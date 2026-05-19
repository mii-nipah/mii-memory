use std::collections::HashMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail, format_err};
use serde_json::json;
use tract_onnx::prelude::*;

pub const EMBEDDING_DIMENSIONS: usize = 384;
pub const MAX_SEQUENCE_LENGTH: usize = 128;
pub const DEFAULT_MODEL_FILENAME: &str = "minilm_model_quint8_avx2.onnx";
pub const DEFAULT_VOCAB_FILENAME: &str = "vocab.txt";
pub const EMBEDDED_MODEL_SIZE: usize = 23_046_789;
pub const EMBEDDED_MODEL_SHA256: &str =
    "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21";
pub const EMBEDDED_VOCAB_SIZE: usize = 231_508;
pub const EMBEDDED_VOCAB_SHA256: &str =
    "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3";

#[cfg(has_embedded_embeddings)]
#[used]
pub static EMBEDDED_MODEL_BYTES: [u8; EMBEDDED_MODEL_SIZE] =
    *include_bytes!("../weights/minilm_model_quint8_avx2.onnx");
#[cfg(has_embedded_embeddings)]
pub static EMBEDDED_VOCAB: &str = include_str!("../weights/vocab.txt");

type RunnableMiniLm = Arc<TypedRunnableModel>;

static EXTERNAL_EMBEDDINGS: OnceLock<EmbeddingPaths> = OnceLock::new();
static MODEL: OnceLock<TractResult<RunnableMiniLm>> = OnceLock::new();
static VOCAB: OnceLock<TractResult<HashMap<String, i64>>> = OnceLock::new();

#[derive(Debug, Clone)]
struct EmbeddingPaths {
    model_path: PathBuf,
    vocab_path: PathBuf,
}

pub fn configure_embeddings_path(path: impl Into<PathBuf>) -> Result<()> {
    if MODEL.get().is_some() || VOCAB.get().is_some() {
        bail!("--embeddings must be configured before embeddings are first used");
    }

    let paths = resolve_embeddings_path(path.into())?;
    EXTERNAL_EMBEDDINGS
        .set(paths)
        .map_err(|_| format_err!("--embeddings was configured more than once"))?;

    Ok(())
}

fn resolve_embeddings_path(path: PathBuf) -> Result<EmbeddingPaths> {
    let (model_path, vocab_path) = if path.is_dir() {
        (
            path.join(DEFAULT_MODEL_FILENAME),
            path.join(DEFAULT_VOCAB_FILENAME),
        )
    } else {
        let vocab_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join(DEFAULT_VOCAB_FILENAME);
        (path, vocab_path)
    };

    if !model_path.is_file() {
        bail!("embedding model file not found at {}", model_path.display());
    }

    if !vocab_path.is_file() {
        bail!(
            "embedding vocabulary file not found at {}",
            vocab_path.display()
        );
    }

    Ok(EmbeddingPaths {
        model_path,
        vocab_path,
    })
}

#[cfg(has_embedded_embeddings)]
pub fn embedded_model_size() -> usize {
    EMBEDDED_MODEL_BYTES.len()
}

#[cfg(has_embedded_embeddings)]
pub fn embedded_model_bytes() -> &'static [u8] {
    &EMBEDDED_MODEL_BYTES
}

pub fn embed_text(text: &str) -> TractResult<Vec<f32>> {
    minilm_embedding(text)
}

pub fn blend(content_embedding: &[f32], tag_embedding: &[f32]) -> Vec<f32> {
    let mut blended = vec![0.0; EMBEDDING_DIMENSIONS];

    for (index, value) in blended.iter_mut().enumerate() {
        *value = content_embedding.get(index).copied().unwrap_or_default() * 0.75
            + tag_embedding.get(index).copied().unwrap_or_default() * 0.25;
    }

    normalize(&mut blended);
    blended
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

pub fn encode_embedding(embedding: &[f32]) -> String {
    serde_json::to_string(embedding).unwrap_or_else(|_| json!([]).to_string())
}

pub fn decode_embedding(raw: &str) -> Vec<f32> {
    let mut embedding = serde_json::from_str::<Vec<f32>>(raw).unwrap_or_default();
    embedding.resize(EMBEDDING_DIMENSIONS, 0.0);
    embedding.truncate(EMBEDDING_DIMENSIONS);
    normalize(&mut embedding);
    embedding
}

fn minilm_embedding(text: &str) -> TractResult<Vec<f32>> {
    let encoded = encode_text(text)?;
    let shape = [1, MAX_SEQUENCE_LENGTH];
    let input_ids = Tensor::from_shape(&shape, &encoded.input_ids)?.into_tvalue();
    let attention_mask = Tensor::from_shape(&shape, &encoded.attention_mask)?.into_tvalue();
    let token_type_ids = Tensor::from_shape(&shape, &encoded.token_type_ids)?.into_tvalue();
    let outputs = load_model()?.run(tvec!(input_ids, attention_mask, token_type_ids))?;
    let last_hidden_state = outputs[0].to_plain_array_view::<f32>()?;
    let hidden_size = last_hidden_state.shape().get(2).copied().unwrap_or(0);
    let mut embedding = vec![0.0; hidden_size];
    let mut token_count = 0.0_f32;

    for token_index in 0..MAX_SEQUENCE_LENGTH {
        if encoded.attention_mask[token_index] == 0 {
            continue;
        }

        token_count += 1.0;
        for hidden_index in 0..hidden_size {
            embedding[hidden_index] += last_hidden_state[[0, token_index, hidden_index]];
        }
    }

    if token_count > 0.0 {
        for value in &mut embedding {
            *value /= token_count;
        }
    }

    embedding.resize(EMBEDDING_DIMENSIONS, 0.0);
    embedding.truncate(EMBEDDING_DIMENSIONS);
    normalize(&mut embedding);
    Ok(embedding)
}

fn load_model() -> TractResult<&'static RunnableMiniLm> {
    MODEL
        .get_or_init(load_model_from_source)
        .as_ref()
        .map_err(|error| format_err!("failed to load MiniLM model: {error}"))
}

fn load_model_from_source() -> TractResult<RunnableMiniLm> {
    if let Some(paths) = EXTERNAL_EMBEDDINGS.get() {
        let model_bytes = fs::read(&paths.model_path).with_context(|| {
            format!(
                "failed to read embedding model {}",
                paths.model_path.display()
            )
        })?;
        let mut model_bytes = Cursor::new(model_bytes);

        return tract_onnx::onnx()
            .model_for_read(&mut model_bytes)?
            .into_optimized()?
            .into_runnable();
    }

    if let Some(model_bytes) = embedded_model_bytes_if_available() {
        let mut model_bytes = Cursor::new(model_bytes);

        return tract_onnx::onnx()
            .model_for_read(&mut model_bytes)?
            .into_optimized()?
            .into_runnable();
    }

    bail!("{}", missing_embeddings_message())
}

#[derive(Debug)]
struct EncodedText {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

fn encode_text(text: &str) -> TractResult<EncodedText> {
    let vocab = vocab()?;
    let pad_id = token_id(vocab, "[PAD]");
    let unknown_id = token_id(vocab, "[UNK]");
    let cls_id = token_id(vocab, "[CLS]");
    let sep_id = token_id(vocab, "[SEP]");
    let mut input_ids = Vec::with_capacity(MAX_SEQUENCE_LENGTH);

    input_ids.push(cls_id);
    for token in basic_tokens(text) {
        for piece in wordpiece(&token, vocab, unknown_id) {
            if input_ids.len() >= MAX_SEQUENCE_LENGTH - 1 {
                break;
            }
            input_ids.push(piece);
        }

        if input_ids.len() >= MAX_SEQUENCE_LENGTH - 1 {
            break;
        }
    }
    input_ids.push(sep_id);

    let mut attention_mask = vec![1; input_ids.len()];
    let mut token_type_ids = vec![0; input_ids.len()];

    input_ids.resize(MAX_SEQUENCE_LENGTH, pad_id);
    attention_mask.resize(MAX_SEQUENCE_LENGTH, 0);
    token_type_ids.resize(MAX_SEQUENCE_LENGTH, 0);

    Ok(EncodedText {
        input_ids,
        attention_mask,
        token_type_ids,
    })
}

fn vocab() -> TractResult<&'static HashMap<String, i64>> {
    VOCAB
        .get_or_init(load_vocab_from_source)
        .as_ref()
        .map_err(|error| format_err!("failed to load MiniLM vocabulary: {error}"))
}

fn load_vocab_from_source() -> TractResult<HashMap<String, i64>> {
    let vocab = if let Some(paths) = EXTERNAL_EMBEDDINGS.get() {
        fs::read_to_string(&paths.vocab_path).with_context(|| {
            format!(
                "failed to read embedding vocabulary {}",
                paths.vocab_path.display()
            )
        })?
    } else if let Some(vocab) = embedded_vocab_if_available() {
        vocab.to_string()
    } else {
        bail!("{}", missing_embeddings_message());
    };

    Ok(vocab
        .lines()
        .enumerate()
        .map(|(index, token)| (token.trim_end().to_string(), index as i64))
        .collect())
}

fn token_id(vocab: &HashMap<String, i64>, token: &str) -> i64 {
    *vocab.get(token).unwrap_or(&100)
}

fn basic_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_whitespace() {
            push_current_token(&mut tokens, &mut current);
        } else if is_punctuation(character) {
            push_current_token(&mut tokens, &mut current);
            tokens.push(character.to_string());
        } else if !character.is_control() {
            current.push(character);
        }
    }

    push_current_token(&mut tokens, &mut current);
    tokens
}

fn push_current_token(tokens: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

fn is_punctuation(character: char) -> bool {
    character.is_ascii_punctuation()
        || matches!(character as u32, 0x2000..=0x206F | 0x2E00..=0x2E7F)
}

fn wordpiece(token: &str, vocab: &HashMap<String, i64>, unknown_id: i64) -> Vec<i64> {
    let characters = token.chars().collect::<Vec<_>>();
    if characters.len() > 100 {
        return vec![unknown_id];
    }

    let mut pieces = Vec::new();
    let mut start = 0;

    while start < characters.len() {
        let mut end = characters.len();
        let mut current = None;

        while start < end {
            let mut piece = String::new();
            if start > 0 {
                piece.push_str("##");
            }
            piece.extend(&characters[start..end]);

            if let Some(id) = vocab.get(piece.as_str()) {
                current = Some(*id);
                break;
            }
            end -= 1;
        }

        let Some(id) = current else {
            return vec![unknown_id];
        };

        pieces.push(id);
        start = end;
    }

    pieces
}

fn missing_embeddings_message() -> &'static str {
    "this mii-memory binary was built without embedded embeddings; pass --embeddings <PATH> or set MII_MEMORY_EMBEDDINGS to a directory containing minilm_model_quint8_avx2.onnx and vocab.txt"
}

#[cfg(has_embedded_embeddings)]
fn embedded_model_bytes_if_available() -> Option<&'static [u8]> {
    Some(embedded_model_bytes())
}

#[cfg(not(has_embedded_embeddings))]
fn embedded_model_bytes_if_available() -> Option<&'static [u8]> {
    None
}

#[cfg(has_embedded_embeddings)]
fn embedded_vocab_if_available() -> Option<&'static str> {
    Some(EMBEDDED_VOCAB)
}

#[cfg(not(has_embedded_embeddings))]
fn embedded_vocab_if_available() -> Option<&'static str> {
    None
}

fn normalize(embedding: &mut [f32]) {
    let length = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();

    if length == 0.0 {
        return;
    }

    for value in embedding {
        *value /= length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(has_embedded_embeddings)]
    use sha2::{Digest, Sha256};

    #[cfg(has_embedded_embeddings)]
    #[test]
    fn related_text_scores_higher_than_unrelated_text() {
        let query = embed_text("rust sqlite memory tags").expect("query embedding");
        let related =
            embed_text("sqlite backed rust memory store with tags").expect("related embedding");
        let unrelated = embed_text("fresh bread and ceramic cups").expect("unrelated embedding");

        assert!(cosine_similarity(&query, &related) > cosine_similarity(&query, &unrelated));
    }

    #[cfg(has_embedded_embeddings)]
    #[test]
    fn minilm_embedding_returns_normalized_vector() {
        let embedding = minilm_embedding("rust sqlite memory tags").expect("MiniLM embedding");
        let length = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();

        assert_eq!(embedding.len(), EMBEDDING_DIMENSIONS);
        assert!(embedding.iter().any(|value| *value != 0.0));
        assert!((length - 1.0).abs() < 0.0001);
    }

    #[cfg(has_embedded_embeddings)]
    #[test]
    fn minilm_model_and_vocab_are_embedded() {
        let model_hash = Sha256::digest(embedded_model_bytes());
        let vocab_hash = Sha256::digest(EMBEDDED_VOCAB.as_bytes());

        assert_eq!(embedded_model_size(), EMBEDDED_MODEL_SIZE);
        assert_eq!(hex::encode(model_hash), EMBEDDED_MODEL_SHA256);
        assert_eq!(EMBEDDED_VOCAB.len(), EMBEDDED_VOCAB_SIZE);
        assert_eq!(hex::encode(vocab_hash), EMBEDDED_VOCAB_SHA256);
    }

    #[cfg(not(has_embedded_embeddings))]
    #[test]
    fn embedding_requires_external_assets_when_not_embedded() {
        let error = embed_text("rust sqlite memory tags")
            .unwrap_err()
            .to_string();

        assert!(error.contains("--embeddings <PATH>"));
    }
}
