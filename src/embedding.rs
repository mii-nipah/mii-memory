use serde_json::json;

pub const EMBEDDING_DIMENSIONS: usize = 128;
pub const EMBEDDED_MODEL_SIZE: usize = 23_046_789;
pub const EMBEDDED_MODEL_SHA256: &str =
    "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21";

#[used]
pub static EMBEDDED_MODEL_BYTES: [u8; EMBEDDED_MODEL_SIZE] =
    *include_bytes!("../weights/minilm_model_quint8_avx2.onnx");

pub fn embedded_model_size() -> usize {
    EMBEDDED_MODEL_BYTES.len()
}

pub fn embedded_model_bytes() -> &'static [u8] {
    &EMBEDDED_MODEL_BYTES
}

pub fn embed_text(text: &str) -> Vec<f32> {
    let mut embedding = vec![0.0; EMBEDDING_DIMENSIONS];
    let tokens = tokenize(text);

    if tokens.is_empty() {
        return embedding;
    }

    for token in tokens {
        add_hashed_feature(&mut embedding, &token, 1.0);

        if token.len() > 4 {
            for gram in token.as_bytes().windows(3) {
                if let Ok(gram) = std::str::from_utf8(gram) {
                    add_hashed_feature(&mut embedding, gram, 0.35);
                }
            }
        }
    }

    normalize(&mut embedding);
    embedding
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

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() || character == '_' || character == '-' {
            current.extend(character.to_lowercase());
            continue;
        }

        if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn add_hashed_feature(embedding: &mut [f32], feature: &str, weight: f32) {
    let hash = fnv1a(feature.as_bytes());
    let index = hash as usize % EMBEDDING_DIMENSIONS;
    let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
    embedding[index] += sign * weight;
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;

    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash
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
    use sha2::{Digest, Sha256};

    #[test]
    fn related_text_scores_higher_than_unrelated_text() {
        let query = embed_text("rust sqlite memory tags");
        let related = embed_text("sqlite backed rust memory store with tags");
        let unrelated = embed_text("fresh bread and ceramic cups");

        assert!(cosine_similarity(&query, &related) > cosine_similarity(&query, &unrelated));
    }

    #[test]
    fn minilm_model_is_embedded() {
        let hash = Sha256::digest(embedded_model_bytes());

        assert_eq!(embedded_model_size(), EMBEDDED_MODEL_SIZE);
        assert_eq!(hex::encode(hash), EMBEDDED_MODEL_SHA256);
    }
}
