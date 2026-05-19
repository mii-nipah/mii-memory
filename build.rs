use std::env;
use std::path::Path;

const MODEL_PATH: &str = "weights/minilm_model_quint8_avx2.onnx";
const VOCAB_PATH: &str = "weights/vocab.txt";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_embedded_embeddings)");
    println!("cargo:rerun-if-changed={MODEL_PATH}");
    println!("cargo:rerun-if-changed={VOCAB_PATH}");

    let embedded_feature_enabled = env::var_os("CARGO_FEATURE_EMBEDDED").is_some();
    let embedded_assets_present =
        Path::new(MODEL_PATH).is_file() && Path::new(VOCAB_PATH).is_file();

    if embedded_feature_enabled && embedded_assets_present {
        println!("cargo:rustc-cfg=has_embedded_embeddings");
    } else if embedded_feature_enabled {
        println!(
            "cargo:warning=embedded feature enabled but weights are not present; this build will require --embeddings <PATH> at runtime"
        );
    }
}
