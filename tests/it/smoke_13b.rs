//! 13B smoke test (design.md §10), ignored by default: it needs the 13.8 GB
//! Tiefighter GGUF in the HF cache and ~15 GB of RAM, and on the Burn reference
//! path every token dequantizes all weights on one thread.
//!
//! Run with: `cargo test --release --test it smoke_13b -- --ignored --nocapture`

use increbot::engine::sampler::argmax;
use increbot::gguf::Gguf;
use increbot::hub;
use increbot::model::{LinearImpl, Llama};
use increbot::tokenizer::Tokenizer;

#[test]
#[ignore = "needs the 13B GGUF and minutes per token on the Burn path"]
fn tiefighter_loads_and_answers() {
    let Some(path) = hf_hub::Cache::from_env()
        .model("TheBloke/LLaMA2-13B-Tiefighter-GGUF".into())
        .get("llama2-13b-tiefighter.Q8_0.gguf")
    else {
        eprintln!("SKIPPED: Tiefighter GGUF not in the HF cache");
        return;
    };
    let gguf = Gguf::open(&path).unwrap();
    // Includes the 32000-entry vocab check against tokenizer.json.
    let tok = Tokenizer::load(
        &hub::resolve("hf-internal-testing/llama-tokenizer", "tokenizer.json").unwrap(),
        &gguf,
    )
    .unwrap();
    let model = Llama::load(&gguf, &Default::default(), LinearImpl::Burn).unwrap();
    drop(gguf);

    let prompt = tok.encode_prompt("The capital of France is").unwrap();
    let mut cache = model.new_cache();
    let logits = model.forward(&prompt, &mut cache).unwrap();
    let next = argmax(&logits);
    let text = tok.decode(&[next]).unwrap();
    eprintln!("next token: {next} {text:?}");
    assert_eq!(text.trim(), "Paris");
}
