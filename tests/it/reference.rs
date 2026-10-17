//! End-to-end reference test (design.md §10): TinyLlama-1.1B Q8_0 against greedy
//! output recorded from llama.cpp (`tests/fixtures/tinyllama_greedy.json`).
//!
//! The fixture was recorded from the Q8_0 file expanded losslessly to F32, with an
//! f32 KV cache, so llama.cpp computes from exactly the weights we dequantize to,
//! with f32 activations. (Run on the Q8_0 file directly, llama.cpp quantizes
//! activations to Q8_0 and differs from us by ~0.15 nats, enough to flip a 0.1-nat
//! near-tie.) The test checks:
//! 1. our tokenizer produces llama.cpp's prompt tokens exactly;
//! 2. teacher-forced on llama.cpp's tokens, our log-probs of its top-5 candidates
//!    stay within `LOGPROB_TOL` at every step (observed: ~2e-4), and our argmax
//!    agrees wherever its top-2 gap is at least `NEAR_TIE`;
//! 3. free-running greedy output matches token for token up to the first near-tie.
//!
//! A RoPE, GQA or layout bug moves log-probs by whole nats, far outside the tolerance.

use std::io::Read;
use std::path::PathBuf;

use increbot::engine::sampler::argmax;
use increbot::gguf::Gguf;
use increbot::hub;
use increbot::model::{LinearImpl, Llama};
use increbot::tokenizer::Tokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const LOGPROB_TOL: f32 = 0.005;
const NEAR_TIE: f32 = 0.01;
// The packed path rounds activations to Q8_0 in decode *and* prefill (the batched
// int8 kernel), like llama.cpp's own Q8_0 path; observed 0.22 nats.
const PACKED_LOGPROB_TOL: f32 = 0.4;
const PACKED_NEAR_TIE: f32 = 0.4;

#[derive(Deserialize)]
struct Fixture {
    model: ModelRef,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ModelRef {
    repo: String,
    file: String,
    sha256: String,
    size: u64,
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    prompt_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    top5: Vec<(u32, f32)>,
}

impl Step {
    fn margin(&self) -> f32 {
        self.top5[0].1 - self.top5[1].1
    }
}

fn model_path(m: &ModelRef) -> Option<PathBuf> {
    // Cache only: the test never downloads (fetch with `hf download <repo> <file>`).
    let path = hf_hub::Cache::from_env().model(m.repo.clone()).get(&m.file);
    if path.is_none() {
        eprintln!(
            "SKIPPED: {}/{} is not in the HF cache; run `hf download {} {}`",
            m.repo, m.file, m.repo, m.file
        );
    }
    path
}

fn sha256(path: &PathBuf) -> String {
    let mut f = std::fs::File::open(path).unwrap();
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    format!("{:x}", h.finalize())
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = logits
        .iter()
        .map(|&l| ((l - max) as f64).exp())
        .sum::<f64>()
        .ln() as f32
        + max;
    logits.iter().map(|&l| l - lse).collect()
}

struct Tolerance {
    logprob: f32,
    near_tie: f32,
}

/// Runs checks 1–3 for every fixture case and returns the worst |Δ logprob|.
fn check_against_fixture(
    model: &Llama,
    tok: &Tokenizer,
    fixture: &Fixture,
    tol: &Tolerance,
) -> f32 {
    let mut overall = 0.0f32;
    for case in &fixture.cases {
        // 1. Tokenizer parity, BOS included.
        assert_eq!(
            tok.encode_prompt(&case.prompt).unwrap(),
            case.prompt_tokens,
            "prompt tokens for {:?}",
            case.prompt
        );

        // 2. Teacher-forced log-probs.
        let mut cache = model.new_cache();
        let mut logits = model.forward(&case.prompt_tokens, &mut cache).unwrap();
        let mut worst = (0.0f32, 0usize);
        for (i, step) in case.steps.iter().enumerate() {
            let lp = log_softmax(&logits);
            for &(id, want) in &step.top5 {
                let diff = (lp[id as usize] - want).abs();
                if diff > worst.0 {
                    worst = (diff, i);
                }
                assert!(
                    diff <= tol.logprob,
                    "{:?} step {i}: logprob of token {id} is {} vs llama.cpp {want} (diff {diff})",
                    case.prompt,
                    lp[id as usize]
                );
            }
            if step.margin() >= tol.near_tie {
                assert_eq!(
                    argmax(&logits),
                    step.top5[0].0,
                    "{:?} step {i}: argmax",
                    case.prompt
                );
            }
            logits = model.forward(&[case.output_tokens[i]], &mut cache).unwrap();
        }
        eprintln!(
            "{:?}: max |Δ logprob| = {:.4} at step {}",
            case.prompt, worst.0, worst.1
        );
        overall = overall.max(worst.0);

        // 3. Free-running greedy up to the first near-tie.
        let check_len = case
            .steps
            .iter()
            .position(|s| s.margin() < tol.near_tie)
            .unwrap_or(case.steps.len());
        let mut cache = model.new_cache();
        let mut logits = model.forward(&case.prompt_tokens, &mut cache).unwrap();
        let mut got = Vec::new();
        for _ in 0..check_len {
            let next = argmax(&logits);
            got.push(next);
            logits = model.forward(&[next], &mut cache).unwrap();
        }
        assert_eq!(
            got,
            case.output_tokens[..check_len],
            "{:?}: greedy tokens",
            case.prompt
        );
        eprintln!(
            "{:?}: greedy matches for the first {check_len} tokens (first near-tie)",
            case.prompt
        );
    }
    overall
}

fn load(linear: LinearImpl) -> Option<(Llama, Tokenizer, Fixture)> {
    let fixture: Fixture = serde_json::from_str(include_str!("../fixtures/tinyllama_greedy.json"))
        .expect("fixture parses");
    let path = model_path(&fixture.model)?;
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        fixture.model.size,
        "model file size"
    );
    assert_eq!(sha256(&path), fixture.model.sha256, "model file sha256");
    let gguf = Gguf::open(&path).unwrap();
    let tok = Tokenizer::load(
        &hub::resolve("hf-internal-testing/llama-tokenizer", "tokenizer.json").unwrap(),
        &gguf,
    )
    .unwrap();
    let model = Llama::load(&gguf, &Default::default(), linear).unwrap();
    Some((model, tok, fixture))
}

/// The f32 reference path (`DenseLinear` = `BurnQLinear` numerics) must agree with
/// llama.cpp to float precision.
#[test]
fn tinyllama_matches_llama_cpp() {
    let Some((model, tok, fixture)) = load(LinearImpl::Dense) else {
        return;
    };
    check_against_fixture(
        &model,
        &tok,
        &fixture,
        &Tolerance {
            logprob: LOGPROB_TOL,
            near_tie: NEAR_TIE,
        },
    );
}

/// `PackedQ8Linear` rounds decode activations to Q8_0 (as llama.cpp's own Q8_0
/// path does), so it agrees with the f32 reference within that rounding, not to
/// float precision. Prefill runs in f32 (dequantize + gemm).
#[test]
fn tinyllama_packed_close_to_llama_cpp() {
    let Some((model, tok, fixture)) = load(LinearImpl::Packed) else {
        return;
    };
    let worst = check_against_fixture(
        &model,
        &tok,
        &fixture,
        &Tolerance {
            logprob: PACKED_LOGPROB_TOL,
            near_tie: PACKED_NEAR_TIE,
        },
    );
    eprintln!("packed: worst |Δ logprob| = {worst:.4}");
}

/// Prefix reuse (design.md §5): truncating the cache to the longest common prefix
/// and prefilling only the suffix gives the same logits as a fresh prefill.
#[test]
fn kv_prefix_reuse_matches_fresh_prefill() {
    let fixture: Fixture = serde_json::from_str(include_str!("../fixtures/tinyllama_greedy.json"))
        .expect("fixture parses");
    let Some(path) = model_path(&fixture.model) else {
        return;
    };
    let gguf = Gguf::open(&path).unwrap();
    let model = Llama::load(&gguf, &Default::default(), LinearImpl::Dense).unwrap();

    let case = &fixture.cases[1];
    let a: Vec<u32> = case
        .prompt_tokens
        .iter()
        .chain(&case.output_tokens[..12])
        .copied()
        .collect();
    // Same first 15 tokens as `a`, then a different continuation.
    let mut b = a[..15].to_vec();
    b.extend_from_slice(&fixture.cases[0].output_tokens[..6]);

    let mut fresh = model.new_cache();
    let want = model.forward(&b, &mut fresh).unwrap();

    let mut cache = model.new_cache();
    model.forward(&a, &mut cache).unwrap();
    let keep = increbot::engine::reusable_prefix(&cache.tokens, &b);
    assert_eq!(keep, 15);
    cache.truncate(keep);
    let got = model.forward(&b[keep..], &mut cache).unwrap();

    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3, "logits differ by up to {max_diff}");
    assert_eq!(argmax(&got), argmax(&want));
}
