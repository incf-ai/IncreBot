//! Q4_K / Q6_K decoding checked against an independent source: TinyLlama's
//! Q4_K_M file and its Q8_0 file were made from the same f16 weights, so every
//! decoded tensor must be close to its Q8_0 counterpart. A layout bug (wrong
//! nibble, scale or bit order) scrambles values and shows up as an error near
//! 100% instead of a few percent. Skipped when either file is missing.

use increbot::gguf::{GgmlType, Gguf};

fn file(name: &str) -> Option<Gguf> {
    let path = hf_hub::Cache::from_env()
        .model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into())
        .get(name);
    match path {
        Some(p) => Some(Gguf::open(&p).unwrap()),
        None => {
            eprintln!("SKIPPED: {name} not in the HF cache");
            None
        }
    }
}

/// RMS(a − b) / RMS(b).
fn relative_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut base) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        diff += ((x - y) as f64).powi(2);
        base += (y as f64).powi(2);
    }
    (diff / base).sqrt()
}

#[test]
fn k_quants_decode_close_to_q8_0() {
    let (Some(k), Some(q8)) = (
        file("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        file("tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
    ) else {
        return;
    };
    let (mut worst_q4, mut worst_q6, mut checked) = (0f64, 0f64, 0);
    for t in &k.tensors {
        if !matches!(t.ty, GgmlType::Q4K | GgmlType::Q6K) {
            continue;
        }
        let a = k.to_f32(t).unwrap();
        let b = q8.to_f32(q8.tensor(&t.name).unwrap()).unwrap();
        assert_eq!(a.len(), b.len(), "{}", t.name);
        let err = relative_rms(&a, &b);
        match t.ty {
            GgmlType::Q4K => worst_q4 = worst_q4.max(err),
            _ => worst_q6 = worst_q6.max(err),
        }
        checked += 1;
    }
    eprintln!(
        "checked {checked} tensors: worst relative RMS error Q4_K {worst_q4:.4}, Q6_K {worst_q6:.4}"
    );
    assert_eq!(checked, 156, "all Q4_K and Q6_K tensors");
    // 4-bit and 6-bit rounding (plus Q8_0's own) stay well under these; a
    // scrambled layout gives ~1.0 or more.
    assert!(worst_q4 < 0.15, "Q4_K error {worst_q4}");
    assert!(worst_q6 < 0.05, "Q6_K error {worst_q6}");
}

/// The whole model on Q4_K_M: the packed kernels (8-bit activations) against
/// f32 weights decoded from the same file (f32 activations), for a long prompt
/// (dequantize + gemm), a short one (integer kernel) and single-token decode.
#[test]
fn packed_q4_k_m_forward_matches_dense() {
    use increbot::model::{LinearImpl, Llama};
    let Some(g) = file("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") else {
        return;
    };
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return;
    }
    let packed = Llama::load(&g, &Default::default(), LinearImpl::Packed).unwrap();
    let dense = Llama::load(&g, &Default::default(), LinearImpl::Dense).unwrap();
    let long: Vec<u32> = (0..150).map(|i| 300 + (i * 37) % 5000).collect();
    let (mut cp, mut cd) = (packed.new_cache(), dense.new_cache());
    let mut steps: Vec<(&str, Vec<u32>)> =
        vec![("prefill 150", long), ("prefill 12", (900..912).collect())];
    steps.extend((0..4).map(|i| ("decode", vec![1000 + i])));
    for (what, toks) in steps {
        let a = packed.forward(&toks, &mut cp).unwrap();
        let b = dense.forward(&toks, &mut cd).unwrap();
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
        let rms = (b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / b.len() as f64).sqrt();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs() as f64)
            .fold(0.0, f64::max);
        eprintln!("{what}: max |packed − dense| {max_diff:.3} (logit RMS {rms:.2})");
        assert_eq!(argmax(&a), argmax(&b), "{what}: top token differs");
        // 8-bit activation rounding only; a kernel bug moves logits by whole units.
        assert!(max_diff < 0.25 * rms, "{what}: logits differ by {max_diff}");
    }
}
