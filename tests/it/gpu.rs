//! The GPU forward pass against the CPU (gpu-plan.md, milestone 2), on TinyLlama
//! Q4_K_M, compared with the dense CPU model (f32 weights decoded from the same
//! file). Decode runs on f32 activations and agrees to f16 KV rounding; prefill
//! quantizes activations to 8 bits like the CPU's packed kernels, so it gets
//! their tolerance (`kquant_decode.rs`). The kernels themselves are checked
//! tightly in `gpu/model.rs`.
//! Runs on the first GPU in `INCREBOT_GPUS`; skipped when that is unset, when
//! the file is missing, or in debug builds.

use increbot::gguf::Gguf;
use increbot::gpu::{Device, GpuModel};
use increbot::model::{LinearImpl, Llama};
use std::sync::atomic::AtomicBool;

fn setup() -> Option<(Llama, Llama, std::sync::Arc<Device>)> {
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return None;
    }
    let Some(pci) = std::env::var("INCREBOT_GPUS")
        .ok()
        .and_then(|v| v.split([',', '=']).next().map(str::to_string))
        .filter(|v| !v.is_empty())
    else {
        eprintln!("SKIPPED: INCREBOT_GPUS is not set");
        return None;
    };
    let path = hf_hub::Cache::from_env()
        .model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into())
        .get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    let Some(path) = path else {
        eprintln!("SKIPPED: TinyLlama Q4_K_M not in the HF cache");
        return None;
    };
    let g = Gguf::open(&path).unwrap();
    let mut packed = Llama::load(&g, &Default::default(), LinearImpl::Packed).unwrap();
    let mut dense = Llama::load(&g, &Default::default(), LinearImpl::Dense).unwrap();
    packed.set_kv_f16(true);
    dense.set_kv_f16(true);
    let dev = Device::open(pci.parse().unwrap()).unwrap();
    Some((packed, dense, dev))
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap()
}

fn compare(what: &str, gpu: &[f32], cpu: &[f32], tol: f64) {
    let rms = (cpu.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / cpu.len() as f64).sqrt();
    let max_diff = gpu
        .iter()
        .zip(cpu)
        .map(|(x, y)| (x - y).abs() as f64)
        .fold(0.0, f64::max);
    eprintln!("{what}: max |gpu − cpu| {max_diff:.4} (logit RMS {rms:.2})");
    assert_eq!(argmax(gpu), argmax(cpu), "{what}: top token differs");
    assert!(max_diff < tol * rms, "{what}: logits differ by {max_diff}");
}

#[test]
fn gpu_forward_matches_cpu() {
    let _cards = crate::CARDS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((packed, dense, dev)) = setup() else {
        return;
    };
    let mut gpu = GpuModel::load(&dev, &packed, &AtomicBool::new(false)).unwrap();
    gpu.activate(2048).unwrap();
    let never = || false;
    let (mut cg, mut cc) = (packed.new_cache(), dense.new_cache());
    let long: Vec<u32> = (0..600).map(|i| 300 + (i * 37) % 5000).collect();
    let mut steps: Vec<(&str, Vec<u32>)> = vec![
        ("prefill 600 (2 chunks)", long),
        ("prefill 12", (900..912).collect()),
    ];
    steps.extend((0..4).map(|i| ("decode", vec![1000 + i])));
    for (what, toks) in steps {
        let a = gpu
            .forward(&packed, &toks, &mut cg, &never)
            .unwrap()
            .unwrap();
        let b = dense.forward(&toks, &mut cc).unwrap();
        // Prefill quantizes activations to 8 bits and decode builds on its KV
        // cache, so the CPU packed path's tolerance; a kernel bug moves logits by
        // whole units.
        compare(what, &a, &b, 0.25);
        assert_eq!(cg.tokens, cc.tokens);
    }

    // The host cache the GPU filled is usable on the CPU, and vice versa: continue
    // on the CPU from the GPU's cache, and on the GPU from a CPU-only cache.
    let a = dense.forward(&[42], &mut cg).unwrap();
    let b = dense.forward(&[42], &mut cc).unwrap();
    compare("cpu after gpu", &a, &b, 0.25);
    let mut fresh = dense.new_cache();
    dense
        .forward(&(0..40).collect::<Vec<_>>(), &mut fresh)
        .unwrap();
    let a = gpu
        .forward(&packed, &[7], &mut fresh, &never)
        .unwrap()
        .unwrap();
    let mut fresh2 = dense.new_cache();
    let b = dense
        .forward(&(0..40).chain([7]).collect::<Vec<_>>(), &mut fresh2)
        .unwrap();
    compare("gpu after cpu", &a, &b, 0.25);
}

#[test]
fn gpu_stop_keeps_finished_chunks() {
    let _cards = crate::CARDS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((packed, _, dev)) = setup() else {
        return;
    };
    let mut gpu = GpuModel::load(&dev, &packed, &AtomicBool::new(false)).unwrap();
    gpu.activate(2048).unwrap();
    let mut cache = packed.new_cache();
    let toks: Vec<u32> = (0..700).map(|i| 300 + i).collect();
    // Stop after the first chunk's layers have all been submitted.
    let calls = std::cell::Cell::new(0);
    let n_layers = packed.config.n_layers;
    let stop = || {
        calls.set(calls.get() + 1);
        calls.get() > n_layers + 1
    };
    let r = gpu.forward(&packed, &toks, &mut cache, &stop).unwrap();
    assert!(r.is_err());
    assert_eq!(cache.len(), 512, "first chunk kept");
    // Finishing the rest gives the same logits as one uninterrupted run.
    let a = gpu
        .forward(&packed, &toks[512..], &mut cache, &|| false)
        .unwrap()
        .unwrap();
    let mut c2 = packed.new_cache();
    let b = gpu
        .forward(&packed, &toks, &mut c2, &|| false)
        .unwrap()
        .unwrap();
    compare("resumed", &a, &b, 0.01);
}
