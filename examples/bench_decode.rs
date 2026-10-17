//! Decode and prefill speed of whole models, for comparing GGUF files (e.g. Q8_0
//! vs Q4_K_M) under the same conditions. Models load one at a time.
//!
//! `cargo run --release --example bench_decode -- [--gpu <pci>] <repo> <file> [<file> …]`
//! (threads from `INCREBOT_THREADS`, default 8, pinned as in the bot;
//! `BENCH_KV_F16=1` for the f16 KV cache; `--gpu 0000:06:00.0` runs the forward
//! pass on that GPU instead, with an f16 cache)

// This tool's whole job is printing a report to the terminal.
#![allow(clippy::print_stdout)]

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use increbot::gguf::Gguf;
use increbot::gpu::{Device, GpuModel};
use increbot::model::{KvCache, LinearImpl, Llama};
use increbot::{hub, runtime};

/// The forward pass under test: the CPU model, or the same model on a GPU.
struct Bench {
    cpu: Llama,
    gpu: Option<GpuModel>,
}

impl Bench {
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        match &mut self.gpu {
            Some(g) => Ok(g
                .forward(&self.cpu, tokens, cache, &|| false)?
                .expect("never stopped")),
            None => self.cpu.forward(tokens, cache),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = match args.iter().position(|a| a == "--gpu") {
        Some(i) => {
            let pci = args.get(i + 1).cloned().unwrap_or_default();
            args.drain(i..i + 2);
            Some(Device::open(pci.parse()?)?)
        }
        None => None,
    };
    anyhow::ensure!(
        args.len() >= 2,
        "usage: bench_decode <repo> <file> [<file> …]"
    );
    let threads = std::env::var("INCREBOT_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    runtime::init_pool(threads, true)?;
    let decode_steps = 32;
    for file in &args[1..] {
        let g = Gguf::open(&hub::resolve(&args[0], file)?)?;
        let t = Instant::now();
        let mut cpu = Llama::load(&g, &Default::default(), LinearImpl::Packed)?;
        // `BENCH_KV_F16=1`: the f16 active KV cache (`runtime.kv_cache_f16`).
        cpu.set_kv_f16(gpu.is_some() || std::env::var_os("BENCH_KV_F16").is_some());
        drop(g);
        println!("{file}: loaded in {:.1}s", t.elapsed().as_secs_f64());
        let gpu = match &gpu {
            Some(dev) => {
                let t = Instant::now();
                let mut m = GpuModel::load(dev, &cpu, &AtomicBool::new(false))?;
                m.activate(2048)?;
                println!(
                    "  uploaded to {} in {:.1}s",
                    dev.pci,
                    t.elapsed().as_secs_f64()
                );
                Some(m)
            }
            None => None,
        };
        let mut model = Bench { cpu, gpu };
        let tok = |i: usize| 300 + (i as u32 * 37) % 20_000;
        let mut cache = model.cpu.new_cache();
        model.forward(&[1], &mut cache)?; // warm-up

        // Cold prefill of ~1,000 tokens, in the bot's 512-token chunks.
        let prompt: Vec<u32> = (0..995).map(tok).collect();
        let t = Instant::now();
        model.forward(&prompt, &mut cache)?;
        let prefill = t.elapsed().as_secs_f64();
        println!(
            "  prefill 995 tokens: {prefill:.1}s ({:.0} tok/s)",
            995.0 / prefill
        );
        report();

        // Warm prefill: 50 new tokens on top.
        let t = Instant::now();
        model.forward(&(2000..2050).map(tok).collect::<Vec<_>>(), &mut cache)?;
        println!(
            "  warm prefill 50 tokens: {:.2}s",
            t.elapsed().as_secs_f64()
        );
        report();

        // Decode at ~1,050 tokens of context.
        let t = Instant::now();
        for i in 0..decode_steps {
            model.forward(&[tok(5000 + i)], &mut cache)?;
        }
        let long = t.elapsed().as_secs_f64() / decode_steps as f64;
        println!("  decode {decode_steps} tokens at ~1,050:");
        report();

        // Decode at short context.
        let mut short_cache = model.cpu.new_cache();
        model.forward(&(0..20).map(tok).collect::<Vec<_>>(), &mut short_cache)?;
        let t = Instant::now();
        for i in 0..decode_steps {
            model.forward(&[tok(6000 + i)], &mut short_cache)?;
        }
        let short = t.elapsed().as_secs_f64() / decode_steps as f64;
        println!(
            "  decode: {:.2} tok/s at short context, {:.2} tok/s at ~1,050 tokens",
            1.0 / short,
            1.0 / long
        );
    }
    Ok(())
}

/// With `INCREBOT_PROFILE=1`: time per section since the last report.
fn report() {
    for (section, d) in increbot::model::profile::take() {
        println!("    {section:<20} {:>8.1} ms", d.as_secs_f64() * 1e3);
    }
}
