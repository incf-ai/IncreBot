//! Where does a 13B decode step spend its time? Times the linear layers alone (raw
//! slices, then through Burn tensors) and the full forward at short and long context.
//!
//! `cargo run --release --example profile_decode` (threads from `INCREBOT_THREADS`,
//! default 8)

// This tool's whole job is printing a report to the terminal.
#![allow(clippy::print_stdout)]

use std::time::Instant;

use increbot::gguf::Gguf;
use increbot::model::{LinearImpl, Llama, PackedQ8Linear, QLinear};
use increbot::{hub, runtime};

fn main() -> anyhow::Result<()> {
    let threads = std::env::var("INCREBOT_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    runtime::init_pool(threads, true)?;
    let path = hub::resolve(
        "TheBloke/LLaMA2-13B-Tiefighter-GGUF",
        "llama2-13b-tiefighter.Q8_0.gguf",
    )?;
    let g = Gguf::open(&path)?;
    let device = Default::default();

    {
        let t = Instant::now();
        let lins: Vec<PackedQ8Linear> = g
            .tensors
            .iter()
            .filter(|t| t.shape().len() == 2 && t.name != "token_embd.weight")
            .map(|t| {
                let s = t.shape();
                PackedQ8Linear::new(g.bytes(t).unwrap(), s[1], s[0])
            })
            .collect();
        println!(
            "{} linears packed in {:.1}s",
            lins.len(),
            t.elapsed().as_secs_f64()
        );
        let xs: Vec<Vec<f32>> = lins.iter().map(|l| vec![0.01; l.in_features()]).collect();
        if std::env::var_os("BENCH_LINEAR").is_some() {
            use increbot::model::packed_q8::Kernel;
            let l = lins.iter().find(|l| l.out_features() == 13824).unwrap();
            let k = l.in_features();
            println!("ffn_up [5120 -> 13824], ms per call (best of 3):");
            println!(
                "{:>5} {:>9} {:>9} {:>9} {:>9}",
                "m", "portable", "avx2", "vnni", "gemm"
            );
            for m in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
                let x: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
                    .collect();
                let time = |f: &dyn Fn() -> Vec<f32>| {
                    (0..3)
                        .map(|_| {
                            let t = Instant::now();
                            std::hint::black_box(f());
                            t.elapsed().as_secs_f64() * 1e3
                        })
                        .fold(f64::MAX, f64::min)
                };
                let p = time(&|| l.batched_with(&x, m, Kernel::Portable));
                let a = time(&|| l.batched_with(&x, m, Kernel::Avx2));
                let v = time(&|| l.batched_with(&x, m, Kernel::AvxVnni));
                let g = if m > 1 {
                    time(&|| l.matmul(&x, m))
                } else {
                    f64::NAN
                };
                println!("{m:>5} {p:>9.2} {a:>9.2} {v:>9.2} {g:>9.2}");
            }
            let xs1: Vec<Vec<f32>> = lins.iter().map(|l| vec![0.01; l.in_features()]).collect();
            for (name, kern) in [
                ("matvec portable", None),
                ("batched avx2 m=1", Some(Kernel::Avx2)),
                ("batched vnni m=1", Some(Kernel::AvxVnni)),
            ] {
                let best = (0..3)
                    .map(|_| {
                        let t = Instant::now();
                        for (l, x) in lins.iter().zip(&xs1) {
                            std::hint::black_box(match kern {
                                None => l.matvec(x),
                                Some(kk) => l.batched_with(x, 1, kk),
                            });
                        }
                        t.elapsed().as_secs_f64()
                    })
                    .fold(f64::MAX, f64::min);
                println!("all 281 linears, {name}: {best:.3}s");
            }
            return Ok(());
        }
        for _ in 0..2 {
            let t = Instant::now();
            for (l, x) in lins.iter().zip(&xs) {
                std::hint::black_box(l.matvec(x));
            }
            println!(
                "all matvecs, raw slices:      {:.3}s",
                t.elapsed().as_secs_f64()
            );
        }
        let m = 512;
        let l = &lins.iter().find(|l| l.out_features() == 13824).unwrap();
        let x = vec![0.01f32; m * l.in_features()];
        for _ in 0..4 {
            let t = Instant::now();
            std::hint::black_box(l.matmul(&x, m));
            let s = t.elapsed().as_secs_f64();
            let flops = 2.0 * (m * l.in_features() * l.out_features()) as f64;
            println!(
                "prefill matmul m={m} [5120->13824]: {:.3}s = {:.0} GFLOP/s incl. dequant",
                s,
                flops / s / 1e9
            );
        }
    }

    if std::env::var_os("MATMUL_ONLY").is_some() {
        return Ok(());
    }
    let mut model = Llama::load(&g, &device, LinearImpl::Packed)?;
    let kv_f16 = std::env::var_os("KV_F16").is_some();
    model.set_kv_f16(kv_f16);
    println!("kv cache f16: {kv_f16}");
    increbot::model::profile::take();
    drop(g);
    let mut cache = model.new_cache();
    model.forward(&[1], &mut cache)?;
    for ctx in [1usize, 1000] {
        if cache.len() < ctx {
            let fill: Vec<u32> = (0..ctx - cache.len())
                .map(|i| 100 + (i as u32 % 1000))
                .collect();
            let t = Instant::now();
            model.forward(&fill, &mut cache)?;
            let s = t.elapsed().as_secs_f64();
            println!(
                "prefill {} tokens: {:.2}s = {:.1} tok/s",
                fill.len(),
                s,
                fill.len() as f64 / s
            );
            print_sections();
        }
        for _ in 0..3 {
            let t = Instant::now();
            model.forward(&[500], &mut cache)?;
            println!(
                "decode step at ctx {:5}: {:.3}s",
                cache.len(),
                t.elapsed().as_secs_f64()
            );
        }
        print_sections();
    }
    for n in [16usize, 50] {
        let warm: Vec<u32> = (0..n as u32).map(|i| 300 + i).collect();
        let t = Instant::now();
        model.forward(&warm, &mut cache)?;
        let s = t.elapsed().as_secs_f64();
        println!("warm prefill {n} tokens at ctx {}: {:.2}s", cache.len(), s);
        print_sections();
    }
    Ok(())
}

fn print_sections() {
    let secs = increbot::model::profile::take();
    let total: f64 = secs.iter().map(|(_, d)| d.as_secs_f64()).sum();
    for (name, d) in secs {
        println!(
            "    {name:18} {:8.3}s {:5.1}%",
            d.as_secs_f64(),
            100.0 * d.as_secs_f64() / total
        );
    }
}
