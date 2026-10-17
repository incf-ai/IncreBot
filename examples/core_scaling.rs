//! Prefill throughput per core type on the i9-14900K (P-cores, their hyperthread
//! siblings, E-cores), to estimate what more CPUs would buy. Uses synthetic Q8_0
//! weights shaped like the 13B's ffn_up (5120 → 13824) and m = 256 rows.
//!
//! `cargo run --release --example core_scaling`

// This tool's whole job is printing a report to the terminal.
#![allow(clippy::print_stdout)]

use std::time::Instant;

use increbot::model::PackedQ8Linear;
use increbot::model::packed_q8::Kernel;

fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
    }
}

fn main() {
    let (k, n, m) = (5120usize, 13824usize, 256usize);
    let m_max = 512;
    let mut bytes = Vec::with_capacity(n * k / 32 * 34);
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..n * k / 32 {
        bytes.extend(half::f16::from_f32(0.01).to_le_bytes());
        for _ in 0..32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bytes.push(s as u8);
        }
    }
    let lin = PackedQ8Linear::new(&bytes, k, n);
    let x: Vec<f32> = (0..m_max * k)
        .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let gops = 2.0 * (m * k * n) as f64 / 1e9;

    let configs: &[(&str, Vec<usize>)] = &[
        ("8 P-cores, 1 thread each", vec![0, 2, 4, 6, 8, 10, 12, 14]),
        ("4 P-cores, 1 thread each", vec![0, 2, 4, 6]),
        (
            "4 P-cores + HT siblings (8 thr)",
            vec![0, 1, 2, 3, 4, 5, 6, 7],
        ),
        ("4 E-cores", vec![16, 17, 18, 19]),
        ("8 E-cores", (16..24).collect()),
        ("4 P-cores, 1 thread each (again)", vec![8, 10, 12, 14]),
        (
            "8 P-cores, 1 thread each (again)",
            vec![0, 2, 4, 6, 8, 10, 12, 14],
        ),
        ("8 P-cores + HT siblings (16 thr)", (0..16).collect()),
        ("16 P-thr + 6 E-cores (22 thr)", (0..22).collect()),
        ("16 P-thr + 8 E-cores (24 thr)", (0..24).collect()),
        ("all 32 logical CPUs", (0..32).collect()),
    ];
    println!("batched VNNI kernel, m = {m}, 5120 -> 13824 ({gops:.1} GOP per call)");
    for (name, cpus) in configs {
        let cpus = cpus.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpus.len())
            .start_handler(move |i| pin(cpus[i]))
            .build()
            .unwrap();
        let best = (0..5)
            .map(|_| {
                let t = Instant::now();
                pool.install(|| {
                    std::hint::black_box(lin.batched_with(&x[..m * k], m, Kernel::AvxVnni))
                });
                t.elapsed().as_secs_f64()
            })
            .fold(f64::MAX, f64::min);
        println!(
            "  {name:34} {:7.1} ms  {:6.0} GOP/s",
            best * 1e3,
            gops / best
        );
    }

    // Where a wide prefill pool starts to pay: ms per call by row count.
    let pools: Vec<(&str, rayon::ThreadPool)> = [
        ("8 P", vec![0, 2, 4, 6, 8, 10, 12, 14]),
        ("16 P-thr", (0..16).collect()),
        ("22 thr", (0..22).collect::<Vec<_>>()),
    ]
    .into_iter()
    .map(|(name, cpus)| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpus.len())
            .start_handler(move |i| pin(cpus[i]))
            .build()
            .unwrap();
        (name, pool)
    })
    .collect();
    print!("\n{:>5}", "m");
    for (name, _) in &pools {
        print!(" {name:>9}");
    }
    println!("   (ms per call, best of 5)");
    for m in [2usize, 4, 8, 16, 32, 64, 128, 256, 512] {
        print!("{m:>5}");
        for (_, pool) in &pools {
            let best = (0..5)
                .map(|_| {
                    let t = Instant::now();
                    pool.install(|| {
                        std::hint::black_box(lin.batched_with(&x[..m * k], m, Kernel::AvxVnni))
                    });
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::MAX, f64::min);
            print!(" {:>9.2}", best * 1e3);
        }
        println!();
    }
}
