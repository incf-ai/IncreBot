//! Measures the numbers increbot's design.md only estimates:
//! memory read bandwidth (threads x pinning x THP), Q8_0 matvec over a
//! 13B-sized weight buffer (portable vs AVX2), and f32 gemm at prefill shapes.

use rayon::prelude::*;
use std::time::Instant;

const MODEL_BYTES: usize = 13_831_319_488; // llama2-13b-tiefighter.Q8_0.gguf
const BLOCK: usize = 34; // f16 d + 32 x i8
const K: usize = 5120; // row length (d_model)
const BPR: usize = K / 32; // blocks per row

struct Buf {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}
impl Buf {
    fn new(len: usize, huge: bool) -> Self {
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert!(ptr != libc::MAP_FAILED, "mmap failed");
            let advice = if huge { libc::MADV_HUGEPAGE } else { libc::MADV_NOHUGEPAGE };
            let rc = libc::madvise(ptr, len, advice);
            if rc != 0 {
                eprintln!("madvise({}) failed: {}", if huge { "HUGEPAGE" } else { "NOHUGEPAGE" }, std::io::Error::last_os_error());
            }
            Buf { ptr: ptr as *mut u8, len }
        }
    }
    fn slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut _, self.len) };
    }
}

fn anon_huge_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("AnonHugePages:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn pin(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn pool(n: usize, pinned: bool) -> rayon::ThreadPool {
    let mut b = rayon::ThreadPoolBuilder::new().num_threads(n);
    if pinned {
        // One thread per physical P-core: logical CPUs 0,2,..,14 on the 14900K.
        b = b.start_handler(|i| pin(2 * i));
    }
    b.build().unwrap()
}

/// Fill with valid-looking Q8_0 blocks: small f16 scale, pseudo-random i8.
fn fill(buf: &mut Buf) {
    let d = half::f16::from_f32(0.01).to_bits().to_le_bytes();
    buf.slice_mut().par_chunks_mut(BLOCK * 4096).enumerate().for_each(|(c, chunk)| {
        let mut x = (c as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1;
        for blk in chunk.chunks_mut(BLOCK) {
            if blk.len() < 2 {
                blk.fill(0);
                continue;
            }
            blk[0] = d[0];
            blk[1] = d[1];
            for q in &mut blk[2..] {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *q = x as u8;
            }
        }
    });
}

fn best<F: FnMut() -> f64>(reps: usize, mut f: F) -> f64 {
    (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min)
}

/// Pure streaming read: sum u64 words. Lower bound on decode time.
fn read_bw(p: &rayon::ThreadPool, buf: &Buf) -> f64 {
    let s = buf.slice();
    let secs = best(5, || {
        let t = Instant::now();
        let sum: u64 = p.install(|| {
            s.par_chunks(1 << 22)
                .map(|c| {
                    let (_, words, _) = unsafe { c.align_to::<u64>() };
                    words.iter().fold(0u64, |a, &w| a.wrapping_add(w))
                })
                .reduce(|| 0, |a, b| a.wrapping_add(b))
        });
        std::hint::black_box(sum);
        t.elapsed().as_secs_f64()
    });
    buf.len as f64 / secs / 1e9
}

#[inline]
fn f16_to_f32(lo: u8, hi: u8) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32()
}

/// Portable Q8_0 x Q8_0 row dot product (activation pre-quantized to Q8_0).
#[inline]
fn dot_row_portable(row: &[u8], xq: &[i8], xd: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, blk) in row.chunks_exact(BLOCK).enumerate() {
        let dw = f16_to_f32(blk[0], blk[1]);
        let w = &blk[2..];
        let x = &xq[b * 32..b * 32 + 32];
        let mut s = 0i32;
        for i in 0..32 {
            s += (w[i] as i8 as i32) * (x[i] as i32);
        }
        acc += dw * xd[b] * s as f32;
    }
    acc
}

/// AVX2 Q8_0 x Q8_0 row dot product (the llama.cpp sign/maddubs trick).
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_avx2(row: &[u8], xq: &[i8], xd: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    for (b, blk) in row.chunks_exact(BLOCK).enumerate() {
        let d = f16_to_f32(blk[0], blk[1]) * xd[b];
        let w = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
        let x = _mm256_loadu_si256(xq.as_ptr().add(b * 32) as *const __m256i);
        let aw = _mm256_sign_epi8(w, w);
        let sx = _mm256_sign_epi8(x, w);
        let p16 = _mm256_maddubs_epi16(aw, sx);
        let p32 = _mm256_madd_epi16(p16, ones);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s = _mm_add_ps(hi, lo);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// One "token": a matvec over every weight byte of a 13B Q8_0 model.
fn q8_matvec(p: &rayon::ThreadPool, buf: &Buf, avx2: bool) -> f64 {
    let row_bytes = BPR * BLOCK;
    let rows = buf.len / row_bytes;
    let s = &buf.slice()[..rows * row_bytes];
    let xq: Vec<i8> = (0..K).map(|i| ((i * 37) % 255) as i8).collect();
    let xd: Vec<f32> = vec![0.02; BPR];
    let mut out = vec![0f32; rows];
    best(5, || {
        let t = Instant::now();
        p.install(|| {
            out.par_chunks_mut(256).enumerate().for_each(|(c, o)| {
                for (j, y) in o.iter_mut().enumerate() {
                    let r = c * 256 + j;
                    let row = &s[r * row_bytes..(r + 1) * row_bytes];
                    *y = if avx2 {
                        unsafe { dot_row_avx2(row, &xq, &xd) }
                    } else {
                        dot_row_portable(row, &xq, &xd)
                    };
                }
            })
        });
        std::hint::black_box(&out);
        t.elapsed().as_secs_f64()
    })
}

/// f32 gemm: [m x k] * [k x n], as in a prefill linear layer.
fn gemm_gflops(p: &rayon::ThreadPool, m: usize, k: usize, n: usize) -> f64 {
    let a = vec![0.5f32; m * k];
    let b = vec![0.25f32; k * n];
    let mut c = vec![0f32; m * n];
    let secs = best(3, || {
        let t = Instant::now();
        p.install(|| unsafe {
            gemm::gemm(
                m, n, k,
                c.as_mut_ptr(), 1, n as isize, false,
                a.as_ptr(), 1, k as isize,
                b.as_ptr(), 1, n as isize,
                0.0, 1.0, false, false, false,
                gemm::Parallelism::Rayon(0),
            )
        });
        std::hint::black_box(&c);
        t.elapsed().as_secs_f64()
    });
    2.0 * (m * k * n) as f64 / secs / 1e9
}

fn main() {
    println!("rayon default threads (from env): {}", rayon::current_num_threads());

    for huge in [true, false] {
        let base = anon_huge_kb();
        let mut buf = Buf::new(MODEL_BYTES, huge);
        let t = Instant::now();
        fill(&mut buf);
        let huge_gb = (anon_huge_kb().saturating_sub(base)) as f64 / 1e6 * 1.024;
        println!(
            "\n=== THP {} : filled {:.1} GB in {:.1}s, AnonHugePages for buffer ~{:.1} GB ===",
            if huge { "MADV_HUGEPAGE" } else { "MADV_NOHUGEPAGE" },
            MODEL_BYTES as f64 / 1e9,
            t.elapsed().as_secs_f64(),
            huge_gb
        );

        println!("read bandwidth (GB/s), and implied decode ceiling (tok/s = GB/s / 13.83):");
        for &(n, pinned) in &[
            (1, false), (2, false), (4, false), (6, false), (8, false), (10, false),
            (4, true), (6, true), (8, true),
        ] {
            let p = pool(n, pinned);
            let bw = read_bw(&p, &buf);
            println!(
                "  threads={:2} {:8}  {:6.1} GB/s  -> {:4.2} tok/s",
                n, if pinned { "pinned" } else { "unpinned" }, bw, bw / 13.83
            );
        }

        println!("Q8_0 matvec over the whole 13.8 GB (one decode token's weight traffic):");
        for &(n, pinned) in &[(4, false), (8, false), (10, false), (8, true)] {
            let p = pool(n, pinned);
            for avx2 in [false, true] {
                let s = q8_matvec(&p, &buf, avx2);
                println!(
                    "  threads={:2} {:8} {:8}  {:6.3} s/token  {:5.2} tok/s  ({:5.1} GB/s)",
                    n,
                    if pinned { "pinned" } else { "unpinned" },
                    if avx2 { "avx2" } else { "portable" },
                    s,
                    1.0 / s,
                    MODEL_BYTES as f64 / s / 1e9
                );
            }
        }
    }

    println!("\n=== f32 gemm (prefill linear layer; k = 5120, n = 13824) ===");
    for &(n_threads, pinned) in &[(8, false), (8, true), (10, false)] {
        let p = pool(n_threads, pinned);
        for m in [64usize, 256, 512] {
            let gf = gemm_gflops(&p, m, 5120, 13824);
            println!(
                "  threads={:2} {:8} m={:4}  {:7.1} GFLOP/s  -> prefill ~{:5.1} tok/s (26 GFLOP/token, matmul only)",
                n_threads, if pinned { "pinned" } else { "unpinned" }, m, gf, gf / 26.0
            );
        }
    }
}
