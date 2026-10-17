//! `PackedQ8Linear` (design.md §3): the original GGUF Q8_0 blocks in one buffer.
//!
//! - Decode (1 row): quantize the activation to Q8_0 (as llama.cpp does) and run
//!   int8 × int8 block dot products, rows split across the rayon pool. A portable,
//!   auto-vectorised kernel: decode is bandwidth bound, and measured within ~3% of
//!   hand-written AVX2.
//! - Short prefill (2..=`BATCH_MAX` rows): the batched kernel reads each weight
//!   block once and dots it with every activation row (all quantized to Q8_0), with
//!   AVX-VNNI `dpbusd` or AVX2 `maddubs` when available. With several rows per
//!   weight load it is compute bound, which is where SIMD pays off (design.md §3).
//! - Long prefill: dequantize the weight to f32 into a shared scratch buffer, in
//!   parallel, then f32 `gemm`.

use std::sync::{LazyLock, Mutex};

use half::f16;
use rayon::prelude::*;

use super::linear::QLinear;
use crate::gguf::{Q8_0_BLOCK_BYTES, QK8_0};

/// Output rows per rayon task in the matvec.
const ROWS_PER_TASK: usize = 64;

/// Above this many rows, dequantize + gemm instead of the batched int8 kernel.
/// Measured crossovers on the 13B's 5120 → 13824 layer (design.md §3): the tiled
/// VNNI kernel beats gemm up to the 512-row prefill chunk, AVX2 up to ~96 rows,
/// and the portable kernel never. Overridable with `INCREBOT_Q8_BATCH_MAX`.
static BATCH_MAX: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("INCREBOT_Q8_BATCH_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(match kernel() {
            Kernel::AvxVnni => 512,
            Kernel::Avx2 => 96,
            Kernel::Portable => 1,
        })
});

/// The batched kernel variant, chosen once from CPU features. `INCREBOT_Q8_KERNEL`
/// (`portable` | `avx2` | `vnni`) forces one, for tests and benchmarks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Portable,
    Avx2,
    AvxVnni,
}

/// `Kernel` without `AvxVnni`: `batched_with`'s per-row loop only ever sees
/// these two, since `AvxVnni` takes its own tiled path and returns before the
/// loop runs. Narrowing the type (instead of matching `Kernel` again and
/// panicking on `AvxVnni`) makes that guarantee checked by the compiler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarKernel {
    Portable,
    Avx2,
}

static KERNEL: LazyLock<Kernel> = LazyLock::new(|| {
    let detected = if is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") {
        Kernel::AvxVnni
    } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Kernel::Avx2
    } else {
        Kernel::Portable
    };
    let forced = match std::env::var("INCREBOT_Q8_KERNEL").as_deref() {
        Ok("portable") => Some(Kernel::Portable),
        Ok("avx2") => Some(Kernel::Avx2),
        Ok("vnni") => Some(Kernel::AvxVnni),
        _ => None,
    };
    let k = forced.unwrap_or(detected);
    tracing::info!(kernel = ?k, "Q8_0 batched kernel");
    k
});

pub fn kernel() -> Kernel {
    *KERNEL
}

/// Dequantized weights for prefill. One buffer for every layer: the engine runs
/// one forward at a time, and the largest weight (the LM head) sets its size.
static SCRATCH: Mutex<Vec<f32>> = Mutex::new(Vec::new());

pub struct PackedQ8Linear {
    /// `out` rows of `in / 32` blocks, 34 bytes each, exactly as in the GGUF.
    blocks: Vec<u8>,
    in_features: usize,
    out_features: usize,
}

impl PackedQ8Linear {
    /// `bytes` holds `out_features` rows of `in_features / 32` Q8_0 blocks.
    pub fn new(bytes: &[u8], in_features: usize, out_features: usize) -> Self {
        assert!(
            in_features.is_multiple_of(QK8_0),
            "in_features {in_features} is not a multiple of {QK8_0}"
        );
        assert_eq!(
            bytes.len(),
            out_features * in_features / QK8_0 * Q8_0_BLOCK_BYTES,
            "Q8_0 byte count"
        );
        Self {
            blocks: bytes.to_vec(),
            in_features,
            out_features,
        }
    }

    fn row_bytes(&self) -> usize {
        self.in_features / QK8_0 * Q8_0_BLOCK_BYTES
    }

    /// `y = W x` for one activation row.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let (xq, xd) = quantize_row_q8_0(x);
        let row_bytes = self.row_bytes();
        let mut out = vec![0f32; self.out_features];
        out.par_chunks_mut(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(chunk, ys)| {
                let first = chunk * ROWS_PER_TASK;
                for (j, y) in ys.iter_mut().enumerate() {
                    let r = first + j;
                    *y = dot_row(&self.blocks[r * row_bytes..(r + 1) * row_bytes], &xq, &xd);
                }
            });
        out
    }

    /// `Y = X Wᵀ` for `m` rows with the batched int8 kernel: every activation row is
    /// quantized to Q8_0, and each weight block is loaded once for all rows.
    pub fn batched(&self, x: &[f32], m: usize) -> Vec<f32> {
        self.batched_with(x, m, kernel())
    }

    pub fn batched_with(&self, x: &[f32], m: usize, kernel: Kernel) -> Vec<f32> {
        let (k, n) = (self.in_features, self.out_features);
        assert_eq!(x.len(), m * k);
        let (xq, xd) = quantize_row_q8_0(x);
        let row_bytes = self.row_bytes();
        let mut y = vec![0f32; m * n];
        let out = SyncPtr(y.as_mut_ptr());
        // `Kernel` has exactly these 3 variants, so this match needs no catch-all:
        // AvxVnni runs its own tiled path and returns directly; the other two name
        // the scalar kernel for the per-row loop below, which only ever sees them.
        let scalar = match kernel {
            Kernel::AvxVnni => {
                // x + 128 as unsigned bytes for dpbusd (see `rows_vnni_2x4`).
                let xu: Vec<u8> = xq.iter().map(|&q| (q as i16 + 128) as u8).collect();
                (0..n.div_ceil(ROWS_PER_TASK))
                    .into_par_iter()
                    .for_each(|chunk| {
                        let out = &out;
                        let (lo, hi) = (chunk * ROWS_PER_TASK, ((chunk + 1) * ROWS_PER_TASK).min(n));
                        let mut prep = RowPrep::default();
                        let mut r = lo;
                        while r < hi {
                            let pair = (r + 1 < hi) as usize + 1;
                            let w0 = &self.blocks[r * row_bytes..(r + 1) * row_bytes];
                            let w1 = &self.blocks[(r + pair - 1) * row_bytes..(r + pair) * row_bytes];
                            // SAFETY: CPU support checked when `kernel` was chosen; each
                            // (j, r) is written once, inside y (see the generic path below).
                            unsafe {
                                rows_vnni_2x4(w0, w1, &xu, &xd, m, k, &mut prep, |j, a, b| {
                                    *out.0.add(j * n + r) = a;
                                    if pair == 2 {
                                        *out.0.add(j * n + r + 1) = b;
                                    }
                                })
                            };
                            r += pair;
                        }
                    });
                return y;
            }
            Kernel::Avx2 => ScalarKernel::Avx2,
            Kernel::Portable => ScalarKernel::Portable,
        };
        (0..n.div_ceil(ROWS_PER_TASK))
            .into_par_iter()
            .for_each(|chunk| {
                let out = &out;
                for r in chunk * ROWS_PER_TASK..((chunk + 1) * ROWS_PER_TASK).min(n) {
                    let wrow = &self.blocks[r * row_bytes..(r + 1) * row_bytes];
                    // SAFETY: each (j, r) is written by exactly one task (r is unique to
                    // this chunk), and j < m, r < n keep it inside y.
                    let store = |j: usize, v: f32| unsafe { *out.0.add(j * n + r) = v };
                    match scalar {
                        ScalarKernel::Portable => rows_portable(wrow, &xq, &xd, m, k, store),
                        // SAFETY: the variant is only selected when the CPU supports it.
                        ScalarKernel::Avx2 => unsafe { rows_avx2(wrow, &xq, &xd, m, k, store) },
                    }
                }
            });
        y
    }

    /// `Y = X Wᵀ` for `m` rows via dequantize + gemm.
    pub fn matmul(&self, x: &[f32], m: usize) -> Vec<f32> {
        let row_bytes = self.row_bytes();
        dequant_matmul(x, m, self.out_features, self.in_features, |r, row| {
            dequantize_row(&self.blocks[r * row_bytes..(r + 1) * row_bytes], row)
        })
    }
}

/// `Y = X Wᵀ` for `m` rows of `x` (`m × k`) and an `n × k` weight given row by
/// row through `dequant_row(r, out)`: the rows are dequantized in parallel into
/// the shared scratch buffer, then multiplied with f32 `gemm`. Long prefills of
/// every packed format go through here.
pub(super) fn dequant_matmul(
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
    dequant_row: impl Fn(usize, &mut [f32]) + Sync,
) -> Vec<f32> {
    let mut scratch = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if scratch.len() < n * k {
        scratch.resize(n * k, 0.0);
    }
    let w = &mut scratch[..n * k];
    w.par_chunks_mut(k)
        .enumerate()
        .for_each(|(r, row)| dequant_row(r, row));

    let mut y = vec![0f32; m * n];
    // SAFETY: all pointers cover the stated shapes: x is m×k row-major (rs = k,
    // cs = 1), W is n×k row-major used as the k×n rhs (rs = 1, cs = k), y is m×n.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            y.as_mut_ptr(),
            1,
            n as isize,
            false,
            x.as_ptr(),
            1,
            k as isize,
            w.as_ptr(),
            k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
    y
}

impl QLinear for PackedQ8Linear {
    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.in_features, "input shape");
        if m == 1 {
            // Decode: bandwidth bound. AVX2 measured ~3% faster than the portable
            // matvec over the 13B's layers; VNNI's per-row setup doesn't pay at m = 1.
            match kernel() {
                Kernel::Portable => self.matvec(x),
                _ => self.batched_with(x, 1, Kernel::Avx2),
            }
        } else if m <= *BATCH_MAX {
            self.batched(x, m)
        } else {
            self.matmul(x, m)
        }
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }
}

/// llama.cpp's `quantize_row_q8_0_ref`: per block, `d = amax / 127` stored as f16,
/// `q = round(x / d)` using the unrounded `d`. Returns values and the f16 scales
/// (as f32).
pub fn quantize_row_q8_0(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    assert!(x.len().is_multiple_of(QK8_0));
    let mut q = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / QK8_0);
    for block in x.as_chunks::<QK8_0>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        d.push(f16::from_f32(scale).to_f32());
        q.extend(block.iter().map(|&v| (v * inv).round() as i8));
    }
    (q, d)
}

/// One output: Σ_blocks d_w · d_x · Σ_i q_w[i] · q_x[i]. Written so the inner
/// 32-wide i8 → i32 product sum auto-vectorises.
#[inline]
fn dot_row(row: &[u8], xq: &[i8], xd: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, block) in row.as_chunks::<Q8_0_BLOCK_BYTES>().0.iter().enumerate() {
        let dw = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let w = &block[2..];
        let x = &xq[b * QK8_0..(b + 1) * QK8_0];
        let mut s = 0i32;
        for i in 0..QK8_0 {
            s += (w[i] as i8 as i32) * (x[i] as i32);
        }
        acc += dw * xd[b] * s as f32;
    }
    acc
}

pub(super) struct SyncPtr(pub(super) *mut f32);
// SAFETY: used only for disjoint writes from rayon tasks (see `batched_with`).
unsafe impl Sync for SyncPtr {}

/// Portable batched kernel: one weight row against `m` activation rows.
fn rows_portable(
    wrow: &[u8],
    xq: &[i8],
    xd: &[f32],
    m: usize,
    k: usize,
    mut store: impl FnMut(usize, f32),
) {
    let nb = k / QK8_0;
    for j in 0..m {
        store(
            j,
            dot_row(wrow, &xq[j * k..(j + 1) * k], &xd[j * nb..(j + 1) * nb]),
        );
    }
}

/// AVX2: `|w|` and `sign(x, w)` make an unsigned × signed pair for
/// `_mm256_maddubs_epi16` (llama.cpp's trick). Our activations are in −127..127, so
/// the i16 pair sums (≤ 2 · 128 · 127) cannot saturate. Rows go in tiles of 4 so
/// each weight block is loaded once per tile while the row stays in L1.
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2(
    wrow: &[u8],
    xq: &[i8],
    xd: &[f32],
    m: usize,
    k: usize,
    mut store: impl FnMut(usize, f32),
) {
    let mut j0 = 0;
    while j0 < m {
        // Compile-time tile sizes keep the accumulators in registers.
        let t = match m - j0 {
            1 => unsafe { tile_avx2::<1>(wrow, xq, xd, j0, k, &mut store) },
            2 => unsafe { tile_avx2::<2>(wrow, xq, xd, j0, k, &mut store) },
            3 => unsafe { tile_avx2::<3>(wrow, xq, xd, j0, k, &mut store) },
            _ => unsafe { tile_avx2::<4>(wrow, xq, xd, j0, k, &mut store) },
        };
        j0 += t;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_avx2<const T: usize>(
    wrow: &[u8],
    xq: &[i8],
    xd: &[f32],
    j0: usize,
    k: usize,
    store: &mut impl FnMut(usize, f32),
) -> usize {
    use std::arch::x86_64::*;
    let nb = k / QK8_0;
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); T];
    for b in 0..nb {
        let blk = &wrow[b * Q8_0_BLOCK_BYTES..];
        let dw = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let w = unsafe { _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i) };
        let aw = _mm256_sign_epi8(w, w);
        for (t, acc) in acc.iter_mut().enumerate() {
            let j = j0 + t;
            let x =
                unsafe { _mm256_loadu_si256(xq.as_ptr().add(j * k + b * QK8_0) as *const __m256i) };
            let sx = _mm256_sign_epi8(x, w);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones);
            *acc = _mm256_fmadd_ps(
                _mm256_set1_ps(dw * xd[j * nb + b]),
                _mm256_cvtepi32_ps(p),
                *acc,
            );
        }
    }
    for (t, a) in acc.iter().enumerate() {
        store(j0 + t, hsum(*a));
    }
    T
}

/// AVX-VNNI, register-tiled: two weight rows × four activation rows per pass.
///
/// `dpbusd` multiplies unsigned × signed bytes, so activations go in as
/// `x + 128` (unsigned) and weights as-is (signed). That avoids llama.cpp's two
/// `sign` operations per pair; the extra `128 · Σw` is cancelled by seeding the
/// accumulator with `−128 · Σw`, itself one `dpbusd` per weight block, shared by
/// all four activation rows. (AVX2's `maddubs` would saturate its i16 pairs on
/// 0..255 × −128..127, so this is VNNI-only.)
/// Per weight row, computed once and reused by every activation tile: the f32
/// block scales and the `−128 · Σw` correction vectors.
#[derive(Default)]
struct RowPrep {
    scale: [Vec<f32>; 2],
    corr: [Vec<std::arch::x86_64::__m256i>; 2],
}

#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn prep_row(
    w: &[u8],
    nb: usize,
    scale: &mut Vec<f32>,
    corr: &mut Vec<std::arch::x86_64::__m256i>,
) {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi8(1);
    let neg128 = _mm256_set1_epi32(-128);
    scale.clear();
    corr.clear();
    for b in 0..nb {
        let o = b * Q8_0_BLOCK_BYTES;
        scale.push(f16::from_le_bytes([w[o], w[o + 1]]).to_f32());
        let a = unsafe { _mm256_loadu_si256(w.as_ptr().add(o + 2) as *const __m256i) };
        // dpbusd(0, 1, w) sums w in groups of four; × −128 gives the correction.
        corr.push(_mm256_mullo_epi32(
            _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones, a),
            neg128,
        ));
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn rows_vnni_2x4(
    w0: &[u8],
    w1: &[u8],
    xu: &[u8],
    xd: &[f32],
    m: usize,
    k: usize,
    prep: &mut RowPrep,
    mut store: impl FnMut(usize, f32, f32),
) {
    let nb = k / QK8_0;
    let RowPrep {
        scale: [s0, s1],
        corr: [c0, c1],
    } = prep;
    unsafe {
        prep_row(w0, nb, s0, c0);
        prep_row(w1, nb, s1, c1);
    }
    let rows = Rows {
        w0,
        w1,
        s0,
        s1,
        c0,
        c1,
    };
    let mut j0 = 0;
    while j0 < m {
        let t = match m - j0 {
            1 => unsafe { tile_vnni::<1>(&rows, xu, xd, j0, k, &mut store) },
            2 => unsafe { tile_vnni::<2>(&rows, xu, xd, j0, k, &mut store) },
            3 => unsafe { tile_vnni::<3>(&rows, xu, xd, j0, k, &mut store) },
            _ => unsafe { tile_vnni::<4>(&rows, xu, xd, j0, k, &mut store) },
        };
        j0 += t;
    }
}

struct Rows<'a> {
    w0: &'a [u8],
    w1: &'a [u8],
    s0: &'a [f32],
    s1: &'a [f32],
    c0: &'a [std::arch::x86_64::__m256i],
    c1: &'a [std::arch::x86_64::__m256i],
}

#[inline]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn tile_vnni<const T: usize>(
    rows: &Rows<'_>,
    xu: &[u8],
    xd: &[f32],
    j0: usize,
    k: usize,
    store: &mut impl FnMut(usize, f32, f32),
) -> usize {
    use std::arch::x86_64::*;
    let nb = k / QK8_0;
    let mut acc0 = [_mm256_setzero_ps(); T];
    let mut acc1 = [_mm256_setzero_ps(); T];
    for b in 0..nb {
        let o = b * Q8_0_BLOCK_BYTES + 2;
        let a0 = unsafe { _mm256_loadu_si256(rows.w0.as_ptr().add(o) as *const __m256i) };
        let a1 = unsafe { _mm256_loadu_si256(rows.w1.as_ptr().add(o) as *const __m256i) };
        let (d0, d1) = (_mm256_set1_ps(rows.s0[b]), _mm256_set1_ps(rows.s1[b]));
        let (c0, c1) = (rows.c0[b], rows.c1[b]);
        for t in 0..T {
            let j = j0 + t;
            let x =
                unsafe { _mm256_loadu_si256(xu.as_ptr().add(j * k + b * QK8_0) as *const __m256i) };
            let dx = _mm256_set1_ps(xd[j * nb + b]);
            let p0 = _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(c0, x, a0));
            let p1 = _mm256_cvtepi32_ps(_mm256_dpbusd_avx_epi32(c1, x, a1));
            acc0[t] = _mm256_fmadd_ps(_mm256_mul_ps(d0, dx), p0, acc0[t]);
            acc1[t] = _mm256_fmadd_ps(_mm256_mul_ps(d1, dx), p1, acc1[t]);
        }
    }
    for t in 0..T {
        store(j0 + t, hsum(acc0[t]), hsum(acc1[t]));
    }
    T
}

#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn hsum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

fn dequantize_row(row: &[u8], out: &mut [f32]) {
    for (block, out) in row
        .as_chunks::<Q8_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<QK8_0>().0)
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for (o, &q) in out.iter_mut().zip(&block[2..]) {
            *o = d * (q as i8) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{deinterleave_q8_0, q8_0_scheme};
    use crate::model::BurnQLinear;
    use proptest::prelude::*;

    fn weights(out_f: usize, in_f: usize, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut bytes = Vec::new();
        for _ in 0..out_f * in_f / QK8_0 {
            let d = (next() % 1000) as f32 * 1e-5 + 1e-4;
            bytes.extend(f16::from_f32(d).to_le_bytes());
            bytes.extend((0..QK8_0).map(|_| next() as u8));
        }
        bytes
    }

    fn activations(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as u64 * 2654435761 + seed) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// Straightforward scalar version of the same computation as `matvec`.
    fn reference_matvec(bytes: &[u8], in_f: usize, out_f: usize, x: &[f32]) -> Vec<f32> {
        let (xq, xd) = quantize_row_q8_0(x);
        let row = in_f / QK8_0 * Q8_0_BLOCK_BYTES;
        (0..out_f)
            .map(|r| {
                let mut acc = 0f32;
                for b in 0..in_f / QK8_0 {
                    let blk = &bytes[r * row + b * Q8_0_BLOCK_BYTES..][..Q8_0_BLOCK_BYTES];
                    let dw = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let s: i32 = (0..QK8_0)
                        .map(|i| blk[2 + i] as i8 as i32 * xq[b * QK8_0 + i] as i32)
                        .sum();
                    acc += dw * xd[b] * s as f32;
                }
                acc
            })
            .collect()
    }

    fn burn_reference(bytes: &[u8], in_f: usize, out_f: usize, x: &[f32], m: usize) -> Vec<f32> {
        let (v, s) = deinterleave_q8_0(bytes);
        let w = burn::tensor::Tensor::<2>::from_data(
            burn::tensor::TensorData::quantized(v, [out_f, in_f], q8_0_scheme(), &s, None),
            &Default::default(),
        );
        BurnQLinear::new(w).forward(x, m)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// Exact check: same math as the scalar reference; only f32 summation
        /// order could differ, and here it doesn't.
        #[test]
        fn matvec_equals_scalar_reference(out_f in 1usize..200, in_blocks in 1usize..12, seed in any::<u64>()) {
            let in_f = in_blocks * QK8_0;
            let bytes = weights(out_f, in_f, seed);
            let x = activations(in_f, seed);
            let lin = PackedQ8Linear::new(&bytes, in_f, out_f);
            prop_assert_eq!(lin.matvec(&x), reference_matvec(&bytes, in_f, out_f, &x));
        }

        /// Approximate check against the f32-activation reference. Rounding each
        /// activation to Q8_0 moves it by at most d_x / 2, so each output may differ by
        /// at most Σ_i |w_i| · d_x(i) / 2. A layout bug (wrong block, wrong row) breaks
        /// that bound; activation rounding can't.
        #[test]
        fn matvec_within_activation_rounding_bound(out_f in 1usize..100, in_blocks in 1usize..12, seed in any::<u64>()) {
            let in_f = in_blocks * QK8_0;
            let bytes = weights(out_f, in_f, seed);
            let x = activations(in_f, seed);
            let got = PackedQ8Linear::new(&bytes, in_f, out_f).matvec(&x);
            let want = burn_reference(&bytes, in_f, out_f, &x, 1);
            let w = crate::gguf::decode_q8_0(&bytes);
            let (_, xd) = quantize_row_q8_0(&x);
            for r in 0..out_f {
                let bound: f32 = (0..in_f).map(|i| w[r * in_f + i].abs() * xd[i / QK8_0] / 2.0).sum();
                let err = (got[r] - want[r]).abs();
                prop_assert!(err <= bound * 1.01 + 1e-6, "row {r}: err {err} > bound {bound}");
            }
        }

        /// The prefill path keeps activations in f32, so it matches the reference closely.
        #[test]
        fn matmul_matches_burn(m in 2usize..9, out_f in 1usize..100, in_blocks in 1usize..8, seed in any::<u64>()) {
            let in_f = in_blocks * QK8_0;
            let bytes = weights(out_f, in_f, seed);
            let x = activations(m * in_f, seed);
            let got = PackedQ8Linear::new(&bytes, in_f, out_f).matmul(&x, m);
            let want = burn_reference(&bytes, in_f, out_f, &x, m);
            for (g, w) in got.iter().zip(&want) {
                prop_assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Every batched kernel equals the scalar reference row by row, up to f32
        /// summation order (SIMD kernels accumulate in 8 lanes).
        #[test]
        fn batched_kernels_match_scalar_reference(
            m in 2usize..11,
            out_f in 1usize..70,
            in_blocks in 1usize..10,
            seed in any::<u64>(),
        ) {
            let in_f = in_blocks * QK8_0;
            let mut bytes = weights(out_f, in_f, seed);
            // Make sure -128 weights occur (the |w| = 128 edge of the sign trick).
            for b in (0..bytes.len()).step_by(Q8_0_BLOCK_BYTES * 3) {
                bytes[b + 2] = 0x80;
            }
            let x = activations(m * in_f, seed);
            let lin = PackedQ8Linear::new(&bytes, in_f, out_f);
            let mut want = Vec::new();
            for j in 0..m {
                want.extend(reference_matvec(&bytes, in_f, out_f, &x[j * in_f..(j + 1) * in_f]));
            }
            let mut kernels = vec![Kernel::Portable];
            if is_x86_feature_detected!("avx2") { kernels.push(Kernel::Avx2); }
            if is_x86_feature_detected!("avxvnni") { kernels.push(Kernel::AvxVnni); }
            for kern in kernels {
                let got = lin.batched_with(&x, m, kern);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    prop_assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "{kern:?} [{i}]: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn quantize_row_handles_zero_block_and_extremes() {
        let mut x = vec![0f32; 64];
        x[32] = -3.0;
        x[33] = 1.5;
        let (q, d) = quantize_row_q8_0(&x);
        assert_eq!(d[0], 0.0);
        assert!(q[..32].iter().all(|&v| v == 0));
        assert_eq!(q[32], -127);
        assert_eq!(q[33], 64); // 1.5 / (3/127) = 63.5, rounds away from zero
    }
}
