//! `PackedKLinear` (design.md §3): GGUF Q4_K or Q6_K blocks kept as-is, for the
//! Q4_K_M files. Decode reads about 4.6–6.6 bits per weight instead of Q8_0's
//! 8.5, so it runs proportionally faster where memory bandwidth is the limit.
//!
//! As in llama.cpp, activations are quantized per 256-element block to 8-bit
//! (`Q8K`: one f32 scale, i8 values, and the sum of every 16 values), so the dot
//! products run in integers:
//! - Q4_K: `Σ_j d·sc_j·Σ q4·q8 − dmin·Σ_j min_j·Σ q8`, the second term from the
//!   precomputed sums;
//! - Q6_K: `d·Σ_g sc_g·(Σ q6·q8 − 32·Σ q8)` with `q6` in 0..64.
//!
//! With AVX2, `maddubs` multiplies the unsigned 4/6-bit weights by the signed
//! activations (pairs stay far from i16 saturation: ≤ 2·63·128), and `madd`
//! applies the per-sub-block scales. Each weight row is unpacked once per tile of
//! up to four activation rows. Long prefills dequantize to f32 and use `gemm`.

use std::sync::LazyLock;

use rayon::prelude::*;

use super::linear::QLinear;
use super::packed_q8::{Kernel, SyncPtr, dequant_matmul, hsum, kernel};
use crate::gguf::kquant::{
    Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K, decode_q4_k_block, decode_q6_k_block, q4_k_parts,
    q6_k_parts, q6_k_value,
};

/// Output rows per rayon task.
const ROWS_PER_TASK: usize = 64;

/// Above this many activation rows, dequantize + gemm instead of the integer
/// kernel. Overridable with `INCREBOT_K_BATCH_MAX`.
static BATCH_MAX: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("INCREBOT_K_BATCH_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(match kernel() {
            Kernel::Portable => 1,
            _ => 64,
        })
});

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KQuant {
    Q4K,
    Q6K,
}

impl KQuant {
    pub fn block_bytes(self) -> usize {
        match self {
            KQuant::Q4K => Q4_K_BLOCK_BYTES,
            KQuant::Q6K => Q6_K_BLOCK_BYTES,
        }
    }

    fn decode_block(self, b: &[u8], out: &mut [f32]) {
        match self {
            KQuant::Q4K => decode_q4_k_block(b, out),
            KQuant::Q6K => decode_q6_k_block(b, out),
        }
    }
}

/// Activation rows quantized per 256-element block (llama.cpp's `block_q8_K`).
pub struct Q8K {
    /// `[m · k]` values in −127..=127.
    pub qs: Vec<i8>,
    /// One scale per block: `x ≈ d · q`.
    pub d: Vec<f32>,
    /// The sum of every 16 consecutive values.
    pub bsums: Vec<i32>,
}

/// Quantizes rows (any number, concatenated) whose length is a multiple of 256.
pub fn quantize_q8_k(x: &[f32]) -> Q8K {
    assert!(
        x.len().is_multiple_of(QK_K),
        "length not a multiple of {QK_K}"
    );
    let mut qs = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / QK_K);
    let mut bsums = Vec::with_capacity(x.len() / 16);
    for block in x.as_chunks::<QK_K>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        d.push(scale);
        let start = qs.len();
        qs.extend(
            block
                .iter()
                .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8),
        );
        bsums.extend(
            qs[start..]
                .as_chunks::<16>()
                .0
                .iter()
                .map(|g| g.iter().map(|&q| q as i32).sum::<i32>()),
        );
    }
    Q8K { qs, d, bsums }
}

/// Scalar Q4_K × Q8K dot product of one weight row with activation row `j`:
/// the same integer sums as the AVX2 kernel, so results agree to rounding order.
pub fn dot_q4k_scalar(row: &[u8], x: &Q8K, j: usize, k: usize) -> f32 {
    let nb = k / QK_K;
    let mut acc = 0f32;
    for (b, blk) in row.as_chunks::<Q4_K_BLOCK_BYTES>().0.iter().enumerate() {
        let (d, dmin, sc, mn, qs) = q4_k_parts(blk);
        let xq = &x.qs[j * k + b * QK_K..];
        let bs = &x.bsums[(j * nb + b) * 16..];
        let mut sumi = 0i32;
        for c in 0..4 {
            let (mut lo, mut hi) = (0i32, 0i32);
            for l in 0..32 {
                let q = qs[c * 32 + l];
                lo += (q & 0x0F) as i32 * xq[c * 64 + l] as i32;
                hi += (q >> 4) as i32 * xq[c * 64 + 32 + l] as i32;
            }
            sumi += sc[2 * c] as i32 * lo + sc[2 * c + 1] as i32 * hi;
        }
        let summ: i32 = (0..8)
            .map(|g| mn[g] as i32 * (bs[2 * g] + bs[2 * g + 1]))
            .sum();
        let xd = x.d[j * nb + b];
        acc += d * xd * sumi as f32 - dmin * xd * summ as f32;
    }
    acc
}

/// Scalar Q6_K × Q8K dot product (see [`dot_q4k_scalar`]).
pub fn dot_q6k_scalar(row: &[u8], x: &Q8K, j: usize, k: usize) -> f32 {
    let nb = k / QK_K;
    let mut acc = 0f32;
    for (b, blk) in row.as_chunks::<Q6_K_BLOCK_BYTES>().0.iter().enumerate() {
        let (d, sc, ql, qh) = q6_k_parts(blk);
        let xq = &x.qs[j * k + b * QK_K..];
        let bs = &x.bsums[(j * nb + b) * 16..];
        let mut sumi = 0i32;
        for g in 0..16 {
            let s: i32 = (g * 16..(g + 1) * 16)
                .zip(&xq[g * 16..])
                .map(|(e, &q)| q6_k_value(ql, qh, e) as i32 * q as i32)
                .sum();
            sumi += sc[g] as i32 * (s - 32 * bs[g]);
        }
        acc += d * x.d[j * nb + b] * sumi as f32;
    }
    acc
}

pub struct PackedKLinear {
    kind: KQuant,
    /// `out` rows of `in / 256` blocks, exactly as in the GGUF.
    blocks: Vec<u8>,
    in_features: usize,
    out_features: usize,
}

impl PackedKLinear {
    pub fn new(kind: KQuant, bytes: &[u8], in_features: usize, out_features: usize) -> Self {
        assert!(
            in_features.is_multiple_of(QK_K),
            "in_features {in_features} is not a multiple of {QK_K}"
        );
        assert_eq!(
            bytes.len(),
            out_features * in_features / QK_K * kind.block_bytes(),
            "{kind:?} byte count"
        );
        Self {
            kind,
            blocks: bytes.to_vec(),
            in_features,
            out_features,
        }
    }

    pub fn kind(&self) -> KQuant {
        self.kind
    }

    fn row(&self, r: usize) -> &[u8] {
        let rb = self.in_features / QK_K * self.kind.block_bytes();
        &self.blocks[r * rb..(r + 1) * rb]
    }

    /// `Y = X Wᵀ` for `m` rows with the integer kernels (`kernel` chooses scalar
    /// or AVX2).
    pub fn integer_with(&self, x: &[f32], m: usize, kernel: Kernel) -> Vec<f32> {
        let (k, n) = (self.in_features, self.out_features);
        assert_eq!(x.len(), m * k);
        let xq = quantize_q8_k(x);
        let mut y = vec![0f32; m * n];
        let out = SyncPtr(y.as_mut_ptr());
        (0..n.div_ceil(ROWS_PER_TASK))
            .into_par_iter()
            .for_each(|chunk| {
                let out = &out;
                for r in chunk * ROWS_PER_TASK..((chunk + 1) * ROWS_PER_TASK).min(n) {
                    let row = self.row(r);
                    // SAFETY: each (j, r) is written by exactly one task (r is unique
                    // to this chunk), and j < m, r < n keep it inside y.
                    let store = |j: usize, v: f32| unsafe { *out.0.add(j * n + r) = v };
                    match kernel {
                        Kernel::Portable => {
                            for j in 0..m {
                                store(
                                    j,
                                    match self.kind {
                                        KQuant::Q4K => dot_q4k_scalar(row, &xq, j, k),
                                        KQuant::Q6K => dot_q6k_scalar(row, &xq, j, k),
                                    },
                                );
                            }
                        }
                        // SAFETY: AVX2 is present whenever an AVX2/VNNI kernel was chosen.
                        _ => unsafe { rows_avx2(self.kind, row, &xq, m, k, store) },
                    }
                }
            });
        y
    }

    /// `Y = X Wᵀ` for `m` rows via dequantize + gemm.
    pub fn matmul(&self, x: &[f32], m: usize) -> Vec<f32> {
        let bb = self.kind.block_bytes();
        dequant_matmul(x, m, self.out_features, self.in_features, |r, out| {
            for (blk, o) in self
                .row(r)
                .chunks_exact(bb)
                .zip(out.as_chunks_mut::<QK_K>().0)
            {
                self.kind.decode_block(blk, o);
            }
        })
    }
}

impl QLinear for PackedKLinear {
    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.in_features, "input shape");
        if m <= *BATCH_MAX {
            self.integer_with(x, m, kernel())
        } else {
            self.matmul(x, m)
        }
    }

    fn k_blocks(&self) -> Option<(KQuant, &[u8])> {
        Some((self.kind, &self.blocks))
    }

    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }
}

/// One weight row against `m` activation rows, in tiles of up to four.
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2(
    kind: KQuant,
    row: &[u8],
    x: &Q8K,
    m: usize,
    k: usize,
    mut store: impl FnMut(usize, f32),
) {
    let mut j0 = 0;
    while j0 < m {
        // Compile-time tile sizes keep the accumulators in registers.
        let t = unsafe {
            match (kind, m - j0) {
                (KQuant::Q4K, 1) => tile_q4k::<1>(row, x, j0, k, &mut store),
                (KQuant::Q4K, 2) => tile_q4k::<2>(row, x, j0, k, &mut store),
                (KQuant::Q4K, 3) => tile_q4k::<3>(row, x, j0, k, &mut store),
                (KQuant::Q4K, _) => tile_q4k::<4>(row, x, j0, k, &mut store),
                (KQuant::Q6K, 1) => tile_q6k::<1>(row, x, j0, k, &mut store),
                (KQuant::Q6K, 2) => tile_q6k::<2>(row, x, j0, k, &mut store),
                (KQuant::Q6K, 3) => tile_q6k::<3>(row, x, j0, k, &mut store),
                (KQuant::Q6K, _) => tile_q6k::<4>(row, x, j0, k, &mut store),
            }
        };
        j0 += t;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_q4k<const T: usize>(
    row: &[u8],
    x: &Q8K,
    j0: usize,
    k: usize,
    store: &mut impl FnMut(usize, f32),
) -> usize {
    use std::arch::x86_64::*;
    let nb = k / QK_K;
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); T];
    let mut acc_min = [0f32; T];
    for b in 0..nb {
        let (d, dmin, sc, mn, qs) = q4_k_parts(&row[b * Q4_K_BLOCK_BYTES..]);
        let mut sumi = [_mm256_setzero_si256(); T];
        for c in 0..4 {
            let q = unsafe { _mm256_loadu_si256(qs.as_ptr().add(c * 32) as *const __m256i) };
            let lo = _mm256_and_si256(q, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            let s_lo = _mm256_set1_epi16(sc[2 * c] as i16);
            let s_hi = _mm256_set1_epi16(sc[2 * c + 1] as i16);
            for (t, s) in sumi.iter_mut().enumerate() {
                let p = unsafe { x.qs.as_ptr().add((j0 + t) * k + b * QK_K + c * 64) };
                let xl = unsafe { _mm256_loadu_si256(p as *const __m256i) };
                let xh = unsafe { _mm256_loadu_si256(p.add(32) as *const __m256i) };
                let pl = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xl), s_lo);
                let ph = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xh), s_hi);
                *s = _mm256_add_epi32(*s, _mm256_add_epi32(pl, ph));
            }
        }
        for t in 0..T {
            let jb = (j0 + t) * nb + b;
            let bs = &x.bsums[jb * 16..jb * 16 + 16];
            let summ: i32 = (0..8)
                .map(|g| mn[g] as i32 * (bs[2 * g] + bs[2 * g + 1]))
                .sum();
            let xd = x.d[jb];
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * xd), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            acc_min[t] += dmin * xd * summ as f32;
        }
    }
    for t in 0..T {
        store(j0 + t, hsum(acc[t]) - acc_min[t]);
    }
    T
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_q6k<const T: usize>(
    row: &[u8],
    x: &Q8K,
    j0: usize,
    k: usize,
    store: &mut impl FnMut(usize, f32),
) -> usize {
    use std::arch::x86_64::*;
    let nb = k / QK_K;
    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(3);
    let mut acc = [_mm256_setzero_ps(); T];
    let mut acc_bias = [0f32; T];
    for b in 0..nb {
        let (d, sc, ql, qh) = q6_k_parts(&row[b * Q6_K_BLOCK_BYTES..]);
        let mut sumi = [_mm256_setzero_si256(); T];
        for h in 0..2 {
            let load = |p: &[u8], o: usize| unsafe {
                _mm256_loadu_si256(p.as_ptr().add(o) as *const __m256i)
            };
            let (qhv, ql0, ql1) = (load(qh, h * 32), load(ql, h * 64), load(ql, h * 64 + 32));
            // The four 32-element quarters of this half (see `q6_k_value`). Masking
            // before the 16-bit shifts keeps bits from crossing byte boundaries.
            let hi = |s: i32| {
                let v = match s {
                    0 => qhv,
                    2 => _mm256_srli_epi16(qhv, 2),
                    4 => _mm256_srli_epi16(qhv, 4),
                    _ => _mm256_srli_epi16(qhv, 6),
                };
                _mm256_slli_epi16(_mm256_and_si256(v, m2), 4)
            };
            let quarters = [
                _mm256_or_si256(_mm256_and_si256(ql0, m4), hi(0)),
                _mm256_or_si256(_mm256_and_si256(ql1, m4), hi(2)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), hi(4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), hi(6)),
            ];
            for (quarter, &q) in quarters.iter().enumerate() {
                let g = h * 8 + quarter * 2;
                // Lanes 0..8 hold elements 0..16 of the quarter (scale g), 8..16 the rest.
                let scales = _mm256_set_m128i(
                    _mm_set1_epi16(sc[g + 1] as i16),
                    _mm_set1_epi16(sc[g] as i16),
                );
                for (t, s) in sumi.iter_mut().enumerate() {
                    let o = (j0 + t) * k + b * QK_K + h * 128 + quarter * 32;
                    let xv = unsafe { _mm256_loadu_si256(x.qs.as_ptr().add(o) as *const __m256i) };
                    *s = _mm256_add_epi32(
                        *s,
                        _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), scales),
                    );
                }
            }
        }
        for t in 0..T {
            let jb = (j0 + t) * nb + b;
            let bs = &x.bsums[jb * 16..jb * 16 + 16];
            let bias: i32 = (0..16).map(|g| sc[g] as i32 * bs[g]).sum();
            let dx = d * x.d[jb];
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            acc_bias[t] += dx * (32 * bias) as f32;
        }
    }
    for t in 0..T {
        store(j0 + t, hsum(acc[t]) - acc_bias[t]);
    }
    T
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::kquant::{decode_q4_k, decode_q6_k, encode};
    use proptest::prelude::*;

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Random weights: `rows × blocks` blocks of `kind`, built with the encoders
    /// so every scale, min and value range (including extremes) is exercised.
    fn weights(kind: KQuant, rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let mut r = rng(seed);
        let mut out = Vec::new();
        for _ in 0..rows * blocks {
            match kind {
                KQuant::Q4K => {
                    let q: [u8; QK_K] = std::array::from_fn(|_| (r() % 16) as u8);
                    let sc = std::array::from_fn(|_| (r() % 64) as u8);
                    let mn = std::array::from_fn(|_| (r() % 64) as u8);
                    let d = (r() % 1000) as f32 * 1e-5;
                    let dmin = (r() % 1000) as f32 * 1e-5;
                    out.extend(encode::q4_k(d, dmin, sc, mn, &q));
                }
                KQuant::Q6K => {
                    let q: [u8; QK_K] = std::array::from_fn(|_| (r() % 64) as u8);
                    let sc = std::array::from_fn(|_| (r() % 256) as u8 as i8);
                    let d = (r() % 1000) as f32 * 1e-5;
                    out.extend(encode::q6_k(d, sc, &q));
                }
            }
        }
        out
    }

    fn activations(n: usize, seed: u64) -> Vec<f32> {
        let mut r = rng(seed);
        (0..n)
            .map(|_| ((r() % 20_001) as f32 - 10_000.0) / 2_500.0)
            .collect()
    }

    /// Exact f32 reference: decoded weights · the quantized activations decoded.
    fn reference(kind: KQuant, w: &[u8], k: usize, n: usize, x: &[f32], m: usize) -> Vec<f32> {
        let wf = match kind {
            KQuant::Q4K => decode_q4_k(w),
            KQuant::Q6K => decode_q6_k(w),
        };
        let xq = quantize_q8_k(x);
        let nb = k / QK_K;
        let mut y = vec![0f32; m * n];
        for j in 0..m {
            for r in 0..n {
                let mut s = 0f64;
                for i in 0..k {
                    s += wf[r * k + i] as f64
                        * (xq.d[j * nb + i / QK_K] * xq.qs[j * k + i] as f32) as f64;
                }
                y[j * n + r] = s as f32;
            }
        }
        y
    }

    fn close(a: &[f32], b: &[f32], what: &str) {
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= 2e-4 * scale,
                "{what}[{i}]: {x} vs {y} (scale {scale})"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        #[test]
        fn integer_kernels_match_the_f32_reference(
            q6 in any::<bool>(),
            m in 1usize..7,
            n in 1usize..80,
            blocks in 1usize..4,
            seed in any::<u64>(),
        ) {
            let kind = if q6 { KQuant::Q6K } else { KQuant::Q4K };
            let k = blocks * QK_K;
            let w = weights(kind, n, blocks, seed);
            let x = activations(m * k, seed ^ 0x5555);
            let lin = PackedKLinear::new(kind, &w, k, n);
            let want = reference(kind, &w, k, n, &x, m);
            close(&lin.integer_with(&x, m, Kernel::Portable), &want, "scalar");
            if is_x86_feature_detected!("avx2") {
                close(&lin.integer_with(&x, m, Kernel::Avx2), &want, "avx2");
            }
        }
    }

    #[test]
    fn dequant_gemm_path_matches_f32_weights() {
        for kind in [KQuant::Q4K, KQuant::Q6K] {
            let (n, blocks, m) = (37, 2, 5);
            let k = blocks * QK_K;
            let w = weights(kind, n, blocks, 7);
            let x = activations(m * k, 9);
            let wf = match kind {
                KQuant::Q4K => decode_q4_k(&w),
                KQuant::Q6K => decode_q6_k(&w),
            };
            let want: Vec<f32> = (0..m * n)
                .map(|i| {
                    let (j, r) = (i / n, i % n);
                    (0..k)
                        .map(|c| wf[r * k + c] as f64 * x[j * k + c] as f64)
                        .sum::<f64>() as f32
                })
                .collect();
            close(
                &PackedKLinear::new(kind, &w, k, n).matmul(&x, m),
                &want,
                "gemm",
            );
        }
    }

    #[test]
    fn q8k_quantization_and_sums() {
        let mut x = activations(512, 3);
        x[300] = 0.0;
        x[..256].iter_mut().for_each(|v| *v = 0.0);
        let q = quantize_q8_k(&x);
        assert_eq!((q.qs.len(), q.d.len(), q.bsums.len()), (512, 2, 32));
        assert_eq!(q.d[0], 0.0);
        assert!(q.qs[..256].iter().all(|&v| v == 0));
        assert!(q.qs.iter().all(|&v| v >= -127));
        assert!(q.qs[256..].iter().any(|&v| v.abs() == 127));
        for (g, &s) in q.bsums.iter().enumerate() {
            assert_eq!(
                s,
                q.qs[g * 16..g * 16 + 16]
                    .iter()
                    .map(|&v| v as i32)
                    .sum::<i32>()
            );
        }
    }
}
