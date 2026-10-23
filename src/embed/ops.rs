//! The pieces both towers share: bf16 linear layers, RMS norms, rotary
//! embeddings and bidirectional attention, all in f32 on the CPU.

use anyhow::Result;
use rayon::prelude::*;

use super::safetensors::{SafeTensors, bf16};
use crate::model::packed_q8::dequant_matmul;

/// A bias-free weight `[out, in]`, kept as bf16 and widened row by row for each matmul.
pub struct Linear {
    weight: Vec<u16>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn load(st: &SafeTensors, name: &str, out_features: usize, in_features: usize) -> Result<Self> {
        let weight = st.bf16(&format!("{name}.weight"), &[out_features, in_features])?;
        Ok(Self { weight, in_features, out_features })
    }

    /// `x Wᵀ` for the `m` rows of `x`.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let (n, k) = (self.out_features, self.in_features);
        debug_assert_eq!(x.len(), m * k);
        dequant_matmul(x, m, n, k, |r, row| {
            for (v, &b) in row.iter_mut().zip(&self.weight[r * k..(r + 1) * k]) {
                *v = bf16(b);
            }
        })
    }
}

/// `x · (mean(x²) + eps)^-½ · weight` over each `weight.len()`-wide row; no weight means 1.
pub struct RmsNorm {
    weight: Option<Vec<f32>>,
    dim: usize,
}

impl RmsNorm {
    pub fn load(st: &SafeTensors, name: &str, dim: usize) -> Result<Self> {
        Ok(Self { weight: Some(st.f32(&format!("{name}.weight"), &[dim])?), dim })
    }

    pub const fn plain(dim: usize) -> Self {
        Self { weight: None, dim }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = x.to_vec();
        self.apply(&mut out);
        out
    }

    pub fn apply(&self, x: &mut [f32]) {
        let d = self.dim;
        x.par_chunks_mut(d).for_each(|row| {
            let ms = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / d as f64;
            let inv = (1.0 / (ms + EPS).sqrt()) as f32;
            match &self.weight {
                Some(w) => row.iter_mut().zip(w).for_each(|(v, w)| *v = *v * inv * w),
                None => row.iter_mut().for_each(|v| *v *= inv),
            }
        });
    }
}

/// `rms_norm_eps` of both towers.
const EPS: f64 = 1e-6;

/// `1 / theta^(2i / dim)` for `i < dim / 2`, in f32 as torch computes it.
pub fn inv_freq(theta: f32, dim: usize) -> Vec<f32> {
    (0..dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32)).collect()
}

/// Rotary embeddings in place on `x` (`[n, width]`, heads of `head_dim`): each head splits into
/// `sections` equal slices, slice `s` rotated (`x · cos + rotate_half(x) · sin`) by its block of
/// `angles(r)` (`head_dim / 2` angles for row `r`, `head_dim / sections / 2` per slice).
pub fn rope(
    x: &mut [f32],
    width: usize,
    head_dim: usize,
    sections: usize,
    angles: impl Fn(usize) -> Vec<f32> + Sync,
) {
    let f = head_dim / sections / 2;
    x.par_chunks_mut(width).enumerate().for_each(|(r, row)| {
        let (cos, sin): (Vec<f32>, Vec<f32>) =
            angles(r).iter().map(|&t| (f64::from(t).cos() as f32, f64::from(t).sin() as f32)).unzip();
        debug_assert_eq!(cos.len(), head_dim / 2);
        for head in row.chunks_exact_mut(head_dim) {
            for (s, slice) in head.chunks_exact_mut(2 * f).enumerate() {
                let (lo, hi) = slice.split_at_mut(f);
                let (cos, sin) = (&cos[s * f..(s + 1) * f], &sin[s * f..(s + 1) * f]);
                for i in 0..f {
                    let (x1, x2) = (lo[i], hi[i]);
                    lo[i] = x1 * cos[i] - x2 * sin[i];
                    hi[i] = x2 * cos[i] + x1 * sin[i];
                }
            }
        }
    });
}

/// Bidirectional attention with scale 1 and grouped key/value heads: `q` is `[n, heads · dh]`,
/// `k` and `v` `[n, kv_heads · dh]`. A `window` keeps only keys within that many positions of
/// the query (inclusive).
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    (heads, kv_heads, dh): (usize, usize, usize),
    window: Option<usize>,
) -> Vec<f32> {
    let (qd, kd) = (heads * dh, kv_heads * dh);
    let group = heads / kv_heads;
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let kh = h / group;
            let mut s = vec![0f32; n * n];
            // SAFETY: q is n × qd and k n × kd, row-major; head h's query columns start at h·dh
            // (row stride qd) and key head kh's at kh·dh, read as Kᵀ (dh × n) with element
            // (c, j) at j·kd + c. s is n × n.
            unsafe {
                gemm::gemm(
                    n, n, dh,
                    s.as_mut_ptr(), 1, n as isize, false,
                    q.as_ptr().add(h * dh), 1, qd as isize,
                    k.as_ptr().add(kh * dh), kd as isize, 1,
                    0.0, 1.0, false, false, false,
                    gemm::Parallelism::None,
                );
            }
            for (i, row) in s.chunks_exact_mut(n).enumerate() {
                if let Some(w) = window {
                    for (j, v) in row.iter_mut().enumerate() {
                        if i.abs_diff(j) > w {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }
                let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for v in row.iter_mut() {
                    *v = (*v - m).exp();
                    sum += *v;
                }
                let inv = 1.0 / sum;
                row.iter_mut().for_each(|v| *v *= inv);
            }
            let mut o = vec![0f32; n * dh];
            // SAFETY: s is n × n row-major; value head kh (n × dh) has element (j, c) at
            // j·kd + kh·dh + c; o is n × dh row-major.
            unsafe {
                gemm::gemm(
                    n, dh, n,
                    o.as_mut_ptr(), 1, dh as isize, false,
                    s.as_ptr(), 1, n as isize,
                    v.as_ptr().add(kh * dh), 1, kd as isize,
                    0.0, 1.0, false, false, false,
                    gemm::Parallelism::None,
                );
            }
            o
        })
        .collect();
    let mut out = vec![0f32; n * qd];
    out.par_chunks_mut(qd).enumerate().for_each(|(r, row)| {
        for (h, o) in per_head.iter().enumerate() {
            row[h * dh..(h + 1) * dh].copy_from_slice(&o[r * dh..(r + 1) * dh]);
        }
    });
    out
}

/// `gelu_pytorch_tanh`.
pub fn gelu_tanh(x: f32) -> f32 {
    let u = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    x / (1.0 + (-2.0 * u).exp())
}

/// `down(gelu_tanh(gate(x)) · up(x))`.
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl Mlp {
    pub fn forward(&self, x: &[f32], n: usize) -> Vec<f32> {
        let mut g = self.gate.forward(x, n);
        let u = self.up.forward(x, n);
        g.par_iter_mut().zip(&u).for_each(|(g, u)| *g = gelu_tanh(*g) * u);
        self.down.forward(&g, n)
    }
}

pub fn add(x: &mut [f32], y: &[f32]) {
    x.par_iter_mut().zip(y).for_each(|(a, b)| *a += b);
}
