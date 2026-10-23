//! d1's SigLIP2 (NaFlex) vision tower and its LFM2 projector on the CPU,
//! loaded from llama.cpp's `mmproj` GGUF: per tile, patch embedding plus the
//! 16 × 16 position grid resized to the tile's patch grid, 27 pre-norm encoder
//! layers with bidirectional attention, the post layer norm, then 2 × 2 pixel
//! unshuffle and a GELU MLP into the language model's width.
//!
//! The attention projections and `ffn_up` are Q8_0 (`PackedQ8Linear`),
//! `ffn_down` f16 and the patch embedding f32; every matmul dequantizes to f32
//! (`dequant_matmul`), so activations are never quantized.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

use super::{PATCH_VALUES, Processed};
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::model::packed_q8::dequant_matmul;
use crate::model::{PackedQ8Linear, QLinear};

/// A weight `[out, in]` with its bias.
pub(crate) struct Linear {
    weight: Weight,
    bias: Vec<f32>,
    /// Keeps the activations f32 on a Q8 weight instead of quantizing them to
    /// int8 too: the projector's input (the tower's outliers) loses ~4% RMS
    /// that way, against ~1% from the weights alone.
    exact: bool,
}

enum Weight {
    Q8(PackedQ8Linear),
    /// f16 rows, little-endian, as in the GGUF.
    F16 { bytes: Vec<u8>, in_features: usize, out_features: usize },
    F32 { values: Vec<f32>, in_features: usize, out_features: usize },
}

impl Linear {
    pub(crate) fn load(g: &Gguf, name: &str) -> Result<Self> {
        let info = g.tensor(&format!("{name}.weight"))?;
        let shape = info.shape();
        ensure!(shape.len() == 2, "{name}: shape {shape:?}");
        let (out_features, in_features) = (shape[0], shape[1]);
        let weight = match info.ty {
            GgmlType::Q8_0 => Weight::Q8(PackedQ8Linear::new(g.bytes(info)?, in_features, out_features)),
            GgmlType::F16 => Weight::F16 { bytes: g.bytes(info)?.to_vec(), in_features, out_features },
            _ => Weight::F32 { values: g.to_f32(info)?, in_features, out_features },
        };
        let bias = g.to_f32(g.tensor(&format!("{name}.bias"))?)?;
        ensure!(bias.len() == out_features, "{name}: bias of {}", bias.len());
        Ok(Self { weight, bias, exact: false })
    }

    /// From f32 rows `[out_features, in_features]`.
    pub(crate) fn from_f32(values: Vec<f32>, in_features: usize, bias: Vec<f32>) -> Self {
        let out_features = bias.len();
        assert_eq!(values.len(), in_features * out_features, "weight size");
        Self { weight: Weight::F32 { values, in_features, out_features }, bias, exact: false }
    }

    pub(crate) fn out_features(&self) -> usize {
        self.bias.len()
    }

    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// The weight as little-endian f16 rows `[out, in]`, for the GPU; `None`
    /// for Q8_0.
    pub(crate) fn f16_bytes(&self) -> Option<std::borrow::Cow<'_, [u8]>> {
        match &self.weight {
            Weight::Q8(_) => None,
            Weight::F16 { bytes, .. } => Some(bytes.into()),
            Weight::F32 { values, .. } => {
                Some(values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect::<Vec<u8>>().into())
            }
        }
    }

    /// `x W^T + b` for the `m` rows of `x`.
    pub(crate) fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut y = match &self.weight {
            Weight::Q8(w) if self.exact => w.matmul(x, m),
            Weight::Q8(w) => w.forward(x, m),
            Weight::F16 { bytes, in_features: k, out_features: n } => {
                dequant_matmul(x, m, *n, *k, |r, row| {
                    let src = &bytes[r * k * 2..(r + 1) * k * 2];
                    for (v, b) in row.iter_mut().zip(src.as_chunks::<2>().0) {
                        *v = half::f16::from_le_bytes(*b).to_f32();
                    }
                })
            }
            Weight::F32 { values, in_features: k, out_features: n } => {
                dequant_matmul(x, m, *n, *k, |r, row| row.copy_from_slice(&values[r * k..(r + 1) * k]))
            }
        };
        let n = self.out_features();
        y.par_chunks_mut(n).for_each(|row| {
            for (v, b) in row.iter_mut().zip(&self.bias) {
                *v += b;
            }
        });
        y
    }
}

pub(crate) struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    pub(crate) fn weight(&self) -> &[f32] {
        &self.weight
    }

    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    pub(crate) fn load(g: &Gguf, name: &str) -> Result<Self> {
        Ok(Self {
            weight: g.to_f32(g.tensor(&format!("{name}.weight"))?)?,
            bias: g.to_f32(g.tensor(&format!("{name}.bias"))?)?,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], eps: f32) -> Vec<f32> {
        let d = self.weight.len();
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, x)| {
            let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
            let var = x.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            for (i, o) in o.iter_mut().enumerate() {
                *o = ((f64::from(x[i]) - mean) * inv) as f32 * self.weight[i] + self.bias[i];
            }
        });
        out
    }
}

struct Layer {
    ln1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln2: LayerNorm,
    up: Linear,
    down: Linear,
}

/// The vision tower and projector.
pub struct Vision {
    width: usize,
    heads: usize,
    eps: f32,
    /// `[width, PATCH_VALUES]`, each row in the processor's `[y, x, channel]` order.
    patch_embd: Linear,
    /// The `side × side` position grid, `[side · side, width]`.
    positions: Vec<f32>,
    side: usize,
    layers: Vec<Layer>,
    post_ln: LayerNorm,
    scale_factor: usize,
    mm1: Linear,
    mm2: Linear,
}

impl Vision {
    /// Loads an `mmproj` GGUF with the LFM2 projector.
    pub fn load(path: &Path) -> Result<Self> {
        let g = Gguf::open(path).with_context(|| format!("opening {}", path.display()))?;
        let projector = g.get_str("clip.projector_type")?;
        ensure!(projector == "lfm2", "projector {projector:?}, not lfm2");
        let width = g.get_u64("clip.vision.embedding_length")? as usize;
        let heads = g.get_u64("clip.vision.attention.head_count")? as usize;
        let blocks = g.get_u64("clip.vision.block_count")? as usize;
        let patch = g.get_u64("clip.vision.patch_size")? as usize;
        let eps = g.get_f32("clip.vision.attention.layer_norm_epsilon")?;
        let scale_factor = g.get_u64("clip.vision.projector.scale_factor")? as usize;
        ensure!(patch * patch * 3 == PATCH_VALUES, "patch size {patch}");
        ensure!(width.is_multiple_of(heads), "{width} wide, {heads} heads");

        let patch_embd = patch_embedding(&g, width, patch)?;
        let pos = g.tensor("v.position_embd.weight")?;
        let positions = g.to_f32(pos)?;
        let side = (positions.len() / width).isqrt();
        ensure!(side * side * width == positions.len(), "position grid {:?}", pos.shape());

        let layers = (0..blocks)
            .map(|i| {
                let p = format!("v.blk.{i}");
                Ok(Layer {
                    ln1: LayerNorm::load(&g, &format!("{p}.ln1"))?,
                    q: Linear::load(&g, &format!("{p}.attn_q"))?,
                    k: Linear::load(&g, &format!("{p}.attn_k"))?,
                    v: Linear::load(&g, &format!("{p}.attn_v"))?,
                    out: Linear::load(&g, &format!("{p}.attn_out"))?,
                    ln2: LayerNorm::load(&g, &format!("{p}.ln2"))?,
                    up: Linear::load(&g, &format!("{p}.ffn_up"))?,
                    down: Linear::load(&g, &format!("{p}.ffn_down"))?,
                })
            })
            .collect::<Result<_>>()?;
        let mm1 = Linear { exact: true, ..Linear::load(&g, "mm.1")? };
        let mm2 = Linear { exact: true, ..Linear::load(&g, "mm.2")? };
        ensure!(
            linear_in(&mm1) == width * scale_factor * scale_factor,
            "projector takes {}",
            linear_in(&mm1)
        );
        Ok(Self {
            width,
            heads,
            eps,

            patch_embd,
            positions,
            side,
            layers,
            post_ln: LayerNorm::load(&g, "v.post_ln")?,
            scale_factor,
            mm1,
            mm2,
        })
    }

    /// The language model width the projector writes.
    pub fn out_width(&self) -> usize {
        self.mm2.out_features()
    }

    /// The patch embeddings plus positions of one tile, `[ph · pw, width]`.
    pub fn embed(&self, values: &[f32], (ph, pw): (usize, usize)) -> Vec<f32> {
        let n = ph * pw;
        assert_eq!(values.len(), n * PATCH_VALUES, "patch values");
        let mut x = self.patch_embd.forward(values, n);
        let pos = self.resized_positions(ph, pw);
        x.iter_mut().zip(&pos).for_each(|(a, b)| *a += b);
        x
    }

    /// One encoder layer on `x` (`n` rows).
    pub fn layer(&self, i: usize, x: &mut [f32], n: usize) {
        let l = &self.layers[i];
        let h = l.ln1.forward(x, self.eps);
        let (q, k, v) = (l.q.forward(&h, n), l.k.forward(&h, n), l.v.forward(&h, n));
        let a = attention(&q, &k, &v, n, self.width, self.heads);
        add(x, &l.out.forward(&a, n));
        let h = l.ln2.forward(x, self.eps);
        let mut u = l.up.forward(&h, n);
        u.par_iter_mut().for_each(|v| *v = gelu_tanh(*v));
        add(x, &l.down.forward(&u, n));
    }

    /// The tower on one tile: `[ph · pw, width]` after the post layer norm.
    pub fn encode(&self, values: &[f32], grid: (usize, usize)) -> Vec<f32> {
        let n = grid.0 * grid.1;
        let mut x = self.embed(values, grid);
        for i in 0..self.layers.len() {
            self.layer(i, &mut x, n);
        }
        self.post_ln.forward(&x, self.eps)
    }

    /// The projector on one encoded tile: `[(ph/2) · (pw/2), out_width]`.
    pub fn project(&self, x: &[f32], (ph, pw): (usize, usize)) -> Vec<f32> {
        let f = self.scale_factor;
        let d = self.width;
        let (oh, ow) = (ph.div_ceil(f), pw.div_ceil(f));
        // Pixel unshuffle: each output token takes the f × f block of patches, rows first. A grid
        // that isn't a multiple of f (never from the processor) is zero padded.
        let mut u = vec![0f32; oh * ow * f * f * d];
        for (t, dst) in u.chunks_exact_mut(f * f * d).enumerate() {
            let (i, j) = (t / ow, t % ow);
            for di in 0..f {
                for dj in 0..f {
                    let (r, c) = (f * i + di, f * j + dj);
                    if r < ph && c < pw {
                        let src = &x[(r * pw + c) * d..(r * pw + c + 1) * d];
                        dst[(di * f + dj) * d..(di * f + dj + 1) * d].copy_from_slice(src);
                    }
                }
            }
        }
        let mut h = self.mm1.forward(&u, oh * ow);
        h.par_iter_mut().for_each(|v| *v = gelu_erf(*v));
        self.mm2.forward(&h, oh * ow)
    }

    /// Every tile of every picture through the tower and projector, in order: the rows that
    /// replace the prompt's `<image>` tokens, `[tokens, out_width]`.
    pub fn features(&self, pictures: &[Processed]) -> Vec<f32> {
        let mut out = Vec::new();
        for (values, grid) in pictures.iter().flat_map(|p| &p.tiles) {
            let x = self.encode(values, *grid);
            out.extend(self.project(&x, *grid));
        }
        out
    }

    /// The position grid bilinearly resized (antialiased, as torch's
    /// `interpolate(.., "bilinear", antialias=True, align_corners=False)`) to `ph × pw`.
    fn resized_positions(&self, ph: usize, pw: usize) -> Vec<f32> {
        let (s, d) = (self.side, self.width);
        let wy = aa_weights(s, ph);
        let wx = aa_weights(s, pw);
        // Columns first: tmp[a][x] = Σ_b wx[x][b] · grid[a][b].
        let mut tmp = vec![0f32; s * pw * d];
        for a in 0..s {
            for (x, (start, w)) in wx.iter().enumerate() {
                let dst = &mut tmp[(a * pw + x) * d..(a * pw + x + 1) * d];
                for (b, &w) in w.iter().enumerate() {
                    let src = &self.positions[(a * s + start + b) * d..(a * s + start + b + 1) * d];
                    dst.iter_mut().zip(src).for_each(|(o, v)| *o += w * v);
                }
            }
        }
        let mut out = vec![0f32; ph * pw * d];
        for (y, (start, w)) in wy.iter().enumerate() {
            let dst = &mut out[y * pw * d..(y + 1) * pw * d];
            for (a, &w) in w.iter().enumerate() {
                let src = &tmp[(start + a) * pw * d..(start + a + 1) * pw * d];
                dst.iter_mut().zip(src).for_each(|(o, v)| *o += w * v);
            }
        }
        out
    }

}

/// Bidirectional multi-head attention over all `n` rows: per head
/// `softmax(Q Kᵀ / √dh) V`, heads in parallel.
pub(crate) fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize, heads: usize) -> Vec<f32> {
    let dh = d / heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let mut s = vec![0f32; n * n];
            // SAFETY: q and k are n × d row-major; head h's columns are h·dh..(h+1)·dh, so
            // Q_h has row stride d and K_hᵀ (dh × n) has element (kk, j) at j·d + kk. s is n × n.
            unsafe {
                gemm::gemm(
                    n, n, dh,
                    s.as_mut_ptr(), 1, n as isize, false,
                    q.as_ptr().add(h * dh), 1, d as isize,
                    k.as_ptr().add(h * dh), d as isize, 1,
                    0.0, scale, false, false, false,
                    gemm::Parallelism::None,
                );
            }
            for row in s.chunks_exact_mut(n) {
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
            // SAFETY: s is n × n row-major; V_h (n × dh) has element (j, c) at j·d + c; o is
            // n × dh row-major.
            unsafe {
                gemm::gemm(
                    n, dh, n,
                    o.as_mut_ptr(), 1, dh as isize, false,
                    s.as_ptr(), 1, n as isize,
                    v.as_ptr().add(h * dh), 1, d as isize,
                    0.0, 1.0, false, false, false,
                    gemm::Parallelism::None,
                );
            }
            o
        })
        .collect();
    let mut out = vec![0f32; n * d];
    out.par_chunks_mut(d).enumerate().for_each(|(r, row)| {
        for (h, o) in per_head.iter().enumerate() {
            row[h * dh..(h + 1) * dh].copy_from_slice(&o[r * dh..(r + 1) * dh]);
        }
    });
    out
}

pub(crate) fn linear_in(l: &Linear) -> usize {
    match &l.weight {
        Weight::Q8(w) => w.in_features(),
        Weight::F16 { in_features, .. } | Weight::F32 { in_features, .. } => *in_features,
    }
}

/// `v.patch_embd`: the conv `[width, 3, patch, patch]` as a linear layer over the processor's
/// `[y, x, channel]` patch values.
fn patch_embedding(g: &Gguf, width: usize, patch: usize) -> Result<Linear> {
    let info: &TensorInfo = g.tensor("v.patch_embd.weight")?;
    ensure!(info.shape() == [width, 3, patch, patch], "patch embedding {:?}", info.shape());
    let conv = g.to_f32(info)?;
    let mut values = vec![0f32; width * PATCH_VALUES];
    for o in 0..width {
        for c in 0..3 {
            for y in 0..patch {
                for x in 0..patch {
                    values[o * PATCH_VALUES + (y * patch + x) * 3 + c] =
                        conv[((o * 3 + c) * patch + y) * patch + x];
                }
            }
        }
    }
    let bias = g.to_f32(g.tensor("v.patch_embd.bias")?)?;
    ensure!(bias.len() == width, "patch embedding bias");
    Ok(Linear {
        weight: Weight::F32 { values, in_features: PATCH_VALUES, out_features: width },
        bias,
        exact: false,
    })
}

pub(crate) fn add(x: &mut [f32], y: &[f32]) {
    x.par_iter_mut().zip(y).for_each(|(a, b)| *a += b);
}

/// torch's antialiased bilinear weights for one axis, `input` to `output` samples: per output,
/// the first input it reads and the normalised weights.
fn aa_weights(input: usize, output: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = input as f64 / output as f64;
    let (support, invscale) = if scale >= 1.0 { (scale, 1.0 / scale) } else { (1.0, 1.0) };
    (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let w: Vec<f64> = (xmin..xmax)
                .map(|j| (1.0 - ((j as f64 - center + 0.5) * invscale).abs()).max(0.0))
                .collect();
            let total: f64 = w.iter().sum();
            let w = w.iter().map(|v| if total > 0.0 { (v / total) as f32 } else { 0.0 }).collect();
            (xmin, w)
        })
        .collect()
}

/// `gelu_pytorch_tanh`.
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    let u = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    x / (1.0 + (-2.0 * u).exp())
}

/// The exact GELU, `x · Φ(x)`.
pub(crate) fn gelu_erf(x: f32) -> f32 {
    let x64 = f64::from(x);
    (0.5 * x64 * erfc(-x64 / std::f64::consts::SQRT_2)) as f32
}

/// `erfc` to about 1.2e-7 (Numerical Recipes' Chebyshev fit).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let r = t * poly.exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_and_gelu() {
        // erfc(0.5) = 0.4795001221869535, erfc(-1) = 1.8427007929497148.
        assert!((erfc(0.5) - 0.479_500_122_186_953_5).abs() < 2e-7);
        assert!((erfc(-1.0) - 1.842_700_792_949_714_8).abs() < 2e-7);
        // torch: gelu(1.0) = 0.8413447, gelu(-2.0) = -0.04550026, gelu_tanh(1.0) = 0.8411920.
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_erf(-2.0) + 0.045_500_26).abs() < 1e-6);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-6);
    }

    #[test]
    fn aa_weights_identity_and_halving() {
        for (i, (start, w)) in aa_weights(4, 4).iter().enumerate() {
            assert_eq!((*start, w[0]), (i, 1.0), "{w:?}");
        }
        let half = aa_weights(4, 2);
        assert_eq!(half[0].0, 0);
        let s: f32 = half[1].1.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }
}
