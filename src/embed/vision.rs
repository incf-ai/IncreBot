//! EmbeddingGemma 2's picture path: Gemma 4's image processor (an
//! aspect-preserving resize to at most 2520 patches of 16 × 16, both sides
//! multiples of 48), its vision tower (patch embedding plus x and y position
//! tables, 16 bidirectional layers with axial rotary embeddings, 3 × 3 average
//! pooling) and `embed_vision` into the text tower's width: one soft token per
//! 3 × 3 patches, at most 280.
//!
//! The resize is torchvision's antialiased bicubic on uint8, bit for bit
//! (`resample::Flavour::Torch`). The processor pads to 2520 patches; the padded
//! keys are masked out and pooled away, so the tower runs on the real ones only.

use std::sync::Arc;

use anyhow::Result;
use rayon::prelude::*;

use super::ops::{Linear, Mlp, RmsNorm, add, attention, inv_freq, rope};
use super::safetensors::{SafeTensors, bf16};
use super::text::HIDDEN as TEXT_HIDDEN;
use crate::classify::vision::Rgb;
use crate::classify::vision::resample::{Flavour, resize};

const WIDTH: usize = 768;
const LAYERS: usize = 16;
const HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const INTERMEDIATE: usize = 3072;
const POSITIONS: usize = 10_240;
const PATCH: usize = 16;
/// Values per patch: 16 · 16 pixels · 3 channels.
pub const PATCH_VALUES: usize = PATCH * PATCH * 3;
const POOL: usize = 3;
/// Soft tokens per picture at most (`max_soft_tokens`).
pub const MAX_SOFT_TOKENS: usize = 280;
const MAX_PATCHES: usize = MAX_SOFT_TOKENS * POOL * POOL;
/// `rescale_factor`.
const RESCALE: f32 = 1.0 / 255.0;
const PREFIX: &str = "vision_tower";

/// A picture as the tower takes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Patches {
    /// `[n, PATCH_VALUES]` in `0..=1`, each patch's pixels rows first, then columns, then channels.
    pub values: Vec<f32>,
    /// Each patch's `(x, y)` in the patch grid, rows first.
    pub positions: Vec<(usize, usize)>,
}

impl Patches {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// `get_aspect_ratio_preserving_size`: the processor's target `(height, width)` for a picture.
pub fn target_size(height: usize, width: usize) -> (usize, usize) {
    let target_px = (MAX_PATCHES * PATCH * PATCH) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let side = POOL * PATCH;
    let floor = |v: f64| (v / side as f64).floor() as usize * side;
    let (h, w) = (floor(factor * height as f64), floor(factor * width as f64));
    let max_side = MAX_SOFT_TOKENS * side;
    // Python's `int(math.floor(a / b))` on the original sides.
    let ratio = |a: usize, b: usize| (a as f64 / b as f64).floor() as usize;
    match (h, w) {
        (0, _) => (side, (ratio(width, height) * side).min(max_side)),
        (_, 0) => ((ratio(height, width) * side).min(max_side), side),
        _ => (h, w),
    }
}

/// The image processor: resize, rescale and patchify.
pub fn preprocess(picture: &Rgb) -> Patches {
    let (h, w) = target_size(picture.height(), picture.width());
    let resized;
    let picture = if (h, w) == (picture.height(), picture.width()) {
        picture
    } else {
        resized = resize(picture, w, h, Flavour::Torch);
        &resized
    };
    let (ph, pw) = (h / PATCH, w / PATCH);
    let data = picture.data();
    let mut values = vec![0f32; ph * pw * PATCH_VALUES];
    values.par_chunks_mut(PATCH_VALUES).enumerate().for_each(|(p, dst)| {
        let (py, px) = (p / pw, p % pw);
        for y in 0..PATCH {
            let start = ((py * PATCH + y) * w + px * PATCH) * 3;
            let src = &data[start..start + PATCH * 3];
            for (d, &s) in dst[y * PATCH * 3..(y + 1) * PATCH * 3].iter_mut().zip(src) {
                *d = f32::from(s) * RESCALE;
            }
        }
    });
    let positions = (0..ph).flat_map(|y| (0..pw).map(move |x| (x, y))).collect();
    Patches { values, positions }
}

struct Layer {
    input_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    pre_ff_norm: RmsNorm,
    mlp: Mlp,
    post_ff_norm: RmsNorm,
}

pub struct Vision {
    input_proj: Linear,
    /// `position_embedding_table`, `[2, POSITIONS, WIDTH]` bf16: x positions, then y.
    positions: Vec<u16>,
    layers: Vec<Layer>,
    freq: Vec<f32>,
    pre_projection_norm: RmsNorm,
    projection: Linear,
}

impl Vision {
    pub fn load(st: &Arc<SafeTensors>) -> Result<Self> {
        let layers = (0..LAYERS)
            .map(|i| {
                let p = format!("{PREFIX}.encoder.layers.{i}");
                let lin = |name: &str, out, inp| Linear::load(st, &format!("{p}.{name}.linear"), out, inp);
                let norm = |name: &str, dim| RmsNorm::load(st, &format!("{p}.{name}"), dim);
                Ok(Layer {
                    input_norm: norm("input_layernorm", WIDTH)?,
                    q: lin("self_attn.q_proj", WIDTH, WIDTH)?,
                    k: lin("self_attn.k_proj", WIDTH, WIDTH)?,
                    v: lin("self_attn.v_proj", WIDTH, WIDTH)?,
                    o: lin("self_attn.o_proj", WIDTH, WIDTH)?,
                    q_norm: norm("self_attn.q_norm", HEAD_DIM)?,
                    k_norm: norm("self_attn.k_norm", HEAD_DIM)?,
                    v_norm: RmsNorm::plain(HEAD_DIM),
                    post_attn_norm: norm("post_attention_layernorm", WIDTH)?,
                    pre_ff_norm: norm("pre_feedforward_layernorm", WIDTH)?,
                    mlp: Mlp {
                        gate: lin("mlp.gate_proj", INTERMEDIATE, WIDTH)?,
                        up: lin("mlp.up_proj", INTERMEDIATE, WIDTH)?,
                        down: lin("mlp.down_proj", WIDTH, INTERMEDIATE)?,
                    },
                    post_ff_norm: norm("post_feedforward_layernorm", WIDTH)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            input_proj: Linear::load(st, &format!("{PREFIX}.patch_embedder.input_proj"), WIDTH, PATCH_VALUES)?,
            positions: st.bf16(
                &format!("{PREFIX}.patch_embedder.position_embedding_table"),
                &[2, POSITIONS, WIDTH],
            )?,
            layers,
            // Each axis rotates a 32-channel slice of every head.
            freq: inv_freq(100.0, HEAD_DIM / 2),
            pre_projection_norm: RmsNorm::plain(WIDTH),
            projection: Linear::load(st, "embed_vision.embedding_projection", TEXT_HIDDEN, WIDTH)?,
        })
    }

    /// The soft tokens for one picture, `[patches / 9, text HIDDEN]`, pooled rows first.
    pub fn soft_tokens(&self, p: &Patches) -> Vec<f32> {
        let n = p.len();
        assert_eq!(p.values.len(), n * PATCH_VALUES, "patch values");
        let scaled: Vec<f32> = p.values.par_iter().map(|v| 2.0 * (v - 0.5)).collect();
        let mut x = self.input_proj.forward(&scaled, n);
        x.par_chunks_mut(WIDTH).zip(&p.positions).for_each(|(row, &(px, py))| {
            let tx = &self.positions[px * WIDTH..(px + 1) * WIDTH];
            let ty = &self.positions[(POSITIONS + py) * WIDTH..(POSITIONS + py + 1) * WIDTH];
            for ((v, &a), &b) in row.iter_mut().zip(tx).zip(ty) {
                *v += bf16(a) + bf16(b);
            }
        });
        for l in &self.layers {
            self.layer(l, &mut x, &p.positions);
        }
        let mut pooled = self.pool(&x, &p.positions);
        self.pre_projection_norm.apply(&mut pooled);
        self.projection.forward(&pooled, pooled.len() / WIDTH)
    }

    fn layer(&self, l: &Layer, x: &mut [f32], positions: &[(usize, usize)]) {
        let n = positions.len();
        let h = l.input_norm.forward(x);
        let (mut q, mut k, mut v) = (l.q.forward(&h, n), l.k.forward(&h, n), l.v.forward(&h, n));
        l.q_norm.apply(&mut q);
        l.k_norm.apply(&mut k);
        l.v_norm.apply(&mut v);
        let angles = |r: usize| {
            let (px, py) = positions[r];
            let along = |p: usize| self.freq.iter().map(move |f| p as f32 * f);
            along(px).chain(along(py)).collect()
        };
        rope(&mut q, WIDTH, HEAD_DIM, 2, angles);
        rope(&mut k, WIDTH, HEAD_DIM, 2, angles);
        let a = attention(&q, &k, &v, n, (HEADS, HEADS, HEAD_DIM), None);
        let mut o = l.o.forward(&a, n);
        l.post_attn_norm.apply(&mut o);
        add(x, &o);

        let h = l.pre_ff_norm.forward(x);
        let mut m = l.mlp.forward(&h, n);
        l.post_ff_norm.apply(&mut m);
        add(x, &m);
    }

    /// The mean of each 3 × 3 block of patches, rows first, times √WIDTH.
    fn pool(&self, x: &[f32], positions: &[(usize, usize)]) -> Vec<f32> {
        let cols = positions.iter().map(|p| p.0 + 1).max().unwrap_or(0) / POOL;
        let mut out = vec![0f32; positions.len() / (POOL * POOL) * WIDTH];
        let ninth = 1.0 / (POOL * POOL) as f32;
        for (row, &(px, py)) in x.chunks_exact(WIDTH).zip(positions) {
            let t = px / POOL + cols * (py / POOL);
            for (o, &v) in out[t * WIDTH..(t + 1) * WIDTH].iter_mut().zip(row) {
                *o += v * ninth;
            }
        }
        let root = (WIDTH as f64).sqrt() as f32;
        out.par_iter_mut().for_each(|v| *v *= root);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_sizes() {
        // The fixtures' pictures: patch counts 2394, 2250, 2340, 2520 and 2304.
        assert_eq!(target_size(480, 640), (672, 912));
        assert_eq!(target_size(777, 333), (1200, 480));
        assert_eq!(target_size(13, 20), (624, 960));
        assert_eq!(target_size(30, 10_000), (48, 13_440));
        assert_eq!(target_size(1, 1), (768, 768));
    }
}
