//! Qwen3-VL's vision tower on the CPU, loaded from llama.cpp's `mmproj` GGUF:
//! the picture resized to whole 32 × 32 blocks (`smart_resize`), 16 × 16
//! patches embedded, the 48 × 48 position grid bilinearly interpolated to the
//! patch grid, 27 pre-norm blocks with 2D rotary attention over every patch,
//! then the merger (2 × 2 patches into one language-model row). Three blocks
//! also feed DeepStack mergers, whose rows the language model adds after its
//! first layers.
//!
//! Patches run in merge-block order throughout (each 2 × 2 block's four
//! patches together, blocks row by row), as the Hugging Face processor lays
//! them out, so the mergers take four consecutive rows.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

use crate::classify::vision::resample::{Flavour, resize};
use crate::classify::vision::siglip2::{LayerNorm, Linear, add, attention, gelu_erf, gelu_tanh, linear_in};
use crate::classify::vision::{PATCH_VALUES, Rgb};
use crate::gguf::Gguf;

pub const PATCH: usize = 16;
/// Patches per merged row, along each side.
pub const MERGE: usize = 2;
/// Pixels per merged row, along each side.
pub const FACTOR: usize = PATCH * MERGE;
/// The processor's `shortest_edge`: smaller pictures are scaled up to this many pixels.
pub const MIN_PIXELS: usize = 65_536;
/// The vision rotary embedding's base.
const ROPE_THETA: f32 = 10_000.0;

/// Python's `round` (half to even).
fn round_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - v.signum() } else { r }
}

/// The processor's `smart_resize`: `(height, width)` in whole [`FACTOR`]
/// blocks, the aspect ratio kept, between [`MIN_PIXELS`] and
/// `max_tokens` merged rows.
pub fn smart_resize(height: usize, width: usize, max_tokens: usize) -> Result<(usize, usize)> {
    ensure!(height > 0 && width > 0, "an empty picture");
    let ratio = height.max(width) as f64 / height.min(width) as f64;
    ensure!(ratio <= 200.0, "the picture is too narrow ({width}×{height})");
    let f = FACTOR as f64;
    let max_pixels = (max_tokens * FACTOR * FACTOR) as f64;
    let (h, w) = (height as f64, width as f64);
    let (mut hb, mut wb) = (f.max(round_even(h / f) * f), f.max(round_even(w / f) * f));
    if hb * wb > max_pixels {
        let beta = (h * w / max_pixels).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// A picture ready for the tower.
pub struct Patches {
    /// `[patches, PATCH_VALUES]`, each patch `[y, x, channel]`, scaled to −1..1.
    pub values: Vec<f32>,
    /// Patches down and across.
    pub grid: (usize, usize),
}

impl Patches {
    /// Merged rows down and across: the language model's picture grid.
    pub fn merged(&self) -> (usize, usize) {
        (self.grid.0 / MERGE, self.grid.1 / MERGE)
    }

    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1 / (MERGE * MERGE)
    }
}

/// The patch at merge-order index `i` of a grid `gw` patches across: its row and column.
fn patch_at(i: usize, gw: usize) -> (usize, usize) {
    let (block, within) = (i / (MERGE * MERGE), i % (MERGE * MERGE));
    let (br, bc) = (block / (gw / MERGE), block % (gw / MERGE));
    (br * MERGE + within / MERGE, bc * MERGE + within % MERGE)
}

/// `picture` resized (bicubic, as the processor) and cut into patches.
pub fn preprocess(picture: &Rgb, max_tokens: usize) -> Result<Patches> {
    let (h, w) = smart_resize(picture.height(), picture.width(), max_tokens)?;
    let img = resize(picture, w, h, Flavour::Torch);
    let (gh, gw) = (h / PATCH, w / PATCH);
    let data = img.data();
    let mut values = vec![0f32; gh * gw * PATCH_VALUES];
    values.par_chunks_mut(PATCH_VALUES).enumerate().for_each(|(i, out)| {
        let (r, c) = patch_at(i, gw);
        for y in 0..PATCH {
            let row = &data[((r * PATCH + y) * w + c * PATCH) * 3..][..PATCH * 3];
            for (o, &v) in out[y * PATCH * 3..][..PATCH * 3].iter_mut().zip(row) {
                *o = (f32::from(v) / 255.0 - 0.5) / 0.5;
            }
        }
    });
    Ok(Patches { values, grid: (gh, gw) })
}

pub(crate) struct Layer {
    pub(crate) ln1: LayerNorm,
    pub(crate) qkv: Linear,
    pub(crate) out: Linear,
    pub(crate) ln2: LayerNorm,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

/// Four patches into one language-model row: a layer norm (its width says
/// whether over each patch before the merge or the merged row after), then a GELU MLP.
pub(crate) struct Merger {
    pub(crate) norm: LayerNorm,
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
}

impl Merger {
    fn forward(&self, x: &[f32], width: usize, eps: f32) -> Vec<f32> {
        let rows = x.len() / (width * MERGE * MERGE);
        let mut h = self.fc1.forward(&self.norm.forward(x, eps), rows);
        h.par_iter_mut().for_each(|v| *v = gelu_erf(*v));
        self.fc2.forward(&h, rows)
    }
}

/// The tower's output for one picture.
pub struct Features {
    /// `[tokens, out_width]`: the rows in place of the picture's tokens.
    pub main: Vec<f32>,
    /// One `[tokens, out_width]` per DeepStack level, added after the language
    /// model's first layers.
    pub deepstack: Vec<Vec<f32>>,
    /// Merged rows down and across.
    pub grid: (usize, usize),
}

pub struct Vision {
    pub(crate) width: usize,
    pub(crate) heads: usize,
    pub(crate) eps: f32,
    /// `[width, PATCH_VALUES]`: both temporal kernels summed (a picture is two equal frames).
    pub(crate) patch_embd: Linear,
    /// The `side × side` position grid, `[side · side, width]`.
    positions: Vec<f32>,
    side: usize,
    pub(crate) layers: Vec<Layer>,
    /// After which block each DeepStack merger reads.
    pub(crate) deepstack: Vec<(usize, Merger)>,
    pub(crate) merger: Merger,
}

impl Vision {
    /// Loads an `mmproj` GGUF with the `qwen3vl_merger` projector.
    pub fn load(path: &Path) -> Result<Self> {
        let g = Gguf::open(path).with_context(|| format!("opening {}", path.display()))?;
        let projector = g.get_str("clip.projector_type")?;
        ensure!(projector == "qwen3vl_merger", "projector {projector:?}, not qwen3vl_merger");
        let width = g.get_u64("clip.vision.embedding_length")? as usize;
        let heads = g.get_u64("clip.vision.attention.head_count")? as usize;
        let blocks = g.get_u64("clip.vision.block_count")? as usize;
        let eps = g.get_f32("clip.vision.attention.layer_norm_epsilon")?;
        ensure!(g.get_u64("clip.vision.patch_size")? as usize == PATCH, "patch size");
        ensure!(g.get_u64("clip.vision.spatial_merge_size")? as usize == MERGE, "merge size");
        ensure!(width.is_multiple_of(heads) && (width / heads).is_multiple_of(4), "{width} wide, {heads} heads");

        // The conv `[width, 3, 16, 16]` per frame, as a linear layer over `[y, x, channel]`.
        let k0 = g.tensor("v.patch_embd.weight")?;
        ensure!(k0.shape() == [width, 3, PATCH, PATCH], "patch embedding {:?}", k0.shape());
        let (k0, k1) = (g.to_f32(k0)?, g.to_f32(g.tensor("v.patch_embd.weight.1")?)?);
        ensure!(k0.len() == k1.len(), "temporal patch kernels differ in size");
        let mut values = vec![0f32; width * PATCH_VALUES];
        for o in 0..width {
            for c in 0..3 {
                for y in 0..PATCH {
                    for x in 0..PATCH {
                        let src = ((o * 3 + c) * PATCH + y) * PATCH + x;
                        values[o * PATCH_VALUES + (y * PATCH + x) * 3 + c] = k0[src] + k1[src];
                    }
                }
            }
        }
        let patch_embd = Linear::from_f32(values, PATCH_VALUES, g.to_f32(g.tensor("v.patch_embd.bias")?)?);
        ensure!(patch_embd.out_features() == width, "patch embedding bias");

        let pos = g.tensor("v.position_embd.weight")?;
        let positions = g.to_f32(pos)?;
        let side = (positions.len() / width).isqrt();
        ensure!(side * side * width == positions.len(), "position grid {:?}", pos.shape());

        let layers = (0..blocks)
            .map(|i| {
                let p = format!("v.blk.{i}");
                Ok(Layer {
                    ln1: LayerNorm::load(&g, &format!("{p}.ln1"))?,
                    qkv: Linear::load(&g, &format!("{p}.attn_qkv"))?,
                    out: Linear::load(&g, &format!("{p}.attn_out"))?,
                    ln2: LayerNorm::load(&g, &format!("{p}.ln2"))?,
                    up: Linear::load(&g, &format!("{p}.ffn_up"))?,
                    down: Linear::load(&g, &format!("{p}.ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(layers.iter().all(|l| l.qkv.out_features() == 3 * width), "fused qkv width");

        let flags = g.get("clip.vision.is_deepstack_layers")?.as_array().context("is_deepstack_layers")?;
        let mut deepstack = Vec::new();
        for (i, f) in flags.iter().enumerate() {
            if matches!(f, crate::gguf::Value::Bool(true)) {
                let p = format!("v.deepstack.{i}");
                deepstack.push((
                    i,
                    Merger {
                        norm: LayerNorm::load(&g, &format!("{p}.norm"))?,
                        fc1: Linear::load(&g, &format!("{p}.fc1"))?,
                        fc2: Linear::load(&g, &format!("{p}.fc2"))?,
                    },
                ));
            }
        }
        let merger = Merger {
            norm: LayerNorm::load(&g, "v.post_ln")?,
            fc1: Linear::load(&g, "mm.0")?,
            fc2: Linear::load(&g, "mm.2")?,
        };
        ensure!(linear_in(&merger.fc1) == width * MERGE * MERGE, "merger takes {}", linear_in(&merger.fc1));
        let out = merger.fc2.out_features();
        for (i, m) in &deepstack {
            ensure!(m.fc2.out_features() == out && *i < blocks, "DeepStack merger {i}");
        }
        Ok(Self { width, heads, eps, patch_embd, positions, side, layers, deepstack, merger })
    }

    /// The language model width the mergers write.
    pub fn out_width(&self) -> usize {
        self.merger.fc2.out_features()
    }

    pub fn deepstack_levels(&self) -> usize {
        self.deepstack.len()
    }

    /// The tower and mergers on one picture.
    pub fn features(&self, p: &Patches) -> Result<Features> {
        let (gh, gw) = p.grid;
        ensure!(gh.is_multiple_of(MERGE) && gw.is_multiple_of(MERGE), "patch grid {gh}×{gw}");
        let n = gh * gw;
        ensure!(p.values.len() == n * PATCH_VALUES, "patch values");
        let d = self.width;
        let mut x = self.patch_embd.forward(&p.values, n);
        add(&mut x, &self.positions(gh, gw));
        let (cos, sin) = rope(n, gw, d / self.heads);
        let mut deepstack = Vec::with_capacity(self.deepstack.len());
        for (i, l) in self.layers.iter().enumerate() {
            let h = l.ln1.forward(&x, self.eps);
            let qkv = l.qkv.forward(&h, n);
            let (mut q, mut k, mut v) = (vec![0f32; n * d], vec![0f32; n * d], vec![0f32; n * d]);
            q.par_chunks_mut(d)
                .zip(k.par_chunks_mut(d))
                .zip(v.par_chunks_mut(d))
                .zip(qkv.par_chunks(3 * d))
                .for_each(|(((q, k), v), src)| {
                    q.copy_from_slice(&src[..d]);
                    k.copy_from_slice(&src[d..2 * d]);
                    v.copy_from_slice(&src[2 * d..]);
                });
            apply_rope(&mut q, &cos, &sin, d / self.heads);
            apply_rope(&mut k, &cos, &sin, d / self.heads);
            let a = attention(&q, &k, &v, n, d, self.heads);
            add(&mut x, &l.out.forward(&a, n));
            let h = l.ln2.forward(&x, self.eps);
            let mut u = l.up.forward(&h, n);
            u.par_iter_mut().for_each(|v| *v = gelu_tanh(*v));
            add(&mut x, &l.down.forward(&u, n));
            for (_, m) in self.deepstack.iter().filter(|(at, _)| *at == i) {
                deepstack.push(m.forward(&x, d, self.eps));
            }
        }
        let main = self.merger.forward(&x, d, self.eps);
        Ok(Features { main, deepstack, grid: p.merged() })
    }

    /// The position grid bilinearly interpolated (corners aligned) to `gh × gw`,
    /// in merge order.
    fn positions(&self, gh: usize, gw: usize) -> Vec<f32> {
        interpolate(&self.positions, self.side, self.width, gh, gw)
    }

    /// The `side × side` position grid and its side.
    pub(crate) fn position_grid(&self) -> (&[f32], usize) {
        (&self.positions, self.side)
    }
}

/// The `s × s` grid `positions` (rows of `d`) bilinearly interpolated (corners
/// aligned) to `gh × gw`, in merge order.
pub(crate) fn interpolate(positions: &[f32], s: usize, d: usize, gh: usize, gw: usize) -> Vec<f32> {
    // torch.linspace(0, s − 1, n): each index's floor, ceil and weight.
    let axis = |n: usize| -> Vec<(usize, usize, f32)> {
        (0..n)
            .map(|i| {
                let v = if n == 1 { 0.0 } else { (s - 1) as f32 * i as f32 / (n - 1) as f32 };
                let lo = (v as usize).min(s - 1);
                (lo, (lo + 1).min(s - 1), v - lo as f32)
            })
            .collect()
    };
    let (ys, xs) = (axis(gh), axis(gw));
    let mut out = vec![0f32; gh * gw * d];
    out.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
        let (r, c) = patch_at(i, gw);
        let ((y0, y1, dy), (x0, x1, dx)) = (ys[r], xs[c]);
        for (idx, w) in [
            (y0 * s + x0, (1.0 - dy) * (1.0 - dx)),
            (y0 * s + x1, (1.0 - dy) * dx),
            (y1 * s + x0, dy * (1.0 - dx)),
            (y1 * s + x1, dy * dx),
        ] {
            o.iter_mut().zip(&positions[idx * d..][..d]).for_each(|(o, p)| *o += w * p);
        }
    });
    out
}

/// The 2D rotary tables, `[n, head_dim / 2]` each, in merge order: the first
/// half of the frequencies turns with the patch's row, the second with its column.
pub(crate) fn rope(n: usize, gw: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let quarter = half / 2;
    let inv: Vec<f32> = (0..quarter).map(|j| 1.0 / ROPE_THETA.powf((2 * j) as f32 / half as f32)).collect();
    let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
    for i in 0..n {
        let (r, c) = patch_at(i, gw);
        for j in 0..half {
            let angle = if j < quarter { r as f32 * inv[j] } else { c as f32 * inv[j - quarter] };
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

/// NeoX-style rotation of every head of `x` (`[n, heads · head_dim]`): pair
/// `(j, j + head_dim / 2)` by row `j` of the tables.
fn apply_rope(x: &mut [f32], cos: &[f32], sin: &[f32], head_dim: usize) {
    let half = head_dim / 2;
    let width = x.len() / (cos.len() / half);
    x.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
        let (cos, sin) = (&cos[i * half..][..half], &sin[i * half..][..half]);
        for head in row.chunks_exact_mut(head_dim) {
            let (a, b) = head.split_at_mut(half);
            for j in 0..half {
                let (x0, x1) = (a[j], b[j]);
                a[j] = x0 * cos[j] - x1 * sin[j];
                b[j] = x1 * cos[j] + x0 * sin[j];
            }
        }
    });
}

/// Checks a picture's merged grid against the model's limit before running it.
pub fn check_tokens(p: &Patches, max_tokens: usize) -> Result<()> {
    if p.tokens() > max_tokens {
        bail!("{} picture tokens, more than {max_tokens}", p.tokens());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_like_the_processor() {
        // 1024 × 768 at 1024 tokens: 32 × 24 blocks fit as they are.
        assert_eq!(smart_resize(768, 1024, 1024).unwrap(), (768, 1024));
        // A large photo is scaled down to at most 1024 blocks.
        let (h, w) = smart_resize(3000, 4000, 1024).unwrap();
        assert!(h * w <= 1024 * FACTOR * FACTOR && h.is_multiple_of(FACTOR) && w.is_multiple_of(FACTOR));
        assert_eq!((h, w), (864, 1152));
        // A small one is scaled up to MIN_PIXELS.
        let (h, w) = smart_resize(100, 100, 1024).unwrap();
        assert!(h * w >= MIN_PIXELS);
        assert_eq!((h, w), (256, 256));
        // Rounding half to even, as Python: 48 / 32 = 1.5 → 2 blocks, 80 / 32 = 2.5 → 2.
        assert_eq!(round_even(1.5), 2.0);
        assert_eq!(round_even(2.5), 2.0);
        assert!(smart_resize(10, 3000, 1024).is_err());
    }

    #[test]
    fn merge_order() {
        // 4 patches across: blocks (0,0) (0,1) then the next block row.
        let order: Vec<_> = (0..8).map(|i| patch_at(i, 4)).collect();
        assert_eq!(order, [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(patch_at(8, 4), (2, 0));
    }
}
