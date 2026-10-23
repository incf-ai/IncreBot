//! Qwen3-VL's language model (/describe-image) on one GPU, built from the
//! Llama path's kernels (`model.rs`).
//!
//! GPU only: the K-quant blocks go straight from the GGUF to VRAM, and only the
//! token embeddings stay on the host. Two things set it apart from Llama:
//! multimodal RoPE, whose (t, h, w) positions the host turns into cos/sin rows
//! for each chunk (`rope_store_neox_rows`), since past a picture they no longer
//! match the cache slots; and DeepStack, which adds the vision tower's earlier
//! features to the picture's rows after the first few layers.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use tracing::info;

use super::model::{
    CHUNK, Linear, MAX_KEYS, Mode, Quantized, SPLIT_KEYS, matmul, matvec, quantize, repack_q6k, round_up,
};
use super::vk::{Buffer, Commands, Device, Push};
use crate::gguf::{GgmlType, Gguf};
use crate::model::KQuant;
use crate::model::lfm2::Embedding;

/// Submissions in flight at once (one layer each).
const RING: usize = 2;

/// LM head rows per matvec dispatch: one workgroup per two rows, and a
/// dispatch's x count may be capped at 65535.
const HEAD_ROWS: usize = 65536;

#[derive(Clone, Debug)]
pub struct Qwen3Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// Frequencies per axis (t, h, w), interleaved: see [`mrope`].
    pub sections: [usize; 3],
    /// Layers after which a DeepStack level is added.
    pub deepstack: usize,
}

impl Qwen3Config {
    fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        ensure!(arch == "qwen3vl", "architecture {arch:?}, expected qwen3vl");
        let u = |k: &str| g.get_u64(&format!("qwen3vl.{k}")).map(|v| v as usize);
        let sections: Vec<usize> = g
            .get("qwen3vl.rope.dimension_sections")?
            .as_array()
            .context("rope.dimension_sections is not an array")?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize).context("a rope section is not an integer"))
            .collect::<Result<_>>()?;
        ensure!(sections.len() >= 3, "rope.dimension_sections {sections:?}");
        let vocab = g.tensor("token_embd.weight")?.shape()[0];
        let c = Self {
            n_layers: u("block_count")?,
            d_model: u("embedding_length")?,
            ffn_dim: u("feed_forward_length")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            vocab,
            rope_theta: g.get_f32("qwen3vl.rope.freq_base")?,
            rms_eps: g.get_f32("qwen3vl.attention.layer_norm_rms_epsilon")?,
            sections: [sections[0], sections[1], sections[2]],
            deepstack: u("n_deepstack_layers")?,
        };
        ensure!(c.head_dim == 128, "head_dim {} unsupported on GPU", c.head_dim);
        ensure!(c.sections.iter().sum::<usize>() == c.head_dim / 2, "rope sections {:?}", c.sections);
        ensure!(
            c.d_model.is_multiple_of(256) && c.ffn_dim.is_multiple_of(256),
            "d_model {} / ffn_dim {} unsupported on GPU",
            c.d_model,
            c.ffn_dim
        );
        ensure!(c.n_heads * c.head_dim == c.d_model, "n_heads · head_dim ≠ d_model");
        ensure!(c.deepstack <= c.n_layers, "{} DeepStack layers", c.deepstack);
        Ok(c)
    }
}

/// A position on each of the time, height and width axes. Text has the same
/// value on all three.
pub type Pos = [u32; 3];

/// The cos and sin rows (`head_dim / 2` each) for `pos`: frequency `j` turns
/// with the height position when `j % 3 == 1`, the width position when
/// `j % 3 == 2` (each within its section's `3 · len`), else the time position.
pub fn mrope(c: &Qwen3Config, pos: &[Pos]) -> (Vec<f32>, Vec<f32>) {
    let half = c.head_dim / 2;
    let inv: Vec<f32> = (0..half).map(|j| 1.0 / c.rope_theta.powf((2 * j) as f32 / c.head_dim as f32)).collect();
    let axis: Vec<usize> = (0..half)
        .map(|j| match j % 3 {
            1 if j < 3 * c.sections[1] => 1,
            2 if j < 3 * c.sections[2] => 2,
            _ => 0,
        })
        .collect();
    let (mut cos, mut sin) = (Vec::with_capacity(pos.len() * half), Vec::with_capacity(pos.len() * half));
    for p in pos {
        for j in 0..half {
            let angle = f64::from(p[axis[j]] as f32 * inv[j]);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

struct Layer {
    buf: Buffer,
    attn_norm: u64,
    ffn_norm: u64,
    q_norm: u64,
    k_norm: u64,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// The KV cache and scratch buffers of the active GPU.
struct Active {
    n_ctx: usize,
    k: Vec<Buffer>,
    v: Vec<Buffer>,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    att: Buffer,
    gate: Buffer,
    up: Buffer,
    last: Buffer,
    att_part: Buffer,
    xq: Buffer,
    xd: Buffer,
    xs: Buffer,
    /// The chunk's cos rows, then its sin rows.
    rope: Buffer,
    /// The chunk's DeepStack rows per level, zero outside the picture.
    deepstack: Buffer,
    /// Host memory.
    logits: Buffer,
    cmds: Commands,
}

pub struct GpuQwen3 {
    pub dev: Arc<Device>,
    pub c: Qwen3Config,
    embed: Embedding,
    layers: Vec<Layer>,
    /// `output_norm` (f32) followed by the LM head, in parts of [`HEAD_ROWS`].
    head: Buffer,
    lm_head: Vec<Linear>,
    active: Option<Active>,
}

/// A linear layer's blocks as the GPU takes them, and its place.
fn blocks(g: &Gguf, name: &str) -> Result<(Vec<u8>, KQuant, u32, u32)> {
    let info = g.tensor(name)?;
    let (k, n) = (info.dims[0] as u32, info.dims[1] as u32);
    let bytes = g.bytes(info)?;
    Ok(match info.ty {
        GgmlType::Q4K => (bytes.to_vec(), KQuant::Q4K, k, n),
        GgmlType::Q6K => (repack_q6k(bytes), KQuant::Q6K, k, n),
        ty => anyhow::bail!("{name} is {ty:?}; the GPU takes Q4_K and Q6_K"),
    })
}

impl GpuQwen3 {
    /// Uploads the GGUF's weights to `dev`; they must be Q4_K/Q6_K (Q4_K_M is).
    pub fn load(dev: &Arc<Device>, path: &Path) -> Result<Self> {
        let start = Instant::now();
        let g = Gguf::open(path)?;
        let c = Qwen3Config::from_gguf(&g)?;
        let embed = Embedding::load(&g, c.vocab, c.d_model)?;
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let p = format!("blk.{i}");
            // f32 vectors first, then the linear layers, each 256-byte aligned.
            let mut size = 0u64;
            let mut vectors = Vec::new();
            let mut vector = |name: &str, len: usize| -> Result<u64> {
                let v = g.to_f32(g.tensor(&format!("{p}.{name}.weight"))?)?;
                ensure!(v.len() == len, "{p}.{name} has {} values, expected {len}", v.len());
                let off = size;
                size += round_up((len * 4) as u64);
                vectors.push((off, v));
                Ok(off)
            };
            let attn_norm = vector("attn_norm", c.d_model)?;
            let ffn_norm = vector("ffn_norm", c.d_model)?;
            let q_norm = vector("attn_q_norm", c.head_dim)?;
            let k_norm = vector("attn_k_norm", c.head_dim)?;
            let mut parts = Vec::new();
            for name in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                let (bytes, kind, k, n) = blocks(&g, &format!("{p}.{name}.weight"))?;
                let off = size;
                size += round_up(bytes.len() as u64);
                parts.push((bytes, Linear { off, kind, k, n }));
            }
            let buf = dev.vram(size)?;
            for (off, v) in &vectors {
                buf.write(*off as usize, v);
            }
            for (bytes, lin) in &parts {
                buf.write(lin.off as usize, bytes);
            }
            let &[wq, wk, wv, wo, gate, up, down] = &parts.iter().map(|&(_, l)| l).collect::<Vec<_>>()[..] else {
                unreachable!("seven linear layers")
            };
            let (d, kvw, ffn) = (c.d_model as u32, (c.n_kv_heads * c.head_dim) as u32, c.ffn_dim as u32);
            for (lin, k, n, name) in
                [(wq, d, d, "attn_q"), (wk, d, kvw, "attn_k"), (wv, d, kvw, "attn_v"), (wo, d, d, "attn_output")]
                    .into_iter()
                    .chain([(gate, d, ffn, "ffn_gate"), (up, d, ffn, "ffn_up"), (down, ffn, d, "ffn_down")])
            {
                ensure!(lin.k == k && lin.n == n, "{p}.{name} is {}×{}, expected {k}×{n}", lin.k, lin.n);
            }
            layers.push(Layer { buf, attn_norm, ffn_norm, q_norm, k_norm, wq, wk, wv, wo, gate, up, down });
        }

        // The LM head in parts of HEAD_ROWS rows, each in the GPU's layout.
        let norm = g.to_f32(g.tensor("output_norm.weight")?)?;
        let info = g.tensor("output.weight")?;
        ensure!(info.shape() == [c.vocab, c.d_model], "output.weight is {:?}", info.shape());
        let kind = match info.ty {
            GgmlType::Q4K => KQuant::Q4K,
            GgmlType::Q6K => KQuant::Q6K,
            ty => anyhow::bail!("output.weight is {ty:?}; the GPU takes Q4_K and Q6_K"),
        };
        let row_bytes = c.d_model / 256 * kind.block_bytes();
        let bytes = g.bytes(info)?;
        let mut size = round_up((c.d_model * 4) as u64);
        let mut parts = Vec::new();
        for rows in bytes.chunks(HEAD_ROWS * row_bytes) {
            // Counted before repacking: Q6_K's GPU layout pads each block.
            let n = rows.len() / row_bytes;
            let rows = match kind {
                KQuant::Q4K => rows.to_vec(),
                KQuant::Q6K => repack_q6k(rows),
            };
            parts.push((rows, Linear { off: size, kind, k: c.d_model as u32, n: n as u32 }));
            size += round_up(parts.last().map_or(0, |(r, _)| r.len()) as u64);
        }
        let head = dev.vram(size)?;
        head.write(0, &norm);
        for (rows, lin) in &parts {
            head.write(lin.off as usize, rows);
        }
        let lm_head = parts.into_iter().map(|(_, l)| l).collect();
        info!(
            gpu = %dev.pci,
            mib = dev.tracker.held() >> 20,
            elapsed_s = start.elapsed().as_secs_f32(),
            "qwen3-vl weights uploaded"
        );
        Ok(Self { dev: dev.clone(), c, embed, layers, head, lm_head, active: None })
    }

    /// The token embeddings, `[tokens, d_model]`.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut x = Vec::with_capacity(tokens.len() * self.c.d_model);
        for &id in tokens {
            ensure!((id as usize) < self.c.vocab, "token id {id} out of range");
            self.embed.row(id as usize, &mut x)?;
        }
        Ok(x)
    }

    /// Allocates the KV cache for `n_ctx` positions and the scratch buffers.
    pub fn activate(&mut self, n_ctx: usize) -> Result<()> {
        if self.active.as_ref().is_some_and(|a| a.n_ctx >= n_ctx) {
            return Ok(());
        }
        self.active = None;
        let c = &self.c;
        ensure!(n_ctx <= MAX_KEYS, "n_ctx {n_ctx} too large for the GPU backend");
        let kvw = (c.n_kv_heads * c.head_dim) as u64;
        let (d, ffn, ch) = (c.d_model as u64, c.ffn_dim as u64, CHUNK as u64);
        let dev = &self.dev;
        let mut k = Vec::with_capacity(c.n_layers);
        let mut v = Vec::with_capacity(c.n_layers);
        for _ in 0..c.n_layers {
            k.push(dev.vram(n_ctx as u64 * kvw * 2)?);
            v.push(dev.vram(n_ctx as u64 * kvw * 2)?);
        }
        self.active = Some(Active {
            n_ctx,
            k,
            v,
            x: dev.vram(ch * d * 4)?,
            h: dev.vram(ch * d * 4)?,
            q: dev.vram(ch * d * 4)?,
            k_new: dev.vram(ch * kvw * 4)?,
            v_new: dev.vram(ch * kvw * 4)?,
            att: dev.vram(ch * d * 4)?,
            gate: dev.vram(ch * ffn * 4)?,
            up: dev.vram(ch * ffn * 4)?,
            last: dev.vram(d * 4)?,
            att_part: dev.vram((c.n_heads * n_ctx.div_ceil(SPLIT_KEYS) * (c.head_dim + 2) * 4) as u64)?,
            xq: dev.vram(ch * d.max(ffn))?,
            xd: dev.vram(ch * d.max(ffn) / 256 * 4)?,
            xs: dev.vram(ch * d.max(ffn) / 16 * 4)?,
            rope: dev.vram(ch * c.head_dim as u64 * 4)?,
            deepstack: dev.vram((c.deepstack.max(1) as u64) * ch * d * 4)?,
            logits: dev.host(c.vocab as u64 * 4)?,
            cmds: Commands::new(dev, RING)?,
        });
        Ok(())
    }

    /// Frees the KV cache and scratch buffers, keeping the weights.
    pub fn deactivate(&mut self) {
        self.active = None;
    }

    /// Runs the rows `x` (`[t, d_model]`, at positions `pos`) into cache slots
    /// `past..past + t` and returns the last row's logits, or `None` when
    /// `stop` turned true first. `deepstack` is the first picture row and each
    /// level's features for the picture's rows.
    pub fn forward(
        &mut self,
        x: &[f32],
        pos: &[Pos],
        deepstack: Option<(usize, &[Vec<f32>])>,
        past: usize,
        stop: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>> {
        let c = &self.c;
        let d = c.d_model;
        let a = self.active.as_mut().context("GPU model is not active")?;
        let n = pos.len();
        ensure!(n > 0 && x.len() == n * d, "{} values for {n} rows", x.len());
        ensure!(past + n <= a.n_ctx, "context overflow on GPU: {past} + {n} > {}", a.n_ctx);
        if let Some((_, levels)) = deepstack {
            ensure!(levels.len() == c.deepstack, "{} DeepStack levels, expected {}", levels.len(), c.deepstack);
        }
        let half = c.head_dim / 2;
        let chunks = n.div_ceil(CHUNK);
        for ci in 0..chunks {
            if stop() {
                a.cmds.wait_all()?;
                return Ok(None);
            }
            let (r0, r1) = (ci * CHUNK, ((ci + 1) * CHUNK).min(n));
            let t = r1 - r0;
            // The previous chunk's layers still read these until they finish.
            a.cmds.wait_all()?;
            a.x.write(0, &x[r0 * d..r1 * d]);
            let (cos, sin) = mrope(c, &pos[r0..r1]);
            a.rope.write(0, &cos);
            a.rope.write(CHUNK * half * 4, &sin);
            // The picture rows within this chunk.
            let stack = deepstack.and_then(|(first, levels)| {
                let rows = levels.first().map_or(0, |l| l.len() / d);
                let (s, e) = (first.max(r0), (first + rows).min(r1));
                (s < e).then_some((s, e, first, levels))
            });
            if let Some((s, e, first, levels)) = stack {
                let mut rows = vec![0f32; t * d];
                for (li, level) in levels.iter().enumerate() {
                    rows[(s - r0) * d..(e - r0) * d].copy_from_slice(&level[(s - first) * d..(e - first) * d]);
                    a.deepstack.write(li * CHUNK * d * 4, &rows);
                }
            }
            let pos0 = past + r0;
            for (li, layer) in self.layers.iter().enumerate() {
                a.cmds.begin()?;
                record_layer(&self.dev, a, c, layer, li, t, pos0)?;
                if stack.is_some() && li < c.deepstack {
                    let count = (t * d) as u32;
                    a.cmds.dispatch(
                        "gpu_add",
                        self.dev.pipes.add,
                        Push::default()
                            .addr(a.deepstack.addr + (li * CHUNK * d * 4) as u64)
                            .addr(a.x.addr)
                            .u32(count),
                        [count.div_ceil(256), 1, 1],
                    )?;
                }
                a.cmds.submit()?;
            }
            if ci + 1 == chunks {
                a.cmds.begin()?;
                a.cmds.dispatch(
                    "gpu_norm",
                    self.dev.pipes.rmsnorm,
                    Push::default()
                        .addr(a.x.addr + ((t - 1) * d * 4) as u64)
                        .addr(self.head.addr)
                        .addr(a.last.addr)
                        .u32(d as u32)
                        .f32(c.rms_eps),
                    [1, 1, 1],
                )?;
                for (i, &lin) in self.lm_head.iter().enumerate() {
                    matvec(
                        &self.dev,
                        &mut a.cmds,
                        &[((self.head.addr + lin.off, lin), a.logits.addr + (i * HEAD_ROWS * 4) as u64)],
                        a.last.addr,
                        Mode::Plain,
                    )?;
                }
                a.cmds.submit()?;
            }
        }
        a.cmds.wait_all()?;
        Ok(Some(a.logits.read(0, c.vocab)))
    }
}

/// Records one decoder layer for `t` rows into cache slots `past..past + t`,
/// their RoPE rows in `a.rope`.
fn record_layer(dev: &Device, a: &mut Active, c: &Qwen3Config, l: &Layer, li: usize, t: usize, past: usize) -> Result<()> {
    let p = &dev.pipes;
    let (d, hd, t32) = (c.d_model as u32, c.head_dim as u32, t as u32);
    let (heads, kv_heads) = (c.n_heads as u32, c.n_kv_heads as u32);
    let w = |lin: Linear| (l.buf.addr + lin.off, lin);
    // RMSNorm of `rows` rows of `width`, weights at `off` in the layer buffer.
    let norm = |cmds: &mut Commands, off: u64, x: u64, y: u64, width: u32, rows: u32| -> Result<()> {
        cmds.dispatch(
            "gpu_norm",
            p.rmsnorm,
            Push::default().addr(x).addr(l.buf.addr + off).addr(y).u32(width).f32(c.rms_eps),
            [rows, 1, 1],
        )
    };
    let qx = Quantized { q: a.xq.addr, d: a.xd.addr, s: a.xs.addr };
    let cmds = &mut a.cmds;
    norm(cmds, l.attn_norm, a.x.addr, a.h.addr, d, t32)?;
    if t == 1 {
        matvec(
            dev,
            cmds,
            &[(w(l.wq), a.q.addr), (w(l.wk), a.k_new.addr), (w(l.wv), a.v_new.addr)],
            a.h.addr,
            Mode::Plain,
        )?;
    } else {
        let xq = quantize(dev, cmds, qx, a.h.addr, d, t32)?;
        matmul(dev, cmds, w(l.wq), xq, a.q.addr, t32, false)?;
        matmul(dev, cmds, w(l.wk), xq, a.k_new.addr, t32, false)?;
        matmul(dev, cmds, w(l.wv), xq, a.v_new.addr, t32, false)?;
    }
    norm(cmds, l.q_norm, a.q.addr, a.q.addr, hd, t32 * heads)?;
    norm(cmds, l.k_norm, a.k_new.addr, a.k_new.addr, hd, t32 * kv_heads)?;
    let pairs = t32 * (heads + 2 * kv_heads) * hd / 2;
    cmds.dispatch(
        "gpu_rope_store",
        p.rope_store_neox_rows,
        Push::default()
            .addr(a.q.addr)
            .addr(a.k_new.addr)
            .addr(a.v_new.addr)
            .addr(a.rope.addr)
            .addr(a.rope.addr + (CHUNK * c.head_dim / 2 * 4) as u64)
            .addr(a.k[li].addr)
            .addr(a.v[li].addr)
            .u32(t32)
            .u32(heads)
            .u32(kv_heads)
            .u32(hd)
            .u32(past as u32),
        [pairs.div_ceil(256), 1, 1],
    )?;
    let attn = Push::default()
        .addr(a.q.addr)
        .addr(a.k[li].addr)
        .addr(a.v[li].addr)
        .addr(if t == 1 { a.att_part.addr } else { a.att.addr })
        .u32(heads)
        .u32(kv_heads)
        .u32(hd)
        .u32(past as u32)
        .f32(1.0 / (c.head_dim as f32).sqrt());
    if t == 1 {
        let splits = (past + 1).div_ceil(SPLIT_KEYS) as u32;
        cmds.dispatch("gpu_attention", p.attention_split, attn.u32(splits), [heads, splits, 1])?;
        cmds.dispatch(
            "gpu_attention",
            p.attention_merge,
            Push::default().addr(a.att_part.addr).addr(a.att.addr).u32(hd).u32(splits),
            [heads, 1, 1],
        )?;
        matvec(dev, cmds, &[(w(l.wo), a.x.addr)], a.att.addr, Mode::Accumulate)?;
    } else {
        cmds.dispatch("gpu_attention", p.flash_attn, attn.u32(t32), [heads, t32.div_ceil(32), 1])?;
        let xq = quantize(dev, cmds, qx, a.att.addr, d, t32)?;
        matmul(dev, cmds, w(l.wo), xq, a.x.addr, t32, true)?;
    }
    norm(cmds, l.ffn_norm, a.x.addr, a.h.addr, d, t32)?;
    if t == 1 {
        matvec(dev, cmds, &[(w(l.gate), a.gate.addr), (w(l.up), 0)], a.h.addr, Mode::Glu)?;
        matvec(dev, cmds, &[(w(l.down), a.x.addr)], a.gate.addr, Mode::Accumulate)?;
    } else {
        let xq = quantize(dev, cmds, qx, a.h.addr, d, t32)?;
        matmul(dev, cmds, w(l.gate), xq, a.gate.addr, t32, false)?;
        matmul(dev, cmds, w(l.up), xq, a.up.addr, t32, false)?;
        let count = t32 * c.ffn_dim as u32;
        cmds.dispatch(
            "gpu_swiglu",
            p.swiglu,
            Push::default().addr(a.gate.addr).addr(a.up.addr).u32(count),
            [count.div_ceil(256), 1, 1],
        )?;
        let xq = quantize(dev, cmds, qx, a.gate.addr, c.ffn_dim as u32, t32)?;
        matmul(dev, cmds, w(l.down), xq, a.x.addr, t32, true)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Qwen3Config {
        Qwen3Config {
            n_layers: 1,
            d_model: 4096,
            ffn_dim: 12288,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            vocab: 16,
            rope_theta: 5_000_000.0,
            rms_eps: 1e-6,
            sections: [24, 20, 20],
            deepstack: 3,
        }
    }

    #[test]
    fn mrope_axes() {
        let c = config();
        // Text: plain RoPE at that position.
        let (cos, sin) = mrope(&c, &[[7, 7, 7]]);
        for j in 0..64 {
            let angle = f64::from(7.0 / 5_000_000f32.powf(j as f32 / 64.0));
            assert!((cos[j] - angle.cos() as f32).abs() < 1e-5 && (sin[j] - angle.sin() as f32).abs() < 1e-5);
        }
        // Only the height position: frequencies 1, 4, … 58 turn.
        let (_, sin) = mrope(&c, &[[0, 3, 0]]);
        let turned: Vec<usize> = (0..64).filter(|&j| sin[j] != 0.0).collect();
        assert_eq!(turned, (1..60).step_by(3).collect::<Vec<_>>());
        // Only the width position: 2, 5, … 59.
        let (_, sin) = mrope(&c, &[[0, 0, 3]]);
        let turned: Vec<usize> = (0..64).filter(|&j| sin[j] != 0.0).collect();
        assert_eq!(turned, (2..60).step_by(3).collect::<Vec<_>>());
    }
}
