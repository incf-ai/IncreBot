//! Qwen3-VL's vision tower ([`crate::describe::vision`]) on one GPU: the same
//! blocks and mergers, its f16 weights in VRAM, f32 activations. The host
//! still cuts the patches and interpolates the position grid; the rest runs
//! as f16-weight matmuls (`matmul_f16`, with the bias, GELU and residual add
//! folded in), layer norms, the 2D rotary embedding and non-causal attention
//! over every patch (`vit_attn`, head size 72).

use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use super::model::round_up;
use super::vk::{Buffer, Commands, Device, Push};
use crate::classify::vision::PATCH_VALUES;
use crate::classify::vision::siglip2::{LayerNorm, Linear, linear_in};
use crate::describe::vision::{self, Features, MERGE, Patches, Vision};

/// Submissions in flight at once (one block each).
const RING: usize = 2;
/// The head size `vit_attn` is built for.
const HEAD_DIM: usize = 72;

/// `matmul_f16` flags.
const BIAS: u32 = 1;
const GELU_TANH: u32 = 2;
const GELU_ERF: u32 = 4;
const ACCUMULATE: u32 = 8;

/// A linear layer's place in its buffer: f16 rows `[n, k]`, then the f32 bias.
#[derive(Clone, Copy)]
struct Lin {
    w: u64,
    b: u64,
    k: u32,
    n: u32,
}

/// A layer norm's weight and bias (f32) in its buffer.
#[derive(Clone, Copy)]
struct Norm {
    w: u64,
    b: u64,
    width: u32,
}

/// Parts gathered for one VRAM buffer, each 256-byte aligned.
#[derive(Default)]
struct Pack {
    size: u64,
    parts: Vec<(u64, Vec<u8>)>,
}

impl Pack {
    fn put(&mut self, bytes: Vec<u8>) -> u64 {
        let off = self.size;
        self.size += round_up(bytes.len() as u64);
        self.parts.push((off, bytes));
        off
    }

    fn floats(&mut self, v: &[f32]) -> u64 {
        self.put(v.iter().flat_map(|f| f.to_le_bytes()).collect())
    }

    fn linear(&mut self, l: &Linear, name: &str) -> Result<Lin> {
        let w = l.f16_bytes().with_context(|| format!("{name} is Q8_0; the GPU tower takes f16"))?;
        let (k, n) = (linear_in(l), l.out_features());
        ensure!(k.is_multiple_of(16), "{name} takes {k} values, not a multiple of 16");
        Ok(Lin { w: self.put(w.into_owned()), b: self.floats(l.bias()), k: k as u32, n: n as u32 })
    }

    fn norm(&mut self, l: &LayerNorm) -> Norm {
        Norm { w: self.floats(l.weight()), b: self.floats(l.bias()), width: l.weight().len() as u32 }
    }

    fn upload(self, dev: &Arc<Device>) -> Result<Buffer> {
        let buf = dev.vram(self.size)?;
        for (off, bytes) in &self.parts {
            buf.write(*off as usize, bytes);
        }
        Ok(buf)
    }
}

struct Layer {
    buf: Buffer,
    ln1: Norm,
    qkv: Lin,
    out: Lin,
    ln2: Norm,
    up: Lin,
    down: Lin,
}

struct Merger {
    buf: Buffer,
    norm: Norm,
    fc1: Lin,
    fc2: Lin,
}

/// Activations for the largest picture.
struct Scratch {
    values: Buffer,
    x: Buffer,
    h: Buffer,
    qkv: Buffer,
    att: Buffer,
    up: Buffer,
    /// The mergers' hidden rows.
    mid: Buffer,
    /// The rotary tables: `cos` then `sin`, `[patches, head_dim / 2]` each.
    rope: Buffer,
    /// Host memory: the main rows, then each DeepStack level's.
    out: Buffer,
}

pub struct GpuVision {
    dev: Arc<Device>,
    width: usize,
    heads: usize,
    eps: f32,
    out_width: usize,
    positions: Vec<f32>,
    side: usize,
    /// The patch embedding (and nothing else).
    patch_buf: Buffer,
    patch: Lin,
    layers: Vec<Layer>,
    /// After which block each DeepStack merger reads.
    deepstack: Vec<(usize, Merger)>,
    merger: Merger,
    max_patches: usize,
    scratch: Scratch,
    cmd: Commands,
}

impl GpuVision {
    /// Uploads `v`'s weights to `dev`, with room for pictures of up to
    /// `max_tokens` merged rows.
    pub fn load(dev: &Arc<Device>, v: &Vision, max_tokens: usize) -> Result<Self> {
        let (d, heads) = (v.width, v.heads);
        ensure!(d / heads == HEAD_DIM, "head size {}; the GPU tower is built for {HEAD_DIM}", d / heads);
        let mut pack = Pack::default();
        let patch = pack.linear(&v.patch_embd, "v.patch_embd")?;
        ensure!(patch.k as usize == PATCH_VALUES, "patch embedding takes {}", patch.k);
        let patch_buf = pack.upload(dev)?;

        let layers = v
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let p = format!("v.blk.{i}");
                let mut pack = Pack::default();
                let ln1 = pack.norm(&l.ln1);
                let qkv = pack.linear(&l.qkv, &format!("{p}.attn_qkv"))?;
                let out = pack.linear(&l.out, &format!("{p}.attn_out"))?;
                let ln2 = pack.norm(&l.ln2);
                let up = pack.linear(&l.up, &format!("{p}.ffn_up"))?;
                let down = pack.linear(&l.down, &format!("{p}.ffn_down"))?;
                Ok(Layer { buf: pack.upload(dev)?, ln1, qkv, out, ln2, up, down })
            })
            .collect::<Result<Vec<_>>>()?;
        let ffn = layers.first().context("no blocks")?.up.n as usize;
        ensure!(layers.iter().all(|l| l.up.n as usize == ffn && l.down.k as usize == ffn), "FFN widths differ");

        let merger = |m: &vision::Merger, name: &str| -> Result<Merger> {
            let mut pack = Pack::default();
            let norm = pack.norm(&m.norm);
            let fc1 = pack.linear(&m.fc1, &format!("{name}.fc1"))?;
            let fc2 = pack.linear(&m.fc2, &format!("{name}.fc2"))?;
            ensure!(norm.width as usize == d || norm.width as usize == d * MERGE * MERGE, "{name} norm width");
            Ok(Merger { buf: pack.upload(dev)?, norm, fc1, fc2 })
        };
        let deepstack =
            v.deepstack.iter().map(|(i, m)| Ok((*i, merger(m, &format!("v.deepstack.{i}"))?))).collect::<Result<Vec<_>>>()?;
        let main = merger(&v.merger, "mm")?;
        let out_width = main.fc2.n as usize;
        let hidden = main.fc1.n.max(deepstack.iter().map(|(_, m)| m.fc1.n).max().unwrap_or(0)) as u64;

        let max_patches = max_tokens * MERGE * MERGE;
        let (n, d64) = (max_patches as u64, d as u64);
        let scratch = Scratch {
            values: dev.vram(n * PATCH_VALUES as u64 * 4)?,
            x: dev.vram(n * d64 * 4)?,
            h: dev.vram(n * d64 * 4)?,
            qkv: dev.vram(n * 3 * d64 * 4)?,
            att: dev.vram(n * d64 * 4)?,
            up: dev.vram(n * ffn as u64 * 4)?,
            mid: dev.vram(max_tokens as u64 * hidden * 4)?,
            rope: dev.vram(n * HEAD_DIM as u64 * 4)?,
            out: dev.host((1 + deepstack.len()) as u64 * max_tokens as u64 * out_width as u64 * 4)?,
        };
        let (grid, side) = v.position_grid();
        Ok(Self {
            dev: dev.clone(),
            width: d,
            heads,
            eps: v.eps,
            out_width,
            positions: grid.to_vec(),
            side,
            patch_buf,
            patch,
            layers,
            deepstack,
            merger: main,
            max_patches,
            scratch,
            cmd: Commands::new(dev, RING)?,
        })
    }

    pub fn out_width(&self) -> usize {
        self.out_width
    }

    pub fn deepstack_levels(&self) -> usize {
        self.deepstack.len()
    }

    /// `y (+)= x · Wᵀ (+ b)` for `m` rows, the activation per `flags`.
    #[allow(clippy::too_many_arguments)]
    fn matmul(&mut self, label: &'static str, x: u64, base: u64, l: Lin, y: u64, m: usize, flags: u32) -> Result<()> {
        let push = Push::default()
            .addr(x)
            .addr(base + l.w)
            .addr(base + l.b)
            .addr(y)
            .u32(m as u32)
            .u32(l.n)
            .u32(l.k)
            .u32(flags | BIAS);
        let pipe = self.dev.pipes.matmul_f16;
        self.cmd.dispatch(label, pipe, push, [l.n.div_ceil(64), (m as u32).div_ceil(64), 1])
    }

    fn layernorm(&mut self, x: u64, base: u64, l: Norm, y: u64, rows: usize) -> Result<()> {
        let push = Push::default().addr(x).addr(base + l.w).addr(base + l.b).addr(y).u32(l.width).f32(self.eps);
        let pipe = self.dev.pipes.layernorm;
        self.cmd.dispatch("vit_layernorm", pipe, push, [rows as u32, 1, 1])
    }

    /// A merger on `x` (`[n, width]` as `[n / 4, 4 · width]`), its rows to
    /// `slot` of the output.
    fn merge(&mut self, which: Option<usize>, n: usize, slot: usize) -> Result<()> {
        let m = match which {
            Some(i) => &self.deepstack[i].1,
            None => &self.merger,
        };
        let (base, norm, fc1, fc2) = (m.buf.addr, m.norm, m.fc1, m.fc2);
        let tokens = n / (MERGE * MERGE);
        let s = &self.scratch;
        let (x, h, mid) = (s.x.addr, s.h.addr, s.mid.addr);
        let out = s.out.addr + (slot * tokens * self.out_width * 4) as u64;
        self.layernorm(x, base, norm, h, n * self.width / norm.width as usize)?;
        self.matmul("vit_merge_fc1", h, base, fc1, mid, tokens, GELU_ERF)?;
        self.matmul("vit_merge_fc2", mid, base, fc2, out, tokens, 0)
    }

    /// The tower and mergers on one picture; `None` when `stop` turned true.
    pub fn features(&mut self, p: &Patches, stop: &dyn Fn() -> bool) -> Result<Option<Features>> {
        let (gh, gw) = p.grid;
        ensure!(gh.is_multiple_of(MERGE) && gw.is_multiple_of(MERGE), "patch grid {gh}×{gw}");
        let n = gh * gw;
        ensure!(n <= self.max_patches, "{n} patches, room for {}", self.max_patches);
        ensure!(p.values.len() == n * PATCH_VALUES, "patch values");
        let (d, half) = (self.width, HEAD_DIM / 2);

        let s = &self.scratch;
        s.values.write(0, &p.values);
        s.x.write(0, &vision::interpolate(&self.positions, self.side, d, gh, gw));
        let (cos, sin) = vision::rope(n, gw, HEAD_DIM);
        s.rope.write(0, &cos);
        s.rope.write(n * half * 4, &sin);
        let (values, x, h, qkv, att, up, rope) = (s.values.addr, s.x.addr, s.h.addr, s.qkv.addr, s.att.addr, s.up.addr, s.rope.addr);

        self.cmd.begin()?;
        self.matmul("vit_patch", values, self.patch_buf.addr, self.patch, x, n, ACCUMULATE)?;
        let mut levels = 0;
        for i in 0..self.layers.len() {
            if i > 0 {
                self.cmd.begin()?;
            }
            let l = &self.layers[i];
            let (base, ln1, wqkv, wout, ln2, wup, wdown) = (l.buf.addr, l.ln1, l.qkv, l.out, l.ln2, l.up, l.down);
            self.layernorm(x, base, ln1, h, n)?;
            self.matmul("vit_qkv", h, base, wqkv, qkv, n, 0)?;
            let push = Push::default().addr(qkv).addr(rope).u32(n as u32).u32(self.heads as u32).u32(HEAD_DIM as u32);
            self.cmd.dispatch("vit_rope", self.dev.pipes.vit_rope, push, [n as u32, 1, 1])?;
            let push = Push::default()
                .addr(qkv)
                .addr(att)
                .u32(n as u32)
                .u32(self.heads as u32)
                .f32(1.0 / (HEAD_DIM as f32).sqrt());
            self.cmd.dispatch("vit_attn", self.dev.pipes.vit_attn, push, [(n as u32).div_ceil(64), self.heads as u32, 1])?;
            self.matmul("vit_out", att, base, wout, x, n, ACCUMULATE)?;
            self.layernorm(x, base, ln2, h, n)?;
            self.matmul("vit_up", h, base, wup, up, n, GELU_TANH)?;
            self.matmul("vit_down", up, base, wdown, x, n, ACCUMULATE)?;
            for j in 0..self.deepstack.len() {
                if self.deepstack[j].0 == i {
                    levels += 1;
                    self.merge(Some(j), n, levels)?;
                }
            }
            self.cmd.submit()?;
            if stop() {
                self.cmd.wait_all()?;
                return Ok(None);
            }
        }
        self.cmd.begin()?;
        self.merge(None, n, 0)?;
        self.cmd.submit()?;
        self.cmd.wait_all()?;

        let tokens = n / (MERGE * MERGE);
        let rows = tokens * self.out_width;
        let read = |slot: usize| self.scratch.out.read::<f32>(slot * rows * 4, rows);
        Ok(Some(Features { main: read(0), deepstack: (1..=levels).map(read).collect(), grid: p.merged() }))
    }
}
