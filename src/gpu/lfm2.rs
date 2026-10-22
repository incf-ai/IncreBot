//! The LFM2 forward pass (d1 classification) on one GPU, built from the Llama
//! path's kernels (`model.rs`) plus `short_conv` and half-split RoPE.
//!
//! Prefill-only: classification runs a shared prefix once and then a few tokens
//! per question, so every linear layer goes through the integer matmul, and the
//! state (conv rows and f16 K/V) never leaves the card. [`GpuLfm2::branches`]
//! snapshots the conv rows after the prefix and restores them before each
//! branch; the K/V past the prefix is simply overwritten.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use tracing::info;

use super::model::{
    CHUNK, Linear, MAX_KEYS, Mode, Quantized, gpu_blocks, matmul, matvec, quantize, round_up,
};
use super::vk::{Buffer, Commands, Device, Push};
use crate::model::lfm2::{Lfm2, Lfm2Config, Mixer as CpuMixer};
use crate::model::QLinear;

/// Submissions in flight at once (one layer each).
const RING: usize = 2;

enum Mixer {
    Conv {
        in_proj: Linear,
        out_proj: Linear,
        /// The `[d_model, taps]` f32 kernel's offset in the layer buffer.
        kernel: u64,
        /// Index of this layer's rows in the conv state.
        slot: usize,
    },
    Attn {
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        q_norm: u64,
        k_norm: u64,
        slot: usize,
    },
}

struct Layer {
    buf: Buffer,
    op_norm: u64,
    ffn_norm: u64,
    mixer: Mixer,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// The state and scratch buffers of the active GPU.
struct Active {
    n_ctx: usize,
    /// f16 K/V per attention layer.
    k: Vec<Buffer>,
    v: Vec<Buffer>,
    /// `[taps - 1, d]` f32 rows of `b · x` per conv layer, and its snapshot.
    conv: Buffer,
    conv_saved: Buffer,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    att: Buffer,
    /// Also holds the conv layers' `in_proj` output `[t, 3d]` (ffn_dim ≥ 3d).
    gate: Buffer,
    up: Buffer,
    last: Buffer,
    xq: Buffer,
    xd: Buffer,
    xs: Buffer,
    /// Host memory.
    logits: Buffer,
    cmds: Commands,
}

pub struct GpuLfm2 {
    pub dev: Arc<Device>,
    c: Lfm2Config,
    layers: Vec<Layer>,
    n_conv: usize,
    n_attn: usize,
    /// `out_norm` (f32) followed by the LM head blocks.
    head: Buffer,
    lm_head: Linear,
    rope: Buffer,
    active: Option<Active>,
}

impl GpuLfm2 {
    /// Copies `cpu`'s weights to `dev`; they must be Q4_K/Q6_K loaded with
    /// `LinearImpl::Packed` (d1-3B Q4_K_M is).
    pub fn load(dev: &Arc<Device>, cpu: &Lfm2) -> Result<Self> {
        let c = cpu.config.clone();
        ensure!(
            c.head_dim.is_multiple_of(64) && c.head_dim <= 128,
            "head_dim {} unsupported on GPU",
            c.head_dim
        );
        ensure!(
            c.d_model.is_multiple_of(256) && c.ffn_dim.is_multiple_of(256),
            "d_model {} / ffn_dim {} unsupported on GPU",
            c.d_model,
            c.ffn_dim
        );
        ensure!(c.ffn_dim >= 3 * c.d_model, "ffn_dim below 3 · d_model");
        ensure!((2..=8).contains(&c.conv_taps), "{} conv taps", c.conv_taps);
        let start = Instant::now();
        let (mut n_conv, mut n_attn) = (0, 0);
        let mut layers = Vec::with_capacity(c.n_layers);
        for (i, l) in cpu.layers.iter().enumerate() {
            // f32 vectors first, then the linear layers, each 256-byte aligned.
            let mut size = 0u64;
            let mut vectors: Vec<(u64, &[f32])> = Vec::new();
            let mut vector = |v: &'_ [f32]| {
                let off = size;
                size += round_up((v.len() * 4) as u64);
                off
            };
            let op_norm = vector(&l.op_norm);
            let ffn_norm = vector(&l.ffn_norm);
            vectors.push((op_norm, &l.op_norm));
            vectors.push((ffn_norm, &l.ffn_norm));
            let mut named: Vec<(&str, &dyn QLinear)> = Vec::new();
            let mixer_vectors: Vec<u64> = match &l.mixer {
                CpuMixer::Conv { in_proj, out_proj, kernel } => {
                    named.extend([("shortconv.in_proj", &**in_proj), ("shortconv.out_proj", &**out_proj)]);
                    let off = vector(kernel);
                    vectors.push((off, kernel));
                    vec![off]
                }
                CpuMixer::Attn { wq, wk, wv, wo, q_norm, k_norm, .. } => {
                    named.extend([
                        ("attn_q", &**wq),
                        ("attn_k", &**wk),
                        ("attn_v", &**wv),
                        ("attn_output", &**wo),
                    ]);
                    let (qo, ko) = (vector(q_norm), vector(k_norm));
                    vectors.push((qo, q_norm));
                    vectors.push((ko, k_norm));
                    vec![qo, ko]
                }
            };
            named.extend([("ffn_gate", &*l.w_gate), ("ffn_up", &*l.w_up), ("ffn_down", &*l.w_down)]);
            let mut parts = Vec::with_capacity(named.len());
            for (name, lin) in named {
                let (kind, blocks) = gpu_blocks(lin, &format!("blk.{i}.{name}"))?;
                let off = size;
                size += round_up(blocks.len() as u64);
                parts.push((
                    blocks,
                    Linear {
                        off,
                        kind,
                        k: lin.in_features() as u32,
                        n: lin.out_features() as u32,
                    },
                ));
            }
            let buf = dev.vram(size)?;
            for (off, v) in vectors {
                buf.write(off as usize, v);
            }
            for (blocks, lin) in &parts {
                buf.write(lin.off as usize, blocks);
            }
            let lins: Vec<Linear> = parts.iter().map(|&(_, l)| l).collect();
            let mixer = match (&l.mixer, &lins[..]) {
                (CpuMixer::Conv { .. }, &[in_proj, out_proj, ..]) => {
                    n_conv += 1;
                    Mixer::Conv {
                        in_proj,
                        out_proj,
                        kernel: mixer_vectors[0],
                        slot: n_conv - 1,
                    }
                }
                (CpuMixer::Attn { .. }, &[wq, wk, wv, wo, ..]) => {
                    n_attn += 1;
                    Mixer::Attn {
                        wq,
                        wk,
                        wv,
                        wo,
                        q_norm: mixer_vectors[0],
                        k_norm: mixer_vectors[1],
                        slot: n_attn - 1,
                    }
                }
                _ => unreachable!("the linear layers follow the mixer"),
            };
            let &[gate, up, down] = &lins[lins.len() - 3..] else {
                unreachable!("three FFN linear layers")
            };
            layers.push(Layer {
                buf,
                op_norm,
                ffn_norm,
                mixer,
                gate,
                up,
                down,
            });
        }
        let (kind, blocks) = gpu_blocks(&*cpu.lm_head, "output")?;
        let norm = round_up((c.d_model * 4) as u64);
        let head = dev.vram(norm + blocks.len() as u64)?;
        head.write(0, &cpu.out_norm);
        head.write(norm as usize, &blocks);
        let lm_head = Linear {
            off: norm,
            kind,
            k: cpu.lm_head.in_features() as u32,
            n: cpu.lm_head.out_features() as u32,
        };
        let half = c.head_dim / 2;
        let rope = dev.vram((c.n_ctx * half * 8) as u64)?;
        rope.write(0, &cpu.rope_cos);
        rope.write(c.n_ctx * half * 4, &cpu.rope_sin);
        info!(
            gpu = %dev.pci,
            mib = dev.tracker.held() >> 20,
            elapsed_s = start.elapsed().as_secs_f32(),
            "lfm2 weights uploaded"
        );
        Ok(Self {
            dev: dev.clone(),
            c,
            layers,
            n_conv,
            n_attn,
            head,
            lm_head,
            rope,
            active: None,
        })
    }

    /// Allocates the state for `n_ctx` positions and the scratch buffers.
    fn activate(&mut self, n_ctx: usize) -> Result<()> {
        if self.active.as_ref().is_some_and(|a| a.n_ctx >= n_ctx) {
            return Ok(());
        }
        self.active = None;
        let c = &self.c;
        ensure!(
            n_ctx <= MAX_KEYS && n_ctx <= c.n_ctx,
            "n_ctx {n_ctx} too large for the GPU backend"
        );
        let kvw = (c.n_kv_heads * c.head_dim) as u64;
        let (d, ffn, ch) = (c.d_model as u64, c.ffn_dim as u64, CHUNK as u64);
        let conv = (self.n_conv * (c.conv_taps - 1) * c.d_model * 4) as u64;
        let dev = &self.dev;
        let mut k = Vec::with_capacity(self.n_attn);
        let mut v = Vec::with_capacity(self.n_attn);
        for _ in 0..self.n_attn {
            k.push(dev.vram(n_ctx as u64 * kvw * 2)?);
            v.push(dev.vram(n_ctx as u64 * kvw * 2)?);
        }
        let qw = (c.n_heads * c.head_dim) as u64;
        self.active = Some(Active {
            n_ctx,
            k,
            v,
            conv: dev.vram(conv)?,
            conv_saved: dev.vram(conv)?,
            x: dev.vram(ch * d * 4)?,
            h: dev.vram(ch * d * 4)?,
            q: dev.vram(ch * qw * 4)?,
            k_new: dev.vram(ch * kvw * 4)?,
            v_new: dev.vram(ch * kvw * 4)?,
            att: dev.vram(ch * d.max(qw) * 4)?,
            gate: dev.vram(ch * ffn * 4)?,
            up: dev.vram(ch * ffn * 4)?,
            last: dev.vram(d * 4)?,
            xq: dev.vram(ch * d.max(ffn))?,
            xd: dev.vram(ch * d.max(ffn) / 256 * 4)?,
            xs: dev.vram(ch * d.max(ffn) / 16 * 4)?,
            logits: dev.host(c.vocab as u64 * 4)?,
            cmds: Commands::new(dev, RING)?,
        });
        Ok(())
    }

    /// Frees the state and scratch buffers, keeping the weights.
    pub fn deactivate(&mut self) {
        self.active = None;
    }

    /// Runs `trunk` from an empty state, then each branch after it (each from
    /// the state the trunk left), and returns every branch's last-position
    /// logits. Like the CPU [`Classifier`](crate::classify::Classifier) path.
    pub fn branches(&mut self, cpu: &Lfm2, trunk: &[u32], branches: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        ensure!(branches.iter().all(|b| !b.is_empty()), "an empty branch");
        let longest = branches.iter().map(|b| b.len()).max().unwrap_or(0);
        self.activate(trunk.len() + longest)?;
        let a = self.active.as_mut().context("GPU model is not active")?;
        a.cmds.wait_all()?;
        a.conv.bytes_mut().fill(0);
        if !trunk.is_empty() {
            self.forward(cpu, trunk, 0, false)?;
        }
        let a = self.active.as_mut().context("GPU model is not active")?;
        let bytes = a.conv.size;
        a.cmds.begin()?;
        a.cmds.copy(&a.conv, 0, &a.conv_saved, 0, bytes)?;
        a.cmds.submit()?;
        let mut out = Vec::with_capacity(branches.len());
        for b in branches {
            let a = self.active.as_mut().context("GPU model is not active")?;
            a.cmds.begin()?;
            a.cmds.copy(&a.conv_saved, 0, &a.conv, 0, bytes)?;
            a.cmds.submit()?;
            out.push(self.forward(cpu, b, trunk.len(), true)?);
        }
        Ok(out)
    }

    /// Runs `tokens` at positions `past..` on the GPU-resident state; with
    /// `logits`, returns the last position's (else an empty vector).
    fn forward(&mut self, cpu: &Lfm2, tokens: &[u32], past: usize, logits: bool) -> Result<Vec<f32>> {
        let c = &self.c;
        let a = self.active.as_mut().context("GPU model is not active")?;
        ensure!(
            past + tokens.len() <= a.n_ctx,
            "context overflow on GPU: {past} + {} > {}",
            tokens.len(),
            a.n_ctx
        );
        let chunks = tokens.chunks(CHUNK).count();
        for (ci, chunk) in tokens.chunks(CHUNK).enumerate() {
            let t = chunk.len();
            let pos = past + ci * CHUNK;
            let x = cpu.embed(chunk)?;
            // The previous chunk's layers still read `x` until they finish.
            a.cmds.wait_all()?;
            a.x.write(0, &x);
            for layer in &self.layers {
                a.cmds.begin()?;
                record_layer(&self.dev, a, c, layer, &self.rope, t, pos)?;
                a.cmds.submit()?;
            }
            if logits && ci + 1 == chunks {
                a.cmds.begin()?;
                a.cmds.dispatch(
                    "gpu_norm",
                    self.dev.pipes.rmsnorm,
                    Push::default()
                        .addr(a.x.addr + ((t - 1) * c.d_model * 4) as u64)
                        .addr(self.head.addr)
                        .addr(a.last.addr)
                        .u32(c.d_model as u32)
                        .f32(c.rms_eps),
                    [1, 1, 1],
                )?;
                matvec(
                    &self.dev,
                    &mut a.cmds,
                    &[((self.head.addr + self.lm_head.off, self.lm_head), a.logits.addr)],
                    a.last.addr,
                    Mode::Plain,
                )?;
                a.cmds.submit()?;
            }
        }
        a.cmds.wait_all()?;
        Ok(if logits { a.logits.read(0, c.vocab) } else { Vec::new() })
    }
}

/// Records one layer for `t` tokens at positions `past..past + t`.
fn record_layer(
    dev: &Device,
    a: &mut Active,
    c: &Lfm2Config,
    l: &Layer,
    rope: &Buffer,
    t: usize,
    past: usize,
) -> Result<()> {
    let p = &dev.pipes;
    let (d, hd, t32) = (c.d_model as u32, c.head_dim as u32, t as u32);
    let w = |lin: Linear| (l.buf.addr + lin.off, lin);
    // RMSNorm of `rows` rows of `width`, weights at `off` in the layer buffer.
    let norm = |cmds: &mut Commands, off: u64, x: u64, y: u64, width: u32, rows: u32| -> Result<()> {
        cmds.dispatch(
            "gpu_norm",
            p.rmsnorm,
            Push::default()
                .addr(x)
                .addr(l.buf.addr + off)
                .addr(y)
                .u32(width)
                .f32(c.rms_eps),
            [rows, 1, 1],
        )
    };
    let qx = Quantized {
        q: a.xq.addr,
        d: a.xd.addr,
        s: a.xs.addr,
    };
    let cmds = &mut a.cmds;
    norm(cmds, l.op_norm, a.x.addr, a.h.addr, d, t32)?;
    let xq = quantize(dev, cmds, qx, a.h.addr, d, t32)?;
    match l.mixer {
        Mixer::Conv {
            in_proj,
            out_proj,
            kernel,
            slot,
        } => {
            let keep = (c.conv_taps - 1) * c.d_model;
            matmul(dev, cmds, w(in_proj), xq, a.gate.addr, t32, false)?;
            cmds.dispatch(
                "gpu_short_conv",
                p.short_conv,
                Push::default()
                    .addr(a.gate.addr)
                    .addr(l.buf.addr + kernel)
                    .addr(a.conv.addr + (slot * keep * 4) as u64)
                    .addr(a.att.addr)
                    .u32(t32)
                    .u32(d)
                    .u32(c.conv_taps as u32),
                [d.div_ceil(256), 1, 1],
            )?;
            let xq = quantize(dev, cmds, qx, a.att.addr, d, t32)?;
            matmul(dev, cmds, w(out_proj), xq, a.x.addr, t32, true)?;
        }
        Mixer::Attn {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            slot,
        } => {
            let (heads, kv_heads) = (c.n_heads as u32, c.n_kv_heads as u32);
            matmul(dev, cmds, w(wq), xq, a.q.addr, t32, false)?;
            matmul(dev, cmds, w(wk), xq, a.k_new.addr, t32, false)?;
            matmul(dev, cmds, w(wv), xq, a.v_new.addr, t32, false)?;
            norm(cmds, q_norm, a.q.addr, a.q.addr, hd, t32 * heads)?;
            norm(cmds, k_norm, a.k_new.addr, a.k_new.addr, hd, t32 * kv_heads)?;
            let half = c.n_ctx * c.head_dim / 2 * 4;
            let pairs = t32 * (heads + 2 * kv_heads) * hd / 2;
            cmds.dispatch(
                "gpu_rope_store",
                p.rope_store_neox,
                Push::default()
                    .addr(a.q.addr)
                    .addr(a.k_new.addr)
                    .addr(a.v_new.addr)
                    .addr(rope.addr)
                    .addr(rope.addr + half as u64)
                    .addr(a.k[slot].addr)
                    .addr(a.v[slot].addr)
                    .u32(t32)
                    .u32(heads)
                    .u32(kv_heads)
                    .u32(hd)
                    .u32(past as u32),
                [pairs.div_ceil(256), 1, 1],
            )?;
            cmds.dispatch(
                "gpu_attention",
                p.flash_attn,
                Push::default()
                    .addr(a.q.addr)
                    .addr(a.k[slot].addr)
                    .addr(a.v[slot].addr)
                    .addr(a.att.addr)
                    .u32(heads)
                    .u32(kv_heads)
                    .u32(hd)
                    .u32(past as u32)
                    .f32(1.0 / (c.head_dim as f32).sqrt())
                    .u32(t32),
                [heads, t32.div_ceil(32), 1],
            )?;
            let xq = quantize(dev, cmds, qx, a.att.addr, heads * hd, t32)?;
            matmul(dev, cmds, w(wo), xq, a.x.addr, t32, true)?;
        }
    }
    norm(cmds, l.ffn_norm, a.x.addr, a.h.addr, d, t32)?;
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
    Ok(())
}
