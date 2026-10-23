//! The Llama forward pass on one GPU (gpu-plan.md §1).
//!
//! Weights are the CPU model's GGUF K-quant blocks, copied as-is into one VRAM
//! buffer per layer. A [`GpuModel`] is either *standby* (weights only) or
//! *active* (also the KV cache and scratch buffers, see [`activate`]).
//!
//! The host [`KvCache`] stays the canonical copy: each forward first uploads the
//! positions the GPU lacks (by longest common token prefix), and the K/V rows it
//! computes are copied back and appended to the host cache as each chunk
//! finishes. So a reply can move to another backend at any chunk boundary with
//! nothing to copy, and freeing a card never loses state.
//!
//! [`activate`]: GpuModel::activate

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use half::f16;
use tracing::info;

use super::vk::{Buffer, Commands, Device, Push};
use crate::gguf::kquant::Q6_K_BLOCK_BYTES;
use crate::model::{KQuant, KvCache, Llama, LlamaConfig, QLinear};

/// Prefill runs in chunks of this many tokens, as on the CPU.
pub const CHUNK: usize = 512;

/// Largest context: split decode attention's partial buffer and positions in
/// u32 indices stay small.
pub(super) const MAX_KEYS: usize = 8192;

/// Keys per workgroup in split decode attention (`attention_split.comp`).
pub(super) const SPLIT_KEYS: usize = 128;

/// Submissions in flight at once. Each is one layer, so a cancel waits for at
/// most this many layers.
const RING: usize = 2;

/// Where a linear layer's blocks sit inside its layer buffer.
#[derive(Clone, Copy)]
pub(super) struct Linear {
    pub(super) off: u64,
    pub(super) kind: KQuant,
    pub(super) k: u32,
    pub(super) n: u32,
}

struct Layer {
    buf: Buffer,
    attn_norm: u64,
    ffn_norm: u64,
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
    /// The tokens whose K/V the GPU cache holds.
    tokens: Vec<u32>,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k_new: Buffer,
    v_new: Buffer,
    att: Buffer,
    gate: Buffer,
    up: Buffer,
    last: Buffer,
    /// Split decode attention's partial results.
    att_part: Buffer,
    /// Prefill activations quantized for the integer matmul: values, block
    /// scales, sums of 16.
    xq: Buffer,
    xd: Buffer,
    xs: Buffer,
    /// Host memory: the logits and the new K/V rows of each chunk.
    logits: Buffer,
    readback: Buffer,
    cmds: Commands,
}

pub struct GpuModel {
    pub dev: Arc<Device>,
    c: LlamaConfig,
    layers: Vec<Layer>,
    /// `out_norm` (f32) followed by the LM head blocks.
    head: Buffer,
    lm_head: Linear,
    rope: Buffer,
    active: Option<Active>,
}

/// Why a forward stopped early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interrupted {
    /// `stop` returned true. The host cache holds every chunk finished so far.
    Stopped,
}

pub(super) fn round_up(v: u64) -> u64 {
    v.next_multiple_of(256)
}

/// A linear layer's weights in the GPU's layout: Q4_K blocks as-is; Q6_K
/// repacked by [`repack_q6k`].
pub(super) fn gpu_blocks<'a>(l: &'a dyn QLinear, name: &str) -> Result<(KQuant, Cow<'a, [u8]>)> {
    let (kind, blocks) = l.k_blocks().with_context(|| {
        format!("{name}: the GPU backend needs Q4_K/Q6_K weights loaded with linear_impl: Packed")
    })?;
    Ok(match kind {
        KQuant::Q4K => (kind, Cow::Borrowed(blocks)),
        KQuant::Q6K => (kind, Cow::Owned(repack_q6k(blocks))),
    })
}

/// Q6_K blocks (210 bytes: ql 128, qh 64, scales 16, d 2) split into the ql+qh
/// parts of every block, then (scales, d, 2 bytes padding) per block, so the
/// shaders load the bulk with aligned 8- and 16-byte reads (common.glsl).
pub fn repack_q6k(blocks: &[u8]) -> Vec<u8> {
    let (blocks, rest) = blocks.as_chunks::<Q6_K_BLOCK_BYTES>();
    assert!(rest.is_empty(), "not a whole number of Q6_K blocks");
    let mut out = Vec::with_capacity(blocks.len() * (192 + 20));
    for b in blocks {
        out.extend_from_slice(&b[..192]);
    }
    for b in blocks {
        out.extend_from_slice(&b[192..]);
        out.extend_from_slice(&[0, 0]);
    }
    out
}

impl GpuModel {
    /// Copies `cpu`'s weights to `dev` (standby). Checks `abort` between layers;
    /// when it is set, frees what was uploaded and fails.
    pub fn load(dev: &Arc<Device>, cpu: &Llama, abort: &AtomicBool) -> Result<Self> {
        let c = cpu.config.clone();
        ensure!(
            c.head_dim.is_multiple_of(64) && c.head_dim <= 128,
            "head_dim {} unsupported on GPU",
            c.head_dim
        );
        ensure!(
            c.d_model.is_multiple_of(256),
            "d_model {} unsupported on GPU",
            c.d_model
        );
        let start = Instant::now();
        let mut layers = Vec::with_capacity(c.n_layers);
        for (i, l) in cpu.layers.iter().enumerate() {
            if abort.load(Ordering::Relaxed) {
                bail!("upload to {} aborted at layer {i}", dev.pci);
            }
            let named = [
                ("attn_q", &*l.wq),
                ("attn_k", &*l.wk),
                ("attn_v", &*l.wv),
                ("attn_output", &*l.wo),
                ("ffn_gate", &*l.w_gate),
                ("ffn_up", &*l.w_up),
                ("ffn_down", &*l.w_down),
            ];
            let norms = round_up((c.d_model * 4) as u64);
            let mut size = 2 * norms;
            let mut parts = Vec::new();
            for (name, lin) in named {
                let (kind, blocks) = gpu_blocks(lin, &format!("blk.{i}.{name}"))?;
                let len = blocks.len() as u64;
                parts.push((size, kind, blocks, lin));
                size += round_up(len);
            }
            let buf = dev.vram(size)?;
            buf.write(0, &l.attn_norm);
            buf.write(norms as usize, &l.ffn_norm);
            let lins: Vec<Linear> = parts
                .iter()
                .map(|&(off, kind, ref blocks, lin)| {
                    buf.write(off as usize, blocks);
                    Linear {
                        off,
                        kind,
                        k: lin.in_features() as u32,
                        n: lin.out_features() as u32,
                    }
                })
                .collect();
            // `named` above has 7 entries; this catches it (instead of silently
            // dropping or panicking) if that ever changes without updating here.
            let [wq, wk, wv, wo, gate, up, down]: [Linear; 7] = lins
                .try_into()
                .map_err(|v: Vec<Linear>| anyhow!("expected 7 linear layers per block, got {}", v.len()))?;
            layers.push(Layer {
                buf,
                attn_norm: 0,
                ffn_norm: norms,
                wq,
                wk,
                wv,
                wo,
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
            "weights uploaded"
        );
        Ok(Self {
            dev: dev.clone(),
            c,
            layers,
            head,
            lm_head,
            rope,
            active: None,
        })
    }

    /// One token through the whole model on a small cache: the logits must be
    /// finite and not all equal. Run after every upload, before the card serves
    /// replies (like the CPU self-test at startup).
    pub fn self_test(&mut self, cpu: &Llama) -> Result<()> {
        let was_active = self.is_active();
        self.activate(16)?;
        let mut cache = cpu.new_cache();
        let logits = self
            .forward(cpu, &[1], &mut cache, &|| false)?
            .map_err(|_| anyhow::anyhow!("self-test interrupted"))?;
        if !was_active {
            self.deactivate();
        }
        ensure!(
            logits.iter().all(|l| l.is_finite()),
            "GPU self-test: non-finite logits"
        );
        let (lo, hi) = logits
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &l| {
                (lo.min(l), hi.max(l))
            });
        ensure!(hi - lo > 1e-3, "GPU self-test: constant logits ({lo})");
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// Allocates the KV cache for `n_ctx` positions and the scratch buffers.
    pub fn activate(&mut self, n_ctx: usize) -> Result<()> {
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
            tokens: Vec::new(),
            x: dev.vram(ch * d * 4)?,
            h: dev.vram(ch * d * 4)?,
            q: dev.vram(ch * d * 4)?,
            k_new: dev.vram(ch * kvw * 4)?,
            v_new: dev.vram(ch * kvw * 4)?,
            att: dev.vram(ch * d * 4)?,
            gate: dev.vram(ch * ffn * 4)?,
            up: dev.vram(ch * ffn * 4)?,
            last: dev.vram(d * 4)?,
            xq: dev.vram(ch * d.max(ffn))?,
            xd: dev.vram(ch * d.max(ffn) / 256 * 4)?,
            xs: dev.vram(ch * d.max(ffn) / 16 * 4)?,
            att_part: dev
                .vram((c.n_heads * n_ctx.div_ceil(SPLIT_KEYS) * (c.head_dim + 2) * 4) as u64)?,
            logits: dev.host(c.vocab as u64 * 4)?,
            readback: dev.host(c.n_layers as u64 * 2 * ch * kvw * 2)?,
            cmds: Commands::new(dev, RING)?,
        });
        Ok(())
    }

    /// Frees the KV cache and scratch buffers, keeping the weights (standby).
    pub fn deactivate(&mut self) {
        self.active = None;
    }

    /// Runs `tokens` after what `cache` holds and returns the last position's
    /// logits, like [`Llama::forward_cancellable`]. `stop` is checked before each
    /// layer; when it fires, the cache keeps every finished chunk and
    /// `Err(Interrupted)` is returned inside the `Ok`.
    pub fn forward(
        &mut self,
        cpu: &Llama,
        tokens: &[u32],
        cache: &mut KvCache,
        stop: &dyn Fn() -> bool,
    ) -> Result<std::result::Result<Vec<f32>, Interrupted>> {
        let c = self.c.clone();
        let a = self.active.as_mut().context("GPU model is not active")?;
        ensure!(!tokens.is_empty(), "forward called with no tokens");
        ensure!(
            cache.len() + tokens.len() <= a.n_ctx,
            "context overflow on GPU: {} + {} > {}",
            cache.len(),
            tokens.len(),
            a.n_ctx
        );
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= c.vocab) {
            bail!("token id {bad} out of range (vocab {})", c.vocab);
        }
        let kvw = c.n_kv_heads * c.head_dim;
        sync_kv(a, cache, c.n_layers, kvw)?;
        cache.reserve_positions(cache.len() + tokens.len());

        let chunks = tokens.chunks(CHUNK).count();
        for (ci, chunk) in tokens.chunks(CHUNK).enumerate() {
            let (t, past) = (chunk.len(), cache.len());
            a.x.write(0, &cpu.embed(chunk));
            for (li, layer) in self.layers.iter().enumerate() {
                if stop() {
                    a.cmds.wait_all()?;
                    a.tokens.truncate(cache.len());
                    return Ok(Err(Interrupted::Stopped));
                }
                a.cmds.begin()?;
                record_layer(&self.dev, a, &c, layer, &self.rope, li, t, past)?;
                let row_bytes = (kvw * 2) as u64;
                let rb = (li * 2 * CHUNK * kvw * 2) as u64;
                let (pos, len) = (past as u64 * row_bytes, t as u64 * row_bytes);
                a.cmds.copy(&a.k[li], pos, &a.readback, rb, len)?;
                a.cmds.copy(
                    &a.v[li],
                    pos,
                    &a.readback,
                    rb + (CHUNK * kvw * 2) as u64,
                    len,
                )?;
                a.cmds.submit()?;
            }
            if ci + 1 == chunks {
                a.cmds.begin()?;
                let d = c.d_model as u32;
                let p = &self.dev.pipes;
                a.cmds.dispatch(
                    "gpu_norm",
                    p.rmsnorm,
                    Push::default()
                        .addr(a.x.addr + ((t - 1) * c.d_model * 4) as u64)
                        .addr(self.head.addr)
                        .addr(a.last.addr)
                        .u32(d)
                        .f32(c.rms_eps),
                    [1, 1, 1],
                )?;
                matvec(
                    &self.dev,
                    &mut a.cmds,
                    &[(
                        (self.head.addr + self.lm_head.off, self.lm_head),
                        a.logits.addr,
                    )],
                    a.last.addr,
                    Mode::Plain,
                )?;
                a.cmds.submit()?;
            }
            a.cmds.wait_all()?;
            for li in 0..c.n_layers {
                let rb = li * 2 * CHUNK * kvw * 2;
                let k: &[f16] = a.readback.slice(rb, t * kvw);
                let v: &[f16] = a.readback.slice(rb + CHUNK * kvw * 2, t * kvw);
                cache.append_f16(li, k, v);
            }
            cache.tokens.extend_from_slice(chunk);
            a.tokens.extend_from_slice(chunk);
        }
        Ok(Ok(a.logits.read(0, c.vocab)))
    }
}

/// Uploads the host cache's positions the GPU cache lacks: everything after the
/// longest common token prefix of the two.
fn sync_kv(a: &mut Active, cache: &KvCache, layers: usize, kvw: usize) -> Result<()> {
    ensure!(cache.kv_width() == kvw, "KV width mismatch");
    let keep = a
        .tokens
        .iter()
        .zip(&cache.tokens)
        .take_while(|(x, y)| x == y)
        .count();
    a.tokens.truncate(keep);
    if keep == cache.len() {
        return Ok(());
    }
    for l in 0..layers {
        let (k, v) = cache
            .f16_rows(l, keep)
            .context("the GPU backend needs an f16 KV cache")?;
        a.k[l].write(keep * kvw * 2, k);
        a.v[l].write(keep * kvw * 2, v);
    }
    a.tokens.extend_from_slice(&cache.tokens[keep..]);
    Ok(())
}

/// Records one decoder layer for `t` tokens at positions `past..past + t`.
#[allow(clippy::too_many_arguments)]
fn record_layer(
    dev: &Device,
    a: &mut Active,
    c: &LlamaConfig,
    l: &Layer,
    rope: &Buffer,
    li: usize,
    t: usize,
    past: usize,
) -> Result<()> {
    let p = &dev.pipes;
    let (d, hd) = (c.d_model as u32, c.head_dim as u32);
    let t32 = t as u32;
    let w = |lin: Linear| (l.buf.addr + lin.off, lin);
    let norm = |cmds: &mut Commands, off: u64, x: u64, y: u64| -> Result<()> {
        cmds.dispatch(
            "gpu_norm",
            p.rmsnorm,
            Push::default()
                .addr(x)
                .addr(l.buf.addr + off)
                .addr(y)
                .u32(d)
                .f32(c.rms_eps),
            [t32, 1, 1],
        )
    };

    let qx = Quantized {
        q: a.xq.addr,
        d: a.xd.addr,
        s: a.xs.addr,
    };
    let cmds = &mut a.cmds;
    norm(cmds, l.attn_norm, a.x.addr, a.h.addr)?;
    if t == 1 {
        matvec(
            dev,
            cmds,
            &[
                (w(l.wq), a.q.addr),
                (w(l.wk), a.k_new.addr),
                (w(l.wv), a.v_new.addr),
            ],
            a.h.addr,
            Mode::Plain,
        )?;
    } else {
        let xq = quantize(dev, cmds, qx, a.h.addr, d, t32)?;
        matmul(dev, cmds, w(l.wq), xq, a.q.addr, t32, false)?;
        matmul(dev, cmds, w(l.wk), xq, a.k_new.addr, t32, false)?;
        matmul(dev, cmds, w(l.wv), xq, a.v_new.addr, t32, false)?;
    }
    let half = c.n_ctx * c.head_dim / 2 * 4;
    let pairs = t32 * (c.n_heads + 2 * c.n_kv_heads) as u32 * hd / 2;
    cmds.dispatch(
        "gpu_rope_store",
        p.rope_store,
        Push::default()
            .addr(a.q.addr)
            .addr(a.k_new.addr)
            .addr(a.v_new.addr)
            .addr(rope.addr)
            .addr(rope.addr + half as u64)
            .addr(a.k[li].addr)
            .addr(a.v[li].addr)
            .u32(t32)
            .u32(c.n_heads as u32)
            .u32(c.n_kv_heads as u32)
            .u32(hd)
            .u32(past as u32),
        [pairs.div_ceil(256), 1, 1],
    )?;
    let attn = Push::default()
        .addr(a.q.addr)
        .addr(a.k[li].addr)
        .addr(a.v[li].addr)
        .addr(if t == 1 { a.att_part.addr } else { a.att.addr })
        .u32(c.n_heads as u32)
        .u32(c.n_kv_heads as u32)
        .u32(hd)
        .u32(past as u32)
        .f32(1.0 / (c.head_dim as f32).sqrt());
    if t == 1 {
        let splits = (past + 1).div_ceil(SPLIT_KEYS) as u32;
        cmds.dispatch(
            "gpu_attention",
            p.attention_split,
            attn.u32(splits),
            [c.n_heads as u32, splits, 1],
        )?;
        cmds.dispatch(
            "gpu_attention",
            p.attention_merge,
            Push::default()
                .addr(a.att_part.addr)
                .addr(a.att.addr)
                .u32(hd)
                .u32(splits),
            [c.n_heads as u32, 1, 1],
        )?;
    } else {
        cmds.dispatch(
            "gpu_attention",
            p.flash_attn,
            attn.u32(t32),
            [c.n_heads as u32, t32.div_ceil(32), 1],
        )?;
    }
    if t == 1 {
        matvec(
            dev,
            cmds,
            &[(w(l.wo), a.x.addr)],
            a.att.addr,
            Mode::Accumulate,
        )?;
    } else {
        let xq = quantize(dev, cmds, qx, a.att.addr, d, t32)?;
        matmul(dev, cmds, w(l.wo), xq, a.x.addr, t32, true)?;
    }
    norm(cmds, l.ffn_norm, a.x.addr, a.h.addr)?;
    if t == 1 {
        matvec(
            dev,
            cmds,
            &[(w(l.gate), a.gate.addr), (w(l.up), 0)],
            a.h.addr,
            Mode::Glu,
        )?;
        matvec(
            dev,
            cmds,
            &[(w(l.down), a.x.addr)],
            a.gate.addr,
            Mode::Accumulate,
        )?;
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

/// How [`matvec`] combines its matrices.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    /// Each matrix writes its own output.
    Plain,
    /// One matrix, added to its output (residual).
    Accumulate,
    /// Gate and up: writes silu(gate) · up to the first output.
    Glu,
}

/// `y_i = W_i · x` for one row, for up to three matrices in one dispatch.
pub(super) fn matvec(dev: &Device, cmds: &mut Commands, mats: &[((u64, Linear), u64)], x: u64, mode: Mode) -> Result<()> {
    assert!((1..=3).contains(&mats.len()));
    let get = |i: usize| mats.get(i).copied();
    let k = mats[0].0.1.k;
    let mut push = Push::default();
    for i in 0..3 {
        push = push.addr(get(i).map_or(0, |((a, _), _)| a));
    }
    push = push.addr(x);
    for i in 0..3 {
        push = push.addr(get(i).map_or(0, |(_, y)| y));
    }
    push = push.u32(k);
    let mut q6k = 0;
    for i in 0..3 {
        let n = get(i).map_or(0, |((_, l), _)| {
            assert_eq!(l.k, k, "matrices in one matvec share k");
            l.n
        });
        if get(i).is_some_and(|((_, l), _)| l.kind == KQuant::Q6K) {
            q6k |= 1 << i;
        }
        push = push.u32(n);
    }
    let rows: u32 = mats.iter().map(|((_, l), _)| l.n).sum();
    let groups = if mode == Mode::Glu {
        mats[0].0.1.n
    } else {
        rows.div_ceil(2)
    };
    cmds.dispatch(
        "gpu_matvec",
        dev.pipes.matvec,
        push.u32(q6k)
            .u32((mode == Mode::Accumulate) as u32)
            .u32((mode == Mode::Glu) as u32),
        [groups, 1, 1],
    )
}

/// The quantized activations [`matmul`] reads: values, scales, sums of 16.
#[derive(Clone, Copy)]
pub(super) struct Quantized {
    pub(super) q: u64,
    pub(super) d: u64,
    pub(super) s: u64,
}

/// Quantizes `m` rows of `k` activations at `x` into `out`.
pub(super) fn quantize(
    dev: &Device,
    cmds: &mut Commands,
    out: Quantized,
    x: u64,
    k: u32,
    m: u32,
) -> Result<Quantized> {
    cmds.dispatch(
        "gpu_quantize",
        dev.pipes.quantize_q8k,
        Push::default()
            .addr(x)
            .addr(out.q)
            .addr(out.d)
            .addr(out.s)
            .u32(k),
        [k / 256, m, 1],
    )?;
    Ok(out)
}

/// `Y (+)= X · Wᵀ` for `m` rows (prefill), with integer dot products.
pub(super) fn matmul(
    dev: &Device,
    cmds: &mut Commands,
    (addr, w): (u64, Linear),
    x: Quantized,
    y: u64,
    m: u32,
    acc: bool,
) -> Result<()> {
    let p = &dev.pipes;
    let (label, pipe) = match w.kind {
        KQuant::Q4K => ("gpu_matmul_q4k", p.mmq_q4k),
        KQuant::Q6K => ("gpu_matmul_q6k", p.mmq_q6k),
    };
    let push = Push::default()
        .addr(addr)
        .addr(x.q)
        .addr(x.d)
        .addr(x.s)
        .addr(y);
    cmds.dispatch(
        label,
        pipe,
        push.u32(w.k).u32(w.n).u32(m).u32(acc as u32),
        [w.n.div_ceil(128), m.div_ceil(64), 1],
    )
}

#[cfg(test)]
mod tests {
    //! The linear-layer kernels against the CPU on random blocks (the encoders
    //! cover every scale and value range). Run on the first GPU in
    //! `INCREBOT_GPUS`; skipped when it is unset.

    // Test-only diagnostics (a mismatch's worst error, "no GPU, skipping").
    #![allow(clippy::print_stderr)]

    use super::*;
    use crate::gguf::kquant::{QK_K, decode_q4_k, decode_q6_k, encode};
    use crate::model::PackedKLinear;
    use crate::model::packed_q8::Kernel;

    fn device() -> Option<Arc<Device>> {
        let pci = std::env::var("INCREBOT_GPUS").ok()?;
        let pci = pci.split([',', '=']).next().filter(|p| !p.is_empty())?;
        Some(Device::open(pci.parse().unwrap()).unwrap())
    }

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn weights(kind: KQuant, rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let mut r = rng(seed);
        let mut out = Vec::new();
        for _ in 0..rows * blocks {
            match kind {
                KQuant::Q4K => {
                    let q: [u8; QK_K] = std::array::from_fn(|_| (r() % 16) as u8);
                    let sc = std::array::from_fn(|_| (r() % 64) as u8);
                    let mn = std::array::from_fn(|_| (r() % 64) as u8);
                    let (d, dmin) = ((r() % 1000) as f32 * 1e-5, (r() % 1000) as f32 * 1e-5);
                    out.extend(encode::q4_k(d, dmin, sc, mn, &q));
                }
                KQuant::Q6K => {
                    let q: [u8; QK_K] = std::array::from_fn(|_| (r() % 64) as u8);
                    let sc = std::array::from_fn(|_| (r() % 256) as u8 as i8);
                    out.extend(encode::q6_k((r() % 1000) as f32 * 1e-5, sc, &q));
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

    fn upload(
        dev: &Arc<Device>,
        kind: KQuant,
        blocks: &[u8],
        k: usize,
        n: usize,
    ) -> (Buffer, Linear) {
        let bytes = match kind {
            KQuant::Q4K => blocks.to_vec(),
            KQuant::Q6K => repack_q6k(blocks),
        };
        let buf = dev.vram(bytes.len() as u64).unwrap();
        buf.write(0, &bytes);
        let lin = Linear {
            off: 0,
            kind,
            k: k as u32,
            n: n as u32,
        };
        (buf, lin)
    }

    fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        eprintln!("{what}: worst error {:.2e} of scale {scale}", worst / scale);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * scale,
                "{what}[{i}]: {g} vs {w} (scale {scale})"
            );
        }
    }

    #[test]
    fn integer_matmul_matches_cpu_integer_kernel() {
        let Some(dev) = device() else {
            eprintln!("SKIPPED: INCREBOT_GPUS is not set");
            return;
        };
        let (k, n, m) = (3 * QK_K, 200, 70);
        for kind in [KQuant::Q4K, KQuant::Q6K] {
            let blocks = weights(kind, n, k / QK_K, 7);
            let x = activations(m * k, 9);
            let want =
                PackedKLinear::new(kind, &blocks, k, n).integer_with(&x, m, Kernel::Portable);
            let (w, lin) = upload(&dev, kind, &blocks, k, n);
            let xb = dev.vram((m * k * 4) as u64).unwrap();
            xb.write(0, &x);
            let (xq, xd, xs) = (
                dev.vram((m * k) as u64).unwrap(),
                dev.vram((m * k / 256 * 4) as u64).unwrap(),
                dev.vram((m * k / 16 * 4) as u64).unwrap(),
            );
            let qx = Quantized {
                q: xq.addr,
                d: xd.addr,
                s: xs.addr,
            };
            let y = dev.host((m * n * 4) as u64).unwrap();
            // Accumulate onto ones to check the residual path too.
            y.write(0, &vec![1f32; m * n]);
            let mut cmds = Commands::new(&dev, 1).unwrap();
            cmds.begin().unwrap();
            quantize(&dev, &mut cmds, qx, xb.addr, k as u32, m as u32).unwrap();
            matmul(&dev, &mut cmds, (w.addr, lin), qx, y.addr, m as u32, true).unwrap();
            cmds.submit().unwrap();
            cmds.wait_all().unwrap();
            let got: Vec<f32> = y.read::<f32>(0, m * n).iter().map(|v| v - 1.0).collect();
            // Same integer sums; floats differ in where large sums are rounded and
            // in the cancellation of the Q4_K min term.
            assert_close(&format!("{kind:?}"), &got, &want, 1e-3);
        }
    }

    #[test]
    fn matvec_modes_match_f32_reference() {
        let Some(dev) = device() else {
            eprintln!("SKIPPED: INCREBOT_GPUS is not set");
            return;
        };
        let k = 2 * QK_K;
        let x = activations(k, 3);
        let reference = |kind: KQuant, blocks: &[u8], n: usize| -> Vec<f32> {
            let w = match kind {
                KQuant::Q4K => decode_q4_k(blocks),
                KQuant::Q6K => decode_q6_k(blocks),
            };
            (0..n)
                .map(|r| w[r * k..][..k].iter().zip(&x).map(|(a, b)| a * b).sum())
                .collect()
        };
        let xb = dev.vram((k * 4) as u64).unwrap();
        xb.write(0, &x);
        // Three matrices of mixed kinds and odd row counts in one dispatch.
        let mats = [(KQuant::Q4K, 37), (KQuant::Q6K, 64), (KQuant::Q4K, 5)];
        let mut ups = Vec::new();
        let mut wants = Vec::new();
        for (i, &(kind, n)) in mats.iter().enumerate() {
            let blocks = weights(kind, n, k / QK_K, 11 + i as u64);
            wants.push(reference(kind, &blocks, n));
            ups.push((
                upload(&dev, kind, &blocks, k, n),
                dev.host((n * 4) as u64).unwrap(),
            ));
        }
        let mut cmds = Commands::new(&dev, 1).unwrap();
        cmds.begin().unwrap();
        let args: Vec<_> = ups
            .iter()
            .map(|((w, l), y)| ((w.addr, *l), y.addr))
            .collect();
        matvec(&dev, &mut cmds, &args, xb.addr, Mode::Plain).unwrap();
        cmds.submit().unwrap();
        cmds.wait_all().unwrap();
        for (i, ((_, l), y)) in ups.iter().enumerate() {
            assert_close(
                &format!("matrix {i}"),
                &y.read::<f32>(0, l.n as usize),
                &wants[i],
                1e-5,
            );
        }

        // Accumulate, and gate/up with SwiGLU.
        let (q6, n) = (KQuant::Q6K, 48);
        let gate = weights(KQuant::Q4K, n, k / QK_K, 21);
        let up = weights(q6, n, k / QK_K, 22);
        let (g, u) = (reference(KQuant::Q4K, &gate, n), reference(q6, &up, n));
        let gw = upload(&dev, KQuant::Q4K, &gate, k, n);
        let uw = upload(&dev, q6, &up, k, n);
        let y = dev.host((n * 4) as u64).unwrap();
        let acc = dev.host((n * 4) as u64).unwrap();
        acc.write(0, &vec![0.5f32; n]);
        let mut cmds = Commands::new(&dev, 1).unwrap();
        cmds.begin().unwrap();
        matvec(
            &dev,
            &mut cmds,
            &[((gw.0.addr, gw.1), y.addr), ((uw.0.addr, uw.1), 0)],
            xb.addr,
            Mode::Glu,
        )
        .unwrap();
        matvec(
            &dev,
            &mut cmds,
            &[((gw.0.addr, gw.1), acc.addr)],
            xb.addr,
            Mode::Accumulate,
        )
        .unwrap();
        cmds.submit().unwrap();
        cmds.wait_all().unwrap();
        let glu: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        assert_close("glu", &y.read::<f32>(0, n), &glu, 1e-5);
        let accd: Vec<f32> = g.iter().map(|v| v + 0.5).collect();
        assert_close("accumulate", &acc.read::<f32>(0, n), &accd, 1e-5);
    }
}
