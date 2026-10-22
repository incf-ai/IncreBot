//! Llama forward pass (design.md §5). Linear layers go through [`QLinear`]; the
//! rest runs on the CPU kernels in `kernels.rs`.

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use tracing::info;

use half::f16;
use half::slice::HalfFloatSliceExt;

use super::kernels::{self, AttnShape};
use super::linear::{BurnQLinear, DenseLinear, QLinear};
use super::packed_k::{KQuant, PackedKLinear};
use super::packed_q8::PackedQ8Linear;
use super::profile::time;
use crate::gguf::{GgmlType, Gguf, q8_0_tensor};

/// Hyperparameters, all read from GGUF metadata.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub n_ctx: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl LlamaConfig {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        ensure!(
            arch == "llama",
            "unsupported architecture {arch:?} (need llama)"
        );
        let n_heads = g.get_u64("llama.attention.head_count")? as usize;
        let d_model = g.get_u64("llama.embedding_length")? as usize;
        let n_kv_heads = match g.metadata.get("llama.attention.head_count_kv") {
            Some(v) => v.as_u64().context("llama.attention.head_count_kv")? as usize,
            None => n_heads,
        };
        ensure!(
            d_model.is_multiple_of(n_heads),
            "embedding_length {d_model} not divisible by {n_heads} heads"
        );
        ensure!(
            n_heads.is_multiple_of(n_kv_heads),
            "{n_heads} heads not divisible by {n_kv_heads} KV heads"
        );
        let head_dim = d_model / n_heads;
        if let Some(v) = g.metadata.get("llama.rope.dimension_count") {
            let rope_dim = v.as_u64().context("llama.rope.dimension_count")? as usize;
            ensure!(
                rope_dim == head_dim,
                "partial RoPE ({rope_dim} of {head_dim}) is not supported"
            );
        }
        let vocab = g
            .get("tokenizer.ggml.tokens")?
            .as_array()
            .context("tokenizer.ggml.tokens is not an array")?
            .len();
        Ok(Self {
            n_layers: g.get_u64("llama.block_count")? as usize,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim: g.get_u64("llama.feed_forward_length")? as usize,
            n_ctx: g.get_u64("llama.context_length")? as usize,
            vocab,
            rope_theta: match g.metadata.get("llama.rope.freq_base") {
                Some(v) => v.as_f32().context("llama.rope.freq_base")?,
                None => 10_000.0,
            },
            rms_eps: g.get_f32("llama.attention.layer_norm_rms_epsilon")?,
        })
    }
}

/// Which [`QLinear`] implementation to load weights into (design.md §3).
/// `Packed` in `config.rsn`; the lowercase aliases are for env overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearImpl {
    /// Burn quantized weights, dequantized per call. The reference.
    #[serde(alias = "burn")]
    Burn,
    /// Dequantized once at load. Same numerics as `Burn`, much faster, 4 B/weight.
    #[serde(alias = "dense")]
    Dense,
    /// The GGUF blocks as-is (Q8_0, Q4_K or Q6_K), with hand-written kernels.
    /// How the bot runs the 13B.
    #[serde(alias = "packed")]
    Packed,
}

/// `Dense` keeps every weight as f32; refuse it for anything but small models.
pub(crate) const DENSE_MAX_PARAMS: u64 = 2_000_000_000;

/// Prefill runs in chunks of this many tokens, where `gemm` is efficient and the
/// activations stay small (design.md §3).
pub(crate) const PREFILL_CHUNK: usize = 512;

/// Prefills of at least this many tokens run on the wider prefill pool, if any
/// (design.md §4). Below it the 22-thread pool measured no faster (`core_scaling`).
pub(crate) const PREFILL_POOL_MIN_TOKENS: usize = 32;

/// The KV cache grows in steps of this many positions (design.md §5).
const KV_CHUNK: usize = 256;

pub(crate) struct Layer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) wq: Box<dyn QLinear>,
    pub(crate) wk: Box<dyn QLinear>,
    pub(crate) wv: Box<dyn QLinear>,
    pub(crate) wo: Box<dyn QLinear>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) w_gate: Box<dyn QLinear>,
    pub(crate) w_up: Box<dyn QLinear>,
    pub(crate) w_down: Box<dyn QLinear>,
}

/// Per-layer K and V, position-major `[len, kv_heads · head_dim]`, in f32 or f16
/// (design.md §3, §5). Appending is `extend_from_slice`, truncating is
/// `Vec::truncate`, and capacity grows in chunks of [`KV_CHUNK`] positions.
/// `tokens` records what the cache holds, for prefix reuse.
#[derive(Clone)]
pub struct KvCache {
    store: KvStore,
    pub tokens: Vec<u32>,
    kv_width: usize,
}

#[derive(Clone)]
enum KvStore {
    F32 { k: Vec<Vec<f32>>, v: Vec<Vec<f32>> },
    F16 { k: Vec<Vec<f16>>, v: Vec<Vec<f16>> },
}

fn grow<T>(bufs: &mut [Vec<T>], have: usize, want: usize) {
    for b in bufs {
        if b.capacity() < have {
            b.reserve_exact(want - b.len());
        }
    }
}

impl KvCache {
    pub(crate) fn new(layers: usize, kv_width: usize, f16: bool) -> Self {
        let store = if f16 {
            KvStore::F16 {
                k: vec![Vec::new(); layers],
                v: vec![Vec::new(); layers],
            }
        } else {
            KvStore::F32 {
                k: vec![Vec::new(); layers],
                v: vec![Vec::new(); layers],
            }
        };
        Self {
            store,
            tokens: Vec::new(),
            kv_width,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn is_f16(&self) -> bool {
        matches!(self.store, KvStore::F16 { .. })
    }

    /// Keeps only the first `n` positions; with `n` = `len()`, drops K/V rows
    /// a cancelled chunk wrote past them.
    pub fn truncate(&mut self, n: usize) {
        if n > self.len() {
            return;
        }
        self.tokens.truncate(n);
        let len = n * self.kv_width;
        match &mut self.store {
            KvStore::F32 { k, v } => k
                .iter_mut()
                .chain(v.iter_mut())
                .for_each(|b| b.truncate(len)),
            KvStore::F16 { k, v } => k
                .iter_mut()
                .chain(v.iter_mut())
                .for_each(|b| b.truncate(len)),
        }
    }

    /// Bytes held (capacity), for memory accounting.
    pub fn bytes(&self) -> usize {
        match &self.store {
            KvStore::F32 { k, v } => k.iter().chain(v).map(|b| b.capacity() * 4).sum(),
            KvStore::F16 { k, v } => k.iter().chain(v).map(|b| b.capacity() * 2).sum(),
        }
    }

    pub(crate) fn reserve(&mut self, positions: usize) {
        let have = positions * self.kv_width;
        let want = positions.next_multiple_of(KV_CHUNK) * self.kv_width;
        match &mut self.store {
            KvStore::F32 { k, v } => {
                grow(k, have, want);
                grow(v, have, want);
            }
            KvStore::F16 { k, v } => {
                grow(k, have, want);
                grow(v, have, want);
            }
        }
    }

    pub(crate) fn append(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        match &mut self.store {
            KvStore::F32 { k, v } => {
                k[layer].extend_from_slice(k_new);
                v[layer].extend_from_slice(v_new);
            }
            KvStore::F16 { k, v } => {
                k[layer].extend(kernels::to_f16_vec(k_new));
                v[layer].extend(kernels::to_f16_vec(v_new));
            }
        }
    }

    pub(crate) fn attention(
        &self,
        layer: usize,
        q: &[f32],
        t: usize,
        past: usize,
        shape: AttnShape,
    ) -> Vec<f32> {
        match &self.store {
            KvStore::F32 { k, v } => kernels::attention(q, &k[layer], &v[layer], t, past, shape),
            KvStore::F16 { k, v } => kernels::attention(q, &k[layer], &v[layer], t, past, shape),
        }
    }

    /// Width of one position's K (or V) row: `kv_heads · head_dim`.
    pub(crate) fn kv_width(&self) -> usize {
        self.kv_width
    }

    /// Layer `layer`'s K and V rows from position `from` on, if the cache is f16
    /// (the GPU backend's format).
    pub(crate) fn f16_rows(&self, layer: usize, from: usize) -> Option<(&[f16], &[f16])> {
        match &self.store {
            KvStore::F16 { k, v } => Some((
                &k[layer][from * self.kv_width..],
                &v[layer][from * self.kv_width..],
            )),
            KvStore::F32 { .. } => None,
        }
    }

    /// Appends f16 rows computed elsewhere (the GPU) to layer `layer`. The caller
    /// extends `tokens` once every layer has its rows.
    pub(crate) fn append_f16(&mut self, layer: usize, k_new: &[f16], v_new: &[f16]) {
        match &mut self.store {
            KvStore::F16 { k, v } => {
                k[layer].extend_from_slice(k_new);
                v[layer].extend_from_slice(v_new);
            }
            KvStore::F32 { k, v } => {
                k[layer].extend(k_new.iter().map(|x| x.to_f32()));
                v[layer].extend(v_new.iter().map(|x| x.to_f32()));
            }
        }
    }

    /// Reserves room for `positions` positions (rounded up to the growth chunk).
    pub(crate) fn reserve_positions(&mut self, positions: usize) {
        self.reserve(positions);
    }

    /// The f16 form for parking in the LRU (design.md §3): a plain copy if the cache
    /// is already f16, lossy (f32 → f16) otherwise.
    pub fn park(&self) -> ParkedKv {
        let to16 = |bufs: &Vec<Vec<f32>>| bufs.iter().map(|b| kernels::to_f16_vec(b)).collect();
        let (k, v, was_f32) = match &self.store {
            KvStore::F16 { k, v } => (k.clone(), v.clone(), false),
            KvStore::F32 { k, v } => (to16(k), to16(v), true),
        };
        ParkedKv {
            k,
            v,
            tokens: self.tokens.clone(),
            kv_width: self.kv_width,
            was_f32,
        }
    }
}

/// A KV cache stored as f16 while it is not the active one.
pub struct ParkedKv {
    k: Vec<Vec<f16>>,
    v: Vec<Vec<f16>>,
    pub tokens: Vec<u32>,
    kv_width: usize,
    was_f32: bool,
}

impl ParkedKv {
    /// Back to an active cache of the precision it was parked from.
    pub fn restore(&self) -> KvCache {
        let store = if self.was_f32 {
            let to32 = |bufs: &Vec<Vec<f16>>| bufs.iter().map(|b| b.to_f32_vec()).collect();
            KvStore::F32 {
                k: to32(&self.k),
                v: to32(&self.v),
            }
        } else {
            KvStore::F16 {
                k: self.k.clone(),
                v: self.v.clone(),
            }
        };
        KvCache {
            store,
            tokens: self.tokens.clone(),
            kv_width: self.kv_width,
        }
    }

    pub fn bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(|b| b.len() * 2).sum()
    }
}

pub struct Llama {
    pub config: LlamaConfig,
    /// `[vocab, d_model]`, decoded to f32 once (design.md §2).
    pub(crate) tok_embd: Vec<f32>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) out_norm: Vec<f32>,
    pub(crate) lm_head: Box<dyn QLinear>,
    /// `[n_ctx, head_dim / 2]`.
    pub(crate) rope_cos: Vec<f32>,
    pub(crate) rope_sin: Vec<f32>,
    kv_f16: bool,
}

impl Llama {
    pub fn load(g: &Gguf, device: &Device, linear_impl: LinearImpl) -> Result<Self> {
        let config = LlamaConfig::from_gguf(g)?;
        info!(?config, ?linear_impl, "loading llama");
        let start = Instant::now();
        if linear_impl == LinearImpl::Dense {
            let params: u64 = g.tensors.iter().map(|t| t.num_elements()).sum();
            ensure!(
                params <= DENSE_MAX_PARAMS,
                "the dense linear impl needs 4 bytes per weight; {params} weights is too many"
            );
        }

        let linear = |name: &str| load_linear(g, name, linear_impl, device);
        let vector = |name: &str, len: usize| load_vector(g, name, len);

        let embd_info = g.tensor("token_embd.weight")?;
        let tok_embd = vector("token_embd.weight", config.vocab * config.d_model)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let p = format!("blk.{i}");
            layers.push(Layer {
                attn_norm: vector(&format!("{p}.attn_norm.weight"), config.d_model)?,
                wq: linear(&format!("{p}.attn_q.weight"))?,
                wk: linear(&format!("{p}.attn_k.weight"))?,
                wv: linear(&format!("{p}.attn_v.weight"))?,
                wo: linear(&format!("{p}.attn_output.weight"))?,
                ffn_norm: vector(&format!("{p}.ffn_norm.weight"), config.d_model)?,
                w_gate: linear(&format!("{p}.ffn_gate.weight"))?,
                w_up: linear(&format!("{p}.ffn_up.weight"))?,
                w_down: linear(&format!("{p}.ffn_down.weight"))?,
            });
            info!(
                layer = i,
                of = config.n_layers,
                elapsed_s = start.elapsed().as_secs_f32(),
                "loaded layer"
            );
        }

        // Some files tie the LM head to the embedding table.
        let lm_head = match g.tensor("output.weight") {
            Ok(_) => linear("output.weight")?,
            Err(_) if matches!(embd_info.ty, GgmlType::Q8_0 | GgmlType::Q4K | GgmlType::Q6K) => {
                linear("token_embd.weight")?
            }
            Err(e) => return Err(e),
        };

        let (rope_cos, rope_sin) = rope_tables(config.head_dim, config.rope_theta, config.n_ctx);
        info!(elapsed_s = start.elapsed().as_secs_f32(), "model loaded");
        Ok(Self {
            out_norm: vector("output_norm.weight", config.d_model)?,
            config,
            tok_embd,
            layers,
            lm_head,
            rope_cos,
            rope_sin,
            kv_f16: false,
        })
    }

    /// Whether new caches are f16 ([`set_kv_f16`](Self::set_kv_f16)).
    pub fn kv_f16(&self) -> bool {
        self.kv_f16
    }

    /// The embedding rows of `tokens`, `[t, d_model]`.
    pub(crate) fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.config.d_model;
        let mut x = Vec::with_capacity(tokens.len() * d);
        for &id in tokens {
            x.extend_from_slice(&self.tok_embd[id as usize * d..][..d]);
        }
        x
    }

    /// A new empty cache, f32 or f16 per [`set_kv_f16`](Self::set_kv_f16).
    pub fn new_cache(&self) -> KvCache {
        let c = &self.config;
        KvCache::new(c.n_layers, c.n_kv_heads * c.head_dim, self.kv_f16)
    }

    /// Store new caches in f16: half the memory and half the bytes attention reads,
    /// at a small precision cost (llama.cpp's default). Off by default, so the
    /// reference test compares against its f32-cache recording.
    pub fn set_kv_f16(&mut self, on: bool) {
        self.kv_f16 = on;
    }

    /// Runs `tokens` after whatever `cache` holds and returns the logits of the last
    /// position. Only the last position goes through the LM head.
    pub fn forward(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        match self.forward_cancellable(tokens, cache, &|| false)? {
            Some(logits) => Ok(logits),
            None => bail!("forward_cancellable returned None despite an always-false cancel closure"),
        }
    }

    /// Like [`forward`](Self::forward), but checks `cancelled` between layers (a cold
    /// prefill can take ~30 s, design.md §5). On cancel the cache is rolled back to
    /// what it held before the call and `None` is returned.
    pub fn forward_cancellable(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<Vec<f32>>> {
        let c = &self.config;
        let past = cache.len();
        ensure!(!tokens.is_empty(), "forward called with no tokens");
        ensure!(
            past + tokens.len() <= c.n_ctx,
            "context overflow: {past} + {} > {}",
            tokens.len(),
            c.n_ctx
        );
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= c.vocab) {
            bail!("token id {bad} out of range (vocab {})", c.vocab);
        }
        cache.reserve(past + tokens.len());

        // Decode and short prefills stay on the global pool; long prefills move
        // wholesale to the wider prefill pool, so the two pools never alternate.
        let mut run = || {
            let mut last = Vec::new();
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                last = self.forward_chunk(chunk, cache, cancelled)?;
            }
            Some(last)
        };
        let out = if tokens.len() >= PREFILL_POOL_MIN_TOKENS {
            crate::runtime::on_prefill_pool(run)
        } else {
            run()
        };
        // Cancelled: keep the chunks that finished (a prefill ahead cut short
        // by a reply still saves that reply their time).
        let Some(last) = out else {
            cache.truncate(cache.len());
            return Ok(None);
        };

        let d = c.d_model;
        let last_row = &last[last.len() - d..];
        let h = time("norm", || {
            kernels::rms_norm(last_row, d, &self.out_norm, c.rms_eps)
        });
        Ok(Some(time("lm_head", || self.lm_head.forward(&h, 1))))
    }

    /// One chunk through every layer; returns the final hidden states `[t, d]`, or
    /// `None` if cancelled (the caller rolls the cache back).
    fn forward_chunk(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Option<Vec<f32>> {
        let c = &self.config;
        let (t, d) = (tokens.len(), c.d_model);
        let past = cache.len();
        let shape = AttnShape {
            heads: c.n_heads,
            kv_heads: c.n_kv_heads,
            head_dim: c.head_dim,
        };

        let mut x = self.embed(tokens);

        for (i, layer) in self.layers.iter().enumerate() {
            if cancelled() {
                return None;
            }
            // Attention block.
            let h = time("norm", || {
                kernels::rms_norm(&x, d, &layer.attn_norm, c.rms_eps)
            });
            let (mut q, mut k, v) = time("linear_qkv", || {
                (
                    layer.wq.forward(&h, t),
                    layer.wk.forward(&h, t),
                    layer.wv.forward(&h, t),
                )
            });
            time("rope", || {
                kernels::rope(
                    &mut q,
                    c.n_heads,
                    c.head_dim,
                    past,
                    &self.rope_cos,
                    &self.rope_sin,
                );
                kernels::rope(
                    &mut k,
                    c.n_kv_heads,
                    c.head_dim,
                    past,
                    &self.rope_cos,
                    &self.rope_sin,
                );
            });
            time("kv_write", || cache.append(i, &k, &v));
            let attn = time("attention", || cache.attention(i, &q, t, past, shape));
            let o = time("linear_o", || layer.wo.forward(&attn, t));
            time("residual", || kernels::add_assign(&mut x, &o));

            // Feed-forward block (SwiGLU).
            let h = time("norm", || {
                kernels::rms_norm(&x, d, &layer.ffn_norm, c.rms_eps)
            });
            let (gate, up) = time("linear_gate_up", || {
                (layer.w_gate.forward(&h, t), layer.w_up.forward(&h, t))
            });
            let act = time("swiglu", || kernels::swiglu(&gate, &up));
            let down = time("linear_down", || layer.w_down.forward(&act, t));
            time("residual", || kernels::add_assign(&mut x, &down));
        }
        cache.tokens.extend_from_slice(tokens);
        Some(x)
    }
}

/// Loads a 2-D weight into the [`QLinear`] `linear_impl` asks for.
pub(crate) fn load_linear(
    g: &Gguf,
    name: &str,
    linear_impl: LinearImpl,
    device: &Device,
) -> Result<Box<dyn QLinear>> {
    let t = g.tensor(name)?;
    let [out_f, in_f] = t.shape()[..] else {
        bail!("tensor {name} is not 2-D")
    };
    if linear_impl == LinearImpl::Packed {
        let bytes = g.bytes(t)?;
        return Ok(match t.ty {
            GgmlType::Q8_0 => Box::new(PackedQ8Linear::new(bytes, in_f, out_f)),
            GgmlType::Q4K => Box::new(PackedKLinear::new(KQuant::Q4K, bytes, in_f, out_f)),
            GgmlType::Q6K => Box::new(PackedKLinear::new(KQuant::Q6K, bytes, in_f, out_f)),
            other => bail!("tensor {name} is {other:?}; packed needs Q8_0, Q4_K or Q6_K"),
        });
    }
    if linear_impl == LinearImpl::Dense && t.ty != GgmlType::Q8_0 {
        // K-quants have no Burn scheme; decode them straight to f32.
        return Ok(Box::new(DenseLinear::from_f32(
            g.to_f32(t)?,
            out_f,
            in_f,
            device,
        )));
    }
    let w = q8_0_tensor(g, t, device).with_context(|| format!("loading {name}"))?;
    Ok(match linear_impl {
        LinearImpl::Burn => Box::new(BurnQLinear::new(w)),
        LinearImpl::Dense => Box::new(DenseLinear::from_quantized(w)),
        // Packed already returned above; a bug there should fail this
        // tensor's load, not panic the whole process.
        LinearImpl::Packed => bail!("linear_impl::Packed reached the dense/burn branch"),
    })
}

/// Loads a tensor of `len` elements as f32.
pub(crate) fn load_vector(g: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
    let v = g.to_f32(g.tensor(name)?)?;
    ensure!(
        v.len() == len,
        "tensor {name} has {} elements, expected {len}",
        v.len()
    );
    Ok(v)
}

/// RoPE tables for the *interleaved* convention: element pair `(2i, 2i + 1)` is
/// rotated by `pos · θ^(-2i / head_dim)`. GGUF `llama` weights need this one; HF's
/// half-split `rotate_half` gives plausible garbage (design.md §5).
/// The same tables serve the half-split convention ([`kernels::rope_neox`]).
pub(crate) fn rope_tables(head_dim: usize, theta: f32, n_ctx: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(n_ctx * half);
    let mut sin = Vec::with_capacity(n_ctx * half);
    for pos in 0..n_ctx {
        for i in 0..half {
            let freq = (theta as f64).powf(-2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}
