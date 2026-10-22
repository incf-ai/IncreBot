//! LFM2 forward pass (LiquidAI's hybrid of gated short convolutions and GQA
//! attention; d1-3B is one). Linear layers go through [`QLinear`]; the rest runs
//! on the CPU kernels in `kernels.rs`.
//!
//! A layer is `x + mixer(norm(x))` then `x + swiglu_ffn(norm(x))`, where the mixer
//! is either a short conv ([`kernels::short_conv`]) or attention with per-head
//! RMSNorm on q and k and half-split RoPE ([`kernels::rope_neox`]).

use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::Device;
use tracing::info;

use super::kernels::{self, AttnShape};
use super::linear::QLinear;
use super::llama::{
    DENSE_MAX_PARAMS, KvCache, LinearImpl, PREFILL_CHUNK, PREFILL_POOL_MIN_TOKENS, load_linear,
    load_vector, rope_tables,
};
use super::profile::time;
use crate::gguf::{GgmlType, Gguf, QK_K, QK8_0, decode_q8_0, kquant};

/// Hyperparameters, all read from GGUF metadata.
#[derive(Clone, Debug)]
pub struct Lfm2Config {
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
    /// Taps of the short conv (`shortconv.l_cache`).
    pub conv_taps: usize,
    /// Per layer: true for attention, false for short conv.
    pub attention: Vec<bool>,
}

impl Lfm2Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        ensure!(arch == "lfm2", "unsupported architecture {arch:?} (need lfm2)");
        let n_layers = g.get_u64("lfm2.block_count")? as usize;
        let n_heads = g.get_u64("lfm2.attention.head_count")? as usize;
        let d_model = g.get_u64("lfm2.embedding_length")? as usize;
        // A per-layer array where 0 marks a conv layer.
        let kv = g
            .get("lfm2.attention.head_count_kv")?
            .as_array()
            .context("lfm2.attention.head_count_kv is not an array")?
            .iter()
            .map(|v| v.as_u64().context("lfm2.attention.head_count_kv entry"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            kv.len() == n_layers,
            "head_count_kv has {} entries for {n_layers} layers",
            kv.len()
        );
        let n_kv_heads = kv.iter().copied().max().unwrap_or(0) as usize;
        ensure!(
            n_kv_heads > 0 && kv.iter().all(|&k| k == 0 || k as usize == n_kv_heads),
            "head_count_kv {kv:?} must be 0 or one shared value"
        );
        ensure!(
            d_model.is_multiple_of(n_heads) && n_heads.is_multiple_of(n_kv_heads),
            "{d_model} / {n_heads} heads / {n_kv_heads} KV heads do not divide"
        );
        Ok(Self {
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim: d_model / n_heads,
            ffn_dim: g.get_u64("lfm2.feed_forward_length")? as usize,
            n_ctx: g.get_u64("lfm2.context_length")? as usize,
            vocab: g.get_u64("lfm2.vocab_size")? as usize,
            rope_theta: g.get_f32("lfm2.rope.freq_base")?,
            rms_eps: g.get_f32("lfm2.attention.layer_norm_rms_epsilon")?,
            conv_taps: g.get_u64("lfm2.shortconv.l_cache")? as usize,
            attention: kv.iter().map(|&k| k > 0).collect(),
        })
    }

    fn attn_shape(&self) -> AttnShape {
        AttnShape {
            heads: self.n_heads,
            kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

pub(crate) enum Mixer {
    Conv {
        in_proj: Box<dyn QLinear>,
        out_proj: Box<dyn QLinear>,
        /// `[d_model, taps]`, oldest tap first.
        kernel: Vec<f32>,
    },
    Attn {
        wq: Box<dyn QLinear>,
        wk: Box<dyn QLinear>,
        wv: Box<dyn QLinear>,
        wo: Box<dyn QLinear>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        /// Index of this layer's K/V in [`Lfm2State::kv`].
        slot: usize,
    },
}

pub(crate) struct Layer {
    pub(crate) op_norm: Vec<f32>,
    pub(crate) mixer: Mixer,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) w_gate: Box<dyn QLinear>,
    pub(crate) w_up: Box<dyn QLinear>,
    pub(crate) w_down: Box<dyn QLinear>,
}

/// The embedding table kept quantized; rows are decoded on lookup (the f32 table
/// of a 128k vocabulary would be 1 GB).
struct Embedding {
    bytes: Vec<u8>,
    ty: GgmlType,
    row_bytes: usize,
    width: usize,
}

impl Embedding {
    fn row(&self, id: usize, out: &mut Vec<f32>) -> Result<()> {
        let b = &self.bytes[id * self.row_bytes..][..self.row_bytes];
        match self.ty {
            GgmlType::F32 => out.extend(b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c))),
            GgmlType::F16 => out.extend(
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| half::f16::from_le_bytes(c).to_f32()),
            ),
            GgmlType::Q8_0 => out.extend(decode_q8_0(b)),
            GgmlType::Q4K => out.extend(kquant::decode_q4_k(b)),
            GgmlType::Q6K => out.extend(kquant::decode_q6_k(b)),
            GgmlType::Other(ty) => bail!("embedding type {ty:?} is not supported"),
        }
        Ok(())
    }
}

/// Everything a sequence carries between calls: the conv layers' last `taps - 1`
/// rows of `b · x` and the attention layers' K/V. Cheap enough to clone, which is
/// how one shared prefix serves several questions.
#[derive(Clone)]
pub struct Lfm2State {
    conv: Vec<Vec<f32>>,
    pub kv: KvCache,
}

impl Lfm2State {
    pub fn len(&self) -> usize {
        self.kv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }
}

pub struct Lfm2 {
    pub config: Lfm2Config,
    embed: Embedding,
    pub(crate) layers: Vec<Layer>,
    pub(crate) out_norm: Vec<f32>,
    pub(crate) lm_head: Box<dyn QLinear>,
    pub(crate) rope_cos: Vec<f32>,
    pub(crate) rope_sin: Vec<f32>,
}

impl Lfm2 {
    /// Loads the model; `max_ctx` caps the RoPE tables (and so the usable context)
    /// below the trained `context_length`.
    pub fn load(g: &Gguf, device: &Device, linear_impl: LinearImpl, max_ctx: usize) -> Result<Self> {
        let mut config = Lfm2Config::from_gguf(g)?;
        config.n_ctx = config.n_ctx.min(max_ctx);
        info!(?config, ?linear_impl, "loading lfm2");
        let start = Instant::now();
        if linear_impl == LinearImpl::Dense {
            let params: u64 = g.tensors.iter().map(|t| t.num_elements()).sum();
            ensure!(
                params <= DENSE_MAX_PARAMS,
                "the dense linear impl needs 4 bytes per weight; {params} weights is too many"
            );
        }
        let d = config.d_model;
        let linear = |name: &str| load_linear(g, name, linear_impl, device);
        let vector = |name: &str, len: usize| load_vector(g, name, len);

        let info = g.tensor("token_embd.weight")?;
        ensure!(
            info.shape() == [config.vocab, d],
            "token_embd.weight is {:?}, expected [{}, {d}]",
            info.shape(),
            config.vocab
        );
        let row_bytes = match info.ty {
            GgmlType::Q8_0 => d / QK8_0 * crate::gguf::Q8_0_BLOCK_BYTES,
            GgmlType::Q4K => d / QK_K * crate::gguf::Q4_K_BLOCK_BYTES,
            GgmlType::Q6K => d / QK_K * crate::gguf::Q6_K_BLOCK_BYTES,
            ty => ty
                .byte_size(d as u64)
                .with_context(|| format!("embedding type {ty:?} is not supported"))?
                as usize,
        };
        let embed = Embedding {
            bytes: g.bytes(info)?.to_vec(),
            ty: info.ty,
            row_bytes,
            width: d,
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        let mut slot = 0;
        for (i, &attn) in config.attention.iter().enumerate() {
            let p = format!("blk.{i}");
            let mixer = if attn {
                slot += 1;
                Mixer::Attn {
                    wq: linear(&format!("{p}.attn_q.weight"))?,
                    wk: linear(&format!("{p}.attn_k.weight"))?,
                    wv: linear(&format!("{p}.attn_v.weight"))?,
                    wo: linear(&format!("{p}.attn_output.weight"))?,
                    q_norm: vector(&format!("{p}.attn_q_norm.weight"), config.head_dim)?,
                    k_norm: vector(&format!("{p}.attn_k_norm.weight"), config.head_dim)?,
                    slot: slot - 1,
                }
            } else {
                Mixer::Conv {
                    in_proj: linear(&format!("{p}.shortconv.in_proj.weight"))?,
                    out_proj: linear(&format!("{p}.shortconv.out_proj.weight"))?,
                    kernel: vector(&format!("{p}.shortconv.conv.weight"), d * config.conv_taps)?,
                }
            };
            layers.push(Layer {
                op_norm: vector(&format!("{p}.attn_norm.weight"), d)?,
                mixer,
                ffn_norm: vector(&format!("{p}.ffn_norm.weight"), d)?,
                w_gate: linear(&format!("{p}.ffn_gate.weight"))?,
                w_up: linear(&format!("{p}.ffn_up.weight"))?,
                w_down: linear(&format!("{p}.ffn_down.weight"))?,
            });
        }

        // The LM head is tied to the embedding table.
        let lm_head = match g.tensor("output.weight") {
            Ok(_) => linear("output.weight")?,
            Err(_) => linear("token_embd.weight")?,
        };
        let (rope_cos, rope_sin) = rope_tables(config.head_dim, config.rope_theta, config.n_ctx);
        info!(elapsed_s = start.elapsed().as_secs_f32(), "model loaded");
        Ok(Self {
            out_norm: vector("token_embd_norm.weight", d)?,
            config,
            embed,
            layers,
            lm_head,
            rope_cos,
            rope_sin,
        })
    }

    /// The embedding rows of `tokens`, `[len, d_model]`.
    pub(crate) fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut x = Vec::with_capacity(tokens.len() * self.embed.width);
        for &id in tokens {
            ensure!((id as usize) < self.config.vocab, "token id {id} out of range");
            self.embed.row(id as usize, &mut x)?;
        }
        Ok(x)
    }

    pub fn new_state(&self) -> Lfm2State {
        let c = &self.config;
        let n_attn = c.attention.iter().filter(|&&a| a).count();
        Lfm2State {
            conv: c
                .attention
                .iter()
                .map(|&a| if a { Vec::new() } else { vec![0.0; (c.conv_taps - 1) * c.d_model] })
                .collect(),
            kv: KvCache::new(n_attn, c.n_kv_heads * c.head_dim, false),
        }
    }

    /// Runs `tokens` after whatever `state` holds and returns the logits of the
    /// last position. Only the last position goes through the LM head.
    pub fn forward(&self, tokens: &[u32], state: &mut Lfm2State) -> Result<Vec<f32>> {
        self.forward_spliced(tokens, state, None)
    }

    /// [`Self::forward`] with the embedding of every `image_token` in `tokens`
    /// replaced, in order, by the next row of `features` (`[n, d_model]`, one
    /// row per such token): the vision tower's output.
    pub fn forward_with_images(
        &self,
        tokens: &[u32],
        state: &mut Lfm2State,
        image_token: u32,
        features: &[f32],
    ) -> Result<Vec<f32>> {
        let d = self.config.d_model;
        let n = tokens.iter().filter(|&&t| t == image_token).count();
        ensure!(
            features.len() == n * d,
            "{n} image tokens but {} feature rows",
            features.len() as f64 / d as f64
        );
        self.forward_spliced(tokens, state, Some((image_token, features)))
    }

    fn forward_spliced(
        &self,
        tokens: &[u32],
        state: &mut Lfm2State,
        images: Option<(u32, &[f32])>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let past = state.len();
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
        state.kv.reserve(past + tokens.len());

        let mut run = || {
            let mut last = Vec::new();
            let mut used = 0;
            for chunk in tokens.chunks(PREFILL_CHUNK) {
                let splice = images.map(|(id, features)| {
                    let n = chunk.iter().filter(|&&t| t == id).count() * c.d_model;
                    used += n;
                    (id, &features[used - n..used])
                });
                last = self.forward_chunk(chunk, state, splice)?;
            }
            Ok::<_, anyhow::Error>(last)
        };
        let last = if tokens.len() >= PREFILL_POOL_MIN_TOKENS {
            crate::runtime::on_prefill_pool(run)?
        } else {
            run()?
        };

        let d = c.d_model;
        let h = time("norm", || {
            kernels::rms_norm(&last[last.len() - d..], d, &self.out_norm, c.rms_eps)
        });
        Ok(time("lm_head", || self.lm_head.forward(&h, 1)))
    }

    /// One chunk through every layer; returns the final hidden states `[t, d]`.
    fn forward_chunk(
        &self,
        tokens: &[u32],
        state: &mut Lfm2State,
        images: Option<(u32, &[f32])>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let (t, d) = (tokens.len(), c.d_model);
        let past = state.len();

        let mut x = self.embed(tokens)?;
        debug_assert_eq!(x.len(), t * d);
        if let Some((id, features)) = images {
            let mut rows = features.chunks_exact(d);
            for (row, _) in x.chunks_exact_mut(d).zip(tokens).filter(|(_, t)| **t == id) {
                row.copy_from_slice(rows.next().context("too few image feature rows")?);
            }
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let h = time("norm", || kernels::rms_norm(&x, d, &layer.op_norm, c.rms_eps));
            let o = match &layer.mixer {
                Mixer::Conv {
                    in_proj,
                    out_proj,
                    kernel,
                } => {
                    let bcx = time("linear_in", || in_proj.forward(&h, t));
                    let y = time("short_conv", || {
                        kernels::short_conv(&bcx, d, kernel, c.conv_taps, &mut state.conv[i])
                    });
                    time("linear_out", || out_proj.forward(&y, t))
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
                    let (q, k, v) = time("linear_qkv", || {
                        (wq.forward(&h, t), wk.forward(&h, t), wv.forward(&h, t))
                    });
                    let (mut q, mut k) = time("qk_norm", || {
                        (
                            kernels::rms_norm(&q, c.head_dim, q_norm, c.rms_eps),
                            kernels::rms_norm(&k, c.head_dim, k_norm, c.rms_eps),
                        )
                    });
                    time("rope", || {
                        let (cos, sin) = (&self.rope_cos, &self.rope_sin);
                        kernels::rope_neox(&mut q, c.n_heads, c.head_dim, past, cos, sin);
                        kernels::rope_neox(&mut k, c.n_kv_heads, c.head_dim, past, cos, sin);
                    });
                    time("kv_write", || state.kv.append(*slot, &k, &v));
                    let attn = time("attention", || {
                        state.kv.attention(*slot, &q, t, past, c.attn_shape())
                    });
                    time("linear_o", || wo.forward(&attn, t))
                }
            };
            time("residual", || kernels::add_assign(&mut x, &o));

            let h = time("norm", || kernels::rms_norm(&x, d, &layer.ffn_norm, c.rms_eps));
            let (gate, up) = time("linear_gate_up", || {
                (layer.w_gate.forward(&h, t), layer.w_up.forward(&h, t))
            });
            let act = time("swiglu", || kernels::swiglu(&gate, &up));
            let down = time("linear_down", || layer.w_down.forward(&act, t));
            time("residual", || kernels::add_assign(&mut x, &down));
        }
        state.kv.tokens.extend_from_slice(tokens);
        Ok(x)
    }
}
