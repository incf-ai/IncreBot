//! EmbeddingGemma 2's text tower (`language_model`): scaled token embeddings
//! (with picture soft tokens spliced in), per-layer inputs projected from them,
//! 24 bidirectional layers (sliding window on most, full attention every
//! sixth), the final norm and the projection to the embedding width per token.

use std::sync::Arc;

use anyhow::{Result, ensure};
use rayon::prelude::*;

use super::ops::{Linear, Mlp, RmsNorm, add, attention, gelu_tanh, inv_freq, rope};
use super::safetensors::{SafeTensors, bf16};

pub const HIDDEN: usize = 512;
pub const LAYERS: usize = 24;
/// The output embedding's width (`embedding_dim`).
pub const EMBED_DIM: usize = 768;
pub const VOCAB: usize = 262_144;
const HEADS: usize = 4;
const INTERMEDIATE: usize = 2048;
/// `hidden_size_per_layer_input`.
const PER_LAYER: usize = 512;
/// Sliding layers attend to keys at most this far from the query, either way.
const WINDOW: usize = 512;
const PREFIX: &str = "language_model";

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
    ple_gate: Linear,
    ple_proj: Linear,
    ple_norm: RmsNorm,
    scalar: f32,
    head_dim: usize,
    kv_heads: usize,
    /// Full attention (every sixth layer) instead of the sliding window.
    full: bool,
}

pub struct Text {
    /// Holds `embed_tokens` (`[VOCAB, HIDDEN]`, bf16), read row by row.
    st: Arc<SafeTensors>,
    ple_proj: Linear,
    ple_norm: RmsNorm,
    layers: Vec<Layer>,
    norm: RmsNorm,
    projection: Linear,
    sliding_freq: Vec<f32>,
    full_freq: Vec<f32>,
}

impl Text {
    pub fn load(st: Arc<SafeTensors>) -> Result<Self> {
        let embed = st.info(&format!("{PREFIX}.embed_tokens.weight"))?;
        ensure!(embed.shape == [VOCAB, HIDDEN], "embed_tokens {:?}", embed.shape);
        let layers = (0..LAYERS)
            .map(|i| {
                let p = format!("{PREFIX}.layers.{i}");
                let full = i % 6 == 5;
                let (head_dim, kv_heads) = if full { (512, 1) } else { (256, 2) };
                let lin = |name: &str, out, inp| Linear::load(&st, &format!("{p}.{name}"), out, inp);
                let norm = |name: &str, dim| RmsNorm::load(&st, &format!("{p}.{name}"), dim);
                Ok(Layer {
                    input_norm: norm("input_layernorm", HIDDEN)?,
                    q: lin("self_attn.q_proj", HEADS * head_dim, HIDDEN)?,
                    k: lin("self_attn.k_proj", kv_heads * head_dim, HIDDEN)?,
                    v: lin("self_attn.v_proj", kv_heads * head_dim, HIDDEN)?,
                    o: lin("self_attn.o_proj", HIDDEN, HEADS * head_dim)?,
                    q_norm: norm("self_attn.q_norm", head_dim)?,
                    k_norm: norm("self_attn.k_norm", head_dim)?,
                    v_norm: RmsNorm::plain(head_dim),
                    post_attn_norm: norm("post_attention_layernorm", HIDDEN)?,
                    pre_ff_norm: norm("pre_feedforward_layernorm", HIDDEN)?,
                    mlp: Mlp {
                        gate: lin("mlp.gate_proj", INTERMEDIATE, HIDDEN)?,
                        up: lin("mlp.up_proj", INTERMEDIATE, HIDDEN)?,
                        down: lin("mlp.down_proj", HIDDEN, INTERMEDIATE)?,
                    },
                    post_ff_norm: norm("post_feedforward_layernorm", HIDDEN)?,
                    ple_gate: lin("ple_block.per_layer_input_gate", PER_LAYER, HIDDEN)?,
                    ple_proj: lin("ple_block.per_layer_projection", HIDDEN, PER_LAYER)?,
                    ple_norm: norm("ple_block.post_per_layer_input_norm", HIDDEN)?,
                    scalar: st.f32(&format!("{p}.layer_scalar"), &[1])?[0],
                    head_dim,
                    kv_heads,
                    full,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            ple_proj: Linear::load(
                &st,
                &format!("{PREFIX}.ple.per_layer_model_projection"),
                LAYERS * PER_LAYER,
                HIDDEN,
            )?,
            ple_norm: RmsNorm::load(&st, &format!("{PREFIX}.ple.per_layer_projection_norm"), PER_LAYER)?,
            layers,
            norm: RmsNorm::load(&st, &format!("{PREFIX}.norm"), HIDDEN)?,
            projection: Linear::load(&st, &format!("{PREFIX}.embedding_projection"), EMBED_DIM, HIDDEN)?,
            sliding_freq: inv_freq(10_000.0, 256),
            full_freq: inv_freq(1_000_000.0, 512),
            st,
        })
    }

    /// The scaled embeddings of `ids`, `[ids.len(), HIDDEN]`.
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let table = self.st.bytes(&format!("{PREFIX}.embed_tokens.weight"))?;
        let scale = 512f64.sqrt() as f32;
        let mut x = vec![0f32; ids.len() * HIDDEN];
        for (row, &id) in x.chunks_exact_mut(HIDDEN).zip(ids) {
            let id = id as usize;
            ensure!(id < VOCAB, "token {id} past the vocabulary");
            let src = &table[id * HIDDEN * 2..(id + 1) * HIDDEN * 2];
            for (v, b) in row.iter_mut().zip(src.as_chunks::<2>().0) {
                *v = bf16(u16::from_le_bytes(*b)) * scale;
            }
        }
        Ok(x)
    }

    /// The tower on input embeddings `x` (`n` rows): the mean of the projected tokens,
    /// L2-normalised, `EMBED_DIM` wide.
    pub fn encode(&self, mut x: Vec<f32>, n: usize) -> Vec<f32> {
        assert_eq!(x.len(), n * HIDDEN, "input embeddings");
        let mut per_layer = self.ple_proj.forward(&x, n);
        let scale = (HIDDEN as f64).powf(-0.5) as f32;
        per_layer.par_iter_mut().for_each(|v| *v *= scale);
        self.ple_norm.apply(&mut per_layer);
        for (i, l) in self.layers.iter().enumerate() {
            self.layer(l, i, &mut x, &per_layer, n);
        }
        self.norm.apply(&mut x);
        let y = self.projection.forward(&x, n);
        let mut mean = vec![0f64; EMBED_DIM];
        for row in y.chunks_exact(EMBED_DIM) {
            mean.iter_mut().zip(row).for_each(|(m, &v)| *m += f64::from(v));
        }
        normalize(&mean)
    }

    fn layer(&self, l: &Layer, i: usize, x: &mut [f32], per_layer: &[f32], n: usize) {
        let (dh, kv) = (l.head_dim, l.kv_heads);
        let h = l.input_norm.forward(x);
        let (mut q, mut k, mut v) = (l.q.forward(&h, n), l.k.forward(&h, n), l.v.forward(&h, n));
        l.q_norm.apply(&mut q);
        l.k_norm.apply(&mut k);
        l.v_norm.apply(&mut v);
        let freq = if l.full { &self.full_freq } else { &self.sliding_freq };
        let angles = |r: usize| freq.iter().map(|f| r as f32 * f).collect();
        rope(&mut q, HEADS * dh, dh, 1, angles);
        rope(&mut k, kv * dh, dh, 1, angles);
        let window = (!l.full).then_some(WINDOW);
        let a = attention(&q, &k, &v, n, (HEADS, kv, dh), window);
        let mut o = l.o.forward(&a, n);
        l.post_attn_norm.apply(&mut o);
        add(x, &o);

        let h = l.pre_ff_norm.forward(x);
        let mut m = l.mlp.forward(&h, n);
        l.post_ff_norm.apply(&mut m);
        add(x, &m);

        let mut g = l.ple_gate.forward(x, n);
        g.par_chunks_mut(PER_LAYER).enumerate().for_each(|(r, row)| {
            let p = &per_layer[(r * LAYERS + i) * PER_LAYER..(r * LAYERS + i + 1) * PER_LAYER];
            row.iter_mut().zip(p).for_each(|(g, p)| *g = gelu_tanh(*g) * p);
        });
        let mut p = l.ple_proj.forward(&g, n);
        l.ple_norm.apply(&mut p);
        add(x, &p);
        x.par_iter_mut().for_each(|v| *v *= l.scalar);
    }
}

/// `v / ‖v‖` in f32.
fn normalize(v: &[f64]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    v.iter().map(|x| (x / norm) as f32).collect()
}
