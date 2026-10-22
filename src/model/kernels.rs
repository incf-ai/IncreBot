//! CPU kernels for everything in the forward pass except the linear layers.
//!
//! These replaced Burn tensor ops: measured on the 13B, burn-flex ran attention,
//! softmax, RoPE and SwiGLU single-threaded (18 s of a 68 s, 1,000-token prefill
//! was attention alone), and its in-place `slice_assign` into the KV cache copied
//! the whole buffer on every decode step. Activations are row-major `[t, width]`
//! f32 slices.

use rayon::prelude::*;

/// `x[i] / rms(row) · w[i]` for each row of width `width`.
pub fn rms_norm(x: &[f32], width: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(width)
        .zip(x.par_chunks(width))
        .for_each(|(o, row)| {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for ((o, &v), &w) in o.iter_mut().zip(row).zip(weight) {
                *o = v * inv * w;
            }
        });
    out
}

/// Interleaved RoPE in place: within each head, pair `(2i, 2i + 1)` of the token at
/// position `pos0 + row` is rotated by angle `pos · θ_i`. `cos`/`sin` are
/// `[n_ctx, head_dim / 2]`.
pub fn rope(x: &mut [f32], heads: usize, head_dim: usize, pos0: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    x.par_chunks_mut(heads * head_dim)
        .enumerate()
        .for_each(|(row, tok)| {
            let c = &cos[(pos0 + row) * half..][..half];
            let s = &sin[(pos0 + row) * half..][..half];
            for head in tok.chunks_exact_mut(head_dim) {
                for i in 0..half {
                    let (a, b) = (head[2 * i], head[2 * i + 1]);
                    head[2 * i] = a * c[i] - b * s[i];
                    head[2 * i + 1] = a * s[i] + b * c[i];
                }
            }
        });
}

/// Half-split (NeoX) RoPE in place: within each head, element `i` pairs with
/// `i + head_dim / 2`, rotated by angle `pos · θ_i`. Same tables as [`rope`]. LFM2
/// uses this one; its GGUF keeps the q/k weights in HF order.
pub fn rope_neox(x: &mut [f32], heads: usize, head_dim: usize, pos0: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    x.par_chunks_mut(heads * head_dim)
        .enumerate()
        .for_each(|(row, tok)| {
            let c = &cos[(pos0 + row) * half..][..half];
            let s = &sin[(pos0 + row) * half..][..half];
            for head in tok.chunks_exact_mut(head_dim) {
                let (lo, hi) = head.split_at_mut(half);
                for i in 0..half {
                    let (a, b) = (lo[i], hi[i]);
                    lo[i] = a * c[i] - b * s[i];
                    hi[i] = a * s[i] + b * c[i];
                }
            }
        });
}

/// LFM2's gated short convolution on `bcx: [t, 3 · width]` (the `in_proj`
/// output, split as `b | c | x`): `c · conv(b · x)`, with a causal depthwise
/// kernel of `taps` taps per channel (`kernel: [width, taps]`, oldest tap
/// first). `state: [taps - 1, width]` holds the `b · x` rows before this chunk,
/// oldest first, and is advanced to end with this chunk's last rows.
pub fn short_conv(bcx: &[f32], width: usize, kernel: &[f32], taps: usize, state: &mut [f32]) -> Vec<f32> {
    let t = bcx.len() / (3 * width);
    let keep = taps - 1;
    // `b · x` for the kept rows and this chunk: [keep + t, width].
    let mut bx = Vec::with_capacity((keep + t) * width);
    bx.extend_from_slice(state);
    for row in bcx.chunks_exact(3 * width) {
        let (b, rest) = row.split_at(width);
        bx.extend(b.iter().zip(&rest[width..]).map(|(b, x)| b * x));
    }
    let mut out = vec![0f32; t * width];
    out.par_chunks_mut(width).enumerate().for_each(|(i, o)| {
        let c = &bcx[i * 3 * width + width..][..width];
        for (ch, o) in o.iter_mut().enumerate() {
            let k = &kernel[ch * taps..][..taps];
            let z: f32 = (0..taps).map(|j| k[j] * bx[(i + j) * width + ch]).sum();
            *o = c[ch] * z;
        }
    });
    state.copy_from_slice(&bx[t * width..]);
    out
}

/// `silu(gate) · up`, elementwise.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; gate.len()];
    out.par_chunks_mut(4096)
        .zip(gate.par_chunks(4096).zip(up.par_chunks(4096)))
        .for_each(|(o, (g, u))| {
            for ((o, &g), &u) in o.iter_mut().zip(g).zip(u) {
                *o = g / (1.0 + (-g).exp()) * u;
            }
        });
    out
}

/// `x += y`, elementwise.
pub fn add_assign(x: &mut [f32], y: &[f32]) {
    x.par_chunks_mut(4096)
        .zip(y.par_chunks(4096))
        .for_each(|(x, y)| {
            for (a, &b) in x.iter_mut().zip(y) {
                *a += b;
            }
        });
}

/// Shapes for [`attention`].
#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// Element type of the KV cache: f32, or f16 to halve the bytes attention reads
/// (design.md §3). Rows are widened to f32 for the dot products.
pub trait KvElem: Copy + Send + Sync + 'static {
    /// `src` as f32, converting into `buf` if needed.
    fn row<'a>(src: &'a [Self], buf: &'a mut [f32]) -> &'a [f32];
}

/// `src` converted to f16 (F16C-accelerated where available).
pub fn to_f16_vec(src: &[f32]) -> Vec<half::f16> {
    use half::slice::HalfFloatSliceExt;
    let mut out = vec![half::f16::ZERO; src.len()];
    out.convert_from_f32_slice(src);
    out
}

impl KvElem for f32 {
    #[inline]
    fn row<'a>(src: &'a [f32], _buf: &'a mut [f32]) -> &'a [f32] {
        src
    }
}

impl KvElem for half::f16 {
    #[inline]
    fn row<'a>(src: &'a [half::f16], buf: &'a mut [f32]) -> &'a [f32] {
        use half::slice::HalfFloatSliceExt;
        let buf = &mut buf[..src.len()];
        src.convert_to_f32_slice(buf);
        buf
    }
}

/// Causal attention for `t` new queries at positions `past..past + t`.
///
/// - `q`: `[t, heads · head_dim]` (RoPE applied).
/// - `k`, `v`: the cache, position-major `[past + t, kv_heads · head_dim]`.
///
/// Query head `h` uses KV head `h / (heads / kv_heads)` (GQA). Returns
/// `[t, heads · head_dim]`. Work is split by head, so one head's keys and values
/// stay in that thread's cache while it walks its queries.
pub fn attention<E: KvElem>(
    q: &[f32],
    k: &[E],
    v: &[E],
    t: usize,
    past: usize,
    shape: AttnShape,
) -> Vec<f32> {
    let AttnShape {
        heads,
        kv_heads,
        head_dim: hd,
    } = shape;
    let group = heads / kv_heads;
    let kv_width = kv_heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();

    // Computed per head as [t, hd], then interleaved back into [t, heads · hd].
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let kvh = h / group;
            let mut out = vec![0f32; t * hd];
            let mut scores = Vec::with_capacity(past + t);
            let mut kbuf = vec![0f32; hd];
            let mut vbuf = vec![0f32; hd];
            for i in 0..t {
                let qi = &q[i * heads * hd + h * hd..][..hd];
                let n = past + i + 1; // causal: keys 0..=past + i
                scores.clear();
                let mut max = f32::NEG_INFINITY;
                for j in 0..n {
                    let kj = E::row(&k[j * kv_width + kvh * hd..][..hd], &mut kbuf);
                    let s = dot(qi, kj) * scale;
                    max = max.max(s);
                    scores.push(s);
                }
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let o = &mut out[i * hd..][..hd];
                for (j, &p) in scores.iter().enumerate() {
                    let vj = E::row(&v[j * kv_width + kvh * hd..][..hd], &mut vbuf);
                    let p = p * inv;
                    for (o, &vv) in o.iter_mut().zip(vj) {
                        *o += p * vv;
                    }
                }
            }
            out
        })
        .collect();

    let mut out = vec![0f32; t * heads * hd];
    out.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(i, row)| {
            for (h, dst) in row.chunks_exact_mut(hd).enumerate() {
                dst.copy_from_slice(&per_head[h][i * hd..][..hd]);
            }
        });
    out
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent accumulators so the loop vectorises without reassociation.
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) % 2001) as f32 / 1000.0
                    - 1.0
            })
            .collect()
    }

    /// Textbook attention: full score matrix, explicit causal mask, explicit GQA.
    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        t: usize,
        past: usize,
        sh: AttnShape,
    ) -> Vec<f32> {
        let (h_n, hd) = (sh.heads, sh.head_dim);
        let group = sh.heads / sh.kv_heads;
        let mut out = vec![0f32; t * h_n * hd];
        for h in 0..h_n {
            let kvh = h / group;
            for i in 0..t {
                let s: Vec<f32> = (0..past + t)
                    .map(|j| {
                        if j > past + i {
                            return f32::NEG_INFINITY;
                        }
                        let mut d = 0.0;
                        for e in 0..hd {
                            d += q[(i * h_n + h) * hd + e] * k[(j * sh.kv_heads + kvh) * hd + e];
                        }
                        d / (hd as f32).sqrt()
                    })
                    .collect();
                let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f32 = e.iter().sum();
                for (j, p) in e.iter().enumerate() {
                    for d in 0..hd {
                        out[(i * h_n + h) * hd + d] += p / z * v[(j * sh.kv_heads + kvh) * hd + d];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn attention_matches_naive_prefill_decode_and_gqa() {
        for (heads, kv_heads, t, past) in [
            (4, 4, 5, 0),
            (4, 4, 1, 9),
            (8, 2, 6, 3),
            (8, 2, 1, 0),
            (6, 1, 3, 4),
        ] {
            let sh = AttnShape {
                heads,
                kv_heads,
                head_dim: 16,
            };
            let q = data(t * heads * 16, 1);
            let k = data((past + t) * kv_heads * 16, 2);
            let v = data((past + t) * kv_heads * 16, 3);
            let got = attention(&q, &k, &v, t, past, sh);
            let want = naive_attention(&q, &k, &v, t, past, sh);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "{sh:?} t={t} past={past}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn f16_cache_attention_close_to_f32() {
        let sh = AttnShape {
            heads: 8,
            kv_heads: 2,
            head_dim: 16,
        };
        let (t, past) = (3, 20);
        let q = data(t * 8 * 16, 1);
        let k = data((past + t) * 2 * 16, 2);
        let v = data((past + t) * 2 * 16, 3);
        let want = attention(&q, &k, &v, t, past, sh);
        let (k16, v16) = (to_f16_vec(&k), to_f16_vec(&v));
        let got = attention(&q, &k16, &v16, t, past, sh);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 2e-3, "{g} vs {w}");
        }
    }

    #[test]
    fn rope_rotates_adjacent_pairs() {
        // One token at position 1, one head of dim 4: pairs (0,1) and (2,3).
        let (cos, sin) = ([1.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.25]);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        rope(&mut x, 1, 4, 1, &cos, &sin);
        assert_eq!(
            x,
            [
                1.0 * 0.0 - 2.0 * 1.0,
                1.0 * 1.0 + 2.0 * 0.0,
                3.0 * 0.5 - 4.0 * 0.25,
                3.0 * 0.25 + 4.0 * 0.5
            ]
        );
    }

    #[test]
    fn rope_neox_rotates_split_halves() {
        // One token at position 1, one head of dim 4: pairs (0,2) and (1,3).
        let (cos, sin) = ([1.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.25]);
        let mut x = [1.0, 2.0, 3.0, 4.0];
        rope_neox(&mut x, 1, 4, 1, &cos, &sin);
        assert_eq!(
            x,
            [
                1.0 * 0.0 - 3.0 * 1.0,
                2.0 * 0.5 - 4.0 * 0.25,
                1.0 * 1.0 + 3.0 * 0.0,
                2.0 * 0.25 + 4.0 * 0.5
            ]
        );
    }

    #[test]
    fn short_conv_is_causal_and_chunks_like_one_pass() {
        let (width, taps) = (3, 3);
        let bcx = data(5 * 3 * width, 7);
        let kernel = data(width * taps, 8);
        // Naive: zero-padded causal conv of b·x, gated by c.
        let bx: Vec<f32> = bcx
            .chunks(3 * width)
            .flat_map(|r| (0..width).map(move |ch| r[ch] * r[2 * width + ch]))
            .collect();
        let mut want = Vec::new();
        for i in 0..5 {
            for ch in 0..width {
                let z: f32 = (0..taps)
                    .filter(|j| i + j >= taps - 1)
                    .map(|j| kernel[ch * taps + j] * bx[(i + j - (taps - 1)) * width + ch])
                    .sum();
                want.push(bcx[i * 3 * width + width + ch] * z);
            }
        }
        let mut state = vec![0f32; (taps - 1) * width];
        let mut got = short_conv(&bcx[..2 * 3 * width], width, &kernel, taps, &mut state);
        got.extend(short_conv(&bcx[2 * 3 * width..], width, &kernel, taps, &mut state));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} vs {w}");
        }
        assert_eq!(state, bx[3 * width..]);
    }

    #[test]
    fn rms_norm_and_swiglu() {
        // Real models use eps > 0, which also keeps an all-zero row finite.
        let out = rms_norm(&[3.0, 4.0, 0.0, 0.0], 2, &[1.0, 2.0], 1e-5);
        let r = (12.5f32 + 1e-5).sqrt();
        assert!(
            (out[0] - 3.0 / r).abs() < 1e-6 && (out[1] - 8.0 / r).abs() < 1e-6,
            "{out:?}"
        );
        assert_eq!(out[2..], [0.0, 0.0]);
        let s = swiglu(&[0.0, 2.0], &[5.0, 3.0]);
        assert_eq!(s[0], 0.0);
        assert!((s[1] - 2.0 / (1.0 + (-2.0f32).exp()) * 3.0).abs() < 1e-6);
    }
}
