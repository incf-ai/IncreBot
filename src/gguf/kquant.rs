//! K-quant block layouts (llama.cpp's `ggml-common.h`), used by the Q4_K_M files:
//! most weights in Q4_K, the sensitive ones and the LM head in Q6_K
//! (design.md §2). Both quantize 256 weights per block ("super-block").
//!
//! Q4_K, 144 bytes: `f16 d`, `f16 dmin`, 12 bytes packing eight 6-bit scales and
//! eight 6-bit mins (one per 32 weights), then 128 bytes of 4-bit values. Weight
//! `x = d·scale_j·q − dmin·min_j`. Each 64-weight chunk stores its first 32
//! weights in the low nibbles of 32 bytes and the next 32 in the high nibbles.
//!
//! Q6_K, 210 bytes: 128 bytes of low 4 bits (`ql`), 64 bytes of high 2 bits
//! (`qh`), sixteen i8 scales (one per 16 weights), `f16 d`. Weight
//! `x = d·scale_g·(q − 32)` with `q` in 0..64.

use half::f16;

/// Weights per K-quant block.
pub const QK_K: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;
pub const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

#[inline]
fn f16_at(b: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
}

/// The eight (scale, min) pairs of a Q4_K block, each 6 bits (llama.cpp's
/// `get_scale_min_k4`): pairs 0–3 sit in the low 6 bits of bytes 0–3 and 4–7;
/// pairs 4–7 put their low 4 bits in bytes 8–11 and their top 2 bits in the
/// spare top bits of bytes 0–7.
#[inline]
pub fn q4_k_scales_mins(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = s[j] & 63;
        mn[j] = s[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        mn[j] = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
    (sc, mn)
}

/// The parts of one Q4_K block: `(d, dmin, scales, mins, qs)`.
#[inline]
pub fn q4_k_parts(b: &[u8]) -> (f32, f32, [u8; 8], [u8; 8], &[u8]) {
    let (sc, mn) = q4_k_scales_mins(&b[4..16]);
    (f16_at(b, 0), f16_at(b, 2), sc, mn, &b[16..Q4_K_BLOCK_BYTES])
}

/// The unsigned 4-bit value of element `e` (0..256) of a Q4_K block's `qs`.
#[inline]
pub fn q4_k_value(qs: &[u8], e: usize) -> u8 {
    let (chunk, i) = (e / 64, e % 64);
    let byte = qs[chunk * 32 + i % 32];
    if i < 32 { byte & 0x0F } else { byte >> 4 }
}

/// Decodes one Q4_K block into 256 floats.
pub fn decode_q4_k_block(b: &[u8], out: &mut [f32]) {
    let (d, dmin, sc, mn, qs) = q4_k_parts(b);
    for (e, o) in out[..QK_K].iter_mut().enumerate() {
        let j = e / 32;
        *o = d * sc[j] as f32 * q4_k_value(qs, e) as f32 - dmin * mn[j] as f32;
    }
}

/// The parts of one Q6_K block: `(d, scales, ql, qh)`.
#[inline]
pub fn q6_k_parts(b: &[u8]) -> (f32, [i8; 16], &[u8], &[u8]) {
    let mut sc = [0i8; 16];
    for (s, &v) in sc.iter_mut().zip(&b[192..208]) {
        *s = v as i8;
    }
    (f16_at(b, 208), sc, &b[..128], &b[128..192])
}

/// The unsigned 6-bit value (0..64) of element `e` (0..256) of a Q6_K block.
/// Each 128-element half uses 64 `ql` bytes and 32 `qh` bytes: element `l` of
/// quarter `k` (32 each) takes `ql`'s low nibble for quarters 0–1 and high
/// nibble for 2–3, and bits `2k..2k+2` of `qh[l]`.
#[inline]
pub fn q6_k_value(ql: &[u8], qh: &[u8], e: usize) -> u8 {
    let (half, i) = (e / 128, e % 128);
    let (quarter, l) = (i / 32, i % 32);
    let lo_byte = ql[half * 64 + (quarter % 2) * 32 + l];
    let lo = if quarter < 2 {
        lo_byte & 0x0F
    } else {
        lo_byte >> 4
    };
    let hi = (qh[half * 32 + l] >> (2 * quarter)) & 3;
    lo | (hi << 4)
}

/// Decodes one Q6_K block into 256 floats.
pub fn decode_q6_k_block(b: &[u8], out: &mut [f32]) {
    let (d, sc, ql, qh) = q6_k_parts(b);
    for (e, o) in out[..QK_K].iter_mut().enumerate() {
        *o = d * sc[e / 16] as f32 * (q6_k_value(ql, qh, e) as i32 - 32) as f32;
    }
}

fn decode_blocks(bytes: &[u8], block_bytes: usize, decode: fn(&[u8], &mut [f32])) -> Vec<f32> {
    assert_eq!(
        bytes.len() % block_bytes,
        0,
        "data is not a whole number of blocks"
    );
    let mut out = vec![0f32; bytes.len() / block_bytes * QK_K];
    for (b, o) in bytes
        .chunks_exact(block_bytes)
        .zip(out.as_chunks_mut::<QK_K>().0)
    {
        decode(b, o);
    }
    out
}

pub fn decode_q4_k(bytes: &[u8]) -> Vec<f32> {
    decode_blocks(bytes, Q4_K_BLOCK_BYTES, decode_q4_k_block)
}

pub fn decode_q6_k(bytes: &[u8]) -> Vec<f32> {
    decode_blocks(bytes, Q6_K_BLOCK_BYTES, decode_q6_k_block)
}

/// Encoders for tests: build blocks from chosen values.
#[cfg(test)]
pub(crate) mod encode {
    use super::*;

    /// A Q4_K block from `d`, `dmin`, 6-bit scales/mins and 256 4-bit values.
    pub fn q4_k(d: f32, dmin: f32, sc: [u8; 8], mn: [u8; 8], q: &[u8; QK_K]) -> Vec<u8> {
        let mut b = vec![0u8; Q4_K_BLOCK_BYTES];
        b[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(dmin).to_le_bytes());
        let s = &mut b[4..16];
        for j in 0..4 {
            s[j] = sc[j] | ((sc[j + 4] >> 4) << 6);
            s[j + 4] = mn[j] | ((mn[j + 4] >> 4) << 6);
            s[j + 8] = (sc[j + 4] & 0x0F) | ((mn[j + 4] & 0x0F) << 4);
        }
        let qs = &mut b[16..];
        for (e, &v) in q.iter().enumerate() {
            let (chunk, i) = (e / 64, e % 64);
            let byte = &mut qs[chunk * 32 + i % 32];
            if i < 32 {
                *byte |= v & 0x0F;
            } else {
                *byte |= (v & 0x0F) << 4;
            }
        }
        b
    }

    /// A Q6_K block from `d`, sixteen scales and 256 6-bit values (0..64).
    pub fn q6_k(d: f32, sc: [i8; 16], q: &[u8; QK_K]) -> Vec<u8> {
        let mut b = vec![0u8; Q6_K_BLOCK_BYTES];
        for (e, &v) in q.iter().enumerate() {
            let (half, i) = (e / 128, e % 128);
            let (quarter, l) = (i / 32, i % 32);
            let lo = &mut b[half * 64 + (quarter % 2) * 32 + l];
            if quarter < 2 {
                *lo |= v & 0x0F;
            } else {
                *lo |= (v & 0x0F) << 4;
            }
            b[128 + half * 32 + l] |= ((v >> 4) & 3) << (2 * quarter);
        }
        for (dst, &s) in b[192..208].iter_mut().zip(&sc) {
            *dst = s as u8;
        }
        b[208..210].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        b
    }
}

#[cfg(test)]
mod tests {
    use super::encode;
    use super::*;

    fn pattern(seed: usize, modulo: usize) -> [u8; QK_K] {
        std::array::from_fn(|i| ((i * 37 + seed * 11 + i / 7) % modulo) as u8)
    }

    #[test]
    fn block_sizes_match_llama_cpp() {
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
    }

    #[test]
    fn q4_k_scales_round_trip_all_6_bit_values() {
        for base in 0..64u8 {
            let sc: [u8; 8] = std::array::from_fn(|j| (base + j as u8 * 9) % 64);
            let mn: [u8; 8] = std::array::from_fn(|j| (base * 3 + j as u8 * 5) % 64);
            let b = encode::q4_k(1.0, 1.0, sc, mn, &[0; QK_K]);
            assert_eq!(q4_k_scales_mins(&b[4..16]), (sc, mn));
        }
    }

    #[test]
    fn q4_k_decode_matches_the_formula() {
        let sc = [1, 5, 17, 63, 0, 32, 40, 9];
        let mn = [0, 63, 2, 7, 33, 1, 60, 12];
        let q = pattern(3, 16);
        let (d, dmin) = (0.0123, 0.0456);
        let b = encode::q4_k(d, dmin, sc, mn, &q);
        let (d, dmin) = (f16_at(&b, 0), f16_at(&b, 2));
        let out = decode_q4_k(&b);
        for e in 0..QK_K {
            let j = e / 32;
            assert_eq!(
                out[e],
                d * sc[j] as f32 * q[e] as f32 - dmin * mn[j] as f32,
                "element {e}"
            );
        }
    }

    #[test]
    fn q4_k_nibble_layout_matches_llama_cpp() {
        // llama.cpp's dequantize_row_q4_K walks 64 elements per 32 qs bytes:
        // low nibbles first, then high nibbles.
        let mut b = encode::q4_k(1.0, 0.0, [1; 8], [0; 8], &[0; QK_K]);
        b[16 + 32 + 5] = 0x9A; // chunk 1, byte 5
        let out = decode_q4_k(&b);
        assert_eq!(out[64 + 5], 10.0);
        assert_eq!(out[64 + 32 + 5], 9.0);
        assert_eq!(out.iter().filter(|&&v| v != 0.0).count(), 2);
    }

    #[test]
    fn q6_k_decode_matches_the_formula() {
        let sc: [i8; 16] = std::array::from_fn(|i| (i as i16 * 9 - 70) as i8);
        let q = pattern(5, 64);
        let b = encode::q6_k(0.0042, sc, &q);
        let d = f16_at(&b, 208);
        let out = decode_q6_k(&b);
        for e in 0..QK_K {
            assert_eq!(
                out[e],
                d * sc[e / 16] as f32 * (q[e] as i32 - 32) as f32,
                "element {e}"
            );
        }
    }

    #[test]
    fn q6_k_bit_layout_matches_llama_cpp() {
        // dequantize_row_q6_K: for l in 0..32, q3 = (ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4
        // lands at y[l + 64]; q4 = (ql[l+32] >> 4) | (qh[l] >> 6) << 4 at y[l + 96].
        let mut b = encode::q6_k(1.0, [1; 16], &[32; QK_K]);
        b[3] = (b[3] & 0x0F) | (0x7 << 4); // ql[3] high nibble
        b[128 + 3] = (b[128 + 3] & !(3 << 4)) | (2 << 4); // qh[3] bits 4..6
        let out = decode_q6_k(&b);
        assert_eq!(out[64 + 3], (0x7 | (2 << 4)) as f32 - 32.0);
        assert_eq!(out.iter().filter(|&&v| v != 0.0).count(), 1);
    }
}
