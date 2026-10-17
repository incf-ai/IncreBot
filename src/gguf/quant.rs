//! Q8_0 block layout: `f16 d` followed by 32 `i8` values, `x = d · q`.

use half::f16;

/// Elements per Q8_0 block.
pub const QK8_0: usize = 32;
/// Bytes per Q8_0 block: f16 scale + 32 × i8.
pub const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

#[inline]
fn block_scale(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[0], block[1]]).to_f32()
}

/// Decodes Q8_0 blocks to f32 in element order.
pub fn decode_q8_0(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(
        bytes.len() % Q8_0_BLOCK_BYTES,
        0,
        "Q8_0 data is not a whole number of blocks"
    );
    let mut out = Vec::with_capacity(bytes.len() / Q8_0_BLOCK_BYTES * QK8_0);
    for block in bytes.as_chunks::<Q8_0_BLOCK_BYTES>().0 {
        let d = block_scale(block);
        out.extend(block[2..].iter().map(|&q| d * (q as i8) as f32));
    }
    out
}

/// Splits interleaved Q8_0 blocks into Burn's layout: all values (element order),
/// then all scales. Scales are returned as f32 holding exact f16 values, so
/// re-encoding them as f16 is lossless.
pub fn deinterleave_q8_0(bytes: &[u8]) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(
        bytes.len() % Q8_0_BLOCK_BYTES,
        0,
        "Q8_0 data is not a whole number of blocks"
    );
    let n_blocks = bytes.len() / Q8_0_BLOCK_BYTES;
    let mut values = Vec::with_capacity(n_blocks * QK8_0);
    let mut scales = Vec::with_capacity(n_blocks);
    for block in bytes.as_chunks::<Q8_0_BLOCK_BYTES>().0 {
        scales.push(block_scale(block));
        values.extend(block[2..].iter().map(|&q| q as i8));
    }
    (values, scales)
}

#[cfg(test)]
pub(crate) fn encode_block(d: f32, q: &[i8; QK8_0]) -> [u8; Q8_0_BLOCK_BYTES] {
    let mut b = [0u8; Q8_0_BLOCK_BYTES];
    b[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
    for (dst, &v) in b[2..].iter_mut().zip(q) {
        *dst = v as u8;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_blocks() -> Vec<u8> {
        let mut q1 = [0i8; QK8_0];
        for (i, v) in q1.iter_mut().enumerate() {
            *v = i as i8 - 16;
        }
        // The format allows -128 even though llama.cpp's quantizer never emits it.
        let mut q2 = [127i8; QK8_0];
        q2[0] = -128;
        q2[31] = -1;
        let mut bytes = encode_block(0.5, &q1).to_vec();
        bytes.extend(encode_block(-0.0078125, &q2));
        bytes
    }

    #[test]
    fn decode_matches_d_times_q() {
        let out = decode_q8_0(&sample_blocks());
        assert_eq!(out.len(), 64);
        for (i, &x) in out[..32].iter().enumerate() {
            assert_eq!(x, 0.5 * (i as f32 - 16.0));
        }
        assert_eq!(out[32], -0.0078125 * -128.0);
        assert_eq!(out[33], -0.0078125 * 127.0);
        assert_eq!(out[63], -0.0078125 * -1.0);
    }

    #[test]
    fn deinterleave_keeps_order_and_scales() {
        let (values, scales) = deinterleave_q8_0(&sample_blocks());
        assert_eq!(scales, vec![0.5, -0.0078125]);
        assert_eq!(values[0], -16);
        assert_eq!(values[32], -128);
        assert_eq!(values[63], -1);
        let decoded = decode_q8_0(&sample_blocks());
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(scales[i / QK8_0] * v as f32, decoded[i]);
        }
    }
}
