//! GGUF tensors → Burn tensors (design.md §2).

use anyhow::{Result, ensure};
use burn::tensor::quantization::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn::tensor::{Device, Tensor, TensorData};

use super::{GgmlType, Gguf, QK8_0, TensorInfo, deinterleave_q8_0};

/// The Burn scheme that represents GGUF Q8_0 exactly: i8 values in −128..127
/// (`Q8F`, not `Q8S`, which would clamp −128), symmetric, one f16 scale per
/// 32 values along the innermost dimension.
pub fn q8_0_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_value(QuantValue::Q8F)
        .with_mode(QuantMode::Symmetric)
        .with_store(QuantStore::Native)
        .per_block([QK8_0 as u8], ScaleDtype::F16)
}

/// Loads a 2-D Q8_0 tensor as a Burn quantized tensor, bit-exact.
pub fn q8_0_tensor(gguf: &Gguf, info: &TensorInfo, device: &Device) -> Result<Tensor<2>> {
    ensure!(
        info.ty == GgmlType::Q8_0,
        "tensor {} is {:?}, expected Q8_0",
        info.name,
        info.ty
    );
    let shape = shape2(info)?;
    let (values, scales) = deinterleave_q8_0(gguf.bytes(info)?);
    let data = TensorData::quantized(values, shape, q8_0_scheme(), &scales, None);
    Ok(Tensor::from_data(data, device))
}

/// Loads any supported tensor decoded to f32.
pub fn f32_tensor<const D: usize>(
    gguf: &Gguf,
    info: &TensorInfo,
    device: &Device,
) -> Result<Tensor<D>> {
    let shape = info.shape();
    ensure!(
        shape.len() == D,
        "tensor {} has rank {}, expected {D}",
        info.name,
        shape.len()
    );
    let data = TensorData::new(gguf.to_f32(info)?, shape);
    Ok(Tensor::from_data(data, device))
}

fn shape2(info: &TensorInfo) -> Result<[usize; 2]> {
    let s = info.shape();
    ensure!(
        s.len() == 2,
        "tensor {} has rank {}, expected 2",
        info.name,
        s.len()
    );
    ensure!(
        s[1].is_multiple_of(QK8_0),
        "tensor {}: inner dim {} is not a multiple of {QK8_0}",
        info.name,
        s[1]
    );
    Ok([s[0], s[1]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::quant::encode_block;
    use crate::gguf::{Q8_0_BLOCK_BYTES, decode_q8_0};

    /// `rows × cols` Q8_0 blocks with varied scales and every i8 value, including −128.
    fn blocks(rows: usize, cols: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut v = -128i32;
        for b in 0..rows * cols / QK8_0 {
            let mut q = [0i8; QK8_0];
            for x in &mut q {
                *x = v as i8;
                v = if v == 127 { -128 } else { v + 1 };
            }
            let d = (b as f32 * 0.37).sin() * 0.05;
            bytes.extend(encode_block(d, &q));
        }
        assert_eq!(bytes.len(), rows * cols / QK8_0 * Q8_0_BLOCK_BYTES);
        bytes
    }

    #[test]
    fn burn_q8f_dequantize_is_bit_exact() {
        let (rows, cols) = (24, 96);
        let bytes = blocks(rows, cols);
        let (values, scales) = deinterleave_q8_0(&bytes);
        assert!(values.contains(&-128));
        let data = TensorData::quantized(values, [rows, cols], q8_0_scheme(), &scales, None);
        let t: Tensor<2> = Tensor::from_data(data, &Default::default());

        let got = t.dequantize().into_data().try_into_vec::<f32>().unwrap();
        let want = decode_q8_0(&bytes);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "element {i}: {g} vs {w}");
        }
    }
}
