//! Linear layers behind one trait (design.md §3).
//!
//! The trait works on plain row-major slices: the forward pass outside the linear
//! layers runs on our own kernels (`kernels.rs`), not Burn tensors. Burn provides
//! the quantized import and the reference implementations here.

use burn::tensor::{Device, Tensor, TensorData};

pub trait QLinear: Send + Sync {
    /// `x`: `[m, in]` row-major → `[m, out]` row-major.
    fn forward(&self, x: &[f32], m: usize) -> Vec<f32>;
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    /// The GGUF K-quant blocks as stored, for uploading to the GPU (gpu-plan.md).
    fn k_blocks(&self) -> Option<(super::KQuant, &[u8])> {
        None
    }
}

fn to_tensor(x: &[f32], m: usize, k: usize, device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(x.to_vec(), [m, k]), device)
}

fn to_vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("f32 tensor")
}

/// Reference implementation: a Burn quantized weight `[out, in]`, dequantized on
/// every call. Bit-exact weights, f32 activations, slow (burn-flex dequantizes on
/// one thread). Used for tests and the small model only.
pub struct BurnQLinear {
    weight: Tensor<2>,
}

impl BurnQLinear {
    pub fn new(weight: Tensor<2>) -> Self {
        Self { weight }
    }
}

impl QLinear for BurnQLinear {
    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let x = to_tensor(x, m, self.in_features(), &self.weight.device());
        // Transpose the *dequantized* tensor: layout ops on a block-quantized tensor
        // re-quantize on burn-flex (design.md §2).
        to_vec(x.matmul(self.weight.clone().dequantize().transpose()))
    }

    fn in_features(&self) -> usize {
        self.weight.dims()[1]
    }

    fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }
}

/// The same weights as [`BurnQLinear`], dequantized once at load instead of on
/// every call: identical f32 weights and the same matmul, so identical results,
/// at 4 bytes per weight. Only for small models (tests, the reference model);
/// [`Llama::load`](super::Llama::load) refuses it above a size limit.
pub struct DenseLinear {
    /// `[in, out]`, already transposed for `x.matmul`.
    weight_t: Tensor<2>,
}

impl DenseLinear {
    pub fn from_quantized(weight: Tensor<2>) -> Self {
        Self {
            weight_t: weight.dequantize().transpose(),
        }
    }

    /// From weights already decoded to f32, `[out, in]` row-major (any GGUF type).
    pub fn from_f32(
        weight: Vec<f32>,
        out_features: usize,
        in_features: usize,
        device: &Device,
    ) -> Self {
        let w =
            Tensor::<2>::from_data(TensorData::new(weight, [out_features, in_features]), device);
        Self {
            weight_t: w.transpose(),
        }
    }
}

impl QLinear for DenseLinear {
    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let x = to_tensor(x, m, self.in_features(), &self.weight_t.device());
        to_vec(x.matmul(self.weight_t.clone()))
    }

    fn in_features(&self) -> usize {
        self.weight_t.dims()[0]
    }

    fn out_features(&self) -> usize {
        self.weight_t.dims()[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{deinterleave_q8_0, q8_0_scheme};

    #[test]
    fn dense_matches_burn_exactly() {
        let (out_f, in_f) = (48, 64);
        let mut bytes = Vec::new();
        for b in 0..out_f * in_f / 32 {
            bytes.extend(half::f16::from_f32(0.01 + b as f32 * 1e-3).to_le_bytes());
            bytes.extend((0..32).map(|i| ((b * 7 + i * 13) % 255) as u8));
        }
        let (v, s) = deinterleave_q8_0(&bytes);
        let device = Default::default();
        let w = || {
            Tensor::<2>::from_data(
                TensorData::quantized(v.clone(), [out_f, in_f], q8_0_scheme(), &s, None),
                &device,
            )
        };
        let x: Vec<f32> = (0..3 * in_f).map(|i| (i as f32 * 0.1).cos()).collect();
        let a = BurnQLinear::new(w()).forward(&x, 3);
        let b = DenseLinear::from_quantized(w()).forward(&x, 3);
        assert_eq!(a, b);
    }
}
