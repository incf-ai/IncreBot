//! A read-only `.safetensors` file, memory-mapped: an 8-byte little-endian
//! header length, a JSON header naming each tensor's dtype, shape and byte
//! range, then the data.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

/// An element type this reader converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I64,
    U8,
    /// Any other, by name, so the error can say which.
    Other,
}

impl Dtype {
    fn parse(s: &str) -> Self {
        match s {
            "BF16" => Self::Bf16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "I64" => Self::I64,
            "U8" => Self::U8,
            _ => Self::Other,
        }
    }

    const fn size(self) -> usize {
        match self {
            Self::U8 | Self::Other => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

/// One tensor's entry in the header.
#[derive(Debug, Clone)]
pub struct Info {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range inside the mapping (header included).
    start: usize,
    end: usize,
}

impl Info {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct SafeTensors {
    mmap: Mmap,
    tensors: HashMap<String, Info>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is opened read-only and the mapping lives in this struct; nothing here
        // writes to it. Another process truncating the file underneath would fault, as with any
        // mapped model file.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
        Self::parse(mmap).with_context(|| format!("parsing safetensors {}", path.display()))
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let head: [u8; 8] = mmap.get(..8).context("shorter than its header")?.try_into()?;
        let len = usize::try_from(u64::from_le_bytes(head))?;
        let base = len.checked_add(8).context("header length")?;
        let json = mmap.get(8..base).context("header past the end")?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(json)?;
        let mut tensors = HashMap::new();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().context("dtype")?;
            let shape = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()).context("dimension"))
                .collect::<Result<Vec<_>>>()?;
            let offsets = v["data_offsets"].as_array().context("data_offsets")?;
            let offset = |i: usize| -> Result<usize> {
                let o = offsets.get(i).and_then(serde_json::Value::as_u64).context("offset")?;
                Ok(base + usize::try_from(o)?)
            };
            let (start, end) = (offset(0)?, offset(1)?);
            let info = Info { dtype: Dtype::parse(dtype), shape, start, end };
            ensure!(start <= end && end <= mmap.len(), "{name}: bytes {start}..{end}");
            if info.dtype != Dtype::Other {
                ensure!(end - start == info.len() * info.dtype.size(), "{name}: {dtype} size");
            }
            tensors.insert(name, info);
        }
        Ok(Self { mmap, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&Info> {
        self.tensors.get(name).with_context(|| format!("no tensor {name}"))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// The raw little-endian bytes of `name`.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.info(name)?;
        Ok(&self.mmap[info.start..info.end])
    }

    /// `name` as bf16 bit patterns, checking its shape.
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let info = self.checked(name, shape)?;
        ensure!(info.dtype == Dtype::Bf16, "{name}: {:?}, not BF16", info.dtype);
        Ok(self.bytes(name)?.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect())
    }

    /// `name` converted to f32 (from BF16, F16 or F32), checking its shape.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let info = self.checked(name, shape)?;
        let bytes = self.bytes(name)?;
        Ok(match info.dtype {
            Dtype::Bf16 => bytes.as_chunks::<2>().0.iter().map(|b| bf16(u16::from_le_bytes(*b))).collect(),
            Dtype::F16 => {
                bytes.as_chunks::<2>().0.iter().map(|b| half::f16::from_le_bytes(*b).to_f32()).collect()
            }
            Dtype::F32 => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            other => bail!("{name}: {other:?}, not a float"),
        })
    }

    /// `name` as i64, checking its dtype.
    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        let info = self.info(name)?;
        ensure!(info.dtype == Dtype::I64, "{name}: {:?}, not I64", info.dtype);
        Ok(self.bytes(name)?.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect())
    }

    fn checked(&self, name: &str, shape: &[usize]) -> Result<&Info> {
        let info = self.info(name)?;
        ensure!(info.shape == shape, "{name}: shape {:?}, expected {shape:?}", info.shape);
        Ok(info)
    }
}

/// A bf16 bit pattern as f32 (exact).
pub fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_values() {
        assert_eq!(bf16(0x3f80), 1.0);
        assert_eq!(bf16(0xc000), -2.0);
        assert_eq!(bf16(0x0000), 0.0);
    }
}
