//! GGUF reader: header, metadata, tensor infos, and decoding of the tensor
//! types we support (Q8_0, Q4_K, Q6_K, F16, F32). See design.md §2 and §5.

mod burn_import;
pub mod kquant;
mod quant;

pub use burn_import::{f32_tensor, q8_0_scheme, q8_0_tensor};
pub use kquant::{Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K};
pub use quant::{Q8_0_BLOCK_BYTES, QK8_0, decode_q8_0, deinterleave_q8_0};

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// A GGML tensor element type. Only the ones we can load are named; any other
/// type is kept as its raw id so the error can name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Q4K,
    Q6K,
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> Self {
        match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            other => Self::Other(other),
        }
    }

    /// Bytes needed for `n` elements, or `None` for an unsupported type.
    pub(crate) fn byte_size(self, n: u64) -> Option<u64> {
        match self {
            Self::F32 => Some(n * 4),
            Self::F16 => Some(n * 2),
            Self::Q8_0 => n
                .is_multiple_of(QK8_0 as u64)
                .then(|| n / QK8_0 as u64 * Q8_0_BLOCK_BYTES as u64),
            Self::Q4K => n
                .is_multiple_of(QK_K as u64)
                .then(|| n / QK_K as u64 * Q4_K_BLOCK_BYTES as u64),
            Self::Q6K => n
                .is_multiple_of(QK_K as u64)
                .then(|| n / QK_K as u64 * Q6_K_BLOCK_BYTES as u64),
            Self::Other(_) => None,
        }
    }
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// Any integer type as `u64`, if it is non-negative.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// One tensor's header entry.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: innermost dimension first.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn num_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Row-major shape (outermost first), the order Burn uses.
    pub fn shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }
}

/// A parsed GGUF file. The tensor data stays memory-mapped until `self` is dropped.
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    by_name: BTreeMap<String, usize>,
    data_start: usize,
    mmap: Mmap,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: the file is opened read-only and we never hand out the mapping beyond
        // `&self`; if another process truncates it while mapped we would fault, which is
        // the usual mmap caveat and acceptable for model files.
        let mmap =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mapping {}", path.display()))?;
        Self::parse(mmap).with_context(|| format!("parsing GGUF {}", path.display()))
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let mut r = Reader { buf: &mmap, pos: 0 };
        ensure!(r.u32()? == MAGIC, "not a GGUF file (bad magic)");
        let version = r.u32()?;
        ensure!(
            matches!(version, 2 | 3),
            "unsupported GGUF version {version} (need 2 or 3)"
        );
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let value = r.value(ty).with_context(|| format!("metadata key {key}"))?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut by_name = BTreeMap::new();
        for i in 0..n_tensors as usize {
            let name = r.string()?;
            let n_dims = r.u32()?;
            ensure!(n_dims <= 4, "tensor {name}: {n_dims} dims");
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ty = GgmlType::from_id(r.u32()?);
            let offset = r.u64()?;
            by_name.insert(name.clone(), i);
            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
            });
        }

        let alignment = match metadata.get("general.alignment") {
            Some(v) => v.as_u64().context("general.alignment is not an integer")?,
            None => DEFAULT_ALIGNMENT,
        };
        ensure!(
            alignment.is_power_of_two(),
            "alignment {alignment} is not a power of two"
        );
        let data_start = (r.pos as u64).next_multiple_of(alignment) as usize;

        let gguf = Gguf {
            version,
            metadata,
            tensors,
            by_name,
            data_start,
            mmap,
        };
        for t in &gguf.tensors {
            // Bounds-check every tensor up front; unsupported types are only an error
            // when someone asks for that tensor (`bytes`), so `inspect` still works.
            if let Some(size) = t.ty.byte_size(t.num_elements()) {
                let end = gguf.data_start as u64 + t.offset + size;
                ensure!(
                    end <= gguf.mmap.len() as u64,
                    "tensor {} runs past end of file",
                    t.name
                );
            }
        }
        Ok(gguf)
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        match self.by_name.get(name) {
            Some(&i) => Ok(&self.tensors[i]),
            None => bail!("tensor {name} not found in GGUF"),
        }
    }

    /// The raw bytes of a tensor. Errors (naming the tensor) for unsupported types.
    pub fn bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let Some(size) = info.ty.byte_size(info.num_elements()) else {
            bail!("tensor {}: unsupported GGML type {:?}", info.name, info.ty);
        };
        let start = self.data_start + info.offset as usize;
        Ok(&self.mmap[start..start + size as usize])
    }

    /// Decodes any supported tensor to f32, row-major.
    pub fn to_f32(&self, info: &TensorInfo) -> Result<Vec<f32>> {
        let bytes = self.bytes(info)?;
        Ok(match info.ty {
            GgmlType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect(),
            GgmlType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| half::f16::from_le_bytes(b).to_f32())
                .collect(),
            GgmlType::Q8_0 => decode_q8_0(bytes),
            GgmlType::Q4K => kquant::decode_q4_k(bytes),
            GgmlType::Q6K => kquant::decode_q6_k(bytes),
            // `bytes()` above already rejects unsupported types; a bug there
            // should fail this tensor's decode, not panic the whole process.
            GgmlType::Other(ty) => bail!("tensor {}: unsupported GGML type {ty:?}", info.name),
        })
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .with_context(|| format!("GGUF metadata key {key} missing"))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.get(key)?
            .as_u64()
            .with_context(|| format!("GGUF metadata key {key} is not an integer"))
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        self.get(key)?
            .as_f32()
            .with_context(|| format!("GGUF metadata key {key} is not a float"))
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.get(key)?
            .as_str()
            .with_context(|| format!("GGUF metadata key {key} is not a string"))
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else {
            bail!("unexpected end of header at byte {}", self.pos)
        };
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        // `take(N)` already guarantees exactly `N` bytes; this can't actually
        // fail, but the function already returns `Result`, so there's no
        // reason to reach for `.unwrap()` over it instead of `?`.
        self.take(N)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("internal error: take({N}) returned the wrong length"))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        // Token strings are not always valid UTF-8 in older files; keep them lossless
        // where possible and never fail the whole load over one token.
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let len = self.u64()? as usize;
                ensure!(len <= self.buf.len(), "array length {len} is implausible");
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.value(elem_ty)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => bail!("unknown metadata value type {other}"),
        })
    }
}
