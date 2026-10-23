//! Google's EmbeddingGemma 2 (`google/embeddinggemma-2`) on the CPU: texts and
//! pictures into one 768-wide, L2-normalised space, for `/embed-compare` and
//! `increbot embed`. Weights stay bf16 (widened row by row in each matmul); the
//! token embedding table stays in the file mapping.
//!
//! A text is `<bos> {PREFIX}{text} <eos>`; a picture is
//! `<bos> <|image> <|image|>×n <image|> <eos>` with the vision tower's soft tokens
//! in place of the `n` image tokens. Either way the text tower runs over the whole
//! sequence and the result is the mean of its tokens, normalised.

pub mod matrix;
pub mod ops;
pub mod safetensors;
pub mod text;
pub mod vision;

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use crate::classify::vision::Rgb;
use safetensors::SafeTensors;
use text::{HIDDEN, Text};
use vision::Vision;

pub use text::EMBED_DIM;
pub use vision::{MAX_SOFT_TOKENS, Patches, preprocess};

pub const REPO: &str = "google/embeddinggemma-2";
/// The task prompt `sentence similarity` puts before every text (pictures get none).
pub const PREFIX: &str = "task: sentence similarity | query: ";
/// Texts longer than this many tokens are refused: attention is quadratic.
pub const MAX_TEXT_TOKENS: usize = 2048;

pub const BOS: u32 = 2;
pub const EOS: u32 = 1;
/// `<|image>`, `<|image|>` and `<image|>`.
pub const BOI: u32 = 255_999;
pub const IMAGE: u32 = 258_880;
pub const EOI: u32 = 258_882;

pub struct Embedder {
    tokenizer: tokenizers::Tokenizer,
    text: Text,
    vision: Vision,
}

impl Embedder {
    /// From the Hugging Face cache (downloading what is missing).
    pub fn load() -> Result<Self> {
        let weights = crate::hub::resolve(REPO, "model.safetensors")?;
        let tokenizer = crate::hub::resolve(REPO, "tokenizer.json")?;
        Self::open(&weights, &tokenizer)
    }

    pub fn open(weights: &Path, tokenizer: &Path) -> Result<Self> {
        let st = Arc::new(SafeTensors::open(weights)?);
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer)
            .map_err(|e| anyhow::anyhow!("{}: {e}", tokenizer.display()))?;
        Ok(Self { tokenizer, vision: Vision::load(&st)?, text: Text::load(st)? })
    }

    /// `text`'s token ids with the prefix, `<bos>` and `<eos>`.
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(format!("{PREFIX}{text}"), true)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokens(text)?;
        ensure!(
            ids.len() <= MAX_TEXT_TOKENS,
            "{} tokens, more than {MAX_TEXT_TOKENS}",
            ids.len()
        );
        let x = self.text.embed(&ids)?;
        Ok(self.text.encode(x, ids.len()))
    }

    /// The vision tower alone: `[n, 512]` soft tokens.
    pub fn soft_tokens(&self, patches: &Patches) -> Vec<f32> {
        self.vision.soft_tokens(patches)
    }

    /// The token ids around `n` soft tokens.
    pub fn image_tokens(n: usize) -> Vec<u32> {
        [BOS, BOI].into_iter().chain(std::iter::repeat_n(IMAGE, n)).chain([EOI, EOS]).collect()
    }

    pub fn embed_patches(&self, patches: &Patches) -> Result<Vec<f32>> {
        let soft = self.soft_tokens(patches);
        let n = soft.len() / HIDDEN;
        let ids = Self::image_tokens(n);
        let mut x = self.text.embed(&ids)?;
        let first = ids.iter().position(|&t| t == IMAGE).context("no image tokens")?;
        x[first * HIDDEN..(first + n) * HIDDEN].copy_from_slice(&soft);
        Ok(self.text.encode(x, ids.len()))
    }

    pub fn embed_image(&self, picture: &Rgb) -> Result<Vec<f32>> {
        self.embed_patches(&preprocess(picture))
    }
}

/// The cosine similarity of two embeddings (normalised or not).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(&p, &q)| f64::from(p) * f64::from(q)).sum::<f64>();
    let d = (dot(a, a) * dot(b, b)).sqrt();
    if d == 0.0 { 0.0 } else { (dot(a, b) / d) as f32 }
}
