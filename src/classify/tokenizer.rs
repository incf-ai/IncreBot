//! d1's tokenizer: the checkpoint's `tokenizer.json` (byte-level BPE) through the `tokenizers`
//! crate (ported from d1-burn, which builds it with `fancy-regex`; this crate uses `onig`, and
//! the golden test checks the two agree on d1's pre-tokenizer).
//!
//! Text is encoded as the remote code does it, with `add_special_tokens=False`: the BOS string is
//! part of the rendered prompt (`prompt::Renderer`), and special-token strings such as
//! `<|im_start|>` in the text still map to their own ids.

use std::fs;
use std::path::Path;

use serde_json::Value;

use super::Error;

/// The tokenizer and the BOS string the prompt starts with.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos: String,
}

impl Tokenizer {
    /// Loads `tokenizer.json` and the `bos_token` of `tokenizer_config.json` from a checkpoint
    /// directory (a Hugging Face snapshot).
    ///
    /// # Errors
    ///
    /// If either file is missing or malformed.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let path = dir.join("tokenizer_config.json");
        let text = fs::read_to_string(&path)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        let config: Value = serde_json::from_str(&text)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        // As runner.py: a BOS that is not a plain string (or none) is no BOS.
        let bos = config
            .get("bos_token")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Self::from_file(&dir.join("tokenizer.json"), bos)
    }

    /// Loads a `tokenizer.json`, with `bos` as the BOS string.
    ///
    /// # Errors
    ///
    /// If the file is missing or malformed.
    pub fn from_file(path: &Path, bos: &str) -> Result<Self, Error> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
        Ok(Self {
            inner,
            bos: bos.to_owned(),
        })
    }

    /// The vocabulary size, special tokens included.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// The text of token `id`, if it exists.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// The BOS string (`<|startoftext|>` for d1-3B); empty if the checkpoint has none.
    #[must_use]
    pub fn bos(&self) -> &str {
        &self.bos
    }

    /// The ids of `text`, with no special tokens added.
    ///
    /// # Errors
    ///
    /// If the tokenizer fails (it does not on valid UTF-8 with this checkpoint).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::new(format!("encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The id of `text` if it encodes to exactly one token.
    ///
    /// # Errors
    ///
    /// As [`Tokenizer::encode`].
    pub fn single(&self, text: &str) -> Result<Option<u32>, Error> {
        Ok(match self.encode(text)?.as_slice() {
            &[id] => Some(id),
            _ => None,
        })
    }
}
