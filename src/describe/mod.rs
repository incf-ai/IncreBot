//! Qwen3-VL-8B-Instruct describing a picture on one GPU: the vision tower
//! ([`GpuVision`]; [`vision`] loads it and keeps the CPU reference) and the
//! language model ([`GpuQwen3`]).
//!
//! One user turn holds the picture then the question; the picture's rows take
//! the `<|image_pad|>` tokens' places, at 3D (time, row, column) positions,
//! and the DeepStack rows join them after the first layers.

pub mod vision;

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};

use crate::classify::vision::Rgb;
use crate::engine::sampler::{Sampler, SamplerParams};
use crate::gpu::qwen3::{GpuQwen3, Pos};
use crate::gpu::qwen3_vision::GpuVision;
use crate::gpu::vk::Device;
use vision::{Features, Vision};

pub const REPO: &str = "Qwen/Qwen3-VL-8B-Instruct-GGUF";
pub const MODEL_FILE: &str = "Qwen3VL-8B-Instruct-Q4_K_M.gguf";
pub const MMPROJ_FILE: &str = "mmproj-Qwen3VL-8B-Instruct-F16.gguf";
/// The GGUF repo has no `tokenizer.json`; the original's is the same vocabulary.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3-VL-8B-Instruct";
/// The picture's limit in merged rows (language-model tokens).
pub const MAX_IMAGE_TOKENS: usize = 1024;

pub const IM_START: u32 = 151_644;
pub const IM_END: u32 = 151_645;
pub const END_OF_TEXT: u32 = 151_643;
pub const VISION_START: u32 = 151_652;
pub const VISION_END: u32 = 151_653;
pub const IMAGE_PAD: u32 = 151_655;
/// Asked when the user gives no question.
pub const DEFAULT_PROMPT: &str = "Describe this image.";
/// The question's limit in tokens.
pub const MAX_PROMPT_TOKENS: usize = 512;

/// How the answer is drawn.
#[derive(Clone, Debug)]
pub struct Params {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for Params {
    /// The model card's settings for pictures.
    fn default() -> Self {
        Self { max_new_tokens: 512, temperature: 0.7, top_k: 20, top_p: 0.8 }
    }
}

pub struct Description {
    pub text: String,
    pub tokens: Vec<u32>,
    /// Picture rows (merged tokens) and the grid they came from.
    pub image_tokens: usize,
    pub grid: (usize, usize),
    pub prompt_tokens: usize,
    pub generated: usize,
    /// True when `max_new_tokens` ran out before the end of the turn.
    pub truncated: bool,
    pub vision_secs: f32,
    pub prefill_secs: f32,
    pub decode_secs: f32,
}

/// A user turn's tokens, ready for the language model.
pub struct Turn {
    pub ids: Vec<u32>,
    pub pos: Vec<Pos>,
    /// The first picture row.
    pub first: usize,
    /// The first answer token's position.
    pub next: u32,
}

pub struct Describer {
    tokenizer: tokenizers::Tokenizer,
    vision: Mutex<GpuVision>,
    gpu: Mutex<GpuQwen3>,
    max_image_tokens: usize,
    params: Params,
}

impl Describer {
    /// Loads the tower and the language model onto `dev`, with room for the
    /// largest picture and a cache for the longest turn allowed.
    pub fn open(dev: &Arc<Device>, model: &Path, mmproj: &Path, tokenizer: &Path, max_image_tokens: usize, params: Params) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{}: {e}", tokenizer.display()))?;
        // Questions are text: `<|im_end|>` typed by a user stays letters.
        tokenizer.set_encode_special_tokens(true);
        let vision = GpuVision::load(dev, &Vision::load(mmproj)?, max_image_tokens)
            .with_context(|| format!("loading {}", mmproj.display()))?;
        let mut gpu = GpuQwen3::load(dev, model).with_context(|| format!("loading {}", model.display()))?;
        ensure!(vision.out_width() == gpu.c.d_model, "the mmproj writes {} wide rows, the model takes {}", vision.out_width(), gpu.c.d_model);
        ensure!(vision.deepstack_levels() == gpu.c.deepstack, "{} DeepStack levels in the mmproj, {} in the model", vision.deepstack_levels(), gpu.c.deepstack);
        gpu.activate(Self::n_ctx(max_image_tokens, &params))?;
        Ok(Self { tokenizer, vision: Mutex::new(vision), gpu: Mutex::new(gpu), max_image_tokens, params })
    }

    fn n_ctx(max_image_tokens: usize, params: &Params) -> usize {
        // The template's own tokens are a handful; 32 covers them.
        (max_image_tokens + MAX_PROMPT_TOKENS + params.max_new_tokens + 32).next_multiple_of(256)
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// The question's tokens, checked against [`MAX_PROMPT_TOKENS`].
    pub fn prompt_tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let prompt = if prompt.trim().is_empty() { DEFAULT_PROMPT } else { prompt.trim() };
        let ids = self.encode(prompt)?;
        ensure!(ids.len() <= MAX_PROMPT_TOKENS, "the question is {} tokens, more than {MAX_PROMPT_TOKENS}", ids.len());
        Ok(ids)
    }

    /// The tower on `picture`; `stop` turning true ends it early with an error.
    pub fn features(&self, picture: &Rgb, stop: &dyn Fn() -> bool) -> Result<Features> {
        let patches = vision::preprocess(picture, self.max_image_tokens)?;
        vision::check_tokens(&patches, self.max_image_tokens)?;
        let mut tower = self.vision.lock().map_err(|_| anyhow!("the describe tower lock is poisoned"))?;
        tower.features(&patches, stop)?.context("stopped")
    }

    /// Describes `picture`, or answers `prompt` about it. `stop` turning true
    /// ends it early with an error.
    pub fn describe(&self, picture: &Rgb, prompt: &str, stop: &dyn Fn() -> bool) -> Result<Description> {
        let question = self.prompt_tokens(prompt)?;
        let start = Instant::now();
        let f = self.features(picture, stop)?;
        let vision_secs = start.elapsed().as_secs_f32();
        let mut d = self.answer(&f, &question, stop)?;
        d.vision_secs = vision_secs;
        Ok(d)
    }

    /// The user turn around a `gh × gw` picture and `question`, up to the
    /// assistant's first token, with each token's 3D position.
    pub fn turn(&self, grid: (usize, usize), question: &[u32]) -> Result<Turn> {
        let (gh, gw) = grid;
        let n = gh * gw;
        let head: Vec<u32> = [IM_START].into_iter().chain(self.encode("user\n")?).chain([VISION_START]).collect();
        let tail: Vec<u32> = [VISION_END]
            .into_iter()
            .chain(question.iter().copied())
            .chain([IM_END])
            .chain(self.encode("\n")?)
            .chain([IM_START])
            .chain(self.encode("assistant\n")?)
            .collect();
        let ids: Vec<u32> = head.iter().copied().chain(std::iter::repeat_n(IMAGE_PAD, n)).chain(tail.iter().copied()).collect();

        // Text at (p, p, p); the picture at (s, s + row, s + column); after it,
        // text resumes at s + the grid's longer side.
        let s = head.len() as u32;
        let mut pos: Vec<Pos> = (0..s).map(|p| [p; 3]).collect();
        pos.extend((0..n as u32).map(|i| [s, s + i / gw as u32, s + i % gw as u32]));
        let mut next = s + gh.max(gw) as u32;
        for _ in &tail {
            pos.push([next; 3]);
            next += 1;
        }
        Ok(Turn { ids, pos, first: head.len(), next })
    }

    /// The logits after the turn around `f`'s rows (for checking against a reference).
    pub fn prefill(&self, f: &Features, question: &[u32]) -> Result<Vec<f32>> {
        let n = f.grid.0 * f.grid.1;
        let Turn { ids, pos, first, .. } = self.turn(f.grid, question)?;
        let mut gpu = self.gpu.lock().map_err(|_| anyhow!("the describe GPU lock is poisoned"))?;
        let d = gpu.c.d_model;
        let mut x = gpu.embed(&ids)?;
        x[first * d..(first + n) * d].copy_from_slice(&f.main);
        gpu.forward(&x, &pos, Some((first, &f.deepstack)), 0, &|| false)?.context("stopped")
    }

    /// The language model's half: the turn around `f`'s rows, then the answer.
    pub fn answer(&self, f: &Features, question: &[u32], stop: &dyn Fn() -> bool) -> Result<Description> {
        let n = f.grid.0 * f.grid.1;
        let Turn { ids, pos, first, mut next } = self.turn(f.grid, question)?;
        let mut gpu = self.gpu.lock().map_err(|_| anyhow!("the describe GPU lock is poisoned"))?;
        let d = gpu.c.d_model;
        let mut x = gpu.embed(&ids)?;
        x[first * d..(first + n) * d].copy_from_slice(&f.main);
        let budget = self.params.max_new_tokens.min(Self::n_ctx(self.max_image_tokens, &self.params) - ids.len());

        let start = Instant::now();
        let mut logits = gpu.forward(&x, &pos, Some((first, &f.deepstack)), 0, stop)?.context("stopped")?;
        let prefill_secs = start.elapsed().as_secs_f32();

        let start = Instant::now();
        let sampler_params = SamplerParams {
            temperature: self.params.temperature,
            top_k: self.params.top_k,
            top_p: self.params.top_p,
            min_p: 0.0,
            repeat: None,
            ..SamplerParams::default()
        };
        let mut sampler = Sampler::new(sampler_params, rand::random());
        let mut out = Vec::new();
        let mut past = ids.len();
        let mut truncated = true;
        while out.len() < budget {
            let id = sampler.sample(&mut logits, &out, out.len());
            if id == IM_END || id == END_OF_TEXT {
                truncated = false;
                break;
            }
            out.push(id);
            if out.len() == budget {
                break;
            }
            let x = gpu.embed(&[id])?;
            logits = gpu.forward(&x, &[[next; 3]], None, past, stop)?.context("stopped")?;
            past += 1;
            next += 1;
        }
        let decode_secs = start.elapsed().as_secs_f32();
        drop(gpu);

        let text = self.tokenizer.decode(&out, true).map_err(|e| anyhow!("decoding: {e}"))?;
        Ok(Description {
            text: text.trim().to_string(),
            generated: out.len(),
            tokens: out,
            image_tokens: n,
            grid: f.grid,
            prompt_tokens: ids.len(),
            truncated,
            vision_secs: 0.0,
            prefill_secs,
            decode_secs,
        })
    }
}
