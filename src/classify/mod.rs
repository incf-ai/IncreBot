//! System One classification with LiquidAI's d1 models: a state (any JSON, or
//! text) and questions in the Decision Index's schema go in; each question's
//! option probabilities come out of one forward pass that stops at the answer
//! slot. `prompt` and `tokenizer` are ported from d1-burn (same author, MIT).
//!
//! Questions on one state share its prefix: the trunk runs once and each
//! question continues a clone of its [`Lfm2State`].

pub mod image;
pub mod vision;
pub mod prompt;
pub mod tokenizer;

use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use tracing::info;

use crate::gguf::Gguf;
use crate::gpu::{Device, GpuLfm2, PciAddr};
use crate::model::{LinearImpl, Lfm2, Lfm2State};
pub use prompt::{Json, Question, Renderer};
pub use tokenizer::Tokenizer;

/// Hugging Face repos of the default model and its tokenizer.
pub const DEFAULT_REPO: &str = "LiquidAI/d1-3B-GGUF";
pub const DEFAULT_FILE: &str = "d1-3B-Q4_K_M.gguf";
/// The SigLIP2 vision tower and projector, in [`DEFAULT_REPO`].
pub const DEFAULT_MMPROJ: &str = "mmproj-d1-3B-Q8_0.gguf";
pub const TOKENIZER_REPO: &str = "LiquidAI/d1-3B";

/// The prompt module's error (kept from d1-burn so the port stays close to it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// One question's result.
#[derive(Debug, Clone)]
pub struct Decision {
    pub name: String,
    /// One per option (noul: yes, no; score: per level), summing to 1.
    pub probabilities: Vec<f64>,
    /// `api.answer`'s JSON: P(yes), or the pick with its confidence.
    pub answer: Value,
}

pub struct Classifier {
    model: Lfm2,
    tokenizer: Tokenizer,
    renderer: Renderer,
    /// Set by [`Classifier::with_gpu`]: runs every forward there.
    gpu: Option<Mutex<GpuLfm2>>,
    /// Set by [`Classifier::with_vision`]: the picture tower for `classify_image`.
    vision: Option<vision::Vision>,
}

impl Classifier {
    /// Loads a d1 GGUF and its `tokenizer.json`; `max_ctx` caps the context.
    pub fn load(
        gguf: &Path,
        tokenizer_json: &Path,
        linear_impl: LinearImpl,
        max_ctx: usize,
    ) -> Result<Self> {
        let g = Gguf::open(gguf).with_context(|| format!("opening {}", gguf.display()))?;
        let model = Lfm2::load(&g, &Default::default(), linear_impl, max_ctx)?;
        let bos_id = g.get_u64("tokenizer.ggml.bos_token_id")? as u32;
        tokenizers::utils::parallelism::set_parallelism(false);
        let probe = Tokenizer::from_file(tokenizer_json, "")?;
        let bos = probe
            .id_to_token(bos_id)
            .with_context(|| format!("BOS id {bos_id} is not in {}", tokenizer_json.display()))?;
        let tokenizer = Tokenizer::from_file(tokenizer_json, &bos)?;
        ensure!(
            tokenizer.vocab_size() <= model.config.vocab,
            "tokenizer has {} tokens, the model {}",
            tokenizer.vocab_size(),
            model.config.vocab
        );
        let renderer = Renderer::new(&tokenizer);
        info!(bos, "classifier ready");
        Ok(Self {
            model,
            tokenizer,
            renderer,
            gpu: None,
            vision: None,
        })
    }

    /// Moves the forward passes to the GPU at `pci` (needs Q4_K/Q6_K weights
    /// and `LinearImpl::Packed`).
    pub fn with_gpu(mut self, pci: PciAddr) -> Result<Self> {
        let dev = Device::open(pci)?;
        self.gpu = Some(Mutex::new(GpuLfm2::load(&dev, &self.model)?));
        Ok(self)
    }

    /// Loads the vision tower and projector from an `mmproj` GGUF
    /// ([`DEFAULT_MMPROJ`]), for [`Classifier::classify_image`].
    pub fn with_vision(mut self, mmproj: &Path) -> Result<Self> {
        let v = vision::Vision::load(mmproj)?;
        ensure!(
            v.out_width() == self.model.config.d_model,
            "the projector writes {} wide, the model takes {}",
            v.out_width(),
            self.model.config.d_model
        );
        self.vision = Some(v);
        info!(mmproj = %mmproj.display(), "classifier vision ready");
        Ok(self)
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The token rows `classify` would run, one per question.
    pub fn rows(&self, state: Option<&Json>, questions: &[Question]) -> Result<Vec<Vec<u32>>> {
        Ok(self.renderer.rows(&self.tokenizer, state, questions)?)
    }

    /// Answers each `(name, question)` about `pictures` and `state`, the
    /// pictures leading the user turn as d1-burn renders them. Runs on the CPU
    /// even with a GPU set: the GPU forward doesn't take image features yet.
    pub fn classify_image(
        &self,
        pictures: &[::image::RgbImage],
        state: Option<&Json>,
        questions: &[(String, Question)],
    ) -> Result<Vec<Decision>> {
        let tower = self.vision.as_ref().context("the classifier has no vision tower")?;
        ensure!(!questions.is_empty(), "no questions");
        ensure!(!pictures.is_empty(), "no pictures");
        let config = vision::Config::default();
        let processed: Vec<vision::Processed> = pictures
            .iter()
            .map(|p| {
                let rgb = vision::Rgb::from_image(p)?;
                Ok(config.process(&vision::cap_pixels(rgb, vision::MAX_PIXELS)))
            })
            .collect::<Result<_>>()?;
        let layouts: Vec<vision::Layout> = processed.iter().map(|p| p.layout).collect();
        let qs: Vec<Question> = questions.iter().map(|(_, q)| q.clone()).collect();
        let rows = vision::rows(&self.renderer, &self.tokenizer, &layouts, state, &qs)?;
        let features = tower.features(&processed);
        let trunk = trunk_len(&rows);
        let image = vision::IMAGE_TOKEN_ID;
        ensure!(
            rows.iter().all(|r| !r[trunk..].contains(&image)),
            "image tokens past the shared prefix"
        );
        let mut base = self.model.new_state();
        if trunk > 0 {
            self.model.forward_with_images(&rows[0][..trunk], &mut base, image, &features)?;
        }
        let logits: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                let mut state: Lfm2State = base.clone();
                self.model.forward(&row[trunk..], &mut state)
            })
            .collect::<Result<_>>()?;
        self.decisions(questions, &logits)
    }

    /// Answers each `(name, question)` about `state` (`None`: no state).
    pub fn classify(
        &self,
        state: Option<&Json>,
        questions: &[(String, Question)],
    ) -> Result<Vec<Decision>> {
        ensure!(!questions.is_empty(), "no questions");
        let qs: Vec<Question> = questions.iter().map(|(_, q)| q.clone()).collect();
        let rows = self.rows(state, &qs)?;

        let trunk = trunk_len(&rows);
        let logits: Vec<Vec<f32>> = match &self.gpu {
            Some(gpu) => {
                let branches: Vec<&[u32]> = rows.iter().map(|r| &r[trunk..]).collect();
                let mut gpu = gpu.lock().unwrap_or_else(|e| e.into_inner());
                gpu.branches(&self.model, &rows[0][..trunk], &branches)?
            }
            None => {
                let mut base = self.model.new_state();
                if trunk > 0 {
                    self.model.forward(&rows[0][..trunk], &mut base)?;
                }
                rows.iter()
                    .map(|row| {
                        let mut state: Lfm2State = base.clone();
                        self.model.forward(&row[trunk..], &mut state)
                    })
                    .collect::<Result<_>>()?
            }
        };
        self.decisions(questions, &logits)
    }

    /// Each question's readout from its last-position logits.
    fn decisions(&self, questions: &[(String, Question)], logits: &[Vec<f32>]) -> Result<Vec<Decision>> {
        let mut out = Vec::with_capacity(questions.len());
        for ((name, q), logits) in questions.iter().zip(logits) {
            let probabilities = prompt::readout(&self.tokenizer, q, &log_softmax(logits))?;
            out.push(Decision {
                name: name.clone(),
                answer: prompt::answer(q, &probabilities),
                probabilities,
            });
        }
        Ok(out)
    }
}

/// The rows' shared trunk, leaving every branch at least its last token.
fn trunk_len(rows: &[Vec<u32>]) -> usize {
    let shortest = rows.iter().map(Vec::len).min().unwrap_or(0);
    let mut trunk = rows[0].len().min(shortest.saturating_sub(1));
    for row in &rows[1..] {
        trunk = trunk.min(row.iter().zip(&rows[0]).take_while(|(a, b)| a == b).count());
    }
    trunk
}

/// Parses `{name: question, ...}` in order.
pub fn parse_questions(text: &str) -> Result<Vec<(String, Question)>> {
    match Json::parse(text)? {
        Json::Object(items) => items
            .iter()
            .map(|(name, q)| {
                Ok((
                    name.clone(),
                    Question::from_json(q).with_context(|| format!("question {name:?}"))?,
                ))
            })
            .collect(),
        _ => anyhow::bail!("questions must be a JSON object of name -> question"),
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + logits.iter().map(|&x| ((x - m) as f64).exp()).sum::<f64>().ln() as f32;
    logits.iter().map(|&x| x - lse).collect()
}
