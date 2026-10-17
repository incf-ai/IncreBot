//! The real model behind the room: increbot's inference thread (design.md §4, §5)
//! with the same job settings the Discord dispatcher uses.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail, ensure};
use increbot::config::Config;
use increbot::engine::Backends;
use increbot::persona::{Active, Persona};
use increbot::engine::worker::{self, CacheKey, EngineHandle, Event, Job};
use increbot::gguf::Gguf;
use increbot::gpu::manager::Gpus;
use increbot::model::Llama;
use increbot::tokenizer::Tokenizer;
use increbot::{hub, runtime};
use tokio::sync::mpsc::{self, error::TryRecvError};

use crate::backend::{Backend, BackendEvent, ReplyRequest, ReplyStats};
use crate::room::ROOM_CHANNEL;

struct Running {
    job: u64,
    prompt_tokens: usize,
    events: mpsc::UnboundedReceiver<Event>,
}

pub struct EngineBackend {
    engine: EngineHandle,
    tok: Arc<Tokenizer>,
    /// The persona's sampler, word biases and reply length, with the tables
    /// built from the vocabulary (the room builds the prompt itself).
    persona: Active,
    timeout: Duration,
    label: String,
    running: Option<Running>,
}

impl EngineBackend {
    /// Wraps an already loaded model (and GPUs, if any). `kv_slots` parked caches
    /// are kept besides the active one (`runtime.kv_cache_slots`); `timeout` is
    /// per reply (`generation.timeout_secs`).
    pub fn new(
        backends: Backends,
        tok: Arc<Tokenizer>,
        persona: Persona,
        timeout: Duration,
        kv_slots: usize,
        label: String,
    ) -> Result<Self> {
        // The template and names only matter for the prompt, which the room builds.
        let persona = Active::new(persona, increbot::config::DEFAULT_TEMPLATE, "bot", &[], &tok)?;
        Ok(Self {
            engine: worker::spawn(backends, tok.clone(), kv_slots)?,
            tok,
            persona,
            timeout,
            label,
            running: None,
        })
    }

    /// Loads the model named in `cfg` the way `increbot serve` does: compute
    /// pool, GGUF, tokenizer, weights, then a one-token self-test. With `gpu`,
    /// also the GPUs in `runtime.gpus` (gpu-plan.md), which take over once their
    /// upload finishes. Replies use `persona`'s sampler settings. `progress`
    /// hears about each step (loading the 13B takes a while).
    pub fn load(cfg: &Config, persona: Persona, gpu: bool, progress: &mut dyn FnMut(&str)) -> Result<Self> {
        persona.check_against(cfg)?;
        ensure!(
            !gpu || !cfg.runtime.gpus.is_empty(),
            "--gpu needs runtime.gpus in the config (or INCREBOT_GPUS)"
        );
        progress(&format!(
            "starting {} compute threads (pinned: {})",
            cfg.runtime.threads, cfg.runtime.pin_threads
        ));
        runtime::init_pool(cfg.runtime.threads, cfg.runtime.pin_threads)?;
        progress(&format!("locating {}/{}", cfg.model.repo, cfg.model.file));
        let gguf = Gguf::open(&hub::resolve(&cfg.model.repo, &cfg.model.file)?)?;
        let tok = Arc::new(Tokenizer::load(
            &hub::resolve(&cfg.model.tokenizer_repo, "tokenizer.json")?,
            &gguf,
        )?);
        progress(&format!("loading weights ({:?})", cfg.model.linear_impl));
        let mut model = Llama::load(&gguf, &Default::default(), cfg.model.linear_impl)?;
        model.set_kv_f16(cfg.runtime.kv_cache_f16);
        drop(gguf);
        ensure!(
            cfg.model.n_ctx <= model.config.n_ctx,
            "n_ctx {} exceeds the model's context {}",
            cfg.model.n_ctx,
            model.config.n_ctx
        );
        progress("self-test");
        let logits = model
            .forward(&[tok.bos], &mut model.new_cache())
            .context("self-test forward")?;
        ensure!(
            logits.iter().all(|l| l.is_finite()),
            "self-test forward produced non-finite logits"
        );
        let model = Arc::new(model);
        let gpus = if gpu {
            progress("starting the GPUs (weights upload in the background)");
            Gpus::start(cfg.runtime.gpu_settings(cfg.model.n_ctx)?, model.clone())
        } else {
            None
        };
        let label = format!(
            "{} ({:?}, {} threads{})",
            cfg.model.file,
            cfg.model.linear_impl,
            cfg.runtime.threads,
            match &gpus {
                Some(g) => format!(", {} GPU(s)", g.cards.len()),
                None => String::new(),
            }
        );
        Self::new(
            Backends::new(model, gpus, cfg.model.n_ctx),
            tok,
            persona,
            Duration::from_secs(cfg.generation.timeout_secs),
            cfg.runtime.kv_cache_slots,
            label,
        )
    }

    fn drain_idle(&mut self) {
        // The inference thread blocks on this channel after each job until read.
        while self.engine.idle.try_recv().is_ok() {}
    }
}

impl Backend for EngineBackend {
    fn describe(&self) -> String {
        self.label.clone()
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tok
            .encode_prompt(text)
            .map_or(usize::MAX, |ids| ids.len())
    }

    fn start(&mut self, req: ReplyRequest) -> Result<()> {
        if self.running.is_some() {
            bail!("a reply is already being generated");
        }
        self.drain_idle();
        let prompt = self.tok.encode_prompt(&req.prompt)?;
        // The penalties see the chat history, not the persona and template
        // around it, as in the bot (design.md §5).
        let history = self.tok.encode_prompt(&req.history)?.split_off(1);
        let p = &self.persona.persona;
        // The persona's phrase bans, speaker labels and the echo of the last
        // reply, as in the bot's dispatcher. The simulator has no reply counts
        // per push, so a `for_replies` ban applies while its phrase is in the
        // unanswered messages (a one-shot's rule).
        let bans = {
            let replies = increbot::persona::one_shot_replies(&req.message);
            let names = req
                .stops
                .iter()
                .filter_map(|s| s.strip_prefix('\n').and_then(|s| s.strip_suffix(':')).map(String::from));
            let bans = self
                .persona
                .bans(&req.message, None, &replies)
                .with_prefixes(names)
                .with_echo(req.previous.as_deref().unwrap_or(""));
            (!bans.is_empty()).then(|| Arc::new(bans))
        };
        let (tx, rx) = mpsc::unbounded_channel();
        let job = Job {
            key: if req.scratch {
                CacheKey::Scratch
            } else {
                CacheKey::Channel(ROOM_CHANNEL)
            },
            prompt: prompt.clone(),
            sampler: self.persona.sampler(&req.message),
            seed: rand::random(),
            history,
            max_new_tokens: p.limits(None).max_new_tokens,
            max_new_tokens_gpu: p.limits(None).max_new_tokens_gpu,
            wrap_up_after: p.limits(None).wrap_up_after,
            wrap_up_after_gpu: p.limits(None).wrap_up_after_gpu,
            stops: req.stops,
            max_lines: p.limits(None).max_lines,
            timeout: self.timeout,
            cancel: req.cancel,
            events: tx,
            best_of: std::num::NonZeroUsize::MIN,
            score: None,
            bans,
        };
        self.engine
            .jobs
            .try_send(job)
            .map_err(|e| anyhow::anyhow!("inference thread unavailable: {e}"))?;
        self.running = Some(Running {
            job: req.job,
            prompt_tokens: prompt.len(),
            events: rx,
        });
        Ok(())
    }

    fn poll(&mut self) -> Vec<(u64, BackendEvent)> {
        self.drain_idle();
        let Some(r) = self.running.as_mut() else {
            return Vec::new();
        };
        let job = r.job;
        let mut out = Vec::new();
        let done = loop {
            match r.events.try_recv() {
                // The simulator has no channel rate limits: never sent.
                Ok(Event::Paused(_)) => {}
                Ok(Event::Started) => out.push((job, BackendEvent::Started)),
                Ok(Event::Text(t)) => out.push((job, BackendEvent::Text(t))),
                Ok(Event::Finished(o)) => {
                    out.push((
                        job,
                        BackendEvent::Finished(ReplyStats {
                            stop: o.stop,
                            prompt_tokens: r.prompt_tokens,
                            reused_tokens: o.reused_tokens,
                            generated_tokens: o.tokens.len(),
                            prefill: o.prefill,
                            decode: o.decode,
                        }),
                    ));
                    break true;
                }
                Ok(Event::Failed(e)) => {
                    out.push((job, BackendEvent::Failed(e)));
                    break true;
                }
                Err(TryRecvError::Empty) => break false,
                Err(TryRecvError::Disconnected) => {
                    out.push((
                        job,
                        BackendEvent::Failed("the inference thread stopped".into()),
                    ));
                    break true;
                }
            }
        };
        if done {
            self.running = None;
        }
        out
    }

    fn reset(&mut self) {
        self.engine.reset(ROOM_CHANNEL);
    }
}
