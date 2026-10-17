//! Where replies come from: the interface between the room and a generator.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use anyhow::Result;
pub use increbot::engine::StopReason;

/// One reply for a backend to generate, built by the room when the job is
/// dispatched (so a merged job sees every message up to then, as on Discord).
#[derive(Clone, Debug)]
pub struct ReplyRequest {
    /// Identifies the job in [`Backend::poll`] results.
    pub job: u64,
    /// The full rendered prompt, ending in `"<bot name>:"`.
    pub prompt: String,
    pub stops: Vec<String>,
    /// A one-shot `/ask`: run in a fresh cache that never evicts the channel's.
    pub scratch: bool,
    /// The chat history in the prompt, as text, for the repetition penalty's
    /// window (design.md §5).
    pub history: String,
    /// The user messages the reply answers, for `prompted_by` word rules.
    pub message: String,
    /// The bot's reply before the trigger, if any: its echo is banned, as in
    /// the bot (design.md §5); `None` for a one-shot.
    pub previous: Option<String>,
    /// Set to stop generation (Esc, `/cancel`, quitting).
    pub cancel: Arc<AtomicBool>,
    /// Who and what the reply answers (for display and for scripted replies).
    pub trigger_author: String,
    pub trigger_text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplyStats {
    pub stop: StopReason,
    pub prompt_tokens: usize,
    /// Prompt tokens served from the KV cache.
    pub reused_tokens: usize,
    pub generated_tokens: usize,
    pub prefill: Duration,
    pub decode: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BackendEvent {
    /// Generation began (prefill is running).
    Started,
    /// Settled text to append to the reply.
    Text(String),
    Finished(ReplyStats),
    Failed(String),
}

/// A reply generator. It runs one job at a time; the app starts the next job
/// only after the previous one finished or failed.
pub trait Backend {
    /// A short label for the status bar and `/stats`.
    fn describe(&self) -> String;

    /// Prompt tokens in `text`, BOS included (used for the history budget).
    fn count_tokens(&self, text: &str) -> usize;

    /// Starts a job. Errors if one is still running.
    fn start(&mut self, req: ReplyRequest) -> Result<()>;

    /// Events since the last call, tagged with their job id. Never blocks.
    fn poll(&mut self) -> Vec<(u64, BackendEvent)>;

    /// Drops the channel's KV cache before the next job (`/reset`).
    fn reset(&mut self);
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn describe(&self) -> String {
        (**self).describe()
    }
    fn count_tokens(&self, text: &str) -> usize {
        (**self).count_tokens(text)
    }
    fn start(&mut self, req: ReplyRequest) -> Result<()> {
        (**self).start(req)
    }
    fn poll(&mut self) -> Vec<(u64, BackendEvent)> {
        (**self).poll()
    }
    fn reset(&mut self) {
        (**self).reset()
    }
}
