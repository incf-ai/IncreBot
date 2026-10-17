//! Where a forward pass runs (gpu-plan.md §4–5): a ladder of backends, the
//! GPUs in preference order and then the CPU.
//!
//! A job starts on the most preferred usable GPU. When that card is taken by
//! another program mid-reply, the forward stops at the next layer, and the same
//! reply continues on the next backend in line from where it stopped: the host
//! KV cache already holds every finished chunk (gpu/model.rs), so nothing is
//! copied and no prompt is re-read beyond the interrupted chunk. A job never
//! moves back up the ladder; the next job starts at the top again.

use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info};

use crate::gpu::manager::{Card, Gpus};
use crate::gpu::model::Interrupted;
use crate::model::{KvCache, Llama};

/// A forward pass the generation loop runs: logits for the last of `tokens`
/// after what `cache` holds, or `None` when `cancelled` fired.
pub trait Forward {
    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut KvCache,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<Vec<f32>>>;

    /// Whether the last forward ran on a GPU (for `max_new_tokens_gpu`).
    fn on_gpu(&self) -> bool {
        false
    }
}

impl Forward for &Llama {
    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut KvCache,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<Vec<f32>>> {
        self.forward_cancellable(tokens, cache, cancelled)
    }
}

pub struct Backends {
    pub cpu: Arc<Llama>,
    pub gpus: Option<Arc<Gpus>>,
    n_ctx: usize,
    /// The current job's place on the ladder: cards before it are not used
    /// again; `cards.len()` means the CPU.
    floor: usize,
    /// The card holding the KV cache and scratch buffers.
    active: Option<usize>,
    /// Where the current job ran, in order.
    route: Vec<String>,
}

impl Backends {
    /// The CPU only.
    pub fn cpu_only(cpu: Llama) -> Self {
        Self::new(Arc::new(cpu), None, 0)
    }

    /// `gpus` (if any) in front of the CPU. GPUs need the CPU model's caches in
    /// f16; `n_ctx` is the KV cache size allocated on the active card.
    pub fn new(cpu: Arc<Llama>, gpus: Option<Arc<Gpus>>, n_ctx: usize) -> Self {
        assert!(
            gpus.is_none() || cpu.kv_f16(),
            "the GPU backend needs f16 KV caches (runtime.kv_cache_f16)"
        );
        Self {
            cpu,
            gpus,
            n_ctx,
            floor: 0,
            active: None,
            route: Vec::new(),
        }
    }

    /// Resets the ladder for a new job.
    pub fn start_job(&mut self) {
        self.floor = 0;
        self.route.clear();
    }

    /// Where the last job ran: `"gpu 0000:06:00.0"`, `"cpu"`, or a migration
    /// such as `"gpu 0000:06:00.0 → cpu"`.
    pub fn route(&self) -> String {
        if self.route.is_empty() {
            return "none".into();
        }
        self.route.join(" → ")
    }

    fn visit(&mut self, name: String) {
        if self.route.last() != Some(&name) {
            self.route.push(name);
        }
    }

    /// Frees the KV cache and scratch of the previously active card, if `i`
    /// takes over.
    fn make_active(&mut self, cards: &[Arc<Card>], i: usize) {
        if let Some(j) = self.active.filter(|&j| j != i) {
            if let Some(m) = cards[j].model().as_mut() {
                m.deactivate();
            }
            cards[j].set_active(false);
        }
        self.active = Some(i);
    }
}

impl Forward for &mut Backends {
    fn on_gpu(&self) -> bool {
        self.route.last().is_some_and(|r| r.starts_with("gpu"))
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut KvCache,
        cancelled: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<Vec<f32>>> {
        let b = &mut **self;
        let past = cache.len();
        let cards = b.gpus.as_ref().map(|g| g.cards.clone()).unwrap_or_default();
        while let Some(card) = cards.get(b.floor) {
            let i = b.floor;
            if !card.usable() {
                b.floor += 1;
                continue;
            }
            b.make_active(&cards, i);
            let mut slot = card.model();
            let Some(model) = slot.as_mut() else {
                b.floor += 1;
                continue;
            };
            if let Err(e) = model.activate(b.n_ctx) {
                error!(gpu = %card.label, error = format!("{e:#}"), "allocating the KV cache failed");
                drop(slot);
                card.fail();
                b.floor += 1;
                continue;
            }
            card.set_active(true);
            b.visit(format!("gpu {}", card.label));
            let rest = &tokens[cache.len() - past..];
            let stop = || cancelled() || card.preempted();
            match model.forward(&b.cpu, rest, cache, &stop) {
                Ok(Ok(logits)) => return Ok(Some(logits)),
                Ok(Err(Interrupted::Stopped)) if cancelled() => return Ok(None),
                Ok(Err(Interrupted::Stopped)) => {
                    info!(
                        gpu = %card.label,
                        kept = cache.len() - past,
                        left = tokens.len() - (cache.len() - past),
                        "preempted mid-reply; continuing on the next backend"
                    );
                }
                Err(e) => {
                    error!(gpu = %card.label, error = format!("{e:#}"), "GPU forward failed; continuing on the next backend");
                    drop(slot);
                    card.fail();
                }
            }
            b.floor += 1;
        }
        b.visit("cpu".into());
        b.cpu
            .forward_cancellable(&tokens[cache.len() - past..], cache, cancelled)
    }
}
