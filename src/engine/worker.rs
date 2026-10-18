//! The inference thread (design.md §4, §5): owns the model and the KV caches and
//! runs one job at a time. The queue lives on the async side; the dispatcher
//! sends a job only after the engine reports idle, so prompts are built late.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tracing::{error, info};

use super::backend::Backends;
use super::sampler::{Sampler, SamplerParams};
use super::{GenerateOutput, GenerateRequest, StopReason, generate};
use crate::gpu::manager::Gpus;
use crate::model::{KvCache, Llama, ParkedKv};
use crate::tokenizer::Tokenizer;

/// Which KV cache a job runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheKey {
    /// A channel or thread's conversation; kept between jobs.
    Channel(u64),
    /// One-shot prompts (`/ask`): a fresh cache, discarded after the job, so they
    /// never evict a channel's cache (design.md §5).
    Scratch,
}

pub struct Job {
    pub key: CacheKey,
    pub prompt: Vec<u32>,
    pub sampler: SamplerParams,
    pub seed: u64,
    /// See [`GenerateRequest::history`].
    pub history: Vec<u32>,
    /// 0 here and in `max_new_tokens_gpu`: only prefill the prompt into the
    /// cache, ahead of a reply (`discord::dispatch::build_warm`); not counted
    /// in the stats.
    pub max_new_tokens: usize,
    /// The limit while the reply runs on a GPU.
    pub max_new_tokens_gpu: usize,
    /// See [`GenerateRequest::wrap_up_after`].
    pub wrap_up_after: usize,
    pub wrap_up_after_gpu: usize,
    pub stops: Vec<String>,
    /// See [`GenerateRequest::max_lines`].
    pub max_lines: Option<std::num::NonZeroUsize>,
    /// See [`GenerateRequest::labels`].
    pub labels: bool,
    /// For the whole job, all candidates together.
    pub timeout: Duration,
    pub cancel: Arc<AtomicBool>,
    pub events: mpsc::UnboundedSender<Event>,
    /// Generate this many candidates (seeds `seed`, `seed + 1`, …) and send
    /// only the one `score` rates best; nothing streams until it is chosen.
    /// 1 = a normal streamed reply.
    pub best_of: std::num::NonZeroUsize,
    pub score: Option<crate::persona::Scorer>,
    /// Phrases rolled back and resampled (`engine::bans`); `None` = off.
    pub bans: Option<Arc<crate::engine::bans::Bans>>,
}

#[derive(Debug)]
pub enum Event {
    /// Generation began (after any queue wait).
    Started,
    /// Settled text to append to the reply.
    Text(String),
    Finished(GenerateOutput),
    Failed(String),
    /// Not from the engine: the dispatcher is holding this reply back for its
    /// channel's pause until then (`discord::dispatch`).
    Paused(std::time::Instant),
}

/// Rolling numbers for `/stats`.
#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    pub jobs: u64,
    pub prompt_tokens: u64,
    pub reused_tokens: u64,
    pub last_prefill_tok_s: f64,
    pub last_decode_tok_s: f64,
    /// Prompt tokens the last reply prefilled, and how many came from the cache.
    pub last_prefill_tokens: usize,
    pub last_reused_tokens: usize,
    /// Tokens the last reply generated.
    pub last_decode_tokens: usize,
    /// Where the last reply ran (`Backends::route`): `"cpu"`, `"gpu 0000:06:00.0"`,
    /// or a migration such as `"gpu 0000:06:00.0 → cpu"`.
    pub last_backend: String,
    /// The last reply's prefill and decode time (set with the rates above),
    /// and its whole time on the engine: cache swap, prefill and decode, from
    /// when the job started (queue wait not included).
    pub last_prefill: Duration,
    pub last_decode: Duration,
    pub last_total: Duration,
    /// The last [`RECENT`] replies of each kind, for medians in `/stats`.
    pub recent: Recent,
    /// The seed prefill (`SeedHandle`): "873 tokens on gpu 0000:06:00.0 in
    /// 1.2 s", "waiting for the GPUs", or empty when none was asked for.
    pub seed: String,
}

/// How many replies of each kind [`Recent`] keeps.
pub const RECENT: usize = 20;

/// One finished reply's timing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub total: Duration,
    pub prefill: Duration,
    /// Decode speed; `None` when it made fewer than 2 tokens.
    pub decode_tok_s: Option<f64>,
}

/// Recent reply timings by where they ran. GPU and CPU speeds differ by an
/// order of magnitude, so they're never mixed; a reply that moved between
/// them mid-way is kept apart too.
#[derive(Clone, Debug, Default)]
pub struct Recent {
    pub gpu: VecDeque<Timing>,
    pub cpu: VecDeque<Timing>,
    pub moved: VecDeque<Timing>,
}

impl Recent {
    /// Records a reply that ran on `route` (`Backends::route`).
    pub fn push(&mut self, route: &str, t: Timing) {
        let list = if route.contains('→') {
            &mut self.moved
        } else if route.starts_with("gpu") {
            &mut self.gpu
        } else {
            &mut self.cpu
        };
        if list.len() == RECENT {
            list.pop_front();
        }
        list.push_back(t);
    }
}

/// "total median 1.42 s, slowest 3.10 s; prefill median 0.31 s; decode median
/// 46.9 tok/s" over `times`; `None` if empty.
pub fn summarize(times: &VecDeque<Timing>) -> Option<String> {
    fn median(mut v: Vec<f64>) -> Option<f64> {
        v.sort_by(f64::total_cmp);
        (!v.is_empty()).then(|| {
            let n = v.len();
            if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
        })
    }
    let total: Vec<f64> = times.iter().map(|t| t.total.as_secs_f64()).collect();
    let slowest = total.iter().copied().fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))))?;
    let mut line = format!(
        "total median {:.2} s, slowest {slowest:.2} s; prefill median {:.2} s",
        median(total.clone())?,
        median(times.iter().map(|t| t.prefill.as_secs_f64()).collect())?
    );
    if let Some(d) = median(times.iter().filter_map(|t| t.decode_tok_s).collect()) {
        line += &format!("; decode median {d:.1} tok/s");
    }
    Some(line)
}

pub struct EngineHandle {
    pub jobs: mpsc::Sender<Job>,
    /// One message per finished job.
    pub idle: mpsc::Receiver<()>,
    pub stats: Arc<Mutex<EngineStats>>,
    pub resets: ResetHandle,
    /// The GPUs, for `/stats`.
    pub gpus: Option<Arc<Gpus>>,
    /// Asks for the persona header to be prefilled ahead of replies.
    pub seed: SeedHandle,
}

/// Asks the inference thread to prefill a prompt's shared start (the persona
/// header) into a seed cache that new channels and one-shot replies start
/// from, so their first prefill skips it (design.md §5). The thread waits
/// until the GPUs have settled: on the first ready card if one comes up, else
/// on the CPU. A newer request replaces an older one (persona reload).
#[derive(Clone)]
pub struct SeedHandle(mpsc::UnboundedSender<Vec<u32>>);

impl SeedHandle {
    pub fn request(&self, header: Vec<u32>) {
        let _ = self.0.send(header);
    }
}

/// How long the GPU managers get to start loading before the seed falls back
/// to the CPU, and the longest wait for a card that keeps loading.
const SEED_SETTLE: Duration = Duration::from_secs(5);
const SEED_MAX_WAIT: Duration = Duration::from_secs(15 * 60);

impl EngineHandle {
    /// Drops a channel's KV cache before the next job (`/super-admin reset`).
    pub fn reset(&self, channel: u64) {
        self.resets.reset(channel);
    }
}

/// Cloneable access to `/super-admin reset` without owning the engine handle.
#[derive(Clone, Default)]
pub struct ResetHandle(Arc<Mutex<Vec<u64>>>);

impl ResetHandle {
    pub fn reset(&self, channel: u64) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(channel);
    }

    fn take(&self) -> Vec<u64> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// Starts the inference thread. `kv_slots` parked caches are kept besides the
/// active one. Fails only if the OS won't hand out a thread (resource
/// exhaustion); there's exactly one inference thread and no fallback, so
/// that's fatal to `serve` and the caller should propagate it up rather than
/// panic (unlike a GPU manager thread, see `gpu::manager::Gpus::start`).
pub fn spawn(mut backends: Backends, tok: Arc<Tokenizer>, kv_slots: usize) -> anyhow::Result<EngineHandle> {
    use anyhow::Context as _;
    let gpus = backends.gpus.clone();
    let (jobs_tx, mut jobs_rx) = mpsc::channel::<Job>(1);
    let (idle_tx, idle_rx) = mpsc::channel(1);
    let (seed_tx, mut seed_rx) = mpsc::unbounded_channel::<Vec<u32>>();
    let stats = Arc::new(Mutex::new(EngineStats::default()));
    let resets = ResetHandle::default();
    let (stats2, resets2) = (stats.clone(), resets.clone());
    std::thread::Builder::new()
        .name("inference".into())
        .spawn(move || {
            // Only to wait on jobs, seed requests and the seed's GPU check at once.
            let rt = match tokio::runtime::Builder::new_current_thread().enable_time().build() {
                Ok(rt) => rt,
                Err(e) => {
                    // Dropping jobs_rx/idle_tx here closes the channels, which
                    // dispatch.rs's inference_thread_gone() treats as fatal and
                    // exits on — the same outcome a panic would reach, without
                    // the unwind.
                    error!(error = %e, "could not build the inference thread's timer runtime");
                    return;
                }
            };
            let mut caches = Caches::new(kv_slots);
            let mut pending: Option<(Vec<u32>, Instant)> = None;
            enum Next {
                Job(Job),
                Seed(Vec<u32>),
                Tick,
                Closed,
            }
            loop {
                if let Some((header, since)) = &pending {
                    if seed_ready(&backends, *since) {
                        seed(&mut backends, &mut caches, &stats2, header);
                        pending = None;
                    }
                }
                let waiting = pending.is_some();
                let next = rt.block_on(async {
                    tokio::select! {
                        job = jobs_rx.recv() => job.map_or(Next::Closed, Next::Job),
                        Some(h) = seed_rx.recv() => Next::Seed(h),
                        _ = tokio::time::sleep(Duration::from_millis(500)), if waiting => Next::Tick,
                    }
                });
                match next {
                    Next::Job(job) => {
                        for channel in resets2.take() {
                            caches.drop_channel(channel);
                        }
                        run_job(&mut backends, &tok, &mut caches, &stats2, job);
                        if idle_tx.blocking_send(()).is_err() {
                            break;
                        }
                    }
                    Next::Seed(h) => {
                        stats2.lock().unwrap_or_else(|e| e.into_inner()).seed = "waiting for the GPUs".into();
                        pending = Some((h, Instant::now()));
                    }
                    Next::Tick => {}
                    Next::Closed => break,
                }
            }
            info!("inference thread exiting");
        })
        .context("spawning the inference thread")?;
    Ok(EngineHandle {
        jobs: jobs_tx,
        idle: idle_rx,
        stats,
        resets,
        gpus,
        seed: SeedHandle(seed_tx),
    })
}

/// Whether to prefill the seed now: no GPUs; or a card is ready; or, after
/// [`SEED_SETTLE`], none is still loading (busy or failed: the CPU does it);
/// or [`SEED_MAX_WAIT`] has passed.
fn seed_ready(backends: &Backends, since: Instant) -> bool {
    let Some(gpus) = &backends.gpus else {
        return true;
    };
    if gpus.any_ready() {
        return true;
    }
    let loading = gpus.status().iter().any(|c| c.state == crate::gpu::policy::State::Loading);
    (since.elapsed() >= SEED_SETTLE && !loading) || since.elapsed() >= SEED_MAX_WAIT
}

/// Prefills `header` into a fresh cache and keeps it as the seed.
fn seed(backends: &mut Backends, caches: &mut Caches, stats: &Mutex<EngineStats>, header: &[u32]) {
    use super::backend::Forward;
    if header.is_empty() {
        return;
    }
    let started = Instant::now();
    let mut cache = backends.cpu.new_cache();
    backends.start_job();
    let mut b = &mut *backends;
    let result = b.forward(header, &mut cache, &|| false);
    let took = started.elapsed();
    let route = backends.route();
    let line = match result {
        Ok(Some(_)) => {
            caches.seed = Some(cache);
            info!(tokens = header.len(), backend = %route, secs = took.as_secs_f32(), "persona header prefilled (seed)");
            format!("{} tokens on {route} in {:.1} s", header.len(), took.as_secs_f32())
        }
        Ok(None) => "cancelled".into(),
        Err(e) => {
            error!(error = %e, "seed prefill failed; replies prefill from scratch");
            caches.seed = None;
            format!("failed: {e}")
        }
    };
    stats.lock().unwrap_or_else(|e| e.into_inner()).seed = line;
}

fn run_job(
    backends: &mut Backends,
    tok: &Tokenizer,
    caches: &mut Caches,
    stats: &Mutex<EngineStats>,
    job: Job,
) {
    // A job cancelled while queued never runs (design.md §5).
    if job.cancel.load(Ordering::Relaxed) {
        let _ = job.events.send(Event::Finished(GenerateOutput {
            tokens: Vec::new(),
            stop: StopReason::Cancelled,
            reused_tokens: 0,
            prefill_tokens: 0,
            prefill: Duration::ZERO,
            decode: Duration::ZERO,
            rollbacks: Vec::new(),
            steered: Vec::new(),
        }));
        return;
    }
    let _ = job.events.send(Event::Started);
    let started = Instant::now();
    let n = job.best_of.get();
    // Runs one candidate and reports whether generation stopped for good
    // (cancelled or timed out: trying another seed won't help). Pulled out
    // of the loop below so the first candidate can seed `best` directly,
    // with no `Option` (and so nothing to `.expect()`) in between.
    let mut run_one = |k: usize| -> (bool, i32, String, Result<GenerateOutput, anyhow::Error>) {
        let mut cache = caches.checkout(job.key, &backends.cpu);
        let mut sampler = Sampler::new(job.sampler.clone(), job.seed.wrapping_add(k as u64));
        let left = job.timeout.saturating_sub(started.elapsed());
        let req = GenerateRequest {
            prompt: &job.prompt,
            max_new_tokens: job.max_new_tokens,
            max_new_tokens_gpu: job.max_new_tokens_gpu,
            wrap_up_after: job.wrap_up_after,
            wrap_up_after_gpu: job.wrap_up_after_gpu,
            stops: job.stops.clone(),
            max_lines: job.max_lines,
            labels: job.labels,
            timeout: Some(left),
            history: &job.history,
            cancel: Some(&job.cancel),
            bans: job.bans.as_deref(),
        };
        let events = job.events.clone();
        let mut text = String::new();
        backends.start_job();
        let result = generate(&mut *backends, tok, &mut cache, &mut sampler, req, |t| {
            if t.is_empty() {
                return;
            }
            if n == 1 {
                let _ = events.send(Event::Text(t.to_string()));
            } else {
                text.push_str(t);
            }
        });
        caches.checkin(job.key, cache);
        let score = match (&result, &job.score) {
            (Ok(_), Some(f)) if n > 1 => f(text.trim()),
            _ => 0,
        };
        let stopped = result.as_ref().map_or(true, |o| matches!(o.stop, StopReason::Cancelled | StopReason::Timeout));
        if n > 1 {
            info!(candidate = k, score, text = %text.trim(), "best-of candidate");
        }
        (stopped, score, text, result)
    };
    let (mut stopped, mut best_score, mut text, mut result) = run_one(0);
    for k in 1..n {
        if stopped {
            break;
        }
        let (s, score, t, r) = run_one(k);
        if score > best_score {
            (best_score, text, result) = (score, t, r);
        }
        stopped = s;
    }
    if n > 1 && !text.is_empty() {
        let _ = job.events.send(Event::Text(text));
    }
    let total = started.elapsed();
    match result {
        Ok(out) if job.max_new_tokens == 0 && job.max_new_tokens_gpu == 0 => {
            info!(
                key = ?job.key,
                stop = ?out.stop,
                prompt_tokens = job.prompt.len(),
                reused = out.reused_tokens,
                prefill_tokens = out.prefill_tokens,
                prefill_s = out.prefill.as_secs_f32(),
                backend = backends.route(),
                "prefilled ahead"
            );
        }
        Ok(out) => {
            info!(
                key = ?job.key,
                seed = job.seed,
                stop = ?out.stop,
                prompt_tokens = job.prompt.len(),
                reused = out.reused_tokens,
                prefill_tokens = out.prefill_tokens,
                prefill_s = out.prefill.as_secs_f32(),
                generated = out.tokens.len(),
                decode_s = out.decode.as_secs_f32(),
                backend = backends.route(),
                "job done"
            );
            let mut s = stats.lock().unwrap_or_else(|e| e.into_inner());
            s.jobs += 1;
            s.prompt_tokens += job.prompt.len() as u64;
            s.reused_tokens += out.reused_tokens as u64;
            // Counts and rates are set together, so /stats shows one reply's numbers.
            if out.prefill_tokens > 0 && !out.prefill.is_zero() {
                s.last_prefill_tok_s = out.prefill_tokens as f64 / out.prefill.as_secs_f64();
                s.last_prefill_tokens = out.prefill_tokens;
                s.last_reused_tokens = out.reused_tokens;
                s.last_prefill = out.prefill;
            }
            let decode_tok_s = (out.tokens.len() > 1 && !out.decode.is_zero())
                .then(|| (out.tokens.len() - 1) as f64 / out.decode.as_secs_f64());
            if let Some(rate) = decode_tok_s {
                s.last_decode_tok_s = rate;
                s.last_decode_tokens = out.tokens.len();
                s.last_decode = out.decode;
            }
            s.last_backend = backends.route();
            s.last_total = total;
            let route = s.last_backend.clone();
            s.recent.push(&route, Timing { total, prefill: out.prefill, decode_tok_s });
            drop(s);
            let _ = job.events.send(Event::Finished(out));
        }
        Err(e) => {
            error!(error = %e, "generation failed");
            let _ = job.events.send(Event::Failed(e.to_string()));
        }
    }
}

/// Removes and returns the first element matching `pred`, wherever it is in
/// the deque (unlike `pop_front`/`pop_back`). `d.remove(i)` can't actually
/// return `None` here (`i` just came from `position` on this same,
/// untouched deque), but letting the two `Option`s compose means there's
/// nothing to `.expect()` either way.
fn remove_where<T>(d: &mut VecDeque<T>, pred: impl Fn(&T) -> bool) -> Option<T> {
    let i = d.iter().position(pred)?;
    d.remove(i)
}

/// One active f32 cache plus an LRU of f16-parked ones (design.md §3, §5).
struct Caches {
    /// The persona header, prefilled (`SeedHandle`): what a new channel's or
    /// a one-shot reply's cache starts as.
    seed: Option<KvCache>,
    active: Option<(u64, KvCache)>,
    /// Most recently used first.
    parked: VecDeque<(u64, ParkedKv)>,
    slots: usize,
}

impl Caches {
    fn new(slots: usize) -> Self {
        Self {
            seed: None,
            active: None,
            parked: VecDeque::new(),
            slots,
        }
    }

    fn checkout(&mut self, key: CacheKey, model: &Llama) -> KvCache {
        let fresh = |seed: &Option<KvCache>| seed.clone().unwrap_or_else(|| model.new_cache());
        let CacheKey::Channel(ch) = key else {
            return fresh(&self.seed);
        };
        match self.active.take() {
            Some((c, cache)) if c == ch => return cache,
            Some((c, cache)) => self.park(c, &cache),
            None => {}
        }
        match remove_where(&mut self.parked, |(c, _)| *c == ch) {
            Some((_, parked)) => parked.restore(),
            None => fresh(&self.seed),
        }
    }

    fn checkin(&mut self, key: CacheKey, cache: KvCache) {
        if let CacheKey::Channel(ch) = key {
            self.active = Some((ch, cache));
        }
    }

    fn park(&mut self, ch: u64, cache: &KvCache) {
        if self.slots == 0 || cache.is_empty() {
            return;
        }
        self.parked.retain(|(c, _)| *c != ch);
        self.parked.push_front((ch, cache.park()));
        self.parked.truncate(self.slots);
    }

    fn drop_channel(&mut self, ch: u64) {
        if self.active.as_ref().is_some_and(|(c, _)| *c == ch) {
            self.active = None;
        }
        self.parked.retain(|(c, _)| *c != ch);
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    fn t(total_ms: u64, prefill_ms: u64, rate: Option<f64>) -> Timing {
        Timing {
            total: Duration::from_millis(total_ms),
            prefill: Duration::from_millis(prefill_ms),
            decode_tok_s: rate,
        }
    }

    #[test]
    fn recent_keeps_kinds_apart_and_the_last_twenty() {
        let mut r = Recent::default();
        r.push("cpu", t(9000, 7000, Some(5.0)));
        r.push("gpu 0000:06:00.0 → cpu", t(4000, 300, Some(9.0)));
        for i in 0..25 {
            r.push("gpu 0000:06:00.0", t(1000 + i, 300, Some(47.0)));
        }
        assert_eq!((r.cpu.len(), r.moved.len(), r.gpu.len()), (1, 1, RECENT));
        assert_eq!(r.gpu[0].total, Duration::from_millis(1005), "oldest dropped");
    }

    #[test]
    fn summaries_use_medians_and_the_slowest() {
        let times: VecDeque<Timing> =
            [t(1000, 200, Some(40.0)), t(3000, 400, Some(50.0)), t(2000, 300, None), t(1500, 250, Some(45.0))].into();
        assert_eq!(
            summarize(&times).unwrap(),
            "total median 1.75 s, slowest 3.00 s; prefill median 0.28 s; decode median 45.0 tok/s"
        );
        let short: VecDeque<Timing> = [t(500, 500, None)].into();
        assert_eq!(summarize(&short).unwrap(), "total median 0.50 s, slowest 0.50 s; prefill median 0.50 s");
        assert_eq!(summarize(&VecDeque::new()), None);
    }
}
