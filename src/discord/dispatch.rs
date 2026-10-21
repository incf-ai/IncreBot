//! The dispatcher task (design.md §4, §6): owns the merging queue and hands the
//! engine one job at a time, building each prompt at dispatch time so a merged
//! job sees every message up to the moment it starts. While nothing is queued,
//! it prefills the prompts of channels quiet for an hour ([`watch_idle`]),
//! so their next reply finds them cached.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use poise::serenity_prelude as serenity;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

use super::remember::{self, command_of};
use super::{Data, resolve_mentions, window_msg};
use crate::chat::sanitize::clean_name;
use crate::chat::gate::is_ignored_text;
use crate::chat::queue::{Enqueued, JobQueue, QueueKey};
use crate::chat::window::{ChannelWindow, WindowMsg, snowflake_at, snowflake_ms};
use crate::chat::{PromptLimits, build_prompt};
use crate::engine::worker::{CacheKey, EngineHandle, Event, Job};

#[derive(Clone)]
pub enum Source {
    /// A channel message. `snapshot` is the channel's history frozen when the
    /// trigger was queued (trigger included); the prompt is built from it.
    Channel {
        trigger_msg: u64,
        snapshot: ChannelWindow,
    },
    Ask {
        author: String,
        prompt: String,
        /// The novelty command it's for, whose persona setup adds a note and
        /// word biases (`persona::Command`); `None` for `/ask`.
        command: Option<crate::persona::CommandName>,
        /// What this reply drew from the command's pools (set-up line, moves,
        /// examples; `Persona::draw`).
        draw: Option<crate::persona::CommandDraw>,
        /// The requester's turn, for a slash command whose answer may go into
        /// the history (`remember`); `None` for `/super-admin shake`.
        asked: Option<Box<super::remember::Asked>>,
    },
}

impl Source {
    /// The command's requester turn, when the persona has its answers kept in
    /// the history (`remember_commands`).
    pub fn remembered(&self, persona: &crate::persona::Persona) -> Option<&super::remember::Asked> {
        match self {
            Source::Ask { asked: Some(a), .. } if persona.remembers(a.slash) => Some(&**a),
            _ => None,
        }
    }

    /// Whether a retry of this output updates the history window: a channel
    /// reply, or a remembered command's answer.
    pub fn keeps_history(&self, persona: &crate::persona::Persona) -> bool {
        matches!(self, Source::Channel { .. }) || self.remembered(persona).is_some()
    }
}

pub struct Trigger {
    pub key: QueueKey,
    pub channel: u64,
    pub source: Source,
    pub requester: u64,
    pub cancel: Arc<AtomicBool>,
    pub events: mpsc::UnboundedSender<Event>,
    /// Earlier outputs a `/retry` should steer away from: added to the
    /// penalties' history, so DRY and the repetition penalty push the new
    /// sample off the old wording.
    pub avoid: Vec<String>,
    /// A reply to a message in its channel: it waits out the channel's pause
    /// (`chat::limits::Pacer`); slash commands' jobs don't.
    pub paced: bool,
}

/// What `/retry` needs to generate an output again (`chat::retry`).
#[derive(Clone)]
pub struct Replay {
    pub source: Source,
    /// The novelty card above the comment; it stays as it was.
    pub header: Option<String>,
    /// The outputs so far, oldest first (`retry::steer`).
    pub avoid: Vec<String>,
}

pub enum EnqueueResult {
    /// Queued; `ahead` counts jobs before it, the running one included.
    Queued {
        ahead: usize,
    },
    Full,
}

pub enum DispatchMsg {
    Enqueue(Box<Trigger>, oneshot::Sender<EnqueueResult>),
    /// Drop a still-queued job with this cancel flag.
    Cancel(Arc<AtomicBool>),
    /// Drop every queued job (the admin `/super-admin clearqueue`); replies with how many.
    /// The running job is not touched.
    Clear(oneshot::Sender<usize>),
    /// Cancel the running job, drop the queue, wait briefly for the engine, ack.
    Shutdown(oneshot::Sender<()>),
    /// Prefill this channel's prompt as its next reply will start, once
    /// nothing else is waiting; any reply cancels it.
    Warm(u64),
}

static QUEUE_LEN: AtomicUsize = AtomicUsize::new(0);

pub fn queue_len() -> usize {
    QUEUE_LEN.load(Ordering::Relaxed)
}

/// The inference thread is gone (it panicked or returned; either way its job
/// channel closed) and nothing can ever reply again. Rather than let the
/// dispatcher quietly stop while the Discord connection stays up looking
/// healthy, exit so systemd restarts the service with a fresh model load
/// (`Restart=on-failure`, deploy.md). Never returns.
fn inference_thread_gone() -> ! {
    error!(
        "the inference thread's job channel is closed (it panicked or exited); \
         exiting so the service restarts with a fresh model load"
    );
    std::process::exit(1);
}

pub async fn run(
    data: Data,
    mut engine: EngineHandle,
    mut rx: mpsc::UnboundedReceiver<DispatchMsg>,
) {
    // Limits per channel and for interactions are applied here, on enqueue.
    let mut queue: JobQueue<Trigger> = JobQueue::new(usize::MAX);
    let mut running: Option<Arc<AtomicBool>> = None;
    // The running job's channel, when it is a paced reply.
    let mut running_paced: Option<u64> = None;
    // When a pausing channel's next reply may start, if nothing else can.
    let mut wake: Option<std::time::Instant> = None;
    // When each waiting paced reply's message was queued, by its cancel
    // flag, and whether it was told it is paused.
    let mut arrived: Vec<(Arc<AtomicBool>, std::time::Instant, bool)> = Vec::new();
    // Channels to prefill, oldest first, and whether the running job is one.
    let mut warm: Vec<u64> = Vec::new();
    let mut warming = false;
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                None => break,
                Some(DispatchMsg::Enqueue(t, reply)) => {
                    if warming && let Some(flag) = &running {
                        flag.store(true, Ordering::Relaxed);
                    }
                    if matches!(t.source, Source::Channel { .. }) {
                        warm.retain(|&c| c != t.channel);
                    }
                    let busy = (running.is_some() && !warming) as usize;
                    // At most `queue_depth` waiting per channel; a reply through an
                    // interaction (keyed `Ask`) also only behind so many in all,
                    // so it finishes before its token expires.
                    let here = queue.count_where(|w| w.channel == t.channel);
                    let interaction = matches!(t.key, QueueKey::Ask(_));
                    let full = here >= data.cfg.runtime.queue_depth
                        || (interaction && queue.len() + busy > data.cfg.generation.interaction_ahead());
                    if !full && t.paced {
                        arrived.push((t.cancel.clone(), std::time::Instant::now(), false));
                    }
                    let res = if full {
                        EnqueueResult::Full
                    } else {
                        match queue.enqueue(t.key, *t) {
                            Enqueued::Queued { ahead } => EnqueueResult::Queued { ahead: ahead + busy },
                            Enqueued::Full => EnqueueResult::Full,
                        }
                    };
                    let _ = reply.send(res);
                }
                Some(DispatchMsg::Cancel(flag)) => {
                    queue.remove_where(|t| Arc::ptr_eq(&t.cancel, &flag));
                }
                Some(DispatchMsg::Clear(done)) => {
                    // Dropping the triggers closes their event channels; each reply
                    // task then removes its "waiting in queue" placeholder.
                    let n = queue.clear().len();
                    info!(dropped = n, "queue cleared by an admin");
                    let _ = done.send(n);
                }
                Some(DispatchMsg::Shutdown(done)) => {
                    while queue.pop().is_some() {}
                    if let Some(flag) = running.take() {
                        flag.store(true, Ordering::Relaxed);
                        let _ = tokio::time::timeout(Duration::from_secs(3), engine.idle.recv()).await;
                    }
                    QUEUE_LEN.store(0, Ordering::Relaxed);
                    let _ = done.send(());
                    break;
                }
                Some(DispatchMsg::Warm(ch)) => {
                    if !warm.contains(&ch) {
                        warm.push(ch);
                    }
                }
            },
            Some(()) = engine.idle.recv(), if running.is_some() => {
                running = None;
                warming = false;
                if let Some(ch) = running_paced.take() {
                    data.pacer().finished(&data.cfg.discord.channel_limits, ch, std::time::Instant::now());
                }
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(wake.unwrap_or_else(std::time::Instant::now))), if wake.is_some() => {}
        }

        if running.is_none() {
            loop {
                // The first job not waiting out its channel's pause.
                let next = {
                    let (pacer, now) = (data.pacer(), std::time::Instant::now());
                    queue.pop_where(|t| !t.paced || pacer.ready_at(t.channel, now).is_none())
                };
                let Some((_, trigger)) = next else { break };
                if trigger.cancel.load(Ordering::Relaxed) {
                    continue;
                }
                let events = trigger.events.clone();
                let paced = trigger.paced.then_some(trigger.channel);
                arrived.retain(|(c, ..)| !Arc::ptr_eq(c, &trigger.cancel));
                match build_job(&data, trigger).await {
                    Ok(job) => {
                        running = Some(job.cancel.clone());
                        running_paced = paced;
                        if engine.jobs.send(job).await.is_err() {
                            inference_thread_gone();
                        }
                        break;
                    }
                    Err(e) => {
                        warn!(error = %e, "could not build a prompt");
                        let _ = events.send(Event::Failed(e.to_string()));
                    }
                }
            }
        }
        while running.is_none() && queue.is_empty() && !warm.is_empty() {
            let ch = warm.remove(0);
            match build_warm(&data, ch).await {
                Ok(Some(job)) => {
                    running = Some(job.cancel.clone());
                    warming = true;
                    if engine.jobs.send(job).await.is_err() {
                        inference_thread_gone();
                    }
                }
                Ok(None) => {}
                Err(e) => warn!(error = %e, "could not build a prompt to prefill"),
            }
        }
        // Dropped from the queue (cleared, cancelled): nothing to show.
        arrived.retain(|(c, ..)| queue.count_where(|t| Arc::ptr_eq(&t.cancel, c)) > 0);
        // A reply the pause holds back is told so, once: its placeholder
        // says so until the pause ends (`reply::stream`).
        {
            let (pacer, now) = (data.pacer(), std::time::Instant::now());
            for (_, (_, t)) in queue.iter() {
                let Some(until) = pacer.ready_at(t.channel, now) else { continue };
                let Some(entry) = arrived.iter_mut().find(|(c, ..)| Arc::ptr_eq(c, &t.cancel)) else { continue };
                let held = pacer.held(t.channel, entry.1);
                if !entry.2 && !held.is_zero() {
                    entry.2 = true;
                    let _ = t.events.send(Event::Paused(until));
                    data.trace_source(t.channel, &t.source, format!(
                        "channel rate limit: held {:.1} s (\"channel rate limit active\" shown)",
                        held.as_secs_f64()
                    ));
                }
            }
        }
        // Nothing can start now but paced replies will: wake when the first may.
        wake = if running.is_none() && !queue.is_empty() {
            let (pacer, now) = (data.pacer(), std::time::Instant::now());
            queue.iter().filter_map(|(_, (_, t))| if t.paced { pacer.ready_at(t.channel, now) } else { None }).min()
        } else {
            None
        };
        QUEUE_LEN.store(queue.len(), Ordering::Relaxed);
    }
    info!("dispatcher stopped");
}

async fn build_job(data: &Data, t: Trigger) -> Result<Job> {
    let cfg = &data.cfg;
    // One persona for the whole job, even if a reload lands meanwhile.
    let persona = data.persona();
    let builder = persona.prompt.clone();
    let count = |s: &str| data.count_tokens(s);
    // A CPU reply skips the persona repeat after the history, which it would
    // otherwise prefill every turn (design.md §7).
    let on_gpu = data.gpus.as_ref().is_some_and(|g| g.any_ready());
    let repeat_persona = on_gpu || !cfg.generation.cpu_drop_persona_repeat;
    // What the reply answers, for `prompted_by` word rules.
    let message = match &t.source {
        Source::Channel { trigger_msg, snapshot } => snapshot.unanswered(data.bot_id(), *trigger_msg),
        Source::Ask { prompt, .. } => prompt.clone(),
    };
    // Replies the bot has made since the latest message with one of some
    // phrases: for `after_replies` notes and `for_replies` bans.
    let replies = |phrases: &[String], span: usize| -> Option<usize> {
        match &t.source {
            Source::Channel { trigger_msg, snapshot } => {
                let p = crate::engine::sampler::Phrases::new(phrases);
                snapshot.replies_since(data.bot_id(), *trigger_msg, span, &|text| {
                    p.found_whole_in(&crate::engine::sampler::normalized_words(text))
                })
            }
            Source::Ask { prompt, .. } => crate::persona::one_shot_replies(prompt)(phrases, span),
        }
    };
    // Notes look further back when they stay on for several replies.
    // The scene a channel reply's prompt gets, for its trace.
    let mut traced_scene: Option<String> = None;
    let note = match &t.source {
        Source::Channel { trigger_msg, snapshot } => {
            let (bot, trigger) = (data.bot_id(), *trigger_msg);
            persona.note(
                &|n| snapshot.since_replies(bot, trigger, n),
                &|n| snapshot.since_replies_by_asker(bot, trigger, n),
                &replies,
            )
        }
        Source::Ask { prompt, command: Some(c), draw, .. } => persona.command_note(*c, prompt, draw.as_ref()),
        Source::Ask { prompt, .. } => persona.note(&|_| prompt.clone(), &|_| prompt.clone(), &replies),
    };
    // The bot's time of day first (`clock`), for every reply.
    let note = match (persona.time_note(std::time::SystemTime::now()), note) {
        (Some(time), Some(n)) => Some(format!("{time}\n{n}")),
        (time, n) => time.or(n),
    };
    let (text, stops, key, history) = match &t.source {
        Source::Channel {
            trigger_msg,
            snapshot,
        } => {
            // Built from the history as it was when the trigger was queued: later
            // messages, edits and deletions don't reach this reply.
            let mut frozen = snapshot.clone();
            let limits = PromptLimits {
                history_messages: cfg.discord.history_messages,
                max_prompt_tokens: cfg.generation.max_prompt_tokens,
                max_message_tokens: cfg.generation.max_message_tokens,
            };
            // What the bot was up to, after an hour without a reply (`chat::idle`).
            let p = &persona.persona;
            let scene = data
                .idle
                .scene(t.channel, snowflake_ms(*trigger_msg), &p.idle_activities, &p.idle_thoughts)
                .map(|s| s.line(&builder.bot_name));
            traced_scene = scene.clone();
            let built = build_prompt(
                &mut frozen,
                &builder,
                data.bot_id(),
                Some(*trigger_msg),
                repeat_persona,
                note.as_deref(),
                scene.as_deref(),
                &limits,
                &count,
            )?;
            // An anchor jump made on the frozen copy carries over to the live
            // window, so later prompts start where this one did (KV prefix reuse).
            if let Some(live) = data.windows.lock().await.get_mut(&t.channel) {
                live.advance_to(frozen.anchor());
            }
            (
                built.text,
                built.stops,
                CacheKey::Channel(t.channel),
                built.history,
            )
        }
        Source::Ask { author, prompt, .. } => {
            let one = crate::chat::build_one_shot(
                &builder,
                author,
                prompt,
                repeat_persona,
                note.as_deref(),
                cfg.generation.max_prompt_tokens,
                &count,
            )?;
            (one.text, one.stops, CacheKey::Scratch, one.history)
        }
    };
    tracing::trace!(prompt = %text, "prompt");
    if let CacheKey::Channel(channel) = key {
        data.last_prompts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(channel, text.clone());
    }
    let prompt = data.tok.encode_prompt(&text)?;
    let stops = persona.stops(stops);
    // A novelty command may set its own length and temperature.
    let command = match &t.source {
        Source::Ask { command, .. } => *command,
        Source::Channel { .. } => None,
    };
    let limits = persona.persona.limits(command);
    // The penalties see the chat history, not the system prompt and template
    // text around it (design.md §5). Tokenized on its own, so token boundaries
    // at line starts can differ slightly from the prompt's.
    let mut history = data.tok.encode_prompt(&history)?.split_off(1);
    for old in &t.avoid {
        history.extend(data.tok.encode_prompt(old)?.split_off(1));
    }
    let bans = {
        // The persona's phrase bans, plus no speaker label at the start
        // (anyone in the prompt, the bot too) and no repeat of the bot's
        // last reply here.
        let names = stops.iter().filter_map(|s| s.strip_prefix('\n').and_then(|s| s.strip_suffix(':')).map(String::from));
        let previous = match &t.source {
            Source::Channel { trigger_msg, snapshot } => snapshot
                .messages()
                .filter(|m| m.author_id == data.bot_id() && m.id < *trigger_msg)
                .max_by_key(|m| m.id)
                .map(|m| m.text.clone()),
            Source::Ask { .. } => None,
        };
        let bans = persona.bans(&message, command, &replies).with_prefixes(names).with_echo(previous.as_deref().unwrap_or(""));
        (!bans.is_empty() || bans.first_chars.is_some()).then(|| std::sync::Arc::new(bans))
    };
    let mut sampler = persona.sampler_for(&message, command);
    // A channel reply is traced: record what its word biases did.
    sampler.observe = matches!(t.source, Source::Channel { .. });
    let best_of = persona.best_of(command, on_gpu);
    // What the reply is built with, in its message's trace.
    if let Source::Channel { trigger_msg, .. } = &t.source {
        let tr = |e: String| data.trace(t.channel, *trigger_msg, e);
        let cut = |s: &str| s.chars().take(300).collect::<String>().replace('\n', " ");
        let persona_file = std::path::Path::new(&cfg.discord.persona_file)
            .file_name()
            .map_or_else(|| cfg.discord.persona_file.clone(), |n| n.to_string_lossy().into_owned());
        let built = format!(
            "{} tokens on {}, persona {persona_file}{}",
            prompt.len(),
            if on_gpu { "a GPU" } else { "the CPU" },
            if repeat_persona { ", persona repeated after the history" } else { "" }
        );
        let sampled = format!(
            "sampler: temperature {}, top_k {}, top_p {}, min_p {}{}{}{}",
            sampler.temperature,
            sampler.top_k,
            sampler.top_p,
            sampler.min_p,
            match &sampler.word_start {
                Some(w) => format!(
                    ", word starts redrawn {}% of the time (temperature {}, top_k {}, top_p {}, min_p {})",
                    w.chance * 100.0,
                    w.temperature,
                    w.top_k,
                    w.top_p,
                    w.min_p
                ),
                None => String::new(),
            },
            match &sampler.words {
                Some(w) => {
                    let (all, on) = w.counts(&sampler.message);
                    format!(", word biases: {all} rules ({on} on for this message)")
                }
                None => ", word biases: none".to_string(),
            },
            if best_of.get() > 1 { format!(", best of {best_of}") } else { String::new() },
        );
        let banned = match &bans {
            Some(b) => format!(
                "phrase bans: {} persona patterns, plus speaker-label and repeat guards{}",
                b.patterns.len(),
                if b.first_chars.is_some() { ", banned first characters" } else { "" }
            ),
            None => "phrase bans: none".to_string(),
        };
        let note = format!("note: {}", note.as_deref().unwrap_or("none triggered"));
        let scene = format!("idle scene: {}", traced_scene.as_deref().unwrap_or("none"));
        tr(format!("prompt built: {built}"));
        if !t.avoid.is_empty() {
            tr(format!("retry: steering away from {} earlier output(s)", t.avoid.len()));
        }
        tr(cut(&note));
        tr(cut(&scene));
        tr(sampled.clone());
        tr(banned.clone());
        // The same in full, with the prompt, for "Admin actions" → Context.
        let mut context = format!("context of the reply to message {trigger_msg}: {built}\n{note}\n{scene}\n");
        for (i, old) in t.avoid.iter().enumerate() {
            context += &format!("steering away from earlier output {}: {old}\n", i + 1);
        }
        context += &format!("{sampled}\n{banned}\n\n──── the prompt, exactly as the model saw it ────\n{text}");
        data.traces.set_context(t.channel, *trigger_msg, context);
    }
    Ok(Job {
        key,
        prompt,
        sampler,
        seed: rand::random(),
        history,
        max_new_tokens: limits.max_new_tokens,
        max_new_tokens_gpu: limits.max_new_tokens_gpu,
        wrap_up_after: limits.wrap_up_after,
        wrap_up_after_gpu: limits.wrap_up_after_gpu,
        stops,
        max_lines: limits.max_lines,
        labels: true,
        timeout: Duration::from_secs(cfg.generation.timeout_secs),
        cancel: t.cancel,
        events: t.events,
        // A novelty command may ask for its best of N (only on a GPU).
        best_of,
        score: match &t.source {
            Source::Ask { prompt, command: Some(c), draw, .. } => Some(c.scorer(prompt, draw.as_ref())),
            _ => None,
        },
        bans,
    })
}

/// A prefill-only job for `channel`'s next reply: its prompt up to the end
/// of the history (the scene, after it, isn't part of it); `None` without a
/// window there.
/// The end is found as where the prompt with one more message and the one
/// without part, so it holds for any template. A message long enough to move
/// the history's anchor still makes the reply re-read it.
async fn build_warm(data: &Data, channel: u64) -> Result<Option<Job>> {
    let cfg = &data.cfg;
    let Some(window) = data.windows.lock().await.get(&channel).cloned() else {
        return Ok(None);
    };
    let persona = data.persona();
    let builder = &persona.prompt;
    let count = |s: &str| data.count_tokens(s);
    let limits = PromptLimits {
        history_messages: cfg.discord.history_messages,
        max_prompt_tokens: cfg.generation.max_prompt_tokens,
        max_message_tokens: cfg.generation.max_message_tokens,
    };
    // One more message by the last author, so the list of names stays the same.
    let last = window.messages().max_by_key(|m| m.id);
    let next = WindowMsg {
        id: last.map_or(0, |m| m.id).max(window.anchor().unwrap_or(0)) + 1,
        author_id: last.map_or(0, |m| m.author_id),
        author: last.map_or_else(|| "someone".to_string(), |m| m.author.clone()),
        text: "hi".into(),
        extras: Default::default(),
        mentions: Vec::new(),
    };
    let mut with = window.clone();
    with.push(next.clone(), usize::MAX);
    let bot = data.bot_id();
    let a = build_prompt(&mut with, builder, bot, Some(next.id), true, None, None, &limits, &count)?;
    let mut without = window;
    without.advance_to(with.anchor());
    let b = build_prompt(&mut without, builder, bot, None, true, None, None, &limits, &count)?;
    let end = a.text.char_indices().zip(b.text.chars()).find(|((_, x), y)| x != y).map_or(a.text.len().min(b.text.len()), |((i, _), _)| i);
    let prompt = data.tok.encode_prompt(&a.text[..end])?;
    let (events, _) = mpsc::unbounded_channel();
    Ok(Some(Job {
        key: CacheKey::Channel(channel),
        prompt,
        sampler: Default::default(),
        seed: 0,
        history: Vec::new(),
        // Nothing to generate: the worker only prefills.
        max_new_tokens: 0,
        max_new_tokens_gpu: 0,
        wrap_up_after: 0,
        wrap_up_after_gpu: 0,
        stops: Vec::new(),
        max_lines: None,
        labels: false,
        timeout: Duration::from_secs(cfg.generation.timeout_secs),
        cancel: Arc::new(AtomicBool::new(false)),
        events,
        best_of: std::num::NonZeroUsize::MIN,
        score: None,
        bans: None,
    }))
}

/// Every half minute, draws the idle scene of each channel the bot hasn't
/// replied in for an hour (`chat::idle`). When nobody else spoke there for
/// that hour either, it has the channel's prompt prefilled too, as its cache
/// has likely given way to other channels': the first reply after the quiet
/// spell doesn't wait for the history to be re-read. (In a busy channel the
/// history would move on before the reply.) Channels the bot hasn't replied
/// in since startup wait for their first reply instead.
pub async fn watch_idle(data: Data) {
    let mut tick = tokio::time::interval(Duration::from_secs(30));
    loop {
        tick.tick().await;
        if data.shutting_down.load(Ordering::Relaxed) {
            break;
        }
        let now = snowflake_ms(snowflake_at(std::time::SystemTime::now()));
        let last: Vec<(u64, u64)> = data
            .windows
            .lock()
            .await
            .iter()
            .filter_map(|(&ch, w)| w.messages().map(|m| m.id).max().map(|id| (ch, snowflake_ms(id))))
            .collect();
        let persona = data.persona();
        let p = &persona.persona;
        for (ch, last) in last {
            if data.idle.prepare(ch, now, &p.idle_activities, &p.idle_thoughts) {
                let quiet = now >= last + crate::chat::idle::IDLE_GAP_MS;
                info!(channel = %data.cfg.discord.label(ch), quiet, "an hour without a reply: idle scene drawn");
                if quiet {
                    let _ = data.dispatch.send(DispatchMsg::Warm(ch));
                }
            }
        }
    }
}

/// At startup (design.md §7): every configured channel gets its history
/// window (empty but for what was said since startup), a scene, and its
/// prompt prefilled, so its first reply starts warm, as after a quiet hour.
/// On a GPU or the CPU alike; the prefills give way to replies.
pub async fn warm_at_boot(data: Data, http: Arc<serenity::Http>) {
    let d = &data.cfg.discord;
    let now = snowflake_ms(snowflake_at(std::time::SystemTime::now()));
    let persona = data.persona();
    let p = &persona.persona;
    let mut seen = Vec::new();
    for c in d.dedicated_channels.iter().chain(&d.shared_channels) {
        if c.id == 0 || seen.contains(&c.id) {
            continue;
        }
        seen.push(c.id);
        {
            let mut windows = data.windows.lock().await;
            let window = windows.entry(c.id).or_insert_with(|| ChannelWindow::after(data.start_anchor));
            if !window.loaded {
                seed_window(&data, &http, c.id, window).await;
            }
        }
        let scene = data.idle.reset(c.id, now, &p.idle_activities, &p.idle_thoughts);
        info!(channel = %d.label(c.id), scene, "startup: prefilling");
        let _ = data.dispatch.send(DispatchMsg::Warm(c.id));
    }
}

/// Cold start for a channel (first trigger since startup or `/super-admin reset`): fetch
/// recent history over REST once; afterwards gateway events keep it current.
/// Runs when the trigger is queued, before its history is frozen.
pub(super) async fn seed_window(
    data: &Data,
    http: &serenity::Http,
    channel: u64,
    window: &mut ChannelWindow,
) {
    let limit = data.cfg.discord.history_messages.min(100) as u8;
    let inflight = data
        .inflight
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let bot = data.bot_id();
    let bot_roles = data.known_bot_roles();
    match serenity::ChannelId::new(channel)
        .messages(http, serenity::GetMessages::new().limit(limit))
        .await
    {
        Ok(msgs) => {
            let persona = data.persona();
            let hidden = |u: u64, id: u64| data.optouts.hidden(u, id);
            let ctx = SeedCtx {
                persona: &persona.persona,
                bot_roles: &bot_roles,
                bot_name: &data.cfg.discord.bot_name,
                hidden: &hidden,
                max: crate::chat::window::max_buffer(data.cfg.discord.history_messages),
            };
            seed_messages(&ctx, bot, &inflight, window, &msgs);
        }
        Err(e) => {
            warn!(
                channel = %data.cfg.discord.label(channel),
                error = %e,
                "could not fetch history (Read Message History permission?)"
            )
        }
    }
    window.loaded = true;
}

/// The filtering and rebuilding of a cold start, on fetched messages (newest
/// first, as REST returns them): the bot's notices and `//` messages are
/// skipped, opted-out users are invisible, remembered command answers are
/// rebuilt (`seed_command_reply`) and everything else enters as it was said.
pub(super) fn seed_messages(
    ctx: &SeedCtx,
    bot: u64,
    inflight: &std::collections::HashSet<u64>,
    window: &mut ChannelWindow,
    msgs: &[serenity::Message],
) {
    // How the fetched messages name their authors, for a command's
    // requester and the people its card mentions.
    let names: HashMap<u64, String> = msgs
        .iter()
        .map(|m| (m.author.id.get(), window_msg(m, ctx.bot_roles, ctx.bot_name).author))
        .collect();
    for m in msgs.iter().rev() {
        let ours = m.author.id.get() == bot;
        if inflight.contains(&m.id.get())
            || is_ignored_text(&m.content)
            || (ctx.hidden)(m.author.id.get(), m.id.get())
        {
            continue;
        }
        if ours && is_command_reply(m) {
            seed_command_reply(ctx, &names, window, m);
            continue;
        }
        if ours && let Some(interaction) = card_part_of(bot, m) {
            seed_follow_up(ctx, window, m, interaction);
            continue;
        }
        let mut msg = window_msg(m, ctx.bot_roles, ctx.bot_name);
        if ours {
            // The bot's notices aren't its words: a leftover notice message
            // ("*(restarting…)*") is skipped whole, a trailing "*(stopped)*"
            // leaves the turn as the live window has it.
            msg.text = remember::without_notices(&msg.text);
            if msg.text.is_empty() {
                continue;
            }
        }
        window.push(msg, ctx.max);
    }
}

/// What [`seed_messages`] needs besides the messages. `names` is filled in
/// there (authors' names as the fetched messages give them, by user ID);
/// callers pass an empty map.
#[derive(Clone, Copy)]
pub(super) struct SeedCtx<'a> {
    pub persona: &'a crate::persona::Persona,
    pub bot_roles: &'a [u64],
    pub bot_name: &'a str,
    /// Whether a user's message is hidden: they are opted out, or it is from
    /// before they opted back in (`OptOuts::hidden`).
    pub hidden: &'a dyn Fn(u64, u64) -> bool,
    pub max: usize,
}

/// A slash command's answer in the REST seed: as live (`reply::stream`), it
/// enters only for a command in `remember_commands`, as the requester's turn
/// from its header (the first line of the first message) and the bot's reply
/// below it, follow-ups included. Never a card as the bot's own turn to copy.
fn seed_command_reply(c: &SeedCtx, names: &HashMap<u64, String>, window: &mut ChannelWindow, m: &serenity::Message) {
    let SeedCtx { persona, bot_roles, bot_name, hidden, max } = *c;
    let Some(serenity::MessageInteractionMetadata::Command(meta)) = m.interaction_metadata.as_deref() else {
        return;
    };
    let (interaction, requester) = (meta.id.get(), meta.user.id.get());
    if meta.original_response_message_id.is_some() {
        return seed_follow_up(c, window, m, interaction);
    }
    // The first message: the header tops it.
    let (header, body) = m.content.split_once('\n').unwrap_or((&m.content, ""));
    if !command_of(header).is_some_and(|c| persona.remembers(c)) {
        return;
    }
    let mut users = m.mentions.clone();
    users.push(meta.user.clone());
    // Anyone the card names by their name here (as their messages are
    // labeled), else as Discord lists them.
    let resolve = |t: &str| {
        let mut t = t.to_string();
        for id in remember::mentioned(t.as_str()) {
            if let Some(n) = names.get(&id) {
                t = t.replace(&format!("<@{id}>"), &format!("@{n}")).replace(&format!("<@!{id}>"), &format!("@{n}"));
            }
        }
        resolve_mentions(&t, &users, bot_roles, bot_name)
    };
    let name = names.get(&requester).cloned().unwrap_or_else(|| clean_name(meta.user.display_name()));
    let Some(mut asked) = remember::asked(header, interaction, (requester, &name), &resolve) else {
        return;
    };
    asked.turn.mentions = remember::mentioned(header)
        .into_iter()
        .filter_map(|id| names.get(&id).cloned().or_else(|| users.iter().find(|u| u.id.get() == id).map(super::mention_name)))
        .collect();
    if asked.about.iter().any(|&u| hidden(u, m.id.get())) || hidden(requester, m.id.get()) {
        return;
    }
    window.push_tied(asked.turn, m.id.get(), asked.about, max);
    if !window.contains(interaction) {
        // Asked before the window's anchor: an exchange lives and dies
        // whole, so no reply without the question (the live window drops
        // the answer when the anchor passes the turn, `prune_ties`).
        return;
    }
    let text = remember::without_notices(&resolve_mentions(body, &m.mentions, bot_roles, bot_name));
    if !text.is_empty() {
        window.push(WindowMsg { text, ..window_msg(m, bot_roles, bot_name) }, max);
        window.tie_another(interaction, m.id.get());
    }
}

/// A later message of the exchange `interaction` (a follow-up, or a retry's
/// extra part): its text below the rest, if the exchange is in the window.
fn seed_follow_up(c: &SeedCtx, window: &mut ChannelWindow, m: &serenity::Message, interaction: u64) {
    if !window.contains(interaction) {
        return;
    }
    let text = remember::without_notices(&window_msg(m, c.bot_roles, c.bot_name).text);
    if !text.is_empty() {
        window.push(WindowMsg { text, ..window_msg(m, c.bot_roles, c.bot_name) }, c.max);
        // A follow-up leaves with the exchange (deleted, opted out).
        window.tie_another(interaction, m.id.get());
    }
}

/// The exchange a plain message of the bot's continues: a retried card's
/// extra part replies to the card (`reply::Sink::post`). The card deleted,
/// Discord no longer says what it replied to; the part then reads as the
/// bot's own turn (a known gap, design.md §6).
fn card_part_of(bot: u64, m: &serenity::Message) -> Option<u64> {
    let card = m.referenced_message.as_deref().filter(|r| r.author.id.get() == bot && is_command_reply(r))?;
    match card.interaction_metadata.as_deref()? {
        serenity::MessageInteractionMetadata::Command(meta) => Some(meta.id.get()),
        _ => None,
    }
}

/// The bot's answers to slash commands (`/ask`, the novelty commands, follow-ups
/// included). Live, the gateway handler skips the bot's own messages; these
/// enter the history only through `seed_command_reply`, as live through
/// `reply::stream`.
fn is_command_reply(m: &serenity::Message) -> bool {
    m.kind == serenity::MessageType::ChatInputCommand || m.interaction_metadata.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A command answer as Discord returns it: `content` from the bot (99),
    /// asked by user 5 ("Alice") in interaction 100; a follow-up if
    /// `original` is set.
    fn answer(id: u64, content: &str, original: Option<u64>, mentions: serde_json::Value) -> serenity::Message {
        let mut meta = serde_json::json!({
            "id": "100", "type": 2, "user": {"id": "5", "username": "alice", "global_name": "Alice", "discriminator": "0"},
            "authorizing_integration_owners": {}
        });
        if let Some(o) = original {
            meta["original_response_message_id"] = o.to_string().into();
        }
        serde_json::from_value(serde_json::json!({
            "id": id.to_string(), "channel_id": "1", "type": 20, "content": content,
            "author": {"id": "99", "username": "floobbot", "discriminator": "0", "bot": true},
            "timestamp": "2026-10-17T00:00:00Z", "tts": false, "mention_everyone": false,
            "mentions": mentions, "mention_roles": [], "attachments": [], "embeds": [],
            "pinned": false, "interaction_metadata": meta
        }))
        .expect("message")
    }

    fn seeded(remember: &[&str], msgs: &[serenity::Message], hidden: &dyn Fn(u64, u64) -> bool) -> Vec<(u64, u64, String)> {
        let mut persona = crate::persona::test_persona();
        persona.remember_commands = remember.iter().map(|s| s.to_string()).collect();
        let names: HashMap<u64, String> = [(6, "Bobby".to_string())].into();
        let ctx = SeedCtx { persona: &persona, bot_roles: &[], bot_name: "floob", hidden, max: 100 };
        let mut w = ChannelWindow::default();
        for m in msgs {
            seed_command_reply(&ctx, &names, &mut w, m);
        }
        w.messages().map(|m| (m.id, m.author_id, m.text.clone())).collect()
    }

    #[test]
    fn remembered_answers_seed_as_the_requester_then_the_bot() {
        let bob = serde_json::json!([{"id": "6", "username": "bob", "global_name": "Bob", "discriminator": "0"}]);
        let roast = answer(200, "🔥 floob roasts <@6>:\nur tail is ugly\n*(stopped)*", None, bob.clone());
        let more = answer(201, "and ur rocks", Some(200), serde_json::json!([]));
        let none = &|_, _| false;
        assert_eq!(
            seeded(&["roast"], &[roast.clone(), more.clone()], none),
            [
                // As the channel labels Bob's messages, not his global name.
                (100, 5, "[used /roast: floob roasts @Bobby]".to_string()),
                (200, 99, "ur tail is ugly".to_string()),
                (201, 99, "and ur rocks".to_string()),
            ]
        );
        // Not listed: nothing, follow-ups included.
        assert!(seeded(&["ask"], &[roast.clone(), more.clone()], none).is_empty());
        // Bob opted out: the whole answer stays out.
        assert!(seeded(&["roast"], &[roast, more], &|u, _| u == 6).is_empty());
        // Discord didn't list the mention: the name the channel knows.
        let unlisted = answer(200, "🔥 floob roasts <@6>:\nugly", None, serde_json::json!([]));
        assert_eq!(seeded(&["roast"], &[unlisted], none)[0].2, "[used /roast: floob roasts @Bobby]");
        // Not in the channel's names: the server nickname the mention carries.
        let carol = serde_json::json!([{"id": "7", "username": "carol", "global_name": "Carol", "discriminator": "0",
            "member": {"nick": "caz", "roles": [], "joined_at": "2024-01-01T00:00:00Z", "deaf": false, "mute": false, "flags": 0}}]);
        let nicked = answer(200, "🔥 floob roasts <@7>:\nugly", None, carol);
        assert_eq!(seeded(&["roast"], &[nicked], none)[0].2, "[used /roast: floob roasts @caz]");
        // An /ask: the question as Alice's own words; a card alone: only her turn.
        let ask = answer(200, "> <@5> asks: is it shiny?\nyes!!", None, serde_json::json!([]));
        assert_eq!(
            seeded(&["ask"], &[ask], none),
            [(100, 5, "is it shiny?".to_string()), (200, 99, "yes!!".to_string())]
        );
        let card = answer(200, "🎲 <@5> bets **3 shiny rocks**. a tie.", None, serde_json::json!([]));
        assert_eq!(seeded(&["gamble"], &[card], none).len(), 1);
        // A follow-up whose answer isn't in the window (deleted, or its
        // command isn't remembered any more) is skipped.
        let orphan = answer(201, "and ur rocks", Some(200), serde_json::json!([]));
        assert!(seeded(&["roast"], &[orphan], none).is_empty());
    }

    #[test]
    fn a_seeded_answer_needs_its_question_inside_the_window() {
        // The interaction (the requester's turn) sorts before the answer; an
        // anchor between them keeps the whole exchange out, follow-ups
        // included, as the live window drops it when the anchor passes the
        // turn (`ChannelWindow::prune_ties`).
        let persona = {
            let mut p = crate::persona::test_persona();
            p.remember_commands = vec!["ask".into()];
            p
        };
        let none = |_, _| false;
        let ctx = SeedCtx { persona: &persona, bot_roles: &[], bot_name: "floob", hidden: &none, max: 100 };
        let mut w = ChannelWindow::after(150);
        seed_command_reply(&ctx, &HashMap::new(), &mut w, &answer(200, "> <@5> asks: hi?\nhewwo", None, serde_json::json!([])));
        assert_eq!(w.messages().count(), 0, "the question is before the anchor");
        seed_command_reply(&ctx, &HashMap::new(), &mut w, &answer(201, "and another thing", Some(200), serde_json::json!([])));
        assert_eq!(w.messages().count(), 0, "its follow-up stays out too");
    }

    #[test]
    fn only_a_listed_commands_source_keeps_history() {
        let mut persona = crate::persona::test_persona();
        persona.remember_commands = vec!["roast".into()];
        let asked = |slash: &'static str| {
            Some(Box::new(super::super::remember::Asked {
                slash,
                turn: crate::chat::window::WindowMsg {
                    id: 1,
                    author_id: 5,
                    author: "Alice".into(),
                    text: "[used a command]".into(),
                    extras: Default::default(),
                    mentions: Vec::new(),
                },
                about: Vec::new(),
            }))
        };
        let ask = |a| Source::Ask { author: "Alice".into(), prompt: "p".into(), command: None, draw: None, asked: a };
        let listed = ask(asked("roast"));
        assert_eq!(listed.remembered(&persona).map(|a| a.slash), Some("roast"));
        assert!(listed.keeps_history(&persona));
        let unlisted = ask(asked("judge"));
        assert!(unlisted.remembered(&persona).is_none());
        assert!(!unlisted.keeps_history(&persona));
        let plain = ask(None);
        assert!(plain.remembered(&persona).is_none() && !plain.keeps_history(&persona));
        let channel = Source::Channel { trigger_msg: 1, snapshot: ChannelWindow::default() };
        assert!(channel.remembered(&persona).is_none());
        assert!(channel.keeps_history(&persona), "a channel reply always does");
    }

    #[test]
    fn command_replies_are_recognized() {
        let mut m = serenity::Message::default();
        assert!(!is_command_reply(&m));
        m.kind = serenity::MessageType::InlineReply;
        assert!(!is_command_reply(&m));
        m.kind = serenity::MessageType::ChatInputCommand;
        assert!(is_command_reply(&m));
        // A follow-up can come as a plain message carrying the interaction.
        let mut f = serenity::Message::default();
        f.interaction_metadata = serde_json::from_value(serde_json::json!({
            "id": "1", "type": 2, "user": {"id": "2", "username": "u", "discriminator": "0"},
            "authorizing_integration_owners": {}
        }))
        .expect("interaction metadata");
        assert!(is_command_reply(&f));
    }
}
