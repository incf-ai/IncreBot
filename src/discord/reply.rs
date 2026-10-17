//! Streaming a reply into Discord messages (design.md §6): placeholder, edits
//! every 1.5 s, stopping 30 s in (`chat::edits`), roll-over at ~1,900 characters,
//! typing indicator (non-streaming replies only), ❌ cancel
//! registration, and fallbacks when the trigger, placeholder or interaction token
//! is gone.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

use poise::serenity_prelude as serenity;
use serenity::{
    ChannelId, CreateAllowedMentions, CreateMessage, EditMessage, MessageId,
};
use tokio::sync::mpsc;
use tracing::warn;

use super::dispatch::{Replay, Source};
use super::{Context, Data, NOTICE_PREFIX};
use crate::chat::retry::{Output, Reuse, steer};
use crate::chat::edits::EditSchedule;
use crate::chat::split::{MESSAGE_LIMIT, StreamingReply};
use crate::chat::window::{WindowMsg, snowflake_at};
use crate::engine::StopReason;
use crate::engine::worker::Event;

const TYPING_INTERVAL: Duration = Duration::from_secs(8);
const PLACEHOLDER: &str = "…";

/// The placeholder while the channel's pause holds the reply back.
const RATE_LIMITED: &str = "*(channel rate limit active)*";

/// The placeholder once the pause is over but the reply hasn't started: it
/// waits for jobs ahead of it.
const IN_QUEUE: &str = "*(waiting in queue)*";

/// How long after a pause ends the placeholder waits for the reply to start
/// before saying it is still in the queue (no flash when it starts at once).
const PAUSE_GRACE: Duration = Duration::from_secs(1);

/// What a streaming message shows: "…" until the first text arrives, then just the
/// text so far (Discord rejects empty messages, hence the placeholder).
fn streaming_text(current: &str) -> &str {
    if current.trim().is_empty() {
        PLACEHOLDER
    } else {
        current
    }
}

pub enum Sink<'a> {
    /// Reply to a channel message.
    Channel {
        http: Arc<serenity::Http>,
        channel: ChannelId,
        reply_to: Option<MessageId>,
    },
    /// Answer a deferred `/ask` or novelty command. A `header` (a novelty card)
    /// tops the first message, with the reply on the lines below it; that
    /// message is posted at once and only ever edited back to the card alone.
    Ask {
        ctx: Context<'a>,
        mention: String,
        header: Option<String>,
        /// The message the header is in, once posted.
        header_msg: Option<MessageId>,
        /// Messages posted as the plain fallback (the interaction token
        /// expired): they enter history as plain turns, as a cold start
        /// reads them, and the answer can't be retried.
        fallen: Vec<MessageId>,
    },
    /// `/retry`: the new output goes into the old one's messages, in order;
    /// leftovers are deleted at the end and, if the retry is dropped, messages
    /// it already overwrote get their old text back.
    Redo {
        http: Arc<serenity::Http>,
        channel: ChannelId,
        reuse: Reuse,
        /// A channel reply: the new text replaces the old in the history window.
        history: bool,
        /// An admin's rewrite of the output finished at this snowflake: it
        /// takes that output's place (`retry::Book::finished_replacing`).
        replaces: Option<u64>,
    },
}

/// A message we posted and may edit.
enum Posted<'a> {
    Msg(MessageId),
    Handle(poise::ReplyHandle<'a>, MessageId),
}

impl Posted<'_> {
    fn id(&self) -> MessageId {
        match self {
            Posted::Msg(id) | Posted::Handle(_, id) => *id,
        }
    }
}

fn no_pings() -> CreateAllowedMentions {
    CreateAllowedMentions::new().replied_user(false)
}

impl<'a> Sink<'a> {
    /// A [`Sink::Redo`] over `messages` (id, text), the output being retried.
    pub fn redo(
        http: Arc<serenity::Http>,
        channel: ChannelId,
        header: Option<String>,
        messages: &[(u64, String)],
        history: bool,
    ) -> Self {
        Sink::Redo {
            http,
            channel,
            reuse: Reuse::new(header, messages),
            history,
            replaces: None,
        }
    }

    /// A [`redo`](Self::redo) that rewrites the output finished at
    /// `finished_id`, wherever it is among the remembered ones.
    pub fn replacing(mut self, finished_id: u64) -> Self {
        if let Sink::Redo { replaces, .. } = &mut self {
            *replaces = Some(finished_id);
        }
        self
    }

    fn http(&self) -> Arc<serenity::Http> {
        match self {
            Sink::Channel { http, .. } | Sink::Redo { http, .. } => http.clone(),
            Sink::Ask { ctx, .. } => ctx.serenity_context().http.clone(),
        }
    }

    fn channel(&self) -> ChannelId {
        match self {
            Sink::Channel { channel, .. } | Sink::Redo { channel, .. } => *channel,
            Sink::Ask { ctx, .. } => ctx.channel_id(),
        }
    }

    /// Posts a new message. `first` makes it the reply to the trigger (or the
    /// interaction response); later parts are plain follow-ups.
    async fn post(&mut self, text: &str, first: bool) -> Option<Posted<'a>> {
        match self {
            Sink::Channel {
                http,
                channel,
                reply_to,
            } => {
                let base = CreateMessage::new()
                    .content(text)
                    .allowed_mentions(no_pings());
                if first && let Some(trigger) = *reply_to {
                    match channel
                        .send_message(&**http, base.clone().reference_message((*channel, trigger)))
                        .await
                    {
                        Ok(m) => return Some(Posted::Msg(m.id)),
                        // The trigger was deleted meanwhile: post without a reference.
                        Err(e) => warn!(error = %e, "reply to trigger failed; posting plainly"),
                    }
                    *reply_to = None;
                }
                match channel.send_message(&**http, base).await {
                    Ok(m) => Some(Posted::Msg(m.id)),
                    Err(e) => {
                        warn!(error = %e, "posting a reply failed");
                        None
                    }
                }
            }
            Sink::Redo {
                http,
                channel,
                reuse,
                ..
            } => {
                // The next old message, else (all gone) a new one.
                while let Some(id) = reuse.next_old() {
                    let edit = EditMessage::new()
                        .content(reuse.shown(id, text))
                        .allowed_mentions(no_pings());
                    if channel.edit_message(&**http, MessageId::new(id), edit).await.is_ok() {
                        return Some(Posted::Msg(MessageId::new(id)));
                    }
                    // Deleted meanwhile: it no longer holds anything.
                    reuse.gone(id);
                }
                let msg = CreateMessage::new()
                    .content(reuse.fresh(text, first))
                    .allowed_mentions(no_pings());
                match channel.send_message(&**http, msg).await {
                    Ok(m) => {
                        reuse.posted(m.id.get(), first);
                        Some(Posted::Msg(m.id))
                    }
                    Err(e) => {
                        warn!(error = %e, "posting a retried reply failed");
                        None
                    }
                }
            }
            Sink::Ask {
                ctx,
                mention,
                header,
                header_msg,
                fallen,
            } => {
                let carded = first && header.is_some();
                let text = match header {
                    Some(h) if first => format!("{h}\n{text}"),
                    _ => text.to_string(),
                };
                let reply = poise::CreateReply::default()
                    .content(text.as_str())
                    .allowed_mentions(no_pings());
                match ctx.send(reply).await {
                    Ok(handle) => match handle.message().await {
                        Ok(m) => {
                            let id = m.id;
                            if carded {
                                *header_msg = Some(id);
                            }
                            Some(Posted::Handle(handle, id))
                        }
                        Err(_) => None,
                    },
                    Err(e) => {
                        // The interaction token expired (15 min): fall back to a
                        // channel message that mentions the user (design.md §6).
                        warn!(error = %e, "interaction follow-up failed; posting in the channel");
                        let content = format!("{mention} {text}");
                        let ping = CreateAllowedMentions::new().users(vec![ctx.author().id]);
                        let msg = CreateMessage::new().content(content).allowed_mentions(ping);
                        let posted = ctx
                            .channel_id()
                            .send_message(ctx.http(), msg)
                            .await
                            .ok()
                            .map(|m| Posted::Msg(m.id));
                        if carded && let Some(p) = &posted {
                            *header_msg = Some(p.id());
                        }
                        if let Some(p) = &posted {
                            fallen.push(p.id());
                        }
                        posted
                    }
                }
            }
        }
    }

    /// The header, if `id` is the message it tops.
    fn header_of(&self, id: MessageId) -> Option<&str> {
        match self {
            Sink::Ask {
                header: Some(h),
                header_msg: Some(m),
                ..
            } if *m == id => Some(h),
            Sink::Redo { reuse, .. } => reuse.header_of(id.get()),
            _ => None,
        }
    }

    /// Characters the header takes from the first message's limit.
    fn header_len(&self) -> usize {
        let header = match self {
            Sink::Ask { header, .. } => header.as_deref(),
            Sink::Redo { reuse, .. } => reuse.header(),
            Sink::Channel { .. } => None,
        };
        header.map_or(0, |h| h.chars().count() + 1)
    }

    /// A retry that finished: deletes the old messages it didn't need.
    async fn drop_unused(&mut self) {
        let Sink::Redo { http, channel, reuse, .. } = self else {
            return;
        };
        for id in reuse.leftovers() {
            let _ = channel.delete_message(&**http, MessageId::new(id)).await;
        }
    }

    /// Edits a posted message; false if it no longer exists (or can't be edited).
    /// The header message keeps its header above `text`.
    async fn edit(&self, posted: &Posted<'a>, text: &str) -> bool {
        let carded;
        let text = match self.header_of(posted.id()) {
            Some(h) => {
                carded = format!("{h}\n{text}");
                carded.as_str()
            }
            None => text,
        };
        match (self, posted) {
            (Sink::Ask { ctx, .. }, Posted::Handle(handle, _)) => {
                let reply = poise::CreateReply::default()
                    .content(text)
                    .allowed_mentions(no_pings());
                handle.edit(*ctx, reply).await.is_ok()
            }
            (_, p) => {
                let edit = EditMessage::new()
                    .content(text)
                    .allowed_mentions(no_pings());
                self.channel()
                    .edit_message(&*self.http(), p.id(), edit)
                    .await
                    .is_ok()
            }
        }
    }

    /// Deletes a posted message; the header message goes back to the header alone,
    /// and an old message a retry reused gets its old text back.
    async fn delete(&self, posted: &Posted<'a>) {
        if let Sink::Redo { reuse, .. } = self
            && let Some(old) = reuse.original(posted.id().get())
        {
            if !self.edit(posted, old).await {
                warn!("could not restore a reply after its retry was dropped");
            }
            return;
        }
        if let Some(h) = self.header_of(posted.id()) {
            let edit = EditMessage::new().content(h).allowed_mentions(no_pings());
            let ok = match (self, posted) {
                (Sink::Ask { ctx, .. }, Posted::Handle(handle, _)) => handle
                    .edit(
                        *ctx,
                        poise::CreateReply::default()
                            .content(h)
                            .allowed_mentions(no_pings()),
                    )
                    .await
                    .is_ok(),
                (_, p) => self
                    .channel()
                    .edit_message(&*self.http(), p.id(), edit)
                    .await
                    .is_ok(),
            };
            if !ok {
                warn!("could not restore a card after its reply was dropped");
            }
            return;
        }
        let _ = self
            .channel()
            .delete_message(&*self.http(), posted.id())
            .await;
    }
}

/// Tracks posted messages for ❌ cancel and keeps them out of history while in flight.
struct Registry<'d> {
    data: &'d Data,
    requester: u64,
    cancel: Arc<AtomicBool>,
    ids: Vec<u64>,
}

impl Registry<'_> {
    fn add(&mut self, id: MessageId) {
        let id = id.get();
        self.ids.push(id);
        self.data
            .cancels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, (self.requester, self.cancel.clone()));
        self.data
            .inflight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id);
    }
}

impl Drop for Registry<'_> {
    fn drop(&mut self) {
        let mut cancels = self.data.cancels.lock().unwrap_or_else(|e| e.into_inner());
        let mut inflight = self.data.inflight.lock().unwrap_or_else(|e| e.into_inner());
        for id in &self.ids {
            cancels.remove(id);
            inflight.remove(id);
        }
    }
}

/// Counts a reply as being written in its channel (`chat::retry`) until dropped;
/// `output` is what it leaves for `/retry`, if it finished; `retried` when it
/// replaces the last output.
struct Writing<'d> {
    data: &'d Data,
    channel: u64,
    output: Option<Output<Replay>>,
    retried: bool,
    replaces: Option<u64>,
}

impl Drop for Writing<'_> {
    fn drop(&mut self) {
        let output = self.output.take();
        match self.replaces {
            Some(old) => self.data.retry().finished_replacing(self.channel, output, old),
            None => self.data.retry().finished(self.channel, output, self.retried),
        }
    }
}

/// Drives one reply from engine events to Discord messages. `replay` is how to
/// generate it again with `/retry`. True if it was generated (finished, even
/// stopped); false if it failed or was dropped from the queue.
pub async fn stream<'a>(
    mut sink: Sink<'a>,
    data: Data,
    mut events: mpsc::UnboundedReceiver<Event>,
    cancel: Arc<AtomicBool>,
    requester: u64,
    ahead: usize,
    replay: Replay,
) -> bool {
    data.retry().started(sink.channel().get());
    let mut writing = Writing {
        data: &data,
        channel: sink.channel().get(),
        output: None,
        retried: matches!(sink, Sink::Redo { .. }),
        replaces: match &sink {
            Sink::Redo { replaces, .. } => *replaces,
            _ => None,
        },
    };
    // The messages that end up holding the output, with their text.
    let mut held: Vec<(u64, String)> = Vec::new();
    let stream_edits = data.cfg.discord.stream_edits;
    let mut reg = Registry {
        data: &data,
        requester,
        cancel,
        ids: Vec::new(),
    };
    let mut reply = StreamingReply::default();
    let mut current: Option<Posted<'a>> = None;
    let mut final_parts: Vec<(MessageId, String)> = Vec::new();
    let mut typing: Option<tokio::task::JoinHandle<()>> = None;
    let mut schedule = EditSchedule::default();
    // Text released since the last edit.
    let mut dirty = false;

    // When the channel's pause (plus the grace) ends, while it holds the reply.
    let mut pause_ends: Option<Instant> = None;
    // The channel message this answers, for its trace (`chat::trace`).
    let traced = match &replay.source {
        Source::Channel { trigger_msg, .. } => Some((sink.channel().get(), *trigger_msg)),
        Source::Ask { .. } => None,
    };
    let tr = |e: String| {
        if let Some((ch, m)) = traced {
            data.trace(ch, m, e);
        }
    };
    let mut started = false;
    // A command answer's requester turn, if it goes into the history.
    let remembered = replay.source.remembered(&data.persona().persona).cloned();

    if ahead > 0 {
        let text = format!("{NOTICE_PREFIX}waiting in queue, {ahead} ahead)*");
        if let Some(p) = sink.post(&text, true).await {
            reg.add(p.id());
            current = Some(p);
        }
    } else if sink.header_len() > 0 {
        // The card shows at once, the reply streaming in below it.
        if let Some(p) = sink.post(PLACEHOLDER, true).await {
            reg.add(p.id());
            current = Some(p);
        }
    }

    let outcome: Option<Result<StopReason, String>> = loop {
        // Wake for a scheduled edit only when there is new text to show, and
        // when a pause ends.
        let now = Instant::now();
        let edit_wait = match (stream_edits && dirty, current.is_some()) {
            (true, true) => schedule.wait(now),
            _ => None,
        };
        let pause_wait = pause_ends.map(|u| u.saturating_duration_since(now));
        let wait = match (edit_wait, pause_wait) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let ev = match wait {
            None => events.recv().await,
            Some(wait) => match tokio::time::timeout(wait, events.recv()).await {
                Ok(ev) => ev,
                Err(_) => {
                    let now = Instant::now();
                    if pause_ends.is_some_and(|u| u <= now) {
                        pause_ends = None;
                        if !started && let Some(p) = &current {
                            sink.edit(p, IN_QUEUE).await;
                        }
                    }
                    if schedule.due(now)
                        && let Some(p) = &current
                    {
                        sink.edit(p, streaming_text(&reply.current)).await;
                        schedule.edited(now);
                        dirty = false;
                    }
                    continue;
                }
            },
        };
        match ev {
            // Merged away, dropped from the queue, or shutdown.
            None => break None,
            Some(Event::Started) => {
                tr("generation started".to_string());
                started = true;
                pause_ends = None;
                // The typing indicator only without streaming edits: Discord keeps it
                // up ~10 s after each signal and clears it early only when the bot
                // *sends* a message, so a reply finished by an edit would leave it
                // lingering. While streaming, the "…" placeholder shows the progress.
                if !stream_edits {
                    let (http, channel) = (sink.http(), sink.channel());
                    typing = Some(tokio::spawn(async move {
                        loop {
                            let _ = channel.broadcast_typing(&*http).await;
                            tokio::time::sleep(TYPING_INTERVAL).await;
                        }
                    }));
                }
                if stream_edits {
                    match &current {
                        Some(p) => {
                            sink.edit(p, PLACEHOLDER).await;
                        }
                        None => {
                            if let Some(p) = sink.post(PLACEHOLDER, true).await {
                                reg.add(p.id());
                                current = Some(p);
                            }
                        }
                    }
                } else if let Some(p) = &current {
                    // A status placeholder: it's no longer waiting.
                    sink.edit(p, PLACEHOLDER).await;
                }
            }
            Some(Event::Text(t)) => {
                // The first message shares Discord's limit with the header.
                let limit = if reply.done.is_empty() {
                    MESSAGE_LIMIT - sink.header_len().min(MESSAGE_LIMIT / 2)
                } else {
                    MESSAGE_LIMIT
                };
                for part in reply.push(&t, limit) {
                    let id = finalize(
                        &mut sink,
                        &mut reg,
                        current.take(),
                        &part,
                        final_parts.is_empty(),
                    )
                    .await;
                    if let Some(id) = id {
                        held.push((id.get(), part.clone()));
                        final_parts.push((id, part));
                    }
                    if stream_edits && let Some(p) = sink.post(PLACEHOLDER, false).await {
                        reg.add(p.id());
                        current = Some(p);
                        schedule.new_message(Instant::now());
                    }
                }
                dirty = true;
                let now = Instant::now();
                schedule.text_arrived(now);
                if stream_edits
                    && schedule.due(now)
                    && let Some(p) = &current
                {
                    if !sink.edit(p, streaming_text(&reply.current)).await {
                        // Placeholder deleted: carry on in a new message.
                        current = sink.post(&reply.current, false).await;
                        if let Some(p) = &current {
                            reg.add(p.id());
                        }
                    }
                    schedule.edited(now);
                    dirty = false;
                }
            }
            Some(Event::Paused(until)) => {
                // Held back by the channel's pause: the placeholder says so
                // now, and that it waits in the queue if it is still waiting
                // once the pause is over.
                match &current {
                    Some(p) => {
                        sink.edit(p, RATE_LIMITED).await;
                    }
                    None => {
                        if let Some(p) = sink.post(RATE_LIMITED, true).await {
                            reg.add(p.id());
                            current = Some(p);
                        }
                    }
                }
                pause_ends = Some(until + PAUSE_GRACE);
            }
            Some(Event::Finished(out)) => {
                tr(format!(
                    "generation finished: stopped by {:?}; {} tokens in {:.1} s; prompt {} reused from cache, {} prefilled in {:.1} s",
                    out.stop,
                    out.tokens.len(),
                    out.decode.as_secs_f64(),
                    out.reused_tokens,
                    out.prefill_tokens,
                    out.prefill.as_secs_f64(),
                ));
                // Word biases that made a difference and phrase bans, in the
                // order generation met them.
                let pct = |p: f32| format!("{:.1}%", p * 100.0);
                let redraws: Vec<bool> = out
                    .steered
                    .iter()
                    .filter_map(|s| match s {
                        crate::engine::Steered::WordStart { changed, .. } => Some(*changed),
                        _ => None,
                    })
                    .collect();
                if !redraws.is_empty() {
                    tr(format!(
                        "  word starts: {} drawn again, {} changed",
                        redraws.len(),
                        redraws.iter().filter(|&&c| c).count()
                    ));
                }
                for s in &out.steered {
                    tr(match s {
                        crate::engine::Steered::Bias { at, before, picked, hit } => format!(
                            "  token {at}: word bias \"{}\" ({:+}): {} → {}{}; after \"…{}\", picked {:?}",
                            hit.stem,
                            hit.bias,
                            pct(hit.before),
                            pct(hit.after),
                            match (hit.top_before, hit.top_after) {
                                (true, false) => ", was the top choice",
                                (false, true) => ", now the top choice",
                                (true, true) => ", still the top choice",
                                _ => "",
                            },
                            before.replace('\n', " "),
                            picked,
                        ),
                        crate::engine::Steered::Ban { at, phrase, kept } => format!(
                            "  token {at}: phrase ban \"{phrase}\" {}",
                            if *kept { "matched but kept (already shown, or out of retries)" } else { "rolled back" }
                        ),
                        crate::engine::Steered::WordStart { at, normal, picked, word, changed, normal_prob, picked_prob, odds } => {
                            // "treat" 61.0% → 52.3%, …: how the redraw moved the odds.
                            let moved = odds
                                .iter()
                                .map(|(t, n, h)| format!("{t:?} {} → {}", pct(*n), pct(*h)))
                                .collect::<Vec<_>>()
                                .join(", ");
                            if *changed {
                                format!(
                                    "  token {at}: word start drawn again, changed: {word:?} (token {picked:?}, {}) instead of the normal draw {normal:?} ({}); odds normal → word start: {moved}",
                                    pct(*picked_prob),
                                    pct(*normal_prob)
                                )
                            } else {
                                format!(
                                    "  token {at}: word start drawn again, same: {word:?} ({}); odds normal → word start: {moved}",
                                    pct(*picked_prob)
                                )
                            }
                        }
                    });
                }
                break Some(Ok(out.stop));
            }
            Some(Event::Failed(e)) => {
                tr(format!("generation failed: {e}"));
                break Some(Err(e));
            }
        }
    };
    if let Some(t) = typing {
        t.abort();
    }

    let shutting_down = data.shutting_down.load(Ordering::Relaxed);
    if outcome.is_none() {
        tr("dropped from the queue (cleared, cancelled while waiting, or restarting): no reply".to_string());
    }
    let done = outcome.is_some();
    let generated = matches!(outcome, Some(Ok(_)));
    match outcome {
        None => {
            if let Some(p) = current.take() {
                if shutting_down {
                    sink.edit(
                        &p,
                        &format!("{NOTICE_PREFIX}restarting, ask again shortly)*"),
                    )
                    .await;
                } else {
                    sink.delete(&p).await;
                }
            }
        }
        Some(result) => {
            let body = reply.current.trim().to_string();
            let notice = match &result {
                Ok(StopReason::Cancelled) if shutting_down => {
                    Some("*(stopped: restarting)*".to_string())
                }
                Ok(StopReason::Cancelled) => Some("*(stopped)*".to_string()),
                Ok(_) if body.is_empty() && final_parts.is_empty() => {
                    Some(format!("{NOTICE_PREFIX}no reply)*"))
                }
                Ok(_) => None,
                Err(_) => Some(format!("{NOTICE_PREFIX}error: generation failed)*")),
            };
            let text = match (&notice, body.is_empty()) {
                (Some(n), true) => n.clone(),
                (Some(n), false) => format!("{body}\n{n}"),
                (None, _) => body.clone(),
            };
            let first = final_parts.is_empty();
            if let Some(id) = finalize(&mut sink, &mut reg, current.take(), &text, first).await {
                held.push((id.get(), text.clone()));
                if !body.is_empty() {
                    final_parts.push((id, body));
                }
            }
            sink.drop_unused().await;
            tr(match &sink {
                Sink::Redo { .. } => format!("written over the earlier reply ({} message(s))", held.len()),
                _ => format!("posted in {} message(s)", held.len()),
            });
            // Finished (even stopped or failed): it can be retried — unless
            // a part fell back to a plain message (the interaction token
            // expired): a retry of that would edit plain messages a cold
            // start reads differently.
            let fell = matches!(&sink, Sink::Ask { fallen, .. } if !fallen.is_empty());
            if !held.is_empty() && !fell {
                let said = final_parts.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join("\n");
                writing.output = Some(Output {
                    replay: Replay {
                        avoid: steer(replay.avoid, &said),
                        ..replay
                    },
                    requester,
                    messages: held.clone(),
                    turn: final_parts.iter().map(|(id, t)| (id.get(), t.clone())).collect(),
                    finished_at: Instant::now(),
                    finished_id: snowflake_at(SystemTime::now()),
                });
            }
        }
    }

    // The final reply enters the channel's history as the bot's turn(s); the
    // gateway handler ignores the bot's own messages (design.md §6). A retried
    // channel reply replaces the old turn. A remembered command's answer
    // enters with its requester's turn, even if only the card is left.
    let parts: Vec<(u64, String)> = final_parts.into_iter().map(|(id, t)| (id.get(), t)).collect();
    match &sink {
        Sink::Channel { channel, .. } => write_turn(&data, channel.get(), &[], parts, None).await,
        Sink::Ask { ctx, mention, header, header_msg: Some(m), fallen, .. } => {
            // A part that fell back to a plain message enters history as the
            // plain turn a cold start reads it back as: its posted content
            // (mention and card included), mentions resolved, notices out.
            let fallen_turns: Vec<(u64, String)> = held
                .iter()
                .filter(|(id, _)| fallen.iter().any(|f| f.get() == *id))
                .filter_map(|(id, text)| {
                    let posted = match header {
                        Some(h) if *id == m.get() => format!("{mention} {h}
{text}"),
                        _ => format!("{mention} {text}"),
                    };
                    let resolved = super::resolve_mentions(
                        &posted,
                        &super::interaction_users(*ctx),
                        &[],
                        &data.cfg.discord.bot_name,
                    );
                    let text = super::remember::without_notices(&resolved);
                    (!text.is_empty()).then_some((*id, text))
                })
                .collect();
            if !fallen_turns.is_empty() {
                write_turn(&data, ctx.channel_id().get(), &[], fallen_turns, None).await;
            }
            // The exchange is remembered only while its header message went
            // out through the interaction (a fallen one has no metadata, so
            // a cold start can't rebuild it).
            if let Some(a) = &remembered
                && !fallen.iter().any(|f| f == m)
            {
                let parts = parts.into_iter().filter(|(id, _)| !fallen.iter().any(|f| f.get() == *id)).collect();
                remember(&data, ctx.channel_id().get(), a, m.get(), parts).await;
            }
        }
        // Dropped, the old messages got their old text back: the turn stays.
        Sink::Redo { channel, history: true, reuse, .. } if done => {
            let tied = remembered.as_ref().map(|a| a.turn.id);
            write_turn(&data, channel.get(), &reuse.old_ids(), parts, tied).await;
        }
        _ => {}
    }
    drop(reg);
    drop(writing);
    generated
}

/// `/undo-retry`: writes `prev`'s text into `cur`'s messages, in order (the card
/// stays on the first), posting or deleting messages where their counts differ,
/// and for a channel reply puts its turn back in the history window. Returns
/// the output now shown, with `cur`'s replay (its steering already covers both),
/// or None if nothing could be written.
pub async fn restore(
    http: Arc<serenity::Http>,
    channel: ChannelId,
    data: &Data,
    cur: &Output<Replay>,
    prev: Output<Replay>,
) -> Option<Output<Replay>> {
    let history = cur.replay.source.keeps_history(&data.persona().persona);
    let mut sink = Sink::redo(http, channel, cur.replay.header.clone(), &cur.messages, history);
    let mut messages = Vec::new();
    let mut turn = Vec::new();
    for (old, text) in &prev.messages {
        let Some(p) = sink.post(text, messages.is_empty()).await else {
            continue;
        };
        let id = p.id().get();
        messages.push((id, text.clone()));
        if let Some((_, t)) = prev.turn.iter().find(|(o, _)| o == old) {
            turn.push((id, t.clone()));
        }
    }
    if messages.is_empty() {
        return None;
    }
    sink.drop_unused().await;
    if let Sink::Redo { reuse, history: true, .. } = &sink {
        let tied = cur.replay.source.remembered(&data.persona().persona).map(|a| a.turn.id);
        write_turn(data, channel.get(), &reuse.old_ids(), turn.clone(), tied).await;
    }
    Some(Output {
        replay: cur.replay.clone(),
        requester: cur.requester,
        messages,
        turn,
        finished_at: Instant::now(),
        finished_id: snowflake_at(SystemTime::now()),
    })
}

/// Puts the bot's turn `parts` (message id, text) in `channel`'s history
/// window in place of the messages `old`, if the channel has one. With
/// `tied`, a command answer's retry, only while its requester's turn is
/// still there (it may have left, or never entered).
async fn write_turn(data: &Data, channel: u64, old: &[u64], parts: Vec<(u64, String)>, tied: Option<u64>) {
    let mut windows = data.windows.lock().await;
    let Some(w) = windows.get_mut(&channel) else { return };
    write_turn_in(w, data.bot_id(), &data.cfg.discord.bot_name, old, parts, tied, crate::chat::window::max_buffer(data.cfg.discord.history_messages));
}

/// [`write_turn`] on the window itself: what the tests (the metamorphic ones
/// in `discord::simtest` included) exercise directly.
pub(super) fn write_turn_in(
    w: &mut crate::chat::window::ChannelWindow,
    bot: u64,
    bot_name: &str,
    old: &[u64],
    parts: Vec<(u64, String)>,
    tied: Option<u64>,
    max: usize,
) {
    let turn = parts
        .into_iter()
        .map(|(id, text)| WindowMsg {
            id,
            author_id: bot,
            author: bot_name.to_string(),
            text,
            extras: Default::default(),
            mentions: Vec::new(),
        })
        .collect();
    w.replace_turn_tied(old, turn, tied, max);
}

/// A remembered command's answer enters `channel`'s history (if it has a
/// window there): `asked`, the requester's turn, tied to `message`, the one
/// the header tops, then the bot's `parts`.
pub(super) async fn remember(data: &Data, channel: u64, asked: &super::remember::Asked, message: u64, parts: Vec<(u64, String)>) {
    let mut windows = data.windows.lock().await;
    let Some(w) = windows.get_mut(&channel) else { return };
    remember_in(
        w,
        &|u| data.optouts.is_out(u),
        data.bot_id(),
        &data.cfg.discord.bot_name,
        asked,
        message,
        parts,
        crate::chat::window::max_buffer(data.cfg.discord.history_messages),
    );
}

/// [`remember`] on the window itself, under one lock (a delete can't land
/// between the requester's turn and the bot's): the tests' entry point.
#[allow(clippy::too_many_arguments)]
pub(super) fn remember_in(
    w: &mut crate::chat::window::ChannelWindow,
    is_out: &dyn Fn(u64) -> bool,
    bot: u64,
    bot_name: &str,
    asked: &super::remember::Asked,
    message: u64,
    parts: Vec<(u64, String)>,
    max: usize,
) {
    // Checked when the answer lands, not when it was asked: anyone involved
    // opting out meanwhile keeps the exchange out, as a cold start will.
    if is_out(asked.turn.author_id) || asked.about.iter().any(|&u| is_out(u)) {
        return;
    }
    w.push_tied(asked.turn.clone(), message, asked.about.clone(), max);
    write_turn_in(w, bot, bot_name, &[], parts, Some(asked.turn.id), max);
}

/// Writes final text into `posted` (or a new message if there is none or the edit
/// fails). Returns the id of the message that holds it.
async fn finalize<'a>(
    sink: &mut Sink<'a>,
    reg: &mut Registry<'_>,
    posted: Option<Posted<'a>>,
    text: &str,
    first: bool,
) -> Option<MessageId> {
    if let Some(p) = posted
        && sink.edit(&p, text).await
    {
        return Some(p.id());
    }
    let p = sink.post(text, first).await?;
    reg.add(p.id());
    Some(p.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::window::ChannelWindow;

    fn asked_by(author_id: u64, about: Vec<u64>) -> crate::discord::remember::Asked {
        crate::discord::remember::Asked {
            slash: "roast",
            turn: WindowMsg {
                id: 10,
                author_id,
                author: "Alice".into(),
                text: "[used /roast]".into(),
                extras: Default::default(),
                mentions: Vec::new(),
            },
            about,
        }
    }

    #[test]
    fn a_remembered_exchange_enters_as_two_turns() {
        let mut w = ChannelWindow::default();
        let none = |_: u64| false;
        remember_in(&mut w, &none, 99, "floob", &asked_by(5, vec![6]), 11, vec![(11, "burn".into())], 80);
        let got: Vec<(u64, u64, String)> = w.messages().map(|m| (m.id, m.author_id, m.text.clone())).collect();
        assert_eq!(got, [(10, 5, "[used /roast]".into()), (11, 99, "burn".into())]);
        // A retry through write_turn_in replaces only the bot's turn.
        write_turn_in(&mut w, 99, "floob", &[11], vec![(11, "worse burn".into())], Some(10), 80);
        assert_eq!(w.messages().map(|m| m.text.as_str()).last(), Some("worse burn"));
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn whoever_opted_out_keeps_the_exchange_out() {
        // The requester, or anyone the card is about, each on their own.
        for out in [5u64, 6] {
            let mut w = ChannelWindow::default();
            let is_out = |u: u64| u == out;
            remember_in(&mut w, &is_out, 99, "floob", &asked_by(5, vec![6]), 11, vec![(11, "burn".into())], 80);
            assert!(w.is_empty(), "user {out} opted out");
        }
    }

    #[test]
    fn placeholder_only_until_the_first_text() {
        assert_eq!(streaming_text(""), "…");
        assert_eq!(streaming_text("  "), "…");
        assert_eq!(streaming_text("hewwo"), "hewwo");
        assert!(!streaming_text("hewwo fren").contains('…'));
    }
}
