//! `/retry-batch` (design.md §6): `/retry`, but [`VERSIONS`] versions are
//! generated without posting, and the requester picks one from a dropdown in
//! an ephemeral panel, which applies it at once (or Cancel). GPU only: refused unless a GPU is
//! ready when it is sent (a GPU lost meanwhile just makes it slower).
//!
//! The retry rules (`chat::retry`) are checked when it is sent and again at
//! the pick, bar the time limit there; the channel's retry lock is held throughout,
//! so `/retry`, `/undo-retry` and `/delete-reply` wait. Once a version came
//! out, the requester waits [`COOLDOWN`] before the next one (admins too,
//! unless `limits.admins_exempt`).
//!
//! An admin's batch ([`force`], from `discord::force`) works on any remembered
//! output under none of those rules but the lock, makes [`VERSIONS_CPU`]
//! versions when no GPU is ready, and at the pick only checks the output is still
//! remembered; it rewrites it in place, with no undo.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use poise::serenity_prelude as serenity;
use serenity::{ComponentInteraction, CreateActionRow};
use tokio::sync::mpsc;
use tracing::{debug, info};

use super::dispatch::{DispatchMsg, EnqueueResult, Replay, Source, Trigger};
use super::shake::cut;
use super::{Context, Data, Error, begin_retry, notice, reply};
use crate::chat::queue::QueueKey;
use crate::chat::limits;
use crate::chat::retry::{self, Output, steer};
use crate::engine::worker::Event;
use crate::engine::{GenerateOutput, StopReason};

/// How many versions to pick from; at most 25, a dropdown's limit.
const VERSIONS: usize = 5;

/// An admin's batch without a GPU: fewer, as the CPU is slow.
const VERSIONS_CPU: usize = 3;

/// How long the panel stays open, generating included.
const TIMEOUT: Duration = Duration::from_secs(600);

/// How long someone waits between batches that
/// generated anything, in any channel.
pub(super) const COOLDOWN: Duration = Duration::from_secs(120);

/// Discord's message length limit; the previews share it.
const DISCORD_LIMIT: usize = 2000;

/// An answer's lifetime meaning "delete the panel now" (0 keeps it).
const GONE: u64 = 1;

/// How long `user` still waits before another batch, if at all.
fn cooldown_left(data: &Data, user: u64, now: Instant) -> Option<Duration> {
    let at = *data.batch_cooldowns.lock().unwrap_or_else(|e| e.into_inner()).get(&user)?;
    COOLDOWN.checked_sub(now.saturating_duration_since(at)).filter(|d| !d.is_zero())
}

/// Holds the channel's retry lock until dropped, however the batch ends.
/// The requester's batch is billed when it starts (`hold`, `batch_cost`
/// toward the limits, and [`COOLDOWN`] from then; neither for those
/// `limits_exempt`) and
/// keeps the bill once `ran`: a version came out and the pick wasn't refused
/// (cancelled or timed out afterwards still counts). Otherwise both are
/// taken back. `force`: an admin's batch, never billed.
struct Lock<'d> {
    data: &'d Data,
    channel: u64,
    user: u64,
    ran: bool,
    force: bool,
    hold: Option<limits::Hold>,
}

impl Drop for Lock<'_> {
    fn drop(&mut self) {
        if self.force {
            self.data.retry().end_force(self.channel);
            return;
        }
        self.data.retry().end(self.channel);
        if !self.ran {
            self.data.limit_release(self.hold.take());
            self.data
                .batch_cooldowns
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&self.user);
        }
    }
}

/// Generate your last reply here 5 ways and pick one (GPU only).
#[poise::command(
    slash_command,
    rename = "retry-batch",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub(super) async fn retry_batch(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data().clone();
    if let Some(n) = super::restarting(&data) {
        return notice(ctx, n).await;
    }
    let user = ctx.author().id.get();
    if let Err(why) = gate(&data, user, Instant::now()) {
        return notice(ctx, why).await;
    }
    // Ephemeral, so the channel gets no new message and stays retryable.
    ctx.defer_ephemeral().await?;
    let (out, source) = match begin_retry(&data, ctx.channel_id().get(), user).await {
        Ok(begun) => begun,
        Err(why) => return notice(ctx, &why).await,
    };
    let handle = ctx
        .send(poise::CreateReply::default().ephemeral(true).content(progress(0, VERSIONS)))
        .await?;
    let ttl_ms = own(ctx, &handle, out, source).await?;
    if ttl_ms > 0 {
        tokio::time::sleep(Duration::from_millis(ttl_ms)).await;
        if let Err(e) = handle.delete(ctx).await {
            debug!("deleting the retry-batch panel: {e}");
        }
    }
    Ok(())
}

/// Why `user` can't start a batch now, beyond `/retry`'s rules: no GPU ready,
/// or [`COOLDOWN`] still running (unless `limits_exempt`).
pub(super) fn gate(data: &Data, user: u64, now: Instant) -> Result<(), String> {
    if !data.gpus.as_ref().is_some_and(|g| g.any_ready()) {
        return Err("This needs the GPU, which isn't ready right now. /retry still works.".to_string());
    }
    if let Some(wait) = data.limit_wait(user) {
        return Err(crate::chat::limits::message(wait));
    }
    if !data.limits_exempt(user)
        && let Some(left) = cooldown_left(data, user, now)
    {
        return Err(format!("Wait {} more seconds before another batch retry.", left.as_secs().max(1)));
    }
    Ok(())
}

/// The requester's batch on `out` (from [`begin_retry`], its lock held), in
/// the panel `handle`. Returns how long the final answer stays.
pub(super) async fn own(
    ctx: Context<'_>,
    handle: &poise::ReplyHandle<'_>,
    out: Output<Replay>,
    source: Source,
) -> Result<u64, Error> {
    let data = ctx.data().clone();
    let user = ctx.author().id.get();
    // Billed now: kept unless nothing comes of it.
    let hold = data.limit_hold(user, data.cfg.discord.limits.batch_cost);
    if !data.limits_exempt(user) {
        data.batch_cooldowns
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(user, Instant::now());
    }
    let mut lock = Lock {
        data: &data,
        channel: ctx.channel_id().get(),
        user,
        ran: false,
        force: false,
        hold,
    };
    run(ctx, &mut lock, handle, out, source, VERSIONS).await
}

/// An admin's batch on `out` (a remembered output), in the panel `handle`
/// (already answered for the click that asked). Returns how long the final
/// answer stays.
pub(super) async fn force(
    ctx: Context<'_>,
    handle: &poise::ReplyHandle<'_>,
    out: Output<Replay>,
) -> Result<u64, Error> {
    let data = ctx.data().clone();
    let channel = ctx.channel_id().get();
    let d = &data.cfg.discord;
    let answer = |s: &str| poise::CreateReply::default().content(s).components(Vec::new());
    let begun = data.retry().begin_force(channel, out.finished_id);
    let out = match begun {
        Ok(out) => out,
        Err(r) => {
            handle.edit(ctx, answer(&super::force::refusal(r, data.cfg.discord.retry.admin_recent))).await?;
            return Ok(d.notice_ttl_ms);
        }
    };
    let mut lock = Lock {
        data: &data,
        channel,
        user: ctx.author().id.get(),
        ran: false,
        force: true,
        hold: None,
    };
    let mut source = out.replay.source.clone();
    if !super::without_optouts(&data, &mut source) {
        handle.edit(ctx, answer(&super::force::refusal(retry::Refusal::Nothing, data.cfg.discord.retry.admin_recent))).await?;
        return Ok(d.notice_ttl_ms);
    }
    let on_gpu = data.gpus.as_ref().is_some_and(|g| g.any_ready());
    let count = if on_gpu { VERSIONS } else { VERSIONS_CPU };
    run(ctx, &mut lock, handle, out, source, count).await
}

fn progress(n: usize, count: usize) -> String {
    format!("Generating {count} versions… ({n}/{count})")
}

/// The panel's component IDs: the command's interaction ID, then a part.
struct Ids(String);

impl Ids {
    fn id(&self, part: &str) -> String {
        format!("{}-{part}", self.0)
    }

    fn part<'a>(&self, custom_id: &'a str) -> Option<&'a str> {
        custom_id.strip_prefix(&self.0)?.strip_prefix('-')
    }

    fn cancel(&self) -> serenity::CreateButton {
        serenity::CreateButton::new(self.id("cancel"))
            .label("Cancel")
            .style(serenity::ButtonStyle::Secondary)
    }
}

/// The next click or pick on the panel before `deadline`.
async fn next_click(ctx: Context<'_>, ids: &Ids, deadline: Instant) -> Option<ComponentInteraction> {
    let prefix = format!("{}-", ids.0);
    serenity::ComponentInteractionCollector::new(&ctx.serenity_context().shard)
        .filter(move |i| i.data.custom_id.starts_with(&prefix))
        .timeout(deadline.saturating_duration_since(Instant::now()))
        .await
}

/// Replaces the panel's message with `content` alone.
fn update(content: impl Into<String>) -> serenity::CreateInteractionResponse {
    serenity::CreateInteractionResponse::UpdateMessage(
        serenity::CreateInteractionResponseMessage::new()
            .content(content)
            .embeds(Vec::new())
            .components(Vec::new()),
    )
}

/// Generates the versions, offers them, and writes the picked one over
/// `out`'s messages. Returns how long the final answer stays.
async fn run(
    ctx: Context<'_>,
    lock: &mut Lock<'_>,
    handle: &poise::ReplyHandle<'_>,
    out: Output<Replay>,
    source: Source,
    count: usize,
) -> Result<u64, Error> {
    let data = lock.data;
    let force = lock.force;
    let d = &data.cfg.discord;
    let http = ctx.serenity_context().http.clone();
    let (channel, user) = (lock.channel, lock.user);
    let ids = Ids(ctx.id().to_string());
    let deadline = Instant::now() + TIMEOUT;
    let replay = Replay {
        source: source.clone(),
        ..out.replay.clone()
    };
    let edit = |content: String, components: Vec<CreateActionRow>| {
        poise::CreateReply::default().content(content).components(components)
    };
    let only_cancel = || vec![CreateActionRow::Buttons(vec![ids.cancel()])];
    handle.edit(ctx, edit(progress(0, count), only_cancel())).await?;
    let tr = |e: String| data.trace_source(channel, &out.replay.source, e);
    tr(format!(
        "{} by user {user}: generating {count} versions",
        if force { "admin batch retry" } else { "/retry-batch" }
    ));

    // One after another, each steered away from the reply and those before.
    let mut versions: Vec<Option<String>> = Vec::new();
    let mut avoid = replay.avoid.clone();
    while versions.len() < count {
        let cancel = Arc::new(AtomicBool::new(false));
        let (events_tx, mut events) = mpsc::unbounded_channel();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let trigger = Trigger {
            key: match source {
                Source::Channel { .. } => QueueKey::Channel(channel),
                Source::Ask { .. } => QueueKey::Ask(rand::random()),
            },
            channel,
            source: source.clone(),
            requester: out.requester,
            cancel: cancel.clone(),
            events: events_tx,
            avoid: avoid.clone(),
            paced: false,
        };
        let sent = data.dispatch.send(DispatchMsg::Enqueue(Box::new(trigger), result_tx));
        let queued = match sent {
            Ok(()) => matches!(result_rx.await, Ok(EnqueueResult::Queued { .. })),
            Err(_) => false,
        };
        if !queued {
            // Offer what came out so far, if anything.
            if versions.iter().any(Option::is_some) {
                break;
            }
            let busy = super::busy();
            handle.edit(ctx, edit(busy, Vec::new())).await?;
            return Ok(d.notice_ttl_ms);
        }
        let mut text = String::new();
        // One watcher for the whole job, so no click falls between events.
        let mut click = std::pin::pin!(next_click(ctx, &ids, deadline));
        let version = loop {
            tokio::select! {
                ev = events.recv() => match ev {
                    Some(Event::Text(t)) => text.push_str(&t),
                    Some(Event::Started | Event::Paused(_)) => {}
                    Some(Event::Finished(_)) => break Some(text.trim().to_string()).filter(|t| !t.is_empty()),
                    // Failed, or dropped from the queue (`/super-admin clearqueue`, shutdown).
                    Some(Event::Failed(_)) | None => break None,
                },
                click = &mut click => {
                    // Cancel (the only button now), or the panel timed out:
                    // stop this job, or drop it from the queue.
                    cancel.store(true, Ordering::Relaxed);
                    let _ = data.dispatch.send(DispatchMsg::Cancel(cancel.clone()));
                    let Some(click) = click else { return Ok(GONE) };
                    click.create_response(&http, update("Cancelled.")).await?;
                    return Ok(d.retrying_ttl_ms);
                }
            }
        };
        if let Some(v) = &version {
            avoid = steer(avoid, v, data.cfg.discord.retry.steer);
            lock.ran = true;
        }
        versions.push(version);
        handle.edit(ctx, edit(progress(versions.len(), count), only_cancel())).await?;
    }

    if versions.iter().all(Option::is_none) {
        handle
            .edit(ctx, edit("Couldn't come up with anything.".to_string(), Vec::new()))
            .await?;
        return Ok(d.notice_ttl_ms);
    }
    let mut content = "Pick one to replace the reply:".to_string();
    // All the previews fit one message.
    let room = ((DISCORD_LIMIT - 100) / versions.len() - 10).min(500);
    for (i, v) in versions.iter().enumerate() {
        let shown = match v {
            Some(v) => cut(v, room),
            None => "*(failed)*".to_string(),
        };
        content.push_str(&format!("\n**{}.** {shown}", i + 1));
    }
    // Failed ones aren't offered: an option can't be disabled.
    let options = versions
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let label = cut(&format!("{}. {}", i + 1, v.as_deref()?), 95);
            Some(serenity::CreateSelectMenuOption::new(label, format!("v{i}")))
        })
        .collect();
    let menu = serenity::CreateSelectMenu::new(ids.id("pick"), serenity::CreateSelectMenuKind::String { options })
        .placeholder("Which one?");
    let components = vec![
        CreateActionRow::SelectMenu(menu),
        CreateActionRow::Buttons(vec![ids.cancel()]),
    ];
    handle.edit(ctx, edit(content, components)).await?;

    // Picking one applies it.
    let (pick, chosen, text) = loop {
        let Some(click) = next_click(ctx, &ids, deadline).await else {
            return Ok(GONE);
        };
        let picked = match (ids.part(&click.data.custom_id).unwrap_or_default(), &click.data.kind) {
            ("pick", serenity::ComponentInteractionDataKind::StringSelect { values }) => values
                .first()
                .and_then(|v| v.strip_prefix('v'))
                .and_then(|n| n.parse::<usize>().ok())
                .and_then(|n| Some((n, versions.get(n)?.clone()?))),
            _ => None,
        };
        if let Some((n, text)) = picked {
            click.create_response(&http, update("…")).await?;
            break (click, n, text);
        }
        if ids.part(&click.data.custom_id) == Some("cancel") {
            tr("batch retry cancelled at the pick".to_string());
            click.create_response(&http, update("Cancelled.")).await?;
            return Ok(d.retrying_ttl_ms);
        }
        // Not ours to act on; Discord still needs an answer.
        click.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
    };
    let answer = |s: &str| serenity::EditInteractionResponse::new().content(s);

    if force {
        // Only that it's still there: the admin skips the rules.
        if !data.retry().still_there(channel, out.finished_id) {
            lock.ran = false;
            tr("batch retry: pick refused, the reply is no longer remembered".to_string());
            pick.edit_response(&http, answer(&super::force::refusal(retry::Refusal::Nothing, data.cfg.discord.retry.admin_recent))).await?;
            return Ok(d.notice_ttl_ms);
        }
    } else {
        // The rules again, bar the time limit.
        let check = data.retry().still_last(channel, &out);
        if let Err(r) = check {
            // Refused: not charged.
            lock.ran = false;
            let why = r.message(&d.retry);
            tr(format!("batch retry: pick refused: {why}"));
            pick.edit_response(&http, answer(&why)).await?;
            return Ok(d.notice_ttl_ms);
        }
    }

    tr(format!("batch retry: version {} picked", chosen + 1));
    write_over(data, http.clone(), ctx.channel_id(), &out, replay, text, force).await;
    // Who, never what (PRIVACY.md).
    info!(user, channel, force, "retry-batch");
    pick.edit_response(&http, answer("Done.")).await?;
    Ok(d.retrying_ttl_ms)
}

/// Writes `text` over `out`'s messages as if it had just been generated (a
/// `/retry` with `replay`): split, history and the retry bookkeeping as
/// usual; with `force`, an admin's rewrite in place.
pub(super) async fn write_over(
    data: &Data,
    http: Arc<serenity::Http>,
    channel: serenity::ChannelId,
    out: &Output<Replay>,
    replay: Replay,
    text: String,
    force: bool,
) {
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let _ = events_tx.send(Event::Text(text));
    let _ = events_tx.send(Event::Finished(GenerateOutput {
        tokens: Vec::new(),
        stop: StopReason::Eos,
        reused_tokens: 0,
        prefill_tokens: 0,
        prefill: Duration::ZERO,
        decode: Duration::ZERO,
        rollbacks: Vec::new(),
        steered: Vec::new(),
    }));
    drop(events_tx);
    let history = replay.source.keeps_history(&data.persona().persona);
    let mut sink = reply::Sink::redo(http, channel, replay.header.clone(), &out.messages, history);
    if force {
        sink = sink.replacing(out.finished_id);
    }
    let cancel = Arc::new(AtomicBool::new(false));
    reply::stream(sink, data.clone(), events_rx, cancel, out.requester, 0, replay).await;
}
