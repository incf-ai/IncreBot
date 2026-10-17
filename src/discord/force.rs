//! "Admin actions" (design.md §6): a right-click (Apps) entry on a message,
//! opening an ephemeral panel. On any of the channel's last `retry::RECENT`
//! outputs: retry, batch-retry, edit or delete it. On it or any of the
//! channel's last traced messages (`chat::trace`): Trace, the log of what
//! happened to the message (the one a reply answers), as a text file; and
//! Context, the exact prompt of the reply to it with what went with it (the
//! note, idle scene, steering, sampler), as another. Admins only, under none of `/retry`'s rules
//! but the channel's lock (`chat::retry`): rewritten in place, it keeps its
//! requester and finish, with no undo.

use std::time::{Duration, Instant};

use poise::serenity_prelude as serenity;
use serenity::{ComponentInteraction, CreateActionRow};
use tracing::{debug, info};

use super::dispatch::{Replay, Source};
use super::shake::cut;
use super::{Context, Data, Error, NOTICE_PREFIX, batch, busy, notice, require_admin, requeue, without_optouts};
use crate::chat::retry::{Output, Refusal};

/// How long the panel stays open.
const TIMEOUT: Duration = Duration::from_secs(600);

/// The edit form's limit (Discord's for a text field); longer replies are cut.
const FORM_LIMIT: usize = 4000;

/// An answer's lifetime meaning "delete the panel now" (0 keeps it).
const GONE: u64 = 1;

/// Why an admin action can't go ahead.
pub(super) fn refusal(r: Refusal) -> &'static str {
    match r {
        Refusal::Retrying => "A retry, undo, edit or delete is already running here; try again once it's done.",
        _ => "That isn't a reply the bot still remembers: only the last 20 in each channel since the last restart.",
    }
}

/// The edit form, filled in with the reply.
#[derive(poise::Modal)]
#[name = "Edit reply"]
struct EditForm {
    #[name = "Text (no pings)"]
    #[paragraph]
    #[max_length = 4000] // FORM_LIMIT
    text: String,
}

/// Holds the channel's lock for an admin action until dropped.
struct Held(Data, u64);

impl Drop for Held {
    fn drop(&mut self) {
        self.0.retry().end_force(self.1);
    }
}

/// Takes the channel's lock for an admin action on `out`, returning the
/// output as it is now; or the reason it can't.
fn hold(data: &Data, channel: u64, out: &Output<Replay>) -> Result<(Held, Output<Replay>), &'static str> {
    let now = data.retry().begin_force(channel, out.finished_id).map_err(refusal)?;
    Ok((Held(data.clone(), channel), now))
}

/// Retry, batch-retry, edit, delete, trace or see the context of this message (admin only).
#[poise::command(
    context_menu_command = "Admin actions",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub(super) async fn admin_actions(ctx: Context<'_>, msg: serenity::Message) -> Result<(), Error> {
    if !require_admin(ctx).await? {
        return Ok(());
    }
    let data = ctx.data().clone();
    if data.shutting_down.load(std::sync::atomic::Ordering::Relaxed) {
        notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await?;
        return Ok(());
    }
    let channel = ctx.channel_id().get();
    let found = data.retry().find(channel, msg.id.get());
    // The message whose trace Trace sends: the one a remembered reply
    // answers, else this one.
    let traced = match found.as_ref().map(|o| &o.replay.source) {
        Some(Source::Channel { trigger_msg, .. }) => *trigger_msg,
        _ => msg.id.get(),
    };
    let has_trace = data.traces.render(channel, traced).is_some();
    let has_context = data.traces.context(channel, traced).is_some();
    if found.is_none() && !has_trace {
        let why = "Nothing to do here: that isn't one of the bot's last 20 replies here, \
                   nor one of this channel's last 20 messages since startup.";
        return notice(ctx, why).await;
    }
    let prefix = ctx.id().to_string();
    let id = |part: &str| format!("{prefix}-{part}");
    let button = |part: &str, label: &str, style| serenity::CreateButton::new(id(part)).label(label).style(style);
    let no_reply = found.is_none();
    let buttons = vec![
        CreateActionRow::Buttons(vec![
            button("retry", "Retry", serenity::ButtonStyle::Primary).disabled(no_reply),
            button("batch", "Batch retry", serenity::ButtonStyle::Primary).disabled(no_reply),
            button("edit", "Edit", serenity::ButtonStyle::Secondary).disabled(no_reply),
            button("delete", "Delete", serenity::ButtonStyle::Danger).disabled(no_reply),
        ]),
        CreateActionRow::Buttons(vec![
            button("trace", "Trace", serenity::ButtonStyle::Secondary).disabled(!has_trace),
            button("context", "Context", serenity::ButtonStyle::Secondary).disabled(!has_context),
            button("close", "Close", serenity::ButtonStyle::Secondary),
        ]),
    ];
    let shown: String = match &found {
        Some(out) => out.messages.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join("\n"),
        None => msg.content.clone(),
    };
    let what = if no_reply { "message" } else { "reply" };
    let mut content = format!("What to do with this {what}?\n> {}", cut(&shown, 300).replace('\n', "\n> "));
    if no_reply {
        content.push_str("\n-# Not one of the bot's remembered replies: only Trace and Context.");
    }
    let handle = ctx
        .send(poise::CreateReply::default().ephemeral(true).content(content).components(buttons))
        .await?;
    let http = ctx.serenity_context().http.clone();
    let d = &data.cfg.discord;
    let deadline = Instant::now() + TIMEOUT;
    let form_id = id("form");
    let say = |text: &str| poise::CreateReply::default().content(text).components(Vec::new());
    let admin = ctx.author().id.get();
    let log = |action: &str| {
        info!(admin, channel, action, "admin action");
        data.trace(channel, traced, format!("admin action by user {admin}: {action}"));
    };

    let ttl_ms = loop {
        let p = format!("{prefix}-");
        let clicks = serenity::ComponentInteractionCollector::new(&ctx.serenity_context().shard)
            .filter(move |i| i.data.custom_id.starts_with(&p))
            .timeout(deadline.saturating_duration_since(Instant::now()));
        let f = form_id.clone();
        let forms = serenity::ModalInteractionCollector::new(&ctx.serenity_context().shard)
            .filter(move |m| m.data.custom_id == f)
            .timeout(deadline.saturating_duration_since(Instant::now()));
        let click: ComponentInteraction = tokio::select! {
            Some(click) = clicks.into_future() => click,
            Some(submit) = forms.into_future() => {
                submit.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
                let text = <EditForm as poise::Modal>::parse(submit.data.clone())
                    .map(|f| f.text.trim().to_string())
                    .unwrap_or_default();
                if text.is_empty() {
                    handle.edit(ctx, say("Nothing to say: to remove it, use Delete.")).await?;
                    break d.notice_ttl_ms;
                }
                let Some(out) = &found else { continue };
                let (held, now) = match hold(&data, channel, out) {
                    Ok(h) => h,
                    Err(why) => {
                        handle.edit(ctx, say(why)).await?;
                        break d.notice_ttl_ms;
                    }
                };
                handle.edit(ctx, say("…")).await?;
                batch::write_over(&data, http.clone(), ctx.channel_id(), &now, now.replay.clone(), text, true).await;
                drop(held);
                log("edit");
                handle.edit(ctx, say("Edited.")).await?;
                break d.retrying_ttl_ms;
            }
            else => break GONE,
        };
        let part = click.data.custom_id.strip_prefix(&format!("{prefix}-")).unwrap_or_default();
        if part == "trace" || part == "context" {
            // Sent as its own message, so the panel stays.
            let (text, file) = if part == "trace" {
                let text = data.traces.render(channel, traced);
                (text.unwrap_or_else(|| "That message is no longer traced.\n".to_string()), "trace.txt")
            } else {
                let text = data.traces.context(channel, traced);
                (text.unwrap_or_else(|| "That reply's context is no longer kept.\n".to_string()), "context.txt")
            };
            let msg = serenity::CreateInteractionResponseMessage::new()
                .ephemeral(true)
                .content(cut(text.lines().next().unwrap_or_default(), 1900))
                .allowed_mentions(serenity::CreateAllowedMentions::new())
                .add_file(serenity::CreateAttachment::bytes(text.into_bytes(), file));
            click.create_response(&http, serenity::CreateInteractionResponse::Message(msg)).await?;
            // Logged, but not into the trace it showed.
            info!(admin, channel, action = part, "admin action");
            continue;
        }
        if part == "edit" {
            // Opened on each click; closing it leaves the panel.
            let text: String = shown.chars().take(FORM_LIMIT).collect();
            let form = <EditForm as poise::Modal>::create(Some(EditForm { text }), form_id.clone());
            click.create_response(&http, form).await?;
            continue;
        }
        click.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
        match part {
            "close" => break GONE,
            "batch" => {
                let Some(out) = found.clone() else { continue };
                log("batch-retry");
                break batch::force(ctx, &handle, out).await?;
            }
            "retry" => {
                let Some(out) = &found else { continue };
                let (held, now) = match hold(&data, channel, out) {
                    Ok(h) => h,
                    Err(why) => {
                        handle.edit(ctx, say(why)).await?;
                        break d.notice_ttl_ms;
                    }
                };
                let mut source = now.replay.source.clone();
                if !without_optouts(&data, &mut source) {
                    handle.edit(ctx, say(refusal(Refusal::Nothing))).await?;
                    break d.notice_ttl_ms;
                }
                let job = requeue(&data, http.clone(), ctx.channel_id(), ctx.id(), &now, source, Some(now.finished_id));
                let Some(job) = job.await else {
                    handle.edit(ctx, say(&busy())).await?;
                    break d.notice_ttl_ms;
                };
                // The lock goes when the new text is in.
                tokio::spawn(async move {
                    job.await;
                    drop(held);
                });
                log("retry");
                handle.edit(ctx, say("Retrying…")).await?;
                break d.retrying_ttl_ms;
            }
            "delete" => {
                let Some(out) = &found else { continue };
                let (held, now) = match hold(&data, channel, out) {
                    Ok(h) => h,
                    Err(why) => {
                        handle.edit(ctx, say(why)).await?;
                        break d.notice_ttl_ms;
                    }
                };
                for (id, _) in &now.messages {
                    // Already deleted by hand: nothing left to do for it.
                    if let Err(e) = ctx.channel_id().delete_message(&http, serenity::MessageId::new(*id)).await {
                        debug!("deleting a reply message: {e}");
                    }
                }
                // Gone from the history now, not when the gateway's delete events arrive.
                if let Some(w) = data.windows.lock().await.get_mut(&channel) {
                    for (id, _) in &now.messages {
                        w.remove(*id);
                    }
                }
                data.retry().remove(channel, now.finished_id);
                drop(held);
                log("delete");
                handle.edit(ctx, say("Deleted.")).await?;
                break d.retrying_ttl_ms;
            }
            _ => {}
        }
    };
    if ttl_ms > 0 {
        tokio::time::sleep(Duration::from_millis(ttl_ms)).await;
        if let Err(e) = handle.delete(ctx).await {
            debug!("deleting the admin panel: {e}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::retry::RECENT;

    #[test]
    fn the_refusal_names_the_limit() {
        assert!(refusal(Refusal::Nothing).contains(&RECENT.to_string()));
    }
}
