//! "Reply actions" (design.md §6): a right-click (Apps) entry on a message,
//! for anyone, opening an ephemeral panel with `/retry`, `/retry-batch`,
//! `/undo-retry` and `/delete-reply` for that reply, plus Stop while it is
//! still being written (as ❌). The slash commands' rules apply, and they act
//! only on the channel's latest reply; what can't be done now shows greyed
//! out with the reason, and is checked again on the click. Opening it isn't
//! channel activity.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use poise::serenity_prelude as serenity;
use serenity::CreateActionRow;
use tracing::{debug, info};

use super::dispatch::DispatchMsg;
use super::{Context, Error, batch, begin_retry, do_delete, do_retry, do_undo, notice, restarting};
use crate::chat::gate::is_admin;

/// How long the panel stays open.
const TIMEOUT: Duration = Duration::from_secs(600);

/// An answer's lifetime meaning "delete the panel now" (0 keeps it).
const GONE: u64 = 1;

const NOT_LATEST: &str =
    "Only the latest reply in a channel can be retried, undone or deleted, by the person who asked for it.";

const NOT_YOURS_TO_STOP: &str = "Only the person who asked for this reply can stop it.";

/// Retry, undo, delete or stop this reply, if it's yours.
#[poise::command(
    context_menu_command = "Reply actions",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub(super) async fn reply_actions(ctx: Context<'_>, msg: serenity::Message) -> Result<(), Error> {
    let data = ctx.data().clone();
    if let Some(n) = restarting(&data) {
        return notice(ctx, n).await;
    }
    let (channel, user, id) = (ctx.channel_id().get(), ctx.author().id.get(), msg.id.get());
    let admin = is_admin(user, &data.cfg.discord);
    let now = Instant::now();

    // Stop: while it's being written, for the requester or an admin.
    let writing = data.cancels.lock().unwrap_or_else(|e| e.into_inner()).get(&id).cloned();
    let stop = writing.as_ref().filter(|(requester, _)| *requester == user || admin).map(|(_, flag)| flag.clone());

    // The rest: only on the latest reply, under the slash commands' rules.
    let latest = data.retry().is_last(channel, id);
    let (retry, undo, delete) = if latest {
        let (book, rules) = (data.retry(), &data.cfg.discord.retry);
        (
            book.can_retry(channel, user, now).map_err(|r| r.message(rules)),
            book.can_undo(channel, user, now).map_err(|r| r.undo_message(rules)),
            book.can_delete(channel, user).map_err(|r| r.delete_message(rules)),
        )
    } else {
        let no = || Err(NOT_LATEST.to_string());
        (no(), no(), no())
    };
    let batch = retry.clone().and_then(|()| batch::gate(&data, user, now));
    let retry = retry.and_then(|()| match data.limit_wait(user) {
        Some(wait) => Err(crate::chat::limits::message(wait)),
        None => Ok(()),
    });

    let actions = [
        ("retry", "Retry", serenity::ButtonStyle::Primary, &retry),
        ("batch", "Batch retry", serenity::ButtonStyle::Primary, &batch),
        ("undo", "Undo retry", serenity::ButtonStyle::Secondary, &undo),
        ("delete", "Delete", serenity::ButtonStyle::Danger, &delete),
    ];
    if stop.is_none() && actions.iter().all(|(.., r)| r.is_err()) {
        // Nothing to offer: just why.
        let why = match (&writing, latest) {
            (Some(_), _) => NOT_YOURS_TO_STOP.to_string(),
            (None, false) => NOT_LATEST.to_string(),
            (None, true) => delete.err().unwrap_or_default(),
        };
        return notice(ctx, why).await;
    }

    // Why each greyed-out one is, those with the same reason on one line.
    let mut why: Vec<(Vec<&str>, &str)> = Vec::new();
    for (_, label, _, r) in &actions {
        if let Err(reason) = r {
            match why.iter_mut().find(|(_, w)| w == reason) {
                Some((labels, _)) => labels.push(label),
                None => why.push((vec![label], reason)),
            }
        }
    }
    let mut content = "What to do with this reply?".to_string();
    for (labels, reason) in &why {
        content.push_str(&format!("\n-# {}: {reason}", labels.join(", ")));
    }
    let prefix = ctx.id().to_string();
    let cid = |part: &str| format!("{prefix}-{part}");
    let mut buttons: Vec<serenity::CreateButton> = actions
        .iter()
        .map(|(part, label, style, r)| {
            serenity::CreateButton::new(cid(part)).label(*label).style(*style).disabled(r.is_err())
        })
        .collect();
    if stop.is_some() {
        buttons.push(serenity::CreateButton::new(cid("stop")).label("Stop").style(serenity::ButtonStyle::Danger));
    }
    let close = serenity::CreateButton::new(cid("close")).label("Close").style(serenity::ButtonStyle::Secondary);
    let components = vec![CreateActionRow::Buttons(buttons), CreateActionRow::Buttons(vec![close])];
    let handle = ctx
        .send(poise::CreateReply::default().ephemeral(true).content(content).components(components))
        .await?;

    let http = ctx.serenity_context().http.clone();
    let d = &data.cfg.discord;
    let deadline = Instant::now() + TIMEOUT;
    let say = |text: &str| poise::CreateReply::default().content(text).components(Vec::new());
    let ttl_ms = loop {
        let p = format!("{prefix}-");
        let click = serenity::ComponentInteractionCollector::new(&ctx.serenity_context().shard)
            .filter(move |i| i.data.custom_id.starts_with(&p))
            .timeout(deadline.saturating_duration_since(Instant::now()))
            .await;
        let Some(click) = click else { break GONE };
        click.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
        let part = click.data.custom_id.strip_prefix(&format!("{prefix}-")).unwrap_or_default();
        if matches!(part, "retry" | "batch" | "undo" | "delete") && !data.retry().is_last(channel, id) {
            handle.edit(ctx, say(NOT_LATEST)).await?;
            break d.notice_ttl_ms;
        }
        let (text, ttl) = match part {
            "close" => break GONE,
            "stop" => {
                let Some(flag) = &stop else { continue };
                info!(message = id, by = user, "cancel requested");
                flag.store(true, Ordering::Relaxed);
                let _ = data.dispatch.send(DispatchMsg::Cancel(flag.clone()));
                ("Stopping.".to_string(), d.retrying_ttl_ms)
            }
            "batch" => {
                if let Err(why) = batch::gate(&data, user, Instant::now()) {
                    (why, d.notice_ttl_ms)
                } else {
                    handle.edit(ctx, say("…")).await?;
                    match begin_retry(&data, channel, user).await {
                        Ok((out, source)) => break batch::own(ctx, &handle, out, source).await?,
                        Err(why) => (why, d.notice_ttl_ms),
                    }
                }
            }
            "retry" | "undo" | "delete" => {
                handle.edit(ctx, say("…")).await?;
                let ch = ctx.channel_id();
                match part {
                    "retry" => do_retry(&data, http.clone(), ch, user, ctx.id()).await,
                    "undo" => do_undo(&data, http.clone(), ch, user).await,
                    _ => do_delete(&data, http.clone(), ch, user).await,
                }
            }
            _ => continue,
        };
        handle.edit(ctx, say(&text)).await?;
        break ttl;
    };
    if ttl_ms > 0 {
        tokio::time::sleep(Duration::from_millis(ttl_ms)).await;
        if let Err(e) = handle.delete(ctx).await {
            debug!("deleting the reply-actions panel: {e}");
        }
    }
    Ok(())
}
