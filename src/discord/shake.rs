//! `/super-admin shake` (design.md §6): the bot says something in a channel
//! named in config, with nothing showing that an admin asked. An ephemeral
//! panel picks the channel and the kind of message ([`Kind`]); the message is
//! posted as a plain one and enters that channel's history as the bot's turn.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use poise::serenity_prelude as serenity;
use serenity::{ChannelId, ComponentInteraction, ComponentInteractionDataKind, CreateActionRow};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::dispatch::{DispatchMsg, EnqueueResult, Source, Trigger};
use super::{Context, Error, NOTICE_PREFIX, notice, persona_name};
use crate::chat::queue::QueueKey;
use crate::chat::window::{ChannelWindow, WindowMsg};
use crate::engine::worker::Event;
use crate::persona::{CommandDraw, CommandName, render_command};

/// Discord's message length limit (replies split earlier, at
/// `split::MESSAGE_LIMIT`); the form allows no more.
const DISCORD_LIMIT: usize = 2000;

/// How long the panel stays open, generating included.
const TIMEOUT: Duration = Duration::from_secs(600);

/// How many lines [`Kind::Bored`] generates to pick from, each from a
/// different set-up line: fewer when no GPU is ready, as the CPU is slow. At
/// most 25, a dropdown's limit.
const BORED_LINES_CPU: usize = 3;
const BORED_LINES_GPU: usize = 10;

/// How many lines [`Kind::Prompted`] and [`Kind::Directed`] generate, all
/// from the admin's set-up line: fewer when no GPU is ready.
const PROMPTED_LINES_CPU: usize = 3;
const PROMPTED_LINES_GPU: usize = 5;

/// Who "says" a bored line's set-up: nobody in the chat.
const NARRATOR: &str = "narrator";

/// The set-up line when the persona has no `Bored` command.
const BORED_PROMPT: &str =
    "*nobody has said anything in here for a long time. {bot} is all alone and says something to the empty chat to get someone to talk.*";

/// An answer's lifetime meaning "delete the panel now" (0 keeps it).
const GONE: u64 = 1;

/// What the bot says: the panel's second dropdown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// The admin's text, exactly.
    Literal,
    /// [`BORED_LINES_CPU`] or [`BORED_LINES_GPU`] generated lines (persona
    /// command `Bored`); the admin picks one from a dropdown or cancels.
    Bored,
    /// As [`Kind::Bored`], but every line answers a set-up line the admin
    /// types ([`PROMPTED_LINES_CPU`] or [`PROMPTED_LINES_GPU`] of them).
    Prompted,
    /// As [`Kind::Prompted`], but with the persona command `Directed` (its
    /// note and examples) in place of `Bored`'s, which would pull every line
    /// back to the empty chat: the bot does what the set-up line says.
    Directed,
}

impl Kind {
    const ALL: &[Kind] = &[Kind::Literal, Kind::Bored, Kind::Prompted, Kind::Directed];

    fn id(self) -> &'static str {
        match self {
            Kind::Literal => "literal",
            Kind::Bored => "bored",
            Kind::Prompted => "prompted",
            Kind::Directed => "directed",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Kind::Literal => "literal: type it",
            Kind::Bored => "bored: pick from generated lines",
            Kind::Prompted => "bored: enter a prompt, pick from lines",
            Kind::Directed => "directed: say what it does, pick from lines",
        }
    }

    fn from_id(id: &str) -> Option<Kind> {
        Self::ALL.iter().copied().find(|k| k.id() == id)
    }

    /// The persona command its generated lines use.
    fn command(self) -> CommandName {
        match self {
            Kind::Directed => CommandName::Directed,
            _ => CommandName::Bored,
        }
    }
}

/// The literal form; Discord caps labels at 45 characters.
#[derive(poise::Modal)]
#[name = "shake"]
struct LiteralForm {
    #[name = "Text (posted exactly, no pings)"]
    #[paragraph]
    #[max_length = 2000] // DISCORD_LIMIT
    text: String,
}

/// The bored form: the picked line, filled in, to edit before posting.
#[derive(poise::Modal)]
#[name = "shake"]
struct BoredForm {
    #[name = "Line (edit before posting, no pings)"]
    #[paragraph]
    #[max_length = 2000] // DISCORD_LIMIT
    text: String,
}

/// The prompted form: the set-up line every line answers, in place of the
/// persona's.
#[derive(poise::Modal)]
#[name = "shake"]
struct PromptForm {
    #[name = "Set-up line ({bot} = the bot's name)"]
    #[placeholder = "*it's raining. {bot} is bored and says something to the empty chat.*"]
    #[paragraph]
    #[max_length = 1000]
    prompt: String,
}

/// The directed form: what the bot does, in place of any set-up line of the
/// persona's.
#[derive(poise::Modal)]
#[name = "shake"]
struct DirectedForm {
    #[name = "Set-up line ({bot} = the bot's name)"]
    #[placeholder = "*{bot} is furious at the goblins who stole his shiny and yells insults at them.*"]
    #[paragraph]
    #[max_length = 1000]
    prompt: String,
}

/// How long the bot shows as typing before posting: 1 s plus 1 s per 100
/// characters, at most 4 s.
fn typing_time(chars: usize) -> Duration {
    Duration::from_millis((1000 + 10 * chars as u64).min(4000))
}

/// The channels offered: those named in config (`dedicated_channels`,
/// `shared_channels`; not their threads, not DMs), labelled with their config
/// name, else their Discord one. At most 25, a dropdown's limit.
async fn channels(ctx: Context<'_>) -> Vec<(ChannelId, String)> {
    let d = &ctx.data().cfg.discord;
    let mut out: Vec<(ChannelId, String)> = Vec::new();
    for c in d.dedicated_channels.iter().chain(&d.shared_channels) {
        if c.id == 0 || out.iter().any(|(id, _)| id.get() == c.id) {
            continue;
        }
        let id = ChannelId::new(c.id);
        let label = if c.name.is_empty() {
            match id.to_channel(ctx).await {
                Ok(serenity::Channel::Guild(g)) => format!("#{}", g.name),
                _ => c.id.to_string(),
            }
        } else {
            c.name.clone()
        };
        out.push((id, label.chars().take(90).collect()));
    }
    if out.len() > 25 {
        warn!(channels = out.len(), "shake: only the first 25 channels fit the dropdown");
        out.truncate(25);
    }
    out
}

/// The panel: what's picked so far, and its components' IDs.
struct Panel {
    /// Prefix of every component ID: the command's interaction ID.
    prefix: String,
    channels: Vec<(ChannelId, String)>,
    here: ChannelId,
    channel: Option<ChannelId>,
    kind: Option<Kind>,
}

impl Panel {
    fn id(&self, part: &str) -> String {
        format!("{}-{part}", self.prefix)
    }

    /// The part of a component ID after the prefix, if it is ours.
    fn part<'a>(&self, custom_id: &'a str) -> Option<&'a str> {
        custom_id.strip_prefix(&self.prefix)?.strip_prefix('-')
    }

    fn components(&self) -> Vec<CreateActionRow> {
        let channels = self
            .channels
            .iter()
            .map(|(id, label)| {
                let label = if *id == self.here { format!("{label} (here)") } else { label.clone() };
                serenity::CreateSelectMenuOption::new(label, id.get().to_string())
                    .default_selection(self.channel == Some(*id))
            })
            .collect();
        let kinds = Kind::ALL
            .iter()
            .map(|k| serenity::CreateSelectMenuOption::new(k.label(), k.id()).default_selection(self.kind == Some(*k)))
            .collect();
        let menu = |id: String, options, placeholder: &str| {
            CreateActionRow::SelectMenu(
                serenity::CreateSelectMenu::new(id, serenity::CreateSelectMenuKind::String { options })
                    .placeholder(placeholder),
            )
        };
        vec![
            menu(self.id("channel"), channels, "Channel"),
            menu(self.id("kind"), kinds, "What to say"),
            CreateActionRow::Buttons(vec![
                serenity::CreateButton::new(self.id("go"))
                    .label("Go")
                    .style(serenity::ButtonStyle::Primary)
                    .disabled(self.channel.is_none() || self.kind.is_none()),
                cancel_button(self),
            ]),
        ]
    }

    /// Notes a dropdown pick; false if it wasn't one of the panel's options.
    fn pick(&mut self, part: &str, value: &str) -> bool {
        match part {
            "channel" => {
                let id = value.parse::<u64>().ok();
                self.channel = self.channels.iter().map(|(c, _)| *c).find(|c| Some(c.get()) == id);
                self.channel.is_some()
            }
            "kind" => {
                self.kind = Kind::from_id(value);
                self.kind.is_some()
            }
            _ => false,
        }
    }
}

fn cancel_button(panel: &Panel) -> serenity::CreateButton {
    serenity::CreateButton::new(panel.id("cancel"))
        .label("Cancel")
        .style(serenity::ButtonStyle::Secondary)
}

/// Replaces the panel's message with `content` and `components`.
fn update(content: impl Into<String>, components: Vec<CreateActionRow>) -> serenity::CreateInteractionResponse {
    serenity::CreateInteractionResponse::UpdateMessage(
        serenity::CreateInteractionResponseMessage::new()
            .content(content)
            .components(components),
    )
}

/// The value a dropdown pick chose, if `i` is one.
fn picked(i: &ComponentInteraction) -> Option<&str> {
    match &i.data.kind {
        ComponentInteractionDataKind::StringSelect { values } => values.first().map(String::as_str),
        _ => None,
    }
}

/// The next click or pick on the panel before `deadline`.
async fn next_click(ctx: Context<'_>, prefix: &str, deadline: Instant) -> Option<ComponentInteraction> {
    let prefix = format!("{prefix}-");
    serenity::ComponentInteractionCollector::new(&ctx.serenity_context().shard)
        .filter(move |i| i.data.custom_id.starts_with(&prefix))
        .timeout(deadline.saturating_duration_since(Instant::now()))
        .await
}

/// The admin action (admin only; checked by `/super-admin`).
pub(super) async fn shake(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data().clone();
    if data.shutting_down.load(Ordering::Relaxed) {
        notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await?;
        return Ok(());
    }
    // Labels may need fetching: answer within Discord's 3 s first.
    ctx.defer_ephemeral().await?;
    let channels = channels(ctx).await;
    if channels.is_empty() {
        notice(ctx, "No channels are set up for this.").await?;
        return Ok(());
    }
    let mut panel = Panel {
        prefix: ctx.id().to_string(),
        channels,
        here: ctx.channel_id(),
        channel: None,
        kind: None,
    };
    let reply = poise::CreateReply::default()
        .ephemeral(true)
        .components(panel.components());
    let handle = ctx.send(reply).await?;
    let http = ctx.serenity_context().http.clone();
    let deadline = Instant::now() + TIMEOUT;
    let form_id = panel.id("form");
    let prompt_id = panel.id("prompt");
    let directed_id = panel.id("directed");

    // Picks, Go and Cancel until something is posted or given up on. Closing
    // the literal or prompt form leaves the panel, so Go can open it again.
    let ttl_ms = loop {
        let ids = [form_id.clone(), prompt_id.clone(), directed_id.clone()];
        let forms = serenity::ModalInteractionCollector::new(&ctx.serenity_context().shard)
            .filter(move |m| ids.contains(&m.data.custom_id))
            .timeout(deadline.saturating_duration_since(Instant::now()));
        let click = tokio::select! {
            Some(click) = next_click(ctx, &panel.prefix, deadline) => click,
            Some(submit) = forms.into_future() => {
                let Some(ch) = panel.channel else { continue };
                if submit.data.custom_id == prompt_id {
                    submit.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
                    let prompt = <PromptForm as poise::Modal>::parse(submit.data.clone())
                        .map(|f| f.prompt.trim().to_string())
                        .unwrap_or_default();
                    break bored(ctx, &panel, &handle, ch, deadline, Kind::Prompted, Some(prompt).filter(|p| !p.is_empty())).await?;
                }
                if submit.data.custom_id == directed_id {
                    submit.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
                    let prompt = <DirectedForm as poise::Modal>::parse(submit.data.clone())
                        .map(|f| f.prompt.trim().to_string())
                        .unwrap_or_default();
                    // Required, so never empty; were it, `Bored`'s lines would do.
                    let kind = if prompt.is_empty() { Kind::Bored } else { Kind::Directed };
                    break bored(ctx, &panel, &handle, ch, deadline, kind, Some(prompt).filter(|p| !p.is_empty())).await?;
                }
                // Replaces the panel, so nothing is posted twice.
                submit.create_response(&http, update("…", Vec::new())).await?;
                let text = <LiteralForm as poise::Modal>::parse(submit.data.clone())
                    .map(|f| f.text)
                    .unwrap_or_default();
                let (answer, ttl_ms) = post(ctx, ch, text, Kind::Literal).await;
                submit.edit_response(&http, serenity::EditInteractionResponse::new().content(answer)).await?;
                break ttl_ms;
            }
            else => {
                // Timed out: the panel goes.
                let _ = handle.delete(ctx).await;
                return Ok(());
            }
        };
        let part = panel.part(&click.data.custom_id).unwrap_or_default().to_string();
        match (part.as_str(), panel.channel, panel.kind) {
            ("cancel", ..) => {
                click.create_response(&http, update("Cancelled.", Vec::new())).await?;
                break data.cfg.discord.retrying_ttl_ms;
            }
            ("go", Some(_), Some(Kind::Literal)) => {
                let form = <LiteralForm as poise::Modal>::create(None, form_id.clone());
                click.create_response(&http, form).await?;
            }
            ("go", Some(_), Some(Kind::Prompted)) => {
                let form = <PromptForm as poise::Modal>::create(None, prompt_id.clone());
                click.create_response(&http, form).await?;
            }
            ("go", Some(_), Some(Kind::Directed)) => {
                let form = <DirectedForm as poise::Modal>::create(None, directed_id.clone());
                click.create_response(&http, form).await?;
            }
            ("go", Some(ch), Some(Kind::Bored)) => {
                click.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
                break bored(ctx, &panel, &handle, ch, deadline, Kind::Bored, None).await?;
            }
            _ => {
                if let Some(value) = picked(&click) {
                    panel.pick(&part, value);
                }
                click.create_response(&http, update("", panel.components())).await?;
            }
        }
    };
    if ttl_ms > 0 {
        tokio::time::sleep(Duration::from_millis(ttl_ms)).await;
        if let Err(e) = handle.delete(ctx).await {
            debug!("deleting the shake panel: {e}");
        }
    }
    Ok(())
}

/// [`Kind::Bored`], [`Kind::Prompted`] and [`Kind::Directed`], after Go (or
/// the prompt form), in the panel `handle`: generates [`BORED_LINES_CPU`] lines one after
/// another ([`BORED_LINES_GPU`] while a GPU is ready), each from a different
/// set-up line of the persona's `Bored` command; with `prompt`,
/// [`PROMPTED_LINES_CPU`] or [`PROMPTED_LINES_GPU`], all from that set-up
/// line instead (the command's note and examples as usual; `kind`'s command,
/// `Bored` or `Directed`). Each is steered
/// away from the lines before; progress shows meanwhile. Then it offers them
/// in a dropdown; "Edit & send" opens the picked one in a form to edit before
/// posting. Closing the form leaves the panel, so the button can open it
/// again. Returns how long the final answer stays.
async fn bored(
    ctx: Context<'_>,
    panel: &Panel,
    handle: &poise::ReplyHandle<'_>,
    ch: ChannelId,
    deadline: Instant,
    kind: Kind,
    prompt: Option<String>,
) -> Result<u64, Error> {
    let data = ctx.data();
    let http = ctx.serenity_context().http.clone();
    let d = &data.cfg.discord;
    let on_gpu = data.gpus.as_ref().is_some_and(|g| g.any_ready());
    let count = match (&prompt, on_gpu) {
        (None, true) => BORED_LINES_GPU,
        (None, false) => BORED_LINES_CPU,
        (Some(_), true) => PROMPTED_LINES_GPU,
        (Some(_), false) => PROMPTED_LINES_CPU,
    };
    let progress = |n: usize| format!("Thinking up {count} lines for <#{}>… ({n}/{count})", ch.get());
    let only_cancel = || vec![CreateActionRow::Buttons(vec![cancel_button(panel)])];
    let show = |content: String, components| poise::CreateReply::default().content(content).components(components);
    handle.edit(ctx, show(progress(0), only_cancel())).await?;

    let command = kind.command();
    let draws = data.persona().persona.draws(command, count, rand::random());
    let mut lines: Vec<Option<String>> = Vec::new();
    for i in 0..count {
        let cancel = Arc::new(AtomicBool::new(false));
        let avoid = lines.iter().flatten().cloned().collect();
        let mut draw = draws.get(i).cloned().unwrap_or_default();
        if let Some(p) = &prompt {
            draw.template = p.clone();
        }
        let mut events = match generate(ctx, ch, cancel.clone(), avoid, command, draw).await {
            Some(events) => events,
            None => {
                let busy = super::busy();
                handle.edit(ctx, show(busy, Vec::new())).await?;
                return Ok(d.notice_ttl_ms);
            }
        };
        let mut text = String::new();
        // One watcher for the whole job, so no click falls between events.
        let mut click = std::pin::pin!(next_click(ctx, &panel.prefix, deadline));
        let line = loop {
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
                    click.create_response(&http, update("Cancelled.", Vec::new())).await?;
                    return Ok(d.retrying_ttl_ms);
                }
            }
        };
        lines.push(line);
        handle.edit(ctx, show(progress(lines.len()), only_cancel())).await?;
    }

    if lines.iter().all(Option::is_none) {
        handle.edit(ctx, show("Couldn't come up with anything.".to_string(), Vec::new())).await?;
        return Ok(d.notice_ttl_ms);
    }
    let mut content = format!("Pick one for <#{}>:", ch.get());
    // All of them fit Discord's 2000 characters.
    let room = (DISCORD_LIMIT - 100) / lines.len() - 10;
    let mut options = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let shown = match line {
            Some(l) => cut(l, room.min(500)),
            None => "*(failed)*".to_string(),
        };
        content.push_str(&format!("\n**{}.** {shown}", i + 1));
        // Failed ones aren't offered: an option can't be disabled.
        if let Some(l) = line {
            options.push((i, cut(&format!("{}. {l}", i + 1), 95)));
        }
    }
    let components = |chosen: Option<usize>| {
        let options = options
            .iter()
            .map(|(i, label)| {
                serenity::CreateSelectMenuOption::new(label, format!("line{i}")).default_selection(chosen == Some(*i))
            })
            .collect();
        let menu = serenity::CreateSelectMenu::new(panel.id("line"), serenity::CreateSelectMenuKind::String { options })
            .placeholder("Post which one?");
        vec![
            CreateActionRow::SelectMenu(menu),
            CreateActionRow::Buttons(vec![
                serenity::CreateButton::new(panel.id("edit"))
                    .label("Edit & send")
                    .style(serenity::ButtonStyle::Primary)
                    .disabled(chosen.is_none()),
                cancel_button(panel),
            ]),
        ]
    };
    let mut chosen: Option<usize> = None;
    handle.edit(ctx, show(content.clone(), components(chosen))).await?;

    let form_id = panel.id("form");
    loop {
        let f = form_id.clone();
        let forms = serenity::ModalInteractionCollector::new(&ctx.serenity_context().shard)
            .filter(move |m| m.data.custom_id == f)
            .timeout(deadline.saturating_duration_since(Instant::now()));
        let click = tokio::select! {
            Some(click) = next_click(ctx, &panel.prefix, deadline) => click,
            Some(submit) = forms.into_future() => {
                // Replaces the panel, so nothing is posted twice.
                submit.create_response(&http, update("…", Vec::new())).await?;
                let text = <BoredForm as poise::Modal>::parse(submit.data.clone())
                    .map(|f| f.text)
                    .unwrap_or_default();
                let edited = chosen.and_then(|n| lines.get(n).cloned().flatten()).as_deref() != Some(text.as_str());
                debug!(edited, "shake: bored line submitted");
                let (answer, ttl_ms) = post(ctx, ch, text, kind).await;
                submit.edit_response(&http, serenity::EditInteractionResponse::new().content(answer)).await?;
                return Ok(ttl_ms);
            }
            else => return Ok(GONE),
        };
        let part = panel.part(&click.data.custom_id).unwrap_or_default();
        match part {
            "cancel" => {
                click.create_response(&http, update("Cancelled.", Vec::new())).await?;
                return Ok(d.retrying_ttl_ms);
            }
            "line" => {
                let n = picked(&click)
                    .and_then(|v| v.strip_prefix("line"))
                    .and_then(|n| n.parse::<usize>().ok())
                    .filter(|n| lines.get(*n).is_some_and(Option::is_some));
                if n.is_some() {
                    chosen = n;
                }
                click.create_response(&http, update(content.clone(), components(chosen))).await?;
            }
            "edit" if chosen.is_some() => {
                let text = chosen.and_then(|n| lines.get(n).cloned().flatten()).unwrap_or_default();
                // The form can't be opened with more than it allows.
                let form = BoredForm { text: text.chars().take(DISCORD_LIMIT).collect() };
                let form = <BoredForm as poise::Modal>::create(Some(form), form_id.clone());
                click.create_response(&http, form).await?;
            }
            _ => {
                // Not ours to act on; Discord still needs an answer.
                click.create_response(&http, serenity::CreateInteractionResponse::Acknowledge).await?;
            }
        }
    }
}

/// `text`, cut to `max` characters with "…" when longer.
pub(super) fn cut(text: &str, max: usize) -> String {
    if text.chars().count() > max {
        format!("{}…", text.chars().take(max.saturating_sub(1)).collect::<String>())
    } else {
        text.to_string()
    }
}

/// Queues one line of `command` for `ch` from `draw` (its set-up line and
/// any examples), steered away from `avoid`; its events, or None when the
/// queue is full.
async fn generate(
    ctx: Context<'_>,
    ch: ChannelId,
    cancel: Arc<AtomicBool>,
    avoid: Vec<String>,
    command: CommandName,
    draw: CommandDraw,
) -> Option<mpsc::UnboundedReceiver<Event>> {
    let data = ctx.data();
    let template = Some(draw.template.clone())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| BORED_PROMPT.to_string());
    let prompt = render_command(&template, &[("bot", &persona_name(ctx))]);
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let trigger = Trigger {
        // One-shot, like `/ask`; never merged with anything.
        key: QueueKey::Ask(rand::random()),
        channel: ch.get(),
        source: Source::Ask {
            author: NARRATOR.to_string(),
            prompt,
            command: Some(command),
            draw: Some(draw),
            asked: None,
        },
        requester: ctx.author().id.get(),
        cancel,
        events: events_tx,
        avoid,
        paced: false,
    };
    data.dispatch.send(DispatchMsg::Enqueue(Box::new(trigger), result_tx)).ok()?;
    match result_rx.await.ok()? {
        EnqueueResult::Queued { .. } => Some(events_rx),
        EnqueueResult::Full => None,
    }
}

/// Posts `text` in `ch` as a plain message, so nothing shows who asked, after
/// showing the bot typing; adds it to the history as the bot's turn. Returns
/// the admin's answer and how long it stays (ms).
async fn post(ctx: Context<'_>, ch: ChannelId, text: String, kind: Kind) -> (String, u64) {
    let data = ctx.data();
    let d = &data.cfg.discord;
    let chars = text.chars().count();
    if text.trim().is_empty() {
        return ("Nothing to say.".to_string(), d.notice_ttl_ms);
    }
    if chars > DISCORD_LIMIT {
        let msg = format!("Too long: {chars} characters, Discord allows {DISCORD_LIMIT}.");
        return (msg, d.notice_ttl_ms);
    }
    let http = &ctx.serenity_context().http;
    let _ = ch.broadcast_typing(http).await;
    tokio::time::sleep(typing_time(chars)).await;
    let msg = serenity::CreateMessage::new()
        .content(text.as_str())
        .allowed_mentions(serenity::CreateAllowedMentions::new());
    let m = match ch.send_message(http, msg).await {
        Ok(m) => m,
        Err(e) => {
            warn!(error = %e, "shake: posting failed");
            return ("Couldn't post it.".to_string(), d.notice_ttl_ms);
        }
    };
    // A reply before it there is no longer the newest thing (`pre_command`
    // only saw the command's channel).
    data.retry().note_activity(ch.get(), m.id.get());
    // The gateway skips the bot's own messages: add it here.
    data.windows
        .lock()
        .await
        .entry(ch.get())
        .or_insert_with(|| ChannelWindow::after(data.start_anchor))
        .push(
            WindowMsg {
                id: m.id.get(),
                author_id: data.bot_id(),
                author: d.bot_name.clone(),
                text,
                extras: Default::default(),
                mentions: Vec::new(),
            },
            data.max_buffer(),
        );
    // Who, never what (PRIVACY.md).
    info!(admin = ctx.author().id.get(), channel = ch.get(), kind = kind.id(), "shake");
    (format!("Sent in <#{}>.", ch.get()), d.retrying_ttl_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_longer_for_longer_text() {
        assert_eq!(typing_time(0).as_millis(), 1000);
        assert_eq!(typing_time(150).as_millis(), 2500);
        assert_eq!(typing_time(DISCORD_LIMIT).as_millis(), 4000);
    }

    #[test]
    fn cut_keeps_short_text_and_marks_cuts() {
        assert_eq!(cut("abc", 3), "abc");
        assert_eq!(cut("abcd", 3), "ab…");
        assert_eq!(cut("äöüß", 2).chars().count(), 2);
    }

    #[test]
    fn bored_lines_fit_a_dropdown_and_a_message() {
        assert!(BORED_LINES_CPU <= BORED_LINES_GPU && BORED_LINES_GPU <= 25);
        let room = (DISCORD_LIMIT - 100) / BORED_LINES_GPU - 10;
        assert!(BORED_LINES_GPU * (room + 10) + 30 <= DISCORD_LIMIT);
    }

    #[test]
    fn kinds_round_trip_their_ids() {
        for &k in Kind::ALL {
            assert_eq!(Kind::from_id(k.id()), Some(k));
            assert!(k.label().len() <= 100, "a dropdown label's limit");
        }
        assert_eq!(Kind::from_id("nope"), None);
    }

    #[test]
    fn only_directed_lines_leave_the_bored_command() {
        assert_eq!(Kind::Directed.command(), CommandName::Directed);
        assert_eq!(Kind::Prompted.command(), CommandName::Bored);
        assert_eq!(Kind::Bored.command(), CommandName::Bored);
    }

    #[test]
    fn the_panel_only_takes_its_own_options() {
        let mut p = Panel {
            prefix: "42".into(),
            channels: vec![(ChannelId::new(7), "#floob".into())],
            here: ChannelId::new(7),
            channel: None,
            kind: None,
        };
        assert_eq!(p.part("42-go"), Some("go"));
        assert_eq!(p.part("420-go"), None);
        assert!(!p.pick("channel", "8"), "not offered");
        assert!(p.pick("channel", "7"));
        assert!(p.pick("kind", "bored"));
        assert_eq!((p.channel, p.kind), (Some(ChannelId::new(7)), Some(Kind::Bored)));
    }
}
