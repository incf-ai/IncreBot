//! Discord integration (design.md §6): poise/serenity glue around the pure chat
//! logic in `crate::chat` and the inference thread in `crate::engine::worker`.

mod actions;
mod batch;
mod dispatch;
mod force;
pub mod hotdog;
pub mod novelty;
mod optout;
mod remember;
mod reply;
mod seeded;
mod shake;
#[cfg(test)]
mod simtest;

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};
use std::time::Instant;

use anyhow::Result;
use poise::serenity_prelude as serenity;
use serenity::{ChannelId, GatewayIntents, Mentionable};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use crate::chat::gate::{
    MessageFacts, in_scope, is_admin, is_ignored_text, mentions_bot, should_reply,
};
use crate::chat::prompt::PromptBuilder;
use crate::persona::{Active, CommandName, Persona};
use crate::chat::queue::QueueKey;
use crate::chat::limits;
use crate::chat::retry;
use crate::chat::sanitize::{Extras, clean_name};
use crate::chat::window::{ChannelWindow, WindowMsg};
use crate::config::{Config, listed};
use crate::engine::worker::{EngineHandle, EngineStats, ResetHandle};
use crate::tokenizer::Tokenizer;
pub use dispatch::DispatchMsg;
pub use optout::OptOuts;
use dispatch::{EnqueueResult, Replay, Source, Trigger};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Context<'a> = poise::Context<'a, Data, Error>;

/// Bot-authored notices (busy, stopped, errors) start with this, so they never
/// enter a prompt as something the bot "said".
pub const NOTICE_PREFIX: &str = "*(";

/// State shared by event handlers, commands, the dispatcher and reply tasks.
pub struct Shared {
    pub cfg: Config,
    pub tok: Arc<Tokenizer>,
    /// The persona file in use, with its prompt builder and sampler tables;
    /// swapped whole by `/super-admin persona-reload`.
    persona: RwLock<Arc<Active>>,
    pub windows: tokio::sync::Mutex<HashMap<u64, ChannelWindow>>,
    /// Thread → parent channel (or `None` for a non-thread), cached per channel.
    parents: Mutex<HashMap<u64, Option<u64>>>,
    pub dispatch: mpsc::UnboundedSender<DispatchMsg>,
    /// Reply message id → (requester, cancel flag), for ❌ (design.md §6).
    pub cancels: Mutex<HashMap<u64, (u64, Arc<AtomicBool>)>>,
    /// Bot messages that are placeholders or still streaming; never history.
    pub inflight: Mutex<HashSet<u64>>,
    pub stats: Arc<Mutex<EngineStats>>,
    pub resets: ResetHandle,
    pub gpus: Option<Arc<crate::gpu::manager::Gpus>>,
    /// Prefills the persona header ahead of replies (at startup and reload).
    pub seed: crate::engine::worker::SeedHandle,
    /// What the bot was up to in each quiet channel (`chat::idle`).
    pub idle: crate::chat::idle::Idle,
    pub started: Instant,
    /// Startup as a message ID: a channel's history starts after it.
    start_anchor: u64,
    pub bot_id: OnceLock<u64>,
    /// The bot account's username and display name, rewritten to the persona
    /// name in prompts (mentions of the bot render as these).
    account_names: OnceLock<Vec<String>>,
    /// Guild → the bot's own (managed) role ids, refreshed from the cache on each
    /// message. Mentioning that role counts as mentioning the bot.
    bot_roles: Mutex<HashMap<u64, Vec<u64>>>,
    /// The last prompt sent to the engine per channel, for `/super-admin context`.
    pub last_prompts: Mutex<HashMap<u64, String>>,
    /// Users who ran `/opt-out`: their messages are dropped on arrival.
    pub optouts: OptOuts,
    pub shutting_down: AtomicBool,
    /// `discord.novelty_key`, behind the fixed novelty commands' results.
    novelty_key: seeded::Key,
    /// Per channel: the last output `/retry` can replace, and activity since.
    retry: Mutex<retry::Book<Replay>>,
    /// Per user: when their last `/retry-batch` generated anything (`batch::COOLDOWN`).
    batch_cooldowns: Mutex<HashMap<u64, Instant>>,
    /// How often each user made the bot generate by command (`chat::limits`).
    limits: Mutex<crate::chat::limits::Limiter>,
    /// Interaction → the bill an `/ask` or novelty command took when allowed.
    command_holds: Mutex<HashMap<u64, limits::Hold>>,
    /// Each channel's pause between replies to its messages.
    pacer: Mutex<limits::Pacer>,
    /// What happened to each channel's recent messages ("Admin actions" → Trace).
    pub traces: crate::chat::trace::Traces,
}

pub type Data = Arc<Shared>;

impl Shared {
    pub fn bot_id(&self) -> u64 {
        self.bot_id.get().copied().unwrap_or(0)
    }

    pub fn retry(&self) -> MutexGuard<'_, retry::Book<Replay>> {
        self.retry.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds `event` to the trace of `message` in `channel` (`chat::trace`).
    pub fn trace(&self, channel: u64, message: u64, event: impl Into<String>) {
        self.traces.add(channel, message, event);
    }

    /// The same for the message a channel reply answers, if `source` is one.
    pub fn trace_source(&self, channel: u64, source: &Source, event: impl Into<String>) {
        if let Source::Channel { trigger_msg, .. } = source {
            self.traces.add(channel, *trigger_msg, event);
        }
    }

    pub fn pacer(&self) -> MutexGuard<'_, limits::Pacer> {
        self.pacer.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn limiter(&self) -> MutexGuard<'_, crate::chat::limits::Limiter> {
        self.limits.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether `user` skips the per-user limits and the batch cooldown: an
    /// admin, while `limits.admins_exempt` is on (off by default).
    pub fn limits_exempt(&self, user: u64) -> bool {
        self.cfg.discord.limits.admins_exempt && is_admin(user, &self.cfg.discord)
    }

    /// How long `user` must still wait before making the bot generate by
    /// command (`chat::limits`); None when they needn't.
    pub fn limit_wait(&self, user: u64) -> Option<std::time::Duration> {
        if self.limits_exempt(user) {
            return None;
        }
        self.limiter().wait(&self.cfg.discord.limits, user, Instant::now())
    }

    /// Bills `user` for making the bot generate by command, starting now, at
    /// `cost` (1, or a batch's), unless they must wait: then the refusal.
    /// None for those [`limits_exempt`](Self::limits_exempt), never billed.
    /// Release it if it doesn't succeed.
    pub fn limit_try_hold(&self, user: u64, cost: f64) -> Result<Option<limits::Hold>, String> {
        if self.limits_exempt(user) {
            return Ok(None);
        }
        let held = self.limiter().try_hold(&self.cfg.discord.limits, user, Instant::now(), cost);
        held.map(Some).map_err(limits::message)
    }

    /// The same, billing whether they must wait or not.
    pub fn limit_hold(&self, user: u64, cost: f64) -> Option<limits::Hold> {
        if self.limits_exempt(user) {
            return None;
        }
        Some(self.limiter().hold(&self.cfg.discord.limits, user, Instant::now(), cost))
    }

    /// Takes back a bill: what it was for didn't succeed.
    pub fn limit_release(&self, hold: Option<limits::Hold>) {
        if let Some(h) = hold {
            self.limiter().release(&self.cfg.discord.limits, h, Instant::now());
        }
    }

    /// The bill an `/ask` or novelty command took when it was allowed, for
    /// its reply to keep or release (`queue_reply`).
    fn take_hold(&self, interaction: u64) -> Option<limits::Hold> {
        self.command_holds.lock().unwrap_or_else(|e| e.into_inner()).remove(&interaction)
    }

    fn keep_hold(&self, interaction: u64, hold: Option<limits::Hold>) {
        if let Some(h) = hold {
            self.command_holds.lock().unwrap_or_else(|e| e.into_inner()).insert(interaction, h);
        }
    }

    pub fn persona(&self) -> Arc<Active> {
        self.persona
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn prompt(&self) -> Arc<PromptBuilder> {
        self.persona().prompt.clone()
    }

    pub fn count_tokens(&self, text: &str) -> usize {
        self.tok
            .encode_prompt(text)
            .map_or(usize::MAX, |ids| ids.len())
    }

    /// Re-reads the persona file and swaps in everything it sets: prompt,
    /// notes, word biases, stop strings, sampler and reply length. On error the
    /// old persona stays (design.md §7). Returns the header size in tokens.
    pub fn reload_persona(&self) -> Result<usize> {
        let path = Path::new(&self.cfg.discord.persona_file);
        let persona = Persona::load(path, &|k| std::env::var(k).ok())?;
        persona.check_against(&self.cfg)?;
        let active = self.activate(persona)?;
        let header = active
            .prompt
            .check_header(self.cfg.generation.header_limit(), &|s| self.count_tokens(s))?;
        if active.persona == self.persona().persona {
            info!(header_tokens = header, persona = %active.persona.line(), "persona reloaded: unchanged");
            // Comments may have changed: keep the new text for persona-dump. The
            // prompts are the same, so the caches still match.
            if active.persona.source.0 != self.persona().persona.source.0 {
                *self.persona.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(active);
            }
            return Ok(header);
        }
        info!(
            header_tokens = header,
            persona = %active.persona.line(),
            "persona reloaded; every channel's next turn is a cold prefill"
        );
        *self.persona.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(active);
        self.request_seed();
        Ok(header)
    }

    /// Asks the engine to prefill the current persona's header (the prompt
    /// with no messages; every reply's prompt starts the same way) into the
    /// cache new channels and one-shot replies start from.
    pub fn request_seed(&self) {
        let header = self.prompt().render(&[]).and_then(|t| self.tok.encode_prompt(&t));
        match header {
            Ok(ids) => self.seed.request(ids),
            Err(e) => tracing::warn!(error = %e, "can't render the persona header to prefill"),
        }
    }

    fn activate(&self, persona: Persona) -> Result<Active> {
        let accounts = self.account_names.get().cloned().unwrap_or_default();
        let d = &self.cfg.discord;
        Active::new(persona, &d.prompt_template, &d.bot_name, &accounts, &self.tok)
    }

    /// Once connected: the account's own names become aliases of the persona.
    fn set_account_names(&self, names: Vec<String>) {
        if self.account_names.set(names).is_err() {
            return;
        }
        match self.activate(self.persona().persona.clone()) {
            Ok(a) => *self.persona.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(a),
            Err(e) => warn!(error = %e, "could not add the account's names as aliases"),
        }
    }

    /// The bot's own role ids in `guild`, as last seen (empty for DMs).
    pub fn bot_roles(&self, guild: Option<u64>) -> Vec<u64> {
        guild
            .and_then(|g| {
                self.bot_roles
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&g)
                    .cloned()
            })
            .unwrap_or_default()
    }

    /// The bot's role ids in every guild seen so far. Role ids are globally
    /// unique, so this serves when the guild isn't known (REST history).
    pub fn known_bot_roles(&self) -> Vec<u64> {
        self.bot_roles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .flatten()
            .copied()
            .collect()
    }

    /// Re-reads the bot's managed role ids in `guild` from the cache.
    fn refresh_bot_roles(&self, ctx: &serenity::Context, guild: serenity::GuildId) -> Vec<u64> {
        let bot = self.bot_id();
        let roles: Vec<u64> = match ctx.cache.guild(guild) {
            Some(g) => g
                .roles
                .values()
                .filter(|r| r.tags.bot_id.is_some_and(|b| b.get() == bot))
                .map(|r| r.id.get())
                .collect(),
            None => return self.bot_roles(Some(guild.get())),
        };
        self.bot_roles
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(guild.get(), roles.clone());
        roles
    }

    fn max_buffer(&self) -> usize {
        crate::chat::window::max_buffer(self.cfg.discord.history_messages)
    }
}

pub fn intents() -> GatewayIntents {
    GatewayIntents::GUILDS
        | GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::GUILD_MESSAGE_REACTIONS
        | GatewayIntents::DIRECT_MESSAGE_REACTIONS
        | GatewayIntents::MESSAGE_CONTENT
}

/// Builds the shared state and the poise framework, and starts the dispatcher.
/// The caller connects the client; the model is already loaded (design.md §6:
/// the bot shows offline until it can answer).
pub fn framework(
    cfg: Config,
    tok: Arc<Tokenizer>,
    active: Active,
    engine: EngineHandle,
    optouts: OptOuts,
) -> (poise::Framework<Data, Error>, Data) {
    let (dispatch_tx, dispatch_rx) = mpsc::unbounded_channel();
    let novelty_key = seeded::Key::new(cfg.discord.novelty_key);
    let retry_rules = cfg.discord.retry.clone();
    let shared = Arc::new(Shared {
        novelty_key,
        cfg,
        tok,
        persona: RwLock::new(Arc::new(active)),
        windows: Default::default(),
        parents: Default::default(),
        dispatch: dispatch_tx,
        cancels: Default::default(),
        inflight: Default::default(),
        stats: engine.stats.clone(),
        resets: engine.resets.clone(),
        gpus: engine.gpus.clone(),
        seed: engine.seed.clone(),
        idle: Default::default(),
        started: Instant::now(),
        start_anchor: crate::chat::window::snowflake_at(std::time::SystemTime::now()),
        bot_id: OnceLock::new(),
        account_names: OnceLock::new(),
        bot_roles: Default::default(),
        last_prompts: Default::default(),
        optouts,
        shutting_down: AtomicBool::new(false),
        retry: Mutex::new(retry::Book::new(retry_rules)),
        batch_cooldowns: Default::default(),
        limits: Default::default(),
        command_holds: Default::default(),
        pacer: Default::default(),
        traces: Default::default(),
    });

    // The persona header, prefilled as soon as the GPUs have settled.
    shared.request_seed();
    let data = shared.clone();
    let engine = Mutex::new(Some((engine, dispatch_rx)));
    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![
                ask(),
                hotdog_rater(),
                kobold_name(),
                lick_test(),
                roast(),
                fortune(),
                hoard_value(),
                kobold_age(),
                judge(),
                trap_check(),
                drink(),
                gamble(),
                dragon_alert(),
                vibe_check(),
                admin(),
                stats(),
                version(),
                optout(),
                optin(),
                retry(),
                batch::retry_batch(),
                force::admin_actions(),
                actions::reply_actions(),
                undo_retry(),
                delete_reply(),
            ],
            // Any command but `/retry`, `/retry-batch`, `/undo-retry`, `/delete-reply`,
            // "Admin actions" and "Reply actions" counts
            // as activity: a reply it follows can no longer be deleted
            // (`chat::retry`; retries don't count commands).
            pre_command: |ctx| {
                Box::pin(async move {
                    if !matches!(&*ctx.command().name, "retry" | "retry-batch" | "undo-retry" | "delete-reply" | "admin_actions" | "reply_actions") {
                        ctx.data().retry().note_activity(ctx.channel_id().get(), ctx.id());
                    }
                })
            },
            // All commands are slash commands. With mention-as-prefix on, poise warns
            // about every "@bot ..." message and logs its text, which PRIVACY.md says
            // the logs never contain.
            prefix_options: poise::PrefixFrameworkOptions {
                mention_as_prefix: false,
                ..Default::default()
            },
            event_handler: |framework, event| Box::pin(on_event(framework, event)),
            on_error: |error| {
                Box::pin(async move {
                    if let Err(e) = poise::builtins::on_error(error).await {
                        error!(error = %e, "error while handling a command error");
                    }
                })
            },
            ..Default::default()
        })
        .setup(move |ctx, ready, framework| {
            let data = data.clone();
            let engine = engine.lock().unwrap_or_else(|e| e.into_inner()).take();
            Box::pin(async move {
                let _ = data.bot_id.set(ready.user.id.get());
                data.set_account_names(vec![
                    ready.user.name.clone(),
                    ready.user.display_name().to_string(),
                ]);
                poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                startup_checks(ctx, &data).await;
                if let Some((engine, rx)) = engine {
                    tokio::spawn(dispatch::run(data.clone(), engine, rx));
                    tokio::spawn(dispatch::watch_idle(data.clone()));
                    tokio::spawn(dispatch::warm_at_boot(data.clone(), ctx.http.clone()));
                }
                info!(user = %ready.user.name, "connected to Discord");
                Ok(data)
            })
        })
        .build();
    (framework, shared)
}

/// Startup checks once connected: every dedicated or shared channel must be in an
/// allowlisted guild (fatal otherwise), and the bot needs Read Message History
/// (warned).
async fn startup_checks(ctx: &serenity::Context, data: &Shared) {
    let cfg = &data.cfg.discord;
    if cfg.guilds.is_empty() {
        warn!("no guilds allowlisted: the bot will only answer allowlisted DMs");
    }
    if cfg.novelty_key == 0 {
        warn!("discord.novelty_key is 0: anyone with the source can compute the fixed novelty results");
    }
    if cfg.dedicated_channels.is_empty() && cfg.shared_channels.is_empty() {
        warn!("no dedicated or shared channels: the bot will only answer allowlisted DMs");
    }
    for ch in cfg.dedicated_channels.iter().chain(&cfg.shared_channels) {
        let channel = match ChannelId::new(ch.id).to_channel(ctx).await {
            Ok(serenity::Channel::Guild(c)) => c,
            Ok(_) => {
                error!(
                    channel = %ch,
                    "configured chat channel is not a guild channel"
                );
                std::process::exit(2);
            }
            Err(e) => {
                warn!(channel = %ch, error = %e, "cannot see configured chat channel");
                continue;
            }
        };
        if !listed(&cfg.guilds, channel.guild_id.get()) {
            error!(
                channel = %ch,
                guild = %cfg.label(channel.guild_id.get()),
                "chat channel is not in an allowlisted guild"
            );
            std::process::exit(2);
        }
        let ok = ctx
            .cache
            .guild(channel.guild_id)
            .and_then(|g| {
                let me = g.members.get(&serenity::UserId::new(data.bot_id()))?;
                Some(g.user_permissions_in(&channel, me).read_message_history())
            })
            .unwrap_or(true); // unknown (not cached yet): don't cry wolf
        if !ok {
            warn!(channel = %ch, "missing Read Message History permission");
        }
    }
}

async fn on_event(
    framework: poise::FrameworkContext<'_, Data, Error>,
    event: &serenity::FullEvent,
) -> Result<(), Error> {
    let data = framework.user_data;
    let ctx = framework.serenity_context;
    match event {
        serenity::FullEvent::Message { new_message } => on_message(ctx, data, new_message).await,
        serenity::FullEvent::MessageUpdate { event, .. } => {
            let Some(text) = &event.content else {
                return Ok(());
            };
            // What a retry counts follows the edit, as the window does below
            // (a channel without a window too): edited into a `//` message,
            // it stops counting; back out of one, it counts again.
            let author = event.author.as_ref().map(|a| a.id.get());
            let hidden = author.is_some_and(|a| data.optouts.hidden(a, event.id.get()));
            data.retry().gateway_edit(
                event.channel_id.get(),
                event.id.get(),
                author,
                data.bot_id(),
                is_ignored_text(text),
                hidden,
            );
            let mut windows = data.windows.lock().await;
            if let Some(w) = windows.get_mut(&event.channel_id.get()) {
                // The bot's own turns change only through the reply code, with
                // its final text (a command's answer has the card above it).
                if w.author_of(event.id.get()) == Some(data.bot_id()) {
                    return Ok(());
                }
                let mentions = event.mentions.clone().unwrap_or_default();
                let extras = Extras {
                    attachments: event.attachments.iter().flatten().map(attachment).collect(),
                    embeds: event
                        .embeds
                        .iter()
                        .flatten()
                        .filter_map(|e| e.title.clone())
                        .collect(),
                    ..Default::default()
                };
                // Edited into a `//` message: forget it.
                if is_ignored_text(text) {
                    w.remove(event.id.get());
                    return Ok(());
                }
                let bot_roles = data.bot_roles(event.guild_id.map(|g| g.get()));
                let names = mention_names(text, &mentions);
                let text =
                    resolve_mentions(text, &mentions, &bot_roles, &data.cfg.discord.bot_name);
                // Edited back out of a `//` message: it becomes visible
                // again, as a restart's seed would make it. The author must
                // not be the bot (guarded above), opted out, or unknown
                // (some update events carry no author); another bot is
                // fine, as its messages enter the window like anyone's.
                if !w.contains(event.id.get()) {
                    // `hidden`, not just `is_out`: editing a message from
                    // before an opt-in doesn't make it visible — it is still
                    // from the hidden period, and a cold start skips it.
                    if let Some(author) = &event.author
                        && author.id.get() != data.bot_id()
                        && !data.optouts.hidden(author.id.get(), event.id.get())
                    {
                        let name = event
                            .member
                            .iter()
                            .flatten()
                            .find_map(|m| m.nick.clone())
                            .unwrap_or_else(|| author.display_name().to_string());
                        w.push(
                            WindowMsg {
                                id: event.id.get(),
                                author_id: author.id.get(),
                                author: clean_name(&name),
                                text,
                                extras,
                                mentions: names,
                            },
                            data.max_buffer(),
                        );
                    }
                    return Ok(());
                }
                w.update(event.id.get(), text, extras, names);
            }
        }
        serenity::FullEvent::MessageDelete {
            channel_id,
            deleted_message_id,
            ..
        } => {
            data.retry().unnote_message(channel_id.get(), deleted_message_id.get());
            if let Some(w) = data.windows.lock().await.get_mut(&channel_id.get()) {
                w.remove(deleted_message_id.get());
            }
        }
        serenity::FullEvent::ReactionAdd { add_reaction } => on_reaction(data, add_reaction).await,
        _ => {}
    }
    Ok(())
}

async fn on_message(ctx: &serenity::Context, data: &Data, msg: &serenity::Message) {
    let bot_id = data.bot_id();
    // `/retry` and `/delete-reply` bookkeeping: the bot's own post, or
    // anyone's message (`//` and opted-out ones block a delete too).
    data.retry().gateway_message(
        msg.channel_id.get(),
        msg.id.get(),
        msg.author.id.get(),
        bot_id,
        is_ignored_text(&msg.content),
        data.optouts.is_out(msg.author.id.get()),
    );
    // The bot's own messages enter history only via the reply code, with final text.
    // `//` messages and opted-out users are invisible to the bot (no reply, no history).
    if msg.author.id.get() == bot_id
        || is_ignored_text(&msg.content)
        || data.optouts.is_out(msg.author.id.get())
    {
        return;
    }
    let cfg = &data.cfg.discord;
    let guild_id = msg.guild_id.map(|g| g.get());
    if guild_id.is_some_and(|g| !listed(&cfg.guilds, g)) {
        return;
    }
    let parent = match msg.guild_id {
        Some(_) => thread_parent(ctx, data, msg.channel_id).await,
        None => None,
    };
    let mut facts = MessageFacts {
        author_id: msg.author.id.get(),
        author_is_bot: msg.author.bot,
        guild_id,
        channel_id: msg.channel_id.get(),
        parent_channel_id: parent,
        ..Default::default()
    };
    // Only dedicated and shared channels (and allowlisted DMs) are kept as history.
    if !in_scope(&facts, cfg) {
        return;
    }
    trace!(channel = msg.channel_id.get(), content = %msg.content, "message");
    let (ch, mid) = (msg.channel_id.get(), msg.id.get());
    data.traces.start(ch, mid, msg.author.id.get(), msg.author.display_name(), &msg.content);
    let bot_roles = match msg.guild_id {
        Some(g) => data.refresh_bot_roles(ctx, g),
        None => Vec::new(),
    };
    data.windows
        .lock()
        .await
        .entry(msg.channel_id.get())
        .or_insert_with(|| ChannelWindow::after(data.start_anchor))
        .push(
            window_msg(msg, &bot_roles, &cfg.bot_name),
            data.max_buffer(),
        );

    facts.mentions_bot = mentions_bot(
        &msg.mentions.iter().map(|u| u.id.get()).collect::<Vec<_>>(),
        &msg.mention_roles
            .iter()
            .map(|r| r.get())
            .collect::<Vec<_>>(),
        bot_id,
        &bot_roles,
    );
    facts.replies_to_bot = msg
        .referenced_message
        .as_ref()
        .is_some_and(|r| r.author.id.get() == bot_id);
    let kind = crate::chat::gate::channel_kind(facts.channel_id, facts.parent_channel_id, cfg);
    let why = match kind {
        _ if facts.author_is_bot => "not replying: the author is a bot",
        _ if facts.guild_id.is_none() => "replying: an allowed DM",
        Some(crate::chat::gate::ChannelKind::Dedicated) => "replying: dedicated channel (every message)",
        Some(crate::chat::gate::ChannelKind::Shared) if facts.mentions_bot => "replying: shared channel, mentions floob",
        Some(crate::chat::gate::ChannelKind::Shared) if facts.replies_to_bot => "replying: shared channel, a reply to floob",
        _ => "not replying: shared channel, floob not mentioned or replied to",
    };
    data.trace(ch, mid, why);
    if !should_reply(&facts, cfg) || data.shutting_down.load(Ordering::Relaxed) {
        if data.shutting_down.load(Ordering::Relaxed) {
            data.trace(ch, mid, "not replying: restarting");
        }
        return;
    }

    // Freeze the history now (trigger included): the reply answers the
    // conversation as it was when it was asked (design.md §6). A channel's first
    // trigger since startup or /super-admin reset loads its recent history first.
    let snapshot = {
        let mut windows = data.windows.lock().await;
        let window = windows
            .entry(msg.channel_id.get())
            .or_insert_with(|| ChannelWindow::after(data.start_anchor));
        if !window.loaded {
            dispatch::seed_window(data, &ctx.http, msg.channel_id.get(), window).await;
            data.trace(ch, mid, format!("history: first reply here since startup or reset, {} messages loaded", window.messages().count()));
        }
        window.clone()
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let source = Source::Channel {
        trigger_msg: msg.id.get(),
        snapshot,
    };
    let replay = Replay {
        source: source.clone(),
        header: None,
        avoid: Vec::new(),
    };
    let trigger = Trigger {
        key: QueueKey::Channel(msg.channel_id.get()),
        channel: msg.channel_id.get(),
        source,
        requester: msg.author.id.get(),
        cancel: cancel.clone(),
        events: events_tx,
        avoid: Vec::new(),
        paced: true,
    };
    if data
        .dispatch
        .send(DispatchMsg::Enqueue(Box::new(trigger), result_tx))
        .is_err()
    {
        return;
    }
    let Ok(result) = result_rx.await else { return };
    let http = ctx.http.clone();
    match result {
        // Queue full: newer messages are ignored (design.md §6). They stay in the
        // history, so a later reply still sees them.
        EnqueueResult::Full => {
            debug!(
                channel = msg.channel_id.get(),
                "queue full: message not answered"
            );
            data.trace(ch, mid, format!(
                "turned away: this channel's queue is full ({} waiting), no reply",
                data.cfg.runtime.queue_depth
            ));
        }
        EnqueueResult::Queued { ahead } => {
            // Every message queued for a reply makes the channel busier,
            // admins' too; one turned away as full doesn't.
            let l = &data.cfg.discord.channel_limits;
            let pressure = data.pacer().arrived(l, msg.channel_id.get(), Instant::now());
            data.trace(ch, mid, format!(
                "queued: {ahead} ahead; channel pressure {pressure:.1} (pause after a reply now {:.1} s)",
                limits::Pacer::delay(l, pressure).as_secs_f64()
            ));
            let sink = reply::Sink::Channel {
                http,
                channel: msg.channel_id,
                reply_to: Some(msg.id),
            };
            tokio::spawn(reply::stream(
                sink,
                data.clone(),
                events_rx,
                cancel,
                msg.author.id.get(),
                ahead,
                replay,
            ));
        }
    }
}

async fn on_reaction(data: &Shared, r: &serenity::Reaction) {
    let serenity::ReactionType::Unicode(emoji) = &r.emoji else {
        return;
    };
    if emoji != "❌" {
        return;
    }
    let Some(user) = r.user_id else { return };
    let entry = data
        .cancels
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&r.message_id.get())
        .cloned();
    let Some((requester, flag)) = entry else {
        return;
    };
    if user.get() == requester || is_admin(user.get(), &data.cfg.discord) {
        info!(
            message = r.message_id.get(),
            by = user.get(),
            "cancel requested"
        );
        flag.store(true, Ordering::Relaxed);
        let _ = data.dispatch.send(DispatchMsg::Cancel(flag));
    }
}

/// The parent channel of a thread, or `None`. Cached per channel.
async fn thread_parent(ctx: &serenity::Context, data: &Shared, channel: ChannelId) -> Option<u64> {
    if let Some(p) = data
        .parents
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&channel.get())
    {
        return *p;
    }
    let parent = match channel.to_channel(ctx).await {
        Ok(serenity::Channel::Guild(c)) if c.thread_metadata.is_some() => {
            c.parent_id.map(|p| p.get())
        }
        _ => None,
    };
    data.parents
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(channel.get(), parent);
    parent
}

fn attachment(a: &serenity::Attachment) -> (String, bool) {
    (
        a.filename.clone(),
        a.content_type
            .as_deref()
            .is_some_and(|t| t.starts_with("image/")),
    )
}

/// `<@id>` / `<@!id>` → `@name` for mentioned users; role and channel mentions →
/// `@role` / `#channel`.
/// Renders Discord mention markup as text: users by display name, the bot's own
/// role as `@<bot_name>` (renamed to the persona later), other roles, channels
/// and unknown users generically.
fn resolve_mentions(
    text: &str,
    mentions: &[serenity::User],
    bot_roles: &[u64],
    bot_name: &str,
) -> String {
    let mut out = text.to_string();
    for r in bot_roles {
        out = out.replace(&format!("<@&{r}>"), &format!("@{bot_name}"));
    }
    for u in mentions {
        let name = format!("@{}", mention_name(u));
        out = out
            .replace(&format!("<@{}>", u.id), &name)
            .replace(&format!("<@!{}>", u.id), &name);
    }
    let mut result = String::with_capacity(out.len());
    let mut rest = out.as_str();
    while let Some(i) = rest.find('<') {
        result.push_str(&rest[..i]);
        let tail = &rest[i..];
        let end = tail.find('>').unwrap_or(0);
        let inner = if end > 0 { &tail[1..end] } else { "" };
        let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        if let Some(id) = inner.strip_prefix("@&").filter(|s| digits(s)) {
            let _ = id;
            result.push_str("@role");
            rest = &tail[end + 1..];
        } else if let Some(id) = inner.strip_prefix('#').filter(|s| digits(s)) {
            let _ = id;
            result.push_str("#channel");
            rest = &tail[end + 1..];
        } else if let Some(id) = inner
            .strip_prefix('@')
            .map(|s| s.trim_start_matches('!'))
            .filter(|s| digits(s))
        {
            let _ = id;
            result.push_str("@someone");
            rest = &tail[end + 1..];
        } else {
            result.push('<');
            rest = &tail[1..];
        }
    }
    result.push_str(rest);
    result
}

/// How a mention of `u` reads: the server nickname a message's mention
/// carries (or one `interaction_users` attached), as the author labels use,
/// else the display name; cleaned.
fn mention_name(u: &serenity::User) -> String {
    clean_name(u.member.as_ref().and_then(|m| m.nick.as_deref()).unwrap_or_else(|| u.display_name()))
}

/// The names [`resolve_mentions`] gives the users of `mentions` that `text`
/// (raw, `<@id>`) mentions: a [`WindowMsg`]'s `mentions`.
fn mention_names(text: &str, mentions: &[serenity::User]) -> Vec<String> {
    let ids = remember::mentioned(text);
    mentions.iter().filter(|u| ids.contains(&u.id.get())).map(mention_name).collect()
}

pub(crate) fn window_msg(msg: &serenity::Message, bot_roles: &[u64], bot_name: &str) -> WindowMsg {
    let name = msg
        .member
        .as_ref()
        .and_then(|m| m.nick.clone())
        .unwrap_or_else(|| msg.author.display_name().to_string());
    WindowMsg {
        id: msg.id.get(),
        author_id: msg.author.id.get(),
        author: clean_name(&name),
        text: resolve_mentions(&msg.content, &msg.mentions, bot_roles, bot_name),
        mentions: mention_names(&msg.content, &msg.mentions),
        extras: Extras {
            attachments: msg.attachments.iter().map(attachment).collect(),
            stickers: msg.sticker_items.iter().map(|s| s.name.clone()).collect(),
            embeds: msg.embeds.iter().filter_map(|e| e.title.clone()).collect(),
            poll: msg.poll.as_ref().and_then(|p| p.question.text.clone()),
        },
    }
}

async fn require_admin(ctx: Context<'_>) -> Result<bool, Error> {
    if is_admin(ctx.author().id.get(), &ctx.data().cfg.discord) {
        return Ok(true);
    }
    notice(ctx, "Admins only.").await?;
    Ok(false)
}

/// Sends an ephemeral refusal or "busy" notice that deletes itself after
/// `discord.notice_ttl_ms`.
async fn notice(ctx: Context<'_>, text: impl Into<String>) -> Result<(), Error> {
    notice_for(ctx, text, ctx.data().cfg.discord.notice_ttl_ms).await
}

/// Sends an ephemeral `text` and deletes it after `ttl_ms` (0 = never). Waits
/// in the command's own task, so callers have nothing left to do after it.
async fn notice_for(ctx: Context<'_>, text: impl Into<String>, ttl_ms: u64) -> Result<(), Error> {
    let handle = ctx
        .send(poise::CreateReply::default().content(text).ephemeral(true))
        .await?;
    if ttl_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(ttl_ms)).await;
        // Fails if the user already dismissed it.
        if let Err(e) = handle.delete(ctx).await {
            debug!("deleting notice: {e}");
        }
    }
    Ok(())
}

/// Ask a one-off question (no channel history).
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn ask(
    ctx: Context<'_>,
    #[description = "Your question"] prompt: String,
) -> Result<(), Error> {
    let data = ctx.data().clone();
    if data.shutting_down.load(Ordering::Relaxed) {
        notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await?;
        return Ok(());
    }
    match data.limit_try_hold(ctx.author().id.get(), 1.0) {
        Ok(hold) => data.keep_hold(ctx.id(), hold),
        Err(why) => return notice(ctx, why).await,
    }
    // Not deferred yet: when the queue is full, "busy" is then the first
    // response, which can be ephemeral (a public defer would make it public).
    let author = speaker_name(ctx).await;
    // The question shows, quoted, above the answer (`remember::ask_header`).
    let header = remember::ask_header(&ctx.author().mention().to_string(), &prompt);
    queue_reply(ctx, author, prompt, Some(header), None, None).await
}

/// The requester and the users the command's options name, for resolving the
/// mentions in its header — by their server nicknames where they have one, as
/// their messages are labeled.
fn interaction_users(ctx: Context<'_>) -> Vec<serenity::User> {
    let mut users = vec![ctx.author().clone()];
    if let poise::Context::Application(a) = ctx {
        if let Some(nick) = a.interaction.member.as_ref().and_then(|m| m.nick.clone()) {
            users[0].global_name = Some(nick);
        }
        let resolved = &a.interaction.data.resolved;
        users.extend(resolved.users.values().map(|u| {
            let mut u = u.clone();
            if let Some(nick) = resolved.members.get(&u.id).and_then(|m| m.nick.clone()) {
                u.global_name = Some(nick);
            }
            u
        }));
    }
    users
}

/// The author's server nickname, else their display name, cleaned for a prompt
/// (and lowercased if the persona says so).
async fn speaker_name(ctx: Context<'_>) -> String {
    let name = ctx
        .author_member()
        .await
        .and_then(|m| m.nick.clone())
        .unwrap_or_else(|| ctx.author().display_name().to_string());
    ctx.data().persona().prompt.user_name(&clean_name(&name))
}

/// Queues a one-shot reply to `prompt`, said by `author`, and streams it as the
/// answer to the interaction (deferring it if it isn't yet), below `header` if
/// given.
/// `command` names the novelty command, for its persona setup.
async fn queue_reply(
    ctx: Context<'_>,
    author: String,
    prompt: String,
    header: Option<String>,
    command: Option<CommandName>,
    draw: Option<crate::persona::CommandDraw>,
) -> Result<(), Error> {
    let data = ctx.data().clone();
    // Billed when it was allowed; kept if the reply is generated.
    let hold = data.take_hold(ctx.id());
    let cancel = Arc::new(AtomicBool::new(false));
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    // The requester's turn, should the answer go into the history.
    let asked = header.as_deref().and_then(|h| {
        let users = interaction_users(ctx);
        let roles = data.bot_roles(ctx.guild_id().map(|g| g.get()));
        let resolve = |t: &str| resolve_mentions(t, &users, &roles, &data.cfg.discord.bot_name);
        let mut asked = remember::asked(h, ctx.id(), (ctx.author().id.get(), &author), &resolve)?;
        asked.turn.mentions = mention_names(h, &users);
        Some(Box::new(asked))
    });
    let is_ask = asked.as_ref().is_some_and(|a| a.slash == "ask");
    let source = Source::Ask { author, prompt, command, draw, asked };
    let replay = Replay {
        source: source.clone(),
        header: header.clone(),
        avoid: Vec::new(),
    };
    let trigger = Trigger {
        key: QueueKey::Ask(ctx.id()),
        channel: ctx.channel_id().get(),
        source,
        requester: ctx.author().id.get(),
        cancel: cancel.clone(),
        events: events_tx,
        avoid: Vec::new(),
        paced: false,
    };
    let queued = match data.dispatch.send(DispatchMsg::Enqueue(Box::new(trigger), result_tx)) {
        Ok(()) => result_rx.await.ok(),
        Err(_) => None,
    };
    let Some(queued) = queued else {
        data.limit_release(hold);
        return Err("dispatcher stopped".into());
    };
    match queued {
        // An interaction must be answered; only the asker sees this.
        EnqueueResult::Full => {
            data.limit_release(hold);
            // The card still shows; only the comment is skipped. An
            // unanswered `/ask` question doesn't.
            if let Some(h) = header.as_ref().filter(|_| !is_ask) {
                let handle = ctx
                    .send(
                        poise::CreateReply::default()
                            .content(h.as_str())
                            .allowed_mentions(serenity::CreateAllowedMentions::new()),
                    )
                    .await?;
                let posted = handle.message().await.ok().map(|m| m.id.get());
                // A newer post than the channel's last output: that can't be
                // retried now.
                if let Some(id) = posted {
                    data.retry().note_bot_post(ctx.channel_id().get(), id);
                }
                if let Some(a) = replay.source.remembered(&data.persona().persona)
                    && let Some(id) = posted
                {
                    reply::remember(&data, ctx.channel_id().get(), a, id, Vec::new()).await;
                }
            }
            let busy = busy();
            notice(ctx, busy).await?;
        }
        EnqueueResult::Queued { ahead } => {
            // Interactions must be acknowledged within 3 s (design.md §6); a
            // no-op for the novelty commands, which deferred already.
            ctx.defer().await?;
            let sink = reply::Sink::Ask {
                ctx,
                mention: ctx.author().mention().to_string(),
                header,
                header_msg: None,
                fallen: Vec::new(),
            };
            let generated = reply::stream(sink, data.clone(), events_rx, cancel, ctx.author().id.get(), ahead, replay).await;
            if !generated {
                data.limit_release(hold);
            }
        }
    }
    Ok(())
}

/// Refuses, ephemerally, while shutting down or when any target has opted out;
/// true to go on. Runs before `defer`, so the refusal can be ephemeral.
async fn novelty_allowed(ctx: Context<'_>, targets: &[&serenity::User]) -> Result<bool, Error> {
    let data = ctx.data();
    let refusal = if data.shutting_down.load(Ordering::Relaxed) {
        format!("{NOTICE_PREFIX}restarting, ask again shortly)*")
    } else if targets.iter().any(|u| data.optouts.is_out(u.id.get())) {
        "That user has opted out.".to_string()
    } else {
        match data.limit_try_hold(ctx.author().id.get(), 1.0) {
            Ok(hold) => {
                data.keep_hold(ctx.id(), hold);
                return Ok(true);
            }
            Err(why) => why,
        }
    };
    notice(ctx, refusal).await?;
    Ok(false)
}

/// How the requester names `user` to the bot: themselves, the bot, or by
/// nickname (else display name).
async fn who(ctx: Context<'_>, user: &serenity::User) -> novelty::Who {
    if user.id == ctx.author().id {
        return novelty::Who::Me;
    }
    if user.id.get() == ctx.data().bot_id() {
        return novelty::Who::Bot;
    }
    let nick = match ctx.guild_id() {
        Some(g) => g.member(ctx, user.id).await.ok().and_then(|m| m.nick),
        None => None,
    };
    let name = clean_name(&nick.unwrap_or_else(|| user.display_name().to_string()));
    novelty::Who::Other(ctx.data().persona().prompt.user_name(&name))
}

/// Posts `card` at once (mentions don't ping) and streams the bot's comment on
/// it into the same message, below the card. The interaction must already be
/// deferred.
async fn card_then_reply(ctx: Context<'_>, card: String, prompt: String) -> Result<(), Error> {
    card_then_reply_for(ctx, card, prompt, None, None).await
}

/// [`card_then_reply`] for a command the persona file can set up (`commands`).
async fn card_then_reply_for(
    ctx: Context<'_>,
    card: String,
    prompt: String,
    command: Option<CommandName>,
    draw: Option<crate::persona::CommandDraw>,
) -> Result<(), Error> {
    let author = speaker_name(ctx).await;
    queue_reply(ctx, author, prompt, Some(card), command, draw).await
}

/// The bot's name in novelty cards and set-up prompts: the persona's
/// ("floob"), not the Discord account's ("FloobBot").
fn persona_name(ctx: Context<'_>) -> String {
    let d = ctx.data();
    d.persona().persona.prompt_name(&d.cfg.discord.bot_name).to_string()
}

/// Measure someone's hotdog. Each user's size comes from their ID and never changes.
#[poise::command(
    slash_command,
    rename = "hotdog-rater",
    nsfw_only,
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn hotdog_rater(
    ctx: Context<'_>,
    #[description = "Whose hotdog to measure"] user: serenity::User,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[&user]).await? {
        return Ok(());
    }
    let key = ctx.data().novelty_key.clone();
    ctx.defer().await?;
    let dog = hotdog::measure(&key, user.id.get());
    // The set-up line, moves and examples come from the persona's pools; the
    // ownership line from who it is, for every persona.
    let mut draw = ctx.data().persona().persona.draw(CommandName::HotdogRater, rand::random()).unwrap_or_default();
    let template = Some(draw.template.clone())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| hotdog::PROMPT.to_string());
    let whose = who(ctx, &user).await;
    let bot = persona_name(ctx);
    draw.facts.push(whose.ownership("hotdog", &speaker_name(ctx).await, &bot));
    let draw = Some(draw);
    let prompt = dog.prompt(&template, &bot, &whose);
    let card = dog.card(&user.mention().to_string());
    card_then_reply_for(ctx, card, prompt, Some(CommandName::HotdogRater), draw).await
}

/// Someone's kobold name. It comes from their ID and never changes.
#[poise::command(
    slash_command,
    rename = "kobold-name",
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn kobold_name(
    ctx: Context<'_>,
    #[description = "Whose kobold name"] user: serenity::User,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[&user]).await? {
        return Ok(());
    }
    let key = ctx.data().novelty_key.clone();
    ctx.defer().await?;
    let name = novelty::kobold_name(&key, user.id.get());
    let prompt = novelty::name_prompt(&persona_name(ctx), &who(ctx, &user).await, &name);
    card_then_reply(ctx, novelty::name_card(&user.mention().to_string(), &name), prompt).await
}

/// The bot judges the flavors of one or two people. The result for each combination never changes.
#[poise::command(
    slash_command,
    rename = "lick-test",
    nsfw_only,
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn lick_test(
    ctx: Context<'_>,
    #[description = "Who to lick (blank: you)"] user: Option<serenity::User>,
    #[description = "Who to match them with (blank: you)"] other: Option<serenity::User>,
) -> Result<(), Error> {
    // Either may be left blank, and a blank one is the caller.
    let user = user.unwrap_or_else(|| ctx.author().clone());
    let other = other.unwrap_or_else(|| ctx.author().clone());
    if !novelty_allowed(ctx, &[&user, &other]).await? {
        return Ok(());
    }
    let key = ctx.data().novelty_key.clone();
    ctx.defer().await?;
    let bot = persona_name(ctx);
    // The same person twice (both blank, say): their inner flavor match instead.
    if user.id == other.id {
        let id = user.id.get();
        let card = novelty::self_lick_card(&key, &bot, ctx.data().bot_id(), (&user.mention().to_string(), id));
        let prompt = novelty::self_lick_prompt(&key, &bot, (&who(ctx, &user).await, id));
        return card_then_reply(ctx, card, prompt).await;
    }
    let (a, b) = (user.id.get(), other.id.get());
    let card = novelty::lick_card(
        &key,
        &bot,
        ctx.data().bot_id(),
        (&user.mention().to_string(), a),
        (&other.mention().to_string(), b),
    );
    let (wa, wb) = (who(ctx, &user).await, who(ctx, &other).await);
    let prompt = novelty::lick_prompt(&key, &bot, (&wa, a), (&wb, b));
    card_then_reply(ctx, card, prompt).await
}

/// The bot roasts someone.
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn roast(
    ctx: Context<'_>,
    #[description = "Who to roast"] user: serenity::User,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[&user]).await? {
        return Ok(());
    }
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let prompt = novelty::roast_prompt(&bot, &who(ctx, &user).await);
    card_then_reply(ctx, novelty::roast_card(&bot, &user.mention().to_string()), prompt).await
}

/// The bot reads your fortune in rat bones (a new one every time).
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn fortune(ctx: Context<'_>) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let omen = novelty::omen();
    let card = novelty::fortune_card(&bot, &ctx.author().mention().to_string(), omen);
    card_then_reply(ctx, card, novelty::fortune_prompt(&bot, omen)).await
}

/// The bot appraises someone's hoard. It comes from their ID and never changes.
#[poise::command(
    slash_command,
    rename = "hoard-value",
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn hoard_value(
    ctx: Context<'_>,
    #[description = "Whose hoard to appraise"] user: serenity::User,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[&user]).await? {
        return Ok(());
    }
    let key = ctx.data().novelty_key.clone();
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let (items, gold) = novelty::hoard(&key, user.id.get());
    let card = novelty::hoard_card(&bot, &user.mention().to_string(), &items, gold);
    let prompt = novelty::hoard_prompt(&bot, &who(ctx, &user).await, &items, gold);
    card_then_reply(ctx, card, prompt).await
}

/// Someone's age in kobold years. It comes from their ID and never changes.
#[poise::command(
    slash_command,
    rename = "kobold-age",
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn kobold_age(
    ctx: Context<'_>,
    #[description = "Whose kobold age"] user: serenity::User,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[&user]).await? {
        return Ok(());
    }
    let key = ctx.data().novelty_key.clone();
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let who = who(ctx, &user).await;
    let age = novelty::kobold_age(&key, user.id.get());
    let card = match who {
        novelty::Who::Bot => novelty::bot_age_card(&bot),
        _ => novelty::age_card(&bot, &user.mention().to_string(), age),
    };
    card_then_reply(ctx, card, novelty::age_prompt(&bot, &who, age)).await
}

/// The bot judges anything out of 10 shiny rocks. The same thing always gets the same score.
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn judge(
    ctx: Context<'_>,
    #[description = "What to judge"]
    #[max_length = 200]
    thing: String,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    let key = ctx.data().novelty_key.clone();
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let thing = novelty::judge_text(&thing);
    let score = novelty::judge_score(&key, &thing);
    let card = novelty::judge_card(&bot, &thing, score);
    card_then_reply(ctx, card, novelty::judge_prompt(&bot, &thing, score)).await
}

/// The bot checks your path for traps.
#[poise::command(
    slash_command,
    rename = "trap-check",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn trap_check(ctx: Context<'_>) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let trap = novelty::trap();
    let card = novelty::trap_card(&bot, &ctx.author().mention().to_string(), trap);
    card_then_reply(ctx, card, novelty::trap_prompt(&bot, trap)).await
}

/// The bot mixes you a drink from cave ingredients.
#[poise::command(
    slash_command,
    nsfw_only,
    install_context = "Guild",
    interaction_context = "Guild"
)]
async fn drink(ctx: Context<'_>) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let d = novelty::drink();
    let card = novelty::drink_card(&bot, &ctx.author().mention().to_string(), d);
    card_then_reply(ctx, card, novelty::drink_prompt(&bot, d)).await
}

/// Bet shiny rocks on a roll of the bones against the bot (nothing is kept).
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn gamble(
    ctx: Context<'_>,
    #[description = "How many shiny rocks to bet"]
    #[min = 1]
    #[max = 1000]
    rocks: u32,
) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let g = novelty::Gamble::roll(rocks);
    let card = g.card(&bot, &ctx.author().mention().to_string());
    card_then_reply(ctx, card, g.prompt(&bot)).await
}

/// Something dragon-shaped is nearby. The bot reacts.
#[poise::command(
    slash_command,
    rename = "dragon-alert",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn dragon_alert(ctx: Context<'_>) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    ctx.defer().await?;
    let bot = persona_name(ctx);
    let sign = novelty::dragon_sign();
    card_then_reply(ctx, novelty::dragon_card(sign), novelty::dragon_prompt(&bot, sign)).await
}

/// The bot sums up the vibe of this channel's recent messages.
#[poise::command(
    slash_command,
    rename = "vibe-check",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn vibe_check(ctx: Context<'_>) -> Result<(), Error> {
    if !novelty_allowed(ctx, &[]).await? {
        return Ok(());
    }
    let data = ctx.data();
    let ch = ctx.channel_id();
    // Only where the bot keeps history.
    if !talks_here(ctx).await {
        let msg = format!(
            "{} only checks the vibe in channels it talks in.",
            data.cfg.discord.bot_name
        );
        // Nothing generated: no bill.
        data.limit_release(data.take_hold(ctx.id()));
        notice(ctx, msg).await?;
        return Ok(());
    }
    ctx.defer().await?;
    let recent: Vec<(String, String)> = {
        let mut windows = data.windows.lock().await;
        let window = windows
            .entry(ch.get())
            .or_insert_with(|| ChannelWindow::after(data.start_anchor));
        if !window.loaded {
            dispatch::seed_window(data, &ctx.serenity_context().http, ch.get(), window).await;
        }
        let msgs: Vec<_> = window
            .messages()
            .filter(|m| !data.optouts.is_out(m.author_id))
            .collect();
        msgs[msgs.len().saturating_sub(novelty::VIBE_MESSAGES)..]
            .iter()
            .map(|m| (m.author.clone(), m.text.clone()))
            .collect()
    };
    let bot = persona_name(ctx);
    card_then_reply(ctx, novelty::vibe_card(&bot), novelty::vibe_prompt(&bot, &recent)).await
}

/// Whether the command's channel is one the bot talks (and keeps history) in:
/// the same scope as channel messages.
async fn talks_here(ctx: Context<'_>) -> bool {
    let data = ctx.data();
    let ch = ctx.channel_id();
    let parent = match ctx.guild_id() {
        Some(_) => thread_parent(ctx.serenity_context(), data, ch).await,
        None => None,
    };
    let facts = MessageFacts {
        author_id: ctx.author().id.get(),
        guild_id: ctx.guild_id().map(|g| g.get()),
        channel_id: ch.get(),
        parent_channel_id: parent,
        ..Default::default()
    };
    in_scope(&facts, &data.cfg.discord)
}

/// What `/super-admin` does; Discord shows these as a dropdown.
#[derive(poise::ChoiceParameter, Clone, Copy, Debug)]
enum AdminAction {
    #[name = "reset: forget this channel's conversation"]
    Reset,
    #[name = "persona-show: the active persona and reminder"]
    PersonaShow,
    #[name = "persona-reload: re-read the persona file"]
    PersonaReload,
    #[name = "persona-dump: the active persona file (.rsn)"]
    PersonaDump,
    #[name = "clearqueue: drop every waiting reply"]
    ClearQueue,
    #[name = "context: the last prompt in this channel"]
    Context,
    #[name = "idle: what the bot was up to here (its scene)"]
    Idle,
    #[name = "idle-next: start a new scene here now"]
    IdleNext,
    #[name = "idle-clear: end this channel's scene"]
    IdleClear,
    // Say-literal (`shake`): no description, on purpose, so the menu doesn't
    // advertise that the bot can be made to speak.
    #[name = "shake"]
    Shake,
    // No description, on purpose: an obscure testing aid (`do_test`).
    #[name = "test"]
    DoTest,
}

/// `/super-admin idle`, `idle-next`, `idle-clear`: this channel's scene
/// (`chat::idle`): show it, start a new one now, or end it.
async fn idle(ctx: Context<'_>, action: AdminAction) -> Result<(), Error> {
    let data = ctx.data();
    let ch = ctx.channel_id().get();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    let persona = data.persona();
    let (p, bot) = (&persona.persona, persona.prompt.bot_name.clone());
    let scene = match action {
        AdminAction::IdleNext => data.idle.next(ch, now, &p.idle_activities, &p.idle_thoughts),
        AdminAction::IdleClear => {
            data.idle.clear(ch, now);
            None
        }
        _ => data.idle.current(ch),
    };
    let body = match (action, scene) {
        (AdminAction::IdleClear, _) => "Scene ended; the next one waits for an hour without a reply.".to_string(),
        (_, None) if p.idle_activities.is_empty() && p.idle_thoughts.is_empty() => {
            "The persona has no idle_activities or idle_thoughts.".to_string()
        }
        (_, None) => "No scene here (an hour without a reply starts one).".to_string(),
        (_, Some(s)) => {
            let state = match s.until_ms {
                None => "starts at the next reply".to_string(),
                Some(u) if u > now => format!("{} more minutes", (u - now) / 60_000),
                Some(_) => "expired".into(),
            };
            format!("{}
({state})", s.line(&bot))
        }
    };
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

/// Admin-only actions.
///
/// One command, so the `/` menu shows a single entry.
#[poise::command(
    slash_command,
    rename = "super-admin",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn admin(
    ctx: Context<'_>,
    #[description = "What to do"] action: AdminAction,
) -> Result<(), Error> {
    if !require_admin(ctx).await? {
        return Ok(());
    }
    match action {
        AdminAction::Reset => reset(ctx).await,
        AdminAction::PersonaShow => persona_show(ctx).await,
        AdminAction::PersonaReload => persona_reload(ctx).await,
        AdminAction::PersonaDump => persona_dump(ctx).await,
        AdminAction::ClearQueue => clearqueue(ctx).await,
        AdminAction::Context => context(ctx).await,
        AdminAction::Idle | AdminAction::IdleNext | AdminAction::IdleClear => idle(ctx, action).await,
        AdminAction::Shake => shake::shake(ctx).await,
        AdminAction::DoTest => do_test(ctx).await,
    }
}

/// Forget this channel's conversation (history window and KV cache; admin only).
async fn reset(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let ch = ctx.channel_id().get();
    // History restarts at this command: the interaction's ID is its timestamp.
    data.windows
        .lock()
        .await
        .insert(ch, ChannelWindow::after(ctx.id()));
    data.last_prompts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&ch);
    data.resets.reset(ch);
    data.retry().forget(ch);
    data.traces.forget(ch);
    // A new scene for the next reply, and its history prefilled ahead.
    let p = data.persona();
    let ms = crate::chat::window::snowflake_ms(ctx.id());
    if data.idle.reset(ch, ms, &p.persona.idle_activities, &p.persona.idle_thoughts) {
        let _ = data.dispatch.send(DispatchMsg::Warm(ch));
    }
    ctx.send(
        poise::CreateReply::default()
            .content("Conversation reset.")
            .ephemeral(true),
    )
    .await?;
    Ok(())
}

/// Show the active persona: name, fingerprint, reminder (admin only).
async fn persona_show(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let a = data.persona();
    let header = a
        .prompt
        .check_header(usize::MAX, &|s| data.count_tokens(s))
        .unwrap_or(0);
    let p = &a.persona;
    // Discord allows 2000 characters; the reminder matters most, then the start
    // of the persona.
    let cut = |s: &str, n: usize| {
        if s.chars().count() > n {
            s.chars().take(n).collect::<String>() + "…"
        } else {
            s.to_string()
        }
    };
    let body = format!(
        "persona: {} — `{}`, {header} tokens with the template header\n\
         {} notes, {} word biases, {} stop strings\n\
         reminder:\n```\n{}\n```persona:\n```\n{}\n```",
        p.line(),
        data.cfg.discord.persona_file,
        p.notes.len(),
        p.discourage.len(),
        p.stop_strings.len(),
        cut(&p.reminder, 600),
        cut(&p.persona, 900),
    );
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

/// The active persona file as it was loaded, comments and all, attached
/// (admin only). Says so when the file on disk has changed since, or an
/// `INCREBOT_DISCOURAGE` override replaces its word list.
async fn persona_dump(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let a = data.persona();
    let path = &data.cfg.discord.persona_file;
    let text = a.persona.source.0.to_string();
    let mut body = format!("persona: {}, loaded from `{path}`.", a.persona.line());
    match std::fs::read_to_string(path) {
        Ok(disk) if disk != text => body += " The file on disk has changed since; persona-reload loads it.",
        Ok(_) => {}
        Err(e) => body += &format!(" (The file can't be read now: {e}.)"),
    }
    if std::env::var_os("INCREBOT_DISCOURAGE").is_some() {
        body += " INCREBOT_DISCOURAGE replaces its discourage list.";
    }
    let name = Path::new(path).file_name().map_or("persona.rsn".into(), |n| n.to_string_lossy().into_owned());
    ctx.send(
        poise::CreateReply::default()
            .content(body)
            .attachment(serenity::CreateAttachment::bytes(text.into_bytes(), name))
            .ephemeral(true),
    )
    .await?;
    Ok(())
}

/// Re-read the persona file and apply all of it at once (admin only).
async fn persona_reload(ctx: Context<'_>) -> Result<(), Error> {
    ctx.defer_ephemeral().await?;
    let msg = match ctx.data().reload_persona() {
        Ok(h) => format!("Reloaded ({h} tokens). Each channel's next reply starts cold."),
        Err(e) => format!("Reload failed, keeping the old persona: {e:#}"),
    };
    ctx.send(poise::CreateReply::default().content(msg).ephemeral(true))
        .await?;
    Ok(())
}

/// Stop the bot from reading your messages; it also forgets the ones it holds.
#[poise::command(
    slash_command,
    rename = "opt-out",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn optout(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let user = ctx.author().id.get();
    let body = match data.optouts.opt_out(user) {
        Ok(new) => {
            forget_author(data, user).await;
            if new {
                info!(opted_out = data.optouts.len(), "a user opted out");
            }
            format!(
                "{}{} won't read or keep your messages from now on, and has forgotten the \
                 ones it was holding. It won't reply to your messages either; `/ask` still \
                 works. Use `/opt-in` to undo this.",
                if new { "Done. " } else { "You're already opted out. " },
                data.cfg.discord.bot_name,
            )
        }
        Err(e) => {
            error!(error = %format!("{e:#}"), "could not save an opt-out");
            "Sorry, that couldn't be saved. Please try again later.".to_string()
        }
    };
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

/// Let the bot read your messages again (undoes /opt-out).
#[poise::command(
    slash_command,
    rename = "opt-in",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn optin(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let body = match data.optouts.opt_in(ctx.author().id.get()) {
        Ok(true) => {
            info!(opted_out = data.optouts.len(), "a user opted back in");
            format!(
                "Done. {} will read your new messages again in the channels it talks in.",
                data.cfg.discord.bot_name
            )
        }
        Ok(false) => "You're already opted in; that's the default.".to_string(),
        Err(e) => {
            error!(error = %format!("{e:#}"), "could not save an opt-in");
            "Sorry, that couldn't be saved. Please try again later.".to_string()
        }
    };
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

/// The checks `/retry` and `/retry-batch` share (`chat::retry`): rules 1–4,
/// then the opt-outs since. On success the channel's retry lock is held (released
/// by `retry().end`) and the output with the source to generate it from is
/// returned; else the reason.
async fn begin_retry(data: &Data, channel: u64, user: u64) -> Result<(retry::Output<Replay>, Source), String> {
    let rules = &data.cfg.discord.retry;
    let begun = data.retry().begin(channel, user, Instant::now());
    let out = begun.map_err(|r| r.message(rules))?;
    let mut source = out.replay.source.clone();
    if !without_optouts(data, &mut source) {
        data.retry().end(channel);
        return Err(retry::Refusal::Nothing.message(rules));
    }
    Ok((out, source))
}

/// What an action answers its caller (ephemeral): the text, and how long it
/// stays (ms).
type Answer = (String, u64);

/// `/retry`'s work, for the slash command and "Reply actions": checks, then
/// queues the retry and streams it in the background. `ask_key` keys a
/// one-shot job in the queue.
async fn do_retry(data: &Data, http: Arc<serenity::Http>, channel: ChannelId, user: u64, ask_key: u64) -> Answer {
    let d = &data.cfg.discord;
    // Billed now, kept if the retry is generated.
    let hold = match data.limit_try_hold(user, 1.0) {
        Ok(hold) => hold,
        Err(why) => return (why, d.notice_ttl_ms),
    };
    let (out, source) = match begin_retry(data, channel.get(), user).await {
        Ok(begun) => begun,
        Err(why) => {
            data.limit_release(hold);
            return (why, d.notice_ttl_ms);
        }
    };
    data.trace_source(channel.get(), &out.replay.source, format!("/retry by user {user}"));
    let Some(job) = requeue(data, http, channel, ask_key, &out, source, None).await else {
        data.retry().end(channel.get());
        data.limit_release(hold);
        return (busy(), d.notice_ttl_ms);
    };
    let data = data.clone();
    tokio::spawn(async move {
        let generated = job.await;
        data.retry().end(channel.get());
        if !generated {
            data.limit_release(hold);
        }
    });
    ("Retrying…".to_string(), d.retrying_ttl_ms)
}

/// `/undo-retry`'s work, for the slash command and "Reply actions".
async fn do_undo(data: &Data, http: Arc<serenity::Http>, channel: ChannelId, user: u64) -> Answer {
    let d = &data.cfg.discord;
    let begun = data.retry().begin_undo(channel.get(), user, Instant::now());
    let (cur, prev) = match begun {
        Ok(pair) => pair,
        Err(r) => return (r.undo_message(&d.retry), d.notice_ttl_ms),
    };
    let restored = reply::restore(http, channel, data, &cur, prev).await;
    let ok = restored.is_some();
    data.trace_source(channel.get(), &cur.replay.source, if ok {
        format!("/undo-retry by user {user}: the earlier reply is back")
    } else {
        format!("/undo-retry by user {user}: couldn't put the earlier reply back")
    });
    data.retry().end_undo(channel.get(), restored);
    if ok {
        ("Undone.".to_string(), d.retrying_ttl_ms)
    } else {
        ("Couldn't put the earlier reply back.".to_string(), d.notice_ttl_ms)
    }
}

/// `/delete-reply`'s work, for the slash command and "Reply actions".
async fn do_delete(data: &Data, http: Arc<serenity::Http>, channel: ChannelId, user: u64) -> Answer {
    let d = &data.cfg.discord;
    let ch = channel.get();
    let begun = data.retry().begin_delete(ch, user);
    let out = match begun {
        Ok(out) => out,
        Err(r) => return (r.delete_message(&d.retry), d.notice_ttl_ms),
    };
    // Nobody may post for a moment after it either, as with `/retry`.
    tokio::time::sleep(d.retry.delete_quiet()).await;
    let quiet = data.retry().still_quiet(ch, &out);
    if let Err(r) = quiet {
        data.retry().end_delete(ch, false);
        return (r.delete_message(&d.retry), d.notice_ttl_ms);
    }
    for (id, _) in &out.messages {
        // Already deleted by hand: nothing left to do for it.
        if let Err(e) = channel.delete_message(&http, serenity::MessageId::new(*id)).await {
            debug!("deleting a reply message: {e}");
        }
    }
    // Gone from the history now, not when the gateway's delete events arrive.
    if let Some(w) = data.windows.lock().await.get_mut(&ch) {
        for (id, _) in &out.messages {
            w.remove(*id);
        }
    }
    data.retry().end_delete(ch, true);
    data.trace_source(ch, &out.replay.source, format!("/delete-reply by user {user}: reply deleted"));
    ("Deleted.".to_string(), d.retrying_ttl_ms)
}

/// The notice while shutting down, if so.
fn restarting(data: &Data) -> Option<String> {
    data.shutting_down
        .load(Ordering::Relaxed)
        .then(|| format!("{NOTICE_PREFIX}restarting, ask again shortly)*"))
}

/// Takes people who opted out since out of a channel reply's frozen history;
/// false if that took what it answered.
fn without_optouts(data: &Data, source: &mut Source) -> bool {
    let Source::Channel { trigger_msg, snapshot } = source else {
        return true;
    };
    let gone: Vec<u64> = snapshot
        .messages()
        .map(|m| m.author_id)
        .filter(|&a| data.optouts.is_out(a))
        .collect();
    for a in gone {
        snapshot.remove_author(a);
    }
    snapshot.messages().any(|m| m.id == *trigger_msg)
}

/// Generate your last reply here again, if nobody has posted since.
///
/// design.md §6.
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn retry(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data().clone();
    if let Some(n) = restarting(&data) {
        return notice(ctx, n).await;
    }
    // Ephemeral, so the channel gets no new message and stays retryable.
    ctx.defer_ephemeral().await?;
    let http = ctx.serenity_context().http.clone();
    let (text, ttl) = do_retry(&data, http, ctx.channel_id(), ctx.author().id.get(), ctx.id()).await;
    notice_for(ctx, text, ttl).await
}

/// The notice for a full queue.
fn busy() -> String {
    format!("{NOTICE_PREFIX}busy: too many replies waiting, try again in a minute)*")
}

/// Queues `out`'s job again from `source` (`out`'s replay, its history
/// perhaps trimmed of opt-outs since) and returns the task that streams the
/// new text into `out`'s messages: a `/retry`, or with `replaces` an admin's
/// rewrite in place; it ends true if the text was generated. None when the
/// queue is full. `ask_key` keys a one-shot
/// job in the queue.
async fn requeue(
    data: &Data,
    http: Arc<serenity::Http>,
    channel: ChannelId,
    ask_key: u64,
    out: &retry::Output<Replay>,
    source: Source,
    replaces: Option<u64>,
) -> Option<impl std::future::Future<Output = bool> + Send + 'static> {
    let history = source.keeps_history(&data.persona().persona);
    let replay = Replay {
        source: source.clone(),
        ..out.replay.clone()
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let (result_tx, result_rx) = tokio::sync::oneshot::channel();
    let trigger = Trigger {
        key: match source {
            Source::Channel { .. } => QueueKey::Channel(channel.get()),
            Source::Ask { .. } => QueueKey::Ask(ask_key),
        },
        channel: channel.get(),
        source,
        requester: out.requester,
        cancel: cancel.clone(),
        events: events_tx,
        avoid: replay.avoid.clone(),
        paced: false,
    };
    data.dispatch.send(DispatchMsg::Enqueue(Box::new(trigger), result_tx)).ok()?;
    let EnqueueResult::Queued { ahead } = result_rx.await.ok()? else {
        return None;
    };
    let mut sink = reply::Sink::redo(http, channel, replay.header.clone(), &out.messages, history);
    if let Some(id) = replaces {
        sink = sink.replacing(id);
    }
    let (data, requester) = (data.clone(), out.requester);
    Some(async move { reply::stream(sink, data, events_rx, cancel, requester, ahead, replay).await })
}

/// Undo your last /retry here: the reply goes back to what it said before.
#[poise::command(
    slash_command,
    rename = "undo-retry",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn undo_retry(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data().clone();
    if let Some(n) = restarting(&data) {
        return notice(ctx, n).await;
    }
    // Ephemeral, so the channel gets no new message and stays retryable.
    ctx.defer_ephemeral().await?;
    let http = ctx.serenity_context().http.clone();
    let (text, ttl) = do_undo(&data, http, ctx.channel_id(), ctx.author().id.get()).await;
    notice_for(ctx, text, ttl).await
}

/// Delete the reply you asked for here, while nobody has posted since.
#[poise::command(
    slash_command,
    rename = "delete-reply",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn delete_reply(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data().clone();
    if let Some(n) = restarting(&data) {
        return notice(ctx, n).await;
    }
    ctx.defer_ephemeral().await?;
    let http = ctx.serenity_context().http.clone();
    let (text, ttl) = do_delete(&data, http, ctx.channel_id(), ctx.author().id.get()).await;
    notice_for(ctx, text, ttl).await
}

/// Drops `user`'s messages from every history window, and the last prompts
/// (`/super-admin context`) of the channels that had any, and their messages'
/// traces ("Admin actions" → Trace). Replies already queued keep the history they
/// froze when they were asked.
async fn forget_author(data: &Shared, user: u64) {
    data.traces.forget_author(user);
    data.retry().forget_author(user);
    let channels: Vec<u64> = data
        .windows
        .lock()
        .await
        .iter_mut()
        .filter_map(|(&ch, w)| w.remove_author(user).then_some(ch))
        .collect();
    let mut prompts = data.last_prompts.lock().unwrap_or_else(|e| e.into_inner());
    for ch in channels {
        prompts.remove(&ch);
    }
}

/// Drop every reply still waiting in the queue (admin only).
async fn clearqueue(ctx: Context<'_>) -> Result<(), Error> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    ctx.data()
        .dispatch
        .send(DispatchMsg::Clear(tx))
        .map_err(|_| "dispatcher stopped")?;
    let n = rx.await.map_err(|_| "dispatcher stopped")?;
    let body = match n {
        0 => "The queue was already empty.".to_string(),
        1 => "Cleared 1 waiting reply. Any reply being written now continues.".to_string(),
        n => format!("Cleared {n} waiting replies. Any reply being written now continues."),
    };
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

/// The first 5 novelty keys (of 0–999,999) under which the caller and the bot
/// get what `hotdog::WISH` asks for, then both their results for keys 0–100,
/// as an attached file (admin only). Read-only: the configured key is only
/// used to mark its rows.
async fn do_test(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let live = data.cfg.discord.novelty_key;
    let me = ctx.author().display_name().to_string();
    let bot = persona_name(ctx);
    let (me_id, bot_id) = (ctx.author().id.get(), data.bot_id());
    ctx.defer_ephemeral().await?;
    // Up to a million keys, off the async threads.
    let (found, last) =
        tokio::task::spawn_blocking(move || hotdog::scan(me_id, bot_id, 0..=999_999, &hotdog::WISH, 5)).await?;
    let text = format!(
        "{}\n{}\n{}",
        hotdog::matches_table(&me, &bot, &hotdog::WISH, &found, last, live),
        hotdog::key_table(&me, me_id, 0..=100, live),
        hotdog::key_table(&bot, bot_id, 0..=100, live),
    );
    let summary = match found.len() {
        0 => format!("No key in 0-{last} where {}.", hotdog::wish_text(&me, &bot, &hotdog::WISH)),
        _ => format!(
            "Keys where {}: {}. Tables for keys 0-100 attached.",
            hotdog::wish_text(&me, &bot, &hotdog::WISH),
            found
                .iter()
                .map(|r| format!("{} (`{}`)", r.key, hotdog::hex_key(r.key)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let reply = poise::CreateReply::default()
        .content(summary)
        .allowed_mentions(serenity::CreateAllowedMentions::new())
        .attachment(serenity::CreateAttachment::bytes(text.into_bytes(), "test.txt"))
        .ephemeral(true);
    ctx.send(reply).await?;
    Ok(())
}

/// Show the last prompt the bot was given in this channel (admin only).
async fn context(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let last = data
        .last_prompts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&ctx.channel_id().get())
        .cloned();
    let reply = match last {
        None => poise::CreateReply::default()
            .content("No reply has been generated in this channel since startup or /super-admin reset."),
        // Prompts run to ~4,000 characters, over Discord's 2,000 limit: attach it.
        Some(text) => poise::CreateReply::default()
            .content(format!(
                "The last prompt in this channel ({} tokens), exactly as the model saw it:",
                data.count_tokens(&text)
            ))
            .attachment(serenity::CreateAttachment::bytes(
                text.into_bytes(),
                "prompt.txt",
            )),
    };
    ctx.send(reply.ephemeral(true)).await?;
    Ok(())
}

/// How long the bot has been running, as `/stats` and `/version` show it: "3h 12m".
fn uptime(running: std::time::Duration) -> String {
    let s = running.as_secs();
    format!("{}h {}m", s / 3600, s / 60 % 60)
}

/// Which build and persona setup the bot is running.
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn version(ctx: Context<'_>) -> Result<(), Error> {
    let body = format!(
        "version: {}\ncommit: {}\nbuilt: {} ({})\npersona: {}\nuptime: {}",
        env!("CARGO_PKG_VERSION"),
        crate::GIT_COMMIT,
        crate::BUILD_TIMESTAMP,
        crate::build_age(std::time::SystemTime::now()),
        ctx.data().persona().persona.line(),
        uptime(ctx.data().started.elapsed()),
    );
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

/// Reply times, speed, queue, cache and persona statistics.
#[poise::command(
    slash_command,
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
async fn stats(ctx: Context<'_>) -> Result<(), Error> {
    let data = ctx.data();
    let s = data.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let hit = if s.prompt_tokens > 0 {
        100.0 * s.reused_tokens as f64 / s.prompt_tokens as f64
    } else {
        0.0
    };
    let header = data
        .prompt()
        .check_header(usize::MAX, &|t| data.count_tokens(t))
        .unwrap_or(0);
    let last = if s.jobs == 0 {
        "no replies yet".to_string()
    } else {
        format!(
            "total: {:.2} s on {}\n\
             prefill: {} tokens in {:.2} s at {:.1} tok/s ({} more reused from cache)\n\
             decode: {} tokens in {:.2} s at {:.2} tok/s",
            s.last_total.as_secs_f64(),
            s.last_backend,
            s.last_prefill_tokens,
            s.last_prefill.as_secs_f64(),
            s.last_prefill_tok_s,
            s.last_reused_tokens,
            s.last_decode_tokens,
            s.last_decode.as_secs_f64(),
            s.last_decode_tok_s,
        )
    };
    // Medians over the last replies, GPU and CPU apart (their speeds differ
    // ~10x); replies that moved between them mid-way apart too.
    let recent: String = [("gpu", &s.recent.gpu), ("cpu", &s.recent.cpu), ("moved gpu → cpu", &s.recent.moved)]
        .into_iter()
        .filter_map(|(kind, times)| {
            crate::engine::worker::summarize(times).map(|line| format!("{kind} (last {}): {line}\n", times.len()))
        })
        .collect();
    let recent = if recent.is_empty() {
        String::new()
    } else {
        format!("recent replies:\n{recent}")
    };
    let gpus = match &data.gpus {
        Some(g) => format!(
            "GPUs:\n{}",
            crate::gpu::manager::describe(&g.status(), std::time::SystemTime::now())
        ),
        None => String::new(),
    };
    let warm = if s.seed.is_empty() { String::new() } else { format!(" (prefilled: {})", s.seed) };
    let body = format!(
        "replies: {}\nlast reply:\n{last}\n{recent}{gpus}prompt cache hits: {hit:.0}% of prompt tokens\nqueue: {} waiting (max {} per channel)\nsystem prompt + header: {header} tokens{warm}\npersona: {}\nuptime: {}\nversion: {}",
        s.jobs,
        dispatch::queue_len(),
        data.cfg.runtime.queue_depth,
        data.persona().persona.line(),
        uptime(data.started.elapsed()),
        crate::version_with_age(std::time::SystemTime::now()),
    );
    ctx.send(poise::CreateReply::default().content(body).ephemeral(true))
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn uptime_reads_hours_and_minutes() {
        use std::time::Duration;
        assert_eq!(uptime(Duration::from_secs(0)), "0h 0m");
        assert_eq!(uptime(Duration::from_secs(3 * 3600 + 12 * 60 + 59)), "3h 12m");
        assert_eq!(uptime(Duration::from_secs(50 * 3600)), "50h 0m");
    }

    #[test]
    fn mentions_resolve() {
        let out = resolve_mentions("hi <@&5> in <#7>, <@9> and <a:x:1>", &[], &[], "IncreBot");
        assert_eq!(out, "hi @role in #channel, @someone and <a:x:1>");
        // The bot's own role reads as the bot, other roles stay generic.
        let out = resolve_mentions("<@&5> hi, <@&6> too", &[], &[5], "IncreBot");
        assert_eq!(out, "@IncreBot hi, @role too");
    }
}
