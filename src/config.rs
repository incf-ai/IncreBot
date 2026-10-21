//! `config.rsn` plus `INCREBOT_<KEY>` env overrides (design.md §8).
//!
//! Every key can be overridden by an env var named after the key alone, without
//! its section (`INCREBOT_THREADS` sets `runtime.threads`). That is unambiguous
//! because key names are unique across sections, which [`Config::load`] asserts.
//! List values are comma-separated, each `id` or `id=name`
//! (`INCREBOT_GUILDS=849937185893384223=KoboldAI,1444870717525983285`).
//!
//! How the bot replies (prompt, notes, word biases, sampler, reply length) is
//! in the persona file `discord.persona_file` names ([`crate::persona`]).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::model::LinearImpl;

pub const DEFAULT_TEMPLATE: &str = "{{ system_prompt }}

### Instruction:
Continue the conversation below as {{ bot_name }}.

{% for m in history %}{{ m.author }}: {{ m.content }}
{% endfor %}
{% if scene %}{{ scene }}

{% endif %}{% if repeat_persona %}{{ reminder }}

{% endif %}{% if note %}{{ note }}

{% endif %}### Instruction:
{% if recent %}The latest messages, which {{ bot_name }} is answering:
{% for m in recent %}{{ m.author }}: {{ m.content }}
{% endfor %}
{% endif %}Continue the conversation as {{ bot_name }}.

### Response:
{% if reply_to %}({{ bot_name }} is replying to {{ reply_to }})
{% endif %}{{ bot_name }}:";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub model: ModelConfig,
    pub runtime: RuntimeConfig,
    pub generation: GenerationConfig,
    pub discord: DiscordConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "Model")]
pub struct ModelConfig {
    pub repo: String,
    pub file: String,
    pub tokenizer_repo: String,
    pub n_ctx: usize,
    pub linear_impl: LinearImpl,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            repo: "TheBloke/LLaMA2-13B-Tiefighter-GGUF".into(),
            file: "llama2-13b-tiefighter.Q8_0.gguf".into(),
            tokenizer_repo: "hf-internal-testing/llama-tokenizer".into(),
            n_ctx: 4096,
            linear_impl: LinearImpl::Packed,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "Runtime")]
pub struct RuntimeConfig {
    pub threads: usize,
    pub pin_threads: bool,
    pub kv_cache_slots: usize,
    /// Keep the active KV cache in f16: half the memory, no measured speed gain
    /// (the f16 → f32 widening in attention offsets the bandwidth saved).
    pub kv_cache_f16: bool,
    /// Replies waiting per channel, behind any running one; newer triggers
    /// there are turned away when full. `/ask` and the other replies through a
    /// Discord interaction are also turned away once more than
    /// [`GenerationConfig::interaction_ahead`] jobs wait in all.
    pub queue_depth: usize,
    /// GPUs to run on when no other program uses them, by PCI address, most
    /// preferred first (gpu-plan.md). Empty = CPU only.
    pub gpus: Vec<Gpu>,
    /// Other programs' VRAM on a card above this many MiB = contended.
    pub gpu_other_mib: u64,
    /// VRAM monitor readings per second.
    pub gpu_poll_hz: u32,
    /// A contended card must be free this long before it is used again; doubles
    /// each time it flaps, up to 10 min.
    pub gpu_holdoff_secs: u64,
    /// Keep the weights on the next free GPU too, as a warm fallback.
    pub gpu_standby: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            threads: 8,
            pin_threads: true,
            kv_cache_slots: 2,
            kv_cache_f16: false,
            queue_depth: 2,
            gpus: Vec::new(),
            gpu_other_mib: 100,
            gpu_poll_hz: 20,
            gpu_holdoff_secs: 30,
            gpu_standby: true,
        }
    }
}

impl RuntimeConfig {
    /// The `[runtime]` GPU settings for the manager.
    pub fn gpu_settings(&self, n_ctx: usize) -> Result<crate::gpu::manager::GpuSettings> {
        Ok(crate::gpu::manager::GpuSettings {
            gpus: self
                .gpus
                .iter()
                .map(|g| {
                    let pci = g.pci.parse().with_context(|| format!("gpu {g}"))?;
                    Ok(crate::gpu::manager::GpuSpec { pci, label: g.to_string(), name: g.name.clone() })
                })
                .collect::<Result<_>>()?,
            other_mib: self.gpu_other_mib,
            poll_hz: self.gpu_poll_hz,
            holdoff: std::time::Duration::from_secs(self.gpu_holdoff_secs),
            standby: self.gpu_standby,
            n_ctx,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "Generation")]
pub struct GenerationConfig {
    pub max_prompt_tokens: usize,
    pub max_message_tokens: usize,
    /// The most tokens the system prompt and fixed template text may take, so
    /// enough is left for the history; 0 = half of `max_prompt_tokens`.
    pub max_header_tokens: usize,
    /// Leave out the persona the default template repeats after the history,
    /// and its copy of the latest messages (`recent`), when no GPU is ready
    /// (design.md §7): both come after the history, so the CPU prefills them
    /// for every reply (~200 tokens, ~7 s with the 13B, for the reminder; up
    /// to ~480 for the copy).
    pub cpu_drop_persona_repeat: bool,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Repeat")]
pub struct Repeat {
    pub penalty: f32,
    /// Window over the most recent `scope` tokens.
    pub last_n: usize,
    pub scope: RepeatScope,
}

impl Default for Repeat {
    fn default() -> Self {
        Self {
            penalty: 1.1,
            last_n: 256,
            scope: RepeatScope::Context,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Dry")]
pub struct Dry {
    pub multiplier: f32,
    pub base: f32,
    /// Repeats up to this many tokens long are free.
    pub allowed_length: usize,
    /// Characters that end a match; any token containing one is a breaker.
    pub sequence_breakers: String,
}

impl Default for Dry {
    fn default() -> Self {
        Self {
            multiplier: 0.8,
            base: 1.75,
            allowed_length: 2,
            sequence_breakers: "\n:\"*".into(),
        }
    }
}

/// A line for the prompt when the user's message has one of `prompted_by`
/// (matched like `Word::prompted_by`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Note")]
pub struct PromptNote {
    pub text: String,
    pub prompted_by: Vec<String>,
    /// Replies the note stays on for after one of its phrases (1 = only the
    /// reply to that message). For the notes that matter most, when users keep
    /// pushing a premise over several messages (design.md §7). 1..=5.
    #[serde(default = "one_turn", skip_serializing_if = "is_one_turn")]
    pub turns: usize,
    /// Only once the bot has made this many replies since the latest message
    /// with one of the phrases, and for `turns` replies from then ("he has
    /// protested enough: now he gives in"). 1..=5; unset = from the message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_replies: Option<usize>,
    /// What the note is for: when several apply, `Safety` ones come first and
    /// the rest follow in file order. Unset = `Personality`.
    #[serde(default, skip_serializing_if = "NoteKind::is_default")]
    pub kind: NoteKind,
}

/// What a [`PromptNote`] is for, which orders it among the notes a reply gets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoteKind {
    /// Keeps the bot safe and in character (adult, unmovable): first.
    Safety,
    /// Who he is and how he behaves.
    #[default]
    Personality,
    /// Jokes and reactions.
    Humor,
}

impl NoteKind {
    fn is_default(&self) -> bool {
        *self == NoteKind::Personality
    }
}

fn one_turn() -> usize {
    1
}

fn is_one_turn(n: &usize) -> bool {
    *n == 1
}

/// A discouraged word stem: a token that makes the word in progress spell at
/// least `min` of its letters (or run past it) gets `bias` added to its logit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Word")]
pub struct Word {
    /// ASCII letters, any case ("gigg" for giggle, giggles, giggling).
    pub stem: String,
    /// Below 0 (only discouraging is allowed): −2.5 makes the word ~12× less
    /// likely.
    pub bias: f32,
    /// Letters of the stem a word must spell before it's biased (1..=stem
    /// length). Raise it when a shorter prefix is a common word: "seek" with 4
    /// spares " see".
    #[serde(default = "default_min_letters")]
    pub min: usize,
    /// Only bias the stem where one of these phrases ends at most `within`
    /// words earlier in the reply ("play with" for "me"); empty = everywhere.
    /// Trigger words match loosely (design.md §5).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub after: Vec<String>,
    /// Words allowed between a trigger and the stem (0 = right after).
    #[serde(default)]
    pub within: usize,
    /// Only bias the stem when one of these phrases appears anywhere in the
    /// user messages the reply answers ("hide and seek"); empty = always.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompted_by: Vec<String>,
    /// Only the word itself: once a word spells exactly the stem, bias the
    /// tokens that would end it (a space or any punctuation), so longer words
    /// stay free ("cock", not "cocktail"). `min` doesn't apply. Env form: `!`
    /// after the bias (`cock=-4!`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub whole_word: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// The most words a `discourage` trigger may be from its stem.
pub const MAX_TRIGGER_WITHIN: usize = 20;

fn default_min_letters() -> usize {
    crate::engine::sampler::STEM_MIN_LETTERS
}

/// Which tokens the repetition penalty looks at. `Context` in `config.rsn`; the
/// lowercase aliases are for env overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepeatScope {
    /// The last `last_n` tokens of the chat history + generated (not the
    /// system prompt or template text).
    #[serde(alias = "context")]
    Context,
    /// Only the tokens generated so far.
    #[serde(alias = "generated")]
    Generated,
}

impl GenerationConfig {
    /// The header limit `max_header_tokens` sets (0 = half the prompt budget).
    pub fn header_limit(&self) -> usize {
        match self.max_header_tokens {
            0 => self.max_prompt_tokens / 2,
            n => n,
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 1024,
            max_message_tokens: 120,
            max_header_tokens: 0,
            cpu_drop_persona_repeat: false,
            timeout_secs: 140,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "Discord")]
pub struct DiscordConfig {
    /// The bot's Discord-facing name (status messages, notices). Inside the
    /// prompt it is rewritten to the persona's `persona_name`.
    pub bot_name: String,
    /// The persona file: the prompt, notes, word biases, sampler and reply
    /// length ([`crate::persona`]). `/super-admin persona-reload` re-reads it.
    pub persona_file: String,
    /// Users who ran `/opt-out`, with when (JSON, written by the bot). Its
    /// directory must be writable; in the service it is a podman volume.
    pub optout_file: String,
    pub prompt_template: String,
    pub history_messages: usize,
    pub stream_edits: bool,
    pub guilds: Vec<Guild>,
    /// Channels (and their threads) where the bot answers every message.
    pub dedicated_channels: Vec<Channel>,
    /// Channels (and their threads) where it answers only @mentions and replies.
    pub shared_channels: Vec<Channel>,
    pub dm_users: Vec<User>,
    pub admins: Vec<User>,
    /// The secret behind the fixed novelty commands' results
    /// (`discord/seeded.rs`), e.g. `0x1234_5678_9abc_def0` from
    /// `openssl rand -hex 8`. Keep it secret and backed up: a new key re-rolls
    /// everyone. 0 (the default) works but is no secret.
    pub novelty_key: u64,
    /// How long an ephemeral refusal or "busy" notice stays before the bot
    /// deletes it, in ms; 0 = never. Answers the caller asked for stay.
    pub notice_ttl_ms: u64,
    /// The same for `/retry`'s "Retrying…", which the redone reply soon makes
    /// redundant.
    pub retrying_ttl_ms: u64,
    /// How often someone may make the bot generate by command
    /// (`chat::limits`).
    pub limits: Limits,
    /// The pause between a channel's replies to its messages.
    pub channel_limits: ChannelLimits,
    /// When `/retry`, `/undo-retry` and `/delete-reply` may go ahead.
    pub retry: RetryRules,
}

/// `/retry`'s rules and its relatives' (`chat::retry`, design.md §6).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "Retry")]
pub struct RetryRules {
    /// A reply can be retried (or a retry undone) this long after it first
    /// finished, in seconds.
    pub window_secs: u64,
    /// At most this many messages floob can see may have come after it.
    pub max_newer: usize,
    /// `/delete-reply` checks again this much later that nothing came since.
    pub delete_quiet_ms: u64,
    /// How many earlier outputs a retry steers away from (DRY and the
    /// repetition penalty).
    pub steer: usize,
    /// How many retries `/undo-retry` can go back through; past this the
    /// oldest retry goes, the original stays.
    pub undo_depth: usize,
    /// How many of each channel's outputs "Admin actions" can reach, the last
    /// included.
    pub admin_recent: usize,
}

impl RetryRules {
    pub fn window(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.window_secs)
    }

    pub fn delete_quiet(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.delete_quiet_ms)
    }
}

impl Default for RetryRules {
    fn default() -> Self {
        Self {
            window_secs: 300,
            max_newer: 5,
            delete_quiet_ms: 1_000,
            steer: 3,
            undo_depth: 10,
            admin_recent: 20,
        }
    }
}

/// Limits on the commands that make the bot generate (`/ask`, the novelty
/// commands, `/retry`, `/retry-batch`; not chatting), per user across
/// channels; admins too unless `admins_exempt` (`chat::limits`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "Limits")]
pub struct Limits {
    /// The least time between two of them.
    pub min_gap_ms: u64,
    /// At most `count` of them in any `ms`; a batch counts as one.
    pub windows: Vec<Window>,
    /// The backoff's extra wait, on top of `min_gap_ms`, is
    /// `backoff_unit_ms × (backoff_growth ^ pressure − 1)`, at most
    /// `backoff_cap_ms`. Each one adds 1 to the pressure (a batch
    /// `batch_cost`); it drains by 1 every `backoff_drain_ms`.
    pub backoff_unit_ms: u64,
    pub backoff_growth: f64,
    pub backoff_drain_ms: u64,
    pub backoff_cap_ms: u64,
    pub batch_cost: f64,
    /// Admins skip these limits and `/retry-batch`'s cooldown. Off by default:
    /// they're held to them like everyone (`/super-admin` and "Admin actions"
    /// are never limited).
    pub admins_exempt: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Window {
    pub count: usize,
    pub ms: u64,
}

/// How long an `/ask` may take, waiting included: Discord's 15-minute
/// interaction tokens, with margin.
pub const INTERACTION_BUDGET_SECS: u64 = 720;

impl GenerationConfig {
    /// How many jobs a reply through a Discord interaction may wait behind
    /// (the running one included), each running to the timeout, and still
    /// finish within [`INTERACTION_BUDGET_SECS`].
    pub fn interaction_ahead(&self) -> usize {
        (INTERACTION_BUDGET_SECS / self.timeout_secs.max(1)).saturating_sub(1) as usize
    }
}

/// The pause between a channel's replies to its messages (`chat::limits`):
/// after each, the next waits `delay_ms` plus a backoff, `backoff_unit_ms ×
/// (backoff_growth ^ pressure − 1)`, at most `backoff_cap_ms`. Each message
/// there that floob replies to adds 1 to the channel's pressure; it drains by
/// 1 every `backoff_drain_ms`. Slash commands neither wait nor add to it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename = "ChannelLimits")]
pub struct ChannelLimits {
    pub delay_ms: u64,
    pub backoff_unit_ms: u64,
    pub backoff_growth: f64,
    pub backoff_drain_ms: u64,
    pub backoff_cap_ms: u64,
}

impl Default for ChannelLimits {
    fn default() -> Self {
        Self {
            delay_ms: 2_000,
            backoff_unit_ms: 1_000,
            backoff_growth: 1.4,
            backoff_drain_ms: 10_000,
            backoff_cap_ms: 5_000,
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            min_gap_ms: 5_000,
            windows: vec![Window { count: 5, ms: 30_000 }, Window { count: 15, ms: 120_000 }],
            backoff_unit_ms: 1_000,
            backoff_growth: 1.4,
            backoff_drain_ms: 15_000,
            backoff_cap_ms: 45_000,
            batch_cost: 10.0,
            admins_exempt: false,
        }
    }
}

/// Writes `name (id)`, or just `id` when there is no name.
fn write_named(f: &mut std::fmt::Formatter<'_>, name: &str, id: &dyn std::fmt::Display) -> std::fmt::Result {
    if name.is_empty() {
        write!(f, "{id}")
    } else {
        write!(f, "{name} ({id})")
    }
}

/// A Discord ID with an optional nickname for logs and errors.
pub trait Named: std::fmt::Display {
    fn id(&self) -> u64;
    fn name(&self) -> &str;
}

/// Whether `list` has an entry for `id`.
pub fn listed<T: Named>(list: &[T], id: u64) -> bool {
    list.iter().any(|e| e.id() == id)
}

/// The Discord list entries. Each is its own type so the name written in
/// `config.rsn` (`Guild { id: 1, name: "x" }`) is checked against the list.
macro_rules! discord_entries {
    ($($ty:ident),*) => {$(
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $ty {
            pub id: u64,
            #[serde(default, skip_serializing_if = "String::is_empty")]
            pub name: String,
        }

        impl Named for $ty {
            fn id(&self) -> u64 {
                self.id
            }
            fn name(&self) -> &str {
                &self.name
            }
        }

        impl From<u64> for $ty {
            fn from(id: u64) -> Self {
                Self { id, name: String::new() }
            }
        }

        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write_named(f, &self.name, &self.id)
            }
        }
    )*};
}
discord_entries!(Guild, Channel, User);

/// A GPU by PCI address, with an optional nickname for logs and `/stats`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    pub pci: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
}

impl From<&str> for Gpu {
    fn from(pci: &str) -> Self {
        Self { pci: pci.into(), name: String::new() }
    }
}

impl std::fmt::Display for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_named(f, &self.name, &self.pci)
    }
}

impl DiscordConfig {
    /// Every configured guild, channel and user entry.
    fn entries(&self) -> impl Iterator<Item = &dyn Named> {
        let g = self.guilds.iter().map(|e| e as &dyn Named);
        let c = self.dedicated_channels.iter().chain(&self.shared_channels).map(|e| e as &dyn Named);
        let u = self.dm_users.iter().chain(&self.admins).map(|e| e as &dyn Named);
        g.chain(c).chain(u)
    }

    /// `name (id)` for a configured ID that has a name, else just the ID.
    pub fn label(&self, id: u64) -> String {
        match self.entries().find(|e| e.id() == id && !e.name().is_empty()) {
            Some(e) => e.to_string(),
            None => id.to_string(),
        }
    }
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            bot_name: "IncreBot".into(),
            persona_file: crate::persona::DEFAULT_PERSONA_FILE.into(),
            optout_file: "data/optouts.json".into(),
            prompt_template: DEFAULT_TEMPLATE.into(),
            history_messages: 20,
            stream_edits: true,
            guilds: Vec::new(),
            dedicated_channels: Vec::new(),
            shared_channels: Vec::new(),
            dm_users: Vec::new(),
            admins: Vec::new(),
            novelty_key: 0,
            notice_ttl_ms: 7_500,
            retrying_ttl_ms: 3_500,
            limits: Limits::default(),
            channel_limits: ChannelLimits::default(),
            retry: RetryRules::default(),
        }
    }
}

impl Config {
    /// Reads `path` if it exists (defaults otherwise), applies env overrides from
    /// `env`, and validates.
    pub fn load(path: &Path, env: &dyn Fn(&str) -> Option<String>) -> Result<Self> {
        let file = match std::fs::read_to_string(path) {
            Ok(text) => rsn::from_str(&text).map_err(|e| {
                let (line, col) = line_col(&text, e.location.start);
                anyhow!(
                    "invalid configuration: {}:{line}:{col}: {}",
                    path.display(),
                    e.kind
                )
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Config::default(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let merged = apply_env(serde_json::to_value(file)?, env)?;
        let config: Config = serde_json::from_value(merged).context("invalid configuration")?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let (r, g, m) = (&self.runtime, &self.generation, &self.model);
        ensure!(r.threads >= 1, "threads must be at least 1");
        ensure!(r.threads <= 32, "threads must be <= 32, got {}", r.threads);
        ensure!(r.queue_depth >= 1, "queue_depth must be at least 1");
        // Interaction tokens expire after 15 min: an /ask must be able to wait
        // behind at least one job that runs to the timeout (design.md §6).
        ensure!(
            2 * g.timeout_secs <= INTERACTION_BUDGET_SECS,
            "2 × timeout_secs must be <= {INTERACTION_BUDGET_SECS} s, got 2 × {}",
            g.timeout_secs
        );
        // Deleting an ephemeral message needs the interaction token (15 min).
        let d = &self.discord;
        ensure!(
            d.notice_ttl_ms <= 600_000 && d.retrying_ttl_ms <= 600_000,
            "notice_ttl_ms and retrying_ttl_ms must be <= 600000 (10 min), got {} and {}",
            d.notice_ttl_ms,
            d.retrying_ttl_ms
        );
        let l = &d.limits;
        ensure!(
            l.backoff_growth >= 1.0 && l.backoff_drain_ms > 0 && l.batch_cost >= 0.0,
            "limits: backoff_growth must be >= 1, backoff_drain_ms > 0 and batch_cost >= 0"
        );
        ensure!(
            l.windows.iter().all(|w| w.count >= 1 && w.ms > 0),
            "limits: every window needs count >= 1 and ms > 0"
        );
        let rr = &d.retry;
        ensure!(
            rr.undo_depth >= 1 && rr.admin_recent >= 1,
            "retry: undo_depth and admin_recent must be at least 1"
        );
        // The delete holds the channel's retry lock while it waits.
        ensure!(
            rr.delete_quiet_ms <= 10_000,
            "retry: delete_quiet_ms must be <= 10000, got {}",
            rr.delete_quiet_ms
        );
        let c = &d.channel_limits;
        ensure!(
            c.backoff_growth >= 1.0 && c.backoff_drain_ms > 0,
            "channel_limits: backoff_growth must be >= 1 and backoff_drain_ms > 0"
        );
        ensure!(
            g.max_message_tokens < g.max_prompt_tokens / 2,
            "max_message_tokens ({}) must be < max_prompt_tokens / 2 ({})",
            g.max_message_tokens,
            g.max_prompt_tokens / 2
        );
        ensure!(
            g.header_limit() + 2 * g.max_message_tokens <= g.max_prompt_tokens,
            "max_header_tokens ({}) + 2 * max_message_tokens ({}) must be <= max_prompt_tokens ({}), so the trigger message fits (it is also copied after the notes)",
            g.header_limit(),
            g.max_message_tokens,
            g.max_prompt_tokens
        );
        if !r.gpus.is_empty() {
            r.gpu_settings(m.n_ctx).context("runtime.gpus")?;
            ensure!(
                r.kv_cache_f16,
                "the GPU backend (runtime.gpus) needs kv_cache_f16: true"
            );
            ensure!(
                m.linear_impl == LinearImpl::Packed,
                "the GPU backend (runtime.gpus) needs linear_impl: Packed"
            );
            ensure!(m.n_ctx <= 8192, "the GPU backend supports n_ctx <= 8192");
            ensure!(
                (1..=200).contains(&r.gpu_poll_hz),
                "gpu_poll_hz must be 1..=200, got {}",
                r.gpu_poll_hz
            );
            ensure!(
                r.gpu_holdoff_secs >= 1,
                "gpu_holdoff_secs must be at least 1"
            );
        }
        ensure!(
            self.discord.history_messages >= 2,
            "history_messages must be at least 2"
        );
        ensure!(
            !self.discord.bot_name.contains([':', '\n', '\r']),
            "bot_name must not contain ':' or line breaks (it starts prompt lines)"
        );
        ensure!(
            !self.discord.bot_name.trim().is_empty(),
            "bot_name must not be empty"
        );
        let d = &self.discord;
        if let Some(c) = d.dedicated_channels.iter().find(|c| listed(&d.shared_channels, c.id)) {
            bail!("channel {c} is in both dedicated_channels and shared_channels");
        }
        let mut names = BTreeMap::new();
        for e in d.entries().filter(|e| !e.name().is_empty()) {
            if let Some(other) = names.insert(e.id(), e) {
                ensure!(
                    other.name() == e.name(),
                    "{other} and {e} are the same ID with different names"
                );
            }
        }
        Ok(())
    }
}

/// 1-based line and column of byte offset `at` in `text`.
pub(crate) fn line_col(text: &str, at: usize) -> (usize, usize) {
    let before = &text[..at.min(text.len())];
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    (
        before.matches('\n').count() + 1,
        before[line_start..].chars().count() + 1,
    )
}

/// Every env key name with its `[section, key]` path in the config. From the
/// defaults, which have every key.
fn env_keys() -> Result<BTreeMap<String, [String; 2]>> {
    let defaults = serde_json::to_value(Config::default())?;
    let mut keys = BTreeMap::new();
    for (section, table) in defaults.as_object().context("config is a struct")? {
        let table = table
            .as_object()
            .context("top-level config values are sections")?;
        for key in table.keys() {
            if let Some([other, _]) = keys.insert(key.clone(), [section.clone(), key.clone()]) {
                bail!("config keys {other}.{key} and {section}.{key} have the same name; env overrides would be ambiguous");
            }
        }
    }
    Ok(keys)
}

/// Overrides keys in `file` (the config as JSON) from `INCREBOT_<KEY>` env vars.
fn apply_env(mut file: Value, env: &dyn Fn(&str) -> Option<String>) -> Result<Value> {
    let defaults = serde_json::to_value(Config::default())?;
    for (key, [section, _]) in env_keys()? {
        let var = format!("INCREBOT_{}", key.to_uppercase());
        let Some(raw) = env(&var) else { continue };
        let value = parse_like(&key, &defaults[&section][&key], &raw).with_context(|| format!("{var}={raw:?}"))?;
        file[&section][&key] = value;
    }
    Ok(file)
}

/// A word list in the `INCREBOT_DISCOURAGE` form:
/// `stem=bias[:min][@after|after/within][^prompt|prompt]`, comma-separated,
/// "_" for a space in a phrase. Empty = no words. Not checked here
/// (`Persona::validate` does).
pub fn parse_discourage(raw: &str) -> Result<Vec<Word>> {
    let list = parse_like("discourage", &Value::Array(Vec::new()), raw)?;
    serde_json::from_value(list).context("discourage")
}

/// Parses an env string for `key` as the same JSON type as `template`.
fn parse_like(key: &str, template: &Value, raw: &str) -> Result<Value> {
    Ok(match template {
        // Enum variants serialize as strings too; their lowercase aliases parse.
        Value::String(_) => Value::from(raw),
        Value::Number(n) if n.is_f64() => {
            Value::from(raw.trim().parse::<f64>().context("expected a number")?)
        }
        // Decimal (signed or up to u64::MAX) or `0x` hex, `_` separators allowed.
        Value::Number(_) => {
            let t = raw.trim().replace('_', "");
            match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                Some(hex) => Value::from(u64::from_str_radix(hex, 16).context("expected a hex integer")?),
                None => match t.parse::<i64>() {
                    Ok(n) => Value::from(n),
                    Err(_) => Value::from(t.parse::<u64>().context("expected an integer")?),
                },
            }
        }
        Value::Bool(_) => Value::Bool(match raw.trim().to_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => bail!("expected a boolean, got {other:?}"),
        }),
        // Comma-separated entries, each `id` or `id=name`: Discord IDs, PCI
        // addresses for `gpus`, or `stem=bias[:min][@after|after/within][^prompt|prompt]`
        // for `discourage` ([`parse_discourage`]). Deserializing then checks
        // the types.
        Value::Array(_) => Value::Array(
            raw.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|item| -> Result<Value> {
                    let (id, name) = item.split_once('=').unwrap_or((item, ""));
                    let (id, name) = (id.trim(), name.trim());
                    let mut entry = serde_json::Map::new();
                    if key == "discourage" {
                        // `stem=bias`, `stem=bias:min`, then optionally
                        // `@play_with|wanna` and `/within`.
                        let expected = || {
                            format!("{item:?}: expected stem=bias[!][:min][@phrase|phrase/within][^phrase|phrase]")
                        };
                        entry.insert("stem".into(), Value::from(id));
                        let phrases = |s: &str| -> Value {
                            Value::Array(s.split('|').map(|t| Value::from(t.trim().replace('_', " "))).collect())
                        };
                        let (name, prompt) = name.split_once('^').map_or((name, None), |(n, p)| (n, Some(p)));
                        if let Some(prompt) = prompt {
                            entry.insert("prompted_by".into(), phrases(prompt));
                        }
                        let (name, cond) = name.split_once('@').map_or((name, None), |(n, c)| (n, Some(c)));
                        if let Some(cond) = cond {
                            let (after, within) =
                                cond.split_once('/').map_or((cond, None), |(a, w)| (a, Some(w)));
                            entry.insert("after".into(), phrases(after));
                            if let Some(w) = within {
                                entry.insert("within".into(), Value::from(w.trim().parse::<u64>().with_context(expected)?));
                            }
                        }
                        let (bias, min) = name.split_once(':').map_or((name, None), |(b, m)| (b, Some(m)));
                        let (bias, whole) = match bias.trim().strip_suffix('!') {
                            Some(b) => (b, true),
                            None => (bias, false),
                        };
                        if whole {
                            entry.insert("whole_word".into(), Value::Bool(true));
                        }
                        let bias = bias.trim().parse::<f64>().with_context(expected)?;
                        entry.insert("bias".into(), Value::from(bias));
                        if let Some(min) = min {
                            entry.insert("min".into(), Value::from(min.trim().parse::<u64>().with_context(expected)?));
                        }
                        return Ok(Value::Object(entry));
                    }
                    if key == "gpus" {
                        entry.insert("pci".into(), Value::from(id));
                    } else {
                        let id = id.parse::<u64>().map(Value::from).unwrap_or_else(|_| Value::from(id));
                        entry.insert("id".into(), id);
                    }
                    if !name.is_empty() {
                        entry.insert("name".into(), Value::from(name));
                    }
                    Ok(Value::Object(entry))
                })
                .collect::<Result<_>>()?,
        ),
        other => bail!("unsupported config type {other:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn discourage_env_whole_word_marker() {
        let w = parse_discourage("cock=-4!, gigg=-2.5:4, cocks=-3!^hotdog").unwrap();
        assert_eq!((w[0].stem.as_str(), w[0].bias, w[0].whole_word), ("cock", -4.0, true));
        assert!(!w[1].whole_word && w[1].min == 4);
        assert!(w[2].whole_word && w[2].prompted_by == ["hotdog"]);
    }

    #[test]
    fn env_integers_take_hex_and_full_u64() {
        let d = Value::from(0u64);
        assert_eq!(parse_like("k", &d, "0x0123_4567_89AB_cdef").unwrap(), Value::from(0x0123_4567_89ab_cdefu64));
        assert_eq!(parse_like("k", &d, "18446744073709551615").unwrap(), Value::from(u64::MAX));
        assert_eq!(parse_like("k", &d, "-3").unwrap(), Value::from(-3));
        assert_eq!(parse_like("k", &d, "1_000").unwrap(), Value::from(1000));
        assert!(parse_like("k", &d, "0xzz").is_err());
        let c = Config::load(Path::new("/nonexistent.rsn"), &|v| {
            (v == "INCREBOT_NOVELTY_KEY").then(|| "0xffff_0000_ffff_0000".to_string())
        })
        .unwrap();
        assert_eq!(c.discord.novelty_key, 0xffff_0000_ffff_0000);
    }

    #[test]
    fn defaults_are_valid_and_keys_unique() {
        Config::default().validate().unwrap();
        let keys = env_keys().unwrap();
        assert_eq!(keys["threads"], ["runtime", "threads"]);
        assert_eq!(keys["persona_file"], ["discord", "persona_file"]);
    }

    #[test]
    fn shipped_config_is_valid() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let c = Config::load(&root.join("config.rsn"), &no_env).unwrap();
        assert_eq!(c.model.linear_impl, LinearImpl::Packed);
        // It spells out the retry rules' defaults.
        let (r, d) = (&c.discord.retry, RetryRules::default());
        assert_eq!(
            (r.window_secs, r.max_newer, r.delete_quiet_ms, r.steer, r.undo_depth, r.admin_recent),
            (d.window_secs, d.max_newer, d.delete_quiet_ms, d.steer, d.undo_depth, d.admin_recent)
        );
        // The persona it names loads and fits the context.
        let p = crate::persona::Persona::load(&root.join(&c.discord.persona_file), &no_env).unwrap();
        p.check_against(&c).unwrap();
    }

    #[test]
    fn env_overrides_by_key_name_without_section() {
        let env = |k: &str| match k {
            "INCREBOT_THREADS" => Some("6".to_string()),
            "INCREBOT_PERSONA_FILE" => Some("p.rsn".to_string()),
            "INCREBOT_GUILDS" => Some("1, 22 = Two,333".to_string()),
            "INCREBOT_PIN_THREADS" => Some("off".to_string()),
            _ => None,
        };
        let c = Config::load(Path::new("/nonexistent/config.rsn"), &env).unwrap();
        assert_eq!(c.runtime.threads, 6);
        assert!(!c.runtime.pin_threads);
        assert_eq!(c.discord.persona_file, "p.rsn");
        let two = Guild { id: 22, name: "Two".into() };
        assert_eq!(c.discord.guilds, vec![1.into(), two, 333.into()]);
        assert_eq!(c.discord.label(22), "Two (22)");
        assert_eq!(c.discord.label(1), "1");
    }

    #[test]
    fn discourage_lists_parse() {
        let d = parse_discourage("gigg=-2.5, Wiggl = -1, seek=-2:4").unwrap();
        let stems: Vec<_> = d.iter().map(|w| (w.stem.as_str(), w.bias, w.min)).collect();
        assert_eq!(stems, [("gigg", -2.5, 3), ("Wiggl", -1.0, 3), ("seek", -2.0, 4)]);
        let d = parse_discourage("me=-3:2@play_with/0, seek=-2.5:4@game|play|hide/6, gigg=-1, tuck=-3:4^tuck_you_in|bedtime")
            .unwrap();
        assert_eq!((d[0].stem.as_str(), d[0].bias, d[0].min), ("me", -3.0, 2));
        assert_eq!((d[0].after.clone(), d[0].within), (vec!["play with".to_string()], 0));
        assert_eq!((d[1].after.clone(), d[1].within), (vec!["game".into(), "play".into(), "hide".into()], 6));
        assert!(d[2].after.is_empty() && d[2].within == 0 && d[2].prompted_by.is_empty());
        assert_eq!(d[3].prompted_by, ["tuck you in", "bedtime"]);
        assert!(parse_discourage("").unwrap().is_empty());
        let err = format!("{:#}", parse_discourage("gigg").unwrap_err());
        assert!(err.contains("stem=bias"), "{err}");
    }

    #[test]
    fn gpu_settings() {
        let env = |k: &str| match k {
            "INCREBOT_GPUS" => Some("0000:06:00.0=upper, 0000:03:00.0".to_string()),
            "INCREBOT_KV_CACHE_F16" => Some("1".to_string()),
            _ => None,
        };
        let c = Config::load(Path::new("/nonexistent/config.rsn"), &env).unwrap();
        let upper = Gpu { pci: "0000:06:00.0".into(), name: "upper".into() };
        assert_eq!(c.runtime.gpus, [upper, "0000:03:00.0".into()]);
        let g = c.runtime.gpu_settings(2048).unwrap();
        assert_eq!(g.gpus[0].pci.to_string(), "0000:06:00.0");
        assert_eq!(g.gpus[0].label, "upper (0000:06:00.0)");
        assert_eq!(g.gpus[1].label, "0000:03:00.0");
        assert_eq!((g.other_mib, g.poll_hz, g.holdoff.as_secs()), (100, 20, 30));
        let bad = |f: &dyn Fn(&mut Config)| {
            let mut c = Config::default();
            c.runtime.gpus = vec!["0000:06:00.0".into()];
            c.runtime.kv_cache_f16 = true;
            f(&mut c);
            format!("{:#}", c.validate().unwrap_err())
        };
        assert!(bad(&|c| c.runtime.kv_cache_f16 = false).contains("kv_cache_f16"));
        let err = bad(&|c| c.runtime.gpus = vec![Gpu { pci: "06:00.0".into(), name: "top".into() }]);
        assert!(err.contains("PCI") && err.contains("top (06:00.0)"), "{err}");
        assert!(bad(&|c| c.model.linear_impl = LinearImpl::Dense).contains("Packed"));
        assert!(bad(&|c| c.runtime.gpu_poll_hz = 0).contains("gpu_poll_hz"));
    }

    #[test]
    fn file_values_then_env_wins() {
        let dir = std::env::temp_dir().join(format!("increbot-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.rsn");
        std::fs::write(
            &path,
            "Config { runtime: Runtime { threads: 4, queue_depth: 1 }, discord: { bot_name: \"Bob\" } }",
        )
        .unwrap();
        let c = Config::load(&path, &no_env).unwrap();
        assert_eq!(
            (
                c.runtime.threads,
                c.runtime.queue_depth,
                c.discord.bot_name.as_str()
            ),
            (4, 1, "Bob")
        );
        let env = |k: &str| (k == "INCREBOT_THREADS").then(|| "2".to_string());
        assert_eq!(Config::load(&path, &env).unwrap().runtime.threads, 2);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn validation_rules() {
        let bad = |f: &dyn Fn(&mut Config)| {
            let mut c = Config::default();
            f(&mut c);
            c.validate().unwrap_err().to_string()
        };
        assert!(bad(&|c| c.runtime.queue_depth = 0).contains("queue_depth"));
        assert!(bad(&|c| c.generation.timeout_secs = 361).contains("720"));
        assert!(bad(&|c| c.generation.max_message_tokens = 600).contains("max_message_tokens"));
        assert!(bad(&|c| c.runtime.threads = 33).contains("32"));
        assert!(bad(&|c| c.discord.bot_name = "flo:ob".into()).contains("bot_name"));
        assert!(bad(&|c| c.discord.bot_name = "a\nb".into()).contains("bot_name"));
        assert!(bad(&|c| c.discord.bot_name = " ".into()).contains("empty"));
        assert!(bad(&|c| c.discord.retry.undo_depth = 0).contains("undo_depth"));
        assert!(bad(&|c| c.discord.retry.admin_recent = 0).contains("admin_recent"));
        assert!(bad(&|c| c.discord.retry.delete_quiet_ms = 10_001).contains("delete_quiet_ms"));
        let mut ok = Config::default();
        ok.runtime.threads = 22;
        ok.validate().unwrap();
    }

    #[test]
    fn retry_rules_from_the_file() {
        let dir = std::env::temp_dir().join(format!("increbot-cfg-retry-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.rsn");
        let load = |text: &str| {
            std::fs::write(&path, text).unwrap();
            Config::load(&path, &no_env).map_err(|e| format!("{e:#}"))
        };
        let c = load("{ discord: { retry: Retry { max_newer: 2, window_secs: 90 } } }").unwrap();
        let r = &c.discord.retry;
        assert_eq!((r.max_newer, r.window_secs, r.undo_depth, r.steer), (2, 90, 10, 3), "the rest default");
        assert_eq!(r.window(), std::time::Duration::from_secs(90));
        assert!(load("{ discord: { retry: { max_newr: 2 } } }").unwrap_err().contains("max_newr"));
        assert!(load("{ discord: { retry: { undo_depth: 0 } } }").unwrap_err().contains("undo_depth"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dedicated_and_shared_channels() {
        let dir = std::env::temp_dir().join(format!("increbot-cfg3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.rsn");
        // The old key is gone: an error instead of silently ignoring channels.
        std::fs::write(&path, "Config { discord: { channels: [1] } }").unwrap();
        assert!(Config::load(&path, &no_env).is_err());
        std::fs::write(
            &path,
            "{ discord: { dedicated_channels: [Channel { id: 1, name: \"#one\" }], \
             shared_channels: [{ id: 2 }, Channel { id: 3 }] } }",
        )
        .unwrap();
        let c = Config::load(&path, &no_env).unwrap();
        let one = Channel { id: 1, name: "#one".into() };
        assert_eq!(
            (c.discord.dedicated_channels, c.discord.shared_channels),
            (vec![one], vec![2.into(), 3.into()])
        );
        let env = |k: &str| (k == "INCREBOT_SHARED_CHANNELS").then(|| "4,5".to_string());
        assert_eq!(
            Config::load(&path, &env).unwrap().discord.shared_channels,
            vec![4.into(), 5.into()]
        );
        let load = |text: &str| {
            std::fs::write(&path, text).unwrap();
            Config::load(&path, &no_env).map_err(|e| format!("{e:#}"))
        };
        let err = load(
            "{ discord: { dedicated_channels: [{ id: 1 }, { id: 2, name: \"#two\" }], \
             shared_channels: [{ id: 2 }] } }",
        )
        .unwrap_err();
        assert!(err.contains("channel #two (2) is in both"), "{err}");
        // The entry type is checked, and bare IDs are no longer entries.
        let err = load("{ discord: { guilds: [Channel { id: 1 }] } }").unwrap_err();
        assert!(err.contains("Guild"), "{err}");
        assert!(load("{ discord: { guilds: [1] } }").is_err());
        // One ID, two names: an error naming both.
        let err = load(
            "{ discord: { dm_users: [User { id: 7, name: \"a\" }], admins: [User { id: 7, name: \"b\" }] } }",
        )
        .unwrap_err();
        assert!(err.contains("a (7) and b (7)"), "{err}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn enums_struct_names_and_error_locations() {
        let dir = std::env::temp_dir().join(format!("increbot-cfg4-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.rsn");
        let load = |text: &str| {
            std::fs::write(&path, text).unwrap();
            Config::load(&path, &no_env).map_err(|e| format!("{e:#}"))
        };
        let c = load(
            "// comment\nConfig {\n    model: Model { linear_impl: Dense, n_ctx: 2_048 },\n}\n",
        )
        .unwrap();
        assert_eq!(c.model.linear_impl, LinearImpl::Dense);
        assert_eq!(c.model.n_ctx, 2048);
        // Struct names are checked when given.
        let err = load("Config { model: Runtime { n_ctx: 2048 } }").unwrap_err();
        assert!(err.contains("Model"), "{err}");
        let err = load("Config {\n  model: { linear_impl: Fast },\n}").unwrap_err();
        assert!(err.contains(":2:"), "{err}");
        // Env overrides take the lowercase names.
        let env = |k: &str| match k {
            "INCREBOT_LINEAR_IMPL" => Some("burn".to_string()),
            _ => None,
        };
        let c = Config::load(&path.with_extension("missing"), &env).unwrap();
        assert_eq!(c.model.linear_impl, LinearImpl::Burn);
        let env = |k: &str| (k == "INCREBOT_LINEAR_IMPL").then(|| "fast".to_string());
        assert!(Config::load(&path.with_extension("missing"), &env).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn unknown_keys_and_bad_env_values_are_errors() {
        let dir = std::env::temp_dir().join(format!("increbot-cfg2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.rsn");
        std::fs::write(&path, "Config { runtime: { thread: 4 } }").unwrap();
        assert!(Config::load(&path, &no_env).is_err());
        std::fs::remove_dir_all(dir).ok();
        let env = |k: &str| (k == "INCREBOT_THREADS").then(|| "many".to_string());
        let err = Config::load(Path::new("/nonexistent"), &env).unwrap_err();
        assert!(format!("{err:#}").contains("INCREBOT_THREADS"), "{err:#}");
    }
}
