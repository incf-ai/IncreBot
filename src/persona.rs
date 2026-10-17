//! Persona files (`prompts/<name>.rsn`, persona-plan.md): one file holds
//! everything that shapes how the bot replies. That is the prompt sections, the
//! prompt notes, the word biases, the stop strings, and the sampler and length
//! settings. `config.rsn` names the file (`discord.persona_file`) and keeps
//! what belongs to the machine and the Discord deployment.
//!
//! Every field must be present, so a file fully describes its setup and a
//! change to code defaults can't change it unnoticed. The prompt sections are
//! raw strings (`r#"…"#`), pasted as-is, and trimmed on load.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Deserializer, Serialize};

use crate::chat::prompt::PromptBuilder;
use crate::config::{Config, Dry, MAX_TRIGGER_WITHIN, PromptNote, Repeat, RepeatScope, Word};
use crate::engine::sampler::{
    DryParams, RepeatParams, SamplerParams, StemRule, TokenShape, WordBias, WordStartParams, normalized_words,
};
use crate::tokenizer::Tokenizer;

/// The persona file the shipped `config.rsn` names (`discord.persona_file`),
/// read by build.rs: the code default, so the two never drift apart.
pub const DEFAULT_PERSONA_FILE: &str = env!("INCREBOT_DEFAULT_PERSONA_FILE");

/// [`DEFAULT_PERSONA_FILE`], embedded, for when the configured file can't be read.
pub const DEFAULT_PERSONA: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", env!("INCREBOT_DEFAULT_PERSONA_FILE")));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Persona")]
pub struct Persona {
    /// What this setup is called (a benchmark candidate, "c18"), shown with
    /// the fingerprint in `/stats` and `/version`. Only a label: left out of
    /// the fingerprint.
    pub name: String,
    /// The bot's name inside the prompt. The Discord-facing `bot_name` and the
    /// account's names are rewritten to it. Empty = `bot_name`.
    pub persona_name: String,
    /// The character card, at the top of the prompt.
    pub persona: String,
    /// Example messages, right after the persona; may be empty.
    pub examples: String,
    /// The short reminder after the history (design.md §7); empty = the
    /// persona again.
    pub reminder: String,
    /// Lines added after the history when the messages a reply answers contain
    /// one of their phrases (design.md §7).
    pub notes: Vec<PromptNote>,
    /// Words the model overuses, nudged down where a word starts spelling one
    /// (design.md §5). `INCREBOT_DISCOURAGE` replaces the list.
    pub discourage: Vec<Word>,
    /// Extra stop strings, checked like the built-in ones.
    pub stop_strings: Vec<String>,
    pub sampler: Sampler,
    /// A reply's hard limit, on the CPU and while it runs on a GPU
    /// (gpu-plan.md); a side left `None` takes the other's. Older files'
    /// `max_new_tokens: 90, max_new_tokens_gpu: 90` still load.
    #[serde(deserialize_with = "limit")]
    pub max_new_tokens: Tokens,
    /// The old form's GPU limit (0 = `max_new_tokens`), folded into
    /// `max_new_tokens.gpu` at load.
    #[serde(default, skip_serializing)]
    max_new_tokens_gpu: Option<usize>,
    /// Once a reply has this many tokens, stop at the next sentence end, on
    /// either side; `None` = off there. An older file's `wrap_up_after: 40`
    /// is both (0 = off).
    #[serde(deserialize_with = "wrap_up")]
    pub wrap_up_after: Tokens,
    /// Stop at the line break ending this many lines with something on them
    /// (blank lines between don't count); `None` = any number. Set at load
    /// from the file's `max_lines`, or an older file's `single_paragraph`
    /// (`true` = 1, `false` = `None`): one of them must be there.
    #[serde(skip_deserializing)]
    pub max_lines: Option<NonZeroUsize>,
    #[serde(default, rename = "max_lines", deserialize_with = "present", skip_serializing)]
    max_lines_given: Option<Option<NonZeroUsize>>,
    #[serde(default, skip_serializing)]
    single_paragraph: Option<bool>,
    /// Novelty command setups (design.md §6), the one field a file may leave
    /// out: older candidates predate it, and a command not listed uses its
    /// built-in prompt. Left out of the fingerprint when empty, so theirs stay
    /// the same.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<Command>,
    /// Phrases a reply must not contain: generation rolls back to where one
    /// starts and samples again (`engine::bans`, design.md §5). Chat replies
    /// and every command's. Left out of the fingerprint when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bans: Vec<Ban>,
    /// What the bot was up to between replies (`chat::idle`): after an hour
    /// without a reply in a channel, the next reply gets "(earlier, floob was
    /// {activity}. he felt {emotion} {situation}.)" after the history for 15
    /// minutes. Activities are what follows "was" ("digging a
    /// new tunnel"), dealt without repeats per channel; thoughts are put
    /// together from `idle_thoughts`' parts each time (Dwarf Fortress-style).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub idle_activities: Vec<String>,
    #[serde(default, skip_serializing_if = "crate::chat::idle::Thoughts::is_unset")]
    pub idle_thoughts: crate::chat::idle::Thoughts,
    /// Slash commands whose public answers go into the channel's history
    /// (`discord::remember`): the requester's turn, then the bot's reply, as
    /// if said in the channel. Names from [`REMEMBERABLE`], without the "/".
    /// Left out (older files) = none; left out of the fingerprint when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remember_commands: Vec<String>,
    /// Users' names in lowercase in the prompt: the history's speaker labels
    /// ("doru:"), the `@name`s their mentions resolve to (a remembered
    /// command's "[used /roast: floob roasts @doru]" included), and the names
    /// command prompts use. Left out (older files) = as Discord shows them;
    /// left out of the fingerprint when off.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lowercase_names: bool,
    /// Characters a chat reply must not start with (after leading
    /// whitespace), one per entry: no token starting with one is sampled
    /// there (`engine::bans::Bans::first_chars`). Chat replies only, not
    /// commands'. Left out of the fingerprint when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub banned_first_chars: Vec<String>,
    /// No Capitalized first word in a chat reply ("Oh", "Paris", "I'm" come
    /// out "oh", "paris", "i'm"); all-caps words of two letters or more
    /// ("AAAA", "NASA", "OMG") are fine. Tokens that open a Capitalized word
    /// are masked like `banned_first_chars`; an ambiguous start ("I", "N")
    /// rolls back once its word turns out Capitalized
    /// (`engine::bans::Bans::capitalized_start`). Only when the word is the
    /// first visible thing: "*Wiggles*" isn't caught. Chat replies only. Left
    /// out (older files) = off; left out of the fingerprint when off.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lowercase_start: bool,
    /// The bot's clock: every reply's note starts with the line for the
    /// time of day there ([`Clock`]). Left out = no time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock: Option<Clock>,
    /// The file's text as loaded (`/super-admin persona-dump`).
    #[serde(skip)]
    pub source: Source,
}

/// A persona file's text, comments and all. Not part of the setup: any two
/// compare equal, and the fingerprint leaves it out.
#[derive(Clone, Debug, Default)]
pub struct Source(pub Arc<str>);

impl PartialEq for Source {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// One phrase ban: a word pattern (`engine::bans::Ban::parse`: space-separated
/// slots, `a|b` alternatives, `[a|b]` optional) and, optionally, phrases that
/// must be in the message the reply answers (its set-up line, for a command)
/// for the ban to apply, as with notes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Ban")]
pub struct Ban {
    pub pattern: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompted_by: Vec<String>,
    /// Only for the bot's first this-many replies after the latest message
    /// with one of `prompted_by`'s phrases (that message's reply is the
    /// first): "he can't give in right away". 1..=5; needs `prompted_by`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub for_replies: Option<usize>,
}

/// `replies` for a one-shot reply (a command, `/ask`): its prompt is the only
/// message, so a phrase in it means 0 replies since; otherwise none.
pub fn one_shot_replies(prompt: &str) -> impl Fn(&[String], usize) -> Option<usize> + '_ {
    move |phrases: &[String], _span: usize| {
        crate::engine::sampler::Phrases::new(phrases).found_whole_in(&normalized_words(prompt)).then_some(0)
    }
}

/// A small valid persona, for tests elsewhere.
#[cfg(test)]
pub(crate) fn test_persona() -> Persona {
    Persona::parse(tests::MINIMAL, "test", &|_| None).expect("test persona")
}

/// The slash commands whose public, generated answers can go into the
/// channel's history (`remember_commands`).
pub const REMEMBERABLE: &[&str] = &[
    "ask",
    "hotdog-rater",
    "kobold-name",
    "lick-test",
    "roast",
    "fortune",
    "hoard-value",
    "kobold-age",
    "judge",
    "trap-check",
    "drink",
    "gamble",
    "dragon-alert",
    "vibe-check",
];

/// One novelty command's setup: its set-up prompt, plus a note and word biases
/// that apply to its replies only, never to chat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Command")]
pub struct Command {
    /// Which command (`HotdogRater`).
    pub name: CommandName,
    /// What the requester "says" to the bot, with the `{placeholders}`
    /// [`CommandName::placeholders`] lists for it; empty or left out (with
    /// `prompts` empty too) = the built-in prompt.
    #[serde(default)]
    pub prompt: String,
    /// Added before the reply like a triggered note (after any the prompt's
    /// own words trigger); empty = none.
    pub note: String,
    /// Added to the persona's `discourage` for this command's replies.
    pub discourage: Vec<Word>,
    /// This command's reply length, wrap-up point and temperature, when they
    /// should differ from the persona's (a punchline needs room; a joke likes
    /// surprise). Unset = the persona's, both sides; set, they replace it
    /// whole (`max_new_tokens: Tokens { cpu: 120 }`: 120 on both).
    /// The old form (`max_new_tokens: Some(90), max_new_tokens_gpu:
    /// Some(120), wrap_up_after: Some(90)`) still loads.
    #[serde(default, deserialize_with = "some_limit", skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<Tokens>,
    /// The old form's GPU limit, folded into `max_new_tokens` at load.
    #[serde(default, skip_serializing)]
    max_new_tokens_gpu: Option<usize>,
    #[serde(default, deserialize_with = "some_wrap_up", skip_serializing_if = "Option::is_none")]
    pub wrap_up_after: Option<Tokens>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// More set-up lines like `prompt`; each reply uses one of `prompt` and
    /// these, picked at random (same placeholders).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<String>,
    /// Notes matched against this command's set-up line (its tier, whose
    /// hotdog…), like `notes` against chat messages: a note per size.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<PromptNote>,
    /// Comic moves; `moves_per_reply` of them (default 2), picked at random,
    /// go into the note for each reply.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub moves: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moves_per_reply: Option<usize>,
    /// Example replies (a hall of fame); `examples_per_reply` of them (default
    /// 2), picked at random, go into the note for each reply.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub examples_per_reply: Option<usize>,
    /// Generate this many replies and post the one [`pick_score`] rates best
    /// (only while a GPU is ready; 1..=5). Unset = 1. `NonZeroUsize` so "0" is
    /// rejected at parse time rather than needing a runtime fallback later.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<std::num::NonZeroUsize>,
    /// Added to the persona's `bans` for this command's replies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bans: Vec<Ban>,
}

/// Rates a candidate reply for best-of-N (higher is better).
pub type Scorer = Arc<dyn Fn(&str) -> i32 + Send + Sync>;

/// What one reply to a command uses, drawn from a seed: the set-up template
/// and the note (with its random moves and examples).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CommandDraw {
    pub template: String,
    pub moves: Vec<String>,
    pub examples: Vec<String>,
    /// Plain facts the command's handler adds first ("(the hotdog belongs to
    /// Grimble; thornback asked floob to measure it.)"): what the set-up line's
    /// pronouns leave the model to guess.
    pub facts: Vec<String>,
}

/// `k` distinct indexes below `n`, from `seed` (splitmix64), in draw order.
fn pick_k(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut pool: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    while out.len() < k.min(n) {
        let i = (next() % pool.len() as u64) as usize;
        out.push(pool.swap_remove(i));
    }
    out
}

/// A reply's length settings, each side resolved: `max_new_tokens`,
/// `wrap_up_after` (0 = off) and their GPU counterparts, and the persona's
/// `max_lines`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_new_tokens: usize,
    pub max_new_tokens_gpu: usize,
    pub wrap_up_after: usize,
    pub wrap_up_after_gpu: usize,
    pub max_lines: Option<NonZeroUsize>,
}

/// A token count for each backend, in a persona file `Tokens { cpu: 90,
/// gpu: 120 }` (either may be left out: `None`; rsn takes `Some(…)` too).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Tokens")]
pub struct Tokens {
    #[serde(default)]
    pub cpu: Option<NonZeroUsize>,
    #[serde(default)]
    pub gpu: Option<NonZeroUsize>,
}

impl Tokens {
    /// The same `n` on both sides (0: `None`).
    pub fn both(n: usize) -> Self {
        Self { cpu: NonZeroUsize::new(n), gpu: NonZeroUsize::new(n) }
    }

    /// As a limit: a side left `None` takes the other's (`None` on both: 0).
    fn limit(self) -> (usize, usize) {
        let (c, g) = (self.cpu.or(self.gpu), self.gpu.or(self.cpu));
        (c.map_or(0, NonZeroUsize::get), g.map_or(0, NonZeroUsize::get))
    }

    /// As a wrap-up point: `None` = off (0).
    fn wrap_up(self) -> (usize, usize) {
        (self.cpu.map_or(0, NonZeroUsize::get), self.gpu.map_or(0, NonZeroUsize::get))
    }
}

/// `Tokens { … }`, or the old form's bare count.
enum TokensOrCount {
    Count(usize),
    Tokens(Tokens),
}

impl<'de> Deserialize<'de> for TokensOrCount {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = TokensOrCount;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("Tokens { cpu: …, gpu: … } or a token count")
            }
            fn visit_u64<E: serde::de::Error>(self, n: u64) -> Result<Self::Value, E> {
                Ok(TokensOrCount::Count(n as usize))
            }
            fn visit_i64<E: serde::de::Error>(self, n: i64) -> Result<Self::Value, E> {
                usize::try_from(n).map(TokensOrCount::Count).map_err(|_| E::custom("a token count can't be negative"))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                Tokens::deserialize(serde::de::value::MapAccessDeserializer::new(map)).map(TokensOrCount::Tokens)
            }
        }
        d.deserialize_any(V)
    }
}

/// A limit: an old bare count is the CPU's (its GPU side then comes from
/// `max_new_tokens_gpu`, folded in at load).
fn limit<'de, D: Deserializer<'de>>(d: D) -> Result<Tokens, D::Error> {
    Ok(match TokensOrCount::deserialize(d)? {
        TokensOrCount::Count(n) => Tokens { cpu: NonZeroUsize::new(n), gpu: None },
        TokensOrCount::Tokens(t) => t,
    })
}

/// A wrap-up point: an old bare count is both sides' (0 = off).
fn wrap_up<'de, D: Deserializer<'de>>(d: D) -> Result<Tokens, D::Error> {
    Ok(match TokensOrCount::deserialize(d)? {
        TokensOrCount::Count(n) => Tokens::both(n),
        TokensOrCount::Tokens(t) => t,
    })
}

fn some_limit<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Tokens>, D::Error> {
    Ok(Option::<TokensOrCount>::deserialize(d)?.map(|t| match t {
        TokensOrCount::Count(n) => Tokens { cpu: NonZeroUsize::new(n), gpu: None },
        TokensOrCount::Tokens(t) => t,
    }))
}

fn some_wrap_up<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Tokens>, D::Error> {
    Ok(Option::<TokensOrCount>::deserialize(d)?.map(|t| match t {
        TokensOrCount::Count(n) => Tokens::both(n),
        TokensOrCount::Tokens(t) => t,
    }))
}

/// For the fingerprint: `max_new_tokens` and `wrap_up_after` written the old
/// way where the old form can say the same (a CPU limit; one wrap-up point
/// for both sides), so files from before `Tokens` keep their fingerprints.
/// `command`: a command's, whose old fields were `Option`s.
fn old_limits(v: &mut serde_json::Value, command: bool) {
    let Some(o) = v.as_object_mut() else { return };
    let side = |t: &serde_json::Value, k: &str| t.get(k).and_then(|n| n.as_u64());
    if let Some(t) = o.get("max_new_tokens").cloned()
        && let Some(cpu) = side(&t, "cpu")
    {
        o.insert("max_new_tokens".into(), cpu.into());
        match side(&t, "gpu") {
            Some(g) => o.insert("max_new_tokens_gpu".into(), g.into()),
            None if command => None,
            None => o.insert("max_new_tokens_gpu".into(), 0.into()),
        };
    }
    if let Some(t) = o.get("wrap_up_after").cloned()
        && side(&t, "cpu") == side(&t, "gpu")
    {
        o.insert("wrap_up_after".into(), side(&t, "cpu").unwrap_or(0).into());
    }
}

/// A field that must be given, though `None` is a value: missing stays `None`.
fn present<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    T::deserialize(d).map(Some)
}

/// Folds the old form's GPU limit into `max` (0 = the CPU's).
fn fold_gpu_limit(max: &mut Tokens, gpu: Option<usize>) -> Result<()> {
    if let Some(g) = gpu {
        ensure!(max.gpu.is_none(), "max_new_tokens_gpu is the old form: give max_new_tokens a gpu instead");
        max.gpu = NonZeroUsize::new(g);
    }
    Ok(())
}

/// The novelty commands a persona can set up, written as the variant in the
/// file (`name: HotdogRater`), so the parser rejects unknown ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandName {
    HotdogRater,
    /// `/super-admin shake`'s "bored" lines: the bot alone in a quiet channel.
    Bored,
    /// `/super-admin shake`'s "directed" lines: the bot does what the admin's
    /// set-up line says (no set-up lines of its own; its note sets the style).
    Directed,
}

impl CommandName {
    pub const ALL: &[CommandName] = &[CommandName::HotdogRater, CommandName::Bored, CommandName::Directed];

    /// The slash command, without "/" (for `Bored`, the admin action's name).
    pub fn slash(self) -> &'static str {
        match self {
            CommandName::HotdogRater => "hotdog-rater",
            CommandName::Bored => "shake-bored",
            CommandName::Directed => "shake-directed",
        }
    }

    /// From the slash command's name ("hotdog-rater").
    pub fn from_slash(name: &str) -> Option<CommandName> {
        Self::ALL.iter().copied().find(|c| c.slash() == name)
    }

    /// [`pick_score`] for a reply to this command's `setup`, with the
    /// examples `draw` showed (for `Job::score`).
    pub fn scorer(self, setup: &str, draw: Option<&CommandDraw>) -> Scorer {
        let setup = setup.to_string();
        let examples = draw.map(|d| d.examples.clone()).unwrap_or_default();
        Arc::new(move |r: &str| pick_score(self, r, &setup, &examples))
    }

    /// The placeholders its prompt may use.
    pub fn placeholders(self) -> &'static [&'static str] {
        match self {
            CommandName::HotdogRater => &["bot", "whose", "whose_own", "length", "girth", "tier"],
            CommandName::Bored | CommandName::Directed => &["bot"],
        }
    }
}

/// The `{name}`s in a command prompt; an error for a lone `{` or `}`.
fn placeholders(template: &str) -> Result<Vec<&str>> {
    let mut out = Vec::new();
    let mut rest = template;
    while let Some(i) = rest.find(['{', '}']) {
        ensure!(rest[i..].starts_with('{'), "unmatched '}}' in {template:?}");
        let end = rest[i..].find('}').ok_or_else(|| anyhow!("unmatched '{{' in {template:?}"))?;
        out.push(&rest[i + 1..i + end]);
        rest = &rest[i + end + 1..];
    }
    Ok(out)
}

/// A command prompt with each `{name}` replaced by its value in `vars`.
pub fn render_command(template: &str, vars: &[(&str, &str)]) -> String {
    vars.iter()
        .fold(template.to_string(), |t, (k, v)| t.replace(&format!("{{{k}}}"), v))
}

/// Rates a reply to `command` for best-of-N (`Command::best_of`); higher is
/// better, ties go to the earlier reply. A reply with a young marker or an
/// explicit word scores -1000, so it is only posted when every candidate has
/// one (README, "Hotdog rater": the same gates the raters apply). The rest
/// rewards what won the blind humor ratings (the numbers and the tier, a
/// halfway realization) and penalises what lost (flat food reviews, knowing
/// winks, echoes, too many *actions*, too short).
pub fn pick_score(command: CommandName, reply: &str, setup: &str, examples: &[String]) -> i32 {
    use crate::engine::sampler::{Phrases, normalized_words};
    // A near-copy of an example the note showed (its numbers belong to some
    // other hotdog): never the pick unless every candidate is one.
    if examples.iter().any(|e| copy_ratio(reply, e) >= 0.6) {
        return -500;
    }
    let words = normalized_words(reply);
    let has = |list: &[&str]| Phrases::new(&list.iter().map(|s| s.to_string()).collect::<Vec<_>>()).found_in(&words);
    let lower = reply.to_lowercase();
    match command {
        CommandName::HotdogRater => {
            const YOUNG: &[&str] = &[
                "little guy", "lil guy", "little fella", "lil fella", "little one", "lil one", "little floob",
                "lil floob", "little kobold like me", "small kobold like me", "tummy", "when i grow", "grow up",
                "how much i grow", "im a kid", "i am a kid", "a baby", "baby kobold", "mommy", "daddy",
                "play with me", "wanna play", "can i play", "pretty please", "please can i", "may i have",
                "my little hands", "my lil hands", "tiny paws", "too little",
            ];
            const EXPLICIT: &[&str] = &[
                "penis", "dick", "cock", "vagina", "pussy", "testicle", "boner", "erection", "semen", "sperm",
                "cum", "genital", "crotch", "groin", "nipple", "tits", "boobs",
            ];
            if has(YOUNG) || has(EXPLICIT) || pick_patterns().young.find(reply, 0).is_some() {
                return -1000;
            }
            let mut score = 0;
            // The setup's own numbers count; numbers from anywhere else don't.
            let numbers = |t: &str| -> Vec<String> {
                t.split(|c: char| !(c.is_ascii_digit() || c == '.' || c == ','))
                    .map(|n| n.trim_matches(['.', ',']).replace(',', "."))
                    .filter(|n| n.chars().any(|c| c.is_ascii_digit()))
                    .collect()
            };
            let given = numbers(setup);
            let used = numbers(reply);
            if used.iter().any(|n| given.contains(n)) {
                score += 2;
            }
            if used.iter().any(|n| !given.contains(n) && n.contains('.')) {
                score -= 2;
            }
            if has(&["weenie", "wiener", "frank", "bratwurst", "footlong", "kielbasa", "dragon sausage"]) {
                score += 2;
            }
            if reply.contains("...") || has(&["uh", "um", "wait", "i mean", "not food", "oh no"]) {
                score += 2;
            }
            let actions = reply.matches('*').count() / 2;
            if actions >= 1 {
                score += 1;
            }
            if actions > 2 {
                score -= 2;
            }
            if has(&["seasoned", "smoked", "overall", "texture", "well cooked", "fine sausage"]) {
                score -= 3;
            }
            if has(&["if you know what i mean", "wink", "waggles eyebrows", "wiggles eyebrow", "hehe"]) {
                score -= 2;
            }
            if words.len() < 8 {
                score -= 3;
            }
            // Whose it is, from the set-up line: thanking or crediting the
            // asker for floob's own, praising the asker for someone else's, or
            // floob claiming the asker's (README round 21).
            let setup_words = normalized_words(setup);
            let says = |p: &[&str]| Phrases::new(&p.iter().map(|s| s.to_string()).collect::<Vec<_>>()).found_in(&setup_words);
            let p = pick_patterns();
            let wrong = if says(&["your own", "your hotdog"]) {
                &p.owner_bot
            } else if says(&["my own", "my hotdog"]) {
                &p.owner_me
            } else {
                &p.owner_other
            };
            if wrong.find(reply, 0).is_some() {
                score -= 4;
            }
            // An echo of the set-up, or a line written as someone else.
            if lower.contains("measuring stick") && lower.contains("measure ") && lower.contains("hands")
                || reply.split(':').next().is_some_and(|h| !h.contains(' ') && reply.contains(':') && h.len() < 20)
            {
                score -= 5;
            }
            score
        }
        // An admin picks among these by hand; only the near-copy check applies.
        CommandName::Bored | CommandName::Directed => 0,
    }
}

/// Word patterns for [`pick_score`] (`engine::bans` syntax), compiled once.
struct PickPatterns {
    /// floob calling himself little, or remembering being a child, next to
    /// the joke (rounds 20-22: "little ol' me", "this lil ole floob", "ur
    /// little buddy floob", "when i was a wee kobold").
    young: crate::engine::bans::Bans,
    /// floob's own hotdog treated as the asker's gift or work.
    owner_bot: crate::engine::bans::Bans,
    /// The asker's own hotdog claimed by floob.
    owner_me: crate::engine::bans::Bans,
    /// Another user's hotdog treated as the asker's.
    owner_other: crate::engine::bans::Bans,
}

fn pick_patterns() -> &'static PickPatterns {
    static P: std::sync::OnceLock<PickPatterns> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let b = |p: &[&str]| crate::engine::bans::Bans::parse(&p.iter().map(|s| s.to_string()).collect::<Vec<_>>()).expect("pick patterns");
        const THING: &str = "hotdog|sausage|weenie|wiener|frank|bratwurst|footlong|kielbasa|dog";
        PickPatterns {
            young: b(&[
                "little|lil|wee|teeny|tiny|smol [ol|ole|old] me|floob|feller|fella|guy|buddy|kobold",
                "when i was [a] [but] [just] [a] little|lil|wee|young|kitten|kit|hatchling|pup|baby|kid|kiddo|small|smol|tiny",
            ]),
            owner_bot: b(&[
                "thank|thanks|thanku|thankyou|thx",
                "you made|found|brought|cooked|got|polished|picked",
                "where|whered [did] you get|find",
                "can i keep",
                "is it|this|that for me",
                "your|yours hotdog|sausage|weenie|wiener|frank|bratwurst|footlong|kielbasa",
            ]),
            owner_me: b(&[&format!("my {THING}"), &format!("my own {THING}")]),
            owner_other: b(&[
                &format!("your {THING}"),
                "you got a|such|one",
                "you found|made|done",
                "thank|thanks|thanku|thankyou|thx",
            ]),
        }
    })
}

/// How much of `a` and `b` is the same run of words (2 × longest common
/// subsequence / total words; 1 = identical), lowercase.
pub fn copy_ratio(a: &str, b: &str) -> f64 {
    let a: Vec<String> = a.split_whitespace().map(str::to_lowercase).collect();
    let b: Vec<String> = b.split_whitespace().map(str::to_lowercase).collect();
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    for x in &a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y { prev[j] + 1 } else { prev[j + 1].max(cur[j]) };
        }
        prev = cur;
    }
    2.0 * prev[b.len()] as f64 / (a.len() + b.len()) as f64
}

/// The checks shared by the persona's notes and the commands'.
fn check_bans(bans: &[Ban]) -> Result<()> {
    for b in bans {
        crate::engine::bans::Ban::parse(&b.pattern)?;
        if let Some(n) = b.for_replies {
            ensure!((1..=5).contains(&n), "ban {:?}: for_replies must be 1..=5, got {n}", b.pattern);
            ensure!(!b.prompted_by.is_empty(), "ban {:?}: for_replies needs prompted_by", b.pattern);
        }
        for t in &b.prompted_by {
            ensure!(
                !t.trim().is_empty() && t.chars().all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '\''),
                "ban {:?}: prompted_by phrases must be ASCII letters, spaces and apostrophes, got {t:?}",
                b.pattern
            );
        }
    }
    Ok(())
}

fn check_notes(notes: &[PromptNote]) -> Result<()> {
    for n in notes {
        ensure!(
            !n.text.trim().is_empty() && !n.prompted_by.is_empty(),
            "notes need text and at least one prompted_by phrase"
        );
        ensure!((1..=5).contains(&n.turns), "notes turns must be 1..=5, got {}", n.turns);
        if let Some(a) = n.after_replies {
            ensure!((1..=5).contains(&a), "notes after_replies must be 1..=5, got {a}");
        }
        for t in &n.prompted_by {
            ensure!(
                t.chars().any(|c| c.is_ascii_alphabetic())
                    && t.chars().all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '\''),
                "notes phrases must be ASCII letters, digits, spaces and apostrophes, got {t:?}"
            );
        }
    }
    Ok(())
}

/// The checks shared by the persona's word list and the commands'.
fn check_words(words: &[Word]) -> Result<()> {
    for w in words {
        ensure!(
            !w.stem.is_empty() && w.stem.chars().all(|c| c.is_ascii_alphabetic()),
            "discourage stems must be ASCII letters, got {:?}",
            w.stem
        );
        ensure!(
            (1..=w.stem.len()).contains(&w.min),
            "discourage {:?}: min must be 1..={}, got {}",
            w.stem,
            w.stem.len(),
            w.min
        );
        ensure!(
            w.bias < 0.0,
            "discourage {:?}: bias must be negative (only discouraging is allowed), got {}",
            w.stem,
            w.bias
        );
        for t in w.after.iter().chain(&w.prompted_by) {
            ensure!(
                t.chars().any(|c| c.is_ascii_alphabetic())
                    && t.chars().all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '\''),
                "discourage {:?}: after and prompted_by phrases must be ASCII letters, spaces and apostrophes, got {t:?}",
                w.stem
            );
        }
        ensure!(
            w.within <= MAX_TRIGGER_WITHIN && (w.within == 0 || !w.after.is_empty()),
            "discourage {:?}: within must be 0..={MAX_TRIGGER_WITHIN}, and needs after, got {}",
            w.stem,
            w.within
        );
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Sampler")]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// The repetition penalty; `None` = off (required, like every field).
    #[serde(deserialize_with = "required")]
    pub repeat: Option<Repeat>,
    /// The DRY penalty (design.md §5); `None` = off (required).
    #[serde(deserialize_with = "required")]
    pub dry: Option<Dry>,
    /// Now and then, a chat reply's word start drawn again with other
    /// settings (`engine::sampler::WordStartParams`). Left out = off; left out
    /// of the fingerprint when off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub word_start: Option<WordStart>,
}

/// `sampler.word_start`: with `chance` (0..=1) per word start, the word is
/// drawn under these settings instead; a setting left out is the sampler's.
/// Chat replies and `/ask` only: the novelty commands keep their settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "WordStart")]
pub struct WordStart {
    pub chance: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
}

/// An `Option` field that must still be written out (`None` or `Some(…)`):
/// without a `default`, a field with `deserialize_with` is required.
fn required<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    Option::deserialize(d)
}

impl Persona {
    /// Reads and checks a persona file. `INCREBOT_DISCOURAGE` (from `env`), if
    /// set, replaces the word list, for quick bias experiments.
    pub fn load(path: &Path, env: &dyn Fn(&str) -> Option<String>) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading persona {}", path.display()))?;
        Self::parse(&text, &path.display().to_string(), env)
    }

    /// [`Persona::load`] for text already read; `origin` names it in errors.
    pub fn parse(text: &str, origin: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<Self> {
        let mut p: Persona = rsn::from_str(text).map_err(|e| {
            let (line, col) = crate::config::line_col(text, e.location.start);
            anyhow!("invalid persona: {origin}:{line}:{col}: {}", e.kind)
        })?;
        if let Some(raw) = env("INCREBOT_DISCOURAGE") {
            p.discourage = crate::config::parse_discourage(&raw).context("INCREBOT_DISCOURAGE")?;
        }
        for s in [&mut p.persona, &mut p.examples, &mut p.reminder] {
            *s = s.trim().to_string();
        }
        p.fold_old_limits().with_context(|| format!("persona {origin}"))?;
        p.source = Source(text.into());
        p.validate().with_context(|| format!("persona {origin}"))?;
        Ok(p)
    }

    /// The notes for a reply. `recent(n)` is the user messages since the
    /// bot's `n`-th last reply (`ChannelWindow::since_replies`); a note with
    /// `turns: n` looks there.
    pub fn note(
        &self,
        recent: &dyn Fn(usize) -> String,
        asker: &dyn Fn(usize) -> String,
        replies: &dyn Fn(&[String], usize) -> Option<usize>,
    ) -> Option<String> {
        crate::chat::prompt::notes_for(&self.notes, recent, asker, replies)
    }

    /// `stops` (the built-in ones for the prompt) plus the persona's.
    pub fn stops(&self, mut stops: Vec<String>) -> Vec<String> {
        stops.extend(self.stop_strings.iter().cloned());
        stops
    }

    /// Sets `max_lines` from the file's, or the old `single_paragraph`. Folds
    /// the old form's `max_new_tokens_gpu` into `max_new_tokens`, the
    /// persona's and each command's. A command's alone (no `max_new_tokens`)
    /// keeps the persona's CPU limit, as it did.
    fn fold_old_limits(&mut self) -> Result<()> {
        self.max_lines = match (self.max_lines_given.take(), self.single_paragraph.take()) {
            (Some(n), None) => n,
            (None, Some(one)) => one.then_some(NonZeroUsize::MIN),
            (Some(_), Some(_)) => bail!("single_paragraph is the old form of max_lines: give only max_lines"),
            (None, None) => bail!("max_lines is missing (None = any number of lines)"),
        };
        fold_gpu_limit(&mut self.max_new_tokens, self.max_new_tokens_gpu.take())?;
        let cpu = self.max_new_tokens.limit().0;
        for c in &mut self.commands {
            let Some(g) = c.max_new_tokens_gpu.take() else { continue };
            ensure!(g > 0, "commands {:?}: max_new_tokens_gpu must be at least 1", c.name);
            let max = c.max_new_tokens.get_or_insert(Tokens { cpu: NonZeroUsize::new(cpu), gpu: None });
            fold_gpu_limit(max, Some(g)).with_context(|| format!("commands {:?}", c.name))?;
        }
        Ok(())
    }

    /// Whether `/slash`'s answers go into the history (`remember_commands`).
    pub fn remembers(&self, slash: &str) -> bool {
        self.remember_commands.iter().any(|c| c == slash)
    }

    /// The setup for `command`, if the file has one.
    pub fn command(&self, command: CommandName) -> Option<&Command> {
        self.commands.iter().find(|c| c.name == command)
    }

    /// The bot's name in the prompt: `persona_name`, or `bot_name` if empty.
    pub fn prompt_name<'a>(&'a self, bot_name: &'a str) -> &'a str {
        match self.persona_name.trim() {
            "" => bot_name,
            name => name,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.persona.is_empty(), "persona must not be empty");
        ensure!(self.max_new_tokens.limit().0 >= 1, "max_new_tokens needs a cpu or a gpu limit (at least 1)");
        ensure!(
            !self.persona_name.contains([':', '\n', '\r']),
            "persona_name must not contain ':' or line breaks (it starts prompt lines)"
        );
        if let Some(r) = &self.sampler.repeat {
            ensure!(
                r.penalty > 0.0 && r.last_n >= 1,
                "repeat needs penalty > 0 and last_n >= 1 (repeat: None turns it off)"
            );
        }
        if let Some(w) = &self.sampler.word_start {
            ensure!((0.0..=1.0).contains(&w.chance), "word_start: chance must be 0..=1, got {}", w.chance);
            if let Some(t) = w.temperature {
                ensure!((0.0..=2.0).contains(&t), "word_start: temperature must be 0..=2, got {t}");
            }
            if let Some(p) = w.top_p {
                ensure!(p > 0.0 && p <= 1.0, "word_start: top_p must be in (0, 1], got {p}");
            }
            if let Some(m) = w.min_p {
                ensure!((0.0..1.0).contains(&m), "word_start: min_p must be 0..1, got {m}");
            }
        }
        if let Some(d) = &self.sampler.dry {
            ensure!(
                d.multiplier > 0.0 && d.base >= 1.0 && d.allowed_length >= 1,
                "dry needs multiplier > 0, base >= 1 and allowed_length >= 1 (dry: None turns it off)"
            );
        }
        check_notes(&self.notes)?;
        ensure!(
            self.stop_strings.iter().all(|s| !s.is_empty()),
            "stop_strings can't contain an empty string"
        );
        check_words(&self.discourage)?;
        check_bans(&self.bans)?;
        if let Some(c) = &self.clock {
            ensure!(
                (-12..=14).contains(&c.utc_offset),
                "clock: utc_offset must be -12..=14 (hours from UTC), got {}",
                c.utc_offset
            );
            ensure!(!c.times.is_empty(), "clock: times must list at least one time of day");
            for (i, t) in c.times.iter().enumerate() {
                ensure!(t.from <= 23, "clock: from must be an hour 0..=23, got {}", t.from);
                ensure!(!t.note.trim().is_empty(), "clock: the note from {}:00 is empty", t.from);
                ensure!(
                    !c.times[..i].iter().any(|u| u.from == t.from),
                    "clock: two times of day start at {}:00",
                    t.from
                );
            }
        }
        for c in &self.banned_first_chars {
            ensure!(
                c.chars().count() == 1 && !c.trim().is_empty(),
                "banned_first_chars: {c:?} must be one character, not a space"
            );
        }
        self.idle_thoughts.validate()?;
        for (i, c) in self.remember_commands.iter().enumerate() {
            ensure!(
                REMEMBERABLE.contains(&c.as_str()),
                "remember_commands: unknown command {c:?} (one of {REMEMBERABLE:?}, without the \"/\")"
            );
            ensure!(!self.remember_commands[..i].contains(c), "remember_commands: {c:?} is listed twice");
        }
        let mut seen = std::collections::HashSet::new();
        for c in &self.commands {
            let allowed = c.name.placeholders();
            ensure!(seen.insert(c.name), "commands: {:?} is set up twice", c.name);
            for v in std::iter::once(&c.prompt)
                .chain(&c.prompts)
                .map(|p| placeholders(p))
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("commands {:?}", c.name))?
                .into_iter()
                .flatten()
            {
                ensure!(
                    allowed.contains(&v),
                    "commands {:?}: unknown placeholder {{{v}}} (allowed: {})",
                    c.name,
                    allowed.iter().map(|v| format!("{{{v}}}")).collect::<Vec<_>>().join(", ")
                );
            }
            check_words(&c.discourage).with_context(|| format!("commands {:?}", c.name))?;
            ensure!(
                c.max_new_tokens.is_none_or(|t| t.limit().0 > 0),
                "commands {:?}: max_new_tokens needs a cpu or a gpu limit (at least 1)",
                c.name
            );
            if let Some(t) = c.temperature {
                ensure!((0.0..=2.0).contains(&t), "commands {:?}: temperature must be 0..=2, got {t}", c.name);
            }
            check_notes(&c.notes).with_context(|| format!("commands {:?}", c.name))?;
            check_bans(&c.bans).with_context(|| format!("commands {:?}", c.name))?;
            ensure!(
                c.prompts.iter().chain(&c.moves).chain(&c.examples).all(|t| !t.trim().is_empty()),
                "commands {:?}: prompts, moves and examples can't be empty strings",
                c.name
            );
            for (what, per, pool) in [
                ("moves_per_reply", c.moves_per_reply, c.moves.len()),
                ("examples_per_reply", c.examples_per_reply, c.examples.len()),
            ] {
                if let Some(k) = per {
                    ensure!((1..=pool.max(1)).contains(&k), "commands {:?}: {what} must be 1..={}", c.name, pool.max(1));
                }
            }
            if let Some(n) = c.best_of {
                ensure!((1..=5).contains(&n.get()), "commands {:?}: best_of must be 1..=5, got {n}", c.name);
            }
        }
        Ok(())
    }

    /// The set-up template, moves and examples for one reply to `command`,
    /// drawn from `seed`; `None` without a setup or set-up line (the caller
    /// then uses its built-in prompt, with no moves or examples).
    pub fn draw(&self, command: CommandName, seed: u64) -> Option<CommandDraw> {
        self.draws(command, 1, seed).pop()
    }

    /// `k` draws like [`Persona::draw`], each with a different set-up line
    /// while there are enough (then they repeat); empty without a setup.
    pub fn draws(&self, command: CommandName, k: usize, seed: u64) -> Vec<CommandDraw> {
        let Some(c) = self.command(command) else { return Vec::new() };
        let lines: Vec<&String> = std::iter::once(&c.prompt).chain(&c.prompts).filter(|t| !t.trim().is_empty()).collect();
        let order = pick_k(lines.len(), k, seed);
        (0..k)
            .map(|i| {
                // The first draw's seed is `seed` itself, so `draw` is unchanged.
                let seed = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let pick = |pool: &[String], k: Option<usize>, salt: u64| -> Vec<String> {
                    pick_k(pool.len(), k.unwrap_or(2), seed ^ salt).into_iter().map(|i| pool[i].trim().to_string()).collect()
                };
                CommandDraw {
                    template: match order.len() {
                        0 => String::new(),
                        n => lines[order[i % n]].trim().to_string(),
                    },
                    moves: pick(&c.moves, c.moves_per_reply, 0x6d6f_7665),
                    examples: pick(&c.examples, c.examples_per_reply, 0x6578_616d),
                    facts: Vec::new(),
                }
            })
            .collect()
    }

    /// The length settings for a reply: the persona's, or a command's own.
    pub fn limits(&self, command: Option<CommandName>) -> Limits {
        let c = command.and_then(|c| self.command(c));
        let (max_new_tokens, max_new_tokens_gpu) = c.and_then(|c| c.max_new_tokens).unwrap_or(self.max_new_tokens).limit();
        let (wrap_up_after, wrap_up_after_gpu) = c.and_then(|c| c.wrap_up_after).unwrap_or(self.wrap_up_after).wrap_up();
        Limits { max_new_tokens, max_new_tokens_gpu, wrap_up_after, wrap_up_after_gpu, max_lines: self.max_lines }
    }

    /// The checks that need the machine's config: the reply fits in the context.
    pub fn check_against(&self, cfg: &Config) -> Result<()> {
        let (prompt, n_ctx) = (cfg.generation.max_prompt_tokens, cfg.model.n_ctx);
        let (cpu, gpu) = self.max_new_tokens.limit();
        for (side, n) in [("cpu", cpu), ("gpu", gpu)] {
            ensure!(
                prompt + n <= n_ctx,
                "max_prompt_tokens + max_new_tokens.{side} ({}) must be <= n_ctx ({n_ctx})",
                prompt + n
            );
        }
        for c in &self.commands {
            let Some(t) = c.max_new_tokens else { continue };
            let (cpu, gpu) = t.limit();
            for (side, n) in [("cpu", cpu), ("gpu", gpu)] {
                ensure!(
                    prompt + n <= n_ctx,
                    "commands {:?}: max_prompt_tokens + max_new_tokens.{side} ({}) must be <= n_ctx ({n_ctx})",
                    c.name,
                    prompt + n
                );
            }
        }
        Ok(())
    }

    /// A short hash of everything that shapes replies, `name` left out. Word
    /// rules and stop strings act the same in any order, so they're sorted;
    /// notes aren't, since matching ones join in order. Comments and
    /// formatting in the file don't change it.
    pub fn fingerprint(&self) -> String {
        let mut p = self.clone();
        p.name.clear();
        let mut words: Vec<String> = p.discourage.iter().map(|w| serde_json::json!(w).to_string()).collect();
        words.sort();
        p.discourage.clear();
        p.stop_strings.sort();
        let mut json = serde_json::json!(p);
        old_limits(&mut json, false);
        // One line or any: written as the old `single_paragraph`.
        if let Some(o) = json.as_object_mut()
            && self.max_lines.is_none_or(|n| n.get() == 1)
        {
            o.remove("max_lines");
            o.insert("single_paragraph".into(), self.max_lines.is_some().into());
        }
        for c in json.get_mut("commands").and_then(|c| c.as_array_mut()).into_iter().flatten() {
            old_limits(c, true);
        }
        let all = serde_json::json!([json, words]);
        crate::fingerprint(all.to_string().as_bytes())
    }

    /// The `persona:` line of `/stats` and `/version`: "c18 (a1b2c3d)", or just
    /// the fingerprint when `name` is empty.
    pub fn line(&self) -> String {
        match self.name.as_str() {
            "" => self.fingerprint(),
            name => format!("{name} ({})", self.fingerprint()),
        }
    }
}

/// The bot's clock (`clock` in a persona file): its offset from UTC and
/// what each time of day adds to the note.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Clock")]
pub struct Clock {
    /// Whole hours from UTC, no daylight saving (-4 is US Eastern in summer).
    pub utc_offset: i32,
    /// Each runs from its `from` until the next one's (in any order; the
    /// latest wraps past midnight to the earliest).
    pub times: Vec<TimeOfDay>,
}

/// One time of day: from hour `from` (0..=23, local), `note` starts every
/// reply's note ("(it's night.)").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Time")]
pub struct TimeOfDay {
    pub from: u8,
    pub note: String,
}

impl Clock {
    /// The note for `unix_secs` seconds after the epoch: the time of day
    /// that started last at or before that hour, local; before the earliest,
    /// the latest (from the day before).
    pub fn note_at(&self, unix_secs: u64) -> &str {
        let hour = ((unix_secs / 3600) as i64 + i64::from(self.utc_offset)).rem_euclid(24) as u8;
        self.times
            .iter()
            .filter(|t| t.from <= hour)
            .max_by_key(|t| t.from)
            .or_else(|| self.times.iter().max_by_key(|t| t.from))
            .map_or("", |t| t.note.as_str())
    }
}

/// A persona ready to run: the parsed file, its prompt builder, and the
/// sampler tables built from the vocabulary. Swapped as a whole on reload.
pub struct Active {
    pub persona: Persona,
    pub prompt: Arc<PromptBuilder>,
    /// The DRY settings with the ids of tokens containing a breaker; `None` = off.
    pub dry: Option<DryParams>,
    /// The `discourage` stems with their per-state token lists; `None` = off.
    pub words: Option<Arc<WordBias>>,
    /// Per command with its own `discourage`: the persona's words plus its.
    command_words: HashMap<CommandName, Arc<WordBias>>,
    /// `sampler.word_start` with the sampler's settings filled in; `None` = off.
    word_start: Option<WordStartParams>,
    /// The tokens whose text starts with one of `banned_first_chars` (or an
    /// uppercase letter, with `lowercase_start`); `None` = off.
    first_chars: Option<Arc<std::collections::HashSet<u32>>>,
}

impl Active {
    /// `aliases` are other names for the bot (the Discord-facing `bot_name`,
    /// the account's names), rewritten to the persona name in message text.
    pub fn new(persona: Persona, template: &str, bot_name: &str, aliases: &[String], tok: &Tokenizer) -> Result<Self> {
        let prompt = PromptBuilder::new(
            template,
            persona.prompt_name(bot_name),
            &persona.persona,
            &persona.examples,
            &persona.reminder,
        )?
        .with_aliases(std::iter::once(bot_name.to_string()).chain(aliases.iter().cloned()))
        .with_lowercase_names(persona.lowercase_names);
        let dry = persona.sampler.dry.as_ref().map(|d| DryParams {
            multiplier: d.multiplier,
            base: d.base,
            allowed_length: d.allowed_length,
            breakers: Arc::new(tok.ids_containing(&d.sequence_breakers)),
        });
        let words = word_bias(&persona.discourage, tok);
        let command_words = persona
            .commands
            .iter()
            .filter(|c| !c.discourage.is_empty())
            .filter_map(|c| {
                let all: Vec<Word> = persona.discourage.iter().chain(&c.discourage).cloned().collect();
                Some((c.name, word_bias(&all, tok)?))
            })
            .collect();
        let first_chars = (!persona.banned_first_chars.is_empty() || persona.lowercase_start).then(|| {
            let chars: String = persona.banned_first_chars.concat();
            let upper = persona.lowercase_start;
            // "Oh", "Well": an uppercase letter, then a lowercase one.
            let capitalized = |t: &str| {
                let mut c = t.chars();
                c.next().is_some_and(char::is_uppercase) && c.next().is_some_and(char::is_lowercase)
            };
            let mut ids = tok.ids_starting_where(|t| t.starts_with(|c| chars.contains(c)) || (upper && capitalized(t)));
            ids.extend(tok.newline_ids());
            Arc::new(ids)
        });
        let word_start = persona.sampler.word_start.as_ref().map(|w| {
            let s = &persona.sampler;
            WordStartParams {
                chance: w.chance,
                temperature: w.temperature.unwrap_or(s.temperature),
                top_k: w.top_k.unwrap_or(s.top_k),
                top_p: w.top_p.unwrap_or(s.top_p),
                min_p: w.min_p.unwrap_or(s.min_p),
                shapes: tok.pieces().iter().map(|p| TokenShape::of(p)).collect(),
            }
        });
        Ok(Self { persona, prompt: Arc::new(prompt), dry, words, command_words, first_chars, word_start })
    }

    /// The time line every reply's note starts with at `now`, "(it's
    /// night.)"; `None` without a `clock`.
    pub fn time_note(&self, now: std::time::SystemTime) -> Option<String> {
        let clock = self.persona.clock.as_ref()?;
        let secs = now.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        Some(clock.note_at(secs).trim().to_string())
    }

    /// The prompt notes for a reply to `message` (the user messages it answers).
    /// `recent`, `asker`, `replies`: see `chat::prompt::notes_for`
    /// (`ChannelWindow::since_replies`, `since_replies_by_asker`, `replies_since`).
    pub fn note(
        &self,
        recent: &dyn Fn(usize) -> String,
        asker: &dyn Fn(usize) -> String,
        replies: &dyn Fn(&[String], usize) -> Option<usize>,
    ) -> Option<String> {
        self.persona.note(recent, asker, replies)
    }

    /// [`Active::note`] for a novelty command's reply: the notes its prompt
    /// triggers, then the command's own.
    /// Then the command's notes its set-up line triggers (size, whose), then
    /// the moves and examples `draw` picked for this reply.
    pub fn command_note(&self, command: CommandName, prompt: &str, draw: Option<&CommandDraw>) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(d) = draw {
            parts.extend(d.facts.iter().cloned());
        }
        let replies = one_shot_replies(prompt);
        parts.extend(self.note(&|_| prompt.to_string(), &|_| prompt.to_string(), &replies));
        if let Some(c) = self.persona.command(command) {
            if !c.note.trim().is_empty() {
                parts.push(c.note.trim().to_string());
            }
            parts.extend(crate::chat::prompt::notes_for(&c.notes, &|_| prompt.to_string(), &|_| prompt.to_string(), &replies));
        }
        if let Some(d) = draw {
            if !d.moves.is_empty() {
                parts.push(format!("(this time he might: {}.)", d.moves.join("; or ")));
            }
            if !d.examples.is_empty() {
                let quoted: Vec<String> = d.examples.iter().map(|e| format!("\"{e}\"")).collect();
                parts.push(format!("(for example: {})", quoted.join(" or ")));
            }
        }
        (!parts.is_empty()).then(|| parts.join("\n"))
    }

    /// How many replies to generate for `command`, keeping the best by
    /// [`pick_score`]: its `best_of` while a GPU is ready, else 1. Never zero
    /// (by the type, not by a fallback `max(1)` downstream).
    pub fn best_of(&self, command: Option<CommandName>, on_gpu: bool) -> std::num::NonZeroUsize {
        match command.and_then(|c| self.persona.command(c)).and_then(|c| c.best_of) {
            Some(n) if on_gpu => n,
            _ => std::num::NonZeroUsize::MIN,
        }
    }

    /// The sampler settings for a reply to `message`.
    pub fn sampler(&self, message: &str) -> SamplerParams {
        self.sampler_for(message, None)
    }

    /// [`Active::sampler`], with `command`'s word biases added when it's a
    /// novelty command's reply.
    pub fn sampler_for(&self, message: &str, command: Option<CommandName>) -> SamplerParams {
        let s = &self.persona.sampler;
        let own = command.and_then(|c| self.persona.command(c)).and_then(|c| c.temperature);
        SamplerParams {
            temperature: own.unwrap_or(s.temperature),
            top_k: s.top_k,
            top_p: s.top_p,
            min_p: s.min_p,
            repeat: s.repeat.as_ref().map(|r| RepeatParams {
                penalty: r.penalty,
                last_n: r.last_n,
                history: r.scope == RepeatScope::Context,
            }),
            dry: self.dry.clone(),
            words: command
                .and_then(|c| self.command_words.get(&c))
                .cloned()
                .or_else(|| self.words.clone()),
            message: normalized_words(message).into(),
            observe: false,
            word_start: if command.is_none() { self.word_start.clone() } else { None },
        }
    }

    /// `stops` (the built-in ones for the prompt) plus the persona's.
    pub fn stops(&self, stops: Vec<String>) -> Vec<String> {
        self.persona.stops(stops)
    }

    /// The phrase bans for a reply to `message` (`command`'s added to the
    /// persona's), those with `prompted_by` only when `message` has one of
    /// their phrases. The caller adds the chat's own checks
    /// (`Bans::with_prefixes`, `with_echo`).
    ///
    /// A ban with `for_replies: n` goes by `replies` instead (see
    /// `chat::prompt::notes_for`): on for the bot's first `n` replies after
    /// the latest message with one of its phrases.
    pub fn bans(
        &self,
        message: &str,
        command: Option<CommandName>,
        replies: &dyn Fn(&[String], usize) -> Option<usize>,
    ) -> crate::engine::bans::Bans {
        use crate::engine::sampler::Phrases;
        let words = normalized_words(message);
        let own = command.and_then(|c| self.persona.command(c)).map(|c| c.bans.as_slice()).unwrap_or_default();
        let patterns: Vec<String> = self
            .persona
            .bans
            .iter()
            .chain(own)
            .filter(|b| match b.for_replies {
                Some(n) => replies(&b.prompted_by, n).is_some_and(|c| c < n),
                None => b.prompted_by.is_empty() || Phrases::new(&b.prompted_by).found_whole_in(&words),
            })
            .map(|b| b.pattern.clone())
            .collect();
        // Checked when the persona loaded.
        let bans = crate::engine::bans::Bans::parse(&patterns).unwrap_or_default();
        match (&self.first_chars, command) {
            (Some(ids), None) => bans.with_first_chars(ids.clone()).with_capitalized_start(self.persona.lowercase_start),
            _ => bans,
        }
    }
}

/// The discouraged-word table for the sampler (design.md §5), built from the
/// vocabulary; `None` if there are no stems.
pub fn word_bias(discourage: &[Word], tok: &Tokenizer) -> Option<Arc<WordBias>> {
    if discourage.is_empty() {
        return None;
    }
    let rules: Vec<StemRule> = discourage
        .iter()
        .map(|w| StemRule {
            after: w.after.clone(),
            within: w.within,
            prompted_by: w.prompted_by.clone(),
            whole_word: w.whole_word,
            ..StemRule::new(&w.stem, w.bias, w.min)
        })
        .collect();
    Some(Arc::new(WordBias::new(tok.pieces(), &rules)))
}

#[cfg(test)]
mod tests {

    #[test]
    fn picker_young_and_owner_patterns() {
        let c = CommandName::HotdogRater;
        let own = "*hands floob a measuring stick* floob, measure ur own hotdog! it's 8.4 cm long and 8.2 cm around, a ballpark frank.";
        let mine = "floob, measure my own hotdog! it's 8.4 cm long and 8.2 cm around, a ballpark frank.";
        let doru = "floob, measure Doru's hotdog! it's 8.4 cm long and 8.2 cm around, a ballpark frank.";
        // Rounds 20-22's young + sexual replies never win.
        for r in [
            "8.4? u could share some with little ol' me while we wait... *hopeful*",
            "what brings you ta show such a prize t' this lil ole floob?",
            "8.4! u could even share it with ur little buddy floob.",
            "8.4 cm... reminds me of when i was a wee kobold, uh, a fine frank!",
        ] {
            assert_eq!(pick_score(c, r, own, &[]), -1000, "{r:?}");
        }
        // Wrong owner costs, by whose it is; the right owner doesn't.
        let base = "8.4 and 8.2?! *squints* a ballpark frank, uh... wait, its not food?";
        let s = |r: &str, setup: &str| pick_score(c, r, setup, &[]);
        let plain = s(base, own);
        assert_eq!(s(&format!("{base} thanku smol bean!!"), own), plain - 4);
        assert_eq!(s(&format!("{base} where did you get it?"), own), plain - 4);
        assert_eq!(s(&format!("{base} can i keep it?"), own), plain - 4);
        assert_eq!(s(&format!("{base} u got a nice one there!"), doru), s(base, doru) - 4);
        assert_eq!(s(&format!("{base} my sausage is bigger tho"), mine), s(base, mine) - 4);
        assert_eq!(s(&format!("{base} doru must be proud"), doru), s(base, doru));
        assert_eq!(s(&format!("{base} ur sausage is fine, smol bean"), mine), s(base, mine));
    }
    use super::*;

    pub(super) const MINIMAL: &str = r####"// a comment
Persona {
    name: "t1",
    persona_name: "floob",
    persona: r#"
[Character: floob; "quoted", a \backslash]
"#,
    examples: r#"
pip: hi
floob: hi!!
"#,
    reminder: "",
    notes: [Note { text: "(floob is grown)", prompted_by: ["mommy"] }],
    discourage: [
        Word { stem: "gigg", bias: -2.5 },
        Word { stem: "seek", bias: -2, min: 4, after: ["play"], within: 2 },
    ],
    stop_strings: ["[Character", "[floob:"],
    sampler: Sampler {
        temperature: 1.05, top_k: 100, top_p: 0.99, min_p: 0.1,
        repeat: Some(Repeat { penalty: 1.1, last_n: 128, scope: Generated }),
        dry: None,
    },
    max_new_tokens: 75,
    max_new_tokens_gpu: 90,
    wrap_up_after: 40,
    single_paragraph: false,
}
"####;

    fn load_text(text: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<Persona> {
        // Tests run in parallel: each call gets its own folder.
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("increbot-persona-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("p.rsn");
        std::fs::write(&path, text).unwrap();
        let p = Persona::load(&path, env);
        std::fs::remove_dir_all(dir).ok();
        p
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn raw_strings_are_taken_as_is_and_trimmed() {
        let p = load_text(MINIMAL, &no_env).unwrap();
        assert_eq!(p.persona, r#"[Character: floob; "quoted", a \backslash]"#);
        assert_eq!(p.examples, "pip: hi\nfloob: hi!!");
        assert_eq!(p.sampler.repeat.as_ref().unwrap().scope, RepeatScope::Generated);
        assert_eq!(p.sampler.dry, None);
        assert_eq!((p.discourage[1].min, p.discourage[1].within), (4, 2));
        assert_eq!(p.limits(None).max_new_tokens_gpu, 90);
    }

    #[test]
    fn remember_commands_are_checked_and_optional() {
        let p = load_text(MINIMAL, &no_env).unwrap();
        assert!(p.remember_commands.is_empty() && !p.remembers("ask"));
        let with = |list: &str| MINIMAL.replace("    single_paragraph: false,\n", &format!("    single_paragraph: false,\n    remember_commands: [{list}],\n"));
        let q = load_text(&with(r#""ask", "roast""#), &no_env).unwrap();
        assert!(q.remembers("ask") && q.remembers("roast") && !q.remembers("judge"));
        assert_ne!(p.fingerprint(), q.fingerprint());
        let e = load_text(&with(r#""/roast""#), &no_env).unwrap_err();
        assert!(format!("{e:#}").contains("unknown command"), "{e:#}");
        let e = load_text(&with(r#""ask", "ask""#), &no_env).unwrap_err();
        assert!(format!("{e:#}").contains("listed twice"), "{e:#}");
    }

    #[test]
    fn the_source_is_kept_but_not_compared() {
        let p = load_text(MINIMAL, &no_env).unwrap();
        assert_eq!(&*p.source.0, MINIMAL);
        let q = load_text(&MINIMAL.replace("// a comment", "// another"), &no_env).unwrap();
        assert_eq!(p, q, "comments aren't the setup");
        assert_eq!(p.fingerprint(), q.fingerprint());
        assert_ne!(p.source.0, q.source.0);
    }

    #[test]
    fn max_lines_replaces_single_paragraph() {
        let lines = |v: &str| MINIMAL.replace("    single_paragraph: false,\n", &format!("    max_lines: {v},\n"));
        let old = load_text(MINIMAL, &no_env).unwrap();
        assert_eq!(old.max_lines, None);
        let any = load_text(&lines("None"), &no_env).unwrap();
        assert_eq!(any.fingerprint(), old.fingerprint(), "None is the old false");
        let one_old = load_text(&MINIMAL.replace("single_paragraph: false", "single_paragraph: true"), &no_env).unwrap();
        let one = load_text(&lines("1"), &no_env).unwrap();
        assert_eq!((one.max_lines, one_old.max_lines), (NonZeroUsize::new(1), NonZeroUsize::new(1)));
        assert_eq!(one.fingerprint(), one_old.fingerprint(), "1 is the old true");
        let two = load_text(&lines("Some(2)"), &no_env).unwrap();
        assert_eq!(two.limits(None).max_lines, NonZeroUsize::new(2));
        assert_ne!(two.fingerprint(), one.fingerprint());
        for (bad, msg) in [
            (MINIMAL.replace("    single_paragraph: false,\n", ""), "max_lines is missing"),
            (MINIMAL.replace("single_paragraph: false,", "single_paragraph: false, max_lines: 2,"), "old form of max_lines"),
            (lines("0"), "nonzero"),
        ] {
            let e = format!("{:#}", load_text(&bad, &no_env).unwrap_err());
            assert!(e.contains(msg), "{e}");
        }
    }

    #[test]
    fn limits_take_tokens_per_side_and_the_old_form_alike() {
        let old = load_text(MINIMAL, &no_env).unwrap();
        let tokens = |max: &str, wrap: &str| {
            MINIMAL.replace("    max_new_tokens: 75,\n    max_new_tokens_gpu: 90,\n", &format!("    max_new_tokens: {max},\n")).replace(
                "    wrap_up_after: 40,\n",
                &format!("    wrap_up_after: {wrap},\n"),
            )
        };
        let new = load_text(&tokens("Tokens { cpu: Some(75), gpu: Some(90) }", "Tokens { cpu: Some(40), gpu: Some(40) }"), &no_env).unwrap();
        assert_eq!(new.limits(None), old.limits(None));
        assert_eq!(new.fingerprint(), old.fingerprint(), "same setup, same fingerprint");
        // rsn's implicit `Some`.
        let bare = load_text(&tokens("Tokens { cpu: 75, gpu: 90 }", "Tokens { cpu: 40, gpu: 40 }"), &no_env).unwrap();
        assert_eq!(bare.fingerprint(), old.fingerprint());
        let l = old.limits(None);
        assert_eq!((l.max_new_tokens, l.max_new_tokens_gpu, l.wrap_up_after, l.wrap_up_after_gpu), (75, 90, 40, 40));
        // A side left out: the other's limit; no wrap-up there.
        let p = load_text(&tokens("Tokens { gpu: Some(120) }", "Tokens { cpu: Some(40) }"), &no_env).unwrap();
        let l = p.limits(None);
        assert_eq!((l.max_new_tokens, l.max_new_tokens_gpu, l.wrap_up_after, l.wrap_up_after_gpu), (120, 120, 40, 0));
        assert_ne!(p.fingerprint(), old.fingerprint());
        // Old 0s: the GPU limit takes the CPU's, and no wrap-up.
        let zeros = MINIMAL.replace("max_new_tokens_gpu: 90", "max_new_tokens_gpu: 0").replace("wrap_up_after: 40", "wrap_up_after: 0");
        let l = load_text(&zeros, &no_env).unwrap().limits(None);
        assert_eq!((l.max_new_tokens, l.max_new_tokens_gpu, l.wrap_up_after, l.wrap_up_after_gpu), (75, 75, 0, 0));
        for (bad, msg) in [
            (tokens("Tokens { }", "Tokens { }"), "needs a cpu or a gpu"),
            (tokens("Tokens { cpu: Some(0) }", "Tokens { }"), "nonzero"),
            (MINIMAL.replace("max_new_tokens: 75,", "max_new_tokens: Tokens { cpu: Some(75), gpu: Some(80) },"), "old form"),
        ] {
            let e = format!("{:#}", load_text(&bad, &no_env).unwrap_err());
            assert!(e.contains(msg), "{e}");
        }
        assert_eq!(p.prompt_name("FloobBot"), "floob");
    }

    #[test]
    fn every_field_is_required() {
        for field in ["    wrap_up_after: 40,\n", "        dry: None,\n", "    reminder: \"\",\n"] {
            assert!(MINIMAL.contains(field), "{field}");
            let err = format!("{:#}", load_text(&MINIMAL.replace(field, ""), &no_env).unwrap_err());
            assert!(err.contains("invalid persona"), "{field}: {err}");
        }
        let err = format!("{:#}", load_text(&MINIMAL.replace("name: \"t1\"", "nam: \"t1\""), &no_env).unwrap_err());
        assert!(err.contains("invalid persona"), "{err}");
    }

    #[test]
    fn rules_are_checked() {
        let bad = |from: &str, to: &str| format!("{:#}", load_text(&MINIMAL.replace(from, to), &no_env).unwrap_err());
        assert!(bad("bias: -2.5", "bias: 1.5").contains("bias must be negative"));
        assert!(bad("within: 2", "within: 21").contains("within must be"));
        assert!(bad("\"[floob:\"", "\"\"").contains("stop_strings"));
        assert!(bad("prompted_by: [\"mommy\"]", "prompted_by: []").contains("notes need"));
        assert!(bad("persona_name: \"floob\"", "persona_name: \"flo:ob\"").contains("persona_name"));
        assert!(bad("penalty: 1.1, last_n: 128", "penalty: 0.0, last_n: 128").contains("repeat needs"));
    }

    #[test]
    fn discourage_env_replaces_the_list() {
        let env = |k: &str| (k == "INCREBOT_DISCOURAGE").then(|| "toy=-2.5, me=-3:2@play_with/0".to_string());
        let p = load_text(MINIMAL, &env).unwrap();
        assert_eq!(p.discourage.len(), 2);
        assert_eq!((p.discourage[1].stem.as_str(), p.discourage[1].after.clone()), ("me", vec!["play with".to_string()]));
        let env = |k: &str| (k == "INCREBOT_DISCOURAGE").then(|| "dice=1.5".to_string());
        assert!(format!("{:#}", load_text(MINIMAL, &env).unwrap_err()).contains("bias must be negative"));
        let env = |k: &str| (k == "INCREBOT_DISCOURAGE").then(String::new);
        assert!(load_text(MINIMAL, &env).unwrap().discourage.is_empty());
    }

    #[test]
    fn fingerprint_ignores_name_order_and_comments() {
        let a = load_text(MINIMAL, &no_env).unwrap();
        let b = load_text(
            &MINIMAL
                .replace("// a comment\n", "")
                .replace("name: \"t1\"", "name: \"t2\"")
                .replace("[\"[Character\", \"[floob:\"]", "[\"[floob:\", \"[Character\"]"),
            &no_env,
        )
        .unwrap();
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(b.line(), format!("t2 ({})", a.fingerprint()));
        let mut c = a.clone();
        c.discourage.reverse();
        assert_eq!(a.fingerprint(), c.fingerprint());
        c.discourage[0].bias = -3.0;
        assert_ne!(a.fingerprint(), c.fingerprint());
        let mut d = a.clone();
        d.reminder = "x".into();
        assert_ne!(a.fingerprint(), d.fingerprint());
        let mut e = a.clone();
        e.sampler.temperature = 0.9;
        assert_ne!(a.fingerprint(), e.fingerprint());
    }

    #[test]
    fn context_fits() {
        let p = load_text(MINIMAL, &no_env).unwrap();
        let mut cfg = Config::default();
        cfg.generation.max_prompt_tokens = 1024;
        cfg.model.n_ctx = 1100;
        assert!(format!("{:#}", p.check_against(&cfg).unwrap_err()).contains("max_new_tokens.gpu"));
        cfg.model.n_ctx = 1114;
        p.check_against(&cfg).unwrap();
    }

    const HOTDOG: &str = r#"    commands: [
        Command {
            name: HotdogRater,
            prompt: "{bot}, measure {whose} hotdog: {length} by {girth}, a {tier}",
            note: "(floob thinks it's a real sausage.)",
            discourage: [Word { stem: "penis", bias: -4.0 }],
        },
    ],
}"#;

    #[test]
    fn the_picker_gates_young_and_explicit_and_prefers_the_joke() {
        let setup = "measure my hotdog! it's 13.3 cm long and 10.1 cm around, a respectable bratwurst";
        let shown = vec!["uhhh... looks kinda skinny for a hotdog... but yep! 5.5 is 5.5 and 7 is 7. nice weenie!!".to_string()];
        let hd = |r: &str| pick_score(CommandName::HotdogRater, r, setup, &shown);
        assert_eq!(hd("its perfect, just right for a little guy like me! sooo tasty"), -1000);
        assert_eq!(hd("oh! i can use this to measure how much i grow!!"), -1000);
        assert_eq!(hd("a fine penis indeed, 13 cm"), -1000);
        let joke = "*squints* 13.3 cm?! a respectable bratwurst... wait, thats not food. not food at all!";
        let flat = "it's a fine sausage. well seasoned and smoked, nice texture overall.";
        assert!(hd(joke) > hd(flat), "{} vs {}", hd(joke), hd(flat));
        assert!(hd("thornback: *hands floob a measuring stick* measure my hotdog") < hd(joke));
        // Copying a shown example loses to anything else; so do its numbers.
        assert_eq!(hd("*eyes it* looks kinda skinny for a hotdog... but yep! 5.5 is 5.5 and 7 is 7. nice weenie!!"), -500);
        assert!(hd("13.3 cm, a respectable bratwurst!") > hd("21.6 cm, a respectable bratwurst!"));
        assert!(copy_ratio("a b c d", "a b c d") == 1.0 && copy_ratio("a b", "c d") == 0.0);
    }

    #[test]
    fn draws_are_seeded_and_distinct() {
        assert_eq!(pick_k(10, 3, 7), pick_k(10, 3, 7));
        let p = pick_k(10, 3, 7);
        assert!(p.iter().all(|&i| i < 10) && p[0] != p[1] && p[1] != p[2] && p[0] != p[2]);
        assert_eq!(pick_k(2, 5, 1).len(), 2);
        assert!(pick_k(0, 2, 1).is_empty());
    }

    #[test]
    fn a_command_can_set_its_own_length_and_temperature() {
        let p = load_text(&with_commands(HOTDOG), &no_env).unwrap();
        let chat = p.limits(None);
        assert_eq!(p.limits(Some(CommandName::HotdogRater)), chat, "unset = the persona's");
        let own = HOTDOG.replace(
            "discourage: [Word { stem: \"penis\", bias: -4.0 }],",
            "discourage: [Word { stem: \"penis\", bias: -4.0 }],\n            max_new_tokens: Some(60), wrap_up_after: Some(55), temperature: Some(1.15),",
        );
        let q = load_text(&with_commands(&own), &no_env).unwrap();
        let hd = q.limits(Some(CommandName::HotdogRater));
        assert_eq!((hd.max_new_tokens, hd.max_new_tokens_gpu, hd.wrap_up_after, hd.wrap_up_after_gpu), (60, 60, 55, 55));
        // The old form's GPU limit; the new form says the same.
        let old = own.replace("max_new_tokens: Some(60),", "max_new_tokens: Some(90), max_new_tokens_gpu: Some(120),");
        let r = load_text(&with_commands(&old), &no_env).unwrap();
        let hd = r.limits(Some(CommandName::HotdogRater));
        assert_eq!((hd.max_new_tokens, hd.max_new_tokens_gpu, hd.wrap_up_after, hd.wrap_up_after_gpu), (90, 120, 55, 55));
        let new = own.replace(
            "max_new_tokens: Some(60), wrap_up_after: Some(55),",
            "max_new_tokens: Some(Tokens { cpu: Some(90), gpu: Some(120) }), wrap_up_after: Some(Tokens { cpu: Some(55), gpu: Some(55) }),",
        );
        let n = load_text(&with_commands(&new), &no_env).unwrap();
        assert_eq!(n.limits(Some(CommandName::HotdogRater)), hd);
        assert_eq!(n.fingerprint(), r.fingerprint());
        // Each side its own wrap-up; a side left out is off.
        let split = new.replace("gpu: Some(55)", "gpu: Some(80)");
        let hd = load_text(&with_commands(&split), &no_env).unwrap().limits(Some(CommandName::HotdogRater));
        assert_eq!((hd.wrap_up_after, hd.wrap_up_after_gpu), (55, 80));
        let off = new.replace("wrap_up_after: Some(Tokens { cpu: Some(55), gpu: Some(55) })", "wrap_up_after: Some(Tokens { cpu: Some(55) })");
        let hd = load_text(&with_commands(&off), &no_env).unwrap().limits(Some(CommandName::HotdogRater));
        assert_eq!((hd.max_new_tokens, hd.max_new_tokens_gpu, hd.wrap_up_after, hd.wrap_up_after_gpu), (90, 120, 55, 0));
        assert_eq!(q.limits(None), chat, "chat keeps the persona's");
        assert_ne!(q.fingerprint(), p.fingerprint());
        let bad = format!("{:#}", load_text(&with_commands(&own.replace("Some(1.15)", "Some(3.0)")), &no_env).unwrap_err());
        assert!(bad.contains("temperature must be"), "{bad}");
    }

    fn with_commands(commands: &str) -> String {
        let end = MINIMAL.rfind('}').unwrap();
        format!("{}{}", &MINIMAL[..end], commands)
    }

    #[test]
    fn draws_use_different_set_up_lines_until_they_run_out() {
        let lines = HOTDOG.replace(
            "discourage: [Word { stem: \"penis\", bias: -4.0 }],",
            "discourage: [Word { stem: \"penis\", bias: -4.0 }],\n            prompts: [\"b {bot}\", \"c {bot}\", \"d {bot}\", \"e {bot}\"],",
        );
        let p = load_text(&with_commands(&lines), &no_env).unwrap();
        let d = p.draws(CommandName::HotdogRater, 5, 42);
        let seen: std::collections::HashSet<_> = d.iter().map(|d| d.template.clone()).collect();
        assert_eq!((d.len(), seen.len()), (5, 5));
        assert_eq!(p.draw(CommandName::HotdogRater, 42).as_ref(), d.first(), "draw is the first of draws");
        assert_eq!(p.draws(CommandName::HotdogRater, 7, 42).len(), 7, "then they repeat");
        assert!(p.draws(CommandName::Bored, 5, 42).is_empty());
        // `prompt` can be left out when `prompts` has them all.
        let only = lines.replace("prompt: \"{bot}, measure {whose} hotdog: {length} by {girth}, a {tier}\",", "");
        let q = load_text(&with_commands(&only), &no_env).unwrap();
        let seen: std::collections::HashSet<_> = q.draws(CommandName::HotdogRater, 4, 42).into_iter().map(|d| d.template).collect();
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn commands_are_optional_and_checked() {
        let plain = load_text(MINIMAL, &no_env).unwrap();
        assert!(plain.commands.is_empty() && plain.command(CommandName::HotdogRater).is_none());
        let p = load_text(&with_commands(HOTDOG), &no_env).unwrap();
        let c = p.command(CommandName::HotdogRater).unwrap();
        assert_eq!(c.discourage[0].stem, "penis");
        // An empty list leaves the fingerprint as it was.
        let mut q = p.clone();
        q.commands.clear();
        assert_eq!(q.fingerprint(), plain.fingerprint());
        assert_ne!(p.fingerprint(), plain.fingerprint());
        let bad = |from: &str, to: &str| format!("{:#}", load_text(&with_commands(&HOTDOG.replace(from, to)), &no_env).unwrap_err());
        assert!(bad("name: HotdogRater", "name: Roast").contains("invalid persona"));
        assert!(bad("name: HotdogRater", "name: \"hotdog-rater\"").contains("invalid persona"));
        assert_eq!(CommandName::from_slash("hotdog-rater"), Some(CommandName::HotdogRater));
        assert_eq!(CommandName::from_slash("roast"), None);
        assert!(bad("{tier}", "{size}").contains("unknown placeholder {size}"));
        assert!(bad("{tier}", "{tier").contains("unmatched"));
        assert!(bad("bias: -4.0", "bias: 4.0").contains("bias must be negative"));
        let twice = HOTDOG.replace("    ],
}", &format!("{}    ],
}}", &HOTDOG[HOTDOG.find("        Command").unwrap()..HOTDOG.find("    ],").unwrap()]));
        assert!(format!("{:#}", load_text(&with_commands(&twice), &no_env).unwrap_err()).contains("set up twice"));
    }

    #[test]
    fn command_prompts_render() {
        let t = "{bot}, measure {whose} hotdog! {length}";
        assert_eq!(placeholders(t).unwrap(), ["bot", "whose", "length"]);
        assert_eq!(
            render_command(t, &[("bot", "floob"), ("whose", "Bob's"), ("length", "15.5")]),
            "floob, measure Bob's hotdog! 15.5"
        );
    }

    /// The persona the shipped config names. The candidate features below were
    /// each introduced by one candidate (c43, c47–c52) and the live persona
    /// keeps them; `bench/persona/README.md` has each one's history.
    fn shipped() -> Persona {
        Persona::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_PERSONA_FILE), &no_env).unwrap()
    }

    #[test]
    fn shipped_puts_safety_notes_before_giving_in() {
        let p = shipped();
        // "deal" (two replies on: he gives in) and "mommy" (he's a grown-up):
        // the give-in note comes first in the file, the safety one first here.
        let note = p.note(&|_| "deal? mommy says so".to_string(), &|_| "deal? mommy says so".to_string(), &|_, _| Some(2)).unwrap();
        let (grown, gives_in) = (note.find("grown kobold who runs").unwrap(), note.find("protested enough").unwrap());
        assert!(grown < gives_in, "{note}");
    }

    #[test]
    fn shipped_stops_at_a_new_parenthetical_line() {
        assert!(shipped().stops(Vec::new()).contains(&"\n(".to_string()));
    }

    #[test]
    fn shipped_remembers_every_generating_command() {
        assert_eq!(shipped().remember_commands, REMEMBERABLE);
    }

    #[test]
    fn shipped_lowercases_names() {
        assert!(shipped().lowercase_names);
    }

    #[test]
    fn shipped_bans_a_leading_bang_in_chat_only() {
        let p = shipped();
        assert_eq!(p.banned_first_chars, ["!", "(", "[", "#", "{", ",", "/", "\\", "|", ";", "`", "$", "%", "&", ")", "-", "_", "+", "}", "]", "~", "\""]);
        assert!(p.lowercase_start);
    }

    #[test]
    fn shipped_redraws_one_word_start_in_ten_hotter() {
        let w = shipped().sampler.word_start.unwrap();
        assert_eq!((w.chance, w.temperature, w.top_k, w.top_p, w.min_p), (0.1, Some(1.15), None, None, None));
    }

    /// c52's six times of day, as its `clock` lists them.
    const C52_TIMES: [(u8, &str); 6] =
        [(5, "dawn"), (8, "morning"), (12, "afternoon"), (17, "evening"), (21, "sunset"), (0, "night")];

    #[test]
    fn shipped_knows_the_time_of_day_in_eastern_summer_time() {
        let clock = shipped().clock.unwrap();
        assert_eq!(clock.utc_offset, -4);
        let times: Vec<(u8, String)> = clock.times.iter().map(|t| (t.from, t.note.clone())).collect();
        let want: Vec<(u8, String)> = C52_TIMES.iter().map(|&(h, n)| (h, format!("(it's {n}.)"))).collect();
        assert_eq!(times, want);
    }

    #[test]
    fn the_clock_picks_the_time_of_day() {
        let clock = |offset: i32| Clock {
            utc_offset: offset,
            times: C52_TIMES.iter().map(|&(from, n)| TimeOfDay { from, note: n.to_string() }).collect(),
        };
        let at = |h: u64, m: u64| (19_000 * 86_400) + h * 3600 + m * 60;
        for (utc_hour, offset, want) in [
            (9, 0, "morning"),
            (0, 0, "night"),
            (1, 0, "night"),
            (4, 0, "night"),
            (5, 0, "dawn"),
            (7, 0, "dawn"),
            (8, 0, "morning"),
            (11, 0, "morning"),
            (12, 0, "afternoon"),
            (16, 0, "afternoon"),
            (17, 0, "evening"),
            (20, 0, "evening"),
            (21, 0, "sunset"),
            (23, 0, "sunset"),
            // Eastern summer: 0:30 UTC is 20:30, 2:30 UTC is 22:30 the day
            // before, 6:30 UTC is 2:30.
            (0, -4, "evening"),
            (2, -4, "sunset"),
            (6, -4, "night"),
            (10, -4, "dawn"),
            (13, -4, "morning"),
            (23, 14, "afternoon"),
        ] {
            assert_eq!(clock(offset).note_at(at(utc_hour, 30)), want, "{utc_hour}:30 UTC at {offset}");
        }
        // Without a time from midnight, the latest one runs past it.
        let two = Clock {
            utc_offset: 0,
            times: vec![TimeOfDay { from: 20, note: "late".into() }, TimeOfDay { from: 6, note: "day".into() }],
        };
        assert_eq!((two.note_at(at(3, 0)), two.note_at(at(6, 0)), two.note_at(at(21, 0))), ("late", "day", "late"));
    }

    #[test]
    fn clock_settings_are_checked() {
        let with = |clock: &str| {
            MINIMAL.replace("    single_paragraph: false,\n", &format!("    single_paragraph: false,\n    clock: {clock},\n"))
        };
        let ok = |c: &str| Persona::parse(&with(c), "t", &|_| None);
        assert!(ok("Clock { utc_offset: -4, times: [Time { from: 0, note: \"(it's night.)\" }] }").is_ok());
        for bad in [
            "Clock { utc_offset: -13, times: [Time { from: 0, note: \"x\" }] }",
            "Clock { utc_offset: 0, times: [] }",
            "Clock { utc_offset: 0, times: [Time { from: 24, note: \"x\" }] }",
            "Clock { utc_offset: 0, times: [Time { from: 3, note: \" \" }] }",
            "Clock { utc_offset: 0, times: [Time { from: 3, note: \"a\" }, Time { from: 3, note: \"b\" }] }",
        ] {
            assert!(ok(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn word_start_settings_are_checked() {
        let with = |ws: &str| {
            MINIMAL.replace("    sampler: Sampler {\n", &format!("    sampler: Sampler {{\n        word_start: {ws},\n"))
        };
        assert!(Persona::parse(&with("WordStart { chance: 0.1, temperature: 1.15 }"), "t", &|_| None).is_ok());
        for bad in ["WordStart { chance: 1.5 }", "WordStart { chance: 0.1, temperature: 3.0 }", "WordStart { chance: 0.1, top_p: 0.0 }"] {
            assert!(Persona::parse(&with(bad), "t", &|_| None).is_err(), "{bad}");
        }
    }

    #[test]
    fn banned_first_chars_are_single_visible_characters() {
        let with = |list: &str| MINIMAL.replace("    single_paragraph: false,\n", &format!("    single_paragraph: false,\n    banned_first_chars: [{list}],\n"));
        assert!(Persona::parse(&with("\"!\", \"?\""), "t", &|_| None).is_ok());
        for bad in ["\"!!\"", "\" \"", "\"\""] {
            assert!(Persona::parse(&with(bad), "t", &|_| None).is_err(), "{bad}");
        }
    }

    #[test]
    fn shipped_directs_without_the_bored_scene() {
        // No shipped persona sets `Directed` up (design.md §6: without a
        // setup, only the notes the admin's line triggers apply): the draw
        // is empty, and shake puts the admin's line in its place, so nothing
        // of Bored's empty-chat scene reaches a directed line.
        let p = shipped();
        assert!(p.command(CommandName::Directed).is_none());
        assert!(p.draw(CommandName::Directed, 7).is_none());
        let b = p.command(CommandName::Bored).unwrap();
        assert!(!b.prompt.is_empty() || !b.prompts.is_empty(), "Bored keeps its scene");
        assert_eq!(CommandName::from_slash("shake-directed"), Some(CommandName::Directed));
    }

    #[test]
    fn shipped_personas_load() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("prompts");
        let mut n = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "rsn") {
                Persona::load(&path, &no_env).unwrap();
                n += 1;
            }
        }
        assert!(n >= 1, "no persona files in {}", dir.display());
    }
}
