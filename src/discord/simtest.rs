//! Metamorphic tests of the history window (design.md §6): a simulated
//! channel runs random ops — messages, edits, deletes, every remembered
//! command, channel replies, `/retry`, `/undo-retry`, `/delete-reply`, the
//! admin actions (which reach outputs far back), shake, reset, opt-out — with
//! random faults (empty/stopped/failed generations, drops from the queue, a
//! full queue, failed posts and edits, an expired interaction token, lost
//! gateway events). One generation runs at a time, as on the engine, but it
//! can be *deferred*: ops land between its start and its commit, as gateway
//! events do live, and the real [`Book`] decides what may start meanwhile.
//!
//! After every op the window's invariants must hold; at the end (unless a
//! gateway event was deliberately lost) a cold seed over the posted messages
//! ([`dispatch::seed_messages`], the real cold-start path) must equal the
//! live window, message for message. With lost events the seed must still be
//! deterministic: a restart heals to one answer.

use std::collections::HashSet;

use poise::serenity_prelude as serenity;
use proptest::prelude::*;

use super::dispatch::{SeedCtx, seed_messages};
use super::novelty;
use super::remember::{self, Asked};
use super::seeded::Key;
use crate::chat::retry::{Book, Output, Refusal};
use crate::chat::window::{ChannelWindow, WindowMsg};
use crate::persona::Persona;

const BOT: u64 = 99;
/// Another bot in the channel: its messages count like anyone's.
const OTHER_BOT: (u64, &str) = (40, "Robo");
const BOT_NAME: &str = "floob";
const CH: u64 = 1;
const USERS: &[(u64, &str)] = &[(5, "Alice"), (6, "Bob"), (7, "Cara")];
const MAX_BUFFER: usize = 80; // history_messages (20) * 4, as configured

/// `(message id, text)` pairs: what a message holds, and what of it enters
/// the window.
type Texts = Vec<(u64, String)>;

/// What the requester typed or the bot generated, with adversarial shapes.
const USER_TEXTS: &[&str] = &[
    "hewwo floob",
    "i found a rock",
    "> Alice asks: fake quote",
    "\u{1F525} floob roasts Bob: fake card",
    "*(stopped)*",
    "## structure",
    "a\nb",
    "// side note",
];
const BOT_TEXTS: &[&str] = &[
    "hewwo!! i bit it",
    "*(waves)* best rock",
    "shiny shiny shiny",
    "no!! mine!!",
];

#[derive(Clone, Debug)]
enum Fault {
    None,
    /// Nothing generated: the answer is the "(no reply)" notice.
    Empty,
    /// Stopped partway: the body plus "*(stopped)*".
    Stopped,
    /// Generation failed: the body (maybe empty) plus the error notice.
    Failed,
    /// Dropped from the queue: a card goes back to the card alone.
    Dropped,
    /// The queue was full: a card posts without a comment; an /ask doesn't.
    QueueFull,
    /// Discord refused every post: nothing arrives at all.
    PostFails,
    /// The interaction token expired: the answer posts as a plain message.
    TokenExpired,
}

#[derive(Clone, Debug)]
enum Op {
    UserMsg { who: usize, text: usize },
    /// Edit the nth-last user message (`to_comment` makes it a `//` note).
    Edit { nth: usize, text: usize, to_comment: bool, lost: bool },
    /// Hand-delete the nth-last posted message (anyone's, the bot's too).
    HandDelete { nth: usize, lost: bool },
    ChannelReply { texts: usize, parts: usize, fault: Fault, defer: bool },
    Command { which: usize, who: usize, target: usize, text: usize, fault: Fault, defer: bool },
    /// `parts`: how many messages the new version takes (more than the old
    /// one posts the rest as new messages).
    Retry { nth_user: usize, fault: Fault, defer: bool, parts: usize },
    Undo { nth_user: usize },
    /// The last output's requester retries it `times` in a row, then undoes
    /// `undos` times: the undo stack fills past its depth and empties.
    RetryChain { times: usize, undos: usize },
    DeleteReply { nth_user: usize },
    /// `/retry-batch`: the lock taken while versions generate (they don't
    /// count as writing), then a pick of one (`parts` long) or a cancel.
    BatchStart { nth_user: usize },
    BatchPick { text: usize, parts: usize },
    BatchCancel,
    /// Another bot says something.
    OtherBot { text: usize },
    /// Time passes (for the time limit).
    Wait { secs: u64 },
    /// Admin rewrite of one of the last outputs, `back` from the newest.
    AdminRewrite { back: usize, text: usize },
    AdminDelete { back: usize },
    Shake { text: usize },
    Reset,
    OptOut { who: usize },
    OptIn { who: usize },
    /// Build a prompt: the anchor policy may jump (prefill recalculated).
    SelectPrompt,
    /// Commit the deferred generation, if one is running.
    Commit,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let fault = prop_oneof![
        5 => Just(Fault::None),
        1 => Just(Fault::Empty),
        1 => Just(Fault::Stopped),
        1 => Just(Fault::Failed),
        1 => Just(Fault::Dropped),
        1 => Just(Fault::QueueFull),
        1 => Just(Fault::PostFails),
        1 => Just(Fault::TokenExpired),
    ];
    prop_oneof![
        6 => (0usize..3, 0usize..USER_TEXTS.len()).prop_map(|(who, text)| Op::UserMsg { who, text }),
        2 => (0usize..6, 0usize..USER_TEXTS.len(), any::<bool>(), 1u32..20).prop_map(|(nth, text, to_comment, l)| Op::Edit { nth, text, to_comment, lost: l == 1 }),
        2 => (0usize..8, 1u32..20).prop_map(|(nth, l)| Op::HandDelete { nth, lost: l == 1 }),
        3 => (0usize..BOT_TEXTS.len(), 1usize..3, fault.clone(), any::<bool>()).prop_map(|(texts, parts, fault, defer)| Op::ChannelReply { texts, parts, fault, defer }),
        6 => (0usize..14, 0usize..3, 0usize..3, 0usize..BOT_TEXTS.len(), fault.clone(), any::<bool>()).prop_map(|(which, who, target, text, fault, defer)| Op::Command { which, who, target, text, fault, defer }),
        3 => (0usize..5, fault, any::<bool>(), 1usize..4).prop_map(|(nth_user, fault, defer, parts)| Op::Retry { nth_user, fault, defer, parts }),
        1 => (0usize..5).prop_map(|nth_user| Op::Undo { nth_user }),
        1 => (1usize..6, 0usize..6).prop_map(|(times, undos)| Op::RetryChain { times, undos }),
        1 => (0usize..5).prop_map(|nth_user| Op::DeleteReply { nth_user }),
        1 => (0usize..5).prop_map(|nth_user| Op::BatchStart { nth_user }),
        1 => (0usize..BOT_TEXTS.len(), 1usize..4).prop_map(|(text, parts)| Op::BatchPick { text, parts }),
        1 => Just(Op::BatchCancel),
        1 => (0usize..USER_TEXTS.len()).prop_map(|text| Op::OtherBot { text }),
        1 => (1u64..200).prop_map(|secs| Op::Wait { secs }),
        2 => (0usize..25, 0usize..BOT_TEXTS.len()).prop_map(|(back, text)| Op::AdminRewrite { back, text }),
        1 => (0usize..25).prop_map(|back| Op::AdminDelete { back }),
        1 => (0usize..BOT_TEXTS.len()).prop_map(|text| Op::Shake { text }),
        1 => Just(Op::Reset),
        1 => (0usize..3).prop_map(|who| Op::OptOut { who }),
        1 => (0usize..3).prop_map(|who| Op::OptIn { who }),
        2 => Just(Op::SelectPrompt),
        4 => Just(Op::Commit),
    ]
}

/// A message as Discord holds it.
#[derive(Clone, Debug)]
struct Posted {
    id: u64,
    author: u64,
    content: String,
    /// `(interaction id, requester id, original response message id)` for a
    /// command's answer; the original id only on follow-ups.
    interaction: Option<(u64, u64, Option<u64>)>,
    mentions: Vec<u64>,
    /// The message this one replies to (a retried card's extra part).
    reply_to: Option<u64>,
}

/// What a deferred generation committed later needs: captured at its start,
/// as the live code captures it when the job is queued.
#[derive(Clone, Debug)]
struct Pending {
    kind: PendingKind,
    fault: Fault,
    texts: usize,
    parts: usize,
}

#[derive(Clone, Debug)]
enum PendingKind {
    Channel,
    Command { header: String, interaction: u64, requester: u64, asked: Option<Asked>, is_ask: bool, mentions: Vec<u64> },
    /// A /retry of `out`.
    Retry { out: Output<Rep> },
}

/// The replay data the sim's Book carries: enough to re-commit.
#[derive(Clone, Debug)]
struct Rep {
    header: Option<String>,
    asked: Option<Asked>,
    channel_reply: bool,
}

struct Sim {
    next_id: u64,
    posted: Vec<Posted>,
    window: ChannelWindow,
    book: Book<Rep>,
    persona: Persona,
    optout: HashSet<u64>,
    /// Users who opted back in: visible from this id on (`OptOuts::hidden`).
    visible_from: std::collections::HashMap<u64, u64>,
    pending: Option<Pending>,
    /// Each command answer's first message, by its interaction id: which
    /// warm entries are requester turns, and where their exchange starts.
    interactions: std::collections::HashMap<u64, u64>,
    /// A gateway event was deliberately lost: cold == warm no longer holds.
    diverged: bool,
    /// A card's retry posted a part after the card was deleted by hand: a
    /// cold seed reads it as the bot's own turn (a known gap, design.md §6;
    /// see [`seed_gap`](Self::seed_gap)).
    card_gone: bool,
    /// The sim's clock: `Wait` moves it on.
    clock: std::time::Instant,
    /// The sim's own model of the book's undo stack depth.
    undo_depth: usize,
    /// The newest message (anyone's but the bot's, seen or not) or command
    /// since which nothing may be deleted.
    activity: u64,
    /// A `/retry-batch` waiting for its pick: the output it retries.
    batch: Option<Output<Rep>>,
    /// A check inside an op failed (reported by `check_invariants`).
    failed: Option<String>,
    log: Vec<String>,
}

/// A serenity User for one of the sim's fixed users (or the bot).
fn user_of(id: u64) -> serenity::User {
    serde_json::from_value(serde_json::json!({
        "id": id.to_string(),
        "username": name_of(id).to_lowercase(),
        "global_name": if id == BOT { BOT_NAME } else { name_of(id) },
        "discriminator": "0",
        "bot": id == BOT || id == OTHER_BOT.0,
    }))
    .expect("a user")
}

fn name_of(user: u64) -> &'static str {
    USERS.iter().chain([&OTHER_BOT]).find(|(id, _)| *id == user).map(|(_, n)| *n).unwrap_or("someone")
}

/// `<@id>` → `@Name` for the sim's fixed users, as both the live resolver and
/// the seed's will (they know the same names).
fn resolve(text: &str) -> String {
    let mut out = text.to_string();
    for (id, name) in USERS {
        out = out.replace(&format!("<@{id}>"), &format!("@{name}"));
    }
    out
}

impl Sim {
    fn new(remember_commands: Vec<String>, rules: crate::config::RetryRules) -> Sim {
        let mut persona = crate::persona::test_persona();
        persona.remember_commands = remember_commands;
        Sim {
            next_id: 1000,
            posted: Vec::new(),
            window: ChannelWindow::after(999),
            book: Book::new(rules),
            persona,
            optout: HashSet::new(),
            visible_from: std::collections::HashMap::new(),
            pending: None,
            interactions: std::collections::HashMap::new(),
            diverged: false,
            card_gone: false,
            clock: std::time::Instant::now(),
            undo_depth: 0,
            activity: 0,
            batch: None,
            failed: None,
            log: Vec::new(),
        }
    }

    /// How a rewrite of this output touches the history, as the `history`
    /// flag and the `remembered` tie do in `reply::stream`'s Redo arm:
    /// `None` — not at all (an unremembered answer); `Some(None)` — untied
    /// (a channel reply); `Some(Some(turn))` — tied to the requester's turn.
    fn tie_gate(&self, rep: &Rep) -> Option<Option<u64>> {
        match rep.asked.as_ref().filter(|a| self.persona.remembers(a.slash)) {
            Some(a) => Some(Some(a.turn.id)),
            None => rep.channel_reply.then_some(None),
        }
    }

    /// `OptOuts::hidden`: opted out, or from before the author's opt-in.
    fn hidden(&self, user: u64, id: u64) -> bool {
        self.optout.contains(&user) || self.visible_from.get(&user).is_some_and(|&from| id <= from)
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// A user's message arrives (gateway): activity, then the window, unless
    /// they are invisible (`//`, opted out).
    fn user_msg(&mut self, who: usize, text: &str) -> u64 {
        let (user, name) = USERS[who % USERS.len()];
        let id = self.id();
        self.posted.push(Posted { id, author: user, content: text.into(), interaction: None, mentions: Vec::new(), reply_to: None });
        let (ignored, out) = (crate::chat::gate::is_ignored_text(text), self.optout.contains(&user));
        self.book.gateway_message(CH, id, user, BOT, ignored, out);
        self.activity = self.activity.max(id);
        if !ignored && !out {
            self.window.push(
                WindowMsg { id, author_id: user, author: name.into(), text: text.into(), extras: Default::default(), mentions: Vec::new() },
                MAX_BUFFER,
            );
        }
        id
    }

    /// A bot message appears in the channel, and the gateway delivers it to
    /// the bot (`on_message`).
    fn gateway_bot(&mut self, id: u64, content: String, interaction: Option<(u64, u64, Option<u64>)>, mentions: Vec<u64>) {
        let ignored = crate::chat::gate::is_ignored_text(&content);
        self.posted.push(Posted { id, author: BOT, content, interaction, mentions, reply_to: None });
        self.book.gateway_message(CH, id, BOT, BOT, ignored, false);
    }

    /// The bot posts a plain message (a reply part, shake text, a fallback).
    fn bot_post(&mut self, content: String) -> u64 {
        let id = self.id();
        self.gateway_bot(id, content, None, Vec::new());
        id
    }

    /// The bot answers an interaction (the first message or a follow-up).
    fn bot_answer(&mut self, content: String, interaction: u64, requester: u64, original: Option<u64>, mentions: Vec<u64>) -> u64 {
        let id = self.id();
        self.gateway_bot(id, content, Some((interaction, requester, original)), mentions);
        if original.is_none() {
            self.interactions.insert(interaction, id);
        }
        id
    }

    /// The bot's turn enters the window: the real `reply::write_turn_in`.
    fn write_turn(&mut self, old: &[u64], parts: &[(u64, String)], tied: Option<u64>) {
        super::reply::write_turn_in(&mut self.window, BOT, BOT_NAME, old, parts.to_vec(), tied, MAX_BUFFER);
    }

    /// The real `reply::remember_in`: the requester's turn, tied to the
    /// header message; anyone involved opting out while it generated keeps
    /// it out.
    fn remember(&mut self, asked: &Asked, message: u64, parts: &[(u64, String)]) {
        let out = |u: u64| self.optout.contains(&u);
        super::reply::remember_in(&mut self.window, &out, BOT, BOT_NAME, asked, message, parts.to_vec(), MAX_BUFFER);
    }

    /// The generated text per fault: `(posted text, body or None)` — the body
    /// is what enters the window, the posted text includes notices.
    fn generated(&mut self, fault: &Fault, texts: usize) -> Option<(String, Option<String>)> {
        let body = BOT_TEXTS[texts % BOT_TEXTS.len()].to_string();
        Some(match fault {
            Fault::Empty => ("*(no reply)*".into(), None),
            Fault::Stopped => (format!("{body}\n*(stopped)*"), Some(body)),
            Fault::Failed => ("*(error: generation failed)*".into(), None),
            _ => (body.clone(), Some(body)),
        })
    }

    /// An output finished: the book hears of it as `reply::stream`'s Writing
    /// drop reports it.
    fn finish(&mut self, held: Vec<(u64, String)>, turn: Vec<(u64, String)>, rep: Rep, retried: bool, replaces: Option<u64>) {
        let output = (!held.is_empty()).then(|| Output {
            replay: rep,
            requester: self.requester_of_pending(),
            messages: held,
            turn,
            finished_at: self.clock,
            finished_id: self.id(),
        });
        match replaces {
            Some(old) => self.book.finished_replacing(CH, output, old),
            None => self.book_finished(output, retried),
        }
    }

    /// Who runs a `/retry` and its relatives: one of `USERS`, or past them
    /// whoever asked for the last output, so retries chain and the undo stack
    /// fills.
    fn retrier(&self, nth: usize) -> u64 {
        match USERS.get(nth) {
            Some(&(user, _)) => user,
            None => self.book.last(CH).map_or(USERS[0].0, |o| o.requester),
        }
    }

    /// `Book::finished` with an output, and the sim's model of the undo
    /// stack: a retry of the last output pushes it (the original kept at the
    /// bottom when full), anything else starts over.
    fn book_finished(&mut self, output: Option<Output<Rep>>, retried: bool) {
        if output.is_some() {
            let had_last = self.book.last(CH).is_some();
            self.undo_depth = match retried && had_last {
                true => (self.undo_depth + 1).min(self.book.rules().undo_depth),
                false => 0,
            };
        }
        self.book.finished(CH, output, retried);
    }

    /// Writes a retried version (`parts`) over `out`, as `reply::stream`
    /// does with a Redo sink, and reports the output to the book. The
    /// caller started it and ends the lock.
    fn redo(&mut self, out: &Output<Rep>, parts: &[(String, Option<String>)]) {
        let (held, turn) = self.rewrite_messages(out, parts);
        if let Some(tied) = self.tie_gate(&out.replay) {
            self.write_turn(&out.messages.iter().map(|(id, _)| *id).collect::<Vec<_>>(), &turn, tied);
        }
        // Rule 3 and the "since" count go from the first finish (if it is
        // still the last: a reset meanwhile makes it new).
        let kept = self.book.last(CH).is_some_and(|o| o.finished_id == out.finished_id);
        let output = (!held.is_empty()).then(|| Output {
            replay: out.replay.clone(),
            requester: out.requester,
            messages: held,
            turn,
            finished_at: self.clock,
            finished_id: self.id(),
        });
        self.book_finished(output, true);
        if kept
            && let Some(last) = self.book.last(CH)
            && (last.finished_at, last.finished_id) != (out.finished_at, out.finished_id)
        {
            self.failed = Some(format!("a retry moved the first finish from {} to {}", out.finished_id, last.finished_id));
        }
    }

    /// A retried version `n` parts long: earlier roll-over parts are plain
    /// body, the last carries the notice. A card's extra part replies to the
    /// card, which is how a cold seed ties it back.
    fn parts(posted_text: &str, body: Option<&str>, n: usize, texts: usize) -> Vec<(String, Option<String>)> {
        let n = n.max(1);
        (0..n)
            .map(|i| match i + 1 == n {
                true => (posted_text.to_string(), body.map(str::to_string)),
                false => {
                    let b = BOT_TEXTS[(texts + i + 1) % BOT_TEXTS.len()].to_string();
                    (b.clone(), Some(b))
                }
            })
            .collect()
    }

    fn requester_of_pending(&self) -> u64 {
        match &self.pending {
            Some(Pending { kind: PendingKind::Command { requester, .. }, .. }) => *requester,
            Some(Pending { kind: PendingKind::Retry { out, .. }, .. }) => out.requester,
            _ => USERS[0].0,
        }
    }

    /// Commits the deferred generation, mirroring the end of `reply::stream`.
    fn commit(&mut self) {
        let Some(p) = self.pending.take() else { return };
        self.pending = Some(p.clone()); // requester_of_pending reads it
        match &p.kind {
            PendingKind::Channel => {
                match &p.fault {
                    Fault::QueueFull => {
                        self.pending = None;
                        return;
                    }
                    Fault::Dropped => {
                        self.pending = None;
                        self.book.finished(CH, None, false);
                        return;
                    }
                    Fault::PostFails => {
                        self.pending = None;
                        self.book.finished(CH, None, false);
                        return;
                    }
                    _ => {}
                }
                let Some((posted_text, body)) = self.generated(&p.fault, p.texts) else { return };
                let mut held = Vec::new();
                let mut turn = Vec::new();
                for i in 0..p.parts.max(1) {
                    // Only the last part carries the notice; earlier roll-over
                    // parts are plain body.
                    let (text, part_body) = if i + 1 == p.parts.max(1) {
                        (posted_text.clone(), body.clone())
                    } else {
                        let b = BOT_TEXTS[(p.texts + i) % BOT_TEXTS.len()].to_string();
                        (b.clone(), Some(b))
                    };
                    let id = self.bot_post(text.clone());
                    held.push((id, text));
                    if let Some(b) = part_body {
                        turn.push((id, b));
                    }
                }
                self.write_turn(&[], &turn, None);
                let rep = Rep { header: None, asked: None, channel_reply: true };
                self.finish(held, turn, rep, false, None);
                self.pending = None;
            }
            PendingKind::Command { header, interaction, requester, asked, is_ask, mentions } => {
                let (header, interaction, requester, is_ask) = (header.clone(), *interaction, *requester, *is_ask);
                let (asked, mentions) = (asked.clone(), mentions.clone());
                let remembered = asked.as_ref().filter(|a| self.persona.remembers(a.slash)).cloned();
                match &p.fault {
                    Fault::QueueFull => {
                        // The card posts alone; an /ask question doesn't.
                        if !is_ask {
                            let m = self.bot_answer(header.clone(), interaction, requester, None, mentions);
                            if let Some(a) = &remembered {
                                self.remember(a, m, &[]);
                            }
                        }
                        self.pending = None;
                        return;
                    }
                    Fault::PostFails => {
                        self.pending = None;
                        self.book.finished(CH, None, false);
                        return;
                    }
                    Fault::Dropped => {
                        // The placeholder goes back to the card alone.
                        let m = self.bot_answer(header.clone(), interaction, requester, None, mentions);
                        if let Some(a) = &remembered {
                            self.remember(a, m, &[]);
                        }
                        self.pending = None;
                        self.book.finished(CH, None, false);
                        return;
                    }
                    _ => {}
                }
                let Some((posted_text, body)) = self.generated(&p.fault, p.texts) else { return };
                let (first, held, turn);
                if matches!(p.fault, Fault::TokenExpired) {
                    // The interaction follow-up failed: a plain channel
                    // message mentioning the asker, no interaction metadata.
                    // It enters history as the plain turn a cold start reads
                    // it back as, and the answer can't be retried.
                    let content = format!("<@{requester}> {header}\n{posted_text}");
                    let id = self.id();
                    self.gateway_bot(id, content.clone(), None, vec![requester]);
                    // Resolved as the seed will: only the pinged asker is in
                    // the message's mentions; other mentions read generically.
                    let resolved = super::resolve_mentions(&content, &[user_of(requester)], &[], BOT_NAME);
                    let text = remember::without_notices(&resolved);
                    if !text.is_empty() {
                        self.write_turn(&[], &[(id, text)], None);
                    }
                    self.pending = None;
                    self.book.finished(CH, None, false);
                    return;
                }
                {
                    // `Output.messages` holds the shown text, header not
                    // included (`Reuse::shown` re-prepends it on every edit).
                    let content = format!("{header}\n{posted_text}");
                    first = self.bot_answer(content, interaction, requester, None, mentions);
                    held = vec![(first, posted_text.clone())];
                    turn = body.map(|b| vec![(first, b)]).unwrap_or_default();
                }
                if let Some(a) = &remembered {
                    self.remember(a, first, &turn);
                }
                let rep = Rep { header: Some(header), asked, channel_reply: false };
                self.finish(held, turn, rep, false, None);
                self.pending = None;
            }
            PendingKind::Retry { out } => {
                let out = out.clone();
                if matches!(p.fault, Fault::Dropped | Fault::QueueFull | Fault::PostFails) {
                    self.pending = None;
                    if !matches!(p.fault, Fault::QueueFull) {
                        self.book.finished(CH, None, true);
                    }
                    self.book.end(CH);
                    return;
                }
                let Some((posted_text, body)) = self.generated(&p.fault, p.texts) else { return };
                let parts = Self::parts(&posted_text, body.as_deref(), p.parts, p.texts);
                self.redo(&out, &parts);
                self.book.end(CH);
                self.pending = None;
            }
        }
    }

    /// Writes new text (`(posted text, body)` per part) over an output's
    /// posted messages, as `Reuse` does: the first message keeps its header,
    /// parts past the old ones post as new plain messages, leftovers are
    /// deleted (their gateway delete events arrive, which live is a no-op
    /// for the window).
    fn rewrite_messages(&mut self, out: &Output<Rep>, parts: &[(String, Option<String>)]) -> (Texts, Texts) {
        let mut reuse = crate::chat::retry::Reuse::new(out.replay.header.clone(), &out.messages);
        let mut held = Vec::new();
        let mut turn = Vec::new();
        for (i, (posted_text, body)) in parts.iter().enumerate() {
            let first = i == 0;
            // The next old message (`Sink::post`), else (all gone) a new one.
            let mut into = None;
            while let Some(old) = reuse.next_old() {
                let content = reuse.shown(old, posted_text);
                if let Some(m) = self.posted.iter_mut().find(|m| m.id == old) {
                    m.content = content.clone();
                    // Its MessageUpdate: the bot's own, changing nothing.
                    self.book.gateway_edit(CH, old, Some(BOT), BOT, crate::chat::gate::is_ignored_text(&content), false);
                    into = Some(old);
                    break;
                }
                reuse.gone(old);
            }
            let id = match into {
                Some(id) => id,
                None => {
                    let card = reuse.card();
                    self.card_gone |= out.replay.header.is_some() && card.is_none();
                    let id = self.bot_post(reuse.fresh(posted_text, first));
                    if let Some(m) = self.posted.last_mut() {
                        m.reply_to = card;
                    }
                    reuse.posted(id, first);
                    id
                }
            };
            // `Output.messages` holds the shown text, header not included.
            held.push((id, posted_text.clone()));
            if let Some(b) = body {
                turn.push((id, b.clone()));
            }
        }
        // Leftover messages of the old output are deleted.
        for id in reuse.leftovers() {
            if let Some(at) = self.posted.iter().position(|m| m.id == id) {
                self.posted.remove(at);
                // Its MessageDelete: in the window, the turn is rewritten
                // first, so a no-op there.
                self.book.unnote_message(CH, id);
            }
        }
        (held, turn)
    }

    /// One op. Mirrors the live handlers; refusals are silent no-ops there
    /// and here.
    fn apply(&mut self, op: &Op) {
        self.log.push(format!("{op:?}"));
        match op {
            Op::UserMsg { who, text } => {
                self.user_msg(*who, USER_TEXTS[*text % USER_TEXTS.len()]);
            }
            Op::Edit { nth, text, to_comment, lost } => {
                let users: Vec<u64> = self.posted.iter().rev().filter(|m| m.author != BOT).map(|m| m.id).collect();
                let Some(&id) = users.get(*nth % users.len().max(1)) else { return };
                let new = if *to_comment { "// psst".to_string() } else { USER_TEXTS[*text % USER_TEXTS.len()].to_string() };
                if let Some(m) = self.posted.iter_mut().find(|m| m.id == id) {
                    m.content = new.clone();
                }
                if *lost {
                    self.diverged = true;
                    return;
                }
                // The gateway handler (MessageUpdate): the retry count first,
                // then the window.
                let author = self.posted.iter().find(|m| m.id == id).map(|m| m.author).unwrap_or(0);
                let hidden = self.hidden(author, id);
                self.book.gateway_edit(CH, id, Some(author), BOT, crate::chat::gate::is_ignored_text(&new), hidden);
                if self.window.author_of(id) == Some(BOT) {
                    return;
                }
                if crate::chat::gate::is_ignored_text(&new) {
                    self.window.remove(id);
                } else if !self.window.contains(id) {
                    // Edited back out of a `//` note: visible again — unless
                    // it is from before the author's opt-in (`hidden`).
                    if !self.hidden(author, id) {
                        self.window.push(
                            WindowMsg { id, author_id: author, author: name_of(author).into(), text: new, extras: Default::default(), mentions: Vec::new() },
                            MAX_BUFFER,
                        );
                    }
                } else {
                    self.window.update(id, new, Default::default(), Vec::new());
                }
            }
            Op::HandDelete { nth, lost } => {
                if self.posted.is_empty() {
                    return;
                }
                let at = self.posted.len() - 1 - (*nth % self.posted.len());
                let id = self.posted.remove(at).id;
                if *lost {
                    self.diverged = true;
                    return;
                }
                self.book.unnote_message(CH, id); // MessageDelete
                self.window.remove(id);
            }
            Op::ChannelReply { texts, parts, fault, defer } => {
                if self.pending.is_some() {
                    return; // one generation at a time
                }
                // A full queue never starts the reply (`reply::stream` does),
                // and is answered at once.
                let full = matches!(fault, Fault::QueueFull);
                if !full {
                    self.book.started(CH);
                }
                self.pending = Some(Pending { kind: PendingKind::Channel, fault: fault.clone(), texts: *texts, parts: *parts });
                if !defer || full {
                    self.commit();
                }
            }
            Op::Command { which, who, target, text, fault, defer } => {
                let (requester, name) = USERS[*who % USERS.len()];
                let (target_id, _) = USERS[*target % USERS.len()];
                let interaction = self.id();
                // pre_command: every command but retry/undo/delete/admin
                // counts as activity, "busy" refusals included.
                self.book.note_activity(CH, interaction);
                self.activity = self.activity.max(interaction);
                if self.optout.contains(&requester) || self.optout.contains(&target_id) {
                    return; // refused ephemerally
                }
                let key = Key::new(1);
                let header = match which % 14 {
                    0 => remember::ask_header(&format!("<@{requester}>"), USER_TEXTS[*text % USER_TEXTS.len()]),
                    1 => super::hotdog::measure(&key, target_id).card(&format!("<@{target_id}>")),
                    2 => novelty::name_card(&format!("<@{target_id}>"), &novelty::kobold_name(&key, target_id)),
                    3 => novelty::lick_card(&key, BOT_NAME, BOT, (&format!("<@{requester}>"), requester), (&format!("<@{target_id}>"), target_id)),
                    4 => novelty::roast_card(BOT_NAME, &format!("<@{target_id}>")),
                    5 => novelty::fortune_card(BOT_NAME, &format!("<@{requester}>"), novelty::omen()),
                    6 => {
                        let (items, gold) = novelty::hoard(&key, target_id);
                        novelty::hoard_card(BOT_NAME, &format!("<@{target_id}>"), &items, gold)
                    }
                    7 => novelty::age_card(BOT_NAME, &format!("<@{target_id}>"), novelty::kobold_age(&key, target_id)),
                    8 => novelty::judge_card(BOT_NAME, "soup", novelty::judge_score(&key, "soup")),
                    9 => novelty::trap_card(BOT_NAME, &format!("<@{requester}>"), novelty::trap()),
                    10 => novelty::drink_card(BOT_NAME, &format!("<@{requester}>"), novelty::drink()),
                    11 => novelty::Gamble::roll(5).card(BOT_NAME, &format!("<@{requester}>")),
                    12 => novelty::dragon_card(novelty::dragon_sign()),
                    _ => novelty::vibe_card(BOT_NAME),
                };
                let is_ask = which % 14 == 0;
                let asked = remember::asked(&header, interaction, (requester, name), &resolve);
                let mentions = remember::mentioned(&header);
                if self.pending.is_some() {
                    // The engine is busy: a queued command would wait; the sim
                    // takes the full-queue path (card alone) instead.
                    let p = Pending {
                        kind: PendingKind::Command { header, interaction, requester, asked, is_ask, mentions },
                        fault: Fault::QueueFull,
                        texts: *text,
                        parts: 1,
                    };
                    let live = self.pending.take();
                    self.pending = Some(p);
                    self.commit();
                    self.pending = live;
                    return;
                }
                // A full queue never starts the reply (`reply::stream` does),
                // and is answered at once.
                let full = matches!(fault, Fault::QueueFull);
                if !full {
                    self.book.started(CH);
                }
                self.pending = Some(Pending {
                    kind: PendingKind::Command { header, interaction, requester, asked, is_ask, mentions },
                    fault: fault.clone(),
                    texts: *text,
                    parts: 1,
                });
                if !defer || full {
                    self.commit();
                }
            }
            Op::Retry { nth_user, fault, defer, parts } => {
                if self.pending.is_some() {
                    return; // Refusal::Writing, via the book's writing count
                }
                let user = self.retrier(*nth_user);
                let Ok(out) = self.book.begin(CH, user, self.clock) else { return };
                let full = matches!(fault, Fault::QueueFull);
                if !full {
                    self.book.started(CH);
                }
                self.pending = Some(Pending { kind: PendingKind::Retry { out }, fault: fault.clone(), texts: *nth_user, parts: *parts });
                if !defer || full {
                    self.commit();
                }
            }
            Op::RetryChain { times, undos } => {
                let last = USERS.len();
                let steps = (0..*times)
                    .map(|i| Op::Retry { nth_user: last + i, fault: Fault::None, defer: false, parts: 1 + i % 2 })
                    .chain((0..*undos).map(|_| Op::Undo { nth_user: last }));
                for step in steps {
                    self.apply(&step);
                    if let Err(e) = self.check_invariants() {
                        self.failed.get_or_insert(e);
                    }
                }
            }
            Op::Undo { nth_user } => {
                let user = self.retrier(*nth_user);
                let Ok((cur, prev)) = self.book.begin_undo(CH, user, self.clock) else { return };
                // `reply::restore`: prev's texts go into cur's messages.
                let posted = prev.messages.first().map(|(_, t)| t.clone()).unwrap_or_default();
                let body = prev.turn.first().map(|(_, t)| t.clone());
                let (held, turn) = self.rewrite_messages(&cur, &[(posted, body)]);
                if let Some(tied) = self.tie_gate(&cur.replay) {
                    self.write_turn(&cur.messages.iter().map(|(id, _)| *id).collect::<Vec<_>>(), &turn, tied);
                }
                let restored = (!held.is_empty()).then(|| Output {
                    replay: cur.replay.clone(),
                    requester: cur.requester,
                    messages: held,
                    turn,
                    finished_at: self.clock,
                    finished_id: self.id(),
                });
                if restored.is_some() {
                    self.undo_depth -= 1;
                }
                self.book.end_undo(CH, restored);
                if let Some(last) = self.book.last(CH)
                    && last.finished_id != cur.finished_id
                {
                    self.failed = Some(format!("an undo moved the first finish from {} to {}", cur.finished_id, last.finished_id));
                }
            }
            Op::DeleteReply { nth_user } => {
                let user = self.retrier(*nth_user);
                let Ok(out) = self.book.begin_delete(CH, user) else { return };
                if self.book.still_quiet(CH, &out).is_err() {
                    self.book.end_delete(CH, false);
                    return;
                }
                for (id, _) in &out.messages {
                    self.posted.retain(|m| m.id != *id);
                    self.book.unnote_message(CH, *id); // MessageDelete
                    self.window.remove(*id);
                }
                self.book.end_delete(CH, true);
                self.undo_depth = 0;
            }
            Op::AdminRewrite { back, text } => {
                let Some(fid) = self.nth_output(*back) else { return };
                let Ok(out) = self.book.begin_force(CH, fid) else { return };
                // `batch::write_over`: the admin's text as if generated.
                let body = BOT_TEXTS[*text % BOT_TEXTS.len()].to_string();
                let history = out.replay.channel_reply || out.replay.asked.as_ref().is_some_and(|a| self.persona.remembers(a.slash));
                let (held, turn) = self.rewrite_messages(&out, &[(body.clone(), Some(body))]);
                if history && let Some(tied) = self.tie_gate(&out.replay) {
                    self.write_turn(&out.messages.iter().map(|(id, _)| *id).collect::<Vec<_>>(), &turn, tied);
                }
                self.finish_force(held, turn, &out);
                self.book.end_force(CH);
            }
            Op::AdminDelete { back } => {
                let Some(fid) = self.nth_output(*back) else { return };
                let Ok(out) = self.book.begin_force(CH, fid) else { return };
                for (id, _) in &out.messages {
                    self.posted.retain(|m| m.id != *id);
                    self.book.unnote_message(CH, *id); // MessageDelete
                    self.window.remove(*id);
                }
                if self.book.last(CH).is_some_and(|o| o.finished_id == fid) {
                    self.undo_depth = 0;
                }
                self.book.remove(CH, fid);
                self.book.end_force(CH);
            }
            Op::Shake { text } => {
                let content = BOT_TEXTS[*text % BOT_TEXTS.len()].to_string();
                let id = self.bot_post(content.clone());
                self.window.push(
                    WindowMsg { id, author_id: BOT, author: BOT_NAME.into(), text: content, extras: Default::default(), mentions: Vec::new() },
                    MAX_BUFFER,
                );
            }
            Op::Reset => {
                let anchor = self.id();
                self.window = ChannelWindow::after(anchor);
                self.book.forget(CH);
                self.undo_depth = 0;
            }
            Op::OptOut { who } => {
                let (user, _) = USERS[*who % USERS.len()];
                self.optout.insert(user);
                self.window.remove_author(user);
                self.book.forget_author(user);
            }
            Op::OptIn { who } => {
                let (user, _) = USERS[*who % USERS.len()];
                if self.optout.remove(&user) {
                    // Visible from now on; what they said while out stays
                    // hidden to a cold start (`OptOuts::hidden`).
                    let from = self.id();
                    self.visible_from.insert(user, from);
                }
            }
            Op::SelectPrompt => {
                self.window.select(20, &|_| true);
            }
            Op::Commit => self.commit(),
            Op::BatchStart { nth_user } => {
                let user = self.retrier(*nth_user);
                // Begun as `/retry`; the versions generate outside the
                // channel's writing count.
                if let Ok(out) = self.book.begin(CH, user, self.clock) {
                    self.batch = Some(out);
                }
            }
            Op::BatchPick { text, parts } => {
                let Some(out) = self.batch.take() else { return };
                // `still_last`: the rules again, bar the time limit.
                let got = self.book.still_last(CH, &out);
                if !self.diverged {
                    let want = self.expected_pick(&out);
                    if got != want {
                        self.failed = Some(format!("batch pick: book says {got:?}, the channel says {want:?}"));
                    }
                }
                if got.is_ok() {
                    // `write_over`: the picked text streams like a retry.
                    self.book.started(CH);
                    let body = BOT_TEXTS[*text % BOT_TEXTS.len()].to_string();
                    let parts = Self::parts(&body, Some(&body), *parts, *text);
                    self.redo(&out, &parts);
                }
                self.book.end(CH); // the Lock's drop
            }
            Op::BatchCancel => {
                if self.batch.take().is_some() {
                    self.book.end(CH);
                }
            }
            Op::OtherBot { text } => {
                let (bot, name) = OTHER_BOT;
                let text = USER_TEXTS[*text % USER_TEXTS.len()];
                let id = self.id();
                self.posted.push(Posted { id, author: bot, content: text.into(), interaction: None, mentions: Vec::new(), reply_to: None });
                let ignored = crate::chat::gate::is_ignored_text(text);
                self.book.gateway_message(CH, id, bot, BOT, ignored, false);
                self.activity = self.activity.max(id);
                if !ignored {
                    self.window.push(
                        WindowMsg { id, author_id: bot, author: name.into(), text: text.into(), extras: Default::default(), mentions: Vec::new() },
                        MAX_BUFFER,
                    );
                }
            }
            Op::Wait { secs } => self.clock += std::time::Duration::from_secs(*secs),
        }
    }

    /// `finished_replacing` for an admin rewrite.
    fn finish_force(&mut self, held: Vec<(u64, String)>, turn: Vec<(u64, String)>, out: &Output<Rep>) {
        self.book.started(CH); // write_over streams: it counts as writing
        let output = (!held.is_empty()).then(|| Output {
            replay: out.replay.clone(),
            requester: out.requester,
            messages: held,
            turn,
            finished_at: self.clock,
            finished_id: self.id(),
        });
        // Rewriting the last output clears its undo stack.
        if output.is_some() && self.book.last(CH).is_some_and(|o| o.finished_id == out.finished_id) {
            self.undo_depth = 0;
        }
        self.book.finished_replacing(CH, output, out.finished_id);
    }

    /// The finish id of the output `back` outputs before the newest the book
    /// still remembers, probed through `find` on the posted messages.
    fn nth_output(&self, back: usize) -> Option<u64> {
        let mut seen = Vec::new();
        for m in self.posted.iter().rev() {
            if m.author == BOT
                && let Some(out) = self.book.find(CH, m.id)
                && !seen.contains(&out.finished_id)
            {
                seen.push(out.finished_id);
            }
        }
        seen.get(back % seen.len().max(1)).copied()
    }

    /// The newest `fetch` posted messages, as REST returns them (newest first).
    fn rest_messages(&self, fetch: usize) -> Vec<serenity::Message> {
        self.posted
            .iter()
            .rev()
            .take(fetch)
            .map(|p| serde_json::from_value(self.message_json(p)).expect("a message"))
            .collect()
    }

    /// The known seeding gap shows: a card's part outlived its card (which
    /// Discord then no longer names), or was posted with it gone.
    fn seed_gap(&self) -> bool {
        self.card_gone || self.posted.iter().any(|p| p.reply_to.is_some_and(|to| !self.posted.iter().any(|r| r.id == to)))
    }

    /// Posted message `p` as Discord's JSON.
    fn message_json(&self, p: &Posted) -> serde_json::Value {
        let user_json = |id: u64, bot: bool| {
            serde_json::json!({
                "id": id.to_string(),
                "username": name_of(id).to_lowercase(),
                "global_name": if id == BOT { BOT_NAME } else { name_of(id) },
                "discriminator": "0",
                "bot": bot,
            })
        };
        let mentions: Vec<_> = p.mentions.iter().map(|&u| user_json(u, u == BOT)).collect();
        let mut v = serde_json::json!({
            "id": p.id.to_string(),
            "channel_id": CH.to_string(),
            "type": if p.interaction.is_some_and(|(.., o)| o.is_none()) { 20 } else { 0 },
            "content": p.content,
            "author": user_json(p.author, p.author == BOT || p.author == OTHER_BOT.0),
            "timestamp": "2026-10-17T00:00:00Z",
            "tts": false,
            "mention_everyone": false,
            "mentions": mentions,
            "mention_roles": [],
            "attachments": [],
            "embeds": [],
            "pinned": false,
        });
        if let Some(to) = p.reply_to {
            v["message_reference"] = serde_json::json!({ "message_id": to.to_string(), "channel_id": CH.to_string() });
            // As Discord fetches it: the replied-to message, unless deleted.
            v["referenced_message"] = self.posted.iter().find(|r| r.id == to).map_or(serde_json::Value::Null, |r| self.message_json(r));
        }
        if let Some((interaction, requester, original)) = p.interaction {
            let mut meta = serde_json::json!({
                "id": interaction.to_string(),
                "type": 2,
                "user": user_json(requester, false),
                "authorizing_integration_owners": {},
            });
            if let Some(o) = original {
                meta["original_response_message_id"] = o.to_string().into();
            }
            v["interaction_metadata"] = meta;
        }
        v
    }

    /// A cold start over the newest `fetch` posted messages, at the live
    /// window's anchor (the real bot fetches `history_messages.min(100)`).
    fn cold(&self, fetch: usize) -> ChannelWindow {
        let mut fresh = ChannelWindow::after(self.window.anchor().unwrap_or(999));
        let hidden = |u: u64, id: u64| self.hidden(u, id);
        let ctx = SeedCtx { persona: &self.persona, bot_roles: &[], bot_name: BOT_NAME, hidden: &hidden, max: MAX_BUFFER };
        seed_messages(&ctx, BOT, &HashSet::new(), &mut fresh, &self.rest_messages(fetch));
        fresh
    }

    /// The oldest posted id a `fetch`-deep cold start sees (0: all of them).
    fn horizon(&self, fetch: usize) -> u64 {
        if self.posted.len() <= fetch {
            return 0;
        }
        self.posted[self.posted.len() - fetch].id
    }

    fn snapshot(w: &ChannelWindow) -> Vec<(u64, u64, String, String)> {
        w.messages().map(|m| (m.id, m.author_id, m.author.clone(), m.text.clone())).collect()
    }

    /// What `/retry` by `user` must answer now, worked out from what the
    /// channel shows (`posted`), not from the book's own counts: the last
    /// output's identity is the book's, everything after it is the log's;
    /// the time is the sim's clock (`Wait`).
    fn expected_retry(&self, user: u64) -> Result<(), Refusal> {
        if self.locked() {
            return Err(Refusal::Retrying);
        }
        let last = self.book.last(CH).ok_or(Refusal::Nothing)?;
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if self.clock.saturating_duration_since(last.finished_at) > self.book.rules().window() {
            return Err(Refusal::TooLate);
        }
        self.expected_since(last)
    }

    /// `/undo-retry`: as `/retry`, with something retried to undo.
    fn expected_undo(&self, user: u64) -> Result<(), Refusal> {
        if self.locked() {
            return Err(Refusal::Retrying);
        }
        let last = self.book.last(CH).filter(|_| self.undo_depth > 0).ok_or(Refusal::Nothing)?;
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if self.clock.saturating_duration_since(last.finished_at) > self.book.rules().window() {
            return Err(Refusal::TooLate);
        }
        self.expected_since(last)
    }

    /// `/delete-reply`: nothing at all since (any message, seen or not, any
    /// counted command, any post of the bot's but the output's own).
    fn expected_delete(&self, user: u64) -> Result<(), Refusal> {
        if self.locked() {
            return Err(Refusal::Retrying);
        }
        let last = self.book.last(CH).ok_or(Refusal::Nothing)?;
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if self.pending.is_some() {
            return Err(Refusal::Writing);
        }
        if self.activity > last.finished_id || self.bot_posted_since(last) {
            return Err(Refusal::NotLast);
        }
        Ok(())
    }

    /// The batch pick of `out`: still the last output, and the "since"
    /// rules; not the time limit.
    fn expected_pick(&self, out: &Output<Rep>) -> Result<(), Refusal> {
        let last = self.book.last(CH).ok_or(Refusal::Nothing)?;
        if last.finished_id != out.finished_id {
            return Err(Refusal::NotLast);
        }
        self.expected_since(last)
    }

    /// A retry, undo or batch holds the channel's lock.
    fn locked(&self) -> bool {
        self.batch.is_some() || matches!(self.pending, Some(Pending { kind: PendingKind::Retry { .. }, .. }))
    }

    /// A message of the bot's newer than `last`'s first finish that isn't
    /// one of its own is still in the channel.
    fn bot_posted_since(&self, last: &Output<Rep>) -> bool {
        let own = |id: u64| last.messages.iter().any(|(m, _)| *m == id);
        self.posted.iter().any(|m| m.id > last.finished_id && m.author == BOT && !own(m.id))
    }

    /// Rules 1 and 2 and nothing being written, from the channel's log.
    fn expected_since(&self, last: &Output<Rep>) -> Result<(), Refusal> {
        if self.pending.is_some() {
            return Err(Refusal::Writing);
        }
        if self.bot_posted_since(last) {
            return Err(Refusal::NotLast);
        }
        let after = |m: &&Posted| m.id > last.finished_id;
        let seen = self
            .posted
            .iter()
            .filter(after)
            .filter(|m| m.author != BOT && !crate::chat::gate::is_ignored_text(&m.content) && !self.hidden(m.author, m.id))
            .count();
        if seen > self.book.rules().max_newer {
            return Err(Refusal::TooMuchSince);
        }
        Ok(())
    }

    /// The invariants that must hold after every op.
    fn check_invariants(&self) -> Result<(), String> {
        // Every user's `/retry` verdict matches the log (not once an event
        // was lost: then the book rightly follows what the bot saw).
        if let Some(e) = &self.failed {
            return Err(e.clone());
        }
        if !self.diverged {
            for &(user, _) in USERS {
                let verdicts = [
                    ("retry", self.book.can_retry(CH, user, self.clock), self.expected_retry(user)),
                    ("undo", self.book.can_undo(CH, user, self.clock), self.expected_undo(user)),
                    ("delete", self.book.can_delete(CH, user), self.expected_delete(user)),
                ];
                for (what, got, want) in verdicts {
                    if got != want {
                        return Err(format!("{what} by {user}: book says {got:?}, the channel says {want:?}"));
                    }
                }
            }
        }
        let msgs = Self::snapshot(&self.window);
        for pair in msgs.windows(2) {
            if pair[0].0 >= pair[1].0 {
                return Err(format!("ids out of order: {} then {}", pair[0].0, pair[1].0));
            }
        }
        if let Some(a) = self.window.anchor()
            && msgs.first().is_some_and(|m| m.0 <= a)
        {
            return Err("a message at or before the anchor".into());
        }
        for (id, author, ..) in &msgs {
            if self.optout.contains(author) {
                return Err(format!("message {id} by opted-out user {author}"));
            }
        }
        Ok(())
    }
}

/// `PROPTEST_CASES` if set (for a long run), else `default`.
/// Retry rules around the defaults: a short or long window against `Wait`'s
/// 1..200 s, no newer messages allowed up to many, a shallow undo stack.
fn rules_strategy() -> impl Strategy<Value = crate::config::RetryRules> {
    let default = crate::config::RetryRules::default();
    prop_oneof![
        2 => Just(default.clone()),
        1 => (prop::sample::select(vec![60u64, 300, 1000]), 0usize..8, 1usize..4).prop_map(move |(window_secs, max_newer, undo_depth)| {
            crate::config::RetryRules { window_secs, max_newer, undo_depth, ..default.clone() }
        }),
    ]
}

fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(64), ..ProptestConfig::default() })]

    /// Random ops and faults; after each, the invariants; at the end, a cold
    /// seed equals the live window unless a gateway event was lost, and with
    /// one it is at least deterministic.
    #[test]
    fn a_simulated_channel_seeds_back_to_its_live_window(
        ops in proptest::collection::vec(op_strategy(), 1..40),
        all in any::<bool>(),
        rules in rules_strategy(),
    ) {
        let listed: Vec<String> = if all {
            crate::persona::REMEMBERABLE.iter().map(|s| s.to_string()).collect()
        } else {
            ["ask", "roast", "gamble"].iter().map(|s| s.to_string()).collect()
        };
        let mut sim = Sim::new(listed, rules);
        for op in &ops {
            sim.apply(op);
            if let Err(e) = sim.check_invariants() {
                prop_assert!(false, "{e}\nops so far:\n{}", sim.log.join("\n"));
            }
        }
        sim.commit(); // flush a deferred generation
        if let Err(e) = sim.check_invariants() {
            prop_assert!(false, "{e}\nops:\n{}", sim.log.join("\n"));
        }
        let cold = Sim::snapshot(&sim.cold(usize::MAX));
        if sim.diverged || sim.seed_gap() {
            // Lost events: cold and warm may differ, but a restart heals to
            // one deterministic answer.
            let again = Sim::snapshot(&sim.cold(usize::MAX));
            prop_assert_eq!(cold, again, "seeding must be deterministic");
        } else {
            let warm = Sim::snapshot(&sim.window);
            prop_assert_eq!(
                cold, warm,
                "cold seed != live window\nops:\n{}", sim.log.join("\n")
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(1000), ..ProptestConfig::default() })]

    /// Long chats: the window outgrows `history_messages` (anchor jumps land
    /// inside exchanges) and the posted channel outgrows the REST fetch. A
    /// full-fetch cold start still equals the live window, and a real-depth
    /// one (the newest 20) equals the live window's matching part: every
    /// entry at or after the fetch horizon, plus the requester turns whose
    /// exchange starts there.
    #[test]
    fn a_long_chat_seeds_its_suffix(
        ops in proptest::collection::vec(op_strategy(), 60..120),
        all in any::<bool>(),
    ) {
        let listed: Vec<String> = if all {
            crate::persona::REMEMBERABLE.iter().map(|s| s.to_string()).collect()
        } else {
            ["ask", "roast", "gamble"].iter().map(|s| s.to_string()).collect()
        };
        let mut sim = Sim::new(listed, Default::default());
        for op in &ops {
            sim.apply(op);
            if let Err(e) = sim.check_invariants() {
                prop_assert!(false, "{e}\nops so far:\n{}", sim.log.join("\n"));
            }
        }
        sim.commit();
        const FETCH: usize = 20;
        if sim.diverged || sim.seed_gap() {
            for fetch in [usize::MAX, FETCH] {
                let (one, two) = (Sim::snapshot(&sim.cold(fetch)), Sim::snapshot(&sim.cold(fetch)));
                prop_assert_eq!(one, two, "seeding must be deterministic");
            }
            return Ok(());
        }
        let warm = Sim::snapshot(&sim.window);
        let full = Sim::snapshot(&sim.cold(usize::MAX));
        prop_assert_eq!(
            &full, &warm,
            "full cold seed != live window\nops:\n{}", sim.log.join("\n")
        );
        let h = sim.horizon(FETCH);
        let cold = Sim::snapshot(&sim.cold(FETCH));
        // An exchange lives and dies whole: a card's later message (a
        // follow-up, a retry's extra part) goes with a card past the horizon.
        let card_cut = |id: u64| {
            let card = |p: &Posted| p.reply_to.or(p.interaction.and_then(|(.., original)| original));
            sim.posted.iter().find(|p| p.id == id).and_then(card).is_some_and(|card| card < h)
        };
        let expected: Vec<_> = warm
            .into_iter()
            .filter(|(id, ..)| {
                (*id >= h && !card_cut(*id)) || sim.interactions.get(id).is_some_and(|&header| header >= h)
            })
            .collect();
        prop_assert_eq!(
            cold, expected,
            "horizon-deep cold seed != live window's suffix (horizon {})\nops:\n{}", h, sim.log.join("\n")
        );
    }
}
