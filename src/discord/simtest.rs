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
use crate::chat::retry::{Book, Output};
use crate::chat::window::{ChannelWindow, WindowMsg};
use crate::persona::Persona;

const BOT: u64 = 99;
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
    Retry { nth_user: usize, fault: Fault, defer: bool },
    Undo { nth_user: usize },
    DeleteReply { nth_user: usize },
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
        3 => (0usize..3, fault, any::<bool>()).prop_map(|(nth_user, fault, defer)| Op::Retry { nth_user, fault, defer }),
        1 => (0usize..3).prop_map(|nth_user| Op::Undo { nth_user }),
        1 => (0usize..3).prop_map(|nth_user| Op::DeleteReply { nth_user }),
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
    /// A /retry of `out`; still_quiet reruns at commit, as live after QUIET.
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
    log: Vec<String>,
}

/// A serenity User for one of the sim's fixed users (or the bot).
fn user_of(id: u64) -> serenity::User {
    serde_json::from_value(serde_json::json!({
        "id": id.to_string(),
        "username": name_of(id).to_lowercase(),
        "global_name": if id == BOT { BOT_NAME } else { name_of(id) },
        "discriminator": "0",
        "bot": id == BOT,
    }))
    .expect("a user")
}

fn name_of(user: u64) -> &'static str {
    USERS.iter().find(|(id, _)| *id == user).map(|(_, n)| *n).unwrap_or("someone")
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
    fn new(remember_commands: Vec<String>) -> Sim {
        let mut persona = crate::persona::test_persona();
        persona.remember_commands = remember_commands;
        Sim {
            next_id: 1000,
            posted: Vec::new(),
            window: ChannelWindow::after(999),
            book: Book::default(),
            persona,
            optout: HashSet::new(),
            visible_from: std::collections::HashMap::new(),
            pending: None,
            interactions: std::collections::HashMap::new(),
            diverged: false,
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

    fn now() -> std::time::Instant {
        std::time::Instant::now()
    }

    /// A user's message arrives (gateway): activity, then the window, unless
    /// they are invisible (`//`, opted out).
    fn user_msg(&mut self, who: usize, text: &str) -> u64 {
        let (user, name) = USERS[who % USERS.len()];
        let id = self.id();
        self.posted.push(Posted { id, author: user, content: text.into(), interaction: None, mentions: Vec::new() });
        self.book.note_activity(CH, id);
        if !crate::chat::gate::is_ignored_text(text) && !self.optout.contains(&user) {
            self.window.push(
                WindowMsg { id, author_id: user, author: name.into(), text: text.into(), extras: Default::default(), mentions: Vec::new() },
                MAX_BUFFER,
            );
        }
        id
    }

    /// The bot posts a plain message (a reply part, shake text, a fallback).
    fn bot_post(&mut self, content: String) -> u64 {
        let id = self.id();
        self.posted.push(Posted { id, author: BOT, content, interaction: None, mentions: Vec::new() });
        id
    }

    /// The bot answers an interaction (the first message or a follow-up).
    fn bot_answer(&mut self, content: String, interaction: u64, requester: u64, original: Option<u64>, mentions: Vec<u64>) -> u64 {
        let id = self.id();
        self.posted.push(Posted { id, author: BOT, content, interaction: Some((interaction, requester, original)), mentions });
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
            finished_at: Self::now(),
            finished_id: self.id(),
        });
        match replaces {
            Some(old) => self.book.finished_replacing(CH, output, old),
            None => self.book.finished(CH, output, retried),
        }
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
                    Fault::Dropped | Fault::QueueFull => {
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
                    self.posted.push(Posted { id, author: BOT, content: content.clone(), interaction: None, mentions: vec![requester] });
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
                // Rule 2: still quiet at commit, as live after QUIET.
                if self.book.still_quiet(CH, &out).is_err() || matches!(p.fault, Fault::Dropped | Fault::QueueFull | Fault::PostFails) {
                    self.pending = None;
                    self.book.finished(CH, None, true);
                    self.book.end(CH);
                    return;
                }
                let Some((posted_text, body)) = self.generated(&p.fault, p.texts) else { return };
                let (held, turn) = self.rewrite_messages(&out, &posted_text, body.as_deref());
                if let Some(tied) = self.tie_gate(&out.replay) {
                    self.write_turn(&out.messages.iter().map(|(id, _)| *id).collect::<Vec<_>>(), &turn, tied);
                }
                let rep = out.replay.clone();
                self.finish(held, turn, rep, true, None);
                self.book.end(CH);
                self.pending = None;
            }
        }
    }

    /// Writes new text over an output's posted messages, as `Reuse` does: the
    /// first message keeps its header, leftovers are deleted (their gateway
    /// delete events arrive, which live is a no-op for the window).
    fn rewrite_messages(&mut self, out: &Output<Rep>, posted_text: &str, body: Option<&str>) -> (Texts, Texts) {
        let mut held = Vec::new();
        let mut turn = Vec::new();
        let mut ids = out.messages.iter().map(|(id, _)| *id);
        let first = ids.next().expect("an output is never empty");
        let content = match &out.replay.header {
            Some(h) => format!("{h}\n{posted_text}"),
            None => posted_text.to_string(),
        };
        if let Some(m) = self.posted.iter_mut().find(|m| m.id == first) {
            m.content = content;
            // `Output.messages` holds the shown text, header not included.
            held.push((first, posted_text.to_string()));
            if let Some(b) = body {
                turn.push((first, b.to_string()));
            }
        }
        // Leftover messages of the old output are deleted.
        for id in ids {
            self.posted.retain(|m| m.id != id);
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
                // The gateway handler (MessageUpdate).
                if self.window.author_of(id) == Some(BOT) {
                    return;
                }
                if crate::chat::gate::is_ignored_text(&new) {
                    self.window.remove(id);
                } else if !self.window.contains(id) {
                    // Edited back out of a `//` note: visible again — unless
                    // it is from before the author's opt-in (`hidden`).
                    let author = self.posted.iter().find(|m| m.id == id).map(|m| m.author).unwrap_or(0);
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
                self.window.remove(id); // MessageDelete
            }
            Op::ChannelReply { texts, parts, fault, defer } => {
                if self.pending.is_some() {
                    return; // one generation at a time
                }
                self.book.started(CH);
                self.pending = Some(Pending { kind: PendingKind::Channel, fault: fault.clone(), texts: *texts, parts: *parts });
                if !defer {
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
                self.book.started(CH);
                self.pending = Some(Pending {
                    kind: PendingKind::Command { header, interaction, requester, asked, is_ask, mentions },
                    fault: fault.clone(),
                    texts: *text,
                    parts: 1,
                });
                if !defer {
                    self.commit();
                }
            }
            Op::Retry { nth_user, fault, defer } => {
                if self.pending.is_some() {
                    return; // Refusal::Writing, via the book's writing count
                }
                let (user, _) = USERS[*nth_user % USERS.len()];
                let Ok(out) = self.book.begin(CH, user, Self::now()) else { return };
                self.pending = Some(Pending { kind: PendingKind::Retry { out }, fault: fault.clone(), texts: *nth_user, parts: 1 });
                if !defer {
                    self.commit();
                }
            }
            Op::Undo { nth_user } => {
                let (user, _) = USERS[*nth_user % USERS.len()];
                let Ok((cur, prev)) = self.book.begin_undo(CH, user, Self::now()) else { return };
                // `reply::restore`: prev's texts go into cur's messages.
                let posted = prev.messages.first().map(|(_, t)| t.clone()).unwrap_or_default();
                let body = prev.turn.first().map(|(_, t)| t.clone());
                let (held, turn) = self.rewrite_messages(&cur, &posted, body.as_deref());
                if let Some(tied) = self.tie_gate(&cur.replay) {
                    self.write_turn(&cur.messages.iter().map(|(id, _)| *id).collect::<Vec<_>>(), &turn, tied);
                }
                let restored = (!held.is_empty()).then(|| Output {
                    replay: cur.replay.clone(),
                    requester: cur.requester,
                    messages: held,
                    turn,
                    finished_at: Self::now(),
                    finished_id: self.id(),
                });
                self.book.end_undo(CH, restored);
            }
            Op::DeleteReply { nth_user } => {
                let (user, _) = USERS[*nth_user % USERS.len()];
                let Ok(out) = self.book.begin_delete(CH, user) else { return };
                if self.book.still_quiet(CH, &out).is_err() {
                    self.book.end_delete(CH, false);
                    return;
                }
                for (id, _) in &out.messages {
                    self.posted.retain(|m| m.id != *id);
                    self.window.remove(*id);
                }
                self.book.end_delete(CH, true);
            }
            Op::AdminRewrite { back, text } => {
                let Some(fid) = self.nth_output(*back) else { return };
                let Ok(out) = self.book.begin_force(CH, fid) else { return };
                // `batch::write_over`: the admin's text as if generated.
                let body = BOT_TEXTS[*text % BOT_TEXTS.len()].to_string();
                let history = out.replay.channel_reply || out.replay.asked.as_ref().is_some_and(|a| self.persona.remembers(a.slash));
                let (held, turn) = self.rewrite_messages(&out, &body, Some(&body));
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
                    self.window.remove(*id);
                }
                self.book.remove(CH, fid);
                self.book.end_force(CH);
            }
            Op::Shake { text } => {
                let content = BOT_TEXTS[*text % BOT_TEXTS.len()].to_string();
                let id = self.bot_post(content.clone());
                self.book.note_activity(CH, id);
                self.window.push(
                    WindowMsg { id, author_id: BOT, author: BOT_NAME.into(), text: content, extras: Default::default(), mentions: Vec::new() },
                    MAX_BUFFER,
                );
            }
            Op::Reset => {
                let anchor = self.id();
                self.window = ChannelWindow::after(anchor);
                self.book.forget(CH);
            }
            Op::OptOut { who } => {
                let (user, _) = USERS[*who % USERS.len()];
                self.optout.insert(user);
                self.window.remove_author(user);
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
            finished_at: Self::now(),
            finished_id: self.id(),
        });
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
            .map(|p| {
                let user_json = |id: u64, bot: bool| {
                    serde_json::json!({
                        "id": id.to_string(),
                        "username": name_of(id).to_lowercase(),
                        "global_name": if bot { BOT_NAME } else { name_of(id) },
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
                    "author": user_json(p.author, p.author == BOT),
                    "timestamp": "2026-10-17T00:00:00Z",
                    "tts": false,
                    "mention_everyone": false,
                    "mentions": mentions,
                    "mention_roles": [],
                    "attachments": [],
                    "embeds": [],
                    "pinned": false,
                });
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
                serde_json::from_value(v).expect("a message")
            })
            .collect()
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

    /// The invariants that must hold after every op.
    fn check_invariants(&self) -> Result<(), String> {
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

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    /// Random ops and faults; after each, the invariants; at the end, a cold
    /// seed equals the live window unless a gateway event was lost, and with
    /// one it is at least deterministic.
    #[test]
    fn a_simulated_channel_seeds_back_to_its_live_window(
        ops in proptest::collection::vec(op_strategy(), 1..40),
        all in any::<bool>(),
    ) {
        let listed: Vec<String> = if all {
            crate::persona::REMEMBERABLE.iter().map(|s| s.to_string()).collect()
        } else {
            ["ask", "roast", "gamble"].iter().map(|s| s.to_string()).collect()
        };
        let mut sim = Sim::new(listed);
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
        if sim.diverged {
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
    #![proptest_config(ProptestConfig { cases: 1000, ..ProptestConfig::default() })]

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
        let mut sim = Sim::new(listed);
        for op in &ops {
            sim.apply(op);
            if let Err(e) = sim.check_invariants() {
                prop_assert!(false, "{e}\nops so far:\n{}", sim.log.join("\n"));
            }
        }
        sim.commit();
        const FETCH: usize = 20;
        if sim.diverged {
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
        let expected: Vec<_> = warm
            .into_iter()
            .filter(|(id, ..)| {
                *id >= h || sim.interactions.get(id).is_some_and(|&header| header >= h)
            })
            .collect();
        prop_assert_eq!(
            cold, expected,
            "horizon-deep cold seed != live window's suffix (horizon {})\nops:\n{}", h, sim.log.join("\n")
        );
    }
}
