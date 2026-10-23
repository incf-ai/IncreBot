//! `/retry` bookkeeping (design.md §6): per channel, the last generated output
//! and whether anyone has spoken since, so a retry only replaces a reply that is
//! still the newest thing in the channel.
//!
//! A retry is refused unless, in that channel:
//! 1. it is the newest output, and no message of the bot's is newer than its
//!    first finish but its own (a `/shake` message, a card left alone, a
//!    message of a reply that left no output; a deleted or ephemeral one
//!    doesn't count);
//! 2. at most `max_newer` messages came after it that the bot can see (in
//!    context: not `//`, not opted-out users'; deleted, edited into `//` or
//!    opted out since, one no longer counts),
//!    timestamped after the output first finished (what was said while it
//!    streamed doesn't count; commands don't count);
//! 3. the output first finished at most `window_secs` ago (a retry or undo
//!    keeps the original's clock);
//! 4. no other retry is running or waiting here.
//!
//! How often anyone may retry at all is `chat::limits`'s business.
//!
//! Only the original requester may retry, and not while another reply is being
//! written in the channel (then "the previous output" is ambiguous). Activity is
//! tracked by snowflake (message or interaction ID), whose timestamp Discord
//! sets, against the finish time taken as a snowflake from the bot's clock
//! (`window::snowflake_at`); only for channels the bot has written in.
//!
//! Each retry pushes the output it replaced onto a per-channel stack (the
//! original at the bottom); `/undo-retry` pops the newest back into the
//! messages, under the same rules. A new reply clears it.
//!
//! `/delete-reply` deletes the last output under stricter rules: rule 1,
//! nothing at all happened since (any message, `//` and opted-out ones
//! included, or any other command), still so `delete_quiet_ms` later, and rule 4; no
//! time limit.
//! Nothing before it becomes deletable or retryable.
//!
//! The admin's right-click actions (`discord::force`) work on any of the last
//! `admin_recent` outputs here, under none of these rules but the lock: rewritten,
//! an output keeps its place, requester and finish, with no undo.
//!
//! The numbers are `discord.retry` in the configuration ([`RetryRules`]).
//!
//! [`Reuse`] plans how a retry's text goes into the old output's messages, and
//! [`steer`] keeps the outputs a retry is pushed away from.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub use crate::config::RetryRules;

/// How many more of the newest messages (users', and the bot's) than
/// `max_newer` a channel keeps the IDs of: enough to count past it after some
/// are deleted.
const KEPT_SPARE: usize = 27;

/// A finished output that can be generated again.
#[derive(Clone, Debug)]
pub struct Output<R> {
    /// What to replay (the job's source, card and so on).
    pub replay: R,
    pub requester: u64,
    /// The messages holding it, in order, with the text each shows (a card
    /// header not included). Never empty.
    pub messages: Vec<(u64, String)>,
    /// The same messages' text as the history window has it: no notices, and a
    /// notice-only message left out.
    pub turn: Vec<(u64, String)>,
    /// When it first finished: a retry or undo keeps the original's.
    pub finished_at: Instant,
    /// The first finish as a snowflake: activity with a larger ID came after
    /// it. Also what names the output among the remembered ones.
    pub finished_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Nothing,
    NotYours,
    TooLate,
    NotLast,
    /// More than `max_newer` messages since (retry and undo only).
    TooMuchSince,
    Writing,
    Retrying,
}

impl Refusal {
    /// The ephemeral reason the caller sees, with the limits in `rules`.
    pub fn message(self, rules: &RetryRules) -> String {
        match self {
            Refusal::Nothing => "There's nothing to retry here.".into(),
            Refusal::NotYours => "Only the person who asked for that reply can retry it.".into(),
            Refusal::TooLate => format!("Too late: a reply can only be retried within {}.", span(rules.window_secs)),
            Refusal::NotLast => "Something newer was posted after that reply, so it can't be retried.".into(),
            Refusal::TooMuchSince => format!(
                "Too much was said after that reply (over {}), so it can't be retried.",
                plural(rules.max_newer as u64, "message")
            ),
            Refusal::Writing => "A reply is still being written here; try again once it's done.".into(),
            Refusal::Retrying => "A retry or undo is already running here.".into(),
        }
    }

    /// The reason for a refused `/undo-retry`.
    pub fn undo_message(self, rules: &RetryRules) -> String {
        match self {
            Refusal::Nothing => "There's no retry to undo here.".into(),
            Refusal::NotYours => "Only the person who asked for that reply can undo its retry.".into(),
            Refusal::TooLate => format!("Too late: a retry can only be undone within {}.", span(rules.window_secs)),
            Refusal::NotLast => "Something newer was posted after that reply, so its retry can't be undone.".into(),
            Refusal::TooMuchSince => format!(
                "Too much was said after that reply (over {}), so its retry can't be undone.",
                plural(rules.max_newer as u64, "message")
            ),
            r => r.message(rules),
        }
    }

    /// The reason for a refused `/delete-reply`.
    pub fn delete_message(self, rules: &RetryRules) -> String {
        match self {
            Refusal::Nothing => "There's no reply to delete here.".into(),
            Refusal::NotYours => "Only the person who asked for that reply can delete it.".into(),
            Refusal::NotLast => "Someone posted after that reply, so it can't be deleted.".into(),
            Refusal::Retrying => "A retry or undo is running here; try again once it's done.".into(),
            r => r.message(rules),
        }
    }
}

/// `secs` for a person: whole minutes as minutes, else seconds.
fn span(secs: u64) -> String {
    match secs {
        s if s >= 60 && s % 60 == 0 => plural(s / 60, "minute"),
        s => plural(s, "second"),
    }
}

fn plural(n: u64, unit: &str) -> String {
    match n {
        1 => format!("1 {unit}"),
        n => format!("{n} {unit}s"),
    }
}

struct Channel<R> {
    /// The newest foreign activity seen (a snowflake), 0 if none.
    activity: u64,
    /// The newest messages in context from anyone but the bot (id, author),
    /// by id, at most `max_newer` + [`KEPT_SPARE`].
    said: VecDeque<(u64, u64)>,
    /// The bot's newest messages still up here, by id, as many.
    posts: VecDeque<u64>,
    /// Replies being written or waiting here.
    writing: usize,
    last: Option<Output<R>>,
    /// Earlier outputs, newest first; with `last`, at most `admin_recent`.
    recent: VecDeque<Output<R>>,
    /// The outputs retries replaced, oldest (the original) first.
    undo: Vec<Output<R>>,
    /// A retry or undo is running.
    retrying: bool,
}

impl<R> Default for Channel<R> {
    fn default() -> Self {
        Self {
            activity: 0,
            said: VecDeque::new(),
            posts: VecDeque::new(),
            writing: 0,
            last: None,
            recent: VecDeque::new(),
            undo: Vec::new(),
            retrying: false,
        }
    }
}

pub struct Book<R> {
    channels: HashMap<u64, Channel<R>>,
    rules: RetryRules,
}

impl<R> Default for Book<R> {
    fn default() -> Self {
        Self::new(RetryRules::default())
    }
}

impl<R> Book<R> {
    pub fn new(rules: RetryRules) -> Self {
        Self {
            channels: HashMap::new(),
            rules,
        }
    }

    /// The limits it applies.
    pub fn rules(&self) -> &RetryRules {
        &self.rules
    }
}

impl<R: Clone> Book<R> {
    /// The gateway delivered message `id` by `author` (the live handler and
    /// the simulator both go through here). The bot's own (`bot`) is a post
    /// for rule 1; anyone else's blocks `/delete-reply`, and counts for rule
    /// 2 unless `ignored` (a `//` message) or `hidden` (its author opted out).
    pub fn gateway_message(&mut self, channel: u64, id: u64, author: u64, bot: u64, ignored: bool, hidden: bool) {
        if author == bot {
            self.note_bot_post(channel, id);
            return;
        }
        self.note_activity(channel, id);
        if !ignored && !hidden {
            self.note_message(channel, id, author);
        }
    }

    /// The gateway delivered an edit of message `id`, now `ignored` (a `//`
    /// message) or not. Into `//`, it stops counting; out of it, it counts
    /// again for its `author` unless `hidden`. Edit events may not say who
    /// wrote it (`None`): then only the first part applies. The bot's own
    /// edits change nothing.
    pub fn gateway_edit(&mut self, channel: u64, id: u64, author: Option<u64>, bot: u64, ignored: bool, hidden: bool) {
        if author == Some(bot) {
            return;
        }
        if ignored {
            self.unnote_message(channel, id);
        } else if let Some(author) = author
            && !hidden
        {
            self.note_message(channel, id, author);
        }
    }

    /// A message from anyone but the bot, or a command other than `/retry`:
    /// what blocks `/delete-reply`.
    pub fn note_activity(&mut self, channel: u64, id: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.activity = c.activity.max(id);
        }
    }

    /// A message in context (not `//`, not opted out) by `author`, anyone
    /// but the bot: rule 2 counts these. Noting one twice is harmless.
    pub fn note_message(&mut self, channel: u64, id: u64, author: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            insert_newest(&mut c.said, (id, author), |&(m, _)| m, kept(&self.rules));
        }
    }

    /// Message `id` was deleted, or edited into a `//` one: neither rule
    /// counts it any more.
    pub fn unnote_message(&mut self, channel: u64, id: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.said.retain(|&(m, _)| m != id);
            c.posts.retain(|&m| m != id);
        }
    }

    /// `user` opted out: what they said no longer counts, anywhere.
    pub fn forget_author(&mut self, user: u64) {
        for c in self.channels.values_mut() {
            c.said.retain(|&(_, a)| a != user);
        }
    }

    /// The bot posted message `id` here (seen by the gateway, or noted at
    /// once where it posts outside an output): rule 1 refuses an output
    /// older than it, unless it is part of that output. Noting one twice is
    /// harmless.
    pub fn note_bot_post(&mut self, channel: u64, id: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            insert_newest(&mut c.posts, id, |&m| m, kept(&self.rules));
        }
    }

    /// The messages of `old` that `new` no longer holds (a retry's or
    /// undo's leftovers, deleted by it) stop counting at once, before the
    /// gateway's delete events arrive.
    fn drop_leftovers(posts: &mut VecDeque<u64>, old: &Output<R>, new: &Output<R>) {
        posts.retain(|id| {
            !old.messages.iter().any(|(m, _)| m == id) || new.messages.iter().any(|(m, _)| m == id)
        });
    }

    /// A reply started (it is queued or being written).
    pub fn started(&mut self, channel: u64) {
        self.channels.entry(channel).or_default().writing += 1;
    }

    /// A reply ended; `output` is set when it can be retried (it finished,
    /// even as stopped or failed, rather than being dropped from the queue).
    /// `retried`: it replaced the last output, which goes on the undo stack;
    /// any other output starts a new one.
    pub fn finished(&mut self, channel: u64, output: Option<Output<R>>, retried: bool) {
        let c = self.channels.entry(channel).or_default();
        c.writing = c.writing.saturating_sub(1);
        let Some(mut output) = output else { return };
        match c.last.take() {
            Some(old) if retried => {
                (output.finished_at, output.finished_id) = (old.finished_at, old.finished_id);
                Self::drop_leftovers(&mut c.posts, &old, &output);
                c.undo.push(old);
                if c.undo.len() > self.rules.undo_depth {
                    c.undo.remove(1);
                }
            }
            Some(old) => {
                c.undo.clear();
                c.recent.push_front(old);
                c.recent.truncate(self.rules.admin_recent.saturating_sub(1));
            }
            None => c.undo.clear(),
        }
        c.last = Some(output);
    }

    /// Rules 1–4 for `user`, changing nothing.
    pub fn can_retry(&self, channel: u64, user: u64, now: Instant) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        if c.retrying {
            return Err(Refusal::Retrying);
        }
        let last = c.last.as_ref().ok_or(Refusal::Nothing)?;
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if now.saturating_duration_since(last.finished_at) > self.rules.window() {
            return Err(Refusal::TooLate);
        }
        Self::since(c, last, &self.rules)
    }

    /// Rules 1 and 2 for `output`, and no reply being written here.
    fn since(c: &Channel<R>, output: &Output<R>, rules: &RetryRules) -> Result<(), Refusal> {
        if c.writing > 0 {
            return Err(Refusal::Writing);
        }
        if Self::posted_since(c, output) {
            return Err(Refusal::NotLast);
        }
        if c.said.iter().filter(|&&(id, _)| id > output.finished_id).count() > rules.max_newer {
            return Err(Refusal::TooMuchSince);
        }
        Ok(())
    }

    /// Rule 1's test: a message of the bot's newer than `output`'s first
    /// finish that isn't one of its own.
    fn posted_since(c: &Channel<R>, output: &Output<R>) -> bool {
        c.posts
            .iter()
            .any(|&id| id > output.finished_id && !output.messages.iter().any(|(m, _)| *m == id))
    }

    /// Checks [`can_retry`](Self::can_retry) and, if it passes, marks a retry
    /// as running here and returns the output to replay.
    pub fn begin(&mut self, channel: u64, user: u64, now: Instant) -> Result<Output<R>, Refusal> {
        self.can_retry(channel, user, now)?;
        self.running(channel)
    }

    /// `/undo-retry`'s rules, changing nothing: those of
    /// [`can_retry`](Self::can_retry), and a retry to undo.
    pub fn can_undo(&self, channel: u64, user: u64, now: Instant) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        if c.retrying {
            return Err(Refusal::Retrying);
        }
        let (Some(last), Some(_)) = (&c.last, c.undo.last()) else {
            return Err(Refusal::Nothing);
        };
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if now.saturating_duration_since(last.finished_at) > self.rules.window() {
            return Err(Refusal::TooLate);
        }
        Self::since(c, last, &self.rules)
    }

    /// `/undo-retry`: checks [`can_undo`](Self::can_undo), marks it running
    /// and returns the current output and the one to put back (still on the
    /// stack until [`end_undo`](Self::end_undo)).
    pub fn begin_undo(&mut self, channel: u64, user: u64, now: Instant) -> Result<(Output<R>, Output<R>), Refusal> {
        self.can_undo(channel, user, now)?;
        let last = self.running(channel)?;
        let prev = self.channels.get(&channel).and_then(|c| c.undo.last().cloned()).ok_or(Refusal::Nothing)?;
        Ok((last, prev))
    }

    /// `/delete-reply`'s rules, changing nothing: nothing at all since the
    /// output, nothing being written, no retry running.
    pub fn can_delete(&self, channel: u64, user: u64) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        if c.retrying {
            return Err(Refusal::Retrying);
        }
        let last = c.last.as_ref().ok_or(Refusal::Nothing)?;
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if c.writing > 0 {
            return Err(Refusal::Writing);
        }
        if c.activity > last.finished_id || Self::posted_since(c, last) {
            return Err(Refusal::NotLast);
        }
        Ok(())
    }

    /// `/delete-reply`: checks [`can_delete`](Self::can_delete), marks it
    /// running and returns the output to delete.
    pub fn begin_delete(&mut self, channel: u64, user: u64) -> Result<Output<R>, Refusal> {
        self.can_delete(channel, user)?;
        self.running(channel)
    }

    /// Marks a retry, undo or delete as running here; the last output.
    fn running(&mut self, channel: u64) -> Result<Output<R>, Refusal> {
        let c = self.channels.get_mut(&channel).ok_or(Refusal::Nothing)?;
        let last = c.last.clone().ok_or(Refusal::Nothing)?;
        c.retrying = true;
        Ok(last)
    }

    /// The last output here, if any.
    pub fn last(&self, channel: u64) -> Option<&Output<R>> {
        self.channels.get(&channel)?.last.as_ref()
    }

    /// Message `id` is part of the last output here.
    pub fn is_last(&self, channel: u64, id: u64) -> bool {
        self.channels
            .get(&channel)
            .and_then(|c| c.last.as_ref())
            .is_some_and(|o| o.messages.iter().any(|(m, _)| *m == id))
    }

    /// The delete is over: when `deleted`, the output and its undo stack go.
    pub fn end_delete(&mut self, channel: u64, deleted: bool) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.retrying = false;
            if deleted {
                c.last = None;
                c.undo.clear();
            }
        }
    }

    /// The undo is over: `restored` is the output now shown (it leaves the
    /// stack and is the last output, keeping the original's finish), None if
    /// nothing could be written.
    pub fn end_undo(&mut self, channel: u64, restored: Option<Output<R>>) {
        let Some(c) = self.channels.get_mut(&channel) else {
            return;
        };
        c.retrying = false;
        // Reset meanwhile: nothing from before it comes back.
        let Some(mut restored) = restored else { return };
        if c.undo.pop().is_some() {
            if let Some(cur) = &c.last {
                (restored.finished_at, restored.finished_id) = (cur.finished_at, cur.finished_id);
                Self::drop_leftovers(&mut c.posts, cur, &restored);
            }
            c.last = Some(restored);
        }
    }

    /// `/delete-reply`'s re-check `delete_quiet_ms` after
    /// [`begin_delete`](Self::begin_delete): still nothing since the output,
    /// and no reply started meanwhile.
    pub fn still_quiet(&self, channel: u64, output: &Output<R>) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        if c.writing > 0 {
            return Err(Refusal::Writing);
        }
        if c.activity > output.finished_id || Self::posted_since(c, output) {
            return Err(Refusal::NotLast);
        }
        Ok(())
    }

    /// `/retry-batch`'s OK at the pick: rules 1 and 2 (`output` is still the
    /// last output here, no other reply finished meanwhile), nothing being
    /// written; not the time limit.
    pub fn still_last(&self, channel: u64, output: &Output<R>) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        match &c.last {
            Some(last) if last.finished_id == output.finished_id => Self::since(c, last, &self.rules),
            Some(_) => Err(Refusal::NotLast),
            None => Err(Refusal::Nothing),
        }
    }

    /// The retry is over (generated, refused or never queued).
    pub fn end(&mut self, channel: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.retrying = false;
        }
    }

    /// `/super-admin reset`: nothing from before it can be retried.
    pub fn forget(&mut self, channel: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.last = None;
            c.recent.clear();
            c.undo.clear();
        }
    }

    fn outputs_mut(c: &mut Channel<R>) -> impl Iterator<Item = &mut Output<R>> {
        c.last.iter_mut().chain(c.recent.iter_mut())
    }

    /// The remembered output holding message `id`, if any.
    pub fn find(&self, channel: u64, id: u64) -> Option<Output<R>> {
        let c = self.channels.get(&channel)?;
        c.last
            .iter()
            .chain(c.recent.iter())
            .find(|o| o.messages.iter().any(|(m, _)| *m == id))
            .cloned()
    }

    /// An admin action on the output that finished at `finished_id`: refused
    /// while a retry, undo or delete runs here, or once it is forgotten.
    /// Marks it running and returns the output as it is now.
    pub fn begin_force(&mut self, channel: u64, finished_id: u64) -> Result<Output<R>, Refusal> {
        let c = self.channels.get_mut(&channel).ok_or(Refusal::Nothing)?;
        if c.retrying {
            return Err(Refusal::Retrying);
        }
        let out = Self::outputs_mut(c).find(|o| o.finished_id == finished_id).ok_or(Refusal::Nothing)?.clone();
        c.retrying = true;
        Ok(out)
    }

    /// The output that finished at `finished_id` is still remembered.
    pub fn still_there(&self, channel: u64, finished_id: u64) -> bool {
        self.channels.get(&channel).is_some_and(|c| {
            c.last.iter().chain(c.recent.iter()).any(|o| o.finished_id == finished_id)
        })
    }

    /// The admin action is over.
    pub fn end_force(&mut self, channel: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.retrying = false;
        }
    }

    /// A reply that rewrote the output finished at `old` ended (see
    /// [`finished`](Self::finished)): `output` takes its place, keeping its
    /// requester and finish, so the usual rules for it stay as they were. No
    /// undo: rewriting the last output clears its stack.
    pub fn finished_replacing(&mut self, channel: u64, output: Option<Output<R>>, old: u64) {
        let c = self.channels.entry(channel).or_default();
        c.writing = c.writing.saturating_sub(1);
        let Some(new) = output else { return };
        let is_last = c.last.as_ref().is_some_and(|o| o.finished_id == old);
        if let Some(before) = c.last.iter().chain(c.recent.iter()).find(|o| o.finished_id == old) {
            Self::drop_leftovers(&mut c.posts, before, &new);
        }
        if let Some(slot) = Self::outputs_mut(c).find(|o| o.finished_id == old) {
            *slot = Output {
                requester: slot.requester,
                finished_at: slot.finished_at,
                finished_id: slot.finished_id,
                ..new
            };
        }
        if is_last {
            c.undo.clear();
        }
    }

    /// The output finished at `finished_id` was deleted by an admin. The last
    /// one going takes its undo stack along, as with `/delete-reply`.
    pub fn remove(&mut self, channel: u64, finished_id: u64) {
        let Some(c) = self.channels.get_mut(&channel) else { return };
        if c.last.as_ref().is_some_and(|o| o.finished_id == finished_id) {
            c.last = None;
            c.undo.clear();
        } else {
            c.recent.retain(|o| o.finished_id != finished_id);
        }
    }
}

/// How many message IDs a channel keeps of each kind.
fn kept(rules: &RetryRules) -> usize {
    rules.max_newer.saturating_add(KEPT_SPARE)
}

/// Inserts `item` into `list` (kept in order of `key`, at most `cap`, the
/// oldest dropped first), unless one with its key is there already.
fn insert_newest<T>(list: &mut VecDeque<T>, item: T, key: impl Fn(&T) -> u64, cap: usize) {
    let k = key(&item);
    let at = list.partition_point(|x| key(x) < k);
    if list.get(at).is_some_and(|x| key(x) == k) {
        return;
    }
    list.insert(at, item);
    if list.len() > cap {
        list.pop_front();
    }
}

/// The outputs to steer a retry away from: the earlier ones plus `said`, the
/// newest `max` kept. An empty output (only a notice) adds nothing.
pub fn steer(mut earlier: Vec<String>, said: &str, max: usize) -> Vec<String> {
    if !said.trim().is_empty() {
        earlier.push(said.to_string());
    }
    let skip = earlier.len().saturating_sub(max);
    earlier.drain(..skip);
    earlier
}

/// Where a retry's messages go: into the old output's messages, in order, then
/// new ones once those run out. The first old message holds the card (header),
/// if any, which stays above the new text.
#[derive(Debug)]
pub struct Reuse {
    header: Option<String>,
    header_msg: Option<u64>,
    unused: VecDeque<u64>,
    /// Each old message still there, with its old text (header not included).
    originals: HashMap<u64, String>,
}

impl Reuse {
    pub fn new(header: Option<String>, messages: &[(u64, String)]) -> Self {
        Self {
            header_msg: header.as_ref().and(messages.first().map(|m| m.0)),
            header,
            unused: messages.iter().map(|m| m.0).collect(),
            originals: messages.iter().cloned().collect(),
        }
    }

    pub fn header(&self) -> Option<&str> {
        self.header.as_deref()
    }

    /// The header, if `id` is the message it tops.
    pub fn header_of(&self, id: u64) -> Option<&str> {
        self.header.as_deref().filter(|_| self.header_msg == Some(id))
    }

    /// The message the header tops, while it is there: a brand-new part
    /// replies to it, so a cold start ties the part back to its card.
    pub fn card(&self) -> Option<u64> {
        self.header.as_ref().and(self.header_msg)
    }

    /// What message `id` shows for `text`: the header above it if it has one.
    pub fn shown(&self, id: u64, text: &str) -> String {
        match self.header_of(id) {
            Some(h) => format!("{h}\n{text}"),
            None => text.to_string(),
        }
    }

    /// The next old message to write into.
    pub fn next_old(&mut self) -> Option<u64> {
        self.unused.pop_front()
    }

    /// An old message turned out to be deleted: it holds nothing now.
    pub fn gone(&mut self, id: u64) {
        self.originals.remove(&id);
        if self.header_msg == Some(id) {
            self.header_msg = None;
        }
    }

    /// What a brand-new message shows (the old ones ran out): the first part
    /// takes the header if its old message is gone.
    pub fn fresh(&self, text: &str, first: bool) -> String {
        match &self.header {
            Some(h) if first && self.header_msg.is_none() => format!("{h}\n{text}"),
            _ => text.to_string(),
        }
    }

    /// A brand-new message was posted for the part [`fresh`](Self::fresh) made.
    pub fn posted(&mut self, id: u64, first: bool) {
        if first && self.header.is_some() && self.header_msg.is_none() {
            self.header_msg = Some(id);
        }
    }

    /// An old message's text before the retry, to put back if it's dropped.
    pub fn original(&self, id: u64) -> Option<&str> {
        self.originals.get(&id).map(String::as_str)
    }

    /// The old messages the new output didn't need (to delete once it's done).
    pub fn leftovers(&mut self) -> Vec<u64> {
        self.unused.drain(..).collect()
    }

    /// Every old message still there, oldest first: the turn a retried channel
    /// reply replaces in the history window.
    pub fn old_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.originals.keys().copied().collect();
        ids.sort_unstable();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const CH: u64 = 1;
    /// The default rules' numbers, which `Book::default()` uses.
    const WINDOW: Duration = Duration::from_secs(300);
    const RECENT: usize = 20;
    const UNDO_MAX: usize = 10;
    const NEWER_KEPT: usize = 5 + KEPT_SPARE;

    #[test]
    fn the_test_numbers_are_the_defaults() {
        let r = RetryRules::default();
        assert_eq!((r.window(), r.admin_recent, r.undo_depth, kept(&r)), (WINDOW, RECENT, UNDO_MAX, NEWER_KEPT));
    }
    const ME: u64 = 7;

    /// An output whose messages start before `finished` (its finish snowflake).
    fn book_with_output(at: Instant, first_msg: u64) -> Book<&'static str> {
        book_with_rules(RetryRules::default(), at, first_msg)
    }

    fn book_with_rules(rules: RetryRules, at: Instant, first_msg: u64) -> Book<&'static str> {
        let mut b = Book::new(rules);
        b.note_activity(CH, 90); // untracked channel: ignored
        b.started(CH);
        b.note_activity(CH, 95); // the trigger, before the reply
        b.finished(
            CH,
            Some(Output {
                replay: "job",
                requester: ME,
                messages: vec![(first_msg - 10, "hi".into())],
                turn: vec![(first_msg - 10, "hi".into())],
                finished_at: at,
                finished_id: first_msg,
            }),
            false,
        );
        b
    }

    #[test]
    fn a_quiet_channel_can_retry_once_at_a_time() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t + Duration::from_secs(10)).expect("allowed");
        assert_eq!(out.replay, "job");
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::Retrying);
        assert!(b.still_quiet(CH, &out).is_ok());
    }

    #[test]
    fn a_newer_reply_is_no_longer_the_last() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).expect("allowed");
        assert!(b.still_last(CH, &out).is_ok());
        // The bot answered someone else meanwhile.
        b.started(CH);
        assert_eq!(b.still_last(CH, &out).unwrap_err(), Refusal::Writing);
        b.finished(CH, Some(Output { finished_id: 120, ..out.clone() }), false);
        assert_eq!(b.still_last(CH, &out).unwrap_err(), Refusal::NotLast);
        b.forget(CH);
        assert_eq!(b.still_last(CH, &out).unwrap_err(), Refusal::Nothing);
    }

    /// A finished output of message `id`, finished at `id`.
    fn reply(b: &mut Book<&'static str>, id: u64) {
        b.started(CH);
        b.finished(
            CH,
            Some(Output {
                replay: "job",
                requester: ME,
                messages: vec![(id, format!("r{id}"))],
                turn: vec![(id, format!("r{id}"))],
                finished_at: Instant::now(),
                finished_id: id,
            }),
            false,
        );
    }

    #[test]
    fn the_last_twenty_outputs_are_remembered() {
        let mut b = Book::default();
        for id in 1..=(RECENT as u64 + 5) {
            reply(&mut b, id);
        }
        assert!(b.find(CH, 5).is_none(), "too old");
        assert_eq!(b.find(CH, 6).map(|o| o.finished_id), Some(6));
        assert_eq!(b.find(CH, RECENT as u64 + 5).map(|o| o.finished_id), Some(RECENT as u64 + 5));
        b.forget(CH);
        assert!(b.find(CH, 10).is_none());
    }

    #[test]
    fn a_forced_rewrite_keeps_its_place_and_finish() {
        let mut b = Book::default();
        for id in [10, 20, 30] {
            reply(&mut b, id);
        }
        let old = b.begin_force(CH, 20).expect("remembered");
        assert_eq!(b.begin_force(CH, 10).unwrap_err(), Refusal::Retrying);
        b.started(CH);
        let new = Output { messages: vec![(20, "new".into())], requester: 99, finished_id: 999, ..old };
        b.finished_replacing(CH, Some(new), 20);
        b.end_force(CH);
        let now = b.find(CH, 20).expect("still there");
        assert_eq!((now.messages[0].1.as_str(), now.requester, now.finished_id), ("new", ME, 20));
        // The last output is still the last, and still the requester's to retry.
        assert_eq!(b.begin(CH, ME, Instant::now()).map(|o| o.finished_id), Ok(30));
    }

    #[test]
    fn a_forced_delete_forgets_just_that_one() {
        let mut b = Book::default();
        for id in [10, 20, 30] {
            reply(&mut b, id);
        }
        b.remove(CH, 20);
        assert!(!b.still_there(CH, 20));
        assert!(b.still_there(CH, 10) && b.still_there(CH, 30));
        b.remove(CH, 30);
        assert_eq!(b.begin(CH, ME, Instant::now()).unwrap_err(), Refusal::Nothing);
        assert_eq!(b.begin_force(CH, 30).unwrap_err(), Refusal::Nothing);
    }

    #[test]
    fn only_the_requester() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        assert_eq!(b.begin(CH, 8, t).unwrap_err(), Refusal::NotYours);
        assert_eq!(b.begin(2, ME, t).unwrap_err(), Refusal::Nothing);
    }

    #[test]
    fn five_messages_since_are_fine_six_are_not() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        for id in 101..=105 {
            b.note_message(CH, id, 8);
        }
        assert!(b.can_retry(CH, ME, t).is_ok());
        b.note_message(CH, 106, 8);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
        // A deleted one no longer counts.
        b.unnote_message(CH, 103);
        assert!(b.begin(CH, ME, t).is_ok());
    }

    #[test]
    fn commands_and_unseen_messages_dont_count() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        for id in 101..=120 {
            b.note_activity(CH, id);
        }
        assert!(b.can_retry(CH, ME, t).is_ok());
        assert_eq!(b.can_delete(CH, ME).unwrap_err(), Refusal::NotLast, "delete still wants nothing since");
    }

    #[test]
    fn talk_while_it_streamed_doesnt_count() {
        let t = Instant::now();
        // Messages at 90..=100, posted at 90, finished at 100.
        let mut b = book_with_output(t, 100);
        for id in 91..=100 {
            b.note_message(CH, id, 8);
        }
        assert!(b.begin(CH, ME, t).is_ok());
    }

    #[test]
    fn a_newer_bot_post_blocks_it() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.note_bot_post(CH, 101);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::NotLast);
        assert_eq!(b.can_undo(CH, ME, t).unwrap_err(), Refusal::Nothing);
        assert_eq!(b.can_delete(CH, ME).unwrap_err(), Refusal::NotLast);
    }

    #[test]
    fn steer_keeps_the_newest_three_and_skips_empty() {
        let v = steer(Vec::new(), "a", 3);
        assert_eq!(v, ["a"]);
        let v = steer(v, "  ", 3);
        assert_eq!(v, ["a"], "a notice-only output adds nothing");
        let v = steer(steer(steer(v, "b", 3), "c", 3), "d", 3);
        assert_eq!(v, ["b", "c", "d"]);
        assert_eq!(steer(v, "e", 1), ["e"], "the configured number is kept");
    }

    fn old() -> Vec<(u64, String)> {
        vec![(10, "one".into()), (11, "two".into()), (12, "three".into())]
    }

    #[test]
    fn reuse_fills_old_messages_in_order_then_leaves_the_rest() {
        let mut r = Reuse::new(None, &old());
        assert_eq!(r.next_old(), Some(10));
        assert_eq!(r.shown(10, "new"), "new");
        assert_eq!(r.next_old(), Some(11));
        assert_eq!(r.leftovers(), [12]);
        assert_eq!(r.next_old(), None);
        assert_eq!(r.old_ids(), [10, 11, 12], "the leftover leaves the history too");
        assert_eq!(r.original(11), Some("two"));
    }

    #[test]
    fn reuse_keeps_the_card_on_the_first_message() {
        let mut r = Reuse::new(Some("CARD".into()), &old());
        assert_eq!(r.header(), Some("CARD"));
        let first = r.next_old().unwrap();
        assert_eq!(r.shown(first, "new"), "CARD\nnew");
        let second = r.next_old().unwrap();
        assert_eq!(r.shown(second, "more"), "more");
        // Restoring puts the old text back below the card.
        assert_eq!(r.shown(first, r.original(first).unwrap()), "CARD\none");
    }

    #[test]
    fn reuse_skips_deleted_messages_and_moves_the_card_to_a_new_one() {
        let mut r = Reuse::new(Some("CARD".into()), &old()[..1]);
        let id = r.next_old().unwrap();
        r.gone(id);
        assert_eq!(r.next_old(), None);
        assert_eq!(r.fresh("new", true), "CARD\nnew");
        r.posted(50, true);
        assert_eq!(r.header_of(50), Some("CARD"));
        assert_eq!(r.fresh("more", false), "more");
        assert!(r.old_ids().is_empty(), "nothing old left to replace in history");
        assert_eq!(r.original(id), None);
    }

    #[test]
    fn reuse_without_a_card_posts_plain_new_messages() {
        let mut r = Reuse::new(None, &[]);
        assert_eq!(r.next_old(), None);
        assert_eq!(r.fresh("x", true), "x");
        r.posted(5, true);
        assert_eq!(r.header_of(5), None);
    }

    #[test]
    fn activity_during_the_quiet_second_cancels_a_delete() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin_delete(CH, ME).unwrap();
        b.note_activity(CH, 120);
        assert_eq!(b.still_quiet(CH, &out).unwrap_err(), Refusal::NotLast);
        b.end_delete(CH, false);
        // The channel isn't quiet anymore.
        assert_eq!(b.begin_delete(CH, ME).unwrap_err(), Refusal::NotLast);
    }

    #[test]
    fn the_pick_counts_what_was_said_meanwhile() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).unwrap();
        for id in 101..=105 {
            b.note_message(CH, id, 8);
        }
        assert!(b.still_last(CH, &out).is_ok());
        b.note_message(CH, 106, 8);
        assert_eq!(b.still_last(CH, &out).unwrap_err(), Refusal::TooMuchSince);
    }

    #[test]
    fn five_minutes_from_the_last_finish() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        assert_eq!(b.begin(CH, ME, t + WINDOW + Duration::from_secs(1)).unwrap_err(), Refusal::TooLate);
        assert!(b.begin(CH, ME, t + WINDOW).is_ok());
    }

    #[test]
    fn a_retry_keeps_the_original_clock() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).unwrap();
        // The retry streams into the same messages.
        b.started(CH);
        let done = t + Duration::from_secs(290);
        b.finished(CH, Some(Output { finished_at: done, finished_id: 200, ..out }), true);
        b.end(CH);
        assert_eq!(b.begin(CH, ME, done + Duration::from_secs(60)).unwrap_err(), Refusal::TooLate);
        // Nor does what was said while it streamed get a pass.
        for id in 101..=106 {
            b.note_message(CH, id, 8);
        }
        assert_eq!(b.begin(CH, ME, done).unwrap_err(), Refusal::TooMuchSince);
    }

    #[test]
    fn an_undo_keeps_the_original_clock() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1"]);
        let (cur, prev) = b.begin_undo(CH, ME, at).unwrap();
        b.end_undo(CH, Some(Output { replay: prev.replay, finished_at: at, finished_id: 500, ..cur }));
        assert_eq!(b.begin(CH, ME, t + WINDOW + Duration::from_secs(1)).unwrap_err(), Refusal::TooLate);
        assert_eq!(b.find(CH, 90).map(|o| o.finished_id), Some(100));
    }

    #[test]
    fn not_while_another_reply_is_written_here() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.started(CH);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::Writing);
        // A dropped retry leaves the last output as it was.
        b.finished(CH, None, true);
        let out = b.begin(CH, ME, t).unwrap();
        b.started(CH);
        assert_eq!(b.still_last(CH, &out).unwrap_err(), Refusal::Writing);
        b.finished(CH, None, true);
        b.end(CH);
        // Another reply that left nothing to remember and posted nothing
        // doesn't block it; one that left a card behind does.
        b.started(CH);
        b.finished(CH, None, false);
        assert!(b.can_retry(CH, ME, t).is_ok());
        b.note_bot_post(CH, 130);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::NotLast);
        // Unless that card is deleted.
        b.unnote_message(CH, 130);
        assert!(b.can_retry(CH, ME, t).is_ok());
    }

    #[test]
    fn a_post_unnoted_before_its_gateway_events_settles() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        // The bot deletes its own message 130 and unnotes it at once.
        b.note_bot_post(CH, 130);
        b.unnote_message(CH, 130);
        assert!(b.can_retry(CH, ME, t).is_ok(), "no wait for the gateway");
        // Its create event may still be on the way, the delete event after it.
        b.note_bot_post(CH, 130);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::NotLast);
        b.unnote_message(CH, 130);
        b.unnote_message(CH, 130);
        assert!(b.can_retry(CH, ME, t).is_ok());
        assert!(b.can_delete(CH, ME).is_ok());
    }

    #[test]
    fn an_older_bot_post_doesnt_block_it() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        // Its own message (posted while it streamed), and a card from before.
        b.note_bot_post(CH, 90);
        b.note_bot_post(CH, 80);
        assert!(b.can_retry(CH, ME, t).is_ok());
        assert!(b.can_delete(CH, ME).is_ok());
    }

    #[test]
    fn a_retrys_extra_part_is_its_own() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).unwrap();
        b.started(CH);
        // The retry reuses 90 and posts one more part, at 150.
        b.note_bot_post(CH, 150);
        let messages = vec![(90, "a".into()), (150, "b".into())];
        b.finished(CH, Some(Output { messages, finished_id: 151, ..out }), true);
        b.end(CH);
        assert!(b.can_retry(CH, ME, t).is_ok(), "150 is newer than the first finish, but it's the reply's own");
        assert!(b.can_undo(CH, ME, t).is_ok());
    }

    #[test]
    fn a_retrys_leftovers_stop_counting_at_once() {
        let t = Instant::now();
        let mut b = Book::default();
        b.started(CH);
        for id in [90, 91, 92] {
            b.note_bot_post(CH, id);
        }
        let messages = vec![(90, "a".into()), (91, "b".into()), (92, "c".into())];
        b.finished(CH, Some(Output { replay: "job", requester: ME, turn: messages.clone(), messages, finished_at: t, finished_id: 100 }), false);
        let out = b.begin(CH, ME, t).unwrap();
        b.started(CH);
        // The retry is one part long: 91 and 92 are deleted, and no gateway
        // event has arrived yet. 92 isn't newer than the finish anyway; a
        // leftover from an earlier retry's extra part would be.
        b.note_bot_post(CH, 160);
        b.finished(CH, Some(Output { messages: vec![(90, "x".into()), (160, "y".into())], finished_id: 161, ..out }), true);
        b.end(CH);
        let out = b.begin(CH, ME, t).unwrap();
        b.started(CH);
        b.finished(CH, Some(Output { messages: vec![(90, "z".into())], finished_id: 170, ..out }), true);
        b.end(CH);
        assert!(b.can_retry(CH, ME, t).is_ok(), "160 was deleted with the retry");
        // An undo brings 160 back as a new message; the one it replaces goes.
        let (cur, prev) = b.begin_undo(CH, ME, t).unwrap();
        b.note_bot_post(CH, 180);
        b.end_undo(CH, Some(Output { messages: vec![(90, "x".into()), (180, "y".into())], ..prev.clone() }));
        let _ = cur;
        assert!(b.can_retry(CH, ME, t).is_ok());
    }

    #[test]
    fn a_bot_post_during_the_pick_blocks_it() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).unwrap();
        b.note_bot_post(CH, 120);
        assert_eq!(b.still_last(CH, &out).unwrap_err(), Refusal::NotLast);
    }

    #[test]
    fn opted_out_authors_stop_counting() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        for id in 101..=103 {
            b.note_message(CH, id, 8);
        }
        for id in 104..=106 {
            b.note_message(CH, id, 9);
        }
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
        b.forget_author(9);
        assert!(b.can_retry(CH, ME, t).is_ok());
    }

    #[test]
    fn an_edit_counts_once_and_into_slashes_not_at_all() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        for id in 101..=105 {
            b.note_message(CH, id, 8);
        }
        // Edits re-note the same message: still five.
        b.note_message(CH, 103, 8);
        b.note_message(CH, 105, 8);
        assert!(b.can_retry(CH, ME, t).is_ok());
        b.note_message(CH, 106, 8);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
        // 104 edited into `//`.
        b.unnote_message(CH, 104);
        assert!(b.can_retry(CH, ME, t).is_ok());
        // And back out of it.
        b.note_message(CH, 104, 8);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
    }

    #[test]
    fn a_newer_bot_post_cancels_a_delete_during_the_quiet_second() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin_delete(CH, ME).unwrap();
        b.note_bot_post(CH, 120);
        assert_eq!(b.still_quiet(CH, &out).unwrap_err(), Refusal::NotLast);
    }

    #[test]
    fn an_admin_rewrite_drops_a_retrys_extra_part_at_once() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        // A retry grew a second part, newer than the kept finish.
        let out = b.begin(CH, ME, t).unwrap();
        b.started(CH);
        b.note_bot_post(CH, 160);
        b.finished(CH, Some(Output { messages: vec![(90, "x".into()), (160, "y".into())], finished_id: 161, ..out }), true);
        b.end(CH);
        assert!(b.can_retry(CH, ME, t).is_ok());
        // An admin rewrites it into one part: 160 is deleted, its gateway
        // event yet to come.
        let cur = b.last(CH).unwrap().clone();
        b.started(CH);
        b.finished_replacing(CH, Some(Output { messages: vec![(90, "z".into())], finished_id: 170, requester: 3, ..cur }), 100);
        let last = b.last(CH).unwrap();
        assert_eq!((last.requester, last.finished_id), (ME, 100), "keeps its requester and finish");
        assert!(b.can_retry(CH, ME, t).is_ok(), "160 no longer counts");
        assert_eq!(b.can_undo(CH, ME, t).unwrap_err(), Refusal::Nothing, "a rewrite clears the stack");
    }

    #[test]
    fn opting_out_forgets_in_every_channel() {
        const OTHER: u64 = 2;
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.started(OTHER);
        b.finished(
            OTHER,
            Some(Output { replay: "job", requester: ME, messages: vec![(90, "hi".into())], turn: vec![], finished_at: t, finished_id: 100 }),
            false,
        );
        for ch in [CH, OTHER] {
            for id in 101..=106 {
                b.note_message(ch, id, 9);
            }
            assert_eq!(b.can_retry(ch, ME, t).unwrap_err(), Refusal::TooMuchSince);
        }
        b.forget_author(9);
        for ch in [CH, OTHER] {
            assert!(b.can_retry(ch, ME, t).is_ok(), "channel {ch}");
        }
    }

    #[test]
    fn only_the_newest_messages_are_kept() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        for id in 101..=(101 + NEWER_KEPT as u64) {
            b.note_message(CH, id, 8);
        }
        // 101 fell off; with the newest gone again, five of those left.
        let newest = 101 + NEWER_KEPT as u64;
        for id in 107..=newest {
            b.unnote_message(CH, id);
        }
        assert!(b.can_retry(CH, ME, t).is_ok());
        // Something older than everything kept doesn't push in.
        b.note_message(CH, 101, 8);
        for id in 107..=newest {
            b.note_message(CH, id, 8);
        }
        for id in 107..=newest {
            b.unnote_message(CH, id);
        }
        assert!(b.can_retry(CH, ME, t).is_ok(), "101 came back in only while there was room");
    }

    const BOT: u64 = 99;

    #[test]
    fn the_gateway_counts_messages_by_kind() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        // `//` and opted-out messages don't count, but block a delete.
        b.gateway_message(CH, 101, 8, BOT, true, false);
        b.gateway_message(CH, 102, 8, BOT, false, true);
        assert!(b.can_retry(CH, ME, t).is_ok());
        assert_eq!(b.can_delete(CH, ME).unwrap_err(), Refusal::NotLast);
        // Anyone else counts, other bots too.
        for (id, author) in [(103, 8), (104, 40), (105, 40), (106, 9), (107, 8)] {
            b.gateway_message(CH, id, author, BOT, false, false);
        }
        assert!(b.can_retry(CH, ME, t).is_ok(), "five");
        b.gateway_message(CH, 108, 40, BOT, false, false);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
    }

    #[test]
    fn the_gateway_tells_the_bots_posts_apart() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.gateway_message(CH, 110, BOT, BOT, false, false);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::NotLast);
        // Its own edit (a stream, say, even into something `//`) changes nothing.
        b.gateway_edit(CH, 110, Some(BOT), BOT, true, false);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::NotLast);
        b.unnote_message(CH, 110);
        // Not counted as activity either: nothing blocks a delete.
        assert!(b.can_delete(CH, ME).is_ok());
    }

    #[test]
    fn the_gateway_recounts_edits() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        for id in 101..=105 {
            b.gateway_message(CH, id, 8, BOT, false, false);
        }
        // A `//` message edited out of it counts from then on.
        b.gateway_message(CH, 106, 8, BOT, true, false);
        assert!(b.can_retry(CH, ME, t).is_ok());
        b.gateway_edit(CH, 106, Some(8), BOT, false, false);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
        // Into `//` again, even with no author on the event: no longer.
        b.gateway_edit(CH, 106, None, BOT, true, false);
        assert!(b.can_retry(CH, ME, t).is_ok());
        // An edit without an author can't count, nor one by someone opted out.
        b.gateway_edit(CH, 106, None, BOT, false, false);
        b.gateway_edit(CH, 106, Some(8), BOT, false, true);
        assert!(b.can_retry(CH, ME, t).is_ok());
        // Another bot's edit counts like anyone's.
        b.gateway_edit(CH, 106, Some(40), BOT, false, false);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
    }

    /// Retries `n` times from `book_with_output`, each output's replay naming
    /// it ("job", "r1", "r2", …); returns when the last finished.
    fn retried(b: &mut Book<&'static str>, t: Instant, names: &[&'static str]) -> Instant {
        let mut at = t;
        for (i, name) in names.iter().enumerate() {
            at += Duration::from_secs(5);
            let out = b.begin(CH, ME, at).unwrap();
            b.started(CH);
            let id = 200 + i as u64;
            b.finished(CH, Some(Output { replay: *name, finished_at: at, finished_id: id, ..out }), true);
            b.end(CH);
        }
        at
    }

    #[test]
    fn undo_pops_back_to_the_original() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        assert_eq!(b.begin_undo(CH, ME, t).unwrap_err(), Refusal::Nothing, "nothing retried yet");
        let at = retried(&mut b, t, &["r1", "r2"]);
        for want in ["r1", "job"] {
            let (cur, prev) = b.begin_undo(CH, ME, at).unwrap();
            assert_eq!(b.begin(CH, ME, at + Duration::from_secs(5)).unwrap_err(), Refusal::Retrying);
            assert_eq!(prev.replay, want);
            b.end_undo(CH, Some(Output { replay: prev.replay, ..cur }));
        }
        assert_eq!(b.begin_undo(CH, ME, at).unwrap_err(), Refusal::Nothing);
        assert!(b.begin(CH, ME, at + Duration::from_secs(5)).is_ok());
    }

    #[test]
    fn a_retry_after_an_undo_pushes_again() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1", "r2"]);
        let (cur, prev) = b.begin_undo(CH, ME, at).unwrap();
        b.end_undo(CH, Some(Output { replay: prev.replay, ..cur }));
        let at = retried(&mut b, at, &["r3"]);
        let (_, prev) = b.begin_undo(CH, ME, at).unwrap();
        assert_eq!(prev.replay, "r1", "r2 was undone; r3 replaced r1");
    }

    #[test]
    fn a_failed_undo_keeps_the_stack() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1"]);
        b.begin_undo(CH, ME, at).unwrap();
        b.end_undo(CH, None);
        let (cur, prev) = b.begin_undo(CH, ME, at).unwrap();
        assert_eq!((cur.replay, prev.replay), ("r1", "job"));
    }

    #[test]
    fn undo_follows_the_retry_rules() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1"]);
        assert_eq!(b.begin_undo(CH, 8, at).unwrap_err(), Refusal::NotYours);
        assert_eq!(b.begin_undo(CH, ME, at + WINDOW + Duration::from_secs(5)).unwrap_err(), Refusal::TooLate);
        for id in 300..306 {
            b.note_message(CH, id, 8);
        }
        assert_eq!(b.begin_undo(CH, ME, at).unwrap_err(), Refusal::TooMuchSince);
    }

    #[test]
    fn a_new_reply_or_reset_clears_the_stack() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1"]);
        b.started(CH);
        assert_eq!(b.begin(CH, ME, at + Duration::from_secs(5)).unwrap_err(), Refusal::Writing);
        let new = Output { replay: "new", requester: ME, messages: vec![(300, "x".into())], turn: Vec::new(), finished_at: at, finished_id: 300 };
        b.finished(CH, Some(new), false);
        assert_eq!(b.begin_undo(CH, ME, at).unwrap_err(), Refusal::Nothing);

        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1"]);
        b.forget(CH);
        assert_eq!(b.begin_undo(CH, ME, at).unwrap_err(), Refusal::Nothing);
    }

    #[test]
    fn the_configured_rules_apply() {
        let rules = RetryRules { window_secs: 90, max_newer: 1, undo_depth: 1, admin_recent: 2, ..Default::default() };
        let t = Instant::now();
        let mut b = book_with_rules(rules.clone(), t, 100);
        assert_eq!(b.begin(CH, ME, t + Duration::from_secs(91)).unwrap_err(), Refusal::TooLate);
        b.note_message(CH, 101, 8);
        assert!(b.can_retry(CH, ME, t + Duration::from_secs(90)).is_ok());
        b.note_message(CH, 102, 8);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);

        // A one-deep stack: the second retry drops the first, the original stays.
        let mut b = book_with_rules(rules.clone(), t, 100);
        let at = retried(&mut b, t, &["r1", "r2"]);
        assert_eq!(b.can_undo(CH, ME, t + Duration::from_secs(91)).unwrap_err(), Refusal::TooLate, "the original's clock");
        let (cur, prev) = b.begin_undo(CH, ME, at).unwrap();
        assert_eq!(prev.replay, "job");
        b.end_undo(CH, Some(Output { replay: prev.replay, ..cur }));
        assert_eq!(b.can_undo(CH, ME, at).unwrap_err(), Refusal::Nothing);

        // The admin menu reaches the last two.
        let mut b = Book::new(rules);
        for id in 1..=3 {
            reply(&mut b, id);
        }
        assert!(b.find(CH, 1).is_none());
        assert!(b.find(CH, 2).is_some() && b.find(CH, 3).is_some());
    }

    #[test]
    fn no_newer_messages_allowed() {
        let rules = RetryRules { max_newer: 0, ..Default::default() };
        let t = Instant::now();
        let mut b = book_with_rules(rules, t, 100);
        assert!(b.can_retry(CH, ME, t).is_ok());
        b.note_message(CH, 101, 8);
        assert_eq!(b.can_retry(CH, ME, t).unwrap_err(), Refusal::TooMuchSince);
    }

    #[test]
    fn the_refusals_name_the_configured_limits() {
        let d = RetryRules::default();
        assert!(Refusal::TooLate.message(&d).contains("within 5 minutes."));
        assert!(Refusal::TooMuchSince.message(&d).contains("over 5 messages"));
        let r = RetryRules { window_secs: 90, max_newer: 1, ..Default::default() };
        assert!(Refusal::TooLate.message(&r).contains("within 90 seconds."));
        assert!(Refusal::TooLate.undo_message(&r).contains("within 90 seconds."));
        assert!(Refusal::TooMuchSince.message(&r).contains("over 1 message)"));
        assert!(Refusal::TooMuchSince.undo_message(&r).contains("over 1 message)"));
        let r = RetryRules { window_secs: 60, ..Default::default() };
        assert!(Refusal::TooLate.message(&r).contains("within 1 minute."));
        assert!(Refusal::TooLate.delete_message(&r).contains("within 1 minute."));
    }

    #[test]
    fn the_stack_keeps_the_original_when_full() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let names: Vec<&'static str> = (0..UNDO_MAX + 2).map(|i| &*format!("r{i}").leak()).collect();
        let mut at = retried(&mut b, t, &names);
        let mut seen = Vec::new();
        while let Ok((cur, prev)) = b.begin_undo(CH, ME, at) {
            seen.push(prev.replay);
            b.end_undo(CH, Some(Output { replay: prev.replay, finished_at: at, ..cur }));
            at += Duration::from_secs(1);
        }
        assert_eq!(seen.len(), UNDO_MAX);
        assert_eq!(seen.last(), Some(&"job"));
        assert_eq!(seen[0], names[names.len() - 2]);
    }

    #[test]
    fn delete_leaves_nothing_to_delete_retry_or_undo() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let at = retried(&mut b, t, &["r1"]);
        assert_eq!(b.begin_delete(CH, 8).unwrap_err(), Refusal::NotYours);
        let out = b.begin_delete(CH, ME).unwrap();
        assert_eq!(out.replay, "r1", "a retried reply can be deleted");
        assert_eq!(b.begin(CH, ME, at + Duration::from_secs(5)).unwrap_err(), Refusal::Retrying);
        b.end_delete(CH, true);
        assert_eq!(b.begin_delete(CH, ME).unwrap_err(), Refusal::Nothing, "not stackable");
        assert_eq!(b.begin_undo(CH, ME, at).unwrap_err(), Refusal::Nothing);
        assert_eq!(b.begin(CH, ME, at + Duration::from_secs(5)).unwrap_err(), Refusal::Nothing);
    }

    #[test]
    fn delete_needs_a_quiet_channel() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.note_activity(CH, 101);
        assert_eq!(b.begin_delete(CH, ME).unwrap_err(), Refusal::NotLast);
        let mut b = book_with_output(t, 100);
        b.started(CH);
        assert_eq!(b.begin_delete(CH, ME).unwrap_err(), Refusal::Writing);
        b.finished(CH, None, false);
        b.begin_delete(CH, ME).unwrap();
        b.end_delete(CH, false);
        assert!(b.begin_delete(CH, ME).is_ok(), "a failed delete keeps the output");
    }

    #[test]
    fn reset_forgets() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.forget(CH);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::Nothing);
    }
}
