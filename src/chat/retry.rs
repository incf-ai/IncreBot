//! `/retry` bookkeeping (design.md §6): per channel, the last generated output
//! and whether anyone has spoken since, so a retry only replaces a reply that is
//! still the newest thing in the channel.
//!
//! A retry is refused unless, in that channel:
//! 1. nothing but `/retry` happened since the output finished generating
//!    (messages from anyone but the bot, and any other command, timestamped
//!    after the finish; what was said while it streamed doesn't count);
//! 2. that is still true [`QUIET`] after the `/retry` (checked again then);
//! 3. the output finished at most [`WINDOW`] ago (a retry restarts the clock);
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
//! `/delete-reply` deletes the last output under rules 1, 2 and 4 (no time
//! limit); nothing before it becomes deletable or retryable.
//!
//! The admin's right-click actions (`discord::force`) work on any of the last
//! [`RECENT`] outputs here, under none of these rules but the lock: rewritten,
//! an output keeps its place, requester and finish, with no undo.
//!
//! [`Reuse`] plans how a retry's text goes into the old output's messages, and
//! [`steer`] keeps the outputs a retry is pushed away from.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

pub const WINDOW: Duration = Duration::from_secs(300);
pub const QUIET: Duration = Duration::from_secs(1);
/// How many earlier outputs a retry steers away from.
pub const STEER_MAX: usize = 3;
/// How many replaced outputs `/undo-retry` can go back through; past this the
/// oldest retry goes, the original stays.
pub const UNDO_MAX: usize = 10;
/// How many outputs per channel the admin's actions can reach, the last
/// included.
pub const RECENT: usize = 20;

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
    pub finished_at: Instant,
    /// The finish as a snowflake: activity with a larger ID came after it.
    pub finished_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Nothing,
    NotYours,
    TooLate,
    NotLast,
    Writing,
    Retrying,
}

impl Refusal {
    /// The ephemeral reason the caller sees.
    pub fn message(self) -> &'static str {
        match self {
            Refusal::Nothing => "There's nothing to retry here.",
            Refusal::NotYours => "Only the person who asked for that reply can retry it.",
            Refusal::TooLate => "Too late: a reply can only be retried within 5 minutes.",
            Refusal::NotLast => "Someone posted after that reply, so it can't be retried.",
            Refusal::Writing => "A reply is still being written here; try again once it's done.",
            Refusal::Retrying => "A retry or undo is already running here.",
        }
    }

    /// The reason for a refused `/undo-retry`.
    pub fn undo_message(self) -> &'static str {
        match self {
            Refusal::Nothing => "There's no retry to undo here.",
            Refusal::NotYours => "Only the person who asked for that reply can undo its retry.",
            Refusal::TooLate => "Too late: a retry can only be undone within 5 minutes.",
            Refusal::NotLast => "Someone posted after that reply, so its retry can't be undone.",
            r => r.message(),
        }
    }

    /// The reason for a refused `/delete-reply`.
    pub fn delete_message(self) -> &'static str {
        match self {
            Refusal::Nothing => "There's no reply to delete here.",
            Refusal::NotYours => "Only the person who asked for that reply can delete it.",
            Refusal::NotLast => "Someone posted after that reply, so it can't be deleted.",
            Refusal::Retrying => "A retry or undo is running here; try again once it's done.",
            r => r.message(),
        }
    }
}

struct Channel<R> {
    /// The newest foreign activity seen (a snowflake), 0 if none.
    activity: u64,
    /// Replies being written or waiting here.
    writing: usize,
    last: Option<Output<R>>,
    /// Earlier outputs, newest first; with `last`, at most [`RECENT`].
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
}

impl<R> Default for Book<R> {
    fn default() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }
}

impl<R: Clone> Book<R> {
    /// A message from anyone but the bot, or a command other than `/retry`.
    pub fn note_activity(&mut self, channel: u64, id: u64) {
        if let Some(c) = self.channels.get_mut(&channel) {
            c.activity = c.activity.max(id);
        }
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
        if output.is_none() {
            return;
        }
        match c.last.take() {
            Some(old) if retried => {
                c.undo.push(old);
                if c.undo.len() > UNDO_MAX {
                    c.undo.remove(1);
                }
            }
            Some(old) => {
                c.undo.clear();
                c.recent.push_front(old);
                c.recent.truncate(RECENT - 1);
            }
            None => c.undo.clear(),
        }
        c.last = output;
    }

    /// Rules 1, 3 and 4 for `user`, changing nothing.
    pub fn can_retry(&self, channel: u64, user: u64, now: Instant) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        if c.retrying {
            return Err(Refusal::Retrying);
        }
        let last = c.last.as_ref().ok_or(Refusal::Nothing)?;
        if last.requester != user {
            return Err(Refusal::NotYours);
        }
        if now.saturating_duration_since(last.finished_at) > WINDOW {
            return Err(Refusal::TooLate);
        }
        if c.writing > 0 {
            return Err(Refusal::Writing);
        }
        if c.activity > last.finished_id {
            return Err(Refusal::NotLast);
        }
        Ok(())
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
        if now.saturating_duration_since(last.finished_at) > WINDOW {
            return Err(Refusal::TooLate);
        }
        if c.writing > 0 {
            return Err(Refusal::Writing);
        }
        if c.activity > last.finished_id {
            return Err(Refusal::NotLast);
        }
        Ok(())
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

    /// `/delete-reply`'s rules, changing nothing: those of
    /// [`can_retry`](Self::can_retry) bar the time limit.
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
        if c.activity > last.finished_id {
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
    /// stack and is the last output), None if nothing could be written.
    pub fn end_undo(&mut self, channel: u64, restored: Option<Output<R>>) {
        let Some(c) = self.channels.get_mut(&channel) else {
            return;
        };
        c.retrying = false;
        // Reset meanwhile: nothing from before it comes back.
        if restored.is_some() && c.undo.pop().is_some() {
            c.last = restored;
        }
    }

    /// Rule 2, [`QUIET`] after [`begin`](Self::begin): still nothing since the
    /// output, and no reply started meanwhile.
    pub fn still_quiet(&self, channel: u64, output: &Output<R>) -> Result<(), Refusal> {
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        if c.writing > 0 {
            return Err(Refusal::Writing);
        }
        if c.activity > output.finished_id {
            return Err(Refusal::NotLast);
        }
        Ok(())
    }

    /// [`still_quiet`](Self::still_quiet), and `output` is still the last
    /// output here (no other reply finished meanwhile): `/retry-batch`'s OK.
    pub fn still_last(&self, channel: u64, output: &Output<R>) -> Result<(), Refusal> {
        self.still_quiet(channel, output)?;
        let c = self.channels.get(&channel).ok_or(Refusal::Nothing)?;
        match &c.last {
            Some(last) if last.finished_id == output.finished_id => Ok(()),
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

/// The outputs to steer a retry away from: the earlier ones plus `said`, the
/// newest [`STEER_MAX`] kept. An empty output (only a notice) adds nothing.
pub fn steer(mut earlier: Vec<String>, said: &str) -> Vec<String> {
    if !said.trim().is_empty() {
        earlier.push(said.to_string());
    }
    let skip = earlier.len().saturating_sub(STEER_MAX);
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

    const CH: u64 = 1;
    const ME: u64 = 7;

    /// An output whose messages start before `finished` (its finish snowflake).
    fn book_with_output(at: Instant, first_msg: u64) -> Book<&'static str> {
        let mut b = Book::default();
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
    fn anything_after_the_finish_blocks_it() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.note_activity(CH, 101);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::NotLast);
    }

    #[test]
    fn talk_while_it_streamed_doesnt_count() {
        let t = Instant::now();
        // Messages at 90..=100, posted at 90, finished at 100.
        let mut b = book_with_output(t, 100);
        b.note_activity(CH, 95);
        b.note_activity(CH, 100);
        assert!(b.begin(CH, ME, t).is_ok());
    }

    #[test]
    fn steer_keeps_the_newest_three_and_skips_empty() {
        let v = steer(Vec::new(), "a");
        assert_eq!(v, ["a"]);
        let v = steer(v, "  ");
        assert_eq!(v, ["a"], "a notice-only output adds nothing");
        let v = steer(steer(steer(v, "b"), "c"), "d");
        assert_eq!(v, ["b", "c", "d"]);
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
    fn activity_during_the_quiet_second_cancels() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).unwrap();
        b.note_activity(CH, 120);
        assert_eq!(b.still_quiet(CH, &out).unwrap_err(), Refusal::NotLast);
        b.end(CH);
        // The channel isn't quiet anymore.
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::NotLast);
    }

    #[test]
    fn five_minutes_from_the_last_finish() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        assert_eq!(b.begin(CH, ME, t + WINDOW + Duration::from_secs(1)).unwrap_err(), Refusal::TooLate);
        assert!(b.begin(CH, ME, t + WINDOW).is_ok());
    }

    #[test]
    fn a_retry_restarts_the_clock() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        let out = b.begin(CH, ME, t).unwrap();
        // The retry streams into the same messages.
        b.started(CH);
        let done = t + Duration::from_secs(290);
        b.finished(CH, Some(Output { finished_at: done, finished_id: 200, ..out }), true);
        b.end(CH);
        assert!(b.begin(CH, ME, done + Duration::from_secs(60)).is_ok(), "well past the original's 300 s");
    }

    #[test]
    fn not_while_another_reply_is_written_here() {
        let t = Instant::now();
        let mut b = book_with_output(t, 100);
        b.started(CH);
        assert_eq!(b.begin(CH, ME, t).unwrap_err(), Refusal::Writing);
        // A dropped reply (cleared from the queue) leaves the last output alone.
        b.finished(CH, None, false);
        let out = b.begin(CH, ME, t).unwrap();
        b.started(CH);
        assert_eq!(b.still_quiet(CH, &out).unwrap_err(), Refusal::Writing);
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
        b.note_activity(CH, 300);
        assert_eq!(b.begin_undo(CH, ME, at).unwrap_err(), Refusal::NotLast);
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
