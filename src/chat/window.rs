//! The prefix-stable history window (design.md §6).
//!
//! History is every message after a per-channel *anchor*. New messages are only
//! appended, so successive prompts share their prefix and the KV cache is reused.
//! When the window grows past `history_messages` or the prompt no longer fits,
//! the anchor jumps forward so only the newest half remains: one cold prefill,
//! then about `history_messages / 2` warm turns.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use super::sanitize::Extras;

/// Discord's epoch (2015-01-01T00:00:00Z) in Unix milliseconds.
const DISCORD_EPOCH_MS: u64 = 1_420_070_400_000;

/// How many messages a channel's window buffers: enough beyond
/// `history_messages` (what a prompt holds) that quiet channels keep context.
pub fn max_buffer(history_messages: usize) -> usize {
    history_messages * 4
}

/// When a Discord ID (snowflake) was created, in Unix milliseconds.
pub fn snowflake_ms(id: u64) -> u64 {
    (id >> 22) + DISCORD_EPOCH_MS
}

impl ChannelWindow {
    /// How long the channel was quiet before `id`: from the newest message
    /// before it (anyone's, the bot's too) to it; `None` with nothing before.
    pub fn gap_before(&self, id: u64) -> Option<u64> {
        let prev = self.messages().map(|m| m.id).filter(|&m| m < id).max()?;
        Some(snowflake_ms(id).saturating_sub(snowflake_ms(prev)))
    }
}

/// The largest Discord ID (snowflake) of anything created at or before `t`: a
/// window anchored there holds only messages sent after `t`. Snowflakes carry
/// their creation time in milliseconds in the bits above 22.
pub fn snowflake_at(t: SystemTime) -> u64 {
    let ms = t
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    (ms.saturating_sub(DISCORD_EPOCH_MS) << 22) | ((1 << 22) - 1)
}

/// Where the messages since the bot's `n`-th last one start in `upto` (all
/// of them, with fewer bot messages).
fn span_start(upto: &[&WindowMsg], bot: u64, n: usize) -> usize {
    let mut start = upto.len();
    let mut seen = 0;
    while start > 0 {
        if upto[start - 1].author_id == bot {
            seen += 1;
            if seen == n.max(1) {
                break;
            }
        }
        start -= 1;
    }
    start
}

#[derive(Clone, Debug, PartialEq)]
pub struct WindowMsg {
    pub id: u64,
    pub author_id: u64,
    /// Already cleaned with [`clean_name`](super::sanitize::clean_name).
    pub author: String,
    /// Raw text with mentions resolved to display names.
    pub text: String,
    pub extras: Extras,
    /// The names its `@mentions` of users were resolved to (cleaned), so the
    /// prompt can rewrite them (`PromptBuilder::lowercase_names`).
    pub mentions: Vec<String>,
}

/// Cloned when a reply is queued: the job's prompt is built from that frozen copy
/// (design.md §6).
#[derive(Clone, Debug, Default)]
pub struct ChannelWindow {
    msgs: VecDeque<WindowMsg>,
    anchor: Option<u64>,
    /// Turns with no message of their own: a slash command's requester turn
    /// (`discord::remember`), keyed by its ID (the interaction's). It leaves
    /// with `message`, the bot's answer that shows it, and with anyone in
    /// `about` opting out.
    ties: Vec<Tie>,
    /// Whether the window was seeded from REST history since startup or `/super-admin reset`.
    pub loaded: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct Tie {
    turn: u64,
    /// The answer's messages. The whole exchange leaves together: deleting
    /// any of them, or anyone in `about` opting out, removes them all and
    /// the turn.
    messages: Vec<u64>,
    about: Vec<u64>,
}

impl ChannelWindow {
    /// An empty, unseeded window holding only messages after `anchor` (a message
    /// ID or [`snowflake_at`]): the cutoff at startup and at `/super-admin reset`. The REST
    /// seed then loads only what was posted since.
    pub fn after(anchor: u64) -> Self {
        Self {
            anchor: Some(anchor),
            ..Self::default()
        }
    }

    pub fn anchor(&self) -> Option<u64> {
        self.anchor
    }

    pub fn messages(&self) -> impl Iterator<Item = &WindowMsg> {
        self.msgs.iter()
    }

    /// The text of the messages that `bot` hasn't answered yet, up to and
    /// including `trigger`: everything after the bot's last message. What a
    /// reply to `trigger` responds to (`prompted_by` word rules).
    pub fn unanswered(&self, bot: u64, trigger: u64) -> String {
        self.since_replies(bot, trigger, 1)
    }

    /// How many replies the bot has made since a push started: the earliest
    /// user message that `hit` accepts since the bot's `span`-th last reply
    /// (up to `trigger`). Everyone's pushes count, so three people asking
    /// for his shovel add up, but only for a reply to someone who pushed in
    /// that span: `None` when the asker (`trigger`'s author) didn't, or
    /// nobody did. The count `for_replies` bans (`span` = their count) and
    /// `after_replies` notes (`after_replies + turns`) go by.
    pub fn replies_since(&self, bot: u64, trigger: u64, span: usize, hit: &dyn Fn(&str) -> bool) -> Option<usize> {
        let upto: Vec<&WindowMsg> = self.msgs.iter().filter(|m| m.id <= trigger).collect();
        let asker = upto.iter().rfind(|m| m.id == trigger)?.author_id;
        let hits: Vec<usize> = (span_start(&upto, bot, span)..upto.len())
            .filter(|&i| upto[i].author_id != bot && hit(&upto[i].text))
            .collect();
        let first = *hits.first()?;
        if !hits.iter().any(|&i| upto[i].author_id == asker) {
            return None;
        }
        Some(upto[first + 1..].iter().filter(|m| m.author_id == bot).count())
    }

    /// [`since_replies`](Self::since_replies), but only the messages of
    /// `trigger`'s author: what a non-safety note looks at, so other people's
    /// chatter in the backlog (a shared channel) doesn't set it off.
    pub fn since_replies_by_asker(&self, bot: u64, trigger: u64, n: usize) -> String {
        let Some(asker) = self.msgs.iter().find(|m| m.id == trigger).map(|m| m.author_id) else {
            return String::new();
        };
        self.since_replies_where(bot, trigger, n, &|m| m.author_id == asker)
    }

    /// The text of the user messages up to and including `trigger` since the
    /// bot's `n`-th last message (`n` = 1 is [`unanswered`](Self::unanswered)):
    /// what a prompt note that stays on for `n` replies looks at.
    pub fn since_replies(&self, bot: u64, trigger: u64, n: usize) -> String {
        self.since_replies_where(bot, trigger, n, &|_| true)
    }

    fn since_replies_where(&self, bot: u64, trigger: u64, n: usize, keep: &dyn Fn(&WindowMsg) -> bool) -> String {
        let upto: Vec<&WindowMsg> = self.msgs.iter().filter(|m| m.id <= trigger).collect();
        upto[span_start(&upto, bot, n)..]
            .iter()
            .filter(|m| m.author_id != bot && keep(m))
            .map(|m| m.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn len(&self) -> usize {
        self.msgs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.msgs.is_empty()
    }

    /// Adds a message (from a gateway event or the REST seed). Ignores messages at
    /// or before the anchor and duplicates; keeps id order. `max_buffer` bounds
    /// memory for channels where the bot is rarely triggered.
    pub fn push(&mut self, msg: WindowMsg, max_buffer: usize) {
        if self.anchor.is_some_and(|a| msg.id <= a) || self.msgs.iter().any(|m| m.id == msg.id) {
            return;
        }
        let at = self
            .msgs
            .iter()
            .position(|m| m.id > msg.id)
            .unwrap_or(self.msgs.len());
        self.msgs.insert(at, msg);
        while self.msgs.len() > max_buffer {
            self.advance_anchor(1);
        }
    }

    /// Adds a turn tied to its answer's `message` (see [`ties`](Self)): the
    /// exchange leaves together when the answer is deleted or anyone in
    /// `about` opts out. A [`push`](Self::push) otherwise.
    pub fn push_tied(&mut self, msg: WindowMsg, message: u64, about: Vec<u64>, max_buffer: usize) {
        let turn = msg.id;
        self.push(msg, max_buffer);
        if self.msgs.iter().any(|m| m.id == turn) && !self.ties.iter().any(|t| t.turn == turn) {
            self.ties.push(Tie { turn, messages: vec![message], about });
        }
    }

    /// Adds `message` (a follow-up of the answer) to `turn`'s tie, if any.
    pub fn tie_another(&mut self, turn: u64, message: u64) {
        if let Some(t) = self.ties.iter_mut().find(|t| t.turn == turn)
            && !t.messages.contains(&message)
        {
            t.messages.push(message);
        }
    }

    /// Who wrote the message `id`, if the window holds it.
    pub fn author_of(&self, id: u64) -> Option<u64> {
        self.msgs.iter().find(|m| m.id == id).map(|m| m.author_id)
    }

    /// Whether the window holds the message `id`.
    pub fn contains(&self, id: u64) -> bool {
        self.msgs.iter().any(|m| m.id == id)
    }

    /// Drops every tie `gone` returns true for, with its whole exchange (the
    /// turn and the answer's messages), and ties whose turn has left.
    fn drop_ties(&mut self, gone: &dyn Fn(&Tie) -> bool) {
        let ids: Vec<u64> = self
            .ties
            .iter()
            .filter(|t| gone(t))
            .flat_map(|t| std::iter::once(t.turn).chain(t.messages.iter().copied()))
            .collect();
        self.msgs.retain(|m| !ids.contains(&m.id));
        self.ties.retain(|t| !ids.contains(&t.turn));
        self.prune_ties();
    }

    /// Applies an edit. Earlier text changing shortens the reusable prefix, which
    /// the cache's longest-common-prefix match absorbs.
    pub fn update(&mut self, id: u64, text: String, extras: Extras, mentions: Vec<String>) {
        if let Some(m) = self.msgs.iter_mut().find(|m| m.id == id) {
            m.text = text;
            m.extras = extras;
            m.mentions = mentions;
        }
    }

    /// Drops message `id`. The answer's first message (the header's) takes
    /// its whole exchange with it, as the REST seed skips a headerless
    /// answer; a later part leaves alone.
    pub fn remove(&mut self, id: u64) {
        self.msgs.retain(|m| m.id != id);
        self.drop_ties(&|t| t.messages.first() == Some(&id));
        for t in &mut self.ties {
            t.messages.retain(|&m| m != id);
        }
    }

    /// A retried reply: the old turn's messages (`old`) leave and the new ones
    /// enter, in id order. The new ones mostly reuse the old IDs.
    pub fn replace_turn(&mut self, old: &[u64], new: Vec<WindowMsg>, max_buffer: usize) {
        self.msgs.retain(|m| !old.contains(&m.id));
        for m in new {
            self.push(m, max_buffer);
        }
    }

    /// [`replace_turn`](Self::replace_turn) for a turn that answers another
    /// turn (`tied`, a remembered command's requester turn): a no-op once that
    /// turn has left the window (deleted, opted out, or never entered), so a
    /// retry can't bring an answer back without what it answers.
    pub fn replace_turn_tied(&mut self, old: &[u64], new: Vec<WindowMsg>, tied: Option<u64>, max_buffer: usize) {
        if tied.is_some_and(|t| !self.contains(t)) {
            // An exchange lives and dies whole: without its turn (deleted,
            // opted out, or behind the anchor) nothing of it comes back.
            return;
        }
        let ids: Vec<u64> = new.iter().map(|m| m.id).collect();
        self.replace_turn(old, new, max_buffer);
        // The tie follows the retry to the new messages (mostly the old IDs).
        if let Some(t) = tied.and_then(|t| self.ties.iter_mut().find(|e| e.turn == t))
            && !ids.is_empty()
        {
            t.messages = ids;
        }
    }

    /// Drops every message by `author` (`/opt-out`), and the whole exchange
    /// of every command about them or asked by them (the answer included, as
    /// the REST seed skips it). Returns whether any was there.
    pub fn remove_author(&mut self, author: u64) -> bool {
        let before = self.msgs.len();
        let asked: Vec<u64> = self.msgs.iter().filter(|m| m.author_id == author).map(|m| m.id).collect();
        self.drop_ties(&|t| t.about.contains(&author) || asked.contains(&t.turn));
        self.msgs.retain(|m| m.author_id != author);
        self.msgs.len() != before
    }

    /// Moves the anchor forward to `anchor` (never back), dropping the messages at
    /// or before it. Used to carry an anchor jump made on a queued job's frozen
    /// copy over to the live window, so later prompts keep sharing a prefix.
    pub fn advance_to(&mut self, anchor: Option<u64>) {
        let Some(a) = anchor else { return };
        if self.anchor.is_some_and(|cur| cur >= a) {
            return;
        }
        while self.msgs.front().is_some_and(|m| m.id <= a) {
            self.msgs.pop_front();
        }
        self.anchor = Some(a);
        self.prune_ties();
    }

    /// Back to empty and unseeded (`/super-admin reset`).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The messages for the next prompt. Applies the anchor policy: while there
    /// are more than `history_messages` messages or `fits` rejects them, keep only
    /// the newest half. Always leaves at least the newest message.
    pub fn select(
        &mut self,
        history_messages: usize,
        fits: &dyn Fn(&[WindowMsg]) -> bool,
    ) -> Vec<WindowMsg> {
        loop {
            let n = self.msgs.len();
            let all: Vec<WindowMsg> = self.msgs.iter().cloned().collect();
            if n <= 1 || (n <= history_messages && fits(&all)) {
                return all;
            }
            let keep = (n.min(history_messages) / 2).max(1);
            self.advance_anchor(n - keep);
        }
    }

    fn advance_anchor(&mut self, drop: usize) {
        for _ in 0..drop {
            if let Some(m) = self.msgs.pop_front() {
                self.anchor = Some(m.id);
            }
        }
        self.prune_ties();
    }

    /// Drops the ties whose turn has left the window (the anchor passed it),
    /// with the rest of their exchange: an answer never outlives its
    /// question, so a cold start (which skips an answer whose question is
    /// behind the anchor) rebuilds the same window.
    fn prune_ties(&mut self) {
        let gone: Vec<u64> = {
            let msgs = &self.msgs;
            self.ties
                .iter()
                .filter(|t| !msgs.iter().any(|m| m.id == t.turn))
                .flat_map(|t| t.messages.iter().copied())
                .collect()
        };
        self.msgs.retain(|m| !gone.contains(&m.id));
        let msgs = &self.msgs;
        self.ties.retain(|t| msgs.iter().any(|m| m.id == t.turn));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(id: u64) -> WindowMsg {
        WindowMsg {
            id,
            author_id: 7,
            author: "A".into(),
            text: format!("m{id}"),
            extras: Extras::default(),
            mentions: Vec::new(),
        }
    }

    fn ids(v: &[WindowMsg]) -> Vec<u64> {
        v.iter().map(|m| m.id).collect()
    }

    #[test]
    fn replies_since_counts_from_the_first_push_in_the_span() {
        const BOT: u64 = 99;
        let mut w = ChannelWindow::default();
        let say = |w: &mut ChannelWindow, id: u64, author_id: u64, text: &str| {
            w.push(WindowMsg { author_id, text: text.into(), ..msg(id) }, 100);
        };
        let give = |t: &str| t.contains("give");
        say(&mut w, 1, 7, "give me ur shovel");
        assert_eq!(w.replies_since(BOT, 1, 1, &give), Some(0));
        say(&mut w, 2, BOT, "NO!!");
        // Someone who didn't push: nothing for their reply.
        say(&mut w, 3, 8, "whats ur favourite rock?");
        assert_eq!(w.replies_since(BOT, 3, 4, &give), None);
        say(&mut w, 4, BOT, "the warm one");
        // A second pusher adds to the first's push.
        say(&mut w, 5, 9, "just give it");
        assert_eq!(w.replies_since(BOT, 5, 4, &give), Some(2));
        say(&mut w, 6, BOT, "nooo");
        // A follow-up without the phrase: still the asker's push in the span.
        say(&mut w, 7, 7, "pleeease");
        assert_eq!(w.replies_since(BOT, 7, 4, &give), Some(3));
        // Out of a 1-reply span: the push was before the bot's last reply.
        assert_eq!(w.replies_since(BOT, 7, 1, &give), None);
    }

    #[test]
    fn asker_scoped_text_leaves_out_other_people() {
        let mut w = ChannelWindow::default();
        for (id, author_id) in [(1, 99), (2, 8), (3, 7), (4, 8), (5, 7)] {
            w.push(WindowMsg { author_id, ..msg(id) }, 100);
        }
        assert_eq!(w.since_replies_by_asker(99, 5, 1), "m3\nm5");
        assert_eq!(w.since_replies(99, 5, 1), "m2\nm3\nm4\nm5");
    }

    #[test]
    fn a_retried_turn_replaces_the_old_one_in_place() {
        let mut w = ChannelWindow::default();
        for id in [1, 2, 3, 4] {
            w.push(msg(id), 100);
        }
        // Old turn: 2 and 3. The retry fits in 2 and needs a new message, 9.
        let new = vec![
            WindowMsg { text: "new2".into(), ..msg(2) },
            WindowMsg { text: "new9".into(), ..msg(9) },
        ];
        w.replace_turn(&[2, 3], new, 100);
        let got: Vec<(u64, &str)> = w.messages().map(|m| (m.id, m.text.as_str())).collect();
        assert_eq!(got, [(1, "m1"), (2, "new2"), (4, "m4"), (9, "new9")]);
        // Nothing old and nothing new: a no-op.
        w.replace_turn(&[], Vec::new(), 100);
        assert_eq!(w.len(), 4);
    }

    #[test]
    fn unanswered_is_everything_since_the_bots_last_message() {
        let mut w = ChannelWindow::default();
        for id in 1..=6 {
            let author_id = if id == 3 { 99 } else { 7 };
            w.push(WindowMsg { author_id, ..msg(id) }, 100);
        }
        assert_eq!(w.unanswered(99, 5), "m4\nm5");
        assert_eq!(w.unanswered(99, 2), "m1\nm2");
        assert_eq!(w.unanswered(42, 2), "m1\nm2");
        // Since the bot's 2nd-last message: its replies are left out.
        for id in 7..=9 {
            let author_id = if id == 7 { 99 } else { 7 };
            w.push(WindowMsg { author_id, ..msg(id) }, 100);
        }
        assert_eq!(w.since_replies(99, 9, 1), "m8\nm9");
        assert_eq!(w.since_replies(99, 9, 2), "m4\nm5\nm6\nm8\nm9");
        assert_eq!(w.since_replies(99, 9, 3), "m1\nm2\nm4\nm5\nm6\nm8\nm9");
    }

    #[test]
    fn the_buffer_holds_four_prompts_of_messages() {
        assert_eq!(max_buffer(20), 80);
        assert_eq!(max_buffer(0), 0);
    }

    #[test]
    fn snowflakes_from_time() {
        // Discord's documented example: 175928847299117063 was created at
        // 1462015105796 ms.
        let t = UNIX_EPOCH + std::time::Duration::from_millis(1_462_015_105_796);
        let at = snowflake_at(t);
        assert!(at >= 175_928_847_299_117_063);
        assert!(at < snowflake_at(t + std::time::Duration::from_millis(1)) - (1 << 22) + 1);
        assert_eq!(snowflake_at(UNIX_EPOCH), (1 << 22) - 1);
    }

    #[test]
    fn a_window_after_a_cutoff_ignores_older_messages() {
        let mut w = ChannelWindow::after(10);
        assert!(!w.loaded);
        for id in [5, 10, 11, 12] {
            w.push(msg(id), 100);
        }
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [11, 12]);
        assert_eq!(w.anchor(), Some(10));
    }

    #[test]
    fn advance_to_only_moves_forward() {
        let mut w = ChannelWindow::default();
        for id in 1..=6 {
            w.push(msg(id), 100);
        }
        w.advance_to(None);
        assert_eq!(w.len(), 6);
        w.advance_to(Some(3));
        assert_eq!(
            (ids(&w.messages().cloned().collect::<Vec<_>>()), w.anchor()),
            (vec![4, 5, 6], Some(3))
        );
        w.advance_to(Some(2)); // never back
        assert_eq!((w.len(), w.anchor()), (3, Some(3)));
        w.push(msg(3), 100); // at or before the anchor: ignored
        assert_eq!(w.len(), 3);
        // A frozen copy's anchor jump carries over to the live window.
        let mut frozen = w.clone();
        frozen.select(2, &|_| true);
        w.push(msg(7), 100);
        w.advance_to(frozen.anchor());
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [6, 7]);
    }

    #[test]
    fn anchor_only_moves_at_the_limit_and_keeps_newest_half() {
        let mut w = ChannelWindow::default();
        let always = |_: &[WindowMsg]| true;
        let mut prev: Vec<u64> = Vec::new();
        for id in 1..=20 {
            w.push(msg(id), 100);
            let sel = ids(&w.select(20, &always));
            // Append-only until the limit: each prompt extends the previous one.
            assert_eq!(&sel[..prev.len()], &prev[..]);
            prev = sel;
        }
        assert_eq!(w.anchor(), None);
        w.push(msg(21), 100);
        let sel = ids(&w.select(20, &always));
        assert_eq!(sel, (12..=21).collect::<Vec<_>>());
        assert_eq!(w.anchor(), Some(11));
        // Messages at or before the anchor are ignored.
        w.push(msg(5), 100);
        assert_eq!(w.len(), 10);
    }

    #[test]
    fn token_budget_halves_until_it_fits() {
        let mut w = ChannelWindow::default();
        for id in 1..=10 {
            w.push(msg(id), 100);
        }
        let fits = |m: &[WindowMsg]| m.len() <= 3;
        assert_eq!(ids(&w.select(20, &fits)), vec![9, 10]); // 10 → 5 → 2
    }

    #[test]
    fn out_of_order_duplicates_edits_and_deletes() {
        let mut w = ChannelWindow::default();
        w.push(msg(3), 100);
        w.push(msg(1), 100);
        w.push(msg(3), 100);
        w.update(1, "edited".into(), Extras::default(), Vec::new());
        w.remove(3);
        let sel = w.select(20, &|_| true);
        assert_eq!(ids(&sel), vec![1]);
        assert_eq!(sel[0].text, "edited");
    }

    #[test]
    fn remove_author_drops_only_theirs() {
        let mut w = ChannelWindow::default();
        for id in 1..=4 {
            w.push(WindowMsg { author_id: id % 2, ..msg(id) }, 100);
        }
        assert!(w.remove_author(1));
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [2, 4]);
        assert!(!w.remove_author(1));
    }

    #[test]
    fn tied_turns_leave_with_their_message_and_their_people() {
        let mut w = ChannelWindow::default();
        // Turn 10 (a command's requester turn) shows in message 11.
        w.push_tied(WindowMsg { author_id: 5, ..msg(10) }, 11, vec![6], 100);
        w.push(WindowMsg { author_id: 99, ..msg(11) }, 100);
        w.push_tied(WindowMsg { author_id: 5, ..msg(20) }, 21, vec![], 100);
        w.push(WindowMsg { author_id: 99, ..msg(21) }, 100);
        w.remove(11);
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [20, 21]);
        // Someone the command was about opts out: the whole exchange goes,
        // the answer included.
        w.push_tied(WindowMsg { author_id: 5, ..msg(30) }, 31, vec![6], 100);
        w.push(WindowMsg { author_id: 99, ..msg(31) }, 100);
        assert!(w.remove_author(6));
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [20, 21]);
        // A follow-up joins the tie: hand-deleting it leaves the rest, and
        // hand-deleting the header message (41) takes the whole exchange.
        w.push_tied(WindowMsg { author_id: 5, ..msg(40) }, 41, vec![7], 100);
        w.push(WindowMsg { author_id: 99, ..msg(41) }, 100);
        w.push(WindowMsg { author_id: 99, ..msg(42) }, 100);
        w.tie_another(40, 42);
        w.tie_another(40, 42); // joining twice is once
        w.tie_another(77, 42); // no such turn: a no-op
        w.remove(42);
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [20, 21, 40, 41]);
        w.remove(41);
        assert_eq!(ids(&w.messages().cloned().collect::<Vec<_>>()), [20, 21]);
        assert_eq!(w.ties.len(), 1, "only the live exchange (20) keeps its tie");
        assert_eq!(w.ties[0].turn, 20);
        // The requester opting out takes all their exchanges whole, answers
        // included (turn 20 is also theirs).
        w.push_tied(WindowMsg { author_id: 5, ..msg(50) }, 51, vec![], 100);
        w.push(WindowMsg { author_id: 99, ..msg(51) }, 100);
        assert!(w.remove_author(5));
        assert!(w.is_empty(), "{:?}", ids(&w.messages().cloned().collect::<Vec<_>>()));
        assert!(w.ties.is_empty());
        // Asking twice keeps one tie per turn (the second message changes nothing).
        let mut d = ChannelWindow::default();
        d.push_tied(msg(1), 2, vec![], 100);
        d.push_tied(msg(1), 9, vec![], 100);
        assert_eq!(d.ties.len(), 1);
        d.remove(9);
        assert!(d.contains(1), "the duplicate's message is no tie");
        assert_eq!(d.author_of(1), Some(7));
        assert_eq!(d.author_of(99), None);
        // A tie whose turn is outside the window is never added.
        let mut a = ChannelWindow::after(50);
        a.push_tied(msg(40), 51, vec![6], 100);
        assert!(a.ties.is_empty());
    }

    #[test]
    fn a_command_answers_retry_keeps_the_requesters_turn() {
        const BOT: u64 = 99;
        let mut w = ChannelWindow::default();
        w.push(msg(1), 100);
        // /roast: the requester's turn (10, the interaction), then the answer (11).
        w.push_tied(WindowMsg { author_id: 5, text: "[used /roast]".into(), ..msg(10) }, 11, vec![6], 100);
        w.push(WindowMsg { author_id: BOT, text: "v1".into(), ..msg(11) }, 100);
        let turn = |text: &str| vec![WindowMsg { author_id: BOT, text: text.into(), ..msg(11) }];
        // A /retry (an admin rewrite and /undo-retry are the same operation).
        w.replace_turn_tied(&[11], turn("v2"), Some(10), 100);
        let texts = |w: &ChannelWindow| w.messages().map(|m| m.text.clone()).collect::<Vec<_>>();
        assert_eq!(texts(&w), ["m1", "[used /roast]", "v2"]);
        w.replace_turn_tied(&[11], turn("v1"), Some(10), 100); // undo
        assert_eq!(texts(&w), ["m1", "[used /roast]", "v1"]);
        // The retry moved the tie to its messages: a follow-up the retry
        // added leaves alone, the header still takes the exchange.
        let two = vec![
            WindowMsg { author_id: BOT, text: "v2a".into(), ..msg(11) },
            WindowMsg { author_id: BOT, text: "v2b".into(), ..msg(12) },
        ];
        w.replace_turn_tied(&[11], two, Some(10), 100);
        assert_eq!(w.ties[0].messages, [11, 12], "the tie follows the retry");
        w.remove(12);
        assert_eq!(texts(&w), ["m1", "[used /roast]", "v2a"]);
        // An empty retry (nothing generated) keeps the tie on the old
        // messages, so hand-deleting the card still drops the turn.
        w.replace_turn_tied(&[11], Vec::new(), Some(10), 100);
        assert_eq!(w.ties[0].messages, [11]);
        w.replace_turn_tied(&[], turn("v1"), Some(10), 100);
        // The roasted user opts out: the whole exchange goes (floob's roast
        // of them too), and a retry no longer brings it back.
        assert!(w.remove_author(6));
        assert_eq!(texts(&w), ["m1"]);
        w.replace_turn_tied(&[11], turn("v3"), Some(10), 100);
        assert_eq!(texts(&w), ["m1"]);
        // A turn behind the anchor is gone like any other: the answer of an
        // exchange the window no longer holds stays out.
        let mut a = ChannelWindow::after(10);
        a.replace_turn_tied(&[], turn("v4"), Some(10), 100);
        assert!(a.is_empty(), "no answer without its question");
        // An untied turn (a channel reply) is never gated.
        w.replace_turn_tied(&[], turn("v4"), None, 100);
        assert_eq!(texts(&w), ["m1", "v4"]);
    }

    #[test]
    fn a_joined_follow_up_leaves_with_the_exchange() {
        // `tie_another` joins the follow-up (3) to the exchange: the person
        // the command was about opting out takes it too.
        let mut f = ChannelWindow::default();
        f.push_tied(WindowMsg { author_id: 5, ..msg(1) }, 2, vec![6], 100);
        f.push(WindowMsg { author_id: 99, ..msg(2) }, 100);
        f.push(WindowMsg { author_id: 99, ..msg(3) }, 100);
        f.tie_another(1, 3);
        assert!(f.remove_author(6));
        assert!(f.is_empty(), "{:?}", ids(&f.messages().cloned().collect::<Vec<_>>()));
    }

    #[test]
    fn ties_leave_when_the_anchor_passes_their_turn() {
        let mut w = ChannelWindow::default();
        w.push_tied(WindowMsg { author_id: 5, ..msg(1) }, 2, vec![6], 100);
        w.push(WindowMsg { author_id: 99, ..msg(2) }, 100);
        for id in 3..=12 {
            w.push(msg(id), 10); // the buffer cap moves the anchor past 1
        }
        assert!(!w.contains(1));
        assert!(w.ties.is_empty(), "no tie without its turn");
        // The anchor policy (select) prunes too, taking the exchange's
        // answer (2) with its turn (1).
        let mut v = ChannelWindow::default();
        v.push_tied(msg(1), 2, vec![], 100);
        for id in 2..=6 {
            v.push(msg(id), 100);
        }
        v.select(4, &|_| true);
        assert!(v.ties.is_empty());
        assert!(!v.contains(2), "the answer leaves with its scrolled-out turn");
        // And a frozen copy's anchor jump carried over (advance_to).
        let mut u = ChannelWindow::default();
        u.push_tied(msg(1), 2, vec![], 100);
        u.push(msg(2), 100);
        u.advance_to(Some(1));
        assert!(u.ties.is_empty());
    }

    #[test]
    fn buffer_is_bounded() {
        let mut w = ChannelWindow::default();
        for id in 1..=50 {
            w.push(msg(id), 40);
        }
        assert_eq!(w.len(), 40);
        assert_eq!(w.anchor(), Some(10));
    }
}
