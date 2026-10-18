//! What the bot was up to between its replies (design.md §7): after an hour
//! without a reply in a channel, however busy it was, a scene is drawn
//! (`discord::dispatch::watch_idle`, or the next reply itself), and the next
//! reply there gets it as a line right after the history, "(earlier, before
//! this chat, floob was digging a new tunnel. he felt satisfied admiring a
//! fine flat rock. he felt annoyed that it is not warm.)", from the persona's
//! `idle_activities` and `idle_thoughts`. The latest messages are copied again
//! after it (`chat::prompt::RECENT_MESSAGES`), so the reply answers them.
//!
//! Activities come from a shuffled deck per channel, without repeats until it
//! runs out (then it reshuffles). Thoughts are put together fresh each time,
//! Dwarf Fortress-style: a random emotion and a random situation, its
//! `{thing}` and `{creature}` filled at random, and about half the time a
//! second feeling. Decks and scenes live in memory only, so a restart starts
//! afresh. A scene lasts [`SCENE_FOR_MS`] from the first reply that gets it;
//! after that the channel has none until the bot is silent there for another
//! hour. `/super-admin reset` and startup (`discord::dispatch::warm_at_boot`)
//! count as such a spell. A reply after one with no scene drawn yet draws one
//! itself. Reply times are in memory too: a channel with no reply since
//! startup counts as silent.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Result, ensure};
use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use serde::{Deserialize, Serialize};

/// This long without a reply in a channel gets a scene.
pub const IDLE_GAP_MS: u64 = 60 * 60 * 1000;
/// How long a scene stays in the prompt, from the first reply that gets it.
pub const SCENE_FOR_MS: u64 = 15 * 60 * 1000;

/// The parts of a Dwarf Fortress-style thought, in a persona file:
/// "felt {emotion} {situation}." and, half the time, "felt {emotion}
/// {afterthought}.". Situations and afterthoughts may use `{thing}` and
/// `{creature}`, filled from those lists (each use drawn on its own).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename = "Thoughts")]
pub struct Thoughts {
    /// "satisfied", "nothing", "grouchy" …: whatever reads after "felt".
    pub emotions: Vec<String>,
    /// "admiring a fine {thing}", "when a {creature} ate his {thing}" …
    pub situations: Vec<String>,
    /// The optional second feeling: "that it is limestone", "about the lack
    /// of chairs at the funeral" …
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub afterthoughts: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub things: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub creatures: Vec<String>,
}

impl Thoughts {
    /// Nothing at all set (left out of a persona's fingerprint).
    pub fn is_unset(&self) -> bool {
        *self == Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.emotions.is_empty() || self.situations.is_empty()
    }

    /// Checked when the persona loads: every placeholder has a list, and
    /// there are no others.
    pub fn validate(&self) -> Result<()> {
        if self.emotions.is_empty() && self.situations.is_empty() {
            return Ok(());
        }
        ensure!(!self.is_empty(), "idle_thoughts need both emotions and situations");
        for t in self.situations.iter().chain(&self.afterthoughts) {
            let rest = t.replace("{thing}", "").replace("{creature}", "");
            ensure!(!rest.contains(['{', '}']), "idle_thoughts: unknown placeholder in {t:?} (allowed: {{thing}}, {{creature}})");
            ensure!(!t.contains("{thing}") || !self.things.is_empty(), "idle_thoughts: {t:?} uses {{thing}} but there are no things");
            ensure!(!t.contains("{creature}") || !self.creatures.is_empty(), "idle_thoughts: {t:?} uses {{creature}} but there are no creatures");
        }
        Ok(())
    }

    /// One thought: "felt satisfied admiring a fine flat rock." and maybe
    /// " felt annoyed that it is limestone.". Empty if there are no parts.
    pub fn make(&self, rng: &mut impl Rng) -> String {
        if self.is_empty() {
            return String::new();
        }
        let e = self.emotions.choose(rng).cloned().unwrap_or_default();
        let sit = self.situations.choose(rng).cloned().unwrap_or_default();
        let mut thought = format!("felt {e} {}.", self.fill(&sit, rng));
        if !self.afterthoughts.is_empty() && rng.random_bool(0.5) {
            let e2 = self.emotions.choose(rng).cloned().unwrap_or_default();
            let a = self.afterthoughts.choose(rng).cloned().unwrap_or_default();
            thought += &format!(" felt {e2} {}.", self.fill(&a, rng));
        }
        thought
    }

    /// `t` with each `{thing}` and `{creature}` replaced by a random one.
    fn fill(&self, t: &str, rng: &mut impl Rng) -> String {
        let mut out = t.to_string();
        for (slot, list) in [("{thing}", &self.things), ("{creature}", &self.creatures)] {
            while let Some(i) = out.find(slot) {
                let pick = list.choose(rng).map_or("", |s| s.as_str());
                out.replace_range(i..i + slot.len(), pick);
            }
        }
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scene {
    pub activity: String,
    pub thought: String,
    /// When it was drawn (ms since the Unix epoch).
    pub drawn_ms: u64,
    /// Until when it stays; `None` until the first reply that gets it.
    pub until_ms: Option<u64>,
}

impl Scene {
    /// The prompt line: earlier, before this chat, `bot` was (activity) …;
    /// each sentence of the thought gets "he". "before this chat" marks it as
    /// background, not something to bring up over what people are saying.
    pub fn line(&self, bot: &str) -> String {
        let mut s = String::new();
        if !self.activity.is_empty() {
            s.push_str(&format!("earlier, before this chat, {bot} was {}.", self.activity.trim().trim_end_matches('.')));
        }
        for t in self.thought.split(". ").map(|t| t.trim().trim_end_matches('.')).filter(|t| !t.is_empty()) {
            if s.is_empty() {
                s.push_str("before this chat, ");
            } else {
                s.push(' ');
            }
            s.push_str(&format!("he {t}."));
        }
        format!("({s})")
    }
}

/// A shuffled order of `0..len`, dealt without repeats.
#[derive(Debug, Default)]
struct Deck {
    order: Vec<usize>,
    len: usize,
}

impl Deck {
    fn deal(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        // A new list (persona reload) or an empty deck: shuffle afresh.
        if self.len != len || self.order.is_empty() {
            self.len = len;
            self.order = (0..len).collect();
            self.order.shuffle(&mut rand::rng());
        }
        self.order.pop()
    }
}

#[derive(Debug, Default)]
struct Channel {
    activities: Deck,
    scene: Option<Scene>,
    /// When a scene was last drawn or ended: one per silent spell.
    drawn_ms: Option<u64>,
    /// The latest reply's message time (`None`: none since startup).
    replied_ms: Option<u64>,
}

/// Every channel's deck and current scene.
#[derive(Debug, Default)]
pub struct Idle {
    channels: Mutex<HashMap<u64, Channel>>,
}

impl Idle {
    /// Draws `channel`'s scene once the bot hasn't replied there for
    /// [`IDLE_GAP_MS`], unless one was drawn since its last reply. Whether it
    /// drew one. A channel with no reply since startup waits for its next
    /// reply, which draws one itself.
    pub fn prepare(&self, channel: u64, now_ms: u64, activities: &[String], thoughts: &Thoughts) -> bool {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let ch = map.entry(channel).or_default();
        let Some(r) = ch.replied_ms else { return false };
        if now_ms < r + IDLE_GAP_MS || ch.drawn_ms.is_some_and(|d| d > r) {
            return false;
        }
        Self::draw(ch, now_ms, None, activities, thoughts);
        ch.scene.is_some()
    }

    /// The scene for a reply in `channel` to a message at `now_ms`, which
    /// counts as the bot's latest reply there from now on. The scene drawn
    /// since the previous reply holds; [`IDLE_GAP_MS`] or more since it (or
    /// none since startup) without one draws one now. It lasts
    /// [`SCENE_FOR_MS`] from the first reply that gets it. `None` without
    /// activities and thoughts.
    pub fn scene(&self, channel: u64, now_ms: u64, activities: &[String], thoughts: &Thoughts) -> Option<Scene> {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let ch = map.entry(channel).or_default();
        let prev = ch.replied_ms;
        let drawn = ch.drawn_ms.is_some_and(|d| prev.is_none_or(|r| d > r));
        if prev.is_none_or(|r| now_ms >= r + IDLE_GAP_MS) && !drawn {
            Self::draw(ch, now_ms, None, activities, thoughts);
        }
        // A retry answers its message again: no later than the latest.
        ch.replied_ms = Some(prev.map_or(now_ms, |r| r.max(now_ms)));
        let s = ch.scene.as_mut()?;
        let until = *s.until_ms.get_or_insert(now_ms + SCENE_FOR_MS);
        (now_ms < until).then(|| s.clone())
    }

    /// Starts a new scene in `channel` now (`/super-admin idle-next`).
    pub fn next(&self, channel: u64, now_ms: u64, activities: &[String], thoughts: &Thoughts) -> Option<Scene> {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let ch = map.entry(channel).or_default();
        Self::draw(ch, now_ms, Some(now_ms + SCENE_FOR_MS), activities, thoughts);
        ch.scene.clone()
    }

    /// `channel`'s scene, expired or not (`/super-admin idle`).
    pub fn current(&self, channel: u64) -> Option<Scene> {
        self.channels.lock().unwrap_or_else(|e| e.into_inner()).get(&channel).and_then(|c| c.scene.clone())
    }

    /// Ends `channel`'s scene; the next one waits for an hour without a reply.
    pub fn clear(&self, channel: u64, now_ms: u64) {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let ch = map.entry(channel).or_default();
        ch.scene = None;
        ch.drawn_ms = Some(now_ms);
    }

    /// Draws a new scene for `channel`'s next reply (`/super-admin reset`,
    /// startup).
    /// Whether it drew one (the caller prefills its prompt).
    pub fn reset(&self, channel: u64, now_ms: u64, activities: &[String], thoughts: &Thoughts) -> bool {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let ch = map.entry(channel).or_default();
        Self::draw(ch, now_ms, None, activities, thoughts);
        ch.scene.is_some()
    }

    fn draw(ch: &mut Channel, now_ms: u64, until_ms: Option<u64>, activities: &[String], thoughts: &Thoughts) {
        let activity = ch.activities.deal(activities.len()).map(|i| activities[i].clone()).unwrap_or_default();
        let thought = thoughts.make(&mut rand::rng());
        ch.drawn_ms = Some(now_ms);
        ch.scene = (!activity.is_empty() || !thought.is_empty()).then_some(Scene { activity, thought, drawn_ms: now_ms, until_ms });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn list(n: usize, p: &str) -> Vec<String> {
        (0..n).map(|i| format!("{p} {i}")).collect()
    }

    fn parts() -> Thoughts {
        let v = |x: &[&str]| x.iter().map(|s| s.to_string()).collect();
        Thoughts {
            emotions: v(&["satisfied", "annoyed", "nothing"]),
            situations: v(&["admiring a fine {thing}", "when a {creature} ate his {thing}"]),
            afterthoughts: v(&["that it is limestone", "about the lack of chairs at the funeral"]),
            things: v(&["flat rock", "bottlecap"]),
            creatures: v(&["cave lizard"]),
        }
    }

    const MIN: u64 = 60 * 1000;

    #[test]
    fn an_hour_without_a_reply_starts_a_scene_that_lasts_fifteen_minutes() {
        let idle = Idle::default();
        let (a, t) = (list(5, "digging"), parts());
        // The first reply since startup draws one, until 15 minutes after it.
        let s = idle.scene(1, 0, &a, &t).unwrap();
        assert_eq!(s.until_ms, Some(15 * MIN));
        assert_eq!(idle.scene(1, 10 * MIN, &a, &t), Some(s.clone()));
        // Then none, while the bot keeps replying.
        assert_eq!(idle.scene(1, 16 * MIN, &a, &t), None);
        assert_eq!(idle.scene(1, 70 * MIN, &a, &t), None);
        // An hour after its last reply, however much others said meanwhile.
        let s2 = idle.scene(1, 131 * MIN, &a, &t).unwrap();
        assert_ne!(s2.activity, s.activity);
        assert_eq!(s2.until_ms, Some(146 * MIN));
        // Other channels have their own.
        assert!(idle.scene(2, 131 * MIN, &a, &t).is_some());
    }

    #[test]
    fn a_scene_is_drawn_an_hour_after_the_last_reply_and_starts_at_the_next() {
        let idle = Idle::default();
        let (a, t) = (list(5, "digging"), parts());
        // No reply since startup: the timer waits for one.
        assert!(!idle.prepare(1, 100 * MIN, &a, &t));
        assert!(idle.scene(1, 100 * MIN, &a, &t).is_some());
        // Last reply at 100: nothing yet at 130, a scene at 161, once.
        assert!(!idle.prepare(1, 130 * MIN, &a, &t));
        assert!(idle.prepare(1, 161 * MIN, &a, &t));
        let s = idle.current(1).unwrap();
        assert_eq!((s.drawn_ms, s.until_ms), (161 * MIN, None));
        assert!(!idle.prepare(1, 162 * MIN, &a, &t));
        // A reply much later keeps it and starts its 15 minutes.
        let got = idle.scene(1, 300 * MIN, &a, &t).unwrap();
        assert_eq!((got.activity.as_str(), got.until_ms), (s.activity.as_str(), Some(315 * MIN)));
        // A retry of an earlier message doesn't move the last reply back.
        assert!(idle.scene(1, 290 * MIN, &a, &t).is_some());
        assert!(!idle.prepare(1, 359 * MIN, &a, &t));
        // Ended by hand: not drawn again until the bot replies and goes silent again.
        assert!(idle.prepare(1, 361 * MIN, &a, &t));
        idle.clear(1, 362 * MIN);
        assert!(!idle.prepare(1, 500 * MIN, &a, &t));
        assert_eq!(idle.scene(1, 600 * MIN, &a, &t), None);
        assert!(idle.prepare(1, 661 * MIN, &a, &t));
    }

    #[test]
    fn a_reset_starts_a_scene_at_the_next_reply_whatever_the_gap() {
        let idle = Idle::default();
        let (a, t) = (list(5, "digging"), parts());
        let s = idle.scene(1, 0, &a, &t).unwrap();
        assert!(idle.reset(1, 2 * MIN, &a, &t));
        let s2 = idle.scene(1, 5 * MIN, &a, &t).unwrap();
        assert_ne!(s2.activity, s.activity);
        assert_eq!(s2.until_ms, Some(20 * MIN));
        // Only that once: later replies keep it.
        assert_eq!(idle.scene(1, 6 * MIN, &a, &t), Some(s2));
        // At startup, before any reply.
        let idle = Idle::default();
        assert!(idle.reset(1, 7 * MIN, &a, &t));
        let s3 = idle.scene(1, 8 * MIN, &a, &t).unwrap();
        assert_eq!(s3.drawn_ms, 7 * MIN);
    }

    #[test]
    fn activities_dont_repeat_until_the_deck_runs_out() {
        let idle = Idle::default();
        let a = list(10, "a");
        let mut seen = std::collections::HashSet::new();
        for i in 0..10u64 {
            let s = idle.next(7, i, &a, &Thoughts::default()).unwrap();
            assert!(seen.insert(s.activity), "repeat before the deck ran out");
        }
        assert!(idle.next(7, 11, &a, &Thoughts::default()).is_some(), "reshuffles");
    }

    #[test]
    fn thoughts_are_put_together_from_parts() {
        let t = parts();
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let made: Vec<String> = (0..40).map(|_| t.make(&mut rng)).collect();
        for m in &made {
            assert!(m.starts_with("felt ") && m.ends_with('.') && !m.contains('{'), "{m}");
        }
        // Varied, and sometimes with a second feeling.
        assert!(made.iter().collect::<std::collections::HashSet<_>>().len() > 10);
        assert!(made.iter().any(|m| m.matches("felt ").count() == 2));
        assert!(made.iter().any(|m| m.matches("felt ").count() == 1));
        let mut bad = parts();
        bad.situations.push("near a {monster}".into());
        assert!(bad.validate().is_err());
        bad = parts();
        bad.creatures.clear();
        assert!(bad.validate().is_err());
        assert!(parts().validate().is_ok() && Thoughts::default().validate().is_ok());
    }

    #[test]
    fn the_line_reads_like_a_note() {
        let s = Scene {
            activity: "digging a new tunnel".into(),
            thought: "felt satisfied admiring a fine flat rock. felt annoyed that it is limestone.".into(),
            drawn_ms: 0,
            until_ms: None,
        };
        assert_eq!(
            s.line("floob"),
            "(earlier, before this chat, floob was digging a new tunnel. he felt satisfied admiring a fine flat rock. he felt annoyed that it is limestone.)"
        );
        assert_eq!(Idle::default().scene(1, 0, &[], &Thoughts::default()), None);
    }
}
