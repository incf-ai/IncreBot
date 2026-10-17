//! What happened to each recent message, for "Admin actions" → Trace (design.md
//! §6): per channel, the last [`KEEP`] messages the bot saw there, each with
//! a timed, ordered list of what was decided and done about it (the reply
//! rule, the queue and the channel pause, what the prompt was built with, how
//! generation ended, retries); and for a message a channel reply answers,
//! that reply's context ("Admin actions" → Context): the exact prompt and
//! what went with it, the latest retry's replacing it. In memory only, never
//! logged: a restart forgets it. Opted-out users' and `//` messages are never
//! recorded.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

/// Messages kept per channel.
pub const KEEP: usize = 20;

/// Events kept per message (a message retried many times stops growing).
const MAX_EVENTS: usize = 200;

struct Trace {
    message: u64,
    author_id: u64,
    author: String,
    preview: String,
    seen: SystemTime,
    start: Instant,
    events: Vec<(Duration, String)>,
    /// The latest reply's context, if a channel reply answered it.
    context: Option<String>,
}

/// Every channel's recent traces.
#[derive(Default)]
pub struct Traces {
    channels: Mutex<HashMap<u64, VecDeque<Trace>>>,
}

/// One traced message, for picking it.
pub struct Listed {
    pub message: u64,
    pub author: String,
    pub preview: String,
    pub seen: SystemTime,
}

impl Traces {
    /// Starts recording `message` in `channel`, dropping the oldest past [`KEEP`].
    pub fn start(&self, channel: u64, message: u64, author_id: u64, author: &str, text: &str) {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let list = map.entry(channel).or_default();
        if list.iter().any(|t| t.message == message) {
            return;
        }
        list.push_back(Trace {
            message,
            author_id,
            author: author.to_string(),
            preview: text.chars().take(80).collect::<String>().replace('\n', " "),
            seen: SystemTime::now(),
            start: Instant::now(),
            events: Vec::new(),
            context: None,
        });
        while list.len() > KEEP {
            list.pop_front();
        }
    }

    /// Adds an event to `message`'s trace, if it is still kept.
    pub fn add(&self, channel: u64, message: u64, event: impl Into<String>) {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let Some(t) = map.get_mut(&channel).and_then(|l| l.iter_mut().find(|t| t.message == message)) else {
            return;
        };
        if t.events.len() < MAX_EVENTS {
            t.events.push((t.start.elapsed(), event.into()));
        }
    }

    /// Sets the context of the reply to `message`, if it is still kept,
    /// replacing an earlier one's.
    pub fn set_context(&self, channel: u64, message: u64, context: String) {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = map.get_mut(&channel).and_then(|l| l.iter_mut().find(|t| t.message == message)) {
            t.context = Some(context);
        }
    }

    /// The context of the reply to `message`, if any.
    pub fn context(&self, channel: u64, message: u64) -> Option<String> {
        let map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        map.get(&channel)?.iter().find(|t| t.message == message)?.context.clone()
    }

    /// `channel`'s traced messages, newest first.
    pub fn list(&self, channel: u64) -> Vec<Listed> {
        let map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        map.get(&channel)
            .map(|l| {
                l.iter()
                    .rev()
                    .map(|t| Listed { message: t.message, author: t.author.clone(), preview: t.preview.clone(), seen: t.seen })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `message`'s trace as text: one `+seconds  event` line each, in order.
    pub fn render(&self, channel: u64, message: u64) -> Option<String> {
        let map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        let t = map.get(&channel)?.iter().find(|t| t.message == message)?;
        let mut out = format!("message {} by {}: {}\n", t.message, t.author, t.preview);
        for (at, e) in &t.events {
            out += &format!("+{:>7.3}s  {e}\n", at.as_secs_f64());
        }
        if t.events.len() >= MAX_EVENTS {
            out += "(more events not kept)\n";
        }
        Some(out)
    }

    /// `/opt-out`: every trace of `user`'s messages goes, and every context,
    /// as any prompt may hold their words.
    pub fn forget_author(&self, user: u64) {
        let mut map = self.channels.lock().unwrap_or_else(|e| e.into_inner());
        for list in map.values_mut() {
            list.retain(|t| t.author_id != user);
            for t in list.iter_mut() {
                t.context = None;
            }
        }
    }

    /// `/super-admin reset`: the channel's traces go too.
    pub fn forget(&self, channel: u64) {
        self.channels.lock().unwrap_or_else(|e| e.into_inner()).remove(&channel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_last_twenty_in_order() {
        let t = Traces::default();
        for m in 1..=25 {
            t.start(1, m, 9, "ann", &format!("hello {m}"));
            t.add(1, m, "replying: dedicated channel");
        }
        t.add(1, 25, "queued");
        let list = t.list(1);
        assert_eq!(list.len(), KEEP);
        assert_eq!((list[0].message, list[KEEP - 1].message), (25, 6));
        assert!(t.render(1, 5).is_none(), "dropped");
        let r = t.render(1, 25).unwrap();
        let lines: Vec<&str> = r.lines().collect();
        assert_eq!(lines[0], "message 25 by ann: hello 25");
        assert!(lines[1].ends_with("replying: dedicated channel") && lines[2].ends_with("queued"));
        assert!(t.list(2).is_empty());
        t.add(2, 1, "nothing to add to");
        t.set_context(1, 25, "first".into());
        t.set_context(1, 25, "retried".into());
        assert_eq!(t.context(1, 25).as_deref(), Some("retried"));
        assert!(t.context(1, 24).is_none() && t.context(1, 5).is_none());
        t.start(3, 1, 8, "bob", "hi");
        t.forget_author(8);
        assert!(t.list(3).is_empty());
        assert!(t.context(1, 25).is_none(), "an opt-out drops every context");
        t.forget(1);
        assert!(t.list(1).is_empty());
    }
}
