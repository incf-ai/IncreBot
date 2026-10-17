//! A backend that needs no model: replies come from a function and stream out a
//! word at a time. `chatsim --mock` uses it, and so do the tests.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use crate::backend::{Backend, BackendEvent, ReplyRequest, ReplyStats, StopReason};

/// How fast scripted replies stream.
#[derive(Clone, Copy, Debug)]
pub enum Pacing {
    /// The whole reply on the first poll.
    Instant,
    /// This many words per poll (deterministic, for tests).
    PerPoll(usize),
    /// One word per interval of wall-clock time (for `--mock`).
    Every(Duration),
}

pub type Responder = Box<dyn FnMut(&ReplyRequest) -> String + Send>;

/// What tests can see of a [`ScriptedBackend`] after handing it to an app.
#[derive(Clone, Default)]
pub struct ScriptHandle {
    requests: Arc<Mutex<Vec<ReplyRequest>>>,
    resets: Arc<AtomicUsize>,
}

impl ScriptHandle {
    /// Every request started so far, oldest first.
    pub fn requests(&self) -> Vec<ReplyRequest> {
        self.requests.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn resets(&self) -> usize {
        self.resets.load(Ordering::Relaxed)
    }
}

struct Active {
    req: ReplyRequest,
    chunks: VecDeque<String>,
    stop: StopReason,
    started: bool,
    generated: usize,
    t0: Instant,
    next_due: Instant,
}

pub struct ScriptedBackend {
    responder: Responder,
    pacing: Pacing,
    active: Option<Active>,
    /// Prompt tokens reused by the previous channel job, as a cache would.
    cached: Vec<String>,
    handle: ScriptHandle,
}

impl ScriptedBackend {
    pub fn new(
        pacing: Pacing,
        responder: impl FnMut(&ReplyRequest) -> String + Send + 'static,
    ) -> Self {
        Self {
            responder: Box::new(responder),
            pacing,
            active: None,
            cached: Vec::new(),
            handle: ScriptHandle::default(),
        }
    }

    /// The `--mock` persona: a short, deterministic reply about the trigger.
    pub fn mock(pacing: Pacing) -> Self {
        Self::new(pacing, mock_reply)
    }

    pub fn handle(&self) -> ScriptHandle {
        self.handle.clone()
    }

    fn take_due(&mut self, now: Instant) -> Vec<String> {
        let Some(a) = self.active.as_mut() else {
            return Vec::new();
        };
        let n = match self.pacing {
            Pacing::Instant => a.chunks.len(),
            Pacing::PerPoll(n) => n.max(1),
            Pacing::Every(d) => {
                let mut n = 0;
                while a.next_due <= now && n < a.chunks.len() {
                    a.next_due += d;
                    n += 1;
                }
                n
            }
        };
        let n = n.min(a.chunks.len());
        a.generated += n;
        a.chunks.drain(..n).collect()
    }
}

/// Splits into words that keep their leading whitespace, so they concatenate back.
fn chunks(text: &str) -> VecDeque<String> {
    let mut out = VecDeque::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_whitespace() && !cur.trim().is_empty() {
            out.push_back(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    match out.back_mut() {
        // Trailing whitespace is not a word of its own.
        Some(last) if cur.trim().is_empty() => last.push_str(&cur),
        _ if !cur.is_empty() => out.push_back(cur),
        _ => {}
    }
    out
}

/// Cuts `text` at the earliest stop string, ignoring ASCII case, as the
/// engine's holdback would.
fn apply_stops(text: &str, stops: &[String]) -> (String, StopReason) {
    let folded = text.to_ascii_lowercase();
    match stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| folded.find(&s.to_ascii_lowercase()))
        .min()
    {
        Some(at) => (text[..at].to_string(), StopReason::StopString),
        None => (text.to_string(), StopReason::Eos),
    }
}

const MOCK_LINES: [&str; 4] = [
    "{author}, that's a good point about \"{text}\".",
    "Ha, I hear you, {author}. \"{text}\" is a fair thing to say.",
    "Interesting, {author}! Tell me more about \"{text}\".",
    "Noted, {author}: \"{text}\". I'm only the mock backend, though.",
];

/// The `--mock` reply: one of a few templates, picked by the trigger text so the
/// same conversation always gets the same replies. Starts with a space, like a
/// model continuing `"<bot>:"`.
pub fn mock_reply(req: &ReplyRequest) -> String {
    let text: String = req.trigger_text.chars().take(60).collect();
    let pick = text
        .bytes()
        .fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
    let line = MOCK_LINES[pick % MOCK_LINES.len()]
        .replace("{author}", &req.trigger_author)
        .replace("{text}", &text);
    format!(" {line}")
}

impl Backend for ScriptedBackend {
    fn describe(&self) -> String {
        "mock (scripted replies, no model)".into()
    }

    fn count_tokens(&self, text: &str) -> usize {
        1 + text.split_whitespace().count()
    }

    fn start(&mut self, req: ReplyRequest) -> Result<()> {
        if self.active.is_some() {
            bail!("a reply is already being generated");
        }
        self.handle.requests.lock().unwrap_or_else(|e| e.into_inner()).push(req.clone());
        let (text, stop) = apply_stops(&(self.responder)(&req), &req.stops);
        let now = Instant::now();
        self.active = Some(Active {
            chunks: chunks(&text),
            stop,
            started: false,
            generated: 0,
            t0: now,
            next_due: now,
            req,
        });
        Ok(())
    }

    fn poll(&mut self) -> Vec<(u64, BackendEvent)> {
        let Some(a) = self.active.as_mut() else {
            return Vec::new();
        };
        let job = a.req.job;
        let mut events = Vec::new();
        if !a.started {
            a.started = true;
            events.push((job, BackendEvent::Started));
        }
        let cancelled = a.req.cancel.load(Ordering::Relaxed);
        if !cancelled {
            let text: String = self.take_due(Instant::now()).concat();
            if !text.is_empty() {
                events.push((job, BackendEvent::Text(text)));
            }
        }
        // One check instead of two: `take_if` decides and takes atomically,
        // so there's nothing for a second `self.active` read to disagree
        // with (and so nothing to `.expect()` on it).
        if let Some(a) = self.active.take_if(|a| cancelled || a.chunks.is_empty()) {
            let words: Vec<String> = a.req.prompt.split_whitespace().map(String::from).collect();
            let reused = if a.req.scratch {
                0
            } else {
                let r = self
                    .cached
                    .iter()
                    .zip(&words)
                    .take_while(|(x, y)| x == y)
                    .count();
                self.cached = words.clone();
                r
            };
            let elapsed = a.t0.elapsed();
            events.push((
                job,
                BackendEvent::Finished(ReplyStats {
                    stop: if cancelled {
                        StopReason::Cancelled
                    } else {
                        a.stop
                    },
                    prompt_tokens: 1 + words.len(),
                    reused_tokens: reused,
                    generated_tokens: a.generated,
                    prefill: Duration::ZERO,
                    decode: elapsed,
                }),
            ));
        }
        events
    }

    fn reset(&mut self) {
        self.cached.clear();
        self.handle.resets.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn req(job: u64, prompt: &str) -> ReplyRequest {
        ReplyRequest {
            job,
            prompt: prompt.into(),
            stops: vec!["\nBob:".into()],
            scratch: false,
            history: String::new(),
            message: "hi".into(),
            previous: None,
            cancel: Arc::new(AtomicBool::new(false)),
            trigger_author: "Alice".into(),
            trigger_text: "hi".into(),
        }
    }

    fn texts(events: &[(u64, BackendEvent)]) -> String {
        events
            .iter()
            .filter_map(|(_, e)| match e {
                BackendEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    fn finished(events: &[(u64, BackendEvent)]) -> Option<ReplyStats> {
        events.iter().find_map(|(_, e)| match e {
            BackendEvent::Finished(s) => Some(s.clone()),
            _ => None,
        })
    }

    #[test]
    fn chunks_concatenate_back() {
        let s = "  one two\n three  ";
        let c = chunks(s);
        assert_eq!(c.iter().map(String::as_str).collect::<String>(), s);
        assert_eq!(c.len(), 3);
        assert!(chunks("").is_empty());
    }

    #[test]
    fn instant_pacing_finishes_in_one_poll() {
        let mut b = ScriptedBackend::new(Pacing::Instant, |_| " hello there".into());
        b.start(req(7, "a b c")).unwrap();
        let ev = b.poll();
        assert_eq!(ev[0], (7, BackendEvent::Started));
        assert_eq!(texts(&ev), " hello there");
        let s = finished(&ev).unwrap();
        assert_eq!(
            (s.stop, s.generated_tokens, s.prompt_tokens),
            (StopReason::Eos, 2, 4)
        );
        assert!(b.poll().is_empty());
    }

    #[test]
    fn per_poll_pacing_streams_words() {
        let mut b = ScriptedBackend::new(Pacing::PerPoll(1), |_| " a b c".into());
        b.start(req(1, "p")).unwrap();
        assert_eq!(texts(&b.poll()), " a");
        assert_eq!(texts(&b.poll()), " b");
        let last = b.poll();
        assert_eq!(texts(&last), " c");
        assert!(finished(&last).is_some());
    }

    #[test]
    fn timed_pacing_releases_words_as_time_passes() {
        let mut b =
            ScriptedBackend::new(Pacing::Every(Duration::from_millis(5)), |_| " a b c".into());
        b.start(req(1, "p")).unwrap();
        let mut all = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while finished(&all).is_none() && Instant::now() < deadline {
            all.extend(b.poll());
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(texts(&all), " a b c");
    }

    #[test]
    fn stop_strings_cut_the_reply() {
        let mut b = ScriptedBackend::new(Pacing::Instant, |_| " sure\nbob: I agree".into());
        b.start(req(1, "p")).unwrap();
        let ev = b.poll();
        assert_eq!(texts(&ev), " sure");
        assert_eq!(finished(&ev).unwrap().stop, StopReason::StopString);
    }

    #[test]
    fn cancel_stops_mid_reply() {
        let mut b = ScriptedBackend::new(Pacing::PerPoll(1), |_| " a b c d".into());
        let r = req(1, "p");
        let flag = r.cancel.clone();
        b.start(r).unwrap();
        b.poll();
        flag.store(true, Ordering::Relaxed);
        let ev = b.poll();
        assert_eq!(texts(&ev), "");
        let s = finished(&ev).unwrap();
        assert_eq!((s.stop, s.generated_tokens), (StopReason::Cancelled, 1));
    }

    #[test]
    fn one_job_at_a_time_and_requests_are_recorded() {
        let mut b = ScriptedBackend::mock(Pacing::PerPoll(1));
        let h = b.handle();
        b.start(req(1, "p")).unwrap();
        assert!(b.start(req(2, "p")).is_err());
        assert_eq!(h.requests().len(), 1);
        b.reset();
        assert_eq!(h.resets(), 1);
    }

    #[test]
    fn reuse_counts_the_shared_prompt_prefix() {
        let mut b = ScriptedBackend::new(Pacing::Instant, |_| " ok".into());
        b.start(req(1, "sys a b")).unwrap();
        assert_eq!(finished(&b.poll()).unwrap().reused_tokens, 0);
        b.start(req(2, "sys a b c")).unwrap();
        assert_eq!(finished(&b.poll()).unwrap().reused_tokens, 3);
        let mut scratch = req(3, "sys a b c");
        scratch.scratch = true;
        b.start(scratch).unwrap();
        assert_eq!(finished(&b.poll()).unwrap().reused_tokens, 0);
        b.reset();
        b.start(req(4, "sys a b c")).unwrap();
        assert_eq!(finished(&b.poll()).unwrap().reused_tokens, 0);
    }

    #[test]
    fn mock_reply_is_deterministic_and_names_the_author() {
        let r = req(1, "p");
        let a = mock_reply(&r);
        assert_eq!(a, mock_reply(&r));
        assert!(a.starts_with(' ') && a.contains("Alice") && a.contains("hi"));
    }
}
