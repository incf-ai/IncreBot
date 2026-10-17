//! The front-end-facing state: a room, a backend, the input line and the scroll
//! position, driven by [`Action`]s and [`App::pump`].

use anyhow::Result;

use crate::backend::{Backend, BackendEvent};
use crate::input::InputLine;
use crate::room::{Effect, Room};

/// Everything a front end can ask for; keys map onto these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Insert(char),
    Backspace,
    Delete,
    DeleteWord,
    ClearLine,
    Left,
    Right,
    Home,
    End,
    HistoryPrev,
    HistoryNext,
    Submit,
    /// Scroll the transcript by lines (up = back in time).
    ScrollUp(usize),
    ScrollDown(usize),
    ScrollBottom,
    /// Stop the reply in progress (Esc).
    Cancel,
    Quit,
}

pub struct App<B> {
    pub room: Room,
    pub backend: B,
    pub input: InputLine,
    /// Lines scrolled back from the newest; 0 follows new messages.
    scroll: usize,
    quit: bool,
}

impl<B: Backend> App<B> {
    /// Fails if the persona's header is over `max_header_tokens`, leaving too
    /// little room for history, as `increbot serve` does at startup.
    pub fn new(room: Room, backend: B) -> Result<Self> {
        room.check_header(&|s| backend.count_tokens(s))?;
        Ok(Self {
            room,
            backend,
            input: InputLine::default(),
            scroll: 0,
            quit: false,
        })
    }

    pub fn should_quit(&self) -> bool {
        self.quit
    }

    pub fn scroll(&self) -> usize {
        self.scroll
    }

    /// Keeps the scroll inside the transcript; the renderer knows its height.
    pub fn clamp_scroll(&mut self, max: usize) {
        self.scroll = self.scroll.min(max);
    }

    pub fn handle(&mut self, action: Action) {
        let input = &mut self.input;
        match action {
            Action::Insert(c) => input.insert(c),
            Action::Backspace => input.backspace(),
            Action::Delete => input.delete(),
            Action::DeleteWord => input.delete_word(),
            Action::ClearLine => input.clear(),
            Action::Left => input.left(),
            Action::Right => input.right(),
            Action::Home => input.home(),
            Action::End => input.end(),
            Action::HistoryPrev => input.history_prev(),
            Action::HistoryNext => input.history_next(),
            Action::ScrollUp(n) => self.scroll = self.scroll.saturating_add(n),
            Action::ScrollDown(n) => self.scroll = self.scroll.saturating_sub(n),
            Action::ScrollBottom => self.scroll = 0,
            Action::Cancel => self.room.cancel(),
            Action::Quit => self.apply(Effect::Quit),
            Action::Submit => {
                let line = self.input.take();
                self.scroll = 0;
                for effect in self.room.submit(&line) {
                    self.apply(effect);
                }
                self.pump();
            }
        }
    }

    fn apply(&mut self, effect: Effect) {
        match effect {
            Effect::Quit => {
                if !self.quit {
                    self.room.submit("/quit");
                }
                self.quit = true;
            }
            Effect::ResetCache => self.backend.reset(),
            Effect::ShowPrompt => {
                let text = match self.room.last_prompt() {
                    None => "No reply has been generated since startup or /reset.".to_string(),
                    Some(p) => format!(
                        "Last prompt ({} tokens), exactly as the model saw it:\n{p}",
                        self.backend.count_tokens(p)
                    ),
                };
                self.room.notice(text);
            }
            Effect::Stats => {
                let text = self.room.stats_text(&self.backend.describe());
                self.room.notice(text);
            }
        }
    }

    /// Feeds backend events to the room and starts the next queued reply.
    /// Returns whether anything changed (i.e. a redraw is due).
    pub fn pump(&mut self) -> bool {
        let events = self.backend.poll();
        let mut changed = !events.is_empty();
        for (job, ev) in events {
            self.room.on_event(job, ev);
        }
        let backend = &self.backend;
        if let Some(req) = self.room.next_request(&|s| backend.count_tokens(s)) {
            changed = true;
            let job = req.job;
            if let Err(e) = self.backend.start(req) {
                self.room.on_event(job, BackendEvent::Failed(e.to_string()));
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::room::{Kind, RoomConfig};
    use crate::scripted::{Pacing, ScriptedBackend};

    fn app(pacing: Pacing) -> App<ScriptedBackend> {
        let backend = ScriptedBackend::new(pacing, |r| format!(" echo {}", r.trigger_text));
        App::new(Room::new(RoomConfig::default()).unwrap(), backend).unwrap()
    }

    fn type_line(app: &mut App<ScriptedBackend>, s: &str) {
        s.chars().for_each(|c| app.handle(Action::Insert(c)));
        app.handle(Action::Submit);
    }

    fn bot_texts(app: &App<ScriptedBackend>) -> Vec<String> {
        app.room
            .messages()
            .iter()
            .filter(|m| m.kind == Kind::Bot)
            .map(|m| m.text.clone())
            .collect()
    }

    #[test]
    fn submit_starts_a_reply_and_pump_streams_it() {
        let mut a = app(Pacing::PerPoll(1));
        type_line(&mut a, "hi there");
        assert!(a.input.is_empty());
        assert_eq!(
            a.backend.handle().requests().len(),
            1,
            "submit starts the job"
        );
        assert!(a.pump());
        assert_eq!(bot_texts(&a), ["echo"]);
        while a.pump() {}
        assert_eq!(bot_texts(&a), ["echo hi there"]);
        assert!(!a.pump(), "idle pumps change nothing");
    }

    #[test]
    fn replies_queue_behind_the_running_one() {
        let mut a = app(Pacing::PerPoll(1));
        type_line(&mut a, "one with a long reply");
        type_line(&mut a, "two");
        type_line(&mut a, "three");
        assert_eq!(a.room.queued(), 2, "each waits for its own reply");
        while a.pump() {}
        assert_eq!(
            bot_texts(&a),
            ["echo one with a long reply", "echo two", "echo three"]
        );
    }

    #[test]
    fn esc_cancels_the_reply() {
        let mut a = app(Pacing::PerPoll(1));
        type_line(&mut a, "a b c d e");
        a.pump();
        a.handle(Action::Cancel);
        while a.pump() {}
        let bot = a
            .room
            .messages()
            .iter()
            .find(|m| m.kind == Kind::Bot)
            .unwrap();
        assert_eq!((bot.text.as_str(), bot.note), ("echo", Some("stopped")));
    }

    #[test]
    fn reset_and_stats_reach_the_backend() {
        let mut a = app(Pacing::Instant);
        let h = a.backend.handle();
        type_line(&mut a, "/reset");
        assert_eq!(h.resets(), 1);
        type_line(&mut a, "/stats");
        let last = a.room.messages().last().unwrap();
        assert!(last.text.contains("backend: mock"), "{}", last.text);
    }

    #[test]
    fn prompt_command_posts_the_prompt_with_its_size() {
        let mut a = app(Pacing::Instant);
        type_line(&mut a, "/prompt");
        assert!(
            a.room
                .messages()
                .last()
                .unwrap()
                .text
                .starts_with("No reply")
        );
        type_line(&mut a, "hi");
        while a.pump() {}
        type_line(&mut a, "/prompt");
        let shown = &a.room.messages().last().unwrap().text;
        let sent = &a.backend.handle().requests()[0].prompt;
        let tokens = a.backend.count_tokens(sent);
        assert_eq!(
            shown,
            &format!("Last prompt ({tokens} tokens), exactly as the model saw it:\n{sent}")
        );
    }

    #[test]
    fn quit_cancels_the_running_reply() {
        let mut a = app(Pacing::PerPoll(1));
        type_line(&mut a, "hello");
        let req = a.backend.handle().requests()[0].clone();
        a.handle(Action::Quit);
        assert!(a.should_quit());
        assert!(req.cancel.load(std::sync::atomic::Ordering::Relaxed));
        let mut b = app(Pacing::Instant);
        type_line(&mut b, "/quit");
        assert!(b.should_quit());
    }

    #[test]
    fn scrolling_clamps_and_submit_returns_to_bottom() {
        let mut a = app(Pacing::Instant);
        a.handle(Action::ScrollUp(5));
        a.handle(Action::ScrollUp(5));
        assert_eq!(a.scroll(), 10);
        a.clamp_scroll(7);
        assert_eq!(a.scroll(), 7);
        a.handle(Action::ScrollDown(3));
        assert_eq!(a.scroll(), 4);
        a.handle(Action::ScrollDown(30));
        assert_eq!(a.scroll(), 0);
        a.handle(Action::ScrollUp(2));
        type_line(&mut a, "x");
        assert_eq!(a.scroll(), 0);
        a.handle(Action::ScrollUp(2));
        a.handle(Action::ScrollBottom);
        assert_eq!(a.scroll(), 0);
    }

    #[test]
    fn editing_actions_reach_the_input_line() {
        let mut a = app(Pacing::Instant);
        for c in "hello world".chars() {
            a.handle(Action::Insert(c));
        }
        a.handle(Action::DeleteWord);
        a.handle(Action::Home);
        a.handle(Action::Delete);
        a.handle(Action::End);
        a.handle(Action::Backspace);
        a.handle(Action::Left);
        a.handle(Action::Right);
        assert_eq!(a.input.text(), "ello");
        a.handle(Action::ClearLine);
        a.handle(Action::HistoryPrev);
        a.handle(Action::HistoryNext);
        assert!(a.input.is_empty());
    }

    #[test]
    fn a_failing_backend_start_becomes_a_notice() {
        struct Broken;
        impl Backend for Broken {
            fn describe(&self) -> String {
                "broken".into()
            }
            fn count_tokens(&self, t: &str) -> usize {
                t.len() / 4 + 1
            }
            fn start(&mut self, _: crate::ReplyRequest) -> Result<()> {
                anyhow::bail!("no engine")
            }
            fn poll(&mut self) -> Vec<(u64, BackendEvent)> {
                Vec::new()
            }
            fn reset(&mut self) {}
        }
        let mut a = App::new(Room::new(RoomConfig::default()).unwrap(), Broken).unwrap();
        "hi".chars().for_each(|c| a.handle(Action::Insert(c)));
        a.handle(Action::Submit);
        let last = a.room.messages().last().unwrap();
        assert_eq!(last.text, "Generation failed: no engine");
        assert_eq!(a.room.activity(), crate::Activity::Idle);
    }
}
