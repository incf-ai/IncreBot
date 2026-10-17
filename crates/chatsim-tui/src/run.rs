//! The event loop: draw, wait briefly for input, apply it, pump the backend.

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use anyhow::Result;
use chatsim_core::{Action, App, Backend};
use ratatui::Terminal;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::{keys, ui};

/// How long to wait for input before redrawing (streaming text, spinner).
pub const TICK: Duration = Duration::from_millis(50);

/// Where terminal events come from: the real terminal, or a script in tests.
pub trait EventSource {
    /// The next event, or `None` if none arrived within `timeout`.
    fn next(&mut self, timeout: Duration) -> io::Result<Option<Event>>;
}

pub struct CrosstermEvents;

impl EventSource for CrosstermEvents {
    fn next(&mut self, timeout: Duration) -> io::Result<Option<Event>> {
        if event::poll(timeout)? {
            event::read().map(Some)
        } else {
            Ok(None)
        }
    }
}

/// A fixed sequence of events and idle ticks. When it runs out it sends
/// Ctrl-C, so a test loop always ends.
#[derive(Default)]
pub struct ScriptedEvents {
    queue: VecDeque<Option<Event>>,
}

impl ScriptedEvents {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn key(mut self, code: KeyCode, modifiers: KeyModifiers) -> Self {
        self.queue
            .push_back(Some(Event::Key(KeyEvent::new(code, modifiers))));
        self
    }

    /// Types `text` character by character.
    pub fn text(self, text: &str) -> Self {
        text.chars()
            .fold(self, |s, c| s.key(KeyCode::Char(c), KeyModifiers::NONE))
    }

    /// Types `text` and presses Enter.
    pub fn line(self, text: &str) -> Self {
        self.text(text).key(KeyCode::Enter, KeyModifiers::NONE)
    }

    /// `n` polls with no input (the backend keeps streaming meanwhile).
    pub fn idle(mut self, n: usize) -> Self {
        self.queue.extend(std::iter::repeat_n(None, n));
        self
    }

    pub fn event(mut self, e: Event) -> Self {
        self.queue.push_back(Some(e));
        self
    }

    pub fn remaining(&self) -> usize {
        self.queue.len()
    }
}

impl EventSource for ScriptedEvents {
    fn next(&mut self, _timeout: Duration) -> io::Result<Option<Event>> {
        Ok(self.queue.pop_front().unwrap_or_else(|| {
            Some(Event::Key(KeyEvent::new(
                KeyCode::Char('c'),
                KeyModifiers::CONTROL,
            )))
        }))
    }
}

/// Runs until the user quits. Draws after every event or tick.
pub fn run<T, B>(
    terminal: &mut Terminal<T>,
    app: &mut App<B>,
    events: &mut impl EventSource,
) -> Result<()>
where
    T: ratatui::backend::Backend,
    T::Error: Send + Sync + 'static,
    B: Backend,
{
    loop {
        let mut max_scroll = 0;
        terminal.draw(|f| max_scroll = ui::draw(f, app))?;
        app.clamp_scroll(max_scroll);
        if app.should_quit() {
            return Ok(());
        }
        match events.next(TICK)? {
            Some(Event::Key(k)) => {
                if let Some(a) = keys::action(k) {
                    app.handle(a);
                }
            }
            Some(Event::Paste(text)) => text.chars().for_each(|c| app.handle(Action::Insert(c))),
            // Resizes just need the redraw at the top of the loop.
            _ => {}
        }
        app.pump();
    }
}
