//! The event loop end to end: scripted key presses go through the key map, the
//! app and the scripted backend, and the result is read off a test terminal.
//! Also checks the `chatsim` binary's command line.

use std::process::Command;

use chatsim_core::{App, Kind, Pacing, Room, RoomConfig, ScriptedBackend};
use chatsim_tui::{ScriptedEvents, run};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{Event, KeyCode, KeyModifiers};

const W: u16 = 70;
const H: u16 = 20;

fn app(pacing: Pacing) -> App<ScriptedBackend> {
    let b = ScriptedBackend::new(pacing, |r| {
        format!(" nice to meet you, {}", r.trigger_author)
    });
    App::new(Room::new(RoomConfig::default()).unwrap(), b).unwrap()
}

fn screen(t: &Terminal<TestBackend>) -> Vec<String> {
    let buf = t.backend().buffer();
    (0..buf.area.height)
        .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect())
        .collect()
}

fn session(app: &mut App<ScriptedBackend>, events: ScriptedEvents) -> Vec<String> {
    let mut t = Terminal::new(TestBackend::new(W, H)).unwrap();
    let mut events = events;
    run(&mut t, app, &mut events).unwrap();
    assert_eq!(events.remaining(), 0, "quit before the script ended");
    screen(&t)
}

fn find(rows: &[String], needle: &str) -> Option<usize> {
    rows.iter().position(|r| r.contains(needle))
}

#[test]
fn typing_a_message_gets_a_streamed_reply() {
    let mut a = app(Pacing::PerPoll(1));
    let rows = session(&mut a, ScriptedEvents::new().line("hello room").idle(10));
    let me = find(&rows, "Alice: hello room").expect("my message");
    let bot = find(&rows, "floob → Alice: nice to meet you, Alice").expect("the reply");
    assert!(bot > me);
    assert!(rows[H as usize - 4].contains("● ready"), "{rows:#?}");
}

#[test]
fn ctrl_c_quits_and_cancels_a_reply_in_progress() {
    let mut a = app(Pacing::PerPoll(1));
    let h = a.backend.handle();
    session(
        &mut a,
        ScriptedEvents::new()
            .line("hi")
            .idle(1)
            .key(KeyCode::Char('c'), KeyModifiers::CONTROL),
    );
    assert!(a.should_quit());
    let req = &h.requests()[0];
    assert!(req.cancel.load(std::sync::atomic::Ordering::Relaxed));
}

#[test]
fn esc_stops_the_reply() {
    let mut a = app(Pacing::PerPoll(1));
    let rows = session(
        &mut a,
        ScriptedEvents::new()
            .line("hi")
            .idle(2)
            .key(KeyCode::Esc, KeyModifiers::NONE)
            .idle(3),
    );
    assert!(find(&rows, "(stopped)").is_some(), "{rows:#?}");
    assert!(find(&rows, "nice to meet you, Alice").is_none());
}

#[test]
fn nick_mode_and_commands_from_the_keyboard() {
    let mut a = app(Pacing::Instant);
    let rows = session(
        &mut a,
        ScriptedEvents::new()
            .line("/mode mention")
            .line("/nick Bob")
            .line("just us humans")
            .line("@IncreBot hey")
            .idle(3)
            .line("/stats")
            .idle(1),
    );
    assert!(find(&rows, "Bob: just us humans").is_some(), "{rows:#?}");
    assert!(
        find(&rows, "floob → Bob: nice to meet you, Bob").is_some(),
        "{rows:#?}"
    );
    assert!(find(&rows, "replies: 1").is_some(), "{rows:#?}");
    assert!(rows[0].contains("mode: mention"));
    assert!(
        find(&rows, "┌ Bob ").is_some(),
        "input box is titled with the nick"
    );
    let bots = a
        .room
        .messages()
        .iter()
        .filter(|m| m.kind == Kind::Bot)
        .count();
    assert_eq!(bots, 1);
}

#[test]
fn input_editing_and_history_recall() {
    let mut a = app(Pacing::Instant);
    session(
        &mut a,
        ScriptedEvents::new()
            .line("/mode mention")
            .text("helo")
            .key(KeyCode::Left, KeyModifiers::NONE)
            .text("l")
            .key(KeyCode::Enter, KeyModifiers::NONE)
            .key(KeyCode::Up, KeyModifiers::NONE)
            .text(" again")
            .key(KeyCode::Enter, KeyModifiers::NONE)
            .text("draft")
            .key(KeyCode::Char('u'), KeyModifiers::CONTROL),
    );
    let mine: Vec<_> = a
        .room
        .messages()
        .iter()
        .filter(|m| m.kind == Kind::User)
        .map(|m| m.text.as_str())
        .collect();
    assert_eq!(mine, ["hello", "hello again"]);
    assert!(a.input.is_empty());
}

#[test]
fn scrolling_back_and_returning() {
    let mut a = app(Pacing::Instant);
    let mut script = ScriptedEvents::new().line("/mode mention");
    for i in 0..40 {
        script = script.line(&format!("msg {i}"));
    }
    let scrolled = script
        .key(KeyCode::PageUp, KeyModifiers::NONE)
        .key(KeyCode::PageUp, KeyModifiers::NONE);
    let rows = session(&mut a, scrolled);
    assert!(find(&rows, "msg 39").is_none(), "{rows:#?}");
    assert!(find(&rows, "scrolled 20 lines").is_some());

    let mut b = app(Pacing::Instant);
    let mut script = ScriptedEvents::new().line("/mode mention");
    for i in 0..40 {
        script = script.line(&format!("msg {i}"));
    }
    let back = script
        .key(KeyCode::Home, KeyModifiers::CONTROL)
        .key(KeyCode::End, KeyModifiers::CONTROL);
    let rows = session(&mut b, back);
    assert!(find(&rows, "msg 39").is_some());
    assert!(find(&rows, "scrolled").is_none());
}

#[test]
fn paste_and_resize_events() {
    let mut a = app(Pacing::Instant);
    let rows = session(
        &mut a,
        ScriptedEvents::new()
            .line("/mode mention")
            .event(Event::Paste("pasted\ntext".into()))
            .key(KeyCode::Enter, KeyModifiers::NONE)
            .event(Event::Resize(40, 10)),
    );
    assert!(find(&rows, "Alice: pasted text").is_some(), "{rows:#?}");
}

#[test]
fn binary_help_and_bad_arguments() {
    let bin = env!("CARGO_BIN_EXE_chatsim");
    let out = Command::new(bin).arg("--help").output().unwrap();
    assert!(out.status.success());
    let help = String::from_utf8_lossy(&out.stdout);
    for flag in ["--mock", "--nick", "--mode", "--config", "--log"] {
        assert!(help.contains(flag), "--help lacks {flag}:\n{help}");
    }
    let out = Command::new(bin)
        .args(["--mock", "--mode", "loud"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown mode"));
}

#[test]
fn binary_rejects_an_invalid_config() {
    let dir = std::env::temp_dir().join(format!("chatsim-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("bad.rsn");
    std::fs::write(&cfg, "Config { generation: { max_new_tokens: \"lots\" } }").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_chatsim"))
        .args(["--mock", "--config"])
        .arg(&cfg)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("invalid configuration"));
}
