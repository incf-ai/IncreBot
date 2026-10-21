//! Whole conversations through the public API with the scripted backend: what
//! the bot is asked (the prompts), what the room shows, and how the cache is used.

use chatsim_core::{Action, App, Kind, Pacing, ReplyMode, Room, RoomConfig, ScriptedBackend};

fn app(cfg: RoomConfig, pacing: Pacing) -> App<ScriptedBackend> {
    let backend = ScriptedBackend::new(pacing, |r| format!(" reply to {}", r.trigger_text));
    App::new(Room::new(cfg).unwrap(), backend).unwrap()
}

fn send(app: &mut App<ScriptedBackend>, line: &str) {
    line.chars().for_each(|c| app.handle(Action::Insert(c)));
    app.handle(Action::Submit);
}

fn settle(app: &mut App<ScriptedBackend>) {
    for _ in 0..10_000 {
        if !app.pump() {
            return;
        }
    }
    panic!("backend never went idle");
}

fn bot_lines(app: &App<ScriptedBackend>) -> Vec<String> {
    app.room
        .messages()
        .iter()
        .filter(|m| m.kind == Kind::Bot)
        .map(|m| m.text.clone())
        .collect()
}

#[test]
fn several_users_take_turns() {
    let mut a = app(RoomConfig::default(), Pacing::Instant);
    send(&mut a, "hi everyone");
    settle(&mut a);
    send(&mut a, "/nick Bob");
    send(&mut a, "hey Alice");
    settle(&mut a);
    send(&mut a, "/as Carol lurking");
    settle(&mut a);

    assert_eq!(
        bot_lines(&a),
        [
            "reply to hi everyone",
            "reply to hey Alice",
            "reply to lurking"
        ]
    );
    let reqs = a.backend.handle().requests();
    let last = &reqs[2].prompt;
    assert!(
        last.contains(
            "Alice: hi everyone\nfloob: reply to hi everyone\n\
             Bob: hey Alice\nfloob: reply to hey Alice\nCarol: lurking\n"
        ),
        "{last}"
    );
    assert!(
        last.ends_with("(floob is replying to Carol)\nfloob:"),
        "{last}"
    );
    for name in ["\nAlice:", "\nBob:", "\nCarol:", "\nfloob:"] {
        assert!(
            reqs[2].stops.iter().any(|s| s == name),
            "missing stop {name}"
        );
    }
    let replies: Vec<_> = a
        .room
        .messages()
        .iter()
        .filter(|m| m.kind == Kind::Bot)
        .collect();
    assert_eq!(replies[1].reply_to.as_deref(), Some("Bob"));
}

#[test]
fn long_chat_keeps_prompts_in_budget_and_mostly_warm() {
    let cfg = RoomConfig {
        history_messages: 8,
        // The persona and examples (~685 words with c55) come before the
        // history and the reminder after it; ~260 words are left for history.
        max_prompt_tokens: 900,
        ..Default::default()
    };
    let mut a = app(cfg, Pacing::Instant);
    let turns = 40;
    for i in 0..turns {
        send(
            &mut a,
            &format!("message number {i} with a few extra words"),
        );
        settle(&mut a);
    }
    let reqs = a.backend.handle().requests();
    assert_eq!(reqs.len(), turns);
    let count = |s: &str| 1 + s.split_whitespace().count();
    for r in &reqs {
        assert!(
            count(&r.prompt) <= 900,
            "prompt over budget: {}",
            count(&r.prompt)
        );
    }
    // Each prompt extends the previous one except where the anchor jumped.
    // Up to the persona's reminder after the history.
    let reminder = RoomConfig::default().persona.reminder;
    let header_end = |p: &str| p.rfind(&format!("\n{reminder}")).unwrap();
    let warm = reqs
        .windows(2)
        .filter(|w| {
            w[1].prompt
                .starts_with(&w[0].prompt[..header_end(&w[0].prompt)])
        })
        .count();
    assert!(
        warm * 2 > turns,
        "only {warm} of {turns} prompts extended the previous one"
    );
    let s = a.room.stats();
    // Every turn re-reads the reminder after the history and some history
    // after an anchor jump, so well under all of it is reused.
    assert!(s.reused_tokens * 5 > s.prompt_tokens * 2, "{s:?}");
    // The oldest messages are out of the bot's history but still on screen.
    let first = a
        .room
        .messages()
        .iter()
        .find(|m| m.text.starts_with("message number 0"))
        .unwrap();
    assert!(!a.room.in_history(first));
    let newest = a
        .room
        .messages()
        .iter()
        .rev()
        .find(|m| m.kind == Kind::Bot)
        .unwrap();
    assert!(a.room.in_history(newest));
}

#[test]
fn busy_channel_queues_four_and_ignores_the_rest() {
    let mut a = app(RoomConfig::default(), Pacing::PerPoll(1));
    send(&mut a, "first question that takes a while to answer");
    for i in 0..5 {
        send(&mut a, &format!("follow-up {i}"));
    }
    settle(&mut a);
    // One running + four waiting; the fifth follow-up found the queue full.
    assert_eq!(
        bot_lines(&a),
        [
            "reply to first question that takes a while to answer",
            "reply to follow-up 0",
            "reply to follow-up 1",
            "reply to follow-up 2",
            "reply to follow-up 3",
        ]
    );
    // Each queued reply sees the history frozen at its trigger, so follow-up 3's
    // prompt ends with it; the ignored follow-up 4 came later.
    let reqs = a.backend.handle().requests();
    assert!(
        reqs[4].prompt.contains(&format!(
            "Alice: follow-up 3\n\n{}",
            RoomConfig::default().persona.reminder
        )),
        "{}",
        reqs[4].prompt
    );
    assert!(!reqs[4].prompt.contains("follow-up 4"));
    // The ignored message is still history: the next reply sees it.
    send(&mut a, "anyone there?");
    settle(&mut a);
    let next = &a.backend.handle().requests()[5].prompt;
    assert!(next.contains("Alice: follow-up 4\n"), "{next}");
}

#[test]
fn mention_mode_ask_and_reset() {
    let cfg = RoomConfig {
        mode: ReplyMode::Mention,
        ..Default::default()
    };
    let mut a = app(cfg, Pacing::Instant);
    send(&mut a, "talking among ourselves");
    settle(&mut a);
    assert!(bot_lines(&a).is_empty());

    send(&mut a, "/ask quick one?");
    settle(&mut a);
    let ask = &a.backend.handle().requests()[0];
    assert!(ask.scratch && !ask.prompt.contains("ourselves"));

    send(&mut a, "@IncreBot what were we saying?");
    settle(&mut a);
    let chan = &a.backend.handle().requests()[1];
    assert!(!chan.scratch);
    assert!(chan.prompt.contains("talking among ourselves"));
    assert!(!chan.prompt.contains("quick one"), "{}", chan.prompt);

    send(&mut a, "/reset");
    assert_eq!(a.backend.handle().resets(), 1);
    send(&mut a, "@IncreBot hello again");
    settle(&mut a);
    let after = &a.backend.handle().requests()[2];
    assert!(!after.prompt.contains("ourselves"), "{}", after.prompt);
    assert_eq!(a.room.stats().jobs, 3);
}

#[test]
fn timed_mock_streams_like_the_tui_uses_it() {
    let mut a = app(
        RoomConfig::default(),
        Pacing::Every(std::time::Duration::from_millis(1)),
    );
    send(&mut a, "stream please");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while a.room.activity() != chatsim_core::Activity::Idle {
        a.pump();
        assert!(std::time::Instant::now() < deadline, "reply never finished");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(bot_lines(&a), ["reply to stream please"]);
}
