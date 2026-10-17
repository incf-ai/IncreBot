//! The room on the real inference thread with TinyLlama-1.1B Q8_0: replies
//! stream, the channel cache is reused between turns, `/ask` doesn't evict it,
//! `/reset` drops it, and Esc stops a reply.
//!
//! Skipped when the model is not in the HF cache, and in debug builds (the
//! kernels are unusably slow without optimisation). Run with:
//! `cargo test --release -p chatsim-core --test engine_tinyllama`

// Test-only diagnostics (skip reasons, intermediate replies).
#![allow(clippy::print_stderr)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use chatsim_core::{Action, Activity, App, EngineBackend, Kind, Room, RoomConfig};
use increbot::gguf::Gguf;
use increbot::hub;
use increbot::model::{LinearImpl, Llama};
use increbot::tokenizer::Tokenizer;

fn backend(max_new_tokens: usize) -> Option<EngineBackend> {
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return None;
    }
    let path = hf_hub::Cache::from_env()
        .model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into())
        .get("tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
    let Some(path) = path else {
        eprintln!("SKIPPED: TinyLlama GGUF not in the HF cache");
        return None;
    };
    let g = Gguf::open(&path).unwrap();
    let tok = Arc::new(
        Tokenizer::load(
            &hub::resolve("hf-internal-testing/llama-tokenizer", "tokenizer.json").unwrap(),
            &g,
        )
        .unwrap(),
    );
    let model = Llama::load(&g, &Default::default(), LinearImpl::Packed).unwrap();
    // Greedy and unpenalised, so replies are repeatable.
    let mut persona = chatsim_core::room::default_persona();
    persona.sampler.temperature = 0.0;
    persona.sampler.repeat = None;
    persona.sampler.dry = None;
    persona.discourage.clear();
    persona.max_new_tokens = increbot::persona::Tokens::both(max_new_tokens);
    persona.wrap_up_after = increbot::persona::Tokens::default();
    Some(
        EngineBackend::new(
            increbot::engine::Backends::cpu_only(model),
            tok,
            persona,
            Duration::from_secs(140),
            2,
            "tinyllama".into(),
        )
        .unwrap(),
    )
}

fn send(app: &mut App<EngineBackend>, line: &str) {
    line.chars().for_each(|c| app.handle(Action::Insert(c)));
    app.handle(Action::Submit);
}

fn settle(app: &mut App<EngineBackend>) {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        app.pump();
        if app.room.activity() == Activity::Idle && app.room.queued() == 0 {
            return;
        }
        assert!(Instant::now() < deadline, "reply took over 2 minutes");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Prompt tokens served from the cache by the most recent job.
fn reused_by_last_job(app: &App<EngineBackend>, before: u64) -> u64 {
    app.room.stats().reused_tokens - before
}

#[test]
fn chat_turns_on_tinyllama() {
    let Some(backend) = backend(16) else { return };
    let mut a = App::new(Room::new(RoomConfig::default()).unwrap(), backend).unwrap();

    send(&mut a, "Hello! What is the capital of France?");
    settle(&mut a);
    let reply = a
        .room
        .messages()
        .iter()
        .rev()
        .find(|m| m.kind == Kind::Bot)
        .expect("a reply");
    eprintln!("turn 1: {:?}", reply.text);
    assert!(!reply.streaming && !reply.text.is_empty());
    assert!(a.room.in_history(reply));
    let s = a.room.stats().clone();
    assert_eq!((s.jobs, s.reused_tokens), (1, 0));
    assert!(s.last_decode_tok_s.is_some());

    // Second turn extends the first prompt: the channel cache is reused.
    send(&mut a, "And of Germany?");
    settle(&mut a);
    let reused = reused_by_last_job(&a, s.reused_tokens);
    eprintln!("turn 2 reused {reused} tokens");
    assert!(reused > 50, "expected a warm prefill, reused {reused}");

    // `/ask` runs in a scratch cache, so the next channel turn is still warm.
    send(&mut a, "/ask Say hi.");
    settle(&mut a);
    let before = a.room.stats().reused_tokens;
    send(&mut a, "Thanks!");
    settle(&mut a);
    assert!(
        reused_by_last_job(&a, before) > 50,
        "/ask evicted the channel cache"
    );

    // `/reset` drops the cache and the history: cold again.
    send(&mut a, "/reset");
    let before = a.room.stats().reused_tokens;
    send(&mut a, "Start over.");
    settle(&mut a);
    assert_eq!(reused_by_last_job(&a, before), 0);
    assert_eq!(a.room.stats().jobs, 5);
}

#[test]
fn esc_stops_a_long_reply() {
    let Some(backend) = backend(200) else { return };
    let mut a = App::new(Room::new(RoomConfig::default()).unwrap(), backend).unwrap();
    send(&mut a, "Write a long story about a dragon.");
    let deadline = Instant::now() + Duration::from_secs(60);
    while !matches!(a.room.activity(), Activity::Writing { .. }) {
        a.pump();
        assert!(Instant::now() < deadline, "no text within a minute");
        std::thread::sleep(Duration::from_millis(5));
    }
    a.handle(Action::Cancel);
    settle(&mut a);
    let m = a
        .room
        .messages()
        .iter()
        .rev()
        .find(|m| m.kind == Kind::Bot)
        .unwrap();
    eprintln!("stopped reply: {:?}", m.text);
    assert_eq!(m.note, Some("stopped"), "{m:?}");
    assert!(
        m.text.split_whitespace().count() < 100,
        "cancel came too late: {m:?}"
    );
}
