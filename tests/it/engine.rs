//! Inference-thread behaviour (design.md §5) on TinyLlama: cache LRU, scratch slot,
//! reset, and cancellation. Skipped when the model is not in the HF cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use increbot::engine::bans::Bans;
use increbot::engine::sampler::SamplerParams;
use increbot::engine::worker::{CacheKey, EngineHandle, Event, Job, spawn};
use increbot::engine::{GenerateOutput, StopReason};
use increbot::gguf::Gguf;
use increbot::hub;
use increbot::model::{LinearImpl, Llama};
use increbot::tokenizer::Tokenizer;
use tokio::sync::mpsc;

fn engine() -> Option<(EngineHandle, Arc<Tokenizer>)> {
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
    let mut model = Llama::load(&g, &Default::default(), LinearImpl::Packed).unwrap();
    model.set_kv_f16(true); // the production setting
    Some((
        spawn(increbot::engine::Backends::cpu_only(model), tok.clone(), 2).unwrap(),
        tok,
    ))
}

struct Outcome {
    started: bool,
    text: String,
    out: GenerateOutput,
}

async fn run(
    h: &mut EngineHandle,
    tok: &Tokenizer,
    key: CacheKey,
    prompt: &str,
    max_new: usize,
    cancel: Arc<AtomicBool>,
    cancel_after_text: bool,
) -> Outcome {
    run_with(h, tok, key, prompt, max_new, cancel, cancel_after_text, None).await
}

#[allow(clippy::too_many_arguments)]
async fn run_with(
    h: &mut EngineHandle,
    tok: &Tokenizer,
    key: CacheKey,
    prompt: &str,
    max_new: usize,
    cancel: Arc<AtomicBool>,
    cancel_after_text: bool,
    bans: Option<Arc<Bans>>,
) -> Outcome {
    run_sampled(h, tok, key, prompt, max_new, cancel, cancel_after_text, bans, SamplerParams::greedy()).await
}

/// [`run_with`] under `sampler` (seed 1) instead of greedy.
#[allow(clippy::too_many_arguments)]
async fn run_sampled(
    h: &mut EngineHandle,
    tok: &Tokenizer,
    key: CacheKey,
    prompt: &str,
    max_new: usize,
    cancel: Arc<AtomicBool>,
    cancel_after_text: bool,
    bans: Option<Arc<Bans>>,
    sampler: SamplerParams,
) -> Outcome {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let job = Job {
        key,
        prompt: tok.encode_prompt(prompt).unwrap(),
        sampler,
        seed: 1,
        history: vec![],
        max_new_tokens: max_new,
        max_new_tokens_gpu: max_new,
                wrap_up_after: 0,
                wrap_up_after_gpu: 0,
stops: vec![],
        max_lines: None,
        labels: false,
        timeout: Duration::from_secs(60),
        cancel: cancel.clone(),
        events: tx,
        best_of: std::num::NonZeroUsize::MIN,
        score: None,
        bans,
    };
    h.jobs.send(job).await.unwrap();
    let (mut started, mut text) = (false, String::new());
    let out = loop {
        match rx.recv().await.expect("engine dropped the job") {
            Event::Started => started = true,
            Event::Text(t) => {
                text += &t;
                if cancel_after_text {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Event::Finished(out) => break out,
            Event::Failed(e) => panic!("job failed: {e}"),
            Event::Paused(_) => {}
        }
    };
    h.idle.recv().await.unwrap();
    Outcome { started, text, out }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_lru_scratch_reset_and_cancel() {
    let Some((mut h, tok)) = engine() else { return };
    let no = || Arc::new(AtomicBool::new(false));
    let a = CacheKey::Channel(1);
    let base = "Alice: Hi! What's a good name for a cat?\nBob: Maybe Whiskers.\nAlice:";

    // Fresh channel: nothing reused.
    let r = run(&mut h, &tok, a, base, 8, no(), false).await;
    assert!(r.started && !r.text.is_empty());
    assert_eq!(r.out.reused_tokens, 0);
    let base_len = tok.encode_prompt(base).unwrap().len();

    // A scratch job (/ask) must not evict channel A's active cache.
    let r = run(
        &mut h,
        &tok,
        CacheKey::Scratch,
        "The capital of France is",
        4,
        no(),
        false,
    )
    .await;
    assert_eq!(r.out.reused_tokens, 0);
    let longer = format!("{base} I like that.\nBob:");
    let r = run(&mut h, &tok, a, &longer, 4, no(), false).await;
    assert!(
        r.out.reused_tokens >= base_len - 2,
        "reused {} of {base_len}",
        r.out.reused_tokens
    );

    // Switch to channel B, then back: A is restored from its f16-parked copy.
    run(
        &mut h,
        &tok,
        CacheKey::Channel(2),
        "Once upon a time",
        4,
        no(),
        false,
    )
    .await;
    let r = run(&mut h, &tok, a, &longer, 4, no(), false).await;
    assert!(
        r.out.reused_tokens >= base_len,
        "reused {} after park/restore",
        r.out.reused_tokens
    );

    // /reset drops A's cache.
    h.reset(1);
    let r = run(&mut h, &tok, a, &longer, 2, no(), false).await;
    assert_eq!(r.out.reused_tokens, 0);

    // Cancelled while queued: never starts.
    let cancelled = Arc::new(AtomicBool::new(true));
    let r = run(&mut h, &tok, a, &longer, 8, cancelled, false).await;
    assert!(!r.started);
    assert_eq!(r.out.stop, StopReason::Cancelled);
    assert!(r.out.tokens.is_empty());

    // Cancelled mid-generation: stops early with what it has.
    let r = run(&mut h, &tok, a, base, 64, no(), true).await;
    assert_eq!(r.out.stop, StopReason::Cancelled);
    assert!(r.out.tokens.len() < 64, "generated {}", r.out.tokens.len());
}

/// Phrase bans (`engine::bans`): a greedy reply, then the same prompt with two
/// of its words banned rolls back, avoids them, and keeps what came before.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn phrase_ban_rolls_back_and_resamples() {
    let Some((mut h, tok)) = engine() else { return };
    let no = || Arc::new(AtomicBool::new(false));
    let prompt = "Alice: Tell me a short story about a cat.\nBob:";
    let free = run(&mut h, &tok, CacheKey::Scratch, prompt, 40, no(), false).await;
    // Two neighbouring plain words, a few words in.
    let ws: Vec<&str> = free.text.split_whitespace().collect();
    let plain = |w: &str| w.len() >= 3 && w.chars().all(|c| c.is_ascii_alphabetic());
    let i = (3..ws.len().saturating_sub(1))
        .find(|&i| plain(ws[i]) && plain(ws[i + 1]))
        .unwrap_or_else(|| panic!("no two plain words to ban in {:?}", free.text));
    let phrase = format!("{} {}", ws[i], ws[i + 1]).to_lowercase();
    let bans = Arc::new(Bans::parse(&[phrase.clone()]).unwrap());
    // Only if the two words are adjacent in the free reply.
    let Some((start, _)) = bans.find(&free.text, 0) else {
        eprintln!("SKIPPED: {phrase:?} not adjacent in {:?}", free.text);
        return;
    };
    let banned = run_with(&mut h, &tok, CacheKey::Scratch, prompt, 40, no(), false, Some(bans.clone())).await;
    assert!(!banned.out.rollbacks.is_empty(), "no rollback: {:?}", banned.text);
    assert!(
        banned.out.rollbacks.iter().all(|r| !r.starts_with("kept:")),
        "{:?}",
        banned.out.rollbacks
    );
    assert_eq!(bans.find(&banned.text, 0), None, "{phrase:?} still in {:?}", banned.text);
    // Greedy: the same text up to the banned phrase.
    let keep = free.text[..start].trim_end();
    assert!(banned.text.starts_with(keep), "{:?} vs {:?}", banned.text, free.text);
    eprintln!("banned {phrase:?}: {:?} -> {:?} ({:?})", free.text, banned.text, banned.out.rollbacks);
}

/// `Bans::first_chars`: a prompt whose replies open with "!" gets a reply that
/// doesn't, without a rollback, and the mask is off once something visible is out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn banned_first_chars_are_never_sampled_first() {
    let Some((mut h, tok)) = engine() else { return };
    let no = || Arc::new(AtomicBool::new(false));
    let prompt = "Alice: hi\nBob: !!! hi !!!\nAlice: yo\nBob: !!! yo !!!\nAlice: hey\nBob:";
    let free = run(&mut h, &tok, CacheKey::Scratch, prompt, 12, no(), false).await;
    let bans = Arc::new(Bans::default().with_first_chars(Arc::new(tok.ids_starting_with("!"))));
    let masked = run_with(&mut h, &tok, CacheKey::Scratch, prompt, 12, no(), false, Some(bans)).await;
    assert!(!masked.text.trim_start().starts_with('!'), "{:?}", masked.text);
    assert!(masked.out.rollbacks.is_empty(), "{:?}", masked.out.rollbacks);
    eprintln!("first chars: {:?} -> {:?}", free.text, masked.text);
}

/// `lowercase_start` (`Bans::capitalized_start` plus its token mask): no
/// Capitalized first word, while all-caps shouting comes out as it would.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capitalized_starts_are_lowered_but_shouting_is_kept() {
    let Some((mut h, tok)) = engine() else { return };
    let no = || Arc::new(AtomicBool::new(false));
    let cap = |t: &str| {
        let mut c = t.chars();
        c.next().is_some_and(char::is_uppercase) && c.next().is_some_and(char::is_lowercase)
    };
    let mut ids = tok.ids_starting_where(|t| t.starts_with('"') || cap(t));
    ids.extend(tok.newline_ids());
    let bans = Arc::new(Bans::default().with_first_chars(Arc::new(ids)).with_capitalized_start(true));
    let check = Bans::default().with_capitalized_start(true);
    for prompt in ["Alice: what's the capital of France?\nBob:", "Alice: Are you OK?\nBob:"] {
        let r = run_with(&mut h, &tok, CacheKey::Scratch, prompt, 12, no(), false, Some(bans.clone())).await;
        assert!(r.text.trim_start().chars().next().is_some_and(|c| !c.is_uppercase() || r.text.trim_start().starts_with("OK")), "{:?}", r.text);
        assert_eq!(check.find(&(r.text.clone() + " "), 0), None, "{:?}", r.text);
    }
    let scream = "Alice: SCREAM\nBob: AAAAA\nAlice: LOUDER\nBob: AAAAAAAAA\nAlice: MORE\nBob:";
    let free = run(&mut h, &tok, CacheKey::Scratch, scream, 8, no(), false).await;
    let masked = run_with(&mut h, &tok, CacheKey::Scratch, scream, 8, no(), false, Some(bans)).await;
    assert_eq!(masked.text, free.text);
}

/// `sampler.word_start` at chance 1: every word start is drawn again (under
/// greedy-free settings), the trace records each, and every one of them is a
/// token that starts a word.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn word_starts_are_redrawn_and_traced() {
    use increbot::engine::Steered;
    use increbot::engine::sampler::{TokenShape, WordStartParams};
    let Some((mut h, tok)) = engine() else { return };
    let no = || Arc::new(AtomicBool::new(false));
    let word_start = WordStartParams {
        chance: 1.0,
        temperature: 1.15,
        top_k: 100,
        top_p: 0.99,
        min_p: 0.1,
        shapes: tok.pieces().iter().map(|p| TokenShape::of(p)).collect(),
    };
    let sampler = SamplerParams { temperature: 1.05, top_k: 100, top_p: 0.99, min_p: 0.1, word_start: Some(word_start), ..SamplerParams::default() };
    let prompt = "Alice: Tell me a short story about a cat.\nBob:";
    // "the" banned: rollbacks discard some redrawn word starts.
    let bans = Some(Arc::new(Bans::parse(&["the".to_string()]).unwrap()));
    let r = run_sampled(&mut h, &tok, CacheKey::Scratch, prompt, 30, no(), false, bans, sampler).await;
    let mut n = 0;
    for s in &r.out.steered {
        let Steered::WordStart { at, picked, word, odds, normal, changed, .. } = s else { continue };
        n += 1;
        // Only redraws the reply kept: its token there is the picked one.
        assert_eq!(&tok.decode(&r.out.tokens[*at..*at + 1]).unwrap(), picked, "token {at} in {:?}", r.text);
        assert!(word.starts_with(picked.trim_start()) && word.starts_with(char::is_alphanumeric), "{word:?} / {picked:?}");
        assert!(odds.iter().any(|o| &o.0 == picked) && odds.iter().any(|o| &o.0 == normal));
        assert!(odds.iter().all(|o| (0.0..=1.0).contains(&o.1) && (0.0..=1.0).contains(&o.2)));
        assert_eq!(*changed, picked != normal);
        eprintln!("  token {at}: {word:?} (was {normal:?}): {odds:?}");
    }
    assert!(n >= 3, "{n} redrawn in {:?}", r.text);
    assert!(!r.out.rollbacks.is_empty() || !r.text.to_lowercase().contains(" the "));
    eprintln!("word starts: {:?} ({n} redrawn, rollbacks {:?})", r.text, r.out.rollbacks);
}

/// The seed prefill (`SeedHandle`): with no GPUs it runs at once on the CPU,
/// and a one-shot reply whose prompt starts with the header reuses it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seed_prefill_is_reused_by_one_shot_replies() {
    let Some((mut h, tok)) = engine() else { return };
    let no = || Arc::new(AtomicBool::new(false));
    let header = "### Instruction:\nYou are Bob, a friendly cat expert. Answer briefly.\n";
    let prompt = format!("{header}Alice: What's a good name for a cat?\nBob:");
    // Before any seed: nothing to reuse.
    let cold = run(&mut h, &tok, CacheKey::Scratch, &prompt, 4, no(), false).await;
    assert_eq!(cold.out.reused_tokens, 0);
    h.seed.request(tok.encode_prompt(header).unwrap());
    let started = std::time::Instant::now();
    while h.stats.lock().unwrap().seed.is_empty() || h.stats.lock().unwrap().seed.starts_with("waiting") {
        assert!(started.elapsed() < Duration::from_secs(60), "seed never prefilled");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let line = h.stats.lock().unwrap().seed.clone();
    assert!(line.contains("on cpu"), "{line}");
    let warm = run(&mut h, &tok, CacheKey::Scratch, &prompt, 4, no(), false).await;
    let header_len = tok.encode_prompt(header).unwrap().len();
    assert!(
        warm.out.reused_tokens + 2 >= header_len,
        "reused {} of a {header_len}-token header ({line})",
        warm.out.reused_tokens
    );
    // Same reply either way (greedy).
    assert_eq!(warm.text, cold.text);
}
