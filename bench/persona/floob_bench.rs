//! Persona benchmark: batches of a user saying "floob" over and over, each
//! reply through the bot's own path (config, persona file, build_prompt,
//! the worker and its sampler) on one GPU, logged as JSONL for analysis.
//!
//! `cargo run --release --example floob_bench -- --gpu <pci> --out <dir>
//!  [--batches 20] [--turns 25] [--seed N] [--config config.local.rsn] [--evict]`
//!
//! - Each batch starts with an empty channel and its own KV cache; turn `t` of
//!   batch `b` uses seed `seed + b * 1000 + t`, so a rerun with the same `--seed`
//!   draws the same random numbers (replies still diverge once a logit changes).
//! - `INCREBOT_*` env vars override the config as in the bot
//!   (`INCREBOT_PERSONA_FILE` picks the persona; `INCREBOT_DISCOURAGE` replaces
//!   its word list, and `INCREBOT_DISCOURAGE=` turns it off); `--gpu` sets `gpus`.
//! - `--evict`: the live service has its standby weights on this card. 2 GiB of
//!   "another program" makes it let go, and is held for the whole run.
//! - Writes `<dir>/meta.json` and `<dir>/batch_NN.jsonl` (one line per turn).
//! - `--scenarios <file.jsonl> [--repeats 4]`: scripted chats instead of
//!   "floob" (README §1). Each line is `{"id", "category", "expect", "turns":
//!   [[{"user", "text"}, …], …]}`: the messages posted before each reply, under
//!   their own names. Each scenario runs `--repeats` times, one batch each, in
//!   file order; `--batches` and `--turns` are ignored.
//! - `INCREBOT_BENCH_LEGACY_NOTES=1`: notes, `after_replies` and `for_replies`
//!   go by everyone's messages, matched loosely, and count every reply since
//!   the latest push, as before 2026-10-15 (README §7, group round): a
//!   baseline for c43. Not covered: `prompted_by` word biases and bans
//!   without `for_replies`, whole words since that evening (c45 round).

// Progress output for a long-running batch loop.
#![allow(clippy::print_stderr)]

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use increbot::chat::window::{ChannelWindow, WindowMsg};
use increbot::chat::{PromptLimits, build_prompt};
use increbot::config::Config;
use increbot::engine::Backends;
use increbot::engine::worker::{CacheKey, Event, Job, spawn};
use increbot::gguf::Gguf;
use increbot::gpu::manager::Gpus;
use increbot::gpu::policy::State;
use increbot::model::Llama;
use increbot::persona::{Active, Persona};
use increbot::tokenizer::Tokenizer;
use serde_json::json;

const USER: u64 = 1116767210254385232;
const USER_NAME: &str = "incf_";
const BOT: u64 = 42;

struct Args {
    gpu: String,
    out: PathBuf,
    batches: usize,
    turns: usize,
    seed: u64,
    config: PathBuf,
    evict: bool,
    scenarios: Option<PathBuf>,
    repeats: usize,
}

/// One batch: the messages posted before each reply.
struct Plan {
    scenario: Option<serde_json::Value>,
    repeat: usize,
    turns: Vec<Vec<(String, String)>>,
    /// A novelty command run as a one-shot reply instead of chat turns.
    command: Option<CommandCase>,
    /// `"idle": true`: the chat opens after a quiet spell, with the scene line
    /// (`chat::idle`) drawn from the persona's lists by the batch's seed and
    /// kept for every turn, as in the bot's first 15 minutes.
    idle: bool,
}

/// `{"command": {"name": "hotdog-rater", "who": "me"|"bot"|<name>, "length",
/// "girth"}, "user": <requester>}`: the prompt the bot builds for it, from the
/// persona's `commands` setup (or the built-in one), its note and word biases.
#[derive(Clone)]
struct CommandCase {
    name: String,
    user: String,
    who: String,
    length: f64,
    girth: f64,
}

fn plans(a: &Args) -> Result<Vec<Plan>> {
    let Some(path) = &a.scenarios else {
        let turn = vec![(USER_NAME.to_string(), "floob".to_string())];
        return Ok((0..a.batches)
            .map(|_| Plan {
                scenario: None,
                repeat: 0,
                turns: vec![turn.clone(); a.turns],
                command: None,
                idle: false,
            })
            .collect());
    };
    let mut out = Vec::new();
    for (n, line) in std::fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), n + 1))?;
        if let Some(c) = v.get("command") {
            let case = (|| -> Option<CommandCase> {
                Some(CommandCase {
                    name: c["name"].as_str()?.to_string(),
                    user: v["user"].as_str()?.to_string(),
                    who: c["who"].as_str()?.to_string(),
                    length: c["length"].as_f64()?,
                    girth: c["girth"].as_f64()?,
                })
            })()
            .with_context(|| format!("{}:{}: a command needs name, who, length, girth and a user", path.display(), n + 1))?;
            if increbot::persona::CommandName::from_slash(&case.name).is_none() {
                bail!("{}:{}: unknown command {:?} (persona::CommandName)", path.display(), n + 1, case.name);
            }
            for repeat in 0..a.repeats {
                out.push(Plan {
                    scenario: Some(json!({ "id": v["id"], "category": v["category"] })),
                    repeat,
                    turns: vec![vec![(case.user.clone(), String::new())]],
                    command: Some(case.clone()),
                    idle: false,
                });
            }
            continue;
        }
        let turns = v["turns"]
            .as_array()
            .filter(|t| !t.is_empty())
            .with_context(|| format!("{}:{}: no turns", path.display(), n + 1))?
            .iter()
            .map(|turn| {
                turn.as_array()
                    .filter(|m| !m.is_empty())
                    .context("a turn must be a non-empty list of messages")?
                    .iter()
                    .map(|m| match (m["user"].as_str(), m["text"].as_str()) {
                        (Some(u), Some(t)) if !u.is_empty() => Ok((u.to_string(), t.to_string())),
                        _ => bail!("a message needs user and text"),
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("{}:{}", path.display(), n + 1))?;
        for repeat in 0..a.repeats {
            out.push(Plan {
                scenario: Some(json!({ "id": v["id"], "category": v["category"] })),
                repeat,
                turns: turns.clone(),
                command: None,
                idle: v["idle"].as_bool().unwrap_or(false),
            });
        }
    }
    Ok(out)
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        gpu: String::new(),
        out: PathBuf::new(),
        batches: 20,
        turns: 25,
        seed: 1,
        config: "config.local.rsn".into(),
        evict: false,
        scenarios: None,
        repeats: 4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--evict" {
            a.evict = true;
            continue;
        }
        let v = it.next().with_context(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--gpu" => a.gpu = v,
            "--out" => a.out = v.into(),
            "--batches" => a.batches = v.parse()?,
            "--turns" => a.turns = v.parse()?,
            "--seed" => a.seed = v.parse()?,
            "--config" => a.config = v.into(),
            "--scenarios" => a.scenarios = Some(v.into()),
            "--repeats" => a.repeats = v.parse()?,
            _ => bail!("unknown flag {flag}"),
        }
    }
    if a.gpu.is_empty() || a.out.as_os_str().is_empty() {
        bail!(
            "usage: floob_bench --gpu <pci> --out <dir> [--batches N] [--turns N] [--seed N] [--config F] [--evict] [--scenarios F [--repeats N]]"
        );
    }
    Ok(a)
}

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("increbot=warn")
        .with_writer(std::io::stderr)
        .try_init();
    let a = parse_args()?;
    let plans = plans(&a)?;
    std::fs::create_dir_all(&a.out)?;
    // Our gpu_other_mib is above the 2 GiB hog, the service's isn't.
    let env = |k: &str| match k {
        "INCREBOT_GPUS" => Some(format!("{}=bench", a.gpu)),
        "INCREBOT_GPU_OTHER_MIB" if a.evict => Some("4000".into()),
        _ => std::env::var(k).ok(),
    };
    let cfg = Config::load(&a.config, &env)?;
    let persona = Persona::load(std::path::Path::new(&cfg.discord.persona_file), &env)?;
    persona.check_against(&cfg)?;
    increbot::runtime::init_pool(cfg.runtime.threads, cfg.runtime.pin_threads)?;
    let gguf = Gguf::open(&increbot::hub::resolve(&cfg.model.repo, &cfg.model.file)?)?;
    let tok = Arc::new(Tokenizer::load(
        &increbot::hub::resolve(&cfg.model.tokenizer_repo, "tokenizer.json")?,
        &gguf,
    )?);
    let mut model = Llama::load(&gguf, &Default::default(), cfg.model.linear_impl)?;
    model.set_kv_f16(cfg.runtime.kv_cache_f16);
    drop(gguf);
    let model = Arc::new(model);

    let _hog = if a.evict {
        let pci: increbot::gpu::PciAddr = a.gpu.parse()?;
        let path = increbot::gpu::monitor::vram_used_path(pci);
        let mib = || increbot::gpu::monitor::read_u64(&path).map_or(0, |b| b >> 20);
        let dev = increbot::gpu::Device::open(pci)?;
        let hog = dev.vram(2 << 30)?;
        let t = Instant::now();
        while mib() > 3072 {
            if t.elapsed() > Duration::from_secs(20) {
                bail!("the service didn't release {}: {} MiB in use", a.gpu, mib());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Some((dev, hog))
    } else {
        None
    };
    let settings = cfg.runtime.gpu_settings(cfg.model.n_ctx).context("no GPU settings")?;
    let gpus = Gpus::start(settings, model.clone()).context("GPUs didn't start")?;
    let t = Instant::now();
    while gpus.cards[0].status().state != State::Ready {
        if t.elapsed() > Duration::from_secs(180) {
            bail!("GPU not ready after 180 s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let d = &cfg.discord;
    let active = Active::new(persona, &d.prompt_template, &d.bot_name, &[], &tok)?;
    let builder = active.prompt.clone();
    let count = |s: &str| tok.encode_prompt(s).map_or(usize::MAX, |v| v.len());
    // The bot refuses to start past this limit, so a run must too.
    let header_tokens = builder.check_header(cfg.generation.header_limit(), &count)?;
    let g = &cfg.generation;
    let p = &active.persona;
    let repeat_persona = gpus.any_ready() || !g.cpu_drop_persona_repeat;
    let legacy = std::env::var("INCREBOT_BENCH_LEGACY_NOTES").is_ok_and(|v| v == "1");
    std::fs::write(
        a.out.join("meta.json"),
        serde_json::to_string_pretty(&json!({
            "gpu": a.gpu,
            "batches": plans.len(),
            "turns": a.turns,
            "legacy_notes": legacy,
            "scenarios": a.scenarios.as_ref().map(|p| p.display().to_string()),
            "repeats": a.scenarios.as_ref().map(|_| a.repeats),
            "seed": a.seed,
            "user": USER_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "persona": p,
            "persona_line": p.line(),
            "header_tokens": header_tokens,
            "config": cfg,
        }))?,
    )?;

    let mut h = spawn(
        Backends::new(model, Some(gpus.clone()), cfg.model.n_ctx),
        tok.clone(),
        cfg.runtime.kv_cache_slots,
    )?;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let limits = PromptLimits {
        history_messages: cfg.discord.history_messages,
        max_prompt_tokens: g.max_prompt_tokens,
        max_message_tokens: g.max_message_tokens,
    };
    let buffer = cfg.discord.history_messages * 4;
    let t_all = Instant::now();
    // Each user name gets its own stable author id; incf_ keeps the real one.
    let mut authors: std::collections::HashMap<String, u64> = [(USER_NAME.to_string(), USER)].into();
    for (batch, plan) in plans.iter().enumerate() {
        let mut log = std::fs::File::create(a.out.join(format!("batch_{batch:02}.jsonl")))?;
        let mut live = ChannelWindow::default();
        let mut id = 1000;
        let t_batch = Instant::now();
        for (turn, posted) in (1..).zip(&plan.turns) {
            for (user, text) in posted {
                id += 1;
                let next = 2000 + authors.len() as u64;
                let author_id = *authors.entry(user.clone()).or_insert(next);
                live.push(
                    WindowMsg {
                        id,
                        author_id,
                        author: user.clone(),
                        text: text.clone(),
                        extras: Default::default(),
                        mentions: Vec::new(),
                    },
                    buffer,
                );
            }
            let trigger = id;
            let seed = a.seed + batch as u64 * 1000 + turn as u64;
            // As the bot after a quiet spell: the scene, picked by the batch's seed.
            let scene = plan.idle.then(|| {
                // (Not `seed % len`: batch × 1000 is a multiple of the list's 100.)
                let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(a.seed + batch as u64 * 1000);
                let activity = rand::seq::IndexedRandom::choose(p.idle_activities.as_slice(), &mut rng).cloned().unwrap_or_default();
                increbot::chat::idle::Scene {
                    activity,
                    thought: p.idle_thoughts.make(&mut rng),
                    drawn_ms: 0,
                    until_ms: None,
                }
                .line(&builder.bot_name)
            });
            let mut draw_used: Option<increbot::persona::CommandDraw> = None;
            // A command case: the one-shot prompt the bot builds, in its own
            // cache, as `/hotdog-rater` does. Otherwise a chat turn.
            let (said, note, text, stops, history_text, messages, key, command_name, posted_log) = match &plan.command {
                Some(c) => {
                    use increbot::discord::{hotdog, novelty::Who};
                    let who = match c.who.as_str() {
                        "me" => Who::Me,
                        "bot" => Who::Bot,
                        name => Who::Other(name.to_string()),
                    };
                    let name = increbot::persona::CommandName::from_slash(&c.name).expect("checked when loading the scenarios");
                    // As the bot: set-up line, moves and examples drawn from the
                    // persona's pools, here from the turn's seed (reproducible).
                    let mut drawn = p.draw(name, seed).unwrap_or_default();
                    let template = Some(drawn.template.clone())
                        .filter(|t| !t.is_empty())
                        .unwrap_or_else(|| hotdog::PROMPT.to_string());
                    // As the bot: the ownership line, for every persona.
                    drawn.facts.push(who.ownership("hotdog", &c.user, &builder.bot_name));
                    let draw = Some(drawn);
                    let said = hotdog::Hotdog::of_size(c.length, c.girth).prompt(&template, &builder.bot_name, &who);
                    let note = active.command_note(name, &said, draw.as_ref());
                    draw_used = draw.clone();
                    let one = increbot::chat::build_one_shot(
                        &builder,
                        &c.user,
                        &said,
                        repeat_persona,
                        note.as_deref(),
                        g.max_prompt_tokens,
                        &count,
                    )?;
                    let posted_log = vec![(c.user.clone(), said.clone())];
                    (said, note, one.text, one.stops, one.history, 1, CacheKey::Scratch, Some(name), posted_log)
                }
                None => {
                    let mut frozen = live.clone();
                    let said = posted.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join("\n");
                    // As the bot: notes look back over the window (`turns`).
                    let replies = |phrases: &[String], span: usize| {
                        let p = increbot::engine::sampler::Phrases::new(phrases);
                        live.replies_since(BOT, trigger, span, &|text| p.found_whole_in(&increbot::engine::sampler::normalized_words(text)))
                    };
                    let note = if legacy {
                        legacy_notes(&p.notes, &live, trigger)
                    } else {
                        active.note(&|n| live.since_replies(BOT, trigger, n), &|n| live.since_replies_by_asker(BOT, trigger, n), &replies)
                    };
                    let built = build_prompt(
                        &mut frozen,
                        &builder,
                        BOT,
                        Some(trigger),
                        repeat_persona,
                        note.as_deref(),
                        scene.as_deref(),
                        &limits,
                        &count,
                    )?;
                    live.advance_to(frozen.anchor());
                    // Each batch its own cache: no prefix reuse across batches.
                    let key = CacheKey::Channel(batch as u64 + 1);
                    (said, note, built.text, built.stops, built.history, built.messages, key, None, posted.clone())
                }
            };
            let stops = active.stops(stops);
            let history = tok.encode_prompt(&history_text)?.split_off(1);
            let prompt = tok.encode_prompt(&text)?;
            let prompt_len = prompt.len();
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            // As the bot: the persona's bans, no speaker label at the start,
            // no repeat of its last reply in this chat.
            let bans = {
                let names = stops.iter().filter_map(|s| s.strip_prefix('\n').and_then(|s| s.strip_suffix(':')).map(String::from));
                let previous = live.messages().filter(|m| m.author_id == BOT).max_by_key(|m| m.id).map(|m| m.text.clone());
                // As the bot: replies since a phrase, in the chat (or the one-shot's prompt).
                let one_shot = increbot::persona::one_shot_replies(&said);
                let replies = |phrases: &[String], span: usize| {
                    if command_name.is_some() {
                        return one_shot(phrases, span);
                    }
                    if legacy {
                        return legacy_replies(&live, trigger, phrases);
                    }
                    let p = increbot::engine::sampler::Phrases::new(phrases);
                    live.replies_since(BOT, trigger, span, &|text| p.found_whole_in(&increbot::engine::sampler::normalized_words(text)))
                };
                let b = active.bans(&said, command_name, &replies).with_prefixes(names).with_echo(previous.as_deref().unwrap_or(""));
                (!b.is_empty()).then(|| Arc::new(b))
            };
            let job = Job {
                key,
                prompt,
                // Everything posted this turn, as the bot's `unanswered` gives;
                // a command adds its own word biases.
                sampler: active.sampler_for(&said, command_name),
                seed,
                history,
                max_new_tokens: p.limits(command_name).max_new_tokens,
                max_new_tokens_gpu: p.limits(command_name).max_new_tokens_gpu,
                wrap_up_after: p.limits(command_name).wrap_up_after,
                wrap_up_after_gpu: p.limits(command_name).wrap_up_after_gpu,
                stops,
                max_lines: p.limits(command_name).max_lines,
                timeout: Duration::from_secs(g.timeout_secs),
                cancel: Arc::default(),
                events: tx,
                // On a GPU, as in the bot: a command's best of N.
                best_of: active.best_of(command_name, true),
                score: command_name.map(|c| c.scorer(&said, draw_used.as_ref())),
                bans,
            };
            let (text, out) = rt.block_on(async {
                h.jobs.send(job).await.ok().context("worker gone")?;
                let mut text = String::new();
                loop {
                    match rx.recv().await.context("worker dropped the job")? {
                        Event::Text(t) => text += &t,
                        Event::Finished(out) => {
                            // Like the dispatcher: the next job only after idle.
                            h.idle.recv().await;
                            break anyhow::Ok((text, out));
                        }
                        Event::Failed(e) => bail!("job failed: {e}"),
                        Event::Started | Event::Paused(_) => {}
                    }
                }
            })?;
            let reply = text.trim().to_string();
            let backend = h.stats.lock().unwrap().last_backend.to_string();
            writeln!(
                log,
                "{}",
                json!({
                    "batch": batch,
                    "turn": turn,
                    "scenario": plan.scenario,
                    "repeat": plan.repeat,
                    "note": note,
                    "scene": scene,
                    "user_messages": posted_log.iter().map(|(u, t)| json!({ "user": u, "text": t })).collect::<Vec<_>>(),
                    "command": command_name.map(|c| c.slash()),
                    "seed": seed,
                    "messages": messages,
                    "prompt_tokens": prompt_len,
                    "reused_tokens": out.reused_tokens,
                    "tokens": out.tokens.len(),
                    "decode_s": out.decode.as_secs_f64(),
                    "stop": format!("{:?}", out.stop),
                    "backend": backend,
                    "rollbacks": out.rollbacks,
                    "reply": reply,
                })
            )?;
            if !reply.is_empty() {
                id += 1;
                live.push(
                    WindowMsg {
                        id,
                        author_id: BOT,
                        author: cfg.discord.bot_name.clone(),
                        text: reply,
                        extras: Default::default(),
                        mentions: Vec::new(),
                    },
                    buffer,
                );
            }
        }
        eprintln!(
            "batch {batch:2}/{} done in {:.1} s ({:.0} s total)",
            plans.len(),
            t_batch.elapsed().as_secs_f32(),
            t_all.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

/// Notes as `chat::prompt::notes_for` chose them before 2026-10-15: every
/// note on everyone's messages since the bot's `turns`-th last reply,
/// matched loosely; `Safety` first.
fn legacy_notes(notes: &[increbot::config::PromptNote], live: &ChannelWindow, trigger: u64) -> Option<String> {
    use increbot::engine::sampler::{Phrases, normalized_words};
    let mut hits: Vec<&increbot::config::PromptNote> = notes
        .iter()
        .filter(|n| {
            if let Some(after) = n.after_replies {
                return legacy_replies(live, trigger, &n.prompted_by).is_some_and(|c| c >= after && c < after + n.turns.max(1));
            }
            Phrases::new(&n.prompted_by).found_in(&normalized_words(&live.since_replies(BOT, trigger, n.turns.max(1))))
        })
        .collect();
    hits.sort_by_key(|n| n.kind != increbot::config::NoteKind::Safety);
    let texts: Vec<&str> = hits.iter().map(|n| n.text.trim()).collect();
    (!texts.is_empty()).then(|| texts.join("\n"))
}

/// `ChannelWindow::replies_since` before 2026-10-15: anyone's latest message
/// with one of `phrases` (loose), and every bot reply after it.
fn legacy_replies(live: &ChannelWindow, trigger: u64, phrases: &[String]) -> Option<usize> {
    use increbot::engine::sampler::{Phrases, normalized_words};
    let p = Phrases::new(phrases);
    let upto: Vec<&WindowMsg> = live.messages().filter(|m| m.id <= trigger).collect();
    let at = upto.iter().rposition(|m| m.author_id != BOT && p.found_in(&normalized_words(&m.text)))?;
    Some(upto[at + 1..].iter().filter(|m| m.author_id == BOT).count())
}
