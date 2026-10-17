//! Labeling inputs and statistics for the persona benchmark (bench/persona/README.md
//! §3–4), over the JSONL that `floob_bench` writes. No model needed.
//!
// This tool's whole job is printing a report to the terminal.
#![allow(clippy::print_stdout)]
//!
//! `cargo run --release --example persona_report -- prep <labels_dir> [--agents N] [--overlap K] <run_dir>...`
//! `cargo run --release --example persona_report -- analyze <run_dir> [<labels_dir>]`
//! `cargo run --release --example persona_report -- header <persona file>...` (header size vs the startup limit)
//! `cargo run --release --example persona_report -- discourage [persona file]` (its bias rules in INCREBOT_DISCOURAGE form)
//! `cargo run --release --example persona_report -- fingerprint [<run_dir>...]` (the `/stats` persona line)
//! `cargo run --release --example persona_report -- pairwise …` (blind pairwise humor ratings; `pairwise.rs`)
//! `cargo run --release --example persona_report -- hotdog …` (hotdog-rater guardrails; `hotdog.rs`)
//! `cargo run --release --example persona_report -- quality <labels_dir> <run_dir>...` (the quality set's report; `quality.rs`)
//!
//! - Scenario runs (`floob_bench --scenarios`) show each turn's real user
//!   messages to the labelers, and `analyze` adds young and borderline rates
//!   by category and lists the scenarios with young replies.
//! - `prep` pools the batches of the given runs, gives each an anonymous
//!   3-digit code (4 digits past 900 batches), deals them (fixed seed) over N agents (default 4) so every
//!   agent gets an even share of every run, and gives K batches (default 3,
//!   taken from each run in turn) to a second agent too (the agreement check).
//!   Writes `agent_N.txt` (what an agent sees) and `key.json` (code → run and
//!   batch; keep it from the agents).
//! - `analyze` prints reply statistics, word and `*action*` frequencies, and,
//!   with labels (`agent_N.jsonl` beside `key.json`; only codes from this run
//!   are used), the young rate with a batch-bootstrap 95% CI, the borderline
//!   rate, breakdowns by batch, turn position and tag, and candidate stems.

mod group;
mod hotdog;
mod pairwise;
mod quality;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};

const AGENTS: usize = 4;
const OVERLAPS: usize = 3;
const SEED: u64 = 20261012;

struct Turn {
    turn: u64,
    tokens: u64,
    decode_s: f64,
    stop: String,
    reply: String,
    /// (user, text) posted before the reply; empty in runs from before
    /// scenarios, where it was always incf_ saying "floob".
    posted: Vec<(String, String)>,
    /// Scenario (id, category) for `--scenarios` runs.
    scenario: Option<(String, String)>,
}

struct Label {
    label: String,
    tags: Vec<String>,
    evidence: Vec<String>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("prep") if args.len() >= 3 => prep(Path::new(&args[1]), &args[2..]),
        Some("analyze") if args.len() >= 2 => analyze(Path::new(&args[1]), args.get(2).map(Path::new)),
        Some("tokens") if args.len() >= 2 => tokens(&args[1..]),
        Some("header") if args.len() >= 2 => header(&args[1..]),
        Some("fingerprint") => fingerprint(&args[1..]),
        Some("discourage") => discourage(args.get(1).map(String::as_str)),
        Some("pairwise") => pairwise::main(&args[1..]),
        Some("hotdog") => hotdog::main(&args[1..]),
        Some("quality") => quality::main(&args[1..]),
        Some("group") => group::main(&args[1..]),
        _ => bail!(
            "usage: persona_report prep <labels_dir> [--agents N] [--overlap K] <run_dir>... | analyze <run_dir> [<labels_dir>] | tokens <stem>... | header <persona file>... | discourage [persona file] | fingerprint [<run_dir>...] | pairwise … | hotdog … | quality <labels_dir> <run_dir>... | group <labels_dir> <run_dir>..."
        ),
    }
}

/// The persona line the bot shows in `/stats` and `/version` ("c18
/// (9e51525)"), for each run from the persona file it saved (`persona.rsn`,
/// under its `env.txt`, so an `INCREBOT_DISCOURAGE` override counts). No runs =
/// the live setup (`config.local.rsn` under this shell's env). Runs from before
/// persona files (0001–0065) have none.
fn fingerprint(runs: &[String]) -> Result<()> {
    if runs.is_empty() {
        let (_, persona) = live()?;
        println!("live: {}", persona.line());
        return Ok(());
    }
    for run in runs {
        let dir = Path::new(run);
        let name = dir.file_name().map_or(run.clone(), |n| n.to_string_lossy().into_owned());
        match run_persona(dir) {
            Ok(p) => println!("{name}: {}", p.line()),
            Err(e) => println!("{name}: - ({e:#})"),
        }
    }
    Ok(())
}

/// The live config and the persona it names.
fn live() -> Result<(increbot::config::Config, increbot::persona::Persona)> {
    let env = |k: &str| std::env::var(k).ok();
    let cfg = increbot::config::Config::load(Path::new("config.local.rsn"), &env)?;
    let persona = increbot::persona::Persona::load(Path::new(&cfg.discord.persona_file), &env)?;
    Ok((cfg, persona))
}

/// A run's saved persona file, under the env the run had.
fn run_persona(dir: &Path) -> Result<increbot::persona::Persona> {
    let file = dir.join("persona.rsn");
    anyhow::ensure!(file.exists(), "no persona.rsn: from before persona files");
    let env: HashMap<String, String> = std::fs::read_to_string(dir.join("env.txt"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    increbot::persona::Persona::load(&file, &|k| env.get(k).cloned())
}

/// A persona file's `discourage` rules in the `INCREBOT_DISCOURAGE` form, one
/// per line, to compare lists. No file = the live persona.
fn discourage(file: Option<&str>) -> Result<()> {
    let persona = match file {
        Some(f) => increbot::persona::Persona::load(Path::new(f), &|k| std::env::var(k).ok())?,
        None => live()?.1,
    };
    for w in &persona.discourage {
        let mut s = format!("{}={}:{}", w.stem, w.bias, w.min);
        let phrases = |p: &[String]| p.iter().map(|t| t.replace(' ', "_")).collect::<Vec<_>>().join("|");
        if !w.after.is_empty() {
            s += &format!("@{}/{}", phrases(&w.after), w.within);
        }
        if !w.prompted_by.is_empty() {
            s += &format!("^{}", phrases(&w.prompted_by));
        }
        println!("{s}");
    }
    Ok(())
}

/// Each persona file's header size (persona sections + template, no history)
/// under `config.local.rsn`, against the limit the bot enforces at startup.
fn header(files: &[String]) -> Result<()> {
    use increbot::config::Config;
    let env = |k: &str| std::env::var(k).ok();
    let cfg = Config::load(Path::new("config.local.rsn"), &env)?;
    let gguf = increbot::gguf::Gguf::open(&increbot::hub::resolve(&cfg.model.repo, &cfg.model.file)?)?;
    let tok = increbot::tokenizer::Tokenizer::load(
        &increbot::hub::resolve(&cfg.model.tokenizer_repo, "tokenizer.json")?,
        &gguf,
    )?;
    let count = |s: &str| tok.encode_prompt(s).map_or(usize::MAX, |v| v.len());
    let limit = cfg.generation.header_limit();
    for f in files {
        let persona = increbot::persona::Persona::load(Path::new(f), &env)?;
        let d = &cfg.discord;
        let active = increbot::persona::Active::new(persona, &d.prompt_template, &d.bot_name, &[], &tok)?;
        let n = active.prompt.check_header(usize::MAX, &count)?;
        let verdict = if n <= limit { "ok" } else { "TOO LONG" };
        println!(
            "{f}: {n} tokens (limit {limit}, history room {}) {verdict}; {}",
            cfg.generation.max_prompt_tokens.saturating_sub(n),
            active.persona.line()
        );
    }
    Ok(())
}

/// The word-start tokens each stem's bias hits anywhere (the rule in
/// `WordBias::new`): leading letters that spell >= `min` (default 3) letters of
/// the stem, or run past it. Collateral shows up as short common words ("see"
/// for "seek"). A stem is `stem` or `stem:min`.
fn tokens(stems: &[String]) -> Result<()> {
    use increbot::config::Config;
    let cfg = Config::load(Path::new("config.local.rsn"), &|k| std::env::var(k).ok())?;
    let gguf = increbot::gguf::Gguf::open(&increbot::hub::resolve(&cfg.model.repo, &cfg.model.file)?)?;
    let tok = increbot::tokenizer::Tokenizer::load(
        &increbot::hub::resolve(&cfg.model.tokenizer_repo, "tokenizer.json")?,
        &gguf,
    )?;
    let pieces = tok.pieces();
    for stem in stems {
        let (stem, min) = stem.split_once(':').unwrap_or((stem, "3"));
        let stem = stem.to_ascii_lowercase();
        let min = min.parse::<usize>()?.clamp(1, stem.len());
        let mut hit: Vec<String> = pieces
            .iter()
            .filter_map(|p| {
                let start = p.find(|c: char| c.is_ascii_alphabetic())?;
                if start == 0 {
                    return None;
                }
                let letters: String = p[start..].chars().take_while(char::is_ascii_alphabetic).collect();
                let l = letters.to_ascii_lowercase();
                (l.len() >= min && (stem.starts_with(&l) || l.starts_with(&stem))).then(|| format!("{p:?}"))
            })
            .collect();
        hit.sort();
        println!("{stem}: {}", hit.join(" "));
    }
    Ok(())
}

fn jsonl(path: &Path) -> Result<Vec<Value>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}

/// The run's batches by file stem ("batch_00"), each turn in order.
fn load_run(run: &Path) -> Result<BTreeMap<String, Vec<Turn>>> {
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(run).with_context(|| format!("reading {}", run.display()))? {
        let path = e?.path();
        let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        if !stem.starts_with("batch_") || path.extension().is_none_or(|x| x != "jsonl") {
            continue;
        }
        let turns = jsonl(&path)?
            .iter()
            .map(|v| Turn {
                turn: v["turn"].as_u64().unwrap_or(0),
                tokens: v["tokens"].as_u64().unwrap_or(0),
                decode_s: v["decode_s"].as_f64().unwrap_or(0.0),
                stop: v["stop"].as_str().unwrap_or("").into(),
                reply: v["reply"].as_str().unwrap_or("").into(),
                posted: v["user_messages"]
                    .as_array()
                    .map(|ms| {
                        ms.iter()
                            .map(|m| {
                                let s = |k: &str| m[k].as_str().unwrap_or("").replace('\n', " ");
                                (s("user"), s("text"))
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                scenario: v["scenario"]["id"].as_str().map(|id| {
                    (id.to_string(), v["scenario"]["category"].as_str().unwrap_or("").to_string())
                }),
            })
            .collect();
        out.insert(stem, turns);
    }
    if out.is_empty() {
        bail!("no batch_*.jsonl in {}", run.display());
    }
    Ok(out)
}

fn prep(out: &Path, args: &[String]) -> Result<()> {
    let (mut agents, mut overlaps, mut runs) = (AGENTS, OVERLAPS, Vec::new());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--agents" => agents = it.next().context("--agents N")?.parse::<usize>()?.max(2),
            "--overlap" => overlaps = it.next().context("--overlap K")?.parse()?,
            _ => runs.push(a.clone()),
        }
    }
    std::fs::create_dir_all(out)?;
    let mut rng = StdRng::seed_from_u64(SEED);
    // Each run's batches shuffled, then dealt run after run with one running
    // counter, so every agent gets an even share of every run.
    let mut by_run = Vec::new();
    for run in &runs {
        let mut batches: Vec<_> = load_run(Path::new(run))?.into_iter().map(|(n, t)| (run.clone(), n, t)).collect();
        batches.shuffle(&mut rng);
        by_run.push(batches);
    }
    let total: usize = by_run.iter().map(Vec::len).sum();
    // 3-digit codes, or 4-digit ones for rounds over 900 batches.
    let (span, first) = if total <= 900 { (900, 100) } else { (9000, 1000) };
    anyhow::ensure!(total <= span, "{total} batches: more than {span} codes; split the round");
    let codes: Vec<usize> = rand::seq::index::sample(&mut rng, span, total).into_iter().map(|i| i + first).collect();
    let mut key = serde_json::Map::new();
    let mut assign: Vec<Vec<(usize, &[Turn])>> = vec![Vec::new(); agents];
    for (i, ((run, name, turns), &code)) in by_run.iter().flatten().zip(&codes).enumerate() {
        key.insert(code.to_string(), json!({ "run": run, "batch": name }));
        assign[i % agents].push((code, turns.as_slice()));
    }
    // Overlap batches: taken from each run in turn (the k-th batch dealt from
    // that run), each also given to another agent, spread over agents.
    let mut overlap = Vec::new();
    for j in 0..overlaps {
        let r = j % by_run.len();
        let idx = by_run[..r].iter().map(Vec::len).sum::<usize>() + j / by_run.len();
        if j / by_run.len() >= by_run[r].len() {
            break;
        }
        let (owner, code) = (idx % agents, codes[idx]);
        let to = (owner + 1 + j) % agents;
        let to = if to == owner { (to + 1) % agents } else { to };
        assign[to].push((code, by_run[r][j / by_run.len()].2.as_slice()));
        overlap.push(code);
    }
    key.insert("overlap".into(), json!(overlap));
    for (n, items) in assign.iter_mut().enumerate() {
        items.shuffle(&mut rng);
        let mut text = String::new();
        for (code, turns) in items.iter() {
            text += &format!("=== BATCH {code} ===\n");
            for t in turns.iter() {
                if t.posted.is_empty() {
                    text += &format!("[turn {}] incf_: floob\n", t.turn);
                }
                for (user, said) in &t.posted {
                    text += &format!("[turn {}] {user}: {said}\n", t.turn);
                }
                text += &format!("[turn {}] floob: {}\n", t.turn, t.reply);
            }
            text += "\n";
        }
        std::fs::write(out.join(format!("agent_{}.txt", n + 1)), text)?;
        let codes: Vec<_> = items.iter().map(|(c, _)| c).collect();
        println!("agent_{}: {codes:?}", n + 1);
    }
    std::fs::write(out.join("key.json"), serde_json::to_string_pretty(&key)?)?;
    println!("overlap: {overlap:?}");
    Ok(())
}

/// Word similarity at or above which a reply counts as a near-repeat (README §4).
const NEAR_REPEAT: f64 = 0.5;
const PARTIAL_REPEAT: f64 = 0.4;

/// 1 − (word-level edit distance / the longer reply's word count), on
/// lowercased words with punctuation dropped: 1 for the same words, about 0.1
/// for unrelated replies, 0.8 for "dont u like it?? i put so much work in to
/// it!!" reworded with two words changed.
fn word_similarity(a: &str, b: &str) -> f64 {
    let (a, b) = (words(a), words(b));
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    1.0 - prev[b.len()] as f64 / a.len().max(b.len()) as f64
}

/// Lowercase words of ASCII letters and apostrophes.
fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !(c.is_ascii_lowercase() || c == '\''))
        .map(|w| w.trim_matches('\''))
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect()
}

/// The `*action*` spans.
fn actions(s: &str) -> Vec<&str> {
    s.split('*').skip(1).step_by(2).filter(|a| !a.trim().is_empty()).collect()
}

/// Labels for `run` by (batch name, turn); the first agent's where two exist.
fn load_labels(run: &Path, dir: &Path) -> Result<HashMap<(String, u64), Label>> {
    let key: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("key.json"))?)?;
    let same = |r: &str| -> bool {
        let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        canon(Path::new(r)) == canon(run)
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            n.starts_with("agent_") && n.ends_with(".jsonl")
        })
        .collect();
    files.sort();
    let mut all: HashMap<(String, u64), Vec<Label>> = HashMap::new();
    for f in &files {
        for v in jsonl(f)? {
            let code = match &v["batch"] {
                Value::Number(n) => n.to_string(),
                Value::String(s) => s.clone(),
                _ => continue,
            };
            let Some(k) = key.get(&code) else { continue };
            if !same(k["run"].as_str().unwrap_or("")) {
                continue;
            }
            let strs = |x: &Value| -> Vec<String> {
                x.as_array()
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_default()
            };
            all.entry((k["batch"].as_str().unwrap_or("").into(), v["turn"].as_u64().unwrap_or(0)))
                .or_default()
                .push(Label {
                    label: v["label"].as_str().unwrap_or("").into(),
                    tags: strs(&v["tags"]),
                    evidence: strs(&v["evidence"]),
                });
        }
    }
    // A list of codes; older passes stored a single number.
    let overlaps: Vec<String> = match key.get("overlap") {
        Some(Value::Array(a)) => a.iter().map(Value::to_string).collect(),
        Some(o) => vec![o.to_string()],
        None => Vec::new(),
    };
    for ov in overlaps {
        let Some(k) = key.get(&ov) else { continue };
        if !same(k["run"].as_str().unwrap_or("")) {
            continue;
        }
        let pairs: Vec<_> = all
            .iter()
            .filter(|((b, _), v)| b == k["batch"].as_str().unwrap_or("") && v.len() == 2)
            .map(|(_, v)| (&v[0].label, &v[1].label))
            .collect();
        let exact = pairs.iter().filter(|(a, b)| a == b).count();
        let young = pairs.iter().filter(|(a, b)| (*a == "young") == (*b == "young")).count();
        println!(
            "overlap batch {ov}: exact agreement {exact}/{n}, young-vs-not {young}/{n}",
            n = pairs.len()
        );
    }
    Ok(all.into_iter().map(|(k, mut v)| (k, v.swap_remove(0))).collect())
}

fn top<K: Ord + Clone + std::fmt::Display>(m: &BTreeMap<K, usize>, n: usize) -> String {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    v.iter().take(n).map(|(k, c)| format!("{k} {c}")).collect::<Vec<_>>().join(", ")
}

fn analyze(run: &Path, labels_dir: Option<&Path>) -> Result<()> {
    let batches = load_run(run)?;
    let all: Vec<(&String, &Turn)> = batches.iter().flat_map(|(b, ts)| ts.iter().map(move |t| (b, t))).collect();
    let n = all.len() as f64;
    println!("{}: {} batches, {} replies", run.display(), batches.len(), all.len());
    let mut stops = BTreeMap::new();
    for (_, t) in &all {
        *stops.entry(t.stop.clone()).or_insert(0) += 1;
    }
    let n_act: Vec<usize> = all.iter().map(|(_, t)| actions(&t.reply).len()).collect();
    println!(
        "tokens/reply mean {:.1}, max {}; decode mean {:.2} s; stops: {}",
        all.iter().map(|(_, t)| t.tokens as f64).sum::<f64>() / n,
        all.iter().map(|(_, t)| t.tokens).max().unwrap_or(0),
        all.iter().map(|(_, t)| t.decode_s).sum::<f64>() / n,
        top(&stops, 5),
    );
    println!(
        "actions/reply mean {:.2}; more than one in {:.0}%",
        n_act.iter().sum::<usize>() as f64 / n,
        100.0 * n_act.iter().filter(|&&a| a > 1).count() as f64 / n
    );

    // Near-repeats: a reply that rewords one of floob's own recent replies.
    // Exact copies are caught by DRY; these change a word or two (design.md §5).
    {
        let (mut prev1, mut partial, mut any3, mut pairs) = (0, 0, 0, 0);
        let mut examples = Vec::new();
        for ts in batches.values() {
            for (i, t) in ts.iter().enumerate().skip(1) {
                pairs += 1;
                let s = word_similarity(&ts[i - 1].reply, &t.reply);
                prev1 += (s >= NEAR_REPEAT) as usize;
                partial += (s >= PARTIAL_REPEAT) as usize;
                let best = ts[i.saturating_sub(3)..i].iter().map(|p| word_similarity(&p.reply, &t.reply)).fold(0.0, f64::max);
                any3 += (best >= NEAR_REPEAT) as usize;
                if s >= NEAR_REPEAT && examples.len() < 3 {
                    examples.push(format!("{:.2} {:?}", s, t.reply.chars().take(70).collect::<String>()));
                }
            }
        }
        let pct = |n: usize| 100.0 * n as f64 / pairs.max(1) as f64;
        println!(
            "near-repeats of the previous reply (word similarity >= {NEAR_REPEAT}): {prev1}/{pairs} ({:.1}%); >= {PARTIAL_REPEAT}: {partial} ({:.1}%); of any of the last 3: {any3} ({:.1}%)",
            pct(prev1),
            pct(partial),
            pct(any3)
        );
        if !examples.is_empty() {
            println!("  e.g. {}", examples.join(" | "));
        }
    }

    let labels = match labels_dir {
        Some(d) => load_labels(run, d)?,
        None => HashMap::new(),
    };
    let is = |b: &str, t: u64, l: &str| labels.get(&(b.to_string(), t)).is_some_and(|x| x.label == l);
    if !labels.is_empty() {
        println!("labeled {}/{}", labels.len(), all.len());
        let names: Vec<&String> = batches.keys().collect();
        let rate = |sel: &[&String], l: &str| -> f64 {
            let (mut hit, mut tot) = (0, 0);
            for b in sel {
                for t in &batches[*b] {
                    if labels.contains_key(&((*b).clone(), t.turn)) {
                        tot += 1;
                        hit += is(b, t.turn, l) as usize;
                    }
                }
            }
            hit as f64 / tot.max(1) as f64
        };
        // Whole batches resampled: turns within a batch are correlated.
        let ci = |l: &str| -> (f64, f64, f64) {
            let mut rng = StdRng::seed_from_u64(1);
            let mut rs: Vec<f64> = (0..5000)
                .map(|_| {
                    let s: Vec<&String> = (0..names.len()).map(|_| names[rng.random_range(0..names.len())]).collect();
                    rate(&s, l)
                })
                .collect();
            rs.sort_by(f64::total_cmp);
            (rate(&names, l), rs[125], rs[4874])
        };
        for (l, what) in [("young", "young (target)"), ("borderline", "borderline (ok)")] {
            let (r, lo, hi) = ci(l);
            println!("{what:16} {:5.1}%  [{:.1}, {:.1}]", 100.0 * r, 100.0 * lo, 100.0 * hi);
        }
        let per: Vec<String> = names.iter().map(|b| format!("{}:{:.0}", &b[6..], 100.0 * rate(&[b], "young"))).collect();
        println!("young % per batch: {}", per.join(" "));
        // Scenario runs: by category, then the scenarios with any young reply.
        let scenario = |b: &String| batches[b].first().and_then(|t| t.scenario.clone());
        if names.iter().any(|b| scenario(b).is_some()) {
            let mut cats: BTreeMap<String, Vec<&String>> = BTreeMap::new();
            let mut ids: BTreeMap<(String, String), Vec<&String>> = BTreeMap::new();
            for b in &names {
                if let Some((id, cat)) = scenario(b) {
                    cats.entry(cat.clone()).or_default().push(b);
                    ids.entry((cat, id)).or_default().push(b);
                }
            }
            for (cat, sel) in &cats {
                let n: usize = sel.iter().map(|b| batches[*b].len()).sum();
                println!(
                    "category {cat:16} young {:5.1}%  borderline {:5.1}%  ({n} replies)",
                    100.0 * rate(sel, "young"),
                    100.0 * rate(sel, "borderline")
                );
            }
            let worst: Vec<String> = ids
                .iter()
                .map(|((_, id), sel)| (id, rate(sel, "young")))
                .filter(|(_, r)| *r > 0.0)
                .map(|(id, r)| format!("{id} {:.0}%", 100.0 * r))
                .collect();
            println!("scenarios with young replies: {}", worst.join(", "));
        }
        for lo in (1..=25).step_by(5) {
            let xs: Vec<_> = labels.iter().filter(|((_, t), _)| (lo..lo + 5).contains(t)).collect();
            if !xs.is_empty() {
                let share = |l: &str| 100.0 * xs.iter().filter(|(_, x)| x.label == l).count() as f64 / xs.len() as f64;
                println!("turns {lo:2}-{:2}: young {:.0}%, borderline {:.0}%", lo + 4, share("young"), share("borderline"));
            }
        }
        let mut tags = BTreeMap::new();
        let mut evidence = BTreeMap::new();
        for x in labels.values() {
            for t in &x.tags {
                *tags.entry(t.clone()).or_insert(0) += 1;
            }
            if x.label == "young" {
                for e in &x.evidence {
                    *evidence.entry(e.to_lowercase()).or_insert(0) += 1;
                }
            }
        }
        println!("tags: {}", top(&tags, 20));
        println!("top young evidence: {}", top(&evidence, 40));
    }

    // Counts are of replies containing the word, stem or action verb.
    let (mut wc, mut sc, mut sy, mut ac) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (b, t) in &all {
        let ws: BTreeSet<String> = words(&t.reply).into_iter().collect();
        let young = is(b, t.turn, "young");
        for w in ws.iter().filter(|w| w.len() > 3) {
            *wc.entry(w.clone()).or_insert(0) += 1;
        }
        let stems: BTreeSet<&str> = ws
            .iter()
            .flat_map(|w| [4, 5, 6].into_iter().filter(|&k| w.len() >= k).map(move |k| &w[..k]))
            .collect();
        for s in stems {
            *sc.entry(s.to_string()).or_insert(0) += 1;
            *sy.entry(s.to_string()).or_insert(0usize) += young as usize;
        }
        for a in actions(&t.reply) {
            if let Some(v) = words(a).into_iter().next() {
                *ac.entry(v).or_insert(0) += 1;
            }
        }
    }
    println!("\naction verbs: {}", top(&ac, 40));
    println!("\nwords: {}", top(&wc, 60));
    if !labels.is_empty() {
        // young occurrences × young share, for stems in at least 5 replies.
        let mut scored: Vec<(f64, &String)> = sc
            .iter()
            .filter(|(s, c)| **c >= 5 && sy[*s] > 0)
            .map(|(s, c)| ((sy[s] * sy[s]) as f64 / *c as f64, s))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        println!("\ncandidate stems (score, stem, replies, young, young share):");
        for (score, s) in scored.iter().take(50) {
            println!("  {score:6.1} {s:8} {:4} {:4} {:3.0}%", sc[*s], sy[*s], 100.0 * sy[*s] as f64 / sc[*s] as f64);
        }
    }
    Ok(())
}
