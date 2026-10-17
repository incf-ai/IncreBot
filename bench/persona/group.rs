//! The group set's report (README.md §2, "Group set"):
//!
//! `persona_report group <labels_dir> <run_dir>...` reads the labelers'
//! `agent_N.jsonl` (the first label of an overlap batch counts) with
//! `key.json`, and the runs' batches, and prints per run: per category the
//! mean `score`, `funny` rate, tag rates and each field's values; stance
//! flips per argument chat; the turn he first gives in per pressure chat;
//! the preference profile. Then the script measures, no labels needed: the
//! notes fired per category and scenario (by kind, from the run's own
//! `persona.rsn`), names (the asker's used, the persona's example names
//! leaking), trait coverage and repeats of an earlier reply in the chat.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

const TAGS: &[&str] = &[
    "hijacks",
    "answers_for_others",
    "speaks_for_user",
    "flip_flops",
    "wrong_addressee",
    "ignores_message",
    "non_sequitur",
    "contradicts",
    "pushover",
    "never_gives_in",
    "under_reacts",
    "over_reacts",
    "flat",
    "breaks_character",
];

/// Labeled fields with a fixed set of values.
const FIELDS: &[&str] = &["context", "answers", "gives_in", "person", "correct", "consistent", "bit", "callback", "mood"];

/// The names in the persona's examples, and two the model made up before
/// (round q01): none is a user in the group sets.
const EXAMPLE_NAMES: &[&str] = &["tib", "wren", "nell", "jory", "bram", "pip", "bartholomew"];

/// Trait buckets: a reply counts for a bucket with any of its word starts.
const TRAITS: &[(&str, &[&str])] = &[
    ("shinies", &["shin", "hoard", "bottlecap", "button"]),
    ("traps", &["trap", "contraption", "gizmo"]),
    ("digging", &["tunnel", "dig", "burrow"]),
    ("snacks", &["snack", "beetle", "bug", "roast", "crunch"]),
    ("warm rock", &["warm rock", "warmest rock"]),
    ("gambling", &["dice", "bet", "gambl", "wager"]),
    ("forgetting", &["forg", "rememb"]),
    ("skittish", &["squeak", "hide", "hides", "hiding", "eep"]),
];

/// One reply's labels.
#[derive(Default, Clone)]
struct Label {
    score: Option<f64>,
    funny: Option<bool>,
    tags: Vec<String>,
    fields: BTreeMap<&'static str, String>,
    stance: Option<String>,
    answer: Option<String>,
}

pub fn main(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("usage: persona_report group <labels_dir> <run_dir>...");
    }
    let labels = Path::new(&args[0]);
    let key: Value = serde_json::from_str(
        &std::fs::read_to_string(labels.join("key.json")).with_context(|| format!("reading {}/key.json", labels.display()))?,
    )?;
    let mut got: HashMap<(String, String, u64), Label> = HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(labels)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("agent_") && n.to_string_lossy().ends_with(".jsonl")))
        .collect();
    files.sort();
    for f in files {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).with_context(|| format!("{}", f.display()))?;
            let code = match &v["batch"] {
                Value::Number(n) => n.to_string(),
                Value::String(s) => s.clone(),
                _ => continue,
            };
            let Some(k) = key.get(&code) else { continue };
            let id = (name(k["run"].as_str().unwrap_or("")), k["batch"].as_str().unwrap_or("").to_string(), v["turn"].as_u64().unwrap_or(0));
            let text = |k: &str| match &v[k] {
                Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_lowercase()),
                Value::Bool(b) => Some(b.to_string()),
                _ => None,
            };
            let label = Label {
                score: v["score"].as_f64(),
                funny: v["funny"].as_bool(),
                tags: v["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default(),
                fields: FIELDS.iter().filter_map(|f| text(f).map(|t| (*f, t))).collect(),
                stance: text("stance"),
                answer: text("answer"),
            };
            got.entry(id).or_insert(label);
        }
    }
    for run in &args[1..] {
        report(Path::new(run), &got)?;
    }
    Ok(())
}

fn name(run: &str) -> String {
    Path::new(run.trim_end_matches('/')).file_name().map_or(run.to_string(), |n| n.to_string_lossy().into_owned())
}

fn pct(k: usize, n: usize) -> f64 {
    if n == 0 { 0.0 } else { 100.0 * k as f64 / n as f64 }
}

#[derive(Default)]
struct Tally {
    n: usize,
    scored: usize,
    score: f64,
    funny: usize,
    rated: usize,
    tags: BTreeMap<&'static str, usize>,
    fields: BTreeMap<&'static str, BTreeMap<String, usize>>,
}

impl Tally {
    fn add(&mut self, l: &Label) {
        self.n += 1;
        if let Some(s) = l.score {
            self.scored += 1;
            self.score += s;
        }
        if let Some(f) = l.funny {
            self.rated += 1;
            self.funny += usize::from(f);
        }
        for t in TAGS {
            if l.tags.iter().any(|x| x == t) {
                *self.tags.entry(t).or_default() += 1;
            }
        }
        for (f, v) in &l.fields {
            *self.fields.entry(f).or_default().entry(v.clone()).or_default() += 1;
        }
    }

    fn line(&self) -> String {
        let mut s = format!(
            "{:>5} replies  score {:.2}  funny {:>4.1}%",
            self.n,
            if self.scored == 0 { 0.0 } else { self.score / self.scored as f64 },
            pct(self.funny, self.rated)
        );
        for t in TAGS {
            let k = self.tags.get(t).copied().unwrap_or(0);
            if k > 0 {
                s += &format!("  {t} {:.1}%", pct(k, self.n));
            }
        }
        for (f, vals) in &self.fields {
            let n: usize = vals.values().sum();
            let parts: Vec<String> = vals.iter().map(|(v, k)| format!("{v} {:.0}%", pct(*k, n))).collect();
            s += &format!("\n             {f} (n={n}): {}", parts.join(", "));
        }
        s
    }
}

/// One logged reply, with what the report needs from the batch log.
struct Reply {
    turn: u64,
    scenario: String,
    category: String,
    asker: String,
    users: Vec<String>,
    reply: String,
    notes: Vec<String>,
    label: Option<Label>,
}

fn report(run: &Path, got: &HashMap<(String, String, u64), Label>) -> Result<()> {
    let rn = name(&run.to_string_lossy());
    // The run's own persona: which note is which kind.
    let persona = increbot::persona::Persona::load(&run.join("persona.rsn"), &|_| None)
        .with_context(|| format!("{}/persona.rsn", run.display()))?;
    let kinds: HashMap<String, (String, String)> = persona
        .notes
        .iter()
        .map(|n| {
            let short: String = n.text.trim().trim_start_matches('(').split_whitespace().take(5).collect::<Vec<_>>().join(" ");
            (n.text.trim().to_string(), (format!("{:?}", n.kind), short))
        })
        .collect();
    let mut chats: Vec<Vec<Reply>> = Vec::new();
    let mut paths: Vec<_> = std::fs::read_dir(run)
        .with_context(|| format!("reading {}", run.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("batch_")) && p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    paths.sort();
    for p in paths {
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let mut chat = Vec::new();
        let mut users: Vec<String> = Vec::new();
        for line in std::fs::read_to_string(&p)?.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line)?;
            let turn = v["turn"].as_u64().unwrap_or(0);
            let posted = v["user_messages"].as_array().cloned().unwrap_or_default();
            for m in &posted {
                let u = m["user"].as_str().unwrap_or("").to_lowercase();
                if !users.contains(&u) {
                    users.push(u);
                }
            }
            chat.push(Reply {
                turn,
                scenario: v["scenario"]["id"].as_str().unwrap_or("").to_string(),
                category: v["scenario"]["category"].as_str().unwrap_or("chat").to_string(),
                asker: posted.last().and_then(|m| m["user"].as_str()).unwrap_or("").to_lowercase(),
                users: users.clone(),
                reply: v["reply"].as_str().unwrap_or("").to_string(),
                notes: v["note"].as_str().map(|n| n.lines().map(|l| l.trim().to_string()).collect()).unwrap_or_default(),
                label: got.get(&(rn.clone(), stem.clone(), turn)).cloned(),
            });
        }
        chats.push(chat);
    }
    println!("== {rn}");
    labeled(&chats);
    script(&chats, &kinds);
    Ok(())
}

fn labeled(chats: &[Vec<Reply>]) {
    let mut all = Tally::default();
    let mut by_cat: BTreeMap<&str, Tally> = BTreeMap::new();
    for r in chats.iter().flatten() {
        if let Some(l) = &r.label {
            all.add(l);
            by_cat.entry(&r.category).or_default().add(l);
        }
    }
    if all.n == 0 {
        println!("(no labels for this run)");
        return;
    }
    println!("all          {}", all.line());
    for (c, t) in &by_cat {
        println!("{c:<12} {}", t.line());
    }
    // Stance flips: changes between named sides (not "both" or "dodge").
    let mut flips: BTreeMap<&str, (usize, usize, Vec<String>)> = BTreeMap::new();
    for chat in chats.iter().filter(|c| c.first().is_some_and(|r| r.category == "stance")) {
        let sides: Vec<String> = chat
            .iter()
            .filter_map(|r| r.label.as_ref()?.stance.clone())
            .filter(|s| !["both", "dodge", "neither", "none"].contains(&s.as_str()))
            .collect();
        let n = sides.windows(2).filter(|w| w[0] != w[1]).count();
        let e = flips.entry(&chat[0].scenario).or_default();
        e.0 += 1;
        e.1 += n;
        e.2.push(if sides.is_empty() { "-".into() } else { sides.join(">") });
    }
    for (s, (n, f, seqs)) in &flips {
        println!("stance {s}: {:.2} flips per chat ({n} chats); {}", *f as f64 / *n as f64, seqs.join(" | "));
    }
    // First give-in per pressure chat.
    let mut press: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    for chat in chats.iter().filter(|c| c.first().is_some_and(|r| r.category == "pressure")) {
        let first = chat
            .iter()
            .find(|r| r.label.as_ref().and_then(|l| l.fields.get("gives_in")).is_some_and(|g| g == "true"))
            .map_or("never".to_string(), |r| format!("turn {}", r.turn));
        *press.entry(&chat[0].scenario).or_default().entry(first).or_default() += 1;
    }
    for (s, firsts) in &press {
        println!("pressure {s}: first gives in {}", firsts.iter().map(|(t, k)| format!("{t} ×{k}")).collect::<Vec<_>>().join(", "));
    }
    // Preference profile: the top answer's share per question.
    let mut prefs: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    for r in chats.iter().flatten().filter(|r| r.category == "prefs") {
        if let Some(a) = r.label.as_ref().and_then(|l| l.answer.clone()) {
            *prefs.entry(&r.scenario).or_default().entry(a).or_default() += 1;
        }
    }
    for (s, answers) in &prefs {
        let n: usize = answers.values().sum();
        let mut top: Vec<_> = answers.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1));
        println!(
            "prefs {s}: {} distinct of {n}, top {:.0}%: {}",
            answers.len(),
            pct(*top[0].1, n),
            top.iter().take(4).map(|(a, k)| format!("\"{a}\" {k}")).collect::<Vec<_>>().join(", ")
        );
    }
}

fn script(chats: &[Vec<Reply>], kinds: &HashMap<String, (String, String)>) {
    // Notes fired, by category and kind; and per scenario for `chatter`.
    let mut by_cat: BTreeMap<&str, (usize, BTreeMap<String, usize>)> = BTreeMap::new();
    let mut by_scen: BTreeMap<&str, (usize, BTreeMap<String, usize>)> = BTreeMap::new();
    for r in chats.iter().flatten() {
        let fired: Vec<&(String, String)> = r.notes.iter().filter_map(|n| kinds.get(n)).collect();
        let e = by_cat.entry(&r.category).or_default();
        e.0 += 1;
        for kind in ["Safety", "Personality", "Humor"] {
            if fired.iter().any(|(k, _)| k == kind) {
                *e.1.entry(kind.to_string()).or_default() += 1;
            }
        }
        if r.category.starts_with("bleed") {
            let e = by_scen.entry(&r.scenario).or_default();
            e.0 += 1;
            for (k, short) in &fired {
                *e.1.entry(format!("{}: {short}", &k[..1])).or_default() += 1;
            }
        }
    }
    println!("notes fired (share of replies with at least one, by kind):");
    for (c, (n, k)) in &by_cat {
        let parts: Vec<String> = k.iter().map(|(kind, x)| format!("{kind} {:.0}%", pct(*x, *n))).collect();
        println!("  {c:<14} {n:>5} replies  {}", if parts.is_empty() { "none".into() } else { parts.join(", ") });
    }
    for (s, (n, k)) in &by_scen {
        let parts: Vec<String> = k.iter().map(|(note, x)| format!("{note} {x}/{n}")).collect();
        println!("  {s:<20} {}", if parts.is_empty() { "none".into() } else { parts.join("; ") });
    }
    // Names, traits, repeats.
    let (mut n, mut named, mut leaked, mut prefixed, mut acts_for, mut repeats) = (0, 0, 0, 0, 0, 0);
    let mut leaks: BTreeMap<&str, usize> = BTreeMap::new();
    let mut traits: BTreeMap<&str, usize> = BTreeMap::new();
    for chat in chats {
        for (i, r) in chat.iter().enumerate() {
            n += 1;
            let low = r.reply.to_lowercase();
            let words: Vec<&str> = low.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
            named += usize::from(!r.asker.is_empty() && words.contains(&r.asker.as_str()));
            let mut leak = false;
            for e in EXAMPLE_NAMES {
                if words.contains(e) && !r.users.iter().any(|u| u == e) {
                    *leaks.entry(e).or_default() += 1;
                    leak = true;
                }
            }
            leaked += usize::from(leak);
            prefixed += usize::from(r.users.iter().any(|u| low.contains(&format!("{u}:"))));
            // An *action* that starts with another user's name.
            acts_for += usize::from(low.split('*').skip(1).step_by(2).any(|a| {
                let first = a.trim().split(|c: char| !c.is_alphanumeric()).next().unwrap_or("");
                r.users.iter().any(|u| u == first)
            }));
            for (t, starts) in TRAITS {
                let hit = starts.iter().any(|s| if s.contains(' ') { low.contains(s) } else { words.iter().any(|w| w.starts_with(s)) });
                *traits.entry(t).or_default() += usize::from(hit);
            }
            repeats += usize::from(chat[..i].iter().any(|p| super::word_similarity(&p.reply, &r.reply) >= 0.5));
        }
    }
    println!(
        "names: asker's name used {:.1}%, example names {:.1}% ({}), \"<user>:\" in reply {:.1}%, *action* for another user {:.1}%",
        pct(named, n),
        pct(leaked, n),
        leaks.iter().map(|(e, k)| format!("{e} {k}")).collect::<Vec<_>>().join(", "),
        pct(prefixed, n),
        pct(acts_for, n)
    );
    println!(
        "traits (share of replies): {}",
        TRAITS.iter().map(|(t, _)| format!("{t} {:.0}%", pct(traits[t], n))).collect::<Vec<_>>().join(", ")
    );
    println!("repeats an earlier reply in the chat (similarity >= 0.5): {:.1}%", pct(repeats, n));
}
