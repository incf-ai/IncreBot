//! The quality set's report (README.md §2, "Quality set"):
//!
//! `persona_report quality <labels_dir> <run_dir>...` reads the labelers'
//! `agent_N.jsonl` (the first label of an overlap batch counts) with
//! `key.json`, and the runs' batches, and prints per run, then per category:
//! replies, mean `score`, `funny` rate and the quality tags' rates. Then the
//! script measures, no labels needed: words per reply, more than one
//! `*action*`, distinct openers (first two words), the commonest stock phrases
//! and openers, and phrase-ban rollbacks.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

const TAGS: &[&str] = &[
    "under_reacts",
    "over_reacts",
    "contradicts",
    "pushover",
    "never_gives_in",
    "non_sequitur",
    "ignores_message",
    "flat",
    "wrong_addressee",
    "breaks_character",
    "not_kobold",
];

/// Phrases he leans on; counted per reply (README round 26's log).
const STOCK: &[&str] = &[
    "u know",
    "you know",
    "eyes suspiciously",
    "puffs up",
    "hehe",
    "hmm",
    "uhh",
    "um",
    "if u say so",
    "if you say so",
    "i dunno",
    "shiny",
    "beetle",
    "tunnel",
    "trap",
];

pub fn main(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("usage: persona_report quality <labels_dir> <run_dir>...");
    }
    let labels = Path::new(&args[0]);
    let key: Value = serde_json::from_str(
        &std::fs::read_to_string(labels.join("key.json")).with_context(|| format!("reading {}/key.json", labels.display()))?,
    )?;
    // (run dir name, batch stem, turn) → (score, funny, tags)
    let mut got: HashMap<(String, String, u64), (Option<f64>, Option<bool>, Vec<String>)> = HashMap::new();
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
            let run = name(k["run"].as_str().unwrap_or(""));
            let id = (run, k["batch"].as_str().unwrap_or("").to_string(), v["turn"].as_u64().unwrap_or(0));
            let tags = v["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default();
            got.entry(id).or_insert((v["score"].as_f64(), v["funny"].as_bool(), tags));
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

#[derive(Default)]
struct Tally {
    n: usize,
    scored: usize,
    score: f64,
    funny: usize,
    rated: usize,
    tags: BTreeMap<&'static str, usize>,
}

impl Tally {
    fn add(&mut self, score: Option<f64>, funny: Option<bool>, tags: &[String]) {
        self.n += 1;
        if let Some(s) = score {
            self.scored += 1;
            self.score += s;
        }
        if let Some(f) = funny {
            self.rated += 1;
            self.funny += usize::from(f);
        }
        for t in TAGS {
            if tags.iter().any(|x| x == t) {
                *self.tags.entry(t).or_default() += 1;
            }
        }
    }

    fn line(&self) -> String {
        let pct = |k: usize, n: usize| if n == 0 { 0.0 } else { 100.0 * k as f64 / n as f64 };
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
        s
    }
}

fn report(run: &Path, got: &HashMap<(String, String, u64), (Option<f64>, Option<bool>, Vec<String>)>) -> Result<()> {
    let rn = name(&run.to_string_lossy());
    let mut all = Tally::default();
    let mut by_cat: BTreeMap<String, Tally> = BTreeMap::new();
    let (mut words, mut multi, mut replies) = (0usize, 0usize, 0usize);
    let mut openers: BTreeMap<String, usize> = BTreeMap::new();
    let mut stock: BTreeMap<&str, usize> = BTreeMap::new();
    let mut rollbacks: BTreeMap<String, usize> = BTreeMap::new();
    let mut paths: Vec<_> = std::fs::read_dir(run)
        .with_context(|| format!("reading {}", run.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("batch_")) && p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    paths.sort();
    for p in paths {
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        for line in std::fs::read_to_string(&p)?.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line)?;
            let reply = v["reply"].as_str().unwrap_or("").to_lowercase();
            let cat = v["scenario"]["category"].as_str().unwrap_or("chat").to_string();
            let turn = v["turn"].as_u64().unwrap_or(0);
            if let Some((score, funny, tags)) = got.get(&(rn.clone(), stem.clone(), turn)) {
                all.add(*score, *funny, tags);
                by_cat.entry(cat).or_default().add(*score, *funny, tags);
            }
            replies += 1;
            let w: Vec<&str> = reply.split_whitespace().collect();
            words += w.len();
            multi += usize::from(reply.matches('*').count() / 2 > 1);
            *openers.entry(w.iter().take(2).copied().collect::<Vec<_>>().join(" ")).or_default() += 1;
            // Whole words: "um" isn't in "drum".
            let padded = format!(" {} ", reply.chars().map(|c| if c.is_alphanumeric() { c } else { ' ' }).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" "));
            for s in STOCK {
                if padded.contains(&format!(" {s} ")) || padded.contains(&format!(" {s}s ")) {
                    *stock.entry(s).or_default() += 1;
                }
            }
            for r in v["rollbacks"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                *rollbacks.entry(r.to_lowercase()).or_default() += 1;
            }
        }
    }
    println!("== {rn}");
    if all.n > 0 {
        println!("all        {}", all.line());
        for (c, t) in &by_cat {
            println!("{c:<10} {}", t.line());
        }
    } else {
        println!("(no labels for this run)");
    }
    let pct = |k: usize| 100.0 * k as f64 / replies.max(1) as f64;
    println!(
        "script: {replies} replies, {:.1} words each, more than one *action* {:.1}%, distinct openers {:.1}%",
        words as f64 / replies.max(1) as f64,
        pct(multi),
        pct(openers.len())
    );
    let mut top: Vec<_> = stock.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "stock phrases (share of replies): {}",
        top.iter().map(|(s, k)| format!("\"{s}\" {:.0}%", pct(*k))).collect::<Vec<_>>().join(", ")
    );
    let mut op: Vec<_> = openers.into_iter().collect();
    op.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "top openers: {}",
        op.iter().take(8).map(|(s, k)| format!("\"{s}\" {k}")).collect::<Vec<_>>().join(", ")
    );
    let total: usize = rollbacks.values().sum();
    let mut rb: Vec<_> = rollbacks.into_iter().collect();
    rb.sort_by(|a, b| b.1.cmp(&a.1));
    println!(
        "ban rollbacks: {total} ({})",
        rb.iter().take(10).map(|(s, k)| format!("\"{s}\" {k}")).collect::<Vec<_>>().join(", ")
    );
    Ok(())
}
