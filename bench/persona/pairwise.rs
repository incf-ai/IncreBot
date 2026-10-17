//! Blind pairwise ratings between two candidates' runs on the same one-shot
//! scenarios (README.md, "Hotdog rater"):
//!
//! `persona_report pairwise prep <out_dir> <run_a> <run_b> [--raters N] [--overlap K] [--seed S]`
//! `persona_report pairwise score <out_dir>`
//! `persona_report pairwise calibrate <out_dir> [--n 20]`
//! `persona_report pairwise winners <out_dir> [--top 20] [--exclude <scenarios.jsonl>]`
//!
//! - `prep` pairs the two runs' replies by scenario id and repeat, shuffles
//!   which run is A and which is B in each pair, deals the pairs out to N
//!   raters (K pairs go to every rater, to measure agreement), and writes
//!   `rater_N.txt` plus `key.json`. Raters write `rater_N.jsonl`: `{"pair": P,
//!   "winner": "A"|"B"|"tie", "funny": ["A", "B"], "reason": "..."}`; `funny`
//!   lists the replies that are funny on their own (any, both or none).
//! - `score` prints each run's wins, losses and ties, its win rate among
//!   length-matched pairs (within 20% of each other's length), how often the
//!   longer reply won the rest, its funny rate (majority of raters per
//!   reply), the overlap agreement, and, once `you.md` is filled in, how often
//!   the AI raters agree with you.
//! - `calibrate` writes `you.md`: N rated pairs for a person to judge blind
//!   (`winner:` A, B or tie, or both / neither; `funny:` A, B, both or none).
//! - `winners` lists the replies that won their pair, most decisive first, to
//!   curate a hall of fame; `--exclude` drops a held-out set's scenarios.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Value, json};

pub fn main(args: &[String]) -> Result<()> {
    let opts = |from: usize| -> HashMap<&str, &str> {
        args[from..].chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].as_str(), c[1].as_str())).collect()
    };
    let num = |o: &HashMap<&str, &str>, k: &str, d: usize| -> Result<usize> {
        o.get(k).map_or(Ok(d), |v| v.parse().with_context(|| format!("{k} {v}")))
    };
    match args.first().map(String::as_str) {
        Some("prep") if args.len() >= 4 => {
            let o = opts(4);
            prep(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                num(&o, "--raters", 3)?,
                num(&o, "--overlap", 6)?,
                num(&o, "--seed", 1)? as u64,
            )
        }
        Some("score") if args.len() == 2 => score(Path::new(&args[1])),
        Some("calibrate") if args.len() >= 2 => calibrate(Path::new(&args[1]), num(&opts(2), "--n", 20)?),
        Some("winners") if args.len() >= 2 => {
            let o = opts(2);
            winners(Path::new(&args[1]), num(&o, "--top", 20)?, o.get("--exclude").map(Path::new))
        }
        _ => bail!(
            "usage: persona_report pairwise prep <out_dir> <run_a> <run_b> [--raters N] [--overlap K] [--seed S] | score <out_dir> | calibrate <out_dir> [--n 20] | winners <out_dir> [--top 20] [--exclude <scenarios.jsonl>]"
        ),
    }
}

/// One reply of a one-shot run: the set-up line, the requester, the reply.
#[derive(Clone)]
struct Reply {
    setup: String,
    user: String,
    text: String,
}

/// `(scenario id, repeat)` → reply, for a one-shot run (one reply per batch).
fn replies(run: &Path) -> Result<HashMap<(String, u64), Reply>> {
    let mut out = HashMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(run)
        .with_context(|| format!("reading {}", run.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("batch_") && n.ends_with(".jsonl")))
        .collect();
    files.sort();
    for f in files {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| !l.trim().is_empty()) {
            let r: Value = serde_json::from_str(line)?;
            let id = r["scenario"]["id"].as_str().unwrap_or_default().to_string();
            let msg = &r["user_messages"][0];
            out.insert(
                (id, r["repeat"].as_u64().unwrap_or(0)),
                Reply {
                    setup: msg["text"].as_str().unwrap_or_default().to_string(),
                    user: msg["user"].as_str().unwrap_or_default().to_string(),
                    text: r["reply"].as_str().unwrap_or_default().to_string(),
                },
            );
        }
    }
    Ok(out)
}

/// A pair as `key.json` keeps it.
#[derive(Clone)]
struct Pair {
    scenario: String,
    repeat: u64,
    /// Run paths for sides A and B.
    a: String,
    b: String,
    setup: String,
    user: String,
    reply_a: String,
    reply_b: String,
}

impl Pair {
    fn run(&self, side: &str) -> &str {
        if side == "A" { &self.a } else { &self.b }
    }
    fn reply(&self, side: &str) -> &str {
        if side == "A" { &self.reply_a } else { &self.reply_b }
    }
}

fn prep(out: &Path, run_a: &Path, run_b: &Path, raters: usize, overlap: usize, seed: u64) -> Result<()> {
    let (a, b) = (replies(run_a)?, replies(run_b)?);
    let mut keys: Vec<(String, u64)> = a.keys().filter(|k| b.contains_key(*k)).cloned().collect();
    if keys.is_empty() {
        bail!("no scenario/repeat in common");
    }
    keys.sort();
    let mut rng = StdRng::seed_from_u64(seed);
    keys.shuffle(&mut rng);
    let mut pairs = Vec::new();
    for (n, k) in keys.iter().enumerate() {
        let flip = rng.random_bool(0.5);
        let ((ra, sa), (rb, sb)) = if flip { ((run_b, &b), (run_a, &a)) } else { ((run_a, &a), (run_b, &b)) };
        let first = &sa[k];
        pairs.push((
            (n + 1).to_string(),
            Pair {
                scenario: k.0.clone(),
                repeat: k.1,
                a: ra.display().to_string(),
                b: rb.display().to_string(),
                setup: first.setup.clone(),
                user: first.user.clone(),
                reply_a: first.text.clone(),
                reply_b: sb[k].text.clone(),
            },
        ));
    }
    let overlap = overlap.min(pairs.len());
    let (shared, rest) = pairs.split_at(overlap);
    std::fs::create_dir_all(out)?;
    for i in 0..raters {
        let mut mine: Vec<&(String, Pair)> = shared.iter().chain(rest.iter().skip(i).step_by(raters.max(1))).collect();
        mine.shuffle(&mut rng);
        let mut text = String::new();
        for (n, p) in mine {
            text += &format!(
                "=== PAIR {n} ===\n{} runs the command: {}\nA: {}\nB: {}\n\n",
                p.user, p.setup, p.reply_a, p.reply_b
            );
        }
        std::fs::write(out.join(format!("rater_{}.txt", i + 1)), text)?;
    }
    let mut key = serde_json::Map::new();
    for (n, p) in &pairs {
        key.insert(
            n.clone(),
            json!({
                "scenario": p.scenario, "repeat": p.repeat, "A": p.a, "B": p.b,
                "setup": p.setup, "user": p.user, "replies": { "A": p.reply_a, "B": p.reply_b },
            }),
        );
    }
    key.insert("overlap".into(), json!(shared.iter().map(|(n, _)| n.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>()));
    std::fs::write(out.join("key.json"), serde_json::to_string_pretty(&Value::Object(key))?)?;
    println!("{} pairs, {raters} raters, {overlap} shared", pairs.len());
    Ok(())
}

/// The ratings in `out`: pairs, the shared pairs, each pair's winners and
/// funny marks (one entry per rater).
struct Ratings {
    pairs: BTreeMap<String, Pair>,
    overlap: BTreeSet<String>,
    votes: HashMap<String, Vec<String>>,
    funny: HashMap<String, Vec<BTreeSet<String>>>,
}

fn load(out: &Path) -> Result<Ratings> {
    let key: serde_json::Map<String, Value> =
        serde_json::from_str(&std::fs::read_to_string(out.join("key.json")).context("key.json")?)?;
    let overlap = key
        .get("overlap")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|v| v.to_string()).collect())
        .unwrap_or_default();
    // Keys from before replies were stored (h01, h02): filled in from the runs.
    let mut cache: HashMap<String, HashMap<(String, u64), Reply>> = HashMap::new();
    let mut pairs = BTreeMap::new();
    for (n, v) in key.iter().filter(|(n, _)| *n != "overlap") {
        let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
        let (scenario, repeat) = (s("scenario"), v["repeat"].as_u64().unwrap_or(0));
        let mut p = Pair {
            scenario: scenario.clone(),
            repeat,
            a: s("A"),
            b: s("B"),
            setup: s("setup"),
            user: s("user"),
            reply_a: v["replies"]["A"].as_str().unwrap_or_default().to_string(),
            reply_b: v["replies"]["B"].as_str().unwrap_or_default().to_string(),
        };
        if v.get("replies").is_none() {
            for side in ["A", "B"] {
                let run = p.run(side).to_string();
                if !cache.contains_key(&run) {
                    cache.insert(run.clone(), replies(Path::new(&run))?);
                }
                if let Some(r) = cache[&run].get(&(scenario.clone(), repeat)) {
                    if side == "A" {
                        p.reply_a = r.text.clone();
                        p.setup = r.setup.clone();
                        p.user = r.user.clone();
                    } else {
                        p.reply_b = r.text.clone();
                    }
                }
            }
        }
        pairs.insert(n.clone(), p);
    }
    let mut votes: HashMap<String, Vec<String>> = HashMap::new();
    let mut funny: HashMap<String, Vec<BTreeSet<String>>> = HashMap::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(out)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("rater_") && n.ends_with(".jsonl")))
        .collect();
    files.sort();
    for f in files {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| !l.trim().is_empty()) {
            let v: Value = serde_json::from_str(line).with_context(|| format!("{}", f.display()))?;
            let pair = match &v["pair"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            votes.entry(pair.clone()).or_default().push(v["winner"].as_str().unwrap_or("tie").to_string());
            if let Some(marks) = v.get("funny").and_then(Value::as_array) {
                funny.entry(pair).or_default().push(marks.iter().filter_map(|m| m.as_str()).map(str::to_uppercase).collect());
            }
        }
    }
    Ok(Ratings { pairs, overlap, votes, funny })
}

/// The winner most raters picked; a split counts as a tie.
fn majority(ws: &[String]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for w in ws {
        *counts.entry(w.as_str()).or_default() += 1;
    }
    match counts.iter().max_by_key(|(_, n)| **n) {
        Some((w, n)) if n * 2 > ws.len() => w.to_string(),
        _ => "tie".into(),
    }
}

/// Whether most raters marked `side` of a pair funny.
fn funny_majority(marks: &[BTreeSet<String>], side: &str) -> bool {
    marks.iter().filter(|m| m.contains(side)).count() * 2 > marks.len()
}

/// A person's `winner:` per pair, and their funny marks ("both" and
/// "neither" as a winner also say both or neither was funny).
fn you_votes(out: &Path) -> (HashMap<String, String>, HashMap<String, BTreeSet<String>>) {
    let (mut wins, mut funny) = (HashMap::new(), HashMap::new());
    let Ok(text) = std::fs::read_to_string(out.join("you.md")) else {
        return (wins, funny);
    };
    let both = || BTreeSet::from(["A".to_string(), "B".to_string()]);
    let mut pair: Option<String> = None;
    for line in text.lines().map(str::trim) {
        if let Some(n) = line.strip_prefix("=== PAIR ").and_then(|r| r.strip_suffix(" ===")) {
            pair = Some(n.to_string());
        }
        let Some(p) = &pair else { continue };
        if let Some(w) = line.strip_prefix("winner:").map(|w| w.trim().to_lowercase()) {
            match w.split_whitespace().next().unwrap_or("") {
                "a" | "b" => {
                    wins.insert(p.clone(), w[..1].to_uppercase());
                }
                "tie" => {
                    wins.insert(p.clone(), "tie".into());
                }
                "both" => {
                    wins.insert(p.clone(), "tie".into());
                    funny.insert(p.clone(), both());
                }
                "neither" => {
                    wins.insert(p.clone(), "tie".into());
                    funny.insert(p.clone(), BTreeSet::new());
                }
                _ => {}
            }
        }
        if let Some(f) = line.strip_prefix("funny:").map(|f| f.trim().to_lowercase()) {
            let marks = match f.split_whitespace().next().unwrap_or("") {
                "both" => Some(both()),
                "none" | "neither" => Some(BTreeSet::new()),
                "a" | "b" => Some(BTreeSet::from([f[..1].to_uppercase()])),
                _ => None,
            };
            if let Some(m) = marks {
                funny.insert(p.clone(), m);
            }
        }
    }
    (wins, funny)
}

fn base(run: &str) -> &str {
    Path::new(run).file_name().and_then(|n| n.to_str()).unwrap_or(run)
}

fn score(out: &Path) -> Result<()> {
    let r = load(out)?;
    let runs: BTreeSet<&str> = r.pairs.values().flat_map(|p| [p.a.as_str(), p.b.as_str()]).collect();
    let mut tally: BTreeMap<&str, [usize; 3]> = runs.iter().map(|&x| (x, [0; 3])).collect(); // win, loss, tie
    let mut matched: BTreeMap<&str, [usize; 2]> = runs.iter().map(|&x| (x, [0; 2])).collect(); // wins, decided
    let mut longer_won = [0usize; 2];
    for (pair, ws) in &r.votes {
        let Some(p) = r.pairs.get(pair) else { continue };
        let w = majority(ws);
        if w == "tie" {
            tally.get_mut(p.a.as_str()).unwrap()[2] += 1;
            tally.get_mut(p.b.as_str()).unwrap()[2] += 1;
            continue;
        }
        let lose = if w == "A" { "B" } else { "A" };
        tally.get_mut(p.run(&w)).unwrap()[0] += 1;
        tally.get_mut(p.run(lose)).unwrap()[1] += 1;
        let (la, lb) = (p.reply_a.chars().count(), p.reply_b.chars().count());
        if la == 0 || lb == 0 {
            continue;
        }
        if la.abs_diff(lb) as f64 <= 0.2 * la.max(lb) as f64 {
            matched.get_mut(p.run(&w)).unwrap()[0] += 1;
            matched.get_mut(p.run(&w)).unwrap()[1] += 1;
            matched.get_mut(p.run(lose)).unwrap()[1] += 1;
        } else {
            longer_won[1] += 1;
            longer_won[0] += usize::from((la > lb) == (w == "A"));
        }
    }
    for (run, [w, l, t]) in &tally {
        let n = w + l + t;
        let rate = if n > 0 { (*w as f64 + *t as f64 / 2.0) / n as f64 } else { 0.0 };
        let [mw, md] = matched[run];
        let m = if md > 0 { format!("{:.0}% of {md}", 100.0 * mw as f64 / md as f64) } else { "-".into() };
        println!("{}: {w} wins, {l} losses, {t} ties of {n}; win rate {:.0}%; length-matched {m}", base(run), 100.0 * rate);
    }
    if longer_won[1] > 0 {
        println!("the longer reply won {}/{} pairs whose lengths differ by more than 20%", longer_won[0], longer_won[1]);
    }
    if !r.funny.is_empty() {
        let mut hits: BTreeMap<&str, [usize; 2]> = runs.iter().map(|&x| (x, [0; 2])).collect();
        for (pair, marks) in &r.funny {
            let Some(p) = r.pairs.get(pair) else { continue };
            for side in ["A", "B"] {
                let h = hits.get_mut(p.run(side)).unwrap();
                h[0] += usize::from(funny_majority(marks, side));
                h[1] += 1;
            }
        }
        for (run, [f, n]) in hits {
            if n > 0 {
                println!("{}: funny {f}/{n} ({:.0}%)", base(run), 100.0 * f as f64 / n as f64);
            }
        }
    }
    let agree: Vec<bool> = r
        .overlap
        .iter()
        .filter_map(|p| r.votes.get(p).filter(|v| v.len() > 1))
        .map(|v| v.iter().all(|w| *w == v[0]))
        .collect();
    println!("shared pairs fully agreed: {}/{}", agree.iter().filter(|a| **a).count(), agree.len());
    let (you, you_funny) = you_votes(out);
    if !you.is_empty() {
        let same = you.iter().filter(|(p, w)| r.votes.get(*p).is_some_and(|v| majority(v) == **w)).count();
        let decided: Vec<_> = you.iter().filter(|(_, w)| *w != "tie").collect();
        let same_d = decided.iter().filter(|(p, w)| r.votes.get(*p).is_some_and(|v| majority(v) == **w)).count();
        println!(
            "AI raters agree with you on {same}/{} calibration pairs; on the {} you decided: {same_d}/{} ({} you called a tie)",
            you.len(),
            decided.len(),
            decided.len(),
            you.len() - decided.len()
        );
        let mut picks: BTreeMap<&str, usize> = runs.iter().map(|&x| (x, 0)).collect();
        for (p, w) in &decided {
            if let Some(pair) = r.pairs.get(*p) {
                *picks.get_mut(pair.run(w)).unwrap() += 1;
            }
        }
        println!("your picks: {}", picks.iter().map(|(r, n)| format!("{} {n}", base(r))).collect::<Vec<_>>().join(", "));
        if !you_funny.is_empty() {
            let mut yr: BTreeMap<&str, [usize; 2]> = runs.iter().map(|&x| (x, [0; 2])).collect();
            for (p, marks) in &you_funny {
                let Some(pair) = r.pairs.get(p) else { continue };
                for side in ["A", "B"] {
                    let y = yr.get_mut(pair.run(side)).unwrap();
                    y[0] += usize::from(marks.contains(side));
                    y[1] += 1;
                }
            }
            println!(
                "your funny marks: {}",
                yr.iter().filter(|(_, [_, n])| *n > 0).map(|(r, [f, n])| format!("{} {f}/{n}", base(r))).collect::<Vec<_>>().join(", ")
            );
            let both: Vec<_> = you_funny.iter().filter(|(p, _)| r.funny.contains_key(*p)).collect();
            if !both.is_empty() {
                let same = both
                    .iter()
                    .flat_map(|(p, m)| ["A", "B"].map(|s| m.contains(s) == funny_majority(&r.funny[*p], s)))
                    .filter(|x| *x)
                    .count();
                println!("AI raters agree with your funny marks on {same}/{} replies", 2 * both.len());
            }
        }
    }
    Ok(())
}

fn calibrate(out: &Path, n: usize) -> Result<()> {
    let r = load(out)?;
    let mut rated: Vec<&String> = r.votes.keys().collect();
    rated.sort_by_key(|p| p.parse::<u64>().unwrap_or(0));
    rated.shuffle(&mut StdRng::seed_from_u64(7));
    let mut chosen: Vec<&String> = rated.into_iter().take(n).collect();
    chosen.sort_by_key(|p| p.parse::<u64>().unwrap_or(0));
    let mut text = String::from(
        "# Calibration: judge each pair blind by the hotdog-rater rules (README).\n\
         # After `winner:` write A, B or tie; after `funny:` write which replies are funny on\n\
         # their own: A, B, both or none. Then run `persona_report pairwise score` on this folder.\n\n",
    );
    for p in &chosen {
        let k = &r.pairs[*p];
        text += &format!(
            "=== PAIR {p} ===\n{} runs the command: {}\nA: {}\nB: {}\nwinner: \nfunny: \n\n",
            k.user, k.setup, k.reply_a, k.reply_b
        );
    }
    std::fs::write(out.join("you.md"), text)?;
    println!("wrote {} ({} pairs)", out.join("you.md").display(), chosen.len());
    Ok(())
}

fn winners(out: &Path, top: usize, exclude: Option<&Path>) -> Result<()> {
    let r = load(out)?;
    let skip: BTreeSet<String> = match exclude {
        Some(f) => std::fs::read_to_string(f)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter_map(|v| v["id"].as_str().map(str::to_string))
            .collect(),
        None => BTreeSet::new(),
    };
    let mut rows: Vec<(f64, usize, &str, &str, &str)> = r
        .votes
        .iter()
        .filter_map(|(pair, ws)| {
            let w = majority(ws);
            let p = r.pairs.get(pair)?;
            (w != "tie" && !skip.contains(&p.scenario)).then(|| {
                let share = ws.iter().filter(|x| **x == w).count() as f64 / ws.len() as f64;
                (share, ws.len(), p.scenario.as_str(), base(p.run(&w)), p.reply(&w))
            })
        })
        .collect();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    for (share, n, scen, run, reply) in rows.into_iter().take(top) {
        println!("{:.0}% of {n} | {scen} | {run} | {reply}", 100.0 * share);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_needs_more_than_half() {
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(majority(&v(&["A", "A", "B"])), "A");
        assert_eq!(majority(&v(&["A", "B"])), "tie");
        assert_eq!(majority(&v(&["B"])), "B");
    }
}
