//! Guardrails for the hotdog rater (README.md, "Hotdog rater"):
//!
//! `persona_report hotdog persona <file.rsn>...` fails if a persona file
//! contains a rater anchor (`bench/persona/anchors.txt`) or something close to
//! one: anchors calibrate the raters and must never be what the bot is shown.
//!
//! `persona_report hotdog copies <run_dir>...` counts replies that copy (word
//! similarity >= 0.6, `persona::copy_ratio`) a quoted line of the run's own
//! `persona.rsn` (examples, moves, set-up lines) or an anchor.
//!
//! `persona_report hotdog owner <run_dir>...` is a rough count of replies that
//! talk as if the wrong person owns the hotdog (the `wrong_owner` tag, by
//! pattern), per whose it is.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use increbot::persona::copy_ratio;
use serde_json::Value;

const THRESHOLD: f64 = 0.6;

pub fn main(args: &[String]) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("persona") if args.len() > 1 => persona(&args[1..]),
        Some("copies") if args.len() > 1 => copies(&args[1..]),
        Some("owner") if args.len() > 1 => owner(&args[1..]),
        _ => bail!("usage: persona_report hotdog persona <file.rsn>... | copies <run_dir>... | owner <run_dir>..."),
    }
}

fn anchors() -> Result<Vec<String>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/persona/anchors.txt");
    Ok(std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

/// The double-quoted strings in a persona file at least `min_len` long; a
/// note's inner quotes ("for example: \"…\"") count separately.
fn quoted(text: &str, min_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut s = String::new();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    if let Some(n) = chars.next() {
                        s.push(n);
                    }
                }
                '"' => break,
                c => s.push(c),
            }
        }
        let inner: Vec<String> = s.split('"').skip(1).step_by(2).filter(|p| p.len() >= min_len).map(str::to_string).collect();
        if !inner.is_empty() {
            out.extend(inner);
        } else if s.len() >= min_len {
            out.push(s);
        }
    }
    out
}

fn persona(files: &[String]) -> Result<()> {
    let anchors = anchors()?;
    let mut bad = 0;
    for f in files {
        let lines = quoted(&std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?, 20);
        for a in &anchors {
            if let Some(hit) = lines.iter().find(|q| copy_ratio(q, a) >= THRESHOLD) {
                bad += 1;
                println!("{f}: contains a rater anchor: {a:?} ~ {hit:?}");
            }
        }
    }
    if bad > 0 {
        bail!("{bad} anchor(s) found");
    }
    println!("ok: no rater anchors");
    Ok(())
}

/// The replies of a one-shot run, with their scenario id and set-up message.
fn rows(run: &Path) -> Result<Vec<Value>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(run)
        .with_context(|| format!("reading {}", run.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("batch_") && n.ends_with(".jsonl")))
        .collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        for line in std::fs::read_to_string(&f)?.lines().filter(|l| !l.trim().is_empty()) {
            out.push(serde_json::from_str(line)?);
        }
    }
    Ok(out)
}

fn name(run: &str) -> &str {
    Path::new(run.trim_end_matches('/')).file_name().and_then(|n| n.to_str()).unwrap_or(run)
}

fn copies(runs: &[String]) -> Result<()> {
    let anchors = anchors()?;
    for run in runs {
        let dir = Path::new(run);
        let mut sources = std::fs::read_to_string(dir.join("persona.rsn")).map(|t| quoted(&t, 20)).unwrap_or_default();
        sources.extend(anchors.iter().cloned());
        let rows = rows(dir)?;
        let mut hits: Vec<(f64, String, String)> = rows
            .iter()
            .filter_map(|r| {
                let reply = r["reply"].as_str().unwrap_or_default();
                sources
                    .iter()
                    .map(|s| (copy_ratio(reply, s), s))
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .filter(|(score, _)| *score >= THRESHOLD)
                    .map(|(score, s)| (score, reply.to_string(), s.clone()))
            })
            .collect();
        println!(
            "{}: {}/{} replies copy a persona line or an anchor ({:.1}%)",
            name(run),
            hits.len(),
            rows.len(),
            100.0 * hits.len() as f64 / rows.len().max(1) as f64
        );
        hits.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (score, reply, src) in hits.iter().take(5) {
            let cut = |s: &str| s.chars().take(90).collect::<String>();
            println!("  {score:.2} {:?}\n       ~ {:?}", cut(reply), cut(src));
        }
    }
    Ok(())
}

const THINGS: &[&str] = &[
    "hotdog", "hot", "weenie", "wiener", "weeny", "sausage", "frank", "bratwurst", "footlong", "kielbasa",
];

/// The reply's words, lowercase, letters and apostrophes only.
fn words(t: &str) -> Vec<String> {
    t.to_lowercase()
        .split(|c: char| !(c.is_ascii_alphabetic() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(|w| w.trim_matches('\'').to_string())
        .collect()
}

/// Whether one of `owners` comes at most 4 words before a hotdog word.
fn owned_by(ws: &[String], owners: &[&str]) -> bool {
    ws.iter().enumerate().any(|(i, w)| {
        owners.contains(&w.as_str()) && ws[i + 1..].iter().take(4).any(|x| THINGS.contains(&x.as_str()))
    })
}

/// Whether `reply` claims the wrong owner. `who` is "me" (the requester's
/// own), "bot" (floob's own) or the other user's name. A pattern, not a label:
/// "mine is way bigger" counts too.
fn wrong_owner(reply: &str, who: &str) -> bool {
    let ws = words(reply);
    let mine = owned_by(&ws, &["my", "mine"]);
    let yours = owned_by(&ws, &["your", "ur", "yer", "yours"]);
    match who {
        // floob's own, called the requester's.
        "bot" => yours && !mine,
        // The requester's own, called floob's.
        "me" => mine,
        // Another user's, called floob's or the requester's.
        other => mine || (yours && !ws.iter().any(|w| *w == other.to_lowercase())),
    }
}

fn owner(runs: &[String]) -> Result<()> {
    for run in runs {
        let mut tally: std::collections::BTreeMap<&str, [usize; 2]> = Default::default();
        for r in rows(Path::new(run))? {
            let id = r["scenario"]["id"].as_str().unwrap_or_default();
            let setup = r["user_messages"][0]["text"].as_str().unwrap_or_default();
            let (kind, who) = if id.contains("-bot-") {
                ("bot", "bot".to_string())
            } else if id.contains("-me-") {
                ("me", "me".to_string())
            } else {
                // "measure Doru's hotdog" / "on Doru's hotdog": the word before "'s hotdog".
                let at = setup.find("'s hotdog").unwrap_or(0);
                let name = setup[..at].rsplit(|c: char| !c.is_alphanumeric() && c != '_').next().unwrap_or("");
                ("other", name.to_string())
            };
            let t = tally.entry(kind).or_default();
            t[0] += usize::from(wrong_owner(r["reply"].as_str().unwrap_or_default(), &who));
            t[1] += 1;
        }
        let (h, n) = tally.values().fold((0, 0), |(h, n), [a, b]| (h + a, n + b));
        let parts: Vec<String> = tally.iter().map(|(k, [a, b])| format!("{k} {a}/{b}")).collect();
        println!("{}: wrong owner (pattern) {h}/{n}: {}", name(run), parts.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_lines_and_inner_quotes() {
        let text = r#"x: "a short one", note: "(for example: \"this one is long enough to count\" or \"so is this second example\")""#;
        assert_eq!(quoted(text, 20), ["this one is long enough to count", "so is this second example"]);
    }

    #[test]
    fn wrong_owner_by_whose_it_is() {
        assert!(wrong_owner("u wanna measure my sausage?", "me"));
        assert!(!wrong_owner("ur sausage is a fine one!", "me"));
        assert!(wrong_owner("ur hotdog is so long!", "bot"));
        assert!(wrong_owner("oh, u want to measure my weenie?", "Doru"));
        assert!(!wrong_owner("doru, ur hotdog is impressive", "Doru"));
        assert!(wrong_owner("ur hot dog is impressive, thornback", "Doru"));
    }
}
