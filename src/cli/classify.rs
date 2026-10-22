//! `increbot classify`: answers System One questions about a state with a d1
//! model and prints one JSON line of answers.

// Printing the answers is this command's whole job.
#![allow(clippy::print_stdout)]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{Map, Value, json};

use super::linear_impl;
use crate::classify::{self, Classifier, Json};
use crate::hub;

#[derive(clap::Args)]
pub struct ClassifyArgs {
    /// Questions, `{"name": {"type": "noul" | "choice" | "score", "instructions": ...,
    /// "criteria": ...}}`, or @file.
    questions: String,
    /// The state: JSON (an object, list, ...), or plain text; @file reads a file.
    #[arg(long)]
    state: Option<String>,
    /// Read `--state` as plain text even if it parses as JSON.
    #[arg(long)]
    text: bool,
    /// Local d1 GGUF path. Overrides --repo/--file.
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = classify::DEFAULT_REPO)]
    repo: String,
    #[arg(long, default_value = classify::DEFAULT_FILE)]
    file: String,
    /// Local tokenizer.json. Defaults to the one in LiquidAI/d1-3B.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Linear layer implementation: packed | burn | dense.
    #[arg(long, default_value = "packed")]
    linear: String,
    /// Context cap, tokens.
    #[arg(long, default_value_t = 8192)]
    max_ctx: usize,
    /// Run on the GPU at this PCI address (e.g. 0000:03:00.0).
    #[arg(long)]
    gpu: Option<String>,
}

fn read_arg(s: &str) -> Result<String> {
    match s.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {path}")),
        None => Ok(s.to_owned()),
    }
}

pub fn classify(args: ClassifyArgs) -> Result<()> {
    let model = match &args.model {
        Some(p) => p.clone(),
        None => hub::resolve(&args.repo, &args.file)?,
    };
    let tokenizer = match &args.tokenizer {
        Some(p) => p.clone(),
        None => hub::resolve(classify::TOKENIZER_REPO, "tokenizer.json")?,
    };
    let questions = classify::parse_questions(&read_arg(&args.questions)?)?;
    let state = match &args.state {
        None => None,
        Some(s) => {
            let s = read_arg(s)?;
            Some(match Json::parse(&s) {
                Ok(j) if !args.text => j,
                _ => Json::String(s),
            })
        }
    };

    let mut classifier = Classifier::load(&model, &tokenizer, linear_impl(&args.linear)?, args.max_ctx)?;
    if let Some(pci) = &args.gpu {
        classifier = classifier.with_gpu(pci.parse()?)?;
    }
    let start = Instant::now();
    let decisions = classifier.classify(state.as_ref(), &questions)?;
    let mut out = Map::new();
    for d in decisions {
        out.insert(d.name, json!({ "answer": d.answer, "probabilities": d.probabilities }));
    }
    tracing::info!(elapsed_s = start.elapsed().as_secs_f32(), "classified");
    println!("{}", Value::Object(out));
    Ok(())
}
