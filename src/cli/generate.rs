//! `increbot generate`: continue a prompt from the command line.

// This command's whole job is printing its result to the terminal
// (cli/mod.rs: non-`serve` commands print on stdout).
#![allow(clippy::print_stdout)]

use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use tracing::info;

use super::{ModelArgs, TOKENIZER_REPO, linear_impl};
use crate::engine::sampler::{Sampler, SamplerParams};
use crate::engine::{GenerateRequest, generate as run_generate};
use crate::gguf::Gguf;
use crate::model::Llama;
use crate::tokenizer::Tokenizer;
use crate::{hub, runtime};

#[derive(clap::Args)]
pub struct GenerateArgs {
    #[command(flatten)]
    model: ModelArgs,
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, env = "INCREBOT_THREADS", default_value_t = 8)]
    threads: usize,
    #[arg(long, env = "INCREBOT_PIN_THREADS", default_value_t = true, action = clap::ArgAction::Set)]
    pin_threads: bool,
    /// Stop strings (repeatable).
    #[arg(long = "stop")]
    stops: Vec<String>,
    /// Print token ids instead of streaming text.
    #[arg(long)]
    ids: bool,
    /// Linear layer implementation: packed, burn (reference, very slow) or dense
    /// (small models only).
    #[arg(long, default_value = "packed")]
    linear: String,
}

pub fn generate(args: GenerateArgs) -> Result<()> {
    let linear_impl = linear_impl(&args.linear)?;
    runtime::init_pool(args.threads, args.pin_threads)?;
    let gguf = Gguf::open(&args.model.path()?)?;
    let tok = Tokenizer::load(&hub::resolve(TOKENIZER_REPO, "tokenizer.json")?, &gguf)?;
    let llama = Llama::load(&gguf, &Default::default(), linear_impl)?;
    drop(gguf); // weights now live in anonymous memory (design.md §5)

    let prompt_ids = tok.encode_prompt(&args.prompt)?;
    let params = if args.temperature > 0.0 {
        SamplerParams {
            temperature: args.temperature,
            ..SamplerParams::default()
        }
    } else {
        SamplerParams::greedy()
    };
    let seed = args.seed.unwrap_or_else(rand::random);
    info!(
        seed,
        threads = args.threads,
        prompt_tokens = prompt_ids.len(),
        ?params,
        "generating"
    );
    let mut sampler = Sampler::new(params, seed);
    let mut cache = llama.new_cache();
    let req = GenerateRequest {
        prompt: &prompt_ids,
        max_new_tokens: args.max_new_tokens,
        max_new_tokens_gpu: args.max_new_tokens,
        wrap_up_after: 0,
        wrap_up_after_gpu: 0,
        stops: args.stops,
        max_lines: None,
        labels: false,
        timeout: None,
        history: &prompt_ids,
        cancel: None,
        bans: None,
    };
    let ids = args.ids;
    let mut out = std::io::stdout();
    let res = run_generate(&llama, &tok, &mut cache, &mut sampler, req, |t| {
        if !ids {
            let _ = out.write_all(t.as_bytes());
            let _ = out.flush();
        }
    })?;
    if ids {
        println!("{:?}", res.tokens);
    } else {
        println!();
    }
    let per = |d: Duration, n: usize| n as f64 / d.as_secs_f64().max(1e-9);
    info!(
        stop = ?res.stop,
        prefill_tokens = res.prefill_tokens,
        prefill_tok_s = per(res.prefill, res.prefill_tokens),
        generated = res.tokens.len(),
        decode_tok_s = per(res.decode, res.tokens.len().saturating_sub(1)),
        "done"
    );
    Ok(())
}
