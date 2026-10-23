//! `increbot describe`: Qwen3-VL-8B describes a picture, or answers a question
//! about it, on one GPU (the vision tower on the CPU).

// Printing the answer is this command's whole job.
#![allow(clippy::print_stdout)]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::classify::image::{self, ImageLimits};
use crate::classify::vision::Rgb;
use crate::config::DescribeConfig;
use crate::describe::Describer;
use crate::hub;

#[derive(clap::Args)]
pub struct DescribeArgs {
    /// The picture (PNG, JPEG or WebP).
    image: PathBuf,
    /// A question about it; without one it is described.
    prompt: Option<String>,
    /// The GPU's PCI address, e.g. 0000:03:00.0.
    #[arg(long)]
    gpu: String,
    /// The answer's limit in tokens.
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    /// The picture's limit in tokens.
    #[arg(long, default_value_t = crate::describe::MAX_IMAGE_TOKENS)]
    max_image_tokens: usize,
    /// 0 picks the likeliest token each time.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
}

pub fn describe(args: DescribeArgs) -> Result<()> {
    let k = DescribeConfig {
        describe_max_new_tokens: args.max_new_tokens,
        describe_temperature: args.temperature,
        ..DescribeConfig::default()
    };
    let bytes = std::fs::read(&args.image).with_context(|| format!("reading {}", args.image.display()))?;
    let limits = ImageLimits { max_bytes: 64 << 20, max_pixels: 100_000_000, max_side: 2048 };
    let picture = image::load(&bytes, &limits).with_context(|| args.image.display().to_string())?;
    let picture = Rgb::from_image(&picture)?;

    let start = Instant::now();
    let dev = crate::gpu::Device::open(args.gpu.parse()?).context("--gpu")?;
    let model = Describer::open(
        &dev,
        &hub::resolve(&k.describe_repo, &k.describe_file)?,
        &hub::resolve(&k.describe_repo, &k.describe_mmproj)?,
        &hub::resolve(&k.describe_tokenizer_repo, "tokenizer.json")?,
        args.max_image_tokens,
        k.params(),
    )?;
    tracing::info!(elapsed_s = start.elapsed().as_secs_f32(), "loaded");
    let d = model.describe(&picture, args.prompt.as_deref().unwrap_or(""), &|| false)?;
    tracing::info!(
        grid = format!("{}×{}", d.grid.1, d.grid.0),
        image_tokens = d.image_tokens,
        prompt_tokens = d.prompt_tokens,
        vision_s = d.vision_secs,
        prefill_s = d.prefill_secs,
        tokens_per_s = d.generated as f32 / d.decode_secs.max(1e-3),
        truncated = d.truncated,
        "described"
    );
    println!("{}", d.text);
    Ok(())
}
