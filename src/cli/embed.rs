//! `increbot embed`: embeds up to a few pictures and texts with EmbeddingGemma 2
//! and prints their cosine similarities as `/embed-compare` shows them.

// Printing the table is this command's whole job.
#![allow(clippy::print_stdout)]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::classify::image::{self, ImageLimits};
use crate::classify::vision::Rgb;
use crate::embed::{Embedder, matrix};

#[derive(clap::Args)]
pub struct EmbedArgs {
    /// A picture (PNG, JPEG or WebP); repeatable.
    #[arg(long)]
    image: Vec<PathBuf>,
    /// A text; repeatable.
    #[arg(long)]
    text: Vec<String>,
    /// Pictures are scaled down first so their longer side is at most this.
    #[arg(long, default_value_t = 1536)]
    max_side: u32,
}

pub fn embed(args: EmbedArgs) -> Result<()> {
    if args.image.len() + args.text.len() < 2 {
        bail!("give at least two of --image and --text");
    }
    let limits = ImageLimits { max_bytes: 64 << 20, max_pixels: 100_000_000, max_side: args.max_side };
    let pictures = args
        .image
        .iter()
        .map(|p| {
            let bytes = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            let picture = image::load(&bytes, &limits).with_context(|| p.display().to_string())?;
            Ok(Rgb::from_image(&picture)?)
        })
        .collect::<Result<Vec<_>>>()?;

    let start = Instant::now();
    let model = Embedder::load()?;
    tracing::info!(elapsed_s = start.elapsed().as_secs_f32(), "loaded");
    let mut images = Vec::new();
    for (p, picture) in args.image.iter().zip(&pictures) {
        let start = Instant::now();
        images.push(model.embed_image(picture)?);
        tracing::info!(
            picture = %p.display(),
            size = format!("{}×{}", picture.width(), picture.height()),
            elapsed_s = start.elapsed().as_secs_f32(),
            "embedded"
        );
    }
    let mut texts = Vec::new();
    for t in &args.text {
        let start = Instant::now();
        texts.push(model.embed_text(t)?);
        tracing::info!(
            tokens = model.tokens(t)?.len(),
            elapsed_s = start.elapsed().as_secs_f32(),
            "embedded a text"
        );
    }
    if let Some(table) = matrix::table(&images, &texts) {
        println!("{table}");
    }
    Ok(())
}
