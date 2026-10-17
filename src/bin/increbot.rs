//! The `increbot` command: `serve` (the Discord bot), `generate`, `inspect`.
//! Everything lives in [`increbot::cli`]; see `increbot --help`.

use clap::Parser;
use increbot::cli::{self, Cli};

fn main() -> anyhow::Result<()> {
    cli::run(Cli::parse())
}
