//! The `increbot` command line. The binary (`src/bin/increbot.rs`) only parses
//! the arguments and calls [`run`]; the commands live here.

mod classify;
mod generate;
mod inspect;
mod serve;

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use crate::hub;
use crate::model::LinearImpl;

pub use classify::ClassifyArgs;
pub use generate::GenerateArgs;

const DEFAULT_REPO: &str = "TheBloke/LLaMA2-13B-Tiefighter-GGUF";
const DEFAULT_FILE: &str = "llama2-13b-tiefighter.Q4_K_M.gguf";
const TOKENIZER_REPO: &str = "hf-internal-testing/llama-tokenizer";

#[derive(Parser)]
#[command(
    name = "increbot",
    about = "increbot: LLaMA2-13B-Tiefighter on CPU with Burn",
    version = crate::VERSION
)]
pub struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the Discord bot (design.md §6). Reads DISCORD_TOKEN from the environment.
    Serve {
        /// Config file; missing means defaults. Env vars INCREBOT_<KEY> override.
        #[arg(long, default_value = "config.rsn")]
        config: PathBuf,
    },
    /// Print a GGUF file's metadata and tensor table.
    Inspect {
        #[command(flatten)]
        model: ModelArgs,
        /// List every tensor, not just the first 12.
        #[arg(long)]
        all: bool,
    },
    /// Generate a continuation of a prompt.
    Generate(GenerateArgs),
    /// Answer System One questions about a state with a d1 model.
    Classify(ClassifyArgs),
}

/// Which GGUF file to use.
#[derive(clap::Args)]
pub struct ModelArgs {
    /// Local GGUF path. Overrides --repo/--file.
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_REPO)]
    repo: String,
    #[arg(long, default_value = DEFAULT_FILE)]
    file: String,
}

impl ModelArgs {
    fn path(&self) -> Result<PathBuf> {
        match &self.model {
            Some(p) => Ok(p.clone()),
            None => hub::resolve(&self.repo, &self.file),
        }
    }
}

/// Parses a linear layer implementation name (`--linear`, `model.linear_impl`).
fn linear_impl(name: &str) -> Result<LinearImpl> {
    Ok(match name {
        "packed" => LinearImpl::Packed,
        "dense" => LinearImpl::Dense,
        "burn" => LinearImpl::Burn,
        other => bail!("unknown linear impl {other:?} (packed | burn | dense)"),
    })
}

/// Sets up logging (`RUST_LOG`, default `info`) and runs the command.
///
/// Logs go to stderr, except that `serve` sends only warnings and errors there and
/// the rest to stdout: podman files container stderr in the journal as errors, so
/// every line would show red. The other commands print their results on stdout.
pub fn run(cli: Cli) -> Result<()> {
    use tracing_subscriber::fmt::writer::{BoxMakeWriter, MakeWriterExt};
    let writer = match cli.cmd {
        Cmd::Serve { .. } => BoxMakeWriter::new(
            std::io::stderr.with_max_level(tracing::Level::WARN).or_else(std::io::stdout),
        ),
        _ => BoxMakeWriter::new(std::io::stderr),
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(writer)
        .init();
    match cli.cmd {
        Cmd::Serve { config } => serve::serve(&config),
        Cmd::Inspect { model, all } => inspect::inspect(&model, all),
        Cmd::Generate(args) => generate::generate(args),
        Cmd::Classify(args) => classify::classify(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_each_command() {
        let cli = Cli::try_parse_from(["increbot", "serve", "--config", "x.rsn"]).unwrap();
        assert!(
            matches!(cli.cmd, Cmd::Serve { config } if config.as_path() == std::path::Path::new("x.rsn"))
        );
        let cli = Cli::try_parse_from(["increbot", "inspect", "--model", "m.gguf"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Inspect { .. }));
        let cli = Cli::try_parse_from(["increbot", "generate", "hi", "--stop", "\n"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Generate(_)));
        let cli = Cli::try_parse_from(["increbot", "classify", "{}", "--state", "hi"]).unwrap();
        assert!(matches!(cli.cmd, Cmd::Classify(_)));
    }

    #[test]
    fn linear_impl_names() {
        assert_eq!(linear_impl("packed").unwrap(), LinearImpl::Packed);
        assert_eq!(linear_impl("dense").unwrap(), LinearImpl::Dense);
        assert_eq!(linear_impl("burn").unwrap(), LinearImpl::Burn);
        assert!(linear_impl("fast").is_err());
    }

    #[test]
    fn version_includes_commit_and_build_time() {
        let v = crate::VERSION;
        assert!(
            v.contains(&format!("(commit {}, built ", crate::GIT_COMMIT)),
            "{v}"
        );
        assert!(v.ends_with(&format!("{})", crate::BUILD_TIMESTAMP)), "{v}");
        assert!(!crate::GIT_COMMIT.is_empty());
    }
}
