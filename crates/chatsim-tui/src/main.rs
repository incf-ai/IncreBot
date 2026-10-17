//! `chatsim`: a console chat room with increbot, for trying the bot's behaviour
//! and persona without Discord.

// Startup diagnostics printed directly, before the TUI takes over the
// terminal (ratatui's alternate screen hasn't started yet).
#![allow(clippy::print_stderr)]

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context as _, Result};
use chatsim_core::room::default_persona;
use chatsim_core::{
    App, Backend, EngineBackend, Pacing, ReplyMode, Room, RoomConfig, ScriptedBackend,
};
use chatsim_tui::{CrosstermEvents, run};
use clap::Parser;
use increbot::config::Config;

#[derive(Parser)]
#[command(
    name = "chatsim",
    about = "A console chat room with increbot: you (and anyone you /nick as) chat, the bot replies"
)]
struct Cli {
    /// Scripted replies instead of the model: no download, starts instantly.
    #[arg(long)]
    mock: bool,
    /// Milliseconds per word for --mock replies.
    #[arg(long, default_value_t = 60)]
    mock_delay_ms: u64,
    /// increbot config file (model, generation and persona settings). Missing
    /// means defaults; INCREBOT_<KEY> env vars override.
    #[arg(long, default_value = "config.rsn")]
    config: PathBuf,
    /// Your display name (change it later with /nick).
    #[arg(long, default_value = "Alice")]
    nick: String,
    /// Reply to every message, or only to @mentions.
    #[arg(long, default_value = "every")]
    mode: ReplyMode,
    /// Channel name shown in the header.
    #[arg(long, default_value = "general")]
    channel: String,
    /// Append logs to this file (the terminal belongs to the UI).
    #[arg(long)]
    log: Option<PathBuf>,
    /// Also run on the GPUs in `runtime.gpus` / `INCREBOT_GPUS` while no other
    /// program uses them (gpu-plan.md).
    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(path) = &cli.log {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file))
            .init();
    }

    let cfg = Config::load(&cli.config, &|k| std::env::var(k).ok())?;
    let persona = match increbot::persona::Persona::load(
        std::path::Path::new(&cfg.discord.persona_file),
        &|k| std::env::var(k).ok(),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: {e:#}; using the built-in persona ({})", increbot::persona::DEFAULT_PERSONA_FILE);
            default_persona()
        }
    };
    let room = Room::new(RoomConfig {
        mode: cli.mode,
        nick: cli.nick.clone(),
        channel_name: cli.channel.clone(),
        ..RoomConfig::from_config(&cfg, persona.clone())
    })?;

    if cli.mock {
        let backend =
            ScriptedBackend::mock(Pacing::Every(Duration::from_millis(cli.mock_delay_ms)));
        start(room, backend)
    } else {
        eprintln!("Loading the model (use --mock to try the UI without it)…");
        let backend = EngineBackend::load(&cfg, persona, cli.gpu, &mut |step| eprintln!("  {step}…"))?;
        start(room, backend)
    }
}

fn start<B: Backend>(room: Room, backend: B) -> Result<()> {
    let mut app = App::new(room, backend)?;
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &mut CrosstermEvents);
    ratatui::restore();
    result
}
