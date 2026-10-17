//! The terminal front end of `chatsim`: maps keys to [`chatsim_core::Action`]s,
//! draws the room with ratatui, and runs the event loop. All behaviour lives in
//! `chatsim-core`; this crate only presents it.

// `clap`, `increbot` and `tracing_subscriber` are real dependencies of this
// package, used by `src/main.rs` (CLI parsing, loading the model, logging
// setup) — just not by this lib target, which `unused_crate_dependencies`
// otherwise flags per-target.
#![allow(unused_crate_dependencies)]

pub mod keys;
pub mod run;
pub mod ui;
pub mod wrap;

pub use run::{CrosstermEvents, EventSource, ScriptedEvents, run};
