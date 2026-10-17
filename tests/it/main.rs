//! Integration tests, one binary: each `tests/*.rs` file would be its own binary,
//! each linking burn, tokenizers and the rest, and Cargo runs them one at a time.

// Test-only diagnostics across every module below (skip reasons, measured
// results). This is the crate root for the whole `it` test binary, so one
// attribute here covers all of them.
#![allow(clippy::print_stderr)]

mod engine;
mod gpu;
mod gpu_sharing;
mod kquant_decode;
mod reference;
mod smoke_13b;
mod tokenizer;

/// The GPU tests (`gpu`, `gpu_sharing`) share the cards; run concurrently, each
/// would see the other's VRAM as another program's.
pub static CARDS: std::sync::Mutex<()> = std::sync::Mutex::new(());
