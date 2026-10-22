//! `increbot inspect`: a GGUF file's metadata and tensor table.

// This command's whole job is printing a report to the terminal (cli/mod.rs:
// non-`serve` commands print on stdout).
#![allow(clippy::print_stdout)]

use anyhow::Result;

use super::ModelArgs;
use crate::gguf::Gguf;

pub fn inspect(model: &ModelArgs, all: bool) -> Result<()> {
    let g = Gguf::open(&model.path()?)?;
    println!(
        "GGUF v{}, {} tensors, {} metadata keys",
        g.version,
        g.tensors.len(),
        g.metadata.len()
    );
    for (k, v) in &g.metadata {
        match v.as_array() {
            Some(a) => println!("  {k}: [{} items]", a.len()),
            None => println!("  {k}: {v:?}"),
        }
    }
    let mut counts = std::collections::BTreeMap::new();
    for t in &g.tensors {
        *counts.entry(format!("{:?}", t.ty)).or_insert(0) += 1;
    }
    println!("tensor types: {counts:?}");
    for t in g.tensors.iter().take(if all { usize::MAX } else { 12 }) {
        println!("  {} {:?} {:?}", t.name, t.ty, t.dims);
    }
    Ok(())
}
