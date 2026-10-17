//! Model file resolution through the Hugging Face cache (design.md §5).

use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::warn;

/// Returns the local path of `repo/file`, from the HF cache (`HF_HOME`) if present,
/// otherwise downloading it.
pub fn resolve(repo: &str, file: &str) -> Result<PathBuf> {
    let cache = hf_hub::Cache::from_env();
    if let Some(path) = cache.model(repo.to_string()).get(file) {
        return Ok(path);
    }
    warn!(
        repo,
        file,
        "not in the HF cache; downloading through hf-hub. For large files prefer: hf download {repo} {file}"
    );
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .build()
        .context("building hf-hub client")?;
    api.model(repo.to_string())
        .get(file)
        .with_context(|| format!("downloading {repo}/{file}"))
}
