//! Embeds build metadata, shown by `--version`, at startup, in `/stats` and by
//! `/version`:
//! - `INCREBOT_BUILD_TIMESTAMP`: the build time (UTC, RFC 3339), and
//!   `INCREBOT_BUILD_EPOCH` the same in Unix seconds (for `/version`'s "ago").
//!   `SOURCE_DATE_EPOCH` overrides it for reproducible builds.
//! - `INCREBOT_GIT_COMMIT`: the short git commit, `-dirty` if tracked files have
//!   uncommitted changes, or `unknown` outside a git checkout. Set
//!   `INCREBOT_GIT_COMMIT` in the environment to supply it where `.git` is absent
//!   (container builds, deploy.md).
//! - The GPU backend's compute shaders (`src/gpu/shaders`), compiled to SPIR-V
//!   with `glslc` into `OUT_DIR`. Without `glslc` the files are left empty and the
//!   bot runs CPU-only (gpu-plan.md); `INCREBOT_GPU_SHADERS_REQUIRED=1` makes that
//!   a build error instead.
//! - `INCREBOT_DEFAULT_PERSONA_FILE`: the persona file the shipped `config.rsn`
//!   names (`discord.persona_file`). It's the code default, and its text is
//!   embedded as the built-in persona, so neither drifts from the shipped config.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let secs = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock before 1970")
                .as_secs()
        });
    println!("cargo:rustc-env=INCREBOT_BUILD_TIMESTAMP={}", rfc3339(secs));
    println!("cargo:rustc-env=INCREBOT_BUILD_EPOCH={secs}");

    let commit = std::env::var("INCREBOT_GIT_COMMIT")
        .ok()
        .filter(|c| !c.trim().is_empty())
        .or_else(git_commit)
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=INCREBOT_GIT_COMMIT={}", commit.trim());

    // Re-run (and so re-stamp) whenever the code, manifest or commit changes, not
    // on every build: an unchanged binary keeps its timestamp. Only existing paths
    // are watched; a missing one would make Cargo re-run this on every build.
    for path in ["src", "Cargo.toml", "Cargo.lock", "build.rs", "config.rsn"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for path in git_watch_paths() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    compile_shaders();
    println!("cargo:rustc-env=INCREBOT_DEFAULT_PERSONA_FILE={}", shipped_persona_file());
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    println!("cargo:rerun-if-env-changed=INCREBOT_GPU_SHADERS_REQUIRED");
    println!("cargo:rerun-if-env-changed=INCREBOT_GIT_COMMIT");
}

/// Each SPIR-V module: output name, source file, extra `glslc` arguments.
const SHADERS: &[(&str, &str, &[&str])] = &[
    ("matvec", "matvec.comp", &[]),
    ("mmq_q4k", "mmq.comp", &["-DQ4K"]),
    ("mmq_q6k", "mmq.comp", &["-DQ6K"]),
    ("quantize_q8k", "quantize_q8k.comp", &[]),
    ("rmsnorm", "rmsnorm.comp", &[]),
    ("rope_store", "rope_store.comp", &[]),
    ("swiglu", "swiglu.comp", &[]),
    ("flash_attn", "flash_attn.comp", &[]),
    ("attention_split", "attention_split.comp", &[]),
    ("attention_merge", "attention_merge.comp", &[]),
];

fn compile_shaders() {
    let src = Path::new("src/gpu/shaders");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let required = std::env::var_os("INCREBOT_GPU_SHADERS_REQUIRED").is_some();
    let have_glslc = Command::new("glslc").arg("--version").output().is_ok();
    if !have_glslc && required {
        panic!("INCREBOT_GPU_SHADERS_REQUIRED is set but glslc is not installed");
    }
    for (name, file, args) in SHADERS {
        let dst = out.join(format!("{name}.spv"));
        if !have_glslc {
            std::fs::write(&dst, []).expect("writing an empty shader");
            continue;
        }
        let status = Command::new("glslc")
            .args(["--target-env=vulkan1.3", "-O"])
            .args(*args)
            .arg(src.join(file))
            .arg("-o")
            .arg(&dst)
            .status()
            .expect("running glslc");
        assert!(status.success(), "glslc failed on {file}");
    }
    if !have_glslc {
        println!("cargo:warning=glslc not found: GPU shaders not built, GPU backend disabled");
    }
}

/// Just the part of `config.rsn` build.rs needs; serde skips the rest.
#[derive(serde::Deserialize)]
#[serde(rename = "Config")]
struct Shipped {
    discord: ShippedDiscord,
}

#[derive(serde::Deserialize)]
#[serde(rename = "Discord")]
struct ShippedDiscord {
    persona_file: String,
}

/// `discord.persona_file` in the shipped `config.rsn`, which must exist.
fn shipped_persona_file() -> String {
    let text = std::fs::read_to_string("config.rsn").expect("reading config.rsn");
    let shipped: Shipped = rsn::from_str(&text).expect("parsing config.rsn");
    let file = shipped.discord.persona_file;
    assert!(Path::new(&file).is_file(), "config.rsn names persona_file {file:?}, which doesn't exist");
    file
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// `abc123def456`, or `abc123def456-dirty` with uncommitted changes to tracked files.
fn git_commit() -> Option<String> {
    let hash = git(&["rev-parse", "--short=12", "HEAD"])?;
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    Some(if dirty { format!("{hash}-dirty") } else { hash })
}

/// HEAD, the index, and the current branch's ref file, if this is a git checkout.
fn git_watch_paths() -> Vec<PathBuf> {
    let Some(dir) = git(&["rev-parse", "--git-dir"]).map(PathBuf::from) else {
        return Vec::new();
    };
    let mut paths = vec![dir.join("HEAD"), dir.join("index"), dir.join("packed-refs")];
    if let Ok(head) = std::fs::read_to_string(dir.join("HEAD"))
        && let Some(r) = head.trim().strip_prefix("ref: ")
    {
        paths.push(dir.join(r));
    }
    paths.retain(|p| Path::new(p).exists());
    paths
}

/// `YYYY-MM-DDTHH:MM:SSZ` for Unix seconds (Howard Hinnant's civil-from-days).
fn rfc3339(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}
