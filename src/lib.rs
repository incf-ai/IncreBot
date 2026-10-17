//! increbot: a Discord bot running LLaMA2-13B-Tiefighter on CPU with Burn.
//! See design.md for the full design.

pub mod chat;
pub mod cli;
pub mod config;
pub mod discord;
pub mod engine;
pub mod gguf;
pub mod gpu;
pub mod hub;
pub mod model;
pub mod persona;
pub mod runtime;
pub mod tokenizer;

/// When this binary was built (UTC, RFC 3339), from `build.rs`.
pub const BUILD_TIMESTAMP: &str = env!("INCREBOT_BUILD_TIMESTAMP");

/// [`BUILD_TIMESTAMP`] in Unix seconds.
pub const BUILD_EPOCH: &str = env!("INCREBOT_BUILD_EPOCH");

/// A short, stable hash of `bytes` (FNV-1a, first 7 hex digits): tells
/// persona setups apart in `/stats`, `/version` and `persona_report
/// fingerprint` (see [`persona::Persona::fingerprint`]).
pub fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{hash:016x}")[..7].to_string()
}

/// How long ago this binary was built, e.g. "123 days 11 hours 17 minutes
/// ago" (see [`age_text`]).
pub fn build_age(now: std::time::SystemTime) -> String {
    let built = BUILD_EPOCH.parse::<u64>().unwrap_or(0);
    let now = now.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    match now.checked_sub(built) {
        Some(secs) => age_text(secs),
        None => "in the future".into(),
    }
}

/// [`VERSION`] with the build's age inside the parentheses: "0.1.0 (commit
/// abc123, built 2026-10-13T03:05:03Z, 5 minutes ago)", for `/stats`.
pub fn version_with_age(now: std::time::SystemTime) -> String {
    format!("{}, {})", VERSION.strip_suffix(')').unwrap_or(VERSION), build_age(now))
}

/// `secs` as "N days N hours N minutes ago", from the largest nonzero unit
/// down ("2 hours 0 minutes ago", "5 minutes ago"), singular for 1, and
/// "just now" under a minute.
pub fn age_text(secs: u64) -> String {
    let (days, hours, minutes) = (secs / 86_400, secs % 86_400 / 3_600, secs % 3_600 / 60);
    let unit = |n: u64, name: &str| format!("{n} {name}{}", if n == 1 { "" } else { "s" });
    let parts = if days > 0 {
        vec![unit(days, "day"), unit(hours, "hour"), unit(minutes, "minute")]
    } else if hours > 0 {
        vec![unit(hours, "hour"), unit(minutes, "minute")]
    } else if minutes > 0 {
        vec![unit(minutes, "minute")]
    } else {
        return "just now".into();
    };
    format!("{} ago", parts.join(" "))
}


/// The git commit this binary was built from (short hash, `-dirty` with
/// uncommitted changes, or `unknown`), from `build.rs`.
pub const GIT_COMMIT: &str = env!("INCREBOT_GIT_COMMIT");

/// `"<version> (commit <hash>, built <timestamp>)"`, for `--version`, the startup
/// log, `/stats` and `/version`.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (commit ",
    env!("INCREBOT_GIT_COMMIT"),
    ", built ",
    env!("INCREBOT_BUILD_TIMESTAMP"),
    ")"
);

#[cfg(test)]
mod age_tests {
    use super::age_text;

    #[test]
    fn fingerprints_are_short_and_stable() {
        // FNV-1a test vector: "a" hashes to af63dc4c8601ec8c.
        assert_eq!(super::fingerprint(b"a"), "af63dc4");
        assert_ne!(super::fingerprint(b"floob"), super::fingerprint(b"floob "));
    }

    #[test]
    fn version_with_age_keeps_the_version_and_adds_the_age() {
        let v = super::version_with_age(std::time::SystemTime::now());
        assert!(v.starts_with(super::VERSION.strip_suffix(')').unwrap()), "{v}");
        assert!(v.ends_with(" ago)") || v.ends_with("just now)"), "{v}");
    }

    #[test]
    fn age_reads_like_the_version_line() {
        assert_eq!(age_text(123 * 86_400 + 11 * 3_600 + 17 * 60 + 5), "123 days 11 hours 17 minutes ago");
        assert_eq!(age_text(86_400 + 60), "1 day 0 hours 1 minute ago");
        assert_eq!(age_text(2 * 3_600), "2 hours 0 minutes ago");
        assert_eq!(age_text(5 * 60 + 59), "5 minutes ago");
        assert_eq!(age_text(59), "just now");
    }
}
