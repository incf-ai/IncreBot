//! Optional per-section timing of the forward pass. Enabled with
//! `INCREBOT_PROFILE=1`; burn-flex executes eagerly, so wall-clock time around
//! each op is the op's cost. The GPU backend adds per-dispatch GPU time
//! from timestamp queries under `gpu_*` sections.

use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("INCREBOT_PROFILE").is_some_and(|v| v != "0"));

// Global, not thread-local: prefill runs on a prefill-pool thread (runtime.rs).
static TOTALS: Mutex<Vec<(&'static str, Duration)>> = Mutex::new(Vec::new());

pub fn enabled() -> bool {
    *ENABLED
}

/// Runs `f`, adding its wall time to `section` when profiling is on.
#[inline]
pub fn time<T>(section: &'static str, f: impl FnOnce() -> T) -> T {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let out = f();
    add(section, start.elapsed());
    out
}

/// Adds `d` to `section` (for times measured elsewhere, e.g. GPU timestamps).
pub fn add(section: &'static str, d: Duration) {
    let mut t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    match t.iter_mut().find(|(s, _)| *s == section) {
        Some((_, total)) => *total += d,
        None => t.push((section, d)),
    }
}

/// Returns and clears the accumulated totals, largest first.
pub fn take() -> Vec<(&'static str, Duration)> {
    let mut v = std::mem::take(&mut *TOTALS.lock().unwrap_or_else(|e| e.into_inner()));
    v.sort_by_key(|&(_, d)| std::cmp::Reverse(d));
    v
}
