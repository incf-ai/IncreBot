//! The compute pools (design.md §4): a global rayon pool for decode, one thread per
//! physical P-core when pinned, and, when `threads > 8`, a separate prefill pool
//! that also uses the P-cores' hyperthread siblings and the E-cores.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use tracing::{info, warn};

/// Logical CPUs of the 14900K's physical P-cores (sibling pairs (0,1), (2,3), …).
const P_CORE_CPUS: [usize; 8] = [0, 2, 4, 6, 8, 10, 12, 14];

/// Prefill pool CPU order: physical P-cores, then their siblings, then E-cores
/// (16–31). Measured per-thread throughput falls in that order (design.md §3).
const PREFILL_CPUS: [usize; 32] = [
    0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];

static PREFILL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

/// Decode (the global pool) gets `min(threads, 8)` threads: it is bandwidth bound
/// and saturates at 8. Above 8, compute-bound prefill gets its own `threads`-wide pool.
pub fn init_pool(threads: usize, pin: bool) -> Result<()> {
    let decode = threads.min(P_CORE_CPUS.len());
    let pin_decode = pin && pinning_possible(&P_CORE_CPUS[..decode]);
    rayon::ThreadPoolBuilder::new()
        .num_threads(decode)
        .thread_name(|i| format!("infer-{i}"))
        .start_handler(move |i| {
            if pin_decode && let Err(e) = pin_to(P_CORE_CPUS[i]) {
                warn!(thread = i, error = %e, "pinning failed; this thread runs unpinned");
            }
        })
        .build_global()
        .context("building the global rayon pool")?;
    info!(threads = decode, pinned = pin_decode, "decode pool ready");

    if threads > decode {
        let pin_prefill =
            pin && threads <= PREFILL_CPUS.len() && pinning_possible(&PREFILL_CPUS[..threads]);
        if pin && !pin_prefill {
            warn!(threads, "prefill pool can't be pinned; running it unpinned");
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("prefill-{i}"))
            .start_handler(move |i| {
                if pin_prefill && let Err(e) = pin_to(PREFILL_CPUS[i]) {
                    warn!(thread = i, error = %e, "pinning failed; this thread runs unpinned");
                }
            })
            .build()
            .context("building the prefill rayon pool")?;
        PREFILL_POOL.set(pool).ok();
        info!(threads, pinned = pin_prefill, "prefill pool ready");
    }
    Ok(())
}

/// Runs `f` on the prefill pool if there is one, else on the calling thread's pool.
pub fn on_prefill_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match PREFILL_POOL.get() {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// Pinning needs every target CPU in our affinity mask (another machine or a
/// restrictive cpuset may lack them). Otherwise run unpinned and say so.
fn pinning_possible(cpus: &[usize]) -> bool {
    let allowed = allowed_cpus();
    if let Some(&cpu) = cpus.iter().find(|c| !allowed.contains(c)) {
        warn!(
            cpu,
            "CPU not in this process's affinity mask; running unpinned"
        );
        return false;
    }
    true
}

fn allowed_cpus() -> Vec<usize> {
    // SAFETY: plain libc calls on a zeroed, correctly sized cpu_set_t.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .collect()
    }
}

fn pin_to(cpu: usize) -> std::io::Result<()> {
    // SAFETY: as above; pins only the calling thread (pid 0).
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}
