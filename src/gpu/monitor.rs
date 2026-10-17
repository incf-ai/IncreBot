//! The VRAM monitor (gpu-plan.md §2): whether another program is using a card.
//!
//! Every tick reads the card's total VRAM use from sysfs (a few µs) and computes
//!
//! ```text
//! other = mem_info_vram_used − baseline − ours − releasing − slack
//! contended ⇔ other > threshold
//! ```
//!
//! `ours` and `releasing` come from the device's [`Tracker`]. Naming the other
//! programs ([`who`]) is the slow path, used only after a detection, for logs and
//! `/stats`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::vk::{PciAddr, Tracker};

const MIB: u64 = 1 << 20;

/// The driver's own overhead on top of our allocations: measured ≈ 0 (32 KiB per
/// device, gpu-plan.md finding), so a constant covers it.
const SLACK: u64 = MIB;

/// An idle card uses ~57 MiB (display/driver). A first reading above this is
/// someone else's memory, not the baseline.
const BASELINE_MAX: u64 = 256 * MIB;

/// While we hold nothing and the total stays within [`STABLE_BAND`] for this
/// long, the baseline follows it, also upwards (slow drift, e.g. a display
/// attached later), as long as the rise is below the threshold.
const STABLE_FOR: Duration = Duration::from_secs(60);
const STABLE_BAND: u64 = 4 * MIB;

/// `mem_info_vram_used` of the card at `pci`.
pub fn vram_used_path(pci: PciAddr) -> PathBuf {
    PathBuf::from(format!("/sys/bus/pci/devices/{pci}/mem_info_vram_used"))
}

pub fn read_u64(path: &Path) -> Result<u64> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    s.trim()
        .parse()
        .with_context(|| format!("parsing {} ({s:?})", path.display()))
}

/// One reading.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reading {
    pub used: u64,
    pub baseline: u64,
    pub ours: u64,
    pub releasing: u64,
    /// Everyone else's VRAM.
    pub other: u64,
    pub contended: bool,
}

/// The idle baseline and the threshold rule, separate from sysfs for testing.
#[derive(Debug)]
pub struct Rule {
    threshold: u64,
    baseline: Option<u64>,
    /// While we hold nothing: the reading the total has stayed near, and since when.
    stable: Option<(u64, Instant)>,
}

impl Rule {
    pub fn new(threshold: u64) -> Self {
        Self {
            threshold,
            baseline: None,
            stable: None,
        }
    }

    pub fn reading(&mut self, now: Instant, used: u64, ours: u64, releasing: u64) -> Reading {
        let idle = ours == 0 && releasing == 0;
        let baseline = match self.baseline {
            None => used.min(BASELINE_MAX),
            Some(b) if idle => b.min(used),
            Some(b) => b,
        };
        let mut baseline = baseline;
        if idle {
            match self.stable {
                Some((v, since)) if used.abs_diff(v) <= STABLE_BAND => {
                    let rise = used.saturating_sub(baseline);
                    if now.saturating_duration_since(since) >= STABLE_FOR && rise <= self.threshold
                    {
                        baseline = used;
                    }
                }
                _ => self.stable = Some((used, now)),
            }
        } else {
            self.stable = None;
        }
        self.baseline = Some(baseline);
        let other = used.saturating_sub(baseline + ours + releasing + SLACK);
        Reading {
            used,
            baseline,
            ours,
            releasing,
            other,
            contended: other > self.threshold,
        }
    }
}

/// Reads one card's sysfs total and applies the rule.
pub struct Monitor {
    path: PathBuf,
    rule: Rule,
}

impl Monitor {
    pub fn new(pci: PciAddr, threshold_mib: u64) -> Result<Self> {
        let path = vram_used_path(pci);
        read_u64(&path)?;
        Ok(Self {
            path,
            rule: Rule::new(threshold_mib * MIB),
        })
    }

    pub fn read(&mut self, now: Instant, tracker: &Tracker) -> Result<Reading> {
        // Our numbers first: an allocation is counted before it is made, so it
        // can't show up in the total before it shows up in `ours`.
        let (ours, releasing) = (tracker.held(), tracker.pending_release(now));
        let used = read_u64(&self.path)?;
        Ok(self.rule.reading(now, used, ours, releasing))
    }
}

/// Another program using a card.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub pid: u32,
    pub name: String,
    pub vram: u64,
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (pid {}, {} MiB)",
            self.name,
            self.pid,
            self.vram / MIB
        )
    }
}

/// The programs other than us using the card at `pci`, largest first: ROCm ones
/// from KFD sysfs, the rest from `/proc/*/fdinfo` (needs `--pid=host` and, for
/// other users' processes, ptrace access; host processes stay unnamed,
/// gpu-plan.md finding 4). Slow (scans every process): call rarely.
pub fn who(pci: PciAddr) -> Vec<Program> {
    let me = std::process::id();
    let mut found: Vec<Program> = Vec::new();
    let mut add = |pid: u32, vram: u64| {
        if pid == me || vram == 0 {
            return;
        }
        match found.iter_mut().find(|p| p.pid == pid) {
            Some(p) => p.vram = p.vram.max(vram),
            None => found.push(Program {
                pid,
                name: std::fs::read_to_string(format!("/proc/{pid}/comm"))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| "?".into()),
                vram,
            }),
        }
    };
    if let Some(gpu_id) = kfd_gpu_id(pci) {
        for (pid, vram) in kfd_users(gpu_id) {
            add(pid, vram);
        }
    }
    for (pid, vram) in fdinfo_users(pci) {
        add(pid, vram);
    }
    found.sort_by_key(|p| std::cmp::Reverse(p.vram));
    found
}

/// KFD's id for the card: the topology node whose `location_id` is
/// `bus << 8 | device << 3 | function`.
fn kfd_gpu_id(pci: PciAddr) -> Option<u32> {
    let want = (pci.bus << 8) | (pci.device << 3) | pci.function;
    for node in std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes")
        .ok()?
        .flatten()
    {
        let props = std::fs::read_to_string(node.path().join("properties")).unwrap_or_default();
        let field = |name: &str| {
            props.lines().find_map(|l| {
                let (k, v) = l.split_once(' ')?;
                (k == name).then(|| v.trim().parse::<u32>().ok()).flatten()
            })
        };
        if field("location_id") == Some(want) && field("domain").unwrap_or(0) == pci.domain {
            let id = read_u64(&node.path().join("gpu_id")).ok()? as u32;
            return (id != 0).then_some(id);
        }
    }
    None
}

/// `(pid, bytes)` from `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`.
fn kfd_users(gpu_id: u32) -> Vec<(u32, u64)> {
    let Ok(dir) = std::fs::read_dir("/sys/class/kfd/kfd/proc") else {
        return Vec::new();
    };
    dir.flatten()
        .filter_map(|e| {
            let pid = e.file_name().to_str()?.parse().ok()?;
            let vram = read_u64(&e.path().join(format!("vram_{gpu_id}"))).ok()?;
            Some((pid, vram))
        })
        .collect()
}

/// `(pid, bytes)` from DRM fdinfo, one entry per client id.
fn fdinfo_users(pci: PciAddr) -> Vec<(u32, u64)> {
    let pdev = pci.to_string();
    let Ok(procs) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for p in procs.flatten() {
        let Some(pid) = p.file_name().to_str().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        let Ok(fds) = std::fs::read_dir(p.path().join("fdinfo")) else {
            continue;
        };
        let mut clients: Vec<(u64, u64)> = Vec::new();
        for fd in fds.flatten() {
            let Ok(text) = std::fs::read_to_string(fd.path()) else {
                continue;
            };
            if let Some((client, vram)) = parse_fdinfo(&text, &pdev)
                && !clients.iter().any(|&(c, _)| c == client)
            {
                clients.push((client, vram));
            }
        }
        let total: u64 = clients.iter().map(|&(_, v)| v).sum();
        if total > 0 {
            out.push((pid, total));
        }
    }
    out
}

/// `(drm-client-id, drm-memory-vram bytes)` of one fdinfo file on device `pdev`.
fn parse_fdinfo(text: &str, pdev: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        text.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            (k.trim() == name).then(|| v.trim().to_string())
        })
    };
    if field("drm-pdev")? != pdev {
        return None;
    }
    let client = field("drm-client-id")?.parse().ok()?;
    let vram = field("drm-memory-vram")?;
    let (num, unit) = vram.split_once(' ').unwrap_or((&vram, ""));
    let n: u64 = num.parse().ok()?;
    let bytes = match unit {
        "KiB" => n << 10,
        "MiB" => n << 20,
        "GiB" => n << 30,
        _ => n,
    };
    Some((client, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: u64 = 100 * MIB;

    #[test]
    fn other_programs_above_threshold_contend() {
        let t0 = Instant::now();
        let mut r = Rule::new(T);
        let a = r.reading(t0, 57 * MIB, 0, 0);
        assert_eq!(a.baseline, 57 * MIB);
        assert!(!a.contended);
        // We hold 8 GiB: still free.
        let b = r.reading(t0, 57 * MIB + (8 << 30), 8 << 30, 0);
        assert_eq!(b.other, 0);
        assert!(!b.contended);
        // Someone takes 200 MiB.
        let c = r.reading(t0, 57 * MIB + (8 << 30) + 200 * MIB, 8 << 30, 0);
        assert!(c.contended);
        assert_eq!(c.other, 199 * MIB);
        // 50 MiB: under the threshold.
        let d = r.reading(t0, 57 * MIB + 50 * MIB, 0, 0);
        assert!(!d.contended);
    }

    #[test]
    fn memory_we_just_freed_is_not_someone_else() {
        let t0 = Instant::now();
        let mut r = Rule::new(T);
        r.reading(t0, 57 * MIB, 0, 0);
        // Freed 8 GiB, the kernel has released only half of it so far.
        let a = r.reading(t0, 57 * MIB + (4 << 30), 0, 5 << 30);
        assert!(!a.contended);
        // While releasing, the baseline doesn't move.
        assert_eq!(a.baseline, 57 * MIB);
    }

    #[test]
    fn busy_card_at_startup_is_not_the_baseline() {
        let t0 = Instant::now();
        let mut r = Rule::new(T);
        let a = r.reading(t0, 20 << 30, 0, 0);
        assert_eq!(a.baseline, BASELINE_MAX);
        assert!(a.contended);
        // It leaves: the baseline drops to the real idle level.
        let b = r.reading(t0, 57 * MIB, 0, 0);
        assert_eq!(b.baseline, 57 * MIB);
        assert!(!b.contended);
    }

    #[test]
    fn baseline_follows_slow_drift_below_threshold_only() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut r = Rule::new(T);
        r.reading(at(0), 57 * MIB, 0, 0);
        // A display adds 40 MiB and stays: adopted after a minute.
        r.reading(at(1), 97 * MIB, 0, 0);
        assert_eq!(r.reading(at(30), 97 * MIB, 0, 0).baseline, 57 * MIB);
        assert_eq!(r.reading(at(62), 97 * MIB, 0, 0).baseline, 97 * MIB);
        // An idle program holding 2 GiB for hours is never the baseline.
        r.reading(at(100), 97 * MIB + (2 << 30), 0, 0);
        let late = r.reading(at(10_000), 97 * MIB + (2 << 30), 0, 0);
        assert_eq!(late.baseline, 97 * MIB);
        assert!(late.contended);
    }

    #[test]
    fn fdinfo_parsing() {
        let text = "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-client-id:\t1234\n\
                    drm-pdev:\t0000:06:00.0\ndrm-memory-vram:\t512028 KiB\ndrm-memory-gtt:\t2048 KiB\n";
        assert_eq!(
            parse_fdinfo(text, "0000:06:00.0"),
            Some((1234, 512_028 << 10))
        );
        assert_eq!(parse_fdinfo(text, "0000:03:00.0"), None);
        assert_eq!(parse_fdinfo("pos:\t0\n", "0000:06:00.0"), None);
    }
}
