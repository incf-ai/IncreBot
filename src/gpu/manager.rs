//! The GPU manager (gpu-plan.md §2–3): one thread per card that reads the VRAM
//! monitor at `gpu_poll_hz`, runs the [`Policy`], uploads the weights on a
//! background thread, and on contention preempts the running reply and frees
//! everything on the card.
//!
//! The inference thread uses a card through [`Card::model`] (locked for the
//! length of one forward call) and stops at the next layer once
//! [`Card::preempted`] is set, so the manager gets the lock within about two
//! layers and drops the model, which frees all of its VRAM.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use tracing::{error, info, warn};

use super::model::GpuModel;
use super::monitor::{Monitor, Program, Reading, who};
use super::policy::{Action, Policy, State};
use super::vk::{Device, PciAddr};
use crate::model::Llama;

const MIB: u64 = 1 << 20;

/// The `[runtime]` GPU settings.
#[derive(Clone, Debug)]
pub struct GpuSettings {
    /// Preference order.
    pub gpus: Vec<GpuSpec>,
    pub other_mib: u64,
    pub poll_hz: u32,
    pub holdoff: Duration,
    /// Keep the weights on every free card, not only the preferred one.
    pub standby: bool,
    /// KV cache positions to allocate on the active card.
    pub n_ctx: usize,
}

/// One configured GPU.
#[derive(Clone, Debug)]
pub struct GpuSpec {
    pub pci: PciAddr,
    /// `name (pci)`, or the PCI address alone: for logs.
    pub label: String,
    /// The name from config, if any; `/stats` shows only this (or the card's
    /// model), never the PCI address.
    pub name: String,
}

/// One preemption, for logs and `/stats`.
#[derive(Clone, Debug)]
pub struct Preemption {
    pub at: SystemTime,
    /// From the reading that saw the other program to all our VRAM freed.
    pub release: Duration,
    /// Everyone else's VRAM at detection.
    pub other: u64,
    /// The programs found on the card afterwards (may be empty: unnamed).
    pub programs: Vec<Program>,
}

/// A card's state for `/stats`.
#[derive(Clone, Debug)]
pub struct CardStatus {
    /// The configured name, else the card's model: no PCI address.
    pub label: String,
    pub state: State,
    /// Whether the KV cache lives here (the card serving jobs).
    pub active: bool,
    pub backoff_left: Option<Duration>,
    pub holdoff: Duration,
    pub reading: Reading,
    pub preemptions: u64,
    pub last_preemption: Option<Preemption>,
}

pub struct Card {
    pub pci: PciAddr,
    /// See [`GpuSpec::label`].
    pub label: String,
    pub dev: Arc<Device>,
    /// The weights (and, on the active card, KV and scratch). `None` while the
    /// card holds nothing of ours.
    model: Mutex<Option<GpuModel>>,
    /// Set the moment the card must not be used: contended, failed, loading.
    preempted: AtomicBool,
    /// Set by the inference thread when a forward failed on this card.
    failed: AtomicBool,
    /// Whether the KV cache lives here; kept by the inference thread.
    active: AtomicBool,
    status: Mutex<CardStatus>,
}

impl Card {
    /// Whether the inference thread may start work on the card.
    pub fn usable(&self) -> bool {
        !self.preempted.load(Ordering::Relaxed)
    }

    /// Checked before every layer: stop, the card is being taken away.
    pub fn preempted(&self) -> bool {
        self.preempted.load(Ordering::Relaxed)
    }

    /// The model slot. Held only by the inference thread for one forward, and by
    /// the manager to install or drop the model.
    pub fn model(&self) -> MutexGuard<'_, Option<GpuModel>> {
        self.model.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reports a failed forward (device lost, out of memory): the manager frees
    /// the card and backs off.
    pub fn fail(&self) {
        self.preempted.store(true, Ordering::Relaxed);
        self.failed.store(true, Ordering::Relaxed);
    }

    pub fn status(&self) -> CardStatus {
        let mut st = self
            .status
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        st.active = self.active.load(Ordering::Relaxed);
        st
    }

    /// Records whether the KV cache lives on this card (inference thread).
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    fn set_status(&self, f: impl FnOnce(&mut CardStatus)) {
        f(&mut self.status.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

/// The cards in preference order, each with its manager thread.
pub struct Gpus {
    pub cards: Vec<Arc<Card>>,
}

impl Gpus {
    /// Opens each card and starts its manager. A card that can't be opened is
    /// logged and skipped; with none left, returns `None` (CPU only).
    pub fn start(settings: GpuSettings, cpu: Arc<Llama>) -> Option<Arc<Gpus>> {
        let mut cards = Vec::new();
        for spec in &settings.gpus {
            match open_card(spec, settings.other_mib) {
                Ok((card, monitor)) => cards.push((card, monitor)),
                Err(e) => {
                    error!(gpu = %spec.label, error = format!("{e:#}"), "GPU unavailable; skipping it")
                }
            }
        }
        if cards.is_empty() {
            return None;
        }
        let gpus = Arc::new(Gpus {
            cards: cards.iter().map(|(c, _)| c.clone()).collect(),
        });
        for (index, (card, monitor)) in cards.into_iter().enumerate() {
            let (gpus, cpu, settings) = (Arc::downgrade(&gpus), cpu.clone(), settings.clone());
            let label = card.label.clone();
            info!(gpu = %label, name = card.dev.name, preference = index, "GPU manager starting");
            // A card with no manager stays `preempted` (its `open_card` default)
            // forever, i.e. unused but harmless: the same outcome as `open_card`
            // failing above, so this degrades the same way instead of taking the
            // whole bot down over one card.
            if let Err(e) = std::thread::Builder::new()
                .name(format!("gpu-{}", card.pci))
                .spawn(move || manage(card, index, monitor, gpus, cpu, settings))
            {
                error!(gpu = %label, error = %e, "could not start the GPU manager thread; this card stays unused");
            }
        }
        Some(gpus)
    }

    pub fn status(&self) -> Vec<CardStatus> {
        self.cards.iter().map(|c| c.status()).collect()
    }

    /// Whether a reply starting now would run on a GPU: some card has the
    /// weights loaded and isn't being taken away.
    pub fn any_ready(&self) -> bool {
        self.cards
            .iter()
            .any(|c| c.usable() && c.status().state == State::Ready)
    }
}

/// One line per card for `/stats`, plus the most recent preemption.
pub fn describe(cards: &[CardStatus], now: SystemTime) -> String {
    let mut out = String::new();
    for c in cards {
        let state = match c.state {
            State::Ready if c.active => "active".to_string(),
            State::Ready => "standby (weights loaded)".to_string(),
            State::Loading => "loading weights".to_string(),
            State::Free => "idle".to_string(),
            State::Backoff => match c.backoff_left {
                Some(left) if left < c.holdoff => {
                    format!(
                        "backed off, back in {} s if it stays free",
                        left.as_secs().max(1)
                    )
                }
                _ => format!(
                    "in use by another program ({} MiB); back {} s after it frees",
                    c.reading.other / MIB,
                    c.holdoff.as_secs()
                ),
            },
        };
        out += &format!("{}: {state}\n", c.label);
    }
    let last = cards
        .iter()
        .filter_map(|c| c.last_preemption.as_ref().map(|p| (&c.label, p)))
        .max_by_key(|(_, p)| p.at);
    if let Some((gpu, p)) = last {
        let ago = now.duration_since(p.at).unwrap_or_default().as_secs();
        let by = if p.programs.is_empty() {
            format!("an unnamed program ({} MiB)", p.other / MIB)
        } else {
            p.programs
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        out += &format!(
            "last preemption: {gpu} {}m {}s ago by {by}; released in {} ms\n",
            ago / 60,
            ago % 60,
            p.release.as_millis()
        );
    }
    out
}

fn open_card(spec: &GpuSpec, other_mib: u64) -> Result<(Arc<Card>, Monitor)> {
    let pci = spec.pci;
    let monitor = Monitor::new(pci, other_mib).context("VRAM monitor (sysfs)")?;
    let dev = Device::open(pci)?;
    let shown = if spec.name.is_empty() { dev.name.clone() } else { spec.name.clone() };
    let card = Arc::new(Card {
        pci,
        label: spec.label.clone(),
        dev,
        model: Mutex::new(None),
        preempted: AtomicBool::new(true),
        failed: AtomicBool::new(false),
        active: AtomicBool::new(false),
        status: Mutex::new(CardStatus {
            label: shown,
            state: State::Free,
            active: false,
            backoff_left: None,
            holdoff: Duration::ZERO,
            reading: Reading::default(),
            preemptions: 0,
            last_preemption: None,
        }),
    });
    Ok((card, monitor))
}

struct Loader {
    handle: JoinHandle<Result<GpuModel>>,
    abort: Arc<AtomicBool>,
}

fn manage(
    card: Arc<Card>,
    index: usize,
    mut monitor: Monitor,
    gpus: std::sync::Weak<Gpus>,
    cpu: Arc<Llama>,
    s: GpuSettings,
) {
    let period = Duration::from_secs_f64(1.0 / s.poll_hz.max(1) as f64);
    let mut policy = Policy::new(s.holdoff);
    let mut loader: Option<Loader> = None;
    let mut sysfs_ok = true;
    let mut next = Instant::now();
    loop {
        let now = Instant::now();
        let reading = match monitor.read(now, &card.dev.tracker) {
            Ok(r) => {
                sysfs_ok = true;
                r
            }
            Err(e) => {
                // Without readings sharing isn't safe: stay off the card.
                if sysfs_ok {
                    warn!(gpu = %card.label, error = format!("{e:#}"), "VRAM monitor failed; treating the card as busy");
                }
                sysfs_ok = false;
                Reading {
                    contended: true,
                    ..Reading::default()
                }
            }
        };
        if reading.contended {
            // Before anything else: the inference thread stops at its next layer.
            card.preempted.store(true, Ordering::Relaxed);
        }
        let Some(all) = gpus.upgrade() else {
            return; // shutting down
        };
        let want = s.standby
            || all.cards[..index]
                .iter()
                .all(|c| c.status().state == State::Backoff);
        drop(all);

        let before = policy.state();
        let mut action = if card.failed.swap(false, Ordering::Relaxed) {
            error!(gpu = %card.label, "inference failed on this GPU; freeing it and backing off");
            policy.failed(now)
        } else {
            policy.tick(now, reading.contended, want)
        };

        if let Some(l) = loader.take_if(|l| l.handle.is_finished()) {
            let aborted = l.abort.load(Ordering::Relaxed);
            match l.handle.join() {
                Ok(Ok(model)) if policy.state() == State::Loading && !aborted => {
                    *card.model() = Some(model);
                    policy.loaded(now, true);
                    card.preempted.store(false, Ordering::Relaxed);
                    info!(gpu = %card.label, mib = card.dev.tracker.held() / MIB, "GPU ready");
                }
                Ok(Ok(_)) => {} // superseded: dropping it frees it
                Ok(Err(e)) => {
                    if !aborted {
                        error!(gpu = %card.label, error = format!("{e:#}"), "uploading the weights failed");
                        policy.loaded(now, false);
                    }
                }
                Err(_) => {
                    error!(gpu = %card.label, "the upload thread panicked");
                    policy.loaded(now, false);
                }
            }
        }

        match action.take() {
            Some(Action::Load) => {
                card.preempted.store(true, Ordering::Relaxed);
                let abort = Arc::new(AtomicBool::new(false));
                let (dev, cpu, ab) = (card.dev.clone(), cpu.clone(), abort.clone());
                info!(gpu = %card.label, "uploading the weights");
                match std::thread::Builder::new()
                    .name(format!("gpu-load-{}", card.pci))
                    .spawn(move || {
                        let mut m = GpuModel::load(&dev, &cpu, &ab)?;
                        m.self_test(&cpu)?;
                        Ok(m)
                    }) {
                    Ok(handle) => loader = Some(Loader { handle, abort }),
                    // Same outcome as a failed upload a few lines below: back off
                    // and retry later, instead of panicking this card's manager
                    // thread and leaving it stuck in `Loading` forever.
                    Err(e) => {
                        error!(gpu = %card.label, error = %e, "could not start the upload thread; backing off");
                        policy.loaded(now, false);
                    }
                }
            }
            Some(Action::Release) => {
                let p = release(
                    &card,
                    &mut loader,
                    now,
                    &reading,
                    policy.state() == State::Backoff,
                );
                if let Some(p) = p {
                    card.set_status(|st| {
                        st.preemptions += 1;
                        st.last_preemption = Some(p);
                    });
                }
            }
            None => {}
        }

        if policy.state() != before {
            info!(
                gpu = %card.label,
                from = ?before,
                to = ?policy.state(),
                holdoff_s = policy.holdoff().as_secs(),
                other_mib = reading.other / MIB,
                "GPU state"
            );
        }
        card.set_status(|st| {
            st.state = policy.state();
            st.backoff_left = policy.backoff_left(now);
            st.holdoff = policy.holdoff();
            st.reading = reading;
        });

        next += period;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}

/// Stops all use of the card and frees everything on it. `contended` (rather
/// than merely unwanted) also names the other programs and returns the record.
fn release(
    card: &Card,
    loader: &mut Option<Loader>,
    detected: Instant,
    reading: &Reading,
    contended: bool,
) -> Option<Preemption> {
    card.preempted.store(true, Ordering::Relaxed);
    if let Some(l) = loader.as_ref() {
        l.abort.store(true, Ordering::Relaxed);
    }
    // Waits for the inference thread to leave its forward (at most ~2 layers).
    let model = card.model().take();
    let stopped = detected.elapsed();
    let had = model.is_some();
    drop(model);
    card.set_active(false);
    let release = detected.elapsed();
    if !contended {
        info!(gpu = %card.label, "weights unloaded (a preferred GPU is ready)");
        return None;
    }
    info!(
        gpu = %card.label,
        release_ms = release.as_secs_f64() * 1e3,
        stop_ms = stopped.as_secs_f64() * 1e3,
        other_mib = reading.other / MIB,
        held_weights = had,
        "another program is using the GPU: freed all our VRAM"
    );
    // The slow path: who is it (logs and /stats only).
    let programs = who(card.pci);
    if programs.is_empty() {
        info!(gpu = %card.label, "the other program couldn't be named (e.g. it runs on the host)");
    } else {
        let names: Vec<String> = programs.iter().map(|p| p.to_string()).collect();
        info!(gpu = %card.label, programs = names.join(", "), "other programs on the GPU");
    }
    Some(Preemption {
        at: SystemTime::now(),
        release,
        other: reading.other,
        programs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_lines() {
        let base = CardStatus {
            label: "upper".into(),
            state: State::Ready,
            active: true,
            backoff_left: None,
            holdoff: Duration::from_secs(30),
            reading: Reading::default(),
            preemptions: 0,
            last_preemption: None,
        };
        let now = SystemTime::now();
        let busy = CardStatus {
            label: "Radeon RX 6600".into(),
            state: State::Backoff,
            backoff_left: Some(Duration::from_secs(30)),
            reading: Reading {
                other: 1200 * MIB,
                ..Reading::default()
            },
            last_preemption: Some(Preemption {
                at: now - Duration::from_secs(125),
                release: Duration::from_millis(70),
                other: 1200 * MIB,
                programs: vec![Program {
                    pid: 7,
                    name: "koboldcpp".into(),
                    vram: 1200 * MIB,
                }],
            }),
            ..base.clone()
        };
        let text = describe(&[base.clone(), busy.clone()], now);
        assert_eq!(
            text,
            "upper: active\n\
             Radeon RX 6600: in use by another program (1200 MiB); back 30 s after it frees\n\
             last preemption: Radeon RX 6600 2m 5s ago by koboldcpp (pid 7, 1200 MiB); released in 70 ms\n"
        );
        let waiting = CardStatus {
            backoff_left: Some(Duration::from_secs(12)),
            last_preemption: None,
            ..busy
        };
        let standby = CardStatus {
            active: false,
            ..base
        };
        assert_eq!(
            describe(&[standby, waiting], now),
            "upper: standby (weights loaded)\nRadeon RX 6600: backed off, back in 12 s if it stays free\n"
        );
    }
}
