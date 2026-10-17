//! How often a user may make the bot generate by command (design.md §6): the
//! [`Limits`] in config, per user across channels. Three layers, the longest
//! wait winning:
//! 1. at least `min_gap_ms` between two;
//! 2. at most `count` in any `ms`, for each window (a batch counts as one);
//! 3. a gentle backoff on top of the gap: each one adds 1 to the user's
//!    pressure (a batch `batch_cost`), which drains steadily, and the extra
//!    wait is `backoff_unit_ms × (backoff_growth ^ pressure − 1)`, at most
//!    `backoff_cap_ms`. At pressure 0 it is nothing. Pressure stops where
//!    the wait reaches its cap, so recovery starts as soon as use slows.
//!
//! A use is billed when it starts, as a [`Hold`] dated then, and kept if it
//! succeeds; one that doesn't is [`release`](Limiter::release)d, as if it never
//! happened. Admins are held to it too unless `limits.admins_exempt`. Kept in
//! memory: a restart forgets it.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::config::{ChannelLimits, Limits};

#[derive(Default)]
struct Use {
    /// When each recent one started, oldest first.
    times: VecDeque<Instant>,
    /// The pressure as of `pressure_at`.
    pressure: f64,
    pressure_at: Option<Instant>,
}

impl Use {
    fn last(&self) -> Option<Instant> {
        self.times.back().copied()
    }

    fn pressure(&self, l: &Limits, t: Instant) -> f64 {
        let Some(at) = self.pressure_at else { return 0.0 };
        let drained = t.saturating_duration_since(at).as_secs_f64() * 1000.0 / l.backoff_drain_ms as f64;
        (self.pressure - drained).max(0.0)
    }
}

/// The pressure at which a backoff of `unit_ms × (growth ^ p − 1)` reaches
/// `cap_ms`: more would add no wait, only a longer recovery, so pressure
/// stops there.
fn pressure_cap(unit_ms: u64, growth: f64, cap_ms: u64) -> f64 {
    if unit_ms == 0 || growth <= 1.0 {
        return f64::INFINITY;
    }
    (1.0 + cap_ms as f64 / unit_ms as f64).ln() / growth.ln()
}

/// The backoff's extra wait at `pressure`.
pub fn extra(l: &Limits, pressure: f64) -> Duration {
    let ms = l.backoff_unit_ms as f64 * (l.backoff_growth.powf(pressure) - 1.0);
    Duration::from_millis(ms.clamp(0.0, l.backoff_cap_ms as f64) as u64)
}

/// The refusal for a wait.
pub fn message(wait: Duration) -> String {
    let secs = wait.as_millis().div_ceil(1000).max(1);
    format!("Slow down: you can do that again in {secs} s.")
}

/// One use's bill, from [`Limiter::try_hold`] or [`Limiter::hold`]: kept
/// unless [`release`](Limiter::release)d.
#[derive(Debug)]
pub struct Hold {
    user: u64,
    at: Instant,
    cost: f64,
}

#[derive(Default)]
pub struct Limiter {
    users: HashMap<u64, Use>,
}

impl Limiter {
    /// How long `user` must still wait before the next one, if at all.
    pub fn wait(&self, l: &Limits, user: u64, now: Instant) -> Option<Duration> {
        let u = self.users.get(&user)?;
        let mut until = Duration::ZERO;
        if let Some(last) = u.last() {
            let gap = Duration::from_millis(l.min_gap_ms);
            // The earliest moment, from now, that is far enough past the last
            // one; the pressure drains meanwhile, so search for it.
            let ok = |after: Duration| {
                let t = now + after;
                t.saturating_duration_since(last) >= gap + extra(l, u.pressure(l, t))
            };
            if !ok(Duration::ZERO) {
                let (mut lo, mut hi) = (Duration::ZERO, gap + Duration::from_millis(l.backoff_cap_ms));
                while hi - lo > Duration::from_millis(50) {
                    let mid = (lo + hi) / 2;
                    if ok(mid) { hi = mid } else { lo = mid }
                }
                until = hi;
            }
        }
        for w in &l.windows {
            let span = Duration::from_millis(w.ms);
            let inside: Vec<Instant> = u.times.iter().copied().filter(|&t| now.saturating_duration_since(t) < span).collect();
            if inside.len() >= w.count {
                // Until enough of them have left the window.
                let frees = inside[inside.len() - w.count] + span;
                until = until.max(frees.saturating_duration_since(now));
            }
        }
        (!until.is_zero()).then_some(until)
    }

    /// Bills `user` for one starting now, if they needn't wait: dated now,
    /// adding `cost` to their pressure (1, or a batch's `batch_cost`).
    /// Otherwise how long they must still wait.
    pub fn try_hold(&mut self, l: &Limits, user: u64, now: Instant, cost: f64) -> Result<Hold, Duration> {
        match self.wait(l, user, now) {
            Some(wait) => Err(wait),
            None => Ok(self.hold(l, user, now, cost)),
        }
    }

    /// Bills `user` for one starting now, waiting or not.
    pub fn hold(&mut self, l: &Limits, user: u64, now: Instant, cost: f64) -> Hold {
        let keep = Duration::from_millis(l.windows.iter().map(|w| w.ms).max().unwrap_or(0).max(l.min_gap_ms));
        // Nobody with nothing left to limit is kept.
        self.users.retain(|_, u| {
            u.times.retain(|&t| now.saturating_duration_since(t) < keep);
            !u.times.is_empty() || u.pressure(l, now) > 0.0
        });
        let u = self.users.entry(user).or_default();
        let top = pressure_cap(l.backoff_unit_ms, l.backoff_growth, l.backoff_cap_ms);
        u.pressure = (u.pressure(l, now) + cost).min(top);
        u.pressure_at = Some(now);
        u.times.push_back(now);
        Hold { user, at: now, cost }
    }

    /// Takes back a bill: the use didn't succeed. Its pressure goes, less
    /// what already drained.
    pub fn release(&mut self, l: &Limits, hold: Hold, now: Instant) {
        let Some(u) = self.users.get_mut(&hold.user) else { return };
        if let Some(i) = u.times.iter().position(|&t| t == hold.at) {
            u.times.remove(i);
        }
        u.pressure = (u.pressure(l, now) - hold.cost).max(0.0);
        u.pressure_at = Some(now);
    }
}

/// The pause between a channel's replies to its messages ([`ChannelLimits`],
/// design.md §6): after one finishes, the next there waits
/// [`Pacer::delay`]; replies elsewhere and slash commands don't. Each
/// message replied to adds 1 to the channel's pressure, which drains steadily.
#[derive(Default)]
pub struct Pacer {
    channels: HashMap<u64, Pace>,
}

#[derive(Default)]
struct Pace {
    pressure: f64,
    pressure_at: Option<Instant>,
    /// The next reply here may start then.
    ready_at: Option<Instant>,
    /// When that pause began (the last reply here finished).
    paused_at: Option<Instant>,
}

impl Pace {
    fn pressure(&self, l: &ChannelLimits, t: Instant) -> f64 {
        let Some(at) = self.pressure_at else { return 0.0 };
        let drained = t.saturating_duration_since(at).as_secs_f64() * 1000.0 / l.backoff_drain_ms as f64;
        (self.pressure - drained).max(0.0)
    }
}

impl Pacer {
    /// The pause at `pressure`: `delay_ms` plus the backoff.
    pub fn delay(l: &ChannelLimits, pressure: f64) -> Duration {
        let extra = (l.backoff_unit_ms as f64 * (l.backoff_growth.powf(pressure) - 1.0)).clamp(0.0, l.backoff_cap_ms as f64);
        Duration::from_millis(l.delay_ms + extra as u64)
    }

    /// A message `channel` will reply to arrived; the channel's pressure now.
    pub fn arrived(&mut self, l: &ChannelLimits, channel: u64, now: Instant) -> f64 {
        let c = self.channels.entry(channel).or_default();
        let top = pressure_cap(l.backoff_unit_ms, l.backoff_growth, l.backoff_cap_ms);
        c.pressure = (c.pressure(l, now) + 1.0).min(top);
        c.pressure_at = Some(now);
        c.pressure
    }

    /// A reply in `channel` finished: the next waits from now.
    pub fn finished(&mut self, l: &ChannelLimits, channel: u64, now: Instant) {
        let c = self.channels.entry(channel).or_default();
        c.ready_at = Some(now + Self::delay(l, c.pressure(l, now)));
        c.paused_at = Some(now);
        // Nothing left to pace is forgotten.
        self.channels.retain(|_, c| c.ready_at.is_some_and(|r| r > now) || c.pressure(l, now) > 0.0);
    }

    /// When the next reply in `channel` may start, if not yet.
    pub fn ready_at(&self, channel: u64, now: Instant) -> Option<Instant> {
        self.channels.get(&channel)?.ready_at.filter(|&r| r > now)
    }

    /// How long a reply to a message that arrived at `arrived` was held back
    /// by `channel`'s latest pause (none if it came after it).
    pub fn held(&self, channel: u64, arrived: Instant) -> Duration {
        let Some(c) = self.channels.get(&channel) else { return Duration::ZERO };
        match (c.paused_at, c.ready_at) {
            (Some(from), Some(until)) => until.saturating_duration_since(arrived.max(from)),
            _ => Duration::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: u64 = 7;

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    #[test]
    fn a_channel_pauses_two_to_seven_seconds() {
        let l = ChannelLimits::default();
        assert_eq!(Pacer::delay(&l, 0.0), secs(2.0));
        assert_eq!(Pacer::delay(&l, 100.0), secs(7.0));
        let mut p = Pacer::default();
        let t = Instant::now();
        assert_eq!(p.ready_at(1, t), None);
        p.arrived(&l, 1, t);
        p.finished(&l, 1, t + secs(5.0));
        // Pressure 1 less what drained in 5 s: a bit over 2 s.
        let r = p.ready_at(1, t + secs(5.0)).expect("paused");
        let pause = r - (t + secs(5.0));
        assert!(pause > secs(2.0) && pause < secs(2.3), "{pause:?}");
        assert_eq!(p.ready_at(2, t), None, "other channels aren't paused");
        // Waiting since before the pause: held all of it; arriving during it,
        // the rest; after it, nothing.
        assert_eq!(p.held(1, t), pause);
        assert_eq!(p.held(1, t + secs(6.0)), pause - secs(1.0));
        assert_eq!(p.held(1, t + secs(9.0)), Duration::ZERO);
        assert_eq!(p.ready_at(1, t + secs(8.0)), None);
        // A flood builds it up to the cap, and no further: a minute after
        // it, the pause is back near its floor.
        for i in 0..30 {
            p.arrived(&l, 1, t + secs(10.0 + i as f64));
        }
        p.finished(&l, 1, t + secs(40.0));
        let pause = p.ready_at(1, t + secs(40.0)).expect("paused") - (t + secs(40.0));
        assert!(pause > secs(6.5) && pause <= secs(7.0), "{pause:?}");
        p.finished(&l, 1, t + secs(100.0));
        let pause = p.ready_at(1, t + secs(100.0)).expect("paused") - (t + secs(100.0));
        assert!(pause < secs(2.1), "{pause:?}");
    }

    #[test]
    fn no_pressure_no_extra() {
        let l = Limits::default();
        assert_eq!(extra(&l, 0.0), Duration::ZERO);
        assert!(extra(&l, 1.0) < secs(0.5));
        assert_eq!(extra(&l, 100.0), Duration::from_millis(l.backoff_cap_ms));
    }

    #[test]
    fn the_first_is_free_then_the_gap() {
        let l = Limits::default();
        let mut lim = Limiter::default();
        let t = Instant::now();
        assert_eq!(lim.wait(&l, ME, t), None);
        lim.hold(&l, ME, t, 1.0);
        let w = lim.wait(&l, ME, t).expect("must wait");
        // 5 s plus a little backoff at pressure ~1.
        assert!(w >= secs(5.0) && w < secs(5.5), "{w:?}");
        assert_eq!(lim.wait(&l, ME, t + secs(6.0)), None);
        assert_eq!(lim.wait(&l, 8, t), None, "others are unaffected");
    }

    #[test]
    fn a_window_holds_them_off() {
        let l = Limits { backoff_unit_ms: 0, ..Limits::default() };
        let mut lim = Limiter::default();
        let t = Instant::now();
        for i in 0..5 {
            lim.hold(&l, ME, t + secs(5.0 * i as f64), 1.0);
        }
        // The 6th at 25 s waits for the first to leave the 30 s window.
        let w = lim.wait(&l, ME, t + secs(25.0)).expect("must wait");
        assert!((w.as_secs_f64() - 5.0).abs() < 0.1, "{w:?}");
    }

    #[test]
    fn a_batch_counts_once_in_windows_but_weighs_on_the_backoff() {
        let l = Limits::default();
        let mut lim = Limiter::default();
        let t = Instant::now();
        lim.hold(&l, ME, t, l.batch_cost);
        let w = lim.wait(&l, ME, t).expect("must wait");
        // About 5 s + 28 s at pressure 10, less what drains meanwhile.
        assert!(w > secs(20.0) && w < secs(33.0), "{w:?}");
        // Drained after 10 × 15 s.
        assert_eq!(lim.wait(&l, ME, t + secs(151.0)), None);
        // Batches past the cap add no lasting pressure: drained as fast.
        let t = t + secs(200.0);
        for i in 0..5 {
            lim.hold(&l, ME, t + secs(i as f64), l.batch_cost);
        }
        let top = pressure_cap(l.backoff_unit_ms, l.backoff_growth, l.backoff_cap_ms);
        let drained = t + secs(4.0 + top * l.backoff_drain_ms as f64 / 1000.0 + 1.0);
        assert_eq!(lim.wait(&l, ME, drained), None);
    }

    #[test]
    fn a_released_hold_leaves_no_trace() {
        let l = Limits::default();
        let mut lim = Limiter::default();
        let t = Instant::now();
        let first = lim.try_hold(&l, ME, t, 1.0).expect("free");
        assert!(lim.try_hold(&l, ME, t, 1.0).is_err(), "billed at the start");
        lim.release(&l, first, t + secs(3.0));
        assert_eq!(lim.wait(&l, ME, t + secs(3.0)), None);
        // A batch released after a minute: its pressure, less what drained, goes.
        let batch = lim.hold(&l, ME, t + secs(10.0), l.batch_cost);
        lim.release(&l, batch, t + secs(70.0));
        assert_eq!(lim.wait(&l, ME, t + secs(70.0)), None);
    }

    #[test]
    fn steady_use_builds_pressure_gently() {
        let l = Limits::default();
        let mut lim = Limiter::default();
        let mut t = Instant::now();
        let mut waits = Vec::new();
        for _ in 0..12 {
            t += lim.wait(&l, ME, t).unwrap_or_default();
            lim.hold(&l, ME, t, 1.0);
            waits.push(lim.wait(&l, ME, t).unwrap_or_default().as_secs_f64());
        }
        assert!(waits.windows(2).all(|w| w[1] >= w[0] - 0.1), "{waits:?}");
        assert!(waits[0] < 6.0 && waits[11] > 6.0, "{waits:?}");
    }
}
