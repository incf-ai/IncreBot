//! The per-card state machine (gpu-plan.md §3), as plain functions of monitor
//! readings and the clock, so it is tested with a fake clock.
//!
//! ```text
//!         start / free for hold-off
//!  Free ───────────────────────────▶ Loading ──▶ Ready (active or standby)
//!   ▲                                   │             │
//!   │                                   │contended    │contended
//!   │  hold-off elapsed, still free     ▼             ▼
//!   └──────────────────────────── Backoff ◀──── (free everything)
//! ```
//!
//! The hold-off doubles on flaps: when the card is contended again during
//! Backoff after having been free, and when it is contended again soon after
//! a Backoff ended. It returns to the base after a quiet spell ([`QUIET`]).

use std::time::{Duration, Instant};

/// Without contention for this long, the hold-off is back to its base value.
pub const QUIET: Duration = Duration::from_secs(600);

/// The hold-off never grows beyond this.
pub const HOLDOFF_CAP: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Nothing of ours on the card.
    Free,
    /// Weights uploading on a background thread.
    Loading,
    /// Weights on the card: usable for jobs.
    Ready,
    /// Another program used the card; we hold nothing and wait until it has been
    /// free for the hold-off.
    Backoff,
}

/// What the caller must do after a [`Policy::tick`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Start uploading the weights.
    Load,
    /// Stop using the card and free every allocation on it now.
    Release,
}

#[derive(Debug)]
pub struct Policy {
    state: State,
    base: Duration,
    holdoff: Duration,
    /// In Backoff: since when the card has been free (`None` while contended).
    free_since: Option<Instant>,
    /// When the last Backoff ended.
    backoff_ended: Option<Instant>,
}

impl Policy {
    pub fn new(base: Duration) -> Self {
        Self {
            state: State::Free,
            base,
            holdoff: base,
            free_since: None,
            backoff_ended: None,
        }
    }

    pub fn state(&self) -> State {
        self.state
    }

    pub fn holdoff(&self) -> Duration {
        self.holdoff
    }

    /// In Backoff: how long until the card may be used again if it stays free.
    pub fn backoff_left(&self, now: Instant) -> Option<Duration> {
        (self.state == State::Backoff).then(|| match self.free_since {
            Some(t) => self
                .holdoff
                .saturating_sub(now.saturating_duration_since(t)),
            None => self.holdoff,
        })
    }

    /// One monitor reading. `want_loaded` says whether this card should hold the
    /// weights at all (standby disabled and a better card ready: no).
    pub fn tick(&mut self, now: Instant, contended: bool, want_loaded: bool) -> Option<Action> {
        match self.state {
            State::Free if contended => {
                self.enter_backoff(now);
                None
            }
            State::Free if want_loaded => {
                self.state = State::Loading;
                Some(Action::Load)
            }
            State::Free => None,
            State::Loading | State::Ready if contended => {
                self.enter_backoff(now);
                Some(Action::Release)
            }
            State::Ready if !want_loaded => {
                self.state = State::Free;
                Some(Action::Release)
            }
            State::Loading | State::Ready => None,
            State::Backoff if contended => {
                // Free again, then contended again: a flap.
                if self.free_since.take().is_some() {
                    self.double();
                }
                None
            }
            State::Backoff => {
                let since = *self.free_since.get_or_insert(now);
                if now.saturating_duration_since(since) < self.holdoff {
                    return None;
                }
                self.state = State::Free;
                self.backoff_ended = Some(now);
                self.tick(now, false, want_loaded)
            }
        }
    }

    /// The upload finished (`ok`) or failed.
    pub fn loaded(&mut self, now: Instant, ok: bool) {
        if self.state != State::Loading {
            return;
        }
        if ok {
            self.state = State::Ready;
        } else {
            self.enter_backoff(now);
        }
    }

    /// The card failed while in use (device lost, out of memory): treated like
    /// contention. Returns the action to take.
    pub fn failed(&mut self, now: Instant) -> Option<Action> {
        match self.state {
            State::Loading | State::Ready => {
                self.enter_backoff(now);
                Some(Action::Release)
            }
            _ => None,
        }
    }

    fn enter_backoff(&mut self, now: Instant) {
        let recent = self
            .backoff_ended
            .is_some_and(|t| now.saturating_duration_since(t) < QUIET);
        if recent {
            self.double();
        } else {
            self.holdoff = self.base;
        }
        self.state = State::Backoff;
        self.free_since = None;
    }

    fn double(&mut self) {
        self.holdoff = (self.holdoff * 2).min(HOLDOFF_CAP.max(self.base));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: Duration = Duration::from_secs(30);

    fn secs(t0: Instant, s: f64) -> Instant {
        t0 + Duration::from_secs_f64(s)
    }

    #[test]
    fn free_card_loads_and_contention_releases() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        assert_eq!(p.tick(t0, false, true), Some(Action::Load));
        assert_eq!(p.state(), State::Loading);
        p.loaded(secs(t0, 1.0), true);
        assert_eq!(p.state(), State::Ready);
        assert_eq!(p.tick(secs(t0, 2.0), false, true), None);
        assert_eq!(p.tick(secs(t0, 3.0), true, true), Some(Action::Release));
        assert_eq!(p.state(), State::Backoff);
        assert_eq!(p.holdoff(), BASE);
    }

    #[test]
    fn contention_while_loading_releases() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        p.tick(t0, false, true);
        assert_eq!(p.tick(secs(t0, 0.5), true, true), Some(Action::Release));
        // A late completion of the aborted upload changes nothing.
        p.loaded(secs(t0, 0.6), true);
        assert_eq!(p.state(), State::Backoff);
    }

    #[test]
    fn backoff_waits_for_holdoff_of_continuous_freedom() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        p.tick(t0, true, true);
        assert_eq!(p.state(), State::Backoff);
        // Contended for a minute: the timer doesn't run, and no doubling.
        for s in 1..60 {
            assert_eq!(p.tick(secs(t0, s as f64), true, true), None);
        }
        assert_eq!(p.holdoff(), BASE);
        // Free from 60 s: loads at 90 s, not before.
        assert_eq!(p.tick(secs(t0, 60.0), false, true), None);
        assert_eq!(
            p.backoff_left(secs(t0, 70.0)),
            Some(Duration::from_secs(20))
        );
        assert_eq!(p.tick(secs(t0, 89.9), false, true), None);
        assert_eq!(p.tick(secs(t0, 90.0), false, true), Some(Action::Load));
    }

    #[test]
    fn flaps_double_the_holdoff_up_to_the_cap() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        p.tick(t0, true, true);
        // Free for 10 s, then contended again: 60 s.
        p.tick(secs(t0, 1.0), false, true);
        p.tick(secs(t0, 11.0), true, true);
        assert_eq!(p.holdoff(), Duration::from_secs(60));
        let mut t = 11.0;
        for _ in 0..10 {
            t += 1.0;
            p.tick(secs(t0, t), false, true);
            t += 1.0;
            p.tick(secs(t0, t), true, true);
        }
        assert_eq!(p.holdoff(), HOLDOFF_CAP);
    }

    #[test]
    fn contention_soon_after_returning_doubles_and_quiet_resets() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        p.tick(t0, true, true);
        p.tick(secs(t0, 1.0), false, true);
        assert_eq!(p.tick(secs(t0, 31.0), false, true), Some(Action::Load));
        p.loaded(secs(t0, 32.0), true);
        // Taken again a minute later: a flap, 60 s.
        assert_eq!(p.tick(secs(t0, 91.0), true, true), Some(Action::Release));
        assert_eq!(p.holdoff(), Duration::from_secs(60));
        p.tick(secs(t0, 92.0), false, true);
        assert_eq!(p.tick(secs(t0, 152.0), false, true), Some(Action::Load));
        p.loaded(secs(t0, 153.0), true);
        // Taken again only after a long quiet spell: back to the base.
        let late = 152.0 + QUIET.as_secs_f64() + 1.0;
        assert_eq!(p.tick(secs(t0, late), true, true), Some(Action::Release));
        assert_eq!(p.holdoff(), BASE);
    }

    #[test]
    fn unwanted_cards_stay_free_or_unload() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        assert_eq!(p.tick(t0, false, false), None);
        assert_eq!(p.state(), State::Free);
        assert_eq!(p.tick(secs(t0, 1.0), false, true), Some(Action::Load));
        p.loaded(secs(t0, 2.0), true);
        assert_eq!(p.tick(secs(t0, 3.0), false, false), Some(Action::Release));
        assert_eq!(p.state(), State::Free);
    }

    #[test]
    fn failures_and_failed_uploads_back_off() {
        let t0 = Instant::now();
        let mut p = Policy::new(BASE);
        p.tick(t0, false, true);
        p.loaded(secs(t0, 1.0), false);
        assert_eq!(p.state(), State::Backoff);
        let mut p = Policy::new(BASE);
        p.tick(t0, false, true);
        p.loaded(secs(t0, 1.0), true);
        assert_eq!(p.failed(secs(t0, 2.0)), Some(Action::Release));
        assert_eq!(p.state(), State::Backoff);
        assert_eq!(p.failed(secs(t0, 3.0)), None);
    }
}
