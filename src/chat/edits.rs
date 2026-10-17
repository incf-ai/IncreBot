//! When a streaming reply's placeholder is edited (design.md §6).
//!
//! The first edit happens as soon as text arrives (the user has already waited
//! through the prefill), then one every [`GAP`]. No edit is made once [`CUTOFF`]
//! has passed since the first text: the message then waits for the final text.
//! With the defaults that is edits at 0, 1.5, 3, 4.5, … s, up to 30 s: a
//! 75-token reply (~14–15 s on Q4_K_M) streams in full, with headroom.

use std::time::{Duration, Instant};

pub const GAP: Duration = Duration::from_millis(1500);
pub const CUTOFF: Duration = Duration::from_secs(30);

#[derive(Debug)]
pub struct EditSchedule {
    gap: Duration,
    cutoff: Duration,
    /// When the first text arrived.
    start: Option<Instant>,
    /// When the next edit may happen; `None` before text or after the cutoff.
    next: Option<Instant>,
}

impl Default for EditSchedule {
    fn default() -> Self {
        Self::new(GAP, CUTOFF)
    }
}

impl EditSchedule {
    pub fn new(gap: Duration, cutoff: Duration) -> Self {
        Self {
            gap,
            cutoff,
            start: None,
            next: None,
        }
    }

    /// Records that text arrived at `now`. The first call makes an edit due at once.
    pub fn text_arrived(&mut self, now: Instant) {
        if self.start.is_none() {
            self.start = Some(now);
            self.next = Some(now);
        }
    }

    /// The pending edit time, unless the cutoff has passed by `now`.
    fn pending(&self, now: Instant) -> Option<Instant> {
        let start = self.start?;
        self.next.filter(|_| now <= start + self.cutoff)
    }

    /// Whether an edit should be made at `now`.
    pub fn due(&self, now: Instant) -> bool {
        self.pending(now).is_some_and(|n| now >= n)
    }

    /// Records an edit made at `now` and schedules the next one, unless it would
    /// fall past the cutoff.
    pub fn edited(&mut self, now: Instant) {
        let Some(start) = self.start else { return };
        let next = now + self.gap;
        self.next = (next <= start + self.cutoff).then_some(next);
    }

    /// How long until the next edit is due (zero if it is due already); `None`
    /// when no edit is pending, so the caller can just wait for events.
    pub fn wait(&self, now: Instant) -> Option<Duration> {
        self.pending(now).map(|n| n.saturating_duration_since(now))
    }

    /// Whether the cutoff has ended streaming edits for this reply.
    pub fn stopped(&self, now: Instant) -> bool {
        self.start.is_some() && self.pending(now).is_none()
    }

    /// Starts over for a new message (roll-over at ~1,900 characters), keeping
    /// the cutoff counted from the reply's first text.
    pub fn new_message(&mut self, now: Instant) {
        if let Some(start) = self.start
            && now < start + self.cutoff
        {
            self.next = Some(now);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Edit times in ms after the first text when text streams in continuously,
    /// polled every 50 ms, and every due edit is made.
    fn edit_times(schedule: &mut EditSchedule, t0: Instant, until_ms: u64) -> Vec<u64> {
        let mut times = Vec::new();
        for t in (0..=until_ms).step_by(50) {
            let now = t0 + ms(t);
            schedule.text_arrived(now);
            if schedule.due(now) {
                schedule.edited(now);
                times.push(t);
            }
        }
        times
    }

    #[test]
    fn nothing_is_due_before_text() {
        let s = EditSchedule::default();
        let now = Instant::now();
        assert!(!s.due(now));
        assert_eq!(s.wait(now), None);
        assert!(!s.stopped(now));
    }

    #[test]
    fn every_1_5_s_until_30_s() {
        let t0 = Instant::now();
        let mut s = EditSchedule::default();
        let times = edit_times(&mut s, t0, 60_000);
        let expected: Vec<u64> = (0..=20).map(|i| i * 1500).collect();
        assert_eq!(times, expected, "0, 1.5, 3, … 30 s");
        assert!(s.stopped(t0 + ms(31_000)));
        assert!(!s.due(t0 + ms(40_000)));
        assert_eq!(s.wait(t0 + ms(40_000)), None);
    }

    #[test]
    fn wait_counts_down_to_the_next_edit() {
        let t0 = Instant::now();
        let mut s = EditSchedule::default();
        s.text_arrived(t0);
        assert_eq!(s.wait(t0), Some(Duration::ZERO));
        s.edited(t0);
        assert_eq!(s.wait(t0 + ms(500)), Some(ms(1000)));
        assert_eq!(s.wait(t0 + ms(2000)), Some(Duration::ZERO));
        assert!(s.due(t0 + ms(1500)));
    }

    #[test]
    fn a_late_edit_pushes_the_next_one_back() {
        // Edits happen when text (or the timer) arrives, not exactly on schedule.
        let t0 = Instant::now();
        let mut s = EditSchedule::default();
        s.text_arrived(t0);
        s.edited(t0);
        s.edited(t0 + ms(4000));
        assert!(!s.due(t0 + ms(5400)));
        assert!(s.due(t0 + ms(5500)));
    }

    #[test]
    fn roll_over_gets_an_edit_until_the_cutoff() {
        let t0 = Instant::now();
        let mut s = EditSchedule::default();
        s.text_arrived(t0);
        s.edited(t0);
        s.new_message(t0 + ms(200));
        assert!(s.due(t0 + ms(200)));
        let mut late = EditSchedule::default();
        late.text_arrived(t0);
        late.edited(t0);
        late.new_message(t0 + ms(35_000));
        assert!(!late.due(t0 + ms(35_000)));
    }

    #[test]
    fn a_pending_edit_lapses_at_the_cutoff() {
        // Text stalls after the first edit: the edit due at 1.5 s is never made,
        // and must not fire (or keep the caller's timer spinning) after 30 s.
        let t0 = Instant::now();
        let mut s = EditSchedule::default();
        s.text_arrived(t0);
        s.edited(t0);
        assert!(s.due(t0 + ms(29_000)));
        assert!(!s.due(t0 + ms(30_500)));
        assert_eq!(s.wait(t0 + ms(30_500)), None);
        assert!(s.stopped(t0 + ms(30_500)));
    }

    #[test]
    fn custom_schedules() {
        let t0 = Instant::now();
        let mut s = EditSchedule::new(ms(4000), ms(10_000));
        assert_eq!(edit_times(&mut s, t0, 20_000), [0, 4000, 8000]);
    }
}
