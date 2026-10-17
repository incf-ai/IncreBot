//! The reply queue (design.md §6): first in, first out, at most `depth` waiting
//! (the dispatcher adds its own limits per channel and for interactions, and
//! may pass over a job whose channel is pausing: [`pop_where`](JobQueue::pop_where)).
//!
//! Every triggering message waits for its own reply; a newer message never
//! replaces or stops an older one. When no slot is free, newer
//! triggers are turned away (`Full`) and the caller ignores them. Prompts are
//! built when a job is dispatched, so a queued job still sees everything said
//! until then. An admin can [`clear`](JobQueue::clear) the waiting jobs.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueueKey {
    Channel(u64),
    /// A one-shot `/ask`, identified by its interaction id.
    Ask(u64),
}

#[derive(Debug, PartialEq)]
pub enum Enqueued {
    /// Added; `ahead` jobs wait before it (the running job not included).
    Queued { ahead: usize },
    /// No free slot: the trigger was not queued.
    Full,
}

pub struct JobQueue<T> {
    waiting: VecDeque<(QueueKey, T)>,
    depth: usize,
}

impl<T> JobQueue<T> {
    pub fn new(depth: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            depth,
        }
    }

    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Queues `trigger` behind the waiting jobs, or returns `Full`.
    pub fn enqueue(&mut self, key: QueueKey, trigger: T) -> Enqueued {
        if self.waiting.len() >= self.depth {
            return Enqueued::Full;
        }
        self.waiting.push_back((key, trigger));
        Enqueued::Queued {
            ahead: self.waiting.len() - 1,
        }
    }

    pub fn pop(&mut self) -> Option<(QueueKey, T)> {
        self.waiting.pop_front()
    }

    /// Removes the first waiting job `ready` accepts, passing over the others.
    pub fn pop_where(&mut self, ready: impl Fn(&T) -> bool) -> Option<(QueueKey, T)> {
        let pos = self.waiting.iter().position(|(_, t)| ready(t))?;
        self.waiting.remove(pos)
    }

    /// How many waiting jobs `pred` accepts.
    pub fn count_where(&self, pred: impl Fn(&T) -> bool) -> usize {
        self.waiting.iter().filter(|(_, t)| pred(t)).count()
    }

    /// Removes the first waiting job whose trigger matches (cancel while queued).
    pub fn remove_where(&mut self, pred: impl Fn(&T) -> bool) -> Option<(QueueKey, T)> {
        let pos = self.waiting.iter().position(|(_, t)| pred(t))?;
        self.waiting.remove(pos)
    }

    /// Drops every waiting job (the admin "clear queue"); returns them.
    pub fn clear(&mut self) -> Vec<(QueueKey, T)> {
        self.waiting.drain(..).collect()
    }

    /// Waiting jobs with their positions (0 = next), for updating "N ahead".
    pub fn iter(&self) -> impl Iterator<Item = (usize, &(QueueKey, T))> {
        self.waiting.iter().enumerate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_can_be_passed_over() {
        let mut q = JobQueue::new(4);
        q.enqueue(QueueKey::Channel(1), 1);
        q.enqueue(QueueKey::Channel(2), 2);
        q.enqueue(QueueKey::Channel(1), 3);
        assert_eq!(q.count_where(|&t| t != 2), 2);
        assert_eq!(q.pop_where(|&t| t != 1), Some((QueueKey::Channel(2), 2)));
        assert_eq!(q.pop(), Some((QueueKey::Channel(1), 1)));
        assert_eq!(q.pop_where(|_| false), None);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn first_in_first_out_and_newer_never_replace_older() {
        let mut q = JobQueue::new(4);
        assert_eq!(
            q.enqueue(QueueKey::Channel(1), "a1"),
            Enqueued::Queued { ahead: 0 }
        );
        assert_eq!(
            q.enqueue(QueueKey::Channel(1), "a2"),
            Enqueued::Queued { ahead: 1 }
        );
        assert_eq!(
            q.enqueue(QueueKey::Ask(9), "ask"),
            Enqueued::Queued { ahead: 2 }
        );
        assert_eq!(
            q.enqueue(QueueKey::Channel(2), "b1"),
            Enqueued::Queued { ahead: 3 }
        );
        assert_eq!(q.pop(), Some((QueueKey::Channel(1), "a1")));
        assert_eq!(q.pop(), Some((QueueKey::Channel(1), "a2")));
    }

    #[test]
    fn full_queue_turns_newer_triggers_away() {
        let mut q = JobQueue::new(2);
        q.enqueue(QueueKey::Channel(1), 1);
        q.enqueue(QueueKey::Channel(1), 2);
        assert_eq!(q.enqueue(QueueKey::Channel(1), 3), Enqueued::Full);
        assert_eq!(q.enqueue(QueueKey::Ask(5), 4), Enqueued::Full);
        assert_eq!(
            q.iter().map(|(_, (_, t))| *t).collect::<Vec<_>>(),
            [1, 2],
            "older ones kept"
        );
        q.pop();
        assert_eq!(
            q.enqueue(QueueKey::Channel(1), 5),
            Enqueued::Queued { ahead: 1 }
        );
    }

    #[test]
    fn cancel_removes_one_and_clear_removes_all() {
        let mut q = JobQueue::new(4);
        for t in 1..=3 {
            q.enqueue(QueueKey::Ask(t), t);
        }
        assert_eq!(q.remove_where(|&t| t == 2), Some((QueueKey::Ask(2), 2)));
        assert_eq!(q.clear().len(), 2);
        assert!(q.is_empty());
        assert_eq!(q.depth(), 4);
    }
}
