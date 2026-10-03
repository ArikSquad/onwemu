//! Minimal kernel state used by the emulator's HLE services.
//!
//! Time is measured in guest microseconds. CPU cycles are accumulated until a
//! whole microsecond elapses, which lets short instruction bursts and long
//! sleeps use the same clock without rounding every operation independently.

use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
/// The states exposed by the small scheduler model.
pub enum ThreadState {
    /// Thread exists but has not been started.
    Dormant,
    /// Thread is eligible to run.
    Ready,
    /// Thread currently owns guest execution.
    Running,
    /// Thread is blocked until an event or deadline.
    Waiting,
    /// Thread is temporarily prevented from running.
    Suspended,
    /// Thread can no longer run.
    Dead,
}
#[derive(Clone, Debug, Serialize)]
/// Guest thread metadata kept by the scheduler.
pub struct Thread {
    /// Guest-visible thread identifier.
    pub id: u32,
    /// Name supplied when the thread was created.
    pub name: String,
    /// PSP scheduling priority.
    pub priority: u8,
    /// Current scheduler state.
    pub state: ThreadState,
    /// Guest time at which a waiting thread becomes ready.
    pub wake_tick: Option<u64>,
}
#[derive(Default)]
/// A deterministic FIFO scheduler for the currently emulated guest threads.
pub struct Scheduler {
    /// Elapsed guest time in microseconds. PSP services that expose time,
    /// including system time, thread delays, vblank pacing, and audio sample
    /// accounting, all use this value for comparisons and deadlines.
    now: u64,
    /// Cycles executed since the last whole microsecond. Guest code progresses
    /// the clock by executing instructions, so fractional microseconds are
    /// accumulated here between whole-unit steps.
    cycle_remainder: u32,
    next_id: u32,
    threads: BTreeMap<u32, Thread>,
    ready: VecDeque<u32>,
}
impl Scheduler {
    /// Emulated Allegrex clock rate. Kernel microsecond timers are derived from
    /// this 333 MHz clock.
    pub const CYCLES_PER_MICROSECOND: u64 = 333;
    /// Return the current guest time in microseconds.
    pub fn now(&self) -> u64 {
        self.now
    }

    /// Create a dormant thread and return its guest-visible identifier.
    pub fn create(&mut self, name: String, priority: u8) -> u32 {
        self.next_id += 1;
        let id = self.next_id;
        self.threads.insert(
            id,
            Thread {
                id,
                name,
                priority,
                state: ThreadState::Dormant,
                wake_tick: None,
            },
        );
        id
    }
    /// Move a known thread to the ready queue.
    ///
    /// Starting an unknown id is a normal failed lookup and returns `false`.
    pub fn start(&mut self, id: u32) -> bool {
        if let Some(t) = self.threads.get_mut(&id) {
            t.state = ThreadState::Ready;
            self.ready.push_back(id);
            true
        } else {
            false
        }
    }
    /// Advance guest time by executed CPU cycles.
    pub fn advance(&mut self, cycles: u64) {
        let total = self.cycle_remainder as u64 + cycles;
        self.now += total / Self::CYCLES_PER_MICROSECOND;
        self.cycle_remainder = (total % Self::CYCLES_PER_MICROSECOND) as u32;
        self.wake_due_threads();
    }
    /// Jump guest time forward by a microsecond delta.
    ///
    /// The frontend uses this when every runnable thread is blocked until a
    /// deadline.
    pub fn advance_micros(&mut self, micros: u64) {
        self.now = self.now.saturating_add(micros);
        self.wake_due_threads();
    }
    fn wake_due_threads(&mut self) {
        for t in self.threads.values_mut() {
            if t.state == ThreadState::Waiting && t.wake_tick.is_some_and(|x| x <= self.now) {
                t.state = ThreadState::Ready;
                t.wake_tick = None;
                self.ready.push_back(t.id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_ids_are_stable_and_unknown_threads_cannot_start() {
        let mut scheduler = Scheduler::default();
        let first = scheduler.create("main".into(), 10);
        let second = scheduler.create("audio".into(), 20);
        assert_eq!((first, second), (1, 2));
        assert_eq!(scheduler.threads[&first].state, ThreadState::Dormant);
        assert!(scheduler.start(first));
        assert_eq!(scheduler.threads[&first].state, ThreadState::Ready);
        assert!(!scheduler.start(99));
    }

    #[test]
    fn cycle_time_keeps_fractional_microseconds_until_the_next_tick() {
        let mut scheduler = Scheduler::default();
        scheduler.advance(332);
        assert_eq!(scheduler.now(), 0);
        scheduler.advance(1);
        assert_eq!(scheduler.now(), 1);
        scheduler.advance_micros(9);
        assert_eq!(scheduler.now(), 10);
    }

    #[test]
    fn due_waiting_threads_become_ready_without_waking_early() {
        let mut scheduler = Scheduler::default();
        let early = scheduler.create("early".into(), 10);
        let due = scheduler.create("due".into(), 10);
        scheduler.threads.get_mut(&early).unwrap().state = ThreadState::Waiting;
        scheduler.threads.get_mut(&early).unwrap().wake_tick = Some(20);
        scheduler.threads.get_mut(&due).unwrap().state = ThreadState::Waiting;
        scheduler.threads.get_mut(&due).unwrap().wake_tick = Some(10);

        scheduler.advance_micros(9);
        assert_eq!(scheduler.threads[&early].state, ThreadState::Waiting);
        assert_eq!(scheduler.threads[&due].state, ThreadState::Waiting);
        scheduler.advance_micros(1);
        assert_eq!(scheduler.threads[&early].state, ThreadState::Waiting);
        assert_eq!(scheduler.threads[&due].state, ThreadState::Ready);
        scheduler.advance_micros(10);
        assert_eq!(scheduler.threads[&early].state, ThreadState::Ready);
    }
}
