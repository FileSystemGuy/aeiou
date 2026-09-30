//! Cross-actor coordination behind the `Coordinator` trait (`NAPKIN_MATH.md` §8.A): a start
//! gate, named barriers, and the end-of-run reduction. `Local` is the in-process
//! implementation for a single host; the TCP coordinator for several hosts is not written yet
//! and will implement the same trait.
//!
//! A barrier's participants are fixed before the run: every instance of every actor template
//! whose body contains `barrier {scope}` (outside any `parallel` or `loader`). An instance
//! that finishes leaves its barriers; if the remaining arrivals then complete a generation,
//! it is released and counted as a departure release, which the report shows because it means
//! the abstract's instances did not all hit the barrier the same number of times.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

pub trait Coordinator: Send + Sync {
    /// Wait until every participant of `scope` has arrived. Returns the time spent waiting.
    fn barrier(&self, scope: &str, aborted: &AtomicBool) -> Result<Duration>;
    /// This participant will never arrive at `scope` again.
    fn leave(&self, scope: &str);
    /// Releases by departure so far, per scope.
    fn departure_releases(&self) -> Vec<(String, u64)>;
}

struct Bar {
    expected: usize,
    arrived: usize,
    generation: u64,
    departure_releases: u64,
}

pub struct Local {
    bars: Mutex<HashMap<String, Bar>>,
    cv: Condvar,
}

impl Local {
    pub fn new(participants: &[(String, usize)]) -> Self {
        let bars = participants
            .iter()
            .map(|(s, n)| (s.clone(), Bar { expected: *n, arrived: 0, generation: 0, departure_releases: 0 }))
            .collect();
        Local { bars: Mutex::new(bars), cv: Condvar::new() }
    }
}

impl Coordinator for Local {
    fn barrier(&self, scope: &str, aborted: &AtomicBool) -> Result<Duration> {
        let t = Instant::now();
        let mut bars = self.bars.lock().unwrap();
        let Some(bar) = bars.get_mut(scope) else { bail!("barrier `{scope}`: no participants registered") };
        bar.arrived += 1;
        if bar.arrived >= bar.expected {
            bar.arrived = 0;
            bar.generation += 1;
            self.cv.notify_all();
            return Ok(t.elapsed());
        }
        let my_gen = bar.generation;
        loop {
            let (guard, _) = self.cv.wait_timeout(bars, Duration::from_millis(50)).unwrap();
            bars = guard;
            if bars.get(scope).map_or(true, |b| b.generation != my_gen) {
                return Ok(t.elapsed());
            }
            if aborted.load(Ordering::Relaxed) {
                bail!("barrier `{scope}`: run aborted while waiting");
            }
        }
    }

    fn leave(&self, scope: &str) {
        let mut bars = self.bars.lock().unwrap();
        if let Some(bar) = bars.get_mut(scope) {
            bar.expected = bar.expected.saturating_sub(1);
            if bar.arrived > 0 && bar.arrived >= bar.expected {
                bar.arrived = 0;
                bar.generation += 1;
                bar.departure_releases += 1;
                self.cv.notify_all();
            }
        }
    }

    fn departure_releases(&self) -> Vec<(String, u64)> {
        let bars = self.bars.lock().unwrap();
        let mut v: Vec<_> = bars.iter().filter(|(_, b)| b.departure_releases > 0).map(|(s, b)| (s.clone(), b.departure_releases)).collect();
        v.sort();
        v
    }
}
