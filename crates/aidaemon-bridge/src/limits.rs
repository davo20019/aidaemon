//! Per-address rate limits and per-route daily byte budgets.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

/// What a rate limit counts.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kind {
    Connect,
    Login,
    Wait,
}

const WINDOW: Duration = Duration::from_secs(60);
/// Addresses tracked at once; past this, expired ones are dropped, and if
/// none are, new addresses are refused (fail closed under a flood).
const MAX_TRACKED: usize = 100_000;

/// Fixed one-minute windows per (kind, address).
#[derive(Default)]
pub(crate) struct RateLimiter {
    entries: HashMap<(Kind, IpAddr), (Instant, u32)>,
}

impl RateLimiter {
    pub(crate) fn allow(&mut self, kind: Kind, ip: IpAddr, limit: u32, now: Instant) -> bool {
        let key = (kind, bucket(ip));
        if !self.entries.contains_key(&key) && self.entries.len() >= MAX_TRACKED {
            self.entries
                .retain(|_, (start, _)| now.duration_since(*start) < WINDOW);
            if self.entries.len() >= MAX_TRACKED {
                return false;
            }
        }
        let entry = self.entries.entry(key).or_insert((now, 0));
        if now.duration_since(entry.0) >= WINDOW {
            *entry = (now, 0);
        }
        if entry.1 >= limit {
            return false;
        }
        entry.1 += 1;
        true
    }
}

/// IPv6 addresses count per /64: one host usually owns the whole prefix.
fn bucket(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(_) => ip,
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return IpAddr::V4(v4);
            }
            let mut segments = v6.segments();
            segments[4..].fill(0);
            IpAddr::V6(segments.into())
        }
    }
}

const DAY: Duration = Duration::from_secs(24 * 60 * 60);

/// Bytes one route forwarded in its current day-long window. Kept by the
/// bridge across daemon logins, so reconnecting does not reset it.
pub(crate) struct Budget {
    start: Instant,
    used: u64,
}

impl Budget {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            start: now,
            used: 0,
        }
    }

    fn roll(&mut self, now: Instant) {
        if now.duration_since(self.start) >= DAY {
            self.start = now;
            self.used = 0;
        }
    }

    /// Count `bytes`; false once the day's budget is spent (the frame that
    /// would overrun it is not forwarded).
    pub(crate) fn charge(&mut self, bytes: u64, limit: u64, now: Instant) -> bool {
        self.roll(now);
        match self.used.checked_add(bytes) {
            Some(total) if total <= limit => {
                self.used = total;
                true
            }
            _ => {
                self.used = limit;
                false
            }
        }
    }

    pub(crate) fn exhausted(&mut self, limit: u64, now: Instant) -> bool {
        self.roll(now);
        self.used >= limit
    }

    pub(crate) fn expired(&self, now: Instant) -> bool {
        now.duration_since(self.start) >= DAY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_resets_each_window() {
        let mut limiter = RateLimiter::default();
        let ip: IpAddr = "192.0.2.1".parse().unwrap();
        let now = Instant::now();
        assert!(limiter.allow(Kind::Login, ip, 2, now));
        assert!(limiter.allow(Kind::Login, ip, 2, now));
        assert!(!limiter.allow(Kind::Login, ip, 2, now));
        assert!(limiter.allow(Kind::Wait, ip, 2, now));
        assert!(limiter.allow(Kind::Login, ip, 2, now + WINDOW));
    }

    #[test]
    fn ipv6_counts_per_prefix() {
        let mut limiter = RateLimiter::default();
        let now = Instant::now();
        let a: IpAddr = "2001:db8::1".parse().unwrap();
        let b: IpAddr = "2001:db8::2".parse().unwrap();
        assert!(limiter.allow(Kind::Connect, a, 1, now));
        assert!(!limiter.allow(Kind::Connect, b, 1, now));
    }

    #[test]
    fn budget_refuses_overrun_and_rolls() {
        let now = Instant::now();
        let mut budget = Budget::new(now);
        assert!(budget.charge(60, 100, now));
        assert!(!budget.charge(60, 100, now));
        assert!(budget.exhausted(100, now));
        assert!(budget.charge(60, 100, now + DAY));
    }
}
