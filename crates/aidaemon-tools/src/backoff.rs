//! Retry delays.

use std::time::Duration;

/// `base * 2^attempt`, capped at `max`.
pub fn exponential_backoff(base: Duration, attempt: u32, max: Duration) -> Duration {
    let multiplier = 1u32.checked_shl(attempt).unwrap_or(u32::MAX);
    base.saturating_mul(multiplier).min(max)
}
