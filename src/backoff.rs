use std::time::Duration;

use rand::Rng;

pub(crate) fn exponential_backoff(base: Duration, attempt: u32, max: Duration) -> Duration {
    let multiplier = 1u32.checked_shl(attempt).unwrap_or(u32::MAX);
    base.saturating_mul(multiplier).min(max)
}

pub(crate) fn jittered_delay(base: Duration, jitter_fraction: f64) -> Duration {
    if base.is_zero() || jitter_fraction <= 0.0 {
        return base;
    }

    let max_jitter = base.mul_f64(jitter_fraction);
    let max_jitter_millis = max_jitter.as_millis().min(u64::MAX as u128) as u64;
    if max_jitter_millis == 0 {
        return base;
    }

    let jitter_millis = rand::thread_rng().gen_range(0..=max_jitter_millis);
    base.saturating_add(Duration::from_millis(jitter_millis))
}

pub(crate) struct ChannelReconnectBackoff {
    initial: Duration,
    max: Duration,
    stable_threshold: Duration,
    current: Duration,
}

impl ChannelReconnectBackoff {
    pub(crate) fn new(initial: Duration, max: Duration, stable_threshold: Duration) -> Self {
        Self {
            initial,
            max,
            stable_threshold,
            current: initial,
        }
    }

    pub(crate) fn next_base_delay(&mut self, ran_for: Duration) -> Duration {
        if ran_for >= self.stable_threshold {
            self.current = self.initial;
        }

        let delay = self.current;
        self.current = self.current.saturating_mul(2).min(self.max);
        delay
    }

    pub(crate) fn next_jittered_delay(
        &mut self,
        ran_for: Duration,
        jitter_fraction: f64,
    ) -> Duration {
        jittered_delay(self.next_base_delay(ran_for), jitter_fraction)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn exponential_backoff_caps_at_max_delay() {
        assert_eq!(
            exponential_backoff(Duration::from_secs(5), 0, Duration::from_secs(60)),
            Duration::from_secs(5)
        );
        assert_eq!(
            exponential_backoff(Duration::from_secs(5), 3, Duration::from_secs(60)),
            Duration::from_secs(40)
        );
        assert_eq!(
            exponential_backoff(Duration::from_secs(5), 8, Duration::from_secs(60)),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn reconnect_backoff_resets_after_stable_run() {
        let mut backoff = ChannelReconnectBackoff::new(
            Duration::from_secs(5),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );

        assert_eq!(
            backoff.next_base_delay(Duration::from_secs(1)),
            Duration::from_secs(5)
        );
        assert_eq!(
            backoff.next_base_delay(Duration::from_secs(1)),
            Duration::from_secs(10)
        );
        assert_eq!(
            backoff.next_base_delay(Duration::from_secs(60)),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn jittered_delay_never_shortens_base_delay() {
        let base = Duration::from_secs(10);
        let jittered = jittered_delay(base, 1.0);
        assert!(jittered >= base);
        assert!(jittered <= Duration::from_secs(20));
    }
}
