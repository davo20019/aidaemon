use std::time::Duration;

use reqwest::header::{HeaderMap, RETRY_AFTER};
use tracing::warn;

const DEFAULT_MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

pub(crate) fn bounded_retry_after(delay: Duration, max_delay: Duration) -> Duration {
    delay.max(Duration::from_secs(1)).min(max_delay)
}

pub(crate) fn retry_after_from_headers(
    headers: &HeaderMap,
    max_delay: Duration,
) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .map(|secs| bounded_retry_after(Duration::from_secs(secs), max_delay))
}

pub(crate) async fn sleep_after_rate_limit(surface: &str, delay: Duration) {
    warn!(
        surface,
        retry_after_secs = delay.as_secs(),
        "Outbound channel rate limited; retrying after delay"
    );
    tokio::time::sleep(delay).await;
}

pub(crate) fn default_retry_after_cap() -> Duration {
    DEFAULT_MAX_RETRY_AFTER
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};

    use super::*;

    #[test]
    fn retry_after_header_parses_seconds_and_clamps() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("999"));

        assert_eq!(
            retry_after_from_headers(&headers, Duration::from_secs(60)),
            Some(Duration::from_secs(60))
        );
    }

    #[test]
    fn retry_after_header_ignores_invalid_values() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, HeaderValue::from_static("not-seconds"));

        assert_eq!(
            retry_after_from_headers(&headers, Duration::from_secs(60)),
            None
        );
    }

    #[test]
    fn retry_after_duration_adds_small_floor_for_zero() {
        assert_eq!(
            bounded_retry_after(Duration::from_secs(0), Duration::from_secs(60)),
            Duration::from_secs(1)
        );
    }
}
