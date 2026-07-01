#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ZeroUnitStyle {
    Keep,
    Trim,
}

pub(crate) fn compact_seconds(secs: i64, zero_style: ZeroUnitStyle) -> String {
    let secs = secs.max(0);
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        let mins = secs / 60;
        let remaining_secs = secs % 60;
        if zero_style == ZeroUnitStyle::Trim && remaining_secs == 0 {
            format!("{}m", mins)
        } else {
            format!("{}m {}s", mins, remaining_secs)
        }
    } else {
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        if zero_style == ZeroUnitStyle::Trim && mins == 0 {
            format!("{}h", hours)
        } else {
            format!("{}h {}m", hours, mins)
        }
    }
}

pub(crate) fn compact_chrono_duration(
    duration: chrono::Duration,
    zero_style: ZeroUnitStyle,
) -> String {
    compact_seconds(duration.num_seconds(), zero_style)
}

pub(crate) fn compact_elapsed_timestamps(
    started_at: Option<&str>,
    completed_at: Option<&str>,
    zero_style: ZeroUnitStyle,
) -> String {
    let Some(start_raw) = started_at else {
        return "n/a".to_string();
    };
    let Some(started) = parse_ts(start_raw) else {
        return "n/a".to_string();
    };
    let Some(end_raw) = completed_at else {
        return "running".to_string();
    };
    let Some(ended) = parse_ts(end_raw) else {
        return "n/a".to_string();
    };

    compact_chrono_duration(ended - started, zero_style)
}

fn parse_ts(ts: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|d| d.with_timezone(&chrono::Utc))
}

#[cfg(test)]
mod tests {
    use chrono::Duration;

    use super::*;

    #[test]
    fn compact_seconds_preserves_zero_units() {
        assert_eq!(compact_seconds(40, ZeroUnitStyle::Keep), "40s");
        assert_eq!(compact_seconds(60, ZeroUnitStyle::Keep), "1m 0s");
        assert_eq!(compact_seconds(3600, ZeroUnitStyle::Keep), "1h 0m");
    }

    #[test]
    fn compact_seconds_can_trim_zero_units() {
        assert_eq!(compact_seconds(60, ZeroUnitStyle::Trim), "1m");
        assert_eq!(compact_seconds(90, ZeroUnitStyle::Trim), "1m 30s");
        assert_eq!(compact_seconds(3600, ZeroUnitStyle::Trim), "1h");
        assert_eq!(compact_seconds(3660, ZeroUnitStyle::Trim), "1h 1m");
    }

    #[test]
    fn compact_chrono_duration_clamps_negative_to_zero() {
        assert_eq!(
            compact_chrono_duration(Duration::seconds(-5), ZeroUnitStyle::Keep),
            "0s"
        );
    }

    #[test]
    fn compact_elapsed_timestamps_handles_missing_and_running() {
        assert_eq!(
            compact_elapsed_timestamps(None, None, ZeroUnitStyle::Keep),
            "n/a"
        );
        assert_eq!(
            compact_elapsed_timestamps(Some("2026-01-01T00:00:00Z"), None, ZeroUnitStyle::Keep),
            "running"
        );
        assert_eq!(
            compact_elapsed_timestamps(
                Some("2026-01-01T00:00:00Z"),
                Some("2026-01-01T01:02:03Z"),
                ZeroUnitStyle::Keep
            ),
            "1h 2m"
        );
    }
}
