//! Times as the owner reads them, in their own time zone: "now", "5m",
//! "3:42 PM", "Yesterday", "Mon, Mar 3".

use chrono::{DateTime, Datelike, NaiveDateTime};

/// The local offset from UTC, in milliseconds.
pub fn local_offset_ms() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        // Minutes behind UTC (300 in New York in winter).
        -(js_sys::Date::new_0().get_timezone_offset() as i64) * 60_000
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        i64::from(chrono::Local::now().offset().local_minus_utc()) * 1000
    }
}

fn local(at_ms: u64) -> Option<NaiveDateTime> {
    let at = i64::try_from(at_ms).ok()?.checked_add(local_offset_ms())?;
    DateTime::from_timestamp_millis(at).map(|t| t.naive_utc())
}

/// The local day `at_ms` falls on, as a number (for day breaks).
pub fn day(at_ms: u64) -> i64 {
    local(at_ms).map_or(0, |t| i64::from(t.date().num_days_from_ce()))
}

/// The UTC date `at_ms` falls on, as `YYYY-MM-DD` (the daemon's usage days).
pub fn utc_date(at_ms: u64) -> String {
    i64::try_from(at_ms)
        .ok()
        .and_then(DateTime::from_timestamp_millis)
        .map_or_else(String::new, |t| t.format("%Y-%m-%d").to_string())
}

/// A count in a few characters: "950", "12.4k", "3.1M".
pub fn compact_count(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}k", n as f64 / 1_000.0),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// "3:42 PM".
pub fn clock(at_ms: u64) -> String {
    local(at_ms).map_or_else(String::new, |t| t.format("%-I:%M %p").to_string())
}

/// When a conversation last moved, for a list: "now", "5m", "3h" within
/// the last day ("18h" for last evening), then its date ("Nov 12";
/// "Oct 30, 2025" in another year).
pub fn stamp(at_ms: u64, now_ms: u64) -> String {
    let secs = now_ms.saturating_sub(at_ms) / 1000;
    match secs {
        0..60 => "now".into(),
        60..3_600 => format!("{}m", secs / 60),
        3_600..86_400 => format!("{}h", secs / 3_600),
        _ => short_date(at_ms, now_ms),
    }
}

/// A list's date: "Oct 30", "Nov 3"; with the year when not this one
/// ("Oct 30, 2025").
pub fn short_date(at_ms: u64, now_ms: u64) -> String {
    let (Some(at), Some(now)) = (local(at_ms), local(now_ms)) else {
        return String::new();
    };
    if at.year() == now.year() {
        at.format("%b %-d").to_string()
    } else {
        at.format("%b %-d, %Y").to_string()
    }
}

/// "Today", "Yesterday", "Tomorrow", "Monday" (the week either side),
/// "Mar 3" or "Mar 3, 2025".
pub fn day_label(at_ms: u64, now_ms: u64) -> String {
    let (Some(at), Some(now)) = (local(at_ms), local(now_ms)) else {
        return String::new();
    };
    let days = (now.date() - at.date()).num_days();
    match days {
        0 => "Today".into(),
        1 => "Yesterday".into(),
        -1 => "Tomorrow".into(),
        2..=6 | -6..=-2 => at.format("%A").to_string(),
        _ if at.year() == now.year() => at.format("%b %-d").to_string(),
        _ => at.format("%b %-d, %Y").to_string(),
    }
}

/// A short age for lists: "now", "5m", "3h", "Yesterday", "Mon", "Mar 3".
pub fn ago(at_ms: u64, now_ms: u64) -> String {
    let secs = now_ms.saturating_sub(at_ms) / 1000;
    match secs {
        0..60 => "now".into(),
        60..3_600 => format!("{}m", secs / 60),
        3_600..86_400 if day(at_ms) == day(now_ms) => format!("{}h", secs / 3_600),
        _ => match local(at_ms).zip(local(now_ms)) {
            Some((at, now)) => {
                let days = (now.date() - at.date()).num_days();
                match days {
                    0 => format!("{}h", secs / 3_600),
                    1 => "Yesterday".into(),
                    2..=6 => at.format("%a").to_string(),
                    _ => at.format("%b %-d").to_string(),
                }
            }
            None => String::new(),
        },
    }
}

/// "1h 30m", "4m", "12s".
pub fn span(secs: u64) -> String {
    match secs {
        0..60 => format!("{secs}s"),
        // Under ten minutes a running clock still moves each second.
        60..600 if !secs.is_multiple_of(60) => format!("{}m {}s", secs / 60, secs % 60),
        60..3_600 => format!("{}m", secs / 60),
        3_600..86_400 => {
            let m = (secs % 3_600) / 60;
            if m == 0 {
                format!("{}h", secs / 3_600)
            } else {
                format!("{}h {m}m", secs / 3_600)
            }
        }
        _ => format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3_600),
    }
}

#[cfg(test)]
mod stamp_tests {
    use super::*;

    #[test]
    fn a_stamp_is_an_age_within_a_day_then_a_date() {
        let now = 1_763_000_000_000;
        assert_eq!(stamp(now - 20_000, now), "now");
        assert_eq!(stamp(now - 5 * 60_000, now), "5m");
        assert_eq!(stamp(now - 3 * 3_600_000, now), "3h");
        assert_eq!(stamp(now - 18 * 3_600_000, now), "18h");
        assert_eq!(
            stamp(now - 3 * 86_400_000, now),
            short_date(now - 3 * 86_400_000, now)
        );
    }
}
