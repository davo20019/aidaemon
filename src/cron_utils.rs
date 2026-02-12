//! Cron schedule parsing and next-run computation utilities.
//!
//! Extracted from the deprecated `scheduler` module. Used by health probes
//! and evergreen goal scheduling.

use chrono::{
    DateTime, Datelike, FixedOffset, Local, LocalResult, NaiveDate, NaiveDateTime, TimeZone,
    Timelike, Utc,
};
use croner::Cron;
use regex::Regex;

/// Parse a human-friendly schedule string into a 5-field cron expression.
/// Supports natural shortcuts and raw cron pass-through.
pub fn parse_schedule(input: &str) -> anyhow::Result<String> {
    let input = input.trim().trim_end_matches(['.', '!', '?']).trim();
    let now_local = Local::now();

    // Simple keyword shortcuts
    match input.to_lowercase().as_str() {
        "hourly" => return Ok("0 * * * *".to_string()),
        "daily" => return Ok("0 0 * * *".to_string()),
        "weekly" => return Ok("0 0 * * 0".to_string()),
        "monthly" => return Ok("0 0 1 * *".to_string()),
        _ => {}
    }

    // "every Nm" / "every N minutes" / "each 5 minutes" / "1 each 5m"
    let re_minutes =
        Regex::new(r"(?i)^(?:\d+\s+)?(?:every|each)\s+(\d+)\s*(?:m|min|mins|minutes?)$")?;
    if let Some(caps) = re_minutes.captures(input) {
        let n: u32 = caps[1].parse()?;
        if n == 0 || n > 59 {
            anyhow::bail!("Minutes interval must be between 1 and 59");
        }
        return Ok(format!("*/{} * * * *", n));
    }

    // "every Nh" / "every N hours" / "each 2h" / "1 each 6 hours"
    let re_hours = Regex::new(r"(?i)^(?:\d+\s+)?(?:every|each)\s+(\d+)\s*(?:h|hrs?|hours?)$")?;
    if let Some(caps) = re_hours.captures(input) {
        let n: u32 = caps[1].parse()?;
        if n == 0 || n > 23 {
            anyhow::bail!("Hours interval must be between 1 and 23");
        }
        return Ok(format!("0 */{} * * *", n));
    }

    // "daily at 9am" / "daily at 14:30" / "daily at 2pm" / "daily at 2:30pm"
    let re_daily = Regex::new(r"(?i)^daily\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    if let Some(caps) = re_daily.captures(input) {
        let (hour, minute) = parse_time_captures(&caps)?;
        return Ok(format!("{} {} * * *", minute, hour));
    }

    // "weekdays at 8:30" / "weekdays at 9am"
    let re_weekdays = Regex::new(r"(?i)^weekdays?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    if let Some(caps) = re_weekdays.captures(input) {
        let (hour, minute) = parse_time_captures(&caps)?;
        return Ok(format!("{} {} * * 1-5", minute, hour));
    }

    // "weekends at 10am"
    let re_weekends = Regex::new(r"(?i)^weekends?\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    if let Some(caps) = re_weekends.captures(input) {
        let (hour, minute) = parse_time_captures(&caps)?;
        return Ok(format!("{} {} * * 0,6", minute, hour));
    }

    // Dynamic relative one-shot parser:
    // accepts variants like "in 2h", "after 90 minutes", "in 1 hour 30 minutes".
    if let Some(duration) = parse_relative_duration(input) {
        let target = now_local + duration;
        return Ok(one_shot_cron_from_local(target));
    }

    // "today at 11:09pm EST" / "tonight at 23:10" / "tomorrow at 9am"
    // Optional timezone token supports common abbreviations and numeric offsets.
    if let Some(target) = parse_day_time_with_optional_timezone(input, now_local) {
        if target <= now_local {
            anyhow::bail!(
                "Schedule time '{}' is in the past for system timezone {}.",
                input,
                system_timezone_display()
            );
        }
        return Ok(one_shot_cron_from_local(target));
    }

    // Dynamic absolute datetime parser:
    // RFC3339, ISO-like forms, and common date/time formats interpreted in system timezone.
    if let Some(target) = parse_absolute_datetime_local(input) {
        if target <= now_local {
            anyhow::bail!(
                "Schedule time '{}' is in the past for system timezone {}.",
                input,
                system_timezone_display()
            );
        }
        return Ok(one_shot_cron_from_local(target));
    }

    // Raw cron pass-through: validate with croner
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.len() == 5 {
        // Try to parse as cron
        input
            .parse::<Cron>()
            .map_err(|e| anyhow::anyhow!("Invalid cron expression '{}': {}", input, e))?;
        return Ok(input.to_string());
    }

    anyhow::bail!(
        "Unrecognized schedule format '{}'. Use natural shortcuts (e.g. 'daily at 9am', 'every 5m', 'in 2h', 'today at 11:09pm EST', 'after 90 minutes', '2026-03-14 09:30') or a 5-field cron expression. Interpreted in system timezone {}.",
        input
        ,
        system_timezone_display()
    )
}

fn one_shot_cron_from_local(target: DateTime<Local>) -> String {
    format!(
        "{} {} {} {} *",
        target.minute(),
        target.hour(),
        target.day(),
        target.month()
    )
}

fn parse_relative_duration(input: &str) -> Option<chrono::Duration> {
    let lower = input.to_lowercase();
    if !lower.contains("in ") && !lower.contains("after ") {
        return None;
    }

    let re =
        Regex::new(r"(?i)(\d+)\s*(weeks?|w|days?|d|hours?|hrs?|h|minutes?|mins?|min|m)").ok()?;
    let mut total = chrono::Duration::zero();
    let mut matched = false;

    for caps in re.captures_iter(input) {
        let n: i64 = caps.get(1)?.as_str().parse().ok()?;
        if n <= 0 {
            continue;
        }
        let unit = caps.get(2)?.as_str().to_lowercase();
        matched = true;
        if unit.starts_with('w') {
            total += chrono::Duration::weeks(n);
        } else if unit.starts_with('d') {
            total += chrono::Duration::days(n);
        } else if unit.starts_with('h') {
            total += chrono::Duration::hours(n);
        } else {
            total += chrono::Duration::minutes(n);
        }
    }

    if matched && total > chrono::Duration::zero() {
        Some(total)
    } else {
        None
    }
}

fn parse_day_time_with_optional_timezone(
    input: &str,
    now_local: DateTime<Local>,
) -> Option<DateTime<Local>> {
    let re = Regex::new(
        r"(?i)^(today|tonight|tomorrow)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s+([A-Za-z]{1,8}|[+-]\d{2}:?\d{2}|Z))?$",
    )
    .ok()?;
    let caps = re.captures(input)?;
    let day_kw = caps.get(1)?.as_str().to_ascii_lowercase();
    let (hour, minute) = parse_time_with_offset_captures(&caps, 2, 3, 4)?;
    let tz_token = caps.get(5).map(|m| m.as_str().trim());

    if let Some(tz_raw) = tz_token {
        let offset = parse_timezone_offset(tz_raw)?;
        let now_in_tz = now_local.with_timezone(&offset);
        let base_date = match day_kw.as_str() {
            "tomorrow" => (now_in_tz + chrono::Duration::days(1)).date_naive(),
            "today" | "tonight" => now_in_tz.date_naive(),
            _ => return None,
        };
        let naive = base_date.and_hms_opt(hour, minute, 0)?;
        let target_in_tz = match offset.from_local_datetime(&naive) {
            LocalResult::Single(dt) => dt,
            LocalResult::Ambiguous(early, _) => early,
            LocalResult::None => return None,
        };
        return Some(target_in_tz.with_timezone(&Local));
    }

    let base_date = match day_kw.as_str() {
        "tomorrow" => (now_local + chrono::Duration::days(1)).date_naive(),
        "today" | "tonight" => now_local.date_naive(),
        _ => return None,
    };
    let naive = base_date.and_hms_opt(hour, minute, 0)?;
    resolve_local_naive(naive)
}

fn parse_timezone_offset(token: &str) -> Option<FixedOffset> {
    let upper = token.trim().to_ascii_uppercase();
    let hours_west = match upper.as_str() {
        "EST" => Some(5),
        "EDT" => Some(4),
        "CST" => Some(6),
        "CDT" => Some(5),
        "MST" => Some(7),
        "MDT" => Some(6),
        "PST" => Some(8),
        "PDT" => Some(7),
        "UTC" | "GMT" | "Z" => Some(0),
        _ => None,
    };
    if let Some(h) = hours_west {
        return if h == 0 {
            FixedOffset::east_opt(0)
        } else {
            FixedOffset::west_opt(h * 3600)
        };
    }

    let re = Regex::new(r"^([+-])(\d{2})(?::?(\d{2}))$").ok()?;
    let caps = re.captures(&upper)?;
    let sign = caps.get(1)?.as_str();
    let hours: i32 = caps.get(2)?.as_str().parse().ok()?;
    let minutes: i32 = caps.get(3)?.as_str().parse().ok()?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    let seconds = hours * 3600 + minutes * 60;
    if sign == "+" {
        FixedOffset::east_opt(seconds)
    } else {
        FixedOffset::west_opt(seconds)
    }
}

fn parse_absolute_datetime_local(input: &str) -> Option<DateTime<Local>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(input) {
        return Some(dt.with_timezone(&Local));
    }

    let dt_formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I%p",
        "%b %d %Y %H:%M",
        "%b %d %Y %I:%M %p",
        "%B %d %Y %H:%M",
        "%B %d %Y %I:%M %p",
    ];
    for fmt in dt_formats {
        if let Ok(naive) = NaiveDateTime::parse_from_str(input, fmt) {
            if let Some(dt) = resolve_local_naive(naive) {
                return Some(dt);
            }
        }
    }

    let date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"];
    for fmt in date_formats {
        if let Ok(date) = NaiveDate::parse_from_str(input, fmt) {
            if let Some(naive) = date.and_hms_opt(0, 0, 0) {
                if let Some(dt) = resolve_local_naive(naive) {
                    return Some(dt);
                }
            }
        }
    }

    None
}

fn resolve_local_naive(naive: NaiveDateTime) -> Option<DateTime<Local>> {
    match Local.from_local_datetime(&naive) {
        LocalResult::Single(dt) => Some(dt),
        LocalResult::Ambiguous(early, _) => Some(early),
        LocalResult::None => None,
    }
}

fn parse_time_with_offset_captures(
    caps: &regex::Captures,
    hour_idx: usize,
    minute_idx: usize,
    ampm_idx: usize,
) -> Option<(u32, u32)> {
    let mut hour: u32 = caps.get(hour_idx)?.as_str().parse().ok()?;
    let minute: u32 = caps
        .get(minute_idx)
        .map_or(Some(0), |m| m.as_str().parse().ok())?;
    if let Some(ampm) = caps.get(ampm_idx) {
        let ampm = ampm.as_str().to_ascii_lowercase();
        if ampm == "pm" && hour < 12 {
            hour += 12;
        } else if ampm == "am" && hour == 12 {
            hour = 0;
        }
    }
    if hour > 23 || minute > 59 {
        return None;
    }
    Some((hour, minute))
}

/// Extract hour and minute from regex captures with optional AM/PM.
fn parse_time_captures(caps: &regex::Captures) -> anyhow::Result<(u32, u32)> {
    let mut hour: u32 = caps[1].parse()?;
    let minute: u32 = caps.get(2).map_or(Ok(0), |m| m.as_str().parse())?;
    if let Some(ampm) = caps.get(3) {
        let ampm = ampm.as_str().to_lowercase();
        if ampm == "pm" && hour < 12 {
            hour += 12;
        } else if ampm == "am" && hour == 12 {
            hour = 0;
        }
    }
    if hour > 23 {
        anyhow::bail!("Hour must be between 0 and 23");
    }
    if minute > 59 {
        anyhow::bail!("Minute must be between 0 and 59");
    }
    Ok((hour, minute))
}

/// Compute the next occurrence from a cron expression using croner.
pub fn compute_next_run(cron_expr: &str) -> anyhow::Result<DateTime<Utc>> {
    Ok(compute_next_run_local(cron_expr)?.with_timezone(&Utc))
}

/// Compute the next occurrence from a cron expression in system-local timezone.
pub fn compute_next_run_local(cron_expr: &str) -> anyhow::Result<DateTime<Local>> {
    let cron: Cron = cron_expr
        .parse()
        .map_err(|e| anyhow::anyhow!("Failed to parse cron '{}': {}", cron_expr, e))?;

    cron.find_next_occurrence(&Local::now(), false)
        .map_err(|e| anyhow::anyhow!("No next occurrence for '{}': {}", cron_expr, e))
}

/// Human-readable system timezone label used in schedule confirmations.
pub fn system_timezone_display() -> String {
    Local::now().format("%Z (UTC%:z)").to_string()
}

/// Heuristic: whether a cron expression looks like a one-shot absolute schedule.
///
/// This is intentionally conservative and should only be used as a fallback when
/// higher-confidence intent classification data is unavailable.
pub fn is_one_shot_schedule(cron_expr: &str) -> bool {
    let parts: Vec<&str> = cron_expr.split_whitespace().collect();
    if parts.len() != 5 {
        return false;
    }

    let dom = parts[2];
    let month = parts[3];
    dom != "*" && month != "*" && !dom.contains('/') && !month.contains('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schedule_keywords() {
        assert_eq!(parse_schedule("hourly").unwrap(), "0 * * * *");
        assert_eq!(parse_schedule("daily").unwrap(), "0 0 * * *");
        assert_eq!(parse_schedule("weekly").unwrap(), "0 0 * * 0");
        assert_eq!(parse_schedule("monthly").unwrap(), "0 0 1 * *");
    }

    #[test]
    fn test_parse_schedule_every_minutes() {
        assert_eq!(parse_schedule("every 5m").unwrap(), "*/5 * * * *");
        assert_eq!(parse_schedule("every 15 minutes").unwrap(), "*/15 * * * *");
        assert_eq!(parse_schedule("every 1 min").unwrap(), "*/1 * * * *");
        assert_eq!(parse_schedule("each 5 minutes").unwrap(), "*/5 * * * *");
        assert_eq!(parse_schedule("each 5m").unwrap(), "*/5 * * * *");
        assert_eq!(parse_schedule("1 each 5m").unwrap(), "*/5 * * * *");
    }

    #[test]
    fn test_parse_schedule_every_hours() {
        assert_eq!(parse_schedule("every 2h").unwrap(), "0 */2 * * *");
        assert_eq!(parse_schedule("every 4 hours").unwrap(), "0 */4 * * *");
        assert_eq!(parse_schedule("each 6h").unwrap(), "0 */6 * * *");
    }

    #[test]
    fn test_parse_schedule_daily_at() {
        assert_eq!(parse_schedule("daily at 9am").unwrap(), "0 9 * * *");
        assert_eq!(parse_schedule("daily at 14:30").unwrap(), "30 14 * * *");
        assert_eq!(parse_schedule("daily at 2pm").unwrap(), "0 14 * * *");
        assert_eq!(parse_schedule("daily at 2:30pm").unwrap(), "30 14 * * *");
        assert_eq!(parse_schedule("daily at 12am").unwrap(), "0 0 * * *");
    }

    #[test]
    fn test_parse_schedule_weekdays() {
        assert_eq!(parse_schedule("weekdays at 8:30").unwrap(), "30 8 * * 1-5");
        assert_eq!(parse_schedule("weekdays at 9am").unwrap(), "0 9 * * 1-5");
    }

    #[test]
    fn test_parse_schedule_weekends() {
        assert_eq!(parse_schedule("weekends at 10am").unwrap(), "0 10 * * 0,6");
    }

    #[test]
    fn test_parse_schedule_cron_passthrough() {
        assert_eq!(parse_schedule("0 9 * * 1-5").unwrap(), "0 9 * * 1-5");
        assert_eq!(parse_schedule("*/5 * * * *").unwrap(), "*/5 * * * *");
    }

    #[test]
    fn test_parse_schedule_tomorrow_at() {
        let cron = parse_schedule("tomorrow at 9am").unwrap();
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], "0");
        assert_eq!(parts[1], "9");
        assert_eq!(parts[4], "*");
    }

    #[test]
    fn test_parse_schedule_day_time_with_timezone_abbrev() {
        let now_est = Local::now().with_timezone(&FixedOffset::west_opt(5 * 3600).unwrap());
        let target_est = now_est + chrono::Duration::minutes(30);
        let day_kw = if target_est.date_naive() == now_est.date_naive() {
            "today"
        } else {
            "tomorrow"
        };
        let input = format!("{} at {} EST", day_kw, target_est.format("%I:%M%P"));
        let cron = parse_schedule(&input).unwrap();
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
    }

    #[test]
    fn test_parse_schedule_tonight_with_trailing_punctuation() {
        let cron = parse_schedule("tonight at 11:09pm EST.").unwrap_or_else(|_| {
            // If 11:09pm EST has already passed for today, use tomorrow in same format.
            parse_schedule("tomorrow at 11:09pm EST.").unwrap()
        });
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
    }

    #[test]
    fn test_is_one_shot_schedule() {
        assert!(is_one_shot_schedule("15 9 12 3 *"));
        assert!(!is_one_shot_schedule("0 */6 * * *"));
        assert!(!is_one_shot_schedule("0 9 * * 1-5"));
    }

    #[test]
    fn test_parse_schedule_invalid() {
        assert!(parse_schedule("never").is_err());
        assert!(parse_schedule("every 0m").is_err());
        assert!(parse_schedule("daily at 25:00").is_err());
    }

    #[test]
    fn test_parse_timezone_offset_abbrev_and_numeric() {
        assert_eq!(
            parse_timezone_offset("EST").unwrap().local_minus_utc(),
            -5 * 3600
        );
        assert_eq!(
            parse_timezone_offset("+05:30").unwrap().local_minus_utc(),
            5 * 3600 + 30 * 60
        );
        assert_eq!(
            parse_timezone_offset("-0400").unwrap().local_minus_utc(),
            -4 * 3600
        );
    }

    #[test]
    fn test_parse_schedule_relative_duration_phrase() {
        let cron = parse_schedule("in 1 hour 30 minutes").unwrap();
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
    }

    #[test]
    fn test_parse_schedule_absolute_datetime_iso() {
        let target = Local::now() + chrono::Duration::hours(2);
        let input = target.format("%Y-%m-%d %H:%M").to_string();
        let cron = parse_schedule(&input).unwrap();
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], target.minute().to_string());
        assert_eq!(parts[1], target.hour().to_string());
    }

    #[test]
    fn test_system_timezone_display_non_empty() {
        assert!(!system_timezone_display().is_empty());
    }

    #[test]
    fn test_compute_next_run() {
        let next = compute_next_run("* * * * *").unwrap();
        assert!(next > Utc::now());
    }

    #[test]
    fn test_compute_next_run_local() {
        let next = compute_next_run_local("* * * * *").unwrap();
        assert!(next > Local::now());
    }
}
