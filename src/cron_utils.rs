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
use std::sync::LazyLock;

static RE_SPACE_COLLAPSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("space collapse regex"));
static RE_SPLIT_NUMBERED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[.;]\s*)\s*\d+[\)\.]\s+").expect("numbered split regex")
});
static RE_SPLIT_DELIMITER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:[.!?]\s+(?:also|and)\s+|;\s*(?:also|and)\s+|;\s+)")
        .expect("delimiter split regex")
});
static RE_SCHEDULE_IN_TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:in|after)\s+\d+\s*(?:w|weeks?|d|days?|h|hrs?|hours?|m|min|mins|minutes?)\b",
    )
    .expect("relative schedule regex")
});
static RE_SCHEDULE_DAY_AT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:today|tonight|tomorrow)\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?(?:\s+(?:[A-Za-z]{1,8}|[+-]\d{2}:?\d{2}|Z))?\b",
    )
    .expect("day schedule regex")
});
static RE_SCHEDULE_MONTH_DAY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:on\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?\b",
    )
    .expect("month-day schedule regex")
});
static RE_SCHEDULE_EVERY_INTERVAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:every|each)\s+\d+\s*(?:m|min|mins|minutes?|h|hrs?|hours?)\b")
        .expect("interval schedule regex")
});
static RE_SCHEDULE_DAILY_AT: LazyLock<Regex> = LazyLock::new(|| {
    let time = r"(?:noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)";
    let time_list = format!(
        r"(?P<times>{time}(?:\s*(?:,|\band\b|&)\s*{time})*)",
        time = time
    );
    Regex::new(&format!(
        r"(?i)\b(?:daily|every\s+day|everyday|each\s+day)\s+at\s+{}",
        time_list
    ))
    .expect("daily-at schedule regex")
});
static RE_SCHEDULE_WEEKDAYS_AT: LazyLock<Regex> = LazyLock::new(|| {
    let time = r"(?:noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)";
    let time_list = format!(
        r"(?P<times>{time}(?:\s*(?:,|\band\b|&)\s*{time})*)",
        time = time
    );
    Regex::new(&format!(
        r"(?i)\b(?:weekdays?|every\s+weekdays?|every\s+weekday|each\s+weekday)\s+at\s+{}",
        time_list
    ))
    .expect("weekdays-at schedule regex")
});
static RE_SCHEDULE_WEEKENDS_AT: LazyLock<Regex> = LazyLock::new(|| {
    let time = r"(?:noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)";
    let time_list = format!(
        r"(?P<times>{time}(?:\s*(?:,|\band\b|&)\s*{time})*)",
        time = time
    );
    Regex::new(&format!(
        r"(?i)\b(?:weekends?|every\s+weekends?|every\s+weekend|each\s+weekend)\s+at\s+{}",
        time_list
    ))
    .expect("weekends-at schedule regex")
});
static RE_SCHEDULE_SPECIFIC_DAYS_AT: LazyLock<Regex> = LazyLock::new(|| {
    let day =
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)";
    let time = r"(?:noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)";
    let time_list = format!(
        r"(?P<times>{time}(?:\s*(?:,|\band\b|&)\s*{time})*)",
        time = time
    );
    Regex::new(&format!(
        r"(?i)\b(?:every|on)\s+(?P<days>{day}(?:\s*(?:,|\band\b|&)\s*{day})*)\s+at\s+{}",
        time_list,
        day = day
    ))
    .expect("specific-days schedule regex")
});
static RE_PREFIX_NOISE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:\d+[\)\.]\s*|,|;|:|-|and\b|also\b|then\b|to\b|for\b)\s*")
        .expect("prefix-noise regex")
});
static RE_FILLER_PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:please\s+)?(?:remind me to|schedule(?:\s+a)?\s+task\s+to|schedule\s+to|set up(?:\s+a)?\s+task\s+to|set up\s+to|i need you to|can you(?:\s+please)?\s+)\s*",
    )
    .expect("filler-prefix regex")
});
static RE_LEADING_CONNECTOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:also|and|then)\b[:,]?\s*").expect("leading-connector regex")
});

/// A schedule/task pair extracted from one message segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleSegment {
    /// Clean task text with schedule/filler language stripped.
    pub description: String,
    /// Human-readable schedule phrase.
    pub schedule_raw: String,
    /// Whether this schedule is one-time.
    pub is_one_shot: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScheduleMatch {
    schedule_raw: String,
    is_one_shot: bool,
    start: usize,
    end: usize,
}

/// Extract the first schedule phrase from arbitrary user text.
///
/// Returns `(schedule_raw, is_one_shot)` when a schedule is detected.
pub fn extract_schedule_from_text(text: &str) -> Option<(String, bool)> {
    let schedule_match = extract_schedule_match(text)?;
    Some((schedule_match.schedule_raw, schedule_match.is_one_shot))
}

/// Extract all schedule/task segments from a potentially multi-request message.
pub fn extract_schedule_segments(text: &str) -> Vec<ScheduleSegment> {
    let mut out = Vec::new();
    for candidate in split_schedule_candidates(text) {
        if let Some(schedule_match) = extract_schedule_match(&candidate) {
            let description = clean_task_description_with_match(
                &candidate,
                &schedule_match.schedule_raw,
                Some((schedule_match.start, schedule_match.end)),
            );
            if !description.trim().is_empty() {
                out.push(ScheduleSegment {
                    description,
                    schedule_raw: schedule_match.schedule_raw,
                    is_one_shot: schedule_match.is_one_shot,
                });
            }
        } else if let Some(previous) = out.last_mut() {
            append_non_schedule_fragment(previous, &candidate);
        }
    }
    out
}

/// Clean task text by stripping the detected schedule expression and
/// common filler prefixes.
pub fn clean_task_description(text: &str, schedule_match: &str) -> String {
    // Re-run schedule extraction on the text to get the precise match range.
    // The caller's `schedule_match` may be normalized (e.g. "weekdays at 9am")
    // which won't match the original text (e.g. "every weekday at 9am").
    let schedule_range = extract_schedule_match(text).map(|sm| (sm.start, sm.end));
    clean_task_description_with_match(text, schedule_match, schedule_range)
}

fn split_schedule_candidates(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let normalized = trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let normalized = RE_SPACE_COLLAPSE
        .replace_all(&normalized, " ")
        .trim()
        .to_string();
    if normalized.is_empty() {
        return Vec::new();
    }

    let mut parts = vec![normalized];
    for splitter in [&*RE_SPLIT_NUMBERED, &*RE_SPLIT_DELIMITER] {
        let mut next_parts = Vec::new();
        for part in parts {
            for piece in splitter.split(&part) {
                let cleaned = piece.trim().trim_matches(['.', '!', '?']).trim();
                if !cleaned.is_empty() {
                    next_parts.push(cleaned.to_string());
                }
            }
        }
        parts = next_parts;
    }

    if parts.is_empty() {
        parts.push(trimmed.to_string());
    }
    parts
}

fn extract_schedule_match(text: &str) -> Option<ScheduleMatch> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    // One-shot relative time, e.g. "in 2h", "after 90 minutes".
    if let Some(m) = RE_SCHEDULE_IN_TIME.find(text) {
        return Some(ScheduleMatch {
            schedule_raw: m.as_str().trim().to_string(),
            is_one_shot: true,
            start: m.start(),
            end: m.end(),
        });
    }

    // One-shot day+time with optional timezone.
    if let Some(m) = RE_SCHEDULE_DAY_AT.find(text) {
        return Some(ScheduleMatch {
            schedule_raw: m.as_str().trim().to_string(),
            is_one_shot: true,
            start: m.start(),
            end: m.end(),
        });
    }

    // One-shot month/day phrases, e.g. "on March 5th at 3pm".
    if let Some(m) = RE_SCHEDULE_MONTH_DAY.find(text) {
        return Some(ScheduleMatch {
            schedule_raw: m.as_str().trim().to_string(),
            is_one_shot: true,
            start: m.start(),
            end: m.end(),
        });
    }

    // Recurring intervals: "every 6h", "each 5 minutes".
    if let Some(m) = RE_SCHEDULE_EVERY_INTERVAL.find(text) {
        return Some(ScheduleMatch {
            schedule_raw: m.as_str().trim().to_string(),
            is_one_shot: false,
            start: m.start(),
            end: m.end(),
        });
    }

    // Recurring "at" schedules; normalized to canonical phrases.
    if let Some(caps) = RE_SCHEDULE_DAILY_AT.captures(text) {
        let m = caps.get(0)?;
        let times = caps.name("times")?.as_str().trim();
        return Some(ScheduleMatch {
            schedule_raw: format!("every day at {}", times),
            is_one_shot: false,
            start: m.start(),
            end: extend_match_past_trailing_timezone(text, m.end()),
        });
    }

    if let Some(caps) = RE_SCHEDULE_WEEKDAYS_AT.captures(text) {
        let m = caps.get(0)?;
        let times = caps.name("times")?.as_str().trim();
        return Some(ScheduleMatch {
            schedule_raw: format!("weekdays at {}", times),
            is_one_shot: false,
            start: m.start(),
            end: extend_match_past_trailing_timezone(text, m.end()),
        });
    }

    if let Some(caps) = RE_SCHEDULE_WEEKENDS_AT.captures(text) {
        let m = caps.get(0)?;
        let times = caps.name("times")?.as_str().trim();
        return Some(ScheduleMatch {
            schedule_raw: format!("weekends at {}", times),
            is_one_shot: false,
            start: m.start(),
            end: extend_match_past_trailing_timezone(text, m.end()),
        });
    }

    // Specific named days: "every Monday and Friday at 9am"
    if let Some(caps) = RE_SCHEDULE_SPECIFIC_DAYS_AT.captures(text) {
        let m = caps.get(0)?;
        let days_str = caps.name("days")?.as_str().trim();
        let times = caps.name("times")?.as_str().trim();
        return Some(ScheduleMatch {
            schedule_raw: format!("{} at {}", normalize_day_names(days_str), times),
            is_one_shot: false,
            start: m.start(),
            end: extend_match_past_trailing_timezone(text, m.end()),
        });
    }

    let lower = text.to_ascii_lowercase();
    for keyword in ["hourly", "daily", "weekly", "monthly"] {
        if lower == keyword {
            return Some(ScheduleMatch {
                schedule_raw: keyword.to_string(),
                is_one_shot: false,
                start: 0,
                end: text.len(),
            });
        }
    }

    None
}

fn clean_task_description_with_match(
    text: &str,
    schedule_match: &str,
    schedule_range: Option<(usize, usize)>,
) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut cleaned = trimmed.to_string();
    if let Some((start, end)) = schedule_range {
        if start < end && end <= cleaned.len() {
            cleaned.replace_range(start..end, " ");
        }
    } else if !schedule_match.trim().is_empty() {
        let escaped = regex::escape(schedule_match.trim());
        let remove_schedule_re =
            Regex::new(&format!(r"(?i){}", escaped)).expect("escaped schedule regex");
        cleaned = remove_schedule_re.replacen(&cleaned, 1, " ").to_string();
    }

    loop {
        let next = RE_PREFIX_NOISE.replace(&cleaned, "").to_string();
        if next == cleaned {
            break;
        }
        cleaned = next;
    }

    loop {
        let next = RE_FILLER_PREFIX.replace(&cleaned, "").to_string();
        if next == cleaned {
            break;
        }
        cleaned = next;
    }

    cleaned = RE_SPACE_COLLAPSE.replace_all(&cleaned, " ").to_string();

    cleaned = cleaned
        .trim()
        .trim_matches(|c: char| c.is_whitespace() || [',', ';', '.', ':', '-', '|'].contains(&c))
        .to_string();

    if cleaned.is_empty() {
        return trimmed.to_string();
    }

    capitalize_first_ascii(&cleaned)
}

fn append_non_schedule_fragment(previous: &mut ScheduleSegment, fragment: &str) {
    let mut cleaned_fragment = fragment.trim().to_string();
    cleaned_fragment = RE_LEADING_CONNECTOR
        .replace(&cleaned_fragment, "")
        .to_string();
    loop {
        let next = RE_PREFIX_NOISE.replace(&cleaned_fragment, "").to_string();
        if next == cleaned_fragment {
            break;
        }
        cleaned_fragment = next;
    }
    cleaned_fragment = cleaned_fragment
        .trim()
        .trim_matches(|c: char| c.is_whitespace() || [',', ';', ':', '-', '|'].contains(&c))
        .to_string();
    if cleaned_fragment.is_empty() {
        return;
    }

    if !previous.description.is_empty() {
        previous.description.push(' ');
    }
    previous.description.push_str(&cleaned_fragment);
    previous.description = RE_SPACE_COLLAPSE
        .replace_all(&previous.description, " ")
        .trim()
        .to_string();
}

/// If the text immediately after `end` starts with a timezone token
/// (e.g. "EST", "UTC", "+05:30"), extend the boundary to include it so it
/// gets stripped from the goal description along with the schedule phrase.
fn extend_match_past_trailing_timezone(text: &str, end: usize) -> usize {
    static RE_TRAILING_TZ: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)^\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT|UTC|GMT|Z|[+-]\d{2}:?\d{2})\b")
            .expect("trailing-tz regex")
    });
    if end >= text.len() {
        return end;
    }
    if let Some(m) = RE_TRAILING_TZ.find(&text[end..]) {
        end + m.end()
    } else {
        end
    }
}

fn capitalize_first_ascii(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::with_capacity(text.len());
            out.push(first.to_ascii_uppercase());
            out.push_str(chars.as_str());
            out
        }
        None => String::new(),
    }
}

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

    // Multi-time variants (minutes must match):
    // - "daily at 6am, 12pm, 6pm"
    // - "every day at 6am, 12pm, 6pm"
    // - "weekdays at 9am and 5pm"
    // - "weekends at noon"
    let re_daily_at = Regex::new(r"(?i)^daily\s+at\s+(.+)$")?;
    if let Some(caps) = re_daily_at.captures(input) {
        let times = parse_time_list(caps.get(1).unwrap().as_str())?;
        return cron_for_recurring_times(times, "*");
    }
    let re_every_day_at = Regex::new(r"(?i)^(?:every\s+day|everyday)\s+at\s+(.+)$")?;
    if let Some(caps) = re_every_day_at.captures(input) {
        let times = parse_time_list(caps.get(1).unwrap().as_str())?;
        return cron_for_recurring_times(times, "*");
    }
    let re_weekdays_at = Regex::new(r"(?i)^weekdays?\s+at\s+(.+)$")?;
    if let Some(caps) = re_weekdays_at.captures(input) {
        let times = parse_time_list(caps.get(1).unwrap().as_str())?;
        return cron_for_recurring_times(times, "1-5");
    }
    let re_every_weekdays_at = Regex::new(r"(?i)^every\s+weekdays?\s+at\s+(.+)$")?;
    if let Some(caps) = re_every_weekdays_at.captures(input) {
        let times = parse_time_list(caps.get(1).unwrap().as_str())?;
        return cron_for_recurring_times(times, "1-5");
    }
    let re_weekends_at = Regex::new(r"(?i)^weekends?\s+at\s+(.+)$")?;
    if let Some(caps) = re_weekends_at.captures(input) {
        let times = parse_time_list(caps.get(1).unwrap().as_str())?;
        return cron_for_recurring_times(times, "0,6");
    }
    let re_every_weekends_at = Regex::new(r"(?i)^every\s+weekends?\s+at\s+(.+)$")?;
    if let Some(caps) = re_every_weekends_at.captures(input) {
        let times = parse_time_list(caps.get(1).unwrap().as_str())?;
        return cron_for_recurring_times(times, "0,6");
    }

    // Specific named days: "Monday and Friday at 9am", "every Tuesday, Thursday at 3pm"
    let re_specific_days_at = Regex::new(
        r"(?i)^(?:every\s+|on\s+)?(?P<days>(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)(?:\s*(?:,|\band\b|&)\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun))*)\s+at\s+(?P<times>.+)$",
    )?;
    if let Some(caps) = re_specific_days_at.captures(input) {
        let days = parse_day_list(caps.name("days").unwrap().as_str())?;
        let times = parse_time_list(caps.name("times").unwrap().as_str())?;
        let dow = days
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        return cron_for_recurring_times(times, &dow);
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

    if let Some(dt) = parse_named_month_datetime_local(input) {
        return Some(dt);
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

fn parse_named_month_datetime_local(input: &str) -> Option<DateTime<Local>> {
    let re = Regex::new(
        r"(?i)^(?:on\s+)?(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?(?:\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)?)?$",
    )
    .ok()?;
    let caps = re.captures(input.trim())?;

    let month = parse_month_name(caps.name("month")?.as_str())?;
    let day: u32 = caps.name("day")?.as_str().parse().ok()?;
    let explicit_year = caps
        .name("year")
        .and_then(|m| m.as_str().parse::<i32>().ok());

    let mut hour: u32 = caps
        .name("hour")
        .and_then(|m| m.as_str().parse::<u32>().ok())
        .unwrap_or(0);
    let minute: u32 = caps
        .name("minute")
        .and_then(|m| m.as_str().parse::<u32>().ok())
        .unwrap_or(0);
    if let Some(ampm) = caps.name("ampm") {
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

    let now_local = Local::now();
    let mut year = explicit_year.unwrap_or(now_local.year());
    let mut candidate = build_named_month_datetime_local(year, month, day, hour, minute)?;
    if explicit_year.is_none() && candidate <= now_local {
        year += 1;
        candidate = build_named_month_datetime_local(year, month, day, hour, minute)?;
    }

    Some(candidate)
}

fn build_named_month_datetime_local(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
) -> Option<DateTime<Local>> {
    let date = NaiveDate::from_ymd_opt(year, month, day)?;
    let naive = date.and_hms_opt(hour, minute, 0)?;
    resolve_local_naive(naive)
}

fn parse_month_name(month: &str) -> Option<u32> {
    match month.trim().to_ascii_lowercase().as_str() {
        "jan" | "january" => Some(1),
        "feb" | "february" => Some(2),
        "mar" | "march" => Some(3),
        "apr" | "april" => Some(4),
        "may" => Some(5),
        "jun" | "june" => Some(6),
        "jul" | "july" => Some(7),
        "aug" | "august" => Some(8),
        "sep" | "sept" | "september" => Some(9),
        "oct" | "october" => Some(10),
        "nov" | "november" => Some(11),
        "dec" | "december" => Some(12),
        _ => None,
    }
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

fn parse_time_list(raw: &str) -> anyhow::Result<Vec<(u32, u32)>> {
    let raw = raw.trim();
    if raw.is_empty() {
        anyhow::bail!("Missing time after 'at'");
    }

    // Split on commas + "and" (case-insensitive); keep it permissive for LLM text.
    let and_re = Regex::new(r"(?i)\s+and\s+")?;
    let normalized = and_re.replace_all(raw, ",").replace('&', ",");
    let parts: Vec<&str> = normalized
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        anyhow::bail!("Missing time after 'at'");
    }

    let mut out = Vec::new();
    for p in parts {
        out.push(parse_time_token(p)?);
    }
    Ok(out)
}

fn parse_time_token(token: &str) -> anyhow::Result<(u32, u32)> {
    let token = token.trim();
    if token.is_empty() {
        anyhow::bail!("Empty time token");
    }
    let lower = token.to_ascii_lowercase();
    if lower == "noon" {
        return Ok((12, 0));
    }
    if lower == "midnight" {
        return Ok((0, 0));
    }

    let re = Regex::new(r"(?i)^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")?;
    let Some(caps) = re.captures(token) else {
        anyhow::bail!("Unrecognized time token '{}'", token);
    };

    let mut hour: u32 = caps.get(1).unwrap().as_str().parse()?;
    let minute: u32 = caps.get(2).map_or(Ok(0), |m| m.as_str().parse())?;
    if let Some(ampm) = caps.get(3) {
        let ampm = ampm.as_str().to_ascii_lowercase();
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

/// Parse a day name (or abbreviation) to cron day-of-week number (0=Sun..6=Sat).
fn day_name_to_cron(name: &str) -> Option<u32> {
    match name.trim().to_ascii_lowercase().as_str() {
        "sun" | "sunday" => Some(0),
        "mon" | "monday" => Some(1),
        "tue" | "tuesday" => Some(2),
        "wed" | "wednesday" => Some(3),
        "thu" | "thursday" => Some(4),
        "fri" | "friday" => Some(5),
        "sat" | "saturday" => Some(6),
        _ => None,
    }
}

/// Normalize a comma/and-separated list of day names into canonical form
/// for display, e.g. "Monday, Wednesday, Friday".
fn normalize_day_names(raw: &str) -> String {
    let and_re = Regex::new(r"(?i)\s+and\s+").expect("and regex");
    let normalized = and_re.replace_all(raw, ", ").replace('&', ", ");
    let parts: Vec<&str> = normalized
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    let full_names: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p.to_ascii_lowercase().as_str() {
            "sun" | "sunday" => Some("Sunday"),
            "mon" | "monday" => Some("Monday"),
            "tue" | "tuesday" => Some("Tuesday"),
            "wed" | "wednesday" => Some("Wednesday"),
            "thu" | "thursday" => Some("Thursday"),
            "fri" | "friday" => Some("Friday"),
            "sat" | "saturday" => Some("Saturday"),
            _ => None,
        })
        .collect();
    if full_names.len() <= 1 {
        return full_names.join("");
    }
    let last = full_names[full_names.len() - 1];
    let rest = &full_names[..full_names.len() - 1];
    format!("{} and {}", rest.join(", "), last)
}

/// Parse a comma/and-separated day-name string into sorted cron day numbers.
fn parse_day_list(raw: &str) -> anyhow::Result<Vec<u32>> {
    let and_re = Regex::new(r"(?i)\s+and\s+")?;
    let normalized = and_re.replace_all(raw, ",").replace('&', ",");
    let parts: Vec<&str> = normalized
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        anyhow::bail!("No day names found");
    }
    let mut days = Vec::new();
    for p in parts {
        let d =
            day_name_to_cron(p).ok_or_else(|| anyhow::anyhow!("Unrecognized day name '{}'", p))?;
        if !days.contains(&d) {
            days.push(d);
        }
    }
    days.sort_unstable();
    Ok(days)
}

fn cron_for_recurring_times(times: Vec<(u32, u32)>, dow: &str) -> anyhow::Result<String> {
    if times.is_empty() {
        anyhow::bail!("Missing time after 'at'");
    }
    let minute = times[0].1;
    if times.iter().any(|(_, m)| *m != minute) {
        anyhow::bail!(
            "Multiple times must share the same minute (e.g. 'daily at 6am, 12pm, 6pm'). For different minutes, create multiple schedules."
        );
    }
    let mut hours: Vec<u32> = times.iter().map(|(h, _)| *h).collect();
    hours.sort_unstable();
    hours.dedup();
    let hour_field = hours
        .into_iter()
        .map(|h| h.to_string())
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{} {} * * {}", minute, hour_field, dow))
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
        assert_eq!(
            parse_schedule("daily at 6am, 12pm, 6pm").unwrap(),
            "0 6,12,18 * * *"
        );
        assert_eq!(
            parse_schedule("daily at 6, 12 and 6 pm").unwrap(),
            "0 6,12,18 * * *"
        );
        assert_eq!(
            parse_schedule("every day at 6, 12 and 6 pm").unwrap(),
            "0 6,12,18 * * *"
        );
        assert_eq!(
            parse_schedule("everyday at 6am, 12pm, 6pm").unwrap(),
            "0 6,12,18 * * *"
        );
    }

    #[test]
    fn test_parse_schedule_weekdays() {
        assert_eq!(parse_schedule("weekdays at 8:30").unwrap(), "30 8 * * 1-5");
        assert_eq!(parse_schedule("weekdays at 9am").unwrap(), "0 9 * * 1-5");
        assert_eq!(
            parse_schedule("weekdays at 9am and 5pm").unwrap(),
            "0 9,17 * * 1-5"
        );
        assert_eq!(
            parse_schedule("every weekday at 9am and 5pm").unwrap(),
            "0 9,17 * * 1-5"
        );
    }

    #[test]
    fn test_parse_schedule_weekends() {
        assert_eq!(parse_schedule("weekends at 10am").unwrap(), "0 10 * * 0,6");
        assert_eq!(
            parse_schedule("weekends at noon, 6pm").unwrap(),
            "0 12,18 * * 0,6"
        );
        assert_eq!(
            parse_schedule("every weekend at noon, 6pm").unwrap(),
            "0 12,18 * * 0,6"
        );
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
    fn test_parse_schedule_named_month_with_ordinal() {
        let cron = parse_schedule("on March 5th at 3pm").unwrap();
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], "0");
        assert_eq!(parts[1], "15");
        assert_eq!(parts[2], "5");
        assert_eq!(parts[3], "3");
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

    #[test]
    fn test_extract_schedule_segments_single_task() {
        let segments =
            extract_schedule_segments("every day at 9am remind me to check server health");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].schedule_raw, "every day at 9am");
        assert_eq!(segments[0].description, "Check server health");
        assert!(!segments[0].is_one_shot);
    }

    #[test]
    fn test_extract_schedule_segments_numbered_multiple_tasks() {
        let segments = extract_schedule_segments(
            "1) every day at 9am check server health. 2) in 2 hours send status report",
        );
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].schedule_raw, "every day at 9am");
        assert_eq!(segments[0].description, "Check server health");
        assert_eq!(segments[1].schedule_raw, "in 2 hours");
        assert_eq!(segments[1].description, "Send status report");
        assert!(segments[1].is_one_shot);
    }

    #[test]
    fn test_extract_schedule_segments_mixed_recurring_and_one_time() {
        let segments = extract_schedule_segments(
            "every 6h monitor API health; also on March 5th at 3pm prepare project status report",
        );
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].schedule_raw, "every 6h");
        assert_eq!(segments[0].description, "Monitor API health");
        assert!(!segments[0].is_one_shot);
        assert_eq!(segments[1].schedule_raw, "on March 5th at 3pm");
        assert_eq!(segments[1].description, "Prepare project status report");
        assert!(segments[1].is_one_shot);
    }

    #[test]
    fn test_extract_schedule_segments_no_schedule_returns_empty() {
        let segments = extract_schedule_segments("please check server health now");
        assert!(segments.is_empty());
    }

    #[test]
    fn test_extract_schedule_segments_preserves_trailing_non_schedule_clause() {
        let segments = extract_schedule_segments(
            "every day at 9am check server status. Also check the backup logs",
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].schedule_raw, "every day at 9am");
        assert_eq!(
            segments[0].description,
            "Check server status check the backup logs"
        );
    }

    #[test]
    fn test_clean_task_description_removes_schedule_and_filler() {
        let cleaned = clean_task_description(
            "on March 5th at 3pm remind me to prepare project status report",
            "on March 5th at 3pm",
        );
        assert_eq!(cleaned, "Prepare project status report");
    }

    #[test]
    fn test_timezone_stripped_from_description_daily() {
        let cleaned = clean_task_description(
            "every day at 9am EST check deployment status",
            "every day at 9am",
        );
        assert_eq!(cleaned, "Check deployment status");
    }

    #[test]
    fn test_timezone_stripped_from_description_utc() {
        let cleaned =
            clean_task_description("every day at 3pm UTC run backup job", "every day at 3pm");
        assert_eq!(cleaned, "Run backup job");
    }

    #[test]
    fn test_weekday_description_cleaned_singular() {
        let cleaned = clean_task_description(
            "every weekday at 9am check build pipeline",
            "weekdays at 9am",
        );
        assert_eq!(cleaned, "Check build pipeline");
    }

    #[test]
    fn test_parse_schedule_specific_days() {
        assert_eq!(
            parse_schedule("Monday and Friday at 3pm").unwrap(),
            "0 15 * * 1,5"
        );
        assert_eq!(
            parse_schedule("every Monday and Friday at 9am").unwrap(),
            "0 9 * * 1,5"
        );
        assert_eq!(
            parse_schedule("every Tuesday, Thursday at 2pm").unwrap(),
            "0 14 * * 2,4"
        );
    }

    #[test]
    fn test_extract_schedule_segments_specific_days() {
        let segments = extract_schedule_segments("every Monday and Friday at 3pm review PRs");
        assert_eq!(segments.len(), 1);
        assert!(!segments[0].is_one_shot);
        assert_eq!(segments[0].description, "Review PRs");
    }

    #[test]
    fn test_extract_schedule_match_timezone_extends_end() {
        let sm = extract_schedule_match("every day at 9am EST check stuff").unwrap();
        // The match end should cover "every day at 9am EST" (including timezone)
        let matched = &"every day at 9am EST check stuff"[sm.start..sm.end];
        assert!(
            matched.contains("EST"),
            "Match should include timezone token"
        );
    }
}
