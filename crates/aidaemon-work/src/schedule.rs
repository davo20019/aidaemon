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

    // A relative delay: "in 2h", "after 90 minutes", "in 1 hour 30 minutes".
    // The whole input must be the delay; anything else is refused rather
    // than guessed (see `parse_relative_duration`).
    if let Some(duration) = parse_relative_duration(input)? {
        let target = now_local + duration;
        return Ok(one_shot_cron_from_local(target));
    }

    // "today at 11:09pm" / "tonight at 23:10" / "tomorrow at 9am -05:00" /
    // "tomorrow at 9am Europe/Madrid": the owner's zone unless a numeric
    // offset, UTC or an IANA zone name follows.
    if let Some(target) = parse_day_time_with_optional_timezone(input, now_local)? {
        if target <= now_local {
            anyhow::bail!(
                "Schedule time '{}' is in the past for system timezone {}.",
                input,
                system_timezone_display()
            );
        }
        return Ok(one_shot_cron_from_local(target));
    }

    // Dynamic absolute datetime parser: RFC3339, ISO-like forms, and common
    // date/time formats, in the owner's zone or a trailing zone token's.
    if let Some(target) = parse_absolute_datetime(input, now_local)? {
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
        "Unrecognized schedule format '{}'. Use 'daily at 9am', 'every 5m', 'in 2 hours', 'tomorrow at 9am', an exact time like 2026-03-14T09:30:00-04:00 or '2026-03-14 09:30', or a 5-field cron expression. Interpreted in the owner's timezone {} unless a time names its zone (an IANA name like Europe/Madrid, a numeric offset or UTC).",
        input,
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

/// A relative delay, only when the whole input is one: `in`/`after`, then one
/// or more `<whole number> <unit>` terms (optionally joined by `and` or a
/// comma). Units are a closed set of whole words: minutes, hours, days,
/// weeks, with their usual short forms (`m`, `min`, `h`, `hr`, `d`, `w`).
///
/// `Ok(None)` when the input does not start with `in`/`after` and a number.
/// When it does but cannot be fully read ("in 2 months", "in 10 days at
/// 9am"), it is an error asking for a precise time: a guess there fired
/// "in 2 months" after 2 minutes and dropped the "at 9am".
fn parse_relative_duration(input: &str) -> anyhow::Result<Option<chrono::Duration>> {
    static LEAD: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)^(?:in|after)\s+\d").expect("relative lead regex"));
    static TERM: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)^(\d+)\s*([a-z]+)(?:\s*,\s*|\s+and\s+|\s*)").expect("relative term regex")
    });
    if !LEAD.is_match(input) {
        return Ok(None);
    }
    let refuse = || {
        anyhow::anyhow!(
            "'{input}' is not a delay this can read exactly. Give a delay in whole minutes, hours, days or weeks (e.g. 'in 90 minutes', 'in 3 days'), or an exact time like 2026-03-14T09:30:00-04:00."
        )
    };
    let mut rest = input
        .split_once(char::is_whitespace)
        .map(|(_, rest)| rest.trim_start())
        .unwrap_or("");
    let mut total = chrono::Duration::zero();
    while !rest.is_empty() {
        let caps = TERM.captures(rest).ok_or_else(refuse)?;
        let n: i64 = caps[1].parse().map_err(|_| refuse())?;
        let unit = caps[2].to_ascii_lowercase();
        let term = match unit.as_str() {
            "m" | "min" | "mins" | "minute" | "minutes" => chrono::Duration::try_minutes(n),
            "h" | "hr" | "hrs" | "hour" | "hours" => chrono::Duration::try_hours(n),
            "d" | "day" | "days" => chrono::Duration::try_days(n),
            "w" | "wk" | "wks" | "week" | "weeks" => chrono::Duration::try_weeks(n),
            _ => None,
        }
        .ok_or_else(refuse)?;
        total += term;
        rest = &rest[caps.get(0).map_or(rest.len(), |m| m.end())..];
    }
    if total <= chrono::Duration::zero() {
        return Err(refuse());
    }
    Ok(Some(total))
}

/// The zone a schedule's wall-clock time is read in.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ScheduleZone {
    /// The owner's own clock (the system zone), with its daylight-saving rules.
    Owner,
    /// A numeric offset or UTC: no daylight saving.
    Fixed(FixedOffset),
    /// A named IANA zone such as `Europe/Madrid`, with its daylight-saving rules.
    Named(chrono_tz::Tz),
}

impl ScheduleZone {
    /// The calendar date `now` falls on in this zone.
    fn date_at(&self, now: DateTime<Local>) -> NaiveDate {
        match self {
            Self::Owner => now.date_naive(),
            Self::Fixed(offset) => now.with_timezone(offset).date_naive(),
            Self::Named(tz) => now.with_timezone(tz).date_naive(),
        }
    }

    /// The instant a wall-clock time in this zone names. A time the clocks
    /// skip (the spring-forward gap) is an error naming it; a time they pass
    /// twice (the fall-back hour) is the earlier of the two.
    fn resolve(&self, naive: NaiveDateTime) -> anyhow::Result<DateTime<Local>> {
        let resolved = match self {
            // The owner's zone by its IANA rules when it has a name: the
            // system clock's own lookup does not always report a fall-back
            // time as ambiguous.
            Self::Owner => {
                match owner_zone_name().and_then(|name| name.parse::<chrono_tz::Tz>().ok()) {
                    Some(tz) => tz
                        .from_local_datetime(&naive)
                        .map(|dt| dt.with_timezone(&Local)),
                    None => Local.from_local_datetime(&naive),
                }
            }
            Self::Fixed(offset) => offset
                .from_local_datetime(&naive)
                .map(|dt| dt.with_timezone(&Local)),
            Self::Named(tz) => tz
                .from_local_datetime(&naive)
                .map(|dt| dt.with_timezone(&Local)),
        };
        match resolved {
            LocalResult::Single(dt) => Ok(dt),
            LocalResult::Ambiguous(earlier, _) => Ok(earlier),
            LocalResult::None => anyhow::bail!(
                "{} does not exist in {}: the clocks skip it for daylight saving. Pick a time outside the skipped hour.",
                naive.format("%Y-%m-%d %H:%M"),
                self.name()
            ),
        }
    }

    fn name(&self) -> String {
        match self {
            Self::Owner => owner_zone_name().unwrap_or_else(system_timezone_display),
            Self::Fixed(offset) => offset.to_string(),
            Self::Named(tz) => tz.name().to_string(),
        }
    }
}

fn parse_day_time_with_optional_timezone(
    input: &str,
    now_local: DateTime<Local>,
) -> anyhow::Result<Option<DateTime<Local>>> {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)^(today|tonight|tomorrow)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s+(\S+))?$",
        )
        .expect("day-time regex")
    });
    let Some(caps) = RE.captures(input) else {
        return Ok(None);
    };
    let Some(day_kw) = caps.get(1).map(|m| m.as_str().to_ascii_lowercase()) else {
        return Ok(None);
    };
    let Some((hour, minute)) = parse_time_with_offset_captures(&caps, 2, 3, 4) else {
        return Ok(None);
    };
    let zone = match caps.get(5).map(|m| m.as_str().trim()) {
        Some(token) => parse_schedule_zone(token)?,
        None => ScheduleZone::Owner,
    };
    let today = zone.date_at(now_local);
    let base_date = match day_kw.as_str() {
        "tomorrow" => today.succ_opt().unwrap_or(today),
        _ => today,
    };
    let Some(naive) = base_date.and_hms_opt(hour, minute, 0) else {
        return Ok(None);
    };
    zone.resolve(naive).map(Some)
}

/// The zone a time names after it: a numeric offset (`-05:00`, `+0530`),
/// `UTC`/`GMT`/`Z`, or any IANA zone name (`Europe/Madrid`,
/// `America/Guayaquil`; exact, or the same name in another letter case),
/// read with that zone's daylight-saving rules. An abbreviation such as EST
/// or PST is refused, not mapped to a fixed offset: "EST" in July is read as
/// EDT by people and would fire an hour off. So are the IANA database's
/// legacy names without a region (`EST`, `CET`, `Japan`): a real zone name
/// has a `/`.
fn parse_schedule_zone(token: &str) -> anyhow::Result<ScheduleZone> {
    static NUMERIC: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^(?:UTC|GMT)?([+-])(\d{2})(?::?(\d{2}))?$").expect("numeric offset regex")
    });
    let token = token.trim();
    let upper = token.to_ascii_uppercase();
    if matches!(upper.as_str(), "UTC" | "GMT" | "Z") {
        return Ok(ScheduleZone::Fixed(
            FixedOffset::east_opt(0).expect("zero offset"),
        ));
    }
    if owner_zone_name().is_some_and(|zone| zone.eq_ignore_ascii_case(token)) {
        return Ok(ScheduleZone::Owner);
    }
    if let Some(caps) = NUMERIC.captures(&upper) {
        let hours: i32 = caps[2].parse()?;
        let minutes: i32 = caps.get(3).map_or(Ok(0), |m| m.as_str().parse())?;
        if hours <= 23 && minutes <= 59 {
            let seconds = hours * 3600 + minutes * 60;
            let offset = if &caps[1] == "+" {
                FixedOffset::east_opt(seconds)
            } else {
                FixedOffset::west_opt(seconds)
            };
            if let Some(offset) = offset {
                return Ok(ScheduleZone::Fixed(offset));
            }
        }
    }
    if token.contains('/') {
        if let Ok(tz) = token.parse::<chrono_tz::Tz>() {
            return Ok(ScheduleZone::Named(tz));
        }
        if let Some(tz) = chrono_tz::TZ_VARIANTS
            .iter()
            .find(|tz| tz.name().eq_ignore_ascii_case(token))
        {
            return Ok(ScheduleZone::Named(*tz));
        }
        anyhow::bail!(
            "'{token}' is not an IANA time zone name. Use a name like Europe/Madrid or America/New_York, a numeric offset like -05:00, or UTC."
        );
    }
    let example = zone_abbreviation_example(&upper).unwrap_or("America/New_York or Europe/Madrid");
    anyhow::bail!(
        "Time zone '{token}' is not one this can resolve exactly: an abbreviation hides daylight saving. Name the IANA zone instead (e.g. {example}), give a numeric offset like -05:00 or UTC, or leave it out to use the owner's timezone ({}).",
        owner_zone_name().unwrap_or_else(system_timezone_display)
    )
}

/// An IANA zone a common abbreviation usually means, for the refusal's hint
/// only: an abbreviation is ambiguous (IST, CST), so it is never resolved.
fn zone_abbreviation_example(upper: &str) -> Option<&'static str> {
    Some(match upper {
        "ET" | "EST" | "EDT" => "America/New_York",
        "CT" | "CST" | "CDT" => "America/Chicago",
        "MT" | "MST" | "MDT" => "America/Denver",
        "PT" | "PST" | "PDT" => "America/Los_Angeles",
        "AKST" | "AKDT" => "America/Anchorage",
        "HST" => "Pacific/Honolulu",
        "BST" | "WET" | "WEST" => "Europe/London",
        "CET" | "CEST" | "MET" | "MEST" => "Europe/Madrid",
        "EET" | "EEST" => "Europe/Athens",
        "MSK" => "Europe/Moscow",
        "IST" => "Asia/Kolkata",
        "JST" => "Asia/Tokyo",
        "KST" => "Asia/Seoul",
        "AEST" | "AEDT" => "Australia/Sydney",
        "NZST" | "NZDT" => "Pacific/Auckland",
        "ECT" => "America/Guayaquil",
        _ => return None,
    })
}

/// An exact date or date-time, read in the owner's zone, or in the zone a
/// trailing token names (`2026-03-14 09:30 Europe/Madrid`). RFC 3339 carries
/// its own offset.
fn parse_absolute_datetime(
    input: &str,
    now_local: DateTime<Local>,
) -> anyhow::Result<Option<DateTime<Local>>> {
    if let Some(dt) = parse_absolute_datetime_in(input, ScheduleZone::Owner, now_local)? {
        return Ok(Some(dt));
    }
    if let Some((rest, token)) = input.rsplit_once(char::is_whitespace) {
        let rest = rest.trim_end();
        // Only a trailing token after a date the parser reads is a zone; the
        // token is then held to `parse_schedule_zone`, so "... EST" is refused.
        // The shape is checked in UTC, which skips no time.
        let probe = ScheduleZone::Fixed(FixedOffset::east_opt(0).expect("zero offset"));
        if !rest.is_empty()
            && parse_absolute_datetime_in(rest, probe, now_local).is_ok_and(|dt| dt.is_some())
        {
            let zone = parse_schedule_zone(token)?;
            return parse_absolute_datetime_in(rest, zone, now_local);
        }
    }
    Ok(None)
}

fn parse_absolute_datetime_in(
    input: &str,
    zone: ScheduleZone,
    now_local: DateTime<Local>,
) -> anyhow::Result<Option<DateTime<Local>>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(input) {
        return Ok(Some(dt.with_timezone(&Local)));
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
            return zone.resolve(naive).map(Some);
        }
    }

    if let Some(dt) = parse_named_month_datetime(input, zone, now_local)? {
        return Ok(Some(dt));
    }

    let date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"];
    for fmt in date_formats {
        if let Ok(date) = NaiveDate::parse_from_str(input, fmt)
            && let Some(naive) = date.and_hms_opt(0, 0, 0)
        {
            return zone.resolve(naive).map(Some);
        }
    }

    Ok(None)
}

fn parse_named_month_datetime(
    input: &str,
    zone: ScheduleZone,
    now_local: DateTime<Local>,
) -> anyhow::Result<Option<DateTime<Local>>> {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)^(?:on\s+)?(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?(?:\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)?)?$",
        )
        .expect("named-month regex")
    });
    let Some(caps) = RE.captures(input.trim()) else {
        return Ok(None);
    };

    let Some(month) = caps
        .name("month")
        .and_then(|m| parse_month_name(m.as_str()))
    else {
        return Ok(None);
    };
    let Some(day) = caps
        .name("day")
        .and_then(|m| m.as_str().parse::<u32>().ok())
    else {
        return Ok(None);
    };
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
        return Ok(None);
    }

    let build = |year: i32| -> anyhow::Result<Option<DateTime<Local>>> {
        let Some(naive) =
            NaiveDate::from_ymd_opt(year, month, day).and_then(|d| d.and_hms_opt(hour, minute, 0))
        else {
            return Ok(None);
        };
        zone.resolve(naive).map(Some)
    };
    let year = explicit_year.unwrap_or_else(|| zone.date_at(now_local).year());
    let candidate = build(year)?;
    if explicit_year.is_none() && candidate.is_some_and(|dt| dt <= now_local) {
        return build(year + 1);
    }
    Ok(candidate)
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

/// Compute a fixed number of occurrences after an already-computed next run,
/// in the zone `next` carries (wall-clock times stay put across a
/// daylight-saving change).
pub fn following_runs<Tz: TimeZone>(
    cron_expr: &str,
    next: &DateTime<Tz>,
    count: usize,
) -> anyhow::Result<Vec<DateTime<Tz>>> {
    let cron: Cron = cron_expr
        .parse()
        .map_err(|e| anyhow::anyhow!("Failed to parse cron '{}': {}", cron_expr, e))?;
    let mut cursor = next.clone();
    let mut runs = Vec::with_capacity(count);
    for _ in 0..count {
        cursor = cron
            .find_next_occurrence(&cursor, false)
            .map_err(|e| anyhow::anyhow!("No next occurrence for '{}': {}", cron_expr, e))?;
        runs.push(cursor.clone());
    }
    Ok(runs)
}

/// The runs an owner confirms before scheduled work is saved: `first` (the
/// already-computed next run) and, for a recurring schedule, the ones after
/// it, `count` in all. A one-time schedule has only its single time. A cron
/// with no later occurrence lists what it has.
pub fn upcoming_runs<Tz: TimeZone>(
    cron_expr: &str,
    first: &DateTime<Tz>,
    one_shot: bool,
    count: usize,
) -> Vec<DateTime<Tz>> {
    let mut runs = vec![first.clone()];
    if !one_shot && count > 1 {
        runs.extend(following_runs(cron_expr, first, count - 1).unwrap_or_default());
    }
    runs.truncate(count.max(1));
    runs
}

#[cfg(test)]
mod following_runs_tests {
    use chrono::{Local, TimeZone};

    #[test]
    fn weekday_runs_skip_the_weekend_and_do_not_repeat_next() {
        let friday = Local
            .with_ymd_and_hms(2026, 11, 20, 8, 0, 0)
            .single()
            .unwrap();
        let runs = super::following_runs("0 8 * * 1-5", &friday, 2).unwrap();
        assert_eq!(runs[0].date_naive().to_string(), "2026-11-23");
        assert_eq!(runs[1].date_naive().to_string(), "2026-11-24");
        assert!(runs.iter().all(|run| run.time() == friday.time()));
    }

    /// US Eastern time (rules since 2007), test-only, so the daylight-saving
    /// cases do not depend on the machine's zone and need no zone crate.
    #[derive(Debug, Clone, Copy)]
    struct Eastern;

    impl Eastern {
        fn nth_sunday(year: i32, month: u32, n: u32) -> chrono::NaiveDate {
            use chrono::Datelike;
            let first = chrono::NaiveDate::from_ymd_opt(year, month, 1).unwrap();
            let to_sunday = (7 - first.weekday().num_days_from_sunday()) % 7;
            first + chrono::Duration::days(i64::from(to_sunday + 7 * (n - 1)))
        }

        fn utc_offset(utc: &chrono::NaiveDateTime) -> chrono::FixedOffset {
            use chrono::Datelike;
            // 2:00 AM local both times: 7:00 UTC in March, 6:00 UTC in November.
            let starts = Self::nth_sunday(utc.year(), 3, 2)
                .and_hms_opt(7, 0, 0)
                .unwrap();
            let ends = Self::nth_sunday(utc.year(), 11, 1)
                .and_hms_opt(6, 0, 0)
                .unwrap();
            let hours = if (starts..ends).contains(utc) { -4 } else { -5 };
            chrono::FixedOffset::east_opt(hours * 3600).unwrap()
        }
    }

    impl TimeZone for Eastern {
        type Offset = chrono::FixedOffset;

        fn from_offset(_offset: &chrono::FixedOffset) -> Self {
            Eastern
        }

        fn offset_from_local_date(
            &self,
            local: &chrono::NaiveDate,
        ) -> chrono::LocalResult<chrono::FixedOffset> {
            self.offset_from_local_datetime(&local.and_hms_opt(0, 0, 0).unwrap())
        }

        fn offset_from_local_datetime(
            &self,
            local: &chrono::NaiveDateTime,
        ) -> chrono::LocalResult<chrono::FixedOffset> {
            // Daylight time first, so an ambiguous time lists the earlier instant first.
            let fits: Vec<chrono::FixedOffset> = [-4, -5]
                .into_iter()
                .map(|hours| chrono::FixedOffset::east_opt(hours * 3600).unwrap())
                .filter(|offset| {
                    Self::utc_offset(
                        &(*local - chrono::Duration::seconds(offset.local_minus_utc().into())),
                    ) == *offset
                })
                .collect();
            match fits.as_slice() {
                [one] => chrono::LocalResult::Single(*one),
                [earlier, later] => chrono::LocalResult::Ambiguous(*earlier, *later),
                _ => chrono::LocalResult::None,
            }
        }

        fn offset_from_utc_date(&self, utc: &chrono::NaiveDate) -> chrono::FixedOffset {
            Self::utc_offset(&utc.and_hms_opt(0, 0, 0).unwrap())
        }

        fn offset_from_utc_datetime(&self, utc: &chrono::NaiveDateTime) -> chrono::FixedOffset {
            Self::utc_offset(utc)
        }
    }

    fn et(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> chrono::DateTime<Eastern> {
        Eastern
            .with_ymd_and_hms(y, mo, d, h, mi, 0)
            .single()
            .unwrap()
    }

    /// Formatted as every owner-facing time is, as seen on Mon Nov 16, 2026.
    fn shown(runs: &[chrono::DateTime<Eastern>]) -> Vec<String> {
        let now = et(2026, 11, 16, 9, 17).fixed_offset();
        runs.iter()
            .map(|run| super::format_owner_time(&run.fixed_offset(), &now, "ET"))
            .collect()
    }

    #[test]
    fn upcoming_runs_keep_the_wall_clock_across_the_fall_back() {
        // Daylight time ends Sun Nov 1, 2026: 9:00 AM stays 9:00 AM ET, and
        // that day's run is 25 hours after the last one.
        let runs = super::upcoming_runs("0 9 * * *", &et(2026, 10, 31, 9, 0), false, 3);
        assert_eq!(
            shown(&runs),
            vec![
                "Sat, Oct 31, 9:00 AM ET",
                "Sun, Nov 1, 9:00 AM ET",
                "Mon, Nov 2, 9:00 AM ET",
            ]
        );
        assert_eq!(runs[0].offset().local_minus_utc(), -4 * 3600);
        assert_eq!(runs[1].offset().local_minus_utc(), -5 * 3600);
        assert_eq!(runs[1] - runs[0], chrono::Duration::hours(25));
    }

    #[test]
    fn upcoming_runs_keep_the_wall_clock_across_the_spring_forward() {
        // Daylight time starts Sun Mar 8, 2026: that day's run is 23 hours later.
        let runs = super::upcoming_runs("30 8 * * *", &et(2026, 3, 7, 8, 30), false, 3);
        assert_eq!(
            shown(&runs),
            vec![
                "Sat, Mar 7, 8:30 AM ET",
                "Sun, Mar 8, 8:30 AM ET",
                "Mon, Mar 9, 8:30 AM ET",
            ]
        );
        assert_eq!(runs[1] - runs[0], chrono::Duration::hours(23));
    }

    #[test]
    fn a_wrong_cadence_shows_in_the_upcoming_runs() {
        // "Every minute" where "every month" was meant reads as minutes apart.
        let first = et(2026, 12, 1, 9, 0);
        assert_eq!(
            shown(&super::upcoming_runs("* * * * *", &first, false, 3)),
            vec![
                "Tue, Dec 1, 9:00 AM ET",
                "Tue, Dec 1, 9:01 AM ET",
                "Tue, Dec 1, 9:02 AM ET"
            ]
        );
        assert_eq!(
            shown(&super::upcoming_runs("0 9 1 * *", &first, false, 3)),
            vec![
                "Tue, Dec 1, 9:00 AM ET",
                "Jan 1, 2027, 9:00 AM ET",
                "Feb 1, 2027, 9:00 AM ET"
            ]
        );
    }

    #[test]
    fn a_one_time_schedule_has_its_single_time() {
        let first = et(2026, 11, 20, 9, 15);
        let runs = super::upcoming_runs("15 9 20 11 *", &first, true, 3);
        assert_eq!(shown(&runs), vec!["Fri, Nov 20, 9:15 AM ET"]);
        assert_eq!(super::upcoming_runs("0 9 * * *", &first, false, 1).len(), 1);
    }
}

/// How late a recurring run may start and still be worth starting: half its
/// period, between 5 minutes and 2 hours. A daily 9:00 AM digest the daemon
/// was off for is not sent at 11 PM; an hourly check missed by a few
/// minutes still runs, and a busy daemon's slow tick never skips one. `None` when the cron has no later occurrence.
pub fn catch_up_window(cron_expr: &str, due: DateTime<Utc>) -> Option<chrono::Duration> {
    let cron: Cron = cron_expr.parse().ok()?;
    let due_local = due.with_timezone(&Local);
    let after = cron.find_next_occurrence(&due_local, false).ok()?;
    let period = after.with_timezone(&Utc) - due;
    Some(
        (period / 2)
            .max(chrono::Duration::minutes(5))
            .min(chrono::Duration::hours(2)),
    )
}

#[cfg(test)]
mod catch_up_tests {
    use super::{catch_up_window, compute_next_run};
    use chrono::{Duration, Utc};

    #[test]
    fn a_late_run_may_start_within_half_its_period_between_5_minutes_and_2_hours() {
        let window = |cron: &str| catch_up_window(cron, compute_next_run(cron).unwrap());
        assert_eq!(window("0 9 * * *"), Some(Duration::hours(2)));
        assert_eq!(window("0 * * * *"), Some(Duration::minutes(30)));
        assert_eq!(window("* * * * *"), Some(Duration::minutes(5)));
        assert_eq!(catch_up_window("not a cron", Utc::now()), None);
    }
}

/// Human-readable system timezone label used in schedule confirmations: the
/// same short label every owner-facing time carries ([`owner_time`]).
pub fn system_timezone_display() -> String {
    owner_zone_label(Local::now().offset().local_minus_utc())
}

/// The IANA name of the owner's time zone. The daemon runs on the owner's
/// machine, so the owner's zone is the system one: `TZ` when it names a zone,
/// else the system setting.
fn owner_zone_name() -> Option<String> {
    static NAME: LazyLock<Option<String>> = LazyLock::new(|| {
        std::env::var("TZ")
            .ok()
            .map(|tz| tz.trim().trim_start_matches(':').to_string())
            .filter(|tz| tz.contains('/') || tz.eq_ignore_ascii_case("UTC"))
            .or_else(|| iana_time_zone::get_timezone().ok())
    });
    NAME.clone()
}

/// The owner's zone as a short label at a given UTC offset.
pub fn owner_zone_label(offset_secs: i32) -> String {
    zone_label(owner_zone_name().as_deref(), offset_secs)
}

/// A short label for a zone: the everyday name for zones people say by
/// name ("ET", "PT"), else the offset ("UTC+5:30"). One label per zone all
/// year, so a time never reads "EST" in one message and "UTC-05:00" in the
/// next.
pub fn zone_label(iana: Option<&str>, offset_secs: i32) -> String {
    const NAMED: &[(&str, &str)] = &[
        ("America/New_York", "ET"),
        ("America/Detroit", "ET"),
        ("America/Toronto", "ET"),
        ("America/Indiana/Indianapolis", "ET"),
        ("America/Kentucky/Louisville", "ET"),
        ("US/Eastern", "ET"),
        ("America/Chicago", "CT"),
        ("America/Winnipeg", "CT"),
        ("US/Central", "CT"),
        ("America/Denver", "MT"),
        ("America/Edmonton", "MT"),
        ("America/Boise", "MT"),
        ("America/Phoenix", "MT"),
        ("US/Mountain", "MT"),
        ("America/Los_Angeles", "PT"),
        ("America/Vancouver", "PT"),
        ("US/Pacific", "PT"),
        ("America/Anchorage", "AKT"),
        ("Pacific/Honolulu", "HT"),
        ("UTC", "UTC"),
        ("Etc/UTC", "UTC"),
        ("Etc/GMT", "UTC"),
    ];
    if let Some(label) = iana.and_then(|name| {
        NAMED
            .iter()
            .find(|(zone, _)| *zone == name)
            .map(|(_, label)| *label)
    }) {
        return label.to_string();
    }
    if offset_secs == 0 {
        return "UTC".to_string();
    }
    let sign = if offset_secs < 0 { '-' } else { '+' };
    let minutes = offset_secs.unsigned_abs() / 60;
    match minutes % 60 {
        0 => format!("UTC{sign}{}", minutes / 60),
        rest => format!("UTC{sign}{}:{rest:02}", minutes / 60),
    }
}

/// A time as the owner reads it, in their zone and one format: "2:39 PM ET"
/// today, "Thu, Nov 13, 2:39 PM ET" another day this year, and
/// "Jan 4, 2027, 2:39 PM ET" another year. Tool outputs keep the RFC 3339
/// value beside it where the model needs one.
pub fn owner_time<Tz: TimeZone>(at: &DateTime<Tz>) -> String {
    let at = at.with_timezone(&Local);
    let label = owner_zone_label(at.offset().local_minus_utc());
    format_owner_time(&at.fixed_offset(), &Local::now().fixed_offset(), &label)
}

/// A moment in full, in the owner's zone, for the model's prompt: the
/// weekday and date it needs plus the same clock and zone label as
/// [`owner_time`] ("Thursday, November 12, 2026, 2:39 PM ET"), so replies
/// speak the owner's time instead of UTC.
pub fn owner_date_time_long<Tz: TimeZone>(at: &DateTime<Tz>) -> String {
    let at = at.with_timezone(&Local);
    let label = owner_zone_label(at.offset().local_minus_utc());
    format_owner_date_time_long(&at.fixed_offset(), &label)
}

/// Testable core of [`owner_date_time_long`]: `at` in the owner's zone.
pub fn format_owner_date_time_long(at: &DateTime<FixedOffset>, zone_label: &str) -> String {
    format!(
        "{}, {} {zone_label}",
        at.format("%A, %B %-d, %Y"),
        at.format("%-I:%M %p")
    )
}

/// [`owner_time`] for an RFC 3339 timestamp; `None` when it does not parse.
pub fn owner_time_rfc3339(value: &str) -> Option<String> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|at| owner_time(&at))
}

/// Testable core of [`owner_time`]: `at` and `now` in the owner's zone.
pub fn format_owner_time(
    at: &DateTime<FixedOffset>,
    now: &DateTime<FixedOffset>,
    zone_label: &str,
) -> String {
    let clock = at.format("%-I:%M %p");
    if at.date_naive() == now.with_timezone(at.offset()).date_naive() {
        format!("{clock} {zone_label}")
    } else if at.year() == now.year() {
        format!("{}, {clock} {zone_label}", at.format("%a, %b %-d"))
    } else {
        format!("{}, {clock} {zone_label}", at.format("%b %-d, %Y"))
    }
}

/// Render a run time the way a person would say it: "today at 1:46 PM
/// (in 1 minute)", "tomorrow at 9:00 AM", "Friday at 9:00 AM", or
/// "on Jul 15 at 9:00 AM" for dates further out.
pub fn humanize_run_time(dt: DateTime<Local>) -> String {
    humanize_run_time_from(dt, Local::now())
}

/// Testable core of [`humanize_run_time`] with an explicit "now".
pub fn humanize_run_time_from(dt: DateTime<Local>, now: DateTime<Local>) -> String {
    let time = dt.format("%-I:%M %p").to_string();
    let days_ahead = (dt.date_naive() - now.date_naive()).num_days();
    match days_ahead {
        0 => {
            let mins = (dt - now).num_minutes();
            if (1..60).contains(&mins) {
                let unit = if mins == 1 { "minute" } else { "minutes" };
                format!("today at {} (in {} {})", time, mins, unit)
            } else {
                format!("today at {}", time)
            }
        }
        1 => format!("tomorrow at {}", time),
        2..=6 => format!("{} at {}", dt.format("%A"), time),
        _ if dt.year() == now.year() => format!("on {} at {}", dt.format("%b %-d"), time),
        _ => format!("on {} at {}", dt.format("%b %-d, %Y"), time),
    }
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
    fn the_prompt_gets_the_full_date_in_the_owners_zone() {
        let eastern = FixedOffset::west_opt(5 * 3600).unwrap();
        let at = eastern.with_ymd_and_hms(2026, 11, 12, 13, 57, 0).unwrap();
        assert_eq!(
            format_owner_date_time_long(&at, "ET"),
            "Thursday, November 12, 2026, 1:57 PM ET"
        );
    }

    #[test]
    fn owner_time_is_one_local_format_with_one_zone_label() {
        let eastern = FixedOffset::west_opt(5 * 3600).unwrap();
        let now = eastern.with_ymd_and_hms(2026, 11, 12, 9, 0, 0).unwrap();
        let today = eastern.with_ymd_and_hms(2026, 11, 12, 14, 39, 0).unwrap();
        let tomorrow = eastern.with_ymd_and_hms(2026, 11, 13, 14, 39, 0).unwrap();
        let next_year = eastern.with_ymd_and_hms(2027, 1, 4, 6, 0, 0).unwrap();
        assert_eq!(format_owner_time(&today, &now, "ET"), "2:39 PM ET");
        assert_eq!(
            format_owner_time(&tomorrow, &now, "ET"),
            "Fri, Nov 13, 2:39 PM ET"
        );
        assert_eq!(
            format_owner_time(&next_year, &now, "ET"),
            "Jan 4, 2027, 6:00 AM ET"
        );
    }

    #[test]
    fn zone_label_names_common_zones_and_falls_back_to_the_offset() {
        assert_eq!(zone_label(Some("America/New_York"), -5 * 3600), "ET");
        // The label does not change with daylight saving time.
        assert_eq!(zone_label(Some("America/New_York"), -4 * 3600), "ET");
        assert_eq!(zone_label(Some("America/Los_Angeles"), -8 * 3600), "PT");
        assert_eq!(zone_label(Some("Asia/Kolkata"), 19_800), "UTC+5:30");
        assert_eq!(zone_label(Some("Europe/Berlin"), 3600), "UTC+1");
        assert_eq!(zone_label(None, -5 * 3600), "UTC-5");
        assert_eq!(zone_label(None, 0), "UTC");
    }

    #[test]
    fn test_humanize_run_time() {
        let now = Local.with_ymd_and_hms(2026, 6, 11, 13, 45, 0).unwrap();
        let in_1m = Local.with_ymd_and_hms(2026, 6, 11, 13, 46, 0).unwrap();
        assert_eq!(
            humanize_run_time_from(in_1m, now),
            "today at 1:46 PM (in 1 minute)"
        );
        let later_today = Local.with_ymd_and_hms(2026, 6, 11, 18, 0, 0).unwrap();
        assert_eq!(humanize_run_time_from(later_today, now), "today at 6:00 PM");
        let tomorrow = Local.with_ymd_and_hms(2026, 6, 12, 9, 0, 0).unwrap();
        assert_eq!(humanize_run_time_from(tomorrow, now), "tomorrow at 9:00 AM");
        // 2026-06-15 is a Monday, 4 days out
        let monday = Local.with_ymd_and_hms(2026, 6, 15, 9, 0, 0).unwrap();
        assert_eq!(humanize_run_time_from(monday, now), "Monday at 9:00 AM");
        let next_month = Local.with_ymd_and_hms(2026, 7, 15, 9, 0, 0).unwrap();
        assert_eq!(
            humanize_run_time_from(next_month, now),
            "on Jul 15 at 9:00 AM"
        );
        let next_year = Local.with_ymd_and_hms(2027, 1, 5, 9, 0, 0).unwrap();
        assert_eq!(
            humanize_run_time_from(next_year, now),
            "on Jan 5, 2027 at 9:00 AM"
        );
    }

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
    fn test_parse_schedule_day_time_with_numeric_offset() {
        let offset = FixedOffset::west_opt(5 * 3600).unwrap();
        let now_there = Local::now().with_timezone(&offset);
        let target = now_there + chrono::Duration::minutes(30);
        let day_kw = if target.date_naive() == now_there.date_naive() {
            "today"
        } else {
            "tomorrow"
        };
        let input = format!("{} at {} -05:00.", day_kw, target.format("%I:%M%P"));
        let cron = parse_schedule(&input).unwrap();
        let local = target.with_timezone(&Local);
        assert_eq!(
            cron,
            format!(
                "{} {} {} {} *",
                local.minute(),
                local.hour(),
                local.day(),
                local.month()
            )
        );
    }

    #[test]
    fn a_zone_abbreviation_is_refused_not_mapped_to_a_fixed_offset() {
        for input in [
            "tomorrow at 9am EST",
            "tomorrow at 9am PST",
            "tomorrow at 9am CET",
        ] {
            let error = parse_schedule(input).unwrap_err().to_string();
            assert!(error.contains("numeric offset"), "{input}: {error}");
        }
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
    fn test_parse_schedule_zone_numeric_utc_owner_and_iana() {
        let seconds = |token: &str| match parse_schedule_zone(token).unwrap() {
            ScheduleZone::Fixed(offset) => offset.local_minus_utc(),
            other => panic!("{token}: {other:?}"),
        };
        assert_eq!(seconds("+05:30"), 5 * 3600 + 30 * 60);
        assert_eq!(seconds("-0400"), -4 * 3600);
        assert_eq!(seconds("UTC"), 0);
        assert_eq!(seconds("UTC-03:00"), -3 * 3600);
        // The owner's IANA zone means "the owner's own clock". A host on UTC
        // (CI runners) names its zone "UTC", which reads as offset zero first.
        if let Some(zone) = owner_zone_name()
            .filter(|zone| !matches!(zone.to_ascii_uppercase().as_str(), "UTC" | "GMT" | "Z"))
        {
            assert_eq!(parse_schedule_zone(&zone).unwrap(), ScheduleZone::Owner);
        }
        // Any other IANA zone, exact or in another letter case.
        for (token, name) in [
            ("Asia/Tokyo", "Asia/Tokyo"),
            ("america/guayaquil", "America/Guayaquil"),
            ("EUROPE/MADRID", "Europe/Madrid"),
        ] {
            if owner_zone_name().is_some_and(|zone| zone.eq_ignore_ascii_case(token)) {
                continue;
            }
            match parse_schedule_zone(token).unwrap() {
                ScheduleZone::Named(tz) => assert_eq!(tz.name(), name),
                other => panic!("{token}: {other:?}"),
            }
        }
        assert!(parse_schedule_zone("EDT").is_err());
    }

    /// Noon UTC on a fixed day, as the owner's clock: a fixed "now" so the
    /// zone tests do not depend on the date they run.
    fn fixed_now(y: i32, mo: u32, d: u32) -> DateTime<Local> {
        Utc.with_ymd_and_hms(y, mo, d, 12, 0, 0)
            .unwrap()
            .with_timezone(&Local)
    }

    fn utc(input: &str, now: DateTime<Local>) -> String {
        parse_absolute_datetime(input, now)
            .unwrap()
            .unwrap_or_else(|| panic!("{input}: not read"))
            .with_timezone(&Utc)
            .to_rfc3339()
    }

    #[test]
    fn a_named_zone_follows_its_daylight_saving_rules() {
        let now = fixed_now(2030, 1, 1);
        // Madrid is UTC+1 in winter and UTC+2 in summer.
        assert_eq!(
            utc("2030-01-15 09:00 Europe/Madrid", now),
            "2030-01-15T08:00:00+00:00"
        );
        assert_eq!(
            utc("2030-07-15 09:00 Europe/Madrid", now),
            "2030-07-15T07:00:00+00:00"
        );
        assert_eq!(
            utc("July 15, 2030 at 9am Europe/Madrid", now),
            "2030-07-15T07:00:00+00:00"
        );
        // Guayaquil keeps UTC-5 all year.
        assert_eq!(
            utc("2030-01-15 09:00 America/Guayaquil", now),
            "2030-01-15T14:00:00+00:00"
        );
        assert_eq!(
            utc("2030-07-15 09:00 America/Guayaquil", now),
            "2030-07-15T14:00:00+00:00"
        );
        assert_eq!(
            utc("2030-07-15 09:00 Asia/Tokyo", now),
            "2030-07-15T00:00:00+00:00"
        );
        // A fall-back time the clocks pass twice is the earlier one (CEST).
        assert_eq!(
            utc("2030-10-27 02:30 Europe/Madrid", now),
            "2030-10-27T00:30:00+00:00"
        );
        // The owner's zone and numeric offsets still read as before.
        assert_eq!(
            utc("2030-07-15 09:00 -05:00", now),
            "2030-07-15T14:00:00+00:00"
        );
        assert_eq!(
            utc("2030-07-15 09:00 UTC", now),
            "2030-07-15T09:00:00+00:00"
        );
    }

    #[test]
    fn tomorrow_at_a_time_in_a_named_zone_uses_that_zones_calendar_and_offset() {
        let resolve = |input: &str, now: DateTime<Local>| {
            parse_day_time_with_optional_timezone(input, now)
                .unwrap()
                .unwrap()
                .with_timezone(&Utc)
                .to_rfc3339()
        };
        assert_eq!(
            resolve("tomorrow at 9am Europe/Madrid", fixed_now(2030, 1, 14)),
            "2030-01-15T08:00:00+00:00"
        );
        assert_eq!(
            resolve("tomorrow at 9am Europe/Madrid", fixed_now(2030, 7, 14)),
            "2030-07-15T07:00:00+00:00"
        );
        assert_eq!(
            resolve("tomorrow at 9am America/Guayaquil", fixed_now(2030, 7, 14)),
            "2030-07-15T14:00:00+00:00"
        );
        // 12:00 UTC on the 14th is already the 14th at 21:00 in Tokyo.
        assert_eq!(
            resolve("tomorrow at 9:30am Asia/Tokyo", fixed_now(2030, 7, 14)),
            "2030-07-15T00:30:00+00:00"
        );
    }

    #[test]
    fn a_time_the_clocks_skip_in_a_named_zone_is_an_error_naming_it() {
        let now = fixed_now(2030, 1, 1);
        let error = parse_absolute_datetime("2030-03-31 02:30 Europe/Madrid", now)
            .unwrap_err()
            .to_string();
        assert!(error.contains("2030-03-31 02:30"), "{error}");
        assert!(error.contains("Europe/Madrid"), "{error}");
        assert!(error.contains("daylight saving"), "{error}");
        let error = parse_day_time_with_optional_timezone(
            "tomorrow at 2:30am Europe/Madrid",
            fixed_now(2030, 3, 30),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("does not exist"), "{error}");
    }

    #[test]
    fn an_abbreviation_or_unknown_zone_after_a_time_is_refused_with_a_hint() {
        let now = fixed_now(2030, 1, 1);
        let error = parse_absolute_datetime("2030-07-15 09:00 EST", now)
            .unwrap_err()
            .to_string();
        assert!(error.contains("America/New_York"), "{error}");
        let error = parse_schedule("tomorrow at 9am CET")
            .unwrap_err()
            .to_string();
        assert!(error.contains("Europe/Madrid"), "{error}");
        // IANA's legacy region-less names are abbreviations too.
        assert!(parse_schedule_zone("EST").is_err());
        assert!(parse_schedule_zone("Japan").is_err());
        let error = parse_absolute_datetime("2030-07-15 09:00 Europe/Atlantis", now)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not an IANA time zone"), "{error}");
        assert!(parse_schedule("tomorrow at 9am Mars/Olympus").is_err());
    }

    #[test]
    fn test_parse_schedule_relative_duration_phrase() {
        let before = Local::now();
        let cron = parse_schedule("in 1 hour 30 minutes").unwrap();
        let expected = before + chrono::Duration::minutes(90);
        let parts: Vec<&str> = cron.split_whitespace().collect();
        assert_eq!(parts[1], expected.hour().to_string());
        for input in [
            "in 2h30m",
            "after 90 minutes",
            "in 1 hour, 30 min",
            "in 1 day and 2 hours",
            "in 3 weeks",
        ] {
            assert!(parse_schedule(input).is_ok(), "{input}");
        }
    }

    #[test]
    fn a_relative_delay_it_cannot_fully_read_is_refused_not_guessed() {
        // "2 months" once read as 2 minutes ("m"), and "at 9am" was dropped.
        for input in [
            "in 2 months",
            "in 10 days at 9am",
            "in 1 year",
            "in 5 minutes or so",
            "after 0 minutes",
        ] {
            let error = parse_schedule(input).unwrap_err().to_string();
            assert!(error.contains("not a delay"), "{input}: {error}");
        }
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
}
