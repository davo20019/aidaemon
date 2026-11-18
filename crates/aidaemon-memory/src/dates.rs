//! Dates the people store keeps (birthdays, important dates). They are
//! machine dates the model fills: ISO 8601 `YYYY-MM-DD`, or `--MM-DD` when
//! the year is unknown. Nothing else is read: a month name or a
//! `05/03`-style date is locale prose (May 3 or March 5?), and a wrong
//! reading sends a reminder on the wrong day.

use chrono::{Datelike, NaiveDate};

/// The categories of person facts whose value is such a date.
pub const DATE_FACT_CATEGORIES: &[&str] = &["birthday", "important_date"];

/// Whether a person-fact `category` holds a date.
pub fn is_date_fact_category(category: &str) -> bool {
    DATE_FACT_CATEGORIES.contains(&category.trim().to_ascii_lowercase().as_str())
}

/// The `(month, day)` of an ISO date `YYYY-MM-DD` or a yearless `--MM-DD`;
/// `None` for anything else.
pub fn parse_month_day(value: &str) -> Option<(u32, u32)> {
    let value = value.trim();
    if let Some(month_day) = value.strip_prefix("--") {
        // Validate against a leap year so --02-29 is a real yearless date.
        let date = NaiveDate::parse_from_str(&format!("2000-{month_day}"), "%Y-%m-%d").ok()?;
        return (month_day.len() == 5).then_some((date.month(), date.day()));
    }
    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d").ok()?;
    (value.len() == 10).then_some((date.month(), date.day()))
}

/// Why a date argument was refused, for the model to correct.
pub fn date_argument_error(value: &str) -> Option<String> {
    parse_month_day(value).is_none().then(|| {
        format!(
            "'{}' is not an ISO date. Give YYYY-MM-DD, or --MM-DD when the year is unknown \
             (May 3 is --05-03); ask the owner when the day and month are ambiguous.",
            value.trim()
        )
    })
}

/// Days from `today` until the date's next anniversary (0 today), or `None`
/// when `value` is not an ISO date or the day does not occur this year
/// (Feb 29 outside a leap year).
pub fn days_until_anniversary(value: &str, today: NaiveDate) -> Option<i64> {
    let (month, day) = parse_month_day(value)?;
    let this_year = NaiveDate::from_ymd_opt(today.year(), month, day)?;
    let diff = (this_year - today).num_days();
    Some(if diff < 0 { diff + 365 } else { diff })
}

/// A value stored before dates were ISO only, rewritten as ISO the way
/// the old reader understood it: `YYYY-M-D`, numeric month first (`3/15`,
/// `03-15`: March 15, as the old code read them), or an English month name
/// and a day (`March 15`, `march 15,`). `None` when `value` is already ISO
/// or the old reader could not read it either; those stay as written.
pub fn legacy_date_to_iso(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || parse_month_day(value).is_some() {
        return None;
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Some(date.format("%Y-%m-%d").to_string());
    }
    // Month-day forms, month first as the old reader meant them, checked
    // against the leap year 2000 so 02-29 stays a real date. (The old
    // reader's own format left out the year and so never matched; its
    // intent, month first, is the reading kept here.)
    for format in ["%Y-%m-%d", "%Y-%m/%d"] {
        if let Ok(date) = NaiveDate::parse_from_str(&format!("2000-{value}"), format) {
            return Some(format!("--{:02}-{:02}", date.month(), date.day()));
        }
    }
    const MONTHS: [&str; 12] = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    let lower = value.to_lowercase();
    for (index, name) in MONTHS.iter().enumerate() {
        if let Some(rest) = lower.strip_prefix(name) {
            let rest = rest.trim().trim_start_matches([',', ' ']);
            let month = index as u32 + 1;
            let day = rest.parse::<u32>().ok()?;
            NaiveDate::from_ymd_opt(2000, month, day)?;
            return Some(format!("--{month:02}-{day:02}"));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn only_iso_dates_are_read() {
        assert_eq!(parse_month_day("2019-04-02"), Some((4, 2)));
        assert_eq!(parse_month_day("--05-03"), Some((5, 3)));
        assert_eq!(parse_month_day("--02-29"), Some((2, 29)));
        for ambiguous in [
            "05/03", "5/3", "05-03", "March 15", "15 March", "2019-4-2", "--13-01",
        ] {
            assert_eq!(parse_month_day(ambiguous), None, "{ambiguous}");
            assert!(date_argument_error(ambiguous).is_some(), "{ambiguous}");
        }
        assert!(date_argument_error("--05-03").is_none());
    }

    #[test]
    fn legacy_dates_convert_as_the_old_reader_read_them() {
        for (legacy, iso) in [
            ("March 15", "--03-15"),
            ("march 5", "--03-05"),
            ("December, 1", "--12-01"),
            ("03-15", "--03-15"),
            ("3/15", "--03-15"),
            // Month first, as the old reader read it: May 3, not March 5.
            ("05/03", "--05-03"),
            ("02-29", "--02-29"),
            ("2019-4-2", "2019-04-02"),
        ] {
            assert_eq!(legacy_date_to_iso(legacy).as_deref(), Some(iso), "{legacy}");
            assert!(parse_month_day(iso).is_some(), "{iso}");
        }
        for untouched in [
            "--03-15",
            "2019-04-02",
            "15 March",
            "March 15th",
            "13/01",
            "February 30",
            "next spring",
            "",
        ] {
            assert_eq!(legacy_date_to_iso(untouched), None, "{untouched}");
        }
    }

    #[test]
    fn days_count_to_the_next_anniversary() {
        let today = day(2026, 5, 1);
        assert_eq!(days_until_anniversary("--05-03", today), Some(2));
        assert_eq!(days_until_anniversary("2019-05-01", today), Some(0));
        assert_eq!(days_until_anniversary("--04-30", today), Some(364));
        // The 31st of a month does not break a later shorter month.
        assert_eq!(
            days_until_anniversary("--06-30", day(2026, 5, 31)),
            Some(30)
        );
        assert_eq!(days_until_anniversary("05/03", today), None);
    }
}
