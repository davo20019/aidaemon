//! Automations: what the owner sees of the two engines that run work on its
//! own. A *scheduled* automation runs a fixed instruction at set times (a
//! scheduled goal, or a reminder that only messages the owner). A *watching*
//! automation works toward an objective within limits the owner approved (a
//! mandate). One vocabulary, one list and one card for both; the engines and
//! their data are unchanged.

use aidaemon_core::mandate_card::MandateConfirmationCard;

/// The two kinds of automation, as the owner and the model name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomationKind {
    /// Runs a fixed instruction at set times (scheduled goals, reminders).
    Scheduled,
    /// Works toward an objective within approved limits (mandates).
    Watching,
}

impl AutomationKind {
    pub const ALL: [Self; 2] = [Self::Scheduled, Self::Watching];

    /// The typed value in tool arguments.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Watching => "watching",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == value.trim())
    }

    /// The kind as the owner reads it.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Scheduled => "Scheduled",
            Self::Watching => "Watching",
        }
    }
}

/// Goal-context key holding a scheduled automation's owner-given name.
/// Mandates keep their name in their own typed column.
pub const NAME_KEY: &str = "automation_name";

/// The owner-given name stored in a scheduled goal's context, if any.
pub fn name_from_context(context: Option<&str>) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(context?).ok()?;
    let name = value.get(NAME_KEY)?.as_str()?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// `context` with the name added, keeping every other key.
pub fn context_with_name(context: Option<&str>, name: &str) -> String {
    let mut object = context
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    object.insert(NAME_KEY.to_string(), serde_json::json!(name.trim()));
    serde_json::Value::Object(object).to_string()
}

/// A goal's lifecycle status as the owner reads it.
pub fn state_words(status: &str) -> &'static str {
    match status {
        "active" => "On",
        "paused" => "Paused",
        "pending_confirmation" | "pending" => "Waiting for your approval",
        "awaiting_input" => "Waiting for your answer",
        "failed" => "Stopped after a problem",
        "completed" => "Done",
        "cancelled" | "abandoned" => "Cancelled",
        _ => "Unknown",
    }
}

/// One schedule of a scheduled automation: its cron (in the owner's zone)
/// and whether it runs once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleSpec {
    pub cron: String,
    pub one_shot: bool,
    /// The next run, already in the owner's format.
    pub next: Option<String>,
}

fn clock(minute: &str, hour: &str, zone: &str) -> Option<String> {
    let minute: u32 = minute.parse().ok()?;
    let hour: u32 = hour.parse().ok()?;
    let time = chrono::NaiveTime::from_hms_opt(hour, minute, 0)?;
    Some(format!("{} {zone}", time.format("%-I:%M %p")))
}

fn weekday(day: &str) -> Option<&'static str> {
    Some(match day.parse::<u8>().ok()? {
        0 | 7 => "Sunday",
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6 => "Saturday",
        _ => return None,
    })
}

fn plural(count: u32, unit: &str) -> String {
    if count == 1 {
        unit.to_string()
    } else {
        format!("{count} {unit}s")
    }
}

/// How often one schedule runs, in plain words in the owner's zone
/// (`zone`, e.g. "ET"): "Every weekday at 8:00 AM ET", "Once, Thu, Nov 13,
/// 9:00 AM ET". A cron the words do not cover says when it runs next.
pub fn cadence_words(schedule: &ScheduleSpec, zone: &str) -> String {
    let fallback = || match &schedule.next {
        Some(next) => format!("On a fixed schedule; next {next}"),
        None => "On a fixed schedule".to_string(),
    };
    if schedule.one_shot {
        return match &schedule.next {
            Some(next) => format!("Once, {next}"),
            None => "Once".to_string(),
        };
    }
    let fields: Vec<&str> = schedule.cron.split_whitespace().collect();
    let [minute, hour, dom, month, dow] = fields.as_slice() else {
        return fallback();
    };
    let every = |count: &str, unit: &str| {
        count
            .strip_prefix("*/")
            .and_then(|n| n.parse::<u32>().ok())
            .map(|n| format!("Every {}", plural(n, unit)))
    };
    match (*minute, *hour, *dom, *month, *dow) {
        ("*", "*", "*", "*", "*") => "Every minute".to_string(),
        (m, "*", "*", "*", "*") if m.starts_with("*/") => {
            every(m, "minute").unwrap_or_else(fallback)
        }
        (m, "*", "*", "*", "*") if m.parse::<u32>().is_ok() => "Every hour".to_string(),
        (m, h, "*", "*", "*") if m.parse::<u32>().is_ok() && h.starts_with("*/") => {
            every(h, "hour").unwrap_or_else(fallback)
        }
        (m, h, d, "*", w) => {
            let Some(at) = clock(m, h, zone) else {
                return fallback();
            };
            match (d, w) {
                ("*", "*") => format!("Every day at {at}"),
                ("*", "1-5") => format!("Every weekday at {at}"),
                ("*", "0,6" | "6,0" | "6,7" | "6-7") => format!("Every weekend at {at}"),
                ("*", days) => {
                    let named: Option<Vec<&str>> = days.split(',').map(weekday).collect();
                    match named {
                        Some(named) if !named.is_empty() => {
                            format!("Every {} at {at}", named.join(" and "))
                        }
                        _ => fallback(),
                    }
                }
                (day, "*") if day.parse::<u32>().is_ok_and(|d| (1..=31).contains(&d)) => {
                    format!("Every month on day {day} at {at}")
                }
                _ => fallback(),
            }
        }
        _ => fallback(),
    }
}

/// Every schedule's cadence, joined.
pub fn cadence(schedules: &[ScheduleSpec], zone: &str) -> String {
    if schedules.is_empty() {
        return "No schedule left – it will not run".to_string();
    }
    schedules
        .iter()
        .map(|schedule| cadence_words(schedule, zone))
        .collect::<Vec<_>>()
        .join("; ")
}

/// A scheduled automation as the owner reads it, in the same five lines as
/// a watching one: what, how often, what it can do, when it tells you,
/// when it ends. A reminder only messages the owner; scheduled work runs
/// with the owner's full access, as it always has.
pub fn scheduled_card(
    what: &str,
    reminder: Option<&str>,
    schedules: &[ScheduleSpec],
    zone: &str,
) -> MandateConfirmationCard {
    let runs_once = !schedules.is_empty() && schedules.iter().all(|s| s.one_shot);
    let (what, can_do, tells_you_when) = match reminder {
        Some(message) => (
            format!("Reminds you: {}", message.trim()),
            "Only sends you this message – nothing else runs".to_string(),
            "it goes off".to_string(),
        ),
        None => (
            what.split_whitespace().collect::<Vec<_>>().join(" "),
            "Anything you can do – it runs with your full access".to_string(),
            "a run finishes, or it hits a problem".to_string(),
        ),
    };
    MandateConfirmationCard {
        what,
        how_often: cadence(schedules, zone),
        can_do,
        tells_you_when,
        ends: if runs_once {
            "After it runs once".to_string()
        } else {
            "Only when you stop it".to_string()
        },
        details: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recurring(cron: &str) -> ScheduleSpec {
        ScheduleSpec {
            cron: cron.to_string(),
            one_shot: false,
            next: Some("Thu, Nov 13, 8:00 AM ET".to_string()),
        }
    }

    #[test]
    fn cadence_reads_in_plain_words_in_the_owners_zone() {
        let cases = [
            ("0 8 * * 1-5", "Every weekday at 8:00 AM ET"),
            ("30 18 * * *", "Every day at 6:30 PM ET"),
            ("0 9 * * 1", "Every Monday at 9:00 AM ET"),
            ("0 9 * * 1,3", "Every Monday and Wednesday at 9:00 AM ET"),
            ("0 9 1 * *", "Every month on day 1 at 9:00 AM ET"),
            ("*/15 * * * *", "Every 15 minutes"),
            ("0 */2 * * *", "Every 2 hours"),
            ("0 * * * *", "Every hour"),
            (
                "0 9 * 1 *",
                "On a fixed schedule; next Thu, Nov 13, 8:00 AM ET",
            ),
        ];
        for (cron, words) in cases {
            assert_eq!(cadence_words(&recurring(cron), "ET"), words, "{cron}");
        }
        let once = ScheduleSpec {
            cron: "0 9 13 11 *".to_string(),
            one_shot: true,
            next: Some("Thu, Nov 13, 9:00 AM ET".to_string()),
        };
        assert_eq!(cadence_words(&once, "ET"), "Once, Thu, Nov 13, 9:00 AM ET");
    }

    #[test]
    fn a_scheduled_card_has_the_five_lines_of_a_watching_one() {
        let work = scheduled_card(
            "Summarize  my inbox",
            None,
            &[recurring("0 8 * * 1-5")],
            "ET",
        );
        assert_eq!(
            work.clone().into_lines(),
            vec![
                "What: Summarize my inbox",
                "How often: Every weekday at 8:00 AM ET",
                "Can do: Anything you can do – it runs with your full access",
                "Tells you when: a run finishes, or it hits a problem",
                "Ends: Only when you stop it",
            ]
        );
        assert_eq!(
            MandateConfirmationCard::from_lines(&work.into_lines()).map(|card| card.what),
            Some("Summarize my inbox".to_string())
        );

        let reminder = scheduled_card(
            "Check the oven.",
            Some("Check the oven."),
            &[ScheduleSpec {
                cron: "5 14 12 11 *".to_string(),
                one_shot: true,
                next: Some("2:05 PM ET".to_string()),
            }],
            "ET",
        );
        assert_eq!(reminder.what, "Reminds you: Check the oven.");
        assert_eq!(
            reminder.can_do,
            "Only sends you this message – nothing else runs"
        );
        assert_eq!(reminder.ends, "After it runs once");
    }

    #[test]
    fn the_name_lives_beside_other_context_keys() {
        let context = crate::reminders::context_for_message("Check the oven.");
        let named = context_with_name(Some(&context), "  Oven  ");
        assert_eq!(name_from_context(Some(&named)).as_deref(), Some("Oven"));
        assert_eq!(
            crate::reminders::message_from_context(Some(&named)).as_deref(),
            Some("Check the oven.")
        );
        assert_eq!(name_from_context(None), None);
        assert_eq!(
            name_from_context(Some(&context_with_name(None, "Inbox"))).as_deref(),
            Some("Inbox")
        );
    }

    #[test]
    fn kinds_parse_from_their_typed_value() {
        assert_eq!(
            AutomationKind::parse("watching"),
            Some(AutomationKind::Watching)
        );
        assert_eq!(
            AutomationKind::parse("scheduled"),
            Some(AutomationKind::Scheduled)
        );
        assert_eq!(AutomationKind::parse("mandate"), None);
    }
}
