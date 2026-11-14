//! The owner's confirmation card for a mandate: five essentials the owner
//! reads at a glance, and the rest as details a channel can fold away.
//!
//! The card travels as [`aidaemon_core::OwnerApprovalRequest::warnings`]
//! lines. The essentials come first, each as `"<label>: <text>"` with a
//! label from [`Essential`], so a renderer can find them with
//! [`Essential::parse_line`] (or rebuild the card with
//! [`MandateConfirmationCard::from_lines`]) instead of guessing from prose.
//! Every other line is a detail.

pub use aidaemon_core::mandate_card::{Essential, MandateConfirmationCard};

/// A whole number of seconds as words: "1 hour", "30 minutes", "2 days".
pub fn duration_words(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let (count, unit) = if seconds > 0 && seconds % 86_400 == 0 {
        (seconds / 86_400, "day")
    } else if seconds > 0 && seconds % 3_600 == 0 {
        (seconds / 3_600, "hour")
    } else if seconds > 0 && seconds % 60 == 0 {
        (seconds / 60, "minute")
    } else {
        (seconds, "second")
    };
    format!("{count} {unit}{}", if count == 1 { "" } else { "s" })
}

/// "hour" for one unit, "2 hours" for more, so "every hour" reads naturally.
fn every_words(seconds: i64) -> String {
    let words = duration_words(seconds);
    words
        .strip_prefix("1 ")
        .map_or(words.clone(), str::to_string)
}

/// How often a mandate reviews: one value when its bounds agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReviewCadence {
    pub min_secs: i64,
    pub default_secs: i64,
    pub max_secs: i64,
}

impl ReviewCadence {
    pub fn render(self) -> String {
        if self.min_secs == self.max_secs {
            return format!("Every {}", every_words(self.max_secs));
        }
        format!(
            "About every {} (between {} and {})",
            every_words(self.default_secs),
            duration_words(self.min_secs),
            duration_words(self.max_secs)
        )
    }
}

/// What a mandate may do on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanDo {
    /// It may look, never change anything.
    ReadOnly,
    /// It may make these changes within these limits.
    Changes {
        /// What it may change, in plain words ("publish posts on X").
        what: Vec<String>,
        per_review: u32,
        per_day: u32,
        min_gap_secs: u32,
    },
}

impl CanDo {
    pub fn render(&self) -> String {
        match self {
            Self::ReadOnly => "Read only – makes no changes".to_string(),
            Self::Changes {
                what,
                per_review,
                per_day,
                min_gap_secs,
            } => {
                let what = if what.is_empty() {
                    "Change things you approved".to_string()
                } else {
                    let mut joined = what.join("; ");
                    if let Some(first) = joined.get(..1) {
                        joined = first.to_uppercase() + &joined[1..];
                    }
                    joined
                };
                let mut limits = vec![format!(
                    "up to {per_review} change{} per review",
                    if *per_review == 1 { "" } else { "s" }
                )];
                if *per_day > 0 {
                    limits.push(format!("{per_day} a day"));
                }
                let mut text = format!("{what} – {}", limits.join(" and "));
                if *min_gap_secs > 0 {
                    text.push_str(&format!(
                        ", at least {} apart",
                        duration_words((*min_gap_secs).into())
                    ));
                }
                text
            }
        }
    }
}

/// Something that makes a mandate message the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeTrigger {
    /// A finding the owner asked to hear about (`owner_alerts`).
    Finding,
    /// A change it made.
    Change,
    /// A question only the owner can answer.
    NeedsInput,
    /// It stopped, failed or paused for safety.
    Problem,
}

impl NoticeTrigger {
    const fn words(self) -> &'static str {
        match self {
            Self::Finding => "it finds something you asked to hear about",
            Self::Change => "it makes a change",
            Self::NeedsInput => "it needs your input",
            Self::Problem => "it hits a problem",
        }
    }

    /// "a, b, or c".
    pub fn render(triggers: &[Self]) -> String {
        let words = triggers
            .iter()
            .map(|trigger| trigger.words())
            .collect::<Vec<_>>();
        match words.as_slice() {
            [] => "only when you ask".to_string(),
            [one] => (*one).to_string(),
            [head @ .., last] => format!("{}, or {last}", head.join(", ")),
        }
    }
}

/// When a mandate ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndLimit {
    /// This long after the owner approves it.
    AfterApproval { secs: i64 },
    /// At this time, already in the owner's format.
    At(String),
    /// No end date.
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ends {
    pub limit: EndLimit,
    /// It also ends when one of its stop conditions is met.
    pub has_stop_conditions: bool,
}

impl Ends {
    pub fn render(&self) -> String {
        let limit = match &self.limit {
            EndLimit::AfterApproval { secs } => {
                Some(format!("{} after you approve", duration_words(*secs)))
            }
            EndLimit::At(at) => Some(at.clone()),
            EndLimit::None => None,
        };
        match (limit, self.has_stop_conditions) {
            (Some(limit), false) => limit,
            (Some(limit), true) => format!("{limit}, or sooner if a stop condition is met"),
            (None, true) => "When a stop condition is met, or when you stop it".to_string(),
            (None, false) => "Only when you stop it".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card() -> MandateConfirmationCard {
        MandateConfirmationCard {
            what: "Check the status page".to_string(),
            how_often: ReviewCadence {
                min_secs: 3_600,
                default_secs: 3_600,
                max_secs: 3_600,
            }
            .render(),
            can_do: CanDo::ReadOnly.render(),
            tells_you_when: NoticeTrigger::render(&[
                NoticeTrigger::Finding,
                NoticeTrigger::NeedsInput,
                NoticeTrigger::Problem,
            ]),
            ends: Ends {
                limit: EndLimit::None,
                has_stop_conditions: false,
            }
            .render(),
            details: vec!["Success criteria: • You hear about outages".to_string()],
        }
    }

    #[test]
    fn a_read_only_hourly_mandate_reads_as_five_plain_lines() {
        let lines = card().into_lines();
        assert_eq!(
            lines,
            vec![
                "What: Check the status page",
                "How often: Every hour",
                "Can do: Read only – makes no changes",
                "Tells you when: it finds something you asked to hear about, it needs your input, or it hits a problem",
                "Ends: Only when you stop it",
                "Success criteria: • You hear about outages",
            ]
        );
        assert_eq!(MandateConfirmationCard::from_lines(&lines), Some(card()));
    }

    #[test]
    fn a_detail_line_is_never_an_essential() {
        assert_eq!(Essential::parse_line("Success criteria: x"), None);
        assert_eq!(
            Essential::parse_line("Can do: Read only – makes no changes"),
            Some((Essential::CanDo, "Read only – makes no changes"))
        );
        let mut lines = card().into_lines();
        lines.remove(1);
        assert_eq!(MandateConfirmationCard::from_lines(&lines), None);
    }

    #[test]
    fn cadence_shows_one_value_when_bounds_agree_and_a_range_otherwise() {
        let fixed = ReviewCadence {
            min_secs: 7_200,
            default_secs: 7_200,
            max_secs: 7_200,
        };
        assert_eq!(fixed.render(), "Every 2 hours");
        let varied = ReviewCadence {
            min_secs: 1_800,
            default_secs: 3_600,
            max_secs: 86_400,
        };
        assert_eq!(
            varied.render(),
            "About every hour (between 30 minutes and 1 day)"
        );
    }

    #[test]
    fn changes_show_their_limits_and_zero_limits_are_read_only() {
        let changes = CanDo::Changes {
            what: vec!["publish public posts and replies on X".to_string()],
            per_review: 1,
            per_day: 8,
            min_gap_secs: 900,
        };
        assert_eq!(
            changes.render(),
            "Publish public posts and replies on X – up to 1 change per review and 8 a day, at least 15 minutes apart"
        );
        assert_eq!(CanDo::ReadOnly.render(), "Read only – makes no changes");
    }

    #[test]
    fn ends_names_the_limit_and_stop_conditions_without_a_none_line() {
        let after = Ends {
            limit: EndLimit::AfterApproval { secs: 3_600 },
            has_stop_conditions: false,
        };
        assert_eq!(after.render(), "1 hour after you approve");
        let at = Ends {
            limit: EndLimit::At("Fri, Nov 13, 2:39 PM ET".to_string()),
            has_stop_conditions: true,
        };
        assert_eq!(
            at.render(),
            "Fri, Nov 13, 2:39 PM ET, or sooner if a stop condition is met"
        );
        let open = Ends {
            limit: EndLimit::None,
            has_stop_conditions: true,
        };
        assert_eq!(
            open.render(),
            "When a stop condition is met, or when you stop it"
        );
    }
}
