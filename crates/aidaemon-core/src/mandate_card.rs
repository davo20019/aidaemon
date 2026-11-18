//! The owner's confirmation card for a mandate as it travels in
//! [`crate::OwnerApprovalRequest::warnings`]: five essentials the owner reads
//! at a glance (each a `"<label>: <text>"` line with a label from
//! [`Essential`]), then every other line as a detail. Here, below both the
//! work crate that builds it and the channels that render it.

/// One of the five things the owner needs to approve a mandate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Essential {
    What,
    HowOften,
    CanDo,
    TellsYouWhen,
    Ends,
}

impl Essential {
    /// In card order.
    pub const ALL: [Self; 5] = [
        Self::What,
        Self::HowOften,
        Self::CanDo,
        Self::TellsYouWhen,
        Self::Ends,
    ];

    /// The label the line starts with, as the owner reads it.
    pub const fn label(self) -> &'static str {
        match self {
            Self::What => "What",
            Self::HowOften => "How often",
            Self::CanDo => "Can do",
            Self::TellsYouWhen => "Tells you when",
            Self::Ends => "Ends",
        }
    }

    /// The essential a card line carries, and its text; `None` for a detail.
    pub fn parse_line(line: &str) -> Option<(Self, &str)> {
        Self::ALL.into_iter().find_map(|essential| {
            line.strip_prefix(essential.label())
                .and_then(|rest| rest.strip_prefix(':'))
                .map(|text| (essential, text.trim()))
        })
    }
}

/// A mandate confirmation as the owner sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MandateConfirmationCard {
    pub what: String,
    pub how_often: String,
    pub can_do: String,
    pub tells_you_when: String,
    pub ends: String,
    /// Everything else, one `"<label>: <text>"` line each, for a "Details"
    /// section.
    pub details: Vec<String>,
}

impl MandateConfirmationCard {
    /// The essentials in card order.
    pub fn essentials(&self) -> [(Essential, &str); 5] {
        [
            (Essential::What, self.what.as_str()),
            (Essential::HowOften, self.how_often.as_str()),
            (Essential::CanDo, self.can_do.as_str()),
            (Essential::TellsYouWhen, self.tells_you_when.as_str()),
            (Essential::Ends, self.ends.as_str()),
        ]
    }

    /// The card as approval-request lines: the essentials, then the details.
    pub fn into_lines(self) -> Vec<String> {
        let mut lines = self
            .essentials()
            .into_iter()
            .map(|(essential, text)| format!("{}: {text}", essential.label()))
            .collect::<Vec<_>>();
        lines.extend(self.details);
        lines
    }

    /// The card a list of approval-request lines carries; `None` unless every
    /// essential is there.
    pub fn from_lines(lines: &[String]) -> Option<Self> {
        let mut essentials: [Option<String>; 5] = Default::default();
        let mut details = Vec::new();
        for line in lines {
            match Essential::parse_line(line) {
                Some((essential, text)) => {
                    let slot = Essential::ALL
                        .iter()
                        .position(|candidate| *candidate == essential)
                        .expect("every essential is listed");
                    essentials[slot] = Some(text.to_string());
                }
                None => details.push(line.clone()),
            }
        }
        let [what, how_often, can_do, tells_you_when, ends] = essentials;
        Some(Self {
            what: what?,
            how_often: how_often?,
            can_do: can_do?,
            tells_you_when: tells_you_when?,
            ends: ends?,
            details,
        })
    }
}
