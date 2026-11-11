//! Decision backends: answer closed questions (yes or no, one of N, a score)
//! about one shared input, each answer with a confidence and a statement of
//! where that confidence came from.
//!
//! A decision backend is not a chat [`Provider`](aidaemon_core::Provider):
//! it writes no text. The caller asks typed questions and gets typed answers
//! back, so nothing is parsed out of prose.
//!
//! Backends:
//! - [`ProviderDecider`] asks any `aidaemon_core::Provider` (hosted or local)
//!   through a JSON schema. Its confidence is [`ConfidenceSource::SelfReported`].
//! - `GeminiDecider` (feature `gemini`) calls Google's Decisions API. Its
//!   confidence is [`ConfidenceSource::Calibrated`].
//! - `ScriptedDecider` (feature `testing`) returns fixed answers.
//!
//! A caller decides which confidence sources it trusts: a model that states
//! its own confidence is often confidently wrong, so a rule like "act only
//! when sure" should not run on a self-reported answer.

use std::collections::BTreeMap;

use async_trait::async_trait;

mod provider;
pub use provider::ProviderDecider;

#[cfg(feature = "gemini")]
mod gemini;
#[cfg(feature = "gemini")]
pub use gemini::GeminiDecider;

#[cfg(feature = "testing")]
pub mod testing;

/// A question or option key: `[A-Za-z0-9_-]`, 1 to 64 characters for
/// questions. Option keys may also hold `:` and `.` and run to 128.
pub type QuestionKey = String;
pub type OptionKey = String;

/// Most options a choice may have, and most levels a score may have.
pub const MAX_OPTIONS: usize = 255;

/// The input every question is asked about.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum InputBlock {
    Text(String),
}

impl InputBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }
}

/// What kind of answer a question wants.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QuestionKind {
    /// A yes-or-no question, optionally with what each side means.
    YesNo {
        true_description: Option<String>,
        false_description: Option<String>,
    },
    /// Pick one option. Option key -> description, 2 to 255 options.
    Choice {
        options: BTreeMap<OptionKey, String>,
    },
    /// An ordered scale, lowest level first, 2 to 255 levels.
    Score { levels: Vec<String> },
}

/// One question about the request's input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Question {
    pub instructions: String,
    pub kind: QuestionKind,
}

impl Question {
    pub fn yes_no(instructions: impl Into<String>) -> Self {
        Self {
            instructions: instructions.into(),
            kind: QuestionKind::YesNo {
                true_description: None,
                false_description: None,
            },
        }
    }

    /// A yes-or-no question with what "yes" and "no" each mean.
    pub fn yes_no_described(
        instructions: impl Into<String>,
        yes: impl Into<String>,
        no: impl Into<String>,
    ) -> Self {
        Self {
            instructions: instructions.into(),
            kind: QuestionKind::YesNo {
                true_description: Some(yes.into()),
                false_description: Some(no.into()),
            },
        }
    }

    pub fn choice<K: Into<String>, D: Into<String>>(
        instructions: impl Into<String>,
        options: impl IntoIterator<Item = (K, D)>,
    ) -> Self {
        Self {
            instructions: instructions.into(),
            kind: QuestionKind::Choice {
                options: options
                    .into_iter()
                    .map(|(key, description)| (key.into(), description.into()))
                    .collect(),
            },
        }
    }

    pub fn score<L: Into<String>>(
        instructions: impl Into<String>,
        levels: impl IntoIterator<Item = L>,
    ) -> Self {
        Self {
            instructions: instructions.into(),
            kind: QuestionKind::Score {
                levels: levels.into_iter().map(Into::into).collect(),
            },
        }
    }
}

/// Questions about one shared input.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct DecisionRequest {
    pub input: Vec<InputBlock>,
    pub questions: BTreeMap<QuestionKey, Question>,
}

impl DecisionRequest {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: vec![InputBlock::text(input)],
            questions: BTreeMap::new(),
        }
    }

    pub fn ask(mut self, key: impl Into<String>, question: Question) -> Self {
        self.questions.insert(key.into(), question);
        self
    }

    pub fn insert(&mut self, key: impl Into<String>, question: Question) {
        self.questions.insert(key.into(), question);
    }

    /// The input as one string, blocks separated by blank lines.
    pub fn input_text(&self) -> String {
        self.input
            .iter()
            .map(|block| match block {
                InputBlock::Text(text) => text.as_str(),
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Checks the limits every backend shares: at least one question, valid
    /// keys, 2..=255 options or levels, distinct option keys.
    pub fn validate(&self) -> Result<(), DecisionError> {
        if self.questions.is_empty() {
            return Err(DecisionError::InvalidRequest("no questions".into()));
        }
        for (key, question) in &self.questions {
            if !valid_key(key, 64, false) {
                return Err(DecisionError::InvalidRequest(format!(
                    "question key {key:?} must match [A-Za-z0-9_-]{{1,64}}"
                )));
            }
            match &question.kind {
                QuestionKind::YesNo { .. } => {}
                QuestionKind::Choice { options } => {
                    if !(2..=MAX_OPTIONS).contains(&options.len()) {
                        return Err(DecisionError::InvalidRequest(format!(
                            "choice {key:?} needs 2 to {MAX_OPTIONS} options, has {}",
                            options.len()
                        )));
                    }
                    if let Some(bad) = options.keys().find(|option| !valid_key(option, 128, true)) {
                        return Err(DecisionError::InvalidRequest(format!(
                            "option key {bad:?} in {key:?} is not valid"
                        )));
                    }
                }
                QuestionKind::Score { levels } => {
                    if !(2..=MAX_OPTIONS).contains(&levels.len()) {
                        return Err(DecisionError::InvalidRequest(format!(
                            "score {key:?} needs 2 to {MAX_OPTIONS} levels, has {}",
                            levels.len()
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn valid_key(key: &str, max: usize, option: bool) -> bool {
    !key.is_empty()
        && key.len() <= max
        && key.chars().all(|c| {
            c.is_ascii_alphanumeric() || c == '_' || c == '-' || (option && (c == ':' || c == '.'))
        })
}

/// Where an answer's confidence came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConfidenceSource {
    /// Calibrated probabilities from a decision model.
    Calibrated,
    /// Derived from token log probabilities of a constrained answer.
    TokenLogprob,
    /// The model stated its own confidence. Measured as unreliable: chat
    /// models report high confidence on wrong answers.
    SelfReported,
}

impl ConfidenceSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Calibrated => "calibrated",
            Self::TokenLogprob => "token_logprob",
            Self::SelfReported => "self_reported",
        }
    }
}

/// An answer's value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AnswerValue {
    /// Probability the answer is yes, in 0..=1.
    YesNo(f32),
    /// The chosen option key.
    Choice(OptionKey),
    /// Expected level index, from 0 for the lowest level.
    Score(f32),
}

/// One question's answer.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Answer {
    pub value: AnswerValue,
    /// Per option (choice) or per level index (score), when the backend
    /// gives them. Sums to 1.
    pub probabilities: Option<BTreeMap<String, f32>>,
    /// Probability that the top answer is right, in 0..=1.
    pub confidence: f32,
    pub confidence_source: ConfidenceSource,
}

impl Answer {
    pub fn new(value: AnswerValue, confidence: f32, confidence_source: ConfidenceSource) -> Self {
        Self {
            value,
            probabilities: None,
            confidence: confidence.clamp(0.0, 1.0),
            confidence_source,
        }
    }

    /// A yes/no answer from `p_yes`; its confidence is `max(p, 1 - p)`.
    pub fn yes_no(p_yes: f32, confidence_source: ConfidenceSource) -> Self {
        let p = p_yes.clamp(0.0, 1.0);
        Self::new(AnswerValue::YesNo(p), p.max(1.0 - p), confidence_source)
    }

    pub fn choice(
        option: impl Into<String>,
        confidence: f32,
        confidence_source: ConfidenceSource,
    ) -> Self {
        Self::new(
            AnswerValue::Choice(option.into()),
            confidence,
            confidence_source,
        )
    }

    pub fn with_probabilities(mut self, probabilities: BTreeMap<String, f32>) -> Self {
        self.probabilities = Some(probabilities);
        self
    }

    /// `p_yes` for a yes/no answer.
    pub fn p_yes(&self) -> Option<f32> {
        match self.value {
            AnswerValue::YesNo(p) => Some(p),
            _ => None,
        }
    }

    /// The chosen option for a choice answer.
    pub fn chosen(&self) -> Option<&str> {
        match &self.value {
            AnswerValue::Choice(option) => Some(option),
            _ => None,
        }
    }

    /// The expected level for a score answer.
    pub fn score(&self) -> Option<f32> {
        match self.value {
            AnswerValue::Score(score) => Some(score),
            _ => None,
        }
    }
}

/// Every question's answer.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct DecisionResponse {
    pub answers: BTreeMap<QuestionKey, Answer>,
    pub usage: aidaemon_core::Usage,
    /// The backend that answered (for telemetry).
    pub backend: String,
}

impl DecisionResponse {
    pub fn new(
        backend: impl Into<String>,
        answers: BTreeMap<QuestionKey, Answer>,
        usage: aidaemon_core::Usage,
    ) -> Self {
        Self {
            answers,
            usage,
            backend: backend.into(),
        }
    }

    /// Checks that every question has an answer of its kind, and nothing
    /// else. Backends call this before returning.
    pub fn check_against(&self, request: &DecisionRequest) -> Result<(), DecisionError> {
        if self.answers.len() != request.questions.len() {
            return Err(DecisionError::Malformed(format!(
                "{} answers for {} questions",
                self.answers.len(),
                request.questions.len()
            )));
        }
        for (key, question) in &request.questions {
            let Some(answer) = self.answers.get(key) else {
                return Err(DecisionError::Malformed(format!("no answer for {key:?}")));
            };
            let fits = match (&question.kind, &answer.value) {
                (QuestionKind::YesNo { .. }, AnswerValue::YesNo(p)) => (0.0..=1.0).contains(p),
                (QuestionKind::Choice { options }, AnswerValue::Choice(option)) => {
                    options.contains_key(option)
                }
                (QuestionKind::Score { levels }, AnswerValue::Score(score)) => {
                    *score >= 0.0 && *score <= (levels.len() - 1) as f32
                }
                _ => false,
            };
            if !fits || !(0.0..=1.0).contains(&answer.confidence) {
                return Err(DecisionError::Malformed(format!(
                    "answer to {key:?} does not fit its question"
                )));
            }
        }
        Ok(())
    }
}

/// What a backend can do.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DeciderCapabilities {
    /// Most questions in one request.
    pub max_questions: usize,
    /// Where this backend's confidence comes from.
    pub confidence_source: ConfidenceSource,
}

impl DeciderCapabilities {
    pub fn new(max_questions: usize, confidence_source: ConfidenceSource) -> Self {
        Self {
            max_questions,
            confidence_source,
        }
    }
}

/// Why a backend has no answer. Mirrors `aidaemon_core::ProviderError`.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum DecisionError {
    #[error("rate limited")]
    RateLimited,
    #[error("transient decision backend error: {0}")]
    Transient(String),
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    /// The backend refused the input on safety grounds.
    #[error("blocked by the backend's safety filter: {0}")]
    SafetyBlock(String),
    /// The backend answered, but not every question, or not in kind.
    #[error("malformed answer: {0}")]
    Malformed(String),
    #[error("{0}")]
    Other(String),
}

impl From<aidaemon_core::ProviderError> for DecisionError {
    fn from(error: aidaemon_core::ProviderError) -> Self {
        use aidaemon_core::ProviderError as P;
        match error {
            P::RateLimited { .. } => Self::RateLimited,
            P::Transient(message) => Self::Transient(message),
            P::InvalidRequest(message) => Self::InvalidRequest(message),
            P::Auth(message) => Self::Auth(message),
            other => Self::Other(other.to_string()),
        }
    }
}

/// A backend that answers closed questions.
///
/// A backend answers every question or returns an error, never a partial
/// set. To add one, copy [`ProviderDecider`] and run
/// `testing::conformance::check_decider` against it.
#[async_trait]
pub trait Decider: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> DeciderCapabilities;
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, DecisionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_choice_with_one_option_is_refused() {
        let request = DecisionRequest::new("hi").ask("q", Question::choice("pick", [("a", "A")]));
        assert!(matches!(
            request.validate(),
            Err(DecisionError::InvalidRequest(_))
        ));
    }

    #[test]
    fn keys_are_checked() {
        let request = DecisionRequest::new("hi").ask("bad key", Question::yes_no("?"));
        assert!(request.validate().is_err());
        let request = DecisionRequest::new("hi").ask(
            "ok",
            Question::choice("pick", [("continuation:msg-1", "A"), ("none", "B")]),
        );
        assert!(request.validate().is_ok());
    }

    #[test]
    fn yes_no_confidence_is_the_larger_side() {
        let answer = Answer::yes_no(0.2, ConfidenceSource::Calibrated);
        assert!((answer.confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn a_response_missing_an_answer_is_malformed() {
        let request = DecisionRequest::new("hi")
            .ask("a", Question::yes_no("?"))
            .ask("b", Question::yes_no("?"));
        let mut answers = BTreeMap::new();
        answers.insert(
            "a".to_string(),
            Answer::yes_no(0.9, ConfidenceSource::Calibrated),
        );
        let response = DecisionResponse::new("x", answers, Default::default());
        assert!(response.check_against(&request).is_err());
    }
}
