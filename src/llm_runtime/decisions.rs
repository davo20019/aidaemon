//! The registry of decisions: small closed questions the daemon may ask a
//! decision backend (`aidaemon_decisions::Decider`) on its own behalf. Each
//! decision has one spec saying what acting on it may do, how long it may
//! take, and which answers are trusted enough to act on.
//!
//! A decision never grants authority, decides completion or loosens a
//! safety check. With no backend configured, its mode `off`, in `shadow`
//! mode, or on any failure, the caller does what it did without one.
//! Every decision runs through `SideCallGateway::decide`.

use std::time::Duration;

use aidaemon_decisions::{Answer, ConfidenceSource, DecisionResponse};

use crate::config::{DecisionMode, DecisionModes};

/// A decision the daemon asks a decision backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Decision {
    /// One call per owner message: how it relates to earlier requests,
    /// whether it is about identity, and which model tier it needs.
    TurnDigest,
    /// Whether a message asks to resume an interrupted task. Only asked
    /// when an interrupted task exists.
    Resume,
    /// Which recalled facts are irrelevant to the message.
    RecallFilter,
    /// Which model tier a new request runs on. Reads the turn digest's
    /// `tier` answer; it makes no call of its own.
    ModelRoute,
    /// A second look at a terminal command's effects. May only add an
    /// approval, never remove one.
    CommandRiskSecond,
}

/// What acting on a decision may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionEffect {
    /// May skip or remove optional work or context. Default: keep it.
    SkipOptional,
    /// May pick one of several paths that hold the same authority. A wrong
    /// pick can still cost quality (a misbound request, a weaker model).
    Route,
    /// May only make a check stricter.
    EscalateOnly,
}

/// The minimum confidence and the confidence sources an answer needs
/// before it is acted on. `key` is a question key, or `"*"` for every
/// question of the decision.
#[derive(Debug, Clone, Copy)]
pub struct ActsOn {
    pub key: &'static str,
    pub min_confidence: f32,
    pub sources: &'static [ConfidenceSource],
}

/// How one decision runs.
#[derive(Debug, Clone, Copy)]
pub struct DecisionSpec {
    /// Telemetry key, unique; recorded as `decision:<purpose>`.
    pub purpose: &'static str,
    pub effect: DecisionEffect,
    /// Total time the backend may take.
    pub deadline: Duration,
    /// Answers that may be acted on. A question with no entry is recorded,
    /// never acted on.
    pub acts_on: &'static [ActsOn],
}

const CALIBRATED: &[ConfidenceSource] = &[ConfidenceSource::Calibrated];
const ANY_SOURCE: &[ConfidenceSource] = &[
    ConfidenceSource::Calibrated,
    ConfidenceSource::TokenLogprob,
    ConfidenceSource::SelfReported,
];

/// Turn-digest question keys.
pub mod digest_keys {
    pub const RELATIONSHIP: &str = "relationship";
    pub const IDENTITY: &str = "identity";
    pub const TIER: &str = "tier";
}

impl Decision {
    #[cfg(test)]
    pub const ALL: [Decision; 5] = [
        Decision::TurnDigest,
        Decision::Resume,
        Decision::RecallFilter,
        Decision::ModelRoute,
        Decision::CommandRiskSecond,
    ];

    pub const fn spec(self) -> DecisionSpec {
        use DecisionEffect::*;
        match self {
            // Relationship acts on a calibrated answer only: a chat model
            // reports high confidence on wrong answers, and today's router
            // already is a chat call. Identity pins a message when it says
            // yes, on any source: an extra pin is cheap, and the keyword
            // check still pins on its own. Tier is read by model routing.
            Decision::TurnDigest => DecisionSpec {
                purpose: "turn_digest",
                effect: Route,
                deadline: Duration::from_millis(1500),
                acts_on: &[
                    ActsOn {
                        key: digest_keys::RELATIONSHIP,
                        min_confidence: 0.6,
                        sources: CALIBRATED,
                    },
                    ActsOn {
                        key: digest_keys::IDENTITY,
                        min_confidence: 0.5,
                        sources: ANY_SOURCE,
                    },
                    ActsOn {
                        key: digest_keys::TIER,
                        min_confidence: 0.6,
                        sources: ANY_SOURCE,
                    },
                ],
            },
            Decision::Resume => DecisionSpec {
                purpose: "resume",
                effect: Route,
                deadline: Duration::from_millis(1500),
                acts_on: &[ActsOn {
                    key: "resume",
                    min_confidence: 0.8,
                    sources: CALIBRATED,
                }],
            },
            // A fact is dropped only when a calibrated answer says it is
            // irrelevant with P(relevant) below 0.15.
            Decision::RecallFilter => DecisionSpec {
                purpose: "recall_filter",
                effect: SkipOptional,
                deadline: Duration::from_millis(1500),
                acts_on: &[ActsOn {
                    key: "*",
                    min_confidence: 0.85,
                    sources: CALIBRATED,
                }],
            },
            // Routing thresholds live in `model_route_tier`; the spec only
            // names the decision for modes and telemetry.
            Decision::ModelRoute => DecisionSpec {
                purpose: "model_route",
                effect: Route,
                deadline: Duration::from_millis(1500),
                acts_on: &[],
            },
            Decision::CommandRiskSecond => DecisionSpec {
                purpose: "command_risk_second",
                effect: EscalateOnly,
                deadline: Duration::from_secs(5),
                acts_on: &[ActsOn {
                    key: "*",
                    min_confidence: 0.9,
                    sources: CALIBRATED,
                }],
            },
        }
    }

    /// This decision's configured mode.
    pub fn mode_in(self, modes: &DecisionModes) -> DecisionMode {
        match self {
            Decision::TurnDigest => modes.turn_digest,
            Decision::Resume => modes.resume,
            Decision::RecallFilter => modes.recall_filter,
            Decision::ModelRoute => modes.model_route,
            Decision::CommandRiskSecond => modes.command_risk_second,
        }
    }
}

/// Why a decision gave no answer to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultReason {
    /// No backend, or the decision's mode is `off`.
    Disabled,
    /// The backend failed or refused.
    Failed(String),
    /// The backend did not answer within the deadline.
    Timeout,
}

/// A backend's answers with the spec that governs acting on them.
#[derive(Debug, Clone)]
pub struct GatedResponse {
    pub response: DecisionResponse,
    acts_on: &'static [ActsOn],
    acting: bool,
}

impl GatedResponse {
    pub(crate) fn new(response: DecisionResponse, spec: &DecisionSpec, acting: bool) -> Self {
        Self {
            response,
            acts_on: spec.acts_on,
            acting,
        }
    }

    /// The answer to `key`, whatever its confidence. For telemetry and
    /// shadow comparisons only.
    #[cfg(test)]
    pub fn observed(&self, key: &str) -> Option<&Answer> {
        self.response.answers.get(key)
    }

    /// The answer to `key` when it may be acted on: the decision is on,
    /// and the answer meets its threshold from a trusted source.
    pub fn acted(&self, key: &str) -> Option<&Answer> {
        if !self.acting {
            return None;
        }
        self.trusted(key)
    }

    /// The answer to `key` when it meets its threshold from a trusted
    /// source, whether or not the decision may act (shadow telemetry).
    pub fn trusted(&self, key: &str) -> Option<&Answer> {
        let answer = self.response.answers.get(key)?;
        let rule = self
            .acts_on
            .iter()
            .find(|rule| rule.key == key)
            .or_else(|| self.acts_on.iter().find(|rule| rule.key == "*"))?;
        (answer.confidence >= rule.min_confidence
            && rule.sources.contains(&answer.confidence_source))
        .then_some(answer)
    }
}

/// What the gate returns.
#[derive(Debug, Clone)]
pub enum DecisionOutcome {
    /// The decision is on; act on the answers `acted` returns.
    Decided(GatedResponse),
    /// The decision is in shadow mode: answers recorded, never acted on.
    Shadow(GatedResponse),
    /// No answers; do what the daemon does without a decision layer.
    UseDefault(DefaultReason),
}

impl DecisionOutcome {
    /// The answer to `key` when it may be acted on.
    pub fn acted(&self, key: &str) -> Option<&Answer> {
        match self {
            Self::Decided(gated) => gated.acted(key),
            _ => None,
        }
    }

    /// The answers, acted on or not.
    pub fn gated(&self) -> Option<&GatedResponse> {
        match self {
            Self::Decided(gated) | Self::Shadow(gated) => Some(gated),
            Self::UseDefault(_) => None,
        }
    }
}

/// The tier a model-routing answer allows, given the digest's `tier`
/// answer. Routing down to the fast model needs a calibrated answer at 0.8
/// or more; routing up to the smart model needs 0.6. Anything else stays on
/// the primary model.
pub fn model_route_tier(answer: &Answer) -> Option<crate::router::Tier> {
    match answer.chosen()? {
        "fast"
            if answer.confidence >= 0.8
                && answer.confidence_source == ConfidenceSource::Calibrated =>
        {
            Some(crate::router::Tier::Fast)
        }
        "smart" if answer.confidence >= 0.6 => Some(crate::router::Tier::Smart),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_decisions::AnswerValue;
    use std::collections::BTreeMap;

    fn response(key: &str, answer: Answer) -> DecisionResponse {
        let mut answers = BTreeMap::new();
        answers.insert(key.to_string(), answer);
        DecisionResponse::new("test", answers, Default::default())
    }

    #[test]
    fn every_decision_has_a_unique_purpose() {
        let purposes: std::collections::HashSet<_> =
            Decision::ALL.iter().map(|d| d.spec().purpose).collect();
        assert_eq!(purposes.len(), Decision::ALL.len());
    }

    #[test]
    fn decisions_that_drop_or_escalate_trust_only_calibrated_answers() {
        for decision in [
            Decision::RecallFilter,
            Decision::CommandRiskSecond,
            Decision::Resume,
        ] {
            for rule in decision.spec().acts_on {
                assert_eq!(rule.sources, CALIBRATED, "{decision:?}");
            }
        }
    }

    #[test]
    fn a_self_reported_relationship_is_never_acted_on() {
        let spec = Decision::TurnDigest.spec();
        let gated = GatedResponse::new(
            response(
                digest_keys::RELATIONSHIP,
                Answer::choice("new_request", 0.99, ConfidenceSource::SelfReported),
            ),
            &spec,
            true,
        );
        assert!(gated.acted(digest_keys::RELATIONSHIP).is_none());
        assert!(gated.observed(digest_keys::RELATIONSHIP).is_some());
    }

    #[test]
    fn shadow_answers_are_never_acted_on() {
        let spec = Decision::TurnDigest.spec();
        let gated = GatedResponse::new(
            response(
                digest_keys::RELATIONSHIP,
                Answer::choice("new_request", 0.99, ConfidenceSource::Calibrated),
            ),
            &spec,
            false,
        );
        assert!(gated.acted(digest_keys::RELATIONSHIP).is_none());
        assert!(gated.trusted(digest_keys::RELATIONSHIP).is_some());
    }

    #[test]
    fn a_wildcard_rule_covers_every_question() {
        let spec = Decision::RecallFilter.spec();
        let gated = GatedResponse::new(
            response("f3", Answer::yes_no(0.05, ConfidenceSource::Calibrated)),
            &spec,
            true,
        );
        assert!(gated.acted("f3").is_some());
        let unsure = GatedResponse::new(
            response("f3", Answer::yes_no(0.3, ConfidenceSource::Calibrated)),
            &spec,
            true,
        );
        assert!(unsure.acted("f3").is_none());
    }

    #[test]
    fn routing_down_needs_a_confident_calibrated_answer() {
        let fast = |confidence, source| Answer::choice("fast", confidence, source);
        assert_eq!(
            model_route_tier(&fast(0.9, ConfidenceSource::Calibrated)),
            Some(crate::router::Tier::Fast)
        );
        assert_eq!(
            model_route_tier(&fast(0.7, ConfidenceSource::Calibrated)),
            None
        );
        assert_eq!(
            model_route_tier(&fast(0.99, ConfidenceSource::SelfReported)),
            None
        );
        assert_eq!(
            model_route_tier(&Answer::choice(
                "smart",
                0.65,
                ConfidenceSource::SelfReported
            )),
            Some(crate::router::Tier::Smart)
        );
        assert_eq!(
            model_route_tier(&Answer::new(
                AnswerValue::YesNo(0.9),
                0.9,
                ConfidenceSource::Calibrated
            )),
            None
        );
    }
}
