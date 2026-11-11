//! Test helpers: [`ScriptedDecider`], [`answer_all`] and the
//! [`conformance`] kit every backend should pass.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use async_trait::async_trait;

use crate::{
    Answer, AnswerValue, ConfidenceSource, Decider, DeciderCapabilities, DecisionError,
    DecisionRequest, DecisionResponse, QuestionKind,
};

type Fallback =
    Box<dyn Fn(&DecisionRequest) -> Result<DecisionResponse, DecisionError> + Send + Sync>;

/// Returns scripted answers in order, then falls back to a closure, and
/// records every request it saw. Requests are validated and scripted
/// answers checked against them, as a real backend would.
pub struct ScriptedDecider {
    source: ConfidenceSource,
    script: Mutex<VecDeque<Result<BTreeMap<String, Answer>, DecisionError>>>,
    fallback: Option<Fallback>,
    calls: Mutex<Vec<DecisionRequest>>,
}

impl ScriptedDecider {
    pub fn new(source: ConfidenceSource) -> Self {
        Self {
            source,
            script: Mutex::new(VecDeque::new()),
            fallback: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Queues one reply.
    pub fn then(self, reply: Result<BTreeMap<String, Answer>, DecisionError>) -> Self {
        self.script.lock().expect("poisoned").push_back(reply);
        self
    }

    /// Answers with `f` once the queue is empty.
    pub fn repeat_with(
        mut self,
        f: impl Fn(&DecisionRequest) -> Result<DecisionResponse, DecisionError> + Send + Sync + 'static,
    ) -> Self {
        self.fallback = Some(Box::new(f));
        self
    }

    /// Answers every request with [`answer_all`] at `p` once the queue is
    /// empty.
    pub fn always(self, p: f32) -> Self {
        let source = self.source;
        self.repeat_with(move |request| {
            Ok(DecisionResponse::new(
                "scripted",
                answer_all(request, p, source),
                Default::default(),
            ))
        })
    }

    /// Every request this decider was asked, in order.
    pub fn calls(&self) -> Vec<DecisionRequest> {
        self.calls.lock().expect("poisoned").clone()
    }

    /// Scripted replies not yet used.
    pub fn remaining(&self) -> usize {
        self.script.lock().expect("poisoned").len()
    }
}

#[async_trait]
impl Decider for ScriptedDecider {
    fn name(&self) -> &str {
        "scripted"
    }

    fn capabilities(&self) -> DeciderCapabilities {
        DeciderCapabilities::new(1000, self.source)
    }

    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, DecisionError> {
        self.calls.lock().expect("poisoned").push(request.clone());
        request.validate()?;
        let next = self.script.lock().expect("poisoned").pop_front();
        let response = match (next, &self.fallback) {
            (Some(reply), _) => DecisionResponse::new("scripted", reply?, Default::default()),
            (None, Some(fallback)) => fallback(request)?,
            (None, None) => {
                return Err(DecisionError::Other(
                    "scripted decider: script exhausted".into(),
                ))
            }
        };
        response.check_against(request)?;
        Ok(response)
    }
}

/// An answer to every question: yes/no at `p_yes`, a choice's first
/// option, a score's lowest level; non-yes/no confidence is
/// `max(p, 1 - p)`.
pub fn answer_all(
    request: &DecisionRequest,
    p_yes: f32,
    source: ConfidenceSource,
) -> BTreeMap<String, Answer> {
    let confidence = p_yes.clamp(0.0, 1.0).max(1.0 - p_yes.clamp(0.0, 1.0));
    request
        .questions
        .iter()
        .map(|(key, question)| {
            let answer = match &question.kind {
                QuestionKind::YesNo { .. } => Answer::yes_no(p_yes, source),
                QuestionKind::Choice { options } => Answer::choice(
                    options.keys().next().cloned().unwrap_or_default(),
                    confidence,
                    source,
                ),
                QuestionKind::Score { .. } => {
                    Answer::new(AnswerValue::Score(0.0), confidence, source)
                }
            };
            (key.clone(), answer)
        })
        .collect()
}

/// Checks any [`Decider`] against the contract every backend shares.
pub mod conformance {
    use super::*;
    use crate::Question;

    /// The question and option keys of [`request`], so a fake backend can
    /// answer it.
    pub struct ConformanceKeys {
        pub yes_no: &'static str,
        pub choice: &'static str,
        /// One valid option of the choice.
        pub choice_option: &'static str,
        pub choice_options: [&'static str; 3],
        pub score: &'static str,
        pub score_levels: usize,
    }

    pub const KEYS: ConformanceKeys = ConformanceKeys {
        yes_no: "asks_question",
        choice: "topic",
        choice_option: "scheduling",
        choice_options: ["billing", "other", "scheduling"],
        score: "urgency",
        score_levels: 3,
    };

    /// The synthetic request the kit asks: one yes/no, one 3-option
    /// choice, one 3-level score.
    pub fn request() -> DecisionRequest {
        DecisionRequest::new(
            "Hi Jordan, can we move Friday's review with Acme Corp to 3pm? -- Alice",
        )
        .ask(
            KEYS.yes_no,
            Question::yes_no_described(
                "Does the message ask the reader something?",
                "It contains a question or request",
                "It only informs",
            ),
        )
        .ask(
            KEYS.choice,
            Question::choice(
                "What is the message mainly about?",
                [
                    ("billing", "Invoices, payments or prices"),
                    ("other", "Anything else"),
                    ("scheduling", "Meetings, times or dates"),
                ],
            ),
        )
        .ask(
            KEYS.score,
            Question::score(
                "How urgent is the message?",
                ["not urgent", "somewhat urgent", "very urgent"],
            ),
        )
    }

    fn check_probabilities(key: &str, probabilities: &BTreeMap<String, f32>) {
        for (option, p) in probabilities {
            assert!(
                (0.0..=1.0).contains(p),
                "probability of {option:?} in {key:?} is {p}, outside 0..=1"
            );
        }
        let sum: f32 = probabilities.values().sum();
        assert!(
            (sum - 1.0).abs() <= 0.02,
            "probabilities of {key:?} sum to {sum}, not 1"
        );
    }

    /// Panics, with a message saying what broke, unless `decider` answers
    /// the kit's request in full and refuses an invalid one.
    pub async fn check_decider(decider: &dyn Decider) {
        let name = decider.name();
        assert!(!name.is_empty(), "decider has an empty name");
        let capabilities = decider.capabilities();
        assert!(
            capabilities.max_questions >= 3,
            "{name}: max_questions {} is below the kit's 3 questions",
            capabilities.max_questions
        );

        let request = request();
        let response = match decider.decide(&request).await {
            Ok(response) => response,
            Err(error) => panic!("{name}: the conformance request failed: {error}"),
        };
        if let Err(error) = response.check_against(&request) {
            panic!("{name}: response does not fit the request: {error}");
        }
        for (key, answer) in &response.answers {
            assert!(
                (0.0..=1.0).contains(&answer.confidence),
                "{name}: confidence of {key:?} is {}, outside 0..=1",
                answer.confidence
            );
            assert_eq!(
                answer.confidence_source, capabilities.confidence_source,
                "{name}: {key:?} reports a confidence source other than the decider's"
            );
            if let Some(probabilities) = &answer.probabilities {
                check_probabilities(key, probabilities);
            }
        }
        let choice = &response.answers[KEYS.choice];
        assert!(
            KEYS.choice_options
                .contains(&choice.chosen().unwrap_or_default()),
            "{name}: choice answer is not one of the options"
        );
        let score = response.answers[KEYS.score].score().unwrap_or(-1.0);
        assert!(
            (0.0..=(KEYS.score_levels - 1) as f32).contains(&score),
            "{name}: score {score} is outside the levels"
        );

        let invalid = DecisionRequest::new("x").ask("only", Question::choice("pick", [("a", "A")]));
        match decider.decide(&invalid).await {
            Err(DecisionError::InvalidRequest(_)) => {}
            other => panic!(
                "{name}: a choice with one option must be refused as InvalidRequest, got {other:?}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Question;

    #[tokio::test]
    async fn scripted_replies_then_fallback() {
        let request = DecisionRequest::new("hi").ask("q", Question::yes_no("?"));
        let decider = ScriptedDecider::new(ConfidenceSource::Calibrated)
            .then(Err(DecisionError::RateLimited))
            .then(Ok(answer_all(&request, 0.9, ConfidenceSource::Calibrated)))
            .always(0.1);
        assert_eq!(
            decider.decide(&request).await,
            Err(DecisionError::RateLimited)
        );
        let second = decider.decide(&request).await.unwrap();
        assert_eq!(second.answers["q"].p_yes(), Some(0.9));
        let third = decider.decide(&request).await.unwrap();
        assert_eq!(third.answers["q"].p_yes(), Some(0.1));
        assert_eq!(decider.calls().len(), 3);
    }

    #[tokio::test]
    async fn an_answer_that_does_not_fit_is_malformed() {
        let request = DecisionRequest::new("hi").ask("q", Question::yes_no("?"));
        let decider = ScriptedDecider::new(ConfidenceSource::Calibrated).then(Ok(BTreeMap::new()));
        assert!(matches!(
            decider.decide(&request).await,
            Err(DecisionError::Malformed(_))
        ));
    }

    #[tokio::test]
    async fn passes_the_conformance_kit() {
        let decider = ScriptedDecider::new(ConfidenceSource::TokenLogprob).always(0.7);
        conformance::check_decider(&decider).await;
    }

    #[tokio::test]
    #[should_panic(expected = "confidence source")]
    async fn the_kit_catches_a_wrong_confidence_source() {
        let decider = ScriptedDecider::new(ConfidenceSource::Calibrated).repeat_with(|request| {
            Ok(DecisionResponse::new(
                "scripted",
                answer_all(request, 0.7, ConfidenceSource::SelfReported),
                Default::default(),
            ))
        });
        conformance::check_decider(&decider).await;
    }
}
