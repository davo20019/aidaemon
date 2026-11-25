//! [`TypesafeDecider`]: TypeSafe's System One API, answered by Jev, which
//! returns calibrated probabilities.
//!
//! Wire format, from TypeSafe's docs (read 2026-11-14):
//! - Endpoint, auth, request and response JSON, error bodies:
//!   <https://docs.typesafe.ai/api/system-one.md> (the OpenAPI spec at
//!   <https://api.typesafe.ai/openapi.json>): `POST /v1/systemone` on
//!   `https://api.typesafe.ai`, `Authorization: Bearer <key>`, body
//!   `{model, state, questions: {name: {type, instructions, criteria}}}`,
//!   reply `{model, answers: {name: {...}}, usage: {input_tokens,
//!   output_tokens}}`. Noul criteria are `{"true", "false"}`; choice
//!   criteria map option name to description; score criteria are an ordered
//!   list of at most 10 levels.
//! - Status codes: <https://docs.typesafe.ai/api/errors.md> (400, 401, 413,
//!   415, 422, 429, 529 overloaded).
//! - Model ids and aliases: <https://docs.typesafe.ai/models.md>.
//! - Confidence: <https://docs.typesafe.ai/confidence.md>. Jev's own
//!   `confidence` is how concentrated the distribution is (0 for an even
//!   split), not the probability that the answer is right. This crate's
//!   [`Answer::confidence`] is that probability, so it is read from
//!   `probabilities` instead: the chosen option's probability for a choice,
//!   the most likely level's for a score, `max(p, 1 - p)` for a noul (which
//!   carries no confidence of its own). Thresholds then mean the same as
//!   with the Gemini backend.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::wire::{f32_field, network_error, probabilities, short};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::{
    Answer, AnswerValue, ConfidenceSource, Decider, DeciderCapabilities, DecisionError,
    DecisionRequest, DecisionResponse, QuestionKind,
};

/// The docs' alias for the newest stable Jev. Pin a versioned id such as
/// `jev-1.14.0` with `with_model` to keep tuned thresholds stable.
const DEFAULT_MODEL: &str = "jev-latest";
const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
/// Score questions take at most this many levels (`ScoreQuestion.criteria`
/// `maxItems` in the OpenAPI spec).
const MAX_SCORE_LEVELS: usize = 10;

/// Calls TypeSafe's System One API. Its confidence is
/// [`ConfidenceSource::Calibrated`].
pub struct TypesafeDecider {
    api_key: String,
    model: String,
    base_url: String,
    name: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for TypesafeDecider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypesafeDecider")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl TypesafeDecider {
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.into(),
            base_url: DEFAULT_BASE_URL.into(),
            name: format!("typesafe:{DEFAULT_MODEL}"),
            client,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.name = format!("typesafe:{}", self.model);
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
}

#[async_trait]
impl Decider for TypesafeDecider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> DeciderCapabilities {
        // The API sets no limit on the number of questions, only on context
        // length; this bound keeps one request reasonable.
        DeciderCapabilities::new(1000, ConfidenceSource::Calibrated)
    }

    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse, DecisionError> {
        request.validate()?;
        let max = self.capabilities().max_questions;
        if request.questions.len() > max {
            return Err(DecisionError::InvalidRequest(format!(
                "{} questions; this backend takes at most {max}",
                request.questions.len()
            )));
        }
        for (key, question) in &request.questions {
            if let QuestionKind::Score { levels } = &question.kind
                && levels.len() > MAX_SCORE_LEVELS
            {
                return Err(DecisionError::InvalidRequest(format!(
                    "score {key:?} has {} levels; this backend takes at most {MAX_SCORE_LEVELS}",
                    levels.len()
                )));
            }
        }
        let reply = self
            .client
            .post(format!("{}/v1/systemone", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&request_body(&self.model, request))
            .send()
            .await
            .map_err(|error| DecisionError::Transient(network_error("TypeSafe API", &error)))?;
        let status = reply.status();
        let body = reply
            .text()
            .await
            .map_err(|error| DecisionError::Transient(network_error("TypeSafe API", &error)))?;
        if !status.is_success() {
            return Err(status_error(status.as_u16(), &body));
        }
        let body: Value = serde_json::from_str(&body)
            .map_err(|error| DecisionError::Malformed(format!("response is not JSON: {error}")))?;
        let answers = parse_answers(&body, request)?;
        let tokens = |field: &str| {
            body.get("usage")
                .and_then(|usage| usage.get(field))
                .and_then(Value::as_u64)
                .unwrap_or(0)
        };
        let usage = aidaemon_core::Usage {
            input_tokens: tokens("input_tokens"),
            output_tokens: tokens("output_tokens"),
            ..Default::default()
        };
        let response = DecisionResponse::new(self.name.clone(), answers, usage);
        response.check_against(request)?;
        Ok(response)
    }
}

fn request_body(model: &str, request: &DecisionRequest) -> Value {
    let mut questions = Map::new();
    for (key, question) in &request.questions {
        let mut entry = Map::new();
        entry.insert("instructions".into(), json!(question.instructions));
        match &question.kind {
            QuestionKind::YesNo {
                true_description,
                false_description,
            } => {
                entry.insert("type".into(), json!("noul"));
                if true_description.is_some() || false_description.is_some() {
                    let mut criteria = Map::new();
                    if let Some(yes) = true_description {
                        criteria.insert("true".into(), json!(yes));
                    }
                    if let Some(no) = false_description {
                        criteria.insert("false".into(), json!(no));
                    }
                    entry.insert("criteria".into(), Value::Object(criteria));
                }
            }
            QuestionKind::Choice { options } => {
                entry.insert("type".into(), json!("choice"));
                entry.insert("criteria".into(), json!(options));
            }
            QuestionKind::Score { levels } => {
                entry.insert("type".into(), json!("score"));
                entry.insert("criteria".into(), json!(levels));
            }
        }
        questions.insert(key.clone(), Value::Object(entry));
    }
    json!({
        "model": model,
        "state": request.input_text(),
        "questions": questions,
    })
}

/// Error bodies are `{"detail": ...}`: a string or object for 400, a list
/// of `{loc, msg, type}` for 422.
fn status_error(status: u16, body: &str) -> DecisionError {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let detail = match parsed.get("detail") {
        Some(Value::String(text)) => short(text),
        Some(Value::Array(items)) => short(
            &items
                .iter()
                .filter_map(|item| item.get("msg").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("; "),
        ),
        Some(other) => short(&other.to_string()),
        None => short(body),
    };
    match status {
        429 => DecisionError::RateLimited,
        401 | 403 => DecisionError::Auth(format!("HTTP {status}: {detail}")),
        // 529 is TypeSafe's "overloaded".
        500..=599 => DecisionError::Transient(format!("HTTP {status}: {detail}")),
        400 | 413 | 415 | 422 => DecisionError::InvalidRequest(format!("HTTP {status}: {detail}")),
        _ => DecisionError::Other(format!("HTTP {status}: {detail}")),
    }
}

fn parse_answers(
    body: &Value,
    request: &DecisionRequest,
) -> Result<BTreeMap<String, Answer>, DecisionError> {
    let Some(answers) = body.get("answers").and_then(Value::as_object) else {
        return Err(DecisionError::Malformed(
            "response has no \"answers\"".into(),
        ));
    };
    if let Some(unknown) = answers
        .keys()
        .find(|key| !request.questions.contains_key(*key))
    {
        return Err(DecisionError::Malformed(format!(
            "answer for unknown question {unknown:?}"
        )));
    }
    let source = ConfidenceSource::Calibrated;
    let mut out = BTreeMap::new();
    for (key, question) in &request.questions {
        let Some(value) = answers.get(key) else {
            return Err(DecisionError::Malformed(format!("no answer for {key:?}")));
        };
        let missing = |field: &str| DecisionError::Malformed(format!("{key:?} has no {field:?}"));
        let expected = match &question.kind {
            QuestionKind::YesNo { .. } => "noul",
            QuestionKind::Choice { .. } => "choice",
            QuestionKind::Score { .. } => "score",
        };
        if let Some(kind) = value.get("type").and_then(Value::as_str)
            && kind != expected
        {
            return Err(DecisionError::Malformed(format!(
                "{key:?} answered as {kind:?}, asked as {expected:?}"
            )));
        }
        let probs = probabilities(value);
        let mut answer = match &question.kind {
            QuestionKind::YesNo { .. } => {
                let p = f32_field(value, "noul").ok_or_else(|| missing("noul"))?;
                Answer::yes_no(p, source)
            }
            QuestionKind::Choice { options } => {
                let choice = value
                    .get("choice")
                    .and_then(Value::as_str)
                    .ok_or_else(|| missing("choice"))?;
                if !options.contains_key(choice) {
                    return Err(DecisionError::Malformed(format!(
                        "{key:?} chose unknown option {choice:?}"
                    )));
                }
                let confidence = probs
                    .as_ref()
                    .and_then(|p| p.get(choice).copied())
                    .or_else(|| {
                        // Without probabilities, invert Jev's confidence
                        // formula, (p_max - 1/n) / (1 - 1/n), back to p_max.
                        let n = options.len() as f32;
                        f32_field(value, "confidence").map(|c| c * (1.0 - 1.0 / n) + 1.0 / n)
                    })
                    .ok_or_else(|| missing("probabilities"))?;
                Answer::choice(choice, confidence, source)
            }
            QuestionKind::Score { levels } => {
                let top = (levels.len() - 1) as f32;
                let score = f32_field(value, "score")
                    .ok_or_else(|| missing("score"))?
                    .clamp(0.0, top);
                let confidence = probs
                    .as_ref()
                    .and_then(|p| p.values().copied().reduce(f32::max))
                    .ok_or_else(|| missing("probabilities"))?;
                Answer::new(AnswerValue::Score(score), confidence, source)
            }
        };
        if let Some(probs) = probs.filter(|p| !p.is_empty()) {
            answer = answer.with_probabilities(probs);
        }
        out.insert(key.clone(), answer);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Question;
    use crate::test_http::{fixed, serve};
    use crate::testing::conformance;
    use std::sync::Arc;
    use tokio::net::TcpListener;

    const KEY: &str = "synthetic-typesafe-key-456";

    /// Answers whatever it is asked, per question type, in the documented
    /// response shape. Choices put `top` on the first option.
    fn answering_with(top: f64) -> impl Fn(&Value) -> (u16, String) + Send + Sync {
        move |body: &Value| {
            let mut answers = Map::new();
            for (key, question) in body["questions"].as_object().unwrap() {
                let answer = match question["type"].as_str().unwrap() {
                    "noul" => json!({"type": "noul", "noul": top}),
                    "choice" => {
                        let options: Vec<&String> =
                            question["criteria"].as_object().unwrap().keys().collect();
                        let n = options.len() as f64;
                        let rest = (1.0 - top) / (n - 1.0);
                        let probabilities: Map<String, Value> = options
                            .iter()
                            .enumerate()
                            .map(|(i, o)| ((*o).clone(), json!(if i == 0 { top } else { rest })))
                            .collect();
                        json!({
                            "type": "choice",
                            "choice": options[0],
                            "confidence": (top - 1.0 / n) / (1.0 - 1.0 / n),
                            "probabilities": probabilities,
                        })
                    }
                    "score" => {
                        let levels = question["criteria"].as_array().unwrap().len();
                        let rest = (1.0 - top) / (levels - 1) as f64;
                        let probabilities: Map<String, Value> = (0..levels)
                            .map(|i| {
                                (
                                    i.to_string(),
                                    json!(if i == levels - 1 { top } else { rest }),
                                )
                            })
                            .collect();
                        let legend: Map<String, Value> = question["criteria"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .enumerate()
                            .map(|(i, level)| (i.to_string(), level.clone()))
                            .collect();
                        json!({
                            "type": "score",
                            "score": (levels - 1) as f64 * top,
                            "confidence": 0.5,
                            "legend": legend,
                            "probabilities": probabilities,
                        })
                    }
                    other => panic!("unexpected type {other}"),
                };
                answers.insert(key.clone(), answer);
            }
            (
                200,
                json!({
                    "model": "jev-1.14.0",
                    "answers": answers,
                    "usage": {"input_tokens": 120, "output_tokens": 12},
                })
                .to_string(),
            )
        }
    }

    fn request() -> DecisionRequest {
        DecisionRequest::new("Please pay invoice 42 by Monday. -- Jordan Lee, Acme Corp")
            .ask("plain", Question::yes_no("Is it a request?"))
            .ask(
                "described",
                Question::yes_no_described(
                    "Is money involved?",
                    "It mentions money",
                    "It does not",
                ),
            )
            .ask(
                "topic",
                Question::choice("Topic?", [("billing", "Money"), ("other", "Else")]),
            )
            .ask(
                "urgency",
                Question::score("Urgency?", ["low", "mid", "high"]),
            )
    }

    #[tokio::test]
    async fn sends_the_api_shape() {
        let (base, seen) = serve(Arc::new(answering_with(0.9))).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        decider.decide(&request()).await.unwrap();

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        let head = seen[0].head.to_ascii_lowercase();
        assert!(head.starts_with("post /v1/systemone "), "{head}");
        assert!(head.contains(&format!(
            "authorization: bearer {}",
            KEY.to_ascii_lowercase()
        )));
        let body = &seen[0].body;
        assert_eq!(body["model"], "jev-latest");
        assert!(body["state"].as_str().unwrap().contains("invoice 42"));
        assert!(body.get("input").is_none());
        let questions = &body["questions"];
        assert_eq!(questions["plain"]["type"], "noul");
        assert_eq!(questions["plain"]["instructions"], "Is it a request?");
        assert!(questions["plain"].get("criteria").is_none());
        assert_eq!(
            questions["described"]["criteria"],
            json!({"true": "It mentions money", "false": "It does not"})
        );
        assert_eq!(questions["topic"]["type"], "choice");
        assert_eq!(
            questions["topic"]["criteria"],
            json!({"billing": "Money", "other": "Else"})
        );
        assert_eq!(questions["urgency"]["type"], "score");
        assert_eq!(
            questions["urgency"]["criteria"],
            json!(["low", "mid", "high"])
        );
    }

    #[tokio::test]
    async fn a_confident_answer_keeps_its_probability_as_confidence() {
        let (base, _) = serve(Arc::new(answering_with(0.95))).await;
        let decider = TypesafeDecider::new(KEY)
            .with_model("jev-1.14.0")
            .with_base_url(base);
        let response = decider.decide(&request()).await.unwrap();

        assert_eq!(response.backend, "typesafe:jev-1.14.0");
        assert_eq!(response.usage.input_tokens, 120);
        assert_eq!(response.usage.output_tokens, 12);
        let plain = &response.answers["plain"];
        assert!((plain.p_yes().unwrap() - 0.95).abs() < 1e-6);
        assert!((plain.confidence - 0.95).abs() < 1e-6);
        let topic = &response.answers["topic"];
        assert_eq!(topic.chosen(), Some("billing"));
        // The chosen option's probability, not Jev's concentration
        // measure (0.9 here).
        assert!((topic.confidence - 0.95).abs() < 1e-6);
        assert_eq!(topic.probabilities.as_ref().unwrap().len(), 2);
        let urgency = &response.answers["urgency"];
        assert!((urgency.score().unwrap() - 1.9).abs() < 1e-6);
        assert!((urgency.confidence - 0.95).abs() < 1e-6);
        assert!(
            response
                .answers
                .values()
                .all(|a| a.confidence_source == ConfidenceSource::Calibrated)
        );
    }

    #[tokio::test]
    async fn a_low_confidence_answer_stays_low() {
        let (base, _) = serve(Arc::new(answering_with(0.55))).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        let response = decider.decide(&request()).await.unwrap();

        let plain = &response.answers["plain"];
        assert!((plain.p_yes().unwrap() - 0.55).abs() < 1e-6);
        assert!((plain.confidence - 0.55).abs() < 1e-6);
        let topic = &response.answers["topic"];
        assert_eq!(topic.chosen(), Some("billing"));
        assert!((topic.confidence - 0.55).abs() < 1e-6);
        // A noul near "no" is a confident no.
        let (base, _) = serve(Arc::new(answering_with(0.1))).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        let only_noul = DecisionRequest::new("hi").ask("plain", Question::yes_no("?"));
        let response = decider.decide(&only_noul).await.unwrap();
        assert!((response.answers["plain"].confidence - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn a_choice_without_probabilities_inverts_the_confidence() {
        let reply = json!({
            "model": "jev-1.14.0",
            "answers": {"topic": {"type": "choice", "choice": "other", "confidence": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        let (base, _) = serve(fixed(200, &reply.to_string())).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        let request = DecisionRequest::new("hi").ask(
            "topic",
            Question::choice("Topic?", [("billing", "Money"), ("other", "Else")]),
        );
        let response = decider.decide(&request).await.unwrap();
        // Two options, confidence 0.5: top probability 0.75.
        assert!((response.answers["topic"].confidence - 0.75).abs() < 1e-6);
    }

    #[tokio::test]
    async fn an_unknown_option_is_malformed() {
        let reply = json!({
            "model": "jev-1.14.0",
            "answers": {"topic": {"type": "choice", "choice": "invented", "confidence": 0.9,
                "probabilities": {"invented": 0.9}}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        let (base, _) = serve(fixed(200, &reply.to_string())).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        let request = DecisionRequest::new("hi").ask(
            "topic",
            Question::choice("Topic?", [("billing", "Money"), ("other", "Else")]),
        );
        assert!(matches!(
            decider.decide(&request).await,
            Err(DecisionError::Malformed(_))
        ));
    }

    #[tokio::test]
    async fn http_errors_map_by_status_without_the_key() {
        let cases: [(u16, &str, &str); 6] = [
            (400, r#"{"detail":"max_tokens_exceeded"}"#, "invalid"),
            (
                422,
                r#"{"detail":[{"loc":["body","state"],"msg":"Field required","type":"missing"}]}"#,
                "invalid",
            ),
            (401, r#"{"detail":"Invalid API key"}"#, "auth"),
            (429, r#"{"detail":"rate limited"}"#, "rate"),
            (529, r#"{"detail":"overloaded"}"#, "transient"),
            (503, "not json", "transient"),
        ];
        for (status, body, expect) in cases {
            let (base, _) = serve(fixed(status, body)).await;
            let decider = TypesafeDecider::new(KEY).with_base_url(base);
            let error = decider.decide(&request()).await.unwrap_err();
            let ok = match expect {
                "invalid" => matches!(error, DecisionError::InvalidRequest(_)),
                "auth" => matches!(error, DecisionError::Auth(_)),
                "rate" => error == DecisionError::RateLimited,
                _ => matches!(error, DecisionError::Transient(_)),
            };
            assert!(ok, "HTTP {status} gave {error:?}");
            assert!(!error.to_string().contains(KEY));
        }
        let (base, _) = serve(fixed(422, r#"{"detail":[{"msg":"Field required"}]}"#)).await;
        let error = TypesafeDecider::new(KEY)
            .with_base_url(base)
            .decide(&request())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Field required"), "{error}");
    }

    #[tokio::test]
    async fn too_many_score_levels_never_reach_the_server() {
        let (base, seen) = serve(Arc::new(answering_with(0.9))).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        let levels: Vec<String> = (0..11).map(|i| format!("level {i}")).collect();
        let request = DecisionRequest::new("hi").ask("s", Question::score("How much?", levels));
        assert!(matches!(
            decider.decide(&request).await,
            Err(DecisionError::InvalidRequest(_))
        ));
        assert!(seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_unreachable_server_is_transient() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let decider = TypesafeDecider::new(KEY).with_base_url(format!("http://{addr}"));
        assert!(matches!(
            decider.decide(&request()).await,
            Err(DecisionError::Transient(_))
        ));
    }

    #[test]
    fn debug_redacts_the_key() {
        let shown = format!("{:?}", TypesafeDecider::new(KEY));
        assert!(!shown.contains(KEY));
        assert!(shown.contains("redacted"));
    }

    #[tokio::test]
    async fn passes_the_conformance_kit() {
        let (base, seen) = serve(Arc::new(answering_with(0.8))).await;
        let decider = TypesafeDecider::new(KEY).with_base_url(base);
        conformance::check_decider(&decider).await;
        assert_eq!(seen.lock().unwrap().len(), 1);
    }
}
