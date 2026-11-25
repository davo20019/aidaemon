//! [`GeminiDecider`]: Google's Decisions API (`gemini-decision-1`), which
//! returns calibrated probabilities.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::wire::{f32_field, network_error, probabilities, short};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::{
    Answer, AnswerValue, ConfidenceSource, Decider, DeciderCapabilities, DecisionError,
    DecisionRequest, DecisionResponse, QuestionKind,
};

const DEFAULT_MODEL: &str = "gemini-decision-1";
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const SAFETY_CODES: [&str; 5] = [
    "safety",
    "prohibited_content",
    "blocklist",
    "image_safety",
    "content_blocked",
];

/// Calls Google's Decisions API. Its confidence is
/// [`ConfidenceSource::Calibrated`].
pub struct GeminiDecider {
    api_key: String,
    model: String,
    base_url: String,
    name: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for GeminiDecider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiDecider")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl GeminiDecider {
    /// The model a decider uses unless [`Self::with_model`] picks another.
    pub const DEFAULT_MODEL: &'static str = DEFAULT_MODEL;

    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.into(),
            base_url: DEFAULT_BASE_URL.into(),
            name: format!("gemini:{DEFAULT_MODEL}"),
            client,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.name = format!("gemini:{}", self.model);
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
}

#[async_trait]
impl Decider for GeminiDecider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> DeciderCapabilities {
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
        let reply = self
            .client
            .post(format!("{}/decisions", self.base_url))
            .header("x-goog-api-key", &self.api_key)
            .json(&request_body(&self.model, request))
            .send()
            .await
            .map_err(|error| DecisionError::Transient(network_error("decisions API", &error)))?;
        let status = reply.status();
        let body = reply
            .text()
            .await
            .map_err(|error| DecisionError::Transient(network_error("decisions API", &error)))?;
        if !status.is_success() {
            return Err(status_error(status.as_u16(), &body));
        }
        let body: Value = serde_json::from_str(&body)
            .map_err(|error| DecisionError::Malformed(format!("response is not JSON: {error}")))?;
        let answers = parse_answers(&body, request)?;
        let usage = aidaemon_core::Usage {
            input_tokens: body
                .pointer("/usage/total_input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
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
                        criteria.insert("true_description".into(), json!(yes));
                    }
                    if let Some(no) = false_description {
                        criteria.insert("false_description".into(), json!(no));
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
        "input": request.input_text(),
        "questions": questions,
    })
}

fn status_error(status: u16, body: &str) -> DecisionError {
    let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let error = parsed.get("error");
    let field = |name: &str| {
        error
            .and_then(|e| e.get(name))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    let message = field("message");
    let detail = if message.is_empty() {
        short(body)
    } else {
        short(&message)
    };
    match status {
        429 => DecisionError::RateLimited,
        401 | 403 => DecisionError::Auth(format!("HTTP {status}: {detail}")),
        500..=599 => DecisionError::Transient(format!("HTTP {status}: {detail}")),
        400 => {
            let is_safety = [field("code"), field("status"), message.clone()]
                .iter()
                .any(|value| SAFETY_CODES.contains(&value.to_ascii_lowercase().as_str()));
            if is_safety {
                DecisionError::SafetyBlock(detail)
            } else {
                DecisionError::InvalidRequest(detail)
            }
        }
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
        let probs = probabilities(value);
        let mut answer = match &question.kind {
            QuestionKind::YesNo { .. } => {
                let p = f32_field(value, "noul").ok_or_else(|| missing("noul"))?;
                let p = p.clamp(0.0, 1.0);
                let confidence = f32_field(value, "confidence").unwrap_or(p.max(1.0 - p));
                Answer::new(AnswerValue::YesNo(p), confidence, source)
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
                let confidence = f32_field(value, "confidence")
                    .or_else(|| probs.as_ref().and_then(|p| p.get(choice).copied()))
                    .ok_or_else(|| missing("confidence"))?;
                Answer::choice(choice, confidence, source)
            }
            QuestionKind::Score { levels } => {
                let top = (levels.len() - 1) as f32;
                let score = f32_field(value, "score")
                    .ok_or_else(|| missing("score"))?
                    .clamp(0.0, top);
                let confidence = f32_field(value, "confidence")
                    .or_else(|| {
                        probs
                            .as_ref()
                            .and_then(|p| p.values().copied().reduce(f32::max))
                    })
                    .ok_or_else(|| missing("confidence"))?;
                Answer::new(AnswerValue::Score(score), confidence, source)
            }
        };
        if let Some(probs) = probs {
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

    const KEY: &str = "synthetic-test-key-123";

    /// Answers whatever it is asked, per question type.
    fn answer_everything(body: &Value) -> (u16, String) {
        let mut answers = Map::new();
        for (key, question) in body["questions"].as_object().unwrap() {
            let answer = match question["type"].as_str().unwrap() {
                "noul" => json!({"noul": 0.85, "confidence": 0.85}),
                "choice" => {
                    let options: Vec<&String> =
                        question["criteria"].as_object().unwrap().keys().collect();
                    let share = 1.0 / options.len() as f64;
                    let probabilities: Map<String, Value> = options
                        .iter()
                        .map(|o| ((*o).clone(), json!(share)))
                        .collect();
                    json!({"choice": options[0], "probabilities": probabilities, "confidence": share})
                }
                "score" => {
                    let levels = question["criteria"].as_array().unwrap().len();
                    let share = 1.0 / levels as f64;
                    let probabilities: Map<String, Value> =
                        (0..levels).map(|i| (i.to_string(), json!(share))).collect();
                    json!({"score": (levels - 1) as f64 / 2.0, "legend": "mean level", "probabilities": probabilities, "confidence": share})
                }
                other => panic!("unexpected type {other}"),
            };
            answers.insert(key.clone(), answer);
        }
        (
            200,
            json!({"answers": answers, "usage": {"total_input_tokens": 321}}).to_string(),
        )
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
    async fn sends_the_api_shape_and_parses_answers() {
        let (base, seen) = serve(Arc::new(answer_everything)).await;
        let decider = GeminiDecider::new(KEY).with_base_url(base);
        let response = decider.decide(&request()).await.unwrap();

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        let head = seen[0].head.to_ascii_lowercase();
        assert!(head.starts_with("post /decisions "), "{head}");
        assert!(head.contains(&format!("x-goog-api-key: {}", KEY.to_ascii_lowercase())));
        let body = &seen[0].body;
        assert_eq!(body["model"], "gemini-decision-1");
        assert!(body["input"].as_str().unwrap().contains("invoice 42"));
        let questions = &body["questions"];
        assert_eq!(questions["plain"]["type"], "noul");
        assert!(questions["plain"].get("criteria").is_none());
        assert_eq!(
            questions["described"]["criteria"],
            json!({"true_description": "It mentions money", "false_description": "It does not"})
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
        assert_eq!(questions["urgency"]["instructions"], "Urgency?");

        assert_eq!(response.backend, "gemini:gemini-decision-1");
        assert_eq!(response.usage.input_tokens, 321);
        assert!((response.answers["plain"].p_yes().unwrap() - 0.85).abs() < 1e-6);
        let topic = &response.answers["topic"];
        assert_eq!(topic.chosen(), Some("billing"));
        assert_eq!(topic.probabilities.as_ref().unwrap().len(), 2);
        assert!((response.answers["urgency"].score().unwrap() - 1.0).abs() < 1e-6);
        assert!(
            response
                .answers
                .values()
                .all(|a| a.confidence_source == ConfidenceSource::Calibrated)
        );
    }

    #[tokio::test]
    async fn rate_limit_maps_to_rate_limited() {
        let (base, _) = serve(fixed(429, r#"{"error":{"code":429}}"#)).await;
        let decider = GeminiDecider::new(KEY).with_base_url(base);
        assert_eq!(
            decider.decide(&request()).await,
            Err(DecisionError::RateLimited)
        );
    }

    #[tokio::test]
    async fn a_safety_400_is_a_safety_block() {
        let (base, _) = serve(fixed(
            400,
            r#"{"error":{"code":"safety","message":"input blocked"}}"#,
        ))
        .await;
        let decider = GeminiDecider::new(KEY).with_base_url(base);
        assert!(matches!(
            decider.decide(&request()).await,
            Err(DecisionError::SafetyBlock(_))
        ));
    }

    #[tokio::test]
    async fn other_errors_map_by_status_without_the_key() {
        for (status, expect) in [(400, "invalid"), (401, "auth"), (503, "transient")] {
            let (base, _) = serve(fixed(
                status,
                r#"{"error":{"code":"bad","message":"nope"}}"#,
            ))
            .await;
            let decider = GeminiDecider::new(KEY).with_base_url(base);
            let error = decider.decide(&request()).await.unwrap_err();
            let ok = match expect {
                "invalid" => matches!(error, DecisionError::InvalidRequest(_)),
                "auth" => matches!(error, DecisionError::Auth(_)),
                _ => matches!(error, DecisionError::Transient(_)),
            };
            assert!(ok, "HTTP {status} gave {error:?}");
            assert!(!error.to_string().contains(KEY));
        }
    }

    #[tokio::test]
    async fn an_unreachable_server_is_transient() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let decider = GeminiDecider::new(KEY).with_base_url(format!("http://{addr}"));
        assert!(matches!(
            decider.decide(&request()).await,
            Err(DecisionError::Transient(_))
        ));
    }

    #[test]
    fn debug_redacts_the_key() {
        let shown = format!("{:?}", GeminiDecider::new(KEY));
        assert!(!shown.contains(KEY));
        assert!(shown.contains("redacted"));
    }

    #[tokio::test]
    async fn passes_the_conformance_kit() {
        let (base, seen) = serve(Arc::new(answer_everything)).await;
        let decider = GeminiDecider::new(KEY).with_base_url(base);
        conformance::check_decider(&decider).await;
        // The invalid request never reached the server.
        assert_eq!(seen.lock().unwrap().len(), 1);
    }
}
