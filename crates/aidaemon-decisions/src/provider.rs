//! [`ProviderDecider`]: asks a chat provider every question at once,
//! through a JSON schema.

use std::collections::BTreeMap;
use std::sync::Arc;

use aidaemon_core::{Message, ModelRequest, Provider, RequestOptions, ResponseFormat, ToolChoice};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::{
    Answer, AnswerValue, ConfidenceSource, Decider, DeciderCapabilities, DecisionError,
    DecisionRequest, DecisionResponse, QuestionKind,
};

const SYSTEM_PROMPT: &str = "You answer closed questions about an input. \
Answer every question. For a yes/no question give p_yes, the probability \
from 0 to 1 that the answer is yes. For a choice give the option key you \
pick and your confidence from 0 to 1 that it is right. For a score give \
the level number you pick and your confidence from 0 to 1. Reply with one \
JSON object only, matching the schema.";

/// Answers questions with any [`Provider`]. Its confidence is the model's
/// own statement, so it is [`ConfidenceSource::SelfReported`].
pub struct ProviderDecider {
    provider: Arc<dyn Provider>,
    model: String,
    name: String,
}

impl std::fmt::Debug for ProviderDecider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderDecider")
            .field("provider", &self.provider.name())
            .field("model", &self.model)
            .finish()
    }
}

impl ProviderDecider {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            provider,
            name: format!("provider:{model}"),
            model,
        }
    }

    fn model_request(&self, request: &DecisionRequest) -> ModelRequest {
        ModelRequest {
            model: self.model.clone(),
            messages: vec![
                Message::system(SYSTEM_PROMPT),
                Message::user(prompt(request)),
            ],
            tools: Vec::new(),
            tool_choice: ToolChoice::None,
            max_output_tokens: Some(1024),
            options: RequestOptions {
                response_format: ResponseFormat::JsonSchema {
                    name: "decisions".into(),
                    schema: schema(request),
                    strict: true,
                },
                reasoning_effort: Some("low".into()),
                ..Default::default()
            },
        }
    }
}

#[async_trait]
impl Decider for ProviderDecider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> DeciderCapabilities {
        DeciderCapabilities::new(64, ConfidenceSource::SelfReported)
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
        let reply = self.provider.complete(&self.model_request(request)).await?;
        let answers = parse_reply(&reply.text, request)?;
        let response = DecisionResponse::new(self.name.clone(), answers, reply.usage);
        response.check_against(request)?;
        Ok(response)
    }
}

fn prompt(request: &DecisionRequest) -> String {
    let mut out = String::from("Input:\n<<<\n");
    out.push_str(&request.input_text());
    out.push_str("\n>>>\n\nQuestions (answer every one):\n");
    for (key, question) in &request.questions {
        out.push_str(&format!("\n[{key}] {}\n", question.instructions));
        match &question.kind {
            QuestionKind::YesNo {
                true_description,
                false_description,
            } => {
                out.push_str(
                    "Type: yes/no. Give p_yes, the probability (0 to 1) that the answer is yes.\n",
                );
                if let Some(yes) = true_description {
                    out.push_str(&format!("Yes means: {yes}\n"));
                }
                if let Some(no) = false_description {
                    out.push_str(&format!("No means: {no}\n"));
                }
            }
            QuestionKind::Choice { options } => {
                out.push_str("Type: choice. Give the option key as \"choice\" and your confidence (0 to 1).\nOptions:\n");
                for (option, description) in options {
                    out.push_str(&format!("- {option}: {description}\n"));
                }
            }
            QuestionKind::Score { levels } => {
                out.push_str("Type: score. Give the level number as \"level\" and your confidence (0 to 1).\nLevels, lowest first:\n");
                for (index, description) in levels.iter().enumerate() {
                    out.push_str(&format!("- {index}: {description}\n"));
                }
            }
        }
    }
    out
}

fn schema(request: &DecisionRequest) -> Value {
    let mut properties = Map::new();
    for (key, question) in &request.questions {
        let property = match &question.kind {
            QuestionKind::YesNo { .. } => json!({
                "type": "object",
                "properties": { "p_yes": { "type": "number" } },
                "required": ["p_yes"],
                "additionalProperties": false,
            }),
            QuestionKind::Choice { options } => json!({
                "type": "object",
                "properties": {
                    "choice": { "type": "string", "enum": options.keys().collect::<Vec<_>>() },
                    "confidence": { "type": "number" },
                },
                "required": ["choice", "confidence"],
                "additionalProperties": false,
            }),
            QuestionKind::Score { levels } => json!({
                "type": "object",
                "properties": {
                    "level": { "type": "integer", "minimum": 0, "maximum": levels.len() - 1 },
                    "confidence": { "type": "number" },
                },
                "required": ["level", "confidence"],
                "additionalProperties": false,
            }),
        };
        properties.insert(key.clone(), property);
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": request.questions.keys().collect::<Vec<_>>(),
        "additionalProperties": false,
    })
}

/// The first JSON object in `text`: from the first `{` to the last `}`, so
/// code fences and prose around it are ignored.
fn extract_object(text: &str) -> Result<Map<String, Value>, DecisionError> {
    let start = text.find('{');
    let end = text.rfind('}');
    let (Some(start), Some(end)) = (start, end) else {
        return Err(DecisionError::Malformed("no JSON object in reply".into()));
    };
    if end < start {
        return Err(DecisionError::Malformed("no JSON object in reply".into()));
    }
    match serde_json::from_str::<Value>(&text[start..=end]) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(DecisionError::Malformed(
            "reply is not a JSON object".into(),
        )),
        Err(error) => Err(DecisionError::Malformed(format!(
            "reply is not valid JSON: {error}"
        ))),
    }
}

fn number(value: &Value, field: &str, key: &str) -> Result<f64, DecisionError> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .ok_or_else(|| DecisionError::Malformed(format!("{key:?} has no numeric {field:?}")))
}

fn parse_reply(
    text: &str,
    request: &DecisionRequest,
) -> Result<BTreeMap<String, Answer>, DecisionError> {
    let object = extract_object(text)?;
    if let Some(unknown) = object
        .keys()
        .find(|key| !request.questions.contains_key(*key))
    {
        return Err(DecisionError::Malformed(format!(
            "answer for unknown question {unknown:?}"
        )));
    }
    let source = ConfidenceSource::SelfReported;
    let mut answers = BTreeMap::new();
    for (key, question) in &request.questions {
        let Some(value) = object.get(key) else {
            return Err(DecisionError::Malformed(format!("no answer for {key:?}")));
        };
        let answer = match &question.kind {
            QuestionKind::YesNo { .. } => {
                Answer::yes_no(number(value, "p_yes", key)? as f32, source)
            }
            QuestionKind::Choice { options } => {
                let choice = value.get("choice").and_then(Value::as_str).ok_or_else(|| {
                    DecisionError::Malformed(format!("{key:?} has no \"choice\""))
                })?;
                if !options.contains_key(choice) {
                    return Err(DecisionError::Malformed(format!(
                        "{key:?} chose unknown option {choice:?}"
                    )));
                }
                let confidence = number(value, "confidence", key)? as f32;
                Answer::choice(choice, confidence, source)
            }
            QuestionKind::Score { levels } => {
                let top = (levels.len() - 1) as f64;
                let level = number(value, "level", key)?.round().clamp(0.0, top);
                let confidence = number(value, "confidence", key)? as f32;
                Answer::new(AnswerValue::Score(level as f32), confidence, source)
            }
        };
        answers.insert(key.clone(), answer);
    }
    Ok(answers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Question;
    use aidaemon_core::testing::{ScriptedProvider, text};

    fn request() -> DecisionRequest {
        DecisionRequest::new("Can we meet at noon on Friday? -- Alice Rivera")
            .ask(
                "is_question",
                Question::yes_no("Does the message ask something?"),
            )
            .ask(
                "topic",
                Question::choice(
                    "What is it about?",
                    [
                        ("meeting", "Scheduling"),
                        ("billing", "Money"),
                        ("other", "Else"),
                    ],
                ),
            )
            .ask(
                "urgency",
                Question::score("How urgent?", ["low", "mid", "high"]),
            )
    }

    fn decider(reply: &str) -> (ProviderDecider, Arc<ScriptedProvider>) {
        let provider = Arc::new(ScriptedProvider::new([text(reply)]));
        (
            ProviderDecider::new(provider.clone(), "test-model"),
            provider,
        )
    }

    const GOOD: &str = r#"{"is_question":{"p_yes":0.9},"topic":{"choice":"meeting","confidence":0.8},"urgency":{"level":1,"confidence":0.6}}"#;

    #[tokio::test]
    async fn answers_every_question_in_one_call() {
        let (decider, provider) = decider(GOOD);
        let response = decider.decide(&request()).await.unwrap();
        assert_eq!(response.backend, "provider:test-model");
        assert_eq!(response.answers["is_question"].p_yes(), Some(0.9));
        assert_eq!(response.answers["topic"].chosen(), Some("meeting"));
        assert_eq!(response.answers["urgency"].score(), Some(1.0));
        assert!(
            response
                .answers
                .values()
                .all(|a| a.confidence_source == ConfidenceSource::SelfReported)
        );
        assert_eq!(response.usage.input_tokens, 100);

        let requests = provider.requests();
        assert_eq!(requests.len(), 1);
        let sent = &requests[0];
        assert!(sent.tools.is_empty());
        assert_eq!(sent.tool_choice, ToolChoice::None);
        assert_eq!(sent.options.reasoning_effort.as_deref(), Some("low"));
        let ResponseFormat::JsonSchema { schema, .. } = &sent.options.response_format else {
            panic!("expected a JSON schema");
        };
        assert_eq!(schema["required"].as_array().unwrap().len(), 3);
        assert_eq!(
            schema["properties"]["topic"]["properties"]["choice"]["enum"],
            json!(["billing", "meeting", "other"])
        );
        assert_eq!(
            schema["properties"]["urgency"]["properties"]["level"]["maximum"],
            json!(2)
        );
    }

    #[tokio::test]
    async fn tolerates_a_code_fence_and_clamps() {
        let reply = "Here you go:\n```json\n{\"is_question\":{\"p_yes\":1.4},\"topic\":{\"choice\":\"other\",\"confidence\":0.5},\"urgency\":{\"level\":7,\"confidence\":0.4}}\n```";
        let (decider, _) = decider(reply);
        let response = decider.decide(&request()).await.unwrap();
        assert_eq!(response.answers["is_question"].p_yes(), Some(1.0));
        assert_eq!(response.answers["urgency"].score(), Some(2.0));
    }

    #[tokio::test]
    async fn a_missing_answer_is_malformed() {
        let (decider, _) = decider(
            r#"{"is_question":{"p_yes":0.9},"topic":{"choice":"meeting","confidence":0.8}}"#,
        );
        assert!(matches!(
            decider.decide(&request()).await,
            Err(DecisionError::Malformed(_))
        ));
    }

    #[tokio::test]
    async fn an_unknown_option_is_malformed() {
        let (decider, _) = decider(
            r#"{"is_question":{"p_yes":0.9},"topic":{"choice":"travel","confidence":0.8},"urgency":{"level":1,"confidence":0.6}}"#,
        );
        assert!(matches!(
            decider.decide(&request()).await,
            Err(DecisionError::Malformed(_))
        ));
    }

    #[tokio::test]
    async fn an_invalid_request_is_not_sent() {
        let (decider, provider) = decider(GOOD);
        let bad = DecisionRequest::new("x").ask("q", Question::choice("pick", [("a", "A")]));
        assert!(matches!(
            decider.decide(&bad).await,
            Err(DecisionError::InvalidRequest(_))
        ));
        assert!(provider.requests().is_empty());
    }

    #[tokio::test]
    async fn provider_errors_map_through() {
        let provider = Arc::new(ScriptedProvider::with_results([Err(
            aidaemon_core::ProviderError::Transient("boom".into()),
        )]));
        let decider = ProviderDecider::new(provider, "m");
        assert!(matches!(
            decider.decide(&request()).await,
            Err(DecisionError::Transient(_))
        ));
    }

    #[tokio::test]
    async fn passes_the_conformance_kit() {
        let reply = crate::testing::conformance::KEYS;
        let body = format!(
            r#"{{"{}":{{"p_yes":0.2}},"{}":{{"choice":"{}","confidence":0.7}},"{}":{{"level":2,"confidence":0.5}}}}"#,
            reply.yes_no, reply.choice, reply.choice_option, reply.score
        );
        let provider = Arc::new(ScriptedProvider::new([]).then_repeat(text(body)));
        crate::testing::conformance::check_decider(&ProviderDecider::new(provider, "m")).await;
    }
}
