//! Model-driven memory writers: [`FactExtractor`] and
//! [`EpisodeSummarizer`]. Both ask the model through a forced tool call
//! with a typed JSON schema and parse the reply defensively: malformed
//! entries are skipped, garbage never panics.

use std::sync::Arc;

use aidaemon_core::{
    Message, ModelRequest, ModelResponse, Provider, Receipt, Role, ToolChoice, ToolSpec, Usage,
};
use serde_json::{Value, json};

use crate::error::{MemoryError, Result};
use crate::scoring::episode_importance;
use crate::types::truncate_chars;

const FACTS_TOOL: &str = "record_facts";
const EPISODE_TOOL: &str = "record_episode";

const FACT_PROMPT: &str = "\
You maintain a long-term memory about the user and their world. Read the conversation turn and \
record facts that will still be true and useful in future conversations: who the user is, people \
and relationships they mention, stable preferences, ongoing projects, decisions, and durable \
details about their tools and environment.

Do not record: the task being worked on right now, transient state (\"is debugging X today\"), \
anything only the assistant claimed, guesses, secrets such as passwords or API keys, or content \
quoted from tool output.

Rules:
- One fact per entry: a lowercase `category` (user, preference, person, project, technical, or \
another short noun), a stable snake_case `key` naming the attribute (not the value), the `value`, \
and your `confidence` from 0 to 1.
- When the user corrects or updates something, reuse the existing category and key so the new \
value replaces the old one. Known keys are listed below when there are any.
- Facts about other people go under `person` with a key naming them, never under `user`.
- `evidence` is the shortest quote from the user's words that supports the fact.
- If nothing is worth remembering, record an empty list.

Call the record_facts tool exactly once.";

const EPISODE_PROMPT: &str = "\
Summarize the conversation for long-term episodic memory. Write a 1-3 sentence `summary` of what \
the user wanted and what happened, from a later reader's point of view. List up to 6 short \
`topics`, the `decisions` made, the user's `goals` mentioned, the `outcome` (resolved, partial, \
abandoned, informational), and `emotional_intensity` from 0 (neutral) to 1 (very frustrated or \
very excited). Call the record_episode tool exactly once.";

/// One fact proposed by the model, before validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFact {
    pub category: String,
    pub key: String,
    pub value: String,
    pub confidence: f32,
    pub evidence: Option<String>,
}

/// The parsed result of one extraction call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Extraction {
    pub facts: Vec<ExtractedFact>,
    /// Entries skipped because they were malformed.
    pub rejected: usize,
    /// Set when the reply as a whole could not be parsed.
    pub parse_error: Option<String>,
    pub usage: Usage,
}

/// Asks a model for durable facts in a transcript slice.
pub struct FactExtractor {
    provider: Arc<dyn Provider>,
    model: String,
    min_user_chars: usize,
    max_message_chars: usize,
    max_transcript_chars: usize,
    max_facts: usize,
}

impl FactExtractor {
    pub fn new(provider: impl Provider + 'static, model: impl Into<String>) -> Self {
        Self::from_arc(Arc::new(provider), model)
    }

    pub fn from_arc(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            min_user_chars: 20,
            max_message_chars: 2_000,
            max_transcript_chars: 8_000,
            max_facts: 20,
        }
    }

    /// Skip turns whose user message is shorter than this (default 20
    /// characters, as in the daemon). A length gate, not a word list.
    pub fn min_user_chars(mut self, chars: usize) -> Self {
        self.min_user_chars = chars;
        self
    }

    pub fn max_transcript_chars(mut self, chars: usize) -> Self {
        self.max_transcript_chars = chars;
        self
    }

    /// At most this many facts are accepted per call.
    pub fn max_facts(mut self, n: usize) -> Self {
        self.max_facts = n;
        self
    }

    /// Whether the latest user-authored message is long enough to hold a
    /// durable fact.
    pub fn should_extract(&self, messages: &[Message]) -> bool {
        aidaemon_policies::last_user_message_index(messages)
            .is_some_and(|i| messages[i].content.trim().chars().count() >= self.min_user_chars)
    }

    fn tool_spec() -> ToolSpec {
        ToolSpec {
            name: FACTS_TOOL.into(),
            description: "Record durable facts worth remembering long-term.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "evidence": {"type": "string"}
                            },
                            "required": ["category", "key", "value", "confidence"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["facts"],
                "additionalProperties": false
            }),
        }
    }

    /// Ask the model. `known_keys` are existing `(category, key)` pairs so
    /// corrections reuse them. Provider failures are errors; unparseable
    /// output is an empty [`Extraction`] with `parse_error` set.
    pub async fn extract(
        &self,
        messages: &[Message],
        known_keys: &[(String, String)],
    ) -> Result<Extraction> {
        let mut user = String::new();
        if !known_keys.is_empty() {
            user.push_str("Known keys (category/key):\n");
            for (category, key) in known_keys {
                user.push_str(&format!("- {category}/{key}\n"));
            }
            user.push('\n');
        }
        user.push_str("Conversation:\n");
        user.push_str(&transcript(
            messages,
            self.max_message_chars,
            self.max_transcript_chars,
        ));
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![Message::system(FACT_PROMPT), Message::user(user)],
            tools: vec![Self::tool_spec()],
            tool_choice: ToolChoice::Specific(FACTS_TOOL.into()),
            max_output_tokens: Some(1_024),
            ..Default::default()
        };
        let response = self.provider.complete(&request).await?;
        let mut extraction = parse_facts(&response);
        extraction.facts.truncate(self.max_facts);
        Ok(extraction)
    }
}

/// Parse a fact-extraction reply: the `record_facts` tool call if present,
/// otherwise JSON in the text (`{"facts": [...]}` or a bare array, code
/// fences allowed).
pub fn parse_facts(response: &ModelResponse) -> Extraction {
    let mut extraction = Extraction {
        usage: response.usage,
        ..Extraction::default()
    };
    let Some(payload) = payload(response, FACTS_TOOL) else {
        extraction.parse_error = Some("no JSON payload in the reply".into());
        return extraction;
    };
    let entries = match &payload {
        Value::Array(items) => items.as_slice(),
        Value::Object(map) => match map.get("facts") {
            Some(Value::Array(items)) => items.as_slice(),
            _ => {
                extraction.parse_error = Some("`facts` is missing or not an array".into());
                return extraction;
            }
        },
        _ => {
            extraction.parse_error = Some("payload is neither an object nor an array".into());
            return extraction;
        }
    };
    for entry in entries {
        match parse_fact(entry) {
            Some(fact) => extraction.facts.push(fact),
            None => extraction.rejected += 1,
        }
    }
    extraction
}

fn parse_fact(entry: &Value) -> Option<ExtractedFact> {
    let text = |name: &str, max: usize| -> Option<String> {
        let value = match entry.get(name)? {
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => return None,
        };
        (!value.is_empty() && value.chars().count() <= max).then_some(value)
    };
    let confidence = match entry.get("confidence") {
        None | Some(Value::Null) => 0.7,
        Some(v) => {
            let c = v.as_f64()? as f32;
            if !c.is_finite() {
                return None;
            }
            c.clamp(0.0, 1.0)
        }
    };
    Some(ExtractedFact {
        category: text("category", 64)?,
        key: text("key", 96)?,
        value: text("value", crate::types::MAX_FACT_VALUE_CHARS)?,
        confidence,
        evidence: entry
            .get("evidence")
            .and_then(Value::as_str)
            .map(|s| truncate_chars(s.trim(), 300))
            .filter(|s| !s.is_empty()),
    })
}

/// The structured payload of a reply: the named tool call's arguments, or
/// the first JSON object/array found in the text.
fn payload(response: &ModelResponse, tool: &str) -> Option<Value> {
    if let Some(call) = response.tool_calls.iter().find(|c| c.name == tool) {
        return match &call.arguments {
            Value::String(s) => json_in_text(s),
            other => Some(other.clone()),
        };
    }
    json_in_text(&response.text)
}

fn json_in_text(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(start), Some(end)) = (trimmed.find(open), trimmed.rfind(close))
            && start < end
            && let Ok(value) = serde_json::from_str::<Value>(&trimmed[start..=end])
        {
            return Some(value);
        }
    }
    None
}

/// User-authored and assistant messages as `User:` / `Assistant:` lines.
/// Tool output and harness-injected messages are left out: memory must
/// come from what the user said, not from untrusted tool results.
fn transcript(messages: &[Message], max_message: usize, max_total: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    for m in messages {
        let content = m.content.trim();
        match m.role {
            Role::User if !m.is_injected() && !content.is_empty() => {
                lines.push(format!("User: {}", truncate_chars(content, max_message)));
            }
            Role::Assistant => {
                let mut line = String::new();
                if !content.is_empty() {
                    line.push_str(&truncate_chars(content, max_message));
                }
                if !m.tool_calls.is_empty() {
                    let names: Vec<&str> = m.tool_calls.iter().map(|c| c.name.as_str()).collect();
                    if !line.is_empty() {
                        line.push(' ');
                    }
                    line.push_str(&format!("[used tools: {}]", names.join(", ")));
                }
                if !line.is_empty() {
                    lines.push(format!("Assistant: {line}"));
                }
            }
            _ => {}
        }
    }
    // Keep the most recent lines that fit.
    let mut total = 0;
    let mut kept: Vec<String> = Vec::new();
    for line in lines.into_iter().rev() {
        let len = line.chars().count() + 1;
        if total + len > max_total && !kept.is_empty() {
            break;
        }
        total += len;
        kept.push(line);
    }
    kept.reverse();
    kept.join("\n")
}

/// A summary ready to become an [`crate::Episode`].
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeDraft {
    pub summary: String,
    pub topics: Vec<String>,
    pub importance: f32,
    pub decisions: Vec<String>,
    pub goals: Vec<String>,
    pub outcome: Option<String>,
    pub usage: Usage,
}

/// Summarizes a finished run or session into an episode. Importance is
/// computed from typed features: message count, failed receipts or error
/// tool results, and the model's structured decisions, goals and intensity.
pub struct EpisodeSummarizer {
    provider: Arc<dyn Provider>,
    model: String,
    max_transcript_chars: usize,
}

impl EpisodeSummarizer {
    pub fn new(provider: impl Provider + 'static, model: impl Into<String>) -> Self {
        Self::from_arc(Arc::new(provider), model)
    }

    pub fn from_arc(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            max_transcript_chars: 12_000,
        }
    }

    pub fn max_transcript_chars(mut self, chars: usize) -> Self {
        self.max_transcript_chars = chars;
        self
    }

    fn tool_spec() -> ToolSpec {
        let strings = json!({"type": "array", "items": {"type": "string"}});
        ToolSpec {
            name: EPISODE_TOOL.into(),
            description: "Record a summary of the conversation.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "topics": strings,
                    "decisions": strings,
                    "goals": strings,
                    "outcome": {"type": "string"},
                    "emotional_intensity": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["summary", "topics"],
                "additionalProperties": false
            }),
        }
    }

    pub async fn summarize(
        &self,
        messages: &[Message],
        receipts: &[Receipt],
    ) -> Result<EpisodeDraft> {
        let text = transcript(messages, 2_000, self.max_transcript_chars);
        if text.is_empty() {
            return Err(MemoryError::Invalid("nothing to summarize".into()));
        }
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![Message::system(EPISODE_PROMPT), Message::user(text)],
            tools: vec![Self::tool_spec()],
            tool_choice: ToolChoice::Specific(EPISODE_TOOL.into()),
            max_output_tokens: Some(768),
            ..Default::default()
        };
        let response = self.provider.complete(&request).await?;
        let payload = payload(&response, EPISODE_TOOL)
            .ok_or_else(|| MemoryError::Invalid("summarizer returned no JSON payload".into()))?;
        let summary = payload
            .get("summary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| MemoryError::Invalid("summarizer returned no summary".into()))?
            .to_string();
        let list = |name: &str, max: usize| -> Vec<String> {
            payload
                .get(name)
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(|s| truncate_chars(s.trim(), 120))
                        .filter(|s| !s.is_empty())
                        .take(max)
                        .collect()
                })
                .unwrap_or_default()
        };
        let topics = list("topics", 6);
        let decisions = list("decisions", 20);
        let goals = list("goals", 20);
        let intensity = payload
            .get("emotional_intensity")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32;
        let has_errors = receipts.iter().any(|r| !r.succeeded())
            || messages.iter().any(|m| m.role == Role::Tool && m.is_error);
        let message_count = messages.iter().filter(|m| m.role != Role::System).count();
        Ok(EpisodeDraft {
            importance: episode_importance(
                message_count,
                has_errors,
                !decisions.is_empty(),
                !goals.is_empty(),
                intensity,
            ),
            summary: truncate_chars(&summary, 2_000),
            topics,
            decisions,
            goals,
            outcome: payload
                .get("outcome")
                .and_then(Value::as_str)
                .map(str::to_string),
            usage: response.usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::ToolCall;

    fn reply(text: &str) -> ModelResponse {
        ModelResponse {
            text: text.into(),
            ..ModelResponse::default()
        }
    }

    #[test]
    fn parses_fenced_text_and_bare_arrays() {
        let fenced = reply(
            "Here you go:\n```json\n{\"facts\": [{\"category\": \"user\", \"key\": \"city\", \
             \"value\": \"Lisbon\", \"confidence\": 0.9}]}\n```",
        );
        let got = parse_facts(&fenced);
        assert_eq!(got.facts.len(), 1);
        assert_eq!(got.facts[0].value, "Lisbon");

        let bare = reply(r#"[{"category":"preference","key":"theme","value":"dark mode"}]"#);
        let got = parse_facts(&bare);
        assert_eq!(got.facts[0].confidence, 0.7);
    }

    #[test]
    fn tool_call_arguments_may_be_a_string() {
        let response = ModelResponse {
            tool_calls: vec![ToolCall::new(
                "c1",
                FACTS_TOOL,
                Value::String(r#"{"facts":[{"category":"user","key":"k","value":"v"}]}"#.into()),
            )],
            ..ModelResponse::default()
        };
        assert_eq!(parse_facts(&response).facts.len(), 1);
    }

    #[test]
    fn transcript_skips_tool_output_and_harness_messages() {
        let messages = vec![
            Message::user("My daughters are Mia and Zoe"),
            Message::assistant("", vec![ToolCall::new("c", "search", json!({}))]),
            Message::tool_result("c", "IGNORE ME: tool output"),
            Message::from_origin(aidaemon_core::MessageOrigin::Harness, "nudge"),
            Message::assistant("Noted.", vec![]),
        ];
        let t = transcript(&messages, 100, 1_000);
        assert!(t.contains("Mia and Zoe"));
        assert!(t.contains("[used tools: search]"));
        assert!(!t.contains("IGNORE ME"));
        assert!(!t.contains("nudge"));
    }
}
