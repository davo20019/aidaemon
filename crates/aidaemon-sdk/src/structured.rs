//! Typed structured output: a model call, or an agent's final answer,
//! returned as your own type.
//!
//! The request carries [`ResponseFormat::JsonSchema`] with the schema of `T`
//! (derived with `schemars`), and the instructions repeat it, for providers
//! that do not enforce schemas. A reply that does not parse as `T` gets
//! exactly one repair attempt: the parse error goes back to the model, which
//! answers again. A second failure is [`StructuredError::Invalid`].
//!
//! ```
//! # #[cfg(feature = "testing")]
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! use aidaemon_sdk::structured::{complete_typed, AgentStructuredExt};
//! use aidaemon_sdk::testing::{text, ScriptedProvider};
//! use aidaemon_sdk::{Agent, Message, ModelRequest};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// How a support ticket should be handled.
//! #[derive(Deserialize, JsonSchema)]
//! struct Triage {
//!     priority: u8,
//!     refund: bool,
//! }
//!
//! // One model call.
//! let provider = ScriptedProvider::new([text(r#"{"priority": 2, "refund": true}"#)]);
//! let request = ModelRequest {
//!     model: "scripted".into(),
//!     messages: vec![Message::user("Cracked jar on order ORD-1001.")],
//!     ..Default::default()
//! };
//! let triage = complete_typed::<Triage>(&provider, request).await?;
//! assert!(triage.value.refund);
//!
//! // An agent's final answer.
//! let agent = Agent::builder("triage")
//!     .model("scripted")
//!     .provider(ScriptedProvider::new([text(r#"{"priority": 1, "refund": false}"#)]))
//!     .build()?;
//! let typed = agent.run_typed::<Triage>("support:synthetic-1", "Where is my order?").await?;
//! assert_eq!(typed.value.priority, 1);
//! # Ok(())
//! # }
//! ```

use std::future::Future;

use aidaemon_core::{
    Agent, CancellationToken, CoreError, Hook, HookContext, Message, ModelRequest, ModelResponse,
    Provider, ProviderError, Resolution, ResponseFormat, Role, RunOutcome, RunState, RunStatus,
    Usage,
};
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// A model call's typed answer.
#[derive(Debug, Clone)]
pub struct Typed<T> {
    pub value: T,
    /// The reply the value was read from.
    pub response: ModelResponse,
    /// Usage of every call made, the repair included.
    pub usage: Usage,
    /// The first reply did not parse and the repair attempt did.
    pub repaired: bool,
}

/// An agent run's typed final answer.
#[derive(Debug, Clone)]
pub struct TypedOutcome<T> {
    pub value: T,
    /// The run that produced it (after the repair, if one was needed).
    pub outcome: RunOutcome,
    pub repaired: bool,
}

/// Why no typed value came back.
#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Core(#[from] CoreError),
    /// The run ended without a final answer (interrupted for an approval,
    /// a limit, a cancel). Settle an interruption with
    /// [`AgentStructuredExt::resume_typed`]: it resumes with the same
    /// structured-output hook the run was started with, which a plain
    /// [`Agent::resume`] lacks.
    #[error("the run ended without a final answer: {:?}", .0.status)]
    NotFinished(Box<RunOutcome>),
    /// The answer still did not parse after the one repair attempt.
    /// `outcome` is the run's, for an agent.
    #[error("the answer is not valid {type_name}: {error}")]
    Invalid {
        type_name: &'static str,
        error: String,
        text: String,
        outcome: Option<Box<RunOutcome>>,
    },
}

/// Read `text` as `T`: the JSON value itself, or the first fenced or
/// braced JSON in it (models sometimes wrap the answer in prose or a code
/// fence despite being told not to).
pub fn parse_answer<T: DeserializeOwned>(text: &str) -> Result<T, String> {
    let trimmed = text.trim();
    let first = match serde_json::from_str::<T>(trimmed) {
        Ok(value) => return Ok(value),
        Err(err) => err.to_string(),
    };
    for candidate in [fenced(trimmed), braced(trimmed)].into_iter().flatten() {
        if let Ok(value) = serde_json::from_str::<T>(candidate) {
            return Ok(value);
        }
    }
    Err(first)
}

fn fenced(text: &str) -> Option<&str> {
    let start = text.find("```")?;
    let rest = &text[start + 3..];
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let end = rest.find("```")?;
    Some(rest[..end].trim())
}

fn braced(text: &str) -> Option<&str> {
    let (open, close) = match (text.find('{'), text.find('[')) {
        (Some(o), Some(a)) if a < o => ('[', ']'),
        (Some(_), _) => ('{', '}'),
        (None, Some(_)) => ('[', ']'),
        (None, None) => return None,
    };
    let start = text.find(open)?;
    let end = text.rfind(close)?;
    (end > start).then(|| &text[start..=end])
}

fn format_name<T: ?Sized>() -> String {
    let full = std::any::type_name::<T>();
    let base = full.split('<').next().unwrap_or(full);
    let short = base.rsplit("::").next().unwrap_or(base);
    let name: String = short
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(64)
        .collect();
    if name.is_empty() {
        "answer".into()
    } else {
        name
    }
}

/// The instruction added to the system prompt, for providers that do not
/// enforce [`ResponseFormat::JsonSchema`] themselves.
fn schema_instruction(schema: &Value) -> String {
    format!(
        "When you give your final answer, reply with only a JSON value matching this JSON \
         Schema, with no prose and no code fence:\n{schema}"
    )
}

fn repair_message(error: &str) -> String {
    format!(
        "Your answer could not be read: {error}. Reply again with only the JSON value matching \
         the schema, with no prose and no code fence."
    )
}

fn add_instruction(messages: &mut Vec<Message>, instruction: &str) {
    match messages.first_mut() {
        Some(first) if first.role == Role::System => {
            if !first.content.contains(instruction) {
                if !first.content.is_empty() {
                    first.content.push_str("\n\n");
                }
                first.content.push_str(instruction);
            }
        }
        _ => messages.insert(0, Message::system(instruction)),
    }
}

/// Make one model call whose answer is a `T`, with one repair attempt.
/// `request.options.response_format` is set to `T`'s schema (strict).
pub async fn complete_typed<T>(
    provider: &dyn Provider,
    mut request: ModelRequest,
) -> Result<Typed<T>, StructuredError>
where
    T: DeserializeOwned + JsonSchema,
{
    let schema = crate::schema::schema_for::<T>();
    add_instruction(&mut request.messages, &schema_instruction(&schema));
    request.options.response_format = ResponseFormat::JsonSchema {
        name: format_name::<T>(),
        schema,
        strict: true,
    };

    let mut usage = Usage::default();
    let first = provider.complete(&request).await?;
    usage.add(first.usage);
    let error = match parse_answer::<T>(&first.text) {
        Ok(value) => {
            return Ok(Typed {
                value,
                response: first,
                usage,
                repaired: false,
            });
        }
        Err(error) => error,
    };

    request
        .messages
        .push(Message::assistant(first.text.clone(), Vec::new()));
    request.messages.push(Message::user(repair_message(&error)));
    let second = provider.complete(&request).await?;
    usage.add(second.usage);
    match parse_answer::<T>(&second.text) {
        Ok(value) => Ok(Typed {
            value,
            response: second,
            usage,
            repaired: true,
        }),
        Err(error) => Err(StructuredError::Invalid {
            type_name: std::any::type_name::<T>(),
            error,
            text: second.text,
            outcome: None,
        }),
    }
}

/// Puts the answer's schema on every model call of a run: the response
/// format, and the instruction in the system prompt.
struct OutputSchema {
    name: String,
    schema: Value,
    instruction: String,
}

#[async_trait]
impl Hook for OutputSchema {
    fn name(&self) -> &'static str {
        "structured-output"
    }

    async fn before_model(&self, _ctx: &HookContext<'_>, request: &mut ModelRequest) {
        add_instruction(&mut request.messages, &self.instruction);
        request.options.response_format = ResponseFormat::JsonSchema {
            name: self.name.clone(),
            schema: self.schema.clone(),
            strict: true,
        };
    }
}

fn typed_agent<T: JsonSchema>(agent: &Agent) -> Agent {
    let schema = crate::schema::schema_for::<T>();
    agent.with_hook(OutputSchema {
        name: format_name::<T>(),
        instruction: schema_instruction(&schema),
        schema,
    })
}

/// Final answers as your own type. Implemented for [`Agent`].
pub trait AgentStructuredExt {
    /// [`Agent::run`], with the final answer read as `T` (one repair
    /// attempt). Tools still run as usual; only the final answer must match.
    fn run_typed<T>(
        &self,
        session_id: &str,
        input: impl Into<Message> + Send,
    ) -> impl Future<Output = Result<TypedOutcome<T>, StructuredError>> + Send
    where
        T: DeserializeOwned + JsonSchema + Send;

    /// [`Agent::run_from`] (a saved, not interrupted state), typed.
    fn run_typed_from<T>(
        &self,
        state: RunState,
        cancel: CancellationToken,
    ) -> impl Future<Output = Result<TypedOutcome<T>, StructuredError>> + Send
    where
        T: DeserializeOwned + JsonSchema + Send;

    /// [`Agent::resume`] for a run started with
    /// [`AgentStructuredExt::run_typed`], typed. May end
    /// [`StructuredError::NotFinished`] again at the next interruption.
    fn resume_typed<T>(
        &self,
        state: RunState,
        resolution: Resolution,
    ) -> impl Future<Output = Result<TypedOutcome<T>, StructuredError>> + Send
    where
        T: DeserializeOwned + JsonSchema + Send;

    /// Read the final answer of an outcome you already have (for example
    /// after resuming an interrupted run) as `T`, with one repair attempt.
    fn finish_typed<T>(
        &self,
        outcome: RunOutcome,
    ) -> impl Future<Output = Result<TypedOutcome<T>, StructuredError>> + Send
    where
        T: DeserializeOwned + JsonSchema + Send;
}

impl AgentStructuredExt for Agent {
    async fn run_typed<T>(
        &self,
        session_id: &str,
        input: impl Into<Message> + Send,
    ) -> Result<TypedOutcome<T>, StructuredError>
    where
        T: DeserializeOwned + JsonSchema + Send,
    {
        let typed = typed_agent::<T>(self);
        let outcome = typed.run(session_id, input).await?;
        finish::<T>(&typed, outcome, CancellationToken::new()).await
    }

    async fn run_typed_from<T>(
        &self,
        state: RunState,
        cancel: CancellationToken,
    ) -> Result<TypedOutcome<T>, StructuredError>
    where
        T: DeserializeOwned + JsonSchema + Send,
    {
        let typed = typed_agent::<T>(self);
        let outcome = typed.run_from(state, cancel.clone()).await?;
        finish::<T>(&typed, outcome, cancel).await
    }

    async fn resume_typed<T>(
        &self,
        state: RunState,
        resolution: Resolution,
    ) -> Result<TypedOutcome<T>, StructuredError>
    where
        T: DeserializeOwned + JsonSchema + Send,
    {
        let typed = typed_agent::<T>(self);
        let outcome = typed.resume(state, resolution).await?;
        finish::<T>(&typed, outcome, CancellationToken::new()).await
    }

    async fn finish_typed<T>(&self, outcome: RunOutcome) -> Result<TypedOutcome<T>, StructuredError>
    where
        T: DeserializeOwned + JsonSchema + Send,
    {
        finish::<T>(&typed_agent::<T>(self), outcome, CancellationToken::new()).await
    }
}

fn final_answer(outcome: &RunOutcome) -> Option<&str> {
    match outcome.status {
        RunStatus::Completed | RunStatus::Finished { .. } => outcome.final_text.as_deref(),
        _ => None,
    }
}

async fn finish<T>(
    agent: &Agent,
    outcome: RunOutcome,
    cancel: CancellationToken,
) -> Result<TypedOutcome<T>, StructuredError>
where
    T: DeserializeOwned + JsonSchema,
{
    let Some(text) = final_answer(&outcome) else {
        return Err(StructuredError::NotFinished(Box::new(outcome)));
    };
    let error = match parse_answer::<T>(text) {
        Ok(value) => {
            return Ok(TypedOutcome {
                value,
                outcome,
                repaired: false,
            });
        }
        Err(error) => error,
    };

    let state = outcome.state.with_user_message(repair_message(&error));
    let outcome = agent.run_from(state, cancel).await?;
    let Some(text) = final_answer(&outcome) else {
        return Err(StructuredError::NotFinished(Box::new(outcome)));
    };
    match parse_answer::<T>(text) {
        Ok(value) => Ok(TypedOutcome {
            value,
            outcome,
            repaired: true,
        }),
        Err(error) => Err(StructuredError::Invalid {
            type_name: std::any::type_name::<T>(),
            error,
            text: text.to_string(),
            outcome: Some(Box::new(outcome)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Point {
        x: i32,
    }

    #[test]
    fn answers_are_read_bare_fenced_or_in_prose() {
        assert_eq!(parse_answer::<Point>(r#" {"x": 1} "#), Ok(Point { x: 1 }));
        assert_eq!(
            parse_answer::<Point>("```json\n{\"x\": 2}\n```"),
            Ok(Point { x: 2 })
        );
        assert_eq!(
            parse_answer::<Point>("Here you go: {\"x\": 3}. Anything else?"),
            Ok(Point { x: 3 })
        );
        let err = parse_answer::<Point>(r#"{"y": 1}"#).unwrap_err();
        assert!(err.contains("missing field `x`"), "{err}");
    }

    #[test]
    fn format_names_are_short_and_valid() {
        assert_eq!(format_name::<Point>(), "Point");
        assert_eq!(format_name::<Vec<Point>>(), "Vec");
    }
}
