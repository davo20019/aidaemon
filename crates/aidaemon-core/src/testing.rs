//! Test helpers: a scripted provider and small builders for responses.
//! Enabled with the `testing` feature so downstream crates can test their own
//! tools and hooks against the real loop without a model.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::message::ToolCall;
use crate::provider::{ModelRequest, ModelResponse, Provider, ProviderError, ToolSpec, Usage};

/// Returns scripted replies in order and records every request it saw.
#[derive(Default)]
pub struct ScriptedProvider {
    script: Mutex<VecDeque<Result<ModelResponse, ProviderError>>>,
    /// Returned once the script is exhausted, if set.
    repeat: Option<ModelResponse>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl ScriptedProvider {
    pub fn new(replies: impl IntoIterator<Item = ModelResponse>) -> Self {
        Self {
            script: Mutex::new(replies.into_iter().map(Ok).collect()),
            ..Self::default()
        }
    }

    /// Script that may include provider errors.
    pub fn with_results(
        results: impl IntoIterator<Item = Result<ModelResponse, ProviderError>>,
    ) -> Self {
        Self {
            script: Mutex::new(results.into_iter().collect()),
            ..Self::default()
        }
    }

    /// After the script runs out, keep returning `reply` forever.
    pub fn then_repeat(mut self, reply: ModelResponse) -> Self {
        self.repeat = Some(reply);
        self
    }

    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().expect("poisoned").clone()
    }

    pub fn remaining(&self) -> usize {
        self.script.lock().expect("poisoned").len()
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.requests
            .lock()
            .expect("poisoned")
            .push(request.clone());
        let next = self.script.lock().expect("poisoned").pop_front();
        match (next, &self.repeat) {
            (Some(result), _) => result,
            (None, Some(reply)) => Ok(reply.clone()),
            (None, None) => Err(ProviderError::Other(
                "scripted provider: script exhausted".into(),
            )),
        }
    }
}

/// Fixed usage on every scripted reply, so budget tests are predictable.
pub const SCRIPTED_USAGE: Usage = Usage {
    input_tokens: 100,
    output_tokens: 20,
    cached_input_tokens: 0,
    cache_creation_tokens: 0,
};

/// A final text answer.
pub fn text(content: impl Into<String>) -> ModelResponse {
    ModelResponse {
        text: content.into(),
        tool_calls: Vec::new(),
        usage: SCRIPTED_USAGE,
    }
}

/// A reply that requests tool calls.
pub fn tool_calls(calls: impl IntoIterator<Item = ToolCall>) -> ModelResponse {
    ModelResponse {
        text: String::new(),
        tool_calls: calls.into_iter().collect(),
        usage: SCRIPTED_USAGE,
    }
}

pub fn call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall::new(id, name, arguments)
}

/// A tool spec with an object schema and the given required keys.
pub fn spec(name: &str, description: &str, required: &[&str]) -> ToolSpec {
    let properties: serde_json::Map<String, Value> = required
        .iter()
        .map(|k| ((*k).to_string(), serde_json::json!({"type": "string"})))
        .collect();
    ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        }),
    }
}

/// Wraps a provider and streams each reply's text word by word before the
/// final event, to exercise streaming consumers.
pub struct StreamingProvider<P>(pub P);

#[async_trait]
impl<P: Provider> Provider for StreamingProvider<P> {
    fn name(&self) -> &str {
        "scripted-streaming"
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.0.complete(request).await
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> futures::stream::BoxStream<'a, Result<crate::StreamEvent, ProviderError>> {
        use futures::StreamExt;
        let fut = async move {
            match self.0.complete(request).await {
                Ok(response) => {
                    let mut events: Vec<Result<crate::StreamEvent, ProviderError>> = response
                        .text
                        .split_inclusive(' ')
                        .map(|w| Ok(crate::StreamEvent::TextDelta(w.to_string())))
                        .collect();
                    events.push(Ok(crate::StreamEvent::Done(response)));
                    events
                }
                Err(err) => vec![Err(err)],
            }
        };
        Box::pin(futures::stream::once(fut).flat_map(futures::stream::iter))
    }
}

/// A tool context for calling tools directly in tests.
pub fn tool_context() -> crate::ToolContext {
    crate::ToolContext::new("test:synthetic-session", "call-1")
}
