use std::future::Future;
use std::marker::PhantomData;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::message::Image;
use crate::provider::{ToolSpec, Usage};
use crate::state::RunId;

/// Effect hints that drive scheduling and policy.
///
/// The default is the conservative one: an unannotated tool is treated as
/// mutating and destructive, so it runs alone and policies may ask before it
/// runs. Mirrors MCP tool annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAnnotations {
    /// Never changes anything outside the agent. Read-only calls in the same
    /// model turn run concurrently.
    pub read_only: bool,
    /// May destroy or overwrite data, or act externally (send, deploy, pay).
    pub destructive: bool,
    /// Safe to re-run with the same arguments (matters for resume/replay).
    pub idempotent: bool,
}

impl ToolAnnotations {
    pub const READ_ONLY: Self = Self {
        read_only: true,
        destructive: false,
        idempotent: true,
    };
    pub const WRITE: Self = Self {
        read_only: false,
        destructive: false,
        idempotent: false,
    };
    pub const DESTRUCTIVE: Self = Self {
        read_only: false,
        destructive: true,
        idempotent: false,
    };
}

impl Default for ToolAnnotations {
    fn default() -> Self {
        Self::DESTRUCTIVE
    }
}

/// What one call will touch, declared by the tool from its arguments.
/// Policies use it for scope checks and sandboxes for confinement. Empty
/// means "not declared", not "touches nothing".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessManifest {
    /// Paths read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<String>,
    /// Paths created, modified or deleted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<String>,
    /// Hosts contacted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hosts: Vec<String>,
    /// Commands executed (first word or full command line).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<String>,
}

impl AccessManifest {
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty()
            && self.writes.is_empty()
            && self.hosts.is_empty()
            && self.commands.is_empty()
    }
}

/// What `before_tool` hooks know about the tool behind a call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInfo {
    pub annotations: ToolAnnotations,
    pub access: AccessManifest,
}

/// Trusted, harness-provided context for one call.
///
/// This is passed beside the model's arguments rather than injected into
/// them, so a model can never spoof a session id or role by inventing a field.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolContext {
    pub run_id: RunId,
    pub session_id: String,
    pub call_id: String,
    pub turn: u32,
    /// Cancelled when the run is cancelled; long tools should watch it.
    pub cancel: CancellationToken,
}

impl ToolContext {
    /// A context with a fresh run id, turn 0 and a new cancel token. The
    /// loop fills in the real values; tests and adapters use this directly.
    pub fn new(session_id: impl Into<String>, call_id: impl Into<String>) -> Self {
        Self {
            run_id: RunId::new(),
            session_id: session_id.into(),
            call_id: call_id.into(),
            turn: 0,
            cancel: CancellationToken::new(),
        }
    }

    pub fn with_run(mut self, run_id: RunId, turn: u32) -> Self {
        self.run_id = run_id;
        self.turn = turn;
        self
    }

    pub fn with_cancel(mut self, cancel: CancellationToken) -> Self {
        self.cancel = cancel;
        self
    }
}

/// What a tool hands back to the model, plus structured details for receipts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text the model will read.
    pub content: String,
    /// Images the model will see with the text (a screenshot, a rendered
    /// page, an image file the tool read).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
    /// The tool ran but the outcome is a failure the model should react to
    /// (non-zero exit, HTTP 4xx, no match). Recorded as a failed receipt.
    pub is_error: bool,
    /// Structured evidence copied into the receipt (exit code, paths, status).
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub details: Value,
    /// Model usage spent inside the tool (for example a sub-agent). The loop
    /// adds it to the run's usage, so budgets cover delegated work.
    #[serde(default)]
    pub usage: Usage,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            is_error: false,
            details: Value::Null,
            usage: Usage::default(),
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            is_error: true,
            details: Value::Null,
            usage: Usage::default(),
        }
    }

    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    pub fn with_images(mut self, images: impl IntoIterator<Item = Image>) -> Self {
        self.images.extend(images);
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }
}

/// A tool failure the harness could not turn into normal output. The loop
/// still reports it to the model as an error result; it never ends the run.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(pub String);

impl From<String> for ToolError {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ToolError {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<std::io::Error> for ToolError {
    fn from(value: std::io::Error) -> Self {
        Self(value.to_string())
    }
}

impl From<serde_json::Error> for ToolError {
    fn from(value: serde_json::Error) -> Self {
        Self(format!("invalid arguments: {value}"))
    }
}

pub type ToolResult = Result<ToolOutput, ToolError>;

/// An LLM-callable capability. Only `spec` and `call` are required.
///
/// Richer semantics (typed effects, access manifests, replay decisions like
/// aidaemon's current `Tool` trait) belong in a separate opt-in trait so the
/// simple case stays simple.
#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default()
    }

    /// Annotations for one call. Tools whose effect depends on the arguments
    /// (a shell running `ls` versus `rm`) override this; the default is the
    /// tool-wide [`Tool::annotations`].
    ///
    /// [`Tool::annotations`] should be the worst-case (most restrictive)
    /// value for the tool; this may only relax it for a specific call. The
    /// result is trusted as given by authority checks (and recorded in the
    /// receipt), so it must be correct for those arguments. Concurrent
    /// scheduling always uses [`Tool::annotations`], never this value.
    fn annotations_for(&self, _arguments: &Value) -> ToolAnnotations {
        self.annotations()
    }

    /// Deterministic argument checks run before any policy or I/O. The
    /// default checks the schema's `required` keys; override to add more.
    fn validate(&self, arguments: &Value) -> Result<(), String> {
        check_required(&self.spec().parameters, arguments)
    }

    /// How long this tool may run, overriding `Limits::tool_timeout`. Tools
    /// that take their own timeout argument (such as a shell) return a value
    /// at least as long as their own maximum.
    fn timeout(&self) -> Option<std::time::Duration> {
        None
    }

    /// Declare what this call will touch. Optional; policies treat an empty
    /// manifest as undeclared.
    fn access(&self, _arguments: &Value) -> AccessManifest {
        AccessManifest::default()
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult;
}

/// Minimal schema check: the arguments must be an object containing every
/// `required` key. Full JSON Schema validation is left to tools that need it.
pub(crate) fn check_required(schema: &Value, arguments: &Value) -> Result<(), String> {
    let wants_object = schema.get("type").and_then(Value::as_str) == Some("object");
    if !wants_object {
        return Ok(());
    }
    let Some(object) = arguments.as_object() else {
        return Err("arguments must be a JSON object".to_string());
    };
    let missing: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|key| !object.contains_key(*key))
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "missing required argument(s): {}",
            missing.join(", ")
        ))
    }
}

/// Build a tool from a closure. Handy for tests and small integrations.
///
/// ```
/// use aidaemon_core::{FnTool, ToolAnnotations, ToolOutput, ToolSpec};
/// use serde_json::{json, Value};
///
/// let spec = ToolSpec {
///     name: "echo".into(),
///     description: "Echo the text back".into(),
///     parameters: json!({"type": "object", "required": ["text"]}),
/// };
/// let echo = FnTool::new(spec, ToolAnnotations::READ_ONLY, |args: Value| async move {
///     Ok(ToolOutput::text(args["text"].as_str().unwrap_or_default().to_string()))
/// });
/// # let _ = echo;
/// ```
pub struct FnTool<F, Fut> {
    spec: ToolSpec,
    annotations: ToolAnnotations,
    func: F,
    _fut: PhantomData<fn() -> Fut>,
}

impl<F, Fut> FnTool<F, Fut>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = ToolResult> + Send,
{
    pub fn new(spec: ToolSpec, annotations: ToolAnnotations, func: F) -> Self {
        Self {
            spec,
            annotations,
            func,
            _fut: PhantomData,
        }
    }
}

#[async_trait]
impl<F, Fut> Tool for FnTool<F, Fut>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = ToolResult> + Send,
{
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn annotations(&self) -> ToolAnnotations {
        self.annotations
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        (self.func)(arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn required_keys_are_checked() {
        let schema = json!({"type": "object", "required": ["path", "text"]});
        assert!(check_required(&schema, &json!({"path": "a", "text": "b"})).is_ok());
        let err = check_required(&schema, &json!({"path": "a"})).unwrap_err();
        assert!(err.contains("text"), "{err}");
        assert!(check_required(&schema, &json!("not an object")).is_err());
    }

    #[test]
    fn unannotated_tools_are_conservative() {
        let ann = ToolAnnotations::default();
        assert!(!ann.read_only);
        assert!(ann.destructive);
    }
}
