use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::agent::Resolution;
use crate::approval::CallEffects;
use crate::approval::SharedDeclines;
use crate::authority::Authority;
use crate::message::Image;
use crate::outcome::FinishOutcome;
use crate::provider::{ToolSpec, Usage};
use crate::state::{RunId, RunState};

/// Who authored the content a tool returned. `Untrusted` is content someone
/// other than the owner wrote (a web page, an inbound message, a third-party
/// file); it may carry instructions aimed at the model. The default is
/// `Untrusted`, so a tool that does not say otherwise taints the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceTrust {
    Trusted,
    #[default]
    Untrusted,
}

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
    /// Sends data or acts outside the owner's machine and workspace, or is
    /// irreversible (send, post, HTTP write, credentialed request, delete,
    /// standing-rule change, payment).
    #[serde(default)]
    pub egress: bool,
}

impl ToolAnnotations {
    pub const READ_ONLY: Self = Self {
        read_only: true,
        destructive: false,
        idempotent: true,
        egress: false,
    };
    pub const WRITE: Self = Self {
        read_only: false,
        destructive: false,
        idempotent: false,
        egress: false,
    };
    pub const DESTRUCTIVE: Self = Self {
        read_only: false,
        destructive: true,
        idempotent: false,
        egress: true,
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
    /// What this call does, from [`Tool::effects`].
    pub effects: CallEffects,
    /// What a read carries out of the machine, from [`Tool::outbound`].
    pub outbound: Outbound,
}

/// What a call carries out of the machine, for a call that only reads. A
/// fetch changes nothing, yet its URL can carry private data to whoever runs
/// the server, so it counts once the run has read untrusted content.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Outbound {
    /// Nothing leaves.
    #[default]
    Nothing,
    /// Exactly these strings leave. A string already present word for word
    /// in what the user or a tool said holds no data the run composed.
    Exact(Vec<String>),
    /// Text the model composed, such as a search query.
    Composed,
}

/// Typed application values a tool can read from its [`ToolContext`],
/// keyed by type: at most one value per type. The application sets them on
/// the agent with [`crate::AgentBuilder::tool_extensions`]; every call's
/// context carries them. Cloning is cheap (values are shared).
#[derive(Clone, Default)]
pub struct Extensions {
    values: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Extensions {
    /// Store `value`, replacing any earlier value of the same type.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.values.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// The value of type `T`, if one was inserted.
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.values
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<T>())
    }
}

impl fmt::Debug for Extensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Extensions")
            .field("len", &self.values.len())
            .finish_non_exhaustive()
    }
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
    /// Application values set with [`crate::AgentBuilder::tool_extensions`].
    pub extensions: Extensions,
    /// The owner's declines in this run, live. A tool that starts another
    /// run shares them with it.
    pub declines: Option<SharedDeclines>,
    /// What this run may do, when a hook enforces it. A tool that starts
    /// another run must not let it do more.
    pub authority: Option<Authority>,
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
            extensions: Extensions::default(),
            declines: None,
            authority: None,
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

    pub fn with_extensions(mut self, extensions: Extensions) -> Self {
        self.extensions = extensions;
        self
    }

    pub fn with_declines(mut self, declines: SharedDeclines) -> Self {
        self.declines = Some(declines);
        self
    }

    pub fn with_authority(mut self, authority: Option<Authority>) -> Self {
        self.authority = authority;
        self
    }
}

/// A tool's request to end the run once its batch completes. The run ends
/// with [`crate::RunStatus::Finished`] and `reply` as its final text, without
/// another model call and without consulting `on_stop` hooks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Finish {
    /// The run's final text, if the tool has one for the user.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply: Option<String>,
    pub outcome: FinishOutcome,
}

impl Finish {
    pub fn new(outcome: FinishOutcome, reply: Option<String>) -> Self {
        Self { reply, outcome }
    }

    /// The work is done; `reply` is the run's final text.
    pub fn succeeded(reply: impl Into<String>) -> Self {
        Self::new(FinishOutcome::Succeeded, Some(reply.into()))
    }

    /// Some of the work is done; `reply` says what is left.
    pub fn partial(reply: impl Into<String>) -> Self {
        Self::new(FinishOutcome::Partial, Some(reply.into()))
    }

    /// The work cannot go on without someone else; `reply` says why.
    pub fn blocked(reply: impl Into<String>) -> Self {
        Self::new(FinishOutcome::Blocked, Some(reply.into()))
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
    /// Overrides [`Tool::source_trust`] for this result (for example a
    /// fetch tool that knows this particular URL is the owner's own).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust: Option<SourceTrust>,
    /// End the run after this call's batch (see [`ToolOutput::finish`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<Finish>,
    /// A paused child run: this call has no result yet, and the run pauses
    /// with it (see [`ToolOutput::suspended`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspended: Option<Box<RunState>>,
}

impl ToolOutput {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            is_error: false,
            details: Value::Null,
            usage: Usage::default(),
            trust: None,
            finish: None,
            suspended: None,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            images: Vec::new(),
            is_error: true,
            details: Value::Null,
            usage: Usage::default(),
            trust: None,
            finish: None,
            suspended: None,
        }
    }

    /// The child this call started is interrupted: pause the calling run
    /// too, with `child` saved in its [`crate::Pending`]. The answer reaches
    /// the tool through [`Tool::resume_child`]. Only a tool whose
    /// [`Tool::suspends`] is true may return this, and `child` must be
    /// interrupted; anything else is a failed call.
    pub fn suspended(child: RunState) -> Self {
        Self {
            suspended: Some(Box::new(child)),
            ..Self::text("")
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

    pub fn with_trust(mut self, trust: SourceTrust) -> Self {
        self.trust = Some(trust);
        self
    }

    /// End the run once the batch this call belongs to completes. Every
    /// call in the batch still runs and gets its receipt; if several calls
    /// finish, the first in call order wins.
    pub fn finish(mut self, finish: Finish) -> Self {
        self.finish = Some(finish);
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
/// Richer per-call semantics (annotations, access, typed effects) are
/// optional methods with conservative defaults, so the simple case stays
/// simple.
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
    /// [`Tool::annotations`] must be the worst-case (most restrictive)
    /// value for the tool; this may only relax it for a specific call. The
    /// result is trusted as given by authority checks (and recorded in the
    /// receipt), so it must be correct for those arguments. Concurrent
    /// scheduling always uses [`Tool::annotations`], never this value.
    /// Crash-recovery replay uses this per-call `idempotent`: a call
    /// declared idempotent for its arguments is re-run, not marked
    /// indeterminate.
    fn annotations_for(&self, _arguments: &Value) -> ToolAnnotations {
        self.annotations()
    }

    /// Who authored what this call returns. The default is `Untrusted`; a
    /// tool whose output is the owner's own (a workspace file, the
    /// kernel's bookkeeping) returns `Trusted`. A result may override this
    /// with [`ToolOutput::with_trust`].
    fn source_trust(&self, _arguments: &Value) -> SourceTrust {
        SourceTrust::Untrusted
    }

    /// What this call carries out of the machine if it only reads. The
    /// default says nothing does.
    fn outbound(&self, _arguments: &Value) -> Outbound {
        Outbound::Nothing
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

    /// The typed effects of this call (effect class and normalized target).
    /// The default is [`CallEffects::none`] for a call that is read-only and
    /// not egress (per [`Tool::annotations_for`]) and [`CallEffects::unknown`]
    /// otherwise: a read from another host still sends the request off the
    /// machine, so a tool annotated `egress` should declare its
    /// [`EffectKey::egress`](crate::EffectKey::egress) target. Recomputed
    /// after a hook rewrites the arguments.
    fn effects(&self, arguments: &Value) -> CallEffects {
        let annotations = self.annotations_for(arguments);
        if annotations.read_only && !annotations.egress {
            CallEffects::none()
        } else {
            CallEffects::unknown()
        }
    }

    /// How many calls of this tool may run together in one batch although
    /// the tool is not read-only: 0 (the default) runs each call alone. A
    /// tool returns more only when it coordinates the shared state its
    /// concurrent calls touch itself (sub-agents whose writes a coordination
    /// ledger arbitrates). Calls queued this way still run after every
    /// earlier call in the batch has been judged, and before any later call
    /// that is not concurrent-safe.
    fn concurrent_peers(&self) -> usize {
        0
    }

    /// Whether this tool may answer with [`ToolOutput::suspended`]. A call
    /// of such a tool always runs alone, because a run pauses on one call at
    /// a time.
    fn suspends(&self) -> bool {
        false
    }

    /// Continue the child run this tool suspended, with the answer to its
    /// interruption, and return what [`Tool::call`] would have returned: a
    /// result, or [`ToolOutput::suspended`] if the child paused again.
    ///
    /// `ctx` is that of the delegating run as it is now, so its authority
    /// and declines are the ones to apply: a resumed child answers to them
    /// exactly as a child started by [`Tool::call`] does. `child` may come
    /// from another process.
    ///
    /// A crash while this runs leaves the delegating call started with no
    /// result, the same as a crash during [`Tool::call`]: on recovery it is
    /// re-run if idempotent and marked indeterminate otherwise.
    async fn resume_child(
        &self,
        _child: RunState,
        _resolution: Resolution,
        _ctx: &ToolContext,
    ) -> ToolResult {
        Err(ToolError(format!(
            "`{}` cannot continue a paused run",
            self.spec().name
        )))
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
    trust: SourceTrust,
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
            trust: SourceTrust::default(),
            func,
            _fut: PhantomData,
        }
    }

    /// Declare the trust class of this tool's output (default `Untrusted`).
    pub fn with_trust(mut self, trust: SourceTrust) -> Self {
        self.trust = trust;
        self
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

    fn source_trust(&self, _arguments: &Value) -> SourceTrust {
        self.trust
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        (self.func)(arguments).await
    }
}

/// The consecutive-call allowance of a tool: how long a run may call it in a
/// row before the same-tool guard treats the streak as a loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SameToolStreak {
    /// The default allowance.
    #[default]
    Standard,
    /// A versatile or research tool whose calls naturally repeat (reading
    /// files, searching, running commands): a longer allowance.
    Versatile,
    /// A step-wise tool driven one small action at a time (computer use):
    /// the longest allowance, and no exemption for distinct arguments.
    Stepwise,
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
    fn default_effects_of_a_read_only_egress_call_are_unknown() {
        let tool = |annotations| {
            FnTool::new(
                ToolSpec {
                    name: "probe".into(),
                    description: "synthetic".into(),
                    parameters: json!({"type": "object"}),
                },
                annotations,
                |_args: Value| async move { Ok(ToolOutput::text("ok")) },
            )
        };
        let local_read = tool(ToolAnnotations::READ_ONLY);
        assert_eq!(local_read.effects(&json!({})), CallEffects::none());
        // Reading from another host still sends the request off the machine.
        let remote_read = tool(ToolAnnotations {
            egress: true,
            ..ToolAnnotations::READ_ONLY
        });
        assert_eq!(remote_read.effects(&json!({})), CallEffects::unknown());
    }

    #[test]
    fn unannotated_tools_are_conservative() {
        let ann = ToolAnnotations::default();
        assert!(!ann.read_only);
        assert!(ann.destructive);
    }
}
