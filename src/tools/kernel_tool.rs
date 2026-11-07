//! [`KernelTool`]: a tool written against the SDK (`aidaemon_core::Tool`
//! plus `aidaemon_tools::ToolSemantics`) as a daemon tool.
//!
//! This is how the daemon uses a tool implemented in an SDK crate, so the
//! implementation lives once. The adapter only translates the call: every
//! description of the call (effects, annotations, access, approval, trust,
//! dispatch facts) is the wrapped tool's own, forwarded unchanged.
//!
//! The daemon's call path carries no kernel run, session or cancellation,
//! so the wrapped tool gets a fresh [`aidaemon_core::ToolContext`]. Wrap
//! only tools that do not read the session id or the context's extensions.

use std::sync::Arc;

use aidaemon_tools::semantics::{
    ApprovalCall, DurableReplayDecision, OwnerApprovalNeed, StableObservationSubject,
    ToolArgumentContractViolation, ToolCallAccessManifest, ToolCallContext, ToolCallOutcome,
    ToolCallSemantics, ToolCapabilities, ToolDispatchFacts, ToolReceiptKind, ToolRole,
    ToolSemantics, TrustRoots,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::types::StatusUpdate;

/// An SDK tool offered to the daemon's dispatcher.
pub struct KernelTool<T> {
    inner: Arc<T>,
    name: String,
    description: String,
    schema: Value,
}

impl<T> KernelTool<T>
where
    T: aidaemon_core::Tool + ToolSemantics + 'static,
{
    pub fn new(tool: T) -> Self {
        let spec = aidaemon_core::Tool::spec(&tool);
        Self {
            schema: json!({
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }),
            name: spec.name,
            description: spec.description,
            inner: Arc::new(tool),
        }
    }

    /// Run the wrapped tool. A result it marks as an error is a failed call.
    async fn run(&self, arguments: &str) -> anyhow::Result<ToolCallOutcome> {
        let arguments: Value = if arguments.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(arguments)
                .map_err(|error| anyhow::anyhow!("Invalid arguments: {error}"))?
        };
        let context = aidaemon_core::ToolContext::new("daemon", &self.name);
        let output = aidaemon_core::Tool::call(self.inner.as_ref(), arguments, &context)
            .await
            .map_err(|error| anyhow::anyhow!(error.0))?;
        if output.is_error {
            anyhow::bail!(output.content);
        }
        let mut outcome = ToolCallOutcome::from_output(output.content);
        outcome.finish = output.finish;
        Ok(outcome)
    }
}

#[async_trait]
impl<T> crate::traits::Tool for KernelTool<T>
where
    T: aidaemon_core::Tool + ToolSemantics + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> Value {
        self.schema.clone()
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self.run(arguments).await?.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.run(arguments).await
    }
}

/// Every method forwards, so the wrapped tool's overrides all apply.
#[async_trait]
impl<T> ToolSemantics for KernelTool<T>
where
    T: aidaemon_core::Tool + ToolSemantics + 'static,
{
    fn canonicalize_arguments(
        &self,
        arguments: &str,
    ) -> Result<String, ToolArgumentContractViolation> {
        self.inner.canonicalize_arguments(arguments)
    }

    fn validate_arguments(&self, arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        self.inner.validate_arguments(arguments)
    }

    async fn on_task_end(&self, task_id: &str, session_id: &str) -> anyhow::Result<()> {
        self.inner.on_task_end(task_id, session_id).await
    }

    async fn session_context_note(&self, session_id: &str) -> Option<String> {
        self.inner.session_context_note(session_id).await
    }

    fn tool_role(&self) -> ToolRole {
        self.inner.tool_role()
    }

    fn dispatch_facts(&self) -> ToolDispatchFacts {
        self.inner.dispatch_facts()
    }

    fn capabilities(&self) -> ToolCapabilities {
        self.inner.capabilities()
    }

    fn owner_control_plane(&self) -> bool {
        self.inner.owner_control_plane()
    }

    fn owner_control_plane_call(&self, arguments: &str) -> bool {
        self.inner.owner_control_plane_call(arguments)
    }

    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        self.inner.owner_approval_request(call, facts).await
    }

    fn receipt_kind(&self, arguments: &str) -> ToolReceiptKind {
        self.inner.receipt_kind(arguments)
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        self.inner.call_semantics(arguments)
    }

    fn call_access_manifest(&self, arguments: &str) -> ToolCallAccessManifest {
        self.inner.call_access_manifest(arguments)
    }

    fn adapter_owned_access_manifest(&self, arguments: &str) -> ToolCallAccessManifest {
        self.inner.adapter_owned_access_manifest(arguments)
    }

    async fn durable_replay_decision(&self, arguments: &str) -> DurableReplayDecision {
        self.inner.durable_replay_decision(arguments).await
    }

    fn stable_observation_subjects(&self) -> Vec<StableObservationSubject> {
        self.inner.stable_observation_subjects()
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        self.inner.call_effects(arguments)
    }

    fn call_annotations_in(
        &self,
        arguments: &str,
        roots: &TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        self.inner.call_annotations_in(arguments, roots)
    }

    fn output_trust(&self, arguments: &str, roots: &TrustRoots<'_>) -> aidaemon_core::SourceTrust {
        self.inner.output_trust(arguments, roots)
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Tool as _;

    /// An SDK-style tool: echoes `text`, fails on `fail`, and declares
    /// non-default semantics so forwarding is visible.
    struct Echo;

    #[async_trait]
    impl aidaemon_core::Tool for Echo {
        fn spec(&self) -> aidaemon_core::ToolSpec {
            aidaemon_core::ToolSpec {
                name: "synthetic_echo".into(),
                description: "Echo text back.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "fail": {"type": "boolean"}},
                    "additionalProperties": false
                }),
            }
        }

        async fn call(
            &self,
            arguments: Value,
            _ctx: &aidaemon_core::ToolContext,
        ) -> aidaemon_core::ToolResult {
            let text = arguments["text"].as_str().unwrap_or_default().to_string();
            if arguments["fail"].as_bool() == Some(true) {
                return Ok(aidaemon_core::ToolOutput::error(format!("refused: {text}")));
            }
            Ok(aidaemon_core::ToolOutput::text(text))
        }
    }

    impl ToolSemantics for Echo {
        fn capabilities(&self) -> ToolCapabilities {
            ToolCapabilities {
                read_only: true,
                idempotent: true,
                ..ToolCapabilities::default()
            }
        }

        fn dispatch_facts(&self) -> ToolDispatchFacts {
            ToolDispatchFacts {
                always_offered: true,
                ..ToolDispatchFacts::default()
            }
        }

        fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
            aidaemon_core::CallEffects::none()
        }
    }

    #[tokio::test]
    async fn the_daemon_runs_an_sdk_tool_unchanged() {
        let tool = KernelTool::new(Echo);
        assert_eq!(tool.name(), "synthetic_echo");
        let schema = tool.schema();
        assert_eq!(schema["name"], "synthetic_echo");
        assert_eq!(schema["description"], "Echo text back.");
        assert_eq!(schema["parameters"]["type"], "object");

        assert_eq!(tool.call(r#"{"text":"hello"}"#).await.unwrap(), "hello");
        let failed = tool.call(r#"{"text":"no","fail":true}"#).await.unwrap_err();
        assert_eq!(failed.to_string(), "refused: no");
        assert!(tool.call("not json").await.is_err());
    }

    #[test]
    fn semantics_are_the_wrapped_tools() {
        let tool = KernelTool::new(Echo);
        assert!(tool.capabilities().read_only);
        assert!(tool.dispatch_facts().always_offered);
        assert_eq!(tool.call_effects("{}"), aidaemon_core::CallEffects::none());
    }
}
