//! Calling a tool directly from a test (`testing` feature).

use async_trait::async_trait;
use tokio::sync::mpsc;

use serde_json::Value;

use crate::OutcomeTool;
use crate::semantics::{ApprovalResponse, ToolCallContext, ToolCallOutcome};
use crate::status::{StatusUpdate, with_status};

/// Calling a tool directly from a test.
#[async_trait]
pub trait ToolCalls: OutcomeTool {
    /// Run with no call facts and return the output text.
    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self
            .execute(arguments, &ToolCallContext::default())
            .await?
            .output)
    }

    /// Run with a progress channel and return the typed outcome.
    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.execute(
            arguments,
            &with_status(ToolCallContext::default(), status_tx),
        )
        .await
    }

    /// Run with a progress channel and these call facts.
    async fn call_with_execution_context(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        facts: ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.execute(arguments, &with_status(facts, status_tx))
            .await
    }
}

impl<T: OutcomeTool + ?Sized> ToolCalls for T {}

/// Dispatcher facts for a direct tool test: the call's session, nothing else.
pub fn call_facts(session_id: &str) -> ToolCallContext {
    ToolCallContext {
        session_id: session_id.to_string(),
        ..Default::default()
    }
}

/// Call `tool` the way the dispatcher does, beside `facts`, and return its
/// output.
pub async fn call_with_facts<T: OutcomeTool + ?Sized>(
    tool: &T,
    arguments: &str,
    facts: ToolCallContext,
) -> anyhow::Result<String> {
    tool.call_with_execution_context(arguments, None, facts)
        .await
        .map(|outcome| outcome.output)
}

/// What [`call_through_kernel`] did with a call.
pub struct KernelCall {
    /// The tool's output, or the kernel's refusal when the owner did not
    /// approve (the tool did not run).
    pub output: String,
    /// The card the owner was asked with, if the tool asked.
    pub asked: Option<aidaemon_core::OwnerApprovalRequest>,
}

/// Call `tool` as the agent kernel does for a tool that judges its own owner
/// approvals: ask `owner_approval_request` with `facts`; when it asks, hand
/// `answer` (`None`: no answer) back through `record_owner_answer` and run
/// the call only when the owner approved. A decline or no answer never runs
/// it, as with the kernel.
pub async fn call_through_kernel<T: OutcomeTool + ?Sized>(
    tool: &T,
    arguments: &str,
    facts: ToolCallContext,
    answer: Option<ApprovalResponse>,
) -> anyhow::Result<KernelCall> {
    let parsed: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    let call = crate::semantics::ApprovalCall::as_sent(&parsed);
    let asked = match tool.owner_approval_request(&call, &facts).await {
        crate::semantics::OwnerApprovalNeed::Ask(request) => Some(request),
        _ => None,
    };
    if asked.is_some() {
        tool.record_owner_answer(&call, &facts, answer.clone())
            .await;
        match answer {
            Some(ApprovalResponse::Deny) => {
                return Ok(KernelCall {
                    output: "Error: Denied by user.".to_string(),
                    asked,
                });
            }
            None => {
                return Ok(KernelCall {
                    output: "Error: no answer from the owner. Action denied.".to_string(),
                    asked,
                });
            }
            Some(_) => {}
        }
    }
    let output = call_with_facts(tool, arguments, facts).await?;
    Ok(KernelCall { output, asked })
}

/// `tool.call_with(arguments, facts)`: [`call_with_facts`] as a method.
#[async_trait]
pub trait CallWithFacts {
    async fn call_with(&self, arguments: &str, facts: ToolCallContext) -> anyhow::Result<String>;
}

#[async_trait]
impl<T: OutcomeTool + ?Sized> CallWithFacts for T {
    async fn call_with(&self, arguments: &str, facts: ToolCallContext) -> anyhow::Result<String> {
        call_with_facts(self, arguments, facts).await
    }
}

/// One unconfined local backend on the current directory, shared by tests
/// that construct a tool directly.
pub fn active_execution_backend() -> crate::backend::SharedExecutionBackend {
    static BACKEND: std::sync::OnceLock<crate::backend::SharedExecutionBackend> =
        std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| {
            std::sync::Arc::new(crate::local_backend::LocalBackend::current_unrestricted())
        })
        .clone()
}
