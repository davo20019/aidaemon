//! Calling a tool directly from a test (`testing` feature).

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::semantics::{ToolCallContext, ToolCallOutcome};
use crate::status::{with_status, StatusUpdate};
use crate::OutcomeTool;

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

/// Call facts for `session_id` and nothing else.
pub fn call_facts(session_id: &str) -> ToolCallContext {
    ToolCallContext {
        session_id: session_id.to_string(),
        ..Default::default()
    }
}
