use async_trait::async_trait;

use super::Agent;
use crate::runtime_ports::{
    AgentIngress, AssistantNoteSink, ChannelAgentRuntime, ChildAgentRequest, ChildAgentRun,
    ChildAgentRuntime, ConversationRequest, ConversationRuntime, DaemonAgentRuntime,
    InboundMessageRequest, SalvagedTaskOutcome,
};
use crate::traits::AgentRole;

fn returned_generated_response(
    captured: Vec<crate::events::CapturedGeneratedResponse>,
    _returned_text: &str,
) -> anyhow::Result<crate::events::CapturedGeneratedResponse> {
    captured.into_iter().next_back().ok_or_else(|| {
        anyhow::anyhow!("returned assistant response lacks an identity from its ingress lifecycle")
    })
}

#[async_trait]
impl ChildAgentRuntime for Agent {
    fn depth(&self) -> usize {
        self.depth()
    }

    fn max_depth(&self) -> usize {
        self.max_depth()
    }

    fn role(&self) -> AgentRole {
        self.role()
    }

    fn tool_names(&self) -> Option<Vec<String>> {
        Some(
            self.tools
                .iter()
                .map(|tool| tool.name().to_string())
                .collect(),
        )
    }

    fn specialist_descriptions(&self) -> Vec<(String, String)> {
        self.specialists
            .llm_visible_kinds()
            .into_iter()
            .map(|(name, description)| (name.to_string(), description))
            .collect()
    }

    async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()> {
        Agent::validate_executor_task_for_spawn(self, task_id, expected_goal_id).await
    }

    async fn run_child(&self, request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun> {
        let result = self
            .spawn_child_with_outcome(
                &request.mission,
                &request.task,
                request.status_tx,
                request.channel_ctx,
                request.user_role,
                request.child_role,
                request.goal_id.as_deref(),
                request.task_id.as_deref(),
                request.project_scope.as_deref(),
                request.specialist.as_deref(),
                request.approval_session_id.as_deref(),
                request.parent_authority,
                request.time_limit,
                request.model_tier,
            )
            .await?;
        Ok(ChildAgentRun {
            session: result.session,
            response: result.response,
            outcome: result.outcome,
            trust: result.trust,
        })
    }

    async fn resolve_child_scope(&self, scope: Vec<String>) -> Vec<String> {
        let mut resolved = Vec::with_capacity(scope.len());
        for path in scope {
            resolved.push(
                crate::agent::coordination::canonical_path(&self.execution_backend, &path)
                    .await
                    .unwrap_or(path),
            );
        }
        resolved
    }

    async fn background_cancel_token(
        &self,
        id: &str,
    ) -> Option<tokio_util::sync::CancellationToken> {
        match self.goal_token_registry.as_ref() {
            Some(registry) => Some(registry.register(id).await),
            None => None,
        }
    }

    async fn release_background_cancel_token(&self, id: &str) {
        if let Some(registry) = self.goal_token_registry.as_ref() {
            registry.remove(id).await;
        }
    }

    /// A child answers to the chat its chain started from; its own session
    /// has no chat to deliver to.
    fn background_delivery_session(&self, call_session: &str) -> String {
        self.approval_session_id
            .clone()
            .unwrap_or_else(|| call_session.to_string())
    }

    async fn salvage_executor_task_outcome(
        &self,
        task_id: &str,
        timeout_secs: u64,
    ) -> Option<SalvagedTaskOutcome> {
        Agent::salvage_executor_task_outcome(self, task_id, timeout_secs)
            .await
            .map(|result| SalvagedTaskOutcome {
                status: result.status,
                details: result.details,
            })
    }

    async fn mark_executor_task_timeout(&self, task_id: &str, timeout_secs: u64) {
        Agent::mark_executor_task_timeout(self, task_id, timeout_secs).await;
    }

    async fn deliver_background_child_result(
        &self,
        router: Option<&std::sync::Weak<dyn crate::runtime_ports::OutboundRouter>>,
        parent_session_id: &str,
        text: &str,
        trust: aidaemon_core::SourceTrust,
    ) -> anyhow::Result<bool> {
        Ok(self
            .deliver_background_spawn_result(router, parent_session_id, text, trust)
            .await?
            .sent)
    }
}

#[async_trait]
impl ConversationRuntime for Agent {
    async fn continue_conversation(
        &self,
        request: ConversationRequest,
    ) -> anyhow::Result<crate::runtime_ports::AgentResponseEnvelope> {
        let (text, captured) = crate::events::capture_generated_responses(
            &request.session_id,
            self.handle_internal_continuation(&request),
        )
        .await;
        let returned_text = text?;
        let captured = returned_generated_response(captured, &returned_text)?;
        let text = captured.content;
        let generated = captured.response;
        Ok(crate::runtime_ports::AgentResponseEnvelope {
            response_id: generated.response_id,
            task_id: generated.task_id,
            turn_id: generated.turn_id,
            text,
            referenced_receipts: generated.referenced_receipts,
            disposition: generated.disposition,
        })
    }

    async fn record_continuation_delivery(
        &self,
        session_id: &str,
        delivery: crate::events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        <Self as AgentIngress>::record_response_delivery(self, session_id, delivery).await
    }
}

#[async_trait]
impl AgentIngress for Agent {
    async fn handle_inbound_message(
        &self,
        request: InboundMessageRequest,
    ) -> anyhow::Result<crate::runtime_ports::AgentResponseEnvelope> {
        let (text, captured) = crate::events::capture_generated_responses(
            &request.session_id,
            self.handle_message_with_attachments_and_ingress(
                &request.session_id,
                &request.user_text,
                &request.attachments,
                request.status_tx,
                request.user_role,
                request.channel_ctx,
                request.heartbeat,
                request.ingress_timing,
            ),
        )
        .await;
        let returned_text = text?;
        let captured = returned_generated_response(captured, &returned_text)?;
        let text = captured.content;
        let generated = captured.response;
        Ok(crate::runtime_ports::AgentResponseEnvelope {
            response_id: generated.response_id,
            task_id: generated.task_id,
            turn_id: generated.turn_id,
            text,
            referenced_receipts: generated.referenced_receipts,
            disposition: generated.disposition,
        })
    }

    async fn record_response_delivery(
        &self,
        session_id: &str,
        delivery: crate::events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            delivery.task_id.trim() != "" && delivery.response_id.trim() != "",
            "delivery event requires generated response and task identities"
        );
        let task_id = delivery.task_id.clone();
        let response_id = delivery.response_id.clone();
        crate::events::EventEmitter::new(self.event_store.clone(), session_id.to_string())
            .with_task_id(task_id.clone())
            .emit(crate::events::EventType::ResponseDelivery, delivery)
            .await?;
        if let Ok(aggregate) = self
            .event_store
            .task_run_aggregate(session_id, &task_id)
            .await
        {
            tracing::info!(
                session_id,
                task_id,
                response_id,
                task_kernel_phase = ?aggregate.lifecycle_phase(),
                "Recorded durable response-delivery transition"
            );
        }
        Ok(())
    }
}

#[async_trait]
impl DaemonAgentRuntime for Agent {
    fn is_sensitive_path(&self, path: &std::path::Path) -> bool {
        self.daemon_paths.is_sensitive(path)
    }

    fn execution_backend(&self) -> crate::execution::SharedExecutionBackend {
        self.execution_backend.clone()
    }

    async fn reload_provider(&self, config: &crate::config::AppConfig) -> anyhow::Result<String> {
        Agent::reload_provider(self, config).await
    }
}

#[async_trait]
impl ChannelAgentRuntime for Agent {
    async fn read_mid_run_message(
        &self,
        _session_id: &str,
        text: &str,
        work: &aidaemon_channels::ports::RunningWork,
    ) -> Option<aidaemon_channels::ports::MidRunReading> {
        aidaemon_runtime::mid_run::read_mid_run_message(&self.side_calls, text, work).await
    }

    async fn terminalize_supervised_task(
        &self,
        session_id: &str,
        status: crate::events::TaskStatus,
        error: String,
    ) -> anyhow::Result<Option<String>> {
        self.event_store
            .terminalize_active_task(
                session_id,
                status,
                Some(error),
                Some("Channel supervisor terminated the running task".to_string()),
            )
            .await
    }

    async fn cancel_active_finite_work_for_session(&self, session_id: &str) -> Vec<String> {
        Agent::cancel_active_finite_work_for_session(self, session_id).await
    }

    async fn running_children(
        &self,
        session_id: &str,
    ) -> Vec<aidaemon_channels::ports::RunningChild> {
        self.running_children.for_session(session_id)
    }

    async fn current_model(&self) -> String {
        Agent::current_model(self).await
    }

    async fn context_debug_settings(
        &self,
        session_id: &str,
        model: &str,
    ) -> (bool, usize, usize, usize, Option<i64>) {
        Agent::context_debug_settings(self, session_id, model).await
    }

    async fn set_model(&self, model: String) {
        Agent::set_model(self, model).await;
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        Agent::list_models(self).await
    }

    async fn clear_model_override(&self) {
        Agent::clear_model_override(self).await;
    }

    async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()> {
        Agent::clear_session_context(self, session_id).await
    }

    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        Agent::clear_session(self, session_id).await
    }
}

#[async_trait]
impl AssistantNoteSink for Agent {
    async fn record_assistant_note(&self, session_id: &str, note: &str) -> anyhow::Result<()> {
        self.record_auxiliary_assistant_note(session_id, note).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persisted_response_is_the_authoritative_delivery_envelope() {
        let captured = crate::events::CapturedGeneratedResponse {
            content: "pre-transform response".to_string(),
            response: crate::events::GeneratedResponseRef {
                response_id: "response-synthetic".to_string(),
                task_id: "task-synthetic".to_string(),
                turn_id: Some("turn-synthetic".to_string()),
                referenced_receipts: Vec::new(),
                disposition: crate::events::AssistantResponseDisposition::Terminal,
            },
        };

        let selected = returned_generated_response(
            vec![captured],
            "sanitized user-facing response with different text",
        )
        .expect("causal identity");
        assert_eq!(selected.response.response_id, "response-synthetic");
        assert_eq!(selected.response.task_id, "task-synthetic");
        assert_eq!(selected.content, "pre-transform response");
    }
}
