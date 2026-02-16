use super::*;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;

pub(super) enum ToolPreludeOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    Proceed,
}

pub(super) struct ToolPreludeCtx<'a> {
    pub resp: &'a ProviderResponse,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub model: &'a str,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a LearningContext,
    pub user_text: &'a str,
    pub policy_bundle: &'a PolicyBundle,
    pub available_capabilities: &'a HashMap<String, ToolCapabilities>,
}

impl Agent {
    pub(super) async fn run_tool_prelude_phase(
        &self,
        ctx: &ToolPreludeCtx<'_>,
    ) -> anyhow::Result<ToolPreludeOutcome> {
        let resp = ctx.resp;
        let emitter = ctx.emitter;
        let task_id = ctx.task_id;
        let session_id = ctx.session_id;
        let model = ctx.model;
        let iteration = ctx.iteration;
        let task_start = ctx.task_start;
        let learning_ctx = ctx.learning_ctx;
        let user_text = ctx.user_text;
        let policy_bundle = ctx.policy_bundle;
        let available_capabilities = ctx.available_capabilities;

        // Persist assistant message with tool calls
        let assistant_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: resp.content.clone(),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: Some(serde_json::to_string(&resp.tool_calls)?),
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        };
        self.append_assistant_message_with_event(
            emitter,
            &assistant_msg,
            model,
            resp.usage.as_ref().map(|u| u.input_tokens),
            resp.usage.as_ref().map(|u| u.output_tokens),
        )
        .await?;

        // Intent gate: on first iteration, require narration before tool calls.
        // Forces the agent to "show its work" so the user can catch misunderstandings.
        if iteration == 1
            && self.depth == 0
            && !resp.tool_calls.is_empty()
            && resp.content.as_ref().is_none_or(|c| c.trim().len() < 20)
        {
            info!(
                session_id,
                "Intent gate: requiring narration before tool execution"
            );
            for tc in &resp.tool_calls {
                let result_text = "[SYSTEM] Before executing tools, briefly state what you \
                    understand the user is asking and what you plan to do. \
                    Then re-issue the tool calls."
                    .to_string();
                let tool_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "tool".to_string(),
                    content: Some(result_text),
                    tool_call_id: Some(tc.id.clone()),
                    tool_name: Some(tc.name.clone()),
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.3,
                    embedding: None,
                };
                self.append_tool_message_with_result_event(
                    emitter,
                    &tool_msg,
                    true,
                    0,
                    None,
                    Some(task_id),
                )
                .await?;
            }
            return Ok(ToolPreludeOutcome::ContinueLoop);
        }

        let uncertainty_threshold =
            current_uncertainty_threshold(self.policy_config.uncertainty_clarify_threshold);
        if self.policy_config.uncertainty_clarify_enforce
            && policy_bundle.uncertainty_score >= uncertainty_threshold
        {
            let has_side_effecting_call = resp
                .tool_calls
                .iter()
                .any(|tc| tool_is_side_effecting(&tc.name, available_capabilities));
            if has_side_effecting_call {
                let clarify = default_clarifying_question(user_text, &[]);
                POLICY_METRICS
                    .uncertainty_clarify_total
                    .fetch_add(1, Ordering::Relaxed);
                info!(
                    session_id,
                    iteration,
                    uncertainty_score = policy_bundle.uncertainty_score,
                    threshold = uncertainty_threshold,
                    clarification = %clarify,
                    "Uncertainty guard triggered before side-effecting tool execution"
                );
                let assistant_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: "assistant".to_string(),
                    content: Some(clarify.clone()),
                    tool_call_id: None,
                    tool_name: None,
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.5,
                    embedding: None,
                };
                self.append_assistant_message_with_event(
                    emitter,
                    &assistant_msg,
                    "system",
                    None,
                    None,
                )
                .await?;
                self.emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    task_start,
                    iteration,
                    learning_ctx.tool_calls.len(),
                    None,
                    Some("Asked clarification due to uncertainty policy.".to_string()),
                )
                .await;
                return Ok(ToolPreludeOutcome::Return(Ok(clarify)));
            }
        }

        Ok(ToolPreludeOutcome::Proceed)
    }
}
