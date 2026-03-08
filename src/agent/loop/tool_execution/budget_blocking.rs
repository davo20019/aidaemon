use crate::agent::*;
use crate::execution_policy::PolicyBundle;

/// Distinguishes temporary blocks (cooldown) from permanent blocks (budget/limit).
/// Cooldown blocks should NOT trigger force_text_response — the tool will be
/// available again in a few iterations. Permanent blocks mean the tool is done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolBlockKind {
    /// Tool was not blocked; proceed with execution.
    NotBlocked,
    /// Tool is in transient-failure cooldown — temporary, don't kill the task.
    Cooldown,
    /// Tool hit a permanent limit (semantic failures, call count, unknown tool).
    HardBlock,
}

pub(super) struct ToolBudgetBlockCtx<'a> {
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub session_id: &'a str,
    pub iteration: usize,
    pub tool_failure_count: &'a HashMap<String, usize>,
    pub tool_transient_failure_count: &'a HashMap<String, usize>,
    pub tool_cooldown_until_iteration: &'a mut HashMap<String, usize>,
    pub tool_call_count: &'a HashMap<String, usize>,
    pub unknown_tools: &'a HashSet<String>,
}

pub(super) struct DuplicateSendFileNoopCtx<'a> {
    pub send_file_key: Option<&'a String>,
    pub successful_send_file_keys: &'a HashSet<String>,
    pub session_id: &'a str,
    pub iteration: usize,
    pub effective_arguments: &'a str,
    pub force_text_response: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub successful_tool_calls: &'a mut usize,
    pub total_successful_tool_calls: &'a mut usize,
    pub tool_call_count: &'a mut HashMap<String, usize>,
    pub learning_ctx: &'a mut LearningContext,
    pub emitter: &'a crate::events::EventEmitter,
    pub task_id: &'a str,
    pub policy_bundle: &'a PolicyBundle,
}

impl Agent {
    pub(super) async fn maybe_block_tool_by_budget(
        &self,
        tc: &ToolCall,
        ctx: &mut ToolBudgetBlockCtx<'_>,
    ) -> anyhow::Result<ToolBlockKind> {
        // `tool_failure_count` tracks the highest repeated semantic failure signature
        // count for this tool in the current task (not aggregate misses).
        let prior_signature_failures = ctx.tool_failure_count.get(&tc.name).copied().unwrap_or(0);
        let prior_transient_failures = ctx
            .tool_transient_failure_count
            .get(&tc.name)
            .copied()
            .unwrap_or(0);
        let prior_calls = ctx.tool_call_count.get(&tc.name).copied().unwrap_or(0);

        if let Some(until_iteration) = ctx.tool_cooldown_until_iteration.get(&tc.name).copied() {
            if ctx.iteration <= until_iteration {
                let result_text = ToolResultNotice::ToolCooldownBlocked {
                    tool_name: tc.name.clone(),
                    until_iteration,
                }
                .render();
                let tool_msg = Message {
                    id: Uuid::new_v4().to_string(),
                    session_id: ctx.session_id.to_string(),
                    role: "tool".to_string(),
                    content: Some(result_text),
                    tool_call_id: Some(tc.id.clone()),
                    tool_name: Some(tc.name.clone()),
                    tool_calls_json: None,
                    created_at: Utc::now(),
                    importance: 0.1,
                    ..Message::runtime_defaults()
                };
                self.append_tool_message_with_result_event(
                    ctx.emitter,
                    &tool_msg,
                    true,
                    0,
                    None,
                    Some(ctx.task_id),
                )
                .await?;
                return Ok(ToolBlockKind::Cooldown);
            }
            // Cooldown has elapsed; allow attempts again.
            ctx.tool_cooldown_until_iteration.remove(&tc.name);
        }

        // Combined web tool budget: web_search + web_fetch together
        let web_search_calls = ctx.tool_call_count.get("web_search").copied().unwrap_or(0);
        let web_fetch_calls = ctx.tool_call_count.get("web_fetch").copied().unwrap_or(0);
        let combined_web_calls = web_search_calls + web_fetch_calls;

        let failure_limit = semantic_failure_limit(&tc.name);
        let blocked = if ctx.unknown_tools.contains(&tc.name) {
            // Tool doesn't exist — block immediately, no retries.
            Some(
                ToolResultNotice::UnknownToolInvented {
                    tool_name: tc.name.clone(),
                }
                .render(),
            )
        } else if prior_signature_failures >= failure_limit {
            Some(
                ToolResultNotice::SemanticErrorLimitBlocked {
                    tool_name: tc.name.clone(),
                    prior_signature_failures,
                    prior_transient_failures,
                }
                .render(),
            )
        } else if tc.name == "web_search" && prior_calls >= 3 {
            Some(ToolResultNotice::WebSearchBudgetBlocked { prior_calls }.render())
        } else if (tc.name == "web_search" || tc.name == "web_fetch") && combined_web_calls >= 6 {
            Some(ToolResultNotice::CombinedWebBudgetBlocked { combined_web_calls }.render())
        } else if tc.name == "web_fetch" && prior_calls >= 4 {
            Some(ToolResultNotice::WebFetchBudgetBlocked { prior_calls }.render())
        } else if tc.name == "spawn_agent" && prior_calls >= 15 {
            // spawn_agent gets a higher cap than generic tools since task leads
            // legitimately spawn many executors, but it must still be bounded
            // to prevent runaway LLM agent spawns.
            Some(ToolResultNotice::SpawnAgentBudgetBlocked { prior_calls }.render())
        } else if prior_calls >= 8
            && !matches!(
                tc.name.as_str(),
                "terminal"
                    | "cli_agent"
                    | "read_file"
                    | "edit_file"
                    | "write_file"
                    | "search_files"
                    | "remember_fact"
                    | "manage_memories"
                    | "web_fetch"
            )
            && !tc.name.contains("__")
        // MCP tools (prefix__name)
        {
            if tc.name == "web_search" && prior_signature_failures == 0 {
                Some(ToolResultNotice::WebSearchBackendSetupHint { prior_calls }.render())
            } else if tc.name == "project_inspect" {
                Some(ToolResultNotice::ProjectInspectBudgetBlocked { prior_calls }.render())
            } else {
                // terminal is expected to be called many times; others are suspicious
                Some(
                    ToolResultNotice::GenericToolBudgetBlocked {
                        tool_name: tc.name.clone(),
                        prior_calls,
                    }
                    .render(),
                )
            }
        } else {
            None
        };

        let Some(result_text) = blocked else {
            return Ok(ToolBlockKind::NotBlocked);
        };

        warn!(
            tool = %tc.name,
            semantic_failures = prior_signature_failures,
            transient_failures = prior_transient_failures,
            calls = prior_calls,
            "Blocking repeated tool call"
        );
        self.emit_warning_decision_point(
            ctx.emitter,
            ctx.task_id,
            ctx.iteration,
            DecisionType::ToolBudgetBlock,
            format!("Blocked tool {} due to repeated failures/calls", tc.name),
            json!({
                "tool": tc.name,
                "prior_semantic_failures": prior_signature_failures,
                "prior_transient_failures": prior_transient_failures,
                "prior_calls": prior_calls
            }),
        )
        .await;
        let tool_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: ctx.session_id.to_string(),
            role: "tool".to_string(),
            content: Some(result_text),
            tool_call_id: Some(tc.id.clone()),
            tool_name: Some(tc.name.clone()),
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.1,
            ..Message::runtime_defaults()
        };

        self.append_tool_message_with_result_event(
            ctx.emitter,
            &tool_msg,
            true,
            0,
            None,
            Some(ctx.task_id),
        )
        .await?;

        // Do NOT count blocked calls as successful progress.
        // Blocked calls feed into is_productive() which gates budget
        // auto-extension — counting them as "progress" allows budgets
        // to keep extending even when the agent is stuck hitting walls.
        // For stall detection, the iteration counter still advances,
        // which is sufficient to prevent infinite loops.
        Ok(ToolBlockKind::HardBlock)
    }

    pub(super) async fn maybe_handle_duplicate_send_file_noop(
        &self,
        tc: &ToolCall,
        ctx: &mut DuplicateSendFileNoopCtx<'_>,
    ) -> anyhow::Result<bool> {
        if tc.name != "send_file"
            || !ctx
                .send_file_key
                .is_some_and(|k| ctx.successful_send_file_keys.contains(k))
        {
            return Ok(false);
        }

        info!(
            ctx.session_id,
            ctx.iteration,
            tool_call_id = %tc.id,
            "Suppressing duplicate send_file call in same task"
        );
        let result_text =
            "Duplicate send_file suppressed: this exact file+caption was already sent in this task."
                .to_string();

        // A duplicate send_file is a strong signal that the model is stuck in
        // a file-delivery loop. Force the remainder of this task into text-only
        // mode so it closes out instead of re-emitting more file sends.
        *ctx.force_text_response = true;
        ctx.pending_system_messages
            .push(SystemDirective::DuplicateSendFileAlreadySent);

        // Count as a successful no-op so stall detection doesn't
        // treat idempotency suppression as lack of progress.
        *ctx.successful_tool_calls += 1;
        *ctx.total_successful_tool_calls += 1;

        // Track total calls per tool
        *ctx.tool_call_count.entry(tc.name.clone()).or_insert(0) += 1;

        // Track tool call for learning
        let tool_summary = format!(
            "{}({})",
            tc.name,
            summarize_tool_args(&tc.name, ctx.effective_arguments)
        );
        ctx.learning_ctx.tool_calls.push(tool_summary);

        let _ = ctx
            .emitter
            .emit(
                EventType::ToolCall,
                ToolCallData::from_tool_call(
                    tc.id.clone(),
                    tc.name.clone(),
                    serde_json::from_str(ctx.effective_arguments).unwrap_or(serde_json::json!({})),
                    Some(ctx.task_id.to_string()),
                )
                .with_policy_metadata(
                    Some(format!("{}:{}:{}", ctx.task_id, tc.name, tc.id)),
                    Some(ctx.policy_bundle.policy.policy_rev),
                    Some(ctx.policy_bundle.risk_score),
                ),
            )
            .await;

        let tool_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: ctx.session_id.to_string(),
            role: "tool".to_string(),
            content: Some(result_text.clone()),
            tool_call_id: Some(tc.id.clone()),
            tool_name: Some(tc.name.clone()),
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.3,
            ..Message::runtime_defaults()
        };
        self.append_tool_message_with_result_event(
            ctx.emitter,
            &tool_msg,
            true,
            0,
            None,
            Some(ctx.task_id),
        )
        .await?;

        if let Some(ref tid) = self.task_id {
            let activity = TaskActivity {
                id: 0,
                task_id: tid.clone(),
                activity_type: "tool_call".to_string(),
                tool_name: Some(tc.name.clone()),
                tool_args: Some(ctx.effective_arguments.chars().take(1000).collect()),
                result: Some(result_text.chars().take(2000).collect()),
                success: Some(true),
                tokens_used: None,
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            if let Err(e) = self.state.log_task_activity(&activity).await {
                warn!(task_id = %tid, error = %e, "Failed to log task activity");
            }
        }

        Ok(true)
    }
}
