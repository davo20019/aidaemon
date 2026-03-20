use super::execution_state::LinearIntentStep;
use super::*;
use crate::execution_policy::{PolicyBundle, VerifyLevel};
use crate::traits::ProviderResponse;
use serde::{Deserialize, Serialize};

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
    pub llm_provider: Arc<dyn ModelProvider>,
    pub iteration: usize,
    pub task_start: Instant,
    pub learning_ctx: &'a mut LearningContext,
    pub evidence_state: &'a EvidenceState,
    pub user_text: &'a str,
    pub policy_bundle: &'a PolicyBundle,
    pub available_capabilities: &'a HashMap<String, ToolCapabilities>,
    pub execution_state: &'a mut ExecutionState,
    pub validation_state: &'a mut ValidationState,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub force_text_response: &'a mut bool,
    pub turn_context: &'a TurnContext,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct PlannedAction {
    pub(crate) tool: String,
    pub(crate) target: String,
    pub(crate) description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct PlanState {
    pub(crate) goal: String,
    pub(crate) success_criteria: Vec<String>,
    pub(crate) first_action: PlannedAction,
    pub(crate) requires_verification: bool,
    pub(crate) risky_actions: Vec<String>,
    pub(crate) version: u32,
    #[serde(default)]
    pub(crate) planned_steps: Vec<PlannedAction>,
}

impl PlanState {
    fn normalize(mut self) -> Self {
        self.goal = self.goal.trim().to_string();
        self.success_criteria = self
            .success_criteria
            .into_iter()
            .map(|criterion| criterion.trim().to_string())
            .filter(|criterion| !criterion.is_empty())
            .collect();
        self.first_action.tool = self.first_action.tool.trim().to_string();
        self.first_action.target = self.first_action.target.trim().to_string();
        self.first_action.description = self.first_action.description.trim().to_string();
        self.risky_actions = self
            .risky_actions
            .into_iter()
            .map(|action| action.trim().to_string())
            .filter(|action| !action.is_empty())
            .collect();
        self.planned_steps = self
            .planned_steps
            .into_iter()
            .map(|mut step| {
                step.tool = step.tool.trim().to_string();
                step.target = step.target.trim().to_string();
                step.description = step.description.trim().to_string();
                step
            })
            .filter(|step| !step.tool.is_empty())
            .collect();
        if self.version == 0 {
            self.version = 1;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CritiqueVerdict {
    Accept,
    Replan,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CritiqueState {
    verdict: CritiqueVerdict,
    issues: Vec<String>,
    summary: String,
}

impl CritiqueState {
    fn normalize(mut self) -> Self {
        self.issues = self
            .issues
            .into_iter()
            .map(|issue| issue.trim().to_string())
            .filter(|issue| !issue.is_empty())
            .collect();
        self.summary = self.summary.trim().to_string();
        self
    }
}

fn pre_execution_plan_schema_json() -> Value {
    json!({
        "type": "object",
        "properties": {
            "goal": { "type": "string" },
            "success_criteria": {
                "type": "array",
                "items": { "type": "string" }
            },
            "first_action": {
                "type": "object",
                "properties": {
                    "tool": { "type": "string" },
                    "target": { "type": "string" },
                    "description": { "type": "string" }
                },
                "required": ["tool", "target", "description"],
                "additionalProperties": false
            },
            "requires_verification": { "type": "boolean" },
            "risky_actions": {
                "type": "array",
                "items": { "type": "string" }
            },
            "version": { "type": "integer", "minimum": 1 },
            "planned_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": { "type": "string" },
                        "target": { "type": "string" },
                        "description": { "type": "string" }
                    },
                    "required": ["tool", "target", "description"],
                    "additionalProperties": false
                }
            }
        },
        "required": [
            "goal",
            "success_criteria",
            "first_action",
            "requires_verification",
            "risky_actions",
            "version"
        ],
        "additionalProperties": false
    })
}

fn pre_execution_critique_schema_json() -> Value {
    json!({
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["accept", "replan"]
            },
            "issues": {
                "type": "array",
                "items": { "type": "string" }
            },
            "summary": { "type": "string" }
        },
        "required": ["verdict", "issues", "summary"],
        "additionalProperties": false
    })
}

fn tool_call_is_side_effecting(
    agent: &Agent,
    tc: &ToolCall,
    available_capabilities: &HashMap<String, ToolCapabilities>,
) -> bool {
    let semantics = agent
        .tools
        .iter()
        .find(|tool| tool.name() == tc.name && tool.is_available())
        .map(|tool| tool.call_semantics(&tc.arguments))
        .unwrap_or_default();

    if !semantics.is_empty() {
        semantics.mutates_state()
    } else {
        tool_is_side_effecting(&tc.name, available_capabilities)
    }
}

fn first_side_effecting_tool_call<'a>(
    agent: &Agent,
    resp: &'a ProviderResponse,
    available_capabilities: &HashMap<String, ToolCapabilities>,
) -> Option<&'a ToolCall> {
    resp.tool_calls
        .iter()
        .find(|tc| tool_call_is_side_effecting(agent, tc, available_capabilities))
}

fn extract_target_preview(arguments: &str) -> Option<String> {
    let parsed = serde_json::from_str::<Value>(arguments).ok()?;
    let Value::Object(map) = parsed else {
        return None;
    };

    for key in [
        "path",
        "file_path",
        "file",
        "filename",
        "url",
        "target",
        "target_path",
        "project_path",
        "project_dir",
        "repo_path",
        "repo_dir",
        "working_dir",
        "directory",
        "dir",
    ] {
        if let Some(value) = map.get(key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    None
}

fn turn_prefers_plain_text_completion(turn_context: &TurnContext) -> bool {
    // ConnectedContentMode::DraftOnly is deliberately excluded here.
    // Keyword-based "authoring only" classification is too brittle —
    // "create 3 blog posts in ~/projects/X and commit" gets misclassified
    // as DraftOnly because it matches "create" + "posts".  The LLM is
    // better at deciding whether tools are needed; hard-blocking them
    // based on keyword heuristics causes false tool disablement.
    // The DraftOnly signal is still used downstream for budget/contract
    // hints, just not for hard tool blocking.
    !turn_context.completion_contract.expects_mutation
        && !turn_context.completion_contract.requires_observation
}

fn summarize_tool_arguments(arguments: &str) -> Value {
    serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({ "raw": arguments }))
}

fn summarize_evidence_state(evidence_state: &EvidenceState) -> Value {
    json!({
        "target": evidence_state.target,
        "record_count": evidence_state.records.len(),
        "records": evidence_state.records.iter().map(|record| {
            json!({
                "kind": record.kind,
                "source": record.source,
                "trust": record.trust,
                "observed_at": record.observed_at,
                "targets": record.targets,
            })
        }).collect::<Vec<_>>(),
        "contradictions": evidence_state.contradictions,
        "post_change_verification_done": evidence_state.post_change_verification_done,
    })
}

fn validate_pre_execution_plan(
    plan: &PlanState,
    tool_call: &ToolCall,
    expected_target: Option<&str>,
) -> Result<(), &'static str> {
    if plan.goal.is_empty() {
        return Err("missing_goal");
    }
    if plan.success_criteria.is_empty() {
        return Err("missing_success_criteria");
    }
    if plan.first_action.tool.is_empty() {
        return Err("missing_first_action_tool");
    }
    if !plan.first_action.tool.eq_ignore_ascii_case(&tool_call.name) {
        return Err("first_action_tool_mismatch");
    }
    if plan.first_action.description.is_empty() {
        return Err("missing_first_action_description");
    }
    if expected_target.is_some() && plan.first_action.target.is_empty() {
        return Err("missing_first_action_target");
    }
    if !plan.requires_verification {
        return Err("missing_verification_requirement");
    }
    if plan.risky_actions.is_empty() {
        return Err("missing_risk_acknowledgment");
    }
    if plan.version == 0 {
        return Err("missing_plan_version");
    }
    if !plan.planned_steps.is_empty() {
        // First planned step must match first_action (tool + target when present)
        let first_step = &plan.planned_steps[0];
        if !first_step
            .tool
            .eq_ignore_ascii_case(&plan.first_action.tool)
        {
            return Err("planned_steps_first_action_mismatch");
        }
        if !plan.first_action.target.is_empty()
            && (first_step.target.is_empty()
                || !first_step
                    .target
                    .eq_ignore_ascii_case(&plan.first_action.target))
        {
            return Err("planned_steps_first_action_target_mismatch");
        }
        // Every planned step must have tool and description
        for step in &plan.planned_steps {
            if step.tool.is_empty() || step.description.is_empty() {
                return Err("planned_steps_incomplete_entry");
            }
        }
    }
    Ok(())
}

fn validate_pre_execution_critique(critique: &CritiqueState) -> Result<(), &'static str> {
    if critique.summary.is_empty() {
        return Err("missing_summary");
    }
    if matches!(critique.verdict, CritiqueVerdict::Replan) && critique.issues.is_empty() {
        return Err("missing_issues_for_replan");
    }
    Ok(())
}

fn critique_budget_available(execution_state: &ExecutionState) -> bool {
    (execution_state.budget.max_llm_calls == 0
        || execution_state.llm_calls_used < execution_state.budget.max_llm_calls)
        && (execution_state.budget.max_validation_rounds == 0
            || execution_state.validation_rounds_used
                < execution_state.budget.max_validation_rounds)
}

fn should_run_pre_execution_critique(
    policy_bundle: &PolicyBundle,
    capabilities: ToolCapabilities,
    execution_state: &ExecutionState,
) -> bool {
    if !critique_budget_available(execution_state) {
        return false;
    }

    // Critique is expensive (~2 extra LLM calls, 1-3 min).  Reserve it for
    // genuinely high-risk operations.  `needs_approval` alone should NOT
    // trigger critique — the interactive approval flow already gates those
    // tools separately.  Routine file writes (write_file, edit_file) were
    // being critiqued on every attempt, causing 7+ min delays for simple tasks.
    capabilities.high_impact_write
        || capabilities.external_side_effect
        || matches!(policy_bundle.policy.verify_level, VerifyLevel::Full)
        || policy_bundle.risk_score >= 0.67
        || policy_bundle.uncertainty_score >= 0.45
}

impl Agent {
    async fn inject_prelude_retry_messages(
        &self,
        emitter: &crate::events::EventEmitter,
        session_id: &str,
        task_id: &str,
        tool_calls: &[ToolCall],
        result_text: String,
    ) -> anyhow::Result<()> {
        for tc in tool_calls {
            let tool_msg = Message {
                id: Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
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
                emitter,
                &tool_msg,
                true,
                0,
                None,
                Some(task_id),
            )
            .await?;
        }

        Ok(())
    }

    async fn request_pre_execution_plan(
        &self,
        llm_provider: Arc<dyn ModelProvider>,
        model: &str,
        user_text: &str,
        assistant_narration: Option<&str>,
        tool_call: &ToolCall,
        capabilities: ToolCapabilities,
    ) -> anyhow::Result<PlanState> {
        let target = extract_target_preview(&tool_call.arguments).unwrap_or_default();
        let messages = vec![
            json!({
                "role": "system",
                "content": "Return only JSON matching the schema. Produce a minimal pre-execution plan for the first risky tool action before execution. The first_action.tool must exactly match the proposed tool call name."
            }),
            json!({
                "role": "user",
                "content": format!(
                    "User request:\n{user_text}\n\nAssistant narration before execution:\n{}\n\nProposed risky tool call:\n{}\n\nReturn a minimal plan for this immediate action. Keep success criteria concrete and short. Mark requires_verification=true for this risky action.",
                    assistant_narration
                        .map(str::trim)
                        .filter(|text| !text.is_empty())
                        .unwrap_or("<none>"),
                    serde_json::to_string_pretty(&json!({
                        "tool": tool_call.name,
                        "target_hint": target,
                        "arguments": summarize_tool_arguments(&tool_call.arguments),
                        "capabilities": {
                            "read_only": capabilities.read_only,
                            "external_side_effect": capabilities.external_side_effect,
                            "needs_approval": capabilities.needs_approval,
                            "idempotent": capabilities.idempotent,
                            "high_impact_write": capabilities.high_impact_write,
                        }
                    }))
                    .unwrap_or_else(|_| tool_call.arguments.clone())
                )
            }),
        ];
        let options = ChatOptions {
            response_mode: crate::traits::ResponseMode::JsonSchema {
                name: "pre_execution_plan_v1".to_string(),
                schema: pre_execution_plan_schema_json(),
                strict: true,
            },
            tool_choice: ToolChoiceMode::None,
            ..ChatOptions::default()
        };
        let response = llm_provider
            .chat_with_options(model, &messages, &[], &options)
            .await?;
        let raw = response
            .content
            .ok_or_else(|| anyhow::anyhow!("pre-execution planning response was empty"))?;
        let plan = serde_json::from_str::<PlanState>(&raw)?;
        Ok(plan.normalize())
    }

    #[allow(clippy::too_many_arguments)]
    async fn request_pre_execution_critique(
        &self,
        llm_provider: Arc<dyn ModelProvider>,
        model: &str,
        user_text: &str,
        assistant_narration: Option<&str>,
        tool_call: &ToolCall,
        plan: &PlanState,
        evidence_state: &EvidenceState,
        capabilities: ToolCapabilities,
        expected_target: Option<&str>,
    ) -> anyhow::Result<CritiqueState> {
        let messages = vec![
            json!({
                "role": "system",
                "content": "Return only JSON matching the schema. You are a brief critique pass for a risky first tool action. Focus only on concrete issues in these categories: wrong target, missing evidence, unverifiable success criteria, unsafe first action. Use verdict=replan only when one of those issues is specific and blocking."
            }),
            json!({
                "role": "user",
                "content": format!(
                    "User request:\n{user_text}\n\nAssistant narration before execution:\n{}\n\nProposed risky tool call:\n{}\n\nPlan under review:\n{}\n\nCurrent evidence snapshot:\n{}\n\nOnly flag concrete blockers. Do not ask for generic caution.",
                    assistant_narration
                        .map(str::trim)
                        .filter(|text| !text.is_empty())
                        .unwrap_or("<none>"),
                    serde_json::to_string_pretty(&json!({
                        "tool": tool_call.name,
                        "target_hint": expected_target,
                        "arguments": summarize_tool_arguments(&tool_call.arguments),
                        "capabilities": {
                            "read_only": capabilities.read_only,
                            "external_side_effect": capabilities.external_side_effect,
                            "needs_approval": capabilities.needs_approval,
                            "idempotent": capabilities.idempotent,
                            "high_impact_write": capabilities.high_impact_write,
                        }
                    }))
                    .unwrap_or_else(|_| tool_call.arguments.clone()),
                    serde_json::to_string_pretty(plan).unwrap_or_else(|_| "<plan unavailable>".to_string()),
                    serde_json::to_string_pretty(&summarize_evidence_state(evidence_state))
                        .unwrap_or_else(|_| "<evidence unavailable>".to_string()),
                )
            }),
        ];
        let options = ChatOptions {
            response_mode: crate::traits::ResponseMode::JsonSchema {
                name: "pre_execution_critique_v1".to_string(),
                schema: pre_execution_critique_schema_json(),
                strict: true,
            },
            tool_choice: ToolChoiceMode::None,
            ..ChatOptions::default()
        };
        let response = llm_provider
            .chat_with_options(model, &messages, &[], &options)
            .await?;
        let raw = response
            .content
            .ok_or_else(|| anyhow::anyhow!("pre-execution critique response was empty"))?;
        let critique = serde_json::from_str::<CritiqueState>(&raw)?;
        Ok(critique.normalize())
    }

    pub(super) async fn run_tool_prelude_phase(
        &self,
        ctx: &mut ToolPreludeCtx<'_>,
    ) -> anyhow::Result<ToolPreludeOutcome> {
        let resp = ctx.resp;
        let emitter = ctx.emitter;
        let task_id = ctx.task_id;
        let session_id = ctx.session_id;
        let model = ctx.model;
        let llm_provider = ctx.llm_provider.clone();
        let iteration = ctx.iteration;
        let task_start = ctx.task_start;
        let learning_ctx = &mut *ctx.learning_ctx;
        let evidence_state = ctx.evidence_state;
        let user_text = ctx.user_text;
        let policy_bundle = ctx.policy_bundle;
        let available_capabilities = ctx.available_capabilities;
        let execution_state = &mut *ctx.execution_state;
        let validation_state = &mut *ctx.validation_state;
        let pending_system_messages = &mut *ctx.pending_system_messages;
        let force_text_response = &mut *ctx.force_text_response;
        let turn_context = ctx.turn_context;
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
            ..Message::runtime_defaults()
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
                    ..Message::runtime_defaults()
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

        if let Some(side_effecting_tool_call) =
            first_side_effecting_tool_call(self, resp, available_capabilities)
        {
            // Memory tools (remember_fact, manage_memories, manage_people) should
            // never be redirected to plain-text completion — the agent legitimately
            // stores information even for conversational requests.
            let all_side_effecting_are_memory = resp
                .tool_calls
                .iter()
                .filter(|tc| tool_call_is_side_effecting(self, tc, available_capabilities))
                .all(|tc| crate::agent::recall_guardrails::is_personal_memory_tool(&tc.name));
            // Sub-sessions (spawned TaskLead/Executor) exist to execute actions —
            // never redirect them to plain-text mode.
            let is_sub_session = session_id.starts_with("sub-");
            if !is_sub_session
                && !all_side_effecting_are_memory
                && turn_prefers_plain_text_completion(turn_context)
            {
                validation_state.note_replan();
                learning_ctx.record_replay_note(
                    ReplayNoteCategory::RetryReason,
                    "text_only_tool_drift",
                    format!(
                        "Retried in plain-text mode after {} drifted into side-effecting execution on a text-only request.",
                        side_effecting_tool_call.name
                    ),
                    true,
                );
                *force_text_response = true;
                pending_system_messages.push(SystemDirective::ToolModeDisabledPlainText);
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::ExecutionStateSnapshot,
                    format!(
                        "Redirecting {} back to plain-text completion",
                        side_effecting_tool_call.name
                    ),
                    json!({
                        "condition": "text_only_turn_side_effecting_tool_drift",
                        "tool": side_effecting_tool_call.name,
                        "loop_repetition_reason": validation_state.loop_repetition_reason,
                    }),
                )
                .await;
                self.inject_prelude_retry_messages(
                    emitter,
                    session_id,
                    task_id,
                    &resp.tool_calls,
                    "[SYSTEM] This request should be answered directly in plain text. Do not call side-effecting tools for it. Write the requested content instead."
                        .to_string(),
                )
                .await?;
                return Ok(ToolPreludeOutcome::ContinueLoop);
            }
        }

        if !resp.tool_calls.is_empty() {
            execution_state.mark_persisted_now();
            self.emit_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::ExecutionStateSnapshot,
                "Execution state snapshot before tool execution".to_string(),
                json!({
                    "condition": "prelude_ready_for_execution",
                    "execution_state": execution_state.clone(),
                    "tool_count": resp.tool_calls.len(),
                }),
            )
            .await;
        }

        let uncertainty_threshold =
            current_uncertainty_threshold(self.policy_config.uncertainty_clarify_threshold);
        if self.policy_config.uncertainty_clarify_enforce
            && policy_bundle.uncertainty_score >= uncertainty_threshold
        {
            let has_side_effecting_call = resp
                .tool_calls
                .iter()
                .any(|tc| tool_call_is_side_effecting(self, tc, available_capabilities));
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
                    ..Message::runtime_defaults()
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

        for tc in &resp.tool_calls {
            if let Some(violation) =
                assess_pre_execution_evidence_gate(&tc.name, &tc.arguments, evidence_state)
            {
                validation_state.record_failure(ValidationFailure::MissingEvidence);
                validation_state.note_retry(LoopRepetitionReason::MissingEvidence);
                learning_ctx.record_replay_note(
                    ReplayNoteCategory::EvidenceGate,
                    "missing_pre_execution_evidence",
                    format!(
                        "Blocked {} until {} evidence exists for {}.",
                        tc.name,
                        format!("{:?}", violation.kind).to_ascii_lowercase(),
                        violation.target.as_deref().unwrap_or("the current target")
                    ),
                    true,
                );
                learning_ctx.record_replay_note(
                    ReplayNoteCategory::RetryReason,
                    "missing_evidence",
                    format!(
                        "Retried after evidence gate blocked {} for {}.",
                        tc.name,
                        violation.target.as_deref().unwrap_or("the current target")
                    ),
                    true,
                );
                self.emit_warning_decision_point(
                    emitter,
                    task_id,
                    iteration,
                    DecisionType::EvidenceGate,
                    format!("Blocked {} until required evidence is gathered", tc.name),
                    json!({
                        "condition": "missing_pre_execution_evidence",
                        "tool": tc.name,
                        "required_evidence_kind": violation.kind,
                        "target": violation.target,
                        "reason": violation.reason,
                        "loop_repetition_reason": validation_state.loop_repetition_reason,
                    }),
                )
                .await;
                self.inject_prelude_retry_messages(
                    emitter,
                    session_id,
                    task_id,
                    &resp.tool_calls,
                    format!(
                        "[SYSTEM] Evidence gate blocked this tool call. {} {}",
                        violation.reason, violation.coaching
                    ),
                )
                .await?;
                return Ok(ToolPreludeOutcome::ContinueLoop);
            }
        }

        let first_risky_tool_call =
            first_side_effecting_tool_call(self, resp, available_capabilities).cloned();
        // Gate: only request the pre-execution plan once per task.
        // Skip if: (a) a side-effecting tool already completed successfully
        // (learning_ctx tracks it), OR (b) a plan was already accepted
        // (current_plan_version is set).  Without check (b), a critique
        // rejection causes ContinueLoop without executing the tool, so the
        // learning_ctx guard stays false and the plan is re-requested every
        // iteration — adding 2 extra LLM calls (plan + critique) per loop.
        let plan_already_generated = execution_state.current_plan_version.is_some();
        if self.depth == 0
            && !plan_already_generated
            && !has_completed_side_effecting_tool_call(learning_ctx, available_capabilities)
        {
            if let Some(first_risky_tool_call) = first_risky_tool_call {
                let capabilities = available_capabilities
                    .get(&first_risky_tool_call.name)
                    .copied()
                    .unwrap_or_default();
                let expected_target = extract_target_preview(&first_risky_tool_call.arguments);

                // Pre-execution planning is a system-initiated quality check,
                // not an agent action. Do not charge it against the execution
                // budget — the agent should not be penalised for the system's
                // own safety overhead.
                match self
                    .request_pre_execution_plan(
                        llm_provider,
                        model,
                        user_text,
                        resp.content.as_deref(),
                        &first_risky_tool_call,
                        capabilities,
                    )
                    .await
                {
                    Ok(plan) => match validate_pre_execution_plan(
                        &plan,
                        &first_risky_tool_call,
                        expected_target.as_deref(),
                    ) {
                        Ok(()) => {
                            execution_state.set_plan_version(plan.version);
                            // Clear any stale plan from previous iterations before
                            // conditionally installing a new one.
                            execution_state.active_linear_intent_plan = None;
                            if !plan.planned_steps.is_empty() {
                                execution_state.install_linear_intent_plan(
                                    plan.version,
                                    plan.planned_steps
                                        .iter()
                                        .enumerate()
                                        .map(|(idx, step)| LinearIntentStep {
                                            step_id: format!(
                                                "plan-v{}-step-{}",
                                                plan.version,
                                                idx + 1
                                            ),
                                            step_index: idx + 1,
                                            tool: step.tool.clone(),
                                            target: step.target.clone(),
                                            description: step.description.clone(),
                                            tool_calls_on_step: 0,
                                            completed: false,
                                            completion_evidence: None,
                                            last_evaluated_at: None,
                                        })
                                        .collect(),
                                );
                            }
                            validation_state.set_plan(plan.version, &plan.success_criteria);
                            validation_state.clear_loop_repetition_reason();
                            learning_ctx.record_replay_note(
                                ReplayNoteCategory::PlanRevision,
                                "plan_accepted",
                                format!(
                                    "Accepted plan v{} for {} targeting {}.",
                                    plan.version,
                                    first_risky_tool_call.name,
                                    expected_target.as_deref().unwrap_or("unspecified target")
                                ),
                                false,
                            );
                            self.emit_decision_point(
                                emitter,
                                task_id,
                                iteration,
                                DecisionType::ExecutionPlanningGate,
                                format!(
                                    "Structured pre-execution plan accepted for {}",
                                    first_risky_tool_call.name
                                ),
                                json!({
                                    "condition": "plan_accepted",
                                    "gate_result": "accepted",
                                    "tool": first_risky_tool_call.name,
                                    "target_hint": expected_target.as_deref(),
                                    "requires_verification": plan.requires_verification,
                                    "success_criteria_count": plan.success_criteria.len(),
                                    "risky_action_count": plan.risky_actions.len(),
                                    "plan": &plan,
                                }),
                            )
                            .await;

                            if should_run_pre_execution_critique(
                                policy_bundle,
                                capabilities,
                                execution_state,
                            ) {
                                // Same principle: critique is system-initiated
                                // quality gating, not agent work. Do not charge
                                // it against the execution budget.
                                match self
                                    .request_pre_execution_critique(
                                        ctx.llm_provider.clone(),
                                        model,
                                        user_text,
                                        resp.content.as_deref(),
                                        &first_risky_tool_call,
                                        &plan,
                                        evidence_state,
                                        capabilities,
                                        expected_target.as_deref(),
                                    )
                                    .await
                                {
                                    Ok(critique) => {
                                        match validate_pre_execution_critique(&critique) {
                                            Ok(())
                                                if matches!(
                                                    critique.verdict,
                                                    CritiqueVerdict::Accept
                                                ) =>
                                            {
                                                learning_ctx.record_replay_note(
                                                    ReplayNoteCategory::PlanRevision,
                                                    "critique_accepted",
                                                    format!(
                                                        "Critique accepted the first {} step.",
                                                        first_risky_tool_call.name
                                                    ),
                                                    false,
                                                );
                                                self.emit_decision_point(
                                                emitter,
                                                task_id,
                                                iteration,
                                                DecisionType::ExecutionCritiquePass,
                                                format!(
                                                    "Pre-execution critique accepted {}",
                                                    first_risky_tool_call.name
                                                ),
                                                json!({
                                                    "condition": "critique_accepted",
                                                    "critique_result": "accepted",
                                                    "tool": first_risky_tool_call.name,
                                                    "target_hint": expected_target.as_deref(),
                                                    "summary": &critique.summary,
                                                    "issues": &critique.issues,
                                                    "risk_score": policy_bundle.risk_score,
                                                    "uncertainty_score": policy_bundle.uncertainty_score,
                                                }),
                                            )
                                            .await;
                                            }
                                            Ok(()) => {
                                                // Clear stale linear intent plan — rejected
                                                // critique means the plan that produced it
                                                // is invalid.
                                                execution_state.active_linear_intent_plan = None;
                                                validation_state.record_failure(
                                                    ValidationFailure::CritiqueRejected,
                                                );
                                                validation_state.note_replan_for(
                                                    LoopRepetitionReason::CritiqueRejected,
                                                );
                                                learning_ctx.record_replay_note(
                                                    ReplayNoteCategory::PlanRevision,
                                                    "critique_rejected",
                                                    format!(
                                                        "Critique rejected the first {} step: {}",
                                                        first_risky_tool_call.name,
                                                        if critique.issues.is_empty() {
                                                            critique.summary.clone()
                                                        } else {
                                                            critique.issues.join("; ")
                                                        }
                                                    ),
                                                    true,
                                                );
                                                learning_ctx.record_replay_note(
                                                    ReplayNoteCategory::RetryReason,
                                                    "critique_rejected",
                                                    format!(
                                                        "Replanned because critique rejected {}.",
                                                        first_risky_tool_call.name
                                                    ),
                                                    true,
                                                );
                                                self.emit_warning_decision_point(
                                                emitter,
                                                task_id,
                                                iteration,
                                                DecisionType::ExecutionCritiquePass,
                                                format!(
                                                    "Pre-execution critique rejected {}",
                                                    first_risky_tool_call.name
                                                ),
                                                json!({
                                                    "condition": "critique_rejected",
                                                    "critique_result": "rejected",
                                                    "tool": first_risky_tool_call.name,
                                                    "target_hint": expected_target.as_deref(),
                                                    "summary": &critique.summary,
                                                    "issues": &critique.issues,
                                                    "risk_score": policy_bundle.risk_score,
                                                    "uncertainty_score": policy_bundle.uncertainty_score,
                                                    "loop_repetition_reason": validation_state.loop_repetition_reason,
                                                }),
                                            )
                                            .await;
                                                let issues = critique.issues.join("; ");
                                                self.inject_prelude_retry_messages(
                                                emitter,
                                                session_id,
                                                task_id,
                                                &resp.tool_calls,
                                                format!(
                                                    "[SYSTEM] Critique pass blocked this risky action. \
                                                     Issues: {}. Re-plan the first action, gather any missing \
                                                     evidence, briefly explain the corrected approach, and then \
                                                     re-issue tool calls.",
                                                    if issues.is_empty() {
                                                        critique.summary
                                                    } else {
                                                        issues
                                                    }
                                                ),
                                            )
                                            .await?;
                                                return Ok(ToolPreludeOutcome::ContinueLoop);
                                            }
                                            Err(reason) => {
                                                self.emit_warning_decision_point(
                                                    emitter,
                                                    task_id,
                                                    iteration,
                                                    DecisionType::ExecutionCritiquePass,
                                                    format!(
                                                        "Pre-execution critique was invalid for {}",
                                                        first_risky_tool_call.name
                                                    ),
                                                    json!({
                                                        "condition": "critique_invalid",
                                                        "critique_result": "invalid",
                                                        "reason": reason,
                                                        "tool": first_risky_tool_call.name,
                                                        "target_hint": expected_target.as_deref(),
                                                        "critique": critique,
                                                    }),
                                                )
                                                .await;
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        self.emit_warning_decision_point(
                                            emitter,
                                            task_id,
                                            iteration,
                                            DecisionType::ExecutionCritiquePass,
                                            format!(
                                                "Pre-execution critique unavailable for {}",
                                                first_risky_tool_call.name
                                            ),
                                            json!({
                                                "condition": "critique_unavailable",
                                                "critique_result": "unavailable",
                                                "reason": "critique_generation_failed",
                                                "tool": first_risky_tool_call.name,
                                                "target_hint": expected_target.as_deref(),
                                                "error": error.to_string(),
                                            }),
                                        )
                                        .await;
                                        warn!(
                                            session_id,
                                            tool = %first_risky_tool_call.name,
                                            error = %error,
                                            "Pre-execution critique pass unavailable; proceeding with existing guards"
                                        );
                                    }
                                }
                            } else if capabilities.high_impact_write
                                || capabilities.external_side_effect
                                || capabilities.needs_approval
                            {
                                self.emit_warning_decision_point(
                                    emitter,
                                    task_id,
                                    iteration,
                                    DecisionType::ExecutionCritiquePass,
                                    format!(
                                        "Pre-execution critique skipped for {} because budget is exhausted",
                                        first_risky_tool_call.name
                                    ),
                                    json!({
                                        "condition": "critique_skipped_budget",
                                        "critique_result": "skipped_budget",
                                        "tool": first_risky_tool_call.name,
                                        "target_hint": expected_target.as_deref(),
                                        "risk_score": policy_bundle.risk_score,
                                        "uncertainty_score": policy_bundle.uncertainty_score,
                                        "llm_calls_used": execution_state.llm_calls_used,
                                        "validation_rounds_used": execution_state.validation_rounds_used,
                                        "budget": execution_state.budget.clone(),
                                    }),
                                )
                                .await;
                            }
                        }
                        Err(reason) => {
                            validation_state.record_failure(ValidationFailure::PlanRejected);
                            validation_state.note_replan_for(LoopRepetitionReason::PlanRejected);
                            learning_ctx.record_replay_note(
                                ReplayNoteCategory::PlanRevision,
                                "plan_rejected",
                                format!(
                                    "Rejected the first {} step because the structured plan failed validation: {}.",
                                    first_risky_tool_call.name, reason
                                ),
                                true,
                            );
                            learning_ctx.record_replay_note(
                                ReplayNoteCategory::RetryReason,
                                "plan_rejected",
                                format!(
                                    "Replanned because the first structured plan for {} was invalid.",
                                    first_risky_tool_call.name
                                ),
                                true,
                            );
                            self.emit_warning_decision_point(
                                emitter,
                                task_id,
                                iteration,
                                DecisionType::ExecutionPlanningGate,
                                format!(
                                    "Structured pre-execution plan rejected for {}",
                                    first_risky_tool_call.name
                                ),
                                json!({
                                    "condition": "plan_rejected",
                                    "gate_result": "rejected",
                                    "reason": reason,
                                    "tool": first_risky_tool_call.name,
                                    "target_hint": expected_target.as_deref(),
                                    "plan": &plan,
                                    "loop_repetition_reason": validation_state.loop_repetition_reason,
                                }),
                            )
                            .await;
                            self.inject_prelude_retry_messages(
                                emitter,
                                session_id,
                                task_id,
                                &resp.tool_calls,
                                format!(
                                    "[SYSTEM] Pre-execution planning check blocked this risky action. \
                                     Reason: {reason}. Reconsider the first risky action, briefly explain \
                                     the plan in user-facing text, and re-issue corrected tool calls."
                                ),
                            )
                            .await?;
                            return Ok(ToolPreludeOutcome::ContinueLoop);
                        }
                    },
                    Err(error) => {
                        self.emit_warning_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::ExecutionPlanningGate,
                            format!(
                                "Structured pre-execution plan unavailable for {}",
                                first_risky_tool_call.name
                            ),
                            json!({
                                "condition": "plan_unavailable",
                                "gate_result": "unavailable",
                                "reason": "plan_generation_failed",
                                "tool": first_risky_tool_call.name,
                                "target_hint": expected_target.as_deref(),
                                "error": error.to_string(),
                            }),
                        )
                        .await;
                        warn!(
                            session_id,
                            tool = %first_risky_tool_call.name,
                            error = %error,
                            "Structured pre-execution planning gate unavailable; proceeding with existing guards"
                        );
                    }
                }
            }
        }

        Ok(ToolPreludeOutcome::Proceed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_plan() -> PlanState {
        PlanState {
            goal: "Post a thread".to_string(),
            success_criteria: vec!["all steps completed".to_string()],
            first_action: PlannedAction {
                tool: "http_request".to_string(),
                target: "https://api.example.com/posts/1".to_string(),
                description: "Post tweet 1".to_string(),
            },
            requires_verification: true,
            risky_actions: vec!["Posting externally".to_string()],
            version: 1,
            planned_steps: vec![PlannedAction {
                tool: "http_request".to_string(),
                target: "https://api.example.com/posts/1".to_string(),
                description: "Post tweet 1".to_string(),
            }],
        }
    }

    fn base_tool_call() -> ToolCall {
        ToolCall {
            id: "tc_1".to_string(),
            name: "http_request".to_string(),
            arguments: "{}".to_string(),
            extra_content: None,
        }
    }

    #[test]
    fn validate_pre_execution_plan_rejects_missing_first_step_target() {
        let mut plan = base_plan();
        plan.planned_steps[0].target.clear();
        let tc = base_tool_call();
        let result = validate_pre_execution_plan(&plan, &tc, Some(&plan.first_action.target));
        assert_eq!(result, Err("planned_steps_first_action_target_mismatch"));
    }

    #[test]
    fn validate_pre_execution_plan_accepts_matching_first_step_target() {
        let plan = base_plan();
        let tc = base_tool_call();
        assert!(validate_pre_execution_plan(&plan, &tc, Some(&plan.first_action.target)).is_ok());
    }
}
