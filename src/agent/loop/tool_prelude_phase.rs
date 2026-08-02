use super::execution_state::LinearIntentStep;
use super::turn_transition::{TurnRestartReason, TurnTransition};
use super::*;
use crate::events::TaskOutcome;
use crate::execution_policy::PolicyBundle;
use crate::traits::ProviderResponse;
use serde::{Deserialize, Serialize};

pub(super) enum ToolPreludeOutcome {
    ContinueLoop,
    Return(anyhow::Result<String>),
    Proceed,
}

impl ToolPreludeOutcome {
    pub(super) fn into_turn_transition(self) -> TurnTransition<()> {
        match self {
            Self::ContinueLoop => TurnTransition::Restart(TurnRestartReason::ToolPreludeRecovery),
            Self::Return(result) => TurnTransition::Finish(result),
            Self::Proceed => TurnTransition::Advance(()),
        }
    }
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
    pub project_instruction_tracker:
        &'a mut Option<crate::project_instructions::ProjectInstructionTracker>,
    pub task_context_tail: &'a mut String,
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

fn plain_text_guard_blocks_tool(tool_name: &str, is_side_effecting: bool) -> bool {
    user_visible_side_effect_guard_blocks_tool(tool_name, is_side_effecting)
}

fn uncertainty_guard_blocks_tool(tool_name: &str, is_side_effecting: bool) -> bool {
    user_visible_side_effect_guard_blocks_tool(tool_name, is_side_effecting)
}

fn user_visible_side_effect_guard_blocks_tool(tool_name: &str, is_side_effecting: bool) -> bool {
    is_side_effecting && tool_name != "spawn_agent"
}

fn scoped_action_matches_tool_call(action: ForbiddenMutationAction, tc: &ToolCall) -> bool {
    let normalized_name = tc.name.replace(['_', '-'], " ").to_ascii_lowercase();
    let command = if matches!(tc.name.as_str(), "terminal" | "run_command") {
        serde_json::from_str::<Value>(&tc.arguments)
            .ok()
            .and_then(|args| {
                ["command", "cmd", "script"]
                    .iter()
                    .find_map(|key| args.get(*key).and_then(Value::as_str).map(str::to_string))
            })
            .unwrap_or_default()
    } else {
        String::new()
    };
    let delegated_instructions = if matches!(tc.name.as_str(), "spawn_agent" | "cli_agent") {
        serde_json::from_str::<Value>(&tc.arguments)
            .ok()
            .and_then(|args| args.as_object().cloned())
            .map(|args| {
                [
                    "mission",
                    "task",
                    "prompt",
                    "command",
                    "description",
                    "system_instruction",
                ]
                .iter()
                .filter_map(|key| args.get(*key).and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
            })
            .unwrap_or_default()
            .to_ascii_lowercase()
    } else {
        String::new()
    };

    let name_has = |keyword: &str| contains_keyword_as_words(&normalized_name, keyword);
    let command_effects = if command.is_empty() {
        crate::traits::ToolMutationEffects::NONE
    } else {
        crate::tools::command_semantics::classify_shell_command(&command).mutation_effects
    };
    let delegated_requests = |keyword: &str| {
        let text = delegated_instructions.trim();
        if text.is_empty()
            || [
                format!("do not {keyword}"),
                format!("don't {keyword}"),
                format!("without {keyword}"),
                format!("how to {keyword}"),
                format!("whether to {keyword}"),
            ]
            .iter()
            .any(|phrase| contains_keyword_as_words(text, phrase))
        {
            return false;
        }
        text.lines()
            .any(|line| line.trim_start().starts_with(&format!("{keyword} ")))
            || [
                format!(" and {keyword}"),
                format!(" then {keyword}"),
                format!(" to {keyword}"),
                format!(" please {keyword}"),
                format!(" must {keyword}"),
            ]
            .iter()
            .any(|phrase| text.contains(phrase))
    };

    match action {
        ForbiddenMutationAction::Create => {
            matches!(tc.name.as_str(), "write_file")
                || name_has("create")
                || command_effects.intersects(
                    crate::traits::ToolMutationEffects::LOCAL_SOURCE_WRITE
                        .union(crate::traits::ToolMutationEffects::LOCAL_WORKSPACE_WRITE),
                )
                || ["create", "write", "build", "make", "generate"]
                    .iter()
                    .any(|keyword| delegated_requests(keyword))
        }
        ForbiddenMutationAction::Delete => {
            name_has("delete")
                || name_has("remove")
                || command_effects.intersects(crate::traits::ToolMutationEffects::DESTRUCTIVE)
                || ["delete", "remove"]
                    .iter()
                    .any(|keyword| delegated_requests(keyword))
        }
        ForbiddenMutationAction::Deploy => {
            name_has("deploy")
                || command_effects.intersects(crate::traits::ToolMutationEffects::REMOTE_DEPLOY)
                || delegated_requests("deploy")
        }
        ForbiddenMutationAction::Publish => {
            name_has("publish")
                || command_effects.intersects(
                    crate::traits::ToolMutationEffects::REMOTE_DEPLOY
                        .union(crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY)
                        .union(crate::traits::ToolMutationEffects::REMOTE_MUTATION),
                )
                || delegated_requests("publish")
        }
        ForbiddenMutationAction::Post => {
            name_has("post")
                || command_effects.intersects(
                    crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY
                        .union(crate::traits::ToolMutationEffects::REMOTE_MUTATION),
                )
                || delegated_requests("post")
        }
        ForbiddenMutationAction::Send => {
            name_has("send")
                || command_effects.intersects(crate::traits::ToolMutationEffects::EXTERNAL_DELIVERY)
                || delegated_requests("send")
        }
    }
}

fn negative_contract_block_reason(
    contract: &CompletionContract,
    tool_call: &ToolCall,
    is_side_effecting: bool,
) -> Option<String> {
    if contract.forbids_mutation {
        if is_side_effecting || matches!(tool_call.name.as_str(), "spawn_agent" | "cli_agent") {
            return Some("all mutation".to_string());
        }
        return None;
    }
    if !user_visible_side_effect_guard_blocks_tool(&tool_call.name, is_side_effecting)
        && !matches!(tool_call.name.as_str(), "spawn_agent" | "cli_agent")
    {
        return None;
    }
    contract
        .forbidden_mutation_actions
        .iter()
        .copied()
        .find(|action| scoped_action_matches_tool_call(*action, tool_call))
        .map(|action| action.as_str().to_string())
}

fn first_plain_text_blocked_tool_call<'a>(
    agent: &Agent,
    resp: &'a ProviderResponse,
    available_capabilities: &HashMap<String, ToolCapabilities>,
) -> Option<&'a ToolCall> {
    resp.tool_calls.iter().find(|tc| {
        plain_text_guard_blocks_tool(
            &tc.name,
            tool_call_is_side_effecting(agent, tc, available_capabilities),
        )
    })
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

/// Normalize a user reply for affirmation matching: trim, lowercase, and strip
/// surrounding punctuation (preserving interior apostrophes for contractions).
fn normalize_affirmation_text(text: &str) -> String {
    text.trim()
        .trim_matches(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
        .to_ascii_lowercase()
}

/// True for short, unambiguous affirmative/approval replies (e.g. "yes",
/// "go ahead", "try that"). A bare affirmation carries no action signal by
/// itself — its intent lives in the proposal the assistant just made.
fn is_short_affirmation_or_approval(text: &str) -> bool {
    let normalized = normalize_affirmation_text(text);
    if normalized.is_empty() {
        return false;
    }
    // Questions are never approvals.
    if normalized.contains('?') {
        return false;
    }
    // Short replies only: a long sentence is doing more than approving.
    if normalized.split_whitespace().count() > 6 {
        return false;
    }
    // Explicit refusal short-circuits.
    if contains_keyword_as_words(&normalized, "no")
        || contains_keyword_as_words(&normalized, "don't")
        || contains_keyword_as_words(&normalized, "stop")
    {
        return false;
    }

    const SINGLE_WORD_AFFIRMATIONS: &[&str] = &[
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "proceed", "sgtm", "please",
    ];
    if SINGLE_WORD_AFFIRMATIONS
        .iter()
        .any(|w| contains_keyword_as_words(&normalized, w))
    {
        return true;
    }

    const APPROVAL_PHRASES: &[&str] = &[
        "go ahead",
        "do it",
        "please do",
        "please proceed",
        "try that",
        "try it",
        "let's do it",
        "lets do it",
        "sounds good",
        "go for it",
        "yes please",
        "that works",
    ];
    APPROVAL_PHRASES
        .iter()
        .any(|phrase| contains_keyword_as_words(&normalized, phrase))
}

/// Inspect the most recent assistant-role message in `recent_messages` and
/// return true if it proposed/offered to do something the user could approve.
fn assistant_proposed_action(recent_messages: &[Value]) -> bool {
    let Some(content) = recent_messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .and_then(|m| m.get("content").and_then(Value::as_str))
    else {
        return false;
    };
    let lower = content.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    const PROPOSAL_PHRASES: &[&str] = &[
        "would you like me to",
        "want me to",
        "do you want me to",
        "shall i",
        "should i",
        "would you like",
        "i can run",
        "i could run",
        "let me know if you'd like",
        "want me to run",
        "like me to",
    ];
    if PROPOSAL_PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // Fallback: a question that also references a concrete action.
    if lower.ends_with('?') {
        const ACTION_WORDS: &[&str] = &[
            "run", "extract", "create", "generate", "build", "fetch", "script", "execute", "try",
        ];
        if ACTION_WORDS
            .iter()
            .any(|w| contains_keyword_as_words(&lower, w))
        {
            return true;
        }
    }

    false
}

/// True when the current turn is a short approval of an action the assistant
/// proposed in its prior turn. Requires BOTH the affirmation and a real prior
/// proposal so genuine text-only turns are unaffected.
fn turn_is_approval_of_prior_proposal(turn_context: &TurnContext) -> bool {
    is_short_affirmation_or_approval(&turn_context.goal_user_text)
        && assistant_proposed_action(&turn_context.recent_messages)
}

/// When the turn is a short approval of the assistant's own prior proposal,
/// return that proposal text so the loop can anchor execution to it.
/// Without the anchor, small models drift into re-planning ("Yes, read the
/// Right one" → six fresh searches and no read).
pub(in crate::agent) fn approved_proposal_text(turn_context: &TurnContext) -> Option<String> {
    if !turn_is_approval_of_prior_proposal(turn_context) {
        return None;
    }
    turn_context
        .recent_messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .and_then(|m| m.get("content").and_then(Value::as_str))
        .map(|content| crate::utils::truncate_str(content.trim(), 600))
}

/// A read-only lookup turn: the user asked a question, so a tool the model
/// reaches for is almost certainly to *observe* state, not to mutate it.
fn turn_is_interrogative_lookup(turn_context: &TurnContext) -> bool {
    let t = turn_context.goal_user_text.trim().to_ascii_lowercase();
    if t.is_empty() {
        return false;
    }
    t.ends_with('?')
        || [
            "how many ",
            "how much ",
            "who ",
            "which ",
            "list ",
            "what are ",
            "are there ",
            "is there ",
            "do i have ",
            "do we have ",
            "any ",
        ]
        .iter()
        .any(|prefix| t.starts_with(prefix))
}

/// Whether the plain-text redirect should spare this blocked tool. Pure-read
/// tools never irreversibly mutate, so blocking them only forces the model to
/// fabricate an answer instead of looking it up — always exempt. `terminal`
/// can also mutate, so it is exempt when the turn is an interrogative lookup
/// OR the command itself classifies as observation-only (find/ls/grep/...);
/// destructive commands remain gated by command-risk approval downstream.
fn plain_text_redirect_exempts_lookup(
    tool_name: &str,
    arguments: &str,
    turn_context: &TurnContext,
) -> bool {
    const PURE_READ_TOOLS: &[&str] = &[
        "read_file",
        "search_files",
        "web_search",
        "web_fetch",
        "read_channel_history",
    ];
    if PURE_READ_TOOLS.contains(&tool_name) {
        return true;
    }
    if !matches!(tool_name, "terminal" | "run_command") {
        return false;
    }
    turn_is_interrogative_lookup(turn_context) || terminal_command_is_read_only(arguments)
}

/// True when a terminal call's command classifies as pure observation
/// (observes state, mutates nothing). Malformed arguments fail closed.
fn terminal_command_is_read_only(arguments: &str) -> bool {
    let Ok(args) = serde_json::from_str::<Value>(arguments) else {
        return false;
    };
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return false;
    };
    let semantics = crate::tools::command_semantics::classify_shell_command(command);
    semantics.observes_state() && !semantics.mutates_state()
}

fn turn_prefers_plain_text_completion(turn_context: &TurnContext) -> bool {
    // A bare affirmation ("yes", "try that") carries no action signal by
    // itself — its intent is the proposal the assistant just made. When the
    // current turn approves a prior proposal, tools must stay allowed so the
    // model can actually carry out the approved action.
    if turn_is_approval_of_prior_proposal(turn_context) {
        return false;
    }
    if turn_context.completion_contract.forbids_mutation {
        return true;
    }
    // Absence of an inferred mutation/observation is not evidence that a tool
    // call is drift. Only an explicit response-only request may arm this
    // redirect. Scoped prohibitions are enforced per action below and must not
    // disable unrelated work.
    let lower = turn_context.goal_user_text.trim().to_ascii_lowercase();
    ["draft only", "answer only", "explain only"]
        .iter()
        .any(|phrase| contains_keyword_as_words(&lower, phrase))
}

/// The text-only drift redirect protects a plain-text turn from drifting
/// into tool execution — but only before any tool has actually run. Once
/// tools have legitimately executed this turn (e.g. exempted lookups), the
/// turn has proven it needs them; redirecting a later call to plain text
/// forces the model to fabricate mid-chain.
fn text_only_redirect_applies(turn_context: &TurnContext, tool_calls_executed: usize) -> bool {
    tool_calls_executed == 0 && turn_prefers_plain_text_completion(turn_context)
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
    // Critique authority comes from the selected tool's declared capability,
    // not a lexical risk/uncertainty score over the user's prose.
    let _ = policy_bundle;
    capabilities.high_impact_write || capabilities.external_side_effect
}

fn tool_call_has_concrete_target(agent: &Agent, tc: &ToolCall) -> bool {
    if agent
        .tools
        .iter()
        .find(|tool| tool.name() == tc.name && tool.is_available())
        .map(|tool| tool.call_semantics(&tc.arguments))
        .is_some_and(|semantics| {
            semantics
                .target_hints
                .iter()
                .any(|target| !target.value.trim().is_empty())
        })
    {
        return true;
    }

    // Compatibility fallback for adapters that have not yet declared typed
    // target hints. This inspects schema fields only; it never interprets the
    // user's sentence or invents a target value.
    arguments_have_concrete_target(&tc.arguments)
}

fn arguments_have_concrete_target(arguments: &str) -> bool {
    let Ok(args) = serde_json::from_str::<Value>(arguments) else {
        return false;
    };
    [
        "resource_id",
        "path",
        "file_path",
        "url",
        "target",
        "target_path",
        "command",
        "project",
        "project_path",
        "project_dir",
        "repo",
        "repository",
        "repo_path",
        "repo_dir",
        "working_dir",
        "directory",
        "dir",
        "account",
        "channel",
        "recipient",
        "to",
        "goal_id",
        "task_id",
        "name",
    ]
    .iter()
    .any(|key| {
        args.get(*key).is_some_and(|value| match value {
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(values) => !values.is_empty(),
            Value::Object(values) => !values.is_empty(),
            Value::Null => false,
            _ => true,
        })
    })
}

fn should_run_pre_execution_gating(tc: &ToolCall) -> bool {
    if tc.name == "terminal" {
        if let Ok(args) = serde_json::from_str::<Value>(&tc.arguments) {
            let action = args.get("action").and_then(|a| a.as_str()).unwrap_or("run");
            if action == "run" {
                if let Some(command) = args.get("command").and_then(|c| c.as_str()) {
                    let assessment = crate::tools::command_risk::classify_command(command);
                    if assessment.level == crate::tools::command_risk::RiskLevel::Safe {
                        return false;
                    }
                }
            }
        }
    }
    true
}

async fn inject_prelude_retry_messages(
    agent: &Agent,
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
        agent
            .append_tool_message_with_result_event(emitter, &tool_msg, true, 0, None, Some(task_id))
            .await?;
    }

    Ok(())
}

async fn defer_prelude_for_new_project_instructions(
    agent: &Agent,
    emitter: &crate::events::EventEmitter,
    session_id: &str,
    task_id: &str,
    tool_calls: &[ToolCall],
    source_paths: &[String],
) -> anyhow::Result<()> {
    let sources = source_paths.join(", ");
    for tool_call in tool_calls {
        let tool_msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "tool".to_string(),
            content: Some(
                ToolResultNotice::ProjectInstructionsDiscovered {
                    tool_name: tool_call.name.clone(),
                    sources: sources.clone(),
                }
                .render(),
            ),
            tool_call_id: Some(tool_call.id.clone()),
            tool_name: Some(tool_call.name.clone()),
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.2,
            ..Message::runtime_defaults()
        };
        agent
            .append_tool_message_with_result_event(emitter, &tool_msg, true, 0, None, Some(task_id))
            .await?;
    }
    Ok(())
}

async fn request_pre_execution_plan(
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
            "content": "Return only JSON matching the schema. You are a brief critique pass for a risky first tool action. Focus only on concrete issues in these categories: wrong target, missing evidence, unverifiable success criteria, unsafe first action. Use verdict=replan only when one of those issues is specific and blocking. A short or vague user message is not, by itself, \"missing evidence\" \u{2014} only flag missing evidence when the action genuinely depends on a prerequisite that has not been established."
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
    services: &super::services::AgentServices<'_>,
    ctx: &mut ToolPreludeCtx<'_>,
) -> anyhow::Result<ToolPreludeOutcome> {
    let agent = services.agent;
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
    let project_instruction_tracker = &mut *ctx.project_instruction_tracker;
    let task_context_tail = &mut *ctx.task_context_tail;
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
    agent
        .append_assistant_message_with_event(
            emitter,
            &assistant_msg,
            model,
            resp.usage.as_ref().map(|u| u.input_tokens),
            resp.usage.as_ref().map(|u| u.output_tokens),
        )
        .await?;

    // Explicit operation-wide read-only instructions remain an authorization
    // boundary. Scoped action matches, however, are language heuristics: they
    // are useful coaching for Guided models but must not intercept an
    // Autonomous model's valid tool call. Deterministic tool/auth/role policy
    // remains enforced in the execution phase for every model.
    let negative_contract_match = resp.tool_calls.iter().find_map(|tc| {
        negative_contract_block_reason(
            &turn_context.completion_contract,
            tc,
            tool_call_is_side_effecting(agent, tc, available_capabilities),
        )
        .map(|restriction| (tc, restriction))
    });
    let enforce_negative_contract =
        if let Some((blocked, restriction)) = negative_contract_match.as_ref() {
            turn_context.completion_contract.forbids_mutation
                || agent
                    .supervision_gate_enforced_with_context(
                        "scoped_negative_contract",
                        model,
                        emitter,
                        task_id,
                        iteration,
                        json!({
                            "tool": blocked.name,
                            "forbidden_action": restriction,
                        }),
                    )
                    .await
        } else {
            false
        };
    if enforce_negative_contract {
        let (blocked, restriction) = negative_contract_match
            .as_ref()
            .expect("negative contract match exists when enforcement is requested");
        agent
            .with_harness_eval(|eval| eval.record_forbidden_mutation_attempt())
            .await;
        validation_state.note_replan();
        agent
            .emit_warning_decision_point(
                emitter,
                task_id,
                iteration,
                DecisionType::ExecutionStateSnapshot,
                format!(
                    "Blocked {} because the user explicitly forbade mutation",
                    blocked.name
                ),
                json!({
                    "condition": "negative_completion_contract",
                    "tool": blocked.name,
                    "forbids_mutation": turn_context.completion_contract.forbids_mutation,
                    "forbidden_action": restriction.clone(),
                }),
            )
            .await;
        let notice = if turn_context.completion_contract.forbids_mutation {
            format!(
                "[SYSTEM] Blocked `{}`: this task has an explicit read-only/report-only contract. \
                 Do not modify files, create artifacts, deploy, publish, post, send, or otherwise \
                 mutate state. Continue only with observation tools, then report findings.",
                blocked.name
            )
        } else {
            format!(
                "[SYSTEM] Blocked `{}` because this task explicitly forbids the `{}` action. \
                 Other authorized work remains allowed. Continue without performing `{}`, then \
                 report what was completed and what remains for the parent or user.",
                blocked.name, restriction, restriction
            )
        };
        inject_prelude_retry_messages(
            agent,
            emitter,
            session_id,
            task_id,
            &resp.tool_calls,
            notice,
        )
        .await?;
        return Ok(ToolPreludeOutcome::ContinueLoop);
    }

    // Intent gate: on first iteration, require narration before tool calls.
    // Forces the agent to "show its work" so the user can catch misunderstandings.
    if iteration == 1
        && agent.depth == 0
        && !resp.tool_calls.is_empty()
        && resp.content.as_ref().is_none_or(|c| c.trim().len() < 20)
        && agent
            .supervision_gate_enforced("intent_narration_gate", model, emitter, task_id, iteration)
            .await
    {
        info!(
            session_id,
            "Intent gate: requiring narration before tool execution"
        );
        agent
            .with_harness_eval(|eval| eval.record_intent_gate_fire())
            .await;
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
            agent
                .append_tool_message_with_result_event(
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
        first_plain_text_blocked_tool_call(agent, resp, available_capabilities)
    {
        // Memory tools (remember_fact, manage_memories, manage_people) should
        // never be redirected to plain-text completion — the agent legitimately
        // stores information even for conversational requests.
        let all_side_effecting_are_memory = resp
            .tool_calls
            .iter()
            .filter(|tc| tool_call_is_side_effecting(agent, tc, available_capabilities))
            .all(|tc| crate::agent::recall_guardrails::is_personal_memory_tool(&tc.name));
        // Read-only lookups (file/web reads, and shell observation on a question
        // turn) must never be redirected to plain text — doing so forces the
        // model to fabricate an answer it should have looked up.
        let all_side_effecting_are_lookup = resp
            .tool_calls
            .iter()
            .filter(|tc| tool_call_is_side_effecting(agent, tc, available_capabilities))
            .all(|tc| plain_text_redirect_exempts_lookup(&tc.name, &tc.arguments, turn_context));
        // Child sessions (spawned TaskLead/Executor) exist to execute actions —
        // never redirect them to plain-text mode. `sub-` is the legacy prefix
        // kept for in-flight tasks; new sessions use `specialist:`.
        let is_child_session =
            session_id.starts_with("sub-") || session_id.starts_with("specialist:");
        // computer_use drives a multi-step desktop GUI flow (inspect → click →
        // type → verify) that is inherently side-effecting and gated by its own
        // approval ladder. Once the model is operating the desktop it must never
        // be redirected to plain-text completion, or the GUI task stalls after
        // the first observation.
        let any_computer_use = resp.tool_calls.iter().any(|tc| tc.name == "computer_use");
        if !is_child_session
            && !any_computer_use
            && !all_side_effecting_are_memory
            && !all_side_effecting_are_lookup
            && text_only_redirect_applies(turn_context, execution_state.tool_calls_used)
            && agent
                .supervision_gate_enforced(
                    "plain_text_drift_redirect",
                    model,
                    emitter,
                    task_id,
                    iteration,
                )
                .await
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
            agent
                .emit_warning_decision_point(
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
            inject_prelude_retry_messages(
                agent,
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
        agent
            .emit_decision_point(
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
        current_uncertainty_threshold(agent.policy_config.uncertainty_clarify_threshold);
    if agent.policy_config.uncertainty_clarify_enforce {
        let unresolved_side_effecting_call = resp.tool_calls.iter().find(|tc| {
            uncertainty_guard_blocks_tool(
                &tc.name,
                tool_call_is_side_effecting(agent, tc, available_capabilities),
            ) && !tool_call_has_concrete_target(agent, tc)
        });
        // Ambiguity is a property of the proposed operation, not of a word in
        // the user's sentence. The model may resolve "it" from conversation
        // history, but a side effect still cannot run until its structured tool
        // call contains a concrete target/resource handle.
        if unresolved_side_effecting_call.is_some()
            && agent
                .supervision_gate_enforced_with_context(
                    "uncertainty_clarify_gate",
                    model,
                    emitter,
                    task_id,
                    iteration,
                    json!({
                        "reason": "ambiguous_side_effect_without_concrete_target",
                        "proposed_tool": unresolved_side_effecting_call.map(|tc| tc.name.as_str()),
                        "uncertainty_score_advisory": policy_bundle.uncertainty_score,
                        "threshold_advisory": uncertainty_threshold,
                    }),
                )
                .await
        {
            let clarify = default_clarifying_question(user_text, &[]);
            POLICY_METRICS
                .uncertainty_clarify_total
                .fetch_add(1, Ordering::Relaxed);
            info!(
                session_id,
                iteration,
                uncertainty_score = policy_bundle.uncertainty_score,
                threshold = uncertainty_threshold,
                tool = unresolved_side_effecting_call.map(|tc| tc.name.as_str()),
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
            agent
                .append_assistant_message_with_event(emitter, &assistant_msg, "system", None, None)
                .await?;
            agent
                .emit_task_end(
                    emitter,
                    task_id,
                    TaskStatus::Completed,
                    TaskOutcome::Partial,
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

    if let Some(tracker) = project_instruction_tracker.as_mut() {
        let mut instruction_targets = Vec::new();
        for tool_call in &resp.tool_calls {
            let call_semantics = agent
                .tools
                .iter()
                .find(|tool| tool.name() == tool_call.name && tool.is_available())
                .map(|tool| tool.call_semantics(&tool_call.arguments))
                .unwrap_or_default();
            for target in super::tool_execution_phase::project_instruction_targets_for_tool_call(
                &tool_call.name,
                &tool_call.arguments,
                &call_semantics,
            ) {
                if !instruction_targets.contains(&target) {
                    instruction_targets.push(target);
                }
            }
        }

        if !instruction_targets.is_empty() {
            match tracker
                .discover_for_targets(
                    crate::execution::active_execution_backend(),
                    &instruction_targets,
                )
                .await
            {
                Ok(Some(instructions)) => {
                    let source_paths = instructions.source_paths();
                    if !task_context_tail.is_empty() {
                        task_context_tail.push_str("\n\n");
                    }
                    task_context_tail.push_str(
                        "[Just-in-time Project Instructions — loaded before subtree work]\n",
                    );
                    task_context_tail.push_str(&instructions.render_for_prompt());
                    info!(
                        session_id,
                        iteration,
                        instruction_targets = ?instruction_targets,
                        instruction_sources = ?source_paths,
                        "Deferred tool batch to load nested project instructions"
                    );
                    agent
                        .emit_decision_point(
                            emitter,
                            task_id,
                            iteration,
                            DecisionType::InstructionsSnapshot,
                            "Loaded nested project instructions before tool execution".to_string(),
                            json!({
                                "condition": "jit_project_instructions_discovered",
                                "targets": &instruction_targets,
                                "sources": &source_paths,
                                "action_executed": false,
                            }),
                        )
                        .await;
                    defer_prelude_for_new_project_instructions(
                        agent,
                        emitter,
                        session_id,
                        task_id,
                        &resp.tool_calls,
                        &source_paths,
                    )
                    .await?;
                    return Ok(ToolPreludeOutcome::ContinueLoop);
                }
                Ok(None) => {}
                Err(error) => {
                    warn!(
                        session_id,
                        iteration,
                        instruction_targets = ?instruction_targets,
                        %error,
                        "Could not perform just-in-time project instruction discovery"
                    );
                }
            }
        }
    }

    for tc in &resp.tool_calls {
        if let Some(violation) =
            assess_pre_execution_evidence_gate(&tc.name, &tc.arguments, evidence_state)
        {
            if !agent
                .supervision_gate_enforced(
                    "pre_execution_evidence_gate",
                    model,
                    emitter,
                    task_id,
                    iteration,
                )
                .await
            {
                // Autonomous tier: telemetry only — let the tool run and the
                // tool's own error surface teach instead of a harness bounce.
                continue;
            }
            agent
                .with_harness_eval(|eval| eval.record_evidence_gate_block())
                .await;
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
            agent
                .emit_warning_decision_point(
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
            inject_prelude_retry_messages(
                agent,
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
        first_side_effecting_tool_call(agent, resp, available_capabilities).cloned();
    // Gate: only request the pre-execution plan once per task.
    // Skip if: (a) a side-effecting tool already completed successfully
    // (learning_ctx tracks it), OR (b) a plan was already accepted
    // (current_plan_version is set).  Without check (b), a critique
    // rejection causes ContinueLoop without executing the tool, so the
    // learning_ctx guard stays false and the plan is re-requested every
    // iteration — adding 2 extra LLM calls (plan + critique) per loop.
    let plan_already_generated = execution_state.current_plan_version.is_some();
    if agent.mandate_execution.is_none()
        && agent.depth == 0
        && !plan_already_generated
        && !has_completed_side_effecting_tool_call(learning_ctx, available_capabilities)
    {
        if let Some(first_risky_tool_call) = first_risky_tool_call {
            if should_run_pre_execution_gating(&first_risky_tool_call)
                && agent
                    .supervision_gate_enforced_with_context(
                        "pre_execution_planning",
                        model,
                        emitter,
                        task_id,
                        iteration,
                        json!({
                            "proposed_tool": first_risky_tool_call.name.as_str(),
                            "target": extract_target_preview(&first_risky_tool_call.arguments),
                        }),
                    )
                    .await
            {
                let capabilities = available_capabilities
                    .get(&first_risky_tool_call.name)
                    .copied()
                    .unwrap_or_default();
                let expected_target = extract_target_preview(&first_risky_tool_call.arguments);

                // Pre-execution planning is a system-initiated quality check,
                // not an agent action. Do not charge it against the execution
                // budget — the agent should not be penalised for the system's
                // own safety overhead.
                match request_pre_execution_plan(
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
                                agent
                                    .with_harness_eval(|eval| {
                                        eval.record_plan_progress(
                                            0,
                                            plan.planned_steps.len() as u32,
                                        )
                                    })
                                    .await;
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
                            agent
                                .emit_decision_point(
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
                                match request_pre_execution_critique(
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
                                                agent.emit_decision_point(
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
                                                agent
                                                    .with_harness_eval(|eval| {
                                                        eval.record_critique_replan()
                                                    })
                                                    .await;
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
                                                agent.emit_warning_decision_point(
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
                                                inject_prelude_retry_messages(
                agent,
                                                emitter,
                                                session_id,
                                                task_id,
                                                &resp.tool_calls,
                                                format!(
                                                    "[SYSTEM] Critique pass blocked this risky action. \
                                                     Issues: {}. Re-plan the first action, gather any missing \
                                                     evidence, briefly explain the corrected approach, and then \
                                                     re-issue tool calls. (SYSTEM NOTE: this rejection came from your own internal safety guardrail, not the external tool. Do not tell the user the tool failed or rejected the input.)",
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
                                                agent
                                                .emit_warning_decision_point(
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
                                        agent
                                            .emit_warning_decision_point(
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
                                agent.emit_warning_decision_point(
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
                            agent.emit_warning_decision_point(
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
                            inject_prelude_retry_messages(
                agent,
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
                        agent
                            .emit_warning_decision_point(
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
    }

    Ok(ToolPreludeOutcome::Proceed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: format!("call-{name}"),
            name: name.to_string(),
            arguments: arguments.to_string(),
            extra_content: None,
        }
    }

    #[test]
    fn resource_handles_are_concrete_structural_targets() {
        assert!(arguments_have_concrete_target(
            r#"{"resource_id":"res_exact"}"#
        ));
        assert!(!arguments_have_concrete_target(r#"{"resource_id":"   "}"#));
        assert!(!arguments_have_concrete_target(r#"{"caption":"send it"}"#));
    }

    #[test]
    fn scoped_deploy_restriction_allows_local_build_but_blocks_deployment() {
        let contract = CompletionContract {
            expects_mutation: true,
            forbidden_mutation_actions: vec![ForbiddenMutationAction::Deploy],
            ..CompletionContract::default()
        };
        let write = tool_call(
            "write_file",
            r#"{"path":"/tmp/site/package.json","content":"{\"scripts\":{\"deploy\":\"wrangler deploy\"}}"}"#,
        );
        let build = tool_call("terminal", r#"{"action":"run","command":"npm run build"}"#);
        let inspect = tool_call("terminal", r#"{"action":"run","command":"rg deploy ."}"#);
        let deploy = tool_call(
            "terminal",
            r#"{"action":"run","command":"npx wrangler deploy"}"#,
        );

        assert_eq!(
            negative_contract_block_reason(&contract, &write, true),
            None
        );
        assert_eq!(
            negative_contract_block_reason(&contract, &build, true),
            None
        );
        assert_eq!(
            negative_contract_block_reason(&contract, &inspect, true),
            None,
            "mentioning deployment in an observation is not a deployment action"
        );
        assert_eq!(
            negative_contract_block_reason(&contract, &deploy, true).as_deref(),
            Some("deploy")
        );
    }

    #[test]
    fn global_read_only_contract_still_blocks_every_side_effect() {
        let contract = CompletionContract {
            forbids_mutation: true,
            ..CompletionContract::default()
        };
        let build = tool_call("terminal", r#"{"action":"run","command":"npm run build"}"#);
        assert_eq!(
            negative_contract_block_reason(&contract, &build, true).as_deref(),
            Some("all mutation")
        );

        let delegated = tool_call(
            "spawn_agent",
            r#"{"mission":"Implement the change","task":"Edit files and deploy it"}"#,
        );
        assert_eq!(
            negative_contract_block_reason(&contract, &delegated, false).as_deref(),
            Some("all mutation")
        );
    }

    #[test]
    fn scoped_and_global_negative_contract_matches_remain_distinguishable() {
        let delegated_post = tool_call(
            "spawn_agent",
            r#"{"mission":"Publish one update","task":"POST exactly once, then verify"}"#,
        );
        let scoped = CompletionContract {
            expects_mutation: true,
            forbidden_mutation_actions: vec![ForbiddenMutationAction::Post],
            ..CompletionContract::default()
        };
        assert_eq!(
            negative_contract_block_reason(&scoped, &delegated_post, false).as_deref(),
            Some("post")
        );

        let read_only = CompletionContract {
            forbids_mutation: true,
            ..CompletionContract::default()
        };
        assert_eq!(
            negative_contract_block_reason(&read_only, &delegated_post, false).as_deref(),
            Some("all mutation"),
            "an explicit global read-only contract is still separately identifiable"
        );
    }

    #[test]
    fn scoped_restriction_follows_delegated_work_without_disabling_other_work() {
        let contract = CompletionContract {
            expects_mutation: true,
            forbidden_mutation_actions: vec![ForbiddenMutationAction::Deploy],
            ..CompletionContract::default()
        };
        let build = tool_call(
            "spawn_agent",
            r#"{"mission":"Build the site","task":"Create and test the site locally"}"#,
        );
        let deploy = tool_call(
            "spawn_agent",
            r#"{"mission":"Ship the site","task":"Deploy the site and return its URL"}"#,
        );

        assert_eq!(
            negative_contract_block_reason(&contract, &build, true),
            None
        );
        assert_eq!(
            negative_contract_block_reason(&contract, &deploy, true).as_deref(),
            Some("deploy")
        );

        let turn = TurnContext {
            goal_user_text: "Build the site locally but do not deploy it.".to_string(),
            completion_contract: contract,
            ..TurnContext::default()
        };
        assert!(!turn_prefers_plain_text_completion(&turn));
    }

    fn turn_with(
        goal_user_text: &str,
        recent_messages: Vec<Value>,
        expects_mutation: bool,
        requires_observation: bool,
    ) -> TurnContext {
        TurnContext {
            goal_user_text: goal_user_text.to_string(),
            recent_messages,
            completion_contract: CompletionContract {
                expects_mutation,
                requires_observation,
                ..CompletionContract::default()
            },
            ..TurnContext::default()
        }
    }

    #[test]
    fn approved_proposal_text_returns_the_proposal_on_short_approval() {
        // "Would you like me to read X?" → "Yes, read the Right one" must
        // anchor the turn to the proposal so the model executes it instead
        // of drifting into re-searching.
        let turn = turn_with(
            "Yes, read the Right one",
            vec![assistant_proposal_msg()],
            false,
            false,
        );
        let proposal = super::approved_proposal_text(&turn).expect("approval should anchor");
        assert!(proposal.contains("Would you like me to"));
    }

    #[test]
    fn approved_proposal_text_ignores_non_approval_turns() {
        let question = turn_with(
            "what's the weather like?",
            vec![assistant_proposal_msg()],
            false,
            false,
        );
        assert!(super::approved_proposal_text(&question).is_none());

        let no_proposal = turn_with("yes", vec![], false, false);
        assert!(super::approved_proposal_text(&no_proposal).is_none());
    }

    #[test]
    fn inferred_text_only_request_does_not_hard_redirect_tools() {
        // A guessed answer/check classification is advisory. Only an explicit
        // user constraint such as "answer only" may hard-disable tools.
        let turn = turn_with(
            "Can you read the file and tell me what it's about? Offer Letter (1).pdf",
            vec![],
            false,
            false,
        );
        assert!(!super::text_only_redirect_applies(&turn, 0));
        assert!(!super::text_only_redirect_applies(&turn, 3));
    }

    #[test]
    fn read_only_terminal_command_exempt_from_plain_text_redirect() {
        // "Can you read the file and tell me what it's about? <file>.pdf" is
        // not interrogative by shape (doesn't end with '?'), but the model's
        // `find` fallback is a pure lookup — blocking it forces fabrication.
        let turn = turn_with(
            "Can you read the file and tell me what it's about? Offer Letter (1).pdf",
            vec![],
            false,
            false,
        );
        let arguments =
            r#"{"action": "run", "command": "find ~ -name \"*Offer Letter*.pdf\" 2>/dev/null"}"#;
        assert!(super::plain_text_redirect_exempts_lookup(
            "terminal", arguments, &turn
        ));
        assert!(super::plain_text_redirect_exempts_lookup(
            "run_command",
            r#"{"command": "find ~ -name \"*Offer Letter*.pdf\" 2>/dev/null"}"#,
            &turn
        ));
    }

    #[test]
    fn mutating_terminal_command_still_redirected_on_text_only_turn() {
        let turn = turn_with("write me a short poem about rust", vec![], false, false);
        let arguments = r#"{"action": "run", "command": "rm -rf ~/tmp/scratch"}"#;
        assert!(!super::plain_text_redirect_exempts_lookup(
            "terminal", arguments, &turn
        ));
    }

    #[test]
    fn malformed_terminal_arguments_not_exempt() {
        let turn = turn_with("write me a short poem about rust", vec![], false, false);
        assert!(!super::plain_text_redirect_exempts_lookup(
            "terminal", "not-json", &turn
        ));
    }

    fn assistant_proposal_msg() -> Value {
        json!({
            "role": "assistant",
            "content": "I found a PDF. Would you like me to try running a Python script to extract the text block-by-block?"
        })
    }

    #[test]
    fn test_should_run_pre_execution_gating() {
        // Safe terminal command -> should NOT run gating
        let safe_tc = ToolCall {
            id: "tc_1".to_string(),
            name: "terminal".to_string(),
            arguments: r#"{"action": "run", "command": "relevo --help"}"#.to_string(),
            extra_content: None,
        };
        assert!(!super::should_run_pre_execution_gating(&safe_tc));

        // Critical terminal command -> SHOULD run gating
        let crit_tc = ToolCall {
            id: "tc_2".to_string(),
            name: "terminal".to_string(),
            arguments: r#"{"action": "run", "command": "rm -rf /"}"#.to_string(),
            extra_content: None,
        };
        assert!(super::should_run_pre_execution_gating(&crit_tc));

        // Malformed arguments -> SHOULD run gating (fail safe)
        let malformed_tc = ToolCall {
            id: "tc_3".to_string(),
            name: "terminal".to_string(),
            arguments: r#"{"action": "run"}"#.to_string(), // missing command
            extra_content: None,
        };
        assert!(super::should_run_pre_execution_gating(&malformed_tc));

        // Non-run action -> SHOULD run gating
        let check_tc = ToolCall {
            id: "tc_4".to_string(),
            name: "terminal".to_string(),
            arguments: r#"{"action": "check", "pid": 1234}"#.to_string(),
            extra_content: None,
        };
        assert!(super::should_run_pre_execution_gating(&check_tc));

        // Non-terminal tool -> SHOULD run gating
        let other_tc = ToolCall {
            id: "tc_5".to_string(),
            name: "http_request".to_string(),
            arguments: r#"{}"#.to_string(),
            extra_content: None,
        };
        assert!(super::should_run_pre_execution_gating(&other_tc));
    }

    #[test]
    fn plain_text_gate_allows_tools_for_approval_of_prior_proposal() {
        // Regression case: bare affirmation in response to an assistant proposal.
        // Even though the contract is text-only, tools must remain allowed.
        let tc = turn_with(
            "Yes, try that",
            vec![
                json!({"role": "user", "content": "Can you extract the text from this PDF?"}),
                assistant_proposal_msg(),
            ],
            false,
            false,
        );
        assert!(!turn_prefers_plain_text_completion(&tc));
    }

    #[test]
    fn plain_question_does_not_hard_disable_tools() {
        let tc = turn_with(
            "what's my name?",
            vec![assistant_proposal_msg()],
            false,
            false,
        );
        assert!(!turn_prefers_plain_text_completion(&tc));
    }

    #[test]
    fn explicit_answer_only_constraint_hard_disables_tools() {
        let tc = turn_with("Answer only; don't change anything.", vec![], false, false);
        assert!(turn_prefers_plain_text_completion(&tc));
    }

    #[test]
    fn lookup_exemption_spares_terminal_on_interrogative_turn() {
        // "How many users?" reaches for terminal/drush to observe state. The
        // plain-text redirect must spare it so the model can run the lookup
        // instead of fabricating an answer.
        let tc = turn_with("How many users?", vec![], false, false);
        assert!(turn_is_interrogative_lookup(&tc));
        assert!(plain_text_redirect_exempts_lookup(
            "terminal",
            r#"{"action": "run", "command": "drush user:list"}"#,
            &tc
        ));
    }

    #[test]
    fn lookup_exemption_spares_any_question_variants() {
        for q in [
            "Who are admin users?",
            "Any blocked/inactive users?",
            "Which modules are enabled?",
            "List the active sessions",
        ] {
            let tc = turn_with(q, vec![], false, false);
            assert!(
                plain_text_redirect_exempts_lookup(
                    "terminal",
                    r#"{"action": "run", "command": "drush user:list"}"#,
                    &tc
                ),
                "terminal lookup should be spared for: {q:?}"
            );
        }
    }

    #[test]
    fn lookup_exemption_does_not_spare_terminal_on_non_question() {
        // A non-interrogative turn drifting into terminal is still redirected
        // (anti-drift protection preserved). Destructive commands also remain
        // gated by command-risk approval downstream.
        let tc = turn_with("Write me a poem about cats.", vec![], false, false);
        assert!(!turn_is_interrogative_lookup(&tc));
        assert!(!plain_text_redirect_exempts_lookup(
            "terminal",
            r#"{"action": "run", "command": "npm run build"}"#,
            &tc
        ));
    }

    #[test]
    fn lookup_exemption_always_spares_pure_read_tools() {
        // Pure reads never irreversibly mutate — exempt regardless of phrasing.
        let tc = turn_with("Summarize the latest news.", vec![], false, false);
        assert!(!turn_is_interrogative_lookup(&tc));
        for tool in ["read_file", "search_files", "web_search", "web_fetch"] {
            assert!(
                plain_text_redirect_exempts_lookup(tool, "{}", &tc),
                "pure-read tool should always be spared: {tool}"
            );
        }
    }

    #[test]
    fn bare_yes_without_proposal_does_not_infer_a_hard_tool_ban() {
        let tc = turn_with(
            "yes",
            vec![
                json!({"role": "user", "content": "Tell me about cats."}),
                json!({"role": "assistant", "content": "Cats are small domesticated mammals."}),
            ],
            false,
            false,
        );
        assert!(!turn_prefers_plain_text_completion(&tc));
    }

    #[test]
    fn is_short_affirmation_or_approval_positive() {
        assert!(is_short_affirmation_or_approval("Yes, try that"));
        assert!(is_short_affirmation_or_approval("go ahead"));
        assert!(is_short_affirmation_or_approval("do it"));
        assert!(is_short_affirmation_or_approval("sure"));
        assert!(is_short_affirmation_or_approval("yes please"));
        assert!(is_short_affirmation_or_approval("please proceed"));
        assert!(is_short_affirmation_or_approval("sgtm"));
        assert!(is_short_affirmation_or_approval("OK!"));
    }

    #[test]
    fn is_short_affirmation_or_approval_negative() {
        assert!(!is_short_affirmation_or_approval(""));
        assert!(!is_short_affirmation_or_approval("what is the weather?"));
        assert!(!is_short_affirmation_or_approval(
            "yes can you also generate a chart and email it to my whole team tomorrow morning"
        ));
        assert!(!is_short_affirmation_or_approval("no don't"));
    }

    #[test]
    fn assistant_proposed_action_positive() {
        assert!(assistant_proposed_action(&[json!({
            "role": "assistant",
            "content": "Would you like me to run that for you?"
        })]));
        assert!(assistant_proposed_action(&[json!({
            "role": "assistant",
            "content": "I can run the extraction script if you want."
        })]));
        assert!(assistant_proposed_action(&[json!({
            "role": "assistant",
            "content": "Should I create the file?"
        })]));
    }

    #[test]
    fn assistant_proposed_action_negative() {
        assert!(!assistant_proposed_action(&[json!({
            "role": "assistant",
            "content": "Cats are small domesticated mammals."
        })]));
        // No assistant message at all.
        assert!(!assistant_proposed_action(&[json!({
            "role": "user",
            "content": "Would you like me to run that?"
        })]));
    }

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

    #[test]
    fn plain_text_guard_allows_internal_delegation() {
        assert!(!plain_text_guard_blocks_tool("spawn_agent", true));
        assert!(plain_text_guard_blocks_tool("terminal", true));
        assert!(!plain_text_guard_blocks_tool("read_file", false));
        assert!(!uncertainty_guard_blocks_tool("spawn_agent", true));
        assert!(uncertainty_guard_blocks_tool("terminal", true));
    }
}
