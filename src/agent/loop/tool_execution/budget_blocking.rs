use crate::agent::trust_tier::ModelTrustTier;
use crate::agent::*;
use crate::execution_policy::PolicyBundle;

/// Per-task tool-call budget caps. These are runaway backstops, not pacing:
/// the repetition guards (same-call hash, consecutive same-tool) catch true
/// loops on every tier. Autonomous-tier models get research-scale caps —
/// twenty distinct web searches is investigation, not a loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolBudgetCaps {
    pub web_search: usize,
    pub web_fetch: usize,
    pub combined_web: usize,
    pub spawn_agent: usize,
    pub http_request: usize,
    pub computer_use: usize,
    pub generic: usize,
}

pub(crate) fn tool_budget_caps(tier: ModelTrustTier) -> ToolBudgetCaps {
    let guided = ToolBudgetCaps {
        web_search: 5,
        web_fetch: 6,
        combined_web: 10,
        spawn_agent: 15,
        http_request: 15,
        computer_use: 40,
        generic: 8,
    };
    match tier {
        ModelTrustTier::Guided => guided,
        ModelTrustTier::Autonomous => ToolBudgetCaps {
            web_search: guided.web_search * 3,
            web_fetch: guided.web_fetch * 3,
            combined_web: guided.combined_web * 3,
            spawn_agent: guided.spawn_agent * 3,
            http_request: guided.http_request * 3,
            computer_use: guided.computer_use * 3,
            generic: guided.generic * 3,
        },
    }
}

/// Tools exempt from the generic per-turn call budget: high-frequency local
/// work tools, MCP tools (`prefix__name`), and `manage_goal_tasks` — task-lead
/// bookkeeping that legitimately needs list + claim/complete calls per task.
fn generic_budget_exempt(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "terminal"
            | "cli_agent"
            | "computer_use"
            | "read_file"
            | "edit_file"
            | "write_file"
            | "search_files"
            | "remember_fact"
            | "manage_memories"
            | "web_fetch"
            | "manage_goal_tasks"
    ) || tool_name.contains("__")
}

#[cfg(test)]
mod caps_tests {
    use super::*;

    #[test]
    fn guided_caps_match_legacy_constants() {
        let caps = tool_budget_caps(ModelTrustTier::Guided);
        assert_eq!(caps.web_search, 5);
        assert_eq!(caps.web_fetch, 6);
        assert_eq!(caps.combined_web, 10);
        assert_eq!(caps.spawn_agent, 15);
        assert_eq!(caps.http_request, 15);
        assert_eq!(caps.computer_use, 40);
        assert_eq!(caps.generic, 8);
    }

    #[test]
    fn autonomous_caps_scale_to_research_volume() {
        let guided = tool_budget_caps(ModelTrustTier::Guided);
        let auto = tool_budget_caps(ModelTrustTier::Autonomous);
        assert_eq!(auto.web_search, 15);
        assert_eq!(auto.web_fetch, 18);
        assert_eq!(auto.combined_web, 30);
        assert_eq!(auto.spawn_agent, 45);
        assert_eq!(auto.http_request, 45);
        assert_eq!(auto.computer_use, 120);
        assert_eq!(auto.generic, 24);
        for (a, g) in [
            (auto.web_search, guided.web_search),
            (auto.web_fetch, guided.web_fetch),
            (auto.combined_web, guided.combined_web),
            (auto.spawn_agent, guided.spawn_agent),
            (auto.http_request, guided.http_request),
            (auto.computer_use, guided.computer_use),
            (auto.generic, guided.generic),
        ] {
            assert!(a > g, "every autonomous cap must exceed its guided cap");
        }
    }
}

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
    pub model: &'a str,
    pub iteration: usize,
    pub tool_failure_count: &'a HashMap<String, usize>,
    pub tool_transient_failure_count: &'a HashMap<String, usize>,
    /// Per-(tool, signature) failure counts for the task. Summed per tool to get
    /// the aggregate failure count, which catches loops where a tool keeps
    /// failing with *different* error signatures (so the per-signature limit
    /// never trips) — e.g. send_file retried on varying rejected paths.
    pub tool_failure_signatures: &'a HashMap<(String, String), usize>,
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

/// A single tool that has failed this many times in a task (across *any* error
/// signatures) is considered stuck in a retry loop and is cooled down, even when
/// each individual signature stays below the per-signature limit. Set above
/// normal multi-step retry counts so legitimate retry-heavy flows aren't caught.
const AGGREGATE_TOOL_FAILURE_LIMIT: usize = 6;

/// Sum of all signature-failure counts for `tool` (aggregate failures for the
/// tool across differing error messages).
pub(super) fn aggregate_tool_failures(
    signatures: &HashMap<(String, String), usize>,
    tool: &str,
) -> usize {
    signatures
        .iter()
        .filter(|((t, _), _)| t == tool)
        .map(|(_, c)| *c)
        .sum()
}

pub(super) async fn maybe_block_tool_by_budget(
    agent: &Agent,
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
    let aggregate_failures = aggregate_tool_failures(ctx.tool_failure_signatures, &tc.name);

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
            agent
                .append_tool_message_with_result_event(
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

    let trust_tier = agent.trust_tier_for_model(ctx.model);
    let caps = tool_budget_caps(trust_tier);
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
    } else if aggregate_failures >= AGGREGATE_TOOL_FAILURE_LIMIT {
        // Stuck in a loop: this tool keeps failing with differing error messages
        // (so the per-signature limit never trips), often interleaved with other
        // tools' successes (so stall_count keeps resetting). Cool it down.
        Some(
            ToolResultNotice::SemanticErrorLimitBlocked {
                tool_name: tc.name.clone(),
                prior_signature_failures: aggregate_failures,
                prior_transient_failures,
            }
            .render(),
        )
    } else if tc.name == "web_search" && prior_calls >= caps.web_search {
        Some(ToolResultNotice::WebSearchBudgetBlocked { prior_calls }.render())
    } else if (tc.name == "web_search" || tc.name == "web_fetch")
        && combined_web_calls >= caps.combined_web
    {
        Some(ToolResultNotice::CombinedWebBudgetBlocked { combined_web_calls }.render())
    } else if tc.name == "web_fetch" && prior_calls >= caps.web_fetch {
        Some(ToolResultNotice::WebFetchBudgetBlocked { prior_calls }.render())
    } else if tc.name == "spawn_agent" && prior_calls >= caps.spawn_agent {
        // spawn_agent gets a higher cap than generic tools since task leads
        // legitimately spawn many executors, but it must still be bounded
        // to prevent runaway LLM agent spawns.
        Some(ToolResultNotice::SpawnAgentBudgetBlocked { prior_calls }.render())
    } else if tc.name == "http_request" && prior_calls >= caps.http_request {
        // http_request gets a higher cap than generic tools because
        // multi-step API workflows (posting threads, paginated APIs,
        // auth checks + retries) legitimately need many sequential calls.
        Some(
            ToolResultNotice::GenericToolBudgetBlocked {
                tool_name: tc.name.clone(),
                prior_calls,
            }
            .render(),
        )
    } else if tc.name == "computer_use" && prior_calls >= caps.computer_use {
        // computer_use drives multi-step GUI flows where each step is a separate
        // call (get_app_state observation + click/type mutation), so a healthy
        // 5–15 step task legitimately makes dozens of calls. The tool enforces
        // its own meaningful-progress budgets internally (max_mutating_actions,
        // max_consecutive_observations); this is only a runaway backstop set far
        // above any real GUI task.
        Some(
            ToolResultNotice::GenericToolBudgetBlocked {
                tool_name: tc.name.clone(),
                prior_calls,
            }
            .render(),
        )
    } else if prior_calls >= caps.generic && !generic_budget_exempt(&tc.name) {
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

    agent
        .persist_gate_fire(
            ctx.emitter,
            ctx.task_id,
            ctx.iteration,
            "tool_budget_block",
            ctx.model,
            trust_tier,
            crate::agent::heuristic_telemetry::HeuristicAction::Enforced,
        )
        .await;

    warn!(
        session_id = %ctx.session_id,
        task_id = %ctx.task_id,
        iteration = ctx.iteration,
        tool = %tc.name,
        semantic_failures = prior_signature_failures,
        transient_failures = prior_transient_failures,
        calls = prior_calls,
        "Blocking repeated tool call"
    );
    agent
        .emit_warning_decision_point(
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

    agent
        .append_tool_message_with_result_event(
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
    agent: &Agent,
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
    agent
        .append_tool_message_with_result_event(
            ctx.emitter,
            &tool_msg,
            true,
            0,
            None,
            Some(ctx.task_id),
        )
        .await?;

    if let Some(ref tid) = agent.task_id {
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
        if let Err(e) = agent.state.log_task_activity(&activity).await {
            warn!(task_id = %tid, error = %e, "Failed to log task activity");
        }
    }

    Ok(true)
}

#[cfg(test)]
mod aggregate_failure_tests {
    use super::*;

    #[test]
    fn aggregate_sums_failures_across_differing_signatures() {
        // The send_file loop: same tool, different error signatures (varying
        // rejected paths), each below the per-signature limit but summing high.
        let mut sigs: HashMap<(String, String), usize> = HashMap::new();
        sigs.insert(("send_file".into(), "outside:/tmp/a".into()), 1);
        sigs.insert(("send_file".into(), "outside:/tmp/b".into()), 1);
        sigs.insert(("send_file".into(), "outside:/tmp/c".into()), 1);
        sigs.insert(("terminal".into(), "exit-1".into()), 2);
        assert_eq!(aggregate_tool_failures(&sigs, "send_file"), 3);
        assert_eq!(aggregate_tool_failures(&sigs, "terminal"), 2);
        assert_eq!(aggregate_tool_failures(&sigs, "read_file"), 0);
    }

    #[test]
    fn aggregate_limit_trips_only_at_threshold() {
        let mut sigs: HashMap<(String, String), usize> = HashMap::new();
        for i in 0..AGGREGATE_TOOL_FAILURE_LIMIT {
            sigs.insert(("send_file".into(), format!("sig-{i}")), 1);
        }
        assert!(
            aggregate_tool_failures(&sigs, "send_file") >= AGGREGATE_TOOL_FAILURE_LIMIT,
            "should reach the aggregate block threshold"
        );
        // One fewer is below threshold.
        sigs.remove(&("send_file".into(), "sig-0".into()));
        assert!(aggregate_tool_failures(&sigs, "send_file") < AGGREGATE_TOOL_FAILURE_LIMIT);
    }
}

#[cfg(test)]
mod exemption_tests {
    use super::*;

    #[test]
    fn task_lead_bookkeeping_tool_is_exempt_from_generic_budget() {
        // A task lead managing several tasks legitimately needs more than
        // the generic 8 calls (list + claim/complete per task).
        assert!(generic_budget_exempt("manage_goal_tasks"));
    }

    #[test]
    fn ordinary_tools_still_hit_the_generic_budget() {
        assert!(!generic_budget_exempt("manage_people"));
        assert!(!generic_budget_exempt("project_inspect"));
    }

    #[test]
    fn legacy_exemptions_preserved() {
        for tool in [
            "terminal",
            "cli_agent",
            "computer_use",
            "read_file",
            "edit_file",
            "write_file",
            "search_files",
            "remember_fact",
            "manage_memories",
            "web_fetch",
        ] {
            assert!(generic_budget_exempt(tool), "{tool} must stay exempt");
        }
        // MCP tools (prefix__name) are exempt too.
        assert!(generic_budget_exempt("github__list_issues"));
    }
}
