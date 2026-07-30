use super::memory_scope::{is_low_signal_goal_text, scope_goal_memory_to_project_hints};
use super::types::OrchestrationCtx;
use crate::agent::response_phase::ResponsePhaseOutcome;
use crate::agent::*;

pub(super) async fn build_goal_feed_forward_context(
    agent: &Agent,
    session_id: &str,
    goal_user_text: &str,
    recent_messages: &[Value],
    project_hints: &[String],
) -> Option<String> {
    let low_signal_without_hints =
        project_hints.is_empty() && is_low_signal_goal_text(goal_user_text);

    let (raw_facts, raw_procedures) = if low_signal_without_hints {
        info!(
            session_id,
            "Skipping goal memory retrieval: low-signal goal text without project hints"
        );
        (Vec::new(), Vec::new())
    } else {
        let memory_query = if project_hints.is_empty() {
            goal_user_text.to_string()
        } else {
            format!(
                "{goal_user_text}\n\nProject context: {}",
                project_hints.join(" ")
            )
        };
        (
            agent
                .state
                .get_relevant_facts(&memory_query, 10)
                .await
                .unwrap_or_default(),
            agent
                .state
                .get_relevant_procedures(&memory_query, 5)
                .await
                .unwrap_or_default(),
        )
    };

    let (relevant_facts, relevant_procedures) =
        scope_goal_memory_to_project_hints(raw_facts, raw_procedures, project_hints);

    if !project_hints.is_empty() && (relevant_facts.is_empty() || relevant_procedures.is_empty()) {
        info!(
            session_id,
            project_hints = ?project_hints,
            facts = relevant_facts.len(),
            procedures = relevant_procedures.len(),
            "Scoped goal memory to project hints"
        );
    }

    if relevant_facts.is_empty()
        && relevant_procedures.is_empty()
        && recent_messages.is_empty()
        && project_hints.is_empty()
    {
        return None;
    }

    let ctx = json!({
        "relevant_facts": relevant_facts.iter().map(|f| {
            json!({"category": f.category, "key": f.key, "value": f.value})
        }).collect::<Vec<_>>(),
        "relevant_procedures": relevant_procedures.iter().map(|p| {
            json!({"name": p.name, "trigger": p.trigger_pattern, "steps": p.steps})
        }).collect::<Vec<_>>(),
        "recent_messages": recent_messages,
        "project_hints": project_hints,
        "task_results": [],
    });
    Some(serde_json::to_string(&ctx).unwrap_or_default())
}

pub(in crate::agent) async fn run_orchestration_phase(
    services: &crate::agent::services::AgentServices<'_>,
    ctx: &mut OrchestrationCtx<'_>,
) -> anyhow::Result<Option<ResponsePhaseOutcome>> {
    let agent = services.agent;
    if let Some(outcome) = super::routes::maybe_handle_generic_cancel_request(agent, ctx).await? {
        record_orchestration_direct_return(agent, &outcome).await;
        return Ok(Some(outcome));
    }

    // Orchestration routing (always-on).
    let complexity = classify_intent_complexity(ctx.user_text);
    let (route, _) = orchestration_route_label(&complexity);
    let tools_required = ctx.execution_requirement.requires_execution();
    let tool_control_enforced = agent.trust_tier_for_model(ctx.model)
        == crate::agent::trust_tier::ModelTrustTier::Guided
        && ctx.execution_requirement.requires_execution();
    let requirement_reasons = ctx.execution_requirement.reason_codes();

    // Per-turn intent-classification telemetry: record the gate result + complexity
    // + route for every (non-cancel) turn so the determination is queryable via the
    // event store. Gated internally by `record_decision_points`; severity Info.
    let (intent_summary, intent_metadata) = intent_decision_telemetry(
        tools_required,
        &complexity,
        route,
        ctx.user_text.chars().count(),
        tool_control_enforced,
        &requirement_reasons,
    );
    agent
        .emit_decision_point(
            ctx.emitter,
            ctx.task_id,
            ctx.iteration,
            crate::events::DecisionType::IntentGate,
            intent_summary,
            intent_metadata,
        )
        .await;

    if agent.harness_eval_enabled() {
        agent
            .with_harness_eval(|eval| eval.record_orchestration_route(route, tools_required))
            .await;
    }
    let outcome = super::routes::route_orchestration_complexity(agent, ctx, complexity).await?;
    record_orchestration_direct_return(agent, &outcome).await;
    Ok(Some(outcome))
}

/// Build the per-turn intent-classification telemetry (summary + metadata) for a
/// `DecisionType::IntentGate` decision point. Pure formatting so it is unit-testable;
/// the emission itself is gated by `record_decision_points` inside `emit_decision_point`.
fn intent_decision_telemetry(
    needs_tools: bool,
    complexity: &IntentComplexity,
    route: &str,
    user_text_len: usize,
    tool_control_enforced: bool,
    requirement_reasons: &[&str],
) -> (String, Value) {
    let complexity_label = match complexity {
        IntentComplexity::Simple => "simple",
        IntentComplexity::Complex => "complex",
        IntentComplexity::Scheduled { .. } => "scheduled",
        IntentComplexity::ScheduledMissingTiming => "scheduled_missing_timing",
    };
    let summary = format!(
        "Intent classified: needs_tools={needs_tools} complexity={complexity_label} route={route}"
    );
    let metadata = json!({
        "condition": "intent_classification",
        "needs_tools": needs_tools,
        "complexity": complexity_label,
        "route": route,
        "user_text_len": user_text_len,
        "tool_control_enforced": tool_control_enforced,
        "requirement_reasons": requirement_reasons,
    });
    (summary, metadata)
}

fn orchestration_route_label(complexity: &IntentComplexity) -> (&'static str, bool) {
    match complexity {
        IntentComplexity::ScheduledMissingTiming => ("clarification_required", false),
        IntentComplexity::Scheduled { .. } => ("direct_reply", false),
        IntentComplexity::Simple => ("default_continue", false),
        IntentComplexity::Complex => ("tools_required", true),
    }
}

async fn record_orchestration_direct_return(agent: &Agent, outcome: &ResponsePhaseOutcome) {
    if !agent.harness_eval_enabled() {
        return;
    }
    if let ResponsePhaseOutcome::Return(result) = outcome {
        agent
            .with_harness_eval(|eval| eval.record_direct_return(true, result.is_ok()))
            .await;
    }
}

#[cfg(test)]
mod intent_telemetry_tests {
    use super::*;

    #[test]
    fn intent_telemetry_maps_fields_and_summary() {
        let (summary, meta) = intent_decision_telemetry(
            true,
            &IntentComplexity::Complex,
            "tools_required",
            42,
            true,
            &["mutation_contract"],
        );
        assert!(summary.contains("needs_tools=true"), "summary: {summary}");
        assert!(summary.contains("complexity=complex"), "summary: {summary}");
        assert!(
            summary.contains("route=tools_required"),
            "summary: {summary}"
        );
        assert_eq!(meta["condition"], "intent_classification");
        assert_eq!(meta["needs_tools"], true);
        assert_eq!(meta["complexity"], "complex");
        assert_eq!(meta["route"], "tools_required");
        assert_eq!(meta["user_text_len"], 42);
        assert_eq!(meta["tool_control_enforced"], true);
        assert_eq!(meta["requirement_reasons"][0], "mutation_contract");
    }

    #[test]
    fn intent_telemetry_labels_each_complexity_variant() {
        let cases = [
            (IntentComplexity::Simple, "simple"),
            (IntentComplexity::Complex, "complex"),
            (
                IntentComplexity::ScheduledMissingTiming,
                "scheduled_missing_timing",
            ),
            (
                IntentComplexity::Scheduled {
                    schedule_raw: "daily".into(),
                    is_one_shot: false,
                },
                "scheduled",
            ),
        ];
        for (complexity, label) in cases {
            let (_s, meta) = intent_decision_telemetry(
                false,
                &complexity,
                "default_continue",
                0,
                false,
                &["none"],
            );
            assert_eq!(meta["complexity"], label, "variant should map to {label}");
        }
    }
}
