use crate::events::TaskOutcome;
use crate::events::{HarnessEvalSnapshot, TaskEndData};

const WARN_OVERALL: f32 = 0.6;
const WARN_ROUTING: f32 = 0.7;

#[derive(Debug, Clone)]
pub struct EvalTaskRow {
    pub task_id: String,
    pub session_id: String,
    pub created_at: String,
    pub task_end: TaskEndData,
    pub eval: HarnessEvalSnapshot,
}

#[derive(Debug, Clone, Default)]
pub struct EvalSummaryStats {
    pub task_count: usize,
    pub overall_p50: f32,
    pub overall_p95: f32,
    pub routing_p50: f32,
    pub routing_p95: f32,
    pub by_task_kind: Vec<(String, usize, f32)>,
    pub by_route: Vec<(String, usize, f32)>,
}

pub fn format_eval_task_report(row: &EvalTaskRow) -> String {
    let eval = &row.eval;
    let end = &row.task_end;
    let mut out = String::new();
    out.push_str("== Harness Eval Task ==\n");
    out.push_str(&format!("- task_id: {}\n", row.task_id));
    out.push_str(&format!("- session_id: {}\n", row.session_id));
    out.push_str(&format!("- recorded_at: {}\n", row.created_at));
    out.push_str(&format!(
        "- outcome: {} (status: {})\n",
        end.effective_outcome().as_str(),
        end.status.as_str()
    ));
    out.push_str(&format!(
        "- completion_task_kind: {}\n",
        eval.completion_task_kind
    ));
    out.push_str(&format!(
        "- orchestration_route: {}\n",
        eval.orchestration_route
    ));
    out.push_str(&format!(
        "- routing: predicted_tools={} used_tools={} direct_return={} drift_failsafe={} escalated={}\n",
        eval.routing.tools_required_predicted,
        eval.routing.tools_actually_used,
        eval.routing.direct_return_attempted,
        eval.routing.route_drift_failsafe,
        eval.routing.model_escalated
    ));
    out.push_str(&format!(
        "- progress: iterations={} tools_ok={}/{} evidence_gain={} stall_guards={} repetition_guards={} deferred_no_tool={} no_progress_iters={}\n",
        eval.progress.iterations,
        eval.progress.tool_calls_succeeded,
        eval.progress.tool_calls_attempted,
        eval.progress.evidence_gain_total,
        eval.progress.stall_guard_fires,
        eval.progress.repetition_guard_fires,
        eval.progress.deferred_no_tool_events,
        eval.progress.no_progress_iterations
    ));
    let contract = &eval.quality.contract;
    out.push_str(&format!(
        "- contract: expects_mutation={} mutations={} requires_observation={} observations={} verification_required={} verifications={} verification_blocks={} fulfilled={}\n",
        contract.expects_mutation,
        contract.mutation_count,
        contract.requires_observation,
        contract.observation_count,
        contract.verification_required,
        contract.verification_count,
        contract.verification_blocks,
        contract.fulfilled
    ));
    out.push_str(&format!(
        "- quality: stop_reason={} contract_fulfilled={} validation_failures={} unrecovered_errors={} approval_denied={}\n",
        eval.quality.stop_reason,
        eval.quality.contract_fulfilled,
        eval.quality.post_exec_validation_failures,
        eval.quality.unrecovered_errors,
        eval.quality.approval_denied
    ));
    out.push_str(&format!(
        "- cost: raw_tokens={} weighted_tokens={} llm_calls={} sub_agent_weighted={} failed_waste={}\n",
        eval.cost.total_input_tokens + eval.cost.total_output_tokens,
        eval.cost.weighted_tokens,
        eval.cost.llm_calls,
        eval.cost.sub_agent_weighted_tokens,
        eval.cost.tokens_failed_waste
    ));
    out.push_str(&format_scores_block(&eval.scores));
    out.push_str(&routing_mismatch_warnings(eval));
    out
}

pub fn format_eval_summary_row(stats: &EvalSummaryStats, hours: i64, root_only: bool) -> String {
    let scope = if root_only {
        "root tasks only"
    } else {
        "including sub-agents"
    };
    let mut out = String::new();
    out.push_str("== Harness Eval Summary ==\n");
    out.push_str(&format!("- window_hours: {hours}\n"));
    out.push_str(&format!("- scope: {scope}\n"));
    out.push_str(&format!("- tasks_with_eval: {}\n", stats.task_count));
    if stats.task_count == 0 {
        out.push_str("(no TaskEnd rows with harness_eval in window)\n");
        return out;
    }
    out.push_str(&format!(
        "- overall: p50={:.2} p95={:.2}\n",
        stats.overall_p50, stats.overall_p95
    ));
    out.push_str(&format!(
        "- routing_accuracy: p50={:.2} p95={:.2}\n",
        stats.routing_p50, stats.routing_p95
    ));
    if !stats.by_task_kind.is_empty() {
        out.push_str("\nBy completion_task_kind:\n");
        for (kind, count, avg) in &stats.by_task_kind {
            out.push_str(&format!("- {kind}: n={count} avg_overall={avg:.2}\n"));
        }
    }
    if !stats.by_route.is_empty() {
        out.push_str("\nBy orchestration_route:\n");
        for (route, count, avg) in &stats.by_route {
            out.push_str(&format!("- {route}: n={count} avg_overall={avg:.2}\n"));
        }
    }
    out
}

pub fn format_diagnose_harness_section(eval: &HarnessEvalSnapshot) -> String {
    let mut out = String::from("\n\n## Harness Effectiveness\n\n");
    out.push_str(&format_scores_block(&eval.scores));
    out.push_str(&format!(
        "- Route: {} (predicted_tools={} used_tools={})\n",
        eval.orchestration_route,
        eval.routing.tools_required_predicted,
        eval.routing.tools_actually_used
    ));
    out.push_str(&format!(
        "- Progress yield: {:.2} (iterations={}, tools_ok={}, deferred_no_tool={}, no_progress_iters={})\n",
        eval.scores.progress_yield,
        eval.progress.iterations,
        eval.progress.tool_calls_succeeded,
        eval.progress.deferred_no_tool_events,
        eval.progress.no_progress_iterations
    ));
    out.push_str(&format!(
        "- Contract: {:.2} fulfilled={} stop_reason={} expects_mutation={} mutations={} validation_failures={}\n",
        eval.scores.contract_fulfillment,
        eval.quality.contract_fulfilled,
        eval.quality.stop_reason,
        eval.quality.contract.expects_mutation,
        eval.quality.contract.mutation_count,
        eval.quality.post_exec_validation_failures
    ));
    out.push_str(&format!(
        "- Cost: weighted_tokens={} raw={} llm_calls={}\n",
        eval.cost.weighted_tokens,
        eval.cost.total_input_tokens + eval.cost.total_output_tokens,
        eval.cost.llm_calls
    ));
    out.push_str(&routing_mismatch_warnings(eval));
    out
}

fn format_scores_block(scores: &crate::events::HarnessScoresPayload) -> String {
    format!(
        "- Scores: overall={} routing={} progress={} contract={} cost={}\n",
        score_label(scores.overall, WARN_OVERALL),
        score_label(scores.routing_accuracy, WARN_ROUTING),
        score_label(scores.progress_yield, WARN_OVERALL),
        score_label(scores.contract_fulfillment, WARN_OVERALL),
        score_label(scores.cost_efficiency, WARN_OVERALL),
    )
}

fn score_label(value: f32, warn_below: f32) -> String {
    let status = if value >= warn_below {
        "ok"
    } else if value >= warn_below - 0.15 {
        "warn"
    } else {
        "bad"
    };
    format!("{value:.2} ({status})")
}

fn routing_mismatch_warnings(eval: &HarnessEvalSnapshot) -> String {
    let mut out = String::new();
    if eval.orchestration_route == "direct_reply" && eval.routing.tools_actually_used {
        out.push_str("- ⚠️ routing mismatch: direct_reply route but tools were used\n");
    }
    if eval.routing.tools_required_predicted
        && !eval.routing.tools_actually_used
        && eval.quality.outcome != TaskOutcome::Succeeded.as_str()
    {
        out.push_str(
            "- ⚠️ routing mismatch: tools required but none used on non-success outcome\n",
        );
    }
    if eval.routing.route_drift_failsafe {
        out.push_str("- ⚠️ route drift failsafe was active this turn\n");
    }
    out
}

pub fn percentile(values: &mut [f32], pct: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * pct).round() as usize;
    values[idx.min(values.len() - 1)]
}

pub fn aggregate_summary(rows: &[EvalTaskRow]) -> EvalSummaryStats {
    let mut stats = EvalSummaryStats {
        task_count: rows.len(),
        ..Default::default()
    };
    if rows.is_empty() {
        return stats;
    }

    let mut overall: Vec<f32> = rows.iter().map(|row| row.eval.scores.overall).collect();
    let mut routing: Vec<f32> = rows
        .iter()
        .map(|row| row.eval.scores.routing_accuracy)
        .collect();
    stats.overall_p50 = percentile(&mut overall.clone(), 0.50);
    stats.overall_p95 = percentile(&mut overall, 0.95);
    stats.routing_p50 = percentile(&mut routing.clone(), 0.50);
    stats.routing_p95 = percentile(&mut routing, 0.95);

    stats.by_task_kind = aggregate_by_key(rows, |row| row.eval.completion_task_kind.clone());
    stats.by_route = aggregate_by_key(rows, |row| row.eval.orchestration_route.clone());
    stats
}

fn aggregate_by_key(
    rows: &[EvalTaskRow],
    key_fn: impl Fn(&EvalTaskRow) -> String,
) -> Vec<(String, usize, f32)> {
    use std::collections::HashMap;
    let mut buckets: HashMap<String, Vec<f32>> = HashMap::new();
    for row in rows {
        buckets
            .entry(key_fn(row))
            .or_default()
            .push(row.eval.scores.overall);
    }
    let mut out: Vec<(String, usize, f32)> = buckets
        .into_iter()
        .map(|(key, scores)| {
            let avg = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
            (key, scores.len(), avg)
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_handles_small_samples() {
        let mut values = vec![0.2, 0.5, 0.9];
        assert!((percentile(&mut values, 0.5) - 0.5).abs() < f32::EPSILON);
    }
}
