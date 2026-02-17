use anyhow::Context;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{Tool, ToolCapabilities, ToolRole};

pub struct PolicyMetricsTool;

#[derive(Debug, Deserialize)]
struct PolicyMetricsArgs {
    #[serde(default)]
    format: Option<String>,
}

#[async_trait]
impl Tool for PolicyMetricsTool {
    fn name(&self) -> &str {
        "policy_metrics"
    }

    fn description(&self) -> &str {
        "Read runtime policy metrics (consultant routing outcomes, no-progress iterations, and failed-task token burn)"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "policy_metrics",
            "description": "Read runtime policy/agent-loop metrics. Use this when the user asks about consultant usefulness, no-progress loops, policy routing behavior, or token cost of failed tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["json", "summary"],
                        "description": "Output format. json returns machine-readable JSON (default). summary returns a concise text summary."
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Universal
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args_input = if arguments.trim().is_empty() {
            "{}"
        } else {
            arguments
        };
        let args: PolicyMetricsArgs = serde_json::from_str(args_input)
            .with_context(|| "policy_metrics arguments must be valid JSON")?;
        let metrics = crate::agent::policy_metrics_snapshot();

        let consultant_total =
            metrics.consultant_direct_return_total + metrics.consultant_fallthrough_total;
        let consultant_direct_return_rate = if consultant_total > 0 {
            metrics.consultant_direct_return_total as f64 / consultant_total as f64
        } else {
            0.0
        };
        let consultant_fallthrough_rate = if consultant_total > 0 {
            metrics.consultant_fallthrough_total as f64 / consultant_total as f64
        } else {
            0.0
        };
        let route_reason_total = metrics.consultant_route_clarification_required_total
            + metrics.consultant_route_tools_required_total
            + metrics.consultant_route_short_correction_direct_reply_total
            + metrics.consultant_route_acknowledgment_direct_reply_total
            + metrics.consultant_route_default_continue_total;
        let route_reason_tools_required_rate = if route_reason_total > 0 {
            metrics.consultant_route_tools_required_total as f64 / route_reason_total as f64
        } else {
            0.0
        };
        let route_reason_return_rate = if route_reason_total > 0 {
            (metrics.consultant_route_clarification_required_total
                + metrics.consultant_route_short_correction_direct_reply_total
                + metrics.consultant_route_acknowledgment_direct_reply_total) as f64
                / route_reason_total as f64
        } else {
            0.0
        };

        if args
            .format
            .as_deref()
            .is_some_and(|fmt| fmt.eq_ignore_ascii_case("summary"))
        {
            return Ok(format!(
                "Policy metrics summary\n\
                 - consultant_direct_return_total: {}\n\
                 - consultant_fallthrough_total: {}\n\
                 - consultant_direct_return_rate: {:.3}\n\
                 - consultant_fallthrough_rate: {:.3}\n\
                 - consultant_route_clarification_required_total: {}\n\
                 - consultant_route_tools_required_total: {}\n\
                 - consultant_route_short_correction_direct_reply_total: {}\n\
                 - consultant_route_acknowledgment_direct_reply_total: {}\n\
                 - consultant_route_default_continue_total: {}\n\
                 - consultant_route_tools_required_rate: {:.3}\n\
                 - consultant_route_return_rate: {:.3}\n\
                 - context_bleed_prevented_total: {}\n\
                 - context_mismatch_preflight_drop_total: {}\n\
                 - followup_mode_overrides_total: {}\n\
                 - cross_scope_blocked_total: {}\n\
                 - tool_schema_contract_rejections_total: {}\n\
                 - route_drift_alert_total: {}\n\
                 - route_drift_failsafe_activation_total: {}\n\
                 - route_failsafe_active_turn_total: {}\n\
                 - tokens_failed_tasks_total: {}\n\
                 - no_progress_iterations_total: {}\n\
                 - deferred_no_tool_forced_required_total: {}\n\
                 - deferred_no_tool_deferral_detected_total: {}\n\
                 - deferred_no_tool_model_switch_total: {}\n\
                 - deferred_no_tool_error_marker_total: {}",
                metrics.consultant_direct_return_total,
                metrics.consultant_fallthrough_total,
                consultant_direct_return_rate,
                consultant_fallthrough_rate,
                metrics.consultant_route_clarification_required_total,
                metrics.consultant_route_tools_required_total,
                metrics.consultant_route_short_correction_direct_reply_total,
                metrics.consultant_route_acknowledgment_direct_reply_total,
                metrics.consultant_route_default_continue_total,
                route_reason_tools_required_rate,
                route_reason_return_rate,
                metrics.context_bleed_prevented_total,
                metrics.context_mismatch_preflight_drop_total,
                metrics.followup_mode_overrides_total,
                metrics.cross_scope_blocked_total,
                metrics.tool_schema_contract_rejections_total,
                metrics.route_drift_alert_total,
                metrics.route_drift_failsafe_activation_total,
                metrics.route_failsafe_active_turn_total,
                metrics.tokens_failed_tasks_total,
                metrics.no_progress_iterations_total,
                metrics.deferred_no_tool_forced_required_total,
                metrics.deferred_no_tool_deferral_detected_total,
                metrics.deferred_no_tool_model_switch_total,
                metrics.deferred_no_tool_error_marker_total,
            ));
        }

        let payload = json!({
            "metrics": metrics,
            "derived": {
                "consultant_total": consultant_total,
                "consultant_direct_return_rate": consultant_direct_return_rate,
                "consultant_fallthrough_rate": consultant_fallthrough_rate,
                "consultant_route_reason_total": route_reason_total,
                "consultant_route_tools_required_rate": route_reason_tools_required_rate,
                "consultant_route_return_rate": route_reason_return_rate,
                "route_drift_total": metrics.route_drift_alert_total,
                "context_integrity_guard_events_total": metrics.context_bleed_prevented_total
                    + metrics.context_mismatch_preflight_drop_total
                    + metrics.followup_mode_overrides_total
                    + metrics.cross_scope_blocked_total,
                "deferred_no_tool_recovery_effectiveness_rate": if metrics.deferred_no_tool_forced_required_total > 0 {
                    metrics
                        .deferred_no_tool_forced_required_total
                        .saturating_sub(metrics.deferred_no_tool_error_marker_total) as f64
                        / metrics.deferred_no_tool_forced_required_total as f64
                } else {
                    0.0
                }
            }
        });
        Ok(serde_json::to_string_pretty(&payload)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_metrics_json_payload() {
        let tool = PolicyMetricsTool;
        let output = tool.call("{}").await.unwrap();
        let parsed: Value = serde_json::from_str(&output).unwrap();

        assert!(parsed.get("metrics").is_some());
        assert!(parsed.get("derived").is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("consultant_direct_return_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("consultant_fallthrough_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("consultant_route_tools_required_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("tokens_failed_tasks_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("no_progress_iterations_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("cross_scope_blocked_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("deferred_no_tool_forced_required_total"))
            .is_some());
        assert!(parsed
            .get("metrics")
            .and_then(|m| m.get("deferred_no_tool_model_switch_total"))
            .is_some());
    }
}
