use serde_json::{json, Value};

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlannerResultStats {
    pub step_count: usize,
    pub success_criteria_count: usize,
    pub contract_present: bool,
    pub contract_changed: bool,
}

impl PlannerResultStats {
    pub(crate) const fn empty() -> Self {
        Self {
            step_count: 0,
            success_criteria_count: 0,
            contract_present: false,
            contract_changed: false,
        }
    }
}

pub(crate) fn planner_result_metadata(
    action: &str,
    model: &str,
    trust_tier: &str,
    stats: PlannerResultStats,
    error: Option<&str>,
) -> Value {
    let mut metadata = json!({
        "component": "task_assessment",
        "action": action,
        "model": model,
        "trust_tier": trust_tier,
        "step_count": stats.step_count,
        "success_criteria_count": stats.success_criteria_count,
        "contract_present": stats.contract_present,
        "contract_changed": stats.contract_changed
    });
    if let Some(error) = error {
        metadata["error"] = Value::String(error.to_string());
    }
    metadata
}

pub(crate) fn planner_skip_metadata(reason: &str, model: &str, trust_tier: &str) -> Value {
    json!({
        "component": "task_assessment",
        "action": "skipped",
        "reason": reason,
        "model": model,
        "trust_tier": trust_tier
    })
}

pub(crate) fn replanner_result_metadata(
    action: &str,
    model: &str,
    trust_tier: &str,
    step_index: usize,
    step_description: &str,
    advanced: bool,
    evidence: Option<&str>,
) -> Value {
    json!({
        "component": "replanner",
        "action": action,
        "model": model,
        "trust_tier": trust_tier,
        "step_index": step_index,
        "step_description": step_description,
        "advanced": advanced,
        "evidence": evidence
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn planner_result_metadata_records_success_shape() {
        let metadata = planner_result_metadata(
            "succeeded",
            "gemma-4-26b",
            "guided",
            PlannerResultStats {
                step_count: 3,
                success_criteria_count: 2,
                contract_present: true,
                contract_changed: true,
            },
            None,
        );
        assert_eq!(metadata["component"], "task_assessment");
        assert_eq!(metadata["action"], "succeeded");
        assert_eq!(metadata["model"], "gemma-4-26b");
        assert_eq!(metadata["trust_tier"], "guided");
        assert_eq!(metadata["step_count"], 3);
        assert_eq!(metadata["success_criteria_count"], 2);
        assert_eq!(metadata["contract_present"], true);
        assert_eq!(metadata["contract_changed"], true);
        assert!(metadata.get("error").is_none());
    }

    #[test]
    fn planner_result_metadata_records_error_shape() {
        let metadata = planner_result_metadata(
            "failed",
            "gemma-4-26b",
            "guided",
            PlannerResultStats::empty(),
            Some("timeout"),
        );
        assert_eq!(metadata["component"], "task_assessment");
        assert_eq!(metadata["action"], "failed");
        assert_eq!(metadata["error"], "timeout");
    }

    #[test]
    fn planner_skip_metadata_records_reason() {
        let metadata = planner_skip_metadata("control_or_ack", "gemma-4-26b", "guided");
        assert_eq!(
            metadata,
            json!({
                "component": "task_assessment",
                "action": "skipped",
                "reason": "control_or_ack",
                "model": "gemma-4-26b",
                "trust_tier": "guided"
            })
        );
    }

    #[test]
    fn replanner_result_metadata_records_shape() {
        let metadata = replanner_result_metadata(
            "advanced",
            "gemma-4-26b",
            "guided",
            2,
            "Run the verification command",
            true,
            Some("tests passed"),
        );
        assert_eq!(metadata["component"], "replanner");
        assert_eq!(metadata["action"], "advanced");
        assert_eq!(metadata["step_index"], 2);
        assert_eq!(metadata["advanced"], true);
        assert_eq!(metadata["evidence"], "tests passed");
    }
}
