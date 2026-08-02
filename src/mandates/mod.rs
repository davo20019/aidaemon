//! Owner-authorized autonomous mandate policy.

pub(crate) mod authority;
pub(crate) mod history;

use serde_json::Value;

use crate::traits::{MandateAuthority, ToolCallEffect, ToolCallSemantics};

pub(crate) fn is_non_delegable_tool(tool_name: &str) -> bool {
    tool_name.starts_with("mcp__") || MandateAuthority::NON_DELEGABLE_TOOLS.contains(&tool_name)
}

/// Extract only explicit owner guidance from a controller goal's otherwise
/// mixed context. This is the sole controller-context field allowed into a
/// mandate child prompt. Keep both entry count and aggregate prompt size
/// bounded even if older data predates the write-side limits.
pub(crate) fn bounded_owner_guidance(goal_context: Option<&str>) -> Vec<String> {
    const MAX_ENTRIES: usize = 10;
    const MAX_ENTRY_TEXT: usize = 1_024;
    const MAX_TOTAL_TEXT: usize = 8 * 1_024;

    let Some(entries) = goal_context
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|value| value.get("owner_guidance").cloned())
        .and_then(|value| value.as_array().cloned())
    else {
        return Vec::new();
    };

    let mut newest_first = Vec::new();
    let mut total_chars = 0usize;
    let mut total_bytes = 0usize;
    for entry in entries.iter().rev().take(MAX_ENTRIES) {
        let Some(raw) = entry
            .get("guidance")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let remaining_chars = MAX_TOTAL_TEXT.saturating_sub(total_chars);
        let remaining_bytes = MAX_TOTAL_TEXT.saturating_sub(total_bytes);
        if remaining_chars == 0 || remaining_bytes == 0 {
            break;
        }
        let max_chars = MAX_ENTRY_TEXT.min(remaining_chars);
        let max_bytes = MAX_ENTRY_TEXT.min(remaining_bytes);
        let mut guidance = String::new();
        for character in raw.chars().take(max_chars) {
            if guidance.len().saturating_add(character.len_utf8()) > max_bytes {
                break;
            }
            guidance.push(character);
        }
        if guidance.is_empty() {
            continue;
        }
        total_chars = total_chars.saturating_add(guidance.chars().count());
        total_bytes = total_bytes.saturating_add(guidance.len());
        newest_first.push(guidance);
    }
    newest_first.reverse();
    newest_first
}

/// Kernel classification for calls made while a mandate controller is the
/// resolved goal. These are exact protocol names/actions, never language
/// inference. Only governed mutations receive an action-budgeted grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MandateCallClass {
    /// Read-only controller protocol state, available without broad read
    /// authority (currently only the current run's task list).
    ProtocolObservation,
    /// Read-only evidence gathering; permitted before ACT when observations
    /// are enabled.
    Observation,
    /// The single durable ACT/WAIT/ASK/STOP commit point.
    RecordDecision,
    /// Goal/task orchestration that is useful only after a current ACT. Child
    /// actions remain independently governed by the same mandate.
    ActControl,
    /// A mutation that must pass the complete owner envelope and consume one
    /// cycle attempt.
    GovernedMutation,
    /// Unknown/admin calls fail closed inside a mandate run.
    Deny,
}

pub(crate) fn classify_mandate_call(
    tool_name: &str,
    arguments: &str,
    semantics: &ToolCallSemantics,
) -> MandateCallClass {
    let action = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("action")
                .and_then(Value::as_str)
                .map(str::to_string)
        });

    if tool_name == "manage_mandates" {
        return if action.as_deref() == Some("record_decision") {
            MandateCallClass::RecordDecision
        } else {
            MandateCallClass::Deny
        };
    }
    if tool_name == "manage_goal_tasks" {
        return match action.as_deref() {
            Some("list_tasks") => MandateCallClass::ProtocolObservation,
            Some("create_task" | "claim_task") => MandateCallClass::ActControl,
            // The mandate lifecycle is committed only through STOP/ASK and
            // runtime finalization. Task mutations/claims are committed only
            // through executor leases, never through a task-lead control call.
            _ => MandateCallClass::Deny,
        };
    }
    if tool_name == "spawn_agent" {
        // Detached executors have a separate progress/completion notification
        // path that can outlive the run fence and surface generated prose to
        // the owner. Mandates use only synchronous, exactly claimed executor
        // children so finalization remains the sole static notification path.
        let synchronous = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .is_some_and(|object| {
                object
                    .get("background")
                    .and_then(Value::as_bool)
                    .is_none_or(|background| !background)
            });
        return if synchronous {
            MandateCallClass::ActControl
        } else {
            MandateCallClass::Deny
        };
    }
    if tool_name == "report_blocker" {
        return MandateCallClass::ActControl;
    }
    // These adapters own opaque nested execution loops. Allowing one as a
    // single observation or mutation would let downstream effects escape
    // target scoping and per-action accounting.
    if is_non_delegable_tool(tool_name) {
        return MandateCallClass::Deny;
    }

    if semantics.mutates_state()
        || matches!(
            semantics.effect,
            ToolCallEffect::Mutation | ToolCallEffect::ObservationAndMutation
        )
    {
        return MandateCallClass::GovernedMutation;
    }
    if semantics.effect == ToolCallEffect::Observation {
        return MandateCallClass::Observation;
    }
    MandateCallClass::Deny
}

/// Enforce the mandate's two-role protocol independently of prompt/tool
/// visibility. The task lead may observe, commit the decision, and orchestrate
/// exact children; only a non-root executor may consume a mutation grant. An
/// executor has one control-plane escape hatch: reporting its own blocker.
pub(crate) fn role_allows_mandate_call(
    class: MandateCallClass,
    tool_name: &str,
    is_task_lead: bool,
) -> bool {
    if is_task_lead {
        return match class {
            MandateCallClass::ProtocolObservation
            | MandateCallClass::Observation
            | MandateCallClass::RecordDecision => true,
            MandateCallClass::ActControl => {
                matches!(tool_name, "manage_goal_tasks" | "spawn_agent")
            }
            MandateCallClass::GovernedMutation | MandateCallClass::Deny => false,
        };
    }

    matches!(class, MandateCallClass::GovernedMutation)
        || (class == MandateCallClass::ActControl && tool_name == "report_blocker")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ToolMutationEffects;

    #[test]
    fn owner_guidance_extraction_ignores_unrelated_controller_context() {
        let context = serde_json::json!({
            "relevant_facts": [{"value": "private"}],
            "recent_messages": ["unrelated history"],
            "owner_guidance": [
                {"guidance": "Prefer thoughtful replies", "recorded_at": "2026-01-01"}
            ]
        });
        assert_eq!(
            bounded_owner_guidance(Some(&context.to_string())),
            vec!["Prefer thoughtful replies"]
        );
    }

    #[test]
    fn owner_guidance_extraction_enforces_entry_and_aggregate_bounds() {
        let entries = (0..12)
            .map(|index| {
                serde_json::json!({
                    "guidance": format!("g{index:02}-{}", "x".repeat(1_100)),
                    "recorded_at": "2026-01-01"
                })
            })
            .collect::<Vec<_>>();
        let context = serde_json::json!({"owner_guidance": entries});
        let guidance = bounded_owner_guidance(Some(&context.to_string()));
        assert_eq!(
            guidance.len(),
            8,
            "8 KiB total bound should admit eight 1,024-byte entries"
        );
        assert!(guidance.iter().all(|entry| entry.chars().count() == 1_024));
        assert!(guidance
            .first()
            .is_some_and(|entry| entry.starts_with("g04-")));
        assert!(guidance
            .last()
            .is_some_and(|entry| entry.starts_with("g11-")));
        assert!(
            guidance
                .iter()
                .map(|entry| entry.chars().count())
                .sum::<usize>()
                <= 8 * 1_024
        );
        assert!(guidance.iter().map(String::len).sum::<usize>() <= 8 * 1_024);
    }

    #[test]
    fn owner_guidance_extraction_enforces_utf8_byte_bounds() {
        let context = serde_json::json!({
            "owner_guidance": [{"guidance": "🦀".repeat(2_000)}]
        });
        let guidance = bounded_owner_guidance(Some(&context.to_string()));
        assert_eq!(guidance.len(), 1);
        assert_eq!(guidance[0].len(), 1_024);
        assert_eq!(guidance[0].chars().count(), 256);
    }

    #[test]
    fn exact_control_calls_do_not_become_general_authority() {
        assert_eq!(
            classify_mandate_call(
                "manage_mandates",
                r#"{"action":"record_decision"}"#,
                &ToolCallSemantics::administrative(),
            ),
            MandateCallClass::RecordDecision
        );
        assert_eq!(
            classify_mandate_call(
                "manage_mandates",
                r#"{"action":"update"}"#,
                &ToolCallSemantics::administrative(),
            ),
            MandateCallClass::Deny
        );
        assert_eq!(
            classify_mandate_call(
                "manage_goal_tasks",
                r#"{"action":"list_tasks"}"#,
                &ToolCallSemantics::observation(),
            ),
            MandateCallClass::ProtocolObservation
        );
        assert_eq!(
            classify_mandate_call(
                "manage_goal_tasks",
                r#"{"action":"complete_goal"}"#,
                &ToolCallSemantics::mutation(),
            ),
            MandateCallClass::Deny
        );
        assert_eq!(
            classify_mandate_call(
                "manage_goal_tasks",
                r#"{"action":"create_task"}"#,
                &ToolCallSemantics::mutation(),
            ),
            MandateCallClass::ActControl
        );
        assert_eq!(
            classify_mandate_call(
                "manage_goal_tasks",
                r#"{"action":"claim_task"}"#,
                &ToolCallSemantics::mutation(),
            ),
            MandateCallClass::ActControl
        );
        for action in [
            "update_task",
            "retry_task",
            "resolve_blocker",
            "complete_goal",
            "fail_goal",
        ] {
            assert_eq!(
                classify_mandate_call(
                    "manage_goal_tasks",
                    &format!(r#"{{"action":"{action}"}}"#),
                    &ToolCallSemantics::mutation(),
                ),
                MandateCallClass::Deny,
                "mandate task control action {action} must fail closed"
            );
        }
        assert_eq!(
            classify_mandate_call(
                "http_request",
                r#"{"method":"POST"}"#,
                &ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION),
            ),
            MandateCallClass::GovernedMutation
        );
        assert_eq!(
            classify_mandate_call(
                "cli_agent",
                r#"{"prompt":"do several things"}"#,
                &ToolCallSemantics::mutation_with(ToolMutationEffects::UNSPECIFIED),
            ),
            MandateCallClass::Deny
        );
        assert_eq!(
            classify_mandate_call(
                "spawn_agent",
                r#"{"mission":"run-bound","task":"exact task","background":false}"#,
                &ToolCallSemantics::mutation(),
            ),
            MandateCallClass::ActControl
        );
        assert_eq!(
            classify_mandate_call(
                "spawn_agent",
                r#"{"mission":"detached","task":"escape","background":true}"#,
                &ToolCallSemantics::mutation(),
            ),
            MandateCallClass::Deny
        );
        for tool_name in [
            "terminal",
            "run_command",
            "browser",
            "computer_use",
            "health_probe",
            "scheduled_goal_runs",
            "read_file",
            "write_file",
            "edit_file",
            "search_files",
            "project_inspect",
            "send_file",
            "git_info",
            "git_commit",
            "check_environment",
        ] {
            assert_eq!(
                classify_mandate_call(
                    tool_name,
                    r#"{"url":"https://api.x.com/2/tweets"}"#,
                    &ToolCallSemantics::observation(),
                ),
                MandateCallClass::Deny,
                "opaque adapter {tool_name} must fail closed even as an observation"
            );
        }
    }

    #[test]
    fn mandate_roles_are_fail_closed_at_dispatch() {
        assert!(role_allows_mandate_call(
            MandateCallClass::RecordDecision,
            "manage_mandates",
            true,
        ));
        assert!(role_allows_mandate_call(
            MandateCallClass::ActControl,
            "manage_goal_tasks",
            true,
        ));
        assert!(role_allows_mandate_call(
            MandateCallClass::ActControl,
            "spawn_agent",
            true,
        ));
        assert!(!role_allows_mandate_call(
            MandateCallClass::GovernedMutation,
            "http_request",
            true,
        ));
        assert!(!role_allows_mandate_call(
            MandateCallClass::ActControl,
            "report_blocker",
            true,
        ));

        assert!(role_allows_mandate_call(
            MandateCallClass::GovernedMutation,
            "http_request",
            false,
        ));
        assert!(role_allows_mandate_call(
            MandateCallClass::ActControl,
            "report_blocker",
            false,
        ));
        for (class, tool) in [
            (MandateCallClass::RecordDecision, "manage_mandates"),
            (MandateCallClass::ProtocolObservation, "manage_goal_tasks"),
            (MandateCallClass::Observation, "web_fetch"),
            (MandateCallClass::ActControl, "manage_goal_tasks"),
            (MandateCallClass::ActControl, "spawn_agent"),
        ] {
            assert!(
                !role_allows_mandate_call(class, tool, false),
                "executor unexpectedly received task-lead authority for {tool}"
            );
        }
    }
}
