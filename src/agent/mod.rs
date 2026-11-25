#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::AtomicU64;
#[cfg(test)]
use std::sync::atomic::Ordering;
#[cfg(test)]
use std::time::Duration;

#[cfg(test)]
use chrono::Utc;
#[cfg(test)]
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
#[cfg(test)]
use tokio::sync::mpsc;

#[cfg(test)]
use crate::events::TaskStatus;
#[cfg(test)]
use crate::events::ToolCallInfo;
#[cfg(test)]
use crate::events::{
    AssistantResponseData, TaskEndData, TaskStartData, ThinkingStartData, ToolResultData,
};
#[cfg(test)]
use crate::events::{DecisionPointData, DecisionType, EventType};
#[cfg(test)]
use crate::execution_policy::{ExecutionPolicy, ModelProfile};
#[cfg(test)]
use crate::traits::Message;
#[cfg(test)]
use crate::traits::ToolCall;
#[cfg(test)]
use crate::traits::ToolCapabilities;
#[cfg(test)]
use crate::traits::{AgentRole, Tool};
#[cfg(test)]
use crate::types::ChannelVisibility;
#[cfg(test)]
pub use crate::types::StatusUpdate;
#[cfg(test)]
use crate::types::{ChannelContext, UserRole};
// `Task` is consumed only by in-tree `#[cfg(test)]` modules that import the
// agent namespace via `use super::*` (e.g. `runtime/goal_failure_tests.rs`).
#[cfg(test)]
use crate::traits::Task;

#[cfg(test)]
#[path = "turn/execution_state.rs"]
mod execution_state;
#[cfg(test)]
pub(crate) use execution_state::ExecutionBudget;
/// The post-run learning context lives in `aidaemon-runtime`.
mod post_task {
    pub(crate) use aidaemon_runtime::turn::post_task::*;
}

#[cfg(feature = "computer_use")]
mod computer_use;

#[path = "runtime/approvals.rs"]
pub(crate) mod approvals;
#[path = "runtime/authority.rs"]
pub(crate) mod authority;
#[path = "turn/bootstrap/mod.rs"]
mod bootstrap;
/// Completion contracts live in `aidaemon-runtime`.
#[cfg(test)]
mod completion_contract {
    pub(in crate::agent) use aidaemon_runtime::turn::completion_contract::*;
}
pub(crate) mod conversation_compaction;
#[path = "runtime/coordination.rs"]
pub(crate) mod coordination;
/// The cacheable core prompt lives in `aidaemon-runtime`.
#[cfg(test)]
mod core_prompt {
    pub(crate) use aidaemon_runtime::prompt::core::*;
}
/// Dialogue state lives in `aidaemon-runtime`.
mod dialogue_state {
    #[cfg(test)]
    pub(in crate::agent) use aidaemon_runtime::turn::dialogue_state::*;
}
#[cfg(test)]
#[path = "runtime/dialogue_state_tests.rs"]
mod dialogue_state_tests;
/// Follow-up classification lives in `aidaemon-runtime`.
mod followup {
    #[cfg(test)]
    pub(in crate::agent) use aidaemon_runtime::turn::followup::*;
}
#[path = "runtime/graceful.rs"]
mod graceful;
#[cfg(test)]
#[path = "runtime/history.rs"]
mod history;
pub(crate) mod memory_plugin;
#[path = "runtime/notes.rs"]
mod notes;
#[path = "runtime/parent_delivery.rs"]
mod parent_delivery;
#[path = "turn/post_turn.rs"]
mod post_turn;
#[path = "runtime/sdk_engine.rs"]
pub(crate) mod sdk_engine;
#[path = "runtime/sdk_provider.rs"]
mod sdk_provider;
#[cfg(test)]
#[path = "turn/services.rs"]
mod services;
pub(crate) mod specialists;
#[path = "runtime/tool_result_fit.rs"]
mod tool_result_fit;
#[path = "runtime/turn_context.rs"]
mod turn_context;
#[path = "runtime/untrusted.rs"]
pub(crate) mod untrusted;
#[cfg(test)]
pub(in crate::agent) use completion_contract::CompletionTaskKind;
#[cfg(test)]
pub(in crate::agent) use followup::FollowupMode;
#[cfg(test)]
pub(in crate::agent) use history::CompletionContract;
#[cfg(test)]
pub(in crate::agent) use history::CompletionProgress;
#[cfg(test)]
pub(in crate::agent) use history::TurnContext;
#[path = "runtime/llm.rs"]
mod llm;
#[path = "turn/main_loop.rs"]
mod main_loop;
#[path = "turn/mandate_run.rs"]
mod mandate_run;
#[path = "turn/message_build.rs"]
mod message_build;
#[path = "runtime/models.rs"]
mod models;
/// Prefix fingerprints live in `aidaemon-runtime`.
mod prefix_fingerprint {
    #[cfg(test)]
    pub(crate) use aidaemon_runtime::prompt::fingerprint::*;
}
#[path = "runtime/resume.rs"]
mod resume;
mod runtime_ports;
#[path = "runtime/spawn.rs"]
mod spawn;
/// Session notes and parent-visible deliveries, on any TurnHost.
#[cfg(test)]
pub(crate) use aidaemon_runtime::turn::notes::TurnNotes;
/// The speech-to-text fallback lives in `aidaemon-runtime`.
#[cfg(test)]
pub(crate) mod stt {
    pub(crate) use aidaemon_runtime::media::stt::*;
}
/// System directives live in `aidaemon-runtime`.
#[cfg(test)]
mod system_directives {
    pub(in crate::agent) use aidaemon_runtime::turn::system_directives::*;
}
#[path = "runtime/system_prompt.rs"]
mod system_prompt;
/// Tool dispatch and the tool roster live in `aidaemon-runtime`
/// (`engine::dispatcher`, `engine::tool_roster`); their tests run here, on
/// the daemon's harness.
#[cfg(test)]
#[path = "tools/tool_defs_tests.rs"]
mod tool_defs_tests;
#[cfg(test)]
#[path = "tools/tool_exec_tests.rs"]
mod tool_exec;
#[path = "turn/tool_execution/mod.rs"]
mod tool_execution;

/// Policy runtime metrics and bounded autotuning live in `aidaemon-runtime`.
mod policy_metrics {
    pub use aidaemon_runtime::policy::metrics::*;
}
pub use policy_metrics::apply_bounded_autotune_from_failure_ratio;
// `PolicyAutotuneSnapshot` is part of the public surface but currently has no
// in-tree consumer (the legacy definition carried ``).
#[allow(unused_imports)]
pub use policy_metrics::PolicyAutotuneSnapshot;

// Free-function helpers (status, resume, project-scope cues, untrusted-reference
// filtering, intent-gate merge). Implementation lives in `agent_helpers.rs`.
#[path = "agent_helpers.rs"]
mod agent_helpers;
#[cfg(test)]
pub use agent_helpers::touch_heartbeat;

/// The agent is `aidaemon_runtime::agent::Agent`; the daemon builds it
/// from its config (`construct.rs`) with its own host ports.
pub use aidaemon_runtime::agent::Agent;
pub(crate) use aidaemon_runtime::agent::{AgentDependencies, AgentParts};
/// Tests read the agent through its host traits.
#[cfg(test)]
pub(crate) use aidaemon_runtime::host::TurnHost as _;
#[cfg(test)]
pub(crate) use spawn::SpawnChildForTest;

#[cfg(test)]
pub(in crate::agent) use system_directives::SystemDirective;

// Goal/task dispatch helpers live in `aidaemon_runtime::goal_dispatch`.
#[path = "goal_dispatch.rs"]
mod goal_dispatch;
#[cfg(test)]
pub(in crate::agent) use goal_dispatch::user_facing_task_description;
#[cfg(test)]
pub(crate) use goal_dispatch::{
    SCHEDULED_AUTONOMOUS_BUDGET_EXTENSIONS, SCHEDULED_AUTONOMOUS_HARD_TOKEN_CAP,
};
#[cfg(test)]
pub(crate) use goal_dispatch::{build_goal_failure_summary, build_goal_task_results_summary};

// A goal's background task lead lives in `aidaemon_runtime::background_task_lead`.
#[path = "background_task_lead.rs"]
mod background_task_lead;

// The daemon's host ports for the agent, and test construction, live in `construct.rs`.
#[path = "construct.rs"]
mod construct;
pub(crate) use construct::host_ports;
#[cfg(test)]
pub(crate) use construct::new_agent;
#[cfg(test)]
pub(crate) use construct::set_context_compaction_tokens_for_test;

#[cfg(test)]
mod final_reply_marker_tests {

    use super::user_facing_task_description;

    struct Agent;

    impl Agent {
        // A lightweight safety net after post_turn.rs's sanitization: only
        // the control markers a model may echo verbatim.
        fn sanitize_final_reply_markers(reply: &str) -> String {
            crate::tools::sanitize::strip_leaked_control_markers(reply)
        }
    }

    #[test]
    fn strips_control_markers_from_final_reply() {
        let reply = "Done.\n\n[SYSTEM] internal note\n[DIAGNOSTIC] trace\n[UNTRUSTED EXTERNAL DATA from 'web_fetch' — test]\npayload\n[END UNTRUSTED EXTERNAL DATA]";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("[SYSTEM]"));
        assert!(!sanitized.contains("[DIAGNOSTIC]"));
        assert!(
            !sanitized.contains("internal note"),
            "SYSTEM content leaked: {sanitized}"
        );
        assert!(!sanitized.contains("UNTRUSTED EXTERNAL DATA"));
        assert!(sanitized.contains("Done."));
    }

    #[test]
    fn strips_diagnostic_blocks_with_content_from_final_reply() {
        let reply = "I encountered an error with the search.\n\n\
            [DIAGNOSTIC] Similar errors resolved before:\n\
            - Used terminal to resolve the issue\n\
              Steps: run cargo build -> fix errors\n\n\
            [TOOL STATS] search_files (24h): 8 calls, 0 failed (0%), avg 296ms\n\
              - 2x: pattern not found\n\n\
            [SYSTEM] This tool has errored 2 semantic times. Do NOT retry it.\n\n\
            I will try a different approach.";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(
            !sanitized.contains("[DIAGNOSTIC]"),
            "DIAGNOSTIC tag leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("Similar errors resolved before"),
            "diagnostic content leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("Used terminal"),
            "solution leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("[TOOL STATS]"),
            "TOOL STATS tag leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("8 calls"),
            "stats content leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("296ms"),
            "stats duration leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("[SYSTEM]"),
            "SYSTEM tag leaked: {sanitized}"
        );
        assert!(
            !sanitized.contains("errored 2 semantic times"),
            "system content leaked: {sanitized}"
        );
        assert!(sanitized.contains("I encountered an error with the search."));
        assert!(sanitized.contains("I will try a different approach."));
    }

    #[test]
    fn strips_prior_turn_markers_from_final_reply() {
        let reply = "Summary [prior turn, truncated]\nNext [prior turn]";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("[prior turn"));
        assert_eq!(sanitized, "Summary\nNext");
    }

    #[test]
    fn identity_lines_stay_as_written_in_a_final_reply() {
        // Replies are cleaned of the daemon's own markers only, never
        // rewritten by their wording (976886d4).
        let reply = "I am a large language model, trained by Google. How can I help?";
        assert_eq!(Agent::sanitize_final_reply_markers(reply), reply);
    }

    #[test]
    fn strips_xml_function_call_blocks_from_final_reply() {
        let reply = "I'll read the most recent 300 lines from that log file.\n\n<function_calls>\n<invoke name=\"terminal\">\n<parameter name=\"command\">tail -n 300 ~/Library/Logs/aidaemon/stdout.log</parameter>\n</invoke>\n</function_calls>\n\nHere's what I found.";
        let sanitized = Agent::sanitize_final_reply_markers(reply);
        assert!(!sanitized.contains("<function_calls>"));
        assert!(!sanitized.contains("<invoke"));
        assert!(!sanitized.contains("<parameter"));
        assert!(!sanitized.contains("tail -n 300"));
        assert!(sanitized.contains("I'll read the most recent 300 lines"));
        assert!(sanitized.contains("Here's what I found."));
    }

    #[test]
    fn scheduler_prose_has_no_lifecycle_meaning_in_progress_descriptions() {
        let cleaned = user_facing_task_description(
            "Scheduled check: Post evening tweet about aidaemon features [SYSTEM: already scheduled and firing now; do not reschedule.]",
        );
        assert_eq!(
            cleaned,
            "Scheduled check: Post evening tweet about aidaemon features"
        );
    }
}

#[cfg(test)]
mod tests;
