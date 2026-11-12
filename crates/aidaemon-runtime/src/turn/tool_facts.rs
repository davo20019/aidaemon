//! What the turn code needs to know about memory tools. The turn treats a
//! tool by these facts, not by its name: the daemon's own memory tools state
//! theirs here, and a replacing memory system's tools get facts derived from
//! what they declare.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::turn::inquiry::{action_from_arguments, capability};
use aidaemon_events::traits::{
    EvidenceAuthority::Canonical,
    EvidencePurpose::{
        Attribution, CausalExplanation, Content, CurrentState, HistoricalRecord, Outcome,
    },
    EvidenceTemporalScope::{Both, Current},
    ToolEvidenceCapability, ToolSemanticScope,
    ToolSemanticScope::{GoalState, UserMemory},
};

/// How the turn treats one memory tool's calls and results.
pub use aidaemon_memory::full::tools::MemoryToolFacts;

static FACTS: LazyLock<RwLock<HashMap<String, Arc<MemoryToolFacts>>>> =
    LazyLock::new(|| RwLock::new(daemon_tool_facts()));

/// The facts of tool `name`, if it is a memory tool.
pub fn memory_tool_facts(name: &str) -> Option<Arc<MemoryToolFacts>> {
    FACTS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(name)
        .cloned()
}

/// The semantic scope of a tool that states none: a memory tool's from its
/// facts, the standard tools' by name.
pub fn fallback_tool_semantic_scope(tool_name: &str) -> Option<ToolSemanticScope> {
    if let Some(facts) = memory_tool_facts(tool_name) {
        return facts.scope;
    }
    match tool_name {
        "web_search" | "web_fetch" | "http_request" | "browser" => {
            Some(ToolSemanticScope::ExternalRemote)
        }
        "read_file" | "search_files" | "edit_file" | "write_file" | "terminal"
        | "project_inspect" => Some(ToolSemanticScope::LocalWorkspace),
        "read_channel_history" | "search_history" => Some(ToolSemanticScope::ConversationHistory),
        "scheduled_goals"
        | "scheduled_goal_runs"
        | "manage_goal_tasks"
        | "goal_trace"
        | "tool_trace"
        | "manage_mandates"
        | "manage_automations" => Some(ToolSemanticScope::GoalState),
        "system_info" | "check_environment" => Some(ToolSemanticScope::HostLocal),
        _ => None,
    }
}

/// The memory tools whose evidence is modeled.
pub fn modeled_memory_tools() -> Vec<String> {
    FACTS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .filter(|(_, facts)| facts.evidence_modeled)
        .map(|(name, _)| name.clone())
        .collect()
}

/// Facts for a tool a memory system contributed, from what it declares. Only
/// the read/write distinction is assumed: such a system may share nothing
/// else with the daemon's memory. The daemon's own tools keep their facts.
pub fn register_memory_system_tool(name: &str, read_only: bool) {
    let mut facts = FACTS
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if DAEMON_TOOLS.contains(&name) {
        return;
    }
    facts
        .entry(name.to_string())
        .or_insert_with(|| Arc::new(memory_system_tool_facts(read_only)));
}

fn memory_system_tool_facts(read_only: bool) -> MemoryToolFacts {
    MemoryToolFacts {
        scope: Some(UserMemory),
        low_signal_result: !read_only,
        summary_from_action: false,
        keep_result_excerpt: read_only,
        evidence: Box::new(move |_| {
            if read_only {
                vec![capability(
                    UserMemory,
                    &[CurrentState, HistoricalRecord, Content, Attribution],
                    Canonical,
                    Both,
                )]
            } else {
                Vec::new()
            }
        }),
        evidence_modeled: true,
        representative_calls: &["{}"],
        output_is_report: None,
        // All such a system declares is whether a tool only reads.
        stores: if read_only { |_| false } else { |_| true },
    }
}

const DAEMON_TOOLS: &[&str] = &[
    "remember_fact",
    "share_memory",
    "manage_memories",
    "manage_people",
];

/// The full memory system's tool facts, with `manage_memories` as the
/// daemon offers it: memory and goal actions in one tool.
fn daemon_tool_facts() -> HashMap<String, Arc<MemoryToolFacts>> {
    let mut facts: HashMap<String, Arc<MemoryToolFacts>> =
        aidaemon_memory::full::tools::tool_facts()
            .into_iter()
            .map(|(name, facts)| (name, Arc::new(facts)))
            .collect();
    facts.insert(
        "manage_memories".to_string(),
        Arc::new(MemoryToolFacts {
            scope: Some(UserMemory),
            low_signal_result: true,
            summary_from_action: true,
            keep_result_excerpt: true,
            evidence: Box::new(manage_memories_evidence),
            evidence_modeled: true,
            representative_calls: &[
                r#"{"action":"search"}"#,
                r#"{"action":"search_episodes"}"#,
                r#"{"action":"list_goals"}"#,
                r#"{"action":"diagnose_scheduled"}"#,
            ],
            output_is_report: Some(manage_memories_output_is_report),
            // Its writes forget, repair or re-scope stored facts, or start work
            // the user asked for (goals, schedules); none stores the turn.
            stores: |_| false,
        }),
    );
    facts
}

fn manage_memories_evidence(arguments: &str) -> Vec<ToolEvidenceCapability> {
    match action_from_arguments(arguments).as_deref() {
        Some("list_goals" | "list_scheduled" | "list_scheduled_matching") => vec![capability(
            GoalState,
            &[CurrentState, Content],
            Canonical,
            Current,
        )],
        Some("diagnose_scheduled") => vec![capability(
            GoalState,
            &[
                CurrentState,
                HistoricalRecord,
                Outcome,
                CausalExplanation,
                Content,
            ],
            Canonical,
            Both,
        )],
        // The memory actions, as the full memory system states them.
        _ => aidaemon_memory::full::tools::manage_memories_evidence(arguments),
    }
}

fn manage_memories_output_is_report(result_text: &str, tool_arguments: Option<&str>) -> bool {
    let action = tool_arguments
        .and_then(|args| serde_json::from_str::<serde_json::Value>(args).ok())
        .and_then(|args| {
            args.get("action")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        });
    if !matches!(
        action.as_deref(),
        Some(
            "list"
                | "health"
                | "search"
                | "search_episodes"
                | "list_goals"
                | "list_scheduled"
                | "list_scheduled_matching"
                | "diagnose_scheduled"
        )
    ) {
        return false;
    }

    let cleaned = crate::turn::loop_utils::strip_appended_diagnostics(result_text);
    let trimmed = cleaned.trim_start();
    !trimmed.starts_with("ERROR:")
        && !trimmed.starts_with("Error:")
        && !trimmed.starts_with("Failed to ")
        && !trimmed.starts_with("Request blocked:")
        && !trimmed.starts_with("Blocked:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memory_systems_tools_are_described_by_what_they_declare() {
        register_memory_system_tool("synthetic_recall", true);
        register_memory_system_tool("synthetic_note", false);

        let read = memory_tool_facts("synthetic_recall").unwrap();
        assert_eq!(read.scope, Some(UserMemory));
        assert!(read.keep_result_excerpt && !read.low_signal_result);
        assert!(!(read.evidence)("{}").is_empty());

        let write = memory_tool_facts("synthetic_note").unwrap();
        assert!(write.low_signal_result && !write.keep_result_excerpt);
        assert!((write.evidence)("{}").is_empty());
    }

    #[test]
    fn a_memory_system_cannot_redefine_the_daemons_tools() {
        register_memory_system_tool("manage_memories", false);
        assert!(
            memory_tool_facts("manage_memories")
                .unwrap()
                .keep_result_excerpt
        );
    }
}
