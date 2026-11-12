//! What the turn code needs to know about memory tools. The turn treats a
//! tool by these facts, not by its name: the daemon's own memory tools state
//! theirs here, and a replacing memory system's tools get facts derived from
//! what they declare.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::agent::inquiry::{action_from_arguments, capability};
use crate::traits::{
    EvidenceAuthority::{Advisory, Canonical},
    EvidencePurpose::{
        Attribution, CausalExplanation, Content, CurrentState, HistoricalRecord, Outcome,
    },
    EvidenceTemporalScope::{Both, Current, Historical},
    ToolEvidenceCapability,
    ToolSemanticScope::{self, GoalState, UserMemory},
};

type Evidence = Box<dyn Fn(&str) -> Vec<ToolEvidenceCapability> + Send + Sync>;

/// How the turn treats one memory tool's calls and results.
pub(in crate::agent) struct MemoryToolFacts {
    pub scope: Option<ToolSemanticScope>,
    /// The result says little worth carrying into a checkpoint.
    pub low_signal_result: bool,
    /// Older calls are summarized by their `action` argument.
    pub summary_from_action: bool,
    /// An older result stays as an excerpt, because it may be the evidence
    /// behind an answer.
    pub keep_result_excerpt: bool,
    /// What a call with these arguments can prove.
    pub evidence: Evidence,
    /// `evidence` is complete enough to rule this tool out for an inquiry.
    pub evidence_modeled: bool,
    pub representative_calls: &'static [&'static str],
    /// A successful read whose text must not be mistaken for a failure.
    pub output_is_report: Option<fn(&str, Option<&str>) -> bool>,
    /// A call with these arguments stores something new in memory (not a
    /// read, a deletion, a visibility change or requested work such as a
    /// goal): what a turn opted out of learning may not do.
    pub stores: fn(&str) -> bool,
}

static FACTS: LazyLock<RwLock<HashMap<String, Arc<MemoryToolFacts>>>> =
    LazyLock::new(|| RwLock::new(daemon_tool_facts()));

/// The facts of tool `name`, if it is a memory tool.
pub(in crate::agent) fn memory_tool_facts(name: &str) -> Option<Arc<MemoryToolFacts>> {
    FACTS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(name)
        .cloned()
}

/// The memory tools whose evidence is modeled.
pub(in crate::agent) fn modeled_memory_tools() -> Vec<String> {
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
pub(in crate::agent) fn register_memory_system_tool(name: &str, read_only: bool) {
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

fn daemon_tool_facts() -> HashMap<String, Arc<MemoryToolFacts>> {
    let write_only = |scope: Option<ToolSemanticScope>,
                      low_signal_result,
                      summary_from_action,
                      stores: fn(&str) -> bool| MemoryToolFacts {
        scope,
        low_signal_result,
        summary_from_action,
        keep_result_excerpt: false,
        evidence: Box::new(|_| Vec::new()),
        evidence_modeled: false,
        representative_calls: &["{}"],
        output_is_report: None,
        stores,
    };
    let manage_memories = MemoryToolFacts {
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
    };
    let manage_people = MemoryToolFacts {
        scope: None,
        low_signal_result: true,
        summary_from_action: false,
        keep_result_excerpt: false,
        evidence: Box::new(|_| {
            vec![capability(
                UserMemory,
                &[CurrentState, HistoricalRecord, Content, Attribution],
                Canonical,
                Both,
            )]
        }),
        evidence_modeled: true,
        representative_calls: &["{}"],
        output_is_report: None,
        stores: |arguments| {
            matches!(
                action_from_arguments(arguments).as_deref(),
                Some("add" | "update" | "add_fact" | "link" | "confirm" | "reconnect")
            )
        },
    };
    [
        (
            "remember_fact",
            write_only(Some(UserMemory), true, true, |arguments| {
                !crate::tools::memory::call_deletes_facts(arguments)
            }),
        ),
        // Widens who may see a fact already stored.
        (
            "share_memory",
            write_only(Some(UserMemory), false, false, |_| false),
        ),
        ("manage_memories", manage_memories),
        ("manage_people", manage_people),
    ]
    .into_iter()
    .map(|(name, facts)| (name.to_string(), Arc::new(facts)))
    .collect()
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
        Some("search_episodes") => vec![capability(
            UserMemory,
            &[HistoricalRecord, Content, Attribution],
            Advisory,
            Historical,
        )],
        Some("list" | "search") | None => vec![capability(
            UserMemory,
            &[CurrentState, HistoricalRecord, Content, Attribution],
            // This operation reads the canonical persisted memory ledger.
            // The stored claim may itself be user-supplied, but whether it
            // exists (or does not exist) is an authoritative store fact.
            // A current lookup can return claims about either current or
            // historical subjects; the receipt separately timestamps the
            // lookup itself.
            Canonical,
            Both,
        )],
        _ => Vec::new(),
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

    let cleaned = crate::agent::loop_utils::strip_appended_diagnostics(result_text);
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
