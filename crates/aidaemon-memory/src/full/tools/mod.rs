//! The full memory system's tools: `remember_fact`, `manage_memories`,
//! `manage_people` and `share_memory`, over the memory stores.
//!
//! Each is an [`OutcomeTool`](aidaemon_tools::OutcomeTool), so a host with its
//! own dispatcher (receipts, approvals, result fitting) runs it directly and
//! keeps the typed outcome; [`kernel_tools`] gives the same tools to a plain
//! `aidaemon_core` agent. [`tool_facts`] says how a host's turn should treat
//! each tool's calls and results.

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_tools::semantics::{
    EvidenceAuthority::{Advisory, Canonical},
    EvidencePurpose::{self, Attribution, Content, CurrentState, HistoricalRecord},
    EvidenceTemporalScope::{Both, Historical},
    ToolEvidenceCapability,
    ToolSemanticScope::{self, UserMemory},
};
use aidaemon_tools::services::SettingsStore;
use aidaemon_tools::{KernelAdapter, OutcomeTool};

use crate::stores::MemoryStores;

mod manage_memories;
mod manage_people;
mod remember_fact;
mod share_memory;

pub use manage_memories::{
    MEMORY_ACTIONS, ManageMemoriesTool, memory_parameters, unknown_action_message,
};
pub use manage_people::ManagePeopleTool;
pub use remember_fact::{RememberFactTool, call_deletes_facts};
pub use share_memory::ShareMemoryTool;

/// The names of the full memory system's tools.
pub const TOOL_NAMES: &[&str] = &[
    "remember_fact",
    "manage_memories",
    "manage_people",
    "share_memory",
];

/// The four tools over `store`. `settings` holds the `people_enabled`
/// switch `manage_people` reads and sets.
pub fn outcome_tools(
    store: Arc<dyn MemoryStores>,
    settings: Arc<dyn SettingsStore>,
) -> Vec<Arc<dyn OutcomeTool>> {
    vec![
        Arc::new(RememberFactTool::new(store.clone())),
        Arc::new(ShareMemoryTool::new(store.clone())),
        Arc::new(ManageMemoriesTool::new(store.clone())),
        Arc::new(ManagePeopleTool::new(settings, store)),
    ]
}

/// [`outcome_tools`] as kernel tools, for an agent without a dispatcher of
/// its own.
pub fn kernel_tools(
    store: Arc<dyn MemoryStores>,
    settings: Arc<dyn SettingsStore>,
) -> Vec<Arc<dyn aidaemon_core::Tool>> {
    outcome_tools(store, settings)
        .into_iter()
        .map(|tool| Arc::new(KernelAdapter::new(tool)) as Arc<dyn aidaemon_core::Tool>)
        .collect()
}

/// Settings kept in memory, for a host with no settings store of its own:
/// `people_enabled` lasts until the process ends.
#[derive(Default)]
pub struct LocalSettings(std::sync::Mutex<HashMap<String, String>>);

#[async_trait::async_trait]
impl SettingsStore for LocalSettings {
    async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
        Ok(self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .cloned())
    }

    async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key.to_string(), value.to_string());
        Ok(())
    }
}

/// What a call with these arguments can prove.
pub type Evidence = Box<dyn Fn(&str) -> Vec<ToolEvidenceCapability> + Send + Sync>;

/// How a host's turn treats one memory tool's calls and results.
pub struct MemoryToolFacts {
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
    /// read, a deletion, a visibility change or requested work): what a turn
    /// opted out of learning may not do.
    pub stores: fn(&str) -> bool,
}

fn capability(
    scope: ToolSemanticScope,
    purposes: &[EvidencePurpose],
    authority: aidaemon_tools::semantics::EvidenceAuthority,
    temporal_scope: aidaemon_tools::semantics::EvidenceTemporalScope,
) -> ToolEvidenceCapability {
    ToolEvidenceCapability::new(scope, purposes, authority, temporal_scope)
}

/// The `action` argument, trimmed, if any.
pub fn action_from_arguments(arguments: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()?
        .get("action")?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Reading the user's memory: what is stored now and was stored before, by
/// whom. Whether a claim exists is an authoritative store fact.
fn memory_read() -> ToolEvidenceCapability {
    capability(
        UserMemory,
        &[CurrentState, HistoricalRecord, Content, Attribution],
        Canonical,
        Both,
    )
}

/// The facts of each of this system's tools, by name.
pub fn tool_facts() -> HashMap<String, MemoryToolFacts> {
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
        representative_calls: &[r#"{"action":"search"}"#, r#"{"action":"search_episodes"}"#],
        output_is_report: Some(manage_memories_output_is_report),
        // Its writes forget, repair or re-scope stored facts; none stores
        // the turn.
        stores: |_| false,
    };
    let manage_people = MemoryToolFacts {
        scope: None,
        low_signal_result: true,
        summary_from_action: false,
        keep_result_excerpt: false,
        evidence: Box::new(|_| vec![memory_read()]),
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
                !call_deletes_facts(arguments)
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
    .map(|(name, facts)| (name.to_string(), facts))
    .collect()
}

/// What a `manage_memories` call can prove: a fact lookup reads the
/// canonical memory ledger; episodes are generated summaries.
pub fn manage_memories_evidence(arguments: &str) -> Vec<ToolEvidenceCapability> {
    match action_from_arguments(arguments).as_deref() {
        Some("search_episodes") => vec![capability(
            UserMemory,
            &[HistoricalRecord, Content, Attribution],
            Advisory,
            Historical,
        )],
        Some("list" | "search") | None => vec![memory_read()],
        _ => Vec::new(),
    }
}

fn manage_memories_output_is_report(result_text: &str, tool_arguments: Option<&str>) -> bool {
    let action = tool_arguments.and_then(action_from_arguments);
    if !matches!(
        action.as_deref(),
        Some("list" | "health" | "search" | "search_episodes")
    ) {
        return false;
    }
    let trimmed = result_text.trim_start();
    !trimmed.starts_with("ERROR:")
        && !trimmed.starts_with("Error:")
        && !trimmed.starts_with("Failed to ")
        && !trimmed.starts_with("Request blocked:")
        && !trimmed.starts_with("Blocked:")
}

/// Stores and helpers for the tools' tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use aidaemon_tools::OutcomeTool;
    use aidaemon_tools::semantics::ToolCallContext;
    use async_trait::async_trait;

    use crate::sqlite::SqliteMemory;

    /// An empty memory store on one in-memory connection.
    pub(crate) async fn memory() -> Arc<SqliteMemory> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::sqlite::migrate(&pool).await.unwrap();
        Arc::new(SqliteMemory::new(
            pool,
            Arc::new(crate::HashEmbedder::new(384)),
        ))
    }

    pub(crate) use super::LocalSettings as Settings;

    /// Calling a tool directly from a test.
    #[async_trait]
    pub(crate) trait ToolCalls: OutcomeTool {
        /// Run with no call facts and return the output text.
        async fn call(&self, arguments: &str) -> anyhow::Result<String> {
            Ok(self
                .execute(arguments, &ToolCallContext::default())
                .await?
                .output)
        }
    }

    impl<T: OutcomeTool + ?Sized> ToolCalls for T {}
}
