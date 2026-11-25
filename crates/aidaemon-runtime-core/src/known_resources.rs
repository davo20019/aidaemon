//! The context that crosses a handoff: parent to child, child back to
//! parent, and one turn to the next.
//!
//! Every boundary used to pass prose only, and the receiving side rebuilt
//! what the sending side already knew. What the work has established is
//! typed: the request it serves, the missions delegated on the way down, and
//! the exact resources its successful calls reached. This module carries
//! those as data, in one form for every boundary.
//!
//! Resources:
//! A run learns concrete identities as it works: the directory a command ran
//! in, the files it read or wrote, the URLs it called. They are recorded as
//! typed target hints on each receipt. Without carrying them across a
//! boundary, the next turn or a spawned child has only prose and rebuilds
//! them, and a rebuilt identity is a guess (`~` expanded to an invented home,
//! a URL route read as a filesystem path).
//!
//! [`KnownResources`] is that handoff, in one form for both boundaries: a
//! child inherits it from the run that spawned it
//! ([`InheritedAuthority::known_resources`]), and a turn
//! receives it from the earlier turns projected into its own transcript. It
//! holds only targets of operations that succeeded. It grants nothing: tool
//! authority and trust still come only from the explicit authority.

use aidaemon_events::EventStore;
use aidaemon_events::run_aggregate::RunAggregate;
use aidaemon_tools::semantics::{ToolOutcomeStatus, ToolTargetHint, ToolTargetHintKind};

/// How many runs, and how deep a spawn tree, one handoff reads.
const MAX_HANDOFF_TASKS: usize = 24;

/// At most this many resources travel; the most recent win.
pub const MAX_KNOWN_RESOURCES: usize = 16;

/// Exact resources, most recent first, without repeats.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct KnownResources(Vec<ToolTargetHint>);

impl KnownResources {
    pub fn as_slice(&self) -> &[ToolTargetHint] {
        &self.0
    }

    /// Add one resource behind those already held. A value that is not an
    /// exact identity (a relative path, a non-HTTP URL, a session-local id)
    /// is dropped; a URL keeps only its scheme, host and path, so no query
    /// token or credential travels.
    pub fn push(&mut self, hint: &ToolTargetHint) {
        if self.0.len() >= MAX_KNOWN_RESOURCES {
            return;
        }
        let Some(hint) = exact_identity(hint) else {
            return;
        };
        if !self.0.contains(&hint) {
            self.0.push(hint);
        }
    }

    pub fn extend<'a>(&mut self, hints: impl IntoIterator<Item = &'a ToolTargetHint>) {
        for hint in hints {
            self.push(hint);
        }
    }

    /// The targets of a run's operations that succeeded, latest first.
    pub fn extend_from_run(&mut self, run: &RunAggregate) {
        let mut succeeded = run
            .operations
            .values()
            .filter(|operation| {
                operation.dispatched
                    && !operation.policy_denied
                    && operation.outcome == Some(ToolOutcomeStatus::Succeeded)
            })
            .collect::<Vec<_>>();
        succeeded.sort_by_key(|operation| std::cmp::Reverse(operation.receipt_sequence));
        for operation in succeeded {
            self.extend(&operation.targets);
        }
    }

    /// The directory the work runs in: the most recent project scope its
    /// calls established, else the repository root around the most recent
    /// known path (an existing `.git` marker, never a guess).
    pub fn working_dir(&self) -> Option<String> {
        if let Some(scope) = self
            .0
            .iter()
            .find(|hint| hint.kind == ToolTargetHintKind::ProjectScope)
        {
            return Some(scope.value.clone());
        }
        self.0
            .iter()
            .filter(|hint| hint.kind == ToolTargetHintKind::Path)
            .find_map(|hint| {
                std::path::Path::new(&hint.value)
                    .ancestors()
                    .find(|dir| dir.join(".git").exists())
                    .map(|dir| dir.to_string_lossy().to_string())
            })
    }
}

/// What one boundary hands over, rendered into the receiving agent's task
/// context. It informs; it never grants (authority is enforced separately).
#[derive(Debug)]
pub struct Handoff<'a> {
    /// The owner's own request the work serves (a child of an owner-authored
    /// request only; another author's words are never relayed as the owner's).
    pub origin_request: Option<&'a str>,
    /// Missions delegated from the root down to this agent, outermost first.
    pub lineage: &'a [String],
    pub resources: &'a KnownResources,
}

static NO_RESOURCES: KnownResources = KnownResources(Vec::new());

impl Default for Handoff<'_> {
    fn default() -> Self {
        Self {
            origin_request: None,
            lineage: &[],
            resources: &NO_RESOURCES,
        }
    }
}

/// At most this much of the root request is relayed.
const HANDOFF_REQUEST_CHARS: usize = 1500;

impl Handoff<'_> {
    pub fn render(&self) -> Option<String> {
        let mut lines = Vec::new();
        if let Some(request) = self.origin_request.map(str::trim).filter(|r| !r.is_empty()) {
            let request: String = request.chars().take(HANDOFF_REQUEST_CHARS).collect();
            lines.push(format!(
                "Request this work serves (the owner's words):\n{request}"
            ));
        }
        if !self.lineage.is_empty() {
            lines.push(format!(
                "Delegated as: {} (your task is the last step)",
                self.lineage.join(" -> ")
            ));
        }
        if let Some(dir) = self.resources.working_dir() {
            lines.push(format!(
                "Working directory: {dir}. Relative paths in the task refer to it."
            ));
        }
        if !self.resources.0.is_empty() {
            let mut known = String::from(
                "Known resources (reached by earlier successful tool calls, most recent first; \
                 use these exact values rather than rebuilding a path or URL from the wording):",
            );
            for hint in &self.resources.0 {
                let kind = match hint.kind {
                    ToolTargetHintKind::Url => "url",
                    ToolTargetHintKind::ProjectScope => "project",
                    _ => "path",
                };
                known.push_str(&format!("\n- {kind}: {}", hint.value));
            }
            lines.push(known);
        }
        if lines.is_empty() {
            return None;
        }
        Some(format!("[Handoff context]\n{}", lines.join("\n")))
    }
}

/// The exact identity a hint names, normalized for the handoff.
fn exact_identity(hint: &ToolTargetHint) -> Option<ToolTargetHint> {
    let value = hint.value.trim();
    match hint.kind {
        kind @ (ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope) => {
            let absolute = std::path::Path::new(value).is_absolute();
            (absolute && !value.contains('\n'))
                .then(|| ToolTargetHint::new(kind, value))
                .flatten()
        }
        ToolTargetHintKind::Url => {
            let mut url = reqwest::Url::parse(value).ok()?;
            if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
                return None;
            }
            url.set_query(None);
            url.set_fragment(None);
            let _ = url.set_username("");
            let _ = url.set_password(None);
            ToolTargetHint::new(ToolTargetHintKind::Url, url.to_string())
        }
        // Session-scoped registry ids mean nothing in another session.
        ToolTargetHintKind::ResourceId => None,
    }
}

/// What an agent inherited from the run that spawned it, as resources and
/// handoff read it (built by `InheritedAuthority::handoff`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Inherited<'a> {
    /// The files or directories the child declared it will change.
    pub declared_scope: &'a [String],
    /// What the parent's work had reached.
    pub known: &'a [ToolTargetHint],
    /// The owner's own request, when the root request was owner-authored.
    pub origin_request: Option<&'a str>,
    /// Missions delegated from the root down to this agent, outermost first.
    pub lineage: &'a [String],
}

/// What the work so far has reached, for a turn's context or for a child it
/// spawns: the child's own declared scope first, then the current run
/// (`current_task`), then the tasks of the earlier turns projected into this
/// turn's transcript (exactly those, so nothing reaches a turn that its
/// transcript, and so its taint, does not already hold), then what the agent
/// inherited (`inherited`).
pub async fn known_resources(
    events: &EventStore,
    inherited: Option<Inherited<'_>>,
    session_id: &str,
    current_task: Option<&str>,
    projected_turn_ids: &[String],
) -> KnownResources {
    let mut known = KnownResources::default();
    for path in inherited
        .map(|inherited| inherited.declared_scope)
        .unwrap_or_default()
    {
        if let Some(hint) = ToolTargetHint::new(ToolTargetHintKind::Path, path) {
            known.push(&hint);
        }
    }
    let started = std::time::Instant::now();
    let mut task_ids = current_task
        .map(str::to_string)
        .into_iter()
        .collect::<Vec<_>>();
    match events
        .task_ids_for_turns(session_id, projected_turn_ids)
        .await
    {
        Ok(ids) => task_ids.extend(
            ids.into_iter()
                .filter(|id| Some(id.as_str()) != current_task),
        ),
        Err(error) => {
            tracing::warn!(session_id, %error, "Could not read earlier tasks' resources");
        }
    }
    let runs = handoff_runs(events, session_id, task_ids).await;
    // The runs' aggregates are independent reads; fold them in walk order.
    let aggregates = futures::future::join_all(
        runs.iter()
            .map(|(task_session, task_id)| events.task_run_aggregate(task_session, task_id)),
    )
    .await;
    for run in aggregates.into_iter().flatten() {
        known.extend_from_run(&run);
    }
    tracing::info!(
        session_id,
        tasks_read = runs.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "Handoff resources read"
    );
    if let Some(inherited) = inherited {
        known.extend(inherited.known);
    }
    known
}

/// The runs a handoff reads, in the order their resources are folded: each
/// task with the runs it spawned, depth first. What a child established
/// returns to its parent's context through the typed `spawned_by` link, not
/// through the prose of its reply. As deep as spawning allowed
/// (`subagents.max_depth`); the walk bounds only how many runs it reads.
async fn handoff_runs(
    events: &EventStore,
    session_id: &str,
    task_ids: Vec<String>,
) -> Vec<(String, String)> {
    let mut pending = task_ids
        .into_iter()
        .rev()
        .map(|task_id| (session_id.to_string(), task_id))
        .collect::<Vec<_>>();
    let mut seen = std::collections::HashSet::new();
    let mut runs = Vec::new();
    while let Some((task_session, task_id)) = pending.pop() {
        if !seen.insert(task_id.clone()) {
            continue;
        }
        if seen.len() > MAX_HANDOFF_TASKS {
            break;
        }
        if let Ok(children) = events.spawned_children(&task_id).await {
            for child in children {
                pending.push((child.session_id, child.task_id));
            }
        }
        runs.push((task_session, task_id));
    }
    runs
}

/// The `[Handoff context]` section for an agent's turn: for a child, the
/// owner's request and the mission chain it inherited; for every agent, the
/// working directory and resources the work has reached.
pub async fn handoff_section(
    events: &EventStore,
    inherited: Option<Inherited<'_>>,
    session_id: &str,
    projected_turn_ids: &[String],
) -> Option<String> {
    let resources = known_resources(events, inherited, session_id, None, projected_turn_ids).await;
    Handoff {
        origin_request: inherited.and_then(|inherited| inherited.origin_request),
        lineage: inherited
            .map(|inherited| inherited.lineage)
            .unwrap_or_default(),
        resources: &resources,
    }
    .render()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint(kind: ToolTargetHintKind, value: &str) -> ToolTargetHint {
        ToolTargetHint::new(kind, value).unwrap()
    }

    #[test]
    fn keeps_exact_identities_and_strips_url_secrets() {
        let mut known = KnownResources::default();
        known.push(&hint(
            ToolTargetHintKind::ProjectScope,
            "/srv/synthetic-app",
        ));
        known.push(&hint(
            ToolTargetHintKind::Url,
            "https://user:pw@api.example.com/v1/items?token=abc#frag",
        ));
        known.push(&hint(ToolTargetHintKind::Path, "relative/file.txt"));
        known.push(&hint(ToolTargetHintKind::Url, "ftp://example.com/x"));
        known.push(&hint(ToolTargetHintKind::ResourceId, "res-1"));
        known.push(&hint(ToolTargetHintKind::Path, "/srv/synthetic-app"));
        assert_eq!(
            known.as_slice(),
            &[
                hint(ToolTargetHintKind::ProjectScope, "/srv/synthetic-app"),
                hint(ToolTargetHintKind::Url, "https://api.example.com/v1/items"),
                hint(ToolTargetHintKind::Path, "/srv/synthetic-app"),
            ]
        );
        let rendered = Handoff {
            resources: &known,
            ..Default::default()
        }
        .render()
        .unwrap();
        assert!(rendered.starts_with("[Handoff context]"));
        assert!(rendered.contains("- project: /srv/synthetic-app"));
        assert!(rendered.contains("Working directory: /srv/synthetic-app."));
        assert!(rendered.contains("- url: https://api.example.com/v1/items"));
        assert!(!rendered.contains("token"));
    }

    fn operation(
        id: &str,
        sequence: u64,
        outcome: &str,
        denied: bool,
        target: &str,
    ) -> aidaemon_events::run_aggregate::RunOperation {
        serde_json::from_value(serde_json::json!({
            "operation_id": id,
            "tool_name": "terminal",
            "dispatched": !denied,
            "policy_denied": denied,
            "outcome": outcome,
            "receipt_sequence": sequence,
            "targets": [{"kind": "Path", "value": target}],
        }))
        .expect("operation")
    }

    /// Only operations that succeeded hand their targets on, latest first:
    /// a failed probe of a guessed path and a refused call never do.
    #[test]
    fn a_run_hands_on_only_what_its_successful_calls_reached() {
        let mut run = RunAggregate::new("synthetic-task");
        for operation in [
            operation("a", 1, "succeeded", false, "/srv/synthetic-app"),
            operation("b", 2, "failed_permanent", false, "/home/guessed/app"),
            operation("c", 3, "blocked", true, "/home/guessed/app/src"),
            operation("d", 4, "succeeded", false, "/srv/synthetic-app/tests"),
        ] {
            run.operations
                .insert(operation.operation_id.clone(), operation);
        }
        let mut known = KnownResources::default();
        known.extend_from_run(&run);
        assert_eq!(
            known.as_slice(),
            &[
                hint(ToolTargetHintKind::Path, "/srv/synthetic-app/tests"),
                hint(ToolTargetHintKind::Path, "/srv/synthetic-app"),
            ]
        );
    }

    /// A child's handoff names the owner's request, the missions delegated
    /// down to it, its working directory and the exact resources.
    #[test]
    fn handoff_renders_request_lineage_and_working_directory() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(repo.path().join(".git")).unwrap();
        let file = repo.path().join("src").join("lib.rs");
        let mut known = KnownResources::default();
        known.push(&hint(ToolTargetHintKind::Path, &file.to_string_lossy()));
        known.push(&hint(
            ToolTargetHintKind::Url,
            "https://api.example.com/v1/items",
        ));
        let lineage = vec![
            "Ship the items API".to_string(),
            "Write its tests".to_string(),
        ];
        let rendered = Handoff {
            origin_request: Some("Add an items endpoint and test it."),
            lineage: &lineage,
            resources: &known,
        }
        .render()
        .unwrap();
        let root = repo.path().to_string_lossy();
        assert!(rendered.contains("(the owner's words):\nAdd an items endpoint and test it."));
        assert!(rendered.contains("Delegated as: Ship the items API -> Write its tests"));
        assert!(rendered.contains(&format!("Working directory: {root}.")));
        assert!(rendered.contains(&format!("- path: {}", file.to_string_lossy())));
        assert!(rendered.contains("- url: https://api.example.com/v1/items"));

        // No repository and no project scope: no working directory is guessed.
        let mut loose = KnownResources::default();
        loose.push(&hint(
            ToolTargetHintKind::Url,
            "https://api.example.com/v1/items",
        ));
        assert_eq!(loose.working_dir(), None);
    }

    async fn append_task_start(
        store: &EventStore,
        session: &str,
        task: &str,
        turn: &str,
        spawned_by: Option<(&str, &str)>,
    ) {
        use aidaemon_events::{Event, EventType, TaskRef, TaskStartData};
        let mut event = Event::new(
            session,
            EventType::TaskStart,
            serde_json::to_value(TaskStartData {
                task_id: task.to_string(),
                description: "synthetic".to_string(),
                parent_task_id: None,
                user_message: None,
                turn_id: Some(turn.to_string()),
                spawned_by: spawned_by.map(|(session_id, task_id)| TaskRef {
                    session_id: session_id.to_string(),
                    task_id: task_id.to_string(),
                }),
            })
            .unwrap(),
        );
        event.task_id = Some(task.to_string());
        event.turn_id = Some(turn.to_string());
        store.append(event).await.unwrap();
    }

    async fn append_receipt(
        store: &EventStore,
        session: &str,
        task: &str,
        call_id: &str,
        path: &str,
        outcome: ToolOutcomeStatus,
    ) {
        use aidaemon_events::traits::{
            ToolCallMetadata, ToolCallSemantics, ToolInvocationStage, ToolVerificationMode,
        };
        use aidaemon_events::{
            Event, EventType, ToolCallData, ToolOutcomeEvidenceSource, ToolReceiptV1,
            ToolResultData,
        };
        let call = ToolCallData {
            tool_call_id: call_id.to_string(),
            name: "read_file".to_string(),
            arguments: serde_json::json!({"path": path}),
            summary: None,
            task_id: Some(task.to_string()),
            idempotency_key: None,
            policy_rev: None,
            risk_score: None,
            stable_operation_key: None,
            obligation_ids: Vec::new(),
            max_operation_attempts: None,
            max_operation_invocations: None,
            operation_lineage: None,
            turn_id: None,
        };
        let mut call_event = Event::new(
            session,
            EventType::ToolCall,
            serde_json::to_value(&call).unwrap(),
        );
        call_event.task_id = Some(task.to_string());
        store.append(call_event).await.unwrap();
        let mut semantics = ToolCallSemantics::observation()
            .with_verification_mode(ToolVerificationMode::ResultContent);
        semantics
            .target_hints
            .push(hint(ToolTargetHintKind::Path, path));
        let metadata = ToolCallMetadata {
            outcome_status: Some(outcome),
            semantics,
            ..ToolCallMetadata::default()
        };
        let mut receipt = ToolReceiptV1::from_metadata(
            &metadata,
            outcome,
            ToolOutcomeEvidenceSource::ToolReported,
            None,
        );
        receipt.result_provenance.result_id = Some(format!("result:{call_id}"));
        receipt.invocation_stage = ToolInvocationStage::Dispatched;
        let result = ToolResultData {
            message_id: None,
            tool_call_id: call_id.to_string(),
            name: "read_file".to_string(),
            result: "synthetic".to_string(),
            success: outcome == ToolOutcomeStatus::Succeeded,
            duration_ms: 1,
            error: None,
            task_id: Some(task.to_string()),
            annotations: Vec::new(),
            turn_id: None,
            attachments: Vec::new(),
            receipt: Some(receipt),
        };
        let mut result_event = Event::new(
            session,
            EventType::ToolResult,
            serde_json::to_value(&result).unwrap(),
        );
        result_event.task_id = Some(task.to_string());
        store.append(result_event).await.unwrap();
    }

    /// The serial walk `known_resources` made before its reads were
    /// batched, kept as the reference its result must match.
    async fn serial_walk_reference(
        events: &EventStore,
        session_id: &str,
        task_ids: Vec<String>,
    ) -> KnownResources {
        let mut known = KnownResources::default();
        let mut pending = task_ids
            .into_iter()
            .rev()
            .map(|task_id| (session_id.to_string(), task_id))
            .collect::<Vec<_>>();
        let mut seen = std::collections::HashSet::new();
        while let Some((task_session, task_id)) = pending.pop() {
            if !seen.insert(task_id.clone()) {
                continue;
            }
            if seen.len() > MAX_HANDOFF_TASKS {
                break;
            }
            if let Ok(run) = events.task_run_aggregate(&task_session, &task_id).await {
                known.extend_from_run(&run);
            }
            if let Ok(children) = events.spawned_children(&task_id).await {
                for child in children {
                    pending.push((child.session_id, child.task_id));
                }
            }
        }
        known
    }

    /// A spawn tree several levels deep hands on the same resources, in the
    /// same order, as the serial depth-first walk did: every level's
    /// successful targets, and never a failed call's.
    #[tokio::test]
    async fn a_multi_level_spawn_tree_hands_on_the_same_resources() {
        use ToolOutcomeStatus::{FailedPermanent, Succeeded};
        let database = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", database.path().display()))
            .await
            .unwrap();
        let store = EventStore::on_paths(
            pool,
            aidaemon_tools::LexicalPaths::daemon_host(),
            std::sync::Arc::new(aidaemon_events::NoEventHost),
        )
        .await
        .unwrap();
        let session = "telegram:synthetic-user-1";
        append_task_start(&store, session, "task-root", "turn-1", None).await;
        append_receipt(
            &store,
            session,
            "task-root",
            "c-root",
            "/srv/syn/root",
            Succeeded,
        )
        .await;
        append_receipt(
            &store,
            session,
            "task-root",
            "c-guess",
            "/home/guessed",
            FailedPermanent,
        )
        .await;
        for (child_session, child, parent_session, parent, turn) in [
            ("specialist:a", "task-a", session, "task-root", "turn-a"),
            ("specialist:b", "task-b", session, "task-root", "turn-b"),
            (
                "specialist:a1",
                "task-a1",
                "specialist:a",
                "task-a",
                "turn-a1",
            ),
            (
                "specialist:a1x",
                "task-a1x",
                "specialist:a1",
                "task-a1",
                "turn-a1x",
            ),
        ] {
            append_task_start(
                &store,
                child_session,
                child,
                turn,
                Some((parent_session, parent)),
            )
            .await;
            append_receipt(
                &store,
                child_session,
                child,
                &format!("c-{child}"),
                &format!("/srv/syn/{child}"),
                Succeeded,
            )
            .await;
        }
        append_task_start(&store, session, "task-later", "turn-2", None).await;
        append_receipt(
            &store,
            session,
            "task-later",
            "c-later",
            "/srv/syn/later",
            Succeeded,
        )
        .await;

        let turns = ["turn-1".to_string(), "turn-2".to_string()];
        let known = known_resources(&store, None, session, None, &turns).await;
        let task_ids = store.task_ids_for_turns(session, &turns).await.unwrap();
        assert_eq!(
            known,
            serial_walk_reference(&store, session, task_ids).await
        );
        let mut paths = known
            .as_slice()
            .iter()
            .map(|hint| hint.value.as_str())
            .collect::<Vec<_>>();
        paths.sort_unstable();
        assert_eq!(
            paths,
            [
                "/srv/syn/later",
                "/srv/syn/root",
                "/srv/syn/task-a",
                "/srv/syn/task-a1",
                "/srv/syn/task-a1x",
                "/srv/syn/task-b",
            ]
        );
    }

    #[test]
    fn is_bounded() {
        let mut known = KnownResources::default();
        for index in 0..(MAX_KNOWN_RESOURCES + 5) {
            known.push(&hint(ToolTargetHintKind::Path, &format!("/srv/f{index}")));
        }
        assert_eq!(known.as_slice().len(), MAX_KNOWN_RESOURCES);
        assert_eq!(Handoff::default().render(), None);
    }
}
