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
//! ([`super::authority::InheritedAuthority::known_resources`]), and a turn
//! receives it from the earlier turns projected into its own transcript. It
//! holds only targets of operations that succeeded. It grants nothing: tool
//! authority and trust still come only from the explicit authority.

use crate::events::RunAggregate;
use crate::traits::{ToolOutcomeStatus, ToolTargetHint, ToolTargetHintKind};

/// How many runs, and how deep a spawn tree, one handoff reads.
const MAX_HANDOFF_TASKS: usize = 24;
const MAX_HANDOFF_DEPTH: usize = 3;

/// At most this many resources travel; the most recent win.
pub(crate) const MAX_KNOWN_RESOURCES: usize = 16;

/// Exact resources, most recent first, without repeats.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub(crate) struct KnownResources(Vec<ToolTargetHint>);

impl KnownResources {
    pub(crate) fn as_slice(&self) -> &[ToolTargetHint] {
        &self.0
    }

    /// Add one resource behind those already held. A value that is not an
    /// exact identity (a relative path, a non-HTTP URL, a session-local id)
    /// is dropped; a URL keeps only its scheme, host and path, so no query
    /// token or credential travels.
    pub(crate) fn push(&mut self, hint: &ToolTargetHint) {
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

    pub(crate) fn extend<'a>(&mut self, hints: impl IntoIterator<Item = &'a ToolTargetHint>) {
        for hint in hints {
            self.push(hint);
        }
    }

    /// The targets of a run's operations that succeeded, latest first.
    pub(crate) fn extend_from_run(&mut self, run: &RunAggregate) {
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
    pub(crate) fn working_dir(&self) -> Option<String> {
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
pub(crate) struct Handoff<'a> {
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
    pub(crate) fn render(&self) -> Option<String> {
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

impl super::Agent {
    /// What the work so far has reached, for this turn's context or for a
    /// child it spawns: the child's own declared scope first, then the
    /// current run (`current_task`), then the tasks of the earlier turns
    /// projected into this turn's transcript (exactly those, so nothing
    /// reaches a turn that its transcript, and so its taint, does not
    /// already hold), then what this agent inherited.
    pub(in crate::agent) async fn known_resources(
        &self,
        session_id: &str,
        current_task: Option<&str>,
        projected_turn_ids: &[String],
    ) -> KnownResources {
        let mut known = KnownResources::default();
        if let Some(declaration) = self
            .inherited_authority
            .as_ref()
            .and_then(|inherited| inherited.child_declaration.as_ref())
        {
            for path in &declaration.scope {
                if let Some(hint) = ToolTargetHint::new(ToolTargetHintKind::Path, path) {
                    known.push(&hint);
                }
            }
        }
        let mut task_ids = current_task
            .map(str::to_string)
            .into_iter()
            .collect::<Vec<_>>();
        match self
            .event_store
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
        // Each task with the runs it spawned, depth first: what a child
        // established returns to its parent's context through the typed
        // `spawned_by` link, not through the prose of its reply.
        let mut pending = task_ids
            .into_iter()
            .rev()
            .map(|task_id| (session_id.to_string(), task_id, 0usize))
            .collect::<Vec<_>>();
        let mut visited = 0usize;
        while let Some((task_session, task_id, depth)) = pending.pop() {
            visited += 1;
            if visited > MAX_HANDOFF_TASKS {
                break;
            }
            if let Ok(run) = self
                .event_store
                .task_run_aggregate(&task_session, &task_id)
                .await
            {
                known.extend_from_run(&run);
            }
            if depth < MAX_HANDOFF_DEPTH {
                if let Ok(children) = self.event_store.spawned_children(&task_id).await {
                    for child in children.into_iter() {
                        pending.push((child.session_id, child.task_id, depth + 1));
                    }
                }
            }
        }
        if let Some(inherited) = self.inherited_authority.as_ref() {
            known.extend(inherited.known_resources.as_slice());
        }
        known
    }

    /// The `[Handoff context]` section for this agent's turn: for a child,
    /// the owner's request and the mission chain it inherited; for every
    /// agent, the working directory and resources the work has reached.
    pub(in crate::agent) async fn handoff_section(
        &self,
        session_id: &str,
        projected_turn_ids: &[String],
    ) -> Option<String> {
        let resources = self
            .known_resources(session_id, None, projected_turn_ids)
            .await;
        let inherited = self.inherited_authority.as_ref();
        Handoff {
            origin_request: inherited
                .filter(|inherited| inherited.origin_trusted)
                .and_then(|inherited| inherited.origin_request.as_deref()),
            lineage: inherited
                .map(|inherited| inherited.lineage.as_slice())
                .unwrap_or_default(),
            resources: &resources,
        }
        .render()
    }
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
