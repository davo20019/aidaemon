//! The exact resources a conversation's own tool calls already reached.
//!
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

    /// The task-context section, or `None` when nothing is known.
    pub(crate) fn render(&self) -> Option<String> {
        if self.0.is_empty() {
            return None;
        }
        let mut out = String::from(
            "[Known resources]\nExact resources this work already reached through successful \
             tool calls, most recent first. When the task refers to one of them, use the exact \
             value here rather than rebuilding a path or URL from the wording:",
        );
        for hint in &self.0 {
            let kind = match hint.kind {
                ToolTargetHintKind::Url => "url",
                _ => "path",
            };
            out.push_str(&format!("\n- {kind}: {}", hint.value));
        }
        Some(out)
    }
}

/// The exact identity a hint names, normalized for the handoff.
fn exact_identity(hint: &ToolTargetHint) -> Option<ToolTargetHint> {
    let value = hint.value.trim();
    match hint.kind {
        ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope => {
            let absolute = std::path::Path::new(value).is_absolute();
            (absolute && !value.contains('\n'))
                .then(|| ToolTargetHint::new(ToolTargetHintKind::Path, value))
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
        for task_id in task_ids {
            if let Ok(run) = self
                .event_store
                .task_run_aggregate(session_id, &task_id)
                .await
            {
                known.extend_from_run(&run);
            }
        }
        if let Some(inherited) = self.inherited_authority.as_ref() {
            known.extend(inherited.known_resources.as_slice());
        }
        known
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
                hint(ToolTargetHintKind::Path, "/srv/synthetic-app"),
                hint(ToolTargetHintKind::Url, "https://api.example.com/v1/items"),
            ]
        );
        let rendered = known.render().unwrap();
        assert!(rendered.starts_with("[Known resources]"));
        assert!(rendered.contains("- path: /srv/synthetic-app"));
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

    #[test]
    fn is_bounded() {
        let mut known = KnownResources::default();
        for index in 0..(MAX_KNOWN_RESOURCES + 5) {
            known.push(&hint(ToolTargetHintKind::Path, &format!("/srv/f{index}")));
        }
        assert_eq!(known.as_slice().len(), MAX_KNOWN_RESOURCES);
        assert_eq!(KnownResources::default().render(), None);
    }
}
