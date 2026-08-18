//! Typed background artifact attribution.
//!
//! A background process may auto-deliver only an exact absolute file target
//! from its enforced write manifest. Shell source, referenced source files,
//! output-looking flags, and checklist prose are not authority and are never
//! parsed to guess effects.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::execution::SharedExecutionBackend;

#[derive(Debug)]
pub struct BackgroundDeliverableContext {
    /// Exact declared file targets modified during this process window.
    pub produced_candidates: Vec<PathBuf>,
    /// Exact declared file targets that were missing or stale at completion.
    pub unconfirmed_candidates: Vec<PathBuf>,
}

#[derive(Debug)]
pub enum AutoSendDecision {
    One(PathBuf),
    None,
    Ambiguous(Vec<PathBuf>),
}

fn is_non_deliverable_sink(path: &Path) -> bool {
    path.starts_with("/dev") || path.starts_with("/proc") || path.starts_with("/sys")
}

/// Attribute outputs from the process capability manifest. Only exact
/// absolute write targets can be auto-delivered; directory grants and inferred
/// paths are never promoted into artifacts.
pub async fn attribute_declared_deliverables_backend(
    backend: SharedExecutionBackend,
    declared_write_paths: &[String],
    command_start: SystemTime,
    command_end: SystemTime,
) -> BackgroundDeliverableContext {
    let window_end = command_end + Duration::from_secs(2);
    let mut produced_candidates = Vec::new();
    let mut unconfirmed_candidates = Vec::new();
    let mut seen = HashSet::new();

    for raw in declared_write_paths {
        let raw = raw.trim();
        if !raw.starts_with('/') {
            continue;
        }
        let candidate = PathBuf::from(raw);
        if is_non_deliverable_sink(&candidate) || !seen.insert(candidate.clone()) {
            continue;
        }
        let Ok(path) = backend.resolve_path(raw).await else {
            unconfirmed_candidates.push(candidate);
            continue;
        };
        match backend.metadata(&path).await {
            Ok(metadata) if metadata.is_file() => {
                let resolved = PathBuf::from(path.as_str());
                if metadata
                    .modified
                    .is_some_and(|modified| modified >= command_start && modified <= window_end)
                {
                    produced_candidates.push(resolved);
                } else {
                    unconfirmed_candidates.push(resolved);
                }
            }
            Ok(_) => {
                // A directory capability authorizes a tree; it does not name
                // one user-facing artifact and must not trigger a scan.
            }
            Err(_) => unconfirmed_candidates.push(PathBuf::from(path.as_str())),
        }
    }

    produced_candidates.sort();
    unconfirmed_candidates.sort();
    BackgroundDeliverableContext {
        produced_candidates,
        unconfirmed_candidates,
    }
}

pub fn auto_send_decision(ctx: &BackgroundDeliverableContext) -> AutoSendDecision {
    match ctx.produced_candidates.as_slice() {
        [] => AutoSendDecision::None,
        [path] => AutoSendDecision::One(path.clone()),
        paths => AutoSendDecision::Ambiguous(paths.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn only_exact_declared_fresh_files_are_candidates() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let declared = root.path().join("declared.txt");
        let undeclared = root.path().join("undeclared.txt");
        let start = SystemTime::now() - Duration::from_secs(2);
        std::fs::write(&declared, "declared").expect("declared fixture");
        std::fs::write(&undeclared, "undeclared").expect("undeclared fixture");
        let end = SystemTime::now();

        let context = attribute_declared_deliverables_backend(
            backend,
            &[declared.to_string_lossy().to_string()],
            start,
            end,
        )
        .await;

        assert_eq!(context.produced_candidates, [declared]);
        assert!(context.unconfirmed_candidates.is_empty());
        assert!(matches!(
            auto_send_decision(&context),
            AutoSendDecision::One(_)
        ));
    }

    #[tokio::test]
    async fn directories_relative_paths_and_sinks_are_not_artifacts() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let now = SystemTime::now();
        let context = attribute_declared_deliverables_backend(
            backend,
            &[
                root.path().to_string_lossy().to_string(),
                "relative.txt".to_string(),
                "/dev/null".to_string(),
            ],
            now - Duration::from_secs(1),
            now,
        )
        .await;

        assert!(context.produced_candidates.is_empty());
        assert!(context.unconfirmed_candidates.is_empty());
        assert!(matches!(
            auto_send_decision(&context),
            AutoSendDecision::None
        ));
    }

    #[tokio::test]
    async fn exact_missing_target_is_reported_without_searching() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let missing = root.path().join("missing.txt");
        let now = SystemTime::now();
        let context = attribute_declared_deliverables_backend(
            backend,
            &[missing.to_string_lossy().to_string()],
            now - Duration::from_secs(1),
            now,
        )
        .await;

        assert!(context.produced_candidates.is_empty());
        assert_eq!(context.unconfirmed_candidates, [missing]);
    }

    #[tokio::test]
    async fn two_declared_fresh_files_are_ambiguous() {
        let backend = crate::execution::active_execution_backend();
        let root = tempfile::tempdir().expect("root");
        let first = root.path().join("a.txt");
        let second = root.path().join("b.txt");
        let start = SystemTime::now() - Duration::from_secs(2);
        std::fs::write(&first, "a").expect("first fixture");
        std::fs::write(&second, "b").expect("second fixture");
        let end = SystemTime::now();
        let context = attribute_declared_deliverables_backend(
            backend,
            &[
                second.to_string_lossy().to_string(),
                first.to_string_lossy().to_string(),
            ],
            start,
            end,
        )
        .await;

        assert!(matches!(
            auto_send_decision(&context),
            AutoSendDecision::Ambiguous(paths) if paths == vec![first, second]
        ));
    }
}
