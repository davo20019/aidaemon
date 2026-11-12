//! Typed background artifact attribution.
//!
//! A background process may auto-deliver only an exact absolute file target
//! from its enforced write manifest. Shell source, referenced source files,
//! output-looking flags, and checklist prose are not authority and are never
//! parsed to guess effects.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::backend::SharedExecutionBackend;

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
