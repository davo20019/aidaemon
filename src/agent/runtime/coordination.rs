//! What one request's tree of plain sub-agents knows about each other: who is
//! working, which files they have seen and changed, and what the coordinator
//! has been told. Held by the [`SpawnFamily`](super::authority::SpawnFamily)
//! and idle until the tree has a child, so a lone agent never pays for it.
//!
//! Files are compared by whole-content hash, never by modification time.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::execution::SharedExecutionBackend;
use aidaemon_tools::semantics::{FileFence, FileWriteGuard};

/// Whole-file content hash; `None` is a file that does not exist.
pub(crate) type FileState = Option<[u8; 32]>;

/// How much baseline content one tree keeps for showing what changed. Past
/// it, baselines are hashes only and a change cannot be shown as a diff.
const RETAINED_CONTENT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn file_state(bytes: Option<&[u8]>) -> FileState {
    bytes.map(|bytes| Sha256::digest(bytes).into())
}

/// One agent in the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Member {
    pub session: String,
    pub mission: String,
    pub specialist: Option<String>,
    /// Files or directories it said it would change (resolved paths).
    pub scope: Vec<String>,
    pub running: bool,
}

/// Who last changed a tracked file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Writer {
    Member(String),
    /// Shell calls of different members overlapped in time: any of them may
    /// have done it, so it counts as a sibling's change for all of them.
    OverlappingShells,
}

#[derive(Debug, Default)]
struct TrackedFile {
    /// The state the ledger last attributed.
    known: FileState,
    last_writer: Option<Writer>,
    /// What each member last saw, by session.
    baselines: HashMap<String, FileState>,
    writers: BTreeSet<String>,
}

#[derive(Debug, Default)]
struct Ledger {
    roster: Vec<Member>,
    files: HashMap<String, TrackedFile>,
    contents: HashMap<[u8; 32], Arc<[u8]>>,
    retained_bytes: usize,
}

impl Ledger {
    fn retain(&mut self, state: FileState, bytes: &[u8]) {
        let Some(hash) = state else { return };
        if self.contents.contains_key(&hash)
            || self.retained_bytes + bytes.len() > RETAINED_CONTENT_BYTES
        {
            return;
        }
        self.retained_bytes += bytes.len();
        self.contents.insert(hash, Arc::from(bytes));
    }
}

#[derive(Debug, Default)]
struct Shared {
    ledger: std::sync::Mutex<Ledger>,
    path_locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

/// The tree's shared knowledge. Clones share it.
#[derive(Debug, Clone, Default)]
pub(crate) struct Coordination(Arc<Shared>);

impl Coordination {
    fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.0.ledger.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Add a member to the roster.
    pub(crate) fn add_member(&self, member: Member) {
        self.ledger().roster.push(member);
    }

    /// `session` has finished working.
    pub(crate) fn finish_member(&self, session: &str) {
        if let Some(member) = self
            .ledger()
            .roster
            .iter_mut()
            .find(|member| member.session == session)
        {
            member.running = false;
        }
    }

    pub(crate) fn roster(&self) -> Vec<Member> {
        self.ledger().roster.clone()
    }

    /// `member` read these bytes of the file at `path`.
    pub(crate) fn record_read(&self, member: &str, path: &str, bytes: &[u8]) {
        let state = file_state(Some(bytes));
        let mut ledger = self.ledger();
        ledger.retain(state, bytes);
        let file = ledger.files.entry(path.to_string()).or_default();
        if file.baselines.is_empty() && file.last_writer.is_none() {
            file.known = state;
        }
        file.baselines.insert(member.to_string(), state);
    }

    /// What `member` last saw of `path`, if it has seen it.
    pub(crate) fn baseline(&self, member: &str, path: &str) -> Option<FileState> {
        self.ledger()
            .files
            .get(path)?
            .baselines
            .get(member)
            .copied()
    }

    /// The per-file lock a write holds from its final check to its landing.
    fn path_lock(&self, path: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.0
            .path_locks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(path.to_string())
            .or_default()
            .clone()
    }
}

/// The path a file tool's argument names, with symlinks resolved so two
/// spellings of one file share an entry. A file that does not exist yet
/// resolves through its parent directory.
pub(crate) async fn canonical_path(
    backend: &SharedExecutionBackend,
    raw: &str,
) -> anyhow::Result<String> {
    let resolved = backend.resolve_path(raw).await?;
    let path = std::path::PathBuf::from(resolved.as_str());
    if let Ok(real) = tokio::fs::canonicalize(&path).await {
        return Ok(real.to_string_lossy().into_owned());
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        if let Ok(real) = tokio::fs::canonicalize(parent).await {
            return Ok(real.join(name).to_string_lossy().into_owned());
        }
    }
    Ok(resolved.as_str().to_string())
}

/// One dispatched call's view of the tree, handed to the file tools.
pub(crate) struct CallFence {
    pub coordination: Coordination,
    pub member: String,
    pub backend: SharedExecutionBackend,
}

#[async_trait]
impl FileFence for CallFence {
    async fn observed(&self, path: &str, bytes: &[u8]) {
        if let Ok(path) = canonical_path(&self.backend, path).await {
            self.coordination.record_read(&self.member, &path, bytes);
        }
    }

    async fn begin_write(&self, _path: &str) -> Result<Box<dyn FileWriteGuard>, String> {
        Err("writes are not fenced yet".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_becomes_that_members_baseline_only() {
        let coordination = Coordination::default();
        coordination.record_read("child-a", "/w/notes.md", b"one");
        assert_eq!(
            coordination.baseline("child-a", "/w/notes.md"),
            Some(file_state(Some(b"one")))
        );
        assert_eq!(coordination.baseline("child-b", "/w/notes.md"), None);
    }

    #[test]
    fn retained_content_stops_at_the_cap_and_counts_a_file_once() {
        let coordination = Coordination::default();
        let chunk = vec![7u8; RETAINED_CONTENT_BYTES - 10];
        coordination.record_read("a", "/w/big", &chunk);
        coordination.record_read("b", "/w/big", &chunk);
        coordination.record_read("a", "/w/small", b"12345");
        coordination.record_read("a", "/w/over", b"12345678901");
        let ledger = coordination.ledger();
        assert_eq!(ledger.retained_bytes, RETAINED_CONTENT_BYTES - 5);
        assert_eq!(ledger.contents.len(), 2);
    }

    #[test]
    fn the_roster_tracks_who_is_still_running() {
        let coordination = Coordination::default();
        coordination.add_member(Member {
            session: "child-a".to_string(),
            mission: "Update the synthetic notes".to_string(),
            specialist: None,
            scope: vec!["/w/notes.md".to_string()],
            running: true,
        });
        coordination.finish_member("child-a");
        assert!(!coordination.roster()[0].running);
    }
}
