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
use aidaemon_core as sdk;
use aidaemon_tools::semantics::{FileFence, FileWriteGuard, FileWriteKind};

/// Whole-file content hash; `None` is a file that does not exist.
pub(crate) type FileState = Option<[u8; 32]>;

/// The most tracked files one shell call re-hashes.
const SHELL_RECHECK_PATHS: usize = 2_000;

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

/// One thing the coordinator should know happened, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LogEntry {
    /// A write was refused because the file had changed under it.
    StaleWriteRefused {
        member: String,
        path: String,
        changed_by: Option<Writer>,
    },
}

impl LogEntry {
    /// The member this entry is about.
    pub(crate) fn member(&self) -> &str {
        match self {
            Self::StaleWriteRefused { member, .. } => member,
        }
    }
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

#[derive(Debug)]
struct OpenShell {
    member: String,
    call_id: String,
    /// Another member's shell call ran at the same time as this one.
    overlapped: bool,
}

#[derive(Debug, Default)]
struct Ledger {
    roster: Vec<Member>,
    files: HashMap<String, TrackedFile>,
    contents: HashMap<[u8; 32], Arc<[u8]>>,
    retained_bytes: usize,
    log: Vec<LogEntry>,
    open_shells: Vec<OpenShell>,
    /// The state each pending write was judged against, by member and call.
    judged: HashMap<(String, String), FileState>,
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

    /// Judge a write `member` proposes to `path`, whose content is `current`.
    pub(crate) fn judge_write(
        &self,
        member: &str,
        path: &str,
        kind: &FileWriteKind,
        current: Option<&[u8]>,
    ) -> WriteVerdict {
        let state = file_state(current);
        let ledger = self.ledger();
        let Some(file) = ledger.files.get(path) else {
            return WriteVerdict::Allow { judged: state };
        };
        let baseline = file.baselines.get(member).copied();
        if baseline == Some(state) {
            return WriteVerdict::Allow { judged: state };
        }
        let attributed = file.known == state;
        let changed_by = attributed.then(|| file.last_writer.clone()).flatten();
        // Only this member changed it since it last looked.
        if changed_by == Some(Writer::Member(member.to_string())) {
            return WriteVerdict::Allow { judged: state };
        }
        // It never saw the file and nobody has changed it.
        if baseline.is_none() && attributed && changed_by.is_none() {
            return WriteVerdict::Allow { judged: state };
        }
        let base = baseline
            .flatten()
            .and_then(|hash| ledger.contents.get(&hash).cloned());
        let current = current.unwrap_or_default();
        let change = describe_change(base.as_deref(), current);
        let stale = StaleWrite {
            changed_by: changed_by.clone(),
            change,
        };
        // A change nobody in the tree made tells the coordinator nothing it
        // could act on, and a member that never read the file only needs to.
        if changed_by.is_none() || baseline.is_none() {
            return WriteVerdict::Reread(stale);
        }
        let overlaps = match kind {
            FileWriteKind::Replace => true,
            FileWriteKind::Append => false,
            FileWriteKind::Edit {
                old_text,
                replace_all,
            } => base
                .as_deref()
                .is_none_or(|base| edit_overlaps(base, current, old_text, *replace_all)),
        };
        if overlaps {
            WriteVerdict::Conflict(stale)
        } else {
            WriteVerdict::Reread(stale)
        }
    }

    /// `member`'s pending call was allowed against `state`.
    pub(crate) fn note_judged(&self, member: &str, call_id: &str, state: FileState) {
        self.ledger()
            .judged
            .insert((member.to_string(), call_id.to_string()), state);
    }

    /// `member`'s write to `path` landed, leaving `bytes` there.
    fn commit_write(&self, member: &str, path: &str, bytes: Option<&[u8]>) {
        let state = file_state(bytes);
        let mut ledger = self.ledger();
        if let Some(bytes) = bytes {
            ledger.retain(state, bytes);
        }
        let file = ledger.files.entry(path.to_string()).or_default();
        file.known = state;
        file.last_writer = Some(Writer::Member(member.to_string()));
        file.baselines.insert(member.to_string(), state);
        file.writers.insert(member.to_string());
    }

    /// `member` started a shell call.
    pub(crate) fn shell_began(&self, member: &str, call_id: &str) {
        let mut ledger = self.ledger();
        let mut overlapped = false;
        for open in ledger.open_shells.iter_mut().filter(|o| o.member != member) {
            open.overlapped = true;
            overlapped = true;
        }
        ledger.open_shells.push(OpenShell {
            member: member.to_string(),
            call_id: call_id.to_string(),
            overlapped,
        });
    }

    /// The tracked paths a finished shell call must look at again.
    fn paths_to_recheck(&self) -> Vec<String> {
        self.ledger()
            .files
            .keys()
            .take(SHELL_RECHECK_PATHS)
            .cloned()
            .collect()
    }

    /// `member`'s shell call ended and the tracked files now hold `observed`:
    /// whatever differs from the ledger is that call's doing, or the window's
    /// if another member's shell ran alongside it.
    fn shell_finished(
        &self,
        member: &str,
        call_id: &str,
        observed: Vec<(String, Option<Vec<u8>>)>,
    ) {
        let mut ledger = self.ledger();
        let overlapped = ledger
            .open_shells
            .iter()
            .position(|open| open.member == member && open.call_id == call_id)
            .map(|at| ledger.open_shells.remove(at).overlapped)
            .unwrap_or(false);
        for (path, bytes) in observed {
            let state = file_state(bytes.as_deref());
            if ledger
                .files
                .get(&path)
                .is_none_or(|file| file.known == state)
            {
                continue;
            }
            if let Some(bytes) = bytes.as_deref() {
                ledger.retain(state, bytes);
            }
            let Some(file) = ledger.files.get_mut(&path) else {
                continue;
            };
            file.known = state;
            file.last_writer = Some(if overlapped {
                Writer::OverlappingShells
            } else {
                Writer::Member(member.to_string())
            });
            file.writers.insert(member.to_string());
        }
    }

    pub(crate) fn log(&self, entry: LogEntry) {
        self.ledger().log.push(entry);
    }

    pub(crate) fn log_entries(&self) -> Vec<LogEntry> {
        self.ledger().log.clone()
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

/// What the ledger says about a proposed write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WriteVerdict {
    Allow {
        judged: FileState,
    },
    /// Refused: the writer re-reads and retries on its own.
    Reread(StaleWrite),
    /// The write would collide with a sibling's change.
    Conflict(StaleWrite),
}

/// Why a write is stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StaleWrite {
    /// `None`: nobody in the tree is known to have made the change.
    pub changed_by: Option<Writer>,
    pub change: String,
}

/// Lines either side of a change that one report shows at most.
const SHOWN_CHANGE_LINES: usize = 40;

/// The part of `current` that differs from `base`, as a byte range aligned to
/// whole lines. One range covers every difference, so two far-apart changes
/// read as one large one.
fn changed_range(base: &[u8], current: &[u8]) -> std::ops::Range<usize> {
    let mut prefix = base.iter().zip(current).take_while(|(a, b)| a == b).count();
    while prefix > 0 && current[prefix - 1] != b'\n' {
        prefix -= 1;
    }
    let longest = base.len().min(current.len()) - prefix;
    let mut suffix = base
        .iter()
        .rev()
        .zip(current.iter().rev())
        .take(longest)
        .take_while(|(a, b)| a == b)
        .count();
    while suffix > 0 && current[current.len() - suffix] != b'\n' && {
        current[current.len() - suffix - 1] != b'\n'
    } {
        suffix -= 1;
    }
    prefix..current.len() - suffix
}

/// Whether editing `old_text` in `current` touches what changed since `base`,
/// or the text is no longer there to edit unambiguously.
fn edit_overlaps(base: &[u8], current: &[u8], old_text: &str, replace_all: bool) -> bool {
    let Ok(text) = std::str::from_utf8(current) else {
        return true;
    };
    let found: Vec<usize> = text.match_indices(old_text).map(|(at, _)| at).collect();
    if found.is_empty() || (found.len() > 1 && !replace_all) {
        return true;
    }
    let changed = changed_range(base, current);
    found.iter().any(|&start| {
        let end = start + old_text.len();
        if changed.is_empty() {
            start < changed.start && changed.start < end
        } else {
            start < changed.end && changed.start < end
        }
    })
}

/// A short account of the change, with line numbers in the current file.
fn describe_change(base: Option<&[u8]>, current: &[u8]) -> String {
    let Some(base) = base else {
        return "(what changed cannot be shown; read the file to see it)".to_string();
    };
    let range = changed_range(base, current);
    let first_line = current[..range.start]
        .iter()
        .filter(|b| **b == b'\n')
        .count()
        + 1;
    let removed_end = base.len() - (current.len() - range.end);
    let side = |bytes: &[u8], mark: char| {
        String::from_utf8_lossy(bytes)
            .lines()
            .take(SHOWN_CHANGE_LINES)
            .map(|line| format!("{mark} {line}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let lines = |bytes: &[u8]| String::from_utf8_lossy(bytes).lines().count();
    format!(
        "At line {first_line} ({} lines before, {} now):\n{}\n{}",
        lines(&base[range.start..removed_end]),
        lines(&current[range.clone()]),
        side(&base[range.start..removed_end], '-'),
        side(&current[range], '+'),
    )
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

/// The current bytes of the file a tool's argument names, if it exists.
async fn read_current(backend: &SharedExecutionBackend, raw: &str) -> Option<Vec<u8>> {
    let path = backend.resolve_path(raw).await.ok()?;
    backend.read(&path).await.ok()
}

/// One dispatched call's view of the tree, handed to the file tools.
pub(crate) struct CallFence {
    pub coordination: Coordination,
    pub member: String,
    pub call_id: String,
    pub backend: SharedExecutionBackend,
}

struct WriteInProgress {
    _held: tokio::sync::OwnedMutexGuard<()>,
    coordination: Coordination,
    member: String,
    raw: String,
    path: String,
    backend: SharedExecutionBackend,
}

#[async_trait]
impl FileWriteGuard for WriteInProgress {
    async fn committed(self: Box<Self>) {
        let bytes = read_current(&self.backend, &self.raw).await;
        self.coordination
            .commit_write(&self.member, &self.path, bytes.as_deref());
    }
}

#[async_trait]
impl FileFence for CallFence {
    async fn observed(&self, path: &str, bytes: &[u8]) {
        if let Ok(path) = canonical_path(&self.backend, path).await {
            self.coordination.record_read(&self.member, &path, bytes);
        }
    }

    /// The hook judged this write before dispatch, so parallel writers can
    /// all have passed it: under the file's lock, whoever comes second finds
    /// the content is no longer what it was judged against.
    async fn begin_write(&self, raw: &str) -> Result<Box<dyn FileWriteGuard>, String> {
        let path = canonical_path(&self.backend, raw)
            .await
            .map_err(|error| error.to_string())?;
        let held = self.coordination.path_lock(&path).lock_owned().await;
        let judged = self
            .coordination
            .ledger()
            .judged
            .remove(&(self.member.clone(), self.call_id.clone()));
        if let Some(judged) = judged {
            let current = read_current(&self.backend, raw).await;
            if file_state(current.as_deref()) != judged {
                return Err(format!(
                    "{raw} changed while this write was waiting. Nothing was written. \
                     Read the file again, then retry against what is there now."
                ));
            }
        }
        Ok(Box::new(WriteInProgress {
            _held: held,
            coordination: self.coordination.clone(),
            member: self.member.clone(),
            raw: raw.to_string(),
            path,
            backend: self.backend.clone(),
        }))
    }
}

/// Prefix of the refusal for a write the writer can fix by re-reading. The
/// run ledger keeps these out of the owner-visible refusals: the write is
/// expected to be retried and to land.
pub(crate) const STALE_WRITE: &str = "[STALE WRITE]";

fn who(changed_by: Option<&Writer>) -> String {
    match changed_by {
        Some(Writer::Member(session)) => format!("another agent ({session})"),
        Some(Writer::OverlappingShells) => "a shell command running alongside yours".to_string(),
        None => "something outside this request".to_string(),
    }
}

fn stale_refusal(path: &str, stale: &StaleWrite) -> String {
    format!(
        "{STALE_WRITE} {path} was changed by {} after you last read it, so this write was \
         not made.\n{}\nRead the file again, then retry against what is there now.",
        who(stale.changed_by.as_ref()),
        stale.change,
    )
}

/// Judges every declared file write once the tree has a child: refuses a
/// write made from a stale view of the file.
pub(in crate::agent) struct CoordinationHook {
    pub family: super::authority::SpawnFamily,
    pub member: String,
    pub tools: Arc<HashMap<String, Arc<dyn crate::traits::Tool>>>,
    pub backend: SharedExecutionBackend,
}

#[async_trait]
impl sdk::Hook for CoordinationHook {
    fn name(&self) -> &'static str {
        "coordination"
    }

    async fn before_tool(
        &self,
        _ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        let Some(coordination) = self.family.active_coordination() else {
            return sdk::ToolDecision::Allow;
        };
        let Some(tool) = self.tools.get(&call.name) else {
            return sdk::ToolDecision::Allow;
        };
        if tool.dispatch_facts().shell_command_line {
            coordination.shell_began(&self.member, &call.id);
            return sdk::ToolDecision::Allow;
        }
        let Some(write) = tool.call_file_write(&call.arguments.to_string()) else {
            return sdk::ToolDecision::Allow;
        };
        let Ok(path) = canonical_path(&self.backend, &write.path).await else {
            return sdk::ToolDecision::Allow;
        };
        let current = read_current(&self.backend, &write.path).await;
        let verdict =
            coordination.judge_write(&self.member, &path, &write.kind, current.as_deref());
        match verdict {
            WriteVerdict::Allow { judged } => {
                coordination.note_judged(&self.member, &call.id, judged);
                sdk::ToolDecision::Allow
            }
            WriteVerdict::Reread(stale) | WriteVerdict::Conflict(stale) => {
                coordination.log(LogEntry::StaleWriteRefused {
                    member: self.member.clone(),
                    path: path.clone(),
                    changed_by: stale.changed_by.clone(),
                });
                sdk::ToolDecision::Deny {
                    reason: stale_refusal(&write.path, &stale),
                }
            }
        }
    }

    /// A shell can change any file, so a tracked file is re-hashed after one.
    async fn after_tool(
        &self,
        _ctx: &sdk::HookContext<'_>,
        receipt: &sdk::Receipt,
        _output: &mut sdk::ToolOutput,
    ) {
        let Some(coordination) = self.family.active_coordination() else {
            return;
        };
        let is_shell = self
            .tools
            .get(&receipt.tool)
            .is_some_and(|tool| tool.dispatch_facts().shell_command_line);
        if !is_shell {
            return;
        }
        let mut observed = Vec::new();
        for path in coordination.paths_to_recheck() {
            let bytes = read_current(&self.backend, &path).await;
            observed.push((path, bytes));
        }
        coordination.shell_finished(&self.member, &receipt.call_id, observed);
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

    fn edit(old_text: &str) -> FileWriteKind {
        FileWriteKind::Edit {
            old_text: old_text.to_string(),
            replace_all: false,
        }
    }

    #[test]
    fn an_edit_away_from_a_siblings_change_is_a_reread_not_a_conflict() {
        let c = Coordination::default();
        let base = b"one\ntwo\nthree\n";
        c.record_read("a", "/w/f", base);
        c.record_read("b", "/w/f", base);
        let after = b"ONE\ntwo\nthree\n";
        c.commit_write("b", "/w/f", Some(after));
        assert!(matches!(
            c.judge_write("a", "/w/f", &edit("three"), Some(after)),
            WriteVerdict::Reread(_)
        ));
    }

    #[test]
    fn an_edit_of_the_changed_lines_or_a_vanished_anchor_is_a_conflict() {
        let c = Coordination::default();
        let base = b"one\ntwo\nthree\n";
        c.record_read("a", "/w/f", base);
        c.record_read("b", "/w/f", base);
        let after = b"ONE more\ntwo\nthree\n";
        c.commit_write("b", "/w/f", Some(after));
        for anchor in ["ONE more", "one"] {
            assert!(
                matches!(
                    c.judge_write("a", "/w/f", &edit(anchor), Some(after)),
                    WriteVerdict::Conflict(_)
                ),
                "{anchor}"
            );
        }
    }

    #[test]
    fn an_overlap_cannot_be_ruled_out_without_the_baseline_content() {
        let c = Coordination::default();
        let filler = vec![1u8; RETAINED_CONTENT_BYTES + 1];
        c.record_read("a", "/w/f", &filler);
        c.commit_write("b", "/w/f", Some(b"zero\nkeep\n"));
        assert!(matches!(
            c.judge_write("a", "/w/f", &edit("keep"), Some(b"zero\nkeep\n")),
            WriteVerdict::Conflict(_)
        ));
    }

    #[test]
    fn replacing_a_file_a_sibling_changed_is_a_conflict() {
        let c = Coordination::default();
        c.record_read("a", "/w/f", b"one\n");
        c.commit_write("b", "/w/f", Some(b"two\n"));
        assert!(matches!(
            c.judge_write("a", "/w/f", &FileWriteKind::Replace, Some(b"two\n")),
            WriteVerdict::Conflict(_)
        ));
    }

    #[test]
    fn a_member_that_never_read_a_siblings_file_rereads() {
        let c = Coordination::default();
        c.commit_write("b", "/w/f", Some(b"two\n"));
        assert!(matches!(
            c.judge_write("a", "/w/f", &FileWriteKind::Replace, Some(b"two\n")),
            WriteVerdict::Reread(_)
        ));
    }

    #[test]
    fn a_change_nobody_in_the_tree_made_is_a_reread() {
        let c = Coordination::default();
        c.record_read("a", "/w/f", b"one\n");
        assert_eq!(
            c.judge_write("a", "/w/f", &FileWriteKind::Replace, Some(b"elsewhere\n")),
            WriteVerdict::Reread(StaleWrite {
                changed_by: None,
                change: describe_change(Some(b"one\n"), b"elsewhere\n"),
            })
        );
    }

    #[test]
    fn a_members_own_change_and_an_untouched_file_are_allowed() {
        let c = Coordination::default();
        c.commit_write("a", "/w/f", Some(b"mine\n"));
        let verdict = c.judge_write("a", "/w/f", &FileWriteKind::Replace, Some(b"mine\n"));
        assert!(matches!(verdict, WriteVerdict::Allow { .. }));
        c.record_read("a", "/w/g", b"seen\n");
        let verdict = c.judge_write("b", "/w/g", &FileWriteKind::Replace, Some(b"seen\n"));
        assert!(matches!(verdict, WriteVerdict::Allow { .. }));
        let verdict = c.judge_write("b", "/w/new", &FileWriteKind::Replace, None);
        assert_eq!(verdict, WriteVerdict::Allow { judged: None });
    }

    #[test]
    fn overlapping_shells_of_different_members_are_blamed_on_the_window() {
        let c = Coordination::default();
        c.record_read("a", "/w/f", b"one\n");
        c.shell_began("a", "1");
        c.shell_began("b", "2");
        c.shell_finished(
            "a",
            "1",
            vec![("/w/f".to_string(), Some(b"two\n".to_vec()))],
        );
        assert_eq!(
            c.ledger().files["/w/f"].last_writer,
            Some(Writer::OverlappingShells)
        );
        assert!(matches!(
            c.judge_write("a", "/w/f", &FileWriteKind::Replace, Some(b"two\n")),
            WriteVerdict::Conflict(_)
        ));
    }

    #[test]
    fn a_shell_with_no_overlap_owns_what_it_changed() {
        let c = Coordination::default();
        c.record_read("a", "/w/f", b"one\n");
        c.shell_began("a", "1");
        c.shell_finished(
            "a",
            "1",
            vec![("/w/f".to_string(), Some(b"two\n".to_vec()))],
        );
        assert!(matches!(
            c.judge_write("a", "/w/f", &FileWriteKind::Replace, Some(b"two\n")),
            WriteVerdict::Allow { .. }
        ));
    }

    fn fence(c: &Coordination, member: &str, call: &str) -> CallFence {
        CallFence {
            coordination: c.clone(),
            member: member.to_string(),
            call_id: call.to_string(),
            backend: crate::execution::active_execution_backend(),
        }
    }

    /// Two writers both passed the hook against the same content: the one
    /// that reaches the file second is told it changed.
    #[tokio::test]
    async fn the_second_of_two_racing_writers_is_stopped_at_the_write() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "base\n").unwrap();
        let raw = file.to_str().unwrap();
        let c = Coordination::default();
        let judged = file_state(Some(b"base\n"));
        c.note_judged("a", "call-1", judged);
        c.note_judged("b", "call-2", judged);

        let first = fence(&c, "a", "call-1").begin_write(raw).await.unwrap();
        std::fs::write(&file, "from a\n").unwrap();
        first.committed().await;

        let second = fence(&c, "b", "call-2").begin_write(raw).await;
        assert!(second.is_err());
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "from a\n");
    }

    /// A write of a path that resolves elsewhere still hits the same entry.
    #[tokio::test]
    async fn two_spellings_of_a_file_share_a_lock_and_an_entry() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "x").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        let backend = crate::execution::active_execution_backend();
        assert_eq!(
            canonical_path(&backend, file.to_str().unwrap())
                .await
                .unwrap(),
            canonical_path(&backend, link.to_str().unwrap())
                .await
                .unwrap(),
        );
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
