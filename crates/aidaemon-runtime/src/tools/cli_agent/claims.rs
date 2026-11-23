//! Working-directory claims: what each running run may change in a folder,
//! so two runs never change the same files at once while runs that change
//! different files (agents compared on one task, each writing its own
//! file) run side by side. A conflict is decided from the runs' typed write
//! scopes, never from their prompts' wording.

use super::*;

/// What a run may change in its folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum WriteScope {
    /// It changes nothing.
    ReadOnly,
    /// It may change anything in the folder: no write paths were given.
    Whole,
    /// Only these paths (files or folders), normalized.
    Paths(Vec<String>),
}

impl WriteScope {
    /// The scope of a run in `mode` with `write_paths`.
    pub(super) fn of(mode: CliWorkspaceMode, write_paths: &[String]) -> Self {
        match mode {
            CliWorkspaceMode::ReadOnly => Self::ReadOnly,
            CliWorkspaceMode::ReadWrite if write_paths.is_empty() => Self::Whole,
            CliWorkspaceMode::ReadWrite => Self::Paths(write_paths.to_vec()),
        }
    }

    /// Whether two runs could change the same file. Paths overlap when one
    /// is the other or holds it (path components, a machine format).
    pub(super) fn overlaps(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ReadOnly, _) | (_, Self::ReadOnly) => false,
            (Self::Whole, _) | (_, Self::Whole) => true,
            (Self::Paths(mine), Self::Paths(theirs)) => mine
                .iter()
                .any(|a| theirs.iter().any(|b| holds(a, b) || holds(b, a))),
        }
    }

    /// The scope in words, for a refusal.
    fn words(&self) -> String {
        match self {
            Self::ReadOnly => "nothing".into(),
            Self::Whole => "anything in the folder".into(),
            Self::Paths(paths) => paths.join(", "),
        }
    }
}

/// `outer` is `inner` or a folder holding it.
fn holds(outer: &str, inner: &str) -> bool {
    let outer = std::path::Path::new(outer.trim_end_matches(['/', '\\']));
    std::path::Path::new(inner).starts_with(outer)
}

/// A running run's claim on a folder, with enough to explain a conflict.
pub(super) struct WorkingDirClaim {
    pub(super) task_id: String,
    pub(super) tool_name: String,
    pub(super) prompt_summary: String,
    pub(super) writes: WriteScope,
}

/// The running claim of `claims` that `writes` would collide with.
pub(super) fn conflicting<'a>(
    claims: &'a [WorkingDirClaim],
    writes: &WriteScope,
) -> Option<&'a WorkingDirClaim> {
    claims.iter().find(|claim| claim.writes.overlaps(writes))
}

/// Why a run cannot start beside `claim` in `dir`.
pub(super) fn conflict_message(dir: &str, claim: &WorkingDirClaim) -> String {
    format!(
        "BLOCKED: Another CLI agent is already working in {dir} and may change the same files \
         (task_id={}, agent={}, prompt=\"{}\", it may change: {}). \
         Wait for it to finish or cancel it, or give this run write_paths that do not overlap.",
        claim.task_id,
        claim.tool_name,
        claim.prompt_summary,
        claim.writes.words()
    )
}

/// Release task `task_id`'s claim on `dir`, if it holds one.
pub(super) fn release_claim(
    claims: &mut HashMap<String, Vec<WorkingDirClaim>>,
    dir: &str,
    task_id: &str,
) {
    if let Some(held) = claims.get_mut(dir) {
        held.retain(|claim| claim.task_id != task_id);
        if held.is_empty() {
            claims.remove(dir);
        }
    }
}

/// The claims map uses a std (not tokio) Mutex so that WorkingDirClaimGuard
/// can release a claim in Drop (Drop cannot await). Never hold this lock
/// across an await point; the compiler enforces it for Send futures.
pub(super) type WorkingDirClaims = std::sync::Mutex<HashMap<String, Vec<WorkingDirClaim>>>;

/// Lock the claims map, recovering from poisoning (a panic while holding the
/// lock must not permanently wedge working-dir dispatch).
pub(super) fn lock_claims(
    claims: &WorkingDirClaims,
) -> std::sync::MutexGuard<'_, HashMap<String, Vec<WorkingDirClaim>>> {
    claims
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// RAII guard that releases a working-dir claim on drop unless disarmed.
///
/// Dispatch futures can be dropped at any await point when the calling
/// session is cancelled or times out. Before this guard existed, an abort
/// between claim insertion and the explicit post-await release leaked the
/// claim permanently, blocking all future dispatches to that directory
/// until a daemon restart. The guard makes release abort-safe.
///
/// Disarm when handing the task off to the background `running` map: from
/// that point the reaper / cancel paths own the release.
pub(super) struct WorkingDirClaimGuard {
    pub(super) claims: Arc<WorkingDirClaims>,
    pub(super) dir: String,
    pub(super) task_id: String,
    pub(super) armed: bool,
}

impl WorkingDirClaimGuard {
    pub(super) fn new(claims: Arc<WorkingDirClaims>, dir: String, task_id: String) -> Self {
        Self {
            claims,
            dir,
            task_id,
            armed: true,
        }
    }

    /// Stop the guard from releasing on drop (ownership of the claim has
    /// been transferred to the background `running` map).
    pub(super) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for WorkingDirClaimGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        release_claim(&mut lock_claims(&self.claims), &self.dir, &self.task_id);
    }
}
