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

    /// The absolute roots it may change, for a run in `dir`: the folder
    /// itself when it may change anything there.
    fn roots<'a>(&'a self, dir: &'a str) -> Vec<&'a str> {
        match self {
            Self::ReadOnly => Vec::new(),
            Self::Whole => vec![dir],
            Self::Paths(paths) => paths.iter().map(String::as_str).collect(),
        }
    }

    /// Whether a run in `dir` with this scope could change a file that a
    /// run in `other_dir` with `other` could. Runs in one folder compare
    /// scopes ([`Self::overlaps`]); runs in different folders compare the
    /// roots they may change, so a run over a whole repository collides
    /// with one in its `src` folder, and two folders' runs given the same
    /// write path collide.
    pub(super) fn collides(&self, dir: &str, other: &Self, other_dir: &str) -> bool {
        if dir == other_dir {
            return self.overlaps(other);
        }
        let theirs = other.roots(other_dir);
        self.roots(dir)
            .iter()
            .any(|a| theirs.iter().any(|b| holds(a, b) || holds(b, a)))
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
pub(super) fn holds(outer: &str, inner: &str) -> bool {
    let outer = std::path::Path::new(outer.trim_end_matches(['/', '\\']));
    std::path::Path::new(inner).starts_with(outer)
}

/// Whether a working-directory claim belongs to an actively executing process
/// or a queued run reserving the folder while waiting for a concurrency slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ClaimState {
    /// Queued in the concurrency-slot line; process has not started yet.
    Reserved,
    /// Process is actively running.
    #[default]
    Active,
}

/// A running run's claim on a folder, with enough to explain a conflict.
pub(super) struct WorkingDirClaim {
    pub(super) task_id: String,
    pub(super) tool_name: String,
    pub(super) prompt_summary: String,
    pub(super) writes: WriteScope,
    pub(super) state: ClaimState,
}

/// The live claim, in any folder, that a run in `dir` with `writes` would
/// collide with, and the folder it holds. `gone` names the tasks whose claim
/// no longer counts: the run asking, and runs that already finished (the
/// reaper releases their claims only at the tool's next call).
pub(super) fn conflicting<'a>(
    claims: &'a HashMap<String, Vec<WorkingDirClaim>>,
    dir: &str,
    writes: &WriteScope,
    gone: impl Fn(&str) -> bool,
) -> Option<(&'a str, &'a WorkingDirClaim)> {
    claims.iter().find_map(|(held_dir, held)| {
        held.iter()
            .filter(|claim| !gone(&claim.task_id))
            .find(|claim| claim.writes.collides(held_dir, writes, dir))
            .map(|claim| (held_dir.as_str(), claim))
    })
}

/// Why a run cannot start beside `claim` in `dir`.
pub(super) fn conflict_message(dir: &str, claim: &WorkingDirClaim) -> String {
    let status = match claim.state {
        ClaimState::Active => "is already working in",
        ClaimState::Reserved => "is already queued (reserved) for",
    };
    format!(
        "BLOCKED: Another CLI agent {status} {dir} and may change the same files \
         (task_id={}, agent={}, prompt=\"{}\", it may change: {}). \
         Wait for it to finish or cancel it, or give this run write_paths that do not overlap.",
        claim.task_id,
        claim.tool_name,
        claim.prompt_summary,
        claim.writes.words()
    )
}

/// Mark a queued run's reserved claim as active once its process starts.
pub(super) fn activate_claim(
    claims: &mut HashMap<String, Vec<WorkingDirClaim>>,
    dir: &str,
    task_id: &str,
) {
    if let Some(held) = claims.get_mut(dir)
        && let Some(claim) = held.iter_mut().find(|claim| claim.task_id == task_id)
    {
        claim.state = ClaimState::Active;
    }
}

/// A launch the owner sent to a coding agent directly that found another
/// run's claim on its folder: it waits for the folder instead of being
/// refused, and takes its own claim once nothing it would collide with is
/// held ([`try_take_folder`]).
pub(super) struct FolderWait {
    pub(super) dir: String,
    pub(super) writes: WriteScope,
    pub(super) prompt_summary: String,
    /// The tool name of the run that holds the folder, from the conflicting
    /// claim.
    pub(super) behind: String,
}

/// What a folder waiter found when it looked again.
pub(super) enum FolderTurn {
    /// It holds its claim now, released on drop until its run is tracked
    /// as started.
    Taken(WorkingDirClaimGuard),
    /// Still held by a run of this tool (`Some`), or an earlier waiter on
    /// the same folder goes first (`None`).
    Waits(Option<String>),
}

/// Take `wait`'s claim for `task_id` if no live claim on its folder would
/// collide with it and no earlier waiter on that folder is still waiting
/// (first come, first served). The claim of a run that already finished
/// does not count: the reaper releases it only at the tool's next call.
pub(super) async fn try_take_folder(
    running: &Mutex<HashMap<String, RunningCliAgent>>,
    claims_map: &Arc<WorkingDirClaims>,
    claims_changed: &Arc<tokio::sync::Notify>,
    task_id: &str,
    tool_name: &str,
    wait: &FolderWait,
) -> FolderTurn {
    // Lock ordering: running -> working_dir_claims.
    let running = running.lock().await;
    let finished = |id: &str| {
        running
            .get(id)
            .is_some_and(|agent| agent.finished.load(Ordering::Acquire))
    };
    let mine = running.get(task_id).map(|agent| agent.accepted_at);
    let earlier_waiter = mine.is_some_and(|accepted| {
        running.iter().any(|(id, agent)| {
            id != task_id
                && !agent.finished.load(Ordering::Acquire)
                && agent
                    .claim_dir
                    .as_deref()
                    .is_some_and(|dir| agent.write_scope.collides(dir, &wait.writes, &wait.dir))
                && agent.waits_for_folder().is_some()
                && (agent.accepted_at, id.as_str()) < (accepted, task_id)
        })
    });
    if earlier_waiter {
        return FolderTurn::Waits(None);
    }
    let mut claims = lock_claims(claims_map);
    let blocker = conflicting(&claims, &wait.dir, &wait.writes, |id| {
        id == task_id || finished(id)
    })
    .map(|(_, claim)| claim.tool_name.clone());
    if let Some(blocker) = blocker {
        return FolderTurn::Waits(Some(blocker));
    }
    claims
        .entry(wait.dir.clone())
        .or_default()
        .push(WorkingDirClaim {
            task_id: task_id.to_string(),
            tool_name: tool_name.to_string(),
            prompt_summary: wait.prompt_summary.clone(),
            writes: wait.writes.clone(),
            state: ClaimState::Reserved,
        });
    FolderTurn::Taken(WorkingDirClaimGuard::new(
        claims_map.clone(),
        claims_changed.clone(),
        wait.dir.clone(),
        task_id.to_string(),
    ))
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
    pub(super) claims_changed: Arc<tokio::sync::Notify>,
    pub(super) dir: String,
    pub(super) task_id: String,
    pub(super) armed: bool,
}

impl WorkingDirClaimGuard {
    pub(super) fn new(
        claims: Arc<WorkingDirClaims>,
        claims_changed: Arc<tokio::sync::Notify>,
        dir: String,
        task_id: String,
    ) -> Self {
        Self {
            claims,
            claims_changed,
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
        self.claims_changed.notify_waiters();
    }
}
