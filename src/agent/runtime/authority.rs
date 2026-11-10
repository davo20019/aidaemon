//! Tool authority, resolved once per turn from explicit facts: who is asking
//! (role and workspace grant), the owner's configured level, the session mode
//! the owner chose, and a scheduled goal's bound workspace. The request
//! assessment never widens or narrows it.
//!
//! The result is the only authority the tool gate reads: the kernel's
//! `AuthorityHook` enforces its level and tool set per call, and the daemon's
//! scope check confines filesystem access to its `task_access`.

use crate::config::OwnerLevel;
use crate::traits::{ToolCallAccessManifest, ToolTargetHint, ToolTargetHintKind};
use crate::types::{UserRole, WorkspaceAccessLevel, WorkspaceGrant};

/// A session mode the owner chose with an explicit command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionMode {
    ReadOnly,
    Full,
    Unrestricted,
}

/// How long a `/readonly` or `/mode unrestricted` lasts.
pub(crate) const SESSION_MODE_TTL_HOURS: i64 = 4;

fn session_mode_key(session_id: &str) -> String {
    format!("authority_mode:{session_id}")
}

/// Persist the owner's session mode and return when it expires. `Full` is the
/// default, so it clears the setting (stored as an empty value, which reads as
/// no mode) instead of recording an expiry.
pub(crate) async fn set_session_mode(
    state: &(impl crate::traits::SettingsStore + ?Sized),
    session_id: &str,
    mode: SessionMode,
    now: chrono::DateTime<chrono::Utc>,
) -> anyhow::Result<chrono::DateTime<chrono::Utc>> {
    let expires_at = now + chrono::Duration::hours(SESSION_MODE_TTL_HOURS);
    let key = session_mode_key(session_id);
    let value = match mode {
        SessionMode::Full => String::new(),
        SessionMode::ReadOnly | SessionMode::Unrestricted => serde_json::json!({
            "mode": if mode == SessionMode::ReadOnly { "read_only" } else { "unrestricted" },
            "expires_at": expires_at.to_rfc3339(),
        })
        .to_string(),
    };
    state.set_setting(&key, &value).await?;
    Ok(expires_at)
}

/// The session mode in force at `now`, with its expiry (`None` expiry for a
/// fail-closed read). A cleared (empty) or valid-but-expired entry is no mode.
/// A read error or an entry that cannot be understood reads as read-only: a
/// narrowing the owner chose must never be lost to a storage fault.
pub(crate) async fn session_mode_with_expiry(
    state: &(impl crate::traits::SettingsStore + ?Sized),
    session_id: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<(SessionMode, Option<chrono::DateTime<chrono::Utc>>)> {
    let fail_closed = || Some((SessionMode::ReadOnly, None));
    let raw = match state.get_setting(&session_mode_key(session_id)).await {
        Ok(Some(raw)) if raw.is_empty() => return None,
        Ok(Some(raw)) => raw,
        Ok(None) => return None,
        Err(error) => {
            tracing::warn!(session_id, %error, "Could not read the session mode; treating the chat as read-only");
            return fail_closed();
        }
    };
    let parsed = serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|value| {
            let expires_at = chrono::DateTime::parse_from_rfc3339(value["expires_at"].as_str()?)
                .ok()?
                .with_timezone(&chrono::Utc);
            let mode = match value["mode"].as_str()? {
                "read_only" => SessionMode::ReadOnly,
                "unrestricted" => SessionMode::Unrestricted,
                "full" => SessionMode::Full,
                _ => return None,
            };
            Some((mode, expires_at))
        });
    match parsed {
        Some((_, expires_at)) if expires_at <= now => None,
        Some((mode, expires_at)) => Some((mode, Some(expires_at))),
        None => {
            tracing::warn!(
                session_id,
                "Unreadable session mode value; treating the chat as read-only"
            );
            fail_closed()
        }
    }
}

pub(crate) async fn session_mode(
    state: &(impl crate::traits::SettingsStore + ?Sized),
    session_id: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<SessionMode> {
    session_mode_with_expiry(state, session_id, now)
        .await
        .map(|(mode, _)| mode)
}

/// The ceiling a running turn (or child) reads before every call: read-only
/// iff the chat it answers to is in `/readonly` right now. It reads the live
/// setting (fail-closed on a read error), so `/readonly` sent mid-run narrows
/// the work already running; leaving the mode never widens a turn, because the
/// turn's own authority was fixed at its start.
pub(crate) struct SessionModeCeiling {
    state: std::sync::Arc<dyn crate::traits::StateStore>,
    mode_session: String,
}

impl SessionModeCeiling {
    pub(crate) fn new(
        state: std::sync::Arc<dyn crate::traits::StateStore>,
        mode_session: impl Into<String>,
    ) -> Self {
        Self {
            state,
            mode_session: mode_session.into(),
        }
    }
}

#[async_trait::async_trait]
impl aidaemon_core::AuthorityCeiling for SessionModeCeiling {
    async fn ceiling(&self) -> Option<aidaemon_core::Authority> {
        match session_mode(self.state.as_ref(), &self.mode_session, chrono::Utc::now()).await {
            Some(SessionMode::ReadOnly) => Some(aidaemon_core::Authority::read_only()),
            _ => None,
        }
    }
}

/// The explicit facts authority is resolved from.
pub(in crate::agent) struct AuthorityFacts<'a> {
    pub user_role: UserRole,
    pub owner_default: OwnerLevel,
    pub session_mode: Option<SessionMode>,
    pub workspace_grant: Option<&'a WorkspaceGrant>,
    pub bound_workspace: Option<&'a str>,
}

/// What this turn may do.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::agent) struct ResolvedAuthority {
    /// Level and tool set, enforced per call by the kernel hook.
    pub authority: aidaemon_core::Authority,
    /// The owner's level (`None` for guests and the public).
    pub owner_level: Option<OwnerLevel>,
    /// Filesystem confinement the scope check applies (`None`: none beyond
    /// the turn's ordinary scope lock and a guest's workspace grant).
    pub task_access: Option<ToolCallAccessManifest>,
}

/// How many plain sub-agents one request may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpawnFamilyLimits {
    /// At the same time, across the whole tree.
    pub max_concurrent: usize,
    /// In total, across the whole tree.
    pub max_total: usize,
    /// The tree tracks files and collisions; off restores unfenced writes.
    pub coordination: bool,
    /// Questions one child may put to the top-level agent itself.
    pub max_escalations_per_child: usize,
    /// Escalations of any kind in the whole tree.
    pub max_escalations_per_request: usize,
}

impl Default for SpawnFamilyLimits {
    fn default() -> Self {
        Self {
            max_concurrent: crate::config::DEFAULT_SUBAGENTS_MAX_CONCURRENT,
            max_total: crate::config::DEFAULT_SUBAGENTS_MAX_PER_REQUEST,
            coordination: true,
            max_escalations_per_child: crate::config::DEFAULT_SUBAGENTS_MAX_ESCALATIONS_PER_CHILD,
            max_escalations_per_request:
                crate::config::DEFAULT_SUBAGENTS_MAX_ESCALATIONS_PER_REQUEST,
        }
    }
}

#[derive(Debug, Default)]
struct SpawnCounts {
    active: usize,
    started: usize,
}

#[derive(Debug)]
struct SpawnFamilyState {
    limits: SpawnFamilyLimits,
    counts: std::sync::Mutex<SpawnCounts>,
    tokens: std::sync::Arc<std::sync::atomic::AtomicU64>,
    coordination: super::coordination::Coordination,
}

/// One request's tree of plain sub-agents: its running count, how many it
/// has started, and the tokens the whole tree has used. Clones share all of
/// it. A runtime handle, never serialized.
#[derive(Debug, Clone)]
pub(crate) struct SpawnFamily(std::sync::Arc<SpawnFamilyState>);

impl PartialEq for SpawnFamily {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

tokio::task_local! {
    static SPAWN_BEGAN: std::sync::Arc<std::sync::atomic::AtomicBool>;
}

/// Holds one running slot in a family until dropped. Its place in the
/// family's total is kept only once the child began running: a spawn that
/// fails during setup gives it back.
#[derive(Debug)]
pub(crate) struct SpawnPermit {
    family: SpawnFamily,
    began: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl SpawnPermit {
    /// Run `future` (the child's setup and run) so that a
    /// [`mark_child_began`] inside it marks this permit's child as begun.
    pub(crate) async fn scope<F: std::future::Future>(&self, future: F) -> F::Output {
        SPAWN_BEGAN.scope(self.began.clone(), future).await
    }
}

/// Run `future` inside `permit`'s scope when there is one.
pub(crate) async fn within_permit<F: std::future::Future>(
    permit: Option<&SpawnPermit>,
    future: F,
) -> F::Output {
    match permit {
        Some(permit) => permit.scope(future).await,
        None => future.await,
    }
}

/// The child of the enclosing [`SpawnPermit::scope`] has finished setup and
/// begun running. Outside any scope it does nothing.
pub(crate) fn mark_child_began() {
    let _ = SPAWN_BEGAN.try_with(|began| began.store(true, std::sync::atomic::Ordering::SeqCst));
}

impl Drop for SpawnPermit {
    fn drop(&mut self) {
        let mut counts = self
            .family
            .0
            .counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        counts.active = counts.active.saturating_sub(1);
        if !self.began.load(std::sync::atomic::Ordering::SeqCst) {
            counts.started = counts.started.saturating_sub(1);
        }
    }
}

impl SpawnFamily {
    pub(crate) fn new(limits: SpawnFamilyLimits) -> Self {
        Self(std::sync::Arc::new(SpawnFamilyState {
            limits,
            counts: std::sync::Mutex::default(),
            tokens: std::sync::Arc::default(),
            coordination: super::coordination::Coordination::new(
                limits.max_escalations_per_child,
                limits.max_escalations_per_request,
            ),
        }))
    }

    /// Whether this tree coordinates its members at all.
    pub(crate) fn coordinates(&self) -> bool {
        self.0.limits.coordination
    }

    /// The tree's shared knowledge, whether or not it has a child yet.
    pub(crate) fn coordination(&self) -> super::coordination::Coordination {
        self.0.coordination.clone()
    }

    /// The tree's shared knowledge, once it has a child: a lone agent has
    /// nobody to coordinate with.
    pub(crate) fn active_coordination(&self) -> Option<&super::coordination::Coordination> {
        let counts = self.0.counts.lock().unwrap_or_else(|e| e.into_inner());
        (self.0.limits.coordination && counts.started > 0).then_some(&self.0.coordination)
    }

    /// The tree's token counter, charged by every run in it.
    pub(crate) fn tokens(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        self.0.tokens.clone()
    }

    /// Admit one more child, or say why not.
    pub(crate) fn try_admit(&self) -> Result<SpawnPermit, String> {
        let limits = self.0.limits;
        let mut counts = self.0.counts.lock().unwrap_or_else(|e| e.into_inner());
        if counts.started >= limits.max_total {
            return Err(format!(
                "This request has already started {} sub-agents, which is the limit. \
                 Finish with the results you have, or do the remaining work directly.",
                limits.max_total
            ));
        }
        if counts.active >= limits.max_concurrent {
            return Err(format!(
                "{} sub-agents are already running for this request, which is the limit. \
                 Wait for one to finish, or do this work directly.",
                limits.max_concurrent
            ));
        }
        counts.active += 1;
        counts.started += 1;
        Ok(SpawnPermit {
            family: self.clone(),
            began: std::sync::Arc::default(),
        })
    }
}

/// What one `spawn_agent` call said about the child it starts.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct ChildDeclaration {
    pub parent_session: String,
    /// Files or directories the child will change, resolved.
    pub scope: Vec<String>,
    /// Who wins where that scope overlaps a running sibling's, if stated.
    pub precedence: Option<super::coordination::Precedence>,
    /// The child's place in spawn order, held until it joins the roster.
    pub claim: Option<std::sync::Arc<super::coordination::ScopeClaim>>,
}

/// What a spawned child inherits from the turn that spawned it: the turn's
/// actual authority (not a re-resolution at spawn time), the chat whose
/// `/readonly` narrows it, the turn's taint, and the root request egress is
/// judged against. The dispatcher hands it to `spawn_agent` in the call
/// context's extensions ([`parent_authority`]), never as an argument.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct InheritedAuthority {
    pub authority: aidaemon_core::Authority,
    pub owner_level: Option<OwnerLevel>,
    pub task_access: Option<ToolCallAccessManifest>,
    /// The chat whose live `/readonly` narrows the child while it runs.
    pub mode_session: Option<String>,
    /// The parent turn had read untrusted content (or started tainted).
    pub untrusted_context: bool,
    /// The root request the verifier judges the child's egress against.
    pub origin_request: Option<String>,
    /// The root request was owner-authored; a child of an untrusted-origin
    /// turn is never trusted (it only ever stays or turns false).
    pub origin_trusted: bool,
    /// The parent turn's trusted file roots, computed by `turn_trust_roots`
    /// on every spawn path: the child receives them and trusts nothing
    /// outside them. Required: a missing value is no trust, never "no
    /// constraint".
    pub trusted_roots: Vec<String>,
    /// Effects the owner declined in the parent run so far (its
    /// `RunState::declined`, which already holds what the parent itself
    /// inherited): the child's run starts with them, so a decline binds the
    /// child for the same effect and target. Required: a child never starts
    /// with fewer declines than its parent had.
    pub declined: Vec<aidaemon_core::EffectKey>,
    /// The parent run's live decline set: the child's run shares it, so a
    /// decline in the child binds the parent (and its other children)
    /// before their next call, and one the parent records after the child
    /// started binds the child. `None` (a spawn outside any running turn,
    /// such as a root goal dispatch) gives the child a set of its own. A
    /// runtime handle, never serialized.
    #[serde(skip)]
    pub shared_declines: Option<aidaemon_core::SharedDeclines>,
    /// The request's sub-agent tree, shared by every plain child in it.
    /// `None` starts a new tree. A runtime handle, never serialized.
    #[serde(skip)]
    pub spawn_family: Option<SpawnFamily>,
    /// What the `spawn_agent` call that starts this child declared. A
    /// runtime handle, never serialized.
    #[serde(skip)]
    pub child_declaration: Option<ChildDeclaration>,
    /// The kernel run that started this child, for tracing the family.
    #[serde(default)]
    pub parent_run: Option<aidaemon_core::RunId>,
}

/// The parent turn's authority the dispatcher handed `spawn_agent`: an
/// application fact (it carries the owner's configured level), so it rides
/// in the call context's extensions rather than as a typed SDK field.
pub(crate) fn parent_authority(
    facts: &crate::traits::ToolCallContext,
) -> Option<&InheritedAuthority> {
    facts.extensions.get::<InheritedAuthority>()
}

/// The union of two decline lists, in order, without repeats.
pub(crate) fn union_declines(
    first: &[aidaemon_core::EffectKey],
    second: &[aidaemon_core::EffectKey],
) -> Vec<aidaemon_core::EffectKey> {
    let mut all = first.to_vec();
    for key in second {
        if !all.contains(key) {
            all.push(key.clone());
        }
    }
    all
}

/// The most of the root request a child inherits (and the verifier is
/// shown at most 4000 characters of arguments anyway).
pub(crate) const INHERITED_REQUEST_CHARS: usize = 4000;

/// Every `InheritedAuthority` a child agent was created with (test-only
/// capture of the real spawn path).
#[cfg(test)]
pub(crate) static CREATED_CHILD_INHERITANCE: std::sync::Mutex<Vec<InheritedAuthority>> =
    std::sync::Mutex::new(Vec::new());

/// A child's own resolution narrowed by what its parent turn had: the
/// narrower authority, the lower owner level (a guest parent's `None` stays
/// `None`), and the parent's workspace confinement when it had one.
pub(in crate::agent) fn narrow_for_child(
    parent: &InheritedAuthority,
    own: ResolvedAuthority,
) -> ResolvedAuthority {
    ResolvedAuthority {
        authority: parent.authority.narrow(&own.authority),
        owner_level: match (parent.owner_level, own.owner_level) {
            (Some(parent), Some(own)) => Some(parent.min(own)),
            _ => None,
        },
        task_access: parent.task_access.clone().or(own.task_access),
    }
}

pub(in crate::agent) fn resolve_authority(facts: &AuthorityFacts<'_>) -> ResolvedAuthority {
    use aidaemon_core::Authority;
    match facts.user_role {
        UserRole::Owner => {
            let owner_level = Some(match facts.session_mode {
                Some(SessionMode::Full) => OwnerLevel::Full,
                Some(SessionMode::Unrestricted) => OwnerLevel::Unrestricted,
                Some(SessionMode::ReadOnly) | None => facts.owner_default,
            });
            if facts.session_mode == Some(SessionMode::ReadOnly) {
                return ResolvedAuthority {
                    authority: Authority::read_only(),
                    owner_level,
                    task_access: None,
                };
            }
            ResolvedAuthority {
                authority: Authority::full(),
                owner_level,
                task_access: facts.bound_workspace.and_then(bound_workspace_access),
            }
        }
        // A guest's grant keeps its own scoping in `tool_exec.rs`; without
        // one, every tool is refused there too.
        UserRole::Guest => ResolvedAuthority {
            authority: match facts.workspace_grant.map(|grant| grant.access) {
                Some(WorkspaceAccessLevel::Edit) => Authority::full(),
                Some(WorkspaceAccessLevel::Read) | None => Authority::read_only(),
            },
            owner_level: None,
            task_access: None,
        },
        UserRole::Public => ResolvedAuthority {
            authority: Authority::no_tools(),
            owner_level: None,
            task_access: None,
        },
    }
}

/// A scheduled goal bound to a workspace writes only inside it and runs
/// there; reads stay free (a grant on the filesystem root the workspace
/// lives under), as do network, commands and credentials.
fn bound_workspace_access(workspace: &str) -> Option<ToolCallAccessManifest> {
    let workspace = workspace.trim();
    if workspace.is_empty() {
        return None;
    }
    // A relative path has no anchor: it would resolve against whatever the
    // daemon's working directory is. Fail closed: a manifest that confines
    // writes to nothing, not an unanchored write root.
    if !std::path::Path::new(workspace).is_absolute() {
        tracing::warn!(
            workspace,
            "bound workspace is not an absolute path; confining writes to nothing"
        );
        return Some(ToolCallAccessManifest::default());
    }
    let write = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, workspace.to_string())?;
    let root = std::path::Path::new(workspace)
        .ancestors()
        .last()
        .map(|root| root.to_string_lossy().into_owned())
        .filter(|root| !root.is_empty())?;
    let read = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, root)?;
    Some(ToolCallAccessManifest {
        execution_cwd: Some(workspace.to_string()),
        read_targets: vec![read],
        write_targets: vec![write],
        adapter_read_targets: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::Authority;

    fn family(max_concurrent: usize, max_total: usize) -> SpawnFamily {
        SpawnFamily::new(SpawnFamilyLimits {
            max_concurrent,
            max_total,
            ..Default::default()
        })
    }

    #[test]
    fn a_family_refuses_past_its_running_limit_until_one_finishes() {
        let family = family(1, 10);
        let first = family.try_admit().unwrap();
        assert!(family.try_admit().unwrap_err().contains("already running"));
        drop(first);
        assert!(family.try_admit().is_ok());
    }

    #[tokio::test]
    async fn a_family_refuses_past_its_total_even_when_nothing_is_running() {
        let family = family(5, 2);
        for _ in 0..2 {
            let permit = family.try_admit().unwrap();
            permit.scope(async { mark_child_began() }).await;
        }
        assert!(family.try_admit().unwrap_err().contains("already started"));
    }

    /// A child that never began running gives its place in the total back;
    /// one that began keeps it.
    #[tokio::test]
    async fn only_a_child_that_began_counts_toward_the_total() {
        let family = family(5, 1);
        let failed_setup = family.try_admit().unwrap();
        failed_setup.scope(async {}).await;
        drop(failed_setup);
        let ran = family.try_admit().expect("a failed setup used no place");
        ran.scope(async { mark_child_began() }).await;
        drop(ran);
        assert!(family.try_admit().unwrap_err().contains("already started"));
    }

    /// Marking outside any permit's scope changes nothing.
    #[test]
    fn marking_outside_a_spawn_is_a_no_op() {
        mark_child_began();
    }

    #[test]
    fn clones_of_a_family_share_its_tokens() {
        let family = family(1, 1);
        family
            .clone()
            .tokens()
            .fetch_add(7, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            family.tokens().load(std::sync::atomic::Ordering::Relaxed),
            7
        );
    }

    async fn temp_store(path: &std::path::Path) -> crate::state::SqliteStateStore {
        let embedding = std::sync::Arc::new(
            crate::memory::embeddings::EmbeddingService::new().expect("embedding service"),
        );
        crate::state::SqliteStateStore::new(path.to_str().unwrap(), 100, None, embedding)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn session_mode_expires() {
        let dir = tempfile::tempdir().unwrap();
        let store = temp_store(&dir.path().join("a.db")).await;
        let t = chrono::Utc::now();
        let expires = set_session_mode(&store, "synthetic-session", SessionMode::ReadOnly, t)
            .await
            .unwrap();
        assert_eq!(expires, t + chrono::Duration::hours(4));
        let before = t + chrono::Duration::minutes(3 * 60 + 59);
        let after = t + chrono::Duration::minutes(4 * 60 + 1);
        assert_eq!(
            session_mode(&store, "synthetic-session", before).await,
            Some(SessionMode::ReadOnly)
        );
        assert_eq!(session_mode(&store, "synthetic-session", after).await, None);
        assert_eq!(session_mode(&store, "other-session", t).await, None);
    }

    #[tokio::test]
    async fn session_mode_full_clears_the_mode() {
        let dir = tempfile::tempdir().unwrap();
        let store = temp_store(&dir.path().join("a.db")).await;
        let t = chrono::Utc::now();
        set_session_mode(&store, "s", SessionMode::Unrestricted, t)
            .await
            .unwrap();
        assert_eq!(
            session_mode(&store, "s", t).await,
            Some(SessionMode::Unrestricted)
        );
        set_session_mode(&store, "s", SessionMode::Full, t)
            .await
            .unwrap();
        assert_eq!(session_mode(&store, "s", t).await, None);
    }

    #[tokio::test]
    async fn session_mode_corrupt_value_reads_read_only() {
        use crate::traits::store_prelude::*;
        let dir = tempfile::tempdir().unwrap();
        let store = temp_store(&dir.path().join("a.db")).await;
        let t = chrono::Utc::now();
        for corrupt in [
            "not json",
            r#"{"mode":"read_only","expires_at":"yesterday"}"#,
            r#"{"mode":"god","expires_at":"2999-01-01T00:00:00Z"}"#,
            r#"{"mode":"read_only"}"#,
        ] {
            store
                .set_setting("authority_mode:corrupt", corrupt)
                .await
                .unwrap();
            assert_eq!(
                session_mode(&store, "corrupt", t).await,
                Some(SessionMode::ReadOnly),
                "{corrupt}"
            );
        }
        // Cleared (empty) and expired-but-valid entries are no mode.
        store
            .set_setting("authority_mode:corrupt", "")
            .await
            .unwrap();
        assert_eq!(session_mode(&store, "corrupt", t).await, None);
    }

    #[tokio::test]
    async fn session_mode_read_error_reads_read_only() {
        struct FailingSettings;
        #[async_trait::async_trait]
        impl crate::traits::SettingsStore for FailingSettings {
            async fn get_setting(&self, _key: &str) -> anyhow::Result<Option<String>> {
                anyhow::bail!("synthetic settings failure")
            }
        }
        assert_eq!(
            session_mode(&FailingSettings, "s", chrono::Utc::now()).await,
            Some(SessionMode::ReadOnly)
        );
    }

    #[tokio::test]
    async fn session_mode_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("restart.db");
        let t = chrono::Utc::now();
        {
            let store = temp_store(&path).await;
            set_session_mode(&store, "synthetic-session", SessionMode::ReadOnly, t)
                .await
                .unwrap();
        }
        let reopened = temp_store(&path).await;
        assert_eq!(
            session_mode(&reopened, "synthetic-session", t).await,
            Some(SessionMode::ReadOnly)
        );
    }

    fn facts(role: UserRole) -> AuthorityFacts<'static> {
        AuthorityFacts {
            user_role: role,
            owner_default: OwnerLevel::Full,
            session_mode: None,
            workspace_grant: None,
            bound_workspace: None,
        }
    }

    fn grant(access: WorkspaceAccessLevel) -> WorkspaceGrant {
        WorkspaceGrant {
            platform: "telegram".to_string(),
            workspace_id: "synthetic-workspace".to_string(),
            channel_id: "synthetic-channel".to_string(),
            user_id: "telegram:synthetic-user-1".to_string(),
            project_root: "/tmp/aidaemon-test/grant".to_string(),
            access,
            expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
        }
    }

    #[test]
    fn resolve_authority_owner_read_only_mode_is_read_only() {
        let resolved = resolve_authority(&AuthorityFacts {
            session_mode: Some(SessionMode::ReadOnly),
            owner_default: OwnerLevel::Unrestricted,
            bound_workspace: Some("/tmp/aidaemon-test/ws"),
            ..facts(UserRole::Owner)
        });
        assert_eq!(resolved.authority, Authority::read_only());
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Unrestricted));
        assert!(resolved.task_access.is_none());
    }

    #[test]
    fn resolve_authority_owner_without_mode_is_full_at_the_configured_level() {
        let resolved = resolve_authority(&facts(UserRole::Owner));
        assert_eq!(resolved.authority, Authority::full());
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Full));
        assert!(resolved.task_access.is_none());

        let resolved = resolve_authority(&AuthorityFacts {
            owner_default: OwnerLevel::Unrestricted,
            ..facts(UserRole::Owner)
        });
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Unrestricted));
    }

    #[test]
    fn resolve_authority_owner_mode_sets_the_level() {
        for (mode, level) in [
            (SessionMode::Full, OwnerLevel::Full),
            (SessionMode::Unrestricted, OwnerLevel::Unrestricted),
        ] {
            let resolved = resolve_authority(&AuthorityFacts {
                session_mode: Some(mode),
                owner_default: if level == OwnerLevel::Full {
                    OwnerLevel::Unrestricted
                } else {
                    OwnerLevel::Full
                },
                ..facts(UserRole::Owner)
            });
            assert_eq!(resolved.authority, Authority::full());
            assert_eq!(resolved.owner_level, Some(level));
            assert!(resolved.task_access.is_none());
        }
    }

    #[test]
    fn resolve_authority_owner_bound_workspace_confines_writes_not_reads() {
        let resolved = resolve_authority(&AuthorityFacts {
            bound_workspace: Some("/tmp/aidaemon-test/ws"),
            ..facts(UserRole::Owner)
        });
        assert_eq!(resolved.authority, Authority::full());
        assert_eq!(resolved.owner_level, Some(OwnerLevel::Full));
        let access = resolved.task_access.expect("bound workspace manifest");
        assert_eq!(
            access.execution_cwd.as_deref(),
            Some("/tmp/aidaemon-test/ws")
        );
        assert_eq!(
            access.write_targets,
            [ToolTargetHint::new(
                ToolTargetHintKind::ProjectScope,
                "/tmp/aidaemon-test/ws".to_string()
            )
            .unwrap()]
        );
        // Reads anywhere: the scope check admits them; writes outside do not.
        let call = |reads: &[&str], writes: &[&str]| ToolCallAccessManifest {
            read_targets: reads
                .iter()
                .map(|p| ToolTargetHint::new(ToolTargetHintKind::Path, p.to_string()).unwrap())
                .collect(),
            write_targets: writes
                .iter()
                .map(|p| ToolTargetHint::new(ToolTargetHintKind::Path, p.to_string()).unwrap())
                .collect(),
            ..ToolCallAccessManifest::default()
        };
        let check = |manifest: &ToolCallAccessManifest| {
            crate::agent::tool_execution::access_manifest_scope_check(
                "write_file",
                manifest,
                Some(&access),
                &[],
                access.execution_cwd.as_deref(),
                &crate::tools::DaemonPaths::default(),
            )
        };
        assert!(check(&call(&["/tmp/aidaemon-test/elsewhere/notes.txt"], &[])).is_none());
        assert!(check(&call(&[], &["/tmp/aidaemon-test/ws/a.txt"])).is_none());
        assert!(check(&call(&[], &["a.txt"])).is_none());
        assert!(matches!(
            check(&call(&[], &["/tmp/aidaemon-test/outside.txt"])),
            Some(crate::agent::tool_execution::ScopeCheck::Rejected(_))
        ));
    }

    #[test]
    fn resolve_authority_guest_read_grant_is_read_only() {
        let grant = grant(WorkspaceAccessLevel::Read);
        let resolved = resolve_authority(&AuthorityFacts {
            workspace_grant: Some(&grant),
            ..facts(UserRole::Guest)
        });
        assert_eq!(resolved.authority, Authority::read_only());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }

    #[test]
    fn resolve_authority_guest_edit_grant_is_full() {
        let grant = grant(WorkspaceAccessLevel::Edit);
        let resolved = resolve_authority(&AuthorityFacts {
            workspace_grant: Some(&grant),
            ..facts(UserRole::Guest)
        });
        assert_eq!(resolved.authority, Authority::full());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }

    #[test]
    fn resolve_authority_guest_without_grant_is_read_only() {
        let resolved = resolve_authority(&AuthorityFacts {
            session_mode: Some(SessionMode::Unrestricted),
            bound_workspace: Some("/tmp/aidaemon-test/ws"),
            ..facts(UserRole::Guest)
        });
        assert_eq!(resolved.authority, Authority::read_only());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }

    /// The integration harness cannot show this: a guest without a grant is
    /// offered no tool, so the kernel refuses an unknown tool before the
    /// dispatcher. Here the dispatcher itself still refuses it.
    #[tokio::test]
    async fn guest_without_grant_still_gets_no_active_workspace_grant() {
        let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
            crate::testing::MockProvider::new(),
            vec![std::sync::Arc::new(crate::tools::WriteFileTool::new(
                crate::execution::active_execution_backend(),
            ))],
            None,
        )
        .await
        .unwrap();
        let ctx = crate::agent::tool_exec::ToolExecCtx {
            file_fence: None,
            mcp_tool: None,
            owner_approval_by_kernel: false,
            session_id: "guest-session",
            task_id: None,
            status_tx: None,
            channel_visibility: crate::types::ChannelVisibility::PrivateGroup,
            channel_id: Some("telegram:synthetic-group"),
            project_scope: None,
            trusted: false,
            user_role: UserRole::Guest,
            workspace_grant: None,
            mandate_authority: None,
            tool_call_id: None,
            mutation_forbidden: false,
            scope_escalation: None,
            turn_authority: None,
            trust_roots: None,
            lone_root_call: false,
        };
        let error = harness
            .agent
            .execute_tool_with_watchdog_outcome(
                "write_file",
                r#"{"path":"/tmp/aidaemon-test/guest.txt","content":"x"}"#,
                &ctx,
            )
            .await
            .expect_err("a guest without a grant is refused");
        assert!(
            error.to_string().contains("no active workspace grant"),
            "{error}"
        );
    }

    fn inherited(
        authority: Authority,
        owner_level: Option<OwnerLevel>,
        task_access: Option<ToolCallAccessManifest>,
    ) -> InheritedAuthority {
        InheritedAuthority {
            authority,
            owner_level,
            task_access,
            mode_session: Some("tg_synthetic_parent".to_string()),
            untrusted_context: false,
            origin_request: Some("Summarize the synthetic notes.".to_string()),
            origin_trusted: true,
            trusted_roots: Vec::new(),
            declined: Vec::new(),
            shared_declines: None,
            spawn_family: None,
            child_declaration: None,
            parent_run: None,
        }
    }

    #[test]
    fn relative_bound_workspace_confines_writes_to_nothing() {
        for workspace in ["relative/ws", "ws", "./ws", "../ws"] {
            let manifest = bound_workspace_access(workspace)
                .unwrap_or_else(|| panic!("{workspace}: still a manifest, so it fails closed"));
            assert!(manifest.write_targets.is_empty(), "{workspace}");
            assert!(manifest.execution_cwd.is_none(), "{workspace}");
        }
        let absolute = bound_workspace_access("/tmp/aidaemon-test/ws").unwrap();
        assert_eq!(absolute.write_targets.len(), 1);
    }

    #[test]
    fn narrow_for_child_never_widens() {
        let bound = bound_workspace_access("/tmp/aidaemon-test/parent-ws");
        assert!(bound.is_some());
        let parent = inherited(
            Authority::read_only(),
            Some(OwnerLevel::Unrestricted),
            bound.clone(),
        );
        let own = ResolvedAuthority {
            authority: Authority::full(),
            owner_level: Some(OwnerLevel::Full),
            task_access: None,
        };
        let child = narrow_for_child(&parent, own);
        assert_eq!(child.authority, Authority::read_only());
        assert_eq!(child.owner_level, Some(OwnerLevel::Full));
        assert_eq!(child.task_access, bound);

        // The lower owner level wins from either side.
        let child = narrow_for_child(
            &inherited(Authority::full(), Some(OwnerLevel::Full), None),
            ResolvedAuthority {
                authority: Authority::full(),
                owner_level: Some(OwnerLevel::Unrestricted),
                task_access: None,
            },
        );
        assert_eq!(child.owner_level, Some(OwnerLevel::Full));

        // A parent without a workspace keeps the child's own.
        let own_bound = bound_workspace_access("/tmp/aidaemon-test/child-ws");
        let child = narrow_for_child(
            &inherited(Authority::full(), Some(OwnerLevel::Full), None),
            ResolvedAuthority {
                authority: Authority::full(),
                owner_level: Some(OwnerLevel::Full),
                task_access: own_bound.clone(),
            },
        );
        assert_eq!(child.task_access, own_bound);

        // A guest parent (no owner level) gives a child none.
        let child = narrow_for_child(
            &inherited(Authority::full(), None, None),
            ResolvedAuthority {
                authority: Authority::full(),
                owner_level: Some(OwnerLevel::Unrestricted),
                task_access: None,
            },
        );
        assert_eq!(child.owner_level, None);
    }

    /// `/mode unrestricted` on a shared chat never relaxes a
    /// guest turn there, or a child spawned from it.
    #[tokio::test]
    async fn unrestricted_never_reaches_a_guest_or_its_child() {
        use crate::agent::untrusted::untrusted_mode;
        let dir = tempfile::tempdir().unwrap();
        let store = temp_store(&dir.path().join("a.db")).await;
        let t = chrono::Utc::now();
        let shared = "tg_synthetic_shared";
        set_session_mode(&store, shared, SessionMode::Unrestricted, t)
            .await
            .unwrap();
        let mode = session_mode(&store, shared, t).await;
        assert_eq!(mode, Some(SessionMode::Unrestricted));

        let guest = resolve_authority(&AuthorityFacts {
            session_mode: mode,
            owner_default: OwnerLevel::Unrestricted,
            ..facts(UserRole::Guest)
        });
        assert_eq!(guest.owner_level, None);
        assert_eq!(
            untrusted_mode(guest.owner_level),
            aidaemon_core::UntrustedMode::Enforce
        );

        // The guest turn's child, even resolving as the owner under the same
        // mode, stays enforced.
        let parent = InheritedAuthority {
            authority: guest.authority.clone(),
            owner_level: guest.owner_level,
            task_access: guest.task_access.clone(),
            mode_session: Some(shared.to_string()),
            untrusted_context: true,
            origin_request: Some("Post this for me.".to_string()),
            origin_trusted: false,
            trusted_roots: Vec::new(),
            declined: Vec::new(),
            shared_declines: None,
            spawn_family: None,
            child_declaration: None,
            parent_run: None,
        };
        let own = resolve_authority(&AuthorityFacts {
            session_mode: mode,
            owner_default: OwnerLevel::Unrestricted,
            ..facts(UserRole::Owner)
        });
        assert_eq!(own.owner_level, Some(OwnerLevel::Unrestricted));
        let child = narrow_for_child(&parent, own);
        assert_eq!(child.owner_level, None);
        assert_eq!(
            untrusted_mode(child.owner_level),
            aidaemon_core::UntrustedMode::Enforce
        );
    }

    #[test]
    fn resolve_authority_public_has_no_tools() {
        let resolved = resolve_authority(&AuthorityFacts {
            session_mode: Some(SessionMode::Full),
            ..facts(UserRole::Public)
        });
        assert_eq!(resolved.authority, Authority::no_tools());
        assert_eq!(resolved.owner_level, None);
        assert!(resolved.task_access.is_none());
    }
}
