//! How turns run on the SDK kernel: the engine settings every agent in a
//! tree shares, and the owner's trusted file roots for one turn.

use tracing::info;

/// Model turns per run. Token budgets, the same-tool guard and the stuck
/// detector bound runaway work; this only stops a run that defeats all three.
/// Long coding tasks routinely need well over a hundred tool rounds.
pub const DEFAULT_MAX_TURNS: u32 = 150;
/// How turns run on the SDK kernel. Held by the root agent and inherited by
/// every child it spawns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdkEngineSettings {
    /// `[agent.engine] model`: pins every turn to this model instead of the
    /// daemon's own selection.
    pub model_override: Option<String>,
    pub max_turns: u32,
    /// `[authority] owner_default`: the owner's level when no session mode
    /// sets one.
    pub owner_default: crate::authority::OwnerLevel,
    /// `[authority] trusted_roots`: directories whose files are the owner's
    /// own, so reading them is not untrusted content.
    pub trusted_roots: Vec<String>,
    /// The configured API keys and tokens the outbound secret scan refuses.
    pub outbound_secrets: crate::approvals::ConfiguredSecrets,
}

impl Default for SdkEngineSettings {
    fn default() -> Self {
        Self {
            model_override: None,
            max_turns: DEFAULT_MAX_TURNS,
            owner_default: crate::authority::OwnerLevel::default(),
            trusted_roots: Vec::new(),
            outbound_secrets: Default::default(),
        }
    }
}

/// Where a turn's trusted file roots come from. Trusted roots, like
/// authority, come only from explicit owner grants: they are trusted only by
/// explicit grant, never by inference from the request. A path the request
/// or a mission names (the primary project scope, a scope a parent
/// handed down) may steer where tools run, but it
/// never vouches for content: callers pass only explicit values here.
#[derive(Default)]
pub struct TrustRootSources<'a> {
    /// Owner configuration: `[authority] trusted_roots`
    /// (`Agent::configured_trust_roots`). Path aliases are not grants.
    pub configured: &'a [String],
    /// The execution workspace (`turn_cwd()`, from config).
    pub cwd: Option<&'a str>,
    /// The turn's explicit workspace binding (`TurnContext::explicit_workspace`:
    /// a goal's `bind_workspace`), never an inferred one.
    pub bound_workspace: Option<&'a str>,
    /// The project root of the chat's active workspace grant (guest turns
    /// in a granted group only).
    pub workspace_grant: Option<&'a str>,
    /// The resolved authority's confinement, passed only alongside an
    /// explicit binding (its write targets derive from it).
    pub task_access: Option<&'a aidaemon_tools::semantics::ToolCallAccessManifest>,
    /// The spawning parent turn's trusted roots: the child receives them and
    /// never trusts a file outside them. `None` for a root turn.
    pub parent_roots: Option<&'a [String]>,
}

/// The owner's trusted file roots for one turn, from explicit grants only
/// (see [`TrustRootSources`]). Every source must be a narrow root (after
/// `~` expansion): the filesystem root, the home directory or a shared
/// temp root would trust every file the owner has. A child's roots are its
/// parent's, plus its own sources that lie inside them.
pub fn turn_trust_roots(sources: TrustRootSources<'_>) -> Vec<String> {
    let TrustRootSources {
        configured,
        cwd,
        bound_workspace,
        workspace_grant,
        task_access,
        parent_roots,
    } = sources;
    let write_targets = task_access
        .into_iter()
        .flat_map(|access| access.write_targets.iter())
        .map(|target| target.value.as_str());
    let mut roots: Vec<String> = configured
        .iter()
        .map(String::as_str)
        .chain(
            [cwd, bound_workspace, workspace_grant]
                .into_iter()
                .flatten(),
        )
        .chain(write_targets)
        .map(|root| shellexpand::tilde(root.trim()).into_owned())
        .filter(|root| is_narrow_root_resolved(root))
        .collect();
    if let Some(parent) = parent_roots {
        let within = aidaemon_tools::semantics::TrustRoots::new(parent);
        roots.retain(|root| within.contains(root));
        roots.splice(0..0, parent.iter().cloned());
    }
    let mut seen = std::collections::HashSet::new();
    roots.retain(|root| seen.insert(root.clone()));
    roots
}

/// Log, once at startup, the trusted roots the owner configured plus the
/// execution workspace, and which of them are dropped as too broad.
pub fn log_configured_trust_roots(configured: &[String], workspace: String) {
    let (kept, dropped): (Vec<String>, Vec<String>) = configured
        .iter()
        .map(|root| shellexpand::tilde(root.trim()).into_owned())
        .chain(std::iter::once(workspace))
        .partition(|root| is_narrow_root_resolved(root));
    info!(
        trusted_roots = ?kept,
        dropped_as_too_broad = ?dropped,
        "Trusted roots: [authority] trusted_roots and the execution workspace \
         (a goal's bind_workspace and a channel workspace grant add to these per turn)"
    );
}

/// `is_narrow_root` for both the root as written and, when it exists, the
/// directory it resolves to: a link to the home directory or `/` is as
/// broad as its target.
fn is_narrow_root_resolved(root: &str) -> bool {
    is_narrow_root(root)
        && std::fs::canonicalize(root)
            .map(|resolved| is_narrow_root(&resolved.to_string_lossy()))
            .unwrap_or(true)
}

/// A directory that holds only the owner's work: not the filesystem root,
/// the home directory or any of its ancestors (`/Users`), nor a shared temp
/// root (`/tmp`, `/private/tmp`, `/var/tmp`, the system temp directory).
/// Folders below them are fine.
fn is_narrow_root(dir: &str) -> bool {
    let dir = std::path::Path::new(dir.trim().trim_end_matches('/'));
    if !dir.is_absolute() || dir.parent().is_none() {
        return false;
    }
    if dirs::home_dir().is_some_and(|home| home.starts_with(dir)) {
        return false;
    }
    let temp = std::env::temp_dir();
    let mut shared: Vec<std::path::PathBuf> = ["/tmp", "/private/tmp", "/var/tmp"]
        .iter()
        .map(std::path::PathBuf::from)
        .collect();
    shared.extend(std::fs::canonicalize(&temp).ok());
    shared.push(temp);
    !shared.iter().any(|root| root.as_path() == dir)
}

/// Where the file tools resolve a relative path: the execution backend's
/// workspace (the daemon's working directory on the local backend).
pub fn turn_cwd(backend: &dyn aidaemon_tools::ExecutionBackend) -> String {
    backend.workspace_root().as_str().to_string()
}
