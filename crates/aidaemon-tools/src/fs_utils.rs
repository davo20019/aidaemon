use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::backend::{BackendKind, BackendPath, ExecutionBackend, ExecutionRequest};
use once_cell::sync::Lazy;

/// Directories to skip during recursive file walks (from the SDK tools crate).
pub use crate::DEFAULT_IGNORE_DIRS;

/// Files/directories that strongly indicate a project/workspace root.
pub const PROJECT_ROOT_MARKERS: &[&str] = &[
    ".git",
    "Cargo.toml",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "yarn.lock",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "CMakeLists.txt",
    "deno.json",
    "deno.jsonc",
    "bun.lockb",
    "wrangler.toml",
    "mix.exs",
    "pubspec.yaml",
    "Package.swift",
];

/// Like [`validate_path`], but with smart project-directory fallback.
///
/// When a relative path doesn't exist under cwd, this checks `~/projects/<path>`
/// before giving up. This avoids expensive `find ~` searches when the user
/// references a project by name (e.g., "blog.aidaemon.ai").
#[allow(dead_code)]
pub fn resolve_project_dir(backend: &dyn ExecutionBackend, path: &str) -> anyhow::Result<PathBuf> {
    let direct = validate_path(backend, path)?;
    if direct.exists() && direct.is_dir() {
        return Ok(direct);
    }

    // Only attempt fallback for simple names (not absolute, no path separators in
    // the original input). This prevents ~/projects/../../../etc/passwd tricks.
    let trimmed = path.trim();
    if trimmed.starts_with('/')
        || trimmed.starts_with('~')
        || trimmed.contains('/')
        || trimmed.contains("..")
    {
        anyhow::bail!("Not a valid directory: {}", path);
    }

    // Try ~/projects/<name>
    if let Some(home) = dirs::home_dir() {
        let fallback = home.join("projects").join(trimmed);
        if fallback.exists() && fallback.is_dir() {
            return Ok(fallback);
        }
    }

    anyhow::bail!(
        "Not a valid directory: {} (also checked ~/projects/{})",
        path,
        trimmed
    )
}

/// Resolves `~` and validates path safety. Returns canonical path.
pub fn validate_path(backend: &dyn ExecutionBackend, path: &str) -> anyhow::Result<PathBuf> {
    if backend.kind() != BackendKind::Local {
        return Ok(PathBuf::from(
            crate::backend::normalize_path_lexically(backend, path)?.as_str(),
        ));
    }

    let expanded = shellexpand::tilde(path).to_string();
    let p = PathBuf::from(&expanded);

    let joined = if p.is_absolute() {
        p.clone()
    } else {
        std::env::current_dir()?.join(&p)
    };

    // Normalize path components to remove `.` and resolve `..` entries
    // (e.g., "/foo/bar/./baz/../qux" -> "/foo/bar/qux")
    let normalized: PathBuf = joined.components().collect();

    // Reject paths that still contain `..` after normalization (shouldn't happen
    // with std::path::Component, but defense-in-depth)
    let normalized_str = normalized.to_string_lossy();
    if normalized_str.contains("/../") || normalized_str.ends_with("/..") {
        anyhow::bail!("Path traversal detected: {}", path);
    }

    Ok(normalized)
}

/// Returns true if the first directory component after root exists on disk.
/// `/tmp/notes_api` -> checks `/tmp` (exists) -> true
/// `/api/notes` -> checks `/api` (doesn't exist) -> false
pub fn first_dir_component_exists(path: &Path) -> bool {
    use std::path::Component;
    let mut components = path.components();
    match components.next() {
        Some(Component::RootDir) => {}
        _ => return true, // relative paths are assumed valid
    }
    match components.next() {
        Some(comp) => Path::new("/").join(comp).exists(),
        None => true, // bare "/" is fine
    }
}

/// Returns true when a path string likely refers to a file rather than a directory.
pub fn path_points_to_file(raw_path: &str) -> bool {
    Path::new(raw_path)
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| name.contains('.') && !name.starts_with('.') && !name.ends_with('.'))
}

fn path_root_looks_like_project(dir: &Path) -> bool {
    PROJECT_ROOT_MARKERS
        .iter()
        .any(|marker| dir.join(marker).exists())
}

/// Find the nearest ancestor that looks like a project root.
pub fn find_nearest_project_root(path: &Path) -> Option<PathBuf> {
    let mut current = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()?.to_path_buf()
    };
    if !current.exists() {
        return None;
    }

    loop {
        if path_root_looks_like_project(&current) {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Normalize a user/tool-supplied scope path.
/// If the path points into an existing project subtree, promote it to the nearest
/// recognizable project root so builds/deploys can reach repo-level files.
pub fn normalize_project_scope_path(
    backend: &dyn ExecutionBackend,
    path: &str,
) -> anyhow::Result<PathBuf> {
    if backend.kind() != BackendKind::Local {
        let mut normalized = validate_path(backend, path)?;
        if path_points_to_file(path.trim_end_matches('/')) {
            if let Some(parent) = normalized.parent() {
                normalized = parent.to_path_buf();
            }
        }
        return Ok(normalized);
    }

    let mut normalized = validate_path(backend, path)?;
    if !first_dir_component_exists(&normalized) {
        anyhow::bail!(
            "Path does not look like a real filesystem location: {}",
            path
        );
    }

    let trimmed = path.trim_end_matches('/');
    if normalized.is_file() || (!normalized.exists() && path_points_to_file(trimmed)) {
        if let Some(parent) = normalized.parent() {
            normalized = parent.to_path_buf();
        }
    }

    if normalized.exists() {
        if let Some(root) = find_nearest_project_root(&normalized) {
            normalized = root;
        }
    }

    Ok(normalized)
}

pub fn token_is_absolute_like(token: &str) -> bool {
    let bytes = token.as_bytes();
    let looks_windows_abs = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/');
    token.starts_with('/')
        || token.starts_with("~/")
        || token.starts_with("./")
        || token.starts_with("../")
        || looks_windows_abs
}

/// Join a possibly-relative token onto `base` and normalize `.` / `..`
/// lexically. Returns `None` when `..` would climb past the filesystem root.
fn join_normalized(base: &Path, token: &str) -> Option<PathBuf> {
    let expanded = shellexpand::tilde(token).to_string();
    let path = PathBuf::from(expanded);
    let joined = if path.is_absolute() {
        path
    } else {
        base.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    return None;
                }
            }
            value => normalized.push(value.as_os_str()),
        }
    }
    Some(normalized)
}

/// The directory a relative path reference is resolved against: the caller's
/// typed workspace when it has one, otherwise the daemon process cwd.
fn relative_reference_base(workspace_anchor: Option<&Path>) -> Option<PathBuf> {
    match workspace_anchor {
        Some(anchor) => Some(anchor.to_path_buf()),
        None => std::env::current_dir().ok(),
    }
}

/// Resolve a token as a filesystem reference only when its shape or the live
/// filesystem provides structural evidence that it is a path.
///
/// Explicit path forms (`/`, `~/`, `./`, `../`, Windows drive roots, and UNC
/// paths) may name a not-yet-created target. Bare relative references must be
/// anchored by an existing first component (or an exact existing entry) in the
/// working directory or an explicitly configured project root. This prevents
/// natural-language compounds such as `pros/cons` from becoming invented local
/// targets without maintaining a vocabulary of prose exceptions.
///
/// Relative references resolve against the daemon process cwd. Callers acting
/// for a task with a typed workspace must use
/// [`resolve_structural_filesystem_reference_from`] so the reference is anchored
/// to that workspace instead.
pub fn resolve_structural_filesystem_reference(
    backend: &dyn ExecutionBackend,
    raw: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    resolve_structural_filesystem_reference_from(backend, raw, alias_roots, None)
}

/// [`resolve_structural_filesystem_reference`] with an explicit anchor for
/// relative references.
///
/// `workspace_anchor` is the task's typed workspace (a goal's bound project
/// scope or a provisioned attempt workspace). When present, `src/content/posts`
/// means that directory *inside the workspace*; the daemon process cwd is an
/// unrelated deployment detail and is never consulted, and no other configured
/// project root may claim the reference either. Without an anchor the cwd and
/// alias-root fallbacks apply as before.
pub fn resolve_structural_filesystem_reference_from(
    backend: &dyn ExecutionBackend,
    raw: &str,
    alias_roots: &[String],
    workspace_anchor: Option<&Path>,
) -> Option<PathBuf> {
    let token = raw.trim();
    if token.is_empty() || token.contains("://") || token.contains('?') || token.contains('*') {
        return None;
    }

    let explicit = token_is_absolute_like(token)
        || token.starts_with("\\\\")
        || (token.starts_with('.') && (token.contains('/') || token.contains('\\')));
    let has_separator = token.contains('/') || token.contains('\\');

    if backend.kind() != BackendKind::Local {
        if explicit {
            return validate_path(backend, token).ok();
        }
        return resolve_projects_folder_alias(backend, token, alias_roots);
    }

    // A relative reference means something only against a workspace the run
    // was given. Without one, the daemon's own working directory is not the
    // user's project: `src/Controller` in a request about another repository
    // resolved into the daemon's checkout. Only the configured project roots
    // (below) may then resolve it.
    if workspace_anchor.is_none() && !explicit {
        return if has_separator {
            resolve_under_alias_roots(backend, token, alias_roots)
        } else {
            None
        };
    }
    let base = relative_reference_base(workspace_anchor)?;
    let direct = join_normalized(&base, token)?;
    // A path-shaped token in prose names this machine's filesystem only when
    // it reaches it: the path exists, or some directory above it other than
    // the filesystem root does (a file about to be created). `/api/items`
    // with no `/api` directory is a URL route or similar notation; making it
    // a filesystem target sends the agent searching a path that cannot exist
    // and lets a failed listing of it count as the observation.
    if explicit && !direct.exists() && !reaches_existing_directory(&direct) {
        return None;
    }
    // A bare existing directory name is still ordinary language ("tests",
    // "docs", "memory") often enough that it cannot establish a hard path
    // obligation. Exact bare files remain structural; directories require an
    // explicit path separator or a grounded semantic project reference.
    if explicit || (direct.exists() && (has_separator || direct.is_file())) {
        return Some(direct);
    }
    if !has_separator {
        return None;
    }

    if let Some(alias_path) = resolve_projects_folder_alias(backend, token, alias_roots) {
        return Some(alias_path);
    }

    let first_component = Path::new(token)
        .components()
        .find_map(|component| match component {
            std::path::Component::Normal(value) => Some(value),
            _ => None,
        })?;
    if base.join(first_component).exists() {
        return Some(direct);
    }
    // Under a typed workspace the reference is either inside that workspace or
    // it is not a filesystem reference for this task. Guessing another project
    // root would relocate the task's authority.
    if workspace_anchor.is_some() {
        return None;
    }

    resolve_under_alias_roots(backend, token, alias_roots)
}

/// A relative reference whose first component exists under a configured
/// project root (`[path_aliases] projects`), joined to that root.
fn resolve_under_alias_roots(
    backend: &dyn ExecutionBackend,
    token: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    if let Some(alias_path) = resolve_projects_folder_alias(backend, token, alias_roots) {
        return Some(alias_path);
    }
    let first_component = Path::new(token)
        .components()
        .find_map(|component| match component {
            std::path::Component::Normal(value) => Some(value),
            _ => None,
        })?;
    for raw_root in alias_roots {
        let Ok(root) = validate_path(backend, raw_root) else {
            continue;
        };
        if !root.is_dir() || !root.join(first_component).exists() {
            continue;
        }
        return Some(root.join(token.replace('\\', "/")));
    }
    None
}

/// Some ancestor of `path` below the filesystem root exists as a directory.
fn reaches_existing_directory(path: &Path) -> bool {
    path.ancestors()
        .skip(1)
        .filter(|ancestor| ancestor.parent().is_some())
        .any(Path::is_dir)
}

fn push_unique_search_root(roots: &mut Vec<PathBuf>, candidate: PathBuf) {
    if candidate.is_dir() && !roots.iter().any(|existing| existing == &candidate) {
        roots.push(candidate);
    }
}

const PROJECT_ROOT_CATALOG_TTL: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct CachedProjectRootCatalog {
    scanned_at: std::time::Instant,
    entries: Vec<(String, PathBuf)>,
}

static PROJECT_ROOT_CATALOG_CACHE: Lazy<Mutex<HashMap<PathBuf, CachedProjectRootCatalog>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn project_root_directory_entries(root: &Path) -> Vec<(String, PathBuf)> {
    if let Ok(cache) = PROJECT_ROOT_CATALOG_CACHE.lock() {
        if let Some(cached) = cache.get(root) {
            if cached.scanned_at.elapsed() <= PROJECT_ROOT_CATALOG_TTL {
                return cached.entries.clone();
            }
        }
    }

    let mut entries = Vec::new();
    let Ok(dir_entries) = std::fs::read_dir(root) else {
        return entries;
    };
    for entry in dir_entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        entries.push((
            entry.file_name().to_string_lossy().to_ascii_lowercase(),
            path,
        ));
    }

    if let Ok(mut cache) = PROJECT_ROOT_CATALOG_CACHE.lock() {
        cache.insert(
            root.to_path_buf(),
            CachedProjectRootCatalog {
                scanned_at: std::time::Instant::now(),
                entries: entries.clone(),
            },
        );
    }

    entries
}

pub fn project_search_roots(
    backend: &dyn ExecutionBackend,
    alias_roots: &[String],
) -> Vec<PathBuf> {
    if backend.kind() != BackendKind::Local {
        let mut roots = Vec::new();
        for raw_root in alias_roots {
            if let Ok(root) = validate_path(backend, raw_root) {
                if !roots.iter().any(|existing| existing == &root) {
                    roots.push(root);
                }
            }
        }
        let workspace = PathBuf::from(backend.workspace_root().as_str());
        if !roots.iter().any(|existing| existing == &workspace) {
            roots.push(workspace);
        }
        return roots;
    }

    let mut roots = Vec::new();
    for raw_root in alias_roots {
        let Ok(root) = validate_path(backend, raw_root) else {
            continue;
        };
        push_unique_search_root(&mut roots, root);
    }

    if let Ok(cwd) = std::env::current_dir() {
        if cwd.file_name().is_some_and(|name| name == "projects") {
            push_unique_search_root(&mut roots, cwd.clone());
        }
        push_unique_search_root(&mut roots, cwd.join("projects"));
    }

    if let Some(home) = dirs::home_dir() {
        push_unique_search_root(&mut roots, home.join("projects"));
    }

    roots
}

pub fn resolve_projects_folder_alias(
    backend: &dyn ExecutionBackend,
    raw_path: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    let trimmed = raw_path.trim();
    if token_is_absolute_like(trimmed) {
        return None;
    }
    let relative = trimmed
        .strip_prefix("./")
        .unwrap_or(trimmed)
        .trim_start_matches('/');
    let starts_with_projects =
        relative.starts_with("projects/") || relative.starts_with("projects\\");
    if !starts_with_projects {
        return None;
    }

    let suffix = relative
        .strip_prefix("projects/")
        .or_else(|| relative.strip_prefix("projects\\"))
        .unwrap_or("");
    for root in project_search_roots(backend, alias_roots) {
        let candidate = if suffix.is_empty() {
            root
        } else {
            root.join(suffix.replace('\\', "/"))
        };
        if backend.kind() != BackendKind::Local {
            return normalize_project_scope_path(backend, candidate.to_string_lossy().as_ref())
                .ok();
        }
        if candidate.parent().is_some_and(|parent| parent.is_dir()) {
            return Some(candidate);
        }
    }
    None
}

fn token_looks_like_named_project(raw: &str) -> bool {
    let token = raw
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
        .trim_end_matches(['.', '!', '?'])
        .to_ascii_lowercase();
    if token.is_empty()
        || token.contains("://")
        || token.contains('/')
        || token.contains('\\')
        || token.len() < 3
    {
        return false;
    }
    if !token.chars().any(|c| c.is_ascii_alphabetic()) || token.chars().all(|c| c.is_ascii_digit())
    {
        return false;
    }
    token.contains('.')
        || token.contains('-')
        || token.contains('_')
        || token.contains("project")
        || token.starts_with("app")
        || token.ends_with("app")
}

pub fn resolve_named_project_root(
    backend: &dyn ExecutionBackend,
    raw_name: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    let token = raw_name
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'))
        .trim_end_matches(['.', '!', '?'])
        .trim();
    if !token_looks_like_named_project(token) {
        return None;
    }

    let target = token.to_ascii_lowercase();
    if backend.kind() != BackendKind::Local {
        return project_search_roots(backend, alias_roots)
            .into_iter()
            .find_map(|root| {
                root.file_name()
                    .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case(&target))
                    .then_some(root)
            });
    }

    for root in project_search_roots(backend, alias_roots) {
        let direct = root.join(token);
        if direct.is_dir() {
            if let Ok(normalized) =
                normalize_project_scope_path(backend, direct.to_string_lossy().as_ref())
            {
                return Some(normalized);
            }
        }

        for (name, path) in project_root_directory_entries(&root) {
            if name != target {
                continue;
            }
            if let Ok(normalized) =
                normalize_project_scope_path(backend, path.to_string_lossy().as_ref())
            {
                return Some(normalized);
            }
        }
    }
    None
}

#[doc(hidden)]
fn explicit_project_search_roots(
    backend: &dyn ExecutionBackend,
    alias_roots: &[String],
) -> Vec<PathBuf> {
    if backend.kind() != BackendKind::Local {
        return alias_roots
            .iter()
            .filter_map(|root| validate_path(backend, root).ok())
            .collect();
    }

    let mut roots = Vec::new();
    for raw_root in alias_roots {
        let Ok(root) = validate_path(backend, raw_root) else {
            continue;
        };
        push_unique_search_root(&mut roots, root);
    }
    roots
}

#[doc(hidden)]
fn resolve_contextual_project_nickname_across_roots(
    backend: &dyn ExecutionBackend,
    raw_name: &str,
    roots: Vec<PathBuf>,
) -> Option<PathBuf> {
    let token = raw_name
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'))
        .trim_end_matches(['.', '!', '?'])
        .trim()
        .to_ascii_lowercase();
    if token.len() < 3
        || token.contains("://")
        || token.contains('/')
        || token.contains('\\')
        || !token.chars().any(|c| c.is_ascii_alphabetic())
        || token.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }

    if backend.kind() != BackendKind::Local {
        return roots.into_iter().find(|root| {
            root.file_name()
                .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case(&token))
        });
    }

    let mut matches = Vec::new();
    let dotted = format!("{token}.");
    let dashed = format!("{token}-");
    let underscored = format!("{token}_");
    for root in roots {
        for (name, path) in project_root_directory_entries(&root) {
            if name != token
                && !name.starts_with(&dotted)
                && !name.starts_with(&dashed)
                && !name.starts_with(&underscored)
            {
                continue;
            }
            let Ok(normalized) =
                normalize_project_scope_path(backend, path.to_string_lossy().as_ref())
            else {
                continue;
            };
            if find_nearest_project_root(&normalized).is_none_or(|root| root != normalized) {
                continue;
            }
            if !matches.iter().any(|existing| existing == &normalized) {
                matches.push(normalized);
            }
        }
    }

    if matches.len() == 1 {
        matches.into_iter().next()
    } else {
        None
    }
}

#[doc(hidden)]
pub fn resolve_contextual_project_nickname(
    backend: &dyn ExecutionBackend,
    raw_name: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    resolve_contextual_project_nickname_across_roots(
        backend,
        raw_name,
        project_search_roots(backend, alias_roots),
    )
}

#[doc(hidden)]
pub fn resolve_contextual_project_nickname_in_explicit_roots(
    backend: &dyn ExecutionBackend,
    raw_name: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    let explicit_roots = explicit_project_search_roots(backend, alias_roots);
    if explicit_roots.is_empty() {
        return resolve_contextual_project_nickname(backend, raw_name, alias_roots);
    }
    resolve_contextual_project_nickname_across_roots(backend, raw_name, explicit_roots)
}

/// Resolve a project-scope reference (path-like token or project nickname) to
/// a project root, with relative path tokens resolved against the daemon
/// process cwd. Task-bound callers use [`resolve_project_scope_reference_from`].
#[doc(hidden)]
pub fn resolve_project_scope_reference(
    backend: &dyn ExecutionBackend,
    raw: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    resolve_project_scope_reference_from(backend, raw, alias_roots, None)
}

/// Resolve a project-scope reference (path-like token or project nickname) to
/// a project root.
///
/// With `workspace_anchor`, a relative path token is joined onto the workspace
/// (never the daemon cwd) and promoted to the nearest project root from there,
/// so `src/content/posts` inside a bound blog checkout scopes to that checkout
/// rather than to whichever repository the daemon happens to run from. Without
/// an anchor the daemon cwd is the base.
pub fn resolve_project_scope_reference_from(
    backend: &dyn ExecutionBackend,
    raw: &str,
    alias_roots: &[String],
    workspace_anchor: Option<&Path>,
) -> Option<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let looks_path_like =
        token_is_absolute_like(trimmed) || trimmed.contains('/') || trimmed.contains('\\');
    if looks_path_like {
        if backend.kind() != BackendKind::Local {
            return normalize_project_scope_path(backend, trimmed).ok();
        }
        let path_for_resolution = if token_is_absolute_like(trimmed) {
            trimmed.to_string()
        } else {
            let base = relative_reference_base(workspace_anchor);
            let base_relative = base.and_then(|base| join_normalized(&base, trimmed));
            if let Some(candidate) = base_relative {
                if candidate.exists() {
                    candidate.to_string_lossy().to_string()
                } else if let Some(alias_candidate) =
                    resolve_projects_folder_alias(backend, trimmed, alias_roots)
                {
                    alias_candidate.to_string_lossy().to_string()
                } else {
                    candidate.to_string_lossy().to_string()
                }
            } else if let Some(alias_candidate) =
                resolve_projects_folder_alias(backend, trimmed, alias_roots)
            {
                alias_candidate.to_string_lossy().to_string()
            } else {
                trimmed.to_string()
            }
        };
        return normalize_project_scope_path(backend, &path_for_resolution).ok();
    }

    resolve_named_project_root(backend, trimmed, alias_roots)
}

/// Returns true if the path matches any sensitive pattern.
/// Uses path-component matching to avoid false positives on substrings
/// (e.g., "my_environment.txt" won't match ".env").
/// Where the pre-change copy of a local file is kept: one copy per file,
/// under `backup_root` (the host's data directory), so edits never leave
/// `.bak` files in the user's tree.
pub fn local_backup_path(backup_root: &Path, path: &Path) -> Option<PathBuf> {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(path.to_string_lossy().as_bytes());
    let tag: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let name = path.file_name()?.to_string_lossy();
    Some(backup_root.join(format!("{tag}-{name}")))
}

/// Keep one pre-change copy of `path` before a write or edit replaces it.
/// Local files go to [`local_backup_path`] under `backup_root` when one is
/// given; otherwise, and on other backends, the copy stays beside the file
/// (`<name>.<ext>.bak`), where that backend can reach it.
/// Best effort: a failed backup never blocks the change.
pub async fn backup_before_overwrite(
    backend: &crate::backend::SharedExecutionBackend,
    path: &BackendPath,
    backup_root: Option<&Path>,
) {
    if backend.kind() == BackendKind::Local {
        if let Some(backup) =
            backup_root.and_then(|root| local_backup_path(root, Path::new(path.as_str())))
        {
            if let Some(dir) = backup.parent() {
                let _ = tokio::fs::create_dir_all(dir).await;
            }
            let _ = tokio::fs::copy(path.as_str(), &backup).await;
            return;
        }
    }
    let backup = path.with_extension(&format!("{}.bak", path.extension().unwrap_or_default()));
    let _ = backend.copy(path, &backup).await;
}

/// Whether a path belongs to the host operating system's protected data
/// plane rather than task/workspace data. Raw file and terminal tools do not
/// receive access to these trees even when a semantic assessment repeats the
/// requested path; host inspection must go through a purpose-built typed tool
/// whose adapter owns and redacts that capability.
///
/// This is a structural filesystem policy, not natural-language matching.
/// Executable and dynamic-library reads needed to start a process remain in
/// `adapter_read_targets` and are evaluated separately from task data.
pub fn is_protected_host_data_path(path: &Path) -> bool {
    if !path.is_absolute() {
        return false;
    }

    #[cfg(unix)]
    {
        const PROTECTED_ROOTS: &[&str] = &[
            "/System",
            "/Library",
            "/bin",
            "/sbin",
            "/usr",
            "/etc",
            "/private/etc",
            "/private/var/root",
            "/var/root",
            "/dev",
            "/proc",
            "/sys",
        ];
        if PROTECTED_ROOTS
            .iter()
            .any(|root| path == Path::new(root) || path.starts_with(root))
        {
            return true;
        }
    }

    #[cfg(windows)]
    {
        let normalized = path
            .to_string_lossy()
            .replace('/', "\\")
            .to_ascii_lowercase();
        if [
            "c:\\windows",
            "c:\\program files",
            "c:\\program files (x86)",
        ]
        .iter()
        .any(|root| normalized == *root || normalized.starts_with(&format!("{root}\\")))
        {
            return true;
        }
    }

    false
}

/// Check if a file appears to be binary by reading first 8KB and looking for null bytes.
#[cfg_attr(not(test), allow(dead_code))]
pub async fn is_binary_file(path: &Path) -> anyhow::Result<bool> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; 8192];
    let n = file.read(&mut buf).await?;
    Ok(buf[..n].contains(&0))
}

/// Output from running a shell command.
#[derive(Debug)]
pub struct CommandOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

/// Run a shell command with timeout. Returns structured output.
pub async fn run_cmd(
    backend: &dyn ExecutionBackend,
    cmd: &str,
    working_dir: Option<&Path>,
    timeout_secs: u64,
) -> anyhow::Result<CommandOutput> {
    let working_dir = match working_dir {
        Some(dir) => Some(backend.resolve_path(&dir.to_string_lossy()).await?),
        None => None,
    };
    run_cmd_backend(backend, cmd, working_dir.as_ref(), timeout_secs).await
}

/// Run a command in the configured execution environment using an opaque
/// backend path for its working directory.
pub async fn run_cmd_backend(
    backend: &dyn ExecutionBackend,
    cmd: &str,
    working_dir: Option<&BackendPath>,
    timeout_secs: u64,
) -> anyhow::Result<CommandOutput> {
    let mut request = ExecutionRequest::shell(cmd);
    request.cwd = working_dir.cloned();
    let output = backend
        .execute(request, Duration::from_secs(timeout_secs))
        .await?;
    if output.timed_out {
        anyhow::bail!("Command timed out after {}s", timeout_secs);
    }
    Ok(CommandOutput {
        exit_code: output.exit_code,
        stdout: output.stdout_lossy(),
        stderr: output.stderr_lossy(),
        duration_ms: output.duration_ms,
    })
}

/// Returns true if the directory entry name should be skipped.
pub fn should_skip_dir(name: &str) -> bool {
    DEFAULT_IGNORE_DIRS.contains(&name)
}

/// Returns true if the filename looks like a test file (test_*.py, *_test.py).
pub fn is_test_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let lower = name.to_lowercase();
    (lower.starts_with("test_") && lower.ends_with(".py"))
        || (lower.ends_with("_test.py"))
        || (lower.starts_with("test_") && lower.ends_with(".js"))
        || (lower.ends_with(".test.js"))
        || (lower.ends_with(".test.ts"))
        || (lower.ends_with(".spec.js"))
        || (lower.ends_with(".spec.ts"))
}

/// Validate Python syntax by running py_compile. Returns None if valid, or
/// the error message if there's a SyntaxError. Runs with a 5-second timeout.
pub async fn validate_python_syntax(backend: &dyn ExecutionBackend, path: &Path) -> Option<String> {
    if path.extension().map(|e| e == "py").unwrap_or(false) {
        let path_str = path.to_string_lossy();
        let cmd = format!(
            "python3 -c \"import py_compile; py_compile.compile('{}', doraise=True)\"",
            path_str.replace('\'', "'\\''")
        );
        match run_cmd(backend, &cmd, None, 5).await {
            Ok(output) if output.exit_code != 0 => {
                let error = if !output.stderr.is_empty() {
                    output.stderr.trim().to_string()
                } else {
                    output.stdout.trim().to_string()
                };
                // Extract just the SyntaxError line for conciseness
                let relevant: String = error
                    .lines()
                    .filter(|l| {
                        l.contains("SyntaxError")
                            || l.contains("IndentationError")
                            || l.contains("TabError")
                            || l.trim().starts_with("File ")
                            || l.trim().starts_with('^')
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if relevant.is_empty() {
                    Some(error)
                } else {
                    Some(relevant)
                }
            }
            _ => None,
        }
    } else {
        None
    }
}

/// Build post-write diagnostics: Python syntax check + test file warning.
/// Returns a string to append to the tool result, or empty string.
pub async fn post_write_diagnostics(backend: &dyn ExecutionBackend, path: &Path) -> String {
    let mut notes = Vec::new();

    // Check Python syntax
    if let Some(syntax_err) = validate_python_syntax(backend, path).await {
        notes.push(format!(
            "\n⚠ SYNTAX ERROR detected in written file:\n{}\nFix the syntax error before proceeding.",
            syntax_err
        ));
    }

    // Warn about test file modification
    if is_test_file(path) {
        notes.push(
            "\n⚠ WARNING: You modified a test file. If your task is to implement code that passes tests, \
you should NOT modify the test file — implement the module to pass the tests as-is."
                .to_string(),
        );
    }

    notes.join("")
}
