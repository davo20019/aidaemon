use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use once_cell::sync::Lazy;
use tokio::io::AsyncReadExt;

use super::process_control::{configure_command_for_process_group, terminate_process_tree};

/// Directories to skip during recursive file walks.
pub const DEFAULT_IGNORE_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".cache",
    ".parcel-cache",
    "coverage",
    ".turbo",
    ".gradle",
    ".idea",
    ".vs",
];

/// Sensitive path patterns that should not be written to.
pub const SENSITIVE_PATTERNS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".env",
    ".key",
    ".pem",
    ".aws/credentials",
    ".netrc",
    ".docker/config.json",
    "credentials",
    "id_rsa",
    "id_ed25519",
];

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

/// Resolves `~` and validates path safety. Returns canonical path.
pub fn validate_path(path: &str) -> anyhow::Result<PathBuf> {
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
pub fn normalize_project_scope_path(path: &str) -> anyhow::Result<PathBuf> {
    let mut normalized = validate_path(path)?;
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

pub fn project_search_roots(alias_roots: &[String]) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for raw_root in alias_roots {
        let Ok(root) = validate_path(raw_root) else {
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

pub fn resolve_projects_folder_alias(raw_path: &str, alias_roots: &[String]) -> Option<PathBuf> {
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
    for root in project_search_roots(alias_roots) {
        let candidate = if suffix.is_empty() {
            root
        } else {
            root.join(suffix.replace('\\', "/"))
        };
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

pub fn resolve_named_project_root(raw_name: &str, alias_roots: &[String]) -> Option<PathBuf> {
    let token = raw_name
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'))
        .trim_end_matches(['.', '!', '?'])
        .trim();
    if !token_looks_like_named_project(token) {
        return None;
    }

    let target = token.to_ascii_lowercase();
    for root in project_search_roots(alias_roots) {
        let direct = root.join(token);
        if direct.is_dir() {
            if let Ok(normalized) = normalize_project_scope_path(direct.to_string_lossy().as_ref())
            {
                return Some(normalized);
            }
        }

        for (name, path) in project_root_directory_entries(&root) {
            if name != target {
                continue;
            }
            if let Ok(normalized) = normalize_project_scope_path(path.to_string_lossy().as_ref()) {
                return Some(normalized);
            }
        }
    }
    None
}

fn explicit_project_search_roots(alias_roots: &[String]) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for raw_root in alias_roots {
        let Ok(root) = validate_path(raw_root) else {
            continue;
        };
        push_unique_search_root(&mut roots, root);
    }
    roots
}

fn resolve_contextual_project_nickname_across_roots(
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
            let Ok(normalized) = normalize_project_scope_path(path.to_string_lossy().as_ref())
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

pub fn resolve_contextual_project_nickname(
    raw_name: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    resolve_contextual_project_nickname_across_roots(raw_name, project_search_roots(alias_roots))
}

pub fn resolve_contextual_project_nickname_in_explicit_roots(
    raw_name: &str,
    alias_roots: &[String],
) -> Option<PathBuf> {
    let explicit_roots = explicit_project_search_roots(alias_roots);
    if explicit_roots.is_empty() {
        return resolve_contextual_project_nickname(raw_name, alias_roots);
    }
    resolve_contextual_project_nickname_across_roots(raw_name, explicit_roots)
}

pub fn resolve_project_scope_reference(raw: &str, alias_roots: &[String]) -> Option<PathBuf> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let looks_path_like =
        token_is_absolute_like(trimmed) || trimmed.contains('/') || trimmed.contains('\\');
    if looks_path_like {
        let path_for_resolution = if token_is_absolute_like(trimmed) {
            trimmed.to_string()
        } else {
            let cwd_relative = validate_path(trimmed).ok();
            if let Some(candidate) = cwd_relative {
                if candidate.exists() {
                    candidate.to_string_lossy().to_string()
                } else if let Some(alias_candidate) =
                    resolve_projects_folder_alias(trimmed, alias_roots)
                {
                    alias_candidate.to_string_lossy().to_string()
                } else {
                    candidate.to_string_lossy().to_string()
                }
            } else if let Some(alias_candidate) =
                resolve_projects_folder_alias(trimmed, alias_roots)
            {
                alias_candidate.to_string_lossy().to_string()
            } else {
                trimmed.to_string()
            }
        };
        return normalize_project_scope_path(&path_for_resolution).ok();
    }

    resolve_named_project_root(trimmed, alias_roots)
}

/// Returns true if the path matches any sensitive pattern.
/// Uses path-component matching to avoid false positives on substrings
/// (e.g., "my_environment.txt" won't match ".env").
pub fn is_sensitive_path(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    SENSITIVE_PATTERNS.iter().any(|pat| {
        if pat.contains('/') {
            // Multi-component patterns (e.g., ".aws/credentials"): substring match is fine
            // because the slash provides enough specificity.
            path_str.contains(pat)
        } else {
            // Single-component patterns: match as an exact path component or filename.
            path.components()
                .any(|c| c.as_os_str().to_string_lossy().eq_ignore_ascii_case(pat))
                || path
                    .file_name()
                    .is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(pat))
        }
    })
}

/// Check if a file appears to be binary by reading first 8KB and looking for null bytes.
pub async fn is_binary_file(path: &Path) -> anyhow::Result<bool> {
    use tokio::io::AsyncReadExt;
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; 8192];
    let n = file.read(&mut buf).await?;
    Ok(buf[..n].contains(&0))
}

/// Format file content with line numbers.
pub fn format_with_line_numbers(content: &str, start_line: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total = start_line + lines.len();
    let width = total.to_string().len().max(3);
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>width$} | {}", start_line + i + 1, line, width = width))
        .collect::<Vec<_>>()
        .join("\n")
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
    cmd: &str,
    working_dir: Option<&Path>,
    timeout_secs: u64,
) -> anyhow::Result<CommandOutput> {
    let start = std::time::Instant::now();

    let mut command = tokio::process::Command::new("sh");
    command.arg("-c").arg(cmd);
    if let Some(dir) = working_dir {
        command.current_dir(dir);
    }
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    command.kill_on_drop(true);
    configure_command_for_process_group(&mut command);

    let mut child = command.spawn()?;
    let child_pid = child.id().unwrap_or(0);
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("Failed to capture stderr"))?;

    let stdout_task = tokio::spawn(async move {
        let mut reader = stdout;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    });
    let stderr_task = tokio::spawn(async move {
        let mut reader = stderr;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await?;
        Ok::<Vec<u8>, std::io::Error>(buf)
    });

    let timeout = Duration::from_secs(timeout_secs);
    let mut timed_out = false;
    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => Some(status),
        Ok(Err(e)) => anyhow::bail!("Command failed to execute: {}", e),
        Err(_) => {
            timed_out = true;
            terminate_process_tree(child_pid, &mut child, Duration::from_secs(2)).await;
            None
        }
    };

    if timed_out {
        stdout_task.abort();
        stderr_task.abort();
        let _ = stdout_task.await;
        let _ = stderr_task.await;
        anyhow::bail!("Command timed out after {}s", timeout_secs);
    }

    let stdout = match stdout_task.await {
        Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).to_string(),
        _ => String::new(),
    };
    let stderr = match stderr_task.await {
        Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).to_string(),
        _ => String::new(),
    };
    let status = status.expect("status must exist when command did not time out");

    let duration_ms = start.elapsed().as_millis() as u64;
    Ok(CommandOutput {
        exit_code: status.code().unwrap_or(-1),
        stdout,
        stderr,
        duration_ms,
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
pub async fn validate_python_syntax(path: &Path) -> Option<String> {
    if path.extension().map(|e| e == "py").unwrap_or(false) {
        let path_str = path.to_string_lossy();
        let cmd = format!(
            "python3 -c \"import py_compile; py_compile.compile('{}', doraise=True)\"",
            path_str.replace('\'', "'\\''")
        );
        match run_cmd(&cmd, None, 5).await {
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
pub async fn post_write_diagnostics(path: &Path) -> String {
    let mut notes = Vec::new();

    // Check Python syntax
    if let Some(syntax_err) = validate_python_syntax(path).await {
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

/// Check if a shell command contains operators that could be dangerous.
pub fn contains_shell_operator(cmd: &str) -> bool {
    for ch in [';', '|', '`', '\n'] {
        if cmd.contains(ch) {
            return true;
        }
    }
    for op in ["&&", "||", "$(", ">(", "<("] {
        if cmd.contains(op) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_validate_path_home() {
        let result = validate_path("~/test.txt");
        assert!(result.is_ok());
        let p = result.unwrap();
        assert!(p.is_absolute());
        assert!(!p.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_validate_path_traversal() {
        let result = validate_path("/tmp/../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_absolute() {
        let result = validate_path("/tmp/test.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_dot_normalized() {
        // Passing "." should resolve to cwd without trailing "/."
        let result = validate_path(".").unwrap();
        let result_str = result.to_string_lossy();
        assert!(
            !result_str.ends_with("/."),
            "validate_path(\".\") should not end with trailing dot, got: {}",
            result_str
        );
        assert_eq!(result, std::env::current_dir().unwrap());
    }

    #[test]
    fn test_validate_path_subdir_dot_normalized() {
        // "src/." should normalize to just "cwd/src"
        let result = validate_path("src/.").unwrap();
        let result_str = result.to_string_lossy();
        assert!(
            !result_str.ends_with("/."),
            "validate_path(\"src/.\") should not end with trailing dot, got: {}",
            result_str
        );
    }

    #[test]
    fn test_first_dir_component_exists() {
        assert!(first_dir_component_exists(Path::new("/tmp/test")));
        assert!(!first_dir_component_exists(Path::new("/api/notes")));
        assert!(first_dir_component_exists(Path::new("src/main.rs")));
    }

    #[test]
    fn test_path_points_to_file() {
        assert!(path_points_to_file("src/main.rs"));
        assert!(path_points_to_file("/tmp/app/package.json"));
        assert!(!path_points_to_file("/tmp/app/src"));
        assert!(!path_points_to_file(".hidden"));
    }

    #[test]
    fn test_find_nearest_project_root_prefers_nearest_marker() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        let app = root.join("apps").join("web");
        let src = app.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(root.join("pnpm-workspace.yaml"), "packages:\n  - apps/*\n").unwrap();
        std::fs::write(app.join("package.json"), r#"{"name":"web"}"#).unwrap();

        let found = find_nearest_project_root(&src).expect("nearest project root");
        assert_eq!(found, app);
    }

    #[test]
    fn test_normalize_project_scope_path_promotes_existing_subdir_to_project_root() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(root.join("package.json"), r#"{"name":"demo"}"#).unwrap();

        let normalized =
            normalize_project_scope_path(src.to_string_lossy().as_ref()).expect("normalized");
        assert_eq!(normalized, root);
    }

    #[test]
    fn test_normalize_project_scope_path_keeps_non_project_target_dir() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("new-site");
        let normalized =
            normalize_project_scope_path(target.to_string_lossy().as_ref()).expect("normalized");
        assert_eq!(normalized, target);
    }

    #[test]
    fn test_normalize_project_scope_path_preserves_existing_dotted_directory() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("blog.aidaemon.ai");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("wrangler.toml"), "name = \"blog\"\n").unwrap();

        let normalized =
            normalize_project_scope_path(target.to_string_lossy().as_ref()).expect("normalized");
        assert_eq!(normalized, target);
    }

    #[test]
    fn test_resolve_named_project_root_from_alias_roots() {
        let dir = tempfile::tempdir().unwrap();
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").unwrap();

        let resolved = resolve_named_project_root(
            "blog.aidaemon.ai",
            &[alias_root.to_string_lossy().to_string()],
        )
        .expect("resolved");
        assert_eq!(resolved, project);
    }

    #[test]
    fn test_resolve_contextual_project_nickname_in_explicit_roots_prefers_alias_roots() {
        let dir = tempfile::tempdir().unwrap();
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").unwrap();

        let resolved = resolve_contextual_project_nickname_in_explicit_roots(
            "fairfax",
            &[alias_root.to_string_lossy().to_string()],
        )
        .expect("resolved");
        assert_eq!(resolved, project);
    }

    #[test]
    fn test_resolve_project_scope_reference_handles_named_and_projects_alias_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").unwrap();
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let named = resolve_project_scope_reference("blog.aidaemon.ai", &alias_roots)
            .expect("named project");
        assert_eq!(named, project);

        let aliased = resolve_project_scope_reference("projects/blog.aidaemon.ai", &alias_roots)
            .expect("aliased project");
        assert_eq!(aliased, project);
    }

    #[test]
    fn test_is_sensitive_path() {
        assert!(is_sensitive_path(Path::new("/home/user/.ssh/id_rsa")));
        assert!(is_sensitive_path(Path::new("/home/user/.env")));
        assert!(is_sensitive_path(Path::new("/home/user/.gnupg/key")));
        assert!(!is_sensitive_path(Path::new("/home/user/code/main.rs")));
    }

    #[tokio::test]
    async fn test_is_binary_file_text() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "Hello, world!\nLine 2\n").unwrap();
        assert!(!is_binary_file(f.path()).await.unwrap());
    }

    #[tokio::test]
    async fn test_is_binary_file_binary() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xD8, 0xFF, 0x00, 0x10]).unwrap();
        assert!(is_binary_file(f.path()).await.unwrap());
    }

    #[test]
    fn test_format_with_line_numbers() {
        let result = format_with_line_numbers("foo\nbar\nbaz", 0);
        assert!(result.contains("  1 | foo"));
        assert!(result.contains("  2 | bar"));
        assert!(result.contains("  3 | baz"));
    }

    #[test]
    fn test_format_with_line_numbers_offset() {
        let result = format_with_line_numbers("line10\nline11", 9);
        assert!(result.contains("10 | line10"));
        assert!(result.contains("11 | line11"));
    }

    #[tokio::test]
    async fn test_run_cmd_echo() {
        let out = run_cmd("echo hello", None, 5).await.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[tokio::test]
    async fn test_run_cmd_timeout() {
        let result = run_cmd("sleep 10", None, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_run_cmd_timeout_kills_child_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("leaked.txt");
        let cmd = format!(
            "(sleep 2; echo leaked > \"{}\") & wait",
            marker.to_string_lossy()
        );

        let result = run_cmd(&cmd, None, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timed out"));

        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(
            !marker.exists(),
            "timed out command should not leave detached child writing files"
        );
    }

    #[tokio::test]
    async fn test_run_cmd_working_dir() {
        let dir = tempfile::tempdir().unwrap();
        let out = run_cmd("pwd", Some(dir.path()), 5).await.unwrap();
        assert_eq!(out.exit_code, 0);
    }

    #[test]
    fn test_should_skip_dir() {
        assert!(should_skip_dir(".git"));
        assert!(should_skip_dir("node_modules"));
        assert!(should_skip_dir("target"));
        assert!(!should_skip_dir("src"));
        assert!(!should_skip_dir("lib"));
    }

    #[test]
    fn test_contains_shell_operator() {
        assert!(contains_shell_operator("ls; rm"));
        assert!(contains_shell_operator("cat | grep"));
        assert!(contains_shell_operator("a && b"));
        assert!(contains_shell_operator("echo $(cmd)"));
        assert!(!contains_shell_operator("cargo build --release"));
        assert!(!contains_shell_operator("ls -la /tmp"));
    }
}
