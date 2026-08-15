//! Project-hint and project-scope extraction from user text and history.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. These helpers resolve filesystem paths and named project
//! nicknames into normalized project scopes, and choose a primary scope for a
//! turn.

use super::*;

#[cfg(test)]
fn is_generic_non_project_token(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "with"
            | "from"
            | "into"
            | "about"
            | "using"
            | "make"
            | "create"
            | "modern"
            | "frontend"
            | "backend"
            | "design"
            | "developer"
            | "senior"
            | "best"
            | "style"
            | "tailwind"
            | "react"
            | "html"
            | "css"
            | "javascript"
            | "typescript"
            | "project"
            | "workspace"
            | "directory"
            | "folder"
    )
}

#[cfg(test)]
fn is_likely_filename(token: &str) -> bool {
    let Some((name, ext)) = token.rsplit_once('.') else {
        return false;
    };
    // If the name portion itself contains dots (e.g. "blog.aidaemon" in "blog.aidaemon.ai"),
    // this looks like a domain name or multi-label hostname, not a plain filename.
    if name.contains('.') {
        return false;
    }
    !name.is_empty()
        && !ext.is_empty()
        && ext.len() <= 8
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
}

#[cfg(test)]
pub(in crate::agent) fn token_looks_like_filesystem_path(token: &str) -> bool {
    crate::tools::fs_utils::resolve_structural_filesystem_reference(token, &[]).is_some()
}

fn token_looks_like_project_scope_path(token: &str, alias_roots: &[String]) -> bool {
    crate::tools::fs_utils::resolve_structural_filesystem_reference(token, alias_roots).is_some()
}

#[cfg(test)]
fn is_common_path_segment(token: &str) -> bool {
    matches!(
        token,
        "users"
            | "user"
            | "home"
            | "workspace"
            | "workspaces"
            | "projects"
            | "repos"
            | "repo"
            | "src"
            | "apps"
            | "app"
            | "packages"
            | "package"
            | "code"
            | "tmp"
            | "var"
            | "usr"
            | "opt"
            | "local"
            | "dev"
            | "documents"
            | "downloads"
            | "desktop"
    )
}

#[cfg(test)]
fn normalize_project_component(raw: &str, allow_plain_names: bool) -> Option<String> {
    let token = raw
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
        .to_ascii_lowercase();
    if token.is_empty() || token.contains("://") {
        return None;
    }
    if token.len() < 3 {
        return None;
    }
    if !token.chars().any(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if is_generic_non_project_token(&token) {
        return None;
    }
    if is_likely_filename(&token) {
        return None;
    }

    if !allow_plain_names {
        let looks_project_like = token.contains("project")
            || token.contains('-')
            || token.contains('_')
            || token.starts_with("app")
            || token.ends_with("app");
        if !looks_project_like {
            return None;
        }
    }

    Some(token)
}

#[cfg(test)]
fn extract_project_hint_from_path_like_token(raw_token: &str) -> Option<String> {
    let trimmed = raw_token
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'))
        .to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    let path_source = if let Some((_, after_scheme)) = trimmed.split_once("://") {
        let (_, path) = after_scheme.split_once('/')?;
        path.to_string()
    } else {
        trimmed
    };

    let mut parts: Vec<String> = Vec::new();
    for raw_part in path_source.split(['/', '\\']) {
        let part = raw_part
            .split(['?', '#'])
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
            .to_ascii_lowercase();
        if !part.is_empty() {
            parts.push(part);
        }
    }

    for part in parts.iter().rev() {
        if is_common_path_segment(part) || is_likely_filename(part) {
            continue;
        }
        if let Some(candidate) = normalize_project_component(part, true) {
            return Some(candidate);
        }
    }

    None
}

#[cfg(test)]
fn normalize_project_hint(raw: &str, path_like: bool) -> Option<String> {
    let uri_like = raw.contains("://");
    if path_like || uri_like {
        if let Some(path_hint) = extract_project_hint_from_path_like_token(raw) {
            return Some(path_hint);
        }
        if uri_like {
            return None;
        }
    }
    normalize_project_component(raw, false)
}

#[cfg(test)]
fn push_project_hint(hints: &mut Vec<String>, hint: String, max_hints: usize) {
    if hints.len() >= max_hints || hints.iter().any(|existing| existing == &hint) {
        return;
    }
    hints.push(hint);
}

#[cfg(test)]
pub(super) fn extract_project_hints_from_text(
    text: &str,
    hints: &mut Vec<String>,
    max_hints: usize,
    path_only: bool,
) {
    for raw in text.split_whitespace() {
        if hints.len() >= max_hints {
            break;
        }
        let uri_like = raw.contains("://");
        let path_like = uri_like
            || crate::tools::fs_utils::resolve_structural_filesystem_reference(raw, &[]).is_some();
        if path_only && !path_like {
            continue;
        }
        if let Some(normalized) = normalize_project_hint(raw, path_like) {
            push_project_hint(hints, normalized, max_hints);
        }
    }
}

#[cfg(test)]
pub(super) fn extract_project_hints_from_history(
    history: &[Message],
    current_user_text: &str,
    max_hints: usize,
    include_history_hints: bool,
) -> Vec<String> {
    let mut hints: Vec<String> = Vec::new();
    extract_project_hints_from_text(current_user_text, &mut hints, max_hints, false);

    if !include_history_hints {
        return hints;
    }

    for msg in history.iter().rev() {
        if hints.len() >= max_hints {
            break;
        }
        if let Some(content) = msg.content.as_deref() {
            match msg.role.as_str() {
                "user" | "assistant" => {
                    extract_project_hints_from_text(content, &mut hints, max_hints, false);
                }
                "tool" => {
                    // Tool payloads are noisy; only trust explicit paths/URIs there.
                    extract_project_hints_from_text(content, &mut hints, max_hints, true);
                }
                _ => {}
            }
        }
    }
    hints
}

pub(super) fn normalize_project_scope_path_with_aliases(
    raw_path: &str,
    alias_roots: &[String],
) -> Option<String> {
    crate::tools::fs_utils::resolve_project_scope_reference(raw_path, alias_roots)
        .map(|path| path.to_string_lossy().to_string())
}

pub(super) fn push_project_scope(scopes: &mut Vec<String>, scope: String, max_scopes: usize) {
    if scopes.len() >= max_scopes || scopes.iter().any(|existing| existing == &scope) {
        return;
    }
    scopes.push(scope);
}

pub(super) fn extract_project_scopes_from_text(
    text: &str,
    scopes: &mut Vec<String>,
    max_scopes: usize,
    alias_roots: &[String],
) {
    extract_project_scopes_from_text_inner(text, scopes, max_scopes, alias_roots);
}

/// Extract only explicit filesystem paths (tokens with `~/`, `/`, `./`, etc.)
/// from the text.  Contextual nickname matching (e.g. "modern" → modern-plants-site)
/// is skipped.  Use this for the current user message to avoid false positives
/// from common English words that happen to match project directory prefixes.
pub(super) fn extract_explicit_path_scopes_from_text(
    text: &str,
    scopes: &mut Vec<String>,
    max_scopes: usize,
    alias_roots: &[String],
) {
    extract_project_scopes_from_text_inner(text, scopes, max_scopes, alias_roots);
}

fn extract_project_scopes_from_text_inner(
    text: &str,
    scopes: &mut Vec<String>,
    max_scopes: usize,
    alias_roots: &[String],
) {
    for raw in text.split_whitespace() {
        if scopes.len() >= max_scopes {
            break;
        }
        let token = raw
            .trim_matches(|c: char| {
                c.is_ascii_whitespace()
                    || matches!(
                        c,
                        '`' | '\''
                            | '"'
                            | ','
                            | ';'
                            | ':'
                            | '.'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                    )
            })
            .trim();
        if token.is_empty() || token.contains("://") {
            continue;
        }
        let scope = if token_looks_like_project_scope_path(token, alias_roots) {
            normalize_project_scope_path_with_aliases(token, alias_roots)
        } else {
            None
        };
        if let Some(scope) = scope {
            push_project_scope(scopes, scope, max_scopes);
        }
    }
}

fn scope_looks_like_project_root(scope: &str) -> bool {
    let backend = crate::execution::active_execution_backend();
    if backend.kind() != crate::execution::BackendKind::Local {
        let Ok(path) = crate::execution::normalize_active_path_lexically(scope) else {
            return false;
        };
        return std::path::Path::new(path.as_str())
            .starts_with(std::path::Path::new(backend.workspace_root().as_str()));
    }

    let Ok(path) = crate::tools::fs_utils::validate_path(scope) else {
        return false;
    };
    if !path.is_dir() {
        return false;
    }
    crate::tools::fs_utils::find_nearest_project_root(&path).is_some_and(|root| root == path)
}

pub(super) fn choose_primary_project_scope(scopes: &[String]) -> Option<String> {
    scopes
        .iter()
        .find(|scope| scope_looks_like_project_root(scope))
        .cloned()
        .or_else(|| scopes.first().cloned())
}

/// When the user mentions multiple paths in a single message (e.g.
/// `~/projects/blog/posts/file.md` and `~/projects/blog/tweets.md`),
/// compute the narrowest scope that encompasses ALL mentioned paths.
///
/// Without this, only the first scope is used, and writes to sibling
/// directories are blocked by the scope guard (e.g. scope locks to
/// `/blog/posts` but the user also asked to write `/blog/tweets.md`).
pub(super) fn unify_current_turn_scopes(scopes: &[String]) -> Option<String> {
    if scopes.len() <= 1 {
        return scopes.first().cloned();
    }

    // Fast path: if one scope is a parent of ALL others, use it directly
    for scope in scopes {
        let prefix = format!("{}/", scope.trim_end_matches('/'));
        if scopes
            .iter()
            .all(|other| other == scope || other.starts_with(&prefix))
        {
            return Some(scope.clone());
        }
    }
    // No single scope is an ancestor of all others — compute common ancestor
    let components: Vec<Vec<&str>> = scopes.iter().map(|s| s.split('/').collect()).collect();
    let first = &components[0];
    let mut common_len = 0;
    for i in 0..first.len() {
        if components.iter().all(|c| c.get(i) == first.get(i)) {
            common_len = i + 1;
        } else {
            break;
        }
    }
    if common_len == 0 {
        return scopes.first().cloned();
    }
    let ancestor: String = first[..common_len].join("/");
    // Safety: the ancestor should be at most 1 level shallower than the
    // shallowest input scope. This prevents computing an overly broad
    // scope (like /Users/name/projects) when unrelated project paths
    // appear together.
    let min_depth = scopes
        .iter()
        .map(|s| s.matches('/').count())
        .min()
        .unwrap_or(0);
    let ancestor_depth = ancestor.matches('/').count();
    if ancestor_depth >= min_depth.saturating_sub(1) {
        Some(ancestor)
    } else {
        scopes.first().cloned()
    }
}

pub(super) fn resolve_primary_project_scope(
    extracted_primary_scope: Option<String>,
    inherited_project_scope: Option<&str>,
    allow_multi_project_scope: bool,
    allow_inherited_scope: bool,
) -> Option<String> {
    if extracted_primary_scope.is_some() {
        return extracted_primary_scope;
    }

    if allow_multi_project_scope {
        return allow_inherited_scope
            .then(|| inherited_project_scope.map(ToOwned::to_owned))
            .flatten();
    }

    if allow_inherited_scope {
        inherited_project_scope.map(ToOwned::to_owned)
    } else {
        None
    }
}

pub(super) fn extract_project_scopes_from_history(
    history: &[Message],
    current_user_text: &str,
    max_scopes: usize,
    include_history_scopes: bool,
    alias_roots: &[String],
) -> Vec<String> {
    let mut scopes = Vec::new();
    extract_project_scopes_from_text(current_user_text, &mut scopes, max_scopes, alias_roots);

    if !include_history_scopes {
        return scopes;
    }

    for msg in history.iter().rev() {
        if scopes.len() >= max_scopes {
            break;
        }
        if msg.role != "user" {
            continue;
        }
        let Some(content) = msg.content.as_deref() else {
            continue;
        };
        extract_project_scopes_from_text(content, &mut scopes, max_scopes, alias_roots);
    }

    scopes
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn project_hint_extraction_finds_project_name() {
        let history = vec![
            msg(
                "user",
                "Please work in test-project and modernize index.html",
            ),
            msg("assistant", "Which sections should I prioritize?"),
            msg("user", "Do what you consider best."),
        ];
        let hints =
            extract_project_hints_from_history(&history, "Do what you consider best.", 6, true);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project in project hints, got {:?}",
            hints
        );
    }
    #[test]
    fn project_hint_extraction_handles_file_uri_paths() {
        let history = vec![msg(
            "assistant",
            "Opened file:///Users/testuser/projects/test-project/index.html",
        )];
        let hints =
            extract_project_hints_from_history(&history, "Do what you consider best.", 8, true);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project from file URI, got {:?}",
            hints
        );
        assert!(
            hints.iter().all(|h| h != "index.html"),
            "filename should not be treated as project hint: {:?}",
            hints
        );
    }
    #[test]
    fn project_hint_extraction_scans_tool_messages_path_only() {
        let history = vec![msg(
            "tool",
            "Using terminal: cd ~/projects/test-project && npm run build",
        )];
        let hints = extract_project_hints_from_history(&history, "make it modern", 8, true);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project from tool output path, got {:?}",
            hints
        );
    }
    #[test]
    fn project_hint_extraction_ignores_history_for_new_tasks() {
        let history = vec![
            msg(
                "user",
                "Please work in ~/projects/blog.aidaemon.ai/src/content/posts",
            ),
            msg("assistant", "Which posts should I update?"),
        ];
        let hints = extract_project_hints_from_history(&history, "Why?", 8, false);
        assert!(
            hints.is_empty(),
            "new-task hints should not inherit prior project context: {:?}",
            hints
        );
    }
    #[test]
    fn project_scope_extraction_uses_explicit_current_path() {
        let history = vec![msg("assistant", "Earlier we touched ~/projects/old-one")];
        let current = "Please work in ~/projects/new-one/src and review the files.";
        let scopes = extract_project_scopes_from_history(&history, current, 4, true, &[]);
        assert!(!scopes.is_empty());
        assert!(
            scopes[0].contains("new-one"),
            "expected first scope to come from current request, got {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_ignores_history_for_new_tasks() {
        let history = vec![msg(
            "assistant",
            "Found symlink: /Users/davidloor/.openclaw and other directories.",
        )];
        let current = "Find all Rust files in the aidaemon project that contain async fn.";
        let scopes = extract_project_scopes_from_history(&history, current, 4, false, &[]);
        assert!(
            scopes.iter().all(|scope| !scope.contains(".openclaw")),
            "new task scope should not inherit prior assistant paths: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_keeps_history_for_followups() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.path().to_string_lossy().to_string();
        let history = vec![
            msg(
                "user",
                &format!("Please work in {} and inspect async functions.", dir_path),
            ),
            msg("assistant", "Should I proceed with a full scan?"),
        ];
        let current = "Yes, do it.";
        let scopes = extract_project_scopes_from_history(&history, current, 4, true, &[]);
        assert!(
            scopes.iter().any(|scope| scope.contains(&*dir_path)),
            "followup should carry prior project scope when explicit: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_carries_exact_path_from_prior_user_request() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project_name = format!("scope-project-{}", uuid::Uuid::new_v4().simple());
        let project = alias_root.join(&project_name);
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let history = vec![msg(
            "user",
            &format!("Please deploy {} when you can.", project.display()),
        )];
        let scopes =
            extract_project_scopes_from_history(&history, "Yes, do it.", 4, true, &alias_roots);
        assert_eq!(scopes, vec![project.to_string_lossy().to_string()]);
    }
    #[test]
    fn choose_primary_project_scope_prefers_real_project_root() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let blog = alias_root.join("blog.aidaemon.ai");
        let logs = root.path().join("Library/Logs/aidaemon");
        std::fs::create_dir_all(&blog).expect("create blog");
        std::fs::write(blog.join("wrangler.toml"), "name = \"blog\"\n").expect("blog wrangler");
        std::fs::create_dir_all(&logs).expect("create logs");

        let chosen = choose_primary_project_scope(&[
            logs.to_string_lossy().to_string(),
            blog.to_string_lossy().to_string(),
        ]);
        assert_eq!(chosen, Some(blog.to_string_lossy().to_string()));
    }
    #[test]
    fn unify_scopes_single_entry_returns_it() {
        let scopes = vec!["/Users/david/projects/blog".to_string()];
        assert_eq!(
            unify_current_turn_scopes(&scopes),
            Some("/Users/david/projects/blog".to_string())
        );
    }
    #[test]
    fn unify_scopes_empty_returns_none() {
        let scopes: Vec<String> = vec![];
        assert_eq!(unify_current_turn_scopes(&scopes), None);
    }
    #[test]
    fn unify_scopes_parent_child_returns_parent() {
        // User mentions ~/projects/blog/posts/file.md and ~/projects/blog/tweets.md
        // which normalize to /blog/posts and /blog respectively
        let scopes = vec![
            "/Users/david/projects/blog/posts".to_string(),
            "/Users/david/projects/blog".to_string(),
        ];
        assert_eq!(
            unify_current_turn_scopes(&scopes),
            Some("/Users/david/projects/blog".to_string())
        );
    }
    #[test]
    fn unify_scopes_siblings_returns_common_ancestor() {
        // User mentions ~/projects/blog/posts/file.md and ~/projects/blog/output/index.html
        let scopes = vec![
            "/Users/david/projects/blog/posts".to_string(),
            "/Users/david/projects/blog/output".to_string(),
        ];
        assert_eq!(
            unify_current_turn_scopes(&scopes),
            Some("/Users/david/projects/blog".to_string())
        );
    }
    #[test]
    fn unify_scopes_different_projects_returns_first_when_ancestor_too_shallow() {
        // Two unrelated projects — common ancestor /tmp is too shallow (only 1 slash)
        let scopes = vec!["/tmp/project-a".to_string(), "/tmp/project-b".to_string()];
        // Common ancestor /tmp has depth 1, min scope depth is 1,
        // ancestor_depth (1) >= min_depth-1 (0) → allowed.
        // This is OK because for such shallow paths, multi_project_scope
        // should handle the case.
        assert_eq!(unify_current_turn_scopes(&scopes), Some("/tmp".to_string()));
    }
    #[test]
    fn legitimate_multiple_projects_are_not_arbitrarily_reduced_to_first() {
        let scopes = vec![
            "/Users/david/projects/blog".to_string(),
            "/Users/david/projects/daemon".to_string(),
        ];
        assert_eq!(
            unify_current_turn_scopes(&scopes),
            Some("/Users/david/projects".to_string())
        );
    }
    #[test]
    fn unify_scopes_three_paths_same_project() {
        let scopes = vec![
            "/Users/david/projects/blog/posts".to_string(),
            "/Users/david/projects/blog/output".to_string(),
            "/Users/david/projects/blog/scripts".to_string(),
        ];
        assert_eq!(
            unify_current_turn_scopes(&scopes),
            Some("/Users/david/projects/blog".to_string())
        );
    }
    #[test]
    fn current_turn_scope_beats_history_scope_even_when_not_yet_on_disk() {
        // Reproduces the bug where a user asks to CREATE a new project
        // (~/projects/ai-news-hub) but the scope lock latches onto an
        // existing project from conversation history (modern-plants-site)
        // because choose_primary_project_scope prefers real project roots.
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects");
        let existing = alias_root.join("modern-plants-site");
        let new_project = alias_root.join("ai-news-hub");
        std::fs::create_dir_all(&existing).expect("create existing");
        std::fs::write(existing.join("package.json"), "{}").expect("package.json");
        // new_project intentionally NOT created — the user is asking to create it

        let current_turn_scopes = vec![new_project.to_string_lossy().to_string()];
        let combined_scopes = vec![
            new_project.to_string_lossy().to_string(),
            existing.to_string_lossy().to_string(),
        ];

        // Old behaviour: choose_primary_project_scope on the combined list
        // would pick the existing project because it's a real root.
        let old_pick = choose_primary_project_scope(&combined_scopes);
        assert_eq!(
            old_pick,
            Some(existing.to_string_lossy().to_string()),
            "sanity: combined list still prefers existing project root"
        );

        // Fixed behaviour: try current-turn scopes first, fall back to combined.
        let fixed_pick = choose_primary_project_scope(&current_turn_scopes)
            .or_else(|| choose_primary_project_scope(&combined_scopes));
        assert_eq!(
            fixed_pick,
            Some(new_project.to_string_lossy().to_string()),
            "current-turn scope should win even if the dir doesn't exist yet"
        );
    }
    #[test]
    fn explicit_scope_wins_over_inherited_scope_when_present() {
        let extracted = Some("/Users/davidloor/projects/terminal.aidaemon.ai".to_string());
        let resolved = resolve_primary_project_scope(
            extracted,
            Some("/Users/davidloor/Library/Logs/aidaemon"),
            false,
            true,
        );
        assert_eq!(
            resolved,
            Some("/Users/davidloor/projects/terminal.aidaemon.ai".to_string())
        );
    }
    #[test]
    fn inherited_project_scope_yields_to_explicit_scope_for_multi_project_requests() {
        let extracted = Some("/Users/davidloor/projects/terminal.aidaemon.ai".to_string());
        let resolved = resolve_primary_project_scope(
            extracted.clone(),
            Some("/Users/davidloor/Library/Logs/aidaemon"),
            true,
            true,
        );
        assert_eq!(resolved, extracted);
    }
    #[test]
    fn inherited_project_scope_is_ignored_when_turn_is_not_local_project_work() {
        let resolved = resolve_primary_project_scope(
            None,
            Some("/Users/davidloor/projects/fairfax-va-site"),
            false,
            false,
        );
        assert_eq!(resolved, None);
    }
    #[test]
    fn project_scope_extraction_ignores_assistant_paths_for_non_local_followup() {
        let history = vec![msg(
            "assistant",
            "I am currently locked to /Users/davidloor/projects/fairfax-va-site.",
        )];

        let scopes = extract_project_scopes_from_history(
            &history,
            "Give me all the info about the top 2.",
            4,
            true,
            &[],
        );
        assert!(
            scopes.is_empty(),
            "non-local followup should not inherit assistant path scope: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_resolves_projects_folder_alias_to_configured_root() {
        let history = vec![];
        let project_name = format!("alias-test-{}", uuid::Uuid::new_v4().simple());
        // Use an absolute path so the test doesn't depend on cwd containing a
        // "projects" directory (which exists locally but not on CI runners).
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        std::fs::create_dir_all(&alias_root).expect("create alias root");
        let project_path = alias_root.join(&project_name);
        let current = format!(
            "Initialize a Vite app at {}",
            project_path.to_string_lossy()
        );
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];
        let scopes =
            extract_project_scopes_from_history(&history, &current, 4, false, &alias_roots);
        let expected = alias_root.join(&project_name).to_string_lossy().to_string();
        assert!(
            scopes.iter().any(|scope| scope == &expected),
            "expected aliased projects scope '{}', got {:?}",
            expected,
            scopes
        );
    }
    #[test]
    fn project_scope_normalization_prefers_absolute_paths_over_aliases() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        std::fs::create_dir_all(&alias_root).expect("create alias root");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];
        let absolute = root.path().join("absolute-target");
        let normalized = normalize_project_scope_path_with_aliases(
            absolute.to_string_lossy().as_ref(),
            &alias_roots,
        )
        .expect("normalized absolute path");
        assert_eq!(normalized, absolute.to_string_lossy());
    }
    #[test]
    fn project_scope_normalization_promotes_existing_src_paths_to_repo_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("blog");
        let src = root.join("src");
        std::fs::create_dir_all(&src).expect("create src");
        std::fs::write(root.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        std::fs::write(src.join("posts.js"), "export default [];\n").expect("posts");

        let normalized = normalize_project_scope_path_with_aliases(
            src.join("posts.js").to_string_lossy().as_ref(),
            &[],
        )
        .expect("normalized");
        assert_eq!(normalized, root.to_string_lossy());
    }
    #[test]
    fn project_scope_extraction_resolves_exact_project_paths() {
        let history = vec![];
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &history,
            &format!("Deploy {}", project.display()),
            4,
            false,
            &alias_roots,
        );
        assert_eq!(scopes, vec![project.to_string_lossy().to_string()]);
    }
    #[test]
    fn project_scope_extraction_rejects_named_project_roots_without_local_cues() {
        let history = vec![];
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &history,
            "Tell me about blog.aidaemon.ai and its latest posts.",
            4,
            false,
            &alias_roots,
        );
        assert!(
            scopes.is_empty(),
            "descriptive external request should not infer project scope: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_rejects_plain_word_nickname_without_local_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &[],
            "Find recruiting studies in Fairfax, Virginia and summarize them.",
            4,
            false,
            &alias_roots,
        );

        assert!(
            scopes.is_empty(),
            "plain language should not infer a project scope: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_words_do_not_turn_plain_nicknames_into_hard_scope() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &[],
            "Check the Fairfax project for broken links.",
            4,
            false,
            &alias_roots,
        );

        assert!(
            scopes.is_empty(),
            "semantic project references must arrive through task assessment: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_rejects_natural_language_slash_phrases() {
        let scopes = extract_project_scopes_from_history(
            &[],
            "The old workflow used to commit, deploy, and reload/restart the daemon.",
            4,
            false,
            &[],
        );

        assert!(
            scopes.is_empty(),
            "natural-language slash phrase should not infer a project scope: {:?}",
            scopes
        );
    }
    #[test]
    fn project_scope_extraction_rejects_globbed_absolute_paths() {
        let root = tempfile::tempdir().expect("tempdir");
        let project = root.path().join("demo-project");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(
            project.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        )
        .expect("cargo");
        let text = format!(
            "Previous failure mentioned {}",
            project.join("reload/restart**").display()
        );

        let mut scopes = Vec::new();
        extract_project_scopes_from_text(&text, &mut scopes, 4, &[]);

        assert!(
            scopes.is_empty(),
            "globbed absolute path should not infer a project scope: {:?}",
            scopes
        );
    }
    #[test]
    fn scope_extraction_rejects_api_endpoint_paths() {
        // /api/notes is a REST API endpoint, not a filesystem path.
        // /tmp/notes_api/ is the actual project directory.
        let mut scopes = Vec::new();
        extract_project_scopes_from_text(
            "Build /api/notes endpoint. Create everything in /tmp/notes_api/",
            &mut scopes,
            5,
            &[],
        );
        // /api doesn't exist on disk, so /api/notes should be rejected
        assert!(
            !scopes.iter().any(|s| s.contains("/api/notes")),
            "API endpoint path should not be extracted as scope, got: {:?}",
            scopes
        );
        // /tmp exists, so /tmp/notes_api should be accepted
        assert!(
            scopes.iter().any(|s| s.contains("notes_api")),
            "Real filesystem path should be extracted, got: {:?}",
            scopes
        );
    }
    #[test]
    fn first_dir_component_exists_filters_correctly() {
        assert!(crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/tmp/test")
        ));
        assert!(!crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/api/notes")
        ));
    }
    #[test]
    fn hostname_urls_not_treated_as_filesystem_paths() {
        // URLs without protocol prefix should NOT be treated as file paths
        assert!(!token_looks_like_filesystem_path(
            "api.waqi.info/feed/miami/?token=demo"
        ));
        assert!(!token_looks_like_filesystem_path(
            "example.com/path/to/resource"
        ));
        assert!(!token_looks_like_filesystem_path("wttr.in/Miami?format=j1"));
        // But real relative paths with dots should still work
        assert!(token_looks_like_filesystem_path("./src/main.rs"));
        assert!(token_looks_like_filesystem_path("../parent/file.txt"));
        // Dot-prefixed directories (hidden dirs) should work
        assert!(token_looks_like_filesystem_path(".hidden/config"));
        // Absolute paths should work
        assert!(token_looks_like_filesystem_path("/tmp/weather.py"));
        assert!(token_looks_like_filesystem_path("/usr/local/bin"));
        // Simple relative paths without dots before slash should work
        assert!(token_looks_like_filesystem_path("src/main.rs"));
    }
    #[test]
    fn url_without_protocol_not_extracted_as_project_scope() {
        let text = "Fetch from api.waqi.info/feed/miami/?token=demo and save to /tmp/weather.py";
        let mut scopes = Vec::new();
        extract_project_scopes_from_text(text, &mut scopes, 5, &[]);
        assert!(
            !scopes
                .iter()
                .any(|s| s.contains("waqi") || s.contains("api.")),
            "URL hostname should not be extracted as scope, got: {:?}",
            scopes
        );
        assert!(
            scopes.iter().any(|s| s.contains("tmp")),
            "Real path /tmp should be extracted, got: {:?}",
            scopes
        );
    }
}
