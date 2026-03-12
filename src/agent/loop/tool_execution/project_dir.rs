use crate::agent::*;

const PATH_ARGUMENT_KEYS: &[&str] = &[
    "path",
    "paths",
    "file_path",
    "working_dir",
    "cwd",
    "directory",
    "project_path",
    "root",
    "src",
    "dst",
    "from",
    "to",
];

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn extract_project_dir_hint(text: &str) -> Option<String> {
    extract_project_dir_hint_with_aliases(text, &[])
}

pub(crate) fn extract_project_dir_hint_with_aliases(
    text: &str,
    alias_roots: &[String],
) -> Option<String> {
    let mut best: Option<(usize, String)> = None;

    for raw in text.split_whitespace() {
        let token = raw
            .trim_matches(|c: char| {
                c.is_ascii_whitespace()
                    || matches!(
                        c,
                        '`' | '\'' | '"' | ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}'
                    )
            })
            // Trim trailing sentence punctuation that follows file extensions.
            // Without this, "event_system.py." extracts as `/path/event_system.py.`
            // and the trailing dot breaks scope locking (makes it look like a
            // non-file path, so normalize_project_dir doesn't strip to parent dir).
            .trim_end_matches(['.', '!', '?'])
            .trim();
        if token.is_empty() || token.contains("://") {
            continue;
        }

        let bytes = token.as_bytes();
        let looks_windows_abs = bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/');
        let looks_path = token.starts_with('/')
            || token.starts_with("~/")
            || token.starts_with("./")
            || token.starts_with("../")
            || looks_windows_abs;
        let normalized = if looks_path {
            normalize_project_dir(token)
        } else {
            crate::agent::should_allow_contextual_project_nickname_scope(text, token)
                .then(|| {
                    crate::tools::fs_utils::resolve_named_project_root(token, alias_roots)
                        .or_else(|| {
                            crate::tools::fs_utils::resolve_contextual_project_nickname_in_explicit_roots(token, alias_roots)
                        })
                })
                .flatten()
                .map(|path| path.to_string_lossy().to_string())
        };
        let Some(normalized) = normalized else {
            continue;
        };
        let score = normalized.matches('/').count() + usize::from(token.starts_with('/'));
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((score, normalized));
        }
    }

    best.map(|(_, path)| path)
}

fn normalize_project_dir(raw_path: &str) -> Option<String> {
    crate::tools::fs_utils::normalize_project_scope_path(raw_path)
        .ok()
        .map(|path| path.to_string_lossy().to_string())
}

fn push_unique_project_dir(collected: &mut Vec<String>, candidate: String) {
    if !collected.iter().any(|existing| existing == &candidate) {
        collected.push(candidate);
    }
}

fn extract_terminal_cd_dirs(command: &str) -> Vec<String> {
    let mut dirs = Vec::new();
    for segment in command.split([';', '\n']) {
        let trimmed = segment.trim();
        let Some(rest) = trimmed.strip_prefix("cd ") else {
            continue;
        };

        let token = if let Some(stripped) = rest.strip_prefix('"') {
            stripped.split('"').next().unwrap_or("").trim()
        } else if let Some(stripped) = rest.strip_prefix('\'') {
            stripped.split('\'').next().unwrap_or("").trim()
        } else {
            rest.split_whitespace().next().unwrap_or("").trim()
        };
        if token.is_empty() || token == "-" {
            continue;
        }
        if let Some(dir) = normalize_project_dir(token) {
            push_unique_project_dir(&mut dirs, dir);
        }
    }
    dirs
}

fn quote_shell_token(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

fn resolve_injected_working_dir(project_dir: &str) -> String {
    let resolved = crate::tools::fs_utils::validate_path(project_dir).ok();
    if let Some(path) = resolved {
        if !path.is_dir() {
            if let Some(parent) = path.parent() {
                if parent.is_dir() {
                    return parent.to_string_lossy().to_string();
                }
            }
        }
    }
    project_dir.to_string()
}

fn collect_project_dirs_from_value(
    value: &Value,
    parent_key: Option<&str>,
    collected: &mut Vec<String>,
) {
    match value {
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return;
            }
            let key_is_path_like = parent_key
                .map(|k| PATH_ARGUMENT_KEYS.contains(&k))
                .unwrap_or(false);
            if !key_is_path_like {
                return;
            }
            if let Some(dir) = normalize_project_dir(trimmed) {
                push_unique_project_dir(collected, dir);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_project_dirs_from_value(item, parent_key, collected);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                if k.starts_with('_') {
                    continue;
                }
                collect_project_dirs_from_value(v, Some(k.as_str()), collected);
            }
        }
        _ => {}
    }
}

pub(super) fn extract_project_dirs_from_tool_args(tool_name: &str, args_json: &str) -> Vec<String> {
    let parsed = match serde_json::from_str::<Value>(args_json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut dirs = Vec::new();
    if tool_name == "terminal" {
        if let Some(cwd) = parsed.get("cwd") {
            collect_project_dirs_from_value(cwd, Some("cwd"), &mut dirs);
        }
        if let Some(working_dir) = parsed.get("working_dir") {
            collect_project_dirs_from_value(working_dir, Some("working_dir"), &mut dirs);
        }
        if let Some(command) = parsed.get("command").and_then(|v| v.as_str()) {
            for dir in extract_terminal_cd_dirs(command) {
                push_unique_project_dir(&mut dirs, dir);
            }
        }
        return dirs;
    }
    if tool_name == "project_inspect" {
        if let Some(path) = parsed.get("path") {
            collect_project_dirs_from_value(path, Some("path"), &mut dirs);
        }
        if let Some(paths) = parsed.get("paths") {
            collect_project_dirs_from_value(paths, Some("paths"), &mut dirs);
        }
        return dirs;
    }

    collect_project_dirs_from_value(&parsed, None, &mut dirs);
    dirs
}

fn project_dir_arg_key_for_tool(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "search_files" | "project_inspect" | "git_info" | "git_commit" | "check_environment" => {
            Some("path")
        }
        "run_command" | "cli_agent" => Some("working_dir"),
        _ => None,
    }
}

pub(super) fn project_dir_from_tool_args(tool_name: &str, args_json: &str) -> Option<String> {
    extract_project_dirs_from_tool_args(tool_name, args_json)
        .into_iter()
        .next()
}

pub(super) fn tool_call_includes_project_path(tool_name: &str, args_json: &str) -> bool {
    project_dir_from_tool_args(tool_name, args_json).is_some()
}

pub(super) fn maybe_inject_project_dir_into_tool_args(
    tool_name: &str,
    args_json: &str,
    known_project_dir: Option<&str>,
) -> Option<(String, String)> {
    let project_dir = known_project_dir?.trim();
    if project_dir.is_empty() {
        return None;
    }

    let mut parsed = serde_json::from_str::<Value>(args_json).ok()?;
    let obj = parsed.as_object_mut()?;
    if tool_name == "terminal" {
        if obj
            .get("cwd")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty())
        {
            return None;
        }
        let command = obj.get("command").and_then(|v| v.as_str())?.trim();
        if command.is_empty() || !extract_terminal_cd_dirs(command).is_empty() {
            return None;
        }
        let injected_dir = resolve_injected_working_dir(project_dir);
        obj.insert(
            "command".to_string(),
            json!(format!(
                "cd {} && {}",
                quote_shell_token(&injected_dir),
                command
            )),
        );
        let updated = serde_json::to_string(&parsed).ok()?;
        return Some((updated, injected_dir));
    }

    let key = project_dir_arg_key_for_tool(tool_name)?;
    if tool_name == "project_inspect"
        && obj
            .get("paths")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter()
                    .filter_map(|entry| entry.as_str())
                    .any(|entry| !entry.trim().is_empty())
            })
    {
        return None;
    }
    if obj
        .get(key)
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.trim().is_empty())
    {
        return None;
    }

    let injected_dir = if matches!(tool_name, "run_command" | "cli_agent") {
        // Scaffolding flow: if the target project dir doesn't exist yet, use
        // the nearest existing parent as working_dir so creation commands can run.
        resolve_injected_working_dir(project_dir)
    } else {
        project_dir.to_string()
    };

    obj.insert(key.to_string(), json!(injected_dir));
    let updated = serde_json::to_string(&parsed).ok()?;
    Some((updated, injected_dir))
}

/// Returns true when the candidate path is an existing directory that looks
/// like a project root (contains a recognized project marker such as
/// `package.json`, `Cargo.toml`, `wrangler.toml`, etc.).
///
/// Used to relax the project scope lock: when the bot intentionally navigates
/// to a *different* but legitimate project, the scope lock should not block it.
pub(super) fn is_recognized_project_root(candidate_path: &str) -> bool {
    let Ok(path) = crate::tools::fs_utils::validate_path(candidate_path) else {
        return false;
    };
    if !path.is_dir() {
        return false;
    }
    crate::tools::fs_utils::find_nearest_project_root(&path).is_some_and(|root| root == path)
}

pub(super) fn scope_allows_project_dir(scope_path: &str, candidate_path: &str) -> bool {
    let Some(scope) = crate::tools::fs_utils::validate_path(scope_path).ok() else {
        return true;
    };
    let Some(candidate) = crate::tools::fs_utils::validate_path(candidate_path).ok() else {
        return true;
    };
    candidate.starts_with(scope)
}

pub(super) fn is_file_recheck_tool(tool_name: &str) -> bool {
    matches!(tool_name, "search_files" | "project_inspect")
}

pub(super) fn extract_project_dir_from_project_inspect_output(output: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(path) = trimmed.strip_prefix("# Project: ") {
            return normalize_project_dir(path.trim());
        }
    }
    None
}

pub(super) fn project_inspect_reports_file_entries(output: &str) -> bool {
    let mut in_structure = false;
    let mut in_code_block = false;

    for line in output.lines() {
        let trimmed = line.trim();
        if !in_structure {
            if trimmed.eq_ignore_ascii_case("## Structure") {
                in_structure = true;
            }
            continue;
        }

        if trimmed == "```" {
            in_code_block = !in_code_block;
            continue;
        }
        if !in_code_block || trimmed.is_empty() {
            continue;
        }

        if !trimmed.ends_with('/') {
            return true;
        }
    }

    false
}

pub(super) fn extract_search_files_scanned_dir(output: &str) -> Option<String> {
    for line in output.lines() {
        let Some(idx) = line.find("files scanned in ") else {
            continue;
        };
        let rest = &line[idx + "files scanned in ".len()..];
        let raw_path = rest
            .trim()
            .trim_end_matches(')')
            .trim_end_matches(':')
            .trim();
        if raw_path.is_empty() {
            continue;
        }
        if let Some(normalized) = normalize_project_dir(raw_path) {
            return Some(normalized);
        }
    }
    None
}

pub(super) fn search_files_result_no_matches(output: &str) -> bool {
    output.to_ascii_lowercase().contains("no matches found")
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn extracts_project_hint_from_user_text() {
        let text = "Look in ~/projects/test-project/ for html files";
        let hint = extract_project_dir_hint(text).expect("project hint");
        assert!(hint.contains("test-project"));
        assert!(hint.starts_with('/'));
    }

    #[test]
    fn extracts_named_project_hint_with_alias_roots() {
        let dir = tempfile::tempdir().expect("tempdir");
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");

        let hint = extract_project_dir_hint_with_aliases(
            "Deploy blog.aidaemon.ai",
            &[alias_root.to_string_lossy().to_string()],
        )
        .expect("project hint");
        assert_eq!(hint, project.to_string_lossy());
    }

    #[test]
    fn does_not_extract_plain_word_nickname_without_local_scope_cues() {
        let dir = tempfile::tempdir().expect("tempdir");
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");

        let hint = extract_project_dir_hint_with_aliases(
            "Find recruiting studies in Fairfax, Virginia and summarize them.",
            &[alias_root.to_string_lossy().to_string()],
        );

        assert!(
            hint.is_none(),
            "plain language should not infer a project hint: {:?}",
            hint
        );
    }

    #[test]
    fn does_not_extract_dotted_nickname_without_local_scope_cues() {
        let dir = tempfile::tempdir().expect("tempdir");
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");

        let hint = extract_project_dir_hint_with_aliases(
            "Tell me about blog.aidaemon.ai and its latest posts.",
            &[alias_root.to_string_lossy().to_string()],
        );

        assert!(
            hint.is_none(),
            "descriptive external request should not infer a project hint: {:?}",
            hint
        );
    }

    #[test]
    fn extracts_plain_word_nickname_with_explicit_project_scope_cues() {
        let dir = tempfile::tempdir().expect("tempdir");
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");

        let hint = extract_project_dir_hint_with_aliases(
            "Check the Fairfax project for broken links.",
            &[alias_root.to_string_lossy().to_string()],
        )
        .expect("project hint");

        assert_eq!(hint, project.to_string_lossy());
    }

    #[test]
    fn injects_project_dir_when_path_missing() {
        let args = r#"{"glob":"*.html"}"#;
        let (updated, injected) =
            maybe_inject_project_dir_into_tool_args("search_files", args, Some("/tmp/myproj"))
                .expect("injection");
        assert_eq!(injected, "/tmp/myproj");
        assert!(updated.contains(r#""path":"/tmp/myproj""#));
    }

    #[test]
    fn does_not_override_existing_project_path() {
        let args = r#"{"glob":"*.html","path":"/tmp/explicit"}"#;
        let updated =
            maybe_inject_project_dir_into_tool_args("search_files", args, Some("/tmp/inferred"));
        assert!(updated.is_none());
    }

    #[test]
    fn parses_project_inspect_and_search_paths() {
        let inspect = "# Project: /tmp/dogs\n\n## Structure\n```\nindex.html\nstyles.css\n```\n";
        let search = "No matches found (0 files scanned in /tmp/dogs)";
        assert_eq!(
            extract_project_dir_from_project_inspect_output(inspect).as_deref(),
            Some("/tmp/dogs")
        );
        assert!(project_inspect_reports_file_entries(inspect));
        assert_eq!(
            extract_search_files_scanned_dir(search).as_deref(),
            Some("/tmp/dogs")
        );
        assert!(search_files_result_no_matches(search));
    }

    #[test]
    fn parses_project_inspect_paths_array() {
        let args = r#"{"paths":["/tmp/dogs","/tmp/cats"]}"#;
        assert_eq!(
            project_dir_from_tool_args("project_inspect", args).as_deref(),
            Some("/tmp/dogs")
        );
    }

    #[test]
    fn does_not_inject_project_dir_when_project_inspect_paths_exist() {
        let args = r#"{"paths":["/tmp/explicit"]}"#;
        let updated =
            maybe_inject_project_dir_into_tool_args("project_inspect", args, Some("/tmp/inferred"));
        assert!(updated.is_none());
    }

    #[test]
    fn preserves_dot_prefixed_directory_hint() {
        let text = "Search under /tmp/.myproject for html files";
        let hint = extract_project_dir_hint(text).expect("project hint");
        assert_eq!(hint, "/tmp/.myproject");
    }

    #[test]
    fn extracts_project_dirs_from_nested_path_arguments() {
        let args =
            r#"{"path":"./workspace/app","changes":[{"file_path":"./workspace/app/src/main.rs"}]}"#;
        let dirs = extract_project_dirs_from_tool_args("edit_file", args);
        assert!(!dirs.is_empty());
        assert!(dirs.iter().any(|d| d.ends_with("/workspace/app")));
    }

    #[test]
    fn extracts_terminal_cd_dirs_for_scope_locking() {
        let args = r#"{"command":"cd ~/projects/demo && npm run build"}"#;
        let dirs = extract_project_dirs_from_tool_args("terminal", args);
        assert!(dirs.iter().any(|d| d.contains("projects/demo")));
    }

    #[test]
    fn injects_terminal_project_dir_when_command_has_no_explicit_cd() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let project = tmp.path().join("myproj");
        std::fs::create_dir_all(&project).expect("create project");
        let args = r#"{"command":"pwd && ls dist"}"#;
        let (updated, injected) = maybe_inject_project_dir_into_tool_args(
            "terminal",
            args,
            Some(project.to_string_lossy().as_ref()),
        )
        .expect("injection");
        assert_eq!(injected, project.to_string_lossy());
        assert!(updated.contains(&format!(
            "cd '{}' && pwd && ls dist",
            project.to_string_lossy()
        )));
    }

    #[test]
    fn does_not_override_terminal_command_with_explicit_cd() {
        let args = r#"{"command":"cd /tmp/explicit && npm run build"}"#;
        let updated =
            maybe_inject_project_dir_into_tool_args("terminal", args, Some("/tmp/inferred"));
        assert!(updated.is_none());
    }

    #[test]
    fn scope_allows_only_descendant_paths() {
        assert!(scope_allows_project_dir("/tmp/a", "/tmp/a/src"));
        assert!(!scope_allows_project_dir("/tmp/a", "/tmp/b/src"));
    }

    #[test]
    fn strips_trailing_sentence_punctuation_from_path_hint() {
        // "event_system.py." — period at end of sentence
        let text = "Fix the bugs in /tmp/debugme/event_system.py. The tests are correct.";
        let hint = extract_project_dir_hint(text).expect("project hint");
        assert_eq!(hint, "/tmp/debugme");
        assert!(!hint.contains("event_system"));

        // Trailing exclamation mark
        let text2 = "Look at /tmp/myproject/main.rs! It has bugs";
        let hint2 = extract_project_dir_hint(text2).expect("project hint");
        assert_eq!(hint2, "/tmp/myproject");

        // No trailing punctuation — should still work
        let text3 = "Fix /tmp/debugme/event_system.py please";
        let hint3 = extract_project_dir_hint(text3).expect("project hint");
        assert_eq!(hint3, "/tmp/debugme");
    }

    #[test]
    fn run_command_injection_falls_back_to_existing_parent_when_target_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let parent = tmp.path().join("projects");
        std::fs::create_dir_all(&parent).expect("create parent");
        let target = parent.join("new-site");
        let args = r#"{"command":"pwd"}"#;

        let (updated, injected) = maybe_inject_project_dir_into_tool_args(
            "run_command",
            args,
            Some(target.to_string_lossy().as_ref()),
        )
        .expect("injection");

        assert_eq!(injected, parent.to_string_lossy());
        assert!(updated.contains(r#""working_dir":"#));
    }

    #[test]
    fn injects_cli_agent_working_dir_when_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let project = tmp.path().join("aidaemon");
        std::fs::create_dir_all(&project).expect("create project");
        let args = r#"{"action":"run","prompt":"inspect the logs"}"#;

        let (updated, injected) = maybe_inject_project_dir_into_tool_args(
            "cli_agent",
            args,
            Some(project.to_string_lossy().as_ref()),
        )
        .expect("injection");

        assert_eq!(injected, project.to_string_lossy());
        assert!(updated.contains(r#""working_dir":"#));
    }

    #[test]
    fn rejects_api_endpoint_paths_as_project_dirs() {
        // /api/notes is a REST API endpoint, not a filesystem path
        // /tmp/notes_api/ is the actual project directory
        let text = "Build /api/notes endpoint. Create everything in /tmp/notes_api/";
        let hint = extract_project_dir_hint(text).expect("project hint");
        assert_eq!(hint, "/tmp/notes_api");
    }

    #[test]
    fn first_dir_component_exists_for_real_paths() {
        assert!(crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/tmp/foo")
        ));
        assert!(crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/usr/bin")
        ));
        assert!(!crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/api/notes")
        ));
        assert!(!crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/v1/status")
        ));
        // Relative paths always pass
        assert!(crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("src/main.rs")
        ));
        // Root itself passes
        assert!(crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/")
        ));
    }

    #[test]
    fn normalizes_existing_src_file_scope_to_repo_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("blog");
        let src = root.join("src");
        std::fs::create_dir_all(&src).expect("create src");
        std::fs::write(root.join("package.json"), r#"{"name":"blog"}"#).expect("package");
        std::fs::write(src.join("posts.js"), "export default [];\n").expect("posts");

        let hint = normalize_project_dir(src.join("posts.js").to_string_lossy().as_ref())
            .expect("project hint");
        assert_eq!(hint, root.to_string_lossy());
    }

    #[test]
    fn extracts_tool_arg_scope_at_repo_root_when_marker_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("blog");
        let src = root.join("src");
        std::fs::create_dir_all(&src).expect("create src");
        std::fs::write(root.join("package.json"), r#"{"name":"blog"}"#).expect("package");
        std::fs::write(src.join("posts.js"), "export default [];\n").expect("posts");

        let args = format!(
            r#"{{"file_path":"{}"}}"#,
            src.join("posts.js").to_string_lossy()
        );
        let extracted = project_dir_from_tool_args("edit_file", &args).expect("project dir");
        assert_eq!(extracted, root.to_string_lossy());
    }

    #[test]
    fn is_recognized_project_root_allows_real_project() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project_a = dir.path().join("project-a");
        let project_b = dir.path().join("project-b");
        std::fs::create_dir_all(&project_a).expect("create project-a");
        std::fs::create_dir_all(&project_b).expect("create project-b");
        // project-a has Cargo.toml
        std::fs::write(project_a.join("Cargo.toml"), "[package]\nname = \"a\"\n").expect("write");
        // project-b has package.json
        std::fs::write(project_b.join("package.json"), r#"{"name":"b"}"#).expect("write");

        assert!(is_recognized_project_root(
            project_a.to_string_lossy().as_ref()
        ));
        assert!(is_recognized_project_root(
            project_b.to_string_lossy().as_ref()
        ));

        // A random dir without project markers is NOT recognized
        let random_dir = dir.path().join("random");
        std::fs::create_dir_all(&random_dir).expect("create random");
        assert!(!is_recognized_project_root(
            random_dir.to_string_lossy().as_ref()
        ));

        // A non-existent dir is NOT recognized
        let nonexistent = dir.path().join("nonexistent");
        assert!(!is_recognized_project_root(
            nonexistent.to_string_lossy().as_ref()
        ));
    }

    #[test]
    fn scope_violation_allows_switch_to_recognized_project_root() {
        // Simulate: primary scope is project-a, bot tries to cd to project-b
        let dir = tempfile::tempdir().expect("tempdir");
        let project_a = dir.path().join("project-a");
        let project_b = dir.path().join("project-b");
        std::fs::create_dir_all(&project_a).expect("create project-a");
        std::fs::create_dir_all(&project_b).expect("create project-b");
        std::fs::write(project_a.join("Cargo.toml"), "[package]\nname = \"a\"\n").expect("write");
        std::fs::write(project_b.join("package.json"), r#"{"name":"b"}"#).expect("write");

        // project-b is outside project-a's scope
        assert!(!scope_allows_project_dir(
            project_a.to_string_lossy().as_ref(),
            project_b.to_string_lossy().as_ref()
        ));

        // but project-b IS a recognized project root, so scope violation should not fire
        assert!(is_recognized_project_root(
            project_b.to_string_lossy().as_ref()
        ));
    }

    #[test]
    fn scope_violation_still_blocks_non_project_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let project_a = dir.path().join("project-a");
        let random = dir.path().join("random-dir");
        std::fs::create_dir_all(&project_a).expect("create project-a");
        std::fs::create_dir_all(&random).expect("create random");
        std::fs::write(project_a.join("Cargo.toml"), "[package]\nname = \"a\"\n").expect("write");
        // random has no project markers

        assert!(!scope_allows_project_dir(
            project_a.to_string_lossy().as_ref(),
            random.to_string_lossy().as_ref()
        ));
        assert!(!is_recognized_project_root(
            random.to_string_lossy().as_ref()
        ));
    }

    proptest! {
        #[test]
        fn extracted_project_dirs_are_deduped_for_duplicate_inputs(name in "[a-z0-9_-]{3,12}") {
            let path = format!("/tmp/{name}");
            let args = json!({
                "paths": [path.clone(), path.clone()],
                "path": path.clone()
            })
            .to_string();
            let dirs = extract_project_dirs_from_tool_args("project_inspect", &args);
            let unique: std::collections::HashSet<String> = dirs.iter().cloned().collect();
            prop_assert_eq!(dirs.len(), unique.len());
        }
    }
}
