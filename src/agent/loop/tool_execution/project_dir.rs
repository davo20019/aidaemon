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

pub(crate) fn extract_project_dir_hint(text: &str) -> Option<String> {
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
        if !looks_path {
            continue;
        }

        let Some(normalized) = normalize_project_dir(token) else {
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
    let mut normalized = crate::tools::fs_utils::validate_path(raw_path).ok()?;

    let trimmed = raw_path.trim_end_matches('/');
    let file_name_looks_like_file = std::path::Path::new(trimmed)
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| name.contains('.') && !name.starts_with('.') && !name.ends_with('.'));
    if file_name_looks_like_file || normalized.is_file() {
        if let Some(parent) = normalized.parent() {
            normalized = parent.to_path_buf();
        }
    }

    Some(normalized.to_string_lossy().to_string())
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
        "run_command" => Some("working_dir"),
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
    let key = project_dir_arg_key_for_tool(tool_name)?;
    let project_dir = known_project_dir?.trim();
    if project_dir.is_empty() {
        return None;
    }

    let mut parsed = serde_json::from_str::<Value>(args_json).ok()?;
    let obj = parsed.as_object_mut()?;
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

    let injected_dir = if tool_name == "run_command" {
        // Scaffolding flow: if the target project dir doesn't exist yet, use
        // the nearest existing parent as working_dir so creation commands can run.
        let resolved = crate::tools::fs_utils::validate_path(project_dir).ok();
        if let Some(path) = resolved {
            if !path.is_dir() {
                if let Some(parent) = path.parent() {
                    if parent.is_dir() {
                        parent.to_string_lossy().to_string()
                    } else {
                        project_dir.to_string()
                    }
                } else {
                    project_dir.to_string()
                }
            } else {
                project_dir.to_string()
            }
        } else {
            project_dir.to_string()
        }
    } else {
        project_dir.to_string()
    };

    obj.insert(key.to_string(), json!(injected_dir));
    let updated = serde_json::to_string(&parsed).ok()?;
    Some((updated, injected_dir))
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
    fn scope_allows_only_descendant_paths() {
        assert!(scope_allows_project_dir("/tmp/a", "/tmp/a/src"));
        assert!(!scope_allows_project_dir("/tmp/a", "/tmp/b/src"));
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
