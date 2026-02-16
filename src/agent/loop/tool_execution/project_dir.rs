use crate::agent::*;

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
    let parsed = serde_json::from_str::<Value>(args_json).ok()?;
    if tool_name == "project_inspect" {
        if let Some(raw_path) = parsed.get("path").and_then(|v| v.as_str()) {
            let trimmed = raw_path.trim();
            if !trimmed.is_empty() {
                return normalize_project_dir(trimmed);
            }
        }
        if let Some(raw_path) = parsed
            .get("paths")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .filter_map(|entry| entry.as_str())
                    .map(str::trim)
                    .find(|entry| !entry.is_empty())
            })
        {
            return normalize_project_dir(raw_path);
        }
        return None;
    }

    let key = project_dir_arg_key_for_tool(tool_name)?;
    let raw = parsed.get(key)?.as_str()?.trim();
    if raw.is_empty() {
        return None;
    }
    normalize_project_dir(raw)
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

    obj.insert(key.to_string(), json!(project_dir));
    let updated = serde_json::to_string(&parsed).ok()?;
    Some((updated, project_dir.to_string()))
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
}
