use serde_json::{json, Value};

use aidaemon_events::traits::{ToolCallSemantics, ToolTargetHintKind};

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

#[cfg(test)]
pub fn extract_project_dir_hint(text: &str) -> Option<String> {
    extract_project_dir_hint_with_aliases(text, &[])
}

#[cfg(test)]
pub fn extract_project_dir_hint_with_aliases(
    text: &str,
    _alias_roots: &[String],
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
        let normalized = looks_path
            .then(|| {
                normalize_project_dir(&*aidaemon_tools::testing::active_execution_backend(), token)
            })
            .flatten();
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

fn normalize_project_dir(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    raw_path: &str,
) -> Option<String> {
    if backend.kind() != aidaemon_tools::BackendKind::Local {
        let mut path = aidaemon_tools::backend::normalize_path_lexically(backend, raw_path).ok()?;
        if aidaemon_tools::fs_utils::path_points_to_file(raw_path) {
            path = path.parent()?;
        }
        return Some(path.to_string());
    }
    aidaemon_tools::fs_utils::normalize_project_scope_path(backend, raw_path)
        .ok()
        .map(|path| path.to_string_lossy().to_string())
}

fn push_unique_project_dir(collected: &mut Vec<String>, candidate: String) {
    if !collected.iter().any(|existing| existing == &candidate) {
        collected.push(candidate);
    }
}

fn resolve_injected_working_dir(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    project_dir: &str,
) -> String {
    if backend.kind() != aidaemon_tools::BackendKind::Local {
        return project_dir.to_string();
    }
    let resolved = aidaemon_tools::fs_utils::validate_path(backend, project_dir).ok();
    if let Some(path) = resolved
        && !path.is_dir()
            && let Some(parent) = path.parent()
                && parent.is_dir() {
                    return parent.to_string_lossy().to_string();
                }
    project_dir.to_string()
}

fn workspace_policy_allows_injected_working_dir(
    candidate: &str,
    workspace_root: &str,
    allow_outside_workspace: bool,
) -> bool {
    allow_outside_workspace || std::path::Path::new(candidate).starts_with(workspace_root)
}

fn active_workspace_policy_allows_injected_working_dir(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    candidate: &str,
) -> bool {
    workspace_policy_allows_injected_working_dir(
        candidate,
        backend.workspace_root().as_str(),
        backend.allows_outside_workspace(),
    )
}

fn collect_project_dirs_from_value(
    backend: &dyn aidaemon_tools::ExecutionBackend,
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
            if let Some(dir) = normalize_project_dir(backend, trimmed) {
                push_unique_project_dir(collected, dir);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_project_dirs_from_value(backend, item, parent_key, collected);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                if k.starts_with('_') {
                    continue;
                }
                collect_project_dirs_from_value(backend, v, Some(k.as_str()), collected);
            }
        }
        _ => {}
    }
}

pub fn extract_project_dirs_from_tool_args(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    tool_name: &str,
    args_json: &str,
) -> Vec<String> {
    let parsed = match serde_json::from_str::<Value>(args_json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut dirs = Vec::new();
    if tool_name == "terminal" {
        if let Some(cwd) = parsed.get("cwd") {
            collect_project_dirs_from_value(backend, cwd, Some("cwd"), &mut dirs);
        }
        if let Some(working_dir) = parsed.get("working_dir") {
            collect_project_dirs_from_value(backend, working_dir, Some("working_dir"), &mut dirs);
        }
        return dirs;
    }
    if tool_name == "project_inspect" {
        if let Some(path) = parsed.get("path") {
            collect_project_dirs_from_value(backend, path, Some("path"), &mut dirs);
        }
        if let Some(paths) = parsed.get("paths") {
            collect_project_dirs_from_value(backend, paths, Some("paths"), &mut dirs);
        }
        return dirs;
    }

    collect_project_dirs_from_value(backend, &parsed, None, &mut dirs);
    dirs
}

/// Return filesystem locations whose first access should trigger nested
/// project-instruction discovery. Broad grep/glob searches are intentionally
/// excluded; high-intent reads, writes, project entry, and command working
/// directories mirror the just-in-time behavior of established coding agents.
pub fn project_instruction_targets_for_tool_call(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    tool_name: &str,
    args_json: &str,
    semantics: &ToolCallSemantics,
) -> Vec<String> {
    if !matches!(
        tool_name,
        "read_file"
            | "write_file"
            | "edit_file"
            | "project_inspect"
            | "git_info"
            | "git_commit"
            | "terminal"
            | "run_command"
    ) {
        return Vec::new();
    }

    let mut targets = Vec::new();
    for hint in &semantics.target_hints {
        if matches!(
            hint.kind,
            ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
        ) && !targets.iter().any(|existing| existing == &hint.value)
        {
            targets.push(hint.value.clone());
        }
    }
    for directory in extract_project_dirs_from_tool_args(backend, tool_name, args_json) {
        if !targets.iter().any(|existing| existing == &directory) {
            targets.push(directory);
        }
    }

    targets
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

pub fn project_dir_from_tool_args(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    tool_name: &str,
    args_json: &str,
) -> Option<String> {
    extract_project_dirs_from_tool_args(backend, tool_name, args_json)
        .into_iter()
        .next()
}

pub fn maybe_inject_project_dir_into_tool_args(
    backend: &dyn aidaemon_tools::ExecutionBackend,
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
        if ["working_dir", "cwd"].iter().any(|key| {
            obj.get(*key)
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.trim().is_empty())
        }) {
            return None;
        }
        let has_program = ["command", "script"].iter().any(|key| {
            obj.get(*key)
                .and_then(|value| value.as_str())
                .is_some_and(|value| !value.trim().is_empty())
        });
        if !has_program {
            return None;
        }
        let injected_dir = resolve_injected_working_dir(backend, project_dir);
        // A shell command may safely reference paths outside a confined
        // workspace while still running from the workspace itself. Do not turn
        // that valid command into a rejected one by auto-prepending `cd` to an
        // out-of-scope inferred path.
        if !active_workspace_policy_allows_injected_working_dir(backend, &injected_dir) {
            return None;
        }
        // Execution cwd is first-class authority. Never hide it by rewriting
        // the shell program with `cd`: authorization, receipts, relative-path
        // resolution, cleanup, and replay must all see the same prepared cwd.
        obj.insert("working_dir".to_string(), json!(injected_dir));
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
        resolve_injected_working_dir(backend, project_dir)
    } else {
        project_dir.to_string()
    };

    obj.insert(key.to_string(), json!(injected_dir));
    let updated = serde_json::to_string(&parsed).ok()?;
    Some((updated, injected_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maybe_inject_project_dir_into_tool_args(
        tool_name: &str,
        args_json: &str,
        known_project_dir: Option<&str>,
    ) -> Option<(String, String)> {
        super::maybe_inject_project_dir_into_tool_args(
            &*aidaemon_tools::testing::active_execution_backend(),
            tool_name,
            args_json,
            known_project_dir,
        )
    }

    fn project_dir_from_tool_args(tool_name: &str, args_json: &str) -> Option<String> {
        super::project_dir_from_tool_args(
            &*aidaemon_tools::testing::active_execution_backend(),
            tool_name,
            args_json,
        )
    }

    fn extract_project_dirs_from_tool_args(tool_name: &str, args_json: &str) -> Vec<String> {
        super::extract_project_dirs_from_tool_args(
            &*aidaemon_tools::testing::active_execution_backend(),
            tool_name,
            args_json,
        )
    }

    fn project_instruction_targets_for_tool_call(
        tool_name: &str,
        args_json: &str,
        semantics: &ToolCallSemantics,
    ) -> Vec<String> {
        super::project_instruction_targets_for_tool_call(
            &*aidaemon_tools::testing::active_execution_backend(),
            tool_name,
            args_json,
            semantics,
        )
    }

    fn normalize_project_dir(raw_path: &str) -> Option<String> {
        super::normalize_project_dir(
            &*aidaemon_tools::testing::active_execution_backend(),
            raw_path,
        )
    }
    use proptest::prelude::*;

    #[test]
    fn extracts_project_hint_from_user_text() {
        let text = "Look in ~/projects/test-project/ for html files";
        let hint = extract_project_dir_hint(text).expect("project hint");
        assert!(hint.contains("test-project"));
        assert!(hint.starts_with('/'));
    }

    #[test]
    fn does_not_extract_plain_word_nickname_without_local_scope_cues() {
        let dir = tempfile::tempdir().expect("tempdir");
        let alias_root = dir.path().join("projects");
        let project = alias_root.join("springfield-il-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"springfield\"\n")
            .expect("wrangler");

        let hint = extract_project_dir_hint_with_aliases(
            "Find recruiting studies in Springfield, Illinois and summarize them.",
            &[alias_root.to_string_lossy().to_string()],
        );

        assert!(
            hint.is_none(),
            "plain language should not infer a project hint: {:?}",
            hint
        );
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
    fn terminal_command_text_does_not_declare_project_scope() {
        let args = r#"{"command":"cd ~/projects/demo && npm run build"}"#;
        let dirs = extract_project_dirs_from_tool_args("terminal", args);
        assert!(dirs.is_empty());
    }

    #[test]
    fn jit_targets_include_high_intent_file_paths_but_not_broad_searches() {
        let read_semantics = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Path, "crates/widget/src/lib.rs");
        let targets = project_instruction_targets_for_tool_call(
            "read_file",
            r#"{"path":"crates/widget/src/lib.rs"}"#,
            &read_semantics,
        );
        assert!(targets
            .iter()
            .any(|target| target == "crates/widget/src/lib.rs"));
        assert!(project_instruction_targets_for_tool_call(
            "search_files",
            r#"{"path":"crates/widget","glob":"*.rs"}"#,
            &read_semantics,
        )
        .is_empty());
    }

    #[test]
    fn jit_targets_use_typed_terminal_cwd_not_command_text() {
        let args =
            r#"{"command":"sed -i '' crates/widget/src/lib.rs && cargo test","cwd":"/tmp/repo"}"#;
        let targets = project_instruction_targets_for_tool_call(
            "terminal",
            args,
            &ToolCallSemantics::mutation(),
        );

        assert!(targets.iter().any(|target| target == "/tmp/repo"));
        assert_eq!(targets, vec!["/tmp/repo".to_string()]);
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
        let parsed: Value = serde_json::from_str(&updated).expect("valid arguments");
        assert_eq!(parsed["working_dir"], project.to_string_lossy().as_ref());
        assert_eq!(parsed["command"], "pwd && ls dist");
    }

    #[test]
    fn confined_workspace_rejects_parent_as_injected_working_dir() {
        let workspace = "/workspace/projects/agent";
        assert!(workspace_policy_allows_injected_working_dir(
            "/workspace/projects/agent/src",
            workspace,
            false
        ));
        assert!(!workspace_policy_allows_injected_working_dir(
            "/workspace/projects",
            workspace,
            false
        ));
        assert!(!workspace_policy_allows_injected_working_dir(
            "/workspace/projects-other",
            workspace,
            false
        ));
        assert!(workspace_policy_allows_injected_working_dir(
            "/workspace/projects",
            workspace,
            true
        ));
    }

    #[test]
    fn shell_cd_does_not_override_typed_working_directory_injection() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let inferred = tmp.path().join("inferred");
        std::fs::create_dir_all(&inferred).expect("create inferred project");
        let args = r#"{"command":"cd /tmp/explicit && npm run build"}"#;
        let (updated, injected) = maybe_inject_project_dir_into_tool_args(
            "terminal",
            args,
            Some(inferred.to_string_lossy().as_ref()),
        )
        .expect("typed cwd injection");
        let parsed: Value = serde_json::from_str(&updated).expect("valid arguments");
        assert_eq!(parsed["working_dir"], injected);
        assert_eq!(parsed["command"], "cd /tmp/explicit && npm run build");
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
        assert!(aidaemon_tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/tmp/foo")
        ));
        assert!(aidaemon_tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/usr/bin")
        ));
        assert!(!aidaemon_tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/api/notes")
        ));
        assert!(!aidaemon_tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/v1/status")
        ));
        // Relative paths always pass
        assert!(aidaemon_tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("src/main.rs")
        ));
        // Root itself passes
        assert!(aidaemon_tools::fs_utils::first_dir_component_exists(
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
