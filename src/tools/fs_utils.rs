//! Path and project helpers for the file tools (defined in `aidaemon-tools`).

pub use aidaemon_tools::fs_utils::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::execution::{BackendKind, BackendPath, ExecutionBackend};
    use std::io::Write;
    #[allow(unused_imports)]
    use std::path::{Path, PathBuf};
    #[allow(unused_imports)]
    use std::time::Duration;

    #[test]
    fn test_validate_path_home() {
        let result = validate_path(&*crate::execution::active_execution_backend(), "~/test.txt");
        assert!(result.is_ok());
        let p = result.unwrap();
        assert!(p.is_absolute());
        assert!(!p.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_validate_path_traversal() {
        let result = validate_path(
            &*crate::execution::active_execution_backend(),
            "/tmp/../../etc/passwd",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_absolute() {
        let result = validate_path(
            &*crate::execution::active_execution_backend(),
            "/tmp/test.txt",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_dot_normalized() {
        // Passing "." should resolve to cwd without trailing "/."
        let result = validate_path(&*crate::execution::active_execution_backend(), ".").unwrap();
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
        let result =
            validate_path(&*crate::execution::active_execution_backend(), "src/.").unwrap();
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

        let normalized = normalize_project_scope_path(
            &*crate::execution::active_execution_backend(),
            src.to_string_lossy().as_ref(),
        )
        .expect("normalized");
        assert_eq!(normalized, root);
    }

    #[test]
    fn test_normalize_project_scope_path_keeps_non_project_target_dir() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("new-site");
        let normalized = normalize_project_scope_path(
            &*crate::execution::active_execution_backend(),
            target.to_string_lossy().as_ref(),
        )
        .expect("normalized");
        assert_eq!(normalized, target);
    }

    #[test]
    fn test_normalize_project_scope_path_preserves_existing_dotted_directory() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("blog.aidaemon.ai");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("wrangler.toml"), "name = \"blog\"\n").unwrap();

        let normalized = normalize_project_scope_path(
            &*crate::execution::active_execution_backend(),
            target.to_string_lossy().as_ref(),
        )
        .expect("normalized");
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
            &*crate::execution::active_execution_backend(),
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
        let project = alias_root.join("springfield-il-site");
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(project.join("wrangler.toml"), "name = \"springfield\"\n").unwrap();

        let resolved = resolve_contextual_project_nickname_in_explicit_roots(
            &*crate::execution::active_execution_backend(),
            "springfield",
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

        let named = resolve_project_scope_reference(
            &*crate::execution::active_execution_backend(),
            "blog.aidaemon.ai",
            &alias_roots,
        )
        .expect("named project");
        assert_eq!(named, project);

        let aliased = resolve_project_scope_reference(
            &*crate::execution::active_execution_backend(),
            "projects/blog.aidaemon.ai",
            &alias_roots,
        )
        .expect("aliased project");
        assert_eq!(aliased, project);
    }

    #[test]
    fn structural_filesystem_references_require_path_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let alias_root = dir.path().join("projects");
        let source = alias_root.join("synthetic-project").join("src");
        std::fs::create_dir_all(&source).unwrap();
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        for prose in [
            "Pros/cons",
            "input/output",
            "client/server",
            "read/write",
            "pass/fail",
            "alpha/beta/gamma",
        ] {
            assert_eq!(
                resolve_structural_filesystem_reference(
                    &*crate::execution::active_execution_backend(),
                    prose,
                    &alias_roots
                ),
                None,
                "unanchored prose compound must not become a path: {prose}"
            );
        }

        assert_eq!(
            resolve_structural_filesystem_reference(
                &*crate::execution::active_execution_backend(),
                "synthetic-project/src/new_module.rs",
                &alias_roots
            ),
            Some(source.join("new_module.rs"))
        );

        let explicit_new_target = dir.path().join("not-created-yet").join("result.md");
        assert_eq!(
            resolve_structural_filesystem_reference(
                &*crate::execution::active_execution_backend(),
                explicit_new_target.to_string_lossy().as_ref(),
                &alias_roots
            ),
            Some(explicit_new_target)
        );
    }

    #[test]
    fn protected_host_data_is_separate_from_workspace_and_temporary_data() {
        #[cfg(unix)]
        {
            assert!(is_protected_host_data_path(Path::new("/etc/hosts")));
            assert!(is_protected_host_data_path(Path::new(
                "/private/var/root/result.txt"
            )));
            assert!(!is_protected_host_data_path(Path::new(
                "/tmp/synthetic-result.txt"
            )));
            assert!(!is_protected_host_data_path(Path::new(
                "/workspace/synthetic-project/src/main.rs"
            )));
        }
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
        let result = aidaemon_tools::format_with_line_numbers("foo\nbar\nbaz", 0);
        assert!(result.contains("  1 | foo"));
        assert!(result.contains("  2 | bar"));
        assert!(result.contains("  3 | baz"));
    }

    #[test]
    fn test_format_with_line_numbers_offset() {
        let result = aidaemon_tools::format_with_line_numbers("line10\nline11", 9);
        assert!(result.contains("10 | line10"));
        assert!(result.contains("11 | line11"));
    }

    #[tokio::test]
    async fn test_run_cmd_echo() {
        let out = run_cmd(
            &*crate::execution::active_execution_backend(),
            "echo hello",
            None,
            5,
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[tokio::test]
    async fn test_run_cmd_timeout() {
        let result = run_cmd(
            &*crate::execution::active_execution_backend(),
            "sleep 10",
            None,
            1,
        )
        .await;
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

        let result = run_cmd(
            &*crate::execution::active_execution_backend(),
            &cmd,
            None,
            1,
        )
        .await;
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
        let out = run_cmd(
            &*crate::execution::active_execution_backend(),
            "pwd",
            Some(dir.path()),
            5,
        )
        .await
        .unwrap();
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

    /// A bound workspace anchors bare relative references even when the daemon
    /// cwd (this repository, which also has `src/`) could claim them.
    #[test]
    fn anchored_relative_reference_resolves_inside_workspace_not_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("blog-site");
        std::fs::create_dir_all(workspace.join("src").join("content").join("posts")).unwrap();
        let cwd = std::env::current_dir().unwrap();
        assert!(
            cwd.join("src").is_dir(),
            "test precondition: the daemon cwd has a competing src/ directory"
        );

        let resolved = resolve_structural_filesystem_reference_from(
            &*crate::execution::active_execution_backend(),
            "src/content/posts",
            &[],
            Some(&workspace),
        )
        .expect("workspace-anchored reference");
        assert_eq!(resolved, workspace.join("src/content/posts"));
        assert!(!resolved.starts_with(&cwd));

        // Without a workspace the run was given, the daemon's own cwd is not
        // an anchor: the reference names nothing.
        let unanchored = resolve_structural_filesystem_reference(
            &*crate::execution::active_execution_backend(),
            "src/content/posts",
            &[],
        );
        assert_eq!(unanchored, None);
    }

    /// Under an anchor, a reference the workspace cannot ground must not be
    /// relocated to another configured project root; that would move the task's
    /// authority to a repository the goal never named.
    #[test]
    fn anchored_reference_never_relocates_to_alias_roots() {
        let dir = tempfile::tempdir().unwrap();
        let alias_root = dir.path().join("projects");
        std::fs::create_dir_all(alias_root.join("other-project").join("src")).unwrap();
        let workspace = dir.path().join("workspace-without-src");
        std::fs::create_dir_all(&workspace).unwrap();
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        assert_eq!(
            resolve_structural_filesystem_reference_from(
                &*crate::execution::active_execution_backend(),
                "src/content/posts",
                &alias_roots,
                Some(&workspace)
            ),
            None
        );

        // Explicit forms still resolve inside the anchor, and a not-yet-created
        // target is allowed there.
        assert_eq!(
            resolve_structural_filesystem_reference_from(
                &*crate::execution::active_execution_backend(),
                "./src/content/posts",
                &alias_roots,
                Some(&workspace)
            ),
            Some(workspace.join("src/content/posts"))
        );
        assert_eq!(
            resolve_structural_filesystem_reference_from(
                &*crate::execution::active_execution_backend(),
                "../escape",
                &alias_roots,
                Some(&workspace)
            ),
            Some(dir.path().join("escape"))
        );
    }

    #[test]
    fn anchored_project_scope_reference_promotes_to_workspace_root() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("blog-site");
        std::fs::create_dir_all(workspace.join(".git")).unwrap();
        std::fs::create_dir_all(workspace.join("src").join("content").join("posts")).unwrap();

        let scope = resolve_project_scope_reference_from(
            &*crate::execution::active_execution_backend(),
            "src/content/posts",
            &[],
            Some(&workspace),
        )
        .expect("anchored project scope");
        assert_eq!(scope, workspace);

        // Negative control: without an anchor the process cwd is the base. Only
        // the base is asserted — whether the reference is promoted to a project
        // root depends on what exists under the cwd, which varies by checkout.
        let cwd = std::env::current_dir().unwrap();
        let unanchored = resolve_project_scope_reference(
            &*crate::execution::active_execution_backend(),
            "src/content/posts",
            &[],
        )
        .expect("unanchored project scope resolves against the process cwd");
        assert!(
            unanchored.starts_with(&cwd),
            "{unanchored:?} should be rooted in {cwd:?}"
        );
        assert!(!unanchored.starts_with(&workspace));
    }
}
