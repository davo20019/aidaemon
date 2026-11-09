//! The file tools confined to a workspace, as a kernel agent runs them.

use aidaemon_core::Tool;
use aidaemon_tools::{EditFile, ProtectedPathGuard, ReadFile, Workspace, WriteFile};
use serde_json::json;

fn workspace() -> (tempfile::TempDir, Workspace) {
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::new(dir.path()).unwrap();
    (dir, workspace)
}

fn ctx() -> aidaemon_core::ToolContext {
    aidaemon_core::ToolContext::new("test:synthetic-user-1", "file_tools")
}

#[tokio::test]
async fn read_pages_by_line_range_and_tail() {
    let (dir, workspace) = workspace();
    std::fs::write(dir.path().join("a.txt"), "one\ntwo\nthree\n").unwrap();
    let read = ReadFile::new(workspace);

    let out = read
        .call(
            json!({"path": "a.txt", "start_line": 2, "end_line": 2}),
            &ctx(),
        )
        .await
        .unwrap();
    assert!(!out.is_error, "{}", out.content);
    assert!(out.content.contains("2 | two") && !out.content.contains("three"));
    assert_eq!(out.details["total_lines"], 3);

    let out = read
        .call(json!({"path": "a.txt", "tail_lines": 1}), &ctx())
        .await
        .unwrap();
    assert!(out.content.contains("3 | three") && !out.content.contains("two"));
}

#[tokio::test]
async fn file_tools_stay_inside_the_workspace() {
    let (_dir, workspace) = workspace();
    let read = ReadFile::new(workspace.clone());
    let out = read
        .call(json!({"path": "../../etc/passwd"}), &ctx())
        .await
        .unwrap();
    assert!(
        out.is_error && out.content.contains("outside the workspace"),
        "{}",
        out.content
    );

    let write = WriteFile::new(workspace);
    let out = write
        .call(
            json!({"path": "/etc/synthetic-escape", "content": "x"}),
            &ctx(),
        )
        .await
        .unwrap();
    assert!(out.is_error, "{}", out.content);
}

#[cfg(unix)]
#[tokio::test]
async fn a_symlink_cannot_lead_out_of_the_workspace() {
    let (dir, workspace) = workspace();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "synthetic secret").unwrap();
    std::os::unix::fs::symlink(outside.path(), dir.path().join("out")).unwrap();

    let read = ReadFile::new(workspace.clone());
    let out = read
        .call(json!({"path": "out/secret.txt"}), &ctx())
        .await
        .unwrap();
    assert!(
        out.is_error && out.content.contains("outside the workspace"),
        "{}",
        out.content
    );

    let write = WriteFile::new(workspace);
    let out = write
        .call(json!({"path": "out/new.txt", "content": "x"}), &ctx())
        .await
        .unwrap();
    assert!(out.is_error);
    assert!(!outside.path().join("new.txt").exists());
}

fn paths(path: &std::path::Path) -> Vec<String> {
    vec![path.display().to_string()]
}

#[tokio::test]
async fn the_guard_refuses_secrets_and_sensitive_writes() {
    let (dir, workspace) = workspace();
    std::fs::write(dir.path().join(".env"), "API_KEY=synthetic").unwrap();
    std::fs::write(dir.path().join(".env.example"), "API_KEY=changeme").unwrap();
    let guard = ProtectedPathGuard::for_workspace(workspace.clone());
    let root = workspace.root();

    let reason = guard
        .refusal("read_file", &paths(&root.join(".env")), &[])
        .await
        .expect("a secret is refused");
    assert!(!reason.contains("synthetic"), "{reason}");
    assert!(guard
        .refusal("read_file", &paths(&root.join(".env.example")), &[])
        .await
        .is_none());
    assert!(guard
        .refusal("read_file", &paths(&root.join("main.rs")), &[])
        .await
        .is_none());
    for tool in ["write_file", "edit_file"] {
        assert!(guard
            .refusal(tool, &[], &paths(&root.join(".env")))
            .await
            .is_some());
        assert!(guard
            .refusal(tool, &[], &paths(&root.join("notes.txt")))
            .await
            .is_none());
    }
    // A tool the guard was not told about is not judged.
    assert!(guard
        .refusal("other_tool", &paths(&root.join(".env")), &[])
        .await
        .is_none());
    drop(dir);
}

#[cfg(unix)]
#[tokio::test]
async fn the_guard_follows_symlinks_to_what_they_name() {
    let (dir, workspace) = workspace();
    let root = workspace.root().to_path_buf();
    std::fs::write(root.join(".env"), "API_KEY=synthetic").unwrap();
    std::fs::create_dir(root.join(".ssh")).unwrap();
    std::os::unix::fs::symlink(root.join(".env"), root.join("notes.txt")).unwrap();
    std::os::unix::fs::symlink(root.join(".ssh"), root.join("keys")).unwrap();
    let guard = ProtectedPathGuard::for_workspace(workspace);

    assert!(guard
        .refusal("read_file", &paths(&root.join("notes.txt")), &[])
        .await
        .is_some());
    // A file not created yet, under a link to a sensitive directory.
    assert!(guard
        .refusal("write_file", &[], &paths(&root.join("keys/new_key")))
        .await
        .is_some());
    drop(dir);
}

#[tokio::test]
async fn writes_report_no_backup_without_one() {
    let (dir, workspace) = workspace();
    std::fs::write(dir.path().join("a.txt"), "old").unwrap();
    let write = WriteFile::new(workspace);
    let out = write
        .call(json!({"path": "a.txt", "content": "new"}), &ctx())
        .await
        .unwrap();
    assert!(!out.is_error, "{}", out.content);
    assert!(!out.content.contains("backup"), "{}", out.content);
}

#[tokio::test]
async fn edit_changes_nothing_when_ambiguous_and_validates_first() {
    let (dir, workspace) = workspace();
    std::fs::write(dir.path().join("a.txt"), "x\nx\n").unwrap();
    let edit = EditFile::new(workspace);

    let out = edit
        .call(
            json!({"path": "a.txt", "old_text": "x", "new_text": "y"}),
            &ctx(),
        )
        .await
        .unwrap();
    assert!(
        out.is_error && out.content.contains("2 occurrences"),
        "{}",
        out.content
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "x\nx\n"
    );

    assert!(edit
        .validate(&json!({"path": "a", "old_text": "", "new_text": "y"}))
        .is_err());
    assert!(edit
        .validate(&json!({"path": "a", "old_text": "y", "new_text": "y"}))
        .is_err());
    assert!(edit
        .validate(&json!({"path": "a", "old_text": "y"}))
        .is_err());
}
