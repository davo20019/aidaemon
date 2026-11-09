//! The file tools confined to a workspace, as a kernel agent runs them.

use aidaemon_core::Tool;
use aidaemon_tools::{EditFile, ReadFile, Workspace, WriteFile};
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

#[tokio::test]
async fn secret_files_are_refused_and_writes_report_no_backup() {
    let (dir, workspace) = workspace();
    std::fs::write(dir.path().join(".env"), "API_KEY=synthetic").unwrap();
    std::fs::write(dir.path().join("a.txt"), "old").unwrap();

    let read = ReadFile::new(workspace.clone());
    let out = read.call(json!({"path": ".env"}), &ctx()).await.unwrap();
    assert!(
        out.is_error && !out.content.contains("synthetic"),
        "{}",
        out.content
    );

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
