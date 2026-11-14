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
    assert!(
        guard
            .refusal("read_file", &paths(&root.join(".env.example")), &[])
            .await
            .is_none()
    );
    assert!(
        guard
            .refusal("read_file", &paths(&root.join("main.rs")), &[])
            .await
            .is_none()
    );
    for tool in ["write_file", "edit_file"] {
        assert!(
            guard
                .refusal(tool, &[], &paths(&root.join(".env")))
                .await
                .is_some()
        );
        assert!(
            guard
                .refusal(tool, &[], &paths(&root.join("notes.txt")))
                .await
                .is_none()
        );
    }
    // A tool the guard was not told about is not judged.
    assert!(
        guard
            .refusal("other_tool", &paths(&root.join(".env")), &[])
            .await
            .is_none()
    );
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

    assert!(
        guard
            .refusal("read_file", &paths(&root.join("notes.txt")), &[])
            .await
            .is_some()
    );
    // A file not created yet, under a link to a sensitive directory.
    assert!(
        guard
            .refusal("write_file", &[], &paths(&root.join("keys/new_key")))
            .await
            .is_some()
    );
    drop(dir);
}

#[tokio::test]
async fn sensitive_reads_cover_more_than_secrets() {
    let (dir, workspace) = workspace();
    let root = workspace.root().to_path_buf();
    std::fs::create_dir(root.join(".git")).unwrap();
    let guard = ProtectedPathGuard::for_workspace(workspace).tool(
        "publisher",
        aidaemon_tools::semantics::FileGuard::SensitiveReads,
    );
    let config = paths(&root.join(".git/config"));
    assert!(guard.refusal("read_file", &config, &[]).await.is_none());
    assert!(guard.refusal("publisher", &config, &[]).await.is_some());
    assert!(
        guard
            .refusal("publisher", &paths(&root.join("page.html")), &[])
            .await
            .is_none()
    );
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

    assert!(
        edit.validate(&json!({"path": "a", "old_text": "", "new_text": "y"}))
            .is_err()
    );
    assert!(
        edit.validate(&json!({"path": "a", "old_text": "y", "new_text": "y"}))
            .is_err()
    );
    assert!(
        edit.validate(&json!({"path": "a", "old_text": "y"}))
            .is_err()
    );
}

/// What a write tool predicts for a call is what the call then writes, and
/// a call that cannot apply is predicted as such, with the tool's reason.
#[tokio::test]
async fn a_predicted_write_is_the_one_the_call_makes() {
    use aidaemon_tools::semantics::ToolSemantics;
    let (dir, workspace) = workspace();
    let file = dir.path().join("a.txt");
    std::fs::write(&file, "one\r\ntwo\r\none\r\n").unwrap();
    let edit = EditFile::new(workspace.clone());
    let write = WriteFile::new(workspace);

    let calls = [
        json!({"path": "a.txt", "old_text": "two\n", "new_text": "TWO\n"}),
        json!({"path": "a.txt", "old_text": "one", "new_text": "ONE", "replace_all": true}),
    ];
    for args in calls {
        let current = std::fs::read(&file).unwrap();
        let predicted = edit
            .file_write_result(&args.to_string(), Some(&current))
            .unwrap()
            .unwrap();
        let out = edit.call(args, &ctx()).await.unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert_eq!(std::fs::read(&file).unwrap(), predicted);
    }
    for args in [
        json!({"path": "a.txt", "old_text": "absent", "new_text": "x"}),
        json!({"path": "a.txt", "old_text": "ONE", "new_text": "x"}),
    ] {
        let current = std::fs::read(&file).unwrap();
        let predicted = edit.file_write_result(&args.to_string(), Some(&current));
        assert!(matches!(predicted, Some(Err(_))), "{args}");
        assert!(edit.call(args, &ctx()).await.unwrap().is_error);
    }
    assert!(matches!(
        edit.file_write_result(
            &json!({"path": "gone.txt", "old_text": "a", "new_text": "b"}).to_string(),
            None
        ),
        Some(Err(_))
    ));

    for args in [
        json!({"path": "a.txt", "content": "more\n", "mode": "append"}),
        json!({"path": "a.txt", "content": "all new\n"}),
    ] {
        let current = std::fs::read(&file).unwrap();
        let predicted = write
            .file_write_result(&args.to_string(), Some(&current))
            .unwrap()
            .unwrap();
        let out = write.call(args, &ctx()).await.unwrap();
        assert!(!out.is_error, "{}", out.content);
        assert_eq!(std::fs::read(&file).unwrap(), predicted);
    }
}

/// A batch of edits applies in order, each to the result of the one before,
/// in one write; `replace_all` works per edit.
#[tokio::test]
async fn edit_batch_applies_in_order() {
    use aidaemon_tools::semantics::ToolSemantics;
    let (dir, workspace) = workspace();
    let file = dir.path().join("a.txt");
    std::fs::write(&file, "alpha\nbeta\ngamma gamma\n").unwrap();
    let edit = EditFile::new(workspace);

    let args = json!({"path": "a.txt", "edits": [
        {"old_text": "alpha", "new_text": "ALPHA"},
        {"old_text": "gamma", "new_text": "delta", "replace_all": true},
        // Only exists once the first edit has applied.
        {"old_text": "ALPHA\nbeta", "new_text": "ALPHA\nBETA"},
    ]});
    assert!(edit.validate(&args).is_ok());
    let predicted = edit
        .file_write_result(&args.to_string(), Some(&std::fs::read(&file).unwrap()))
        .unwrap()
        .unwrap();
    let out = edit.call(args, &ctx()).await.unwrap();
    assert!(!out.is_error, "{}", out.content);
    assert!(
        out.content
            .contains("applied 3 edits, replaced 4 occurrences"),
        "{}",
        out.content
    );
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "ALPHA\nBETA\ndelta delta\n"
    );
    assert_eq!(std::fs::read(&file).unwrap(), predicted);
}

/// When any edit of a batch fails, the file is unchanged and the reply names
/// the failing edit and why.
#[tokio::test]
async fn edit_batch_writes_nothing_when_one_fails() {
    use aidaemon_tools::semantics::ToolSemantics;
    let (dir, workspace) = workspace();
    let file = dir.path().join("a.txt");
    std::fs::write(&file, "one\ntwo\ntwo\n").unwrap();
    let edit = EditFile::new(workspace);

    for (args, why) in [
        (
            json!({"path": "a.txt", "edits": [
                {"old_text": "one", "new_text": "ONE"},
                {"old_text": "absent", "new_text": "x"},
            ]}),
            "Text not found",
        ),
        (
            json!({"path": "a.txt", "edits": [
                {"old_text": "one", "new_text": "ONE"},
                {"old_text": "two", "new_text": "TWO"},
            ]}),
            "2 occurrences",
        ),
        // The first edit removes what the second looks for.
        (
            json!({"path": "a.txt", "edits": [
                {"old_text": "one\n", "new_text": "uno\n"},
                {"old_text": "one", "new_text": "x"},
            ]}),
            "Text not found",
        ),
    ] {
        let predicted =
            edit.file_write_result(&args.to_string(), Some(&std::fs::read(&file).unwrap()));
        assert!(matches!(predicted, Some(Err(_))), "{args}");
        let out = edit.call(args.clone(), &ctx()).await.unwrap();
        assert!(out.is_error, "{args}");
        assert!(
            out.content.contains("Edit 2 of 2") && out.content.contains(why),
            "{}",
            out.content
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "one\ntwo\ntwo\n");
    }

    for bad in [
        json!({"path": "a", "edits": []}),
        json!({"path": "a", "edits": [{"old_text": "", "new_text": "y"}]}),
        json!({"path": "a", "edits": [{"old_text": "x", "new_text": "x"}]}),
        json!({"path": "a", "edits": [{"old_text": "x"}]}),
        json!({"path": "a", "old_text": "x", "new_text": "y",
               "edits": [{"old_text": "x", "new_text": "y"}]}),
    ] {
        assert!(edit.validate(&bad).is_err(), "{bad}");
    }
}
