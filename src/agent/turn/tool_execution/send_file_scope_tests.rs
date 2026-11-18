//! The send_file adapter against the task scope check.

use crate::traits::{ToolTargetHint, ToolTargetHintKind};
use aidaemon_runtime::turn::tool_dispatch::access_manifest_scope_violation;

/// Live 2026-11-11: a sub-agent confined to its project took a browser
/// screenshot (saved in the daemon inbox) and `send_file` was refused as
/// an "unauthorized write" of that screenshot. Sending a file reads it;
/// the inbox is the adapter's own; a file elsewhere stays out of scope.
#[test]
fn a_confined_child_can_send_its_own_screenshot_but_nothing_else() {
    use crate::traits::ToolSemantics as _;
    let tmp = tempfile::tempdir().expect("tempdir");
    let inbox = tmp.path().join("files/inbox");
    let project = tmp.path().join("projects/acme-site");
    let elsewhere = tmp.path().join("private");
    for dir in [&inbox, &project, &elsewhere] {
        std::fs::create_dir_all(dir).expect("dir");
    }
    let (media_tx, _media_rx) = tokio::sync::mpsc::channel(1);
    let tool = crate::tools::SendFileTool::new(
        crate::execution::active_execution_backend(),
        media_tx,
        &[tmp.path().to_string_lossy().to_string()],
        &inbox.to_string_lossy(),
    );
    let task = crate::traits::ToolCallAccessManifest {
        read_targets: vec![
            ToolTargetHint::new(ToolTargetHintKind::ProjectScope, project.to_string_lossy())
                .unwrap(),
        ],
        write_targets: vec![
            ToolTargetHint::new(ToolTargetHintKind::ProjectScope, project.to_string_lossy())
                .unwrap(),
        ],
        ..Default::default()
    };
    let check = |path: &std::path::Path| {
        let args = serde_json::json!({ "file_path": path.to_string_lossy() }).to_string();
        let mut manifest = tool.call_access_manifest(&args);
        assert!(manifest.write_targets.is_empty(), "sending writes nothing");
        manifest.adapter_read_targets = tool
            .adapter_owned_access_manifest(&args)
            .adapter_read_targets;
        access_manifest_scope_violation(
            "send_file",
            &manifest,
            Some(&task),
            &[],
            None,
            &crate::tools::DaemonPaths::default(),
        )
    };
    assert_eq!(check(&inbox.join("5a947c87_screenshot.png")), None);
    assert_eq!(check(&project.join("homepage.png")), None);
    let refused = check(&elsewhere.join("notes.txt")).expect("outside the scope");
    assert!(refused.contains("unauthorized reads"), "{refused}");
    assert!(check(&inbox.join("../private/notes.txt")).is_some());
}
