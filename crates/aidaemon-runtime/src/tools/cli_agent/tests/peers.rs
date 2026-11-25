//! Tests for `CliAgentTool` peer coordination: worktree isolation, narrowed
//! subpath authority, permit parking without deadlock, and direct-agent peer
//! routing.

use super::*;
use crate::tools::cli_agent::peers as peer_mod;
use aidaemon_coding_agents::peer::{PeerLimits, PeerWorkspaceMode};

#[test]
fn subpath_validation_enforces_narrowing_and_blocks_escape() {
    let parent = vec!["/synthetic/repo/src".to_string()];

    // Inside parent scope: allowed.
    let ok = peer_mod::validate_narrowed_write_paths(
        &["/synthetic/repo/src/api.rs".to_string()],
        Some("/synthetic/repo"),
        &parent,
        true,
    )
    .unwrap();
    assert_eq!(ok, vec!["/synthetic/repo/src/api.rs"]);

    // Relative path resolving inside parent scope: allowed.
    let ok_rel = peer_mod::validate_narrowed_write_paths(
        &["src/ui/mod.rs".to_string()],
        Some("/synthetic/repo"),
        &parent,
        true,
    )
    .unwrap();
    assert_eq!(ok_rel, vec!["/synthetic/repo/src/ui/mod.rs"]);

    // Traversal escaping parent scope: refused.
    let escaped = peer_mod::validate_narrowed_write_paths(
        &["/synthetic/repo/src/../../etc/passwd".to_string()],
        Some("/synthetic/repo"),
        &parent,
        true,
    );
    assert!(escaped.is_err());

    // Sibling path outside parent scope: refused.
    let outside = peer_mod::validate_narrowed_write_paths(
        &["/synthetic/repo/Cargo.toml".to_string()],
        Some("/synthetic/repo"),
        &parent,
        true,
    );
    assert!(outside.is_err());
}

/// A symlink inside the lead's folder that points out of it does not widen
/// a peer's write root.
#[cfg(unix)]
#[test]
fn subpath_narrowing_follows_symlinks() {
    let lead = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(elsewhere.path(), lead.path().join("link")).unwrap();
    let parent = vec![lead.path().to_string_lossy().into_owned()];
    let escaped = peer_mod::validate_narrowed_write_paths(
        &["link/inside".to_string()],
        Some(&parent[0]),
        &parent,
        true,
    );
    assert!(escaped.is_err(), "{escaped:?}");
    std::fs::create_dir(lead.path().join("real")).unwrap();
    let inside = peer_mod::validate_narrowed_write_paths(
        &["real/new-file".to_string()],
        Some(&parent[0]),
        &parent,
        true,
    )
    .unwrap();
    assert!(inside[0].ends_with("real/new-file"));
}

#[tokio::test]
async fn permit_slot_parks_and_restores_without_deadlock() {
    // Pool of size 1: if a waiting caller did not park its permit, the child
    // could never acquire a slot.
    let limiter = Arc::new(Semaphore::new(1));
    let finished = Arc::new(AtomicBool::new(false));
    let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let permit = limiter.clone().acquire_owned().await.unwrap();
    let slot =
        peer_mod::PermitSlot::new(permit, limiter.clone(), finished.clone(), in_flight.clone());

    assert_eq!(limiter.available_permits(), 0);
    let guard = slot.park();
    assert_eq!(limiter.available_permits(), 1);
    assert_eq!(in_flight.load(Ordering::Acquire), 1);

    // Child acquires and releases the freed slot while parent is parked.
    let child_permit = limiter
        .clone()
        .try_acquire_owned()
        .expect("child must get parked slot");
    drop(child_permit);

    // Dropping the guard restores the parent's permit and unpauses idle timer.
    drop(guard);
    assert_eq!(limiter.available_permits(), 0);
    assert_eq!(in_flight.load(Ordering::Acquire), 0);

    // Releasing on exit frees the slot cleanly.
    finished.store(true, Ordering::Release);
    slot.release();
    assert_eq!(limiter.available_permits(), 1);
}

#[tokio::test]
async fn worktree_provisioning_creates_isolated_branch_and_directory() {
    let repo = tempfile::tempdir().unwrap();
    let root = repo.path().to_string_lossy().into_owned();
    let backend = aidaemon_tools::testing::active_execution_backend();

    // Initialize a real temporary git repo with one commit.
    let git = |args: &[&str]| {
        let mut cmd = std::process::Command::new("git");
        cmd.arg("-C").arg(&root).args(args);
        cmd.env("GIT_AUTHOR_NAME", "Alice Rivera");
        cmd.env("GIT_AUTHOR_EMAIL", "alice@example.com");
        cmd.env("GIT_COMMITTER_NAME", "Alice Rivera");
        cmd.env("GIT_COMMITTER_EMAIL", "alice@example.com");
        let status = cmd.status().unwrap();
        assert!(status.success(), "git {:?} failed", args);
    };
    git(&["init"]);
    std::fs::write(repo.path().join("README.md"), "synthetic repo\n").unwrap();
    git(&["add", "README.md"]);
    git(&["commit", "-m", "initial"]);

    let tree_id = format!("t{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
    let worktree = peer_mod::provision_peer_worktree(backend.as_ref(), &root, &tree_id, "peer-1")
        .await
        .unwrap();
    let (wt_path, branch) = (worktree.path.clone(), worktree.branch.clone());

    assert!(std::path::Path::new(&wt_path).join("README.md").is_file());
    assert!(
        branch.starts_with(&format!("aidaemon/peer-{tree_id}-peer-1-")),
        "{branch}"
    );
    assert_ne!(wt_path, root);
    let tree_dir = std::path::Path::new(&wt_path)
        .parent()
        .unwrap()
        .to_path_buf();

    // A peer's uncommitted work is committed to its branch, which is kept.
    std::fs::write(
        std::path::Path::new(&wt_path).join("notes.md"),
        "peer work\n",
    )
    .unwrap();
    let released = peer_mod::release_peer_worktree(backend.as_ref(), &worktree)
        .await
        .unwrap();
    assert_eq!(released.kept_branch.as_deref(), Some(branch.as_str()));
    assert!(released.committed);
    assert!(!std::path::Path::new(&wt_path).exists());
    assert!(
        !tree_dir.exists(),
        "the tree's folder goes with its last worktree"
    );
    let shown = std::process::Command::new("git")
        .args(["-C", &root, "show", &format!("{branch}:notes.md")])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&shown.stdout), "peer work\n");

    // A worktree with no work: its branch goes too.
    let idle = peer_mod::provision_peer_worktree(backend.as_ref(), &root, &tree_id, "peer-2")
        .await
        .unwrap();
    let released = peer_mod::release_peer_worktree(backend.as_ref(), &idle)
        .await
        .unwrap();
    assert_eq!(
        released,
        peer_mod::ReleasedWorktree {
            kept_branch: None,
            committed: false
        }
    );
    let branches = std::process::Command::new("git")
        .args(["-C", &root, "branch", "--list", &idle.branch])
        .output()
        .unwrap();
    assert!(branches.stdout.is_empty());
}

#[tokio::test]
async fn enable_peer_coordination_and_run_child_peer_in_direct_agent_session() {
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    tool.enable_peer_coordination_with_limits(
        "echo",
        vec![],
        PeerLimits {
            max_depth: 2,
            max_peers: 4,
            max_messages: 16,
            max_board_wait_secs: 5,
        },
    )
    .await
    .unwrap();

    // Pin the session to a direct coding agent (AIdaemon LLM bypassed).
    crate::session_model::set(
        tool.state.as_ref(),
        "app:direct-peer-test",
        &crate::session_model::SessionModelSetting {
            pinned: None,
            effort: None,
            agent: Some("bash-agent".into()),
        },
    )
    .await
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let dir_str = dir.path().to_string_lossy().into_owned();
    let coord = tool.peer_coord.get().expect("peer_coord enabled");
    let session = coord.start_root_tree(
        "bash-agent",
        "Lead task",
        peer_mod::PeerRootContext {
            lead_mode: Some(aidaemon_coding_agents::PermissionMode::FullAccess),
            session_id: "app:direct-peer-test".into(),
            grant_session: "app:direct-peer-test".into(),
            working_dir: Some(dir_str.clone()),
            read_paths: vec![dir_str.clone()],
            write_paths: vec![dir_str.clone()],
            enforce_native_access: false,
            project: None,
            detached_judge: None,
            spawn_family: None,
            status_tx: None,
            may_send_files: false,
        },
        &["bash-agent".into()],
    );

    // Launch a read-only child peer synchronously via the RuntimePeerLauncher.
    use aidaemon_coding_agents::peer::{PeerLaunchRequest, PeerLauncher};
    let (peer_id, depth, cancel) = session
        .tree
        .reserve_child(
            "peer-root",
            "bash-agent",
            "Inspect dir",
            PeerWorkspaceMode::ReadOnly,
        )
        .unwrap();
    let launched = session
        .launcher
        .launch(PeerLaunchRequest {
            tree_id: session.tree.id().to_string(),
            peer_id,
            parent_peer_id: "peer-root".into(),
            depth,
            agent: "bash-agent".into(),
            title: "Inspect dir".into(),
            prompt: "echo peer-hello".into(),
            workspace: PeerWorkspaceMode::ReadOnly,
            write_paths: Vec::new(),
            continue_run: None,
            wait: true,
            mcp_endpoint: None,
            mcp_keep_alive: None,
            cancel,
        })
        .await
        .unwrap();

    assert!(launched.finished);
    assert_eq!(launched.success, Some(true));
    assert!(
        launched
            .output
            .as_deref()
            .unwrap_or_default()
            .contains("peer-hello"),
        "unexpected peer output: {:?}",
        launched.output
    );
}

/// A peer's `send_file` passes each check in turn: the owner's private
/// conversation, file delivery set up, the run's folders (secrets never),
/// then the conversation's channel.
#[tokio::test]
async fn a_peer_sends_files_only_to_a_private_owner_chat_from_its_folders() {
    use aidaemon_coding_agents::peer::PeerLauncher;
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    tool.enable_peer_coordination_with_limits("echo", vec![], PeerLimits::default())
        .await
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let dir_str = std::fs::canonicalize(dir.path())
        .unwrap()
        .to_string_lossy()
        .into_owned();
    std::fs::write(dir.path().join("shot.png"), b"synthetic png").unwrap();
    std::fs::write(dir.path().join(".env"), b"SECRET=synthetic").unwrap();
    let coord = tool.peer_coord.get().expect("peer_coord enabled");
    let root = |may_send_files: bool| peer_mod::PeerRootContext {
        lead_mode: Some(aidaemon_coding_agents::PermissionMode::FullAccess),
        session_id: "app:send-file-test".into(),
        grant_session: "app:send-file-test".into(),
        working_dir: Some(dir_str.clone()),
        read_paths: vec![dir_str.clone()],
        write_paths: Vec::new(),
        enforce_native_access: false,
        project: None,
        detached_judge: None,
        spawn_family: None,
        status_tx: None,
        may_send_files,
    };

    let shared = coord.start_root_tree("bash-agent", "Lead", root(false), &[]);
    let refused = shared
        .launcher
        .send_file("peer-root", "shot.png", None)
        .await
        .unwrap_err();
    assert!(refused.contains("private conversation"), "{refused}");

    let private = coord.start_root_tree("bash-agent", "Lead", root(true), &[]);
    let unset = private
        .launcher
        .send_file("peer-root", "shot.png", None)
        .await
        .unwrap_err();
    assert!(unset.contains("no file delivery"), "{unset}");

    let inbox = tempfile::tempdir().unwrap();
    tool.enable_peer_file_delivery(inbox.path());
    let secret = private
        .launcher
        .send_file("peer-root", ".env", None)
        .await
        .unwrap_err();
    assert!(secret.contains("never sent"), "{secret}");
    let missing = private
        .launcher
        .send_file("peer-root", "nope.png", None)
        .await
        .unwrap_err();
    assert!(missing.contains("no file at"), "{missing}");
    // Every check passed: only the (unwired) channel is left.
    let channel = private
        .launcher
        .send_file("peer-root", "shot.png", Some("Synthetic shot"))
        .await
        .unwrap_err();
    assert!(channel.contains("not connected"), "{channel}");
}
