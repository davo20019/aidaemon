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

/// A worktree peer resumed in a later exchange, after its lead's end
/// released its worktree, gets its kept branch checked out again; a branch
/// that held nothing (deleted) gives it a fresh worktree.
#[tokio::test]
async fn a_released_peer_worktree_is_restored_from_its_branch() {
    let repo = tempfile::tempdir().unwrap();
    let root = repo.path().to_string_lossy().into_owned();
    let backend = aidaemon_tools::testing::active_execution_backend();
    let git = |args: &[&str]| {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(args)
            .env("GIT_AUTHOR_NAME", "Alice Rivera")
            .env("GIT_AUTHOR_EMAIL", "alice@example.com")
            .env("GIT_COMMITTER_NAME", "Alice Rivera")
            .env("GIT_COMMITTER_EMAIL", "alice@example.com")
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
    };
    git(&["init"]);
    std::fs::write(repo.path().join("README.md"), "synthetic repo\n").unwrap();
    git(&["add", "README.md"]);
    git(&["commit", "-m", "initial"]);
    let tree_id = format!("t{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);

    let first = peer_mod::provision_peer_worktree(backend.as_ref(), &root, &tree_id, "peer-1")
        .await
        .unwrap();
    std::fs::write(
        std::path::Path::new(&first.path).join("notes.md"),
        "peer work\n",
    )
    .unwrap();
    peer_mod::release_peer_worktree(backend.as_ref(), &first)
        .await
        .unwrap();
    assert!(!std::path::Path::new(&first.path).exists());

    let restored =
        peer_mod::restore_peer_worktree(backend.as_ref(), &root, &tree_id, "peer-1", &first.branch)
            .await
            .unwrap();
    assert_eq!(restored.branch, first.branch);
    assert_ne!(restored.path, first.path);
    assert_eq!(
        std::fs::read_to_string(std::path::Path::new(&restored.path).join("notes.md")).unwrap(),
        "peer work\n",
        "the resumed peer finds its earlier work"
    );
    let released = peer_mod::release_peer_worktree(backend.as_ref(), &restored)
        .await
        .unwrap();
    assert_eq!(released.kept_branch.as_deref(), Some(first.branch.as_str()));

    // A branch that held nothing was deleted: a fresh worktree instead.
    let idle = peer_mod::provision_peer_worktree(backend.as_ref(), &root, &tree_id, "peer-2")
        .await
        .unwrap();
    peer_mod::release_peer_worktree(backend.as_ref(), &idle)
        .await
        .unwrap();
    let fresh =
        peer_mod::restore_peer_worktree(backend.as_ref(), &root, &tree_id, "peer-2", &idle.branch)
            .await
            .unwrap();
    assert_ne!(fresh.branch, idle.branch);
    assert!(
        std::path::Path::new(&fresh.path)
            .join("README.md")
            .is_file()
    );
    peer_mod::release_peer_worktree(backend.as_ref(), &fresh)
        .await
        .unwrap();
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
        None,
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
            requested_by: "peer-root".into(),
            depth,
            agent: "bash-agent".into(),
            title: "Inspect dir".into(),
            prompt: "echo peer-hello".into(),
            workspace: PeerWorkspaceMode::ReadOnly,
            write_paths: Vec::new(),
            files: Vec::new(),
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

    let shared = coord.start_root_tree("bash-agent", "Lead", root(false), &[], None);
    let refused = shared
        .launcher
        .send_file("peer-root", "shot.png", None)
        .await
        .unwrap_err();
    assert!(refused.contains("private conversation"), "{refused}");

    let private = coord.start_root_tree("bash-agent", "Lead", root(true), &[], None);
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

/// A peer's `share_files` copies what its tree may read (and the owner's
/// uploads in the inbox) into each recipient's own folder, where a
/// confined recipient's sandbox reaches it; anything else is refused.
#[tokio::test]
async fn a_peer_shares_files_into_the_recipients_folder() {
    use aidaemon_coding_agents::peer::{PeerLauncher, PeerState, PeerUpdate};
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    tool.enable_peer_coordination_with_limits("echo", vec![], PeerLimits::default())
        .await
        .unwrap();
    let lead = tempfile::tempdir().unwrap();
    let lead_dir = std::fs::canonicalize(lead.path()).unwrap();
    std::fs::write(lead_dir.join("review.md"), b"synthetic review").unwrap();
    let inbox = tempfile::tempdir().unwrap();
    std::fs::write(inbox.path().join("shot.png"), b"synthetic png").unwrap();
    tool.enable_peer_file_delivery(inbox.path());
    let lead_str = lead_dir.to_string_lossy().into_owned();
    let coord = tool.peer_coord.get().expect("peer_coord enabled");
    let session = coord.start_root_tree(
        "bash-agent",
        "Lead",
        peer_mod::PeerRootContext {
            lead_mode: Some(aidaemon_coding_agents::PermissionMode::FullAccess),
            session_id: "app:share-files-test".into(),
            grant_session: "app:share-files-test".into(),
            working_dir: Some(lead_str.clone()),
            read_paths: vec![lead_str.clone()],
            write_paths: Vec::new(),
            enforce_native_access: false,
            project: None,
            detached_judge: None,
            spawn_family: None,
            status_tx: None,
            may_send_files: false,
        },
        &[],
        None,
    );
    // A peer working in its own folder, as a worktree peer does.
    let (peer_id, _, _) = session
        .tree
        .reserve_child(
            "peer-root",
            "bash-agent",
            "Review",
            PeerWorkspaceMode::ReadOnly,
        )
        .unwrap();
    let own = tempfile::tempdir().unwrap();
    session.tree.update_peer(
        &peer_id,
        PeerUpdate {
            state: PeerState::Running,
            run_id: None,
            working_dir: Some(own.path().to_string_lossy().into_owned()),
            branch: None,
            output: None,
            resumable: None,
        },
    );

    let inbox_shot = inbox.path().join("shot.png").to_string_lossy().into_owned();
    let note = session
        .launcher
        .share_files(
            "peer-root",
            &["review.md".to_string(), inbox_shot],
            std::slice::from_ref(&peer_id),
        )
        .await
        .unwrap();
    let copies: Vec<std::path::PathBuf> = note
        .lines()
        .skip(1)
        .map(|line| {
            line.trim_start_matches("- `")
                .trim_end_matches(['`', ']'])
                .into()
        })
        .collect();
    assert_eq!(copies.len(), 2, "{note}");
    assert!(
        copies.iter().all(|copy| copy.starts_with(own.path())),
        "{note}"
    );
    assert_eq!(std::fs::read(&copies[0]).unwrap(), b"synthetic review");
    assert_eq!(std::fs::read(&copies[1]).unwrap(), b"synthetic png");

    let unknown = session
        .launcher
        .share_files("peer-root", &["review.md".to_string()], &["peer-9".into()])
        .await
        .unwrap_err();
    assert!(unknown.contains("unknown peer"), "{unknown}");
    // Neither the tree's folders, the inbox nor a temp folder.
    let home = dirs::home_dir().unwrap();
    let outside = tempfile::Builder::new()
        .prefix("aidaemon-share-test")
        .tempfile_in(&home)
        .unwrap();
    let refused = session
        .launcher
        .share_files(
            "peer-root",
            &[outside.path().to_string_lossy().into_owned()],
            std::slice::from_ref(&peer_id),
        )
        .await
        .unwrap_err();
    assert!(refused.contains("outside your folders"), "{refused}");
}

/// An owner's conversation keeps one tree for its lead agent: the next run
/// continues it (board and finished peers), unless the owner asked for a
/// fresh session, the folder changed, or the earlier run still runs.
#[tokio::test]
async fn a_conversations_next_lead_run_continues_its_tree() {
    use aidaemon_coding_agents::peer::{PeerState, PeerUpdate};
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    tool.enable_peer_coordination_with_limits("echo", vec![], PeerLimits::default())
        .await
        .unwrap();
    let coord = tool.peer_coord.get().expect("peer_coord enabled");
    let root = |dir: &str| peer_mod::PeerRootContext {
        lead_mode: Some(aidaemon_coding_agents::PermissionMode::FullAccess),
        session_id: "app:synthetic-continue".into(),
        grant_session: "app:synthetic-continue".into(),
        working_dir: Some(dir.to_string()),
        read_paths: vec![dir.to_string()],
        write_paths: Vec::new(),
        enforce_native_access: false,
        project: None,
        detached_judge: None,
        spawn_family: None,
        status_tx: None,
        may_send_files: false,
    };
    let continuity = |fresh: bool, run: &str| {
        Some(peer_mod::TreeContinuity {
            lead: peer_mod::ConversationLead {
                session_id: "app:synthetic-continue".into(),
                agent: "bash-agent".into(),
            },
            fresh,
            run_id: run.into(),
        })
    };
    let end = |tree: &aidaemon_coding_agents::peer::PeerTree| {
        tree.cancel_subtree("peer-root");
        tree.update_peer(
            "peer-root",
            PeerUpdate {
                state: PeerState::Completed,
                run_id: None,
                working_dir: None,
                branch: None,
                output: None,
                resumable: None,
            },
        );
    };

    let first = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/a"),
        &[],
        continuity(false, "run-1"),
    );
    first
        .tree
        .post_message("peer-root", None, "plan", "HERON note")
        .unwrap();
    // A follow-up queued while the first run still runs is prepared on a
    // stand-in tree, and the conversation keeps the running one.
    let queued = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/a"),
        &[],
        continuity(false, "run-2"),
    );
    assert_ne!(queued.tree.id(), first.tree.id());
    let mut binding = peer_mod::ActivePeerBinding {
        tree: queued.tree.clone(),
        peer_id: "peer-root".into(),
        permit_slots: queued.permit_slots.clone(),
        is_root: true,
        launcher: Some(queued.launcher.clone()),
        branch: None,
        rebind: queued.rebind.clone(),
        conversation: None,
    };
    // Still running at start: it stays on its own tree.
    let mut early = binding.clone();
    assert!(!early.adopt_conversation_tree(&queued.scoped));
    assert_eq!(early.tree.id(), queued.tree.id());
    // Once the first run ended, starting takes the conversation's tree,
    // and the endpoint the agent was given reaches it.
    end(&first.tree);
    assert!(binding.adopt_conversation_tree(&queued.scoped));
    assert_eq!(binding.tree.id(), first.tree.id());
    assert_eq!(
        binding.tree.peer("peer-root").unwrap().run_id.as_deref(),
        Some("run-2")
    );
    assert_eq!(
        binding.tree.read_messages("peer-root", 0, false)[0].body,
        "HERON note"
    );
    assert!(
        !binding.adopt_conversation_tree(&queued.scoped),
        "only once"
    );
    end(&binding.tree);

    // The real case: one run ends, the next continues its tree.
    let a = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/a"),
        &[],
        continuity(true, "run-3"),
    );
    a.tree
        .post_message("peer-root", None, "plan", "HERON first")
        .unwrap();
    end(&a.tree);
    let b = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/a"),
        &[],
        continuity(false, "run-4"),
    );
    assert_eq!(b.tree.id(), a.tree.id(), "the follow-up continues the tree");
    let board = b.tree.read_messages("peer-root", 0, false);
    assert_eq!(board.len(), 1);
    assert_eq!(board[0].body, "HERON first");
    assert_eq!(
        b.tree.peer("peer-root").unwrap().run_id.as_deref(),
        Some("run-4")
    );
    end(&b.tree);

    // A fresh session or another folder starts over.
    let fresh = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/a"),
        &[],
        continuity(true, "run-5"),
    );
    assert_ne!(fresh.tree.id(), b.tree.id());
    end(&fresh.tree);
    let moved = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/b"),
        &[],
        continuity(false, "run-6"),
    );
    assert_ne!(moved.tree.id(), fresh.tree.id());
    // Without continuity (a delegation from a regular chat) it is always new.
    end(&moved.tree);
    let other = coord.start_root_tree("bash-agent", "Lead", root("/w/b"), &[], None);
    assert_ne!(other.tree.id(), moved.tree.id());
    end(&other.tree);

    // A follow-up queued in another folder can't take the conversation's
    // tree; once that lead ended, its own tree is the conversation's.
    let in_a = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/d"),
        &[],
        continuity(true, "run-9"),
    );
    let elsewhere = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/e"),
        &[],
        continuity(false, "run-10"),
    );
    let mut elsewhere_binding = peer_mod::ActivePeerBinding {
        tree: elsewhere.tree.clone(),
        peer_id: "peer-root".into(),
        permit_slots: elsewhere.permit_slots.clone(),
        is_root: true,
        launcher: Some(elsewhere.launcher.clone()),
        branch: None,
        rebind: elsewhere.rebind.clone(),
        conversation: None,
    };
    end(&in_a.tree);
    assert!(!elsewhere_binding.adopt_conversation_tree(&elsewhere.scoped));
    end(&elsewhere.tree);
    let next = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/e"),
        &[],
        continuity(false, "run-11"),
    );
    assert_eq!(
        next.tree.id(),
        elsewhere.tree.id(),
        "the next exchange continues the run that ran last"
    );
    end(&next.tree);

    // A run ends (root marked done), the queued follow-up takes the tree
    // before the ended run's cleanup: that late cleanup leaves the new
    // run's root alone.
    let ending = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/c"),
        &[],
        continuity(true, "run-7"),
    );
    ending.tree.set_peer_run_id("peer-root", "run-7");
    let ending_binding = peer_mod::ActivePeerBinding {
        tree: ending.tree.clone(),
        peer_id: "peer-root".into(),
        permit_slots: ending.permit_slots.clone(),
        is_root: true,
        launcher: Some(ending.launcher.clone()),
        branch: None,
        rebind: None,
        conversation: None,
    };
    let follow = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root("/w/c"),
        &[],
        continuity(false, "run-8"),
    );
    let mut follow_binding = peer_mod::ActivePeerBinding {
        tree: follow.tree.clone(),
        peer_id: "peer-root".into(),
        permit_slots: follow.permit_slots.clone(),
        is_root: true,
        launcher: Some(follow.launcher.clone()),
        branch: None,
        rebind: follow.rebind.clone(),
        conversation: None,
    };
    ending.tree.update_peer(
        "peer-root",
        PeerUpdate {
            state: PeerState::Completed,
            run_id: None,
            working_dir: None,
            branch: None,
            output: None,
            resumable: None,
        },
    );
    assert!(follow_binding.adopt_conversation_tree(&follow.scoped));
    ending_binding.cleanup_children("run-7").await;
    let root_now = follow_binding.tree.peer("peer-root").unwrap();
    assert_eq!(
        root_now.state,
        PeerState::Running,
        "the late cleanup left it alone"
    );
    assert_eq!(root_now.run_id.as_deref(), Some("run-8"));
    follow_binding.cleanup_children("run-8").await;
    assert_eq!(
        follow_binding.tree.peer("peer-root").unwrap().state,
        PeerState::Cancelled,
        "its own run's cleanup still ends it"
    );
}

/// A conversation's tree survives a daemon restart: saved when its lead
/// run ends, read back for the next exchange.
#[tokio::test]
async fn a_conversations_tree_survives_a_restart() {
    use aidaemon_coding_agents::peer::{PeerState, PeerUpdate};
    let (tool, _db) = setup_bash_tool().await;
    let tool = Arc::new(tool);
    tool.enable_peer_coordination_with_limits("echo", vec![], PeerLimits::default())
        .await
        .unwrap();
    let coord = tool.peer_coord.get().expect("peer_coord enabled");
    let root = || peer_mod::PeerRootContext {
        lead_mode: Some(aidaemon_coding_agents::PermissionMode::FullAccess),
        session_id: "app:synthetic-restart".into(),
        grant_session: "app:synthetic-restart".into(),
        working_dir: Some("/w/r".into()),
        read_paths: vec!["/w/r".into()],
        write_paths: Vec::new(),
        enforce_native_access: false,
        project: None,
        detached_judge: None,
        spawn_family: None,
        status_tx: None,
        may_send_files: false,
    };
    let lead = peer_mod::ConversationLead {
        session_id: "app:synthetic-restart".into(),
        agent: "bash-agent".into(),
    };
    let continuity = |run: &str| {
        Some(peer_mod::TreeContinuity {
            lead: lead.clone(),
            fresh: false,
            run_id: run.into(),
        })
    };
    let first = coord.start_root_tree("bash-agent", "Lead", root(), &[], continuity("run-1"));
    first
        .tree
        .post_message("peer-root", None, "plan", "HERON before restart")
        .unwrap();
    let binding = peer_mod::ActivePeerBinding {
        tree: first.tree.clone(),
        peer_id: "peer-root".into(),
        permit_slots: first.permit_slots.clone(),
        is_root: true,
        launcher: Some(first.launcher.clone()),
        branch: None,
        rebind: None,
        conversation: first.conversation.clone(),
    };
    first.tree.update_peer(
        "peer-root",
        PeerUpdate {
            state: PeerState::Completed,
            run_id: None,
            working_dir: None,
            branch: None,
            output: None,
            resumable: None,
        },
    );
    binding.cleanup_children("run-1").await;
    binding.save_conversation_tree().await;

    coord.forget_conversation_trees();
    coord.load_conversation_tree(&lead).await;
    let next = coord.start_root_tree("bash-agent", "Lead", root(), &[], continuity("run-2"));
    assert_eq!(next.tree.id(), first.tree.id());
    let board = next.tree.read_messages("peer-root", 0, false);
    assert_eq!(board.len(), 1);
    assert_eq!(board[0].body, "HERON before restart");
    assert_eq!(
        next.tree.peer("peer-root").unwrap().run_id.as_deref(),
        Some("run-2")
    );

    // Nothing saved for another conversation: a fresh tree.
    let other = peer_mod::ConversationLead {
        session_id: "app:synthetic-other".into(),
        agent: "bash-agent".into(),
    };
    coord.load_conversation_tree(&other).await;
    let fresh = coord.start_root_tree(
        "bash-agent",
        "Lead",
        root(),
        &[],
        Some(peer_mod::TreeContinuity {
            lead: other,
            fresh: false,
            run_id: "run-3".into(),
        }),
    );
    assert_ne!(fresh.tree.id(), first.tree.id());
}
