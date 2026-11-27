//! Background launches past the concurrency cap wait in line for a slot.

use super::*;

/// Records every run change, as the app is told of them.
#[derive(Default)]
struct Watcher(std::sync::Mutex<Vec<aidaemon_channels::turns::RunChange>>);

impl aidaemon_channels::turns::TurnObserver for Watcher {
    fn turn_started(&self, _turn: &aidaemon_channels::turns::TurnStart<'_>) {}
    fn turn_status(
        &self,
        _session_id: &str,
        _turn: u64,
        _update: &aidaemon_tools::status::StatusUpdate,
    ) {
    }
    fn turn_finished(&self, _turn: &aidaemon_channels::turns::TurnFinish<'_>) {}
    fn run_changed(&self, change: &aidaemon_channels::turns::RunChange) {
        self.0.lock().unwrap().push(change.clone());
    }
}

impl Watcher {
    /// What the app was last told of run `task_id`.
    fn last(&self, task_id: &str) -> Option<aidaemon_channels::turns::RunChange> {
        let id = cli_run_id(task_id);
        self.0
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|change| change.id == id)
            .cloned()
    }

    /// Wait up to `secs` for run `task_id` to end, as the app is told.
    async fn ended(&self, task_id: &str, secs: u64) -> aidaemon_channels::turns::RunChange {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
        loop {
            let last = self.last(task_id).expect("the app was told of the run");
            if last.view.phase.is_terminal() || tokio::time::Instant::now() >= deadline {
                return last;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// A tool with room for one agent at a time, its runs reported to a
/// watcher.
async fn one_slot_tool() -> (
    CliAgentTool,
    Arc<crate::run_state::RunStates>,
    Arc<Watcher>,
    tempfile::NamedTempFile,
) {
    let (mut tool, db) = setup_bash_tool().await;
    tool.max_concurrent = 1;
    tool.concurrency_limiter = Arc::new(Semaphore::new(1));
    let runs = Arc::new(crate::run_state::RunStates::default());
    let watch = aidaemon_channels::turns::TurnWatch::default();
    let watcher = Arc::new(Watcher::default());
    watch.set(watcher.clone());
    runs.report_to(watch);
    tool.set_runs(runs.clone());
    (tool, runs, watcher, db)
}

async fn launch(tool: &CliAgentTool, chat: &str, args: &str) -> String {
    tool.call_with_status_under_contract(
        args,
        None,
        false,
        false,
        &aidaemon_tools::testing::call_facts(chat),
    )
    .await
    .unwrap()
}

/// The task ids of the invocations logged so far, oldest first: one per
/// agent process started.
async fn started_task_ids(tool: &CliAgentTool) -> Vec<String> {
    let mut ids: Vec<String> = tool
        .state
        .get_cli_agent_invocations(64)
        .await
        .unwrap()
        .into_iter()
        .filter_map(|invocation| invocation.task_id)
        .collect();
    ids.reverse();
    ids
}

/// The background runs on record for `chat` (what a restart would report),
/// by run id.
async fn on_record(tool: &CliAgentTool, chat: &str) -> Vec<String> {
    let mut ids: Vec<String> = tool
        .state
        .background_spawns_for_session(chat)
        .await
        .unwrap()
        .into_iter()
        .map(|spawn| spawn.id)
        .collect();
    ids.sort();
    ids
}

/// Every slot taken, a background launch is accepted at once: it has its
/// task id and its record (reading "Waiting for a free slot"), and it starts
/// by itself, in the order launches came in, once a running one ends.
#[tokio::test]
async fn a_background_launch_past_the_cap_waits_and_starts_when_a_slot_frees() {
    let (tool, runs, watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-queue";

    let first = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 2; echo first-done","async_mode":true}"#,
    )
    .await;
    assert!(first.contains("started in background"), "{first}");
    let first_id = extract_task_id_from_background_message(&first);

    let second = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"echo second-done","async_mode":true}"#,
    )
    .await;
    assert!(
        second.starts_with("Started in the background")
            && second.contains("is queued (task_id=")
            && second.contains("next in line")
            && second.contains("starts by itself"),
        "{second}"
    );
    let second_id = extract_task_id_from_background_message(&second);

    let third = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"echo third-done","async_mode":true}"#,
    )
    .await;
    assert!(third.contains("1 queued ahead of it"), "{third}");
    let third_id = extract_task_id_from_background_message(&third);

    // Queued: a run on record, waiting, with no process started.
    let waiting = runs.get(&cli_run_id(&second_id)).unwrap();
    assert_eq!(waiting.phase, crate::run_state::RunPhase::Running);
    assert_eq!(waiting.activity.as_deref(), Some(WAITING_FOR_SLOT));
    assert!(waiting.agent.is_some(), "{waiting:?}");
    // The app's row reads "<agent> · Waiting for a free slot".
    let told = watcher.last(&second_id).unwrap();
    assert_eq!(told.activity.as_deref(), Some(WAITING_FOR_SLOT));
    assert!(told.agent.is_some(), "{told:?}");
    assert_eq!(started_task_ids(&tool).await, vec![first_id.clone()]);
    let check = tool.handle_check(&second_id, chat).await.unwrap();
    assert!(check.contains("is queued"), "{check}");
    // Running and queued alike, each is on record until its process ends.
    let mut expected: Vec<String> = [&first_id, &second_id, &third_id]
        .iter()
        .map(|id| cli_run_id(id))
        .collect();
    expected.sort();
    assert_eq!(on_record(&tool, chat).await, expected);
    let listed = tool.handle_list().await.unwrap();
    assert!(
        listed.contains("queued, waiting for a free slot"),
        "{listed}"
    );

    // Each ends as an immediate launch does, the queued ones in order.
    for id in [&first_id, &second_id, &third_id] {
        let run = watcher.ended(id, 20).await;
        assert_eq!(
            run.view.phase,
            crate::run_state::RunPhase::Succeeded,
            "{run:?}"
        );
        assert_eq!(run.activity, None, "{run:?}");
    }
    assert_eq!(
        started_task_ids(&tool).await,
        vec![first_id, second_id.clone(), third_id]
    );
    assert!(on_record(&tool, chat).await.is_empty());
    let result = tool.handle_check(&second_id, chat).await.unwrap();
    assert!(result.contains("second-done"), "{result}");
}

/// A queued run the owner stops before it starts never runs: it ends
/// Stopped by the owner, its working-dir claim (taken when it was accepted,
/// so a run that would change the same files is refused up front) is
/// released, and its place in line is free.
#[tokio::test]
async fn a_queued_run_stopped_before_it_starts_never_runs_and_frees_its_claim() {
    use crate::tools::traits::ToolSemantics as _;
    let (tool, runs, watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-queue-stop";
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().to_string_lossy().to_string();
    let normalized = CliAgentTool::normalize_working_dir(&tool.backend, &dir_path)
        .await
        .unwrap();

    let first = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#,
    )
    .await;
    let first_id = extract_task_id_from_background_message(&first);
    let queued = launch(
        &tool,
        chat,
        &format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"touch never-ran","working_dir":"{dir_path}","async_mode":true}}"#
        ),
    )
    .await;
    assert!(queued.contains("is queued"), "{queued}");
    let queued_id = extract_task_id_from_background_message(&queued);
    assert!(lock_claims(&tool.working_dir_claims).contains_key(&normalized));

    // A run that would change the same files is refused up front.
    let conflicting = launch(
        &tool,
        chat,
        &format!(
            r#"{{"action":"run","tool":"bash-agent","prompt":"echo second","working_dir":"{dir_path}","async_mode":true}}"#
        ),
    )
    .await;
    assert!(
        conflicting.contains("BLOCKED") && conflicting.contains(&queued_id),
        "{conflicting}"
    );

    // The owner's Stop, as the app sends it.
    let jobs = tool.background_jobs(chat).await;
    assert!(jobs.iter().any(|job| job.key == queued_id), "{jobs:?}");
    runs.owner_stopping(&cli_run_id(&queued_id));
    assert!(tool.stop_background_job(&queued_id).await);

    let run = watcher.ended(&queued_id, 5).await;
    assert_eq!(
        run.view.phase,
        crate::run_state::RunPhase::Stopped,
        "{run:?}"
    );
    assert!(run.view.stopped_by_owner, "{run:?}");
    assert!(!lock_claims(&tool.working_dir_claims).contains_key(&normalized));
    assert!(
        !on_record(&tool, chat)
            .await
            .contains(&cli_run_id(&queued_id))
    );

    // Its place is free: once the running one ends, the next launch starts
    // at once, and the stopped one never does.
    tool.handle_cancel(&first_id).await.unwrap();
    let next = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"echo next","async_mode":true}"#,
    )
    .await;
    assert!(next.contains("started in background"), "{next}");
    let next_id = extract_task_id_from_background_message(&next);
    assert_eq!(
        watcher.ended(&next_id, 20).await.view.phase,
        crate::run_state::RunPhase::Succeeded
    );
    assert_eq!(started_task_ids(&tool).await, vec![first_id, next_id]);
    assert!(!dir.path().join("never-ran").exists());
}

/// The coordinator cancels a queued run with `action="cancel"`: it is told
/// the run never started, and the run ends Stopped (not by the owner).
#[tokio::test]
async fn cancelling_a_queued_run_says_it_never_started() {
    let (tool, _runs, watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-queue-cancel";
    launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 30","async_mode":true}"#,
    )
    .await;
    let queued = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"echo never","async_mode":true}"#,
    )
    .await;
    let queued_id = extract_task_id_from_background_message(&queued);

    let cancelled = tool
        .call(&format!(r#"{{"action":"cancel","task_id":"{queued_id}"}}"#))
        .await
        .unwrap();
    assert!(cancelled.contains("before it started"), "{cancelled}");
    let run = watcher.ended(&queued_id, 5).await;
    assert_eq!(
        run.view.phase,
        crate::run_state::RunPhase::Stopped,
        "{run:?}"
    );
    assert!(!run.view.stopped_by_owner, "{run:?}");
    assert_eq!(started_task_ids(&tool).await.len(), 1);

    tool.handle_cancel_all(chat).await.unwrap();
}

/// A queued run's time is counted from when its process started, not from
/// when it was queued.
#[tokio::test]
async fn a_started_queued_run_counts_its_time_from_its_start() {
    let (tool, _runs, _watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-queue-time";
    launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 1","async_mode":true}"#,
    )
    .await;
    let queued = launch(
        &tool,
        chat,
        r#"{"action":"run","tool":"bash-agent","prompt":"sleep 5","async_mode":true}"#,
    )
    .await;
    let queued_id = extract_task_id_from_background_message(&queued);
    assert_eq!(
        tool.running.lock().await[&queued_id].process.started_at(),
        None
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let (accepted, since) = loop {
        {
            let running = tool.running.lock().await;
            let agent = &running[&queued_id];
            if agent.process.started_at().is_some() {
                break (agent.accepted_at, agent.running_since());
            }
        }
        assert!(tokio::time::Instant::now() < deadline, "it never started");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(
        since.duration_since(accepted) >= Duration::from_millis(500),
        "its time starts when it started, after its wait"
    );
    tool.handle_cancel_all(chat).await.unwrap();
}

/// Set `chat` to a coding agent: each owner message goes to it directly.
async fn set_direct(tool: &CliAgentTool, chat: &str) {
    crate::session_model::set(
        tool.state.as_ref(),
        chat,
        &crate::session_model::SessionModelSetting {
            agent: Some("bash-agent".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
}

/// A launch in `dir` of `prompt`, in the background.
fn in_dir(dir: &str, prompt: &str) -> String {
    format!(
        r#"{{"action":"run","tool":"bash-agent","prompt":"{prompt}","working_dir":"{dir}","async_mode":true}}"#
    )
}

/// In a conversation set to a coding agent, a message sent while another
/// run works in the same folder waits for it: accepted at once with its
/// record reading which agent it waits on, no process started, and it
/// starts by itself once the first run ends.
#[tokio::test]
async fn a_direct_agent_launch_in_a_taken_folder_waits_and_starts_when_it_frees() {
    let (tool, runs, watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-direct-folder";
    set_direct(&tool, chat).await;
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().to_string_lossy().to_string();

    let first = launch(&tool, chat, &in_dir(&dir_path, "sleep 2; echo first-done")).await;
    assert!(first.contains("started in background"), "{first}");
    let first_id = extract_task_id_from_background_message(&first);

    let second = launch(&tool, chat, &in_dir(&dir_path, "echo second-done")).await;
    assert!(
        !second.contains("BLOCKED")
            && second.contains("is queued (task_id=")
            && second.contains("waits for it to finish")
            && second.contains("starts by itself"),
        "{second}"
    );
    let second_id = extract_task_id_from_background_message(&second);

    let holder = tool.agent_named("bash-agent").unwrap().name().to_string();
    let waiting = runs.get(&cli_run_id(&second_id)).unwrap();
    assert_eq!(waiting.phase, crate::run_state::RunPhase::Running);
    assert_eq!(
        waiting.activity.as_deref(),
        Some(format!("Waiting for {holder} to finish in this folder").as_str())
    );
    assert_eq!(started_task_ids(&tool).await, vec![first_id.clone()]);
    let listed = tool.handle_list().await.unwrap();
    assert!(
        listed.contains(&format!("waiting for {holder} to finish in its folder")),
        "{listed}"
    );

    for id in [&first_id, &second_id] {
        let run = watcher.ended(id, 20).await;
        assert_eq!(
            run.view.phase,
            crate::run_state::RunPhase::Succeeded,
            "{run:?}"
        );
        assert_eq!(run.activity, None, "{run:?}");
    }
    assert_eq!(
        started_task_ids(&tool).await,
        vec![first_id, second_id.clone()]
    );
    let result = tool.handle_check(&second_id, chat).await.unwrap();
    assert!(result.contains("second-done"), "{result}");
}

/// A direct-agent launch stopped while it waits for its folder never
/// starts, and holds no claim on the folder.
#[tokio::test]
async fn a_direct_agent_launch_stopped_while_it_waits_for_its_folder_never_runs() {
    let (tool, runs, watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-direct-folder-stop";
    set_direct(&tool, chat).await;
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().to_string_lossy().to_string();

    let first = launch(&tool, chat, &in_dir(&dir_path, "sleep 30")).await;
    let first_id = extract_task_id_from_background_message(&first);
    let waiting = launch(&tool, chat, &in_dir(&dir_path, "touch never-ran")).await;
    assert!(waiting.contains("is queued"), "{waiting}");
    let waiting_id = extract_task_id_from_background_message(&waiting);

    runs.owner_stopping(&cli_run_id(&waiting_id));
    assert!(tool.stop_background_job(&waiting_id).await);
    let run = watcher.ended(&waiting_id, 5).await;
    assert_eq!(
        run.view.phase,
        crate::run_state::RunPhase::Stopped,
        "{run:?}"
    );
    assert!(
        lock_claims(&tool.working_dir_claims)
            .values()
            .flatten()
            .all(|claim| claim.task_id != waiting_id)
    );

    tool.handle_cancel(&first_id).await.unwrap();
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(started_task_ids(&tool).await, vec![first_id]);
    assert!(!dir.path().join("never-ran").exists());
}

/// Outside a conversation set to a coding agent (the model delegating), a
/// launch in a folder another run holds is still refused.
#[tokio::test]
async fn a_delegated_launch_in_a_taken_folder_is_still_refused() {
    let (tool, _runs, _watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-delegated-folder";
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().to_string_lossy().to_string();

    let first = launch(&tool, chat, &in_dir(&dir_path, "sleep 30")).await;
    let first_id = extract_task_id_from_background_message(&first);
    let second = launch(&tool, chat, &in_dir(&dir_path, "echo second")).await;
    assert!(
        second.contains("BLOCKED") && second.contains(&first_id),
        "{second}"
    );
    assert_eq!(started_task_ids(&tool).await, vec![first_id.clone()]);
    tool.handle_cancel(&first_id).await.unwrap();
}

/// A folder waiter in a subfolder must not jump ahead of an earlier waiter
/// targeting the enclosing parent folder, whereas an earlier waiter with a
/// disjoint `write_paths` scope does not block a later waiter.
#[tokio::test]
async fn folder_waiter_blocks_subfolder_waiter_but_allows_disjoint_write_paths() {
    let (tool, _runs, _watcher, _db) = one_slot_tool().await;
    let claims = tool.working_dir_claims.clone();
    let notify = tool.claims_changed.clone();

    // Seed an earlier waiter waiting on the parent folder `/w/synthetic`.
    let earlier_shared = RunShared::default();
    *earlier_shared.folder_wait.lock().unwrap() = Some("bash-agent".to_string());
    let earlier_call = RunCall {
        task_id: "early-parent".to_string(),
        tool_name: "bash-agent".to_string(),
        session_id: "app:synthetic-wait-order".to_string(),
        goal_id: None,
        delegated_task_id: None,
        label: "parent task".to_string(),
        short_summary: "parent".to_string(),
        async_mode: true,
        status_tx: None,
        grant_session: "app:synthetic-wait-order".to_string(),
        new_folders: Vec::new(),
        began: chrono::Utc::now(),
        owner_follow_up: false,
        direct_agent: true,
        fresh_session: false,
        is_peer_run: false,
        continues_session: false,
        peer_binding: None,
    };
    let earlier_at = Instant::now() - Duration::from_secs(5);
    let later_shared = RunShared::default();
    *later_shared.folder_wait.lock().unwrap() = Some("bash-agent".to_string());
    let later_at = Instant::now();
    {
        let mut table = tool.running.lock().await;
        table.insert(
            "early-parent".to_string(),
            tracked(
                &earlier_call,
                &earlier_shared,
                earlier_at,
                Some("/w/synthetic".to_string()),
                WriteScope::Whole,
                4096,
            ),
        );
        table.insert(
            "later-sub".to_string(),
            tracked(
                &RunCall {
                    task_id: "later-sub".to_string(),
                    ..earlier_call.clone()
                },
                &later_shared,
                later_at,
                Some("/w/synthetic/src".to_string()),
                WriteScope::Whole,
                4096,
            ),
        );
        table.insert(
            "later-disjoint".to_string(),
            tracked(
                &RunCall {
                    task_id: "later-disjoint".to_string(),
                    ..earlier_call.clone()
                },
                &later_shared,
                later_at,
                Some("/w/synthetic".to_string()),
                WriteScope::Paths(vec!["/w/synthetic/b.md".to_string()]),
                4096,
            ),
        );
    }

    // Later waiter targeting a subfolder `/w/synthetic/src` collides with
    // `/w/synthetic` (Whole), so it must wait.
    let sub_wait = FolderWait {
        dir: "/w/synthetic/src".to_string(),
        writes: WriteScope::Whole,
        prompt_summary: "sub".to_string(),
        behind: "bash-agent".to_string(),
    };
    assert!(
        matches!(
            try_take_folder(
                &tool.running,
                &claims,
                &notify,
                "later-sub",
                "bash-agent",
                &sub_wait,
            )
            .await,
            FolderTurn::Waits(_)
        ),
        "subfolder waiter should wait behind earlier parent-folder waiter"
    );

    // Replace the earlier waiter with a narrow `write_paths` scope on `a.md`.
    tool.running.lock().await.insert(
        "early-parent".to_string(),
        tracked(
            &earlier_call,
            &earlier_shared,
            earlier_at,
            Some("/w/synthetic".to_string()),
            WriteScope::Paths(vec!["/w/synthetic/a.md".to_string()]),
            4096,
        ),
    );

    // A later waiter targeting disjoint `/w/synthetic/b.md` does NOT collide
    // and can take the folder immediately.
    let disjoint_wait = FolderWait {
        dir: "/w/synthetic".to_string(),
        writes: WriteScope::Paths(vec!["/w/synthetic/b.md".to_string()]),
        prompt_summary: "disjoint".to_string(),
        behind: "bash-agent".to_string(),
    };
    let turn = try_take_folder(
        &tool.running,
        &claims,
        &notify,
        "later-disjoint",
        "bash-agent",
        &disjoint_wait,
    )
    .await;
    assert!(
        matches!(turn, FolderTurn::Taken(_)),
        "disjoint write_paths waiter should not be blocked by earlier narrow waiter"
    );
}

/// When a direct-agent follow-up wakes from waiting on its folder, it
/// refreshes and resumes the prior run's session even if `RunState::ended_at`
/// has not yet been stamped, as long as durable completion (`output_summary`)
/// is recorded.
#[tokio::test]
async fn folder_queued_direct_run_refreshes_session_before_ended_at_is_stamped() {
    let (tool, runs, _watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-direct-refresh";
    set_direct(&tool, chat).await;
    let dir = tempfile::TempDir::new().unwrap();
    let dir_path = dir.path().to_string_lossy().to_string();

    // Register a fake `claude` tool so `prepare` supports `resume`.
    {
        let mut tools = tool.tools.write().unwrap();
        tools.insert(
            "claude".to_string(),
            CliToolEntry {
                command: "claude".to_string(),
                args: vec!["-p".to_string()],
                description: "Synthetic Claude".to_string(),
                timeout: Duration::from_secs(10),
                max_output_chars: 10_000,
                is_dynamic: false,
            },
        );
    }

    // Record an earlier kept run in SQLite whose durable completion is saved,
    // while `RunState` in memory still has `ended_at == None`.
    let earlier_task = "first123";
    let earlier_run = cli_run_id(earlier_task);
    let inv_id = tool
        .state
        .log_cli_agent_start(
            Some(earlier_task),
            chat,
            "claude",
            "first turn",
            Some(&dir_path),
        )
        .await
        .unwrap();
    tool.state
        .note_cli_agent_run(inv_id, &earlier_run, Some("checked"))
        .await
        .unwrap();
    tool.state
        .set_cli_agent_keeps_session(inv_id, true)
        .await
        .unwrap();
    tool.state
        .set_cli_agent_session(inv_id, "sess-synthetic-42")
        .await
        .unwrap();
    tool.state
        .log_cli_agent_complete(inv_id, Some(0), "first turn done", true, 1.0)
        .await
        .unwrap();

    // Leave `RunState::ended_at` unstamped (still Running in the registry).
    runs.begin(&earlier_run, chat, "first turn", None);
    assert!(runs.get(&earlier_run).unwrap().ended_at.is_none());

    // Prepare a fresh second launch (before the first had finished).
    let coding_agent = tool.agent_named("claude").unwrap();
    let mut prep = PreparedRun {
        command: "claude".to_string(),
        args: vec!["-p".to_string()],
        base_launch: aidaemon_coding_agents::Launch {
            args: vec!["-p".to_string()],
            start: 0,
        },
        enforce_native_access: false,
        read_paths: vec![dir_path.clone()],
        model: None,
        can_host: false,
        mode: aidaemon_coding_agents::PermissionMode::FullAccess,
        coding_agent,
        nested: None,
        detached_judge: None,
        spawn_family: None,
        timeout: Duration::from_secs(10),
        max_output: 10_000,
        workspace_mode: CliWorkspaceMode::ReadWrite,
        working_dir: Some(dir_path.clone()),
        claim_dir: Some(dir_path.clone()),
        write_scope: WriteScope::Paths(vec![dir_path.clone()]),
        write_paths: vec![dir_path.clone()],
        prompt_placement: aidaemon_coding_agents::PromptPlacement::LastArgument,
        env: Vec::new(),
        env_remove: Vec::new(),
        signed: aidaemon_coding_agents::SignedIn::default(),
        host: None,
        keep_alive: None,
        mcp_guard: None,
        keeps_session: true,
        continues: None,
        mcp_servers: Vec::new(),
        prompt: "second turn".to_string(),
        task: "second turn".to_string(),
        final_prompt: "second turn".to_string(),
        prior_context: None,
        claim: None,
        folder_wait: None,
        thread_wait: None,
    };
    let call = RunCall {
        task_id: "second456".to_string(),
        tool_name: "claude".to_string(),
        session_id: chat.to_string(),
        goal_id: None,
        delegated_task_id: None,
        label: "second turn".to_string(),
        short_summary: "second turn".to_string(),
        async_mode: true,
        status_tx: None,
        grant_session: chat.to_string(),
        new_folders: Vec::new(),
        began: chrono::Utc::now(),
        owner_follow_up: false,
        direct_agent: true,
        fresh_session: false,
        is_peer_run: false,
        continues_session: false,
        peer_binding: None,
    };

    // The peer endpoint's guard: dropping it revokes the run's MCP token.
    struct DropFlag(Arc<std::sync::atomic::AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let mcp_dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    prep.mcp_guard = Some(Box::new(DropFlag(mcp_dropped.clone())));
    // Queued while it looked like a new session: history was put first.
    let prefix = "## Prior conversation\nuser: first turn\n\n## Current request\n";
    prep.prior_context = Some(prefix.to_string());
    prep.final_prompt = format!("{prefix}second turn");

    super::super::queue::refresh_direct_session(
        &tool.run_handles(),
        &tool.running,
        &call,
        &mut prep,
    )
    .await;
    assert!(
        !mcp_dropped.load(std::sync::atomic::Ordering::SeqCst),
        "re-preparing a queued launch must keep its peer MCP endpoint alive"
    );
    let handed_off = prep.take_keep_alive();
    assert!(handed_off.is_some(), "the watcher gets the peer guard");
    drop(handed_off);
    assert!(mcp_dropped.load(std::sync::atomic::Ordering::SeqCst));

    assert_eq!(
        prep.final_prompt, "second turn",
        "a continued session already holds the history"
    );
    assert!(prep.prior_context.is_none());
    let continued = prep
        .continues
        .expect("should resume earlier kept session even before ended_at is stamped");
    assert_eq!(continued.run, earlier_run);
    assert_eq!(continued.agent_session, "sess-synthetic-42");
    assert!(
        prep.args
            .windows(2)
            .any(|w| w[0] == "--resume" && w[1] == "sess-synthetic-42"),
        "re-prepared args should resume the session: {:?}",
        prep.args
    );
    assert!(
        !prep.args.iter().any(|arg| arg == "--json-schema"),
        "direct_agent refresh must not attach the 3-sentence report schema: {:?}",
        prep.args
    );
}

/// Addressing a live running session whose host accepts mid-run notes delivers
/// the note directly to that host and records `last_note_at` on the run state.
#[tokio::test]
async fn continue_run_steers_live_host_that_accepts_mid_run_notes() {
    use std::sync::Mutex as StdMutex;

    struct SteerableHost {
        notes: Arc<StdMutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl aidaemon_coding_agents::AgentHost for SteerableHost {
        async fn on_line(&self, line: &str) -> Vec<String> {
            vec![line.to_string()]
        }
        fn answers(&self) -> aidaemon_coding_agents::HostAnswers {
            Default::default()
        }
        fn accepts_mid_run(&self) -> bool {
            true
        }
        async fn send_note(&self, note: &str) -> Result<(), String> {
            self.notes.lock().unwrap().push(note.to_string());
            Ok(())
        }
        async fn close(&self) {}
    }

    let (tool, _runs, _watcher, _db) = one_slot_tool().await;
    let chat = "app:synthetic-steer";
    let task_id = "steer123";
    let run_id = cli_run_id(task_id);
    let notes = Arc::new(StdMutex::new(Vec::new()));
    let host: Arc<dyn aidaemon_coding_agents::AgentHost> = Arc::new(SteerableHost {
        notes: notes.clone(),
    });

    let handles = tool.run_handles();
    handles.runs.begin(&run_id, chat, "Synthetic Codex", None);
    handles.runs.set_agent(&run_id, "Codex");
    handles.runs.set_accepts_mid_run(&run_id, true);

    let shared = RunShared::default();
    shared.process.start(
        aidaemon_tools::backend::ProcessHandle::new(1, None, None),
        Some(host),
    );
    let call = RunCall {
        task_id: task_id.to_string(),
        tool_name: "bash-agent".to_string(),
        session_id: chat.to_string(),
        goal_id: None,
        delegated_task_id: None,
        label: "Synthetic Codex".to_string(),
        short_summary: "Synthetic Codex".to_string(),
        async_mode: true,
        status_tx: None,
        grant_session: chat.to_string(),
        new_folders: Vec::new(),
        began: chrono::Utc::now(),
        owner_follow_up: false,
        direct_agent: false,
        fresh_session: false,
        is_peer_run: false,
        continues_session: false,
        peer_binding: None,
    };
    tool.running.lock().await.insert(
        task_id.to_string(),
        tracked(
            &call,
            &shared,
            Instant::now(),
            None,
            WriteScope::ReadOnly,
            10_000,
        ),
    );

    let before = chrono::Utc::now();
    let call_id = "follow-up-synthetic-steer";
    crate::helper_run::mark_owner_follow_up(call_id);
    let facts = ToolCallContext {
        session_id: chat.to_string(),
        tool_call_id: Some(call_id.to_string()),
        ..Default::default()
    };
    let output = tool
        .call_with_status_under_contract(
            &json!({
                "action": "run",
                "continue_run": run_id,
                "prompt": "also check the synthetic edge case",
                "async_mode": true,
            })
            .to_string(),
            None,
            false,
            false,
            &facts,
        )
        .await
        .expect("mid-run note should succeed");

    assert!(output.contains("Added your note"), "{output}");
    assert_eq!(
        *notes.lock().unwrap(),
        vec!["also check the synthetic edge case".to_string()]
    );
    let state_after = handles.runs.get(&run_id).expect("run still tracked");
    assert!(
        state_after.last_note_at.is_some_and(|at| at >= before),
        "last_note_at should be stamped on the steered run"
    );
}
