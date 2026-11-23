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
