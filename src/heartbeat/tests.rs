use super::*;
use crate::memory::embeddings::EmbeddingService;
use crate::state::SqliteStateStore;
use crate::traits::{Goal, GoalSchedule, Mandate, MandateAuthority, Task, TaskActivity};
use std::sync::atomic::{AtomicUsize, Ordering};

fn synthetic_task(desc: &str, status: &str) -> Task {
    Task {
        id: format!("task-{desc}"),
        goal_id: "goal-1".to_string(),
        description: desc.to_string(),
        status: status.to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: String::new(),
        started_at: None,
        completed_at: None,
    }
}

#[test]
fn scheduled_run_reconciliation_terminalizes_split_brain_task_graphs() {
    let mut run = crate::traits::GoalRun::new("goal-1", "default", "scheduled");
    run.status = "running".to_string();

    assert_eq!(
        scheduled_run_reconciliation_status(
            &run,
            &[
                synthetic_task("root", "completed"),
                synthetic_task("child", "blocked")
            ],
        ),
        Some("failed")
    );
    assert_eq!(
        scheduled_run_reconciliation_status(&run, &[synthetic_task("root", "completed")],),
        Some("completed")
    );
    assert_eq!(
        scheduled_run_reconciliation_status(
            &run,
            &[
                synthetic_task("root", "completed"),
                synthetic_task("child", "running")
            ],
        ),
        None
    );
}

#[test]
fn terminal_run_tasks_never_apply_schedule_backpressure() {
    let mut run = crate::traits::GoalRun::new("goal-1", "default", "scheduled");
    run.status = "completed".to_string();
    let mut task = synthetic_task("child", "blocked");
    task.blocker = Some("Historical unresolved child".to_string());

    assert!(!task_blocks_later_schedule_fire(&run, &task));
}

#[test]
fn only_open_manual_run_can_reactivate_stalled_pending_work() {
    let pending = synthetic_task("manual root", "pending");
    let candidates = vec![&pending];
    let mut run = crate::traits::GoalRun::new("goal-1", "default", "manual");

    assert!(stranded_manual_run_matches_pending_tasks(
        &run,
        std::slice::from_ref(&pending),
        &candidates,
    ));

    run.trigger_type = "scheduled".to_string();
    assert!(!stranded_manual_run_matches_pending_tasks(
        &run,
        std::slice::from_ref(&pending),
        &candidates,
    ));

    run.trigger_type = "manual".to_string();
    run.status = "completed".to_string();
    assert!(!stranded_manual_run_matches_pending_tasks(
        &run,
        std::slice::from_ref(&pending),
        &candidates,
    ));

    let other = synthetic_task("different task", "pending");
    run.status = "pending".to_string();
    assert!(!stranded_manual_run_matches_pending_tasks(
        &run,
        std::slice::from_ref(&other),
        &candidates,
    ));
}

#[test]
fn scheduled_blocker_only_retires_the_prior_scheduled_occurrence() {
    let blocked = synthetic_task("scheduled blocker", "blocked");
    let mut provider_blocked = blocked.clone();
    provider_blocked.blocker = Some(
        "LLM error: Codex stream failed: Our servers are currently overloaded. Please try again later."
            .to_string(),
    );
    let pending = synthetic_task("still in flight", "pending");
    let mut scheduled_run = crate::traits::GoalRun::new("goal-1", "default", "scheduled");
    scheduled_run.status = "blocked".to_string();
    let mut manual_run = crate::traits::GoalRun::new("goal-1", "default", "manual");
    manual_run.status = "blocked".to_string();

    assert!(!task_blocks_later_schedule_fire(&scheduled_run, &blocked,));
    // A blocked manual or recovery occurrence is equally terminal for
    // backpressure: it must never suppress the recurring schedule forever.
    assert!(!task_blocks_later_schedule_fire(&manual_run, &blocked));
    let mut recovery_run = crate::traits::GoalRun::new("goal-1", "default", "recovery");
    recovery_run.status = "blocked".to_string();
    assert!(!task_blocks_later_schedule_fire(&recovery_run, &blocked));
    assert!(!task_blocks_later_schedule_fire(
        &scheduled_run,
        &provider_blocked,
    ));
    assert!(!task_blocks_later_schedule_fire(
        &manual_run,
        &provider_blocked,
    ));
    // A still-running manual occurrence keeps its backpressure.
    manual_run.status = "running".to_string();
    assert!(task_blocks_later_schedule_fire(&manual_run, &blocked));
    assert_eq!(
        scheduled_run_reconciliation_status(
            &manual_run,
            &[synthetic_task("manual root", "completed")]
        ),
        Some("completed")
    );
    assert!(!task_blocks_later_schedule_fire(&scheduled_run, &pending,));

    scheduled_run.status = "running".to_string();
    assert!(task_blocks_later_schedule_fire(&scheduled_run, &blocked));
    assert!(task_blocks_later_schedule_fire(&scheduled_run, &pending));
    assert!(task_is_terminal_schedule_failure(&blocked));
}

fn due_mandate_controller(session_id: &str) -> (Goal, Mandate) {
    let goal = Goal::new_continuous(
        "Steward an account autonomously",
        session_id,
        Some(10_000),
        Some(100_000),
    );
    let mut mandate = Mandate::new(
        &goal.id,
        None,
        "Maintain a useful and authentic account presence",
        session_id,
        MandateAuthority::default(),
        60,
        3_600,
        300,
    );
    mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
    (goal, mandate)
}

#[tokio::test]
async fn queued_notices_reach_owner_history_and_timeouts_stay_answerable() {
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::{DialogueStateStore, NotificationEntry, QuestionKind};

    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    let session_id = "telegram:test_bot:synthetic-owner";
    let state: Arc<dyn StateStore> = harness.state.clone();
    let (goal, mandate) = due_mandate_controller(session_id);
    state
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();
    let agent = Arc::new(harness.agent);
    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.set_agent(Arc::downgrade(&agent));

    // A routine queued notice: visible to the agent, but not a question.
    let stalled = NotificationEntry::new(
        &goal.id,
        session_id,
        "stalled",
        "Your weekly digest goal has not made progress in 3 days.",
    );
    coordinator
        .record_delivered_notification(&stalled, &stalled.message)
        .await;
    assert!(harness
        .state
        .get_dialogue_state(session_id)
        .await
        .unwrap()
        .and_then(|state| state.open_question)
        .is_none());

    // The ASK-timeout notice says the question can still be answered, so
    // the owner's next reply must be bound to the (already resumed) mandate.
    let timeout = NotificationEntry::new(
        &goal.id,
        session_id,
        "mandate_ask_timeout",
        "Your automation kept going without your answer. It asked: 'May I post twice a day?'",
    );
    coordinator
        .record_delivered_notification(&timeout, &timeout.message)
        .await;

    let history = state.get_history(session_id, 10).await.unwrap();
    let recorded = history
        .iter()
        .filter_map(|message| message.content.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(recorded.contains("has not made progress in 3 days"));
    assert!(recorded.contains("May I post twice a day?"));
    let question = harness
        .state
        .get_dialogue_state(session_id)
        .await
        .unwrap()
        .and_then(|state| state.open_question)
        .expect("timeout notice binds the owner's next reply");
    assert_eq!(question.kind, QuestionKind::MandateInput);
    assert_eq!(question.mandate_id.as_deref(), Some(mandate.id.as_str()));
}

#[tokio::test]
async fn due_mandate_creates_a_distinct_run_without_a_goal_schedule() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap(),
    );
    let (goal, mandate) = due_mandate_controller("owner-session");
    state
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.check_due_mandates().await;

    let run = state
        .get_current_goal_run(&goal.id)
        .await
        .unwrap()
        .expect("mandate review should create an open run");
    assert_eq!(run.trigger_type, "mandate");
    assert!(run.schedule_id.is_none());
    let tasks = state.get_tasks_for_goal_run(&run.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(run.root_task_id.as_deref(), Some(tasks[0].id.as_str()));
    assert!(tasks[0]
        .description
        .starts_with("Run one bounded autonomous review"));
    assert!(tasks[0]
        .description
        .contains("isolated built-in mandate protocol"));
    assert!(!tasks[0].description.contains(&mandate.objective));
    assert!(!tasks[0].description.contains("allowed tools"));
    let context: serde_json::Value =
        serde_json::from_str(tasks[0].context.as_deref().expect("minimal fence context")).unwrap();
    assert_eq!(context["mandate_id"], mandate.id);
    assert_eq!(context["mandate_version"], mandate.version);
    assert_eq!(context["provenance"], "runtime_mandate_fence_only");
    assert_eq!(context.as_object().unwrap().len(), 3);
    assert!(state
        .get_schedules_for_goal(&goal.id)
        .await
        .unwrap()
        .is_empty());
    assert!(state
        .get_scheduled_run_state(&goal.id)
        .await
        .unwrap()
        .is_none());
    assert!(state
        .get_mandate(&mandate.id)
        .await
        .unwrap()
        .unwrap()
        .review_lease_token
        .is_some());
}

#[tokio::test]
async fn mandate_budget_deferral_waits_for_the_next_utc_reset() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap(),
    );
    let (mut goal, mandate) = due_mandate_controller("owner-session");
    goal.tokens_used_today = 95_000;
    goal.tokens_used_day = chrono::Utc::now().date_naive().to_string();
    state
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let before = chrono::Utc::now();
    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.check_due_mandates().await;

    assert!(state
        .get_current_goal_run(&goal.id)
        .await
        .unwrap()
        .is_none());
    let deferred = state.get_mandate(&mandate.id).await.unwrap().unwrap();
    assert!(deferred.review_lease_token.is_none());
    let retry_at = chrono::DateTime::parse_from_rfc3339(&deferred.next_review_at)
        .unwrap()
        .with_timezone(&chrono::Utc);
    let expected_midnight = chrono::DateTime::<Utc>::from_naive_utc_and_offset(
        before
            .date_naive()
            .succ_opt()
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
        Utc,
    );
    assert_eq!(retry_at, expected_midnight);
}

#[tokio::test]
async fn open_mandate_run_prevents_a_duplicate_review_after_lease_release() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap(),
    );
    let (goal, mandate) = due_mandate_controller("owner-session");
    state
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.check_due_mandates().await;
    let first_run = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
    let claimed = state.get_mandate(&mandate.id).await.unwrap().unwrap();
    let lease_token = claimed.review_lease_token.as_deref().unwrap();
    assert!(state
        .release_mandate_review_lease(
            &mandate.id,
            lease_token,
            &(chrono::Utc::now() - chrono::Duration::seconds(1)).to_rfc3339(),
        )
        .await
        .unwrap());

    coordinator.check_due_mandates().await;

    let still_open = state.get_current_goal_run(&goal.id).await.unwrap().unwrap();
    assert_eq!(still_open.id, first_run.id);
    assert_eq!(state.get_goal_runs(&goal.id).await.unwrap().len(), 1);
    assert_eq!(
        state
            .get_tasks_for_goal_run(&first_run.id)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn generic_orphan_dispatch_leaves_non_root_mandate_tasks_untouched() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            Arc::new(EmbeddingService::new().unwrap()),
        )
        .await
        .unwrap(),
    );
    let (goal, mandate) = due_mandate_controller("owner-session");
    state
        .create_mandate_controller(&goal, &mandate)
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.check_due_mandates().await;
    let run = state
        .get_current_goal_run(&goal.id)
        .await
        .unwrap()
        .expect("mandate run");
    let root_id = run.root_task_id.clone().expect("mandate root");
    let mut root = state.get_task(&root_id).await.unwrap().expect("root task");
    root.status = "running".to_string();
    root.started_at = Some((chrono::Utc::now() - chrono::Duration::minutes(20)).to_rfc3339());
    state.update_task(&root).await.unwrap();

    let non_root = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Model-authored exact action task".to_string(),
        status: "pending".to_string(),
        priority: "high".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 0,
        created_at: (chrono::Utc::now() - chrono::Duration::minutes(5)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&non_root).await.unwrap();
    assert!(state
        .get_tasks_for_goal_run(&run.id)
        .await
        .unwrap()
        .iter()
        .any(|task| task.id == non_root.id));

    coordinator.dispatch_pending_tasks().await;

    let after = state
        .get_task(&non_root.id)
        .await
        .unwrap()
        .expect("non-root task remains");
    assert_eq!(after.status, "pending");
    assert!(after.agent_id.is_none());
    assert!(after.started_at.is_none());
    assert!(state
        .get_current_task_attempt(&non_root.id)
        .await
        .unwrap()
        .is_none());
    assert!(
        state
            .get_pending_notifications(10)
            .await
            .unwrap()
            .is_empty(),
        "generic orphan recovery must not announce or claim mandate child work"
    );
}

#[test]
fn goal_completion_header_is_honest_about_incomplete_tasks() {
    // Live repro (goal df0925b4, 2026-07-02): the watchdog killed the task
    // lead mid-run, no partial_success metadata was written, and the user
    // received "Goal completed:" over a summary that itself said "2/4
    // tasks completed" — two steps silently missing. The header must be
    // derived from the actual task rows, not from metadata a dead lead
    // never wrote.
    let tasks = vec![
        synthetic_task("check disk space", "completed"),
        synthetic_task("count rs files", "completed"),
        synthetic_task("find 3 largest files", "pending"),
        synthetic_task("write summary report", "pending"),
    ];
    let header = goal_completion_header(&tasks);
    assert!(
        header.starts_with("Goal partially completed (2/4 tasks)"),
        "got: {header}"
    );
    assert!(header.contains("find 3 largest files"));
    assert!(header.contains("write summary report"));

    // All tasks done → plain completion header.
    let done = vec![
        synthetic_task("a", "completed"),
        synthetic_task("b", "completed"),
    ];
    assert_eq!(goal_completion_header(&done), "Goal completed:");

    // No task rows at all (goal completed without decomposition) → plain.
    assert_eq!(goal_completion_header(&[]), "Goal completed:");
}

#[test]
fn daily_budget_only_blocks_on_current_day_usage() {
    let today = "2026-06-28";
    // Over budget, usage recorded today -> exhausted.
    assert!(daily_budget_exhausted(Some(500_000), 964_666, today, today));
    // Over budget, but usage is from a PRIOR day (reset hasn't run) -> NOT
    // exhausted, so the goal can fire and reset instead of deadlocking.
    assert!(!daily_budget_exhausted(
        Some(500_000),
        964_666,
        "2026-06-27",
        today
    ));
    // Under budget today -> not exhausted.
    assert!(!daily_budget_exhausted(
        Some(500_000),
        100_000,
        today,
        today
    ));
    // No daily budget configured -> never exhausted.
    assert!(!daily_budget_exhausted(None, 999_999, today, today));

    assert!(!daily_budget_has_run_capacity(
        Some(1_000_000),
        Some(400_000),
        798_965,
        today,
        today,
    ));
    assert!(daily_budget_has_run_capacity(
        Some(1_000_000),
        Some(400_000),
        598_965,
        today,
        today,
    ));
    assert!(daily_budget_has_run_capacity(
        Some(1_000_000),
        Some(400_000),
        999_999,
        "2026-06-27",
        today,
    ));
}

#[test]
fn mandate_sqlite_contention_uses_a_short_expiry_bounded_retry() {
    assert!(is_sqlite_busy_error(&anyhow::anyhow!(
        "error returned from database: (code: 5) database is locked"
    )));
    assert!(!is_sqlite_busy_error(&anyhow::anyhow!(
        "controller goal is missing"
    )));

    let now = chrono::DateTime::parse_from_rfc3339("2026-08-04T05:00:00Z")
        .unwrap()
        .with_timezone(&Utc);
    let goal = crate::traits::Goal::new_continuous("retry", "owner", None, None);
    let mut mandate = Mandate::new(
        &goal.id,
        None,
        "retry transient contention",
        "owner",
        crate::traits::MandateAuthority::default(),
        1_800,
        86_400,
        1_800,
    );
    mandate.expires_at = Some("2026-08-04T05:00:20Z".to_string());
    assert_eq!(
        mandate_transient_retry_at(&mandate, now),
        "2026-08-04T05:00:20+00:00"
    );
}

async fn test_state_store() -> Arc<dyn StateStore> {
    // Persist the temp DB file for the test process's lifetime. If the
    // NamedTempFile were dropped here, the path would be unlinked and a
    // later (lazily-opened) pool connection would re-create an empty
    // database — surfacing as "no such table" under parallel test load on
    // CI. keep() leaks the file, which is fine for short-lived test runs.
    let (_db_file, db_path) = tempfile::NamedTempFile::new().unwrap().keep().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    Arc::new(
        SqliteStateStore::new(db_path.to_str().unwrap(), 100, None, embedding_service)
            .await
            .unwrap(),
    )
}

#[tokio::test]
async fn coordinator_defaults_task_inactivity_timeout_to_300() {
    let (_tx, rx) = mpsc::channel::<()>(1);
    let state = test_state_store().await;
    let hb = HeartbeatCoordinator::new(state, 5, 2, rx, None, None, None);
    assert_eq!(hb.task_inactivity_timeout_secs, 300);
}

#[tokio::test]
async fn coordinator_set_task_inactivity_timeout_updates_field() {
    let (_tx, rx) = mpsc::channel::<()>(1);
    let state = test_state_store().await;
    let mut hb = HeartbeatCoordinator::new(state, 5, 2, rx, None, None, None);
    hb.set_task_inactivity_timeout(777);
    assert_eq!(hb.task_inactivity_timeout_secs, 777);
}

#[tokio::test]
async fn test_new_continuous_goal() {
    let goal = Goal::new_continuous("Test continuous goal", "system", Some(5000), Some(20000));
    assert_eq!(goal.domain, "orchestration");
    assert_eq!(goal.goal_type, "continuous");
    assert_eq!(goal.status, "active");
    assert_eq!(goal.priority, "low");
    assert_eq!(goal.budget_per_check, Some(5000));
    assert_eq!(goal.budget_daily, Some(20000));
    assert_eq!(goal.tokens_used_today, 0);
    assert!(goal.last_useful_action.is_none());
    assert_eq!(goal.session_id, "system");
}

#[tokio::test]
async fn test_heartbeat_job_fires() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let (wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let mut coordinator = HeartbeatCoordinator::new(state, 1, 3, wake_rx, None, None, None);

    coordinator.register_job("test_job", Duration::from_secs(0), move || {
        let c = counter_clone.clone();
        async move {
            c.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    });

    let shutdown = CancellationToken::new();
    let host = tokio::spawn(coordinator.into_host().run(shutdown.clone()));
    tokio::time::sleep(Duration::from_millis(100)).await;
    shutdown.cancel();
    host.await.unwrap();

    assert!(
        counter.load(Ordering::SeqCst) >= 1,
        "Job should have fired at least once"
    );

    drop(wake_tx); // Keep sender alive until here
}

async fn reminder_test_setup() -> (
    Arc<dyn StateStore>,
    Arc<crate::channels::ChannelHub>,
    Arc<crate::testing::TestChannel>,
    HeartbeatCoordinator,
    tempfile::NamedTempFile,
) {
    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );
    let channel = Arc::new(crate::testing::TestChannel::new());
    let session_map: crate::channels::SessionMap = Arc::new(tokio::sync::RwLock::new(
        HashMap::from([("test_session".to_string(), "test".to_string())]),
    ));
    let hub = Arc::new(crate::channels::ChannelHub::new(
        vec![channel.clone() as Arc<dyn crate::traits::HubChannel>],
        session_map,
    ));
    let outbound: Arc<dyn OutboundRouter> = hub.clone();
    let coordinator = HeartbeatCoordinator::new(
        state.clone(),
        1,
        3,
        wake_rx,
        Some(Arc::downgrade(&outbound)),
        None,
        None,
    );
    (state, hub, channel, coordinator, db_file)
}

fn due_schedule(goal_id: &str, cron_expr: &str, is_one_shot: bool) -> GoalSchedule {
    let now = chrono::Utc::now().to_rfc3339();
    GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal_id.to_string(),
        cron_expr: cron_expr.to_string(),
        tz: "local".to_string(),
        original_schedule: None,
        fire_policy: "coalesce".to_string(),
        is_one_shot,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.clone(),
        created_at: now.clone(),
        updated_at: now,
    }
}

#[tokio::test]
async fn test_reminder_fast_path_one_shot_delivers_and_completes() {
    let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

    let mut goal = Goal::new_deferred_finite("Remind me to call my daughter", "test_session");
    goal.status = "active".to_string();
    state.create_goal(&goal).await.unwrap();
    let schedule = due_schedule(&goal.id, "46 13 11 6 *", true);
    state.create_goal_schedule(&schedule).await.unwrap();

    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();

    // The reminder itself is the only message — no "Running scheduled
    // task", no progress updates.
    let msgs = channel.messages_for("test_session").await;
    assert_eq!(msgs, vec!["⏰ Reminder: call your daughter".to_string()]);

    // Goal completed with notified_at set (suppresses the generic
    // "Goal completed" notification) and the one-shot schedule removed.
    let g = state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(g.status, "completed");
    assert!(g.notified_at.is_some());
    assert!(state
        .get_schedules_for_goal(&goal.id)
        .await
        .unwrap()
        .is_empty());

    // A completed task records the run; nothing is left pending for the
    // task-lead pipeline.
    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].status, "completed");
}

/// The typed reminder message decides the fast path, whatever wording
/// the model used for the description.
#[tokio::test]
async fn test_typed_reminder_message_uses_fast_path() {
    let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

    let mut goal =
        Goal::new_deferred_finite("Remind Alice Rivera to check the oven", "test_session");
    goal.status = "active".to_string();
    goal.context = Some(crate::reminders::context_for_message("Check the oven."));
    state.create_goal(&goal).await.unwrap();
    let schedule = due_schedule(&goal.id, "46 13 11 6 *", true);
    state.create_goal_schedule(&schedule).await.unwrap();

    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();

    let msgs = channel.messages_for("test_session").await;
    assert_eq!(msgs, vec!["⏰ Reminder: Check the oven.".to_string()]);
    let g = state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(g.status, "completed");
}

#[tokio::test]
async fn test_reminder_fast_path_recurring_keeps_goal_active() {
    let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

    let mut goal = Goal::new_continuous("Remind me to take my meds", "test_session", None, None);
    goal.status = "active".to_string();
    state.create_goal(&goal).await.unwrap();
    let schedule = due_schedule(&goal.id, "0 9 * * *", false);
    state.create_goal_schedule(&schedule).await.unwrap();

    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();

    let msgs = channel.messages_for("test_session").await;
    assert_eq!(msgs, vec!["⏰ Reminder: take your meds".to_string()]);

    // Recurring reminders stay active with their schedule intact.
    let g = state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(g.status, "active");
    assert_eq!(
        state.get_schedules_for_goal(&goal.id).await.unwrap().len(),
        1
    );
}

#[tokio::test]
async fn test_non_reminder_schedule_creates_pending_task() {
    let (state, _hub, channel, coordinator, _db_file) = reminder_test_setup().await;

    let mut goal = Goal::new_deferred_finite("Check the deploy status", "test_session");
    goal.status = "active".to_string();
    state.create_goal(&goal).await.unwrap();
    let schedule = due_schedule(&goal.id, "46 13 11 6 *", true);
    state.create_goal_schedule(&schedule).await.unwrap();

    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();

    // Non-reminder goals keep the normal pipeline: a task is created and
    // the user is told the scheduled task is running.
    let msgs = channel.messages_for("test_session").await;
    let local_hour = chrono::Timelike::hour(&chrono::Local::now());
    if crate::traits::NotificationEntry::routine_delivery_allowed_at_local_hour(local_hour) {
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].starts_with("⏳ **Scheduled run in progress**"));
        assert!(msgs[0].contains("Check the deploy status"));
    } else {
        assert!(msgs.is_empty(), "routine progress must respect quiet hours");
    }
    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    // No agent wired in this test, so the task stays pending for the
    // orphan dispatcher.
    assert_eq!(tasks[0].status, "pending");
}

#[tokio::test]
async fn test_heartbeat_telemetry_tracks_failures_and_recovery() {
    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );
    let telemetry = Arc::new(HeartbeatTelemetry::new());
    let mut coordinator =
        HeartbeatCoordinator::new(state, 3600, 3, wake_rx, None, None, Some(telemetry.clone()));

    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_clone = attempts.clone();
    coordinator.register_job("test_job", Duration::from_secs(0), move || {
        let a = attempts_clone.clone();
        async move {
            let n = a.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                anyhow::bail!("first run fails");
            }
            Ok(())
        }
    });

    let shutdown = CancellationToken::new();
    let host = coordinator.into_host();
    let waker = host.waker();
    let host = tokio::spawn(host.run(shutdown.clone()));
    tokio::time::sleep(Duration::from_millis(100)).await;

    let first = telemetry.snapshots();
    let first_job = first
        .iter()
        .find(|j| j.name == "test_job")
        .expect("telemetry row should exist");
    assert_eq!(first_job.consecutive_failures, 1);
    assert!(first_job.last_error.is_some());
    assert!(first_job.last_run_at.is_some());

    waker.wake();
    tokio::time::sleep(Duration::from_millis(100)).await;
    shutdown.cancel();
    host.await.unwrap();

    let second = telemetry.snapshots();
    let second_job = second
        .iter()
        .find(|j| j.name == "test_job")
        .expect("telemetry row should exist");
    assert_eq!(second_job.consecutive_failures, 0);
    assert!(second_job.last_error.is_none());
    assert!(second_job.last_success_at.is_some());
}

#[tokio::test]
async fn test_due_one_shot_schedule_creates_task_and_deletes_schedule() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_finite("Send deployment reminder", "session-1");
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: true,
        is_paused: false,
        last_run_at: None,
        next_run_at: due_ts,
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1, "One execution task should be created");
    assert_eq!(tasks[0].status, "pending");
    assert!(
        tasks[0].description.starts_with("Execute scheduled goal:"),
        "Task description should indicate scheduled execution"
    );
    assert!(
        !tasks[0].idempotent,
        "scheduled orchestration roots must never auto-retry after ambiguous writes"
    );
    assert_eq!(tasks[0].max_retries, 0);

    let sched = state.get_goal_schedule(&schedule.id).await.unwrap();
    assert!(
        sched.is_none(),
        "One-shot schedules should be deleted after firing"
    );
}

/// A continuous goal idle >30 days must KEEP firing (not be silently
/// skipped forever) and must surface a one-time alert to the user.
#[tokio::test]
async fn test_idle_continuous_goal_keeps_firing_and_alerts_user() {
    let state = test_state_store().await;

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();

    let mut goal =
        Goal::new_continuous("Post daily tweets", "session-1", Some(5000), Some(500_000));
    // Idle for 40 days — well past the 30-day auto-retire threshold.
    goal.last_useful_action = Some((now - chrono::Duration::days(40)).to_rfc3339());
    state.create_goal(&goal).await.unwrap();

    let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 9 * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("0 9 * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: due_ts,
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    // Keeps firing: a task is created instead of being silently skipped.
    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(
        tasks.len(),
        1,
        "Idle continuous goal should still fire (create a task), not be skipped"
    );
    assert_eq!(tasks[0].status, "pending");

    // Surfaces a one-time alert so the idle goal is no longer silent.
    let notifs = state.get_pending_notifications(10).await.unwrap();
    let alerts: Vec<_> = notifs
        .iter()
        .filter(|n| n.notification_type == "evergreen_alert" && n.goal_id == goal.id)
        .collect();
    assert_eq!(
        alerts.len(),
        1,
        "Idle continuous goal should enqueue exactly one evergreen_alert notification"
    );
}

/// The idle alert must fire at most once per idle episode, even if the
/// goal stays idle across many ticks (e.g. an open task keeps coalescing
/// the fire so `last_useful_action` never resets).
#[tokio::test]
async fn test_idle_continuous_goal_alerts_only_once() {
    let state = test_state_store().await;

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();

    let mut goal =
        Goal::new_continuous("Post daily tweets", "session-1", Some(5000), Some(500_000));
    goal.last_useful_action = Some((now - chrono::Duration::days(40)).to_rfc3339());
    state.create_goal(&goal).await.unwrap();

    // A pre-existing open task makes the fire coalesce (back off) every
    // time, so `last_useful_action` never resets and the goal stays idle.
    let open_task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "in-flight work".to_string(),
        status: "pending".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now_ts.clone(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&open_task).await.unwrap();

    let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: due_ts,
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);

    // Fire repeatedly while the goal stays idle.
    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();
    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();
    coordinator
        .fire_due_schedule(schedule.clone())
        .await
        .unwrap();

    let alerts = state
        .get_pending_notifications(20)
        .await
        .unwrap()
        .into_iter()
        .filter(|n| n.notification_type == "evergreen_alert" && n.goal_id == goal.id)
        .count();
    assert_eq!(
        alerts, 1,
        "Idle alert must be sent at most once, not every tick"
    );

    // Coalescing is still respected: no new execution tasks were created
    // while the open task is in flight.
    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(
        tasks.len(),
        1,
        "No new task should be created while coalescing"
    );
}

#[tokio::test]
async fn test_coalesce_policy_one_shot_backs_off_when_open_task_exists() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_finite("Coalesce test", "session-1");
    state.create_goal(&goal).await.unwrap();

    // Existing open task should block coalesced firing.
    let existing_task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Existing work".to_string(),
        status: "running".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("agent-1".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: Some(chrono::Utc::now().to_rfc3339()),
        completed_at: None,
    };
    state.create_task(&existing_task).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: true,
        is_paused: false,
        last_run_at: None,
        next_run_at: due_ts,
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    // No new tasks should be created when coalescing and open work exists.
    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, existing_task.id);

    // One-shot schedules should back off instead of hot-looping.
    let updated_sched = state
        .get_goal_schedule(&schedule.id)
        .await
        .unwrap()
        .expect("schedule should still exist");
    let next = chrono::DateTime::parse_from_rfc3339(&updated_sched.next_run_at).unwrap();
    assert!(next.with_timezone(&chrono::Utc) > now);
}

#[tokio::test]
async fn test_closed_run_blocker_does_not_block_next_schedule_fire() {
    let state = test_state_store().await;
    let goal = Goal::new_continuous("Daily blog", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let historical_run = state
        .start_goal_run(&goal.id, "manual", None, None)
        .await
        .unwrap();
    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let historical_blocker = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Old deployment attempt".to_string(),
        status: "blocked".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: Some("Deployment credentials were unavailable".to_string()),
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now_ts.clone(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&historical_blocker).await.unwrap();
    state
        .finish_goal_run(
            &historical_run.id,
            "failed",
            Some("Archived failed deployment run"),
        )
        .await
        .unwrap();

    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 2);
    assert!(tasks.iter().any(|task| task.id == historical_blocker.id));
    assert!(
        tasks
            .iter()
            .any(|task| task.description.starts_with("Scheduled check:")),
        "a blocker retained in a closed run must not suppress future scheduled work"
    );
}

#[tokio::test]
async fn test_blocked_scheduled_run_does_not_coalesce_later_occurrence() {
    let state = test_state_store().await;
    let goal = Goal::new_continuous("Daily synthetic journal", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: Some((now - chrono::Duration::minutes(2)).to_rfc3339()),
        next_run_at: (now - chrono::Duration::minutes(1)).to_rfc3339(),
        created_at: now_ts.clone(),
        updated_at: now_ts.clone(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let blocked_run = state
        .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
        .await
        .unwrap();
    let blocked_task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Review the latest synthetic repository work".to_string(),
        status: "blocked".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: Some("No verifiable new material was available for this run".to_string()),
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now_ts.clone(),
        started_at: Some(now_ts.clone()),
        completed_at: Some(now_ts),
    };
    state.create_task(&blocked_task).await.unwrap();
    state
        .finish_goal_run(
            &blocked_run.id,
            "blocked",
            Some("This scheduled occurrence could not proceed."),
        )
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.fire_due_schedule(schedule).await.unwrap();

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 2);
    assert!(tasks.iter().any(|task| task.id == blocked_task.id));
    assert!(tasks
        .iter()
        .any(|task| task.description.starts_with("Scheduled check:")));

    let runs = state.get_goal_runs(&goal.id).await.unwrap();
    let retired = runs
        .iter()
        .find(|run| run.id == blocked_run.id)
        .expect("the prior run remains available as history");
    assert_eq!(retired.status, "failed");
    assert!(retired.completed_at.is_some());
    assert_ne!(
        state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("the later occurrence has an open run")
            .id,
        blocked_run.id
    );
}

#[tokio::test]
async fn scheduled_lifecycle_reconciles_before_the_next_fire_is_due() {
    let state = test_state_store().await;
    let goal = Goal::new_continuous("Synthetic future schedule", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    state
        .create_goal_schedule(&GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("0 6 * * *".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some(now_ts.clone()),
            next_run_at: (now + chrono::Duration::days(1)).to_rfc3339(),
            created_at: now_ts.clone(),
            updated_at: now_ts.clone(),
        })
        .await
        .unwrap();
    let run = state
        .start_goal_run(&goal.id, "scheduled", None, None)
        .await
        .unwrap();
    let completed_root = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Synthetic scheduled root".to_string(),
        status: "completed".to_string(),
        priority: "normal".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: Some("root finished".to_string()),
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now_ts.clone(),
        started_at: Some(now_ts.clone()),
        completed_at: Some(now_ts.clone()),
    };
    state.create_task(&completed_root).await.unwrap();
    let blocked_child = Task {
        id: uuid::Uuid::new_v4().to_string(),
        description: "Synthetic blocked child".to_string(),
        status: "blocked".to_string(),
        task_order: 1,
        result: None,
        blocker: Some("Typed terminal blocker".to_string()),
        ..completed_root
    };
    state.create_task(&blocked_child).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.reconcile_open_scheduled_runs().await;

    let reconciled = state
        .get_goal_runs(&goal.id)
        .await
        .unwrap()
        .into_iter()
        .find(|candidate| candidate.id == run.id)
        .unwrap();
    assert_eq!(reconciled.status, "failed");
    assert!(reconciled.completed_at.is_some());
}

#[tokio::test]
async fn escalated_goal_gets_one_automatic_recovery_run_whose_root_survives_dispatch() {
    let (_db_file, db_path) = tempfile::NamedTempFile::new().unwrap().keep().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let sqlite = Arc::new(
        SqliteStateStore::new(db_path.to_str().unwrap(), 100, None, embedding_service)
            .await
            .unwrap(),
    );
    let pool = sqlite.pool();
    let state: Arc<dyn StateStore> = sqlite.clone();
    let goal = Goal::new_continuous(
        "Publish the synthetic daily digest",
        "session-1",
        None,
        None,
    );
    state.create_goal(&goal).await.unwrap();
    let now = chrono::Utc::now();
    let escalated_at = (now - chrono::Duration::hours(12)).to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 6 * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("0 6 * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: now.to_rfc3339(),
        created_at: escalated_at.clone(),
        updated_at: escalated_at.clone(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();
    let failed_run = state
        .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
        .await
        .unwrap();
    state
        .finish_goal_run(&failed_run.id, "failed", Some("synthetic"))
        .await
        .unwrap();
    // Escalate through the store's own failure accounting so the paused
    // schedule is owned by the recovery machine.
    for _ in 0..3 {
        let run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        state
            .finish_goal_run(&run.id, "failed", Some("synthetic"))
            .await
            .unwrap();
    }
    let recovery = state
        .get_scheduled_recovery_state(&goal.id)
        .await
        .unwrap()
        .expect("recovery state");
    assert_eq!(
        recovery.disposition,
        crate::traits::ScheduledRecoveryDisposition::Escalated
    );
    // Age the escalation past the cool-down.
    sqlx::query("UPDATE scheduled_recovery_state SET updated_at = ? WHERE goal_id = ?")
        .bind(&escalated_at)
        .bind(&goal.id)
        .execute(&pool)
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let runs = state.get_goal_runs(&goal.id).await.unwrap();
    let recovery_run = runs
        .iter()
        .find(|run| run.trigger_type == "recovery")
        .expect("automatic recovery run");
    assert!(matches!(
        recovery_run.status.as_str(),
        "pending" | "running"
    ));
    let root = state
        .get_task(recovery_run.root_task_id.as_deref().expect("root task"))
        .await
        .unwrap()
        .expect("root task row");
    assert_ne!(
        root.status, "cancelled",
        "recovery root must not be retired as stale"
    );
    assert!(root
        .description
        .contains("automatic recovery attempt 1 of 3"));
    let state_after = state
        .get_scheduled_recovery_state(&goal.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(state_after.recovery_attempts, 1);

    // A second tick inside the cool-down launches nothing more.
    coordinator.tick().await.unwrap();
    let runs = state.get_goal_runs(&goal.id).await.unwrap();
    assert_eq!(
        runs.iter()
            .filter(|run| run.trigger_type == "recovery")
            .count(),
        1
    );
}

#[tokio::test]
async fn test_open_run_blocker_coalesces_next_schedule_fire() {
    let state = test_state_store().await;
    let goal = Goal::new_continuous("Daily tweet", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let open_run = state
        .start_goal_run(&goal.id, "manual", None, None)
        .await
        .unwrap();
    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let active_blocker = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Await editorial approval".to_string(),
        status: "blocked".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: Some("Waiting for a user decision".to_string()),
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: now_ts.clone(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&active_blocker).await.unwrap();

    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, active_blocker.id);
    assert_eq!(
        state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .expect("blocked run should remain open")
            .id,
        open_run.id
    );

    let updated_schedule = state
        .get_goal_schedule(&schedule.id)
        .await
        .unwrap()
        .expect("recurring schedule should remain active");
    assert!(updated_schedule.last_run_at.is_none());
    assert!(
        chrono::DateTime::parse_from_rfc3339(&updated_schedule.next_run_at)
            .unwrap()
            .with_timezone(&chrono::Utc)
            > now
    );
}

#[tokio::test]
async fn test_provider_overload_blocker_does_not_coalesce_next_schedule_fire() {
    let state = test_state_store().await;
    let goal = Goal::new_continuous("Daily synthetic report", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
        created_at: now_ts.clone(),
        updated_at: now_ts.clone(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let overloaded_run = state
        .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
        .await
        .unwrap();
    let overloaded_task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Generate the scheduled report".to_string(),
        status: "blocked".to_string(),
        priority: "low".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("synthetic-worker".to_string()),
        context: None,
        result: None,
        error: None,
        blocker: Some(
            "LLM error: Codex stream failed: Our servers are currently overloaded. Please try again later."
                .to_string(),
        ),
        idempotent: true,
        retry_count: 1,
        max_retries: 1,
        created_at: now_ts.clone(),
        started_at: Some(now_ts.clone()),
        completed_at: Some(now_ts),
    };
    state.create_task(&overloaded_task).await.unwrap();
    state
        .finish_goal_run(
            &overloaded_run.id,
            "blocked",
            Some("The model service was overloaded."),
        )
        .await
        .unwrap();
    let blocked_before_fire = state
        .get_current_goal_run(&goal.id)
        .await
        .unwrap()
        .expect("the overloaded scheduled run should still be open");
    assert_eq!(blocked_before_fire.id, overloaded_run.id);
    assert_eq!(blocked_before_fire.status, "blocked");
    assert!(blocked_before_fire.completed_at.is_none());

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let runs = state.get_goal_runs(&goal.id).await.unwrap();
    let prior_run = runs
        .iter()
        .find(|run| run.id == overloaded_run.id)
        .expect("the overloaded run remains in history");
    assert_eq!(prior_run.status, "failed");
    assert!(prior_run.completed_at.is_some());

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 2);
    assert!(tasks.iter().any(|task| task.id == overloaded_task.id));
    assert!(
        tasks
            .iter()
            .any(|task| task.description.starts_with("Scheduled check:")),
        "a provider-overload blocker from the prior run must not suppress the next scheduled run"
    );
}

#[tokio::test]
async fn test_pending_task_with_interrupted_dependency_does_not_block_next_schedule_fire() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_continuous("Daily dependency cleanup test", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let dependency = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Fetch prerequisite data".to_string(),
        status: "interrupted".to_string(),
        priority: "low".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: Some("agent-1".to_string()),
        context: None,
        result: None,
        error: Some("interrupted".to_string()),
        blocker: None,
        idempotent: true,
        retry_count: 1,
        max_retries: 1,
        created_at: (now - chrono::Duration::minutes(10)).to_rfc3339(),
        started_at: Some((now - chrono::Duration::minutes(9)).to_rfc3339()),
        completed_at: Some((now - chrono::Duration::minutes(5)).to_rfc3339()),
    };
    state.create_task(&dependency).await.unwrap();

    let dependent = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Analyze prerequisite data".to_string(),
        status: "pending".to_string(),
        priority: "low".to_string(),
        task_order: 2,
        parallel_group: None,
        depends_on: Some(serde_json::json!([dependency.id.clone()]).to_string()),
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 1,
        created_at: (now - chrono::Duration::minutes(9)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&dependent).await.unwrap();

    let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "* * * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("* * * * *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: None,
        next_run_at: due_ts,
        created_at: now_ts.clone(),
        updated_at: now_ts,
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    let dependent_after = tasks
        .iter()
        .find(|t| t.id == dependent.id)
        .expect("dependent task should still exist");
    assert_eq!(dependent_after.status, "blocked");

    assert!(
        tasks
            .iter()
            .any(|t| t.description.starts_with("Scheduled check:")),
        "new scheduled run should be enqueued after unfulfillable pending dependency is blocked"
    );
}

#[tokio::test]
async fn test_multiple_due_schedules_always_fire_enqueues_multiple_tasks() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_continuous("Take medicine", "session-1", Some(5000), Some(20000));
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let now_ts = now.to_rfc3339();
    let due_ts = (now - chrono::Duration::minutes(2)).to_rfc3339();

    for _ in 0..3 {
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "* * * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("* * * * *".to_string()),
            fire_policy: "always_fire".to_string(),
            is_one_shot: true,
            is_paused: false,
            last_run_at: None,
            next_run_at: due_ts.clone(),
            created_at: now_ts.clone(),
            updated_at: now_ts.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();
    }

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(
        tasks.len(),
        3,
        "always_fire should enqueue multiple due runs"
    );

    let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
    assert!(
        schedules.is_empty(),
        "one-shot schedules should be deleted after firing"
    );
}

#[tokio::test]
async fn test_daily_budget_reset() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let sqlite_state = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );
    let state: Arc<dyn StateStore> = sqlite_state.clone();

    // Create a continuous goal with some tokens used
    let goal = Goal::new_continuous("Test budget goal", "system", Some(5000), Some(20000));
    state.create_goal(&goal).await.unwrap();

    // Manually set prior-day usage via the concrete pool.
    sqlx::query(
        "UPDATE goals
         SET tokens_used_today = 1500, tokens_used_day = date('now', '-1 day')
         WHERE id = ?",
    )
    .bind(&goal.id)
    .execute(&sqlite_state.pool())
    .await
    .unwrap();

    // Reset
    let count = state.reset_daily_token_budgets().await.unwrap();
    assert!(count >= 1, "Should have reset at least one goal");

    // Verify it's 0 now
    let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(
        updated.tokens_used_today, 0,
        "tokens_used_today should be reset to 0"
    );
}

#[tokio::test]
async fn test_daily_budget_reset_preserves_same_day_usage() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let sqlite_state = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );
    let state: Arc<dyn StateStore> = sqlite_state.clone();
    let goal = Goal::new_continuous("Same-day budget", "system", Some(5000), Some(20000));
    state.create_goal(&goal).await.unwrap();
    sqlx::query(
        "UPDATE goals
         SET tokens_used_today = 1500, tokens_used_day = date('now')
         WHERE id = ?",
    )
    .bind(&goal.id)
    .execute(&sqlite_state.pool())
    .await
    .unwrap();

    let count = state.reset_daily_token_budgets().await.unwrap();
    assert_eq!(count, 0);
    let updated = state.get_goal(&goal.id).await.unwrap().unwrap();
    assert_eq!(updated.tokens_used_today, 1500);
}

#[tokio::test]
async fn test_dispatch_no_agent_reverts_claim() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    // Create an active goal with a pending task (no running tasks = orphaned)
    let goal = Goal::new_finite("Build website", "session-1");
    state.create_goal(&goal).await.unwrap();

    let task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Deploy to production".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&task).await.unwrap();

    // Create coordinator with NO agent reference
    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(
        state.clone(),
        60,
        3,
        wake_rx,
        None, // no hub
        None, // no goal_token_registry
        None, // no telemetry
    );
    // agent is None by default — dispatch will fail to spawn

    coordinator.dispatch_pending_tasks().await;

    // Task must be back to "pending", NOT stranded in "claimed"
    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(
        tasks[0].status, "pending",
        "Task should be reverted to pending when no agent is available"
    );
    assert!(
        tasks[0].agent_id.is_none(),
        "agent_id should be cleared on revert"
    );
    assert!(
        tasks[0].started_at.is_none(),
        "started_at should be cleared on revert"
    );

    // A stalled notification should have been enqueued
    let notifications = state.get_pending_notifications(10).await.unwrap();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].notification_type, "stalled");
    assert_eq!(notifications[0].goal_id, goal.id);
}

#[tokio::test]
async fn heartbeat_recovers_stranded_manual_run_before_dispatch() {
    let state = test_state_store().await;
    let mut goal = Goal::new_continuous(
        "Publish one diary entry",
        "session-1",
        Some(5_000),
        Some(20_000),
    );
    goal.status = "stalled".to_string();
    state.create_goal(&goal).await.unwrap();

    let root = synthetic_task("manual root", "pending");
    let mut root = Task {
        goal_id: goal.id.clone(),
        created_at: (chrono::Utc::now() - chrono::Duration::minutes(15)).to_rfc3339(),
        ..root
    };
    root.id = uuid::Uuid::new_v4().to_string();
    let run = state
        .start_goal_run(&goal.id, "manual", None, Some(&root.id))
        .await
        .unwrap();
    state.create_task(&root).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.dispatch_pending_tasks().await;

    assert_eq!(
        state.get_goal(&goal.id).await.unwrap().unwrap().status,
        "active"
    );
    assert_eq!(
        state.get_tasks_for_goal(&goal.id).await.unwrap()[0].status,
        "pending"
    );
    assert_eq!(state.get_goal_runs(&goal.id).await.unwrap().len(), 1);
    assert_eq!(state.get_goal_runs(&goal.id).await.unwrap()[0].id, run.id);

    let notifications = state.get_pending_notifications(10).await.unwrap();
    assert!(notifications
        .iter()
        .all(|entry| entry.notification_type != "status_update"));
    assert!(notifications
        .iter()
        .any(|entry| entry.notification_type == "stalled"));
}

#[tokio::test]
async fn active_task_lead_owns_retries_and_pending_dispatch() {
    let state = test_state_store().await;
    let goal = Goal::new_finite("Build and deploy a site", "session-1");
    state.create_goal(&goal).await.unwrap();
    let old = (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339();
    let failed = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Research visual direction".to_string(),
        status: "failed".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: Some("Research was incorporated directly.".to_string()),
        error: Some("Late persistence race".to_string()),
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: old.clone(),
        started_at: None,
        completed_at: Some(old.clone()),
    };
    let pending = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Return the deployment URL".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: old,
        started_at: None,
        completed_at: None,
    };
    state.create_task(&failed).await.unwrap();
    state.create_task(&pending).await.unwrap();

    let registry = GoalTokenRegistry::new();
    let _active_run = registry
        .try_acquire_run(&goal.id)
        .expect("test task lead should own run");
    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, Some(registry), None);

    coordinator.auto_retry_failed_tasks().await;
    coordinator.dispatch_pending_tasks().await;

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(
        tasks
            .iter()
            .find(|task| task.id == failed.id)
            .unwrap()
            .status,
        "failed"
    );
    assert_eq!(
        tasks
            .iter()
            .find(|task| task.id == pending.id)
            .unwrap()
            .status,
        "pending"
    );
    assert!(state
        .get_pending_notifications(10)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn closed_scheduled_run_children_are_neither_retried_nor_dispatched() {
    let state = test_state_store().await;
    let goal = Goal::new_continuous("Publish a daily post", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let cycle_start = now - chrono::Duration::hours(2);
    state
        .create_goal_schedule(&GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 8 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: None,
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: Some(cycle_start.to_rfc3339()),
            next_run_at: (now + chrono::Duration::hours(22)).to_rfc3339(),
            created_at: cycle_start.to_rfc3339(),
            updated_at: cycle_start.to_rfc3339(),
        })
        .await
        .unwrap();

    let failed = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Deploy the post".to_string(),
        status: "failed".to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: Some("ambiguous deploy result".to_string()),
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: (cycle_start + chrono::Duration::minutes(5)).to_rfc3339(),
        started_at: None,
        completed_at: Some((cycle_start + chrono::Duration::minutes(6)).to_rfc3339()),
    };
    state.create_task(&failed).await.unwrap();
    let pending = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Verify the old deployment".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 2,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: (cycle_start + chrono::Duration::minutes(7)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&pending).await.unwrap();
    let run = state
        .get_current_goal_run(&goal.id)
        .await
        .unwrap()
        .expect("task creation should bind both children to one implicit run");
    state
        .finish_goal_run(&run.id, "failed", Some("synthetic closed run"))
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.auto_retry_failed_tasks().await;
    coordinator.dispatch_pending_tasks().await;

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    assert_eq!(
        tasks
            .iter()
            .find(|task| task.id == failed.id)
            .unwrap()
            .status,
        "failed",
        "closed-run failures must not be promoted back to pending"
    );
    let pending_after = tasks.iter().find(|task| task.id == pending.id).unwrap();
    assert_eq!(pending_after.status, "cancelled");
    assert!(!pending_after.idempotent);
    assert_eq!(pending_after.max_retries, 0);
}

/// Live repro (goal 9a744834, 2026-07-04): a scheduled "post one tweet"
/// goal posted 5 real duplicate tweets in one cycle because orphaned
/// pending tasks kept getting redispatched after the tweet had already
/// posted successfully (downstream verification/task-tracking hiccups
/// made it look unfinished). This is the regression test for the fix:
/// dispatch_pending_tasks must close out orphaned tasks instead of
/// redispatching once a successful mutating http_request is logged for
/// the goal's current cycle.
#[tokio::test]
async fn test_dispatch_skips_redispatch_after_mutating_success_this_cycle() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_finite("Post one tweet", "session-1");
    state.create_goal(&goal).await.unwrap();

    // Schedule fired 10 minutes ago — that's the current cycle's start.
    let cycle_start = chrono::Utc::now() - chrono::Duration::minutes(10);
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 9 * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: None,
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: Some(cycle_start.to_rfc3339()),
        next_run_at: (chrono::Utc::now() + chrono::Duration::hours(23)).to_rfc3339(),
        created_at: cycle_start.to_rfc3339(),
        updated_at: cycle_start.to_rfc3339(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    // The task lead's original task already posted the tweet successfully.
    let posted_task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Scheduled check: Post one short tweet".to_string(),
        status: "interrupted".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: (cycle_start + chrono::Duration::minutes(1)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&posted_task).await.unwrap();
    state
        .log_task_activity(&TaskActivity {
            id: 0,
            task_id: posted_task.id.clone(),
            activity_type: crate::traits::REMOTE_CHANGE_ACTIVITY.to_string(),
            tool_name: Some("http_request".to_string()),
            tool_args: Some(r#"{"method":"POST","url":"https://api.x.com/2/tweets"}"#.to_string()),
            result: Some("HTTP 201 Created".to_string()),
            success: Some(true),
            tokens_used: None,
            created_at: (cycle_start + chrono::Duration::minutes(2)).to_rfc3339(),
        })
        .await
        .unwrap();

    // A leftover subtask the orchestrator created before things got
    // confused, now orphaned (pending, >60s old, nothing running).
    let orphaned = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Post the tweet".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 1,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&orphaned).await.unwrap();
    state
        .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
            goal_id: goal.id.clone(),
            root_task_id: posted_task.id.clone(),
            effective_budget_per_check: 5000,
            tokens_used: 0,
            budget_extensions_count: 0,
            health: Default::default(),
            created_at: cycle_start.to_rfc3339(),
            updated_at: cycle_start.to_rfc3339(),
        })
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    // agent is None — if the gate didn't fire, this would fall through to
    // the "no agent available" revert-to-pending + stalled-notification path.
    coordinator.dispatch_pending_tasks().await;

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    let posted_after = tasks.iter().find(|t| t.id == posted_task.id).unwrap();
    assert_eq!(
        posted_after.status, "completed",
        "a successful mutation receipt should reconcile its interrupted task"
    );
    let orphaned_after = tasks.iter().find(|t| t.id == orphaned.id).unwrap();
    assert_eq!(
        orphaned_after.status, "completed",
        "orphaned task should be closed out, not redispatched, once a \
         mutating success is already logged for this cycle"
    );
    assert!(
        state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .is_none(),
        "a fully reconciled run must not remain open after orphan closeout"
    );
    assert_eq!(
        state.get_goal_runs(&goal.id).await.unwrap()[0].status,
        "completed"
    );

    let notifications = state.get_pending_notifications(10).await.unwrap();
    assert!(
        notifications
            .iter()
            .all(|n| n.notification_type != "stalled"),
        "should not fall through to the stalled-notification path once the \
         idempotency gate has closed out the orphaned task"
    );
}

/// A successful mutation in an earlier run is not a receipt for a new
/// explicit manual run. This reproduces task 30e0c819 (2026-08-01):
/// `trigger_now` created a new run without changing the cron schedule's
/// `last_run_at`, and goal-wide timestamp matching auto-closed the new task
/// from the previous run's tweet receipt.
#[tokio::test]
async fn test_manual_run_is_not_suppressed_by_prior_run_mutating_success() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_continuous("Post one useful tweet", "session-1", None, None);
    state.create_goal(&goal).await.unwrap();

    let now = chrono::Utc::now();
    let prior_cycle = now - chrono::Duration::minutes(15);
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 9 * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: None,
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: Some(prior_cycle.to_rfc3339()),
        next_run_at: (now + chrono::Duration::hours(20)).to_rfc3339(),
        created_at: prior_cycle.to_rfc3339(),
        updated_at: prior_cycle.to_rfc3339(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let first_run = state
        .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
        .await
        .unwrap();
    let first_task = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Scheduled check: Post one useful tweet".to_string(),
        status: "completed".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: Some("Posted and verified.".to_string()),
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 0,
        created_at: prior_cycle.to_rfc3339(),
        started_at: Some(prior_cycle.to_rfc3339()),
        completed_at: Some((prior_cycle + chrono::Duration::minutes(1)).to_rfc3339()),
    };
    state.create_task(&first_task).await.unwrap();
    state
        .log_task_activity(&TaskActivity {
            id: 0,
            task_id: first_task.id.clone(),
            activity_type: crate::traits::REMOTE_CHANGE_ACTIVITY.to_string(),
            tool_name: Some("http_request".to_string()),
            tool_args: Some(r#"{"method":"POST","url":"https://api.x.com/2/tweets"}"#.to_string()),
            result: Some("HTTP 201 Created".to_string()),
            success: Some(true),
            tokens_used: None,
            created_at: (prior_cycle + chrono::Duration::minutes(1)).to_rfc3339(),
        })
        .await
        .unwrap();
    state
        .finish_goal_run(&first_run.id, "completed", Some("First tweet posted."))
        .await
        .unwrap();

    let second_run = state
        .start_goal_run(&goal.id, "manual", None, None)
        .await
        .unwrap();
    let pending = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Manual scheduled run: Post one useful tweet".to_string(),
        status: "pending".to_string(),
        priority: "high".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: true,
        retry_count: 0,
        max_retries: 3,
        created_at: (now - chrono::Duration::minutes(2)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&pending).await.unwrap();
    state
        .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
            goal_id: goal.id.clone(),
            root_task_id: pending.id.clone(),
            effective_budget_per_check: 5000,
            tokens_used: 0,
            budget_extensions_count: 0,
            health: Default::default(),
            created_at: (now - chrono::Duration::minutes(3)).to_rfc3339(),
            updated_at: (now - chrono::Duration::minutes(3)).to_rfc3339(),
        })
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.dispatch_pending_tasks().await;

    let pending_after = state.get_task(&pending.id).await.unwrap().unwrap();
    assert_ne!(
        pending_after.status, "completed",
        "the previous run's mutation receipt must not auto-close a new manual run"
    );
    assert!(!pending_after
        .result
        .as_deref()
        .is_some_and(|result| result.contains("mutating action")));
    assert_eq!(
        state
            .get_current_goal_run(&goal.id)
            .await
            .unwrap()
            .unwrap()
            .id,
        second_run.id,
        "the new manual run must remain eligible for dispatch"
    );
}

/// Control for the test above: without any logged mutating success, the
/// existing orphan-dispatch behavior (revert to pending + stalled
/// notification, since there's no agent) must still happen — the new
/// gate must not fire on vibes.
#[tokio::test]
async fn test_dispatch_still_redispatches_without_prior_mutating_success() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let goal = Goal::new_finite("Post one tweet", "session-1");
    state.create_goal(&goal).await.unwrap();

    let cycle_start = chrono::Utc::now() - chrono::Duration::minutes(10);
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 9 * * *".to_string(),
        tz: "local".to_string(),
        original_schedule: None,
        fire_policy: "coalesce".to_string(),
        is_one_shot: false,
        is_paused: false,
        last_run_at: Some(cycle_start.to_rfc3339()),
        next_run_at: (chrono::Utc::now() + chrono::Duration::hours(23)).to_rfc3339(),
        created_at: cycle_start.to_rfc3339(),
        updated_at: cycle_start.to_rfc3339(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let orphaned = Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        description: "Post the tweet".to_string(),
        status: "pending".to_string(),
        priority: "medium".to_string(),
        task_order: 0,
        parallel_group: None,
        depends_on: None,
        agent_id: None,
        context: None,
        result: None,
        error: None,
        blocker: None,
        idempotent: false,
        retry_count: 0,
        max_retries: 3,
        created_at: (chrono::Utc::now() - chrono::Duration::seconds(120)).to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    state.create_task(&orphaned).await.unwrap();
    state
        .upsert_scheduled_run_state(&crate::traits::ScheduledRunState {
            goal_id: goal.id.clone(),
            root_task_id: orphaned.id.clone(),
            effective_budget_per_check: 5000,
            tokens_used: 0,
            budget_extensions_count: 0,
            health: Default::default(),
            created_at: cycle_start.to_rfc3339(),
            updated_at: cycle_start.to_rfc3339(),
        })
        .await
        .unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let coordinator = HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.dispatch_pending_tasks().await;

    let tasks = state.get_tasks_for_goal(&goal.id).await.unwrap();
    let orphaned_after = tasks.iter().find(|t| t.id == orphaned.id).unwrap();
    assert_eq!(
        orphaned_after.status, "pending",
        "without a logged mutating success, normal orphan-dispatch behavior \
         (revert to pending since there's no agent) must be unchanged"
    );

    let notifications = state.get_pending_notifications(10).await.unwrap();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].notification_type, "stalled");
}

#[tokio::test]
async fn test_deferred_finite_goal_fires() {
    // Deprecated: deferred finite goals are now represented as one-shot goal_schedules.
    // This behavior is covered by test_due_one_shot_schedule_creates_task_and_deletes_schedule().
}

#[test]
fn task_inactivity_secs_uses_last_activity_then_started_at() {
    let now = chrono::DateTime::parse_from_rfc3339("2026-06-23T00:10:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    // Last activity 2 min ago → 120s.
    assert_eq!(
        task_inactivity_secs(Some("2026-06-23T00:08:00Z"), "2026-06-23T00:00:00Z", now),
        120
    );
    // No activity → falls back to started_at (10 min ago → 600s).
    assert_eq!(task_inactivity_secs(None, "2026-06-23T00:00:00Z", now), 600);
    // Unparseable inputs → 0 (never panics).
    assert_eq!(task_inactivity_secs(Some("garbage"), "garbage", now), 0);
    // SQLite-datetime format ('YYYY-MM-DD HH:MM:SS', the stored task_activity format)
    // must parse identically — last activity 5 min ago → 300s.
    assert_eq!(
        task_inactivity_secs(Some("2026-06-23 00:05:00"), "2026-06-23 00:00:00", now),
        300
    );
}

#[test]
fn task_escalations_wait_for_same_turn_state_reconciliation() {
    let now = chrono::DateTime::parse_from_rfc3339("2026-07-31T16:00:10Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    assert!(!task_escalation_has_settled("2026-07-31T16:00:05Z", now));
    assert!(task_escalation_has_settled("2026-07-31T16:00:01Z", now));
    assert!(task_escalation_has_settled("legacy timestamp", now));
}

#[tokio::test]
async fn test_stale_pending_confirmation_cleanup() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let embedding_service = Arc::new(EmbeddingService::new().unwrap());
    let state: Arc<dyn StateStore> = Arc::new(
        SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap(),
    );

    let mut goal = Goal::new_deferred_finite("Remind me tomorrow", "session-1");
    goal.created_at = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
    goal.updated_at = goal.created_at.clone();
    state.create_goal(&goal).await.unwrap();
    let schedule = GoalSchedule {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: goal.id.clone(),
        cron_expr: "0 9 12 2 *".to_string(),
        tz: "local".to_string(),
        original_schedule: Some("0 9 12 2 *".to_string()),
        fire_policy: "coalesce".to_string(),
        is_one_shot: true,
        is_paused: false,
        last_run_at: None,
        next_run_at: (chrono::Utc::now() + chrono::Duration::hours(2)).to_rfc3339(),
        created_at: goal.created_at.clone(),
        updated_at: goal.updated_at.clone(),
    };
    state.create_goal_schedule(&schedule).await.unwrap();

    let (_wake_tx, wake_rx) = mpsc::channel::<()>(1);
    let mut coordinator =
        HeartbeatCoordinator::new(state.clone(), 60, 3, wake_rx, None, None, None);
    coordinator.tick().await.unwrap();

    let updated = state
        .get_goal(&goal.id)
        .await
        .unwrap()
        .expect("goal should exist");
    assert_eq!(
        updated.status, "cancelled",
        "Stale pending_confirmation goal should be auto-cancelled"
    );

    // Stale pending-confirmation cleanup should also remove schedules.
    let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
    assert!(schedules.is_empty());
}
