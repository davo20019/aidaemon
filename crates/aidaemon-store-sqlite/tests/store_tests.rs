use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider};
use aidaemon_core::{
    Agent, CancellationToken, Event, Hook, HookContext, InterruptKind, Interruption, Pending,
    ReceiptStatus, Resolution, RunId, RunState, RunStatus, RunStore, Tool, ToolAnnotations,
    ToolCall, ToolContext, ToolDecision, ToolInfo, ToolOutput, ToolResult, ToolSpec,
};
use aidaemon_store_sqlite::{latest_version, Error, SqliteStore};
use async_trait::async_trait;
use serde_json::{json, Value};

fn db_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("runs.db")
}

struct Echo;
#[async_trait]
impl Tool for Echo {
    fn spec(&self) -> ToolSpec {
        spec("echo", "echo text", &["text"])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }
    async fn call(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        Ok(ToolOutput::text(args["text"].as_str().unwrap_or_default())
            .with_details(json!({"n": 1})))
    }
}

/// Interrupts every call to `deploy`.
struct AskBeforeDeploy;
#[async_trait]
impl Hook for AskBeforeDeploy {
    async fn before_tool(
        &self,
        _: &HookContext<'_>,
        call: &ToolCall,
        _: &ToolInfo,
    ) -> ToolDecision {
        if call.name == "deploy" {
            ToolDecision::Interrupt {
                reason: "needs approval".into(),
                kind: aidaemon_core::InterruptKind::default(),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

fn deploy_tool() -> impl Tool {
    aidaemon_core::FnTool::new(
        spec("deploy", "deploy the app", &[]),
        ToolAnnotations::DESTRUCTIVE,
        |_args: Value| async { Ok(ToolOutput::text("deployed")) },
    )
}

// ---------- migrations ----------

#[tokio::test]
async fn migrations_apply_once_and_reopen_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(db_path(&dir)).await.unwrap();
    assert_eq!(store.schema_version().await.unwrap(), latest_version());
    let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM schema_migrations")
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(rows, latest_version());
    let mode: String = sqlx::query_scalar("PRAGMA journal_mode")
        .fetch_one(store.pool())
        .await
        .unwrap();
    assert_eq!(mode.to_lowercase(), "wal");
    store.close().await;

    for _ in 0..3 {
        let again = SqliteStore::open(db_path(&dir)).await.unwrap();
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM schema_migrations")
            .fetch_one(again.pool())
            .await
            .unwrap();
        assert_eq!(rows, latest_version(), "no migration re-applied");
        again.close().await;
    }
}

#[tokio::test]
async fn concurrent_opens_migrate_exactly_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = db_path(&dir);
    let opens = (0..6).map(|_| {
        let path = path.clone();
        tokio::spawn(async move { SqliteStore::open(path).await })
    });
    for handle in opens {
        handle.await.unwrap().unwrap();
    }
    let store = SqliteStore::open(&path).await.unwrap();
    assert_eq!(store.schema_version().await.unwrap(), latest_version());
}

#[tokio::test]
async fn newer_schema_version_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(db_path(&dir)).await.unwrap();
    sqlx::query("INSERT INTO schema_migrations (version, applied_at) VALUES (?, 'later')")
        .bind(latest_version() + 1)
        .execute(store.pool())
        .await
        .unwrap();
    store.close().await;

    let err = SqliteStore::open(db_path(&dir)).await.unwrap_err();
    match &err {
        Error::SchemaTooNew { found, supported } => {
            assert_eq!(*found, latest_version() + 1);
            assert_eq!(*supported, latest_version());
        }
        other => panic!("expected SchemaTooNew, got {other:?}"),
    }
    assert!(err.to_string().contains("newer than this build"), "{err}");
}

// ---------- events ----------

#[tokio::test]
async fn events_append_in_order_with_per_run_sequence() {
    let store = SqliteStore::open_in_memory().await.unwrap();
    let (a, b) = (RunId::new(), RunId::new());
    let first = Event::RunStarted {
        session_id: "s1".into(),
    };
    let second = Event::TurnStarted { turn: 1 };
    assert_eq!(store.append(a, &first).await.unwrap(), 1);
    assert_eq!(store.append(b, &first).await.unwrap(), 1);
    assert_eq!(store.append(a, &second).await.unwrap(), 2);

    assert_eq!(store.events(a).await.unwrap(), vec![first.clone(), second]);
    assert_eq!(store.events(b).await.unwrap(), vec![first]);
    let raw = store.raw_events(a).await.unwrap();
    assert_eq!(raw[0].event_type, "run_started");
    assert_eq!(raw[1].event_type, "turn_started");
    assert!(raw[0].created_at.ends_with('Z'));

    let edit = sqlx::query("UPDATE run_events SET event_type = 'x'")
        .execute(store.pool())
        .await;
    assert!(edit.is_err(), "journal is append-only");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_appends_get_unique_gap_free_sequence_numbers() {
    let dir = tempfile::tempdir().unwrap();
    // Two stores (two pools) on one file, as two processes would be.
    let one = SqliteStore::open(db_path(&dir)).await.unwrap();
    let two = SqliteStore::open(db_path(&dir)).await.unwrap();
    let run = RunId::new();
    let mut handles = Vec::new();
    for i in 0..60u32 {
        let store = if i % 2 == 0 { one.clone() } else { two.clone() };
        handles.push(tokio::spawn(async move {
            store
                .append(run, &Event::TurnStarted { turn: i })
                .await
                .unwrap()
        }));
    }
    let mut seqs = Vec::new();
    for h in handles {
        seqs.push(h.await.unwrap());
    }
    seqs.sort_unstable();
    assert_eq!(seqs, (1..=60).collect::<Vec<i64>>());

    let raw = one.raw_events(run).await.unwrap();
    assert_eq!(raw.len(), 60);
    assert!(raw.windows(2).all(|w| w[0].seq + 1 == w[1].seq));
}

// ---------- states ----------

#[tokio::test]
async fn state_round_trips_exactly_and_upserts() {
    let store = Arc::new(SqliteStore::open_in_memory().await.unwrap());
    let agent = Agent::builder("worker")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([
            tool_calls([call("c1", "echo", json!({"text": "hi"}))]),
            text("done"),
        ]))
        .tool(Echo)
        .store(store.clone())
        .build()
        .unwrap();
    let out = agent.run("session-1", "say hi").await.unwrap();
    assert!(out.is_completed());

    let loaded = store.load_state(out.state.run_id).await.unwrap().unwrap();
    assert_eq!(loaded, out.state, "stored state equals the outcome state");
    assert_eq!(loaded.receipts[0].details, json!({"n": 1}));

    // Saving again replaces, never duplicates.
    let mut changed = loaded.clone();
    changed.nudges = 7;
    store.save_state(&changed).await.unwrap();
    assert_eq!(
        store.load_state(changed.run_id).await.unwrap(),
        Some(changed)
    );
    assert_eq!(store.runs_for_session("session-1").await.unwrap().len(), 1);
    assert!(store.load_state(RunId::new()).await.unwrap().is_none());

    // The loop journaled the run through the trait.
    let events = store.events(out.state.run_id).await.unwrap();
    assert!(matches!(events.first(), Some(Event::RunStarted { .. })));
    assert!(matches!(events.last(), Some(Event::Finished { .. })));
}

#[tokio::test]
async fn session_queries_and_interrupted_runs() {
    let store = Arc::new(SqliteStore::open_in_memory().await.unwrap());
    let build = |provider: ScriptedProvider| {
        Agent::builder("ops")
            .allow_unguarded()
            .model("m")
            .provider(provider)
            .tool(deploy_tool())
            .hook(AskBeforeDeploy)
            .store(store.clone())
            .build()
            .unwrap()
    };

    let first = build(ScriptedProvider::new([text("hello")]))
        .run("session-a", "hi")
        .await
        .unwrap();
    let paused = build(ScriptedProvider::new([tool_calls([call(
        "d1",
        "deploy",
        json!({}),
    )])]))
    .run("session-a", "deploy it")
    .await
    .unwrap();
    assert!(matches!(paused.status, RunStatus::Interrupted { .. }));
    build(ScriptedProvider::new([text("other")]))
        .run("session-b", "unrelated")
        .await
        .unwrap();

    let runs = store.runs_for_session("session-a").await.unwrap();
    let ids: Vec<RunId> = runs.iter().map(|s| s.run_id).collect();
    assert_eq!(ids, vec![first.state.run_id, paused.state.run_id]);
    let latest = store
        .latest_state_for_session("session-a")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(latest.run_id, paused.state.run_id);
    assert!(store
        .latest_state_for_session("nobody")
        .await
        .unwrap()
        .is_none());

    let interrupted = store.interrupted_runs().await.unwrap();
    assert_eq!(interrupted.len(), 1);
    assert_eq!(interrupted[0], paused.state);
    assert!(store.in_flight_runs().await.unwrap().is_empty());

    // Settle it from the stored copy; it leaves the interrupted set.
    let done = build(ScriptedProvider::new([text("shipped")]))
        .resume(interrupted[0].clone(), Resolution::Approve)
        .await
        .unwrap();
    assert!(done.is_completed());
    assert!(store.interrupted_runs().await.unwrap().is_empty());
}

// ---------- crash recovery ----------

/// Blocks forever on its first call (the "crash"), succeeds afterwards.
struct HangsOnce {
    calls: Arc<AtomicUsize>,
}
#[async_trait]
impl Tool for HangsOnce {
    fn spec(&self) -> ToolSpec {
        spec("write_report", "hangs once", &[])
    }
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
            egress: false,
        }
    }
    async fn call(&self, _args: Value, _ctx: &ToolContext) -> ToolResult {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            std::future::pending::<()>().await;
        }
        Ok(ToolOutput::text("report written"))
    }
}

#[tokio::test]
async fn crashed_run_is_found_after_reopen_and_finished() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let seen: Arc<Mutex<Option<RunId>>> = Arc::default();

    // Process one: dies while the tool runs.
    {
        let store = Arc::new(SqliteStore::open(db_path(&dir)).await.unwrap());
        let sink = seen.clone();
        let agent = Agent::builder("worker")
            .allow_unguarded()
            .model("m")
            .provider(ScriptedProvider::new([tool_calls([call(
                "c1",
                "write_report",
                json!({}),
            )])]))
            .tool(HangsOnce {
                calls: calls.clone(),
            })
            .store(store.clone())
            .event_sink(move |id, _e: &Event| {
                sink.lock().unwrap().get_or_insert(id);
            })
            .build()
            .unwrap();
        let crashed =
            tokio::time::timeout(Duration::from_millis(300), agent.run("s-crash", "report")).await;
        assert!(crashed.is_err(), "first run must not finish");
        store.close().await;
    }

    // Process two: reopen the same file and recover.
    let store = Arc::new(SqliteStore::open(db_path(&dir)).await.unwrap());
    let stranded = store.in_flight_runs().await.unwrap();
    assert_eq!(stranded.len(), 1);
    let state = stranded.into_iter().next().unwrap();
    assert_eq!(Some(state.run_id), *seen.lock().unwrap());
    assert_eq!(state.session_id, "s-crash");

    let agent = Agent::builder("worker")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([text("report handled")]))
        .tool(HangsOnce {
            calls: calls.clone(),
        })
        .store(store.clone())
        .build()
        .unwrap();
    let out = agent
        .run_from(state.clone(), CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(out.status, RunStatus::Completed);
    assert_eq!(out.final_text.as_deref(), Some("report handled"));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "idempotent call re-run once"
    );
    assert_eq!(out.receipts()[0].status, ReceiptStatus::Succeeded);

    assert!(store.in_flight_runs().await.unwrap().is_empty());
    let saved = store.load_state(state.run_id).await.unwrap().unwrap();
    assert_eq!(saved, out.state);
    let events = store.events(state.run_id).await.unwrap();
    assert!(events.iter().any(|e| matches!(e, Event::Recovered { .. })));
    assert!(matches!(events.last(), Some(Event::Finished { .. })));
}

#[tokio::test]
async fn erase_session_removes_only_that_sessions_runs_and_keeps_the_guard() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(db_path(&dir)).await.unwrap();
    let erased = aidaemon_core::RunState::new("aidaemon", "s-erased", "private text");
    let kept = aidaemon_core::RunState::new("aidaemon", "s-kept", "other text");
    for state in [&erased, &kept] {
        store
            .append(state.run_id, &Event::TurnStarted { turn: 1 })
            .await
            .unwrap();
        store.upsert_state(state).await.unwrap();
    }

    let mut tx = store.pool().begin().await.unwrap();
    aidaemon_store_sqlite::erase_session_in_tx(&mut tx, "s-erased")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert!(store.state(erased.run_id).await.unwrap().is_none());
    assert!(store.events(erased.run_id).await.unwrap().is_empty());
    assert!(store.state(kept.run_id).await.unwrap().is_some());
    assert_eq!(store.events(kept.run_id).await.unwrap().len(), 1);

    // The append-only guard is back in force for every other delete.
    let blocked = sqlx::query("DELETE FROM run_events WHERE run_id = ?")
        .bind(kept.run_id.to_string())
        .execute(store.pool())
        .await;
    let message = blocked
        .expect_err("run_events must stay append-only")
        .to_string();
    assert!(message.contains("append-only"), "wrong failure: {message}");
}

/// `/wipe` on a pool the run store was never opened on has nothing to erase
/// and must not fail.
#[tokio::test]
async fn erase_session_without_a_journal_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}?mode=rwc", db_path(&dir).display());
    let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
    let mut tx = pool.begin().await.unwrap();
    aidaemon_store_sqlite::erase_session_in_tx(&mut tx, "s-never-journaled")
        .await
        .unwrap();
    tx.commit().await.unwrap();
}

#[tokio::test]
async fn a_paused_parent_keeps_its_child_and_only_the_parent_is_listed() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(db_path(&dir)).await.unwrap();

    let pending = |call: &str, child: Option<RunState>| Pending {
        calls: vec![ToolCall::new(call, "send", json!({}))],
        next: 0,
        interruption: Interruption {
            call: ToolCall::new("k1", "send", json!({})),
            reason: "the owner must confirm".to_string(),
            kind: InterruptKind::default(),
        },
        hook_index: None,
        hook_chain: Vec::new(),
        child: child.map(Box::new),
    };
    let mut parent = RunState::new("parent", "telegram:synthetic-user-1", "go");
    let mut child = RunState::new("child", "telegram:synthetic-user-1/sub/c1", "task")
        .with_parent(parent.run_id);
    child.pending = Some(pending("k1", None));
    parent.pending = Some(pending("c1", Some(child.clone())));
    store.upsert_state(&child).await.unwrap();
    store.upsert_state(&parent).await.unwrap();

    let mut listed = store.interrupted_runs().await.unwrap();
    assert_eq!(listed.len(), 1);
    let listed = listed.pop().unwrap();
    assert_eq!(listed, parent);
    assert_eq!(listed.waiting_run().unwrap().run_id, child.run_id);
}

#[tokio::test]
async fn a_run_is_ended_only_while_its_last_event_is_the_end() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(db_path(&dir)).await.unwrap();
    let run = RunId::new();
    assert!(!store.run_ended(run).await.unwrap());

    let finished = Event::Finished {
        status: RunStatus::Cancelled,
        usage: Default::default(),
        turns: 1,
    };
    store.append(run, &finished).await.unwrap();
    assert!(store.run_ended(run).await.unwrap());

    store.append(run, &Event::RunResumed).await.unwrap();
    assert!(!store.run_ended(run).await.unwrap());
}

#[tokio::test]
async fn a_run_paused_for_an_answer_has_not_ended() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(db_path(&dir)).await.unwrap();
    let run = RunId::new();
    let paused = Event::Finished {
        status: RunStatus::Interrupted {
            interruption: Interruption {
                call: ToolCall::new("c1", "terminal", serde_json::json!({})),
                reason: "needs the owner".to_string(),
                kind: InterruptKind::Untrusted,
            },
        },
        usage: Default::default(),
        turns: 1,
    };
    store.append(run, &paused).await.unwrap();
    assert!(
        !store.run_ended(run).await.unwrap(),
        "a pause is not an end: a restart must still find the run"
    );
}
