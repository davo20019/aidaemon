use super::*;
use crate::state::SqliteStateStore;
use crate::stores::{NotificationStore, StateStore};
use aidaemon_memory::embeddings::EmbeddingService;
use std::sync::Arc;

/// A background spawn ends the turn from the tool's typed outcome, not
/// from its wording: the finish and the handoff fact are set whatever
/// the message says.
#[test]
fn background_spawn_hands_the_turn_off() {
    let outcome = background_spawn_outcome("Summarize the synthetic report");
    assert!(outcome.metadata.background_handoff);
    assert!(
        !outcome.metadata.background_started,
        "a sub-agent is not a tracked process launch"
    );
    let finish = outcome.finish.expect("the turn hands off");
    assert_eq!(finish.outcome, aidaemon_core::FinishOutcome::Partial);
    assert_eq!(
        finish.reply, None,
        "the dispatcher writes the handoff reply"
    );
}

#[test]
fn truncate_utf8_ascii() {
    assert_eq!(truncate_utf8("hello world", 5), "hello");
    assert_eq!(truncate_utf8("hello", 10), "hello");
    assert_eq!(truncate_utf8("hello", 5), "hello");
}

#[test]
fn truncate_utf8_multibyte() {
    // Each emoji is 4 bytes
    let s = "🔥🔥🔥";
    assert_eq!(s.len(), 12);
    // Limit 4 should include exactly the first emoji
    assert_eq!(truncate_utf8(s, 4), "🔥");
    // Limit 5 should still only include the first emoji (next starts at 4, ends at 8)
    assert_eq!(truncate_utf8(s, 5), "🔥");
    // Limit 8 should include two emojis
    assert_eq!(truncate_utf8(s, 8), "🔥🔥");
    // Limit 1 — no full character fits, but char_indices first is (0, '🔥')
    // take_while(|&i| i <= 1) → only i=0 qualifies
    assert_eq!(truncate_utf8(s, 1), "");
}

#[test]
fn truncate_utf8_mixed() {
    let s = "hi🌍!";
    // 'h'=1, 'i'=1, '🌍'=4, '!'=1 → total 7
    assert_eq!(truncate_utf8(s, 3), "hi");
    assert_eq!(truncate_utf8(s, 6), "hi🌍");
    assert_eq!(truncate_utf8(s, 7), "hi🌍!");
}

#[test]
fn truncate_utf8_empty() {
    assert_eq!(truncate_utf8("", 10), "");
    assert_eq!(truncate_utf8("", 0), "");
}

/// A synthetic ended run record.
pub(super) fn ended_run(
    phase: RunPhase,
    // The folders of steps no receipt confirmed (each a file write).
    unconfirmed: &[&str],
    first_step: Option<i64>,
    finished: i64,
    waited: u64,
) -> crate::run_state::RunState {
    let start = chrono::DateTime::parse_from_rfc3339("2026-11-13T02:05:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    crate::run_state::RunState {
        id: "background_spawn:synthetic".to_string(),
        chat: "telegram:synthetic-chat".to_string(),
        label: "Slow count".to_string(),
        phase,
        requested_at: None,
        started_at: start,
        ended_at: Some(start + chrono::Duration::seconds(finished)),
        work_began: first_step.map(|secs| start + chrono::Duration::seconds(secs)),
        waited: Duration::from_secs(waited),
        step: None,
        activity: None,
        agent: None,
        edited: Vec::new(),
        continues: None,
        thread: None,
        keeps_session: false,
        result: None,
        asked: None,
        steps: Vec::new(),
        evidence: crate::run_state::RunEvidence {
            unconfirmed: unconfirmed
                .iter()
                .map(|path| crate::run_state::UnconfirmedStep {
                    effect: aidaemon_tools::semantics::ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
                    target: Some(path.to_string()),
                })
                .collect(),
            ..Default::default()
        },
        failure: (phase == RunPhase::Failed).then_some(crate::run_state::RunFailure::NoCause),
        card: Default::default(),
        background: true,
        request: None,
        made_folders: Vec::new(),
    }
}

/// The result's header is the run's phase, its one evidence line names
/// only typed action steps nothing verified, and its footer is the measured
/// time: what the helper wrote is only the body.
#[test]
fn a_runs_result_renders_from_its_record() {
    let stopped = render_run_result(
        &ended_run(RunPhase::Stopped, &[], None, 60, 0),
        "WordPress install",
        "The site is running.",
        8_000,
        false,
    );
    assert!(
        stopped.starts_with("\u{26a0}\u{fe0f} Didn't finish in the background: WordPress install\n\nThe site is running."),
        "{stopped}"
    );
    assert!(!stopped.contains("Finished in the background"));

    let succeeded = render_run_result(
        &ended_run(RunPhase::Succeeded, &[], Some(2), 62, 0),
        "Slow count",
        "Done.",
        8_000,
        false,
    );
    assert!(
        succeeded.starts_with("\u{2705} Finished in the background: Slow count\n\nDone."),
        "{succeeded}"
    );
    assert!(!succeeded.contains("confirm"), "{succeeded}");
    assert!(succeeded.contains("Took 1m"), "{succeeded}");

    let unconfirmed = render_run_result(
        &ended_run(RunPhase::Succeeded, &["/srv/site/shots"], None, 60, 0),
        "Site install",
        "Done.",
        8_000,
        false,
    );
    // The outcome leads and the report follows; what only the report
    // vouches for is said after it (live 2026-11-13: "Its report is the
    // only record of 2 steps" was the result's first line).
    assert!(
        unconfirmed.starts_with(
            "\u{2705} Finished in the background: Site install\n\nDone.\n\n\
             Only the report above shows the files it wrote in /srv/site/shots."
        ),
        "{unconfirmed}"
    );
}

/// How a run ended is said once: when its card shows it, the result
/// carries the body and the measured time, never the card's header
/// again (live 2026-11-13: "Finished in the background: X" on the card
/// and again on the result).
#[test]
fn the_result_never_repeats_the_cards_header() {
    for phase in [RunPhase::Succeeded, RunPhase::Stopped, RunPhase::Failed] {
        let run = ended_run(phase, &["/srv/site/shots"], Some(2), 62, 0);
        let message = render_run_result(&run, "Slow count", "Counted to 30.", 8_000, true);
        for header in [phase.headline(), phase.helper_headline(false)] {
            assert!(!message.contains(header), "{header:?} in {message:?}");
        }
        assert!(!message.contains("Slow count"), "{message}");
        // A failed run leads with why it failed, from its record; any
        // other's leads with its report, and what that report alone
        // records follows it.
        match phase {
            RunPhase::Failed => {
                let reason = run.reason().expect("a failed run has its reason");
                assert!(
                    message.starts_with(&format!("{reason}\n\nCounted to 30.")),
                    "{message}"
                );
            }
            _ => assert!(
                message.starts_with(
                    "Counted to 30.\n\nOnly the report above shows the files it \
                     wrote in /srv/site/shots."
                ),
                "{message}"
            ),
        }
        assert!(message.contains("Took 1m"), "{message}");
    }
}

/// A helper's title, and each plan item, names the folder the work
/// creates, so the owner's first "Started in the background" lines say
/// where it goes (live 2026-11-13: the new -6 folders showed up first in
/// a status reply).
#[tokio::test]
async fn titles_and_plan_items_name_the_folder_the_work_creates() {
    let tool = SpawnAgentTool::new_deferred(8000, 300);
    let schema = tool.schema();
    let title = schema["parameters"]["properties"]["title"]["description"]
        .as_str()
        .unwrap();
    assert!(
        title.contains("when the work creates or works in a folder, that folder's name"),
        "{title}"
    );
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let plans = aidaemon_work::tools::track_requirements::TrackRequirementsTool::new(Arc::new(
        aidaemon_work::plans::PlanStore::new(pool).await.unwrap(),
    ));
    assert!(Tool::description(&plans).contains("name the folder or file an item creates"));
}

#[test]
fn deferred_initialization_not_set() {
    let tool = SpawnAgentTool::new_deferred(8000, 300);
    let result = tool.get_agent();
    assert!(result.is_err());
    assert!(result.err().unwrap().to_string().contains("not set"));
}

#[test]
fn deferred_hub_not_set() {
    let tool = SpawnAgentTool::new_deferred(8000, 300);
    assert!(tool.get_hub().is_none());
}

#[test]
fn spawn_args_background_default() {
    let json = r#"{"mission": "test", "task": "do stuff"}"#;
    let args: SpawnArgs = serde_json::from_str(json).unwrap();
    assert!(!args.background);
}

#[test]
fn spawn_args_background_true() {
    let json = r#"{"mission": "test", "task": "do stuff", "background": true}"#;
    let args: SpawnArgs = serde_json::from_str(json).unwrap();
    assert!(args.background);
}

fn spawn_args(extra: serde_json::Value) -> SpawnArgs {
    let mut args = serde_json::json!({"mission": "m", "task": "t"});
    args.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::from_value(args).unwrap()
}

#[test]
fn a_child_is_narrowed_only_when_asked() {
    assert!(
        requested_child_authority(&spawn_args(serde_json::json!({})))
            .unwrap()
            .is_none()
    );

    let requested =
        requested_child_authority(&spawn_args(serde_json::json!({"access": "read_only"})))
            .unwrap()
            .unwrap();
    assert_eq!(requested.level, aidaemon_core::AccessLevel::ReadOnly);
    assert_eq!(requested.tools, None);

    let requested = requested_child_authority(&spawn_args(
        serde_json::json!({"tools": ["read_file", " search_files "]}),
    ))
    .unwrap()
    .unwrap();
    assert_eq!(requested.level, aidaemon_core::AccessLevel::Write);
    assert_eq!(
        requested.tools.unwrap().into_iter().collect::<Vec<_>>(),
        ["read_file", "search_files"]
    );

    assert!(requested_child_authority(&spawn_args(serde_json::json!({"tools": [" "]}))).is_err());
}

#[test]
fn asking_for_more_than_the_parent_has_gives_nothing_extra() {
    let parent = aidaemon_core::Authority::read_only();
    let requested = requested_child_authority(&spawn_args(serde_json::json!({"access": "write"})))
        .unwrap()
        .unwrap();
    assert_eq!(
        parent.narrow(&requested).level,
        aidaemon_core::AccessLevel::ReadOnly
    );
}

/// The parent turn's authority reaches spawn_agent only through the
/// dispatcher's typed facts; an argument of the same name is ignored.
#[test]
fn parent_authority_comes_from_the_call_facts() {
    let parent = crate::authority::InheritedAuthority {
        authority: aidaemon_core::Authority::read_only(),
        owner_level: Some(crate::authority::OwnerLevel::Full),
        task_access: None,
        mode_session: Some("tg:123".to_string()),
        untrusted_context: true,
        origin_request: Some("Summarize the synthetic notes.".to_string()),
        origin_trusted: true,
        trusted_roots: Vec::new(),
        declined: vec![aidaemon_core::EffectKey::egress(
            "https",
            "h.example.invalid",
            None,
        )],
        shared_declines: None,
        learning: None,
        spawn_family: None,
        child_declaration: None,
        parent_run: None,
        known_resources: Default::default(),
        card_title: None,
        card_steps: Vec::new(),
        lineage: Vec::new(),
        parent_task: None,
    };
    let mut facts = ToolCallContext::default();
    assert!(crate::authority::parent_authority(&facts).is_none());
    facts.extensions.insert(parent.clone());
    assert_eq!(crate::authority::parent_authority(&facts), Some(&parent));
    // The model cannot supply one: SpawnArgs has no such argument.
    let json = serde_json::json!({
        "mission": "test",
        "task": "do stuff",
        "_parent_authority": parent,
    });
    assert!(serde_json::from_value::<SpawnArgs>(json).is_ok());
}

#[test]
fn child_channel_context_takes_the_call_visibility() {
    let facts = ToolCallContext {
        channel_visibility: Some(ChannelVisibility::Public),
        ..Default::default()
    };
    assert_eq!(
        build_child_channel_context(&facts).visibility,
        ChannelVisibility::Public
    );
    assert_eq!(
        build_child_channel_context(&ToolCallContext::default()).visibility,
        ChannelVisibility::Internal
    );
}

#[test]
fn child_channel_context_takes_the_call_trust() {
    let facts = ToolCallContext {
        trusted_session: true,
        channel_visibility: Some(ChannelVisibility::Internal),
        ..Default::default()
    };
    let channel_ctx = build_child_channel_context(&facts);
    assert_eq!(channel_ctx.visibility, ChannelVisibility::Internal);
    assert!(channel_ctx.trusted);
    assert!(!build_child_channel_context(&ToolCallContext::default()).trusted);
}

#[tokio::test]
async fn executor_task_lock_deduplicates_concurrent_spawns() {
    let tool = SpawnAgentTool::new_deferred(8000, 300);
    assert!(tool.try_begin_executor_task("task-1").await);
    assert!(
        !tool.try_begin_executor_task("task-1").await,
        "Second acquire should be rejected while first is active"
    );
    tool.finish_executor_task("task-1").await;
    assert!(
        tool.try_begin_executor_task("task-1").await,
        "Task lock should be reusable after release"
    );
}

#[tokio::test]
async fn background_notification_reaches_the_queue() {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    let embedding_service = Arc::new(EmbeddingService::deterministic());
    let state = Arc::new(
        SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
            .await
            .unwrap(),
    );
    let state_dyn: Arc<dyn StateStore> = state.clone();

    enqueue_background_notification(
        Some(&(state_dyn.clone() as Arc<dyn crate::stores::NotificationStore>)),
        "goal_spawn_test",
        "sess_spawn_test",
        "completed",
        "Background task complete.\nMission: test",
        "spawn_test",
        false,
    )
    .await;

    let pending = state.get_pending_notifications(10).await.unwrap();
    assert!(pending.iter().any(|entry| {
        entry.goal_id == "goal_spawn_test"
            && entry.session_id == "sess_spawn_test"
            && entry.notification_type == "completed"
            && entry.message.contains("Mission: test")
    }));
}
