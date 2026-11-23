//! `manage_automations` over reminders, scheduled work, watches and health
//! checks in a scratch state store.

use super::*;
use crate::stores::StateStore;
use aidaemon_core::ChannelVisibility;
use aidaemon_tools::semantics::ToolCallContext;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::testing::CallWithFacts;
use aidaemon_tools::testing::ToolCalls as _;
use aidaemon_work::automations;
use aidaemon_work::tools::manage_mandates::ManageMandatesTool;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;

fn owner() -> ToolCallContext {
    ToolCallContext {
        session_id: "owner-session".into(),
        user_role: Some(UserRole::Owner),
        channel_visibility: Some(ChannelVisibility::Private),
        ..Default::default()
    }
}

fn approved_owner() -> ToolCallContext {
    ToolCallContext {
        kernel_approved: Some(aidaemon_tools::semantics::ApprovalScope::Call),
        ..owner()
    }
}

struct Fixture {
    state: Arc<dyn StateStore>,
    tool: ManageAutomationsTool,
    /// Scheduled automations by name, recorded when created.
    scheduled: std::sync::Mutex<std::collections::HashMap<String, String>>,
    /// Keeps the test database alive.
    _harness: Harness,
}

/// The state the daemon's test agent is built on: a scratch state store
/// with the event store's tables on the same database.
struct Harness {
    state: Arc<crate::state::SqliteStateStore>,
    _db_file: tempfile::NamedTempFile,
}

async fn harness() -> Harness {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().to_str().unwrap().to_string();
    let embeddings = Arc::new(aidaemon_memory::embeddings::EmbeddingService::deterministic());
    let state = Arc::new(
        crate::state::SqliteStateStore::for_tests(&db_path, 100, None, embeddings)
            .await
            .unwrap(),
    );
    aidaemon_events::EventStore::on_paths(
        state.pool(),
        aidaemon_tools::LexicalPaths::daemon_host(),
        Arc::new(crate::history::HistoryEventHost),
    )
    .await
    .unwrap();
    Harness {
        state,
        _db_file: db_file,
    }
}

async fn fixture() -> Fixture {
    let harness = harness().await;
    let state: Arc<dyn StateStore> = harness.state.clone();
    let tool = ManageAutomationsTool::new(state.clone());
    Fixture {
        state,
        tool,
        scheduled: Default::default(),
        _harness: harness,
    }
}

impl Fixture {
    async fn reminder(&self, name: &str, when: &str) -> String {
        let output = crate::tools::set_reminder::SetReminderTool::new(self.state.clone())
            .call_with(
                &json!({"message": "Check the oven.", "when": when, "name": name}).to_string(),
                owner(),
            )
            .await
            .unwrap();
        self.record(name).await;
        output
    }

    /// Remember the goal now carrying `name`, if one does.
    async fn record(&self, name: &str) {
        let found = self
            .state
            .get_scheduled_goals()
            .await
            .unwrap()
            .into_iter()
            .find(|goal| {
                automations::name_from_context(goal.context.as_deref()).as_deref() == Some(name)
            });
        if let Some(goal) = found {
            self.scheduled
                .lock()
                .unwrap()
                .insert(name.to_string(), goal.id);
        }
    }

    async fn scheduled_work(&self, name: &str, schedule: &str) -> String {
        let output = crate::tools::schedule_task::ScheduleTaskTool::new(self.state.clone())
            .call_with(
                &json!({
                    "task": "Summarize the synthetic inbox",
                    "when": schedule,
                    "name": name,
                })
                .to_string(),
                approved_owner(),
            )
            .await
            .unwrap();
        self.record(name).await;
        output
    }

    async fn watching(&self, name: &str) -> String {
        ManageMandatesTool::new(self.state.clone())
            .call_with(
                &json!({
                    "action": "create",
                    "name": name,
                    "objective": "Watch the synthetic status page",
                    "max_mutating_actions_per_cycle": 0,
                    "min_review_minutes": 60,
                    "max_review_minutes": 60,
                    "default_review_minutes": 60,
                    "success_criteria": ["The status page is reviewed without unauthorized action"],
                })
                .to_string(),
                approved_owner(),
            )
            .await
            .unwrap()
    }

    async fn call(&self, args: Value) -> String {
        self.tool
            .call_with(&args.to_string(), owner())
            .await
            .unwrap()
    }

    async fn scheduled_status(&self, name: &str) -> String {
        let id = self.scheduled.lock().unwrap().get(name).cloned();
        let id = id.unwrap_or_else(|| panic!("no scheduled automation named {name}"));
        self.state.get_goal(&id).await.unwrap().unwrap().status
    }

    async fn mandate_status(&self, name: &str) -> String {
        self.state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .into_iter()
            .find(|mandate| mandate.has_name(name))
            .map(|mandate| mandate.status.as_str().to_string())
            .unwrap_or_default()
    }
}

#[tokio::test]
async fn one_list_shows_both_kinds_with_cadence_state_and_next_run() {
    let f = fixture().await;
    assert!(
        f.reminder("Oven", "every weekday at 8am")
            .await
            .starts_with("Reminder set")
    );
    let work = f.scheduled_work("Inbox digest", "every day at 9am").await;
    assert!(work.starts_with("Scheduled \"Inbox digest\""), "{work}");
    let watch = f.watching("Status watch").await;
    assert!(watch.contains("Activated mandate"), "{watch}");

    let list = f.call(json!({"action": "list"})).await;
    assert!(list.starts_with("Your automations (3):"), "{list}");
    for row in [
        "\"Oven\" · Scheduled · Every weekday at 8:00 AM",
        "\"Inbox digest\" · Scheduled · Every day at 9:00 AM",
        "\"Status watch\" · Watching · Every hour · On · next ",
    ] {
        assert!(list.contains(row), "missing {row:?} in {list}");
    }
    assert!(
        !list.contains("mandate") && !list.contains("goal"),
        "{list}"
    );

    // The typed kind narrows the list.
    let watching = f.call(json!({"action": "list", "kind": "watching"})).await;
    assert!(
        watching.starts_with("Your watching automations (1):"),
        "{watching}"
    );
    assert!(!watching.contains("Oven"), "{watching}");
}

#[tokio::test]
async fn show_gives_the_same_five_line_card_for_both_kinds() {
    let f = fixture().await;
    f.scheduled_work("Inbox digest", "every day at 9am").await;
    f.watching("Status watch").await;

    let scheduled = f
        .call(json!({"action": "show", "name": "inbox digest"}))
        .await;
    for line in [
        "\"Inbox digest\" (Scheduled)",
        "What: Summarize the synthetic inbox",
        "How often: Every day at 9:00 AM",
        "Can do: Anything you can do – it runs with your full access",
        "Tells you when: a run finishes, or it hits a problem",
        "Ends: Only when you stop it",
        "State: On · next ",
    ] {
        assert!(scheduled.contains(line), "missing {line:?} in {scheduled}");
    }
    let watching = f
        .call(json!({"action": "show", "name": "Status watch"}))
        .await;
    for line in [
        "\"Status watch\" (Watching)",
        "What: Watch the synthetic status page",
        "How often: Every hour",
        "Can do: Read only – makes no changes",
        "Tells you when: ",
        "Ends: Only when you stop it",
    ] {
        assert!(watching.contains(line), "missing {line:?} in {watching}");
    }
}

#[tokio::test]
async fn cancel_by_name_works_for_each_kind() {
    let f = fixture().await;
    f.reminder("Oven", "every weekday at 8am").await;
    f.watching("Status watch").await;

    let cancelled = f.call(json!({"action": "cancel", "name": "oven"})).await;
    assert_eq!(
        cancelled,
        "Cancelled \"Oven\" (scheduled). It won't run again."
    );
    assert_eq!(f.scheduled_status("Oven").await, "cancelled");

    let cancelled = f
        .call(json!({"action": "cancel", "name": "Status watch"}))
        .await;
    assert_eq!(
        cancelled,
        "Cancelled \"Status watch\" (watching). It won't run again."
    );
    assert_eq!(f.mandate_status("Status watch").await, "cancelled");

    let list = f.call(json!({"action": "list"})).await;
    assert_eq!(list, "Your automations: none.");
    let missing = f.call(json!({"action": "pause", "name": "Oven"})).await;
    assert!(
        missing.starts_with("No automation is named \"Oven\""),
        "{missing}"
    );
}

#[tokio::test]
async fn a_client_lists_every_automation_and_changes_one_by_id() {
    let f = fixture().await;
    f.reminder("Oven", "every weekday at 8am").await;
    f.watching("Status watch").await;
    let entries = f.tool.entries().await.unwrap();
    assert_eq!(entries.len(), 2, "{entries:?}");
    let oven = entries
        .iter()
        .find(|e| e.name.as_deref() == Some("Oven"))
        .unwrap()
        .clone();
    assert_eq!(oven.kind, automations::AutomationKind::Scheduled);
    assert!(oven.how_often.starts_with("Every weekday at 8:00 AM"));
    assert!(oven.next.is_some());
    // Watching ones of every session, changed in their own session.
    let watch = entries
        .iter()
        .find(|e| e.name.as_deref() == Some("Status watch"))
        .unwrap()
        .clone();
    let (changed, said) = f.tool.owner_change(&watch.id, "pause").await.unwrap();
    assert!(changed, "{said}");
    assert_eq!(f.mandate_status("Status watch").await, "paused");
    let (changed, _) = f.tool.owner_change(&watch.id, "resume").await.unwrap();
    assert!(changed);
    // Only scheduled work runs now; an unknown change is refused.
    let (changed, said) = f.tool.owner_change(&watch.id, "run_now").await.unwrap();
    assert!(
        !changed && said.starts_with("Only scheduled work"),
        "{said}"
    );
    let (changed, _) = f.tool.owner_change(&oven.id, "edit").await.unwrap();
    assert!(!changed);
    let (changed, said) = f.tool.owner_change(&oven.id, "cancel").await.unwrap();
    assert!(changed, "{said}");
    assert_eq!(f.scheduled_status("Oven").await, "cancelled");
    let (changed, said) = f.tool.owner_change(&oven.id, "pause").await.unwrap();
    assert!(!changed && said == "That automation is gone.", "{said}");
}

/// A client sees what each automation did: its latest runs, newest
/// first, with what they said; and one whose runs kept failing reads as
/// paused after failures, not as on with no next run (research
/// 2026-11-15: silent failure is what owners complain about most).
#[tokio::test]
async fn a_client_sees_recent_runs_and_an_automation_held_after_failures() {
    let f = fixture().await;
    f.scheduled_work("Repo check", "every day at 8am").await;
    let goal_id = f.scheduled.lock().unwrap()["Repo check"].clone();
    let entries = f.tool.entries().await.unwrap();
    assert!(entries[0].runs.is_empty());
    assert_eq!(entries[0].status, "active");
    for attempt in 1..=3 {
        let run = f
            .state
            .start_goal_run(&goal_id, "manual", None, None)
            .await
            .unwrap();
        assert!(
            f.state
                .finish_goal_run(
                    &run.id,
                    "failed",
                    Some(&format!("The synthetic token expired ({attempt}).")),
                )
                .await
                .unwrap()
        );
    }
    let entry = f.tool.entries().await.unwrap().remove(0);
    assert_eq!(entry.status, automations::HELD_AFTER_FAILURES);
    assert!(entry.next.is_none(), "a held automation has no next run");
    assert_eq!(entry.runs.len(), 3);
    assert_eq!(entry.runs[0].status, "failed");
    assert_eq!(
        entry.runs[0].summary.as_deref(),
        Some("The synthetic token expired (3).")
    );
    assert_eq!(
        automations::state_words(&entry.status),
        "Paused after repeated failures"
    );
}

/// A run the daemon missed (it was off when the run was due) is in the
/// automation's recent runs as `missed`, with what happened; three
/// misses in a row do not hold it after failures.
#[tokio::test]
async fn a_client_sees_a_missed_run_and_misses_never_hold_an_automation() {
    let f = fixture().await;
    f.scheduled_work("Repo check", "every day at 8am").await;
    let goal_id = f.scheduled.lock().unwrap()["Repo check"].clone();
    let due = (chrono::Utc::now() - chrono::Duration::hours(26)).to_rfc3339();
    for _ in 0..3 {
        f.state
            .record_missed_goal_run(
                &goal_id,
                None,
                &due,
                "The daemon wasn't running when this was due.",
            )
            .await
            .unwrap();
    }
    let entry = f.tool.entries().await.unwrap().remove(0);
    assert_eq!(entry.status, "active", "misses are not failures");
    assert!(entry.next.is_some(), "it still has a next run");
    assert_eq!(entry.runs.len(), 3);
    assert_eq!(entry.runs[0].status, aidaemon_work::goals::GOAL_RUN_MISSED);
    assert_eq!(
        entry.runs[0].summary.as_deref(),
        Some("The daemon wasn't running when this was due.")
    );
}

/// A client sees an automation's details and changes scheduled work in
/// place, as others' Scheduled pages do: name, when (in words) and what
/// its runs do; a reminder's message stays fixed, and a watching
/// automation changes by asking (research 2026-11-15).
#[tokio::test]
async fn a_client_edits_scheduled_work_in_place() {
    let f = fixture().await;
    f.scheduled_work("Inbox digest", "every day at 9am").await;
    f.reminder("Oven", "every weekday at 8am").await;
    f.watching("Status watch").await;
    let entries = f.tool.entries().await.unwrap();
    let find = |entries: &[AutomationEntry], name: &str| {
        entries
            .iter()
            .find(|e| e.name.as_deref() == Some(name))
            .unwrap()
            .clone()
    };
    let digest = find(&entries, "Inbox digest");
    let edit = digest.edit.clone().expect("scheduled work is editable");
    assert_eq!(edit.when, "every day at 9am");
    assert_eq!(
        edit.instructions.as_deref(),
        Some("Summarize the synthetic inbox")
    );
    assert!(
        digest
            .details
            .iter()
            .any(|line| line.starts_with("Can do: "))
    );
    // What it spends is part of its details (its limits, today's use).
    assert!(
        digest
            .details
            .iter()
            .any(|line| line.starts_with("Spend: ")),
        "{:?}",
        digest.details
    );
    assert!(find(&entries, "Oven").edit.unwrap().instructions.is_none());
    assert!(find(&entries, "Status watch").edit.is_none());

    let (changed, said) = f
        .tool
        .owner_edit(
            &digest.id,
            Some("Morning digest"),
            Some("every weekday at 8am"),
            Some("Summarize the synthetic inbox; skip newsletters"),
            None,
        )
        .await
        .unwrap();
    assert!(changed, "{said}");
    let after = find(&f.tool.entries().await.unwrap(), "Morning digest");
    assert!(
        after.how_often.starts_with("Every weekday at 8:00 AM"),
        "{}",
        after.how_often
    );
    assert_eq!(
        after.edit.as_ref().unwrap().instructions.as_deref(),
        Some("Summarize the synthetic inbox; skip newsletters")
    );
    assert_eq!(
        after.what,
        "Summarize the synthetic inbox; skip newsletters"
    );
    // Quiet unless something changes: the card says so.
    assert_eq!(
        after.edit.as_ref().and_then(|e| e.notify),
        Some(automations::NotifyMode::EveryRun)
    );
    let (changed, said) = f
        .tool
        .owner_edit(
            &after.id,
            None,
            None,
            None,
            Some(automations::NotifyMode::Changes),
        )
        .await
        .unwrap();
    assert!(changed, "{said}");
    let quiet = find(&f.tool.entries().await.unwrap(), "Morning digest");
    assert_eq!(
        quiet.edit.as_ref().and_then(|e| e.notify),
        Some(automations::NotifyMode::Changes)
    );
    assert!(
        quiet
            .details
            .iter()
            .any(|line| line
                == "Tells you when: only when a run finds something new, or hits a problem"),
        "{:?}",
        quiet.details
    );

    // Unchanged fields change nothing; a reminder's message is fixed;
    // a watching automation is changed by asking.
    let (changed, said) = f
        .tool
        .owner_edit(&after.id, Some("Morning digest"), None, None, None)
        .await
        .unwrap();
    assert!(!changed && said == "Nothing changed.", "{said}");
    let oven = find(&f.tool.entries().await.unwrap(), "Oven");
    let (changed, _) = f
        .tool
        .owner_edit(
            &oven.id,
            None,
            None,
            Some("Check the synthetic stove"),
            None,
        )
        .await
        .unwrap();
    assert!(!changed);
    let watch = find(&f.tool.entries().await.unwrap(), "Status watch");
    let (changed, said) = f
        .tool
        .owner_edit(&watch.id, Some("Renamed"), None, None, None)
        .await
        .unwrap();
    assert!(!changed && said.contains("ask in a conversation"), "{said}");
}

#[tokio::test]
async fn pause_and_resume_by_name_round_trip_for_each_kind() {
    let f = fixture().await;
    f.scheduled_work("Inbox digest", "every day at 9am").await;
    f.watching("Status watch").await;
    for (name, kind) in [("Inbox digest", "scheduled"), ("Status watch", "watching")] {
        let paused = f.call(json!({"action": "pause", "name": name})).await;
        assert_eq!(
            paused,
            format!("Paused \"{name}\" ({kind}). It won't run until you resume it.")
        );
        let again = f.call(json!({"action": "pause", "name": name})).await;
        assert_eq!(
            again,
            format!("Couldn't pause \"{name}\" ({kind}): it is Paused.")
        );
        let resumed = f.call(json!({"action": "resume", "name": name})).await;
        assert!(
            resumed.starts_with(&format!("Resumed \"{name}\" ({kind}).")),
            "{resumed}"
        );
    }
    assert_eq!(f.scheduled_status("Inbox digest").await, "active");
    assert_eq!(f.mandate_status("Status watch").await, "active");
}

#[tokio::test]
async fn the_typed_kind_picks_between_two_automations_with_one_name() {
    let f = fixture().await;
    f.reminder("Prices", "every weekday at 8am").await;
    f.watching("Prices").await;

    let ambiguous = f.call(json!({"action": "pause", "name": "Prices"})).await;
    assert!(
        ambiguous.starts_with("More than one automation matches"),
        "{ambiguous}"
    );
    assert_eq!(f.scheduled_status("Prices").await, "active");
    assert_eq!(f.mandate_status("Prices").await, "active");

    let paused = f
        .call(json!({"action": "pause", "name": "Prices", "kind": "watching"}))
        .await;
    assert!(
        paused.starts_with("Paused \"Prices\" (watching)"),
        "{paused}"
    );
    assert_eq!(f.mandate_status("Prices").await, "paused");
    assert_eq!(f.scheduled_status("Prices").await, "active");

    let cancelled = f
        .call(json!({"action": "cancel", "name": "Prices", "kind": "scheduled"}))
        .await;
    assert!(
        cancelled.starts_with("Cancelled \"Prices\" (scheduled)"),
        "{cancelled}"
    );
    assert_eq!(f.mandate_status("Prices").await, "paused");
}

#[tokio::test]
async fn a_name_already_in_use_is_refused_and_unnamed_ones_go_by_ref() {
    let f = fixture().await;
    f.watching("Oven").await;
    let refused = f.reminder("oven", "in 2h").await;
    assert!(
        refused.starts_with("An automation is already named \"oven\" (a watching one)"),
        "{refused}"
    );

    let set = crate::tools::set_reminder::SetReminderTool::new(f.state.clone())
        .call_with(
            &json!({"message": "Stretch.", "when": "every day at 3pm"}).to_string(),
            owner(),
        )
        .await
        .unwrap();
    assert!(set.starts_with("Reminder set"), "{set}");
    let list = f.call(json!({"action": "list", "kind": "scheduled"})).await;
    let row = list
        .lines()
        .find(|line| line.contains("Reminds you: Stretch."))
        .unwrap_or_else(|| panic!("{list}"));
    let reference = row
        .rsplit_once("[ref ")
        .and_then(|(_, rest)| rest.strip_suffix(']'))
        .unwrap();
    let cancelled = f.call(json!({"action": "cancel", "id": reference})).await;
    assert!(
        cancelled.starts_with("Cancelled Reminds you: Stretch."),
        "{cancelled}"
    );
}

#[tokio::test]
async fn only_the_owner_in_a_private_chat_sees_automations() {
    let f = fixture().await;
    for facts in [
        ToolCallContext {
            user_role: Some(UserRole::Guest),
            ..owner()
        },
        ToolCallContext {
            channel_visibility: Some(ChannelVisibility::Public),
            ..owner()
        },
        ToolCallContext {
            channel_visibility: Some(ChannelVisibility::Internal),
            ..owner()
        },
    ] {
        let answer = f
            .tool
            .call_with(&json!({"action": "list"}).to_string(), facts)
            .await
            .unwrap();
        assert_eq!(answer, OWNER_ONLY);
    }
}

#[tokio::test]
async fn edit_reschedules_renames_and_snoozes_one_scheduled_automation() {
    let f = fixture().await;
    f.scheduled_work("Inbox digest", "every day at 9am").await;

    let edited = f
        .call(json!({"action": "edit", "name": "Inbox digest", "when": "every weekday at 7am", "new_name": "Morning digest"}))
        .await;
    assert!(
        edited.starts_with("Edited \"Inbox digest\": renamed it \"Morning digest\"; its schedule is now Every weekday at 7:00 AM"),
        "{edited}"
    );
    let id = f
        .scheduled
        .lock()
        .unwrap()
        .get("Inbox digest")
        .cloned()
        .unwrap();
    let schedules = f.state.get_schedules_for_goal(&id).await.unwrap();
    assert_eq!(schedules.len(), 1);
    assert_eq!(schedules[0].cron_expr, "0 7 * * 1-5");
    let shown = f
        .call(json!({"action": "show", "name": "morning digest"}))
        .await;
    assert!(
        shown.contains("How often: Every weekday at 7:00 AM"),
        "{shown}"
    );

    let snoozed = f
        .call(json!({"action": "edit", "name": "Morning digest", "snooze_until": "in 3 days"}))
        .await;
    assert!(snoozed.contains("its next run moved to"), "{snoozed}");
    let schedule = f.state.get_schedules_for_goal(&id).await.unwrap().remove(0);
    // The cadence stays; only the next run moved.
    assert_eq!(schedule.cron_expr, "0 7 * * 1-5");
    let next = chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at).unwrap();
    assert!(
        next > chrono::Utc::now() + chrono::Duration::days(2),
        "{next}"
    );

    // A recurring snooze, a one-time reschedule of recurring work and a
    // name already taken are refused, and change nothing.
    for (args, refusal) in [
        (
            json!({"action": "edit", "name": "Morning digest", "snooze_until": "every day at 8am"}),
            "`snooze_until` is one time",
        ),
        (
            json!({"action": "edit", "name": "Morning digest", "when": "tomorrow at 9am"}),
            "recurs; to make it a one-time one",
        ),
    ] {
        let answer = f.call(args).await;
        assert!(answer.contains(refusal), "{answer}");
    }
    f.watching("Status watch").await;
    let taken = f
        .call(json!({"action": "edit", "name": "Morning digest", "new_name": "Status watch"}))
        .await;
    assert!(
        taken.starts_with("An automation is already named"),
        "{taken}"
    );
    assert_eq!(
        f.state.get_schedules_for_goal(&id).await.unwrap()[0].cron_expr,
        "0 7 * * 1-5"
    );
}

#[tokio::test]
async fn several_refs_cancel_together_and_nothing_cancels_by_matching_words() {
    let f = fixture().await;
    f.reminder("Oven", "every weekday at 8am").await;
    f.reminder("Stretch", "every day at 3pm").await;
    f.reminder("Water", "every day at 4pm").await;
    let refs: Vec<String> = ["Oven", "Stretch"]
        .iter()
        .map(|name| f.scheduled.lock().unwrap().get(*name).cloned().unwrap()[..8].to_string())
        .collect();
    let cancelled = f.call(json!({"action": "cancel", "ids": refs})).await;
    assert_eq!(cancelled.lines().count(), 2, "{cancelled}");
    assert_eq!(f.scheduled_status("Oven").await, "cancelled");
    assert_eq!(f.scheduled_status("Stretch").await, "cancelled");
    assert_eq!(f.scheduled_status("Water").await, "active");
    // A word from a name is not a ref.
    let outcome = f
        .tool
        .call_with_execution_context(
            &json!({"action": "cancel", "ids": ["Water"]}).to_string(),
            None,
            owner(),
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.metadata.outcome_status,
        Some(aidaemon_tools::semantics::ToolOutcomeStatus::CompletedWithNegativeResult)
    );
    assert_eq!(f.scheduled_status("Water").await, "active");
}

#[tokio::test]
async fn run_now_starts_a_run_of_named_scheduled_work() {
    let f = fixture().await;
    f.scheduled_work("Inbox digest", "every day at 9am").await;
    let outcome = f
        .tool
        .call_with_execution_context(
            &json!({"action": "run_now", "name": "Inbox digest"}).to_string(),
            None,
            owner(),
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.metadata.outcome_status,
        Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded),
        "{}",
        outcome.output
    );
    let id = f
        .scheduled
        .lock()
        .unwrap()
        .get("Inbox digest")
        .cloned()
        .unwrap();
    assert_eq!(f.state.get_tasks_for_goal(&id).await.unwrap().len(), 1);
    // A paused one does not run, and that is no success.
    f.call(json!({"action": "pause", "name": "Inbox digest"}))
        .await;
    let refused = f
        .tool
        .call_with_execution_context(
            &json!({"action": "run_now", "id": id}).to_string(),
            None,
            owner(),
        )
        .await
        .unwrap();
    assert_ne!(
        refused.metadata.outcome_status,
        Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded),
        "{}",
        refused.output
    );
}

#[tokio::test]
async fn scheduled_history_keeps_its_existing_output() {
    let f = fixture().await;
    f.scheduled_work("Repo check", "every day at 8am").await;
    let goal_id = f.scheduled.lock().unwrap()["Repo check"].clone();
    let run = f
        .state
        .start_goal_run(&goal_id, "manual", None, None)
        .await
        .unwrap();
    f.state
        .finish_goal_run(&run.id, "completed", Some("Synthetic repository is clean."))
        .await
        .unwrap();

    let history = f
        .call(json!({"action": "history", "name": "Repo check"}))
        .await;
    assert!(
        history.starts_with("**Scheduled Run History**"),
        "{history}"
    );
    assert!(
        history.contains("- Goal: Summarize the synthetic inbox"),
        "{history}"
    );
    assert!(
        history.contains("Synthetic repository is clean."),
        "{history}"
    );
}

#[tokio::test]
async fn watching_history_uses_goal_runs_and_says_decisions_in_words() {
    let f = fixture().await;
    f.watching("Status watch").await;
    let mandate = f
        .state
        .list_mandates(Some("owner-session"), false)
        .await
        .unwrap()
        .into_iter()
        .find(|mandate| mandate.has_name("Status watch"))
        .unwrap();
    let claimed = f
        .state
        .claim_due_mandates(1, "history-test", 300)
        .await
        .unwrap();
    let leased = claimed
        .into_iter()
        .find(|item| item.id == mandate.id)
        .unwrap();
    let root_task = aidaemon_work::goals::Task {
        id: uuid::Uuid::new_v4().to_string(),
        goal_id: mandate.goal_id.clone(),
        description: "Review the synthetic status page".to_string(),
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
        idempotent: false,
        retry_count: 0,
        max_retries: 0,
        created_at: chrono::Utc::now().to_rfc3339(),
        started_at: None,
        completed_at: None,
    };
    let run = f
        .state
        .create_mandate_review_run(
            &mandate.id,
            leased.review_lease_token.as_deref().unwrap(),
            &uuid::Uuid::new_v4().to_string(),
            &root_task,
        )
        .await
        .unwrap();
    f.state
        .claim_task_with_lease(&root_task.id, "history-task-lead", None, 300)
        .await
        .unwrap()
        .unwrap();
    let decision = aidaemon_work::mandates::MandateDecisionCycle::new(
        &mandate.id,
        &run.id,
        aidaemon_work::mandates::MandateDecisionOutcome::Wait,
        "The synthetic status page is healthy, so no intervention is needed.",
        mandate.version,
    );
    f.state
        .record_mandate_decision(&decision, None, None)
        .await
        .unwrap();
    f.state
        .finish_goal_run(&run.id, "completed", Some("mandate_non_action_satisfied"))
        .await
        .unwrap();

    let history = f
        .call(json!({"action": "history", "name": "Status watch", "kind": "watching"}))
        .await;
    assert!(history.starts_with("**Watching History**"), "{history}");
    assert!(
        history.contains(
            "Decision: Wait — The synthetic status page is healthy, so no intervention is needed."
        ),
        "{history}"
    );
    assert!(
        !history.contains("mandate_non_action_satisfied"),
        "{history}"
    );
    assert!(!history.contains("goal_run_id"), "{history}");
}

#[tokio::test]
async fn health_check_history_shows_stored_status_latency_and_error() {
    let f = fixture().await;
    let store = Arc::new(
        aidaemon_work::health::HealthProbeStore::new(f._harness.state.pool())
            .await
            .unwrap(),
    );
    let tool = ManageAutomationsTool::new(f.state.clone()).with_health_probes(Some(store.clone()));
    let probe = aidaemon_work::health::HealthProbe::new(
        "Synthetic site".into(),
        aidaemon_work::health::ProbeType::Http,
        "https://status.example.test/".into(),
        "*/5 * * * *".into(),
        "tool".into(),
    );
    store.upsert_probe(&probe).await.unwrap();
    store
        .insert_result(
            &aidaemon_work::health::ProbeResult::new(
                probe.id.clone(),
                aidaemon_work::health::probes::ProbeStatus::Unhealthy,
            )
            .with_latency(425)
            .with_error("Synthetic upstream timeout"),
        )
        .await
        .unwrap();

    let history = tool
        .call_with(
            &json!({"action": "history", "id": probe.id, "kind": "probe"}).to_string(),
            owner(),
        )
        .await
        .unwrap();
    assert!(history.starts_with("**Health Check History**"), "{history}");
    assert!(history.contains("status: unhealthy"), "{history}");
    assert!(history.contains("latency: 425 ms"), "{history}");
    assert!(
        history.contains("error: Synthetic upstream timeout"),
        "{history}"
    );
}

#[tokio::test]
async fn health_checks_are_listed_paused_and_cancelled_here() {
    let f = fixture().await;
    let store = Arc::new(
        aidaemon_work::health::HealthProbeStore::new(f._harness.state.pool())
            .await
            .unwrap(),
    );
    let tool = ManageAutomationsTool::new(f.state.clone()).with_health_probes(Some(store.clone()));
    let mut probe = aidaemon_work::health::HealthProbe::new(
        "Synthetic site".to_string(),
        aidaemon_work::health::ProbeType::Http,
        "https://status.example.test/".to_string(),
        "*/5 * * * *".to_string(),
        "tool".to_string(),
    );
    probe.next_run_at = chrono::Utc::now() + chrono::Duration::minutes(5);
    store.upsert_probe(&probe).await.unwrap();
    let call = |args: Value| {
        let tool = &tool;
        async move { tool.call_with(&args.to_string(), owner()).await.unwrap() }
    };
    let list = call(json!({"action": "list", "kind": "probe"})).await;
    assert!(
        list.contains("\"Synthetic site\" · Health check · Every 5 minutes · On"),
        "{list}"
    );
    let paused = call(json!({"action": "pause", "name": "synthetic site"})).await;
    assert!(
        paused.starts_with("Paused \"Synthetic site\" (probe)"),
        "{paused}"
    );
    assert!(store.get_probe(&probe.id).await.unwrap().unwrap().is_paused);
    let cancelled = call(json!({"action": "cancel", "name": "Synthetic site"})).await;
    assert!(
        cancelled.starts_with("Cancelled \"Synthetic site\" (probe)"),
        "{cancelled}"
    );
    assert!(store.get_probe(&probe.id).await.unwrap().is_none());
}

#[tokio::test]
async fn health_check_edit_renames_reschedules_and_reads_back() {
    let f = fixture().await;
    let store = Arc::new(
        aidaemon_work::health::HealthProbeStore::new(f._harness.state.pool())
            .await
            .unwrap(),
    );
    let tool = ManageAutomationsTool::new(f.state.clone()).with_health_probes(Some(store.clone()));
    let mut probe = aidaemon_work::health::HealthProbe::new(
        "Synthetic status".into(),
        aidaemon_work::health::ProbeType::Http,
        "https://status.example.test/".into(),
        "*/5 * * * *".into(),
        "tool".into(),
    );
    probe.next_run_at = chrono::Utc::now() + chrono::Duration::minutes(5);
    store.upsert_probe(&probe).await.unwrap();
    let original_next = store
        .get_probe(&probe.id)
        .await
        .unwrap()
        .unwrap()
        .next_run_at;
    let call = |args: Value| {
        let tool = &tool;
        async move { tool.call_with(&args.to_string(), owner()).await.unwrap() }
    };
    let renamed = call(
        json!({"action":"edit", "id":probe.id, "kind":"probe", "new_name":"Synthetic uptime"}),
    )
    .await;
    assert!(renamed.contains("renamed it"), "{renamed}");
    let saved = store.get_probe(&probe.id).await.unwrap().unwrap();
    assert_eq!(saved.name, "Synthetic uptime");
    assert_eq!(saved.next_run_at, original_next);
    let invalid =
        call(json!({"action":"edit", "id":probe.id, "kind":"probe", "when":"not a schedule"}))
            .await;
    assert!(invalid.contains("Couldn't read the schedule"), "{invalid}");
    let changed =
        call(json!({"action":"edit", "id":probe.id, "kind":"probe", "when":"*/10 * * * *"})).await;
    assert!(changed.contains("Next run:"), "{changed}");
    assert_eq!(
        store.get_probe(&probe.id).await.unwrap().unwrap().schedule,
        "*/10 * * * *"
    );
    let entries = tool.entries().await.unwrap();
    let entry = entries.iter().find(|entry| entry.id == probe.id).unwrap();
    assert_eq!(entry.edit.as_ref().unwrap().when, "*/10 * * * *");
    let (ok, message) = tool
        .owner_edit(&probe.id, Some("Synthetic web"), None, None, None)
        .await
        .unwrap();
    assert!(ok, "{message}");
    assert_eq!(
        store.get_probe(&probe.id).await.unwrap().unwrap().name,
        "Synthetic web"
    );
}

#[tokio::test]
async fn config_health_check_edit_stays_configured() {
    let f = fixture().await;
    let store = Arc::new(
        aidaemon_work::health::HealthProbeStore::new(f._harness.state.pool())
            .await
            .unwrap(),
    );
    let tool = ManageAutomationsTool::new(f.state.clone()).with_health_probes(Some(store.clone()));
    let probe = aidaemon_work::health::HealthProbe::new(
        "Configured status".into(),
        aidaemon_work::health::ProbeType::Http,
        "https://status.example.test/".into(),
        "*/5 * * * *".into(),
        "config".into(),
    );
    store.upsert_probe(&probe).await.unwrap();
    let response = tool.call_with(&json!({"action":"edit", "id":probe.id, "kind":"probe", "new_name":"New status", "when":"*/10 * * * *"}).to_string(), owner()).await.unwrap();
    assert!(response.contains("stays as configured"), "{response}");
    let saved = store.get_probe(&probe.id).await.unwrap().unwrap();
    assert_eq!(saved.name, probe.name);
    assert_eq!(saved.schedule, probe.schedule);
    assert!(
        tool.entries()
            .await
            .unwrap()
            .iter()
            .find(|entry| entry.id == probe.id)
            .unwrap()
            .edit
            .is_none()
    );
}

#[tokio::test]
async fn a_moved_schedule_action_names_its_new_home_and_runs_nothing() {
    let f = fixture().await;
    f.reminder("Oven", "every weekday at 8am").await;
    let id = f.scheduled.lock().unwrap().get("Oven").cloned().unwrap();
    let memories = crate::tools::manage_memories::ManageMemoriesTool::new(
        f.state.clone(),
        f._harness.state.memory_stores(),
    );
    let outcome = memories
        .call_with_execution_context(
            &json!({"action": "cancel_scheduled", "goal_id": id}).to_string(),
            None,
            owner(),
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.metadata.outcome_status,
        Some(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent)
    );
    assert!(
        outcome.output.contains("Use manage_automations cancel"),
        "{}",
        outcome.output
    );
    assert_eq!(f.scheduled_status("Oven").await, "active");
}
