//! What one run did, for the app's run view (protocol 1.46, `get_run`): a
//! launched coding agent's invocation and its kept events
//! (`cli_agent_events`), read back into typed protocol items. The events'
//! data is the daemon's own stored form
//! ([`aidaemon_coding_agents::AgentEvent::to_record`] and the run's
//! permission records), a machine format it wrote itself.

use aidaemon_client_protocol::{
    RunDetail, RunPermission, RunPermissionOutcome, RunPlanItem, RunPlanStatus, RunStepItem,
    RunStepState, RunUsage,
};
use aidaemon_tools::services::{CliAgentEvent, CliAgentRun};
use serde_json::Value;

use crate::traits::StateStore;

/// Run `run_id` as its record keeps it. A run with no coding agent's
/// invocation behind it has its id alone.
pub(crate) async fn run_detail(state: &dyn StateStore, run_id: &str) -> anyhow::Result<RunDetail> {
    let Some(run) = state.cli_agent_run(run_id).await? else {
        return Ok(RunDetail {
            run_id: run_id.to_string(),
            agent_session_id: None,
            model: None,
            working_dir: None,
            task: String::new(),
            result: None,
            steps: Vec::new(),
            plan: Vec::new(),
            permissions: Vec::new(),
            usage: None,
        });
    };
    let events = state.get_cli_agent_events(run.invocation).await?;
    Ok(detail(run_id, run, &events))
}

fn detail(run_id: &str, run: CliAgentRun, events: &[CliAgentEvent]) -> RunDetail {
    let mut detail = RunDetail {
        run_id: run_id.to_string(),
        agent_session_id: run.agent_session,
        model: None,
        working_dir: run.working_dir,
        task: run.task.unwrap_or(run.prompt_summary),
        result: run.output_summary.filter(|text| !text.trim().is_empty()),
        steps: Vec::new(),
        plan: Vec::new(),
        permissions: Vec::new(),
        usage: None,
    };
    // Each step's place by the agent's own id, for its end.
    let mut by_id: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for event in events {
        let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        let text = |key: &str| data.get(key).and_then(Value::as_str).map(str::to_string);
        // The stored kinds (`AgentEvent::to_record`, the permission record).
        match event.kind.as_str() {
            "session" => {
                if detail.agent_session_id.is_none() {
                    detail.agent_session_id = text("id");
                }
                // The model it reported working with (its latest word).
                if let Some(model) = text("model").filter(|model| !model.trim().is_empty()) {
                    detail.model = Some(model);
                }
            }
            "step" => {
                if let Some(id) = text("id") {
                    by_id.insert(id, detail.steps.len());
                }
                detail.steps.push(RunStepItem {
                    words: text("words").unwrap_or_default(),
                    state: RunStepState::Started,
                    at_ms: at_ms(&event.at),
                    by_helper: data.get("parent").is_some_and(|parent| !parent.is_null()),
                });
            }
            "step_ended" => {
                if let Some(step) = text("id")
                    .and_then(|id| by_id.get(&id).copied())
                    .and_then(|index| detail.steps.get_mut(index))
                {
                    step.state = if data.get("failed").and_then(Value::as_bool) == Some(true) {
                        RunStepState::Failed
                    } else {
                        RunStepState::Done
                    };
                }
            }
            "plan" => {
                detail.plan = data
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|item| {
                        Some(RunPlanItem {
                            text: item.get("text")?.as_str()?.to_string(),
                            status: match item.get("status").and_then(Value::as_str) {
                                Some("pending") => RunPlanStatus::Pending,
                                Some("in_progress") => RunPlanStatus::InProgress,
                                Some("done") => RunPlanStatus::Done,
                                _ => RunPlanStatus::Unknown,
                            },
                        })
                    })
                    .collect();
            }
            "usage" => {
                let count = |key: &str| data.get(key).and_then(Value::as_u64).unwrap_or(0);
                detail.usage = Some(RunUsage {
                    input_tokens: count("input_tokens"),
                    output_tokens: count("output_tokens"),
                    cost_usd: data.get("cost_usd").and_then(Value::as_f64),
                });
            }
            "permission" => detail.permissions.push(RunPermission {
                label: text("label").unwrap_or_default(),
                outcome: match text("outcome").as_deref() {
                    Some("allowed") => RunPermissionOutcome::Allowed,
                    Some("refused") => RunPermissionOutcome::Refused,
                    Some("declined") => RunPermissionOutcome::Declined,
                    Some("unanswered") => RunPermissionOutcome::Unanswered,
                    _ => RunPermissionOutcome::Unknown,
                },
                reason: text("reason"),
            }),
            _ => {}
        }
    }
    // Once its turn ended, a step with no end on record is done: the
    // agent moved past it.
    if events.iter().any(|event| event.kind == "turn_ended") {
        for step in &mut detail.steps {
            if step.state == RunStepState::Started {
                step.state = RunStepState::Done;
            }
        }
    }
    detail
}

/// How many ended runs a conversation's history brings back.
const PAST_RUNS: i64 = 200;

/// Keep `run` as it ended (`client_runs`): its last word replaces any
/// earlier one.
pub(crate) async fn keep(
    pool: &sqlx::SqlitePool,
    run: &aidaemon_client_protocol::RunSummary,
) -> anyhow::Result<()> {
    let Some(ended) = run.ended_at_ms else {
        return Ok(());
    };
    sqlx::query(
        "INSERT INTO client_runs (run_id, session_id, ended_at_ms, data) VALUES (?, ?, ?, ?)
         ON CONFLICT(run_id) DO UPDATE SET session_id = excluded.session_id,
             ended_at_ms = excluded.ended_at_ms, data = excluded.data",
    )
    .bind(&run.id)
    .bind(&run.session_id)
    .bind(i64::try_from(ended).unwrap_or(i64::MAX))
    .bind(serde_json::to_string(run)?)
    .execute(pool)
    .await?;
    Ok(())
}

/// Conversation `session_id`'s newest kept runs, oldest first. A record
/// this build cannot read is left out.
pub(crate) async fn past(
    pool: &sqlx::SqlitePool,
    session_id: &str,
) -> anyhow::Result<Vec<aidaemon_client_protocol::RunSummary>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT data FROM client_runs WHERE session_id = ? ORDER BY ended_at_ms DESC LIMIT ?",
    )
    .bind(session_id)
    .bind(PAST_RUNS)
    .fetch_all(pool)
    .await?;
    let mut runs: Vec<aidaemon_client_protocol::RunSummary> = rows
        .into_iter()
        .filter_map(|(data,)| serde_json::from_str(&data).ok())
        .collect();
    runs.sort_by_key(|run| run.started_at_ms);
    Ok(runs)
}

/// A stored event time (SQLite's `datetime('now')`, UTC) in ms.
fn at_ms(at: &str) -> Option<u64> {
    let at = chrono::NaiveDateTime::parse_from_str(at, "%Y-%m-%d %H:%M:%S").ok()?;
    u64::try_from(at.and_utc().timestamp_millis()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(seq: i64, kind: &str, data: &str) -> CliAgentEvent {
        CliAgentEvent {
            seq,
            at: "2026-01-02 03:04:05".into(),
            kind: kind.into(),
            data: data.into(),
        }
    }

    #[test]
    fn a_runs_kept_events_read_back_as_its_steps_plan_permissions_and_usage() {
        let run = CliAgentRun {
            invocation: 7,
            agent_name: "codex".into(),
            agent_session: None,
            working_dir: Some("/w/synthetic".into()),
            prompt_summary: "Review the slug".into(),
            task: Some("Review the slug module".into()),
            output_summary: Some("Fixed one bug".into()),
        };
        let events = [
            event(
                1,
                "session",
                r#"{"id":"thread-1","model":"synthetic-large"}"#,
            ),
            event(
                2,
                "step",
                r#"{"id":"i1","parent":null,"tool":{"kind":"run","program":"pytest"},"words":"Running pytest"}"#,
            ),
            event(3, "step_ended", r#"{"id":"i1","failed":true}"#),
            event(
                4,
                "step",
                r#"{"id":"i2","parent":"t1","tool":{"kind":"edit","path":"/w/synthetic/slug.py"},"words":"Editing slug.py"}"#,
            ),
            event(
                5,
                "plan",
                r#"[{"text":"Read it","status":"done"},{"text":"Fix it","status":"in_progress"}]"#,
            ),
            event(
                6,
                "permission",
                r#"{"action":"write","label":"write: /w/synthetic/slug.py","outcome":"allowed","reason":null}"#,
            ),
            event(
                7,
                "usage",
                r#"{"input_tokens":120,"output_tokens":30,"cost_usd":0.01}"#,
            ),
            event(8, "turn_ended", r#"{"failed":false}"#),
        ];
        let detail = detail("cli-agent-1", run, &events);
        assert_eq!(detail.agent_session_id.as_deref(), Some("thread-1"));
        assert_eq!(detail.model.as_deref(), Some("synthetic-large"));
        assert_eq!(detail.task, "Review the slug module");
        assert_eq!(detail.steps.len(), 2);
        assert_eq!(detail.steps[0].words, "Running pytest");
        assert_eq!(detail.steps[0].state, RunStepState::Failed);
        assert_eq!(detail.steps[0].at_ms, Some(1_767_323_045_000));
        // No end on record, but its turn ended: done.
        assert_eq!(detail.steps[1].state, RunStepState::Done);
        assert!(detail.steps[1].by_helper);
        assert_eq!(detail.plan[1].status, RunPlanStatus::InProgress);
        assert_eq!(detail.permissions[0].outcome, RunPermissionOutcome::Allowed);
        assert_eq!(detail.usage.as_ref().map(|u| u.output_tokens), Some(30));
    }

    #[tokio::test]
    async fn an_ended_run_is_kept_for_its_conversation() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE client_runs (run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
             ended_at_ms INTEGER NOT NULL, data TEXT NOT NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        let mut run: aidaemon_client_protocol::RunSummary =
            serde_json::from_value(serde_json::json!({
                "id": "cli-agent-1", "session_id": "app:synthetic", "label": "Review",
                "agent": "Codex", "phase": "running", "background": true,
                "started_at_ms": 1000, "group": "task-1"
            }))
            .unwrap();
        // Running: nothing kept yet.
        keep(&pool, &run).await.unwrap();
        assert!(past(&pool, "app:synthetic").await.unwrap().is_empty());
        run.phase = aidaemon_client_protocol::RunPhase::Succeeded;
        run.ended_at_ms = Some(5000);
        run.edited = vec!["/w/synthetic/review.md".into()];
        keep(&pool, &run).await.unwrap();
        keep(&pool, &run).await.unwrap();
        let kept = past(&pool, "app:synthetic").await.unwrap();
        assert_eq!(kept, vec![run]);
        assert!(past(&pool, "app:other").await.unwrap().is_empty());
    }
}
