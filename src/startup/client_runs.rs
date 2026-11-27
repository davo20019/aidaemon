//! What one run did, for the app's run view (protocol 1.46, `get_run`): a
//! launched coding agent's invocation and its kept events
//! (`cli_agent_events`), read back into typed protocol items. The events'
//! data is the daemon's own stored form
//! ([`aidaemon_coding_agents::AgentEvent::to_record`] and the run's
//! permission records), a machine format it wrote itself.

use aidaemon_client_protocol::{
    RunDetail, RunPermission, RunPermissionOutcome, RunPlanItem, RunPlanStatus, RunStepItem,
    RunStepKind, RunStepState, RunUsage,
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
            can_continue: false,
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
    let speaks = speaks_a_protocol(&run.agent_name);
    Ok(detail(run_id, run, &events, speaks))
}

/// Whether agent `name`'s output is its protocol's stream (typed events),
/// never a report in words: what it left when it did not finish is raw
/// protocol data, even when it failed before reporting a step.
fn speaks_a_protocol(name: &str) -> bool {
    aidaemon_runtime::tools::agent_mode::agent_named(name)
        .is_some_and(|agent| agent.reports_steps())
}

/// `speaks`: its agent's output is a protocol stream ([`speaks_a_protocol`]).
fn detail(run_id: &str, run: CliAgentRun, events: &[CliAgentEvent], speaks: bool) -> RunDetail {
    let mut detail = RunDetail {
        run_id: run_id.to_string(),
        agent_session_id: run.agent_session,
        can_continue: false,
        model: None,
        working_dir: run.working_dir,
        task: run.task.unwrap_or(run.prompt_summary),
        // Its report: what it said at its end. An agent whose protocol
        // was read as typed events (its events are on record) that did not
        // finish left its raw stream there, not a report: its steps and how
        // it ended say what happened instead.
        result: run
            .output_summary
            .filter(|text| !text.trim().is_empty())
            .filter(|_| run.success == Some(true) || !(speaks || !events.is_empty())),
        steps: Vec::new(),
        plan: Vec::new(),
        permissions: Vec::new(),
        usage: None,
    };
    // The steps that are the agent's calls handing work to another agent.
    let mut hand_off_calls: Vec<usize> = Vec::new();
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
                let tool = data.get("tool");
                let (kind, path) = step_kind(tool);
                if tool
                    .and_then(|tool| tool.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(aidaemon_coding_agents::is_hand_off_tool)
                {
                    hand_off_calls.push(detail.steps.len());
                }
                detail.steps.push(RunStepItem {
                    words: text("words").unwrap_or_default(),
                    state: RunStepState::Started,
                    at_ms: at_ms(&event.at),
                    by_helper: data.get("parent").is_some_and(|parent| !parent.is_null()),
                    kind,
                    path,
                    run: None,
                    agent: None,
                });
            }
            // The daemon's own record of a run this one's agent handed work
            // to (`peer_run`, kept when it launched it).
            "peer_run" => {
                let agent = text("agent").unwrap_or_else(|| "another agent".to_string());
                let follow_up = data.get("follow_up").and_then(Value::as_bool) == Some(true);
                detail.steps.push(RunStepItem {
                    words: if follow_up {
                        format!("Sent a follow-up to {agent}")
                    } else {
                        format!("Handed work to {agent}")
                    },
                    state: RunStepState::Done,
                    at_ms: at_ms(&event.at),
                    by_helper: false,
                    kind: RunStepKind::HandOff,
                    path: None,
                    run: text("run_id"),
                    agent: Some(agent),
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
    // A hand-off call that went through is said by the daemon's own
    // record of the run it started (agent and run, which opens it): the
    // call's own step would say it twice, and between two records it kept
    // them from reading as one row (live 2026-11-27). A failed one stays.
    if detail
        .steps
        .iter()
        .any(|step| step.kind == RunStepKind::HandOff)
    {
        let mut index = 0;
        detail.steps.retain(|step| {
            let keep = !(hand_off_calls.contains(&index) && step.state == RunStepState::Done);
            index += 1;
            keep
        });
    }
    // A follow-up goes on in the session its agent kept.
    detail.can_continue = run.keeps_session && detail.agent_session_id.is_some();
    detail
}

/// How many ended runs a conversation's history brings back.
const PAST_RUNS: i64 = 200;

/// Keep `run` as it stands now (`client_runs`), on every change the app is
/// told of: its last word replaces any earlier one, so a run a restart cuts
/// off still has its record ([`close_cut_off`] ends it at the next start).
/// A running record never replaces an ended one (writes can land out of
/// order). A running run sorts by its start, an ended one by its end.
pub(crate) async fn keep(
    pool: &sqlx::SqlitePool,
    run: &aidaemon_client_protocol::RunSummary,
) -> anyhow::Result<()> {
    let at = run.ended_at_ms.unwrap_or(run.started_at_ms);
    sqlx::query(
        "INSERT INTO client_runs (run_id, session_id, ended_at_ms, data) VALUES (?, ?, ?, ?)
         ON CONFLICT(run_id) DO UPDATE SET session_id = excluded.session_id,
             ended_at_ms = excluded.ended_at_ms, data = excluded.data
         WHERE json_extract(client_runs.data, '$.ended_at_ms') IS NULL
            OR json_extract(excluded.data, '$.ended_at_ms') IS NOT NULL",
    )
    .bind(&run.id)
    .bind(&run.session_id)
    .bind(i64::try_from(at).unwrap_or(i64::MAX))
    .bind(serde_json::to_string(run)?)
    .execute(pool)
    .await?;
    Ok(())
}

/// Why a run kept as running at startup ended: the daemon stopped.
const CUT_OFF_REASON: &str = "A daemon restart stopped it before it finished";

/// `run`, kept as running, as a stop at `now` left it; `agent` is its
/// coding agent's invocation and the events kept as it ran, when it was
/// one.
fn cut_off(
    mut run: aidaemon_client_protocol::RunSummary,
    agent: Option<(&CliAgentRun, &[CliAgentEvent])>,
    now: u64,
) -> aidaemon_client_protocol::RunSummary {
    run.phase = aidaemon_client_protocol::RunPhase::Stopped;
    run.ended_at_ms = Some(now.max(run.started_at_ms));
    run.step.clear();
    run.reason = Some(CUT_OFF_REASON.to_string());
    if let Some((invocation, events)) = agent {
        for path in aidaemon_runtime::tools::cli_agent::RunProgress::of(events).edited {
            if !run.edited.contains(&path) {
                run.edited.push(path);
            }
        }
        run.can_continue = invocation.keeps_session && invocation.agent_session.is_some();
    }
    run
}

/// End every run kept as running: at startup each was cut off by the
/// daemon's stop. Its record says so; a coding agent's gets the files it
/// edited by the events kept as it ran, and whether a follow-up can go on
/// in its session. Run before any client is served, so no live run is
/// among them. Returns how many it ended.
pub(crate) async fn close_cut_off(
    pool: &sqlx::SqlitePool,
    state: &dyn StateStore,
) -> anyhow::Result<usize> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT data FROM client_runs WHERE json_extract(data, '$.ended_at_ms') IS NULL",
    )
    .fetch_all(pool)
    .await?;
    let now = u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0);
    let mut closed = 0;
    for (data,) in rows {
        let Ok(run) = serde_json::from_str::<aidaemon_client_protocol::RunSummary>(&data) else {
            continue;
        };
        let agent = match state.cli_agent_run(&run.id).await {
            Ok(Some(invocation)) => {
                let events = state
                    .get_cli_agent_events(invocation.invocation)
                    .await
                    .unwrap_or_default();
                Some((invocation, events))
            }
            _ => None,
        };
        let run = cut_off(
            run,
            agent.as_ref().map(|(run, events)| (run, &events[..])),
            now,
        );
        keep(pool, &run).await?;
        closed += 1;
    }
    Ok(closed)
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

/// Whether `thread` is one of conversation `session_id`'s coding agent
/// runs: only those can be named there.
pub(crate) async fn is_agent_of(
    state: &dyn StateStore,
    session_id: &str,
    thread: &str,
) -> Result<bool, String> {
    Ok(state
        .cli_agent_run(thread)
        .await
        .map_err(|error| error.to_string())?
        .is_some_and(|run| run.session_id == session_id))
}

/// Name the agent session whose thread is `thread` in conversation
/// `session_id` ([`is_agent_of`] first); `None` takes the name away.
pub(crate) async fn rename_agent(
    pool: &sqlx::SqlitePool,
    session_id: &str,
    thread: &str,
    name: Option<&str>,
) -> Result<(), String> {
    match name {
        Some(name) => sqlx::query(
            "INSERT INTO agent_names (thread_id, session_id, name) VALUES (?, ?, ?)
             ON CONFLICT(thread_id) DO UPDATE SET name = excluded.name",
        )
        .bind(thread)
        .bind(session_id)
        .bind(name)
        .execute(pool)
        .await
        .map(|_| ()),
        None => sqlx::query("DELETE FROM agent_names WHERE thread_id = ?")
            .bind(thread)
            .execute(pool)
            .await
            .map(|_| ()),
    }
    .map_err(|error| error.to_string())
}

/// The names the owner gave conversation `session_id`'s agents, by thread.
pub(crate) async fn agent_names(
    pool: &sqlx::SqlitePool,
    session_id: &str,
) -> anyhow::Result<std::collections::BTreeMap<String, String>> {
    let rows: Vec<(String, String)> =
        sqlx::query_as("SELECT thread_id, name FROM agent_names WHERE session_id = ?")
            .bind(session_id)
            .fetch_all(pool)
            .await?;
    Ok(rows.into_iter().collect())
}

/// A stored event time (SQLite's `datetime('now')`, UTC) in ms.
fn at_ms(at: &str) -> Option<u64> {
    let at = chrono::NaiveDateTime::parse_from_str(at, "%Y-%m-%d %H:%M:%S").ok()?;
    u64::try_from(at.and_utc().timestamp_millis()).ok()
}

/// A kept step's kind and the file it names, from its stored tool
/// (`aidaemon_coding_agents::AgentEvent::to_record`'s `kind` values).
fn step_kind(tool: Option<&Value>) -> (RunStepKind, Option<String>) {
    let field = |key: &str| {
        tool.and_then(|tool| tool.get(key))
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
    };
    let kind = match field("kind").as_deref() {
        Some("read") => RunStepKind::Read,
        Some("search") => RunStepKind::Search,
        Some("edit") => RunStepKind::Edit,
        Some("run") => RunStepKind::Run,
        Some("fetch") => RunStepKind::Fetch,
        Some("delegate") => RunStepKind::Delegate,
        Some("plan") => RunStepKind::Plan,
        Some("other")
            if field("name")
                .is_some_and(|name| aidaemon_coding_agents::is_coordination_tool(&name)) =>
        {
            RunStepKind::Coordinate
        }
        Some("other") => RunStepKind::Other,
        _ => RunStepKind::Unknown,
    };
    let path = matches!(kind, RunStepKind::Read | RunStepKind::Edit)
        .then(|| field("path"))
        .flatten();
    (kind, path)
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
            session_id: "app:synthetic".into(),
            agent_name: "codex".into(),
            agent_session: None,
            working_dir: Some("/w/synthetic".into()),
            prompt_summary: "Review the slug".into(),
            task: Some("Review the slug module".into()),
            success: Some(true),
            output_summary: Some("Fixed one bug".into()),
            keeps_session: true,
            thread: None,
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
            event(
                8,
                "step",
                r#"{"id":"i3","parent":null,"tool":{"kind":"other","name":"mcp_aidaemon_peers_post_board"},"words":"Posting to the agents' board"}"#,
            ),
            event(
                9,
                "step",
                r#"{"id":"i4","parent":null,"tool":{"kind":"other","name":"mcp__synthetic__lookup"},"words":"Using mcp__synthetic__lookup"}"#,
            ),
            event(
                9,
                "step",
                r#"{"id":"i5","parent":null,"tool":{"kind":"other","name":"mcp_aidaemon_peers_delegate_cli"},"words":"Handing work to another agent"}"#,
            ),
            event(9, "step_ended", r#"{"id":"i5","failed":false}"#),
            event(
                9,
                "peer_run",
                r#"{"run_id":"cli-agent-5e6f","agent":"Codex","follow_up":false}"#,
            ),
            event(10, "turn_ended", r#"{"failed":false}"#),
        ];
        let detail = detail("cli-agent-1", run, &events, false);
        assert_eq!(detail.agent_session_id.as_deref(), Some("thread-1"));
        assert!(detail.can_continue, "its agent kept the session it named");
        assert_eq!(detail.model.as_deref(), Some("synthetic-large"));
        assert_eq!(detail.task, "Review the slug module");
        assert_eq!(detail.steps.len(), 5);
        assert_eq!(detail.steps[0].words, "Running pytest");
        // Each step's kind and, for a file it changed, the file.
        let kinds: Vec<RunStepKind> = detail.steps.iter().map(|step| step.kind).collect();
        assert_eq!(
            kinds,
            [
                RunStepKind::Run,
                RunStepKind::Edit,
                RunStepKind::Coordinate,
                RunStepKind::Other,
                RunStepKind::HandOff
            ]
        );
        // A hand-off names its agent and the run it started.
        assert_eq!(detail.steps[4].words, "Handed work to Codex");
        assert_eq!(detail.steps[4].run.as_deref(), Some("cli-agent-5e6f"));
        assert_eq!(detail.steps[0].path, None);
        assert_eq!(
            detail.steps[1].path.as_deref(),
            Some("/w/synthetic/slug.py")
        );
        assert_eq!(detail.steps[0].state, RunStepState::Failed);
        assert_eq!(detail.steps[0].at_ms, Some(1_767_323_045_000));
        // No end on record, but its turn ended: done.
        assert_eq!(detail.steps[1].state, RunStepState::Done);
        assert!(detail.steps[1].by_helper);
        assert_eq!(detail.plan[1].status, RunPlanStatus::InProgress);
        assert_eq!(detail.permissions[0].outcome, RunPermissionOutcome::Allowed);
        assert_eq!(detail.usage.as_ref().map(|u| u.output_tokens), Some(30));
    }

    /// Live 2026-11-23: a Codex run stopped by the owner showed its raw
    /// protocol stream (`{"type":"thread.started"...}`) as its report.
    #[test]
    fn a_structured_agent_that_did_not_finish_has_no_report() {
        let run = |success| CliAgentRun {
            invocation: 8,
            session_id: "app:synthetic".into(),
            agent_name: "codex".into(),
            agent_session: None,
            working_dir: None,
            prompt_summary: "Review".into(),
            task: None,
            output_summary: Some(r#"{"type":"thread.started","thread_id":"t-1"}"#.into()),
            success,
            keeps_session: false,
            thread: None,
        };
        let events = [event(1, "session", r#"{"id":"t-1","model":null}"#)];
        // One that failed before reporting anything left its protocol's
        // error line, not a report.
        let mut early = run(Some(false));
        early.agent_name = "agy".into();
        early.output_summary = Some(r#"{"message":"Failed to start","type":"error"}"#.into());
        assert_eq!(detail("cli-agent-9", early, &[], true).result, None);
        assert_eq!(
            detail("cli-agent-8", run(Some(false)), &events, false).result,
            None
        );
        assert!(
            detail("cli-agent-8", run(Some(true)), &events, false)
                .result
                .is_some()
        );
        // An agent read as plain text keeps what it printed.
        assert!(
            detail("cli-agent-8", run(Some(false)), &[], false)
                .result
                .is_some()
        );
    }

    #[tokio::test]
    async fn a_run_is_kept_for_its_conversation_as_it_goes_and_as_it_ended() {
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
        // Running: kept as it stands, so a stop does not lose it.
        let running = run.clone();
        keep(&pool, &running).await.unwrap();
        assert_eq!(
            past(&pool, "app:synthetic").await.unwrap(),
            vec![running.clone()]
        );
        run.phase = aidaemon_client_protocol::RunPhase::Succeeded;
        run.ended_at_ms = Some(5000);
        run.edited = vec!["/w/synthetic/review.md".into()];
        keep(&pool, &run).await.unwrap();
        keep(&pool, &run).await.unwrap();
        // A running write that lands late never replaces its end.
        keep(&pool, &running).await.unwrap();
        let kept = past(&pool, "app:synthetic").await.unwrap();
        assert_eq!(kept, vec![run]);
        assert!(past(&pool, "app:other").await.unwrap().is_empty());
    }

    /// A run a stop cut off reads as stopped by it, with the files its
    /// agent edited by its kept events and whether its session goes on.
    #[test]
    fn a_run_kept_as_running_reads_as_cut_off_by_the_stop() {
        let run: aidaemon_client_protocol::RunSummary = serde_json::from_value(serde_json::json!({
            "id": "cli-agent-2", "session_id": "app:synthetic", "label": "Fix",
            "agent": "Codex", "phase": "running", "background": true,
            "step": "Editing lib.rs", "started_at_ms": 1000,
            "edited": ["/w/synthetic/a.rs"]
        }))
        .unwrap();
        let invocation = CliAgentRun {
            invocation: 9,
            session_id: "app:synthetic".into(),
            agent_name: "codex".into(),
            agent_session: Some("thread-9".into()),
            working_dir: Some("/w/synthetic".into()),
            prompt_summary: "Fix".into(),
            task: None,
            success: Some(false),
            output_summary: None,
            keeps_session: true,
            thread: None,
        };
        let events = [
            event(
                1,
                "step",
                r#"{"id":"e1","parent":null,"tool":{"kind":"edit","path":"/w/synthetic/b.rs"},"words":"Editing b.rs"}"#,
            ),
            event(2, "step_ended", r#"{"id":"e1","failed":false}"#),
        ];
        let ended = cut_off(run, Some((&invocation, &events)), 9000);
        assert_eq!(ended.phase, aidaemon_client_protocol::RunPhase::Stopped);
        assert_eq!(ended.ended_at_ms, Some(9000));
        assert!(ended.step.is_empty());
        assert_eq!(ended.reason.as_deref(), Some(CUT_OFF_REASON));
        assert_eq!(ended.edited, vec!["/w/synthetic/a.rs", "/w/synthetic/b.rs"]);
        assert!(ended.can_continue);
    }

    /// A name the owner gives an agent is its conversation's, by thread,
    /// until taken away.
    #[tokio::test]
    async fn an_agent_keeps_the_name_it_was_given() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE agent_names (thread_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
             name TEXT NOT NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        rename_agent(&pool, "app:synthetic", "cli-agent-1", Some("Designer"))
            .await
            .unwrap();
        rename_agent(&pool, "app:synthetic", "cli-agent-1", Some("Developer"))
            .await
            .unwrap();
        let names = agent_names(&pool, "app:synthetic").await.unwrap();
        assert_eq!(
            names.get("cli-agent-1").map(String::as_str),
            Some("Developer")
        );
        assert!(agent_names(&pool, "app:other").await.unwrap().is_empty());
        rename_agent(&pool, "app:synthetic", "cli-agent-1", None)
            .await
            .unwrap();
        assert!(
            agent_names(&pool, "app:synthetic")
                .await
                .unwrap()
                .is_empty()
        );
    }
}
