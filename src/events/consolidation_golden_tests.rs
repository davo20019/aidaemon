//! Consolidation's behaviour over a synthetic history, down to the rows it
//! leaves behind. `golden` records those rows as the previous implementation
//! wrote them, so a change to what is learned has to change this file too.

use crate::events::EventStoreForTests as _;
use std::collections::BTreeMap;
use std::sync::Arc;

use aidaemon_memory::encode_embedding;
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde_json::{json, Value};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};

use super::{ConsolidationResult, Consolidator};
use crate::config::ProviderKind;
use crate::events::{
    DecisionPointData, DecisionType, DiagnosticSeverity, Event, EventStore, EventType, TaskEndData,
    TaskOutcome, TaskStatus,
};
use crate::llm_runtime::SharedLlmRuntime;
use crate::memory::embeddings::EmbeddingService;
use crate::plans::{PlanStatus, PlanStore, StepStatus, TaskPlan};
use crate::testing::MockProvider;

const GOLDEN: &str = include_str!("testdata/consolidation.golden");

/// One pass configuration: with neither model nor embeddings, or with both
/// and the evidence gate on.
#[derive(Clone, Copy, Debug)]
struct Setup {
    name: &'static str,
    with_models: bool,
    evidence_gate: bool,
}

const SETUPS: [Setup; 2] = [
    Setup {
        name: "plain",
        with_models: false,
        evidence_gate: false,
    },
    Setup {
        name: "models_and_gate",
        with_models: true,
        evidence_gate: true,
    },
];

const OWNER_SESSIONS: [&str; 2] = ["owner-round-one", "owner-round-two"];
const MANDATE_SESSION: &str = "owner-looking-mandate";
const WORKER_SESSION: &str = "specialist:executor:worker";

/// Events in the order they happened, one second apart.
struct History {
    session: String,
    clock: DateTime<Utc>,
    events: Vec<Event>,
}

impl History {
    fn new(session: &str, start: DateTime<Utc>) -> Self {
        Self {
            session: session.to_string(),
            clock: start,
            events: Vec::new(),
        }
    }

    fn push(&mut self, event_type: EventType, data: Value) {
        let mut event = Event::new(self.session.clone(), event_type, data);
        event.created_at = self.clock;
        self.clock += Duration::seconds(1);
        self.events.push(event);
    }

    fn user(&mut self, content: &str) {
        self.push(
            EventType::UserMessage,
            json!({"content": content, "user_role": "owner"}),
        );
    }

    fn internal_continuation(&mut self, content: &str) {
        self.push(
            EventType::UserMessage,
            json!({
                "content": content,
                "user_role": "owner",
                "annotations": ["internal_continuation"]
            }),
        );
    }

    fn policy(&mut self, task_id: &str, access: &str) {
        self.push(
            EventType::MemoryPolicyCompiled,
            json!({
                "task_id": task_id,
                "access": access,
                "reason_code": "synthetic_test_policy",
                "retrieval_suppressed": false,
                "persistence_suppressed": access == "suppressed"
            }),
        );
    }

    fn start(&mut self, task_id: &str, description: &str) {
        self.policy(task_id, "allowed");
        self.push(
            EventType::TaskStart,
            json!({"task_id": task_id, "description": description}),
        );
    }

    fn call(&mut self, task_id: &str, name: &str, summary: Option<&str>) {
        self.push(
            EventType::ToolCall,
            json!({
                "tool_call_id": format!("call-{}", self.events.len()),
                "name": name,
                "arguments": {},
                "summary": summary,
                "task_id": task_id
            }),
        );
    }

    fn remember(&mut self, arguments: Value) {
        self.remember_with("remember_fact", arguments);
    }

    fn remember_with(&mut self, tool: &str, arguments: Value) {
        self.push(
            EventType::ToolCall,
            json!({
                "tool_call_id": format!("call-{}", self.events.len()),
                "name": tool,
                "arguments": arguments
            }),
        );
    }

    fn result(&mut self, task_id: Option<&str>, name: &str, success: bool, text: &str) {
        self.push(
            EventType::ToolResult,
            json!({
                "tool_call_id": format!("call-{}", self.events.len()),
                "name": name,
                "result": text,
                "success": success,
                "duration_ms": 5,
                "task_id": task_id
            }),
        );
    }

    fn error(&mut self, task_id: Option<&str>, tool: Option<&str>, message: &str) {
        self.push(
            EventType::Error,
            json!({
                "message": message,
                "error_type": "tool_error",
                "task_id": task_id,
                "tool_name": tool
            }),
        );
    }

    fn end(&mut self, task_id: &str, outcome: TaskOutcome, duration_secs: u64) {
        self.push(
            EventType::TaskEnd,
            serde_json::to_value(TaskEndData {
                task_id: task_id.to_string(),
                status: if outcome == TaskOutcome::Failed {
                    TaskStatus::Failed
                } else {
                    TaskStatus::Completed
                },
                outcome: Some(outcome),
                duration_secs,
                iterations: 2,
                tool_calls_count: 2,
                error: None,
                summary: None,
                efficiency: None,
                turn_id: None,
                completion_proof: None,
                harness_eval: None,
            })
            .expect("serialize task end"),
        );
    }

    fn decision(
        &mut self,
        task_id: &str,
        code: Option<&str>,
        decision_type: DecisionType,
        severity: DiagnosticSeverity,
        metadata: Value,
    ) {
        self.push(
            EventType::DecisionPoint,
            serde_json::to_value(DecisionPointData {
                decision_type,
                task_id: task_id.to_string(),
                iteration: 1,
                severity,
                code: code.map(str::to_string),
                metadata,
                summary: "synthetic decision".to_string(),
            })
            .expect("serialize decision"),
        );
    }

    fn warning(&mut self, task_id: &str, code: &str, metadata: Value) {
        self.decision(
            task_id,
            Some(code),
            DecisionType::StoppingCondition,
            DiagnosticSeverity::Warning,
            metadata,
        );
    }
}

/// Every known code, with the metadata branches each one has.
fn failure_codes() -> Vec<(&'static str, Value)> {
    vec![
        (
            "repetitive_call_detection",
            json!({"tool": "terminal", "count": 4, "action": "hard_stop"}),
        ),
        (
            "repetitive_call_detection",
            json!({"tool": "web_fetch", "count": 0, "action": "redirect"}),
        ),
        ("repetitive_call_detection", json!({})),
        (
            "consecutive_same_tool_detection",
            json!({"tool": "browser", "consecutive_count": 7, "unique_args": 7, "is_diverse": true}),
        ),
        (
            "consecutive_same_tool_detection",
            json!({"tool": "terminal", "consecutive_count": 6, "unique_args": 2}),
        ),
        (
            "alternating_pattern_detection",
            json!({"tools": ["terminal", "edit_file"]}),
        ),
        ("alternating_pattern_detection", json!({})),
        (
            "tool_budget_block",
            json!({"tool": "http_request", "prior_semantic_failures": 3, "prior_calls": 9}),
        ),
        (
            "tool_budget_block",
            json!({"tool": "web_search", "prior_calls": 9}),
        ),
        (
            "missing_pre_execution_evidence",
            json!({"tool": "edit_file", "required_evidence_kind": "file_read", "target": "src/main.rs"}),
        ),
        ("missing_pre_execution_evidence", json!({})),
        ("plan_rejected", json!({})),
        ("critique_rejected", json!({})),
        ("target_scope_violation", json!({"tool": "write_file"})),
        ("tool_contract_violation", json!({"tool": "read_file"})),
        ("contradictory_file_evidence", json!({})),
        ("verification_pending", json!({})),
        ("verification_unavailable_in_phase", json!({})),
        ("validation_budget_exhausted", json!({})),
        ("execution_budget_exhausted", json!({})),
        ("pre_tool_deferral_stall", json!({})),
        ("stall", json!({"stall_mode": "no_new_evidence"})),
        ("stall", json!({})),
        ("hard_iteration_cap", json!({})),
        ("task_timeout", json!({})),
        ("task_token_budget", json!({})),
        ("scheduled_run_budget", json!({})),
        ("goal_daily_token_budget", json!({})),
        ("daily_token_budget", json!({})),
        ("task_token_budget_warning", json!({})),
        ("soft_iteration_warning", json!({})),
    ]
}

/// How each group of failure codes ends, so every confidence bonus is hit.
const CODE_GROUPS: usize = 6;

fn code_group_task(group: usize) -> String {
    format!("task-codes-{group}")
}

fn owner_history(session: &str, start: DateTime<Utc>) -> Vec<Event> {
    let mut h = History::new(session, start);

    // Fact provenance, starting before any user message.
    h.remember(json!({"category": "profile", "key": "no_user", "value": "anything"}));
    h.user("Alice Rivera works at Acme Corp and lives in Lisbon");
    h.remember(json!({"category": "profile", "key": "city", "value": "LISBON"}));
    h.remember(json!({"facts": [
        {"category": "profile", "key": "employer", "value": "acme corp"},
        {"category": "profile", "key": "derived_one", "value": "not in the message"},
        {"category": "profile", "value": "entry without a key"}
    ]}));
    h.remember(json!({"category": "profile", "key": "short", "value": "abc"}));
    h.remember(json!({"category": "profile", "key": "absent", "value": "lisbon"}));
    h.remember(json!({"category": "profile", "key": "superseded", "value": "lisbon"}));
    h.remember(json!({"category": "profile", "key": "no_index", "value": "lisbon"}));
    h.remember_with(
        "some_other_tool",
        json!({"category": "profile", "key": "other_tool", "value": "lisbon"}),
    );
    h.internal_continuation("continuing: lisbon was stated here only by the harness");
    h.remember(json!({"category": "profile", "key": "after_continuation", "value": "acme"}));
    h.user(&format!(
        "Jordan Lee mudó a Quito. {}",
        "ñandú añejo ".repeat(40)
    ));
    h.remember(json!({"category": "profile", "key": "friend_city", "value": "quito"}));

    // Successful multi-tool tasks, including one whose name collides with a
    // stored procedure and one that duplicates a stored trigger.
    h.start("task-success-alpha", "Ship the Acme Corp release");
    h.call("task-success-alpha", "terminal", Some("cargo test"));
    h.result(Some("task-success-alpha"), "terminal", true, "tests passed");
    h.call("task-success-alpha", "git_commit", Some("commit the bump"));
    h.end("task-success-alpha", TaskOutcome::Succeeded, 30);

    h.start("task-research", "Compare hosting prices for Acme Corp");
    h.call("task-research", "web_search", Some("prices"));
    h.call("task-research", "web_fetch", None);
    h.end("task-research", TaskOutcome::Succeeded, 12);

    h.start("task-two-step-model", "Rotate the staging certificates");
    h.call("task-two-step-model", "terminal", Some("renew"));
    h.call("task-two-step-model", "terminal", Some("reload"));
    h.end("task-two-step-model", TaskOutcome::Succeeded, 8);

    h.start("task-skipped-by-model", "Tidy one specific scratch file");
    h.call("task-skipped-by-model", "read_file", Some("scratch"));
    h.call("task-skipped-by-model", "edit_file", Some("tidy"));
    h.end("task-skipped-by-model", TaskOutcome::Succeeded, 4);

    h.start("task-duplicate", "A trigger memory already holds");
    h.call("task-duplicate", "read_file", Some("a"));
    h.call("task-duplicate", "edit_file", Some("b"));
    h.end("task-duplicate", TaskOutcome::Succeeded, 3);

    // Outcomes that must not become procedures but still count as expertise.
    h.start("task-browser-failed", "Fill the signup form");
    h.call("task-browser-failed", "browser", Some("open"));
    h.call("task-browser-failed", "browser", Some("click"));
    h.end("task-browser-failed", TaskOutcome::Failed, 9);

    h.start("task-partial", "Clean the logs");
    h.call("task-partial", "terminal", Some("rm old"));
    h.call("task-partial", "terminal", Some("rm older"));
    h.end("task-partial", TaskOutcome::Partial, 6);

    h.start("task-single-tool", "Say the date");
    h.call("task-single-tool", "date", Some("now"));
    h.end("task-single-tool", TaskOutcome::Succeeded, 1);

    h.start("task-unended", "Never finishes");
    h.call("task-unended", "terminal", Some("sleep"));
    h.call("task-unended", "terminal", Some("sleep more"));

    // A recovered error with decision points and user messages inside its
    // window: only work events count toward the nine-event limit.
    h.error(
        Some("task-unended"),
        Some("terminal"),
        "[SYSTEM] wrapper line\nconnection refused at /srv/app/db.rs:10:4\n[UNTRUSTED EXTERNAL DATA noise",
    );
    for _ in 0..4 {
        h.decision(
            "task-unended",
            Some("stall"),
            DecisionType::StoppingCondition,
            DiagnosticSeverity::Info,
            json!({}),
        );
        h.user("are you still there?");
    }
    for _ in 0..8 {
        h.result(Some("task-unended"), "terminal", false, "still down");
    }
    h.result(
        Some("task-unended"),
        "run_command",
        true,
        "database started",
    );

    // An error whose recovery falls just past the window.
    h.error(None, Some("terminal"), "disk quota exceeded on /var/data");
    for _ in 0..9 {
        h.result(None, "terminal", false, "still full");
    }
    h.result(None, "run_command", true, "too late to count");

    // An error with no tool, a long recovery text, and one that is only
    // scaffolding and leaves an empty pattern.
    h.error(None, None, "unexpected token in configuration");
    h.result(None, "edit_file", true, &"fixed configuration ".repeat(30));
    h.error(None, Some("web_fetch"), "[SYSTEM] only scaffolding here");
    h.result(None, "web_search", true, "found another way");

    // Every failure-pattern code, spread over task endings. Half of each
    // group comes before a successful tool result, so it earns the recovery
    // bonus, and half after.
    let codes = failure_codes();
    for group in 0..CODE_GROUPS {
        let task_id = code_group_task(group);
        h.policy(&task_id, "allowed");
        let group_codes: Vec<&(&str, Value)> = codes
            .iter()
            .enumerate()
            .filter(|(index, _)| index % CODE_GROUPS == group)
            .map(|(_, code)| code)
            .collect();
        let middle = group_codes.len() / 2;
        for (index, (code, metadata)) in group_codes.iter().enumerate() {
            if index == middle {
                h.result(Some(&task_id), "terminal", false, "no luck");
                h.result(Some(&task_id), "edit_file", true, "worked");
            }
            h.warning(&task_id, code, metadata.clone());
            // A repeat in the same task is the same lesson: the first wins.
            h.warning(
                &task_id,
                code,
                json!({"tool": metadata["tool"], "count": 99, "action": metadata["action"],
                       "tools": metadata["tools"], "stall_mode": metadata["stall_mode"],
                       "required_evidence_kind": metadata["required_evidence_kind"]}),
            );
        }
        match group {
            0 => h.end(&task_id, TaskOutcome::Failed, 5),
            1 => h.end(&task_id, TaskOutcome::Partial, 5),
            2 => h.end(&task_id, TaskOutcome::Succeeded, 5),
            3 => {} // no end at all
            4 | 5 => {
                h.end(&task_id, TaskOutcome::Failed, 5);
                if group == 5 {
                    h.end(&task_id, TaskOutcome::Failed, 5);
                }
            }
            _ => unreachable!(),
        }
    }

    // Decisions that teach nothing.
    h.policy("task-no-lesson", "allowed");
    h.decision(
        "task-no-lesson",
        Some("hard_iteration_cap"),
        DecisionType::StoppingCondition,
        DiagnosticSeverity::Info,
        json!({}),
    );
    h.warning("task-no-lesson", "a_code_nobody_knows", json!({}));
    h.push(
        EventType::DecisionPoint,
        json!({"task_id": "task-no-lesson", "code": "stall", "severity": "warning"}),
    );
    // No code: the decision type names the lesson.
    h.decision(
        "task-no-lesson",
        None,
        DecisionType::RepetitiveCallDetection,
        DiagnosticSeverity::Error,
        json!({"tool": "fallback_tool", "count": 2, "action": "hard_stop"}),
    );
    // Indexed under no task: not part of any task's decisions.
    h.warning("task-unindexed-decision", "task_timeout", json!({}));

    // A suppressed task and one with no policy at all.
    h.policy("task-suppressed", "suppressed");
    h.push(
        EventType::TaskStart,
        json!({"task_id": "task-suppressed", "description": "Private errand"}),
    );
    h.call("task-suppressed", "terminal", Some("secret"));
    h.call("task-suppressed", "terminal", Some("secret two"));
    h.end("task-suppressed", TaskOutcome::Succeeded, 2);
    h.push(
        EventType::TaskStart,
        json!({"task_id": "task-no-policy", "description": "Unclassified errand"}),
    );
    h.call("task-no-policy", "terminal", Some("x"));
    h.call("task-no-policy", "terminal", Some("y"));
    h.end("task-no-policy", TaskOutcome::Succeeded, 2);

    h.events
}

/// Rewrites what `append` would refuse or derive: old rows and damaged ones.
async fn damage_rows(pool: &SqlitePool, session: &str) {
    for statement in [
        "UPDATE events SET task_id = NULL
         WHERE session_id = ? AND event_type = 'decision_point'
           AND json_extract(data, '$.task_id') = 'task-unindexed-decision'",
        "UPDATE events SET task_id = NULL
         WHERE session_id = ? AND event_type = 'task_end'
           AND json_extract(data, '$.task_id') = 'task-codes-4'",
        "UPDATE events SET tool_name = NULL
         WHERE session_id = ? AND event_type = 'tool_call'
           AND json_extract(data, '$.arguments.key') = 'no_index'",
        "UPDATE events SET data = '{\"damaged\": true}'
         WHERE session_id = ? AND id = (
             SELECT MAX(id) FROM events
             WHERE session_id = ?1 AND event_type = 'task_end' AND task_id = 'task-codes-5')",
    ] {
        sqlx::query(statement)
            .bind(session)
            .execute(pool)
            .await
            .expect("damage row");
    }
}

fn plan(
    id: &str,
    session: &str,
    description: &str,
    steps: &[&str],
    completed: usize,
    task_id: Option<&str>,
    age_minutes: i64,
) -> TaskPlan {
    let mut plan = TaskPlan::new(
        session,
        description,
        description,
        steps.iter().map(|step| step.to_string()).collect(),
        "synthetic",
    );
    plan.id = id.to_string();
    plan.status = PlanStatus::Completed;
    plan.task_id = task_id.map(str::to_string);
    plan.updated_at = Utc::now() - Duration::minutes(age_minutes);
    for (index, step) in plan.steps.iter_mut().enumerate() {
        step.status = if index < completed {
            StepStatus::Completed
        } else {
            StepStatus::Pending
        };
    }
    plan
}

/// Oldest last in `updated_at DESC`, so each round's order is fixed.
fn plans_for(session: &str) -> Vec<TaskPlan> {
    vec![
        plan(
            &format!("plan-one-{session}"),
            session,
            "Publish the Acme Corp newsletter",
            &["write it", "review it", "send it"],
            3,
            None,
            1,
        ),
        plan(
            &format!("plan-two-{session}"),
            session,
            "A partly finished migration",
            &["dump", "load", "verify"],
            2,
            Some("task-success-alpha"),
            2,
        ),
        plan(
            &format!("plan-short-{session}"),
            session,
            "Only one step finished",
            &["first", "second"],
            1,
            None,
            3,
        ),
        plan(
            &format!("plan-private-{session}"),
            session,
            "A plan from a suppressed task",
            &["a", "b"],
            2,
            Some("task-suppressed"),
            4,
        ),
        plan(
            &format!("plan-dupe-{session}"),
            session,
            "Publish the Acme Corp newsletter",
            &["again", "and again"],
            2,
            None,
            5,
        ),
    ]
}

async fn seed_memory(pool: &SqlitePool, embeddings: &EmbeddingService) {
    let at = "2026-01-01T00:00:00+00:00";
    sqlx::query(
        "INSERT INTO procedures (name, trigger_pattern, steps, success_count, failure_count,
                                 avg_duration_secs, last_used_at, created_at, updated_at)
         VALUES ('auto_task-suc', 'an older trigger', '[\"older step\"]', 5, 2, 9.0, ?1, ?1, ?1)",
    )
    .bind(at)
    .execute(pool)
    .await
    .expect("seed procedure");
    let duplicate = embeddings
        .embed("A trigger memory already holds".to_string())
        .await
        .expect("embedding");
    sqlx::query(
        "INSERT INTO procedures (name, trigger_pattern, trigger_embedding, steps, created_at, updated_at)
         VALUES ('already-known', 'A trigger memory already holds', ?, '[\"x\",\"y\"]', ?2, ?2)",
    )
    .bind(encode_embedding(&duplicate))
    .bind(at)
    .execute(pool)
    .await
    .expect("seed duplicate");
    sqlx::query(
        "INSERT INTO error_solutions (error_pattern, domain, solution_summary, solution_steps,
                                      success_count, failure_count, created_at)
         VALUES ('connection refused at <path>:<line>', 'terminal',
                 'Used run_command to resolve', '[\"older fix\"]', 3, 1, ?)",
    )
    .bind(at)
    .execute(pool)
    .await
    .expect("seed error solution");
    sqlx::query(
        "INSERT INTO expertise (domain, tasks_attempted, tasks_succeeded, tasks_failed,
                                current_level, confidence_score, common_errors, created_at, updated_at)
         VALUES ('git', 10, 8, 2, 'competent', 0.7, '[\"merge conflict\"]', ?1, ?1)",
    )
    .bind(at)
    .execute(pool)
    .await
    .expect("seed expertise");
    sqlx::query(
        "INSERT INTO behavior_patterns (pattern_type, description, trigger_context, action,
                                        confidence, occurrence_count, created_at)
         VALUES ('failure', 'an older description', 'plan_rejected',
                 'replan_first_risky_step_before_execution', 0.9, 3, ?)",
    )
    .bind(at)
    .execute(pool)
    .await
    .expect("seed behavior pattern");
    for key in [
        "no_user",
        "city",
        "employer",
        "derived_one",
        "short",
        "no_index",
        "other_tool",
        "after_continuation",
        "friend_city",
        "untouched",
    ] {
        sqlx::query(
            "INSERT INTO facts (category, key, value, source, created_at, updated_at)
             VALUES ('profile', ?, 'seeded value', 'agent', ?2, ?2)",
        )
        .bind(key)
        .bind(at)
        .execute(pool)
        .await
        .expect("seed fact");
    }
    sqlx::query(
        "INSERT INTO facts (category, key, value, source, created_at, updated_at, superseded_at)
         VALUES ('profile', 'superseded', 'old value', 'agent', ?1, ?1, ?1)",
    )
    .bind(at)
    .execute(pool)
    .await
    .expect("seed superseded fact");
}

/// Replies for the first procedures, in order: a rewrite, a refusal, a step
/// count the evidence gate rejects, and garbage. After these the mock says
/// "Mock response", which is not JSON, so the rest stay unchanged.
fn model_replies() -> Vec<crate::traits::ProviderResponse> {
    [
        r#"{"name":"tested-release","trigger_pattern":"when shipping a tested release",
            "steps":["run the tests","commit"],"skip":false}"#,
        r#"{"skip":true}"#,
        r#"{"name":"certificate-rotation","trigger_pattern":"when certificates expire",
            "steps":["renew and reload"],"skip":false}"#,
        "not json at all",
    ]
    .into_iter()
    .map(MockProvider::text_response)
    .collect()
}

type Rows = Vec<Vec<Option<String>>>;
type Snapshot = BTreeMap<String, Rows>;

async fn rows(pool: &SqlitePool, sql: &str) -> Rows {
    sqlx::query(sqlx::AssertSqlSafe(sql))
        .fetch_all(pool)
        .await
        .unwrap_or_else(|error| panic!("{sql}: {error}"))
        .iter()
        .map(|row| {
            (0..row.len())
                .map(|index| row.get::<Option<String>, _>(index))
                .collect()
        })
        .collect()
}

/// The rows consolidation owns, without their timestamps of writing.
async fn snapshot(pool: &SqlitePool, results: Rows) -> Snapshot {
    let mut snapshot = Snapshot::new();
    snapshot.insert("results".to_string(), results);
    for (table, sql) in [
        (
            "procedures",
            "SELECT name, trigger_pattern, steps, CAST(success_count AS TEXT),
                    CAST(failure_count AS TEXT), CAST(avg_duration_secs AS TEXT),
                    CAST(last_used_at IS NOT NULL AS TEXT),
                    CAST(trigger_embedding IS NOT NULL AS TEXT)
             FROM procedures ORDER BY name",
        ),
        (
            "error_solutions",
            "SELECT error_pattern, domain, solution_summary, solution_steps,
                    CAST(success_count AS TEXT), CAST(failure_count AS TEXT),
                    CAST(last_used_at IS NOT NULL AS TEXT),
                    CAST(error_embedding IS NOT NULL AS TEXT)
             FROM error_solutions ORDER BY error_pattern, domain, solution_summary",
        ),
        (
            "expertise",
            "SELECT domain, CAST(tasks_attempted AS TEXT), CAST(tasks_succeeded AS TEXT),
                    CAST(tasks_failed AS TEXT), current_level,
                    CAST(confidence_score AS TEXT), common_errors,
                    CAST(last_task_at IS NOT NULL AS TEXT)
             FROM expertise ORDER BY domain",
        ),
        (
            "behavior_patterns",
            "SELECT pattern_type, description, trigger_context, action,
                    CAST(confidence AS TEXT), CAST(occurrence_count AS TEXT)
             FROM behavior_patterns ORDER BY pattern_type, trigger_context, action",
        ),
        (
            "facts",
            "SELECT category, key, value, source, first_seen_at, source_excerpt,
                    CAST(superseded_at IS NOT NULL AS TEXT)
             FROM facts ORDER BY id",
        ),
        (
            "memory_embeddings",
            "SELECT owner_type, embedding_purpose, CAST(COUNT(*) AS TEXT)
             FROM memory_embeddings GROUP BY owner_type, embedding_purpose
             ORDER BY owner_type, embedding_purpose",
        ),
        (
            "events",
            "SELECT session_id, CAST(consolidated_at IS NOT NULL AS TEXT), CAST(COUNT(*) AS TEXT)
             FROM events GROUP BY session_id, consolidated_at IS NOT NULL
             ORDER BY session_id, consolidated_at IS NOT NULL",
        ),
    ] {
        snapshot.insert(table.to_string(), rows(pool, sql).await);
    }
    snapshot
}

fn result_row(session: &str, result: &ConsolidationResult) -> Vec<Option<String>> {
    [
        session.to_string(),
        result.events_processed.to_string(),
        result.events_consolidated.to_string(),
        result.procedures_created.to_string(),
        result.error_solutions_created.to_string(),
        result.expertise_updated.to_string(),
        result.behavior_patterns_recorded.to_string(),
        result.episodes_created.to_string(),
    ]
    .into_iter()
    .map(Some)
    .collect()
}

/// Consolidate the whole history on a fresh in-memory database.
async fn run(setup: Setup) -> Snapshot {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect("sqlite::memory:")
        .await
        .expect("in-memory database");
    aidaemon_memory::sqlite::migrate(&pool)
        .await
        .expect("memory tables");
    let event_store = Arc::new(EventStore::new(pool.clone()).await.expect("event store"));
    let plan_store = Arc::new(PlanStore::new(pool.clone()).await.expect("plan store"));
    let embeddings = Arc::new(EmbeddingService::new().expect("embedding service"));
    let memory: Arc<dyn aidaemon_memory::stores::MemoryStores> = Arc::new(
        crate::memory::sqlite_memory(pool.clone(), embeddings.clone()),
    );
    seed_memory(&pool, &embeddings).await;

    let provider = Arc::new(MockProvider::with_responses(if setup.with_models {
        model_replies()
    } else {
        Vec::new()
    }));
    let runtime = setup.with_models.then(|| {
        SharedLlmRuntime::new(
            provider.clone(),
            None,
            ProviderKind::OpenaiCompatible,
            "mock".to_string(),
        )
    });
    // The gateway records its calls in its own database, so they stay out
    // of the history this test consolidates.
    let (side_calls, _gateway_events) =
        crate::llm_runtime::gateway::test_support::gateway_with(provider).await;
    let consolidator = Consolidator::new(
        event_store.clone(),
        plan_store.clone(),
        pool.clone(),
        memory,
        runtime,
        side_calls,
        setup.with_models.then(|| embeddings.clone()),
    )
    .with_learning_evidence_gate(setup.evidence_gate);

    let base = Utc.with_ymd_and_hms(2026, 3, 1, 9, 0, 0).unwrap();
    let mut results = Rows::new();
    for (round, session) in OWNER_SESSIONS.iter().enumerate() {
        let start = base + Duration::hours(round as i64);
        for event in owner_history(session, start) {
            event_store.append(event).await.expect("append");
        }
        damage_rows(&pool, session).await;
        for plan in plans_for(session) {
            plan_store.create(&plan).await.expect("create plan");
        }
        results.push(result_row(
            session,
            &consolidate(&consolidator, session).await,
        ));
        // Nothing new: the second pass over a session changes no row.
        results.push(result_row(
            session,
            &consolidate(&consolidator, session).await,
        ));
    }

    // Sessions that must never teach anything.
    let mut mandate = History::new(MANDATE_SESSION, base + Duration::hours(5));
    mandate.push(
        EventType::UserMessage,
        json!({"content": "scheduled work", "user_role": "owner", "execution_origin": "mandate"}),
    );
    mandate.start("task-mandate", "Post the daily update");
    mandate.call("task-mandate", "terminal", Some("post"));
    mandate.call("task-mandate", "terminal", Some("confirm"));
    mandate.end("task-mandate", TaskOutcome::Succeeded, 5);
    let mut worker = History::new(WORKER_SESSION, base + Duration::hours(6));
    worker.start("task-worker", "Do a delegated step");
    worker.call("task-worker", "terminal", Some("a"));
    worker.call("task-worker", "terminal", Some("b"));
    worker.end("task-worker", TaskOutcome::Succeeded, 5);
    for event in mandate.events.into_iter().chain(worker.events) {
        event_store.append(event).await.expect("append");
    }
    for session in [MANDATE_SESSION, WORKER_SESSION] {
        results.push(result_row(
            session,
            &consolidate(&consolidator, session).await,
        ));
    }

    snapshot(&pool, results).await
}

async fn consolidate(consolidator: &Consolidator, session: &str) -> ConsolidationResult {
    consolidator
        .consolidate_session(session)
        .await
        .expect("consolidate session")
}

fn golden() -> BTreeMap<String, Snapshot> {
    serde_json::from_str(GOLDEN).expect("golden file parses")
}

fn assert_snapshots_equal(left: &Snapshot, right: &Snapshot, context: &str) {
    assert_eq!(
        left.keys().collect::<Vec<_>>(),
        right.keys().collect::<Vec<_>>(),
        "{context}: tables"
    );
    for (table, left_rows) in left {
        let right_rows = &right[table];
        for (index, (a, b)) in left_rows.iter().zip(right_rows).enumerate() {
            assert_eq!(a, b, "{context}: {table} row {index}");
        }
        assert_eq!(
            left_rows.len(),
            right_rows.len(),
            "{context}: {table} row count"
        );
    }
}

/// Regenerates the golden file. Run with `UPDATE_CONSOLIDATION_GOLDEN=1`
/// only to record a deliberate change to what consolidation writes.
async fn update_golden_if_asked() {
    if std::env::var_os("UPDATE_CONSOLIDATION_GOLDEN").is_none() {
        return;
    }
    let mut all = BTreeMap::new();
    for setup in SETUPS {
        all.insert(setup.name.to_string(), run(setup).await);
    }
    std::fs::write(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/events/testdata/consolidation.golden"
        ),
        serde_json::to_string_pretty(&all).expect("serialize golden") + "\n",
    )
    .expect("write golden");
}

#[tokio::test]
async fn consolidation_writes_the_recorded_rows() {
    update_golden_if_asked().await;
    let golden = golden();
    for setup in SETUPS {
        let rows = run(setup).await;
        assert_snapshots_equal(&rows, &golden[setup.name], setup.name);
    }
}

/// The scenario has to keep reaching what it exists to compare.
#[tokio::test]
async fn the_scenario_reaches_every_path() {
    for setup in SETUPS {
        let snapshot = run(setup).await;
        let table = |name: &str| &snapshot[name];
        let has = |rows: &Rows, text: &str| {
            rows.iter()
                .any(|row| row.iter().flatten().any(|value| value.contains(text)))
        };
        assert!(table("behavior_patterns").len() > 25, "{setup:?}");
        assert!(has(table("behavior_patterns"), "soft_iteration_warning"));
        assert!(has(table("behavior_patterns"), "unknown_pair"));
        assert!(has(table("behavior_patterns"), "fallback_tool"));
        // Known procedure names keep their definition.
        let stored = table("procedures")
            .iter()
            .find(|row| row[0].as_deref() == Some("auto_task-suc"));
        if !setup.with_models {
            let stored = stored.expect("collided procedure");
            assert_eq!(stored[1].as_deref(), Some("an older trigger"));
            assert_eq!(stored[3].as_deref(), Some("7"));
        }
        assert!(has(table("procedures"), "plan_plan-one"));
        assert!(!has(table("procedures"), "plan_plan-sho"));
        assert!(!has(table("procedures"), "Private errand"));
        assert!(!has(table("procedures"), "Unclassified errand"));
        assert!(!has(table("procedures"), "Post the daily update"));
        // The window ignores decision points; just past it does not count.
        assert!(has(table("error_solutions"), "connection refused"));
        assert!(!has(table("error_solutions"), "disk quota"));
        // The empty pattern left by a scaffolding-only error is kept.
        assert!(table("error_solutions")
            .iter()
            .any(|row| row[0].as_deref() == Some("")));
        // Consolidation itself never embeds into the index.
        assert!(table("memory_embeddings").is_empty());
        assert!(has(table("facts"), "user_stated"));
        if setup.with_models {
            assert!(has(table("procedures"), "tested-release"));
            assert!(!has(table("procedures"), "certificate-rotation"));
            assert!(!has(table("procedures"), "auto_task-dup"));
        } else {
            assert!(has(table("procedures"), "auto_task-dup"));
        }
    }
}
