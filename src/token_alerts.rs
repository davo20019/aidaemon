#![allow(dead_code)] // Phase 1 token alert code is staged but not yet wired into runtime.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::Context;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use crate::traits::{NotificationEntry, StateStore};

const TOKENS_THRESHOLD_15M: i64 = 200_000;
const CALLS_THRESHOLD_10M: i64 = 40;
const WAIT_TASK_CALLS_THRESHOLD_10M: i64 = 10;
const ALERT_COOLDOWN_MINUTES: i64 = 30;
const ALERT_GROWTH_FACTOR: f64 = 1.25;
const MAX_GOAL_ALERTS_PER_TICK: usize = 5;
const MAX_SESSION_ALERTS_PER_TICK: usize = 5;

#[derive(Debug, Clone)]
struct WaitTaskSpike {
    task_id: String,
    task_description: String,
    tokens_10m: i64,
    calls_10m: i64,
}

#[derive(Debug, Clone)]
struct GoalSpike {
    goal_id: String,
    session_id: String,
    goal_description: String,
    tokens_15m: i64,
    calls_10m: i64,
    wait_task: Option<WaitTaskSpike>,
}

#[derive(Debug, Clone)]
struct SessionSpike {
    session_id: String,
    tokens_15m: i64,
    calls_10m: i64,
}

/// Phase 1 token spike detection:
/// - Detect high token/call bursts on active goals and sessions.
/// - Enqueue critical notifications with cooldown + growth gating.
/// - No automatic cancellation yet.
pub async fn run_phase1_token_alert_scan(
    state: Arc<dyn StateStore>,
    pool: SqlitePool,
    owner_alert_sessions: Vec<String>,
) -> anyhow::Result<usize> {
    let owner_alert_sessions = dedupe_sessions(owner_alert_sessions);
    let now = Utc::now();

    let goal_spikes = detect_goal_spikes(&pool).await?;
    let session_spikes = detect_session_spikes(&pool).await?;

    if goal_spikes.is_empty() && session_spikes.is_empty() {
        return Ok(0);
    }

    let mut queued = 0usize;
    let mut alerted_sessions_this_tick: HashSet<String> = HashSet::new();

    for spike in goal_spikes.into_iter().take(MAX_GOAL_ALERTS_PER_TICK) {
        if !alerted_sessions_this_tick.insert(spike.session_id.clone()) {
            continue;
        }
        if !should_alert_now(
            &pool,
            "goal",
            &spike.goal_id,
            spike.tokens_15m,
            spike.calls_10m,
            now,
        )
        .await?
        {
            continue;
        }

        let message = build_goal_alert_message(&spike);
        let mut targets = owner_alert_sessions.clone();
        targets.push(spike.session_id.clone());
        targets = dedupe_sessions(targets);

        let mut any_enqueued = false;
        for target in targets {
            let entry = NotificationEntry::new(&spike.goal_id, &target, "token_alert", &message);
            match state.enqueue_notification(&entry).await {
                Ok(_) => {
                    any_enqueued = true;
                }
                Err(e) => {
                    warn!(
                        goal_id = %spike.goal_id,
                        session_id = %target,
                        error = %e,
                        "Failed to enqueue goal token alert"
                    );
                }
            }
        }

        if any_enqueued {
            upsert_alert_state(
                &pool,
                "goal",
                &spike.goal_id,
                spike.tokens_15m,
                spike.calls_10m,
                now,
            )
            .await?;
            queued += 1;
        }
    }

    for spike in session_spikes.into_iter().take(MAX_SESSION_ALERTS_PER_TICK) {
        if alerted_sessions_this_tick.contains(&spike.session_id) {
            continue;
        }
        if !should_alert_now(
            &pool,
            "session",
            &spike.session_id,
            spike.tokens_15m,
            spike.calls_10m,
            now,
        )
        .await?
        {
            continue;
        }

        let message = build_session_alert_message(&spike);
        let mut targets = owner_alert_sessions.clone();
        if !is_internal_session(&spike.session_id) {
            targets.push(spike.session_id.clone());
        }
        targets = dedupe_sessions(targets);

        if targets.is_empty() {
            continue;
        }

        let synthetic_goal_id = format!("token-monitor:{}", sanitize_scope_id(&spike.session_id));
        let mut any_enqueued = false;
        for target in targets {
            let entry =
                NotificationEntry::new(&synthetic_goal_id, &target, "token_alert", &message);
            match state.enqueue_notification(&entry).await {
                Ok(_) => {
                    any_enqueued = true;
                }
                Err(e) => {
                    warn!(
                        monitored_session = %spike.session_id,
                        target_session = %target,
                        error = %e,
                        "Failed to enqueue session token alert"
                    );
                }
            }
        }

        if any_enqueued {
            upsert_alert_state(
                &pool,
                "session",
                &spike.session_id,
                spike.tokens_15m,
                spike.calls_10m,
                now,
            )
            .await?;
            queued += 1;
            alerted_sessions_this_tick.insert(spike.session_id.clone());
        }
    }

    if queued > 0 {
        info!(queued, "Token spike detector queued alerts");
    }
    Ok(queued)
}

fn dedupe_sessions(sessions: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for session in sessions {
        let trimmed = session.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            out.push(trimmed.to_string());
        }
    }
    out
}

fn is_internal_session(session_id: &str) -> bool {
    session_id.starts_with("background:")
        || session_id.starts_with("sub-")
        || session_id.starts_with("scheduled_")
        || session_id == "system"
}

fn sanitize_scope_id(scope: &str) -> String {
    scope
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn format_tokens(tokens: i64) -> String {
    let negative = tokens < 0;
    let digits = tokens.abs().to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    if negative {
        format!("-{}", out)
    } else {
        out
    }
}

fn truncate_line(input: &str, max_chars: usize) -> String {
    let compact = input.replace('\n', " ");
    let mut out: String = compact.chars().take(max_chars).collect();
    if compact.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn build_goal_alert_message(spike: &GoalSpike) -> String {
    let mut msg = format!(
        "Token alert: high LLM usage detected for an active goal.\n\
         Goal: {}\n\
         Goal ID: {}\n\
         Last 15m: {} tokens\n\
         Last 10m: {} LLM calls\n",
        truncate_line(&spike.goal_description, 140),
        spike.goal_id,
        format_tokens(spike.tokens_15m),
        spike.calls_10m
    );

    if let Some(wait) = &spike.wait_task {
        msg.push_str(&format!(
            "Runaway pattern: wait-like task loop.\n\
             Task: {} (id: {})\n\
             Last 10m on this task: {} tokens, {} LLM calls\n",
            truncate_line(&wait.task_description, 120),
            wait.task_id,
            format_tokens(wait.tokens_10m),
            wait.calls_10m
        ));
    }

    msg.push_str(&format!(
        "Suggested action: cancel/pause this schedule if unexpected.\n\
         - manage_memories(action='cancel_scheduled', goal_id='{}')\n\
         - manage_memories(action='pause_scheduled', goal_id='{}')",
        spike.goal_id, spike.goal_id
    ));

    msg
}

fn build_session_alert_message(spike: &SessionSpike) -> String {
    format!(
        "Token alert: high LLM usage detected for session `{}`.\n\
         Last 15m: {} tokens\n\
         Last 10m: {} LLM calls\n\
         Suggested action: inspect active scheduled goals and cancel/pause unexpected ones.",
        spike.session_id,
        format_tokens(spike.tokens_15m),
        spike.calls_10m
    )
}

async fn should_alert_now(
    pool: &SqlitePool,
    scope_type: &str,
    scope_id: &str,
    tokens: i64,
    calls: i64,
    now: DateTime<Utc>,
) -> anyhow::Result<bool> {
    let row = sqlx::query(
        "SELECT last_alert_at, last_metric_tokens, last_metric_calls
         FROM token_alert_state
         WHERE scope_type = ? AND scope_id = ?",
    )
    .bind(scope_type)
    .bind(scope_id)
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        return Ok(true);
    };

    let last_alert_at: String = row.get("last_alert_at");
    let last_tokens: i64 = row.get("last_metric_tokens");
    let last_calls: i64 = row.get("last_metric_calls");

    let parsed = DateTime::parse_from_rfc3339(&last_alert_at)
        .with_context(|| format!("invalid token_alert_state.last_alert_at: {}", last_alert_at));
    let Ok(last_alert_dt) = parsed else {
        return Ok(true);
    };
    let last_alert_dt = last_alert_dt.with_timezone(&Utc);

    let within_cooldown = now - last_alert_dt < ChronoDuration::minutes(ALERT_COOLDOWN_MINUTES);
    if !within_cooldown {
        return Ok(true);
    }

    let growth_tokens = ((last_tokens.max(1) as f64) * ALERT_GROWTH_FACTOR).ceil() as i64;
    let growth_calls = ((last_calls.max(1) as f64) * ALERT_GROWTH_FACTOR).ceil() as i64;
    let significant_growth = tokens >= growth_tokens || calls >= growth_calls;

    Ok(significant_growth)
}

async fn upsert_alert_state(
    pool: &SqlitePool,
    scope_type: &str,
    scope_id: &str,
    tokens: i64,
    calls: i64,
    now: DateTime<Utc>,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO token_alert_state (scope_type, scope_id, last_alert_at, last_metric_tokens, last_metric_calls)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(scope_type, scope_id) DO UPDATE SET
            last_alert_at = excluded.last_alert_at,
            last_metric_tokens = excluded.last_metric_tokens,
            last_metric_calls = excluded.last_metric_calls",
    )
    .bind(scope_type)
    .bind(scope_id)
    .bind(now.to_rfc3339())
    .bind(tokens)
    .bind(calls)
    .execute(pool)
    .await?;

    Ok(())
}

async fn detect_goal_spikes(pool: &SqlitePool) -> anyhow::Result<Vec<GoalSpike>> {
    let mut by_goal: HashMap<String, GoalSpike> = HashMap::new();

    let rows = sqlx::query(
        "SELECT
            g.id AS goal_id,
            g.session_id AS session_id,
            g.description AS goal_description,
            COALESCE(SUM(a.tokens_used), 0) AS tokens_15m,
            COALESCE(SUM(CASE
                WHEN julianday(a.created_at) >= julianday('now', '-10 minutes') THEN 1
                ELSE 0
            END), 0) AS calls_10m
         FROM goals g
         JOIN tasks t ON t.goal_id = g.id
         JOIN task_activity a ON a.task_id = t.id
         WHERE g.domain = 'orchestration'
           AND g.status = 'active'
           AND a.activity_type = 'llm_call'
           AND a.tokens_used IS NOT NULL
           AND julianday(a.created_at) >= julianday('now', '-15 minutes')
         GROUP BY g.id, g.session_id, g.description
         HAVING COALESCE(SUM(a.tokens_used), 0) >= ?
            OR COALESCE(SUM(CASE
                WHEN julianday(a.created_at) >= julianday('now', '-10 minutes') THEN 1
                ELSE 0
            END), 0) >= ?
         ORDER BY tokens_15m DESC",
    )
    .bind(TOKENS_THRESHOLD_15M)
    .bind(CALLS_THRESHOLD_10M)
    .fetch_all(pool)
    .await?;

    for row in rows {
        let spike = GoalSpike {
            goal_id: row.get::<String, _>("goal_id"),
            session_id: row.get::<String, _>("session_id"),
            goal_description: row.get::<String, _>("goal_description"),
            tokens_15m: row.get::<i64, _>("tokens_15m"),
            calls_10m: row.get::<i64, _>("calls_10m"),
            wait_task: None,
        };
        by_goal.insert(spike.goal_id.clone(), spike);
    }

    let wait_rows = sqlx::query(
        "SELECT
            g.id AS goal_id,
            g.session_id AS session_id,
            g.description AS goal_description,
            t.id AS task_id,
            t.description AS task_description,
            COALESCE(SUM(a.tokens_used), 0) AS tokens_10m,
            COUNT(*) AS calls_10m
         FROM goals g
         JOIN tasks t ON t.goal_id = g.id
         JOIN task_activity a ON a.task_id = t.id
         WHERE g.domain = 'orchestration'
           AND g.status = 'active'
           AND a.activity_type = 'llm_call'
           AND a.tokens_used IS NOT NULL
           AND julianday(a.created_at) >= julianday('now', '-10 minutes')
           AND lower(trim(t.description)) LIKE 'wait%'
         GROUP BY g.id, g.session_id, g.description, t.id, t.description
         HAVING COUNT(*) >= ?
         ORDER BY calls_10m DESC",
    )
    .bind(WAIT_TASK_CALLS_THRESHOLD_10M)
    .fetch_all(pool)
    .await?;

    for row in wait_rows {
        let goal_id: String = row.get("goal_id");
        let wait_spike = WaitTaskSpike {
            task_id: row.get("task_id"),
            task_description: row.get("task_description"),
            tokens_10m: row.get("tokens_10m"),
            calls_10m: row.get("calls_10m"),
        };

        if let Some(existing) = by_goal.get_mut(&goal_id) {
            existing.wait_task = Some(wait_spike);
            if existing.calls_10m < CALLS_THRESHOLD_10M {
                existing.calls_10m = CALLS_THRESHOLD_10M;
            }
            continue;
        }

        by_goal.insert(
            goal_id.clone(),
            GoalSpike {
                goal_id,
                session_id: row.get("session_id"),
                goal_description: row.get("goal_description"),
                tokens_15m: row.get("tokens_10m"),
                calls_10m: row.get("calls_10m"),
                wait_task: Some(wait_spike),
            },
        );
    }

    let mut spikes: Vec<GoalSpike> = by_goal.into_values().collect();
    spikes.sort_by(|a, b| b.tokens_15m.cmp(&a.tokens_15m));
    Ok(spikes)
}

async fn detect_session_spikes(pool: &SqlitePool) -> anyhow::Result<Vec<SessionSpike>> {
    let rows = sqlx::query(
        "SELECT
            session_id,
            COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens_15m,
            COALESCE(SUM(CASE
                WHEN julianday(created_at) >= julianday('now', '-10 minutes') THEN 1
                ELSE 0
            END), 0) AS calls_10m
         FROM token_usage
         WHERE julianday(created_at) >= julianday('now', '-15 minutes')
         GROUP BY session_id
         HAVING COALESCE(SUM(input_tokens + output_tokens), 0) >= ?
            OR COALESCE(SUM(CASE
                WHEN julianday(created_at) >= julianday('now', '-10 minutes') THEN 1
                ELSE 0
            END), 0) >= ?
         ORDER BY tokens_15m DESC",
    )
    .bind(TOKENS_THRESHOLD_15M)
    .bind(CALLS_THRESHOLD_10M)
    .fetch_all(pool)
    .await?;

    let mut spikes = Vec::with_capacity(rows.len());
    for row in rows {
        spikes.push(SessionSpike {
            session_id: row.get("session_id"),
            tokens_15m: row.get("tokens_15m"),
            calls_10m: row.get("calls_10m"),
        });
    }
    Ok(spikes)
}
