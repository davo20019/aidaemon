use anyhow::Context;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};
use std::path::{Path, PathBuf};
use std::str::FromStr;

fn runtime_working_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn resolve_runtime_env_file_path(working_dir: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("AIDAEMON_ENV_FILE") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            let candidate = PathBuf::from(trimmed);
            return if candidate.is_absolute() {
                candidate
            } else {
                working_dir.join(candidate)
            };
        }
    }
    working_dir.join(".env")
}

fn canonical_task_outcome(value: &serde_json::Value) -> Option<&str> {
    let status = match value.get("status").and_then(|status| status.as_str()) {
        Some(status @ ("completed" | "failed" | "cancelled")) => status,
        _ => return None,
    };
    if let Some(outcome) = value.get("outcome") {
        return match outcome.as_str() {
            Some("succeeded") => Some("succeeded"),
            Some("partial") => Some("partial"),
            Some("failed") => Some("failed"),
            _ => None,
        };
    }
    match status {
        "completed" => Some("succeeded"),
        "failed" | "cancelled" => Some("failed"),
        _ => None,
    }
}

fn canonical_error_type(value: &serde_json::Value) -> &str {
    match value
        .get("error_type")
        .and_then(|error_type| error_type.as_str())
    {
        Some(
            error_type @ ("tool_error" | "llm_error" | "timeout" | "rate_limit"
            | "permission_denied" | "internal" | "cancelled"),
        ) => error_type,
        _ => "unknown",
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct TelemetryReconciliationCounts {
    correlated: usize,
    token_only: usize,
    event_only: usize,
    duplicate_token_rows: usize,
    duplicate_event_rows: usize,
    legacy_token_rows: usize,
    legacy_event_rows: usize,
}

fn telemetry_reconciliation_counts(
    token_call_ids: &[Option<String>],
    event_rows: &[(Option<String>, bool)],
) -> TelemetryReconciliationCounts {
    let mut token_frequencies = std::collections::HashMap::<String, usize>::new();
    let mut legacy_token_rows = 0usize;
    for call_id in token_call_ids {
        match call_id.as_deref().filter(|call_id| !call_id.is_empty()) {
            Some(call_id) => *token_frequencies.entry(call_id.to_string()).or_insert(0) += 1,
            None => legacy_token_rows += 1,
        }
    }

    let mut event_frequencies = std::collections::HashMap::<String, usize>::new();
    let mut legacy_event_rows = 0usize;
    for (call_id, usage_present) in event_rows {
        if call_id.as_deref().is_none_or(str::is_empty) {
            legacy_event_rows += 1;
        }
        if !usage_present {
            continue;
        }
        if let Some(call_id) = call_id.as_deref().filter(|call_id| !call_id.is_empty()) {
            *event_frequencies.entry(call_id.to_string()).or_insert(0) += 1;
        }
    }

    let token_ids: std::collections::HashSet<&String> = token_frequencies.keys().collect();
    let event_ids: std::collections::HashSet<&String> = event_frequencies.keys().collect();
    TelemetryReconciliationCounts {
        correlated: token_ids.intersection(&event_ids).count(),
        token_only: token_ids.difference(&event_ids).count(),
        event_only: event_ids.difference(&token_ids).count(),
        duplicate_token_rows: token_frequencies
            .values()
            .map(|count| count.saturating_sub(1))
            .sum(),
        duplicate_event_rows: event_frequencies
            .values()
            .map(|count| count.saturating_sub(1))
            .sum(),
        legacy_token_rows,
        legacy_event_rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_task_outcome_rejects_unrecognized_values() {
        assert_eq!(
            canonical_task_outcome(&json!({"status": "completed", "outcome": "mostly_done"})),
            None
        );
        assert_eq!(
            canonical_task_outcome(&json!({"status": "completed"})),
            Some("succeeded")
        );
        assert_eq!(
            canonical_task_outcome(&json!({"outcome": "succeeded"})),
            None
        );
    }

    #[test]
    fn canonical_error_type_rejects_unrecognized_values() {
        assert_eq!(
            canonical_error_type(&json!({"error_type": "networkish"})),
            "unknown"
        );
        assert_eq!(
            canonical_error_type(&json!({"error_type": "tool_error"})),
            "tool_error"
        );
    }

    #[test]
    fn reconciliation_counts_duplicate_and_legacy_rows() {
        let counts = telemetry_reconciliation_counts(
            &[
                Some("call-a".to_string()),
                Some("call-a".to_string()),
                Some("call-b".to_string()),
                None,
            ],
            &[
                (Some("call-a".to_string()), true),
                (Some("call-c".to_string()), true),
                (Some("call-c".to_string()), true),
                (None, true),
                (None, false),
                (Some("call-d".to_string()), false),
            ],
        );

        assert_eq!(counts.correlated, 1);
        assert_eq!(counts.token_only, 1);
        assert_eq!(counts.event_only, 1);
        assert_eq!(counts.duplicate_token_rows, 1);
        assert_eq!(counts.duplicate_event_rows, 1);
        assert_eq!(counts.legacy_token_rows, 1);
        assert_eq!(counts.legacy_event_rows, 2);
    }
}

async fn print_eval_task(pool: &SqlitePool, task_id: &str) -> anyhow::Result<()> {
    use aidaemon::TaskEndData;
    use aidaemon::harness_eval::report::{format_eval_task_report, EvalTaskRow};

    let row = sqlx::query(
        r#"
        SELECT session_id, created_at, data
        FROM events
        WHERE event_type = 'task_end' AND task_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await?
    .with_context(|| format!("no TaskEnd found for task_id={task_id}"))?;

    let session_id: String = row.get("session_id");
    let created_at: String = row.get("created_at");
    let data_json: String = row.get("data");
    let task_end: TaskEndData = serde_json::from_str(&data_json).context("parse TaskEnd JSON")?;
    let eval = task_end
        .harness_eval
        .clone()
        .context("TaskEnd has no harness_eval snapshot (enable [diagnostics.harness_eval])")?;

    let report = format_eval_task_report(&EvalTaskRow {
        task_id: task_id.to_string(),
        session_id,
        created_at,
        task_end,
        eval,
    });
    println!("{report}");
    Ok(())
}

async fn print_eval_summary(pool: &SqlitePool, hours: i64, root_only: bool) -> anyhow::Result<()> {
    use aidaemon::TaskEndData;
    use aidaemon::harness_eval::report::{aggregate_summary, format_eval_summary_row, EvalTaskRow};

    let rows = sqlx::query(
        r#"
        SELECT session_id, task_id, created_at, data
        FROM events
        WHERE event_type = 'task_end'
          AND created_at >= datetime('now', printf('-%d hours', ?))
          AND json_extract(data, '$.harness_eval') IS NOT NULL
        ORDER BY created_at DESC
        "#,
    )
    .bind(hours)
    .fetch_all(pool)
    .await?;

    let mut eval_rows = Vec::new();
    for row in rows {
        let data_json: String = row.get("data");
        let Ok(task_end) = serde_json::from_str::<TaskEndData>(&data_json) else {
            continue;
        };
        let Some(eval) = task_end.harness_eval.clone() else {
            continue;
        };
        if root_only && eval.depth > 0 {
            continue;
        }
        if root_only && eval.parent_task_id.is_some() {
            continue;
        }
        eval_rows.push(EvalTaskRow {
            task_id: row.get("task_id"),
            session_id: row.get("session_id"),
            created_at: row.get("created_at"),
            task_end,
            eval,
        });
    }

    let stats = aggregate_summary(&eval_rows);
    println!("{}", format_eval_summary_row(&stats, hours, root_only));
    Ok(())
}

async fn record_fixture_from_session(
    pool: &SqlitePool,
    session_id: &str,
    task_id: Option<&str>,
    output: Option<&str>,
    include_text: bool,
) -> anyhow::Result<()> {
    use aidaemon::{EventType, TaskEndData, ToolCallData, UserMessageData};
    use aidaemon::harness_eval::fixture::{build_recorded_fixture, fixtures_dir};

    let rows = if let Some(task_id) = task_id {
        sqlx::query(
            r#"
            SELECT event_type, task_id, tool_name, data, created_at
            FROM events
            WHERE session_id = ? AND task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(task_id)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query(
            r#"
            SELECT event_type, task_id, tool_name, data, created_at
            FROM events
            WHERE session_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(pool)
        .await?
    };

    let mut task_end: Option<TaskEndData> = None;
    let mut resolved_task_id = task_id.map(str::to_string);
    let mut user_text = String::new();
    let mut tool_names = Vec::new();

    for row in &rows {
        let event_type: String = row.get("event_type");
        let data_json: String = row.get("data");
        if event_type == EventType::TaskEnd.as_str() {
            if let Ok(end) = serde_json::from_str::<TaskEndData>(&data_json) {
                if end.harness_eval.is_some() {
                    resolved_task_id = Some(end.task_id.clone());
                    task_end = Some(end);
                }
            }
        } else if event_type == EventType::UserMessage.as_str() && user_text.is_empty() {
            if let Ok(msg) = serde_json::from_str::<UserMessageData>(&data_json) {
                user_text = msg.content;
            }
        } else if event_type == EventType::ToolCall.as_str() {
            if let Ok(data) = serde_json::from_str::<ToolCallData>(&data_json) {
                tool_names.push(data.name);
            }
        }
    }

    let task_end = task_end.context("no TaskEnd with harness_eval found for session/task")?;
    let eval = task_end
        .harness_eval
        .as_ref()
        .context("TaskEnd missing harness_eval")?;
    let task_id = resolved_task_id.context("could not resolve task_id")?;
    if user_text.is_empty() {
        user_text = task_end
            .summary
            .clone()
            .unwrap_or_else(|| "(unknown user text)".to_string());
    }

    let name = task_id.chars().take(32).collect::<String>();
    let mut fixture =
        build_recorded_fixture(&name, session_id, &user_text, eval, &task_end, &tool_names);
    if include_text {
        if let Some(summary) = task_end.summary.as_ref() {
            fixture.expect.response_contains = vec![summary.chars().take(40).collect()];
        }
    }

    let output_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| fixtures_dir().join(format!("{name}.yaml")));
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let yaml = serde_yaml::to_string(&fixture)?;
    std::fs::write(&output_path, yaml)?;
    println!("Recorded fixture -> {}", output_path.display());
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let msg_search = args
        .windows(2)
        .find(|w| w[0] == "--search")
        .map(|w| w[1].clone());
    let msg_search_limit = args
        .windows(2)
        .find(|w| w[0] == "--search-limit")
        .map(|w| w[1].parse::<i64>())
        .transpose()?
        .unwrap_or(10)
        .clamp(1, 200);
    let msg_search_context = args
        .windows(2)
        .find(|w| w[0] == "--search-context")
        .map(|w| w[1].parse::<i64>())
        .transpose()?
        .unwrap_or(6)
        .clamp(0, 50);
    let task_filter = args
        .windows(2)
        .find(|w| w[0] == "--task")
        .map(|w| w[1].clone());
    let inv_filter = args
        .windows(2)
        .find(|w| w[0] == "--invocation")
        .map(|w| w[1].parse::<i64>())
        .transpose()?;
    let session_filter = args
        .windows(2)
        .find(|w| w[0] == "--session")
        .map(|w| w[1].clone());
    let repair_stale_cli_hours = args
        .windows(2)
        .find(|w| w[0] == "--repair-stale-cli")
        .map(|w| w[1].parse::<i64>())
        .transpose()?;
    let token_hours = args
        .windows(2)
        .find(|w| w[0] == "--token-hours")
        .map(|w| w[1].parse::<i64>())
        .transpose()?
        .unwrap_or(7)
        .clamp(1, 720);
    let eval_task = args
        .windows(2)
        .find(|w| w[0] == "--eval-task")
        .map(|w| w[1].clone());
    let eval_summary = args.iter().any(|arg| arg == "--eval-summary");
    let eval_hours = args
        .windows(2)
        .find(|w| w[0] == "--eval-hours")
        .map(|w| w[1].parse::<i64>())
        .transpose()?
        .unwrap_or(24)
        .clamp(1, 720);
    let eval_include_subagents = args.iter().any(|arg| arg == "--eval-include-subagents");
    let record_fixture_session = args
        .windows(2)
        .find(|w| w[0] == "--record-fixture")
        .map(|w| w[1].clone());
    let record_fixture_output = args
        .windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| w[1].clone());
    let record_fixture_include_text = args.iter().any(|arg| arg == "--include-text");

    let env_path = resolve_runtime_env_file_path(&runtime_working_dir());
    if env_path.exists() {
        let _ = dotenvy::from_path(&env_path);
    }

    let db_path = std::env::var("AIDAEMON_DB_PATH").unwrap_or_else(|_| "aidaemon.db".to_string());
    let key = std::env::var("AIDAEMON_ENCRYPTION_KEY")
        .context("AIDAEMON_ENCRYPTION_KEY is not set in environment/.env")?;
    if key.trim().is_empty() {
        anyhow::bail!("AIDAEMON_ENCRYPTION_KEY is empty");
    }

    let escaped_key = key.replace('\'', "''");
    let opts = SqliteConnectOptions::from_str(&format!("sqlite:{}", db_path))?
        .pragma("key", format!("'{}'", escaped_key))
        .pragma("journal_mode", "WAL");

    let pool = SqlitePool::connect_with(opts).await?;

    if let Some(task_id) = eval_task.as_deref() {
        print_eval_task(&pool, task_id).await?;
        return Ok(());
    }
    if eval_summary {
        print_eval_summary(&pool, eval_hours, !eval_include_subagents).await?;
        return Ok(());
    }
    if let Some(session_id) = record_fixture_session.as_deref() {
        record_fixture_from_session(
            &pool,
            session_id,
            task_filter.as_deref(),
            record_fixture_output.as_deref(),
            record_fixture_include_text,
        )
        .await?;
        return Ok(());
    }

    if let Some(needle) = msg_search.as_ref() {
        println!("== Message Search ==");
        println!(
            "- needle={:?} limit={} context={}",
            needle, msg_search_limit, msg_search_context
        );

        let rows = sqlx::query(
            r#"
            WITH convo AS (
                SELECT
                    id AS event_id,
                    COALESCE(NULLIF(CAST(json_extract(data, '$.message_id') AS TEXT), ''), CAST(id AS TEXT)) AS message_id,
                    session_id,
                    CASE event_type
                        WHEN 'user_message' THEN 'user'
                        WHEN 'assistant_response' THEN 'assistant'
                        WHEN 'tool_result' THEN 'tool'
                        ELSE event_type
                    END AS role,
                    COALESCE(tool_name, CAST(json_extract(data, '$.name') AS TEXT)) AS tool_name,
                    created_at,
                    CASE event_type
                        WHEN 'user_message' THEN CAST(json_extract(data, '$.content') AS TEXT)
                        WHEN 'assistant_response' THEN CAST(json_extract(data, '$.content') AS TEXT)
                        WHEN 'tool_result' THEN CAST(json_extract(data, '$.result') AS TEXT)
                        ELSE NULL
                    END AS content
                FROM events
                WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
            )
            SELECT message_id, event_id, session_id, role, tool_name, created_at,
                   substr(COALESCE(content, ''), 1, 240) AS content_preview
            FROM convo
            WHERE COALESCE(content, '') LIKE '%' || ? || '%'
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(needle)
        .bind(msg_search_limit)
        .fetch_all(&pool)
        .await?;

        if rows.is_empty() {
            println!("(no matches)");
        } else {
            for row in &rows {
                let msg_id: String = row.get("message_id");
                let event_id: i64 = row.get("event_id");
                let session_id: String = row.get("session_id");
                let role: String = row.get("role");
                let tool_name: Option<String> = row.try_get("tool_name").unwrap_or(None);
                let created_at: String = row.get("created_at");
                let preview: String = row.get("content_preview");

                println!(
                    "- msg_id={} event_id={} session={} role={} tool={:?} at={}\n  {}",
                    msg_id,
                    event_id,
                    session_id,
                    role,
                    tool_name,
                    created_at,
                    preview.replace('\n', " ")
                );

                if msg_search_context > 0 {
                    // Surrounding context inside the same session for quick forensics.
                    let before = sqlx::query(
                        r#"
                        WITH convo AS (
                            SELECT
                                session_id,
                                CASE event_type
                                    WHEN 'user_message' THEN 'user'
                                    WHEN 'assistant_response' THEN 'assistant'
                                    WHEN 'tool_result' THEN 'tool'
                                    ELSE event_type
                                END AS role,
                                COALESCE(tool_name, CAST(json_extract(data, '$.name') AS TEXT)) AS tool_name,
                                created_at,
                                CASE event_type
                                    WHEN 'user_message' THEN CAST(json_extract(data, '$.content') AS TEXT)
                                    WHEN 'assistant_response' THEN CAST(json_extract(data, '$.content') AS TEXT)
                                    WHEN 'tool_result' THEN CAST(json_extract(data, '$.result') AS TEXT)
                                    ELSE NULL
                                END AS content
                            FROM events
                            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
                        )
                        SELECT role, tool_name, created_at,
                               substr(COALESCE(content, ''), 1, 140) AS content_preview
                        FROM convo
                        WHERE session_id = ?
                          AND created_at < ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        "#,
                    )
                    .bind(&session_id)
                    .bind(&created_at)
                    .bind(msg_search_context)
                    .fetch_all(&pool)
                    .await?;

                    let after = sqlx::query(
                        r#"
                        WITH convo AS (
                            SELECT
                                session_id,
                                CASE event_type
                                    WHEN 'user_message' THEN 'user'
                                    WHEN 'assistant_response' THEN 'assistant'
                                    WHEN 'tool_result' THEN 'tool'
                                    ELSE event_type
                                END AS role,
                                COALESCE(tool_name, CAST(json_extract(data, '$.name') AS TEXT)) AS tool_name,
                                created_at,
                                CASE event_type
                                    WHEN 'user_message' THEN CAST(json_extract(data, '$.content') AS TEXT)
                                    WHEN 'assistant_response' THEN CAST(json_extract(data, '$.content') AS TEXT)
                                    WHEN 'tool_result' THEN CAST(json_extract(data, '$.result') AS TEXT)
                                    ELSE NULL
                                END AS content
                            FROM events
                            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
                        )
                        SELECT role, tool_name, created_at,
                               substr(COALESCE(content, ''), 1, 140) AS content_preview
                        FROM convo
                        WHERE session_id = ?
                          AND created_at > ?
                        ORDER BY created_at ASC
                        LIMIT ?
                        "#,
                    )
                    .bind(&session_id)
                    .bind(&created_at)
                    .bind(msg_search_context)
                    .fetch_all(&pool)
                    .await?;

                    if !before.is_empty() || !after.is_empty() {
                        println!("  -- context --");
                        for ctx_row in before.iter().rev() {
                            println!(
                                "  - {} tool={:?} at={}  {}",
                                ctx_row.get::<String, _>("role"),
                                ctx_row
                                    .try_get::<Option<String>, _>("tool_name")
                                    .unwrap_or(None),
                                ctx_row.get::<String, _>("created_at"),
                                ctx_row
                                    .get::<String, _>("content_preview")
                                    .replace('\n', " ")
                            );
                        }
                        println!(
                            "  - {} tool={:?} at={}  {}",
                            role,
                            tool_name,
                            created_at,
                            preview.replace('\n', " ")
                        );
                        for ctx_row in after {
                            println!(
                                "  - {} tool={:?} at={}  {}",
                                ctx_row.get::<String, _>("role"),
                                ctx_row
                                    .try_get::<Option<String>, _>("tool_name")
                                    .unwrap_or(None),
                                ctx_row.get::<String, _>("created_at"),
                                ctx_row
                                    .get::<String, _>("content_preview")
                                    .replace('\n', " ")
                            );
                        }
                    }
                }
            }
        }
        println!();
    }

    if let Some(hours) = repair_stale_cli_hours {
        println!("== Repair Stale CLI Agent Invocations ==");
        let result = sqlx::query(
            r#"
            UPDATE cli_agent_invocations
               SET completed_at = started_at,
                   exit_code = NULL,
                   output_summary = 'STALE: closed by db_probe repair at ' || datetime('now') || ' (no completion recorded)',
                   success = 0,
                   duration_secs = 0.0
             WHERE completed_at IS NULL
               AND started_at < datetime('now', '-' || ? || ' hours')
            "#,
        )
        .bind(hours)
        .execute(&pool)
        .await?;
        println!(
            "- closed {} invocation(s) older than {} hours",
            result.rows_affected(),
            hours
        );
    }

    println!("== Recent CLI Agent Invocations ==");
    let invocations = sqlx::query(
        r#"
        SELECT id, agent_name, prompt_summary, started_at, completed_at, success, exit_code, duration_secs
        FROM cli_agent_invocations
        ORDER BY id DESC
        LIMIT 12
        "#,
    )
    .fetch_all(&pool)
    .await?;
    for row in invocations {
        println!(
            "- id={} agent={} success={:?} exit={:?} started={} completed={:?} dur={:?}s\n  prompt={}",
            row.get::<i64, _>("id"),
            row.get::<String, _>("agent_name"),
            row.try_get::<Option<i64>, _>("success").unwrap_or(None),
            row.try_get::<Option<i64>, _>("exit_code").unwrap_or(None),
            row.get::<String, _>("started_at"),
            row.try_get::<Option<String>, _>("completed_at").unwrap_or(None),
            row.try_get::<Option<f64>, _>("duration_secs").unwrap_or(None),
            row.get::<String, _>("prompt_summary")
        );
    }

    println!("\n== Open CLI Agent Invocations (completed_at IS NULL) ==");
    match sqlx::query(
        r#"
        SELECT id, session_id, agent_name, prompt_summary, started_at
        FROM cli_agent_invocations
        WHERE completed_at IS NULL
        ORDER BY started_at DESC
        LIMIT 20
        "#,
    )
    .fetch_all(&pool)
    .await
    {
        Ok(rows) => {
            if rows.is_empty() {
                println!("(none)");
            } else {
                for row in rows {
                    println!(
                        "- id={} session={} agent={} started={}\n  prompt={}",
                        row.get::<i64, _>("id"),
                        row.get::<String, _>("session_id"),
                        row.get::<String, _>("agent_name"),
                        row.get::<String, _>("started_at"),
                        row.get::<String, _>("prompt_summary")
                    );
                }
            }
        }
        Err(e) => {
            println!("(failed to query open invocations: {})", e);
        }
    }

    println!("\n== Token Usage (Last {} Hours) ==", token_hours);
    match sqlx::query(
        r#"
        SELECT
          COUNT(*) AS request_count,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
          COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens
        FROM token_usage
        WHERE created_at >= datetime('now', '-' || ? || ' hours')
        "#,
    )
    .bind(token_hours)
    .fetch_one(&pool)
    .await
    {
        Ok(row) => {
            let reqs: i64 = row.get("request_count");
            let input: i64 = row.get("input_tokens");
            let output: i64 = row.get("output_tokens");
            let cached: i64 = row.get("cached_input_tokens");
            let cache_created: i64 = row.get("cache_creation_input_tokens");
            let fresh = input.saturating_sub(cached);
            println!(
                "- requests={} input_tokens={} cached_input_tokens={} fresh_input_tokens={} cache_creation_input_tokens={} output_tokens={} total_tokens={}",
                reqs,
                input,
                cached,
                fresh,
                cache_created,
                output,
                input + output
            );
        }
        Err(e) => {
            println!("(failed to query token_usage totals: {})", e);
        }
    }

    match sqlx::query(
        r#"
        SELECT
          session_id,
          COUNT(*) AS request_count,
          COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens,
          COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
          MIN(created_at) AS first_at,
          MAX(created_at) AS last_at
        FROM token_usage
        WHERE created_at >= datetime('now', '-' || ? || ' hours')
        GROUP BY session_id
        ORDER BY total_tokens DESC
        LIMIT 15
        "#,
    )
    .bind(token_hours)
    .fetch_all(&pool)
    .await
    {
        Ok(rows) => {
            if rows.is_empty() {
                println!("(no token_usage rows in last 7 hours)");
            } else {
                println!("Top sessions (by tokens):");
                for row in rows {
                    println!(
                        "- session={} tokens={} cached_input_tokens={} requests={} first_at={:?} last_at={:?}",
                        row.get::<String, _>("session_id"),
                        row.get::<i64, _>("total_tokens"),
                        row.get::<i64, _>("cached_input_tokens"),
                        row.get::<i64, _>("request_count"),
                        row.try_get::<Option<String>, _>("first_at").unwrap_or(None),
                        row.try_get::<Option<String>, _>("last_at").unwrap_or(None),
                    );
                }
            }
        }
        Err(e) => {
            println!("(failed to query token_usage by session: {})", e);
        }
    }

    match sqlx::query(
        r#"
        SELECT
          strftime('%Y-%m-%d %H:00', created_at) AS hour,
          COUNT(*) AS request_count,
          COALESCE(SUM(input_tokens + output_tokens), 0) AS total_tokens
        FROM token_usage
        WHERE created_at >= datetime('now', '-' || ? || ' hours')
        GROUP BY hour
        ORDER BY hour ASC
        "#,
    )
    .bind(token_hours)
    .fetch_all(&pool)
    .await
    {
        Ok(rows) => {
            if !rows.is_empty() {
                println!("Hourly:");
                for row in rows {
                    println!(
                        "- {}  tokens={} requests={}",
                        row.get::<String, _>("hour"),
                        row.get::<i64, _>("total_tokens"),
                        row.get::<i64, _>("request_count"),
                    );
                }
            }
        }
        Err(e) => {
            println!("(failed to query token_usage hourly: {})", e);
        }
    }

    println!(
        "\n== Telemetry Reconciliation (Last {} Hours) ==",
        token_hours
    );
    let token_call_ids: Vec<Option<String>> = sqlx::query_scalar(
        r#"
        SELECT call_id FROM token_usage
        WHERE created_at >= datetime('now', '-' || ? || ' hours')
        "#,
    )
    .bind(token_hours)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let llm_rows = sqlx::query(
        r#"
        SELECT
          json_extract(data, '$.call_id') AS call_id,
          json_extract(data, '$.token_usage_present') AS token_usage_present
        FROM events
        WHERE event_type = 'llm_call'
          AND created_at >= datetime('now', '-' || ? || ' hours')
        "#,
    )
    .bind(token_hours)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let mut events_with_usage = 0i64;
    let mut event_rows = Vec::with_capacity(llm_rows.len());
    for row in &llm_rows {
        let call_id = row.try_get::<Option<String>, _>("call_id").unwrap_or(None);
        let token_usage_present = row
            .try_get::<Option<i64>, _>("token_usage_present")
            .unwrap_or(None)
            == Some(1);
        if token_usage_present {
            events_with_usage += 1;
        }
        event_rows.push((call_id, token_usage_present));
    }
    let reconciliation = telemetry_reconciliation_counts(&token_call_ids, &event_rows);
    println!(
        "- token_rows={} llm_events={} llm_events_token_usage_present={} correlated={} token_only={} event_only={} duplicate_token_rows={} duplicate_event_rows={} legacy_token_rows={} legacy_event_rows={}",
        token_call_ids.len(),
        llm_rows.len(),
        events_with_usage,
        reconciliation.correlated,
        reconciliation.token_only,
        reconciliation.event_only,
        reconciliation.duplicate_token_rows,
        reconciliation.duplicate_event_rows,
        reconciliation.legacy_token_rows,
        reconciliation.legacy_event_rows,
    );

    println!("\n== Task Outcomes (Last {} Hours) ==", token_hours);
    let task_end_rows = sqlx::query(
        r#"
        SELECT data
        FROM events
        WHERE event_type = 'task_end'
          AND created_at >= datetime('now', '-' || ? || ' hours')
        "#,
    )
    .bind(token_hours)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let mut by_status: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut by_outcome: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut unknown = 0u64;
    for row in task_end_rows {
        let data_str: String = row.get("data");
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&data_str) else {
            unknown += 1;
            continue;
        };
        // Whitelist of known terminal statuses; anything else is bucketed as
        // "unknown". Not an unwrap_or — arbitrary status strings must NOT pass
        // through.
        #[allow(clippy::manual_unwrap_or)]
        let status = match value.get("status").and_then(|v| v.as_str()) {
            Some(status @ ("completed" | "failed" | "cancelled")) => status,
            _ => "unknown",
        };
        *by_status.entry(status.to_string()).or_insert(0) += 1;
        let outcome = canonical_task_outcome(&value).unwrap_or_else(|| {
            unknown += 1;
            "unknown"
        });
        *by_outcome.entry(outcome.to_string()).or_insert(0) += 1;
    }
    println!("- by_status: {:?}", by_status);
    println!("- by_outcome: {:?}", by_outcome);
    if unknown > 0 {
        println!("- malformed_or_unknown={}", unknown);
    }

    println!("\n== Errors by Type (Last {} Hours) ==", token_hours);
    let error_rows = sqlx::query(
        r#"
        SELECT data
        FROM events
        WHERE event_type = 'error'
          AND created_at >= datetime('now', '-' || ? || ' hours')
        "#,
    )
    .bind(token_hours)
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let mut by_error_type: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for row in error_rows {
        let data_str: String = row.get("data");
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&data_str) else {
            *by_error_type.entry("unknown".to_string()).or_insert(0) += 1;
            continue;
        };
        let error_type = canonical_error_type(&value);
        *by_error_type.entry(error_type.to_string()).or_insert(0) += 1;
    }
    println!("- by_error_type: {:?}", by_error_type);

    if let Some(inv_id) = inv_filter {
        println!("\n== Invocation {} Details ==", inv_id);
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, agent_name, started_at, completed_at, success, exit_code, duration_secs,
                   prompt_summary, output_summary
            FROM cli_agent_invocations
            WHERE id = ?
            "#,
        )
        .bind(inv_id)
        .fetch_all(&pool)
        .await?;
        for row in rows {
            println!(
                "- id={} session={} agent={} success={:?} exit={:?} started={} completed={:?} dur={:?}s\n  prompt={}\n  output={}",
                row.get::<i64, _>("id"),
                row.get::<String, _>("session_id"),
                row.get::<String, _>("agent_name"),
                row.try_get::<Option<i64>, _>("success").unwrap_or(None),
                row.try_get::<Option<i64>, _>("exit_code").unwrap_or(None),
                row.get::<String, _>("started_at"),
                row.try_get::<Option<String>, _>("completed_at").unwrap_or(None),
                row.try_get::<Option<f64>, _>("duration_secs").unwrap_or(None),
                row.get::<String, _>("prompt_summary"),
                row.try_get::<Option<String>, _>("output_summary")
                    .unwrap_or(None)
                    .unwrap_or_default()
                    .replace('\n', " ")
            );
        }
    }

    println!("\n== Recent Task Events ==");
    let events = sqlx::query(
        r#"
        SELECT id, event_type, task_id, tool_name, created_at
        FROM events
        WHERE event_type IN ('task_start', 'tool_call', 'tool_result', 'task_end', 'error')
        ORDER BY id DESC
        LIMIT 30
        "#,
    )
    .fetch_all(&pool)
    .await?;
    for row in events {
        println!(
            "- id={} type={} task_id={:?} tool={:?} at={}",
            row.get::<i64, _>("id"),
            row.get::<String, _>("event_type"),
            row.try_get::<Option<String>, _>("task_id").unwrap_or(None),
            row.try_get::<Option<String>, _>("tool_name")
                .unwrap_or(None),
            row.get::<String, _>("created_at"),
        );
    }

    println!("\n== Recent LLM Calls ==");
    let llm_events = sqlx::query(
        r#"
        SELECT
            task_id,
            created_at,
            json_extract(data, '$.iteration') AS iteration,
            json_extract(data, '$.model') AS model,
            json_extract(data, '$.fell_back') AS fell_back,
            json_extract(data, '$.latency_ms') AS latency_ms,
            json_extract(data, '$.input_tokens') AS input_tokens,
            json_extract(data, '$.output_tokens') AS output_tokens,
            json_extract(data, '$.cached_input_tokens') AS cached_input_tokens,
            json_extract(data, '$.fresh_input_tokens') AS fresh_input_tokens,
            json_extract(data, '$.cache_creation_input_tokens') AS cache_creation_input_tokens
        FROM events
        WHERE event_type = 'llm_call'
        ORDER BY id DESC
        LIMIT 30
        "#,
    )
    .fetch_all(&pool)
    .await?;
    if llm_events.is_empty() {
        println!("(none)");
    } else {
        for row in llm_events {
            println!(
                "- task_id={:?} iter={} model={} fell_back={} latency_ms={} in_tok={} cached_in_tok={:?} fresh_in_tok={:?} cache_create_tok={:?} out_tok={} at={}",
                row.try_get::<Option<String>, _>("task_id").unwrap_or(None),
                row.try_get::<Option<i64>, _>("iteration").unwrap_or(None).unwrap_or(0),
                row.try_get::<Option<String>, _>("model").unwrap_or(None).unwrap_or_default(),
                row.try_get::<Option<i64>, _>("fell_back").unwrap_or(None) == Some(1),
                row.try_get::<Option<i64>, _>("latency_ms").unwrap_or(None).unwrap_or(0),
                row.try_get::<Option<i64>, _>("input_tokens").unwrap_or(None).unwrap_or(0),
                row.try_get::<Option<i64>, _>("cached_input_tokens").unwrap_or(None),
                row.try_get::<Option<i64>, _>("fresh_input_tokens").unwrap_or(None),
                row.try_get::<Option<i64>, _>("cache_creation_input_tokens")
                    .unwrap_or(None),
                row.try_get::<Option<i64>, _>("output_tokens").unwrap_or(None).unwrap_or(0),
                row.get::<String, _>("created_at"),
            );
        }
    }

    println!("\n== Recent cli_agent Tool Events ==");
    let cli_events = sqlx::query(
        r#"
        SELECT id, session_id, event_type, task_id, tool_name, created_at,
               substr(data, 1, 260) AS data_preview
        FROM events
        WHERE tool_name = 'cli_agent'
        ORDER BY id DESC
        LIMIT 40
        "#,
    )
    .fetch_all(&pool)
    .await?;
    if cli_events.is_empty() {
        println!("(none)");
    } else {
        for row in cli_events {
            println!(
                "- id={} session={} type={} task_id={:?} at={}\n  data={}",
                row.get::<i64, _>("id"),
                row.get::<String, _>("session_id"),
                row.get::<String, _>("event_type"),
                row.try_get::<Option<String>, _>("task_id").unwrap_or(None),
                row.get::<String, _>("created_at"),
                row.try_get::<Option<String>, _>("data_preview")
                    .unwrap_or(None)
                    .unwrap_or_default()
                    .replace('\n', " ")
            );
        }
    }

    if let Some(task_id) = task_filter {
        println!("\n== Task {} Full Event Stream ==", task_id);
        let rows = sqlx::query(
            r#"
            SELECT id, event_type, tool_name, created_at, substr(data, 1, 600) AS data_preview
            FROM events
            WHERE task_id = ?
            ORDER BY id ASC
            "#,
        )
        .bind(&task_id)
        .fetch_all(&pool)
        .await?;
        for row in rows {
            println!(
                "- id={} type={} tool={:?} at={}\n  data={}",
                row.get::<i64, _>("id"),
                row.get::<String, _>("event_type"),
                row.try_get::<Option<String>, _>("tool_name")
                    .unwrap_or(None),
                row.get::<String, _>("created_at"),
                row.try_get::<Option<String>, _>("data_preview")
                    .unwrap_or(None)
                    .unwrap_or_default()
                    .replace('\n', " ")
            );
        }

        println!("\n== Task {} LLM Calls ==", task_id);
        let llm_rows = sqlx::query(
            r#"
            SELECT
                json_extract(data, '$.iteration') AS iteration,
                json_extract(data, '$.model') AS model,
                json_extract(data, '$.final_model') AS final_model,
                json_extract(data, '$.fell_back') AS fell_back,
                json_extract(data, '$.attempts') AS attempts,
                json_extract(data, '$.latency_ms') AS latency_ms,
                json_extract(data, '$.build_ms') AS build_ms,
                json_extract(data, '$.input_tokens') AS input_tokens,
                json_extract(data, '$.output_tokens') AS output_tokens,
                json_extract(data, '$.cached_input_tokens') AS cached_input_tokens,
                json_extract(data, '$.fresh_input_tokens') AS fresh_input_tokens,
                json_extract(data, '$.cache_creation_input_tokens') AS cache_creation_input_tokens,
                json_extract(data, '$.est_input_tokens') AS est_input_tokens
            FROM events
            WHERE event_type = 'llm_call' AND task_id = ?
            ORDER BY id ASC
            "#,
        )
        .bind(&task_id)
        .fetch_all(&pool)
        .await?;
        if llm_rows.is_empty() {
            println!("(none)");
        } else {
            for row in llm_rows {
                let model = row.try_get::<Option<String>, _>("model").unwrap_or(None);
                let final_model = row
                    .try_get::<Option<String>, _>("final_model")
                    .unwrap_or(None);
                let fell_back =
                    row.try_get::<Option<i64>, _>("fell_back").unwrap_or(None) == Some(1);
                let model_label = match final_model {
                    Some(fm) if Some(&fm) != model.as_ref() => {
                        format!("{} -> {}", model.unwrap_or_default(), fm)
                    }
                    _ => model.unwrap_or_default(),
                };
                println!(
                    "- iter={} model={} fell_back={} attempts={} latency_ms={} build_ms={} in_tok={} cached_in_tok={:?} fresh_in_tok={:?} cache_create_tok={:?} out_tok={} est_in_tok={}",
                    row.try_get::<Option<i64>, _>("iteration").unwrap_or(None).unwrap_or(0),
                    model_label,
                    fell_back,
                    row.try_get::<Option<i64>, _>("attempts").unwrap_or(None).unwrap_or(0),
                    row.try_get::<Option<i64>, _>("latency_ms").unwrap_or(None).unwrap_or(0),
                    row.try_get::<Option<i64>, _>("build_ms").unwrap_or(None).unwrap_or(0),
                    row.try_get::<Option<i64>, _>("input_tokens").unwrap_or(None).unwrap_or(0),
                    row.try_get::<Option<i64>, _>("cached_input_tokens").unwrap_or(None),
                    row.try_get::<Option<i64>, _>("fresh_input_tokens").unwrap_or(None),
                    row.try_get::<Option<i64>, _>("cache_creation_input_tokens")
                        .unwrap_or(None),
                    row.try_get::<Option<i64>, _>("output_tokens").unwrap_or(None).unwrap_or(0),
                    row.try_get::<Option<i64>, _>("est_input_tokens").unwrap_or(None).unwrap_or(0),
                );
            }
        }
    }

    if let Some(session_id) = session_filter.as_deref() {
        println!("\n== Recent Session {} Events ==", session_id);
        let rows = sqlx::query(
            r#"
            SELECT id, event_type, tool_name, task_id, created_at, substr(data, 1, 420) AS data_preview
            FROM events
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 80
            "#,
        )
        .bind(session_id)
        .fetch_all(&pool)
        .await?;
        if rows.is_empty() {
            println!("(none)");
        } else {
            for row in rows {
                println!(
                    "- id={} type={} tool={:?} task_id={:?} at={}\n  data={}",
                    row.get::<i64, _>("id"),
                    row.get::<String, _>("event_type"),
                    row.try_get::<Option<String>, _>("tool_name")
                        .unwrap_or(None),
                    row.try_get::<Option<String>, _>("task_id").unwrap_or(None),
                    row.get::<String, _>("created_at"),
                    row.try_get::<Option<String>, _>("data_preview")
                        .unwrap_or(None)
                        .unwrap_or_default()
                        .replace('\n', " ")
                );
            }
        }

        println!("\n== Recent Session {} Messages ==", session_id);
        let msgs = sqlx::query(
            r#"
            WITH convo AS (
                SELECT
                    COALESCE(NULLIF(CAST(json_extract(data, '$.message_id') AS TEXT), ''), CAST(id AS TEXT)) AS message_id,
                    session_id,
                    CASE event_type
                        WHEN 'user_message' THEN 'user'
                        WHEN 'assistant_response' THEN 'assistant'
                        WHEN 'tool_result' THEN 'tool'
                        ELSE event_type
                    END AS role,
                    COALESCE(tool_name, CAST(json_extract(data, '$.name') AS TEXT)) AS tool_name,
                    created_at,
                    CASE event_type
                        WHEN 'user_message' THEN CAST(json_extract(data, '$.content') AS TEXT)
                        WHEN 'assistant_response' THEN CAST(json_extract(data, '$.content') AS TEXT)
                        WHEN 'tool_result' THEN CAST(json_extract(data, '$.result') AS TEXT)
                        ELSE NULL
                    END AS content
                FROM events
                WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
            )
            SELECT message_id, role, tool_name, created_at,
                   substr(COALESCE(content, ''), 1, 280) AS content_preview
            FROM convo
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 80
            "#,
        )
        .bind(session_id)
        .fetch_all(&pool)
        .await?;
        if msgs.is_empty() {
            println!("(none)");
        } else {
            for row in msgs {
                println!(
                    "- {} {} tool={:?} at={}\n  {}",
                    row.get::<String, _>("message_id"),
                    row.get::<String, _>("role"),
                    row.try_get::<Option<String>, _>("tool_name")
                        .unwrap_or(None),
                    row.get::<String, _>("created_at"),
                    row.get::<String, _>("content_preview").replace('\n', " ")
                );
            }
        }
    }

    println!("\n== Recent Messages ==");
    let messages = sqlx::query(
        r#"
        WITH convo AS (
            SELECT
                COALESCE(NULLIF(CAST(json_extract(data, '$.message_id') AS TEXT), ''), CAST(id AS TEXT)) AS message_id,
                CASE event_type
                    WHEN 'user_message' THEN 'user'
                    WHEN 'assistant_response' THEN 'assistant'
                    WHEN 'tool_result' THEN 'tool'
                    ELSE event_type
                END AS role,
                COALESCE(tool_name, CAST(json_extract(data, '$.name') AS TEXT)) AS tool_name,
                created_at,
                CASE event_type
                    WHEN 'user_message' THEN CAST(json_extract(data, '$.content') AS TEXT)
                    WHEN 'assistant_response' THEN CAST(json_extract(data, '$.content') AS TEXT)
                    WHEN 'tool_result' THEN CAST(json_extract(data, '$.result') AS TEXT)
                    ELSE NULL
                END AS content
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
        )
        SELECT message_id, role, tool_name, substr(COALESCE(content, ''), 1, 180) AS content, created_at
        FROM convo
        ORDER BY created_at DESC
        LIMIT 20
        "#,
    )
    .fetch_all(&pool)
    .await?;
    for row in messages {
        println!(
            "- {} {} tool={:?} at={}\n  {}",
            row.get::<String, _>("message_id"),
            row.get::<String, _>("role"),
            row.try_get::<Option<String>, _>("tool_name")
                .unwrap_or(None),
            row.get::<String, _>("created_at"),
            row.try_get::<Option<String>, _>("content")
                .unwrap_or(None)
                .unwrap_or_default()
                .replace('\n', " ")
        );
    }

    println!("\n== Dynamic CLI Agent Config ==");
    let dyn_agents = sqlx::query(
        r#"
        SELECT id, name, command, args_json, enabled, created_at
        FROM dynamic_cli_agents
        ORDER BY id ASC
        "#,
    )
    .fetch_all(&pool)
    .await?;
    if dyn_agents.is_empty() {
        println!("(none)");
    } else {
        for row in dyn_agents {
            println!(
                "- id={} name={} command={} enabled={} created_at={}\n  args_json={}",
                row.get::<i64, _>("id"),
                row.get::<String, _>("name"),
                row.get::<String, _>("command"),
                row.get::<i64, _>("enabled"),
                row.get::<String, _>("created_at"),
                row.get::<String, _>("args_json")
            );
        }
    }

    // === Goal Scheduling Diagnostics ===
    println!("\n== Active Scheduled Goals ==");
    let sched_goals = sqlx::query(
        r#"SELECT g.id, substr(g.description, 1, 100) AS desc, g.goal_type, g.session_id,
                  s.id AS sched_id, s.cron_expr, s.fire_policy, s.is_paused, s.is_one_shot,
                  s.last_run_at, s.next_run_at, s.original_schedule
           FROM goals g
           JOIN goal_schedules s ON s.goal_id = g.id
           WHERE g.status = 'active' AND g.domain = 'orchestration'
           ORDER BY s.next_run_at ASC"#,
    )
    .fetch_all(&pool)
    .await?;
    if sched_goals.is_empty() {
        println!("(none)");
    } else {
        for r in &sched_goals {
            println!(
                "- goal={} sched={} cron='{}' policy={} paused={} one_shot={}\n  next={} last={}\n  desc={}",
                &r.get::<String, _>("id")[..8],
                &r.get::<String, _>("sched_id")[..8],
                r.get::<String, _>("cron_expr"),
                r.get::<String, _>("fire_policy"),
                r.get::<i64, _>("is_paused"),
                r.get::<i64, _>("is_one_shot"),
                r.get::<String, _>("next_run_at"),
                r.try_get::<Option<String>, _>("last_run_at").unwrap_or(None).unwrap_or_else(|| "never".to_string()),
                r.get::<String, _>("desc"),
            );
        }
    }

    println!("\n== Recent Scheduled Tasks (last 10) ==");
    let sched_tasks = sqlx::query(
        r#"SELECT t.id, t.goal_id, substr(t.description, 1, 80) AS desc, t.status,
                  t.created_at, t.started_at, t.completed_at, t.retry_count, t.agent_id
           FROM tasks t
           WHERE t.description LIKE 'Scheduled check:%' OR t.description LIKE 'Execute scheduled goal:%'
           ORDER BY t.created_at DESC LIMIT 10"#,
    )
    .fetch_all(&pool)
    .await?;
    if sched_tasks.is_empty() {
        println!("(none)");
    } else {
        for r in &sched_tasks {
            println!(
                "- task={} goal={} status={} retry={} agent={}\n  created={} started={} completed={}\n  desc={}",
                &r.get::<String, _>("id")[..8],
                &r.get::<String, _>("goal_id")[..8],
                r.get::<String, _>("status"),
                r.get::<i64, _>("retry_count"),
                r.try_get::<Option<String>, _>("agent_id").unwrap_or(None).unwrap_or_else(|| "none".to_string()),
                &r.get::<String, _>("created_at")[..19],
                r.try_get::<Option<String>, _>("started_at").unwrap_or(None).map(|s| s[..19].to_string()).unwrap_or_else(|| "none".to_string()),
                r.try_get::<Option<String>, _>("completed_at").unwrap_or(None).map(|s| s[..19].to_string()).unwrap_or_else(|| "none".to_string()),
                r.get::<String, _>("desc"),
            );
        }
    }

    pool.close().await;
    Ok(())
}
