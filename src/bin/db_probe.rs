use anyhow::Context;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};
use std::str::FromStr;

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

    let _ = dotenvy::dotenv();

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
          COALESCE(SUM(output_tokens), 0) AS output_tokens
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
            println!(
                "- requests={} input_tokens={} output_tokens={} total_tokens={}",
                reqs,
                input,
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
                        "- session={} tokens={} requests={} first_at={:?} last_at={:?}",
                        row.get::<String, _>("session_id"),
                        row.get::<i64, _>("total_tokens"),
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
        .bind(task_id)
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

    pool.close().await;
    Ok(())
}
