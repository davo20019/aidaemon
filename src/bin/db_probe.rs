#![allow(clippy::items_after_test_module)]

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

/// Lower time bound for `events.created_at` comparisons, formatted the way
/// the event store writes timestamps (RFC3339, `+00:00` offset).
///
/// `events.created_at` must NOT be compared against `datetime('now', ...)`:
/// SQLite compares TEXT timestamps as strings, and the space-separated
/// `datetime()` format sorts below the `T` separator, which silently degrades
/// the filter to calendar-day granularity.
fn events_cutoff_rfc3339(now: chrono::DateTime<chrono::Utc>, hours: i64) -> String {
    (now - chrono::Duration::hours(hours)).to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
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

#[cfg(test)]
fn telemetry_reconciliation_counts(
    token_call_ids: &[Option<String>],
    event_rows: &[(Option<String>, bool)],
) -> TelemetryReconciliationCounts {
    telemetry_reconciliation_counts_windowed(token_call_ids, event_rows, token_call_ids, event_rows)
}

/// Reconcile a requested diagnostic window against the complete durable
/// counterpart index. Window membership selects which rows are reported; it
/// must never decide whether a call is matched. Otherwise widening the window
/// can relabel the same call from `token_only` to correlated merely because its
/// counterpart timestamp fell just outside the narrower cutoff.
fn telemetry_reconciliation_counts_windowed(
    window_token_call_ids: &[Option<String>],
    window_event_rows: &[(Option<String>, bool)],
    all_token_call_ids: &[Option<String>],
    all_event_rows: &[(Option<String>, bool)],
) -> TelemetryReconciliationCounts {
    let mut token_frequencies = std::collections::HashMap::<String, usize>::new();
    let mut legacy_token_rows = 0usize;
    for call_id in window_token_call_ids {
        match call_id.as_deref().filter(|call_id| !call_id.is_empty()) {
            Some(call_id) => *token_frequencies.entry(call_id.to_string()).or_insert(0) += 1,
            None => legacy_token_rows += 1,
        }
    }

    let mut event_frequencies = std::collections::HashMap::<String, usize>::new();
    let mut legacy_event_rows = 0usize;
    for (call_id, usage_present) in window_event_rows {
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
    let all_token_ids = all_token_call_ids
        .iter()
        .filter_map(|call_id| call_id.as_deref())
        .filter(|call_id| !call_id.is_empty())
        .collect::<std::collections::HashSet<_>>();
    let all_usage_event_ids = all_event_rows
        .iter()
        .filter(|(_, usage_present)| *usage_present)
        .filter_map(|(call_id, _)| call_id.as_deref())
        .filter(|call_id| !call_id.is_empty())
        .collect::<std::collections::HashSet<_>>();
    TelemetryReconciliationCounts {
        correlated: token_ids
            .iter()
            .filter(|call_id| all_usage_event_ids.contains(call_id.as_str()))
            .count(),
        token_only: token_ids
            .iter()
            .filter(|call_id| !all_usage_event_ids.contains(call_id.as_str()))
            .count(),
        event_only: event_ids
            .iter()
            .filter(|call_id| !all_token_ids.contains(call_id.as_str()))
            .count(),
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

/// Diagnostic split of `token_only` reconciliation rows: token_usage rows
/// with a call_id that no usage-bearing `llm_call` event accounts for.
/// Mirrors the `token_only` definition in `telemetry_reconciliation_counts`
/// (unique call_ids vs events with `token_usage_present=true`).
#[derive(Debug, Default, PartialEq, Eq)]
struct TokenOnlyBreakdown {
    /// (session_id, token-only call count) sorted by count descending.
    by_session: Vec<(String, usize)>,
    /// call_id has no llm_call event at all — the call never reached the
    /// event store (e.g. background LLM use outside the agent loop).
    event_missing: usize,
    /// llm_call event exists but reported token_usage_present=false.
    event_usage_flag_false: usize,
}

fn token_only_breakdown(
    token_rows: &[(Option<String>, String)],
    event_rows: &[(Option<String>, bool)],
) -> TokenOnlyBreakdown {
    let mut usage_event_ids = std::collections::HashSet::<&str>::new();
    let mut any_event_ids = std::collections::HashSet::<&str>::new();
    for (call_id, usage_present) in event_rows {
        if let Some(call_id) = call_id.as_deref().filter(|call_id| !call_id.is_empty()) {
            any_event_ids.insert(call_id);
            if *usage_present {
                usage_event_ids.insert(call_id);
            }
        }
    }

    let mut seen = std::collections::HashSet::<&str>::new();
    let mut by_session = std::collections::HashMap::<&str, usize>::new();
    let mut event_missing = 0usize;
    let mut event_usage_flag_false = 0usize;
    for (call_id, session_id) in token_rows {
        let Some(call_id) = call_id.as_deref().filter(|call_id| !call_id.is_empty()) else {
            continue;
        };
        if usage_event_ids.contains(call_id) || !seen.insert(call_id) {
            continue;
        }
        *by_session.entry(session_id.as_str()).or_insert(0) += 1;
        if any_event_ids.contains(call_id) {
            event_usage_flag_false += 1;
        } else {
            event_missing += 1;
        }
    }

    let mut by_session: Vec<(String, usize)> = by_session
        .into_iter()
        .map(|(session, count)| (session.to_string(), count))
        .collect();
    by_session.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    TokenOnlyBreakdown {
        by_session,
        event_missing,
        event_usage_flag_false,
    }
}

fn handholding_detail_label(reason: Option<&str>, error: Option<&str>) -> String {
    if let Some(reason) = reason.filter(|value| !value.is_empty()) {
        return format!("reason={reason}");
    }
    if let Some(error) = error.filter(|value| !value.is_empty()) {
        return format!("error={error}");
    }
    "detail=-".to_string()
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
    fn token_only_breakdown_groups_by_session_and_splits_event_presence() {
        let token_rows: Vec<(Option<String>, String)> = vec![
            // correlated — not token-only
            (Some("call-a".to_string()), "sess-1".to_string()),
            // event exists but token_usage_present=false
            (Some("call-b".to_string()), "sess-1".to_string()),
            // no llm_call event at all
            (Some("call-c".to_string()), "sess-2".to_string()),
            (Some("call-d".to_string()), "sess-2".to_string()),
            // legacy row (no call_id) — not token-only
            (None, "sess-3".to_string()),
        ];
        let event_rows: Vec<(Option<String>, bool)> = vec![
            (Some("call-a".to_string()), true),
            (Some("call-b".to_string()), false),
        ];

        let breakdown = token_only_breakdown(&token_rows, &event_rows);
        assert_eq!(
            breakdown.by_session,
            vec![("sess-2".to_string(), 2), ("sess-1".to_string(), 1)]
        );
        assert_eq!(breakdown.event_missing, 2);
        assert_eq!(breakdown.event_usage_flag_false, 1);
    }

    #[test]
    fn events_cutoff_string_compares_correctly_against_rfc3339_timestamps() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-06-11T15:40:00+00:00")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let cutoff = events_cutoff_rfc3339(now, 7);

        // Same-day event older than the window must sort BELOW the cutoff.
        assert!("2026-06-11T07:00:00.974627+00:00" < cutoff.as_str());
        // Events inside the window (with and without fractional seconds)
        // must sort at-or-above it.
        assert!("2026-06-11T09:00:00+00:00" >= cutoff.as_str());
        assert!("2026-06-11T08:40:00.000001+00:00" >= cutoff.as_str());
        // Prior-day events stay excluded.
        assert!("2026-06-10T23:59:59+00:00" < cutoff.as_str());

        // The legacy space-format bound this replaces wrongly included
        // same-day events older than the window ('T' > ' ').
        assert!("2026-06-11T07:00:00.974627+00:00" >= "2026-06-11 08:40:00");
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

    #[test]
    fn reconciliation_category_does_not_depend_on_diagnostic_window() {
        let narrow_tokens = vec![Some("call-boundary".to_string())];
        let narrow_events = Vec::new();
        let all_tokens = narrow_tokens.clone();
        let all_events = vec![(Some("call-boundary".to_string()), true)];

        let narrow = telemetry_reconciliation_counts_windowed(
            &narrow_tokens,
            &narrow_events,
            &all_tokens,
            &all_events,
        );
        let wide = telemetry_reconciliation_counts_windowed(
            &narrow_tokens,
            &all_events,
            &all_tokens,
            &all_events,
        );

        assert_eq!(narrow.token_only, 0);
        assert_eq!(wide.token_only, 0);
        assert_eq!(narrow.correlated, wide.correlated);
    }

    #[test]
    fn handholding_detail_prefers_reason_then_error() {
        assert_eq!(
            handholding_detail_label(Some("control_command"), Some("ignored")),
            "reason=control_command"
        );
        assert_eq!(
            handholding_detail_label(None, Some("planner_returned_none")),
            "error=planner_returned_none"
        );
        assert_eq!(handholding_detail_label(None, None), "detail=-");
    }

    #[test]
    fn completion_claim_requires_exact_response_receipt_and_obligation_edges() {
        let receipt = aidaemon::CompletionProofReference {
            receipt_id: "tool-call-synthetic".to_string(),
            result_id: Some("sha256:synthetic".to_string()),
            obligation_ids: vec!["task:task-synthetic/evidence:0".to_string()],
        };
        let task = FabricationTask {
            session_id: "telegram:synthetic-user-1".to_string(),
            last_ts: "2026-08-15T00:00:00Z".to_string(),
            last_reply: "I published the synthetic result.".to_string(),
            response_id: "response-synthetic".to_string(),
            referenced_receipts: vec![receipt.clone()],
            tool_calls: 1,
        };
        let mut proof = aidaemon::TaskCompletionProofData {
            schema_version: 1,
            task_id: "task-synthetic".to_string(),
            request_turn_id: Some("turn-synthetic".to_string()),
            response_message_ids: vec![task.response_id.clone()],
            receipt_refs: vec![receipt],
            closed_at: "2026-08-15T00:00:01Z".to_string(),
            proof_basis: None,
        };

        assert!(completion_claim_has_closed_proof(
            "task-synthetic",
            &task,
            Some(&proof)
        ));
        proof.response_message_ids = vec!["different-response".to_string()];
        assert!(!completion_claim_has_closed_proof(
            "task-synthetic",
            &task,
            Some(&proof)
        ));
        proof.response_message_ids = vec![task.response_id.clone()];
        proof.receipt_refs[0].obligation_ids.clear();
        assert!(!completion_claim_has_closed_proof(
            "task-synthetic",
            &task,
            Some(&proof)
        ));
    }
}

async fn print_eval_task(pool: &SqlitePool, task_id: &str) -> anyhow::Result<()> {
    use aidaemon::harness_eval::report::{format_eval_task_report, EvalTaskRow};
    use aidaemon::TaskEndData;

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
    use aidaemon::harness_eval::report::{aggregate_summary, format_eval_summary_row, EvalTaskRow};
    use aidaemon::TaskEndData;

    let rows = sqlx::query(
        r#"
        SELECT session_id, task_id, created_at, data
        FROM events
        WHERE event_type = 'task_end'
          AND created_at >= ?
          AND json_extract(data, '$.harness_eval') IS NOT NULL
        ORDER BY created_at DESC
        "#,
    )
    .bind(events_cutoff_rfc3339(chrono::Utc::now(), hours))
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

async fn print_handholding_summary(pool: &SqlitePool, hours: i64) -> anyhow::Result<()> {
    let cutoff = events_cutoff_rfc3339(chrono::Utc::now(), hours);
    println!(
        "== Hand-Holding Telemetry Summary (Last {} Hours) ==",
        hours
    );

    let rows = sqlx::query(
        r#"
        SELECT
          json_extract(data, '$.metadata.component') AS component,
          json_extract(data, '$.metadata.action') AS action,
          json_extract(data, '$.metadata.reason') AS reason,
          json_extract(data, '$.metadata.error') AS error,
          json_extract(data, '$.metadata.model') AS model,
          json_extract(data, '$.metadata.trust_tier') AS trust_tier,
          COUNT(*) AS fires
        FROM events
        WHERE event_type = 'decision_point'
          AND json_extract(data, '$.decision_type') = 'hand_holding_telemetry'
          AND created_at >= ?
        GROUP BY component, action, reason, error, model, trust_tier
        ORDER BY fires DESC, component, action
        "#,
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        println!("- no hand_holding_telemetry decision points found");
    } else {
        println!("Hand-holding events:");
        for row in rows {
            println!(
                "- component={} action={} {} model={} tier={} fires={}",
                row.try_get::<Option<String>, _>("component")?
                    .unwrap_or_else(|| "-".to_string()),
                row.try_get::<Option<String>, _>("action")?
                    .unwrap_or_else(|| "-".to_string()),
                handholding_detail_label(
                    row.try_get::<Option<String>, _>("reason")?.as_deref(),
                    row.try_get::<Option<String>, _>("error")?.as_deref(),
                ),
                row.try_get::<Option<String>, _>("model")?
                    .unwrap_or_else(|| "-".to_string()),
                row.try_get::<Option<String>, _>("trust_tier")?
                    .unwrap_or_else(|| "-".to_string()),
                row.get::<i64, _>("fires"),
            );
        }
    }

    let outcome_rows = sqlx::query(
        r#"
        SELECT
          json_extract(dp.data, '$.metadata.component') AS component,
          json_extract(dp.data, '$.metadata.action') AS action,
          json_extract(dp.data, '$.metadata.reason') AS reason,
          json_extract(dp.data, '$.metadata.error') AS error,
          COALESCE(
            json_extract(te.data, '$.outcome'),
            CASE json_extract(te.data, '$.status')
              WHEN 'completed' THEN 'succeeded'
              ELSE 'failed'
            END
          ) AS outcome,
          COUNT(DISTINCT dp.task_id) AS tasks,
          COUNT(*) AS fires,
          ROUND(AVG(json_extract(te.data, '$.duration_secs')), 1) AS avg_secs,
          ROUND(AVG(json_extract(te.data, '$.efficiency.llm_calls')), 1) AS avg_llm_calls,
          ROUND(AVG(
            json_extract(te.data, '$.efficiency.input_tokens')
            + json_extract(te.data, '$.efficiency.output_tokens')
          ), 0) AS avg_tokens
        FROM events dp
        LEFT JOIN events te ON te.task_id = dp.task_id AND te.event_type = 'task_end'
        WHERE dp.event_type = 'decision_point'
          AND json_extract(dp.data, '$.decision_type') = 'hand_holding_telemetry'
          AND dp.created_at >= ?
        GROUP BY component, action, reason, error, outcome
        ORDER BY component, action, reason, error, outcome
        "#,
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await?;

    println!("\nJoined to task outcomes:");
    if outcome_rows.is_empty() {
        println!("- no hand-holding events to join");
    } else {
        for row in outcome_rows {
            println!(
                "- component={} action={} {} outcome={} tasks={} fires={} avg_secs={} avg_llm_calls={} avg_tokens={}",
                row.try_get::<Option<String>, _>("component")?
                    .unwrap_or_else(|| "-".to_string()),
                row.try_get::<Option<String>, _>("action")?
                    .unwrap_or_else(|| "-".to_string()),
                handholding_detail_label(
                    row.try_get::<Option<String>, _>("reason")?.as_deref(),
                    row.try_get::<Option<String>, _>("error")?.as_deref(),
                ),
                row.try_get::<Option<String>, _>("outcome")?
                    .unwrap_or_else(|| "-".to_string()),
                row.get::<i64, _>("tasks"),
                row.get::<i64, _>("fires"),
                row.try_get::<Option<f64>, _>("avg_secs")?.unwrap_or(0.0),
                row.try_get::<Option<f64>, _>("avg_llm_calls")?
                    .unwrap_or(0.0),
                row.try_get::<Option<f64>, _>("avg_tokens")?.unwrap_or(0.0),
            );
        }
    }

    let gate_rows = sqlx::query(
        r#"
        SELECT
          json_extract(data, '$.metadata.heuristic') AS heuristic,
          json_extract(data, '$.metadata.action') AS action,
          json_extract(data, '$.metadata.tier') AS tier,
          json_extract(data, '$.metadata.model') AS model,
          COUNT(*) AS fires
        FROM events
        WHERE event_type = 'decision_point'
          AND json_extract(data, '$.decision_type') = 'gate_telemetry'
          AND json_extract(data, '$.metadata.code') = 'supervision_gate_fire'
          AND created_at >= ?
        GROUP BY heuristic, action, tier, model
        ORDER BY fires DESC
        "#,
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await?;

    println!("\nSupervision gates:");
    if gate_rows.is_empty() {
        println!("- no supervision_gate_fire events found");
    } else {
        for row in gate_rows {
            println!(
                "- heuristic={} action={} tier={} model={} fires={}",
                row.try_get::<Option<String>, _>("heuristic")?
                    .unwrap_or_else(|| "-".to_string()),
                row.try_get::<Option<String>, _>("action")?
                    .unwrap_or_else(|| "-".to_string()),
                row.try_get::<Option<String>, _>("tier")?
                    .unwrap_or_else(|| "-".to_string()),
                row.try_get::<Option<String>, _>("model")?
                    .unwrap_or_else(|| "-".to_string()),
                row.get::<i64, _>("fires"),
            );
        }
    }

    Ok(())
}

/// Per-task aggregate used by the fabrication audit.
struct FabricationTask {
    session_id: String,
    last_ts: String,
    last_reply: String,
    response_id: String,
    referenced_receipts: Vec<aidaemon::CompletionProofReference>,
    tool_calls: i64,
}

fn completion_claim_has_closed_proof(
    task_id: &str,
    task: &FabricationTask,
    proof: Option<&aidaemon::TaskCompletionProofData>,
) -> bool {
    let Some(proof) = proof else {
        return false;
    };
    if proof.task_id != task_id
        || task.response_id.is_empty()
        || task.referenced_receipts.is_empty()
        || !proof.response_message_ids.contains(&task.response_id)
    {
        return false;
    }
    task.referenced_receipts.iter().all(|response_ref| {
        !response_ref.obligation_ids.is_empty()
            && proof.receipt_refs.iter().any(|closed_ref| {
                closed_ref.receipt_id == response_ref.receipt_id
                    && closed_ref.result_id == response_ref.result_id
                    && response_ref
                        .obligation_ids
                        .iter()
                        .all(|id| closed_ref.obligation_ids.contains(id))
            })
    })
}

/// Post-hoc proof audit: for each task in the window, fold all final response
/// and task-end records. Any task whose typed contract or actual receipt
/// accounting is material must carry a closed response-to-receipt proof edge.
/// No response wording classifier participates.
async fn print_fabrication_audit(pool: &SqlitePool, hours: i64) -> anyhow::Result<()> {
    let cutoff = events_cutoff_rfc3339(chrono::Utc::now(), hours);
    println!("== Fabrication Audit (Last {} Hours) ==", hours);
    println!(
        "(material outcomes must reference receipts closed by the exact task-end proof graph)\n"
    );

    let rows = sqlx::query(
        r#"
        SELECT
          created_at,
          session_id,
          json_extract(data, '$.task_id') AS task_id,
          COALESCE(json_extract(data, '$.message_id'), '') AS response_id,
          COALESCE(json_extract(data, '$.content'), '') AS content,
          COALESCE(json_extract(data, '$.referenced_receipts'), '[]') AS referenced_receipts,
          COALESCE(json_array_length(json_extract(data, '$.tool_calls')), 0) AS tool_calls
        FROM events
        WHERE event_type = 'assistant_response'
          AND created_at >= ?
          AND json_extract(data, '$.task_id') IS NOT NULL
        ORDER BY created_at ASC
        "#,
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await?;

    let mut tasks: std::collections::HashMap<String, FabricationTask> =
        std::collections::HashMap::new();
    for row in &rows {
        let task_id: String = row.get("task_id");
        let content: String = row.get("content");
        let tool_calls: i64 = row.get("tool_calls");
        let response_id: String = row.get("response_id");
        let referenced_receipts = row
            .try_get::<String, _>("referenced_receipts")
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();
        let created_at: String = row.get("created_at");
        let session_id: String = row.get("session_id");
        let entry = tasks.entry(task_id).or_insert_with(|| FabricationTask {
            session_id,
            last_ts: String::new(),
            last_reply: String::new(),
            response_id: String::new(),
            referenced_receipts: Vec::new(),
            tool_calls: 0,
        });
        entry.tool_calls += tool_calls;
        // The user-facing completion is the latest non-empty assistant content.
        if !content.trim().is_empty() {
            entry.last_ts = created_at;
            entry.last_reply = content;
            entry.response_id = response_id;
            entry.referenced_receipts = referenced_receipts;
        }
    }

    let proof_rows = sqlx::query(
        r#"
        SELECT
          json_extract(data, '$.task_id') AS task_id,
          json_extract(data, '$.completion_proof') AS completion_proof,
          COALESCE(json_extract(data, '$.harness_eval.quality.contract.expects_mutation'), 0) AS expects_mutation,
          COALESCE(json_extract(data, '$.harness_eval.quality.contract.requires_observation'), 0) AS requires_observation,
          COALESCE(json_extract(data, '$.harness_eval.quality.contract.mutation_count'), 0) AS mutation_count,
          COALESCE(json_extract(data, '$.harness_eval.quality.contract.observation_count'), 0) AS observation_count,
          COALESCE(json_extract(data, '$.outcome'),
                   CASE json_extract(data, '$.status') WHEN 'completed' THEN 'succeeded' ELSE 'failed' END) AS outcome
        FROM events
        WHERE event_type = 'task_end'
          AND created_at >= ?
          AND json_extract(data, '$.task_id') IS NOT NULL
        ORDER BY created_at ASC
        "#,
    )
    .bind(&cutoff)
    .fetch_all(pool)
    .await?;
    let mut proofs = std::collections::HashMap::new();
    let mut material_tasks = std::collections::HashSet::new();
    for row in proof_rows {
        let task_id: String = row.get("task_id");
        let succeeded = row.get::<String, _>("outcome") == "succeeded";
        let material = succeeded
            && (row.get::<i64, _>("expects_mutation") != 0
                || row.get::<i64, _>("requires_observation") != 0
                || row.get::<i64, _>("mutation_count") > 0
                || row.get::<i64, _>("observation_count") > 0);
        if material {
            material_tasks.insert(task_id.clone());
        }
        if let Some(raw) = row.try_get::<Option<String>, _>("completion_proof")? {
            if let Ok(proof) = serde_json::from_str::<aidaemon::TaskCompletionProofData>(&raw) {
                proofs.insert(task_id, proof);
            }
        }
    }

    let total = tasks.len();
    let mut claim_tasks = 0usize;
    let mut proven_claims = 0usize;
    let mut candidates: Vec<(String, &FabricationTask, &'static str)> = Vec::new();
    for (id, task) in &tasks {
        if material_tasks.contains(id) {
            claim_tasks += 1;
            let proof = proofs.get(id);
            if completion_claim_has_closed_proof(id, task, proof) {
                proven_claims += 1;
            } else {
                let reason = if task.referenced_receipts.is_empty() {
                    if task.tool_calls == 0 {
                        "legacy_zero_tool_calls"
                    } else {
                        "missing_response_receipt_refs"
                    }
                } else if proof.is_none() {
                    "missing_task_end_proof"
                } else {
                    "response_task_proof_mismatch"
                };
                candidates.push((id.clone(), task, reason));
            }
        }
    }

    println!("- tasks with a final reply:            {}", total);
    println!("- ...with a material typed outcome:    {}", claim_tasks);
    println!("- ...with closed structural proof:     {}", proven_claims);
    println!(
        "- ...without closed proof (candidates): {}",
        candidates.len()
    );
    if claim_tasks > 0 {
        println!(
            "- unproven-claim rate:                  {:.0}%",
            candidates.len() as f64 / claim_tasks as f64 * 100.0
        );
    }

    candidates.sort_by(|a, b| b.1.last_ts.cmp(&a.1.last_ts));
    println!("\nCandidates (newest first):");
    if candidates.is_empty() {
        println!("- none — every successful material task has an exact closed proof graph");
    }
    for (id, task, reason) in &candidates {
        let snippet: String = task
            .last_reply
            .chars()
            .take(160)
            .collect::<String>()
            .replace('\n', " ");
        println!(
            "- [{}] reason={} task={} session={} tool_calls={} refs={}\n    \"{}\"",
            task.last_ts,
            reason,
            &id[..id.len().min(8)],
            task.session_id,
            task.tool_calls,
            task.referenced_receipts.len(),
            snippet,
        );
    }

    Ok(())
}

async fn record_fixture_from_session(
    pool: &SqlitePool,
    session_id: &str,
    task_id: Option<&str>,
    output: Option<&str>,
    include_text: bool,
) -> anyhow::Result<()> {
    use aidaemon::harness_eval::fixture::{build_recorded_fixture, fixtures_dir};
    use aidaemon::{EventType, TaskEndData, ToolCallData, UserMessageData};

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
    let retry_stalled_goal = args
        .windows(2)
        .find(|w| w[0] == "--retry-stalled-goal")
        .map(|w| w[1].clone());
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
    let handholding_summary = args.iter().any(|arg| arg == "--handholding-summary");
    let fabrication_audit = args.iter().any(|arg| arg == "--fabrication-audit");
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
    let prompt_hash = args
        .windows(2)
        .find(|w| w[0] == "--prompt")
        .map(|w| w[1].clone());

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

    if let Some(goal_id) = retry_stalled_goal.as_deref() {
        let result = sqlx::query(
            r#"
            UPDATE goals
               SET status = 'active',
                   dispatch_failures = 0,
                   completed_at = NULL,
                   updated_at = datetime('now')
             WHERE id = ?
               AND status = 'stalled'
               AND EXISTS (
                    SELECT 1
                      FROM goal_schedules
                     WHERE goal_schedules.goal_id = goals.id
               )
            "#,
        )
        .bind(goal_id)
        .execute(&pool)
        .await?;
        anyhow::ensure!(
            result.rows_affected() == 1,
            "goal was not a stalled scheduled goal or did not exist"
        );
        println!("Reactivated stalled scheduled goal {goal_id}; dispatch_failures reset to 0.");
        return Ok(());
    }

    if args.iter().any(|a| a == "--dynamic-bots") {
        let rows = sqlx::query(
            "SELECT id, channel_type, allowed_user_ids, extra_config, created_at, bot_token, app_token \
             FROM dynamic_bots ORDER BY created_at ASC",
        )
        .fetch_all(&pool)
        .await?;
        println!("== Dynamic Bots ({}) — tokens REDACTED ==", rows.len());
        let redact = |t: &str| -> String {
            if t.starts_with("keychain:") {
                "keychain-ref".to_string()
            } else {
                format!("<{}-char literal token>", t.len())
            }
        };
        for row in rows {
            let id: i64 = row.get("id");
            let channel: String = row.get("channel_type");
            let allowed: String = row.get("allowed_user_ids");
            let extra: Option<String> = row.get("extra_config");
            let created: String = row.get("created_at");
            let bot_token: String = row.get("bot_token");
            let app_token: Option<String> = row.get("app_token");
            println!("- id={} channel={} created={}", id, channel, created);
            println!("    allowed_user_ids = {}", allowed);
            println!("    extra_config     = {}", extra.as_deref().unwrap_or("-"));
            println!(
                "    bot_token={}  app_token={}",
                redact(&bot_token),
                app_token
                    .as_deref()
                    .map(redact)
                    .unwrap_or_else(|| "-".into())
            );
        }
        return Ok(());
    }

    if let Some(hash) = prompt_hash.as_deref() {
        // Prefix match for convenience: `--prompt 4848ae26` finds the full hash.
        let rows = sqlx::query(
            "SELECT hash, content, created_at FROM prompt_snapshots WHERE hash LIKE ? || '%' ORDER BY created_at DESC LIMIT 5",
        )
        .bind(hash)
        .fetch_all(&pool)
        .await?;
        if rows.is_empty() {
            println!("No prompt snapshot matching hash prefix {:?}", hash);
        }
        for row in rows {
            let full_hash: String = row.get("hash");
            let created_at: String = row.get("created_at");
            let content: String = row.get("content");
            println!("== Prompt Snapshot {} (saved {}) ==", full_hash, created_at);
            println!("{}", content);
        }
        return Ok(());
    }
    if let Some(task_id) = eval_task.as_deref() {
        print_eval_task(&pool, task_id).await?;
        return Ok(());
    }
    if eval_summary {
        print_eval_summary(&pool, eval_hours, !eval_include_subagents).await?;
        return Ok(());
    }
    if handholding_summary {
        print_handholding_summary(&pool, eval_hours).await?;
        return Ok(());
    }
    if fabrication_audit {
        print_fabrication_audit(&pool, eval_hours).await?;
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
    let token_cutoff = events_cutoff_rfc3339(chrono::Utc::now(), token_hours);
    match sqlx::query(
        r#"
        SELECT
          COUNT(*) AS request_count,
          COALESCE(SUM(input_tokens), 0) AS input_tokens,
          COALESCE(SUM(output_tokens), 0) AS output_tokens,
          COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
          COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens
        FROM token_usage
        WHERE created_at >= ?
        "#,
    )
    .bind(&token_cutoff)
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
        WHERE created_at >= ?
        GROUP BY session_id
        ORDER BY total_tokens DESC
        LIMIT 15
        "#,
    )
    .bind(&token_cutoff)
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
        WHERE created_at >= ?
        GROUP BY hour
        ORDER BY hour ASC
        "#,
    )
    .bind(&token_cutoff)
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
    let token_rows: Vec<(Option<String>, String)> = sqlx::query(
        r#"
        SELECT call_id, session_id FROM token_usage
        WHERE created_at >= ?
        "#,
    )
    .bind(&token_cutoff)
    .fetch_all(&pool)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|row| {
        (
            row.try_get::<Option<String>, _>("call_id").unwrap_or(None),
            row.try_get::<String, _>("session_id").unwrap_or_default(),
        )
    })
    .collect();
    let token_call_ids: Vec<Option<String>> = token_rows
        .iter()
        .map(|(call_id, _)| call_id.clone())
        .collect();
    let llm_rows = sqlx::query(
        r#"
        SELECT
          json_extract(data, '$.call_id') AS call_id,
          json_extract(data, '$.token_usage_present') AS token_usage_present,
          json_extract(data, '$.token_usage_evidence') AS token_usage_evidence,
          json_extract(data, '$.est_input_tokens') AS est_input_tokens,
          json_extract(data, '$.failed') AS failed
        FROM events
        WHERE event_type = 'llm_call'
          AND created_at >= ?
        "#,
    )
    .bind(events_cutoff_rfc3339(chrono::Utc::now(), token_hours))
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let mut events_with_usage = 0i64;
    let mut estimate_only_calls = 0u64;
    let mut estimate_only_input_tokens = 0u64;
    let mut failed_calls_without_provider_usage = 0u64;
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
        let evidence = row
            .try_get::<Option<String>, _>("token_usage_evidence")
            .unwrap_or(None);
        if evidence.as_deref() == Some("estimated_input_only") {
            estimate_only_calls = estimate_only_calls.saturating_add(1);
            estimate_only_input_tokens = estimate_only_input_tokens.saturating_add(
                row.try_get::<Option<i64>, _>("est_input_tokens")
                    .unwrap_or(None)
                    .and_then(|value| u64::try_from(value).ok())
                    .unwrap_or_default(),
            );
        }
        let failed = row.try_get::<Option<i64>, _>("failed").unwrap_or(None) == Some(1);
        if failed && !token_usage_present {
            failed_calls_without_provider_usage =
                failed_calls_without_provider_usage.saturating_add(1);
        }
        event_rows.push((call_id, token_usage_present));
    }
    // Counterpart presence is resolved over the complete durable index. The
    // requested time window selects the reported cohort only; it cannot change
    // a call's matched/unmatched category.
    let all_token_call_ids =
        sqlx::query_scalar::<_, Option<String>>("SELECT call_id FROM token_usage")
            .fetch_all(&pool)
            .await
            .unwrap_or_default();
    let all_event_rows = sqlx::query(
        "SELECT json_extract(data, '$.call_id') AS call_id,
                json_extract(data, '$.token_usage_present') AS token_usage_present
         FROM events WHERE event_type = 'llm_call'",
    )
    .fetch_all(&pool)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|row| {
        let call_id = row.try_get::<Option<String>, _>("call_id").unwrap_or(None);
        let usage_present = row
            .try_get::<Option<i64>, _>("token_usage_present")
            .unwrap_or(None)
            == Some(1);
        (call_id, usage_present)
    })
    .collect::<Vec<_>>();
    let reconciliation = telemetry_reconciliation_counts_windowed(
        &token_call_ids,
        &event_rows,
        &all_token_call_ids,
        &all_event_rows,
    );
    println!(
        "- provider_usage_unavailable_failed_calls={} estimate_only_calls={} estimated_input_tokens={} (estimate; not provider-measured billing)",
        failed_calls_without_provider_usage, estimate_only_calls, estimate_only_input_tokens
    );
    println!(
        "- token_rows={} llm_events={} llm_events_token_usage_present={} correlated={} token_only={} event_only={} duplicate_token_rows={} duplicate_event_rows={} unattributed_legacy_token_rows={} unattributed_legacy_event_rows={}",
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
    if reconciliation.token_only > 0 {
        let breakdown = token_only_breakdown(&token_rows, &all_event_rows);
        println!(
            "- token_only split: event_missing={} (no llm_call event; likely LLM use outside the agent loop) event_usage_flag_false={}",
            breakdown.event_missing, breakdown.event_usage_flag_false,
        );
        println!("- token_only by session:");
        for (session, count) in breakdown.by_session.iter().take(10) {
            println!("  - session={} token_only_calls={}", session, count);
        }
        if breakdown.by_session.len() > 10 {
            println!("  - (+{} more sessions)", breakdown.by_session.len() - 10);
        }
    }

    println!("\n== Task Outcomes (Last {} Hours) ==", token_hours);
    let task_end_rows = sqlx::query(
        r#"
        SELECT data
        FROM events
        WHERE event_type = 'task_end'
          AND created_at >= ?
        "#,
    )
    .bind(events_cutoff_rfc3339(chrono::Utc::now(), token_hours))
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
          AND created_at >= ?
        "#,
    )
    .bind(events_cutoff_rfc3339(chrono::Utc::now(), token_hours))
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
            json_extract(data, '$.prompt_ms') AS prompt_ms,
            json_extract(data, '$.decode_ms') AS decode_ms,
            json_extract(data, '$.input_tokens') AS input_tokens,
            json_extract(data, '$.output_tokens') AS output_tokens,
            json_extract(data, '$.cached_input_tokens') AS cached_input_tokens,
            json_extract(data, '$.fresh_input_tokens') AS fresh_input_tokens,
            json_extract(data, '$.cache_creation_input_tokens') AS cache_creation_input_tokens
            ,json_extract(data, '$.est_input_tokens') AS est_input_tokens
            ,json_extract(data, '$.token_usage_evidence') AS token_usage_evidence
            ,json_extract(data, '$.failed') AS failed
            ,json_extract(data, '$.provider_error_kind') AS provider_error_kind
            ,json_extract(data, '$.provider_status') AS provider_status
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
            let latency_ms = row
                .try_get::<Option<i64>, _>("latency_ms")
                .unwrap_or(None)
                .unwrap_or(0);
            let prefill_ms = row.try_get::<Option<f64>, _>("prompt_ms").unwrap_or(None);
            let decode_ms = row.try_get::<Option<f64>, _>("decode_ms").unwrap_or(None);
            // The slice of wall-clock the server did NOT attribute to prefill or
            // decode: queue wait + transport. A large overhead on a warm call is
            // the contention signal (another session evicting/sharing the KV).
            let overhead_ms = match (prefill_ms, decode_ms) {
                (Some(p), Some(d)) => Some((latency_ms as f64 - p - d).round() as i64),
                _ => None,
            };
            println!(
                "- task_id={:?} iter={} model={} fell_back={} latency_ms={} prefill_ms={:?} decode_ms={:?} overhead_ms={:?} in_tok={} cached_in_tok={:?} fresh_in_tok={:?} cache_create_tok={:?} out_tok={} est_in_tok={:?} usage_evidence={:?} failed={} failure_kind={:?} provider_status={:?} at={}",
                row.try_get::<Option<String>, _>("task_id").unwrap_or(None),
                row.try_get::<Option<i64>, _>("iteration").unwrap_or(None).unwrap_or(0),
                row.try_get::<Option<String>, _>("model").unwrap_or(None).unwrap_or_default(),
                row.try_get::<Option<i64>, _>("fell_back").unwrap_or(None) == Some(1),
                latency_ms,
                prefill_ms.map(|v| v.round() as i64),
                decode_ms.map(|v| v.round() as i64),
                overhead_ms,
                row.try_get::<Option<i64>, _>("input_tokens").unwrap_or(None).unwrap_or(0),
                row.try_get::<Option<i64>, _>("cached_input_tokens").unwrap_or(None),
                row.try_get::<Option<i64>, _>("fresh_input_tokens").unwrap_or(None),
                row.try_get::<Option<i64>, _>("cache_creation_input_tokens")
                    .unwrap_or(None),
                row.try_get::<Option<i64>, _>("output_tokens").unwrap_or(None).unwrap_or(0),
                row.try_get::<Option<i64>, _>("est_input_tokens").unwrap_or(None),
                row.try_get::<Option<String>, _>("token_usage_evidence").unwrap_or(None),
                row.try_get::<Option<i64>, _>("failed").unwrap_or(None) == Some(1),
                row.try_get::<Option<String>, _>("provider_error_kind").unwrap_or(None),
                row.try_get::<Option<i64>, _>("provider_status").unwrap_or(None),
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
            SELECT id, event_type, tool_name, created_at, data,
                   substr(data, 1, 600) AS data_preview
            FROM events
            WHERE task_id = ?
            ORDER BY id ASC
            "#,
        )
        .bind(&task_id)
        .fetch_all(&pool)
        .await?;
        for row in rows {
            let raw_data = row.get::<String, _>("data");
            let parsed = serde_json::from_str::<serde_json::Value>(&raw_data).ok();
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
            if let Some(data) = parsed.as_ref() {
                if row.get::<String, _>("event_type") == "task_contract_compiled" {
                    let contract = data.get("contract").unwrap_or(&serde_json::Value::Null);
                    let evidence = contract
                        .get("evidence_requirements")
                        .and_then(serde_json::Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let invocations = data
                        .get("required_invocations")
                        .and_then(serde_json::Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    println!(
                        "  contract: task_kind={} requires_observation={} evidence={} invocations={} allowed_tools={}",
                        contract.get("task_kind").and_then(serde_json::Value::as_str).unwrap_or("unknown"),
                        contract.get("requires_observation").and_then(serde_json::Value::as_bool).unwrap_or(false),
                        evidence.len(),
                        invocations.len(),
                        contract.get("allowed_tool_names").map_or_else(|| "[]".to_string(), serde_json::Value::to_string),
                    );
                    for (index, requirement) in evidence.iter().enumerate() {
                        println!(
                            "  evidence[{index}]: purpose={} authority={} temporal={} scopes={} target={} receipt={}",
                            requirement.get("purpose").and_then(serde_json::Value::as_str).unwrap_or("unknown"),
                            requirement.get("minimum_authority").and_then(serde_json::Value::as_str).unwrap_or("unknown"),
                            requirement.get("temporal_scope").and_then(serde_json::Value::as_str).unwrap_or("unknown"),
                            requirement.get("acceptable_scopes").map_or_else(|| "[]".to_string(), serde_json::Value::to_string),
                            requirement.get("target").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                            requirement.get("receipt").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                        );
                    }
                    for (index, predicate) in invocations.iter().enumerate() {
                        println!("  invocation[{index}]: {predicate}");
                    }
                }
                if row.get::<String, _>("event_type") == "tool_call" {
                    println!(
                        "  claim: call_id={} stable_key={} obligations={} max_attempts={} max_invocations={} lineage={}",
                        data.get("tool_call_id").and_then(serde_json::Value::as_str).unwrap_or("none"),
                        data.get("stable_operation_key").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                        data.get("obligation_ids").map_or_else(|| "[]".to_string(), serde_json::Value::to_string),
                        data.get("max_operation_attempts").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                        data.get("max_operation_invocations").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                        data.get("operation_lineage").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                    );
                }
                if let Some(receipt) = data.get("receipt") {
                    println!(
                        "  receipt: call_id={} outcome={} evidence={} exit_code={} http_status={} result_id={} completeness={}/{} preflight={} closes={} continues={}",
                        data.get("tool_call_id").and_then(|value| value.as_str()).unwrap_or("none"),
                        receipt.get("outcome_status").and_then(|value| value.as_str()).unwrap_or("unknown"),
                        receipt.get("outcome_evidence").and_then(|value| value.as_str()).unwrap_or("unknown"),
                        receipt.get("exit_code").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                        receipt.get("http_status").map_or_else(|| "none".to_string(), serde_json::Value::to_string),
                        receipt.pointer("/result_provenance/result_id").and_then(|value| value.as_str()).unwrap_or("none"),
                        receipt.pointer("/result_provenance/model_view_completeness").and_then(|value| value.as_str()).unwrap_or("unknown"),
                        receipt.pointer("/result_provenance/durable_view_completeness").and_then(|value| value.as_str()).unwrap_or("unknown"),
                        receipt.pointer("/authorization_preflight/status").and_then(|value| value.as_str()).unwrap_or("none"),
                        receipt.get("completion_obligation_ids").map_or_else(|| "[]".to_string(), serde_json::Value::to_string),
                        receipt.get("continuation_obligation_ids").map_or_else(|| "[]".to_string(), serde_json::Value::to_string),
                    );
                }
                let metadata = data.get("metadata");
                let contract_scope = metadata
                    .and_then(|value| value.get("contract_scope_task_id"))
                    .and_then(|value| value.as_str());
                let adopted =
                    metadata.and_then(|value| value.get("contract_adopted_from_task_ids"));
                if contract_scope.is_some() || adopted.is_some() {
                    println!(
                        "  contract_lineage: scope_task_id={} adopted_from={}",
                        contract_scope.unwrap_or("unbound"),
                        adopted.map_or_else(|| "[]".to_string(), serde_json::Value::to_string),
                    );
                }
            }
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
                json_extract(data, '$.prompt_ms') AS prompt_ms,
                json_extract(data, '$.decode_ms') AS decode_ms,
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
                let latency_ms = row
                    .try_get::<Option<i64>, _>("latency_ms")
                    .unwrap_or(None)
                    .unwrap_or(0);
                let prefill_ms = row.try_get::<Option<f64>, _>("prompt_ms").unwrap_or(None);
                let decode_ms = row.try_get::<Option<f64>, _>("decode_ms").unwrap_or(None);
                let overhead_ms = match (prefill_ms, decode_ms) {
                    (Some(p), Some(d)) => Some((latency_ms as f64 - p - d).round() as i64),
                    _ => None,
                };
                println!(
                    "- iter={} model={} fell_back={} attempts={} latency_ms={} prefill_ms={:?} decode_ms={:?} overhead_ms={:?} build_ms={} in_tok={} cached_in_tok={:?} fresh_in_tok={:?} cache_create_tok={:?} out_tok={} est_in_tok={}",
                    row.try_get::<Option<i64>, _>("iteration").unwrap_or(None).unwrap_or(0),
                    model_label,
                    fell_back,
                    row.try_get::<Option<i64>, _>("attempts").unwrap_or(None).unwrap_or(0),
                    latency_ms,
                    prefill_ms.map(|v| v.round() as i64),
                    decode_ms.map(|v| v.round() as i64),
                    overhead_ms,
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
