//! Event store implementation using SQLite.
//!
//! The EventStore provides CRUD operations for events, with support for:
//! - Efficient querying by session, time window, and event type
//! - Conversation history retrieval from canonical events
//! - Consolidation tracking and pruning

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use sqlx::{Row, SqlitePool};
use tracing::{info, warn};

use super::conversation_turn::{group_rows_into_turns, FetchedRow, FetchedTurn};
use super::{
    DecisionPointData, DecisionType, Event, EventType, InteractionRequestedData,
    InteractionResolvedData, LlmCallData, PolicyDecisionData, ResourceRegisteredData, TaskEndData,
    TaskStatus, ToolCallData, ToolResultData,
};
use crate::traits::{Message, TokenUsage};

#[derive(Debug, Clone)]
pub struct GeneratedResponseRef {
    pub response_id: String,
    pub task_id: String,
    pub turn_id: Option<String>,
    pub referenced_receipts: Vec<super::CompletionProofReference>,
}

/// Exact durable execution evidence that caused a runtime continuation.
///
/// Both halves are required: the call owns the requested arguments while the
/// terminal result owns the structured outcome and completion semantics. The
/// result provenance ID prevents an earlier background-transition receipt for
/// the same call from being mistaken for the eventual terminal outcome.
#[derive(Debug, Clone)]
pub struct ContinuationToolEvidence {
    pub call: ToolCallData,
    pub result: ToolResultData,
}

/// The event store backed by SQLite.
pub struct EventStore {
    pool: SqlitePool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskWindowStats {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    #[serde(default)]
    pub interrupted: u64,
    pub stalled: u64,
    pub error_events: u64,
    pub outcome_succeeded: u64,
    pub outcome_partial: u64,
    pub outcome_failed: u64,
    pub outcome_unknown: u64,
    pub completion_rate: f64,
    pub error_rate: f64,
    pub stall_rate: f64,
    pub semantic_success_rate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolStats {
    pub total_calls: u64,
    pub successful: u64,
    pub failed: u64,
    pub avg_duration_ms: u64,
    /// (error pattern, count), top 3.
    pub common_errors: Vec<(String, u64)>,
}

struct AtomicMandateReceiptProjection {
    mandate_id: String,
    mandate_version: i64,
    decision_cycle_id: String,
    reserved_action_attempt: i64,
    action_digest: String,
    tool_call_id: String,
    status: &'static str,
    outcome_evidence: Option<&'static str>,
    http_status: Option<i64>,
    exit_code: Option<i32>,
    completed_at: String,
}

fn mandate_projection_from_tool_result(
    result: &ToolResultData,
    completed_at: &str,
) -> anyhow::Result<Option<AtomicMandateReceiptProjection>> {
    let Some(receipt) = result.receipt.as_ref() else {
        return Ok(None);
    };
    let Some(grant) = receipt.mandate_authority.as_ref() else {
        return Ok(None);
    };
    anyhow::ensure!(
        grant.counts_toward_cycle_budget
            && grant.reserved_action_attempt > 0
            && grant.tool_call_id.as_deref() == Some(result.tool_call_id.as_str()),
        "mandate receipt carries an invalid action-bound grant"
    );
    let evidence = match receipt.outcome_evidence {
        crate::events::ToolOutcomeEvidenceSource::ToolReported => Some("tool_reported"),
        crate::events::ToolOutcomeEvidenceSource::StructuredMetadata => Some("structured_metadata"),
        crate::events::ToolOutcomeEvidenceSource::DurableReplay
        | crate::events::ToolOutcomeEvidenceSource::LegacyText => None,
    };
    let clean_transport = receipt.transport_error.is_none()
        && !receipt.timed_out
        && !receipt.background_started
        && !receipt.detached
        && !receipt.completion_notifications_enabled;
    let exit_success = receipt.exit_code.is_none_or(|code| code == 0);
    let http_success = result.name != "http_request"
        || receipt
            .http_status
            .is_some_and(|status| (200..300).contains(&status) && status != 202);
    let strict_success = receipt.schema_version == crate::events::ToolReceiptV1::SCHEMA_VERSION
        && receipt.outcome_status == crate::traits::ToolOutcomeStatus::Succeeded
        && evidence.is_some()
        && clean_transport
        && exit_success
        && http_success
        && receipt.semantics.mutates_state();
    let explicit_failure = receipt.schema_version == crate::events::ToolReceiptV1::SCHEMA_VERSION
        && receipt.outcome_status.is_failure()
        && evidence.is_some()
        && clean_transport
        && receipt.semantics.mutates_state();
    let status = if strict_success {
        "succeeded"
    } else if explicit_failure {
        "failed"
    } else {
        "ambiguous"
    };
    Ok(Some(AtomicMandateReceiptProjection {
        mandate_id: grant.mandate_id.clone(),
        mandate_version: grant.mandate_version,
        decision_cycle_id: grant.decision_cycle_id.clone(),
        reserved_action_attempt: grant.reserved_action_attempt,
        action_digest: grant.action_digest.clone(),
        tool_call_id: result.tool_call_id.clone(),
        status,
        outcome_evidence: evidence,
        http_status: receipt.http_status.map(i64::from),
        exit_code: receipt.exit_code,
        completed_at: completed_at.to_string(),
    }))
}

/// Aggregated latency + token telemetry over recent `LlmCall` events.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LlmStats {
    pub total_calls: u64,
    pub failed_calls: u64,
    pub avg_latency_ms: u64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub max_latency_ms: u64,
    pub fell_back_count: u64,
    pub avg_input_tokens: u64,
    pub avg_output_tokens: u64,
}

/// Per-task rollup of `LlmCall` telemetry. Powers the self-diagnosis LLM
/// summary (Tier 1) and the per-turn efficiency reflection signal (Tier 2).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TaskLlmSummary {
    pub total_calls: u64,
    pub failed_calls: u64,
    /// Most recent terminal provider error, when any call failed.
    pub last_error: Option<String>,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cached_input_tokens: u64,
    pub cached_input_token_samples: u64,
    pub total_cache_creation_input_tokens: u64,
    pub cache_creation_input_token_samples: u64,
    /// Sum of estimated input tokens over calls that carried an estimate.
    pub total_est_input_tokens: u64,
    /// Sum of actual input tokens over calls that carried an estimate
    /// (paired denominator for drift, so the comparison is apples-to-apples).
    pub actual_input_tokens_with_est: u64,
    /// Number of calls that carried an input-token estimate.
    pub est_samples: u64,
    pub avg_latency_ms: u64,
    pub p50_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub max_latency_ms: u64,
    /// Iteration (1-based) of the slowest call in the task.
    pub max_latency_iteration: u32,
    pub fell_back_count: u64,
    /// Sum of provider attempts across calls (attempts > calls ⇒ retries).
    pub total_attempts: u64,
    /// Model that produced the final call (helps distinguish model vs logic).
    pub final_model: Option<String>,
}

impl TaskLlmSummary {
    /// Signed est-vs-actual input-token drift over calls that had an estimate.
    /// Positive ⇒ estimate ran high (over-counted); negative ⇒ under-counted.
    pub fn est_input_drift(&self) -> i64 {
        self.total_est_input_tokens as i64 - self.actual_input_tokens_with_est as i64
    }

    /// Whether this turn looks notably inefficient and worth flagging.
    /// Tuned to be quiet on healthy turns: fires on terminal provider failure,
    /// retries/fallback, heavy iteration loops, or large est-vs-actual token drift.
    pub fn is_inefficient(&self) -> bool {
        let retried = self.total_attempts > self.total_calls;
        let looped = self.total_calls >= 8;
        let drift_ratio = if self.actual_input_tokens_with_est > 0 {
            (self.est_input_drift().unsigned_abs() as f64)
                / (self.actual_input_tokens_with_est as f64)
        } else {
            0.0
        };
        let big_drift = self.est_samples > 0 && drift_ratio >= 0.30;
        self.failed_calls > 0 || self.fell_back_count > 0 || retried || looped || big_drift
    }
}

impl Default for TaskWindowStats {
    fn default() -> Self {
        Self {
            total: 0,
            completed: 0,
            failed: 0,
            cancelled: 0,
            interrupted: 0,
            stalled: 0,
            error_events: 0,
            outcome_succeeded: 0,
            outcome_partial: 0,
            outcome_failed: 0,
            outcome_unknown: 0,
            completion_rate: 1.0,
            error_rate: 0.0,
            stall_rate: 0.0,
            semantic_success_rate: 1.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolicyGraduationReport {
    pub window_days: u32,
    pub observed_days: f64,
    pub total_decisions: u64,
    pub diverged_decisions: u64,
    pub divergence_rate: f64,
    pub current: TaskWindowStats,
    pub previous: TaskWindowStats,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionWriteDrift {
    pub session_id: String,
    pub message_rows: u64,
    pub event_rows: u64,
    pub delta: i64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WriteConsistencyReport {
    pub generated_at: String,
    pub conversation_event_rows: u64,
    pub missing_message_id_events: u64,
    pub global_delta: i64,
    pub session_mismatch_count: u64,
    pub stale_task_starts: u64,
    pub top_session_drifts: Vec<SessionWriteDrift>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct WriteConsistencyThresholds {
    pub max_abs_global_delta: u64,
    pub max_session_mismatch_count: u64,
    pub max_stale_task_starts: u64,
    pub max_missing_message_id_events: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WriteConsistencyGateStatus {
    pub passed: bool,
    pub reasons: Vec<String>,
    pub thresholds: WriteConsistencyThresholds,
}

impl PolicyGraduationReport {
    pub fn gate_passes(&self, max_divergence: f64) -> bool {
        if self.observed_days < self.window_days as f64 {
            return false;
        }
        if self.total_decisions == 0 {
            return false;
        }
        if self.divergence_rate >= max_divergence {
            return false;
        }
        // No-regression gate:
        // completion must not decrease; error/stall must not increase.
        let completion_ok = self.current.completion_rate >= self.previous.completion_rate;
        let error_ok = self.current.error_rate <= self.previous.error_rate;
        let stall_ok = self.current.stall_rate <= self.previous.stall_rate;
        completion_ok && error_ok && stall_ok
    }
}

impl Default for WriteConsistencyThresholds {
    fn default() -> Self {
        Self {
            // Canonical event-path defaults.
            max_abs_global_delta: 3,
            max_session_mismatch_count: 0,
            max_stale_task_starts: 0,
            max_missing_message_id_events: 0,
        }
    }
}

impl WriteConsistencyReport {
    pub fn evaluate_gate(&self) -> WriteConsistencyGateStatus {
        self.evaluate_gate_with(WriteConsistencyThresholds::default())
    }

    pub fn evaluate_gate_with(
        &self,
        thresholds: WriteConsistencyThresholds,
    ) -> WriteConsistencyGateStatus {
        let mut reasons = Vec::new();

        let abs_global_delta = self.global_delta.unsigned_abs();
        if abs_global_delta > thresholds.max_abs_global_delta {
            reasons.push(format!(
                "global delta {} exceeds threshold {}",
                abs_global_delta, thresholds.max_abs_global_delta
            ));
        }
        if self.session_mismatch_count > thresholds.max_session_mismatch_count {
            reasons.push(format!(
                "session mismatch count {} exceeds threshold {}",
                self.session_mismatch_count, thresholds.max_session_mismatch_count
            ));
        }

        if self.stale_task_starts > thresholds.max_stale_task_starts {
            reasons.push(format!(
                "stale task starts {} exceeds threshold {}",
                self.stale_task_starts, thresholds.max_stale_task_starts
            ));
        }

        if self.missing_message_id_events > thresholds.max_missing_message_id_events {
            reasons.push(format!(
                "events missing message_id {} exceeds threshold {}",
                self.missing_message_id_events, thresholds.max_missing_message_id_events
            ));
        }

        WriteConsistencyGateStatus {
            passed: reasons.is_empty(),
            reasons,
            thresholds,
        }
    }
}

impl EventStore {
    pub async fn event_watermark(&self) -> anyhow::Result<i64> {
        Ok(
            sqlx::query_scalar("SELECT COALESCE(MAX(id), 0) FROM events")
                .fetch_one(&self.pool)
                .await?,
        )
    }

    pub async fn session_context_boundary(&self, session_id: &str) -> anyhow::Result<Option<i64>> {
        Ok(sqlx::query_scalar(
            "SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?)
    }

    /// Turn identities whose compiled task policy excludes them from every
    /// automatic memory pipeline. Canonical history queries intentionally do
    /// not use this filter; memory consumers opt in explicitly.
    pub async fn memory_suppressed_turn_ids(
        &self,
        session_id: &str,
    ) -> anyhow::Result<std::collections::HashSet<String>> {
        Ok(sqlx::query_scalar(
            "SELECT DISTINCT turn_id FROM events
             WHERE session_id = ? AND event_type = 'memory_policy_compiled'
               AND turn_id IS NOT NULL
               AND json_extract(data, '$.access') = 'suppressed'",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .collect())
    }

    /// Return the exact generated-response identity for a completed agent
    /// handoff. The immutable pre-dispatch watermark is the selector; content
    /// is only an integrity check. This prevents an identical historical reply
    /// from being attributed to the current delivery.
    pub async fn generated_response_after(
        &self,
        session_id: &str,
        event_watermark: i64,
        content: &str,
    ) -> anyhow::Result<Option<GeneratedResponseRef>> {
        let row = sqlx::query(
            "SELECT id, task_id, turn_id, data
             FROM events
             WHERE session_id = ? AND id > ? AND event_type = 'assistant_response'
               AND json_type(data, '$.message_id') = 'text'
             ORDER BY id DESC LIMIT 1",
        )
        .bind(session_id)
        .bind(event_watermark)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let raw: String = row.get("data");
        let data: serde_json::Value = serde_json::from_str(&raw)?;
        anyhow::ensure!(
            data.get("content").and_then(serde_json::Value::as_str) == Some(content),
            "latest generated response after dispatch does not match returned content"
        );
        let response_id = data
            .get("message_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("assistant response is missing message_id"))?
            .to_string();
        let task_id = row
            .try_get::<Option<String>, _>("task_id")?
            .or_else(|| {
                data.get("task_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .ok_or_else(|| anyhow::anyhow!("assistant response is missing task_id"))?;
        let referenced_receipts = data
            .get("referenced_receipts")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default();
        Ok(Some(GeneratedResponseRef {
            response_id,
            task_id,
            turn_id: row.try_get("turn_id")?,
            referenced_receipts,
        }))
    }

    /// All durable structured receipts recorded under one exact task. The
    /// response and TaskEnd proof graphs both use this projection so audit
    /// cannot drift between two independent correlation rules.
    pub async fn task_completion_proof_references(
        &self,
        task_id: &str,
    ) -> anyhow::Result<Vec<super::CompletionProofReference>> {
        let rows = sqlx::query(
            "SELECT json_extract(data, '$.tool_call_id') AS receipt_id,
                    json_extract(data, '$.receipt.result_provenance.result_id') AS result_id,
                    json_extract(data, '$.receipt.completion_obligation_ids') AS obligation_ids
             FROM events
             WHERE event_type = 'tool_result' AND task_id = ?
               AND json_extract(data, '$.receipt.schema_version') IS NOT NULL
               AND json_array_length(
                    COALESCE(json_extract(data, '$.receipt.completion_obligation_ids'), '[]')
                   ) > 0
             ORDER BY id ASC",
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;
        let mut seen = std::collections::HashSet::new();
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let receipt_id = row.try_get::<String, _>("receipt_id").ok()?;
                seen.insert(receipt_id.clone())
                    .then(|| super::CompletionProofReference {
                        receipt_id,
                        result_id: row.try_get("result_id").ok().flatten(),
                        obligation_ids: row
                            .try_get::<Option<String>, _>("obligation_ids")
                            .ok()
                            .flatten()
                            .and_then(|raw| serde_json::from_str(&raw).ok())
                            .unwrap_or_default(),
                    })
            })
            .collect())
    }

    /// Load the exact parent receipt named by an internal continuation edge.
    ///
    /// Correlation is intentionally structural and fail-closed: session,
    /// parent task, tool-call ID, and result-provenance ID must all agree.
    pub async fn continuation_tool_evidence(
        &self,
        session_id: &str,
        parent_task_id: &str,
        parent_tool_call_id: &str,
        parent_result_id: &str,
    ) -> anyhow::Result<Option<ContinuationToolEvidence>> {
        let row = sqlx::query(
            r#"
            SELECT call_event.data AS call_data, result_event.data AS result_data
            FROM events AS result_event
            JOIN events AS call_event
              ON call_event.session_id = result_event.session_id
             AND call_event.task_id = result_event.task_id
             AND call_event.event_type = 'tool_call'
             AND json_extract(call_event.data, '$.tool_call_id') =
                 json_extract(result_event.data, '$.tool_call_id')
            WHERE result_event.session_id = ?
              AND result_event.task_id = ?
              AND result_event.event_type = 'tool_result'
              AND json_extract(result_event.data, '$.tool_call_id') = ?
              AND json_extract(
                    result_event.data,
                    '$.receipt.result_provenance.result_id'
                  ) = ?
            ORDER BY result_event.id DESC, call_event.id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .bind(parent_task_id)
        .bind(parent_tool_call_id)
        .bind(parent_result_id)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let call: ToolCallData = serde_json::from_str(&row.get::<String, _>("call_data"))?;
        let result: ToolResultData = serde_json::from_str(&row.get::<String, _>("result_data"))?;
        anyhow::ensure!(
            call.task_id.as_deref() == Some(parent_task_id)
                && result.task_id.as_deref() == Some(parent_task_id)
                && call.tool_call_id == parent_tool_call_id
                && result.tool_call_id == parent_tool_call_id,
            "continuation receipt correlation changed after query"
        );
        Ok(Some(ContinuationToolEvidence { call, result }))
    }

    pub async fn task_response_message_ids(&self, task_id: &str) -> anyhow::Result<Vec<String>> {
        Ok(sqlx::query_scalar(
            "SELECT json_extract(data, '$.message_id')
             FROM events
             WHERE event_type = 'assistant_response' AND task_id = ?
               AND json_type(data, '$.message_id') = 'text'
             ORDER BY id ASC",
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?)
    }

    /// Create a new EventStore with the given database pool.
    /// This also runs migrations to create/update the events table.
    pub async fn new(pool: SqlitePool) -> anyhow::Result<Self> {
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Get the underlying database pool (for sharing with other components)
    pub fn pool(&self) -> SqlitePool {
        self.pool.clone()
    }

    /// Run database migrations for the events table
    async fn migrate(&self) -> anyhow::Result<()> {
        crate::db::migrations::migrate_events(&self.pool).await
    }

    // =========================================================================
    // Write Operations
    // =========================================================================

    /// Append a new event to the store. Returns the assigned event ID.
    pub async fn append(&self, event: Event) -> anyhow::Result<i64> {
        let data_json = serde_json::to_string(&event.data)?;
        let event_type_str = event.event_type.as_str();
        let created_at_str = event.created_at.to_rfc3339();

        let mandate_projection = if event.event_type == EventType::ToolResult {
            match serde_json::from_value::<ToolResultData>(event.data.clone()) {
                Ok(result) => mandate_projection_from_tool_result(&result, &created_at_str)?,
                Err(_) => None,
            }
        } else {
            None
        };

        let mut tx = self.pool.begin().await?;

        let result = sqlx::query(
            r#"
            INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name, turn_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&event.session_id)
        .bind(event_type_str)
        .bind(&data_json)
        .bind(&created_at_str)
        .bind(&event.task_id)
        .bind(&event.tool_name)
        .bind(&event.turn_id)
        .execute(&mut *tx)
        .await?;

        let event_id = result.last_insert_rowid();
        if let Some(projection) = mandate_projection {
            let updated = sqlx::query(
                "UPDATE mandate_mutation_attempts
                 SET status = ?, outcome_evidence = ?, http_status = ?, exit_code = ?,
                     completed_at = ?
                 WHERE mandate_id = ? AND mandate_version = ?
                   AND decision_cycle_id = ? AND reserved_action_attempt = ?
                   AND action_digest = ? AND tool_call_id = ?
                   AND status = 'reserved' AND dispatch_claimed_at IS NOT NULL",
            )
            .bind(projection.status)
            .bind(projection.outcome_evidence)
            .bind(projection.http_status)
            .bind(projection.exit_code)
            .bind(&projection.completed_at)
            .bind(&projection.mandate_id)
            .bind(projection.mandate_version)
            .bind(&projection.decision_cycle_id)
            .bind(projection.reserved_action_attempt)
            .bind(&projection.action_digest)
            .bind(&projection.tool_call_id)
            .execute(&mut *tx)
            .await?;
            anyhow::ensure!(
                updated.rows_affected() == 1,
                "mandate mutation receipt did not match its claimed reservation"
            );
        }
        tx.commit().await?;
        if matches!(event_type_str, "user_message" | "assistant_response") {
            if let Err(error) =
                crate::state::sqlite::history_search::project_event(&self.pool, event_id).await
            {
                // Never couple the canonical append transaction to FTS health.
                tracing::debug!(%error, event_id, "Deferred exact-history projection");
            }
        }
        Ok(event_id)
    }

    /// Project one canonical user-message event into the derived memory graph.
    ///
    /// Callers invoke this only after the task's semantic capability policy has
    /// been compiled. Keeping projection out of [`append`](Self::append) closes
    /// the former ordering gap where a no-memory request was persisted to the
    /// memory graph before its constraint had been assessed.
    pub async fn project_user_message_memory_span(&self, event_id: i64) -> anyhow::Result<()> {
        crate::state::sqlite::memory::project_event_span(&self.pool, event_id).await
    }

    /// Atomically append the canonical model-call event and its aggregate
    /// token-usage projection. Keeping both writes on this store's shared
    /// SQLite transaction prevents token-only and event-only telemetry rows.
    pub async fn append_llm_call_with_token_usage(
        &self,
        event: Event,
        usage: Option<&TokenUsage>,
        call_id: &str,
    ) -> anyhow::Result<i64> {
        anyhow::ensure!(
            event.event_type == EventType::LlmCall,
            "atomic model telemetry append requires an llm_call event"
        );
        anyhow::ensure!(!call_id.trim().is_empty(), "model call_id is required");

        let data_json = serde_json::to_string(&event.data)?;
        let created_at_str = event.created_at.to_rfc3339();
        let mut tx = self.pool.begin().await?;
        let result = sqlx::query(
            r#"
            INSERT INTO events (session_id, event_type, data, created_at, task_id, tool_name, turn_id)
            VALUES (?, 'llm_call', ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&event.session_id)
        .bind(&data_json)
        .bind(&created_at_str)
        .bind(&event.task_id)
        .bind(&event.tool_name)
        .bind(&event.turn_id)
        .execute(&mut *tx)
        .await?;

        if let Some(usage) = usage {
            sqlx::query(
                "INSERT INTO token_usage (
                    session_id, model, input_tokens, output_tokens,
                    cached_input_tokens, cache_creation_input_tokens, call_id, created_at
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .bind(&event.session_id)
            .bind(&usage.model)
            .bind(i64::from(usage.input_tokens))
            .bind(i64::from(usage.output_tokens))
            .bind(usage.cached_input_tokens.map(i64::from))
            .bind(usage.cache_creation_input_tokens.map(i64::from))
            .bind(call_id)
            .bind(&created_at_str)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(result.last_insert_rowid())
    }

    /// Mark events as consolidated
    pub async fn mark_consolidated(&self, event_ids: &[i64]) -> anyhow::Result<()> {
        if event_ids.is_empty() {
            return Ok(());
        }

        let now = Utc::now().to_rfc3339();
        let placeholders: Vec<String> = event_ids.iter().map(|_| "?".to_string()).collect();
        let query = format!(
            "UPDATE events SET consolidated_at = ? WHERE id IN ({})",
            placeholders.join(",")
        );

        let mut q = sqlx::query(&query).bind(&now);
        for id in event_ids {
            q = q.bind(id);
        }
        q.execute(&self.pool).await?;

        Ok(())
    }

    // =========================================================================
    // Read Operations - General Queries
    // =========================================================================

    /// Query events for a session within a time window
    pub async fn query_events(
        &self,
        session_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<Event>> {
        let since_str = since.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND created_at >= ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query events eligible for implicit conversation context. Unlike audit,
    /// memory, and explicit history queries, this respects the durable task
    /// boundary and excludes delayed events whose turn originated before it.
    pub async fn query_context_events(
        &self,
        session_id: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<Vec<Event>> {
        let since_str = since.to_rfc3339();
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ?1
              AND created_at >= ?2
              AND id > COALESCE(
                (SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?1),
                0)
              AND (
                turn_id IS NULL
                OR (SELECT MIN(origin.id)
                    FROM events origin
                    WHERE origin.session_id = ?1
                      AND origin.turn_id = events.turn_id) > COALESCE(
                    (SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?1),
                    0)
              )
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query events by type for a session
    pub async fn query_events_by_types(
        &self,
        session_id: &str,
        types: &[EventType],
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        if types.is_empty() {
            return Ok(vec![]);
        }

        let type_strs: Vec<&str> = types.iter().map(|t| t.as_str()).collect();
        let placeholders: Vec<String> = types.iter().map(|_| "?".to_string()).collect();

        let query = format!(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND event_type IN ({})
            ORDER BY created_at DESC
            LIMIT ?
            "#,
            placeholders.join(",")
        );

        let mut q = sqlx::query(&query).bind(session_id);
        for type_str in type_strs {
            q = q.bind(type_str);
        }
        q = q.bind(limit as i64);

        let rows = q.fetch_all(&self.pool).await?;
        self.rows_to_events(rows)
    }

    /// Query recent events for a session (all types)
    pub async fn query_recent_events(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        // Reverse to get chronological order
        let mut events = self.rows_to_events(rows)?;
        events.reverse();
        Ok(events)
    }

    /// Reconstruct approvals that have a durable request but no later durable
    /// resolution. The projection deliberately reads the full interaction
    /// lifecycle for the session: silently dropping an old unresolved request
    /// because of an arbitrary recency limit would be unsafe.
    pub async fn get_pending_interactions(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Vec<InteractionRequestedData>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ?
              AND event_type IN ('interaction_requested', 'interaction_resolved')
            ORDER BY id ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        let events = self.rows_to_events(rows)?;
        let mut pending =
            std::collections::HashMap::<String, (i64, InteractionRequestedData)>::new();
        for event in events {
            match event.event_type {
                EventType::InteractionRequested => {
                    if let Ok(data) = event.parse_data::<InteractionRequestedData>() {
                        pending.insert(data.interaction_id.clone(), (event.id, data));
                    }
                }
                EventType::InteractionResolved => {
                    if let Ok(data) = event.parse_data::<InteractionResolvedData>() {
                        pending.remove(&data.interaction_id);
                    }
                }
                _ => {}
            }
        }
        let mut pending = pending.into_values().collect::<Vec<_>>();
        pending.sort_by_key(|(event_id, _)| *event_id);
        Ok(pending.into_iter().map(|(_, data)| data).collect())
    }

    /// Reconstruct unresolved approval interactions belonging to one task.
    /// Completion uses this as a final fail-closed check so a model response
    /// cannot race ahead of an approval resolution.
    pub async fn get_pending_task_interactions(
        &self,
        task_id: &str,
    ) -> anyhow::Result<Vec<InteractionRequestedData>> {
        let events = self.query_task_events(task_id).await?;
        let mut pending = std::collections::HashMap::<String, InteractionRequestedData>::new();
        for event in events {
            match event.event_type {
                EventType::InteractionRequested => {
                    if let Ok(data) = event.parse_data::<InteractionRequestedData>() {
                        pending.insert(data.interaction_id.clone(), data);
                    }
                }
                EventType::InteractionResolved => {
                    if let Ok(data) = event.parse_data::<InteractionResolvedData>() {
                        pending.remove(&data.interaction_id);
                    }
                }
                _ => {}
            }
        }
        Ok(pending.into_values().collect())
    }

    /// Query events for a specific task
    pub async fn query_task_events(&self, task_id: &str) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query events for a specific task scoped to a session.
    pub async fn query_task_events_for_session(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND task_id = ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query recent task_end events for a session.
    /// When failures_only is true, only failed task_end events are returned.
    pub async fn query_recent_task_ends(
        &self,
        session_id: &str,
        failures_only: bool,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        let fetch_limit = if failures_only {
            limit.saturating_mul(8)
        } else {
            limit
        }
        .max(1);
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ?
              AND event_type = 'task_end'
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(fetch_limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut events = self.rows_to_events(rows)?;
        if failures_only {
            events.retain(|e| {
                e.parse_data::<TaskEndData>()
                    .ok()
                    .is_some_and(|d| d.effective_outcome() != crate::events::TaskOutcome::Succeeded)
            });
        }
        events.truncate(limit.max(1));
        Ok(events)
    }

    /// Query decision_point events for a specific task scoped to a session.
    pub async fn query_decision_points(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND task_id = ? AND event_type = 'decision_point'
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Query recent intent-gate decision_point events scoped to a session.
    /// Returned in reverse-chronological order.
    pub async fn query_recent_intent_gate_decision_points(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        let fetch_limit = limit.max(1).saturating_mul(5).max(20);
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND event_type = 'decision_point'
            ORDER BY created_at DESC
            LIMIT ?
            "#,
        )
        .bind(session_id)
        .bind(fetch_limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut events = self.rows_to_events(rows)?;
        events.retain(|e| {
            e.parse_data::<DecisionPointData>()
                .ok()
                .is_some_and(|d| d.decision_type == DecisionType::IntentGate)
        });
        events.truncate(limit.max(1));
        Ok(events)
    }

    /// Get unconsolidated events for a session (for consolidation)
    pub async fn query_unconsolidated(&self, session_id: &str) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND consolidated_at IS NULL
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        self.rows_to_events(rows)
    }

    /// Get sessions with unconsolidated events older than a cutoff
    pub async fn get_sessions_needing_consolidation(&self) -> anyhow::Result<Vec<String>> {
        let rows = sqlx::query(
            r#"
            SELECT DISTINCT session_id
            FROM events
            WHERE consolidated_at IS NULL
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get("session_id")).collect())
    }

    /// Get sessions with old unconsolidated events (before cutoff)
    pub async fn get_sessions_with_old_unconsolidated_events(
        &self,
        before: DateTime<Utc>,
    ) -> anyhow::Result<Vec<String>> {
        let before_str = before.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT DISTINCT session_id
            FROM events
            WHERE consolidated_at IS NULL AND created_at < ?
            "#,
        )
        .bind(&before_str)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get("session_id")).collect())
    }

    // =========================================================================
    // Read Operations - Conversation History
    // =========================================================================

    /// Get conversation history for a session (for LLM context)
    /// Returns runtime messages projected from canonical events.
    pub async fn get_conversation_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        let events = self
            .query_events_by_types(
                session_id,
                &[
                    EventType::UserMessage,
                    EventType::AssistantResponse,
                    EventType::ToolResult,
                ],
                limit * 3, // Fetch more to account for tool results
            )
            .await?;

        // Convert events to runtime Message format.
        // Reverse to chronological (query returns newest-first).
        let mut messages = Vec::new();
        for event in events.into_iter().rev() {
            if let Some(msg) = crate::events::turn_from_event(
                event.id,
                &event.session_id,
                event.event_type.as_str(),
                &event.data,
                event.created_at,
            )
            .map(|turn| turn.into_message())
            {
                messages.push(msg);
            }
        }

        Ok(crate::conversation::truncate_with_anchor(messages, limit))
    }

    // =========================================================================
    // Read Operations - Turn-Anchored Conversation Fetch (Pillar B)
    // =========================================================================

    /// Fetch all whole turns whose turn-start sequence is `>= anchor_turn_seq`,
    /// ordered by `(turn_seq, msg_seq)` — id-only ordering, no timestamps.
    ///
    /// `turn_seq = MIN(events.id)` per `turn_id`, computed at fetch time via a
    /// join subquery (immutable: a later, higher id cannot lower the MIN).
    /// `msg_seq = events.id`. Legacy `turn_id IS NULL` rows are excluded.
    /// Each turn's `terminal_status` comes from a LEFT JOIN to its latest
    /// `task_end` (`MAX(id)` among that turn's `task_end` rows — latest-wins),
    /// folded into the same query (no N+1). `task_end` is not part of the
    /// reconstructed conversation rows.
    ///
    /// `anchor_turn_seq = 0` is a full-session scan; do NOT use it as a
    /// cold-start init path for long-lived sessions — use
    /// [`Self::get_recent_turns_page`] for bounded, reverse-walked init.
    pub async fn get_turns_from_anchor(
        &self,
        session_id: &str,
        anchor_turn_seq: i64,
    ) -> anyhow::Result<Vec<FetchedTurn>> {
        let rows = sqlx::query(
            r#"
            SELECT e.id, e.event_type, e.data, e.created_at, e.turn_id, t.turn_seq, s.status
            FROM events e
            JOIN (
                SELECT turn_id, MIN(id) AS turn_seq
                FROM events
                WHERE session_id = ?1 AND turn_id IS NOT NULL
                GROUP BY turn_id
                HAVING MIN(id) > COALESCE(
                    (SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?1),
                    0)
            ) t ON e.turn_id = t.turn_id
            LEFT JOIN (
                SELECT te.turn_id,
                       json_extract(te.data, '$.status') AS status,
                       te.id
                FROM events te
                WHERE te.session_id = ?1 AND te.turn_id IS NOT NULL
                  AND te.event_type = 'task_end'
                  AND te.id = (SELECT MAX(te2.id) FROM events te2
                               WHERE te2.session_id = ?1
                                 AND te2.turn_id = te.turn_id
                                 AND te2.event_type = 'task_end')
            ) s ON e.turn_id = s.turn_id
            WHERE e.session_id = ?1
              AND e.turn_id IS NOT NULL
              AND t.turn_seq >= ?2
              AND e.event_type IN ('user_message','assistant_response','tool_result')
            ORDER BY t.turn_seq ASC, e.id ASC
            "#,
        )
        .bind(session_id)
        .bind(anchor_turn_seq)
        .fetch_all(&self.pool)
        .await?;

        Ok(group_rows_into_turns(self.fetched_rows(session_id, rows)?))
    }

    /// Reverse-walk a bounded page of the most recent turns for cold-start
    /// init. Turns are selected newest→oldest (`turn_seq DESC`, `LIMIT`) in a
    /// `selected_turns` CTE BEFORE message rows are expanded, so `LIMIT` counts
    /// whole turns — a large turn is never split or skipped. The returned rows
    /// are grouped oldest→newest within the page. For the next reverse page,
    /// pass `before_turn_seq = page.first().turn_seq`.
    pub async fn get_recent_turns_page(
        &self,
        session_id: &str,
        before_turn_seq: Option<i64>,
        limit: usize,
    ) -> anyhow::Result<Vec<FetchedTurn>> {
        let rows = sqlx::query(
            r#"
            WITH turn_starts AS (
                SELECT turn_id, MIN(id) AS turn_seq
                FROM events
                WHERE session_id = ?1 AND turn_id IS NOT NULL
                GROUP BY turn_id
                HAVING MIN(id) > COALESCE(
                    (SELECT cleared_after_id FROM session_context_boundaries WHERE session_id = ?1),
                    0)
            ),
            selected_turns AS (
                SELECT turn_id, turn_seq
                FROM turn_starts
                WHERE (?2 IS NULL OR turn_seq < ?2)
                ORDER BY turn_seq DESC
                LIMIT ?3
            )
            SELECT e.id, e.event_type, e.data, e.created_at, e.turn_id,
                   selected_turns.turn_seq, s.status
            FROM selected_turns
            JOIN events e
              ON e.session_id = ?1 AND e.turn_id = selected_turns.turn_id
            LEFT JOIN (
                SELECT te.turn_id,
                       json_extract(te.data, '$.status') AS status
                FROM events te
                WHERE te.session_id = ?1
                  AND te.turn_id IS NOT NULL
                  AND te.event_type = 'task_end'
                  AND te.id = (
                      SELECT MAX(te2.id)
                      FROM events te2
                      WHERE te2.session_id = ?1
                        AND te2.turn_id = te.turn_id
                        AND te2.event_type = 'task_end'
                  )
            ) s
              ON s.turn_id = selected_turns.turn_id
            WHERE e.event_type IN ('user_message','assistant_response','tool_result')
            ORDER BY selected_turns.turn_seq ASC, e.id ASC
            "#,
        )
        .bind(session_id)
        .bind(before_turn_seq)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        Ok(group_rows_into_turns(self.fetched_rows(session_id, rows)?))
    }

    /// Project turn-anchored query rows into [`FetchedRow`]s: hydrate each
    /// conversation row through `turn_from_event(...).into_message()` (turn_id
    /// flows through) and carry the per-turn `turn_seq` + latest terminal
    /// `status`. No `created_at` ordering anywhere; rows arrive pre-ordered by
    /// `(turn_seq, id)`.
    fn fetched_rows(
        &self,
        session_id: &str,
        rows: Vec<sqlx::sqlite::SqliteRow>,
    ) -> anyhow::Result<Vec<FetchedRow>> {
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id: i64 = row.get("id");
            let event_type: String = row.get("event_type");
            let data_str: String = row.get("data");
            let created_at_str: String = row.get("created_at");
            let turn_id: Option<String> = row.get("turn_id");
            let turn_seq: i64 = row.get("turn_seq");
            let status_str: Option<String> = row.get("status");

            let data: serde_json::Value = serde_json::from_str(&data_str)?;
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let Some(message) =
                crate::events::turn_from_event(id, session_id, &event_type, &data, created_at)
                    .map(|turn| turn.into_message())
            else {
                continue;
            };

            let terminal_status = status_str.as_deref().and_then(TaskStatus::from_str);

            out.push(FetchedRow {
                turn_id,
                turn_seq,
                terminal_status,
                message,
            });
        }
        Ok(out)
    }

    // =========================================================================
    // Read Operations - Specific Queries for Context
    // =========================================================================

    /// Get the most recent error for a session
    pub async fn get_last_error(&self, session_id: &str) -> anyhow::Result<Option<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND event_type = 'error'
            ORDER BY created_at DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let events = self.rows_to_events(rows)?;
        Ok(events.into_iter().next())
    }

    /// Get the current active task (TaskStart without matching TaskEnd).
    ///
    /// This intentionally has no wall-clock cutoff. Activity age is a stale
    /// task policy concern, not an identity concern: a long-running task must
    /// remain resumable until its matching terminal event exists.
    pub async fn get_active_task(&self, session_id: &str) -> anyhow::Result<Option<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT start.id, start.session_id, start.event_type, start.data,
                   start.created_at, start.consolidated_at, start.task_id,
                   start.tool_name, start.turn_id
            FROM events AS start
            WHERE start.session_id = ?
              AND start.event_type = 'task_start'
              AND start.task_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM events AS terminal
                  WHERE terminal.session_id = start.session_id
                    AND terminal.task_id = start.task_id
                    AND terminal.event_type = 'task_end'
              )
            ORDER BY start.id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(self.rows_to_events(rows)?.into_iter().next())
    }

    /// Latest cumulative policy-counter checkpoint for every daemon boot.
    /// Each boot emits cumulative values, so returning only its newest row is
    /// essential to avoid double counting.
    pub async fn latest_policy_metrics_by_boot(
        &self,
    ) -> anyhow::Result<Vec<crate::events::PolicyMetricsSnapshotData>> {
        let rows = sqlx::query(
            r#"
            SELECT e.data
            FROM events AS e
            JOIN (
                SELECT json_extract(data, '$.boot_id') AS boot_id, MAX(id) AS latest_id
                FROM events
                WHERE event_type = 'policy_metrics_snapshot'
                  AND json_valid(data)
                  AND json_extract(data, '$.boot_id') IS NOT NULL
                GROUP BY json_extract(data, '$.boot_id')
            ) AS latest ON latest.latest_id = e.id
            ORDER BY e.id DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                let data: String = row.get("data");
                Ok(serde_json::from_str(&data)?)
            })
            .collect()
    }

    /// Reconcile stale TaskStart events that never received a matching TaskEnd.
    ///
    /// This emits synthetic failed TaskEnd events so UI/DB task state self-heals
    /// even when an agent process died outside channel watchdog loops.
    pub async fn reconcile_stale_task_starts(
        &self,
        stale_after_secs: i64,
        batch_size: usize,
    ) -> anyhow::Result<u64> {
        let stale_after_secs = stale_after_secs.max(1);
        let cutoff = Utc::now() - Duration::seconds(stale_after_secs);
        let cutoff_str = cutoff.to_rfc3339();

        let rows = sqlx::query(
            r#"
            SELECT s.session_id AS session_id,
                   s.task_id AS task_id,
                   MIN(s.created_at) AS started_at
            FROM events s
            WHERE s.event_type = 'task_start'
              AND s.task_id IS NOT NULL
              AND s.created_at < ?
              AND NOT EXISTS (
                SELECT 1
                FROM events e
                WHERE e.session_id = s.session_id
                  AND e.task_id = s.task_id
                  AND e.event_type = 'task_end'
              )
              AND NOT EXISTS (
                SELECT 1
                FROM events a
                WHERE a.session_id = s.session_id
                  AND a.task_id = s.task_id
                  AND a.created_at >= ?
              )
            GROUP BY s.session_id, s.task_id
            ORDER BY MIN(s.created_at) ASC
            LIMIT ?
            "#,
        )
        .bind(&cutoff_str)
        .bind(&cutoff_str)
        .bind(batch_size.max(1) as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut reconciled = 0u64;
        for row in rows {
            let session_id: String = row.get("session_id");
            let task_id: String = row.get("task_id");
            let started_at_raw: String = row.get("started_at");

            // Re-check both terminal state and activity to avoid racing a live
            // task that emitted an event after the candidate query.
            let is_terminal_or_active = sqlx::query(
                r#"
                SELECT 1
                FROM events
                WHERE session_id = ?
                  AND task_id = ?
                  AND (event_type = 'task_end' OR created_at >= ?)
                LIMIT 1
                "#,
            )
            .bind(&session_id)
            .bind(&task_id)
            .bind(&cutoff_str)
            .fetch_optional(&self.pool)
            .await?
            .is_some();
            if is_terminal_or_active {
                continue;
            }

            let started_at = DateTime::parse_from_rfc3339(&started_at_raw)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(cutoff);
            let duration_secs = (Utc::now() - started_at).num_seconds().max(0) as u64;
            let stale_after_mins = (stale_after_secs / 60).max(1);

            let event = Event::new(
                session_id.clone(),
                EventType::TaskEnd,
                serde_json::to_value(TaskEndData {
                    task_id: task_id.clone(),
                    status: TaskStatus::Failed,
                    outcome: Some(crate::events::TaskOutcome::Failed),
                    duration_secs,
                    iterations: 0,
                    tool_calls_count: 0,
                    error: Some(format!(
                        "Auto-failed by watchdog after {} minute(s) without task_end",
                        stale_after_mins
                    )),
                    summary: Some("Recovered stale in-flight task".to_string()),
                    efficiency: None,
                    // Watchdog-synthesized TaskEnd has no in-process turn
                    // context; legacy/unscoped => None.
                    turn_id: None,
                    completion_proof: None,
                    harness_eval: None,
                })?,
            );
            self.append(event).await?;
            reconciled += 1;
            info!(
                session_id = %session_id,
                task_id = %task_id,
                duration_secs,
                "Reconciled stale task_start with synthetic task_end"
            );
        }

        Ok(reconciled)
    }

    /// Get recent tool calls for a session
    pub async fn get_recent_tool_calls(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Event>> {
        self.query_events_by_types(
            session_id,
            &[EventType::ToolCall, EventType::ToolResult],
            limit,
        )
        .await
    }

    /// Resolve the latest state of an opaque resource handle inside one
    /// session. The latest lifecycle event wins, so an invalidation prevents a
    /// stale locator from being used even when an older registration exists.
    pub async fn get_resource(
        &self,
        session_id: &str,
        resource_id: &str,
    ) -> anyhow::Result<Option<ResourceRegisteredData>> {
        let row = sqlx::query(
            r#"
            SELECT event_type, data
            FROM events
            WHERE session_id = ?
              AND event_type IN ('resource_registered', 'resource_invalidated')
              AND json_extract(data, '$.resource_id') = ?
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .bind(resource_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else {
            return Ok(None);
        };
        let event_type: String = row.get("event_type");
        if event_type == EventType::ResourceInvalidated.as_str() {
            return Ok(None);
        }
        let data: String = row.get("data");
        Ok(Some(serde_json::from_str(&data)?))
    }

    /// Return the latest durable result for an exact idempotency key.
    pub async fn get_tool_result_by_idempotency_key(
        &self,
        session_id: &str,
        idempotency_key: &str,
    ) -> anyhow::Result<Option<ToolResultData>> {
        let row = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE session_id = ?
              AND event_type = 'tool_result'
              AND json_extract(data, '$.receipt.idempotency_key') = ?
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .bind(idempotency_key)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let data: String = row.get("data");
        Ok(Some(serde_json::from_str(&data)?))
    }

    /// Whether execution was durably claimed for this key without a result.
    /// This is an indeterminate side effect after a crash and must be
    /// reconciled rather than blindly replayed.
    pub async fn has_unresolved_tool_call_for_idempotency_key(
        &self,
        session_id: &str,
        idempotency_key: &str,
    ) -> anyhow::Result<bool> {
        let row = sqlx::query(
            r#"
            SELECT 1
            FROM events
            WHERE session_id = ?
              AND event_type = 'tool_call'
              AND json_extract(data, '$.idempotency_key') = ?
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .bind(idempotency_key)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.is_some())
    }

    pub async fn get_tool_stats(
        &self,
        tool_name: &str,
        since: DateTime<Utc>,
    ) -> anyhow::Result<ToolStats> {
        let since_str = since.to_rfc3339();
        let rows = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE event_type = 'tool_result'
              AND tool_name = ?
              AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 500
            "#,
        )
        .bind(tool_name)
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        let mut total_calls = 0u64;
        let mut successful = 0u64;
        let mut failed = 0u64;
        let mut duration_sum_ms: u128 = 0;
        let mut error_counts: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        for row in rows {
            let data_str: String = row.get("data");
            let tr: ToolResultData = match serde_json::from_str(&data_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if is_synthetic_tool_result(&tr) {
                continue;
            }

            total_calls += 1;
            duration_sum_ms += tr.duration_ms as u128;
            if tr.success {
                successful += 1;
                continue;
            }
            failed += 1;

            let raw_error = tr.error.as_deref().unwrap_or(&tr.result);
            let normalized = normalize_tool_error_text(raw_error);
            let pattern = crate::memory::procedures::extract_error_pattern(&normalized);
            if !pattern.trim().is_empty() {
                *error_counts.entry(pattern).or_insert(0) += 1;
            }
        }

        let avg_duration_ms = if total_calls == 0 {
            0
        } else {
            (duration_sum_ms / total_calls as u128) as u64
        };

        let mut common_errors: Vec<(String, u64)> = error_counts.into_iter().collect();
        common_errors.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        common_errors.truncate(3);

        Ok(ToolStats {
            total_calls,
            successful,
            failed,
            avg_duration_ms,
            common_errors,
        })
    }

    /// Aggregate latency + token telemetry over recent `LlmCall` events.
    /// Powers the dashboard `/api/llm/latency` endpoint and ad-hoc diagnostics.
    pub async fn get_llm_stats(&self, since: DateTime<Utc>) -> anyhow::Result<LlmStats> {
        let since_str = since.to_rfc3339();
        let rows = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE event_type = 'llm_call'
              AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 2000
            "#,
        )
        .bind(&since_str)
        .fetch_all(&self.pool)
        .await?;

        let mut latencies: Vec<u64> = Vec::with_capacity(rows.len());
        let mut latency_sum: u128 = 0;
        let mut input_sum: u128 = 0;
        let mut output_sum: u128 = 0;
        let mut fell_back_count = 0u64;
        let mut failed_calls = 0u64;

        for row in rows {
            let data_str: String = row.get("data");
            let call: LlmCallData = match serde_json::from_str(&data_str) {
                Ok(v) => v,
                Err(_) => continue,
            };
            latencies.push(call.latency_ms);
            latency_sum += call.latency_ms as u128;
            input_sum += call.input_tokens as u128;
            output_sum += call.output_tokens as u128;
            if call.fell_back {
                fell_back_count += 1;
            }
            if call.failed {
                failed_calls += 1;
            }
        }

        let total_calls = latencies.len() as u64;
        if total_calls == 0 {
            return Ok(LlmStats::default());
        }

        latencies.sort_unstable();
        let percentile = |sorted: &[u64], pct: f64| -> u64 {
            if sorted.is_empty() {
                return 0;
            }
            // Nearest-rank percentile.
            let rank = (pct / 100.0 * sorted.len() as f64).ceil() as usize;
            let idx = rank.saturating_sub(1).min(sorted.len() - 1);
            sorted[idx]
        };

        Ok(LlmStats {
            total_calls,
            failed_calls,
            avg_latency_ms: (latency_sum / total_calls as u128) as u64,
            p50_latency_ms: percentile(&latencies, 50.0),
            p95_latency_ms: percentile(&latencies, 95.0),
            max_latency_ms: *latencies.last().unwrap_or(&0),
            fell_back_count,
            avg_input_tokens: (input_sum / total_calls as u128) as u64,
            avg_output_tokens: (output_sum / total_calls as u128) as u64,
        })
    }

    /// Roll up `LlmCall` telemetry for a single task. Used by `self_diagnose`
    /// (Tier 1) and the per-turn efficiency signal at task end (Tier 2).
    pub async fn get_task_llm_stats(&self, task_id: &str) -> anyhow::Result<TaskLlmSummary> {
        let rows = sqlx::query(
            r#"
            SELECT data
            FROM events
            WHERE event_type = 'llm_call'
              AND task_id = ?
            ORDER BY created_at ASC
            LIMIT 2000
            "#,
        )
        .bind(task_id)
        .fetch_all(&self.pool)
        .await?;

        let mut summary = TaskLlmSummary::default();
        let mut latencies: Vec<u64> = Vec::with_capacity(rows.len());
        let mut latency_sum: u128 = 0;
        let mut max_latency = 0u64;

        for row in rows {
            let data_str: String = row.get("data");
            let call: LlmCallData = match serde_json::from_str(&data_str) {
                Ok(v) => v,
                Err(_) => continue,
            };
            summary.total_calls += 1;
            if call.failed {
                summary.failed_calls += 1;
                if call.error.is_some() {
                    summary.last_error = call.error.clone();
                }
            }
            summary.total_input_tokens += call.input_tokens as u64;
            summary.total_output_tokens += call.output_tokens as u64;
            if let Some(cached) = call.cached_input_tokens {
                summary.cached_input_token_samples += 1;
                summary.total_cached_input_tokens += cached as u64;
            }
            if let Some(created) = call.cache_creation_input_tokens {
                summary.cache_creation_input_token_samples += 1;
                summary.total_cache_creation_input_tokens += created as u64;
            }
            summary.total_attempts += call.attempts.max(1) as u64;
            if call.fell_back {
                summary.fell_back_count += 1;
            }
            if let Some(est) = call.est_input_tokens {
                summary.est_samples += 1;
                summary.total_est_input_tokens += est as u64;
                summary.actual_input_tokens_with_est += call.input_tokens as u64;
            }
            latencies.push(call.latency_ms);
            latency_sum += call.latency_ms as u128;
            if call.latency_ms >= max_latency {
                max_latency = call.latency_ms;
                summary.max_latency_iteration = call.iteration.unwrap_or(0);
            }
            // Last row wins (ASC order) so this is the final model used.
            summary.final_model = call.final_model.or(Some(call.model));
        }

        if summary.total_calls == 0 {
            return Ok(summary);
        }

        latencies.sort_unstable();
        let percentile = |sorted: &[u64], pct: f64| -> u64 {
            if sorted.is_empty() {
                return 0;
            }
            let rank = (pct / 100.0 * sorted.len() as f64).ceil() as usize;
            let idx = rank.saturating_sub(1).min(sorted.len() - 1);
            sorted[idx]
        };

        summary.avg_latency_ms = (latency_sum / summary.total_calls as u128) as u64;
        summary.p50_latency_ms = percentile(&latencies, 50.0);
        summary.p95_latency_ms = percentile(&latencies, 95.0);
        summary.max_latency_ms = max_latency;
        Ok(summary)
    }

    /// Get the last completed task for a session
    pub async fn get_last_completed_task(&self, session_id: &str) -> anyhow::Result<Option<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE session_id = ? AND event_type = 'task_end'
            ORDER BY created_at DESC
            LIMIT 1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        let events = self.rows_to_events(rows)?;
        Ok(events.into_iter().next())
    }

    /// Query all events of a single type in [start, end).
    pub async fn query_events_by_type_between(
        &self,
        event_type: EventType,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<Vec<Event>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, event_type, data, created_at, consolidated_at, task_id, tool_name, turn_id
            FROM events
            WHERE event_type = ? AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC
            "#,
        )
        .bind(event_type.as_str())
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(&self.pool)
        .await?;
        self.rows_to_events(rows)
    }

    /// Return the earliest created_at for an event type.
    pub async fn earliest_event_time_by_type(
        &self,
        event_type: EventType,
    ) -> anyhow::Result<Option<DateTime<Utc>>> {
        let row = sqlx::query(
            r#"
            SELECT created_at
            FROM events
            WHERE event_type = ?
            ORDER BY created_at ASC
            LIMIT 1
            "#,
        )
        .bind(event_type.as_str())
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let raw: String = row.get("created_at");
        let parsed = DateTime::parse_from_rfc3339(&raw)
            .map(|dt| dt.with_timezone(&Utc))
            .ok();
        Ok(parsed)
    }

    /// Build a graduation report for policy routing gate checks.
    pub async fn policy_graduation_report(
        &self,
        window_days: u32,
    ) -> anyhow::Result<PolicyGraduationReport> {
        let now = Utc::now();
        let window = Duration::days(window_days as i64);
        let start_current = now - window;
        let start_previous = start_current - window;

        let decisions = self
            .query_events_by_type_between(EventType::PolicyDecision, start_current, now)
            .await?;
        let mut total_decisions = 0u64;
        let mut diverged_decisions = 0u64;
        for event in decisions {
            if let Ok(data) = event.parse_data::<PolicyDecisionData>() {
                total_decisions += 1;
                if data.diverged {
                    diverged_decisions += 1;
                }
            }
        }
        let divergence_rate = if total_decisions > 0 {
            diverged_decisions as f64 / total_decisions as f64
        } else {
            0.0
        };

        let current = self
            .task_window_stats(start_current, now)
            .await
            .unwrap_or_default();
        let previous = self
            .task_window_stats(start_previous, start_current)
            .await
            .unwrap_or_default();

        let observed_days = match self
            .earliest_event_time_by_type(EventType::PolicyDecision)
            .await?
        {
            Some(first) => (now - first).num_seconds().max(0) as f64 / 86_400.0,
            None => 0.0,
        };

        Ok(PolicyGraduationReport {
            window_days,
            observed_days,
            total_decisions,
            diverged_decisions,
            divergence_rate,
            current,
            previous,
        })
    }

    /// Return canonical write-path consistency metrics from the event stream.
    ///
    /// `messages` was intentionally removed when events became canonical. The
    /// projection measured here is therefore the stable `message_id` identity
    /// carried by each conversation event. A non-zero delta identifies missing
    /// or duplicate identities without reviving the obsolete dual-write path.
    pub async fn write_consistency_report(
        &self,
        top_n_sessions: usize,
    ) -> anyhow::Result<WriteConsistencyReport> {
        let conversation_event_rows: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let stable_message_rows: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM (
                SELECT session_id,
                       CAST(json_extract(data, '$.message_id') AS TEXT) AS message_id
                FROM events
                WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
                  AND json_extract(data, '$.message_id') IS NOT NULL
                  AND TRIM(CAST(json_extract(data, '$.message_id') AS TEXT)) != ''
                GROUP BY session_id, message_id
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let session_mismatch_count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM (
                SELECT session_id,
                       COUNT(*) AS event_rows,
                       COUNT(DISTINCT CASE
                           WHEN json_extract(data, '$.message_id') IS NOT NULL
                            AND TRIM(CAST(json_extract(data, '$.message_id') AS TEXT)) != ''
                           THEN CAST(json_extract(data, '$.message_id') AS TEXT)
                       END) AS message_rows
                FROM events
                WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
                GROUP BY session_id
                HAVING event_rows != message_rows
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let top_session_drifts = if top_n_sessions == 0 {
            Vec::new()
        } else {
            sqlx::query(
                r#"
                SELECT session_id,
                       COUNT(*) AS event_rows,
                       COUNT(DISTINCT CASE
                           WHEN json_extract(data, '$.message_id') IS NOT NULL
                            AND TRIM(CAST(json_extract(data, '$.message_id') AS TEXT)) != ''
                           THEN CAST(json_extract(data, '$.message_id') AS TEXT)
                       END) AS message_rows
                FROM events
                WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
                GROUP BY session_id
                HAVING event_rows != message_rows
                ORDER BY ABS(event_rows - message_rows) DESC, session_id ASC
                LIMIT ?
                "#,
            )
            .bind(top_n_sessions as i64)
            .fetch_all(&self.pool)
            .await?
            .into_iter()
            .map(|row| {
                let message_rows = row.get::<i64, _>("message_rows");
                let event_rows = row.get::<i64, _>("event_rows");
                SessionWriteDrift {
                    session_id: row.get("session_id"),
                    message_rows: to_u64(message_rows),
                    event_rows: to_u64(event_rows),
                    delta: event_rows - message_rows,
                }
            })
            .collect()
        };

        let missing_message_id_events: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM events
            WHERE event_type IN ('user_message', 'assistant_response', 'tool_result')
              AND (
                json_extract(data, '$.message_id') IS NULL
                OR TRIM(CAST(json_extract(data, '$.message_id') AS TEXT)) = ''
              )
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        let stale_task_starts: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM (
                SELECT s.session_id, s.task_id
                FROM events s
                WHERE s.event_type = 'task_start'
                  AND s.task_id IS NOT NULL
                GROUP BY s.session_id, s.task_id
                HAVING NOT EXISTS (
                    SELECT 1
                    FROM events e
                    WHERE e.session_id = s.session_id
                      AND e.task_id = s.task_id
                      AND e.event_type = 'task_end'
                )
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(0);

        Ok(WriteConsistencyReport {
            generated_at: Utc::now().to_rfc3339(),
            conversation_event_rows: to_u64(conversation_event_rows),
            missing_message_id_events: to_u64(missing_message_id_events),
            global_delta: conversation_event_rows - stable_message_rows,
            session_mismatch_count: to_u64(session_mismatch_count),
            stale_task_starts: to_u64(stale_task_starts),
            top_session_drifts,
        })
    }

    async fn task_window_stats(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> anyhow::Result<TaskWindowStats> {
        let task_ends = self
            .query_events_by_type_between(EventType::TaskEnd, start, end)
            .await?;
        let errors = self
            .query_events_by_type_between(EventType::Error, start, end)
            .await?;

        let mut stats = TaskWindowStats {
            total: task_ends.len() as u64,
            ..TaskWindowStats::default()
        };
        for event in task_ends {
            if let Ok(data) = event.parse_data::<TaskEndData>() {
                match data.status {
                    TaskStatus::Completed => stats.completed += 1,
                    TaskStatus::Failed => stats.failed += 1,
                    TaskStatus::Cancelled => stats.cancelled += 1,
                    TaskStatus::Interrupted => stats.interrupted += 1,
                }
                match data.effective_outcome() {
                    crate::events::TaskOutcome::Succeeded => stats.outcome_succeeded += 1,
                    crate::events::TaskOutcome::Partial => stats.outcome_partial += 1,
                    crate::events::TaskOutcome::Failed => stats.outcome_failed += 1,
                }
                let stalled = data
                    .error
                    .as_deref()
                    .map(|e| e.to_ascii_lowercase().contains("stalled"))
                    .unwrap_or(false)
                    || data
                        .summary
                        .as_deref()
                        .map(|s| s.to_ascii_lowercase().contains("stalled"))
                        .unwrap_or(false);
                if stalled {
                    stats.stalled += 1;
                }
            } else {
                stats.outcome_unknown += 1;
            }
        }
        stats.error_events = errors.len() as u64;

        if stats.total > 0 {
            stats.completion_rate = stats.completed as f64 / stats.total as f64;
            stats.error_rate = stats.error_events as f64 / stats.total as f64;
            stats.stall_rate = stats.stalled as f64 / stats.total as f64;
            let semantic_known = stats.total.saturating_sub(stats.outcome_unknown);
            if semantic_known > 0 {
                stats.semantic_success_rate =
                    stats.outcome_succeeded as f64 / semantic_known as f64;
            }
        }

        Ok(stats)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn rows_to_events(&self, rows: Vec<sqlx::sqlite::SqliteRow>) -> anyhow::Result<Vec<Event>> {
        let mut events = Vec::new();
        for row in rows {
            let id: i64 = row.get("id");
            let session_id: String = row.get("session_id");
            let event_type_str: String = row.get("event_type");
            let data_str: String = row.get("data");
            let created_at_str: String = row.get("created_at");
            let consolidated_at_str: Option<String> = row.get("consolidated_at");
            let task_id: Option<String> = row.get("task_id");
            let tool_name: Option<String> = row.get("tool_name");
            let turn_id: Option<String> = row.get("turn_id");

            let event_type = match EventType::from_str(&event_type_str) {
                Some(et) => et,
                None => {
                    warn!("Unknown event type: {}", event_type_str);
                    continue;
                }
            };

            let data: serde_json::Value = serde_json::from_str(&data_str)?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let consolidated_at = consolidated_at_str.and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc))
            });

            events.push(Event {
                id,
                session_id,
                event_type,
                data,
                created_at,
                consolidated_at,
                task_id,
                tool_name,
                turn_id,
            });
        }
        Ok(events)
    }
}

fn to_u64(value: i64) -> u64 {
    if value <= 0 {
        0
    } else {
        value as u64
    }
}

fn normalize_tool_error_text(raw: &str) -> std::borrow::Cow<'_, str> {
    crate::traits::extract_primary_message_content(raw, &[])
}

fn is_synthetic_tool_result(tr: &ToolResultData) -> bool {
    tr.success
        && tr.duration_ms == 0
        && tr.error.is_none()
        && crate::traits::message_content_is_structural_only(&tr.result, &tr.annotations)
}

/// Builder for emitting events with a consistent session context
pub struct EventEmitter {
    store: Arc<EventStore>,
    session_id: String,
    current_task_id: Option<String>,
}

impl EventEmitter {
    pub fn new(store: Arc<EventStore>, session_id: impl Into<String>) -> Self {
        Self {
            store,
            session_id: session_id.into(),
            current_task_id: None,
        }
    }

    pub fn with_task_id(mut self, task_id: impl Into<String>) -> Self {
        self.current_task_id = Some(task_id.into());
        self
    }

    pub fn set_task_id(&mut self, task_id: Option<String>) {
        self.current_task_id = task_id;
    }

    pub fn task_id(&self) -> Option<&str> {
        self.current_task_id.as_deref()
    }

    pub async fn session_context_boundary(&self) -> anyhow::Result<Option<i64>> {
        self.store.session_context_boundary(&self.session_id).await
    }

    /// Emit an event with the current context
    pub async fn emit<T: serde::Serialize>(
        &self,
        event_type: EventType,
        data: T,
    ) -> anyhow::Result<i64> {
        let mut json_data = serde_json::to_value(data)?;

        // Inject task_id if present and not already in data
        if let Some(task_id) = &self.current_task_id {
            if let Some(obj) = json_data.as_object_mut() {
                if !obj.contains_key("task_id") {
                    obj.insert("task_id".to_string(), serde_json::json!(task_id));
                }
            }
        }

        let event = Event::new(&self.session_id, event_type, json_data);
        self.store.append(event).await
    }

    /// Emit an `llm_call` event and correlated token projection as one durable
    /// transaction. This is intentionally specialized so unrelated event
    /// writers cannot accidentally couple themselves to the projection table.
    pub async fn emit_model_call_with_token_usage(
        &self,
        mut data: LlmCallData,
        usage: Option<&TokenUsage>,
        call_id: &str,
    ) -> anyhow::Result<i64> {
        if data.task_id.trim().is_empty() {
            if let Some(task_id) = &self.current_task_id {
                data.task_id = task_id.clone();
            }
        }
        let event = Event::new(
            &self.session_id,
            EventType::LlmCall,
            serde_json::to_value(data)?,
        );
        self.store
            .append_llm_call_with_token_usage(event, usage, call_id)
            .await
    }

    /// Get the underlying store
    pub fn store(&self) -> Arc<EventStore> {
        self.store.clone()
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use serde_json::json;

    async fn setup_store() -> (EventStore, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let db_url = format!("sqlite:{}", db_file.path().display());
        let pool = SqlitePool::connect(&db_url).await.expect("connect sqlite");
        let store = EventStore::new(pool).await.expect("init event store");
        (store, db_file)
    }

    #[tokio::test]
    async fn user_message_memory_projection_requires_explicit_allowed_stage() {
        let (store, _database) = setup_store().await;
        sqlx::query(
            "CREATE TABLE memory_spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                span_kind TEXT NOT NULL,
                source_event_id INTEGER,
                source_episode_id INTEGER,
                session_id TEXT,
                channel_id TEXT,
                role TEXT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                privacy TEXT NOT NULL DEFAULT 'global',
                observed_from TEXT,
                observed_to TEXT,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                deleted_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&store.pool)
        .await
        .unwrap();
        sqlx::query(
            "CREATE UNIQUE INDEX idx_memory_spans_source_event
             ON memory_spans(source_event_id) WHERE source_event_id IS NOT NULL",
        )
        .execute(&store.pool)
        .await
        .unwrap();
        let event_id = store
            .append(Event::new(
                "session-memory-boundary",
                EventType::UserMessage,
                json!({
                    "task_id": "task-memory-boundary",
                    "turn_id": "turn-memory-boundary",
                    "content": "Synthetic content that must stay outside automatic memory"
                }),
            ))
            .await
            .unwrap();

        let before_projection: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
                .bind(event_id)
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(
            before_projection, 0,
            "canonical append must not imply memory persistence"
        );

        store
            .append(Event::new(
                "session-memory-boundary",
                EventType::MemoryPolicyCompiled,
                json!({
                    "task_id": "task-memory-boundary",
                    "turn_id": "turn-memory-boundary",
                    "access": "suppressed",
                    "reason_code": "current_assessment_denies_user_memory",
                    "retrieval_suppressed": true,
                    "persistence_suppressed": true
                }),
            ))
            .await
            .unwrap();
        store
            .project_user_message_memory_span(event_id)
            .await
            .unwrap();

        let after_suppressed_projection: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM memory_spans WHERE source_event_id = ?")
                .bind(event_id)
                .fetch_one(&store.pool)
                .await
                .unwrap();
        assert_eq!(after_suppressed_projection, 0);
    }

    #[tokio::test]
    async fn generated_response_identity_is_bounded_by_dispatch_watermark() {
        let (store, _database) = setup_store().await;
        let content = "synthetic identical response";
        for (task_id, message_id) in [("task-old", "response-old")] {
            store
                .append(Event::new(
                    "session-a",
                    EventType::AssistantResponse,
                    json!({
                        "task_id": task_id,
                        "turn_id": "turn-old",
                        "message_id": message_id,
                        "content": content,
                        "model": "test",
                        "referenced_receipts": []
                    }),
                ))
                .await
                .unwrap();
        }
        let watermark = store.event_watermark().await.unwrap();
        store
            .append(Event::new(
                "session-a",
                EventType::AssistantResponse,
                json!({
                    "task_id": "task-current",
                    "turn_id": "turn-current",
                    "message_id": "response-current",
                    "content": content,
                    "model": "test",
                    "referenced_receipts": []
                }),
            ))
            .await
            .unwrap();

        let generated = store
            .generated_response_after("session-a", watermark, content)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(generated.response_id, "response-current");
        assert_eq!(generated.task_id, "task-current");
        assert_eq!(generated.turn_id.as_deref(), Some("turn-current"));
    }

    #[tokio::test]
    async fn completion_proof_references_only_graph_satisfying_receipts() {
        let (store, _database) = setup_store().await;
        for (call_id, obligations) in [
            ("receipt-incidental", json!([])),
            (
                "receipt-proof",
                json!(["task:synthetic-task/obligation:evidence:0"]),
            ),
        ] {
            store
                .append(Event::new(
                    "session-a",
                    EventType::ToolResult,
                    json!({
                        "task_id": "synthetic-task",
                        "tool_call_id": call_id,
                        "name": "read_file",
                        "result": "synthetic",
                        "success": true,
                        "duration_ms": 1,
                        "receipt": {
                            "schema_version": crate::events::ToolReceiptV1::SCHEMA_VERSION,
                            "result_provenance": {"result_id": format!("result:{call_id}")},
                            "completion_obligation_ids": obligations
                        }
                    }),
                ))
                .await
                .unwrap();
        }

        let proof = store
            .task_completion_proof_references("synthetic-task")
            .await
            .unwrap();
        assert_eq!(proof.len(), 1);
        assert_eq!(proof[0].receipt_id, "receipt-proof");
        assert_eq!(
            proof[0].obligation_ids,
            ["task:synthetic-task/obligation:evidence:0"]
        );
    }

    #[tokio::test]
    async fn continuation_evidence_requires_exact_task_call_and_result_identity() {
        let (store, _database) = setup_store().await;
        let call = ToolCallData {
            tool_call_id: "call-exact".to_string(),
            name: "terminal".to_string(),
            arguments: json!({"command": "/usr/bin/false", "working_dir": "/tmp"}),
            summary: None,
            task_id: Some("task-parent".to_string()),
            idempotency_key: None,
            policy_rev: None,
            risk_score: None,
            turn_id: None,
        };
        store
            .append(Event::new(
                "session-a",
                EventType::ToolCall,
                serde_json::to_value(&call).unwrap(),
            ))
            .await
            .unwrap();
        let metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult),
            exit_code: Some(1),
            semantics: crate::tools::command_semantics::classify_shell_command("/usr/bin/false"),
            ..crate::traits::ToolCallMetadata::default()
        };
        let mut receipt = crate::events::ToolReceiptV1::from_metadata(
            &metadata,
            crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult,
            crate::events::ToolOutcomeEvidenceSource::StructuredMetadata,
            None,
        );
        receipt.result_provenance.result_id = Some("result-exact".to_string());
        let result = ToolResultData {
            message_id: None,
            tool_call_id: "call-exact".to_string(),
            name: "terminal".to_string(),
            result: String::new(),
            success: true,
            duration_ms: 1,
            error: None,
            task_id: Some("task-parent".to_string()),
            annotations: Vec::new(),
            turn_id: None,
            attachments: Vec::new(),
            receipt: Some(receipt),
        };
        store
            .append(Event::new(
                "session-a",
                EventType::ToolResult,
                serde_json::to_value(&result).unwrap(),
            ))
            .await
            .unwrap();

        let found = store
            .continuation_tool_evidence("session-a", "task-parent", "call-exact", "result-exact")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(found.call.arguments["command"], "/usr/bin/false");
        assert_eq!(found.result.receipt.unwrap().exit_code, Some(1));
        assert!(store
            .continuation_tool_evidence("session-a", "task-parent", "call-exact", "result-other",)
            .await
            .unwrap()
            .is_none());
        assert!(store
            .continuation_tool_evidence("session-a", "task-other", "call-exact", "result-exact",)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn append_persists_and_reads_back_turn_id() {
        let (store, _db) = setup_store().await;
        let data = serde_json::json!({"content": "hi", "turn_id": "turn-abc"});
        let ev = Event::new("sess-1", EventType::UserMessage, data);
        assert_eq!(
            ev.turn_id.as_deref(),
            Some("turn-abc"),
            "Event::new extracts turn_id from data"
        );
        let id = store.append(ev).await.unwrap();
        let rows = store
            .query_events("sess-1", Utc::now() - Duration::days(1))
            .await
            .unwrap();
        let got = rows.iter().find(|e| e.id == id).unwrap();
        assert_eq!(got.turn_id.as_deref(), Some("turn-abc"));
    }

    #[tokio::test]
    async fn append_turn_id_null_when_absent() {
        let (store, _db) = setup_store().await;
        let ev = Event::new(
            "sess-1",
            EventType::UserMessage,
            serde_json::json!({"content": "hi"}),
        );
        assert!(ev.turn_id.is_none());
        let id = store.append(ev).await.unwrap();
        let rows = store
            .query_events("sess-1", Utc::now() - Duration::days(1))
            .await
            .unwrap();
        assert!(rows.iter().find(|e| e.id == id).unwrap().turn_id.is_none());
    }

    #[tokio::test]
    async fn model_call_and_token_projection_commit_atomically() {
        let (store, _db) = setup_store().await;
        sqlx::query(
            "CREATE TABLE token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cached_input_tokens INTEGER,
                cache_creation_input_tokens INTEGER,
                call_id TEXT,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&store.pool)
        .await
        .expect("token projection table");

        let call_id = "synthetic-call-atomic";
        let event = Event::new(
            "synthetic-session",
            EventType::LlmCall,
            json!({
                "call_id": call_id,
                "task_id": "synthetic-task",
                "model": "synthetic-model",
                "latency_ms": 1,
                "token_usage_present": true
            }),
        );
        let usage = TokenUsage {
            input_tokens: 11,
            output_tokens: 7,
            model: "synthetic-model".to_string(),
            ..TokenUsage::default()
        };

        store
            .append_llm_call_with_token_usage(event, Some(&usage), call_id)
            .await
            .expect("atomic telemetry append");

        let event_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE event_type = 'llm_call'")
                .fetch_one(&store.pool)
                .await
                .expect("event count");
        let token_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM token_usage WHERE call_id = ?")
                .bind(call_id)
                .fetch_one(&store.pool)
                .await
                .expect("token count");
        assert_eq!((event_count, token_count), (1, 1));
    }

    #[tokio::test]
    async fn model_call_event_rolls_back_when_token_projection_fails() {
        let (store, _db) = setup_store().await;
        let call_id = "synthetic-call-rollback";
        let event = Event::new(
            "synthetic-session",
            EventType::LlmCall,
            json!({
                "call_id": call_id,
                "task_id": "synthetic-task",
                "model": "synthetic-model",
                "latency_ms": 1,
                "token_usage_present": true
            }),
        );
        let usage = TokenUsage {
            input_tokens: 3,
            output_tokens: 2,
            model: "synthetic-model".to_string(),
            ..TokenUsage::default()
        };

        assert!(store
            .append_llm_call_with_token_usage(event, Some(&usage), call_id)
            .await
            .is_err());
        let event_count: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM events WHERE event_type = 'llm_call'")
                .fetch_one(&store.pool)
                .await
                .expect("event count");
        assert_eq!(event_count, 0, "failed projection must roll back event");
    }

    async fn append_event_at(
        store: &EventStore,
        session_id: &str,
        event_type: EventType,
        data: serde_json::Value,
        created_at: DateTime<Utc>,
    ) {
        let mut event = Event::new(session_id, event_type, data);
        event.created_at = created_at;
        store.append(event).await.expect("append event");
    }

    async fn append_policy_decision(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        diverged: bool,
        created_at: DateTime<Utc>,
    ) {
        let payload = PolicyDecisionData {
            task_id: task_id.to_string(),
            old_model: "old-model".to_string(),
            new_model: "new-model".to_string(),
            old_tier: "primary".to_string(),
            new_profile: "balanced".to_string(),
            diverged,
            policy_enforce: false,
            risk_score: 0.3,
            uncertainty_score: 0.2,
        };
        append_event_at(
            store,
            session_id,
            EventType::PolicyDecision,
            serde_json::to_value(payload).expect("serialize policy decision"),
            created_at,
        )
        .await;
    }

    async fn append_task_end(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        status: TaskStatus,
        created_at: DateTime<Utc>,
        error: Option<&str>,
        summary: Option<&str>,
    ) {
        let payload = TaskEndData {
            task_id: task_id.to_string(),
            status,
            outcome: Some(match status {
                TaskStatus::Completed => crate::events::TaskOutcome::Succeeded,
                TaskStatus::Cancelled | TaskStatus::Failed => crate::events::TaskOutcome::Failed,
                TaskStatus::Interrupted => crate::events::TaskOutcome::Partial,
            }),
            duration_secs: 1,
            iterations: 1,
            tool_calls_count: 0,
            error: error.map(str::to_string),
            summary: summary.map(str::to_string),
            efficiency: None,
            turn_id: None,
            completion_proof: None,
            harness_eval: None,
        };
        append_event_at(
            store,
            session_id,
            EventType::TaskEnd,
            serde_json::to_value(payload).expect("serialize task end"),
            created_at,
        )
        .await;
    }

    async fn append_task_start(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        created_at: DateTime<Utc>,
    ) {
        append_event_at(
            store,
            session_id,
            EventType::TaskStart,
            json!({
                "task_id": task_id,
                "description": format!("task {}", task_id)
            }),
            created_at,
        )
        .await;
    }

    async fn append_decision_point(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        created_at: DateTime<Utc>,
    ) {
        append_event_at(
            store,
            session_id,
            EventType::DecisionPoint,
            json!({
                "decision_type":"intent_gate",
                "task_id": task_id,
                "iteration": 1,
                "metadata":{"needs_tools":true},
                "summary":"intent gate forced tool mode"
            }),
            created_at,
        )
        .await;
    }

    struct ToolResultFixture<'a> {
        tool: &'a str,
        success: bool,
        duration_ms: u64,
        result: &'a str,
        error: Option<&'a str>,
        created_at: DateTime<Utc>,
    }

    async fn append_tool_result(
        store: &EventStore,
        session_id: &str,
        fixture: ToolResultFixture<'_>,
    ) {
        let mut payload = json!({
            "tool_call_id": format!(
                "tc-{}-{}",
                fixture.tool,
                fixture.created_at.timestamp_nanos_opt().unwrap_or(0)
            ),
            "name": fixture.tool,
            "result": fixture.result,
            "success": fixture.success,
            "duration_ms": fixture.duration_ms,
        });
        if let Some(err) = fixture.error {
            payload["error"] = json!(err);
        }
        append_event_at(
            store,
            session_id,
            EventType::ToolResult,
            payload,
            fixture.created_at,
        )
        .await;
    }

    #[tokio::test]
    async fn graduation_report_passes_with_low_divergence_and_no_regression() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-pass";

        // Ensure observed_days >= 7
        append_policy_decision(&store, session, "old-task", false, now - Duration::days(8)).await;
        for i in 0..20 {
            append_policy_decision(
                &store,
                session,
                &format!("cur-{i}"),
                false,
                now - Duration::hours(6) + Duration::minutes(i as i64),
            )
            .await;
        }

        // Previous window: weaker quality
        append_task_end(
            &store,
            session,
            "prev-1",
            TaskStatus::Completed,
            now - Duration::days(10),
            None,
            Some("completed"),
        )
        .await;
        append_task_end(
            &store,
            session,
            "prev-2",
            TaskStatus::Failed,
            now - Duration::days(9),
            Some("stalled waiting for output"),
            Some("stalled"),
        )
        .await;
        append_event_at(
            &store,
            session,
            EventType::Error,
            json!({"message":"previous error"}),
            now - Duration::days(9),
        )
        .await;

        // Current window: improved quality
        append_task_end(
            &store,
            session,
            "cur-1",
            TaskStatus::Completed,
            now - Duration::days(2),
            None,
            Some("done"),
        )
        .await;
        append_task_end(
            &store,
            session,
            "cur-2",
            TaskStatus::Completed,
            now - Duration::days(1),
            None,
            Some("done"),
        )
        .await;

        let report = store.policy_graduation_report(7).await.expect("report");
        assert!(report.observed_days >= 7.0);
        assert_eq!(report.total_decisions, 20);
        assert_eq!(report.diverged_decisions, 0);
        assert!(report.gate_passes(0.05));
        assert!(report.current.completion_rate >= report.previous.completion_rate);
        assert!(report.current.error_rate <= report.previous.error_rate);
        assert!(report.current.stall_rate <= report.previous.stall_rate);
    }

    #[tokio::test]
    async fn graduation_report_fails_when_divergence_exceeds_threshold() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-diverge";

        append_policy_decision(&store, session, "old-task", false, now - Duration::days(8)).await;
        for i in 0..20 {
            append_policy_decision(
                &store,
                session,
                &format!("cur-{i}"),
                i < 2,
                now - Duration::hours(3) + Duration::minutes(i as i64),
            )
            .await;
        }

        // Keep quality metrics equal so divergence is the failing reason.
        append_task_end(
            &store,
            session,
            "prev-1",
            TaskStatus::Completed,
            now - Duration::days(9),
            None,
            Some("done"),
        )
        .await;
        append_task_end(
            &store,
            session,
            "cur-1",
            TaskStatus::Completed,
            now - Duration::days(1),
            None,
            Some("done"),
        )
        .await;

        let report = store.policy_graduation_report(7).await.expect("report");
        assert!(report.observed_days >= 7.0);
        assert!(report.divergence_rate > 0.05);
        assert!(!report.gate_passes(0.05));
    }

    #[tokio::test]
    async fn graduation_report_fails_when_observation_window_is_too_short() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-short-window";

        // Earliest policy decision is only 2 days old.
        for i in 0..8 {
            append_policy_decision(
                &store,
                session,
                &format!("cur-{i}"),
                false,
                now - Duration::days(2) + Duration::hours(i as i64),
            )
            .await;
        }

        append_task_end(
            &store,
            session,
            "cur-1",
            TaskStatus::Completed,
            now - Duration::hours(12),
            None,
            Some("done"),
        )
        .await;

        let report = store.policy_graduation_report(7).await.expect("report");
        assert!(report.observed_days < 7.0);
        assert!(!report.gate_passes(0.05));
    }

    #[tokio::test]
    async fn query_recent_task_ends_and_decision_points_are_session_scoped() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_task_end(
            &store,
            "s1",
            "task-failed",
            TaskStatus::Failed,
            now - Duration::minutes(2),
            Some("boom"),
            None,
        )
        .await;
        append_task_end(
            &store,
            "s1",
            "task-ok",
            TaskStatus::Completed,
            now - Duration::minutes(1),
            None,
            Some("ok"),
        )
        .await;
        append_task_end(
            &store,
            "s2",
            "task-s2",
            TaskStatus::Failed,
            now - Duration::minutes(1),
            Some("other"),
            None,
        )
        .await;
        append_decision_point(&store, "s1", "task-failed", now - Duration::minutes(2)).await;
        append_decision_point(&store, "s2", "task-failed", now - Duration::minutes(2)).await;

        let s1_failed = store
            .query_recent_task_ends("s1", true, 10)
            .await
            .expect("query failed");
        assert_eq!(s1_failed.len(), 1);
        assert_eq!(s1_failed[0].session_id, "s1");

        let s1_decisions = store
            .query_decision_points("s1", "task-failed")
            .await
            .expect("query decision points");
        assert_eq!(s1_decisions.len(), 1);
        assert_eq!(s1_decisions[0].session_id, "s1");
    }

    #[tokio::test]
    async fn query_recent_intent_gate_decision_points_filters_and_scopes() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_decision_point(&store, "s1", "task-1", now - Duration::minutes(3)).await;
        append_event_at(
            &store,
            "s1",
            EventType::DecisionPoint,
            json!({
                "decision_type":"stopping_condition",
                "task_id":"task-1",
                "iteration":2,
                "metadata":{"reason":"stall"},
                "summary":"stopping condition fired"
            }),
            now - Duration::minutes(2),
        )
        .await;
        append_decision_point(&store, "s2", "task-2", now - Duration::minutes(1)).await;

        let s1_recent = store
            .query_recent_intent_gate_decision_points("s1", 10)
            .await
            .expect("query recent intent gate decision points");
        assert_eq!(s1_recent.len(), 1);
        assert_eq!(s1_recent[0].session_id, "s1");
        let parsed = s1_recent[0]
            .parse_data::<DecisionPointData>()
            .expect("parse decision point");
        assert_eq!(parsed.decision_type, DecisionType::IntentGate);
    }

    #[tokio::test]
    async fn reconcile_stale_task_starts_appends_failed_task_end() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        // Stale task with no task_end -> should be reconciled.
        append_task_start(
            &store,
            "s-reconcile",
            "task-stale",
            now - Duration::minutes(10),
        )
        .await;

        // Stale task with task_end already present -> should be ignored.
        append_task_start(
            &store,
            "s-reconcile",
            "task-complete",
            now - Duration::minutes(10),
        )
        .await;
        append_task_end(
            &store,
            "s-reconcile",
            "task-complete",
            TaskStatus::Completed,
            now - Duration::minutes(9),
            None,
            Some("ok"),
        )
        .await;

        // Recent task start -> should remain active.
        append_task_start(
            &store,
            "s-reconcile",
            "task-recent",
            now - Duration::minutes(1),
        )
        .await;

        // An old task that is still emitting events is active, not stale. This
        // covers long-running tools and detached continuation turns.
        append_task_start(
            &store,
            "s-reconcile",
            "task-active",
            now - Duration::minutes(10),
        )
        .await;
        append_decision_point(
            &store,
            "s-reconcile",
            "task-active",
            now - Duration::minutes(1),
        )
        .await;

        let reconciled = store
            .reconcile_stale_task_starts(300, 10)
            .await
            .expect("reconcile stale starts");
        assert_eq!(reconciled, 1);

        let stale_events = store
            .query_task_events_for_session("s-reconcile", "task-stale")
            .await
            .expect("query stale task events");
        assert_eq!(stale_events.len(), 2, "task-stale should have start+end");
        assert_eq!(stale_events[1].event_type, EventType::TaskEnd);
        let stale_end = stale_events[1]
            .parse_data::<TaskEndData>()
            .expect("parse stale task_end");
        assert_eq!(stale_end.status, TaskStatus::Failed);
        assert!(
            stale_end
                .error
                .as_deref()
                .is_some_and(|e| e.contains("Auto-failed by watchdog")),
            "synthetic task_end should include watchdog reason"
        );

        let recent_events = store
            .query_task_events_for_session("s-reconcile", "task-recent")
            .await
            .expect("query recent task events");
        assert_eq!(recent_events.len(), 1, "recent task should stay open");

        let active_events = store
            .query_task_events_for_session("s-reconcile", "task-active")
            .await
            .expect("query active task events");
        assert_eq!(
            active_events.len(),
            2,
            "recent activity should keep an old task open"
        );

        // Running again should be idempotent.
        let reconciled_again = store
            .reconcile_stale_task_starts(300, 10)
            .await
            .expect("second reconcile");
        assert_eq!(reconciled_again, 0);
    }

    #[tokio::test]
    async fn conversation_history_preserves_tool_call_extra_content() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_event_at(
            &store,
            "s-extra",
            EventType::AssistantResponse,
            json!({
                "message_id": "assistant-msg-1",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "name": "run_command",
                    "arguments": { "command": "ls -la" },
                    "extra_content": { "thought_signature": "sig-123" }
                }],
                "model": "gemini-2.5-pro",
                "input_tokens": 12,
                "output_tokens": 3
            }),
            now,
        )
        .await;

        let history = store
            .get_conversation_history("s-extra", 10)
            .await
            .expect("conversation history");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "assistant");

        let tool_calls_json = history[0]
            .tool_calls_json
            .as_deref()
            .expect("assistant tool calls should exist");
        let tool_calls: Vec<crate::traits::ToolCall> =
            serde_json::from_str(tool_calls_json).expect("parse tool calls");
        assert_eq!(tool_calls.len(), 1);
        let extra = tool_calls[0]
            .extra_content
            .as_ref()
            .expect("extra_content should be preserved");
        assert_eq!(extra["thought_signature"], "sig-123");
    }

    #[tokio::test]
    async fn write_consistency_report_compares_events_with_stable_message_ids() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();

        append_event_at(
            &store,
            "s-no-messages",
            EventType::UserMessage,
            json!({
                "content": "hello from event stream",
                "message_id": "event-msg-1",
                "has_attachments": false
            }),
            now,
        )
        .await;

        let report = store
            .write_consistency_report(5)
            .await
            .expect("write consistency");
        assert_eq!(report.conversation_event_rows, 1);
        assert_eq!(report.missing_message_id_events, 0);
        assert_eq!(report.global_delta, 0);
        assert_eq!(report.session_mismatch_count, 0);
        assert!(report.top_session_drifts.is_empty());
        assert!(
            report.evaluate_gate().passed,
            "canonical events should pass with unique stable message IDs"
        );
    }

    #[tokio::test]
    async fn write_consistency_report_counts_missing_message_ids() {
        let (store, _db_file) = setup_store().await;
        append_event_at(
            &store,
            "s-drift",
            EventType::UserMessage,
            json!({
                "content": "hello from event stream",
                "message_id": null,
                "has_attachments": false
            }),
            Utc::now(),
        )
        .await;

        let report = store
            .write_consistency_report(5)
            .await
            .expect("write consistency");

        assert_eq!(report.conversation_event_rows, 1);
        assert_eq!(report.missing_message_id_events, 1);
        assert_eq!(report.global_delta, 1);
        assert_eq!(report.session_mismatch_count, 1);
        assert_eq!(report.top_session_drifts.len(), 1);
        assert_eq!(report.top_session_drifts[0].session_id, "s-drift");
        assert_eq!(report.top_session_drifts[0].message_rows, 0);
        assert_eq!(report.top_session_drifts[0].event_rows, 1);
        assert_eq!(report.top_session_drifts[0].delta, 1);
        assert!(
            !report.evaluate_gate().passed,
            "default gate should fail when event payloads are missing message_id"
        );
    }

    #[tokio::test]
    async fn write_consistency_report_detects_duplicate_stable_message_ids_and_honors_limit() {
        let (store, _db_file) = setup_store().await;
        for (session, message_id) in [
            ("s-duplicate-a", "same-a"),
            ("s-duplicate-a", "same-a"),
            ("s-duplicate-b", "same-b"),
            ("s-duplicate-b", "same-b"),
        ] {
            append_event_at(
                &store,
                session,
                EventType::UserMessage,
                json!({
                    "content": "synthetic event",
                    "message_id": message_id,
                    "has_attachments": false
                }),
                Utc::now(),
            )
            .await;
        }

        let report = store
            .write_consistency_report(1)
            .await
            .expect("write consistency");
        assert_eq!(report.conversation_event_rows, 4);
        assert_eq!(report.global_delta, 2);
        assert_eq!(report.session_mismatch_count, 2);
        assert_eq!(report.top_session_drifts.len(), 1);
    }

    #[tokio::test]
    async fn active_task_does_not_expire_after_one_hour() {
        let (store, _db_file) = setup_store().await;
        append_task_start(
            &store,
            "s-long-task",
            "task-long-running",
            Utc::now() - Duration::hours(3),
        )
        .await;

        let active = store
            .get_active_task("s-long-task")
            .await
            .expect("active task lookup")
            .expect("old unresolved task remains active");
        assert_eq!(active.task_id.as_deref(), Some("task-long-running"));
    }

    #[tokio::test]
    async fn policy_metric_history_keeps_only_latest_cumulative_snapshot_per_boot() {
        let (store, _db_file) = setup_store().await;
        for (boot_id, samples) in [("boot-a", 1_u64), ("boot-b", 4), ("boot-a", 7)] {
            let mut metrics = crate::agent::policy_metrics_snapshot();
            metrics.tool_exposure_samples = samples;
            append_event_at(
                &store,
                "synthetic-policy-session",
                EventType::PolicyMetricsSnapshot,
                serde_json::to_value(crate::events::PolicyMetricsSnapshotData {
                    schema_version: crate::events::PolicyMetricsSnapshotData::SCHEMA_VERSION,
                    boot_id: boot_id.to_string(),
                    metrics,
                })
                .expect("serialize policy snapshot"),
                Utc::now(),
            )
            .await;
        }

        let snapshots = store
            .latest_policy_metrics_by_boot()
            .await
            .expect("durable policy history");
        assert_eq!(snapshots.len(), 2);
        assert_eq!(
            snapshots
                .iter()
                .find(|snapshot| snapshot.boot_id == "boot-a")
                .expect("boot-a")
                .metrics
                .tool_exposure_samples,
            7
        );
    }

    #[tokio::test]
    async fn get_tool_stats_aggregates_and_groups_errors() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-tool-stats-1";

        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: true,
                duration_ms: 100,
                result: "ok",
                error: None,
                created_at: now - Duration::minutes(50),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: true,
                duration_ms: 300,
                result: "ok",
                error: None,
                created_at: now - Duration::minutes(40),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: false,
                duration_ms: 200,
                result: "Error: Connection timed out at /tmp/foo.rs:12:3",
                error: Some("Error: Connection timed out at /tmp/foo.rs:12:3"),
                created_at: now - Duration::minutes(30),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "terminal",
                success: false,
                duration_ms: 400,
                result: "Error: Connection timed out at /tmp/bar.rs:99:1",
                error: Some("Error: Connection timed out at /tmp/bar.rs:99:1"),
                created_at: now - Duration::minutes(20),
            },
        )
        .await;

        let stats = store
            .get_tool_stats("terminal", now - Duration::hours(24))
            .await
            .expect("tool stats");

        assert_eq!(stats.total_calls, 4);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 2);
        assert_eq!(stats.avg_duration_ms, 250);
        assert_eq!(stats.common_errors.len(), 1);
        assert_eq!(stats.common_errors[0].1, 2);
    }

    #[tokio::test]
    async fn get_tool_stats_excludes_synthetic_system_results() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-tool-stats-2";

        // Synthetic: success + duration 0 + no error + [SYSTEM] prefix.
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 0,
                result: "[SYSTEM] You have already called web_search 3 times.",
                error: None,
                created_at: now - Duration::minutes(10),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 0,
                result: "[SYSTEM] BLOCKED: repetitive tool call",
                error: None,
                created_at: now - Duration::minutes(9),
            },
        )
        .await;
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 0,
                result: "[SYSTEM] Before executing tools, briefly state what you understand...",
                error: None,
                created_at: now - Duration::minutes(8),
            },
        )
        .await;

        // Real execution result
        append_tool_result(
            &store,
            session,
            ToolResultFixture {
                tool: "web_search",
                success: true,
                duration_ms: 120,
                result: "some results",
                error: None,
                created_at: now - Duration::minutes(7),
            },
        )
        .await;

        let stats = store
            .get_tool_stats("web_search", now - Duration::hours(24))
            .await
            .expect("tool stats");

        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.successful, 1);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.avg_duration_ms, 120);
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_llm_call(
        store: &EventStore,
        session_id: &str,
        task_id: &str,
        iteration: u32,
        latency_ms: u64,
        input_tokens: u32,
        est_input_tokens: Option<u32>,
        fell_back: bool,
        attempts: u32,
        final_model: Option<&str>,
        cached_input_tokens: Option<u32>,
        cache_creation_input_tokens: Option<u32>,
        created_at: DateTime<Utc>,
    ) {
        let payload = LlmCallData {
            call_id: None,
            call_purpose: None,
            task_id: task_id.to_string(),
            iteration: Some(iteration),
            model: "primary-model".to_string(),
            final_model: final_model.map(str::to_string),
            fell_back,
            attempts,
            latency_ms,
            prompt_ms: None,
            decode_ms: None,
            input_tokens,
            output_tokens: 100,
            cached_input_tokens,
            cache_creation_input_tokens,
            fresh_input_tokens: cached_input_tokens
                .map(|cached| input_tokens.saturating_sub(cached)),
            est_input_tokens,
            tool_calls_count: 0,
            offered_tools: Vec::new(),
            chosen_tools: Vec::new(),
            build_ms: Some(5),
            prefix_hash_system: None,
            prefix_hash_pre_boundary: None,
            tool_defs_hash: None,
            session_summary_hash: None,
            tail_hash: None,
            prefix_hash_archived: None,
            boundary_pos: None,
            message_count: None,
            force_text: false,
            token_usage_present: true,
            failed: false,
            error: None,
        };
        append_event_at(
            store,
            session_id,
            EventType::LlmCall,
            serde_json::to_value(payload).expect("serialize llm call"),
            created_at,
        )
        .await;
    }

    #[tokio::test]
    async fn get_task_llm_stats_aggregates_latency_and_drift() {
        let (store, _db_file) = setup_store().await;
        let now = Utc::now();
        let session = "s-llm-task";
        let task = "task-llm-1";

        // Two calls in this task; second is slower, fell back, and retried.
        append_llm_call(
            &store,
            session,
            task,
            1,
            100,
            1000,
            Some(1200),
            false,
            1,
            None,
            Some(700),
            Some(50),
            now - Duration::minutes(2),
        )
        .await;
        append_llm_call(
            &store,
            session,
            task,
            2,
            500,
            2000,
            Some(2000),
            true,
            3,
            Some("fallback-model"),
            Some(1000),
            None,
            now - Duration::minutes(1),
        )
        .await;
        // A call belonging to a different task must be excluded.
        append_llm_call(
            &store,
            session,
            "other-task",
            1,
            9999,
            50,
            None,
            false,
            1,
            None,
            Some(49),
            None,
            now - Duration::seconds(30),
        )
        .await;

        let s = store
            .get_task_llm_stats(task)
            .await
            .expect("task llm stats");

        assert_eq!(s.total_calls, 2);
        assert_eq!(s.total_input_tokens, 3000);
        assert_eq!(s.total_cached_input_tokens, 1700);
        assert_eq!(s.cached_input_token_samples, 2);
        assert_eq!(s.total_cache_creation_input_tokens, 50);
        assert_eq!(s.cache_creation_input_token_samples, 1);
        assert_eq!(s.total_attempts, 4);
        assert_eq!(s.fell_back_count, 1);
        assert_eq!(s.max_latency_ms, 500);
        assert_eq!(s.max_latency_iteration, 2);
        assert_eq!(s.est_samples, 2);
        // est 1200+2000=3200 vs actual 1000+2000=3000 ⇒ over-estimated by 200.
        assert_eq!(s.total_est_input_tokens, 3200);
        assert_eq!(s.actual_input_tokens_with_est, 3000);
        assert_eq!(s.est_input_drift(), 200);
        assert_eq!(s.final_model.as_deref(), Some("fallback-model"));
        // Fallback + retries ⇒ flagged inefficient.
        assert!(s.is_inefficient());
    }

    #[tokio::test]
    async fn get_task_llm_stats_surfaces_failed_call_error() {
        let (store, _db_file) = setup_store().await;
        let task = "task-llm-failed";
        append_event_at(
            &store,
            "s-llm-failed",
            EventType::LlmCall,
            json!({
                "task_id": task,
                "iteration": 8,
                "model": "gpt-test",
                "final_model": "gpt-test",
                "attempts": 2,
                "latency_ms": 42,
                "failed": true,
                "error": "LLM request was malformed (400)"
            }),
            Utc::now(),
        )
        .await;

        let summary = store
            .get_task_llm_stats(task)
            .await
            .expect("failed-call stats");
        assert_eq!(summary.total_calls, 1);
        assert_eq!(summary.failed_calls, 1);
        assert_eq!(summary.total_attempts, 2);
        assert_eq!(
            summary.last_error.as_deref(),
            Some("LLM request was malformed (400)")
        );
        assert!(summary.is_inefficient());
    }

    #[test]
    fn task_llm_summary_healthy_turn_is_not_flagged() {
        let s = TaskLlmSummary {
            total_calls: 3,
            total_attempts: 3,
            fell_back_count: 0,
            est_samples: 3,
            total_est_input_tokens: 1050,
            actual_input_tokens_with_est: 1000,
            ..Default::default()
        };
        // 5% drift, no retries/fallbacks, light loop ⇒ healthy.
        assert_eq!(s.est_input_drift(), 50);
        assert!(!s.is_inefficient());
    }

    #[test]
    fn task_llm_summary_large_token_drift_is_flagged() {
        let s = TaskLlmSummary {
            total_calls: 2,
            total_attempts: 2,
            fell_back_count: 0,
            est_samples: 2,
            total_est_input_tokens: 2000,
            actual_input_tokens_with_est: 1000,
            ..Default::default()
        };
        // 100% over-estimate ⇒ flagged even without fallbacks/retries.
        assert!(s.is_inefficient());
    }

    #[test]
    fn write_consistency_gate_can_be_tuned_with_custom_thresholds() {
        let report = WriteConsistencyReport {
            generated_at: Utc::now().to_rfc3339(),
            conversation_event_rows: 10,
            missing_message_id_events: 1,
            global_delta: 2,
            session_mismatch_count: 1,
            stale_task_starts: 0,
            top_session_drifts: Vec::new(),
        };

        let strict = report.evaluate_gate_with(WriteConsistencyThresholds {
            max_abs_global_delta: 0,
            max_session_mismatch_count: 0,
            max_stale_task_starts: 0,
            max_missing_message_id_events: 0,
        });
        assert!(!strict.passed);
        assert!(!strict.reasons.is_empty());

        let relaxed = report.evaluate_gate_with(WriteConsistencyThresholds {
            max_abs_global_delta: 2,
            max_session_mismatch_count: 1,
            max_stale_task_starts: 0,
            max_missing_message_id_events: 1,
        });
        assert!(relaxed.passed);
    }
}

#[cfg(test)]
mod turn_anchored_tests {
    use super::*;
    use serde_json::json;

    /// Build an `EventStore` over a temp DB. The backing tempfile is leaked so
    /// the open pool keeps a valid file for the lifetime of the test without
    /// the caller having to thread a guard through every helper.
    async fn test_event_store() -> EventStore {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let path = db_file.into_temp_path().keep().expect("keep temp db path");
        let db_url = format!("sqlite:{}", path.display());
        let pool = SqlitePool::connect(&db_url).await.expect("connect sqlite");
        EventStore::new(pool).await.expect("init event store")
    }

    async fn append_user(store: &EventStore, session: &str, turn_id: &str, content: &str) {
        let ev = Event::new(
            session,
            EventType::UserMessage,
            json!({ "content": content, "turn_id": turn_id }),
        );
        store.append(ev).await.expect("append user_message");
    }

    async fn append_assistant(store: &EventStore, session: &str, turn_id: &str, content: &str) {
        let ev = Event::new(
            session,
            EventType::AssistantResponse,
            json!({ "content": content, "turn_id": turn_id }),
        );
        store.append(ev).await.expect("append assistant_response");
    }

    async fn append_tool(store: &EventStore, session: &str, turn_id: &str, content: &str) {
        let ev = Event::new(
            session,
            EventType::ToolResult,
            json!({
                "tool_call_id": format!("tc-{turn_id}-{content}"),
                "name": "terminal",
                "result": content,
                "success": true,
                "duration_ms": 1,
                "turn_id": turn_id,
            }),
        );
        store.append(ev).await.expect("append tool_result");
    }

    async fn append_legacy_user(store: &EventStore, session: &str, content: &str) {
        // Legacy row: no turn_id in data ⇒ turn_id column NULL.
        let ev = Event::new(
            session,
            EventType::UserMessage,
            json!({ "content": content }),
        );
        assert!(ev.turn_id.is_none());
        store.append(ev).await.expect("append legacy user_message");
    }

    async fn append_task_end(store: &EventStore, session: &str, turn_id: &str, status: &str) {
        let ev = Event::new(
            session,
            EventType::TaskEnd,
            json!({ "status": status, "turn_id": turn_id }),
        );
        store.append(ev).await.expect("append task_end");
    }

    #[tokio::test]
    async fn turn_anchored_fetch_orders_by_turn_then_msg_seq() {
        let store = test_event_store().await;
        // Turn A: user(id1) assistant(id2). Turn B: user(id3) tool(id4).
        append_user(&store, "sess", "turn-A", "a-user").await;
        append_assistant(&store, "sess", "turn-A", "a-asst").await;
        append_user(&store, "sess", "turn-B", "b-user").await;
        append_tool(&store, "sess", "turn-B", "b-tool").await;
        let turns = store.get_turns_from_anchor("sess", 0).await.unwrap();
        // Two whole turns, in turn_seq order; within each, msg_seq (id) order.
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].turn_id.as_deref(), Some("turn-A"));
        assert_eq!(
            turns[0]
                .messages
                .iter()
                .map(|m| m.role.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "assistant"]
        );
        assert_eq!(turns[1].turn_id.as_deref(), Some("turn-B"));
    }

    #[tokio::test]
    async fn turn_anchored_fetch_late_write_sorts_last_within_its_turn() {
        let store = test_event_store().await;
        append_user(&store, "sess", "turn-A", "a-user").await; // id1
        append_assistant(&store, "sess", "turn-A", "a-asst").await; // id2
        append_user(&store, "sess", "turn-B", "b-user").await; // id3
                                                               // Late write under already-finished turn-A (id4 > id3) — a background notifier.
        append_tool(&store, "sess", "turn-A", "late-tool").await; // id4
        let turns = store.get_turns_from_anchor("sess", 0).await.unwrap();
        // turn-A's turn_seq is MIN(id)=1 < turn-B's 3, so turn-A still sorts first;
        // the late tool (id4) sorts LAST inside turn-A by msg_seq.
        assert_eq!(turns[0].turn_id.as_deref(), Some("turn-A"));
        assert_eq!(
            turns[0].messages.last().unwrap().content.as_deref(),
            Some("late-tool")
        );
        assert_eq!(turns[1].turn_id.as_deref(), Some("turn-B"));
    }

    #[tokio::test]
    async fn turn_anchored_fetch_respects_anchor_floor_and_excludes_legacy_null() {
        let store = test_event_store().await;
        append_legacy_user(&store, "sess", "legacy").await; // turn_id NULL
        append_user(&store, "sess", "turn-A", "a-user").await; // turn_seq = its id
        append_user(&store, "sess", "turn-B", "b-user").await;
        let all = store.get_turns_from_anchor("sess", 0).await.unwrap();
        // Legacy NULL-turn rows are excluded from reconstruction (covered by summary).
        assert!(all.iter().all(|t| t.turn_id.is_some()));
        // Anchor at turn-B's turn_seq drops turn-A.
        let b_seq = all
            .iter()
            .find(|t| t.turn_id.as_deref() == Some("turn-B"))
            .unwrap()
            .turn_seq;
        let from_b = store.get_turns_from_anchor("sess", b_seq).await.unwrap();
        assert_eq!(from_b.len(), 1);
        assert_eq!(from_b[0].turn_id.as_deref(), Some("turn-B"));
    }

    #[tokio::test]
    async fn turn_seq_is_immutable_across_late_writes() {
        let store = test_event_store().await;
        append_user(&store, "sess", "turn-A", "a-user").await; // id1, turn_seq = 1
        append_assistant(&store, "sess", "turn-A", "a-asst").await;
        let before = store.get_turns_from_anchor("sess", 0).await.unwrap();
        let a_seq = before
            .iter()
            .find(|t| t.turn_id.as_deref() == Some("turn-A"))
            .unwrap()
            .turn_seq;
        append_tool(&store, "sess", "turn-A", "late-tool").await; // higher id, must not lower MIN
        let after = store.get_turns_from_anchor("sess", 0).await.unwrap();
        let a_seq2 = after
            .iter()
            .find(|t| t.turn_id.as_deref() == Some("turn-A"))
            .unwrap()
            .turn_seq;
        assert_eq!(
            a_seq, a_seq2,
            "turn_seq = MIN(id) is immutable; the anchor relies on this"
        );
    }

    #[tokio::test]
    async fn fetch_groups_turn_with_no_user_message() {
        // Scheduled/background turn: starts with a tool/assistant, no user_message.
        let store = test_event_store().await;
        append_assistant(&store, "sess", "turn-bg", "bg-asst").await;
        append_tool(&store, "sess", "turn-bg", "bg-tool").await;
        let turns = store.get_turns_from_anchor("sess", 0).await.unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].turn_id.as_deref(), Some("turn-bg"));
        assert!(
            turns[0].messages.iter().all(|m| m.role != "user"),
            "no synthesized user message"
        );
    }

    #[tokio::test]
    async fn fetch_carries_latest_terminal_status() {
        let store = test_event_store().await;
        append_user(&store, "sess", "turn-A", "a-user").await;
        append_task_end(&store, "sess", "turn-A", "failed").await; // earlier
        append_task_end(&store, "sess", "turn-A", "completed").await; // latest wins
        let turns = store.get_turns_from_anchor("sess", 0).await.unwrap();
        let a = turns
            .iter()
            .find(|t| t.turn_id.as_deref() == Some("turn-A"))
            .unwrap();
        assert_eq!(a.terminal_status, Some(TaskStatus::Completed));
    }

    #[tokio::test]
    async fn recent_turn_page_limits_turns_not_message_rows() {
        let store = test_event_store().await;
        // Newest turn has more messages than the page's turn limit. The page must
        // still return every row in that selected turn.
        append_user(&store, "sess", "turn-A", "a-user").await;
        append_user(&store, "sess", "turn-B", "b-user").await;
        append_assistant(&store, "sess", "turn-B", "b-a1").await;
        append_tool(&store, "sess", "turn-B", "b-t1").await;
        append_assistant(&store, "sess", "turn-B", "b-a2").await;

        let page1 = store.get_recent_turns_page("sess", None, 1).await.unwrap();
        assert_eq!(page1.len(), 1);
        assert_eq!(page1[0].turn_id.as_deref(), Some("turn-B"));
        assert_eq!(
            page1[0].messages.len(),
            4,
            "LIMIT applies to turns, not rows"
        );

        let page2 = store
            .get_recent_turns_page("sess", Some(page1[0].turn_seq), 1)
            .await
            .unwrap();
        assert_eq!(page2.len(), 1);
        assert_eq!(page2[0].turn_id.as_deref(), Some("turn-A"));
    }
}
