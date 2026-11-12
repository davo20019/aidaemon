//! Policy runtime metrics and bounded autotuning.
//!
//! Moved verbatim from `agent/mod.rs` (Phase 5 decoupling). No logic changes:
//! this is a pure relocation of the policy metrics counters and the
//! uncertainty-threshold autotune helpers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::sync::LazyLock as Lazy;

use crate::events::{HarnessEvalSnapshot, LlmPayloadInvalidMetric, PolicyMetricsData};

pub(in crate::agent) struct PolicyRuntimeMetrics {
    pub(in crate::agent) tool_exposure_samples: AtomicU64,
    pub(in crate::agent) tool_exposure_before_sum: AtomicU64,
    pub(in crate::agent) tool_exposure_after_sum: AtomicU64,
    pub(in crate::agent) tool_schema_contract_rejections_total: AtomicU64,
    pub(in crate::agent) ambiguity_detected_total: AtomicU64,
    pub(in crate::agent) uncertainty_clarify_total: AtomicU64,
    pub(in crate::agent) context_refresh_total: AtomicU64,
    pub(in crate::agent) escalation_total: AtomicU64,
    pub(in crate::agent) fallback_expansion_total: AtomicU64,
    pub(in crate::agent) response_direct_return_total: AtomicU64,
    pub(in crate::agent) response_fallthrough_total: AtomicU64,
    pub(in crate::agent) orchestration_route_clarification_required_total: AtomicU64,
    pub(in crate::agent) orchestration_route_tools_required_total: AtomicU64,
    pub(in crate::agent) orchestration_route_short_correction_direct_reply_total: AtomicU64,
    pub(in crate::agent) orchestration_route_acknowledgment_direct_reply_total: AtomicU64,
    pub(in crate::agent) orchestration_route_default_continue_total: AtomicU64,
    pub(in crate::agent) context_bleed_prevented_total: AtomicU64,
    pub(in crate::agent) context_mismatch_preflight_drop_total: AtomicU64,
    pub(in crate::agent) followup_mode_overrides_total: AtomicU64,
    pub(in crate::agent) cross_scope_blocked_total: AtomicU64,
    pub(in crate::agent) tokens_failed_tasks_total: AtomicU64,
    pub(in crate::agent) est_input_token_samples: AtomicU64,
    pub(in crate::agent) est_input_tokens_total: AtomicU64,
    pub(in crate::agent) est_msg_tokens_total: AtomicU64,
    pub(in crate::agent) est_tool_tokens_total: AtomicU64,
    pub(in crate::agent) est_tool_tokens_high_share_total: AtomicU64,
    pub(in crate::agent) est_tool_tokens_high_abs_total: AtomicU64,
    pub(in crate::agent) no_progress_iterations_total: AtomicU64,
    pub(in crate::agent) deferred_no_tool_forced_required_total: AtomicU64,
    pub(in crate::agent) deferred_no_tool_deferral_detected_total: AtomicU64,
    pub(in crate::agent) deferred_no_tool_model_switch_total: AtomicU64,
    pub(in crate::agent) deferred_no_tool_error_marker_total: AtomicU64,
    pub(in crate::agent) llm_payload_invalid_total: AtomicU64,
    pub(in crate::agent) harness_eval_tasks_total: AtomicU64,
    pub(in crate::agent) harness_eval_overall_milli_sum: AtomicU64,
}

impl PolicyRuntimeMetrics {
    const fn new() -> Self {
        Self {
            tool_exposure_samples: AtomicU64::new(0),
            tool_exposure_before_sum: AtomicU64::new(0),
            tool_exposure_after_sum: AtomicU64::new(0),
            tool_schema_contract_rejections_total: AtomicU64::new(0),
            ambiguity_detected_total: AtomicU64::new(0),
            uncertainty_clarify_total: AtomicU64::new(0),
            context_refresh_total: AtomicU64::new(0),
            escalation_total: AtomicU64::new(0),
            fallback_expansion_total: AtomicU64::new(0),
            response_direct_return_total: AtomicU64::new(0),
            response_fallthrough_total: AtomicU64::new(0),
            orchestration_route_clarification_required_total: AtomicU64::new(0),
            orchestration_route_tools_required_total: AtomicU64::new(0),
            orchestration_route_short_correction_direct_reply_total: AtomicU64::new(0),
            orchestration_route_acknowledgment_direct_reply_total: AtomicU64::new(0),
            orchestration_route_default_continue_total: AtomicU64::new(0),
            context_bleed_prevented_total: AtomicU64::new(0),
            context_mismatch_preflight_drop_total: AtomicU64::new(0),
            followup_mode_overrides_total: AtomicU64::new(0),
            cross_scope_blocked_total: AtomicU64::new(0),
            tokens_failed_tasks_total: AtomicU64::new(0),
            est_input_token_samples: AtomicU64::new(0),
            est_input_tokens_total: AtomicU64::new(0),
            est_msg_tokens_total: AtomicU64::new(0),
            est_tool_tokens_total: AtomicU64::new(0),
            est_tool_tokens_high_share_total: AtomicU64::new(0),
            est_tool_tokens_high_abs_total: AtomicU64::new(0),
            no_progress_iterations_total: AtomicU64::new(0),
            deferred_no_tool_forced_required_total: AtomicU64::new(0),
            deferred_no_tool_deferral_detected_total: AtomicU64::new(0),
            deferred_no_tool_model_switch_total: AtomicU64::new(0),
            deferred_no_tool_error_marker_total: AtomicU64::new(0),
            llm_payload_invalid_total: AtomicU64::new(0),
            harness_eval_tasks_total: AtomicU64::new(0),
            harness_eval_overall_milli_sum: AtomicU64::new(0),
        }
    }
}

pub(in crate::agent) static POLICY_METRICS: Lazy<PolicyRuntimeMetrics> =
    Lazy::new(PolicyRuntimeMetrics::new);
const MAX_LLM_PAYLOAD_INVALID_METRIC_KEYS: usize = 512;
const LLM_PAYLOAD_INVALID_OVERFLOW_PROVIDER: &str = "__other__";
const LLM_PAYLOAD_INVALID_OVERFLOW_MODEL: &str = "__other__";
const LLM_PAYLOAD_INVALID_OVERFLOW_REASON: &str = "__other__";
type PayloadInvalidMetricKey = (String, String, String);
type PayloadInvalidMetricMap = HashMap<PayloadInvalidMetricKey, u64>;
static LLM_PAYLOAD_INVALID_BREAKDOWN: Lazy<std::sync::Mutex<PayloadInvalidMetricMap>> =
    Lazy::new(|| std::sync::Mutex::new(HashMap::new()));

pub(in crate::agent) fn provider_kind_metric_label(
    kind: crate::config::ProviderKind,
) -> &'static str {
    match kind {
        crate::config::ProviderKind::OpenaiCompatible => "openai_compatible",
        crate::config::ProviderKind::XaiNative => "xai_native",
        crate::config::ProviderKind::GoogleGenai => "google_genai",
        crate::config::ProviderKind::Anthropic => "anthropic",
        crate::config::ProviderKind::OpenaiChatgpt => "openai_chatgpt",
    }
}

pub(in crate::agent) fn record_llm_payload_invalid_metric(
    provider: &str,
    model: &str,
    reason: &str,
) {
    POLICY_METRICS
        .llm_payload_invalid_total
        .fetch_add(1, Ordering::Relaxed);

    let Ok(mut breakdown) = LLM_PAYLOAD_INVALID_BREAKDOWN.lock() else {
        return;
    };
    let key = (provider.to_string(), model.to_string(), reason.to_string());
    if let Some(count) = breakdown.get_mut(&key) {
        *count = count.saturating_add(1);
        return;
    }

    if breakdown.len() >= MAX_LLM_PAYLOAD_INVALID_METRIC_KEYS {
        let overflow_key = (
            LLM_PAYLOAD_INVALID_OVERFLOW_PROVIDER.to_string(),
            LLM_PAYLOAD_INVALID_OVERFLOW_MODEL.to_string(),
            LLM_PAYLOAD_INVALID_OVERFLOW_REASON.to_string(),
        );
        let count = breakdown.entry(overflow_key).or_insert(0);
        *count = count.saturating_add(1);
        return;
    }

    breakdown.insert(key, 1);
}

fn llm_payload_invalid_breakdown_snapshot() -> Vec<LlmPayloadInvalidMetric> {
    let Ok(breakdown) = LLM_PAYLOAD_INVALID_BREAKDOWN.lock() else {
        return Vec::new();
    };
    let mut rows: Vec<LlmPayloadInvalidMetric> = breakdown
        .iter()
        .map(
            |((provider, model, reason), count)| LlmPayloadInvalidMetric {
                provider: provider.clone(),
                model: model.clone(),
                reason: reason.clone(),
                count: *count,
            },
        )
        .collect();
    rows.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| a.provider.cmp(&b.provider))
            .then_with(|| a.model.cmp(&b.model))
            .then_with(|| a.reason.cmp(&b.reason))
    });
    rows
}

struct PolicyRuntimeTunables {
    initialized: AtomicBool,
    // Stored as basis points (e.g. 0.55 => 5500) for lock-free updates.
    uncertainty_threshold_bp: AtomicU64,
}

impl PolicyRuntimeTunables {
    const fn new() -> Self {
        Self {
            initialized: AtomicBool::new(false),
            uncertainty_threshold_bp: AtomicU64::new(5500),
        }
    }
}

static POLICY_TUNABLES: Lazy<PolicyRuntimeTunables> = Lazy::new(PolicyRuntimeTunables::new);
static POLICY_METRICS_BOOT_ID: Lazy<String> = Lazy::new(|| uuid::Uuid::new_v4().to_string());

pub fn policy_metrics_boot_id() -> &'static str {
    POLICY_METRICS_BOOT_ID.as_str()
}

pub fn durable_policy_metrics_snapshot() -> crate::events::PolicyMetricsSnapshotData {
    crate::events::PolicyMetricsSnapshotData {
        schema_version: crate::events::PolicyMetricsSnapshotData::SCHEMA_VERSION,
        boot_id: policy_metrics_boot_id().to_string(),
        metrics: policy_metrics_snapshot(),
    }
}

pub fn policy_metrics_snapshot() -> PolicyMetricsData {
    PolicyMetricsData {
        tool_exposure_samples: POLICY_METRICS.tool_exposure_samples.load(Ordering::Relaxed),
        tool_exposure_before_sum: POLICY_METRICS
            .tool_exposure_before_sum
            .load(Ordering::Relaxed),
        tool_exposure_after_sum: POLICY_METRICS
            .tool_exposure_after_sum
            .load(Ordering::Relaxed),
        tool_schema_contract_rejections_total: POLICY_METRICS
            .tool_schema_contract_rejections_total
            .load(Ordering::Relaxed),
        ambiguity_detected_total: POLICY_METRICS
            .ambiguity_detected_total
            .load(Ordering::Relaxed),
        uncertainty_clarify_total: POLICY_METRICS
            .uncertainty_clarify_total
            .load(Ordering::Relaxed),
        context_refresh_total: POLICY_METRICS.context_refresh_total.load(Ordering::Relaxed),
        escalation_total: POLICY_METRICS.escalation_total.load(Ordering::Relaxed),
        fallback_expansion_total: POLICY_METRICS
            .fallback_expansion_total
            .load(Ordering::Relaxed),
        response_direct_return_total: POLICY_METRICS
            .response_direct_return_total
            .load(Ordering::Relaxed),
        response_fallthrough_total: POLICY_METRICS
            .response_fallthrough_total
            .load(Ordering::Relaxed),
        orchestration_route_clarification_required_total: POLICY_METRICS
            .orchestration_route_clarification_required_total
            .load(Ordering::Relaxed),
        orchestration_route_tools_required_total: POLICY_METRICS
            .orchestration_route_tools_required_total
            .load(Ordering::Relaxed),
        orchestration_route_short_correction_direct_reply_total: POLICY_METRICS
            .orchestration_route_short_correction_direct_reply_total
            .load(Ordering::Relaxed),
        orchestration_route_acknowledgment_direct_reply_total: POLICY_METRICS
            .orchestration_route_acknowledgment_direct_reply_total
            .load(Ordering::Relaxed),
        orchestration_route_default_continue_total: POLICY_METRICS
            .orchestration_route_default_continue_total
            .load(Ordering::Relaxed),
        context_bleed_prevented_total: POLICY_METRICS
            .context_bleed_prevented_total
            .load(Ordering::Relaxed),
        context_mismatch_preflight_drop_total: POLICY_METRICS
            .context_mismatch_preflight_drop_total
            .load(Ordering::Relaxed),
        followup_mode_overrides_total: POLICY_METRICS
            .followup_mode_overrides_total
            .load(Ordering::Relaxed),
        cross_scope_blocked_total: POLICY_METRICS
            .cross_scope_blocked_total
            .load(Ordering::Relaxed),
        tokens_failed_tasks_total: POLICY_METRICS
            .tokens_failed_tasks_total
            .load(Ordering::Relaxed),
        est_input_token_samples: POLICY_METRICS
            .est_input_token_samples
            .load(Ordering::Relaxed),
        est_input_tokens_total: POLICY_METRICS
            .est_input_tokens_total
            .load(Ordering::Relaxed),
        est_msg_tokens_total: POLICY_METRICS.est_msg_tokens_total.load(Ordering::Relaxed),
        est_tool_tokens_total: POLICY_METRICS.est_tool_tokens_total.load(Ordering::Relaxed),
        est_tool_tokens_high_share_total: POLICY_METRICS
            .est_tool_tokens_high_share_total
            .load(Ordering::Relaxed),
        est_tool_tokens_high_abs_total: POLICY_METRICS
            .est_tool_tokens_high_abs_total
            .load(Ordering::Relaxed),
        no_progress_iterations_total: POLICY_METRICS
            .no_progress_iterations_total
            .load(Ordering::Relaxed),
        deferred_no_tool_forced_required_total: POLICY_METRICS
            .deferred_no_tool_forced_required_total
            .load(Ordering::Relaxed),
        deferred_no_tool_deferral_detected_total: POLICY_METRICS
            .deferred_no_tool_deferral_detected_total
            .load(Ordering::Relaxed),
        deferred_no_tool_model_switch_total: POLICY_METRICS
            .deferred_no_tool_model_switch_total
            .load(Ordering::Relaxed),
        deferred_no_tool_error_marker_total: POLICY_METRICS
            .deferred_no_tool_error_marker_total
            .load(Ordering::Relaxed),
        llm_payload_invalid_total: POLICY_METRICS
            .llm_payload_invalid_total
            .load(Ordering::Relaxed),
        llm_payload_invalid_breakdown: llm_payload_invalid_breakdown_snapshot(),
        harness_eval_tasks_total: POLICY_METRICS
            .harness_eval_tasks_total
            .load(Ordering::Relaxed),
        harness_eval_overall_avg: {
            let tasks = POLICY_METRICS
                .harness_eval_tasks_total
                .load(Ordering::Relaxed);
            if tasks == 0 {
                0.0
            } else {
                POLICY_METRICS
                    .harness_eval_overall_milli_sum
                    .load(Ordering::Relaxed) as f64
                    / tasks as f64
                    / 1000.0
            }
        },
    }
}

pub(super) fn record_harness_eval_task(snapshot: &HarnessEvalSnapshot) {
    POLICY_METRICS
        .harness_eval_tasks_total
        .fetch_add(1, Ordering::Relaxed);
    let milli = (f64::from(snapshot.scores.overall) * 1000.0).round() as u64;
    POLICY_METRICS
        .harness_eval_overall_milli_sum
        .fetch_add(milli, Ordering::Relaxed);
}

pub(super) fn record_failed_task_tokens(tokens_used: u64) {
    POLICY_METRICS
        .tokens_failed_tasks_total
        .fetch_add(tokens_used, Ordering::Relaxed);
}

pub fn init_policy_tunables_once(base_uncertainty_threshold: f32) {
    if POLICY_TUNABLES
        .initialized
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        let bp = (base_uncertainty_threshold.clamp(0.0, 1.0) * 10_000.0) as u64;
        POLICY_TUNABLES
            .uncertainty_threshold_bp
            .store(bp, Ordering::SeqCst);
    }
}

pub(in crate::agent) fn current_uncertainty_threshold(default_threshold: f32) -> f32 {
    if POLICY_TUNABLES.initialized.load(Ordering::SeqCst) {
        let bp = POLICY_TUNABLES
            .uncertainty_threshold_bp
            .load(Ordering::SeqCst);
        (bp as f32 / 10_000.0).clamp(0.0, 1.0)
    } else {
        default_threshold
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PolicyAutotuneSnapshot {
    pub uncertainty_threshold: f32,
}

pub fn policy_autotune_snapshot(default_threshold: f32) -> PolicyAutotuneSnapshot {
    PolicyAutotuneSnapshot {
        uncertainty_threshold: current_uncertainty_threshold(default_threshold),
    }
}

/// Bounded auto-tuning: adjust uncertainty threshold within safe bounds.
/// Returns (old, new) when a change is applied.
pub fn apply_bounded_autotune_from_failure_ratio(
    failure_ratio: f64,
    enforce: bool,
) -> Option<(f32, f32)> {
    if !enforce {
        return None;
    }
    let old_bp = POLICY_TUNABLES
        .uncertainty_threshold_bp
        .load(Ordering::SeqCst);
    let old = old_bp as f32 / 10_000.0;
    let mut next = old;
    // High failure ratio -> tighten policy (ask clarification earlier).
    if failure_ratio >= 0.25 {
        next = (next - 0.02).max(0.45);
    // Low failure ratio -> relax slightly.
    } else if failure_ratio <= 0.05 {
        next = (next + 0.01).min(0.75);
    }
    if (next - old).abs() < f32::EPSILON {
        return None;
    }
    let next_bp = (next * 10_000.0) as u64;
    POLICY_TUNABLES
        .uncertainty_threshold_bp
        .store(next_bp, Ordering::SeqCst);
    Some((old, next))
}
