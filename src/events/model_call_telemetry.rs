//! Shared model-call telemetry recording for root and background LLM calls.

use std::sync::Arc;
use std::time::Duration;

use tracing::warn;
use uuid::Uuid;

use crate::traits::{ProviderResponse, TokenUsage, TokenUsageStore};

use super::{EventEmitter, EventStore, LlmCallData};

/// Input for a single model-call telemetry recording operation.
#[derive(Debug, Clone)]
pub struct ModelCallTelemetryInput {
    pub session_id: String,
    pub task_id: String,
    pub call_purpose: Option<String>,
    pub iteration: Option<u32>,
    pub llm_call: LlmCallData,
    pub token_usage: Option<TokenUsage>,
}

/// Record detailed `llm_call` event and optional correlated `token_usage` row.
pub async fn record_model_call_telemetry(
    emitter: &EventEmitter,
    _state: &dyn TokenUsageStore,
    mut input: ModelCallTelemetryInput,
) {
    let call_id = input
        .llm_call
        .call_id
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    input.llm_call.call_id = Some(call_id.clone());
    input.llm_call.call_purpose = input
        .call_purpose
        .clone()
        .or_else(|| input.llm_call.call_purpose.clone());
    input.llm_call.token_usage_present = input.token_usage.is_some();

    if let Err(error) = emitter
        .emit_model_call_with_token_usage(input.llm_call, input.token_usage.as_ref(), &call_id)
        .await
    {
        warn!(
            call_id = %call_id,
            session_id = %input.session_id,
            task_id = %input.task_id,
            call_purpose = ?input.call_purpose,
            error = %error,
            "Model-call telemetry transaction failed; neither canonical event nor token projection committed"
        );
    }
}

/// Convenience wrapper for non-agent-loop LLM calls that do not have iteration
/// or prompt-prefix metadata but still need correlated detailed and aggregate
/// telemetry.
pub async fn record_background_model_call_telemetry(
    event_store: Arc<EventStore>,
    state: &dyn TokenUsageStore,
    session_id: &str,
    call_purpose: &str,
    model: &str,
    response: &ProviderResponse,
    latency: Duration,
) {
    let emitter =
        EventEmitter::new(event_store, session_id.to_string()).with_task_id(session_id.to_string());
    let (
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_creation_input_tokens,
        fresh_input_tokens,
    ) = response
        .usage
        .as_ref()
        .map(|usage| {
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.cache_creation_input_tokens,
                usage.fresh_input_tokens(),
            )
        })
        .unwrap_or((0, 0, None, None, None));
    // Server prefill/decode split — background jobs (summarization, extraction)
    // are decode-heavy, so this attributes their latency directly.
    let (prompt_ms, decode_ms) = response
        .usage
        .as_ref()
        .map(|usage| (usage.prompt_ms, usage.decode_ms))
        .unwrap_or((None, None));

    record_model_call_telemetry(
        &emitter,
        state,
        ModelCallTelemetryInput {
            session_id: session_id.to_string(),
            task_id: session_id.to_string(),
            call_purpose: Some(call_purpose.to_string()),
            iteration: None,
            llm_call: LlmCallData {
                call_id: None,
                call_purpose: Some(call_purpose.to_string()),
                task_id: session_id.to_string(),
                iteration: None,
                model: model.to_string(),
                final_model: Some(
                    response
                        .usage
                        .as_ref()
                        .map(|usage| usage.model.clone())
                        .unwrap_or_else(|| model.to_string()),
                ),
                fell_back: false,
                attempts: 1,
                latency_ms: latency.as_millis() as u64,
                prompt_ms,
                decode_ms,
                input_tokens,
                output_tokens,
                cached_input_tokens,
                cache_creation_input_tokens,
                fresh_input_tokens,
                est_input_tokens: None,
                tool_calls_count: response.tool_calls.len() as u32,
                offered_tools: Vec::new(),
                chosen_tools: Vec::new(),
                build_ms: None,
                prefix_hash_system: None,
                prefix_hash_pre_boundary: None,
                tool_defs_hash: None,
                session_summary_hash: None,
                tail_hash: None,
                prefix_hash_archived: None,
                boundary_pos: None,
                projected_source_message_ids: Vec::new(),
                projected_source_turn_ids: Vec::new(),
                message_count: None,
                force_text: false,
                token_usage_present: response.usage.is_some(),
                failed: false,
                error: None,
            },
            token_usage: response.usage.clone(),
        },
    )
    .await;
}
