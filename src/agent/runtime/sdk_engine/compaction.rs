//! Run compaction: when a run outgrows the context window, its older rounds
//! are replaced by a model-written summary instead of being dropped.
//!
//! The cut only moves forward and sits at the start of a model round, so the
//! sent transcript is `prefix + summary + run[covered..]`: byte-stable between
//! compactions (the provider's prompt cache keeps working) and never splits a
//! tool call from its result. Quota fitting (`fit_run_values`) stays as the
//! fallback when a summary cannot be produced.

use super::model_call::{with_idle_timeout, STREAMING_MAX_MODEL_CALL};
use super::*;
pub(super) use aidaemon_runtime::engine::compaction::*;

impl Agent {
    /// The run view to send: compacted when the run outgrows `budget`. A
    /// failed summary leaves the view uncompacted and quota fitting applies.
    pub(super) async fn sdk_compacted_run(
        &self,
        scope: &TurnScope<'_>,
        model: &str,
        prefix_tokens: usize,
        run: &[Value],
        budget: usize,
    ) -> Vec<Value> {
        let current = scope.compaction.lock().ok().and_then(|c| c.clone());
        let view = compacted_view(run, current.as_ref());
        let view_tokens = crate::context_budget::estimate_multimodal_message_tokens(&view);
        // Mandate runs spend a durable shared balance through a lease; the
        // summarizer would bypass it, so they keep quota fitting.
        if self.mandate_execution.is_some()
            || prefix_tokens + view_tokens <= budget * COMPACT_AT_PERCENT / 100
        {
            return view;
        }
        let covered = current.as_ref().map_or(0, |c| c.covered);
        let keep_tokens = budget.saturating_sub(prefix_tokens) * KEEP_RECENT_PERCENT / 100;
        let Some(cut) = compaction_cut(&run[covered..], keep_tokens).map(|cut| covered + cut)
        else {
            return view;
        };
        if cut <= covered {
            return view;
        }
        let request = summary_request(
            current.as_ref().map(|c| c.summary.as_str()),
            &run[covered..cut],
        );
        let options = crate::traits::ChatOptions {
            tool_choice: crate::traits::ToolChoiceMode::None,
            max_tokens_override: Some(SUMMARY_MAX_OUTPUT_TOKENS),
            ..crate::traits::ChatOptions::default()
        };
        let snapshot = self.llm_runtime.snapshot();
        let mut telemetry = crate::agent::llm::LlmCallTelemetry::default();
        let idle = self.sdk_model_call_timeout();
        let response = with_idle_timeout(
            idle,
            idle.max(STREAMING_MAX_MODEL_CALL),
            self.call_llm_with_recovery(
                snapshot.provider(),
                snapshot.router(),
                model,
                &request,
                &[],
                &options,
                &mut telemetry,
                None,
                super::model_call::first_attempt_deadline(idle),
            ),
        )
        .await;
        let summary = match response {
            Ok(Ok(response)) => {
                if let Some(usage) = response.usage.as_ref() {
                    // Real spend: it counts toward the run's budgets. A stop
                    // it triggers is enforced by the next model call.
                    let iteration = scope.model_calls.load(Ordering::Relaxed) as usize;
                    let _ = sdk_charge_budgets(self, scope, usage, iteration).await;
                }
                response.content.unwrap_or_default()
            }
            Ok(Err(error)) => {
                warn!(task_id = scope.task_id, %error, "SDK engine: run compaction failed; fitting by quota");
                return view;
            }
            Err(limit) => {
                warn!(
                    task_id = scope.task_id,
                    ?limit,
                    "SDK engine: run compaction timed out; fitting by quota"
                );
                return view;
            }
        };
        if summary.trim().is_empty() {
            warn!(
                task_id = scope.task_id,
                "SDK engine: run compaction returned no summary; fitting by quota"
            );
            return view;
        }
        info!(
            task_id = scope.task_id,
            covered_before = covered,
            covered = cut,
            run_messages = run.len(),
            summary_chars = summary.len(),
            "SDK engine: compacted the run's older rounds"
        );
        let compaction = RunCompaction {
            covered: cut,
            summary: summary.trim().to_string(),
        };
        let view = compacted_view(run, Some(&compaction));
        if let Ok(mut slot) = scope.compaction.lock() {
            *slot = Some(compaction);
        }
        view
    }
}
