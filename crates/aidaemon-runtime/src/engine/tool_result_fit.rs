//! Prepare a tool result for the model before the model sees it.
//!
//! The tool-result boundary: untrusted output is sanitized and wrapped as
//! data, and an oversized result is paged (`read_file`), spilled to a file
//! the model can page through, or compressed head+tail. The transformation is recorded as a
//! decision point so `db_probe --task` explains what the model saw.

use aidaemon_events::{EventEmitter, EventType};
use aidaemon_tools::semantics::ToolCallMetadata;

use crate::host::TurnHost;

/// Pretty-printed JSON bodies above this size are compacted losslessly.
const STRUCTURED_JSON_COMPACT_MIN_CHARS: usize = 2_000;

pub struct FitScope<'a> {
    pub tool_name: &'a str,
    /// The called tool's dispatch facts: how its result is fitted.
    pub facts: aidaemon_tools::semantics::ToolDispatchFacts,
    pub session_id: &'a str,
    pub task_id: &'a str,
    pub model: &'a str,
    pub emitter: &'a EventEmitter,
    /// Mandate runs never spill: their workers have no follow-up read path.
    pub mandate_execution: bool,
    /// Who authored what this exact call returned.
    pub trust: aidaemon_core::SourceTrust,
}

/// Fitting a tool result for the model, on any [`TurnHost`].
#[async_trait::async_trait]
pub trait ToolResultFit: TurnHost {
    /// Fit `result_text` for the model and record its provenance on
    /// `result_metadata`, so the receipt says how much of the result the
    /// model saw. Call before building the receipt.
    async fn fit_tool_result_for_model(
        &self,
        scope: &FitScope<'_>,
        result_text: String,
        result_is_err: bool,
        result_metadata: &mut ToolCallMetadata,
    ) -> String {
        if result_metadata.result_provenance.is_none() {
            result_metadata.result_provenance = Some(
                aidaemon_tools::semantics::ToolResultProvenance::from_authoritative_result(
                    &result_text,
                    result_metadata,
                    aidaemon_tools::semantics::ToolResultContentSource::ToolOutput,
                ),
            );
        }
        // This is the one place a tool result is marked as data: it feeds
        // both the model's view and the stored history. Untrusted output is
        // sanitized and wrapped from the call's typed trust, errors included:
        // a remote server can write its own error text. Capping comes first,
        // so a cut result is still one intact block.
        let untrusted = scope.trust == aidaemon_core::SourceTrust::Untrusted;
        let result_text = if untrusted && !result_metadata.untrusted_verbatim {
            aidaemon_tools::sanitize::sanitize_external_content(&result_text)
        } else if !untrusted && !result_is_err && scope.facts.result_fit.strip_control_markers {
            aidaemon_tools::sanitize::strip_internal_control_markers(&result_text)
        } else {
            result_text
        };
        let result_text = self
            .cap_tool_result(scope, result_text, result_is_err, result_metadata)
            .await;
        let result_text = if untrusted {
            aidaemon_tools::sanitize::wrap_untrusted_output(scope.tool_name, &result_text)
        } else {
            result_text
        };
        if let Some(provenance) = result_metadata.result_provenance.as_mut()
            && provenance.model_view_completeness
                != aidaemon_tools::semantics::ToolResultCompleteness::Truncated
        {
            provenance.model_visible_chars = result_text.chars().count();
        }
        result_text
    }

    async fn cap_tool_result(
        &self,
        scope: &FitScope<'_>,
        mut result_text: String,
        result_is_err: bool,
        result_metadata: &mut ToolCallMetadata,
    ) -> String {
        let tool_name = scope.tool_name;
        let fit = scope.facts.result_fit;

        // A process's errors can be extremely large (its output, stack
        // traces). Truncate aggressively to prevent context blow-up and
        // runaway retries.
        if fit.cap_large_errors && result_is_err {
            let char_len = result_text.chars().count();
            if char_len > 2000 {
                let head: String = result_text.chars().take(500).collect();
                let tail: String = result_text
                    .chars()
                    .rev()
                    .take(500)
                    .collect::<Vec<char>>()
                    .into_iter()
                    .rev()
                    .collect();
                result_text = format!(
                    "{}\n\n[... {} error output truncated ({} chars total) ...]\n\n{}",
                    head, tool_name, char_len, tail
                );
                if let Some(provenance) = result_metadata.result_provenance.as_mut() {
                    provenance.mark_model_view_truncated(&result_text);
                }
            }
        }

        // The cap is per-model so small-context models get tighter results
        // while big-context models keep more.
        if !self.turn_runtime().context_window_config.enabled {
            return result_text;
        }
        let max_chars = self
            .turn_runtime()
            .context_window_config
            .tool_result_chars_for(scope.model);

        // A file read's typed metadata gives line-boundary paging with an
        // explicit continuation hint instead of destructive mid-drop
        // compression: the model can keep reading from the exact cut point.
        let read_file_metadata = (!result_is_err)
            .then_some(result_metadata.read_file.as_ref())
            .flatten();
        if let Some(read_metadata) = read_file_metadata {
            if result_text.chars().count() > max_chars {
                result_text =
                    aidaemon_tools::render_read_file_output_within(read_metadata, max_chars);
                if let Some(provenance) = result_metadata.result_provenance.as_mut() {
                    provenance.mark_model_view_truncated(&result_text);
                }
            }
            return result_text;
        }

        // Structured API bodies: pretty-printed JSON costs ~2.5-3x the tokens
        // of its compact form with identical information. Compact it
        // losslessly BEFORE the size checks; this can also keep a result
        // inline that would otherwise spill. Small bodies stay pretty-printed
        // for model readability.
        if fit.compact_json
            && !result_is_err
            && result_text.chars().count() > STRUCTURED_JSON_COMPACT_MIN_CHARS
            && let Some(compacted) = crate::context_budget::compact_embedded_json(&result_text)
        {
            tracing::info!(
                tool = %tool_name,
                original_chars = result_text.chars().count(),
                compact_chars = compacted.chars().count(),
                "Losslessly compacted embedded JSON in tool result"
            );
            result_text = compacted;
        }

        // Oversized successful results are spilled to a temp file when the
        // model has a tool that pages through one (`pages_spilled_results`): it
        // then pages / jq / greps the full data instead of losing the middle
        // to lossy head+tail compression. No recovery path → fall back to
        // lossy compression. Errors are never spilled.
        let original_chars = result_text.chars().count();
        if original_chars <= max_chars {
            return result_text;
        }
        // A result that is the model's working data (the desktop's
        // accessibility tree, `preserve_inline`) must stay inline and
        // complete so the model can target an element directly. Spilling it
        // sends the planner into a paging loop instead of clicking; head+tail
        // compression would drop mid-tree controls. So never spill or
        // compress it.
        let keep_inline = fit.preserve_inline || result_metadata.preserve_inline;
        let has_fs_recovery = self
            .tools()
            .iter()
            .any(|t| t.dispatch_facts().result_fit.pages_spilled_results && t.is_available());
        let spilled =
            if !result_is_err && has_fs_recovery && !keep_inline && !scope.mandate_execution {
                aidaemon_tools::result_spill::build_spilled_preview_for_backend(
                    self.execution_backend(),
                    tool_name,
                    scope.session_id,
                    &result_text,
                    max_chars,
                )
                .await
            } else {
                None
            };
        let was_spilled = spilled.is_some();
        result_text = match spilled {
            Some(preview) => {
                tracing::info!(
                    target: "inline_dump_spill",
                    tool = %tool_name,
                    "Large tool result spilled to file this turn"
                );
                preview
            }
            None if keep_inline => result_text,
            None => crate::context_budget::compress_tool_result(tool_name, &result_text, max_chars),
        };
        if !keep_inline && let Some(provenance) = result_metadata.result_provenance.as_mut() {
            provenance.mark_model_view_truncated(&result_text);
            if was_spilled {
                provenance.source =
                    aidaemon_tools::semantics::ToolResultContentSource::SpillPreview;
            }
        }
        // Record how the harness transformed the result before the model saw
        // it, so a single `db_probe --task` query explains "why did the model
        // do that".
        if !result_is_err {
            let shown_chars = result_text.chars().count();
            let decision_type = if was_spilled {
                "tool_result_spilled"
            } else if keep_inline {
                "tool_result_kept_inline"
            } else {
                "tool_result_compressed"
            };
            let _ = scope
                .emitter
                .emit(
                    EventType::DecisionPoint,
                    serde_json::json!({
                        "decision_type": decision_type,
                        "name": tool_name,
                        "task_id": scope.task_id,
                        "original_chars": original_chars,
                        "shown_chars": shown_chars,
                        "hidden_chars": original_chars.saturating_sub(shown_chars),
                        "max_chars": max_chars,
                    }),
                )
                .await;
        }
        result_text
    }
}

impl<H: TurnHost + ?Sized> ToolResultFit for H {}
