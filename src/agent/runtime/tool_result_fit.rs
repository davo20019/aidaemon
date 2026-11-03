//! Prepare a tool result for the model before the model sees it.
//!
//! The tool-result boundary: untrusted output is sanitized and wrapped as
//! data, and an oversized result is paged (`read_file`), spilled to a file
//! the model can page through, or compressed head+tail. The transformation is recorded as a
//! decision point so `db_probe --task` explains what the model saw.

use crate::events::{EventEmitter, EventType};
use crate::traits::ToolCallMetadata;

use super::Agent;

/// Pretty-printed JSON bodies above this size are compacted losslessly.
const STRUCTURED_JSON_COMPACT_MIN_CHARS: usize = 2_000;

pub(crate) struct FitScope<'a> {
    pub tool_name: &'a str,
    pub session_id: &'a str,
    pub task_id: &'a str,
    pub model: &'a str,
    pub emitter: &'a EventEmitter,
    /// Mandate runs never spill: their workers have no follow-up read path.
    pub mandate_execution: bool,
}

impl Agent {
    /// Fit `result_text` for the model and record its provenance on
    /// `result_metadata`, so the receipt says how much of the result the
    /// model saw. Call before building the receipt.
    pub(crate) async fn fit_tool_result_for_model(
        &self,
        scope: &FitScope<'_>,
        result_text: String,
        result_is_err: bool,
        result_metadata: &mut ToolCallMetadata,
    ) -> String {
        if result_metadata.result_provenance.is_none() {
            result_metadata.result_provenance = Some(
                crate::traits::ToolResultProvenance::from_authoritative_result(
                    &result_text,
                    result_metadata,
                    crate::traits::ToolResultContentSource::ToolOutput,
                ),
            );
        }
        // Untrusted output (web, MCP, terminal, channel history...) is
        // sanitized and wrapped so the model treats it as data, not
        // instructions. Errors come from the daemon itself.
        let result_text = if result_is_err {
            result_text
        } else if !crate::tools::sanitize::is_trusted_tool(scope.tool_name) {
            let body = if result_metadata.untrusted_verbatim {
                result_text
            } else {
                crate::tools::sanitize::sanitize_external_content(&result_text)
            };
            crate::tools::sanitize::wrap_untrusted_output(scope.tool_name, &body)
        } else if scope.tool_name == "terminal" {
            crate::tools::sanitize::strip_internal_control_markers(&result_text)
        } else {
            result_text
        };
        let result_text = self
            .cap_tool_result(scope, result_text, result_is_err, result_metadata)
            .await;
        if let Some(provenance) = result_metadata.result_provenance.as_mut() {
            if provenance.model_view_completeness
                != crate::traits::ToolResultCompleteness::Truncated
            {
                provenance.model_visible_chars = result_text.chars().count();
            }
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

        // `cli_agent` errors can be extremely large (process output, stack
        // traces). Truncate aggressively to prevent context blow-up and
        // runaway retries.
        if tool_name == "cli_agent" && result_is_err {
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
                    "{}\n\n[... cli_agent error output truncated ({} chars total) ...]\n\n{}",
                    head, char_len, tail
                );
                if let Some(provenance) = result_metadata.result_provenance.as_mut() {
                    provenance.mark_model_view_truncated(&result_text);
                }
            }
        }

        // The cap is per-model so small-context models get tighter results
        // while big-context models keep more.
        if !self.context_window_config.enabled {
            return result_text;
        }
        let max_chars = self
            .context_window_config
            .tool_result_chars_for(scope.model);

        // read_file results with typed metadata get line-boundary paging with
        // an explicit continuation hint instead of destructive mid-drop
        // compression: the model can keep reading from the exact cut point.
        let read_file_metadata = (tool_name == "read_file" && !result_is_err)
            .then_some(result_metadata.read_file.as_ref())
            .flatten();
        if let Some(read_metadata) = read_file_metadata {
            if result_text.chars().count() > max_chars {
                result_text =
                    crate::tools::render_read_file_output_within(read_metadata, max_chars);
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
        if matches!(tool_name, "http_request" | "web_fetch")
            && !result_is_err
            && result_text.chars().count() > STRUCTURED_JSON_COMPACT_MIN_CHARS
        {
            if let Some(compacted) = crate::utils::compact_embedded_json(&result_text) {
                tracing::info!(
                    tool = %tool_name,
                    original_chars = result_text.chars().count(),
                    compact_chars = compacted.chars().count(),
                    "Losslessly compacted embedded JSON in tool result"
                );
                result_text = compacted;
            }
        }

        // Oversized successful results are spilled to a temp file when the
        // model has a filesystem recovery path (read_file or terminal): it
        // then pages / jq / greps the full data instead of losing the middle
        // to lossy head+tail compression. No recovery path → fall back to
        // lossy compression. Errors are never spilled.
        let original_chars = result_text.chars().count();
        if original_chars <= max_chars {
            return result_text;
        }
        // The computer_use accessibility tree is the model's working data: it
        // must stay inline and complete so the model can target an element
        // directly. Spilling it sends the planner into a read_file/terminal
        // loop instead of clicking; head+tail compression would drop mid-tree
        // controls. So never spill or compress it.
        let keep_inline = tool_name == "computer_use" || result_metadata.preserve_inline;
        let has_fs_recovery = self
            .tools
            .iter()
            .any(|t| matches!(t.name(), "read_file" | "terminal") && t.is_available());
        let spilled =
            if !result_is_err && has_fs_recovery && !keep_inline && !scope.mandate_execution {
                crate::tools::result_spill::build_spilled_preview_for_backend(
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
            None => crate::memory::context_window::compress_tool_result(
                tool_name,
                &result_text,
                max_chars,
            ),
        };
        if !keep_inline {
            if let Some(provenance) = result_metadata.result_provenance.as_mut() {
                provenance.mark_model_view_truncated(&result_text);
                if was_spilled {
                    provenance.source = crate::traits::ToolResultContentSource::SpillPreview;
                }
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
