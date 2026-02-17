use crate::agent::recall_guardrails::filter_tool_defs_for_personal_memory;
use crate::agent::tool_loop_state::{IterationProgress, ToolLoopState};
use crate::agent::*;
use crate::execution_policy::PolicyBundle;

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_post_tool_iteration_controls(
        &self,
        session_id: &str,
        iteration: usize,
        task_tokens_used: u64,
        successful_tool_calls: usize,
        iteration_had_tool_failures: bool,
        restrict_to_personal_memory_tools: bool,
        base_tool_defs: &[Value],
        available_capabilities: &HashMap<String, ToolCapabilities>,
        policy_bundle: &PolicyBundle,
        total_tool_calls_attempted: usize,
        has_active_goal: bool,
        total_successful_tool_calls: &mut usize,
        force_text_response: &mut bool,
        pending_system_messages: &mut Vec<String>,
        tool_defs: &mut Vec<Value>,
        stall_count: &mut usize,
        deferred_no_tool_streak: &mut usize,
        consecutive_clean_iterations: &mut usize,
        fallback_expanded_once: &mut bool,
    ) {
        // Escalating early-stop nudges: remind the LLM with increasing urgency
        // to stop exploring and respond. After a hard threshold, strip tools
        // entirely to force a text response on the next iteration.
        const NUDGE_INTERVAL: usize = 6;
        const FORCE_TEXT_BASE: usize = 30;
        const FORCE_TEXT_GOAL_BACKED: usize = 55;
        let force_text_at = if has_active_goal {
            FORCE_TEXT_GOAL_BACKED
        } else {
            FORCE_TEXT_BASE
        };
        if total_tool_calls_attempted > 0
            && total_tool_calls_attempted.is_multiple_of(NUDGE_INTERVAL)
            && total_tool_calls_attempted < force_text_at
        {
            let critical_threshold = force_text_at.saturating_sub(6);
            let important_threshold = force_text_at / 2;
            let urgency = if total_tool_calls_attempted >= critical_threshold {
                format!(
                    "[SYSTEM] CRITICAL: You have used {} tokens across {} tool calls. \
                     Stop immediately and respond to the user with what you have \
                     before the hard limit ({} calls). No more exploration.",
                    task_tokens_used, total_tool_calls_attempted, force_text_at
                )
            } else if total_tool_calls_attempted >= important_threshold {
                format!(
                    "[SYSTEM] IMPORTANT: You have used {} tokens in {} tool calls. \
                     You MUST stop calling tools soon and respond to the user. \
                     Hard limit for this task is {} tool calls.",
                    task_tokens_used, total_tool_calls_attempted, force_text_at
                )
            } else {
                format!(
                    "[SYSTEM] You have used {} tokens in {} tool calls. If you have \
                     enough information to answer the user's question, stop calling \
                     tools and respond now with your findings (hard limit: {} calls).",
                    task_tokens_used, total_tool_calls_attempted, force_text_at
                )
            };
            pending_system_messages.push(urgency);
            info!(
                session_id,
                total_tool_calls_attempted, "Early-stop nudge injected (escalating)"
            );
        }

        // Hard force-stop: after FORCE_TEXT_AT tool calls, strip tools on
        // the next LLM call so the model MUST produce a text response.
        if total_tool_calls_attempted >= force_text_at && !*force_text_response {
            *force_text_response = true;
            pending_system_messages.push(format!(
                "[SYSTEM] Tool limit reached ({} calls). You must now respond to the user \
with a summary of everything you found. No more tool calls are available.",
                force_text_at
            ));
            warn!(
                session_id,
                total_tool_calls_attempted,
                force_text_at,
                has_active_goal,
                "Force-text response activated — tools stripped"
            );
        }

        // Update stall detection and fallback state.
        // NOTE: `total_successful_tool_calls` is already incremented inline
        // per successful tool call during execution. Seed the extracted state
        // with this iteration's pre-progress baseline so applying progress
        // preserves the existing semantics.
        let pre_iteration_total_success =
            total_successful_tool_calls.saturating_sub(successful_tool_calls);
        let mut loop_state = ToolLoopState {
            stall_count: *stall_count,
            total_successful_tool_calls: pre_iteration_total_success,
            consecutive_clean_iterations: *consecutive_clean_iterations,
            fallback_expanded_once: *fallback_expanded_once,
        };
        let loop_signals = loop_state.apply_iteration(IterationProgress {
            successful_tool_calls,
            iteration_had_tool_failures,
        });

        *stall_count = loop_state.stall_count;
        *total_successful_tool_calls = loop_state.total_successful_tool_calls;
        *consecutive_clean_iterations = loop_state.consecutive_clean_iterations;
        *fallback_expanded_once = loop_state.fallback_expanded_once;

        if loop_signals.no_progress {
            POLICY_METRICS
                .no_progress_iterations_total
                .fetch_add(1, Ordering::Relaxed);
        } else {
            *deferred_no_tool_streak = 0;
        }

        // Fallback expansion: widen tool set once after exactly two no-progress iterations.
        if loop_signals.should_attempt_fallback_expansion {
            let previous_count = tool_defs.len();
            let widened = self.filter_tool_definitions_for_policy(
                base_tool_defs,
                available_capabilities,
                &policy_bundle.policy,
                policy_bundle.risk_score,
                true,
            );
            let widened = if restrict_to_personal_memory_tools {
                filter_tool_defs_for_personal_memory(&widened)
            } else {
                widened
            };
            if !widened.is_empty() {
                POLICY_METRICS
                    .fallback_expansion_total
                    .fetch_add(1, Ordering::Relaxed);
                *tool_defs = widened;
                info!(
                    session_id,
                    iteration,
                    previous_count,
                    widened_count = tool_defs.len(),
                    "No-progress fallback expansion applied"
                );
            }
        }
    }
}
