use crate::agent::post_task;
use crate::agent::recall_guardrails::filter_tool_defs_for_personal_memory;
use crate::agent::tool_loop_state::{IterationProgress, ToolLoopState};
use crate::agent::*;
use crate::execution_policy::PolicyBundle;

use crate::agent::loop_utils::build_task_boundary_hint;

pub(super) struct PostToolIterationInputs<'a> {
    pub session_id: &'a str,
    pub iteration: usize,
    pub task_tokens_used: u64,
    pub successful_tool_calls: usize,
    pub iteration_had_tool_failures: bool,
    pub restrict_to_personal_memory_tools: bool,
    pub base_tool_defs: &'a [Value],
    pub available_capabilities: &'a HashMap<String, ToolCapabilities>,
    pub policy_bundle: &'a PolicyBundle,
    pub total_tool_calls_attempted: usize,
    pub has_active_goal: bool,
    pub completed_tool_calls: &'a [String],
    pub recent_tool_names: &'a VecDeque<String>,
    pub user_text: &'a str,
}

pub(super) struct PostToolIterationState<'a> {
    pub total_successful_tool_calls: &'a mut usize,
    pub force_text_response: &'a mut bool,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub tool_defs: &'a mut Vec<Value>,
    pub stall_count: &'a mut usize,
    pub deferred_no_tool_streak: &'a mut usize,
    pub consecutive_clean_iterations: &'a mut usize,
    pub fallback_expanded_once: &'a mut bool,
}

impl Agent {
    fn apply_read_saturation_controls(
        &self,
        session_id: &str,
        pending_system_messages: &mut Vec<SystemDirective>,
        tool_defs: &mut Vec<Value>,
        base_tool_defs: &[Value],
        recent_tool_names: &VecDeque<String>,
    ) {
        // Read-saturation: three-tier escalation for excessive consecutive reads.
        //
        // Tier 1 (3+ reads): Forceful nudge — command the agent to start acting.
        // Tier 2 (5+ reads): Hard escalation — strip read tools from tool_defs
        //   so the model literally cannot call read_file.
        // Restore: when consecutive_reads drops below threshold, restore read tools.
        const READ_SATURATION_THRESHOLD: usize = 3;
        const READ_SATURATION_ESCALATION: usize = 5;
        let read_only_tools = [
            "read_file",
            "search_files",
            "project_inspect",
            "terminal_read",
        ];
        let tool_def_name = |def: &Value| -> String {
            def.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string()
        };
        let is_read_only = |name: &str| -> bool {
            if read_only_tools.contains(&name) {
                return true;
            }
            if let Some(suffix) = name.split("__").last() {
                read_only_tools.contains(&suffix)
            } else {
                false
            }
        };
        let consecutive_reads = recent_tool_names
            .iter()
            .rev()
            .take_while(|name| is_read_only(name.as_str()))
            .count();

        // Also check a sliding window: if 7+ of the last 8 tools are reads,
        // escalate even if a single write_file broke the consecutive streak.
        // (Was 5/6 but that's too aggressive for legitimate multi-file tasks
        //  like "read blog post + read existing tweets + write new tweets".)
        const SLIDING_WINDOW: usize = 8;
        const SLIDING_READ_THRESHOLD: usize = 7;
        let window_read_count = recent_tool_names
            .iter()
            .rev()
            .take(SLIDING_WINDOW)
            .filter(|name| is_read_only(name.as_str()))
            .count();
        let window_total = recent_tool_names.len().min(SLIDING_WINDOW);
        let sliding_saturated =
            window_total >= SLIDING_WINDOW && window_read_count >= SLIDING_READ_THRESHOLD;

        if consecutive_reads >= READ_SATURATION_ESCALATION || sliding_saturated {
            let before_len = tool_defs.len();
            tool_defs.retain(|def| {
                let name = tool_def_name(def);
                !is_read_only(&name)
            });
            let stripped = before_len.saturating_sub(tool_defs.len());
            let read_desc = if sliding_saturated && consecutive_reads < READ_SATURATION_ESCALATION {
                format!(
                    "{} of your last {} tool calls were read-only",
                    window_read_count, window_total
                )
            } else {
                format!("{} read-only calls in a row", consecutive_reads)
            };
            pending_system_messages.push(SystemDirective::ReadSaturationCritical { read_desc });
            info!(
                session_id,
                consecutive_reads,
                window_read_count,
                sliding_saturated,
                stripped_tools = stripped,
                "Read-saturation escalation applied"
            );
            return;
        }

        if consecutive_reads >= READ_SATURATION_THRESHOLD {
            pending_system_messages
                .push(SystemDirective::ReadSaturationWarning { consecutive_reads });
            info!(
                session_id,
                consecutive_reads, "Read-saturation nudge injected"
            );
            return;
        }

        let has_read_file = tool_defs.iter().any(|def| {
            let name = tool_def_name(def);
            name == "read_file" || name.ends_with("__read_file")
        });
        if !has_read_file {
            for base_def in base_tool_defs.iter() {
                let name = tool_def_name(base_def);
                if is_read_only(&name) {
                    tool_defs.push(base_def.clone());
                }
            }
            info!(session_id, "Read tools restored after non-read action");
        }
    }

    fn apply_terminal_after_edit_nudge(
        &self,
        session_id: &str,
        pending_system_messages: &mut Vec<SystemDirective>,
        recent_tool_names: &VecDeque<String>,
    ) {
        // Terminal-after-edit nudge: if the agent has made edits but then runs
        // terminal 2+ consecutive times without making more edits, nudge it to
        // analyze test failures and fix remaining bugs instead of re-running tests.
        const TERMINAL_AFTER_EDIT_THRESHOLD: usize = 2;
        let has_edit = recent_tool_names.iter().any(|n| n == "edit_file");
        if !has_edit {
            return;
        }
        let consecutive_terminals = recent_tool_names
            .iter()
            .rev()
            .take_while(|name| name.as_str() != "edit_file")
            .filter(|name| name.as_str() == "terminal")
            .count();
        if consecutive_terminals >= TERMINAL_AFTER_EDIT_THRESHOLD {
            pending_system_messages.push(SystemDirective::TerminalAfterEdit {
                consecutive_terminals,
            });
            info!(
                session_id,
                consecutive_terminals, "Terminal-after-edit nudge injected"
            );
        }
    }

    pub(super) fn apply_post_tool_iteration_controls(
        &self,
        inputs: PostToolIterationInputs<'_>,
        state: PostToolIterationState<'_>,
    ) {
        let PostToolIterationInputs {
            session_id,
            iteration,
            task_tokens_used,
            successful_tool_calls,
            iteration_had_tool_failures,
            restrict_to_personal_memory_tools,
            base_tool_defs,
            available_capabilities,
            policy_bundle,
            total_tool_calls_attempted,
            has_active_goal,
            completed_tool_calls,
            recent_tool_names,
            user_text,
        } = inputs;
        let PostToolIterationState {
            total_successful_tool_calls,
            force_text_response,
            pending_system_messages,
            tool_defs,
            stall_count,
            deferred_no_tool_streak,
            consecutive_clean_iterations,
            fallback_expanded_once,
        } = state;

        // Prioritized control engine:
        // 1) hard response coercion (early-stop/force-text),
        // 2) no-progress state transitions and fallback expansion,
        // 3) read-saturation shaping,
        // 4) terminal-after-edit nudge.

        // Escalating early-stop nudges: remind the LLM with increasing urgency
        // to stop exploring and respond. After a hard threshold, strip tools
        // entirely to force a text response on the next iteration.
        const NUDGE_INTERVAL: usize = 10;
        const FORCE_TEXT_BASE: usize = 40;
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
            let task_hint = build_task_boundary_hint(user_text, 150);
            let task_anchor = if task_hint.is_empty() {
                String::new()
            } else {
                format!("\nCurrent task: {}", task_hint)
            };
            let severity = if total_tool_calls_attempted >= critical_threshold {
                EarlyStopSeverity::Critical
            } else if total_tool_calls_attempted >= important_threshold {
                EarlyStopSeverity::Important
            } else {
                EarlyStopSeverity::Normal
            };
            pending_system_messages.push(SystemDirective::EarlyStopUrgency {
                task_tokens_used,
                total_tool_calls_attempted,
                force_text_at,
                task_anchor,
                severity,
            });
            info!(
                session_id,
                total_tool_calls_attempted, "Early-stop nudge injected (escalating)"
            );
        }

        // Hard force-stop: after FORCE_TEXT_AT tool calls, strip tools on
        // the next LLM call so the model MUST produce a text response.
        if total_tool_calls_attempted >= force_text_at && !*force_text_response {
            *force_text_response = true;
            let activity = post_task::categorize_tool_calls(completed_tool_calls);
            let activity_section = if activity.is_empty() {
                String::new()
            } else {
                format!(
                    "\nHere is what you actually did (use this as ground truth):\n{}\n",
                    activity
                )
            };
            let force_task_hint = build_task_boundary_hint(user_text, 200);
            let force_task_anchor = if force_task_hint.is_empty() {
                String::new()
            } else {
                format!("User's request: {}\n\n", force_task_hint)
            };
            pending_system_messages.push(SystemDirective::ForceTextToolLimitReached {
                force_text_at,
                force_task_anchor,
                activity_section,
            });
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
        // Cap at 20 tools to avoid context budget bloat. Widening from 12→36 tools
        // adds ~7200 tokens of schemas (9255 - 2043), which drops model_budget from
        // ~18000 to ~10600 and actually makes things worse for coding tasks.
        if loop_signals.should_attempt_fallback_expansion {
            let previous_count = tool_defs.len();
            let mut widened = self.filter_tool_definitions_for_policy(
                base_tool_defs,
                available_capabilities,
                &policy_bundle.policy,
                policy_bundle.risk_score,
                true,
            );
            widened =
                self.restrict_connected_api_setup_tools_for_request(inputs.user_text, &widened);
            widened =
                self.ensure_connected_api_tools_exposed(inputs.user_text, &widened, base_tool_defs);
            widened.truncate(20);
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

        self.apply_read_saturation_controls(
            session_id,
            pending_system_messages,
            tool_defs,
            base_tool_defs,
            recent_tool_names,
        );

        self.apply_terminal_after_edit_nudge(
            session_id,
            pending_system_messages,
            recent_tool_names,
        );

        self.apply_edit_stall_write_hint(
            session_id,
            pending_system_messages,
            recent_tool_names,
            iteration_had_tool_failures,
        );

        self.apply_memory_search_saturation_controls(
            session_id,
            pending_system_messages,
            tool_defs,
            recent_tool_names,
        );

        self.apply_research_synthesis_nudge(session_id, pending_system_messages, recent_tool_names);

        self.apply_build_fix_cycle_nudge(
            session_id,
            iteration,
            pending_system_messages,
            recent_tool_names,
        );
    }

    fn apply_memory_search_saturation_controls(
        &self,
        session_id: &str,
        pending_system_messages: &mut Vec<SystemDirective>,
        tool_defs: &mut Vec<Value>,
        recent_tool_names: &VecDeque<String>,
    ) {
        // Memory search saturation: after many consecutive memory tool calls
        // (manage_memories, remember_fact), the model is often stuck in a search
        // loop trying to verify facts or searching with slightly different queries.
        //
        // Tier 1 (4+ consecutive): Nudge — tell model to stop and respond.
        // Tier 2 (7+ consecutive OR sliding window): Strip memory tools so the
        //   model literally cannot call them.
        const MEMORY_NUDGE_THRESHOLD: usize = 4;
        const MEMORY_STRIP_THRESHOLD: usize = 7;
        let memory_tools = ["manage_memories", "remember_fact", "manage_people"];
        let is_memory_tool = |name: &str| -> bool { memory_tools.contains(&name) };

        let consecutive_memory_calls = recent_tool_names
            .iter()
            .rev()
            .take_while(|name| is_memory_tool(name.as_str()))
            .count();

        // Sliding window: if 6+ of the last 8 tools are memory tools, escalate
        // even if a single non-memory tool broke the consecutive streak.
        const SLIDING_WINDOW: usize = 8;
        const SLIDING_MEMORY_THRESHOLD: usize = 6;
        let window_memory_count = recent_tool_names
            .iter()
            .rev()
            .take(SLIDING_WINDOW)
            .filter(|name| is_memory_tool(name.as_str()))
            .count();
        let window_total = recent_tool_names.len().min(SLIDING_WINDOW);
        let sliding_saturated =
            window_total >= SLIDING_WINDOW && window_memory_count >= SLIDING_MEMORY_THRESHOLD;

        let tool_def_name = |def: &Value| -> String {
            def.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string()
        };

        if consecutive_memory_calls >= MEMORY_STRIP_THRESHOLD || sliding_saturated {
            let before_len = tool_defs.len();
            tool_defs.retain(|def| {
                let name = tool_def_name(def);
                !is_memory_tool(&name)
            });
            let stripped = before_len.saturating_sub(tool_defs.len());
            pending_system_messages.push(SystemDirective::MemorySearchSaturation {
                consecutive_memory_calls,
            });
            info!(
                session_id,
                consecutive_memory_calls,
                window_memory_count,
                sliding_saturated,
                stripped_tools = stripped,
                "Memory search saturation escalation — memory tools stripped"
            );
            return;
        }

        if consecutive_memory_calls >= MEMORY_NUDGE_THRESHOLD {
            pending_system_messages.push(SystemDirective::MemorySearchSaturation {
                consecutive_memory_calls,
            });
            info!(
                session_id,
                consecutive_memory_calls, "Memory search saturation nudge injected"
            );
        }
    }

    fn apply_research_synthesis_nudge(
        &self,
        session_id: &str,
        pending_system_messages: &mut Vec<SystemDirective>,
        recent_tool_names: &VecDeque<String>,
    ) {
        // Research synthesis nudge: after 3+ consecutive web searches, nudge
        // the model to evaluate whether it has enough evidence to answer.
        // This lets the model reason about information sufficiency instead of
        // relying on hard numeric limits.
        const RESEARCH_NUDGE_THRESHOLD: usize = 3;
        let research_tools = ["web_search", "web_fetch"];
        let consecutive_searches = recent_tool_names
            .iter()
            .rev()
            .take_while(|name| research_tools.contains(&name.as_str()))
            .count();

        if consecutive_searches >= RESEARCH_NUDGE_THRESHOLD {
            pending_system_messages.push(SystemDirective::ResearchSynthesisNudge {
                consecutive_searches,
            });
            info!(
                session_id,
                consecutive_searches, "Research synthesis nudge injected"
            );
        }
    }

    fn apply_edit_stall_write_hint(
        &self,
        session_id: &str,
        pending_system_messages: &mut Vec<SystemDirective>,
        recent_tool_names: &VecDeque<String>,
        iteration_had_tool_failures: bool,
    ) {
        // Nudge the model to use write_file instead of edit_file when it appears
        // stuck in an edit loop.  Two detection modes:
        //
        // 1. Strict consecutive: 3+ consecutive edit_file calls with failures.
        // 2. Sliding window: in the last 8 tool calls, if 4+ are edit_file and
        //    the pattern includes interleaved read_file (classic failed-edit
        //    retry: edit→read→edit→read...), inject the hint.  This catches
        //    loops where the model alternates between reading the file and
        //    retrying edits.
        // Mode 1: strict consecutive (requires current iteration to have failures)
        if iteration_had_tool_failures {
            let consecutive_edits = recent_tool_names
                .iter()
                .rev()
                .take_while(|name| name.as_str() == "edit_file")
                .count();
            if consecutive_edits >= 3 {
                pending_system_messages.push(SystemDirective::EditStallWriteFileHint);
                info!(
                    session_id,
                    consecutive_edits, "Edit-stall write_file hint injected (consecutive)"
                );
                return;
            }
        }

        // Mode 2: sliding window — edit/read interleave pattern.
        // Check regardless of current iteration failures — the model often
        // alternates between terminal (succeeds) and edit_file (fails),
        // so the failure guard would prevent detection on terminal iterations.
        const WINDOW: usize = 8;
        const EDIT_THRESHOLD: usize = 4;
        let window: Vec<&str> = recent_tool_names
            .iter()
            .rev()
            .take(WINDOW)
            .map(|s| s.as_str())
            .collect();
        let edit_count = window.iter().filter(|&&n| n == "edit_file").count();
        let read_count = window.iter().filter(|&&n| n == "read_file").count();
        if edit_count >= EDIT_THRESHOLD && read_count >= 1 {
            pending_system_messages.push(SystemDirective::EditStallWriteFileHint);
            info!(
                session_id,
                edit_count, read_count, "Edit-stall write_file hint injected (sliding window)"
            );
        }
    }

    /// Detect build→edit→build→edit cycles (common with weaker models).
    /// When the agent keeps alternating between running build/test commands
    /// and editing files without converging, nudge it to step back and
    /// write the complete file from scratch.
    fn apply_build_fix_cycle_nudge(
        &self,
        session_id: &str,
        iteration: usize,
        pending_system_messages: &mut Vec<SystemDirective>,
        recent_tool_names: &VecDeque<String>,
    ) {
        // Only activate after enough iterations to be confident it's a cycle
        const MIN_ITERATION: usize = 15;
        if iteration < MIN_ITERATION {
            return;
        }

        const WINDOW: usize = 10;
        let build_tools = ["terminal", "run_command"];
        let modify_tools = ["write_file", "edit_file"];

        let window: Vec<&str> = recent_tool_names
            .iter()
            .rev()
            .take(WINDOW)
            .map(|s| s.as_str())
            .collect();

        let build_count = window.iter().filter(|&&n| build_tools.contains(&n)).count();
        let modify_count = window
            .iter()
            .filter(|&&n| modify_tools.contains(&n))
            .count();

        // Both building and modifying heavily in the same window = cycle
        if build_count >= 3 && modify_count >= 3 {
            pending_system_messages.push(SystemDirective::BuildFixCycleNudge);
            info!(
                session_id,
                iteration, build_count, modify_count, "Build-fix cycle nudge injected"
            );
        }
    }
}
