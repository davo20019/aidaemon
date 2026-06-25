#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) enum ToolResultNotice {
    HardPolicyToolBudgetBlocked {
        policy_tool_budget: usize,
        tool_name: String,
    },
    PersonalMemoryToolsOnly {
        tool_name: String,
    },
    ScopeLockBlockedResult {
        tool_name: String,
        reason: String,
    },
    DeterministicArgumentContractBlocked {
        tool_name: String,
        reason: String,
    },
    RunCommandPolicyAutoRoutedToTerminal,
    CliAgentInlineBoundary {
        task_hint: String,
    },
    RequiredFileRecheckCompleted,
    SendFileSucceededStopAndReply,
    RecoverableFilePathMiss {
        tool_name: String,
    },
    TransientFailureCooldown {
        tool_name: String,
        cooldown_until: usize,
        cooldown_iters: usize,
    },
    SpecialistFailurePivot,
    BrowserLaunchFallback,
    OffTargetFactStorageRequest,
    MissingGoalIdManageMemories,
    MissingGoalIdGeneric,
    /// A page read failed (HTTP error, timeout, navigation failure). The
    /// model must pivot to a different source URL instead of retrying the
    /// same one (observed: same unreadable URL re-fetched 4x, then give-up).
    FetchFailedTryDifferentSource {
        tool_name: String,
        url: String,
    },
    WriteFileJsonRecovery {
        path: String,
    },
    EditFileJsonRecovery {
        path: String,
    },
    EditFileTextNotFoundRecovery {
        path: String,
    },
    EditFileReplaceAllRecovery {
        path: String,
    },
    ToolContractFailureRetry {
        tool_name: String,
    },
    EnvironmentFailureGuidance {
        tool_name: String,
    },
    LogicFailureReplan {
        tool_name: String,
    },
    ErrorCoaching {
        key_line: String,
    },
    PermissionDeniedCoaching {
        key_line: String,
    },
    SemanticFailureCoaching {
        semantic_count: usize,
        key_line: String,
    },
    RepeatedReadCached {
        repetitive_count: usize,
        cached_content: String,
        tool_name: String,
    },
    RepeatedReadBlocked {
        tool_name: String,
        repetitive_count: usize,
    },
    SemanticReadReplay {
        rendered_output: String,
    },
    SemanticReadPartialOverlap {
        covered_intervals: String,
        uncovered_intervals: String,
    },
    RepeatedApiCallBlocked {
        tool_name: String,
        repetitive_count: usize,
        previous_result_hint: String,
    },
    RepeatedTerminalBlockedAfterEdits {
        repetitive_count: usize,
    },
    RepetitiveToolBlocked {
        tool_name: String,
        repetitive_count: usize,
    },
    ToolCooldownBlocked {
        tool_name: String,
        until_iteration: usize,
    },
    ToolNotCurrentlyExposed {
        tool_name: String,
    },
    UnknownToolInvented {
        tool_name: String,
    },
    SemanticErrorLimitBlocked {
        tool_name: String,
        prior_signature_failures: usize,
        prior_transient_failures: usize,
    },
    WebSearchBudgetBlocked {
        prior_calls: usize,
    },
    CombinedWebBudgetBlocked {
        combined_web_calls: usize,
    },
    WebFetchBudgetBlocked {
        prior_calls: usize,
    },
    SpawnAgentBudgetBlocked {
        prior_calls: usize,
    },
    WebSearchBackendSetupHint {
        prior_calls: usize,
    },
    ProjectInspectBudgetBlocked {
        prior_calls: usize,
    },
    GenericToolBudgetBlocked {
        tool_name: String,
        prior_calls: usize,
    },
    PathAutoInjectedFromProjectContext {
        injected_dir: String,
    },
    InternalEditFileRecoverySucceeded {
        read_note: String,
    },
}

impl ToolResultNotice {
    pub(in crate::agent) fn render(&self) -> String {
        match self {
            Self::HardPolicyToolBudgetBlocked {
                policy_tool_budget,
                tool_name,
            } => format!(
                "[SYSTEM] Hard tool budget reached: {} calls allowed per turn for this policy profile. \
                     This call to `{}` was blocked. Synthesize and answer now.",
                policy_tool_budget, tool_name
            ),
            Self::PersonalMemoryToolsOnly { tool_name } => format!(
                "[SYSTEM] Personal-memory recall should only use `manage_people` / `manage_memories` \
                             unless the user explicitly requested broader verification. \
                             Do not call `{}` for this query.",
                tool_name
            ),
            Self::ScopeLockBlockedResult { tool_name, reason } => format!(
                "[SYSTEM] Scope lock blocked `{}`: {}. Continue with tools that stay inside the active request scope.",
                tool_name, reason
            ),
            Self::DeterministicArgumentContractBlocked { tool_name, reason } => format!(
                "[SYSTEM] Blocked `{}` by deterministic argument contract: {}. \
Continue with tools that directly match the user request.",
                tool_name, reason
            ),
            Self::RunCommandPolicyAutoRoutedToTerminal => {
                "[SYSTEM] run_command was blocked by policy; auto-routed to `terminal`."
                    .to_string()
            }
            Self::CliAgentInlineBoundary { task_hint } => format!(
                "[SYSTEM] cli_agent completed. USER REQUEST SUMMARY (untrusted): {}. \
                 Unless the user explicitly asked for more work, stop calling tools and \
                 reply to the user now with what was completed. Do NOT explore other \
                 projects or start unrelated tasks.",
                task_hint
            ),
            Self::RequiredFileRecheckCompleted => {
                "[SYSTEM] Required file re-check completed. You may now synthesize findings."
                    .to_string()
            }
            Self::SendFileSucceededStopAndReply => {
                "[SYSTEM] send_file succeeded. If an active requirement checklist exists, update it now before replying; otherwise, unless the user explicitly requested additional files or modifications, stop calling tools and reply to the user now.".to_string()
            }
            Self::RecoverableFilePathMiss { tool_name } => format!(
                "[SYSTEM] Recoverable file/path miss for `{}`. \
This did NOT consume semantic lockout budget. Recheck the target path first \
(project_inspect/search_files/read_file) and retry with the exact path.",
                tool_name
            ),
            Self::TransientFailureCooldown {
                tool_name,
                cooldown_until,
                cooldown_iters,
            } => format!(
                "[SYSTEM] Detected transient failure for `{}` (timeouts/network/rate limits). \
Avoid retrying this tool until iteration {} (cooldown {} iterations). Use another approach for now. \
Only report attempts that were actually executed; do not describe retries that were blocked or skipped.",
                tool_name, cooldown_until, cooldown_iters
            ),
            Self::SpecialistFailurePivot => {
                "[SYSTEM] Specialist delegation failed. Do NOT retry the same delegation unchanged. \
Continue the original request now with available direct tools. If completion is impossible, report \
the concrete blocker and verified partial results. Do NOT claim you are monitoring the connection \
or will retry later unless a real background task or retry was scheduled."
                    .to_string()
            }
            Self::BrowserLaunchFallback => {
                "[SYSTEM] browser launch failed. Do NOT retry browser for this verification step. \
Use another available approach now: `terminal` with curl/wrangler/HTTP status checks, \
or an HTTP-capable fetch tool if available. Verify the URL or deployment with concrete HTTP evidence."
                    .to_string()
            }
            Self::OffTargetFactStorageRequest => {
                "[SYSTEM] The previous tool call was off-target for this request. \
The user appears to be asking you to learn/remember/save facts. Use `remember_fact` (batch with `facts` when needed) and do NOT call scheduled-goal tools."
                    .to_string()
            }
            Self::MissingGoalIdManageMemories => {
                "[SYSTEM] The previous `manage_memories` call was underspecified (`goal_id` missing). \
Do NOT retry the same action blindly. Switch to `manage_memories(action='list_scheduled')` \
to retrieve exact IDs (or ask the user for the goal ID), then retry the intended action with `goal_id`."
                    .to_string()
            }
            Self::MissingGoalIdGeneric => {
                "[SYSTEM] The previous tool call was underspecified (`goal_id` missing). \
Do NOT retry the same call. If this is scheduled-goal run forensics, first call \
`manage_memories(action='list_scheduled')` to get a concrete `goal_id`, then retry. \
If the user is asking to store facts, use `remember_fact` instead."
                    .to_string()
            }
            Self::FetchFailedTryDifferentSource { tool_name, url } => format!(
                "[SYSTEM] `{}` could not read {} (blocked, timed out, or unreadable). \
Do NOT retry the same URL — the result will not change. Pick a DIFFERENT URL from your \
most recent web_search results (the next-best result covering the same topic) and fetch \
that instead. If several sources fail, answer with what you verified and name the \
sources you could not reach.",
                tool_name, url
            ),
            Self::WriteFileJsonRecovery { path } => format!(
                "[SYSTEM] write_file recovery: The file content has characters that broke \
JSON encoding in the tool call arguments (backslashes, quotes, etc.). \
Retry write_file but carefully escape ALL backslashes (\\\\) and quotes (\\\") in the content string. \
For code with many special chars (regex, JSON), double-check each backslash is escaped as \\\\. \
If write_file fails again, use terminal with a SHORT heredoc:\n\
cat > {} << 'PYEOF'\n<content here>\nPYEOF\n\
Keep heredoc commands under 3000 chars to avoid approval message truncation.",
                path
            ),
            Self::EditFileJsonRecovery { path } => format!(
                "[SYSTEM] edit_file recovery: The edit content has characters that broke \
JSON encoding (backslashes, quotes, unicode escapes). Do NOT retry edit_file with the same content. \
Instead, use the terminal tool with `sed` for small targeted fixes:\n\
  sed -i '' 's/old_pattern/new_pattern/' {}\n\
Or rewrite the entire file using a QUOTED heredoc:\n\
  cat > {} << 'PYEOF'\n<full corrected file>\nPYEOF\n\
The single-quoted delimiter prevents shell expansion.",
                path, path
            ),
            Self::EditFileTextNotFoundRecovery { path } => format!(
                "[SYSTEM] edit_file recovery: do NOT ask the user for file contents yet. \
Call read_file(path=\"{}\") now, then retry edit_file with exact copied old_text. \
If the user asked for a full rewrite, use write_file for full content replacement.",
                path
            ),
            Self::EditFileReplaceAllRecovery { path } => format!(
                "[SYSTEM] edit_file recovery: disambiguate by either setting replace_all=true \
or expanding old_text with nearby unique context from read_file(path=\"{}\").",
                path
            ),
            Self::ToolContractFailureRetry { tool_name } => format!(
                "[SYSTEM] `{}` failed because the tool call contract was wrong. \
Fix the arguments or required fields and retry once with corrected parameters. \
Do NOT change target scope until the call shape is valid.",
                tool_name
            ),
            Self::EnvironmentFailureGuidance { tool_name } => format!(
                "[SYSTEM] `{}` failed because the environment or target state is missing or unavailable. \
Gather the missing evidence, inspect the target, or ask for the missing permission/configuration. \
Do NOT blindly retry the same call.",
                tool_name
            ),
            Self::LogicFailureReplan { tool_name } => format!(
                "[SYSTEM] `{}` failed because the approach or target appears wrong. \
Do NOT repeat the same call. Change approach, reduce scope, or tell the user what remains blocked.",
                tool_name
            ),
            Self::ErrorCoaching { key_line } => format!(
                "[SYSTEM] IMPORTANT — The error says: \"{}\"\n\
         Do NOT repeat the same command. Analyze what this error means and use a DIFFERENT approach.\n\
         If the error indicates something doesn't exist or isn't available, \
         research alternatives before trying again.",
                key_line
            ),
            Self::PermissionDeniedCoaching { key_line } => format!(
                "[SYSTEM] PERMISSION ERROR — \"{}\"\n\
         This path is NOT writable (system directory, wrong permissions, or read-only). \
         Do NOT retry the same path. Instead:\n\
         1. Explain to the user WHY the operation failed (permission denied)\n\
         2. Suggest an alternative writable location (e.g., home directory ~/)\n\
         3. If the user insists, they need to run the operation manually with sudo\n\
         Do NOT keep retrying write_file/edit_file to the same path.",
                key_line
            ),
            Self::SemanticFailureCoaching {
                semantic_count,
                key_line,
            } => {
                let error_context = if key_line.is_empty() {
                    String::new()
                } else {
                    format!(" The error was: \"{}\".", key_line)
                };
                format!(
                    "[SYSTEM] This tool has errored {} semantic times.{} \
         Do NOT retry this tool. Use a DIFFERENT tool or approach, \
         or respond to the user with what you know.",
                    semantic_count, error_context
                )
            }
            Self::RepeatedReadCached {
                repetitive_count,
                cached_content,
                tool_name,
            } => format!(
                "[SYSTEM] You already read this content {} times. Here it is again — \
                         do NOT read it again.\n\n\
                         --- CACHED CONTENT ---\n{}\n--- END CACHED CONTENT ---\n\n\
                         IMPORTANT: Use the content above to answer the user or continue the \
                         requested work. Do not call `{}` again.",
                repetitive_count, cached_content, tool_name
            ),
            Self::RepeatedReadBlocked {
                tool_name,
                repetitive_count,
            } => format!(
                "[SYSTEM] BLOCKED: You already called `{}` with these exact same arguments {} times. \
                         The file content should be in your conversation history. \
                         Use the retained content to answer the user or continue the requested work. \
                         Do not try to read the same content again.",
                tool_name, repetitive_count
            ),
            Self::SemanticReadReplay { rendered_output } => format!(
                "{}\n\n[SYSTEM] This content came from the complete task-local file artifact. \
                 Do not read the same file or range again. Use it to answer the user or continue \
                 the requested work.",
                rendered_output
            ),
            Self::SemanticReadPartialOverlap {
                covered_intervals,
                uncovered_intervals,
            } => format!(
                "[SYSTEM] This read overlaps content already retained for the file. \
                 Covered intervals: {}. Uncovered intervals: {}. \
                 For sequential inspection, request only each uncovered non-overlapping interval. \
                 If you are looking for specific text, use `search_files` first and then make one \
                 exact surrounding `read_file` call. Do not scan with overlapping ranges.",
                covered_intervals, uncovered_intervals
            ),
            Self::RepeatedApiCallBlocked {
                tool_name,
                repetitive_count,
                previous_result_hint,
            } => format!(
                "[SYSTEM] BLOCKED: You already called `{}` with these exact same arguments {} times. \
                         Retrying with identical arguments will NOT produce a different result.\n\n\
                         Previous result: {}\n\n\
                         You MUST either:\n\
                         - Try DIFFERENT parameters (different URL, query params, or method)\n\
                         - Tell the user what happened honestly (e.g., \"the API returned 404 — \
                         this resource may not exist\")\n\
                         - Use an alternative approach (e.g., `web_fetch` to access the web page directly)\n\n\
                         Do NOT say your requests are \"being blocked\" — be specific about the actual error.",
                tool_name, repetitive_count, previous_result_hint
            ),
            Self::RepeatedTerminalBlockedAfterEdits { repetitive_count } => format!(
                "[SYSTEM] BLOCKED: You already ran this exact terminal command {} times. \
                         Re-running tests will NOT fix bugs — you must edit the code first.\n\n\
                         NEXT STEP: Use `edit_file` to fix the next bug, THEN run tests once to verify.\n\
                         Look at the test failure output you already have and identify which file and line needs fixing.",
                repetitive_count
            ),
            Self::RepetitiveToolBlocked {
                tool_name,
                repetitive_count,
            } => format!(
                "[SYSTEM] BLOCKED: You already called `{}` with these exact same arguments {} times \
                                 and got the same result. Repeating it will NOT produce a different outcome.\n\n\
                                 You MUST change your approach. Options:\n\
                                 - Use DIFFERENT arguments or a different command\n\
                                 - If you're missing information (URL, credentials, deployment method), \
                                 ASK the user instead of guessing\n\
                                 - If this sub-task is blocked, skip it and tell the user what you \
                                 accomplished and what still needs their input",
                tool_name, repetitive_count
            ),
            Self::ToolCooldownBlocked {
                tool_name,
                until_iteration,
            } => format!(
                "[SYSTEM] Tool '{}' is in transient-failure cooldown until iteration {}. \
                     Do not call it yet; use a different approach first.",
                tool_name, until_iteration
            ),
            Self::ToolNotCurrentlyExposed { tool_name } => format!(
                "[SYSTEM] '{}' exists, but it is not available in your current tool list for this turn. \
                 Only call tools that are currently exposed. Do NOT guess or force hidden tool names.",
                tool_name
            ),
            Self::UnknownToolInvented { tool_name } => format!(
                "[SYSTEM] '{}' is not a real tool. It does NOT exist. \
                 You MUST use one of the actual available tools or respond with text. \
                 Do NOT invent tool names.",
                tool_name
            ),
            Self::SemanticErrorLimitBlocked {
                tool_name,
                prior_signature_failures,
                prior_transient_failures,
            } => format!(
                "[SYSTEM] Tool '{}' has hit the repeated semantic error limit ({}x same failure signature) \
                 (and {} transient failures). \
                 Do not call it again. Use a different approach or \
                 answer the user with what you have.",
                tool_name, prior_signature_failures, prior_transient_failures
            ),
            Self::WebSearchBudgetBlocked { prior_calls } => format!(
                "[SYSTEM] You have already called web_search {} times. \
                 Synthesize your answer from the results you have.",
                prior_calls
            ),
            Self::CombinedWebBudgetBlocked { combined_web_calls } => format!(
                "[SYSTEM] You have made {} combined web calls (web_search + web_fetch). \
                 Stop searching and synthesize your answer from the results you already have.",
                combined_web_calls
            ),
            Self::WebFetchBudgetBlocked { prior_calls } => format!(
                "[SYSTEM] You have already called web_fetch {} times. \
                 Synthesize your answer from the pages you have already fetched.",
                prior_calls
            ),
            Self::SpawnAgentBudgetBlocked { prior_calls } => format!(
                "[SYSTEM] You have already spawned {} sub-agents this turn. \
                 This is the maximum. Synthesize results from the agents you \
                 have already spawned and respond to the user.",
                prior_calls
            ),
            Self::WebSearchBackendSetupHint { prior_calls } => format!(
                "[SYSTEM] web_search returned no useful results {} times. \
                     The DuckDuckGo backend is likely blocked.\n\n\
                     Tell the user web search is not working and suggest they set up Brave Search:\n\
                     1. Get a free API key at https://brave.com/search/api/ (free tier = 2000 queries/month)\n\
                     2. Paste the API key in this chat\n\n\
                     When the user provides a Brave API key, use manage_config to:\n\
                     - set search.backend to '\"brave\"'\n\
                     - set search.api_key to '\"THEIR_KEY\"'\n\
                     Then tell them to type /reload to apply the changes.\n\
                     Alternatively, if they run a self-hosted SearxNG instance, ask them for \
                     its base URL and set search.searxng_url to it.",
                prior_calls
            ),
            Self::ProjectInspectBudgetBlocked { prior_calls } => format!(
                "[SYSTEM] You have already called project_inspect {} times this turn. \
                     Do not call project_inspect again now.\n\n\
                     Move forward by synthesizing what you already learned, then:\n\
                     - use search_files/read_file on the most relevant directories\n\
                     - ask the user to narrow scope if many folders remain\n\
                     - if you need many directories in one pass next turn, call project_inspect \
                       once with {{\"paths\":[\"/dir1\",\"/dir2\",...]}}.",
                prior_calls
            ),
            Self::GenericToolBudgetBlocked {
                tool_name,
                prior_calls,
            } => format!(
                "[SYSTEM] You have already called '{}' {} times this turn. \
                     Do not call it again. Use the results you already have to \
                     answer the user's question now.",
                tool_name, prior_calls
            ),
            Self::PathAutoInjectedFromProjectContext { injected_dir } => format!(
                "[SYSTEM] Path was auto-injected from known project context: {}",
                injected_dir
            ),
            Self::InternalEditFileRecoverySucceeded { read_note } => format!(
                "[SYSTEM] Internal edit_file recovery succeeded: {}. Retried once with exact on-disk text matched via whitespace-tolerant mapping.",
                read_note
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ToolResultNotice;

    #[test]
    fn send_file_success_render_matches_previous_text() {
        assert_eq!(
            ToolResultNotice::SendFileSucceededStopAndReply.render(),
            "[SYSTEM] send_file succeeded. If an active requirement checklist exists, update it now before replying; otherwise, unless the user explicitly requested additional files or modifications, stop calling tools and reply to the user now."
        );
    }

    #[test]
    fn cli_agent_inline_boundary_render_includes_summary() {
        let rendered = ToolResultNotice::CliAgentInlineBoundary {
            task_hint: "fix the bug".to_string(),
        }
        .render();
        assert!(rendered.contains("USER REQUEST SUMMARY (untrusted): fix the bug."));
        assert!(rendered.contains("Do NOT explore other projects"));
    }

    #[test]
    fn transient_failure_cooldown_render_mentions_iteration() {
        let rendered = ToolResultNotice::TransientFailureCooldown {
            tool_name: "web_fetch".to_string(),
            cooldown_until: 7,
            cooldown_iters: 2,
        }
        .render();
        assert!(rendered.contains("Detected transient failure for `web_fetch`"));
        assert!(rendered.contains("until iteration 7"));
        assert!(rendered.contains("cooldown 2 iterations"));
    }

    #[test]
    fn specialist_failure_pivot_requires_direct_completion_without_future_promises() {
        let rendered = ToolResultNotice::SpecialistFailurePivot.render();
        assert!(rendered.contains("direct tools"));
        assert!(rendered.contains("Do NOT retry the same delegation"));
        assert!(rendered.contains("Do NOT claim"));
        assert!(rendered.contains("monitoring"));
    }

    #[test]
    fn browser_launch_fallback_render_points_to_http_verification() {
        let rendered = ToolResultNotice::BrowserLaunchFallback.render();
        assert!(rendered.contains("browser launch failed"));
        assert!(rendered.contains("terminal"));
        assert!(rendered.contains("curl"));
        assert!(rendered.contains("HTTP"));
        assert!(rendered.contains("Do NOT retry browser"));
    }

    #[test]
    fn hard_policy_tool_budget_blocked_render_matches_previous_text() {
        assert_eq!(
            ToolResultNotice::HardPolicyToolBudgetBlocked {
                policy_tool_budget: 12,
                tool_name: "web_search".to_string(),
            }
            .render(),
            "[SYSTEM] Hard tool budget reached: 12 calls allowed per turn for this policy profile. This call to `web_search` was blocked. Synthesize and answer now."
        );
    }

    #[test]
    fn tool_not_currently_exposed_render_mentions_current_tool_list() {
        let rendered = ToolResultNotice::ToolNotCurrentlyExposed {
            tool_name: "cli_agent".to_string(),
        }
        .render();
        assert!(rendered.contains("'cli_agent' exists"));
        assert!(rendered.contains("current tool list"));
        assert!(rendered.contains("Do NOT guess or force hidden tool names"));
    }

    #[test]
    fn semantic_read_notices_are_task_neutral_and_precise() {
        let replay = ToolResultNotice::SemanticReadReplay {
            rendered_output: "File: resume.md\n1 | experience".to_string(),
        }
        .render();
        assert!(replay.contains("complete task-local file artifact"));
        assert!(replay.contains("answer the user"));
        assert!(!replay.contains("write_file"));

        let overlap = ToolResultNotice::SemanticReadPartialOverlap {
            covered_intervals: "10-20".to_string(),
            uncovered_intervals: "1-9, 21-30".to_string(),
        }
        .render();
        assert!(overlap.contains("1-9, 21-30"));
        assert!(overlap.contains("search_files"));
        assert!(overlap.contains("non-overlapping"));
    }
}

#[cfg(test)]
mod fetch_pivot_tests {
    use super::*;

    #[test]
    fn fetch_failure_notice_pivots_to_a_different_source() {
        let rendered = ToolResultNotice::FetchFailedTryDifferentSource {
            tool_name: "web_fetch".to_string(),
            url: "https://www.olympics.com/squad".to_string(),
        }
        .render();
        assert!(rendered.contains("web_fetch"));
        assert!(rendered.contains("https://www.olympics.com/squad"));
        assert!(rendered.contains("Do NOT retry"));
        assert!(rendered.contains("DIFFERENT URL"));
        assert!(rendered.contains("web_search results"));
    }
}
