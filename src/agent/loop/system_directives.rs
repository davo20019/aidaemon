#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) enum EarlyStopSeverity {
    Normal,
    Important,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::agent) enum SystemDirective {
    RouteFailsafeActive,
    FreshConversationContext,
    EmptyResponseRetry,
    TruncationRecoveryUseWriteFile,
    ToolModeDisabledPlainText,
    TaskTokenBudgetWarning {
        used: u64,
        budget: u64,
        pct: u64,
        task_anchor: String,
    },
    TaskBudgetAutoExtended {
        old_budget: i64,
        new_budget: i64,
        extension: usize,
        max_extensions: usize,
    },
    TaskBudgetExtensionApproved {
        old_budget: i64,
        new_budget: i64,
    },
    GlobalDailyBudgetExtensionApproved {
        old_budget: i64,
        new_budget: i64,
    },
    GoalDailyBudgetAutoExtended {
        old_budget: i64,
        new_budget: i64,
        extension: usize,
        max_extensions: usize,
    },
    GoalDailyBudgetExtensionApproved {
        old_budget: i64,
        new_budget: i64,
    },
    RoutingContractEnforcement,
    ContradictoryFileEvidenceExplicitPath {
        dir: String,
    },
    ContradictoryFileEvidenceRecheckRequired,
    DeferredToolCallRequired,
    DeferredProvideConcreteResults,
    RecoveryModeModelSwitch,
    NoEvidenceRespondKnownUnknown,
    CliAgentPresentResults,
    CliAgentTaskBoundary {
        task_hint: String,
    },
    ReadSaturationCritical {
        read_desc: String,
    },
    ReadSaturationWarning {
        consecutive_reads: usize,
    },
    TerminalAfterEdit {
        consecutive_terminals: usize,
    },
    EarlyStopUrgency {
        task_tokens_used: u64,
        total_tool_calls_attempted: usize,
        force_text_at: usize,
        task_anchor: String,
        severity: EarlyStopSeverity,
    },
    ForceTextToolLimitReached {
        force_text_at: usize,
        force_task_anchor: String,
        activity_section: String,
    },
    EditStallWriteFileHint,
    DuplicateSendFileAlreadySent,
    HardPolicyToolBudgetReached {
        policy_tool_budget: usize,
    },
    PersonalMemoryRecheckLimitReached,
    ScopeLockBlocked {
        tool_name: String,
        reason: String,
    },
    ArgumentContractBlocked {
        tool_name: String,
        reason: String,
        coaching: String,
    },
    HardToolLimitReached,
    BackgroundHandoff {
        notifications_active: bool,
    },
    SchedulingOwnerOnly,
    KnowledgeIntentDirectAnswer,
    DelegationModeActive,
    GoalCreationOwnerOnly,
}

impl SystemDirective {
    pub(in crate::agent) fn render(&self) -> String {
        match self {
            Self::RouteFailsafeActive => "[SYSTEM] Route fail-safe is active for this session. Use explicit tools/results, avoid direct-return shortcuts, and prioritize concrete execution evidence.".to_string(),
            Self::FreshConversationContext => "This is a fresh conversation context. There are no previous tasks. Focus exclusively on the current user request. Do not reference or repeat tool calls from any prior context.".to_string(),
            Self::EmptyResponseRetry => "[SYSTEM] Your previous reply was empty (no text and no tool calls). This retry is running with reduced conversation history to recover. You MUST either (1) call the required tools, or (2) reply with a concrete blocker and the missing info. Do NOT return an empty response.".to_string(),
            Self::TruncationRecoveryUseWriteFile => "[SYSTEM] Your previous response was cut off because it exceeded the maximum output token limit. Do NOT generate long content inline. Instead, use the write_file tool to save long content (articles, code, etc.) to a file, then summarize what you wrote in a short reply. Keep your direct response brief.".to_string(),
            Self::ToolModeDisabledPlainText => "[SYSTEM] Tool mode is disabled for this turn. Respond with plain text only. Do NOT emit tool calls.".to_string(),
            Self::TaskTokenBudgetWarning {
                used,
                budget,
                pct,
                task_anchor,
            } => format!(
                "[SYSTEM] TOKEN BUDGET WARNING: You have used {} of {} tokens ({}%). \
                 You are approaching the task token limit. Wrap up your work and \
                 respond to the user about THEIR CURRENT REQUEST immediately.{}",
                used, budget, pct, task_anchor
            ),
            Self::TaskBudgetAutoExtended {
                old_budget,
                new_budget,
                extension,
                max_extensions,
            } => format!(
                "[SYSTEM] Token budget auto-extended from {} to {} ({}/{} extensions). \
                 Continue working.",
                old_budget, new_budget, extension, max_extensions
            ),
            Self::TaskBudgetExtensionApproved {
                old_budget,
                new_budget,
            } => format!(
                "[SYSTEM] Task token budget extension approved by owner: {} -> {}. \
                 Continue working.",
                old_budget, new_budget
            ),
            Self::GlobalDailyBudgetExtensionApproved {
                old_budget,
                new_budget,
            } => format!(
                "[SYSTEM] Global daily token budget extension approved by owner: {} -> {}. \
                 Continue working.",
                old_budget, new_budget
            ),
            Self::GoalDailyBudgetAutoExtended {
                old_budget,
                new_budget,
                extension,
                max_extensions,
            } => format!(
                "[SYSTEM] Goal daily token budget auto-extended from {} to {} ({}/{} extensions). \
                 Continue working.",
                old_budget, new_budget, extension, max_extensions
            ),
            Self::GoalDailyBudgetExtensionApproved {
                old_budget,
                new_budget,
            } => format!(
                "[SYSTEM] Goal daily token budget extension approved by owner: {} -> {}. \
                 Continue working.",
                old_budget, new_budget
            ),
            Self::RoutingContractEnforcement => "[SYSTEM] ROUTING CONTRACT ENFORCEMENT: This turn requires tool execution. Ignore prior-turn outputs, run the required tool call(s) for the current user message, and then answer with concrete results.".to_string(),
            Self::ContradictoryFileEvidenceExplicitPath { dir } => format!(
                "[SYSTEM] Contradictory file evidence detected for {}: one tool found files while another reported no matches. \
                 You MUST run an explicit-path re-check (search_files/project_inspect) before answering.",
                dir
            ),
            Self::ContradictoryFileEvidenceRecheckRequired => "[SYSTEM] Contradictory file evidence was detected (one tool found files while another reported no matches). Before answering, you MUST run at least one file re-check tool with an explicit path (e.g. search_files or project_inspect with path).".to_string(),
            Self::DeferredToolCallRequired => "[SYSTEM] HARD REQUIREMENT: your next reply MUST include at least one tool call. Do NOT return planning text like \"I'll do X\". Text-only replies are invalid for this request.".to_string(),
            Self::DeferredProvideConcreteResults => "[SYSTEM] You narrated future work instead of providing results. Execute any remaining required tools, or return concrete outcomes and blockers now.".to_string(),
            Self::RecoveryModeModelSwitch => "[SYSTEM] Recovery mode: a model switch was applied because prior replies kept promising actions without tool calls. Call the required tools now and return concrete results.".to_string(),
            Self::NoEvidenceRespondKnownUnknown => "[SYSTEM] You have searched across multiple tools and keep finding no evidence. Stop searching and respond with what is known/unknown.".to_string(),
            Self::CliAgentPresentResults => "[SYSTEM] The CLI agent completed successfully and returned substantive results. Present those results to the user directly now. Do NOT claim you cannot complete the request.".to_string(),
            Self::CliAgentTaskBoundary { task_hint } => format!(
                "[SYSTEM] TASK BOUNDARY: cli_agent delegation is complete. \
                 USER REQUEST SUMMARY (untrusted): {}. Review whether the request is \
                 already satisfied. If yes, reply with a concise completion summary. \
                 Do not start unrelated work.",
                task_hint
            ),
            Self::ReadSaturationCritical { read_desc } => format!(
                "[SYSTEM] CRITICAL: {} \
                 without making meaningful changes. Read tools have been REMOVED.\n\n\
                 You MUST act NOW with one of these tools:\n\
                 - `write_file` to rewrite the file with all your fixes applied\n\
                 - `edit_file` to apply a specific fix to the code\n\
                 - `terminal` to run tests or commands\n\n\
                 You already have enough information from your previous reads. \
                 Write the corrected code NOW.",
                read_desc
            ),
            Self::ReadSaturationWarning { consecutive_reads } => format!(
                "[SYSTEM] WARNING: You have called read-only tools {} times in a row. \
                 STOP reading and ACT NOW.\n\n\
                 You MUST use `edit_file`, `write_file`, or `terminal` as your NEXT tool call. \
                 Do NOT call read_file again — you already have the information you need. \
                 If you read again, your read tools will be removed.",
                consecutive_reads
            ),
            Self::TerminalAfterEdit {
                consecutive_terminals,
            } => format!(
                "[SYSTEM] You have run terminal commands {} times since your last edit \
                 without making any new edits. If tests are still failing:\n\n\
                 1. Look at the FAILING TEST NAMES — they tell you which file has the bug\n\
                 2. Read THAT file (not one you already fixed)\n\
                 3. Compare expected vs actual values in the test output to identify the fix\n\
                 4. Use `edit_file` to fix it, then run tests ONCE to verify\n\n\
                 IMPORTANT: If you already fixed bugs in one file but other tests still fail, \
                 the remaining bugs are in DIFFERENT files. Move on to those files.",
                consecutive_terminals
            ),
            Self::EarlyStopUrgency {
                task_tokens_used,
                total_tool_calls_attempted,
                force_text_at,
                task_anchor,
                severity,
            } => match severity {
                EarlyStopSeverity::Critical => format!(
                    "[SYSTEM] CRITICAL: You have used {} tokens across {} tool calls. \
                     Stop immediately and respond to the user about THEIR REQUEST \
                     before the hard limit ({} calls). No more exploration.{}",
                    task_tokens_used, total_tool_calls_attempted, force_text_at, task_anchor
                ),
                EarlyStopSeverity::Important => format!(
                    "[SYSTEM] IMPORTANT: You have used {} tokens in {} tool calls. \
                     You MUST stop calling tools soon and respond about the user's request. \
                     Hard limit for this task is {} tool calls.{}",
                    task_tokens_used, total_tool_calls_attempted, force_text_at, task_anchor
                ),
                EarlyStopSeverity::Normal => format!(
                    "[SYSTEM] You have used {} tokens in {} tool calls. If you have \
                     enough information, stop calling tools and respond now with your \
                     findings about the user's request (hard limit: {} calls).{}",
                    task_tokens_used, total_tool_calls_attempted, force_text_at, task_anchor
                ),
            },
            Self::ForceTextToolLimitReached {
                force_text_at,
                force_task_anchor,
                activity_section,
            } => format!(
                "[SYSTEM] Tool limit reached ({} calls). No more tool calls available.\n\
                 {}{}\
                 You MUST now respond with a concise summary:\n\
                 1. What you accomplished (files modified, bugs fixed, features added)\n\
                 2. What remains unfinished and why\n\
                 3. Any test results or verification status\n\n\
                 Do NOT restate the original task or say what you would do next. \
                 Do NOT answer questions from old conversation history. \
                 Focus only on concrete results and outcomes for the CURRENT task.",
                force_text_at, force_task_anchor, activity_section
            ),
            Self::EditStallWriteFileHint => "[SYSTEM] You have failed edit_file 3+ times in a row. The old_text is not matching the actual file content. STOP using edit_file. Instead:\n1. Use `read_file` to see the CURRENT file content\n2. Use `write_file` to rewrite the ENTIRE file with all your changes applied\n\nwrite_file is more reliable than edit_file when the file has been modified.".to_string(),
            Self::DuplicateSendFileAlreadySent => "[SYSTEM] The requested file was already sent in this task. Stop calling send_file and reply with plain text only.".to_string(),
            Self::HardPolicyToolBudgetReached { policy_tool_budget } => format!(
                "[SYSTEM] Hard tool budget reached ({} calls). No more tool calls available.\n\n\
                 You MUST now respond with a concise summary:\n\
                 1. What you accomplished (files modified, bugs fixed, features added)\n\
                 2. What remains unfinished and why\n\
                 3. Any test results or verification status\n\n\
                 Do NOT restate the original task or say what you would do next. \
                 Focus only on concrete results and outcomes.",
                policy_tool_budget
            ),
            Self::PersonalMemoryRecheckLimitReached => "[SYSTEM] You already performed the allowed targeted memory re-check(s). Stop calling tools and answer directly with what you know.".to_string(),
            Self::ScopeLockBlocked { tool_name, reason } => format!(
                "[SYSTEM] The previous `{}` tool call was blocked by deterministic scope locks ({}). Use paths/tool args aligned with the current request scope.",
                tool_name, reason
            ),
            Self::ArgumentContractBlocked {
                tool_name,
                reason,
                coaching,
            } => format!(
                "[SYSTEM] The previous `{}` tool call was blocked ({}). {}",
                tool_name, reason, coaching
            ),
            Self::HardToolLimitReached => "[SYSTEM] Tool limit reached. No more tool calls available.\n\n\
                 You MUST now respond with a concise summary:\n\
                 1. What you accomplished (files modified, bugs fixed, features added)\n\
                 2. What remains unfinished and why\n\
                 3. Any test results or verification status\n\n\
                 Do NOT restate the original task or say what you would do next. \
                 Focus only on concrete results and outcomes.".to_string(),
            Self::BackgroundHandoff {
                notifications_active,
            } => {
                if *notifications_active {
                    "[SYSTEM] A background task is now running and completion notifications are enabled. Do NOT call additional tools or poll status in this turn. Reply to the user now that work continues in background and results will be sent automatically.".to_string()
                } else {
                    "[SYSTEM] A background task was moved to the background. Do NOT call additional tools or poll status in this turn. Reply to the user now with the current status.".to_string()
                }
            }
            Self::SchedulingOwnerOnly => "[SYSTEM] Scheduling goals is owner-only. Handle this request directly without creating a goal.".to_string(),
            Self::KnowledgeIntentDirectAnswer => "[SYSTEM] Consultant classified this turn as knowledge. Provide the best direct answer now. Use tools only if needed to verify or retrieve missing facts.".to_string(),
            Self::DelegationModeActive => "[SYSTEM] Delegation mode active. Use `cli_agent` for execution tasks. `terminal`, `browser`, and `run_command` are hidden in this turn.".to_string(),
            Self::GoalCreationOwnerOnly => "[SYSTEM] Creating goals is owner-only. Handle this request directly without creating a goal.".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EarlyStopSeverity, SystemDirective};

    #[test]
    fn background_handoff_render_matches_notification_state() {
        let with_notify = SystemDirective::BackgroundHandoff {
            notifications_active: true,
        }
        .render();
        let without_notify = SystemDirective::BackgroundHandoff {
            notifications_active: false,
        }
        .render();

        assert!(with_notify.contains("completion notifications are enabled"));
        assert!(without_notify.contains("moved to the background"));
        assert!(!without_notify.contains("results will be sent automatically"));
    }

    #[test]
    fn route_failsafe_render_is_stable() {
        assert!(SystemDirective::RouteFailsafeActive
            .render()
            .contains("Route fail-safe is active"));
    }

    #[test]
    fn duplicate_send_file_render_matches_previous_text() {
        assert_eq!(
            SystemDirective::DuplicateSendFileAlreadySent.render(),
            "[SYSTEM] The requested file was already sent in this task. Stop calling send_file and reply with plain text only."
        );
    }

    #[test]
    fn force_text_tool_limit_render_preserves_sections() {
        let rendered = SystemDirective::ForceTextToolLimitReached {
            force_text_at: 40,
            force_task_anchor: "User's request: fix tests\n\n".to_string(),
            activity_section:
                "\nHere is what you actually did (use this as ground truth):\n1. terminal(...)\n"
                    .to_string(),
        }
        .render();
        assert_eq!(
            rendered,
            "[SYSTEM] Tool limit reached (40 calls). No more tool calls available.\n\
                 User's request: fix tests\n\n\
\nHere is what you actually did (use this as ground truth):\n1. terminal(...)\n\
                 You MUST now respond with a concise summary:\n\
                 1. What you accomplished (files modified, bugs fixed, features added)\n\
                 2. What remains unfinished and why\n\
                 3. Any test results or verification status\n\n\
                 Do NOT restate the original task or say what you would do next. \
                 Do NOT answer questions from old conversation history. \
                 Focus only on concrete results and outcomes for the CURRENT task."
        );
    }

    #[test]
    fn early_stop_urgency_render_matches_previous_text() {
        let rendered = SystemDirective::EarlyStopUrgency {
            task_tokens_used: 1200,
            total_tool_calls_attempted: 15,
            force_text_at: 40,
            task_anchor: "\nCurrent task: fix the parser".to_string(),
            severity: EarlyStopSeverity::Important,
        }
        .render();
        assert_eq!(
            rendered,
            "[SYSTEM] IMPORTANT: You have used 1200 tokens in 15 tool calls. \
                     You MUST stop calling tools soon and respond about the user's request. \
                     Hard limit for this task is 40 tool calls.\nCurrent task: fix the parser"
        );
    }
}
