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
    TruncationRecoveryTextContinuation {
        truncated_tail: String,
    },
    ToolModeDisabledPlainText,
    ApproachPivotRequired {
        attempt: usize,
        failure_record: String,
    },
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
    GlobalDailyBudgetAutoExtended {
        old_budget: i64,
        new_budget: i64,
        extension: usize,
        max_extensions: usize,
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
    CompletionVerificationRequired {
        target_hint: Option<String>,
    },
    DeferredToolCallRequired,
    DeferredProvideConcreteResults,
    StructuredToolResultSynthesis {
        tool_name: String,
        excerpt: String,
    },
    SuccessfulToolEvidenceMustBeUsed,
    EvidenceGroundingRequired,
    LiveWorkPivotRequired,
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
    ResearchSynthesisNudge {
        consecutive_searches: usize,
    },
    MemorySearchSaturation {
        consecutive_memory_calls: usize,
    },
    EditStallWriteFileHint,
    BuildFixCycleNudge,
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
    /// A specific tool hit its per-tool call limit but other tools remain
    /// available. The model should switch to a different tool.
    SpecificToolBlocked {
        tool_name: String,
    },
    BackgroundHandoff {
        notifications_active: bool,
    },
    /// A background process was launched but the user's request has more
    /// steps (e.g., "start server in background, then test it with curl").
    /// The agent should continue working on the remaining steps.
    BackgroundProcessContinue,
    SchedulingOwnerOnly,
    DelegationModeActive,
    GoalCreationOwnerOnly,
    ReflectionDiagnosis {
        tool_name: String,
        root_cause: String,
        recommended_action: String,
    },
    /// Injected immediately after an external mutation tool call fails.
    ExternalMutationFailed {
        tool_name: String,
        status_code: Option<u16>,
        error_hint: String,
    },
    /// Injected before final response when the outcome ledger has mixed results.
    OutcomeReconciliation(String),
    /// Task plan context injected each iteration so the model sees its plan.
    TaskPlanContext(String),
    /// The completion contract expects a file mutation (write/rewrite/create)
    /// but no write_file or edit_file tool was called.  Nudge the model to
    /// complete the requested file modification before declaring completion.
    MutationStillRequired,
    /// The model's final answer was mostly harness scaffolding (tool-result
    /// envelopes, [SYSTEM] notices) and was gutted by sanitization. Ask it to
    /// restate the answer in plain language once before any deterministic
    /// fallback ships.
    FinalAnswerRejectedInternalMarkers,
    /// The model produced a very short response after significant work (many
    /// tool calls) for a multi-part request. Nudge it to provide a comprehensive
    /// response addressing all parts of the user's request.
    ResponseQualityNudge {
        user_text_hint: String,
    },
    /// The model asked the user to upload/provide a file it can locate
    /// itself with search tools. Injected once per turn to force a retry.
    LocateFileInsteadOfAsking {
        user_text_hint: String,
    },
    /// The candidate final reply enumerates name-like list entries that
    /// appear in no tool output (and not in the user's message) this turn —
    /// the signature of fabricated list content (e.g. invented roster
    /// members). Injected once per turn to force a grounded rewrite.
    UngroundedListEntities {
        entities: Vec<String>,
    },
    /// The candidate reply denies or asserts a specific personal fact about an
    /// entity the user named, but no memory lookup for that entity grounded this
    /// turn. Force a search before the denial/assertion is allowed through.
    UnsearchedEntityDenial {
        entities: Vec<String>,
    },
    /// The candidate final reply is an enumeration answer built from web
    /// research, but fewer than two source pages were successfully read
    /// (0 = snippets only). Injected once per turn: fetch a second
    /// independent source, or explicitly caveat the single-sourcing.
    SingleSourceEnumeration {
        sources_read: usize,
    },
    /// N consecutive tool calls returned nothing. The search term is likely
    /// wrong — reorient instead of repeating or concluding absence.
    EmptyResultStreakVaryTerms {
        streak: usize,
    },
    /// The user approved the assistant's own immediately-prior proposal.
    /// Anchor the turn to that proposal so the model executes exactly it.
    ApprovedProposalAnchor {
        proposal: String,
    },
    /// Injected when plan detection heuristics identify a multi-step task
    /// that benefits from structured execution with verification.
    PlanSuggestion {
        hint: String,
    },
    /// User challenged the immediately previous answer ("Are you sure?").
    /// Anchor the model to that exchange only.
    ReaffirmationChallengeAnchor {
        prior_user_request: Option<String>,
        prior_assistant_reply: String,
    },
    /// A follow-up question carries its person referent only via a pronoun
    /// ("...what can you infer about her?"). Anchor the pronoun to the prior
    /// exchange's subject and force a memory lookup so it is not bound to a
    /// salient pinned-profile person (e.g. the pinned partner).
    CoreferenceGroundingRequired {
        prior_user_request: Option<String>,
        prior_assistant_reply: String,
    },
    /// The model is trying to finish but its self-registered requirement
    /// checklist still has unchecked items. Soft verification: injected once per
    /// task, then the turn is allowed to finish regardless.
    ChecklistVerificationRequired {
        items: Vec<String>,
    },
}

impl SystemDirective {
    pub(in crate::agent) fn render(&self) -> String {
        match self {
            Self::RouteFailsafeActive => "[SYSTEM] Route fail-safe is active for this session. Use explicit tools/results, avoid direct-return shortcuts, and prioritize concrete execution evidence.".to_string(),
            Self::FreshConversationContext => "This is a fresh conversation context. There are no previous tasks. Focus exclusively on the current user request. Do not reference or repeat tool calls from any prior context.".to_string(),
            Self::EmptyResponseRetry => "[SYSTEM] Your previous reply was empty (no text and no tool calls). This retry is running with reduced conversation history to recover. You MUST either (1) call the required tools, or (2) reply with a concrete blocker and the missing info. Do NOT return an empty response.".to_string(),
            Self::TruncationRecoveryUseWriteFile => "[SYSTEM] Your previous response was cut off because it exceeded the maximum output token limit. Do NOT generate long content inline. Choose by where the content lives: (1) If the content already exists in a file (e.g. a spilled tool result or fetched data), do NOT regenerate it — extract or format the part the user needs into a clean file with a tool (terminal/grep), then deliver it with the send_file tool. (2) If you must author the content yourself (a long report, a large code file), write it in chunks: call write_file with the first chunk, then call write_file with mode=\"append\" for each additional chunk. Keep your direct reply to the user brief.".to_string(),
            Self::TruncationRecoveryTextContinuation { truncated_tail } => format!(
                "[SYSTEM] Your previous text response was cut off mid-sentence due to output token limits. \
                 The partial response has been saved. Continue your response from EXACTLY where it was cut off. \
                 Your response was cut off at: \"...{}\"\n\n\
                 IMPORTANT: Start your continuation directly from the cutoff point. Do NOT repeat content \
                 that was already generated. Keep the continuation brief and complete the thought.",
                truncated_tail
            ),
            Self::ToolModeDisabledPlainText => "[SYSTEM] Tool mode is disabled for this turn. Respond with plain text only. Do NOT emit tool calls.".to_string(),
            Self::ApproachPivotRequired {
                attempt,
                failure_record,
            } => format!(
                "[SYSTEM] Your current approach is not working — do NOT retry it. \
                 This is approach pivot #{} of this task.\n\n{}\n\
                 Choose a FUNDAMENTALLY different method to accomplish the user's \
                 original request: different tools, a different strategy, or a \
                 different order of operations. Account for the state-changing \
                 actions already performed — verify before re-doing anything.",
                attempt, failure_record
            ),
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
            Self::GlobalDailyBudgetAutoExtended {
                old_budget,
                new_budget,
                extension,
                max_extensions,
            } => format!(
                "[SYSTEM] Global daily token budget auto-extended from {} to {} ({}/{} extensions). \
                 Continue working.",
                old_budget, new_budget, extension, max_extensions
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
            Self::CompletionVerificationRequired { target_hint } => {
                let target = target_hint
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .map(|value| format!(" against {}", value))
                    .unwrap_or_default();
                format!(
                    "[SYSTEM] You have not yet verified the requested outcome{}. Before answering, run a read-only verification step that checks the actual result. If you changed something, re-check after the change. Do NOT claim success until that verification is done.",
                    target
                )
            }
            Self::DeferredToolCallRequired => "[SYSTEM] HARD REQUIREMENT: your next reply MUST include at least one tool call. Do NOT return planning text like \"I'll do X\". Text-only replies are invalid for this request.".to_string(),
            Self::DeferredProvideConcreteResults => "[SYSTEM] You narrated future work instead of providing results. Execute any remaining required tools, or return concrete outcomes and blockers now.".to_string(),
            Self::StructuredToolResultSynthesis { tool_name, excerpt } => format!(
                "[SYSTEM] You already have the structured result from `{}`. Do NOT call more tools unless verification is still genuinely required. Summarize only what this result actually shows. For any tool-derived claim, only cite filenames, paths, status codes, errors, IDs, values, counts, test names, field names, or other specifics that appear in the excerpt. If any detail is missing or ambiguous, say so instead of inferring it.\n\nResult excerpt:\n{}",
                tool_name, excerpt
            ),
            Self::SuccessfulToolEvidenceMustBeUsed => "[SYSTEM] You already have successful live tool results in this turn. Do NOT claim you cannot browse, access current data, or only provide guidance. Use the actual tool results already in context and answer with concrete findings now.".to_string(),
            Self::EvidenceGroundingRequired => "[SYSTEM] The user is challenging whether a previously mentioned result, error, or detail was real. Do NOT defend prior assistant prose from memory. Only claim filenames, paths, status codes, errors, IDs, values, counts, test names, lines, field names, or other specifics if they appear in actual tool evidence already in context. Quote the exact line when helpful. If the evidence is partial, ambiguous, or unavailable, say that plainly and ask to re-check rather than inferring missing details.".to_string(),
            Self::LiveWorkPivotRequired => "[SYSTEM] You summarized failed live attempts instead of completing the request. Do NOT stop with a \"What I tried\" / \"Current status\" summary while tools still remain. Change strategy now: if an API call returned HTTP 4xx or bad parameters, simplify the request, use `http_request` for APIs, keep `web_fetch` for readable pages only, or fall back to `web_search`/site search/browser and then answer with concrete findings.".to_string(),
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
                 You already have enough information from your previous reads. \
                 Answer the user now, or use the appropriate non-read action tool if the request \
                 requires an action. Do not claim information is unavailable merely because read \
                 tools were removed.",
                read_desc
            ),
            Self::ReadSaturationWarning { consecutive_reads } => format!(
                "[SYSTEM] WARNING: You have called read-only tools {} times in a row. \
                 STOP reading and use the information already available. \
                 Answer the user now, or use an appropriate action tool only if the request \
                 requires one. Do NOT call read_file again. \
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
                 IMPORTANT: First, ANSWER any questions the user asked — check their request \
                 carefully and respond to every part you can from what you already know or discovered.\n\
                 Then briefly summarize:\n\
                 1. What you accomplished (files modified, bugs fixed, features added)\n\
                 2. What remains unfinished and why\n\n\
                 Do NOT list iteration numbers or raw tool names in your response. \
                 Do NOT promise future actions like \"let me try...\" — your tools have been disabled.\n\
                 Write a natural, user-friendly response — not a system log.",
                force_text_at, force_task_anchor, activity_section
            ),
            Self::ResearchSynthesisNudge { consecutive_searches } => format!(
                "[SYSTEM] You have done {} consecutive web searches. \
                 PAUSE and evaluate: do you have enough information to answer the user's question?\n\n\
                 - If YES: Stop searching and synthesize a comprehensive response from the evidence you already gathered.\n\
                 - If NO: Continue searching, but use a DIFFERENT search strategy (different keywords, a specific source, or web_fetch on a promising URL from your results).\n\n\
                 Most questions can be answered well with 2-3 good searches. More searches with similar queries \
                 will return similar results. Synthesize what you have rather than searching for perfection.",
                consecutive_searches
            ),
            Self::MemorySearchSaturation { consecutive_memory_calls } => format!(
                "[SYSTEM] You have called memory tools {} times in a row. \
                 STOP searching memory and RESPOND to the user NOW.\n\n\
                 You already have the information you need from your earlier searches. \
                 Synthesize what you found and compose your reply. \
                 If you stored new facts, confirm what was stored. \
                 If you searched for existing facts, share what you found.\n\n\
                 Do NOT call manage_memories or remember_fact again.",
                consecutive_memory_calls
            ),
            Self::EditStallWriteFileHint => "[SYSTEM] You have failed edit_file 3+ times in a row. The old_text is not matching the actual file content. STOP using edit_file. Instead:\n1. Use `read_file` to see the CURRENT file content\n2. Use `write_file` to rewrite the ENTIRE file with all your changes applied\n\nwrite_file is more reliable than edit_file when the file has been modified.".to_string(),
            Self::BuildFixCycleNudge => "[SYSTEM] DETECTED: Build-fix cycle. You have been alternating between editing files and running build/test commands many times without converging. STOP and take a different approach:\n1. Use `read_file` to see the CURRENT state of the file\n2. Think carefully about ALL the errors at once\n3. Use `write_file` to rewrite the ENTIRE file with ALL fixes applied in one shot\n4. Only then run the build/test command ONCE\n\nDo NOT continue making incremental edits — rewrite the file completely.".to_string(),
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
            Self::SpecificToolBlocked { tool_name } => format!(
                "[SYSTEM] The `{}` tool has reached its call limit for this task and is no longer available. \
                 However, your OTHER tools (write_file, edit_file, terminal, etc.) are still fully available. \
                 Continue working on the task using your remaining tools. Do NOT give up or summarize — \
                 proceed with the next step of the user's request.",
                tool_name
            ),
            Self::BackgroundHandoff {
                notifications_active,
            } => {
                if *notifications_active {
                    "[SYSTEM] A background task is now running and completion notifications are enabled. Do NOT call additional tools or poll status in this turn. Reply to the user now that work continues in background and results will be sent automatically.".to_string()
                } else {
                    "[SYSTEM] A background task was moved to the background. Do NOT call additional tools or poll status in this turn. Reply to the user now with the current status.".to_string()
                }
            }
            Self::BackgroundProcessContinue => "[SYSTEM] A background process was launched successfully and is now running. Continue with the remaining steps of the user's request (e.g., testing endpoints, verifying output). The background process is already running — proceed directly with the next action.".to_string(),
            Self::SchedulingOwnerOnly => "[SYSTEM] Scheduling goals is owner-only. Handle this request directly without creating a goal.".to_string(),
            Self::DelegationModeActive => "[SYSTEM] Delegation mode active. Use `cli_agent` for execution tasks. `terminal`, `browser`, and `run_command` are hidden in this turn.".to_string(),
            Self::GoalCreationOwnerOnly => "[SYSTEM] Creating goals is owner-only. Handle this request directly without creating a goal.".to_string(),
            Self::ReflectionDiagnosis {
                tool_name,
                root_cause,
                recommended_action,
            } => format!(
                "[SYSTEM] SELF-DIAGNOSIS for `{}`: {}.\n\
                 ACTION REQUIRED: {}.\n\
                 Do NOT repeat the same failing approach. \
                 If you cannot fix the issue, report the actual error honestly to the user.",
                tool_name, root_cause, recommended_action
            ),
            Self::ExternalMutationFailed {
                tool_name,
                status_code,
                error_hint,
            } => {
                let status_part = status_code
                    .map(|c| format!(" (HTTP {})", c))
                    .unwrap_or_default();
                format!(
                    "[SYSTEM] The previous `{}`{} FAILED: {}. \
                     Do NOT proceed as if it succeeded. Either retry with corrected \
                     parameters, or acknowledge the failure in your response.",
                    tool_name, status_part, error_hint
                )
            }
            Self::OutcomeReconciliation(summary) => summary.clone(),
            Self::TaskPlanContext(plan) => plan.clone(),
            Self::MutationStillRequired => "[SYSTEM] INCOMPLETE: Your request requires modifying or creating a file, but you have NOT called write_file or edit_file yet. You have the information from your reads — now WRITE the file. Use write_file to save the result, then provide a brief summary of what you changed.".to_string(),
            Self::FinalAnswerRejectedInternalMarkers => "[SYSTEM] Your previous answer was rejected: it consisted of internal tool-envelope markers instead of an answer. Do NOT quote tool output wrappers, [SYSTEM] lines, or bracketed markers. In plain language, state your final answer to the user's request now — what you found or did, and any honest limitation (e.g. the requested item was not found).".to_string(),
            Self::ResponseQualityNudge { user_text_hint } => format!(
                "[SYSTEM] Your response was too brief and did not address the user's full request. \
                 The user asked: \"{}\"\n\n\
                 You completed significant work using multiple tools. Now write a comprehensive response that:\n\
                 1. Explains WHAT you did (each change/action)\n\
                 2. Explains WHY you made each choice\n\
                 3. Shows relevant results (test output, file paths, etc.)\n\
                 4. Answers any specific questions the user asked\n\n\
                 Do NOT print raw tool-call syntax like call:terminal or <|tool_call>. \
                 If you still need a tool, call it using the structured tool interface. \
                 Otherwise, write a clear, structured response.",
                user_text_hint
            ),
            Self::EmptyResultStreakVaryTerms { streak } => format!(
                "[SYSTEM] {} consecutive tool calls returned no results. An empty result for one \
                 search term is NOT evidence of absence — the term is probably wrong. Before \
                 searching again: (1) re-read the earlier messages in this conversation — your own \
                 prior replies may already contain the exact filename, path, or answer; use it \
                 directly. (2) vary the term: substrings, initials, abbreviations, fewer words \
                 (e.g. files named with 'NC' will not match 'non-compete'). (3) filename listings \
                 (ls/find/glob) do NOT match file contents — use a content search (grep -r, \
                 mdfind) when the term may only appear inside the file.",
                streak
            ),
            Self::ApprovedProposalAnchor { proposal } => format!(
                "[SYSTEM] The user just approved YOUR OWN proposal quoted below. Execute exactly \
                 that — do not re-plan, re-search, or substitute a different action. If the \
                 proposal names a specific file, command, or target, act on that exact target now.\n\
                 Your proposal: \"{}\"",
                proposal
            ),
            Self::SingleSourceEnumeration { sources_read } => {
                let basis = if *sources_read == 0 {
                    "search snippets only — no source page was successfully read"
                } else {
                    "a single source page"
                };
                format!(
                    "[SYSTEM] CORROBORATION CHECK: your enumerated answer rests on {}. \
                     Before finalizing, do ONE of the following: (a) fetch one more \
                     independent source page with web_fetch — pick a different domain \
                     from your search results — and confirm the entries; or (b) state \
                     explicitly in your reply that the answer is based on {} and may be \
                     incomplete, naming the source(s) you used. Do not present an \
                     uncorroborated enumeration as definitive.",
                    basis,
                    if *sources_read == 0 {
                        "search snippets only"
                    } else {
                        "a single source"
                    },
                )
            }
            Self::UngroundedListEntities { entities } => format!(
                "[SYSTEM] GROUNDING CHECK FAILED: your draft reply lists entries that do not \
                 appear in ANY tool output from this task: {}. Inventing list entries is a \
                 critical error. Rewrite your answer using ONLY items literally present in \
                 the tool results you received. If the data you gathered does not contain \
                 the complete list, fetch a source page with web_fetch first, or state \
                 exactly which part you could not verify instead of filling the gaps.",
                entities.join(", ")
            ),
            Self::UnsearchedEntityDenial { entities } => format!(
                "[SYSTEM] GROUNDING CHECK: your draft answers a question about {} but you did \
                 not search memory for it this turn. Call manage_memories (and manage_people if \
                 available) for {} BEFORE answering. If you genuinely find nothing after \
                 searching, say so plainly — do not assert or deny a relationship you did not \
                 look up, and never assume a partner is a child's biological parent; phrase any \
                 such inference tentatively.",
                entities.join(", "),
                entities.join(", ")
            ),
            Self::LocateFileInsteadOfAsking { user_text_hint } => format!(
                "[SYSTEM] The user referenced a file by name. Do NOT ask the user to upload it or \
                 provide a path — you have tools to locate it yourself. \
                 Call search_files with a glob built from the filename; if the default directory \
                 has no match, retry with \"path\": \"~\" to search the home directory. \
                 Then read the file and answer the user's question. \
                 User request: \"{}\"",
                user_text_hint
            ),
            Self::PlanSuggestion { hint } => hint.clone(),
            Self::ReaffirmationChallengeAnchor {
                prior_user_request,
                prior_assistant_reply,
            } => {
                let mut lines = vec![
                    "[SYSTEM] REAFFIRMATION CHALLENGE: The user is challenging your immediately previous answer, not an earlier topic.".to_string(),
                    "Reply ONLY about the exchange below. Do NOT mention unrelated earlier topics, tools, goals, or session activity.".to_string(),
                ];
                if let Some(request) = prior_user_request {
                    lines.push(format!(
                        "Prior user request: \"{}\"",
                        crate::utils::truncate_str(request, 240)
                    ));
                }
                lines.push(format!(
                    "Your prior answer: \"{}\"",
                    crate::utils::truncate_str(prior_assistant_reply, 400)
                ));
                lines.push(
                    "Either reaffirm that answer with brief justification, or correct it if you were wrong. Keep the reply focused on this exchange only.".to_string(),
                );
                lines.join("\n")
            }
            Self::CoreferenceGroundingRequired {
                prior_user_request,
                prior_assistant_reply,
            } => {
                let mut lines = vec![
                    "[SYSTEM] PRONOUN REFERENT: The user's pronoun (\"her\"/\"him\"/\"she\"/\"he\") refers to the subject of the exchange below — NOT to anyone in your pinned core profile (e.g. the pinned partner or children). Do NOT assume it means the partner.".to_string(),
                ];
                if let Some(request) = prior_user_request {
                    lines.push(format!(
                        "Immediately prior user message: \"{}\"",
                        crate::utils::truncate_str(request, 240)
                    ));
                }
                lines.push(format!(
                    "Your prior answer: \"{}\"",
                    crate::utils::truncate_str(prior_assistant_reply, 400)
                ));
                lines.push(
                    "Before answering, call manage_memories to look up that specific person. If you find nothing, say so plainly instead of substituting a different person.".to_string(),
                );
                lines.join("\n")
            }
            Self::ChecklistVerificationRequired { items } => format!(
                "[SYSTEM] You are about to finish, but these tracked requirements are still \
                 unchecked:\n{}\n\
                 Complete them now — for a file deliverable, call send_file with the produced \
                 file's path. If you are intentionally skipping any, call track_requirements and \
                 mark it 'deferred' with a brief reason. Do not finish while a required item is \
                 still pending.",
                items
                    .iter()
                    .map(|i| format!("- {i}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
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
    fn checklist_verification_render_lists_items_and_steers_send_file() {
        let rendered = SystemDirective::ChecklistVerificationRequired {
            items: vec!["send the latency file to the user".to_string()],
        }
        .render();
        assert!(rendered.contains("[SYSTEM]"));
        assert!(rendered.contains("send the latency file to the user"));
        assert!(rendered.contains("send_file"));
        assert!(rendered.contains("deferred"));
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
    fn successful_tool_evidence_render_mentions_live_results() {
        let rendered = SystemDirective::SuccessfulToolEvidenceMustBeUsed.render();
        assert!(rendered.contains("successful live tool results"));
        assert!(rendered.contains("answer with concrete findings now"));
    }

    #[test]
    fn evidence_grounding_render_requires_exact_evidence() {
        let rendered = SystemDirective::EvidenceGroundingRequired.render();
        assert!(rendered.contains("result, error, or detail"));
        assert!(rendered.contains("actual tool evidence"));
        assert!(rendered.contains("rather than inferring"));
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
                 IMPORTANT: First, ANSWER any questions the user asked — check their request \
                 carefully and respond to every part you can from what you already know or discovered.\n\
                 Then briefly summarize:\n\
                 1. What you accomplished (files modified, bugs fixed, features added)\n\
                 2. What remains unfinished and why\n\n\
                 Do NOT list iteration numbers or raw tool names in your response. \
                 Do NOT promise future actions like \"let me try...\" — your tools have been disabled.\n\
                 Write a natural, user-friendly response — not a system log."
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

    #[test]
    fn reflection_diagnosis_render_includes_root_cause_and_action() {
        let rendered = SystemDirective::ReflectionDiagnosis {
            tool_name: "http_request".to_string(),
            root_cause: "Using the wrong hostname for the API".to_string(),
            recommended_action: "Change the base URL to https://example.com/api/v2".to_string(),
        }
        .render();

        assert!(rendered.contains("SELF-DIAGNOSIS"));
        assert!(rendered.contains("http_request"));
        assert!(rendered.contains("wrong hostname"));
        assert!(rendered.contains("Change the base URL"));
        assert!(rendered.contains("Do NOT repeat the same failing approach"));
    }

    #[test]
    fn external_mutation_failed_directive_renders() {
        let directive = SystemDirective::ExternalMutationFailed {
            tool_name: "http_request".to_string(),
            status_code: Some(403),
            error_hint: "duplicate content".to_string(),
        };
        let rendered = directive.render();
        assert!(rendered.contains("[SYSTEM]"));
        assert!(rendered.contains("FAILED"));
        assert!(rendered.contains("403"));
        assert!(rendered.contains("http_request"));
        assert!(rendered.contains("Do NOT proceed as if it succeeded"));
    }

    #[test]
    fn truncation_text_continuation_render_includes_tail() {
        let rendered = SystemDirective::TruncationRecoveryTextContinuation {
            truncated_tail: "according to my".to_string(),
        }
        .render();
        assert!(rendered.contains("[SYSTEM]"));
        assert!(rendered.contains("cut off mid-sentence"));
        assert!(rendered.contains("according to my"));
        assert!(rendered.contains("Continue your response"));
        assert!(rendered.contains("Do NOT repeat content"));
    }

    #[test]
    fn mutation_still_required_render_mentions_write_file() {
        let rendered = SystemDirective::MutationStillRequired.render();
        assert!(rendered.contains("[SYSTEM]"));
        assert!(rendered.contains("write_file"));
        assert!(rendered.contains("NOT called"));
    }

    #[test]
    fn outcome_reconciliation_directive_renders() {
        let directive =
            SystemDirective::OutcomeReconciliation("[SYSTEM] 1 of 2 attempts failed.".to_string());
        let rendered = directive.render();
        assert!(rendered.contains("1 of 2 attempts failed"));
    }

    #[test]
    fn reaffirmation_challenge_anchor_render_focuses_prior_exchange() {
        let rendered = SystemDirective::ReaffirmationChallengeAnchor {
            prior_user_request: Some("How many R's in strawberry?".to_string()),
            prior_assistant_reply: "There are 3 R's in strawberry.".to_string(),
        }
        .render();

        assert!(rendered.contains("REAFFIRMATION CHALLENGE"));
        assert!(rendered.contains("How many R's in strawberry?"));
        assert!(rendered.contains("There are 3 R's in strawberry."));
        assert!(rendered.contains("Do NOT mention unrelated earlier topics"));
    }

    #[test]
    fn coreference_grounding_render_anchors_pronoun_and_forces_lookup() {
        let rendered = SystemDirective::CoreferenceGroundingRequired {
            prior_user_request: Some("Where is my mom from?".to_string()),
            prior_assistant_reply: "I don't have info about Carol Mendez.".to_string(),
        }
        .render();

        assert!(rendered.contains("PRONOUN REFERENT"));
        assert!(rendered.contains("Where is my mom from?"));
        assert!(rendered.contains("Carol Mendez"));
        // Must steer away from the pinned partner and toward a lookup.
        assert!(rendered.contains("NOT"));
        assert!(rendered.contains("manage_memories"));
    }

    #[test]
    fn truncation_recovery_directive_covers_both_branches() {
        let text = SystemDirective::TruncationRecoveryUseWriteFile.render();
        // existing-file branch → deliver via send_file, don't regenerate
        assert!(text.contains("send_file"), "must mention send_file");
        // model-authored branch → append chunks
        assert!(
            text.contains("mode=\"append\""),
            "must mention append chunks"
        );
    }

    #[test]
    fn read_saturation_directives_are_task_neutral() {
        for rendered in [
            SystemDirective::ReadSaturationWarning {
                consecutive_reads: 5,
            }
            .render(),
            SystemDirective::ReadSaturationCritical {
                read_desc: "Repeated reads".to_string(),
            }
            .render(),
        ] {
            assert!(rendered.contains("Answer the user"));
            assert!(!rendered.contains("MUST use `write_file`"));
            assert!(!rendered.contains("Write the corrected code"));
        }
    }
}

#[cfg(test)]
mod unsearched_entity_tests {
    use super::*;

    #[test]
    fn renders_unsearched_entity_denial() {
        let d = SystemDirective::UnsearchedEntityDenial {
            entities: vec!["Caro".into()],
        };
        let msg = d.render();
        assert!(msg.contains("Caro"));
        assert!(msg.to_lowercase().contains("search"));
        assert!(msg.contains("biological"), "honesty clause must be present");
    }
}

#[cfg(test)]
mod corroboration_tests {
    use super::*;

    #[test]
    fn single_source_enumeration_directive_demands_second_source_or_caveat() {
        let zero = SystemDirective::SingleSourceEnumeration { sources_read: 0 }.render();
        assert!(zero.contains("CORROBORATION CHECK"));
        assert!(zero.contains("search snippets only"));
        assert!(zero.contains("web_fetch"));

        let one = SystemDirective::SingleSourceEnumeration { sources_read: 1 }.render();
        assert!(one.contains("a single source"));
        assert!(one.contains("different domain"));
    }
}
