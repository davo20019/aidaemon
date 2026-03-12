use super::*;
use crate::llm_markers::INTENT_GATE_MARKER;
use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FollowupMode {
    NewTask,
    Followup,
    ClarificationAnswer,
}

impl FollowupMode {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            FollowupMode::NewTask => "new_task",
            FollowupMode::Followup => "followup",
            FollowupMode::ClarificationAnswer => "clarification_answer",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TurnContextReason {
    DefaultNewTask,
    ExplicitFollowup,
    ClarificationAnswer,
    ContextDependentQuestion,
    FollowupOverrideStandalone,
    FollowupOverrideMismatchPreflight,
    CarryoverSanitized,
}

impl TurnContextReason {
    pub(super) fn as_code(&self) -> &'static str {
        match self {
            TurnContextReason::DefaultNewTask => "default_new_task",
            TurnContextReason::ExplicitFollowup => "explicit_followup",
            TurnContextReason::ClarificationAnswer => "clarification_answer",
            TurnContextReason::ContextDependentQuestion => "context_dependent_question",
            TurnContextReason::FollowupOverrideStandalone => "followup_override_standalone",
            TurnContextReason::FollowupOverrideMismatchPreflight => {
                "followup_override_mismatch_preflight"
            }
            TurnContextReason::CarryoverSanitized => "carryover_sanitized",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum CompletionTaskKind {
    #[default]
    Conversational,
    Answer,
    Compose,
    Check,
    Find,
    Change,
    Deliver,
    Schedule,
    Monitor,
    Diagnose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerificationTargetKind {
    Url,
    Path,
    ProjectScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerificationTarget {
    pub kind: VerificationTargetKind,
    pub value: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CompletionContract {
    pub task_kind: CompletionTaskKind,
    pub expects_mutation: bool,
    pub requires_observation: bool,
    pub requires_reverification_after_mutation: bool,
    pub explicit_verification_requested: bool,
    pub connected_content_mode: super::intent_routing::ConnectedContentMode,
    pub verification_targets: Vec<VerificationTarget>,
}

impl CompletionContract {
    pub(super) fn primary_target_hint(&self) -> Option<String> {
        self.verification_targets
            .first()
            .map(|target| target.value.clone())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CompletionProgress {
    pub observation_count: usize,
    pub mutation_count: usize,
    pub verification_count: usize,
    pub verification_pending: bool,
}

impl CompletionProgress {
    pub(super) fn new(contract: &CompletionContract) -> Self {
        Self {
            verification_pending: contract.requires_observation,
            ..Self::default()
        }
    }

    pub(super) fn mark_mutation(&mut self, contract: &CompletionContract) {
        self.mutation_count = self.mutation_count.saturating_add(1);
        if contract.requires_reverification_after_mutation {
            self.verification_pending = true;
        }
    }

    pub(super) fn mark_observation(&mut self, contract: &CompletionContract, matched_target: bool) {
        self.observation_count = self.observation_count.saturating_add(1);
        if !contract.requires_observation {
            return;
        }
        if matched_target || contract.verification_targets.is_empty() {
            self.verification_pending = false;
            self.verification_count = self.verification_count.saturating_add(1);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct TurnContext {
    pub goal_user_text: String,
    pub recent_messages: Vec<Value>,
    pub project_hints: Vec<String>,
    pub primary_project_scope: Option<String>,
    pub allow_multi_project_scope: bool,
    pub followup_mode: Option<FollowupMode>,
    pub reasons: Vec<TurnContextReason>,
    pub completion_contract: CompletionContract,
}

const GOAL_CONTEXT_RECENT_MESSAGES_LIMIT: usize = 6;
const GOAL_CONTEXT_HINT_HISTORY_LIMIT: usize = 30;
const GOAL_CONTEXT_MAX_PROJECT_HINTS: usize = 8;
const GOAL_CONTEXT_MAX_PROJECT_SCOPES: usize = 6;
static HTTP_URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\bhttps?://[^\s"'()<>]+"#).expect("valid http url regex"));

fn find_previous_turns(
    history: &[Message],
    current_user_text: &str,
) -> (Option<String>, Option<String>) {
    // History includes the current user message (already persisted). Walk backwards to find:
    // - previous assistant message (optional; usually a clarifying question)
    // - previous user message (the original request)
    let mut saw_current_user = false;
    let mut prev_assistant: Option<String> = None;
    let mut prev_user: Option<String> = None;
    for msg in history.iter().rev() {
        match msg.role.as_str() {
            "user" => {
                if !saw_current_user {
                    saw_current_user = true;
                    continue;
                }
                if let Some(content) = msg.content.as_deref() {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        prev_user = Some(trimmed.to_string());
                        break;
                    }
                }
            }
            "assistant" => {
                if saw_current_user && prev_assistant.is_none() {
                    if let Some(content) = msg.content.as_deref() {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case(current_user_text) {
                            prev_assistant = Some(trimmed.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    (prev_assistant, prev_user)
}

fn assistant_message_looks_like_clarifying_question(message: &str) -> bool {
    let trimmed = message.trim();
    if !trimmed.contains('?') {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let clarifying_markers = [
        "which",
        "what",
        "how",
        "do you want",
        "would you like",
        "should i",
        "can you clarify",
        "any specific",
        "do you prefer",
        "prefer",
        "what style",
        "what elements",
    ];
    clarifying_markers.iter().any(|m| lower.contains(m))
}

fn looks_like_explicit_task_switch(lower_text: &str) -> bool {
    lower_text.starts_with("new task")
        || lower_text.starts_with("different task")
        || lower_text.starts_with("instead ")
        || lower_text.starts_with("forget that")
        || lower_text.starts_with("ignore that")
}

fn looks_like_style_followup(lower_text: &str) -> bool {
    let style_markers = [
        "do what you consider",
        "do what you think",
        "as you see fit",
        "you decide",
        "your call",
        "best judgment",
        "best judgement",
        "whatever you think",
    ];
    style_markers.iter().any(|m| lower_text.contains(m))
}

fn looks_like_context_dependent_followup_question(lower_text: &str) -> bool {
    if !looks_like_question_request(lower_text) || lower_text.chars().count() > 160 {
        return false;
    }

    let status_followup_prefixes = [
        "did you ",
        "were you able",
        "have you ",
        "what did you find",
        "which one",
        "which ones",
        "what happened",
        "how many",
        "why did",
        "why was",
        "why were",
    ];
    let shared_context_markers = [
        "those",
        "these",
        "them",
        "earlier",
        "before",
        "previous",
        "last",
        "again",
        "already",
        "the api",
        "the result",
        "the results",
        "the output",
        "the response",
        "our chat",
        "the convo",
        "the conversation",
        "you just",
        "you sent",
        "you found",
        "you pulled",
    ];
    if status_followup_prefixes
        .iter()
        .any(|prefix| lower_text.starts_with(prefix))
    {
        return text_contains_any_phrase(lower_text, &shared_context_markers);
    }

    let explanation_followup_prefixes = [
        "what does",
        "what do",
        "can you explain",
        "could you explain",
        "can you clarify",
        "could you clarify",
        "can you help me understand",
        "could you help me understand",
    ];
    if !explanation_followup_prefixes
        .iter()
        .any(|prefix| lower_text.starts_with(prefix))
    {
        return false;
    }

    let deictic_markers = ["it", "that", "this", "these", "those", "them"];
    let explanation_cues = [
        "mean",
        "means",
        "explain",
        "clarify",
        "understand",
        "interpret",
    ];

    text_contains_any_phrase(lower_text, &explanation_cues)
        && (text_contains_any_phrase(lower_text, &deictic_markers)
            || text_contains_any_phrase(lower_text, &shared_context_markers))
}

fn looks_like_artifact_inspection_request(lower_text: &str) -> bool {
    let mentions_artifact = text_contains_any_phrase(
        lower_text,
        &[
            "doc",
            "document",
            "file",
            "attachment",
            "attached",
            "screenshot",
            "image",
            "photo",
            "picture",
            "pdf",
            "note",
        ],
    );
    if !mentions_artifact {
        return false;
    }

    text_contains_any_phrase(
        lower_text,
        &[
            "check", "read", "review", "inspect", "look at", "open", "analyze", "analyse", "fix",
            "issue", "problem", "error",
        ],
    )
}

fn looks_like_standalone_goal_request(lower_text: &str) -> bool {
    let word_count = lower_text.split_whitespace().count();
    if word_count < 8 {
        return false;
    }

    let asks_for_uninterrupted_execution = contains_keyword_as_words(lower_text, "dont ask")
        || contains_keyword_as_words(lower_text, "don't ask")
        || contains_keyword_as_words(lower_text, "without asking")
        || contains_keyword_as_words(lower_text, "just do it");

    let has_action_verb = [
        "compare",
        "analyze",
        "analyse",
        "build",
        "create",
        "write",
        "read",
        "parse",
        "scan",
        "search",
        "find",
        "clean",
        "delete",
        "install",
        "fix",
        "refactor",
        "audit",
        "summarize",
        "review",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(lower_text, kw));

    let has_scope_detail = lower_text.contains('/')
        || lower_text.contains(".json")
        || lower_text.contains("all my projects")
        || lower_text.contains("across all")
        || lower_text.contains("dependencies")
        || lower_text.contains("versions")
        || lower_text.contains("node_modules");

    (asks_for_uninterrupted_execution && has_action_verb)
        || (word_count >= 14 && has_action_verb && has_scope_detail)
}

fn payload_has_self_contained_detail(text: &str) -> bool {
    let meaningful_tokens = text
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-')
                .to_ascii_lowercase()
        })
        .filter(|token| !token.is_empty())
        .filter(|token| {
            !matches!(
                token.as_str(),
                "a" | "an"
                    | "the"
                    | "to"
                    | "for"
                    | "at"
                    | "on"
                    | "in"
                    | "of"
                    | "my"
                    | "your"
                    | "our"
                    | "me"
                    | "you"
                    | "it"
                    | "this"
                    | "that"
                    | "these"
                    | "those"
                    | "them"
                    | "task"
                    | "goal"
                    | "job"
                    | "schedule"
                    | "scheduled"
                    | "daily"
                    | "every"
                    | "day"
                    | "weekdays"
                    | "weekends"
                    | "recurring"
                    | "reminder"
                    | "remind"
                    | "set"
                    | "setup"
                    | "up"
                    | "create"
                    | "build"
                    | "write"
                    | "edit"
                    | "update"
                    | "delete"
                    | "remove"
                    | "deploy"
                    | "publish"
                    | "post"
                    | "send"
                    | "email"
                    | "message"
                    | "upload"
                    | "install"
                    | "connect"
                    | "enable"
                    | "disable"
                    | "restart"
                    | "reload"
                    | "commit"
                    | "push"
                    | "can"
                    | "could"
                    | "would"
                    | "please"
            )
        })
        .count();

    meaningful_tokens >= 3
}

fn looks_like_self_contained_mutation_request(current: &str, lower_text: &str) -> bool {
    let trimmed = current.trim();
    if trimmed.is_empty() || trimmed.split_whitespace().count() < 6 {
        return false;
    }

    if let Some((schedule_raw, _)) = crate::cron_utils::extract_schedule_from_text(trimmed) {
        let cleaned = crate::cron_utils::clean_task_description(trimmed, &schedule_raw);
        return payload_has_self_contained_detail(&cleaned);
    }

    let has_mutation_cue = [
        "set up", "setup", "create", "build", "write", "edit", "update", "delete", "remove",
        "deploy", "publish", "post", "send", "email", "message", "upload", "install", "connect",
        "enable", "disable", "restart", "reload", "commit", "push",
    ]
    .iter()
    .any(|phrase| contains_keyword_as_words(lower_text, phrase));
    if !has_mutation_cue {
        return false;
    }

    let has_structured_target = looks_like_short_command_request(trimmed)
        || HTTP_URL_RE.is_match(trimmed)
        || super::user_text_references_filesystem_path(trimmed)
        || super::text_has_explicit_project_scope_cues(lower_text)
        || lower_text.contains(".json")
        || lower_text.contains(".toml")
        || lower_text.contains(".yaml")
        || lower_text.contains(".yml")
        || lower_text.contains(".md")
        || lower_text.contains(".ts")
        || lower_text.contains(".js")
        || lower_text.contains(".rs");

    has_structured_target && payload_has_self_contained_detail(trimmed)
}

fn looks_like_short_command_request(current: &str) -> bool {
    let trimmed = current.trim();
    if trimmed.is_empty() || trimmed.ends_with('?') {
        return false;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() < 2 || words.len() > 12 {
        return false;
    }

    let first = words
        .first()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
                .to_ascii_lowercase()
        })
        .unwrap_or_default();
    if !matches!(
        first.as_str(),
        "run"
            | "build"
            | "deploy"
            | "publish"
            | "commit"
            | "push"
            | "post"
            | "restart"
            | "reload"
            | "check"
            | "inspect"
            | "debug"
            | "review"
            | "analyze"
            | "analyse"
            | "search"
            | "find"
            | "open"
            | "show"
    ) {
        return false;
    }

    let lower = trimmed.to_ascii_lowercase();
    let has_cli_token = [
        "build",
        "deploy",
        "publish",
        "commit",
        "push",
        "post",
        "wrangler",
        "npm",
        "pnpm",
        "yarn",
        "cargo",
        "pytest",
        "test",
        "tests",
        "git",
        "docker",
        "kubectl",
        "logs",
        "log",
        "restart",
        "reload",
        "server",
        "service",
        "branch",
        "repo",
        "repository",
        "diff",
        "migration",
        "migrations",
        "schema",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    let has_structured_target = words.iter().skip(1).any(|word| {
        let token = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_');
        !token.is_empty()
            && (token_looks_like_filesystem_path(token)
                || token.starts_with('-')
                || token.contains('.')
                || token.contains('/')
                || token.contains('\\')
                || token.chars().any(|c| c.is_ascii_digit()))
    }) || has_cli_token;
    if !has_structured_target {
        return false;
    }

    let deictic_only = words.iter().skip(1).all(|word| {
        matches!(
            word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'' && c != '-')
                .to_ascii_lowercase()
                .as_str(),
            "it" | "this" | "that" | "them" | "those" | "these" | "again" | "now" | "please"
        )
    });

    !deictic_only
}

fn looks_like_scope_carryover_ack(current: &str) -> bool {
    let trimmed = current.trim();
    if trimmed.is_empty() || trimmed.ends_with('?') || trimmed.chars().count() > 80 {
        return false;
    }

    let lower = trimmed.to_ascii_lowercase();
    [
        "yes",
        "go ahead",
        "do it",
        "continue",
        "keep going",
        "proceed",
        "sounds good",
        "ok",
        "okay",
        "sure",
        "same repo",
        "same project",
    ]
    .iter()
    .any(|phrase| contains_keyword_as_words(&lower, phrase))
}

fn looks_like_multi_project_request(lower_text: &str) -> bool {
    contains_keyword_as_words(lower_text, "all my projects")
        || contains_keyword_as_words(lower_text, "across all projects")
        || contains_keyword_as_words(lower_text, "across my projects")
        || contains_keyword_as_words(lower_text, "every project")
        || (contains_keyword_as_words(lower_text, "all projects")
            && contains_keyword_as_words(lower_text, "compare"))
}

fn text_contains_any_phrase(text: &str, phrases: &[&str]) -> bool {
    phrases
        .iter()
        .any(|phrase| contains_keyword_as_words(text, phrase))
}

fn should_allow_verification_nickname_scope(text: &str, token: &str) -> bool {
    super::should_allow_contextual_project_nickname_scope(text, token)
}

fn extract_verification_project_scopes(
    text: &str,
    scopes: &mut Vec<String>,
    max_scopes: usize,
    alias_roots: &[String],
) {
    for raw in text.split_whitespace() {
        if scopes.len() >= max_scopes {
            break;
        }
        let token = raw
            .trim_matches(|c: char| {
                c.is_ascii_whitespace()
                    || matches!(
                        c,
                        '`' | '\''
                            | '"'
                            | ','
                            | ';'
                            | ':'
                            | '.'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                    )
            })
            .trim();
        if token.is_empty() || token.contains("://") {
            continue;
        }

        let scope = if token_looks_like_filesystem_path(token) {
            normalize_project_scope_path_with_aliases(token, alias_roots)
        } else {
            should_allow_verification_nickname_scope(text, token)
                .then(|| {
                    crate::tools::fs_utils::resolve_named_project_root(token, alias_roots)
                        .or_else(|| {
                            crate::tools::fs_utils::resolve_contextual_project_nickname_in_explicit_roots(token, alias_roots)
                        })
                })
                .flatten()
                .map(|path| path.to_string_lossy().to_string())
        };

        if let Some(scope) = scope {
            push_project_scope(scopes, scope, max_scopes);
        }
    }
}

fn extract_verification_targets(text: &str, alias_roots: &[String]) -> Vec<VerificationTarget> {
    let mut targets = Vec::new();

    for capture in HTTP_URL_RE.captures_iter(text) {
        let raw = capture
            .get(0)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim_end_matches(['.', ',', ';', ')', ']', '}'])
            .to_string();
        if raw.is_empty()
            || targets.iter().any(|existing: &VerificationTarget| {
                existing.kind == VerificationTargetKind::Url && existing.value == raw
            })
        {
            continue;
        }
        targets.push(VerificationTarget {
            kind: VerificationTargetKind::Url,
            value: raw,
        });
    }

    let mut scopes = Vec::new();
    extract_verification_project_scopes(text, &mut scopes, 4, alias_roots);
    for scope in scopes {
        if targets.iter().any(|existing| existing.value == scope) {
            continue;
        }
        targets.push(VerificationTarget {
            kind: VerificationTargetKind::ProjectScope,
            value: scope,
        });
    }

    if targets.is_empty() && super::user_text_references_filesystem_path(text) {
        for raw in text.split_whitespace() {
            let token = raw
                .trim_matches(|c: char| {
                    c.is_ascii_whitespace()
                        || matches!(
                            c,
                            '`' | '\''
                                | '"'
                                | ','
                                | ';'
                                | ':'
                                | '.'
                                | '!'
                                | '?'
                                | '('
                                | ')'
                                | '['
                                | ']'
                                | '{'
                                | '}'
                        )
                })
                .trim();
            if token.is_empty() || token.contains("://") || !token_looks_like_filesystem_path(token)
            {
                continue;
            }
            if let Ok(path) = crate::tools::fs_utils::validate_path(token) {
                let value = path.to_string_lossy().to_string();
                if !targets.iter().any(|existing| existing.value == value) {
                    targets.push(VerificationTarget {
                        kind: VerificationTargetKind::Path,
                        value,
                    });
                }
            }
        }
    }

    targets
}

#[derive(Debug, Clone, Default)]
struct CompletionSignals {
    is_question: bool,
    asks_schedule: bool,
    asks_monitor: bool,
    asks_check: bool,
    asks_find: bool,
    asks_deliver: bool,
    asks_change: bool,
    asks_diagnose: bool,
    has_verification_target: bool,
    claimed_side_effect: bool,
    explicit_verification_requested: bool,
    observable_target_request: bool,
    visible_state_problem: bool,
}

fn looks_like_question_request(lower_text: &str) -> bool {
    lower_text.ends_with('?')
        || [
            "what ", "when ", "where ", "why ", "who ", "how ", "is ", "are ", "do ", "does ",
            "did ", "can ", "could ", "will ", "would ",
        ]
        .iter()
        .any(|prefix| lower_text.starts_with(prefix))
}

fn infer_completion_signals(
    lower_text: &str,
    verification_targets: &[VerificationTarget],
) -> CompletionSignals {
    let has_verification_target = !verification_targets.is_empty();
    let is_question = looks_like_question_request(lower_text);
    let asks_schedule = text_contains_any_phrase(
        lower_text,
        &[
            "remind me",
            "schedule",
            "set a reminder",
            "add reminder",
            "scheduled task",
            "scheduled goal",
            "recurring task",
            "recurring goal",
        ],
    ) || crate::cron_utils::extract_schedule_from_text(lower_text).is_some();
    let asks_monitor =
        text_contains_any_phrase(lower_text, &["monitor", "watch", "keep an eye on"]);
    let asks_check = text_contains_any_phrase(
        lower_text,
        &[
            "check",
            "verify",
            "confirm",
            "see if",
            "test whether",
            "test if",
            "is there",
            "do i have",
            "did it",
            "did you",
            "status",
        ],
    );
    let asks_find = text_contains_any_phrase(
        lower_text,
        &["find", "locate", "list", "show me", "search for", "look up"],
    );
    let asks_deliver = text_contains_any_phrase(
        lower_text,
        &[
            "send",
            "post this",
            "post it",
            "post to",
            "post on",
            "upload",
            "tweet",
            "email",
            "message",
            "share",
        ],
    );
    let asks_change = text_contains_any_phrase(
        lower_text,
        &[
            "change", "update", "edit", "write", "create", "delete", "remove", "deploy", "build",
            "connect", "set up", "setup", "install", "restart", "reload", "enable", "disable",
            "remember", "store", "save", "note",
        ],
    );
    let visible_state_problem = text_contains_any_phrase(
        lower_text,
        &[
            "still dont see",
            "still don't see",
            "not showing",
            "doesnt show",
            "doesn't show",
            "isnt showing",
            "isn't showing",
            "not visible",
            "missing from",
            "missing on",
            "broken on",
            "not working",
            "failed to load",
            "in production",
            "on the site",
            "on the page",
            "go live",
        ],
    );
    let asks_diagnose = visible_state_problem
        || text_contains_any_phrase(
            lower_text,
            &[
                "fix",
                "debug",
                "diagnose",
                "troubleshoot",
                "why is",
                "why isnt",
                "why isn't",
                "issue",
                "problem",
                "error",
                "fails to",
                "failing to",
            ],
        );
    let claimed_side_effect = text_contains_any_phrase(
        lower_text,
        &[
            "did it",
            "did you",
            "did that work",
            "did this work",
            "went through",
            "was it sent",
            "was it posted",
            "was it deployed",
        ],
    );
    let explicit_verification_requested = text_contains_any_phrase(
        lower_text,
        &[
            "verify",
            "confirm",
            "make sure",
            "double check",
            "double-check",
            "validate",
            "look it up",
            "look this up",
        ],
    );
    let observable_target_request = has_verification_target
        && text_contains_any_phrase(
            lower_text,
            &[
                "here",
                "there",
                "read",
                "open",
                "summarize",
                "show me",
                "what's on",
                "what is on",
                "what does",
                "what do you see",
                "in this file",
                "on this page",
                "on this site",
                "at this url",
                "at this link",
            ],
        );

    CompletionSignals {
        is_question,
        asks_schedule,
        asks_monitor,
        asks_check,
        asks_find,
        asks_deliver,
        asks_change,
        asks_diagnose,
        has_verification_target,
        claimed_side_effect,
        explicit_verification_requested,
        observable_target_request,
        visible_state_problem,
    }
}

fn infer_completion_task_kind(signals: &CompletionSignals) -> CompletionTaskKind {
    if signals.asks_schedule {
        return CompletionTaskKind::Schedule;
    }
    if signals.asks_monitor {
        return CompletionTaskKind::Monitor;
    }
    if signals.asks_diagnose {
        return CompletionTaskKind::Diagnose;
    }
    if signals.asks_deliver {
        return CompletionTaskKind::Deliver;
    }
    if signals.asks_change {
        return CompletionTaskKind::Change;
    }
    if signals.asks_check {
        return CompletionTaskKind::Check;
    }
    if signals.asks_find {
        return CompletionTaskKind::Find;
    }
    if signals.observable_target_request {
        return CompletionTaskKind::Answer;
    }
    if signals.is_question {
        return CompletionTaskKind::Answer;
    }

    CompletionTaskKind::Conversational
}

fn infer_completion_contract(text: &str, alias_roots: &[String]) -> CompletionContract {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return CompletionContract::default();
    }

    let verification_targets = extract_verification_targets(text, alias_roots);
    let signals = infer_completion_signals(&lower, &verification_targets);
    let connected_content_mode = super::intent_routing::classify_connected_content_mode(text);
    let mut task_kind = infer_completion_task_kind(&signals);
    if matches!(
        connected_content_mode,
        super::intent_routing::ConnectedContentMode::DraftOnly
    ) {
        task_kind = CompletionTaskKind::Compose;
    }

    let expects_mutation = if connected_content_mode.expects_live_delivery() {
        true
    } else if connected_content_mode.is_authoring_only() {
        false
    } else {
        matches!(
            task_kind,
            CompletionTaskKind::Change
                | CompletionTaskKind::Deliver
                | CompletionTaskKind::Schedule
                | CompletionTaskKind::Monitor
                | CompletionTaskKind::Diagnose
        )
    };
    let requires_observation = signals.explicit_verification_requested
        || signals.observable_target_request
        || signals.visible_state_problem
        || task_kind == CompletionTaskKind::Diagnose
        || (matches!(
            task_kind,
            CompletionTaskKind::Check | CompletionTaskKind::Find
        ) && (signals.has_verification_target || signals.claimed_side_effect));
    let requires_reverification_after_mutation = matches!(
        task_kind,
        CompletionTaskKind::Diagnose | CompletionTaskKind::Monitor
    ) || (expects_mutation
        && (signals.explicit_verification_requested
            || text_contains_any_phrase(&lower, &["deploy", "publish", "release", "go live"])
            || signals.visible_state_problem));

    CompletionContract {
        task_kind,
        expects_mutation,
        requires_observation,
        requires_reverification_after_mutation,
        explicit_verification_requested: signals.explicit_verification_requested,
        connected_content_mode,
        verification_targets,
    }
}

fn sanitize_carryover_blocks(input: &str) -> (String, bool) {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return (String::new(), false);
    }
    let markers = ["Original request:", "Assistant asked:", "Follow-up:"];
    let mut sanitized = trimmed.to_string();
    let mut changed = false;
    for marker in markers {
        if sanitized.contains(marker) {
            changed = true;
            sanitized = sanitized.replace(marker, "");
        }
    }
    (sanitized.trim().to_string(), changed)
}

fn classify_followup_mode(
    current: &str,
    prev_assistant: Option<&str>,
) -> (FollowupMode, Vec<TurnContextReason>) {
    let mut reasons = Vec::new();
    let trimmed = current.trim();
    if trimmed.is_empty() {
        reasons.push(TurnContextReason::DefaultNewTask);
        return (FollowupMode::NewTask, reasons);
    }
    let lower = trimmed.to_ascii_lowercase();
    let is_short = trimmed.chars().count() <= 260;
    if !is_short || looks_like_explicit_task_switch(&lower) {
        reasons.push(TurnContextReason::DefaultNewTask);
        return (FollowupMode::NewTask, reasons);
    }

    if looks_like_standalone_goal_request(&lower) {
        reasons.push(TurnContextReason::FollowupOverrideStandalone);
        reasons.push(TurnContextReason::DefaultNewTask);
        return (FollowupMode::NewTask, reasons);
    }
    if looks_like_self_contained_mutation_request(trimmed, &lower) {
        reasons.push(TurnContextReason::FollowupOverrideStandalone);
        reasons.push(TurnContextReason::DefaultNewTask);
        return (FollowupMode::NewTask, reasons);
    }
    if looks_like_short_command_request(trimmed) {
        reasons.push(TurnContextReason::FollowupOverrideStandalone);
        reasons.push(TurnContextReason::DefaultNewTask);
        return (FollowupMode::NewTask, reasons);
    }

    let ack_like = contains_keyword_as_words(&lower, "yes")
        || contains_keyword_as_words(&lower, "confirm")
        || contains_keyword_as_words(&lower, "go ahead")
        || contains_keyword_as_words(&lower, "do it")
        || contains_keyword_as_words(&lower, "sure")
        || contains_keyword_as_words(&lower, "ok")
        || contains_keyword_as_words(&lower, "okay")
        || contains_keyword_as_words(&lower, "sounds good")
        || contains_keyword_as_words(&lower, "just use");
    let concise_ack_like = ack_like && trimmed.chars().count() <= 80;
    let explicit_followup = lower.starts_with("also ")
        || lower.starts_with("and ")
        || lower.starts_with("plus ")
        || concise_ack_like
        || looks_like_style_followup(&lower);
    if explicit_followup {
        reasons.push(TurnContextReason::ExplicitFollowup);
        return (FollowupMode::Followup, reasons);
    }

    if prev_assistant.is_some() && looks_like_context_dependent_followup_question(&lower) {
        reasons.push(TurnContextReason::ContextDependentQuestion);
        return (FollowupMode::Followup, reasons);
    }

    if prev_assistant.is_some_and(|prev| {
        assistant_message_looks_like_clarifying_question(prev)
            && !trimmed.trim_end().ends_with('?')
            && !looks_like_explicit_task_switch(&lower)
            && !looks_like_artifact_inspection_request(&lower)
    }) {
        reasons.push(TurnContextReason::ClarificationAnswer);
        return (FollowupMode::ClarificationAnswer, reasons);
    }

    reasons.push(TurnContextReason::DefaultNewTask);
    (FollowupMode::NewTask, reasons)
}

fn has_project_scope_divergence_with_aliases(
    prev_user_text: &str,
    current: &str,
    alias_roots: &[String],
) -> bool {
    let mut prev_scopes = Vec::new();
    let mut current_scopes = Vec::new();
    extract_project_scopes_from_text(prev_user_text, &mut prev_scopes, 6, alias_roots);
    extract_project_scopes_from_text(current, &mut current_scopes, 6, alias_roots);
    if !prev_scopes.is_empty() && !current_scopes.is_empty() {
        return !current_scopes
            .iter()
            .any(|scope| prev_scopes.iter().any(|prev| prev == scope));
    }

    let mut prev_hints = Vec::new();
    let mut current_hints = Vec::new();
    extract_project_hints_from_text(prev_user_text, &mut prev_hints, 6, false);
    extract_project_hints_from_text(current, &mut current_hints, 6, false);
    if prev_hints.is_empty() || current_hints.is_empty() {
        return false;
    }
    !current_hints
        .iter()
        .any(|hint| prev_hints.iter().any(|p| p == hint))
}

#[cfg(test)]
fn looks_like_followup_reply(current: &str, prev_assistant: Option<&str>) -> bool {
    let trimmed = current.trim();
    if trimmed.is_empty() {
        return false;
    }
    let (mode, _) = classify_followup_mode(current, prev_assistant);
    mode != FollowupMode::NewTask
}

fn trim_assistant_context_content(content: &str) -> String {
    let trimmed = content.trim();
    if let Some((before, _)) = trimmed.split_once(INTENT_GATE_MARKER) {
        before.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

fn is_low_signal_http_metadata_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("content-type:")
        || lower.starts_with("content-length:")
        || lower.starts_with("cache-control:")
        || lower.starts_with("date:")
        || lower.starts_with("etag:")
        || lower.starts_with("expires:")
        || lower.starts_with("last-modified:")
        || lower.starts_with("location:")
        || lower.starts_with("server:")
        || lower.starts_with("strict-transport-security:")
        || lower.starts_with("vary:")
        || lower.starts_with("via:")
        || lower.starts_with("x-")
}

fn is_low_info_recent_tool_context(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "write_file"
            | "edit_file"
            | "manage_memories"
            | "manage_people"
            | "remember_fact"
            | "check_environment"
    )
}

fn is_low_signal_tool_context_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.is_empty()
        || is_low_signal_http_metadata_line(line)
        || lower == "[truncated]"
        || lower.starts_with("exit code:")
        || lower.starts_with("[mode:")
}

fn summarize_http_request_context(primary: &str) -> Option<String> {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    if !first.to_ascii_lowercase().starts_with("http ") {
        return Some(first.to_string());
    }

    let detail = lines.find(|line| !is_low_signal_http_metadata_line(line));
    match detail {
        Some(detail) if !detail.eq_ignore_ascii_case(first) => Some(format!("{first} | {detail}")),
        _ => Some(first.to_string()),
    }
}

fn summarize_generic_tool_context(primary: &str) -> Option<String> {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !is_low_signal_tool_context_line(line));
    let first = lines.next()?;
    let second = lines.next();
    match second {
        Some(second) if !second.eq_ignore_ascii_case(first) => Some(format!("{first} | {second}")),
        _ => Some(first.to_string()),
    }
}

fn summarize_recent_tool_context(msg: &Message) -> Option<String> {
    let tool_name = msg.tool_name.as_deref()?;
    if is_low_info_recent_tool_context(tool_name) {
        return None;
    }

    let primary = msg.primary_content()?;
    let summary = match tool_name {
        "http_request" => summarize_http_request_context(&primary)?,
        _ => summarize_generic_tool_context(&primary)?,
    };
    Some(format!(
        "{}: {}",
        tool_name,
        truncate_for_resume(&summary, 240)
    ))
}

fn extract_recent_parent_messages(history: &[Message], max_messages: usize) -> Vec<Value> {
    let mut rows_rev: Vec<Value> = Vec::new();
    let mut recent_tool_rows = 0usize;

    for msg in history.iter().rev() {
        let row = match msg.role.as_str() {
            "user" => {
                let Some(raw) = msg.content.as_deref().map(str::trim) else {
                    continue;
                };
                if raw.is_empty() {
                    continue;
                }
                Some(json!({
                    "role": msg.role,
                    "content": truncate_for_resume(raw, 500),
                }))
            }
            "assistant" => {
                let Some(raw) = msg.content.as_deref().map(str::trim) else {
                    continue;
                };
                if raw.is_empty() {
                    continue;
                }
                let content = trim_assistant_context_content(raw);
                if content.trim().is_empty() {
                    continue;
                }
                Some(json!({
                    "role": msg.role,
                    "content": truncate_for_resume(content.trim(), 500),
                }))
            }
            "tool" if recent_tool_rows < 2 => summarize_recent_tool_context(msg).map(|content| {
                recent_tool_rows += 1;
                json!({
                    "role": "tool",
                    "content": content,
                })
            }),
            _ => None,
        };

        if let Some(row) = row {
            rows_rev.push(row);
            if rows_rev.len() >= max_messages {
                break;
            }
        }
    }

    rows_rev.reverse();
    rows_rev
}

fn is_generic_non_project_token(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "with"
            | "from"
            | "into"
            | "about"
            | "using"
            | "make"
            | "create"
            | "modern"
            | "frontend"
            | "backend"
            | "design"
            | "developer"
            | "senior"
            | "best"
            | "style"
            | "tailwind"
            | "react"
            | "html"
            | "css"
            | "javascript"
            | "typescript"
            | "project"
            | "workspace"
            | "directory"
            | "folder"
    )
}

fn is_likely_filename(token: &str) -> bool {
    let Some((name, ext)) = token.rsplit_once('.') else {
        return false;
    };
    // If the name portion itself contains dots (e.g. "blog.aidaemon" in "blog.aidaemon.ai"),
    // this looks like a domain name or multi-label hostname, not a plain filename.
    if name.contains('.') {
        return false;
    }
    !name.is_empty()
        && !ext.is_empty()
        && ext.len() <= 8
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
}

fn token_looks_like_filesystem_path(token: &str) -> bool {
    // Reject URL-like tokens without a protocol prefix (e.g., "api.waqi.info/feed/miami").
    // These contain dots before the first slash, resembling hostnames, not file paths.
    if let Some(slash_idx) = token.find('/') {
        let before_slash = &token[..slash_idx];
        if before_slash.contains('.') && !before_slash.starts_with('.') {
            return false;
        }
        // Also reject tokens containing '?' (query strings) — never valid in file paths.
        if token.contains('?') {
            return false;
        }
    }

    let bytes = token.as_bytes();
    let looks_windows_abs = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/');
    token.starts_with('/')
        || token.starts_with("~/")
        || token.starts_with("./")
        || token.starts_with("../")
        || token.contains('/')
        || token.contains('\\')
        || looks_windows_abs
}

fn token_looks_like_project_scope_path(token: &str, alias_roots: &[String]) -> bool {
    if !token_looks_like_filesystem_path(token) {
        return false;
    }

    // History often contains natural-language slash phrases ("reload/restart")
    // and markdown emphasis artifacts ("path**"). Those should not become the
    // sticky project scope for later turns.
    if token.contains('*') || token.contains('?') {
        return false;
    }

    let bytes = token.as_bytes();
    let looks_windows_abs = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/');
    if token.starts_with('/')
        || token.starts_with("~/")
        || token.starts_with("./")
        || token.starts_with("../")
        || looks_windows_abs
    {
        return true;
    }

    let Some(first_segment) = token
        .split(['/', '\\'])
        .find(|segment| !segment.trim().is_empty())
    else {
        return false;
    };

    let cwd_has_segment = std::env::current_dir()
        .ok()
        .is_some_and(|cwd| cwd.join(first_segment).exists());
    if cwd_has_segment {
        return true;
    }

    alias_roots
        .iter()
        .filter_map(|root| crate::tools::fs_utils::validate_path(root).ok())
        .any(|root| root.join(first_segment).exists())
}

fn is_common_path_segment(token: &str) -> bool {
    matches!(
        token,
        "users"
            | "user"
            | "home"
            | "workspace"
            | "workspaces"
            | "projects"
            | "repos"
            | "repo"
            | "src"
            | "apps"
            | "app"
            | "packages"
            | "package"
            | "code"
            | "tmp"
            | "var"
            | "usr"
            | "opt"
            | "local"
            | "dev"
            | "documents"
            | "downloads"
            | "desktop"
    )
}

fn normalize_project_component(raw: &str, allow_plain_names: bool) -> Option<String> {
    let token = raw
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
        .to_ascii_lowercase();
    if token.is_empty() || token.contains("://") {
        return None;
    }
    if token.len() < 3 {
        return None;
    }
    if !token.chars().any(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if is_generic_non_project_token(&token) {
        return None;
    }
    if is_likely_filename(&token) {
        return None;
    }

    if !allow_plain_names {
        let looks_project_like = token.contains("project")
            || token.contains('-')
            || token.contains('_')
            || token.starts_with("app")
            || token.ends_with("app");
        if !looks_project_like {
            return None;
        }
    }

    Some(token)
}

fn extract_project_hint_from_path_like_token(raw_token: &str) -> Option<String> {
    let trimmed = raw_token
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'))
        .to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    let path_source = if let Some((_, after_scheme)) = trimmed.split_once("://") {
        let (_, path) = after_scheme.split_once('/')?;
        path.to_string()
    } else {
        trimmed
    };

    let mut parts: Vec<String> = Vec::new();
    for raw_part in path_source.split(['/', '\\']) {
        let part = raw_part
            .split(['?', '#'])
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
            .to_ascii_lowercase();
        if !part.is_empty() {
            parts.push(part);
        }
    }

    for part in parts.iter().rev() {
        if is_common_path_segment(part) || is_likely_filename(part) {
            continue;
        }
        if let Some(candidate) = normalize_project_component(part, true) {
            return Some(candidate);
        }
    }

    None
}

fn normalize_project_hint(raw: &str, path_like: bool) -> Option<String> {
    let uri_like = raw.contains("://");
    if path_like || uri_like {
        if let Some(path_hint) = extract_project_hint_from_path_like_token(raw) {
            return Some(path_hint);
        }
        if uri_like {
            return None;
        }
    }
    normalize_project_component(raw, false)
}

fn push_project_hint(hints: &mut Vec<String>, hint: String, max_hints: usize) {
    if hints.len() >= max_hints || hints.iter().any(|existing| existing == &hint) {
        return;
    }
    hints.push(hint);
}

fn extract_project_hints_from_text(
    text: &str,
    hints: &mut Vec<String>,
    max_hints: usize,
    path_only: bool,
) {
    for raw in text.split_whitespace() {
        if hints.len() >= max_hints {
            break;
        }
        let uri_like = raw.contains("://");
        let path_like = uri_like
            || raw.contains('/')
            || raw.contains('\\')
            || raw.starts_with("./")
            || raw.starts_with("../")
            || raw.starts_with("~/");
        if path_only && !path_like {
            continue;
        }
        if let Some(normalized) = normalize_project_hint(raw, path_like) {
            push_project_hint(hints, normalized, max_hints);
        }
    }
}

fn extract_project_hints_from_history(
    history: &[Message],
    current_user_text: &str,
    max_hints: usize,
    include_history_hints: bool,
) -> Vec<String> {
    let mut hints: Vec<String> = Vec::new();
    extract_project_hints_from_text(current_user_text, &mut hints, max_hints, false);

    if !include_history_hints {
        return hints;
    }

    for msg in history.iter().rev() {
        if hints.len() >= max_hints {
            break;
        }
        if let Some(content) = msg.content.as_deref() {
            match msg.role.as_str() {
                "user" | "assistant" => {
                    extract_project_hints_from_text(content, &mut hints, max_hints, false);
                }
                "tool" => {
                    // Tool payloads are noisy; only trust explicit paths/URIs there.
                    extract_project_hints_from_text(content, &mut hints, max_hints, true);
                }
                _ => {}
            }
        }
    }
    hints
}

fn normalize_project_scope_path_with_aliases(
    raw_path: &str,
    alias_roots: &[String],
) -> Option<String> {
    crate::tools::fs_utils::resolve_project_scope_reference(raw_path, alias_roots)
        .map(|path| path.to_string_lossy().to_string())
}

fn push_project_scope(scopes: &mut Vec<String>, scope: String, max_scopes: usize) {
    if scopes.len() >= max_scopes || scopes.iter().any(|existing| existing == &scope) {
        return;
    }
    scopes.push(scope);
}

fn extract_project_scopes_from_text(
    text: &str,
    scopes: &mut Vec<String>,
    max_scopes: usize,
    alias_roots: &[String],
) {
    for raw in text.split_whitespace() {
        if scopes.len() >= max_scopes {
            break;
        }
        let token = raw
            .trim_matches(|c: char| {
                c.is_ascii_whitespace()
                    || matches!(
                        c,
                        '`' | '\''
                            | '"'
                            | ','
                            | ';'
                            | ':'
                            | '.'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                    )
            })
            .trim();
        if token.is_empty() || token.contains("://") {
            continue;
        }
        let scope = if token_looks_like_project_scope_path(token, alias_roots) {
            normalize_project_scope_path_with_aliases(token, alias_roots)
        } else {
            super::should_allow_contextual_project_nickname_scope(text, token)
                .then(|| {
                    crate::tools::fs_utils::resolve_named_project_root(token, alias_roots)
                        .or_else(|| {
                            crate::tools::fs_utils::resolve_contextual_project_nickname_in_explicit_roots(token, alias_roots)
                        })
                })
                .flatten()
                .map(|path| path.to_string_lossy().to_string())
        };
        if let Some(scope) = scope {
            push_project_scope(scopes, scope, max_scopes);
        }
    }
}

fn scope_looks_like_project_root(scope: &str) -> bool {
    let Ok(path) = crate::tools::fs_utils::validate_path(scope) else {
        return false;
    };
    if !path.is_dir() {
        return false;
    }
    crate::tools::fs_utils::find_nearest_project_root(&path).is_some_and(|root| root == path)
}

fn choose_primary_project_scope(scopes: &[String]) -> Option<String> {
    scopes
        .iter()
        .find(|scope| scope_looks_like_project_root(scope))
        .cloned()
        .or_else(|| scopes.first().cloned())
}

fn turn_allows_inherited_project_scope(
    current_user_text: &str,
    current_user_scopes: &[String],
) -> bool {
    if !current_user_scopes.is_empty() {
        return true;
    }

    if super::user_explicitly_requests_local_file_inspection(current_user_text) {
        return true;
    }

    let lower = current_user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if super::text_has_explicit_project_scope_cues(&lower)
        || looks_like_short_command_request(current_user_text)
    {
        return true;
    }

    [
        "cargo",
        "npm",
        "pnpm",
        "yarn",
        "pytest",
        "wrangler",
        "docker",
        "kubectl",
        "build",
        "compile",
        "deploy",
        "lint",
        "format",
        "fmt",
        "commit",
        "branch",
        "diff",
        "logs",
        "log",
        "migration",
        "migrations",
        "schema",
        "daemon",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw))
}

fn resolve_primary_project_scope(
    extracted_primary_scope: Option<String>,
    inherited_project_scope: Option<&str>,
    allow_multi_project_scope: bool,
    allow_inherited_scope: bool,
) -> Option<String> {
    if extracted_primary_scope.is_some() {
        return extracted_primary_scope;
    }

    if allow_multi_project_scope {
        return allow_inherited_scope
            .then(|| inherited_project_scope.map(ToOwned::to_owned))
            .flatten();
    }

    if allow_inherited_scope {
        inherited_project_scope.map(ToOwned::to_owned)
    } else {
        None
    }
}

fn extract_project_scopes_from_history(
    history: &[Message],
    current_user_text: &str,
    max_scopes: usize,
    include_history_scopes: bool,
    alias_roots: &[String],
) -> Vec<String> {
    let mut scopes = Vec::new();
    extract_project_scopes_from_text(current_user_text, &mut scopes, max_scopes, alias_roots);

    if !include_history_scopes {
        return scopes;
    }

    for msg in history.iter().rev() {
        if scopes.len() >= max_scopes {
            break;
        }
        if msg.role != "user" {
            continue;
        }
        let Some(content) = msg.content.as_deref() else {
            continue;
        };
        extract_project_scopes_from_text(content, &mut scopes, max_scopes, alias_roots);
    }

    scopes
}

impl Agent {
    pub(crate) async fn record_auxiliary_assistant_note(
        &self,
        session_id: &str,
        content: &str,
    ) -> anyhow::Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let emitter = crate::events::EventEmitter::new(self.event_store.clone(), session_id);
        let msg = Message {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "assistant".to_string(),
            content: Some(trimmed.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.2,
            ..Message::runtime_defaults()
        };
        self.append_assistant_message_with_event(&emitter, &msg, "system", None, None)
            .await
    }

    pub(super) async fn build_turn_context_from_recent_history(
        &self,
        session_id: &str,
        user_text: &str,
    ) -> TurnContext {
        let current = user_text.trim();
        if current.is_empty() {
            return TurnContext::default();
        }

        let history = self
            .state
            .get_history(session_id, GOAL_CONTEXT_HINT_HISTORY_LIMIT)
            .await
            .unwrap_or_default();
        let (prev_assistant, prev_user) = find_previous_turns(&history, current);
        let (mut followup_mode, mut reasons) =
            classify_followup_mode(current, prev_assistant.as_deref());

        let mut goal_user_text = current.to_string();
        if followup_mode != FollowupMode::NewTask {
            let mismatch_preflight_drop = prev_user.as_deref().is_some_and(|prev| {
                has_project_scope_divergence_with_aliases(
                    prev,
                    current,
                    &self.path_aliases.projects,
                )
            });
            if mismatch_preflight_drop {
                followup_mode = FollowupMode::NewTask;
                reasons.push(TurnContextReason::FollowupOverrideMismatchPreflight);
                reasons.push(TurnContextReason::DefaultNewTask);
                POLICY_METRICS
                    .context_mismatch_preflight_drop_total
                    .fetch_add(1, Ordering::Relaxed);
                POLICY_METRICS
                    .followup_mode_overrides_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        if followup_mode != FollowupMode::NewTask {
            if let Some(prev_user_text) = prev_user
                .as_deref()
                .filter(|prev| !prev.trim().eq_ignore_ascii_case(current))
            {
                let mut combined = String::new();
                combined.push_str("Original request:\n");
                combined.push_str(&truncate_for_resume(prev_user_text.trim(), 2000));
                if let Some(prev_assistant_text) = prev_assistant.as_deref() {
                    let trimmed = prev_assistant_text.trim();
                    if !trimmed.is_empty() && trimmed.contains('?') {
                        combined.push_str("\n\nAssistant asked:\n");
                        combined.push_str(&truncate_for_resume(trimmed, 800));
                    }
                }
                combined.push_str("\n\nFollow-up:\n");
                combined.push_str(&truncate_for_resume(current, 800));
                goal_user_text = combined;
            }
        } else {
            let (sanitized, changed) = sanitize_carryover_blocks(current);
            if changed {
                reasons.push(TurnContextReason::CarryoverSanitized);
                POLICY_METRICS
                    .context_bleed_prevented_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            if !sanitized.is_empty() {
                goal_user_text = sanitized;
            }
        }

        let include_history_context = followup_mode != FollowupMode::NewTask;
        let project_hints = extract_project_hints_from_history(
            &history,
            current,
            GOAL_CONTEXT_MAX_PROJECT_HINTS,
            include_history_context,
        );
        let mut current_project_scopes = Vec::new();
        extract_project_scopes_from_text(
            current,
            &mut current_project_scopes,
            GOAL_CONTEXT_MAX_PROJECT_SCOPES,
            &self.path_aliases.projects,
        );
        let allow_scope_carryover = if looks_like_scope_carryover_ack(current) {
            let mut prior_user_scopes = Vec::new();
            if let Some(prev_user_text) = prev_user.as_deref() {
                extract_project_scopes_from_text(
                    prev_user_text,
                    &mut prior_user_scopes,
                    GOAL_CONTEXT_MAX_PROJECT_SCOPES,
                    &self.path_aliases.projects,
                );
            }
            !prior_user_scopes.is_empty()
                || prev_user.as_deref().is_some_and(|prev_user_text| {
                    turn_allows_inherited_project_scope(prev_user_text, &prior_user_scopes)
                })
        } else {
            turn_allows_inherited_project_scope(current, &current_project_scopes)
        };
        let project_scopes = extract_project_scopes_from_history(
            &history,
            current,
            GOAL_CONTEXT_MAX_PROJECT_SCOPES,
            include_history_context && allow_scope_carryover,
            &self.path_aliases.projects,
        );
        let allow_multi_project_scope =
            looks_like_multi_project_request(&current.to_ascii_lowercase());
        let primary_project_scope = resolve_primary_project_scope(
            choose_primary_project_scope(&project_scopes),
            self.inherited_project_scope.as_deref(),
            allow_multi_project_scope,
            allow_scope_carryover,
        );
        let completion_contract =
            infer_completion_contract(&goal_user_text, &self.path_aliases.projects);
        TurnContext {
            goal_user_text,
            recent_messages: if include_history_context {
                extract_recent_parent_messages(&history, GOAL_CONTEXT_RECENT_MESSAGES_LIMIT)
            } else {
                Vec::new()
            },
            project_hints,
            primary_project_scope,
            allow_multi_project_scope,
            followup_mode: Some(followup_mode),
            reasons,
            completion_contract,
        }
    }

    pub(super) async fn append_message_canonical(&self, msg: &Message) -> anyhow::Result<()> {
        self.state.append_message(msg).await
    }

    pub(super) async fn append_user_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        user_role: crate::types::UserRole,
        channel_ctx: &ChannelContext,
        has_attachments: bool,
    ) -> anyhow::Result<()> {
        let normalized_msg = msg.with_inferred_annotations();
        emitter
            .emit(
                EventType::UserMessage,
                json!({
                    "content": normalized_msg.content.clone().unwrap_or_default(),
                    "message_id": normalized_msg.id.clone(),
                    "has_attachments": has_attachments,
                    "annotations": normalized_msg.annotations.clone(),
                    // Provenance for downstream projections/consolidation.
                    "channel_visibility": channel_ctx.visibility.to_string(),
                    "channel_id": channel_ctx.channel_id.clone(),
                    "platform": channel_ctx.platform.clone(),
                    "sender_id": channel_ctx.sender_id.clone(),
                    "user_role": user_role.to_string(),
                }),
            )
            .await?;
        self.append_message_canonical(normalized_msg.as_ref())
            .await?;
        Ok(())
    }

    pub(super) async fn append_assistant_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        model: &str,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) -> anyhow::Result<()> {
        let normalized_msg = msg.with_inferred_annotations();
        let tool_calls = normalized_msg.tool_calls_json.as_ref().and_then(|raw| {
            serde_json::from_str::<Vec<ToolCall>>(raw)
                .ok()
                .map(|calls| {
                    calls
                        .into_iter()
                        .map(|tc| ToolCallInfo {
                            id: tc.id,
                            name: tc.name,
                            arguments: serde_json::from_str(&tc.arguments)
                                .unwrap_or(serde_json::json!({})),
                            extra_content: tc.extra_content,
                        })
                        .collect::<Vec<_>>()
                })
        });
        emitter
            .emit(
                EventType::AssistantResponse,
                AssistantResponseData {
                    message_id: Some(normalized_msg.id.clone()),
                    content: normalized_msg.content.clone(),
                    model: model.to_string(),
                    tool_calls,
                    input_tokens,
                    output_tokens,
                    annotations: normalized_msg.annotations.clone(),
                },
            )
            .await?;
        self.append_message_canonical(normalized_msg.as_ref())
            .await?;
        Ok(())
    }

    pub(super) async fn append_tool_message_with_result_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        success: bool,
        duration_ms: u64,
        error: Option<String>,
        task_id: Option<&str>,
    ) -> anyhow::Result<()> {
        let normalized_msg = msg.with_inferred_annotations();
        emitter
            .emit(
                EventType::ToolResult,
                ToolResultData {
                    message_id: Some(normalized_msg.id.clone()),
                    tool_call_id: normalized_msg
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| normalized_msg.id.clone()),
                    name: normalized_msg
                        .tool_name
                        .clone()
                        .unwrap_or_else(|| "system".to_string()),
                    result: normalized_msg.content.clone().unwrap_or_default(),
                    success,
                    duration_ms,
                    error,
                    task_id: task_id.map(str::to_string),
                    annotations: normalized_msg.annotations.clone(),
                },
            )
            .await?;
        self.append_message_canonical(normalized_msg.as_ref())
            .await?;
        Ok(())
    }

    pub(super) async fn load_initial_history(
        &self,
        session_id: &str,
        user_text: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        match self
            .event_store
            .get_conversation_history(session_id, limit)
            .await
        {
            Ok(history) if !history.is_empty() => {
                return Ok(history);
            }
            Ok(_) => {}
            Err(e) => {
                warn!(
                    session_id,
                    error = %e,
                    "Event history load failed; falling back to state context retrieval"
                );
            }
        }

        self.state.get_context(session_id, user_text, limit).await
    }

    pub(super) async fn load_recent_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        match self
            .event_store
            .get_conversation_history(session_id, limit)
            .await
        {
            Ok(history) if !history.is_empty() => Ok(history),
            Ok(_) => self.state.get_history(session_id, limit).await,
            Err(e) => {
                warn!(
                    session_id,
                    error = %e,
                    "Event recent-history load failed; falling back to state history retrieval"
                );
                self.state.get_history(session_id, limit).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use proptest::prelude::*;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn followup_detects_answer_to_clarifying_question() {
        let followup =
            "You are a senior designer and frontend developer. Do what you consider best.";
        let prev = "Which elements do you want me to modernize?";
        assert!(looks_like_followup_reply(followup, Some(prev)));
    }

    #[test]
    fn followup_rejects_explicit_task_switch() {
        let followup = "New task: build a dashboard from scratch.";
        let prev = "Should I continue with this page or focus on another one?";
        assert!(!looks_like_followup_reply(followup, Some(prev)));
    }

    #[test]
    fn followup_rejects_standalone_request_with_do_it_suffix() {
        let followup = "Compare the package.json files across all my projects in ~/projects. Which ones share dependencies? Don't ask questions, just do it.";
        let prev = "Which directories are taking up the most space?";
        assert!(!looks_like_followup_reply(followup, Some(prev)));
    }

    #[test]
    fn followup_accepts_concise_do_it_ack() {
        let followup = "Yes, do it.";
        let prev = "Should I proceed with this change?";
        assert!(looks_like_followup_reply(followup, Some(prev)));
    }

    #[test]
    fn followup_accepts_context_dependent_status_question() {
        let followup = "Did you find those 10 in the API?";
        let prev = "I fetched the search results and sent the JSON export.";
        let (mode, reasons) = classify_followup_mode(followup, Some(prev));
        assert_eq!(mode, FollowupMode::Followup);
        assert!(reasons.contains(&TurnContextReason::ContextDependentQuestion));
    }

    #[test]
    fn followup_accepts_context_dependent_explanation_question() {
        let followup = "What does it mean?";
        let prev = "I found several matching studies and sent the JSON export.";
        let (mode, reasons) = classify_followup_mode(followup, Some(prev));
        assert_eq!(mode, FollowupMode::Followup);
        assert!(reasons.contains(&TurnContextReason::ContextDependentQuestion));
    }

    #[test]
    fn generic_explanation_question_without_context_markers_stays_new_task() {
        let current = "Can you explain Rust ownership?";
        let prev = "I fetched the search results and sent the JSON export.";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::DefaultNewTask));
    }

    #[test]
    fn generic_short_question_without_context_markers_stays_new_task() {
        let current = "What is the weather in Paris?";
        let prev = "I fetched the search results and sent the JSON export.";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::DefaultNewTask));
    }

    #[test]
    fn artifact_request_does_not_inherit_previous_topic_as_clarification_answer() {
        let current = "Check the doc and fix the issue.";
        let prev = "Would you like me to get more detailed information for any specific trial(s)?";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::DefaultNewTask));
    }

    #[test]
    fn project_hint_extraction_finds_project_name() {
        let history = vec![
            msg(
                "user",
                "Please work in test-project and modernize index.html",
            ),
            msg("assistant", "Which sections should I prioritize?"),
            msg("user", "Do what you consider best."),
        ];
        let hints =
            extract_project_hints_from_history(&history, "Do what you consider best.", 6, true);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project in project hints, got {:?}",
            hints
        );
    }

    #[test]
    fn project_hint_extraction_handles_file_uri_paths() {
        let history = vec![msg(
            "assistant",
            "Opened file:///Users/testuser/projects/test-project/index.html",
        )];
        let hints =
            extract_project_hints_from_history(&history, "Do what you consider best.", 8, true);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project from file URI, got {:?}",
            hints
        );
        assert!(
            hints.iter().all(|h| h != "index.html"),
            "filename should not be treated as project hint: {:?}",
            hints
        );
    }

    #[test]
    fn project_hint_extraction_scans_tool_messages_path_only() {
        let history = vec![msg(
            "tool",
            "Using terminal: cd ~/projects/test-project && npm run build",
        )];
        let hints = extract_project_hints_from_history(&history, "make it modern", 8, true);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project from tool output path, got {:?}",
            hints
        );
    }

    #[test]
    fn project_hint_extraction_ignores_history_for_new_tasks() {
        let history = vec![
            msg(
                "user",
                "Please work in ~/projects/blog.aidaemon.ai/src/content/posts",
            ),
            msg("assistant", "Which posts should I update?"),
        ];
        let hints = extract_project_hints_from_history(&history, "Why?", 8, false);
        assert!(
            hints.is_empty(),
            "new-task hints should not inherit prior project context: {:?}",
            hints
        );
    }

    #[test]
    fn project_scope_extraction_uses_explicit_current_path() {
        let history = vec![msg("assistant", "Earlier we touched ~/projects/old-one")];
        let current = "Please work in ~/projects/new-one/src and review the files.";
        let scopes = extract_project_scopes_from_history(&history, current, 4, true, &[]);
        assert!(!scopes.is_empty());
        assert!(
            scopes[0].contains("new-one"),
            "expected first scope to come from current request, got {:?}",
            scopes
        );
    }

    #[test]
    fn project_scope_extraction_ignores_history_for_new_tasks() {
        let history = vec![msg(
            "assistant",
            "Found symlink: /Users/davidloor/.openclaw and other directories.",
        )];
        let current = "Find all Rust files in the aidaemon project that contain async fn.";
        let scopes = extract_project_scopes_from_history(&history, current, 4, false, &[]);
        assert!(
            scopes.iter().all(|scope| !scope.contains(".openclaw")),
            "new task scope should not inherit prior assistant paths: {:?}",
            scopes
        );
    }

    #[test]
    fn project_scope_extraction_keeps_history_for_followups() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dir_path = dir.path().to_string_lossy().to_string();
        let history = vec![
            msg(
                "user",
                &format!("Please work in {} and inspect async functions.", dir_path),
            ),
            msg("assistant", "Should I proceed with a full scan?"),
        ];
        let current = "Yes, do it.";
        let scopes = extract_project_scopes_from_history(&history, current, 4, true, &[]);
        assert!(
            scopes.iter().any(|scope| scope.contains(&*dir_path)),
            "followup should carry prior project scope when explicit: {:?}",
            scopes
        );
    }

    #[test]
    fn command_style_reply_to_question_starts_new_task() {
        let (mode, reasons) = classify_followup_mode(
            "Run build, and deploy it",
            Some("Do you have all the information from your facts?"),
        );
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::FollowupOverrideStandalone));
    }

    #[test]
    fn project_scope_extraction_resolves_contextual_nickname_from_prior_user_request() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let nickname = format!("scope-nick-{}", uuid::Uuid::new_v4().simple());
        let project_name = format!("{nickname}.aidaemon.ai");
        let project = alias_root.join(&project_name);
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let history = vec![msg(
            "user",
            &format!("Please deploy the {} project when you can.", nickname),
        )];
        let scopes =
            extract_project_scopes_from_history(&history, "Yes, do it.", 4, true, &alias_roots);
        assert_eq!(scopes, vec![project.to_string_lossy().to_string()]);
    }

    #[test]
    fn choose_primary_project_scope_prefers_real_project_root() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let blog = alias_root.join("blog.aidaemon.ai");
        let logs = root.path().join("Library/Logs/aidaemon");
        std::fs::create_dir_all(&blog).expect("create blog");
        std::fs::write(blog.join("wrangler.toml"), "name = \"blog\"\n").expect("blog wrangler");
        std::fs::create_dir_all(&logs).expect("create logs");

        let chosen = choose_primary_project_scope(&[
            logs.to_string_lossy().to_string(),
            blog.to_string_lossy().to_string(),
        ]);
        assert_eq!(chosen, Some(blog.to_string_lossy().to_string()));
    }

    #[test]
    fn explicit_scope_wins_over_inherited_scope_when_present() {
        let extracted = Some("/Users/davidloor/projects/terminal.aidaemon.ai".to_string());
        let resolved = resolve_primary_project_scope(
            extracted,
            Some("/Users/davidloor/Library/Logs/aidaemon"),
            false,
            true,
        );
        assert_eq!(
            resolved,
            Some("/Users/davidloor/projects/terminal.aidaemon.ai".to_string())
        );
    }

    #[test]
    fn inherited_project_scope_yields_to_explicit_scope_for_multi_project_requests() {
        let extracted = Some("/Users/davidloor/projects/terminal.aidaemon.ai".to_string());
        let resolved = resolve_primary_project_scope(
            extracted.clone(),
            Some("/Users/davidloor/Library/Logs/aidaemon"),
            true,
            true,
        );
        assert_eq!(resolved, extracted);
    }

    #[test]
    fn inherited_project_scope_is_ignored_when_turn_is_not_local_project_work() {
        let resolved = resolve_primary_project_scope(
            None,
            Some("/Users/davidloor/projects/fairfax-va-site"),
            false,
            false,
        );
        assert_eq!(resolved, None);
    }

    #[test]
    fn inherited_project_scope_allowed_for_workspace_command_turns() {
        assert!(turn_allows_inherited_project_scope("run cargo test", &[]));
        assert!(looks_like_scope_carryover_ack("Yes, do it."));
    }

    #[test]
    fn inherited_project_scope_blocked_for_external_followup_without_local_cues() {
        assert!(!turn_allows_inherited_project_scope(
            "Give me more details of the ones offered by next oncology.",
            &[]
        ));
    }

    #[test]
    fn project_scope_extraction_ignores_assistant_paths_for_non_local_followup() {
        let history = vec![msg(
            "assistant",
            "I am currently locked to /Users/davidloor/projects/fairfax-va-site.",
        )];

        let scopes = extract_project_scopes_from_history(
            &history,
            "Give me all the info about the top 2.",
            4,
            true,
            &[],
        );
        assert!(
            scopes.is_empty(),
            "non-local followup should not inherit assistant path scope: {:?}",
            scopes
        );
    }

    #[test]
    fn visible_issue_contract_requires_observation_and_reverification() {
        let contract = infer_completion_contract(
            "I still don't see the posts here: https://blog.aidaemon.ai",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Diagnose);
        assert!(contract.requires_observation);
        assert!(contract.requires_reverification_after_mutation);
        assert_eq!(contract.verification_targets.len(), 1);
        assert_eq!(
            contract.verification_targets[0],
            VerificationTarget {
                kind: VerificationTargetKind::Url,
                value: "https://blog.aidaemon.ai".to_string(),
            }
        );
    }

    #[test]
    fn create_record_contract_does_not_force_verification() {
        let contract = infer_completion_contract("Create the remote record.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn still_phrase_alone_does_not_force_diagnose() {
        let contract = infer_completion_contract("I still want you to deploy the app.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(contract.requires_reverification_after_mutation);
    }

    #[test]
    fn target_reference_in_change_request_does_not_force_verification() {
        let contract = infer_completion_contract(
            "Update /tmp/aidaemon/config.toml to point at the new endpoint.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
        assert_eq!(contract.verification_targets.len(), 1);
    }

    #[test]
    fn reading_target_requires_observation_without_change() {
        // "post" triggers connected content mode DraftThenDeliver (as both
        // a content delivery resource and a delivery verb), which forces
        // expects_mutation=true even though task_kind remains Answer.
        let contract = infer_completion_contract(
            "Read https://blog.aidaemon.ai and summarize the latest post.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Answer);
        assert!(contract.expects_mutation);
        assert!(contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn deploy_and_verify_stays_change_with_reverification() {
        let contract = infer_completion_contract(
            "Deploy the app and verify it is live at https://blog.aidaemon.ai",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(contract.requires_observation);
        assert!(contract.requires_reverification_after_mutation);
    }

    #[test]
    fn schedule_request_tracks_mutation_without_forcing_verification() {
        let contract = infer_completion_contract("Remind me tomorrow at 9am to call Alice.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Schedule);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn detailed_schedule_request_breaks_followup_carryover() {
        let current = "Can you set up a daily scheduled task at 6:00 am to publish the blog with honest reflections about recent errors and fixes.";
        let (mode, reasons) =
            classify_followup_mode(current, Some("Would you like me to schedule that?"));
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::FollowupOverrideStandalone));
    }

    #[test]
    fn deictic_schedule_followup_stays_contextual() {
        let current = "Schedule it for every day at 6am.";
        let (mode, reasons) =
            classify_followup_mode(current, Some("Would you like me to schedule that?"));
        assert_ne!(mode, FollowupMode::NewTask);
        assert!(!reasons.contains(&TurnContextReason::FollowupOverrideStandalone));
    }

    #[test]
    fn generic_check_request_does_not_force_verification_without_target() {
        let contract = infer_completion_contract("Check system health.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Check);
        assert!(!contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn targeted_check_request_requires_observation_without_mutation() {
        let contract = infer_completion_contract("Check https://status.example.com.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Check);
        assert!(!contract.expects_mutation);
        assert!(contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn deliver_request_does_not_force_observation_without_explicit_verification() {
        let contract = infer_completion_contract("Email this note to Alice.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Deliver);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn connected_content_draft_only_request_uses_compose_contract() {
        let contract = infer_completion_contract("Help me write a tweet about our launch.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Compose);
        assert!(!contract.expects_mutation);
        assert_eq!(
            contract.connected_content_mode,
            super::super::intent_routing::ConnectedContentMode::DraftOnly
        );
    }

    #[test]
    fn connected_content_draft_then_deliver_request_keeps_delivery_contract() {
        let contract = infer_completion_contract(
            "Can you post a tweet about your new stuff and make it engaging so people want to comment?",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Deliver);
        assert!(contract.expects_mutation);
        assert_eq!(
            contract.connected_content_mode,
            super::super::intent_routing::ConnectedContentMode::DraftThenDeliver
        );
    }

    #[test]
    fn generic_find_request_does_not_force_verification_without_target() {
        // "note" triggers asks_change which takes priority over asks_find,
        // so this is classified as Change with expects_mutation=true.
        let contract =
            infer_completion_contract("Find the most relevant note from last week.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
    }

    #[test]
    fn short_command_request_requires_structured_target() {
        assert!(looks_like_short_command_request("run cargo test"));
        assert!(!looks_like_short_command_request("check that"));
        assert!(!looks_like_short_command_request("show it again"));
    }

    #[test]
    fn completion_progress_resets_after_mutation_and_clears_after_observation() {
        let contract = infer_completion_contract(
            "I still don't see the posts here: https://blog.aidaemon.ai",
            &[],
        );
        let mut progress = CompletionProgress::new(&contract);
        assert!(progress.verification_pending);

        progress.mark_observation(&contract, true);
        assert!(!progress.verification_pending);

        progress.mark_mutation(&contract);
        assert!(progress.verification_pending);

        progress.mark_observation(&contract, true);
        assert!(!progress.verification_pending);
        assert_eq!(progress.verification_count, 2);
    }

    #[test]
    fn project_scope_extraction_resolves_projects_folder_alias_to_configured_root() {
        let history = vec![];
        let mut project_name = format!("alias-test-{}", uuid::Uuid::new_v4().simple());
        let cwd = std::env::current_dir().expect("cwd");
        let mut cwd_candidate = cwd.join("projects").join(&project_name);
        if cwd_candidate.exists() {
            project_name = format!("alias-test-{}", uuid::Uuid::new_v4().simple());
            cwd_candidate = cwd.join("projects").join(&project_name);
        }
        assert!(
            !cwd_candidate.exists(),
            "cwd candidate unexpectedly exists: {}",
            cwd_candidate.display()
        );
        let current = format!("Initialize a Vite app at projects/{}", project_name);
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        std::fs::create_dir_all(&alias_root).expect("create alias root");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];
        let scopes =
            extract_project_scopes_from_history(&history, &current, 4, false, &alias_roots);
        let expected = alias_root.join(&project_name).to_string_lossy().to_string();
        assert!(
            scopes.iter().any(|scope| scope == &expected),
            "expected aliased projects scope '{}', got {:?}",
            expected,
            scopes
        );
    }

    #[test]
    fn project_scope_normalization_prefers_absolute_paths_over_aliases() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        std::fs::create_dir_all(&alias_root).expect("create alias root");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];
        let absolute = root.path().join("absolute-target");
        let normalized = normalize_project_scope_path_with_aliases(
            absolute.to_string_lossy().as_ref(),
            &alias_roots,
        )
        .expect("normalized absolute path");
        assert_eq!(normalized, absolute.to_string_lossy());
    }

    #[test]
    fn project_scope_normalization_promotes_existing_src_paths_to_repo_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("blog");
        let src = root.join("src");
        std::fs::create_dir_all(&src).expect("create src");
        std::fs::write(root.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        std::fs::write(src.join("posts.js"), "export default [];\n").expect("posts");

        let normalized = normalize_project_scope_path_with_aliases(
            src.join("posts.js").to_string_lossy().as_ref(),
            &[],
        )
        .expect("normalized");
        assert_eq!(normalized, root.to_string_lossy());
    }

    #[test]
    fn project_scope_extraction_resolves_named_project_roots() {
        let history = vec![];
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &history,
            "Deploy blog.aidaemon.ai",
            4,
            false,
            &alias_roots,
        );
        assert_eq!(scopes, vec![project.to_string_lossy().to_string()]);
    }

    #[test]
    fn project_scope_extraction_rejects_named_project_roots_without_local_cues() {
        let history = vec![];
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"blog\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &history,
            "Tell me about blog.aidaemon.ai and its latest posts.",
            4,
            false,
            &alias_roots,
        );
        assert!(
            scopes.is_empty(),
            "descriptive external request should not infer project scope: {:?}",
            scopes
        );
    }

    #[test]
    fn project_scope_extraction_rejects_plain_word_nickname_without_local_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &[],
            "Find recruiting studies in Fairfax, Virginia and summarize them.",
            4,
            false,
            &alias_roots,
        );

        assert!(
            scopes.is_empty(),
            "plain language should not infer a project scope: {:?}",
            scopes
        );
    }

    #[test]
    fn project_scope_extraction_allows_plain_word_nickname_with_explicit_project_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let scopes = extract_project_scopes_from_history(
            &[],
            "Check the Fairfax project for broken links.",
            4,
            false,
            &alias_roots,
        );

        assert_eq!(scopes, vec![project.to_string_lossy().to_string()]);
    }

    #[test]
    fn project_scope_extraction_rejects_natural_language_slash_phrases() {
        let scopes = extract_project_scopes_from_history(
            &[],
            "The old workflow used to commit, deploy, and reload/restart the daemon.",
            4,
            false,
            &[],
        );

        assert!(
            scopes.is_empty(),
            "natural-language slash phrase should not infer a project scope: {:?}",
            scopes
        );
    }

    #[test]
    fn project_scope_extraction_rejects_globbed_absolute_paths() {
        let root = tempfile::tempdir().expect("tempdir");
        let project = root.path().join("demo-project");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(
            project.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
        )
        .expect("cargo");
        let text = format!(
            "Previous failure mentioned {}",
            project.join("reload/restart**").display()
        );

        let mut scopes = Vec::new();
        extract_project_scopes_from_text(&text, &mut scopes, 4, &[]);

        assert!(
            scopes.is_empty(),
            "globbed absolute path should not infer a project scope: {:?}",
            scopes
        );
    }

    #[test]
    fn named_project_scope_divergence_breaks_followup_carryover() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let dogs = alias_root.join("dogs-project");
        let blog = alias_root.join("blog.aidaemon.ai");
        std::fs::create_dir_all(&dogs).expect("create dogs");
        std::fs::create_dir_all(&blog).expect("create blog");
        std::fs::write(dogs.join("package.json"), r#"{"name":"dogs"}"#).expect("dogs package");
        std::fs::write(blog.join("wrangler.toml"), "name = \"blog\"\n").expect("blog wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        assert!(has_project_scope_divergence_with_aliases(
            "Deploy dogs-project",
            "Now deploy blog.aidaemon.ai",
            &alias_roots,
        ));
    }

    #[test]
    fn recent_parent_messages_strip_intent_gate_payload() {
        let history = vec![
            msg("user", "Build a site"),
            msg(
                "assistant",
                "Sure, I can help.\n[INTENT_GATE] {\"can_answer_now\":false}",
            ),
        ];
        let messages = extract_recent_parent_messages(&history, 6);
        assert_eq!(messages.len(), 2);
        let assistant_content = messages[1]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(!assistant_content.contains("[INTENT_GATE]"));
        assert_eq!(assistant_content, "Sure, I can help.");
    }

    #[test]
    fn recent_parent_messages_include_recent_api_and_file_tool_summaries() {
        let history = vec![
            msg("user", "Look up those studies."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "http_request",
                    "HTTP 200 OK\ncontent-type: application/json\n\n{\"studies\":[]}",
                )),
                tool_call_id: Some("call-http".to_string()),
                tool_name: Some("http_request".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some("File sent: studies.json (127 KB)".to_string()),
                tool_call_id: Some("call-file".to_string()),
                tool_name: Some("send_file".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "I fetched the results and sent the file."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| content.contains("http_request: HTTP 200 OK"))
        }));
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| content.contains("send_file: File sent: studies.json"))
        }));
    }

    #[test]
    fn recent_parent_messages_preserve_http_error_detail_beyond_headers() {
        let history = vec![
            msg("user", "Check that trial ID."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "http_request",
                    "HTTP 404 Not Found\ncontent-type: application/json\n\nNot Found: NCT05178195",
                )),
                tool_call_id: Some("call-http".to_string()),
                tool_name: Some("http_request".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg(
                "assistant",
                "The API could not find that study, but I should confirm the exact error.",
            ),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| {
                        content
                            .contains("http_request: HTTP 404 Not Found | Not Found: NCT05178195")
                    })
        }));
    }

    #[test]
    fn recent_parent_messages_include_web_fetch_error_summaries() {
        let history = vec![
            msg("user", "Check that page."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some(crate::tools::sanitize::wrap_untrusted_output(
                    "web_fetch",
                    "Error fetching https://example.com/missing: HTTP 404 Not Found",
                )),
                tool_call_id: Some("call-web-fetch".to_string()),
                tool_name: Some("web_fetch".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "That page fetch failed."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row.get("content").and_then(|v| v.as_str()).is_some_and(|content| {
                    content.contains("web_fetch: Error fetching https://example.com/missing: HTTP 404 Not Found")
                    })
        }));
    }

    #[test]
    fn recent_parent_messages_include_generic_terminal_evidence_summary() {
        let history = vec![
            msg("user", "Run the tests."),
            Message {
                id: uuid::Uuid::new_v4().to_string(),
                session_id: "test-session".to_string(),
                role: "tool".to_string(),
                content: Some("pytest\nAssertionError: expected 1 but got 2".to_string()),
                tool_call_id: Some("call-terminal".to_string()),
                tool_name: Some("terminal".to_string()),
                tool_calls_json: None,
                created_at: Utc::now(),
                importance: 0.5,
                ..Message::runtime_defaults()
            },
            msg("assistant", "The tests failed."),
        ];

        let messages = extract_recent_parent_messages(&history, 6);
        assert!(messages.iter().any(|row| {
            row.get("role").and_then(|v| v.as_str()) == Some("tool")
                && row
                    .get("content")
                    .and_then(|v| v.as_str())
                    .is_some_and(|content| {
                        content.contains("terminal: pytest | AssertionError: expected 1 but got 2")
                    })
        }));
    }

    #[tokio::test]
    async fn build_turn_context_omits_history_context_for_new_tasks() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Please work in ~/projects/blog.aidaemon.ai/src/content/posts",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Which posts should I update?"))
            .await
            .expect("append prior assistant");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", "Why?")
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::NewTask));
        assert!(
            turn_context.recent_messages.is_empty(),
            "new tasks should not include prior recent_messages: {:?}",
            turn_context.recent_messages
        );
        assert!(
            turn_context.project_hints.is_empty(),
            "new tasks should not inherit prior project hints: {:?}",
            turn_context.project_hints
        );
    }

    #[tokio::test]
    async fn build_turn_context_does_not_reuse_old_local_scope_for_external_followup() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Please work in /Users/davidloor/projects/fairfax-va-site and inspect the logs.",
            ))
            .await
            .expect("append old local user");
        harness
            .state
            .append_message(&msg(
                "assistant",
                "I am currently locked to /Users/davidloor/projects/fairfax-va-site.",
            ))
            .await
            .expect("append old local assistant");
        harness
            .state
            .append_message(&msg(
                "user",
                "Find melanoma clinical trials recruiting near Fairfax, Virginia.",
            ))
            .await
            .expect("append external user");
        harness
            .state
            .append_message(&msg(
                "assistant",
                "I found 10 recruiting trials near Fairfax, VA with male participants.",
            ))
            .await
            .expect("append external assistant");

        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history(
                "test-session",
                "Give me all the info about the top 2.",
            )
            .await;

        assert_eq!(turn_context.primary_project_scope, None);
    }

    #[tokio::test]
    async fn build_turn_context_keeps_detailed_schedule_request_separate_from_previous_task() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg(
                "user",
                "Can you post your daily blog post on your blog?",
            ))
            .await
            .expect("append prior user");
        harness
            .state
            .append_message(&msg("assistant", "Would you like me to schedule that too?"))
            .await
            .expect("append prior assistant");

        let current = "Can you set up a daily scheduled task at 6:00 am to publish the blog with honest reflections about recent errors and fixes.";
        let turn_context = harness
            .agent
            .build_turn_context_from_recent_history("test-session", current)
            .await;

        assert_eq!(turn_context.followup_mode, Some(FollowupMode::NewTask));
        assert_eq!(turn_context.goal_user_text, current);
        assert!(turn_context.recent_messages.is_empty());
        assert_eq!(
            turn_context.completion_contract.task_kind,
            CompletionTaskKind::Schedule
        );
        assert!(!turn_context.completion_contract.requires_observation);
    }

    proptest! {
        #[test]
        fn sanitize_carryover_blocks_removes_known_markers(payload in ".*") {
            let input = format!(
                "Original request:\n{}\n\nAssistant asked:\n{}\n\nFollow-up:\n{}",
                payload, payload, payload
            );
            let (sanitized, changed) = sanitize_carryover_blocks(&input);
            prop_assert!(changed);
            prop_assert!(!sanitized.contains("Original request:"));
            prop_assert!(!sanitized.contains("Assistant asked:"));
            prop_assert!(!sanitized.contains("Follow-up:"));
        }
    }

    proptest! {
        #[test]
        fn standalone_cross_project_requests_do_not_classify_as_followup(
            scope in "[a-z0-9_-]{3,20}",
            verb in prop::sample::select(vec!["compare", "analyze", "scan", "review", "audit"]),
        ) {
            let current = format!(
                "{} dependencies across all my projects in ~/projects/{}. Don't ask questions, just do it.",
                verb, scope
            );
            let (mode, reasons) = classify_followup_mode(
                &current,
                Some("Which directories should I inspect?")
            );
            prop_assert_eq!(mode, FollowupMode::NewTask);
            prop_assert!(reasons.contains(&TurnContextReason::FollowupOverrideStandalone));
        }
    }

    #[test]
    fn scope_extraction_rejects_api_endpoint_paths() {
        // /api/notes is a REST API endpoint, not a filesystem path.
        // /tmp/notes_api/ is the actual project directory.
        let mut scopes = Vec::new();
        extract_project_scopes_from_text(
            "Build /api/notes endpoint. Create everything in /tmp/notes_api/",
            &mut scopes,
            5,
            &[],
        );
        // /api doesn't exist on disk, so /api/notes should be rejected
        assert!(
            !scopes.iter().any(|s| s.contains("/api/notes")),
            "API endpoint path should not be extracted as scope, got: {:?}",
            scopes
        );
        // /tmp exists, so /tmp/notes_api should be accepted
        assert!(
            scopes.iter().any(|s| s.contains("notes_api")),
            "Real filesystem path should be extracted, got: {:?}",
            scopes
        );
    }

    #[test]
    fn first_dir_component_exists_filters_correctly() {
        assert!(crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/tmp/test")
        ));
        assert!(!crate::tools::fs_utils::first_dir_component_exists(
            std::path::Path::new("/api/notes")
        ));
    }

    #[test]
    fn hostname_urls_not_treated_as_filesystem_paths() {
        // URLs without protocol prefix should NOT be treated as file paths
        assert!(!token_looks_like_filesystem_path(
            "api.waqi.info/feed/miami/?token=demo"
        ));
        assert!(!token_looks_like_filesystem_path(
            "example.com/path/to/resource"
        ));
        assert!(!token_looks_like_filesystem_path("wttr.in/Miami?format=j1"));
        // But real relative paths with dots should still work
        assert!(token_looks_like_filesystem_path("./src/main.rs"));
        assert!(token_looks_like_filesystem_path("../parent/file.txt"));
        // Dot-prefixed directories (hidden dirs) should work
        assert!(token_looks_like_filesystem_path(".hidden/config"));
        // Absolute paths should work
        assert!(token_looks_like_filesystem_path("/tmp/weather.py"));
        assert!(token_looks_like_filesystem_path("/usr/local/bin"));
        // Simple relative paths without dots before slash should work
        assert!(token_looks_like_filesystem_path("src/main.rs"));
    }

    #[test]
    fn url_without_protocol_not_extracted_as_project_scope() {
        let text = "Fetch from api.waqi.info/feed/miami/?token=demo and save to /tmp/weather.py";
        let mut scopes = Vec::new();
        extract_project_scopes_from_text(text, &mut scopes, 5, &[]);
        assert!(
            !scopes
                .iter()
                .any(|s| s.contains("waqi") || s.contains("api.")),
            "URL hostname should not be extracted as scope, got: {:?}",
            scopes
        );
        assert!(
            scopes.iter().any(|s| s.contains("tmp")),
            "Real path /tmp should be extracted, got: {:?}",
            scopes
        );
    }

    #[test]
    fn verification_targets_do_not_resolve_plain_word_nicknames_without_local_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let targets = extract_verification_targets(
            "Find recruiting studies in Fairfax, Virginia and summarize them.",
            &alias_roots,
        );

        assert!(
            targets
                .iter()
                .all(|target| target.kind != VerificationTargetKind::ProjectScope),
            "plain-word nickname should not resolve without local scope cues: {:?}",
            targets
        );
    }

    #[test]
    fn verification_targets_allow_plain_word_nicknames_with_explicit_project_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        assert!(should_allow_verification_nickname_scope(
            "Check the Fairfax project for broken links.",
            "Fairfax"
        ));
        assert_eq!(
            crate::tools::fs_utils::resolve_contextual_project_nickname_in_explicit_roots(
                "Fairfax",
                &alias_roots
            ),
            Some(project.clone())
        );

        let targets = extract_verification_targets(
            "Check the Fairfax project for broken links.",
            &alias_roots,
        );

        assert!(
            targets.iter().any(|target| {
                target.kind == VerificationTargetKind::ProjectScope
                    && target.value == project.to_string_lossy()
            }),
            "explicit local scope cue should still allow nickname resolution: {:?}",
            targets
        );
    }
}
