//! Follow-up / task-switch classification for incoming user turns.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. Owns [`FollowupMode`], [`TurnContextReason`], and the
//! `looks_like_*` / `classify_followup_mode` family.

use super::completion_contract::{looks_like_question_request, HTTP_URL_RE};
use super::project_scope::{
    extract_project_hints_from_text, extract_project_scopes_from_text,
    token_looks_like_filesystem_path,
};
use super::*;

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

pub(super) fn find_previous_turns(
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
            "assistant" if saw_current_user && prev_assistant.is_none() => {
                if let Some(content) = msg.content.as_deref() {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case(current_user_text) {
                        prev_assistant = Some(trimmed.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    (prev_assistant, prev_user)
}

pub(super) fn assistant_message_looks_like_clarifying_question(message: &str) -> bool {
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
        "want me to",
        "should i",
        "shall i",
        "can you clarify",
        "any specific",
        "do you prefer",
        "prefer",
        "what style",
        "what elements",
    ];
    clarifying_markers.iter().any(|m| lower.contains(m))
}

pub(super) fn looks_like_explicit_task_switch(lower_text: &str) -> bool {
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

/// Detect strong followup indicators that override the length heuristic.
/// These are phrases that unambiguously reference prior conversation context,
/// e.g. "follow-up on the newsletter you just created".
fn has_strong_followup_indicators(lower_text: &str) -> bool {
    // Explicit followup phrases
    let followup_phrases = [
        "follow-up on",
        "follow up on",
        "following up on",
        "back to the",
        "back to what you",
        "continuing from",
        "continuation of",
        "regarding what you",
        "about what you just",
    ];
    if followup_phrases.iter().any(|p| lower_text.contains(p)) {
        return true;
    }
    // Recency markers that reference something the agent just did
    let recency_refs = [
        "you just created",
        "you just made",
        "you just wrote",
        "you just generated",
        "you just built",
        "you just saved",
        "we just discussed",
        "we just talked about",
        "the one you just",
        "the file you just",
        "the script you just",
        "that you just",
    ];
    recency_refs.iter().any(|r| lower_text.contains(r))
}

pub(super) fn looks_like_context_dependent_followup_question(lower_text: &str) -> bool {
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

/// Detect messages that explicitly reference a prior unanswered request:
/// "You didn't respond my question", "answer my question", "you ignored my request".
/// These are meta-commentary about the prior interaction and must keep context.
pub(super) fn looks_like_unanswered_request_reference(lower: &str) -> bool {
    if lower.chars().count() > 120 {
        return false;
    }
    let complaint_cues = [
        "didn't respond",
        "didn't answer",
        "did not respond",
        "did not answer",
        "didn't reply",
        "did not reply",
        "never responded",
        "never answered",
        "never replied",
        "ignored",
        "skipped",
        "didn't address",
        "did not address",
        "what about",
    ];
    let response_verbs = ["answer", "respond", "reply", "address"];
    let prior_request_references = [
        "my question",
        "my request",
        "my message",
        "my earlier question",
        "my earlier request",
        "my last question",
        "my last request",
        "what i asked",
        "the question i asked",
        "the request i made",
    ];

    let references_prior_request = prior_request_references
        .iter()
        .any(|phrase| contains_keyword_as_words(lower, phrase));
    if !references_prior_request {
        return false;
    }

    let has_complaint_cue = complaint_cues
        .iter()
        .any(|phrase| contains_keyword_as_words(lower, phrase));
    let asks_for_response = response_verbs
        .iter()
        .any(|phrase| contains_keyword_as_words(lower, phrase));

    has_complaint_cue || asks_for_response
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

pub(super) fn looks_like_standalone_goal_request(lower_text: &str) -> bool {
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

pub(super) fn looks_like_self_contained_mutation_request(current: &str, lower_text: &str) -> bool {
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

pub(super) fn looks_like_short_command_request(current: &str) -> bool {
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

pub(super) fn looks_like_scope_carryover_ack(current: &str) -> bool {
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

pub(super) fn looks_like_multi_project_request(lower_text: &str) -> bool {
    contains_keyword_as_words(lower_text, "all my projects")
        || contains_keyword_as_words(lower_text, "across all projects")
        || contains_keyword_as_words(lower_text, "across my projects")
        || contains_keyword_as_words(lower_text, "every project")
        || (contains_keyword_as_words(lower_text, "all projects")
            && contains_keyword_as_words(lower_text, "compare"))
}

pub(super) fn text_contains_any_phrase(text: &str, phrases: &[&str]) -> bool {
    phrases
        .iter()
        .any(|phrase| contains_keyword_as_words(text, phrase))
}

pub(super) fn sanitize_carryover_blocks(input: &str) -> (String, bool) {
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

pub(super) fn classify_followup_mode(
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

    // Synthetic re-engagement messages from background command completions
    // must always be follow-ups so the original task context is preserved.
    if trimmed.starts_with("[Background command completed]") {
        reasons.push(TurnContextReason::ExplicitFollowup);
        return (FollowupMode::Followup, reasons);
    }

    // Strong followup indicators that should override the length heuristic.
    // Phrases like "follow-up on the X you just created" clearly reference
    // prior context regardless of message length.
    if has_strong_followup_indicators(&lower) && !looks_like_explicit_task_switch(&lower) {
        reasons.push(TurnContextReason::ExplicitFollowup);
        return (FollowupMode::Followup, reasons);
    }

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

    // Messages that reference a prior unanswered request ("You didn't respond
    // my question", "answer my question", "you ignored my request") must keep
    // prior context so the LLM can find the original question.
    if prev_assistant.is_some() && looks_like_unanswered_request_reference(&lower) {
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

pub(super) fn has_project_scope_divergence_with_aliases(
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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
    fn followup_accepts_answer_to_want_me_to_clarification() {
        let current = "Post it.";
        let prev = "Please answer directly: Want me to tweak this or post it?";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::ClarificationAnswer);
        assert!(reasons.contains(&TurnContextReason::ClarificationAnswer));
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
    fn unanswered_request_complaint_classified_as_followup() {
        let prev = "I searched for AI news and found several results.";
        let current = "You didn't respond my question";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::Followup);
        assert!(reasons.contains(&TurnContextReason::ContextDependentQuestion));
    }
    #[test]
    fn answer_my_question_classified_as_followup() {
        let prev = "Here is the latest result excerpt.";
        let current = "answer my question please";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::Followup);
        assert!(reasons.contains(&TurnContextReason::ContextDependentQuestion));
    }
    #[test]
    fn ignored_my_request_classified_as_followup() {
        let prev = "I posted the deployment logs.";
        let current = "You ignored my request";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::Followup);
        assert!(reasons.contains(&TurnContextReason::ContextDependentQuestion));
    }
    #[test]
    fn what_about_my_request_classified_as_followup() {
        let prev = "I summarized the bug reports.";
        let current = "What about my request?";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::Followup);
        assert!(reasons.contains(&TurnContextReason::ContextDependentQuestion));
    }
    #[test]
    fn unanswered_request_reference_uses_word_boundaries() {
        let prev = "I searched the docs and posted the summary.";
        let current = "Please answer myquestion now";
        let (mode, reasons) = classify_followup_mode(current, Some(prev));
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::DefaultNewTask));
    }
    #[test]
    fn unanswered_complaint_without_prev_assistant_stays_new_task() {
        // No previous assistant message — can't be a followup
        let current = "You didn't respond my question";
        let (mode, reasons) = classify_followup_mode(current, None);
        assert_eq!(mode, FollowupMode::NewTask);
        assert!(reasons.contains(&TurnContextReason::DefaultNewTask));
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
    fn short_command_request_requires_structured_target() {
        assert!(looks_like_short_command_request("run cargo test"));
        assert!(!looks_like_short_command_request("check that"));
        assert!(!looks_like_short_command_request("show it again"));
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
}
