use regex::Regex;

use super::{IntentGateDecision, ENABLE_SCHEDULE_HEURISTICS};

/// Complexity classification for orchestration routing.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum IntentComplexity {
    /// Answer from memory/knowledge, no executor needed.
    Knowledge,
    /// Simple task — falls through to full agent loop.
    Simple,
    /// Multi-step complex task, create a goal and fall through to current agent loop.
    Complex,
    /// User asks for recurring/ongoing behavior but did not provide timing.
    ScheduledMissingTiming,
    /// Scheduled task intent requiring deferred/recurring goal creation.
    Scheduled {
        schedule_raw: String,
        schedule_cron: Option<String>,
        is_one_shot: bool,
        schedule_type_explicit: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConnectedApiIntent {
    RuntimeCapabilityValidation,
    ReadAction,
    WriteAction,
}

/// Check if a phrase appears as complete words in text (word-boundary matching).
/// Splits on whitespace, trims surrounding punctuation (preserving apostrophes),
/// then checks for consecutive word matches. Case-insensitive.
///
/// Works for single keywords ("deploy"), multi-word phrases ("set up"),
/// and contractions ("i'll check").
pub(super) fn contains_keyword_as_words(text: &str, keyword: &str) -> bool {
    let normalize = |w: &str| -> String {
        w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
            .to_lowercase()
    };
    let text_words: Vec<String> = text
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty())
        .collect();
    let kw_words: Vec<String> = keyword
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty())
        .collect();
    if kw_words.is_empty() {
        return false;
    }
    text_words
        .windows(kw_words.len())
        .any(|window| window == kw_words.as_slice())
}

/// Detect "about this scheduled goal" meta-queries so they aren't misread as
/// fresh scheduling requests when quoted text contains timing words.
fn is_schedule_reference_query(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let mentions_schedule_subject = contains_keyword_as_words(&lower, "scheduled goal")
        || contains_keyword_as_words(&lower, "recurring goal")
        || contains_keyword_as_words(&lower, "schedule");

    if !mentions_schedule_subject {
        return false;
    }

    contains_keyword_as_words(&lower, "details about")
        || contains_keyword_as_words(&lower, "tell me about")
        || contains_keyword_as_words(&lower, "show me")
        || contains_keyword_as_words(&lower, "list")
        || contains_keyword_as_words(&lower, "what is")
        || contains_keyword_as_words(&lower, "what's")
        || contains_keyword_as_words(&lower, "explain")
        || contains_keyword_as_words(&lower, "describe")
        || lower.contains("scheduled goal:")
}

/// Detect obvious scheduling phrases in user text as a fallback when the model
/// omits schedule fields in [INTENT_GATE].
///
/// Returns (schedule_raw, is_one_shot).
pub(super) fn detect_schedule_heuristic(user_text: &str) -> Option<(String, bool)> {
    let text = user_text.trim();
    if text.is_empty() {
        return None;
    }
    if is_schedule_reference_query(text) {
        return None;
    }
    crate::cron_utils::extract_schedule_from_text(text)
}

/// Detect recurring-intent language when the user did not provide concrete timing.
/// Used to prevent accidental fallback into non-recurring "complex" goals.
pub(super) fn looks_like_recurring_intent_without_timing(user_text: &str) -> bool {
    if detect_schedule_heuristic(user_text).is_some() {
        return false;
    }

    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let re_times_per = match Regex::new(r"(?i)\b\d+\s+times?\s+per\s+(day|week|month)\b") {
        Ok(re) => re,
        Err(_) => return false,
    };
    if re_times_per.is_match(user_text) {
        return true;
    }

    for kw in [
        "monitor",
        "recurring",
        "ongoing",
        "long-term",
        "long term",
        "regularly",
        "consistently",
        "every day",
        "each day",
        "per day",
        "per week",
        "per month",
    ] {
        if contains_keyword_as_words(&lower, kw) {
            return true;
        }
    }

    false
}

/// Detect legacy internal-maintenance intents that should run via native
/// heartbeat/memory jobs rather than goal orchestration.
pub(super) fn is_internal_maintenance_intent(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if lower == "maintain knowledge base: process embeddings, consolidate memories, decay old facts"
        || lower
            == "maintain memory health: prune old events, clean up retention, remove stale data"
    {
        return true;
    }

    let knowledge_maintenance = contains_keyword_as_words(&lower, "process embeddings")
        && contains_keyword_as_words(&lower, "consolidate memories")
        && (contains_keyword_as_words(&lower, "decay old facts")
            || contains_keyword_as_words(&lower, "memory decay"));

    let memory_health = contains_keyword_as_words(&lower, "prune old events")
        && (contains_keyword_as_words(&lower, "clean up retention")
            || contains_keyword_as_words(&lower, "retention cleanup"))
        && contains_keyword_as_words(&lower, "stale data");

    knowledge_maintenance || memory_health
}

pub(super) fn infer_intent_gate(user_text: &str, _analysis: &str) -> IntentGateDecision {
    // Deterministic tool-need overrides:
    // 1) explicit filesystem paths
    // 2) explicit local-environment execution/inspection requests
    // 3) connected external API/integration work:
    //    - runtime capability validation
    //    - connected API reads
    //    - connected API writes
    //
    // These requests cannot be satisfied by text-only consultant output and
    // must run through the tool loop, regardless of model intent-gate output.
    if super::user_text_references_filesystem_path(user_text)
        || user_text_requires_local_tool_execution(user_text)
        || classify_connected_api_intent(user_text).is_some()
    {
        return IntentGateDecision {
            can_answer_now: Some(false),
            needs_tools: Some(true),
            needs_clarification: Some(false),
            clarifying_question: None,
            missing_info: Vec::new(),
            complexity: None,
            cancel_intent: None,
            cancel_scope: None,
            is_acknowledgment: None,
            schedule: None,
            schedule_type: None,
            schedule_cron: None,
            domains: Vec::new(),
        };
    }

    // No lexical guessing fallback: rely on explicit model intent-gate fields.
    // Missing fields simply stay None.
    IntentGateDecision {
        can_answer_now: None,
        needs_tools: None,
        needs_clarification: None,
        clarifying_question: None,
        missing_info: Vec::new(),
        complexity: None,
        cancel_intent: None,
        cancel_scope: None,
        is_acknowledgment: None,
        schedule: None,
        schedule_type: None,
        schedule_cron: None,
        domains: Vec::new(),
    }
}

fn user_text_requires_local_tool_execution(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let explicit_run_request = [
        "run the command",
        "run this command",
        "execute the command",
        "execute this command",
        "run command",
        "execute command",
        "run in terminal",
        "execute in terminal",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    let local_scope_request = [
        "on this machine",
        "on this system",
        "on this host",
        "on this computer",
        "on my machine",
        "on my system",
        "on my host",
        "on my computer",
        "in this environment",
        "in my environment",
        "locally",
        "installed here",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    let installed_wording = contains_keyword_as_words(&lower, "installed")
        || contains_keyword_as_words(&lower, "is installed");
    let installed_version_query = (contains_keyword_as_words(&lower, "what version of")
        || contains_keyword_as_words(&lower, "which version of"))
        && installed_wording
        && local_scope_request;

    let quoted_command_execution = lower.contains('`')
        && (contains_keyword_as_words(&lower, "run")
            || contains_keyword_as_words(&lower, "execute"));

    explicit_run_request
        || local_scope_request
        || installed_version_query
        || quoted_command_execution
}

fn contains_any_as_words(text: &str, keywords: &[&str]) -> bool {
    keywords
        .iter()
        .any(|kw| contains_keyword_as_words(text, kw))
}

pub(super) fn user_text_requests_runtime_capability_validation(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    // Narrow deterministic trigger: only fire when the user explicitly asks
    // for a live check/verification of current capability or connection state.
    for phrase in [
        "check if you can",
        "check whether you can",
        "see if you can",
        "see whether you can",
        "verify you can",
        "verify if you can",
        "verify whether you can",
        "test if you can",
        "test whether you can",
        "confirm you can",
        "confirm whether you can",
        "make sure you can",
        "check if you have access",
        "check whether you have access",
        "verify you have access",
        "check if you're connected",
        "check whether you're connected",
        "verify you're connected",
        "see if you're connected",
        "see whether you're connected",
        "check if it is connected",
        "check whether it is connected",
        "check if this is connected",
        "check whether this is connected",
        "are you connected to",
        "are you hooked up to",
    ] {
        if contains_keyword_as_words(&lower, phrase) {
            return true;
        }
    }

    let has_validation_verb = ["check", "verify", "test", "confirm", "validate", "see"]
        .iter()
        .any(|kw| contains_keyword_as_words(&lower, kw));
    let asks_about_current_access_or_connection = [
        "do you have access",
        "you have access",
        "are you connected",
        "you're connected",
        "it is connected",
        "this is connected",
        "whether you can",
    ]
    .iter()
    .any(|kw| contains_keyword_as_words(&lower, kw));

    has_validation_verb && asks_about_current_access_or_connection
}

fn mentions_connected_api_target(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    const TARGET_MARKERS: &[&str] = &[
        "api",
        "apis",
        "oauth",
        "webhook",
        "connected account",
        "connected service",
        "service account",
        "integration",
        "github",
        "gitlab",
        "jira",
        "linear",
        "notion",
        "slack",
        "discord",
        "twitter",
        "bluesky",
        "mastodon",
        "linkedin",
        "reddit",
        "gmail",
        "email",
        "calendar",
        "google calendar",
        "google drive",
        "google docs",
        "google sheets",
        "drive",
        "stripe",
        "airtable",
        "salesforce",
        "hubspot",
        "shopify",
        "zendesk",
        "intercom",
        "confluence",
        "trello",
        "asana",
        "clickup",
        "dropbox",
        "youtube",
        "figma",
    ];

    contains_any_as_words(&lower, TARGET_MARKERS)
}

fn mentions_connected_api_resource(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    const RESOURCE_MARKERS: &[&str] = &[
        "issue",
        "issues",
        "pull request",
        "pull requests",
        "pr",
        "prs",
        "ticket",
        "tickets",
        "task",
        "tasks",
        "comment",
        "comments",
        "reply",
        "replies",
        "message",
        "messages",
        "email",
        "emails",
        "draft",
        "drafts",
        "inbox",
        "calendar event",
        "calendar events",
        "event",
        "events",
        "page",
        "pages",
        "document",
        "documents",
        "doc",
        "docs",
        "sheet",
        "sheets",
        "record",
        "records",
        "row",
        "rows",
        "database",
        "file",
        "files",
        "folder",
        "folders",
        "repository",
        "repositories",
        "repo",
        "repos",
        "post",
        "posts",
        "tweet",
        "tweets",
        "thread",
        "threads",
        "invoice",
        "invoices",
        "customer",
        "customers",
        "contact",
        "contacts",
        "lead",
        "leads",
    ];

    contains_any_as_words(&lower, RESOURCE_MARKERS)
}

fn mentions_connected_api_account_scope(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "my", "our", "for me", "from my", "to my", "in my", "on my", "with my",
        ],
    )
}

pub(super) fn user_text_requests_connected_api_write_action(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if !mentions_connected_api_target(&lower) {
        return false;
    }

    let has_strong_write_verb = contains_any_as_words(
        &lower,
        &[
            "post", "publish", "send", "reply", "comment", "share", "upload", "submit", "tweet",
            "retweet", "message", "dm",
        ],
    );
    // "open" is excluded — it conflicts with "open issues" (adjective meaning
    // "not-closed") and "open GitHub" (launch).  It rarely signals a write action.
    let has_scoped_write_verb = contains_any_as_words(
        &lower,
        &[
            "create", "update", "edit", "modify", "delete", "remove", "close", "reopen", "merge",
            "archive", "schedule", "cancel",
        ],
    ) && mentions_connected_api_resource(&lower);

    has_strong_write_verb || has_scoped_write_verb
}

pub(super) fn user_text_requests_connected_api_read_action(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if !mentions_connected_api_target(&lower) {
        return false;
    }

    // "read" is excluded — it conflicts with "read the docs" (local reading)
    // and is rarely used for API fetching ("list my issues", not "read my issues").
    let has_read_verb = contains_any_as_words(
        &lower,
        &[
            "check", "list", "show", "fetch", "get", "view", "inspect", "look up", "lookup",
            "find", "search", "pull", "sync",
        ],
    );

    has_read_verb
        && (mentions_connected_api_account_scope(&lower) || mentions_connected_api_resource(&lower))
}

pub(super) fn classify_connected_api_intent(user_text: &str) -> Option<ConnectedApiIntent> {
    if user_text_requests_runtime_capability_validation(user_text) {
        Some(ConnectedApiIntent::RuntimeCapabilityValidation)
    } else if user_text_requests_connected_api_write_action(user_text) {
        Some(ConnectedApiIntent::WriteAction)
    } else if user_text_requests_connected_api_read_action(user_text) {
        Some(ConnectedApiIntent::ReadAction)
    } else {
        None
    }
}

/// Classify user intent complexity for orchestration routing.
///
/// Uses the LLM-provided `complexity` field from the `[INTENT_GATE]` JSON.
/// Falls back to `Simple` when the field is absent or unrecognized.
///
/// Guardrails override the LLM's "complex" classification for messages that
/// are clearly simple — the consultant LLM over-classifies short commands,
/// acknowledgments, and single-action requests as complex.
pub(super) fn classify_intent_complexity(
    user_text: &str,
    intent_gate: &IntentGateDecision,
) -> (IntentComplexity, Vec<String>) {
    // Heuristic schedule extraction: if the model omitted schedule fields but the
    // user message contains a concrete schedule phrase, treat it as scheduled
    // rather than falling back into the tool loop (which can spiral).
    if ENABLE_SCHEDULE_HEURISTICS
        && intent_gate.schedule.is_none()
        && intent_gate.schedule_cron.is_none()
    {
        if let Some((schedule_raw, is_one_shot)) = detect_schedule_heuristic(user_text) {
            return (
                IntentComplexity::Scheduled {
                    schedule_raw,
                    schedule_cron: None,
                    is_one_shot,
                    schedule_type_explicit: false,
                },
                vec![],
            );
        }
    }

    // If user clearly wants recurring behavior but no timing could be extracted,
    // ask for schedule details instead of silently creating a non-recurring goal.
    if ENABLE_SCHEDULE_HEURISTICS
        && intent_gate.schedule.is_none()
        && intent_gate.schedule_cron.is_none()
        && looks_like_recurring_intent_without_timing(user_text)
    {
        return (IntentComplexity::ScheduledMissingTiming, vec![]);
    }

    // Schedule takes priority over all other classifications.
    if let Some(ref schedule_raw) = intent_gate.schedule {
        let schedule_type_explicit = intent_gate.schedule_type.is_some();
        let is_one_shot = intent_gate.schedule_type.as_deref() == Some("one_shot");
        return (
            IntentComplexity::Scheduled {
                schedule_raw: schedule_raw.clone(),
                schedule_cron: intent_gate.schedule_cron.clone(),
                is_one_shot,
                schedule_type_explicit,
            },
            vec![],
        );
    }
    if let Some(ref schedule_cron) = intent_gate.schedule_cron {
        let schedule_type_explicit = intent_gate.schedule_type.is_some();
        let is_one_shot = intent_gate.schedule_type.as_deref() == Some("one_shot");
        return (
            IntentComplexity::Scheduled {
                schedule_raw: schedule_cron.clone(),
                schedule_cron: Some(schedule_cron.clone()),
                is_one_shot,
                schedule_type_explicit,
            },
            vec![],
        );
    }

    if intent_gate.can_answer_now.unwrap_or(false) && !intent_gate.needs_tools.unwrap_or(false) {
        return (IntentComplexity::Knowledge, vec![]);
    }

    // Consultant-empty fallback: when the model omits `complexity`, promote
    // obviously cross-project / multi-question, multi-step requests to Complex.
    if intent_gate.complexity.is_none() && looks_like_complex_request_fallback(user_text) {
        return (IntentComplexity::Complex, vec![]);
    }

    // When can_answer_now=false, don't classify as Knowledge even if
    // complexity="knowledge" — the model can't answer, so we should
    // try tools (memory search, manage_people, etc.) as Simple.
    match intent_gate.complexity.as_deref() {
        Some("knowledge") => (IntentComplexity::Simple, vec![]),
        Some("complex") => (IntentComplexity::Complex, vec![]),
        _ => (IntentComplexity::Simple, vec![]),
    }
}

fn looks_like_complex_request_fallback(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let cross_project_scope = contains_keyword_as_words(&lower, "all my projects")
        || contains_keyword_as_words(&lower, "across all projects")
        || contains_keyword_as_words(&lower, "across my projects")
        || (contains_keyword_as_words(&lower, "all projects")
            && contains_keyword_as_words(&lower, "compare"));

    let action_markers = [
        "compare",
        "analyze",
        "analyse",
        "identify",
        "find",
        "count",
        "calculate",
        "summarize",
        "report",
        "list",
    ];
    let action_hits = action_markers
        .iter()
        .filter(|marker| contains_keyword_as_words(&lower, marker))
        .count();
    let multi_question = user_text.matches('?').count() >= 2;
    let compound_request =
        lower.contains(" and ") || lower.contains(" then ") || lower.contains(" also ");

    cross_project_scope
        || (action_hits >= 3 && (multi_question || compound_request))
        || (action_hits >= 2 && user_text.chars().count() >= 140)
}

#[cfg(test)]
mod intent_routing_path_override_tests {
    use super::*;

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_urls() {
        let d = infer_intent_gate("https://example.com/foo/bar", "");
        assert!(
            d.needs_tools.is_none(),
            "expected needs_tools=None for URL, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_common_slash_shorthand() {
        for text in ["3/4", "2/14", "yes/no", "w/o"] {
            let d = infer_intent_gate(text, "");
            assert!(
                d.needs_tools.is_none(),
                "expected needs_tools=None for '{}', got {:?}",
                text,
                d.needs_tools
            );
        }
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_unix_paths() {
        let d = infer_intent_gate("/Users/alice/project/file.txt", "");
        assert_eq!(d.needs_tools, Some(true));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_tilde_paths() {
        let d = infer_intent_gate("~/project/file.txt", "");
        assert_eq!(d.needs_tools, Some(true));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_windows_paths() {
        let d = infer_intent_gate(r"C:\Users\alice\file.txt", "");
        assert_eq!(d.needs_tools, Some(true));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_local_installed_version_query() {
        let d = infer_intent_gate("What version of rustc is installed on this machine?", "");
        assert_eq!(d.needs_tools, Some(true));
        assert_eq!(d.can_answer_now, Some(false));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_explicit_run_command_request() {
        let d = infer_intent_gate(
            "Run the command `cat /nonexistent/file.txt` and tell me what happens",
            "",
        );
        assert_eq!(d.needs_tools, Some(true));
        assert_eq!(d.can_answer_now, Some(false));
    }

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_generic_version_question() {
        let d = infer_intent_gate("What version of rustc was released in 2021?", "");
        assert!(
            d.needs_tools.is_none(),
            "expected needs_tools=None for generic version question, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_non_local_installed_question() {
        let d = infer_intent_gate("What version of Python is installed at the company?", "");
        assert!(
            d.needs_tools.is_none(),
            "expected needs_tools=None for non-local installed wording, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_runtime_capability_validation() {
        let d = infer_intent_gate("Can you check if you can post to Twitter right now?", "");
        assert_eq!(d.needs_tools, Some(true));
        assert_eq!(d.can_answer_now, Some(false));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_connection_state_validation() {
        let d = infer_intent_gate(
            "Please verify you're connected to GitHub before answering.",
            "",
        );
        assert_eq!(d.needs_tools, Some(true));
        assert_eq!(d.can_answer_now, Some(false));
    }

    #[test]
    fn infer_intent_gate_does_not_force_tools_for_generic_capability_question() {
        let d = infer_intent_gate("Can you help me write a tweet?", "");
        assert!(
            d.needs_tools.is_none(),
            "expected needs_tools=None for generic capability question, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn classify_connected_api_intent_detects_runtime_validation() {
        assert_eq!(
            classify_connected_api_intent(
                "Please check whether you're connected to GitHub before you answer."
            ),
            Some(ConnectedApiIntent::RuntimeCapabilityValidation)
        );
    }

    #[test]
    fn classify_connected_api_intent_detects_write_actions() {
        assert_eq!(
            classify_connected_api_intent("Create a GitHub issue for this regression."),
            Some(ConnectedApiIntent::WriteAction)
        );
        assert_eq!(
            classify_connected_api_intent("Post this update to LinkedIn."),
            Some(ConnectedApiIntent::WriteAction)
        );
    }

    #[test]
    fn classify_connected_api_intent_detects_read_actions() {
        assert_eq!(
            classify_connected_api_intent("List my open GitHub issues."),
            Some(ConnectedApiIntent::ReadAction)
        );
        assert_eq!(
            classify_connected_api_intent("Check my calendar events for tomorrow."),
            Some(ConnectedApiIntent::ReadAction)
        );
    }

    #[test]
    fn classify_connected_api_intent_avoids_local_or_implementation_prompts() {
        assert_eq!(
            classify_connected_api_intent("Can you rewrite this paragraph?"),
            None
        );
        assert_eq!(
            classify_connected_api_intent("Write a GitHub Actions workflow for this repo."),
            None
        );
        assert_eq!(
            classify_connected_api_intent("Read the GitHub Actions docs and summarize them."),
            None
        );
    }
}
