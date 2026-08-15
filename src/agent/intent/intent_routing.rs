#[cfg(test)]
use super::IntentGateDecision;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConnectedApiIntent {
    RuntimeCapabilityValidation,
    ReadAction,
    WriteAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ConnectedContentMode {
    #[default]
    None,
    #[cfg(test)]
    DraftOnly,
    #[cfg(test)]
    DeliverOnly,
    #[cfg(test)]
    DraftThenDeliver,
}

impl ConnectedContentMode {
    pub(super) fn is_authoring_only(self) -> bool {
        #[cfg(test)]
        {
            return matches!(self, Self::DraftOnly);
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    #[cfg(test)]
    pub(super) fn expects_live_delivery(self) -> bool {
        matches!(self, Self::DeliverOnly | Self::DraftThenDeliver)
    }
}

/// Check if a phrase appears as complete words in text (word-boundary matching).
/// Splits on whitespace, trims surrounding punctuation (preserving apostrophes),
/// then checks for consecutive word matches. Case-insensitive.
///
/// Works for single keywords ("deploy"), multi-word phrases ("set up"),
/// and contractions ("i'll check").
pub(crate) fn contains_keyword_as_words(text: &str, keyword: &str) -> bool {
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

/// Returns `true` when this turn is a background-output delivery injected by
/// the terminal tool's re-engagement path (see `tools/terminal.rs`).
/// These turns already contain the command output and must not be forced
/// through the tool loop — the model only needs to interpret and reply.
#[cfg(test)]
fn is_observation_delivery_turn(user_text: &str) -> bool {
    user_text.contains("[Background command completed]")
}

#[cfg(test)]
pub(super) fn infer_intent_gate(user_text: &str, _analysis: &str) -> IntentGateDecision {
    // Short-circuit: background-output delivery turns never need tools.
    // The terminal re-engagement injects a synthetic message starting with
    // "[Background command completed]" — the output is already present;
    // forcing tool_choice=required causes degenerate repetition on some models.
    if is_observation_delivery_turn(user_text) {
        return IntentGateDecision {
            needs_tools: Some(false),
            ..Default::default()
        };
    }

    let user_text = crate::channels::attachments::user_authored_text(user_text);
    // Deterministic tool-need overrides:
    // 1) explicit filesystem paths
    // 2) explicit local-environment execution/inspection requests
    // 3) connected external API/integration work:
    //    - runtime capability validation
    //    - connected API reads
    //    - connected API writes
    //
    // These requests cannot be satisfied by a text-only first-pass response and
    // must run through the tool loop.
    if super::user_text_references_filesystem_path(&user_text)
        || user_text_requires_local_tool_execution(&user_text)
        || user_text_requests_auth_or_integration_management(&user_text)
        || classify_connected_api_intent(&user_text).is_some()
        || recognized_artifact_creation_request(&user_text)
    {
        return IntentGateDecision {
            needs_tools: Some(true),
            ..Default::default()
        };
    }

    // No lexical guessing fallback: missing signals simply stay None.
    IntentGateDecision::default()
}

/// Deterministic lexical recognition for straightforward file/artifact
/// creation. It deliberately requires both an authoring action and an explicit
/// artifact type so ordinary questions about writing or presentations do not
/// get routed into execution.
#[cfg(test)]
pub(crate) fn recognized_artifact_creation_request(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let creation_action = [
        "create", "make", "generate", "build", "produce", "prepare", "export",
    ]
    .iter()
    .any(|keyword| contains_keyword_as_words(&lower, keyword));
    if !creation_action {
        return false;
    }

    [
        "file",
        "document",
        "docx",
        "pdf",
        "spreadsheet",
        "workbook",
        "xlsx",
        "csv",
        "presentation",
        "powerpoint",
        "pptx",
        "slide deck",
        "slides",
        "report",
        "image",
        "png",
        "jpeg",
        "jpg",
        "archive",
        "zip",
    ]
    .iter()
    .any(|keyword| contains_keyword_as_words(&lower, keyword))
}

#[cfg(test)]
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

#[cfg(test)]
fn contains_any_as_words(text: &str, keywords: &[&str]) -> bool {
    keywords
        .iter()
        .any(|kw| contains_keyword_as_words(text, kw))
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
#[cfg(test)]
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

#[cfg(test)]
fn mentions_connected_api_account_scope(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "my account",
            "our account",
            "your account",
            "their account",
            "his account",
            "her account",
            "for me",
            "for us",
            "for you",
            "for them",
            "for him",
            "for her",
            "from my",
            "from our",
            "from your",
            "from their",
            "from his",
            "from her",
            "to my",
            "to our",
            "to your",
            "to their",
            "to his",
            "to her",
            "in my",
            "in our",
            "in your",
            "in their",
            "in his",
            "in her",
            "on my",
            "on our",
            "on your",
            "on their",
            "on his",
            "on her",
            "with my",
            "with our",
            "with your",
            "with their",
            "with his",
            "with her",
            "using my",
            "using our",
            "using your",
            "using their",
            "using his",
            "using her",
        ],
    )
}

#[cfg(test)]
fn mentions_existing_connected_content_payload(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "this update",
            "this post",
            "this tweet",
            "this thread",
            "this email",
            "this message",
            "this reply",
            "this comment",
            "the following",
            "below",
            "above",
            "already drafted",
            "existing draft",
            "prepared draft",
        ],
    )
}

#[cfg(test)]
fn has_content_delivery_resource(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "post",
            "posts",
            "tweet",
            "tweets",
            "thread",
            "threads",
            "email",
            "emails",
            "message",
            "messages",
            "reply",
            "replies",
            "comment",
            "comments",
            "caption",
            "captions",
            "announcement",
            "announcements",
        ],
    )
}

#[cfg(test)]
fn has_content_delivery_verb(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "post",
            "publish",
            "send",
            "reply",
            "comment",
            "share",
            "upload",
            "submit",
            "retweet",
            "schedule",
            "tweet this",
            "email this",
            "message this",
            "dm this",
        ],
    )
}

#[cfg(test)]
fn looks_like_direct_reply_format_instruction(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "reply with",
            "reply only with",
            "respond with",
            "respond only with",
            "answer with",
            "answer only with",
        ],
    )
}

#[cfg(test)]
fn has_content_live_delivery_cue(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    mentions_connected_api_account_scope(&lower)
        || contains_any_as_words(
            &lower,
            &[
                "right now",
                "right away",
                "immediately",
                "send it",
                "post it",
                "publish it",
                "submit it",
                "upload it",
                "schedule it",
                "using the connected account",
                "using the connected service",
                "using my connected account",
                "using our connected account",
                "using your connected account",
                "using their connected account",
                "using his connected account",
                "using her connected account",
            ],
        )
        || contains_any_as_words(&lower, &["connected account", "connected service"])
}

#[cfg(test)]
fn has_content_drafting_verb(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() || !has_content_delivery_resource(&lower) {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "write",
            "draft",
            "compose",
            "brainstorm",
            "generate",
            "polish",
            "rewrite",
            "revise",
            "make",
            "create",
            "come up with",
            "help me write",
            "help me draft",
            "give me",
        ],
    )
}

#[cfg(test)]
fn has_content_new_payload_shape(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() || !has_content_delivery_resource(&lower) {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "a post",
            "a tweet",
            "a thread",
            "an email",
            "a message",
            "a reply",
            "a comment",
            "a caption",
            "an announcement",
            "about",
        ],
    )
}

#[cfg(test)]
fn has_content_copywriting_cue(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() || !has_content_delivery_resource(&lower) {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "engaging",
            "catchy",
            "compelling",
            "punchy",
            "polished",
            "professional",
            "friendly",
            "funny",
            "thoughtful",
            "make it",
            "so people",
            "want to comment",
            "call to action",
            "cta",
            "hook",
            "tone",
            "voice",
            "caption",
            "copy",
        ],
    )
}

#[cfg(test)]
fn has_explicit_do_not_deliver_cue(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "do not post",
            "don't post",
            "dont post",
            "do not send",
            "don't send",
            "dont send",
            "do not publish",
            "don't publish",
            "dont publish",
            "just draft",
            "draft only",
            "do not share",
            "don't share",
            "dont share",
        ],
    )
}

#[cfg(test)]
pub(super) fn classify_connected_content_mode(user_text: &str) -> ConnectedContentMode {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() || !has_content_delivery_resource(&lower) {
        return ConnectedContentMode::None;
    }

    let has_connected_target = mentions_connected_api_target(&lower);
    let has_existing_payload = mentions_existing_connected_content_payload(&lower);
    let has_live_delivery_cue = has_content_live_delivery_cue(&lower);
    let has_delivery_verb = has_content_delivery_verb(&lower);
    let has_drafting_verb = has_content_drafting_verb(&lower);
    let has_new_payload_shape = has_content_new_payload_shape(&lower);
    let has_copywriting_cue = has_content_copywriting_cue(&lower);
    let needs_new_copy = has_drafting_verb || has_new_payload_shape || has_copywriting_cue;

    if !has_connected_target
        && !has_existing_payload
        && !has_live_delivery_cue
        && looks_like_direct_reply_format_instruction(&lower)
    {
        return ConnectedContentMode::None;
    }

    if has_explicit_do_not_deliver_cue(&lower) {
        return ConnectedContentMode::DraftOnly;
    }

    if has_existing_payload {
        return ConnectedContentMode::DeliverOnly;
    }

    if has_live_delivery_cue || has_delivery_verb {
        if needs_new_copy {
            return ConnectedContentMode::DraftThenDeliver;
        }
        return ConnectedContentMode::DeliverOnly;
    }

    if needs_new_copy {
        return ConnectedContentMode::DraftOnly;
    }

    ConnectedContentMode::None
}

#[cfg(test)]
pub(crate) fn content_authoring_request_is_text_only(user_text: &str) -> bool {
    classify_connected_content_mode(user_text).is_authoring_only()
}

#[cfg(test)]
pub(crate) fn user_text_requests_auth_or_integration_management(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let has_auth_surface = mentions_connected_api_target(&lower)
        || contains_any_as_words(
            &lower,
            &[
                "oauth",
                "api key",
                "access token",
                "refresh token",
                "bearer token",
                "credentials",
                "credential",
                "auth profile",
                "webhook",
                "integration",
                "connected account",
                "connected service",
                "service account",
            ],
        );
    if !has_auth_surface {
        return false;
    }

    contains_any_as_words(
        &lower,
        &[
            "connect",
            "reconnect",
            "disconnect",
            "authorize",
            "reauthorize",
            "authenticate",
            "log in",
            "login",
            "sign in",
            "sign into",
            "grant access",
            "revoke access",
            "configure",
            "set up",
            "setup",
            "register",
            "repair connection",
            "refresh token",
            "verify connection",
            "manage oauth",
            "manage http auth",
            "manage api",
        ],
    )
}

#[cfg(test)]
pub(super) fn user_text_requests_connected_api_write_action(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    let content_mode = classify_connected_content_mode(&lower);
    let has_explicit_connected_target = mentions_connected_api_target(&lower);
    let has_live_content_delivery_intent = content_mode.expects_live_delivery();
    if !has_explicit_connected_target && !has_live_content_delivery_intent {
        return false;
    }

    if content_mode.is_authoring_only() {
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

    if has_content_delivery_resource(&lower) {
        return has_strong_write_verb && has_live_content_delivery_intent;
    }

    has_strong_write_verb || has_scoped_write_verb
}

#[cfg(test)]
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

#[cfg(test)]
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
    fn infer_intent_gate_ignores_paths_in_channel_file_stub() {
        use crate::channels::attachments::format_file_stub;

        let stub = format_file_stub(
            "photo.jpg",
            221 * 1024,
            "image/jpeg",
            std::path::Path::new("/Users/alice/.aidaemon/files/inbox/photo.jpg"),
        );
        let d = infer_intent_gate(&stub, "");
        assert_ne!(
            d.needs_tools,
            Some(true),
            "channel file stub paths must not force tools, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_user_path_with_attachment_stub() {
        use std::path::PathBuf;

        use crate::channels::attachments::{build_inbound_text, message_attachment};

        let attachments = vec![message_attachment(
            PathBuf::from("/tmp/inbox/doc.pdf"),
            "doc.pdf".to_string(),
            "application/pdf".to_string(),
            100,
        )];
        let inbound = build_inbound_text("Please read ~/project/Cargo.toml", &attachments);
        let d = infer_intent_gate(&inbound, "");
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
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_explicit_run_command_request() {
        let d = infer_intent_gate(
            "Run the command `cat /nonexistent/file.txt` and tell me what happens",
            "",
        );
        assert_eq!(d.needs_tools, Some(true));
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
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_connection_state_validation() {
        let d = infer_intent_gate(
            "Please verify you're connected to GitHub before answering.",
            "",
        );
        assert_eq!(d.needs_tools, Some(true));
    }

    #[test]
    fn infer_intent_gate_forces_tools_for_auth_management_request() {
        let d = infer_intent_gate("Connect my Twitter account so you can post for me.", "");
        assert_eq!(d.needs_tools, Some(true));
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
    fn infer_intent_gate_does_not_force_tools_for_simple_reply_format_request() {
        let d = infer_intent_gate(
            "Chrome smoke test 2026-03-30T02:27:33.908Z: what is 2+2? Reply with just 4.",
            "",
        );
        assert!(
            d.needs_tools.is_none(),
            "expected needs_tools=None for simple reply-format request, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn classify_connected_content_mode_ignores_generic_reply_formatting() {
        assert_eq!(
            classify_connected_content_mode(
                "Chrome smoke test 2026-03-30T02:27:33.908Z: what is 2+2? Reply with just 4."
            ),
            ConnectedContentMode::None
        );
        assert_eq!(
            classify_connected_api_intent(
                "Chrome smoke test 2026-03-30T02:27:33.908Z: what is 2+2? Reply with just 4."
            ),
            None
        );
    }

    #[test]
    fn classify_connected_content_mode_keeps_explicit_live_reply_requests() {
        assert_eq!(
            classify_connected_content_mode("Reply to this tweet right now."),
            ConnectedContentMode::DeliverOnly
        );
        assert_eq!(
            classify_connected_api_intent("Reply to this tweet right now."),
            Some(ConnectedApiIntent::WriteAction)
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
        assert_eq!(
            classify_connected_api_intent("Draft the copy and post it to LinkedIn right now."),
            Some(ConnectedApiIntent::WriteAction)
        );
    }

    #[test]
    fn classify_connected_api_intent_treats_explicit_social_delivery_as_write_action() {
        assert_eq!(
            classify_connected_api_intent(
                "Can you post a tweet about your new stuff and make it engaging so people want to comment?"
            ),
            Some(ConnectedApiIntent::WriteAction)
        );
        assert_eq!(
            classify_connected_content_mode(
                "Can you post a tweet about your new stuff and make it engaging so people want to comment?"
            ),
            ConnectedContentMode::DraftThenDeliver
        );
        assert!(!content_authoring_request_is_text_only(
            "Can you post a tweet about your new stuff and make it engaging so people want to comment?"
        ));
    }

    #[test]
    fn classify_connected_content_mode_detects_draft_only_requests() {
        assert_eq!(
            classify_connected_content_mode("Can you help me write a tweet about our launch?"),
            ConnectedContentMode::DraftOnly
        );
        assert!(content_authoring_request_is_text_only(
            "Can you help me write a tweet about our launch?"
        ));
    }

    #[test]
    fn classify_connected_content_mode_detects_existing_payload_delivery() {
        assert_eq!(
            classify_connected_content_mode("Post this tweet right now."),
            ConnectedContentMode::DeliverOnly
        );
    }

    #[test]
    fn classify_connected_content_mode_respects_do_not_post_cues() {
        assert_eq!(
            classify_connected_content_mode("Draft a tweet about our launch but don't post it."),
            ConnectedContentMode::DraftOnly
        );
    }

    #[test]
    fn classify_connected_api_intent_detects_account_scoped_delivery_without_service_name() {
        assert_eq!(
            classify_connected_api_intent("Can you post a tweet on your account?"),
            Some(ConnectedApiIntent::WriteAction)
        );
        assert!(!content_authoring_request_is_text_only(
            "Can you post a tweet on your account?"
        ));
    }

    #[test]
    fn classify_connected_api_intent_detects_existing_payload_delivery_without_service_name() {
        assert_eq!(
            classify_connected_api_intent("Send this email for me right now."),
            Some(ConnectedApiIntent::WriteAction)
        );
        assert!(!content_authoring_request_is_text_only(
            "Send this email for me right now."
        ));
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
        // "post" triggers both delivery resource and delivery verb detection,
        // resulting in DraftThenDeliver (live delivery), which qualifies as WriteAction.
        assert_eq!(
            classify_connected_api_intent("Draft a LinkedIn post about our launch."),
            Some(ConnectedApiIntent::WriteAction)
        );
    }

    #[test]
    fn auth_management_detection_matches_explicit_connection_requests() {
        assert!(user_text_requests_auth_or_integration_management(
            "Reconnect my GitHub OAuth integration."
        ));
        assert!(user_text_requests_auth_or_integration_management(
            "Set up the Twitter connected account."
        ));
        assert!(!user_text_requests_auth_or_integration_management(
            "Draft a tweet about our launch."
        ));
    }

    #[test]
    fn test_intent_gate_background_delivery_turn_needs_no_tools() {
        let msg = "[Background command completed]\nCommand: `find . -name '*.log' -size +1M`\nOutput:\n./logs/app.log\n./logs/error.log\n\nThis command was part of your previous task.";
        let d = infer_intent_gate(msg, "");
        assert_eq!(
            d.needs_tools,
            Some(false),
            "background-output delivery turn must not require tools, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn test_intent_gate_normal_local_request_still_needs_tools() {
        let d = infer_intent_gate("what's the biggest file on my computer?", "");
        assert_eq!(
            d.needs_tools,
            Some(true),
            "local filesystem request must still require tools, got {:?}",
            d.needs_tools
        );
    }

    #[test]
    fn exact_school_presentation_request_requires_execution() {
        let request = "Can you create a pptx about Ecuador. Make it beautiful as I will present in my daughter's school.";

        assert!(recognized_artifact_creation_request(request));
        assert_eq!(infer_intent_gate(request, "").needs_tools, Some(true));
    }

    #[test]
    fn presentation_advice_does_not_require_artifact_execution() {
        let request = "How do I make presentations beautiful for a school audience?";

        assert!(!recognized_artifact_creation_request(request));
        assert_eq!(infer_intent_gate(request, "").needs_tools, None);
    }

    #[test]
    fn test_intent_gate_delivery_marker_overrides_filesystem_paths() {
        let msg = "[Background command completed]\nCommand: `ls /var/log`\nOutput:\n/var/log/syslog\n/var/log/kern.log\n/Users/alice/project/build.log\n\nThis command was part of your previous task.";
        let d = infer_intent_gate(msg, "");
        assert_eq!(
            d.needs_tools,
            Some(false),
            "background-output delivery marker must override filesystem-path detection, got {:?}",
            d.needs_tools
        );
    }
}
