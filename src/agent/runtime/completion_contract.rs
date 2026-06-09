//! Completion-contract inference: deriving what a turn must observe/mutate/verify.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. Owns the completion types ([`CompletionContract`],
//! [`CompletionProgress`], [`CompletionTaskKind`], [`VerificationTarget`],
//! [`VerificationTargetKind`], `CompletionSignals`) and the `infer_*` /
//! `extract_verification_*` helpers.

use super::followup::text_contains_any_phrase;
use super::project_scope::{
    normalize_project_scope_path_with_aliases, push_project_scope, token_looks_like_filesystem_path,
};
use once_cell::sync::Lazy;
use regex::Regex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum CompletionTaskKind {
    #[default]
    Conversational,
    Answer,
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
    /// Count of external mutation attempts that failed (4xx/5xx, etc.)
    pub failed_external_mutation_count: usize,
    /// Count of external mutation attempts that succeeded
    pub successful_external_mutation_count: usize,
    /// True after we have already given the LLM one reconciliation-informed retry.
    pub external_mutation_reconciliation_attempted: bool,
    /// Cumulative count of times the verification guard blocked completion.
    /// Unlike `stall_count` (which is shared with other stall mechanisms and
    /// gets reset), this counter only increments and is used as a safety valve
    /// to prevent infinite verification loops.
    pub verification_block_count: usize,
    /// Count of times the response-quality nudge has been injected.
    /// Used to prevent infinite nudge loops — only fire once.
    pub quality_nudge_count: usize,
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

    pub(super) fn mark_failed_external_mutation(&mut self) {
        self.failed_external_mutation_count += 1;
    }

    pub(super) fn mark_successful_external_mutation(&mut self) {
        self.successful_external_mutation_count += 1;
    }

    pub(super) fn mark_external_mutation_reconciliation_attempted(&mut self) {
        self.external_mutation_reconciliation_attempted = true;
    }

    pub(super) fn clear_failed_external_mutation_gate(&mut self) {
        self.failed_external_mutation_count = 0;
        self.external_mutation_reconciliation_attempted = false;
    }
}

pub(super) static HTTP_URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\bhttps?://[^\s"'()<>]+"#).expect("valid http url regex"));

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
    /// A question whose answer requires observing live external/system state
    /// (e.g. "how many users?", "who are the admins?", "any blocked users?").
    /// These have no file/URL target and no "show me / on this site" phrasing,
    /// so they are NOT `observable_target_request`, yet they still need a tool
    /// to observe. Without this signal the contract treats them as plain-text
    /// knowledge questions and the prelude hard-blocks the lookup tool.
    live_state_query: bool,
}

pub(super) fn looks_like_question_request(lower_text: &str) -> bool {
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
            "change",
            "update",
            "edit",
            "write",
            "rewrite",
            "overwrite",
            "modify",
            "replace",
            "create",
            "delete",
            "remove",
            "deploy",
            "build",
            "connect",
            "set up",
            "setup",
            "install",
            "restart",
            "reload",
            "enable",
            "disable",
            "remember",
            "store",
            "save",
            "note",
            "pull",
            "push",
            "run",
            "execute",
            "fetch",
            "merge",
            "start",
            "stop",
            "compile",
            "download",
            "clone",
            "migrate",
            "fix",
            "retry",
            "redo",
            "rerun",
            "try again",
            "do it again",
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
                "fixing",
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

    // If the message is a question AND the only change-triggering keywords are
    // memory verbs ("remember", "store", "save", "note"), demote asks_change
    // to false. "What did I ask you to remember?" is recall, not mutation.
    let asks_change = if asks_change && is_question {
        // If only memory verbs triggered asks_change, it's a recall question.
        // Check if any NON-memory change keyword is present.
        text_contains_any_phrase(
            lower_text,
            &[
                "change",
                "update",
                "edit",
                "write",
                "rewrite",
                "overwrite",
                "modify",
                "replace",
                "create",
                "delete",
                "remove",
                "deploy",
                "build",
                "connect",
                "set up",
                "setup",
                "install",
                "restart",
                "reload",
                "enable",
                "disable",
                "pull",
                "push",
                "run",
                "execute",
                "fetch",
                "merge",
                "start",
                "stop",
                "compile",
                "download",
                "clone",
                "migrate",
                "fix",
                "retry",
                "redo",
                "rerun",
                "try again",
                "do it again",
            ],
        )
    } else {
        asks_change
    };

    // Interrogatives that enumerate or quantify live system/entity state. These
    // need a tool to observe the answer even though they carry no file/URL
    // target. Scoped to questions so plain conversational text is unaffected.
    let live_state_query = is_question
        && (lower_text.starts_with("any ")
            || text_contains_any_phrase(
                lower_text,
                &[
                    "how many",
                    "how much",
                    "who are",
                    "who is",
                    "who's",
                    "list the",
                    "list all",
                    "what are the",
                    "which ",
                    "are there any",
                    "is there any",
                    "do i have any",
                    "do we have any",
                ],
            ));

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
        live_state_query,
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

pub(super) fn infer_completion_contract(text: &str, alias_roots: &[String]) -> CompletionContract {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return CompletionContract::default();
    }

    let verification_targets = extract_verification_targets(text, alias_roots);
    let signals = infer_completion_signals(&lower, &verification_targets);
    let connected_content_mode = super::intent_routing::classify_connected_content_mode(text);
    let task_kind = infer_completion_task_kind(&signals);
    // DraftOnly used to override task_kind to Compose and force
    // expects_mutation=false, but this caused false tool disablement
    // for requests like "create blog posts in ~/projects/X and commit".
    // Keyword-based content-mode classification is too brittle to make
    // hard execution decisions.  DraftOnly is now advisory only — it
    // still influences system prompt hints and budget routing but does
    // not override the signal-derived task kind or mutation expectation.

    let expects_mutation = if connected_content_mode.expects_live_delivery() {
        true
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
        || signals.live_state_query
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn live_state_questions_require_observation_so_lookup_tools_are_not_blocked() {
        // Regression: factual questions about live system state were classified
        // as plain-text knowledge questions (requires_observation=false), so the
        // prelude hard-blocked the shell/lookup tool and the model fabricated an
        // answer or returned a bare "Done.".
        for q in [
            "How many users?",
            "Who are admin users?",
            "Any blocked/inactive users?",
            "Which modules are enabled?",
            "How much disk is free?",
        ] {
            let contract = infer_completion_contract(q, &[]);
            assert!(
                contract.requires_observation,
                "expected requires_observation for live-state question: {q:?}"
            );
            assert!(
                !contract.expects_mutation,
                "live-state question should not expect mutation: {q:?}"
            );
        }
        // "How many users?" carries no file/URL target and requests no explicit
        // verification, so requires_observation alone does NOT arm a completion
        // verification block (has_concrete_verification_reason stays false) —
        // confirming requires_observation flips off the plain-text gate without
        // introducing loop risk for a plain answer.
        let bare = infer_completion_contract("How many users?", &[]);
        assert!(!bare.explicit_verification_requested);
        assert!(bare.verification_targets.is_empty());
    }
    #[test]
    fn conversational_questions_stay_plain_text() {
        // A question that does not enumerate live state must not be flagged.
        let contract = infer_completion_contract("What do you think of this idea?", &[]);
        assert!(!contract.requires_observation);
        assert!(!contract.expects_mutation);
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
    fn rewrite_request_expects_mutation() {
        let contract = infer_completion_contract(
            "Rewrite ~/projects/blog/tweets.md so the thread better promotes the blog content.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
    }
    #[test]
    fn connected_content_draft_only_request_preserves_signal_derived_task_kind() {
        // DraftOnly is advisory only — it no longer overrides task_kind to
        // Compose or forces expects_mutation=false.  The signal-derived
        // classification is preserved so that tool blocking decisions are
        // based on actual intent signals, not keyword-based content mode.
        let contract = infer_completion_contract("Help me write a tweet about our launch.", &[]);
        assert_eq!(
            contract.connected_content_mode,
            super::super::intent_routing::ConnectedContentMode::DraftOnly
        );
        // task_kind comes from signals, not content mode override — DraftOnly
        // no longer forces Compose.
        assert!(
            contract.expects_mutation
                || contract.task_kind == CompletionTaskKind::Conversational
                || contract.task_kind == CompletionTaskKind::Answer
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
    fn recall_question_about_remembered_facts_does_not_expect_mutation() {
        // "What did I ask you to remember?" contains "remember" but is a
        // recall question, not a mutation request. Should NOT expect mutation.
        let contract = infer_completion_contract(
            "What do you know about my coding preferences? What did I ask you to remember?",
            &[],
        );
        assert!(
            !contract.expects_mutation,
            "Recall question about 'remember' should not expect mutation, got task_kind={:?}",
            contract.task_kind
        );
    }
    #[test]
    fn store_request_with_remember_does_expect_mutation() {
        // "Remember that I prefer dark themes" is a store request, not recall.
        let contract =
            infer_completion_contract("Remember that I prefer dark themes and large fonts", &[]);
        assert!(
            contract.expects_mutation,
            "Store request with 'remember' should expect mutation"
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
    #[test]
    fn failed_external_mutation_tracking() {
        let contract = CompletionContract {
            task_kind: CompletionTaskKind::Deliver,
            expects_mutation: true,
            requires_observation: false,
            ..Default::default()
        };
        let mut progress = CompletionProgress::new(&contract);
        assert_eq!(progress.failed_external_mutation_count, 0);
        assert!(!progress.external_mutation_reconciliation_attempted);

        progress.mark_failed_external_mutation();
        assert_eq!(progress.failed_external_mutation_count, 1);
        assert!(!progress.external_mutation_reconciliation_attempted);

        progress.mark_successful_external_mutation();
        assert_eq!(progress.successful_external_mutation_count, 1);

        progress.mark_external_mutation_reconciliation_attempted();
        assert!(progress.external_mutation_reconciliation_attempted);

        progress.clear_failed_external_mutation_gate();
        assert_eq!(progress.failed_external_mutation_count, 0);
        assert!(!progress.external_mutation_reconciliation_attempted);
    }

    // Regression tests from the 2026-06-06 attribution run: the mutation
    // expectation misfired in BOTH directions across builds. Read-only
    // phrasings were blocked with expects_mutation=true on the older build,
    // while the turn-10 delete request must keep expects_mutation=true so
    // the zero-tool fabrication guard can fire.
    #[test]
    fn read_only_file_inspection_does_not_expect_mutation() {
        let contract = infer_completion_contract(
            "Read each of the three files back and tell me which one is longest.",
            &[],
        );
        assert!(
            !contract.expects_mutation,
            "read-only inspection must not expect a mutation (got {:?})",
            contract.task_kind
        );

        let contract = infer_completion_contract(
            "Open all four files and tell me which two are most similar in length.",
            &[],
        );
        assert!(!contract.expects_mutation);
    }

    #[test]
    fn delete_and_create_requests_expect_mutation() {
        // Turn-10 fabrication case: "delete" must keep the contract armed.
        let contract = infer_completion_contract("Delete the cachetest2 folder entirely.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);

        let contract = infer_completion_contract(
            "Make a folder ~/tmp/cachetest2 and create four files in it.",
            &[],
        );
        assert!(contract.expects_mutation);

        let contract =
            infer_completion_contract("Remove west.txt and execute tally.py once more.", &[]);
        assert!(contract.expects_mutation);
    }
}
