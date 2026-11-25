//! Follow-up / task-switch classification for incoming user turns.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. Owns [`FollowupMode`], [`TurnContextReason`], and the
//! `looks_like_*` / `classify_followup_mode` family.

use super::project_scope::extract_project_scopes_from_text;
use aidaemon_events::traits::Message;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FollowupMode {
    NewTask,
    Followup,
    ClarificationAnswer,
}

impl FollowupMode {
    pub fn as_str(self) -> &'static str {
        match self {
            FollowupMode::NewTask => "new_task",
            FollowupMode::Followup => "followup",
            FollowupMode::ClarificationAnswer => "clarification_answer",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnContextReason {
    DefaultNewTask,
    InternalContinuation,
    #[cfg(test)]
    ExplicitFollowup,
    ClarificationAnswer,
    ContextDependentQuestion,
    #[cfg(test)]
    FollowupOverrideStandalone,
    FollowupOverrideMismatchPreflight,
    CarryoverSanitized,
}

impl TurnContextReason {
    pub fn as_code(&self) -> &'static str {
        match self {
            TurnContextReason::DefaultNewTask => "default_new_task",
            TurnContextReason::InternalContinuation => "internal_continuation",
            #[cfg(test)]
            TurnContextReason::ExplicitFollowup => "explicit_followup",
            TurnContextReason::ClarificationAnswer => "clarification_answer",
            TurnContextReason::ContextDependentQuestion => "context_dependent_question",
            #[cfg(test)]
            TurnContextReason::FollowupOverrideStandalone => "followup_override_standalone",
            TurnContextReason::FollowupOverrideMismatchPreflight => {
                "followup_override_mismatch_preflight"
            }
            TurnContextReason::CarryoverSanitized => "carryover_sanitized",
        }
    }
}

pub fn find_previous_turns(
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

pub fn sanitize_carryover_blocks(input: &str) -> (String, bool) {
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

pub fn has_project_scope_divergence_with_aliases(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    prev_user_text: &str,
    current: &str,
    alias_roots: &[String],
) -> bool {
    let mut prev_scopes = Vec::new();
    let mut current_scopes = Vec::new();
    extract_project_scopes_from_text(backend, prev_user_text, &mut prev_scopes, 6, alias_roots);
    extract_project_scopes_from_text(backend, current, &mut current_scopes, 6, alias_roots);
    if !prev_scopes.is_empty() && !current_scopes.is_empty() {
        return !current_scopes
            .iter()
            .any(|scope| prev_scopes.iter().any(|prev| prev == scope));
    }

    // Plain-language project names are not identities. If either turn lacks
    // an exact structural scope, semantic request assessment decides whether
    // it is a continuation; a nickname mismatch cannot override that state.
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_project_scope_divergence_with_aliases(
        prev_user_text: &str,
        current: &str,
        alias_roots: &[String],
    ) -> bool {
        super::has_project_scope_divergence_with_aliases(
            &*aidaemon_tools::testing::active_execution_backend(),
            prev_user_text,
            current,
            alias_roots,
        )
    }
    use proptest::prelude::*;

    #[test]
    fn previous_turn_resolution_uses_canonical_message_topology() {
        fn message(role: &str, content: &str) -> Message {
            Message {
                content: Some(content.to_string()),
                ..Message::new_runtime(uuid::Uuid::new_v4().to_string(), "session", role)
            }
        }

        let history = vec![
            message("user", "Send me the investor PDF."),
            message("assistant", "The conversion failed."),
            message("user", "Retry"),
            message("assistant", "That converter also failed."),
            message("user", "Can you try one more time?"),
        ];
        let (_, previous_user) = find_previous_turns(&history, "Can you try one more time?");
        assert_eq!(previous_user.as_deref(), Some("Retry"));
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
            &format!("Deploy {}", dogs.display()),
            &format!("Now deploy {}", blog.display()),
            &alias_roots,
        ));
    }

    #[test]
    fn plain_project_names_cannot_override_semantic_continuation() {
        assert!(!has_project_scope_divergence_with_aliases(
            "Keep working on synthetic-dogs-project",
            "Use synthetic-blog-project for that part",
            &[],
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
}
