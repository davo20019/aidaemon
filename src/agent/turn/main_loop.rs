//! The turn's setup lives in `aidaemon-runtime`
//! ([`aidaemon_runtime::engine::turn_engine::main_loop`]); its tests stay
//! here, on the daemon's agent.

#[cfg(test)]
use super::*;
#[cfg(test)]
use aidaemon_runtime::engine::turn_engine::main_loop::*;

#[cfg(test)]
#[path = "characterization_tests.rs"]
mod characterization_tests;

#[cfg(test)]
mod offered_tool_tests {
    use super::*;

    fn names(defs: &[Value]) -> Vec<&str> {
        defs.iter()
            .filter_map(|d| d["function"]["name"].as_str())
            .collect()
    }

    /// The offered list follows the resolved authority's tool set: every
    /// tool when it names none, only the named ones otherwise, none for an
    /// empty set.
    #[test]
    fn only_an_owners_unbound_request_has_a_guessed_scope() {
        use crate::types::UserRole;
        assert!(scopes_are_only_a_guess(
            UserRole::Owner,
            false,
            false,
            false,
            false
        ));
        assert!(!scopes_are_only_a_guess(
            UserRole::Guest,
            false,
            false,
            false,
            false
        ));
        assert!(!scopes_are_only_a_guess(
            UserRole::Owner,
            true,
            false,
            false,
            false
        ));
        assert!(!scopes_are_only_a_guess(
            UserRole::Owner,
            false,
            true,
            false,
            false
        ));
        assert!(!scopes_are_only_a_guess(
            UserRole::Owner,
            false,
            false,
            true,
            false
        ));
        assert!(!scopes_are_only_a_guess(
            UserRole::Owner,
            false,
            false,
            false,
            true
        ));
    }

    #[test]
    fn offered_tools_follow_the_resolved_authority_tool_set() {
        let defs: Vec<Value> = ["http_request", "read_file", "write_file"]
            .iter()
            .map(|name| json!({"function": {"name": name}}))
            .collect();

        let mut all = defs.clone();
        retain_authorized_tool_definitions(&mut all, &aidaemon_core::Authority::read_only());
        assert_eq!(names(&all), ["http_request", "read_file", "write_file"]);

        let mut some = defs.clone();
        let authority = aidaemon_core::Authority {
            level: aidaemon_core::AccessLevel::Write,
            tools: Some(["read_file".to_string()].into_iter().collect()),
        };
        retain_authorized_tool_definitions(&mut some, &authority);
        assert_eq!(names(&some), ["read_file"]);

        let mut none = defs;
        retain_authorized_tool_definitions(&mut none, &aidaemon_core::Authority::no_tools());
        assert!(none.is_empty());
    }
}

#[cfg(test)]
mod stuck_fallback_tests {
    use super::*;

    #[test]
    fn exact_antecedent_is_projected_as_bounded_typed_context_without_word_matching() {
        let recent = vec![
            json!({
                "message_id": "unrelated-user",
                "role": "user",
                "content": "Unrelated setup"
            }),
            json!({
                "message_id": "antecedent-user",
                "role": "user",
                "content": "Synthetic volatile value"
            }),
            json!({
                "message_id": "antecedent-assistant",
                "role": "assistant",
                "content": "Acknowledged"
            }),
            json!({
                "message_id": "current-user",
                "role": "user",
                "content": "Use the prior value"
            }),
        ];

        let projection = project_required_antecedent_context(&recent, "antecedent-user")
            .expect("typed relationship projection");

        assert_eq!(
            projection.source_message_ids,
            ["antecedent-user", "antecedent-assistant"]
        );
        assert!(projection.rendered.contains("Synthetic volatile value"));
        assert!(projection.rendered.contains("Acknowledged"));
        assert!(!projection.rendered.contains("Unrelated setup"));
        assert!(!projection.rendered.contains("Use the prior value"));
        assert!(!projection.truncated);
    }

    #[test]
    fn required_context_projection_reports_truncation() {
        let recent = vec![json!({
            "message_id": "antecedent-user",
            "role": "user",
            "content": "x".repeat(8_000)
        })];
        let projection =
            project_required_antecedent_context(&recent, "antecedent-user").expect("projection");
        assert!(projection.truncated);
        assert!(projection.rendered.contains("\"truncated\":true"));
        assert!(projection.rendered.chars().count() < 5_000);
    }

    #[test]
    fn missing_exact_antecedent_cannot_close_conversation_obligation() {
        let contract = CompletionContract {
            requires_observation: true,
            evidence_requirements: vec![crate::traits::RequestEvidenceRequirement {
                summary: "Exact preceding setup value".to_string(),
                acceptable_scopes: vec![crate::traits::ToolSemanticScope::ConversationHistory],
                purpose: crate::traits::EvidencePurpose::Content,
                minimum_authority: crate::traits::EvidenceAuthority::Canonical,
                temporal_scope: crate::traits::EvidenceTemporalScope::Historical,
                required_content_markers: Vec::new(),
                receipt: None,
                target: None,
            }],
            ..CompletionContract::default()
        };
        let recent = vec![
            json!({
                "message_id": "unrelated-user",
                "role": "user",
                "content": "VIOLET HARBOR"
            }),
            json!({
                "message_id": "different-user",
                "role": "user",
                "content": "This is a different setup."
            }),
        ];
        let mut progress = CompletionProgress::new(&contract, "task-child");

        assert!(
            assimilate_visible_antecedent_context(
                &contract,
                &mut progress,
                &recent,
                "antecedent-user",
            )
            .is_none()
        );
        assert!(progress.verification_pending);
    }

    #[test]
    fn provisional_new_task_relationship_can_name_but_not_adopt_an_antecedent() {
        let recent = vec![json!({
            "message_id": "synthetic-prior-user",
            "role": "assistant",
            "content": "The previous device check requires live health evidence."
        })];
        let context = relationship_conversation_context(
            Some(FollowupMode::NewTask),
            Some("Prior device discussion"),
            &recent,
        )
        .expect("bounded relationship context");
        assert!(context.contains("message_id=synthetic-prior-user"));
        assert!(context.contains("Prior device discussion"));
    }

    #[test]
    fn followup_relationship_context_retains_the_preceding_exchange() {
        let recent = vec![json!({
            "message_id": "synthetic-user-1",
            "role": "user",
            "content": "Check the first device."
        })];
        let context =
            relationship_conversation_context(Some(FollowupMode::Followup), None, &recent)
                .expect("follow-up context");
        assert!(context.contains("Check the first device."));
    }
}
