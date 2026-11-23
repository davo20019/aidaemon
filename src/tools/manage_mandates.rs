//! `manage_mandates` lives in `aidaemon_work::tools::manage_mandates`, where
//! its tests are; the daemon gives it the skill library for strategies.

pub use aidaemon_work::tools::manage_mandates::*;

#[cfg(test)]
use crate::startup::tool_config::strategy_skills_dir;

// These stay: they pin a strategy read through the daemon's skill library (`strategy_skills_dir`).
#[cfg(test)]
mod tests {
    /// The dispatcher's facts for the owner in a private chat.
    fn owner_facts() -> crate::traits::ToolCallContext {
        crate::traits::ToolCallContext {
            session_id: "owner-session".into(),
            user_role: Some(crate::types::UserRole::Owner),
            channel_visibility: Some(crate::types::ChannelVisibility::Private),
            ..Default::default()
        }
    }

    /// The owner's facts for a call the agent kernel judged (the owner
    /// confirmed it where `owner_approval_request` asks).
    fn approved_owner_facts() -> crate::traits::ToolCallContext {
        crate::traits::ToolCallContext {
            kernel_approved: Some(crate::traits::ApprovalScope::Call),
            ..owner_facts()
        }
    }

    use super::*;
    use crate::testing::CallWithFacts;
    use crate::testing::{MockProvider, setup_test_agent};
    use crate::traits::*;
    use serde_json::Value;

    /// A strategy skill edited while the owner's card is open is never
    /// pinned: the commit refuses and nothing is stored.
    #[tokio::test]
    async fn a_strategy_edited_after_the_card_is_refused() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let skills_dir = tempfile::TempDir::new().unwrap();
        let mut skill = crate::skills::Skill {
            name: "acme-stewardship".to_string(),
            description: "Careful account strategy".to_string(),
            triggers: vec![],
            body: "Prefer useful replies.".to_string(),
            origin: Some("custom".to_string()),
            source: Some("filesystem".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &skill).unwrap();
        let tool = ManageMandatesTool::new(state.clone())
            .with_strategy_skills(Some(strategy_skills_dir(skills_dir.path().to_path_buf())));
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(30)).to_rfc3339();
        let args = format!(
            r#"{{
                    "action":"create",
                    "objective":"Steward the Acme account",
                    "allow_observations":false,
                    "constraints":["No fabrication"],
                    "success_criteria":["Owner ends stewardship"],
                    "stop_conditions":["Credentials are revoked"],
                    "strategy_skill":"acme-stewardship",
                    "expires_at":"{expires_at}"
                }}"#,
        );
        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        assert!(request.subject.contains("strategy_sha256"));

        skill.body = "Post as often as possible.".to_string();
        crate::skills::write_skill_to_file(skills_dir.path(), &skill).unwrap();

        let error = tool
            .call_with(&args, approved_owner_facts())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("changed after the owner was asked"),
            "{error}"
        );
        assert!(
            state
                .list_mandates(Some("owner-session"), true)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn pinned_strategy_is_content_addressed_and_confirmation_is_complete() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let skills_dir = tempfile::TempDir::new().unwrap();
        let original_skill = crate::skills::Skill {
            name: "x-stewardship".to_string(),
            description: "Thoughtful account strategy".to_string(),
            triggers: vec![],
            body: "Prefer useful replies and verify outcomes.".to_string(),
            origin: Some("custom".to_string()),
            source: Some("filesystem".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &original_skill).unwrap();

        let tool = ManageMandatesTool::new(state.clone())
            .with_strategy_skills(Some(strategy_skills_dir(skills_dir.path().to_path_buf())));
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(30)).to_rfc3339();
        let args = format!(
            r#"{{
                    "action":"create",
                    "objective":"Steward @aidaemon_ai thoughtfully",
                    "allow_observations":false,
                    "constraints":["No fabrication"],
                    "success_criteria":["Owner ends stewardship"],
                    "stop_conditions":["Credentials are revoked"],
                    "strategy_skill":"x-stewardship",
                    "expires_at":"{expires_at}"
                }}"#,
        );

        let crate::traits::OwnerApprovalNeed::Ask(request) = tool
            .owner_approval(
                &serde_json::from_str::<Value>(&args).unwrap(),
                &owner_facts(),
            )
            .await
        else {
            panic!("create asks the owner");
        };
        for expected in [
            "What:",
            "How often:",
            "Can do:",
            "Tells you when:",
            "Ends:",
            "Constraints:",
            "Success criteria:",
            "Stop conditions:",
            "Guidance:",
            "Access summary:",
            "Review depth:",
            "Policy reference:",
        ] {
            assert!(
                request
                    .warnings
                    .iter()
                    .any(|warning| warning.contains(expected)),
                "missing confirmation field {expected}"
            );
        }
        let rendered = request.warnings.join("\n");
        for internal_term in [
            "sha256:",
            "Allowed targets:",
            "Exact operation scopes:",
            "allowed_query_params",
            "mutation_effects",
            "Confirmation binding:",
        ] {
            assert!(
                !rendered.contains(internal_term),
                "confirmation leaked internal term {internal_term}"
            );
        }

        tool.call_with(&args, approved_owner_facts()).await.unwrap();

        // The snapshot is pinned when the confirmed call stores the mandate:
        // a later filesystem edit does not alter it.
        let changed_skill = crate::skills::Skill {
            body: "A later filesystem edit must not alter the pinned snapshot.".to_string(),
            ..original_skill
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &changed_skill).unwrap();

        let mandate = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap()
            .pop()
            .unwrap();
        let strategy = mandate.strategy.expect("pinned strategy");
        assert_eq!(strategy.skill_name, "x-stewardship");
        assert!(strategy.body.contains("Prefer useful replies"));
        assert!(!strategy.body.contains("later filesystem edit"));
        assert_eq!(strategy.content_sha256.len(), 64);
    }
}
