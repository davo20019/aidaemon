//! Skills live in `aidaemon_skills`. The daemon keeps its old paths to them
//! and the model it gives skill promotion.

pub use aidaemon_skills::*;

/// Skill promotion, with the daemon's side-call gateway as its model.
pub mod promotion {
    use std::path::PathBuf;
    use std::sync::Arc;

    use serde_json::Value;

    pub use aidaemon_skills::promotion::*;

    use crate::llm_runtime::gateway::SideCallGateway;
    use crate::llm_runtime::side_calls::SideCall;
    use crate::traits::StateStore;

    /// Skill drafts are written through the `skill_promotion` side call.
    struct SideCallPromotion(SideCallGateway);

    #[async_trait::async_trait]
    impl PromotionModel for SideCallPromotion {
        async fn write_skill(&self, messages: &[Value]) -> anyhow::Result<String> {
            // Background work does not need the strongest model: the spec runs
            // on the cheap model.
            Ok(self
                .0
                .ask(
                    SideCall::SkillPromotion,
                    messages,
                    &Default::default(),
                    None,
                )
                .await?
                .text)
        }
    }

    /// The daemon's promoter: drafts in its state store, written through
    /// its side-call gateway.
    pub fn promoter(
        state: Arc<dyn StateStore>,
        side_calls: SideCallGateway,
        skills_dir: PathBuf,
        evidence_gate_enforce: bool,
    ) -> SkillPromoter {
        SkillPromoter::new(
            state,
            Arc::new(SideCallPromotion(side_calls)),
            skills_dir,
            evidence_gate_enforce,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::llm_runtime::gateway::test_support::{
            failed_calls, gateway_with, ok_calls, provider_answering, provider_failing,
        };
        use crate::traits::store_prelude::*;
        use crate::traits::Procedure;

        const VALUABLE_SKILL: &str = "---\nname: deploy-rust-service\n\
            description: Build, validate, and deploy a Rust service safely.\n\
            triggers: deploy service, release rust app\n---\n\
            1. Run the formatter, the linter with warnings denied, and the full test suite.\n\
            2. Build the release artifact and verify the configuration values for Acme Corp.\n\
            3. Deploy the artifact, restart the service, and check its health before reporting.";

        /// A promoter over a fresh store holding one procedure used `successes`
        /// times without a failure, and its own temp skills directory.
        async fn promoter_with(
            gateway: SideCallGateway,
            successes: i32,
        ) -> (SkillPromoter, Arc<dyn StateStore>, tempfile::TempDir) {
            let file = tempfile::NamedTempFile::new().unwrap();
            let path = file.path().to_str().unwrap().to_string();
            // The store keeps the file open; the file outlives the test.
            std::mem::forget(file);
            let embeddings = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
            let store = crate::state::SqliteStateStore::new(&path, 100, None, embeddings)
                .await
                .unwrap();
            store
                .upsert_procedure(&Procedure {
                    id: 0,
                    name: "deploy rust service".to_string(),
                    trigger_pattern: "deploy the service for Acme Corp".to_string(),
                    steps: vec![
                        "run the formatter, the linter and the tests".to_string(),
                        "build the release artifact and restart the service".to_string(),
                    ],
                    success_count: successes,
                    failure_count: 0,
                    avg_duration_secs: None,
                    last_used_at: None,
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                })
                .await
                .unwrap();
            let state: Arc<dyn StateStore> = Arc::new(store);
            let skills_dir = tempfile::tempdir().unwrap();
            let promoter = promoter(
                state.clone(),
                gateway,
                skills_dir.path().to_path_buf(),
                false,
            );
            (promoter, state, skills_dir)
        }

        #[tokio::test]
        async fn a_promotion_call_creates_a_draft_and_is_recorded() {
            let (gateway, events) = gateway_with(provider_answering(VALUABLE_SKILL)).await;
            let (promoter, state, _skills_dir) = promoter_with(gateway, 5).await;

            assert_eq!(promoter.run_promotion_cycle().await.unwrap(), 1);
            assert_eq!(state.get_pending_skill_drafts().await.unwrap().len(), 1);
            assert_eq!(ok_calls(&events, "skill_promotion").await, 1);
        }

        #[tokio::test]
        async fn a_failed_promotion_call_creates_no_draft_and_is_recorded() {
            let (gateway, events) = gateway_with(provider_failing("503")).await;
            let (promoter, state, _skills_dir) = promoter_with(gateway, 5).await;

            assert_eq!(promoter.run_promotion_cycle().await.unwrap(), 0);
            assert_eq!(state.get_pending_skill_drafts().await.unwrap().len(), 0);
            assert!(failed_calls(&events, "skill_promotion").await >= 1);
            assert_eq!(ok_calls(&events, "skill_promotion").await, 0);
        }
    }
}
