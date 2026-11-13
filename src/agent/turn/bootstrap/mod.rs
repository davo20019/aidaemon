//! The start of a turn lives in `aidaemon-runtime`
//! ([`aidaemon_runtime::turn::bootstrap`]); the agent runs it as its
//! [`aidaemon_runtime::host::TurnHost`].

/// Task planning and the turn digest live in `aidaemon-runtime`.
#[cfg(test)]
pub(crate) mod task_planning {
    pub(crate) use aidaemon_runtime::turn::task_planning::*;
}

pub(in crate::agent) use aidaemon_runtime::turn::bootstrap::{
    BootstrapCtx, BootstrapData, BootstrapOutcome,
};

/// Bootstrap one turn of `services.agent`.
pub(in crate::agent) async fn run_bootstrap(
    services: &super::services::AgentServices<'_>,
    ctx: &BootstrapCtx<'_>,
) -> anyhow::Result<BootstrapOutcome> {
    aidaemon_runtime::turn::bootstrap::run_bootstrap(services.agent, ctx).await
}

#[cfg(test)]
mod mandate_bootstrap_isolation_tests {
    use super::*;
    use crate::agent::*;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::{Mandate, MandateAuthority, MandateStore, MessageStore, TaskAttempt};

    fn attempt(id: &str, task_id: &str) -> TaskAttempt {
        TaskAttempt {
            id: id.to_string(),
            task_id: task_id.to_string(),
            goal_run_id: "mandate-bootstrap-run".to_string(),
            worker_profile_id: None,
            worker_instance_id: "mandate-bootstrap-worker".to_string(),
            lease_token: format!("lease-{id}"),
            status: "running".to_string(),
            lease_expires_at: (chrono::Utc::now() + chrono::Duration::minutes(3)).to_rfc3339(),
            last_heartbeat_at: chrono::Utc::now().to_rfc3339(),
            workspace_id: None,
            started_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
        }
    }

    async fn run_isolated_turn(
        agent: &Agent,
        session_id: &str,
        user_text: &str,
    ) -> Box<BootstrapData> {
        let services = crate::agent::services::AgentServices::new(agent);
        match run_bootstrap(
            &services,
            &BootstrapCtx {
                session_id,
                user_text,
                attachments: &[],
                status_tx: None,
                user_role: UserRole::Owner,
                channel_ctx: &ChannelContext::internal(),
                ingress_timing: None,
                internal_continuation: false,
                parent_task_id: None,
            },
        )
        .await
        .expect("isolated mandate bootstrap")
        {
            BootstrapOutcome::Continue(data) => data,
            BootstrapOutcome::Return(_) => {
                panic!("generic bootstrap shortcut diverted a mandate worker")
            }
        }
    }

    async fn assert_durable_mandate_origin(agent: &Agent, session_id: &str) {
        let events = agent
            .event_store
            .query_events_by_types(session_id, &[EventType::UserMessage], 10)
            .await
            .expect("query mandate user message");
        let user_message = events
            .first()
            .expect("mandate bootstrap emitted a canonical user message");
        assert_eq!(
            user_message
                .data
                .get("execution_origin")
                .and_then(serde_json::Value::as_str),
            Some("mandate")
        );
    }

    fn assert_isolated(data: &BootstrapData, forbidden: &[&str]) {
        assert!(data.active_untrusted_external_reference_skills.is_empty());
        assert!(!data.restrict_untrusted_external_reference_tools);
        assert!(data.session_summary.is_none());
        assert!(data.project_instruction_tracker.is_none());
        assert!(data.llm_router.is_none());
        assert!(data.turn_context.assessment_recent_messages.is_empty());
        assert!(data.turn_context.primary_project_scope.is_none());
        assert_eq!(
            data.turn_context.goal_user_text,
            "Execute the built-in bounded mandate protocol for this exact worker fence."
        );
        let prompt = format!(
            "{}\n{}\n{}",
            data.core_prompt_bytes,
            data.fresh_task_context_tail,
            data.continuation_task_context_tail
        );
        for sentinel in forbidden {
            assert!(
                !prompt.contains(sentinel),
                "isolated bootstrap leaked {sentinel:?}"
            );
        }
    }

    #[tokio::test]
    async fn mandate_lead_and_executor_bypass_shortcuts_skills_and_history_routing() {
        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("setup test agent");

        let skill_dir = tempfile::tempdir().expect("skill tempdir");
        std::fs::write(
            skill_dir.path().join("private-api-guide.md"),
            "---\nname: private-api-guide\ndescription: PRIVATE SKILL DESCRIPTION\ntriggers: stop, private-api-guide\nsource: docs\n---\nPRIVATE SKILL BODY\n",
        )
        .expect("write custom skill");
        harness.agent.skills_dir = skill_dir.path().to_path_buf();
        harness.agent.skill_cache = crate::skills::SkillCache::new(skill_dir.path().to_path_buf());

        let goal = crate::traits::Goal::new_continuous(
            "Mandate bootstrap controller",
            "owner-mandate-bootstrap",
            Some(10_000),
            Some(50_000),
        );
        let authority = MandateAuthority::default();
        let mandate = Mandate::new(
            &goal.id,
            None,
            "Exercise the bounded protocol",
            &goal.session_id,
            authority.clone(),
            60,
            3_600,
            300,
        );
        harness
            .state
            .create_mandate_controller(&goal, &mandate)
            .await
            .expect("persist mandate controller");

        for session_id in ["mandate-lead-session", "mandate-executor-session"] {
            let prior = Message {
                content: Some(
                    "PRIVATE PRIOR HISTORY in /tmp/private-project with PRIVATE PROJECT SCOPE"
                        .to_string(),
                ),
                ..Message::new_runtime(uuid::Uuid::new_v4().to_string(), session_id, "assistant")
            };
            harness
                .state
                .append_message(&prior)
                .await
                .expect("persist prior history sentinel");
        }

        let root = attempt("root-attempt", "mandate-root-task");
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            authority.clone(),
            &goal.id,
            &root.task_id,
            &root.id,
            &root,
        );
        let lead = run_isolated_turn(&harness.agent, "mandate-lead-session", "stop").await;
        assert_durable_mandate_origin(&harness.agent, "mandate-lead-session").await;
        assert_eq!(lead.user_text, "stop");
        assert_isolated(
            &lead,
            &[
                "PRIVATE SKILL DESCRIPTION",
                "PRIVATE SKILL BODY",
                "PRIVATE PRIOR HISTORY",
                "PRIVATE PROJECT SCOPE",
                "/tmp/private-project",
            ],
        );

        let executor = attempt("executor-attempt", "mandate-executor-task");
        harness.agent.set_test_mandate_execution(
            &mandate.id,
            mandate.version,
            authority,
            &goal.id,
            &root.task_id,
            &root.id,
            &executor,
        );
        let executor_data = run_isolated_turn(
            &harness.agent,
            "mandate-executor-session",
            "$private-api-guide inspect /tmp/private-project",
        )
        .await;
        assert_durable_mandate_origin(&harness.agent, "mandate-executor-session").await;
        assert_isolated(
            &executor_data,
            &[
                "PRIVATE SKILL DESCRIPTION",
                "PRIVATE SKILL BODY",
                "PRIVATE PRIOR HISTORY",
                "PRIVATE PROJECT SCOPE",
            ],
        );
        assert!(executor_data.core_prompt_bytes.contains("role: executor"));
    }
}
