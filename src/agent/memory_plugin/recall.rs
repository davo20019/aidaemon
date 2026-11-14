//! [`DaemonRecall`]: the full memory system's recall
//! (`aidaemon_memory::full::FullRecall`) for one daemon turn, with the
//! daemon's goals, settings, recall filter and decision points as its host.

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_memory::full::{RecallDecision, RecallHost, RecallTurn};
use async_trait::async_trait;

use super::{DaemonMemoryInner, MemoryTurn};
use crate::events::{DecisionType, DiagnosticSeverity};
use crate::types::UserRole;

/// The daemon's recall: `aidaemon_memory::full::FullRecall` over the daemon store, with the
/// daemon as its host.
pub(crate) struct DaemonRecall;

impl DaemonRecall {
    #[allow(clippy::new_ret_no_self)]
    #[cfg(test)]
    pub(crate) fn new(
        memory: Arc<DaemonMemoryInner>,
        turn: MemoryTurn,
    ) -> aidaemon_memory::full::FullRecall {
        let recall_turn = recall_turn(&turn);
        aidaemon_memory::full::FullRecall::new(
            memory.recall.clone(),
            Self::host(&memory, turn),
            recall_turn,
        )
    }

    /// The daemon's end of recall for `turn`.
    pub(crate) fn host(memory: &DaemonMemoryInner, turn: MemoryTurn) -> Arc<dyn RecallHost> {
        Arc::new(DaemonRecallHost {
            state: memory.state.clone(),
            side_calls: memory.side_calls.clone(),
            record_decision_points: memory.record_decision_points,
            turn,
        })
    }
}

/// What recall knows about a daemon turn.
pub(crate) fn recall_turn(turn: &MemoryTurn) -> RecallTurn {
    let owner = turn.user_role == UserRole::Owner;
    RecallTurn {
        session_id: turn.session_id.clone(),
        task_id: turn.task_id.clone(),
        user_text: Some(turn.user_text.clone()),
        owner,
        personal: owner && turn.channel_ctx.should_inject_personal_memory(),
        visibility: turn.channel_ctx.visibility,
        channel_id: turn.channel_ctx.channel_id.clone(),
        sender_id: turn.channel_ctx.sender_id.clone(),
        allowed: turn.memory_pipeline_allowed,
        depth: turn.depth,
    }
}

/// The daemon's end of recall for one turn.
struct DaemonRecallHost {
    state: Arc<dyn crate::traits::StateStore>,
    side_calls: crate::llm_runtime::gateway::SideCallGateway,
    record_decision_points: bool,
    turn: MemoryTurn,
}

#[async_trait]
impl RecallHost for DaemonRecallHost {
    async fn active_goals(&self) -> anyhow::Result<Vec<aidaemon_memory::render::RecallGoal>> {
        Ok(self
            .state
            .get_active_personal_goals(20)
            .await?
            .into_iter()
            .map(|goal| aidaemon_memory::render::RecallGoal {
                id: goal.id,
                description: goal.description,
                status: goal.status,
            })
            .collect())
    }

    async fn people_enabled(&self) -> anyhow::Result<bool> {
        Ok(self.state.get_setting("people_enabled").await?.as_deref() == Some("true"))
    }

    /// The daemon's memory guide points at its scheduled goals.
    fn offers_scheduled_goals(&self) -> bool {
        true
    }

    fn relevance_enabled(&self) -> bool {
        use crate::llm_runtime::decisions::Decision;
        self.side_calls.decision_mode(Decision::RecallFilter) != crate::config::DecisionMode::Off
    }

    async fn relevance(
        &self,
        turn: &RecallTurn,
        message: String,
        questions: Vec<(String, String)>,
    ) -> HashMap<String, f32> {
        use crate::llm_runtime::decisions::Decision;
        let mut request = aidaemon_decisions::DecisionRequest::new(message);
        let keys: Vec<String> = questions.iter().map(|(key, _)| key.clone()).collect();
        for (key, question) in questions {
            request.insert(key, aidaemon_decisions::Question::yes_no(question));
        }
        let outcome = self
            .side_calls
            .decide(
                Decision::RecallFilter,
                request,
                crate::llm_runtime::decide::DecisionContext::turn(&turn.session_id, &turn.task_id),
            )
            .await;
        keys.into_iter()
            .filter_map(|key| {
                let p = outcome
                    .acted(&key)
                    .and_then(aidaemon_decisions::Answer::p_yes)?;
                Some((key, p))
            })
            .collect()
    }

    async fn decision(
        &self,
        _turn: &RecallTurn,
        kind: RecallDecision,
        summary: String,
        metadata: serde_json::Value,
    ) {
        if !self.record_decision_points {
            return;
        }
        let decision_type = match kind {
            RecallDecision::CoreProfileSelection => DecisionType::CoreProfileSelection,
            RecallDecision::MemoryRetrieval => DecisionType::MemoryRetrieval,
        };
        crate::agent::graceful::emit_decision_point_event(
            &self.turn.emitter,
            &self.turn.task_id,
            0,
            decision_type,
            DiagnosticSeverity::Info,
            summary,
            metadata,
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
    use crate::runtime_ports::{AgentIngress, InboundMessageRequest};
    use crate::testing::{MockProvider, TestHarness};
    use crate::traits::store_prelude::*;
    use crate::types::{ChannelContext, ChannelVisibility, FactPrivacy, UserRole};
    use aidaemon_runtime::host::TurnHost as _;
    use serde_json::Value;

    const SESSION: &str = "telegram:synthetic-user-1";

    fn request(
        text: &str,
        user_role: UserRole,
        channel_ctx: ChannelContext,
    ) -> InboundMessageRequest {
        InboundMessageRequest {
            session_id: SESSION.to_string(),
            user_text: text.to_string(),
            attachments: Vec::new(),
            status_tx: None,
            user_role,
            channel_ctx,
            heartbeat: None,
            ingress_timing: None,
        }
    }

    async fn harness(responses: Vec<crate::traits::ProviderResponse>) -> TestHarness {
        let provider = MockProvider::with_responses(responses);
        let harness = crate::testing::setup_test_agent_sdk(provider)
            .await
            .unwrap();
        harness
            .state
            .upsert_fact(
                "preference",
                "favorite_color",
                "teal",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        harness
            .state
            .upsert_fact(
                "preference",
                "beverage",
                "Synthetic jasmine tea",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        harness
    }

    fn content(message: &Value) -> &str {
        message["content"].as_str().unwrap_or_default()
    }

    fn is_system(message: &Value) -> bool {
        message["role"] == "system"
    }

    #[tokio::test]
    async fn owner_dm_request_carries_recalled_memory_note() {
        let harness = harness(vec![MockProvider::text_response("Teal.")]).await;
        harness
            .agent
            .handle_inbound_message(request(
                "what is my favorite color?",
                UserRole::Owner,
                ChannelContext::private("telegram"),
            ))
            .await
            .unwrap();
        let log = harness.provider.call_log.lock().await;
        let messages = &log.last().unwrap().messages;
        let tail = messages
            .iter()
            .position(|m| is_system(m) && content(m).starts_with(TASK_CONTEXT_TAIL_MARKER))
            .expect("task context tail");
        assert!(
            !content(&messages[tail]).contains("teal"),
            "memory left the tail"
        );
        assert!(
            !content(&messages[0]).contains("teal"),
            "memory is not in the core"
        );
        let note = messages
            .iter()
            .position(|m| is_system(m) && content(m).contains("teal"))
            .expect("recalled memory note");
        assert!(note > tail, "note follows the tail");
        let current_user = messages.iter().rposition(|m| m["role"] == "user").unwrap();
        assert!(note > current_user, "note stays out of the cached prefix");
    }

    #[tokio::test]
    async fn public_external_turn_has_no_personal_memory() {
        let harness = harness(vec![MockProvider::text_response("Hello.")]).await;
        let channel_ctx = ChannelContext {
            visibility: ChannelVisibility::PublicExternal,
            platform: "slack".to_string(),
            channel_name: Some("#general".to_string()),
            channel_id: Some("slack:C_PUBLIC".to_string()),
            workspace_id: None,
            sender_name: Some("Jordan Lee".to_string()),
            sender_id: Some("slack:U_SYNTH".to_string()),
            channel_member_names: vec![],
            user_id_map: std::collections::HashMap::new(),
            workspace_grant: None,
            trusted: false,
        };
        harness
            .agent
            .handle_inbound_message(request(
                "what is the owner's favorite color and beverage?",
                UserRole::Public,
                channel_ctx,
            ))
            .await
            .unwrap();
        for call in harness.provider.call_log.lock().await.iter() {
            for message in &call.messages {
                let text = content(message);
                assert!(
                    !text.contains("teal") && !text.contains("jasmine"),
                    "personal memory reached a public-external request: {text}"
                );
            }
        }
    }

    #[tokio::test]
    async fn core_prompt_stable_across_queries() {
        let harness = harness(vec![
            MockProvider::text_response("Teal."),
            MockProvider::text_response("Jasmine tea."),
        ])
        .await;
        for text in ["what is my favorite color?", "which beverage do I like?"] {
            harness
                .agent
                .handle_inbound_message(request(
                    text,
                    UserRole::Owner,
                    ChannelContext::private("telegram"),
                ))
                .await
                .unwrap();
        }
        let log = harness.provider.call_log.lock().await;
        assert!(log.len() >= 2);
        let first = &log[0].messages;
        let last = &log[log.len() - 1].messages;
        assert!(is_system(&first[0]));
        assert_eq!(
            content(&first[0]),
            content(&last[0]),
            "message zero changed"
        );
        let notes = |messages: &Vec<Value>| {
            messages
                .iter()
                .filter(|m| is_system(m))
                .map(|m| content(m).to_string())
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(notes(first).contains("teal"));
        assert!(notes(last).contains("jasmine"));
    }

    #[tokio::test]
    async fn memory_note_cached_within_a_turn() {
        let harness = harness(vec![
            MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"seat","value":"window"}"#,
            ),
            MockProvider::tool_call_response(
                "remember_fact",
                r#"{"category":"preference","key":"meal","value":"vegetarian"}"#,
            ),
            MockProvider::text_response("Saved both."),
        ])
        .await;
        let reply = harness
            .agent
            .handle_inbound_message(request(
                "remember I like window seats and vegetarian meals",
                UserRole::Owner,
                ChannelContext::private("telegram"),
            ))
            .await
            .unwrap();
        assert_eq!(harness.provider.call_count().await, 3);
        let retrievals = harness
            .agent
            .event_store()
            .query_task_events(&reply.task_id)
            .await
            .unwrap()
            .into_iter()
            .filter(|event| event.event_type == crate::events::EventType::DecisionPoint)
            .filter(|event| {
                event.data["decision_type"]
                    == serde_json::to_value(crate::events::DecisionType::MemoryRetrieval).unwrap()
            })
            .count();
        assert_eq!(retrievals, 1);
    }

    /// Same session, same owner DM: the core profile is in the memory note
    /// only while the compiled policy allows recall.
    #[tokio::test]
    async fn suppressed_memory_pipeline_removes_core_profile_from_the_same_session() {
        use super::DaemonRecall;
        use aidaemon_policies::Recall;
        use std::sync::Arc;

        let harness = crate::testing::setup_test_agent_root(MockProvider::new())
            .await
            .unwrap();
        harness
            .state
            .upsert_fact(
                "personal",
                "private_profile_sentinel",
                "SYNTHETIC_CORE_PROFILE_VALUE",
                "owner_dm",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        let state: Arc<dyn crate::traits::StateStore> = harness.state.clone();
        let memory = super::super::DaemonMemory::new(
            state,
            harness.state.memory_stores(),
            harness.agent.event_store().clone(),
            harness.agent.llm_runtime().clone(),
            harness.agent.side_calls().clone(),
            crate::config::ContextWindowConfig::default(),
            10,
            false,
            "",
        );
        let base = crate::agent::memory_plugin::test_support::turn(
            "memory-policy-session",
            "Use the current owner context.",
        )
        .await;
        let turn = |allowed: bool| super::MemoryTurn {
            memory_pipeline_allowed: allowed,
            memory_persistence_allowed: allowed,
            ..base.clone()
        };
        let allowed = DaemonRecall::new(memory.inner.clone(), turn(true))
            .render("memory-policy-session", "", 5)
            .await
            .unwrap_or_default();
        assert!(
            allowed.contains("SYNTHETIC_CORE_PROFILE_VALUE"),
            "{allowed}"
        );
        let suppressed = DaemonRecall::new(memory.inner.clone(), turn(false))
            .render("memory-policy-session", "", 5)
            .await
            .unwrap_or_default();
        assert!(!suppressed.contains("SYNTHETIC_CORE_PROFILE_VALUE"));
    }

    /// A delegated agent in an owner DM still gets the pinned identity facts.
    #[tokio::test]
    async fn child_owner_dm_turn_keeps_critical_identity_facts() {
        use super::DaemonRecall;
        use aidaemon_policies::Recall;
        use std::sync::Arc;

        let harness = crate::testing::setup_test_agent_root(MockProvider::new())
            .await
            .unwrap();
        harness
            .state
            .upsert_fact(
                "user",
                "name",
                "Alice Rivera",
                "user",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        let state: Arc<dyn crate::traits::StateStore> = harness.state.clone();
        let memory = super::super::DaemonMemory::new(
            state,
            harness.state.memory_stores(),
            harness.agent.event_store().clone(),
            harness.agent.llm_runtime().clone(),
            harness.agent.side_calls().clone(),
            crate::config::ContextWindowConfig::default(),
            10,
            false,
            "",
        );
        let turn = super::MemoryTurn {
            depth: 1,
            ..crate::agent::memory_plugin::test_support::turn("child-session", "Draft the reply.")
                .await
        };
        let note = DaemonRecall::new(memory.inner.clone(), turn)
            .render("child-session", "", 5)
            .await
            .unwrap_or_default();
        assert!(note.contains("Owner name → Alice Rivera"), "{note}");
    }
}
