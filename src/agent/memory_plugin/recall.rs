//! [`DaemonRecall`]: the daemon's SQLite memory, rendered as one system note
//! by `MemoryInjection`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tracing::{info, warn};

use super::{DaemonMemoryInner, MemoryTurn};
use crate::agent::recall_guardrails::{
    build_critical_facts_prompt_block, extract_critical_fact_summary,
};
use crate::events::{DecisionType, DiagnosticSeverity};
use crate::skills::{self, MemoryContext};
use crate::types::{ChannelVisibility, UserRole};

/// Identity/profile categories pinned as critical facts in owner DMs.
const IDENTITY_CATEGORIES: &[&str] = &[
    "identity",
    "personal",
    "profile",
    "user",
    "assistant",
    "bot",
    "relationship",
    "preference",
    "family",
];

/// A recall read that fails soft: the section renders empty, and the error is
/// logged so a broken store does not look like an empty memory.
fn or_logged<T: Default, E: std::fmt::Display>(result: Result<T, E>, section: &str) -> T {
    result.unwrap_or_else(|error| {
        warn!(section, %error, "memory recall read failed; section left empty");
        T::default()
    })
}

/// Recall over the daemon store for one turn. It already knows the channel,
/// role and memory policy, so the `Recall` contract's session and query are
/// enough.
pub(crate) struct DaemonRecall {
    memory: Arc<DaemonMemoryInner>,
    turn: MemoryTurn,
}

impl DaemonRecall {
    pub(crate) fn new(memory: Arc<DaemonMemoryInner>, turn: MemoryTurn) -> Self {
        Self { memory, turn }
    }

    /// Starts the recall filter over `facts`, when the decision layer runs
    /// it. Resolves to the ids of facts to drop: only a calibrated answer
    /// with P(relevant) below 0.15 drops one, and facts in identity
    /// categories are never asked about, so they are always kept. In shadow
    /// mode, or on any failure, nothing is dropped.
    fn spawn_fact_filter(
        &self,
        facts: &[aidaemon_memory::records::Fact],
    ) -> Option<tokio::task::JoinHandle<Option<std::collections::HashSet<i64>>>> {
        use crate::llm_runtime::decisions::Decision;
        let side_calls = self.memory.side_calls.clone();
        if side_calls.decision_mode(Decision::RecallFilter) == crate::config::DecisionMode::Off {
            return None;
        }
        let mut request = aidaemon_decisions::DecisionRequest::new(format!(
            "User message:\n{}",
            crate::utils::truncate_str(&self.turn.user_text, 3000)
        ));
        let mut asked = std::collections::BTreeMap::new();
        for fact in facts
            .iter()
            .filter(|fact| !IDENTITY_CATEGORIES.contains(&fact.category.as_str()))
        {
            let key = format!("f{}", asked.len());
            request.insert(
                key.clone(),
                aidaemon_decisions::Question::yes_no(format!(
                    "Would this remembered fact help answer or carry out the user's message? Fact: [{}] {}: {}",
                    fact.category,
                    fact.key,
                    crate::utils::truncate_str(&fact.value, 300)
                )),
            );
            asked.insert(key, fact.id);
        }
        if asked.is_empty() {
            return None;
        }
        let session = self.turn.session_id.clone();
        let task = self.turn.task_id.clone();
        Some(tokio::spawn(async move {
            let outcome = side_calls
                .decide(
                    Decision::RecallFilter,
                    request,
                    crate::llm_runtime::decide::DecisionContext::turn(&session, &task),
                )
                .await;
            Some(
                asked
                    .iter()
                    .filter(|(key, _)| {
                        outcome
                            .acted(key)
                            .and_then(aidaemon_decisions::Answer::p_yes)
                            .is_some_and(|p| p < 0.5)
                    })
                    .map(|(_, id)| *id)
                    .collect(),
            )
        }))
    }

    /// The daemon memory block: critical facts, core profile, then the
    /// query-ranked memory sections. Channel, role and policy scoping are
    /// applied here.
    async fn render_block(&self) -> String {
        let memory = &self.memory;
        let turn = &self.turn;
        let state = &memory.state;
        let store = &memory.store;
        let channel_ctx = &turn.channel_ctx;
        let user_role = turn.user_role;
        let allowed = turn.memory_pipeline_allowed;
        // Memory retrieval is current-request-scoped.
        let query = turn.user_text.as_str();
        let inject_personal =
            user_role == UserRole::Owner && channel_ctx.should_inject_personal_memory();

        // Facts: always channel-scoped semantic retrieval, so unrelated facts
        // never bleed into prompts for unrelated queries.
        let facts = if !allowed {
            vec![]
        } else {
            or_logged(
                store
                    .get_relevant_facts_for_channel(
                        query,
                        memory.max_facts,
                        channel_ctx.channel_id.as_deref(),
                        channel_ctx.visibility,
                        user_role == UserRole::Owner,
                    )
                    .await,
                "facts",
            )
        };

        // The optional recall filter (decision layer) asks which of these
        // facts matter to the message. It runs beside the lookups below and
        // is awaited where the facts are rendered.
        let fact_filter = self.spawn_fact_filter(&facts);

        // Critical facts (identity/profile): bounded identity categories only,
        // owner DMs (delegated agents included).
        let mut critical_fact_summary = if allowed && inject_personal {
            let mut identity_facts = Vec::new();
            for category in IDENTITY_CATEGORIES {
                identity_facts.append(&mut or_logged(
                    store.get_facts(Some(category)).await,
                    "critical_facts",
                ));
            }
            extract_critical_fact_summary(&identity_facts)
        } else {
            Default::default()
        };
        if let Some(configured_name) = memory.assistant_name.clone() {
            critical_fact_summary.assistant_name = Some(configured_name);
        }

        // Episodes: channel-scoped for non-DM channels.
        let episodes = if !allowed {
            vec![]
        } else if inject_personal {
            or_logged(store.get_relevant_episodes(query, 3).await, "episodes")
        } else {
            match channel_ctx.visibility {
                ChannelVisibility::PublicExternal => vec![],
                _ => or_logged(
                    store
                        .get_relevant_episodes_for_channel(
                            query,
                            3,
                            channel_ctx.channel_id.as_deref(),
                        )
                        .await,
                    "episodes",
                ),
            }
        };

        // Personal goals/profile remain DM-only. Operational failure patterns
        // are safe more broadly: they encode agent-side recovery guidance, not
        // user-private preferences.
        let goals = if allowed && inject_personal {
            or_logged(state.get_active_personal_goals(20).await, "goals")
        } else {
            vec![]
        };
        let patterns =
            if !allowed || matches!(channel_ctx.visibility, ChannelVisibility::PublicExternal) {
                vec![]
            } else if inject_personal {
                or_logged(store.get_behavior_patterns(0.5).await, "patterns")
            } else {
                or_logged(store.get_behavior_patterns(0.5).await, "patterns")
                    .into_iter()
                    .filter(|pattern| pattern.pattern_type == "failure")
                    .collect()
            };
        // Procedures and error solutions are operational: always load, except
        // on PublicExternal where everything is restricted.
        let (procedures, error_solutions) =
            if !allowed || matches!(channel_ctx.visibility, ChannelVisibility::PublicExternal) {
                (vec![], vec![])
            } else {
                (
                    or_logged(store.get_relevant_procedures(query, 5).await, "procedures"),
                    or_logged(
                        store.get_relevant_error_solutions(query, 5).await,
                        "error_solutions",
                    ),
                )
            };
        let profile = if allowed && inject_personal {
            or_logged(store.get_user_profile().await, "profile")
        } else {
            None
        };

        // People context: resolve the current speaker (only when enabled).
        let people_enabled = allowed
            && or_logged(state.get_setting("people_enabled").await, "people_enabled").as_deref()
                == Some("true");
        let (people, current_person, current_person_facts) = if !people_enabled {
            (vec![], None, vec![])
        } else if inject_personal {
            // In owner DMs the people list only selects the privacy rules; the
            // owner's own profile arrives through the core profile.
            (
                or_logged(store.get_all_people().await, "people"),
                None,
                vec![],
            )
        } else if let Some(ref sender_id) = channel_ctx.sender_id {
            match store.get_person_by_platform_id(sender_id).await {
                Ok(Some(person)) => {
                    if let Err(error) = store.touch_person_interaction(person.id).await {
                        warn!(%error, "failed to record person interaction");
                    }
                    let facts = or_logged(
                        store.get_person_facts(person.id, None).await,
                        "person_facts",
                    );
                    (vec![], Some(person), facts)
                }
                Ok(None) => (vec![], None, vec![]),
                Err(error) => {
                    warn!(%error, "memory recall read failed; section left empty");
                    (vec![], None, vec![])
                }
            }
        } else {
            (vec![], None, vec![])
        };

        // Core profile: owner DMs at the top level, with a per-session stable
        // entity selection.
        let core_profile = if allowed && inject_personal && turn.depth == 0 {
            let cached_ids = memory
                .core_profile_ids
                .read()
                .await
                .get(&turn.session_id)
                .cloned();
            let (profile_str, new_ids, digest) = or_logged(
                aidaemon_memory::profile::build_core_profile(
                    store.as_ref(),
                    cached_ids,
                    people_enabled,
                )
                .await,
                "core_profile",
            );
            if let Some(ids) = new_ids {
                memory
                    .core_profile_ids
                    .write()
                    .await
                    .insert(turn.session_id.clone(), ids);
            }
            // The selection digest (id + content hash per entity) makes a
            // profile change self-explaining across turns.
            if !digest.is_empty() {
                self.decision_point(
                    DecisionType::CoreProfileSelection,
                    format!("Core profile selection: {} entities", digest.len()),
                    json!({
                        "code": "core_profile_digest",
                        "count": digest.len(),
                        "entities": digest
                            .iter()
                            .map(|(id, h)| json!({ "id": id, "h": h }))
                            .collect::<Vec<_>>(),
                    }),
                )
                .await;
            }
            profile_str
        } else {
            String::new()
        };

        let facts = match fact_filter {
            Some(filter) => match filter.await {
                Ok(Some(drop)) if !drop.is_empty() => {
                    let before = facts.len();
                    let kept: Vec<_> = facts
                        .into_iter()
                        .filter(|fact| !drop.contains(&fact.id))
                        .collect();
                    info!(
                        dropped = before - kept.len(),
                        kept = kept.len(),
                        "Recall filter dropped facts irrelevant to the message"
                    );
                    kept
                }
                _ => facts,
            },
            None => facts,
        };

        let memory_context = MemoryContext {
            facts: &facts,
            episodes: &episodes,
            goals: &goals,
            patterns: &patterns,
            procedures: &procedures,
            error_solutions: &error_solutions,
            profile: profile.as_ref(),
            people: &people,
            current_person: current_person.as_ref(),
            current_person_facts: &current_person_facts,
        };
        let report = skills::render_memory_section(&memory_context, memory.max_facts);
        if let Err(error) = store
            .record_memory_recall(&report.rendered_fact_ids, &report.rendered_episode_ids)
            .await
        {
            tracing::debug!(%error, "Failed to record rendered memory recall");
        }
        self.decision_point(
            DecisionType::MemoryRetrieval,
            format!(
                "Memory fetched/rendered: facts={}/{} episodes={}/{} procedures={}/{}",
                facts.len(),
                report.rendered_fact_ids.len(),
                episodes.len(),
                report.rendered_episode_ids.len(),
                procedures.len(),
                report.rendered_procedure_ids.len()
            ),
            json!({
                "policy": if allowed { "allowed" } else { "suppressed" },
                "pipeline_suppressed": !allowed,
                "fetched": {
                    "facts": facts.len(),
                    "episodes": episodes.len(),
                    "goals": goals.len(),
                    "patterns": patterns.len(),
                    "procedures": procedures.len(),
                    "error_solutions": error_solutions.len(),
                    "people": people.len(),
                    "current_person_facts": current_person_facts.len()
                },
                "rendered": {
                    "fact_ids": report.rendered_fact_ids,
                    "episode_ids": report.rendered_episode_ids,
                    "goal_ids": report.rendered_goal_ids,
                    "procedure_ids": report.rendered_procedure_ids,
                    "error_solution_ids": report.rendered_error_solution_ids,
                    "pattern_ids": report.rendered_pattern_ids
                }
            }),
        )
        .await;
        info!(
            session_id = %turn.session_id,
            facts = facts.len(),
            episodes = episodes.len(),
            goals = goals.len(),
            patterns = patterns.len(),
            procedures = procedures.len(),
            "Memory context loaded"
        );

        let mut block =
            build_critical_facts_prompt_block(&critical_fact_summary).unwrap_or_default();
        if !core_profile.is_empty() {
            if !block.is_empty() {
                block.push_str("\n\n");
            }
            block.push_str(&core_profile);
        }
        block.push_str(&report.prompt);
        block.trim_start().to_string()
    }

    async fn decision_point(
        &self,
        decision_type: DecisionType,
        summary: String,
        metadata: serde_json::Value,
    ) {
        if !self.memory.record_decision_points {
            return;
        }
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

#[async_trait]
impl aidaemon_policies::Recall for DaemonRecall {
    /// Channel-scoped fact lines, for generic callers.
    async fn recall(&self, _session_id: &str, query: &str, limit: usize) -> Vec<String> {
        if !self.turn.memory_pipeline_allowed {
            return Vec::new();
        }
        let channel_ctx = &self.turn.channel_ctx;
        let facts = self
            .memory
            .store
            .get_relevant_facts_for_channel(
                query,
                limit,
                channel_ctx.channel_id.as_deref(),
                channel_ctx.visibility,
                self.turn.user_role == UserRole::Owner,
            )
            .await;
        or_logged(facts, "facts")
            .into_iter()
            .map(|fact| format!("[{}] {}: {}", fact.category, fact.key, fact.value))
            .collect()
    }

    /// The full daemon memory block with its section headings.
    async fn render(&self, _session_id: &str, _query: &str, _limit: usize) -> Option<String> {
        let block = self.render_block().await;
        (!block.trim().is_empty()).then_some(block)
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
    use crate::runtime_ports::{AgentIngress, InboundMessageRequest};
    use crate::testing::{MockProvider, TestHarness};
    use crate::traits::store_prelude::*;
    use crate::types::{ChannelContext, ChannelVisibility, FactPrivacy, UserRole};
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
            .event_store
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
            harness.agent.llm_runtime.clone(),
            harness.agent.side_calls.clone(),
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
            harness.agent.llm_runtime.clone(),
            harness.agent.side_calls.clone(),
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
