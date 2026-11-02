//! [`SdkMemory`]: the `aidaemon-memory` crate as the memory plugin, through
//! the SDK's public seams only.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use aidaemon_memory::{
    Memory, MemoryBuilder, MemoryRecall, MemoryWriter, RecallTool, RememberTool,
};
use async_trait::async_trait;

use super::{MemoryParts, MemoryPlugin, MemoryTurn};
use crate::types::{ChannelVisibility, UserRole};

/// Long-term memory from `aidaemon-memory`: injected recall, run-end fact
/// extraction, and the `remember` / `recall` tools. Owner-DM memory is one
/// scope; every other conversation keeps its own, so group and DM memories
/// stay apart. Public-external turns get no memory.
pub(crate) struct SdkMemory {
    memory: Memory,
    /// Each session's memory scope, set when its turn starts.
    scopes: Arc<RwLock<SessionScopes>>,
}

/// Sessions whose scope is kept. Run-end extraction resolves the scope after
/// the turn, so an entry must outlive its turn; the least recently started
/// session is evicted past this bound, far beyond any number of turns in
/// flight at once.
const MAX_SESSION_SCOPES: usize = 1024;

/// Session scopes, bounded by [`MAX_SESSION_SCOPES`].
#[derive(Default)]
struct SessionScopes {
    /// Session id → (scope, turn-start sequence).
    entries: HashMap<String, (String, u64)>,
    next: u64,
}

impl SessionScopes {
    fn get(&self, session_id: &str) -> Option<&str> {
        self.entries
            .get(session_id)
            .map(|(scope, _)| scope.as_str())
    }

    fn insert(&mut self, session_id: String, scope: String) {
        self.next += 1;
        self.entries.insert(session_id, (scope, self.next));
        if self.entries.len() > MAX_SESSION_SCOPES {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, (_, seq))| *seq)
                .map(|(session, _)| session.clone());
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
    }
}

impl SdkMemory {
    pub(crate) fn new(builder: MemoryBuilder) -> Self {
        let scopes: Arc<RwLock<SessionScopes>> = Arc::default();
        let resolve = scopes.clone();
        let memory = builder
            .scope_resolver(move |session_id| {
                resolve
                    .read()
                    .ok()
                    .and_then(|scopes| scopes.get(session_id).map(str::to_string))
                    .unwrap_or_else(|| session_scope(session_id))
            })
            .build();
        Self { memory, scopes }
    }
}

fn session_scope(session_id: &str) -> String {
    format!("session:{session_id}")
}

/// The scope a turn reads and writes.
fn turn_scope(turn: &MemoryTurn) -> String {
    if turn.user_role == UserRole::Owner && turn.channel_ctx.should_inject_personal_memory() {
        "owner".to_string()
    } else {
        session_scope(&turn.session_id)
    }
}

#[async_trait]
impl MemoryPlugin for SdkMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        if turn.channel_ctx.visibility == ChannelVisibility::PublicExternal {
            return MemoryParts::default();
        }
        if let Ok(mut scopes) = self.scopes.write() {
            scopes.insert(turn.session_id.clone(), turn_scope(turn));
        }
        let mut parts = MemoryParts::default();
        if turn.memory_pipeline_allowed {
            parts
                .hooks
                .push(Arc::new(aidaemon_policies::MemoryInjection::new(
                    MemoryRecall::new(self.memory.clone()),
                )));
            parts
                .tools
                .push(Arc::new(RecallTool::new(self.memory.clone())));
        }
        if turn.memory_persistence_allowed {
            parts
                .tools
                .push(Arc::new(RememberTool::new(self.memory.clone())));
            if !turn.memory_isolated() {
                parts
                    .hooks
                    .push(Arc::new(MemoryWriter::new(self.memory.clone())));
            }
        }
        parts
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[test]
    fn session_scopes_evict_the_least_recently_started_session() {
        use super::{SessionScopes, MAX_SESSION_SCOPES};

        let mut scopes = SessionScopes::default();
        for index in 0..MAX_SESSION_SCOPES {
            scopes.insert(format!("session-{index}"), "owner".to_string());
        }
        // A new turn in the oldest session makes session-1 the oldest.
        scopes.insert("session-0".to_string(), "owner".to_string());
        scopes.insert("session-new".to_string(), "session:new".to_string());

        assert_eq!(scopes.entries.len(), MAX_SESSION_SCOPES);
        assert_eq!(scopes.get("session-0"), Some("owner"));
        assert_eq!(scopes.get("session-1"), None);
        assert_eq!(scopes.get("session-new"), Some("session:new"));
    }

    use aidaemon_memory::{HashEmbedder, InMemoryMemoryStore, Memory};

    use super::SdkMemory;
    use crate::runtime_ports::{AgentIngress, InboundMessageRequest};
    use crate::testing::MockProvider;
    use crate::traits::store_prelude::*;
    use crate::types::{ChannelContext, UserRole};

    fn request(text: &str) -> InboundMessageRequest {
        InboundMessageRequest {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_text: text.to_string(),
            attachments: Vec::new(),
            status_tx: None,
            user_role: UserRole::Owner,
            channel_ctx: ChannelContext::private("telegram"),
            heartbeat: None,
            ingress_timing: None,
        }
    }

    fn tool_names(call: &crate::testing::MockChatCall) -> Vec<String> {
        call.tools
            .iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|function| function.get("name"))
                    .or_else(|| tool.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    #[tokio::test]
    async fn sdk_memory_plugin_replaces_daemon_memory() {
        let assessment = || {
            MockProvider::semantic_task_assessment(
                "answer",
                false,
                false,
                &[],
                "new_request",
                "general",
            )
        };
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response(
                "remember",
                r#"{"category":"preference","key":"favorite_color","value":"teal"}"#,
            ),
            MockProvider::text_response("Saved."),
            MockProvider::text_response("Teal."),
        ])
        .with_task_assessments(vec![assessment(), assessment()]);
        let mut harness = crate::testing::setup_test_agent_root(provider)
            .await
            .unwrap();
        let memory = Memory::builder(
            Arc::new(InMemoryMemoryStore::new()),
            Arc::new(HashEmbedder::default()),
        );
        harness
            .agent
            .set_memory_plugin(Arc::new(SdkMemory::new(memory)));

        harness
            .agent
            .handle_inbound_message(request("Note my favorite color: teal."))
            .await
            .unwrap();
        harness
            .agent
            .handle_inbound_message(request("What is my favorite color?"))
            .await
            .unwrap();

        let log = harness.provider.call_log.lock().await;
        let last = log.last().unwrap();
        assert!(
            last.messages
                .iter()
                .filter(|m| m["role"] == "system")
                .any(|m| m["content"].as_str().unwrap_or_default().contains("teal")),
            "turn 2 has no recalled note"
        );
        for call in log.iter() {
            let names = tool_names(call);
            assert!(!names.contains(&"remember_fact".to_string()), "{names:?}");
        }
        assert!(tool_names(&log[0]).contains(&"remember".to_string()));
        drop(log);
        assert!(harness.state.get_facts(None).await.unwrap().is_empty());
    }
}
