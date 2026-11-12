//! [`SystemMemory`]: any `aidaemon_policies::MemorySystem` as the daemon's
//! memory plugin. The daemon decides the turn's scope and permissions from
//! its users and channels; the memory system decides how to remember and
//! recall within them. `aidaemon_memory::ScopedMemory` is the built-in
//! system; another backend implements the same trait in its own crate.

use std::sync::Arc;

use aidaemon_policies::MemorySystem;
use async_trait::async_trait;

use super::{MemoryParts, MemoryPlugin, MemoryTurn};
use crate::types::{ChannelVisibility, UserRole};

/// A whole memory system plugged into daemon turns.
// Only the `sdk-memory` backend constructs it today; the seam is not gated.
#[cfg_attr(not(feature = "sdk-memory"), allow(dead_code))]
pub(crate) struct SystemMemory {
    system: Arc<dyn MemorySystem>,
}

#[cfg_attr(not(feature = "sdk-memory"), allow(dead_code))]
impl SystemMemory {
    pub(crate) fn new(system: Arc<dyn MemorySystem>) -> Self {
        Self { system }
    }
}

/// The memory a turn reads and writes, or `None` for no memory. Owner-DM
/// memory is one scope; every other conversation keeps its own, so group
/// and DM memories stay apart. Public-external turns get no memory.
fn turn_scope(turn: &MemoryTurn) -> Option<String> {
    if turn.channel_ctx.visibility == ChannelVisibility::PublicExternal {
        None
    } else if turn.user_role == UserRole::Owner && turn.channel_ctx.should_inject_personal_memory()
    {
        Some("owner".to_string())
    } else {
        Some(format!("session:{}", turn.session_id))
    }
}

/// What the daemon allows the memory system on `turn`.
fn system_turn(turn: &MemoryTurn) -> Option<aidaemon_policies::MemoryTurn> {
    Some(aidaemon_policies::MemoryTurn {
        session_id: turn.session_id.clone(),
        scope: turn_scope(turn)?,
        read: turn.memory_pipeline_allowed,
        write: turn.memory_persistence_allowed,
        explicit_writes_only: turn.memory_isolated(),
        learning: turn.learning_consent.clone(),
        host: Default::default(),
    })
}

/// The daemon's [`MemoryTurn`] for a memory system that reads its host's
/// facts (`host`). Such a system scopes the turn itself, so a turn with no
/// scope of its own (public-external) is passed on with an empty one.
pub(super) fn host_system_turn(
    turn: &MemoryTurn,
    host: aidaemon_policies::HostTurn,
) -> aidaemon_policies::MemoryTurn {
    aidaemon_policies::MemoryTurn {
        session_id: turn.session_id.clone(),
        scope: turn_scope(turn).unwrap_or_default(),
        read: turn.memory_pipeline_allowed,
        write: turn.memory_persistence_allowed,
        explicit_writes_only: turn.memory_isolated(),
        learning: turn.learning_consent.clone(),
        host,
    }
}

#[async_trait]
impl MemoryPlugin for SystemMemory {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let Some(system_turn) = system_turn(turn) else {
            return MemoryParts::default();
        };
        let parts = self.system.turn(&system_turn).await;
        MemoryParts {
            hooks: parts.hooks,
            tools: parts.tools,
            ..MemoryParts::default()
        }
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;
    use crate::types::ChannelContext;

    #[tokio::test]
    async fn the_daemon_decides_scope_and_permissions() {
        let owner = super::super::test_support::turn("telegram:synthetic-user-1", "hi").await;
        let decided = system_turn(&owner).unwrap();
        assert_eq!(decided.scope, "owner");
        assert!(decided.read && decided.write && !decided.explicit_writes_only);

        let child = MemoryTurn {
            depth: 1,
            ..owner.clone()
        };
        assert!(system_turn(&child).unwrap().explicit_writes_only);

        let guest = MemoryTurn {
            user_role: UserRole::Guest,
            ..owner.clone()
        };
        assert_eq!(
            system_turn(&guest).unwrap().scope,
            "session:telegram:synthetic-user-1"
        );

        let public = MemoryTurn {
            channel_ctx: ChannelContext {
                visibility: ChannelVisibility::PublicExternal,
                ..owner.channel_ctx.clone()
            },
            ..owner
        };
        assert!(system_turn(&public).is_none());
    }
}

#[cfg(all(test, feature = "sdk-memory"))]
mod tests {
    use std::sync::Arc;

    use aidaemon_memory::{HashEmbedder, InMemoryMemoryStore, Memory};

    use super::SystemMemory;
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
    async fn memory_system_replaces_daemon_memory() {
        let provider = MockProvider::with_responses(vec![
            MockProvider::tool_call_response(
                "remember",
                r#"{"category":"preference","key":"favorite_color","value":"teal"}"#,
            ),
            MockProvider::text_response("Saved."),
            MockProvider::text_response("Teal."),
        ]);
        let mut harness = crate::testing::setup_test_agent_root(provider)
            .await
            .unwrap();
        let memory = Memory::builder(
            Arc::new(InMemoryMemoryStore::new()),
            Arc::new(HashEmbedder::default()),
        );
        harness
            .agent
            .set_memory_plugin(Arc::new(SystemMemory::new(Arc::new(
                aidaemon_memory::ScopedMemory::new(memory),
            ))));

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
