//! The Node service from `aidaemon-nodes`, plus the daemon's conversation
//! ingress: Node turns answered by the agent.

use std::sync::Arc;

use async_trait::async_trait;

use aidaemon_nodes::domain::AuthenticatedNodeContext;
pub use aidaemon_nodes::service::*;

use crate::agent::Agent;
use crate::traits::MessageAttachment;
use crate::types::UserRole;

/// Node turns answered by the daemon's agent.
pub struct AgentNodeConversationIngress {
    agent: Arc<Agent>,
}

impl AgentNodeConversationIngress {
    pub fn new(agent: Arc<Agent>) -> Self {
        Self { agent }
    }
}

#[async_trait]
impl NodeConversationIngress for AgentNodeConversationIngress {
    async fn respond(
        &self,
        context: &AuthenticatedNodeContext,
        text: &str,
    ) -> anyhow::Result<String> {
        self.agent
            .handle_message(
                &context.conversation_session_id,
                text,
                None,
                UserRole::Public,
                node_channel_context(context),
                None,
            )
            .await
    }

    async fn respond_with_attachment(
        &self,
        context: &AuthenticatedNodeContext,
        text: &str,
        attachment: MessageAttachment,
    ) -> anyhow::Result<String> {
        self.agent
            .handle_message_with_attachments(
                &context.conversation_session_id,
                text,
                &[attachment],
                None,
                UserRole::Public,
                node_channel_context(context),
                None,
            )
            .await
    }
}
