//! The Node service from `aidaemon-nodes`, plus the daemon's conversation
//! ingress: Node turns answered by the agent.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use aidaemon_nodes::domain::AuthenticatedNodeContext;
pub use aidaemon_nodes::service::*;

use crate::agent::Agent;
use crate::traits::MessageAttachment;
use crate::types::{ChannelContext, ChannelVisibility, UserRole};

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
        let channel_context = ChannelContext {
            visibility: ChannelVisibility::PublicExternal,
            platform: "node".to_string(),
            channel_name: Some("Companion".to_string()),
            channel_id: Some(format!("node:{}", context.node_channel_id)),
            workspace_id: None,
            sender_name: Some("Companion user".to_string()),
            sender_id: Some(format!("node-principal:{}", context.node_id)),
            channel_member_names: Vec::new(),
            user_id_map: HashMap::new(),
            workspace_grant: None,
            trusted: false,
        };
        self.agent
            .handle_message(
                &context.conversation_session_id,
                text,
                None,
                UserRole::Public,
                channel_context,
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
        let channel_context = node_channel_context(context);
        self.agent
            .handle_message_with_attachments(
                &context.conversation_session_id,
                text,
                &[attachment],
                None,
                UserRole::Public,
                channel_context,
                None,
            )
            .await
    }
}

fn node_channel_context(context: &AuthenticatedNodeContext) -> ChannelContext {
    ChannelContext {
        visibility: ChannelVisibility::PublicExternal,
        platform: "node".to_string(),
        channel_name: Some("Companion".to_string()),
        channel_id: Some(format!("node:{}", context.node_channel_id)),
        workspace_id: None,
        sender_name: Some("Companion user".to_string()),
        sender_id: Some(format!("node-principal:{}", context.node_id)),
        channel_member_names: Vec::new(),
        user_id_map: HashMap::new(),
        workspace_grant: None,
        trusted: false,
    }
}
