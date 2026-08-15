use std::sync::Arc;

use async_trait::async_trait;

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{Channel, ChannelCapabilities};
use crate::types::{ApprovalResponse, MediaMessage};

use super::domain::NODE_CHANNEL_NAME;
use super::store::NodeStore;

pub struct NodeChannel {
    store: Arc<NodeStore>,
}

impl NodeChannel {
    pub fn new(store: Arc<NodeStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Channel for NodeChannel {
    fn name(&self) -> String {
        NODE_CHANNEL_NAME.to_string()
    }
    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: false,
            inline_buttons: false,
            media: false,
            max_message_len: 4096,
        }
    }
    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        self.store
            .append_outbound_for_conversation(session_id, text)
            .await?;
        Ok(())
    }
    async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
        anyhow::bail!("Node media delivery is not enabled")
    }
    async fn request_approval(
        &self,
        _session_id: &str,
        _command: &str,
        _risk_level: RiskLevel,
        _warnings: &[String],
        _permission_mode: PermissionMode,
        _one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        Ok(ApprovalResponse::Deny)
    }
    async fn request_goal_confirmation(
        &self,
        _session_id: &str,
        _goal_description: &str,
        _details: &[String],
        _style: crate::types::GoalConfirmationStyle,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }
}
