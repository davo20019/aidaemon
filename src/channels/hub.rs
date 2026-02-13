use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use crate::config::QueuePolicyConfig;
use crate::queue_policy::{should_shed_due_to_overload, SessionFairnessBudget};
use crate::queue_telemetry::{QueuePressure, QueueTelemetry};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::traits::Channel;
use crate::types::{ApprovalResponse, MediaMessage};

/// Shared map of session_id → channel name.
/// Written by channels when they receive incoming messages,
/// read by the hub to route outbound messages (approvals, media, notifications).
pub type SessionMap = Arc<RwLock<HashMap<String, String>>>;

/// Central router for outbound messages across all channels.
///
/// The hub routes approval requests, media, and notifications to the
/// correct channel based on which channel originated the session.
/// Unknown sessions are refused (returns None) to prevent cross-channel
/// privacy leaks.
pub struct ChannelHub {
    /// Registered channels. Uses RwLock to support dynamic registration.
    channels: RwLock<Vec<Arc<dyn Channel>>>,
    session_map: SessionMap,
    queue_telemetry: Option<Arc<QueueTelemetry>>,
    queue_policy: Option<QueuePolicyConfig>,
}

impl ChannelHub {
    pub fn new(channels: Vec<Arc<dyn Channel>>, session_map: SessionMap) -> Self {
        Self {
            channels: RwLock::new(channels),
            session_map,
            queue_telemetry: None,
            queue_policy: None,
        }
    }

    pub fn with_queue_telemetry(mut self, queue_telemetry: Arc<QueueTelemetry>) -> Self {
        self.queue_telemetry = Some(queue_telemetry);
        self
    }

    pub fn with_queue_policy(mut self, queue_policy: QueuePolicyConfig) -> Self {
        self.queue_policy = Some(queue_policy);
        self
    }

    /// Register a new channel dynamically.
    /// Returns the channel name after registration.
    #[allow(dead_code)]
    pub async fn register_channel(&self, channel: Arc<dyn Channel>) -> String {
        let name = channel.name();
        let mut channels = self.channels.write().await;
        channels.push(channel);
        info!(channel = %name, total = channels.len(), "Registered new channel");
        name
    }

    /// Get a reference to the shared session map.
    #[allow(dead_code)]
    pub fn session_map(&self) -> &SessionMap {
        &self.session_map
    }

    /// Find the channel that owns a session.
    /// Returns None for unknown sessions to prevent cross-channel privacy leaks.
    async fn channel_for_session(&self, session_id: &str) -> Option<Arc<dyn Channel>> {
        let map = self.session_map.read().await;
        let channels = self.channels.read().await;
        if let Some(channel_name) = map.get(session_id) {
            if let Some(ch) = channels.iter().find(|c| &c.name() == channel_name) {
                return Some(ch.clone());
            }
        }
        None
    }

    /// Request approval through a channel that supports inline buttons.
    ///
    /// Used for UX flows that require button consistency (for example scheduled
    /// goal confirmation) while preserving text fallback in non-inline channels.
    pub async fn request_inline_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse> {
        let channel = self
            .channel_for_session(session_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("No channel found for session {}", session_id))?;
        if !channel.capabilities().inline_buttons {
            anyhow::bail!(
                "Channel {} does not support inline approval buttons",
                channel.name()
            );
        }
        channel
            .request_approval(session_id, command, risk_level, warnings, permission_mode)
            .await
    }

    /// Route approval requests from tools to the appropriate channel.
    ///
    /// Each approval is handled in its own task so the listener doesn't
    /// block while waiting for the user to respond.
    pub async fn approval_listener(self: Arc<Self>, mut rx: mpsc::Receiver<ApprovalRequest>) {
        let mut fair_session_budget: SessionFairnessBudget = HashMap::new();
        loop {
            let request = match rx.recv().await {
                Some(r) => r,
                None => break, // channel closed
            };
            let approval_depth = rx.len().saturating_add(1);
            let mut pressure = QueuePressure::Normal;
            if let Some(queue_telemetry) = &self.queue_telemetry {
                queue_telemetry.mark_approval_received();
                let observation = queue_telemetry.observe_approval_depth(approval_depth);
                pressure = observation.pressure;
                if observation.entered_warning {
                    warn!(
                        queue = "approval",
                        depth = approval_depth,
                        "Approval queue entered warning state"
                    );
                }
                if observation.entered_overload {
                    warn!(
                        queue = "approval",
                        depth = approval_depth,
                        "Approval queue entered overload state"
                    );
                }
            }

            let should_shed = if let Some(queue_policy) = &self.queue_policy {
                should_shed_due_to_overload(
                    &queue_policy.lanes.approval,
                    pressure,
                    &mut fair_session_budget,
                    &request.session_id,
                )
            } else {
                false
            };

            if should_shed {
                let mut had_error = false;
                if request.response_tx.send(ApprovalResponse::Deny).is_err() {
                    had_error = true;
                    warn!(
                        session_id = %request.session_id,
                        "Approval response receiver dropped before overload-shed deny could be sent"
                    );
                }
                if let Some(queue_telemetry) = &self.queue_telemetry {
                    queue_telemetry.mark_approval_dropped(1);
                    if had_error {
                        queue_telemetry.mark_approval_failed();
                    }
                    queue_telemetry.mark_approval_completed();
                }
                warn!(
                    session_id = %request.session_id,
                    "Dropping approval request due to configured overload shedding policy"
                );
                continue;
            }

            let hub = self.clone();
            tokio::spawn(async move {
                let queue_telemetry = hub.queue_telemetry.clone();
                let channel = hub.channel_for_session(&request.session_id).await;
                let mut had_error = false;
                let response = match channel {
                    Some(ch) => {
                        match ch
                            .request_approval(
                                &request.session_id,
                                &request.command,
                                request.risk_level,
                                &request.warnings,
                                request.permission_mode,
                            )
                            .await
                        {
                            Ok(resp) => resp,
                            Err(e) => {
                                warn!("Approval request failed on {}: {}", ch.name(), e);
                                had_error = true;
                                ApprovalResponse::Deny
                            }
                        }
                    }
                    None => {
                        warn!(
                            "No channel found for session {}, denying",
                            request.session_id
                        );
                        had_error = true;
                        ApprovalResponse::Deny
                    }
                };
                if request.response_tx.send(response).is_err() {
                    had_error = true;
                    warn!(
                        session_id = %request.session_id,
                        "Approval response receiver dropped before response could be sent"
                    );
                }
                if let Some(queue_telemetry) = queue_telemetry {
                    if had_error {
                        queue_telemetry.mark_approval_failed();
                    }
                    queue_telemetry.mark_approval_completed();
                }
            });
        }
    }

    /// Route media messages from tools to the appropriate channel.
    pub async fn media_listener(self: Arc<Self>, mut rx: mpsc::Receiver<MediaMessage>) {
        let mut fair_session_budget: SessionFairnessBudget = HashMap::new();
        loop {
            let msg = match rx.recv().await {
                Some(m) => m,
                None => break, // channel closed
            };
            let media_depth = rx.len().saturating_add(1);
            let mut pressure = QueuePressure::Normal;
            if let Some(queue_telemetry) = &self.queue_telemetry {
                queue_telemetry.mark_media_received();
                let observation = queue_telemetry.observe_media_depth(media_depth);
                pressure = observation.pressure;
                if observation.entered_warning {
                    warn!(
                        queue = "media",
                        depth = media_depth,
                        "Media queue entered warning state"
                    );
                }
                if observation.entered_overload {
                    warn!(
                        queue = "media",
                        depth = media_depth,
                        "Media queue entered overload state; shedding non-critical media work"
                    );
                }
            }

            let should_shed = if let Some(queue_policy) = &self.queue_policy {
                should_shed_due_to_overload(
                    &queue_policy.lanes.media,
                    pressure,
                    &mut fair_session_budget,
                    &msg.session_id,
                )
            } else {
                false
            };

            if should_shed {
                let mut had_error = false;
                if let Some(channel) = self.channel_for_session(&msg.session_id).await {
                    if let Err(e) = channel
                        .send_text(
                            &msg.session_id,
                            "[Media skipped due high system load. Please retry shortly.]",
                        )
                        .await
                    {
                        had_error = true;
                        warn!(
                            "Failed to send overload media fallback via {}: {}",
                            channel.name(),
                            e
                        );
                    }
                } else {
                    had_error = true;
                    warn!(
                        "No channel found for overloaded media session {}",
                        msg.session_id
                    );
                }
                if let Some(queue_telemetry) = &self.queue_telemetry {
                    queue_telemetry.mark_media_dropped();
                    if had_error {
                        queue_telemetry.mark_media_failed();
                    }
                    queue_telemetry.mark_media_completed();
                }
                continue;
            }

            let mut had_error = false;
            if let Some(channel) = self.channel_for_session(&msg.session_id).await {
                if channel.capabilities().media {
                    if let Err(e) = channel.send_media(&msg.session_id, &msg).await {
                        had_error = true;
                        warn!("Failed to send media via {}: {}", channel.name(), e);
                    }
                } else {
                    // Channel doesn't support media — send caption as text
                    if let Err(e) = channel
                        .send_text(&msg.session_id, &format!("[Media] {}", msg.caption))
                        .await
                    {
                        had_error = true;
                        warn!("Failed to send media caption via {}: {}", channel.name(), e);
                    }
                }
            } else {
                had_error = true;
                warn!("No channel found for media session {}", msg.session_id);
            }
            if let Some(queue_telemetry) = &self.queue_telemetry {
                if had_error {
                    queue_telemetry.mark_media_failed();
                }
                queue_telemetry.mark_media_completed();
            }
        }
    }

    /// Send text to the channel that owns a specific session.
    #[allow(dead_code)]
    pub async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        if let Some(channel) = self.channel_for_session(session_id).await {
            channel.send_text(session_id, text).await
        } else {
            anyhow::bail!("No channel found for session {}", session_id)
        }
    }

    /// Send media to the channel that owns a specific session.
    /// Falls back to text caption for channels without media support.
    pub async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        if let Some(channel) = self.channel_for_session(session_id).await {
            if channel.capabilities().media {
                channel.send_media(session_id, media).await
            } else {
                channel
                    .send_text(session_id, &format!("[File] {}", media.caption))
                    .await
            }
        } else {
            anyhow::bail!("No channel found for session {}", session_id)
        }
    }

    /// Broadcast text to a list of session IDs (e.g., trigger notifications).
    /// Errors are logged but don't stop the broadcast.
    pub async fn broadcast_text(&self, session_ids: &[String], text: &str) {
        for session_id in session_ids {
            if let Some(channel) = self.channel_for_session(session_id).await {
                if let Err(e) = channel.send_text(session_id, text).await {
                    warn!(
                        channel = channel.name(),
                        session_id, "Broadcast send failed: {}", e
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    use async_trait::async_trait;
    use tokio::sync::Mutex;

    use crate::tools::command_risk::{PermissionMode, RiskLevel};
    use crate::traits::{Channel, ChannelCapabilities};
    use crate::types::{ApprovalResponse, MediaMessage};

    /// A test channel with a configurable name, used to verify routing.
    struct NamedTestChannel {
        channel_name: String,
        messages: Mutex<Vec<(String, String)>>, // (session_id, text)
    }

    impl NamedTestChannel {
        fn new(name: &str) -> Self {
            Self {
                channel_name: name.to_string(),
                messages: Mutex::new(Vec::new()),
            }
        }

        async fn captured_messages(&self) -> Vec<(String, String)> {
            self.messages.lock().await.clone()
        }
    }

    #[async_trait]
    impl Channel for NamedTestChannel {
        fn name(&self) -> String {
            self.channel_name.clone()
        }

        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities {
                markdown: true,
                inline_buttons: false,
                media: false,
                max_message_len: 4096,
            }
        }

        async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
            self.messages
                .lock()
                .await
                .push((session_id.to_string(), text.to_string()));
            Ok(())
        }

        async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn request_approval(
            &self,
            _session_id: &str,
            _command: &str,
            _risk_level: RiskLevel,
            _warnings: &[String],
            _permission_mode: PermissionMode,
        ) -> anyhow::Result<ApprovalResponse> {
            Ok(ApprovalResponse::AllowOnce)
        }
    }

    fn empty_session_map() -> SessionMap {
        Arc::new(RwLock::new(HashMap::new()))
    }

    fn session_map_with(entries: Vec<(&str, &str)>) -> SessionMap {
        let mut map = HashMap::new();
        for (session, channel) in entries {
            map.insert(session.to_string(), channel.to_string());
        }
        Arc::new(RwLock::new(map))
    }

    #[tokio::test]
    async fn test_channel_for_session_known() {
        let ch_telegram: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("slack"));

        let session_map = session_map_with(vec![("sess_1", "slack")]);
        let hub = ChannelHub::new(vec![ch_telegram, ch_slack], session_map);

        let found = hub.channel_for_session("sess_1").await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().name(), "slack");
    }

    #[tokio::test]
    async fn test_channel_for_session_unknown_returns_none() {
        let ch_telegram: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("slack"));

        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![ch_telegram, ch_slack], session_map);

        // Unknown session should return None to prevent cross-channel leaks
        let found = hub.channel_for_session("unknown_session").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_channel_for_session_empty() {
        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![], session_map);

        let found = hub.channel_for_session("any_session").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_send_text_routes_correctly() {
        let ch_telegram = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack = Arc::new(NamedTestChannel::new("slack"));

        let ch_telegram_dyn: Arc<dyn Channel> = ch_telegram.clone();
        let ch_slack_dyn: Arc<dyn Channel> = ch_slack.clone();

        let session_map = session_map_with(vec![("sess_1", "slack")]);
        let hub = ChannelHub::new(vec![ch_telegram_dyn, ch_slack_dyn], session_map);

        hub.send_text("sess_1", "Hello Slack!").await.unwrap();

        // Slack channel should have the message
        let slack_msgs = ch_slack.captured_messages().await;
        assert_eq!(slack_msgs.len(), 1);
        assert_eq!(slack_msgs[0].0, "sess_1");
        assert_eq!(slack_msgs[0].1, "Hello Slack!");

        // Telegram channel should have no messages
        let telegram_msgs = ch_telegram.captured_messages().await;
        assert_eq!(telegram_msgs.len(), 0);
    }

    #[tokio::test]
    async fn test_send_text_no_channels_errors() {
        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![], session_map);

        let result = hub.send_text("sess_1", "Hello?").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No channel found"),
            "Expected 'No channel found' error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_broadcast_sends_to_all() {
        let ch_telegram = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack = Arc::new(NamedTestChannel::new("slack"));

        let ch_telegram_dyn: Arc<dyn Channel> = ch_telegram.clone();
        let ch_slack_dyn: Arc<dyn Channel> = ch_slack.clone();

        let session_map =
            session_map_with(vec![("sess_telegram", "telegram"), ("sess_slack", "slack")]);
        let hub = ChannelHub::new(vec![ch_telegram_dyn, ch_slack_dyn], session_map);

        let ids = vec!["sess_telegram".to_string(), "sess_slack".to_string()];
        hub.broadcast_text(&ids, "Broadcast!").await;

        let telegram_msgs = ch_telegram.captured_messages().await;
        assert_eq!(telegram_msgs.len(), 1);
        assert_eq!(telegram_msgs[0].1, "Broadcast!");

        let slack_msgs = ch_slack.captured_messages().await;
        assert_eq!(slack_msgs.len(), 1);
        assert_eq!(slack_msgs[0].1, "Broadcast!");
    }

    #[tokio::test]
    async fn test_register_channel_dynamically() {
        let session_map = session_map_with(vec![("sess_1", "discord")]);
        let hub = ChannelHub::new(vec![], session_map);

        // Initially no channels, so send_text should fail
        assert!(hub.send_text("sess_1", "test").await.is_err());

        // Register a channel dynamically
        let ch_discord: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("discord"));
        let name = hub.register_channel(ch_discord).await;
        assert_eq!(name, "discord");

        // Now send_text should succeed
        let result = hub.send_text("sess_1", "Hello Discord!").await;
        assert!(result.is_ok());
    }
}
