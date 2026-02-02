use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, RwLock};
use tracing::warn;

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
/// If the session's channel is unknown, it falls back to the first
/// registered channel.
pub struct ChannelHub {
    channels: Vec<Arc<dyn Channel>>,
    session_map: SessionMap,
}

impl ChannelHub {
    pub fn new(channels: Vec<Arc<dyn Channel>>, session_map: SessionMap) -> Self {
        Self {
            channels,
            session_map,
        }
    }

    /// Get a reference to the shared session map.
    #[allow(dead_code)]
    pub fn session_map(&self) -> &SessionMap {
        &self.session_map
    }

    /// Find the channel that owns a session, falling back to the first channel.
    async fn channel_for_session(&self, session_id: &str) -> Option<Arc<dyn Channel>> {
        let map = self.session_map.read().await;
        if let Some(channel_name) = map.get(session_id) {
            if let Some(ch) = self.channels.iter().find(|c| c.name() == channel_name) {
                return Some(ch.clone());
            }
        }
        // Fallback: first registered channel
        self.channels.first().cloned()
    }

    /// Route approval requests from tools to the appropriate channel.
    ///
    /// Each approval is handled in its own task so the listener doesn't
    /// block while waiting for the user to respond.
    pub async fn approval_listener(self: Arc<Self>, mut rx: mpsc::Receiver<ApprovalRequest>) {
        loop {
            let request = match rx.recv().await {
                Some(r) => r,
                None => break, // channel closed
            };

            let hub = self.clone();
            tokio::spawn(async move {
                let channel = hub.channel_for_session(&request.session_id).await;
                let response = match channel {
                    Some(ch) => {
                        match ch
                            .request_approval(&request.session_id, &request.command)
                            .await
                        {
                            Ok(resp) => resp,
                            Err(e) => {
                                warn!("Approval request failed on {}: {}", ch.name(), e);
                                ApprovalResponse::Deny
                            }
                        }
                    }
                    None => {
                        warn!(
                            "No channel found for session {}, denying",
                            request.session_id
                        );
                        ApprovalResponse::Deny
                    }
                };
                let _ = request.response_tx.send(response);
            });
        }
    }

    /// Route media messages from tools to the appropriate channel.
    pub async fn media_listener(self: Arc<Self>, mut rx: mpsc::Receiver<MediaMessage>) {
        loop {
            let msg = match rx.recv().await {
                Some(m) => m,
                None => break, // channel closed
            };

            if let Some(channel) = self.channel_for_session(&msg.session_id).await {
                if channel.capabilities().media {
                    if let Err(e) = channel.send_media(&msg.session_id, &msg).await {
                        warn!("Failed to send media via {}: {}", channel.name(), e);
                    }
                } else {
                    // Channel doesn't support media — send caption as text
                    if let Err(e) = channel
                        .send_text(&msg.session_id, &format!("[Media] {}", msg.caption))
                        .await
                    {
                        warn!("Failed to send media caption via {}: {}", channel.name(), e);
                    }
                }
            } else {
                warn!("No channel found for media session {}", msg.session_id);
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
