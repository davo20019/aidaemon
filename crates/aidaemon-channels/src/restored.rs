//! Messages a previous run of the daemon accepted to wait for their turn,
//! run after a restart.
//!
//! The task registry journals its queues ([`crate::tasks::QueueSnapshot`])
//! and the daemon restores them at boot; nothing is running then to drain
//! them, so each restored session's waiting messages start a turn of their
//! own here, in order, on the chat's surface as the hub reaches it.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::ports::OutboundRouter;
use crate::turns::{Turn, TurnRunner, TurnSurface};

/// A chat as the hub reaches it: what a restored message's turn says goes
/// through the hub to the chat's bot, with no typing indicator (the turn's
/// own status updates still show where the chat shows them).
pub struct RouterSurface {
    router: Arc<dyn OutboundRouter>,
    session_id: String,
    platform: &'static str,
}

impl RouterSurface {
    /// The surface of `session_id`, a chat on `platform` (the waiting
    /// message's `ChannelContext::platform`).
    pub fn new(router: Arc<dyn OutboundRouter>, session_id: String, platform: &str) -> Self {
        // The platform names the bots record on their receipts: a machine
        // format the daemon itself writes into the context.
        let platform = match platform {
            "app" => "app",
            "telegram" => "telegram",
            "slack" => "slack",
            "discord" => "discord",
            _ => "internal",
        };
        Self {
            router,
            session_id,
            platform,
        }
    }
}

#[async_trait]
impl TurnSurface for RouterSurface {
    fn platform(&self) -> &'static str {
        self.platform
    }

    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        Ok(self
            .router
            .send_text_tracked(&self.session_id, text)
            .await?
            .into_iter()
            .collect())
    }

    async fn edit(&self, message_id: &str, text: &str) -> anyhow::Result<bool> {
        self.router
            .edit_text(&self.session_id, message_id, text)
            .await
    }

    fn show_working(&self, _stop: CancellationToken, _heartbeat: Arc<AtomicU64>, _stale: u64) {}
}

impl TurnRunner {
    /// Run `session_id`'s waiting messages, restored after a restart: the
    /// first starts a turn, and each after it runs when the one before
    /// ends, as messages queued behind any turn do. Nothing when they were
    /// already taken (a turn the owner started since drained them).
    pub async fn run_restored(self, surface: Arc<dyn TurnSurface>, session_id: &str) {
        let registry = self.deps.task_registry.clone();
        if registry.has_running_task(session_id).await {
            // The running turn drains them when it ends.
            return;
        }
        let Some(first) = registry.pop_all_queued_messages(session_id).await else {
            return;
        };
        tracing::info!(session_id, "Running a message a restart left waiting");
        self.run(
            surface,
            Turn {
                session_id: session_id.to_string(),
                text: first.text,
                attachments: first.attachments,
                user_role: first.user_role,
                channel_ctx: first.channel_ctx,
                dedup_key: None,
                ingress_timing: None,
            },
        )
        .await;
    }
}
