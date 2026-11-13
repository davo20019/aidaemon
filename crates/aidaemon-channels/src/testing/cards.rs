//! A chat that keeps cards, as the owner sees it: a [`CardChannel`] behind
//! a real [`ChannelHub`], for tests of tools that report through the hub.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};

use crate::hub::{ChannelCapabilities, ChannelHub, HubChannel, MediaMessage, SessionMap};
use aidaemon_tools::semantics::{ApprovalResponse, PermissionMode, RiskLevel};

/// A channel whose messages can be edited: each message keeps its id and
/// its latest text, so a test reads the chat exactly as the owner sees it.
#[derive(Default)]
pub struct CardChannel {
    shown: Mutex<Vec<(String, String)>>,
}

#[async_trait]
impl HubChannel for CardChannel {
    fn name(&self) -> String {
        "cards".to_string()
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
        self.send_text_tracked(session_id, text).await.map(|_| ())
    }

    async fn send_text_tracked(
        &self,
        _session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>> {
        let mut shown = self.shown.lock().await;
        let id = format!("m{}", shown.len() + 1);
        shown.push((id.clone(), text.to_string()));
        Ok(Some(id))
    }

    async fn edit_text(
        &self,
        _session_id: &str,
        message_id: &str,
        text: &str,
    ) -> anyhow::Result<bool> {
        let mut shown = self.shown.lock().await;
        match shown.iter_mut().find(|(id, _)| id == message_id) {
            Some((_, shown)) => {
                *shown = text.to_string();
                Ok(true)
            }
            None => Ok(false),
        }
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
        _one_time_only: bool,
    ) -> anyhow::Result<ApprovalResponse> {
        Ok(ApprovalResponse::AllowOnce)
    }
}

/// One chat on a [`CardChannel`], with turns run the way the channel
/// runner runs them: the turn's status drives its live card, the card ends
/// by the one rule when the turn does, a kept card is registered, and only
/// then is the turn over.
pub struct CardChat {
    pub hub: Arc<ChannelHub>,
    pub channel: Arc<CardChannel>,
    pub session: String,
}

impl CardChat {
    pub async fn new(session: &str) -> Self {
        let channel = Arc::new(CardChannel::default());
        let session_map: SessionMap = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
        session_map
            .write()
            .await
            .insert(session.to_string(), "cards".to_string());
        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn HubChannel>],
            session_map,
        ));
        Self {
            hub,
            channel,
            session: session.to_string(),
        }
    }

    pub fn outbound(&self) -> std::sync::Weak<dyn crate::ports::OutboundRouter> {
        let hub: Arc<dyn crate::ports::OutboundRouter> = self.hub.clone();
        Arc::downgrade(&hub)
    }

    /// Run `start` as one turn and send `reply(&output)` as its reply.
    pub async fn turn<T, F, Fut>(&self, start: F, reply: impl FnOnce(&T) -> String) -> T
    where
        F: FnOnce(mpsc::Sender<aidaemon_tools::status::StatusUpdate>) -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        use crate::live_status::{HubSurfaceSink, LiveStatus};
        let (tx, mut rx) = mpsc::channel(256);
        let turn = crate::turns::TurnOver::default();
        let output = turn.within(start(tx)).await;
        let sink = HubSurfaceSink::new(self.hub.clone(), self.session.clone());
        let mut live = LiveStatus::new();
        while let Ok(update) = rx.try_recv() {
            live.note(&update);
        }
        live.flush_if_changed(&sink).await;
        let reply = reply(&output);
        if !reply.trim().is_empty() {
            let _ = self.channel.send_text(&self.session, &reply).await;
        }
        live.close(&sink).await;
        if let Some(id) = live.kept_card_message_id() {
            self.hub
                .register_checklist_surface(&self.session, &id)
                .await;
        }
        turn.end();
        output
    }

    /// The chat as the owner sees it now, oldest message first.
    pub async fn shown(&self) -> Vec<String> {
        self.channel
            .shown
            .lock()
            .await
            .iter()
            .map(|(_, text)| text.clone())
            .collect()
    }

    /// Wait (at most `limit`) until a message satisfies `found`.
    pub async fn wait_for(&self, limit: Duration, found: impl Fn(&str) -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + limit;
        while tokio::time::Instant::now() < deadline {
            if self.shown().await.iter().any(|text| found(text)) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        false
    }

    /// One background run's chat, checked against the rules every run
    /// follows: exactly one card, settled to its final header `header`,
    /// the result in its own message that never repeats the header, and no
    /// counter in text, echoed request or internal handoff text anywhere.
    pub async fn assert_one_settled_card(&self, header: &str, result: &str) {
        let shown = self.shown().await;
        let cards: Vec<&String> = shown
            .iter()
            .filter(|text| text.contains("in the background:"))
            .filter(|text| !text.starts_with("Started in the background"))
            .collect();
        assert_eq!(cards.len(), 1, "exactly one card: {shown:#?}");
        assert!(cards[0].starts_with(header), "settled header: {shown:#?}");
        let results: Vec<&String> = shown.iter().filter(|text| text.contains(result)).collect();
        assert_eq!(results.len(), 1, "one result: {shown:#?}");
        assert!(
            !results[0].contains("in the background:"),
            "the result never repeats the header: {shown:#?}"
        );
        assert!(
            results[0].contains("Measured: Took"),
            "the result carries the measured time: {shown:#?}"
        );
        for text in &shown {
            for banned in [
                "min)",
                "Still on it",
                "Still working",
                "Request:",
                "Preparing your result",
                "taking a little longer",
            ] {
                assert!(!text.contains(banned), "{banned:?} in {text:?}");
            }
        }
    }
}
