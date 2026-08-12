use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use sha2::{Digest, Sha256};

use crate::config::NodesConfig;

use super::speech::NodeSpeechSynthesizer;
use super::store::{NodeOutboxReceipt, NodeStore, QueuedAudioAnnouncement};

#[derive(Debug, Clone)]
pub struct AnnouncementDelivery {
    pub queued: QueuedAudioAnnouncement,
    pub receipt: Option<NodeOutboxReceipt>,
}

#[derive(Clone)]
pub struct NodeAnnouncementService {
    store: Arc<NodeStore>,
    config: NodesConfig,
    speech: Arc<dyn NodeSpeechSynthesizer>,
}

impl NodeAnnouncementService {
    pub fn new(
        store: Arc<NodeStore>,
        config: NodesConfig,
        speech: Arc<dyn NodeSpeechSynthesizer>,
    ) -> Self {
        Self {
            store,
            config,
            speech,
        }
    }

    pub async fn queue_and_wait(
        &self,
        selector: Option<&str>,
        text: &str,
    ) -> anyhow::Result<AnnouncementDelivery> {
        anyhow::ensure!(
            self.config.announcements.enabled,
            "Node audio announcements are disabled"
        );
        anyhow::ensure!(
            self.config.speech.enabled,
            "Node speech synthesis is disabled"
        );
        let text = text.trim();
        anyhow::ensure!(!text.is_empty(), "announcement text is empty");
        anyhow::ensure!(
            text.chars().count() <= self.config.announcements.max_text_chars,
            "announcement text exceeds the configured limit"
        );

        self.cleanup_media(None).await;
        let target = self
            .store
            .resolve_audio_announcement_target(selector)
            .await?;
        let maximum_audio_bytes = target
            .maximum_audio_bytes
            .map(|limit| limit as usize)
            .unwrap_or(self.config.announcements.max_audio_bytes)
            .min(self.config.announcements.max_audio_bytes);
        anyhow::ensure!(
            maximum_audio_bytes > 0,
            "Node reported an invalid audio limit"
        );

        let output_dir = std::path::PathBuf::from(
            shellexpand::tilde(&self.config.retention.media_dir).into_owned(),
        )
        .join("announcements");
        let artifact = self
            .speech
            .synthesize(text, &output_dir, maximum_audio_bytes)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Node speech synthesis did not produce audio"))?;
        let result = async {
            anyhow::ensure!(
                artifact.content_type == "audio/wav",
                "Node announcements require audio/wav"
            );
            let bytes = tokio::fs::read(&artifact.path).await?;
            anyhow::ensure!(
                bytes.len() as u64 == artifact.size_bytes,
                "synthesized announcement size changed"
            );
            anyhow::ensure!(
                !bytes.is_empty() && bytes.len() <= maximum_audio_bytes,
                "synthesized announcement exceeds the target limit"
            );
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            self.store
                .queue_audio_announcement(
                    &target,
                    &artifact.content_type,
                    artifact.size_bytes,
                    &sha256,
                    &artifact.path.to_string_lossy(),
                    self.config.announcements.ttl_seconds,
                    self.config.announcements.max_pending_per_node,
                )
                .await
        }
        .await;
        let queued = match result {
            Ok(queued) => queued,
            Err(error) => {
                let _ = tokio::fs::remove_file(&artifact.path).await;
                return Err(error);
            }
        };

        let receipt = self.wait_for_receipt(&queued).await?;
        Ok(AnnouncementDelivery { queued, receipt })
    }

    async fn wait_for_receipt(
        &self,
        queued: &QueuedAudioAnnouncement,
    ) -> anyhow::Result<Option<NodeOutboxReceipt>> {
        if self.config.announcements.ack_wait_seconds == 0 {
            return Ok(None);
        }
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs(self.config.announcements.ack_wait_seconds);
        loop {
            if let Some(receipt) = self
                .store
                .node_outbox_receipt(&queued.node_id, queued.cursor)
                .await?
            {
                return Ok(Some(receipt));
            }
            if tokio::time::Instant::now() >= deadline || Utc::now() >= queued.expires_at {
                return Ok(None);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn cleanup_media(&self, node_id: Option<&str>) {
        let Ok(candidates) = self.store.outbound_media_cleanup_candidates(node_id).await else {
            return;
        };
        for candidate in candidates {
            match tokio::fs::remove_file(&candidate.local_path).await {
                Ok(())
                    if self
                        .store
                        .mark_outbound_media_deleted(&candidate.media_id)
                        .await
                        .is_err() => {}
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let _ = self
                        .store
                        .mark_outbound_media_deleted(&candidate.media_id)
                        .await;
                }
                Err(_) => {}
            }
        }
    }
}
