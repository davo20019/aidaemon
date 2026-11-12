//! Health probes live in `aidaemon_work::health`. The daemon keeps its old
//! paths to them and delivers their alerts through the channel hub.

pub use aidaemon_work::health::*;

use crate::channels::ChannelHub;

#[async_trait::async_trait]
impl AlertDelivery for ChannelHub {
    async fn broadcast_text(&self, session_ids: &[String], text: &str) {
        ChannelHub::broadcast_text(self, session_ids, text).await;
    }
}
