//! Health probes live in `aidaemon_work::health`. The daemon keeps its old
//! paths to them and delivers their alerts through the channel hub.

pub use aidaemon_work::health::*;

use std::sync::Arc;

use crate::channels::ChannelHub;

/// Delivers health alerts through the daemon's channel hub.
pub struct HubAlerts(pub Arc<ChannelHub>);

#[async_trait::async_trait]
impl AlertDelivery for HubAlerts {
    async fn broadcast_text(&self, session_ids: &[String], text: &str) {
        self.0.broadcast_text(session_ids, text).await;
    }
}
