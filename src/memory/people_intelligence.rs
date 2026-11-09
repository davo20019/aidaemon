//! People intelligence: the SDK memory crate's daily upkeep, gated by the
//! daemon's runtime setting.

use std::sync::Arc;

use crate::config::PeopleConfig;
use crate::traits::StateStore;

pub struct PeopleIntelligence {
    state: Arc<dyn StateStore>,
    upkeep: aidaemon_memory::people_intelligence::PeopleIntelligence,
}

impl PeopleIntelligence {
    pub fn new(
        state: Arc<dyn StateStore>,
        store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
        config: PeopleConfig,
    ) -> Self {
        let upkeep = aidaemon_memory::people_intelligence::PeopleIntelligence::new(
            store,
            aidaemon_memory::people_intelligence::PeopleIntelligenceConfig {
                fact_retention_days: config.fact_retention_days,
                reconnect_reminder_days: config.reconnect_reminder_days,
            },
        );
        Self { state, upkeep }
    }

    /// Runs only while the owner has people intelligence enabled
    /// (`people_enabled` setting).
    pub async fn run_daily_checks(&self) {
        let enabled = matches!(
            self.state.get_setting("people_enabled").await,
            Ok(Some(value)) if value == "true"
        );
        if enabled {
            self.upkeep.run_daily_checks().await;
        }
    }
}
