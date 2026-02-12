use std::sync::Arc;
use tracing::{debug, error, info};

use crate::config::PeopleConfig;
use crate::traits::StateStore;

/// Background tasks for people intelligence:
/// - Upcoming date reminders
/// - Stale fact pruning
/// - Reconnect suggestions
pub struct PeopleIntelligence {
    state: Arc<dyn StateStore>,
    config: PeopleConfig,
}

impl PeopleIntelligence {
    pub fn new(state: Arc<dyn StateStore>, config: PeopleConfig) -> Self {
        Self { state, config }
    }

    pub async fn run_daily_checks(&self) {
        // Check runtime setting — skip if people intelligence is disabled
        let enabled = match self.state.get_setting("people_enabled").await {
            Ok(Some(val)) => val == "true",
            _ => false,
        };
        if !enabled {
            return;
        }

        // 1. Prune stale auto-extracted facts
        match self
            .state
            .prune_stale_person_facts(self.config.fact_retention_days)
            .await
        {
            Ok(pruned) => {
                if pruned > 0 {
                    info!(
                        "People intelligence: pruned {} stale auto-extracted facts (>{} days old, unconfirmed)",
                        pruned, self.config.fact_retention_days
                    );
                }
            }
            Err(e) => {
                error!("People intelligence: failed to prune stale facts: {}", e);
            }
        }

        // 2. Check for upcoming dates (within 14 days)
        match self.state.get_people_with_upcoming_dates(14).await {
            Ok(upcoming) => {
                if !upcoming.is_empty() {
                    info!(
                        "People intelligence: {} upcoming dates within 14 days",
                        upcoming.len()
                    );
                    for (person, fact) in &upcoming {
                        debug!(
                            "People intelligence: upcoming date for {} — [{}/{}]",
                            person.name, fact.category, fact.key
                        );
                    }
                }
            }
            Err(e) => {
                error!("People intelligence: failed to check upcoming dates: {}", e);
            }
        }

        // 3. Check for reconnect suggestions
        match self
            .state
            .get_people_needing_reconnect(self.config.reconnect_reminder_days)
            .await
        {
            Ok(people) => {
                if !people.is_empty() {
                    info!(
                        "People intelligence: {} people haven't been contacted in >{} days",
                        people.len(),
                        self.config.reconnect_reminder_days
                    );
                    for person in &people {
                        debug!(
                            "People intelligence: reconnect suggestion — {}",
                            person.name
                        );
                    }
                }
            }
            Err(e) => {
                error!(
                    "People intelligence: failed to check reconnect suggestions: {}",
                    e
                );
            }
        }
    }
}
