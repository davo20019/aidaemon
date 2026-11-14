//! People intelligence: daily upkeep of the people memory.

use std::sync::Arc;

use tracing::{debug, error, info};

use crate::stores::PeopleStore;

/// Background tasks for people intelligence:
/// - Upcoming date reminders
/// - Stale fact pruning
/// - Reconnect suggestions
pub struct PeopleIntelligence {
    state: Arc<dyn PeopleStore>,
    config: PeopleIntelligenceConfig,
}

/// How long unconfirmed auto-extracted person facts live, and how long
/// without contact before a reconnect suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeopleIntelligenceConfig {
    pub fact_retention_days: u32,
    pub reconnect_reminder_days: u32,
}

impl PeopleIntelligence {
    pub fn new(state: Arc<dyn PeopleStore>, config: PeopleIntelligenceConfig) -> Self {
        Self { state, config }
    }

    /// Prune stale facts and log upcoming dates and reconnect suggestions.
    /// The host decides whether people intelligence is enabled.
    pub async fn run_daily_checks(&self) {
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
