use crate::config::ModelsConfig;

/// Tier-based model selection.
#[derive(Debug, Clone, Copy)]
pub enum Tier {
    Fast,
    Primary,
    Smart,
}

pub struct Router {
    models: ModelsConfig,
}

impl Router {
    pub fn new(models: ModelsConfig) -> Self {
        Self { models }
    }

    pub fn select(&self, tier: Tier) -> &str {
        match tier {
            Tier::Fast => &self.models.fast,
            Tier::Primary => &self.models.primary,
            Tier::Smart => &self.models.smart,
        }
    }
}
