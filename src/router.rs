use std::fmt;

use crate::config::ModelsConfig;
use crate::execution_policy::ModelProfile;

/// Tier-based model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Fast,
    Primary,
    Smart,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Fast => write!(f, "fast"),
            Tier::Primary => write!(f, "primary"),
            Tier::Smart => write!(f, "smart"),
        }
    }
}

#[derive(Clone)]
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

    /// Thin profile-to-model mapping used by policy-driven routing.
    /// Cheap -> fast, Balanced -> primary, Strong -> smart.
    pub fn select_for_profile(&self, profile: ModelProfile) -> &str {
        match profile {
            ModelProfile::Cheap => &self.models.fast,
            ModelProfile::Balanced => &self.models.primary,
            ModelProfile::Strong => &self.models.smart,
        }
    }

    /// Returns true when all three tier models resolve to the same string,
    /// meaning auto-routing would be pointless.
    pub fn is_uniform(&self) -> bool {
        self.models.fast == self.models.primary && self.models.primary == self.models.smart
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_uniform_true() {
        let models = ModelsConfig {
            primary: "gpt-4o".to_string(),
            fast: "gpt-4o".to_string(),
            smart: "gpt-4o".to_string(),
        };
        let router = Router::new(models);
        assert!(router.is_uniform());
    }

    #[test]
    fn test_is_uniform_false() {
        let models = ModelsConfig {
            primary: "gpt-4o".to_string(),
            fast: "gpt-4o-mini".to_string(),
            smart: "gpt-4o".to_string(),
        };
        let router = Router::new(models);
        assert!(!router.is_uniform());
    }

    #[test]
    fn test_display_tier() {
        assert_eq!(Tier::Fast.to_string(), "fast");
        assert_eq!(Tier::Primary.to_string(), "primary");
        assert_eq!(Tier::Smart.to_string(), "smart");
    }

    #[test]
    fn test_select_for_profile() {
        let models = ModelsConfig {
            primary: "primary-model".to_string(),
            fast: "fast-model".to_string(),
            smart: "smart-model".to_string(),
        };
        let router = Router::new(models);
        assert_eq!(router.select(Tier::Fast), "fast-model");
        assert_eq!(router.select(Tier::Primary), "primary-model");
        assert_eq!(router.select(Tier::Smart), "smart-model");
        assert_eq!(router.select_for_profile(ModelProfile::Cheap), "fast-model");
        assert_eq!(
            router.select_for_profile(ModelProfile::Balanced),
            "primary-model"
        );
        assert_eq!(
            router.select_for_profile(ModelProfile::Strong),
            "smart-model"
        );
    }
}
