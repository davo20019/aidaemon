use std::fmt;

use crate::config::ModelsConfig;
use crate::execution_policy::ModelProfile;

/// Tier-based model selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
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
        let mut models = models;
        if models.default_model.trim().is_empty() {
            models.default_model = models.primary.trim().to_string();
        }
        if models.fallback_models.is_empty() {
            for legacy in [&models.smart, &models.fast] {
                let candidate = legacy.trim();
                if candidate.is_empty() || candidate == models.default_model {
                    continue;
                }
                if !models.fallback_models.iter().any(|m| m == candidate) {
                    models.fallback_models.push(candidate.to_string());
                }
            }
        } else {
            let mut deduped = Vec::new();
            for raw in &models.fallback_models {
                let candidate = raw.trim();
                if candidate.is_empty() || candidate == models.default_model {
                    continue;
                }
                if !deduped.iter().any(|m: &String| m == candidate) {
                    deduped.push(candidate.to_string());
                }
            }
            models.fallback_models = deduped;
        }
        Self { models }
    }

    pub fn default_model(&self) -> &str {
        &self.models.default_model
    }

    pub fn fallback_models(&self) -> &[String] {
        &self.models.fallback_models
    }

    pub fn first_fallback(&self) -> Option<&str> {
        self.models.fallback_models.first().map(String::as_str)
    }

    pub fn all_models_ordered(&self) -> Vec<String> {
        let mut out = vec![self.models.default_model.clone()];
        for fallback in &self.models.fallback_models {
            if !out.iter().any(|m| m == fallback) {
                out.push(fallback.clone());
            }
        }
        out
    }

    pub fn select(&self, tier: Tier) -> &str {
        match tier {
            Tier::Fast => self.first_fallback().unwrap_or(&self.models.default_model),
            Tier::Primary => &self.models.default_model,
            Tier::Smart => &self.models.default_model,
        }
    }

    /// Profile mapping on top of default+fallback:
    /// - Cheap -> first fallback when available (cost-oriented)
    /// - Balanced/Strong -> default model
    pub fn select_for_profile(&self, profile: ModelProfile) -> &str {
        match profile {
            ModelProfile::Cheap => self.first_fallback().unwrap_or(&self.models.default_model),
            ModelProfile::Balanced | ModelProfile::Strong => &self.models.default_model,
        }
    }

    /// Returns true when there are no distinct fallback models.
    pub fn is_uniform(&self) -> bool {
        self.models
            .fallback_models
            .iter()
            .all(|m| m == &self.models.default_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_uniform_true() {
        let models = ModelsConfig {
            default_model: "gpt-4o".to_string(),
            fallback_models: Vec::new(),
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
            default_model: "gpt-4o".to_string(),
            fallback_models: vec!["gpt-4o-mini".to_string()],
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
            default_model: "primary-model".to_string(),
            fallback_models: vec!["fast-model".to_string(), "backup-model".to_string()],
            primary: "primary-model".to_string(),
            fast: "fast-model".to_string(),
            smart: "smart-model".to_string(),
        };
        let router = Router::new(models);
        assert_eq!(router.select(Tier::Fast), "fast-model");
        assert_eq!(router.select(Tier::Primary), "primary-model");
        assert_eq!(router.select(Tier::Smart), "primary-model");
        assert_eq!(router.select_for_profile(ModelProfile::Cheap), "fast-model");
        assert_eq!(
            router.select_for_profile(ModelProfile::Balanced),
            "primary-model"
        );
        assert_eq!(
            router.select_for_profile(ModelProfile::Strong),
            "primary-model"
        );
        assert_eq!(
            router.all_models_ordered(),
            vec![
                "primary-model".to_string(),
                "fast-model".to_string(),
                "backup-model".to_string()
            ]
        );
    }
}
