//! Model routing: which provider family a deployment uses, its configured
//! models, and the tier-based [`Router`] that picks a model for a request.

use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    #[default]
    OpenaiCompatible,
    XaiNative,
    GoogleGenai,
    Anthropic,
    /// ChatGPT Plus/Pro/Business subscription via Codex OAuth. Uses stored
    /// OAuth credentials rather than `api_key`.
    OpenaiChatgpt,
}

impl ProviderKind {
    /// The provider family as the owner knows it.
    pub fn plain_name(self) -> &'static str {
        match self {
            Self::OpenaiCompatible => "an OpenAI-compatible provider",
            Self::XaiNative => "xAI",
            Self::GoogleGenai => "Google Gemini",
            Self::Anthropic => "Anthropic",
            Self::OpenaiChatgpt => "your ChatGPT subscription",
        }
    }
}

/// What a provider reload changed, said in plain words.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderReload {
    pub old_provider: ProviderKind,
    pub new_provider: ProviderKind,
    pub old_model: String,
    pub new_model: String,
    /// A model the owner had picked by hand was dropped for automatic choice.
    pub manual_model_cleared: bool,
}

impl ProviderReload {
    /// Only what changed; "Nothing else changed." when nothing did.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.old_provider != self.new_provider {
            parts.push(format!(
                "Now using {} ({}).",
                self.new_provider.plain_name(),
                self.new_model
            ));
        } else if self.old_model != self.new_model {
            parts.push(format!(
                "The model is now {} (was {}).",
                self.new_model, self.old_model
            ));
        }
        if self.manual_model_cleared {
            parts.push("The model is picked automatically again.".to_string());
        }
        if parts.is_empty() {
            "Nothing else changed.".to_string()
        } else {
            parts.join(" ")
        }
    }
}

#[cfg(test)]
mod reload_summary_tests {
    use super::*;

    fn reload(old: ProviderKind, new: ProviderKind, models: (&str, &str)) -> ProviderReload {
        ProviderReload {
            old_provider: old,
            new_provider: new,
            old_model: models.0.to_string(),
            new_model: models.1.to_string(),
            manual_model_cleared: false,
        }
    }

    #[test]
    fn a_reload_says_only_what_changed() {
        let same = reload(
            ProviderKind::OpenaiChatgpt,
            ProviderKind::OpenaiChatgpt,
            ("gpt-6-sol", "gpt-6-sol"),
        );
        assert_eq!(same.summary(), "Nothing else changed.");
        let switched = reload(
            ProviderKind::OpenaiCompatible,
            ProviderKind::Anthropic,
            ("mock-model", "claude-x"),
        );
        assert_eq!(switched.summary(), "Now using Anthropic (claude-x).");
        let mut model_only = reload(
            ProviderKind::Anthropic,
            ProviderKind::Anthropic,
            ("claude-a", "claude-b"),
        );
        model_only.manual_model_cleared = true;
        assert_eq!(
            model_only.summary(),
            "The model is now claude-b (was claude-a). The model is picked automatically again."
        );
        assert!(!switched.summary().contains("Openai"));
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ModelsConfig {
    /// Canonical default model used for normal requests.
    /// TOML key: `provider.models.default`
    #[serde(default, rename = "default")]
    pub default_model: String,
    /// Ordered fallback models used only when calls fail.
    /// TOML key: `provider.models.fallback = [\"...\"]`
    #[serde(default, rename = "fallback")]
    pub fallback_models: Vec<String>,
    /// Legacy key kept for backward compatibility.
    #[serde(default)]
    pub primary: String,
    /// Legacy key kept for backward compatibility.
    #[serde(default)]
    pub fast: String,
    /// Legacy key kept for backward compatibility.
    #[serde(default)]
    pub smart: String,
}

impl ModelsConfig {
    /// Normalize legacy and new model keys into:
    /// - `default_model`
    /// - `fallback_models` (ordered, deduped, excluding default)
    ///
    /// Legacy fields (`primary`, `fast`, `smart`) remain populated for
    /// compatibility with older code paths and existing config tooling.
    pub fn apply_defaults(&mut self, provider_kind: &ProviderKind) {
        let provider_default = match provider_kind {
            ProviderKind::GoogleGenai => "gemini-3-flash-preview".to_string(),
            ProviderKind::Anthropic => "claude-sonnet-4-20250514".to_string(),
            ProviderKind::XaiNative => "grok-4".to_string(),
            ProviderKind::OpenaiCompatible => "openai/gpt-4o".to_string(),
            ProviderKind::OpenaiChatgpt => "gpt-5.6-terra".to_string(),
        };

        let default_model = if !self.default_model.trim().is_empty() {
            self.default_model.trim().to_string()
        } else if !self.primary.trim().is_empty() {
            self.primary.trim().to_string()
        } else {
            provider_default
        };

        // If fallback[] was not explicitly provided, derive it from legacy tier keys.
        if self.fallback_models.is_empty() {
            for legacy in [&self.smart, &self.fast] {
                let candidate = legacy.trim();
                if candidate.is_empty() || candidate == default_model {
                    continue;
                }
                if !self.fallback_models.iter().any(|m| m == candidate) {
                    self.fallback_models.push(candidate.to_string());
                }
            }
        } else {
            // Sanitize explicit fallback[].
            let mut deduped = Vec::new();
            for raw in &self.fallback_models {
                let candidate = raw.trim();
                if candidate.is_empty() || candidate == default_model {
                    continue;
                }
                if !deduped.iter().any(|m: &String| m == candidate) {
                    deduped.push(candidate.to_string());
                }
            }
            self.fallback_models = deduped;
        }

        self.default_model = default_model.clone();
        self.primary = default_model.clone();

        let first_fallback = self
            .fallback_models
            .first()
            .cloned()
            .unwrap_or_else(|| default_model.clone());
        if self.fast.trim().is_empty() {
            self.fast = first_fallback.clone();
        }
        if self.smart.trim().is_empty() {
            self.smart = first_fallback;
        }
    }
}

/// How strong a model a request needs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelProfile {
    Cheap,
    Balanced,
    Strong,
}

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
            Tier::Fast => nonempty_or(&self.models.fast, &self.models.default_model),
            Tier::Primary => &self.models.default_model,
            Tier::Smart => nonempty_or(&self.models.smart, &self.models.default_model),
        }
    }

    /// Profile mapping on top of default+fallback:
    /// Cheap uses the configured fast model, Balanced uses the canonical
    /// default, and Strong uses the configured smart model. Deployments that
    /// prefer one model can keep all three keys equal; deployments that provide
    /// distinct models now receive real cost/capability adaptation.
    pub fn select_for_profile(&self, profile: ModelProfile) -> &str {
        match profile {
            ModelProfile::Cheap => self.select(Tier::Fast),
            ModelProfile::Balanced => self.select(Tier::Primary),
            ModelProfile::Strong => self.select(Tier::Smart),
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

fn nonempty_or<'a>(candidate: &'a str, fallback: &'a str) -> &'a str {
    if candidate.trim().is_empty() {
        fallback
    } else {
        candidate
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
