//! Capability-floor checks for vision+tools computer-use loops.

use crate::config::{ProviderKind, VisionConfig};

const DEFAULT_VISION_MODEL_PATTERNS: &[&str] = &[
    "gpt-4o",
    "gpt-4",
    "gpt-5",
    "o1",
    "o3",
    "o4",
    "gemini",
    "claude-3",
    "claude-sonnet",
    "claude-opus",
    "claude-haiku",
    "fable",
    "gemma",
    "llava",
    "qwen-vl",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "internvl",
    "vision",
    "pixtral",
    "mistral-large",
];

/// The GUI loop speaks the OpenAI-style multimodal wire format (image_url
/// content blocks; tool screenshots re-enter as user-role observations).
/// Providers qualify when their adapter consumes that format: the
/// openai_compatible adapter natively, and google_genai via its
/// `openai_content_to_gemini_parts` translation (image_url → inlineData).
/// Anthropic/xAI native adapters have no verified translation yet.
pub fn provider_supports_computer_use(kind: ProviderKind) -> bool {
    matches!(
        kind,
        ProviderKind::OpenaiCompatible | ProviderKind::GoogleGenai
    )
}

pub fn model_supports_computer_use_vision(model: &str, vision: &VisionConfig) -> bool {
    if vision.model_patterns.is_empty() {
        let model_lower = model.to_ascii_lowercase();
        return DEFAULT_VISION_MODEL_PATTERNS
            .iter()
            .any(|pattern| model_lower.contains(&pattern.to_ascii_lowercase()));
    }
    vision.model_supports_vision(model)
}

pub fn model_meets_computer_use_floor(
    model: &str,
    vision: &VisionConfig,
    provider_kind: ProviderKind,
) -> bool {
    provider_supports_computer_use(provider_kind)
        && model_supports_computer_use_vision(model, vision)
}

/// Pick the first model in `chain` that meets the computer-use capability floor.
pub fn pick_capable_model(
    chain: &[String],
    vision: &VisionConfig,
    provider_kind: ProviderKind,
) -> Result<String, String> {
    if !provider_supports_computer_use(provider_kind) {
        return Err(
            "computer_use requires a provider whose adapter speaks the OpenAI-style \
             multimodal wire (openai_compatible or google_genai); native Anthropic/xAI \
             adapters are not supported for GUI loops yet"
                .to_string(),
        );
    }
    if !vision.enabled {
        return Err(
            "Vision is disabled in config — enable [files] vision_enabled for computer_use"
                .to_string(),
        );
    }
    for model in chain {
        if model_meets_computer_use_floor(model, vision, provider_kind) {
            return Ok(model.clone());
        }
    }
    Err(format!(
        "No model in the configured chain supports computer_use (vision+tools+OpenAI wire). \
         Chain tried: {}. Add a vision-capable model to default_model/fallback_models \
         or configure [files] vision_model_patterns.",
        chain.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FilesConfig;

    fn vision() -> VisionConfig {
        VisionConfig::from_files(&FilesConfig {
            vision_enabled: true,
            ..FilesConfig::default()
        })
    }

    #[test]
    fn gpt4o_meets_floor_on_openai_compatible() {
        assert!(model_meets_computer_use_floor(
            "gpt-4o",
            &vision(),
            ProviderKind::OpenaiCompatible
        ));
    }

    #[test]
    fn google_genai_gemini_meets_floor() {
        // Live 2026-07-12: with gemini-3.5-flash as primary, computer_use
        // refused with "requires an OpenAI-compatible multimodal provider" and
        // the agent fell back to brittle AppleScript. The google_genai adapter
        // consumes the same OpenAI-style multimodal wire (image_url →
        // inlineData) and tool screenshots re-enter as user-role observations,
        // so the GUI loop's requirements are met.
        assert!(model_meets_computer_use_floor(
            "gemini-3.5-flash",
            &vision(),
            ProviderKind::GoogleGenai
        ));
        let picked = pick_capable_model(
            &["gemini-3.5-flash".to_string()],
            &vision(),
            ProviderKind::GoogleGenai,
        )
        .unwrap();
        assert_eq!(picked, "gemini-3.5-flash");
    }

    #[test]
    fn anthropic_provider_fails_floor() {
        assert!(!model_meets_computer_use_floor(
            "gpt-4o",
            &vision(),
            ProviderKind::Anthropic
        ));
    }

    #[test]
    fn pick_first_capable_from_chain() {
        let chain = vec!["text-only-mini".to_string(), "gpt-4o-mini".to_string()];
        let picked = pick_capable_model(&chain, &vision(), ProviderKind::OpenaiCompatible).unwrap();
        assert_eq!(picked, "gpt-4o-mini");
    }
}
