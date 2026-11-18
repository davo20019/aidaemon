//! `generate_image` lives in `aidaemon_runtime::tools::image_generation`;
//! the live tests on the daemon's stored ChatGPT login stay here.

pub use aidaemon_runtime::tools::image_generation::GenerateImageTool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::chatgpt_login as chatgpt_codex;
    use crate::traits::ToolCalls as _;
    use aidaemon_runtime::tools::image_generation::{
        ChatGptSubscriptionImageBackend, ImageGenerationBackend, ImageGenerationReference,
        ImageGenerationRequest,
    };
    use aidaemon_tools::semantics::ToolOutcomeStatus;
    use aidaemon_tools::sniff_image_mime;
    use serde_json::json;
    use std::path::PathBuf;

    #[tokio::test]
    #[ignore = "uses stored ChatGPT subscription credentials and consumes image usage"]
    async fn live_chatgpt_subscription_generation() {
        assert_eq!(
            std::env::var("AIDAEMON_LIVE_TEST").as_deref(),
            Ok("1"),
            "set AIDAEMON_LIVE_TEST=1 to authorize the live request"
        );
        assert!(
            chatgpt_codex::is_connected(),
            "run `aidaemon auth login openai` before the live test"
        );
        let output_dir = std::env::var_os("AIDAEMON_LIVE_IMAGE_DIR")
            .map(PathBuf::from)
            .expect("set AIDAEMON_LIVE_IMAGE_DIR to an explicit artifact directory");
        let tool = GenerateImageTool::chatgpt_subscription(
            output_dir,
            chatgpt_codex::shared_credential_manager(),
        )
        .expect("construct ChatGPT subscription image tool");

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "prompt": "A clean test card: one cobalt-blue circle centered on a warm white background, flat vector style, crisp edges, no shadows, no text"
                })
                .to_string(),
                None,
            )
            .await
            .expect("execute live image generation");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded),
            "live generation failed: {}",
            outcome.output
        );
        let attachment = outcome
            .metadata
            .attachments
            .first()
            .expect("live result should include an image attachment");
        assert!(
            attachment.size_bytes > 1_024,
            "live image is implausibly small: {} bytes",
            attachment.size_bytes
        );
        assert_eq!(attachment.mime_type, "image/png");
        eprintln!("live_image_path={}", attachment.local_path);
        eprintln!("live_image_bytes={}", attachment.size_bytes);
    }

    #[tokio::test]
    #[ignore = "uses stored ChatGPT subscription credentials, a local reference image, and consumes image usage"]
    async fn live_chatgpt_subscription_reference_edit() {
        assert_eq!(
            std::env::var("AIDAEMON_LIVE_TEST").as_deref(),
            Ok("1"),
            "set AIDAEMON_LIVE_TEST=1 to authorize the live request"
        );
        assert!(
            chatgpt_codex::is_connected(),
            "run `aidaemon auth login openai` before the live test"
        );
        let reference_path = std::env::var_os("AIDAEMON_LIVE_REFERENCE_IMAGE")
            .map(PathBuf::from)
            .expect("set AIDAEMON_LIVE_REFERENCE_IMAGE to an explicit PNG, JPEG, or WebP path");
        let output_dir = std::env::var_os("AIDAEMON_LIVE_IMAGE_DIR")
            .map(PathBuf::from)
            .expect("set AIDAEMON_LIVE_IMAGE_DIR to an explicit artifact directory");
        std::fs::create_dir_all(&output_dir).expect("create live image output directory");

        let bytes = std::fs::read(&reference_path).expect("read live reference image");
        let mime_type = match sniff_image_mime(&bytes) {
            Some("image/png") => "image/png",
            Some("image/jpeg") => "image/jpeg",
            Some("image/webp") => "image/webp",
            other => panic!("unsupported live reference image type: {other:?}"),
        };
        let filename = reference_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("reference-image")
            .to_string();
        let backend =
            ChatGptSubscriptionImageBackend::new(None, chatgpt_codex::shared_credential_manager())
                .expect("construct ChatGPT subscription image backend");
        let generated = backend
            .generate(&ImageGenerationRequest {
                prompt: "Create a clean, neutral portrait card using the supplied photo as the identity reference. Preserve the person's face, facial proportions, skin tone, eyes, nose, smile, hairline, and hair texture faithfully. No text.".to_string(),
                references: vec![ImageGenerationReference {
                    bytes,
                    filename,
                    mime_type: mime_type.to_string(),
                }],
                turn_id: Some(format!("live-reference-edit-{}", uuid::Uuid::new_v4())),
            })
            .await
            .expect("execute live reference edit");

        assert!(
            generated.bytes.len() > 1_024,
            "live reference edit is implausibly small: {} bytes",
            generated.bytes.len()
        );
        assert!(
            sniff_image_mime(&generated.bytes).is_some(),
            "live reference edit is not a supported image"
        );
        let output_path = output_dir.join("live_reference_edit.png");
        std::fs::write(&output_path, &generated.bytes).expect("write live reference edit");
        eprintln!("live_reference_edit_path={}", output_path.display());
        eprintln!("live_reference_edit_bytes={}", generated.bytes.len());
    }
}
