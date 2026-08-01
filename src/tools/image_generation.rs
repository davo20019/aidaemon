//! Provider-neutral image generation tool and provider adapters.
//!
//! The agent sees one stable `generate_image` contract. Backends translate that
//! request into each provider's API, so image generation is independent of the
//! provider handling the surrounding conversation.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::channels::attachments::{save_tool_observation_image, sniff_image_mime};
use crate::oauth::chatgpt_codex::{self, ChatGptCredentialManager, CODEX_BACKEND_BASE_URL};
use crate::traits::{
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolMutationEffects, ToolOutcomeStatus, ToolVerificationMode,
};
use crate::types::StatusUpdate;

const CHATGPT_IMAGE_MODEL: &str = "gpt-image-2";
const CHATGPT_IMAGE_BACKEND: &str = "chatgpt_subscription";
const ORIGINATOR: &str = "aidaemon";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Provider-independent request passed to an image generation backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    /// Correlates a request with the agent turn when the provider supports it.
    pub turn_id: Option<String>,
}

/// Provider-independent result returned to the tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
    pub model: Option<String>,
    pub dimensions: Option<String>,
}

/// How the tool should classify a backend failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGenerationErrorKind {
    Retryable,
    Permanent,
    Blocked,
}

/// Typed backend error that remains meaningful across providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationError {
    pub kind: ImageGenerationErrorKind,
    pub message: String,
    pub http_status: Option<u16>,
}

impl ImageGenerationError {
    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            kind: ImageGenerationErrorKind::Retryable,
            message: message.into(),
            http_status: None,
        }
    }

    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            kind: ImageGenerationErrorKind::Permanent,
            message: message.into(),
            http_status: None,
        }
    }

    pub fn blocked(message: impl Into<String>) -> Self {
        Self {
            kind: ImageGenerationErrorKind::Blocked,
            message: message.into(),
            http_status: None,
        }
    }

    pub fn with_http_status(mut self, status: u16) -> Self {
        self.http_status = Some(status);
        self
    }

    fn outcome_status(&self) -> ToolOutcomeStatus {
        match self.kind {
            ImageGenerationErrorKind::Retryable => ToolOutcomeStatus::FailedRetryable,
            ImageGenerationErrorKind::Permanent => ToolOutcomeStatus::FailedPermanent,
            ImageGenerationErrorKind::Blocked => ToolOutcomeStatus::Blocked,
        }
    }
}

impl fmt::Display for ImageGenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ImageGenerationError {}

/// Adapter interface for provider-specific image APIs.
///
/// New providers implement this trait; the tool schema, attachment handling,
/// delivery flow, and agent behavior do not need to change.
#[async_trait]
pub trait ImageGenerationBackend: Send + Sync {
    fn name(&self) -> &str;

    async fn generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<GeneratedImage, ImageGenerationError>;
}

/// Image adapter backed by the user's ChatGPT subscription OAuth login.
pub struct ChatGptSubscriptionImageBackend {
    client: reqwest::Client,
    base_url: String,
    credentials: Arc<ChatGptCredentialManager>,
}

impl ChatGptSubscriptionImageBackend {
    pub fn new(base_url: Option<&str>) -> Result<Self, String> {
        Ok(Self {
            client: crate::providers::build_http_client(DEFAULT_TIMEOUT)?,
            base_url: normalize_base_url(base_url),
            credentials: chatgpt_codex::shared_credential_manager(),
        })
    }

    #[cfg(test)]
    fn with_dependencies(
        base_url: &str,
        client: reqwest::Client,
        credentials: Arc<ChatGptCredentialManager>,
    ) -> Self {
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            credentials,
        }
    }
}

fn normalize_base_url(base_url: Option<&str>) -> String {
    base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(CODEX_BACKEND_BASE_URL)
        .trim_end_matches('/')
        .to_string()
}

#[derive(Debug, Deserialize)]
struct ChatGptImageResponse {
    #[serde(default)]
    data: Vec<ChatGptImageData>,
    #[serde(default)]
    size: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatGptImageData {
    b64_json: String,
}

#[async_trait]
impl ImageGenerationBackend for ChatGptSubscriptionImageBackend {
    fn name(&self) -> &str {
        CHATGPT_IMAGE_BACKEND
    }

    async fn generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<GeneratedImage, ImageGenerationError> {
        let credentials = self
            .credentials
            .usable_credentials(&self.client)
            .await
            .map_err(|error| ImageGenerationError::blocked(error.to_string()))?;

        let url = format!("{}/images/generations", self.base_url);
        let body = json!({
            "prompt": request.prompt,
            "background": "auto",
            "model": CHATGPT_IMAGE_MODEL,
            "quality": "auto",
            "size": "auto"
        });
        let mut request_builder = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {}", credentials.access_token),
            )
            .header("chatgpt-account-id", credentials.account_id)
            .header("originator", ORIGINATOR)
            .header("User-Agent", image_user_agent())
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .json(&body);
        if let Some(turn_id) = request
            .turn_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            request_builder = request_builder.header("x-codex-image-turn-id", turn_id);
        }

        let response = request_builder.send().await.map_err(|error| {
            ImageGenerationError::retryable(format!(
                "ChatGPT image request could not be sent: {error}"
            ))
        })?;
        let status = response.status();
        let response_body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(chatgpt_image_http_error(status.as_u16(), &response_body));
        }

        let parsed: ChatGptImageResponse =
            serde_json::from_str(&response_body).map_err(|error| {
                ImageGenerationError::retryable(format!(
                    "ChatGPT image generation returned unexpected JSON: {error}"
                ))
            })?;
        let encoded = parsed
            .data
            .into_iter()
            .next()
            .map(|data| data.b64_json)
            .ok_or_else(|| {
                ImageGenerationError::permanent("ChatGPT image generation returned no image data")
            })?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded.trim().as_bytes())
            .map_err(|error| {
                ImageGenerationError::permanent(format!(
                    "ChatGPT image generation returned invalid base64: {error}"
                ))
            })?;

        Ok(GeneratedImage {
            bytes,
            mime_type: Some("image/png".to_string()),
            model: Some(CHATGPT_IMAGE_MODEL.to_string()),
            dimensions: parsed.size,
        })
    }
}

fn image_user_agent() -> String {
    format!(
        "aidaemon/{} ({}; {})",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

fn chatgpt_image_http_error(status: u16, body: &str) -> ImageGenerationError {
    let details = truncate_error_body(body);
    let error = match status {
        401 => ImageGenerationError::blocked(format!(
            "ChatGPT subscription auth rejected. Re-connect with `aidaemon auth login openai`. \
             Details: {details}"
        )),
        403 => ImageGenerationError::blocked(format!(
            "ChatGPT image generation is unavailable for this account or workspace. \
             Details: {details}"
        )),
        429 => ImageGenerationError::retryable(format!(
            "ChatGPT image generation usage limit reached or temporarily throttled. \
             Details: {details}"
        )),
        500..=599 => ImageGenerationError::retryable(format!(
            "ChatGPT image generation is temporarily unavailable. Details: {details}"
        )),
        _ => ImageGenerationError::permanent(format!(
            "ChatGPT image generation failed with HTTP {status}. Details: {details}"
        )),
    };
    error.with_http_status(status)
}

fn truncate_error_body(body: &str) -> String {
    const LIMIT: usize = 500;
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "no response body".to_string();
    }
    if trimmed.chars().count() <= LIMIT {
        return trimmed.to_string();
    }
    let prefix: String = trimmed.chars().take(LIMIT).collect();
    format!("{prefix}…")
}

/// Stable agent-facing image generation tool.
pub struct GenerateImageTool {
    backend: Arc<dyn ImageGenerationBackend>,
    inbox_dir: PathBuf,
}

impl GenerateImageTool {
    pub fn new(backend: Arc<dyn ImageGenerationBackend>, inbox_dir: PathBuf) -> Self {
        Self { backend, inbox_dir }
    }

    pub fn chatgpt_subscription(inbox_dir: PathBuf) -> Result<Self, String> {
        Ok(Self::new(
            Arc::new(ChatGptSubscriptionImageBackend::new(None)?),
            inbox_dir,
        ))
    }

    async fn generate_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Value = serde_json::from_str(arguments)?;
        let prompt = args
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(prompt) = prompt else {
            return Ok(error_outcome(
                ImageGenerationError::permanent("Missing required parameter: prompt"),
                self.call_semantics(arguments),
            ));
        };

        if let Some(status_tx) = status_tx {
            let _ = status_tx
                .send(StatusUpdate::ToolProgress {
                    name: self.name().to_string(),
                    chunk: "Generating image…".to_string(),
                })
                .await;
        }

        let request = ImageGenerationRequest {
            prompt: prompt.to_string(),
            turn_id: args
                .get("_turn_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
        };
        let generated = match self.backend.generate(&request).await {
            Ok(generated) => generated,
            Err(error) => {
                return Ok(error_outcome(error, self.call_semantics(arguments)));
            }
        };

        let detected_mime = match sniff_image_mime(&generated.bytes) {
            Some(mime_type) => mime_type,
            None => {
                return Ok(error_outcome(
                    ImageGenerationError::permanent(
                        "Image provider returned data that is not a supported PNG, JPEG, GIF, or WebP image",
                    ),
                    self.call_semantics(arguments),
                ));
            }
        };
        let filename = format!("generated_image.{}", extension_for_mime(detected_mime));
        let attachment = match save_tool_observation_image(
            &self.inbox_dir,
            &generated.bytes,
            &filename,
            detected_mime,
            self.name(),
        ) {
            Ok(attachment) => attachment,
            Err(error) => {
                return Ok(error_outcome(
                    ImageGenerationError::permanent(format!(
                        "Generated image could not be saved: {error}"
                    )),
                    self.call_semantics(arguments),
                ));
            }
        };

        let resource_id = attachment.resource_id.as_deref().unwrap_or("unavailable");
        let model = generated.model.as_deref().unwrap_or("provider default");
        let dimensions = generated
            .dimensions
            .as_deref()
            .map(|value| format!("\nDimensions: {value}"))
            .unwrap_or_default();
        let output = format!(
            "Generated image successfully.\nBackend: {}\nModel: {model}{dimensions}\nSaved to: {}\nResource: {resource_id}\nThe image is attached for vision inspection. To deliver it to the user, call `send_file` with resource_id `{resource_id}`.",
            self.backend.name(),
            attachment.local_path,
        );

        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: self.call_semantics(arguments),
                attachments: vec![attachment],
                ..ToolCallMetadata::default()
            },
        })
    }
}

fn extension_for_mime(mime_type: &str) -> &'static str {
    match mime_type {
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "png",
    }
}

fn error_outcome(error: ImageGenerationError, semantics: ToolCallSemantics) -> ToolCallOutcome {
    ToolCallOutcome {
        output: format!("Error: Image generation failed: {error}"),
        metadata: ToolCallMetadata {
            outcome_status: Some(error.outcome_status()),
            http_status: error.http_status,
            semantics,
            ..ToolCallMetadata::default()
        },
    }
}

#[async_trait]
impl Tool for GenerateImageTool {
    fn name(&self) -> &str {
        "generate_image"
    }

    fn description(&self) -> &str {
        "Generate a new raster image from a text prompt using the configured image backend. The backend is independent of the conversational model provider. The result is attached for inspection; use send_file with its resource_id to deliver it to the user."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "generate_image",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Complete image brief: subject, composition, style, lighting, colors, and required text."
                    }
                },
                "required": ["prompt"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self.generate_outcome(arguments, None).await?.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.generate_outcome(arguments, status_tx).await
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
            .with_verification_mode(ToolVerificationMode::ResultContent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::post;
    use axum::{Json, Router};
    use chrono::Utc;

    #[derive(Debug, Default)]
    struct CapturedRequest {
        authorization: String,
        account_id: String,
        originator: String,
        turn_id: String,
        body: Value,
    }

    async fn image_response(
        State(captured): State<Arc<StdMutex<CapturedRequest>>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let mut captured = captured.lock().expect("capture request");
        captured.authorization = header(&headers, "authorization");
        captured.account_id = header(&headers, "chatgpt-account-id");
        captured.originator = header(&headers, "originator");
        captured.turn_id = header(&headers, "x-codex-image-turn-id");
        captured.body = body;

        let png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        Json(json!({
            "created": 1,
            "data": [{
                "b64_json": base64::engine::general_purpose::STANDARD.encode(png)
            }],
            "size": "1024x1024"
        }))
    }

    fn header(headers: &HeaderMap, name: &str) -> String {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string()
    }

    struct AlternateProviderBackend;

    #[async_trait]
    impl ImageGenerationBackend for AlternateProviderBackend {
        fn name(&self) -> &str {
            "alternate_provider"
        }

        async fn generate(
            &self,
            _request: &ImageGenerationRequest,
        ) -> Result<GeneratedImage, ImageGenerationError> {
            Ok(GeneratedImage {
                bytes: vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
                mime_type: Some("image/png".to_string()),
                model: Some("alternate-image-model".to_string()),
                dimensions: None,
            })
        }
    }

    #[tokio::test]
    async fn provider_adapter_is_swappable_without_changing_the_tool_contract() {
        let temp = tempfile::tempdir().expect("create temp inbox");
        let tool = GenerateImageTool::new(
            Arc::new(AlternateProviderBackend),
            temp.path().to_path_buf(),
        );

        let outcome = tool
            .call_with_status_outcome(&json!({"prompt": "A portable prompt"}).to_string(), None)
            .await
            .expect("generate with alternate adapter");

        assert_eq!(outcome.metadata.attachments.len(), 1);
        assert!(outcome.output.contains("Backend: alternate_provider"));
        assert!(outcome.output.contains("Model: alternate-image-model"));
    }

    #[tokio::test]
    async fn chatgpt_backend_authenticates_saves_and_attaches_image() {
        let captured = Arc::new(StdMutex::new(CapturedRequest::default()));
        let app = Router::new()
            .route("/images/generations", post(image_response))
            .with_state(captured.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("read test server address");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let credentials =
            ChatGptCredentialManager::with_credentials(chatgpt_codex::ChatGptCredentials {
                access_token: "synthetic-access".to_string(),
                refresh_token: "synthetic-refresh".to_string(),
                account_id: "acct-synthetic".to_string(),
                expires_at: Utc::now() + chrono::Duration::hours(1),
            });
        let backend = ChatGptSubscriptionImageBackend::with_dependencies(
            &format!("http://{address}"),
            reqwest::Client::new(),
            Arc::new(credentials),
        );
        let temp = tempfile::tempdir().expect("create temp inbox");
        let tool = GenerateImageTool::new(Arc::new(backend), temp.path().to_path_buf());

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "prompt": "A tiny red fox in watercolor",
                    "_turn_id": "turn-synthetic-1"
                })
                .to_string(),
                None,
            )
            .await
            .expect("generate image");
        server.abort();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert_eq!(outcome.metadata.attachments.len(), 1);
        let attachment = &outcome.metadata.attachments[0];
        assert_eq!(attachment.mime_type, "image/png");
        assert_eq!(attachment.source_tool.as_deref(), Some("generate_image"));
        assert!(attachment
            .resource_id
            .as_deref()
            .is_some_and(|id| id.starts_with("res_")));
        assert_eq!(
            std::fs::read(&attachment.local_path).expect("read saved image"),
            [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        );
        assert!(outcome.output.contains("send_file"));

        let captured = captured.lock().expect("read request capture");
        assert_eq!(captured.authorization, "Bearer synthetic-access");
        assert_eq!(captured.account_id, "acct-synthetic");
        assert_eq!(captured.originator, "aidaemon");
        assert_eq!(captured.turn_id, "turn-synthetic-1");
        assert_eq!(captured.body["prompt"], "A tiny red fox in watercolor");
        assert_eq!(captured.body["model"], CHATGPT_IMAGE_MODEL);
        assert_eq!(captured.body["background"], "auto");
        assert_eq!(captured.body["quality"], "auto");
        assert_eq!(captured.body["size"], "auto");
        assert!(captured.body.get("n").is_none());
    }

    #[test]
    fn schema_and_capabilities_are_provider_neutral() {
        let credentials =
            ChatGptCredentialManager::with_credentials(chatgpt_codex::ChatGptCredentials {
                access_token: "a".to_string(),
                refresh_token: "r".to_string(),
                account_id: "acct".to_string(),
                expires_at: Utc::now() + chrono::Duration::hours(1),
            });
        let backend = ChatGptSubscriptionImageBackend::with_dependencies(
            "http://example.invalid",
            reqwest::Client::new(),
            Arc::new(credentials),
        );
        let tool = GenerateImageTool::new(Arc::new(backend), PathBuf::from("/tmp"));
        let schema = tool.schema();

        assert_eq!(schema["name"], "generate_image");
        assert_eq!(schema["parameters"]["required"], json!(["prompt"]));
        assert_eq!(
            schema["parameters"]["additionalProperties"],
            Value::Bool(false)
        );
        assert!(!tool.capabilities().read_only);
        assert!(tool.capabilities().external_side_effect);
        assert!(!tool.capabilities().needs_approval);
    }

    #[test]
    fn http_errors_have_actionable_provider_specific_messages() {
        let auth = chatgpt_image_http_error(401, "expired");
        assert_eq!(auth.kind, ImageGenerationErrorKind::Blocked);
        assert!(auth.message.contains("aidaemon auth login openai"));
        assert_eq!(auth.http_status, Some(401));

        let rate_limit = chatgpt_image_http_error(429, "limit");
        assert_eq!(rate_limit.kind, ImageGenerationErrorKind::Retryable);
        assert_eq!(rate_limit.http_status, Some(429));
    }

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
        let tool = GenerateImageTool::chatgpt_subscription(output_dir)
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
}
