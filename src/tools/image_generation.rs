//! Provider-neutral image generation tool and provider adapters.
//!
//! The agent sees one stable `generate_image` contract. Backends translate that
//! request into each provider's API, so image generation is independent of the
//! provider handling the surrounding conversation.

use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::Digest;
use tokio::sync::mpsc;

use crate::channels::attachments::{save_tool_observation_image, sniff_image_mime};
use crate::events::{
    Event, EventStore, EventType, ResourceInvalidatedData, ResourceRegisteredData,
};
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
const MAX_REFERENCE_IMAGES: usize = 16;
const MAX_REFERENCE_IMAGE_BYTES: u64 = 50 * 1_048_576;

/// One validated image supplied to the provider as an edit/reference input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationReference {
    pub bytes: Vec<u8>,
    pub filename: String,
    pub mime_type: String,
}

/// Provider-independent request passed to an image generation backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    /// Ordered reference inputs. Identity-critical references belong first.
    pub references: Vec<ImageGenerationReference>,
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

    async fn send_authenticated(
        &self,
        url: &str,
        body: &Value,
        turn_id: Option<&str>,
        credentials: &chatgpt_codex::ChatGptCredentials,
    ) -> Result<reqwest::Response, ImageGenerationError> {
        let mut request_builder = self
            .client
            .post(url)
            .header(
                "Authorization",
                format!("Bearer {}", credentials.access_token),
            )
            .header("chatgpt-account-id", &credentials.account_id)
            .header("originator", ORIGINATOR)
            .header("User-Agent", image_user_agent())
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .json(body);
        if let Some(turn_id) = turn_id.map(str::trim).filter(|value| !value.is_empty()) {
            request_builder = request_builder.header("x-codex-image-turn-id", turn_id);
        }
        request_builder.send().await.map_err(|error| {
            ImageGenerationError::retryable(format!(
                "ChatGPT image request could not be sent: {error}"
            ))
        })
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
        let mut credentials = self
            .credentials
            .usable_credentials(&self.client)
            .await
            .map_err(|error| ImageGenerationError::blocked(error.to_string()))?;

        let endpoint = if request.references.is_empty() {
            "generations"
        } else {
            "edits"
        };
        let url = format!("{}/images/{endpoint}", self.base_url);
        let mut body = json!({
            "prompt": request.prompt,
            "background": "auto",
            "model": CHATGPT_IMAGE_MODEL,
            "quality": "auto",
            "size": "auto"
        });
        if !request.references.is_empty() {
            body["images"] = Value::Array(
                request
                    .references
                    .iter()
                    .map(|reference| {
                        json!({
                            "image_url": format!(
                                "data:{};base64,{}",
                                reference.mime_type,
                                base64::engine::general_purpose::STANDARD.encode(&reference.bytes)
                            )
                        })
                    })
                    .collect(),
            );
        }
        let turn_id = request
            .turn_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let mut response = self
            .send_authenticated(&url, &body, turn_id, &credentials)
            .await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            let rejected = credentials.access_token.clone();
            let _ = response.bytes().await;
            credentials = self
                .credentials
                .refresh_after_rejection(&self.client, &rejected)
                .await
                .map_err(|error| ImageGenerationError::blocked(error.to_string()))?;
            response = self
                .send_authenticated(&url, &body, turn_id, &credentials)
                .await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED {
                let response_body = response.text().await.unwrap_or_default();
                self.credentials
                    .mark_rejected(
                        &credentials.access_token,
                        "ChatGPT image authentication remained rejected after automatic refresh",
                    )
                    .await;
                return Err(chatgpt_image_http_error(401, &response_body));
            }
        }
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

fn prompt_claims_reference_images(prompt: &str) -> bool {
    let normalized = prompt.to_ascii_lowercase();
    [
        "attached image",
        "attached photo",
        "attached picture",
        "attached reference",
        "provided image",
        "provided photo",
        "provided picture",
        "shared image",
        "shared photo",
        "shared picture",
        "reference image",
        "reference photo",
        "reference picture",
        "input image",
        "edit this image",
        "edit this photo",
        "same face",
        "keep her face",
        "keep his face",
        "preserve her face",
        "preserve his face",
        "preserve the face",
        "preserve bella's face",
        "preserve bella’s face",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

/// Stable agent-facing image generation tool.
pub struct GenerateImageTool {
    backend: Arc<dyn ImageGenerationBackend>,
    inbox_dir: PathBuf,
    event_store: Option<Arc<EventStore>>,
}

impl GenerateImageTool {
    pub fn new(backend: Arc<dyn ImageGenerationBackend>, inbox_dir: PathBuf) -> Self {
        Self {
            backend,
            inbox_dir,
            event_store: None,
        }
    }

    pub fn chatgpt_subscription(inbox_dir: PathBuf) -> Result<Self, String> {
        Ok(Self::new(
            Arc::new(ChatGptSubscriptionImageBackend::new(None)?),
            inbox_dir,
        ))
    }

    pub fn with_event_store(mut self, event_store: Arc<EventStore>) -> Self {
        self.event_store = Some(event_store);
        self
    }

    async fn resolve_resource(
        &self,
        session_id: &str,
        resource_id: &str,
    ) -> Result<ResourceRegisteredData, ImageGenerationError> {
        if session_id.is_empty() {
            return Err(ImageGenerationError::permanent(
                "Reference images require the current session context",
            ));
        }
        let store = self.event_store.as_ref().ok_or_else(|| {
            ImageGenerationError::permanent("The reference image registry is unavailable")
        })?;
        let resource = store
            .get_resource(session_id, resource_id)
            .await
            .map_err(|error| {
                ImageGenerationError::retryable(format!(
                    "Could not read reference image {resource_id}: {error}"
                ))
            })?
            .ok_or_else(|| {
                ImageGenerationError::permanent(format!(
                    "Reference image {resource_id} is unknown, expired, or invalidated in this session"
                ))
            })?;
        if resource.kind != "file" {
            return Err(ImageGenerationError::permanent(format!(
                "Reference {resource_id} is {}, not an image file",
                resource.kind
            )));
        }
        Ok(resource)
    }

    async fn invalidate_resource(&self, session_id: &str, resource_id: &str, reason: &str) {
        let Some(store) = &self.event_store else {
            return;
        };
        let data = ResourceInvalidatedData {
            schema_version: ResourceInvalidatedData::SCHEMA_VERSION,
            resource_id: resource_id.to_string(),
            reason: reason.to_string(),
            turn_id: None,
        };
        if let Ok(value) = serde_json::to_value(data) {
            let _ = store
                .append(Event::new(
                    session_id,
                    EventType::ResourceInvalidated,
                    value,
                ))
                .await;
        }
    }

    async fn resolve_reference_images(
        &self,
        args: &Value,
    ) -> Result<Vec<ImageGenerationReference>, ImageGenerationError> {
        let Some(values) = args.get("reference_image_resource_ids") else {
            return Ok(Vec::new());
        };
        let values = values.as_array().ok_or_else(|| {
            ImageGenerationError::permanent(
                "reference_image_resource_ids must be an array of resource IDs",
            )
        })?;
        if values.is_empty() {
            return Ok(Vec::new());
        }
        if values.len() > MAX_REFERENCE_IMAGES {
            return Err(ImageGenerationError::permanent(format!(
                "At most {MAX_REFERENCE_IMAGES} reference images are supported"
            )));
        }

        let session_id = args
            .get("_session_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let mut seen = HashSet::new();
        let mut references = Vec::with_capacity(values.len());
        for (index, value) in values.iter().enumerate() {
            let resource_id = value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    ImageGenerationError::permanent(format!(
                        "Reference image {} has an empty or invalid resource ID",
                        index + 1
                    ))
                })?;
            if !seen.insert(resource_id.to_string()) {
                return Err(ImageGenerationError::permanent(format!(
                    "Reference image {resource_id} was provided more than once"
                )));
            }

            let resource = self.resolve_resource(session_id, resource_id).await?;
            let bytes = match tokio::fs::read(&resource.locator).await {
                Ok(bytes) => bytes,
                Err(error) => {
                    self.invalidate_resource(session_id, resource_id, "reference file is missing")
                        .await;
                    return Err(ImageGenerationError::permanent(format!(
                        "Reference image {resource_id} could not be read: {error}. Reattach the image and try again."
                    )));
                }
            };
            if bytes.len() as u64 > MAX_REFERENCE_IMAGE_BYTES {
                return Err(ImageGenerationError::permanent(format!(
                    "Reference image {resource_id} is larger than 50 MB"
                )));
            }

            let expected_sha256 = resource.sha256.as_deref().ok_or_else(|| {
                ImageGenerationError::permanent(format!(
                    "Reference image {resource_id} has no integrity receipt. Reattach the image and try again."
                ))
            })?;
            let actual_sha256 = format!("{:x}", sha2::Sha256::digest(&bytes));
            if actual_sha256 != expected_sha256 {
                self.invalidate_resource(
                    session_id,
                    resource_id,
                    "reference file content changed after registration",
                )
                .await;
                return Err(ImageGenerationError::permanent(format!(
                    "Reference image {resource_id} changed after registration and was invalidated. Reattach the image and try again."
                )));
            }

            let mime_type = match sniff_image_mime(&bytes) {
                Some("image/png") => "image/png",
                Some("image/jpeg") => "image/jpeg",
                Some("image/webp") => "image/webp",
                Some(other) => {
                    return Err(ImageGenerationError::permanent(format!(
                        "Reference image {resource_id} uses unsupported format {other}; use PNG, JPEG, or WebP"
                    )));
                }
                None => {
                    return Err(ImageGenerationError::permanent(format!(
                        "Reference {resource_id} is not a valid PNG, JPEG, or WebP image"
                    )));
                }
            };
            references.push(ImageGenerationReference {
                bytes,
                filename: resource.display_name,
                mime_type: mime_type.to_string(),
            });
        }
        Ok(references)
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

        let references = match self.resolve_reference_images(&args).await {
            Ok(references) => references,
            Err(error) => return Ok(error_outcome(error, self.call_semantics(arguments))),
        };
        if references.is_empty() && prompt_claims_reference_images(prompt) {
            return Ok(error_outcome(
                ImageGenerationError::permanent(
                    "The prompt asks to use or preserve reference images, but no actual image resources were supplied. Call generate_image again with reference_image_resource_ids containing the exact IDs from the relevant [Resource: res_…] lines; put the identity-critical face image first.",
                ),
                self.call_semantics(arguments),
            ));
        }

        if let Some(status_tx) = status_tx {
            let chunk = if references.is_empty() {
                "Generating image…".to_string()
            } else {
                format!(
                    "Generating image from {} reference{}…",
                    references.len(),
                    if references.len() == 1 { "" } else { "s" }
                )
            };
            let _ = status_tx
                .send(StatusUpdate::ToolProgress {
                    name: self.name().to_string(),
                    chunk,
                })
                .await;
        }

        let request = ImageGenerationRequest {
            prompt: prompt.to_string(),
            references,
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
        let reference_count = request.references.len();
        let output = format!(
            "Generated image successfully.\nBackend: {}\nModel: {model}{dimensions}\nReference images used: {reference_count}\nSaved to: {}\nResource: {resource_id}\nThe image is attached for vision inspection. To deliver it to the user, call `send_file` with resource_id `{resource_id}`.",
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
        "Generate or edit an image. When the request refers to attached, shared, previous, or reference images—or asks to preserve visual identity—pass their exact IDs in reference_image_resource_ids, with the identity-critical image first. Without them, generation is text-only. The result is attached; use send_file with its resource_id to deliver it."
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
                    },
                    "reference_image_resource_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^res_[A-Za-z0-9_-]+$"
                        },
                        "maxItems": MAX_REFERENCE_IMAGES,
                        "uniqueItems": true,
                        "description": "Ordered image resource IDs. Include referenced attachments and put the identity-critical image first."
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

    #[derive(Default)]
    struct CapturingBackend {
        requests: Arc<StdMutex<Vec<ImageGenerationRequest>>>,
    }

    #[async_trait]
    impl ImageGenerationBackend for CapturingBackend {
        fn name(&self) -> &str {
            "capturing"
        }

        async fn generate(
            &self,
            request: &ImageGenerationRequest,
        ) -> Result<GeneratedImage, ImageGenerationError> {
            self.requests
                .lock()
                .expect("capture image request")
                .push(request.clone());
            Ok(GeneratedImage {
                bytes: vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
                mime_type: Some("image/png".to_string()),
                model: Some("capture-model".to_string()),
                dimensions: None,
            })
        }
    }

    async fn test_event_store() -> Arc<EventStore> {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("connect test event store");
        Arc::new(
            EventStore::new(pool)
                .await
                .expect("migrate test event store"),
        )
    }

    async fn register_image_resource(
        store: &EventStore,
        session_id: &str,
        resource_id: &str,
        path: &std::path::Path,
        display_name: &str,
        mime_type: &str,
    ) {
        let data = ResourceRegisteredData {
            schema_version: ResourceRegisteredData::SCHEMA_VERSION,
            resource_id: resource_id.to_string(),
            kind: "file".to_string(),
            locator: path.to_string_lossy().into_owned(),
            display_name: display_name.to_string(),
            mime_type: Some(mime_type.to_string()),
            size_bytes: std::fs::metadata(path).ok().map(|metadata| metadata.len()),
            sha256: crate::channels::attachments::sha256_file(path),
            provenance: crate::events::ResourceProvenance::CurrentAttachment,
            produced_by_tool_call_id: None,
            source_tool: None,
            task_id: Some("task-reference-test".to_string()),
            turn_id: Some("turn-reference-test".to_string()),
        };
        store
            .append(Event::new(
                session_id,
                EventType::ResourceRegistered,
                serde_json::to_value(data).expect("serialize resource"),
            ))
            .await
            .expect("register image resource");
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

    #[tokio::test]
    async fn chatgpt_backend_uses_json_edits_for_ordered_reference_images() {
        let captured = Arc::new(StdMutex::new(CapturedRequest::default()));
        let app = Router::new()
            .route("/images/edits", post(image_response))
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
        let result = backend
            .generate(&ImageGenerationRequest {
                prompt: "Preserve the face from the first reference".to_string(),
                references: vec![
                    ImageGenerationReference {
                        bytes: b"FIRST_REFERENCE_BYTES".to_vec(),
                        filename: "bella-primary.jpg".to_string(),
                        mime_type: "image/jpeg".to_string(),
                    },
                    ImageGenerationReference {
                        bytes: b"SECOND_REFERENCE_BYTES".to_vec(),
                        filename: "bella-secondary.png".to_string(),
                        mime_type: "image/png".to_string(),
                    },
                ],
                turn_id: Some("turn-edit-1".to_string()),
            })
            .await
            .expect("edit from reference images");
        server.abort();

        assert_eq!(result.dimensions.as_deref(), Some("1024x1024"));
        let captured = captured.lock().expect("read edit request capture");
        assert_eq!(captured.authorization, "Bearer synthetic-access");
        assert_eq!(captured.account_id, "acct-synthetic");
        assert_eq!(captured.originator, "aidaemon");
        assert_eq!(captured.turn_id, "turn-edit-1");
        assert_eq!(captured.body["model"], CHATGPT_IMAGE_MODEL);
        assert_eq!(
            captured.body["prompt"],
            "Preserve the face from the first reference"
        );
        let images = captured.body["images"]
            .as_array()
            .expect("ordered image reference array");
        assert_eq!(images.len(), 2);
        assert_eq!(
            images[0]["image_url"],
            format!(
                "data:image/jpeg;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(b"FIRST_REFERENCE_BYTES")
            )
        );
        assert_eq!(
            images[1]["image_url"],
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(b"SECOND_REFERENCE_BYTES")
            )
        );
        assert!(captured.body.get("input_fidelity").is_none());
    }

    #[tokio::test]
    async fn tool_resolves_session_resources_and_preserves_reference_order() {
        let temp = tempfile::tempdir().expect("create temp inbox");
        let primary_path = temp.path().join("primary.jpg");
        let secondary_path = temp.path().join("secondary.png");
        let primary_bytes = [0xff, 0xd8, 0xff, b'P', b'R', b'I', b'M', b'A', b'R', b'Y'];
        let secondary_bytes = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, b'S', b'E', b'C', b'O', b'N', b'D',
            b'A', b'R', b'Y',
        ];
        std::fs::write(&primary_path, primary_bytes).expect("write primary reference");
        std::fs::write(&secondary_path, secondary_bytes).expect("write secondary reference");

        let store = test_event_store().await;
        register_image_resource(
            &store,
            "slack:test",
            "res_primary",
            &primary_path,
            "bella-primary.jpg",
            "image/jpeg",
        )
        .await;
        register_image_resource(
            &store,
            "slack:test",
            "res_secondary",
            &secondary_path,
            "bella-secondary.png",
            "image/png",
        )
        .await;

        let backend = Arc::new(CapturingBackend::default());
        let captured = backend.requests.clone();
        let tool =
            GenerateImageTool::new(backend, temp.path().to_path_buf()).with_event_store(store);
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "prompt": "Keep Bella's exact face",
                    "reference_image_resource_ids": ["res_primary", "res_secondary"],
                    "_session_id": "slack:test",
                    "_turn_id": "turn-follow-up"
                })
                .to_string(),
                None,
            )
            .await
            .expect("generate from registered references");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert!(outcome.output.contains("Reference images used: 2"));
        let requests = captured.lock().expect("read captured backend requests");
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.turn_id.as_deref(), Some("turn-follow-up"));
        assert_eq!(request.references.len(), 2);
        assert_eq!(request.references[0].filename, "bella-primary.jpg");
        assert_eq!(request.references[0].mime_type, "image/jpeg");
        assert_eq!(request.references[0].bytes, primary_bytes);
        assert_eq!(request.references[1].filename, "bella-secondary.png");
        assert_eq!(request.references[1].mime_type, "image/png");
        assert_eq!(request.references[1].bytes, secondary_bytes);
    }

    #[tokio::test]
    async fn tool_rejects_cross_session_reference_without_calling_backend() {
        let temp = tempfile::tempdir().expect("create temp inbox");
        let path = temp.path().join("private.png");
        std::fs::write(&path, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
            .expect("write reference");
        let store = test_event_store().await;
        register_image_resource(
            &store,
            "slack:private-a",
            "res_private",
            &path,
            "private.png",
            "image/png",
        )
        .await;

        let backend = Arc::new(CapturingBackend::default());
        let captured = backend.requests.clone();
        let tool =
            GenerateImageTool::new(backend, temp.path().to_path_buf()).with_event_store(store);
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "prompt": "Use another session's image",
                    "reference_image_resource_ids": ["res_private"],
                    "_session_id": "slack:private-b"
                })
                .to_string(),
                None,
            )
            .await
            .expect("return a structured reference error");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome.output.contains("unknown, expired, or invalidated"));
        assert!(captured
            .lock()
            .expect("read captured backend requests")
            .is_empty());
    }

    #[tokio::test]
    async fn tool_invalidates_reference_when_registered_content_changes() {
        let temp = tempfile::tempdir().expect("create temp inbox");
        let path = temp.path().join("bella.jpg");
        std::fs::write(&path, [0xff, 0xd8, 0xff, b'O', b'L', b'D'])
            .expect("write original reference");
        let store = test_event_store().await;
        register_image_resource(
            &store,
            "slack:test",
            "res_changed",
            &path,
            "bella.jpg",
            "image/jpeg",
        )
        .await;
        std::fs::write(&path, [0xff, 0xd8, 0xff, b'N', b'E', b'W'])
            .expect("replace registered reference");

        let backend = Arc::new(CapturingBackend::default());
        let captured = backend.requests.clone();
        let tool = GenerateImageTool::new(backend, temp.path().to_path_buf())
            .with_event_store(store.clone());
        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "prompt": "Use the changed image",
                    "reference_image_resource_ids": ["res_changed"],
                    "_session_id": "slack:test"
                })
                .to_string(),
                None,
            )
            .await
            .expect("return a structured integrity error");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome.output.contains("changed after registration"));
        assert!(store
            .get_resource("slack:test", "res_changed")
            .await
            .expect("query invalidated resource")
            .is_none());
        assert!(captured
            .lock()
            .expect("read captured backend requests")
            .is_empty());
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
            schema["parameters"]["properties"]["reference_image_resource_ids"]["maxItems"],
            json!(MAX_REFERENCE_IMAGES)
        );
        assert_eq!(
            schema["parameters"]["properties"]["reference_image_resource_ids"]["uniqueItems"],
            Value::Bool(true)
        );
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
    async fn tool_refuses_silent_text_only_generation_for_reference_prompt() {
        let temp = tempfile::tempdir().expect("create temp inbox");
        let backend = Arc::new(CapturingBackend::default());
        let captured = backend.requests.clone();
        let tool = GenerateImageTool::new(backend, temp.path().to_path_buf());

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "prompt": "Use the two attached reference photos and preserve Bella's face exactly"
                })
                .to_string(),
                None,
            )
            .await
            .expect("return missing-reference error");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome
            .output
            .contains("no actual image resources were supplied"));
        assert!(outcome.output.contains("reference_image_resource_ids"));
        assert!(captured
            .lock()
            .expect("read captured backend requests")
            .is_empty());
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
        let backend = ChatGptSubscriptionImageBackend::new(None)
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
