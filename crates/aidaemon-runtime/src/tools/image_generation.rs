//! Provider-neutral image generation tool and provider adapters.
//!
//! The agent sees one stable `generate_image` contract. Backends translate that
//! request into each provider's API, so image generation is independent of the
//! provider handling the surrounding conversation.

use aidaemon_tools::semantics::ToolSemantics;
use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::Digest;
use tokio::sync::mpsc;

use aidaemon_channels::attachments::{save_tool_observation_image, sniff_image_mime};
use aidaemon_events::EventStore;
use aidaemon_events::payloads::ResourceRegisteredData;
use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolMutationEffects, ToolOutcomeStatus, ToolVerificationMode,
};
use aidaemon_tools::status::StatusUpdate;

pub const OPENAI_IMAGES_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_IMAGE_MODEL: &str = "gpt-image-2.5-sunburst";
const OPENAI_IMAGE_BACKEND: &str = "openai_images_api";
const UNCONFIGURED_IMAGE_BACKEND: &str = "unconfigured";
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

/// Image adapter backed by OpenAI's official metered Images API
/// (`POST /v1/images/generations` and `POST /v1/images/edits`).
pub struct OpenAiImagesBackend {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAiImagesBackend {
    /// Create a backend authenticated with an OpenAI API key.
    pub fn new(base_url: Option<&str>, api_key: String) -> Result<Self, String> {
        let api_key = api_key.trim().to_string();
        if api_key.is_empty() {
            return Err("OpenAI API key cannot be empty".to_string());
        }
        Ok(Self {
            client: crate::chat_bridge::build_http_client(DEFAULT_TIMEOUT)?,
            base_url: normalize_base_url(base_url),
            api_key,
            model: OPENAI_IMAGE_MODEL.to_string(),
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let model = model.into();
        if !model.trim().is_empty() {
            self.model = model.trim().to_string();
        }
        self
    }

    #[cfg(test)]
    fn with_dependencies(base_url: &str, client: reqwest::Client, api_key: String) -> Self {
        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: OPENAI_IMAGE_MODEL.to_string(),
        }
    }

    async fn send_generation(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<reqwest::Response, ImageGenerationError> {
        let url = format!("{}/images/generations", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": request.prompt,
            "background": "auto",
            "quality": "auto",
            "size": "auto"
        });
        self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("User-Agent", image_user_agent())
            .header("accept", "application/json")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                ImageGenerationError::retryable(format!(
                    "OpenAI image generation request could not be sent: {error}"
                ))
            })
    }

    async fn send_edit(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<reqwest::Response, ImageGenerationError> {
        let url = format!("{}/images/edits", self.base_url);
        let boundary = format!("----aidaemon-image-{}", uuid::Uuid::new_v4().simple());
        let body = build_multipart_edit_body(&boundary, &self.model, request);
        self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("User-Agent", image_user_agent())
            .header("accept", "application/json")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .map_err(|error| {
                ImageGenerationError::retryable(format!(
                    "OpenAI image edit request could not be sent: {error}"
                ))
            })
    }
}

/// Placeholder backend used when no metered image API key is configured.
/// Keeps the `generate_image` tool registered and returns an actionable
/// configuration message instead of silently failing or calling unofficial
/// subscription endpoints.
pub struct UnconfiguredImageBackend;

#[async_trait]
impl ImageGenerationBackend for UnconfiguredImageBackend {
    fn name(&self) -> &str {
        UNCONFIGURED_IMAGE_BACKEND
    }

    async fn generate(
        &self,
        _request: &ImageGenerationRequest,
    ) -> Result<GeneratedImage, ImageGenerationError> {
        Err(ImageGenerationError::blocked(
            "Image generation requires an OpenAI API key (ChatGPT subscription sign-in does not \
             include image generation). Set `OPENAI_API_KEY` in the daemon environment or \
             configure an OpenAI API provider in `config.toml`.",
        ))
    }
}

fn normalize_base_url(base_url: Option<&str>) -> String {
    base_url
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_IMAGES_BASE_URL)
        .trim_end_matches('/')
        .to_string()
}

fn sanitize_multipart_filename(filename: &str, index: usize, mime_type: &str) -> String {
    let cleaned: String = filename
        .chars()
        .filter(|ch| !matches!(ch, '\r' | '\n' | '"' | '\\') && !ch.is_control())
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        format!("reference_{}.{}", index + 1, extension_for_mime(mime_type))
    } else {
        trimmed.to_string()
    }
}

/// Build an RFC 7578 `multipart/form-data` payload for OpenAI's
/// `POST /v1/images/edits` endpoint, preserving the exact order of
/// `request.references` via `image[]` file parts.
fn build_multipart_edit_body(
    boundary: &str,
    model: &str,
    request: &ImageGenerationRequest,
) -> Vec<u8> {
    let estimated_bytes: usize = request
        .references
        .iter()
        .map(|reference| reference.bytes.len() + 256)
        .sum::<usize>()
        + request.prompt.len()
        + 512;
    let mut out = Vec::with_capacity(estimated_bytes);

    let push_text_field = |buf: &mut Vec<u8>, name: &str, value: &str| {
        buf.extend_from_slice(b"--");
        buf.extend_from_slice(boundary.as_bytes());
        buf.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"");
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(b"\"\r\n\r\n");
        buf.extend_from_slice(value.as_bytes());
        buf.extend_from_slice(b"\r\n");
    };

    push_text_field(&mut out, "model", model);
    push_text_field(&mut out, "prompt", &request.prompt);
    push_text_field(&mut out, "background", "auto");
    push_text_field(&mut out, "quality", "auto");
    push_text_field(&mut out, "size", "auto");

    for (index, reference) in request.references.iter().enumerate() {
        let filename =
            sanitize_multipart_filename(&reference.filename, index, &reference.mime_type);
        out.extend_from_slice(b"--");
        out.extend_from_slice(boundary.as_bytes());
        out.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"");
        out.extend_from_slice(filename.as_bytes());
        out.extend_from_slice(b"\"\r\nContent-Type: ");
        out.extend_from_slice(reference.mime_type.as_bytes());
        out.extend_from_slice(b"\r\n\r\n");
        out.extend_from_slice(&reference.bytes);
        out.extend_from_slice(b"\r\n");
    }

    out.extend_from_slice(b"--");
    out.extend_from_slice(boundary.as_bytes());
    out.extend_from_slice(b"--\r\n");
    out
}

#[derive(Debug, Deserialize)]
struct OpenAiImageResponse {
    #[serde(default)]
    data: Vec<OpenAiImageData>,
    #[serde(default)]
    size: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiImageData {
    b64_json: String,
}

#[async_trait]
impl ImageGenerationBackend for OpenAiImagesBackend {
    fn name(&self) -> &str {
        OPENAI_IMAGE_BACKEND
    }

    async fn generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<GeneratedImage, ImageGenerationError> {
        let response = if request.references.is_empty() {
            self.send_generation(request).await?
        } else {
            self.send_edit(request).await?
        };
        let status = response.status();
        let response_body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(openai_image_http_error(status.as_u16(), &response_body));
        }

        let parsed: OpenAiImageResponse =
            serde_json::from_str(&response_body).map_err(|error| {
                ImageGenerationError::retryable(format!(
                    "OpenAI image generation returned unexpected JSON: {error}"
                ))
            })?;
        let encoded = parsed
            .data
            .into_iter()
            .next()
            .map(|data| data.b64_json)
            .ok_or_else(|| {
                ImageGenerationError::permanent("OpenAI image generation returned no image data")
            })?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded.trim().as_bytes())
            .map_err(|error| {
                ImageGenerationError::permanent(format!(
                    "OpenAI image generation returned invalid base64: {error}"
                ))
            })?;

        Ok(GeneratedImage {
            bytes,
            mime_type: Some("image/png".to_string()),
            model: Some(self.model.clone()),
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

fn openai_image_http_error(status: u16, body: &str) -> ImageGenerationError {
    let details = truncate_error_body(body);
    let error = match status {
        401 => ImageGenerationError::blocked(format!(
            "OpenAI Images API authentication rejected. Check your OpenAI API key \
             (`OPENAI_API_KEY` or `[provider]` in config.toml). Details: {details}"
        )),
        403 => ImageGenerationError::blocked(format!(
            "OpenAI Images API access denied for this project or organization. Details: {details}"
        )),
        429 => ImageGenerationError::retryable(format!(
            "OpenAI Images API rate limit or quota reached. Details: {details}"
        )),
        500..=599 => ImageGenerationError::retryable(format!(
            "OpenAI Images API is temporarily unavailable. Details: {details}"
        )),
        _ => ImageGenerationError::permanent(format!(
            "OpenAI image generation failed with HTTP {status}. Details: {details}"
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

    pub fn openai_api(inbox_dir: PathBuf, api_key: String) -> Result<Self, String> {
        Ok(Self::new(
            Arc::new(OpenAiImagesBackend::new(None, api_key)?),
            inbox_dir,
        ))
    }

    pub fn unconfigured(inbox_dir: PathBuf) -> Self {
        Self::new(Arc::new(UnconfiguredImageBackend), inbox_dir)
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
        // Best effort: a handle that stays registered is checked again on use.
        if let Some(store) = &self.event_store {
            let _ = store
                .invalidate_resource(session_id, resource_id, reason)
                .await;
        }
    }

    async fn resolve_reference_images(
        &self,
        args: &Value,
        facts: &ToolCallContext,
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

        let session_id = facts.session_id.as_str();
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
            let actual_sha256 = hex::encode(sha2::Sha256::digest(&bytes));
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
        facts: &ToolCallContext,
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

        let references = match self.resolve_reference_images(&args, facts).await {
            Ok(references) => references,
            Err(error) => return Ok(error_outcome(error, self.call_semantics(arguments))),
        };

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
            turn_id: facts
                .turn_id
                .as_deref()
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

        // The image is the request's own artifact: a later send_file of it
        // (by path or resource id, from any run of the request) delivers
        // what this call made, and the untrusted-call check knows so (live
        // 2026-11-17: a mascot this call generated was carded to the owner
        // as a file "not identified as an artifact" of the request).
        if let Some(produced) = aidaemon_tools::semantics::produced_files(facts) {
            produced.note_created(&attachment.local_path, attachment.resource_id.as_deref());
        }
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
            finish: None,
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
        finish: None,
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

    async fn execute(
        &self,
        arguments: &str,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let status_tx = aidaemon_tools::status::status_sender(facts);
        let facts = facts.clone();

        self.generate_outcome(arguments, status_tx, &facts).await
    }
}

impl aidaemon_tools::semantics::ToolSemantics for GenerateImageTool {
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::EXTERNAL_WRITE
    }

    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Trusted
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::LOCAL_WORKSPACE_WRITE)
            .with_verification_mode(ToolVerificationMode::ResultContent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_events::{Event, EventType};
    use aidaemon_tools::testing::ToolCalls as _;
    use std::sync::Mutex as StdMutex;

    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::post;
    use axum::{Json, Router};

    #[derive(Debug, Default)]
    struct CapturedRequest {
        authorization: String,
        content_type: String,
        has_account_id: bool,
        has_originator: bool,
        has_codex_turn_id: bool,
        body: Value,
        raw_body: Vec<u8>,
    }

    fn synthetic_png_response() -> Json<Value> {
        let png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        Json(json!({
            "created": 1,
            "data": [{
                "b64_json": base64::engine::general_purpose::STANDARD.encode(png)
            }],
            "size": "1024x1024"
        }))
    }

    async fn image_json_response(
        State(captured): State<Arc<StdMutex<CapturedRequest>>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        let mut captured = captured.lock().expect("capture request");
        captured.authorization = header(&headers, "authorization");
        captured.content_type = header(&headers, "content-type");
        captured.has_account_id = headers.contains_key("chatgpt-account-id");
        captured.has_originator = headers.contains_key("originator");
        captured.has_codex_turn_id = headers.contains_key("x-codex-image-turn-id");
        captured.body = body;
        synthetic_png_response()
    }

    async fn image_multipart_response(
        State(captured): State<Arc<StdMutex<CapturedRequest>>>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Json<Value> {
        let mut captured = captured.lock().expect("capture multipart request");
        captured.authorization = header(&headers, "authorization");
        captured.content_type = header(&headers, "content-type");
        captured.has_account_id = headers.contains_key("chatgpt-account-id");
        captured.has_originator = headers.contains_key("originator");
        captured.has_codex_turn_id = headers.contains_key("x-codex-image-turn-id");
        captured.raw_body = body.to_vec();
        synthetic_png_response()
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
            EventStore::on_paths(
                pool,
                aidaemon_tools::LexicalPaths::daemon_host(),
                Arc::new(aidaemon_events::NoEventHost),
            )
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
            sha256: aidaemon_channels::attachments::sha256_file(path),
            provenance: aidaemon_events::payloads::ResourceProvenance::CurrentAttachment,
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
    async fn a_prompt_about_a_reference_face_without_ids_still_generates() {
        // The prompt's words never refuse a call; the schema asks the model
        // for reference ids and a call without them is text-only.
        let temp = tempfile::tempdir().expect("create temp inbox");
        let tool = GenerateImageTool::new(
            Arc::new(AlternateProviderBackend),
            temp.path().to_path_buf(),
        );
        let outcome = tool
            .call_with_status_outcome(
                &json!({"prompt": "Keep the same face as the reference image of Mia"}).to_string(),
                None,
            )
            .await
            .expect("generate without references");
        assert_eq!(outcome.metadata.attachments.len(), 1);
    }

    #[tokio::test]
    async fn openai_backend_authenticates_saves_and_attaches_image() {
        let captured = Arc::new(StdMutex::new(CapturedRequest::default()));
        let app = Router::new()
            .route("/images/generations", post(image_json_response))
            .with_state(captured.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("read test server address");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let backend = OpenAiImagesBackend::with_dependencies(
            &format!("http://{address}"),
            reqwest::Client::new(),
            "sk-synthetic-test-key".to_string(),
        );
        let temp = tempfile::tempdir().expect("create temp inbox");
        let tool = GenerateImageTool::new(Arc::new(backend), temp.path().to_path_buf());

        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "prompt": "A tiny red fox in watercolor"
                })
                .to_string(),
                None,
                aidaemon_tools::semantics::ToolCallContext {
                    turn_id: Some("turn-synthetic-1".into()),
                    ..Default::default()
                },
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
        assert!(
            attachment
                .resource_id
                .as_deref()
                .is_some_and(|id| id.starts_with("res_"))
        );
        assert_eq!(
            std::fs::read(&attachment.local_path).expect("read saved image"),
            [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        );
        assert!(outcome.output.contains("send_file"));

        let captured = captured.lock().expect("read request capture");
        assert_eq!(captured.authorization, "Bearer sk-synthetic-test-key");
        assert!(!captured.has_account_id);
        assert!(!captured.has_originator);
        assert!(!captured.has_codex_turn_id);
        assert_eq!(captured.body["prompt"], "A tiny red fox in watercolor");
        assert_eq!(captured.body["model"], OPENAI_IMAGE_MODEL);
        assert_eq!(captured.body["background"], "auto");
        assert_eq!(captured.body["quality"], "auto");
        assert_eq!(captured.body["size"], "auto");
        assert!(captured.body.get("n").is_none());
    }

    #[tokio::test]
    async fn openai_backend_uses_multipart_edits_for_ordered_reference_images() {
        let captured = Arc::new(StdMutex::new(CapturedRequest::default()));
        let app = Router::new()
            .route("/images/edits", post(image_multipart_response))
            .with_state(captured.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let address = listener.local_addr().expect("read test server address");
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let backend = OpenAiImagesBackend::with_dependencies(
            &format!("http://{address}"),
            reqwest::Client::new(),
            "sk-synthetic-test-key".to_string(),
        );
        let result = backend
            .generate(&ImageGenerationRequest {
                prompt: "Preserve the face from the first reference".to_string(),
                references: vec![
                    ImageGenerationReference {
                        bytes: b"FIRST_REFERENCE_BYTES".to_vec(),
                        filename: "mia-primary.jpg".to_string(),
                        mime_type: "image/jpeg".to_string(),
                    },
                    ImageGenerationReference {
                        bytes: b"SECOND_REFERENCE_BYTES".to_vec(),
                        filename: "zoe-\"quoted\"\r\n.png".to_string(),
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
        assert_eq!(captured.authorization, "Bearer sk-synthetic-test-key");
        assert!(!captured.has_account_id);
        assert!(!captured.has_originator);
        assert!(!captured.has_codex_turn_id);
        assert!(
            captured
                .content_type
                .starts_with("multipart/form-data; boundary=----aidaemon-image-"),
            "{}",
            captured.content_type
        );
        let body_text = String::from_utf8_lossy(&captured.raw_body);
        assert!(body_text.contains(&format!(
            "Content-Disposition: form-data; name=\"model\"\r\n\r\n{OPENAI_IMAGE_MODEL}\r\n"
        )));
        assert!(body_text.contains(
            "Content-Disposition: form-data; name=\"prompt\"\r\n\r\nPreserve the face from the first reference\r\n"
        ));
        let first_idx = body_text
            .find("name=\"image[]\"; filename=\"mia-primary.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nFIRST_REFERENCE_BYTES\r\n")
            .expect("first reference part");
        let second_idx = body_text
            .find("name=\"image[]\"; filename=\"zoe-quoted.png\"\r\nContent-Type: image/png\r\n\r\nSECOND_REFERENCE_BYTES\r\n")
            .expect("second reference part with sanitized filename");
        assert!(first_idx < second_idx, "reference order must be preserved");
    }

    #[tokio::test]
    async fn unconfigured_backend_blocks_with_setup_instructions() {
        let temp = tempfile::tempdir().expect("create temp inbox");
        let tool = GenerateImageTool::unconfigured(temp.path().to_path_buf());
        let outcome = tool
            .call_with_status_outcome(
                &json!({"prompt": "A watercolor lighthouse"}).to_string(),
                None,
            )
            .await
            .expect("return structured blocked error");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Blocked)
        );
        assert!(outcome.output.contains("OPENAI_API_KEY"));
        assert!(outcome.output.contains("config.toml"));
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
            .call_with_execution_context(
                &json!({
                    "prompt": "Keep Bella's exact face",
                    "reference_image_resource_ids": ["res_primary", "res_secondary"]
                })
                .to_string(),
                None,
                aidaemon_tools::semantics::ToolCallContext {
                    session_id: "slack:test".into(),
                    turn_id: Some("turn-follow-up".into()),
                    ..Default::default()
                },
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
            .call_with_execution_context(
                &json!({
                    "prompt": "Use another session's image",
                    "reference_image_resource_ids": ["res_private"]
                })
                .to_string(),
                None,
                aidaemon_tools::testing::call_facts("slack:private-b"),
            )
            .await
            .expect("return a structured reference error");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome.output.contains("unknown, expired, or invalidated"));
        assert!(
            captured
                .lock()
                .expect("read captured backend requests")
                .is_empty()
        );
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
            .call_with_execution_context(
                &json!({
                    "prompt": "Use the changed image",
                    "reference_image_resource_ids": ["res_changed"]
                })
                .to_string(),
                None,
                aidaemon_tools::testing::call_facts("slack:test"),
            )
            .await
            .expect("return a structured integrity error");

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::FailedPermanent)
        );
        assert!(outcome.output.contains("changed after registration"));
        assert!(
            store
                .get_resource("slack:test", "res_changed")
                .await
                .expect("query invalidated resource")
                .is_none()
        );
        assert!(
            captured
                .lock()
                .expect("read captured backend requests")
                .is_empty()
        );
    }

    #[test]
    fn schema_and_capabilities_are_provider_neutral() {
        let backend = OpenAiImagesBackend::with_dependencies(
            "http://example.invalid",
            reqwest::Client::new(),
            "sk-synthetic".to_string(),
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
        let auth = openai_image_http_error(401, "invalid_api_key");
        assert_eq!(auth.kind, ImageGenerationErrorKind::Blocked);
        assert!(auth.message.contains("OPENAI_API_KEY"));
        assert_eq!(auth.http_status, Some(401));

        let forbidden = openai_image_http_error(403, "org_not_verified");
        assert_eq!(forbidden.kind, ImageGenerationErrorKind::Blocked);
        assert_eq!(forbidden.http_status, Some(403));

        let rate_limit = openai_image_http_error(429, "limit");
        assert_eq!(rate_limit.kind, ImageGenerationErrorKind::Retryable);
        assert_eq!(rate_limit.http_status, Some(429));
    }
}
