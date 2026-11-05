//! Google Gemini provider (the Generative Language API's `generateContent`).

use std::time::Duration;

use aidaemon_core::{
    Image, Message, ModelRequest, ModelResponse, Provider, ProviderError, ResponseFormat, Role,
    ToolCall, ToolChoice, Usage,
};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use reqwest::StatusCode;
use serde_json::{json, Map, Value};

use crate::transport::{
    body_excerpt, build_client, header_map, map_reqwest_error, map_status, retry_after,
    ConfigError, Secret, DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT,
};

pub const GOOGLE_DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// A system instruction containing this marker asks for a text-only reply:
/// function calling and Search grounding are switched off for the call,
/// whatever tools the request carries.
pub const TEXT_ONLY_RESPONSE_MARKER: &str = "[TEXT_ONLY_RESPONSE_MODE]";

/// The [`ToolCall::provider_data`] keys this provider writes and replays:
/// the Gemini part fields that sit beside `functionCall` (the thinking
/// models' thought signature, under either spelling). Other keys (another
/// provider's data after a failover) are not sent to Gemini.
pub const GOOGLE_PART_FIELDS: [&str; 2] = ["thought_signature", "thoughtSignature"];

/// A provider for Google Gemini over the Generative Language API
/// (`POST {base_url}/models/{model}:generateContent`, key in
/// `x-goog-api-key`). Gemini does not stream here: `stream` yields the
/// whole reply as one `Done`.
///
/// - Leading system messages become `system_instruction`; a later system
///   message (or any hook note) stays in place as a `<system-reminder>`
///   user part, so the cached prefix is not rewritten.
/// - With no tools and [`ToolChoice::Auto`], Google Search grounding is
///   enabled and grounding sources are appended to the text.
/// - Thought signatures on `functionCall` parts are kept in
///   [`ToolCall::provider_data`] and replayed beside the call. If Gemini
///   still rejects the history for a missing signature, the unsigned calls
///   (and their results) are dropped and the request is sent once more,
///   unless [`aidaemon_core::RequestOptions::single_attempt`] is set.
/// - `usage.input_tokens` is `promptTokenCount` (it includes the cached
///   `cachedContentTokenCount`); output counts candidates plus thoughts.
#[derive(Debug, Clone)]
pub struct GoogleGenai {
    name: String,
    client: reqwest::Client,
    base_url: String,
    headers: HeaderMap,
    timeout: Duration,
}

impl GoogleGenai {
    pub fn builder() -> GoogleGenaiBuilder {
        GoogleGenaiBuilder::default()
    }

    /// The JSON body this provider would send for `request`.
    pub fn request_body(&self, request: &ModelRequest) -> Value {
        build_body(request)
    }

    /// The URL this provider posts `model`'s requests to.
    pub fn endpoint(&self, model: &str) -> String {
        format!(
            "{}/models/{}:generateContent",
            self.base_url,
            model.trim().trim_start_matches("models/")
        )
    }

    /// One POST: the parsed body on success, else the rejection.
    async fn send(
        &self,
        url: &str,
        body: &Value,
    ) -> Result<Result<Value, Rejection>, ProviderError> {
        let response = self
            .client
            .post(url)
            .timeout(self.timeout)
            .headers(self.headers.clone())
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = response.status();
        let headers = response.headers().clone();
        let text = response.text().await.map_err(map_reqwest_error)?;
        if !status.is_success() {
            return Ok(Err(Rejection {
                status,
                headers,
                body: text,
            }));
        }
        serde_json::from_str(&text)
            .map(Ok)
            .map_err(|err| ProviderError::Malformed {
                message: format!(
                    "Gemini response is not JSON ({err}): {}",
                    body_excerpt(&text)
                ),
                parse: true,
            })
    }
}

struct Rejection {
    status: StatusCode,
    headers: HeaderMap,
    body: String,
}

impl Rejection {
    fn into_error(self) -> ProviderError {
        gemini_error(self.status, &self.headers, &self.body)
    }
}

/// Builder for [`GoogleGenai`].
#[derive(Debug, Clone)]
pub struct GoogleGenaiBuilder {
    name: String,
    base_url: String,
    api_key: Option<Secret>,
    headers: Vec<(String, String)>,
    timeout: Duration,
    connect_timeout: Duration,
}

impl Default for GoogleGenaiBuilder {
    fn default() -> Self {
        Self {
            name: "google".into(),
            base_url: GOOGLE_DEFAULT_BASE_URL.into(),
            api_key: None,
            headers: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

impl GoogleGenaiBuilder {
    /// Name reported by [`Provider::name`] (default `google`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Required. Sent as `x-goog-api-key` (never in the URL).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(Secret(key.into()));
        self
    }

    /// API root, without `/models/...` (default
    /// `https://generativelanguage.googleapis.com/v1beta`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Extra header on every request.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Whole-request timeout (default 10 min).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Connection timeout (default 30 s).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<GoogleGenai, ConfigError> {
        let api_key = self
            .api_key
            .filter(|k| !k.0.is_empty())
            .ok_or(ConfigError::MissingApiKey)?;
        let mut headers = header_map(&[("x-goog-api-key".to_string(), api_key.0)])?;
        headers.extend(header_map(&self.headers)?);
        Ok(GoogleGenai {
            name: self.name,
            client: build_client(self.connect_timeout)?,
            base_url: self.base_url.trim_end_matches('/').to_string(),
            headers,
            timeout: self.timeout,
        })
    }
}

#[async_trait]
impl Provider for GoogleGenai {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let url = self.endpoint(&request.model);
        let body = self.request_body(request);
        let rejection = match self.send(&url, &body).await? {
            Ok(data) => return Ok(parse_response(&data)),
            Err(rejection) => rejection,
        };
        if !request.options.single_attempt
            && rejection.status == StatusCode::BAD_REQUEST
            && is_missing_thought_signature_error(&rejection.body)
        {
            if let Some(retry) = body_without_unsigned_calls(&body) {
                return match self.send(&url, &retry).await? {
                    Ok(data) => Ok(parse_response(&data)),
                    Err(rejection) => Err(rejection.into_error()),
                };
            }
        }
        Err(rejection.into_error())
    }
}

// ---------------------------------------------------------------- errors

/// Map a rejected call. Gemini states a 429's wait in the body
/// (`RetryInfo.retryDelay`) and answers a bad key with 400
/// `API_KEY_INVALID`; everything else follows the shared mapping.
fn gemini_error(status: StatusCode, headers: &HeaderMap, body: &str) -> ProviderError {
    match status.as_u16() {
        429 => ProviderError::RateLimited {
            retry_after: retry_after(headers).or_else(|| body_retry_delay(body)),
        },
        400 if body.contains("API_KEY_INVALID") => {
            ProviderError::Auth(format!("HTTP 400: {}", body_excerpt(body)))
        }
        _ => map_status(status, headers, body),
    }
}

/// `retryDelay` (`"30s"`, `"1.5s"`) from a google.rpc.RetryInfo detail.
fn body_retry_delay(body: &str) -> Option<Duration> {
    let value: Value = serde_json::from_str(body).ok()?;
    value
        .pointer("/error/details")?
        .as_array()?
        .iter()
        .filter_map(|detail| detail.get("retryDelay").and_then(Value::as_str))
        .filter_map(|delay| delay.trim().strip_suffix('s')?.parse::<f64>().ok())
        .find(|secs| secs.is_finite() && *secs >= 0.0)
        .map(Duration::from_secs_f64)
}

fn is_missing_thought_signature_error(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("missing a thought_signature")
        || lower.contains("missing thought_signature")
        || lower.contains("missing thoughtsignature")
}

// ----------------------------------------------------------- request body

/// Recursively strip fields Gemini rejects in function and response
/// schemas: `$schema` and `additionalProperties`.
fn strip_unsupported_fields(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("$schema");
            map.remove("additionalProperties");
            for v in map.values_mut() {
                strip_unsupported_fields(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                strip_unsupported_fields(v);
            }
        }
        _ => {}
    }
}

/// System messages before the first other message, hook notes excepted:
/// they become `system_instruction`. A hook note changes per request, so
/// it stays in place in the transcript instead.
fn leading_system(messages: &[Message]) -> usize {
    messages
        .iter()
        .position(|m| m.role != Role::System || m.is_hook_note())
        .unwrap_or(messages.len())
}

/// A system message kept in place, delimited as harness context.
fn in_place_system_text(text: &str) -> String {
    format!("<system-reminder>\n{text}\n</system-reminder>")
}

fn build_body(request: &ModelRequest) -> Value {
    let leading = leading_system(&request.messages);
    let system_parts: Vec<Value> = request.messages[..leading]
        .iter()
        .map(|m| json!({"text": m.content}))
        .collect();
    let text_only = request.messages[..leading]
        .iter()
        .any(|m| m.content.contains(TEXT_ONLY_RESPONSE_MARKER));

    let has_function_tools = !request.tools.is_empty();
    let grounding =
        matches!(request.tool_choice, ToolChoice::Auto) && !has_function_tools && !text_only;
    let disable_function_calling =
        !has_function_tools || text_only || matches!(request.tool_choice, ToolChoice::None);

    let mut body = json!({ "contents": wire_contents(&request.messages[leading..]) });
    if !system_parts.is_empty() {
        body["system_instruction"] = json!({ "parts": system_parts });
    }

    let mut tools = Vec::new();
    if has_function_tools {
        let declarations: Vec<Value> = request
            .tools
            .iter()
            .map(|spec| {
                let mut parameters = spec.parameters.clone();
                strip_unsupported_fields(&mut parameters);
                json!({
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": parameters,
                })
            })
            .collect();
        tools.push(json!({ "function_declarations": declarations }));
    }
    if grounding {
        tools.push(json!({ "google_search": {} }));
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }

    if disable_function_calling {
        body["tool_config"] = json!({ "function_calling_config": { "mode": "NONE" } });
    } else {
        match &request.tool_choice {
            ToolChoice::Required => {
                body["tool_config"] = json!({ "function_calling_config": { "mode": "ANY" } });
            }
            ToolChoice::Specific(name) => {
                body["tool_config"] = json!({
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [name],
                    }
                });
            }
            ToolChoice::Auto | ToolChoice::None => {}
        }
    }

    let mut generation = Map::new();
    match &request.options.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject => {
            generation.insert("response_mime_type".into(), json!("application/json"));
        }
        ResponseFormat::JsonSchema { schema, .. } => {
            let mut schema = schema.clone();
            strip_unsupported_fields(&mut schema);
            generation.insert("response_mime_type".into(), json!("application/json"));
            generation.insert("response_schema".into(), schema);
        }
    }
    if let Some(max) = request.max_output_tokens {
        generation.insert("max_output_tokens".into(), json!(max));
    }
    if !generation.is_empty() {
        body["generation_config"] = Value::Object(generation);
    }
    body
}

/// Map the transcript (after the system instruction) to Gemini `contents`.
///
/// - A system message stays in place as a `<system-reminder>` text part,
///   joined to the previous user content when there is one.
/// - Tool results become one `function`-role `functionResponse` content
///   each, named after the call they answer. Images from a run of tool
///   results follow the run in one user content, labelled with their call.
fn wire_contents(messages: &[Message]) -> Vec<Value> {
    let mut contents: Vec<Value> = Vec::new();
    let mut tool_images: Vec<Value> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role != Role::Tool && !tool_images.is_empty() {
            contents.push(json!({"role": "user", "parts": std::mem::take(&mut tool_images)}));
        }
        match message.role {
            Role::System => {
                if message.content.trim().is_empty() {
                    continue;
                }
                let part = json!({"text": in_place_system_text(&message.content)});
                match contents.last_mut() {
                    Some(last) if last["role"] == "user" => {
                        if let Some(parts) = last["parts"].as_array_mut() {
                            parts.push(part);
                        }
                    }
                    _ => contents.push(json!({"role": "user", "parts": [part]})),
                }
            }
            Role::User => {
                let mut parts = Vec::with_capacity(message.images.len() + 1);
                if !message.content.is_empty() {
                    parts.push(json!({"text": message.content}));
                }
                parts.extend(message.images.iter().map(image_part));
                parts.extend(message.audio.iter().filter_map(|audio| {
                    let media_type = audio.media_type()?;
                    Some(json!({"inlineData": {"mimeType": media_type, "data": audio.data}}))
                }));
                if !parts.is_empty() {
                    contents.push(json!({"role": "user", "parts": parts}));
                }
            }
            Role::Assistant => {
                let mut parts = Vec::new();
                if !message.content.is_empty() {
                    parts.push(json!({"text": message.content}));
                }
                parts.extend(message.tool_calls.iter().map(function_call_part));
                if !parts.is_empty() {
                    contents.push(json!({"role": "model", "parts": parts}));
                }
            }
            Role::Tool => {
                let call_id = message.tool_call_id.as_deref().unwrap_or_default();
                let name = called_tool_name(&messages[..index], call_id);
                contents.push(json!({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": name,
                            "response": function_response(&message.content, message.is_error),
                        }
                    }]
                }));
                if !message.images.is_empty() {
                    tool_images.push(json!({
                        "text": format!("[Images returned by tool call {call_id}]"),
                    }));
                    tool_images.extend(message.images.iter().map(image_part));
                }
            }
        }
    }
    if !tool_images.is_empty() {
        contents.push(json!({"role": "user", "parts": tool_images}));
    }
    contents
}

/// The name of the call `call_id` in the transcript before a tool result.
fn called_tool_name<'a>(before: &'a [Message], call_id: &str) -> &'a str {
    before
        .iter()
        .rev()
        .flat_map(|m| m.tool_calls.iter())
        .find(|call| call.id == call_id)
        .map(|call| call.name.as_str())
        .unwrap_or("unknown_tool")
}

/// `functionResponse.response` is a protobuf Struct, so always an object:
/// a JSON object result is sent as is, anything else under `result` (or
/// `error` for a failed call).
fn function_response(content: &str, is_error: bool) -> Value {
    let key = if is_error { "error" } else { "result" };
    match serde_json::from_str::<Value>(content) {
        Ok(Value::Object(object)) => Value::Object(object),
        Ok(other) => json!({ key: other }),
        Err(_) => json!({ key: content }),
    }
}

/// A replayed call, with its thought signature beside `functionCall` under
/// both spellings.
fn function_call_part(call: &ToolCall) -> Value {
    let args = if call.arguments.is_object() {
        call.arguments.clone()
    } else {
        json!({})
    };
    let mut part = Map::new();
    part.insert(
        "functionCall".into(),
        json!({"name": call.name, "args": args}),
    );
    if let Some(data) = call.provider_data.as_ref().and_then(Value::as_object) {
        for key in GOOGLE_PART_FIELDS {
            if let Some(value) = data.get(key) {
                part.insert(key.to_string(), value.clone());
            }
        }
        inject_thought_signature_aliases(&mut part);
    }
    Value::Object(part)
}

fn inject_thought_signature_aliases(part: &mut Map<String, Value>) {
    let sig = part
        .get("thought_signature")
        .cloned()
        .or_else(|| part.get("thoughtSignature").cloned());
    if let Some(sig) = sig {
        part.entry("thought_signature".to_string())
            .or_insert_with(|| sig.clone());
        part.entry("thoughtSignature".to_string()).or_insert(sig);
    }
}

fn image_part(image: &Image) -> Value {
    match image {
        Image::Base64 { media_type, data } => {
            json!({"inlineData": {"mimeType": media_type, "data": data}})
        }
        Image::Url { url } => match parse_data_url_image(url) {
            Some((mime_type, data)) => {
                json!({"inlineData": {"mimeType": mime_type, "data": data}})
            }
            // generateContent cannot fetch arbitrary URLs; say where the
            // image is rather than drop it silently.
            None => json!({"text": format!("[Image at {url}]")}),
        },
    }
}

fn parse_data_url_image(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(";base64,")?;
    meta.starts_with("image/")
        .then(|| (meta.to_string(), data.to_string()))
}

// ------------------------------------------------ thought-signature retry

fn part_has_thought_signature(part: &Value) -> bool {
    part.get("thought_signature").is_some() || part.get("thoughtSignature").is_some()
}

/// Gemini thinking models can reject historical `functionCall` parts that
/// carry no thought signature. Strip those parts, and the function
/// responses paired with them, as a one-shot recovery. Returns the
/// sanitized contents and the dropped (calls, responses).
fn strip_unsigned_function_call_history(contents: &[Value]) -> (Vec<Value>, usize, usize) {
    let mut sanitized = Vec::with_capacity(contents.len());
    let mut pending_response_drops = 0usize;
    let mut dropped_calls = 0usize;
    let mut dropped_responses = 0usize;

    for msg in contents {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "model" {
            let Some(parts) = msg.get("parts").and_then(Value::as_array) else {
                sanitized.push(msg.clone());
                continue;
            };
            let mut kept_parts = Vec::with_capacity(parts.len());
            let mut dropped_in_msg = 0usize;
            for part in parts {
                if part.get("functionCall").is_none() || part_has_thought_signature(part) {
                    kept_parts.push(part.clone());
                } else {
                    dropped_calls += 1;
                    dropped_in_msg += 1;
                }
            }
            pending_response_drops = pending_response_drops.saturating_add(dropped_in_msg);
            if !kept_parts.is_empty() {
                let mut kept_msg = msg.clone();
                kept_msg["parts"] = Value::Array(kept_parts);
                sanitized.push(kept_msg);
            }
            continue;
        }
        if role == "function" && pending_response_drops > 0 {
            pending_response_drops -= 1;
            dropped_responses += 1;
            continue;
        }
        sanitized.push(msg.clone());
    }
    (sanitized, dropped_calls, dropped_responses)
}

/// The retry body: `body` with unsigned call history stripped, or `None`
/// when there was nothing to strip (a retry would fail the same way).
fn body_without_unsigned_calls(body: &Value) -> Option<Value> {
    let contents = body.get("contents")?.as_array()?;
    let (sanitized, dropped_calls, _) = strip_unsigned_function_call_history(contents);
    if dropped_calls == 0 {
        return None;
    }
    let mut retry = body.clone();
    retry["contents"] = Value::Array(sanitized);
    Some(retry)
}

// ---------------------------------------------------------------- response

fn blocked_safety_categories(ratings: Option<&Vec<Value>>) -> Vec<String> {
    let mut categories: Vec<String> = Vec::new();
    for rating in ratings.into_iter().flatten() {
        if !rating
            .get("blocked")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        if let Some(category) = rating.get("category").and_then(Value::as_str) {
            if !categories.iter().any(|c| c == category) {
                categories.push(category.to_string());
            }
        }
    }
    categories
}

fn response_note(
    finish_reason: Option<&str>,
    prompt_block_reason: Option<&str>,
    prompt_blocked_categories: &[String],
    candidate_blocked_categories: &[String],
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(reason) = prompt_block_reason {
        parts.push(format!("prompt blocked ({reason})"));
    }
    if !prompt_blocked_categories.is_empty() {
        parts.push(format!(
            "prompt safety categories: {}",
            prompt_blocked_categories.join(", ")
        ));
    }
    if let Some(reason) = finish_reason {
        match reason.to_ascii_uppercase().as_str() {
            "STOP" => {}
            "MAX_TOKENS" => parts.push("output cut off at the token limit (MAX_TOKENS)".into()),
            _ => parts.push(format!("finish reason: {reason}")),
        }
    }
    if !candidate_blocked_categories.is_empty() {
        parts.push(format!(
            "candidate safety categories: {}",
            candidate_blocked_categories.join(", ")
        ));
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

fn parse_usage(data: &Value) -> Usage {
    let Some(usage) = data.get("usageMetadata") else {
        return Usage::default();
    };
    let count = |key: &str| usage.get(key).and_then(Value::as_u64).unwrap_or(0);
    let input_tokens = count("promptTokenCount");
    Usage {
        input_tokens,
        output_tokens: count("candidatesTokenCount") + count("thoughtsTokenCount"),
        cached_input_tokens: count("cachedContentTokenCount").min(input_tokens),
        cache_creation_tokens: 0,
    }
}

/// A stable id for a call Gemini sent without one: the same reply always
/// yields the same ids, and calls within a reply differ.
fn synthetic_call_id(response_id: &str, index: usize, part: &Value) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(response_id.as_bytes());
    hasher.update(index.to_le_bytes());
    hasher.update(part.to_string().as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest[..12].iter().map(|b| format!("{b:02x}")).collect();
    format!("call_{hex}")
}

/// Read a `generateContent` reply.
fn parse_response(data: &Value) -> ModelResponse {
    let usage = parse_usage(data);
    let prompt_feedback = data.get("promptFeedback");
    let prompt_block_reason = prompt_feedback
        .and_then(|pf| pf.get("blockReason"))
        .and_then(Value::as_str);
    let prompt_blocked_categories = blocked_safety_categories(
        prompt_feedback
            .and_then(|pf| pf.get("safetyRatings"))
            .and_then(Value::as_array),
    );

    let Some(candidate) = data.pointer("/candidates/0") else {
        let note = response_note(None, prompt_block_reason, &prompt_blocked_categories, &[])
            .unwrap_or_else(|| "no candidates returned by provider".to_string());
        return ModelResponse {
            usage,
            note: Some(note),
            ..Default::default()
        };
    };

    let finish_reason = candidate.get("finishReason").and_then(Value::as_str);
    let candidate_blocked_categories =
        blocked_safety_categories(candidate.get("safetyRatings").and_then(Value::as_array));
    let mut note = response_note(
        finish_reason,
        prompt_block_reason,
        &prompt_blocked_categories,
        &candidate_blocked_categories,
    );

    let response_id = data
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let parts = candidate
        .pointer("/content/parts")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(chunk) = part.get("text").and_then(Value::as_str) {
            if part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                reasoning.push_str(chunk);
            } else {
                text.push_str(chunk);
            }
        }
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            let arguments = match fc.get("args") {
                Some(Value::Object(args)) => Value::Object(args.clone()),
                _ => json!({}),
            };
            let id = fc
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| synthetic_call_id(response_id, index, part));
            let extra: Map<String, Value> = part
                .as_object()
                .into_iter()
                .flatten()
                .filter(|(key, _)| key.as_str() != "functionCall")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            let mut call = ToolCall::new(id, name, arguments);
            if !extra.is_empty() {
                call.provider_data = Some(Value::Object(extra));
            }
            tool_calls.push(call);
        }
    }

    // Google Search grounding sources.
    let sources: Vec<String> = candidate
        .pointer("/groundingMetadata/groundingChunks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|chunk| {
            let web = chunk.get("web")?;
            let uri = web.get("uri")?.as_str()?;
            let title = web.get("title").and_then(Value::as_str).unwrap_or(uri);
            Some(format!("- [{title}]({uri})"))
        })
        .collect();
    if !sources.is_empty() {
        text.push_str("\n\nSources:\n");
        text.push_str(&sources.join("\n"));
    }

    if text.trim().is_empty() && reasoning.trim().is_empty() && tool_calls.is_empty() {
        let empty = format!(
            "empty response (finishReason={}, parts={})",
            finish_reason.unwrap_or("unknown"),
            parts.len()
        );
        note = Some(match note {
            Some(existing) if !existing.trim().is_empty() => format!("{existing}; {empty}"),
            _ => empty,
        });
    }

    ModelResponse {
        text,
        tool_calls,
        usage,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::{RequestOptions, ToolSpec};
    use reqwest::header::HeaderValue;

    fn request(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: "gemini-test".into(),
            messages,
            ..Default::default()
        }
    }

    fn tool(name: &str, parameters: Value) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: "test tool".into(),
            parameters,
        }
    }

    fn assert_no_key(value: &Value, key: &str, path: &str) {
        match value {
            Value::Object(map) => {
                assert!(!map.contains_key(key), "found {key:?} at {path}");
                for (k, v) in map {
                    assert_no_key(v, key, &format!("{path}.{k}"));
                }
            }
            Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    assert_no_key(v, key, &format!("{path}[{i}]"));
                }
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ builder

    #[test]
    fn builder_requires_a_key_and_sets_the_endpoint() {
        assert!(matches!(
            GoogleGenai::builder().build(),
            Err(ConfigError::MissingApiKey)
        ));
        assert!(matches!(
            GoogleGenai::builder().api_key("").build(),
            Err(ConfigError::MissingApiKey)
        ));
        let provider = GoogleGenai::builder()
            .api_key("synthetic-key")
            .base_url("http://localhost:9/v1beta/")
            .header("x-extra", "1")
            .name("gemini")
            .build()
            .unwrap();
        assert_eq!(provider.name(), "gemini");
        assert_eq!(
            provider.endpoint("models/gemini-2.5-flash"),
            "http://localhost:9/v1beta/models/gemini-2.5-flash:generateContent"
        );
        assert!(provider.headers.contains_key("x-goog-api-key"));
        assert!(provider.headers.contains_key("x-extra"));
        assert!(!format!("{provider:?}").contains("synthetic-key"));
    }

    // -------------------------------------------------------- body building

    #[test]
    fn leading_system_messages_become_the_system_instruction() {
        let body = build_body(&request(vec![
            Message::system("Core."),
            Message::system("Be brief."),
            Message::user("Hi, I am Alice Rivera."),
        ]));
        assert_eq!(
            body["system_instruction"]["parts"],
            json!([{"text": "Core."}, {"text": "Be brief."}])
        );
        assert_eq!(
            body["contents"],
            json!([{"role": "user", "parts": [{"text": "Hi, I am Alice Rivera."}]}])
        );
    }

    #[test]
    fn later_system_messages_stay_in_place() {
        let body = build_body(&request(vec![
            Message::system("Core."),
            Message::user("Earlier turn"),
            Message::assistant("Reply", vec![]),
            Message::system("Task tail."),
            Message::user("Now"),
            Message::hook_note("Recalled: Acme Corp."),
        ]));
        assert_eq!(body["system_instruction"]["parts"][0]["text"], "Core.");
        let contents = body["contents"].as_array().unwrap();
        let roles: Vec<&str> = contents
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "model", "user", "user"]);
        assert_eq!(
            contents[2]["parts"][0]["text"],
            "<system-reminder>\nTask tail.\n</system-reminder>"
        );
        // A trailing note joins the user turn before it.
        assert_eq!(contents[3]["parts"][0]["text"], "Now");
        assert_eq!(
            contents[3]["parts"][1]["text"],
            "<system-reminder>\nRecalled: Acme Corp.\n</system-reminder>"
        );
    }

    #[test]
    fn a_leading_hook_note_is_not_hoisted() {
        let body = build_body(&request(vec![
            Message::system("Core."),
            Message::hook_note("Recalled memory."),
            Message::user("Hi"),
        ]));
        assert_eq!(
            body["system_instruction"]["parts"],
            json!([{"text": "Core."}])
        );
        assert_eq!(
            body["contents"][0]["parts"][0]["text"],
            "<system-reminder>\nRecalled memory.\n</system-reminder>"
        );
    }

    #[test]
    fn tools_are_declarations_with_unsupported_schema_fields_stripped() {
        let mut req = request(vec![Message::user("read it")]);
        req.tools = vec![
            tool(
                "clean_tool",
                json!({"type": "object", "properties": {"a": {"type": "string"}}}),
            ),
            tool(
                "mcp__server__read_file",
                json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "path": {"type": "string", "description": "The file path"},
                        "config": {
                            "type": "object",
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "properties": {"items": {"type": "array", "items": {
                                "$schema": "http://json-schema.org/draft-07/schema#",
                                "type": "string"
                            }}}
                        }
                    },
                    "required": ["path"]
                }),
            ),
        ];
        let body = build_body(&req);
        assert_no_key(&body["tools"], "$schema", "tools");
        assert_no_key(&body["tools"], "additionalProperties", "tools");
        let decls = body["tools"][0]["function_declarations"]
            .as_array()
            .unwrap();
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0]["name"], "clean_tool");
        assert_eq!(decls[1]["name"], "mcp__server__read_file");
        assert_eq!(decls[1]["parameters"]["required"][0], "path");
        assert_eq!(
            decls[1]["parameters"]["properties"]["path"]["type"],
            "string"
        );
        // Tools present: no grounding, auto choice sends no tool_config.
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert!(body.get("tool_config").is_none());
    }

    #[test]
    fn no_tools_with_auto_choice_grounds_with_google_search() {
        let body = build_body(&request(vec![Message::user("latest news")]));
        assert_eq!(body["tools"], json!([{"google_search": {}}]));
        assert_eq!(
            body["tool_config"]["function_calling_config"]["mode"],
            "NONE"
        );

        let mut req = request(vec![Message::user("latest news")]);
        req.tool_choice = ToolChoice::None;
        let body = build_body(&req);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn text_only_marker_disables_grounding_and_function_calls() {
        let mut req = request(vec![
            Message::system(format!("{TEXT_ONLY_RESPONSE_MARKER}\ntext only")),
            Message::user("what is rust?"),
        ]);
        let body = build_body(&req);
        assert!(body.get("tools").is_none());
        assert_eq!(
            body["tool_config"]["function_calling_config"]["mode"],
            "NONE"
        );

        req.tools = vec![tool("search_files", json!({"type": "object"}))];
        req.tool_choice = ToolChoice::Required;
        let body = build_body(&req);
        assert_eq!(
            body["tool_config"]["function_calling_config"]["mode"],
            "NONE"
        );
    }

    #[test]
    fn tool_choice_maps_to_function_calling_modes() {
        let mut req = request(vec![Message::user("run a tool")]);
        req.tools = vec![tool("search_files", json!({"type": "object"}))];

        req.tool_choice = ToolChoice::Required;
        assert_eq!(
            build_body(&req)["tool_config"],
            json!({"function_calling_config": {"mode": "ANY"}})
        );
        req.tool_choice = ToolChoice::Specific("search_files".into());
        assert_eq!(
            build_body(&req)["tool_config"],
            json!({"function_calling_config": {
                "mode": "ANY", "allowed_function_names": ["search_files"]
            }})
        );
        req.tool_choice = ToolChoice::None;
        assert_eq!(
            build_body(&req)["tool_config"],
            json!({"function_calling_config": {"mode": "NONE"}})
        );
    }

    #[test]
    fn json_modes_and_max_tokens_go_in_generation_config() {
        let mut req = request(vec![Message::user("json please")]);
        req.options = RequestOptions {
            response_format: ResponseFormat::JsonObject,
            ..Default::default()
        };
        assert_eq!(
            build_body(&req)["generation_config"],
            json!({"response_mime_type": "application/json"})
        );

        req.max_output_tokens = Some(512);
        req.options.response_format = ResponseFormat::JsonSchema {
            name: "intent_gate_v1".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "additionalProperties": false,
                "properties": {"can_answer_now": {"type": "boolean"}},
                "required": ["can_answer_now"]
            }),
            strict: true,
        };
        let body = build_body(&req);
        let generation = &body["generation_config"];
        assert_eq!(generation["response_mime_type"], "application/json");
        assert_eq!(generation["max_output_tokens"], 512);
        assert_eq!(
            generation["response_schema"]["required"][0],
            "can_answer_now"
        );
        assert_no_key(&generation["response_schema"], "$schema", "schema");
        assert_no_key(
            &generation["response_schema"],
            "additionalProperties",
            "schema",
        );

        let body = build_body(&request(vec![Message::user("plain")]));
        assert!(body.get("generation_config").is_none());
    }

    #[test]
    fn user_audio_becomes_inline_data() {
        let request = ModelRequest {
            model: "gemini-test".into(),
            messages: vec![Message::user("Transcribe.")
                .with_audio([aidaemon_core::Audio::new("opus", "QUJD")])],
            ..Default::default()
        };
        let body = build_body(&request);
        assert_eq!(
            body["contents"][0]["parts"][1],
            json!({"inlineData": {"mimeType": "audio/ogg", "data": "QUJD"}})
        );
    }

    #[test]
    fn images_become_inline_data() {
        let body = build_body(&request(vec![Message::user("what is this").with_images(
            vec![
                Image::base64("image/png", "AQID"),
                Image::url("data:image/jpeg;base64,BAUG"),
                Image::url("https://example.com/cat.png"),
            ],
        )]));
        let parts = &body["contents"][0]["parts"];
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(
            parts[1],
            json!({"inlineData": {"mimeType": "image/png", "data": "AQID"}})
        );
        assert_eq!(
            parts[2],
            json!({"inlineData": {"mimeType": "image/jpeg", "data": "BAUG"}})
        );
        assert_eq!(parts[3]["text"], "[Image at https://example.com/cat.png]");
    }

    #[test]
    fn tool_results_are_named_function_responses_and_images_follow_the_run() {
        let calls = vec![
            ToolCall::new("c1", "terminal", json!({"command": "ls"})),
            ToolCall::new("c2", "screenshot", json!({})),
        ];
        let mut shot = Message::tool_result("c2", "[1,2]");
        shot.images = vec![Image::base64("image/png", "AQID")];
        let mut failed = Message::tool_result("ghost", "boom");
        failed.is_error = true;
        let body = build_body(&request(vec![
            Message::user("look"),
            Message::assistant("Looking.", calls),
            Message::tool_result("c1", r#"{"stdout": "README.md"}"#),
            shot,
            failed,
            Message::user("thanks"),
        ]));
        let contents = body["contents"].as_array().unwrap();
        let roles: Vec<&str> = contents
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert_eq!(
            roles,
            ["user", "model", "function", "function", "function", "user", "user"]
        );
        assert_eq!(contents[1]["parts"][0]["text"], "Looking.");
        assert_eq!(
            contents[1]["parts"][1]["functionCall"],
            json!({"name": "terminal", "args": {"command": "ls"}})
        );
        let response = |i: usize| &contents[i]["parts"][0]["functionResponse"];
        assert_eq!(response(2)["name"], "terminal");
        assert_eq!(response(2)["response"], json!({"stdout": "README.md"}));
        assert_eq!(response(3)["name"], "screenshot");
        assert_eq!(response(3)["response"], json!({"result": [1, 2]}));
        assert_eq!(response(4)["name"], "unknown_tool");
        assert_eq!(response(4)["response"], json!({"error": "boom"}));
        assert_eq!(
            contents[5]["parts"][0]["text"],
            "[Images returned by tool call c2]"
        );
        assert_eq!(contents[5]["parts"][1]["inlineData"]["data"], "AQID");
        assert_eq!(contents[6]["parts"][0]["text"], "thanks");
    }

    #[test]
    fn replayed_calls_carry_their_signature_under_both_spellings() {
        for data in [
            json!({"thought_signature": "sig-synthetic"}),
            json!({"thoughtSignature": "sig-synthetic"}),
        ] {
            let call = ToolCall::new("c1", "write_file", json!({"path": "/tmp/a.txt"}))
                .with_provider_data(data);
            let body = build_body(&request(vec![
                Message::user("write"),
                Message::assistant("", vec![call]),
            ]));
            let part = &body["contents"][1]["parts"][0];
            assert_eq!(part["thought_signature"], "sig-synthetic");
            assert_eq!(part["thoughtSignature"], "sig-synthetic");
            assert_eq!(body["contents"][1]["parts"].as_array().unwrap().len(), 1);
        }
        // Another provider's data is not sent to Gemini.
        let call = ToolCall::new("c1", "t", json!({}))
            .with_provider_data(json!({"anthropic_thinking_blocks": []}));
        let body = build_body(&request(vec![Message::assistant("", vec![call])]));
        assert_eq!(
            body["contents"][0]["parts"][0],
            json!({"functionCall": {"name": "t", "args": {}}})
        );
    }

    // ----------------------------------------------------- response parsing

    #[test]
    fn parses_text_reasoning_usage_and_grounding_sources() {
        let response = parse_response(&json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [
                    {"text": "Considering Acme Corp.", "thought": true},
                    {"text": "Acme Corp "},
                    {"text": "is fictional."}
                ]},
                "groundingMetadata": {"groundingChunks": [
                    {"web": {"uri": "https://example.com/acme", "title": "Acme"}}
                ]}
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "cachedContentTokenCount": 70,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 3
            }
        }));
        assert_eq!(
            response.text,
            "Acme Corp is fictional.\n\nSources:\n- [Acme](https://example.com/acme)"
        );
        assert_eq!(
            response.reasoning.as_deref(),
            Some("Considering Acme Corp.")
        );
        assert_eq!(
            response.usage,
            Usage {
                input_tokens: 100,
                output_tokens: 8,
                cached_input_tokens: 70,
                cache_creation_tokens: 0
            }
        );
        assert_eq!(response.usage.billable_tokens(), 38);
        assert!(response.note.is_none());
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn function_calls_keep_their_thought_signature_and_round_trip() {
        let data = json!({
            "responseId": "resp-synthetic",
            "candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": " terminal ", "args": {"command": "ls projects"}},
                 "thoughtSignature": "sig-synthetic"},
                {"functionCall": {"name": "get_time", "args": {"city": "Springfield"}}},
                {"functionCall": {"id": "fc-7", "name": "noop"}}
            ]}}]
        });
        let response = parse_response(&data);
        assert_eq!(response.tool_calls.len(), 3);
        let first = &response.tool_calls[0];
        assert_eq!(first.name, "terminal");
        assert_eq!(first.arguments, json!({"command": "ls projects"}));
        assert_eq!(
            first.provider_data,
            Some(json!({"thoughtSignature": "sig-synthetic"}))
        );
        assert!(first.id.starts_with("call_"));
        assert!(response.tool_calls[1].provider_data.is_none());
        assert_ne!(first.id, response.tool_calls[1].id);
        assert_eq!(response.tool_calls[2].id, "fc-7");
        assert_eq!(response.tool_calls[2].arguments, json!({}));
        // Deterministic: the same reply yields the same ids.
        assert_eq!(parse_response(&data).tool_calls, response.tool_calls);

        // Replay: the signature goes back beside the call, and the result
        // is named after it.
        let body = build_body(&request(vec![
            Message::user("list"),
            Message::assistant("", vec![first.clone()]),
            Message::tool_result(first.id.clone(), r#"{"result": "site-cars"}"#),
        ]));
        let part = &body["contents"][1]["parts"][0];
        assert_eq!(part["thought_signature"], "sig-synthetic");
        assert_eq!(part["thoughtSignature"], "sig-synthetic");
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"]["name"],
            "terminal"
        );
    }

    #[test]
    fn safety_and_finish_reasons_become_notes() {
        let response = parse_response(&json!({
            "candidates": [{
                "finishReason": "SAFETY",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": true},
                    {"category": "HARM_CATEGORY_HARASSMENT", "blocked": false}
                ],
                "content": {"role": "model", "parts": []}
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 0}
        }));
        assert!(response.text.is_empty() && response.tool_calls.is_empty());
        let note = response.note.unwrap();
        assert!(note.contains("finish reason: SAFETY"), "{note}");
        assert!(note.contains("HARM_CATEGORY_DANGEROUS_CONTENT"), "{note}");
        assert!(!note.contains("HARASSMENT"), "{note}");
        assert!(note.contains("empty response (finishReason=SAFETY, parts=0)"));

        let response = parse_response(&json!({
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "blocked": true}]
            },
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 0}
        }));
        let note = response.note.unwrap();
        assert!(note.contains("prompt blocked (SAFETY)"), "{note}");
        assert!(note.contains("HARM_CATEGORY_HATE_SPEECH"), "{note}");
        assert_eq!(response.usage.input_tokens, 8);

        assert_eq!(
            parse_response(&json!({})).note.as_deref(),
            Some("no candidates returned by provider")
        );

        let response = parse_response(&json!({"candidates": [{
            "finishReason": "MAX_TOKENS",
            "content": {"parts": [{"text": "partial"}]}
        }]}));
        assert!(response.note.unwrap().contains("token limit"));

        let response = parse_response(&json!({"candidates": [{
            "finishReason": "STOP",
            "content": {"role": "model", "parts": []}
        }]}));
        assert_eq!(
            response.note.as_deref(),
            Some("empty response (finishReason=STOP, parts=0)")
        );
    }

    // --------------------------------------------- thought-signature retry

    #[test]
    fn unsigned_call_history_and_its_responses_are_stripped() {
        let contents = vec![
            json!({"role": "user", "parts": [{"text": "do it"}]}),
            json!({"role": "model", "parts": [
                {"text": "Writing."},
                {"functionCall": {"name": "write_file", "args": {"path": "/tmp/out.txt"}}}
            ]}),
            json!({"role": "function", "parts": [{"functionResponse": {
                "name": "write_file", "response": {"result": "ok"}
            }}]}),
            json!({"role": "model", "parts": [
                {"functionCall": {"name": "read_file", "args": {}},
                 "thoughtSignature": "sig-synthetic"}
            ]}),
            json!({"role": "function", "parts": [{"functionResponse": {
                "name": "read_file", "response": {"result": "x"}
            }}]}),
            json!({"role": "model", "parts": [
                {"functionCall": {"name": "write_file", "args": {}}}
            ]}),
            json!({"role": "function", "parts": [{"functionResponse": {
                "name": "write_file", "response": {"result": "ok"}
            }}]}),
            json!({"role": "model", "parts": [{"text": "Done"}]}),
        ];
        let (sanitized, dropped_calls, dropped_responses) =
            strip_unsigned_function_call_history(&contents);
        assert_eq!((dropped_calls, dropped_responses), (2, 2));
        let roles: Vec<&str> = sanitized
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "model", "model", "function", "model"]);
        assert_eq!(sanitized[1]["parts"], json!([{"text": "Writing."}]));
        assert_eq!(
            sanitized[2]["parts"][0]["thoughtSignature"],
            "sig-synthetic"
        );
        assert_eq!(sanitized[4]["parts"][0]["text"], "Done");

        let body = json!({"contents": contents, "tools": []});
        let retry = body_without_unsigned_calls(&body).unwrap();
        assert_eq!(retry["contents"], json!(sanitized));
        assert_eq!(retry["tools"], json!([]));

        // Nothing to strip: no retry.
        let signed = json!({"contents": [contents[0].clone(), contents[3].clone()]});
        assert!(body_without_unsigned_calls(&signed).is_none());
    }

    #[test]
    fn detects_the_missing_signature_rejection() {
        assert!(is_missing_thought_signature_error(
            r#"{"error":{"code":400,"message":"Function call is missing a thought_signature in functionCall parts."}}"#
        ));
        assert!(is_missing_thought_signature_error(
            "missing thoughtSignature"
        ));
        assert!(!is_missing_thought_signature_error("bad tool schema"));
    }

    // -------------------------------------------------------- error mapping

    #[test]
    fn errors_map_to_typed_kinds() {
        let none = HeaderMap::new();
        let status = |code: u16| StatusCode::from_u16(code).unwrap();

        let quota = r#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED",
            "details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"30s"}]}}"#;
        assert_eq!(
            gemini_error(status(429), &none, quota),
            ProviderError::RateLimited {
                retry_after: Some(Duration::from_secs(30))
            }
        );
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("2"));
        assert_eq!(
            gemini_error(status(429), &headers, quota),
            ProviderError::RateLimited {
                retry_after: Some(Duration::from_secs(2))
            }
        );

        let bad_key = r#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT",
            "details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}"#;
        match gemini_error(status(400), &none, bad_key) {
            ProviderError::Auth(message) => assert!(message.contains("API key not valid")),
            other => panic!("unexpected {other:?}"),
        }

        let overflow = r#"{"error":{"code":400,"message":"The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."}}"#;
        assert_eq!(
            gemini_error(status(400), &none, overflow),
            ProviderError::ContextOverflow {
                message: "HTTP 400: The input token count (1200000) exceeds the maximum number of tokens allowed (1048576).".into(),
                limit: Some(1_048_576)
            }
        );
        assert!(matches!(
            gemini_error(
                status(404),
                &none,
                r#"{"error":{"message":"models/x is not found"}}"#
            ),
            ProviderError::NotFound(_)
        ));
        assert!(matches!(
            gemini_error(status(403), &none, "denied"),
            ProviderError::Auth(_)
        ));
        assert!(matches!(
            gemini_error(status(402), &none, "billing"),
            ProviderError::Billing { .. }
        ));
        assert!(matches!(
            gemini_error(status(400), &none, r#"{"error":{"message":"bad schema"}}"#),
            ProviderError::InvalidRequest(_)
        ));
        assert!(matches!(
            gemini_error(status(503), &none, "overloaded"),
            ProviderError::Transient(_)
        ));
    }
}
