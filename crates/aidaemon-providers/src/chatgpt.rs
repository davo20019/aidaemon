//! [`ChatGpt`]: models on a ChatGPT subscription, through the Codex
//! backend's Responses API (`chatgpt.com/backend-api/codex/responses`).
//!
//! Calls draw on the user's ChatGPT Plus/Pro/Business plan instead of a
//! metered API key. Credentials come from a [`ChatGptAuth`]: the app owns
//! the OAuth login and token storage, the provider only asks for a usable
//! token and reports one the backend rejected.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use aidaemon_core::{
    Message, ModelRequest, ModelResponse, Provider, ProviderError, ResponseFormat, Role,
    StreamEvent, ToolCall, ToolChoice, ToolSpec, Usage, collect_stream,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::{Map, Value, json};

use crate::sse::SseEvent;
use crate::transport::{
    ConfigError, SseDecoder, body_excerpt, build_client, drive_response, map_reqwest_error,
    map_status, parse_arguments, raw_arguments,
};

/// The Codex backend root.
pub const CHATGPT_DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// Models reachable with a ChatGPT subscription. The backend has no model
/// listing; unknown ids still pass through, and the API decides.
pub const CHATGPT_KNOWN_MODELS: &[&str] = &[
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.6-pro",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
];

/// Reasoning-effort values the backend accepts, weakest to strongest. Not
/// enforced: each model supports its own subset and the API decides.
pub const CHATGPT_REASONING_EFFORTS: &[&str] = &[
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
];

/// Identifies the calling tool to OpenAI; third-party clients declare
/// themselves instead of impersonating the Codex CLI.
const DEFAULT_ORIGINATOR: &str = "aidaemon";

/// An access token and the ChatGPT account it acts for.
#[derive(Clone, PartialEq, Eq)]
pub struct ChatGptCredentials {
    pub access_token: String,
    pub account_id: String,
}

impl std::fmt::Debug for ChatGptCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatGptCredentials")
            .field("access_token", &"<redacted>")
            .field("account_id", &self.account_id)
            .finish()
    }
}

/// Where [`ChatGpt`] gets its credentials. Implementations refresh near
/// expiry themselves and serialize refresh-token rotation.
#[async_trait]
pub trait ChatGptAuth: Send + Sync {
    /// Credentials usable now.
    async fn credentials(&self) -> Result<ChatGptCredentials, String>;

    /// The backend rejected `rejected_token` before its expiry (revoked):
    /// refresh once and return the new credentials.
    async fn refresh_after_rejection(
        &self,
        rejected_token: &str,
    ) -> Result<ChatGptCredentials, String>;

    /// The refreshed token was rejected too; the login needs repair.
    async fn mark_rejected(&self, _token: &str, _detail: &str) {}
}

/// Fixed credentials that never refresh, for tests and short-lived tools.
pub struct StaticChatGptAuth(pub ChatGptCredentials);

#[async_trait]
impl ChatGptAuth for StaticChatGptAuth {
    async fn credentials(&self) -> Result<ChatGptCredentials, String> {
        Ok(self.0.clone())
    }

    async fn refresh_after_rejection(&self, _: &str) -> Result<ChatGptCredentials, String> {
        Err("static credentials cannot be refreshed".into())
    }
}

/// A ChatGPT-subscription model backend.
///
/// - Leading system messages become `instructions`, with a stable
///   `prompt_cache_key` derived from them so requests sharing them land on
///   the same cache shard; later system messages (hook notes, mid-run
///   directives) stay in place as `developer` input.
/// - `store` is always false: the app owns the history, so the encrypted
///   reasoning of a tool-calling reply rides on its first tool call
///   (`REASONING_ITEMS_KEY`) and is replayed beside it.
/// - No output cap is sent: the backend rejects `max_output_tokens`.
/// - A 401 refreshes the token once and replays the request (inference is
///   side-effect free); a second 401 is reported to the [`ChatGptAuth`].
/// - The reply streams; reasoning summaries become
///   [`ModelResponse::reasoning`].
pub struct ChatGpt {
    client: reqwest::Client,
    base_url: String,
    auth: Arc<dyn ChatGptAuth>,
    reasoning_effort: Option<String>,
    originator: String,
    user_agent: String,
    timeout: Duration,
}

/// Builder for [`ChatGpt`].
pub struct ChatGptBuilder {
    base_url: String,
    auth: Option<Arc<dyn ChatGptAuth>>,
    reasoning_effort: Option<String>,
    originator: String,
    user_agent: Option<String>,
    timeout: Duration,
}

impl ChatGpt {
    pub fn builder() -> ChatGptBuilder {
        ChatGptBuilder {
            base_url: CHATGPT_DEFAULT_BASE_URL.into(),
            auth: None,
            reasoning_effort: None,
            originator: DEFAULT_ORIGINATOR.into(),
            user_agent: None,
            timeout: Duration::from_secs(600),
        }
    }
}

impl std::fmt::Debug for ChatGpt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatGpt")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl ChatGptBuilder {
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        let url = url.trim().trim_end_matches('/');
        if !url.is_empty() {
            self.base_url = url.to_string();
        }
        self
    }

    pub fn auth(mut self, auth: impl ChatGptAuth + 'static) -> Self {
        self.auth = Some(Arc::new(auth));
        self
    }

    pub fn auth_arc(mut self, auth: Arc<dyn ChatGptAuth>) -> Self {
        self.auth = Some(auth);
        self
    }

    /// The default effort for requests that set none
    /// ([`aidaemon_core::RequestOptions::reasoning_effort`]).
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        let effort = effort.into();
        self.reasoning_effort = (!effort.trim().is_empty()).then_some(effort);
        self
    }

    /// The `originator` header naming the calling tool.
    pub fn originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// The whole request's timeout (default 600s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn build(self) -> Result<ChatGpt, ConfigError> {
        let auth = self
            .auth
            .ok_or(ConfigError::Missing("ChatGPT credentials (auth)"))?;
        Ok(ChatGpt {
            client: build_client(Duration::from_secs(30))?,
            base_url: self.base_url,
            auth,
            reasoning_effort: self.reasoning_effort,
            user_agent: self.user_agent.unwrap_or_else(|| {
                format!(
                    "{}/{} ({}; {})",
                    self.originator,
                    env!("CARGO_PKG_VERSION"),
                    std::env::consts::OS,
                    std::env::consts::ARCH
                )
            }),
            originator: self.originator,
            timeout: self.timeout,
        })
    }
}

impl ChatGpt {
    async fn send(
        &self,
        credentials: &ChatGptCredentials,
        body: &Value,
    ) -> Result<reqwest::Response, ProviderError> {
        let mut post = self.client.post(format!("{}/responses", self.base_url));
        // The backend routes a conversation to its prompt cache by the
        // `session-id` header, not by `prompt_cache_key` alone: without it,
        // calls in one run land on cold shards (as Codex does, both carry
        // the same id).
        if let Some(key) = session_id(body) {
            post = post.header("session-id", key);
        }
        post.timeout(self.timeout)
            .bearer_auth(&credentials.access_token)
            .header("chatgpt-account-id", &credentials.account_id)
            .header("OpenAI-Beta", "responses=experimental")
            .header("originator", &self.originator)
            .header("User-Agent", &self.user_agent)
            .header("accept", "text/event-stream")
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)
    }

    /// Authenticate, send, and recover once from a revoked token.
    async fn open(&self, request: &ModelRequest) -> Result<reqwest::Response, ProviderError> {
        let auth_error = |error: String| ProviderError::Auth(error);
        let mut credentials = self.auth.credentials().await.map_err(auth_error)?;
        let mut body = build_body(request, self.reasoning_effort.as_deref());
        let mut response = self.send(&credentials, &body).await?;
        // The backend is authoritative about token validity: a token can be
        // revoked before its expiry.
        if response.status().as_u16() == 401 {
            let rejected = credentials.access_token.clone();
            let _ = response.bytes().await;
            credentials = self
                .auth
                .refresh_after_rejection(&rejected)
                .await
                .map_err(auth_error)?;
            response = self.send(&credentials, &body).await?;
            if response.status().as_u16() == 401 {
                let detail = annotate(401, &response.text().await.unwrap_or_default());
                self.auth
                    .mark_rejected(&credentials.access_token, &detail)
                    .await;
                return Err(ProviderError::Auth(detail));
            }
        }
        // Encrypted reasoning is bound to the model that wrote it, so a
        // transcript that crossed models can be refused. It is an optimization:
        // the request stands without it.
        if response.status().as_u16() == 400
            && !request.options.single_attempt
            && strip_reasoning(&mut body)
        {
            let _ = response.bytes().await;
            response = self.send(&credentials, &body).await?;
        }
        let status = response.status();
        if !status.is_success() {
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            return Err(chatgpt_error(status, &headers, &body));
        }
        Ok(response)
    }
}

#[async_trait]
impl Provider for ChatGpt {
    fn name(&self) -> &str {
        "chatgpt"
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        collect_stream(self.stream(request)).await
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        Box::pin(
            stream::once(self.open(request)).flat_map(|opened| match opened {
                Ok(response) => drive_response(response, CodexDecoder::default()),
                Err(error) => stream::iter([Err(error)]).boxed(),
            }),
        )
    }
}

/// The client versions the model listing asks as, in turn. The backend
/// hides models that need a newer client than the one named (0.99 saw only
/// an internal model; 0.166 saw the GPT-6 family), so a far-future version
/// comes first, and a version known to answer follows in case the backend
/// refuses it.
pub const CHATGPT_LIST_CLIENT_VERSIONS: &[&str] = &["999.0.0", "0.166.0"];

/// A model the ChatGPT backend offers the signed-in account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatGptModel {
    pub id: String,
    /// The name the backend shows people, when it gives one.
    pub display_name: Option<String>,
    /// The backend's one-line description, when it gives one.
    pub description: Option<String>,
    /// Shown in a model picker (the backend can list a model it hides).
    pub listed: bool,
}

impl ChatGpt {
    /// The models the backend offers this account
    /// (`GET {base}/models?client_version=..`), with a 401 refreshed once.
    pub async fn list_models(&self) -> Result<Vec<ChatGptModel>, ProviderError> {
        let mut last_error = None;
        for version in CHATGPT_LIST_CLIENT_VERSIONS {
            match self.list_models_as(version).await {
                Ok(models) if models.iter().any(|m| m.listed) => return Ok(models),
                Ok(models) => {
                    last_error = Some(ProviderError::NotFound(format!(
                        "model listing as client {version} showed no models to pick ({})",
                        models
                            .iter()
                            .map(|m| m.id.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )))
                }
                Err(error @ ProviderError::Auth(_)) => return Err(error),
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| ProviderError::NotFound("no model listing".into())))
    }

    async fn list_models_as(&self, version: &str) -> Result<Vec<ChatGptModel>, ProviderError> {
        let auth_error = |error: String| ProviderError::Auth(error);
        let mut credentials = self.auth.credentials().await.map_err(auth_error)?;
        let mut response = self.get_models(&credentials, version).await?;
        if response.status().as_u16() == 401 {
            let rejected = credentials.access_token.clone();
            let _ = response.bytes().await;
            credentials = self
                .auth
                .refresh_after_rejection(&rejected)
                .await
                .map_err(auth_error)?;
            response = self.get_models(&credentials, version).await?;
        }
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_status(status, &headers, &body));
        }
        Ok(parse_model_listing(&body))
    }

    /// The plan's usage limits as the backend reports them
    /// (`GET .../backend-api/wham/usage`, what Codex's `/status` shows),
    /// with a 401 refreshed once.
    pub async fn usage(&self) -> Result<crate::usage::AccountUsage, ProviderError> {
        let auth_error = |error: String| ProviderError::Auth(error);
        let url = crate::usage::chatgpt_usage_url(&self.base_url);
        let get = |credentials: ChatGptCredentials| {
            self.client
                .get(&url)
                .timeout(Duration::from_secs(10))
                .bearer_auth(credentials.access_token)
                .header("chatgpt-account-id", credentials.account_id)
                .header("originator", &self.originator)
                .header("User-Agent", &self.user_agent)
                .send()
        };
        let credentials = self.auth.credentials().await.map_err(auth_error)?;
        let rejected = credentials.access_token.clone();
        let mut response = get(credentials).await.map_err(map_reqwest_error)?;
        if response.status().as_u16() == 401 {
            let _ = response.bytes().await;
            let credentials = self
                .auth
                .refresh_after_rejection(&rejected)
                .await
                .map_err(auth_error)?;
            response = get(credentials).await.map_err(map_reqwest_error)?;
        }
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(map_status(status, &headers, &body));
        }
        crate::usage::parse_chatgpt_usage(&body, SystemTime::now()).ok_or_else(|| {
            ProviderError::InvalidRequest(format!(
                "the usage answer had no limits: {}",
                body_excerpt(&body)
            ))
        })
    }

    async fn get_models(
        &self,
        credentials: &ChatGptCredentials,
        version: &str,
    ) -> Result<reqwest::Response, ProviderError> {
        self.client
            .get(format!("{}/models?client_version={version}", self.base_url))
            .timeout(Duration::from_secs(30))
            .bearer_auth(&credentials.access_token)
            .header("chatgpt-account-id", &credentials.account_id)
            .header("originator", &self.originator)
            .header("User-Agent", &self.user_agent)
            .send()
            .await
            .map_err(map_reqwest_error)
    }
}

/// The models in a listing body: `{"models": [{"slug", ..}]}` as the Codex
/// backend answers, or `{"data": [{"id"}]}` as OpenAI's API does.
pub fn parse_model_listing(body: &str) -> Vec<ChatGptModel> {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return Vec::new();
    };
    let text = |m: &Value, key: &str| {
        m[key]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let mut seen = HashSet::new();
    value["models"]
        .as_array()
        .or_else(|| value["data"].as_array())
        .into_iter()
        .flatten()
        .filter_map(|m| {
            let id = text(m, "slug").or_else(|| text(m, "id"))?;
            seen.insert(id.clone()).then(|| ChatGptModel {
                display_name: text(m, "display_name"),
                description: text(m, "description"),
                listed: m["visibility"]
                    .as_str()
                    .is_none_or(|v| v.eq_ignore_ascii_case("list")),
                id,
            })
        })
        .collect()
}

/// Make an auth rejection legible: the backend's own message is terse, and
/// a bad token needs the user to sign in again.
fn annotate(status: u16, body: &str) -> String {
    let body = body_excerpt(body);
    match status {
        401 => format!("ChatGPT subscription auth rejected; sign in again. Details: {body}"),
        _ => body,
    }
}

/// A rejected call. A 429 is either throttling (worth a short wait) or the
/// plan's usage limit (`usage_limit_reached`, or a rate-limit window at
/// 100%), which only its reset cures; the plan not covering Codex at all
/// (`usage_not_included`) is a billing failure.
fn chatgpt_error(
    status: reqwest::StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> ProviderError {
    if status.as_u16() == 429
        && let Some(error) = plan_limit(headers, body, SystemTime::now())
    {
        tracing::warn!(%error, "ChatGPT plan limit");
        return error;
    }
    map_status(status, headers, body)
}

/// The plan-limit error a 429 states, if it is one. The reset comes from the
/// body (`resets_at`, epoch seconds, or `resets_in_seconds`), else from the
/// exhausted window's `x-codex-<window>-reset-at` header.
fn plan_limit(
    headers: &reqwest::header::HeaderMap,
    body: &str,
    now: SystemTime,
) -> Option<ProviderError> {
    let value: Option<Value> = serde_json::from_str(body).ok();
    let error = value.as_ref().and_then(|v| v.get("error"));
    let field = |name: &str| error.and_then(|e| e.get(name));
    let kind = field("type")
        .or_else(|| field("code"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let message = body_excerpt(body);
    if kind == "usage_not_included" {
        return Some(ProviderError::Billing {
            message: format!("This ChatGPT plan does not include Codex usage: {message}"),
            affordable_tokens: None,
        });
    }
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite())
    };
    let epoch =
        |secs: f64| (secs > 0.0).then(|| SystemTime::UNIX_EPOCH + Duration::from_secs_f64(secs));
    // The window that is used up, and when it resets.
    let exhausted_window = ["primary", "secondary"].iter().find_map(|window| {
        let used = header(&format!("x-codex-{window}-used-percent"))?;
        (used >= 100.0).then(|| header(&format!("x-codex-{window}-reset-at")).and_then(epoch))
    });
    if kind != "usage_limit_reached" && exhausted_window.is_none() {
        return None;
    }
    let resets_at = field("resets_at")
        .and_then(Value::as_f64)
        .and_then(epoch)
        .or_else(|| {
            field("resets_in_seconds")
                .and_then(Value::as_f64)
                .filter(|secs| secs.is_finite() && *secs >= 0.0)
                .map(|secs| now + Duration::from_secs_f64(secs))
        })
        .or(exhausted_window.flatten());
    let plan = field("plan_type")
        .and_then(Value::as_str)
        .map(|plan| format!(" ({plan} plan)"))
        .unwrap_or_default();
    Some(ProviderError::QuotaExhausted {
        message: format!("ChatGPT usage limit reached{plan}: {message}"),
        resets_at,
        per_model: false,
        in_stream: false,
    })
}

/// The [`ToolCall::provider_data`] key holding a reply's encrypted reasoning.
pub const REASONING_ITEMS_KEY: &str = "chatgpt_reasoning_items";

/// A stable cache-routing key. The seed is hashed so a conversation id never
/// leaves the machine.
fn prompt_cache_key(seed: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(seed.as_bytes());
    let hex: String = digest[..12].iter().map(|b| format!("{b:02x}")).collect();
    format!("aidaemon-{hex}")
}

/// The `session-id` header for a body: its cache-routing key.
fn session_id(body: &Value) -> Option<&str> {
    body.get("prompt_cache_key").and_then(Value::as_str)
}

/// The Responses `input` items and `instructions` for a transcript.
///
/// The leading system messages become `instructions`; later ones stay in
/// place as `developer` input, since rewriting `instructions` (which precede
/// every item) would invalidate the prompt cache for the whole request.
/// A tool result whose call is not pending is dropped: the endpoint rejects
/// it with a 400.
fn build_input(messages: &[Message]) -> (String, Vec<Value>) {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    let mut pending = HashSet::new();
    for message in messages {
        match message.role {
            Role::System => {
                if message.content.is_empty() {
                } else if input.is_empty() {
                    instructions.push(message.content.clone());
                } else {
                    input.push(json!({
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": message.content}],
                    }));
                }
            }
            Role::Tool => {
                let call_id = message.tool_call_id.as_deref().unwrap_or_default();
                if !pending.remove(call_id) {
                    continue;
                }
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.content,
                }));
            }
            Role::Assistant => {
                input.extend(
                    message
                        .tool_calls
                        .iter()
                        .filter_map(|call| call.provider_data.as_ref())
                        .filter_map(|data| data.get(REASONING_ITEMS_KEY))
                        .filter_map(Value::as_array)
                        .flatten()
                        .cloned(),
                );
                if !message.content.is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": message.content}],
                    }));
                }
                for call in &message.tool_calls {
                    let arguments = raw_arguments(&call.arguments)
                        .map(str::to_string)
                        .unwrap_or_else(|| call.arguments.to_string());
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": arguments,
                    }));
                    if !call.id.is_empty() {
                        pending.insert(call.id.clone());
                    }
                }
            }
            Role::User => {
                let mut content = vec![json!({"type": "input_text", "text": message.content})];
                content.extend(
                    message
                        .images
                        .iter()
                        .map(|image| json!({"type": "input_image", "image_url": image.to_url()})),
                );
                input.push(json!({"type": "message", "role": "user", "content": content}));
            }
        }
    }
    (instructions.join("\n\n"), input)
}

fn build_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool_parameters(&tool.parameters),
            })
        })
        .collect()
}

/// A tool's parameters in the schema subset the backend compiles. Newer
/// models compile every tool schema strictly and refuse the whole request
/// over keywords outside that subset (HTTP 400, "Unimplemented keys"):
/// `uniqueItems`, and `not` (which gpt-6.2 started refusing). The tool
/// still checks its own arguments, so dropping them loses nothing. A
/// combinator (`allOf`/`anyOf`/`oneOf`) left with only empty branches is
/// dropped too. Property names are never filtered.
fn tool_parameters(schema: &Value) -> Value {
    let mut schema = sanitize_schema(schema, false);
    every_property_required(&mut schema);
    schema
}

/// Strict compilation makes the model send every property. One the tool
/// leaves optional becomes required and nullable, so the model can say
/// "unset" with `null` (which the dispatcher drops) instead of inventing a
/// value: live 2026-11-20, `read_file` got `start_line: 0, end_line: 0,
/// tail_lines: 1` and refused the call. Nested objects alike.
fn every_property_required(schema: &mut Value) {
    let Some(map) = schema.as_object_mut() else {
        return;
    };
    if matches!(map.get("properties"), Some(Value::Object(_))) {
        let required: Vec<String> = map
            .get("required")
            .and_then(Value::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        let Some(Value::Object(properties)) = map.get_mut("properties") else {
            return;
        };
        let mut names = Vec::with_capacity(properties.len());
        for (name, property) in properties.iter_mut() {
            every_property_required(property);
            if !required.contains(name) {
                make_nullable(property);
            }
            names.push(Value::String(name.clone()));
        }
        map.insert("required".into(), Value::Array(names));
    }
    if let Some(items) = map.get_mut("items") {
        every_property_required(items);
    }
}

/// `property` also accepts `null`.
fn make_nullable(property: &mut Value) {
    let Some(map) = property.as_object_mut() else {
        return;
    };
    match map.get_mut("type") {
        Some(Value::String(kind)) if kind != "null" => {
            let kind = kind.clone();
            map.insert("type".into(), json!([kind, "null"]));
        }
        Some(Value::Array(kinds)) => {
            if !kinds.iter().any(|k| k == "null") {
                kinds.push(json!("null"));
            }
        }
        Some(_) => {}
        None => {
            let original = Value::Object(std::mem::take(map));
            map.insert("anyOf".into(), json!([original, {"type": "null"}]));
            return;
        }
    }
    if let Some(Value::Array(values)) = map.get_mut("enum")
        && !values.iter().any(Value::is_null)
    {
        values.push(Value::Null);
    }
}

const UNCOMPILED_KEYWORDS: [&str; 2] = ["uniqueItems", "not"];

fn sanitize_schema(schema: &Value, names: bool) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                if !names && UNCOMPILED_KEYWORDS.contains(&key.as_str()) {
                    continue;
                }
                let child_names =
                    !names && matches!(key.as_str(), "properties" | "$defs" | "definitions");
                let value = sanitize_schema(value, child_names);
                if !names
                    && matches!(key.as_str(), "allOf" | "anyOf" | "oneOf")
                    && value
                        .as_array()
                        .is_some_and(|branches| branches.iter().all(is_empty_object))
                {
                    continue;
                }
                out.insert(key.clone(), value);
            }
            Value::Object(out)
        }
        Value::Array(items) => {
            Value::Array(items.iter().map(|v| sanitize_schema(v, false)).collect())
        }
        other => other.clone(),
    }
}

fn is_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(Map::is_empty)
}

fn build_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Specific(name) => json!({"type": "function", "name": name}),
    }
}

/// The body for `POST /responses`. The Codex backend allowlists parameters
/// and rejects others with a 400, `max_output_tokens` among them, so no
/// output cap is sent.
fn build_body(request: &ModelRequest, default_effort: Option<&str>) -> Value {
    let (instructions, input) = build_input(&request.messages);
    let mut body = Map::new();
    body.insert("model".into(), json!(request.model));
    body.insert("input".into(), json!(input));
    body.insert("stream".into(), json!(true));
    body.insert("store".into(), json!(false));
    if !instructions.is_empty() {
        body.insert(
            "prompt_cache_key".into(),
            json!(prompt_cache_key(
                request
                    .options
                    .cache_key
                    .as_deref()
                    .unwrap_or(&instructions)
            )),
        );
        body.insert("instructions".into(), json!(instructions));
    }
    if !request.tools.is_empty() {
        body.insert("tools".into(), json!(build_tools(&request.tools)));
        body.insert(
            "tool_choice".into(),
            build_tool_choice(&request.tool_choice),
        );
    }
    let effort = request
        .options
        .reasoning_effort
        .as_deref()
        .or(default_effort);
    if let Some(effort) = effort {
        // The backend spells "reasoning off" `none`.
        let effort = if effort == "off" { "none" } else { effort };
        body.insert("reasoning".into(), json!({"effort": effort}));
        if effort != "none" {
            body.insert("include".into(), json!(["reasoning.encrypted_content"]));
        }
    }
    match &request.options.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject => {
            body.insert("text".into(), json!({"format": {"type": "json_object"}}));
        }
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            body.insert(
                "text".into(),
                json!({"format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }}),
            );
        }
    }
    Value::Object(body)
}

/// Removes replayed reasoning from a request body; true when there was any.
fn strip_reasoning(body: &mut Value) -> bool {
    let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return false;
    };
    let before = input.len();
    input.retain(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"));
    input.len() != before
}

/// Accumulates the Responses event stream.
///
/// The Codex backend ends with `response.completed` carrying `"output": []`
/// (it echoes nothing when `store` is false), so the real payload is the
/// items from `response.output_item.done`.
#[derive(Default)]
struct CodexDecoder {
    delta_text: String,
    items: Vec<Value>,
    completed: Option<Value>,
    failure: Option<ProviderError>,
}

impl SseDecoder for CodexDecoder {
    fn on_event(
        &mut self,
        event: &SseEvent,
        deltas: &mut Vec<String>,
    ) -> Result<bool, ProviderError> {
        if event.data == "[DONE]" {
            return Ok(false);
        }
        // One unreadable frame is not worth failing the turn; a stream that
        // never completes is caught in `finish`.
        let Ok(event) = serde_json::from_str::<Value>(&event.data) else {
            return Ok(false);
        };
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    self.delta_text.push_str(delta);
                    deltas.push(delta.to_string());
                }
            }
            "response.output_item.done" => {
                if let Some(item) = event.get("item") {
                    self.items.push(item.clone());
                }
            }
            "response.completed" => {
                self.completed = event.get("response").cloned();
                return Ok(true);
            }
            "response.failed" | "response.incomplete" => {
                let message = event
                    .pointer("/response/error/message")
                    .or_else(|| event.pointer("/response/incomplete_details/reason"))
                    .and_then(Value::as_str)
                    .unwrap_or("the model stopped before completing the response");
                self.failure = Some(stream_failure(&event, message));
            }
            "error" => {
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified stream error");
                self.failure = Some(stream_failure(&event, message));
            }
            _ => {}
        }
        Ok(false)
    }

    fn finish(&mut self, _terminated: bool) -> Result<ModelResponse, ProviderError> {
        let items = std::mem::take(&mut self.items);
        let delta_text = std::mem::take(&mut self.delta_text);
        if let Some(completed) = self.completed.take() {
            let output = match completed.get("output").and_then(Value::as_array) {
                Some(output) if !output.is_empty() => output.clone(),
                _ => items,
            };
            let mut response = parse_output(&output);
            if response.text.is_empty() && response.tool_calls.is_empty() {
                response.text = delta_text;
            }
            response.usage = parse_usage(&completed);
            return Ok(response);
        }
        if let Some(failure) = self.failure.take() {
            return Err(failure);
        }
        // The stream died before completing: hand back what arrived.
        let mut response = parse_output(&items);
        if response.text.is_empty() {
            response.text = delta_text;
        }
        if response.text.is_empty() && response.tool_calls.is_empty() {
            return Err(ProviderError::Malformed {
                message: "Codex stream ended with no response events".into(),
                parse: false,
            });
        }
        response.note = Some(
            "Codex stream ended without a completion event; response may be truncated.".into(),
        );
        Ok(response)
    }
}

/// A typed error for a failure event, so recovery can tell overload and
/// rate limits apart from refusals.
fn stream_failure(event: &Value, message: &str) -> ProviderError {
    let field = |name: &str| {
        event
            .get(name)
            .or_else(|| event.pointer(&format!("/error/{name}")))
            .or_else(|| event.pointer(&format!("/response/error/{name}")))
    };
    let code = field("code")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    let status = field("status").and_then(Value::as_u64);
    tracing::warn!(code = %code, status = ?status, event = %event, "Codex stream failure event");
    // A stated HTTP status leads the message in the transport's own
    // `HTTP <status>: ` form, so the daemon reads the status, not the words.
    let message = match status {
        Some(status) => format!("HTTP {status}: Codex stream failed: {message}"),
        None => format!("Codex stream failed: {message}"),
    };
    // The Responses API's documented failure codes, compared exactly.
    let overloaded = matches!(
        code.as_str(),
        "server_is_overloaded" | "slow_down" | "server_error" | "service_unavailable"
    )
        // A failure event with neither a status nor a code carries no
        // structured signal at all; its "overloaded" wording is then the
        // only one, so it alone is read, and only in that case.
        || (code.is_empty()
            && status.is_none()
            && message.to_ascii_lowercase().contains("overloaded"));
    match code.as_str() {
        "usage_limit_reached" => {
            let resets_at = field("resets_at")
                .and_then(Value::as_f64)
                .filter(|secs| *secs > 0.0)
                .map(|secs| SystemTime::UNIX_EPOCH + Duration::from_secs_f64(secs));
            return ProviderError::QuotaExhausted {
                message,
                resets_at,
                per_model: false,
                in_stream: true,
            };
        }
        "context_length_exceeded" => {
            return ProviderError::ContextOverflow {
                message,
                limit: None,
            };
        }
        "insufficient_quota" => {
            return ProviderError::Billing {
                message,
                affordable_tokens: None,
            };
        }
        _ => {}
    }
    match status {
        Some(429) => ProviderError::RateLimited { retry_after: None },
        Some(408 | 500 | 502 | 503 | 504) => ProviderError::Transient(message),
        _ if code == "rate_limit_exceeded" => ProviderError::RateLimited { retry_after: None },
        _ if overloaded => ProviderError::Transient(message),
        _ => ProviderError::Other(message),
    }
}

fn parse_output(items: &[Value]) -> ModelResponse {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut reasoning_items = Vec::new();
    let mut tool_calls = Vec::new();
    let parts = |item: &Value, field: &str| -> Vec<String> {
        item.get(field)
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => text.push_str(&parts(item, "content").concat()),
            Some("reasoning") => {
                reasoning.push_str(&parts(item, "summary").concat());
                if let Some(encrypted) = item.get("encrypted_content").filter(|v| v.is_string()) {
                    // No `id`: with `store` false the backend has no such item.
                    reasoning_items.push(json!({
                        "type": "reasoning",
                        "summary": item.get("summary").cloned().unwrap_or_else(|| json!([])),
                        "encrypted_content": encrypted,
                    }));
                }
            }
            Some("function_call") => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                tool_calls.push(ToolCall::new(id, name, parse_arguments(arguments)));
            }
            _ => {}
        }
    }
    if let Some(first) = tool_calls
        .first_mut()
        .filter(|_| !reasoning_items.is_empty())
    {
        first.provider_data = Some(json!({ REASONING_ITEMS_KEY: reasoning_items }));
    }
    ModelResponse {
        text,
        tool_calls,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        ..Default::default()
    }
}

fn parse_usage(completed: &Value) -> Usage {
    let usage = completed.get("usage");
    let count = |pointer: &str| {
        usage
            .and_then(|u| u.pointer(pointer))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    Usage {
        input_tokens: count("/input_tokens"),
        output_tokens: count("/output_tokens"),
        cached_input_tokens: count("/input_tokens_details/cached_tokens"),
        cache_creation_tokens: 0,
    }
}

#[cfg(test)]
mod strict_tool_tests {
    use super::*;

    #[test]
    fn optional_properties_become_required_and_nullable() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "mode": {"type": "string", "enum": ["overwrite", "append"]},
                "opts": {"type": "object", "properties": {"deep": {"type": "boolean"}}, "required": []}
            },
            "required": ["path"],
            "additionalProperties": false
        });
        let out = tool_parameters(&schema);
        let required: Vec<&str> = out["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(required.len(), 4);
        assert_eq!(out["properties"]["path"]["type"], "string");
        assert_eq!(
            out["properties"]["start_line"]["type"],
            json!(["integer", "null"])
        );
        assert_eq!(
            out["properties"]["mode"]["enum"],
            json!(["overwrite", "append", null])
        );
        assert_eq!(
            out["properties"]["opts"]["properties"]["deep"]["type"],
            json!(["boolean", "null"])
        );
        assert_eq!(out["properties"]["opts"]["required"], json!(["deep"]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_listing_reads_both_shapes() {
        let codex = r#"{"models":[
            {"slug":"gpt-6.2-sol","display_name":"GPT-6.2 Sol","description":"Newest.","visibility":"list"},
            {"slug":"gpt-6-sol","visibility":"hide"},
            {"slug":"gpt-6.2-sol"}
        ]}"#;
        let models = parse_model_listing(codex);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].display_name.as_deref(), Some("GPT-6.2 Sol"));
        assert!(models[0].listed && !models[1].listed);
        let api = r#"{"data":[{"id":"gpt-6-luna"}]}"#;
        assert_eq!(parse_model_listing(api)[0].id, "gpt-6-luna");
        assert!(parse_model_listing("not json").is_empty());
    }
    use aidaemon_core::{Image, RequestOptions};

    fn request(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: "gpt-6-sol".into(),
            messages,
            ..Default::default()
        }
    }

    fn decode(events: &[Value]) -> Result<ModelResponse, ProviderError> {
        let mut decoder = CodexDecoder::default();
        let mut deltas = Vec::new();
        for event in events {
            let done = decoder.on_event(
                &SseEvent {
                    event: None,
                    data: event.to_string(),
                },
                &mut deltas,
            )?;
            if done {
                return decoder.finish(true);
            }
        }
        decoder.finish(false)
    }

    #[test]
    fn leading_system_messages_are_instructions_and_later_ones_stay_in_place() {
        let (instructions, input) = build_input(&[
            Message::system("You are terse."),
            Message::system("Be safe."),
            Message::user("hi"),
            Message::hook_note("[Context Coverage] …"),
        ]);
        assert_eq!(instructions, "You are terse.\n\nBe safe.");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "developer");
        assert_eq!(input[1]["content"][0]["text"], "[Context Coverage] …");
    }

    #[test]
    fn tool_calls_and_results_round_trip_and_orphans_are_dropped() {
        let call = ToolCall::new("c1", "terminal", json!({"command": "ls"}));
        let (_, input) = build_input(&[
            Message::user("list"),
            Message::assistant("Looking.", vec![call]),
            Message::tool_result("c1", "README.md"),
            Message::tool_result("ghost", "never asked"),
        ]);
        assert_eq!(input[1]["content"][0]["text"], "Looking.");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["arguments"], r#"{"command":"ls"}"#);
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["output"], "README.md");
        assert_eq!(input.len(), 4);
    }

    #[test]
    fn user_images_become_input_images() {
        let image = Image::base64("image/png", "AQID");
        let (_, input) = build_input(&[Message::user("what is this").with_images(vec![image])]);
        assert_eq!(input[0]["content"][1]["type"], "input_image");
        assert!(
            input[0]["content"][1]["image_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn the_body_streams_stores_nothing_and_sends_no_output_cap() {
        let mut req = request(vec![Message::system("sys"), Message::user("hi")]);
        req.max_output_tokens = Some(500);
        let body = build_body(&req, None);
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert!(body.get("max_output_tokens").is_none());
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["prompt_cache_key"], json!(prompt_cache_key("sys")));
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn reasoning_effort_comes_from_the_request_else_the_default_and_off_is_none() {
        let mut req = request(vec![Message::user("hi")]);
        assert_eq!(
            build_body(&req, Some("high"))["reasoning"]["effort"],
            "high"
        );
        req.options = RequestOptions {
            reasoning_effort: Some("off".into()),
            ..Default::default()
        };
        assert_eq!(
            build_body(&req, Some("high"))["reasoning"]["effort"],
            "none"
        );
        assert!(
            build_body(&request(vec![]), None)
                .get("reasoning")
                .is_none()
        );
    }

    #[test]
    fn tools_choice_and_json_schema_map() {
        let mut req = request(vec![Message::user("hi")]);
        req.tools = vec![ToolSpec {
            name: "read_file".into(),
            description: "Read".into(),
            parameters: json!({"type": "object"}),
        }];
        req.tool_choice = ToolChoice::Specific("read_file".into());
        req.options.response_format = ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({"type": "object"}),
            strict: true,
        };
        let body = build_body(&req, None);
        assert_eq!(body["tools"][0]["name"], "read_file");
        assert_eq!(
            body["tool_choice"],
            json!({"type": "function", "name": "read_file"})
        );
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["strict"], true);
    }

    #[test]
    fn a_reasoning_request_asks_for_encrypted_reasoning_unless_reasoning_is_off() {
        let mut req = request(vec![Message::user("hi")]);
        assert_eq!(
            build_body(&req, Some("medium"))["include"],
            json!(["reasoning.encrypted_content"])
        );
        assert!(build_body(&req, None).get("include").is_none());
        req.options.reasoning_effort = Some("off".into());
        assert!(build_body(&req, Some("medium")).get("include").is_none());
    }

    #[test]
    fn a_conversation_key_routes_the_cache_and_is_not_sent_raw() {
        let mut req = request(vec![Message::system("sys 12:01"), Message::user("hi")]);
        req.options.cache_key = Some("telegram:synthetic-user-1".into());
        let first = build_body(&req, None);
        req.messages[0] = Message::system("sys 12:02");
        let second = build_body(&req, None);
        assert_eq!(first["prompt_cache_key"], second["prompt_cache_key"]);
        assert!(
            !first["prompt_cache_key"]
                .as_str()
                .unwrap()
                .contains("synthetic-user-1")
        );
        // The backend routes by the session-id header: same key, both places.
        assert_eq!(session_id(&first), first["prompt_cache_key"].as_str());
    }

    /// The reasoning that led to a tool call goes back just before that call,
    /// without its server-side id (nothing is stored under `store: false`).
    #[test]
    fn encrypted_reasoning_is_replayed_before_the_call_it_produced() {
        let response = decode(&[
            json!({"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "sealed"}}),
            json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c1", "name": "terminal", "arguments": "{}"}}),
            json!({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        ])
        .unwrap();
        let (_, input) = build_input(&[
            Message::user("go"),
            Message::assistant(String::new(), response.tool_calls),
            Message::tool_result("c1", "done"),
        ]);
        let kinds: Vec<&str> = input.iter().map(|i| i["type"].as_str().unwrap()).collect();
        assert_eq!(
            kinds,
            [
                "message",
                "reasoning",
                "function_call",
                "function_call_output"
            ]
        );
        assert_eq!(input[1]["encrypted_content"], "sealed");
        assert!(input[1].get("id").is_none());
    }

    #[test]
    fn reasoning_without_a_tool_call_is_not_kept() {
        let response = decode(&[
            json!({"type": "response.output_item.done", "item": {"type": "reasoning", "summary": [], "encrypted_content": "sealed"}}),
            json!({"type": "response.output_item.done", "item": {"type": "message", "content": [{"text": "Hi"}]}}),
            json!({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}}),
        ])
        .unwrap();
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.text, "Hi");
    }

    #[test]
    fn streamed_items_text_tool_calls_reasoning_and_usage_are_read() {
        let response = decode(&[
            json!({"type": "response.output_text.delta", "delta": "Hel"}),
            json!({"type": "response.output_item.done", "item": {"type": "reasoning", "summary": [{"text": "Thought."}]}}),
            json!({"type": "response.output_item.done", "item": {"type": "message", "content": [{"text": "Hello"}]}}),
            json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c1", "name": "terminal", "arguments": "{\"command\":\"ls\"}"}}),
            json!({"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c2", "name": "", "arguments": "{}"}}),
            json!({"type": "response.completed", "response": {"output": [], "usage": {"input_tokens": 100, "output_tokens": 7, "input_tokens_details": {"cached_tokens": 64}}}}),
        ])
        .unwrap();
        assert_eq!(response.text, "Hello");
        assert_eq!(response.reasoning.as_deref(), Some("Thought."));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].arguments, json!({"command": "ls"}));
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.cached_input_tokens, 64);
    }

    #[test]
    fn a_cut_off_stream_returns_its_partial_text_with_a_note() {
        let response =
            decode(&[json!({"type": "response.output_text.delta", "delta": "partial"})]).unwrap();
        assert_eq!(response.text, "partial");
        assert!(response.note.unwrap().contains("truncated"));
    }

    #[test]
    fn failure_events_are_typed() {
        let rate = decode(&[
            json!({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}),
        ]);
        assert_eq!(rate, Err(ProviderError::RateLimited { retry_after: None }));
        let overload = decode(&[json!({"type": "error", "message": "Model is overloaded"})]);
        assert!(matches!(overload, Err(ProviderError::Transient(_))));
        let coded =
            decode(&[json!({"type": "error", "code": "server_is_overloaded", "message": "x"})]);
        assert!(matches!(coded, Err(ProviderError::Transient(_))));
        // A code is compared whole: a refusal code that merely contains
        // the word is not an overload or a rate limit.
        let refusal = decode(&[
            json!({"type": "error", "code": "invalid_prompt_overload_policy", "message": "Model is overloaded"}),
        ]);
        assert!(matches!(refusal, Err(ProviderError::Other(_))));
        let not_rate = decode(&[
            json!({"type": "error", "code": "rate_limit_policy_violation", "message": "x"}),
        ]);
        assert!(matches!(not_rate, Err(ProviderError::Other(_))));
        let overflow = decode(&[
            json!({"type": "response.failed", "response": {"error": {"code": "context_length_exceeded", "message": "Your input is too big."}}}),
        ]);
        assert!(matches!(
            overflow,
            Err(ProviderError::ContextOverflow { limit: None, .. })
        ));
        let status = decode(&[json!({"type": "error", "status": 503, "message": "x"})]);
        assert!(matches!(status, Err(ProviderError::Transient(m)) if m.starts_with("HTTP 503: ")));
        let refused = decode(&[
            json!({"type": "response.failed", "response": {"error": {"message": "policy"}}}),
        ]);
        assert!(matches!(refused, Err(ProviderError::Other(m)) if m.contains("policy")));
        assert!(matches!(
            decode(&[]),
            Err(ProviderError::Malformed { parse: false, .. })
        ));
    }

    #[test]
    fn auth_rejections_are_explained() {
        assert!(annotate(401, "{}").contains("sign in again"));
        assert_eq!(annotate(500, "boom"), "boom");
    }

    fn headers(pairs: &[(&'static str, &'static str)]) -> reqwest::header::HeaderMap {
        let mut map = reqwest::header::HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, reqwest::header::HeaderValue::from_static(value));
        }
        map
    }

    fn status(code: u16) -> reqwest::StatusCode {
        reqwest::StatusCode::from_u16(code).unwrap()
    }

    /// The Codex backend's plan-limit body: the reset is epoch seconds.
    #[test]
    fn a_plan_limit_is_a_used_up_quota_with_its_reset() {
        let body = r#"{"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"plus","resets_at":1800000000}}"#;
        match chatgpt_error(status(429), &headers(&[]), body) {
            ProviderError::QuotaExhausted {
                message,
                resets_at,
                per_model,
                in_stream,
            } => {
                assert!(!in_stream, "a refused request used nothing");
                assert_eq!(
                    resets_at,
                    Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000))
                );
                assert!(!per_model);
                assert!(message.contains("plus plan"), "{message}");
                assert!(message.contains("The usage limit has been reached"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_plan_limit_without_a_body_reset_takes_the_exhausted_window() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let window = headers(&[
            ("x-codex-primary-used-percent", "42"),
            ("x-codex-primary-reset-at", "1700000100"),
            ("x-codex-secondary-used-percent", "100"),
            ("x-codex-secondary-reset-at", "1700050000"),
        ]);
        let reset = |error: Option<ProviderError>| match error {
            Some(ProviderError::QuotaExhausted { resets_at, .. }) => resets_at,
            other => panic!("unexpected {other:?}"),
        };
        assert_eq!(
            reset(plan_limit(&window, "{}", now)),
            Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_050_000))
        );
        let relative = r#"{"error":{"code":"usage_limit_reached","resets_in_seconds":600}}"#;
        assert_eq!(
            reset(plan_limit(&headers(&[]), relative, now)),
            Some(now + Duration::from_secs(600))
        );
    }

    #[test]
    fn throttling_and_an_uncovered_plan_are_not_a_used_up_quota() {
        let throttled = headers(&[("retry-after", "3"), ("x-codex-primary-used-percent", "60")]);
        assert_eq!(
            chatgpt_error(
                status(429),
                &throttled,
                r#"{"error":{"message":"slow down"}}"#
            ),
            ProviderError::RateLimited {
                retry_after: Some(Duration::from_secs(3))
            }
        );
        assert!(matches!(
            chatgpt_error(
                status(429),
                &headers(&[]),
                r#"{"error":{"type":"usage_not_included","message":"Upgrade"}}"#
            ),
            ProviderError::Billing {
                affordable_tokens: None,
                ..
            }
        ));
    }

    #[test]
    fn a_plan_limit_failure_event_is_a_used_up_quota() {
        let capped = decode(&[json!({"type": "response.failed", "response": {"error": {
            "code": "usage_limit_reached", "message": "limit", "resets_at": 1800000000
        }}})]);
        assert!(matches!(
            capped,
            Err(ProviderError::QuotaExhausted {
                resets_at: Some(_),
                in_stream: true,
                ..
            })
        ));
        // Reported inside an answer already started: not proof of no spend.
        assert!(!capped.unwrap_err().proves_nothing_spent());
    }

    #[test]
    fn tool_schemas_drop_unique_items_at_any_depth() {
        let mut req = request(vec![Message::user("hi")]);
        req.tools = vec![ToolSpec {
            name: "generate_image".into(),
            description: "Draw".into(),
            parameters: json!({"type": "object", "properties": {"ids": {
                "type": "array", "uniqueItems": true, "maxItems": 4,
                "items": {"type": "array", "uniqueItems": true}
            }}}),
        }];
        let body = build_body(&req, None);
        let ids = &body["tools"][0]["parameters"]["properties"]["ids"];
        assert!(ids.get("uniqueItems").is_none());
        assert!(ids["items"].get("uniqueItems").is_none());
        assert_eq!(ids["maxItems"], 4);
    }

    #[test]
    fn tool_schemas_drop_not_but_keep_a_property_named_not() {
        let mut req = request(vec![Message::user("hi")]);
        req.tools = vec![ToolSpec {
            name: "read_file".into(),
            description: "Read".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "not": {"type": "string"},
                    "tail": {"type": "integer"},
                    "start": {"type": "integer"}
                },
                "allOf": [
                    {"not": {"required": ["tail", "start"]}},
                    {"not": {"required": ["tail", "end"]}}
                ],
                "anyOf": [{"required": ["tail"]}, {"not": {"required": ["start"]}}]
            }),
        }];
        let body = build_body(&req, None);
        let params = &body["tools"][0]["parameters"];
        assert!(params.get("allOf").is_none());
        // Optional, so nullable under strict compilation.
        assert_eq!(
            params["properties"]["not"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(params["anyOf"], json!([{"required": ["tail"]}, {}]));
        assert!(!params.to_string().contains("\"not\":{\"required"));
    }
}
