//! [`Xai`]: Grok models through xAI's Responses API (`POST {base_url}/responses`).
//!
//! Text replies use the Responses API without streaming. A request that asks
//! for JSON ([`ResponseFormat::JsonObject`] or [`ResponseFormat::JsonSchema`])
//! goes to xAI's OpenAI-compatible Chat Completions endpoint instead, through
//! an [`OpenAiCompatible`] pointed at the same base URL, key and headers.

use std::time::Duration;

use aidaemon_core::{
    Image, Message, ModelRequest, ModelResponse, Provider, ProviderError, ResponseFormat, Role,
    StreamEvent, ToolCall, ToolChoice, ToolSpec, Usage,
};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use reqwest::header::HeaderMap;
use serde_json::{json, Map, Value};

use crate::transport::{
    build_client, header_map, map_reqwest_error, map_status, parse_arguments, raw_arguments,
    ConfigError, Secret, DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT,
};
use crate::OpenAiCompatible;

/// xAI's API root.
pub const XAI_DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// A provider for xAI's Responses API.
///
/// - Every system message stays where it is, as a `system` input item: the
///   leading ones form the cached prefix, and later ones (hook notes,
///   mid-run directives) are not hoisted, so the prefix is not rewritten.
/// - Tools are sent with `parallel_tool_calls: true`; `ToolChoice::Auto`
///   sends no `tool_choice`.
/// - The output cap is the request's `max_output_tokens`, else the
///   builder's [`XaiBuilder::max_tokens`], else none.
/// - Cache reads come from `input_tokens_details.cached_tokens` (or the
///   Chat Completions spelling, when a proxy answers in that shape).
/// - No reasoning effort is sent: Grok reasoning models reject the
///   parameter. Reasoning summaries the API returns become
///   [`ModelResponse::reasoning`].
/// - Replies are not streamed: `stream` yields one `Done`.
#[derive(Debug, Clone)]
pub struct Xai {
    name: String,
    client: reqwest::Client,
    endpoint: String,
    headers: HeaderMap,
    timeout: Duration,
    max_tokens: Option<u32>,
    json_fallback: OpenAiCompatible,
}

/// Builder for [`Xai`].
#[derive(Debug, Clone)]
pub struct XaiBuilder {
    name: String,
    base_url: String,
    api_key: Option<Secret>,
    headers: Vec<(String, String)>,
    timeout: Duration,
    connect_timeout: Duration,
    max_tokens: Option<u32>,
}

impl Default for XaiBuilder {
    fn default() -> Self {
        Self {
            name: "xai".into(),
            base_url: XAI_DEFAULT_BASE_URL.into(),
            api_key: None,
            headers: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            max_tokens: None,
        }
    }
}

impl Xai {
    pub fn builder() -> XaiBuilder {
        XaiBuilder::default()
    }

    /// The JSON body this provider would send to `/responses` for `request`.
    pub fn request_body(&self, request: &ModelRequest) -> Value {
        build_body(request, self.max_tokens)
    }

    /// The request the Chat Completions fallback receives, when it differs
    /// from the caller's: the builder's output cap applies there too.
    fn fallback_request(&self, request: &ModelRequest) -> Option<ModelRequest> {
        if request.max_output_tokens.is_some() || self.max_tokens.is_none() {
            return None;
        }
        let mut request = request.clone();
        request.max_output_tokens = self.max_tokens;
        Some(request)
    }

    async fn respond(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let response = self
            .client
            .post(&self.endpoint)
            .timeout(self.timeout)
            .headers(self.headers.clone())
            .json(&self.request_body(request))
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.text().await.map_err(map_reqwest_error)?;
        if !status.is_success() {
            return Err(map_status(status, &headers, &body));
        }
        parse_payload(&body)
    }
}

impl XaiBuilder {
    /// Name reported by [`Provider::name`] (default `xai`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Required. Sent as `Authorization: Bearer <key>`.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(Secret(key.into()));
        self
    }

    /// API root, without `/responses` (default `https://api.x.ai/v1`). Must
    /// be HTTPS, except for a proxy on localhost.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        let url = url.trim().trim_end_matches('/');
        if !url.is_empty() {
            self.base_url = url.to_string();
        }
        self
    }

    /// Extra header on every request, including the JSON-mode fallback.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Output cap for requests that set no `max_output_tokens`. Default:
    /// none sent (the API's own default applies).
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens.max(1));
        self
    }

    /// Whole-request timeout (default 10 min). Replies are not streamed, so
    /// this bounds the model's full generation time.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Connection timeout (default 30 s).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<Xai, ConfigError> {
        let api_key = self
            .api_key
            .filter(|k| !k.0.is_empty())
            .ok_or(ConfigError::MissingApiKey)?;
        if !base_url_is_safe(&self.base_url) {
            return Err(ConfigError::Missing(
                "an https base_url (plain http is allowed only for localhost)",
            ));
        }
        let mut headers =
            header_map(&[("authorization".to_string(), format!("Bearer {}", api_key.0))])?;
        headers.extend(header_map(&self.headers)?);
        let mut fallback = OpenAiCompatible::builder()
            .name(self.name.clone())
            .base_url(self.base_url.clone())
            .api_key(api_key.0)
            .timeout(self.timeout)
            .connect_timeout(self.connect_timeout);
        for (name, value) in &self.headers {
            fallback = fallback.header(name.clone(), value.clone());
        }
        Ok(Xai {
            name: self.name,
            client: build_client(self.connect_timeout)?,
            endpoint: format!("{}/responses", self.base_url),
            headers,
            timeout: self.timeout,
            max_tokens: self.max_tokens,
            json_fallback: fallback.build()?,
        })
    }
}

/// HTTPS anywhere, plain HTTP only on localhost: the API key travels in a
/// header.
fn base_url_is_safe(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    match url.scheme() {
        "https" => true,
        "http" => matches!(
            url.host_str().unwrap_or_default(),
            "localhost" | "127.0.0.1" | "[::1]" | "::1"
        ),
        _ => false,
    }
}

/// The Responses endpoint is not given a response format; JSON requests go
/// to Chat Completions.
fn uses_chat_completions(request: &ModelRequest) -> bool {
    !matches!(request.options.response_format, ResponseFormat::Text)
}

#[async_trait]
impl Provider for Xai {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        if uses_chat_completions(request) {
            let adjusted = self.fallback_request(request);
            return self
                .json_fallback
                .complete(adjusted.as_ref().unwrap_or(request))
                .await;
        }
        self.respond(request).await
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        if uses_chat_completions(request) {
            return match self.fallback_request(request) {
                // The caller's request as is: stream it through.
                None => self.json_fallback.stream(request),
                // An adjusted copy cannot outlive this call while borrowed
                // by the inner stream; drain it into an owned list.
                Some(adjusted) => Box::pin(
                    stream::once(async move {
                        let events: Vec<_> = self.json_fallback.stream(&adjusted).collect().await;
                        stream::iter(events)
                    })
                    .flatten(),
                ),
            };
        }
        Box::pin(stream::once(async move {
            self.respond(request).await.map(StreamEvent::Done)
        }))
    }
}

/// The input items for a transcript. System messages keep their position
/// as `system` items. Images from a run of consecutive tool results follow
/// it in one user message (a function output is text only), labelled with
/// the call each image came from.
fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::with_capacity(messages.len());
    let mut tool_images: Vec<Value> = Vec::new();
    for message in messages {
        if message.role != Role::Tool && !tool_images.is_empty() {
            input.push(user_item(std::mem::take(&mut tool_images)));
        }
        let has_text = !message.content.trim().is_empty();
        match message.role {
            Role::System => {
                if has_text {
                    input.push(json!({"role": "system", "content": message.content}));
                }
            }
            Role::User if message.images.is_empty() => {
                if has_text {
                    input.push(json!({"role": "user", "content": message.content}));
                }
            }
            Role::User => {
                let mut content = Vec::with_capacity(message.images.len() + 1);
                if has_text {
                    content.push(json!({"type": "input_text", "text": message.content}));
                }
                content.extend(message.images.iter().map(image_part));
                input.push(user_item(content));
            }
            Role::Assistant => {
                if has_text {
                    input.push(json!({"role": "assistant", "content": message.content}));
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
                }
            }
            Role::Tool => {
                let call_id = message.tool_call_id.as_deref().unwrap_or_default();
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.content,
                }));
                if !message.images.is_empty() {
                    tool_images.push(json!({
                        "type": "input_text",
                        "text": format!("[Images returned by tool call {call_id}]"),
                    }));
                    tool_images.extend(message.images.iter().map(image_part));
                }
            }
        }
    }
    if !tool_images.is_empty() {
        input.push(user_item(tool_images));
    }
    input
}

fn user_item(content: Vec<Value>) -> Value {
    json!({"role": "user", "content": content})
}

fn image_part(image: &Image) -> Value {
    json!({"type": "input_image", "image_url": image.to_url()})
}

fn build_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect()
}

/// `None` for `Auto`, the API's default.
fn build_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Specific(name) => Some(json!({"type": "function", "name": name})),
    }
}

/// The body for `POST /responses`. A tool choice without tools is dropped.
fn build_body(request: &ModelRequest, default_max_tokens: Option<u32>) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(request.model));
    body.insert("input".into(), json!(build_input(&request.messages)));
    if let Some(max) = request.max_output_tokens.or(default_max_tokens) {
        body.insert("max_output_tokens".into(), json!(max));
    }
    if !request.tools.is_empty() {
        body.insert("tools".into(), json!(build_tools(&request.tools)));
        body.insert("parallel_tool_calls".into(), json!(true));
        if let Some(choice) = build_tool_choice(&request.tool_choice) {
            body.insert("tool_choice".into(), choice);
        }
    }
    Value::Object(body)
}

/// Read a reply body: the Responses shape, or the Chat Completions shape a
/// compatible proxy may answer with.
fn parse_payload(body: &str) -> Result<ModelResponse, ProviderError> {
    let data: Value = serde_json::from_str(body).map_err(|err| ProviderError::Malformed {
        message: format!("xAI reply is not JSON: {err}"),
        parse: true,
    })?;
    if data.get("output").is_some() || data.get("output_text").is_some() {
        return parse_responses(&data);
    }
    if data.get("choices").is_some() {
        return parse_chat_completions(&data);
    }
    Err(ProviderError::Malformed {
        message: "xAI reply has neither `output` nor `choices`".into(),
        parse: false,
    })
}

/// A tool call from a Responses item or a Chat Completions `tool_calls`
/// entry. Arguments may be a JSON string or an already-parsed value.
fn parse_tool_call(value: &Value) -> Option<ToolCall> {
    let function = value.get("function");
    let name = value
        .get("name")
        .or_else(|| function.and_then(|f| f.get("name")))
        .and_then(Value::as_str)?
        .trim();
    if name.is_empty() {
        return None;
    }
    let id = value
        .get("call_id")
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = match value
        .get("arguments")
        .or_else(|| function.and_then(|f| f.get("arguments")))
    {
        Some(Value::String(raw)) => parse_arguments(raw),
        None | Some(Value::Null) => json!({}),
        Some(other) => other.clone(),
    };
    Some(ToolCall::new(id, name, arguments))
}

fn push_text(chunks: &mut Vec<String>, text: Option<&str>) {
    if let Some(text) = text.map(str::trim).filter(|t| !t.is_empty()) {
        chunks.push(text.to_string());
    }
}

fn items(value: Option<&Value>) -> impl Iterator<Item = &Value> {
    value.and_then(Value::as_array).into_iter().flatten()
}

fn parse_responses(data: &Value) -> Result<ModelResponse, ProviderError> {
    let mut text = Vec::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();
    for item in items(data.get("output")) {
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "function_call" | "tool_call" => tool_calls.extend(parse_tool_call(item)),
            "message" => {
                for part in items(item.get("content")) {
                    match part.get("type").and_then(Value::as_str).unwrap_or_default() {
                        "output_text" | "input_text" | "text" => {
                            push_text(&mut text, part.get("text").and_then(Value::as_str))
                        }
                        "function_call" | "tool_call" => tool_calls.extend(parse_tool_call(part)),
                        _ => {}
                    }
                }
            }
            "reasoning" => {
                for part in items(item.get("summary")).chain(items(item.get("content"))) {
                    push_text(&mut reasoning, part.get("text").and_then(Value::as_str));
                }
            }
            _ => {}
        }
    }
    // `output_text` is a convenience copy of the message text: use it only
    // when the items carried none, so text is not doubled.
    if text.is_empty() {
        push_text(&mut text, data.get("output_text").and_then(Value::as_str));
    }
    if text.is_empty() && tool_calls.is_empty() {
        return Err(ProviderError::Malformed {
            message: "xAI reply has no output text and no tool calls".into(),
            parse: false,
        });
    }
    let note = (data.get("status").and_then(Value::as_str) == Some("incomplete")).then(|| {
        let reason = data
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            .unwrap_or("unspecified");
        format!("xAI reply is incomplete ({reason}).")
    });
    Ok(ModelResponse {
        text: text.join("\n"),
        tool_calls,
        usage: parse_usage(data),
        reasoning: (!reasoning.is_empty()).then(|| reasoning.join("\n")),
        note,
    })
}

fn parse_chat_completions(data: &Value) -> Result<ModelResponse, ProviderError> {
    let message = data
        .pointer("/choices/0/message")
        .ok_or_else(|| ProviderError::Malformed {
            message: "xAI reply has no choices[0].message".into(),
            parse: false,
        })?;
    Ok(ModelResponse {
        text: message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        tool_calls: items(message.get("tool_calls"))
            .filter_map(parse_tool_call)
            .collect(),
        usage: parse_usage(data),
        ..Default::default()
    })
}

/// Usage in either spelling. `input_tokens` includes cache reads, as the
/// core convention expects.
fn parse_usage(data: &Value) -> Usage {
    let Some(usage) = data.get("usage") else {
        return Usage::default();
    };
    let first = |pointers: &[&str]| {
        pointers
            .iter()
            .find_map(|p| usage.pointer(p).and_then(Value::as_u64))
            .unwrap_or(0)
    };
    Usage {
        input_tokens: first(&["/input_tokens", "/prompt_tokens"]),
        output_tokens: first(&["/output_tokens", "/completion_tokens"]),
        cached_input_tokens: first(&[
            "/input_tokens_details/cached_tokens",
            "/prompt_tokens_details/cached_tokens",
            "/cached_tokens",
        ]),
        cache_creation_tokens: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::RequestOptions;

    fn request(messages: Vec<Message>) -> ModelRequest {
        ModelRequest {
            model: "grok-4".into(),
            messages,
            ..Default::default()
        }
    }

    #[test]
    fn the_builder_requires_a_key_and_a_safe_base_url() {
        assert!(matches!(
            Xai::builder().build(),
            Err(ConfigError::MissingApiKey)
        ));
        assert!(matches!(
            Xai::builder().api_key("").build(),
            Err(ConfigError::MissingApiKey)
        ));
        let remote_http = Xai::builder()
            .api_key("k")
            .base_url("http://xai.example.com/v1")
            .build();
        assert!(matches!(remote_http, Err(ConfigError::Missing(_))));
        for local in ["http://localhost:8080/v1", "http://127.0.0.1:9/v1/"] {
            let built = Xai::builder().api_key("k").base_url(local).build().unwrap();
            assert!(
                built.endpoint.ends_with("/v1/responses"),
                "{}",
                built.endpoint
            );
        }
        assert!(Xai::builder()
            .api_key("k")
            .header("bad header", "x")
            .build()
            .is_err());
        let default = Xai::builder().api_key("k").build().unwrap();
        assert_eq!(default.name(), "xai");
        assert_eq!(default.endpoint, "https://api.x.ai/v1/responses");
        let named = Xai::builder().api_key("k").name("grok").build().unwrap();
        assert_eq!(named.name(), "grok");
    }

    #[test]
    fn tool_calls_and_results_become_function_items() {
        let call = ToolCall::new("call_1", "search_files", json!({"q": "Cargo.toml"}));
        let input = build_input(&[
            Message::system("You are helpful."),
            Message::user("find files"),
            Message::assistant("", vec![call]),
            Message::tool_result("call_1", r#"["Cargo.toml"]"#),
        ]);
        assert_eq!(
            input[0],
            json!({"role": "system", "content": "You are helpful."})
        );
        assert_eq!(input[1], json!({"role": "user", "content": "find files"}));
        // Empty assistant text sends no message item.
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["arguments"], r#"{"q":"Cargo.toml"}"#);
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["output"], r#"["Cargo.toml"]"#);
        assert_eq!(input.len(), 4);
    }

    #[test]
    fn unparseable_arguments_replay_verbatim() {
        let call = ToolCall::new("c1", "terminal", parse_arguments(r#"{"command":"#));
        let input = build_input(&[Message::assistant("", vec![call])]);
        assert_eq!(input[0]["arguments"], r#"{"command":"#);
    }

    #[test]
    fn later_system_messages_stay_in_place() {
        let input = build_input(&[
            Message::system("You are terse."),
            Message::user("hi"),
            Message::assistant("Hello.", vec![]),
            Message::hook_note("[Recalled] Alice Rivera works at Acme Corp."),
            Message::system("Finish the task now."),
            Message::user("and now?"),
        ]);
        let roles: Vec<_> = input.iter().map(|i| i["role"].clone()).collect();
        assert_eq!(
            roles,
            ["system", "user", "assistant", "system", "system", "user"]
        );
        assert_eq!(
            input[3]["content"],
            "[Recalled] Alice Rivera works at Acme Corp."
        );
        assert_eq!(input[4]["content"], "Finish the task now.");
        // Nothing is hoisted into `instructions`.
        let body = build_body(
            &request(vec![Message::system("s"), Message::user("u")]),
            None,
        );
        assert!(body.get("instructions").is_none());
        assert_eq!(body["input"][0]["role"], "system");
    }

    #[test]
    fn user_and_tool_images_become_input_images() {
        let image = Image::base64("image/png", "AQID");
        let mut shot = Message::tool_result("c1", "screenshot taken");
        shot.images = vec![Image::url("https://example.com/shot.png")];
        let input = build_input(&[
            Message::user("what is this").with_images(vec![image]),
            Message::assistant("", vec![ToolCall::new("c1", "screenshot", json!({}))]),
            shot,
            Message::user("and that?"),
        ]);
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][1]["type"], "input_image");
        assert!(input[0]["content"][1]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,"));
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[3]["role"], "user");
        assert!(input[3]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("c1"));
        assert_eq!(
            input[3]["content"][1]["image_url"],
            "https://example.com/shot.png"
        );
        assert_eq!(input[4]["content"], "and that?");
    }

    #[test]
    fn body_carries_tools_choice_and_output_cap() {
        let mut req = request(vec![Message::user("hi")]);
        let body = build_body(&req, None);
        assert_eq!(body["model"], "grok-4");
        assert!(body.get("max_output_tokens").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("stream").is_none());
        assert!(body.get("reasoning").is_none());

        assert_eq!(build_body(&req, Some(900))["max_output_tokens"], 900);
        req.max_output_tokens = Some(500);
        assert_eq!(build_body(&req, Some(900))["max_output_tokens"], 500);

        req.tools = vec![ToolSpec {
            name: "read_file".into(),
            description: "Read".into(),
            parameters: json!({"type": "object"}),
        }];
        let body = build_body(&req, None);
        assert_eq!(
            body["tools"][0],
            json!({"type": "function", "name": "read_file", "description": "Read",
                   "parameters": {"type": "object"}})
        );
        assert_eq!(body["parallel_tool_calls"], true);
        assert!(body.get("tool_choice").is_none(), "auto is the default");
        for (choice, wire) in [
            (ToolChoice::None, json!("none")),
            (ToolChoice::Required, json!("required")),
            (
                ToolChoice::Specific("read_file".into()),
                json!({"type": "function", "name": "read_file"}),
            ),
        ] {
            req.tool_choice = choice;
            assert_eq!(build_body(&req, None)["tool_choice"], wire);
        }
        // A forced choice without tools is dropped.
        req.tools.clear();
        assert!(build_body(&req, None).get("tool_choice").is_none());
    }

    #[test]
    fn json_formats_route_to_chat_completions_with_the_output_cap() {
        let mut req = request(vec![Message::user("hi")]);
        assert!(!uses_chat_completions(&req));
        req.options = RequestOptions {
            response_format: ResponseFormat::JsonObject,
            ..Default::default()
        };
        assert!(uses_chat_completions(&req));
        req.options.response_format = ResponseFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({"type": "object"}),
            strict: true,
        };
        assert!(uses_chat_completions(&req));

        let capped = Xai::builder().api_key("k").max_tokens(700).build().unwrap();
        let adjusted = capped.fallback_request(&req).expect("cap applied");
        assert_eq!(adjusted.max_output_tokens, Some(700));
        assert_eq!(adjusted.options, req.options);
        req.max_output_tokens = Some(50);
        assert!(capped.fallback_request(&req).is_none(), "request cap wins");
        let uncapped = Xai::builder().api_key("k").build().unwrap();
        assert!(uncapped
            .fallback_request(&request(vec![Message::user("hi")]))
            .is_none());
    }

    #[test]
    fn responses_text_tool_calls_reasoning_and_usage_are_read() {
        let payload = json!({
            "id": "resp_1",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Look it up."}]},
                {"type": "message", "content": [{"type": "output_text", "text": "I will search now."}]},
                {"type": "function_call", "call_id": "call_42", "name": " search_files ",
                 "arguments": "{\"q\":\"README\"}"},
                {"type": "function_call", "call_id": "call_43", "name": "", "arguments": "{}"}
            ],
            "output_text": "I will search now.",
            "usage": {"input_tokens": 100, "output_tokens": 8,
                      "input_tokens_details": {"cached_tokens": 64}}
        });
        let parsed = parse_payload(&payload.to_string()).unwrap();
        assert_eq!(
            parsed.text, "I will search now.",
            "output_text is not doubled"
        );
        assert_eq!(parsed.reasoning.as_deref(), Some("Look it up."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id, "call_42");
        assert_eq!(parsed.tool_calls[0].name, "search_files");
        assert_eq!(parsed.tool_calls[0].arguments, json!({"q": "README"}));
        assert_eq!(parsed.usage.input_tokens, 100);
        assert_eq!(parsed.usage.output_tokens, 8);
        assert_eq!(parsed.usage.cached_input_tokens, 64);
        assert_eq!(parsed.usage.billable_tokens(), 36 + 8);
        assert!(parsed.note.is_none());
    }

    #[test]
    fn top_level_output_text_alone_is_read_and_incomplete_is_noted() {
        let payload = json!({
            "output_text": "Partial answer",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"}
        });
        let parsed = parse_payload(&payload.to_string()).unwrap();
        assert_eq!(parsed.text, "Partial answer");
        assert!(parsed.note.unwrap().contains("max_output_tokens"));
        assert_eq!(parsed.usage, Usage::default());
    }

    #[test]
    fn chat_completions_shape_is_supported() {
        let payload = json!({
            "choices": [{"message": {
                "content": "Fallback text",
                "tool_calls": [{"id": "call_7", "type": "function",
                    "function": {"name": "get_time", "arguments": {"city": "Springfield"}}}]
            }}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5,
                      "prompt_tokens_details": {"cached_tokens": 2}}
        });
        let parsed = parse_payload(&payload.to_string()).unwrap();
        assert_eq!(parsed.text, "Fallback text");
        assert_eq!(parsed.tool_calls[0].id, "call_7");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"city": "Springfield"})
        );
        assert_eq!(parsed.usage.input_tokens, 3);
        assert_eq!(parsed.usage.output_tokens, 5);
        assert_eq!(parsed.usage.cached_input_tokens, 2);
    }

    #[test]
    fn unreadable_replies_are_malformed() {
        assert!(matches!(
            parse_payload("not json"),
            Err(ProviderError::Malformed { parse: true, .. })
        ));
        for shape in [r#"{"id": "x"}"#, r#"{"output": []}"#, r#"{"choices": []}"#] {
            assert!(
                matches!(
                    parse_payload(shape),
                    Err(ProviderError::Malformed { parse: false, .. })
                ),
                "{shape}"
            );
        }
    }

    #[test]
    fn conformance_kit_bodies_parse() {
        let plain = json!({"id": "resp_plain", "status": "completed", "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Hello there."}]}],
            "usage": {"input_tokens": 20, "output_tokens": 3,
                      "input_tokens_details": {"cached_tokens": 8}}});
        let parsed = parse_payload(&plain.to_string()).unwrap();
        assert_eq!(parsed.text, "Hello there.");
        assert_eq!(parsed.usage.cached_input_tokens, 8);
        let tool = json!({"id": "resp_tool", "status": "completed", "output": [
            {"type": "function_call", "call_id": "call_a", "name": "get_time",
             "arguments": "{\"city\": \"Springfield\"}"}],
            "usage": {"input_tokens": 30, "output_tokens": 9}});
        let parsed = parse_payload(&tool.to_string()).unwrap();
        assert_eq!(parsed.tool_calls[0].name, "get_time");
        assert_eq!(
            parsed.tool_calls[0].arguments,
            json!({"city": "Springfield"})
        );
    }

    #[test]
    fn http_errors_map_to_typed_kinds() {
        let headers = HeaderMap::new();
        let status = |code| reqwest::StatusCode::from_u16(code).unwrap();
        assert!(matches!(
            map_status(status(402), &headers, r#"{"error":"credits exhausted"}"#),
            ProviderError::Billing { .. }
        ));
        assert!(matches!(
            map_status(status(404), &headers, "model grok-0 not found"),
            ProviderError::NotFound(_)
        ));
        assert!(matches!(
            map_status(
                status(400),
                &headers,
                r#"{"error":{"message":"This model's maximum context length is 131072 tokens"}}"#
            ),
            ProviderError::ContextOverflow {
                limit: Some(131072),
                ..
            }
        ));
    }
}
