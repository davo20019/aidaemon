//! Anthropic Messages API provider.

use std::collections::BTreeMap;
use std::time::Duration;

use aidaemon_core::{
    collect_stream, Message, ModelRequest, ModelResponse, Provider, ProviderError, Role,
    StreamEvent, ToolCall, ToolChoice, Usage,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::HeaderMap;
use serde_json::{json, Value};

use crate::sse::SseEvent;
use crate::transport::{
    body_excerpt, build_client, drive, header_map, parse_arguments, ConfigError, Secret,
    SseDecoder, DEFAULT_CONNECT_TIMEOUT, DEFAULT_TIMEOUT,
};

pub const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 4096;

/// A provider for the Anthropic Messages API (`POST {base_url}/v1/messages`)
/// with SSE streaming and prompt caching of the stable prefix (system
/// prompt and tool definitions).
///
/// Usage mapping: Anthropic reports `input_tokens` *excluding* cache reads
/// and writes. To match the core convention (where `input_tokens` includes
/// cached input and [`Usage::billable_tokens`] subtracts it), the reported
/// `input_tokens` here is `input_tokens + cache_read_input_tokens +
/// cache_creation_input_tokens`.
#[derive(Debug, Clone)]
pub struct Anthropic {
    name: String,
    client: reqwest::Client,
    endpoint: String,
    headers: HeaderMap,
    timeout: Duration,
    max_tokens: u32,
    prompt_caching: bool,
}

impl Anthropic {
    pub fn builder() -> AnthropicBuilder {
        AnthropicBuilder::default()
    }

    /// The JSON body this provider would send for `request`.
    pub fn request_body(&self, request: &ModelRequest) -> Value {
        build_body(request, self.max_tokens, self.prompt_caching)
    }
}

/// Builder for [`Anthropic`].
#[derive(Debug, Clone)]
pub struct AnthropicBuilder {
    name: String,
    base_url: String,
    api_key: Option<Secret>,
    version: String,
    headers: Vec<(String, String)>,
    timeout: Duration,
    connect_timeout: Duration,
    max_tokens: u32,
    prompt_caching: bool,
}

impl Default for AnthropicBuilder {
    fn default() -> Self {
        Self {
            name: "anthropic".into(),
            base_url: ANTHROPIC_DEFAULT_BASE_URL.into(),
            api_key: None,
            version: ANTHROPIC_VERSION.into(),
            headers: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            max_tokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
            prompt_caching: true,
        }
    }
}

impl AnthropicBuilder {
    /// Name reported by [`Provider::name`] (default `anthropic`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Required. Sent as `x-api-key`.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(Secret(key.into()));
        self
    }

    /// API root, without `/v1/messages` (default `https://api.anthropic.com`).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// `anthropic-version` header (default `2023-06-01`).
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Extra header on every request (e.g. `anthropic-beta`).
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// `max_tokens` used when the request does not set `max_output_tokens`
    /// (default 4096; the API requires a value).
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens.max(1);
        self
    }

    /// Mark the system prompt and tool definitions with `cache_control`
    /// (default on).
    pub fn prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    /// Whole-request timeout, including the streamed body (default 10 min).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Connection timeout (default 30 s).
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<Anthropic, ConfigError> {
        let api_key = self
            .api_key
            .filter(|k| !k.0.is_empty())
            .ok_or(ConfigError::MissingApiKey)?;
        let mut headers = header_map(&[
            ("x-api-key".to_string(), api_key.0),
            ("anthropic-version".to_string(), self.version),
        ])?;
        headers.extend(header_map(&self.headers)?);
        Ok(Anthropic {
            name: self.name,
            client: build_client(self.connect_timeout)?,
            endpoint: format!("{}/v1/messages", self.base_url.trim_end_matches('/')),
            headers,
            timeout: self.timeout,
            max_tokens: self.max_tokens,
            prompt_caching: self.prompt_caching,
        })
    }
}

#[async_trait]
impl Provider for Anthropic {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        collect_stream(self.stream(request)).await
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        let http = self
            .client
            .post(&self.endpoint)
            .timeout(self.timeout)
            .headers(self.headers.clone())
            .header("accept", "text/event-stream")
            .json(&self.request_body(request));
        drive(http, MessagesDecoder::default())
    }
}

/// Anthropic rejects `oneOf`, `anyOf`, `allOf` and `not` at the top level of
/// `input_schema`. Drop them there: they only express cross-field rules, which
/// the tool's own `validate` still enforces when it is called. Nested
/// combinators are accepted and kept.
fn input_schema(parameters: &Value) -> Value {
    let mut schema = parameters.clone();
    if let Some(object) = schema.as_object_mut() {
        for key in ["oneOf", "anyOf", "allOf", "not"] {
            object.remove(key);
        }
        object
            .entry("type")
            .or_insert_with(|| Value::String("object".to_string()));
    }
    schema
}

fn ephemeral() -> Value {
    json!({"type": "ephemeral"})
}

fn build_body(request: &ModelRequest, default_max_tokens: u32, caching: bool) -> Value {
    let mut system: Vec<Value> = request
        .messages
        .iter()
        .filter(|m| m.role == Role::System && !m.content.is_empty())
        .map(|m| json!({"type": "text", "text": m.content}))
        .collect();
    if caching {
        if let Some(last) = system.last_mut() {
            last["cache_control"] = ephemeral();
        }
    }

    let mut body = json!({
        "model": request.model,
        "max_tokens": request.max_output_tokens.unwrap_or(default_max_tokens),
        "messages": wire_messages(&request.messages),
        "stream": true,
    });
    if !system.is_empty() {
        body["system"] = Value::Array(system);
    }
    if !request.tools.is_empty() {
        let mut tools: Vec<Value> = request
            .tools
            .iter()
            .map(|spec| {
                json!({
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": input_schema(&spec.parameters),
                })
            })
            .collect();
        if caching {
            if let Some(last) = tools.last_mut() {
                last["cache_control"] = ephemeral();
            }
        }
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = match &request.tool_choice {
            ToolChoice::Auto => json!({"type": "auto"}),
            ToolChoice::None => json!({"type": "none"}),
            ToolChoice::Required => json!({"type": "any"}),
            ToolChoice::Specific(name) => json!({"type": "tool", "name": name}),
        };
    }
    body
}

/// Map the transcript to Anthropic's strictly alternating user/assistant
/// messages of content blocks. System messages are lifted out; tool results
/// become `tool_result` blocks in a user message; consecutive messages with
/// the same wire role are merged.
fn wire_messages(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<(&'static str, Vec<Value>)> = Vec::new();
    for message in messages {
        let (role, blocks) = match message.role {
            Role::System => continue,
            Role::User => ("user", text_block(&message.content)),
            Role::Tool => (
                "user",
                vec![{
                    let mut block = json!({
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
                        "content": message.content,
                    });
                    if message.is_error {
                        block["is_error"] = json!(true);
                    }
                    block
                }],
            ),
            Role::Assistant => {
                let mut blocks = text_block(&message.content);
                blocks.extend(message.tool_calls.iter().map(|call| {
                    let input = if call.arguments.is_object() {
                        call.arguments.clone()
                    } else {
                        json!({})
                    };
                    json!({"type": "tool_use", "id": call.id, "name": call.name, "input": input})
                }));
                ("assistant", blocks)
            }
        };
        if blocks.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some((last_role, last_blocks)) if *last_role == role => {
                // Anthropic wants tool_result blocks before any text in a
                // user turn.
                if role == "user" && is_tool_result(&blocks[0]) {
                    let split = last_blocks
                        .iter()
                        .position(|b| !is_tool_result(b))
                        .unwrap_or(last_blocks.len());
                    for (offset, block) in blocks.into_iter().enumerate() {
                        last_blocks.insert(split + offset, block);
                    }
                } else {
                    last_blocks.extend(blocks);
                }
            }
            _ => out.push((role, blocks)),
        }
    }
    out.into_iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect()
}

fn text_block(text: &str) -> Vec<Value> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![json!({"type": "text", "text": text})]
    }
}

fn is_tool_result(block: &Value) -> bool {
    block.get("type").and_then(Value::as_str) == Some("tool_result")
}

#[derive(Debug)]
enum Block {
    Text,
    ToolUse {
        id: String,
        name: String,
        initial_input: Value,
        json: String,
    },
    Other,
}

#[derive(Debug, Default)]
struct MessagesDecoder {
    text: String,
    blocks: BTreeMap<u64, Block>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_creation: u64,
}

impl MessagesDecoder {
    fn absorb_usage(&mut self, usage: &Value) {
        let set = |slot: &mut u64, key: &str| {
            if let Some(v) = usage.get(key).and_then(Value::as_u64) {
                *slot = v;
            }
        };
        set(&mut self.input_tokens, "input_tokens");
        set(&mut self.output_tokens, "output_tokens");
        set(&mut self.cache_read, "cache_read_input_tokens");
        set(&mut self.cache_creation, "cache_creation_input_tokens");
    }
}

impl SseDecoder for MessagesDecoder {
    fn on_event(
        &mut self,
        event: &SseEvent,
        deltas: &mut Vec<String>,
    ) -> Result<bool, ProviderError> {
        let data: Value = match serde_json::from_str(&event.data) {
            Ok(data) => data,
            Err(err) => {
                return Err(ProviderError::Transient(format!(
                    "malformed stream event ({err}): {}",
                    body_excerpt(&event.data)
                )))
            }
        };
        let kind = data
            .get("type")
            .and_then(Value::as_str)
            .or(event.event.as_deref())
            .unwrap_or_default();
        let index = data.get("index").and_then(Value::as_u64).unwrap_or(0);
        match kind {
            "message_start" => {
                if let Some(usage) = data.pointer("/message/usage") {
                    self.absorb_usage(usage);
                }
            }
            "content_block_start" => {
                let block = data.get("content_block").cloned().unwrap_or(Value::Null);
                let parsed = match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block.get("text").and_then(Value::as_str).unwrap_or("");
                        if !text.is_empty() {
                            self.text.push_str(text);
                            deltas.push(text.to_string());
                        }
                        Block::Text
                    }
                    Some("tool_use") => Block::ToolUse {
                        id: str_field(&block, "id"),
                        name: str_field(&block, "name"),
                        initial_input: block.get("input").cloned().unwrap_or(json!({})),
                        json: String::new(),
                    },
                    _ => Block::Other,
                };
                self.blocks.insert(index, parsed);
            }
            "content_block_delta" => {
                let delta = data.get("delta").cloned().unwrap_or(Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
                        if !text.is_empty() {
                            self.text.push_str(text);
                            deltas.push(text.to_string());
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(Block::ToolUse { json, .. }) = self.blocks.get_mut(&index) {
                            json.push_str(
                                delta
                                    .get("partial_json")
                                    .and_then(Value::as_str)
                                    .unwrap_or(""),
                            );
                        }
                    }
                    // thinking / signature / citations deltas are not part
                    // of the provider-neutral response.
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(usage) = data.get("usage") {
                    self.absorb_usage(usage);
                }
            }
            "message_stop" => return Ok(true),
            "error" => {
                let message = data
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error");
                let kind = data
                    .pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("error");
                return Err(ProviderError::Transient(format!(
                    "stream error ({kind}): {message}"
                )));
            }
            // ping, content_block_stop, unknown future events
            _ => {}
        }
        Ok(false)
    }

    fn finish(&mut self, terminated: bool) -> Result<ModelResponse, ProviderError> {
        if !terminated {
            return Err(ProviderError::Transient(
                "stream ended before message_stop".into(),
            ));
        }
        let tool_calls = std::mem::take(&mut self.blocks)
            .into_values()
            .filter_map(|block| match block {
                Block::ToolUse {
                    id,
                    name,
                    initial_input,
                    json,
                } => {
                    let arguments = if json.trim().is_empty() {
                        initial_input
                    } else {
                        parse_arguments(&json)
                    };
                    Some(ToolCall::new(id, name, arguments))
                }
                _ => None,
            })
            .collect();
        Ok(ModelResponse {
            text: std::mem::take(&mut self.text),
            tool_calls,
            usage: Usage {
                input_tokens: self.input_tokens + self.cache_read + self.cache_creation,
                output_tokens: self.output_tokens,
                cached_input_tokens: self.cache_read,
                cache_creation_tokens: self.cache_creation,
            },
        })
    }
}

fn str_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod input_schema_tests {
    use super::input_schema;
    use serde_json::json;

    #[test]
    fn top_level_combinators_are_dropped_and_nested_ones_kept() {
        let schema = json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "targets": {"type": "array", "items": {"oneOf": [{"type": "string"}]}}
            },
            "required": ["path"],
            "allOf": [{"not": {"required": ["tail_lines", "start_line"]}}],
            "anyOf": [{"required": ["path"]}]
        });
        let sent = input_schema(&schema);
        assert!(sent.get("allOf").is_none() && sent.get("anyOf").is_none());
        assert_eq!(sent["required"], json!(["path"]));
        assert_eq!(
            sent["properties"]["targets"]["items"]["oneOf"],
            json!([{"type": "string"}])
        );
    }
}
