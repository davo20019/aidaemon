use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use zeroize::Zeroize;

use crate::config::SlotRoutingConfig;
use crate::providers::ProviderError;
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

pub struct OpenAiCompatibleProvider {
    client: Client,
    base_url: String,
    api_key: String,
    gateway_token: Option<String>,
    extra_headers: HashMap<String, String>,
    is_cloudflare_gateway: bool,
    max_tokens: Option<u32>,
    /// When set, includes the `reasoning` parameter in requests to enable
    /// thinking/reasoning tokens (supported by OpenRouter, Anthropic, etc.).
    /// Values: "low", "medium", "high", "xhigh"
    pub reasoning_effort: Option<String>,
    /// Opt-in llama.cpp chat-template thinking. Disabled by default because
    /// cloud OpenAI-compatible APIs may reject these llama.cpp-only fields.
    llama_cpp_thinking: bool,
    /// Opt-in llama.cpp slot routing. When `enabled`, every request carries an
    /// `id_slot` field: the per-call override when present, else `background_slot`.
    /// When disabled, `id_slot` is never emitted (cloud-API safe).
    slot_routing: SlotRoutingConfig,
    /// Opt-in SSE streaming transport. Deltas are accumulated into the same
    /// response shape as a non-streaming call; a stream that dies after
    /// partial text is reported as a `length` cutoff so truncation recovery
    /// continues it instead of losing the response.
    streaming: bool,
    /// Models whose final (no-tool-call) answers are re-issued through a
    /// tool-less, schema-constrained `{reasoning, response}` second pass.
    /// For models (local Gemma) that emit raw undelimited chain-of-thought AS
    /// their answer body on challenge/truncation-recovery turns; a JSON
    /// grammar cannot coexist with tool calls, hence intercept-and-reissue.
    /// Matched by exact model name; empty (off) by default.
    structured_answer_models: Vec<String>,
}

impl Drop for OpenAiCompatibleProvider {
    fn drop(&mut self) {
        self.api_key.zeroize();
        if let Some(token) = self.gateway_token.as_mut() {
            token.zeroize();
        }
    }
}

/// Validate the base URL for security.
/// - HTTPS is required for remote URLs to protect API keys in transit
/// - HTTP is allowed only for localhost/127.0.0.1 (local LLM servers)
fn validate_base_url(base_url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(base_url)
        .map_err(|e| format!("Invalid base_url '{}': {}", base_url, e))?;

    let scheme = parsed.scheme();
    let host = parsed.host_str().unwrap_or("");

    match scheme {
        "https" => Ok(()), // HTTPS is always allowed
        "http" => {
            // HTTP only allowed for localhost
            let is_localhost =
                host == "localhost" || host == "127.0.0.1" || host == "[::1]" || host == "::1";

            if is_localhost {
                warn!(
                    "Using unencrypted HTTP for local LLM server at '{}'. \
                     API key will be transmitted in cleartext.",
                    base_url
                );
                Ok(())
            } else {
                Err(format!(
                    "HTTP is not allowed for remote URLs (base_url: '{}'). \
                     Use HTTPS to protect your API key in transit. \
                     HTTP is only permitted for localhost.",
                    base_url
                ))
            }
        }
        _ => Err(format!(
            "Unsupported URL scheme '{}' in base_url '{}'. Only http and https are allowed.",
            scheme, base_url
        )),
    }
}

fn is_cloudflare_ai_gateway_base(base_url: &str) -> bool {
    let parsed = match reqwest::Url::parse(base_url) {
        Ok(url) => url,
        Err(_) => return false,
    };
    matches!(
        parsed.host_str(),
        Some(host) if host.eq_ignore_ascii_case("gateway.ai.cloudflare.com")
    )
}

fn normalize_tool_name(name: &str) -> String {
    name.trim().to_string()
}

/// Why a response `content` string looks like it may carry leaked model
/// reasoning that the provider/server failed to split into a structured
/// `reasoning`/`reasoning_content` field.
///
/// Diagnostic only — this classification never changes delivery; it exists so
/// the next real occurrence of the intermittent reasoning-leak (raw
/// chain-of-thought reaching a user) is captured with ground truth. The
/// distinction matters for the fix: a `Delimited` leak can be stripped
/// deterministically at the boundary, whereas `UndelimitedProse` cannot and is
/// the only case that would justify envelope/schema enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningLeakSignal {
    /// Content carries an explicit thinking delimiter (e.g. `<think>`).
    Delimited,
    /// No delimiter, but content reads as first-person meta-deliberation about
    /// the user/task, or echoes daemon-internal prompt scaffolding.
    UndelimitedProse,
}

/// Heuristically classify whether `content` looks like leaked reasoning.
///
/// Intended to be called only when the structured reasoning field came back
/// empty; a non-empty reasoning field means the split already worked. Tuned for
/// precision on the scaffold-echo and delimiter signals (near-zero false
/// positives); the prose path requires two distinct deliberation markers so a
/// single incidental phrase in a genuine answer does not trip it.
fn classify_possible_reasoning_leak(content: &str) -> Option<ReasoningLeakSignal> {
    let trimmed = content.trim();
    if trimmed.len() < 40 {
        return None;
    }
    let lower = trimmed.to_lowercase();

    // Explicit thinking delimiters emitted by various chat templates. Any hit is
    // an unambiguous split failure.
    const DELIMITERS: &[&str] = &[
        "<think>",
        "</think>",
        "◁think▷",
        "◁/think▷",
        "<start_of_thinking>",
        "</start_of_thinking>",
        "<end_of_thinking>",
        "<|thinking|>",
        "<|end_thinking|>",
    ];
    if DELIMITERS.iter().any(|d| lower.contains(d)) {
        return Some(ReasoningLeakSignal::Delimited);
    }

    // Daemon-internal prompt scaffolding echoed back into an outbound reply is a
    // strong leak signal: these markers are internal and should never appear in
    // user-visible content.
    const SCAFFOLD_ECHOES: &[&str] = &["original request:", "follow-up:"];
    if SCAFFOLD_ECHOES.iter().any(|m| lower.contains(m)) {
        return Some(ReasoningLeakSignal::UndelimitedProse);
    }

    // First-person meta-deliberation about the user/task. Two distinct markers
    // required to fire.
    const DELIBERATION_MARKERS: &[&str] = &[
        "the user is",
        "the user wants",
        "the user said",
        "it seems the user",
        "since they said",
        "i should acknowledge",
        "i should respond",
        "i'll acknowledge",
        "i will acknowledge",
        "let me think",
        "wait, looking",
        "communication preferences",
    ];
    let hits = DELIBERATION_MARKERS
        .iter()
        .filter(|m| lower.contains(**m))
        .count();
    if hits >= 2 {
        return Some(ReasoningLeakSignal::UndelimitedProse);
    }

    None
}

/// Outcome of parsing the structured `{reasoning, response}` second-pass body.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StructuredAnswerParse {
    /// Valid JSON with a non-blank `response`.
    Complete { reasoning: String, response: String },
    /// Invalid JSON (typically max_tokens truncation mid-string) but the
    /// user-visible `response` prefix was salvageable.
    Partial { response: String },
    /// Nothing user-deliverable could be extracted.
    Unusable,
}

/// What to deliver after the structured second pass.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SecondPassResolution {
    Deliver {
        response: String,
        reasoning: Option<String>,
        truncated: bool,
    },
    /// Truncation on the first attempt: retry once with a larger token cap.
    RetryLarger,
    /// Second pass unusable but the draft is not reasoning-shaped: ship it.
    UseDraft,
    /// Second pass unusable AND the draft looks like leaked reasoning:
    /// deliver an honest failure instead of raw chain-of-thought.
    Suppress,
}

/// Instruction appended at the END of the message list (not the system
/// prompt) for the structured second pass, so the llama.cpp prompt cache
/// reuses the first pass's prefill for the shared prefix — a system-prompt
/// edit would force a full cold re-prefill of the whole conversation.
const STRUCTURED_ANSWER_INSTRUCTION: &str = "Format your reply as a JSON object with two \
string fields, in this order: \"reasoning\" then \"response\". Put ALL analysis, counting, \
double-checking, self-correction, and notes about this conversation in \"reasoning\" — none \
of it in \"response\". \"response\" is the only text the user will see: the final, polished \
reply to my last message — direct and complete, with no deliberation, no apologies for \
earlier drafts, and no commentary about the conversation or your own process.";

/// User-visible text for the one combination where nothing safe can ship:
/// the second pass produced nothing usable AND the draft is leaked reasoning.
const STRUCTURED_ANSWER_SUPPRESSED_MSG: &str = "I ran into a formatting problem while \
finalizing this reply and don't have a clean answer to send. Please ask again.";

/// JSON schema for the structured final answer. The user-visible field is
/// named `response` (not `answer`) because serde_json serializes object keys
/// alphabetically and llama.cpp compiles the schema into a grammar in wire
/// order: `reasoning` must decode BEFORE the user-visible field so the model
/// drains its deliberation first ("rea" < "res"). Renaming either field can
/// silently flip that order and re-open the leak.
fn structured_answer_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "All deliberation, checking, and self-correction. Never shown to the user."
            },
            "response": {
                "type": "string",
                "description": "The final user-visible reply. Polished and complete; no deliberation."
            }
        },
        "required": ["reasoning", "response"],
        "additionalProperties": false
    })
}

fn append_answer_instruction(messages: &[Value]) -> Vec<Value> {
    let mut out = messages.to_vec();
    if let Some(last) = out.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some("user") {
            if let Some(text) = last.get("content").and_then(Value::as_str) {
                let combined = format!("{}\n\n{}", text, STRUCTURED_ANSWER_INSTRUCTION);
                last["content"] = json!(combined);
                return out;
            }
        }
    }
    out.push(json!({"role": "user", "content": STRUCTURED_ANSWER_INSTRUCTION}));
    out
}

/// Best-effort extraction of a string field's value from truncated/invalid
/// JSON: finds `"<key>"`, skips to the opening quote, then decodes JSON
/// string escapes until the closing quote or end of input.
fn salvage_json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let start = text.find(&needle)? + needle.len();
    let mut chars = text[start..].chars().peekable();
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
    if chars.next() != Some(':') {
        return None;
    }
    while matches!(chars.peek(), Some(c) if c.is_whitespace()) {
        chars.next();
    }
    if chars.next() != Some('"') {
        return None;
    }
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('b') => out.push('\u{0008}'),
                Some('f') => out.push('\u{000C}'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                        out.push(ch);
                    }
                }
                // Covers \" \\ \/ — the escaped char is itself.
                Some(other) => out.push(other),
                None => break,
            },
            _ => out.push(c),
        }
    }
    Some(out)
}

fn parse_structured_answer(raw: &str) -> StructuredAnswerParse {
    let mut text = raw.trim();
    // A non-grammar backend (e.g. cloud fallback) may fence the JSON.
    if let Some(rest) = text.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.trim_start();
        text = rest.strip_suffix("```").map(str::trim_end).unwrap_or(rest);
    }
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(text) {
        let response = obj.get("response").and_then(Value::as_str).unwrap_or("");
        if !response.trim().is_empty() {
            let reasoning = obj.get("reasoning").and_then(Value::as_str).unwrap_or("");
            return StructuredAnswerParse::Complete {
                reasoning: reasoning.to_string(),
                response: response.to_string(),
            };
        }
        return StructuredAnswerParse::Unusable;
    }
    match salvage_json_string_field(text, "response") {
        Some(response) if !response.trim().is_empty() => {
            StructuredAnswerParse::Partial { response }
        }
        _ => StructuredAnswerParse::Unusable,
    }
}

fn resolve_structured_answer(
    parse: StructuredAnswerParse,
    second_truncated: bool,
    already_retried: bool,
    draft_leaky: bool,
) -> SecondPassResolution {
    let can_retry = second_truncated && !already_retried;
    match parse {
        StructuredAnswerParse::Complete {
            reasoning,
            response,
        } => SecondPassResolution::Deliver {
            response,
            reasoning: (!reasoning.trim().is_empty()).then_some(reasoning),
            truncated: false,
        },
        StructuredAnswerParse::Partial { response } => {
            if can_retry {
                SecondPassResolution::RetryLarger
            } else {
                SecondPassResolution::Deliver {
                    response,
                    reasoning: None,
                    truncated: true,
                }
            }
        }
        StructuredAnswerParse::Unusable => {
            if can_retry {
                SecondPassResolution::RetryLarger
            } else if !draft_leaky {
                SecondPassResolution::UseDraft
            } else {
                SecondPassResolution::Suppress
            }
        }
    }
}

fn needs_structured_answer_pass(
    models: &[String],
    model: &str,
    tools: &[Value],
    options: &ChatOptions,
    response: &ProviderResponse,
) -> bool {
    // No tools on the request means this is not the agent's answer turn (the
    // second pass itself runs tool-less — this is also the recursion guard).
    // Callers that already asked for structured output manage their own shape.
    if tools.is_empty() || !matches!(options.response_mode, ResponseMode::Text) {
        return false;
    }
    if !models.iter().any(|m| m == model) {
        return false;
    }
    response.tool_calls.is_empty()
        && response
            .content
            .as_deref()
            .is_some_and(|c| !c.trim().is_empty())
}

fn structured_answer_chat_options(base: &ChatOptions) -> ChatOptions {
    let mut opts = base.clone();
    opts.response_mode = ResponseMode::JsonSchema {
        name: "final_answer".to_string(),
        schema: structured_answer_schema(),
        strict: true,
    };
    opts.tool_choice = ToolChoiceMode::None;
    // Thinking OFF: the reasoning field replaces the thinking channel instead
    // of duplicating it (duplication doubles decode and drives truncation).
    opts.reasoning_effort_override = Some("off".to_string());
    opts
}

fn merge_token_usage(base: Option<TokenUsage>, extra: Option<TokenUsage>) -> Option<TokenUsage> {
    fn sum_u32(x: Option<u32>, y: Option<u32>) -> Option<u32> {
        match (x, y) {
            (None, v) | (v, None) => v,
            (Some(x), Some(y)) => Some(x + y),
        }
    }
    fn sum_f64(x: Option<f64>, y: Option<f64>) -> Option<f64> {
        match (x, y) {
            (None, v) | (v, None) => v,
            (Some(x), Some(y)) => Some(x + y),
        }
    }
    match (base, extra) {
        (None, u) | (u, None) => u,
        (Some(a), Some(b)) => Some(TokenUsage {
            input_tokens: a.input_tokens + b.input_tokens,
            output_tokens: a.output_tokens + b.output_tokens,
            cached_input_tokens: sum_u32(a.cached_input_tokens, b.cached_input_tokens),
            cache_creation_input_tokens: sum_u32(
                a.cache_creation_input_tokens,
                b.cache_creation_input_tokens,
            ),
            model: a.model,
            prompt_ms: sum_f64(a.prompt_ms, b.prompt_ms),
            decode_ms: sum_f64(a.decode_ms, b.decode_ms),
        }),
    }
}

impl OpenAiCompatibleProvider {
    #[allow(dead_code)]
    pub fn new(base_url: &str, api_key: &str) -> Result<Self, String> {
        Self::new_with_gateway_token(base_url, api_key, None)
    }

    pub fn new_with_gateway_token(
        base_url: &str,
        api_key: &str,
        gateway_token: Option<&str>,
    ) -> Result<Self, String> {
        Self::new_with_gateway_token_and_headers(base_url, api_key, gateway_token, None)
    }

    pub fn new_with_gateway_token_and_headers(
        base_url: &str,
        api_key: &str,
        gateway_token: Option<&str>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> Result<Self, String> {
        Self::new_with_all_options(base_url, api_key, gateway_token, extra_headers, None)
    }

    pub fn new_with_all_options(
        base_url: &str,
        api_key: &str,
        gateway_token: Option<&str>,
        extra_headers: Option<HashMap<String, String>>,
        max_tokens: Option<u32>,
    ) -> Result<Self, String> {
        // Validate URL security before creating provider
        validate_base_url(base_url)?;

        let client = crate::providers::build_http_client(Duration::from_secs(300))?;
        let normalized_base_url = base_url.trim_end_matches('/').to_string();

        Ok(Self {
            client,
            is_cloudflare_gateway: is_cloudflare_ai_gateway_base(&normalized_base_url),
            base_url: normalized_base_url,
            api_key: api_key.to_string(),
            gateway_token: gateway_token.map(|s| s.to_string()),
            extra_headers: extra_headers.unwrap_or_default(),
            max_tokens,
            reasoning_effort: None,
            llama_cpp_thinking: false,
            slot_routing: SlotRoutingConfig::default(),
            streaming: false,
            structured_answer_models: Vec::new(),
        })
    }

    /// Enable reasoning/thinking tokens with the given effort level.
    pub fn with_reasoning_effort(mut self, effort: Option<String>) -> Self {
        self.reasoning_effort = effort;
        self
    }

    /// Enable llama.cpp chat-template thinking controls.
    pub fn with_llama_cpp_thinking(mut self, enabled: bool) -> Self {
        self.llama_cpp_thinking = enabled;
        self
    }

    /// Configure llama.cpp slot routing for this provider. When disabled (the
    /// default), no `id_slot` field is ever emitted.
    pub fn with_slot_routing(mut self, slot_routing: SlotRoutingConfig) -> Self {
        self.slot_routing = slot_routing;
        self
    }

    /// Enable SSE streaming transport (off by default).
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Enable the structured final-answer second pass for these exact model
    /// names (off by default — empty list).
    pub fn with_structured_answer_models(mut self, models: Vec<String>) -> Self {
        self.structured_answer_models = models;
        self
    }

    /// Parse a chat-completions response body (shared by the buffered and
    /// streaming transports — the stream accumulator reconstructs this
    /// exact shape).
    fn parse_chat_response_body(data: &Value, model: &str) -> anyhow::Result<ProviderResponse> {
        let choice = data["choices"].get(0).ok_or_else(|| {
            error!("Provider response missing choices[0]");
            ProviderError::malformed_shape(
                "Malformed response from LLM provider (missing choices[0])",
            )
        })?;
        let message = choice.get("message").ok_or_else(|| {
            error!("Provider response missing choices[0].message");
            ProviderError::malformed_shape(
                "Malformed response from LLM provider (missing choices[0].message)",
            )
        })?;

        let content = message
            .get("content")
            .and_then(crate::agent::vision::content_value_as_text);

        let mut tool_calls = Vec::new();
        if let Some(tcs) = message["tool_calls"].as_array() {
            debug!(
                "Raw tool_calls from provider: {}",
                serde_json::to_string(tcs).unwrap_or_default()
            );
            for tc in tcs {
                let extra_content = tc.get("extra_content").filter(|v| !v.is_null()).cloned();

                tool_calls.push(ToolCall {
                    id: tc["id"].as_str().unwrap_or("").to_string(),
                    name: normalize_tool_name(tc["function"]["name"].as_str().unwrap_or("")),
                    arguments: tc["function"]["arguments"]
                        .as_str()
                        .unwrap_or("{}")
                        .to_string(),
                    extra_content,
                });
            }
        }

        // llama.cpp reports server-side timing in a top-level `timings` object
        // (sibling of `usage`): `prompt_ms` is prefill, `predicted_ms` is decode.
        // Capturing these splits end-to-end latency into its two components so a
        // slow call can be attributed to cold prefill vs long decode vs queue
        // contention without guessing from token counts.
        let timings = data.get("timings");
        let prompt_ms = timings
            .and_then(|t| t.get("prompt_ms"))
            .and_then(Value::as_f64);
        let decode_ms = timings
            .and_then(|t| t.get("predicted_ms"))
            .and_then(Value::as_f64);

        let usage = data.get("usage").and_then(|u| {
            Some(TokenUsage {
                input_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
                output_tokens: u.get("completion_tokens")?.as_u64()? as u32,
                cached_input_tokens: Self::cached_input_tokens_from_usage(u),
                cache_creation_input_tokens: None,
                model: model.to_string(),
                prompt_ms,
                decode_ms,
            })
        });

        // Detect token-limit truncation: when finish_reason is "length", the
        // model hit its max_tokens ceiling and the response was cut off
        // mid-generation.  Surface this so the agent loop can retry or degrade
        // gracefully instead of treating the empty/broken output as intentional.
        let finish_reason = choice
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let response_note = if finish_reason == "length" {
            warn!(
                model,
                output_tokens = usage.as_ref().map(|u| u.output_tokens).unwrap_or(0),
                "LLM response truncated at token limit (finish_reason=length)"
            );
            Some(
                "Response was truncated because it hit the model's maximum output token limit. \
                  The output may be incomplete or missing tool calls."
                    .to_string(),
            )
        } else {
            None
        };

        // Extract reasoning/thinking tokens from the response.
        // OpenRouter returns reasoning as `message.reasoning` (string) or
        // `message.reasoning_details` (array of objects with `text` field).
        let thinking = message
            .get("reasoning")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                message
                    .get("reasoning_content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .or_else(|| {
                message
                    .get("reasoning_details")
                    .and_then(|v| v.as_array())
                    .map(|details| {
                        details
                            .iter()
                            .filter_map(|d| d.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .filter(|s| !s.is_empty())
            });

        if thinking.is_some() {
            info!(
                model,
                thinking_len = thinking.as_ref().map(|t| t.len()).unwrap_or(0),
                "Reasoning tokens received from provider"
            );
        }

        // Diagnostic (no behavior change): when the structured reasoning field is
        // empty but `content` reads as reasoning, the provider/server did not
        // separate the model's chain-of-thought — the exact condition behind the
        // intermittent reasoning-leak (raw scratchpad reaching a user). Capture
        // the final model + a bounded snippet so the next real occurrence reveals
        // whether the leak is delimited (deterministic strip suffices) or
        // undelimited prose (needs envelope/schema enforcement). See
        // docs/final-answer-structured-output-audit.md.
        if thinking.as_deref().map(str::trim).unwrap_or("").is_empty() {
            if let Some(content_str) = content.as_deref() {
                if let Some(signal) = classify_possible_reasoning_leak(content_str) {
                    let snippet: String = content_str.trim().chars().take(500).collect();
                    warn!(
                        model,
                        signal = ?signal,
                        content_len = content_str.len(),
                        has_tool_calls = !tool_calls.is_empty(),
                        snippet = %snippet,
                        "Possible reasoning leak: content looks reasoning-shaped but \
                         structured reasoning field is empty (candidate final-answer leak)"
                    );
                }
            }
        }

        Ok(ProviderResponse {
            content,
            tool_calls,
            usage,
            thinking,
            response_note,
        })
    }

    /// Send the request with SSE streaming and accumulate deltas into a
    /// non-streaming-shaped body. A stream that dies or stalls after
    /// partial text is finalized as a `length` cutoff so the agent loop's
    /// truncation recovery continues the response instead of losing it.
    async fn execute_streaming_request(
        &self,
        request: reqwest::RequestBuilder,
        model: &str,
        hard_timeout: Duration,
    ) -> anyhow::Result<Value> {
        use futures::StreamExt;
        // Max silence between chunks before declaring the stream stalled.
        const STREAM_CHUNK_GAP_TIMEOUT: Duration = Duration::from_secs(120);
        let deadline = tokio::time::Instant::now() + hard_timeout;

        let resp = request.send().await.map_err(|e| {
            error!("HTTP request failed: {}", e);
            anyhow::Error::from(ProviderError::network(&e))
        })?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            error!(status = %status, "Provider API error: {}", text);
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        let mut framer = crate::providers::streaming::SseFramer::default();
        let mut acc = crate::providers::streaming::StreamAccumulator::default();
        let mut stream = resp.bytes_stream();
        let interrupted = 'consume: loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let gap = STREAM_CHUNK_GAP_TIMEOUT.min(remaining);
            if gap.is_zero() {
                break 'consume true;
            }
            match tokio::time::timeout(gap, stream.next()).await {
                Ok(Some(Ok(bytes))) => {
                    for payload in framer.feed(&bytes) {
                        if !acc.apply_payload(&payload) {
                            break 'consume false; // [DONE]
                        }
                    }
                }
                Ok(Some(Err(e))) => {
                    if !acc.has_partial_output() {
                        error!(model, "Stream error before any output: {}", e);
                        return Err(ProviderError::network(&e).into());
                    }
                    warn!(
                        model,
                        error = %e,
                        "Stream died mid-response; recovering partial output as length cutoff"
                    );
                    break 'consume true;
                }
                Ok(None) => break 'consume false, // graceful end without [DONE]
                Err(_elapsed) => {
                    if !acc.has_partial_output() {
                        return Err(ProviderError::timeout_msg(
                            "LLM streaming call timed out with no output",
                        )
                        .into());
                    }
                    warn!(
                        model,
                        "Stream stalled; recovering partial output as length cutoff"
                    );
                    break 'consume true;
                }
            }
        };
        Ok(acc.finalize(interrupted))
    }

    fn with_auth_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut request = request.header("Authorization", format!("Bearer {}", self.api_key));
        if let Some(token) = self.gateway_token.as_deref() {
            if token.is_empty() {
                for (k, v) in &self.extra_headers {
                    request = request.header(k, v);
                }
                return request;
            }
            request = request.header("cf-aig-authorization", format!("Bearer {}", token));
        }

        for (k, v) in &self.extra_headers {
            request = request.header(k, v);
        }
        request
    }

    fn parse_models_response(text: &str) -> anyhow::Result<Vec<String>> {
        let data: Value = serde_json::from_str(text)?;
        // OpenAI format: { "data": [{ "id": "model-name" }, ...] }
        let models = data["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        Ok(models)
    }

    fn cached_input_tokens_from_usage(usage: &Value) -> Option<u32> {
        usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens"))
            .and_then(Value::as_u64)
            .or_else(|| {
                usage
                    .get("input_tokens_details")
                    .and_then(|details| details.get("cached_tokens"))
                    .and_then(Value::as_u64)
            })
            .or_else(|| usage.get("cached_tokens").and_then(Value::as_u64))
            .map(|tokens| tokens.min(u32::MAX as u64) as u32)
    }

    fn cloudflare_models_fallback_url(&self) -> String {
        if self.base_url.ends_with("/compat") {
            format!("{}/v1/models", self.base_url)
        } else {
            format!("{}/compat/v1/models", self.base_url)
        }
    }

    fn build_request_body(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> Value {
        // Strip extra_content from tool_calls before sending — the OpenAI-compatible
        // endpoint doesn't understand it (it's used internally for Gemini native round-trip).
        let mut messages_cleaned: Vec<Value> = messages.to_vec();
        for msg in &mut messages_cleaned {
            if let Some(tcs) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                for tc in tcs {
                    if let Some(obj) = tc.as_object_mut() {
                        obj.remove("extra_content");
                    }
                }
            }
        }

        let mut body = json!({
            "model": model,
            "messages": messages_cleaned,
        });

        // Per-call override takes priority (e.g. billing-aware retry with reduced cap).
        if let Some(override_mt) = options.max_tokens_override {
            body["max_tokens"] = json!(override_mt);
        } else if let Some(max_tokens) = self.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }

        if !tools.is_empty() {
            body["tools"] = json!(tools);
            // Enable parallel tool calls so the model can batch independent
            // tool invocations in a single turn, reducing iteration count.
            body["parallel_tool_calls"] = json!(true);
        }
        if !tools.is_empty() {
            match &options.tool_choice {
                ToolChoiceMode::Auto => {}
                ToolChoiceMode::None => body["tool_choice"] = json!("none"),
                ToolChoiceMode::Required => body["tool_choice"] = json!("required"),
                ToolChoiceMode::Specific(name) => {
                    body["tool_choice"] = json!({
                        "type": "function",
                        "function": { "name": name }
                    });
                }
            }
        } else if !matches!(
            options.tool_choice,
            ToolChoiceMode::Auto | ToolChoiceMode::None
        ) {
            // `None` with no tools is a no-op by definition (aux JSON-schema
            // calls pass it explicitly) — only Required/Specific indicate a
            // real caller bug worth warning about.
            warn!(
                tool_choice = ?options.tool_choice,
                "Ignoring tool_choice that requires tools because no tools were provided"
            );
        }

        // Include reasoning/thinking tokens when configured.
        // Works with any OpenAI-compatible provider that supports the
        // `reasoning` parameter (e.g., thinking models like Kimi K2 Thinking,
        // DeepSeek R1, Claude with extended thinking).
        // Per-call override takes priority (e.g., truncation recovery reduces thinking).
        // "off" means: disable reasoning entirely (don't include the parameter).
        let effective_reasoning = options
            .reasoning_effort_override
            .as_deref()
            .or(self.reasoning_effort.as_deref());
        if let Some(effort) = effective_reasoning {
            if effort != "off" {
                body["reasoning"] = json!({
                    "effort": effort,
                });
            }
            // else: "off" → omit reasoning param to disable thinking entirely
        }

        if self.llama_cpp_thinking && effective_reasoning != Some("off") {
            body["chat_template_kwargs"] = json!({
                "enable_thinking": true,
            });
            body["reasoning_format"] = json!("deepseek");
        }

        // llama.cpp KV-cache slot pinning. Only emitted when explicitly enabled
        // — cloud APIs reject unknown params, so disabled is the safe default.
        // The interactive loop passes `Some(interactive_slot)`; every other call
        // leaves `id_slot` as `None` and is mapped to the background slot.
        if self.slot_routing.enabled {
            body["id_slot"] = json!(options.id_slot.unwrap_or(self.slot_routing.background_slot));
        }

        if self.streaming {
            body["stream"] = json!(true);
            // Ask for usage on the final chunk so token accounting (budgets,
            // billing-aware retries) keeps working under streaming.
            body["stream_options"] = json!({ "include_usage": true });
        }

        // OpenAI audio models require `modalities` when `input_audio` blocks are present.
        // Text-only agent responses: modalities = ["text"] only (no audio output).
        // If a provider rejects this, it may also require `audio: { voice, format }` —
        // validate against your configured audio model and adjust if needed.
        if crate::providers::multimodal::messages_contain_audio_blocks(messages) {
            body["modalities"] = json!(["text"]);
        }

        match &options.response_mode {
            ResponseMode::Text => {}
            ResponseMode::JsonObject => {
                body["response_format"] = json!({ "type": "json_object" });
            }
            ResponseMode::JsonSchema {
                name,
                schema,
                strict,
            } => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": strict
                    }
                });
            }
        }

        body
    }

    /// Execute one buffered/streaming chat request. This is the raw
    /// transport; `chat_with_options` layers the structured final-answer
    /// interception on top of it.
    async fn execute_chat_request(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        let body = self.build_request_body(model, messages, tools, options);

        // DEBUG: Log tool count and reasoning config sent to provider
        let tool_count = body
            .get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let has_reasoning = body.get("reasoning").is_some();
        info!(
            model,
            tool_count, has_reasoning, "Sending request to LLM provider"
        );

        let url = format!("{}/chat/completions", self.base_url);
        info!(
            model,
            url = %url,
            tools = tools.len(),
            response_mode = ?options.response_mode,
            tool_choice = ?options.tool_choice,
            "Calling LLM API"
        );

        let request = self
            .with_auth_headers(self.client.post(&url))
            .header("Content-Type", "application/json")
            .json(&body);

        // Safety-net timeout independent of reqwest's client-level timeout.
        // reqwest's timeout can be bypassed when the server trickles data
        // (e.g. keep-alive pings, chunked encoding). This hard wall-clock
        // cap ensures we never block the agent loop indefinitely.
        const LLM_CALL_HARD_TIMEOUT: Duration = Duration::from_secs(360);

        if self.streaming {
            let data = self
                .execute_streaming_request(request, model, LLM_CALL_HARD_TIMEOUT)
                .await?;
            return Self::parse_chat_response_body(&data, model);
        }

        let (resp, text) = match tokio::time::timeout(LLM_CALL_HARD_TIMEOUT, async {
            let resp = request.send().await.map_err(|e| {
                error!("HTTP request failed: {}", e);
                anyhow::Error::from(ProviderError::network(&e))
            })?;
            let status = resp.status();
            let text = resp.text().await.map_err(|e| {
                error!("Failed to read response body: {}", e);
                anyhow::Error::from(ProviderError::network(&e))
            })?;
            Ok::<(u16, String), anyhow::Error>((status.as_u16(), text))
        })
        .await
        {
            Ok(Ok((status_code, text))) => (status_code, text),
            Ok(Err(e)) => return Err(e),
            Err(_elapsed) => {
                error!(
                    model,
                    timeout_secs = LLM_CALL_HARD_TIMEOUT.as_secs(),
                    "LLM API call exceeded hard timeout"
                );
                return Err(ProviderError::timeout_msg(
                    "LLM API call timed out (hard wall-clock limit)",
                )
                .into());
            }
        };

        let status = reqwest::StatusCode::from_u16(resp).unwrap_or(reqwest::StatusCode::OK);

        if !status.is_success() {
            error!(status = %status, "Provider API error: {}", text);
            debug!(
                "Failed request body: {}",
                serde_json::to_string(&body).unwrap_or_default()
            );
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        // Safely truncate for debug logging, respecting UTF-8 char boundaries
        let truncated = if text.len() > 2000 {
            let mut end = 2000;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            &text[..end]
        } else {
            &text
        };
        debug!("Provider response: {}", truncated);

        let data: Value = serde_json::from_str(&text).map_err(|e| {
            error!("Failed to parse provider response JSON: {}", e);
            ProviderError::malformed_parse(format!(
                "Malformed response from LLM provider (JSON parse error: {})",
                e
            ))
        })?;
        Self::parse_chat_response_body(&data, model)
    }

    /// Second, tool-less, grammar-constrained pass over the same conversation
    /// producing the final user-visible answer as `{reasoning, response}`.
    ///
    /// Exists because some local models (Gemma) emit raw undelimited
    /// chain-of-thought AS the answer body on challenge/truncation-recovery
    /// turns, and no thinking-config lever cleans it — the only reliable fix
    /// is an output boundary on the answer turn. A JSON grammar cannot
    /// coexist with live tools, hence intercept-and-reissue. Never errors:
    /// every failure mode degrades to the draft (if clean) or an honest
    /// failure notice (if the draft is leaked reasoning).
    async fn run_structured_answer_pass(
        &self,
        model: &str,
        messages: &[Value],
        options: &ChatOptions,
        first: ProviderResponse,
    ) -> ProviderResponse {
        let draft = first.content.clone().unwrap_or_default();
        let draft_leak = classify_possible_reasoning_leak(&draft);
        info!(
            model,
            draft_len = draft.len(),
            draft_leak = ?draft_leak,
            "Structured answer pass: re-issuing final answer through schema"
        );
        let second_messages = append_answer_instruction(messages);
        let mut second_options = structured_answer_chat_options(options);
        let mut usage = first.usage.clone();
        let mut retried = false;
        let resolution = loop {
            let second = match self
                .execute_chat_request(model, &second_messages, &[], &second_options)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    warn!(model, error = %e, "Structured answer pass request failed");
                    // Same policy as an unusable parse; transport retries are
                    // the caller's job, not ours.
                    break resolve_structured_answer(
                        StructuredAnswerParse::Unusable,
                        false,
                        retried,
                        draft_leak.is_some(),
                    );
                }
            };
            usage = merge_token_usage(usage, second.usage.clone());
            // parse_chat_response_body sets response_note only for
            // finish_reason=length, so its presence means truncation.
            let second_truncated = second.response_note.is_some();
            let parse = parse_structured_answer(second.content.as_deref().unwrap_or(""));
            match resolve_structured_answer(parse, second_truncated, retried, draft_leak.is_some())
            {
                SecondPassResolution::RetryLarger => {
                    retried = true;
                    let bumped = options
                        .max_tokens_override
                        .or(self.max_tokens)
                        .unwrap_or(2048)
                        .saturating_mul(2);
                    info!(
                        model,
                        bumped, "Structured answer truncated; retrying with larger cap"
                    );
                    second_options.max_tokens_override = Some(bumped);
                }
                other => break other,
            }
        };
        match resolution {
            SecondPassResolution::Deliver {
                response,
                reasoning,
                truncated,
            } => {
                // Permanent telemetry: the "no rework in response" instruction
                // is soft on a contract-unreliable model — this is how we find
                // out when it stops holding.
                if let Some(signal) = classify_possible_reasoning_leak(&response) {
                    warn!(
                        model,
                        signal = ?signal,
                        "Structured answer still reasoning-shaped after schema pass"
                    );
                }
                let response_note = truncated.then(|| {
                    "Final answer was truncated at the token limit; the delivered text may be \
                     incomplete."
                        .to_string()
                });
                ProviderResponse {
                    content: Some(response),
                    tool_calls: Vec::new(),
                    usage,
                    thinking: reasoning.or(first.thinking),
                    response_note,
                }
            }
            SecondPassResolution::UseDraft => {
                info!(
                    model,
                    "Structured answer pass unusable; draft is clean, shipping draft"
                );
                ProviderResponse { usage, ..first }
            }
            SecondPassResolution::Suppress => {
                warn!(
                    model,
                    draft_snippet = %draft.chars().take(300).collect::<String>(),
                    "Structured answer pass unusable AND draft is reasoning-shaped; \
                     suppressing draft"
                );
                ProviderResponse {
                    content: Some(STRUCTURED_ANSWER_SUPPRESSED_MSG.to_string()),
                    tool_calls: Vec::new(),
                    usage,
                    thinking: first.thinking,
                    response_note: Some(
                        "Structured answer pass failed and the draft looked like leaked \
                         reasoning; delivered an honest failure notice instead."
                            .to_string(),
                    ),
                }
            }
            SecondPassResolution::RetryLarger => unreachable!("retry handled in loop"),
        }
    }
}

#[async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
    async fn chat(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
    ) -> anyhow::Result<ProviderResponse> {
        self.chat_with_options(model, messages, tools, &ChatOptions::default())
            .await
    }

    async fn chat_with_options(
        &self,
        model: &str,
        messages: &[Value],
        tools: &[Value],
        options: &ChatOptions,
    ) -> anyhow::Result<ProviderResponse> {
        let first = self
            .execute_chat_request(model, messages, tools, options)
            .await?;
        if !needs_structured_answer_pass(
            &self.structured_answer_models,
            model,
            tools,
            options,
            &first,
        ) {
            return Ok(first);
        }
        Ok(self
            .run_structured_answer_pass(model, messages, options, first)
            .await)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let primary_url = format!("{}/models", self.base_url);
        let primary_resp = self
            .with_auth_headers(self.client.get(&primary_url))
            .send()
            .await?;

        let primary_status = primary_resp.status();
        let primary_text = primary_resp.text().await?;
        if primary_status.is_success() {
            return Self::parse_models_response(&primary_text);
        }

        let should_try_cf_fallback =
            self.is_cloudflare_gateway && matches!(primary_status.as_u16(), 404 | 405);
        if !should_try_cf_fallback {
            anyhow::bail!(
                "Failed to list models at '{}' ({}): {}",
                primary_url,
                primary_status,
                primary_text
            );
        }

        let fallback_url = self.cloudflare_models_fallback_url();
        let fallback_resp = self
            .with_auth_headers(self.client.get(&fallback_url))
            .send()
            .await?;
        let fallback_status = fallback_resp.status();
        let fallback_text = fallback_resp.text().await?;

        if fallback_status.is_success() {
            return Self::parse_models_response(&fallback_text);
        }

        anyhow::bail!(
            "Failed to list models at '{}' ({}): {}. Fallback '{}' ({}): {}",
            primary_url,
            primary_status,
            primary_text,
            fallback_url,
            fallback_status,
            fallback_text
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_https_accepted() {
        let result = validate_base_url("https://api.openai.com");
        assert!(result.is_ok());
    }

    #[test]
    fn reasoning_leak_delimited_think_tags() {
        let c = "<think>The user is testing me, I should keep it brief.</think>\n\nUnderstood.";
        assert_eq!(
            classify_possible_reasoning_leak(c),
            Some(ReasoningLeakSignal::Delimited)
        );
    }

    #[test]
    fn reasoning_leak_scaffold_echo_is_prose_signal() {
        // The 2026-06-30 incident: internal "Original request:" / "Follow-up:"
        // scaffolding echoed verbatim into an outbound reply.
        let c = "It's almost like they are providing a log or a summary of their own \
                 test.\nOriginal request: [text]\nFollow-up: [text]\nI'll acknowledge it.";
        assert_eq!(
            classify_possible_reasoning_leak(c),
            Some(ReasoningLeakSignal::UndelimitedProse)
        );
    }

    #[test]
    fn reasoning_leak_undelimited_deliberation_prose() {
        // Reconstructed from the incident: first-person meta-deliberation with
        // multiple distinct markers, no delimiter, no scaffold echo.
        let c = "It seems the user is providing context or summarizing what they were \
                 doing. Since they said \"nothing needed\", I should acknowledge the \
                 explanation. I'll keep it brief as per the communication preferences.";
        assert_eq!(
            classify_possible_reasoning_leak(c),
            Some(ReasoningLeakSignal::UndelimitedProse)
        );
    }

    #[test]
    fn reasoning_leak_clean_answer_does_not_fire() {
        // The actual clean answer the live server produced for the same prompt.
        let c = "Yes, a brief acknowledgment like \"Understood\" is polite and confirms \
                 you are functioning correctly.";
        assert_eq!(classify_possible_reasoning_leak(c), None);
    }

    #[test]
    fn reasoning_leak_single_marker_does_not_fire() {
        // One incidental deliberation phrase in a genuine answer must not trip it.
        let c = "The user is asking about the weather, and it looks clear all week.";
        assert_eq!(classify_possible_reasoning_leak(c), None);
    }

    #[test]
    fn reasoning_leak_short_content_does_not_fire() {
        assert_eq!(classify_possible_reasoning_leak("<think>hi</think>"), None);
    }

    // === Structured final-answer second pass ===

    #[test]
    fn structured_answer_schema_orders_reasoning_before_response() {
        // llama.cpp builds its grammar in wire order, and serde_json serializes
        // keys alphabetically — the drain field must decode before the
        // user-visible one or the whole mechanism is defeated.
        let serialized = serde_json::to_string(&structured_answer_schema()).unwrap();
        let r = serialized.find("\"reasoning\"").expect("reasoning field");
        let a = serialized.find("\"response\"").expect("response field");
        assert!(
            r < a,
            "reasoning must serialize before response: {serialized}"
        );
        let schema = structured_answer_schema();
        assert_eq!(schema["additionalProperties"], json!(false));
        let required = schema["required"].as_array().unwrap();
        assert_eq!(required.len(), 2);
    }

    #[test]
    fn parse_structured_answer_complete() {
        let raw = r#"{"reasoning":"6 r's, 9.11 < 9.8","response":"There are 6 r's, and 9.11 is smaller."}"#;
        match parse_structured_answer(raw) {
            StructuredAnswerParse::Complete {
                reasoning,
                response,
            } => {
                assert_eq!(reasoning, "6 r's, 9.11 < 9.8");
                assert_eq!(response, "There are 6 r's, and 9.11 is smaller.");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn parse_structured_answer_fenced_json() {
        // A non-grammar backend (cloud fallback) may fence the JSON.
        let raw = "```json\n{\"reasoning\":\"r\",\"response\":\"clean\"}\n```";
        match parse_structured_answer(raw) {
            StructuredAnswerParse::Complete { response, .. } => assert_eq!(response, "clean"),
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn parse_structured_answer_truncated_mid_response_salvages_prefix() {
        // max_tokens hit mid-string: invalid JSON, but the user-visible prefix
        // is recoverable (with escapes decoded).
        let raw = r#"{"reasoning":"long deliberation","response":"Line one.\nLine tw"#;
        match parse_structured_answer(raw) {
            StructuredAnswerParse::Partial { response } => {
                assert_eq!(response, "Line one.\nLine tw");
            }
            other => panic!("expected Partial, got {other:?}"),
        }
    }

    #[test]
    fn parse_structured_answer_truncated_mid_reasoning_is_unusable() {
        let raw = r#"{"reasoning":"I will now re-count the letters carefu"#;
        assert!(matches!(
            parse_structured_answer(raw),
            StructuredAnswerParse::Unusable
        ));
    }

    #[test]
    fn parse_structured_answer_escapes_in_salvage() {
        let raw = r#"{"reasoning":"r","response":"She said \"hi\" A\\B"#;
        match parse_structured_answer(raw) {
            StructuredAnswerParse::Partial { response } => {
                assert_eq!(response, "She said \"hi\" A\\B");
            }
            other => panic!("expected Partial, got {other:?}"),
        }
    }

    #[test]
    fn parse_structured_answer_garbage_is_unusable() {
        assert!(matches!(
            parse_structured_answer("not json at all"),
            StructuredAnswerParse::Unusable
        ));
        assert!(matches!(
            parse_structured_answer(""),
            StructuredAnswerParse::Unusable
        ));
    }

    #[test]
    fn parse_structured_answer_empty_response_field_is_unusable() {
        let raw = r#"{"reasoning":"thought about it","response":"   "}"#;
        assert!(matches!(
            parse_structured_answer(raw),
            StructuredAnswerParse::Unusable
        ));
    }

    #[test]
    fn append_answer_instruction_extends_trailing_user_text() {
        let messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"Are you sure?"}),
        ];
        let out = append_answer_instruction(&messages);
        assert_eq!(out.len(), 2);
        let content = out[1]["content"].as_str().unwrap();
        assert!(content.starts_with("Are you sure?"));
        assert!(content.contains("\"response\""));
        // Prefix untouched so the llama.cpp prompt cache reuses the first
        // pass's prefill.
        assert_eq!(out[0], messages[0]);
    }

    #[test]
    fn append_answer_instruction_appends_new_user_message_after_non_user() {
        let messages = vec![
            json!({"role":"user","content":"do it"}),
            json!({"role":"assistant","content":null,"tool_calls":[]}),
            json!({"role":"tool","content":"ok","tool_call_id":"1"}),
        ];
        let out = append_answer_instruction(&messages);
        assert_eq!(out.len(), 4);
        assert_eq!(out[3]["role"], "user");
        assert!(out[3]["content"].as_str().unwrap().contains("\"response\""));
    }

    #[test]
    fn append_answer_instruction_appends_new_user_message_for_block_content() {
        let messages = vec![json!({"role":"user","content":[{"type":"text","text":"hi"}]})];
        let out = append_answer_instruction(&messages);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1]["role"], "user");
    }

    fn text_response(content: &str) -> ProviderResponse {
        ProviderResponse {
            content: Some(content.to_string()),
            tool_calls: vec![],
            usage: None,
            thinking: None,
            response_note: None,
        }
    }

    #[test]
    fn needs_pass_fires_for_flagged_model_final_answer_with_tools_live() {
        let models = vec!["gemma-4-26b".to_string()];
        let tools = vec![json!({"type":"function"})];
        assert!(needs_structured_answer_pass(
            &models,
            "gemma-4-26b",
            &tools,
            &ChatOptions::default(),
            &text_response("Here is your answer.")
        ));
    }

    #[test]
    fn needs_pass_skips_unflagged_model() {
        let models = vec!["gemma-4-26b".to_string()];
        let tools = vec![json!({"type":"function"})];
        assert!(!needs_structured_answer_pass(
            &models,
            "other-model",
            &tools,
            &ChatOptions::default(),
            &text_response("answer")
        ));
        assert!(!needs_structured_answer_pass(
            &[],
            "gemma-4-26b",
            &tools,
            &ChatOptions::default(),
            &text_response("answer")
        ));
    }

    #[test]
    fn needs_pass_skips_when_no_tools_live() {
        // Second pass itself runs with no tools — this is the recursion guard.
        let models = vec!["gemma-4-26b".to_string()];
        assert!(!needs_structured_answer_pass(
            &models,
            "gemma-4-26b",
            &[],
            &ChatOptions::default(),
            &text_response("answer")
        ));
    }

    #[test]
    fn needs_pass_skips_tool_call_responses() {
        let models = vec!["gemma-4-26b".to_string()];
        let tools = vec![json!({"type":"function"})];
        let mut resp = text_response("thinking about which tool");
        resp.tool_calls.push(ToolCall {
            id: "1".into(),
            name: "web_search".into(),
            arguments: "{}".into(),
            extra_content: None,
        });
        assert!(!needs_structured_answer_pass(
            &models,
            "gemma-4-26b",
            &tools,
            &ChatOptions::default(),
            &resp
        ));
    }

    #[test]
    fn needs_pass_skips_non_text_response_mode() {
        // Callers that already requested structured output manage their own shape.
        let models = vec!["gemma-4-26b".to_string()];
        let tools = vec![json!({"type":"function"})];
        let opts = ChatOptions {
            response_mode: ResponseMode::JsonObject,
            ..Default::default()
        };
        assert!(!needs_structured_answer_pass(
            &models,
            "gemma-4-26b",
            &tools,
            &opts,
            &text_response("{}")
        ));
    }

    #[test]
    fn needs_pass_skips_blank_content() {
        let models = vec!["gemma-4-26b".to_string()];
        let tools = vec![json!({"type":"function"})];
        assert!(!needs_structured_answer_pass(
            &models,
            "gemma-4-26b",
            &tools,
            &ChatOptions::default(),
            &text_response("   ")
        ));
        let mut resp = text_response("");
        resp.content = None;
        assert!(!needs_structured_answer_pass(
            &models,
            "gemma-4-26b",
            &tools,
            &ChatOptions::default(),
            &resp
        ));
    }

    #[test]
    fn structured_answer_options_disable_thinking_and_tools() {
        let base = ChatOptions {
            id_slot: Some(3),
            max_tokens_override: Some(1000),
            reasoning_effort_override: Some("high".into()),
            ..Default::default()
        };
        let opts = structured_answer_chat_options(&base);
        assert_eq!(opts.reasoning_effort_override.as_deref(), Some("off"));
        assert_eq!(opts.tool_choice, ToolChoiceMode::None);
        // Same slot: the second pass shares the first pass's KV prefix.
        assert_eq!(opts.id_slot, Some(3));
        assert_eq!(opts.max_tokens_override, Some(1000));
        match &opts.response_mode {
            ResponseMode::JsonSchema { name, strict, .. } => {
                assert_eq!(name, "final_answer");
                assert!(*strict);
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn second_pass_request_body_has_no_thinking_and_no_tools() {
        let provider = OpenAiCompatibleProvider::new("http://localhost:8081/v1", "k")
            .unwrap()
            .with_llama_cpp_thinking(true);
        let opts = structured_answer_chat_options(&ChatOptions::default());
        let body = provider.build_request_body(
            "gemma-4-26b",
            &[json!({"role":"user","content":"hi"})],
            &[],
            &opts,
        );
        assert!(body.get("tools").is_none());
        assert!(body.get("reasoning").is_none());
        assert!(
            body.get("chat_template_kwargs").is_none(),
            "thinking must be OFF on the answer pass"
        );
        assert_eq!(body["response_format"]["type"], "json_schema");
    }

    #[test]
    fn resolve_complete_delivers_answer_with_reasoning() {
        let parse = StructuredAnswerParse::Complete {
            reasoning: "checked twice".into(),
            response: "It is 6.".into(),
        };
        assert_eq!(
            resolve_structured_answer(parse, false, false, true),
            SecondPassResolution::Deliver {
                response: "It is 6.".into(),
                reasoning: Some("checked twice".into()),
                truncated: false,
            }
        );
    }

    #[test]
    fn resolve_partial_truncated_first_try_retries_larger() {
        let parse = StructuredAnswerParse::Partial {
            response: "half an ans".into(),
        };
        assert_eq!(
            resolve_structured_answer(parse, true, false, true),
            SecondPassResolution::RetryLarger
        );
    }

    #[test]
    fn resolve_partial_after_retry_delivers_truncated() {
        let parse = StructuredAnswerParse::Partial {
            response: "half an ans".into(),
        };
        assert_eq!(
            resolve_structured_answer(parse, true, true, true),
            SecondPassResolution::Deliver {
                response: "half an ans".into(),
                reasoning: None,
                truncated: true,
            }
        );
    }

    #[test]
    fn resolve_unusable_truncated_first_try_retries() {
        assert_eq!(
            resolve_structured_answer(StructuredAnswerParse::Unusable, true, false, true),
            SecondPassResolution::RetryLarger
        );
    }

    #[test]
    fn resolve_unusable_clean_draft_falls_back_to_draft() {
        // Most drafts are clean — a failed second pass must not lose them.
        assert_eq!(
            resolve_structured_answer(StructuredAnswerParse::Unusable, false, false, false),
            SecondPassResolution::UseDraft
        );
        assert_eq!(
            resolve_structured_answer(StructuredAnswerParse::Unusable, true, true, false),
            SecondPassResolution::UseDraft
        );
    }

    #[test]
    fn resolve_unusable_leaky_draft_suppresses() {
        // The one case that must never ship: raw chain-of-thought.
        assert_eq!(
            resolve_structured_answer(StructuredAnswerParse::Unusable, false, false, true),
            SecondPassResolution::Suppress
        );
        assert_eq!(
            resolve_structured_answer(StructuredAnswerParse::Unusable, true, true, true),
            SecondPassResolution::Suppress
        );
    }

    #[test]
    fn merge_token_usage_sums_both_passes() {
        let first = TokenUsage {
            input_tokens: 14_000,
            output_tokens: 300,
            cached_input_tokens: Some(1_000),
            cache_creation_input_tokens: None,
            model: "gemma-4-26b".into(),
            prompt_ms: Some(2_000.0),
            decode_ms: Some(9_000.0),
        };
        let second = TokenUsage {
            input_tokens: 14_400,
            output_tokens: 200,
            cached_input_tokens: Some(14_000),
            cache_creation_input_tokens: None,
            model: "gemma-4-26b".into(),
            prompt_ms: Some(500.0),
            decode_ms: Some(6_000.0),
        };
        let merged = merge_token_usage(Some(first), Some(second)).unwrap();
        assert_eq!(merged.input_tokens, 28_400);
        assert_eq!(merged.output_tokens, 500);
        assert_eq!(merged.cached_input_tokens, Some(15_000));
        assert_eq!(merged.prompt_ms, Some(2_500.0));
        assert_eq!(merged.decode_ms, Some(15_000.0));
        assert_eq!(merged.model, "gemma-4-26b");

        assert!(merge_token_usage(None, None).is_none());
        let only_second = merge_token_usage(
            None,
            Some(TokenUsage {
                input_tokens: 5,
                output_tokens: 7,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_eq!(only_second.input_tokens, 5);
        assert_eq!(only_second.output_tokens, 7);
    }

    #[test]
    fn parse_leaves_leaked_content_verbatim() {
        // The diagnostic must not alter delivery: reasoning-shaped content with no
        // structured reasoning field still round-trips unchanged.
        let leaked = "It seems the user is testing me. I should acknowledge. Understood.";
        let body = json!({
            "choices": [{
                "message": { "role": "assistant", "content": leaked },
                "finish_reason": "stop"
            }]
        });
        let resp =
            OpenAiCompatibleProvider::parse_chat_response_body(&body, "gemma-4-26b").unwrap();
        assert_eq!(resp.content.as_deref(), Some(leaked));
        assert!(resp.thinking.is_none());
    }

    #[test]
    fn test_http_localhost_accepted() {
        let result = validate_base_url("http://localhost:8080");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_127_accepted() {
        let result = validate_base_url("http://127.0.0.1:1234");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_ipv6_localhost_accepted() {
        let result = validate_base_url("http://[::1]:8080");
        assert!(result.is_ok());
    }

    #[test]
    fn test_http_remote_rejected() {
        let result = validate_base_url("http://api.example.com");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("HTTP is not allowed"),
            "Expected HTTP rejection error, got: {}",
            err
        );
    }

    #[test]
    fn test_ftp_rejected() {
        let result = validate_base_url("ftp://example.com");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Unsupported URL scheme"),
            "Expected unsupported scheme error, got: {}",
            err
        );
    }

    #[test]
    fn test_invalid_url_rejected() {
        let result = validate_base_url("not a url");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Invalid base_url"),
            "Expected invalid URL error, got: {}",
            err
        );
    }

    #[test]
    fn test_trailing_slash_trimmed() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1/", "test-key");
        assert!(
            provider.is_ok(),
            "Provider::new should succeed with trailing slash"
        );
        let provider = provider.unwrap();
        assert!(
            !provider.base_url.ends_with('/'),
            "base_url should not end with slash, got: {}",
            provider.base_url
        );
    }

    #[test]
    fn test_build_request_body_applies_required_tool_choice_and_json_schema() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"plan the task"})];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "search project files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    }
                }
            }
        })];
        let options = ChatOptions {
            response_mode: ResponseMode::JsonSchema {
                name: "intent_gate_v1".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "needs_tools": {"type": "boolean"}
                    },
                    "required": ["needs_tools"],
                    "additionalProperties": false
                }),
                strict: true,
            },
            tool_choice: ToolChoiceMode::Required,
            ..ChatOptions::default()
        };

        let body = provider.build_request_body("gpt-4o-mini", &messages, &tools, &options);

        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "intent_gate_v1"
        );
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn test_build_request_body_slot_routing_enabled_defaults_to_background_slot() {
        let provider = OpenAiCompatibleProvider::new("http://localhost:8080/v1", "test-key")
            .expect("provider should initialize")
            .with_slot_routing(SlotRoutingConfig {
                enabled: true,
                interactive_slot: 0,
                background_slot: 1,
            });
        let messages = vec![json!({"role":"user","content":"hi"})];
        // id_slot: None → provider maps to background_slot (1).
        let options = ChatOptions::default();
        let body = provider.build_request_body("local-model", &messages, &[], &options);
        assert_eq!(
            body["id_slot"], 1,
            "id_slot:None should map to background_slot when routing enabled"
        );
    }

    #[test]
    fn test_build_request_body_slot_routing_enabled_honors_explicit_id_slot() {
        let provider = OpenAiCompatibleProvider::new("http://localhost:8080/v1", "test-key")
            .expect("provider should initialize")
            .with_slot_routing(SlotRoutingConfig {
                enabled: true,
                interactive_slot: 0,
                background_slot: 1,
            });
        let messages = vec![json!({"role":"user","content":"hi"})];
        let options = ChatOptions {
            id_slot: Some(0),
            ..ChatOptions::default()
        };
        let body = provider.build_request_body("local-model", &messages, &[], &options);
        assert_eq!(
            body["id_slot"], 0,
            "explicit id_slot:Some(0) (interactive) should be used verbatim"
        );
    }

    #[test]
    fn test_build_request_body_slot_routing_disabled_omits_id_slot() {
        // Disabled (the default) — no id_slot key at all (cloud-API safe).
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"hi"})];
        // Even an explicit id_slot must NOT leak when routing is disabled.
        let options = ChatOptions {
            id_slot: Some(0),
            ..ChatOptions::default()
        };
        let body = provider.build_request_body("gpt-4o-mini", &messages, &[], &options);
        assert!(
            body.get("id_slot").is_none(),
            "id_slot must be omitted entirely when slot routing is disabled"
        );
    }

    #[test]
    fn llama_cpp_thinking_disabled_preserves_request_shape() {
        let provider = OpenAiCompatibleProvider::new("http://localhost:8080/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"hi"})];

        let body =
            provider.build_request_body("gemma-4-26b", &messages, &[], &ChatOptions::default());

        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body.get("reasoning_format").is_none());
    }

    #[test]
    fn llama_cpp_thinking_enabled_adds_template_controls() {
        let provider = OpenAiCompatibleProvider::new("http://localhost:8080/v1", "test-key")
            .expect("provider should initialize")
            .with_llama_cpp_thinking(true);
        let messages = vec![json!({"role":"user","content":"hi"})];

        let body =
            provider.build_request_body("gemma-4-26b", &messages, &[], &ChatOptions::default());

        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(body["reasoning_format"], "deepseek");
    }

    #[test]
    fn llama_cpp_thinking_off_override_restores_disabled_request_shape() {
        let provider = OpenAiCompatibleProvider::new("http://localhost:8080/v1", "test-key")
            .expect("provider should initialize")
            .with_llama_cpp_thinking(true);
        let messages = vec![json!({"role":"user","content":"hi"})];
        let options = ChatOptions {
            reasoning_effort_override: Some("off".to_string()),
            ..ChatOptions::default()
        };

        let body = provider.build_request_body("gemma-4-26b", &messages, &[], &options);

        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body.get("reasoning_format").is_none());
    }

    #[test]
    fn reasoning_content_is_private_and_separate_from_final_content() {
        let response = OpenAiCompatibleProvider::parse_chat_response_body(
            &json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "final answer",
                        "reasoning_content": "private trace"
                    }
                }]
            }),
            "gemma-4-26b",
        )
        .expect("response should parse");

        assert_eq!(response.thinking.as_deref(), Some("private trace"));
        assert_eq!(response.content.as_deref(), Some("final answer"));
    }

    #[test]
    fn parses_llama_cpp_timings_into_prefill_decode_split() {
        let response = OpenAiCompatibleProvider::parse_chat_response_body(
            &json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": { "role": "assistant", "content": "ok" }
                }],
                "usage": { "prompt_tokens": 12000, "completion_tokens": 200 },
                "timings": {
                    "prompt_n": 12000,
                    "prompt_ms": 34000.5,
                    "predicted_n": 200,
                    "predicted_ms": 6500.0
                }
            }),
            "gemma-4-26b",
        )
        .expect("response should parse");

        let usage = response.usage.expect("usage present");
        assert_eq!(usage.prompt_ms, Some(34000.5));
        assert_eq!(usage.decode_ms, Some(6500.0));
    }

    #[test]
    fn timings_absent_leaves_split_none() {
        let response = OpenAiCompatibleProvider::parse_chat_response_body(
            &json!({
                "choices": [{
                    "finish_reason": "stop",
                    "message": { "role": "assistant", "content": "ok" }
                }],
                "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
            }),
            "gemma-4-26b",
        )
        .expect("response should parse");

        let usage = response.usage.expect("usage present");
        assert_eq!(usage.prompt_ms, None);
        assert_eq!(usage.decode_ms, None);
    }

    #[test]
    fn test_build_request_body_ignores_non_auto_tool_choice_without_tools() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"answer in json"})];
        let options = ChatOptions {
            response_mode: ResponseMode::JsonObject,
            tool_choice: ToolChoiceMode::Required,
            ..ChatOptions::default()
        };

        let body = provider.build_request_body("gpt-4o-mini", &messages, &[], &options);

        assert!(body.get("tool_choice").is_none());
        assert_eq!(body["response_format"]["type"], "json_object");
    }

    // ---- Pillar A: provider-conversion order-preservation & determinism (Task 8) ----

    /// Pillar A tail-at-boundary design requires the OpenAI adapter to emit
    /// system messages INLINE, in source order — never hoisted to a top-level
    /// field, never merged with adjacent system messages. Feed the canonical
    /// `[system(core), assistant, user, system(tail), user]` sequence and assert
    /// the emitted `messages` array preserves count, order, and roles exactly.
    #[test]
    fn test_pillar_a_openai_preserves_message_count_order_and_roles() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![
            json!({"role":"system","content":"core prompt"}),
            json!({"role":"assistant","content":"prior turn"}),
            json!({"role":"user","content":"first ask"}),
            json!({"role":"system","content":"tail directive"}),
            json!({"role":"user","content":"second ask"}),
        ];

        let body =
            provider.build_request_body("gpt-4o-mini", &messages, &[], &ChatOptions::default());

        let out = body["messages"].as_array().expect("messages array");
        // Count preserved — no merge/hoist drops a message.
        assert_eq!(out.len(), messages.len(), "message count must be preserved");
        // Order + roles preserved element-for-element; system stays INLINE.
        let expected_roles = ["system", "assistant", "user", "system", "user"];
        for (i, expected_role) in expected_roles.iter().enumerate() {
            assert_eq!(
                out[i]["role"], *expected_role,
                "role at index {i} must be preserved inline (no hoist/merge)"
            );
        }
        // The two system messages remain distinct and in place — not merged.
        assert_eq!(out[0]["content"], "core prompt");
        assert_eq!(out[3]["content"], "tail directive");
        // No top-level `system` field — OpenAI keeps system inline.
        assert!(
            body.get("system").is_none(),
            "OpenAI adapter must not hoist system to a top-level field"
        );
    }

    /// Cache-prefix stability at the adapter boundary: converting `seq` and
    /// `seq + [tool exchange]` yields message arrays whose first `len(seq)`
    /// elements are byte-identical (a tool exchange appended after the prefix
    /// does not perturb the prefix).
    #[test]
    fn test_pillar_a_openai_prefix_stable_when_tool_exchange_appended() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let seq = vec![
            json!({"role":"system","content":"core prompt"}),
            json!({"role":"assistant","content":"prior turn"}),
            json!({"role":"user","content":"first ask"}),
            json!({"role":"system","content":"tail directive"}),
            json!({"role":"user","content":"second ask"}),
        ];
        let mut seq_extended = seq.clone();
        seq_extended.push(json!({
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": { "name": "search_files", "arguments": "{}" }
            }]
        }));
        seq_extended.push(json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "tool output"
        }));

        let body_a = provider.build_request_body("gpt-4o-mini", &seq, &[], &ChatOptions::default());
        let body_b =
            provider.build_request_body("gpt-4o-mini", &seq_extended, &[], &ChatOptions::default());

        let arr_a = body_a["messages"].as_array().expect("messages array");
        let arr_b = body_b["messages"].as_array().expect("messages array");
        assert_eq!(arr_b.len(), arr_a.len() + 2);
        // First len(seq) elements byte-identical (compare serialized prefix).
        for i in 0..seq.len() {
            assert_eq!(
                serde_json::to_string(&arr_a[i]).unwrap(),
                serde_json::to_string(&arr_b[i]).unwrap(),
                "prefix element {i} must be byte-identical when a tool exchange is appended"
            );
        }
    }

    /// Tool-array emission order (Step 3): the adapter is a faithful passthrough.
    /// It emits `body["tools"]` in the EXACT incoming order — it does NOT sort.
    /// Sorting is upstream's responsibility (Task 6); the adapter must preserve
    /// whatever order it receives so `tool_defs_hash` flips only on membership
    /// change, never on a re-sort at the boundary.
    #[test]
    fn test_pillar_a_openai_tools_array_preserves_incoming_order() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");
        let messages = vec![json!({"role":"user","content":"do it"})];
        let tool = |name: &str| {
            json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": format!("tool {name}"),
                    "parameters": {"type": "object", "properties": {}}
                }
            })
        };

        // Already name-sorted incoming order → emitted in that exact order.
        let sorted = vec![tool("a_tool"), tool("b_tool"), tool("c_tool")];
        let body =
            provider.build_request_body("gpt-4o-mini", &messages, &sorted, &ChatOptions::default());
        let emitted: Vec<&str> = body["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            emitted,
            ["a_tool", "b_tool", "c_tool"],
            "adapter must emit tools in incoming (already-sorted) order"
        );

        // Scrambled incoming order → emitted in that SAME scrambled order,
        // proving the adapter does NOT sort (passthrough only).
        let scrambled = vec![tool("c_tool"), tool("a_tool"), tool("b_tool")];
        let body_scrambled = provider.build_request_body(
            "gpt-4o-mini",
            &messages,
            &scrambled,
            &ChatOptions::default(),
        );
        let emitted_scrambled: Vec<&str> = body_scrambled["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            emitted_scrambled,
            ["c_tool", "a_tool", "b_tool"],
            "adapter must NOT reorder/sort tools — it is a faithful passthrough"
        );
    }

    // ---- Pillar B (Task 9): cross-turn prefix invariant at the adapter ----

    /// Cross-turn archived stability at the adapter boundary. Pillar B inserts
    /// whole ARCHIVED turns BETWEEN the stable core (index 0) and the per-task
    /// `[Task Context]` tail. When a later turn archives one MORE whole turn, the
    /// payload it builds extends the `core + archived[..N-1]` region without
    /// rewriting any earlier element. The OpenAI adapter must preserve that
    /// element-wise: converting the turn-2 sequence and the turn-3 sequence
    /// yields message arrays whose shared `core + archived` prefix is
    /// byte-identical, element-for-element. This is the adapter-boundary
    /// companion to the full-loop call_log assertion in
    /// `integration_tests::pillar_b_cross_turn_archived_prefix_is_byte_identical`.
    #[test]
    fn test_pillar_b_openai_cross_turn_archived_prefix_is_byte_identical() {
        let provider = OpenAiCompatibleProvider::new("https://api.openai.com/v1", "test-key")
            .expect("provider should initialize");

        let core = json!({"role":"system","content":"core prompt (stable)"});
        // One archived turn = user + assistant rendered pair.
        let archived_turn_1 = [
            json!({"role":"user","content":"archived turn 1 ask"}),
            json!({"role":"assistant","content":"archived turn 1 reply"}),
        ];
        let archived_turn_2 = [
            json!({"role":"user","content":"archived turn 2 ask"}),
            json!({"role":"assistant","content":"archived turn 2 reply"}),
        ];

        // Turn 2 build: core + [archived turn 1] + tail + current user.
        let turn2 = vec![
            core.clone(),
            archived_turn_1[0].clone(),
            archived_turn_1[1].clone(),
            json!({"role":"system","content":"[Task Context] turn 2 tail"}),
            json!({"role":"user","content":"turn 2 current ask"}),
        ];
        // Turn 3 build: core + [archived turn 1, archived turn 2] + tail + user.
        // The archived region EXTENDS turn 2's: turn 1 stays in place, turn 2 is
        // appended after it, then a fresh transient tail + current user.
        let turn3 = vec![
            core.clone(),
            archived_turn_1[0].clone(),
            archived_turn_1[1].clone(),
            archived_turn_2[0].clone(),
            archived_turn_2[1].clone(),
            json!({"role":"system","content":"[Task Context] turn 3 tail"}),
            json!({"role":"user","content":"turn 3 current ask"}),
        ];

        let body2 =
            provider.build_request_body("gpt-4o-mini", &turn2, &[], &ChatOptions::default());
        let body3 =
            provider.build_request_body("gpt-4o-mini", &turn3, &[], &ChatOptions::default());
        let arr2 = body2["messages"].as_array().expect("messages array");
        let arr3 = body3["messages"].as_array().expect("messages array");

        // Shared stable prefix = core (1) + archived turn 1 (2) = 3 elements.
        // Those three must be byte-identical element-for-element across turns;
        // turn 3 appends archived turn 2 + its own tail after them.
        let shared = 3;
        for i in 0..shared {
            assert_eq!(
                serde_json::to_string(&arr2[i]).unwrap(),
                serde_json::to_string(&arr3[i]).unwrap(),
                "core+archived[..N-1] element {i} must be byte-identical across turns"
            );
        }
        // Sanity: turn 3 genuinely carries the additional archived turn inline,
        // in source order, immediately after the shared prefix.
        assert_eq!(arr3[3]["content"], "archived turn 2 ask");
        assert_eq!(arr3[4]["content"], "archived turn 2 reply");
    }

    #[test]
    fn test_cached_input_tokens_from_openai_usage_details() {
        let usage = json!({
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 75
            }
        });

        assert_eq!(
            OpenAiCompatibleProvider::cached_input_tokens_from_usage(&usage),
            Some(75)
        );
    }

    #[test]
    fn test_detects_cloudflare_gateway_host() {
        assert!(is_cloudflare_ai_gateway_base(
            "https://gateway.ai.cloudflare.com/v1/acct/gw/compat"
        ));
        assert!(!is_cloudflare_ai_gateway_base("https://api.openai.com/v1"));
    }

    #[test]
    fn test_cloudflare_models_fallback_url_when_base_has_compat() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token(
            "https://gateway.ai.cloudflare.com/v1/a/g/compat",
            "test-key",
            None,
        )
        .expect("provider should initialize");
        assert_eq!(
            provider.cloudflare_models_fallback_url(),
            "https://gateway.ai.cloudflare.com/v1/a/g/compat/v1/models"
        );
    }

    #[test]
    fn test_cloudflare_models_fallback_url_when_base_has_no_compat() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token(
            "https://gateway.ai.cloudflare.com/v1/a/g",
            "test-key",
            None,
        )
        .expect("provider should initialize");
        assert_eq!(
            provider.cloudflare_models_fallback_url(),
            "https://gateway.ai.cloudflare.com/v1/a/g/compat/v1/models"
        );
    }

    #[test]
    fn test_with_auth_headers_includes_gateway_header_when_set() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token(
            "https://api.openai.com/v1",
            "test-key",
            Some("cf-gateway-token"),
        )
        .expect("provider should initialize");
        let request = provider
            .with_auth_headers(provider.client.get("https://example.com/models"))
            .build()
            .expect("request should build");

        assert_eq!(
            request.headers().get("Authorization").unwrap(),
            "Bearer test-key"
        );
        assert_eq!(
            request.headers().get("cf-aig-authorization").unwrap(),
            "Bearer cf-gateway-token"
        );
    }

    #[test]
    fn test_with_auth_headers_includes_extra_headers() {
        let provider = OpenAiCompatibleProvider::new_with_gateway_token_and_headers(
            "https://api.openai.com/v1",
            "test-key",
            None,
            Some(HashMap::from([(
                "x-team".to_string(),
                "agents".to_string(),
            )])),
        )
        .expect("provider should initialize");
        let request = provider
            .with_auth_headers(provider.client.get("https://example.com/models"))
            .build()
            .expect("request should build");

        assert_eq!(request.headers().get("x-team").unwrap(), "agents");
    }

    #[test]
    fn test_parse_models_response_parses_openai_shape() {
        let models = OpenAiCompatibleProvider::parse_models_response(
            r#"{"data":[{"id":"gpt-4o-mini"},{"id":"gpt-4.1"}]}"#,
        )
        .expect("models should parse");
        assert_eq!(models, vec!["gpt-4o-mini", "gpt-4.1"]);
    }

    #[test]
    fn test_normalize_tool_name_trims_whitespace() {
        assert_eq!(normalize_tool_name(" terminal "), "terminal");
    }
}
