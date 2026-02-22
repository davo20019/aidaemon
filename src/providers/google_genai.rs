use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};
use zeroize::Zeroize;

use crate::llm_markers::CONSULTANT_TEXT_ONLY_MARKER;
use crate::providers::ProviderError;
use crate::traits::{
    ChatOptions, ModelProvider, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
    ToolChoiceMode,
};

/// Recursively strip fields unsupported by Gemini API from a JSON value.
/// Google Gemini rejects `$schema` and `additionalProperties` in function parameter schemas.
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

fn blocked_safety_categories(ratings: Option<&Vec<Value>>) -> Vec<String> {
    let mut categories = Vec::new();
    if let Some(ratings) = ratings {
        for rating in ratings {
            let blocked = rating
                .get("blocked")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !blocked {
                continue;
            }
            if let Some(category) = rating.get("category").and_then(|v| v.as_str()) {
                if !categories.iter().any(|c| c == category) {
                    categories.push(category.to_string());
                }
            }
        }
    }
    categories
}

fn build_gemini_response_note(
    finish_reason: Option<&str>,
    prompt_block_reason: Option<&str>,
    prompt_blocked_categories: &[String],
    candidate_blocked_categories: &[String],
) -> Option<String> {
    let mut parts = Vec::new();

    if let Some(reason) = prompt_block_reason {
        parts.push(format!("prompt blocked ({})", reason));
    }
    if !prompt_blocked_categories.is_empty() {
        parts.push(format!(
            "prompt safety categories: {}",
            prompt_blocked_categories.join(", ")
        ));
    }
    if let Some(reason) = finish_reason {
        let upper = reason.to_ascii_uppercase();
        if upper != "STOP" && upper != "MAX_TOKENS" {
            parts.push(format!("finish reason: {}", reason));
        }
    }
    if !candidate_blocked_categories.is_empty() {
        parts.push(format!(
            "candidate safety categories: {}",
            candidate_blocked_categories.join(", ")
        ));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("; "))
    }
}

fn normalize_tool_name(name: &str) -> String {
    name.trim().to_string()
}

fn part_has_thought_signature(part: &Value) -> bool {
    part.get("thought_signature").is_some() || part.get("thoughtSignature").is_some()
}

fn inject_thought_signature_aliases(part: &mut serde_json::Map<String, Value>) {
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

/// Gemini thinking models can reject historical functionCall parts that are
/// missing thought signatures. Strip those parts (and paired function responses)
/// as a one-shot malformed-request recovery strategy.
fn strip_unsigned_function_call_history(contents: &[Value]) -> (Vec<Value>, usize, usize) {
    let mut sanitized = Vec::with_capacity(contents.len());
    let mut pending_response_drops = 0usize;
    let mut dropped_calls = 0usize;
    let mut dropped_responses = 0usize;

    for msg in contents {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "model" {
            let Some(parts) = msg.get("parts").and_then(|p| p.as_array()) else {
                sanitized.push(msg.clone());
                continue;
            };

            let mut kept_parts = Vec::with_capacity(parts.len());
            let mut dropped_in_msg = 0usize;
            for part in parts {
                if part.get("functionCall").is_none() {
                    kept_parts.push(part.clone());
                    continue;
                }

                if part_has_thought_signature(part) {
                    kept_parts.push(part.clone());
                } else {
                    dropped_calls += 1;
                    dropped_in_msg += 1;
                }
            }

            pending_response_drops = pending_response_drops.saturating_add(dropped_in_msg);
            if !kept_parts.is_empty() {
                let mut kept_msg = msg.clone();
                kept_msg["parts"] = json!(kept_parts);
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

fn is_missing_thought_signature_error(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("missing a thought_signature")
        || lower.contains("missing thought_signature")
        || lower.contains("missing thoughtsignature")
}

pub struct GoogleGenAiProvider {
    client: Client,
    base_url: String,
    api_key: String,
    extra_headers: HashMap<String, String>,
}

impl Drop for GoogleGenAiProvider {
    fn drop(&mut self) {
        self.api_key.zeroize();
    }
}

impl GoogleGenAiProvider {
    pub fn new_with_base_url_and_headers(
        api_key: &str,
        base_url: Option<&str>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> Self {
        let client = crate::providers::build_http_client(Duration::from_secs(120))
            .unwrap_or_else(|e| panic!("failed to build HTTP client: {e}"));
        let normalized_base_url = base_url
            .unwrap_or("https://generativelanguage.googleapis.com/v1beta")
            .trim_end_matches('/')
            .to_string();
        Self {
            client,
            base_url: normalized_base_url,
            api_key: api_key.to_string(),
            extra_headers: extra_headers.unwrap_or_default(),
        }
    }

    fn with_extra_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (k, v) in &self.extra_headers {
            request = request.header(k, v);
        }
        request
    }

    /// Convert OpenAI-format messages to Gemini "contents" + "system_instruction"
    fn convert_messages(&self, messages: &[Value]) -> (Option<Value>, Vec<Value>) {
        let mut system_instruction: Option<Value> = None;
        let mut contents = Vec::new();

        for msg in messages {
            let role = msg["role"].as_str().unwrap_or("user");

            match role {
                "system" => {
                    // Gemini supports top-level system_instruction (for 1.5+ models)
                    // If multiple system messages exist, we concatenate them (or just take the last one, but concatenation is safer)
                    let text = msg["content"].as_str().unwrap_or("");
                    if let Some(existing) = &mut system_instruction {
                        // Append to existing parts
                        if let Some(parts) = existing["parts"].as_array_mut() {
                            parts.push(json!({"text": text}));
                        }
                    } else {
                        system_instruction = Some(json!({
                            "parts": [{ "text": text }]
                        }));
                    }
                }
                "user" => {
                    let text = msg["content"].as_str().unwrap_or("");
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": text }]
                    }));
                }
                "assistant" => {
                    let mut parts = Vec::new();
                    // Text content
                    if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            parts.push(json!({"text": text}));
                        }
                    }
                    // Tool calls
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tc in tool_calls {
                            let function_call = json!({
                                "name": tc["function"]["name"],
                                "args": serde_json::from_str::<Value>(tc["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}))
                            });
                            // Merge extra fields (thought_signature etc.) as siblings of functionCall in the part
                            let mut part_obj = json!({ "functionCall": function_call });
                            if let Some(extra) = tc.get("extra_content").and_then(|e| e.as_object())
                            {
                                if let Some(part_map) = part_obj.as_object_mut() {
                                    for (k, v) in extra {
                                        part_map.insert(k.clone(), v.clone());
                                    }
                                    inject_thought_signature_aliases(part_map);
                                }
                            }
                            parts.push(part_obj);
                        }
                    }
                    contents.push(json!({
                        "role": "model",
                        "parts": parts
                    }));
                }
                "tool" => {
                    // Tool response
                    let _tool_call_id = msg["tool_call_id"].as_str().unwrap_or("");
                    let _tool_name = msg["name"].as_str().unwrap_or(""); // OpenAI puts name here? Or in tool_call?
                                                                         // In our trait Message, we have tool_name. usage in openai_compatible.rs seems to verify this.
                                                                         // But actually OpenAI format for tool response is: { role: tool, tool_call_id: ..., content: ... }
                                                                         // Gemini needs:
                                                                         // { role: "function", parts: [{ "functionResponse": { "name": ..., "response": ... } }] }
                                                                         // Note: Gemini API requires previous message to be "model" with "functionCall".

                    let content_str = msg["content"].as_str().unwrap_or("");
                    // Gemini's functionResponse.response maps to protobuf Struct,
                    // which must be a JSON object — never an array or primitive.
                    let response_json = match serde_json::from_str::<Value>(content_str) {
                        Ok(Value::Object(obj)) => Value::Object(obj),
                        Ok(other) => json!({ "result": other }),
                        Err(_) => json!({ "result": content_str }),
                    };

                    // Hack: We need the tool name. OpenAI format msg has it sometimes?
                    // Our internal Message struct has it. The `messages` passed here are straight JSON from agent.rs.
                    // Checking agent.rs:
                    // if let Some(ref tname) = msg.tool_name { m["name"] = json!(tname); }
                    // So yes, "name" should be present.

                    let name = msg
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("unknown_tool");

                    contents.push(json!({
                        "role": "function", // API expects "function" role or just parts with functionResponse?
                        // Actually documentation says role should be 'function' (v1beta).
                        // wait, checking recent docs... v1beta 'role' can be 'user' or 'model'. Function responses are usually 'function'?
                        // Actually, for generateContent, it's:
                        // role: "function" is NOT standard. It's usually "user" or "model" or implicitly handled?
                        // "role": "function" was used in PaLM. Gemini uses "function_response" part in a "user" (or separate) role?
                        // Docs: "The role of the author... 'user' or 'model'."
                        // Checks: https://ai.google.dev/gemini-api/docs/function-calling
                        // "Send a message... with the function response... role='function'"
                        // So 'function' IS a valid role.
                        "parts": [{
                            "functionResponse": {
                                "name": name,
                                "response": response_json
                            }
                        }]
                    }));
                }
                _ => {}
            }
        }
        (system_instruction, contents)
    }

    /// Convert OpenAI tool definitions to Gemini "tools".
    /// When `include_grounding` is true, appends Google Search grounding.
    /// Expose convert_tools for integration tests.
    #[cfg(test)]
    pub fn convert_tools_for_test(
        &self,
        tools: &[Value],
        include_grounding: bool,
    ) -> Option<Vec<Value>> {
        self.convert_tools(tools, include_grounding)
    }

    fn convert_tools(&self, tools: &[Value], include_grounding: bool) -> Option<Vec<Value>> {
        let mut gemini_tools = Vec::new();

        if !tools.is_empty() {
            let mut function_declarations = Vec::new();
            for tool in tools {
                if let Some(func) = tool.get("function") {
                    let mut params = func["parameters"].clone();
                    strip_unsupported_fields(&mut params);
                    function_declarations.push(json!({
                        "name": func["name"],
                        "description": func.get("description").unwrap_or(&json!("")),
                        "parameters": params
                    }));
                }
            }
            gemini_tools.push(json!({ "function_declarations": function_declarations }));
        }

        if include_grounding {
            gemini_tools.push(json!({ "google_search": {} }));
        }

        if gemini_tools.is_empty() {
            None
        } else {
            Some(gemini_tools)
        }
    }

    /// Detect whether the system instruction explicitly requests text-only consultant mode.
    fn is_consultant_text_only_mode(system_instruction: &Option<Value>) -> bool {
        system_instruction
            .as_ref()
            .and_then(|sys| sys.get("parts"))
            .and_then(|parts| parts.as_array())
            .is_some_and(|parts| {
                parts.iter().any(|part| {
                    part.get("text")
                        .and_then(|t| t.as_str())
                        .is_some_and(|text| text.contains(CONSULTANT_TEXT_ONLY_MARKER))
                })
            })
    }

    fn build_request_body(
        &self,
        system_instruction: Option<Value>,
        contents: Vec<Value>,
        tools: &[Value],
        options: &ChatOptions,
    ) -> (Value, bool, bool, bool) {
        let consultant_text_only_mode = Self::is_consultant_text_only_mode(&system_instruction);
        let has_function_tools = !tools.is_empty();
        let include_grounding = matches!(options.tool_choice, ToolChoiceMode::Auto)
            && !has_function_tools
            && !consultant_text_only_mode;
        let disable_function_calling = !has_function_tools
            || consultant_text_only_mode
            || matches!(options.tool_choice, ToolChoiceMode::None);
        let gemini_tools = self.convert_tools(tools, include_grounding);

        let mut body = json!({
            "contents": contents,
        });

        if let Some(ref sys) = system_instruction {
            body["system_instruction"] = sys.clone();
        }
        if let Some(ref gt) = gemini_tools {
            body["tools"] = json!(gt);
        }
        if disable_function_calling {
            body["tool_config"] = json!({ "function_calling_config": { "mode": "NONE" } });
        } else {
            match &options.tool_choice {
                ToolChoiceMode::Required => {
                    body["tool_config"] = json!({ "function_calling_config": { "mode": "ANY" } });
                }
                ToolChoiceMode::Specific(name) => {
                    body["tool_config"] = json!({
                        "function_calling_config": {
                            "mode": "ANY",
                            "allowed_function_names": [name]
                        }
                    });
                }
                ToolChoiceMode::Auto | ToolChoiceMode::None => {}
            }
        }

        match &options.response_mode {
            ResponseMode::Text => {}
            ResponseMode::JsonObject => {
                body["generation_config"] = json!({
                    "response_mime_type": "application/json"
                });
            }
            ResponseMode::JsonSchema { schema, .. } => {
                let mut stripped = schema.clone();
                strip_unsupported_fields(&mut stripped);
                body["generation_config"] = json!({
                    "response_mime_type": "application/json",
                    "response_schema": stripped
                });
            }
        }

        (
            body,
            consultant_text_only_mode,
            include_grounding,
            disable_function_calling,
        )
    }

    /// Parse a Gemini generateContent response into a ProviderResponse.
    fn parse_response(&self, data: &Value, model: &str) -> anyhow::Result<ProviderResponse> {
        let usage = data.get("usageMetadata").and_then(|u| {
            Some(TokenUsage {
                input_tokens: u.get("promptTokenCount")?.as_u64()? as u32,
                output_tokens: u.get("candidatesTokenCount")?.as_u64()? as u32,
                model: model.to_string(),
            })
        });

        let prompt_feedback = data.get("promptFeedback");
        let prompt_block_reason = prompt_feedback
            .and_then(|pf| pf.get("blockReason"))
            .and_then(|v| v.as_str());
        let prompt_blocked_categories = blocked_safety_categories(
            prompt_feedback
                .and_then(|pf| pf.get("safetyRatings"))
                .and_then(|v| v.as_array()),
        );

        let Some(candidate) = data["candidates"].get(0) else {
            warn!(
                model,
                prompt_block_reason = prompt_block_reason.unwrap_or(""),
                prompt_blocked_categories = ?prompt_blocked_categories,
                "Gemini returned no candidates"
            );
            debug!(
                model,
                response_json = %serde_json::to_string(data).unwrap_or_else(|_| "<unserializable>".to_string()),
                "Gemini raw response JSON (no candidates)"
            );
            let response_note = build_gemini_response_note(
                None,
                prompt_block_reason,
                &prompt_blocked_categories,
                &[],
            )
            .or_else(|| Some("no candidates returned by provider".to_string()));
            return Ok(ProviderResponse {
                content: None,
                tool_calls: vec![],
                usage,
                thinking: None,
                response_note,
            });
        };

        let finish_reason = candidate.get("finishReason").and_then(|v| v.as_str());
        let candidate_blocked_categories = blocked_safety_categories(
            candidate
                .get("safetyRatings")
                .and_then(|ratings| ratings.as_array()),
        );
        let mut response_note = build_gemini_response_note(
            finish_reason,
            prompt_block_reason,
            &prompt_blocked_categories,
            &candidate_blocked_categories,
        );

        let empty_parts = vec![];
        let content_parts = candidate["content"]["parts"]
            .as_array()
            .unwrap_or(&empty_parts);
        let content_parts_len = content_parts.len();

        let mut final_text = String::new();
        let mut thinking_text = String::new();
        let mut tool_calls = Vec::new();

        for part in content_parts {
            if let Some(text) = part.get("text").and_then(|s| s.as_str()) {
                // Gemini thinking models return thought parts (thought: true).
                // Capture them separately — they can be used as fallback when
                // the model produces no regular text content.
                let is_thought = part
                    .get("thought")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_thought {
                    info!(
                        model,
                        thought_len = text.len(),
                        thought_preview = %text.chars().take(300).collect::<String>(),
                        "Gemini thinking output"
                    );
                    thinking_text.push_str(text);
                    continue;
                }
                final_text.push_str(text);
            }
            if let Some(fc) = part.get("functionCall") {
                let name = normalize_tool_name(fc["name"].as_str().unwrap_or(""));
                let args = fc["args"].clone();
                let args_str = serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string());

                let mut extra = serde_json::Map::new();
                if let Some(obj) = part.as_object() {
                    for (k, v) in obj {
                        if k != "functionCall" {
                            extra.insert(k.clone(), v.clone());
                        }
                    }
                }

                let extra_content = if extra.is_empty() {
                    None
                } else {
                    Some(Value::Object(extra))
                };

                tool_calls.push(ToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4()),
                    name,
                    arguments: args_str,
                    extra_content,
                });
            }
        }

        // Extract grounding sources from Google Search
        if let Some(metadata) = candidate.get("groundingMetadata") {
            if let Some(chunks) = metadata.get("groundingChunks").and_then(|c| c.as_array()) {
                let sources: Vec<String> = chunks
                    .iter()
                    .filter_map(|chunk| {
                        let web = chunk.get("web")?;
                        let uri = web.get("uri")?.as_str()?;
                        let title = web.get("title").and_then(|t| t.as_str()).unwrap_or(uri);
                        Some(format!("- [{}]({})", title, uri))
                    })
                    .collect();
                if !sources.is_empty() {
                    final_text.push_str("\n\nSources:\n");
                    final_text.push_str(&sources.join("\n"));
                }
            }
        }

        // Diagnostics: Some Gemini models occasionally return a candidate with
        // empty content parts and no tool calls. Surface this so the agent can
        // produce a meaningful fallback and logs capture the cause.
        let is_empty_response = final_text.trim().is_empty()
            && thinking_text.trim().is_empty()
            && tool_calls.is_empty();
        if is_empty_response {
            let extra = format!(
                "empty response (finishReason={}, parts={})",
                finish_reason.unwrap_or("unknown"),
                content_parts_len
            );
            response_note = Some(match response_note {
                Some(existing) if !existing.trim().is_empty() => format!("{}; {}", existing, extra),
                _ => extra,
            });
            warn!(
                model,
                finish_reason = finish_reason.unwrap_or(""),
                prompt_block_reason = prompt_block_reason.unwrap_or(""),
                prompt_blocked_categories = ?prompt_blocked_categories,
                candidate_blocked_categories = ?candidate_blocked_categories,
                parts = content_parts_len,
                "Gemini returned empty response (no text/thought/tool calls)"
            );
            debug!(
                model,
                response_json = %serde_json::to_string(data).unwrap_or_else(|_| "<unserializable>".to_string()),
                "Gemini raw response JSON (empty response)"
            );
        }

        Ok(ProviderResponse {
            content: if final_text.is_empty() {
                None
            } else {
                Some(final_text)
            },
            tool_calls,
            usage,
            thinking: if thinking_text.is_empty() {
                None
            } else {
                Some(thinking_text)
            },
            response_note,
        })
    }
}

#[async_trait]
impl ModelProvider for GoogleGenAiProvider {
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
        let (system_instruction, contents) = self.convert_messages(messages);
        let original_contents = contents.clone();
        let has_function_tools = !tools.is_empty();
        let (body, consultant_text_only_mode, include_grounding, disable_function_calling) =
            self.build_request_body(system_instruction, contents, tools, options);

        // Use header-based authentication instead of URL query parameter
        // to avoid API key exposure in logs, proxies, and error messages
        let url = format!("{}/models/{}:generateContent", self.base_url, model);

        info!(
            model,
            url_prefix = %self.base_url,
            consultant_text_only_mode,
            has_function_tools,
            include_grounding,
            disable_function_calling,
            response_mode = ?options.response_mode,
            tool_choice = ?options.tool_choice,
            "Calling Google GenAI"
        );

        if has_function_tools
            && matches!(
                options.tool_choice,
                ToolChoiceMode::Required | ToolChoiceMode::Specific(_)
            )
            && disable_function_calling
        {
            warn!(
                tool_choice = ?options.tool_choice,
                "Requested required/specific tool_choice but function calling is disabled by mode constraints"
            );
        }

        let request = self
            .with_extra_headers(
                self.client
                    .post(&url)
                    .header("x-goog-api-key", &self.api_key),
            )
            .json(&body);
        let resp = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                error!("Google GenAI HTTP request failed: {}", e);
                return Err(ProviderError::network(&e).into());
            }
        };

        let status = resp.status();
        let text = resp.text().await.map_err(|e| {
            error!("Failed to read response body: {}", e);
            ProviderError::network(&e)
        })?;

        if !status.is_success() {
            if status.as_u16() == 400 && is_missing_thought_signature_error(&text) {
                let (sanitized_contents, dropped_calls, dropped_responses) =
                    strip_unsigned_function_call_history(&original_contents);
                if dropped_calls > 0 {
                    let mut retry_body = body.clone();
                    retry_body["contents"] = json!(sanitized_contents);
                    warn!(
                        model,
                        dropped_calls,
                        dropped_responses,
                        "Google GenAI rejected missing thought signatures; retrying with unsigned function-call history stripped"
                    );
                    let retry_request = self
                        .with_extra_headers(
                            self.client
                                .post(&url)
                                .header("x-goog-api-key", &self.api_key),
                        )
                        .json(&retry_body);
                    let retry_resp = match retry_request.send().await {
                        Ok(r) => r,
                        Err(e) => {
                            error!("Google GenAI retry HTTP request failed: {}", e);
                            return Err(ProviderError::network(&e).into());
                        }
                    };

                    let retry_status = retry_resp.status();
                    let retry_text = retry_resp.text().await.map_err(|e| {
                        error!("Failed to read retry response body: {}", e);
                        ProviderError::network(&e)
                    })?;
                    if retry_status.is_success() {
                        let data: Value = serde_json::from_str(&retry_text).map_err(|e| {
                            error!("Failed to parse Google GenAI retry response JSON: {}", e);
                            ProviderError::malformed_parse(format!(
                                "Malformed response from LLM provider (JSON parse error: {})",
                                e
                            ))
                        })?;
                        return self.parse_response(&data, model);
                    }
                    error!(
                        status = %retry_status,
                        "Google GenAI API error after thought-signature recovery retry: {}",
                        retry_text
                    );
                    return Err(
                        ProviderError::from_status(retry_status.as_u16(), &retry_text).into(),
                    );
                }
            }
            error!(status = %status, "Google GenAI API error: {}", text);
            return Err(ProviderError::from_status(status.as_u16(), &text).into());
        }

        let data: Value = serde_json::from_str(&text).map_err(|e| {
            error!("Failed to parse Google GenAI response JSON: {}", e);
            ProviderError::malformed_parse(format!(
                "Malformed response from LLM provider (JSON parse error: {})",
                e
            ))
        })?;
        self.parse_response(&data, model)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        // Use header-based authentication instead of URL query parameter
        let url = format!("{}/models?page_size=50", self.base_url);
        let resp = self
            .with_extra_headers(
                self.client
                    .get(&url)
                    .header("x-goog-api-key", &self.api_key),
            )
            .send()
            .await?;

        if !resp.status().is_success() {
            anyhow::bail!("Failed to list models: {}", resp.status());
        }

        let data: Value = resp.json().await?;
        let models = data["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        m["name"]
                            .as_str()
                            .map(|s| s.trim_start_matches("models/").to_string())
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

#[cfg(test)]
impl GoogleGenAiProvider {
    pub fn new(api_key: &str) -> Self {
        Self::new_with_base_url_and_headers(api_key, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn provider() -> GoogleGenAiProvider {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .no_proxy()
            .build()
            .expect("failed to build HTTP client");
        GoogleGenAiProvider {
            client,
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            api_key: "fake-key".to_string(),
            extra_headers: HashMap::new(),
        }
    }

    /// Helper: build an OpenAI-format tool definition from a parameters object.
    fn openai_tool(name: &str, params: Value) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": "test tool",
                "parameters": params
            }
        })
    }

    /// Recursively check that no object in `value` contains a `$schema` key.
    fn assert_no_schema_field(value: &Value, path: &str) {
        match value {
            Value::Object(map) => {
                assert!(
                    !map.contains_key("$schema"),
                    "Found '$schema' at path: {}",
                    path
                );
                for (k, v) in map {
                    assert_no_schema_field(v, &format!("{}.{}", path, k));
                }
            }
            Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    assert_no_schema_field(v, &format!("{}[{}]", path, i));
                }
            }
            _ => {}
        }
    }

    #[test]
    fn test_strips_top_level_schema() {
        let p = provider();
        let tools = vec![openai_tool(
            "mytool",
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                }
            }),
        )];

        let result = p.convert_tools_for_test(&tools, false).unwrap();
        assert_no_schema_field(&json!(result), "tools");
    }

    #[test]
    fn test_strips_nested_schema() {
        let p = provider();
        let tools = vec![openai_tool(
            "mytool",
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "$schema": "http://json-schema.org/draft-07/schema#",
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            }),
        )];

        let result = p.convert_tools_for_test(&tools, false).unwrap();
        assert_no_schema_field(&json!(result), "tools");
    }

    #[test]
    fn test_preserves_valid_fields() {
        let p = provider();
        let tools = vec![openai_tool(
            "mytool",
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The name" },
                    "count": { "type": "integer" }
                },
                "required": ["name"]
            }),
        )];

        let result = p.convert_tools_for_test(&tools, false).unwrap();
        let params = &result[0]["function_declarations"][0]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["name"]["type"], "string");
        assert_eq!(params["properties"]["count"]["type"], "integer");
        assert_eq!(params["required"][0], "name");
    }

    #[test]
    fn test_multiple_tools_all_stripped() {
        let p = provider();
        let tools = vec![
            openai_tool(
                "clean_tool",
                json!({
                    "type": "object",
                    "properties": { "a": { "type": "string" } }
                }),
            ),
            openai_tool(
                "dirty_tool",
                json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": { "b": { "type": "string" } }
                }),
            ),
        ];

        let result = p.convert_tools_for_test(&tools, false).unwrap();
        assert_no_schema_field(&json!(result), "tools");

        // Both tools should still be present
        let decls = result[0]["function_declarations"].as_array().unwrap();
        assert_eq!(decls.len(), 2);
        assert_eq!(decls[0]["name"], "clean_tool");
        assert_eq!(decls[1]["name"], "dirty_tool");
    }

    /// Simulates an MCP tool schema as returned by a real MCP server.
    #[test]
    fn test_mcp_style_schema_stripped() {
        let p = provider();
        let tools = vec![openai_tool(
            "mcp__server__read_file",
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read"
                    }
                },
                "required": ["path"]
            }),
        )];

        let result = p.convert_tools_for_test(&tools, false).unwrap();
        assert_no_schema_field(&json!(result), "tools");
        assert_no_additional_properties(&json!(result), "tools");
    }

    #[test]
    fn test_detects_consultant_text_only_marker() {
        let sys = Some(json!({
            "parts": [
                { "text": "prefix" },
                { "text": "[CONSULTANT_TEXT_ONLY_MODE]\ntext only" }
            ]
        }));
        assert!(GoogleGenAiProvider::is_consultant_text_only_mode(&sys));
    }

    #[test]
    fn test_request_body_consultant_mode_disables_grounding_and_function_calls() {
        let p = provider();
        let system_instruction = Some(json!({
            "parts": [{ "text": "[CONSULTANT_TEXT_ONLY_MODE]\ntext only" }]
        }));
        let contents = vec![json!({
            "role": "user",
            "parts": [{ "text": "what is rust?" }]
        })];

        let (body, consultant_mode, include_grounding, disable_fn_calling) =
            p.build_request_body(system_instruction, contents, &[], &ChatOptions::default());

        assert!(consultant_mode);
        assert!(!include_grounding);
        assert!(disable_fn_calling);
        assert!(body.get("tools").is_none());
        assert_eq!(
            body["tool_config"]["function_calling_config"]["mode"],
            "NONE"
        );
    }

    #[test]
    fn test_request_body_with_no_tools_disables_function_calls_and_keeps_grounding() {
        let p = provider();
        let system_instruction = Some(json!({
            "parts": [{ "text": "normal system prompt" }]
        }));
        let contents = vec![json!({
            "role": "user",
            "parts": [{ "text": "latest news" }]
        })];

        let (body, consultant_mode, include_grounding, disable_fn_calling) =
            p.build_request_body(system_instruction, contents, &[], &ChatOptions::default());

        assert!(!consultant_mode);
        assert!(include_grounding);
        assert!(disable_fn_calling);
        assert_eq!(
            body["tool_config"]["function_calling_config"]["mode"],
            "NONE"
        );
        let tools = body["tools"].as_array().expect("tools should be present");
        assert!(
            tools.iter().any(|t| t.get("google_search").is_some()),
            "expected google_search grounding when not in consultant mode"
        );
    }

    #[test]
    fn test_request_body_required_tool_choice_sets_any_mode() {
        let p = provider();
        let system_instruction = Some(json!({
            "parts": [{ "text": "normal system prompt" }]
        }));
        let contents = vec![json!({
            "role": "user",
            "parts": [{ "text": "run a tool" }]
        })];
        let tools = vec![openai_tool(
            "search_files",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } }
            }),
        )];
        let options = ChatOptions {
            response_mode: ResponseMode::Text,
            tool_choice: ToolChoiceMode::Required,
        };

        let (body, consultant_mode, include_grounding, disable_fn_calling) =
            p.build_request_body(system_instruction, contents, &tools, &options);

        assert!(!consultant_mode);
        assert!(!include_grounding);
        assert!(!disable_fn_calling);
        assert_eq!(
            body["tool_config"]["function_calling_config"]["mode"],
            "ANY"
        );
    }

    #[test]
    fn test_request_body_json_schema_sets_generation_config_and_strips_unsupported_fields() {
        let p = provider();
        let system_instruction = Some(json!({
            "parts": [{ "text": "normal system prompt" }]
        }));
        let contents = vec![json!({
            "role": "user",
            "parts": [{ "text": "return intent gate json" }]
        })];
        let options = ChatOptions {
            response_mode: ResponseMode::JsonSchema {
                name: "intent_gate_v1".to_string(),
                schema: json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "can_answer_now": { "type": "boolean" }
                    },
                    "required": ["can_answer_now"]
                }),
                strict: true,
            },
            tool_choice: ToolChoiceMode::Auto,
        };

        let (body, _, _, _) = p.build_request_body(system_instruction, contents, &[], &options);

        assert_eq!(
            body["generation_config"]["response_mime_type"],
            "application/json"
        );
        assert_no_schema_field(
            &body["generation_config"]["response_schema"],
            "generation_config.response_schema",
        );
        assert_no_additional_properties(
            &body["generation_config"]["response_schema"],
            "generation_config.response_schema",
        );
    }

    /// Verify that thought_signature from Gemini thinking models survives the
    /// parse → convert round-trip (response parsing → message conversion).
    #[test]
    fn test_thought_signature_round_trip() {
        let p = provider();

        // 1. Simulate a Gemini response with thought_signature on a functionCall part
        let gemini_response = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "terminal",
                            "args": { "command": "ls projects" }
                        },
                        "thought_signature": "abc123-sig"
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        });

        // 2. Parse the response — extra_content should capture thought_signature
        let parsed = p
            .parse_response(&gemini_response, "gemini-2.5-flash")
            .unwrap();
        assert_eq!(parsed.tool_calls.len(), 1);
        let tc = &parsed.tool_calls[0];
        assert_eq!(tc.name, "terminal");
        let extra = tc
            .extra_content
            .as_ref()
            .expect("extra_content should be present");
        assert_eq!(extra["thought_signature"], "abc123-sig");

        // 3. Build an assistant message in OpenAI wire format (as agent.rs does)
        let mut val = json!({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments,
            }
        });
        if let Some(ref extra) = tc.extra_content {
            val["extra_content"] = extra.clone();
        }

        // 4. Build a Gemini-format message array with the assistant + tool result
        let messages = vec![
            json!({
                "role": "assistant",
                "tool_calls": [val]
            }),
            json!({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": "terminal",
                "content": "{\"result\": \"site-cars\"}"
            }),
        ];

        // 5. Convert back to Gemini format — thought_signature must be preserved
        let (_sys, contents) = p.convert_messages(&messages);
        assert_eq!(contents.len(), 2);

        // The model message should have the functionCall part with thought_signature
        let model_msg = &contents[0];
        assert_eq!(model_msg["role"], "model");
        let parts = model_msg["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert!(parts[0].get("functionCall").is_some());
        assert_eq!(
            parts[0]["thought_signature"], "abc123-sig",
            "thought_signature must survive the round-trip"
        );
        assert_eq!(
            parts[0]["thoughtSignature"], "abc123-sig",
            "camelCase thoughtSignature alias must also be present"
        );
    }

    #[test]
    fn test_convert_messages_normalizes_thought_signature_aliases() {
        let p = provider();
        let messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": "{\"path\":\"/tmp/a.txt\",\"content\":\"hello\"}"
                },
                "extra_content": {
                    "thoughtSignature": "sig-camel-only"
                }
            }]
        })];

        let (_sys, contents) = p.convert_messages(&messages);
        let parts = contents[0]["parts"].as_array().expect("parts");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["thoughtSignature"], "sig-camel-only");
        assert_eq!(parts[0]["thought_signature"], "sig-camel-only");
    }

    #[test]
    fn test_strip_unsigned_function_call_history_drops_orphaned_function_response() {
        let contents = vec![
            json!({
                "role": "user",
                "parts": [{"text": "do it"}]
            }),
            json!({
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "write_file",
                        "args": {"path": "/tmp/out.txt", "content": "x"}
                    }
                }]
            }),
            json!({
                "role": "function",
                "parts": [{
                    "functionResponse": {
                        "name": "write_file",
                        "response": {"result": "ok"}
                    }
                }]
            }),
            json!({
                "role": "model",
                "parts": [{"text": "Done"}]
            }),
        ];

        let (sanitized, dropped_calls, dropped_responses) =
            strip_unsigned_function_call_history(&contents);
        assert_eq!(dropped_calls, 1);
        assert_eq!(dropped_responses, 1);
        assert_eq!(sanitized.len(), 2);
        assert_eq!(sanitized[0]["role"], "user");
        assert_eq!(sanitized[1]["role"], "model");
        assert_eq!(sanitized[1]["parts"][0]["text"], "Done");
    }

    #[test]
    fn test_parse_response_surfaces_finish_reason_and_safety_block() {
        let p = provider();
        let gemini_response = json!({
            "candidates": [{
                "finishReason": "SAFETY",
                "safetyRatings": [
                    { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "blocked": true }
                ],
                "content": { "role": "model", "parts": [] }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 0
            }
        });

        let parsed = p
            .parse_response(&gemini_response, "gemini-2.5-flash")
            .unwrap();

        assert!(parsed.content.is_none());
        assert!(parsed.tool_calls.is_empty());
        let note = parsed
            .response_note
            .as_deref()
            .expect("expected response note");
        assert!(note.contains("finish reason: SAFETY"));
        assert!(note.contains("HARM_CATEGORY_DANGEROUS_CONTENT"));
    }

    #[test]
    fn test_parse_response_empty_parts_adds_response_note() {
        let p = provider();
        let gemini_response = json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": { "role": "model", "parts": [] }
            }],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 0
            }
        });

        let parsed = p
            .parse_response(&gemini_response, "gemini-2.5-flash-lite")
            .unwrap();

        assert!(parsed.content.is_none());
        assert!(parsed.tool_calls.is_empty());
        assert!(parsed.thinking.is_none());
        let note = parsed
            .response_note
            .as_deref()
            .expect("expected response note");
        assert!(note.contains("empty response"));
        assert!(note.contains("finishReason=STOP"));
        assert!(note.contains("parts=0"));
    }

    #[test]
    fn test_parse_response_with_prompt_feedback_only_returns_note() {
        let p = provider();
        let gemini_response = json!({
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [
                    { "category": "HARM_CATEGORY_HATE_SPEECH", "blocked": true }
                ]
            },
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 0
            }
        });

        let parsed = p
            .parse_response(&gemini_response, "gemini-2.5-flash")
            .unwrap();

        assert!(parsed.content.is_none());
        assert!(parsed.tool_calls.is_empty());
        let note = parsed
            .response_note
            .as_deref()
            .expect("expected response note");
        assert!(note.contains("prompt blocked (SAFETY)"));
        assert!(note.contains("HARM_CATEGORY_HATE_SPEECH"));
    }

    #[test]
    fn test_normalize_tool_name_trims_whitespace() {
        assert_eq!(normalize_tool_name(" terminal "), "terminal");
    }

    /// Recursively check that no object in `value` contains an `additionalProperties` key.
    fn assert_no_additional_properties(value: &Value, path: &str) {
        match value {
            Value::Object(map) => {
                assert!(
                    !map.contains_key("additionalProperties"),
                    "Found 'additionalProperties' at path: {}",
                    path
                );
                for (k, v) in map {
                    assert_no_additional_properties(v, &format!("{}.{}", path, k));
                }
            }
            Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    assert_no_additional_properties(v, &format!("{}[{}]", path, i));
                }
            }
            _ => {}
        }
    }
}
