//! MCP tools exposed as kernel [`Tool`]s.

use std::collections::HashSet;
use std::sync::Arc;

use aidaemon_core::{
    CancellationToken, Image, Tool, ToolAnnotations, ToolContext, ToolError, ToolOutput,
    ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::client::{Inner, McpError, McpOptions};

const MAX_TOOL_NAME: usize = 64;

/// Helpers over a raw `tools/list` entry.
pub(crate) struct McpToolDef;

impl McpToolDef {
    pub(crate) fn is_valid(def: &Value) -> bool {
        def.get("name")
            .and_then(Value::as_str)
            .is_some_and(|n| !n.is_empty())
    }
}

/// `"{server}__{tool}"` limited to `[a-zA-Z0-9_-]` and 64 characters.
pub fn tool_name(server: &str, tool: &str) -> String {
    let mut name: String = format!("{server}__{tool}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    name.truncate(MAX_TOOL_NAME);
    name
}

/// Map MCP tool annotations to kernel ones. Absent hints take the MCP
/// defaults (`readOnlyHint=false`, `destructiveHint=true`,
/// `idempotentHint=false`), so an unannotated tool is mutating and
/// destructive. A read-only tool is never destructive and, unless it says
/// otherwise, safe to re-run.
pub fn map_annotations(annotations: Option<&Value>) -> ToolAnnotations {
    let hint = |key: &str| {
        annotations
            .and_then(|a| a.get(key))
            .and_then(Value::as_bool)
    };
    if hint("readOnlyHint") == Some(true) {
        return ToolAnnotations {
            read_only: true,
            destructive: false,
            idempotent: hint("idempotentHint").unwrap_or(true),
            egress: false,
        };
    }
    let destructive = hint("destructiveHint").unwrap_or(true);
    ToolAnnotations {
        read_only: false,
        destructive,
        idempotent: hint("idempotentHint").unwrap_or(false),
        // A mutating tool reaches outside the workspace unless the server
        // says it is closed-world (`openWorldHint=false`); MCP's default
        // for that hint is open.
        egress: destructive || hint("openWorldHint").unwrap_or(true),
    }
}

/// A `tools/list` `inputSchema` as model-facing function parameters: an
/// object schema with `properties` (MCP allows a bare `{"type": "object"}`)
/// and no `$schema` (Gemini rejects it). A missing or non-object schema
/// becomes an empty object schema.
pub fn normalize_schema(schema: Option<&Value>) -> Value {
    let Some(Value::Object(schema)) = schema else {
        return json!({"type": "object", "properties": {}});
    };
    let mut schema = schema.clone();
    schema.remove("$schema");
    schema.entry("properties").or_insert_with(|| json!({}));
    Value::Object(schema)
}

/// One MCP server tool. Calls go through the shared server connection.
#[derive(Clone)]
pub struct McpTool {
    server: Arc<Inner>,
    remote_name: String,
    definition: Value,
    spec: ToolSpec,
    annotations: ToolAnnotations,
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTool")
            .field("name", &self.spec.name)
            .field("remote_name", &self.remote_name)
            .field("annotations", &self.annotations)
            .finish()
    }
}

impl McpTool {
    fn from_def(server: &Arc<Inner>, def: &Value, name: String) -> Self {
        let remote_name = def["name"].as_str().unwrap_or_default().to_string();
        let description = def
            .get("description")
            .and_then(Value::as_str)
            .or_else(|| def.get("title").and_then(Value::as_str))
            .map(str::to_string)
            .unwrap_or_else(|| format!("`{remote_name}` from MCP server `{}`", server.name));
        Self {
            server: server.clone(),
            annotations: server
                .annotations
                .unwrap_or_else(|| map_annotations(def.get("annotations"))),
            spec: ToolSpec {
                name,
                description,
                parameters: normalize_schema(def.get("inputSchema")),
            },
            definition: def.clone(),
            remote_name,
        }
    }

    /// A tool with no server behind it: named, described and annotated as
    /// `definition` under `options` would be on a live server, but every
    /// call fails with [`McpError::Closed`]. For checking how a definition
    /// maps, and for tests that hold a tool without calling it.
    pub fn detached(server: &str, definition: &Value, options: &McpOptions) -> Self {
        let name = tool_name(
            server,
            definition
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        Self::from_def(
            &Arc::new(Inner::detached(server, options)),
            definition,
            name,
        )
    }

    /// The tool's name on the MCP server (unprefixed, unsanitized).
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    pub fn server_name(&self) -> &str {
        &self.server.name
    }

    /// The raw `tools/list` entry this tool was built from.
    pub fn definition(&self) -> &Value {
        &self.definition
    }

    /// Send `tools/call` and return the server's result unrendered, for
    /// callers with their own rendering ([`Tool::call`] uses
    /// [`render_result`]). `isError` results are `Ok`; only transport and
    /// JSON-RPC failures are `Err`.
    pub async fn call_raw(
        &self,
        arguments: Value,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value, McpError> {
        self.server
            .request_cancellable(
                "tools/call",
                json!({"name": self.remote_name, "arguments": arguments}),
                None,
                cancel,
            )
            .await
    }
}

/// Build tools, keeping kernel names unique even when sanitizing or
/// truncation makes two collide (later ones get a `_2`, `_3`... suffix).
pub(crate) fn tools_from_defs(server: &Arc<Inner>, defs: &[Value]) -> Vec<McpTool> {
    let mut used = HashSet::new();
    let mut tools = Vec::with_capacity(defs.len());
    for def in defs {
        let Some(remote) = def.get("name").and_then(Value::as_str) else {
            continue;
        };
        let base = tool_name(&server.name, remote);
        let mut name = base.clone();
        let mut n = 2;
        while !used.insert(name.clone()) {
            let suffix = format!("_{n}");
            let mut stem = base.clone();
            stem.truncate(MAX_TOOL_NAME - suffix.len());
            name = format!("{stem}{suffix}");
            n += 1;
        }
        tools.push(McpTool::from_def(server, def, name));
    }
    tools
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn annotations(&self) -> ToolAnnotations {
        self.annotations
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let arguments = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };
        match self.call_raw(arguments, Some(&ctx.cancel)).await {
            Ok(result) => Ok(render_result(&self.server.name, &result)),
            // The server answered with a JSON-RPC error (unknown tool, bad
            // params): the model can react to it.
            Err(err @ McpError::Rpc { .. }) => Ok(ToolOutput::error(err.to_string())
                .with_details(json!({"server": self.server.name, "transport": "rpc_error"}))),
            // Crashed, timed out, cancelled: the call's fate is unknown.
            Err(err) => Err(ToolError(err.to_string())),
        }
    }
}

/// Turn a `tools/call` result into model text, images and receipt details.
/// Image blocks keep a one-line summary in the text and are also attached
/// as [`Image`]s, so vision models see them.
pub fn render_result(server: &str, result: &Value) -> ToolOutput {
    let mut parts = Vec::new();
    let mut images = Vec::new();
    let mut non_text = 0usize;
    for block in result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
        if kind == "text" {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                parts.push(text.to_string());
            }
            continue;
        }
        non_text += 1;
        parts.push(summarize_block(kind, block));
        if kind == "image" {
            let data = block.get("data").and_then(Value::as_str);
            let mime = block.get("mimeType").and_then(Value::as_str);
            if let (Some(data), Some(mime)) = (data, mime) {
                images.push(Image::base64(mime, data));
            }
        }
    }

    let structured = result
        .get("structuredContent")
        .filter(|v| !v.is_null())
        .cloned();
    let mut content = parts.join("\n");
    if content.trim().is_empty()
        && let Some(structured) = &structured
    {
        content = structured.to_string();
    }

    let mut details = Map::new();
    details.insert("server".into(), json!(server));
    if non_text > 0 {
        details.insert("non_text_blocks".into(), json!(non_text));
    }
    if let Some(structured) = structured {
        details.insert("structured_content".into(), structured);
    }

    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let output = if is_error {
        ToolOutput::error(content)
    } else {
        ToolOutput::text(content)
    };
    output
        .with_images(images)
        .with_details(Value::Object(details))
}

fn summarize_block(kind: &str, block: &Value) -> String {
    let mime = block
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let approx_bytes = |data: Option<&Value>| {
        data.and_then(Value::as_str)
            .map(|d| d.len().saturating_mul(3) / 4)
            .unwrap_or(0)
    };
    match kind {
        "image" | "audio" => format!(
            "[{kind} content: {mime}, ~{} bytes]",
            approx_bytes(block.get("data"))
        ),
        "resource_link" => format!(
            "[resource link: {}]",
            block
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ),
        "resource" => {
            let resource = block.get("resource");
            let uri = resource
                .and_then(|r| r.get("uri"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            match resource.and_then(|r| r.get("text")).and_then(Value::as_str) {
                Some(text) => format!("[resource: {uri}]\n{text}"),
                None => format!(
                    "[resource: {uri}, binary ~{} bytes]",
                    approx_bytes(resource.and_then(|r| r.get("blob")))
                ),
            }
        }
        "" => "[untyped content block]".to_string(),
        other => format!("[{other} content]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_prefixed_sanitized_and_bounded() {
        assert_eq!(tool_name("fs", "read_file"), "fs__read_file");
        assert_eq!(
            tool_name("my server", "files.read/v2"),
            "my_server__files_read_v2"
        );
        assert_eq!(tool_name("srv", "héllo"), "srv__h_llo");
        let long = tool_name("srv", &"x".repeat(200));
        assert_eq!(long.len(), 64);
        assert!(long.starts_with("srv__"));
    }

    #[test]
    fn annotations_follow_mcp_defaults() {
        assert_eq!(map_annotations(None), ToolAnnotations::DESTRUCTIVE);
        assert_eq!(
            map_annotations(Some(&json!({}))),
            ToolAnnotations::DESTRUCTIVE
        );
        assert_eq!(
            map_annotations(Some(&json!({"readOnlyHint": true}))),
            ToolAnnotations::READ_ONLY
        );
        assert_eq!(
            map_annotations(Some(
                &json!({"readOnlyHint": true, "destructiveHint": true})
            )),
            ToolAnnotations::READ_ONLY,
            "destructiveHint is meaningless for read-only tools"
        );
        assert_eq!(
            map_annotations(Some(
                &json!({"destructiveHint": false, "idempotentHint": true})
            )),
            ToolAnnotations {
                read_only: false,
                destructive: false,
                idempotent: true,
                egress: true
            }
        );
        assert_eq!(
            map_annotations(Some(&json!({"readOnlyHint": false}))),
            ToolAnnotations::DESTRUCTIVE
        );
    }

    #[test]
    fn schemas_are_normalized_for_providers() {
        assert_eq!(
            normalize_schema(Some(&json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["a"],
                "properties": {"a": {"type": "string"}}
            }))),
            json!({
                "type": "object",
                "required": ["a"],
                "properties": {"a": {"type": "string"}}
            })
        );
        assert_eq!(
            normalize_schema(Some(&json!({"type": "object"}))),
            json!({"type": "object", "properties": {}})
        );
        for missing in [None, Some(&Value::Null), Some(&json!("not a schema"))] {
            assert_eq!(
                normalize_schema(missing),
                json!({"type": "object", "properties": {}})
            );
        }
    }

    #[test]
    fn results_render_text_summaries_and_structured_content() {
        let out = render_result(
            "s",
            &json!({
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "image", "mimeType": "image/png", "data": "AAAA"},
                    {"type": "text", "text": "two"},
                    {"type": "resource_link", "uri": "file:///a.txt", "name": "a"}
                ],
                "structuredContent": {"k": 1}
            }),
        );
        assert!(!out.is_error);
        assert_eq!(
            out.content,
            "one\n[image content: image/png, ~3 bytes]\ntwo\n[resource link: file:///a.txt]"
        );
        assert_eq!(out.details["structured_content"], json!({"k": 1}));
        assert_eq!(out.details["non_text_blocks"], json!(2));
        assert_eq!(out.images, vec![Image::base64("image/png", "AAAA")]);

        let only_structured =
            render_result("s", &json!({"content": [], "structuredContent": {"a": 2}}));
        assert_eq!(only_structured.content, r#"{"a":2}"#);

        let err = render_result(
            "s",
            &json!({"content": [{"type": "text", "text": "nope"}], "isError": true}),
        );
        assert!(err.is_error);
        assert_eq!(err.content, "nope");
    }
}
