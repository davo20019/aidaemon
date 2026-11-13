//! An MCP server's tool as the registry hands it to an agent: an
//! [`OutcomeTool`] judged at its worst whatever the server claims.

use async_trait::async_trait;
use serde_json::Value;
use tracing::info;

use aidaemon_tools::OutcomeTool;
use aidaemon_tools::semantics::{
    ToolCallContext, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolDispatchFacts,
    ToolMutationEffects, ToolSemantics, TrustRoots,
};

/// A tool proxied from an MCP server as the registry offers it: the
/// transport's [`crate::McpTool`] does the call; this keeps the name, schema
/// and rendering the model sees and the safeguards below.
pub struct RegisteredMcpTool {
    /// The name advertised to the LLM (`{server}__{tool}`, unsanitized).
    tool_name: String,
    tool_schema: Value,
    tool: crate::McpTool,
}

/// Every daemon MCP server starts with these: its tools are annotated at
/// their worst whatever the server's hints claim (see
/// [`mcp_tool_capabilities`]).
pub fn server_options() -> crate::McpOptions {
    crate::McpOptions {
        annotations: Some(aidaemon_core::ToolAnnotations::DESTRUCTIVE),
        ..Default::default()
    }
}

impl RegisteredMcpTool {
    /// The registry's view of a tool from `server`.
    pub fn wrap(server: &str, tool: crate::McpTool) -> Self {
        let tool_name = format!("{server}__{}", tool.remote_name());
        let tool_schema = serde_json::json!({
            "name": tool_name,
            "description": tool.definition()["description"].as_str().unwrap_or(""),
            "parameters": aidaemon_core::Tool::spec(&tool).parameters,
        });
        Self {
            tool_name,
            tool_schema,
            tool,
        }
    }

    /// A tool built as the registry builds one, with no process behind it
    /// (for tests).
    pub fn detached(server: &str, definition: Value) -> Self {
        Self::wrap(
            server,
            crate::McpTool::detached(server, &definition, &server_options()),
        )
    }
}

/// The capabilities of an MCP server's tool: always the conservative ones
/// (mutating, open-world, destructive, non-idempotent, needs approval).
///
/// MCP `ToolAnnotations` (`readOnlyHint`, `openWorldHint`,
/// `destructiveHint`, `idempotentHint`) are hints the server declares about
/// itself; the MCP specification says to treat them as untrusted. They never
/// relax a safeguard here: nothing that gates a decision (the authority
/// hook's read-only level, the profile's tool pruning, the untrusted-content
/// verifier) reads them.
pub fn mcp_tool_capabilities() -> ToolCapabilities {
    ToolCapabilities {
        read_only: false,
        external_side_effect: true,
        needs_approval: true,
        idempotent: false,
        high_impact_write: true,
    }
}

#[async_trait]
impl OutcomeTool for RegisteredMcpTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        self.tool_schema["description"].as_str().unwrap_or("")
    }

    fn schema(&self) -> Value {
        self.tool_schema.clone()
    }

    async fn execute(
        &self,
        arguments: &str,
        _facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output: anyhow::Result<String> = async move {
            let args: Value =
                serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()));

            // Audit log: truncate args for logging to avoid flooding logs with large payloads
            let args_preview: String = {
                let s = args.to_string();
                aidaemon_tools::text::truncate_str(&s, 203)
            };

            // Threat detection: flag suspicious patterns in tool arguments
            detect_suspicious_args(&self.tool_name, &args_preview);

            info!(mcp_tool = %self.tool_name, args = %args_preview, "MCP tool call started");

            let start = std::time::Instant::now();
            let result = self
                .tool
                .call_raw(args, None)
                .await
                .map_err(anyhow::Error::from)
                .and_then(|result| super::render::format_tool_call_result(&result));
            let elapsed = start.elapsed();

            match &result {
                Ok(output) => {
                    // Threat detection: flag suspicious patterns in tool output
                    detect_suspicious_output(&self.tool_name, output);

                    info!(
                        mcp_tool = %self.tool_name,
                        duration_ms = elapsed.as_millis() as u64,
                        output_bytes = output.len(),
                        "MCP tool call completed"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        mcp_tool = %self.tool_name,
                        duration_ms = elapsed.as_millis() as u64,
                        error = %e,
                        "MCP tool call failed"
                    );
                }
            }

            result
        }
        .await;
        output.map(ToolCallOutcome::from_output)
    }
}

impl ToolSemantics for RegisteredMcpTool {
    fn capabilities(&self) -> ToolCapabilities {
        mcp_tool_capabilities()
    }

    /// The server is an opaque process: a mandate never delegates its tools.
    fn dispatch_facts(&self) -> ToolDispatchFacts {
        ToolDispatchFacts {
            opaque_execution: true,
            ..Default::default()
        }
    }

    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Untrusted
    }

    /// An MCP server is opaque: whatever a call does, its effects are not
    /// declared, so a decline in the run always asks again.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::unknown()
    }

    /// Never from the server's hints ([`server_options`]): every call is
    /// egress, not read-only, destructive and non-idempotent, so the
    /// authority hook, the untrusted verifier and the approval policy judge
    /// it at its worst.
    fn call_annotations_in(
        &self,
        _arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        aidaemon_core::Tool::annotations(&self.tool)
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::DESTRUCTIVE),
        )
    }
}

/// Patterns in MCP tool arguments that may indicate an attack or misuse.
/// Each entry is (pattern, description) — matched case-insensitively.
const SUSPICIOUS_ARG_PATTERNS: &[(&str, &str)] = &[
    ("etc/passwd", "system password file access"),
    ("etc/shadow", "shadow password file access"),
    (".ssh/", "SSH key directory access"),
    (".env", "environment file access"),
    ("config.toml", "aidaemon config file access"),
    ("aidaemon.db", "aidaemon database access"),
    ("api_key", "potential API key extraction"),
    ("bot_token", "potential bot token extraction"),
    ("encryption_key", "potential encryption key extraction"),
    ("curl ", "network request via shell"),
    ("wget ", "network request via shell"),
    ("nc ", "netcat usage"),
    ("base64", "data encoding/exfiltration"),
    ("eval(", "code evaluation"),
    ("exec(", "code execution"),
    ("| sh", "shell pipe execution"),
    ("| bash", "shell pipe execution"),
    ("; rm ", "destructive command chaining"),
    ("chmod 777", "overly permissive file permissions"),
];

/// Patterns in MCP tool output that may indicate data exfiltration or compromise.
const SUSPICIOUS_OUTPUT_PATTERNS: &[(&str, &str)] = &[
    ("sk-", "potential OpenAI API key in output"),
    ("ghp_", "potential GitHub token in output"),
    ("-----BEGIN", "potential private key in output"),
    ("PRIVATE KEY", "potential private key in output"),
    ("password", "potential password in output"),
    ("bot_token", "potential bot token reference in output"),
];

/// Check MCP tool arguments for suspicious patterns and log warnings.
fn detect_suspicious_args(tool_name: &str, args: &str) {
    let lower = args.to_lowercase();
    for (pattern, description) in SUSPICIOUS_ARG_PATTERNS {
        if lower.contains(&pattern.to_lowercase()) {
            tracing::warn!(
                mcp_tool = %tool_name,
                threat = %description,
                "SECURITY: suspicious pattern detected in MCP tool arguments"
            );
        }
    }
}

/// Check MCP tool output for suspicious patterns and log warnings.
fn detect_suspicious_output(tool_name: &str, output: &str) {
    // Only check first ~4096 bytes to avoid scanning huge outputs
    let check = if output.len() > 4096 {
        let boundary = aidaemon_tools::text::floor_char_boundary(output, 4096);
        &output[..boundary]
    } else {
        output
    };
    let lower = check.to_lowercase();
    for (pattern, description) in SUSPICIOUS_OUTPUT_PATTERNS {
        if lower.contains(&pattern.to_lowercase()) {
            tracing::warn!(
                mcp_tool = %tool_name,
                threat = %description,
                "SECURITY: suspicious pattern detected in MCP tool output"
            );
        }
    }
}
