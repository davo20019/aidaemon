pub mod registry;
mod render;

use async_trait::async_trait;
use serde_json::Value;
use tracing::info;

use crate::traits::{Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects};

pub use registry::McpRegistry;

/// A tool proxied from an MCP server: the SDK's tool does the call, the
/// daemon keeps the name, schema and rendering the model sees and the
/// safeguards below.
pub struct McpTool {
    /// The name advertised to the LLM (`{server}__{tool}`, unsanitized).
    tool_name: String,
    tool_schema: Value,
    tool: aidaemon_mcp::McpTool,
}

/// Every daemon MCP server starts with these: its tools are annotated at
/// their worst whatever the server's hints claim (see
/// [`mcp_tool_capabilities`]).
pub(crate) fn server_options() -> aidaemon_mcp::McpOptions {
    aidaemon_mcp::McpOptions {
        annotations: Some(aidaemon_core::ToolAnnotations::DESTRUCTIVE),
        ..Default::default()
    }
}

impl McpTool {
    /// The daemon's view of a tool from `server`.
    pub(crate) fn wrap(server: &str, tool: aidaemon_mcp::McpTool) -> Self {
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

    /// A tool built as the registry builds one, with no process behind it.
    #[cfg(test)]
    pub(crate) fn detached_for_test(server: &str, definition: Value) -> Self {
        Self::wrap(
            server,
            aidaemon_mcp::McpTool::detached(server, &definition, &server_options()),
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
pub(crate) fn mcp_tool_capabilities() -> ToolCapabilities {
    ToolCapabilities {
        read_only: false,
        external_side_effect: true,
        needs_approval: true,
        idempotent: false,
        high_impact_write: true,
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        self.tool_schema["description"].as_str().unwrap_or("")
    }

    fn schema(&self) -> Value {
        self.tool_schema.clone()
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value =
            serde_json::from_str(arguments).unwrap_or(Value::Object(Default::default()));

        // Audit log: truncate args for logging to avoid flooding logs with large payloads
        let args_preview: String = {
            let s = args.to_string();
            crate::utils::truncate_str(&s, 203)
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
            .and_then(|result| render::format_tool_call_result(&result));
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
}

impl crate::traits::ToolSemantics for McpTool {
    fn capabilities(&self) -> ToolCapabilities {
        mcp_tool_capabilities()
    }

    /// The server is an opaque process: a mandate never delegates its tools.
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            opaque_execution: true,
            ..Default::default()
        }
    }

    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
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
        _roots: &crate::traits::TrustRoots<'_>,
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
        let boundary = crate::utils::floor_char_boundary(output, 4096);
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

#[cfg(test)]
mod annotation_tests {
    use super::*;

    fn mcp_tool(annotations: Value) -> McpTool {
        // The definition carries the server's hints, as `tools/list` does.
        McpTool::detached_for_test(
            "acme",
            serde_json::json!({
                "name": "synthetic_tool",
                "description": "",
                "inputSchema": {"type": "object"},
                "annotations": annotations,
            }),
        )
    }

    /// The model sees `{server}__{tool}` with the name the server gave,
    /// not the SDK's sanitized one, and the server's own description.
    #[test]
    fn wrapped_tools_keep_the_daemon_name_and_schema() {
        let tool = McpTool::detached_for_test(
            "acme",
            serde_json::json!({
                "name": "files.read",
                "title": "Reads files",
                "inputSchema": {"$schema": "x", "type": "object"},
            }),
        );
        assert_eq!(tool.name(), "acme__files.read");
        assert_eq!(
            tool.schema(),
            serde_json::json!({
                "name": "acme__files.read",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            })
        );
    }

    /// A mandate never delegates an MCP tool, whatever the server and tool
    /// are called: the tool declares opaque execution.
    #[test]
    fn mcp_tools_are_never_delegated_to_a_mandate() {
        let tool = McpTool::detached_for_test(
            "x",
            serde_json::json!({"name": "post", "inputSchema": {"type": "object"}}),
        );
        assert_eq!(tool.name(), "x__post");
        assert!(crate::mandates::is_non_delegable(&tool));
        assert!(!crate::mandates::is_non_delegable_tool("x__post"));
    }

    fn hook_ctx(untrusted_seen: bool) -> aidaemon_core::HookContext<'static> {
        aidaemon_core::HookContext {
            run_id: aidaemon_core::RunId::new(),
            agent_id: "aidaemon",
            session_id: "telegram:synthetic-user-1",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: aidaemon_core::Usage::default(),
            run_usage: aidaemon_core::Usage::default(),
            untrusted_seen,
            declined: &[],
        }
    }

    /// A server's hints are its own claims: a tool claiming to be read-only
    /// and closed-world is still egress, refused under read-only authority,
    /// checked by the untrusted verifier after untrusted content, opaque in
    /// its effects and untrusted in its output.
    #[tokio::test]
    async fn mcp_hints_never_loosen_safeguards() {
        use crate::traits::ToolSemantics;
        use aidaemon_core::Hook;
        let roots = crate::traits::TrustRoots::new(&[]);
        let claims_read_only =
            mcp_tool(serde_json::json!({"readOnlyHint": true, "openWorldHint": false}));
        let unhinted = mcp_tool(serde_json::json!({}));
        for tool in [&claims_read_only, &unhinted] {
            let annotations = tool.call_annotations_in("{}", &roots);
            assert!(annotations.egress, "{annotations:?}");
            assert!(!annotations.read_only, "{annotations:?}");
            let caps = tool.capabilities();
            assert!(!caps.read_only && caps.needs_approval && caps.external_side_effect);
            assert!(!caps.idempotent && caps.high_impact_write);
            assert_eq!(
                tool.call_effects("{}"),
                aidaemon_core::CallEffects::unknown()
            );
            assert_eq!(
                tool.output_trust("{}", &roots),
                aidaemon_core::SourceTrust::Untrusted
            );

            let info = aidaemon_core::ToolInfo {
                annotations,
                access: aidaemon_core::AccessManifest::default(),
                effects: tool.call_effects("{}"),
            };
            let call = aidaemon_core::ToolCall::new("c1", "synthetic_tool", serde_json::json!({}));
            let read_only =
                aidaemon_core::AuthorityHook::new(aidaemon_core::Authority::read_only());
            assert!(
                matches!(
                    read_only.before_tool(&hook_ctx(false), &call, &info).await,
                    aidaemon_core::ToolDecision::Deny { .. }
                ),
                "read-only authority must refuse it"
            );
            let untrusted = aidaemon_core::UntrustedHook::new(
                aidaemon_core::UntrustedMode::Enforce,
                |_, content| content.to_string(),
            );
            assert!(
                matches!(
                    untrusted.before_tool(&hook_ctx(true), &call, &info).await,
                    aidaemon_core::ToolDecision::Interrupt { .. }
                ),
                "after untrusted content it must go through the verifier"
            );
        }
    }
}
