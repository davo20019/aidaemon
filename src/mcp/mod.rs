mod client;
pub mod registry;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tracing::{info, warn};

use crate::config::McpServerConfig;
use crate::traits::{
    Tool, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolVerificationMode,
};

pub use client::McpClient;
pub use registry::McpRegistry;

/// A tool proxied from an MCP server.
pub struct McpTool {
    /// The name advertised to the LLM (may be prefixed for namespacing).
    tool_name: String,
    /// The raw name used when calling the MCP server (always unprefixed).
    server_tool_name: String,
    server_name: Option<String>,
    tool_schema: Value,
    client: Arc<McpClient>,
    registry: Option<McpRegistry>,
    capabilities: ToolCapabilities,
}

impl McpTool {
    #[allow(dead_code)] // Used by discover_mcp_tools for static configs
    pub fn new(
        tool_name: String,
        tool_schema: Value,
        client: Arc<McpClient>,
        capabilities: ToolCapabilities,
    ) -> Self {
        let server_tool_name = tool_name.clone();
        Self {
            tool_name,
            server_tool_name,
            server_name: None,
            tool_schema,
            client,
            registry: None,
            capabilities,
        }
    }

    /// Create a tool with a prefixed name for the LLM but a raw name for the server.
    pub fn with_prefix(
        server_tool_name: String,
        prefixed_name: String,
        tool_schema: Value,
        client: Arc<McpClient>,
        server_name: String,
        registry: McpRegistry,
        capabilities: ToolCapabilities,
    ) -> Self {
        Self {
            tool_name: prefixed_name,
            server_tool_name,
            server_name: Some(server_name),
            tool_schema,
            client,
            registry: Some(registry),
            capabilities,
        }
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
/// verifier, transport retries) reads them.
pub(crate) fn mcp_tool_capabilities() -> ToolCapabilities {
    ToolCapabilities {
        read_only: false,
        external_side_effect: true,
        needs_approval: true,
        idempotent: false,
        high_impact_write: true,
    }
}

fn is_recoverable_transport_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_ascii_lowercase();
    if msg.contains("mcp tool reported iserror=true") {
        return false;
    }

    msg.contains("closed stdout")
        || msg.contains("broken pipe")
        || msg.contains("connection reset")
        || msg.contains("connection refused")
        || msg.contains("timed out")
        || msg.contains("io error")
}

/// Retrying after a transport failure is safe only when the adapter contract
/// guarantees that the call is both read-only and idempotent. A mutating call
/// may have committed before its response was lost; replaying it would turn one
/// dispatcher grant and one reserved attempt into two external effects.
fn should_retry_transport_failure(capabilities: ToolCapabilities, error: &anyhow::Error) -> bool {
    capabilities.read_only && capabilities.idempotent && is_recoverable_transport_error(error)
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
        let mut result = self
            .client
            .call_tool(&self.server_tool_name, args.clone())
            .await;

        let first_error_message = result.as_ref().err().map(ToString::to_string);
        let should_attempt_recovery = result
            .as_ref()
            .err()
            .is_some_and(|error| should_retry_transport_failure(self.capabilities, error));

        if should_attempt_recovery {
            if let (Some(registry), Some(server_name)) = (&self.registry, &self.server_name) {
                let first_msg = first_error_message
                    .clone()
                    .unwrap_or_else(|| "unknown MCP transport error".to_string());
                if let Err(ref first_err) = result {
                    warn!(
                        mcp_tool = %self.tool_name,
                        server = %server_name,
                        error = %first_err,
                        "MCP transport failure detected, attempting automatic server restart"
                    );
                }

                match registry.restart_server(server_name).await {
                    Ok(_) => {
                        result = registry
                            .call_server_tool(server_name, &self.server_tool_name, args.clone())
                            .await;
                        match &result {
                            Ok(_) => {
                                warn!(
                                    mcp_tool = %self.tool_name,
                                    server = %server_name,
                                    "MCP server restart succeeded; tool call recovered"
                                );
                            }
                            Err(retry_err) => {
                                result = Err(anyhow::anyhow!(
                                    "MCP transport failed and retry after auto-restart also failed. First error: {}. Retry error: {}",
                                    first_msg,
                                    retry_err
                                ));
                            }
                        }
                    }
                    Err(restart_err) => {
                        result = Err(anyhow::anyhow!(
                            "MCP transport failed and automatic restart failed. First error: {}. Restart error: {}",
                            first_msg,
                            restart_err
                        ));
                    }
                }
            }
        }
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
        self.capabilities
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

    /// Never from the server's hints: every call is egress, not read-only,
    /// destructive and non-idempotent, so the authority hook, the untrusted
    /// verifier and the approval policy judge it at its worst.
    fn call_annotations_in(
        &self,
        _arguments: &str,
        _roots: &crate::traits::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        aidaemon_core::ToolAnnotations {
            read_only: false,
            destructive: true,
            idempotent: false,
            egress: true,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        if self.capabilities.read_only {
            return ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent);
        }
        let mut effects = ToolMutationEffects::REMOTE_MUTATION;
        if self.capabilities.high_impact_write {
            effects = effects.union(ToolMutationEffects::DESTRUCTIVE);
        }
        ToolCallSemantics::mutation_with(effects)
    }
}

/// Discover and register all tools from configured MCP servers.
#[allow(dead_code)] // Kept for backward compatibility; static configs now go through McpRegistry
pub async fn discover_mcp_tools(
    mcp_configs: &std::collections::HashMap<String, McpServerConfig>,
) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
    let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

    for (server_name, config) in mcp_configs {
        info!(server_name, command = %config.command, "Connecting to MCP server");

        match McpClient::spawn(&config.command, &config.args, &config.env).await {
            Ok(client) => {
                let client = Arc::new(client);

                match client.list_tools().await {
                    Ok(tool_defs) => {
                        for td in tool_defs {
                            let name = td["name"].as_str().unwrap_or("unknown").to_string();
                            let desc = td["description"].as_str().unwrap_or("").to_string();
                            let mut schema = td["inputSchema"].clone();
                            // Strip $schema — Google Gemini API rejects it
                            if let Some(obj) = schema.as_object_mut() {
                                obj.remove("$schema");
                            }

                            let tool_schema = serde_json::json!({
                                "name": name,
                                "description": desc,
                                "parameters": schema,
                            });

                            info!(server_name, tool_name = %name, "Registered MCP tool");
                            tools.push(Arc::new(McpTool::new(
                                name,
                                tool_schema,
                                Arc::clone(&client),
                                mcp_tool_capabilities(),
                            )));
                        }
                    }
                    Err(e) => {
                        tracing::error!(server_name, "Failed to list MCP tools: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::error!(server_name, "Failed to spawn MCP server: {}", e);
            }
        }
    }

    Ok(tools)
}

#[cfg(test)]
mod annotation_tests {
    use super::*;

    fn mcp_tool(annotations: Value) -> McpTool {
        McpTool::new(
            "synthetic_tool".to_string(),
            // The server's hints travel with the definition; nothing reads them.
            serde_json::json!({
                "name": "synthetic_tool",
                "description": "",
                "parameters": {"type": "object"},
                "annotations": annotations,
            }),
            Arc::new(McpClient::idle_for_test()),
            mcp_tool_capabilities(),
        )
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
mod tests {
    use super::{is_recoverable_transport_error, should_retry_transport_failure};
    use crate::traits::ToolCapabilities;

    fn capabilities(read_only: bool, idempotent: bool) -> ToolCapabilities {
        ToolCapabilities {
            read_only,
            idempotent,
            external_side_effect: true,
            needs_approval: !read_only,
            high_impact_write: !read_only,
        }
    }

    #[test]
    fn transport_error_classifier_ignores_tool_level_is_error() {
        let err = anyhow::anyhow!("MCP tool reported isError=true: Not allowed");
        assert!(!is_recoverable_transport_error(&err));
    }

    #[test]
    fn transport_error_classifier_flags_dead_process_signals() {
        let err = anyhow::anyhow!("MCP server closed stdout (empty response)");
        assert!(is_recoverable_transport_error(&err));
    }

    #[test]
    fn transport_recovery_retries_only_read_only_idempotent_calls() {
        let err = anyhow::anyhow!("MCP server closed stdout (empty response)");

        assert!(should_retry_transport_failure(
            capabilities(true, true),
            &err
        ));
        assert!(!should_retry_transport_failure(
            capabilities(true, false),
            &err
        ));
        assert!(!should_retry_transport_failure(
            capabilities(false, true),
            &err
        ));
        assert!(!should_retry_transport_failure(
            capabilities(false, false),
            &err
        ));
    }
}
