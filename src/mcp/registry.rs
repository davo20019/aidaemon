use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::{store_in_keychain, McpServerConfig};
use crate::mcp::{McpClient, McpTool};
use crate::traits::{DynamicMcpServer, StateStore, Tool};

/// A running MCP server entry in the registry.
pub struct McpServerEntry {
    pub name: String,
    pub config: McpServerConfig,
    pub client: Arc<McpClient>,
    pub tools: Vec<Arc<dyn Tool>>,
    pub triggers: Vec<String>,
    pub db_id: Option<i64>,
}

/// Summary info about a registered MCP server (for list action).
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used by ManageMcpTool list action
pub struct ServerInfo {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub tool_names: Vec<String>,
    pub env_keys: Vec<String>,
    pub triggers: Vec<String>,
    pub db_id: Option<i64>,
    pub enabled: bool,
}

/// Runtime registry of MCP servers. Clone is cheap (Arc-based).
#[derive(Clone)]
pub struct McpRegistry {
    servers: Arc<RwLock<HashMap<String, McpServerEntry>>>,
    state: Arc<dyn StateStore>,
}

impl McpRegistry {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
            state,
        }
    }

    /// Validate an MCP server name: lowercase alphanumeric + underscores, 1-32 chars.
    fn validate_server_name(name: &str) -> anyhow::Result<()> {
        if name.is_empty() || name.len() > 32 {
            anyhow::bail!("Server name must be 1-32 characters (got {})", name.len());
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            anyhow::bail!(
                "Server name '{}' contains invalid characters. Only lowercase letters, digits, and underscores are allowed.",
                name
            );
        }
        Ok(())
    }

    /// Validate an environment variable key name: alphanumeric + underscores, 1-64 chars.
    fn validate_env_key(key: &str) -> anyhow::Result<()> {
        if key.is_empty() || key.len() > 64 {
            anyhow::bail!("Env key must be 1-64 characters (got {})", key.len());
        }
        if !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            anyhow::bail!(
                "Env key '{}' contains invalid characters. Only letters, digits, and underscores are allowed.",
                key
            );
        }
        Ok(())
    }

    /// Add and spawn an MCP server. Returns the list of registered tool names.
    pub async fn add_server(
        &self,
        name: &str,
        config: McpServerConfig,
        persist: bool,
    ) -> anyhow::Result<Vec<String>> {
        Self::validate_server_name(name)?;

        // Spawn the MCP client
        let client = Arc::new(McpClient::spawn(&config.command, &config.args, &config.env).await?);

        // List tools from the server
        let tool_defs = client.list_tools().await?;

        let mut tool_names = Vec::new();
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();

        for td in &tool_defs {
            let raw_name = td["name"].as_str().unwrap_or("unknown").to_string();
            let prefixed_name = format!("{}__{}", name, raw_name);
            let desc = td["description"].as_str().unwrap_or("").to_string();
            let mut schema = td["inputSchema"].clone();
            // Strip $schema — Google Gemini API rejects it in function parameters
            if let Some(obj) = schema.as_object_mut() {
                obj.remove("$schema");
            }

            let tool_schema = serde_json::json!({
                "name": prefixed_name,
                "description": desc,
                "parameters": schema,
            });

            tool_names.push(prefixed_name.clone());
            tools.push(Arc::new(McpTool::with_prefix(
                raw_name.clone(),
                prefixed_name,
                tool_schema,
                Arc::clone(&client),
                name.to_string(),
                self.clone(),
            )));
        }

        // Keep explicit server aliases only. We no longer infer trigger keywords
        // from tool names to avoid accidental activation by natural-language text.
        let triggers: Vec<String> = vec![name.to_lowercase()];

        info!(
            server = name,
            tools = ?tool_names,
            triggers = ?triggers,
            "Registered MCP server"
        );

        // Persist to DB if requested
        let db_id = if persist {
            let env_keys: Vec<String> = config.env.keys().cloned().collect();
            let server_record = DynamicMcpServer {
                id: 0,
                name: name.to_string(),
                command: config.command.clone(),
                args_json: serde_json::to_string(&config.args)?,
                env_keys_json: serde_json::to_string(&env_keys)?,
                triggers_json: serde_json::to_string(&triggers)?,
                enabled: true,
                created_at: String::new(),
            };
            let id = self.state.save_dynamic_mcp_server(&server_record).await?;
            Some(id)
        } else {
            None
        };

        let entry = McpServerEntry {
            name: name.to_string(),
            config,
            client,
            tools,
            triggers,
            db_id,
        };

        self.servers.write().await.insert(name.to_string(), entry);

        Ok(tool_names)
    }

    /// Remove and shut down an MCP server.
    pub async fn remove_server(&self, name: &str) -> anyhow::Result<()> {
        let entry = self
            .servers
            .write()
            .await
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("MCP server '{}' not found", name))?;

        // Shut down the process
        entry.client.shutdown().await;

        // Delete from DB if persisted
        if let Some(id) = entry.db_id {
            self.state.delete_dynamic_mcp_server(id).await?;
        }

        // Clean up keychain entries for env keys
        let env_keys: Vec<String> = serde_json::from_str(&serde_json::to_string(
            &entry.config.env.keys().collect::<Vec<_>>(),
        )?)
        .unwrap_or_default();
        for key in &env_keys {
            let keychain_key = format!("mcp_{}_{}", name, key);
            if let Err(e) = delete_from_keychain(&keychain_key) {
                warn!(key = %keychain_key, error = %e, "Failed to delete keychain entry");
            }
        }

        info!(server = name, "Removed MCP server");
        Ok(())
    }

    /// Store an env var value in the OS keychain for an MCP server.
    pub async fn set_server_env(&self, name: &str, key: &str, value: &str) -> anyhow::Result<()> {
        Self::validate_server_name(name)?;
        Self::validate_env_key(key)?;
        let keychain_key = format!("mcp_{}_{}", name, key);
        store_in_keychain(&keychain_key, value)?;

        // Update the env_keys in memory and DB.
        // Store only a sentinel — real value stays in keychain only.
        // restart_server() resolves the actual value from keychain before spawning.
        let mut servers = self.servers.write().await;
        if let Some(entry) = servers.get_mut(name) {
            entry
                .config
                .env
                .insert(key.to_string(), "<from_keychain>".to_string());

            // Update DB record with new env keys
            if let Some(id) = entry.db_id {
                let env_keys: Vec<String> = entry.config.env.keys().cloned().collect();
                let server_record = DynamicMcpServer {
                    id,
                    name: name.to_string(),
                    command: entry.config.command.clone(),
                    args_json: serde_json::to_string(&entry.config.args)?,
                    env_keys_json: serde_json::to_string(&env_keys)?,
                    triggers_json: serde_json::to_string(&entry.triggers)?,
                    enabled: true,
                    created_at: String::new(),
                };
                self.state.update_dynamic_mcp_server(&server_record).await?;
            }
        }

        info!(server = name, key, "Stored MCP server env var in keychain");
        Ok(())
    }

    /// Restart an MCP server (re-spawn with fresh env from keychain).
    pub async fn restart_server(&self, name: &str) -> anyhow::Result<Vec<String>> {
        let (config, db_id) = {
            let servers = self.servers.read().await;
            let entry = servers
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("MCP server '{}' not found", name))?;
            (entry.config.clone(), entry.db_id)
        };

        // Shut down old instance
        if let Some(entry) = self.servers.write().await.remove(name) {
            entry.client.shutdown().await;
        }

        // Resolve env from keychain
        let mut resolved_config = config.clone();
        let env_keys: Vec<String> = config.env.keys().cloned().collect();
        for key in &env_keys {
            let keychain_key = format!("mcp_{}_{}", name, key);
            match resolve_from_keychain(&keychain_key) {
                Ok(val) => {
                    resolved_config.env.insert(key.clone(), val);
                }
                Err(e) => {
                    warn!(key = %keychain_key, error = %e, "Failed to resolve env from keychain");
                }
            }
        }

        // Re-add (without persisting again since it's already in DB)
        let tool_names = self.add_server(name, resolved_config, false).await?;

        // Restore the db_id
        if let Some(id) = db_id {
            if let Some(entry) = self.servers.write().await.get_mut(name) {
                entry.db_id = Some(id);
            }
        }

        Ok(tool_names)
    }

    async fn find_dynamic_server_by_name(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<DynamicMcpServer>> {
        let db_servers = self.state.list_dynamic_mcp_servers().await?;
        Ok(db_servers.into_iter().find(|s| s.name == name))
    }

    async fn spawn_dynamic_server_from_record(
        &self,
        server: &DynamicMcpServer,
    ) -> anyhow::Result<Vec<String>> {
        let args: Vec<String> = serde_json::from_str(&server.args_json).unwrap_or_default();
        let env_keys: Vec<String> = serde_json::from_str(&server.env_keys_json).unwrap_or_default();

        // Resolve env values from keychain
        let mut env = HashMap::new();
        for key in &env_keys {
            let keychain_key = format!("mcp_{}_{}", server.name, key);
            match resolve_from_keychain(&keychain_key) {
                Ok(val) => {
                    env.insert(key.clone(), val);
                }
                Err(e) => {
                    warn!(
                        server = %server.name,
                        key = %keychain_key,
                        error = %e,
                        "Failed to resolve env from keychain, server may fail"
                    );
                }
            }
        }

        let config = McpServerConfig {
            command: server.command.clone(),
            args,
            env,
        };

        let tools = self.add_server(&server.name, config, false).await?;

        // Restore db_id/triggers from DB record.
        if let Some(entry) = self.servers.write().await.get_mut(&server.name) {
            entry.db_id = Some(server.id);
            let db_triggers: Vec<String> =
                serde_json::from_str(&server.triggers_json).unwrap_or_default();
            if !db_triggers.is_empty() {
                entry.triggers = db_triggers;
            }
        }

        Ok(tools)
    }

    /// Disable a dynamic MCP server (persisted) and stop it if currently running.
    pub async fn disable_server(&self, name: &str) -> anyhow::Result<()> {
        Self::validate_server_name(name)?;

        let dynamic = self.find_dynamic_server_by_name(name).await?;
        let mut dynamic = match dynamic {
            Some(server) => server,
            None => {
                if self
                    .servers
                    .read()
                    .await
                    .get(name)
                    .is_some_and(|entry| entry.db_id.is_none())
                {
                    anyhow::bail!(
                        "MCP server '{}' is statically configured and cannot be disabled dynamically",
                        name
                    );
                }
                anyhow::bail!("Dynamic MCP server '{}' not found", name);
            }
        };

        if dynamic.enabled {
            dynamic.enabled = false;
            self.state.update_dynamic_mcp_server(&dynamic).await?;
        }

        if let Some(entry) = self.servers.write().await.remove(name) {
            entry.client.shutdown().await;
        }

        info!(server = name, "Dynamic MCP server disabled");
        Ok(())
    }

    /// Enable a dynamic MCP server (persisted) and ensure it is running.
    pub async fn enable_server(&self, name: &str) -> anyhow::Result<Vec<String>> {
        Self::validate_server_name(name)?;

        let mut dynamic = match self.find_dynamic_server_by_name(name).await? {
            Some(server) => server,
            None => anyhow::bail!("Dynamic MCP server '{}' not found", name),
        };

        // Already running and enabled: return current tool list.
        if dynamic.enabled {
            let servers = self.servers.read().await;
            if let Some(entry) = servers.get(name) {
                let tool_names = entry.tools.iter().map(|t| t.name().to_string()).collect();
                return Ok(tool_names);
            }
        }

        // Replace any stale runtime instance before re-spawning.
        if let Some(entry) = self.servers.write().await.remove(name) {
            entry.client.shutdown().await;
        }

        let tools = self.spawn_dynamic_server_from_record(&dynamic).await?;

        if !dynamic.enabled {
            dynamic.enabled = true;
            self.state.update_dynamic_mcp_server(&dynamic).await?;
        }

        info!(server = name, "Dynamic MCP server enabled");
        Ok(tools)
    }

    /// List all registered servers.
    pub async fn list_servers(&self) -> Vec<ServerInfo> {
        let servers = self.servers.read().await;
        servers
            .values()
            .map(|e| ServerInfo {
                name: e.name.clone(),
                command: e.config.command.clone(),
                args: e.config.args.clone(),
                tool_names: e.tools.iter().map(|t| t.name().to_string()).collect(),
                env_keys: e.config.env.keys().cloned().collect(),
                triggers: e.triggers.clone(),
                db_id: e.db_id,
                enabled: true,
            })
            .collect()
    }

    /// List known MCP servers with enabled status, including disabled dynamic servers.
    pub async fn list_servers_with_status(&self) -> anyhow::Result<Vec<ServerInfo>> {
        let mut merged: HashMap<String, ServerInfo> = self
            .list_servers()
            .await
            .into_iter()
            .map(|server| (server.name.clone(), server))
            .collect();

        let dynamic_servers = self.state.list_dynamic_mcp_servers().await?;
        for dynamic in dynamic_servers {
            let args: Vec<String> = serde_json::from_str(&dynamic.args_json).unwrap_or_default();
            let env_keys: Vec<String> =
                serde_json::from_str(&dynamic.env_keys_json).unwrap_or_default();
            let triggers: Vec<String> =
                serde_json::from_str(&dynamic.triggers_json).unwrap_or_default();

            if let Some(existing) = merged.get_mut(&dynamic.name) {
                existing.enabled = dynamic.enabled;
                existing.db_id = Some(dynamic.id);
                if existing.triggers.is_empty() && !triggers.is_empty() {
                    existing.triggers = triggers.clone();
                }
                continue;
            }

            merged.insert(
                dynamic.name.clone(),
                ServerInfo {
                    name: dynamic.name,
                    command: dynamic.command,
                    args,
                    tool_names: Vec::new(),
                    env_keys,
                    triggers,
                    db_id: Some(dynamic.id),
                    enabled: dynamic.enabled,
                },
            );
        }

        let mut list: Vec<ServerInfo> = merged.into_values().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(list)
    }

    /// Load enabled dynamic MCP servers from the database and spawn them.
    pub async fn load_from_db(&self) -> anyhow::Result<()> {
        let db_servers = self.state.list_dynamic_mcp_servers().await?;

        for server in db_servers {
            if !server.enabled {
                continue;
            }

            match self.spawn_dynamic_server_from_record(&server).await {
                Ok(tools) => {
                    info!(
                        server = %server.name,
                        tools = ?tools,
                        "Loaded dynamic MCP server from database"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        server = %server.name,
                        error = %e,
                        "Failed to spawn dynamic MCP server from database"
                    );
                }
            }
        }

        Ok(())
    }

    /// Shut down all MCP server processes.
    pub async fn shutdown_all(&self) {
        let mut servers = self.servers.write().await;
        for (name, entry) in servers.drain() {
            info!(server = %name, "Shutting down MCP server");
            entry.client.shutdown().await;
        }
    }

    /// Return tools from MCP servers only through explicit invocation references.
    ///
    /// Supported forms:
    /// - `mcp:server_name`
    /// - `$mcp:server_name`
    /// - `use mcp server_name`
    /// - `/mcp server_name`
    pub async fn match_tools(&self, user_message: &str) -> Vec<Arc<dyn Tool>> {
        let refs = extract_explicit_mcp_server_refs(user_message);
        if refs.is_empty() {
            return Vec::new();
        }

        let servers = self.servers.read().await;

        let mut matched_tools = Vec::new();
        for entry in servers.values() {
            let server_name = entry.name.to_lowercase();
            let matches = refs.iter().any(|r| r == &server_name)
                || entry
                    .triggers
                    .iter()
                    .map(|t| t.to_lowercase())
                    .any(|alias| refs.iter().any(|r| r == &alias));
            if matches {
                matched_tools.extend(entry.tools.iter().cloned());
            }
        }

        matched_tools
    }

    /// Find a specific tool by name across all MCP servers.
    pub async fn find_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let servers = self.servers.read().await;
        for entry in servers.values() {
            for tool in &entry.tools {
                if tool.name() == name {
                    return Some(tool.clone());
                }
            }
        }
        None
    }

    /// Call a specific raw MCP tool on a named server.
    pub async fn call_server_tool(
        &self,
        server_name: &str,
        raw_tool_name: &str,
        arguments: Value,
    ) -> anyhow::Result<String> {
        let client = {
            let servers = self.servers.read().await;
            let entry = servers
                .get(server_name)
                .ok_or_else(|| anyhow::anyhow!("MCP server '{}' not found", server_name))?;
            entry.client.clone()
        };

        client.call_tool(raw_tool_name, arguments).await
    }

    /// Return all MCP tool definitions (for listing purposes).
    #[allow(dead_code)] // Available for future use (e.g. debug/dashboard)
    pub async fn all_tool_definitions(&self) -> Vec<Value> {
        let servers = self.servers.read().await;
        let mut defs = Vec::new();
        for entry in servers.values() {
            for tool in &entry.tools {
                defs.push(serde_json::json!({
                    "type": "function",
                    "function": tool.schema()
                }));
            }
        }
        defs
    }
}

// ==================== Helper functions ====================

fn normalize_server_ref(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .to_ascii_lowercase()
}

fn extract_explicit_mcp_server_refs(user_message: &str) -> Vec<String> {
    let lower = user_message.to_ascii_lowercase();
    let mut refs: Vec<String> = Vec::new();

    for token in lower.split_whitespace() {
        let candidate = token
            .strip_prefix("$mcp:")
            .or_else(|| token.strip_prefix("mcp:"));
        if let Some(raw) = candidate {
            let norm = normalize_server_ref(raw);
            if !norm.is_empty() && !refs.contains(&norm) {
                refs.push(norm);
            }
        }
    }

    let words: Vec<&str> = lower.split_whitespace().collect();
    for window in words.windows(3) {
        if window[0] == "use" && window[1] == "mcp" {
            let norm = normalize_server_ref(window[2]);
            if !norm.is_empty() && !refs.contains(&norm) {
                refs.push(norm);
            }
        }
    }
    for window in words.windows(2) {
        if window[0] == "/mcp" {
            let norm = normalize_server_ref(window[1]);
            if !norm.is_empty() && !refs.contains(&norm) {
                refs.push(norm);
            }
        }
    }

    refs
}

fn resolve_from_keychain(field_name: &str) -> anyhow::Result<String> {
    crate::config::resolve_from_keychain(field_name)
}

fn delete_from_keychain(field_name: &str) -> anyhow::Result<()> {
    crate::config::delete_from_keychain(field_name)
}

#[cfg(test)]
mod tests {
    use super::extract_explicit_mcp_server_refs;

    #[test]
    fn extracts_prefixed_mcp_refs() {
        let refs = extract_explicit_mcp_server_refs("please use mcp:github and $mcp:slack");
        assert!(refs.contains(&"github".to_string()));
        assert!(refs.contains(&"slack".to_string()));
    }

    #[test]
    fn extracts_use_mcp_form() {
        let refs = extract_explicit_mcp_server_refs("Use MCP browser_tools for this");
        assert_eq!(refs, vec!["browser_tools".to_string()]);
    }

    #[test]
    fn extracts_slash_mcp_form() {
        let refs = extract_explicit_mcp_server_refs("/mcp github");
        assert_eq!(refs, vec!["github".to_string()]);
    }

    #[test]
    fn does_not_guess_from_natural_language() {
        let refs = extract_explicit_mcp_server_refs("can you search github issues?");
        assert!(refs.is_empty());
    }
}
