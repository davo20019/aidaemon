use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::config::AppConfig;
use crate::traits::Tool;

pub struct ConfigManagerTool {
    config_path: PathBuf,
}

impl ConfigManagerTool {
    pub fn new(config_path: PathBuf) -> Self {
        Self { config_path }
    }

    /// Rotate backups (3-deep ring) and create a new .bak from current config.
    /// Rotation: .bak.2 is deleted, .bak.1 → .bak.2, .bak → .bak.1, current → .bak
    async fn create_backup(&self) -> anyhow::Result<()> {
        let bak = self.config_path.with_extension("toml.bak");
        let bak1 = self.config_path.with_extension("toml.bak.1");
        let bak2 = self.config_path.with_extension("toml.bak.2");

        // Rotate: .bak.1 → .bak.2
        if bak1.exists() {
            let _ = tokio::fs::rename(&bak1, &bak2).await;
        }
        // Rotate: .bak → .bak.1
        if bak.exists() {
            let _ = tokio::fs::rename(&bak, &bak1).await;
        }
        // Current → .bak
        tokio::fs::copy(&self.config_path, &bak).await?;
        info!(backup = %bak.display(), "Config backup created (3-deep rotation)");
        Ok(())
    }

    /// Restore from the first available backup in the chain.
    async fn restore_backup(&self) -> anyhow::Result<String> {
        let candidates = [
            self.config_path.with_extension("toml.bak"),
            self.config_path.with_extension("toml.bak.1"),
            self.config_path.with_extension("toml.bak.2"),
        ];

        for candidate in &candidates {
            if candidate.exists() {
                tokio::fs::copy(candidate, &self.config_path).await?;
                warn!(source = %candidate.display(), "Config restored from backup");
                return Ok(format!(
                    "Config restored from {}. Run /reload to apply.",
                    candidate.display()
                ));
            }
        }

        anyhow::bail!("No backups found (checked .bak, .bak.1, .bak.2)")
    }

    /// Validate that a config string parses correctly as AppConfig.
    fn validate_config(content: &str) -> Result<(), String> {
        // First check it's valid TOML
        let _doc: toml::Table = content
            .parse()
            .map_err(|e| format!("Invalid TOML syntax: {}", e))?;

        // Then check it deserializes into our AppConfig
        toml::from_str::<AppConfig>(content)
            .map_err(|e| format!("Invalid config structure: {}", e))?;

        Ok(())
    }
}

#[derive(Deserialize)]
struct ConfigArgs {
    action: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    value: String,
}

#[async_trait]
impl Tool for ConfigManagerTool {
    fn name(&self) -> &str {
        "manage_config"
    }

    fn description(&self) -> &str {
        "Read or update aidaemon's own config.toml. Automatically backs up before changes and validates before saving."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_config",
            "description": "Read or update aidaemon's own config.toml. Backs up before writing and validates changes. Use 'restore' to rollback if something goes wrong. After updating, tell the user to run /reload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "get", "set", "restore"],
                        "description": "'read' = full config, 'get' = read key, 'set' = update key (auto-backup + validate), 'restore' = rollback to last backup"
                    },
                    "key": {
                        "type": "string",
                        "description": "TOML key path, e.g. 'provider.models.primary'. Required for get/set."
                    },
                    "value": {
                        "type": "string",
                        "description": "New value (TOML literal, e.g. '\"gemini-2.0-flash\"' or '[\"ls\", \"git\"]'). Required for set."
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ConfigArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "read" => {
                let content = tokio::fs::read_to_string(&self.config_path).await?;
                Ok(format!("Current config.toml:\n\n{}", content))
            }
            "get" => {
                if args.key.is_empty() {
                    return Ok("Error: 'key' is required for 'get' action.".to_string());
                }
                let content = tokio::fs::read_to_string(&self.config_path).await?;
                let doc: toml::Value = content.parse()?;
                let value = navigate_toml(&doc, &args.key);
                match value {
                    Some(v) => Ok(format!("{} = {}", args.key, v)),
                    None => Ok(format!("Key '{}' not found in config.", args.key)),
                }
            }
            "set" => {
                if args.key.is_empty() || args.value.is_empty() {
                    return Ok("Error: 'key' and 'value' are required for 'set' action.".to_string());
                }

                let content = tokio::fs::read_to_string(&self.config_path).await?;
                let mut doc: toml::Table = content.parse()?;

                // Parse the new value as TOML
                let new_value: toml::Value = args.value.parse().or_else(|_| {
                    let wrapped = format!("v = {}", args.value);
                    let table: toml::Table = wrapped.parse()?;
                    Ok::<toml::Value, toml::de::Error>(table["v"].clone())
                })?;

                set_toml_value(&mut doc, &args.key, new_value)?;

                let new_content = toml::to_string_pretty(&toml::Value::Table(doc))?;

                // Validate before writing
                if let Err(e) = Self::validate_config(&new_content) {
                    return Ok(format!(
                        "Refused to save: {}.\n\nThe config was NOT modified. Fix the value and try again.",
                        e
                    ));
                }

                // Backup current config before writing
                if let Err(e) = self.create_backup().await {
                    warn!("Failed to create backup: {}", e);
                    // Continue anyway — better to fix config than refuse
                }

                tokio::fs::write(&self.config_path, &new_content).await?;

                Ok(format!(
                    "Updated {} = {}\n\nBackup saved to config.toml.bak. Config validated and saved. Run /reload to apply changes.",
                    args.key, args.value
                ))
            }
            "restore" => {
                match self.restore_backup().await {
                    Ok(msg) => Ok(msg),
                    Err(e) => Ok(format!("Restore failed: {}", e)),
                }
            }
            _ => Ok(format!("Unknown action: {}. Use 'read', 'get', 'set', or 'restore'.", args.action)),
        }
    }
}

/// Navigate a TOML value by a dotted key path like "provider.models.primary".
fn navigate_toml<'a>(value: &'a toml::Value, path: &str) -> Option<&'a toml::Value> {
    let mut current = value;
    for key in path.split('.') {
        current = current.get(key)?;
    }
    Some(current)
}

/// Set a value in a TOML table at a dotted key path.
fn set_toml_value(table: &mut toml::Table, path: &str, value: toml::Value) -> anyhow::Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        anyhow::bail!("Empty key path");
    }

    let mut current = table;
    for &key in &parts[..parts.len() - 1] {
        current = current
            .entry(key)
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("'{}' is not a table", key))?;
    }

    let last_key = parts.last().unwrap();
    current.insert(last_key.to_string(), value);
    Ok(())
}
