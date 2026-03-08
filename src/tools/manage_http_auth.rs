use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::{AppConfig, HttpAuthProfile, HttpAuthType};
use crate::oauth::SharedHttpProfiles;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::http_request::HttpRequestTool;
use crate::tools::terminal::ApprovalRequest;
use crate::traits::{StateStore, Tool, ToolCapabilities};
use crate::types::ApprovalResponse;

const MAX_VERIFY_TIMEOUT_SECS: u64 = 60;
const DEFAULT_VERIFY_TIMEOUT_SECS: u64 = 20;

#[derive(Deserialize, Default)]
struct ManageHttpAuthArgs {
    action: String,
    profile: Option<String>,
    auth_type: Option<String>,
    allowed_domains: Option<Vec<String>>,
    header_name: Option<String>,
    username: Option<String>,
    user_id: Option<String>,
    url: Option<String>,
    method: Option<String>,
    timeout_secs: Option<u64>,
    #[serde(default)]
    _session_id: String,
}

#[derive(Clone)]
struct ManualProfileResolution {
    auth_type: HttpAuthType,
    allowed_domains: Vec<String>,
    live_profile: HttpAuthProfile,
    missing_direct_fields: Vec<&'static str>,
    missing_secret_fields: Vec<&'static str>,
    detected_secret_fields: Vec<&'static str>,
    configured_secret_fields: Vec<&'static str>,
}

impl ManualProfileResolution {
    fn status_label(&self) -> &'static str {
        if !self.missing_direct_fields.is_empty() || !self.missing_secret_fields.is_empty() {
            "incomplete"
        } else if !self.detected_secret_fields.is_empty() {
            "stored-secrets-detected"
        } else {
            "ready"
        }
    }
}

pub struct ManageHttpAuthTool {
    config_path: PathBuf,
    profiles: SharedHttpProfiles,
    approval_tx: mpsc::Sender<ApprovalRequest>,
    state_store: Arc<dyn StateStore>,
}

impl ManageHttpAuthTool {
    pub fn new(
        config_path: PathBuf,
        profiles: SharedHttpProfiles,
        approval_tx: mpsc::Sender<ApprovalRequest>,
        state_store: Arc<dyn StateStore>,
    ) -> Self {
        Self {
            config_path,
            profiles,
            approval_tx,
            state_store,
        }
    }

    fn validate_profile_name(raw: &str) -> anyhow::Result<String> {
        let trimmed = raw.trim();
        anyhow::ensure!(!trimmed.is_empty(), "Profile name must not be empty");
        anyhow::ensure!(
            trimmed
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')),
            "Profile name '{}' is invalid. Use only letters, numbers, '_' or '-'.",
            trimmed
        );
        Ok(trimmed.to_ascii_lowercase())
    }

    fn parse_auth_type(raw: &str) -> anyhow::Result<HttpAuthType> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "oauth1a" | "oauth_1a" | "oauth-1a" => Ok(HttpAuthType::Oauth1a),
            "bearer" => Ok(HttpAuthType::Bearer),
            "header" => Ok(HttpAuthType::Header),
            "basic" => Ok(HttpAuthType::Basic),
            other => anyhow::bail!(
                "Unknown auth_type '{}'. Use one of: oauth1a, bearer, header, basic.",
                other
            ),
        }
    }

    fn auth_type_name(auth_type: &HttpAuthType) -> &'static str {
        match auth_type {
            HttpAuthType::Oauth1a => "oauth1a",
            HttpAuthType::Bearer => "bearer",
            HttpAuthType::Header => "header",
            HttpAuthType::Basic => "basic",
        }
    }

    fn required_secret_fields(auth_type: &HttpAuthType) -> &'static [&'static str] {
        match auth_type {
            HttpAuthType::Oauth1a => &[
                "api_key",
                "api_secret",
                "access_token",
                "access_token_secret",
            ],
            HttpAuthType::Bearer => &["token"],
            HttpAuthType::Header => &["header_value"],
            HttpAuthType::Basic => &["password"],
        }
    }

    fn required_direct_fields(auth_type: &HttpAuthType) -> &'static [&'static str] {
        match auth_type {
            HttpAuthType::Oauth1a | HttpAuthType::Bearer => &[],
            HttpAuthType::Header => &["header_name"],
            HttpAuthType::Basic => &["username"],
        }
    }

    fn optional_direct_fields(auth_type: &HttpAuthType) -> &'static [&'static str] {
        match auth_type {
            HttpAuthType::Oauth1a => &["user_id"],
            _ => &[],
        }
    }

    fn is_direct_field_relevant(auth_type: &HttpAuthType, field: &str) -> bool {
        Self::required_direct_fields(auth_type).contains(&field)
            || Self::optional_direct_fields(auth_type).contains(&field)
    }

    fn keychain_field_name(profile: &str, field: &str) -> String {
        format!("http_auth_{}_{}", profile, field)
    }

    fn normalize_domains(domains: &[String]) -> anyhow::Result<Vec<String>> {
        let normalized: Vec<String> = domains
            .iter()
            .map(|domain| domain.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|domain| !domain.is_empty())
            .map(|domain| {
                anyhow::ensure!(
                    !domain.contains("://")
                        && !domain.contains('/')
                        && !domain.contains(char::is_whitespace),
                    "Allowed domain '{}' must be a bare host/domain, not a URL or path.",
                    domain
                );
                Ok(domain)
            })
            .collect::<anyhow::Result<_>>()?;
        anyhow::ensure!(
            !normalized.is_empty(),
            "`allowed_domains` must contain at least one hostname."
        );
        Ok(normalized)
    }

    async fn load_config_doc(&self) -> anyhow::Result<toml::Table> {
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        Ok(content.parse()?)
    }

    fn validate_config(content: &str) -> Result<(), String> {
        let _doc: toml::Table = content
            .parse()
            .map_err(|e| format!("Invalid TOML syntax: {}", e))?;
        let expanded = crate::config::expand_env_vars(content).map_err(|e| format!("{}", e))?;
        toml::from_str::<AppConfig>(&expanded)
            .map_err(|e| format!("Invalid config structure: {}", e))?;
        Ok(())
    }

    #[cfg(unix)]
    fn set_owner_only_permissions(path: &Path) {
        use std::os::unix::fs::PermissionsExt;

        if let Ok(metadata) = std::fs::metadata(path) {
            let mut perms = metadata.permissions();
            perms.set_mode(0o600);
            let _ = std::fs::set_permissions(path, perms);
        }
    }

    #[cfg(not(unix))]
    fn set_owner_only_permissions(_path: &Path) {}

    async fn create_backup(&self) -> anyhow::Result<()> {
        let bak = self.config_path.with_extension("toml.bak");
        let bak1 = self.config_path.with_extension("toml.bak.1");
        let bak2 = self.config_path.with_extension("toml.bak.2");

        if bak1.exists() {
            let _ = tokio::fs::rename(&bak1, &bak2).await;
        }
        if bak.exists() {
            let _ = tokio::fs::rename(&bak, &bak1).await;
        }
        tokio::fs::copy(&self.config_path, &bak).await?;
        Self::set_owner_only_permissions(&bak);
        Self::set_owner_only_permissions(&bak1);
        Self::set_owner_only_permissions(&bak2);
        Ok(())
    }

    async fn save_config_doc(&self, doc: &toml::Table) -> anyhow::Result<()> {
        let rendered = toml::to_string_pretty(&toml::Value::Table(doc.clone()))?;
        if let Err(err) = Self::validate_config(&rendered) {
            anyhow::bail!("Refused to save invalid config: {}", err);
        }
        if let Err(err) = self.create_backup().await {
            warn!(error = %err, "Failed to create config backup before manage_http_auth change");
        }
        tokio::fs::write(&self.config_path, &rendered).await?;
        Self::set_owner_only_permissions(&self.config_path);
        Ok(())
    }

    fn http_auth_table(doc: &toml::Table) -> Option<&toml::Table> {
        doc.get("http_auth").and_then(toml::Value::as_table)
    }

    fn http_auth_table_mut(doc: &mut toml::Table) -> anyhow::Result<&mut toml::Table> {
        doc.entry("http_auth")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`http_auth` is not a table"))
    }

    fn profile_table<'a>(doc: &'a toml::Table, profile: &str) -> Option<&'a toml::Table> {
        Self::http_auth_table(doc)?
            .get(profile)
            .and_then(toml::Value::as_table)
    }

    fn profile_table_mut<'a>(
        doc: &'a mut toml::Table,
        profile: &str,
    ) -> anyhow::Result<&'a mut toml::Table> {
        Self::http_auth_table_mut(doc)?
            .entry(profile.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("`http_auth.{}` is not a table", profile))
    }

    fn redact_profile_error(error: &str, profile: &HttpAuthProfile) -> String {
        let mut redacted = error.to_string();
        for secret in profile.credential_values() {
            if secret.len() >= 4 {
                redacted = redacted.replace(secret, "[REDACTED]");
            }
        }
        redacted
    }

    fn read_string_field(table: &toml::Table, field: &str) -> Option<String> {
        table
            .get(field)
            .and_then(toml::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
    }

    fn read_allowed_domains(table: &toml::Table) -> Vec<String> {
        table
            .get("allowed_domains")
            .and_then(toml::Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(|value| value.to_ascii_lowercase())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn resolve_manual_profile(
        &self,
        profile_name: &str,
        table: &toml::Table,
    ) -> anyhow::Result<ManualProfileResolution> {
        let auth_type_raw = Self::read_string_field(table, "auth_type")
            .ok_or_else(|| anyhow::anyhow!("Profile '{}' is missing auth_type", profile_name))?;
        let auth_type = Self::parse_auth_type(&auth_type_raw)?;
        let allowed_domains = Self::read_allowed_domains(table);

        let mut missing_direct_fields = Vec::new();
        if allowed_domains.is_empty() {
            missing_direct_fields.push("allowed_domains");
        }
        for field in Self::required_direct_fields(&auth_type) {
            if Self::read_string_field(table, field).is_none() {
                missing_direct_fields.push(field);
            }
        }

        let mut live_profile = HttpAuthProfile {
            auth_type: auth_type.clone(),
            allowed_domains: allowed_domains.clone(),
            api_key: None,
            api_secret: None,
            access_token: None,
            access_token_secret: None,
            user_id: Self::read_string_field(table, "user_id"),
            token: None,
            header_name: Self::read_string_field(table, "header_name"),
            header_value: None,
            username: Self::read_string_field(table, "username"),
            password: None,
        };

        let mut missing_secret_fields = Vec::new();
        let mut detected_secret_fields = Vec::new();
        let mut configured_secret_fields = Vec::new();

        for field in Self::required_secret_fields(&auth_type) {
            let key_name = Self::keychain_field_name(profile_name, field);
            let inline_value = Self::read_string_field(table, field);
            if let Some(ref value) = inline_value {
                if value == "keychain" {
                    match crate::config::resolve_from_keychain(&key_name) {
                        Ok(secret) if !secret.is_empty() => {
                            configured_secret_fields.push(*field);
                            Self::set_secret_field(&mut live_profile, field, Some(secret));
                        }
                        _ => missing_secret_fields.push(*field),
                    }
                    continue;
                }

                configured_secret_fields.push(*field);
                Self::set_secret_field(&mut live_profile, field, Some(value.clone()));
                continue;
            }

            match crate::config::resolve_from_keychain(&key_name) {
                Ok(secret) if !secret.is_empty() => {
                    detected_secret_fields.push(*field);
                    Self::set_secret_field(&mut live_profile, field, Some(secret));
                }
                _ => missing_secret_fields.push(*field),
            }
        }

        Ok(ManualProfileResolution {
            auth_type,
            allowed_domains,
            live_profile,
            missing_direct_fields,
            missing_secret_fields,
            detected_secret_fields,
            configured_secret_fields,
        })
    }

    fn set_secret_field(profile: &mut HttpAuthProfile, field: &str, value: Option<String>) {
        match field {
            "api_key" => profile.api_key = value,
            "api_secret" => profile.api_secret = value,
            "access_token" => profile.access_token = value,
            "access_token_secret" => profile.access_token_secret = value,
            "token" => profile.token = value,
            "header_value" => profile.header_value = value,
            "password" => profile.password = value,
            _ => {}
        }
    }

    async fn sync_manual_profile(
        &self,
        doc: &mut toml::Table,
        profile_name: &str,
    ) -> anyhow::Result<ManualProfileResolution> {
        let Some(profile_table) = Self::profile_table(doc, profile_name) else {
            anyhow::bail!("Profile '{}' was not found after update", profile_name);
        };
        let mut resolution = self.resolve_manual_profile(profile_name, profile_table)?;
        if !resolution.detected_secret_fields.is_empty() {
            {
                let profile_table = Self::profile_table_mut(doc, profile_name)?;
                for field in &resolution.detected_secret_fields {
                    profile_table.insert(
                        (*field).to_string(),
                        toml::Value::String("keychain".to_string()),
                    );
                }
            }
            self.save_config_doc(doc).await?;
            let profile_table = Self::profile_table(doc, profile_name).ok_or_else(|| {
                anyhow::anyhow!("Profile '{}' disappeared during sync", profile_name)
            })?;
            resolution = self.resolve_manual_profile(profile_name, profile_table)?;
        }
        self.profiles
            .write()
            .await
            .insert(profile_name.to_string(), resolution.live_profile.clone());
        Ok(resolution)
    }

    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
        risk_level: RiskLevel,
        warnings: Vec<String>,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level,
                warnings,
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;

        match tokio::time::timeout(Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Ok(ApprovalResponse::Deny),
            Err(_) => Ok(ApprovalResponse::Deny),
        }
    }

    async fn is_oauth_managed_profile(&self, profile_name: &str) -> anyhow::Result<bool> {
        Ok(self
            .state_store
            .get_oauth_connection(profile_name)
            .await?
            .is_some())
    }

    async fn handle_list(&self) -> anyhow::Result<String> {
        let doc = self.load_config_doc().await?;
        let manual_names: Vec<String> = Self::http_auth_table(&doc)
            .map(|table| table.keys().cloned().collect())
            .unwrap_or_default();
        let oauth_connections = self.state_store.list_oauth_connections().await?;

        let mut lines = Vec::new();
        if manual_names.is_empty() {
            lines.push("No manual API auth profiles configured yet.".to_string());
        } else {
            lines.push("Manual API auth profiles:".to_string());
            for name in manual_names {
                let table = Self::profile_table(&doc, &name)
                    .ok_or_else(|| anyhow::anyhow!("Profile '{}' is not a table", name))?;
                match self.resolve_manual_profile(&name, table) {
                    Ok(resolution) => {
                        let domains = if resolution.allowed_domains.is_empty() {
                            "(none)".to_string()
                        } else {
                            resolution.allowed_domains.join(", ")
                        };
                        lines.push(format!(
                            "- {} [{}] domains: {} ({})",
                            name,
                            Self::auth_type_name(&resolution.auth_type),
                            domains,
                            resolution.status_label()
                        ));
                    }
                    Err(err) => {
                        lines.push(format!("- {} [invalid] {}", name, err));
                    }
                }
            }
        }

        if !oauth_connections.is_empty() {
            if !lines.is_empty() {
                lines.push(String::new());
            }
            lines.push("OAuth-managed profiles:".to_string());
            for connection in oauth_connections {
                lines.push(format!(
                    "- {} [{}] manage with OAuth flow tools",
                    connection.service, connection.auth_type
                ));
            }
        }

        if lines.is_empty() {
            lines.push("No API auth profiles are configured yet.".to_string());
        } else {
            lines.push(String::new());
            lines.push(
                "Use `upsert` to create/update a manual profile, `describe` to inspect it, and `verify` to bind stored secrets and refresh it before live API calls."
                    .to_string(),
            );
        }

        Ok(lines.join("\n"))
    }

    async fn handle_describe(&self, profile_name: &str) -> anyhow::Result<String> {
        let doc = self.load_config_doc().await?;
        let Some(profile_table) = Self::profile_table(&doc, profile_name) else {
            if self.is_oauth_managed_profile(profile_name).await? {
                return Ok(format!(
                    "Profile '{}' is managed by OAuth. Use the OAuth management flow for connection state; this tool only manages manual HTTP auth profiles.",
                    profile_name
                ));
            }
            return Ok(format!(
                "Manual auth profile '{}' was not found. Use action='upsert' to create it.",
                profile_name
            ));
        };
        let resolution = self.resolve_manual_profile(profile_name, profile_table)?;

        let mut lines = vec![
            format!("Manual API auth profile `{}`", profile_name),
            format!(
                "- auth_type: `{}`",
                Self::auth_type_name(&resolution.auth_type)
            ),
            format!(
                "- allowed_domains: {}",
                if resolution.allowed_domains.is_empty() {
                    "(none)".to_string()
                } else {
                    resolution.allowed_domains.join(", ")
                }
            ),
            format!("- status: {}", resolution.status_label()),
        ];

        if let Some(header_name) = resolution.live_profile.header_name.as_deref() {
            lines.push(format!("- header_name: `{}`", header_name));
        }
        if let Some(username) = resolution.live_profile.username.as_deref() {
            lines.push(format!("- username: `{}`", username));
        }
        if let Some(user_id) = resolution.live_profile.user_id.as_deref() {
            lines.push(format!("- user_id: `{}`", user_id));
        }

        if !resolution.configured_secret_fields.is_empty() {
            lines.push(format!(
                "- configured secret fields: {}",
                resolution.configured_secret_fields.join(", ")
            ));
        }
        if !resolution.detected_secret_fields.is_empty() {
            lines.push(format!(
                "- stored but not yet bound in config: {}",
                resolution.detected_secret_fields.join(", ")
            ));
        }
        if !resolution.missing_direct_fields.is_empty() {
            lines.push(format!(
                "- missing direct fields: {}",
                resolution.missing_direct_fields.join(", ")
            ));
        }
        if !resolution.missing_secret_fields.is_empty() {
            lines.push(format!(
                "- missing secure credentials: {}",
                resolution.missing_secret_fields.join(", ")
            ));
        }

        if !resolution.missing_secret_fields.is_empty() {
            lines.push(String::new());
            lines.push("Store missing credentials securely with:".to_string());
            for field in &resolution.missing_secret_fields {
                lines.push(format!(
                    "- `aidaemon keychain set {}`",
                    Self::keychain_field_name(profile_name, field)
                ));
            }
        }

        lines.push(String::new());
        lines.push(
            "After storing or rotating credentials, run `verify` for this profile before making live API calls."
                .to_string(),
        );

        Ok(lines.join("\n"))
    }

    async fn handle_upsert(
        &self,
        profile_name: &str,
        args: &ManageHttpAuthArgs,
    ) -> anyhow::Result<String> {
        if self.is_oauth_managed_profile(profile_name).await? {
            return Ok(format!(
                "Profile '{}' is already managed by OAuth. Use the OAuth flow for that service name, or choose a different manual profile name.",
                profile_name
            ));
        }

        let session_id = if args._session_id.is_empty() {
            "unknown"
        } else {
            args._session_id.as_str()
        };
        let mut doc = self.load_config_doc().await?;
        let existing_auth_type = Self::profile_table(&doc, profile_name)
            .and_then(|table| Self::read_string_field(table, "auth_type"));
        let auth_type = if let Some(ref raw) = args.auth_type {
            Self::parse_auth_type(raw)?
        } else if let Some(raw) = existing_auth_type {
            Self::parse_auth_type(&raw)?
        } else {
            anyhow::bail!("action='upsert' requires auth_type when creating a new profile.");
        };
        let existing_allowed_domains = Self::profile_table(&doc, profile_name)
            .map(Self::read_allowed_domains)
            .unwrap_or_default();
        let allowed_domains = if let Some(ref domains) = args.allowed_domains {
            Self::normalize_domains(domains)?
        } else if !existing_allowed_domains.is_empty() {
            existing_allowed_domains
        } else {
            anyhow::bail!("action='upsert' requires allowed_domains when creating a new profile.");
        };

        let approval_description = format!(
            "Update manual API auth profile '{}' (type={}, domains={})",
            profile_name,
            Self::auth_type_name(&auth_type),
            allowed_domains.join(", ")
        );
        match self
            .request_approval(
                session_id,
                &approval_description,
                RiskLevel::High,
                vec![
                    "Modifies API auth configuration".to_string(),
                    "Can change which domains receive credentials".to_string(),
                ],
            )
            .await?
        {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
            ApprovalResponse::Deny => {
                return Ok("Manual API auth profile update denied by user.".to_string());
            }
        }

        {
            let profile_table = Self::profile_table_mut(&mut doc, profile_name)?;
            profile_table.insert(
                "auth_type".to_string(),
                toml::Value::String(Self::auth_type_name(&auth_type).to_string()),
            );
            profile_table.insert(
                "allowed_domains".to_string(),
                toml::Value::Array(
                    allowed_domains
                        .iter()
                        .cloned()
                        .map(toml::Value::String)
                        .collect(),
                ),
            );

            for field in ["header_name", "username", "user_id"] {
                if !Self::is_direct_field_relevant(&auth_type, field) {
                    profile_table.remove(field);
                    continue;
                }
                let incoming = match field {
                    "header_name" => args.header_name.as_deref(),
                    "username" => args.username.as_deref(),
                    "user_id" => args.user_id.as_deref(),
                    _ => None,
                };
                if let Some(value) = incoming {
                    let trimmed = value.trim();
                    if trimmed.is_empty() {
                        profile_table.remove(field);
                    } else {
                        profile_table
                            .insert(field.to_string(), toml::Value::String(trimmed.to_string()));
                    }
                }
            }

            for field in [
                "api_key",
                "api_secret",
                "access_token",
                "access_token_secret",
                "token",
                "header_value",
                "password",
            ] {
                if !Self::required_secret_fields(&auth_type).contains(&field) {
                    profile_table.remove(field);
                }
            }
        }

        self.save_config_doc(&doc).await?;
        let resolution = self.sync_manual_profile(&mut doc, profile_name).await?;
        info!(
            profile = %profile_name,
            auth_type = Self::auth_type_name(&resolution.auth_type),
            "Saved manual API auth profile"
        );

        let mut lines = vec![
            format!("Saved manual API auth profile `{}`.", profile_name),
            format!(
                "- auth_type: `{}`",
                Self::auth_type_name(&resolution.auth_type)
            ),
            format!(
                "- allowed_domains: {}",
                resolution.allowed_domains.join(", ")
            ),
        ];

        if !resolution.detected_secret_fields.is_empty() {
            lines.push(format!(
                "- bound stored credentials from keychain/env: {}",
                resolution.detected_secret_fields.join(", ")
            ));
        }
        if !resolution.missing_direct_fields.is_empty() {
            lines.push(format!(
                "- missing direct fields: {}",
                resolution.missing_direct_fields.join(", ")
            ));
        }
        if !resolution.missing_secret_fields.is_empty() {
            lines.push(format!(
                "- missing secure credentials: {}",
                resolution.missing_secret_fields.join(", ")
            ));
            lines.push(String::new());
            lines.push("Store the missing credentials with:".to_string());
            for field in &resolution.missing_secret_fields {
                lines.push(format!(
                    "- `aidaemon keychain set {}`",
                    Self::keychain_field_name(profile_name, field)
                ));
            }
            lines.push(String::new());
            lines.push(
                "After storing them, run `verify` for this profile to bind the secrets and refresh the live auth profile."
                    .to_string(),
            );
        } else {
            lines.push("- runtime profile refreshed".to_string());
        }

        Ok(lines.join("\n"))
    }

    async fn handle_remove(&self, profile_name: &str, session_id: &str) -> anyhow::Result<String> {
        let mut doc = self.load_config_doc().await?;
        let Some(http_auth_table) = Self::http_auth_table_mut(&mut doc).ok() else {
            return Ok(format!(
                "Manual auth profile '{}' was not found.",
                profile_name
            ));
        };

        if !http_auth_table.contains_key(profile_name) {
            if self.is_oauth_managed_profile(profile_name).await? {
                return Ok(format!(
                    "Profile '{}' is OAuth-managed. Use the OAuth flow to disconnect it; this tool only removes manual profiles.",
                    profile_name
                ));
            }
            return Ok(format!(
                "Manual auth profile '{}' was not found.",
                profile_name
            ));
        }

        let approval_description = format!(
            "Remove manual API auth profile '{}' and delete its stored credentials",
            profile_name
        );
        match self
            .request_approval(
                session_id,
                &approval_description,
                RiskLevel::High,
                vec![
                    "Deletes API auth configuration".to_string(),
                    "Deletes stored credentials for this profile".to_string(),
                ],
            )
            .await?
        {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
            ApprovalResponse::Deny => {
                return Ok("Manual API auth profile removal denied by user.".to_string());
            }
        }

        http_auth_table.remove(profile_name);
        if http_auth_table.is_empty() {
            doc.remove("http_auth");
        }
        self.save_config_doc(&doc).await?;

        for field in [
            "api_key",
            "api_secret",
            "access_token",
            "access_token_secret",
            "token",
            "header_value",
            "password",
        ] {
            let key = Self::keychain_field_name(profile_name, field);
            if let Err(err) = crate::config::delete_from_keychain(&key) {
                warn!(key = %key, error = %err, "Failed to delete stored credential for removed HTTP auth profile");
            }
        }

        self.profiles.write().await.remove(profile_name);
        Ok(format!(
            "Removed manual API auth profile '{}' and purged its stored credentials.",
            profile_name
        ))
    }

    async fn handle_verify(
        &self,
        profile_name: &str,
        url: Option<&str>,
        method: Option<&str>,
        timeout_secs: Option<u64>,
        session_id: &str,
    ) -> anyhow::Result<String> {
        let mut doc = self.load_config_doc().await?;
        let Some(_) = Self::profile_table(&doc, profile_name) else {
            if self.is_oauth_managed_profile(profile_name).await? {
                return Ok(format!(
                    "Profile '{}' is OAuth-managed. Use the OAuth connection tools to inspect it; this tool verifies manual profiles only.",
                    profile_name
                ));
            }
            return Ok(format!(
                "Manual auth profile '{}' was not found.",
                profile_name
            ));
        };

        let resolution = self.sync_manual_profile(&mut doc, profile_name).await?;
        if !resolution.missing_direct_fields.is_empty()
            || !resolution.missing_secret_fields.is_empty()
        {
            let mut lines = vec![format!(
                "Profile '{}' is not ready for live API calls yet.",
                profile_name
            )];
            if !resolution.missing_direct_fields.is_empty() {
                lines.push(format!(
                    "- missing direct fields: {}",
                    resolution.missing_direct_fields.join(", ")
                ));
            }
            if !resolution.missing_secret_fields.is_empty() {
                lines.push(format!(
                    "- missing secure credentials: {}",
                    resolution.missing_secret_fields.join(", ")
                ));
                lines.push("Store them with:".to_string());
                for field in &resolution.missing_secret_fields {
                    lines.push(format!(
                        "- `aidaemon keychain set {}`",
                        Self::keychain_field_name(profile_name, field)
                    ));
                }
            }
            return Ok(lines.join("\n"));
        }

        let Some(url) = url.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(format!(
                "Profile '{}' is ready. Runtime auth state refreshed for `{}` across domains: {}.",
                profile_name,
                Self::auth_type_name(&resolution.auth_type),
                resolution.allowed_domains.join(", ")
            ));
        };

        let verify_method = method.unwrap_or("GET").trim().to_ascii_uppercase();
        if !matches!(verify_method.as_str(), "GET" | "HEAD") {
            return Ok("verify currently supports only GET or HEAD probes. Use http_request for other methods.".to_string());
        }

        let parsed_url = reqwest::Url::parse(url)
            .map_err(|err| anyhow::anyhow!("Invalid verify URL: {}", err))?;
        if parsed_url.scheme() != "https" {
            return Ok("Verification probe blocked: only HTTPS URLs are allowed.".to_string());
        }
        let request_host = parsed_url.host_str().unwrap_or("");
        let domain_ok = resolution
            .allowed_domains
            .iter()
            .any(|domain| HttpRequestTool::domain_matches(request_host, domain));
        if !domain_ok {
            return Ok(format!(
                "Verification probe blocked: '{}' is not in the allowed domains for profile '{}'.",
                request_host, profile_name
            ));
        }

        let approval_description = format!(
            "Verify manual API auth profile '{}' with {} {}",
            profile_name, verify_method, url
        );
        match self
            .request_approval(
                session_id,
                &approval_description,
                RiskLevel::Medium,
                vec![
                    "Authenticated HTTP probe to external API".to_string(),
                    "Read-only verification request".to_string(),
                ],
            )
            .await?
        {
            ApprovalResponse::AllowOnce
            | ApprovalResponse::AllowSession
            | ApprovalResponse::AllowAlways => {}
            ApprovalResponse::Deny => {
                return Ok("Verification probe denied by user.".to_string());
            }
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(
                timeout_secs
                    .unwrap_or(DEFAULT_VERIFY_TIMEOUT_SECS)
                    .min(MAX_VERIFY_TIMEOUT_SECS),
            ))
            .build()
            .map_err(|err| anyhow::anyhow!("Failed to build HTTP client: {}", err))?;

        let builder = match verify_method.as_str() {
            "HEAD" => client.head(url),
            _ => client.get(url),
        };
        let builder = HttpRequestTool::apply_auth(
            builder,
            &resolution.live_profile,
            &verify_method,
            url,
            None,
            None,
        )
        .map_err(|err| {
            anyhow::anyhow!(
                "Failed to apply auth for profile '{}': {}",
                profile_name,
                Self::redact_profile_error(&err.to_string(), &resolution.live_profile)
            )
        })?;

        let response = builder.send().await.map_err(|err| {
            anyhow::anyhow!(
                "Verification request failed: {}",
                Self::redact_profile_error(&err.to_string(), &resolution.live_profile)
            )
        })?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        if verify_method == "HEAD" {
            return Ok(format!(
                "Verification probe completed for '{}': {} {} (content-type: {}).",
                profile_name, verify_method, status, content_type
            ));
        }

        let body = response.text().await.unwrap_or_default();
        let snippet = if body.is_empty() {
            "(empty body)".to_string()
        } else {
            let boundary = crate::utils::floor_char_boundary(&body, 400);
            let truncated = body.len() > boundary;
            let mut snippet = body[..boundary].to_string();
            if truncated {
                snippet.push_str("...");
            }
            snippet
        };
        Ok(format!(
            "Verification probe completed for '{}': {} {} (content-type: {}).\nBody preview:\n{}",
            profile_name, verify_method, status, content_type, snippet
        ))
    }
}

#[async_trait]
impl Tool for ManageHttpAuthTool {
    fn name(&self) -> &str {
        "manage_http_auth"
    }

    fn description(&self) -> &str {
        "Create, inspect, verify, and remove generic API auth profiles for HTTP APIs"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_http_auth",
            "description": "Create, inspect, verify, and remove manual HTTP auth profiles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "describe", "upsert", "remove", "verify"],
                        "description": "Action"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile name"
                    },
                    "auth_type": {
                        "type": "string",
                        "enum": ["oauth1a", "bearer", "header", "basic"],
                        "description": "Auth type for upsert"
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Allowed API domains"
                    },
                    "header_name": {
                        "type": "string",
                        "description": "Required for header auth"
                    },
                    "username": {
                        "type": "string",
                        "description": "Required for basic auth"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Optional OAuth1a user/account id"
                    },
                    "url": {
                        "type": "string",
                        "description": "Optional safe verify URL"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "HEAD"],
                        "description": "Verify method"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Verify timeout"
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageHttpAuthArgs = serde_json::from_str(arguments)?;
        match args.action.as_str() {
            "list" => self.handle_list().await,
            "describe" => {
                let profile = Self::validate_profile_name(
                    args.profile
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("action='describe' requires 'profile'"))?,
                )?;
                self.handle_describe(&profile).await
            }
            "upsert" => {
                let profile = Self::validate_profile_name(
                    args.profile
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("action='upsert' requires 'profile'"))?,
                )?;
                self.handle_upsert(&profile, &args).await
            }
            "remove" => {
                let profile = Self::validate_profile_name(
                    args.profile
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("action='remove' requires 'profile'"))?,
                )?;
                let session_id = if args._session_id.is_empty() {
                    "unknown"
                } else {
                    args._session_id.as_str()
                };
                self.handle_remove(&profile, session_id).await
            }
            "verify" => {
                let profile = Self::validate_profile_name(
                    args.profile
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("action='verify' requires 'profile'"))?,
                )?;
                let session_id = if args._session_id.is_empty() {
                    "unknown"
                } else {
                    args._session_id.as_str()
                };
                self.handle_verify(
                    &profile,
                    args.url.as_deref(),
                    args.method.as_deref(),
                    args.timeout_secs,
                    session_id,
                )
                .await
            }
            other => Ok(format!(
                "Unknown action '{}'. Use: list, describe, upsert, remove, verify.",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use tempfile::{NamedTempFile, TempDir};

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;

    static ENV_LOCK: Lazy<std::sync::Mutex<()>> = Lazy::new(|| std::sync::Mutex::new(()));

    fn restore_env_var(name: &str, old_value: Option<String>) {
        if let Some(value) = old_value {
            std::env::set_var(name, value);
        } else {
            std::env::remove_var(name);
        }
    }

    async fn test_tool(
        config_path: PathBuf,
        profiles: SharedHttpProfiles,
    ) -> anyhow::Result<ManageHttpAuthTool> {
        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new()?);
        let state = Arc::new(SqliteStateStore::new(&db_path, 32, None, embedding_service).await?);
        let (approval_tx, mut approval_rx) = mpsc::channel::<ApprovalRequest>(4);
        tokio::spawn(async move {
            while let Some(request) = approval_rx.recv().await {
                let _ = request.response_tx.send(ApprovalResponse::AllowOnce);
            }
        });
        Ok(ManageHttpAuthTool::new(
            config_path,
            profiles,
            approval_tx,
            state as Arc<dyn StateStore>,
        ))
    }

    fn write_minimal_config(path: &Path, extra: &str) {
        std::fs::write(
            path,
            format!(
                "[provider]\napi_key = \"test-key\"\n{}\n",
                if extra.is_empty() {
                    String::new()
                } else {
                    format!("\n{}", extra)
                }
            ),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn upsert_creates_profile_and_reports_missing_secret_commands() {
        let config_file = NamedTempFile::new().unwrap();
        write_minimal_config(config_file.path(), "");
        let profiles = Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let tool = test_tool(config_file.path().to_path_buf(), profiles.clone())
            .await
            .unwrap();

        let result = tool
            .call(
                r#"{
                    "action": "upsert",
                    "profile": "stripe",
                    "auth_type": "bearer",
                    "allowed_domains": ["api.stripe.com"],
                    "_session_id": "test"
                }"#,
            )
            .await
            .unwrap();

        assert!(result.contains("Saved manual API auth profile `stripe`."));
        assert!(result.contains("aidaemon keychain set http_auth_stripe_token"));
        let saved = std::fs::read_to_string(config_file.path()).unwrap();
        assert!(saved.contains("[http_auth.stripe]"));
        assert!(saved.contains("auth_type = \"bearer\""));
        assert!(saved.contains("allowed_domains = [\"api.stripe.com\"]"));
        assert!(!saved.contains("token = \"keychain\""));

        let runtime = profiles.read().await;
        assert!(runtime.contains_key("stripe"));
        assert_eq!(runtime["stripe"].allowed_domains, vec!["api.stripe.com"]);
    }

    #[tokio::test]
    async fn verify_binds_detected_env_secret_and_refreshes_runtime_profile() {
        let _guard = ENV_LOCK.lock().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.toml");
        let env_path = temp_dir.path().join(".env");
        write_minimal_config(
            &config_path,
            r#"
[http_auth.demo]
auth_type = "bearer"
allowed_domains = ["api.example.com"]
"#,
        );
        std::fs::write(&env_path, "HTTP_AUTH_DEMO_TOKEN=secret-demo-token\n").unwrap();

        let old_no_keychain = std::env::var("AIDAEMON_NO_KEYCHAIN").ok();
        let old_runtime_env = std::env::var(crate::RUNTIME_ENV_FILE_ENV_KEY).ok();
        std::env::set_var("AIDAEMON_NO_KEYCHAIN", "1");
        std::env::set_var(
            crate::RUNTIME_ENV_FILE_ENV_KEY,
            env_path.to_string_lossy().to_string(),
        );

        let profiles = Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new()));
        let tool = test_tool(config_path.clone(), profiles.clone())
            .await
            .unwrap();
        let result = tool
            .call(r#"{"action":"verify","profile":"demo"}"#)
            .await
            .unwrap();

        restore_env_var("AIDAEMON_NO_KEYCHAIN", old_no_keychain);
        restore_env_var(crate::RUNTIME_ENV_FILE_ENV_KEY, old_runtime_env);

        assert!(result.contains("Profile 'demo' is ready."));
        let saved = std::fs::read_to_string(&config_path).unwrap();
        assert!(saved.contains("token = \"keychain\""));

        let runtime = profiles.read().await;
        let profile = runtime.get("demo").expect("runtime profile inserted");
        assert_eq!(profile.token.as_deref(), Some("secret-demo-token"));
    }
}
