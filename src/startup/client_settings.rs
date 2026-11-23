//! The Settings page of the owner's apps, on the daemon: every setting the
//! config file holds or the example config documents, runtime bots and MCP
//! servers, and each change an app makes. Config changes run through
//! `manage_config`'s own path ([`ConfigManagerTool::run_owner_call`]): the
//! same validation, backups and refusals as a change asked for in chat, and
//! the same approvals, asked by the gateway.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use aidaemon_client_gateway::{SettingsDone, SettingsGate};
use aidaemon_client_protocol::{
    Applies, ApplyHow, BotInfo, CredentialState, DatabaseAction, McpAction, McpServerInfo,
    ProviderChoice, Risk, Setting, SettingGroup, SettingKind, Settings, SettingsChange,
};
use serde_json::{Value, json};

use crate::startup::client_providers::{self, ProviderFlow};
use crate::startup::settings_upkeep::{self, DbUpkeep, Trial};
use crate::tools::ConfigManagerTool;
use crate::traits::StateStore;
use aidaemon_runtime::permissions::{AgentChoice, OwnerLevel, Permissions};

/// What the example config documents about one setting.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Documented {
    /// TOML text of the example's value.
    example: String,
    description: String,
    /// In a list of tables (`[[telegram_bots]]`): only an entry's own
    /// settings, never offered unset.
    in_list: bool,
}

/// The example config, by dotted key (list indexes removed), and the order
/// its tables come in.
struct Catalogue {
    keys: HashMap<String, Documented>,
    order: Vec<String>,
}

static CATALOGUE: LazyLock<Catalogue> =
    LazyLock::new(|| catalogue(include_str!("../../config.toml.example")));

/// Tables whose entries are named by the owner (`[mcp.<name>]`): their
/// example entries are illustrations, not settings to offer.
const NAMED_TABLES: &[&str] = &["mcp", "http_auth", "oauth", "cli_agents.tools", "exposure"];

fn section_of(line: &str) -> Option<(String, bool)> {
    let line = line.trim_start_matches('#').trim();
    let list = line.starts_with("[[");
    let inner = line.strip_prefix('[')?;
    let inner = inner.strip_prefix('[').unwrap_or(inner);
    let end = inner.find(']')?;
    let name = inner[..end].trim().replace('"', "");
    let ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    ok.then_some((name, list))
}

/// `key = value  # note` (commented out or not) as key, value, note.
fn entry_of(line: &str) -> Option<(String, String, String)> {
    let line = line.trim_start_matches('#').trim();
    let (key, rest) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty()
        || !key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
    {
        return None;
    }
    let rest = rest.trim();
    // The value ends where a ` #` outside a string starts the note.
    let mut in_string = None;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        match (in_string, c) {
            (None, '"' | '\'') => in_string = Some(c),
            (Some(q), c) if c == q => in_string = None,
            (None, '#') => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    let value = rest[..end].trim().to_string();
    let note = rest[end..].trim_start_matches('#').trim().to_string();
    toml::from_str::<toml::Table>(&format!("v = {value}")).ok()?;
    Some((key.to_string(), value, note))
}

fn catalogue(text: &str) -> Catalogue {
    let mut keys = HashMap::new();
    let mut order = Vec::new();
    let mut section: Option<(String, bool)> = None;
    let mut notes: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            notes.clear();
            continue;
        }
        if let Some(found) = section_of(line) {
            let top = found.0.split('.').next().unwrap_or_default().to_string();
            if !order.contains(&top) {
                order.push(top);
            }
            section = Some(found);
            notes.clear();
            continue;
        }
        if let Some((key, value, note)) = entry_of(line) {
            let Some((name, list)) = section.as_ref() else {
                continue;
            };
            let path = format!("{name}.{key}");
            let mut description = notes.join(" ");
            if !note.is_empty() {
                if !description.is_empty() {
                    description.push(' ');
                }
                description.push_str(&note);
            }
            keys.entry(path).or_insert(Documented {
                example: value,
                description,
                in_list: *list,
            });
            notes.clear();
            continue;
        }
        if let Some(note) = line.strip_prefix('#') {
            notes.push(note.trim().to_string());
        }
    }
    Catalogue { keys, order }
}

/// `a[2].b` as `a.b`: the example config's key for a list entry's setting.
fn plain_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut skipping = false;
    for c in key.chars() {
        match c {
            '[' => skipping = true,
            ']' => skipping = false,
            c if !skipping => out.push(c),
            _ => {}
        }
    }
    out
}

/// Every leaf of `value` under `path`: tables open up, a list of tables
/// opens by index, any other value is one leaf.
fn leaves(path: &str, value: &toml::Value, out: &mut Vec<(String, toml::Value)>) {
    match value {
        toml::Value::Table(table) if !table.is_empty() => {
            for (key, value) in table {
                let child = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                leaves(&child, value, out);
            }
        }
        toml::Value::Array(items)
            if !items.is_empty() && items.iter().all(toml::Value::is_table) =>
        {
            for (index, item) in items.iter().enumerate() {
                leaves(&format!("{path}[{index}]"), item, out);
            }
        }
        _ => out.push((path.to_string(), value.clone())),
    }
}

fn kind_of(value: &toml::Value) -> SettingKind {
    match value {
        toml::Value::String(_) | toml::Value::Datetime(_) => SettingKind::Text,
        toml::Value::Integer(_) | toml::Value::Float(_) => SettingKind::Number,
        toml::Value::Boolean(_) => SettingKind::Bool,
        toml::Value::Array(_) => SettingKind::List,
        toml::Value::Table(_) => SettingKind::Table,
    }
}

fn kind_of_text(text: &str) -> SettingKind {
    toml::from_str::<toml::Table>(&format!("v = {text}"))
        .ok()
        .and_then(|t| t.get("v").map(kind_of))
        .unwrap_or(SettingKind::Text)
}

/// When a change to `key` takes effect: providers and models reload in
/// place; the rest is read at start.
fn applies_of(key: &str) -> Applies {
    // AIdaemon's level and the coding agents' choices are live
    // (`aidaemon_runtime::permissions`): every write takes them again.
    if key == "authority.owner_default"
        || key.starts_with("cli_agents.permissions")
        || key.starts_with("cli_agents.models")
    {
        return Applies::Now;
    }
    match key.split(['.', '[']).next().unwrap_or_default() {
        "provider" | "decisions" | "surfaces" => Applies::Reload,
        _ => Applies::Restart,
    }
}

/// A change that can cut every app off (the gateway's address, or the
/// tunnel that publishes it): tried, and rolled back unless an app
/// reconnects.
fn on_trial(key: &str) -> bool {
    let top = key.split(['.', '[']).next().unwrap_or_default();
    match top {
        "exposure" => true,
        "client_gateway" => {
            key == "client_gateway"
                || matches!(
                    key.rsplit('.').next(),
                    Some("bind" | "port" | "enabled" | "token_file")
                )
        }
        _ => false,
    }
}

/// Settings that let the daemon start programs or act with more reach;
/// changing them from an app asks, beyond what `manage_config` guards.
fn host_guarded(key: &str) -> bool {
    let top = key.split(['.', '[']).next().unwrap_or_default();
    matches!(
        top,
        "mcp" | "cli_agents" | "execution" | "users" | "authority" | "nodes"
    ) || key == "terminal.permission_mode"
        || key.starts_with("terminal.allowed")
}

fn guarded(key: &str) -> bool {
    ConfigManagerTool::is_guarded_key(key) || host_guarded(key) || on_trial(key)
}

fn risk_of(risk: aidaemon_core::ApprovalRisk) -> Risk {
    match risk {
        aidaemon_core::ApprovalRisk::Low => Risk::Low,
        aidaemon_core::ApprovalRisk::Medium => Risk::Medium,
        aidaemon_core::ApprovalRisk::High => Risk::High,
        aidaemon_core::ApprovalRisk::Critical => Risk::Critical,
        _ => Risk::High,
    }
}

/// The first paragraph of a `manage_config` reply: what changed, without
/// the chat instructions ("Run /reload") the app shows its own way.
fn first_paragraph(text: &str) -> String {
    text.split("\n\n").next().unwrap_or(text).trim().to_string()
}

/// Whether a change under `[authority]` widens what the daemon may do.
/// `None`: not an authority setting.
fn widens_authority(key: &str, value: Option<&str>, current: &toml::Table) -> Option<bool> {
    if key != "authority" && !key.starts_with("authority.") && !key.starts_with("authority[") {
        return None;
    }
    let parsed = value.and_then(|text| {
        toml::from_str::<toml::Table>(&format!("v = {text}"))
            .ok()
            .and_then(|t| t.get("v").cloned())
    });
    Some(match key {
        // Wider when the new level allows more than the current one (unset
        // is Standard); a value this daemon cannot read is asked.
        "authority.owner_default" => {
            let level_of = |value: Option<&toml::Value>| -> Option<OwnerLevel> {
                match value {
                    None => Some(OwnerLevel::default()),
                    Some(value) => value.clone().try_into().ok(),
                }
            };
            let now = level_of(
                current
                    .get("authority")
                    .and_then(|authority| authority.get("owner_default")),
            )
            .unwrap_or_default();
            if value.is_some() && parsed.is_none() {
                return Some(true);
            }
            level_of(parsed.as_ref()).is_none_or(|next| next > now)
        }
        "authority.trusted_roots" => {
            let now: HashSet<String> = current
                .get("authority")
                .and_then(|a| a.get("trusted_roots"))
                .and_then(toml::Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            match parsed.as_ref().and_then(toml::Value::as_array) {
                Some(items) => items
                    .iter()
                    .any(|v| v.as_str().is_none_or(|root| !now.contains(root))),
                None => value.is_some(),
            }
        }
        // Removing a setting or the table goes back to the defaults.
        _ => value.is_some(),
    })
}

fn provider_args(action: &str, provider: &ProviderChoice) -> Value {
    json!({
        "action": action,
        "provider": provider.preset,
        "base_url": provider.base_url,
        "primary_model": provider.primary_model,
        "fast_model": provider.fast_model,
        "smart_model": provider.smart_model,
    })
}

/// The settings side of the daemon's client host.
/// The refusal for a failover change made from an out-of-date page.
pub(crate) const STALE_FAILOVERS: &str =
    "The providers changed since this page was read; it shows them as they are now. Try again.";

pub(crate) struct SettingsHost {
    pub config_path: PathBuf,
    pub tool: ConfigManagerTool,
    pub state: Arc<dyn StateStore>,
    pub mcp: Option<crate::mcp::McpRegistry>,
    /// For `/reload`; `None` in tests.
    pub commands: Option<crate::channels::commands::CommandContext>,
    /// Saved changes waiting for a reload, and for a restart.
    pub pending: Mutex<(bool, bool)>,
    pub trial: Option<Arc<Trial>>,
    /// A ChatGPT sign-in or key check under way from the Providers tab.
    pub flow: ProviderFlow,
    /// AIdaemon's permission level and the coding agents' choices, live.
    pub permissions: Permissions,
}

impl SettingsHost {
    pub(crate) fn new(
        config_path: PathBuf,
        state: Arc<dyn StateStore>,
        mcp: Option<crate::mcp::McpRegistry>,
        commands: Option<crate::channels::commands::CommandContext>,
    ) -> Self {
        let trial = settings_upkeep::arm_trial(&config_path);
        // The daemon's live value; one of its own read from the file when
        // there is no daemon (tests).
        let permissions = match crate::startup::permissions::daemon() {
            Some((permissions, _)) => permissions.clone(),
            None => std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|text| text.parse::<toml::Table>().ok())
                .map(|table| Permissions::from_config(&table))
                .unwrap_or_default(),
        };
        Self {
            tool: ConfigManagerTool::new(
                crate::config::config_file(&config_path),
                crate::config::secret_store().clone(),
            )
            .with_permissions(permissions.clone()),
            permissions,
            config_path,
            state,
            mcp,
            commands,
            pending: Mutex::new((false, false)),
            trial,
            flow: ProviderFlow::default(),
        }
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, (bool, bool)> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn note_saved(&self, applies: Applies) {
        let mut pending = self.pending();
        match applies {
            Applies::Reload => pending.0 = true,
            Applies::Restart => pending.1 = true,
            _ => {}
        }
    }

    fn current(&self) -> anyhow::Result<toml::Table> {
        Ok(std::fs::read_to_string(&self.config_path)?.parse()?)
    }

    pub(crate) fn client_seen(&self) {
        if let Some(trial) = &self.trial {
            trial.confirm();
        }
    }

    pub(crate) async fn settings(&self) -> anyhow::Result<Settings> {
        let current = self.current()?;
        let mut found = Vec::new();
        leaves("", &toml::Value::Table(current.clone()), &mut found);
        let catalogue = &*CATALOGUE;
        let mut groups: BTreeMap<String, Vec<Setting>> = BTreeMap::new();
        let mut present: HashSet<String> = HashSet::new();
        for (key, value) in found {
            let plain = plain_key(&key);
            let documented = catalogue.keys.get(&plain);
            let secret = ConfigManagerTool::is_secret_key(&key);
            present.insert(plain.clone());
            let setting = Setting {
                value: (!secret).then(|| value.to_string()),
                kind: if secret {
                    SettingKind::Secret
                } else {
                    kind_of(&value)
                },
                secret_set: secret && value.as_str().is_none_or(|v| !v.trim().is_empty()),
                description: documented
                    .map(|d| d.description.clone())
                    .unwrap_or_default(),
                example: documented.filter(|_| !secret).map(|d| d.example.clone()),
                applies: applies_of(&key),
                guarded: guarded(&key),
                key,
            };
            groups
                .entry(group_of(&setting.key))
                .or_default()
                .push(setting);
        }
        // What the example documents and the file leaves to its default.
        for (key, documented) in &catalogue.keys {
            if documented.in_list
                || present.contains(key)
                || NAMED_TABLES
                    .iter()
                    .any(|t| key.starts_with(&format!("{t}.")))
                || present
                    .iter()
                    .any(|have| key.starts_with(&format!("{have}.")))
            {
                continue;
            }
            let secret = ConfigManagerTool::is_secret_key(key);
            groups.entry(group_of(key)).or_default().push(Setting {
                key: key.clone(),
                value: None,
                kind: if secret {
                    SettingKind::Secret
                } else {
                    kind_of_text(&documented.example)
                },
                secret_set: false,
                description: documented.description.clone(),
                example: (!secret).then(|| documented.example.clone()),
                applies: applies_of(key),
                guarded: guarded(key),
            });
        }
        let rank = |name: &str| {
            catalogue
                .order
                .iter()
                .position(|n| n == name)
                .unwrap_or(usize::MAX)
        };
        let mut groups: Vec<SettingGroup> = groups
            .into_iter()
            .map(|(name, mut settings)| {
                settings.sort_by(|a, b| a.key.cmp(&b.key));
                SettingGroup { name, settings }
            })
            .collect();
        groups.sort_by(|a, b| rank(&a.name).cmp(&rank(&b.name)).then(a.name.cmp(&b.name)));

        let bots = self
            .state
            .get_dynamic_bots()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|bot| BotInfo {
                id: bot.id,
                platform: bot.channel_type,
                allowed_user_ids: bot.allowed_user_ids,
                created_at: bot.created_at,
            })
            .collect();
        let in_file: HashSet<String> = current
            .get("mcp")
            .and_then(toml::Value::as_table)
            .map(|t| t.keys().cloned().collect())
            .unwrap_or_default();
        let mcp_servers = match &self.mcp {
            Some(registry) => registry
                .list_servers_with_status()
                .await
                .unwrap_or_default()
                .into_iter()
                .map(|server| McpServerInfo {
                    from_config: in_file.contains(&server.name),
                    command: std::iter::once(server.command)
                        .chain(server.args)
                        .collect::<Vec<_>>()
                        .join(" "),
                    name: server.name,
                    enabled: server.enabled,
                    tools: server.tool_names,
                })
                .collect(),
            None => Vec::new(),
        };
        let db_path = current
            .get("state")
            .and_then(|s| s.get("db_path"))
            .and_then(toml::Value::as_str)
            .unwrap_or("aidaemon.db")
            .to_string();
        let (reload_pending, restart_pending) = *self.pending();
        let (sign_in, checked) = self.flow.shown().await;
        let mut connected = client_providers::connected(&current);
        super::provider_usage::fill(&current, &mut connected).await;
        Ok(Settings {
            connected,
            chatgpt_signed_in: client_providers::chatgpt_signed_in(),
            sign_in,
            checked,
            config_path: self.config_path.display().to_string(),
            groups,
            bots,
            mcp_servers,
            reload_pending,
            restart_pending,
            notices: settings_upkeep::notices(&self.config_path)
                .into_iter()
                .chain(crate::model_upgrade::notices(&self.config_path))
                .collect(),
            providers: crate::setup::model::options(),
            database_path: crate::startup::db_security::resolve_db_path(
                &self.config_path,
                &db_path,
            )
            .display()
            .to_string(),
            coding_agents: coding_agents(&self.permissions).await,
            permission_level: Some(wire_level(self.permissions.level())),
            database_encrypted: cfg!(feature = "encryption")
                && std::env::var("AIDAEMON_ALLOW_PLAINTEXT_DB").is_err(),
        })
    }

    /// The `manage_config` call a config change is.
    fn config_call(&self, change: &SettingsChange) -> Result<Option<Value>, String> {
        Ok(Some(match change {
            SettingsChange::Set { key, value } => {
                let key = key.trim();
                if key.is_empty() {
                    return Err("Name the setting to change".to_string());
                }
                match key {
                    "state.db_path" => {
                        return Err("Move the database with the Database actions, so its \
                                    data goes with it"
                            .to_string());
                    }
                    "state.encryption_key" => {
                        return Err("Rotate the database key with the Database actions; a \
                                    new key set here would lock the data out"
                            .to_string());
                    }
                    _ => {}
                }
                match value {
                    Some(value) => json!({"action": "set", "key": key, "value": value}),
                    None => json!({"action": "unset", "key": key}),
                }
            }
            SettingsChange::SwitchProvider { provider, .. } => {
                provider_args("switch_provider", provider)
            }
            SettingsChange::AddFailover { provider, .. } => {
                provider_args("add_failover_provider", provider)
            }
            SettingsChange::RemoveFailover { index, .. } => {
                json!({"action": "remove_failover_provider", "failover_index": index})
            }
            SettingsChange::MoveFailover { index, to, .. } => json!({
                "action": "move_failover_provider",
                "failover_index": index,
                "failover_to_index": to,
            }),
            SettingsChange::RestoreBackup => json!({"action": "restore"}),
            _ => return Ok(None),
        }))
    }

    pub(crate) async fn gate(&self, change: &SettingsChange) -> Result<SettingsGate, String> {
        let ask = |summary: String, other_device: bool| SettingsGate::Ask {
            summary,
            warnings: Vec::new(),
            risk: Risk::High,
            other_device,
        };
        // A change to failover `index` made from a page that showed another
        // provider there (the order changed since it was read) is refused:
        // it would move or remove the wrong one.
        if let SettingsChange::MoveFailover { index, expect, .. }
        | SettingsChange::RemoveFailover { index, expect } = change
            && !expect.is_empty()
        {
            let current = self.current().map_err(|e| e.to_string())?;
            let there = client_providers::connected(&current)
                .into_iter()
                .find(|p| p.failover == Some(*index))
                .map(|p| p.name);
            if there.as_deref() != Some(expect.as_str()) {
                return Err(STALE_FAILOVERS.to_string());
            }
        }
        if let SettingsChange::Set { key, value } = change {
            let current = self.current().map_err(|e| e.to_string())?;
            if let Some(widens) = widens_authority(key, value.as_deref(), &current) {
                return Ok(if widens {
                    ask(
                        format!(
                            "Widen the assistant's authority: set `{key}` to {}",
                            value.as_deref().unwrap_or("its default")
                        ),
                        true,
                    )
                } else {
                    SettingsGate::Free
                });
            }
        }
        if let Some(call) = self.config_call(change)? {
            // A provider change that stores a key asks even when the
            // config tool cannot word it yet (the key is not stored).
            if let Some(key) = self.key_change(change)? {
                return Ok(ask(key.summary, false));
            }
            if let crate::traits::OwnerApprovalNeed::Ask(request) =
                self.tool.owner_approval(&call).await
            {
                return Ok(SettingsGate::Ask {
                    summary: request.summary,
                    warnings: request.warnings,
                    risk: risk_of(request.risk),
                    other_device: false,
                });
            }
            return Ok(match change {
                SettingsChange::Set { key, value } if host_guarded(key) || on_trial(key) => ask(
                    format!(
                        "Change `{key}` to {}",
                        value.as_deref().unwrap_or("its default")
                    ),
                    false,
                ),
                SettingsChange::RestoreBackup => ask(
                    "Put config.toml back as it was before the last change".to_string(),
                    false,
                ),
                _ => SettingsGate::Free,
            });
        }
        Ok(match change {
            SettingsChange::Secret { key, .. } => {
                if crate::config::keychain_name_for(key).is_none() {
                    return Err(format!(
                        "`{key}` is not read from the keychain, so a secret cannot be stored for it"
                    ));
                }
                ask(format!("Store a new secret for `{key}`"), false)
            }
            SettingsChange::Mcp { name, action } => match action {
                McpAction::Enable | McpAction::Restart => {
                    let registry = self.mcp.as_ref().ok_or("MCP servers are not running")?;
                    let verb = if *action == McpAction::Enable {
                        "enable"
                    } else {
                        "restart"
                    };
                    match registry.pending_launch(verb, name).await {
                        Some((command, args)) => ask(
                            format!(
                                "Start MCP server `{name}`: runs `{}`",
                                std::iter::once(command)
                                    .chain(args)
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            ),
                            false,
                        ),
                        None => SettingsGate::Free,
                    }
                }
                McpAction::Disable | McpAction::Remove => SettingsGate::Free,
                McpAction::Unknown => return Err("Unknown MCP action".to_string()),
            },
            SettingsChange::Database { action } => match action {
                DatabaseAction::MoveTo { path } => {
                    if path.trim().is_empty() {
                        return Err("Name where the database goes".to_string());
                    }
                    ask(
                        format!("Move the database to {} at the next restart", path.trim()),
                        false,
                    )
                }
                DatabaseAction::RotateKey => {
                    if !cfg!(feature = "encryption") {
                        return Err("This build has no database encryption".to_string());
                    }
                    ask(
                        "Re-encrypt the database under a new key at the next restart".to_string(),
                        false,
                    )
                }
                DatabaseAction::Cancel => SettingsGate::Free,
                DatabaseAction::Unknown => return Err("Unknown database action".to_string()),
            },
            SettingsChange::CheckAgents => SettingsGate::Free,
            // Which model an agent works with widens nothing it may do.
            SettingsChange::SetAgentModel { agent, model } => {
                agent_model_choice(agent, model.as_deref())?;
                SettingsGate::Free
            }
            SettingsChange::SetAgentMode { agent, mode } => {
                let coding_agent = aidaemon_runtime::tools::agent_mode::agent_named(agent)
                    .ok_or_else(|| format!("There is no coding agent named `{agent}`"))?;
                // Never looser than AIdaemon itself.
                self.permissions
                    .may_pick(&*coding_agent, agent_mode(*mode)?)?;
                match agent_mode(*mode)? {
                    // Nothing it does is checked from then on.
                    aidaemon_runtime::tools::agent_mode::PermissionMode::FullAccess => ask(
                        format!(
                            "Give {} full access: it runs on this machine with nothing asked, \
                             and your rules don't see what it does.",
                            coding_agent.name()
                        ),
                        false,
                    ),
                    _ => SettingsGate::Free,
                }
            }
            SettingsChange::SetPermissionLevel { level } => {
                let next = owner_level(*level)?;
                let now = self.permissions.level();
                if next <= now {
                    SettingsGate::Free
                } else if next >= OwnerLevel::FullAccess {
                    // As raising `[authority]` from the Advanced tab.
                    ask(
                        "Give AIdaemon full access: every terminal command, including \
                         dangerous ones (rm, sudo), runs without asking, and coding agents may \
                         be given full access too."
                            .to_string(),
                        true,
                    )
                } else {
                    ask(
                        format!(
                            "Raise AIdaemon's permission level from {} to {}",
                            now.title(),
                            next.title()
                        ),
                        true,
                    )
                }
            }
            SettingsChange::AgentSameAsAidaemon { agent } => {
                let coding_agent = aidaemon_runtime::tools::agent_mode::agent_named(agent)
                    .ok_or_else(|| format!("There is no coding agent named `{agent}`"))?;
                let cap = aidaemon_runtime::permissions::agent_cap(self.permissions.level());
                let runs = aidaemon_runtime::tools::agent_mode::modes_within(&*coding_agent, cap)
                    .last()
                    .copied();
                if runs == Some(aidaemon_runtime::tools::agent_mode::PermissionMode::FullAccess) {
                    ask(
                        format!(
                            "Give {} full access, as AIdaemon has: it runs on this machine with \
                             nothing asked, and your rules don't see what it does.",
                            coding_agent.name()
                        ),
                        false,
                    )
                } else {
                    SettingsGate::Free
                }
            }
            SettingsChange::ForgetSignIns => ask(
                "Forget every sign-in the agent's browser kept. Its open tabs close, and you \
                 sign in again next time."
                    .to_string(),
                false,
            ),
            SettingsChange::RemoveBot { .. }
            | SettingsChange::Apply { .. }
            | SettingsChange::CheckKey { .. }
            | SettingsChange::SignIn { .. }
            | SettingsChange::FinishSignIn { .. }
            | SettingsChange::CancelSignIn => SettingsGate::Free,
            _ => return Err("This change is not known to this daemon".to_string()),
        })
    }

    pub(crate) async fn change(
        &self,
        change: SettingsChange,
        approved: bool,
    ) -> Result<SettingsDone, String> {
        if let Some(call) = self.config_call(&change)? {
            // Its key goes into the slot the provider reads first; the
            // earlier one comes back if the change does not go through.
            let key = self.key_change(&change)?;
            // A switch that stores a new main key keeps the one it replaces
            // where `manage_config` finds it: the old main provider becomes
            // a failover with its key.
            let outgoing = matches!(change, SettingsChange::SwitchProvider { .. }) && key.is_some();
            if outgoing {
                stash_outgoing_main_key();
            }
            let stored_key = match key {
                Some(key) => match self.store_provider_key(key).await {
                    Ok(earlier) => Some(earlier),
                    Err(reason) => {
                        drop_outgoing_main_key();
                        return Err(reason);
                    }
                },
                None => None,
            };
            let before = std::fs::read_to_string(&self.config_path).ok();
            let said = self.failover_words(&change);
            let mut done = self.config_change(&change, call, approved).await;
            // A failover change is said with the names Settings shows.
            if let (Ok(done), Some(said)) = (&mut done, said) {
                done.message = said;
            }
            // `manage_config` explains some refusals in an answer that
            // changed nothing: the key must not stay behind then either.
            if stored_key.is_some()
                && let Ok(answer) = &done
                && std::fs::read_to_string(&self.config_path).ok() == before
            {
                done = Err(answer.message.clone());
            }
            match (&done, stored_key) {
                (Err(_), Some(earlier)) => earlier.put_back(),
                (Ok(_), Some(_)) => self.flow.clear_checked().await,
                _ => {}
            }
            if outgoing {
                drop_outgoing_main_key();
            }
            return done;
        }
        self.other_change(change, approved).await
    }

    /// What a failover move or removal did, in the names Settings shows
    /// ("OpenRouter is now Failover 2 after a reload."); `None` for other changes, or
    /// when the provider is not found.
    fn failover_words(&self, change: &SettingsChange) -> Option<String> {
        let (index, to) = match change {
            SettingsChange::MoveFailover { index, to, .. } => (*index, Some(*to)),
            SettingsChange::RemoveFailover { index, .. } => (*index, None),
            _ => return None,
        };
        let current = self.current().ok()?;
        let name = client_providers::connected(&current)
            .into_iter()
            .find(|p| p.failover == Some(index))?
            .name;
        Some(match to {
            Some(to) => format!("{name} is now Failover {} after a reload.", to + 1),
            None => format!("Removed {name}; it stops being tried after a reload."),
        })
    }

    /// The key a provider change stores: its approval card's words, the
    /// keychain name it goes under and the key. `None`: the change stores
    /// none (its provider takes no key, or the main provider keeps its own).
    /// A switch with no key borrows the one a failover of the same provider
    /// has; else the key is asked for, so the main slot's earlier key (made
    /// for another provider) is never sent to this one.
    fn key_change(&self, change: &SettingsChange) -> Result<Option<ProviderKey>, String> {
        let (provider, key, failover) = match change {
            SettingsChange::SwitchProvider { provider, key } => (provider, key, false),
            SettingsChange::AddFailover { provider, key } => (provider, key, true),
            _ => return Ok(None),
        };
        let entry = crate::setup::model::preset(&provider.preset).ok_or("Unknown provider")?;
        let key = key
            .as_ref()
            .map(|k| k.0.trim().to_string())
            .filter(|k| !k.is_empty());
        if !entry.needs_api_key() {
            if key.is_some() {
                return Err(format!("{} takes no API key", entry.display_name));
            }
            if provider.preset == "openai_chatgpt" && !client_providers::chatgpt_signed_in() {
                return Err("Sign in to ChatGPT first".to_string());
            }
            return Ok(None);
        }
        let current = self.current().map_err(|e| e.to_string())?;
        let model = if provider.primary_model.trim().is_empty() {
            entry.models.primary
        } else {
            provider.primary_model.trim()
        };
        let (value, words) = match key {
            Some(key) => (key, "with a new API key".to_string()),
            None if failover => {
                return Err(format!("{} needs its API key", entry.display_name));
            }
            None => {
                let connected = client_providers::connected(&current);
                let keeps_own = connected.first().is_some_and(|main| {
                    main.preset == provider.preset
                        && matches!(
                            main.credential,
                            CredentialState::KeyStored | CredentialState::KeyInFile
                        )
                });
                if keeps_own {
                    return Ok(None);
                }
                let borrowed = connected.iter().skip(1).find_map(|p| {
                    (p.preset == provider.preset && p.credential == CredentialState::KeyStored)
                        .then(|| {
                            let name = crate::config::keychain_name_for(&p.key_setting)?;
                            let key = crate::config::resolve_from_keychain(&name).ok()?;
                            Some((key, p.failover.unwrap_or_default()))
                        })
                        .flatten()
                });
                match borrowed {
                    Some((key, index)) => {
                        (key, format!("with the API key failover {} uses", index + 1))
                    }
                    None => {
                        return Err(format!(
                            "{} needs its API key: paste it to make it the main provider",
                            entry.display_name
                        ));
                    }
                }
            }
        };
        let setting = if failover {
            format!(
                "provider.fallbacks[{}].api_key",
                client_providers::failover_count(&current)
            )
        } else {
            "provider.api_key".to_string()
        };
        let slot = crate::config::keychain_name_for(&setting)
            .ok_or_else(|| format!("`{setting}` is not read from the keychain"))?;
        let summary = if failover {
            format!(
                "Add {} ({model}) as a failover provider, {words}",
                entry.display_name
            )
        } else {
            format!(
                "Make {} ({model}) the main provider, {words}",
                entry.display_name
            )
        };
        Ok(Some(ProviderKey {
            summary,
            slot,
            value,
            entry,
            base_url: provider.base_url.trim().to_string(),
        }))
    }

    /// Check `key` with its provider, then store it in its slot, keeping
    /// what was there to put back.
    async fn store_provider_key(&self, key: ProviderKey) -> Result<EarlierSecret, String> {
        let entry = key.entry;
        let base = if key.base_url.is_empty() {
            entry.base_url
        } else {
            key.base_url.as_str()
        };
        if let Some(endpoint) = aidaemon_providers::key_check::KeyEndpoint::new(entry.kind, base) {
            use aidaemon_providers::key_check::KeyCheck;
            if let KeyCheck::Rejected(detail) =
                aidaemon_providers::key_check::check_key(&endpoint, &key.value).await
            {
                return Err(format!(
                    "{} did not accept that key ({detail}); nothing was changed",
                    entry.display_name
                ));
            }
        }
        use aidaemon_secrets::SecretStore as _;
        let store = crate::config::secret_store();
        let earlier = store.get(&key.slot).ok().map(|s| s.into_exposed());
        crate::config::store_in_keychain(&key.slot, &key.value)
            .map_err(|e| format!("Could not store the key: {e}"))?;
        Ok(EarlierSecret {
            slot: key.slot,
            earlier,
        })
    }

    /// A change `manage_config` makes.
    async fn config_change(
        &self,
        change: &SettingsChange,
        call: Value,
        approved: bool,
    ) -> Result<SettingsDone, String> {
        let key = match change {
            SettingsChange::Set { key, .. } => key.trim().to_string(),
            _ => "provider".to_string(),
        };
        let trial = on_trial(&key);
        let trial_before = settings_upkeep::trial_active(&self.config_path);
        if trial {
            settings_upkeep::begin_trial(&self.config_path)
                .map_err(|e| format!("Could not keep the current config to roll back to: {e}"))?;
        }
        let applies = applies_of(&key);
        let result = self.tool.run_owner_call(&call, approved).await;
        let text = match result {
            Ok(text) => text,
            Err(reason) => {
                if trial && !trial_before {
                    // Nothing changed: there is nothing to roll back.
                    settings_upkeep::cancel_trial(&self.config_path);
                }
                return Err(first_paragraph(&reason));
            }
        };
        self.note_saved(applies);
        if matches!(change, SettingsChange::RestoreBackup) {
            self.note_saved(Applies::Restart);
        }
        // Watching and control switch at once: turning control off gives
        // every screen back to the agent now.
        if key.starts_with("surfaces")
            && let Some(registry) = crate::startup::surfaces::running()
            && let Ok(text) = std::fs::read_to_string(&self.config_path)
            && let Ok(config) = toml::from_str::<SurfacesOnly>(&text)
        {
            registry.set_config(config.surfaces.registry_config());
        }
        Ok(SettingsDone {
            message: first_paragraph(&text),
            applies: Some(applies),
        })
    }

    /// A change that is not a config file edit.
    async fn other_change(
        &self,
        change: SettingsChange,
        approved: bool,
    ) -> Result<SettingsDone, String> {
        let now = |message: String| {
            Ok(SettingsDone {
                message,
                applies: Some(Applies::Now),
            })
        };
        match change {
            SettingsChange::CheckKey { provider, key } => {
                now(self.flow.check_key(&provider, &key.0).await?)
            }
            SettingsChange::SignIn { paste } => {
                let message = self
                    .flow
                    .sign_in(paste, |ok, message| {
                        if let Some(gateway) = crate::startup::client_gateway::running() {
                            gateway.settings_updated(ok, message);
                        }
                    })
                    .await?;
                now(message)
            }
            SettingsChange::FinishSignIn { redirect } => {
                now(self.flow.finish_sign_in(&redirect).await?)
            }
            SettingsChange::CancelSignIn => now(self.flow.cancel_sign_in().await),
            SettingsChange::Secret { key, value } => {
                let name = crate::config::keychain_name_for(&key)
                    .ok_or_else(|| format!("`{key}` is not read from the keychain"))?;
                if value.0.trim().is_empty() {
                    return Err("The secret is empty".to_string());
                }
                crate::config::store_in_keychain(&name, value.0.trim())
                    .map_err(|e| format!("Could not store the secret: {e}"))?;
                let call = json!({"action": "set", "key": key, "value": "keychain"});
                self.tool
                    .run_owner_call(&call, approved)
                    .await
                    .map_err(|reason| first_paragraph(&reason))?;
                let applies = applies_of(&key);
                self.note_saved(applies);
                Ok(SettingsDone {
                    message: format!("Stored the secret for `{key}`."),
                    applies: Some(applies),
                })
            }
            SettingsChange::RemoveBot { bot_id } => {
                let known = self
                    .state
                    .get_dynamic_bots()
                    .await
                    .map_err(|e| e.to_string())?
                    .iter()
                    .any(|bot| bot.id == bot_id);
                if !known {
                    return Err(format!("There is no bot {bot_id}"));
                }
                self.state
                    .delete_dynamic_bot(bot_id)
                    .await
                    .map_err(|e| format!("Could not remove the bot: {e}"))?;
                self.note_saved(Applies::Restart);
                Ok(SettingsDone {
                    message: format!("Removed bot {bot_id}. It stops at the next restart."),
                    applies: Some(Applies::Restart),
                })
            }
            SettingsChange::Mcp { name, action } => {
                let registry = self.mcp.as_ref().ok_or("MCP servers are not running")?;
                let done = match action {
                    McpAction::Enable => registry
                        .enable_server(&name)
                        .await
                        .map(|tools| format!("Enabled `{name}` ({} tools).", tools.len())),
                    McpAction::Disable => registry
                        .disable_server(&name)
                        .await
                        .map(|_| format!("Disabled `{name}`.")),
                    McpAction::Restart => registry
                        .restart_server(&name)
                        .await
                        .map(|tools| format!("Restarted `{name}` ({} tools).", tools.len())),
                    McpAction::Remove => registry
                        .remove_server(&name)
                        .await
                        .map(|_| format!("Removed `{name}`.")),
                    McpAction::Unknown => return Err("Unknown MCP action".to_string()),
                };
                done.map(|message| SettingsDone {
                    message,
                    applies: Some(Applies::Now),
                })
                .map_err(|e| e.to_string())
            }
            SettingsChange::Apply { how } => match how {
                ApplyHow::Reload => {
                    let commands = self
                        .commands
                        .as_ref()
                        .ok_or("This daemon cannot reload its config in place")?;
                    let report = commands.reload_config().await?;
                    self.pending().0 = false;
                    Ok(SettingsDone {
                        message: first_paragraph(&report),
                        applies: Some(Applies::Now),
                    })
                }
                ApplyHow::Restart => {
                    tokio::spawn(async {
                        // Let the reply reach the app first.
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        settings_upkeep::restart_in_place();
                    });
                    Ok(SettingsDone {
                        message: "Restarting. The app reconnects on its own.".to_string(),
                        applies: Some(Applies::Now),
                    })
                }
                ApplyHow::Unknown => Err("Unknown way to apply".to_string()),
            },
            SettingsChange::Database { action } => {
                let cancel = matches!(action, DatabaseAction::Cancel);
                let message = match action {
                    DatabaseAction::MoveTo { path } => {
                        let path = path.trim().to_string();
                        settings_upkeep::schedule(
                            &self.config_path,
                            &DbUpkeep::MoveTo { path: path.clone() },
                        )
                        .map_err(|e| e.to_string())?;
                        format!("The database moves to {path} at the next restart.")
                    }
                    DatabaseAction::RotateKey => {
                        settings_upkeep::schedule(&self.config_path, &DbUpkeep::RotateKey)
                            .map_err(|e| e.to_string())?;
                        "The database key is rotated at the next restart.".to_string()
                    }
                    DatabaseAction::Cancel => {
                        if settings_upkeep::cancel(&self.config_path) {
                            "Cancelled the scheduled database upkeep.".to_string()
                        } else {
                            "Nothing was scheduled.".to_string()
                        }
                    }
                    DatabaseAction::Unknown => return Err("Unknown database action".to_string()),
                };
                if !cancel {
                    self.note_saved(Applies::Restart);
                }
                Ok(SettingsDone {
                    message,
                    applies: Some(Applies::Restart),
                })
            }
            SettingsChange::SetPermissionLevel { level } => {
                let level = owner_level(level)?;
                aidaemon_runtime::permissions::save_level(
                    &crate::config::config_file(&self.config_path),
                    &self.permissions,
                    level,
                )
                .await
                .map_err(|error| format!("The level could not be saved: {error}"))?;
                Ok(SettingsDone {
                    message: format!("AIdaemon's permission level is now {}.", level.title()),
                    applies: Some(Applies::Now),
                })
            }
            SettingsChange::AgentSameAsAidaemon { agent } => {
                let coding_agent = aidaemon_runtime::tools::agent_mode::agent_named(&agent)
                    .ok_or_else(|| format!("There is no coding agent named `{agent}`"))?;
                aidaemon_runtime::permissions::save_agent(
                    &crate::config::config_file(&self.config_path),
                    &self.permissions,
                    &agent,
                    Some(AgentChoice::Same),
                )
                .await
                .map_err(|error| format!("The mode could not be saved: {error}"))?;
                Ok(SettingsDone {
                    message: format!("{} now follows AIdaemon's level.", coding_agent.name()),
                    applies: Some(Applies::Now),
                })
            }
            SettingsChange::SetAgentModel { agent, model } => {
                let (coding_agent, model) = agent_model_choice(&agent, model.as_deref())?;
                aidaemon_runtime::permissions::save_agent_model(
                    &crate::config::config_file(&self.config_path),
                    &self.permissions,
                    &agent,
                    model.as_deref(),
                )
                .await
                .map_err(|error| format!("The model could not be saved: {error}"))?;
                Ok(SettingsDone {
                    message: match model {
                        Some(model) => format!("{} now works with {model}.", coding_agent.name()),
                        None => format!(
                            "{} now works with its own default model.",
                            coding_agent.name()
                        ),
                    },
                    applies: Some(Applies::Now),
                })
            }
            SettingsChange::CheckAgents => {
                aidaemon_runtime::tools::agent_mode::check_all().await;
                Ok(SettingsDone {
                    message: "Checked whether each coding agent is available.".to_string(),
                    applies: Some(Applies::Now),
                })
            }
            SettingsChange::SetAgentMode { agent, mode } => {
                use aidaemon_runtime::tools::agent_mode;
                let coding_agent = agent_mode::agent_named(&agent)
                    .ok_or_else(|| format!("There is no coding agent named `{agent}`"))?;
                let mode = agent_mode(mode)?;
                self.permissions.may_pick(&*coding_agent, mode)?;
                aidaemon_runtime::permissions::save_agent(
                    &crate::config::config_file(&self.config_path),
                    &self.permissions,
                    &agent,
                    Some(AgentChoice::Mode(mode)),
                )
                .await
                .map_err(|error| format!("The mode could not be saved: {error}"))?;
                Ok(SettingsDone {
                    message: format!("{} now runs as you chose.", coding_agent.name()),
                    applies: Some(Applies::Now),
                })
            }
            SettingsChange::ForgetSignIns => {
                let message = crate::startup::surfaces::forget_sign_ins().await?;
                Ok(SettingsDone {
                    message,
                    applies: Some(Applies::Now),
                })
            }
            _ => Err("This change is not known to this daemon".to_string()),
        }
    }
}

/// The `[surfaces]` table of a config file, alone.
#[derive(serde::Deserialize)]
struct SurfacesOnly {
    #[serde(default)]
    surfaces: crate::config::SurfacesSettings,
}

/// A key a provider change stores.
struct ProviderKey {
    /// The approval card's words.
    summary: String,
    /// The keychain name it goes under.
    slot: String,
    value: String,
    entry: &'static aidaemon_providers::catalog::CatalogEntry,
    base_url: String,
}

/// Copy the main provider's stored API key to where a switch reads the
/// key of the provider it demotes to a failover.
fn stash_outgoing_main_key() {
    use aidaemon_secrets::SecretStore as _;
    use aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY;
    let store = crate::config::secret_store();
    if let Ok(key) = store.get("api_key")
        && let Err(error) = store.set(PROVIDER_OUTGOING_API_KEY, key.expose())
    {
        tracing::warn!(%error, "Could not keep the outgoing main provider's key");
    }
}

/// Forget the copy [`stash_outgoing_main_key`] made (the switch moved it,
/// or did not go through).
fn drop_outgoing_main_key() {
    use aidaemon_secrets::SecretStore as _;
    let store = crate::config::secret_store();
    if store.contains(aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY) {
        let _ = store.delete(aidaemon_secrets::names::PROVIDER_OUTGOING_API_KEY);
    }
}

/// What a keychain slot held before a provider key replaced it.
struct EarlierSecret {
    slot: String,
    earlier: Option<String>,
}

impl EarlierSecret {
    /// Put it back: the change it was made for did not go through.
    fn put_back(self) {
        use aidaemon_secrets::SecretStore as _;
        let store = crate::config::secret_store();
        let restored = match &self.earlier {
            Some(value) => store.set(&self.slot, value),
            None => store.delete(&self.slot),
        };
        if let Err(error) = restored {
            tracing::warn!(slot = %self.slot, %error, "Could not put back a provider key");
        }
    }
}

/// The group a setting is shown in: its config table.
fn group_of(key: &str) -> String {
    key.split(['.', '[']).next().unwrap_or_default().to_string()
}

/// The protocol's level as the runtime's; `Err` for one this daemon does
/// not know.
fn owner_level(level: aidaemon_client_protocol::PermissionLevel) -> Result<OwnerLevel, String> {
    use aidaemon_client_protocol::PermissionLevel as Wire;
    match level {
        Wire::ReadOnly => Ok(OwnerLevel::ReadOnly),
        Wire::Standard => Ok(OwnerLevel::Full),
        Wire::FullAccess => Ok(OwnerLevel::FullAccess),
        _ => Err("This permission level is not known to this daemon".to_string()),
    }
}

/// The runtime's level as the protocol's. `unrestricted` (written by hand)
/// runs every command unasked, so it shows as Full access.
fn wire_level(level: OwnerLevel) -> aidaemon_client_protocol::PermissionLevel {
    use aidaemon_client_protocol::PermissionLevel as Wire;
    match level {
        OwnerLevel::ReadOnly => Wire::ReadOnly,
        OwnerLevel::Full => Wire::Standard,
        OwnerLevel::FullAccess | OwnerLevel::Unrestricted => Wire::FullAccess,
    }
}

/// The protocol's mode as the runtime's; `Err` for one this daemon does
/// not know.
fn agent_mode(
    mode: aidaemon_client_protocol::AgentMode,
) -> Result<aidaemon_runtime::tools::agent_mode::PermissionMode, String> {
    use aidaemon_client_protocol::AgentMode as Wire;
    use aidaemon_runtime::tools::agent_mode::PermissionMode;
    match mode {
        Wire::Checked => Ok(PermissionMode::Checked),
        Wire::EveryCommand => Ok(PermissionMode::EveryCommand),
        Wire::FullAccess => Ok(PermissionMode::FullAccess),
        _ => Err("This mode is not known to this daemon".to_string()),
    }
}

/// The coding agents the daemon can hand work to, each with its mode.
/// The agent named `agent` and the model chosen for it (trimmed; `None`:
/// its own default), when it can take one and, if it listed its models
/// when asked, the model is one of them. `Err`: why not, in plain words.
fn agent_model_choice(
    agent: &str,
    model: Option<&str>,
) -> Result<
    (
        std::sync::Arc<dyn aidaemon_runtime::tools::agent_mode::CodingAgent>,
        Option<String>,
    ),
    String,
> {
    use aidaemon_runtime::tools::agent_mode;
    let coding_agent = agent_mode::agent_named(agent)
        .ok_or_else(|| format!("There is no coding agent named `{agent}`"))?;
    if !coding_agent.takes_model() {
        return Err(format!(
            "{} can't be given a model: it works with its own.",
            coding_agent.name()
        ));
    }
    let model = model
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(str::to_string);
    if let Some(model) = model.as_deref()
        && let Some(listed) = agent_mode::models_of(agent)
        && !listed.available.is_empty()
        && !listed.available.iter().any(|offered| offered.id == model)
    {
        return Err(format!(
            "{} doesn't list a model `{model}`. Pick one of the models it offers.",
            coding_agent.name()
        ));
    }
    Ok((coding_agent, model))
}

async fn coding_agents(permissions: &Permissions) -> Vec<aidaemon_client_protocol::CodingAgent> {
    use aidaemon_client_protocol::AgentMode as Wire;
    use aidaemon_runtime::tools::agent_mode::{self, PermissionMode};
    let wire = |mode: PermissionMode| match mode {
        PermissionMode::Checked => Wire::Checked,
        PermissionMode::EveryCommand => Wire::EveryCommand,
        PermissionMode::FullAccess => Wire::FullAccess,
    };
    use aidaemon_client_protocol::AgentReadiness;
    // An agent not asked yet is asked now (the page shows its answer).
    agent_mode::ensure_checked().await;
    let cap = aidaemon_runtime::permissions::agent_cap(permissions.level());
    let mut out = Vec::new();
    for info in agent_mode::agents() {
        let agent = &info.agent;
        // The models it listed when asked (only one that takes a model).
        let listed = agent
            .takes_model()
            .then(|| agent_mode::models_of(&info.name))
            .flatten();
        let (readiness, readiness_reason, readiness_fix) =
            match agent_mode::readiness_of(&info.name) {
                agent_mode::Readiness::Ready => {
                    (AgentReadiness::Ready, String::new(), String::new())
                }
                agent_mode::Readiness::NotUsable { reason, fix } => {
                    (AgentReadiness::NotUsable, reason, fix.unwrap_or_default())
                }
                agent_mode::Readiness::Unchecked => {
                    (AgentReadiness::Unchecked, String::new(), String::new())
                }
            };
        out.push(aidaemon_client_protocol::CodingAgent {
            readiness,
            readiness_reason,
            readiness_fix,
            program: agent.name().to_string(),
            // What it runs at now: the owner's choice under AIdaemon's level.
            mode: permissions.agent_mode(&info.name, &**agent).map(wire),
            same_as_aidaemon: permissions.choice(&info.name) == Some(AgentChoice::Same),
            within_level: agent_mode::modes_within(&**agent, cap)
                .into_iter()
                .map(wire)
                .collect(),
            capped: matches!(
                permissions.choice(&info.name),
                Some(AgentChoice::Mode(mode)) if mode > cap
            ),
            modes: agent.modes().iter().copied().map(wire).collect(),
            note: agent.note().to_string(),
            account: agent.account().to_string(),
            takes_model: agent.takes_model(),
            model: agent
                .takes_model()
                .then(|| permissions.model_of(&info.name))
                .flatten(),
            default_model: listed.as_ref().and_then(|models| models.current.clone()),
            models: listed
                .map(|models| {
                    models
                        .available
                        .into_iter()
                        .map(|model| aidaemon_client_protocol::AgentModelOption {
                            id: model.id,
                            name: model.name,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            name: info.name,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_config_documents_settings_with_their_notes() {
        let catalogue = &*CATALOGUE;
        let backend = catalogue.keys.get("execution.backend").expect("documented");
        assert_eq!(backend.example, "\"local\"");
        assert!(backend.description.contains("docker"), "{backend:?}");
        // A commented-out table's settings count too.
        assert!(catalogue.keys.contains_key("execution.docker.image"));
        assert!(
            catalogue
                .keys
                .get("provider.fallbacks.kind")
                .is_none_or(|d| d.in_list)
        );
        assert_eq!(catalogue.order.first().map(String::as_str), Some("agent"));
    }

    #[test]
    fn notes_end_at_a_hash_outside_strings() {
        assert_eq!(
            entry_of(r##"# color = "#fff" # a colour"##),
            Some(("color".into(), "\"#fff\"".into(), "a colour".into()))
        );
        assert_eq!(
            entry_of("# Set allow_outside_workspace = false to scope"),
            None
        );
    }

    #[test]
    fn list_entries_flatten_by_index_and_read_as_their_documented_key() {
        let table: toml::Table = r#"
[[telegram_bots]]
name = "synthetic"
allowed_user_ids = [1]
"#
        .parse()
        .unwrap();
        let mut out = Vec::new();
        leaves("", &toml::Value::Table(table), &mut out);
        let keys: Vec<&str> = out.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            ["telegram_bots[0].allowed_user_ids", "telegram_bots[0].name"]
        );
        assert_eq!(plain_key("telegram_bots[0].name"), "telegram_bots.name");
    }

    #[test]
    fn only_widening_authority_needs_another_device() {
        let current: toml::Table = "[authority]\ntrusted_roots = [\"/a\"]\n".parse().unwrap();
        let widens = |key, value: Option<&str>| widens_authority(key, value, &current);
        assert_eq!(
            widens("authority.owner_default", Some("\"unrestricted\"")),
            Some(true)
        );
        assert_eq!(
            widens("authority.owner_default", Some("\"full\"")),
            Some(false)
        );
        assert_eq!(widens("authority.owner_default", None), Some(false));
        assert_eq!(
            widens("authority.trusted_roots", Some("[\"/a\", \"/b\"]")),
            Some(true)
        );
        assert_eq!(widens("authority.trusted_roots", Some("[]")), Some(false));
        assert_eq!(widens("authority", None), Some(false));
        assert_eq!(widens("memory.retention_days", Some("3")), None);
    }

    #[test]
    fn the_gateways_address_is_tried_and_providers_reload() {
        assert!(on_trial("client_gateway.port"));
        assert!(on_trial("exposure.cloudflare.hostname"));
        assert!(!on_trial("client_gateway.retention_days"));
        assert_eq!(applies_of("provider.models.default"), Applies::Reload);
        assert_eq!(applies_of("memory.retention_days"), Applies::Restart);
        assert!(guarded("provider.base_url"));
        assert!(guarded("mcp.files.command"));
        assert!(!guarded("memory.retention_days"));
    }

    async fn host(config: &str) -> (tempfile::TempDir, SettingsHost) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, config).unwrap();
        let db = dir.path().join("state.db");
        let state = Arc::new(
            crate::state::SqliteStateStore::for_tests(
                db.to_str().unwrap(),
                100,
                None,
                Arc::new(crate::memory::embeddings::EmbeddingService::deterministic()),
            )
            .await
            .unwrap(),
        );
        let host = SettingsHost::new(path, state, None, None);
        (dir, host)
    }

    /// Each coding agent is listed with its mode; checked is the default
    /// for one the daemon can host, full access asks first, and the choice
    /// is kept per agent.
    #[tokio::test]
    async fn a_coding_agents_mode_is_listed_set_and_full_access_asks() {
        use aidaemon_client_protocol::{AgentMode, CodingAgent};
        use aidaemon_runtime::tools::agent_mode::{self, AgentInfo};
        let registry = agent_mode::CodingAgents::builtin();
        agent_mode::publish(Arc::new(move || {
            vec![
                AgentInfo {
                    name: "synthetic-codex".into(),
                    agent: registry.for_program("codex"),
                    command: "/synthetic/bin/codex".into(),
                    prefix: Vec::new(),
                    // Not asked: it does not run here.
                    local: false,
                },
                AgentInfo {
                    name: "synthetic-coder".into(),
                    agent: registry.for_program("synthetic-coder"),
                    command: "/synthetic/bin/synthetic-coder".into(),
                    prefix: Vec::new(),
                    local: false,
                },
            ]
        }));
        let (_dir, host) = host(CONFIG).await;
        let agents = coding_agents(&host.permissions).await;
        let codex: &CodingAgent = agents.iter().find(|a| a.name == "synthetic-codex").unwrap();
        assert_eq!(codex.program, "Codex");
        assert_eq!(codex.mode, Some(AgentMode::Checked));
        assert_eq!(
            codex.readiness,
            aidaemon_client_protocol::AgentReadiness::Unchecked
        );
        assert_eq!(codex.modes.len(), 3);
        let other = agents.iter().find(|a| a.name == "synthetic-coder").unwrap();
        assert_eq!(other.mode, None);
        assert_eq!(other.modes, vec![AgentMode::FullAccess]);
        assert!(!other.note.is_empty());

        // Standard: full access is above AIdaemon's own level, refused with
        // why, and not offered.
        assert_eq!(
            codex.within_level,
            vec![AgentMode::EveryCommand, AgentMode::Checked]
        );
        assert!(other.within_level.is_empty());
        let full = SettingsChange::SetAgentMode {
            agent: "synthetic-codex".into(),
            mode: AgentMode::FullAccess,
        };
        assert!(host.gate(&full).await.unwrap_err().contains("Standard"));
        // Raising AIdaemon to full access asks (on another device), then
        // offers it, and full access for the agent asks too.
        let raise = SettingsChange::SetPermissionLevel {
            level: aidaemon_client_protocol::PermissionLevel::FullAccess,
        };
        assert!(matches!(
            host.gate(&raise).await.unwrap(),
            SettingsGate::Ask {
                other_device: true,
                ..
            }
        ));
        host.change(raise, true).await.unwrap();
        assert_eq!(
            host.settings().await.unwrap().permission_level,
            Some(aidaemon_client_protocol::PermissionLevel::FullAccess)
        );
        assert!(matches!(
            host.gate(&full).await.unwrap(),
            SettingsGate::Ask { .. }
        ));
        host.change(full, true).await.unwrap();
        let same = SettingsChange::AgentSameAsAidaemon {
            agent: "synthetic-coder".into(),
        };
        assert!(matches!(
            host.gate(&same).await.unwrap(),
            SettingsGate::Ask { .. }
        ));
        host.change(same, true).await.unwrap();
        let agents = coding_agents(&host.permissions).await;
        let other = agents.iter().find(|a| a.name == "synthetic-coder").unwrap();
        assert!(other.same_as_aidaemon);
        assert_eq!(other.mode, Some(AgentMode::FullAccess));
        // Back to Standard: free; the agent picked above it is capped and
        // says so, the one following the level runs nothing.
        let lower = SettingsChange::SetPermissionLevel {
            level: aidaemon_client_protocol::PermissionLevel::Standard,
        };
        assert_eq!(host.gate(&lower).await.unwrap(), SettingsGate::Free);
        host.change(lower, false).await.unwrap();
        let agents = coding_agents(&host.permissions).await;
        let codex = agents.iter().find(|a| a.name == "synthetic-codex").unwrap();
        assert_eq!(codex.mode, Some(AgentMode::Checked));
        assert!(codex.capped);
        let other = agents.iter().find(|a| a.name == "synthetic-coder").unwrap();
        assert_eq!(other.mode, None);
        // All of it is in the file.
        let text = std::fs::read_to_string(&host.config_path).unwrap();
        assert!(text.contains("owner_default = \"full\""), "{text}");
        assert!(text.contains("synthetic-codex = \"full_access\""), "{text}");
        assert!(text.contains("synthetic-coder = \"same\""), "{text}");
        let every = SettingsChange::SetAgentMode {
            agent: "synthetic-codex".into(),
            mode: AgentMode::EveryCommand,
        };
        assert_eq!(host.gate(&every).await.unwrap(), SettingsGate::Free);
        host.change(every, false).await.unwrap();
        let agents = coding_agents(&host.permissions).await;
        let codex = agents.iter().find(|a| a.name == "synthetic-codex").unwrap();
        assert_eq!(codex.mode, Some(AgentMode::EveryCommand));
        // One whose run could not sign in shows why, until asked again.
        agent_mode::mark_not_usable(
            "synthetic-codex",
            "It couldn't sign in on its last run.",
            Some("Sign in again."),
        );
        let agents = coding_agents(&host.permissions).await;
        let codex = agents.iter().find(|a| a.name == "synthetic-codex").unwrap();
        assert_eq!(
            codex.readiness,
            aidaemon_client_protocol::AgentReadiness::NotUsable
        );
        assert_eq!(codex.readiness_fix, "Sign in again.");
        assert_eq!(
            host.gate(&SettingsChange::CheckAgents).await.unwrap(),
            SettingsGate::Free
        );
        // A mode its kind does not offer is refused.
        let refused = SettingsChange::SetAgentMode {
            agent: "synthetic-coder".into(),
            mode: AgentMode::Checked,
        };
        assert!(host.change(refused, false).await.is_err());
    }

    const CONFIG: &str = r#"
[provider]
kind = "openai_compatible"
api_key = "keychain"
base_url = "https://h.example.invalid/v1"

[provider.models]
default = "synthetic-model-1"
"#;

    fn set(key: &str, value: Option<&str>) -> SettingsChange {
        SettingsChange::Set {
            key: key.into(),
            value: value.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn settings_show_the_file_and_the_documented_defaults_with_secrets_hidden() {
        let (_dir, host) = host(CONFIG).await;
        let settings = host.settings().await.unwrap();
        let all: Vec<&Setting> = settings.groups.iter().flat_map(|g| &g.settings).collect();
        let key = all.iter().find(|s| s.key == "provider.api_key").unwrap();
        assert_eq!(key.kind, SettingKind::Secret);
        assert_eq!(key.value, None);
        assert!(key.secret_set && key.guarded);
        let model = all
            .iter()
            .find(|s| s.key == "provider.models.default")
            .unwrap();
        assert_eq!(model.value.as_deref(), Some("\"synthetic-model-1\""));
        assert_eq!(model.applies, Applies::Reload);
        // Documented in the example, left to its default here.
        let backend = all.iter().find(|s| s.key == "execution.backend").unwrap();
        assert_eq!(backend.value, None);
        assert_eq!(backend.example.as_deref(), Some("\"local\""));
        assert_eq!(settings.groups[0].name, "agent");
        assert!(!settings.restart_pending);
    }

    #[tokio::test]
    async fn a_plain_change_is_saved_through_manage_config_and_waits_for_a_restart() {
        let (_dir, host) = host(CONFIG).await;
        let change = set("checkpoints.enabled", Some("true"));
        assert_eq!(host.gate(&change).await.unwrap(), SettingsGate::Free);
        let done = host.change(change, false).await.unwrap();
        assert_eq!(done.applies, Some(Applies::Restart));
        let text = std::fs::read_to_string(&host.config_path).unwrap();
        assert!(text.contains("enabled = true"), "{text}");
        assert!(host.settings().await.unwrap().restart_pending);
        // A key the daemon does not read is refused, as in chat.
        let refused = host
            .change(set("checkpoints.no_such_setting", Some("1")), false)
            .await
            .unwrap_err();
        assert!(refused.contains("does not read"), "{refused}");
    }

    #[tokio::test]
    async fn guarded_changes_ask_and_widening_authority_needs_another_device() {
        let (_dir, host) = host(CONFIG).await;
        let gate = host
            .gate(&set(
                "provider.base_url",
                Some("\"https://other.example.invalid\""),
            ))
            .await
            .unwrap();
        assert!(matches!(
            gate,
            SettingsGate::Ask {
                other_device: false,
                ..
            }
        ));
        // Unapproved, the tool itself refuses it.
        assert!(
            host.change(
                set(
                    "provider.base_url",
                    Some("\"https://other.example.invalid\"")
                ),
                false
            )
            .await
            .is_err()
        );
        let raise = host
            .gate(&set("authority.owner_default", Some("\"unrestricted\"")))
            .await
            .unwrap();
        assert!(matches!(
            raise,
            SettingsGate::Ask {
                other_device: true,
                ..
            }
        ));
        let lower = host
            .gate(&set("authority.owner_default", Some("\"full\"")))
            .await
            .unwrap();
        assert_eq!(lower, SettingsGate::Free);
        let mcp = host
            .gate(&set("mcp.files.command", Some("\"x\"")))
            .await
            .unwrap();
        assert!(matches!(mcp, SettingsGate::Ask { .. }));
    }

    #[tokio::test]
    async fn the_database_moves_only_by_its_own_action() {
        let (_dir, host) = host(CONFIG).await;
        assert!(
            host.gate(&set("state.db_path", Some("\"/x.db\"")))
                .await
                .is_err()
        );
        assert!(host.gate(&set("state.encryption_key", None)).await.is_err());
        let change = SettingsChange::Database {
            action: DatabaseAction::MoveTo {
                path: "/data/synthetic.db".into(),
            },
        };
        assert!(matches!(
            host.gate(&change).await.unwrap(),
            SettingsGate::Ask { .. }
        ));
        host.change(change, true).await.unwrap();
        let settings = host.settings().await.unwrap();
        assert!(settings.notices[0].contains("/data/synthetic.db"));
        let cancel = SettingsChange::Database {
            action: DatabaseAction::Cancel,
        };
        assert_eq!(host.gate(&cancel).await.unwrap(), SettingsGate::Free);
        host.change(cancel, false).await.unwrap();
        assert!(host.settings().await.unwrap().notices.is_empty());
    }

    #[tokio::test]
    async fn an_address_change_goes_on_trial_and_a_refused_one_does_not() {
        let (_dir, host) = host(CONFIG).await;
        // Refused by the config check (not a port): no trial is left behind.
        assert!(
            host.change(set("client_gateway.port", Some("\"high\"")), true)
                .await
                .is_err()
        );
        assert!(!settings_upkeep::trial_active(&host.config_path));
        host.change(set("client_gateway.port", Some("8799")), true)
            .await
            .unwrap();
        assert!(settings_upkeep::trial_active(&host.config_path));
    }

    #[tokio::test]
    async fn a_failover_change_from_an_out_of_date_page_is_refused() {
        let _keychain = crate::testing::KeychainIsolation::acquire();
        let (_dir, host) = host(
            r#"
[provider]
kind = "anthropic"
api_key = "keychain"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
api_key = "keychain"

[[provider.fallbacks]]
kind = "anthropic"
api_key = "keychain"
"#,
        )
        .await;
        let shown = host.settings().await.unwrap().connected;
        let (first, second) = (shown[1].name.clone(), shown[2].name.clone());
        assert_ne!(first, second);
        // What the page shows at failover 1 is there: the move goes ahead.
        let up = SettingsChange::MoveFailover {
            index: 1,
            to: 0,
            expect: second.clone(),
        };
        assert_eq!(host.gate(&up).await.unwrap(), SettingsGate::Free);
        // The page shows another provider there: refused, for either change.
        let stale = SettingsChange::MoveFailover {
            index: 1,
            to: 0,
            expect: first.clone(),
        };
        assert_eq!(host.gate(&stale).await.unwrap_err(), STALE_FAILOVERS);
        let remove = SettingsChange::RemoveFailover {
            index: 0,
            expect: second.clone(),
        };
        assert_eq!(host.gate(&remove).await.unwrap_err(), STALE_FAILOVERS);
        // Said with the names the page shows.
        assert_eq!(
            host.failover_words(&up).unwrap(),
            format!("{second} is now Failover 1 after a reload.")
        );
        assert_eq!(
            host.failover_words(&SettingsChange::RemoveFailover {
                index: 0,
                expect: String::new(),
            })
            .unwrap(),
            format!("Removed {first}; it stops being tried after a reload.")
        );
    }

    #[tokio::test]
    async fn a_new_main_provider_never_inherits_another_provider_s_key() {
        let _keychain = crate::testing::KeychainIsolation::acquire();
        let (_dir, host) = host(
            r#"
[provider]
kind = "anthropic"
api_key = "keychain"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://openrouter.ai/api/v1"
api_key = "keychain"
"#,
        )
        .await;
        crate::config::store_in_keychain("api_key", "synthetic-main-key").unwrap();
        crate::config::store_in_keychain("provider_fallback_0_api_key", "synthetic-or-key")
            .unwrap();
        let switch = |preset: &str| SettingsChange::SwitchProvider {
            provider: ProviderChoice {
                preset: preset.into(),
                ..ProviderChoice::default()
            },
            key: None,
        };
        // Another provider with no key of its own: asked for, not the
        // main slot's key.
        let refused = host.gate(&switch("openai")).await.unwrap_err();
        assert!(refused.contains("needs its API key"), "{refused}");
        // A failover's provider brings the failover's key.
        let borrowed = host.key_change(&switch("openrouter")).unwrap().unwrap();
        assert_eq!(borrowed.slot, "api_key");
        assert_eq!(borrowed.value, "synthetic-or-key");
        assert!(
            borrowed.summary.contains("failover 1"),
            "{}",
            borrowed.summary
        );
        // The same provider keeps its own.
        assert!(
            host.key_change(&switch("anthropic_native"))
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_secret_for_a_setting_the_keychain_does_not_hold_is_refused() {
        let (_dir, host) = host(CONFIG).await;
        let change = SettingsChange::Secret {
            key: "provider.models.default".into(),
            value: aidaemon_client_protocol::SecretText("synthetic-secret".into()),
        };
        let refused = host.gate(&change).await.unwrap_err();
        assert!(!refused.contains("synthetic-secret"));
        assert!(
            host.change(SettingsChange::RemoveBot { bot_id: 99 }, false)
                .await
                .is_err()
        );
    }
}
