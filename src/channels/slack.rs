//! The daemon's side of the Slack bot ([`SlackChannel`],
//! `aidaemon_channel_slack::bot`): the access lists and workspace grants in
//! its config file, project roots on its execution backend, the shared
//! slash commands on its stores, and `!restart`.

use std::sync::Arc;

pub use aidaemon_channel_slack::bot::{SlackBot as SlackChannel, spawn_slack_channel};
use aidaemon_channel_slack::bot::{SlackBotSettings, SlackHost, WorkspaceGrantMutation};
use async_trait::async_trait;
use tracing::info;

use crate::channels::ChannelRuntimeDeps;
use crate::types::{UserRole, WorkspaceGrant};

/// A Slack bot on the daemon's services.
pub(crate) fn new_slack_bot(
    app_token: &str,
    bot_token: &str,
    allowed_user_ids: Vec<String>,
    owner_user_ids: Vec<String>,
    workspace_grants: Vec<WorkspaceGrant>,
    use_threads: bool,
    runtime: ChannelRuntimeDeps,
) -> SlackChannel {
    SlackChannel::new(
        SlackBotSettings {
            app_token: app_token.to_string(),
            bot_token: bot_token.to_string(),
            allowed_user_ids,
            owner_user_ids,
            workspace_grants,
            use_threads,
        },
        runtime.bot_deps(),
        Arc::new(DaemonSlackHost {
            runtime,
            bot_token: bot_token.to_string(),
        }),
    )
}

async fn load_workspace_grants_from_config(
    config_path: &std::path::Path,
) -> anyhow::Result<Vec<WorkspaceGrant>> {
    let _read_guard = super::claim_config::CONFIG_WRITE_LOCK.lock().await;
    let content = tokio::fs::read_to_string(config_path).await?;
    let doc: toml::Table = content.parse()?;
    let grants = doc
        .get("users")
        .and_then(toml::Value::as_table)
        .and_then(|users| users.get("workspace_grants"))
        .cloned()
        .map(|value| value.try_into())
        .transpose()?
        .unwrap_or_default();
    Ok(grants)
}

/// The daemon's services behind one Slack bot.
struct DaemonSlackHost {
    runtime: ChannelRuntimeDeps,
    /// Which bot this is, to find its record when it was connected at
    /// runtime.
    bot_token: String,
}

#[async_trait]
impl aidaemon_channels::bot::SharedCommandDispatch for DaemonSlackHost {
    async fn dispatch(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String> {
        self.runtime
            .command_context()
            .dispatch(cmd, args, session_id, sender_id, user_role)
            .await
    }
}

#[async_trait]
impl SlackHost for DaemonSlackHost {
    fn restart(&self) {
        restart_process();
    }

    /// Persist Slack allowlist and owner IDs together so auto-claim cannot leave
    /// a user authorized but accidentally downgraded (or vice versa).
    async fn persist_access_ids(
        &self,
        allowed_ids: &[String],
        owner_ids: &[String],
    ) -> anyhow::Result<()> {
        // A bot connected at runtime lives in the database: its claim is
        // saved on its own row (with no owner ids, its allowlist holds its
        // owners), never in the config, where it would be lost or would
        // overwrite the configured bot's access.
        let dynamic = self
            .runtime
            .state
            .get_dynamic_bots()
            .await
            .unwrap_or_default();
        if dynamic
            .iter()
            .any(|bot| bot.channel_type == "slack" && bot.bot_token == self.bot_token)
        {
            self.runtime
                .state
                .update_dynamic_bot_allowed_users(&self.bot_token, allowed_ids)
                .await?;
            info!("Persisted Slack access IDs to the connected bot's record");
            return Ok(());
        }
        let ids_toml = toml::Value::Array(
            allowed_ids
                .iter()
                .map(|id| toml::Value::String(id.clone()))
                .collect(),
        );
        super::claim_config::save_allowed_ids(
            &self.runtime.config_path,
            "slack",
            "slack_bots",
            &self.bot_token,
            ids_toml,
            |doc| {
                let users = doc
                    .entry("users".to_string())
                    .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                    .as_table_mut()
                    .ok_or_else(|| anyhow::anyhow!("[users] must be a TOML table"))?;
                let owner_table = users
                    .entry("owner_ids".to_string())
                    .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                    .as_table_mut()
                    .ok_or_else(|| anyhow::anyhow!("[users.owner_ids] must be a TOML table"))?;
                owner_table.insert(
                    "slack".to_string(),
                    toml::Value::Array(
                        owner_ids
                            .iter()
                            .map(|id| toml::Value::String(id.clone()))
                            .collect(),
                    ),
                );
                Ok(())
            },
        )
        .await?;
        info!("Persisted Slack access IDs to config.toml");
        Ok(())
    }

    /// Apply one grant mutation against the latest on-disk configuration. Each
    /// Slack bot keeps a local cache for message routing, but config writes must
    /// merge from disk so one bot cannot erase another bot's grants with a
    /// stale snapshot.
    async fn mutate_workspace_grants(
        &self,
        mutation: WorkspaceGrantMutation,
    ) -> anyhow::Result<(Vec<WorkspaceGrant>, bool)> {
        let _write_guard = super::claim_config::CONFIG_WRITE_LOCK.lock().await;
        let content = tokio::fs::read_to_string(&self.runtime.config_path).await?;
        let mut doc: toml::Table = content.parse()?;
        let users = doc
            .entry("users".to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()))
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("[users] must be a TOML table"))?;

        let mut grants: Vec<WorkspaceGrant> = users
            .get("workspace_grants")
            .cloned()
            .map(|value| value.try_into())
            .transpose()?
            .unwrap_or_default();
        grants.retain(WorkspaceGrant::is_active);

        let affected = match mutation {
            WorkspaceGrantMutation::Upsert(grant) => {
                grants.retain(|existing| {
                    !(existing.platform == grant.platform
                        && existing.workspace_id == grant.workspace_id
                        && existing.channel_id == grant.channel_id
                        && existing.user_id == grant.user_id)
                });
                grants.push(grant);
                true
            }
            WorkspaceGrantMutation::Revoke {
                platform,
                workspace_id,
                channel_id,
                user_id,
            } => {
                let before = grants.len();
                grants.retain(|grant| {
                    !(grant.platform == platform
                        && grant.workspace_id == workspace_id
                        && grant.channel_id == channel_id
                        && grant.user_id == user_id)
                });
                grants.len() != before
            }
        };

        users.insert(
            "workspace_grants".to_string(),
            toml::Value::try_from(&grants)?,
        );

        crate::config::config_file(&self.runtime.config_path)
            .save_table(&doc)
            .await?;
        info!(count = grants.len(), "Persisted Slack workspace grants");
        Ok((grants, affected))
    }

    async fn load_workspace_grants(&self) -> anyhow::Result<Vec<WorkspaceGrant>> {
        load_workspace_grants_from_config(&self.runtime.config_path).await
    }

    async fn resolve_workspace_project_root(&self, raw_path: &str) -> anyhow::Result<String> {
        let backend = self.runtime.agent.execution_backend();
        anyhow::ensure!(
            backend.kind() == crate::execution::BackendKind::Local,
            "Workspace collaboration is currently available only with the local execution backend"
        );

        let resolved = backend.resolve_path(raw_path).await?;
        let metadata = backend
            .metadata(&resolved)
            .await
            .map_err(|_| anyhow::anyhow!("Project directory does not exist"))?;
        anyhow::ensure!(metadata.is_dir(), "The project path must be a directory");

        let canonical = backend.canonicalize(&resolved).await?;
        let project_root = std::path::PathBuf::from(canonical.as_str());
        anyhow::ensure!(
            project_root.parent().is_some(),
            "The filesystem root cannot be delegated"
        );
        if let Ok(home) = backend.canonicalize(backend.home_hint()).await {
            anyhow::ensure!(
                canonical != home,
                "The home directory cannot be delegated; choose one project"
            );
        }
        anyhow::ensure!(
            !self.runtime.agent.is_sensitive_path(&project_root),
            "A sensitive directory cannot be delegated"
        );
        anyhow::ensure!(
            crate::tools::fs_utils::find_nearest_project_root(&project_root)
                .is_some_and(|root| root == project_root),
            "The path must be a project root (for example, a Git, Cargo, Node, Python, Go, or similar project)"
        );

        Ok(canonical.as_str().to_string())
    }
}

/// Replace the current process with a fresh instance of itself.
fn restart_process() {
    crate::startup::settings_upkeep::restart_in_place();
}

#[cfg(test)]
#[path = "slack_tests.rs"]
mod turn_tests;
