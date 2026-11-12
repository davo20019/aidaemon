//! Owner commands an application can add to the shared ones
//! ([`crate::slash::SharedCommands::extensions`]): `/reload` over the
//! host's config, and `/checkpoints` and `/rollback` over the filesystem
//! checkpoints of `aidaemon_tools` (feature `checkpoints`).

use std::sync::Arc;

#[cfg(feature = "checkpoints")]
use aidaemon_tools::semantics::UserRole;
use async_trait::async_trait;

use crate::slash::{SlashCall, SlashCommand};

/// Reloads the application's config.
#[async_trait]
pub trait ConfigReload: Send + Sync {
    /// `Ok` says what changed; `Err` says what went wrong (and what was
    /// restored).
    async fn reload_config(&self) -> Result<String, String>;
}

/// `/reload`: reload the host's config.
pub struct ReloadCommand {
    reload: Arc<dyn ConfigReload>,
}

impl ReloadCommand {
    pub fn new(reload: Arc<dyn ConfigReload>) -> Self {
        Self { reload }
    }
}

#[async_trait]
impl SlashCommand for ReloadCommand {
    fn handles(&self, cmd: &str) -> bool {
        cmd == "/reload"
    }

    async fn run(&self, _cmd: &str, _args: &str, _call: &SlashCall<'_>) -> String {
        match self.reload.reload_config().await {
            Ok(summary) => format!("Config reloaded. {summary}"),
            Err(error) => error,
        }
    }
}

/// `/checkpoints` and `/rollback`: the filesystem checkpoints.
#[cfg(feature = "checkpoints")]
pub struct CheckpointCommands;

#[cfg(feature = "checkpoints")]
#[async_trait]
impl SlashCommand for CheckpointCommands {
    fn handles(&self, cmd: &str) -> bool {
        matches!(cmd, "/checkpoints" | "/rollback")
    }

    async fn run(&self, cmd: &str, args: &str, call: &SlashCall<'_>) -> String {
        if cmd == "/checkpoints" {
            self.handle_checkpoints(call.user_role).await
        } else {
            self.handle_rollback(args, call.session_id, call.user_role)
                .await
        }
    }
}

#[cfg(feature = "checkpoints")]
impl CheckpointCommands {
    async fn handle_checkpoints(&self, user_role: UserRole) -> String {
        if user_role != UserRole::Owner {
            return "Only the owner can inspect filesystem checkpoints.".to_string();
        }
        match aidaemon_tools::checkpoints::active_manager() {
            Some(manager) => manager.list_text(30, None).await,
            None => "Filesystem checkpoints are disabled.".to_string(),
        }
    }

    async fn handle_rollback(&self, args: &str, session_id: &str, user_role: UserRole) -> String {
        if user_role != UserRole::Owner {
            return "Only the owner can roll back filesystem checkpoints.".to_string();
        }
        let Some(manager) = aidaemon_tools::checkpoints::active_manager() else {
            return "Filesystem checkpoints are disabled.".to_string();
        };
        let mut parts = args.split_whitespace();
        let first = parts.next();
        if first.is_some_and(|value| value.eq_ignore_ascii_case("confirm")) {
            let Some(token) = parts.next() else {
                return "Usage: /rollback confirm <token>".to_string();
            };
            if parts.next().is_some() {
                return "Usage: /rollback confirm <token>".to_string();
            }
            return match manager.apply_rollback(session_id, token).await {
                Ok(result) => result.render(),
                Err(error) => format!("Rollback failed: {error}"),
            };
        }
        if parts.next().is_some() {
            return "Usage: /rollback [checkpoint-id]".to_string();
        }
        match manager.prepare_rollback(session_id, first).await {
            Ok(preview) => preview.render(),
            Err(error) => format!("Could not prepare rollback: {error}"),
        }
    }
}
