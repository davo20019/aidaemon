//! `/trustall`: the owner's switch for terminal trust-all mode (every
//! command auto-approved). It is an owner command, not a tool action, so no
//! model call can turn it on.

use std::sync::{Arc, OnceLock};

use aidaemon_channels::slash::{CommandCategory, CommandDef, SlashCall, SlashCommand};
use async_trait::async_trait;

use crate::tools::TerminalTool;
use crate::types::UserRole;

/// The running terminal tool, published at startup.
static TERMINAL: OnceLock<Arc<TerminalTool>> = OnceLock::new();

pub(crate) fn publish(terminal: Arc<TerminalTool>) {
    let _ = TERMINAL.set(terminal);
}

pub(crate) struct TrustAllCommand;

#[async_trait]
impl SlashCommand for TrustAllCommand {
    fn handles(&self, cmd: &str) -> bool {
        cmd == "/trustall"
    }

    fn commands(&self) -> Vec<CommandDef> {
        vec![CommandDef {
            name: "trustall",
            description: "Run every terminal command without asking (on/off)",
            usage: Some("/trustall on|off"),
            category: CommandCategory::Core,
        }]
    }

    async fn run(&self, _cmd: &str, args: &str, call: &SlashCall<'_>) -> String {
        if call.user_role != UserRole::Owner {
            return "Only the owner can change terminal trust-all mode.".to_string();
        }
        let Some(terminal) = TERMINAL.get() else {
            return "The terminal tool is not enabled.".to_string();
        };
        run_with(terminal, args).await
    }
}

async fn run_with(terminal: &TerminalTool, args: &str) -> String {
    let on = match args.trim().to_ascii_lowercase().as_str() {
        "on" => true,
        "off" => false,
        _ => {
            let state = if terminal.trust_all_enabled().await {
                "on: every terminal command runs without asking"
            } else {
                "off: risky terminal commands ask you first"
            };
            return format!("Trust-all mode is {state}. Use /trustall on or /trustall off.");
        }
    };
    match terminal.set_trust_all(on).await {
        Ok(()) if on => "Trust-all mode is on: every terminal command, including dangerous \
                         ones (rm, sudo), now runs without asking, until /trustall off."
            .to_string(),
        Ok(()) => "Trust-all mode is off: risky terminal commands ask you first.".to_string(),
        Err(error) => format!("Trust-all mode was not changed: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The owner's command turns trust-all on and off, persisted for the
    /// terminal's backend, and reports the state with no argument.
    #[tokio::test]
    async fn the_owner_switches_trust_all_on_and_off() {
        let pool_file = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", pool_file.path().display()))
            .await
            .unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let terminal = TerminalTool::new(
            crate::execution::active_execution_backend(),
            Vec::new(),
            crate::tools::ApprovalBroker::new(tx),
            1,
            1000,
            crate::types::PermissionMode::Default,
            pool.clone(),
        )
        .await;
        let persisted = || async {
            sqlx::query_scalar::<_, String>(
                "SELECT prefix FROM terminal_backend_allowed_prefixes WHERE prefix = '*'",
            )
            .fetch_all(&pool)
            .await
            .unwrap()
        };

        assert!(run_with(&terminal, "").await.contains("is off"));
        assert!(run_with(&terminal, "on").await.contains("is on"));
        assert!(terminal.trust_all_enabled().await);
        assert_eq!(persisted().await.len(), 1);
        assert!(run_with(&terminal, "off").await.contains("is off"));
        assert!(!terminal.trust_all_enabled().await);
        assert!(persisted().await.is_empty());
    }
}
