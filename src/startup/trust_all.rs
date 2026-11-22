//! `/trustall`: the owner's switch for AIdaemon's Full access level (every
//! terminal command runs unasked). It sets the same `[authority]
//! owner_default` the Settings page does (`startup::permissions`), so the
//! two can never disagree. It is an owner command, not a tool action, so no
//! model call can turn it on.
//!
//! Lowering is immediate. Raising takes the same check as Settings, so no
//! one device or chat raises the level alone: from an app the gateway
//! routes `/trustall on` into Settings' change (another device approves);
//! from a chat the owner's paired devices approve it. With no paired
//! device to ask, the chat confirms it itself (`/trustall on confirm`), as
//! a lone device confirms its own card in Settings.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use aidaemon_channels::slash::{CommandCategory, CommandDef, SlashCall, SlashCommand};
use aidaemon_runtime::permissions::{OwnerLevel, Permissions, save_level};
use aidaemon_tools::config_file::ConfigFile;
use async_trait::async_trait;

use crate::types::UserRole;

/// How long a chat's own confirmation stays open.
const CONFIRM_WITHIN: Duration = Duration::from_secs(120);

/// Chats asked to confirm Full access themselves, and when.
static ASKED: LazyLock<Mutex<HashMap<String, Instant>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

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
            return "Only the owner can change AIdaemon's permission level.".to_string();
        }
        let Some((permissions, config)) = crate::startup::permissions::daemon() else {
            return "AIdaemon's permission level is not available yet.".to_string();
        };
        // Exact command words.
        match args.trim().to_ascii_lowercase().as_str() {
            "on" if permissions.level() < OwnerLevel::FullAccess => {
                ask_to_raise(call.session_id).await
            }
            "on confirm" => {
                if !take_confirmation(call.session_id, Instant::now()) {
                    return "Nothing to confirm: send /trustall on first.".to_string();
                }
                run_with(permissions, config, "on").await
            }
            other => run_with(permissions, config, other).await,
        }
    }
}

/// Raising the level from a chat: the owner's paired devices approve it;
/// with none, the chat confirms it itself.
async fn ask_to_raise(session_id: &str) -> String {
    // An app's `/trustall on` is Settings' change (the gateway routes it);
    // one reaching here from an app confirms nothing itself.
    if session_id.starts_with(aidaemon_client_gateway::APP_SESSION_PREFIX) {
        return "Raise AIdaemon's permission level in Settings → AIdaemon permissions.".to_string();
    }
    let change = aidaemon_client_protocol::SettingsChange::SetPermissionLevel {
        level: aidaemon_client_protocol::PermissionLevel::FullAccess,
    };
    if let Some(gateway) = crate::startup::client_gateway::running()
        && let Some(asked) = gateway.ask_devices_for_chat(change, session_id).await
    {
        return asked.unwrap_or_else(|reason| format!("Trust-all was not changed: {reason}"));
    }
    ask_confirmation(session_id, Instant::now());
    format!(
        "Full access runs every terminal command, including dangerous ones (rm, sudo), without \
         asking. No paired device can approve it, so this chat confirms it: send /trustall on \
         confirm within {} minutes.",
        CONFIRM_WITHIN.as_secs() / 60
    )
}

fn ask_confirmation(session_id: &str, now: Instant) {
    let mut asked = ASKED.lock().unwrap_or_else(|p| p.into_inner());
    asked.retain(|_, at| now.duration_since(*at) < CONFIRM_WITHIN);
    asked.insert(session_id.to_string(), now);
}

/// Whether `session_id` was asked to confirm, within the time; the ask is
/// used up either way.
fn take_confirmation(session_id: &str, now: Instant) -> bool {
    let mut asked = ASKED.lock().unwrap_or_else(|p| p.into_inner());
    asked
        .remove(session_id)
        .is_some_and(|at| now.duration_since(at) < CONFIRM_WITHIN)
}

async fn run_with(permissions: &Permissions, config: &ConfigFile, args: &str) -> String {
    let level = permissions.level();
    let target = match args.trim().to_ascii_lowercase().as_str() {
        "on" => OwnerLevel::FullAccess,
        // Off from Full access is Standard; a level below it stays.
        "off" if level >= OwnerLevel::FullAccess => OwnerLevel::Full,
        "off" => level,
        _ => {
            let state = if permissions.trust_all() {
                "on: every terminal command runs without asking"
            } else {
                "off: risky terminal commands ask you first"
            };
            return format!(
                "Trust-all is {state} (AIdaemon's permission level: {}). Use /trustall on or \
                 /trustall off, or Settings → AIdaemon permissions.",
                level.title()
            );
        }
    };
    if target != level
        && let Err(error) = save_level(config, permissions, target).await
    {
        return format!("Trust-all was not changed: {error}");
    }
    if target == OwnerLevel::FullAccess {
        "Trust-all is on (AIdaemon's permission level: Full access): every terminal command, \
         including dangerous ones (rm, sudo), now runs without asking, until /trustall off."
            .to_string()
    } else {
        format!(
            "Trust-all is off (AIdaemon's permission level: {}): risky terminal commands ask \
             you first.",
            target.title()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The owner's command sets AIdaemon's level in config.toml, reports it
    /// with no argument, and a restart reads what it left.
    #[tokio::test]
    async fn the_owner_switches_trust_all_on_and_off() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, concat!("# owner's notes\n", "[provider]\nkind = \"openai_compatible\"\napi_key = \"keychain\"\nbase_url = \"https://h.example.invalid/v1\"\n\n[provider.models]\ndefault = \"synthetic-model-1\"\n")).unwrap();
        let config = crate::config::config_file(&path);
        let start = || async { Permissions::from_config(&config.load_table().await.unwrap()) };
        let permissions = start().await;

        assert!(run_with(&permissions, &config, "").await.contains("is off"));
        assert!(
            run_with(&permissions, &config, "on")
                .await
                .contains("is on")
        );
        assert!(permissions.trust_all());
        assert_eq!(start().await.level(), OwnerLevel::FullAccess);
        assert!(
            run_with(&permissions, &config, "off")
                .await
                .contains("is off")
        );
        assert!(!permissions.trust_all());
        // The next start reads it off.
        assert_eq!(start().await.level(), OwnerLevel::Full);
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("# owner's notes")
        );

        // Off from Read only stays Read only.
        save_level(&config, &permissions, OwnerLevel::ReadOnly)
            .await
            .unwrap();
        assert!(
            run_with(&permissions, &config, "off")
                .await
                .contains("Read only")
        );
        assert_eq!(start().await.level(), OwnerLevel::ReadOnly);
    }

    /// A chat's own confirmation is for that chat only, once, and only
    /// while it is fresh.
    #[test]
    fn a_chat_confirms_its_own_ask_once_while_fresh() {
        let now = Instant::now();
        assert!(!take_confirmation("telegram:synthetic-1", now));
        ask_confirmation("telegram:synthetic-1", now);
        assert!(!take_confirmation("telegram:synthetic-2", now));
        assert!(take_confirmation("telegram:synthetic-1", now));
        assert!(!take_confirmation("telegram:synthetic-1", now), "used up");
        ask_confirmation("telegram:synthetic-1", now);
        assert!(!take_confirmation(
            "telegram:synthetic-1",
            now + CONFIRM_WITHIN + Duration::from_secs(1)
        ));
    }

    /// An app's `/trustall on` never confirms itself here: the gateway
    /// routes it into Settings' change.
    #[tokio::test]
    async fn an_app_session_is_sent_to_settings() {
        let reply = ask_to_raise("app:synthetic").await;
        assert!(reply.contains("Settings"), "{reply}");
        let now = Instant::now();
        assert!(!take_confirmation("app:synthetic", now));
    }
}
