//! The daemon's slash commands: the shared ones from
//! [`aidaemon_channels::slash`] on the daemon's stores and agent, plus
//! `/work` (`aidaemon_work`), `/checkpoints` and `/rollback`, and `/reload`
//! on the daemon's config file.

use std::path::PathBuf;
use std::sync::Arc;

use aidaemon_channels::admin_commands::{CheckpointCommands, ConfigReload, ReloadCommand};
use aidaemon_channels::slash::{ChatMode, CommandHost, ScheduledItem, SharedCommands};
pub(crate) use aidaemon_channels::slash::{CommandCategory, CommandDef};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::agent::authority::SessionMode;
use crate::config::AppConfig;
use crate::runtime_ports::DaemonAgentRuntime;
use crate::tasks::TaskRegistry;
use crate::traits::{ConversationSummary, Message, StateStore, TokenUsageRecord};
use crate::types::UserRole;

/// Shared command dispatcher for commands that behave identically across
/// Telegram, Slack, and Discord channels.
pub(crate) struct CommandContext {
    pub agent: Arc<dyn DaemonAgentRuntime>,
    pub state: Arc<dyn StateStore>,
    pub task_registry: Arc<TaskRegistry>,
    pub config_path: PathBuf,
}

#[async_trait]
impl aidaemon_channels::bot::SharedCommandDispatch for CommandContext {
    async fn dispatch(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String> {
        CommandContext::dispatch(self, cmd, args, session_id, sender_id, user_role).await
    }
}

impl CommandContext {
    /// Try to handle a shared command. Returns `Some(reply)` if the command was
    /// recognised, `None` if the channel should handle it itself.
    pub(crate) async fn dispatch(
        &self,
        cmd: &str,
        args: &str,
        session_id: &str,
        sender_id: Option<&str>,
        user_role: UserRole,
    ) -> Option<String> {
        self.shared()
            .dispatch(cmd, args, session_id, sender_id, user_role)
            .await
    }

    /// Reload the config file (restoring its backup when it no longer
    /// loads). `Ok` says what changed; `Err` says what went wrong.
    pub(crate) async fn reload_config(&self) -> Result<String, String> {
        reload_config(self.agent.as_ref(), &self.config_path).await
    }

    fn shared(&self) -> SharedCommands {
        SharedCommands {
            agent: self.agent.clone(),
            host: Arc::new(DaemonCommandHost {
                state: self.state.clone(),
            }),
            task_registry: self.task_registry.clone(),
            extensions: vec![
                Arc::new(aidaemon_work::chat_command::WorkCommand::new(
                    self.state.clone(),
                )),
                Arc::new(ReloadCommand::new(Arc::new(DaemonConfigReload {
                    agent: self.agent.clone(),
                    config_path: self.config_path.clone(),
                }))),
                Arc::new(CheckpointCommands),
                Arc::new(crate::startup::client_gateway::DeviceCommands),
            ],
        }
    }
}

/// The daemon's services behind the shared commands.
struct DaemonCommandHost {
    state: Arc<dyn StateStore>,
}

#[async_trait]
impl CommandHost for DaemonCommandHost {
    async fn set_next_message_private(
        &self,
        session_id: &str,
        sender_id: Option<&str>,
        on: bool,
    ) -> anyhow::Result<()> {
        crate::agent::memory_plugin::set_next_message_private(
            self.state.as_ref(),
            session_id,
            sender_id,
            on,
        )
        .await
    }

    async fn chat_mode(
        &self,
        session_id: &str,
        now: DateTime<Utc>,
    ) -> Option<(ChatMode, Option<DateTime<Utc>>)> {
        crate::agent::authority::session_mode_with_expiry(self.state.as_ref(), session_id, now)
            .await
            .map(|(mode, until)| (chat_mode(mode), until))
    }

    async fn set_chat_mode(
        &self,
        session_id: &str,
        mode: ChatMode,
        now: DateTime<Utc>,
    ) -> anyhow::Result<DateTime<Utc>> {
        let mode = match mode {
            ChatMode::ReadOnly => SessionMode::ReadOnly,
            ChatMode::Full => SessionMode::Full,
            ChatMode::Unrestricted => SessionMode::Unrestricted,
        };
        crate::agent::authority::set_session_mode(self.state.as_ref(), session_id, mode, now).await
    }

    fn chat_mode_hours(&self) -> i64 {
        crate::agent::authority::SESSION_MODE_TTL_HOURS
    }

    async fn history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>> {
        self.state.get_history(session_id, limit).await
    }

    async fn conversation_summary(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ConversationSummary>> {
        self.state.get_conversation_summary(session_id).await
    }

    fn estimate_message_tokens(&self, messages: &[serde_json::Value]) -> usize {
        crate::context_budget::estimate_multimodal_message_tokens(messages)
    }

    fn estimate_text_tokens(&self, text: &str) -> usize {
        crate::context_budget::estimate_tokens(text)
    }

    async fn token_usage_since(&self, since: &str) -> anyhow::Result<Vec<TokenUsageRecord>> {
        self.state.get_token_usage_since(since).await
    }

    async fn next_scheduled(&self) -> Option<ScheduledItem> {
        let goals = self.state.get_scheduled_goals().await.ok()?;
        let mut next: Option<(DateTime<Utc>, String)> = None;
        for goal in goals.into_iter().filter(|goal| goal.status == "active") {
            let Ok(schedules) = self.state.get_schedules_for_goal(&goal.id).await else {
                continue;
            };
            for schedule in schedules.iter().filter(|schedule| !schedule.is_paused) {
                let Ok(at) = DateTime::parse_from_rfc3339(&schedule.next_run_at) else {
                    continue;
                };
                let at = at.with_timezone(&Utc);
                if next.as_ref().is_none_or(|(earliest, _)| at < *earliest) {
                    // The owner's name for it reads better than the instruction.
                    let what =
                        aidaemon_work::automations::name_from_context(goal.context.as_deref())
                            .unwrap_or_else(|| goal.description.clone());
                    next = Some((at, what));
                }
            }
        }
        next.map(|(at, what)| ScheduledItem {
            what,
            when: aidaemon_work::schedule::owner_time(&at),
        })
    }

    fn owner_time(&self, at: DateTime<Utc>) -> Option<String> {
        Some(aidaemon_work::schedule::owner_time(&at))
    }
}

fn chat_mode(mode: SessionMode) -> ChatMode {
    match mode {
        SessionMode::ReadOnly => ChatMode::ReadOnly,
        SessionMode::Full => ChatMode::Full,
        SessionMode::Unrestricted => ChatMode::Unrestricted,
    }
}

/// `/reload` on the daemon's config file.
struct DaemonConfigReload {
    agent: Arc<dyn DaemonAgentRuntime>,
    config_path: PathBuf,
}

#[async_trait]
impl ConfigReload for DaemonConfigReload {
    async fn reload_config(&self) -> Result<String, String> {
        reload_config(self.agent.as_ref(), &self.config_path).await
    }
}

/// Reload the config file, restoring its backup when it no longer loads.
async fn reload_config(
    agent: &dyn DaemonAgentRuntime,
    config_path: &std::path::Path,
) -> Result<String, String> {
    match AppConfig::load(config_path) {
        Ok(new_config) => agent
            .reload_provider(&new_config)
            .await
            .map_err(|e| format!("Provider reload failed: {}", e)),
        Err(e) => {
            // Config is broken — try to auto-restore from backup
            let backup = config_path.with_extension("toml.bak");
            Err(if backup.exists() {
                if tokio::fs::copy(&backup, config_path).await.is_ok() {
                    format!(
                        "Config reload failed: {}\n\nAuto-restored from backup. Config is back to the previous working state.",
                        e
                    )
                } else {
                    format!(
                        "Config reload failed: {}\n\nBackup restore also failed. Manual intervention needed.",
                        e
                    )
                }
            } else {
                format!("Config reload failed: {}\n\nNo backup available.", e)
            })
        }
    }
}

#[cfg(test)]
mod mode_command_tests {
    use super::*;
    use crate::traits::store_prelude::*;

    /// No host-local `HH:MM` clock time appears in a reply.
    fn assert_no_clock_time(reply: &str) {
        let bytes = reply.as_bytes();
        for i in 0..bytes.len().saturating_sub(4) {
            let w = &bytes[i..i + 5];
            assert!(
                !(w[0].is_ascii_digit()
                    && w[1].is_ascii_digit()
                    && w[2] == b':'
                    && w[3].is_ascii_digit()
                    && w[4].is_ascii_digit()),
                "clock time in {reply:?}"
            );
        }
    }

    /// `/private` opts the chat's next message out of learning, once.
    #[tokio::test]
    async fn private_opts_the_next_message_out_once() {
        use crate::agent::memory_plugin::take_next_message_private;

        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state = Arc::clone(&harness.state);
        let ctx = CommandContext {
            agent: Arc::new(harness.agent),
            state: state.clone(),
            task_registry: Arc::new(TaskRegistry::new(10)),
            config_path: PathBuf::new(),
        };
        let session = "telegram:synthetic-user-1";
        assert!(!take_next_message_private(state.as_ref(), session, None).await);
        let reply = ctx
            .dispatch("/private", "", session, None, UserRole::Guest)
            .await
            .unwrap();
        assert!(reply.contains("won't be remembered"), "{reply}");
        assert!(take_next_message_private(state.as_ref(), session, None).await);
        assert!(!take_next_message_private(state.as_ref(), session, None).await);

        ctx.dispatch("/private", "", session, None, UserRole::Owner)
            .await;
        ctx.dispatch("/private", "off", session, None, UserRole::Owner)
            .await;
        assert!(!take_next_message_private(state.as_ref(), session, None).await);
        assert!(ctx
            .dispatch("/private", "bogus", session, None, UserRole::Owner)
            .await
            .unwrap()
            .starts_with("Usage"));
    }

    /// In a shared chat `/private` belongs to the person who sent it: only
    /// their next message consumes it, and nobody else can cancel it.
    #[tokio::test]
    async fn private_belongs_to_its_sender_in_a_shared_chat() {
        use crate::agent::memory_plugin::take_next_message_private;

        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state = Arc::clone(&harness.state);
        let ctx = CommandContext {
            agent: Arc::new(harness.agent),
            state: state.clone(),
            task_registry: Arc::new(TaskRegistry::new(10)),
            config_path: PathBuf::new(),
        };
        let session = "slack:C-synthetic-shared";
        let alice = Some("slack:U-synthetic-alice");
        let bob = Some("slack:U-synthetic-bob");

        ctx.dispatch("/private", "", session, alice, UserRole::Guest)
            .await
            .unwrap();
        // Bob's message does not consume Alice's request...
        assert!(!take_next_message_private(state.as_ref(), session, bob).await);
        // ...and Bob cannot cancel it.
        ctx.dispatch("/private", "off", session, bob, UserRole::Guest)
            .await
            .unwrap();
        assert!(take_next_message_private(state.as_ref(), session, alice).await);
        assert!(!take_next_message_private(state.as_ref(), session, alice).await);
    }

    #[tokio::test]
    async fn mode_commands_owner_only() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state = Arc::clone(&harness.state);
        let ctx = CommandContext {
            agent: Arc::new(harness.agent),
            state: state.clone(),
            task_registry: Arc::new(TaskRegistry::new(10)),
            config_path: PathBuf::new(),
        };
        let session = "telegram:synthetic-user-1";
        for role in [UserRole::Guest, UserRole::Public] {
            for (cmd, args) in [
                ("/readonly", ""),
                ("/mode", "readonly"),
                ("/mode", "unrestricted"),
            ] {
                assert_eq!(
                    ctx.dispatch(cmd, args, session, None, role)
                        .await
                        .as_deref(),
                    Some("Only the owner can change the mode.")
                );
            }
        }
        assert_eq!(
            state
                .get_setting(&format!("authority_mode:{session}"))
                .await
                .unwrap(),
            None
        );

        let reply = ctx
            .dispatch("/readonly", "", session, None, UserRole::Owner)
            .await
            .unwrap();
        assert_eq!(
            reply,
            "Read-only for this chat for the next 4 hours, starting now, including work already running. aidaemon can look but not change anything. Send /mode full to end it early."
        );
        let shown = ctx
            .dispatch("/mode", "", session, None, UserRole::Owner)
            .await
            .unwrap();
        assert!(shown.starts_with("Mode: read-only (about "), "{shown}");
        assert!(shown.ends_with(" left)."), "{shown}");
        assert_no_clock_time(&shown);
        assert_eq!(
            ctx.dispatch("/mode", "bogus", session, None, UserRole::Owner)
                .await
                .as_deref(),
            Some("Usage: /mode [full|unrestricted|readonly]")
        );
        assert_eq!(
            ctx.dispatch("/mode", "full", session, None, UserRole::Owner)
                .await
                .as_deref(),
            Some("Back to full access from your next message.")
        );
        assert_eq!(
            ctx.dispatch("/mode", "", session, None, UserRole::Owner)
                .await
                .as_deref(),
            Some("Mode: full.")
        );
        let unrestricted = ctx
            .dispatch("/mode", "unrestricted", session, None, UserRole::Owner)
            .await
            .unwrap();
        assert_eq!(
            unrestricted,
            "Unrestricted for this chat for the next 4 hours, starting with your next message."
        );
        let shown = ctx
            .dispatch("/mode", "", session, None, UserRole::Owner)
            .await
            .unwrap();
        assert!(shown.starts_with("Mode: unrestricted (about "), "{shown}");
        assert_no_clock_time(&shown);
    }
}
