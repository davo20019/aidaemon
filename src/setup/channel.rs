//! The chat step of first-run setup and the first conversation, on the
//! running daemon. A bot connected here starts with no allowed users and
//! claims the first person who messages it (as `/connect` does); the step
//! shows connected once that person's message has reached the daemon, and
//! the bot has answered it in the chat. Finishing or skipping the step posts
//! the first conversation's greeting.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aidaemon_client_protocol::{ChannelWait, SetupAction, SetupState, SetupStep};

use super::progress::{self, Progress};
use crate::traits::StateStore;

/// The first conversation's title.
const GREETING_TITLE: &str = "Welcome";

/// What the agent says first: who it is, what it may do on its own and
/// what it asks about, and what to try.
pub(crate) fn greeting(agent_name: &str, on_server: bool) -> (String, Vec<String>) {
    // The default name reads as the product's.
    let name = match agent_name.trim() {
        "" | "aidaemon" => "AIDaemon",
        name => name,
    };
    let place = if on_server {
        "I run on your server, around the clock"
    } else {
        "I run on this computer. While it sleeps, I'm offline"
    };
    let text = format!(
        "Hi, I'm {name}. I research, write, organize files, run commands and keep an eye on \
         things for you on a schedule. {place}.\n\n\
         I read and look things up on my own. Before anything that sends, posts, deletes or \
         spends, I ask you first.\n\n\
         Try one of these, or ask for anything. Want a different name or personality? Just say \
         so. To talk to me from your phone, open Settings and choose Reach it from your phone."
    );
    // Something that finishes in a minute first, then one that pays off
    // tomorrow.
    let suggestions = vec![
        if on_server {
            "What can you do on this server?".to_string()
        } else {
            "Look through my Downloads folder and suggest what to clean up".to_string()
        },
        "Find today's three biggest tech stories and summarize each in two lines".to_string(),
        "Every weekday at 8, send me a short summary of news on a topic I pick".to_string(),
    ];
    (text, suggestions)
}

/// The chat step, on the running daemon's client host.
pub(crate) struct ChannelSetup {
    pub config_path: PathBuf,
    pub state: Arc<dyn StateStore>,
    pub events: Arc<crate::events::EventStore>,
}

impl ChannelSetup {
    fn progress(&self) -> Option<Progress> {
        progress::load(&self.config_path)
    }

    /// Where setup stands; `None` once it is done.
    pub(crate) async fn state(&self, sessions: &[String]) -> Option<SetupState> {
        let progress = self.progress()?;
        if progress.channel_done {
            // Only the greeting is left: post it now.
            self.greet(progress).await;
            return None;
        }
        Some(self.snapshot(&progress, sessions))
    }

    fn snapshot(&self, progress: &Progress, sessions: &[String]) -> SetupState {
        SetupState {
            step: SetupStep::Channel,
            model: (!progress.model.is_empty()).then(|| progress.model.clone()),
            channels: super::model::channels(),
            channel: progress.bot.clone().map(|(platform, bot)| ChannelWait {
                connected: sessions
                    .iter()
                    .any(|channel| channel_is(channel, &platform)),
                platform,
                bot,
                claim_code: progress.bot_claim_code.clone(),
            }),
            on_server: progress.on_server,
            ..SetupState::default()
        }
    }

    pub(crate) async fn step(
        &self,
        action: SetupAction,
        sessions: &[String],
    ) -> Result<SetupState, String> {
        let mut progress = match self.progress() {
            Some(progress) => progress,
            // Set up already: the owner asks for the chat step from
            // Settings. It opens again; the greeting is not posted twice.
            None if matches!(action, SetupAction::Detect) => {
                let progress = Progress {
                    channel_done: false,
                    greeted: true,
                    bot: None,
                    ..progress::load_any(&self.config_path)
                };
                progress::save(&self.config_path, &progress).map_err(|e| e.to_string())?;
                progress
            }
            None => return Err("This daemon is already set up".into()),
        };
        match action {
            SetupAction::ConnectChannel {
                platform,
                token,
                app_token,
            } => {
                let (bot, claim_code) = connect_bot(
                    self.state.as_ref(),
                    &platform,
                    token.0.trim(),
                    app_token.as_ref().map(|t| t.0.trim()),
                )
                .await?;
                progress.bot = Some((platform.clone(), bot));
                progress.bot_claim_code = claim_code;
                progress::save(&self.config_path, &progress).map_err(|e| e.to_string())?;
                // Bots start with the daemon: restart in place, and the app
                // reconnects to a daemon running it.
                tokio::spawn(async {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    crate::startup::settings_upkeep::restart_in_place();
                });
                Ok(self.snapshot(&progress, sessions))
            }
            SetupAction::SkipChannel | SetupAction::Finish => {
                progress.channel_done = true;
                progress::save(&self.config_path, &progress).map_err(|e| e.to_string())?;
                self.greet(progress).await;
                Ok(SetupState {
                    step: SetupStep::Done,
                    ..SetupState::default()
                })
            }
            SetupAction::Detect => Ok(self.snapshot(&progress, sessions)),
            _ => Err("The model is already connected".into()),
        }
    }

    /// Post the first conversation's greeting, once.
    async fn greet(&self, mut progress: Progress) {
        if progress.greeted {
            return;
        }
        let Some(gateway) = crate::startup::client_gateway::running() else {
            return;
        };
        let name = agent_name_of(&self.config_path);
        let (text, suggestions) = greeting(&name, progress.on_server);
        let session_id = aidaemon_client_gateway::new_app_session_id();
        if let Err(error) = self.keep(&session_id, &text).await {
            tracing::warn!(%error, "Could not keep the greeting in the conversation's history");
        }
        gateway
            .post_greeting(&session_id, GREETING_TITLE, &text, suggestions)
            .await;
        progress.greeted = true;
        if let Err(error) = progress::save(&self.config_path, &progress) {
            tracing::warn!(%error, "Could not note the greeting");
        }
    }

    /// The greeting in the session's history, so the agent reads it with the
    /// owner's first reply and the app shows it after a reload.
    async fn keep(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        let message_id = uuid::Uuid::new_v4().to_string();
        let turn_id = format!("setup-greeting-{message_id}");
        self.events
            .append(crate::events::Event::new(
                session_id,
                crate::events::EventType::AssistantResponse,
                serde_json::json!({
                    "message_id": message_id,
                    "content": text,
                    "turn_id": turn_id,
                    "model": "setup",
                }),
            ))
            .await?;
        let mut message = crate::traits::Message::new_runtime(&message_id, session_id, "assistant");
        message.content = Some(text.to_string());
        message.turn_id = Some(turn_id);
        self.state.append_message(&message).await
    }
}

/// The configured agent name, read without loading the whole config.
fn agent_name_of(config_path: &Path) -> String {
    std::fs::read_to_string(config_path)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|t| {
            t.get("agent")
                .and_then(|a| a.get("name"))
                .and_then(toml::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_default()
}

/// Whether a session's channel (`telegram`, `telegram:name`...) is on
/// `platform`.
fn channel_is(channel: &str, platform: &str) -> bool {
    channel == platform || channel.starts_with(&format!("{platform}:"))
}

/// Check a bot's tokens with its platform and keep it as a runtime bot.
/// Returns the bot as the owner finds it and, for a new bot, the one-time
/// code whose direct message claims it.
async fn connect_bot(
    state: &dyn StateStore,
    platform: &str,
    token: &str,
    app_token: Option<&str>,
) -> Result<(String, Option<String>), String> {
    use crate::channels::connect;
    if token.is_empty() {
        return Err("Paste the bot's token".into());
    }
    let (name, app_token) = match platform {
        "telegram" => (
            format!("@{}", connect::validate_telegram_token(token).await?),
            None,
        ),
        "discord" => (connect::validate_discord_token(token).await?, None),
        "slack" => {
            let app = app_token
                .filter(|t| !t.is_empty())
                .ok_or("Slack also needs the app-level token (xapp-…)")?;
            let (bot, team) = connect::validate_slack_tokens(token, app).await?;
            (format!("{bot} in {team}"), Some(app.to_string()))
        }
        other => return Err(format!("{other} is not a chat this daemon connects")),
    };
    if connect::check_bot_exists(state, platform, token).await? {
        return Ok((name, None));
    }
    let extra = if platform == "telegram" {
        serde_json::json!({ "username": name.trim_start_matches('@') }).to_string()
    } else {
        "{}".to_string()
    };
    let claim_code = aidaemon_channels::claim::new_claim_code();
    let extra = aidaemon_channels::claim::extra_with_claim_code(&extra, &claim_code);
    connect::persist_dynamic_bot(
        state,
        &crate::traits::DynamicBot {
            id: 0,
            channel_type: platform.to_string(),
            bot_token: token.to_string(),
            app_token,
            // Empty: the direct message carrying the claim code claims it.
            allowed_user_ids: Vec::new(),
            extra_config: extra,
            created_at: String::new(),
        },
    )
    .await?;
    Ok((name, Some(claim_code)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_greeting_says_what_asks_and_where_it_runs() {
        let (text, suggestions) = greeting("Synthetic Agent", false);
        assert!(text.starts_with("Hi, I'm Synthetic Agent."));
        assert!(text.contains("sends, posts, deletes or spends"));
        assert!(text.contains("While it sleeps, I'm offline"));
        assert_eq!(suggestions.len(), 3);
        assert!(suggestions[0].contains("Downloads"));
        let (server, on_server) = greeting("", true);
        assert!(!on_server[0].contains("Downloads"));
        assert!(server.starts_with("Hi, I'm AIDaemon."));
        assert!(!server.contains("sleeps"));
    }

    #[tokio::test]
    async fn the_chat_step_opens_again_from_settings_without_a_second_greeting() {
        use crate::events::EventStoreForTests;
        use crate::state::SqliteStateStore;
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("state.db");
        let db = db.to_str().unwrap();
        let embeddings = Arc::new(crate::memory::embeddings::EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(db, 100, None, embeddings)
                .await
                .unwrap(),
        );
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{db}"))
            .await
            .unwrap();
        let events = Arc::new(crate::events::EventStore::new(pool).await.unwrap());
        let config_path = dir.path().join("config.toml");
        let setup = ChannelSetup {
            config_path: config_path.clone(),
            state,
            events,
        };
        // Finished: nothing left, and any other step is refused.
        progress::save(
            &config_path,
            &Progress {
                channel_done: true,
                greeted: true,
                model: "Synthetic model".into(),
                ..Progress::default()
            },
        )
        .unwrap();
        assert!(setup.state(&[]).await.is_none());
        assert!(setup.step(SetupAction::SkipChannel, &[]).await.is_err());

        let opened = setup.step(SetupAction::Detect, &[]).await.unwrap();
        assert_eq!(opened.step, SetupStep::Channel);
        assert_eq!(opened.model.as_deref(), Some("Synthetic model"));
        let progress = progress::load(&config_path).unwrap();
        assert!(!progress.channel_done && progress.greeted);

        let done = setup.step(SetupAction::SkipChannel, &[]).await.unwrap();
        assert_eq!(done.step, SetupStep::Done);
        assert!(progress::load(&config_path).is_none());
    }

    #[test]
    fn a_session_channel_matches_its_platform_only() {
        assert!(channel_is("telegram", "telegram"));
        assert!(channel_is("telegram:synthetic_bot", "telegram"));
        assert!(!channel_is("telegramx", "telegram"));
        assert!(!channel_is("app", "telegram"));
    }
}
