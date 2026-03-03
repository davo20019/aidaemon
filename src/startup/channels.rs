use std::path::PathBuf;
use std::sync::{Arc, Weak};

use tracing::{info, warn};

use crate::agent::Agent;
#[cfg(feature = "discord")]
use crate::channels::DiscordChannel;
#[cfg(feature = "slack")]
use crate::channels::SlackChannel;
use crate::channels::{ChannelHub, SessionMap, TelegramChannel};
use crate::config::AppConfig;
use crate::state::SqliteStateStore;
use crate::tasks::TaskRegistry;
use crate::traits::store_prelude::*;
use crate::traits::Channel;

pub struct ChannelBundle {
    pub channels: Vec<Arc<dyn Channel>>,
    pub telegram_bots: Vec<Arc<TelegramChannel>>,
    pub dynamic_telegram_bots: Vec<Arc<TelegramChannel>>,
    #[cfg(feature = "discord")]
    pub discord_bots: Vec<Arc<DiscordChannel>>,
    #[cfg(feature = "discord")]
    pub dynamic_discord_bots: Vec<Arc<DiscordChannel>>,
    #[cfg(feature = "slack")]
    pub slack_bots: Vec<Arc<SlackChannel>>,
    #[cfg(feature = "slack")]
    pub dynamic_slack_bots: Vec<Arc<SlackChannel>>,
}

impl ChannelBundle {
    pub fn set_channel_hub_for_all(&self, weak_hub: Weak<ChannelHub>) {
        for tg in &self.telegram_bots {
            tg.set_channel_hub(weak_hub.clone());
        }
        for tg in &self.dynamic_telegram_bots {
            tg.set_channel_hub(weak_hub.clone());
        }
        #[cfg(feature = "discord")]
        {
            for dc in &self.discord_bots {
                dc.set_channel_hub(weak_hub.clone());
            }
            for dc in &self.dynamic_discord_bots {
                dc.set_channel_hub(weak_hub.clone());
            }
        }
        #[cfg(feature = "slack")]
        {
            for sc in &self.slack_bots {
                sc.set_channel_hub(weak_hub.clone());
            }
            for sc in &self.dynamic_slack_bots {
                sc.set_channel_hub(weak_hub.clone());
            }
        }
    }

    pub async fn send_startup_notifications(&self, config: &AppConfig) {
        if let Some(first_tg) = self.telegram_bots.first() {
            if let Some(first_config) = config.all_telegram_bots().first() {
                if !first_config.allowed_user_ids.is_empty() {
                    let mode = if first_config.webhook.enabled {
                        "webhook"
                    } else {
                        "polling"
                    };
                    let msg = format!("aidaemon is online ({})", mode);
                    for user_id in &first_config.allowed_user_ids {
                        let _ = first_tg.send_text(&user_id.to_string(), &msg).await;
                    }
                }
            }
        }
    }

    pub fn spawn_all(self) {
        #[cfg(feature = "discord")]
        {
            for dc in self.discord_bots {
                tokio::spawn(async move {
                    dc.start_with_retry().await;
                });
            }
            for dc in self.dynamic_discord_bots {
                tokio::spawn(async move {
                    dc.start_with_retry().await;
                });
            }
        }

        #[cfg(feature = "slack")]
        {
            for sc in self.slack_bots {
                tokio::spawn(async move {
                    sc.start_with_retry().await;
                });
            }
            for sc in self.dynamic_slack_bots {
                tokio::spawn(async move {
                    sc.start_with_retry().await;
                });
            }
        }

        for tg in self.dynamic_telegram_bots {
            tokio::spawn(async move {
                tg.start_with_retry().await;
            });
        }
        for tg in self.telegram_bots {
            tokio::spawn(async move {
                tg.start_with_retry().await;
            });
        }
    }
}

fn parse_owner_ids(config: &AppConfig, platform: &str) -> Vec<u64> {
    config
        .users
        .owner_ids
        .get(platform)
        .map(|ids| ids.iter().filter_map(|id| id.parse::<u64>().ok()).collect())
        .unwrap_or_default()
}

fn parse_u64_ids(ids: &[String]) -> Vec<u64> {
    ids.iter().filter_map(|s| s.parse::<u64>().ok()).collect()
}

/// Extract a slug from a bot token (e.g. "123456:ABC..." → "bot-123456").
fn slug_from_bot_token(token: &str) -> String {
    let id_part = token.split(':').next().unwrap_or("unknown");
    format!("bot-{}", id_part)
}

#[allow(clippy::too_many_arguments)]
pub async fn build_channels(
    config: &AppConfig,
    agent: Arc<Agent>,
    config_path: PathBuf,
    session_map: SessionMap,
    task_registry: Arc<TaskRegistry>,
    inbox_dir: &str,
    state: Arc<SqliteStateStore>,
    watchdog_stale_threshold_secs: u64,
) -> ChannelBundle {
    let telegram_owner_ids = parse_owner_ids(config, "telegram");
    let inbox_dir = PathBuf::from(inbox_dir);
    let files_enabled = config.files.enabled;
    let max_file_size_mb = config.files.max_file_size_mb;
    let terminal_web_app_url = config.terminal.effective_web_app_url();
    let terminal_allowed_prefixes = config.terminal.allowed_prefixes.clone();

    let make_telegram = |bot_token: &str,
                         allowed_user_ids: Vec<u64>,
                         webhook: crate::config::TelegramWebhookConfig|
     -> Arc<TelegramChannel> {
        Arc::new(TelegramChannel::new(
            bot_token,
            allowed_user_ids,
            webhook,
            telegram_owner_ids.clone(),
            Arc::clone(&agent),
            config_path.clone(),
            session_map.clone(),
            task_registry.clone(),
            files_enabled,
            inbox_dir.clone(),
            max_file_size_mb,
            state.clone(),
            watchdog_stale_threshold_secs,
            terminal_web_app_url.clone(),
            terminal_allowed_prefixes.clone(),
        ))
    };

    // Collect occupied ports from config bots with explicit webhook
    let defaults = &config.telegram_webhook_defaults;
    let mut occupied_ports: std::collections::HashSet<u16> = std::collections::HashSet::new();
    for bot_config in config.all_telegram_bots() {
        if bot_config.webhook.enabled {
            if let Some(addr) = &bot_config.webhook.listen_addr {
                if let Some(port) = addr.rsplit(':').next().and_then(|p| p.parse::<u16>().ok()) {
                    occupied_ports.insert(port);
                }
            }
        }
    }

    let telegram_bots: Vec<Arc<TelegramChannel>> = config
        .all_telegram_bots()
        .into_iter()
        .map(|bot_config| {
            let webhook = if bot_config.webhook.enabled {
                // Explicit webhook config takes precedence
                bot_config.webhook.clone()
            } else if defaults.enabled && defaults.base_domain.is_some() {
                // Auto-derive from global defaults
                let slug = slug_from_bot_token(&bot_config.bot_token);
                let port = defaults.next_available_port(&mut occupied_ports);
                info!(slug = %slug, port, "Auto-deriving webhook config from global defaults for config bot");
                defaults.derive_webhook_config(&slug, port)
            } else {
                bot_config.webhook.clone()
            };
            info!("Registering Telegram bot (username will be fetched from API)");
            make_telegram(
                &bot_config.bot_token,
                bot_config.allowed_user_ids.clone(),
                webhook,
            )
        })
        .collect();

    #[cfg(feature = "discord")]
    let discord_owner_ids = parse_owner_ids(config, "discord");
    #[cfg(feature = "discord")]
    let make_discord = |bot_token: &str,
                        allowed_user_ids: Vec<u64>,
                        guild_id: Option<u64>|
     -> Arc<DiscordChannel> {
        Arc::new(DiscordChannel::new(
            bot_token,
            allowed_user_ids,
            discord_owner_ids.clone(),
            guild_id,
            Arc::clone(&agent),
            config_path.clone(),
            session_map.clone(),
            task_registry.clone(),
            files_enabled,
            inbox_dir.clone(),
            max_file_size_mb,
            state.clone(),
            watchdog_stale_threshold_secs,
        ))
    };
    #[cfg(feature = "discord")]
    let discord_bots: Vec<Arc<DiscordChannel>> = config
        .all_discord_bots()
        .into_iter()
        .map(|bot_config| {
            info!("Registering Discord bot (username will be fetched from API)");
            make_discord(
                &bot_config.bot_token,
                bot_config.allowed_user_ids.clone(),
                bot_config.guild_id,
            )
        })
        .collect();

    #[cfg(feature = "slack")]
    let make_slack = |app_token: &str,
                      bot_token: &str,
                      allowed_user_ids: Vec<String>,
                      use_threads: bool|
     -> Arc<SlackChannel> {
        Arc::new(SlackChannel::new(
            app_token,
            bot_token,
            allowed_user_ids,
            use_threads,
            Arc::clone(&agent),
            config_path.clone(),
            session_map.clone(),
            task_registry.clone(),
            files_enabled,
            inbox_dir.clone(),
            max_file_size_mb,
            state.clone(),
            watchdog_stale_threshold_secs,
        ))
    };
    #[cfg(feature = "slack")]
    let slack_bots: Vec<Arc<SlackChannel>> = config
        .all_slack_bots()
        .into_iter()
        .map(|bot_config| {
            info!("Registering Slack bot (bot name will be fetched from API)");
            make_slack(
                &bot_config.app_token,
                &bot_config.bot_token,
                bot_config.allowed_user_ids.clone(),
                bot_config.use_threads,
            )
        })
        .collect();

    let mut dynamic_telegram_bots: Vec<Arc<TelegramChannel>> = Vec::new();
    #[cfg(feature = "discord")]
    let mut dynamic_discord_bots: Vec<Arc<DiscordChannel>> = Vec::new();
    #[cfg(feature = "slack")]
    let mut dynamic_slack_bots: Vec<Arc<SlackChannel>> = Vec::new();

    match state.get_dynamic_bots().await {
        Ok(bots) => {
            for bot in bots {
                match bot.channel_type.as_str() {
                    "telegram" => {
                        let allowed_user_ids: Vec<u64> = parse_u64_ids(&bot.allowed_user_ids);
                        let webhook = if defaults.enabled && defaults.base_domain.is_some() {
                            let extra: serde_json::Value =
                                serde_json::from_str(&bot.extra_config).unwrap_or_default();
                            let slug = extra["username"]
                                .as_str()
                                .filter(|s| !s.is_empty())
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| slug_from_bot_token(&bot.bot_token));
                            let port = defaults.next_available_port(&mut occupied_ports);
                            info!(bot_id = bot.id, slug = %slug, port, "Auto-deriving webhook config from global defaults for dynamic bot");
                            defaults.derive_webhook_config(&slug, port)
                        } else {
                            crate::config::TelegramWebhookConfig::default()
                        };
                        info!(bot_id = bot.id, "Loading dynamic Telegram bot");
                        dynamic_telegram_bots.push(make_telegram(
                            &bot.bot_token,
                            allowed_user_ids,
                            webhook,
                        ));
                    }
                    #[cfg(feature = "discord")]
                    "discord" => {
                        let allowed_user_ids: Vec<u64> = parse_u64_ids(&bot.allowed_user_ids);
                        let extra: serde_json::Value =
                            serde_json::from_str(&bot.extra_config).unwrap_or_default();
                        let guild_id = extra["guild_id"].as_u64();
                        info!(bot_id = bot.id, "Loading dynamic Discord bot");
                        dynamic_discord_bots.push(make_discord(
                            &bot.bot_token,
                            allowed_user_ids,
                            guild_id,
                        ));
                    }
                    #[cfg(not(feature = "discord"))]
                    "discord" => {
                        warn!(
                            bot_id = bot.id,
                            "Skipping dynamic Discord bot (feature disabled)"
                        );
                    }
                    #[cfg(feature = "slack")]
                    "slack" => {
                        if let Some(app_token) = &bot.app_token {
                            let extra: serde_json::Value =
                                serde_json::from_str(&bot.extra_config).unwrap_or_default();
                            let use_threads = extra["use_threads"].as_bool().unwrap_or(false);
                            info!(bot_id = bot.id, "Loading dynamic Slack bot");
                            dynamic_slack_bots.push(make_slack(
                                app_token,
                                &bot.bot_token,
                                bot.allowed_user_ids.clone(),
                                use_threads,
                            ));
                        }
                    }
                    #[cfg(not(feature = "slack"))]
                    "slack" => {
                        warn!(
                            bot_id = bot.id,
                            "Skipping dynamic Slack bot (feature disabled)"
                        );
                    }
                    _ => {
                        warn!(
                            bot_id = bot.id,
                            channel_type = %bot.channel_type,
                            "Unknown dynamic bot channel type, skipping"
                        );
                    }
                }
            }
        }
        Err(e) => {
            warn!("Failed to load dynamic bots: {}", e);
        }
    }

    let mut channels: Vec<Arc<dyn Channel>> = telegram_bots
        .iter()
        .map(|t| t.clone() as Arc<dyn Channel>)
        .collect();
    channels.extend(
        dynamic_telegram_bots
            .iter()
            .map(|t| t.clone() as Arc<dyn Channel>),
    );
    #[cfg(feature = "discord")]
    {
        channels.extend(discord_bots.iter().map(|d| d.clone() as Arc<dyn Channel>));
        channels.extend(
            dynamic_discord_bots
                .iter()
                .map(|d| d.clone() as Arc<dyn Channel>),
        );
    }
    #[cfg(feature = "slack")]
    {
        channels.extend(slack_bots.iter().map(|s| s.clone() as Arc<dyn Channel>));
        channels.extend(
            dynamic_slack_bots
                .iter()
                .map(|s| s.clone() as Arc<dyn Channel>),
        );
    }

    info!(count = channels.len(), "Channels registered");

    ChannelBundle {
        channels,
        telegram_bots,
        dynamic_telegram_bots,
        #[cfg(feature = "discord")]
        discord_bots,
        #[cfg(feature = "discord")]
        dynamic_discord_bots,
        #[cfg(feature = "slack")]
        slack_bots,
        #[cfg(feature = "slack")]
        dynamic_slack_bots,
    }
}
