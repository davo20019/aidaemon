use std::path::Path;
use std::sync::Arc;

use crate::config::AppConfig;
use crate::memory::embeddings::EmbeddingService;
use crate::nodes::domain::{NodeAction, CHILD_COMPANION_POLICY};
use crate::state::SqliteStateStore;
use anyhow::Context;
use sqlx::{Row, SqlitePool};

pub fn handle_node_command(args: &[String], config_path: &Path) -> anyhow::Result<()> {
    if matches!(
        args.first().map(String::as_str),
        None | Some("-h" | "--help")
    ) {
        print_help();
        return Ok(());
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let mut config = AppConfig::load(config_path)?;
        crate::startup::db_security::enforce_database_encryption(&mut config, config_path).await?;
        let resolved =
            crate::startup::db_security::resolve_db_path(config_path, &config.state.db_path);
        config.state.db_path = resolved.to_string_lossy().into_owned();
        let embeddings =
            Arc::new(EmbeddingService::new().map_err(|error| {
                anyhow::anyhow!("Failed to initialize database support: {error}")
            })?);
        let state = SqliteStateStore::new(
            &config.state.db_path,
            config.state.working_memory_cap,
            config.state.encryption_key.as_deref(),
            embeddings,
        )
        .await?;
        let store = Arc::new(super::store::NodeStore::new(
            state.pool(),
            super::auth::load_or_create_instance_key()?,
        ));
        let service = super::service::NodeService::new(store.clone(), config.nodes.clone());
        match args[0].as_str() {
            "pair" => {
                let kind = option(args, "--kind").unwrap_or("k10");
                let name = option(args, "--name").unwrap_or("K10 Companion");
                let owner = option(args, "--owner").unwrap_or("parent");
                let policy = option(args, "--policy").unwrap_or(CHILD_COMPANION_POLICY);
                let offer = service
                    .create_pairing_offer(owner, kind, name, policy)
                    .await?;
                println!("Pairing offer created for {name} ({kind}).");
                println!("Offer ID: {}", offer.offer_id);
                println!("One-time secret: {}", offer.offer_secret);
                println!("Expires: {}", offer.expires_at.to_rfc3339());
                println!("Keep this one-time secret private; it cannot be displayed again.");
            }
            "list" => {
                let nodes = store.list_nodes().await?;
                if nodes.is_empty() {
                    println!("No Nodes enrolled.");
                }
                for node in nodes {
                    println!(
                        "{}  {}  kind={}  policy={}  status={}",
                        node.node_id,
                        node.display_name,
                        node.kind,
                        node.policy_profile,
                        if node.revoked_at.is_some() {
                            "revoked"
                        } else {
                            "active"
                        }
                    );
                }
            }
            "status" => {
                let health = super::store::node_health_snapshot(
                    store.pool(),
                    args.get(1).map(String::as_str),
                )
                .await?;
                let age_seconds = health
                    .last_seen_at
                    .map(|seen| (chrono::Utc::now() - seen).num_seconds().max(0));
                println!(
                    "{}  connection={}  last_seen_age_seconds={}  runtime={}  firmware={}",
                    health.display_name,
                    match age_seconds {
                        Some(age) if age <= 90 => "recently_connected",
                        Some(_) => "stale",
                        None => "never_connected",
                    },
                    age_seconds
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    health.runtime_version.as_deref().unwrap_or("unknown"),
                    health.firmware_version.as_deref().unwrap_or("unknown"),
                );
                if let Some(recovery) = health.recovery {
                    println!("Recovery: {}", serde_json::to_string(&recovery)?);
                } else {
                    println!("Recovery: not reported by this Runtime.");
                }
            }
            "revoke" => {
                let node_id = args
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Usage: aidaemon node revoke <node_id>"))?;
                anyhow::ensure!(
                    store.revoke_node(node_id).await?,
                    "Node was not found or already revoked"
                );
                println!(
                    "Revoked Node {node_id}; active sessions and credentials are invalid now."
                );
            }
            "authorize" | "deny" => {
                let node_id = args
                    .get(1)
                    .ok_or_else(|| anyhow::anyhow!("Node ID is required"))?;
                let action = NodeAction::parse(
                    args.get(2)
                        .ok_or_else(|| anyhow::anyhow!("Node action is required"))?,
                )?;
                store
                    .set_authorization(
                        node_id,
                        action,
                        args[0] == "authorize",
                        serde_json::json!({}),
                    )
                    .await?;
                println!("Node authorization updated; existing sessions were closed.");
            }
            "announce" => {
                let selector = args.get(1).ok_or_else(|| {
                    anyhow::anyhow!("Usage: aidaemon node announce <node-name-or-id> <text>")
                })?;
                let text = args.get(2..).unwrap_or_default().join(" ");
                anyhow::ensure!(
                    !text.trim().is_empty(),
                    "Usage: aidaemon node announce <node-name-or-id> <text>"
                );
                let speech = super::speech::configured_synthesizer(&config.nodes.speech)?;
                let announcements = super::announcement::NodeAnnouncementService::new(
                    store.clone(),
                    config.nodes.clone(),
                    speech,
                );
                let delivery = announcements.queue_and_wait(Some(selector), &text).await?;
                let status = delivery
                    .receipt
                    .as_ref()
                    .map(|receipt| receipt.status.as_str())
                    .unwrap_or("queued");
                println!(
                    "Announcement for {}: {}.",
                    delivery.queued.display_name, status
                );
            }
            "monitor" => {
                handle_monitor_command(&args[1..], &config, state.pool()).await?;
            }
            other => anyhow::bail!("Unknown node command: {other}"),
        }
        anyhow::Ok(())
    })
}

async fn handle_monitor_command(
    args: &[String],
    config: &AppConfig,
    pool: SqlitePool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.nodes.monitoring.enabled,
        "Node environmental monitoring is disabled"
    );
    let command = args.first().map(String::as_str).unwrap_or("help");
    if matches!(command, "help" | "-h" | "--help") {
        print_monitor_help();
        return Ok(());
    }
    let owner_session = resolve_monitor_owner_session(args, config, &pool).await?;
    let service =
        super::monitoring::NodeMonitoringService::new(pool, config.nodes.monitoring.clone());
    match command {
        "create" => {
            let capability = match required_option(args, "--capability")? {
                "temperature" => "sensor.environment.temperature",
                "humidity" => "sensor.environment.humidity",
                _ => anyhow::bail!("--capability must be temperature or humidity"),
            };
            let request = super::monitoring::CreateNodeMonitor {
                name: required_option(args, "--name")?.to_string(),
                owner_session_id: owner_session.clone(),
                node: option(args, "--node").map(str::to_string),
                capability_id: capability.to_string(),
                comparison: super::monitoring::MonitorComparison::parse(required_option(
                    args,
                    "--comparison",
                )?)?,
                threshold: parse_option(args, "--threshold")?,
                clear_threshold: parse_option(args, "--clear-threshold")?,
                duration_seconds: parse_optional(args, "--condition-seconds")?.unwrap_or(0),
                stale_after_seconds: parse_optional(args, "--stale-seconds")?,
                offline_after_seconds: parse_optional(args, "--offline-seconds")?,
                repeat_seconds: parse_optional(args, "--repeat-seconds")?.unwrap_or(0),
                send_recovery: !args.iter().any(|arg| arg == "--no-recovery"),
                duration_minutes: parse_option(args, "--duration-minutes")?,
                mandate_id: option(args, "--mandate").map(str::to_string),
            };
            let monitor = service.create(request).await?;
            println!(
                "Created monitor {} for {} (status={}, expires={}).",
                monitor.monitor_id, monitor.node, monitor.status, monitor.expires_at
            );
        }
        "list" => {
            let monitors = service.list(&owner_session).await?;
            if monitors.is_empty() {
                println!("No Node monitors for the selected owner session.");
            }
            for monitor in monitors {
                println!(
                    "{}  {}  node={}  status={}  threshold_state={}  availability={}  expires={}",
                    monitor.monitor_id,
                    monitor.name,
                    monitor.node,
                    monitor.status,
                    monitor.threshold_state,
                    monitor.availability_state,
                    monitor.expires_at
                );
            }
        }
        "history" => {
            let monitor_id = args.get(1).context("monitor ID is required")?;
            let since_hours = parse_optional(args, "--since-hours")?.unwrap_or(24);
            let limit = parse_optional(args, "--limit")?.unwrap_or(100);
            let monitor = service.get(monitor_id, &owner_session).await?;
            let events = service
                .history(monitor_id, &owner_session, since_hours, limit)
                .await?;
            let readings = service
                .sensor_history(monitor_id, &owner_session, since_hours, limit)
                .await?;
            println!(
                "Monitor {}: status={}, threshold_state={}, availability={}, sensor_rows={}, events={}.",
                monitor.name,
                monitor.status,
                monitor.threshold_state,
                monitor.availability_state,
                readings.len(),
                events.len()
            );
            for event in events {
                println!(
                    "{}  {}  delivered={}",
                    event.created_at, event.event_kind, event.delivered
                );
            }
        }
        "pause" | "resume" | "cancel" => {
            let monitor_id = args.get(1).context("monitor ID is required")?;
            let monitor = service
                .change_status(monitor_id, &owner_session, command)
                .await?;
            println!("Monitor {} is {}.", monitor.name, monitor.status);
        }
        other => anyhow::bail!("Unknown node monitor command: {other}"),
    }
    Ok(())
}

async fn resolve_monitor_owner_session(
    args: &[String],
    config: &AppConfig,
    pool: &SqlitePool,
) -> anyhow::Result<String> {
    if let Some(session) = option(args, "--session") {
        anyhow::ensure!(!session.trim().is_empty(), "--session cannot be empty");
        return Ok(session.trim().to_string());
    }
    let allowed: std::collections::HashSet<i64> = config
        .all_telegram_bots()
        .into_iter()
        .flat_map(|bot| bot.allowed_user_ids)
        .filter_map(|id| i64::try_from(id).ok())
        .collect();
    anyhow::ensure!(
        !allowed.is_empty(),
        "No Telegram owner is configured; pass --session explicitly"
    );
    let rows = sqlx::query(
        "SELECT session_id FROM session_channels
         WHERE channel_name = 'telegram' OR channel_name LIKE 'telegram:%'
         ORDER BY updated_at DESC",
    )
    .fetch_all(pool)
    .await?;
    let candidate = rows
        .into_iter()
        .map(|row| row.get::<String, _>("session_id"))
        .find(|session| {
            crate::session::telegram_chat_id_from_session(session)
                .is_some_and(|chat_id| allowed.contains(&chat_id))
        });
    candidate
        .context("Could not resolve a private Telegram owner session; pass --session explicitly")
}

fn required_option<'a>(args: &'a [String], name: &str) -> anyhow::Result<&'a str> {
    option(args, name).ok_or_else(|| anyhow::anyhow!("{name} is required"))
}

fn parse_option<T>(args: &[String], name: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    required_option(args, name)?
        .parse()
        .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
}

fn parse_optional<T>(args: &[String], name: &str) -> anyhow::Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    option(args, name)
        .map(|value| {
            value
                .parse()
                .map_err(|error| anyhow::anyhow!("invalid {name}: {error}"))
        })
        .transpose()
}

fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn print_help() {
    println!("Usage: aidaemon node <command>");
    println!("  pair [--kind k10] [--name NAME] [--owner OWNER] [--policy PROFILE]");
    println!("  list");
    println!("  status [node-name-or-id]");
    println!("  authorize <node_id> <action>");
    println!("  deny <node_id> <action>");
    println!("  announce <node-name-or-id> <text>");
    println!("  monitor <create|list|history|pause|resume|cancel> [options]");
    println!("  revoke <node_id>");
}

fn print_monitor_help() {
    println!("Usage: aidaemon node monitor <command>");
    println!("  create --name NAME [--node NAME] --capability temperature|humidity");
    println!("         --comparison above|below --threshold N --clear-threshold N");
    println!("         --duration-minutes N [--condition-seconds N] [--stale-seconds N]");
    println!("         [--offline-seconds N] [--repeat-seconds N] [--no-recovery]");
    println!("  list");
    println!("  history <monitor_id> [--since-hours N] [--limit N]");
    println!("  pause|resume|cancel <monitor_id>");
    println!("Owner session is resolved from the most recently active configured private Telegram Channel; use --session only for explicit local administration.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn owner_session_resolution_prefers_most_recent_matching_namespace() {
        let config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test-key"

            [telegram]
            bot_token = "synthetic-token"
            allowed_user_ids = [42]
            "#,
        )
        .unwrap();
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE session_channels (
                session_id TEXT PRIMARY KEY, channel_name TEXT NOT NULL, updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO session_channels (session_id, channel_name, updated_at)
             VALUES ('oldbot:42', 'telegram:oldbot', '2026-08-03T00:00:00Z'),
                    ('currentbot:42', 'telegram:currentbot', '2026-08-04T00:00:00Z'),
                    ('otherbot:99', 'telegram:otherbot', '2026-08-05T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let resolved = resolve_monitor_owner_session(&[], &config, &pool)
            .await
            .unwrap();
        assert_eq!(resolved, "currentbot:42");
    }
}
