//! Setup mode: what the daemon runs when it has no config. The Client
//! Gateway serves on its usual loopback address with no agent, channels or
//! background work behind it, the local token is minted for an app on this
//! machine, and a one-time code is printed for a device elsewhere. Once a
//! model passes its test the config is written and setup mode returns, so
//! the daemon starts with it in the same process.

use std::path::Path;
use std::sync::Arc;

use aidaemon_client_gateway::setup::{SetupDeps, SetupGateway};
use tokio_util::sync::CancellationToken;

/// How often a fresh code is printed while nobody has paired (each lasts
/// ten minutes).
const CODE_EVERY: std::time::Duration = std::time::Duration::from_secs(9 * 60);

/// The daemon's database, opened as the daemon opens it (its key made or
/// read first), for the devices paired during setup.
async fn open_pool(config_path: &Path) -> anyhow::Result<sqlx::SqlitePool> {
    let mut config: crate::config::AppConfig = toml::from_str(super::model::STARTER)
        .and_then(|mut table: toml::Table| {
            table.insert(
                "provider".into(),
                toml::Value::Table(toml::Table::from_iter([(
                    "api_key".to_string(),
                    toml::Value::String(String::new()),
                )])),
            );
            toml::Value::Table(table).try_into()
        })
        .map_err(|e| anyhow::anyhow!("the starter config: {e}"))?;
    crate::startup::db_security::enforce_database_encryption(&mut config, config_path).await?;
    let db_path = crate::startup::db_security::resolve_db_path(config_path, &config.state.db_path);
    let mut options = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal);
    if let Some(key) = config
        .state
        .encryption_key
        .as_deref()
        .filter(|k| !k.trim().is_empty())
    {
        options = options.pragma("key", format!("'{}'", key.trim().replace('\'', "''")));
    }
    Ok(sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await?)
}

/// Run setup mode until a model is connected and the config written.
pub(crate) async fn run(config_path: &Path) -> anyhow::Result<()> {
    let db_path = crate::startup::db_security::resolve_db_path(config_path, "aidaemon.db");
    crate::single_instance::acquire(&std::path::PathBuf::from(format!(
        "{}.lock",
        db_path.display()
    )))?;
    let pool = open_pool(config_path).await?;
    let log = aidaemon_store_sqlite::SqliteClientEventLog::open(pool.clone())
        .await
        .map_err(|e| anyhow::anyhow!("client event log: {e}"))?;
    let gateway_config = crate::startup::client_gateway::with_installed_app(
        aidaemon_client_gateway::ClientGatewayConfig {
            enabled: true,
            // A second daemon on one machine needs another port.
            port: std::env::var("AIDAEMON_GATEWAY_PORT")
                .ok()
                .and_then(|p| p.trim().parse().ok())
                .unwrap_or(aidaemon_client_gateway::ClientGatewayConfig::default().port),
            ..Default::default()
        },
    );
    crate::startup::client_gateway::local_token_or_mint(&gateway_config)?;

    let stop = CancellationToken::new();
    let host = super::model::ModelSetup::new(config_path.to_path_buf(), stop.clone());
    {
        let host = host.clone();
        tokio::spawn(async move { host.detect().await });
    }
    let gateway = SetupGateway::start(SetupDeps {
        host: Arc::new(super::model::Host(host.clone())),
        updates: host.updates(),
        log,
        config: gateway_config.clone(),
        token_file: aidaemon_client_gateway::token::token_file(&gateway_config),
        stop: stop.clone(),
    })
    .await?;

    let listener = aidaemon_client_gateway::setup::bind(&gateway).await?;
    eprintln!();
    eprintln!("  AIdaemon is not set up yet.");
    eprintln!("  On this computer: open the AIdaemon app.");
    eprintln!(
        "  From another device: reach this machine's port {} (an SSH tunnel or your",
        gateway_config.port
    );
    eprintln!("  own HTTPS tunnel), open the link below and enter the code.");
    eprintln!("  In a terminal instead: stop this and run `aidaemon setup`.");
    {
        let gateway = gateway.clone();
        let stop = stop.clone();
        tokio::spawn(async move {
            while !stop.is_cancelled() {
                match gateway.console_offer().await {
                    Some(offer) => {
                        eprintln!();
                        eprintln!("  Link: {}", gateway.web_link(&offer));
                        eprintln!("  Code: {}  (valid 10 minutes, one device)", offer.code);
                        eprintln!();
                    }
                    None => return,
                }
                tokio::select! {
                    _ = tokio::time::sleep(CODE_EVERY) => {}
                    _ = stop.cancelled() => return,
                }
            }
        });
    }

    let served = aidaemon_client_gateway::setup::serve(gateway, listener).await;
    pool.close().await;
    served?;
    if !config_path.exists() {
        anyhow::bail!("setup stopped before a model was connected");
    }
    eprintln!("  Set up. Starting AIdaemon…");
    Ok(())
}
