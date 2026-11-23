//! What the host configures: [`BridgeSettings`], the [`BridgeHost`]
//! seam, and [`spawn_if_configured`].

use super::*;

#[derive(Clone)]
pub(crate) struct RuntimeConfig {
    pub(crate) ws_url: String,
    pub(crate) user_id: String,
    /// The configured device id, if any; otherwise one is derived from this
    /// install's bridge key once it is loaded (`default_device_id`).
    pub(crate) device_id: Option<String>,
    pub(crate) shell: String,
    pub(crate) default_cwd: PathBuf,
    pub(crate) inbox_dir: PathBuf,
    pub(crate) max_upload_bytes: usize,
    pub(crate) review_profiles: HashMap<String, ReviewProfile>,
    pub(crate) auth: BridgeAuth,
}

pub(crate) fn validate_daemon_ws_url(ws_url: &str) -> anyhow::Result<()> {
    let parsed = reqwest::Url::parse(ws_url)?;
    match parsed.scheme() {
        "wss" => Ok(()),
        other => anyhow::bail!(
            "terminal daemon websocket URL must use wss:// (found {}://)",
            other
        ),
    }
}

/// What the bridge reads from the host's configuration.
#[derive(Debug, Clone, Default)]
pub struct BridgeSettings {
    /// Master switch (`[terminal].bridge_enabled`).
    pub enabled: bool,
    /// The Telegram owner the bridge acts for. `None` keeps the bridge off.
    pub user_id: Option<u64>,
    /// The relay's daemon websocket (`wss://`, or `ws://` on loopback only).
    pub ws_url: String,
    /// The shell for terminal sessions; `$SHELL`, then `/bin/bash`, when unset.
    pub shell: Option<String>,
    /// Where uploaded images land (relative paths resolve against the
    /// working directory).
    pub inbox_dir: String,
    /// The largest accepted upload in bytes; 0 refuses uploads.
    pub max_upload_bytes: u64,
    /// A fixed device id; derived from this install's bridge key when unset.
    pub device_id: Option<String>,
    /// Telegram bot tokens from the config, used to mint a connect token.
    pub bot_tokens: Vec<String>,
    /// A static daemon connect token.
    pub static_connect_token: Option<String>,
    /// Fall back to the static token when bot-proof minting fails.
    pub allow_static_token_fallback: bool,
    /// Configured review agents; each supported agent overrides the
    /// built-in profile of the same name.
    pub review_agents: Vec<ReviewAgentSettings>,
}

/// A configured CLI agent the bridge may run for a code review.
#[derive(Debug, Clone, Default)]
pub struct ReviewAgentSettings {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub timeout_secs: Option<u64>,
    pub max_output_chars: Option<usize>,
}

/// A bot added at runtime (not in the config file).
#[derive(Debug, Clone, Default)]
pub struct RuntimeBot {
    /// `telegram`, `discord` or `slack`.
    pub channel_type: String,
    pub bot_token: String,
    /// Empty: anyone may use it.
    pub allowed_user_ids: Vec<String>,
}

/// A one-time code that attaches a client to a relay session.
#[derive(Debug, Clone)]
pub struct AttachHandoff {
    pub code: String,
    pub relay_session_id: String,
    pub expires_at_unix: i64,
}

/// What the bridge needs from its host: bots added at runtime and the relay
/// session records shared with the host's Telegram `/agent` command.
#[async_trait::async_trait]
pub trait BridgeHost: Send + Sync {
    /// Bots added at runtime; their Telegram tokens can mint a connect token.
    async fn runtime_bots(&self) -> anyhow::Result<Vec<RuntimeBot>>;
    /// Creates a handoff code for `relay_session_id`.
    async fn create_handoff_code(
        &self,
        relay_session_id: &str,
        owner_user_id: u64,
    ) -> anyhow::Result<AttachHandoff>;
    /// Looks up an unexpired, unconsumed handoff code.
    async fn resolve_handoff_code(&self, code: &str) -> anyhow::Result<AttachHandoff>;
    /// Marks a handoff code used.
    async fn consume_handoff_code(&self, code: &str) -> anyhow::Result<AttachHandoff>;
    /// Records the owner's most recently active relay session.
    async fn set_last_active_relay_session_id(
        &self,
        owner_user_id: u64,
        relay_session_id: &str,
    ) -> anyhow::Result<()>;
    /// The owner's most recently active relay session.
    async fn get_last_active_relay_session_id(
        &self,
        owner_user_id: u64,
    ) -> anyhow::Result<Option<String>>;
    /// Binds a Telegram session to a relay session.
    async fn bind_telegram_session_to_relay(
        &self,
        telegram_session_id: &str,
        relay_session_id: &str,
    ) -> anyhow::Result<()>;
    /// The relay session bound to a Telegram session, if still fresh.
    async fn resolve_relay_for_telegram_session(
        &self,
        telegram_session_id: &str,
    ) -> anyhow::Result<Option<String>>;
}

/// Starts the bridge in the background when the settings allow it. Returns
/// false when it stays off (disabled, no owner, an insecure URL, no auth).
pub fn spawn_if_configured(
    settings: BridgeSettings,
    host: Arc<dyn BridgeHost>,
    secrets: Arc<dyn SecretStore>,
) -> bool {
    if !settings.enabled {
        info!("Terminal bridge disabled by config ([terminal].bridge_enabled = false)");
        return false;
    }

    let Some(user_id) = settings.user_id else {
        warn!("Terminal bridge disabled: unable to resolve Telegram owner user_id");
        return false;
    };

    let ws_url = settings.ws_url.clone();
    if let Err(err) = validate_daemon_ws_url(&ws_url) {
        warn!(
            error = %err,
            ws_url = %ws_url,
            "Terminal bridge disabled: insecure or invalid daemon websocket URL"
        );
        return false;
    }

    let shell = settings
        .shell
        .clone()
        .or_else(|| std::env::var("SHELL").ok())
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "/bin/bash".to_string());

    let default_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let inbox_dir = resolve_terminal_inbox_dir(&settings.inbox_dir, &default_cwd);
    let max_upload_bytes = usize::try_from(settings.max_upload_bytes).unwrap_or(usize::MAX);
    let device_id = settings.device_id.clone();

    let bot_tokens = settings.bot_tokens.clone();
    let static_token = settings.static_connect_token.clone();
    let allow_static_fallback = settings.allow_static_token_fallback;
    let auth = if !bot_tokens.is_empty() {
        let mint_url = match daemon_connect_token_mint_url(&ws_url) {
            Ok(url) => url,
            Err(err) => {
                warn!(
                    error = %err,
                    "Terminal bridge disabled: invalid daemon websocket URL for token minting"
                );
                return false;
            }
        };
        let fallback_static_token = if allow_static_fallback {
            static_token.clone()
        } else {
            None
        };
        if fallback_static_token.is_some() {
            warn!(
                "Terminal bridge auth mode: auto-bootstrap daemon token with static token fallback enabled"
            );
        } else if static_token.is_some() {
            info!(
                "Terminal bridge auth mode: auto-bootstrap daemon token; static token fallback disabled"
            );
        } else {
            info!("Terminal bridge auth mode: auto-bootstrap daemon token");
        }
        BridgeAuth::BotProof {
            mint_url,
            bot_tokens,
            fallback_static_token,
        }
    } else if let Some(token) = static_token {
        warn!("Terminal bridge auth mode: static daemon token (no bot-proof auto-bootstrap)");
        BridgeAuth::StaticToken(token)
    } else {
        info!(
            "Terminal bridge disabled: no daemon token configured and no Telegram bot token available for secure auto-bootstrap"
        );
        return false;
    };

    let review_profiles = build_review_profiles(&settings.review_agents);

    let runtime = RuntimeConfig {
        ws_url,
        user_id: user_id.to_string(),
        device_id: device_id.as_deref().map(sanitize_device_id),
        shell,
        default_cwd,
        inbox_dir,
        max_upload_bytes,
        review_profiles,
        auth,
    };

    if TERMINAL_BRIDGE_TASK_RUNNING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        info!("Terminal bridge already running; skipping duplicate startup");
        return true;
    }

    tokio::spawn(async move {
        match TerminalBridge::new(runtime, host, secrets).await {
            Ok(mut bridge) => bridge.run_forever().await,
            Err(err) => error!(error = %err, "Failed to initialize terminal bridge"),
        }
        TERMINAL_BRIDGE_TASK_RUNNING.store(false, Ordering::Release);
    });
    true
}
