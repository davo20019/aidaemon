//! The model step of first-run setup, the one step that must pass: find
//! what this machine already has (a ChatGPT sign-in, a provider key in the
//! environment), try each, or connect a provider the owner picks. Nothing
//! is saved until a test call answers; then the config is written through
//! `manage_config`'s own `switch_provider`, the same path a provider change
//! takes later, and the daemon starts with it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aidaemon_client_protocol::{
    Billing, ChannelOption, FoundModel, ProviderChoice, ProviderOption, SetupAction, SetupState,
    SetupStep, SignIn,
};
use aidaemon_secrets::SecretStore;
use serde_json::json;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

use crate::config::ProviderConfig;
use crate::tools::ConfigManagerTool;

/// The presets offered, in order: the subscription sign-in first (it bills
/// a plan the owner already has), then API-key providers. Local models are
/// not offered.
const OFFERED: &[&str] = &[
    "openai_chatgpt",
    "openai",
    "anthropic_native",
    "google_ai_studio",
    "openrouter",
    "xai",
    "deepseek",
    "moonshot",
    "minimax",
    "custom_openai_compatible",
];

/// Where to get each preset's key.
fn key_url(preset: &str) -> &'static str {
    match preset {
        "openai" => "https://platform.openai.com/api-keys",
        "anthropic_native" => "https://console.anthropic.com/settings/keys",
        "google_ai_studio" => "https://aistudio.google.com/apikey",
        "openrouter" => "https://openrouter.ai/keys",
        "xai" => "https://console.x.ai",
        "deepseek" => "https://platform.deepseek.com/api_keys",
        "moonshot" => "https://platform.moonshot.ai/",
        "minimax" => "https://platform.minimax.io/user-center/basic-information",
        _ => "",
    }
}

/// Provider keys a machine may already hold in its environment.
const ENV_KEYS: &[(&str, &str)] = &[
    ("OPENAI_API_KEY", "openai"),
    ("ANTHROPIC_API_KEY", "anthropic_native"),
    ("GEMINI_API_KEY", "google_ai_studio"),
    ("GOOGLE_API_KEY", "google_ai_studio"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("XAI_API_KEY", "xai"),
    ("DEEPSEEK_API_KEY", "deepseek"),
];

const CHATGPT: &str = "openai_chatgpt";

fn preset(id: &str) -> Option<aidaemon_runtime::tools::config_manager::PresetInfo> {
    aidaemon_runtime::tools::config_manager::provider_presets()
        .into_iter()
        .find(|p| p.id == id)
}

fn billing(preset: &str) -> Billing {
    if preset == CHATGPT {
        Billing::Plan
    } else {
        Billing::PerToken
    }
}

/// Every provider offered.
pub(crate) fn options() -> Vec<ProviderOption> {
    OFFERED
        .iter()
        .filter_map(|id| preset(id))
        .map(|p| ProviderOption {
            preset: p.id.to_string(),
            name: p.display_name.to_string(),
            billing: billing(p.id),
            needs_key: p.needs_api_key,
            key_url: key_url(p.id).to_string(),
            needs_base_url: p.requires_custom_base_url,
            model: p.primary.to_string(),
        })
        .collect()
}

/// The chats this build can connect.
pub(crate) fn channels() -> Vec<ChannelOption> {
    #[allow(unused_mut)]
    let mut out = vec![ChannelOption {
        platform: "telegram".into(),
        name: "Telegram".into(),
        help_url: "https://t.me/BotFather".into(),
        needs_app_token: false,
    }];
    #[cfg(feature = "slack")]
    out.push(ChannelOption {
        platform: "slack".into(),
        name: "Slack".into(),
        help_url: "https://api.slack.com/apps".into(),
        needs_app_token: true,
    });
    #[cfg(feature = "discord")]
    out.push(ChannelOption {
        platform: "discord".into(),
        name: "Discord".into(),
        help_url: "https://discord.com/developers/applications".into(),
        needs_app_token: false,
    });
    out
}

/// A model connection to try: a provider choice and, for one that takes a
/// key, the key.
#[derive(Clone)]
pub(crate) struct Candidate {
    pub choice: ProviderChoice,
    pub key: Option<String>,
}

impl Candidate {
    fn name(&self) -> String {
        let display = preset(&self.choice.preset)
            .map(|p| p.display_name.to_string())
            .unwrap_or_else(|| self.choice.preset.clone());
        let model = if self.choice.primary_model.is_empty() {
            preset(&self.choice.preset)
                .map(|p| p.primary.to_string())
                .unwrap_or_default()
        } else {
            self.choice.primary_model.clone()
        };
        if model.is_empty() {
            display
        } else {
            format!("{display} · {model}")
        }
    }
}

/// The config every setup starts from: the app's gateway on, the rest at
/// its defaults (which ask before anything is sent, posted, deleted or
/// spent).
pub(crate) const STARTER: &str = r#"# Written by first-run setup. Change it from the app's Settings page, or here.

[agent]
name = "aidaemon"

[client_gateway]
enabled = true

[state]
db_path = "aidaemon.db"
"#;

fn switch_call(choice: &ProviderChoice) -> serde_json::Value {
    json!({
        "action": "switch_provider",
        "provider": choice.preset,
        "base_url": choice.base_url,
        "primary_model": choice.primary_model,
        "fast_model": choice.fast_model,
        "smart_model": choice.smart_model,
    })
}

/// `STARTER` with `candidate`'s provider written in by `switch_provider`
/// at `path`, reading keys from `secrets`.
async fn write_with(
    path: &Path,
    candidate: &Candidate,
    secrets: Arc<dyn SecretStore>,
) -> Result<String, String> {
    std::fs::write(path, STARTER).map_err(|e| format!("Could not write the config: {e}"))?;
    let tool = ConfigManagerTool::new(crate::config::config_file(path), secrets);
    let reply = tool
        .run_owner_call(&switch_call(&candidate.choice), true)
        .await?;
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
    let written = table
        .get("provider")
        .and_then(|p| p.get("kind"))
        .and_then(toml::Value::as_str)
        .is_some();
    if written {
        Ok(text)
    } else {
        // `switch_provider` explains a refusal in its reply.
        Err(reply.split("\n\n").next().unwrap_or(&reply).to_string())
    }
}

/// A scratch directory, removed when dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Result<Self, String> {
        let path = std::env::temp_dir().join(format!("aidaemon-setup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
        Ok(Self(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The provider `candidate` makes, without saving anything.
async fn provider_of(candidate: &Candidate) -> Result<ProviderConfig, String> {
    let dir = TempDir::new()?;
    let secrets = aidaemon_secrets::MemorySecretStore::new();
    if let Some(key) = &candidate.key {
        secrets
            .set(
                &aidaemon_secrets::names::provider_field(None, "api_key"),
                key,
            )
            .map_err(|e| e.to_string())?;
    }
    let text = write_with(&dir.0.join("config.toml"), candidate, Arc::new(secrets)).await?;
    let table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
    let mut provider = table
        .get("provider")
        .and_then(toml::Value::as_table)
        .cloned()
        .ok_or("No provider was written")?;
    if let Some(key) = &candidate.key {
        provider.insert("api_key".into(), toml::Value::String(key.clone()));
    }
    toml::Value::Table(provider)
        .try_into()
        .map_err(|e| format!("The provider settings do not read back: {e}"))
}

/// Try `candidate` with one test call.
pub(crate) async fn test(candidate: &Candidate) -> Result<(), String> {
    let provider = provider_of(candidate).await?;
    crate::providers::connection_test::try_connection(&provider).await
}

/// Write the config for `candidate` (its key into the secret store first),
/// then the setup progress. Returns notices for the owner.
pub(crate) async fn save(
    config_path: &Path,
    candidate: &Candidate,
    on_server: bool,
) -> Result<Vec<String>, String> {
    let mut notices = Vec::new();
    if let Some(key) = &candidate.key {
        let name = aidaemon_secrets::names::provider_field(None, "api_key");
        crate::config::store_in_keychain(&name, key)
            .map_err(|e| format!("Could not store the key: {e}"))?;
        if let Some(note) = crate::config::secret_store().fallback_notice() {
            notices.push(note);
        }
    }
    let staged = config_path.with_extension("toml.setup");
    let written = write_with(&staged, candidate, crate::config::shared_secret_store()).await;
    if let Err(reason) = written {
        let _ = std::fs::remove_file(&staged);
        return Err(reason);
    }
    std::fs::rename(&staged, config_path).map_err(|e| format!("Could not save the config: {e}"))?;
    for leftover in [
        staged.with_extension("setup.bak"),
        staged.with_extension("setup.lastgood"),
    ] {
        let _ = std::fs::remove_file(leftover);
    }
    super::progress::save(
        config_path,
        &super::progress::Progress {
            on_server,
            model: candidate.name(),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Could not note the setup: {e}"))?;
    Ok(notices)
}

/// What the model step knows.
#[derive(Default)]
struct Inner {
    step: SetupStep,
    found: Vec<(FoundModel, Candidate)>,
    /// The sign-in waiting for a pasted address.
    pasted: Option<aidaemon_providers::chatgpt_login::LoginAttempt>,
    sign_in: Option<SignIn>,
    model: Option<String>,
    notices: Vec<String>,
    /// Set up from another device.
    on_server: bool,
}

/// The model step, as the setup gateway's host.
pub(crate) struct ModelSetup {
    config_path: PathBuf,
    inner: Mutex<Inner>,
    updates: watch::Sender<SetupState>,
    /// Cancelled once the config is written: setup mode ends.
    done: CancellationToken,
}

impl ModelSetup {
    pub(crate) fn new(config_path: PathBuf, done: CancellationToken) -> Arc<Self> {
        let (updates, _) = watch::channel(SetupState::default());
        Arc::new(Self {
            config_path,
            inner: Mutex::new(Inner::default()),
            updates,
            done,
        })
    }

    pub(crate) fn updates(&self) -> watch::Receiver<SetupState> {
        self.updates.subscribe()
    }

    fn changed(&self) {
        self.updates.send_modify(|s| s.notices.clear());
    }

    fn snapshot(inner: &Inner, local: bool) -> SetupState {
        SetupState {
            step: inner.step,
            found: inner.found.iter().map(|(f, _)| f.clone()).collect(),
            providers: options(),
            model: inner.model.clone(),
            sign_in: inner.sign_in.clone(),
            channels: channels(),
            channel: None,
            notices: inner.notices.clone(),
            on_server: !local,
        }
    }

    /// What this machine already has, each tried once.
    pub(crate) async fn detect(&self) {
        let mut candidates: Vec<(Candidate, String)> = Vec::new();
        if crate::providers::chatgpt_login::is_connected() {
            candidates.push((
                Candidate {
                    choice: ProviderChoice {
                        preset: CHATGPT.into(),
                        ..ProviderChoice::default()
                    },
                    key: None,
                },
                "ChatGPT sign-in on this computer".into(),
            ));
        }
        for (var, preset) in ENV_KEYS {
            let Some(key) = std::env::var(var)
                .ok()
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
            else {
                continue;
            };
            if candidates.iter().any(|(c, _)| c.choice.preset == *preset) {
                continue;
            }
            candidates.push((
                Candidate {
                    choice: ProviderChoice {
                        preset: (*preset).into(),
                        ..ProviderChoice::default()
                    },
                    key: Some(key),
                },
                format!("{var} in this computer's environment"),
            ));
        }
        let tests = candidates
            .into_iter()
            .map(|(candidate, source)| async move {
                let result = test(&candidate).await;
                (candidate, source, result)
            });
        let found = futures::future::join_all(tests)
            .await
            .into_iter()
            .map(|(candidate, source, result)| {
                (
                    FoundModel {
                        preset: candidate.choice.preset.clone(),
                        name: candidate.name(),
                        source,
                        billing: billing(&candidate.choice.preset),
                        works: result.is_ok(),
                        detail: result.err().unwrap_or_default(),
                    },
                    candidate,
                )
            })
            .collect();
        self.inner.lock().await.found = found;
        self.changed();
    }

    /// `candidate` passed its test (or is about to be tested): save it and
    /// end setup mode.
    async fn connect(&self, candidate: Candidate, tested: bool) -> Result<(), String> {
        if !tested {
            test(&candidate).await?;
        }
        let on_server = self.inner.lock().await.on_server;
        let notices = save(&self.config_path, &candidate, on_server).await?;
        {
            let mut inner = self.inner.lock().await;
            inner.step = SetupStep::Starting;
            inner.model = Some(candidate.name());
            inner.sign_in = None;
            inner.notices = notices;
        }
        self.changed();
        // Let the answer reach the app before the gateway restarts.
        let done = self.done.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            done.cancel();
        });
        Ok(())
    }

    fn chatgpt() -> Candidate {
        Candidate {
            choice: ProviderChoice {
                preset: CHATGPT.into(),
                ..ProviderChoice::default()
            },
            key: None,
        }
    }

    async fn sign_in(self: &Arc<Self>, paste: bool) -> Result<(), String> {
        let manager = crate::providers::chatgpt_login::shared_credential_manager();
        let mut notice = None;
        let attempt = match manager.start_login(!paste).await {
            Ok(attempt) => attempt,
            Err(error) if !paste => {
                notice = Some(format!(
                    "{error}. Sign in on any device and paste the address the browser ends on."
                ));
                manager
                    .start_login(false)
                    .await
                    .map_err(|e| e.to_string())?
            }
            Err(error) => return Err(error.to_string()),
        };
        let sign_in = SignIn {
            url: attempt.url.clone(),
            paste: !attempt.on_loopback(),
        };
        {
            let mut inner = self.inner.lock().await;
            inner.sign_in = Some(sign_in);
            inner.notices = notice.into_iter().collect();
            inner.pasted = None;
            if !attempt.on_loopback() {
                inner.pasted = Some(attempt);
                return Ok(());
            }
        }
        let this = self.clone();
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            let result = manager.finish_on_loopback(&client, attempt).await;
            let outcome = match result {
                Ok(_) => this.connect(Self::chatgpt(), false).await,
                Err(error) => Err(error.to_string()),
            };
            if let Err(reason) = outcome {
                let mut inner = this.inner.lock().await;
                inner.sign_in = None;
                inner.notices = vec![format!("The sign-in did not finish: {reason}")];
            }
            this.changed();
        });
        Ok(())
    }

    async fn finish_sign_in(&self, redirect: &str) -> Result<(), String> {
        let attempt = self
            .inner
            .lock()
            .await
            .pasted
            .take()
            .ok_or("There is no sign-in waiting for an address; start it again")?;
        let manager = crate::providers::chatgpt_login::shared_credential_manager();
        let client = reqwest::Client::new();
        if let Err(error) = manager
            .finish_with_redirect(&client, &attempt, redirect)
            .await
        {
            // The same sign-in may be pasted again.
            self.inner.lock().await.pasted = Some(attempt);
            return Err(format!("That did not finish the sign-in: {error}"));
        }
        self.connect(Self::chatgpt(), false).await
    }

    async fn act(self: &Arc<Self>, action: SetupAction, local: bool) -> Result<(), String> {
        if !local {
            self.inner.lock().await.on_server = true;
        }
        if self.inner.lock().await.step == SetupStep::Starting {
            return Err("A model is connected; the daemon is starting with it".into());
        }
        match action {
            SetupAction::Detect => {
                self.detect().await;
                Ok(())
            }
            SetupAction::UseFound { index } => {
                let found = self
                    .inner
                    .lock()
                    .await
                    .found
                    .get(index as usize)
                    .cloned()
                    .ok_or("There is no such model")?;
                if !found.0.works {
                    return Err(format!("That one did not answer: {}", found.0.detail));
                }
                self.connect(found.1, true).await
            }
            SetupAction::SignIn { paste } => self.sign_in(paste || !local).await,
            SetupAction::FinishSignIn { redirect } => self.finish_sign_in(&redirect).await,
            SetupAction::ConnectModel { provider, key } => {
                let preset = preset(&provider.preset).ok_or("Unknown provider")?;
                let key = key
                    .map(|k| k.0.trim().to_string())
                    .filter(|k| !k.is_empty());
                if preset.needs_api_key && key.is_none() {
                    return Err(format!("{} needs an API key", preset.display_name));
                }
                if preset.id == CHATGPT {
                    return Err("Sign in to ChatGPT instead of entering a key".into());
                }
                self.connect(
                    Candidate {
                        choice: provider,
                        key,
                    },
                    false,
                )
                .await
            }
            SetupAction::ConnectChannel { .. } | SetupAction::SkipChannel | SetupAction::Finish => {
                Err("Connect a model first".into())
            }
            SetupAction::Unknown => Err("Unknown setup step".into()),
        }
    }
}

/// The model step as the setup gateway's host.
pub(crate) struct Host(pub Arc<ModelSetup>);

#[async_trait::async_trait]
impl aidaemon_client_gateway::setup::SetupHost for Host {
    async fn state(&self, local: bool) -> SetupState {
        let inner = self.0.inner.lock().await;
        ModelSetup::snapshot(&inner, local && !inner.on_server)
    }

    async fn step(&self, action: SetupAction, local: bool) -> Result<SetupState, String> {
        self.0.act(action, local).await?;
        Ok(self.state(local).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_client_gateway::setup::SetupHost;

    #[test]
    fn the_subscription_comes_first_and_local_models_are_not_offered() {
        let options = options();
        assert_eq!(options[0].preset, CHATGPT);
        assert_eq!(options[0].billing, Billing::Plan);
        assert!(!options[0].needs_key);
        assert!(options.iter().all(|o| o.preset != "ollama"));
        assert!(
            options
                .iter()
                .filter(|o| o.preset != CHATGPT)
                .all(|o| o.billing == Billing::PerToken)
        );
        assert!(options.iter().any(|o| o.preset == "openai" && o.needs_key));
    }

    /// An OpenAI-compatible endpoint that answers key `sk-synthetic-good`
    /// and refuses any other.
    async fn fake_provider() -> String {
        use axum::http::{HeaderMap, StatusCode};
        use axum::response::IntoResponse;
        async fn complete(headers: HeaderMap) -> axum::response::Response {
            let good = headers.get("authorization").and_then(|v| v.to_str().ok())
                == Some("Bearer sk-synthetic-good");
            if !good {
                return (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(json!({"error": {"message": "bad key", "type": "invalid_api_key"}})),
                )
                    .into_response();
            }
            let chunk = |delta: serde_json::Value, finish: serde_json::Value| {
                json!({"id": "c1", "object": "chat.completion.chunk", "created": 1,
                       "model": "synthetic-model",
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            };
            let body = format!(
                "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                chunk(
                    json!({"role": "assistant", "content": "ready"}),
                    json!(null)
                ),
                chunk(json!({}), json!("stop")),
            );
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                body,
            )
                .into_response()
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(complete));
        tokio::spawn(async move { axum::serve(listener, app).await });
        base
    }

    #[tokio::test]
    async fn nothing_is_saved_until_the_model_answers_then_the_daemon_can_start_with_it() {
        let isolation = crate::testing::KeychainIsolation::acquire();
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let base_url = fake_provider().await;
        let done = CancellationToken::new();
        let setup = ModelSetup::new(config_path.clone(), done.clone());
        let connect = |key: &str| SetupAction::ConnectModel {
            provider: ProviderChoice {
                preset: "custom_openai_compatible".into(),
                base_url: base_url.clone(),
                primary_model: "synthetic-model".into(),
                ..ProviderChoice::default()
            },
            key: Some(aidaemon_client_protocol::SecretText(key.into())),
        };

        let refused = setup
            .act(connect("sk-synthetic-bad"), true)
            .await
            .unwrap_err();
        assert!(refused.contains("did not accept"), "{refused}");
        assert!(!config_path.exists(), "a failed test saves nothing");
        assert!(!done.is_cancelled());

        setup.act(connect("sk-synthetic-good"), true).await.unwrap();
        let state = Host(setup.clone()).state(true).await;
        assert_eq!(state.step, SetupStep::Starting);
        assert!(state.model.unwrap().contains("synthetic-model"));
        tokio::time::timeout(std::time::Duration::from_secs(5), done.cancelled())
            .await
            .expect("setup mode ends once the config is written");

        // The key went to the secret store, not the file, and the config
        // loads as the daemon loads it.
        let text = std::fs::read_to_string(&config_path).unwrap();
        assert!(!text.contains("sk-synthetic-good"), "{text}");
        let config = crate::config::AppConfig::load(&config_path).unwrap();
        assert_eq!(config.provider.api_key, "sk-synthetic-good");
        assert!(config.client_gateway.enabled);
        let progress = super::super::progress::load(&config_path).unwrap();
        assert!(!progress.channel_done && !progress.on_server);
        drop(isolation);
    }

    #[tokio::test]
    async fn a_candidate_reads_back_as_the_provider_the_daemon_will_build() {
        let candidate = Candidate {
            choice: ProviderChoice {
                preset: "openrouter".into(),
                primary_model: "synthetic/model-1".into(),
                ..ProviderChoice::default()
            },
            key: Some("sk-synthetic-not-a-key".into()),
        };
        let provider = provider_of(&candidate).await.unwrap();
        assert_eq!(provider.api_key, "sk-synthetic-not-a-key");
        assert_eq!(provider.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(provider.models.primary, "synthetic/model-1");
        // Without its key, a key provider is refused before any call.
        let keyless = Candidate {
            key: None,
            ..candidate
        };
        assert!(provider_of(&keyless).await.is_err());
    }
}
