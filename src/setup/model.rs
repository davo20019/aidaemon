//! The model step of first-run setup, the one step that must pass: find
//! what this machine already has (a ChatGPT sign-in, a provider key in the
//! environment), try each, or connect a provider the owner picks. Nothing
//! is saved until a test call answers; then the config is written through
//! `manage_config`'s own `switch_provider`, the same path a provider change
//! takes later, and the daemon starts with it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aidaemon_client_protocol::{
    Billing, ChannelOption, CheckedKey, FoundModel, ProviderChoice, ProviderOption, SetupAction,
    SetupState, SetupStep, SignIn,
};
use aidaemon_secrets::SecretStore;
use serde_json::json;
use tokio::sync::{Mutex, watch};
use tokio_util::sync::CancellationToken;

use crate::config::ProviderConfig;
use crate::tools::ConfigManagerTool;

use aidaemon_providers::catalog::{self, CatalogEntry};

const CHATGPT: &str = "openai_chatgpt";

/// What setup offers from the SDK's provider catalog: the hosted
/// providers (the subscription sign-in first, as the catalog orders them)
/// and a model server on this machine (Ollama), without the advanced
/// routes (a Cloudflare gateway, Anthropic through OpenRouter) a Settings
/// change still reaches.
fn offered(entry: &CatalogEntry) -> bool {
    !entry.gateway_token && entry.id != "anthropic_openrouter"
}

/// A provider that runs on the owner's machine and takes no key.
fn local(entry: &CatalogEntry) -> bool {
    entry.billing == catalog::Billing::Local
}

/// The key a custom endpoint is called with when the owner gave none: a
/// self-hosted server (llama.cpp, LM Studio, vLLM) ignores it.
const NO_KEY: &str = "none";

/// Whether `entry` may be connected with no key: the owner's own endpoint.
fn key_optional(entry: &CatalogEntry) -> bool {
    entry.needs_api_key() && entry.custom_base_url && !entry.gateway_token
}

/// A local provider's name without its "(local)".
fn local_name(entry: &CatalogEntry) -> &'static str {
    entry
        .display_name
        .split(" (")
        .next()
        .unwrap_or(entry.display_name)
}

pub(crate) fn preset(id: &str) -> Option<&'static CatalogEntry> {
    catalog::find(id)
}

fn billing(entry: &CatalogEntry) -> Billing {
    match entry.billing {
        catalog::Billing::Plan => Billing::Plan,
        catalog::Billing::Local => Billing::Local,
        _ => Billing::PerToken,
    }
}

/// Every provider offered.
pub(crate) fn options() -> Vec<ProviderOption> {
    catalog::CATALOG
        .iter()
        .filter(|entry| offered(entry))
        .map(|entry| ProviderOption {
            preset: entry.id.to_string(),
            name: entry.display_name.to_string(),
            billing: billing(entry),
            needs_key: entry.needs_api_key(),
            key_url: entry.key_url().to_string(),
            needs_base_url: entry.custom_base_url,
            model: entry.models.primary.to_string(),
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
                .map(|p| p.models.primary.to_string())
                .unwrap_or_default()
        } else {
            self.choice.primary_model.clone()
        };
        // llama.cpp names a model by its file's path: the file says enough.
        let model = match model.rsplit_once(['/', '\\']) {
            Some((_, file)) if model.ends_with(".gguf") => file.to_string(),
            _ => model,
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
    // A custom endpoint's preset, or a local server's, names no models:
    // the one the owner named does every job unless they name others.
    let custom = preset(&choice.preset).is_some_and(|p| p.models.primary.is_empty());
    let or_primary = |model: &str| {
        if model.is_empty() && custom {
            choice.primary_model.clone()
        } else {
            model.to_string()
        }
    };
    json!({
        "action": "switch_provider",
        "provider": choice.preset,
        "base_url": choice.base_url,
        "primary_model": choice.primary_model,
        "fast_model": or_primary(&choice.fast_model),
        "smart_model": or_primary(&choice.smart_model),
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

/// Try `candidate` with one test call: the provider built from the
/// config it would write, the way the daemon will build it, asked by the
/// SDK's connection test. Nothing else exists yet in setup mode (no stores,
/// no side-call gateway), and nothing is saved until it answers.
pub(crate) async fn test(candidate: &Candidate) -> Result<(), String> {
    let config = provider_of(candidate).await?;
    let (provider, _) = crate::providers::factory::build_provider(&config)
        .map_err(|error| format!("Could not set up the provider: {error}"))?;
    let model = crate::router::Router::new(config.models.clone())
        .default_model()
        .to_string();
    aidaemon_providers::catalog::try_connection(provider.as_ref(), &model).await
}

/// Check `key` with `entry`'s provider (nothing is saved) and read the
/// models its account offers.
pub(crate) async fn check_key(
    entry: &'static CatalogEntry,
    base_url: &str,
    key: &str,
) -> Result<CheckedKey, String> {
    let base = if base_url.trim().is_empty() {
        entry.base_url
    } else {
        base_url.trim()
    };
    let keyless = local(entry) || (key_optional(entry) && key.is_empty());
    let key = if key.is_empty() && key_optional(entry) {
        NO_KEY
    } else {
        key
    };
    if !keyless
        && let Some(endpoint) = aidaemon_providers::key_check::KeyEndpoint::new(entry.kind, base)
    {
        use aidaemon_providers::key_check::KeyCheck;
        if let KeyCheck::Rejected(detail) =
            aidaemon_providers::key_check::check_key(&endpoint, key).await
        {
            return Err(format!("The provider did not accept that key ({detail})"));
        }
    }
    let models = catalog::list_models(entry.kind, base, key)
        .await
        .map_err(|why| {
            if keyless {
                format!(
                    "No model server answered at {base} ({why}). Is it running there? If it uses \
                     another port, enter its address."
                )
            } else {
                format!("The key works, but its models could not be listed: {why}")
            }
        })?;
    if keyless && models.is_empty() {
        return Err(format!(
            "{} is running but has no models yet. Pull one first.",
            entry.display_name
        ));
    }
    Ok(CheckedKey {
        preset: entry.id.to_string(),
        suggested: catalog::preselect(&models, entry.models.primary),
        models,
    })
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
    // A setup on another port (a second daemon) keeps it.
    if let Some(port) = std::env::var("AIDAEMON_GATEWAY_PORT")
        .ok()
        .and_then(|p| p.trim().parse::<i64>().ok())
    {
        let text = std::fs::read_to_string(&staged).map_err(|e| e.to_string())?;
        let mut table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
        if let Some(gateway) = table
            .get_mut("client_gateway")
            .and_then(toml::Value::as_table_mut)
        {
            gateway.insert("port".into(), toml::Value::Integer(port));
        }
        std::fs::write(
            &staged,
            toml::to_string_pretty(&table).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    std::fs::rename(&staged, config_path).map_err(|e| format!("Could not save the config: {e}"))?;
    // `switch_provider` backs up the staged file before it writes it.
    for leftover in [
        staged.with_extension("toml.bak"),
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
    /// The sign-in a pasted address may finish (its loopback wait, if any,
    /// holds it too).
    pasted: Option<Arc<aidaemon_providers::chatgpt_login::LoginAttempt>>,
    /// The sign-in under way, which another "Sign in" joins.
    attempt: Option<Arc<aidaemon_providers::chatgpt_login::LoginAttempt>>,
    /// The loopback wait, stopped when its sign-in ends another way.
    waiting: Option<tokio::task::AbortHandle>,
    /// Bumped by each sign-in started or finished, so a loopback wait that
    /// outlived its sign-in changes nothing.
    generation: u64,
    sign_in: Option<SignIn>,
    /// A key checked, with its account's models.
    checked: Option<CheckedKey>,
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
            checked: inner.checked.clone(),
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
        // Keys in the environment, each with a model its account offers.
        let mut unusable: Vec<FoundModel> = Vec::new();
        for found in catalog::keys_in_env() {
            if !offered(found.entry) {
                continue;
            }
            let source = format!("{} in this computer's environment", found.var);
            let model = match catalog::list_models(
                found.entry.kind,
                found.entry.base_url,
                &found.key,
            )
            .await
            {
                Ok(models) => catalog::preselect(&models, found.entry.models.primary),
                Err(_) => None,
            };
            let Some(model) = model else {
                unusable.push(FoundModel {
                    preset: found.entry.id.to_string(),
                    name: found.entry.display_name.to_string(),
                    source,
                    billing: billing(found.entry),
                    works: false,
                    detail: "connect it below to pick one of its models".into(),
                });
                continue;
            };
            candidates.push((
                Candidate {
                    choice: ProviderChoice {
                        preset: found.entry.id.into(),
                        primary_model: model,
                        ..ProviderChoice::default()
                    },
                    key: Some(found.key),
                },
                source,
            ));
        }
        // A model server on this machine, with the models it has.
        for entry in catalog::CATALOG.iter().filter(|e| local(e) && offered(e)) {
            let Ok(models) = catalog::list_models(entry.kind, entry.base_url, "").await else {
                continue;
            };
            let Some(model) = catalog::preselect(&models, "").or_else(|| models.first().cloned())
            else {
                continue;
            };
            candidates.push((
                Candidate {
                    choice: ProviderChoice {
                        preset: entry.id.into(),
                        primary_model: model,
                        ..ProviderChoice::default()
                    },
                    key: None,
                },
                format!("{} running on this computer", local_name(entry)),
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
                        billing: preset(&candidate.choice.preset)
                            .map(billing)
                            .unwrap_or(Billing::PerToken),
                        works: result.is_ok(),
                        detail: result.err().unwrap_or_default(),
                    },
                    candidate,
                )
            })
            .collect();
        let mut inner = self.inner.lock().await;
        inner.found = found;
        inner.found.extend(unusable.into_iter().map(|f| {
            (
                f,
                Candidate {
                    choice: ProviderChoice::default(),
                    key: None,
                },
            )
        }));
        drop(inner);
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

    /// Start a ChatGPT sign-in. It waits on this machine's loopback port,
    /// so a browser here finishes it by itself; with `paste` (the owner is
    /// on another device), or when no port can be bound, the address the
    /// browser ends on finishes it too.
    async fn sign_in(self: &Arc<Self>, paste: bool) -> Result<(), String> {
        // One under way is joined, not replaced (another device's press
        // would cancel the page its owner is signing in on).
        {
            let mut inner = self.inner.lock().await;
            if let Some(attempt) = inner.attempt.clone()
                && let Some(sign_in) = inner.sign_in.as_mut()
            {
                if paste && !sign_in.paste {
                    sign_in.paste = true;
                    inner.pasted = Some(attempt);
                }
                drop(inner);
                self.changed();
                return Ok(());
            }
        }
        let manager = crate::providers::chatgpt_login::shared_credential_manager();
        let mut notice = None;
        let attempt = match manager.start_login(true).await {
            Ok(attempt) => attempt,
            Err(error) => {
                notice = Some(format!(
                    "{error}. Sign in on any device and paste the address the browser ends on."
                ));
                manager
                    .start_login(false)
                    .await
                    .map_err(|e| e.to_string())?
            }
        };
        let attempt = Arc::new(attempt);
        let loopback = attempt.on_loopback();
        let paste = paste || !loopback;
        let generation = {
            let mut inner = self.inner.lock().await;
            inner.generation += 1;
            if let Some(waiting) = inner.waiting.take() {
                waiting.abort();
            }
            inner.sign_in = Some(SignIn {
                url: attempt.url.clone(),
                paste,
            });
            inner.notices = notice.into_iter().collect();
            inner.pasted = paste.then(|| attempt.clone());
            inner.attempt = Some(attempt.clone());
            inner.generation
        };
        if !loopback {
            return Ok(());
        }
        let this = self.clone();
        let task = tokio::spawn(async move {
            let client = reqwest::Client::new();
            let result = manager.finish_on_loopback(&client, &attempt).await;
            {
                let mut inner = this.inner.lock().await;
                if inner.generation != generation {
                    return;
                }
                inner.generation += 1;
                inner.waiting = None;
                inner.pasted = None;
                inner.attempt = None;
            }
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
        let mut inner = self.inner.lock().await;
        if inner.generation == generation {
            inner.waiting = Some(task.abort_handle());
        }
        Ok(())
    }

    async fn finish_sign_in(&self, redirect: &str) -> Result<(), String> {
        let (attempt, generation) = {
            let inner = self.inner.lock().await;
            let attempt = inner
                .pasted
                .clone()
                .ok_or("There is no sign-in waiting for an address; start it again")?;
            (attempt, inner.generation)
        };
        let manager = crate::providers::chatgpt_login::shared_credential_manager();
        let client = reqwest::Client::new();
        // The same sign-in may be pasted again after a failure.
        if let Err(error) = manager
            .finish_with_redirect(&client, &attempt, redirect)
            .await
        {
            return Err(format!("That did not finish the sign-in: {error}"));
        }
        {
            let mut inner = self.inner.lock().await;
            if inner.generation == generation {
                inner.generation += 1;
                inner.pasted = None;
                inner.attempt = None;
                if let Some(waiting) = inner.waiting.take() {
                    waiting.abort();
                }
            }
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
            SetupAction::CheckKey { provider, key } => {
                let entry = preset(&provider.preset).ok_or("Unknown provider")?;
                let checked = check_key(entry, &provider.base_url, key.0.trim()).await?;
                self.inner.lock().await.checked = Some(checked);
                Ok(())
            }
            SetupAction::FinishSignIn { redirect } => self.finish_sign_in(&redirect).await,
            SetupAction::ConnectModel { provider, key } => {
                let preset = preset(&provider.preset).ok_or("Unknown provider")?;
                let key = key
                    .map(|k| k.0.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .or_else(|| key_optional(preset).then(|| NO_KEY.to_string()));
                if preset.needs_api_key() && key.is_none() {
                    return Err(format!("{} needs an API key", preset.display_name));
                }
                if provider.primary_model.trim().is_empty() && preset.models.primary.is_empty() {
                    return Err("Pick a model".into());
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
    fn the_subscription_comes_first_and_a_local_server_is_offered_without_a_key() {
        let options = options();
        assert_eq!(options[0].preset, CHATGPT);
        assert_eq!(options[0].billing, Billing::Plan);
        assert!(!options[0].needs_key);
        for local in ["ollama", "llamacpp"] {
            let option = options.iter().find(|o| o.preset == local).unwrap();
            assert_eq!(option.billing, Billing::Local);
            assert!(!option.needs_key);
        }
        assert!(
            options
                .iter()
                .filter(|o| o.preset != CHATGPT && o.billing != Billing::Local)
                .all(|o| o.billing == Billing::PerToken)
        );
        assert!(options.iter().any(|o| o.preset == "openai" && o.needs_key));
    }

    #[tokio::test]
    async fn a_local_server_and_the_owners_own_endpoint_connect_without_a_key() {
        // Ollama: no key, and the model named does every job.
        let ollama = provider_of(&Candidate {
            choice: ProviderChoice {
                preset: "ollama".into(),
                primary_model: "synthetic-local".into(),
                ..ProviderChoice::default()
            },
            key: None,
        })
        .await
        .unwrap();
        assert_eq!(ollama.base_url, "http://localhost:11434/v1");
        assert_eq!(ollama.models.primary, "synthetic-local");
        assert_eq!(ollama.models.fast, "synthetic-local");
        // llama.cpp: its default address unless the owner names one.
        let llama = |base_url: &str| Candidate {
            choice: ProviderChoice {
                preset: "llamacpp".into(),
                base_url: base_url.into(),
                primary_model: "synthetic-local".into(),
                ..ProviderChoice::default()
            },
            key: None,
        };
        assert_eq!(
            Candidate {
                choice: ProviderChoice {
                    preset: "llamacpp".into(),
                    primary_model: "/home/synthetic/models/synthetic-2b.gguf".into(),
                    ..ProviderChoice::default()
                },
                key: None,
            }
            .name(),
            "llama.cpp (local) · synthetic-2b.gguf"
        );
        let default = provider_of(&llama("")).await.unwrap();
        assert_eq!(default.base_url, "http://localhost:8080/v1");
        let moved = provider_of(&llama("http://localhost:8081/v1"))
            .await
            .unwrap();
        assert_eq!(moved.base_url, "http://localhost:8081/v1");
        assert!(!key_optional(preset("llama.cpp").unwrap()));
        // The owner's endpoint with no key: its models are listed.
        let base = fake_provider().await;
        let custom = preset("custom_openai_compatible").unwrap();
        let checked = check_key(custom, &base, "").await.unwrap();
        assert!(checked.models.contains(&"synthetic-model".to_string()));
        // A hosted provider still needs its key.
        assert!(!key_optional(preset("openai").unwrap()));
        // Nothing listening: the answer says where it looked.
        let nothing = check_key(custom, "http://127.0.0.1:9/v1", "").await;
        assert!(nothing.unwrap_err().contains("No model server answered"));
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
        async fn models() -> axum::Json<serde_json::Value> {
            axum::Json(json!({"object": "list", "data": [
                {"id": "synthetic-other"}, {"id": "synthetic-model"}
            ]}))
        }
        let app = axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(complete))
            .route("/v1/models", axum::routing::get(models));
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

        // A key is checked and the account's models read before choosing.
        setup
            .act(
                SetupAction::CheckKey {
                    provider: ProviderChoice {
                        preset: "custom_openai_compatible".into(),
                        base_url: base_url.clone(),
                        ..ProviderChoice::default()
                    },
                    key: aidaemon_client_protocol::SecretText("sk-synthetic-good".into()),
                },
                true,
            )
            .await
            .unwrap();
        let checked = Host(setup.clone()).state(true).await.checked.unwrap();
        assert_eq!(checked.models, ["synthetic-other", "synthetic-model"]);
        assert_eq!(
            checked.suggested, None,
            "a custom endpoint has no usual model"
        );
        assert!(!config_path.exists());

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
        // A custom endpoint's one model does every job.
        assert_eq!(config.provider.models.fast, "synthetic-model");
        assert!(!dir.path().join("config.toml.toml.bak").exists());
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
