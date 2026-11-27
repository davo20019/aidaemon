//! The Providers tab of the owner's apps, on the daemon: the providers the
//! config uses (main first, then each failover), how each is signed in, a
//! key checked against its provider before it is saved, and the ChatGPT
//! subscription sign-in. The config itself changes only through
//! `manage_config` (`client_settings`); this module holds what the tab
//! shows besides the file and the sign-in under way.

use std::sync::Arc;

use aidaemon_client_protocol::{
    CheckedKey, ConnectedProvider, CredentialState, ProviderChoice, SignIn,
};
use aidaemon_providers::catalog::{self, CatalogEntry};
use aidaemon_secrets::SecretStore as _;
use tokio::sync::Mutex;

const CHATGPT_KIND: &str = "openai_chatgpt";

/// `url` without a trailing slash, lower-cased.
fn plain_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_ascii_lowercase()
}

/// The catalog entry a provider table was written from: the same kind and
/// address (an entry with no address of its own matches a table with
/// none). ChatGPT is the one entry of its kind, whatever its address.
fn entry_of(kind: &str, base_url: &str) -> Option<&'static CatalogEntry> {
    let of_kind = || {
        catalog::CATALOG
            .iter()
            .filter(move |entry| entry.kind_name() == kind)
    };
    if kind == CHATGPT_KIND {
        return of_kind().next();
    }
    let base = plain_url(base_url);
    of_kind()
        .find(|entry| !entry.custom_base_url && plain_url(entry.base_url) == base)
        .or_else(|| {
            // Written with the address left out: the provider's own.
            (base.is_empty())
                .then(|| of_kind().find(|entry| !entry.custom_base_url))
                .flatten()
        })
        .or_else(|| {
            // An address the owner named: the entry that takes one (a
            // Cloudflare gateway by its host).
            let gateway = host_of(&base) == "gateway.ai.cloudflare.com";
            of_kind().find(|entry| entry.custom_base_url && entry.gateway_token == gateway)
        })
}

/// The host part of `url`, for naming an endpoint no preset matches.
fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    rest.split(['/', ':']).next().unwrap_or(rest)
}

/// The models a provider table names: its default first, then fast,
/// smart and the fallbacks, each once.
fn models_of(table: &toml::Table) -> Vec<String> {
    let Some(models) = table.get("models").and_then(toml::Value::as_table) else {
        return Vec::new();
    };
    let mut out: Vec<String> = Vec::new();
    let mut push = |model: &str| {
        let model = model.trim();
        if !model.is_empty() && !out.iter().any(|m| m == model) {
            out.push(model.to_string());
        }
    };
    for name in ["default", "primary", "fast", "smart"] {
        if let Some(model) = models.get(name).and_then(toml::Value::as_str) {
            push(model);
        }
    }
    if let Some(list) = models.get("fallback").and_then(toml::Value::as_array) {
        for model in list.iter().filter_map(toml::Value::as_str) {
            push(model);
        }
    }
    out
}

/// The models a provider table uses for the hardest work and for quick
/// jobs, as the daemon fills them (a job left unnamed takes the catalog's
/// model for that provider). Empty for an unknown kind.
fn jobs_of(table: &toml::Table, kind: &str) -> (String, String) {
    let Ok(kind) = toml::Value::String(kind.to_string()).try_into::<crate::config::ProviderKind>()
    else {
        return Default::default();
    };
    let mut models: crate::config::ModelsConfig = table
        .get("models")
        .cloned()
        .and_then(|value| value.try_into().ok())
        .unwrap_or_default();
    models.apply_defaults(&kind);
    (models.smart, models.fast)
}

/// The models a provider's catalog entry offers; none when no entry
/// matches or its models depend on the address.
fn offered_of(entry: Option<&CatalogEntry>) -> Vec<String> {
    entry
        .map(|entry| entry.offered.models())
        .unwrap_or_default()
        .iter()
        .map(|model| model.to_string())
        .collect()
}

/// One provider table as the tab shows it. `failover`: its index, or
/// `None` for the main provider. `stored`: whether a secret is stored under
/// a keychain name; `signed_in`: whether ChatGPT is.
fn connected_of(
    table: &toml::Table,
    failover: Option<u32>,
    stored: &dyn Fn(&str) -> bool,
    signed_in: bool,
) -> ConnectedProvider {
    let text = |key: &str| {
        table
            .get(key)
            .and_then(toml::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string()
    };
    let kind = text("kind");
    let base_url = text("base_url");
    let entry = entry_of(&kind, &base_url);
    let key_setting = match failover {
        None => "provider.api_key".to_string(),
        Some(index) => format!("provider.fallbacks[{index}].api_key"),
    };
    let name = match entry {
        Some(entry) if entry.custom_base_url && !base_url.is_empty() => {
            format!("{} · {}", entry.display_name, host_of(&base_url))
        }
        Some(entry) => entry.display_name.to_string(),
        None if !base_url.is_empty() => format!("{kind} · {}", host_of(&base_url)),
        None if kind.is_empty() => "Not set".to_string(),
        None => kind.clone(),
    };
    let api_key = text("api_key");
    let credential = if kind == CHATGPT_KIND {
        if signed_in {
            CredentialState::SignedIn
        } else {
            CredentialState::SignedOut
        }
    } else if api_key == "keychain" {
        let stored_now =
            crate::config::keychain_name_for(&key_setting).is_some_and(|name| stored(&name));
        if stored_now {
            CredentialState::KeyStored
        } else {
            CredentialState::KeyMissing
        }
    } else if !api_key.is_empty() && api_key != "ollama" {
        CredentialState::KeyInFile
    } else if entry.is_some_and(|entry| !entry.needs_api_key()) {
        CredentialState::NotNeeded
    } else {
        CredentialState::KeyMissing
    };
    let (hard_model, quick_model) = jobs_of(table, &kind);
    ConnectedProvider {
        failover,
        preset: entry.map(|e| e.id.to_string()).unwrap_or_default(),
        name,
        base_url,
        models: models_of(table),
        hard_model,
        quick_model,
        offered_models: offered_of(entry),
        credential,
        key_setting: if kind == CHATGPT_KIND {
            String::new()
        } else {
            key_setting
        },
        usage: None,
    }
}

/// The providers `config` uses: the main one, then each failover in order.
pub(crate) fn connected(config: &toml::Table) -> Vec<ConnectedProvider> {
    let store = crate::config::secret_store();
    let stored = |name: &str| store.contains(name);
    connected_with(config, &stored, chatgpt_signed_in())
}

/// The providers `config` uses, as [`connected`] lists them, for their
/// names and presets only: no credential is looked up, so each reads as
/// signed out or missing its key.
pub(crate) fn listed(config: &toml::Table) -> Vec<ConnectedProvider> {
    connected_with(config, &|_| false, false)
}

fn connected_with(
    config: &toml::Table,
    stored: &dyn Fn(&str) -> bool,
    signed_in: bool,
) -> Vec<ConnectedProvider> {
    let Some(provider) = config.get("provider").and_then(toml::Value::as_table) else {
        return Vec::new();
    };
    let mut out = vec![connected_of(provider, None, stored, signed_in)];
    let fallbacks = provider
        .get("fallbacks")
        .or_else(|| provider.get("failover"))
        .and_then(toml::Value::as_array);
    for (index, table) in fallbacks
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_table)
        .enumerate()
    {
        out.push(connected_of(table, Some(index as u32), stored, signed_in));
    }
    out
}

/// How many failovers `config` has: the index the next one takes.
pub(crate) fn failover_count(config: &toml::Table) -> usize {
    config
        .get("provider")
        .and_then(|p| p.get("fallbacks").or_else(|| p.get("failover")))
        .and_then(toml::Value::as_array)
        .map_or(0, Vec::len)
}

pub(crate) fn chatgpt_signed_in() -> bool {
    crate::providers::chatgpt_login::is_connected()
}

/// A sign-in or key check under way from the Providers tab.
#[derive(Default)]
struct FlowState {
    sign_in: Option<SignIn>,
    /// The sign-in a pasted address may finish (its loopback wait, if any,
    /// holds it too).
    pasted: Option<Arc<aidaemon_providers::chatgpt_login::LoginAttempt>>,
    /// The sign-in under way, which another "Sign in" joins.
    attempt: Option<Arc<aidaemon_providers::chatgpt_login::LoginAttempt>>,
    /// The loopback wait, stopped when the sign-in ends another way.
    waiting: Option<tokio::task::AbortHandle>,
    checked: Option<CheckedKey>,
    /// Bumped by each sign-in started, finished or cancelled, so a loopback
    /// wait that outlived its sign-in changes nothing.
    generation: u64,
}

impl FlowState {
    /// Drop the sign-in under way (and stop its loopback wait).
    fn end_sign_in(&mut self) -> bool {
        self.generation += 1;
        self.pasted = None;
        self.attempt = None;
        if let Some(waiting) = self.waiting.take() {
            waiting.abort();
        }
        self.sign_in.take().is_some()
    }
}

/// What the Providers tab has under way, shared with the loopback wait.
#[derive(Clone, Default)]
pub(crate) struct ProviderFlow(Arc<Mutex<FlowState>>);

impl ProviderFlow {
    /// The sign-in and the key checked, for the settings.
    pub(crate) async fn shown(&self) -> (Option<SignIn>, Option<CheckedKey>) {
        let state = self.0.lock().await;
        (state.sign_in.clone(), state.checked.clone())
    }

    /// Check `key` with `provider`'s preset: nothing is saved.
    pub(crate) async fn check_key(
        &self,
        provider: &ProviderChoice,
        key: &str,
    ) -> Result<String, String> {
        let entry = crate::setup::model::preset(&provider.preset).ok_or("Unknown provider")?;
        if !entry.needs_api_key() {
            return Err(format!("{} takes no API key", entry.display_name));
        }
        if key.trim().is_empty() {
            return Err("Paste the key first".to_string());
        }
        if entry.custom_base_url && provider.base_url.trim().is_empty() {
            return Err("Name the provider's address first".to_string());
        }
        let checked = crate::setup::model::check_key(entry, &provider.base_url, key.trim()).await?;
        let message = format!(
            "The key works. Its {} account offers {} models.",
            entry.display_name,
            checked.models.len()
        );
        self.0.lock().await.checked = Some(checked);
        Ok(message)
    }

    /// Forget the key check (its key was saved, or another provider was
    /// picked).
    pub(crate) async fn clear_checked(&self) {
        self.0.lock().await.checked = None;
    }

    /// Start a ChatGPT sign-in. It waits on this machine's loopback port,
    /// so a browser here finishes it by itself (`finished` is told how it
    /// went). With `paste` (the owner may be on another device, whose
    /// browser cannot reach that port), or when no port can be bound, the
    /// address the browser ends on finishes it too. One already under way
    /// is joined, not replaced (another device, or a second press, would
    /// otherwise cancel the page its owner is signing in on); Cancel starts
    /// over.
    pub(crate) async fn sign_in(
        &self,
        paste: bool,
        finished: impl FnOnce(bool, String) + Send + 'static,
    ) -> Result<String, String> {
        {
            let mut state = self.0.lock().await;
            if let Some(attempt) = state.attempt.clone()
                && let Some(sign_in) = state.sign_in.as_mut()
            {
                if paste && !sign_in.paste {
                    sign_in.paste = true;
                    state.pasted = Some(attempt);
                }
                return Ok("Finish the sign-in under way, or cancel it to start over.".to_string());
            }
        }
        let manager = crate::providers::chatgpt_login::shared_credential_manager();
        let (attempt, fell_back) = match manager.start_login(true).await {
            Ok(attempt) => (attempt, false),
            // No loopback port: paste instead.
            Err(_) => (
                manager
                    .start_login(false)
                    .await
                    .map_err(|e| e.to_string())?,
                true,
            ),
        };
        let attempt = Arc::new(attempt);
        let loopback = attempt.on_loopback();
        let paste = paste || !loopback;
        let generation = {
            let mut state = self.0.lock().await;
            state.end_sign_in();
            state.sign_in = Some(SignIn {
                url: attempt.url.clone(),
                paste,
            });
            state.pasted = paste.then(|| attempt.clone());
            state.attempt = Some(attempt.clone());
            state.generation
        };
        if !loopback {
            return Ok(if fell_back {
                "The sign-in port on the daemon's computer is busy: sign in, then paste the \
                 address the browser ends on."
                    .to_string()
            } else {
                "Sign in, then paste the address the browser ends on.".to_string()
            });
        }
        let flow = self.clone();
        let task = tokio::spawn(async move {
            let client = reqwest::Client::new();
            let result = manager.finish_on_loopback(&client, &attempt).await;
            {
                let mut state = flow.0.lock().await;
                if state.generation != generation {
                    return;
                }
                state.waiting = None;
                state.end_sign_in();
            }
            match result {
                Ok(_) => finished(true, "Signed in to ChatGPT.".to_string()),
                Err(error) => finished(false, format!("The sign-in did not finish: {error}")),
            }
        });
        {
            let mut state = self.0.lock().await;
            if state.generation == generation {
                state.waiting = Some(task.abort_handle());
            }
        }
        Ok("Finish signing in in the browser.".to_string())
    }

    /// Finish a sign-in with the address its browser ended on.
    pub(crate) async fn finish_sign_in(&self, redirect: &str) -> Result<String, String> {
        let (attempt, generation) = {
            let state = self.0.lock().await;
            let attempt = state
                .pasted
                .clone()
                .ok_or("There is no sign-in waiting for an address; start it again")?;
            (attempt, state.generation)
        };
        let manager = crate::providers::chatgpt_login::shared_credential_manager();
        let client = reqwest::Client::new();
        match manager
            .finish_with_redirect(&client, &attempt, redirect.trim())
            .await
        {
            Ok(_) => {
                let mut state = self.0.lock().await;
                if state.generation == generation {
                    state.end_sign_in();
                }
                Ok("Signed in to ChatGPT.".to_string())
            }
            // The same sign-in may be pasted again.
            Err(error) => Err(format!("That did not finish the sign-in: {error}")),
        }
    }

    pub(crate) async fn cancel_sign_in(&self) -> String {
        if self.0.lock().await.end_sign_in() {
            "Stopped the sign-in.".to_string()
        } else {
            "No sign-in was under way.".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> toml::Table {
        text.parse().unwrap()
    }

    /// A second "Sign in" (another device, or a second press) joins the
    /// sign-in under way instead of cancelling its page; one from a device
    /// that may be elsewhere lets an address be pasted from then on.
    #[tokio::test]
    async fn a_second_sign_in_joins_the_one_under_way() {
        use aidaemon_providers::chatgpt_login::{ChatGptCredentialManager, LoginSettings};
        let manager = ChatGptCredentialManager::new(
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
            LoginSettings::new("aidaemon", "aidaemon auth login openai"),
        );
        let attempt = Arc::new(manager.start_login(false).await.unwrap());
        let flow = ProviderFlow::default();
        {
            let mut state = flow.0.lock().await;
            state.sign_in = Some(SignIn {
                url: attempt.url.clone(),
                paste: false,
            });
            state.attempt = Some(attempt.clone());
            state.generation = 7;
        }

        let message = flow
            .sign_in(true, |_, _| panic!("a joined sign-in starts no wait"))
            .await
            .unwrap();
        assert!(message.contains("under way"), "{message}");
        let state = flow.0.lock().await;
        let sign_in = state.sign_in.as_ref().unwrap();
        assert_eq!(sign_in.url, attempt.url);
        assert!(sign_in.paste);
        assert!(Arc::ptr_eq(state.pasted.as_ref().unwrap(), &attempt));
        assert_eq!(state.generation, 7);
    }

    #[test]
    fn a_provider_with_only_a_key_shows_the_catalogs_models_by_job() {
        let config = table(
            r#"
kind = "anthropic"
[models]
default = "claude-haiku-5-5"
"#,
        );
        assert_eq!(
            jobs_of(&config, "anthropic"),
            (
                "claude-opus-5-5".to_string(),
                "claude-haiku-5-5".to_string()
            )
        );
        assert_eq!(jobs_of(&config, "not-a-kind"), Default::default());
    }

    #[test]
    fn the_main_provider_and_each_failover_read_as_their_presets() {
        let config = table(
            r#"
[provider]
kind = "openai_chatgpt"
base_url = "https://api.openai.com/v1"
api_key = ""
[provider.models]
default = "synthetic-sol"
fast = "synthetic-luna"
fallback = ["synthetic-terra", "synthetic-sol"]

[[provider.fallbacks]]
kind = "anthropic"
api_key = "keychain"
[provider.fallbacks.models]
default = "synthetic-haiku"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://openrouter.ai/api/v1/"
api_key = "sk-synthetic-inline"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://llm.example.com/v1"
api_key = "keychain"
"#,
        );
        let stored = |name: &str| name == "provider_fallback_0_api_key";
        let all = connected_with(&config, &stored, false);
        assert_eq!(all.len(), 4);

        assert_eq!(all[0].failover, None);
        assert_eq!(all[0].preset, "openai_chatgpt");
        assert_eq!(all[0].credential, CredentialState::SignedOut);
        assert!(all[0].key_setting.is_empty());
        assert_eq!(
            all[0].models,
            ["synthetic-sol", "synthetic-luna", "synthetic-terra"]
        );

        assert_eq!(all[1].failover, Some(0));
        assert_eq!(all[1].preset, "anthropic_native");
        assert_eq!(all[1].credential, CredentialState::KeyStored);
        assert_eq!(all[1].key_setting, "provider.fallbacks[0].api_key");

        assert_eq!(all[2].preset, "openrouter");
        assert_eq!(all[2].credential, CredentialState::KeyInFile);

        // No preset has that address: the custom endpoint, by its host.
        assert_eq!(all[3].preset, "custom_openai_compatible");
        assert!(all[3].name.ends_with("llm.example.com"), "{}", all[3].name);
        assert_eq!(all[3].credential, CredentialState::KeyMissing);
        assert_eq!(failover_count(&config), 3);
    }

    #[test]
    fn each_provider_offers_its_catalog_models_and_servers_offer_none() {
        let config = table(
            r#"
[provider]
kind = "anthropic"
api_key = "keychain"

[[provider.fallbacks]]
kind = "openai_chatgpt"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://api.deepseek.com"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "https://llm.example.com/v1"

[[provider.fallbacks]]
kind = "openai_compatible"
base_url = "http://localhost:11434/v1"

[[provider.fallbacks]]
kind = "not-a-kind"
"#,
        );
        let all = connected_with(&config, &|_| false, true);
        assert_eq!(
            all[0].offered_models,
            [
                "claude-fable-5-2",
                "claude-opus-5-5",
                "claude-sonnet-5-5",
                "claude-haiku-5-5"
            ]
        );
        assert_eq!(all[1].preset, "openai_chatgpt");
        assert_eq!(
            all[1].offered_models,
            aidaemon_providers::CHATGPT_KNOWN_MODELS
        );
        assert!(all[1].offered_models.iter().any(|m| m == "gpt-6-astra"));
        assert_eq!(all[2].preset, "deepseek");
        assert_eq!(
            all[2].offered_models,
            ["deepseek-chat", "deepseek-reasoner"]
        );
        // Servers whose models depend on the address, and an unknown kind.
        assert_eq!(all[3].preset, "custom_openai_compatible");
        assert!(all[3].offered_models.is_empty());
        assert_eq!(all[4].preset, "ollama");
        assert!(all[4].offered_models.is_empty());
        assert!(all[5].offered_models.is_empty());
    }

    #[test]
    fn no_provider_table_reads_as_none() {
        assert!(connected_with(&table("[agent]\nname = \"x\""), &|_| false, true).is_empty());
        assert_eq!(failover_count(&table("")), 0);
    }
}
