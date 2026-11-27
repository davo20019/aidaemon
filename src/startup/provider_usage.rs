//! What each connected provider's account has used and has left, for the
//! Providers tab: an OpenRouter key's balance (the provider with a public
//! usage endpoint; see `aidaemon_providers::usage`). Readings are cached a
//! minute, and Settings waits only briefly for a new one, so opening the
//! page never stalls on a provider.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::{Duration, Instant, SystemTime};

use aidaemon_client_protocol::{
    ConnectedProvider, CredentialState, ProviderUsage, UsageBalance, UsageWindow,
};
use aidaemon_providers::usage::{AccountUsage, Balance};
use tokio::sync::Mutex;
use tracing::debug;

/// How long a reading is shown before it is read again.
const FRESH_FOR: Duration = Duration::from_secs(60);
/// How long Settings waits for new readings before showing the page with
/// the last ones (or none).
const WAIT_AT_MOST: Duration = Duration::from_secs(4);

/// Where a provider's usage is read from.
#[derive(Debug, Clone, PartialEq)]
enum Source {
    OpenRouter { base_url: String, api_key: String },
}

impl Source {
    /// The cache key: the account, never the secret itself.
    fn cache_key(&self, key_setting: &str) -> String {
        match self {
            Source::OpenRouter { base_url, .. } => format!("openrouter {base_url} {key_setting}"),
        }
    }
}

/// A reading: when it was taken, and what was read (`None` when the
/// provider would not say, so a failure is not retried at once).
type Reading = (Instant, Option<ProviderUsage>);

/// The readings so far, by account.
static CACHE: LazyLock<Mutex<HashMap<String, Reading>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The provider tables `config` lists, main first, as `connected` orders them.
fn tables(config: &toml::Table) -> Vec<&toml::Table> {
    let Some(provider) = config.get("provider").and_then(toml::Value::as_table) else {
        return Vec::new();
    };
    let fallbacks = provider
        .get("fallbacks")
        .or_else(|| provider.get("failover"))
        .and_then(toml::Value::as_array);
    std::iter::once(provider)
        .chain(
            fallbacks
                .into_iter()
                .flatten()
                .filter_map(toml::Value::as_table),
        )
        .collect()
}

/// Whether `url`'s host is OpenRouter's. A machine format (a URL), matched
/// on its parsed host.
fn is_openrouter(url: &str) -> bool {
    reqwest::Url::parse(url.trim())
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "openrouter.ai")
}

/// Where `provider` (written as `table`) reports its usage, if anywhere.
/// `key`: reads a stored secret by its keychain name.
fn source_of(
    provider: &ConnectedProvider,
    table: &toml::Table,
    key: &dyn Fn(&str) -> Option<String>,
) -> Option<Source> {
    let base_url = if provider.base_url.is_empty() {
        aidaemon_providers::catalog::CATALOG
            .iter()
            .find(|entry| entry.id == provider.preset)?
            .base_url
            .to_string()
    } else {
        provider.base_url.clone()
    };
    if !is_openrouter(&base_url) {
        return None;
    }
    let api_key = match provider.credential {
        CredentialState::KeyStored => {
            key(&crate::config::keychain_name_for(&provider.key_setting)?)?
        }
        // The file as written: a `${ENV_VAR}` reference is expanded here,
        // as loading the config does, or the reference itself would go out
        // as the key (OpenRouter: 401 on each read). Unset: no usage.
        CredentialState::KeyInFile => crate::config::expand_env_vars(
            table.get("api_key").and_then(toml::Value::as_str)?.trim(),
        )
        .ok()?,
        _ => return None,
    };
    (!api_key.is_empty()).then_some(Source::OpenRouter { base_url, api_key })
}

fn ms(at: SystemTime) -> u64 {
    at.duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn cents(dollars: f64) -> u64 {
    (dollars.max(0.0) * 100.0).round() as u64
}

/// A reading as the tab shows it.
fn shown(usage: AccountUsage, now: SystemTime) -> ProviderUsage {
    ProviderUsage {
        plan: usage.plan.unwrap_or_default(),
        windows: usage
            .windows
            .into_iter()
            .map(|w| UsageWindow {
                minutes: (w.length.as_secs() / 60) as u32,
                used_percent: w.used_percent.round().clamp(0.0, 100.0) as u8,
                resets_at_ms: w.resets_at.map(ms),
            })
            .collect(),
        balance: usage.balance.map(|b: Balance| UsageBalance {
            used_cents: cents(b.used),
            limit_cents: b.limit.map(cents),
            remaining_cents: b.remaining.map(cents),
        }),
        read_at_ms: ms(now),
    }
}

async fn read(source: &Source) -> Option<ProviderUsage> {
    let result = match source {
        Source::OpenRouter { base_url, api_key } => {
            aidaemon_providers::usage::openrouter_usage(base_url, api_key).await
        }
    };
    match result {
        Ok(usage) => Some(shown(usage, SystemTime::now())),
        Err(error) => {
            debug!(%error, "Provider usage not read");
            None
        }
    }
}

/// Fill in `usage` on each of `providers` (as `connected` listed them from
/// `config`) that reports it, reading anew what is older than a minute.
pub(crate) async fn fill(config: &toml::Table, providers: &mut [ConnectedProvider]) {
    let key = |name: &str| crate::config::resolve_from_keychain(name).ok();
    let tables = tables(config);
    let sources: Vec<Option<(String, Source)>> = providers
        .iter()
        .zip(tables.iter().map(Some).chain(std::iter::repeat(None)))
        .map(|(provider, table)| {
            let source = source_of(provider, table?, &key)?;
            Some((source.cache_key(&provider.key_setting), source))
        })
        .collect();
    // What is stale, once per account, read together.
    let stale: Vec<(String, Source)> = {
        let cache = CACHE.lock().await;
        let mut seen = std::collections::HashSet::new();
        sources
            .iter()
            .flatten()
            .filter(|(key, _)| {
                cache
                    .get(key)
                    .is_none_or(|(at, _)| at.elapsed() >= FRESH_FOR)
                    && seen.insert(key.clone())
            })
            .cloned()
            .collect()
    };
    if !stale.is_empty() {
        let reads = futures::future::join_all(stale.iter().map(|(_, source)| read(source)));
        if let Ok(readings) = tokio::time::timeout(WAIT_AT_MOST, reads).await {
            let mut cache = CACHE.lock().await;
            for ((key, _), reading) in stale.into_iter().zip(readings) {
                // A failed read keeps the last good one on show.
                let kept = reading.or_else(|| cache.get(&key).and_then(|(_, u)| u.clone()));
                cache.insert(key, (Instant::now(), kept));
            }
        }
    }
    let cache = CACHE.lock().await;
    for (provider, source) in providers.iter_mut().zip(sources) {
        provider.usage = source.and_then(|(key, _)| cache.get(&key).and_then(|(_, u)| u.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(credential: CredentialState, preset: &str, base_url: &str) -> ConnectedProvider {
        ConnectedProvider {
            failover: Some(0),
            preset: preset.into(),
            name: "Synthetic".into(),
            base_url: base_url.into(),
            models: Vec::new(),
            hard_model: String::new(),
            quick_model: String::new(),
            offered_models: Vec::new(),
            credential,
            key_setting: "provider.fallbacks[0].api_key".into(),
            usage: None,
        }
    }

    #[test]
    fn only_openrouter_reports_usage() {
        let table = toml::Table::new();
        let stored = |_: &str| Some("sk-synthetic".to_string());
        assert_eq!(
            source_of(
                &provider(CredentialState::SignedIn, "openai_chatgpt", ""),
                &table,
                &stored
            ),
            None
        );
        assert_eq!(
            source_of(
                &provider(CredentialState::KeyStored, "openrouter", ""),
                &table,
                &stored
            ),
            Some(Source::OpenRouter {
                base_url: "https://openrouter.ai/api/v1".into(),
                api_key: "sk-synthetic".into(),
            })
        );
        assert_eq!(
            source_of(
                &provider(CredentialState::KeyStored, "anthropic", ""),
                &table,
                &stored
            ),
            None
        );
        // An OpenRouter key that is missing reads nothing.
        assert_eq!(
            source_of(
                &provider(CredentialState::KeyMissing, "openrouter", ""),
                &table,
                &stored
            ),
            None
        );
    }

    #[test]
    fn a_key_in_the_file_is_read_from_its_table() {
        let table: toml::Table = r#"api_key = "sk-in-file""#.parse().unwrap();
        let source = source_of(
            &provider(
                CredentialState::KeyInFile,
                "",
                "https://openrouter.ai/api/v1/",
            ),
            &table,
            &|_| None,
        );
        assert!(
            matches!(source, Some(Source::OpenRouter { api_key, .. }) if api_key == "sk-in-file")
        );
    }

    /// A key written as an unset `${ENV_VAR}` reference reads no usage:
    /// the reference itself never goes out as the key.
    #[test]
    fn an_unset_env_reference_in_the_file_is_not_sent_as_the_key() {
        let table: toml::Table = r#"api_key = "${AIDAEMON_USAGE_TEST_NEVER_SET_KEY}""#
            .parse()
            .unwrap();
        let source = source_of(
            &provider(
                CredentialState::KeyInFile,
                "",
                "https://openrouter.ai/api/v1/",
            ),
            &table,
            &|_| None,
        );
        assert!(source.is_none());
    }

    #[test]
    fn readings_become_whole_percents_and_cents() {
        let usage = AccountUsage {
            plan: Some("plus".into()),
            windows: vec![aidaemon_providers::usage::UsageWindow {
                length: Duration::from_secs(18_000),
                used_percent: 42.6,
                resets_at: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            }],
            balance: Some(Balance {
                used: 3.456,
                limit: None,
                remaining: None,
            }),
        };
        let shown = shown(usage, SystemTime::UNIX_EPOCH + Duration::from_secs(5));
        assert_eq!(shown.windows[0].minutes, 300);
        assert_eq!(shown.windows[0].used_percent, 43);
        assert_eq!(shown.windows[0].resets_at_ms, Some(1_700_000_000_000));
        assert_eq!(shown.balance.unwrap().used_cents, 346);
        assert_eq!(shown.read_at_ms, 5_000);
    }
}
