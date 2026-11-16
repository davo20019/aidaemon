//! The providers this crate speaks to, as one table: how each is reached,
//! what it starts with, how it is paid for and where its key comes from.
//! An application builds its "connect a model" step from it: offer the
//! entries, find keys already in the environment ([`keys_in_env`]), check a
//! key ([`crate::key_check`]), read the account's real model list
//! ([`list_models`]) and prove the connection with one call
//! ([`try_connection`]) before saving anything.
//!
//! The starting models are a default to show, not a promise: model names
//! change faster than releases, so a setup screen should offer what
//! [`list_models`] returns and use [`Models::primary`] only to preselect it
//! when the account has it.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::{Message, ModelRequest, Provider, ProviderError, RequestOptions};
use serde_json::Value;

use crate::routing::ProviderKind;
use crate::transport::ConfigError;
use crate::{Anthropic, GoogleGenai, OpenAiCompatible, Xai};

/// How a provider is paid for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Billing {
    /// A subscription the owner already pays for; it has usage limits.
    Plan,
    /// Billed per token by the provider.
    PerToken,
    /// Runs on hardware the owner runs.
    Local,
}

/// How a provider proves who is calling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Credential {
    /// An API key, read from the first of `env` that is set.
    ApiKey {
        env: &'static [&'static str],
        /// Where to make one.
        key_url: &'static str,
    },
    /// The ChatGPT subscription sign-in (feature `chatgpt-login`).
    ChatGptSignIn,
    /// None (a local server).
    None,
}

/// The models a connection starts with, by job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Models {
    pub primary: &'static str,
    /// Quick, cheap work.
    pub fast: &'static str,
    /// The hardest work.
    pub smart: &'static str,
}

/// The models a provider offers, for a settings screen to offer as
/// choices before (or without) asking the provider for its listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfferedModels {
    /// These models, the catalog's own first.
    Listed(&'static [&'static str]),
    /// Whatever the server at the owner's address serves: the catalog
    /// cannot know them.
    ByAddress,
}

impl OfferedModels {
    /// The models, empty for [`OfferedModels::ByAddress`].
    pub fn models(self) -> &'static [&'static str] {
        match self {
            OfferedModels::Listed(models) => models,
            OfferedModels::ByAddress => &[],
        }
    }
}

/// One provider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogEntry {
    /// Stable id (`anthropic_native`, `openai_chatgpt`, ...).
    pub id: &'static str,
    /// Other names it is found by.
    pub aliases: &'static [&'static str],
    pub display_name: &'static str,
    pub kind: ProviderKind,
    /// Empty: the provider's own default.
    pub base_url: &'static str,
    pub models: Models,
    /// The models it offers to pick from.
    pub offered: OfferedModels,
    pub credential: Credential,
    pub billing: Billing,
    /// The owner must name the address.
    pub custom_base_url: bool,
    /// Takes a Cloudflare AI Gateway token as well.
    pub gateway_token: bool,
}

impl CatalogEntry {
    /// It is connected with an API key.
    pub fn needs_api_key(&self) -> bool {
        matches!(self.credential, Credential::ApiKey { .. })
    }

    /// Where to make a key; empty when there is none to make.
    pub fn key_url(&self) -> &'static str {
        match self.credential {
            Credential::ApiKey { key_url, .. } => key_url,
            _ => "",
        }
    }

    /// The `provider.kind` value this entry writes.
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            ProviderKind::OpenaiCompatible => "openai_compatible",
            ProviderKind::XaiNative => "xai_native",
            ProviderKind::GoogleGenai => "google_genai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenaiChatgpt => "openai_chatgpt",
        }
    }

    /// A provider for this entry with `key`, at `base_url` (empty: the
    /// entry's own). `Err` for an entry not reached with a key.
    pub fn provider(&self, key: &str, base_url: &str) -> Result<Arc<dyn Provider>, ConfigError> {
        let base = if base_url.trim().is_empty() {
            self.base_url
        } else {
            base_url.trim()
        };
        Ok(match self.kind {
            ProviderKind::Anthropic => {
                let mut builder = Anthropic::builder().api_key(key);
                if !base.is_empty() {
                    builder = builder.base_url(base);
                }
                Arc::new(builder.build()?)
            }
            ProviderKind::GoogleGenai => {
                let mut builder = GoogleGenai::builder().api_key(key);
                if !base.is_empty() {
                    builder = builder.base_url(base);
                }
                Arc::new(builder.build()?)
            }
            ProviderKind::XaiNative => {
                let mut builder = Xai::builder().api_key(key);
                if !base.is_empty() {
                    builder = builder.base_url(base);
                }
                Arc::new(builder.build()?)
            }
            ProviderKind::OpenaiCompatible => {
                let mut builder = OpenAiCompatible::builder().api_key(key);
                if !base.is_empty() {
                    builder = builder.base_url(base);
                }
                Arc::new(builder.build()?)
            }
            ProviderKind::OpenaiChatgpt => {
                // It signs in; build it with `ChatGpt::builder` and its auth.
                return Err(ConfigError::Missing("a ChatGPT sign-in (it takes no key)"));
            }
        })
    }
}

const fn same(model: &'static str) -> Models {
    Models {
        primary: model,
        fast: model,
        smart: model,
    }
}

/// Every provider, hosted ones first (the subscription sign-in leads).
pub const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "openai_chatgpt",
        aliases: &[
            "chatgpt",
            "chatgptsubscription",
            "codex",
            "openaisubscription",
        ],
        display_name: "OpenAI (ChatGPT subscription)",
        kind: ProviderKind::OpenaiChatgpt,
        base_url: "",
        models: Models {
            primary: "gpt-6-sol",
            fast: "gpt-6-luna",
            smart: "gpt-6-sol",
        },
        offered: OfferedModels::Listed(crate::CHATGPT_KNOWN_MODELS),
        credential: Credential::ChatGptSignIn,
        billing: Billing::Plan,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "anthropic_native",
        aliases: &["anthropic", "claude"],
        display_name: "Anthropic",
        kind: ProviderKind::Anthropic,
        base_url: "https://api.anthropic.com/v1",
        models: Models {
            primary: "claude-sonnet-5-5",
            fast: "claude-haiku-5-5",
            smart: "claude-opus-5-5",
        },
        offered: OfferedModels::Listed(&[
            "claude-fable-5-2",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-haiku-5-5",
        ]),
        credential: Credential::ApiKey {
            env: &["ANTHROPIC_API_KEY"],
            key_url: "https://console.anthropic.com/settings/keys",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "openai",
        aliases: &[],
        display_name: "OpenAI",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://api.openai.com/v1",
        models: same("gpt-5"),
        offered: OfferedModels::Listed(&["gpt-5"]),
        credential: Credential::ApiKey {
            env: &["OPENAI_API_KEY"],
            key_url: "https://platform.openai.com/api-keys",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "google_ai_studio",
        aliases: &["google", "googleaistudio", "gemini"],
        display_name: "Google AI Studio",
        kind: ProviderKind::GoogleGenai,
        base_url: "",
        models: Models {
            primary: "gemini-4-hydrogen",
            fast: "gemini-3.5-flash-lite",
            smart: "gemini-4-argon",
        },
        offered: OfferedModels::Listed(&[
            "gemini-4-hydrogen",
            "gemini-4-argon",
            "gemini-3.5-flash-lite",
        ]),
        credential: Credential::ApiKey {
            env: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            key_url: "https://aistudio.google.com/apikey",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "openrouter",
        aliases: &[],
        display_name: "OpenRouter",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://openrouter.ai/api/v1",
        models: same("moonshotai/kimi-k2.5"),
        offered: OfferedModels::ByAddress,
        credential: Credential::ApiKey {
            env: &["OPENROUTER_API_KEY"],
            key_url: "https://openrouter.ai/keys",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "xai",
        aliases: &["x.ai", "grok"],
        display_name: "xAI (Grok)",
        kind: ProviderKind::XaiNative,
        base_url: "https://api.x.ai/v1",
        models: same("grok-4"),
        offered: OfferedModels::Listed(&["grok-4"]),
        credential: Credential::ApiKey {
            env: &["XAI_API_KEY"],
            key_url: "https://console.x.ai",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "deepseek",
        aliases: &[],
        display_name: "DeepSeek",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://api.deepseek.com",
        models: Models {
            primary: "deepseek-chat",
            fast: "deepseek-chat",
            smart: "deepseek-reasoner",
        },
        offered: OfferedModels::Listed(&["deepseek-chat", "deepseek-reasoner"]),
        credential: Credential::ApiKey {
            env: &["DEEPSEEK_API_KEY"],
            key_url: "https://platform.deepseek.com/api_keys",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "moonshot",
        aliases: &["moonshotai", "kimi", "kimik2", "kimik25"],
        display_name: "Moonshot AI (Kimi)",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://api.moonshot.ai/v1",
        models: Models {
            primary: "kimi-k2.5",
            fast: "kimi-k2.5",
            smart: "kimi-k2-thinking",
        },
        offered: OfferedModels::Listed(&["kimi-k2.5", "kimi-k2-thinking"]),
        credential: Credential::ApiKey {
            env: &["MOONSHOT_API_KEY"],
            key_url: "https://platform.moonshot.ai/",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "minimax",
        aliases: &[],
        display_name: "MiniMax",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://api.minimax.io/v1",
        models: Models {
            primary: "MiniMax-M2.5",
            fast: "MiniMax-M2.5-highspeed",
            smart: "MiniMax-M2.5",
        },
        offered: OfferedModels::Listed(&["MiniMax-M2.5", "MiniMax-M2.5-highspeed"]),
        credential: Credential::ApiKey {
            env: &["MINIMAX_API_KEY"],
            key_url: "https://platform.minimax.io/user-center/basic-information",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "custom_openai_compatible",
        aliases: &["custom", "openaicompatible"],
        display_name: "Another OpenAI-compatible service",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "",
        models: same(""),
        offered: OfferedModels::ByAddress,
        credential: Credential::ApiKey {
            env: &[],
            key_url: "",
        },
        billing: Billing::PerToken,
        custom_base_url: true,
        gateway_token: false,
    },
    CatalogEntry {
        id: "cloudflare_gateway",
        aliases: &["cloudflare", "cloudflareaigateway", "aigateway"],
        display_name: "Cloudflare AI Gateway",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://gateway.ai.cloudflare.com/v1/<ACCOUNT_ID>/<GATEWAY_ID>/compat",
        models: same(""),
        offered: OfferedModels::ByAddress,
        credential: Credential::ApiKey {
            env: &[],
            key_url: "https://dash.cloudflare.com/",
        },
        billing: Billing::PerToken,
        custom_base_url: true,
        gateway_token: true,
    },
    CatalogEntry {
        id: "anthropic_openrouter",
        aliases: &["anthropicopenrouter"],
        display_name: "Anthropic (via OpenRouter)",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "https://openrouter.ai/api/v1",
        models: Models {
            primary: "anthropic/claude-sonnet-4",
            fast: "anthropic/claude-haiku-4",
            smart: "anthropic/claude-opus-4",
        },
        offered: OfferedModels::Listed(&[
            "anthropic/claude-sonnet-4",
            "anthropic/claude-opus-4",
            "anthropic/claude-haiku-4",
        ]),
        credential: Credential::ApiKey {
            env: &[],
            key_url: "https://openrouter.ai/keys",
        },
        billing: Billing::PerToken,
        custom_base_url: false,
        gateway_token: false,
    },
    CatalogEntry {
        id: "ollama",
        aliases: &["ollamalocal"],
        display_name: "Ollama (local)",
        kind: ProviderKind::OpenaiCompatible,
        base_url: "http://localhost:11434/v1",
        models: same(""),
        offered: OfferedModels::ByAddress,
        credential: Credential::None,
        billing: Billing::Local,
        custom_base_url: false,
        gateway_token: false,
    },
];

fn normalize(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '.')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The entry named `name` (its id, display name or an alias; case, spaces
/// and punctuation do not matter).
pub fn find(name: &str) -> Option<&'static CatalogEntry> {
    let needle = normalize(name);
    CATALOG.iter().find(|entry| {
        normalize(entry.id) == needle
            || normalize(entry.display_name) == needle
            || entry.aliases.iter().any(|alias| normalize(alias) == needle)
    })
}

/// The models a provider of `kind` starts with, when one catalog entry is
/// that kind. `None` for OpenAI-compatible servers: which models they offer
/// depends on the address.
pub fn models_for_kind(kind: ProviderKind) -> Option<Models> {
    let mut entries = CATALOG.iter().filter(|entry| entry.kind == kind);
    match (entries.next(), entries.next()) {
        (Some(entry), None) => Some(entry.models),
        _ => None,
    }
}

/// A provider key found in the environment.
#[derive(Clone)]
pub struct EnvKey {
    pub entry: &'static CatalogEntry,
    /// The variable it was in.
    pub var: &'static str,
    pub key: String,
}

impl std::fmt::Debug for EnvKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvKey")
            .field("entry", &self.entry.id)
            .field("var", &self.var)
            .field("key", &"<redacted>")
            .finish()
    }
}

/// Every provider key set in the environment, in catalog order, one per
/// provider.
pub fn keys_in_env() -> Vec<EnvKey> {
    keys_in(|var| std::env::var(var).ok())
}

/// [`keys_in_env`] over any lookup.
pub fn keys_in(lookup: impl Fn(&str) -> Option<String>) -> Vec<EnvKey> {
    let mut found = Vec::new();
    for entry in CATALOG {
        let Credential::ApiKey { env, .. } = entry.credential else {
            continue;
        };
        if let Some((var, key)) = env.iter().find_map(|var| {
            lookup(var)
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .map(|k| (*var, k))
        }) {
            found.push(EnvKey { entry, var, key });
        }
    }
    found
}

const LIST_TIMEOUT: Duration = Duration::from_secs(15);

/// The models `key` can use at a provider of `kind` (at `base_url`, empty:
/// the provider's own), as the provider lists them. The ChatGPT
/// subscription has no listing: its known models come back.
pub async fn list_models(
    kind: ProviderKind,
    base_url: &str,
    key: &str,
) -> Result<Vec<String>, String> {
    if kind == ProviderKind::OpenaiChatgpt {
        return Ok(crate::CHATGPT_KNOWN_MODELS
            .iter()
            .map(|m| m.to_string())
            .collect());
    }
    let client = reqwest::Client::builder()
        .timeout(LIST_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;
    let base = base_url.trim().trim_end_matches('/');
    let request = match kind {
        ProviderKind::Anthropic => {
            let base = if base.is_empty() {
                "https://api.anthropic.com"
            } else {
                base.trim_end_matches("/v1")
            };
            client
                .get(format!("{base}/v1/models?limit=1000"))
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
        }
        ProviderKind::GoogleGenai => {
            let base = if base.is_empty() {
                "https://generativelanguage.googleapis.com"
            } else {
                base.trim_end_matches("/v1beta").trim_end_matches("/v1")
            };
            client
                .get(format!("{base}/v1beta/models?pageSize=1000"))
                .header("x-goog-api-key", key)
        }
        ProviderKind::XaiNative => {
            let base = if base.is_empty() {
                "https://api.x.ai/v1"
            } else {
                base
            };
            client.get(format!("{base}/models")).bearer_auth(key)
        }
        ProviderKind::OpenaiCompatible | ProviderKind::OpenaiChatgpt => {
            let base = if base.is_empty() {
                crate::OPENAI_DEFAULT_BASE_URL
            } else {
                base
            };
            let request = client.get(format!("{base}/models"));
            if key.is_empty() {
                request
            } else {
                request.bearer_auth(key)
            }
        }
    };
    let response = request.send().await.map_err(|e| {
        if e.is_timeout() {
            "the provider timed out".to_string()
        } else {
            "could not reach the provider".to_string()
        }
    })?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("the provider answered HTTP {}", status.as_u16()));
    }
    Ok(model_ids(kind, &body))
}

/// The model ids in a listing's body.
fn model_ids(kind: ProviderKind, body: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return Vec::new();
    };
    match kind {
        ProviderKind::GoogleGenai => value["models"]
            .as_array()
            .into_iter()
            .flatten()
            // Only models that chat.
            .filter(|m| {
                m["supportedGenerationMethods"]
                    .as_array()
                    .is_none_or(|ways| ways.iter().any(|w| w == "generateContent"))
            })
            .filter_map(|m| m["name"].as_str())
            .map(|name| name.trim_start_matches("models/").to_string())
            .collect(),
        _ => value["data"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m["id"].as_str().map(str::to_string))
            .collect(),
    }
}

/// Of `offered`, the one to preselect: `preferred` when offered, else none
/// (the owner picks).
pub fn preselect(offered: &[String], preferred: &str) -> Option<String> {
    offered.iter().find(|m| m.as_str() == preferred).cloned()
}

/// How long [`try_connection`] waits.
const TRY_WITHIN: Duration = Duration::from_secs(60);

/// One short call to `model` with no tools and a single attempt, to prove
/// the connection works before an application saves it. `Err`: why it did
/// not answer, in plain words.
pub async fn try_connection(provider: &dyn Provider, model: &str) -> Result<(), String> {
    let request = ModelRequest {
        model: model.to_string(),
        messages: vec![Message::user(
            "This is a connection test. Reply with the single word: ready",
        )],
        max_output_tokens: Some(256),
        options: RequestOptions {
            single_attempt: true,
            ..Default::default()
        },
        ..Default::default()
    };
    match tokio::time::timeout(TRY_WITHIN, provider.complete(&request)).await {
        Err(_) => Err("The provider did not answer within a minute".to_string()),
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(plain(&error)),
    }
}

fn plain(error: &ProviderError) -> String {
    match error {
        ProviderError::Auth(_) => "The provider did not accept the key or sign-in".to_string(),
        ProviderError::Billing { .. } => {
            "The account has no credit left; add some and try again".to_string()
        }
        ProviderError::QuotaExhausted { .. } => {
            "The plan's usage limit is used up for now".to_string()
        }
        ProviderError::RateLimited { .. } => {
            "The provider is busy (rate limited); try again in a minute".to_string()
        }
        ProviderError::InvalidRequest(detail) => {
            format!("The provider refused the test: {detail}")
        }
        ProviderError::Transient(detail) => format!("Could not reach the provider: {detail}"),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_is_found_by_its_id_and_ids_are_unique() {
        for entry in CATALOG {
            assert_eq!(find(entry.id).map(|e| e.id), Some(entry.id));
        }
        let mut ids: Vec<&str> = CATALOG.iter().map(|e| e.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), CATALOG.len());
        assert_eq!(find("Claude").map(|e| e.id), Some("anthropic_native"));
        assert_eq!(find("chat gpt").map(|e| e.id), Some("openai_chatgpt"));
    }

    #[test]
    fn each_hosted_entry_offers_its_own_models_and_servers_offer_none() {
        for entry in CATALOG {
            let offered = entry.offered.models();
            if entry.offered == OfferedModels::ByAddress {
                // The endpoint determines its models; only configured models are pickable.
                assert!(offered.is_empty(), "{}", entry.id);
                continue;
            }
            for model in [entry.models.primary, entry.models.fast, entry.models.smart] {
                assert!(offered.contains(&model), "{} lacks {model}", entry.id);
            }
            let mut unique = offered.to_vec();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), offered.len(), "{} repeats a model", entry.id);
        }
        assert_eq!(
            find("anthropic").unwrap().offered.models(),
            [
                "claude-fable-5-2",
                "claude-opus-5-5",
                "claude-sonnet-5-5",
                "claude-haiku-5-5"
            ]
        );
        assert_eq!(
            find("chatgpt").unwrap().offered.models(),
            crate::CHATGPT_KNOWN_MODELS
        );
        for id in [
            "openrouter",
            "custom_openai_compatible",
            "cloudflare_gateway",
            "ollama",
        ] {
            assert!(find(id).unwrap().offered.models().is_empty(), "{id}");
        }
    }

    #[test]
    fn keys_in_the_environment_are_found_once_per_provider() {
        let env = |var: &str| match var {
            "GOOGLE_API_KEY" => Some("g-synthetic".to_string()),
            "GEMINI_API_KEY" => Some(" ".to_string()),
            "ANTHROPIC_API_KEY" => Some("sk-ant-synthetic".to_string()),
            _ => None,
        };
        let found = keys_in(env);
        let ids: Vec<(&str, &str)> = found.iter().map(|k| (k.entry.id, k.var)).collect();
        assert_eq!(
            ids,
            [
                ("anthropic_native", "ANTHROPIC_API_KEY"),
                ("google_ai_studio", "GOOGLE_API_KEY")
            ]
        );
        assert!(!format!("{:?}", found[0]).contains("synthetic"));
    }

    #[test]
    fn listings_read_as_model_ids() {
        assert_eq!(
            model_ids(
                ProviderKind::OpenaiCompatible,
                r#"{"data":[{"id":"synthetic-a"},{"id":"synthetic-b"}]}"#
            ),
            ["synthetic-a", "synthetic-b"]
        );
        assert_eq!(
            model_ids(
                ProviderKind::GoogleGenai,
                r#"{"models":[{"name":"models/synthetic-chat","supportedGenerationMethods":["generateContent"]},{"name":"models/synthetic-embed","supportedGenerationMethods":["embedContent"]}]}"#
            ),
            ["synthetic-chat"]
        );
        let offered = vec!["a".to_string(), "b".to_string()];
        assert_eq!(preselect(&offered, "b").as_deref(), Some("b"));
        assert_eq!(preselect(&offered, "c"), None);
    }
}
