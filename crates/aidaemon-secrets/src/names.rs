//! The account names aidaemon's configuration files its secrets under.
//!
//! A config value of `"keychain"` is read from the [`SecretStore`] under one
//! of these names; the env-file mode upper-cases them (`api_key` is
//! `API_KEY`). Changing one orphans every secret already stored under it.
//!
//! [`SecretStore`]: crate::SecretStore

/// A provider field: `api_key` for the primary provider, `{prefix}_api_key`
/// for a fallback (see [`provider_fallback_prefix`]).
pub fn provider_field(prefix: Option<&str>, field: &str) -> String {
    prefix
        .map(|prefix| format!("{}_{}", prefix, field))
        .unwrap_or_else(|| field.to_string())
}

/// The prefix of fallback provider `index`: `provider_fallback_{index}` under
/// the primary provider, `{parent}_fallback_{index}` under another fallback.
pub fn provider_fallback_prefix(parent: Option<&str>, index: usize) -> String {
    parent
        .map(|prefix| format!("{}_fallback_{}", prefix, index))
        .unwrap_or_else(|| format!("provider_fallback_{}", index))
}

/// A header name lower-cased with every other character than ASCII letters
/// and digits turned into `_`.
pub fn normalized_header(header_name: &str) -> String {
    header_name
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// An extra provider header: `provider_header_{name}` for the primary
/// provider, `{prefix}_header_{name}` for a fallback.
pub fn provider_header(prefix: Option<&str>, header_name: &str) -> String {
    let normalized = normalized_header(header_name);
    prefix
        .map(|prefix| format!("{}_header_{}", prefix, normalized))
        .unwrap_or_else(|| format!("provider_header_{}", normalized))
}

/// Bot token `index` of a channel: the first bot uses `first`
/// (`bot_token`, `discord_bot_token`), later ones `{indexed}_{index}`
/// (`telegram_bot_token_1`).
pub fn bot_token(first: &str, indexed: &str, index: usize) -> String {
    if index == 0 {
        first.to_string()
    } else {
        format!("{}_{}", indexed, index)
    }
}

/// A field of an HTTP auth profile: `http_auth_{profile}_{field}`.
pub fn http_auth(profile: &str, field: &str) -> String {
    format!("http_auth_{}_{}", profile, field)
}

/// An OAuth secret of a service: `oauth_{service}_{suffix}` (`client_id`,
/// `client_secret`, `access_token`, `refresh_token`, ...).
pub fn oauth(service: &str, suffix: &str) -> String {
    format!("oauth_{service}_{suffix}")
}

/// An OTLP exporter header: `otlp_header_{name}`, lower-cased.
pub fn otlp_header(header_name: &str) -> String {
    format!("otlp_header_{}", header_name.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_daemons_stored_accounts() {
        assert_eq!(provider_field(None, "api_key"), "api_key");
        let fallback = provider_fallback_prefix(None, 0);
        assert_eq!(fallback, "provider_fallback_0");
        assert_eq!(
            provider_field(Some(&fallback), "api_key"),
            "provider_fallback_0_api_key"
        );
        assert_eq!(
            provider_fallback_prefix(Some(&fallback), 1),
            "provider_fallback_0_fallback_1"
        );
        assert_eq!(
            provider_header(None, "X-Api-Key"),
            "provider_header_x_api_key"
        );
        assert_eq!(
            provider_header(Some("provider_fallback_2"), "Authorization"),
            "provider_fallback_2_header_authorization"
        );
        assert_eq!(bot_token("bot_token", "telegram_bot_token", 0), "bot_token");
        assert_eq!(
            bot_token("bot_token", "telegram_bot_token", 2),
            "telegram_bot_token_2"
        );
        assert_eq!(http_auth("twitter", "api_key"), "http_auth_twitter_api_key");
        assert_eq!(oauth("github", "client_id"), "oauth_github_client_id");
        assert_eq!(otlp_header("X-Honeycomb"), "otlp_header_x-honeycomb");
    }
}
