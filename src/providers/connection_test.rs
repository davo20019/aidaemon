//! One short call that proves a model connection works, made while the
//! daemon is being set up. Nothing else exists yet (no stores, agent or
//! side-call gateway), and nothing is saved until it answers: it builds the
//! candidate provider the way the daemon will and asks it for one word,
//! with no tools and a single attempt.

use crate::config::ProviderConfig;

/// How long the test call may take, sign-in refresh included.
const TEST_WITHIN: std::time::Duration = std::time::Duration::from_secs(60);

/// Ask `config`'s default model for a word. `Err`: why it did not answer,
/// in plain words.
pub(crate) async fn try_connection(config: &ProviderConfig) -> Result<(), String> {
    let (backend, _) = super::factory::build_provider(config)
        .map_err(|error| format!("Could not set up the provider: {error}"))?;
    let model = crate::router::Router::new(config.models.clone())
        .default_model()
        .to_string();
    let request = aidaemon_core::ModelRequest {
        model,
        messages: vec![aidaemon_core::Message::user(
            "This is a connection test. Reply with the single word: ready",
        )],
        max_output_tokens: Some(256),
        options: aidaemon_core::RequestOptions {
            single_attempt: true,
            ..Default::default()
        },
        ..Default::default()
    };
    match tokio::time::timeout(TEST_WITHIN, backend.complete(&request)).await {
        Err(_) => Err("The provider did not answer within a minute".to_string()),
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(plain(&error)),
    }
}

fn plain(error: &aidaemon_core::ProviderError) -> String {
    use aidaemon_core::ProviderError as E;
    match error {
        E::Auth(_) => "The provider did not accept the key or sign-in".to_string(),
        E::Billing { .. } => "The account has no credit left; add some and try again".to_string(),
        E::QuotaExhausted { .. } => "The plan's usage limit is used up for now".to_string(),
        E::RateLimited { .. } => {
            "The provider is busy (rate limited); try again in a minute".to_string()
        }
        E::InvalidRequest(detail) => format!("The provider refused the test: {detail}"),
        E::Transient(detail) => format!("Could not reach the provider: {detail}"),
        other => other.to_string(),
    }
}
