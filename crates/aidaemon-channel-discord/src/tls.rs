//! TLS for the gateway websocket: rustls on aws-lc-rs, checked by the OS trust
//! store through the same platform verifier reqwest uses. The provider is
//! named because a build can link both rustls backends, and rustls's
//! `ClientConfig::builder()` panics when it has to choose.

use std::sync::Arc;

use rustls::ClientConfig;
use rustls_platform_verifier::BuilderVerifierExt;

pub(crate) fn client_config() -> Result<Arc<ClientConfig>, rustls::Error> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_platform_verifier()?
        .with_no_client_auth();
    Ok(Arc::new(config))
}
