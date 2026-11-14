//! TLS for the sockets reqwest does not open (websockets, IMAP).
//!
//! rustls on aws-lc-rs, checked by the OS trust store through the same
//! platform verifier reqwest uses, so a socket trusts exactly what the
//! daemon's HTTPS client trusts. The provider is named here because the build
//! links both rustls backends (aws-lc-rs and ring), and rustls's
//! `ClientConfig::builder()` panics when it has to choose between them.

use std::sync::Arc;

pub use rustls::ClientConfig;
use rustls_platform_verifier::BuilderVerifierExt;

/// A client config for one connection.
pub fn client_config() -> Result<Arc<ClientConfig>, rustls::Error> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()?
        .with_platform_verifier()?
        .with_no_client_auth();
    Ok(Arc::new(config))
}

#[cfg(test)]
mod tests {
    #[test]
    fn builds_without_a_process_default_provider() {
        assert!(super::client_config().is_ok());
    }
}
