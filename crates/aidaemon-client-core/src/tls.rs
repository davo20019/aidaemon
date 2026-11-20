//! TLS for the link's websocket and its HTTP calls: rustls on aws-lc-rs,
//! checked by the OS trust store through the platform verifier. The
//! provider is named because a build can link both rustls backends, and
//! rustls's `ClientConfig::builder()` panics when it has to choose.
//!
//! Android is the exception: its platform verifier needs a Java component
//! registered from the app's activity, which an app built by nsh does not
//! ship, so there the bundled Mozilla roots (`webpki-roots`) decide.

use std::sync::Arc;

use rustls::ClientConfig;

pub(crate) fn client_config() -> Result<Arc<ClientConfig>, rustls::Error> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let builder =
        ClientConfig::builder_with_provider(provider).with_safe_default_protocol_versions()?;
    #[cfg(not(target_os = "android"))]
    let builder = {
        use rustls_platform_verifier::BuilderVerifierExt;
        builder.with_platform_verifier()?
    };
    #[cfg(target_os = "android")]
    let builder = builder.with_root_certificates(rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    });
    Ok(Arc::new(builder.with_no_client_auth()))
}

/// The HTTP client for sign-in, pairing and uploads, on [`client_config`].
pub(crate) fn http_client() -> reqwest::Client {
    client_config()
        .ok()
        .and_then(|tls| {
            reqwest::Client::builder()
                .tls_backend_preconfigured((*tls).clone())
                .build()
                .ok()
        })
        .unwrap_or_default()
}
