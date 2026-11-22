//! The Node Gateway on the local network: an HTTPS listener on this
//! computer's network address, so a Node on the same network reaches the
//! Gateway with no tunnel or cable.
//!
//! No public CA vouches for a private address, so the daemon keeps a CA of
//! its own ([`LanAuthority`], its key owner-only on disk) and issues the
//! listener a certificate for its current address, again when the address
//! changes. Nodes learn the CA over a channel they already trust (a USB
//! relay or a tunnel: the heartbeat's `gateway_trust`) and verify the
//! listener's chain and name against it, so the private key never leaves
//! this computer and an address change needs nothing new on the Node.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Context as _;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, SanType, SerialNumber,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// The CA's name, as Nodes see it in the chain.
const CA_NAME: &str = "AIDaemon local Node Gateway CA";
/// How often the listener checks whether this computer's address changed.
const ADDRESS_CHECK: Duration = Duration::from_secs(30);

/// The daemon's own CA for its local-network listener.
pub struct LanAuthority {
    key: KeyPair,
    ca: CertificateDer<'static>,
    ca_pem: String,
}

impl LanAuthority {
    /// The CA whose key is kept at `path` (PEM, owner-only), made on first
    /// use. The certificate is rebuilt from the key with fixed fields, so
    /// it verifies the same for Nodes that stored an earlier copy.
    pub fn load_or_create(path: &Path) -> anyhow::Result<Self> {
        let key = match std::fs::read_to_string(path) {
            Ok(pem) => KeyPair::from_pem(&pem)
                .with_context(|| format!("{} holds no CA key", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
                write_owner_only(path, &key.serialize_pem())
                    .with_context(|| format!("could not save the CA key at {}", path.display()))?;
                key
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("could not read the CA key at {}", path.display()));
            }
        };
        // The same key and fields each time (the signature bytes differ:
        // ECDSA), so a Node holding an earlier copy still verifies with it.
        let ca = ca_params().self_signed(&key)?;
        Ok(Self {
            ca_pem: ca.pem(),
            ca: ca.der().clone(),
            key,
        })
    }

    /// The CA certificate Nodes trust, PEM.
    pub fn ca_pem(&self) -> &str {
        &self.ca_pem
    }

    /// A certificate for the listener at `address`: the address is its
    /// common name, a DNS name (TLS stacks that match names as text) and an
    /// IP address (those that match addresses), signed by this CA.
    pub fn issue(
        &self,
        address: IpAddr,
    ) -> anyhow::Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
        let leaf_key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
        let mut params = CertificateParams::default();
        let text = address.to_string();
        params.distinguished_name = DistinguishedName::new();
        params
            .distinguished_name
            .push(DnType::CommonName, text.clone());
        params.subject_alt_names = vec![
            SanType::DnsName(text.try_into()?),
            SanType::IpAddress(address),
        ];
        let now = time::OffsetDateTime::now_utc();
        params.not_before = now - time::Duration::days(1);
        params.not_after = now + time::Duration::days(3650);
        params.key_usages = vec![
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::KeyAgreement,
        ];
        params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ServerAuth];
        let issuer = Issuer::new(ca_params(), &self.key);
        let leaf = params.signed_by(&leaf_key, &issuer)?;
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(leaf_key.serialize_der()));
        Ok((vec![leaf.der().clone(), self.ca.clone()], key))
    }
}

/// The CA's fixed fields: the same certificate from the same key.
fn ca_params() -> CertificateParams {
    let mut params = CertificateParams::default();
    params.distinguished_name = DistinguishedName::new();
    params.distinguished_name.push(DnType::CommonName, CA_NAME);
    params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    params.serial_number = Some(SerialNumber::from(1u64));
    params.not_before = time::macros::datetime!(2025-01-01 00:00 UTC);
    params.not_after = time::macros::datetime!(2050-01-01 00:00 UTC);
    params
}

fn write_owner_only(path: &Path, contents: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    let _ = std::fs::remove_file(&tmp);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        use std::io::Write;
        let mut file = options.open(&tmp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// This computer's address on its main network (the one its default route
/// uses), `None` with no network. Sends nothing: connecting a UDP socket
/// only picks the route.
pub fn local_address() -> Option<IpAddr> {
    let socket = std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    // TEST-NET-1 (RFC 5737): never routed, only used to pick the route.
    socket.connect((Ipv4Addr::new(192, 0, 2, 1), 9)).ok()?;
    let address = socket.local_addr().ok()?.ip();
    (!address.is_loopback() && !address.is_unspecified()).then_some(address)
}

/// Marks requests that came through the local-network listener (a request
/// extension), so the Gateway can say how each Node is connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrivedOnLocalNetwork;

/// Where the local-network listener is, for the Nodes: its origin now and
/// the CA that vouches for it.
pub struct LanEndpoint {
    pub port: u16,
    pub authority: Arc<LanAuthority>,
    /// The address to name instead of this computer's current one (tests).
    pub address: Option<IpAddr>,
}

impl LanEndpoint {
    /// `https://<this computer's address>:<port>`, `None` with no network.
    pub fn origin(&self) -> Option<String> {
        self.address
            .or_else(local_address)
            .map(|address| origin(address, self.port))
    }
}

fn origin(address: IpAddr, port: u16) -> String {
    match address {
        IpAddr::V4(v4) => format!("https://{v4}:{port}"),
        IpAddr::V6(v6) => format!("https://[{v6}]:{port}"),
    }
}

/// A TLS configuration for `address`, issued by `authority`.
fn server_config(
    authority: &LanAuthority,
    address: IpAddr,
) -> anyhow::Result<Arc<rustls::ServerConfig>> {
    let (chain, key) = authority.issue(address)?;
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()?
    .with_no_client_auth()
    .with_single_cert(chain, key)?;
    Ok(Arc::new(config))
}

/// Serve `app` over HTTPS on every interface at `port`, with a certificate
/// for this computer's current address, reissued when it changes.
pub async fn serve(
    app: axum::Router,
    authority: Arc<LanAuthority>,
    port: u16,
) -> anyhow::Result<()> {
    let listener =
        tokio::net::TcpListener::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port))
            .await
            .with_context(|| format!("could not listen on port {port}"))?;
    let address = local_address().unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
    serve_on(listener, app, authority, address).await
}

/// [`serve`] on `listener`, with a certificate first issued for `address`.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    authority: Arc<LanAuthority>,
    address: IpAddr,
) -> anyhow::Result<()> {
    let port = listener.local_addr()?.port();
    let app = app.layer(axum::Extension(ArrivedOnLocalNetwork));
    let current = Arc::new(RwLock::new((address, server_config(&authority, address)?)));
    tracing::info!(origin = %origin(address, port), "Node Gateway listening on the local network");
    {
        let current = current.clone();
        let authority = authority.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(ADDRESS_CHECK).await;
                let Some(now) = local_address() else { continue };
                let changed = current.read().map(|c| c.0 != now).unwrap_or(false);
                if changed {
                    match server_config(&authority, now) {
                        Ok(config) => {
                            if let Ok(mut c) = current.write() {
                                *c = (now, config);
                            }
                            tracing::info!(origin = %origin(now, port), "Node Gateway's local address changed");
                        }
                        Err(error) => {
                            tracing::warn!(%error, "could not reissue the local certificate")
                        }
                    }
                }
            }
        });
    }
    loop {
        let (tcp, _) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(error) => {
                tracing::debug!(%error, "local Node Gateway could not accept");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        // Small request and reply messages: send each at once rather than
        // wait for the Node's (delayed) acknowledgement of the last one.
        let _ = tcp.set_nodelay(true);
        let config = match current.read() {
            Ok(c) => c.1.clone(),
            Err(_) => continue,
        };
        let app = app.clone();
        tokio::spawn(async move {
            let tls = match tokio_rustls::TlsAcceptor::from(config).accept(tcp).await {
                Ok(tls) => tls,
                Err(error) => {
                    tracing::debug!(%error, "local Node Gateway TLS handshake failed");
                    return;
                }
            };
            let service = TowerToHyperService::new(app);
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(tls), service)
                .await
            {
                tracing::debug!(%error, "local Node Gateway connection ended with an error");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ca_key_is_made_once_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("node-lan-ca.key");
        let first = LanAuthority::load_or_create(&path).unwrap();
        let again = LanAuthority::load_or_create(&path).unwrap();
        assert_eq!(first.key.serialize_pem(), again.key.serialize_pem());
        assert!(first.ca_pem().starts_with("-----BEGIN CERTIFICATE-----"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    /// A Node that stored the CA before a restart reaches the restarted
    /// listener by address, verifying chain and name; one trusting another
    /// daemon's CA is refused in the handshake.
    #[tokio::test]
    async fn a_node_trusting_the_ca_reaches_the_listener() {
        let dir = tempfile::tempdir().unwrap();
        let stored_by_the_node = LanAuthority::load_or_create(&dir.path().join("ca.key"))
            .unwrap()
            .ca_pem()
            .to_string();
        // The daemon restarted: the same key, a fresh CA certificate.
        let authority = Arc::new(LanAuthority::load_or_create(&dir.path().join("ca.key")).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app =
            axum::Router::new().route("/node/v1/ping", axum::routing::get(|| async { "pong" }));
        let address: IpAddr = "127.0.0.1".parse().unwrap();
        tokio::spawn(serve_on(listener, app, authority.clone(), address));

        let ca = reqwest::Certificate::from_pem(stored_by_the_node.as_bytes()).unwrap();
        let trusting = reqwest::Client::builder()
            .tls_certs_only([ca])
            .build()
            .unwrap();
        let body = trusting
            .get(format!("https://127.0.0.1:{port}/node/v1/ping"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(body, "pong");

        // Another daemon's CA does not vouch for this one.
        let other = LanAuthority::load_or_create(&dir.path().join("other.key")).unwrap();
        let stranger = reqwest::Client::builder()
            .tls_certs_only([reqwest::Certificate::from_pem(other.ca_pem().as_bytes()).unwrap()])
            .build()
            .unwrap();
        assert!(
            stranger
                .get(format!("https://127.0.0.1:{port}/node/v1/ping"))
                .send()
                .await
                .is_err()
        );
    }

    #[test]
    fn an_origin_names_the_address_and_port() {
        assert_eq!(
            origin("192.168.1.20".parse().unwrap(), 8443),
            "https://192.168.1.20:8443"
        );
        assert_eq!(
            origin("fd00::1".parse().unwrap(), 8443),
            "https://[fd00::1]:8443"
        );
    }
}
