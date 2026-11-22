//! The daemon's side: serve a router to the peers [`Access`] lets in, each
//! request carrying the peer's key as a [`RemotePeer`] extension.

use std::fmt;
use std::sync::Arc;

use anyhow::Context as _;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use iroh::Endpoint;
use iroh::endpoint::Connection;
use tracing::{debug, warn};

use crate::{EndpointAddr, EndpointId, Relays, RemoteUrl, SecretKey, Side, bind};

/// Which peers may open a connection.
#[derive(Clone, Default)]
pub enum Access {
    /// Any key; the gateway's own sign-in still applies.
    #[default]
    Any,
    /// Only keys the check accepts.
    Only(Arc<dyn Fn(&EndpointId) -> bool + Send + Sync>),
}

impl Access {
    fn allows(&self, peer: &EndpointId) -> bool {
        match self {
            Self::Any => true,
            Self::Only(check) => check(peer),
        }
    }
}

impl fmt::Debug for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Any => "Access::Any",
            Self::Only(_) => "Access::Only(..)",
        })
    }
}

/// The key of the peer a request came from, as a request extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RemotePeer(pub EndpointId);

impl AsRef<[u8]> for RemotePeer {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

/// Serves a router to the peers [`Access`] lets in.
pub struct RemoteListener {
    endpoint: Endpoint,
    accept: tokio::task::JoinHandle<()>,
}

impl RemoteListener {
    pub async fn bind(
        secret: SecretKey,
        relays: &Relays,
        access: Access,
        app: axum::Router,
    ) -> anyhow::Result<Self> {
        let endpoint = bind(secret, relays, Side::Daemon).await?;
        let accept = tokio::spawn(accept_loop(endpoint.clone(), access, app));
        Ok(Self { endpoint, accept })
    }

    /// The daemon's key: its address for life.
    pub fn id(&self) -> EndpointId {
        self.endpoint.id()
    }

    /// Where peers dial now: the key, the home relay once connected, and
    /// direct addresses.
    pub fn addr(&self) -> EndpointAddr {
        self.endpoint.addr()
    }

    /// The address clients keep: the key and the home relay, once there is
    /// one.
    pub fn url(&self) -> RemoteUrl {
        RemoteUrl {
            id: self.id(),
            relay: self.addr().relay_urls().next().cloned(),
        }
    }

    /// Wait until the endpoint is connected to its home relay.
    pub async fn online(&self) {
        self.endpoint.online().await;
    }

    pub async fn close(self) {
        self.accept.abort();
        self.endpoint.close().await;
    }
}

async fn accept_loop(endpoint: Endpoint, access: Access, app: axum::Router) {
    while let Some(incoming) = endpoint.accept().await {
        let access = access.clone();
        let app = app.clone();
        tokio::spawn(async move {
            let conn = match incoming.await {
                Ok(conn) => conn,
                Err(error) => {
                    debug!(%error, "Remote connection failed during the handshake");
                    return;
                }
            };
            let peer = conn.remote_id();
            if !access.allows(&peer) {
                debug!(peer = %peer.fmt_short(), "Remote connection refused: key not allowed");
                conn.close(1u32.into(), b"not allowed");
                return;
            }
            debug!(peer = %peer.fmt_short(), "Remote connection accepted");
            serve_connection(conn, peer, app).await;
        });
    }
}

async fn serve_connection(conn: Connection, peer: EndpointId, app: axum::Router) {
    let app = app.layer(axum::Extension(RemotePeer(peer)));
    while let Ok((send, recv)) = conn.accept_bi().await {
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            let io = TokioIo::new(tokio::io::join(recv, send));
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .with_upgrades()
                .await
            {
                debug!(%error, "Remote stream ended with an error");
            }
        });
    }
}

/// The endpoint key kept at `path` (hex, owner-only), made on first use.
/// The key is the daemon's address for life: a new one unpairs every app.
pub fn load_or_create_key(path: &std::path::Path) -> anyhow::Result<SecretKey> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let key: SecretKey = text
                .trim()
                .parse()
                .with_context(|| format!("{} holds no endpoint key", path.display()))?;
            keep_owner_only(path)?;
            Ok(key)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let key = SecretKey::generate();
            write_owner_only(path, &hex(&key.to_bytes())).with_context(|| {
                format!("could not save the endpoint key at {}", path.display())
            })?;
            Ok(key)
        }
        Err(error) => Err(error)
            .with_context(|| format!("could not read the endpoint key at {}", path.display())),
    }
}

/// Narrow a key file others can read (copied, restored from a backup) back
/// to owner-only: whoever reads it can answer as this daemon.
fn keep_owner_only(path: &std::path::Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)?.permissions().mode();
        if mode & 0o077 != 0 {
            warn!(
                path = %path.display(),
                mode = format!("{:o}", mode & 0o777),
                "The endpoint key was readable by others; narrowed to owner-only"
            );
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .with_context(|| format!("could not narrow {}", path.display()))?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn write_owner_only(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    // Created owner-only, never widened and narrowed after.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_is_made_once_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("remote-key");
        let first = load_or_create_key(&path).unwrap();
        let again = load_or_create_key(&path).unwrap();
        assert_eq!(first.public(), again.public());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_key_file_others_can_read_is_narrowed() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("remote-key");
        let first = load_or_create_key(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let again = load_or_create_key(&path).unwrap();
        assert_eq!(first.public(), again.public());
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn a_damaged_key_file_is_an_error_not_a_new_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("remote-key");
        std::fs::write(&path, "not a key").unwrap();
        assert!(load_or_create_key(&path).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not a key");
    }
}
