//! A loopback bridge to a daemon reached over iroh: an HTTP server on
//! `127.0.0.1` inside the client that forwards each request and WebSocket
//! to the daemon over the device's iroh connection. Whatever addresses the
//! gateway by URL (pictures and files the UI loads itself, live views,
//! conversation shells, screen sharing) uses the bridge's base instead of
//! the `iroh://` address and works unchanged.
//!
//! The base carries a random secret path segment, so another program or
//! user on this machine that finds the port cannot use it. The bridge adds
//! no credentials: each request brings its own token or signed ticket, and
//! the daemon honours them only from this device's endpoint.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt as _, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::sync::Mutex;
use tracing::debug;

use crate::remote::{Dialer, RemoteUrl, connection, forget};
use crate::signin::SignInError;

type Body = BoxBody<Bytes, hyper::Error>;

/// Running bridges' bases, by dialing key and daemon.
type Running = Mutex<HashMap<([u8; 32], aidaemon_remote::EndpointId), String>>;

fn bridges() -> &'static Running {
    static BRIDGES: OnceLock<Running> = OnceLock::new();
    BRIDGES.get_or_init(Mutex::default)
}

/// The base (`http://127.0.0.1:<port>/<secret>`) that reaches the daemon at
/// `gateway` (`iroh://…`) as `dialer`, starting the bridge on first use.
/// Starting it opens no connection; the first request dials.
pub async fn base(gateway: &str, dialer: &Dialer) -> Result<String, SignInError> {
    let url = RemoteUrl::parse(gateway)
        .ok_or_else(|| SignInError::Refused(format!("not a daemon address: {gateway}")))?;
    let slot = (dialer.key().to_bytes(), url.id);
    let mut running = bridges().lock().await;
    if let Some(base) = running.get(&slot) {
        return Ok(base.clone());
    }
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .await
        .map_err(|e| SignInError::Failed(format!("could not start the local bridge: {e}")))?;
    let port = listener
        .local_addr()
        .map_err(|e| SignInError::Failed(e.to_string()))?
        .port();
    let prefix = format!(
        "/{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let base = format!("http://127.0.0.1:{port}{prefix}");
    tokio::spawn(serve(
        listener,
        Arc::new(Route {
            prefix,
            url,
            dialer: dialer.clone(),
        }),
    ));
    running.insert(slot, base.clone());
    Ok(base)
}

struct Route {
    prefix: String,
    url: RemoteUrl,
    dialer: Dialer,
}

async fn serve(listener: tokio::net::TcpListener, route: Arc<Route>) {
    loop {
        let tcp = match listener.accept().await {
            Ok((tcp, _)) => tcp,
            Err(error) => {
                debug!(%error, "The local bridge could not accept");
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
        };
        let route = route.clone();
        tokio::spawn(async move {
            let service = hyper::service::service_fn(move |request| {
                let route = route.clone();
                async move { Ok::<_, Infallible>(forward(&route, request).await) }
            });
            if let Err(error) = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(tcp), service)
                .with_upgrades()
                .await
            {
                debug!(%error, "A local bridge connection ended with an error");
            }
        });
    }
}

fn plain(status: StatusCode, text: &'static str) -> Response<Body> {
    let mut response = Response::new(
        Full::new(Bytes::from_static(text.as_bytes()))
            .map_err(|never| match never {})
            .boxed(),
    );
    *response.status_mut() = status;
    response
}

/// The request's path and query with the bridge's secret taken off, or
/// `None` when it does not start with it.
fn strip(prefix: &str, path_and_query: &str) -> Option<String> {
    let rest = path_and_query.strip_prefix(prefix)?;
    match rest.chars().next() {
        None => Some("/".into()),
        Some('/') => Some(rest.into()),
        Some('?') => Some(format!("/{rest}")),
        Some(_) => None,
    }
}

async fn forward(route: &Route, mut request: Request<Incoming>) -> Response<Body> {
    let path = request
        .uri()
        .path_and_query()
        .map_or("/", |p| p.as_str())
        .to_owned();
    let Some(path) = strip(&route.prefix, &path) else {
        return plain(StatusCode::NOT_FOUND, "not found");
    };
    let Ok(uri) = path.parse() else {
        return plain(StatusCode::BAD_REQUEST, "bad path");
    };
    *request.uri_mut() = uri;
    let upgrade = request
        .headers()
        .contains_key(hyper::header::UPGRADE)
        .then(|| hyper::upgrade::on(&mut request));

    // A stale connection fails at the stream: dial again once.
    let mut stream = None;
    for _ in 0..2 {
        match connection(&route.url, &route.dialer).await {
            Ok(conn) => match conn.open_stream().await {
                Ok(opened) => {
                    stream = Some(opened);
                    break;
                }
                Err(_) => forget(&route.url, &route.dialer).await,
            },
            Err(error) => {
                debug!(%error, "The local bridge could not reach the daemon");
                return plain(StatusCode::BAD_GATEWAY, "the daemon is not reachable");
            }
        }
    }
    let Some(stream) = stream else {
        return plain(StatusCode::BAD_GATEWAY, "the daemon is not reachable");
    };
    let (mut sender, conn) = match hyper::client::conn::http1::handshake(TokioIo::new(stream)).await
    {
        Ok(pair) => pair,
        Err(_) => return plain(StatusCode::BAD_GATEWAY, "the daemon did not answer"),
    };
    tokio::spawn(async move {
        if let Err(error) = conn.with_upgrades().await {
            debug!(%error, "A bridged stream ended with an error");
        }
    });
    let mut response = match sender.send_request(request).await {
        Ok(response) => response,
        Err(_) => return plain(StatusCode::BAD_GATEWAY, "the daemon did not answer"),
    };
    if response.status() == StatusCode::SWITCHING_PROTOCOLS
        && let Some(upgrade) = upgrade
    {
        let theirs = hyper::upgrade::on(&mut response);
        tokio::spawn(async move {
            if let (Ok(ours), Ok(theirs)) = (upgrade.await, theirs.await) {
                let _ = tokio::io::copy_bidirectional(
                    &mut TokioIo::new(ours),
                    &mut TokioIo::new(theirs),
                )
                .await;
            }
        });
    }
    response.map(|body| body.boxed())
}

#[cfg(test)]
mod tests {
    use super::strip;

    #[test]
    fn only_paths_under_the_secret_are_forwarded() {
        let prefix = "/s3cret";
        assert_eq!(
            strip(prefix, "/s3cret/client/v1/files/a/b/c.png").as_deref(),
            Some("/client/v1/files/a/b/c.png")
        );
        assert_eq!(strip(prefix, "/s3cret").as_deref(), Some("/"));
        assert_eq!(strip(prefix, "/s3cret?x=1").as_deref(), Some("/?x=1"));
        assert_eq!(strip(prefix, "/client/v1/files/a"), None);
        assert_eq!(strip(prefix, "/s3cretX/client"), None);
    }
}
