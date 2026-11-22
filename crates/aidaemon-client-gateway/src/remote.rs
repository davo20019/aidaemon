//! The gateway as a peer-to-peer transport serves it (iroh): the same
//! routes, behind a guard keyed on the peer's transport key.
//!
//! Over such a transport the gateway is reachable from the internet, so:
//! - a peer with no session may reach only pairing and sign-in;
//! - an access token works only from the peer key that signed in for it,
//!   so a token lifted from one device is useless from another;
//! - the owner's local token is refused: it is for this machine.
//!
//! The transport puts the peer's key on each request as an extension of
//! type `P` (`aidaemon_remote::RemotePeer`); a request without one is
//! refused. The gateway's own sign-in then decides as on loopback.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use aidaemon_client_protocol::{AccessGrant, ErrorCode};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::Method;
use axum::middleware::Next;
use axum::response::Response;
use sha2::{Digest, Sha256};

use crate::ClientGateway;
use crate::bus::now_ms;
use crate::server::{bearer, error_response};

/// The largest sign-in reply read back to bind its token.
const MAX_GRANT_BYTES: usize = 64 * 1024;

/// Access tokens and the peer key each was issued to.
struct Bindings<P> {
    tokens: Mutex<HashMap<String, (P, u64)>>,
}

impl<P: Clone + Eq> Bindings<P> {
    fn bind(&self, token: &str, peer: P, expires_at_ms: u64) {
        let mut tokens = self.tokens.lock().unwrap_or_else(|p| p.into_inner());
        let now = now_ms();
        tokens.retain(|_, (_, expires)| *expires > now);
        tokens.insert(sha(token), (peer, expires_at_ms));
    }

    /// Whether `token` was issued to `peer` and has not run out.
    fn holds(&self, token: &str, peer: &P) -> bool {
        let tokens = self.tokens.lock().unwrap_or_else(|p| p.into_inner());
        tokens
            .get(&sha(token))
            .is_some_and(|(owner, expires)| owner == peer && *expires > now_ms())
    }

    /// Whether `peer` holds any live token: it signed in as a paired device.
    fn signed_in(&self, peer: &P) -> bool {
        let tokens = self.tokens.lock().unwrap_or_else(|p| p.into_inner());
        let now = now_ms();
        tokens
            .values()
            .any(|(owner, expires)| owner == peer && *expires > now)
    }
}

fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// The routes a peer with no session may reach: redeeming a pairing offer,
/// waiting on it, and signing in with a paired key.
fn before_sign_in(method: &Method, path: &str) -> bool {
    match *method {
        Method::POST => matches!(
            path,
            "/client/v1/enrollments/redeem"
                | "/client/v1/sessions/challenge"
                | "/client/v1/sessions"
        ),
        Method::GET => path
            .strip_prefix("/client/v1/enrollments/")
            .is_some_and(|id| !id.is_empty() && !id.contains('/')),
        _ => false,
    }
}

/// The gateway's routes behind the peer guard, for a transport that puts
/// each request's peer key on it as a `P` extension.
pub fn remote_router<P>(gateway: Arc<ClientGateway>) -> Router
where
    P: Clone + Eq + Hash + Send + Sync + 'static,
{
    let bindings = Arc::new(Bindings::<P> {
        tokens: Mutex::default(),
    });
    crate::router(gateway.clone()).layer(axum::middleware::from_fn_with_state(
        (gateway, bindings),
        guard::<P>,
    ))
}

type GuardState<P> = (Arc<ClientGateway>, Arc<Bindings<P>>);

async fn guard<P>(
    State((gateway, bindings)): State<GuardState<P>>,
    request: Request,
    next: Next,
) -> Response
where
    P: Clone + Eq + Hash + Send + Sync + 'static,
{
    let Some(peer) = request.extensions().get::<P>().cloned() else {
        return error_response(
            ErrorCode::Forbidden,
            "This route is served to known peers only",
        );
    };
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    if before_sign_in(&method, &path) {
        let response = next.run(request).await;
        if method == Method::POST && path == "/client/v1/sessions" {
            return bind_grant(response, &bindings, peer).await;
        }
        return response;
    }
    let allowed = match bearer(request.headers()) {
        // A token works from the peer it was issued to, and only while the
        // gateway still honours it (revoking a device ends it).
        Some(token) => bindings.holds(token, &peer) && gateway.principal(token).is_some(),
        // The client socket takes its token in the header here: a token in
        // `hello` would get past this guard.
        None if path == "/client/v1/ws" => false,
        // Ticket sockets and signed file links: only for a signed-in peer.
        None => bindings.signed_in(&peer),
    };
    if !allowed {
        return error_response(
            ErrorCode::Unauthorized,
            "Sign in with this device's key first (the local token is not taken remotely)",
        );
    }
    next.run(request).await
}

/// Read a successful sign-in reply and tie its token to `peer`.
async fn bind_grant<P: Clone + Eq>(
    response: Response,
    bindings: &Bindings<P>,
    peer: P,
) -> Response {
    if !response.status().is_success() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, MAX_GRANT_BYTES).await else {
        return error_response(ErrorCode::Internal, "The sign-in reply could not be read");
    };
    if let Ok(grant) = serde_json::from_slice::<AccessGrant>(&bytes) {
        bindings.bind(&grant.access_token, peer, grant.expires_at_ms);
    }
    Response::from_parts(parts, Body::from(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_pairing_and_sign_in_are_open() {
        assert!(before_sign_in(
            &Method::POST,
            "/client/v1/enrollments/redeem"
        ));
        assert!(before_sign_in(&Method::GET, "/client/v1/enrollments/of-1"));
        assert!(before_sign_in(
            &Method::POST,
            "/client/v1/sessions/challenge"
        ));
        assert!(before_sign_in(&Method::POST, "/client/v1/sessions"));
        assert!(!before_sign_in(&Method::GET, "/client/v1/enrollments/"));
        assert!(!before_sign_in(
            &Method::GET,
            "/client/v1/enrollments/of-1/x"
        ));
        assert!(!before_sign_in(&Method::POST, "/client/v1/pairing/offers"));
        assert!(!before_sign_in(&Method::POST, "/client/v1/commands"));
        assert!(!before_sign_in(&Method::GET, "/client/v1/ws"));
        assert!(!before_sign_in(&Method::GET, "/client/v1/devices"));
    }

    #[test]
    fn a_token_belongs_to_the_peer_it_was_issued_to() {
        let bindings = Bindings::<u8> {
            tokens: Mutex::default(),
        };
        let later = now_ms() + 60_000;
        bindings.bind("acs_one", 1, later);
        assert!(bindings.holds("acs_one", &1));
        assert!(!bindings.holds("acs_one", &2));
        assert!(!bindings.holds("acs_other", &1));
        assert!(bindings.signed_in(&1));
        assert!(!bindings.signed_in(&2));
        bindings.bind("acs_old", 3, now_ms().saturating_sub(1));
        assert!(!bindings.holds("acs_old", &3));
        assert!(!bindings.signed_in(&3));
    }
}
