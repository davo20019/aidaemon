//! The gateway as a peer-to-peer transport serves it: the same routes,
//! behind a guard keyed on the peer's transport key. iroh
//! (`aidaemon-remote`) is the transport the daemon ships, behind its
//! `remote` feature; any other that authenticates its peers plugs in the
//! same way: serve [`remote_router`] with each request's peer key as a `P`
//! extension, and admit peers at the handshake with
//! [`crate::pairing::Pairing::admits`].
//!
//! Over such a transport the gateway is reachable from the internet, so:
//! - a peer with no session may reach only pairing and sign-in;
//! - an access token works only from the peer key that signed in for it,
//!   so a token lifted from one device is useless from another;
//! - the client socket may sign in by itself instead of bringing a token:
//!   its upgrade carries a proof ([`SOCKET_SIGN_IN_HEADER`]) bound to this
//!   daemon and the peer's key, taken once ([`Pairing::sign_in_socket`]),
//!   and the token it earns is bound to the peer as a sign-in's is;
//! - the owner's local token is refused: it is for this machine.
//!
//! The transport puts the peer's key on each request as an extension of
//! type `P` (`aidaemon_remote::RemotePeer`, its bytes through `AsRef`); a
//! request without one is refused. The gateway's own sign-in then decides
//! as on loopback, and a sign-in that succeeds records the peer's key as
//! the device's, so the transport admits it from then on
//! ([`crate::pairing::Pairing::admits`]).

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use aidaemon_client_protocol::{AccessGrant, ErrorCode, SOCKET_SIGN_IN_HEADER, SocketSignIn};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, Method, header};
use axum::middleware::Next;
use axum::response::Response;
use tokio_util::sync::CancellationToken;

use crate::ClientGateway;
use crate::bus::now_ms;
use crate::server::{bearer, error_response};

#[cfg(doc)]
use crate::pairing::Pairing;

/// The largest sign-in reply read back to bind its token.
const MAX_GRANT_BYTES: usize = 64 * 1024;

/// One access token as the guard keeps it: the peer it was issued to, when
/// it runs out, and what revoking its device cancels.
struct Binding<P> {
    peer: P,
    expires_at_ms: u64,
    revoked: CancellationToken,
}

impl<P: Eq> Binding<P> {
    fn live_for(&self, peer: &P, now: u64) -> bool {
        self.peer == *peer && self.expires_at_ms > now && !self.revoked.is_cancelled()
    }
}

/// Access tokens and the peer key each was issued to.
struct Bindings<P> {
    tokens: Mutex<HashMap<String, Binding<P>>>,
}

impl<P: Clone + Eq> Bindings<P> {
    /// Tie `token` to `peer` until `expires_at_ms`, or until `revoked` (its
    /// device's revocation) is cancelled.
    fn bind(&self, token: &str, peer: P, expires_at_ms: u64, revoked: CancellationToken) {
        let mut tokens = self.tokens.lock().unwrap_or_else(|p| p.into_inner());
        let now = now_ms();
        tokens.retain(|_, b| b.expires_at_ms > now && !b.revoked.is_cancelled());
        tokens.insert(
            sha(token),
            Binding {
                peer,
                expires_at_ms,
                revoked,
            },
        );
    }

    /// Whether `token` was issued to `peer` and has not run out.
    fn holds(&self, token: &str, peer: &P) -> bool {
        let tokens = self.tokens.lock().unwrap_or_else(|p| p.into_inner());
        tokens
            .get(&sha(token))
            .is_some_and(|b| b.live_for(peer, now_ms()))
    }

    /// Whether `peer` holds any live token of a device not revoked since:
    /// it signed in as a paired device.
    fn signed_in(&self, peer: &P) -> bool {
        let tokens = self.tokens.lock().unwrap_or_else(|p| p.into_inner());
        let now = now_ms();
        tokens.values().any(|b| b.live_for(peer, now))
    }
}

/// What revoking the device `token` belongs to cancels; a token the gateway
/// no longer knows gets one already cancelled.
fn revocation_of(gateway: &ClientGateway, token: &str) -> CancellationToken {
    gateway
        .principal(token)
        .map(|principal| principal.revoked)
        .unwrap_or_else(|| {
            let gone = CancellationToken::new();
            gone.cancel();
            gone
        })
}

use crate::util::sha;

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
    P: Clone + Eq + Hash + AsRef<[u8]> + Send + Sync + 'static,
{
    guarded::<P>(gateway.clone(), crate::exposed_router(gateway))
}

/// The bridge's routes ([`crate::server::bridge_routes`]: sign-in, commands
/// and message files only) behind the same peer guard, for the bridge
/// connector, which puts the device key its handshake proved on each
/// request as `P`. Build it once per connector: the guard's token bindings
/// live in it.
pub fn bridge_router<P>(gateway: Arc<ClientGateway>) -> Router
where
    P: Clone + Eq + Hash + AsRef<[u8]> + Send + Sync + 'static,
{
    guarded::<P>(gateway.clone(), crate::server::bridge_routes(gateway))
}

fn guarded<P>(gateway: Arc<ClientGateway>, routes: Router) -> Router
where
    P: Clone + Eq + Hash + AsRef<[u8]> + Send + Sync + 'static,
{
    let bindings = Arc::new(Bindings::<P> {
        tokens: Mutex::default(),
    });
    routes.layer(axum::middleware::from_fn_with_state(
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
    P: Clone + Eq + Hash + AsRef<[u8]> + Send + Sync + 'static,
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
            return bind_grant(response, &gateway, &bindings, peer).await;
        }
        return response;
    }
    if method == Method::GET
        && path == "/client/v1/ws"
        && bearer(request.headers()).is_none()
        && let Some(proof) = request.headers().get(SOCKET_SIGN_IN_HEADER)
    {
        return match sign_in_socket(&gateway, &bindings, peer, proof).await {
            Ok(token) => {
                // The socket goes on as one that brought this token: the
                // route checks it as it checks any.
                let mut request = request;
                request.headers_mut().insert(header::AUTHORIZATION, token);
                next.run(request).await
            }
            Err(error) => error_response(error.code, &error.message),
        };
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

/// Sign a socket's device in from the proof its upgrade carried, tie the
/// token to `peer` and give it back as the upgrade's `Authorization`.
async fn sign_in_socket<P: Clone + Eq + AsRef<[u8]>>(
    gateway: &ClientGateway,
    bindings: &Bindings<P>,
    peer: P,
    proof: &HeaderValue,
) -> Result<HeaderValue, aidaemon_client_protocol::ProtocolError> {
    let malformed = || {
        aidaemon_client_protocol::ProtocolError::new(
            ErrorCode::Unauthorized,
            "The socket's sign-in is malformed",
        )
    };
    let proof: SocketSignIn = proof
        .to_str()
        .ok()
        .and_then(|text| serde_json::from_str(text).ok())
        .ok_or_else(malformed)?;
    let grant = gateway
        .pairing
        .sign_in_socket(&proof, &hex::encode(peer.as_ref()))
        .await?;
    let revoked = revocation_of(gateway, &grant.access_token);
    bindings.bind(&grant.access_token, peer, grant.expires_at_ms, revoked);
    HeaderValue::from_str(&format!("Bearer {}", grant.access_token)).map_err(|_| malformed())
}

/// Read a successful sign-in reply and tie its token to `peer`.
async fn bind_grant<P: Clone + Eq + AsRef<[u8]>>(
    response: Response,
    gateway: &ClientGateway,
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
        // A client too old to name its endpoint at sign-in is admitted by
        // the one it signed in from.
        gateway
            .pairing
            .record_endpoint(&grant.device_id, &hex::encode(peer.as_ref()))
            .await;
        let revoked = revocation_of(gateway, &grant.access_token);
        bindings.bind(&grant.access_token, peer, grant.expires_at_ms, revoked);
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
        let revoked = CancellationToken::new();
        bindings.bind("acs_one", 1, later, revoked.clone());
        assert!(bindings.holds("acs_one", &1));
        assert!(!bindings.holds("acs_one", &2));
        assert!(!bindings.holds("acs_other", &1));
        assert!(bindings.signed_in(&1));
        assert!(!bindings.signed_in(&2));
        bindings.bind(
            "acs_old",
            3,
            now_ms().saturating_sub(1),
            CancellationToken::new(),
        );
        assert!(!bindings.holds("acs_old", &3));
        assert!(!bindings.signed_in(&3));
        // Revoking the device ends what its peer may reach at once, signed
        // file links included, not when the token would have run out.
        revoked.cancel();
        assert!(!bindings.holds("acs_one", &1));
        assert!(!bindings.signed_in(&1));
    }
}
