//! Two endpoints on this machine, no relay: HTTP and a WebSocket through
//! the router, the peer's key reaching the handler, and `Access` refusing a
//! key it does not know.

use std::collections::HashSet;
use std::sync::Arc;

use aidaemon_remote::{Access, Relays, RemoteClient, RemoteListener, RemotePeer, SecretKey};
use axum::Extension;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use http_body_util::Full;

fn app() -> axum::Router {
    axum::Router::new()
        .route(
            "/whoami",
            get(|Extension(RemotePeer(peer)): Extension<RemotePeer>| async move {
                peer.to_string()
            }),
        )
        .route(
            "/echo",
            get(|upgrade: WebSocketUpgrade| async move {
                upgrade.on_upgrade(echo).into_response()
            }),
        )
}

async fn echo(mut socket: WebSocket) {
    while let Some(Ok(Message::Text(text))) = socket.recv().await {
        if socket.send(Message::Text(text)).await.is_err() {
            break;
        }
    }
}

fn get_request(path: &str) -> http::Request<Full<Bytes>> {
    http::Request::get(path).body(Full::default()).unwrap()
}

#[tokio::test]
async fn http_and_websocket_reach_the_router_with_the_peer_key() {
    let listener =
        RemoteListener::bind(SecretKey::generate(), &Relays::Disabled, Access::Any, app())
            .await
            .unwrap();
    let client = RemoteClient::bind(SecretKey::generate(), &Relays::Disabled)
        .await
        .unwrap();

    let conn = client.connect(listener.addr()).await.unwrap();
    assert_eq!(conn.daemon(), listener.id());

    let response = conn.send(get_request("/whoami")).await.unwrap();
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(response.body().as_ref(), client.id().to_string().as_bytes());

    // A second request on the same connection takes a new stream.
    let missing = conn.send(get_request("/nowhere")).await.unwrap();
    assert_eq!(missing.status(), http::StatusCode::NOT_FOUND);

    let mut socket = conn
        .websocket("/echo", &http::HeaderMap::new())
        .await
        .unwrap();
    socket
        .send(tokio_tungstenite::tungstenite::Message::text("ping"))
        .await
        .unwrap();
    let reply = socket.next().await.unwrap().unwrap();
    assert_eq!(reply.into_text().unwrap().as_str(), "ping");

    client.close().await;
    listener.close().await;
}

#[tokio::test]
async fn access_refuses_keys_it_does_not_know() {
    let paired = SecretKey::generate();
    let allowed: HashSet<_> = [paired.public()].into();
    let listener = RemoteListener::bind(
        SecretKey::generate(),
        &Relays::Disabled,
        Access::Only(Arc::new(move |peer| allowed.contains(peer))),
        app(),
    )
    .await
    .unwrap();

    let stranger = RemoteClient::bind(SecretKey::generate(), &Relays::Disabled)
        .await
        .unwrap();
    // The handshake itself succeeds; the listener closes the connection
    // before serving a stream.
    let refused = match stranger.connect(listener.addr()).await {
        Ok(conn) => conn.send(get_request("/whoami")).await.is_err(),
        Err(_) => true,
    };
    assert!(refused, "a stranger's request was served");

    let friend = RemoteClient::bind(paired, &Relays::Disabled).await.unwrap();
    let conn = friend.connect(listener.addr()).await.unwrap();
    let response = conn.send(get_request("/whoami")).await.unwrap();
    assert_eq!(response.status(), http::StatusCode::OK);

    stranger.close().await;
    friend.close().await;
    listener.close().await;
}

#[tokio::test]
async fn a_client_refuses_a_daemon_with_another_key() {
    let listener =
        RemoteListener::bind(SecretKey::generate(), &Relays::Disabled, Access::Any, app())
            .await
            .unwrap();
    let client = RemoteClient::bind(SecretKey::generate(), &Relays::Disabled)
        .await
        .unwrap();
    // The right addresses, the wrong key: the TLS handshake must fail.
    let impostor = SecretKey::generate().public();
    let addr =
        aidaemon_remote::EndpointAddr::from_parts(impostor, listener.addr().addrs.iter().cloned());
    assert!(client.connect(addr).await.is_err());

    client.close().await;
    listener.close().await;
}
