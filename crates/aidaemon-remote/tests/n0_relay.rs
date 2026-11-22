//! Live: two endpoints that know each other only by key and relay URL (no
//! direct addresses), the way a paired phone dials its daemon, through the
//! project's relay and through n0's (addresses from before the project ran
//! its own). Needs the internet, so ignored by default:
//! `cargo test -p aidaemon-remote --test n0_relay -- --ignored`.

use aidaemon_remote::{Access, EndpointAddr, Relays, RemoteClient, RemoteListener, SecretKey};
use axum::routing::get;
use bytes::Bytes;
use http_body_util::Full;

#[tokio::test]
#[ignore = "needs the internet and the project's relay"]
async fn dials_through_the_projects_relay() {
    dials_by_key_and_relay_only(Relays::Aidaemon).await;
}

#[tokio::test]
#[ignore = "needs the internet and n0's public relays"]
async fn dials_through_n0s_relays() {
    dials_by_key_and_relay_only(Relays::N0).await;
}

async fn dials_by_key_and_relay_only(relays: Relays) {
    let app = axum::Router::new().route("/ping", get(|| async { "pong" }));
    let listener = RemoteListener::bind(SecretKey::generate(), &relays, Access::Any, app)
        .await
        .unwrap();
    listener.online().await;
    let relay = listener
        .addr()
        .relay_urls()
        .next()
        .cloned()
        .expect("online without a home relay");
    println!("daemon {} via {relay}", listener.id());

    if relays == Relays::Aidaemon {
        assert_eq!(relay.host_str(), Some("relay.aidaemon.ai"));
    }

    let client = RemoteClient::bind(SecretKey::generate(), &relays)
        .await
        .unwrap();
    let conn = client
        .connect(EndpointAddr::new(listener.id()).with_relay_url(relay))
        .await
        .unwrap();
    let response = conn
        .send(
            http::Request::get("/ping")
                .body(Full::<Bytes>::default())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.body().as_ref(), b"pong");

    client.close().await;
    listener.close().await;
}
