//! The bridge end to end, in process: real WebSockets on a loopback port,
//! a daemon logging in with a test P-256 key, browsers relaying through it.

use std::net::SocketAddr;
use std::time::Duration;

use aidaemon_bridge::{Bridge, Config};
use aidaemon_bridge_proto::ids::route_id;
use aidaemon_bridge_proto::login::sign_login;
use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature, SigningKey};
use serde_json::{Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

const WAIT: Duration = Duration::from_secs(5);

fn test_config() -> Config {
    Config {
        connects_per_ip_per_minute: 10_000,
        logins_per_ip_per_minute: 10_000,
        waits_per_ip_per_minute: 10_000,
        ..Config::default()
    }
}

async fn start(config: Config) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(Bridge::new(config).serve(listener));
    addr
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_slice(&[seed; 32]).unwrap()
}

fn uncompressed(key: &SigningKey) -> Vec<u8> {
    key.verifying_key().to_sec1_point(false).as_bytes().to_vec()
}

/// The next data or close message, skipping pings; `None` once the stream ends.
async fn next(ws: &mut Ws) -> Option<Message> {
    loop {
        match tokio::time::timeout(WAIT, futures_next(ws)).await {
            Ok(Some(Ok(Message::Ping(_) | Message::Pong(_)))) => continue,
            Ok(Some(Ok(message))) => return Some(message),
            Ok(Some(Err(_)) | None) => return None,
            Err(_) => panic!("timed out waiting for a message"),
        }
    }
}

async fn futures_next(
    ws: &mut Ws,
) -> Option<Result<Message, tokio_tungstenite::tungstenite::Error>> {
    use futures_util::StreamExt;
    ws.next().await
}

async fn send(ws: &mut Ws, message: Message) {
    use futures_util::SinkExt;
    ws.send(message).await.unwrap();
}

async fn send_json(ws: &mut Ws, value: Value) {
    send(ws, Message::text(value.to_string())).await;
}

async fn recv_json(ws: &mut Ws) -> Value {
    match next(ws).await {
        Some(Message::Text(text)) => serde_json::from_str(&text).unwrap(),
        other => panic!("expected JSON, got {other:?}"),
    }
}

async fn recv_binary(ws: &mut Ws) -> Vec<u8> {
    match next(ws).await {
        Some(Message::Binary(bytes)) => bytes.to_vec(),
        other => panic!("expected binary, got {other:?}"),
    }
}

/// The socket closes (with `code`, when given) and nothing else arrives first.
async fn expect_closed(ws: &mut Ws, code: Option<u16>) {
    match next(ws).await {
        Some(Message::Close(frame)) => {
            if let Some(code) = code {
                let got = frame.map(|frame| u16::from(frame.code));
                assert_eq!(got, Some(code));
            }
        }
        None => assert!(code.is_none(), "closed without a close frame"),
        Some(other) => panic!("expected close, got {other:?}"),
    }
}

async fn open_daemon(addr: SocketAddr) -> (Ws, [u8; 32]) {
    let (mut ws, _) = connect_async(format!("ws://{addr}/v1/daemon"))
        .await
        .unwrap();
    let challenge = recv_json(&mut ws).await;
    assert_eq!(challenge["type"], "challenge");
    let nonce: [u8; 32] = hex::decode(challenge["nonce"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap();
    (ws, nonce)
}

async fn login(addr: SocketAddr, key: &SigningKey) -> (Ws, String) {
    let (mut ws, nonce) = open_daemon(addr).await;
    let login_pub = uncompressed(key);
    let sig = sign_login(key, &nonce);
    send_json(
        &mut ws,
        json!({"type": "login", "login_pub": hex::encode(&login_pub), "sig": hex::encode(sig)}),
    )
    .await;
    let ready = recv_json(&mut ws).await;
    assert_eq!(ready["type"], "ready");
    let route = hex::encode(route_id(&login_pub));
    assert_eq!(ready["route_id"], route.as_str());
    (ws, route)
}

async fn browser(addr: SocketAddr, route: &str) -> Ws {
    connect_async(format!("ws://{addr}/v1/connect?route={route}"))
        .await
        .unwrap()
        .0
}

/// A browser connected and opened on the daemon: `(browser, conn)`.
async fn joined(addr: SocketAddr, daemon: &mut Ws, route: &str) -> (Ws, u32) {
    let ws = browser(addr, route).await;
    let open = recv_json(daemon).await;
    assert_eq!(open["type"], "open");
    (ws, open["conn"].as_u64().unwrap() as u32)
}

fn prefixed(conn: u32, payload: &[u8]) -> Vec<u8> {
    let mut frame = conn.to_be_bytes().to_vec();
    frame.extend_from_slice(payload);
    frame
}

/// Promote `conn` and wait until its browser task has taken the promotion
/// (a frame sent after it arrives after it).
async fn promote(daemon: &mut Ws, browser: &mut Ws, conn: u32) {
    send_json(daemon, json!({"type": "promote", "conn": conn})).await;
    send(daemon, Message::binary(prefixed(conn, b"sync"))).await;
    assert_eq!(recv_binary(browser).await, b"sync");
}

#[tokio::test]
async fn login_succeeds_and_healthz_answers() {
    let addr = start(test_config()).await;
    login(addr, &key(1)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(response.starts_with("HTTP/1.1 200"));
    assert!(response.ends_with("ok"));
}

#[tokio::test]
async fn login_with_a_bad_signature_is_refused() {
    let addr = start(test_config()).await;
    let (mut ws, nonce) = open_daemon(addr).await;
    let mut other = nonce;
    other[0] ^= 1;
    let sig = sign_login(&key(1), &other);
    send_json(
        &mut ws,
        json!({"type": "login", "login_pub": hex::encode(uncompressed(&key(1))), "sig": hex::encode(sig)}),
    )
    .await;
    expect_closed(&mut ws, Some(1008)).await;
}

#[tokio::test]
async fn login_with_a_compressed_key_is_refused() {
    let addr = start(test_config()).await;
    let (mut ws, nonce) = open_daemon(addr).await;
    let compressed = key(1)
        .verifying_key()
        .to_sec1_point(true)
        .as_bytes()
        .to_vec();
    let sig = sign_login(&key(1), &nonce);
    send_json(
        &mut ws,
        json!({"type": "login", "login_pub": hex::encode(compressed), "sig": hex::encode(sig)}),
    )
    .await;
    expect_closed(&mut ws, Some(1008)).await;
}

#[tokio::test]
async fn login_signed_without_the_prefix_is_refused() {
    let addr = start(test_config()).await;
    let (mut ws, nonce) = open_daemon(addr).await;
    let mut message = b"aidaemon-bridge-hs-v1:".to_vec();
    message.extend_from_slice(&nonce);
    let sig: Signature = key(1).sign(&message);
    send_json(
        &mut ws,
        json!({"type": "login", "login_pub": hex::encode(uncompressed(&key(1))), "sig": hex::encode(sig.to_bytes())}),
    )
    .await;
    expect_closed(&mut ws, Some(1008)).await;
}

#[tokio::test]
async fn connect_to_an_unknown_route_is_refused() {
    let addr = start(test_config()).await;
    let route = hex::encode([7u8; 32]);
    assert!(
        connect_async(format!("ws://{addr}/v1/connect?route={route}"))
            .await
            .is_err()
    );
    // Uppercase hex is not a route.
    let (_daemon, route) = login(addr, &key(1)).await;
    let upper = route.to_ascii_uppercase();
    assert!(
        connect_async(format!("ws://{addr}/v1/connect?route={upper}"))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn frames_relay_both_ways_with_connection_ids() {
    let addr = start(test_config()).await;
    let (mut daemon, route) = login(addr, &key(1)).await;
    let (mut first, conn1) = joined(addr, &mut daemon, &route).await;
    let (mut second, conn2) = joined(addr, &mut daemon, &route).await;
    assert_ne!(conn1, conn2);

    send(&mut first, Message::binary(vec![1, 2, 3])).await;
    assert_eq!(recv_binary(&mut daemon).await, prefixed(conn1, &[1, 2, 3]));
    send(&mut second, Message::binary(vec![4])).await;
    assert_eq!(recv_binary(&mut daemon).await, prefixed(conn2, &[4]));

    send(&mut daemon, Message::binary(prefixed(conn2, &[9, 9]))).await;
    assert_eq!(recv_binary(&mut second).await, vec![9, 9]);
    send(&mut daemon, Message::binary(prefixed(conn1, &[8]))).await;
    assert_eq!(recv_binary(&mut first).await, vec![8]);
}

#[tokio::test]
async fn promote_raises_the_frame_limit() {
    let addr = start(test_config()).await;
    let (mut daemon, route) = login(addr, &key(1)).await;

    let (mut early, conn) = joined(addr, &mut daemon, &route).await;
    send(&mut early, Message::binary(vec![0u8; 1025])).await;
    expect_closed(&mut early, Some(1009)).await;
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "close", "conn": conn})
    );

    let (mut late, conn) = joined(addr, &mut daemon, &route).await;
    send(&mut late, Message::binary(vec![0u8; 1024])).await;
    assert_eq!(recv_binary(&mut daemon).await.len(), 4 + 1024);
    promote(&mut daemon, &mut late, conn).await;
    send(&mut late, Message::binary(vec![5u8; 64 * 1024])).await;
    let frame = recv_binary(&mut daemon).await;
    assert_eq!(frame.len(), 4 + 64 * 1024);
    assert_eq!(&frame[..4], conn.to_be_bytes());
    // Over the data limit even after promotion.
    send(&mut late, Message::binary(vec![0u8; 64 * 1024 + 1])).await;
    expect_closed(&mut late, None).await;
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "close", "conn": conn})
    );
}

#[tokio::test]
async fn a_handshake_not_promoted_in_time_is_closed() {
    let addr = start(Config {
        handshake_timeout: Duration::from_millis(300),
        ..test_config()
    })
    .await;
    let (mut daemon, route) = login(addr, &key(1)).await;
    let (mut slow, conn) = joined(addr, &mut daemon, &route).await;
    let (mut quick, quick_conn) = joined(addr, &mut daemon, &route).await;
    promote(&mut daemon, &mut quick, quick_conn).await;
    expect_closed(&mut slow, Some(1008)).await;
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "close", "conn": conn})
    );
    // The promoted one stays.
    tokio::time::sleep(Duration::from_millis(400)).await;
    send(&mut quick, Message::binary(vec![1])).await;
    assert_eq!(recv_binary(&mut daemon).await, prefixed(quick_conn, &[1]));
}

#[tokio::test]
async fn pairing_wait_join_and_join_failed() {
    let addr = start(test_config()).await;
    let (mut daemon, _route) = login(addr, &key(1)).await;
    let (mut waiting, _) = connect_async(format!("ws://{addr}/v1/wait?nameplate=7K3Q"))
        .await
        .unwrap();
    // The same nameplate cannot wait twice; bad nameplates are refused.
    for nameplate in ["7K3Q", "7k3q", "7K3I", "7K3QX"] {
        assert!(
            connect_async(format!("ws://{addr}/v1/wait?nameplate={nameplate}"))
                .await
                .is_err(),
            "{nameplate}"
        );
    }

    send_json(&mut daemon, json!({"type": "join", "nameplate": "ABCD"})).await;
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "join_failed", "nameplate": "ABCD"})
    );

    send_json(&mut daemon, json!({"type": "join", "nameplate": "7K3Q"})).await;
    let open = recv_json(&mut daemon).await;
    assert_eq!(open["type"], "open");
    let conn = open["conn"].as_u64().unwrap() as u32;
    send(&mut waiting, Message::binary(vec![1, 1])).await;
    assert_eq!(recv_binary(&mut daemon).await, prefixed(conn, &[1, 1]));
    send(&mut daemon, Message::binary(prefixed(conn, &[2]))).await;
    assert_eq!(recv_binary(&mut waiting).await, vec![2]);

    // Joined once, the nameplate is free again, and a second join fails.
    send_json(&mut daemon, json!({"type": "join", "nameplate": "7K3Q"})).await;
    assert_eq!(recv_json(&mut daemon).await["type"], "join_failed");
    let (_again, _) = connect_async(format!("ws://{addr}/v1/wait?nameplate=7K3Q"))
        .await
        .unwrap();
}

#[tokio::test]
async fn a_wait_ends_after_its_timeout_and_frees_the_nameplate() {
    let addr = start(Config {
        wait_timeout: Duration::from_millis(200),
        ..test_config()
    })
    .await;
    let (mut waiting, _) = connect_async(format!("ws://{addr}/v1/wait?nameplate=WXYZ"))
        .await
        .unwrap();
    expect_closed(&mut waiting, Some(1000)).await;
    connect_async(format!("ws://{addr}/v1/wait?nameplate=WXYZ"))
        .await
        .unwrap();
}

#[tokio::test]
async fn a_new_login_replaces_the_old_daemon() {
    let addr = start(test_config()).await;
    let (mut old, route) = login(addr, &key(1)).await;
    let (mut browser_old, _) = joined(addr, &mut old, &route).await;
    let (mut new, same_route) = login(addr, &key(1)).await;
    assert_eq!(route, same_route);
    expect_closed(&mut old, Some(1001)).await;
    expect_closed(&mut browser_old, Some(1000)).await;
    let (mut browser_new, conn) = joined(addr, &mut new, &route).await;
    send(&mut browser_new, Message::binary(vec![3])).await;
    assert_eq!(recv_binary(&mut new).await, prefixed(conn, &[3]));
}

#[tokio::test]
async fn per_daemon_connection_and_handshake_caps() {
    let addr = start(Config {
        max_conns_per_daemon: 3,
        max_handshaking_per_daemon: 2,
        ..test_config()
    })
    .await;
    let (mut daemon, route) = login(addr, &key(1)).await;
    let (mut a, conn_a) = joined(addr, &mut daemon, &route).await;
    let (_b, _) = joined(addr, &mut daemon, &route).await;
    // A third handshake while two are handshaking is refused.
    let mut refused = browser(addr, &route).await;
    expect_closed(&mut refused, Some(1013)).await;
    // Once one is promoted a third fits, and then the connection cap is hit.
    promote(&mut daemon, &mut a, conn_a).await;
    let (_c, _) = joined(addr, &mut daemon, &route).await;
    let mut refused = browser(addr, &route).await;
    expect_closed(&mut refused, Some(1013)).await;
    // A join is refused by the same caps.
    let (_waiting, _) = connect_async(format!("ws://{addr}/v1/wait?nameplate=0000"))
        .await
        .unwrap();
    send_json(&mut daemon, json!({"type": "join", "nameplate": "0000"})).await;
    assert_eq!(recv_json(&mut daemon).await["type"], "join_failed");
}

#[tokio::test]
async fn a_browser_text_frame_is_refused() {
    let addr = start(test_config()).await;
    let (mut daemon, route) = login(addr, &key(1)).await;
    let (mut ws, conn) = joined(addr, &mut daemon, &route).await;
    send(&mut ws, Message::text("{\"type\":\"promote\",\"conn\":1}")).await;
    expect_closed(&mut ws, Some(1003)).await;
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "close", "conn": conn})
    );
}

#[tokio::test]
async fn close_propagates_both_ways() {
    let addr = start(test_config()).await;
    let (mut daemon, route) = login(addr, &key(1)).await;

    let (mut by_daemon, conn) = joined(addr, &mut daemon, &route).await;
    send(&mut daemon, Message::binary(prefixed(conn, &[7]))).await;
    send_json(&mut daemon, json!({"type": "close", "conn": conn})).await;
    // Frames sent before the close still arrive.
    assert_eq!(recv_binary(&mut by_daemon).await, vec![7]);
    expect_closed(&mut by_daemon, Some(1000)).await;

    let (mut by_browser, conn) = joined(addr, &mut daemon, &route).await;
    by_browser.close(None).await.unwrap();
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "close", "conn": conn})
    );
}

#[tokio::test]
async fn unknown_daemon_messages_end_the_session() {
    let addr = start(test_config()).await;
    for bad in [
        json!({"type": "open", "conn": 1}),
        json!({"type": "promote", "conn": 1, "extra": true}),
        json!({"type": "join", "nameplate": "abcd"}),
    ] {
        let (mut daemon, route) = login(addr, &key(1)).await;
        let (mut ws, _) = joined(addr, &mut daemon, &route).await;
        send_json(&mut daemon, bad).await;
        expect_closed(&mut daemon, Some(1008)).await;
        expect_closed(&mut ws, Some(1000)).await;
    }
    // A binary frame too short for a connection id, too.
    let (mut daemon, _) = login(addr, &key(1)).await;
    send(&mut daemon, Message::binary(vec![0, 0, 1])).await;
    expect_closed(&mut daemon, Some(1008)).await;
}

#[tokio::test]
async fn daily_budget_closes_the_connection() {
    let addr = start(Config {
        daily_bytes_per_route: 10,
        ..test_config()
    })
    .await;
    let (mut daemon, route) = login(addr, &key(1)).await;
    let (mut ws, conn) = joined(addr, &mut daemon, &route).await;
    send(&mut ws, Message::binary(vec![0u8; 8])).await;
    assert_eq!(recv_binary(&mut daemon).await.len(), 12);
    send(&mut ws, Message::binary(vec![0u8; 8])).await;
    assert_eq!(
        recv_json(&mut daemon).await,
        json!({"type": "close", "conn": conn})
    );
    expect_closed(&mut ws, Some(1000)).await;
    // Spent: new connections are refused until the day rolls over.
    let mut refused = browser(addr, &route).await;
    expect_closed(&mut refused, Some(1013)).await;
}

#[tokio::test]
async fn per_ip_login_rate_limit() {
    let addr = start(Config {
        logins_per_ip_per_minute: 2,
        ..test_config()
    })
    .await;
    open_daemon(addr).await;
    open_daemon(addr).await;
    assert!(
        connect_async(format!("ws://{addr}/v1/daemon"))
            .await
            .is_err()
    );
}
