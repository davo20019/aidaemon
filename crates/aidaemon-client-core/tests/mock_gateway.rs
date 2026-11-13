#![allow(clippy::result_large_err)] // The handshake callback's error type is tungstenite's.

//! The link against a mock gateway: reconnecting resumes from the last
//! event and resends unanswered commands with their own ids.

use std::time::Duration;

use aidaemon_client_core::link::{self, Auth, ClientIdentity, LinkConfig, Update};
use aidaemon_client_protocol::*;
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

fn event(seq: u64, event: Event) -> ServerFrame {
    ServerFrame::Event {
        seq: Some(seq),
        at_ms: seq,
        session_id: Some("app:s1".into()),
        event,
    }
}

fn text(frame: &ServerFrame) -> Message {
    Message::text(serde_json::to_string(frame).unwrap())
}

async fn read_client(
    socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
) -> ClientFrame {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(10), socket.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let Message::Text(text) = message {
            let frame: ClientFrame = serde_json::from_str(&text).unwrap();
            if frame != ClientFrame::Ping {
                return frame;
            }
        }
    }
}

/// The next update that is not `Connecting`.
async fn next_update(updates: &mut mpsc::UnboundedReceiver<Update>) -> Update {
    loop {
        let update = tokio::time::timeout(Duration::from_secs(10), updates.recv())
            .await
            .unwrap()
            .unwrap();
        if update != Update::Connecting {
            return update;
        }
    }
}

#[tokio::test]
async fn the_link_resumes_and_resends_unanswered_commands() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "ws://127.0.0.1:{}/client/v1/ws",
        listener.local_addr().unwrap().port()
    );
    let (updates_tx, mut updates) = mpsc::unbounded_channel();
    let commands = link::spawn(
        LinkConfig {
            url,
            auth: Auth::LocalToken("acg_synthetic".into()),
            client: ClientIdentity {
                kind: ClientKind::Tui,
                name: "aidaemon-tui".into(),
                version: None,
            },
            device: Some("Synthetic Laptop".into()),
            capabilities: Vec::new(),
        },
        updates_tx,
    );

    // First connection: no cursor; three events; a command goes
    // unanswered; then the gateway goes away.
    let (stream, _) = listener.accept().await.unwrap();
    let mut socket = tokio_tungstenite::accept_hdr_async(
        stream,
        |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
            assert_eq!(request.headers()["authorization"], "Bearer acg_synthetic");
            Ok(response)
        },
    )
    .await
    .unwrap();
    let ClientFrame::Hello { resume_from, .. } = read_client(&mut socket).await else {
        panic!("hello first");
    };
    assert_eq!(resume_from, None);
    assert_eq!(next_update(&mut updates).await, Update::Connected);
    for seq in 1..=3 {
        socket
            .send(text(&event(
                seq,
                Event::AssistantMessage {
                    session_id: "app:s1".into(),
                    text: format!("note {seq}"),
                },
            )))
            .await
            .unwrap();
    }
    for _ in 1..=3 {
        assert!(matches!(next_update(&mut updates).await, Update::Frame(_)));
    }
    commands
        .send((
            "req-1".into(),
            Command::StopTurn {
                session_id: "app:s1".into(),
            },
        ))
        .unwrap();
    let ClientFrame::Command { request_id, .. } = read_client(&mut socket).await else {
        panic!("the command");
    };
    assert_eq!(request_id, "req-1");
    drop(socket);
    assert!(matches!(
        next_update(&mut updates).await,
        Update::Disconnected(_)
    ));

    // Second connection: resumes after seq 3 and resends req-1.
    let (stream, _) = listener.accept().await.unwrap();
    let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
    let ClientFrame::Hello {
        resume_from,
        client,
        ..
    } = read_client(&mut socket).await
    else {
        panic!("hello first");
    };
    assert_eq!(resume_from, Some(3));
    assert_eq!(client.kind, ClientKind::Tui);
    let ClientFrame::Command { request_id, .. } = read_client(&mut socket).await else {
        panic!("the resent command");
    };
    assert_eq!(request_id, "req-1");
    socket
        .send(text(&ServerFrame::Ack {
            request_id: "req-1".into(),
            result: CommandResult::Stopped {
                report: "Nothing is running in this chat right now.".into(),
            },
        }))
        .await
        .unwrap();
    assert_eq!(next_update(&mut updates).await, Update::Connected);
    assert!(matches!(
        next_update(&mut updates).await,
        Update::Frame(ServerFrame::Ack { .. })
    ));
}
