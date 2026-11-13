#![allow(clippy::result_large_err)] // The handshake callback's error type is tungstenite's.

//! The TUI against a mock gateway: reconnecting resumes from the last
//! event and resends unanswered commands with their own ids; the screen
//! shows approval cards from their typed facts and keys answer them.

use std::time::Duration;

use aidaemon_client_protocol::*;
use aidaemon_tui::app::{Action, App, Focus};
use aidaemon_tui::link::{self, LinkConfig, Update};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use futures::{SinkExt, StreamExt};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
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

async fn next_update(updates: &mut mpsc::UnboundedReceiver<Update>) -> Update {
    tokio::time::timeout(Duration::from_secs(10), updates.recv())
        .await
        .unwrap()
        .unwrap()
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
            token: "acg_synthetic".into(),
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

fn approval() -> Approval {
    Approval {
        id: "a1".into(),
        session_id: "telegram:synthetic-user-1".into(),
        kind: ApprovalKind::Command,
        risk: Risk::High,
        summary: "Your assistant wants to delete the synthetic drafts".into(),
        detail: Some("rm -r /tmp/synthetic-drafts".into()),
        card: Some(ApprovalCard {
            asked_by: None,
            action: CardAction::Command {
                line: "rm -r /tmp/synthetic-drafts".into(),
                title: Some("Delete the drafts".into()),
                looks_inside: Vec::new(),
            },
            reason: Some(AskReason::AfterADecline),
            impact: Some(Impact::Deletes {
                targets: vec!["/tmp/synthetic-drafts".into()],
            }),
        }),
        warnings: Vec::new(),
        options: vec![
            ApprovalDecision::ApproveOnce,
            ApprovalDecision::ApproveSession,
            ApprovalDecision::Deny,
        ],
        requested_at_ms: 1,
        expires_at_ms: None,
        channel: Some("telegram".into()),
    }
}

fn screen(app: &App) -> String {
    let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    let mut text = String::new();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            text.push_str(buffer[(x, y)].symbol());
        }
        text.push('\n');
    }
    text
}

fn press(app: &mut App, code: KeyCode) -> Option<Action> {
    app.key(KeyEvent::new(code, KeyModifiers::NONE))
}

#[test]
fn the_card_shows_its_facts_and_keys_answer_it() {
    let mut app = App::default();
    app.connected = true;
    app.state.apply(event(
        1,
        Event::ApprovalRequested {
            approval: approval(),
        },
    ));
    app.state.apply(event(
        2,
        Event::TaskUpdated {
            task: TaskSummary {
                id: "b:telegram:synthetic-user-1#4".into(),
                session_id: "telegram:synthetic-user-1".into(),
                number: 4,
                description: "Working on your message".into(),
                state: TaskState::Running,
                started_at_ms: 1,
                finished_at_ms: None,
                error: None,
            },
        },
    ));
    let shown = screen(&app);
    for expected in [
        "delete the synthetic drafts",
        "Deletes /tmp/synthetic-drafts",
        "You said no to an earlier step",
        "[y] allow once",
        "[s] allow this session",
        "[n] deny",
        "also in telegram",
        "#4 Working on your message",
    ] {
        assert!(shown.contains(expected), "missing {expected:?} in\n{shown}");
    }

    // Tab reaches the card; `a` is not offered, `y` answers.
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.focus, Focus::Approval);
    assert_eq!(press(&mut app, KeyCode::Char('a')), None);
    assert_eq!(
        press(&mut app, KeyCode::Char('y')),
        Some(Action::Send(Command::ResolveApproval {
            approval_id: "a1".into(),
            decision: ApprovalDecision::ApproveOnce,
        }))
    );
    // The work pane cancels the selected entry.
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.focus, Focus::Tasks);
    assert_eq!(
        press(&mut app, KeyCode::Char('c')),
        Some(Action::Send(Command::CancelTask {
            task_id: "b:telegram:synthetic-user-1#4".into(),
        }))
    );
}

#[test]
fn typing_starts_a_session_then_follows_it() {
    let mut app = App::default();
    for c in "Draft the report".chars() {
        press(&mut app, KeyCode::Char(c));
    }
    let Some(Action::Send(Command::SendMessage {
        session_id, text, ..
    })) = press(&mut app, KeyCode::Enter)
    else {
        panic!("a message");
    };
    assert_eq!((session_id, text.as_str()), (None, "Draft the report"));
    let ack = ServerFrame::Ack {
        request_id: "r".into(),
        result: CommandResult::MessageAccepted {
            session_id: "app:new".into(),
            new_session: true,
        },
    };
    app.state.apply(ack.clone());
    app.after_frame(&ack);
    assert_eq!(app.selected.as_deref(), Some("app:new"));
    // `/stop` stops the session's work.
    for c in "/stop".chars() {
        press(&mut app, KeyCode::Char(c));
    }
    assert_eq!(
        press(&mut app, KeyCode::Enter),
        Some(Action::Send(Command::StopTurn {
            session_id: "app:new".into()
        }))
    );
}
