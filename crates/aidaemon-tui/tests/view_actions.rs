//! The agent-driven view in the TUI: over the real link against a mock
//! gateway, a view action moves the screen and shows a note of what moved
//! and why; no view action ever makes the TUI answer an approval or send
//! anything but a history read; a move that arrives while the owner types
//! is only offered.

use std::time::{Duration, Instant};

use aidaemon_client_core::link::{self, Auth, Update};
use aidaemon_client_core::state::ViewRequest;
use aidaemon_client_protocol::*;
use aidaemon_tui::app::{Action, App, Focus};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use futures::{SinkExt, StreamExt};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

fn screen(app: &App) -> String {
    let mut terminal = Terminal::new(TestBackend::new(160, 40)).unwrap();
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

fn approval() -> Approval {
    Approval {
        id: "a1".into(),
        session_id: "app:s1".into(),
        kind: ApprovalKind::Command,
        risk: Risk::Medium,
        summary: "Your assistant wants to publish the synthetic report".into(),
        detail: None,
        card: None,
        warnings: Vec::new(),
        options: vec![ApprovalDecision::ApproveOnce, ApprovalDecision::Deny],
        requested_at_ms: 1,
        expires_at_ms: None,
        channel: None,
    }
}

fn session(id: &str, title: &str) -> SessionSummary {
    SessionSummary {
        id: id.into(),
        channel: "app".into(),
        title: Some(title.into()),
        last_activity_ms: None,
        writable: true,
    }
}

fn view(action: ViewAction, reason: &str) -> ServerFrame {
    ServerFrame::Event {
        seq: None,
        at_ms: 9,
        session_id: Some("app:s1".into()),
        event: Event::ViewAction {
            action,
            reason: reason.into(),
        },
    }
}

fn text(frame: &ServerFrame) -> Message {
    Message::text(serde_json::to_string(frame).unwrap())
}

/// An app with two sessions, a waiting card and running work.
fn busy_app() -> App {
    let mut app = App::default();
    app.update(Update::Connected);
    app.update(Update::Frame(ServerFrame::Snapshot {
        seq: 3,
        state: Snapshot {
            sessions: vec![
                session("app:s1", "Synthetic report"),
                session("app:s2", "Plan for Acme Corp"),
            ],
            running_tasks: vec![TaskSummary {
                id: "b:app:s1#4".into(),
                session_id: "app:s1".into(),
                number: 4,
                description: "Building the report".into(),
                state: TaskState::Running,
                started_at_ms: 1,
                finished_at_ms: None,
                error: None,
            }],
            pending_approvals: vec![approval()],
        },
    }));
    app
}

#[tokio::test]
async fn a_view_action_over_the_link_moves_the_screen_with_a_note() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "ws://127.0.0.1:{}/client/v1/ws",
        listener.local_addr().unwrap().port()
    );
    let (updates_tx, mut updates) = mpsc::unbounded_channel();
    let _commands = link::spawn(
        aidaemon_tui::link_config(url, Auth::LocalToken("acg_synthetic".into()), None),
        updates_tx,
    );
    let (stream, _) = listener.accept().await.unwrap();
    let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
    let hello = loop {
        if let Some(Ok(Message::Text(text))) = socket.next().await {
            break serde_json::from_str::<ClientFrame>(&text).unwrap();
        }
    };
    let ClientFrame::Hello { capabilities, .. } = hello else {
        panic!("hello first");
    };
    assert!(capabilities.contains(&Capability::ViewActions));
    socket
        .send(text(&ServerFrame::Snapshot {
            seq: 1,
            state: Snapshot {
                sessions: vec![
                    session("app:s1", "Synthetic report"),
                    session("app:s2", "Plan for Acme Corp"),
                ],
                running_tasks: Vec::new(),
                pending_approvals: Vec::new(),
            },
        }))
        .await
        .unwrap();
    socket
        .send(text(&view(
            ViewAction::OpenSession {
                session_id: "app:s2".into(),
            },
            "You asked for yesterday's plan",
        )))
        .await
        .unwrap();

    let mut app = App::default();
    let mut sent = Vec::new();
    while app.selected.as_deref() != Some("app:s2") {
        let update = tokio::time::timeout(Duration::from_secs(10), updates.recv())
            .await
            .expect("an update in time")
            .expect("the link runs");
        sent.extend(app.update(update));
    }
    let shown = screen(&app);
    assert!(
        shown.contains("Moved to open Plan for Acme Corp (You asked for yesterday's plan)"),
        "{shown}"
    );
    assert!(shown.contains("Ctrl-B back"), "{shown}");
    assert!(sent.iter().all(|command| matches!(
        command,
        Command::ListSessions { .. } | Command::GetHistory { .. }
    )));
    // Ctrl-B goes back to where the owner was.
    app.key(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::CONTROL));
    assert_eq!(app.selected, None);
    assert!(app.note.is_none());
}

/// Every kind of view action, applied with a card waiting: the TUI asks
/// for nothing but history, the card stays, and a key pressed right after
/// the move is never an answer.
#[test]
fn no_view_action_answers_the_card_or_sends_anything_but_a_read() {
    let actions = [
        ViewAction::OpenSession {
            session_id: "app:s2".into(),
        },
        ViewAction::ShowApproval {
            approval_id: "a1".into(),
        },
        ViewAction::ShowTask {
            task_id: "b:app:s1#4".into(),
        },
        ViewAction::FocusPane {
            pane: Pane::Approvals,
        },
        ViewAction::FocusPane { pane: Pane::Tasks },
        ViewAction::NewSession,
        ViewAction::ShowDevices,
        ViewAction::Unknown,
    ];
    for action in actions {
        let mut app = busy_app();
        // The card has been on screen long enough to be armed...
        let start = Instant::now();
        app.observe_card(start - Duration::from_secs(5));
        let moved_at = Instant::now();
        let commands = app.view_request(
            ViewRequest {
                action: action.clone(),
                reason: "Look".into(),
            },
            moved_at,
        );
        assert!(
            commands
                .iter()
                .all(|c| matches!(c, Command::GetHistory { .. })),
            "{action:?} sent {commands:?}"
        );
        assert!(app.state.approvals.contains_key("a1"), "{action:?}");
        // ...but a key right after the move answers nothing.
        let pressed = app.key_at(
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE),
            moved_at,
        );
        assert!(
            !matches!(pressed, Some(Action::Send(Command::ResolveApproval { .. }))),
            "{action:?} let a key answer the card at once: {pressed:?}"
        );
    }
}

#[test]
fn show_approval_brings_the_card_up_armed_afresh() {
    let mut app = busy_app();
    let mut second = approval();
    second.id = "a2".into();
    second.summary = "Your assistant wants to email the synthetic invoice".into();
    second.requested_at_ms = 5;
    app.update(Update::Frame(ServerFrame::Event {
        seq: Some(4),
        at_ms: 4,
        session_id: Some("app:s1".into()),
        event: Event::ApprovalRequested { approval: second },
    }));
    let at = Instant::now();
    app.view_request(
        ViewRequest {
            action: ViewAction::ShowApproval {
                approval_id: "a2".into(),
            },
            reason: "You asked what is waiting".into(),
        },
        at,
    );
    assert_eq!(app.shown_approval().map(|a| a.id.as_str()), Some("a2"));
    assert!(screen(&app).contains("email the synthetic invoice"));
    assert!(app.armed_card(at).is_none());
    // Once read, the owner answers the card that is shown, not the oldest.
    let later = at + Duration::from_secs(1);
    assert_eq!(
        app.key_at(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE), later),
        Some(Action::Send(Command::ResolveApproval {
            approval_id: "a2".into(),
            decision: ApprovalDecision::Deny,
        }))
    );
}

#[test]
fn a_move_while_typing_is_only_offered() {
    let mut app = busy_app();
    for c in "draft a reply".chars() {
        app.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
    }
    let commands = app.view_request(
        ViewRequest {
            action: ViewAction::OpenSession {
                session_id: "app:s2".into(),
            },
            reason: "Newer plan".into(),
        },
        Instant::now(),
    );
    assert!(commands.is_empty());
    assert_eq!(app.selected, None, "the message keeps its target");
    assert_eq!(app.input, "draft a reply");
    assert!(app.holds_move());
    assert!(screen(&app).contains("Ctrl-G to go there"));
    let go = app.key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::CONTROL));
    assert_eq!(app.selected.as_deref(), Some("app:s2"));
    assert!(matches!(go, Some(Action::Send(Command::GetHistory { .. }))));
    assert_eq!(app.focus, Focus::Chat);
}
