//! The TUI's screen: approval cards from their typed facts and keys that
//! answer them only once armed, Esc stopping work, the palette switching
//! sessions, and a status row that names the connection's state.

use std::time::{Duration, Instant};

use aidaemon_client_core::Update;
use aidaemon_client_protocol::*;
use aidaemon_tui::app::{Action, App, Focus};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

fn event(seq: u64, event: Event) -> ServerFrame {
    ServerFrame::Event {
        seq: Some(seq),
        at_ms: seq,
        session_id: Some("app:s1".into()),
        event,
    }
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

fn task() -> TaskSummary {
    TaskSummary {
        id: "b:telegram:synthetic-user-1#4".into(),
        session_id: "telegram:synthetic-user-1".into(),
        number: 4,
        description: "Working on your message".into(),
        state: TaskState::Running,
        started_at_ms: 1,
        finished_at_ms: None,
        error: None,
    }
}

pub fn screen(app: &App) -> String {
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

fn press_at(app: &mut App, code: KeyCode, at: Instant) -> Option<Action> {
    app.key_at(KeyEvent::new(code, KeyModifiers::NONE), at)
}

fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        press(app, KeyCode::Char(c));
    }
}

#[test]
fn the_card_shows_its_facts_and_keys_answer_it_once_armed() {
    let mut app = App::default();
    app.update(Update::Connected);
    app.update(Update::Frame(event(
        1,
        Event::ApprovalRequested {
            approval: approval(),
        },
    )));
    app.update(Update::Frame(event(2, Event::TaskUpdated { task: task() })));
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
        "Connected",
        "1 approval waiting",
    ] {
        assert!(shown.contains(expected), "missing {expected:?} in\n{shown}");
    }

    // A key pressed as the card appears is typed, not taken as an answer.
    let shown_at = Instant::now();
    app.observe_card(shown_at);
    assert_eq!(press_at(&mut app, KeyCode::Char('y'), shown_at), None);
    assert_eq!(app.input, "y");
    // With text in the box, the card's keys are just letters.
    let later = shown_at + Duration::from_secs(1);
    assert_eq!(press_at(&mut app, KeyCode::Char('n'), later), None);
    assert_eq!(app.input, "yn");
    app.input.clear();
    // Armed and with the box empty: `a` is not offered, `y` answers.
    assert_eq!(press_at(&mut app, KeyCode::Char('a'), later), None);
    app.input.clear();
    assert_eq!(
        press_at(&mut app, KeyCode::Char('y'), later),
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
    // Esc comes back to the message box.
    press(&mut app, KeyCode::Esc);
    assert_eq!(app.focus, Focus::Chat);
}

#[test]
fn typing_starts_a_session_follows_it_and_esc_stops_it() {
    let mut app = App::default();
    let shown = screen(&app);
    assert!(shown.contains("Connecting..."), "{shown}");
    assert!(
        shown.contains("Type a message to start your first session"),
        "{shown}"
    );
    type_text(&mut app, "Draft the report");
    let Some(Action::Send(Command::SendMessage {
        session_id, text, ..
    })) = press(&mut app, KeyCode::Enter)
    else {
        panic!("a message");
    };
    assert_eq!((session_id, text.as_str()), (None, "Draft the report"));
    app.update(Update::Frame(ServerFrame::Ack {
        request_id: "r".into(),
        result: CommandResult::MessageAccepted {
            session_id: "app:new".into(),
            new_session: true,
        },
    }));
    assert_eq!(app.selected.as_deref(), Some("app:new"));
    // Nothing runs yet: Esc says so and sends nothing.
    assert_eq!(press(&mut app, KeyCode::Esc), None);
    app.update(Update::Frame(ServerFrame::Event {
        seq: Some(1),
        at_ms: 1,
        session_id: Some("app:new".into()),
        event: Event::TurnStarted {
            turn_id: "t1".into(),
            session_id: "app:new".into(),
            origin: "app".into(),
        },
    }));
    assert!(screen(&app).contains("Esc to stop"));
    let stop = Some(Action::Send(Command::StopTurn {
        session_id: "app:new".into(),
    }));
    assert_eq!(press(&mut app, KeyCode::Esc), stop);
    // `/stop` does the same.
    type_text(&mut app, "/stop");
    assert_eq!(press(&mut app, KeyCode::Enter), stop);
    // `/st` and Tab completes.
    type_text(&mut app, "/st");
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.input, "/stop ");
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

#[test]
fn the_palette_switches_sessions_and_the_status_row_names_the_connection() {
    let mut app = App::default();
    app.update(Update::Connected);
    app.update(Update::Frame(ServerFrame::Ack {
        request_id: "r1".into(),
        result: CommandResult::Sessions {
            sessions: vec![
                session("app:s1", "Synthetic report"),
                session("app:s2", "Plan for Acme Corp"),
            ],
            next_cursor: None,
        },
    }));
    app.key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));
    type_text(&mut app, "acme");
    assert!(screen(&app).contains("Sessions and commands: acme"));
    assert_eq!(
        press(&mut app, KeyCode::Enter),
        Some(Action::Send(Command::GetHistory {
            session_id: "app:s2".into(),
            before: None,
            limit: 100,
        }))
    );
    assert_eq!(app.selected.as_deref(), Some("app:s2"));
    assert!(app.palette.is_none());
    // Alt-Down moves to the next session in one key.
    app.key(KeyEvent::new(KeyCode::Down, KeyModifiers::ALT));
    assert_eq!(app.selected.as_deref(), Some("app:s1"));

    app.update(Update::Disconnected(aidaemon_client_core::Disconnect {
        kind: aidaemon_client_core::DropKind::DaemonNotRunning,
        reason: "nothing is listening".into(),
        retry_in: Duration::from_secs(4),
    }));
    let shown = screen(&app);
    assert!(shown.contains("Daemon not running"), "{shown}");
    // Ctrl-C with an empty box quits.
    assert_eq!(
        app.key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Some(Action::Quit)
    );
}
