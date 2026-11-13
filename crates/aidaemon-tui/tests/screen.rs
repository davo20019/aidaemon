//! The TUI's screen: approval cards from their typed facts, keys that
//! answer them, and a typed message that starts a session and follows it.

use aidaemon_client_protocol::*;
use aidaemon_tui::app::{Action, App, Focus};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

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
