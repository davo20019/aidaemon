//! The app on the real engine, headless, fed gateway frames as the link
//! would deliver them.

use aidaemon_client_core::Update;
use aidaemon_client_protocol::ServerFrame;
use nsh_app_aidaemon::{Offline, feed};
use nsh_engine::{EngineConfig, Points, Root, Size};
use nsh_testing::Headless;

fn frame(json: &str) -> Update {
    Update::Frame(serde_json::from_str::<ServerFrame>(json).expect("a frame"))
}

/// The feed is one per process: tests take turns.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn start() -> Headless {
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    Headless::start(config).unwrap()
}

#[test]
fn a_streamed_reply_an_approval_and_the_turn_s_activity() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-1","channel":"app","title":"Fix the flaky test","writable":true}]}}"#),
    ]);
    app.wait_for("Fix the flaky test")?;
    app.press("Fix the flaky test")?;
    feed([
        frame(r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-1","event":{"type":"user_message","session_id":"app:synthetic-1","text":"Why does it fail on CI?","origin":"app"}}"#),
        frame(r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-1","event":{"type":"turn_started","turn_id":"t1","session_id":"app:synthetic-1","origin":"app"}}"#),
        frame(r#"{"type":"event","at_ms":3,"session_id":"app:synthetic-1","event":{"type":"turn_status","turn_id":"t1","status":{"kind":"tool_started","name":"terminal","summary":"cargo test -q"}}}"#),
        frame(r#"{"type":"event","at_ms":4,"session_id":"app:synthetic-1","event":{"type":"turn_delta","turn_id":"t1","text":"It races on the **shared** port."}}"#),
        frame(r#"{"type":"event","seq":3,"at_ms":5,"session_id":"app:synthetic-1","event":{"type":"approval_requested","approval":{"id":"ap1","session_id":"app:synthetic-1","kind":"command","risk":"medium","summary":"Run a command","options":["approve_once","deny"],"requested_at_ms":5,"card":{"action":{"kind":"command","line":"git push origin fix-port"}}}}}"#),
    ]);
    app.wait_for("Why does it fail on CI?")?;
    app.wait_for("terminal")?;
    app.wait_for("git push origin fix-port")?;
    app.wait_for("Allow once")?;
    feed([frame(
        r#"{"type":"event","seq":4,"at_ms":6,"session_id":"app:synthetic-1","event":{"type":"turn_completed","turn_id":"t1","text":"It races on the shared port.","state":"completed"}}"#,
    )]);
    app.wait_for("Turn finished.")?;
    Ok(())
}

#[test]
fn offline_says_why() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    feed([Update::Disconnected(aidaemon_client_core::Disconnect {
        kind: aidaemon_client_core::DropKind::DaemonNotRunning,
        reason: "refused".into(),
        retry_in: std::time::Duration::from_secs(2),
    })]);
    app.wait_for("The daemon is not running")?;
    Ok(())
}
