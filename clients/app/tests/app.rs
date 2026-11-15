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
    start_at(1280.0, 820.0)
}

fn start_at(width: f32, height: f32) -> Headless {
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(width), Points(height));
    Headless::start(config).unwrap()
}

/// How wide the conversation page was laid out.
fn conversation_width(app: &Headless) -> f32 {
    app.screen()
        .layout_nodes()
        .iter()
        .find(|n| n.test_id.as_deref() == Some("conversation") && !n.hidden)
        .map(|n| n.bounds.size.width.0)
        .unwrap_or(0.0)
}

#[test]
fn the_conversation_fills_a_phone_and_the_room_beside_the_panes() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    // A phone: the page alone, as wide as the screen.
    let mut app = start_at(390.0, 844.0);
    app.wait_for("What should we work on?")?;
    let phone = conversation_width(&app);
    assert!(
        phone > 380.0,
        "conversation is {phone} pt wide on a 390 pt screen"
    );
    // A roomy window: the page takes what the panes leave.
    let mut app = start();
    app.wait_for("What should we work on?")?;
    let desk = conversation_width(&app);
    assert!(
        desk > 600.0,
        "conversation is {desk} pt wide in a 1280 pt window"
    );
    Ok(())
}

#[test]
fn a_streamed_reply_an_approval_and_the_turn_s_activity() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-1","channel":"app","title":"Fix the flaky test","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Fix the flaky test")?;
    app.press("Fix the flaky test")?;
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-1","event":{"type":"user_message","session_id":"app:synthetic-1","text":"Why does it fail on CI?","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-1","event":{"type":"turn_started","turn_id":"t1","session_id":"app:synthetic-1","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","at_ms":3,"session_id":"app:synthetic-1","event":{"type":"turn_status","turn_id":"t1","status":{"kind":"tool_started","name":"terminal","summary":"cargo test -q"}}}"#,
        ),
        frame(
            r#"{"type":"event","at_ms":4,"session_id":"app:synthetic-1","event":{"type":"turn_delta","turn_id":"t1","text":"It races on the **shared** port."}}"#,
        ),
        frame(
            r#"{"type":"event","seq":3,"at_ms":5,"session_id":"app:synthetic-1","event":{"type":"approval_requested","approval":{"id":"ap1","session_id":"app:synthetic-1","kind":"command","risk":"medium","summary":"Run a command","options":["approve_once","deny"],"requested_at_ms":5,"card":{"action":{"kind":"command","line":"git push origin fix-port"}}}}}"#,
        ),
    ]);
    app.wait_for("Why does it fail on CI?")?;
    app.wait_for("terminal")?;
    app.wait_for("git push origin fix-port")?;
    app.wait_for("Allow once")?;
    // The conversation waits on its owner: it leads the list.
    app.wait_for("Needs you")?;
    app.wait_for("text:terminal: cargo test -q...  ·  1 tool so far")?;
    feed([frame(
        r#"{"type":"event","seq":4,"at_ms":6,"session_id":"app:synthetic-1","event":{"type":"turn_completed","turn_id":"t1","text":"It races on the shared port.","state":"completed"}}"#,
    )]);
    app.wait_for("Turn finished.")?;
    // The reply keeps a line of what its turn did.
    app.wait_for("Used 1 tool: terminal")?;
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

#[test]
fn the_agent_moves_the_view_and_says_why() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-2","channel":"app","title":"Plan for Acme Corp","writable":true}]}}"#,
        ),
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"event":{"type":"view_action","action":{"kind":"open_session","session_id":"app:synthetic-2"},"reason":"the plan is ready"}}"#,
        ),
    ]);
    app.wait_for("Moved to Plan for Acme Corp · the plan is ready")?;
    Ok(())
}

#[test]
fn the_settings_page_reads_the_settings_and_says_what_changed() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([Update::Connected]);
    app.press("Settings: change how the daemon runs")?;
    app.wait_for("Reading settings…")?;
    feed([frame(
        r#"{"type":"event","at_ms":1,"event":{"type":"settings_updated","ok":true,"message":"Updated memory.retention_days = 30"}}"#,
    )]);
    app.wait_for("Updated memory.retention_days = 30")?;
    Ok(())
}

#[test]
fn first_run_connects_a_model_then_offers_a_chat_and_greets() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    feed([
        Update::Connected,
        frame(
            r#"{"type":"welcome","version":{"major":1,"minor":5},"owner":{},"latest_seq":0,"setup":{"step":"model","found":[{"preset":"openai_chatgpt","name":"OpenAI (ChatGPT subscription) · gpt-synthetic","source":"ChatGPT sign-in on this computer","billing":"plan","works":true}],"providers":[{"preset":"openrouter","name":"OpenRouter","billing":"per_token","needs_key":true,"model":"synthetic/model-1"}]}}"#,
        ),
    ]);
    app.wait_for("Connect a model")?;
    app.wait_for("ChatGPT sign-in on this computer · uses your plan · has limits")?;
    app.wait_for("pay per token")?;
    // A key provider opens its form.
    app.press("Use a key")?;
    app.wait_for("Check key")?;
    // The key checks out: the account's own models, the usual one chosen.
    feed([frame(
        r#"{"type":"event","at_ms":1,"event":{"type":"setup_updated","setup":{"step":"model","providers":[{"preset":"openrouter","name":"OpenRouter","billing":"per_token","needs_key":true,"model":"synthetic/model-1"}],"checked":{"preset":"openrouter","models":["synthetic/model-1","synthetic/model-2"],"suggested":"synthetic/model-1"}}}}"#,
    )]);
    app.wait_for("The key works. Its account offers 2 models.")?;
    // The usual model is chosen in the picker.
    app.wait_for("synthetic/model-1")?;
    app.wait_for("Test and connect")?;
    // The daemon starts with the model, then asks about a chat.
    feed([frame(
        r#"{"type":"event","at_ms":1,"event":{"type":"setup_updated","setup":{"step":"starting","model":"OpenRouter · synthetic/model-1"}}}"#,
    )]);
    app.wait_for(
        "OpenRouter · synthetic/model-1 answered its test. AIdaemon is starting with it…",
    )?;
    feed([frame(
        r#"{"type":"event","at_ms":2,"event":{"type":"setup_updated","setup":{"step":"channel","model":"OpenRouter · synthetic/model-1","channels":[{"platform":"telegram","name":"Telegram","help_url":"https://t.me/BotFather"}],"channel":{"platform":"telegram","bot":"@synthetic_bot"}}}}"#,
    )]);
    app.wait_for("Reach it from your phone")?;
    app.wait_for("Waiting for your first message")?;
    feed([frame(
        r#"{"type":"event","at_ms":3,"event":{"type":"setup_updated","setup":{"step":"channel","channel":{"platform":"telegram","bot":"@synthetic_bot","connected":true}}}}"#,
    )]);
    app.wait_for("Connected: your agent answered you as @synthetic_bot.")?;
    app.press("Continue")?;
    // Done: the greeting's conversation opens, with prompts to press.
    feed([
        frame(
            r#"{"type":"event","at_ms":4,"event":{"type":"setup_updated","setup":{"step":"done"}}}"#,
        ),
        frame(
            r#"{"type":"event","seq":1,"at_ms":5,"session_id":"app:welcome","event":{"type":"session_created","session":{"id":"app:welcome","channel":"app","title":"Welcome","writable":true}}}"#,
        ),
        frame(
            r#"{"type":"event","seq":2,"at_ms":6,"session_id":"app:welcome","event":{"type":"assistant_message","session_id":"app:welcome","text":"Hi, I'm AIdaemon.","suggestions":["What can you do on this computer?"]}}"#,
        ),
    ]);
    app.wait_for("Hi, I'm AIdaemon.")?;
    app.wait_for("What can you do on this computer?")?;
    Ok(())
}
