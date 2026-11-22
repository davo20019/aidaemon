//! The app on the real engine, headless, fed gateway frames as the link
//! would deliver them.

use aidaemon_client_core::Update;
use aidaemon_client_protocol::ServerFrame;
use nsh_app_aidaemon::{Offline, feed, sent};
use nsh_engine::{EngineConfig, Points, Root, Size};
use nsh_testing::Headless;

fn frame(json: &str) -> Update {
    Update::Frame(serde_json::from_str::<ServerFrame>(json).expect("a frame"))
}

/// The id of the last command of `kind` the app sent, once it has.
fn wait_sent(kind: &str) -> Result<String, String> {
    for _ in 0..100 {
        if let Some(id) = sent(kind) {
            return Ok(id);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Err(format!("no {kind} was sent"))
}

/// The id of a command of `kind` sent after `before` (the last such id
/// when the step began): earlier tests' commands stay in the list.
fn wait_sent_new(kind: &str, before: Option<String>) -> Result<String, String> {
    for _ in 0..100 {
        if let Some(id) = sent(kind).filter(|id| Some(id) != before.as_ref()) {
            return Ok(id);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Err(format!("no new {kind} was sent"))
}

/// The feed is one per process: tests take turns.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn start() -> Headless {
    start_at(1280.0, 820.0)
}

/// The tests never reach a daemon on this machine: owner pairing (the
/// phone step's QR code) signs in with the local token, so point it at a
/// file that does not exist.
fn no_local_daemon() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: set once, before any test starts an engine that reads it.
        unsafe {
            std::env::remove_var("AIDAEMON_CLIENT_TOKEN");
            std::env::set_var(
                "AIDAEMON_CLIENT_TOKEN_FILE",
                std::env::temp_dir().join("aidaemon-app-tests-no-daemon/client-token"),
            );
        }
    });
}

fn start_at(width: f32, height: f32) -> Headless {
    no_local_daemon();
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(width), Points(height));
    Headless::start(config).unwrap()
}

/// A desktop window, with a mouse to hover with.
fn start_desktop() -> Headless {
    no_local_daemon();
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.os = nsh_core::platform::Os::MacOs;
    config.environment.idiom = nsh_core::platform::Idiom::Desktop;
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
    app.wait_for("What can I help you with?")?;
    let phone = conversation_width(&app);
    assert!(
        phone > 380.0,
        "conversation is {phone} pt wide on a 390 pt screen"
    );
    // A roomy window: the page takes what the panes leave.
    let mut app = start();
    app.wait_for("What can I help you with?")?;
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
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-1","channel":"app","title":"Fix the flaky test","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Fix the flaky test")?;
    app.press("Fix the flaky test >> nth=0")?;
    // A conversation away from code has no details pane: its steps are
    // in the chat, as they run and under the reply after.
    assert!(app.press("Steps").is_err());
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-1","event":{"type":"user_message","session_id":"app:synthetic-1","text":"Why does it fail on CI?","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-1","event":{"type":"turn_started","turn_id":"t1","session_id":"app:synthetic-1","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","at_ms":3,"session_id":"app:synthetic-1","event":{"type":"turn_status","turn_id":"t1","status":{"kind":"tool_started","name":"terminal","summary":"cargo test -q","activity":"run"}}}"#,
        ),
        // The command's output, its colours a terminal's.
        frame(
            r#"{"type":"event","at_ms":3,"session_id":"app:synthetic-1","event":{"type":"turn_status","turn_id":"t1","status":{"kind":"tool_progress","name":"terminal","chunk":"running 3 tests\n\u001b[31mtest port_race ... FAILED\u001b[0m\n"}}}"#,
        ),
        frame(
            r#"{"type":"event","at_ms":4,"session_id":"app:synthetic-1","event":{"type":"turn_delta","turn_id":"t1","text":"It races on the **shared** port."}}"#,
        ),
        frame(
            r#"{"type":"event","seq":3,"at_ms":5,"session_id":"app:synthetic-1","event":{"type":"approval_requested","approval":{"id":"ap1","session_id":"app:synthetic-1","kind":"command","risk":"medium","summary":"Run a command","options":["approve_once","deny"],"requested_at_ms":5,"card":{"action":{"kind":"command","line":"git push origin fix-port"}}}}}"#,
        ),
    ]);
    app.wait_for("Why does it fail on CI?")?;
    app.wait_for("label:Tools so far")?;
    app.wait_for("1 tool")?;
    app.wait_for("git push origin fix-port")?;
    app.wait_for("Allow once")?;
    // The conversation waits on its owner: it leads the list.
    app.wait_for("Needs you")?;
    // Waiting on its owner, the working line says so, not the last tool.
    app.wait_for("text:Waiting for your approval")?;
    feed([frame(
        r#"{"type":"event","seq":4,"at_ms":6,"session_id":"app:synthetic-1","event":{"type":"turn_completed","turn_id":"t1","text":"It races on the shared port.","state":"completed"}}"#,
    )]);
    // The reply keeps a line of what its turn did, in words; opened, each
    // call reads as what it did to what.
    app.wait_for("Ran 1 command")?;
    app.press("Ran 1 command")?;
    app.wait_for("text:Ran")?;
    app.wait_for("text:cargo test -q")?;
    Ok(())
}

#[test]
fn a_site_scoped_click_card_names_the_site_and_the_button() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-2","channel":"app","title":"Order the synthetic parts","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Order the synthetic parts")?;
    app.press("Order the synthetic parts >> nth=0")?;
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":5,"session_id":"app:synthetic-2","event":{"type":"approval_requested","approval":{"id":"ap-click","session_id":"app:synthetic-2","kind":"command","risk":"medium","summary":"Your assistant wants to click a button on https://shop.acme.example in the browser.","detail":"button#checkout","options":["approve_once","approve_session","deny"],"requested_at_ms":5,"card":{"action":{"kind":"click_on_site","site":"https://shop.acme.example","target":"button#checkout"}}}}}"#,
    )]);
    app.wait_for("text:Click buttons on https://shop.acme.example")?;
    app.wait_for("button#checkout")?;
    app.wait_for("Allow this session")?;
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
    app.wait_for("Can't reach the daemon")?;
    Ok(())
}

#[test]
fn a_dropped_connection_does_not_keep_a_turn_spinning() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-12","channel":"app","title":"Check the backups","writable":true}]}}"#,
        ),
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-12","event":{"type":"turn_started","turn_id":"t12","session_id":"app:synthetic-12","origin":"app"}}"#,
        ),
    ]);
    app.wait_for("label:Working")?;
    // Offline past the quiet few seconds: the app no longer knows.
    feed([Update::Disconnected(aidaemon_client_core::Disconnect {
        kind: aidaemon_client_core::DropKind::Dropped,
        reason: "the socket ended".into(),
        retry_in: std::time::Duration::from_secs(2),
    })]);
    app.wait_for_within(
        "label:Was working when the connection dropped",
        std::time::Duration::from_secs(10),
    )?;
    // Back, with the finish it missed: the ring goes.
    feed([
        Update::Connected,
        frame(
            r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-12","event":{"type":"turn_completed","turn_id":"t12","text":"The backups ran.","state":"completed"}}"#,
        ),
    ]);
    app.wait_for_gone(
        "label:Was working when the connection dropped",
        std::time::Duration::from_secs(5),
    )?;
    Ok(())
}

#[test]
fn the_name_carries_the_connection_as_a_dot() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    feed([Update::Connected]);
    app.wait_for("label:Connected to the daemon")?;
    Ok(())
}

#[test]
fn the_name_goes_home() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-9","channel":"app","title":"Garden notes","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Garden notes")?;
    app.press("Garden notes >> nth=0")?;
    app.wait_for_gone(
        "What can I help you with?",
        std::time::Duration::from_secs(5),
    )?;
    app.press("AIDaemon, home")?;
    app.wait_for("What can I help you with?")?;
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
    app.wait_for("Moved to Plan for Acme Corp")?;
    app.wait_for("the plan is ready")?;
    Ok(())
}

/// A daemon's settings, as `get_settings` answers.
const SETTINGS: &str = r#"{"settings": {"bots": [{"allowed_user_ids": ["telegram:synthetic-user-1"], "created_at": "2026-01-01 00:00:00", "id": 3, "platform": "telegram"}], "chatgpt_signed_in": true, "checked": {"models": ["gpt-x", "gpt-y"], "preset": "openai", "suggested": "gpt-x"}, "config_path": "/home/synthetic/.aidaemon/config.toml", "connected": [{"credential": "signed_in", "models": ["synthetic-model-1"], "hard_model": "synthetic-strong", "quick_model": "synthetic-quick", "name": "OpenAI (ChatGPT subscription)", "preset": "openai_chatgpt", "usage": {"plan": "plus", "windows": [{"minutes": 300, "used_percent": 42}, {"minutes": 10080, "used_percent": 18}], "read_at_ms": 1700000000000}}, {"base_url": "https://api.example.com/v1", "credential": "key_in_file", "failover": 0, "key_setting": "provider.fallbacks[0].api_key", "models": ["gpt-x"], "offered_models": ["gpt-x", "gpt-astra-synthetic"], "quick_model": "gpt-x", "name": "OpenAI", "preset": "openai"}, {"credential": "key_stored", "failover": 1, "key_setting": "provider.fallbacks[1].api_key", "models": ["synthetic-a"], "name": "Anthropic", "preset": "anthropic", "usage": {"balance": {"used_cents": 350, "limit_cents": 1000, "remaining_cents": 650}, "read_at_ms": 1700000000000}}], "database_encrypted": true, "database_path": "/home/synthetic/.aidaemon/aidaemon.db", "groups": [{"name": "provider", "settings": [{"applies": "reload", "description": "The model most turns use.", "example": "\"openai/gpt-4o\"", "guarded": false, "key": "provider.models.default", "kind": "text", "secret_set": false, "value": "\"synthetic-model-1\""}, {"applies": "reload", "guarded": true, "key": "provider.api_key", "kind": "secret", "secret_set": true}]}], "mcp_servers": [{"command": "synthetic-mcp --root /tmp", "enabled": true, "from_config": false, "name": "files", "tools": ["read_file"]}], "notices": ["The database moves to /data/aidaemon.db at the next restart."], "providers": [{"billing": "per_token", "key_url": "https://platform.openai.com/api-keys", "model": "gpt-x", "name": "OpenAI", "needs_base_url": false, "needs_key": true, "preset": "openai"}], "reload_pending": true, "restart_pending": false}, "type": "settings"}"#;

#[test]
fn the_terminal_settings_are_one_page_of_models_reached_by_keyboard() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    no_local_daemon();
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.idiom = nsh_core::platform::Idiom::Terminal;
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What can I help you with?")?;
    let before = sent("get_settings");
    feed([Update::Connected]);
    // From the keyboard: the command palette, then Settings.
    app.press_key("mod+k")?;
    app.wait_for("Settings: change how the daemon runs")?;
    app.type_text("label:Commands", "settings")?;
    app.press_key("Enter")?;
    let id = wait_sent_new("get_settings", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{SETTINGS}}}"#
    ))]);
    // No tabs to reach: the models per job are the page.
    app.wait_for("synthetic-strong")?;
    app.wait_for("synthetic-quick")?;
    // What each account has used, where its provider says.
    app.wait_for("Signed in to ChatGPT · Plus plan")?;
    app.wait_for("5-hour limit")?;
    app.wait_for("42% used")?;
    app.wait_for("Weekly limit")?;
    app.wait_for("$6.50 left of $10.00")?;
    app.wait_for("On this device")?;
    if app.shows("Connections") || app.shows("Advanced") {
        return Err("the terminal shows the settings tabs".into());
    }
    // Failovers are tried in order: the last moves up, not down.
    if app.shows("label:Move Anthropic down") || app.shows("label:Move OpenAI up") {
        return Err("a failover offers a move past an end".into());
    }
    app.press("label:Move Anthropic up")?;
    let mut moved = false;
    for _ in 0..100 {
        moved = nsh_app_aidaemon::settings_sent().iter().any(|c| {
            serde_json::from_str::<serde_json::Value>(c).is_ok_and(|c| {
                c["change"]
                    == serde_json::json!({"kind": "move_failover", "index": 1, "to": 0, "expect": "Anthropic"})
            })
        });
        if moved {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    if !moved {
        return Err("no move_failover was sent".into());
    }
    for (id, json) in nsh_app_aidaemon::settings_sent_ids() {
        if json.contains("move_failover") {
            feed([frame(&format!(
                r#"{{"type":"ack","request_id":"{id}","result":{{"type":"settings_changed","outcome":{{"kind":"done","message":"Moved"}}}}}}"#
            ))]);
        }
    }
    // Answered, the cards take presses again.
    let mut pressed = Err(String::new());
    for _ in 0..100 {
        pressed = app.press("label:Change the models OpenAI uses");
        if pressed.is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    pressed?;
    app.wait_for("label:Hard work model")?;
    Ok(())
}

#[test]
fn the_settings_page_reads_the_settings_and_says_what_changed() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    // Earlier tests' requests stay in the sent list: wait for this one's.
    let before = sent("get_settings");
    app.press("Settings: change how the daemon runs")?;
    app.wait_for("Reading settings…")?;
    feed([frame(
        r#"{"type":"event","at_ms":1,"event":{"type":"settings_updated","ok":true,"message":"Updated memory.retention_days = 30"}}"#,
    )]);
    app.wait_for("Updated memory.retention_days = 30")?;
    // The settings arrive: a value edits as TOML in a code editor, and
    // another key is typed with the known keys suggested.
    let id = wait_sent_new("get_settings", before)?;
    let result = SETTINGS;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{result}}}"#
    ))]);
    // What Enter does while the agent works is this device's choice.
    app.wait_for("On this device")?;
    app.wait_for("While the agent works, Enter >> nth=0")?;
    // Providers name a model per job; a key in the file offers the keychain,
    // and a failover never offers the provider already in use.
    app.press("Models")?;
    app.wait_for("synthetic-strong")?;
    app.wait_for("synthetic-quick")?;
    app.wait_for("Key in config.toml")?;
    app.wait_for("Move to keychain")?;
    // Each provider's models per job are the owner's to pick, failovers
    // included; an emptied job goes back to the catalog's model.
    app.press("label:Change the models OpenAI uses")?;
    app.wait_for("label:Hard work model")?;
    // The provider's catalog models are a tap away; free text still wins.
    app.press("label:Use gpt-astra-synthetic for hard work")?;
    app.clear("label:Hard work model")?;
    app.type_text("label:Hard work model", "gpt-strong-synthetic")?;
    // The current model is shown; emptied, the catalog's returns.
    app.clear("label:Quick jobs model")?;
    app.press("Save models")?;
    app.wait_for_gone("label:Hard work model", std::time::Duration::from_secs(5))?;
    let mut changes = Vec::new();
    for _ in 0..100 {
        changes = nsh_app_aidaemon::settings_sent();
        if changes.len() >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    for (id, _) in nsh_app_aidaemon::settings_sent_ids() {
        feed([frame(&format!(
            r#"{{"type":"ack","request_id":"{id}","result":{{"type":"settings_changed","outcome":{{"kind":"done","message":"Saved"}}}}}}"#
        ))]);
    }
    let changes: Vec<serde_json::Value> = changes
        .iter()
        .filter_map(|c| serde_json::from_str::<serde_json::Value>(c).ok())
        .map(|c| c["change"].clone())
        .collect();
    let hard = serde_json::json!({"kind": "set", "key": "provider.fallbacks[0].models.smart", "value": "\"gpt-strong-synthetic\""});
    let reset = serde_json::json!({"kind": "set", "key": "provider.fallbacks[0].models.fast"});
    if !changes.contains(&hard) || !changes.contains(&reset) {
        return Err(format!("the model changes sent: {changes:?}"));
    }
    feed([frame(
        r#"{"type":"event","at_ms":2,"event":{"type":"settings_updated","ok":true,"message":"Updated provider.fallbacks[0].models.smart"}}"#,
    )]);
    app.wait_for("Updated provider.fallbacks[0].models.smart")?;
    app.press("Add a failover")?;
    app.wait_for("Tried when the providers before it fail.")?;
    app.wait_for("Every provider here is already in use.")?;
    if app.shows("Billed per use") {
        return Err("the failover list offers OpenAI again".into());
    }
    app.press("Advanced")?;
    app.wait_for("Another setting")?;
    app.type_text("label:Setting key", "provider.mo")?;
    app.wait_for("provider.models.default")?;
    app.press("provider.models.default")?;
    // A value opens in the code editor.
    app.press("Model providers · 2 changed")?;
    app.press("Edit >> nth=0")?;
    app.wait_for("label:Value of provider.models.default")?;
    Ok(())
}

#[test]
fn a_coding_agent_s_mode_is_picked_and_full_access_waits_for_approval() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    let before = sent("get_settings");
    feed([Update::Connected]);
    app.press("Settings: change how the daemon runs")?;
    let id = wait_sent_new("get_settings", before)?;
    let result = SETTINGS.replace(
        r#""restart_pending": false}"#,
        r#""restart_pending": false, "coding_agents": [{"name": "claude", "program": "Claude Code", "mode": "checked", "modes": ["checked", "every_command", "full_access"]}, {"name": "synthetic-agent", "program": "Acme Coder", "modes": ["full_access"], "note": "Acme Coder's requests cannot be checked."}]}"#,
    );
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{result}}}"#
    ))]);
    app.wait_for("Full access: Runs on this machine with nothing checked. Fastest; your rules don't see what it does.")?;
    app.wait_for("Claude Code")?;
    app.wait_for("synthetic-agent")?;
    // An agent with no mode yet says it may not run, and why it offers
    // fewer modes.
    app.wait_for("Not allowed to run delegated work yet.")?;
    app.wait_for("Acme Coder's requests cannot be checked.")?;
    app.wait_for("Checked by AIDaemon >> nth=0")?;
    // Full access goes to the daemon, which asks first.
    let before = nsh_app_aidaemon::settings_sent_ids().len();
    app.press("label:Claude Code permission mode")?;
    app.press("Full access")?;
    let mut change = None;
    for _ in 0..100 {
        change = nsh_app_aidaemon::settings_sent_ids()
            .into_iter()
            .skip(before)
            .find(|(_, c)| c.contains("set_agent_mode"));
        if change.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (id, json) = change.ok_or("no set_agent_mode was sent")?;
    let sent: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    if sent["change"]
        != serde_json::json!({"kind": "set_agent_mode", "agent": "claude", "mode": "full_access"})
    {
        return Err(format!("the change sent: {}", sent["change"]));
    }
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"settings_changed","outcome":{{"kind":"awaiting_approval","approval_id":"ap-synthetic-1"}}}}}}"#
    ))]);
    app.wait_for("Waiting for your approval")?;
    Ok(())
}

#[test]
fn each_coding_agent_says_whether_it_is_ready_and_check_again_asks() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    let before = sent("get_settings");
    feed([Update::Connected]);
    app.press("Settings: change how the daemon runs")?;
    let id = wait_sent_new("get_settings", before)?;
    let result = SETTINGS.replace(
        r#""restart_pending": false}"#,
        r#""restart_pending": false, "coding_agents": [{"name": "claude", "program": "Claude Code", "mode": "checked", "modes": ["checked", "every_command", "full_access"], "readiness": "ready"}, {"name": "synthetic-agent", "program": "Acme Coder", "mode": "full_access", "modes": ["full_access"], "readiness": "not_usable", "readiness_reason": "Acme Coder is not signed in.", "readiness_fix": "Run acme-coder login, then check again."}, {"name": "other-agent", "program": "Other Coder", "modes": ["full_access"]}]}"#,
    );
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{result}}}"#
    ))]);
    app.wait_for("Ready")?;
    app.wait_for("Unavailable: Acme Coder is not signed in.")?;
    app.wait_for("Run acme-coder login, then check again.")?;
    app.wait_for("Not checked yet")?;
    let before = nsh_app_aidaemon::settings_sent_ids().len();
    app.press("Check again")?;
    let mut change = None;
    for _ in 0..100 {
        change = nsh_app_aidaemon::settings_sent_ids()
            .into_iter()
            .skip(before)
            .find(|(_, c)| c.contains("check_agents"));
        if change.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let (_, json) = change.ok_or("no check_agents was sent")?;
    let sent: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    if sent["change"] != serde_json::json!({"kind": "check_agents"}) {
        return Err(format!("the change sent: {}", sent["change"]));
    }
    Ok(())
}

/// Wait for a setup step naming `what`, sent after the first `before`;
/// its request id.
fn wait_setup_sent(before: usize, what: &str) -> Result<String, String> {
    for _ in 0..150 {
        if let Some((id, _)) = nsh_app_aidaemon::setup_sent()
            .into_iter()
            .skip(before)
            .find(|(_, step)| step.contains(what))
        {
            return Ok(id);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Err(format!("no setup step `{what}` was sent"))
}

#[test]
fn settings_opens_the_phone_step_and_not_now_leaves_it() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    // Earlier tests' requests stay in the sent list: wait for this one's.
    let before = sent("get_settings");
    app.press("Settings: change how the daemon runs")?;
    let id = wait_sent_new("get_settings", before)?;
    let result = SETTINGS;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{result}}}"#
    ))]);
    app.wait_for("On this device")?;
    // A daemon with no coding agents shows no modes to pick for them.
    if app.shows("Full access: Runs on this machine with nothing checked. Fastest; your rules don't see what it does.") {
        return Err("coding agent modes show with no agent to pick for".into());
    }
    app.press("Connections")?;
    let before = nsh_app_aidaemon::setup_sent().len();
    app.press("Reach it from your phone")?;
    let id = wait_setup_sent(before, "detect")?;
    // The daemon opens its chat step again.
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"setup","setup":{{"step":"channel","channels":[{{"platform":"telegram","name":"Telegram","help_url":"https://t.me/BotFather"}}]}}}}}}"#
    ))]);
    app.wait_for("Telegram")?;
    app.press("Telegram")?;
    app.wait_for("Make a Telegram bot")?;
    app.press("Not now")?;
    let id = wait_setup_sent(before, "skip_channel")?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"setup","setup":{{"step":"done"}}}}}}"#
    ))]);
    app.wait_for_gone("Make a Telegram bot", std::time::Duration::from_secs(5))?;
    Ok(())
}

#[test]
fn first_run_connects_a_model_then_greets_with_the_phone_step_left_for_later() -> Result<(), String>
{
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    let before = nsh_app_aidaemon::setup_sent().len();
    feed([
        Update::Connected,
        frame(
            r#"{"type":"welcome","version":{"major":1,"minor":5},"owner":{},"latest_seq":0,"setup":{"step":"model","found":[{"preset":"openai_chatgpt","name":"OpenAI (ChatGPT subscription) · gpt-synthetic","source":"ChatGPT sign-in on this computer","billing":"plan","works":true}],"providers":[{"preset":"openrouter","name":"OpenRouter","billing":"per_token","needs_key":true,"model":"synthetic/model-1"},{"preset":"ollama","name":"Ollama (local)","billing":"local","needs_key":false}]}}"#,
        ),
    ]);
    app.wait_for("Connect a model")?;
    // The model found here is one click; the rest wait.
    app.wait_for("Found: OpenAI (ChatGPT subscription) · gpt-synthetic")?;
    app.wait_for("Continue")?;
    app.wait_for(
        "ChatGPT sign-in on this computer · uses your plan · has limits. It answered a test just now.",
    )?;
    if app.shows("pay per token") {
        return Err("every provider shows before Use something else".into());
    }
    app.press("Use something else")?;
    app.wait_for("pay per token")?;
    // A model server on this machine takes no key and no sign-in.
    app.wait_for("runs on your own machine · free")?;
    app.press("Use it")?;
    app.wait_for("Find its models")?;
    if app.shows("label:API key") || app.shows("Paste the key") {
        return Err("a local server is asked for a key".into());
    }
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
    // The daemon starts with the model; the chat step waits for Settings,
    // so the first conversation opens next.
    feed([frame(
        r#"{"type":"event","at_ms":1,"event":{"type":"setup_updated","setup":{"step":"starting","model":"OpenRouter · synthetic/model-1"}}}"#,
    )]);
    app.wait_for(
        "OpenRouter · synthetic/model-1 answered its test. AIDaemon is starting with it…",
    )?;
    feed([frame(
        r#"{"type":"event","at_ms":2,"event":{"type":"setup_updated","setup":{"step":"channel","model":"OpenRouter · synthetic/model-1","channels":[{"platform":"telegram","name":"Telegram","help_url":"https://t.me/BotFather"}]}}}"#,
    )]);
    app.wait_for("Opening your first conversation")?;
    if app.shows("Reach it from your phone") {
        return Err("first run asks for a bot token".into());
    }
    wait_setup_sent(before, "skip_channel")?;
    // Done: the greeting's conversation opens, with prompts to press.
    feed([
        frame(
            r#"{"type":"event","at_ms":4,"event":{"type":"setup_updated","setup":{"step":"done"}}}"#,
        ),
        frame(
            r#"{"type":"event","seq":1,"at_ms":5,"session_id":"app:welcome","event":{"type":"session_created","session":{"id":"app:welcome","channel":"app","title":"Welcome","writable":true}}}"#,
        ),
        frame(
            r#"{"type":"event","seq":2,"at_ms":6,"session_id":"app:welcome","event":{"type":"assistant_message","session_id":"app:welcome","text":"Hi, I'm AIDaemon.","suggestions":["What can you do on this computer?"]}}"#,
        ),
    ]);
    app.wait_for("Hi, I'm AIDaemon.")?;
    app.wait_for("What can you do on this computer?")?;
    Ok(())
}

#[test]
fn a_snapshot_mid_turn_shows_what_the_turn_is_doing() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"snapshot","seq":7,"state":{"sessions":[{"id":"app:synthetic-3","channel":"app","title":"Count the files","writable":true}],"running_tasks":[{"id":"b:app:synthetic-3#1","session_id":"app:synthetic-3","number":1,"description":"Count the files","state":"running","started_at_ms":1}],"pending_approvals":[],"live_turns":[{"session_id":"app:synthetic-3","turn_id":"t9","started_at_ms":1,"status":{"kind":"tool_started","name":"terminal","summary":"find . -name '*.rs'"},"tools":["read_file","terminal"]}]}}"#,
        ),
    ]);
    app.wait_for("Count the files")?;
    app.press("Count the files >> nth=0")?;
    // The working line says what it is doing; the tools so far are a
    // line over it. The turn is also a running task: Stop alone stops it,
    // no Cancel beside it.
    app.wait_for("text:terminal: find . -name '*.rs'...")?;
    // The count of tools ends the line and opens them in place.
    app.press("label:Tools so far")?;
    // A tool the daemon gave no kind for reads as its name in words.
    app.wait_for("Read file")?;
    // Stop is the composer's round button while it is empty.
    app.wait_for("label:Stop")?;
    assert!(app.press("Cancel").is_err());
    Ok(())
}

#[test]
fn a_file_write_gives_a_conversation_its_changes_tab() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-2","channel":"app","title":"Add the contact form","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Add the contact form")?;
    app.press("Add the contact form >> nth=0")?;
    // A conversation that has not touched code has no details pane.
    assert!(app.press("Changes and steps").is_err());
    // The turn's first file write gives it Changes and opens them.
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-2","event":{"type":"turn_started","turn_id":"t2","session_id":"app:synthetic-2","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","at_ms":2,"session_id":"app:synthetic-2","event":{"type":"turn_status","turn_id":"t2","status":{"kind":"tool_completed","name":"edit_file","summary":"src/contact.rs"}}}"#,
        ),
    ]);
    app.wait_for("Reading changes…")?;
    // The changes arrive; a file opens on its diff, which fills the pane.
    let patch = r#"diff --git a/src/contact.rs b/src/contact.rs\n--- a/src/contact.rs\n+++ b/src/contact.rs\n@@ -1,2 +1,3 @@\n-fn send() {}\n+fn send() {\n+}\n fn main() {}\n"#;
    let id = wait_sent("get_changes")?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"changes","changes":{{"repos":[{{"root":"/home/synthetic/acme-site","branch":"main","files":[{{"path":"src/contact.rs","status":"modified","added":2,"removed":1,"by_session":true}}],"patch":"{patch}"}}]}}}}}}"#
    ))]);
    app.wait_for("acme-site")?;
    app.press("src/contact.rs, modified, written in this conversation")?;
    app.wait_for("All files")?;
    app.wait_for("fn main() {}")?;
    // Undo asks first; Cancel leaves the file as it is.
    app.answer_dialogs_myself();
    app.press("Undo changes…")?;
    let asked = app.pending_dialog().and_then(|d| d.title.clone());
    assert_eq!(
        asked.as_deref(),
        Some("Undo the changes to src/contact.rs?")
    );
    app.answer_dialog(nsh_testing::DialogAnswer::Press("Cancel".into()))?;
    assert!(sent("revert_file").is_none());
    app.press("Side by side")?;
    app.press("Next change")?;
    app.press("All files")?;
    app.wait_for("acme-site")?;
    Ok(())
}

#[test]
fn a_project_conversation_browses_its_files_read_only() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"rp","result":{"type":"projects","projects":[{"id":"p-acme","name":"Acme site","path":"/home/synthetic/acme-site"}]}}"#,
        ),
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-files","channel":"app","title":"Tidy the forms","writable":true,"project_id":"p-acme"}]}}"#,
        ),
    ]);
    app.wait_for("Tidy the forms")?;
    app.press("Tidy the forms >> nth=0")?;
    // A conversation in a project with a folder works on code: its details
    // have Files beside Changes.
    app.wait_for("label:Files")?;
    let before = sent("list_project_files");
    app.press("label:Files")?;
    let id = wait_sent_new("list_project_files", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"project_dir","dir":{{"project_id":"p-acme","branch":"main","entries":[{{"name":"src","kind":"folder","changed":true}},{{"name":".env","kind":"file","size_bytes":64,"secret":true}},{{"name":"README.md","kind":"file","size_bytes":1229}}]}}}}}}"#
    ))]);
    app.wait_for("README.md")?;
    app.wait_for("1.2 KB")?;
    app.wait_for("label:.env, holds secrets")?;
    // A folder opens in place; the path above goes back.
    app.press("label:src, folder, has changes")?;
    let id = wait_sent_new("list_project_files", Some(id))?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"project_dir","dir":{{"project_id":"p-acme","path":"src","entries":[{{"name":"contact.rs","kind":"file","size_bytes":26,"status":"modified"}}]}}}}}}"#
    ))]);
    app.wait_for("contact.rs")?;
    // A file opens filling the pane, its text read-only.
    let before = sent("read_project_file");
    app.press("label:contact.rs, modified")?;
    let id = wait_sent_new("read_project_file", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"project_file","file":{{"project_id":"p-acme","path":"src/contact.rs","size_bytes":26,"version":"sha256:00","text":"fn send() {{}}\nfn main() {{}}\n","status":"modified"}}}}}}"#
    ))]);
    app.wait_for("src/contact.rs")?;
    app.wait_for("Copy path")?;
    // The answer is in (the path and Copy show while loading): the same read
    // still unanswered is not sent again.
    app.wait_for("fn main() {}")?;
    // Full screen: the file fills the window, read again; Exit puts it
    // back in the details pane, still open.
    let before = sent("read_project_file");
    app.press("label:Full screen")?;
    let id = wait_sent_new("read_project_file", before)?;
    app.wait_for("Acme site files")?;
    app.wait_for("label:Exit full screen")?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"project_file","file":{{"project_id":"p-acme","path":"src/contact.rs","size_bytes":26,"version":"sha256:00","text":"fn send() {{}}\nfn main() {{}}\n","status":"modified"}}}}}}"#
    ))]);
    app.wait_for("fn main() {}")?;
    app.press("label:Exit full screen")?;
    wait_sent_new("read_project_file", Some(id))?;
    app.wait_for("label:Full screen")?;
    app.wait_for_gone("Acme site files", std::time::Duration::from_secs(5))?;
    app.press("label:src")?;
    app.wait_for("contact.rs")?;
    Ok(())
}

#[test]
fn a_projects_files_open_from_its_page_without_a_conversation() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"rp","result":{"type":"projects","projects":[{"id":"p-docs","name":"Docs site","path":"/home/synthetic/docs-site"}]}}"#,
        ),
        frame(r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[]}}"#),
    ]);
    app.wait_for("Docs site")?;
    app.press("Project Docs site, 0 conversations")?;
    app.wait_for("label:This project's files")?;
    let before = sent("list_project_files");
    app.press("label:This project's files")?;
    let id = wait_sent_new("list_project_files", before)?;
    app.wait_for("Docs site files")?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"project_dir","dir":{{"project_id":"p-docs","branch":"main","entries":[{{"name":"index.md","kind":"file","size_bytes":12}}]}}}}}}"#
    ))]);
    app.wait_for("index.md")?;
    Ok(())
}

#[test]
fn chats_filter_by_source_and_a_group_asks_before_posting() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-5","channel":"app","title":"Weekend plan","writable":true},{"id":"synthetic_coding_bot:7","channel":"telegram:synthetic_coding_bot","title":"Fix the footer","writable":true},{"id":"synthetic_coding_bot:-9","channel":"telegram:synthetic_coding_bot","title":"Team release notes","writable":true,"group":true}]}}"#,
        ),
    ]);
    app.wait_for("Weekend plan")?;
    // Search is an icon until pressed: then the line is the field, and
    // Cancel clears it and brings the name back.
    app.wait_for("AIDaemon")?;
    app.press("Search conversations")?;
    app.type_text("label:Search conversations", "footer")?;
    app.wait_for_gone("Weekend plan", std::time::Duration::from_secs(5))?;
    app.wait_for("Fix the footer")?;
    app.press("Cancel")?;
    app.wait_for("Weekend plan")?;
    app.wait_for("AIDaemon")?;
    // Each says where it came from; one filter narrows the list.
    app.wait_for("Telegram group")?;
    app.press("Show conversations from: All")?;
    app.wait_for("Groups")?;
    app.press("Groups")?;
    app.wait_for_gone("Weekend plan", std::time::Duration::from_secs(5))?;
    app.wait_for("Team release notes")?;
    // A group: posting from here is the owner's explicit choice.
    app.press("Team release notes >> nth=0")?;
    app.wait_for("Post in this group")?;
    app.press("Post in this group")?;
    app.wait_for("Also posted in synthetic_coding_bot · group")?;
    app.wait_for("label:Telegram")?;
    // A direct chat posts there too, and says so.
    app.press("Show conversations from: Groups")?;
    app.press("All")?;
    app.press("Fix the footer >> nth=0")?;
    app.wait_for("Also posted in synthetic_coding_bot")?;
    Ok(())
}

#[test]
fn projects_group_their_conversations_and_start_new_ones_in_their_folder() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    // Code before any project exists says how to make the first.
    app.press("Code")?;
    app.wait_for("Which project is this for?")?;
    app.wait_for("New project")?;
    app.press("Chat")?;
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"projects","projects":[{"id":"project-acme","name":"Acme site","path":"/home/synthetic/acme-site"},{"id":"project-notes","name":"Design notes"}]}}"#,
        ),
        frame(
            r#"{"type":"ack","request_id":"r1","result":{"type":"sessions","sessions":[{"id":"app:synthetic-3","channel":"app","title":"Contact form copy","writable":true,"project_id":"project-acme"},{"id":"app:synthetic-5","channel":"app","title":"Old landing page","writable":true,"archived":true,"project_id":"project-acme"},{"id":"app:synthetic-4","channel":"app","title":"Grocery list","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Acme site")?;
    app.wait_for("Design notes")?;
    // Every conversation is listed, one in a project tagged with it.
    app.wait_for("Grocery list")?;
    app.wait_for("Contact form copy")?;
    // A press goes inside a project: its folder and its conversations.
    app.press("Project Acme site, 1 conversations")?;
    app.wait_for("Back to all conversations")?;
    app.wait_for_gone("Grocery list", std::time::Duration::from_secs(5))?;
    app.wait_for("Contact form copy")?;
    // Its archived conversations are a press away at the top of its page,
    // and Back returns to the project.
    app.press("Archived conversations in this project")?;
    app.wait_for("Archived in Acme site")?;
    app.wait_for("Old landing page")?;
    app.press("Back to the project")?;
    app.wait_for_gone("Old landing page", std::time::Duration::from_secs(5))?;
    app.wait_for("Contact form copy")?;
    // New conversation, inside a project, starts one there.
    app.press("New conversation >> nth=0")?;
    app.wait_for("Which project is this for?")?;
    app.wait_for("/home/synthetic/acme-site")?;
    // The header's switch says what a new conversation is: Chat leaves
    // the project and brings the question; Code shows the project on a
    // card, the one this device last worked in.
    app.press("Chat")?;
    app.wait_for("What can I help you with?")?;
    app.press("Code")?;
    app.wait_for("Which project is this for?")?;
    app.wait_for("/home/synthetic/acme-site")?;
    // The card opens the projects with New project last; a press picks
    // one, and the card names it.
    app.press("Project: Acme site, change")?;
    app.wait_for("New project")?;
    app.press("Design notes >> nth=0")?;
    app.wait_for("Project: Design notes, change")?;
    // New project opens the form for one.
    app.press("Project: Design notes, change")?;
    app.press("New project >> nth=0")?;
    app.wait_for("Project name")?;
    // A conversation's menu moves it with the same picker; No project
    // takes it out of one.
    app.press_key("escape")?;
    app.press("Back to all conversations >> nth=0")?;
    // The sheet says which conversation it is about: another may be the
    // one marked as open.
    open_menu(&mut app, "Grocery list >> nth=0", "Move to project…")?;
    assert!(
        app.screen()
            .layout_nodes()
            .iter()
            .any(|n| n.test_id.as_deref() == Some("menu-title") && !n.hidden),
        "the menu names its conversation"
    );
    let before = sent("set_session_project");
    app.press("Move to project…")?;
    app.wait_for("Move to project")?;
    app.press("Acme site")?;
    wait_sent_new("set_session_project", before)?;
    // A new conversation from the keyboard, in a project's conversation,
    // stays in that project.
    app.press("Contact form copy >> nth=0")?;
    app.wait_for("Acme site · /home/synthetic/acme-site")?;
    app.press_key("mod+n")?;
    app.wait_for("Which project is this for?")?;
    // The paperclip offers pictures and files (and the camera, where a
    // phone's browser has one).
    app.press("Attach")?;
    app.wait_for("Pictures")?;
    app.wait_for("Files")?;
    Ok(())
}

/// Whether `first` comes before `second` among the texts on screen.
fn above(screen: &nsh_testing::Screen, first: &str, second: &str) -> bool {
    let texts = screen.texts();
    let at = |name: &str| texts.iter().position(|t| *t == name);
    matches!((at(first), at(second)), (Some(a), Some(b)) if a < b)
}

/// Long-presses `target` until its menu shows `item`. On a loaded
/// runner a press landing while a just-reordered list redraws can be
/// lost, and the menu opens a frame or more later.
fn open_menu(app: &mut Headless, target: &str, item: &str) -> Result<(), String> {
    for _ in 0..3 {
        app.long_press(target)?;
        if app
            .wait_until(item, std::time::Duration::from_secs(3), |s| s.shows(item))
            .is_ok()
        {
            return Ok(());
        }
    }
    app.wait_for(item)
}

#[test]
fn projects_can_be_pinned_dragged_into_order_and_deleted() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"projects","projects":[{"id":"project-acme","name":"Acme site","path":"/home/synthetic/acme-site"},{"id":"project-beta","name":"Beta app"},{"id":"project-charlie","name":"Charlie docs"},{"id":"project-delta","name":"Delta api"},{"id":"project-echo","name":"Echo blog"}]}}"#,
        ),
        frame(
            r#"{"type":"ack","request_id":"r1","result":{"type":"sessions","sessions":[{"id":"app:synthetic-6","channel":"app","title":"Contact form copy","writable":true,"project_id":"project-acme"}]}}"#,
        ),
    ]);
    // Home has room for three and All projects: Echo blog is left out.
    app.wait_for("Acme site")?;
    app.wait_for("All projects")?;
    assert!(!app.screen().shows("Echo blog"));
    // Pinned from the list of every project, it is first there and on Home.
    app.press("All projects")?;
    app.wait_for("Echo blog")?;
    open_menu(
        &mut app,
        "Project Echo blog, 0 conversations",
        "Pin to the top",
    )?;
    app.press("Pin to the top")?;
    app.wait_for("label:Project Echo blog, pinned, 0 conversations")?;
    assert!(above(app.screen(), "Echo blog", "Acme site"));
    // Its handle drags a project a place up.
    assert!(above(app.screen(), "Charlie docs", "Delta api"));
    app.drag(
        "id:drag:project-delta",
        nsh_engine::Point {
            x: Points(0.0),
            y: Points(-45.0),
        },
    )?;
    app.wait_until(
        "Delta api above Charlie docs",
        std::time::Duration::from_secs(5),
        |s| above(s, "Delta api", "Charlie docs"),
    )?;
    // The menu moves one too, for the keyboard.
    open_menu(&mut app, "Project Delta api, 0 conversations", "Move down")?;
    app.press("Move down")?;
    app.wait_until(
        "Charlie docs above Delta api",
        std::time::Duration::from_secs(5),
        |s| above(s, "Charlie docs", "Delta api"),
    )?;
    app.press("Back to all conversations >> nth=0")?;
    app.wait_for("label:Project Echo blog, pinned, 0 conversations")?;
    // Home's projects drag the same way, in the main list.
    assert!(above(app.screen(), "Acme site", "Beta app"));
    app.drag(
        "id:drag:project-beta",
        nsh_engine::Point {
            x: Points(0.0),
            y: Points(-31.0),
        },
    )?;
    app.wait_until(
        "Beta app above Acme site",
        std::time::Duration::from_secs(5),
        |s| above(s, "Beta app", "Acme site"),
    )?;
    assert!(above(app.screen(), "Echo blog", "Beta app"));
    // Delete asks first and says what happens to its conversations and
    // folder.
    open_menu(
        &mut app,
        "Project Acme site, 1 conversations",
        "Delete project…",
    )?;
    app.answer_dialogs_myself();
    let before = sent("delete_project");
    app.press("Delete project…")?;
    let asked = app
        .pending_dialog()
        .map(|d| (d.title.clone(), d.message.clone()));
    assert_eq!(
        asked,
        Some((
            Some("Delete Acme site?".to_owned()),
            Some(
                "Its conversation moves to No project. The folder /home/synthetic/acme-site stays as it is."
                    .to_owned()
            )
        ))
    );
    app.answer_dialog(nsh_testing::DialogAnswer::Press("Delete".into()))?;
    let id = wait_sent_new("delete_project", before)?;
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"event":{"type":"project_deleted","project_id":"project-acme"}}"#,
        ),
        frame(&format!(
            r#"{{"type":"ack","request_id":"{id}","result":{{"type":"projects","projects":[{{"id":"project-beta","name":"Beta app"}},{{"id":"project-charlie","name":"Charlie docs"}},{{"id":"project-delta","name":"Delta api"}},{{"id":"project-echo","name":"Echo blog"}}]}}}}"#
        )),
    ]);
    app.wait_for("Deleted Acme site")?;
    app.wait_for_gone("Acme site", std::time::Duration::from_secs(5))?;
    app.wait_for("Contact form copy")?;
    Ok(())
}

#[test]
fn a_file_the_agent_sends_shows_with_its_reply() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-files","channel":"app","title":"Quarterly numbers","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Quarterly numbers")?;
    app.press("Quarterly numbers")?;
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-files","event":{"type":"assistant_message","session_id":"app:synthetic-files","text":"Here is the report.","attachments":[{"id":"0123456789abcdef01234567","name":"synthetic-report.pdf","mime":"application/pdf","size_bytes":4,"url":"/client/v1/files/0123456789abcdef01234567/sig/synthetic-report.pdf"}]}}"#,
    )]);
    app.wait_for("Here is the report.")?;
    app.wait_for("synthetic-report.pdf")?;
    Ok(())
}

#[test]
fn a_sent_picture_shows_at_its_size_before_it_loads() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-pic","channel":"app","title":"Look at this","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Look at this")?;
    app.press("Look at this")?;
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-pic","event":{"type":"user_message","session_id":"app:synthetic-pic","text":"What is in it?","origin":"app","attachments":[{"id":"0123456789abcdef01234567","name":"shot.png","mime":"image/png","size_bytes":4,"url":"/client/v1/files/0123456789abcdef01234567/sig/shot.png","width":800,"height":400}]}}"#,
    )]);
    app.wait_for("What is in it?")?;
    let picture = app
        .screen()
        .layout_nodes()
        .into_iter()
        .find(|n| n.test_id.as_deref() == Some("picture"))
        .ok_or("no picture on screen")?;
    let (w, h) = (picture.bounds.size.width.0, picture.bounds.size.height.0);
    // Laid out before it loads: its 2:1 shape, at most 320 wide.
    assert!(w > 100.0 && w <= 320.0, "{w} x {h}");
    assert!((w / h - 2.0).abs() < 0.01, "{w} x {h}");
    // A press opens it over the page (not a browser tab); Close closes it.
    app.press("Picture: shot.png")?;
    app.wait_for("Open in browser")?;
    app.wait_for("Close")?;
    app.press("Close")?;
    app.wait_for_gone("Open in browser", std::time::Duration::from_secs(5))?;
    Ok(())
}

/// A run that was due while the daemon was off reads as missed, in plain
/// words, in the last-run line and in History; not as failed.
#[test]
fn a_missed_run_reads_as_missed_in_the_last_run_line_and_history() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    app.press("Automations: what runs on its own")?;
    app.wait_for("Reading your automations…")?;
    feed([frame(
        r#"{"type":"ack","request_id":"r9","result":{"type":"automations","automations":[{"id":"goal-1","kind":"scheduled","name":"Inbox digest","what":"Summarize the synthetic inbox","how_often":"Every day at 9:00 AM ET","status":"active","edit":{"when":"every day at 9am","instructions":"Summarize the synthetic inbox"},"recent_runs":[{"started_ms":1767088800000,"finished_ms":1767092400000,"outcome":"missed","summary":"The daemon wasn't running when this was due."},{"started_ms":1767002400000,"finished_ms":1767002460000,"outcome":"done","summary":"3 new messages, 1 needs a reply."}]}]}}"#,
    )]);
    app.wait_for(r"text:/^Last run due /")?;
    app.wait_for("The daemon wasn't running when this was due.")?;
    // Its recent runs, under Details, say the same.
    app.press("Details")?;
    app.wait_for("Recent runs")?;
    app.wait_for("Missed")?;
    app.wait_for("3 new messages, 1 needs a reply.")?;
    Ok(())
}

#[test]
fn automations_list_and_ask_before_cancelling() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    app.press("Automations: what runs on its own")?;
    app.wait_for("Reading your automations…")?;
    feed([frame(
        r#"{"type":"ack","request_id":"r9","result":{"type":"automations","automations":[{"id":"goal-1","kind":"scheduled","name":"Inbox digest","what":"Summarize the synthetic inbox","how_often":"Every weekday at 9:00 AM ET","status":"active","details":["Can do: Read only – makes no changes"],"edit":{"when":"every weekday at 9am","instructions":"Summarize the synthetic inbox"}},{"id":"mandate-1","kind":"watching","name":"Status watch","what":"Watch the synthetic status page","how_often":"Every hour","status":"paused","recent_runs":[{"started_ms":1767002400000,"finished_ms":1767002460000,"outcome":"done","summary":"No change on the synthetic status page."}]},{"id":"goal-2","kind":"scheduled","name":"Repo check","what":"Check the synthetic repo","how_often":"Every day at 8:00 AM ET","status":"held_after_failures","edit":{"when":"every day at 8am","instructions":"Check the synthetic repo"},"recent_runs":[{"started_ms":1767002400000,"outcome":"failed","summary":"The synthetic token expired."},{"started_ms":1766916000000,"outcome":"done","summary":"2 open PRs."}]}]}}"#,
    )]);
    // What each one did, not only what it will do.
    app.wait_for("It hasn't run yet.")?;
    // Runs that reported something since the last look are marked.
    app.wait_for("New since you last looked")?;
    app.wait_for("The synthetic token expired.")?;
    // Paused after its runs kept failing: the way back is a good run.
    app.wait_for("Paused after failures")?;
    app.wait_for("Try again")?;
    app.press("Details >> nth=2")?;
    app.wait_for("2 open PRs.")?;
    app.wait_for("Recent runs")?;
    app.press("Hide details")?;
    app.wait_for_gone("2 open PRs.", std::time::Duration::from_secs(5))?;
    // A watching automation changes by asking; scheduled work in place.
    app.wait_for("Change in chat")?;
    app.press("Edit >> nth=0")?;
    app.wait_for("Changes apply from its next run.")?;
    app.wait_for("label:When it runs")?;
    app.press("Discard")?;
    app.wait_for_gone(
        "Changes apply from its next run.",
        std::time::Duration::from_secs(5),
    )?;
    app.wait_for("Inbox digest")?;
    app.wait_for("Every weekday at 9:00 AM ET")?;
    // A paused one resumes; only scheduled work runs now.
    app.wait_for("Paused")?;
    app.wait_for("Resume")?;
    app.wait_for("Run now")?;
    // Cancelling sits under Details, and asks first.
    app.press("Details >> nth=0")?;
    app.press("Cancel automation")?;
    app.wait_for("Cancel it for good?")?;
    app.press("Keep it")?;
    app.wait_for_gone("Cancel it for good?", std::time::Duration::from_secs(5))?;
    // A change's answer brings the list back, and says what happened.
    feed([frame(
        r#"{"type":"ack","request_id":"r10","result":{"type":"automations","automations":[],"note":"Cancelled \"Inbox digest\" (scheduled). It won't run again.","changed":true}}"#,
    )]);
    app.wait_for("Nothing runs on its own yet")?;
    Ok(())
}

/// Editing scheduled work in place: Save waits for a change, sends it, and
/// the answer brings the list back as the daemon now has it.
#[test]
fn an_automation_is_edited_in_place_and_saved() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    app.press("Automations: what runs on its own")?;
    app.wait_for("Reading your automations…")?;
    feed([frame(
        r#"{"type":"ack","request_id":"r9","result":{"type":"automations","automations":[{"id":"goal-1","kind":"scheduled","name":"Inbox digest","what":"Summarize the synthetic inbox","how_often":"Every weekday at 9:00 AM ET","status":"active","edit":{"when":"every weekday at 9am","instructions":"Summarize the synthetic inbox","notify":"every_run"}}]}}"#,
    )]);
    app.wait_for("Inbox digest")?;
    app.press("Edit")?;
    app.wait_for("Changes apply from its next run.")?;
    // What its runs tell you is part of it.
    app.wait_for("Every run")?;
    app.type_text("label:When it runs", " and saturdays")?;
    app.press("Save")?;
    // The form closes; the answer brings the list as it is now.
    app.wait_for_gone(
        "Changes apply from its next run.",
        std::time::Duration::from_secs(5),
    )?;
    feed([frame(
        r#"{"type":"ack","request_id":"r10","result":{"type":"automations","automations":[{"id":"goal-1","kind":"scheduled","name":"Inbox digest","what":"Summarize the synthetic inbox","how_often":"Every weekday and Saturday at 9:00 AM ET","status":"active","edit":{"when":"every weekday at 9am and saturdays","instructions":"Summarize the synthetic inbox"}}],"note":"Rescheduled \"Inbox digest\".","changed":true}}"#,
    )]);
    app.wait_for("Every weekday and Saturday at 9:00 AM ET")?;
    Ok(())
}

#[test]
fn resting_the_pointer_on_a_conversation_previews_it() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start_desktop();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-6","channel":"app","title":"Weekend plan","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Weekend plan")?;
    app.hover("Weekend plan >> nth=0")?;
    app.wait_for("Open it to read it.")?;
    Ok(())
}

/// A recognizer that hears one sentence and ends when stopped.
struct Hears;

struct Session(nsh_core::speech::SpeechSink);

impl nsh_core::speech::SpeechControl for Session {
    fn stop(&mut self) {
        self.0.result("and add a test", true);
        self.0.end(None);
    }
    fn abort(&mut self) {}
}

impl nsh_core::speech::SpeechBackend for Hears {
    fn start(
        &self,
        _: nsh_core::speech::SpeechRequest,
        events: nsh_core::speech::SpeechSink,
    ) -> Box<dyn nsh_core::speech::SpeechControl> {
        events.start();
        events.result("and add", false);
        Box::new(Session(events))
    }
}

#[test]
fn speaking_writes_into_the_composer() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    no_local_daemon();
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.os = nsh_core::platform::Os::MacOs;
    config.environment.idiom = nsh_core::platform::Idiom::Desktop;
    config.speech = Some(std::sync::Arc::new(Hears));
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What can I help you with?")?;
    app.type_text("label:Message", "Fix the footer")?;
    app.press("Speak instead of typing")?;
    app.wait_for("Stop listening")?;
    app.press("Stop listening")?;
    app.wait_for("Speak instead of typing")?;
    app.wait_for("Fix the footer and add a test")?;
    Ok(())
}

#[test]
fn a_reply_finishing_elsewhere_says_so_in_the_app() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-7","channel":"app","title":"Draft the invoice","writable":true},{"id":"app:synthetic-8","channel":"app","title":"Plan the trip","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Plan the trip")?;
    app.press("Plan the trip >> nth=0")?;
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-7","event":{"type":"turn_started","turn_id":"t7","session_id":"app:synthetic-7","origin":"app"}}"#,
    )]);
    // Working elsewhere: its row turns a ring where its time was, in its
    // place in the list (no section of its own).
    app.wait_for("label:Working")?;
    feed([frame(
        r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-7","event":{"type":"turn_completed","turn_id":"t7","text":"The invoice is drafted.","state":"completed"}}"#,
    )]);
    // "Done", what finished under it, and a press opens it.
    app.wait_for("Done: Draft the invoice")?;
    // It can be closed without opening anything.
    app.wait_for("label:Dismiss")?;
    app.press("label:Open Draft the invoice")?;
    app.wait_for("The invoice is drafted.")?;
    Ok(())
}

#[test]
fn the_composer_picks_a_model_and_an_effort_on_a_phone() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start_at(390.0, 844.0);
    app.wait_for("What can I help you with?")?;
    let before = sent("list_models");
    feed([Update::Connected]);
    let id = wait_sent_new("list_models", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"models","options":{{"providers":[
            {{"id":"main","name":"Anthropic","billing":"per_token","models":[{{"id":"synthetic-claude","efforts":["low","medium","high"]}}]}},
            {{"id":"failover-0","name":"Google Gemini","billing":"per_token","models":[{{"id":"synthetic-gemini"}}]}}
        ]}}}}}}"#
    ))]);
    app.wait_for("Model and effort: Auto")?;
    // The picker's row stays on the screen, beside nothing that wraps it.
    let row = app
        .screen()
        .layout_nodes()
        .into_iter()
        .find(|n| n.test_id.as_deref() == Some("model-picker") && !n.hidden)
        .ok_or("no model picker on screen")?;
    let right = row.bounds.origin.x.0 + row.bounds.size.width.0;
    assert!(
        right <= 390.0,
        "the picker ends at {right} pt on a 390 pt screen"
    );
    // One picker holds the models (their providers under them) and the
    // efforts of the model in effect.
    app.press("Model and effort: Auto")?;
    app.wait_for("Google Gemini")?;
    app.wait_for("Low effort")?;
    app.press("synthetic-gemini")?;
    app.wait_for("Model and effort: synthetic-gemini")?;
    // Gemini takes no effort: the picker drops the efforts.
    app.press("Model and effort: synthetic-gemini")?;
    app.wait_for("Google Gemini")?;
    app.wait_for_gone("Low effort", std::time::Duration::from_secs(2))?;
    // An existing conversation's pick goes to the daemon at once (in a
    // window, where the list of conversations shows).
    drop(app);
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    let id = wait_sent_new("list_models", Some(id))?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"models","options":{{"providers":[
            {{"id":"main","name":"Anthropic","billing":"per_token","models":[{{"id":"synthetic-claude","efforts":["low","medium","high"]}}]}},
            {{"id":"failover-0","name":"Google Gemini","billing":"per_token","models":[{{"id":"synthetic-gemini"}}]}}
        ]}}}}}}"#
    ))]);
    feed([frame(
        r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-m","channel":"app","title":"Pick a model","writable":true}]}}"#,
    )]);
    app.wait_for("Pick a model")?;
    app.press("Pick a model >> nth=0")?;
    app.wait_for("Model and effort: Auto")?;
    app.press("Model and effort: Auto")?;
    app.press("synthetic-claude")?;
    app.wait_for("Model and effort: synthetic-claude")?;
    wait_sent_new("set_session_model", None)?;
    app.press("Model and effort: synthetic-claude")?;
    app.press("High effort")?;
    app.wait_for("Model and effort: synthetic-claude · High")?;
    Ok(())
}

#[test]
fn archived_conversations_wait_under_archived_and_queued_messages_can_be_removed()
-> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-9","channel":"app","title":"Old receipts","writable":true,"archived":true},{"id":"app:synthetic-10","channel":"app","title":"Write the newsletter","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Write the newsletter")?;
    // An archived conversation is not in the list: the filter by Recent
    // offers the archived, which open as their own page.
    app.wait_for_gone("Old receipts", std::time::Duration::from_secs(2))?;
    app.press("Show conversations from: All")?;
    app.press("Archived")?;
    app.wait_for("Hold a conversation to bring it back.")?;
    app.wait_for("Old receipts")?;
    app.press("Back to all conversations")?;
    app.wait_for_gone("Old receipts", std::time::Duration::from_secs(2))?;
    // A search finds it among the others, and says it was put away.
    app.press("Search conversations")?;
    app.type_text("label:Search conversations", "receipts")?;
    app.wait_for("Old receipts")?;
    app.wait_for("Archived")?;
    app.press("Cancel")?;
    // Archiving says so, and Undo on the note takes it back.
    open_menu(&mut app, "Write the newsletter >> nth=0", "Archive")?;
    let before = sent("archive_session");
    app.press("Archive")?;
    wait_sent_new("archive_session", before)?;
    app.wait_for("Undo")?;
    let before = sent("archive_session");
    app.press("Undo")?;
    wait_sent_new("archive_session", before)?;
    // A message waiting for the running work shows above the composer.
    app.press("Write the newsletter >> nth=0")?;
    feed([frame(
        r#"{"type":"ack","request_id":"q1","result":{"type":"queued","session_id":"app:synthetic-10","messages":[{"id":4,"text":"Then add the event dates","queued_at_ms":1,"after":true}]}}"#,
    )]);
    app.wait_for("Waiting to send")?;
    app.wait_for("Then add the event dates")?;
    // Sent with Queue: it says it runs after.
    app.wait_for("After")?;
    app.press("Remove from the queue")?;
    wait_sent("edit_queued")?;
    feed([frame(
        r#"{"type":"ack","request_id":"q2","result":{"type":"queued","session_id":"app:synthetic-10","messages":[]}}"#,
    )]);
    app.wait_for_gone(
        "Then add the event dates",
        std::time::Duration::from_secs(2),
    )?;
    Ok(())
}

#[test]
fn the_terminal_build_shows_tool_output_as_text() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    no_local_daemon();
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.idiom = nsh_core::platform::Idiom::Terminal;
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-11","channel":"app","title":"Run the build","writable":true}]}}"#,
        ),
    ]);
    // The terminal starts on the conversation; the list is a key away.
    app.press("label:Conversations")?;
    app.wait_for("Run the build")?;
    app.press("Run the build >> nth=0")?;
    // Work on code has the details pane; its Steps tab shows the output.
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-11","event":{"type":"turn_started","turn_id":"t11","session_id":"app:synthetic-11","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","at_ms":2,"session_id":"app:synthetic-11","event":{"type":"turn_status","turn_id":"t11","status":{"kind":"tool_completed","name":"edit_file","summary":"build.rs"}}}"#,
        ),
    ]);
    // Work on code gets the details; Steps is the tab beside Changes.
    // A terminal sends Ctrl+I as Tab: the steps' key there is O.
    app.wait_for(r"text:/O steps/")?;
    app.wait_for("label:Changes and steps")?;
    app.press("label:Changes and steps")?;
    app.press("label:Steps")?;
    feed([
        // Colours, a redrawn progress line and a wide character.
        frame(
            r#"{"type":"event","at_ms":2,"session_id":"app:synthetic-11","event":{"type":"turn_status","turn_id":"t11","status":{"kind":"tool_progress","name":"terminal","chunk":"\u001b[32mCompiling\u001b[0m 10%\rCompiling 100%\n✅ 漢字 build ok\n"}}}"#,
        ),
    ]);
    // The working line says the output's latest line, without escapes.
    app.wait_for("text:terminal: ✅ 漢字 build ok")?;
    // Steps shows it as text: the progress line at its last state.
    app.wait_for("Compiling 100%\n✅ 漢字 build ok")?;
    Ok(())
}

#[test]
fn a_handoff_card_opens_the_live_view_and_watching_needs_no_card() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-live","channel":"app","title":"Book the synthetic table","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Book the synthetic table")?;
    app.press("Book the synthetic table >> nth=0")?;
    feed([
        frame(
            r#"{"type":"event","at_ms":1,"event":{"type":"surface_updated","surface":{"id":"srf_live_1","kind":"browser_tab","session_id":"app:synthetic-live","title":"Sign in","site":"login.acme.example","width":1280,"height":720,"controller":{"kind":"agent"},"handoff_pending":true,"interactive":true}}}"#,
        ),
        frame(
            r#"{"type":"event","seq":1,"at_ms":2,"session_id":"app:synthetic-live","event":{"type":"approval_requested","approval":{"id":"ap-live","session_id":"app:synthetic-live","kind":"handoff","risk":"low","summary":"Sign in to your Acme account on login.acme.example, then press Done.","options":["approve_once","deny"],"requested_at_ms":2,"card":{"action":{"kind":"take_over","surface_id":"srf_live_1","what":"Sign in to your Acme account","site":"login.acme.example","keeps_sign_in":true}}}}}"#,
        ),
    ]);
    // The step is the heading; the site and how to do it come under it.
    app.wait_for("Sign in to your Acme account")?;
    app.wait_for("On login.acme.example. Do it in the live view or at the computer, then press Done. The agent waits and never sees what you type. A sign-in here is kept for later tasks.")?;
    app.wait_for("Not now")?;
    let before = sent("open_surface");
    app.press("Open the live view")?;
    wait_sent_new("open_surface", before)?;
    app.wait_for("Connecting to the screen…")?;
    app.wait_for("Done")?;
    // The X hides the view; the agent keeps waiting.
    app.press("label:Hide the live view")?;
    app.wait_for("Still waiting for you: open the live view again from the card")?;
    // Watching needs no card: the header offers the browser while it is
    // open in the conversation.
    let before = sent("open_surface");
    app.press("label:The agent is waiting for you in its browser")?;
    wait_sent_new("open_surface", before)?;
    app.wait_for("Connecting to the screen…")?;
    // Watching, a tap can point the agent at something.
    app.press("Point")?;
    app.wait_for("Tap what you mean")?;
    // The view is built once per opening: the client's other changes (the
    // page's new title here) keep its state (a zoom, the keyboard, Point).
    feed([frame(
        r#"{"type":"event","at_ms":3,"event":{"type":"surface_updated","surface":{"id":"srf_live_1","kind":"browser_tab","session_id":"app:synthetic-live","title":"Secure area","site":"login.acme.example","width":1280,"height":720,"controller":{"kind":"agent"},"handoff_pending":true,"interactive":true}}}"#,
    )]);
    app.wait_for("Secure area · login.acme.example")?;
    app.wait_for("Tap what you mean")?;
    Ok(())
}

/// A scheduled run waiting on a sign-in says so on its automation's card,
/// with the way to do it.
#[test]
fn an_automation_waiting_on_a_sign_in_says_so_on_its_card() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"event","at_ms":1,"event":{"type":"surface_updated","surface":{"id":"srf_auto_1","kind":"browser_tab","session_id":"telegram:synthetic-user-1","title":"Sign in","site":"bank.acme.example","width":1280,"height":720,"controller":{"kind":"agent"},"handoff_pending":true,"interactive":true}}}"#,
        ),
    ]);
    app.press("Automations: what runs on its own")?;
    app.wait_for("Reading your automations…")?;
    feed([frame(
        r#"{"type":"ack","request_id":"r9","result":{"type":"automations","automations":[{"id":"goal-7","kind":"scheduled","name":"Statement download","what":"Download the synthetic statement","how_often":"Every month on the 1st","status":"active","session_id":"telegram:synthetic-user-1","recent_runs":[{"started_ms":1767002400000,"outcome":"running"}]}]}}"#,
    )]);
    app.wait_for("Waiting for you on bank.acme.example")?;
    let before = sent("open_surface");
    app.press("Open the live view")?;
    wait_sent_new("open_surface", before)?;
    Ok(())
}

#[test]
fn a_sent_message_can_be_sent_again_or_edited_first() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-again","channel":"app","title":"Open the sign-in page","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Open the sign-in page")?;
    app.press("Open the sign-in page >> nth=0")?;
    // A message no reply came to: nothing failed, so there is no Retry,
    // and the message itself can go again.
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-again","event":{"type":"user_message","session_id":"app:synthetic-again","text":"Open login.acme.example and hand it over","origin":"app"}}"#,
    )]);
    app.wait_for("Open login.acme.example and hand it over")?;
    let before = sent("send_message");
    app.press("Send again")?;
    let again = wait_sent_new("send_message", before)?;
    // Edit puts it in the composer; Send sends what is there.
    app.press("Edit and send again")?;
    app.press("Send")?;
    wait_sent_new("send_message", Some(again))?;
    // While the conversation works, a message goes from the composer.
    feed([frame(
        r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-again","event":{"type":"turn_started","turn_id":"t1","session_id":"app:synthetic-again","origin":"app"}}"#,
    )]);
    app.wait_for_gone("Send again", std::time::Duration::from_secs(2))?;
    assert!(app.press("Edit and send again").is_ok());
    Ok(())
}

#[test]
fn a_reply_starting_in_a_long_conversation_is_in_view() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What can I help you with?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-long","channel":"app","title":"A long synthetic chat","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("A long synthetic chat")?;
    let before = sent("get_history");
    app.press("A long synthetic chat >> nth=0")?;
    let id = wait_sent_new("get_history", before)?;
    let messages = (0..120)
        .map(|i| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            serde_json::json!({"id": format!("m{i}"), "role": role, "text": format!("Synthetic message {i}.\n\nWith a second paragraph."), "at_ms": 1000 + i})
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"history","session_id":"app:synthetic-long","messages":[{messages}]}}}}"#
    ))]);
    app.wait_for("Synthetic message 119.")?;
    feed([frame(
        r#"{"type":"event","seq":1,"at_ms":5000,"session_id":"app:synthetic-long","event":{"type":"turn_started","turn_id":"t1","session_id":"app:synthetic-long","origin":"app"}}"#,
    )]);
    // The reply's working line, at the end, in view: not an old message.
    app.wait_for("Working…")?;
    Ok(())
}

/// "Pair a phone" answers the press: it shows the code, or here (no
/// daemon at the address) why not. Its task ran under the button that
/// setting the phase rebuilds, so it ended before it sent anything and the
/// press did nothing.
#[test]
fn pair_a_phone_answers_the_press() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    // A port nothing listens on, and a token: this computer runs the daemon.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| e.to_string())?
        .port();
    // SAFETY: the tests take turns (TURN), and no other thread reads these.
    unsafe {
        std::env::set_var("AIDAEMON_CLIENT_TOKEN", "acg_synthetic");
        std::env::set_var("AIDAEMON_GATEWAY_URL", format!("http://127.0.0.1:{port}"));
    }
    let mut app = start_desktop();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    let before = sent("get_settings");
    app.press("Settings: change how the daemon runs")?;
    let id = wait_sent_new("get_settings", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{SETTINGS}}}"#
    ))]);
    let before = sent("list_devices");
    app.press("Connections")?;
    let id = wait_sent_new("list_devices", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"devices","devices":[],"can_change":true}}}}"#
    ))]);
    app.wait_for("Pair a phone")?;
    app.press("Pair a phone")?;
    let answered = app.wait_for("text:/^Could not reach AIDaemon/");
    // SAFETY: as above.
    unsafe {
        std::env::remove_var("AIDAEMON_CLIENT_TOKEN");
        std::env::remove_var("AIDAEMON_GATEWAY_URL");
    }
    answered?;
    Ok(())
}

/// The System page lists the owner's devices with how each reaches the
/// daemon, from the protocol's own golden reply.
#[test]
fn the_system_page_lists_devices_and_how_they_connect() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start_desktop();
    app.wait_for("What can I help you with?")?;
    feed([Update::Connected]);
    let before = sent("get_system");
    app.press("System")?;
    let id = wait_sent_new("get_system", before)?;
    let fixture =
        include_str!("../../../crates/client-protocol/tests/fixtures/server_ack_system.golden");
    let mut reply: serde_json::Value = serde_json::from_str(fixture).map_err(|e| e.to_string())?;
    reply["request_id"] = serde_json::Value::String(id);
    feed([frame(&reply.to_string())]);
    app.wait_for("Devices")?;
    app.wait_for("Synthetic Companion")?;
    app.wait_for("text:/Local network · direct/")?;
    app.wait_for("Online")?;
    Ok(())
}
