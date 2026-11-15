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

fn start_at(width: f32, height: f32) -> Headless {
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(width), Points(height));
    Headless::start(config).unwrap()
}

/// A desktop window, with a mouse to hover with.
fn start_desktop() -> Headless {
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
    app.press("Fix the flaky test >> nth=0")?;
    // A conversation away from code has its steps a press away.
    app.press("Steps")?;
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
    app.wait_for("terminal")?;
    app.wait_for("test port_race ... FAILED")?;
    app.wait_for("git push origin fix-port")?;
    app.wait_for("Allow once")?;
    // The conversation waits on its owner: it leads the list.
    app.wait_for("Needs you")?;
    // Waiting on its owner, the working line says so, not the last tool.
    app.wait_for("text:Waiting for your approval")?;
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

/// A daemon's settings, as `get_settings` answers.
const SETTINGS: &str = r#"{"settings": {"bots": [{"allowed_user_ids": ["telegram:synthetic-user-1"], "created_at": "2026-01-01 00:00:00", "id": 3, "platform": "telegram"}], "chatgpt_signed_in": true, "checked": {"models": ["gpt-x", "gpt-y"], "preset": "openai", "suggested": "gpt-x"}, "config_path": "/home/synthetic/.aidaemon/config.toml", "connected": [{"credential": "signed_in", "models": ["synthetic-model-1"], "hard_model": "synthetic-strong", "quick_model": "synthetic-quick", "name": "OpenAI (ChatGPT subscription)", "preset": "openai_chatgpt"}, {"base_url": "https://api.example.com/v1", "credential": "key_in_file", "failover": 0, "key_setting": "provider.fallbacks[0].api_key", "models": ["gpt-x"], "offered_models": ["gpt-x", "gpt-astra-synthetic"], "quick_model": "gpt-x", "name": "OpenAI", "preset": "openai"}, {"credential": "key_stored", "failover": 1, "key_setting": "provider.fallbacks[1].api_key", "models": ["synthetic-a"], "name": "Anthropic", "preset": "anthropic"}], "database_encrypted": true, "database_path": "/home/synthetic/.aidaemon/aidaemon.db", "groups": [{"name": "provider", "settings": [{"applies": "reload", "description": "The model most turns use.", "example": "\"openai/gpt-4o\"", "guarded": false, "key": "provider.models.default", "kind": "text", "secret_set": false, "value": "\"synthetic-model-1\""}, {"applies": "reload", "guarded": true, "key": "provider.api_key", "kind": "secret", "secret_set": true}]}], "mcp_servers": [{"command": "synthetic-mcp --root /tmp", "enabled": true, "from_config": false, "name": "files", "tools": ["read_file"]}], "notices": ["The database moves to /data/aidaemon.db at the next restart."], "providers": [{"billing": "per_token", "key_url": "https://platform.openai.com/api-keys", "model": "gpt-x", "name": "OpenAI", "needs_base_url": false, "needs_key": true, "preset": "openai"}], "reload_pending": true, "restart_pending": false}, "type": "settings"}"#;

#[test]
fn the_terminal_settings_are_one_page_of_models_reached_by_keyboard() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.idiom = nsh_core::platform::Idiom::Terminal;
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What should we work on?")?;
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
    app.wait_for("Hard work: synthetic-strong · Quick jobs: synthetic-quick")?;
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
    app.wait_for("What should we work on?")?;
    feed([Update::Connected]);
    app.press("Settings: change how the daemon runs")?;
    app.wait_for("Reading settings…")?;
    feed([frame(
        r#"{"type":"event","at_ms":1,"event":{"type":"settings_updated","ok":true,"message":"Updated memory.retention_days = 30"}}"#,
    )]);
    app.wait_for("Updated memory.retention_days = 30")?;
    // The settings arrive: a value edits as TOML in a code editor, and
    // another key is typed with the known keys suggested.
    let id = wait_sent("get_settings")?;
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
    app.wait_for("Hard work: synthetic-strong · Quick jobs: synthetic-quick")?;
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

#[test]
fn a_snapshot_mid_turn_shows_what_the_turn_is_doing() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"snapshot","seq":7,"state":{"sessions":[{"id":"app:synthetic-3","channel":"app","title":"Count the files","writable":true}],"running_tasks":[{"id":"b:app:synthetic-3#1","session_id":"app:synthetic-3","number":1,"description":"Count the files","state":"running","started_at_ms":1}],"pending_approvals":[],"live_turns":[{"session_id":"app:synthetic-3","turn_id":"t9","started_at_ms":1,"status":{"kind":"tool_started","name":"terminal","summary":"find . -name '*.rs'"},"tools":["read_file","terminal"]}]}}"#,
        ),
    ]);
    app.wait_for("Count the files")?;
    app.press("Count the files >> nth=0")?;
    app.press("Steps")?;
    // The working line and the steps both say what it is doing.
    app.wait_for("text:terminal: find . -name '*.rs'...  ·  2 tools so far")?;
    app.wait_for("read_file")?;
    Ok(())
}

#[test]
fn a_file_write_gives_a_conversation_its_changes_tab() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-2","channel":"app","title":"Add the contact form","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Add the contact form")?;
    app.press("Add the contact form >> nth=0")?;
    // A conversation that has not touched code has its steps alone, a
    // press away.
    app.press("Steps")?;
    app.wait_for("This turn")?;
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
    // Its steps one press away, and back.
    app.press("Steps")?;
    app.wait_for("This turn")?;
    app.press("Changes")?;
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
    app.wait_for("Undo changes")?;
    app.press("Side by side")?;
    app.press("Next change")?;
    app.press("All files")?;
    app.wait_for("acme-site")?;
    Ok(())
}

#[test]
fn chats_filter_by_source_and_a_group_asks_before_posting() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-5","channel":"app","title":"Weekend plan","writable":true},{"id":"synthetic_coding_bot:7","channel":"telegram:synthetic_coding_bot","title":"Fix the footer","writable":true},{"id":"synthetic_coding_bot:-9","channel":"telegram:synthetic_coding_bot","title":"Team release notes","writable":true,"group":true}]}}"#,
        ),
    ]);
    app.wait_for("Weekend plan")?;
    // Where each came from, as filters.
    app.wait_for("Telegram · synthetic_coding_bot · group")?;
    app.press("Groups")?;
    app.wait_for_gone("Weekend plan", std::time::Duration::from_secs(5))?;
    app.wait_for("Team release notes")?;
    // A group: posting from here is the owner's explicit choice.
    app.press("Team release notes >> nth=0")?;
    app.wait_for("Post in this group")?;
    app.press("Post in this group")?;
    app.wait_for("Also posted in Telegram · synthetic_coding_bot · group")?;
    // A direct chat posts there too, and says so.
    app.press("All")?;
    app.press("Fix the footer >> nth=0")?;
    app.wait_for("Also posted in Telegram · synthetic_coding_bot")?;
    Ok(())
}

#[test]
fn projects_group_their_conversations_and_start_new_ones_in_their_folder() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"projects","projects":[{"id":"project-acme","name":"Acme site","path":"/home/synthetic/acme-site"},{"id":"project-notes","name":"Design notes"}]}}"#,
        ),
        frame(
            r#"{"type":"ack","request_id":"r1","result":{"type":"sessions","sessions":[{"id":"app:synthetic-3","channel":"app","title":"Contact form copy","writable":true,"project_id":"project-acme"},{"id":"app:synthetic-4","channel":"app","title":"Grocery list","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Acme site")?;
    app.wait_for("Design notes")?;
    // A conversation in no project is listed on its own.
    app.wait_for("Conversations")?;
    app.wait_for("Grocery list")?;
    // A project opens on its conversations.
    app.press("Project Acme site, 1 conversations")?;
    app.wait_for("Contact form copy")?;
    // Its "+" starts a conversation there: the header names the project.
    app.press("New conversation in this project >> nth=0")?;
    app.wait_for("Explain this project")?;
    app.wait_for("Acme site · /home/synthetic/acme-site")?;
    // A new conversation says where it works: Chat leaves the project and
    // brings the everyday starters; In a project goes back to one.
    app.press("Chat")?;
    app.wait_for("Set a reminder")?;
    app.press("In a project")?;
    app.wait_for("Explain this project")?;
    // A new conversation from the keyboard, in a project's conversation,
    // stays in that project.
    app.press("Contact form copy >> nth=0")?;
    app.press_key("mod+n")?;
    app.wait_for("Explain this project")?;
    app.wait_for("Acme site · /home/synthetic/acme-site")?;
    Ok(())
}

#[test]
fn a_file_the_agent_sends_shows_with_its_reply() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
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
    app.wait_for("What should we work on?")?;
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
    Ok(())
}

#[test]
fn automations_list_and_ask_before_cancelling() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
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
    app.wait_for(r"text:/^Last run .* · Failed: The synthetic token expired\.$/")?;
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
    app.wait_for("Scheduled · Every weekday at 9:00 AM ET")?;
    // A paused one resumes; only scheduled work runs now.
    app.wait_for("Paused")?;
    app.wait_for("Resume")?;
    app.wait_for("Run now")?;
    // Cancelling asks first.
    app.press("Cancel >> nth=0")?;
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
    app.wait_for("What should we work on?")?;
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
    app.wait_for("Scheduled · Every weekday and Saturday at 9:00 AM ET")?;
    Ok(())
}

#[test]
fn resting_the_pointer_on_a_conversation_previews_it() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start_desktop();
    app.wait_for("What should we work on?")?;
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
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.os = nsh_core::platform::Os::MacOs;
    config.environment.idiom = nsh_core::platform::Idiom::Desktop;
    config.speech = Some(std::sync::Arc::new(Hears));
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What should we work on?")?;
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
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-7","channel":"app","title":"Draft the invoice","writable":true},{"id":"app:synthetic-8","channel":"app","title":"Plan the trip","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Plan the trip")?;
    app.press("Plan the trip >> nth=0")?;
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-7","event":{"type":"turn_started","turn_id":"t7","session_id":"app:synthetic-7","origin":"app"}}"#,
        ),
        frame(
            r#"{"type":"event","seq":2,"at_ms":2,"session_id":"app:synthetic-7","event":{"type":"turn_completed","turn_id":"t7","text":"The invoice is drafted.","state":"completed"}}"#,
        ),
    ]);
    app.wait_for("Done: Draft the invoice")?;
    Ok(())
}

#[test]
fn the_composer_picks_a_model_and_an_effort_on_a_phone() -> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start_at(390.0, 844.0);
    app.wait_for("What should we work on?")?;
    let before = sent("list_models");
    feed([Update::Connected]);
    let id = wait_sent_new("list_models", before)?;
    feed([frame(&format!(
        r#"{{"type":"ack","request_id":"{id}","result":{{"type":"models","options":{{"providers":[
            {{"id":"main","name":"Anthropic","billing":"per_token","models":[{{"id":"synthetic-claude","efforts":["low","medium","high"]}}]}},
            {{"id":"failover-0","name":"Google Gemini","billing":"per_token","models":[{{"id":"synthetic-gemini"}}]}}
        ]}}}}}}"#
    ))]);
    app.wait_for("Auto model")?;
    app.wait_for("Auto effort")?;
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
    // Gemini takes no effort: its choice goes.
    app.press("Auto model")?;
    app.press("synthetic-gemini · Google Gemini")?;
    app.wait_for("synthetic-gemini · Google Gemini")?;
    app.wait_for_gone("Auto effort", std::time::Duration::from_secs(5))?;
    // An existing conversation's pick goes to the daemon at once (in a
    // window, where the list of conversations shows).
    drop(app);
    let mut app = start();
    app.wait_for("What should we work on?")?;
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
    app.wait_for("Auto model")?;
    app.press("Auto model")?;
    app.press("synthetic-claude · Anthropic")?;
    app.wait_for("synthetic-claude · Anthropic")?;
    wait_sent_new("set_session_model", None)?;
    app.wait_for("Auto effort")?;
    Ok(())
}

#[test]
fn archived_conversations_wait_under_archived_and_queued_messages_can_be_removed()
-> Result<(), String> {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = start();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-9","channel":"app","title":"Old receipts","writable":true,"archived":true},{"id":"app:synthetic-10","channel":"app","title":"Write the newsletter","writable":true}]}}"#,
        ),
    ]);
    app.wait_for("Write the newsletter")?;
    // An archived conversation is not in the list, only counted under it.
    app.wait_for("Archived · 1")?;
    app.wait_for_gone("Old receipts", std::time::Duration::from_secs(2))?;
    app.press("Archived · 1")?;
    app.wait_for("Old receipts")?;
    app.press("Back to your conversations")?;
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
    let mut config = EngineConfig::new(Root::of::<Offline>());
    config.viewport = Size::new(Points(1280.0), Points(820.0));
    config.environment.idiom = nsh_core::platform::Idiom::Terminal;
    let mut app = Headless::start(config).unwrap();
    app.wait_for("What should we work on?")?;
    feed([
        Update::Connected,
        frame(
            r#"{"type":"ack","request_id":"r0","result":{"type":"sessions","sessions":[{"id":"app:synthetic-11","channel":"app","title":"Run the build","writable":true}]}}"#,
        ),
    ]);
    // A terminal sends Ctrl+I as Tab: the steps' key there is O.
    app.wait_for(r"text:/O steps/")?;
    // The terminal starts on the conversation; the list is a key away.
    app.press("label:Conversations")?;
    app.wait_for("Run the build")?;
    app.press("Run the build >> nth=0")?;
    app.press("label:Steps")?;
    feed([
        frame(
            r#"{"type":"event","seq":1,"at_ms":1,"session_id":"app:synthetic-11","event":{"type":"turn_started","turn_id":"t11","session_id":"app:synthetic-11","origin":"app"}}"#,
        ),
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
