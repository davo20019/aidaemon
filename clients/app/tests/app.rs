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
        r#"{"type":"ack","request_id":"r9","result":{"type":"automations","automations":[{"id":"goal-1","kind":"scheduled","name":"Inbox digest","what":"Summarize the synthetic inbox","how_often":"Every weekday at 9:00 AM ET","status":"active"},{"id":"mandate-1","kind":"watching","name":"Status watch","what":"Watch the synthetic status page","how_often":"Every hour","status":"paused"}]}}"#,
    )]);
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
