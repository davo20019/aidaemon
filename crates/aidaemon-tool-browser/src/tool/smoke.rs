//! Ignored real-Chrome smoke tests for the browser tool.
//!
//! Run locally (requires Chrome/Chromium installed):
//! ```bash
//! cargo test --features browser browser_smoke_real_chrome -- --ignored --nocapture
//! ```

use aidaemon_tools::testing::ToolCalls as _;
use std::time::Duration;

use serde_json::json;
use tempfile::TempDir;
use tokio::sync::mpsc;

use crate::settings::BrowserSettings;
use crate::tool::BrowserTool;
use aidaemon_tools::semantics::ApprovalResponse;
use aidaemon_tools::testing::CallWithFacts;

/// Run `arguments` as the kernel does, the owner approving any card for the
/// session.
async fn call_approved(tool: &BrowserTool, arguments: &str, session: &str) -> String {
    aidaemon_tools::testing::call_through_kernel(
        tool,
        arguments,
        aidaemon_tools::testing::call_facts(session),
        Some(ApprovalResponse::AllowSession),
    )
    .await
    .expect("browser call")
    .output
}

fn smoke_browser_config(user_data_dir: &str) -> BrowserSettings {
    BrowserSettings {
        headless: true,
        user_data_dir: Some(user_data_dir.to_string()),
        session_isolation: Some(crate::settings::SessionIsolation::BrowserContext),
        ..BrowserSettings::default()
    }
}

fn write_smoke_fixture(dir: &TempDir) -> String {
    let path = dir.path().join("page.html");
    std::fs::write(
        &path,
        r#"<!DOCTYPE html>
<html><body>
<input id="name" type="text" value="old">
<button id="go" type="button">Go</button>
<div id="out">hello smoke</div>
<script>
  document.getElementById('go').addEventListener('click', function() {
    document.getElementById('name').value = 'new';
    document.getElementById('out').textContent = 'clicked';
  });
  console.error('smoke-console-error');
</script>
</body></html>"#,
    )
    .expect("write fixture");
    format!("file://{}", path.display())
}

#[tokio::test]
#[ignore = "requires local Chrome; run with --ignored"]
async fn browser_smoke_real_chrome() {
    let profile = TempDir::new().expect("temp profile");
    let fixture = TempDir::new().expect("temp fixture");
    let fixture_url = write_smoke_fixture(&fixture);

    let (media_tx, _media_rx) = mpsc::channel(4);
    let tool = BrowserTool::new(
        smoke_browser_config(profile.path().to_str().unwrap()),
        media_tx,
        profile.path().join("inbox"),
        aidaemon_tools::testing::active_execution_backend(),
    )
    .expect("construct browser tool");

    let session = "smoke-session";

    let nav = tool
        .call_with(
            &json!({
                "action": "navigate",
                "url": fixture_url,
            })
            .to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await
        .expect("navigate");
    assert!(nav.starts_with("Navigated to"), "navigate failed: {nav}");

    let fill = call_approved(
        &tool,
        &json!({
            "action": "fill",
            "selector": "#name",
            "value": "replaced",
        })
        .to_string(),
        session,
    )
    .await;
    assert_eq!(fill, "Filled '#name'");

    let click = call_approved(
        &tool,
        &json!({
            "action": "click",
            "selector": "#go",
        })
        .to_string(),
        session,
    )
    .await;
    assert!(click.contains("Clicked element '#go'"), "{click}");

    tokio::time::sleep(Duration::from_millis(300)).await;

    let text = tool
        .call_with(
            &json!({
                "action": "get_text",
                "selector": "#out",
            })
            .to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await
        .expect("get_text");
    assert!(
        text.contains("clicked"),
        "expected post-click text, got: {text}"
    );

    let logs = tool
        .call_with(
            &json!({
                "action": "get_console_logs",
            })
            .to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await
        .expect("get_console_logs");
    assert!(
        logs.contains("smoke-console-error") || logs.contains("Console logs"),
        "expected console capture, got: {logs}"
    );

    let shot = tool
        .call_with(
            &json!({
                "action": "screenshot",
            })
            .to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await
        .expect("screenshot");
    assert!(
        shot.contains("Screenshot captured"),
        "screenshot failed: {shot}"
    );

    let close = tool
        .call_with(
            &json!({ "action": "close" }).to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await
        .expect("close");
    assert!(
        close.contains("closed") || close.contains("Closed"),
        "close message unexpected: {close}"
    );
}

#[tokio::test]
#[ignore = "requires local Chrome; run with --ignored"]
async fn browser_render_pdf_real_chrome() {
    let fixture = TempDir::new().expect("temp fixture");
    let source = fixture.path().join("designed-document.html");
    std::fs::write(
        &source,
        r#"<!doctype html>
<style>
@page { size: letter; margin: 0 }
* { box-sizing: border-box }
html, body { margin: 0; background: #08352d; color: white }
.page { width: 8.5in; height: 11in; padding: 1in; page-break-after: always }
</style>
<section class="page"><h1>General PDF renderer</h1><p>Page one.</p></section>
<section class="page"><h2>Second page</h2><p>Backgrounds and CSS page size are preserved.</p></section>"#,
    )
    .expect("write PDF fixture");

    let config = BrowserSettings {
        headless: true,
        user_data_dir: None,
        session_isolation: Some(crate::settings::SessionIsolation::BrowserContext),
        ..BrowserSettings::default()
    };
    let (media_tx, _media_rx) = mpsc::channel(4);
    let tool = BrowserTool::new(
        config,
        media_tx,
        fixture.path().join("inbox"),
        aidaemon_tools::testing::active_execution_backend(),
    )
    .expect("construct browser tool");

    let rendered = tool
        .call_with(
            &json!({
                "action": "render_pdf",
                "source_path": source,
                "output_filename": "designed-document.pdf",
            })
            .to_string(),
            aidaemon_tools::testing::call_facts("pdf-smoke"),
        )
        .await
        .expect("render action");
    assert!(
        rendered.starts_with("PDF rendered with Chromium"),
        "render failed: {rendered}"
    );
    let output_path = rendered
        .split_once("saved to: ")
        .and_then(|(_, rest)| rest.lines().next())
        .map(std::path::PathBuf::from)
        .expect("render response should expose output path");
    let bytes = std::fs::read(&output_path).expect("read rendered PDF");
    assert!(bytes.starts_with(b"%PDF-"));
    assert!(bytes.len() > 1_000, "real PDF should be nontrivial");

    if let Ok(qa_dir) = std::env::var("AIDAEMON_DEPLOY_QA_DIR") {
        let qa_dir = std::path::PathBuf::from(qa_dir);
        std::fs::create_dir_all(&qa_dir).expect("create persistent PDF QA directory");
        let qa_pdf = qa_dir.join("deployed-render-regression.pdf");
        let qa_html = qa_dir.join("deployed-render-regression.html");
        std::fs::copy(&output_path, &qa_pdf).expect("preserve rendered PDF for visual QA");
        std::fs::copy(&source, &qa_html).expect("preserve source HTML for visual QA");
        println!("QA_PDF={}", qa_pdf.display());
    }

    let _ = tool.call(&json!({ "action": "close" }).to_string()).await;
}

/// Live surfaces on a real headless Chrome: the page streams frames, the
/// owner's tap focuses a field, their text and Enter reach it, and nothing
/// typed is readable from the page text the agent gets.
#[tokio::test]
#[ignore = "requires local Chrome; run with --ignored"]
async fn live_surface_smoke_real_chrome() {
    use aidaemon_surfaces::{
        InputEvent, KeyAction, LiveMode, PointerAction, PointerButton, StreamParams,
        SurfaceRegistry, SurfacesConfig,
    };
    let profile = TempDir::new().expect("temp profile");
    let fixture = TempDir::new().expect("temp fixture");
    let path = fixture.path().join("signin.html");
    std::fs::write(
        &path,
        r#"<!DOCTYPE html><html><body style="margin:0">
<input id="pw" type="password" style="position:absolute;left:0;top:0;width:100vw;height:50vh;font-size:40px">
<div id="out">waiting</div>
<script>
document.getElementById('pw').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('out').textContent =
    'submitted ' + document.getElementById('pw').value.length;
});
</script></body></html>"#,
    )
    .expect("write fixture");
    let (media_tx, _media_rx) = mpsc::channel(4);
    let tool = BrowserTool::new(
        BrowserSettings {
            headless: true,
            user_data_dir: Some(profile.path().to_str().unwrap().to_string()),
            ..BrowserSettings::default()
        },
        media_tx,
        profile.path().join("inbox"),
        aidaemon_tools::testing::active_execution_backend(),
    )
    .expect("construct browser tool");
    let session = "live-smoke";
    // Served over local HTTP (the tool opens no file: URLs).
    let page = std::fs::read(&path).expect("fixture");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        while let Ok((mut stream, _)) = listener.accept().await {
            let page = page.clone();
            tokio::spawn(async move {
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request).await;
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    page.len()
                );
                let _ = stream.write_all(head.as_bytes()).await;
                let _ = stream.write_all(&page).await;
            });
        }
    });
    let nav = call_approved(
        &tool,
        &json!({"action": "navigate", "url": format!("http://127.0.0.1:{port}/signin")})
            .to_string(),
        session,
    )
    .await;
    assert!(nav.starts_with("Navigated to"), "navigate failed: {nav}");

    let registry = SurfaceRegistry::new(SurfacesConfig::default());
    let surfaces = tool.surface_backend();
    surfaces.set_notifier(registry.notifier());
    registry.add_backend(surfaces);
    let surface = registry
        .find_for_agent(session, None)
        .await
        .expect("the tab is a surface");
    let mut live = registry
        .open_live(
            &surface.id,
            "smoke-phone",
            LiveMode::Control,
            StreamParams::default(),
        )
        .await
        .expect("open live");
    let frame = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(frame) = live.frames.borrow_and_update().clone() {
                return frame;
            }
            live.frames.changed().await.expect("stream open");
        }
    })
    .await
    .expect("a frame within 10 s");
    assert_eq!(&frame.data[..2], &[0xff, 0xd8], "a JPEG");
    assert!(frame.w > 0 && frame.h > 0);

    for action in [PointerAction::Down, PointerAction::Up] {
        registry
            .input(
                &live.live_id,
                &InputEvent::Pointer {
                    action,
                    x: 0.5,
                    y: 0.2,
                    button: PointerButton::Left,
                },
            )
            .await
            .expect("tap");
    }
    let focused = registry
        .focused(&surface.id)
        .await
        .expect("focus")
        .expect("the tapped field has the keyboard");
    assert_eq!(focused.what, "a password field");
    let [_, y, w, h] = focused.rect.expect("the field's place on the frame");
    assert!(w > 0.5 && h > 0.2 && y < 0.1, "{:?}", focused.rect);
    registry
        .input(
            &live.live_id,
            &InputEvent::Text {
                text: "synthetic-pw".into(),
            },
        )
        .await
        .expect("type");
    for action in [KeyAction::Down, KeyAction::Up] {
        registry
            .input(
                &live.live_id,
                &InputEvent::Key {
                    action,
                    code: "Enter".into(),
                    key: "Enter".into(),
                    mods: vec![],
                },
            )
            .await
            .expect("enter");
    }
    let pointed = registry
        .describe_point(&surface.id, 0.5, 0.2)
        .await
        .expect("describe the point");
    assert_eq!(pointed.what, "a password field");
    assert_eq!(pointed.selector.as_deref(), Some("#pw"));
    assert!(pointed.text.is_none(), "a field's value is never described");
    // The picture keeps up: a change after typing arrives as a new frame.
    let before = live
        .frames
        .borrow_and_update()
        .as_ref()
        .map(|f| f.seq)
        .unwrap_or(0);
    registry
        .input(&live.live_id, &InputEvent::Text { text: "x".into() })
        .await
        .expect("type more");
    let after = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            live.frames.changed().await.expect("stream open");
            let seq = live
                .frames
                .borrow_and_update()
                .as_ref()
                .map(|f| f.seq)
                .unwrap_or(0);
            if seq > before {
                return seq;
            }
        }
    })
    .await;
    assert!(
        after.is_ok(),
        "no new frame after typing: the picture stalled at {before}"
    );
    let observed = registry.observe(&surface.id).await.expect("observe");
    assert!(observed.password_field_present);
    registry.close_live(&live.live_id);
    registry
        .release_control(&surface.id, "smoke-phone")
        .unwrap();

    let text = tool
        .call_with(
            &json!({"action": "get_text", "selector": "body"}).to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await
        .expect("get_text");
    assert!(
        text.contains("submitted 12"),
        "the typed text and Enter reached the page: {text}"
    );
    assert!(
        !text.contains("synthetic-pw"),
        "the page text never carries the password"
    );
    let _ = tool
        .call_with(
            &json!({"action": "close"}).to_string(),
            aidaemon_tools::testing::call_facts(session),
        )
        .await;
}
