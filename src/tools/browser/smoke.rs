//! Ignored real-Chrome smoke tests for the browser tool.
//!
//! Run locally (requires Chrome/Chromium installed):
//! ```bash
//! cargo test --features browser browser_smoke_real_chrome -- --ignored --nocapture
//! ```

use std::time::Duration;

use serde_json::json;
use tempfile::TempDir;
use tokio::sync::mpsc;

use crate::config::BrowserConfig;
use crate::tools::terminal::ApprovalRequest;
use crate::tools::{ApprovalBroker, BrowserTool};
use crate::traits::Tool;
use crate::types::ApprovalResponse;

fn auto_approve_broker() -> ApprovalBroker {
    let (tx, mut rx) = mpsc::channel::<ApprovalRequest>(8);
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            let _ = req.response_tx.send(ApprovalResponse::AllowSession);
        }
    });
    ApprovalBroker::new(tx)
}

fn smoke_browser_config(user_data_dir: &str) -> BrowserConfig {
    BrowserConfig {
        enabled: true,
        headless: true,
        user_data_dir: Some(user_data_dir.to_string()),
        session_isolation: Some(crate::config::SessionIsolation::BrowserContext),
        ..BrowserConfig::default()
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
        auto_approve_broker(),
        profile.path().join("inbox"),
        None,
    )
    .expect("construct browser tool");

    let session = "smoke-session";

    let nav = tool
        .call(
            &json!({
                "action": "navigate",
                "url": fixture_url,
                "_session_id": session
            })
            .to_string(),
        )
        .await
        .expect("navigate");
    assert!(nav.starts_with("Navigated to"), "navigate failed: {nav}");

    let fill = tool
        .call(
            &json!({
                "action": "fill",
                "selector": "#name",
                "value": "replaced",
                "_session_id": session
            })
            .to_string(),
        )
        .await
        .expect("fill");
    assert_eq!(fill, "Filled '#name'");

    let click = tool
        .call(
            &json!({
                "action": "click",
                "selector": "#go",
                "_session_id": session
            })
            .to_string(),
        )
        .await
        .expect("click");
    assert!(click.contains("Clicked element '#go'"), "{click}");

    tokio::time::sleep(Duration::from_millis(300)).await;

    let text = tool
        .call(
            &json!({
                "action": "get_text",
                "selector": "#out",
                "_session_id": session
            })
            .to_string(),
        )
        .await
        .expect("get_text");
    assert!(
        text.contains("clicked"),
        "expected post-click text, got: {text}"
    );

    let logs = tool
        .call(
            &json!({
                "action": "get_console_logs",
                "_session_id": session
            })
            .to_string(),
        )
        .await
        .expect("get_console_logs");
    assert!(
        logs.contains("smoke-console-error") || logs.contains("Console logs"),
        "expected console capture, got: {logs}"
    );

    let shot = tool
        .call(
            &json!({
                "action": "screenshot",
                "_session_id": session
            })
            .to_string(),
        )
        .await
        .expect("screenshot");
    assert!(
        shot.contains("Screenshot captured"),
        "screenshot failed: {shot}"
    );

    let close = tool
        .call(&json!({ "action": "close", "_session_id": session }).to_string())
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

    let config = BrowserConfig {
        enabled: true,
        headless: true,
        user_data_dir: None,
        session_isolation: Some(crate::config::SessionIsolation::BrowserContext),
        ..BrowserConfig::default()
    };
    let (media_tx, _media_rx) = mpsc::channel(4);
    let tool = BrowserTool::new(
        config,
        media_tx,
        auto_approve_broker(),
        fixture.path().join("inbox"),
        None,
    )
    .expect("construct browser tool");

    let rendered = tool
        .call(
            &json!({
                "action": "render_pdf",
                "source_path": source,
                "output_filename": "designed-document.pdf",
                "_session_id": "pdf-smoke"
            })
            .to_string(),
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
