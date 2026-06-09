//! Smoke tests proving `BrowserTool` dispatch works against `MockBackend`
//! (no real Chrome). Covers one observation action and one mutation action.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use super::backend::{MockBackend, MockCall};
use super::BrowserTool;
use crate::traits::Tool;

fn mock_tool() -> (
    BrowserTool,
    Arc<MockBackend>,
    mpsc::Receiver<crate::types::MediaMessage>,
) {
    let backend = Arc::new(MockBackend::new());
    let (media_tx, media_rx) = mpsc::channel(8);
    let tool = BrowserTool::with_backend(backend.clone(), media_tx);
    (tool, backend, media_rx)
}

#[tokio::test]
async fn dispatch_observation_navigate_routes_through_backend() {
    let (tool, backend, _rx) = mock_tool();

    let args = json!({ "action": "navigate", "url": "https://example.com/" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Navigated to https://example.com/");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::EnsureReady),
        "ensure_ready should be called: {calls:?}"
    );
    assert!(
        calls.contains(&MockCall::CurrentPage),
        "current_page should be called: {calls:?}"
    );
    assert!(
        calls.contains(&MockCall::Goto("https://example.com/".to_string())),
        "goto should record the URL: {calls:?}"
    );
}

#[tokio::test]
async fn dispatch_mutation_click_routes_through_backend() {
    let (tool, backend, _rx) = mock_tool();

    let args = json!({ "action": "click", "selector": "#submit" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Clicked element '#submit'");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::EnsureReady),
        "ensure_ready should be called: {calls:?}"
    );
    assert!(
        calls.contains(&MockCall::Click("#submit".to_string())),
        "click should record the selector: {calls:?}"
    );
}
