//! Smoke tests proving `BrowserTool` dispatch works against `MockBackend`
//! (no real Chrome). Covers observation and mutation actions and locks down the
//! preservation-sensitive dispatch paths (selector vs. no-selector routing,
//! `execute_js` null/undefined handling, screenshot target selection) that
//! later tasks will touch.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

use super::backend::{MockBackend, MockCall};
use super::BrowserTool;
use crate::tools::ApprovalBroker;
use crate::traits::Tool;
use crate::types::{ApprovalResponse, MediaKind, MediaMessage};

fn mock_tool() -> (
    BrowserTool,
    Arc<MockBackend>,
    mpsc::Receiver<crate::types::MediaMessage>,
) {
    mock_tool_with(MockBackend::new())
}

/// Dispatch-smoke helper: wires an AUTO-APPROVING approval broker so these
/// pre-Task-7 tests exercise the action dispatch paths (navigate/click/fill/
/// execute_js) without each one being blocked by the approval gate. The
/// missing-channel (fail-safe Deny) path is exercised explicitly by
/// `no_channel_tool()` / the Task-7 missing-channel test.
fn mock_tool_with(
    backend: MockBackend,
) -> (
    BrowserTool,
    Arc<MockBackend>,
    mpsc::Receiver<crate::types::MediaMessage>,
) {
    let backend = Arc::new(backend);
    let (media_tx, media_rx) = mpsc::channel(8);
    let (broker, _recorder) = spawn_responder(ApprovalResponse::AllowSession);
    let tool = BrowserTool::with_backend_and_approval(
        backend.clone(),
        media_tx,
        broker,
        Duration::from_secs(5),
    );
    (tool, backend, media_rx)
}

/// Build a tool with NO approval channel (every approval-requiring action fails
/// safe to Deny).
fn no_channel_tool(
    backend: MockBackend,
) -> (
    BrowserTool,
    Arc<MockBackend>,
    mpsc::Receiver<crate::types::MediaMessage>,
) {
    let backend = Arc::new(backend);
    let (media_tx, media_rx) = mpsc::channel(8);
    let tool = BrowserTool::with_backend(backend.clone(), media_tx);
    (tool, backend, media_rx)
}

#[tokio::test]
async fn dispatch_observation_navigate_routes_through_backend() {
    let (tool, backend, _rx) = mock_tool();

    let args =
        json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Navigated to https://example.com/");

    // Lock down the call ordering: ensure_ready, then create_page (this session's
    // first action), then goto, then a bounded navigation-readiness wait (Task 13
    // — replaces the old fixed 2s sleep), then a url() read for final-committed-URL
    // revalidation (Task 8). The session registry mints a fresh page id, so
    // assert the create_page record exists and goto follows it.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert_eq!(calls.len(), 5, "expected 5 recorded calls: {calls:?}");
    assert_eq!(calls[0], MockCall::EnsureReady);
    assert!(
        matches!(calls[1], MockCall::CreatePage(_)),
        "navigate's first action must create the session page: {calls:?}"
    );
    assert_eq!(
        calls[2],
        MockCall::Goto("https://example.com/".to_string()),
        "navigate must goto after creating the page: {calls:?}"
    );
    assert_eq!(
        calls[3],
        MockCall::WaitForNavigation,
        "navigate must wait for navigation readiness (no fixed sleep): {calls:?}"
    );
    assert_eq!(
        calls[4],
        MockCall::Url,
        "navigate must read the committed url to revalidate it: {calls:?}"
    );
}

#[tokio::test]
async fn dispatch_mutation_click_routes_through_backend() {
    let (tool, backend, _rx) = mock_tool();

    let args = json!({ "action": "click", "selector": "#submit", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Clicked element '#submit'");

    // Exact sequence: ensure_ready -> create_page -> click(selector) -> a
    // navigation-readiness probe (Task 13 nav-race; the non-navigating default
    // returns fast via the short settle while still recording the probe).
    // Asserts the click is recorded and that we did NOT route through
    // goto/find_element.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert_eq!(calls.len(), 4, "expected 4 recorded calls: {calls:?}");
    assert_eq!(calls[0], MockCall::EnsureReady);
    assert!(
        matches!(calls[1], MockCall::CreatePage(_)),
        "click's first action must create the session page: {calls:?}"
    );
    assert_eq!(
        calls[2],
        MockCall::Click("#submit".to_string()),
        "click should click after creating the page: {calls:?}"
    );
    assert_eq!(
        calls[3],
        MockCall::WaitForNavigation,
        "click must run the nav-race probe (no fixed sleep): {calls:?}"
    );
}

#[tokio::test]
async fn get_text_with_selector_routes_through_inner_text() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("hello inner"));

    let args = json!({ "action": "get_text", "selector": "#headline", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "hello inner", "should return the mocked inner_text");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::InnerText("#headline".to_string())),
        "with a selector, get_text must route through inner_text: {calls:?}"
    );
    assert!(
        !calls.contains(&MockCall::BodyText),
        "with a selector, get_text must NOT route through body_text: {calls:?}"
    );
}

/// Task 14 — selector-faithful dispatch contract.
///
/// `get_text` must forward the selector to `inner_text` UNCHANGED, regardless
/// of what characters it contains.  The old implementation built a JS string
/// `document.querySelector('<selector>').innerText`, which broke on backslashes,
/// newlines, and un-escapable CSS escapes, and could inject arbitrary JS via a
/// crafted selector.  The fix moves text extraction to the element handle, so
/// the selector is only ever passed to `find_element` (a CDP DOM query — no JS
/// string building) and the text comes from `Element::inner_text()`.
///
/// These tests verify the *dispatch contract*: the selector arrives at
/// `InnerText(selector)` byte-for-byte unchanged.  The absence of any
/// `Evaluate` call with the selector embedded in it proves the interpolation is
/// gone.  Real DOM-level extraction with adversarial selectors (actual Chrome,
/// actual pages) is covered by the deferred real-Chrome smoke suite (Task 18).
#[tokio::test]
async fn get_text_selector_with_single_quote_passes_through_unchanged() {
    // `input[name='x']` — contains a single-quote that the old code escaped to
    // `\'`, mutating the selector before passing it on.
    let selector = "input[name='x']";
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("field text"));

    let args = json!({ "action": "get_text", "selector": selector, "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "field text");

    let calls = backend.calls();
    let calls = calls.lock().await;

    // The selector must arrive at inner_text byte-for-byte.
    assert!(
        calls.contains(&MockCall::InnerText(selector.to_string())),
        "selector with single-quote must pass through to inner_text unchanged: {calls:?}"
    );
    // No Evaluate call may contain the selector (interpolation is gone).
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Evaluate(s) if s.contains(selector))),
        "selector must not be interpolated into an Evaluate call: {calls:?}"
    );
}

#[tokio::test]
async fn get_text_selector_with_backslash_passes_through_unchanged() {
    // `a\\b` — a raw backslash (CSS escape introducer). The old JS-string
    // approach mis-handled this; element-handle extraction avoids any JS source.
    let selector = r"a\b";
    let (tool, backend, _rx) =
        mock_tool_with(MockBackend::new().with_text_result("backslash text"));

    let args = json!({ "action": "get_text", "selector": selector, "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "backslash text");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::InnerText(selector.to_string())),
        "selector with backslash must pass through to inner_text unchanged: {calls:?}"
    );
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Evaluate(s) if s.contains(selector))),
        "backslash selector must not be interpolated into an Evaluate call: {calls:?}"
    );
}

#[tokio::test]
async fn get_text_selector_with_newline_passes_through_unchanged() {
    // A newline inside the selector would break the single-line JS string
    // built by the old code, causing a syntax error in Chrome's JS engine.
    let selector = "div[data-x=\"line1\nline2\"]";
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("newline text"));

    let args = json!({ "action": "get_text", "selector": selector, "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "newline text");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::InnerText(selector.to_string())),
        "selector with newline must pass through to inner_text unchanged: {calls:?}"
    );
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Evaluate(s) if s.contains("line1"))),
        "newline selector must not be interpolated into an Evaluate call: {calls:?}"
    );
}

#[tokio::test]
async fn get_text_selector_with_css_escape_passes_through_unchanged() {
    // `#\31 23` — a CSS-escaped numeric id (ID starting with a digit).
    // `\31 ` is the CSS escape for `1`, making the selector `#123` after
    // browser parsing. The backslash-space sequence must survive intact.
    let selector = r"#\31 23";
    let (tool, backend, _rx) =
        mock_tool_with(MockBackend::new().with_text_result("escaped id text"));

    let args = json!({ "action": "get_text", "selector": selector, "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "escaped id text");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::InnerText(selector.to_string())),
        "CSS-escaped selector must pass through to inner_text unchanged: {calls:?}"
    );
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Evaluate(s) if s.contains(selector))),
        "CSS-escaped selector must not be interpolated into an Evaluate call: {calls:?}"
    );
}

#[tokio::test]
async fn get_text_with_selector_never_calls_evaluate() {
    // Regardless of the selector, get_text must NEVER call evaluate.
    // The old code always went through evaluate(format!(...selector...));
    // the fixed code goes through inner_text only.
    let selectors = &[
        "#simple",
        "input[name='user']",
        r"div\backslash",
        "p[data-v=\"a\nb\"]",
        r"#\31 23",
    ];

    for selector in selectors {
        let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("t"));

        let args = json!({ "action": "get_text", "selector": selector, "_session_id": "sess-a" });
        tool.call(&args.to_string()).await.unwrap();

        let calls = backend.calls();
        let calls = calls.lock().await;
        assert!(
            !calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
            "get_text with selector '{selector}' must NEVER call evaluate: {calls:?}"
        );
    }
}

#[tokio::test]
async fn get_text_without_selector_routes_through_body_text() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("hello body"));

    let args = json!({ "action": "get_text", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "hello body", "should return the mocked body_text");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::BodyText),
        "without a selector, get_text must route through body_text: {calls:?}"
    );
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::InnerText(_))),
        "without a selector, get_text must NOT route through inner_text: {calls:?}"
    );
}

#[tokio::test]
async fn execute_js_json_null_renders_as_null_string() {
    // Some(Value::Null) — a genuine JS `null` return — must serialize to "null",
    // distinct from the undefined/void path.
    let (tool, backend, _rx) =
        mock_tool_with(MockBackend::new().with_eval_result(Some(serde_json::Value::Null)));

    let args = json!({ "action": "execute_js", "script": "return null;", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "null", "JSON null must render as the string \"null\"");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Evaluate("return null;".to_string())),
        "execute_js must route through evaluate: {calls:?}"
    );
}

#[tokio::test]
async fn execute_js_undefined_renders_as_no_return_value() {
    // None — a void/`undefined` return — must reproduce the "(no return value)"
    // fallback, NOT "null".
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_eval_result(None));

    let args = json!({ "action": "execute_js", "script": "void 0;", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(
        out, "(no return value)",
        "undefined/void must render as the (no return value) fallback"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Evaluate("void 0;".to_string())),
        "execute_js must route through evaluate: {calls:?}"
    );
}

#[tokio::test]
async fn screenshot_with_selector_records_element_target() {
    let (tool, backend, rx) = mock_tool();
    // Channel ACCEPTS the delivery → the tool reports honest success.
    let media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "selector": "#hero", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("delivered to chat"),
        "screenshot should report honest delivery: {out}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    // A selector capture passes full_page through (false), but it is ignored.
    assert!(
        calls.contains(&MockCall::Screenshot(Some("#hero".to_string()), false)),
        "with a selector, screenshot must target the element: {calls:?}"
    );
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Screenshot(None, _))),
        "with a selector, screenshot must NOT capture the page: {calls:?}"
    );

    // The screenshot bytes were pushed onto the media channel as a Photo.
    let captured = media.messages.lock().await;
    assert_eq!(captured.len(), 1, "exactly one media message enqueued");
    assert!(
        matches!(captured[0].kind, MediaKind::Photo { .. }),
        "screenshot media must be a Photo"
    );
}

#[tokio::test]
async fn fill_result_contains_selector_but_not_value() {
    // Security: the fill result must NOT echo back the typed value (could be a
    // password or API token). The first 23 chars of the value are the specific
    // slice the old code returned via truncate_str — assert those are absent too.
    let (tool, _backend, _rx) = mock_tool();

    let value = "hunter2-super-secret-token-1234567890";
    let args = json!({ "action": "fill", "selector": "#password", "value": value, "_session_id": "sess-a" });
    let result = tool.call(&args.to_string()).await.unwrap();

    // The selector must appear in the result so callers know which field was filled.
    assert!(
        result.contains("#password"),
        "fill result must contain the selector: {result}"
    );

    // The value (or its 23-char prefix) must NOT appear in the result.
    let prefix_23 = crate::utils::truncate_str(value, 23);
    assert!(
        !result.contains(value),
        "fill result must not contain the full value: {result}"
    );
    assert!(
        !result.contains(prefix_23.as_str()),
        "fill result must not contain the 23-char truncated prefix of the value: {result}"
    );
}

// =============================================================================
// Task 2: fill replaces existing content (not appends)
// =============================================================================

/// Core regression: filling a field that already has "old@example.com" must
/// produce ONLY "new@example.com". The mock asserts dispatch-level semantics —
/// `action_fill` must call `replace_text`, not the old `click`+`type_text`
/// append sequence. True DOM-level replacement (that `this.value = ''` clears
/// the native input state) is covered by the ignored real-Chrome smoke suite
/// (Task 18, not in this slice).
#[tokio::test]
async fn fill_replaces_existing_value_not_appends() {
    let (tool, backend, _rx) = mock_tool();

    // The mock "field" conceptually starts with old@example.com; with replace
    // semantics the recorded call carries exactly the new value.
    let args = json!({
        "action": "fill",
        "selector": "#email",
        "value": "new@example.com",
        "_session_id": "sess-a"
    });
    let result = tool.call(&args.to_string()).await.unwrap();

    // Action should succeed and report the selector.
    assert!(
        result.contains("#email"),
        "fill result must contain selector: {result}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;

    // Must have recorded a ReplaceText with exactly the new value (replace
    // semantics). The old append sequence (Click + TypeText) must NOT appear.
    assert!(
        calls.contains(&MockCall::ReplaceText(
            "#email".to_string(),
            "new@example.com".to_string()
        )),
        "fill must dispatch replace_text with new@example.com: {calls:?}"
    );
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::TypeText(..))),
        "fill must NOT use type_text (append path): {calls:?}"
    );
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Click(..))),
        "fill must NOT use click (old append preamble): {calls:?}"
    );
}

/// Parametrized value round-trip: each value must appear unchanged in the
/// recorded ReplaceText call AND must NOT appear in the returned result string
/// (Task 1 secret-safety invariant holds for all value types).
#[tokio::test]
async fn fill_replace_value_cases_round_trip_and_stay_secret() {
    let cases: &[(&str, &str)] = &[
        ("#empty", ""),
        ("#unicode", "héllo wörld 日本語 🚀"),
        ("#multiline", "line1\nline2"),
        ("#password", "P@ssw0rd!'\"<script>"),
    ];

    for (selector, value) in cases {
        let (tool, backend, _rx) = mock_tool();

        let args = json!({ "action": "fill", "selector": selector, "value": value, "_session_id": "sess-a" });
        let result = tool.call(&args.to_string()).await.unwrap();

        // Result must contain selector (callers know which field was filled).
        assert!(
            result.contains(*selector),
            "fill result must contain selector '{selector}' for value '{value}': {result}"
        );

        // Result must NOT contain the value (secret-safety, Task 1 invariant).
        if !value.is_empty() {
            assert!(
                !result.contains(*value),
                "fill result must not echo value for selector '{selector}': {result}"
            );
        }

        let calls = backend.calls();
        let calls = calls.lock().await;

        // The value must round-trip unchanged into the ReplaceText record.
        assert!(
            calls.contains(&MockCall::ReplaceText(
                selector.to_string(),
                value.to_string()
            )),
            "replace_text call must carry exact value for selector '{selector}': {calls:?}"
        );
    }
}

#[tokio::test]
async fn screenshot_without_selector_defaults_to_viewport() {
    let (tool, backend, rx) = mock_tool();
    let media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("delivered to chat"),
        "screenshot should report honest delivery: {out}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    // DEFAULT (no full_page arg) is a VIEWPORT capture: full_page=false.
    assert!(
        calls.contains(&MockCall::Screenshot(None, false)),
        "without a selector or full_page, screenshot must capture the viewport: {calls:?}"
    );
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Screenshot(Some(_), _))),
        "without a selector, screenshot must NOT target an element: {calls:?}"
    );

    let captured = media.messages.lock().await;
    assert_eq!(captured.len(), 1, "exactly one media message enqueued");
    assert!(
        matches!(captured[0].kind, MediaKind::Photo { .. }),
        "screenshot media must be a Photo"
    );
}

/// `full_page: true` must pass `full_page=true` through to the backend.
#[tokio::test]
async fn screenshot_full_page_true_passes_through() {
    let (tool, backend, rx) = mock_tool();
    let _media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "full_page": true, "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(out.contains("delivered to chat"), "should succeed: {out}");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Screenshot(None, true)),
        "full_page:true must reach the backend as true: {calls:?}"
    );
}

/// HONEST-FAILURE regression (the confirmed bug): when the channel REJECTS the
/// delivery (e.g. Telegram PHOTO_INVALID_DIMENSIONS), the tool must NOT claim it
/// was sent — it returns an error stating the image was not delivered.
#[tokio::test]
async fn screenshot_delivery_failure_is_reported_honestly() {
    let (tool, _backend, rx) = mock_tool();
    // Channel REJECTS the delivery.
    let _media = spawn_media_responder(rx, Err("PHOTO_INVALID_DIMENSIONS".to_string()));

    let args = json!({ "action": "screenshot", "_session_id": "sess-a" });
    // The Tool::call wrapper surfaces an internal Err(String) as Ok("Error: ..").
    let msg = tool.call(&args.to_string()).await.unwrap();
    assert!(
        msg.starts_with("Error:"),
        "a rejected delivery must surface as an error: {msg}"
    );
    assert!(
        msg.contains("could NOT be delivered") || msg.contains("not sent"),
        "must honestly report non-delivery: {msg}"
    );
    assert!(
        !msg.contains("captured and delivered to chat"),
        "must NOT make a success claim: {msg}"
    );
    assert!(
        msg.contains("PHOTO_INVALID_DIMENSIONS"),
        "the channel's reason should be surfaced: {msg}"
    );
}

/// HONEST-SUCCESS: when the channel ACCEPTS the delivery, the tool reports a
/// success string and a Photo was enqueued.
#[tokio::test]
async fn screenshot_delivery_success_is_reported() {
    let (tool, _backend, rx) = mock_tool();
    let media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("captured and delivered to chat"),
        "successful delivery should be reported as delivered: {out}"
    );
    let captured = media.messages.lock().await;
    assert_eq!(captured.len(), 1, "a Photo must have been enqueued");
    assert!(matches!(captured[0].kind, MediaKind::Photo { .. }));
}

#[tokio::test]
async fn screenshot_call_with_status_outcome_includes_vision_attachment() {
    use crate::traits::{AttachmentProvenance, ToolCallOutcome};

    let (tool, _backend, rx) = mock_tool();
    let _media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "_session_id": "sess-a" });
    let outcome: ToolCallOutcome = tool
        .call_with_status_outcome(&args.to_string(), None)
        .await
        .unwrap();
    assert_eq!(outcome.metadata.attachments.len(), 1);
    let attachment = &outcome.metadata.attachments[0];
    assert_eq!(attachment.provenance, AttachmentProvenance::ToolObservation);
    assert_eq!(attachment.source_tool.as_deref(), Some("browser"));
    assert!(std::path::Path::new(&attachment.local_path).exists());
}

/// OVERSIZED (over photo caps, under doc cap): delivered as a DOCUMENT using the
/// persisted inbox copy (also used for agent vision context).
#[tokio::test]
async fn screenshot_oversized_delivered_as_document() {
    let oversized = png_header(8000, 8000);
    let backend = MockBackend::new().with_screenshot_bytes(oversized);
    let (tool, _backend, rx) = mock_tool_with(backend);
    let media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "full_page": true, "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("delivered to chat as a file"),
        "oversized capture should report delivery as a file: {out}"
    );
    assert!(
        out.contains("Saved to:"),
        "screenshot should report inbox path for vision: {out}"
    );

    let captured = media.messages.lock().await;
    assert_eq!(captured.len(), 1, "exactly one media message enqueued");
    let (path, filename) = match &captured[0].kind {
        MediaKind::Document {
            file_path,
            filename,
        } => (file_path.clone(), filename.clone()),
        _ => panic!("oversized capture must be enqueued as a Document"),
    };
    assert_eq!(
        filename, "screenshot.png",
        "filename should be screenshot.png"
    );
    assert!(
        std::path::Path::new(&path).exists(),
        "inbox screenshot file must persist for vision context: {path}"
    );
}

/// DOCUMENT DELIVERY FAILURE: channel rejects delivery; inbox copy still saved for vision.
#[tokio::test]
async fn screenshot_document_delivery_failure_is_reported_and_inbox_persists() {
    let oversized = png_header(8000, 8000);
    let backend = MockBackend::new().with_screenshot_bytes(oversized);
    let (tool, _backend, rx) = mock_tool_with(backend);
    let media = spawn_media_responder(rx, Err("file too large".to_string()));

    let args = json!({ "action": "screenshot", "full_page": true, "_session_id": "sess-a" });
    let msg = tool.call(&args.to_string()).await.unwrap();
    assert!(
        msg.starts_with("Error:"),
        "a rejected document delivery must surface as an error: {msg}"
    );
    assert!(
        msg.contains("could NOT be delivered") || msg.contains("not sent"),
        "must honestly report non-delivery: {msg}"
    );

    let captured = media.messages.lock().await;
    assert_eq!(captured.len(), 1);
    let path = match &captured[0].kind {
        MediaKind::Document { file_path, .. } => file_path.clone(),
        _ => panic!("must be a Document"),
    };
    assert!(
        std::path::Path::new(&path).exists(),
        "inbox screenshot file must persist even when chat delivery fails: {path}"
    );
}

/// Unit test for the pure size-decision helper. Exercises all three branches
/// (Photo / Document / TooLarge) cheaply with byte-length numbers — no real
/// 49MB allocation. `oversize_reason` mirrors `screenshot_oversize_reason`'s
/// Option output: `None` = within photo caps, `Some(_)` = over photo caps.
#[test]
fn screenshot_delivery_kind_decides_by_size() {
    use super::ScreenshotDelivery;

    // Within photo caps (no oversize reason) → Photo, regardless of byte length.
    assert_eq!(
        super::screenshot_delivery_kind(1_000, None),
        ScreenshotDelivery::Photo
    );

    // Over photo caps but under the document cap → Document.
    assert_eq!(
        super::screenshot_delivery_kind(1_000, Some("dimensions 8000x8000".to_string())),
        ScreenshotDelivery::Document
    );
    assert_eq!(
        super::screenshot_delivery_kind(
            super::MAX_SCREENSHOT_DOCUMENT_BYTES,
            Some("dimensions 8000x8000".to_string())
        ),
        ScreenshotDelivery::Document
    );

    // Over photo caps AND over the document cap → TooLarge.
    assert_eq!(
        super::screenshot_delivery_kind(
            super::MAX_SCREENSHOT_DOCUMENT_BYTES + 1,
            Some("dimensions 8000x8000".to_string())
        ),
        ScreenshotDelivery::TooLarge
    );
}

/// MISSING SESSION ID: a screenshot with an empty `_session_id` is refused
/// before any capture or enqueue.
#[tokio::test]
async fn screenshot_empty_session_id_refused() {
    let (tool, backend, rx) = mock_tool();
    let media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "_session_id": "" });
    let msg = tool.call(&args.to_string()).await.unwrap();
    assert!(
        msg.starts_with("Error:") && msg.contains("session id"),
        "empty session id must be refused: {msg}"
    );
    // No capture and no enqueue happened.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Screenshot(..))),
        "must refuse before capturing: {calls:?}"
    );
    let captured = media.messages.lock().await;
    assert!(captured.is_empty(), "must enqueue nothing");
}

/// URL REDACTION: a page URL carrying a query token + fragment must have BOTH
/// stripped from the caption and the result, while host+path survive.
#[tokio::test]
async fn screenshot_caption_redacts_query_and_fragment() {
    let backend =
        MockBackend::new().with_url("https://host.example/dash/board?token=SECRET#section");
    let (tool, _backend, rx) = mock_tool_with(backend);
    let media = spawn_media_responder(rx, Ok(()));

    let args = json!({ "action": "screenshot", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();

    // Result string: no token, no fragment; host + path present.
    assert!(
        !out.contains("SECRET"),
        "result must not leak the token: {out}"
    );
    assert!(
        !out.contains("section"),
        "result must not leak the fragment: {out}"
    );
    assert!(
        out.contains("host.example/dash/board"),
        "host + path should be present: {out}"
    );

    // Caption (what reaches the channel) is likewise redacted.
    let captured = media.messages.lock().await;
    assert_eq!(captured.len(), 1);
    let cap = &captured[0].caption;
    assert!(
        !cap.contains("SECRET"),
        "caption must not leak the token: {cap}"
    );
    assert!(
        !cap.contains("section"),
        "caption must not leak the fragment: {cap}"
    );
    assert!(
        cap.contains("host.example/dash/board"),
        "caption keeps host + path: {cap}"
    );
}

/// Unit test for the dependency-free PNG dimension parser.
#[test]
fn png_dimensions_parses_valid_header_and_rejects_others() {
    // Valid PNG header encoding 1280x720.
    let bytes = png_header(1280, 720);
    assert_eq!(super::png_dimensions(&bytes), Some((1280, 720)));

    // Not a PNG (wrong signature).
    assert_eq!(super::png_dimensions(&[0x00, 0x01, 0x02, 0x03]), None);

    // Too short to contain an IHDR.
    assert_eq!(
        super::png_dimensions(&[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        None
    );
}

/// Build a minimal valid PNG header (8-byte signature + an IHDR whose width and
/// height fields encode `w`/`h`). Only the first 24 bytes matter for
/// `png_dimensions`; the rest of a real PNG is irrelevant to the dimension parse.
fn png_header(w: u32, h: u32) -> Vec<u8> {
    let mut bytes = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    // IHDR chunk length (13) + type "IHDR" (bytes [8..16]); contents unimportant
    // here except width@[16..20] and height@[20..24].
    bytes.extend_from_slice(&13u32.to_be_bytes());
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&w.to_be_bytes());
    bytes.extend_from_slice(&h.to_be_bytes());
    bytes
}

// =============================================================================
// Task 3: session-scoped page state
// =============================================================================

/// Helper: collect the synthetic page ids minted by every `create_page` call.
async fn create_page_ids(backend: &MockBackend) -> Vec<String> {
    let calls = backend.calls();
    let calls = calls.lock().await;
    calls
        .iter()
        .filter_map(|c| match c {
            MockCall::CreatePage(id) => Some(id.clone()),
            _ => None,
        })
        .collect()
}

/// Two different sessions must each get their OWN page (distinct `create_page`
/// calls / ids), and a session's subsequent action must REUSE its page rather
/// than mint a new one. This is the core isolation guarantee: session B's
/// actions never operate on session A's tab.
#[tokio::test]
async fn two_sessions_get_distinct_pages() {
    let (tool, backend, _rx) = mock_tool();

    // Session A navigates.
    let a = json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" });
    tool.call(&a.to_string()).await.unwrap();

    // Session B navigates.
    let b = json!({ "action": "navigate", "url": "https://b.example/", "_session_id": "sess-b" });
    tool.call(&b.to_string()).await.unwrap();

    // Exactly two create_page calls so far, with DISTINCT ids — the two sessions
    // hold different pages.
    let ids = create_page_ids(&backend).await;
    assert_eq!(
        ids.len(),
        2,
        "two distinct sessions must each create a page: {ids:?}"
    );
    assert_ne!(
        ids[0], ids[1],
        "the two sessions must hold different page ids: {ids:?}"
    );

    // A SECOND action on session A must REUSE A's page — no new create_page.
    let a2 = json!({ "action": "get_text", "_session_id": "sess-a" });
    tool.call(&a2.to_string()).await.unwrap();

    let ids_after = create_page_ids(&backend).await;
    assert_eq!(
        ids_after.len(),
        2,
        "session A's second action must reuse its page (no new create_page): {ids_after:?}"
    );
}

/// An empty (or missing) `_session_id` must be rejected BEFORE the browser is
/// launched: no `ensure_ready`, no `create_page`. Proven via the mock's call log
/// being empty.
#[tokio::test]
async fn empty_session_id_rejected_before_launch() {
    // Explicit empty session id.
    let (tool, backend, _rx) = mock_tool();
    let args = json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.to_lowercase().contains("session id"),
        "empty session id must produce a session-id error: {out}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.is_empty(),
        "no browser launch (ensure_ready/create_page) may happen for an empty session id: {calls:?}"
    );
    drop(calls);

    // Missing _session_id altogether (defaults to "" in dispatch) — same outcome.
    let (tool2, backend2, _rx2) = mock_tool();
    let missing = json!({ "action": "navigate", "url": "https://a.example/" });
    let out2 = tool2.call(&missing.to_string()).await.unwrap();
    assert!(
        out2.to_lowercase().contains("session id"),
        "missing session id must produce a session-id error: {out2}"
    );
    assert!(
        backend2.calls().lock().await.is_empty(),
        "no browser launch may happen for a missing session id"
    );
}

/// Two sessions can both obtain their pages and then reuse them across several
/// sequential actions without deadlock or cross-contamination. True parallel
/// execution (different sessions running concurrently while one holds its action
/// lock) is exercised by integration; the per-session action lock is a distinct
/// `Arc<Mutex<()>>` per session, so distinct sessions never block on each other.
#[tokio::test]
async fn two_sessions_reuse_pages_without_deadlock() {
    let (tool, backend, _rx) = mock_tool();

    for _ in 0..3 {
        let a = json!({ "action": "get_text", "_session_id": "sess-a" });
        tool.call(&a.to_string()).await.unwrap();
        let b = json!({ "action": "get_text", "_session_id": "sess-b" });
        tool.call(&b.to_string()).await.unwrap();
    }

    // Despite six actions, only two pages were ever created (one per session).
    let ids = create_page_ids(&backend).await;
    assert_eq!(
        ids.len(),
        2,
        "each session creates exactly one page across repeated actions: {ids:?}"
    );
    assert_ne!(
        ids[0], ids[1],
        "the two sessions' pages must differ: {ids:?}"
    );
}

// =============================================================================
// Task 5: explicit tab management
// =============================================================================

/// Extract the opaque tab id reported by a `new_tab` / `Opened new tab <id>`
/// result, or by the active line of a list_tabs result. Used to drive
/// subsequent switch/close actions without hardcoding ids.
fn tab_id_from_opened(result: &str) -> String {
    // "Opened new tab <id> ..." — the id is the 4th whitespace token.
    result
        .split_whitespace()
        .nth(3)
        .expect("opened-tab result must contain an id")
        .to_string()
}

/// `list_tabs` reports the session's tabs with opaque ids + titles + REDACTED
/// origins. A path/query in a tab's URL must NOT leak into the output.
#[tokio::test]
async fn list_tabs_reports_tabs_with_redacted_origins() {
    // The mock's pages report a URL with a path + secret query string. new_tab
    // snapshots this into the session, so list_tabs must redact it to the origin.
    let secret = "https://app.example/dashboard?session_token=SECRET123&reset=abc";
    let (tool, _backend, _rx) = mock_tool_with(MockBackend::new().with_url(secret));

    // Open the session's first tab via navigate.
    let nav =
        json!({ "action": "navigate", "url": "https://first.example/", "_session_id": "sess-a" });
    tool.call(&nav.to_string()).await.unwrap();

    // new_tab — its page reports the secret url; only the origin must be shown.
    let nt = json!({ "action": "new_tab", "_session_id": "sess-a" });
    tool.call(&nt.to_string()).await.unwrap();

    let out = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    // Two tabs, with the active marker present.
    assert!(out.contains("Open tabs (2)"), "expected two tabs: {out}");
    assert!(
        out.contains("[active]"),
        "an active tab must be marked: {out}"
    );

    // CRITICAL: no path/query leak. The secret token and the path must be absent.
    assert!(
        !out.contains("session_token"),
        "list_tabs must redact query strings (secret leaked): {out}"
    );
    assert!(
        !out.contains("SECRET123"),
        "list_tabs must not surface the secret token: {out}"
    );
    assert!(
        !out.contains("/dashboard"),
        "list_tabs must redact the path: {out}"
    );
}

/// `new_tab` adds a tab AND makes it active, so a subsequent action operates on
/// the new tab's page (a distinct page id from the original).
#[tokio::test]
async fn new_tab_adds_and_activates_tab() {
    let (tool, backend, _rx) = mock_tool();

    // First tab via navigate.
    let nav = json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" });
    tool.call(&nav.to_string()).await.unwrap();

    let ids_before = create_page_ids(&backend).await;
    assert_eq!(ids_before.len(), 1, "one page so far: {ids_before:?}");

    // Open a second tab.
    let nt = json!({ "action": "new_tab", "_session_id": "sess-a" });
    let out = tool.call(&nt.to_string()).await.unwrap();
    assert!(
        out.contains("Opened new tab"),
        "new_tab should report id: {out}"
    );

    let ids_after = create_page_ids(&backend).await;
    assert_eq!(
        ids_after.len(),
        2,
        "new_tab must create a second page: {ids_after:?}"
    );

    // list_tabs shows two tabs with the second active (new_tab activates it).
    let list = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(list.contains("Open tabs (2)"), "two tabs expected: {list}");
    // The active tab id must be the newly opened one (last created page id).
    let new_id = &ids_after[1];
    let active_line = list
        .lines()
        .find(|l| l.contains("[active]"))
        .expect("an active tab line must exist");
    assert!(
        active_line.contains(new_id.as_str()),
        "the newly opened tab must be active: active='{active_line}' new_id='{new_id}'"
    );
}

/// Regression for `target=_blank`: a click that spawns a popup must report the
/// new tab's id, and `list_tabs` must then show BOTH tabs — the new tab is
/// discoverable and switchable, never silently left behind.
#[tokio::test]
async fn popup_after_click_is_discoverable() {
    let (tool, backend, _rx) = mock_tool();

    // Establish the session's first tab.
    let nav = json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" });
    tool.call(&nav.to_string()).await.unwrap();

    // The session's active tab is the page minted by navigate — it is the
    // legitimate opener for a popup this session spawns.
    let opener_id = create_page_ids(&backend).await[0].clone();

    // Script a popup that will appear the next time list_targets runs (i.e. the
    // diff after the click). Its opener is the clicking session's active tab, so
    // it is attributed to this session. Give it a URL with a secret path/query
    // to also prove redaction flows through to list_tabs.
    backend
        .script_popup_with_opener(
            "popup-target-xyz",
            "Popup Page",
            "https://popup.example/oauth?code=TOPSECRET",
            Some(&opener_id),
        )
        .await;

    // Click — should detect the popup and report its tab id.
    let out = tool
        .call(
            &json!({ "action": "click", "selector": "#open", "_session_id": "sess-a" }).to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.contains("opened new tab") && out.contains("popup-target-xyz"),
        "click must report the spawned popup's tab id: {out}"
    );

    // list_tabs now shows BOTH the original and the popup tab.
    let list = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(
        list.contains("Open tabs (2)"),
        "both tabs must be listed: {list}"
    );
    assert!(
        list.contains("popup-target-xyz"),
        "the popup tab must be discoverable via list_tabs: {list}"
    );
    // Redaction still holds for the popup's url.
    assert!(
        !list.contains("TOPSECRET") && !list.contains("/oauth"),
        "popup url must be redacted to origin: {list}"
    );

    // The original tab stays active (popups are registered, not auto-activated).
    let active_line = list
        .lines()
        .find(|l| l.contains("[active]"))
        .expect("an active tab line must exist");
    assert!(
        !active_line.contains("popup-target-xyz"),
        "popup must NOT be auto-activated: {active_line}"
    );
}

/// Cross-session security: a net-new target whose CDP `openerId` is NOT the
/// clicking session's active tab (it belongs to another session, or was opened
/// independently) must NOT be attributed to the clicking session. Otherwise,
/// under concurrent timing, session A's click could capture session B's
/// freshly-opened tab and then read/switch its page — a cross-session leak.
#[tokio::test]
async fn popup_with_foreign_opener_is_not_attributed() {
    let (tool, backend, _rx) = mock_tool();

    // Session A establishes its first tab.
    let nav = json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" });
    tool.call(&nav.to_string()).await.unwrap();

    // Script a net-new target whose opener is some OTHER target (not A's active
    // tab) — modeling a tab spawned by a different session or independently.
    backend
        .script_popup_with_opener(
            "foreign-target-zzz",
            "Foreign Page",
            "https://foreign.example/secret",
            Some("some-other-sessions-target"),
        )
        .await;

    // A clicks. The new target appears in the global set, but its opener is not
    // A's active tab, so the click must NOT report a new tab.
    let out = tool
        .call(
            &json!({ "action": "click", "selector": "#open", "_session_id": "sess-a" }).to_string(),
        )
        .await
        .unwrap();
    assert!(
        !out.contains("opened new tab") && !out.contains("foreign-target-zzz"),
        "a foreign-opener target must not be reported as A's popup: {out}"
    );
    assert_eq!(
        out, "Clicked element '#open'",
        "click should report a plain click when no popup is attributed: {out}"
    );

    // list_tabs for session A must NOT include the foreign target — only A's
    // original tab remains.
    let list = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(
        list.contains("Open tabs (1)"),
        "session A must still have exactly one tab: {list}"
    );
    assert!(
        !list.contains("foreign-target-zzz"),
        "the foreign-opener target must NOT be discoverable in A's session: {list}"
    );
}

/// `switch_tab` to a valid tab changes the active page; a subsequent action then
/// runs on the switched-to tab.
#[tokio::test]
async fn switch_tab_changes_active_page() {
    let (tool, backend, _rx) = mock_tool();

    // First tab.
    let nav = json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" });
    tool.call(&nav.to_string()).await.unwrap();
    let first_id = create_page_ids(&backend).await[0].clone();

    // Second tab (now active).
    tool.call(&json!({ "action": "new_tab", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    // Switch back to the first tab by its id.
    let out = tool
        .call(
            &json!({ "action": "switch_tab", "tab_id": first_id, "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.contains(&format!("Switched to tab {first_id}")),
        "switch_tab must confirm the active tab: {out}"
    );

    // list_tabs now marks the first tab active again.
    let list = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    let active_line = list
        .lines()
        .find(|l| l.contains("[active]"))
        .expect("an active tab line must exist");
    assert!(
        active_line.contains(first_id.as_str()),
        "after switch, the first tab must be active: {active_line}"
    );
}

/// Cross-session safety: a session must NOT be able to switch to or close
/// another session's tab. An unknown id, and another session's real id, both
/// error.
#[tokio::test]
async fn switch_and_close_reject_unknown_and_cross_session_tabs() {
    let (tool, backend, _rx) = mock_tool();

    // Session A has one tab.
    tool.call(
        &json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    let a_tab = create_page_ids(&backend).await[0].clone();

    // Session B has one tab — capture B's real tab id.
    tool.call(
        &json!({ "action": "navigate", "url": "https://b.example/", "_session_id": "sess-b" })
            .to_string(),
    )
    .await
    .unwrap();
    let b_tab = create_page_ids(&backend).await[1].clone();
    assert_ne!(a_tab, b_tab, "the two sessions must hold different tab ids");

    // A switching to a totally unknown id errors.
    let out = tool
        .call(
            &json!({ "action": "switch_tab", "tab_id": "no-such-tab", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.starts_with("Error:") && out.to_lowercase().contains("unknown tab"),
        "switch to unknown tab must error: {out}"
    );

    // A switching to B's REAL tab id must be rejected (cross-session).
    let out = tool
        .call(
            &json!({ "action": "switch_tab", "tab_id": b_tab, "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.starts_with("Error:") && out.to_lowercase().contains("does not belong"),
        "A must not switch to B's tab: {out}"
    );

    // A closing B's REAL tab id must be rejected too.
    let out = tool
        .call(
            &json!({ "action": "close_tab", "tab_id": b_tab, "_session_id": "sess-a" }).to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.starts_with("Error:") && out.to_lowercase().contains("does not belong"),
        "A must not close B's tab: {out}"
    );

    // Sanity: B can still switch to its own tab.
    let out = tool
        .call(
            &json!({ "action": "switch_tab", "tab_id": b_tab, "_session_id": "sess-b" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.contains("Switched to tab"),
        "B must be able to switch to its own tab: {out}"
    );
}

/// `close_tab` removes the tab and reports the new active tab; closing the
/// active tab promotes a remaining one.
#[tokio::test]
async fn close_tab_removes_and_reports_new_active() {
    let (tool, _backend, _rx) = mock_tool();

    // First tab + a second (active) tab.
    tool.call(
        &json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    let opened = tool
        .call(&json!({ "action": "new_tab", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    let second_id = tab_id_from_opened(&opened);

    // Close the active (second) tab — the first must become active again.
    let out = tool
        .call(
            &json!({ "action": "close_tab", "tab_id": second_id, "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.contains(&format!("Closed tab {second_id}")) && out.contains("Active tab is now"),
        "close_tab must report removal and the new active tab: {out}"
    );

    // Only one tab remains.
    let list = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(
        list.contains("Open tabs (1)"),
        "one tab must remain: {list}"
    );
    assert!(
        !list.contains(second_id.as_str()),
        "the closed tab must be gone: {list}"
    );
}

/// `redact_origin` must strip embedded userinfo (credentials), the path, query,
/// and fragment — leaving only `scheme://host[:port]`. Anything in those parts
/// can be a secret and must never reach a tab listing.
#[test]
fn redact_origin_strips_credentials_path_and_query() {
    let redacted = super::redact_origin("https://user:s3cr3t@example.com/admin?token=ABC");

    // The credential, query key, query value, and path must all be gone.
    assert!(
        !redacted.contains("s3cr3t"),
        "embedded password must be stripped: {redacted}"
    );
    assert!(
        !redacted.contains("user"),
        "embedded username must be stripped: {redacted}"
    );
    assert!(
        !redacted.contains("token"),
        "query key must be stripped: {redacted}"
    );
    assert!(
        !redacted.contains("ABC"),
        "query value must be stripped: {redacted}"
    );
    assert!(
        !redacted.contains("/admin"),
        "path must be stripped: {redacted}"
    );
    // The host must survive so the listing stays useful.
    assert!(
        redacted.contains("example.com"),
        "host must be preserved: {redacted}"
    );
    assert_eq!(
        redacted, "https://example.com",
        "exact origin must be scheme://host with nothing else: {redacted}"
    );

    // Schemeless host/path: the path must still be stripped.
    let schemeless = super::redact_origin("host.com/path?x=secret");
    assert_eq!(
        schemeless, "host.com",
        "schemeless input must strip path/query: {schemeless}"
    );
}

/// Existing single-page actions still operate on the active tab after the
/// multi-tab refactor (the active tab IS the page).
#[tokio::test]
async fn single_page_actions_use_active_tab() {
    let (tool, backend, _rx) =
        mock_tool_with(MockBackend::new().with_text_result("active tab text"));

    // Navigate establishes the active tab.
    tool.call(
        &json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();

    // get_text routes through the active tab's page.
    let out = tool
        .call(&json!({ "action": "get_text", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert_eq!(out, "active tab text");

    // Still exactly one page (no spurious tab creation).
    let ids = create_page_ids(&backend).await;
    assert_eq!(
        ids.len(),
        1,
        "single-page flow must not create extra tabs: {ids:?}"
    );
}

// =============================================================================
// Task 7: request approval at execution time
// =============================================================================

/// Captures every `ApprovalRequest.command` the gate sends, so tests can count
/// prompts and assert prompt content (secret-exclusion). Backed by a spawned
/// task that drains the broker channel and replies with a scripted response.
#[derive(Clone)]
struct ApprovalRecorder {
    commands: Arc<Mutex<Vec<String>>>,
}

impl ApprovalRecorder {
    async fn commands(&self) -> Vec<String> {
        self.commands.lock().await.clone()
    }

    async fn count(&self) -> usize {
        self.commands.lock().await.len()
    }
}

/// Spawn a responder that replies to every approval request with `reply`,
/// recording the prompt `command` of each. Returns the broker (to inject into
/// the tool) and the recorder (to assert on).
fn spawn_responder(reply: ApprovalResponse) -> (ApprovalBroker, ApprovalRecorder) {
    let (tx, mut rx) = mpsc::channel(8);
    let broker = ApprovalBroker::new(tx);
    let recorder = ApprovalRecorder {
        commands: Arc::new(Mutex::new(Vec::new())),
    };
    let commands = recorder.commands.clone();
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            commands.lock().await.push(req.command.clone());
            let _ = req.response_tx.send(reply.clone());
        }
    });
    (broker, recorder)
}

/// Spawn a responder that NEVER replies (drops the oneshot sender), but still
/// records each request — used for the timeout path.
fn spawn_silent_responder() -> (ApprovalBroker, ApprovalRecorder) {
    let (tx, mut rx) = mpsc::channel(8);
    let broker = ApprovalBroker::new(tx);
    let recorder = ApprovalRecorder {
        commands: Arc::new(Mutex::new(Vec::new())),
    };
    let commands = recorder.commands.clone();
    tokio::spawn(async move {
        while let Some(req) = rx.recv().await {
            commands.lock().await.push(req.command.clone());
            // Drop req (and its response_tx) without replying → the gate times out.
            drop(req);
        }
    });
    (broker, recorder)
}

/// Captures the `MediaMessage`s a media responder drained, so a test can assert
/// what (if anything) was enqueued and how many.
#[derive(Clone)]
struct MediaRecorder {
    messages: Arc<Mutex<Vec<CapturedMedia>>>,
}

/// A `MediaMessage` with its `result_tx` already consumed (replied to). Carries
/// only the inspectable fields a test asserts on.
struct CapturedMedia {
    #[allow(dead_code)]
    session_id: String,
    caption: String,
    kind: MediaKind,
}

/// Spawn a media responder that drains `media_rx`, records each `MediaMessage`,
/// and replies on its `result_tx` with the given outcome — modeling the channel
/// either accepting (`Ok`) or rejecting (`Err`) the delivery. Mirrors the
/// approval-responder pattern. Returns a recorder for post-hoc assertions.
fn spawn_media_responder(
    mut rx: mpsc::Receiver<MediaMessage>,
    outcome: Result<(), String>,
) -> MediaRecorder {
    let recorder = MediaRecorder {
        messages: Arc::new(Mutex::new(Vec::new())),
    };
    let messages = recorder.messages.clone();
    tokio::spawn(async move {
        while let Some(mut msg) = rx.recv().await {
            let result_tx = msg.result_tx.take();
            messages.lock().await.push(CapturedMedia {
                session_id: msg.session_id.clone(),
                caption: msg.caption.clone(),
                kind: msg.kind,
            });
            if let Some(result_tx) = result_tx {
                let _ = result_tx.send(outcome.clone());
            }
        }
    });
    recorder
}

fn approving_tool(
    backend: MockBackend,
    reply: ApprovalResponse,
) -> (BrowserTool, Arc<MockBackend>, ApprovalRecorder) {
    let backend = Arc::new(backend);
    let (media_tx, _media_rx) = mpsc::channel(8);
    let (broker, recorder) = spawn_responder(reply);
    let tool = BrowserTool::with_backend_and_approval(
        backend.clone(),
        media_tx,
        broker,
        Duration::from_secs(5),
    );
    (tool, backend, recorder)
}

async fn calls_contains(backend: &MockBackend, needle: &MockCall) -> bool {
    backend.calls().lock().await.contains(needle)
}

/// ALLOW: an approving responder lets a navigate reach the backend and succeed.
#[tokio::test]
async fn approval_allow_lets_navigate_reach_backend() {
    let (tool, backend, recorder) =
        approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    let out = tool
        .call(
            &json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();

    assert_eq!(out, "Navigated to https://example.com/");
    assert!(
        calls_contains(
            &backend,
            &MockCall::Goto("https://example.com/".to_string())
        )
        .await,
        "an approved navigate must reach the backend goto"
    );
    assert_eq!(
        recorder.count().await,
        1,
        "navigate must prompt exactly once"
    );
}

/// DENY: a denying responder blocks the navigation — the backend goto is never
/// recorded and the result reports denial.
#[tokio::test]
async fn approval_deny_blocks_navigation_before_backend() {
    let (tool, backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::Deny);

    let out = tool
        .call(
            &json!({ "action": "navigate", "url": "https://evil.example/", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();

    assert!(
        out.to_lowercase().contains("denied"),
        "denied navigate must report denial: {out}"
    );
    // The denied action must NOT have touched the backend at all.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(
            c,
            MockCall::Goto(_) | MockCall::CreatePage(_) | MockCall::EnsureReady
        )),
        "a denied navigation must never reach the backend: {calls:?}"
    );
}

/// DENY on a consequential click: the backend click is never recorded.
#[tokio::test]
async fn approval_deny_blocks_consequential_click_before_backend() {
    let (tool, backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::Deny);

    // "#delete" → consequential (point-of-action), so it always prompts.
    let out = tool
        .call(
            &json!({ "action": "click", "selector": "#delete", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();

    assert!(
        out.to_lowercase().contains("denied"),
        "denied click must report denial: {out}"
    );
    assert!(
        !calls_contains(&backend, &MockCall::Click("#delete".to_string())).await,
        "a denied click must never reach the backend"
    );
}

/// TIMEOUT: a responder that never replies → the gate auto-denies fast, with no
/// backend call.
#[tokio::test]
async fn approval_timeout_denies_without_backend() {
    let backend = Arc::new(MockBackend::new());
    let (media_tx, _media_rx) = mpsc::channel(8);
    let (broker, recorder) = spawn_silent_responder();
    // ~50ms timeout so the test runs fast.
    let tool = BrowserTool::with_backend_and_approval(
        backend.clone(),
        media_tx,
        broker,
        Duration::from_millis(50),
    );

    let start = std::time::Instant::now();
    let out = tool
        .call(
            &json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    let elapsed = start.elapsed();

    assert!(
        out.to_lowercase().contains("denied"),
        "timed-out approval must deny: {out}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "timeout path must resolve quickly, took {elapsed:?}"
    );
    assert_eq!(
        recorder.count().await,
        1,
        "the request was sent (and recorded) before timing out"
    );
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Goto(_))),
        "a timed-out navigation must never reach the backend: {calls:?}"
    );
}

/// MISSING CHANNEL: with no broker, a mutation is denied with no backend call,
/// while an observation still runs.
#[tokio::test]
async fn missing_channel_denies_mutation_but_allows_observation() {
    // `no_channel_tool` constructs the tool with NO approval channel.
    let (tool, backend, _rx) = no_channel_tool(MockBackend::new());

    // A navigation requires approval → fail safe to Deny, no backend.
    let nav = tool
        .call(
            &json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        nav.to_lowercase().contains("approval") || nav.to_lowercase().contains("denied"),
        "navigate must be denied with no channel: {nav}"
    );
    assert!(
        backend.calls().lock().await.is_empty(),
        "a denied navigation must not touch the backend with no channel"
    );

    // An observation (get_text) is free — it runs without a channel.
    let obs = tool
        .call(&json!({ "action": "get_text", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(
        !obs.to_lowercase().contains("denied"),
        "an observation must not be denied with no channel: {obs}"
    );
    assert!(
        calls_contains(&backend, &MockCall::BodyText).await,
        "the observation must have reached the backend"
    );
}

/// OBSERVATION-FREE: even with a responder that would DENY if asked, an
/// observation runs WITHOUT sending any approval request.
#[tokio::test]
async fn observations_never_prompt() {
    let (tool, backend, recorder) = approving_tool(MockBackend::new(), ApprovalResponse::Deny);

    for action in &["get_text", "screenshot", "list_tabs"] {
        let out = tool
            .call(&json!({ "action": action, "_session_id": "sess-obs" }).to_string())
            .await
            .unwrap();
        assert!(
            !out.to_lowercase().contains("denied"),
            "observation '{action}' must not be denied: {out}"
        );
    }

    assert_eq!(
        recorder.count().await,
        0,
        "observations must never send an approval request: {:?}",
        recorder.commands().await
    );
    // And they reached the backend (proof they actually ran).
    assert!(
        calls_contains(&backend, &MockCall::BodyText).await,
        "get_text observation must have run"
    );
}

/// SESSION-LEVEL REUSE: AllowSession on the first navigation suppresses the
/// prompt for a SECOND navigation in the same session (one prompt total), yet
/// both reach the backend.
#[tokio::test]
async fn session_approval_suppresses_second_navigation_prompt() {
    let (tool, backend, recorder) =
        approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    tool.call(
        &json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    assert_eq!(recorder.count().await, 1, "first navigation must prompt");

    tool.call(
        &json!({ "action": "navigate", "url": "https://b.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    assert_eq!(
        recorder.count().await,
        1,
        "second navigation in an approved session must NOT prompt again"
    );

    // Both navigations reached the backend.
    assert!(
        calls_contains(&backend, &MockCall::Goto("https://a.example/".to_string())).await
            && calls_contains(&backend, &MockCall::Goto("https://b.example/".to_string())).await,
        "both navigations must reach the backend"
    );
}

/// POINT-OF-ACTION: every `execute_js` call prompts even after AllowSession —
/// session approval must NOT suppress execute_js prompts. Also proves the prompt
/// never contains the script body.
#[tokio::test]
async fn execute_js_prompts_every_call_and_hides_script() {
    let (tool, backend, recorder) =
        approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    let sentinel = "SECRET_IN_SCRIPT";
    let script = format!("var token = '{sentinel}'; return token;");

    for _ in 0..2 {
        tool.call(
            &json!({ "action": "execute_js", "script": script, "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        recorder.count().await,
        2,
        "each execute_js must prompt, even after AllowSession: {:?}",
        recorder.commands().await
    );

    // The prompt must NOT leak the script body / sentinel.
    for cmd in recorder.commands().await {
        assert!(
            !cmd.contains(sentinel),
            "execute_js prompt must not contain the script body: {cmd}"
        );
    }

    // Both evaluations reached the backend (they were each approved).
    assert!(
        calls_contains(&backend, &MockCall::Evaluate(script.clone())).await,
        "approved execute_js must reach the backend"
    );
}

/// POINT-OF-ACTION: a consequential click prompts even after a prior ordinary
/// (navigation) AllowSession marked the session approved.
#[tokio::test]
async fn consequential_click_prompts_despite_session_approval() {
    let (tool, _backend, recorder) =
        approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    // Ordinary navigation marks the session approved (one prompt).
    tool.call(
        &json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    assert_eq!(recorder.count().await, 1);

    // A consequential click ("#delete") must still prompt despite the approved
    // session.
    tool.call(
        &json!({ "action": "click", "selector": "#delete", "_session_id": "sess-a" }).to_string(),
    )
    .await
    .unwrap();
    assert_eq!(
        recorder.count().await,
        2,
        "consequential click must prompt even in an approved session"
    );
}

/// SECRET-SAFE PROMPT: a consequential fill prompt must not leak the typed value.
#[tokio::test]
async fn consequential_fill_prompt_hides_value() {
    let (tool, _backend, recorder) =
        approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    // "#submit-password" selector is opaque/hyphenated (not consequential), so
    // force point-of-action via a standalone consequential token in the selector.
    let secret_value = "hunter2-super-secret";
    tool.call(
        &json!({
            "action": "fill",
            "selector": "#submit",
            "value": secret_value,
            "_session_id": "sess-a"
        })
        .to_string(),
    )
    .await
    .unwrap();

    let cmds = recorder.commands().await;
    assert_eq!(
        cmds.len(),
        1,
        "consequential fill must prompt once: {cmds:?}"
    );
    assert!(
        !cmds[0].contains(secret_value),
        "fill prompt must not leak the typed value: {}",
        cmds[0]
    );
    // But it should name the field so the user knows what's happening.
    assert!(
        cmds[0].contains("#submit"),
        "fill prompt should identify the target selector: {}",
        cmds[0]
    );
}

/// ALLOW_ONCE on ordinary navigation must NOT mark the session approved: a
/// second ordinary navigation prompts again.
#[tokio::test]
async fn allow_once_does_not_persist_session_approval() {
    let (tool, _backend, recorder) =
        approving_tool(MockBackend::new(), ApprovalResponse::AllowOnce);

    tool.call(
        &json!({ "action": "navigate", "url": "https://a.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    tool.call(
        &json!({ "action": "navigate", "url": "https://b.example/", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();

    assert_eq!(
        recorder.count().await,
        2,
        "AllowOnce must not persist session approval — each navigation prompts"
    );
}

/// ADMINISTRATIVE actions (close/set_mode/close_tab) never prompt, even with a
/// would-deny responder.
#[tokio::test]
async fn administrative_actions_never_prompt() {
    let (tool, _backend, recorder) = approving_tool(MockBackend::new(), ApprovalResponse::Deny);

    let out = tool
        .call(&json!({ "action": "close", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(
        !out.to_lowercase().contains("denied"),
        "close (administrative) must not be denied: {out}"
    );
    assert_eq!(
        recorder.count().await,
        0,
        "administrative actions must not prompt"
    );
}

// =============================================================================
// Task 8 — request-level network policy (SSRF) enforcement
// =============================================================================
//
// These tests pin the two enforcement points that ARE feasible without a real
// Chrome (and thus testable in CI against the mock backend):
//
//   1. The validator is applied to every tool-initiated navigation
//      (`navigate`, `new_tab`) — already exercised elsewhere but re-asserted
//      here for the network-policy contract.
//   2. The FINAL committed page URL is revalidated after `goto` settles, so a
//      server-side redirect that lands on a blocked host is caught even though
//      per-request subresource interception is deferred (see module docs in
//      `web_fetch::classify_blocked_host` and the Task 8 report).
//
// Per-request subresource/XHR/WebSocket interception is NOT implemented (see the
// CDP feasibility finding); these tests deliberately do not claim it.

/// RED→GREEN: a public URL that *redirects* to `127.0.0.1` (modeled by the mock
/// reporting a loopback committed URL after goto) must be treated as BLOCKED.
/// The action must NOT report success, and the error must name only the host
/// CLASS — never the loopback URL or its secret query string.
#[tokio::test]
async fn navigate_redirect_to_loopback_is_blocked_on_final_url() {
    // Public URL passes the pre-flight check, but after goto the page's
    // committed URL is a loopback address carrying a secret token.
    let committed = "http://127.0.0.1:8080/admin?token=SUPERSECRET";
    let (tool, _backend, _rec) = approving_tool(
        MockBackend::new().with_url(committed),
        ApprovalResponse::AllowSession,
    );

    let out = tool
        .call(
            &json!({
                "action": "navigate",
                "url": "https://public-redirector.example/go",
                "_session_id": "sess-a"
            })
            .to_string(),
        )
        .await
        .unwrap();

    // Must be reported as blocked, not as a successful navigation.
    assert!(
        !out.starts_with("Navigated to"),
        "a redirect to loopback must not be reported as a successful navigation: {out}"
    );
    assert!(
        out.to_lowercase().contains("block") && out.to_lowercase().contains("loopback"),
        "blocked navigation must name the loopback host class: {out}"
    );
    // Secret-safety: neither the loopback URL nor its query/token may leak.
    assert!(
        !out.contains("127.0.0.1")
            && !out.contains("SUPERSECRET")
            && !out.contains("token")
            && !out.contains("admin"),
        "blocked-navigation error must not leak the URL/path/query/credentials: {out}"
    );
}

/// A redirect that lands on an ALLOWED public URL still reports success.
#[tokio::test]
async fn navigate_redirect_to_public_is_allowed() {
    let (tool, _backend, _rec) = approving_tool(
        MockBackend::new().with_url("https://final.example/page"),
        ApprovalResponse::AllowSession,
    );

    let out = tool
        .call(
            &json!({
                "action": "navigate",
                "url": "https://start.example/",
                "_session_id": "sess-a"
            })
            .to_string(),
        )
        .await
        .unwrap();

    assert!(
        out.starts_with("Navigated to"),
        "a redirect to a public host must still succeed: {out}"
    );
}

/// The pre-flight validator still blocks a tool-initiated navigation whose URL
/// is itself a private/link-local target, and names only the host class.
#[tokio::test]
async fn navigate_to_metadata_endpoint_is_blocked_preflight() {
    let (tool, backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    let out = tool
        .call(
            &json!({
                "action": "navigate",
                "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
                "_session_id": "sess-a"
            })
            .to_string(),
        )
        .await
        .unwrap();

    assert!(
        out.to_lowercase().contains("block") && out.to_lowercase().contains("link-local"),
        "metadata endpoint must be blocked with its host class: {out}"
    );
    assert!(
        !out.contains("security-credentials") && !out.contains("meta-data"),
        "block error must not echo the metadata path: {out}"
    );
    // A pre-flight block must never reach the backend.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Goto(_))),
        "a pre-flight-blocked navigation must never goto: {calls:?}"
    );
}

/// `new_tab` with a private URL is blocked pre-flight (host class only).
#[tokio::test]
async fn new_tab_to_private_url_is_blocked() {
    let (tool, _backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::AllowSession);

    let out = tool
        .call(
            &json!({
                "action": "new_tab",
                "url": "http://10.0.0.5/internal?secret=abc",
                "_session_id": "sess-a"
            })
            .to_string(),
        )
        .await
        .unwrap();

    assert!(
        out.to_lowercase().contains("block") && out.to_lowercase().contains("private network"),
        "new_tab to a private URL must be blocked with its host class: {out}"
    );
    assert!(
        !out.contains("10.0.0.5") && !out.contains("secret"),
        "new_tab block error must not leak the URL/query: {out}"
    );
}

/// DEFERRED — per-request CDP subresource interception.
///
/// This stub documents the boundary we DO NOT currently enforce, and is a
/// placeholder for a future real-Chrome integration test. It is `#[ignore]`d so
/// it never runs in CI (no real Chrome) and never fakes a pass.
///
/// ## Why it's deferred (chromiumoxide 0.8 feasibility finding)
///
/// chromiumoxide 0.8 exposes request interception ONLY as a browser-global flag
/// (`BrowserConfig::enable_request_intercept`, mapped to `config.request_intercept`
/// and applied at target creation in `handler/target.rs`). There is no
/// `Page::set_request_interception` and no public per-request continue/abort
/// callback. When interception is enabled, `NetworkManager::on_fetch_request_paused`
/// (handler/network.rs:135) deliberately does NOT auto-continue
/// (`if !user_request_interception_enabled && protocol_..._enabled`), so EVERY
/// paused request would hang unless an external consumer drains
/// `Fetch.requestPaused` (forwarded to `event_listener`s via `consume_event!`)
/// and issues `ContinueRequest`/`FailRequest` itself. That requires a dedicated,
/// never-failing per-browser pump whose stall would deadlock ALL browser
/// traffic, and it cannot be exercised by the mock backend. Shipping it would
/// trade a real, testable boundary (pre-flight + final-URL revalidation) for a
/// fragile, untestable one — so subresource/XHR/WebSocket interception is
/// deferred pending a chromiumoxide upgrade or a raw-CDP pump with auto-continue
/// fallback. See the Task 8 report.
/// PART 1 — STATE NEUTRALIZATION: a navigate whose final committed URL is a
/// blocked host must not only return the block error, it must also RESET the page
/// to about:blank so the blocked content is not left committed/observable. We
/// assert both: the block error AND a recorded `goto("about:blank")` after the
/// original goto.
#[tokio::test]
async fn navigate_blocked_redirect_resets_page_to_about_blank() {
    let committed = "http://127.0.0.1/secret";
    let (tool, backend, _rec) = approving_tool(
        MockBackend::new().with_url(committed),
        ApprovalResponse::AllowSession,
    );

    let out = tool
        .call(
            &json!({
                "action": "navigate",
                "url": "https://public-redirector.example/go",
                "_session_id": "sess-a"
            })
            .to_string(),
        )
        .await
        .unwrap();

    // Still reported as blocked (host class only).
    assert!(
        out.to_lowercase().contains("block") && out.to_lowercase().contains("loopback"),
        "blocked redirect must report the loopback host class: {out}"
    );
    assert!(
        !out.contains("127.0.0.1") && !out.contains("/secret"),
        "block error must not leak the URL/path: {out}"
    );

    // The committed state must have been neutralized via goto("about:blank").
    let calls = backend.calls();
    let calls = calls.lock().await;
    let goto_count = calls
        .iter()
        .filter(|c| matches!(c, MockCall::Goto(_)))
        .count();
    assert!(
        calls.contains(&MockCall::Goto("about:blank".to_string())),
        "blocked redirect must reset the page to about:blank: {calls:?}"
    );
    // There must be the original goto to the requested URL AND the about:blank
    // reset (two gotos total), proving the reset happens AFTER landing.
    assert_eq!(
        goto_count, 2,
        "expected the original goto plus the about:blank reset: {calls:?}"
    );
}

/// PART 2 — OBSERVATION REFUSED: when the page's LIVE committed URL is a blocked
/// host (e.g. reached via a post-load JS-redirect / meta-refresh), get_text,
/// screenshot, and execute_js must each REFUSE before reading/capturing/
/// evaluating. The error names ONLY the host class (no IP/path/query), and the
/// underlying read/capture/evaluate call must NOT have been recorded.
#[tokio::test]
async fn observation_actions_refuse_on_blocked_current_url() {
    // The page is currently sitting on a link-local metadata endpoint with a
    // secret path — modeling a post-load redirect we never approved.
    let blocked = "http://169.254.169.254/latest/meta-data/iam/security-credentials/";

    // get_text refuses before body_text/inner_text.
    {
        let (tool, backend, _rec) = approving_tool(
            MockBackend::new().with_url(blocked),
            ApprovalResponse::AllowSession,
        );
        let out = tool
            .call(&json!({ "action": "get_text", "_session_id": "sess-a" }).to_string())
            .await
            .unwrap();
        assert!(
            out.to_lowercase().contains("block") && out.to_lowercase().contains("link-local"),
            "get_text on a blocked URL must refuse with the host class: {out}"
        );
        assert!(
            !out.contains("169.254.169.254")
                && !out.contains("security-credentials")
                && !out.contains("meta-data"),
            "get_text refusal must not leak the URL/path: {out}"
        );
        let calls = backend.calls();
        let calls = calls.lock().await;
        assert!(
            !calls.contains(&MockCall::BodyText)
                && !calls.iter().any(|c| matches!(c, MockCall::InnerText(_))),
            "get_text must refuse BEFORE reading any text: {calls:?}"
        );
    }

    // screenshot refuses before capturing.
    {
        let (tool, backend, _rec) = approving_tool(
            MockBackend::new().with_url(blocked),
            ApprovalResponse::AllowSession,
        );
        let out = tool
            .call(&json!({ "action": "screenshot", "_session_id": "sess-a" }).to_string())
            .await
            .unwrap();
        assert!(
            out.to_lowercase().contains("block") && out.to_lowercase().contains("link-local"),
            "screenshot on a blocked URL must refuse with the host class: {out}"
        );
        assert!(
            !out.contains("169.254.169.254")
                && !out.contains("security-credentials")
                && !out.contains("meta-data"),
            "screenshot refusal must not leak the URL/path: {out}"
        );
        let calls = backend.calls();
        let calls = calls.lock().await;
        assert!(
            !calls.iter().any(|c| matches!(c, MockCall::Screenshot(..))),
            "screenshot must refuse BEFORE capturing: {calls:?}"
        );
    }

    // execute_js refuses before evaluating (after the approval gate).
    {
        let (tool, backend, _rec) = approving_tool(
            MockBackend::new().with_url(blocked),
            ApprovalResponse::AllowSession,
        );
        let out = tool
            .call(
                &json!({
                    "action": "execute_js",
                    "script": "document.body.innerText",
                    "_session_id": "sess-a"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(
            out.to_lowercase().contains("block") && out.to_lowercase().contains("link-local"),
            "execute_js on a blocked URL must refuse with the host class: {out}"
        );
        assert!(
            !out.contains("169.254.169.254")
                && !out.contains("security-credentials")
                && !out.contains("meta-data"),
            "execute_js refusal must not leak the URL/path: {out}"
        );
        let calls = backend.calls();
        let calls = calls.lock().await;
        assert!(
            !calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
            "execute_js must refuse BEFORE evaluating: {calls:?}"
        );
    }
}

/// PART 2 — PRIVATE-NETWORK CLASS: the same refusal holds for an RFC1918 private
/// host, proving the gate uses the shared classifier rather than a special case.
#[tokio::test]
async fn observation_actions_refuse_on_private_current_url() {
    let blocked = "http://10.0.0.5/internal?secret=abc";
    let (tool, backend, _rec) = approving_tool(
        MockBackend::new().with_url(blocked),
        ApprovalResponse::AllowSession,
    );

    let out = tool
        .call(&json!({ "action": "get_text", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();
    assert!(
        out.to_lowercase().contains("block") && out.to_lowercase().contains("private network"),
        "get_text on a private URL must refuse with the host class: {out}"
    );
    assert!(
        !out.contains("10.0.0.5") && !out.contains("secret") && !out.contains("internal"),
        "refusal must not leak the URL/query: {out}"
    );
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.contains(&MockCall::BodyText),
        "get_text must refuse before reading on a private host: {calls:?}"
    );
}

/// REGRESSION: with the live URL on a PUBLIC host, get_text / screenshot /
/// execute_js still proceed normally (the gate only blocks blocked hosts).
#[tokio::test]
async fn observation_actions_allowed_on_public_current_url() {
    let public = "https://app.example/dashboard";

    // get_text proceeds.
    {
        let (tool, backend, _rec) = approving_tool(
            MockBackend::new()
                .with_url(public)
                .with_text_result("public body"),
            ApprovalResponse::AllowSession,
        );
        let out = tool
            .call(&json!({ "action": "get_text", "_session_id": "sess-a" }).to_string())
            .await
            .unwrap();
        assert_eq!(out, "public body", "public get_text must proceed: {out}");
        assert!(
            calls_contains(&backend, &MockCall::BodyText).await,
            "public get_text must reach body_text"
        );
    }

    // screenshot proceeds. A media responder drains the channel and confirms
    // delivery so the tool's bounded delivery-await resolves promptly.
    {
        let (tool, backend, rx) = mock_tool_with(MockBackend::new().with_url(public));
        let _media = spawn_media_responder(rx, Ok(()));
        let out = tool
            .call(&json!({ "action": "screenshot", "_session_id": "sess-a" }).to_string())
            .await
            .unwrap();
        assert!(
            out.contains("delivered to chat"),
            "public screenshot must proceed: {out}"
        );
        assert!(
            calls_contains(&backend, &MockCall::Screenshot(None, false)).await,
            "public screenshot must reach the capture"
        );
    }

    // execute_js proceeds.
    {
        let (tool, backend, _rec) = approving_tool(
            MockBackend::new().with_url(public),
            ApprovalResponse::AllowSession,
        );
        let out = tool
            .call(
                &json!({ "action": "execute_js", "script": "1 + 1", "_session_id": "sess-a" })
                    .to_string(),
            )
            .await
            .unwrap();
        assert!(
            !out.to_lowercase().contains("block"),
            "public execute_js must not be blocked: {out}"
        );
        assert!(
            calls_contains(&backend, &MockCall::Evaluate("1 + 1".to_string())).await,
            "public execute_js must reach evaluate"
        );
    }
}

#[tokio::test]
#[ignore = "requires real Chrome + CDP Fetch pump; deferred (see doc comment)"]
async fn deferred_per_request_subresource_interception_stub() {
    // Intentionally empty: documents the deferred boundary without asserting a
    // capability we do not have. When implemented, this should drive a real
    // page whose loaded subresource targets a private IP and assert the
    // subresource request is failed with BlockedByClient.
}

// =============================================================================
// Task 9 — Constrain JavaScript execution
// =============================================================================
//
// JS-originated subresource/XHR/WebSocket interception shares the Task 8
// deferral (chromiumoxide 0.8 browser-global only; no per-page seam; would
// deadlock/untestable). The live-URL gate (`ensure_current_url_allowed`) is
// the feasible mitigation: an approved execute_js still cannot read out a
// private host the page redirected to post-load. See the `#[ignore]`d stub
// above for the full CDP feasibility note.

/// SCRIPT SIZE CAP: a script larger than MAX_SCRIPT_BYTES (64 KiB) is rejected
/// with a clear size error BEFORE evaluation AND before the approval prompt
/// is sent.
#[tokio::test]
async fn oversized_script_rejected_before_evaluate() {
    // Build a script slightly over 64 KiB.
    let script = "x".repeat(64 * 1024 + 1);
    let (tool, backend, recorder) = approving_tool(MockBackend::new(), ApprovalResponse::AllowOnce);

    let args = serde_json::json!({
        "action": "execute_js",
        "script": script,
        "_session_id": "sess-a"
    });
    let out = tool.call(&args.to_string()).await.unwrap();

    // The error must name the size and the limit.
    assert!(
        out.to_lowercase().contains("script too large") || out.to_lowercase().contains("too large"),
        "oversized script must produce a size error: {out}"
    );
    assert!(
        out.contains("65536") || out.contains("64"),
        "size error must name the size limit: {out}"
    );

    // No Evaluate call must have been recorded.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
        "oversized script must not be evaluated: {calls:?}"
    );

    // Bonus: no approval prompt should have been sent (check is pre-approval).
    assert_eq!(
        recorder.count().await,
        0,
        "oversized script must not produce an approval prompt (rejected before gate): {}",
        recorder.count().await
    );
}

/// BROWSER-MANAGEMENT API DENYLIST: scripts referencing window.open, chrome.*,
/// or the debugger escape pattern are rejected without evaluation.
#[tokio::test]
async fn browser_management_api_denied_not_evaluated() {
    let denied_scripts: &[(&str, &str)] = &[
        // window.open spawns tabs outside the session/tab model.
        ("window.open('https://evil.example')", "window.open"),
        // chrome.* namespace gives access to privileged browser APIs.
        ("chrome.debugger.attach({tabId:1},'1.3')", "chrome.debugger"),
        ("chrome.management.getAll()", "chrome.management"),
        (
            "chrome.tabs.query({active:true},function(t){})",
            "chrome.tabs",
        ),
        // Bare `chrome.` prefix — any chrome namespace access is blocked.
        ("chrome.runtime.sendMessage('ext-id',{})", "chrome.runtime"),
    ];

    for (script, denied_capability) in denied_scripts {
        let (tool, backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::AllowOnce);

        let args = serde_json::json!({
            "action": "execute_js",
            "script": script,
            "_session_id": "sess-a"
        });
        let out = tool.call(&args.to_string()).await.unwrap();

        // Error must name the capability class, not echo the full script.
        assert!(
            out.to_lowercase().contains("error"),
            "denied script must produce an error [{denied_capability}]: {out}"
        );
        // Must not echo the whole script body back to the user.
        // (The error may be longer than the script — it explains the policy —
        // but must not literally contain the script text itself.)
        assert!(
            !out.contains(*script),
            "error must not echo the whole script body [{denied_capability}]: {out}"
        );

        let calls = backend.calls();
        let calls = calls.lock().await;
        assert!(
            !calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
            "denied browser-management script must not be evaluated [{denied_capability}]: {calls:?}"
        );
    }

    // CONTROL: a benign script must NOT be blocked.
    let (tool, backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::AllowOnce);
    let benign = "document.title";
    let args = serde_json::json!({
        "action": "execute_js",
        "script": benign,
        "_session_id": "sess-control"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        !out.to_lowercase().contains("browser management")
            && !out.to_lowercase().contains("not allowed"),
        "benign script must not be rejected by the denylist: {out}"
    );
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
        "benign script must reach evaluate: {calls:?}"
    );
}

/// PRIVATE-NETWORK BYPASS: an approved execute_js whose page's live URL is a
/// blocked host must be refused BEFORE evaluation — the Task 8 live-URL gate
/// (`ensure_current_url_allowed`) still holds for execute_js.
#[tokio::test]
async fn approved_execute_js_refused_on_blocked_current_url() {
    // Page is sitting on the AWS metadata endpoint.
    let blocked = "http://169.254.169.254/latest/meta-data/";
    let (tool, backend, recorder) = approving_tool(
        MockBackend::new().with_url(blocked),
        ApprovalResponse::AllowOnce,
    );

    let args = serde_json::json!({
        "action": "execute_js",
        "script": "document.body.innerText",
        "_session_id": "sess-a"
    });
    let out = tool.call(&args.to_string()).await.unwrap();

    // Blocked by the live-URL gate, even though the user approved.
    assert!(
        out.to_lowercase().contains("block") && out.to_lowercase().contains("link-local"),
        "execute_js on a blocked URL must refuse with the host class: {out}"
    );
    // Must not leak the URL or path.
    assert!(
        !out.contains("169.254.169.254") && !out.contains("meta-data"),
        "refusal must not leak the URL/path: {out}"
    );
    // Approval was prompted (the gate runs before the live-URL check but after
    // the constraint check); script was NOT evaluated.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
        "execute_js on a blocked URL must not be evaluated: {calls:?}"
    );
    // The approval prompt was sent (constraint checks pass for a benign script,
    // approval gate runs, then the live-URL check fires before evaluate).
    assert_eq!(
        recorder.count().await,
        1,
        "approval gate must have been reached before the live-URL check"
    );
}

/// RESULT REDACTION — execute_js: when the script returns a value matching a
/// secret pattern (e.g. an API key), the returned result must be redacted
/// before reaching the caller.
#[tokio::test]
async fn execute_js_result_is_redacted() {
    // An sk- prefixed key that matches the SECRET_PATTERNS "API key" regex.
    // Pattern: sk-[a-zA-Z0-9]{20,}
    let raw_secret = "sk-abc12345678901234567890";
    let (tool, backend, _rec) = approving_tool(
        MockBackend::new()
            .with_eval_result(Some(serde_json::Value::String(raw_secret.to_string()))),
        ApprovalResponse::AllowOnce,
    );

    let args = serde_json::json!({
        "action": "execute_js",
        "script": "getSecret()",
        "_session_id": "sess-a"
    });
    let out = tool.call(&args.to_string()).await.unwrap();

    // The raw secret must not appear in the output.
    assert!(
        !out.contains(raw_secret),
        "execute_js result must have the secret redacted: {out}"
    );
    // The redaction marker must be present.
    assert!(
        out.contains("[REDACTED:") || out.contains("REDACTED"),
        "execute_js result must contain a redaction marker: {out}"
    );
    // The evaluate call DID happen — redaction is applied to the RESULT, not
    // the script. The script itself was benign.
    assert!(
        calls_contains(&backend, &MockCall::Evaluate("getSecret()".to_string())).await,
        "the evaluate must have been called for a valid script: backend calls"
    );
}

/// RESULT REDACTION — get_text: text extracted from the DOM that contains a
/// secret pattern is also redacted (same spec requirement: "returned DOM
/// content passes existing secret redaction").
#[tokio::test]
async fn get_text_result_is_redacted() {
    // A Bearer token in the page text.
    // Pattern: Bearer\s+[a-zA-Z0-9\-._~+/]+=*
    let raw_secret = "Bearer eyJhbGciOiJSUzI1NiJ9";
    let (tool, backend, _rec) = approving_tool(
        MockBackend::new().with_text_result(raw_secret),
        ApprovalResponse::AllowSession,
    );

    let args = serde_json::json!({
        "action": "get_text",
        "_session_id": "sess-a"
    });
    let out = tool.call(&args.to_string()).await.unwrap();

    // The raw bearer token must not survive in the output.
    assert!(
        !out.contains("eyJhbGciOiJSUzI1NiJ9"),
        "get_text result must have the bearer token redacted: {out}"
    );
    assert!(
        out.contains("[REDACTED:") || out.contains("REDACTED"),
        "get_text result must contain a redaction marker: {out}"
    );
    // body_text was still called (redaction applied to result, not blocked).
    assert!(
        calls_contains(&backend, &MockCall::BodyText).await,
        "body_text must have been called: backend calls"
    );
}

/// DENIED SCRIPT NOT EVALUATED (Task 7 confirmation): a user Deny on execute_js
/// means no Evaluate is recorded. This is Task 7 behavior; included here for
/// completeness per the spec.
#[tokio::test]
async fn denied_execute_js_not_evaluated() {
    let (tool, backend, _rec) = approving_tool(MockBackend::new(), ApprovalResponse::Deny);

    let args = serde_json::json!({
        "action": "execute_js",
        "script": "document.title",
        "_session_id": "sess-a"
    });
    let out = tool.call(&args.to_string()).await.unwrap();

    assert!(
        out.to_lowercase().contains("denied"),
        "Deny response must produce a denied message: {out}"
    );
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.iter().any(|c| matches!(c, MockCall::Evaluate(_))),
        "a denied execute_js must not be evaluated: {calls:?}"
    );
}

// ============================================================================
// Task 10: handler-health monitoring + disconnect recovery
// ============================================================================

use super::backend::{is_connection_error, BrowserBackend, FailOp};

/// `is_connection_error` recognizes transport/connection-death patterns but NOT
/// ordinary page errors. This is the classifier that gates ALL recovery.
#[test]
fn connection_error_classifier_only_matches_transport_failures() {
    // Transport/connection-class failures → true.
    for e in &[
        "WebSocket connection closed: Sender was dropped",
        "connection reset by peer",
        "Ws(connection aborted)",
        "the channel closed unexpectedly",
        "request did not resolve",
        "no response from the browser",
        "the browser was closed",
        "Transport error",
    ] {
        assert!(
            is_connection_error(e),
            "should be classified as a connection error: {e}"
        );
    }

    // Ordinary page-level errors → false (must NOT trigger a reconnect).
    for e in &[
        "Element not found '#missing': node not found",
        "JavaScript execution failed: ReferenceError: x is not defined",
        "Timeout: element '#foo' not found after 10s",
        "Failed to navigate to https://x.test: net::ERR_NAME_NOT_RESOLVED",
        "Navigation blocked: target is a loopback address",
    ] {
        assert!(
            !is_connection_error(e),
            "ordinary page error must NOT be a connection error: {e}"
        );
    }
}

/// HANDLER-EXIT DETECTION: when the mock reports its handler dead, `ensure_ready`
/// (driven by the first action) must NOT treat the connection as healthy — it
/// relaunches/reconnects (a fresh EnsureReady) and the action then works.
#[tokio::test]
async fn dead_handler_triggers_relaunch_on_next_action() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("alive"));

    // First action establishes the connection.
    let args = json!({ "action": "get_text", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "alive");
    assert!(
        backend.is_connected().await,
        "connection should be live after first action"
    );

    // Connection dies: the CDP handler exits, but a naive is_some() check would
    // still report healthy. Prove our health gate catches it.
    backend.mark_handler_dead();
    assert!(
        !backend.is_connected().await,
        "a dead handler must report NOT connected (the core bug)"
    );

    // Next action must observe the dead handler and relaunch (fresh EnsureReady),
    // then succeed.
    let out2 = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out2, "alive", "action after relaunch should succeed");
    assert!(
        backend.is_connected().await,
        "relaunch should have revived the connection"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    let ensure_count = calls
        .iter()
        .filter(|c| matches!(c, MockCall::EnsureReady))
        .count();
    assert_eq!(
        ensure_count, 2,
        "ensure_ready must have run again after the handler died: {calls:?}"
    );
}

/// OBSERVATION RETRY: `get_text` hits a one-shot connection error on the first
/// `body_text`, then after exactly ONE reconnect + a fresh page the retry
/// succeeds and returns the text.
#[tokio::test]
async fn observation_retries_once_after_connection_error() {
    let (tool, backend, _rx) =
        mock_tool_with(MockBackend::new().with_text_result("recovered text"));
    backend
        .fail_once_with_connection_error_on(FailOp::BodyText)
        .await;

    let args = json!({ "action": "get_text", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(
        out, "recovered text",
        "observation should retry and return the text after reconnect"
    );

    assert_eq!(
        backend.reconnect_count().await,
        1,
        "exactly one reconnect for an observation connection error"
    );

    // The op was attempted twice: once (failed) + once (retry succeeded).
    let calls = backend.calls();
    let calls = calls.lock().await;
    let body_calls = calls
        .iter()
        .filter(|c| matches!(c, MockCall::BodyText))
        .count();
    assert_eq!(
        body_calls, 2,
        "body_text should be attempted once then retried once: {calls:?}"
    );
}

/// MUTATION NON-REPLAY: `click` hits a connection error → the action surfaces an
/// error, the click is attempted only ONCE (no silent second Click), and a
/// reconnect happens for recovery (so the NEXT action works) — but the mutation
/// is NOT replayed.
#[tokio::test]
async fn mutation_is_not_replayed_after_connection_error() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new());
    backend
        .fail_once_with_connection_error_on(FailOp::Click)
        .await;

    let args = json!({ "action": "click", "selector": "#buy-now", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.to_lowercase().contains("could not be confirmed")
            || out.to_lowercase().contains("not retried")
            || out.to_lowercase().contains("connection was lost"),
        "a mutation disconnect must surface a clear non-replay error: {out}"
    );

    // Reconnect happened (recovery for subsequent actions)...
    assert_eq!(
        backend.reconnect_count().await,
        1,
        "the connection should be restored for the next action"
    );

    // ...but the click was attempted exactly ONCE (NEVER replayed).
    let calls = backend.calls();
    let calls = calls.lock().await;
    let click_calls = calls
        .iter()
        .filter(|c| matches!(c, MockCall::Click(_)))
        .count();
    assert_eq!(
        click_calls, 1,
        "click must NOT be auto-replayed after a disconnect: {calls:?}"
    );
}

/// MUTATION NON-REPLAY (fill): same contract as click — a fill that disconnects
/// is never silently re-typed.
#[tokio::test]
async fn fill_is_not_replayed_after_connection_error() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new());
    backend
        .fail_once_with_connection_error_on(FailOp::ReplaceText)
        .await;

    let args = json!({
        "action": "fill",
        "selector": "#card-number",
        "value": "4111111111111111",
        "_session_id": "sess-a"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.to_lowercase().contains("connection was lost")
            || out.to_lowercase().contains("could not be confirmed"),
        "a fill disconnect must surface a clear non-replay error: {out}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    let fill_calls = calls
        .iter()
        .filter(|c| matches!(c, MockCall::ReplaceText(..)))
        .count();
    assert_eq!(
        fill_calls, 1,
        "fill must NOT be auto-replayed after a disconnect: {calls:?}"
    );
    // The secret value must never appear in the surfaced error.
    assert!(
        !out.contains("4111111111111111"),
        "the fill value must never leak into the error message: {out}"
    );
}

/// NEGATIVE CONTROL: a normal "element not found" error is NOT a connection
/// error — no reconnect, surfaced verbatim. `wait` on a missing element times
/// out (an ordinary error).
#[tokio::test]
async fn ordinary_error_does_not_trigger_reconnect() {
    // `screenshot` element-not-found is an ordinary error in real Chrome; the
    // mock never errors there, so instead drive a clearly-ordinary path: a
    // `switch_tab` to an unknown id returns a normal (non-connection) error.
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new());

    // Establish the session first.
    let _ = tool
        .call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    let out = tool
        .call(
            &json!({ "action": "switch_tab", "tab_id": "does-not-exist", "_session_id": "sess-a" })
                .to_string(),
        )
        .await
        .unwrap();
    assert!(
        out.to_lowercase().contains("unknown tab"),
        "ordinary error should be surfaced verbatim: {out}"
    );
    assert_eq!(
        backend.reconnect_count().await,
        0,
        "an ordinary error must NOT trigger a reconnect"
    );
}

// =============================================================================
// Task 11 — graceful Chromium shutdown + idle session/context eviction
// =============================================================================

use super::session::{BrowserSessionRegistry, EvictedSession};
use tokio::time::Instant;

/// A launched-browser `close` must perform a GRACEFUL close (not just an abort).
#[tokio::test]
async fn close_on_launched_browser_records_graceful_shutdown() {
    let (tool, backend, _rx) = mock_tool();

    // Establish a session so a browser is "live", then close.
    tool.call(
        &json!({ "action": "navigate", "url": "https://example.com", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    let out = tool
        .call(&json!({ "action": "close" }).to_string())
        .await
        .unwrap();
    assert!(out.contains("closed"), "close should report closed: {out}");

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Shutdown { graceful: true }),
        "launched close must record a graceful shutdown: {calls:?}"
    );
}

/// `set_mode` that actually changes the mode must reuse the graceful shutdown
/// path to tear the old browser down (launched → graceful) so the next action
/// relaunches in the new mode. The mock models this teardown
/// (`set_headless_mode` records a `Shutdown` on a real change while a browser is
/// live, mirroring `ChromiumoxideBackend::set_headless_mode` →
/// `graceful_teardown`), so the test can verify the REUSE — not merely that the
/// switch was routed. True teardown against real Chrome is exercised in Task 18.
#[tokio::test]
async fn set_mode_change_reuses_graceful_shutdown() {
    let (tool, backend, _rx) = mock_tool();

    // Bring a browser up (the mock defaults to headless), then switch to visible
    // — a REAL mode change while a browser is live, which must reuse the graceful
    // teardown path.
    tool.call(
        &json!({ "action": "navigate", "url": "https://example.com", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    let out = tool
        .call(&json!({ "action": "set_mode", "value": "visible" }).to_string())
        .await
        .unwrap();
    assert!(
        out.to_lowercase().contains("visible"),
        "set_mode should confirm the new mode: {out}"
    );
    {
        let calls = backend.calls();
        let calls = calls.lock().await;
        assert!(
            calls.contains(&MockCall::SetHeadlessMode(false)),
            "set_mode must route the mode switch to the backend: {calls:?}"
        );
        // The real mode change reused the graceful teardown path.
        assert!(
            calls.contains(&MockCall::Shutdown { graceful: true }),
            "a real set_mode change must reuse the graceful shutdown path: {calls:?}"
        );
    }

    // A no-op same-mode call (switch to visible again) must NOT tear the browser
    // down again — bring a browser back up first, then re-issue the SAME mode.
    tool.call(
        &json!({ "action": "navigate", "url": "https://example.com", "_session_id": "sess-a" })
            .to_string(),
    )
    .await
    .unwrap();
    let shutdowns_before = {
        let calls = backend.calls();
        let calls = calls.lock().await;
        calls
            .iter()
            .filter(|c| matches!(c, MockCall::Shutdown { .. }))
            .count()
    };
    tool.call(&json!({ "action": "set_mode", "value": "visible" }).to_string())
        .await
        .unwrap();
    let calls = backend.calls();
    let calls = calls.lock().await;
    let shutdowns_after = calls
        .iter()
        .filter(|c| matches!(c, MockCall::Shutdown { .. }))
        .count();
    assert_eq!(
        shutdowns_before, shutdowns_after,
        "a no-op same-mode set_mode must NOT record a teardown: {calls:?}"
    );
}

/// An ATTACHED browser must DETACH on shutdown — never issue a graceful
/// browser-close command (that would kill the user's own Chrome).
#[tokio::test]
async fn shutdown_on_attached_browser_detaches_without_closing() {
    let backend = Arc::new(MockBackend::new().attached());
    let msg = backend.shutdown().await.unwrap();
    assert!(
        msg.to_lowercase().contains("still running"),
        "attached shutdown should reassure the user's browser keeps running: {msg}"
    );
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Shutdown { graceful: false }),
        "attached shutdown must NOT be graceful (no browser-close command): {calls:?}"
    );
    assert!(
        !calls.contains(&MockCall::Shutdown { graceful: true }),
        "attached shutdown must never issue a graceful browser-close: {calls:?}"
    );
}

/// Forced-cleanup fallback: when the graceful close errors/times out, shutdown
/// still completes (does not hang) and leaves the backend in a clean,
/// disconnected state.
#[tokio::test]
async fn shutdown_forces_cleanup_when_graceful_close_fails() {
    let backend = Arc::new(MockBackend::new().with_failing_graceful_close());
    backend.ensure_ready().await.unwrap();
    assert!(backend.is_connected().await, "precondition: connected");

    // Bounded so a hang fails the test rather than wedging CI.
    let msg = tokio::time::timeout(Duration::from_secs(2), backend.shutdown())
        .await
        .expect("shutdown must not hang on a failing graceful close")
        .unwrap();
    assert!(
        msg.contains("closed"),
        "forced cleanup still reports closed: {msg}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Shutdown { graceful: false }),
        "failing graceful close must fall back to a forced (non-graceful) teardown: {calls:?}"
    );
    drop(calls);
    assert!(
        !backend.is_connected().await,
        "after forced cleanup the backend must be disconnected"
    );
}

/// `shutdown` is idempotent: a second call on an already-torn-down backend is a
/// safe no-op that does not hang.
#[tokio::test]
async fn shutdown_is_idempotent() {
    let backend = Arc::new(MockBackend::new());
    backend.ensure_ready().await.unwrap();
    let _ = backend.shutdown().await.unwrap();
    let second = tokio::time::timeout(Duration::from_secs(2), backend.shutdown())
        .await
        .expect("second shutdown must not hang")
        .unwrap();
    // The mock always returns the closed message; the real backend returns the
    // "no browser was active" no-op. Either is a clean, non-hanging result.
    assert!(!second.is_empty());
}

/// Deterministic idle eviction: sessions older than the threshold are removed
/// and their tab target ids + context ids are returned for disposal, while a
/// recently-used session survives. Evicted sessions' approval is dropped, and
/// the backend disposes the returned resources.
#[tokio::test]
async fn evict_idle_removes_stale_sessions_and_disposes_resources() {
    let registry = BrowserSessionRegistry::new();
    let backend = Arc::new(MockBackend::new().with_isolated_contexts());
    // A page handle for the seeded tabs (the backend is only used for disposal).
    let (_id, _ctx, page) = backend.create_page().await.unwrap();

    let now = Instant::now();
    let old = now - Duration::from_secs(60 * 60); // 1h ago — idle
    let recent = now - Duration::from_secs(60); // 1m ago — fresh

    // Two idle sessions (one with a context, one with two tabs sharing a ctx),
    // and one recently-used session that must survive.
    registry
        .seed_session_for_test(
            "idle-a",
            old,
            vec![("t-a1".into(), Some("ctx-a".into()))],
            Arc::clone(&page),
            /* approved */ true,
        )
        .await;
    registry
        .seed_session_for_test(
            "idle-b",
            old,
            vec![
                ("t-b1".into(), Some("ctx-b".into())),
                ("t-b2".into(), Some("ctx-b".into())),
            ],
            Arc::clone(&page),
            true,
        )
        .await;
    registry
        .seed_session_for_test(
            "fresh",
            recent,
            vec![("t-f1".into(), Some("ctx-f".into()))],
            Arc::clone(&page),
            true,
        )
        .await;

    let mut evicted = registry.evict_idle(now, Duration::from_secs(30 * 60)).await;
    evicted.sort_by(|a, b| a.session_id.cmp(&b.session_id));

    assert_eq!(
        evicted,
        vec![
            EvictedSession {
                session_id: "idle-a".into(),
                tab_target_ids: vec!["t-a1".into()],
                context_ids: vec!["ctx-a".into()],
            },
            EvictedSession {
                session_id: "idle-b".into(),
                // two tabs, but one DISTINCT context id (deduped).
                tab_target_ids: vec!["t-b1".into(), "t-b2".into()],
                context_ids: vec!["ctx-b".into()],
            },
        ],
        "only idle sessions evicted, with deduped context ids: {evicted:?}"
    );

    // Idle sessions are gone; the fresh one survives.
    assert!(!registry.has_session_for_test("idle-a").await);
    assert!(!registry.has_session_for_test("idle-b").await);
    assert!(registry.has_session_for_test("fresh").await);

    // Approval for evicted sessions is dropped (safe-default re-approve); the
    // fresh session keeps its approval.
    assert!(!registry.is_session_approved("idle-a").await);
    assert!(!registry.is_session_approved("idle-b").await);
    assert!(registry.is_session_approved("fresh").await);

    // The caller disposes the returned resources via the backend.
    for ev in &evicted {
        backend
            .dispose_session(&ev.tab_target_ids, &ev.context_ids)
            .await;
    }
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(calls.contains(&MockCall::CloseTarget("t-a1".into())));
    assert!(calls.contains(&MockCall::CloseTarget("t-b1".into())));
    assert!(calls.contains(&MockCall::CloseTarget("t-b2".into())));
    assert!(calls.contains(&MockCall::DisposeContext("ctx-a".into())));
    assert!(calls.contains(&MockCall::DisposeContext("ctx-b".into())));
    assert!(
        !calls.contains(&MockCall::DisposeContext("ctx-f".into())),
        "the surviving session's context must NOT be disposed: {calls:?}"
    );
}

/// Nothing is evicted when every session is within the idle threshold.
#[tokio::test]
async fn evict_idle_keeps_all_fresh_sessions() {
    let registry = BrowserSessionRegistry::new();
    let backend = Arc::new(MockBackend::new().with_isolated_contexts());
    let (_id, _ctx, page) = backend.create_page().await.unwrap();
    let now = Instant::now();
    registry
        .seed_session_for_test(
            "fresh",
            now - Duration::from_secs(60),
            vec![("t1".into(), Some("ctx".into()))],
            page,
            true,
        )
        .await;
    let evicted = registry.evict_idle(now, Duration::from_secs(30 * 60)).await;
    assert!(evicted.is_empty(), "no fresh session should be evicted");
    assert!(registry.has_session_for_test("fresh").await);
}

/// INVARIANT: a session with a HELD action lock is never evicted, even when it
/// is past the idle threshold. An action that outruns the idle window must not
/// have its tab targets / browser context disposed out from under the live page
/// handle (use-after-dispose). The unheld idle sibling IS evicted in the same
/// sweep, proving the skip is targeted, not a blanket bail-out.
#[tokio::test]
async fn evict_idle_skips_session_with_held_action_lock() {
    let registry = BrowserSessionRegistry::new();
    let backend = Arc::new(MockBackend::new().with_isolated_contexts());
    let (_id, _ctx, page) = backend.create_page().await.unwrap();

    let now = Instant::now();
    let old = now - Duration::from_secs(60 * 60); // 1h ago — both idle.

    // One idle session that is MID-ACTION (its action lock is held), and one
    // idle session that is not.
    registry
        .seed_session_for_test(
            "busy",
            old,
            vec![("t-busy".into(), Some("ctx-busy".into()))],
            Arc::clone(&page),
            true,
        )
        .await;
    registry
        .seed_session_for_test(
            "free",
            old,
            vec![("t-free".into(), Some("ctx-free".into()))],
            Arc::clone(&page),
            true,
        )
        .await;

    // Model "busy" running an action: hold an owned guard on its action lock for
    // the duration of the eviction sweep.
    let busy_lock = registry
        .action_lock_for_test("busy")
        .await
        .expect("busy session seeded");
    let _held = busy_lock.lock_owned().await;

    let evicted = registry.evict_idle(now, Duration::from_secs(30 * 60)).await;

    // Only the unheld idle session is evicted; the busy one survives untouched.
    assert_eq!(
        evicted,
        vec![EvictedSession {
            session_id: "free".into(),
            tab_target_ids: vec!["t-free".into()],
            context_ids: vec!["ctx-free".into()],
        }],
        "only the unheld idle session is evicted: {evicted:?}"
    );
    assert!(
        registry.has_session_for_test("busy").await,
        "a session with a held action lock must survive eviction"
    );
    assert!(!registry.has_session_for_test("free").await);

    // The busy session's resources are NOT returned for disposal, so disposing
    // the evicted set never touches them.
    for ev in &evicted {
        backend
            .dispose_session(&ev.tab_target_ids, &ev.context_ids)
            .await;
    }
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        !calls.contains(&MockCall::CloseTarget("t-busy".into())),
        "the busy session's tab target must NOT be closed: {calls:?}"
    );
    assert!(
        !calls.contains(&MockCall::DisposeContext("ctx-busy".into())),
        "the busy session's context must NOT be disposed: {calls:?}"
    );
    assert!(
        calls.contains(&MockCall::CloseTarget("t-free".into())),
        "the free session's tab target should be closed: {calls:?}"
    );
}

/// Opportunistic eviction: creating a NEW session sweeps an already-idle sibling
/// (its tabs/context get disposed via the backend). Driven through the real
/// `get_or_create_page` path, with the idle session seeded far in the past so it
/// crosses the default threshold.
#[tokio::test]
async fn creating_new_session_opportunistically_evicts_idle_sibling() {
    let registry = BrowserSessionRegistry::new();
    let backend = Arc::new(MockBackend::new().with_isolated_contexts());
    let (_id, _ctx, page) = backend.create_page().await.unwrap();

    // Seed an idle session older than the DEFAULT threshold (30 min).
    registry
        .seed_session_for_test(
            "idle",
            Instant::now() - Duration::from_secs(45 * 60),
            vec![("t-idle".into(), Some("ctx-idle".into()))],
            page,
            true,
        )
        .await;

    // Creating a brand-new session must opportunistically evict the idle one.
    let (_p, _lock) = registry
        .get_or_create_page("new-session", &*backend)
        .await
        .unwrap();

    assert!(
        !registry.has_session_for_test("idle").await,
        "idle sibling must be evicted when a new session is created"
    );
    assert!(registry.has_session_for_test("new-session").await);

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::CloseTarget("t-idle".into()))
            && calls.contains(&MockCall::DisposeContext("ctx-idle".into())),
        "the idle sibling's tab + context must be disposed via the backend: {calls:?}"
    );
}

// =============================================================================
// Task 13 — deterministic bounded waits (fake clock)
// =============================================================================
//
// These tests run under `#[tokio::test(start_paused = true)]`: the runtime
// auto-advances the clock to the next pending timer whenever it goes idle, so a
// poll loop that sleeps `WAIT_POLL_INTERVAL` between probes drives forward with
// NO real time elapsing and NO manual `advance()` needed. A condition scripted
// to become true after N polls (via `MockElementState`) resolves instantly; a
// never-satisfied condition hits the bounded deadline instantly. This proves the
// waits are bounded AND that they don't hang.

use super::backend::MockElementState;

/// Build a tool with short, paused-clock-friendly timeouts (element/nav/action
/// all 4s) so the bounded-timeout deadline is reached in a small number of
/// polls under the fake clock.
fn wait_tool(backend: MockBackend) -> (BrowserTool, Arc<MockBackend>) {
    let backend = Arc::new(backend);
    let (media_tx, _rx) = mpsc::channel(8);
    let (broker, _rec) = spawn_responder(ApprovalResponse::AllowSession);
    let tool = BrowserTool::with_backend_and_approval(
        backend.clone(),
        media_tx,
        broker,
        Duration::from_secs(5),
    )
    .with_timeouts(
        Duration::from_secs(4),
        Duration::from_secs(4),
        Duration::from_secs(4),
    );
    (tool, backend)
}

// --- present ---------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn wait_present_succeeds_when_element_appears_before_deadline() {
    let state = MockElementState {
        present_after: Some(2),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args = json!({ "action": "wait", "selector": "#go", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Element '#go' found", "present should succeed: {out}");
}

#[tokio::test(start_paused = true)]
async fn wait_present_times_out_when_element_never_appears() {
    let state = MockElementState {
        present_after: Some(u64::MAX),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args = json!({ "action": "wait", "selector": "#never", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error: Timeout") && out.contains("#never") && out.contains("4s"),
        "present should bound-timeout: {out}"
    );
}

// --- visible ---------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn wait_visible_succeeds_when_element_becomes_visible() {
    let state = MockElementState {
        visible_after: Some(3),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args =
        json!({ "action": "wait", "selector": "#v", "condition": "visible", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(
        out, "Element '#v' is visible",
        "visible should succeed: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_visible_times_out_when_never_visible() {
    let state = MockElementState {
        visible_after: Some(u64::MAX),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args =
        json!({ "action": "wait", "selector": "#v", "condition": "visible", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error: Timeout") && out.contains("not visible"),
        "visible should bound-timeout: {out}"
    );
}

// --- enabled ---------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn wait_enabled_succeeds_when_element_becomes_enabled() {
    let state = MockElementState {
        enabled_after: Some(2),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args =
        json!({ "action": "wait", "selector": "#e", "condition": "enabled", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(
        out, "Element '#e' is enabled",
        "enabled should succeed: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_enabled_times_out_when_never_enabled() {
    let state = MockElementState {
        enabled_after: Some(u64::MAX),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args =
        json!({ "action": "wait", "selector": "#e", "condition": "enabled", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error: Timeout") && out.contains("not enabled"),
        "enabled should bound-timeout: {out}"
    );
}

// --- hidden ----------------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn wait_hidden_succeeds_when_element_becomes_hidden() {
    // Starts visible, becomes hidden after 3 polls.
    let state = MockElementState {
        hidden_after: Some(3),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args =
        json!({ "action": "wait", "selector": "#h", "condition": "hidden", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(
        out, "Element '#h' is hidden",
        "hidden should succeed: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_hidden_times_out_when_element_stays_visible() {
    // hidden_after = MAX means it stays visible forever, so `hidden` never holds.
    let state = MockElementState {
        hidden_after: Some(u64::MAX),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args =
        json!({ "action": "wait", "selector": "#h", "condition": "hidden", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error: Timeout") && out.contains("still visible"),
        "hidden should bound-timeout: {out}"
    );
}

// --- text_contains ---------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn wait_text_contains_succeeds_when_text_appears() {
    let state = MockElementState {
        text: Some("Order complete: thank you".to_string()),
        text_after: Some(2),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args = json!({
        "action": "wait",
        "selector": "#status",
        "condition": "text_contains",
        "text": "complete",
        "_session_id": "s"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("text contains the expected value"),
        "text_contains should succeed: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_text_contains_times_out_when_text_never_matches() {
    // Text is present immediately but never contains the needle.
    let state = MockElementState {
        text: Some("still loading...".to_string()),
        text_after: Some(0),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args = json!({
        "action": "wait",
        "selector": "#status",
        "condition": "text_contains",
        "text": "complete",
        "_session_id": "s"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error: Timeout") && out.contains("did not contain"),
        "text_contains should bound-timeout: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_text_contains_requires_a_needle() {
    let (tool, _b) = wait_tool(MockBackend::new());
    let args = json!({
        "action": "wait",
        "selector": "#status",
        "condition": "text_contains",
        "_session_id": "s"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error") && out.contains("text_contains") && out.contains("text"),
        "text_contains without a needle should error: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_rejects_unknown_condition() {
    let (tool, _b) = wait_tool(MockBackend::new());
    let args = json!({
        "action": "wait",
        "selector": "#x",
        "condition": "exists",
        "_session_id": "s"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error") && out.contains("Invalid wait condition"),
        "unknown condition should error: {out}"
    );
}

#[tokio::test(start_paused = true)]
async fn wait_per_call_timeout_overrides_default_and_clamps() {
    // A per-call timeout_secs of 2 overrides the 4s default; the bounded timeout
    // message reflects the override. (Clamping of an absurd value is asserted at
    // the config level; here we prove the override path is wired.)
    let state = MockElementState {
        present_after: Some(u64::MAX),
        ..Default::default()
    };
    let (tool, _b) = wait_tool(MockBackend::new().with_element_state(state));
    let args = json!({
        "action": "wait",
        "selector": "#never",
        "timeout_secs": 2,
        "_session_id": "s"
    });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("Error: Timeout") && out.contains("2s"),
        "per-call timeout_secs should override the default: {out}"
    );
}

// --- navigate readiness ----------------------------------------------------

#[tokio::test(start_paused = true)]
async fn navigate_waits_for_navigation_readiness_not_a_fixed_sleep() {
    // The navigating mock resolves wait_for_navigation immediately, so navigate
    // returns promptly under the fake clock (no 2s blind sleep).
    let (tool, backend) = wait_tool(MockBackend::new().with_click_navigates());
    let args = json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Navigated to https://example.com/");
    let calls = backend.calls();
    let calls = calls.lock().await;
    // Goto, then a navigation-readiness probe, then the committed-URL revalidation.
    let goto_pos = calls
        .iter()
        .position(|c| matches!(c, MockCall::Goto(_)))
        .expect("goto recorded");
    let nav_pos = calls
        .iter()
        .position(|c| matches!(c, MockCall::WaitForNavigation))
        .expect("nav-readiness probe recorded");
    let url_pos = calls
        .iter()
        .position(|c| matches!(c, MockCall::Url))
        .expect("url revalidation recorded");
    assert!(
        goto_pos < nav_pos && nav_pos < url_pos,
        "navigate must goto -> wait-for-nav -> revalidate url: {calls:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn navigate_bounds_when_navigation_never_settles() {
    // Opt in to never-settling: wait_for_navigation sleeps its full 4s bound.
    // Under the paused clock this completes instantly (fake clock auto-advances),
    // proving navigate is bounded (never hangs) even when `load` never fires,
    // and still revalidates the committed URL afterward.
    let (tool, backend) = wait_tool(MockBackend::new().with_nav_never_settles());
    let args = json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Navigated to https://example.com/");
    assert!(
        calls_contains(&backend, &MockCall::WaitForNavigation).await,
        "navigate must still record the bounded navigation wait"
    );
}

// --- click nav-race --------------------------------------------------------

#[tokio::test(start_paused = true)]
async fn click_that_navigates_waits_for_navigation() {
    // A click that triggers navigation: wait_for_navigation resolves immediately,
    // so the nav branch of the race wins and the probe is recorded.
    let (tool, backend) = wait_tool(MockBackend::new().with_click_navigates());
    let args = json!({ "action": "click", "selector": "#go", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Clicked element '#go'");
    let calls = backend.calls();
    let calls = calls.lock().await;
    let click_pos = calls
        .iter()
        .position(|c| matches!(c, MockCall::Click(_)))
        .expect("click recorded");
    let nav_pos = calls
        .iter()
        .position(|c| matches!(c, MockCall::WaitForNavigation))
        .expect("nav probe recorded");
    assert!(
        click_pos < nav_pos,
        "a navigating click must wait for navigation after clicking: {calls:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn click_that_does_not_navigate_returns_fast_via_settle() {
    // Non-navigating click: wait_for_navigation returns immediately (default fast
    // path — no sleep). The click nav-race still launches the nav probe (recorded),
    // but the result is the fast settle path. Under the paused clock the test
    // completes instantly. Use `with_nav_never_settles()` to exercise the
    // bounded-timeout path specifically.
    let (tool, backend) = wait_tool(MockBackend::new());
    let args = json!({ "action": "click", "selector": "#noop", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Clicked element '#noop'");
    // The race still launched the nav probe (recorded), but the result is the
    // fast settle path — the click succeeded without a popup/navigation.
    assert!(
        calls_contains(&backend, &MockCall::Click("#noop".to_string())).await,
        "click must be recorded"
    );
}

#[tokio::test(start_paused = true)]
async fn click_navrace_still_detects_popup() {
    // A click that opens a popup (target=_blank): the popup is revealed on click;
    // the nav-race settle elapses; popup detection still attributes + reports it.
    let (tool, backend) = wait_tool(MockBackend::new());

    // Prime the session so it has an active tab id (the legitimate opener).
    let prime = json!({ "action": "list_tabs", "_session_id": "s" });
    tool.call(&prime.to_string()).await.unwrap();
    let opener = backend
        .calls()
        .lock()
        .await
        .iter()
        .rev()
        .find_map(|c| match c {
            MockCall::CreatePage(id) => Some(id.clone()),
            _ => None,
        })
        .expect("session's first page id");

    backend
        .script_popup_with_opener(
            "popup-xyz",
            "Popup",
            "https://popup.example/oauth?code=SECRET",
            Some(&opener),
        )
        .await;

    let args = json!({ "action": "click", "selector": "#open", "_session_id": "s" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.contains("opened new tab") && out.contains("popup-xyz"),
        "click nav-race must still detect the spawned popup: {out}"
    );
}

// =============================================================================
// Task 15 — browser diagnostics (console logs + network errors)
// =============================================================================

#[tokio::test]
async fn get_console_logs_returns_scripted_entries_for_active_tab() {
    let backend = MockBackend::new().with_console_logs(vec![
        ("error", "smoke-console-error"),
        ("log", "hello from page"),
    ]);
    let (tool, _backend, _rec) = mock_tool_with(backend);

    let prime = json!({ "action": "list_tabs", "_session_id": "sess-a" });
    tool.call(&prime.to_string()).await.unwrap();

    let out = tool
        .call(&json!({ "action": "get_console_logs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    assert!(out.contains("smoke-console-error"), "{out}");
    assert!(out.contains("[error]"), "{out}");
    assert!(!out.contains("token"), "{out}");
}

#[tokio::test]
async fn get_network_errors_redacts_url_origin() {
    let backend = MockBackend::new().with_network_errors(vec![(
        "https://api.example.com/secret/path?token=SECRET",
        "Document: net::ERR_FAILED",
    )]);
    let (tool, _backend, _rec) = mock_tool_with(backend);

    tool.call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    let out = tool
        .call(&json!({ "action": "get_network_errors", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    assert!(out.contains("https://api.example.com"), "{out}");
    assert!(out.contains("net::ERR_FAILED"), "{out}");
    assert!(!out.contains("SECRET"), "{out}");
    assert!(!out.contains("/secret"), "{out}");
}

#[tokio::test]
async fn get_console_logs_unknown_tab_is_rejected() {
    let (tool, _backend, _rec) = mock_tool();

    tool.call(&json!({ "action": "list_tabs", "_session_id": "sess-a" }).to_string())
        .await
        .unwrap();

    let out = tool
        .call(
            &json!({
                "action": "get_console_logs",
                "tab_id": "not-a-real-tab",
                "_session_id": "sess-a"
            })
            .to_string(),
        )
        .await
        .unwrap();

    assert!(out.contains("Unknown tab"), "{out}");
}
