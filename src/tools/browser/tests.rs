//! Smoke tests proving `BrowserTool` dispatch works against `MockBackend`
//! (no real Chrome). Covers observation and mutation actions and locks down the
//! preservation-sensitive dispatch paths (selector vs. no-selector routing,
//! `execute_js` null/undefined handling, screenshot target selection) that
//! later tasks will touch.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use super::backend::{MockBackend, MockCall};
use super::BrowserTool;
use crate::traits::Tool;
use crate::types::{MediaKind, MediaMessage};

fn mock_tool() -> (
    BrowserTool,
    Arc<MockBackend>,
    mpsc::Receiver<crate::types::MediaMessage>,
) {
    mock_tool_with(MockBackend::new())
}

fn mock_tool_with(
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

    let args = json!({ "action": "navigate", "url": "https://example.com/" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Navigated to https://example.com/");

    // Lock down the call ordering: ensure_ready, then current_page, then goto.
    // Task 3 will add session/ordering logic; an exact-sequence anchor catches
    // regressions there.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert_eq!(
        *calls,
        vec![
            MockCall::EnsureReady,
            MockCall::CurrentPage,
            MockCall::Goto("https://example.com/".to_string()),
        ],
        "navigate should ensure_ready -> current_page -> goto in order: {calls:?}"
    );
}

#[tokio::test]
async fn dispatch_mutation_click_routes_through_backend() {
    let (tool, backend, _rx) = mock_tool();

    let args = json!({ "action": "click", "selector": "#submit" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Clicked element '#submit'");

    // Exact sequence: ensure_ready -> current_page -> click(selector). Asserts
    // the click is recorded and that we did NOT route through goto/find_element.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert_eq!(
        *calls,
        vec![
            MockCall::EnsureReady,
            MockCall::CurrentPage,
            MockCall::Click("#submit".to_string()),
        ],
        "click should ensure_ready -> current_page -> click in order: {calls:?}"
    );
}

#[tokio::test]
async fn get_text_with_selector_routes_through_inner_text() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("hello inner"));

    let args = json!({ "action": "get_text", "selector": "#headline" });
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

#[tokio::test]
async fn get_text_without_selector_routes_through_body_text() {
    let (tool, backend, _rx) = mock_tool_with(MockBackend::new().with_text_result("hello body"));

    let args = json!({ "action": "get_text" });
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

    let args = json!({ "action": "execute_js", "script": "return null;" });
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

    let args = json!({ "action": "execute_js", "script": "void 0;" });
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
    let (tool, backend, mut rx) = mock_tool();

    let args = json!({ "action": "screenshot", "selector": "#hero" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.starts_with("Screenshot taken and sent to chat."),
        "screenshot should report success: {out}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Screenshot(Some("#hero".to_string()))),
        "with a selector, screenshot must target the element: {calls:?}"
    );
    assert!(
        !calls.contains(&MockCall::Screenshot(None)),
        "with a selector, screenshot must NOT capture the full page: {calls:?}"
    );

    // The screenshot bytes are pushed onto the media channel as a Photo.
    let media: MediaMessage = rx.try_recv().expect("a media message should be sent");
    assert!(
        matches!(media.kind, MediaKind::Photo { .. }),
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
    let args = json!({ "action": "fill", "selector": "#password", "value": value });
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
        "value": "new@example.com"
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

        let args = json!({ "action": "fill", "selector": selector, "value": value });
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
async fn screenshot_without_selector_records_full_page_target() {
    let (tool, backend, mut rx) = mock_tool();

    let args = json!({ "action": "screenshot" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert!(
        out.starts_with("Screenshot taken and sent to chat."),
        "screenshot should report success: {out}"
    );

    let calls = backend.calls();
    let calls = calls.lock().await;
    assert!(
        calls.contains(&MockCall::Screenshot(None)),
        "without a selector, screenshot must capture the full page: {calls:?}"
    );
    assert!(
        !calls
            .iter()
            .any(|c| matches!(c, MockCall::Screenshot(Some(_)))),
        "without a selector, screenshot must NOT target an element: {calls:?}"
    );

    let media: MediaMessage = rx.try_recv().expect("a media message should be sent");
    assert!(
        matches!(media.kind, MediaKind::Photo { .. }),
        "screenshot media must be a Photo"
    );
}
