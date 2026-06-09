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

    let args =
        json!({ "action": "navigate", "url": "https://example.com/", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Navigated to https://example.com/");

    // Lock down the call ordering: ensure_ready, then create_page (this session's
    // first action), then goto. The session registry mints a fresh page id, so
    // assert the create_page record exists and goto follows it.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert_eq!(calls.len(), 3, "expected 3 recorded calls: {calls:?}");
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
}

#[tokio::test]
async fn dispatch_mutation_click_routes_through_backend() {
    let (tool, backend, _rx) = mock_tool();

    let args = json!({ "action": "click", "selector": "#submit", "_session_id": "sess-a" });
    let out = tool.call(&args.to_string()).await.unwrap();
    assert_eq!(out, "Clicked element '#submit'");

    // Exact sequence: ensure_ready -> create_page -> click(selector). Asserts
    // the click is recorded and that we did NOT route through goto/find_element.
    let calls = backend.calls();
    let calls = calls.lock().await;
    assert_eq!(calls.len(), 3, "expected 3 recorded calls: {calls:?}");
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
    let (tool, backend, mut rx) = mock_tool();

    let args = json!({ "action": "screenshot", "selector": "#hero", "_session_id": "sess-a" });
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
async fn screenshot_without_selector_records_full_page_target() {
    let (tool, backend, mut rx) = mock_tool();

    let args = json!({ "action": "screenshot", "_session_id": "sess-a" });
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
