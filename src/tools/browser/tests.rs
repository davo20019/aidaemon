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
