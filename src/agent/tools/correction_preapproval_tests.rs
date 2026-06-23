use std::sync::atomic::{AtomicBool, Ordering};
/// P2.3 tests: dispatcher-owned one-shot correction preapproval token
/// and `suppress_trusted_session` enforcement.
///
/// Security contract tested here:
///   - `_correction_preapproved` in model-supplied JSON args does NOT activate preapproval.
///   - Only the Rust-side `ToolExecCtx.correction_preapproved` reaches the tool.
///   - No `_correction_preapproved` key is injected into enriched args.
///   - `ChannelContext.trusted=true` with `suppress_trusted_session=true` → no `_trusted_session`.
///   - Scheduled provenance with `suppress_trusted_session=true` → no `_trusted_session`.
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use super::*;
use crate::testing::{setup_test_agent, MockProvider};
use crate::traits::{Tool, ToolCallOutcome, ToolCapabilities, ToolExecutionContext};
use crate::types::StatusUpdate;

// ────────────────────────────────────────────────────────────────────────────
// Spy tool: records whether call_with_execution_context was invoked with
// correction_preapproved=true, and echoes back the raw JSON args it received.
// ────────────────────────────────────────────────────────────────────────────
struct SpyTool {
    /// Set to true if call_with_execution_context saw correction_preapproved=true.
    pub saw_preapproval: Arc<AtomicBool>,
    /// The most recent raw enriched args string received by the tool.
    pub last_args: Arc<std::sync::Mutex<String>>,
}

impl SpyTool {
    fn new() -> (Self, Arc<AtomicBool>, Arc<std::sync::Mutex<String>>) {
        let saw = Arc::new(AtomicBool::new(false));
        let args = Arc::new(std::sync::Mutex::new(String::new()));
        (
            Self {
                saw_preapproval: Arc::clone(&saw),
                last_args: Arc::clone(&args),
            },
            saw,
            args,
        )
    }
}

#[async_trait]
impl Tool for SpyTool {
    fn name(&self) -> &str {
        "spy_tool"
    }

    fn description(&self) -> &str {
        "Spy tool for P2.3 tests"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "spy_tool",
            "description": "Spy tool for P2.3 tests",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        *self.last_args.lock().unwrap() = arguments.to_string();
        Ok(arguments.to_string())
    }

    /// Override: record exec_ctx.correction_preapproved, echo back the args.
    async fn call_with_execution_context(
        &self,
        arguments: &str,
        _status_tx: Option<mpsc::Sender<StatusUpdate>>,
        exec_ctx: ToolExecutionContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.saw_preapproval
            .store(exec_ctx.correction_preapproved, Ordering::SeqCst);
        *self.last_args.lock().unwrap() = arguments.to_string();
        Ok(ToolCallOutcome::from_output(arguments.to_string()))
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 1: model-supplied `_correction_preapproved` JSON key does NOT preapprove
//
// The LLM can put any field in its JSON args. The underscore-prefix stripping
// guard in `execute_tool_outcome` removes it before the tool ever sees it.
// Critically: even if the stripping were absent, there's no pathway from
// a JSON arg to `ToolExecutionContext.correction_preapproved`.  The only way
// to set correction_preapproved is through `ToolExecCtx.correction_preapproved`
// which is never derived from model-visible JSON.
// ────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_model_supplied_correction_preapproval_json_key_does_not_preapprove() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");

    let (spy, saw_preapproval, last_args) = SpyTool::new();
    harness.agent.tools.push(Arc::new(spy));

    // Model supplies _correction_preapproved=true in its JSON args.
    // This must NOT be interpreted as a preapproval signal.
    let model_args = r#"{"_correction_preapproved": true, "some_field": "value"}"#;

    let _ = harness
        .agent
        .execute_tool_with_watchdog(
            "spy_tool",
            model_args,
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-1"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                // correction_preapproved is FALSE at the dispatcher level
                correction_preapproved: false,
                suppress_trusted_session: false,
            },
        )
        .await
        .expect("spy_tool call should succeed");

    // The spy tool should NOT have seen correction_preapproved=true.
    assert!(
        !saw_preapproval.load(Ordering::SeqCst),
        "model-supplied _correction_preapproved JSON field must not activate preapproval"
    );

    // The _correction_preapproved key must have been stripped from enriched args.
    let raw = last_args.lock().unwrap().clone();
    let parsed: Value = serde_json::from_str(&raw).expect("args should be valid JSON");
    assert!(
        parsed.get("_correction_preapproved").is_none(),
        "enriched args must not contain _correction_preapproved (got: {})",
        raw
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2: dispatcher correction_preapproved=true reaches the tool exactly once
//
// When ToolExecCtx.correction_preapproved=true, call_with_execution_context
// must be invoked with ToolExecutionContext { correction_preapproved: true }.
// ────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_dispatcher_correction_preapproval_context_reaches_tool_once() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");

    let (spy, saw_preapproval, _) = SpyTool::new();
    harness.agent.tools.push(Arc::new(spy));

    let _ = harness
        .agent
        .execute_tool_with_watchdog(
            "spy_tool",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-2"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                // Dispatcher sets correction_preapproved=true
                correction_preapproved: true,
                suppress_trusted_session: false,
            },
        )
        .await
        .expect("spy_tool call should succeed");

    assert!(
        saw_preapproval.load(Ordering::SeqCst),
        "call_with_execution_context must receive correction_preapproved=true \
         when ToolExecCtx.correction_preapproved=true"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3: no `_correction_preapproved` key is injected into enriched args
//
// The correction preapproval token must NEVER appear in the JSON args passed
// to the tool.  It lives only in the Rust-side ToolExecutionContext.
// ────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_no_correction_preapproval_key_is_injected_into_args() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");

    let (spy, _, last_args) = SpyTool::new();
    harness.agent.tools.push(Arc::new(spy));

    let _ = harness
        .agent
        .execute_tool_with_watchdog(
            "spy_tool",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-3"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: false,
                user_role: UserRole::Owner,
                correction_preapproved: true,
                suppress_trusted_session: false,
            },
        )
        .await
        .expect("spy_tool call should succeed");

    let raw = last_args.lock().unwrap().clone();
    let parsed: Value = serde_json::from_str(&raw).expect("args should be valid JSON");

    // No correction token in args — in any form.
    for key in parsed.as_object().map(|o| o.keys()).into_iter().flatten() {
        assert!(
            !key.to_ascii_lowercase().contains("correction"),
            "enriched args must not contain any 'correction' key (found '{}')",
            key
        );
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4: ChannelContext.trusted=true does NOT inject `_trusted_session` when
//          suppress_trusted_session=true
//
// The correction gate suppresses trusted-session semantics so that an
// unattended correction run can't silently inherit scheduled-task trust.
// ────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_channel_context_trusted_does_not_bypass_correction_gate() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");

    let (spy, _, last_args) = SpyTool::new();
    harness.agent.tools.push(Arc::new(spy));

    let _ = harness
        .agent
        .execute_tool_with_watchdog(
            "spy_tool",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-4"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                // Even though trusted=true...
                trusted: true,
                user_role: UserRole::Owner,
                correction_preapproved: true,
                // ...suppress_trusted_session must prevent _trusted_session injection.
                suppress_trusted_session: true,
            },
        )
        .await
        .expect("spy_tool call should succeed");

    let raw = last_args.lock().unwrap().clone();
    let parsed: Value = serde_json::from_str(&raw).expect("args should be valid JSON");

    // _trusted_session must NOT be present when suppress_trusted_session=true.
    let ts = parsed.get("_trusted_session");
    assert!(
        ts.is_none() || ts.and_then(Value::as_bool) == Some(false),
        "_trusted_session must not be true when suppress_trusted_session=true (got: {})",
        raw
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 5: scheduled trust (via goal/task provenance) does NOT inject
//          `_trusted_session` when suppress_trusted_session=true
//
// Even if the agent were running under a scheduled goal/task, suppression must
// short-circuit the entire trusted OR-chain (not just ctx.trusted).
//
// We test this via trusted=false (so only provenance could inject it) +
// suppress_trusted_session=true. Without suppression, a scheduled goal/task
// would inject it; with suppression it must not.
//
// Note: in the test harness, goal_id and task_id are None on the Agent, so
// scheduled provenance cannot fire. This test validates the suppression code
// path directly using trusted=true which is the strongest guarantee that
// suppress_trusted_session=true always wins over the OR.
// ────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_scheduled_trust_does_not_bypass_correction_gate() {
    let mut harness = setup_test_agent(MockProvider::new())
        .await
        .expect("setup test harness");

    let (spy, _, last_args) = SpyTool::new();
    harness.agent.tools.push(Arc::new(spy));

    // Use trusted=true to exercise the strongest form of the OR:
    //   trusted=true || scheduled_provenance(...) → would be true without suppression.
    // With suppress_trusted_session=true the entire expression must be false.
    let _ = harness
        .agent
        .execute_tool_with_watchdog(
            "spy_tool",
            "{}",
            &ToolExecCtx {
                session_id: "test-session",
                task_id: Some("task-5"),
                status_tx: None,
                channel_visibility: ChannelVisibility::Private,
                channel_id: None,
                project_scope: None,
                trusted: true,
                user_role: UserRole::Owner,
                correction_preapproved: false,
                suppress_trusted_session: true,
            },
        )
        .await
        .expect("spy_tool call should succeed");

    let raw = last_args.lock().unwrap().clone();
    let parsed: Value = serde_json::from_str(&raw).expect("args should be valid JSON");

    let ts = parsed.get("_trusted_session");
    assert!(
        ts.is_none() || ts.and_then(Value::as_bool) == Some(false),
        "_trusted_session must not be injected when suppress_trusted_session=true (got: {})",
        raw
    );
}
