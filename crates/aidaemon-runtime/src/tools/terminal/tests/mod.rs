//! Tests of the terminal tool. Shared helpers live here; the tests are
//! grouped by concern in the child modules.

use super::*;
use crate::tools::traits::NotificationStore;
use crate::tools::traits::ToolSemantics;
use aidaemon_core::testing::ScriptedProvider;
use aidaemon_tools::testing::CallWithFacts;
use aidaemon_tools::testing::ToolCalls as _;

/// The work stores (goals, notifications) on a fresh database at `path`.
async fn work_store(path: &str) -> aidaemon_work::sqlite::SqliteWorkStore {
    aidaemon_work::sqlite::open_for_tests(path).await
}
use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::Duration;

/// The dispatcher's context for an owner's call the kernel approved.
fn kernel_owner_ctx() -> ToolCallContext {
    ToolCallContext {
        kernel_approved: Some(crate::tools::traits::ApprovalScope::Call),
        user_role: Some(aidaemon_tools::semantics::UserRole::Owner),
        ..Default::default()
    }
}

fn owner_trust() -> TerminalTrust {
    TerminalTrust {
        session_id: "telegram:synthetic-owner".to_string(),
        owner: true,
        ..TerminalTrust::default()
    }
}

async fn semantic_gateway(provider: Arc<ScriptedProvider>) -> crate::llm::gateway::SideCallGateway {
    crate::llm::gateway::test_support::gateway_with(provider)
        .await
        .0
}

async fn host_and_sandboxed_terminals() -> (TerminalTool, TerminalTool) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let pool = SqlitePool::connect(&format!("sqlite:{}", db_file.path().display()))
        .await
        .unwrap();
    let host = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec![],
        1,
        1000,
        PermissionMode::Default,
        pool.clone(),
    )
    .await;
    let sandboxed = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec![],
        1,
        1000,
        PermissionMode::Default,
        pool,
    )
    .await
    .with_confinement(aidaemon_tools::TerminalConfinement::Sandbox);
    (host, sandboxed)
}

struct PendingConversationRuntime;

#[async_trait::async_trait]
impl ConversationRuntime for PendingConversationRuntime {
    async fn continue_conversation(
        &self,
        _request: ConversationRequest,
    ) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
        std::future::pending().await
    }

    async fn record_continuation_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

struct AnsweringConversationRuntime;

#[async_trait::async_trait]
impl ConversationRuntime for AnsweringConversationRuntime {
    async fn continue_conversation(
        &self,
        _request: ConversationRequest,
    ) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
        Ok(aidaemon_channels::ports::AgentResponseEnvelope {
            response_id: "r".to_string(),
            task_id: "t".to_string(),
            turn_id: None,
            text: "Drupal is installed.".to_string(),
            referenced_receipts: Vec::new(),
            disposition: Default::default(),
            model: None,
        })
    }

    async fn record_continuation_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Works for longer than the idle limit while beating its heartbeat.
struct BusyConversationRuntime;

#[async_trait::async_trait]
impl ConversationRuntime for BusyConversationRuntime {
    async fn continue_conversation(
        &self,
        request: ConversationRequest,
    ) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
        for _ in 0..10 {
            tokio::time::sleep(Duration::from_millis(300)).await;
            crate::turn::helpers::touch_heartbeat(&request.heartbeat);
        }
        AnsweringConversationRuntime
            .continue_conversation(request)
            .await
    }

    async fn record_continuation_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Watched follow-ups: a runtime that reports one step, then answers.
struct ReportingConversationRuntime(aidaemon_channels::turns::TurnWatch);

#[async_trait::async_trait]
impl ConversationRuntime for ReportingConversationRuntime {
    async fn continue_conversation(
        &self,
        request: ConversationRequest,
    ) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
        if let Some(tx) = request.status_tx.as_ref() {
            let _ = tx.send(StatusUpdate::Thinking(0)).await;
        }
        AnsweringConversationRuntime
            .continue_conversation(request)
            .await
    }

    async fn record_continuation_delivery(
        &self,
        _session_id: &str,
        _delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn turn_watch(&self) -> Option<aidaemon_channels::turns::TurnWatch> {
        Some(self.0.clone())
    }
}

#[derive(Default)]
struct Recorded(std::sync::Mutex<Vec<String>>);

impl aidaemon_channels::turns::TurnObserver for Recorded {
    fn turn_started(&self, _turn: &aidaemon_channels::turns::TurnStart<'_>) {
        self.0.lock().unwrap().push("owner turn".into());
    }
    fn followup_started(&self, session_id: &str, _turn: u64, description: &str) {
        self.0
            .lock()
            .unwrap()
            .push(format!("started {session_id}: {description}"));
    }
    fn turn_status(&self, _session_id: &str, _turn: u64, update: &StatusUpdate) {
        if matches!(update, StatusUpdate::Thinking(_)) {
            self.0.lock().unwrap().push("thinking".into());
        }
    }
    fn turn_finished(&self, turn: &aidaemon_channels::turns::TurnFinish<'_>) {
        self.0.lock().unwrap().push(format!(
            "finished {:?} reply={:?} handed_off={}",
            turn.end, turn.reply, turn.handed_off
        ));
    }
}

fn extract_pid_from_background_message(msg: &str) -> u32 {
    let marker = "pid=";
    let start = msg
        .find(marker)
        .expect("background response should include pid")
        + marker.len();
    let digits: String = msg[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().expect("pid should parse as u32")
}

async fn card_chat_terminal(chat: &aidaemon_channels::testing::CardChat) -> TerminalTool {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_path = db_file.path().display().to_string();
    aidaemon_tools::testing::delete_at_exit(db_file.into_temp_path().keep().unwrap());
    let state = Arc::new(work_store(&db_path).await);
    let (_approval_tx_raw, _approval_rx) = mpsc::channel::<ApprovalRequest>(1);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec!["*".to_string()],
        1,
        4000,
        PermissionMode::Yolo,
        state.pool(),
    )
    .await
    .with_stores(state.clone(), state);
    tool.set_hub(chat.outbound());
    tool
}

async fn make_tool_with_no_perm_prefixes() -> TerminalTool {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    // Empty permanent prefix list — only session approvals exist.
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec![],
        1,
        1000,
        PermissionMode::Default,
        pool,
    )
    .await;
    aidaemon_tools::testing::delete_at_exit(db_file.into_temp_path().keep().unwrap());
    tool
}

// ─────────────────────────────────────────────────────────────────────────
// P2.3 terminal tests
// ─────────────────────────────────────────────────────────────────────────

/// Helper that creates a TerminalTool with an open-ended approval channel
/// (receiver kept alive) so approval requests don't immediately error.
/// Returns (tool, approval_rx) — keep approval_rx alive for the test duration.
async fn make_tool_with_open_approval_channel()
-> (TerminalTool, tokio::sync::mpsc::Receiver<ApprovalRequest>) {
    let db_file = tempfile::NamedTempFile::new().unwrap();
    let db_url = format!("sqlite:{}", db_file.path().display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();
    let (approval_tx_raw, approval_rx) = mpsc::channel::<ApprovalRequest>(4);
    let _approval_tx = crate::approvals::broker::ApprovalBroker::new(approval_tx_raw);
    let tool = TerminalTool::new(
        aidaemon_tools::testing::active_execution_backend(),
        vec![],
        1,
        1000,
        PermissionMode::Default,
        pool,
    )
    .await;
    aidaemon_tools::testing::delete_at_exit(db_file.into_temp_path().keep().unwrap());
    (tool, approval_rx)
}

mod approvals;
mod background;
mod basics;
mod command_checks;
mod continuation;
mod receipts;
mod sandbox;
mod trust;

mod call_effects_tests;
mod low_disk_tests;
mod policy_dump_probe;
