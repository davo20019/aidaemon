//! A real Client Gateway on a scripted agent, for trying the web client and
//! the TUI by hand or from a browser test, with synthetic data only.
//!
//! `cargo run -p aidaemon-client-gateway --example demo_gateway [-- PORT]`
//!
//! It prints a pairing link (approved by itself) and a local token file for
//! the TUI. The agent answers "Done: ..." and streams a status first; a
//! message with "approve" asks for an approval and waits; "what's waiting"
//! moves the sender's view to the oldest waiting approval; "open the plan"
//! opens the synthetic plan session (both through the agent-driven view).

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use aidaemon_channels::hub::{ApprovalRequest, ChannelHub, SessionChannelStore};
use aidaemon_channels::ports::{
    AgentIngress, AgentResponseEnvelope, ChannelAgentRuntime, InboundMessageRequest,
};
use aidaemon_channels::turns::{TurnDeps, TurnWatch};
use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps};
use aidaemon_client_protocol::{HistoryMessage, SessionSummary, ViewAction};
use aidaemon_store_sqlite::SqliteClientEventLog;
use aidaemon_tools::command_risk::{PermissionMode, RiskLevel};
use aidaemon_tools::semantics::{ApprovalKind, ApprovalResponse};
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

static GATEWAY: OnceLock<Arc<ClientGateway>> = OnceLock::new();

struct Agent {
    approvals: mpsc::Sender<ApprovalRequest>,
}

#[async_trait]
impl AgentIngress for Agent {
    async fn handle_inbound_message(
        &self,
        request: InboundMessageRequest,
    ) -> anyhow::Result<AgentResponseEnvelope> {
        let text = request.user_text.to_lowercase();
        if let Some(status) = request.status_tx.as_ref() {
            let _ = status
                .send(StatusUpdate::ToolStart {
                    name: "terminal".into(),
                    summary: "reading the synthetic notes".into(),
                    kind: aidaemon_core::card::ActivityKind::Read,
                })
                .await;
            tokio::time::sleep(Duration::from_millis(600)).await;
        }
        let gateway = GATEWAY.get().cloned();
        let reply = if text.contains("approve") {
            let (response_tx, response_rx) = oneshot::channel();
            self.approvals
                .send(ApprovalRequest {
                    command: "report --publish --to h.example.invalid".into(),
                    session_id: request.session_id.clone(),
                    risk_level: RiskLevel::High,
                    warnings: vec!["Sends the synthetic report outside this computer.".into()],
                    permission_mode: PermissionMode::Default,
                    response_tx,
                    kind: ApprovalKind::Command,
                    card: None,
                    expires_at_ms: None,
                })
                .await?;
            match response_rx.await {
                Ok(ApprovalResponse::Deny) => "Not published.".to_string(),
                Ok(response) => format!("Published ({response:?})."),
                Err(_) => "No answer.".to_string(),
            }
        } else if text.contains("waiting") {
            let gateway = gateway.expect("started");
            let (_, snapshot) = gateway.snapshot().await;
            match snapshot.pending_approvals.first() {
                Some(approval) => {
                    let moved = gateway.views().show(
                        &request.session_id,
                        ViewAction::ShowApproval {
                            approval_id: approval.id.clone(),
                        },
                        "You asked what is waiting",
                    );
                    format!("One approval is waiting: {} ({moved:?})", approval.summary)
                }
                None => "Nothing is waiting.".to_string(),
            }
        } else if text.contains("open the plan") {
            let moved = GATEWAY.get().expect("started").views().show(
                &request.session_id,
                ViewAction::OpenSession {
                    session_id: "app:plan".into(),
                },
                "You asked for the plan",
            );
            format!("Opened the plan ({moved:?}).")
        } else {
            format!("Done: {}", request.user_text)
        };
        Ok(AgentResponseEnvelope {
            response_id: "r".into(),
            task_id: "t".into(),
            turn_id: None,
            text: reply,
            referenced_receipts: Vec::new(),
            disposition: Default::default(),
        })
    }
}

#[async_trait]
impl ChannelAgentRuntime for Agent {
    async fn terminalize_supervised_task(
        &self,
        _: &str,
        _: aidaemon_events::TaskStatus,
        _: String,
    ) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
    async fn cancel_active_finite_work_for_session(&self, _: &str) -> Vec<String> {
        Vec::new()
    }
    async fn current_model(&self) -> String {
        "scripted".into()
    }
    async fn context_debug_settings(
        &self,
        _: &str,
        _: &str,
    ) -> (bool, usize, usize, usize, Option<i64>) {
        (false, 0, 0, 0, None)
    }
    async fn set_model(&self, _: String) {}
    async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        Ok(Vec::new())
    }
    async fn clear_model_override(&self) {}
    async fn clear_session_context(&self, _: &str) -> anyhow::Result<()> {
        Ok(())
    }
    async fn clear_session(&self, _: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

struct NoSessions;
impl SessionChannelStore for NoSessions {}

struct Host;

#[async_trait]
impl ClientHost for Host {
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        Ok(vec![
            SessionSummary {
                id: "app:plan".into(),
                channel: "app".into(),
                title: Some("Plan for Acme Corp".into()),
                last_activity_ms: Some(now_ms() - 3_600_000),
                writable: true,
            },
            SessionSummary {
                id: "telegram:synthetic-user-1".into(),
                channel: "telegram".into(),
                title: Some("Alice Rivera on Telegram".into()),
                last_activity_ms: Some(now_ms() - 26 * 3_600_000),
                writable: false,
            },
        ])
    }
    async fn history(&self, session_id: &str, _: usize) -> anyhow::Result<Vec<HistoryMessage>> {
        let lines: &[(&str, &str)] = match session_id {
            "app:plan" => &[
                ("user", "Draft next week's plan for Acme Corp"),
                (
                    "assistant",
                    "Monday: review the synthetic report. Tuesday: the demo.",
                ),
            ],
            _ => &[("user", "What's on today?"), ("assistant", "Two meetings.")],
        };
        let count = lines.len() as u64;
        Ok(lines
            .iter()
            .enumerate()
            .map(|(n, (role, text))| HistoryMessage {
                id: format!("{session_id}-m{n}"),
                role: role.to_string(),
                text: text.to_string(),
                // A minute apart, ending an hour ago.
                at_ms: now_ms() - 3_600_000 - (count - n as u64) * 60_000,
            })
            .collect())
    }
    async fn slash(&self, _: &str, command: &str, _: &str) -> Option<String> {
        match command {
            "devices" => Some("Paired devices:\n• Synthetic Phone (Web)".into()),
            "clear" => Some("Context cleared.".into()),
            _ => None,
        }
    }
    async fn system(&self) -> Option<aidaemon_client_protocol::SystemStatus> {
        use aidaemon_client_protocol::*;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or_default();
        Some(SystemStatus {
            version: env!("CARGO_PKG_VERSION").into(),
            uptime_secs: 5_400,
            model: Some("model-a".into()),
            fallback_models: vec!["model-b".into()],
            usage: (0..7)
                .map(|d| UsageDay {
                    day: format!("2026-01-{:02}", 7 - d),
                    model: "model-a".into(),
                    input_tokens: 40_000 + d * 9_000,
                    output_tokens: 3_000 + d * 400,
                    requests: 20 + d,
                })
                .collect(),
            model_calls: Some(ModelCalls {
                total: 182,
                failed: 3,
                fell_back: 1,
                p50_ms: 1_400,
                p95_ms: 5_200,
                max_ms: 9_800,
            }),
            schedules: vec![ScheduleInfo {
                goal_id: "g-1".into(),
                description: "Send the weekly summary for Acme Corp".into(),
                status: "active".into(),
                schedule: "every monday at 9".into(),
                paused: false,
                one_shot: false,
                next_run_ms: Some(now + 3 * 3_600_000),
                last_run_ms: Some(now - 4 * 86_400_000),
            }],
            jobs: vec![
                JobInfo {
                    name: "consolidation".into(),
                    every_secs: 3_600,
                    running: false,
                    consecutive_failures: 0,
                    last_success_ms: Some(now - 600_000),
                    last_error: None,
                },
                JobInfo {
                    name: "embeddings".into(),
                    every_secs: 300,
                    running: false,
                    consecutive_failures: 2,
                    last_success_ms: Some(now - 1_800_000),
                    last_error: Some("The embedding model is still downloading".into()),
                },
            ],
            queues: vec![QueueInfo {
                name: "triggers".into(),
                depth: 0,
                capacity: 64,
                dropped: 0,
                failed: 0,
                overloaded: false,
            }],
            checks: vec![HealthCheckInfo {
                name: "Acme site".into(),
                target: "https://h.example.invalid".into(),
                status: CheckStatus::Healthy,
                paused: false,
                consecutive_failures: 0,
                latency_ms: Some(140),
                checked_ms: Some(now - 120_000),
                error: None,
            }],
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|p| p.parse().ok())
        .unwrap_or(8799);
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .min_connections(1)
        .idle_timeout(None)
        .max_lifetime(None)
        .connect("sqlite::memory:")
        .await?;
    let log = SqliteClientEventLog::open(pool).await?;
    let (approvals_tx, approvals_rx) = mpsc::channel(16);
    let agent = Arc::new(Agent {
        approvals: approvals_tx.clone(),
    });
    let session_map = aidaemon_channels::hub::SessionMap::default();
    let hub = Arc::new(ChannelHub::new(Vec::new(), session_map.clone()));
    tokio::spawn(hub.clone().approval_listener(approvals_rx));
    let dir = tempfile::tempdir()?;
    let token_file = dir.path().join("client-token.sha256");
    let token = aidaemon_client_gateway::token::mint(&token_file)?;
    let token_copy = dir.path().join("client-token");
    std::fs::write(&token_copy, &token)?;
    let gateway = ClientGateway::start(GatewayDeps {
        turns: TurnDeps {
            agent,
            session_map,
            sessions: Arc::new(NoSessions),
            task_registry: Arc::new(aidaemon_channels::tasks::TaskRegistry::new(50)),
            watchdog_stale_threshold_secs: 0,
            watch: TurnWatch::default(),
        },
        hub,
        host: Arc::new(Host),
        log,
        config: ClientGatewayConfig {
            enabled: true,
            // The nsh app's web build (`nsh package --target web` in
            // clients/app), served at /client/v1/next/.
            app_dir: std::env::var("AIDAEMON_APP_DIR").ok(),
            ..ClientGatewayConfig::default()
        },
        token_file,
        approvals: Some(approvals_tx),
        endpoints: aidaemon_core::EndpointSource::fixed(Vec::new()),
    })
    .await?;
    let _ = GATEWAY.set(gateway.clone());
    let offer = gateway
        .pairing()
        .offer(None)
        .await
        .map_err(|e| anyhow::anyhow!(e.message))?;
    {
        let gateway = gateway.clone();
        let offer_id = offer.offer_id.clone();
        tokio::spawn(async move {
            // The owner's confirmation, given as soon as a device redeems.
            loop {
                if gateway.pairing().decide(&offer_id, true).await.is_ok() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        });
    }
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!(
        "pair: http://127.0.0.1:{port}/client/v1/app#offer={}&code={}&fp={}",
        offer.offer_id, offer.code, offer.fingerprint
    );
    println!(
        "tui:  aidaemon-tui --url ws://127.0.0.1:{port}/client/v1/ws --token-file {}",
        token_copy.display()
    );
    axum::serve(listener, aidaemon_client_gateway::router(gateway)).await?;
    Ok(())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}
