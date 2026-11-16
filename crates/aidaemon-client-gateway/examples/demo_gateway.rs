//! A real Client Gateway on a scripted agent, for trying the web client and
//! the TUI by hand or from a browser test, with synthetic data only.
//!
//! `cargo run -p aidaemon-client-gateway --example demo_gateway [-- PORT]`
//!
//! It prints a pairing link (approved by itself) and a local token file for
//! the TUI. The agent answers "Done: ..." and streams a status first; a
//! message with "approve" asks for an approval and waits; "what's waiting"
//! moves the sender's view to the oldest waiting approval; "open the plan"
//! opens the synthetic plan session (both through the agent-driven view);
//! "markdown" answers with a longer formatted reply.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use aidaemon_channels::hub::{ApprovalRequest, ChannelHub, SessionChannelStore};
use aidaemon_channels::ports::{
    AgentIngress, AgentResponseEnvelope, ChannelAgentRuntime, InboundMessageRequest,
};
use aidaemon_channels::turns::{TurnDeps, TurnWatch};
use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps};
use aidaemon_client_protocol::{
    Billing, CheckedKey, ConnectedProvider, CredentialState, FileChange, FileStatus,
    HistoryMessage, MemoryItem, ProjectSummary, ProviderChoice, ProviderOption, RepoChanges,
    SearchHit, SessionChanges, SessionSummary, Settings, SettingsChange, SignIn, SlashCommand,
    ViewAction,
};
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
            // "slow" keeps it working a while, to watch a working row.
            let pause = if text.contains("slow") { 30_000 } else { 600 };
            tokio::time::sleep(Duration::from_millis(pause)).await;
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
                Ok(ApprovalResponse::AllowOnce) => "Published the synthetic report.".to_string(),
                Ok(_) => {
                    "Published the synthetic report. Publishing reports is allowed from now on."
                        .to_string()
                }
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
        } else if text.contains("markdown") {
            "## Plan for Acme Corp\n\nBuilt `intervals` in `~/projects/acme-eval/cache-test` with `merge_overlapping` and a `pytest-asyncio` check. Three things this week:\n\n1. **Review** the synthetic report with Jordan Lee.\n2. Ship the demo build to `staging`.\n3. Draft the follow-up for Alice Rivera.\n\nThe build command:\n\n```sh\ncargo build --release --features browser\n```\n\n> The demo needs the staging key; it is not in the repo.\n\nDone by **Friday**, if the report lands on Tuesday.".to_string()
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

/// Names and archive flags the owner set, by session id.
type Marks = std::sync::Mutex<std::collections::HashMap<String, (Option<String>, bool)>>;

fn marks() -> &'static Marks {
    static MARKS: OnceLock<Marks> = OnceLock::new();
    MARKS.get_or_init(Default::default)
}

/// First-run setup, tried with `AIDAEMON_DEMO_SETUP=1` (a model is found
/// here) or `=none` (none is): `None` once it is done.
fn demo_setup() -> &'static std::sync::Mutex<Option<aidaemon_client_protocol::SetupState>> {
    use aidaemon_client_protocol::{Billing, FoundModel, ProviderOption, SetupState, SetupStep};
    static SETUP: OnceLock<std::sync::Mutex<Option<SetupState>>> = OnceLock::new();
    SETUP.get_or_init(|| {
        let mode = std::env::var("AIDAEMON_DEMO_SETUP").unwrap_or_default();
        let provider = |preset: &str, name: &str, billing, needs_key| ProviderOption {
            preset: preset.into(),
            name: name.into(),
            billing,
            needs_key,
            key_url: String::new(),
            needs_base_url: false,
            model: String::new(),
        };
        std::sync::Mutex::new((!mode.is_empty()).then(|| SetupState {
            step: SetupStep::Model,
            found: if mode == "none" {
                Vec::new()
            } else {
                vec![FoundModel {
                    preset: "openai_chatgpt".into(),
                    name: "OpenAI (ChatGPT subscription) · gpt-synthetic".into(),
                    source: "ChatGPT sign-in on this computer".into(),
                    billing: Billing::Plan,
                    works: true,
                    detail: String::new(),
                }]
            },
            providers: vec![
                provider(
                    "openai_chatgpt",
                    "OpenAI (ChatGPT subscription)",
                    Billing::Plan,
                    false,
                ),
                provider("anthropic_native", "Anthropic", Billing::PerToken, true),
                provider("openai", "OpenAI", Billing::PerToken, true),
                provider("google_ai_studio", "Google Gemini", Billing::PerToken, true),
                provider("openrouter", "OpenRouter", Billing::PerToken, true),
                provider("xai", "xAI (Grok)", Billing::PerToken, true),
                provider("deepseek", "DeepSeek", Billing::PerToken, true),
                provider("moonshot", "Moonshot AI (Kimi)", Billing::PerToken, true),
                provider("minimax", "MiniMax", Billing::PerToken, true),
                provider(
                    "cloudflare_gateway",
                    "Cloudflare AI Gateway",
                    Billing::PerToken,
                    true,
                ),
                provider("ollama", "Ollama (local)", Billing::Local, false),
                provider("llamacpp", "llama.cpp (local)", Billing::Local, false),
            ],
            ..SetupState::default()
        }))
    })
}

fn demo_channel_step() -> aidaemon_client_protocol::SetupState {
    use aidaemon_client_protocol::{ChannelOption, SetupState, SetupStep};
    let chat = |platform: &str, name: &str, needs_app_token| ChannelOption {
        platform: platform.into(),
        name: name.into(),
        help_url: "https://example.invalid/make-a-bot".into(),
        needs_app_token,
    };
    SetupState {
        step: SetupStep::Channel,
        model: Some("OpenAI (ChatGPT subscription) · gpt-synthetic".into()),
        channels: vec![
            chat("telegram", "Telegram", false),
            chat("slack", "Slack", true),
            chat("discord", "Discord", false),
        ],
        ..SetupState::default()
    }
}

#[async_trait]
impl ClientHost for Host {
    async fn setup(&self) -> Option<aidaemon_client_protocol::SetupState> {
        demo_setup().lock().unwrap().clone()
    }
    async fn setup_step(
        &self,
        action: aidaemon_client_protocol::SetupAction,
    ) -> Result<aidaemon_client_protocol::SetupState, String> {
        use aidaemon_client_protocol::{SetupAction, SetupState, SetupStep};
        let next = match action {
            SetupAction::UseFound { .. } | SetupAction::ConnectModel { .. } => {
                // The daemon would restart with the model; then the chat step.
                tokio::spawn(async {
                    tokio::time::sleep(Duration::from_millis(1500)).await;
                    *demo_setup().lock().unwrap() = Some(demo_channel_step());
                    if let Some(gateway) = GATEWAY.get() {
                        gateway.setup_updated(demo_channel_step());
                    }
                });
                SetupState {
                    step: SetupStep::Starting,
                    model: Some("OpenAI (ChatGPT subscription) · gpt-synthetic".into()),
                    ..SetupState::default()
                }
            }
            // A key or a local server "checks out": its models to pick from.
            SetupAction::CheckKey { provider, .. } => {
                let mut state = demo_setup()
                    .lock()
                    .unwrap()
                    .clone()
                    .ok_or("The demo is already set up")?;
                state.checked = Some(aidaemon_client_protocol::CheckedKey {
                    preset: provider.preset,
                    models: vec!["synthetic-small:4b".into(), "synthetic-large:70b".into()],
                    suggested: Some("synthetic-large:70b".into()),
                });
                state
            }
            SetupAction::Detect => demo_channel_step(),
            SetupAction::SkipChannel | SetupAction::Finish => {
                static GREETED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                demo_setup().lock().unwrap().take();
                if !GREETED.swap(true, std::sync::atomic::Ordering::Relaxed)
                    && let Some(gateway) = GATEWAY.get()
                {
                    gateway
                        .post_greeting(
                            &aidaemon_client_gateway::new_app_session_id(),
                            "Welcome",
                            "Hi, I'm AIdaemon. I research, write, organize files, run commands \
                             and keep an eye on things for you on a schedule. I run on this \
                             computer. While it sleeps, I'm offline.\n\nI read and look things \
                             up on my own. Before anything that sends, posts, deletes or spends, \
                             I ask you first.\n\nTry one of these, or ask for anything.",
                            vec![
                                "Look through my Downloads folder and suggest what to clean up"
                                    .into(),
                                "Find today's three biggest tech stories and summarize each in \
                                 two lines"
                                    .into(),
                                "Every weekday at 8, send me a short summary of news on a topic \
                                 I pick"
                                    .into(),
                            ],
                        )
                        .await;
                }
                return Ok(SetupState {
                    step: SetupStep::Done,
                    ..SetupState::default()
                });
            }
            _ => return Err("The demo connects no chat bots".into()),
        };
        Ok(next)
    }
    async fn rename_session(&self, id: &str, title: Option<&str>) -> Result<(), String> {
        marks().lock().unwrap().entry(id.into()).or_default().0 = title.map(str::to_owned);
        Ok(())
    }
    async fn archive_session(&self, id: &str, archived: bool) -> Result<(), String> {
        marks().lock().unwrap().entry(id.into()).or_default().1 = archived;
        Ok(())
    }
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let marks = marks().lock().unwrap().clone();
        let marked = move |mut s: SessionSummary| {
            if let Some((title, archived)) = marks.get(&s.id) {
                if title.is_some() {
                    s.title = title.clone();
                }
                s.archived = *archived;
            }
            s
        };
        Ok([
            SessionSummary {
                archived: false,
                id: "app:plan".into(),
                channel: "app".into(),
                title: Some("Plan for Acme Corp".into()),
                last_activity_ms: Some(now_ms() - 3_600_000),
                writable: true,
                project_id: None,
                group: false,
            },
            SessionSummary {
                archived: false,
                id: "telegram:synthetic-user-1".into(),
                channel: "telegram:synthetic_coding_bot".into(),
                title: Some("Fix the contact form".into()),
                last_activity_ms: Some(now_ms() - 26 * 3_600_000),
                writable: true,
                project_id: Some("project-acme".into()),
                group: false,
            },
            SessionSummary {
                archived: false,
                id: "telegram:synthetic-group-1".into(),
                channel: "telegram:synthetic_coding_bot".into(),
                title: Some("Release notes for the team".into()),
                last_activity_ms: Some(now_ms() - 30 * 3_600_000),
                writable: true,
                project_id: None,
                group: true,
            },
        ]
        .into_iter()
        // A busy day and older days, so the sidebar's long lists show.
        .chain((1..=12).map(|n| SessionSummary {
            archived: false,
            id: format!("app:busy-{n}"),
            channel: "app".into(),
            title: Some(format!("Synthetic task {n} for Acme Corp")),
            last_activity_ms: Some(now_ms() - n * 240_000),
            writable: true,
            project_id: None,
            group: false,
        }))
        .chain((3..=8).map(|days| SessionSummary {
            archived: false,
            id: format!("app:older-{days}"),
            channel: "app".into(),
            title: Some(format!("Notes from {days} days ago")),
            last_activity_ms: Some(now_ms() - days * 86_400_000),
            writable: true,
            project_id: None,
            group: false,
        }))
        .map(marked)
        .collect())
    }
    async fn automations(
        &self,
    ) -> anyhow::Result<Vec<aidaemon_client_protocol::AutomationSummary>> {
        use aidaemon_client_protocol::{AutomationKind, AutomationSummary};
        Ok(vec![
            AutomationSummary {
                id: "goal-digest".into(),
                kind: AutomationKind::Scheduled,
                name: Some("Inbox digest".into()),
                what: "Summarize the synthetic inbox and send the highlights".into(),
                how_often: "Every weekday at 9:00 AM ET".into(),
                status: "active".into(),
                next_run_ms: Some(now_ms() + 5 * 3_600_000),
                recent_runs: Vec::new(),
                details: vec!["Can do: Read only – makes no changes".into()],
                edit: Some(aidaemon_client_protocol::AutomationEdit {
                    when: "every weekday at 9am".into(),
                    instructions: Some(
                        "Summarize the synthetic inbox and send the highlights".into(),
                    ),
                    notify: Some(aidaemon_client_protocol::AutomationNotify::EveryRun),
                }),
            },
            AutomationSummary {
                id: "mandate-status".into(),
                kind: AutomationKind::Watching,
                name: Some("Status watch".into()),
                what: "Watch the synthetic status page and tell me when it changes".into(),
                how_often: "Every hour".into(),
                status: "paused".into(),
                next_run_ms: None,
                recent_runs: Vec::new(),
                details: Vec::new(),
                edit: None,
            },
            AutomationSummary {
                id: "probe-site".into(),
                kind: AutomationKind::Probe,
                name: Some("Acme site up".into()),
                what: "Check https://acme.example.invalid answers".into(),
                how_often: "Every 5 minutes".into(),
                status: "failed".into(),
                next_run_ms: Some(now_ms() + 300_000),
                recent_runs: Vec::new(),
                details: Vec::new(),
                edit: None,
            },
        ])
    }
    async fn change_automation(
        &self,
        id: &str,
        change: aidaemon_client_protocol::AutomationChange,
    ) -> Result<(bool, String), String> {
        Ok((true, format!("{change:?} {id}: done (demo).")))
    }
    async fn projects(&self) -> anyhow::Result<Vec<ProjectSummary>> {
        Ok(vec![ProjectSummary {
            id: "project-acme".into(),
            name: "Acme site".into(),
            path: Some("/home/synthetic/acme-site".into()),
            channels: vec!["telegram:synthetic_coding_bot".into()],
        }])
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
                attachments: Vec::new(),
                id: format!("{session_id}-m{n}"),
                role: role.to_string(),
                text: text.to_string(),
                // A minute apart, ending an hour ago.
                at_ms: now_ms() - 3_600_000 - (count - n as u64) * 60_000,
                tools: if *role == "assistant" {
                    vec!["read_file".into(), "web_search".into(), "read_file".into()]
                } else {
                    Vec::new()
                },
                calls: if *role == "assistant" {
                    demo_calls()
                } else {
                    Vec::new()
                },
            })
            .collect())
    }
    async fn changes(&self, session_id: &str) -> anyhow::Result<SessionChanges> {
        if session_id != "app:plan" && session_id != "telegram:synthetic-user-1" {
            return Ok(SessionChanges::default());
        }
        let file = |path: &str, status, added, removed, by_session| FileChange {
            path: path.into(),
            status,
            added,
            removed,
            by_session,
        };
        Ok(SessionChanges {
            last_check: Some(aidaemon_client_protocol::CheckRun {
                command: "cargo test -q contact".into(),
                passed: true,
                exit_code: None,
                at_ms: now_ms() - 60_000,
            }),
            repos: vec![RepoChanges {
                root: "/home/synthetic/acme-site".into(),
                branch: Some("feature/contact-form".into()),
                files: vec![
                    file("src/contact.rs", FileStatus::Modified, 4, 1, true),
                    file("templates/contact.html", FileStatus::Untracked, 3, 0, true),
                    file("README.md", FileStatus::Modified, 1, 1, false),
                ],
                patch: "diff --git a/src/contact.rs b/src/contact.rs\n\
                        --- a/src/contact.rs\n+++ b/src/contact.rs\n\
                        @@ -1,3 +1,6 @@\n-fn send() {}\n+fn send(form: &Form) -> Result<()> {\n\
                        +    form.validate()?;\n+    mailer::deliver(form)\n+}\n\
                         \n fn main() {}\n\
                        diff --git a/templates/contact.html b/templates/contact.html\n\
                        new file\n--- /dev/null\n+++ b/templates/contact.html\n\
                        @@ -0,0 +1,3 @@\n+<form method=\"post\">\n+  <input name=\"email\">\n+</form>\n\
                        diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n\
                        @@ -1 +1 @@\n-# Acme site\n+# Acme Corp site\n"
                    .into(),
                truncated: false,
            }],
        })
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
            runs: vec![
                GoalRunInfo {
                    goal_id: "g-digest".into(),
                    description: "Morning digest for Acme Corp".into(),
                    status: "completed".into(),
                    outcome: Some("Sent the digest: 3 new issues, 1 release".into()),
                    started_ms: now - 5 * 3_600_000,
                    finished_ms: Some(now - 5 * 3_600_000 + 90_000),
                    session_id: Some("app:plan".into()),
                },
                GoalRunInfo {
                    goal_id: "g-watch".into(),
                    description: "Watch the synthetic report folder".into(),
                    status: "completed".into(),
                    outcome: None,
                    started_ms: now - 9 * 3_600_000,
                    finished_ms: Some(now - 9 * 3_600_000 + 20_000),
                    session_id: None,
                },
            ],
        })
    }
    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        let words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();
        let mut hits = Vec::new();
        for session in ["app:plan", "telegram:synthetic-user-1"] {
            for message in self.history(session, 50).await? {
                let lower = message.text.to_lowercase();
                if words.iter().all(|w| lower.contains(w.as_str())) {
                    hits.push(SearchHit {
                        session_id: session.into(),
                        message_id: message.id,
                        role: message.role,
                        snippet: message.text,
                        at_ms: message.at_ms,
                    });
                }
            }
        }
        hits.truncate(limit);
        Ok(hits)
    }
    fn commands(&self) -> Vec<SlashCommand> {
        [
            ("tasks", "What is running now"),
            ("cancel", "Stop running work"),
            ("clear", "Start this conversation afresh"),
            ("goals", "Your scheduled goals"),
        ]
        .into_iter()
        .map(|(name, description)| SlashCommand {
            name: name.into(),
            description: description.into(),
        })
        .collect()
    }
    async fn memories(&self) -> anyhow::Result<Vec<MemoryItem>> {
        let forgotten = FORGOTTEN.lock().unwrap().clone();
        Ok([
            (
                "1",
                "preference",
                "coffee",
                "Alice Rivera takes her coffee black",
            ),
            ("2", "work", "employer", "Works at Acme Corp"),
            ("3", "family", "daughters", "Daughters Mia and Zoe"),
        ]
        .into_iter()
        .filter(|(id, ..)| !forgotten.contains(&id.to_string()))
        .map(|(id, category, key, value)| MemoryItem {
            id: id.into(),
            category: category.into(),
            key: key.into(),
            value: value.into(),
            source: "conversation".into(),
            updated_ms: now_ms() - 86_400_000,
        })
        .collect())
    }
    async fn forget(&self, id: &str) -> anyhow::Result<bool> {
        let mut forgotten = FORGOTTEN.lock().unwrap();
        if forgotten.iter().any(|f| f == id) || !["1", "2", "3"].contains(&id) {
            return Ok(false);
        }
        forgotten.push(id.to_string());
        Ok(true)
    }
    async fn settings(&self) -> anyhow::Result<Option<Settings>> {
        Ok(Some(demo_settings().lock().unwrap().clone()))
    }
    async fn settings_gate(
        &self,
        _: &SettingsChange,
    ) -> Result<aidaemon_client_gateway::SettingsGate, String> {
        Ok(aidaemon_client_gateway::SettingsGate::Free)
    }
    async fn change_settings(
        &self,
        change: SettingsChange,
        _: bool,
    ) -> Result<aidaemon_client_gateway::SettingsDone, String> {
        let mut settings = demo_settings().lock().unwrap();
        let entry = |preset: &str, settings: &Settings| {
            settings
                .providers
                .iter()
                .find(|p| p.preset == preset)
                .cloned()
                .ok_or("Unknown provider".to_string())
        };
        let made = |option: &ProviderOption, choice: &ProviderChoice, failover: Option<u32>| {
            ConnectedProvider {
                failover,
                preset: option.preset.clone(),
                name: option.name.clone(),
                base_url: choice.base_url.clone(),
                models: vec![if choice.primary_model.is_empty() {
                    option.model.clone()
                } else {
                    choice.primary_model.clone()
                }],
                hard_model: String::new(),
                quick_model: String::new(),
                offered_models: Vec::new(),
                credential: if option.needs_key {
                    CredentialState::KeyStored
                } else {
                    CredentialState::SignedIn
                },
                key_setting: if !option.needs_key {
                    String::new()
                } else if let Some(i) = failover {
                    format!("provider.fallbacks[{i}].api_key")
                } else {
                    "provider.api_key".into()
                },
            }
        };
        let message = match change {
            SettingsChange::SwitchProvider { provider, .. } => {
                let option = entry(&provider.preset, &settings)?;
                settings.connected[0] = made(&option, &provider, None);
                settings.reload_pending = true;
                format!("Switched the main provider to {}.", option.name)
            }
            SettingsChange::AddFailover { provider, .. } => {
                let option = entry(&provider.preset, &settings)?;
                let index = settings.connected.len() as u32 - 1;
                settings
                    .connected
                    .push(made(&option, &provider, Some(index)));
                settings.reload_pending = true;
                format!("Added {} as failover {}.", option.name, index + 1)
            }
            SettingsChange::RemoveFailover { index, .. } => {
                let removed = settings.connected.remove(index as usize + 1);
                for (at, p) in settings.connected.iter_mut().enumerate().skip(1) {
                    p.failover = Some(at as u32 - 1);
                }
                settings.reload_pending = true;
                format!(
                    "Removed {}; it stops being tried after a reload.",
                    removed.name
                )
            }
            SettingsChange::MoveFailover { index, to, .. } => {
                let count = settings.connected.len() - 1;
                if index as usize >= count || to as usize >= count {
                    return Err("No such failover".into());
                }
                let moved = settings.connected.remove(index as usize + 1);
                let name = moved.name.clone();
                settings.connected.insert(to as usize + 1, moved);
                for (at, p) in settings.connected.iter_mut().enumerate().skip(1) {
                    p.failover = Some(at as u32 - 1);
                    if !p.key_setting.is_empty() {
                        p.key_setting = format!("provider.fallbacks[{}].api_key", at - 1);
                    }
                }
                settings.reload_pending = true;
                format!("{name} is now Failover {} after a reload.", to + 1)
            }
            SettingsChange::CheckKey { provider, key } => {
                if key.0.contains("bad") {
                    return Err("The provider did not accept that key (HTTP 401)".into());
                }
                let option = entry(&provider.preset, &settings)?;
                settings.checked = Some(CheckedKey {
                    preset: option.preset.clone(),
                    models: vec![option.model.clone(), "synthetic-model-2".into()],
                    suggested: Some(option.model.clone()),
                });
                "The key works.".into()
            }
            SettingsChange::SignIn { paste } => {
                settings.sign_in = Some(SignIn {
                    url: "https://auth.example.com/oauth/authorize?client=synthetic".into(),
                    paste,
                });
                "Open the sign-in page.".into()
            }
            SettingsChange::FinishSignIn { .. } => {
                settings.sign_in = None;
                settings.chatgpt_signed_in = true;
                "Signed in.".into()
            }
            SettingsChange::CancelSignIn => {
                settings.sign_in = None;
                "Stopped the sign-in.".into()
            }
            SettingsChange::Secret { key, .. } => {
                for p in settings
                    .connected
                    .iter_mut()
                    .filter(|p| p.key_setting == key)
                {
                    p.credential = CredentialState::KeyStored;
                }
                format!("Stored the secret for `{key}`.")
            }
            SettingsChange::Apply { .. } => {
                settings.reload_pending = false;
                "Reloaded.".into()
            }
            _ => "Done.".into(),
        };
        Ok(aidaemon_client_gateway::SettingsDone {
            message,
            applies: None,
        })
    }
}

/// Synthetic settings: a ChatGPT main provider and two failovers.
fn demo_settings() -> &'static std::sync::Mutex<Settings> {
    static SETTINGS: OnceLock<std::sync::Mutex<Settings>> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let option = |preset: &str, name: &str, billing, needs_key, model: &str| ProviderOption {
            preset: preset.into(),
            name: name.into(),
            billing,
            needs_key,
            key_url: if needs_key {
                "https://example.com/keys".into()
            } else {
                String::new()
            },
            needs_base_url: preset == "custom_openai_compatible",
            model: model.into(),
        };
        let connected =
            |failover, preset: &str, name: &str, models: &[&str], credential, key: &str| {
                ConnectedProvider {
                    failover,
                    preset: preset.into(),
                    name: name.into(),
                    base_url: String::new(),
                    models: models.iter().map(|m| m.to_string()).collect(),
                    hard_model: models.first().map(|m| m.to_string()).unwrap_or_default(),
                    quick_model: models.last().map(|m| m.to_string()).unwrap_or_default(),
                    offered_models: models.iter().map(|m| m.to_string()).collect(),
                    credential,
                    key_setting: key.into(),
                }
            };
        std::sync::Mutex::new(Settings {
            config_path: "/home/synthetic/.aidaemon/config.toml".into(),
            providers: vec![
                option(
                    "openai_chatgpt",
                    "OpenAI (ChatGPT subscription)",
                    Billing::Plan,
                    false,
                    "gpt-synthetic-1",
                ),
                option(
                    "anthropic_native",
                    "Anthropic",
                    Billing::PerToken,
                    true,
                    "claude-synthetic",
                ),
                option(
                    "openai",
                    "OpenAI",
                    Billing::PerToken,
                    true,
                    "gpt-synthetic-2",
                ),
                option(
                    "openrouter",
                    "OpenRouter",
                    Billing::PerToken,
                    true,
                    "synthetic/model-1",
                ),
                option(
                    "custom_openai_compatible",
                    "Custom (OpenAI-compatible)",
                    Billing::PerToken,
                    true,
                    "",
                ),
            ],
            connected: vec![
                connected(
                    None,
                    "openai_chatgpt",
                    "OpenAI (ChatGPT subscription)",
                    &["gpt-synthetic-1", "gpt-synthetic-mini", "gpt-synthetic-0"],
                    CredentialState::SignedIn,
                    "",
                ),
                connected(
                    Some(0),
                    "anthropic_native",
                    "Anthropic",
                    &["claude-synthetic-fast", "claude-synthetic"],
                    CredentialState::KeyStored,
                    "provider.fallbacks[0].api_key",
                ),
                connected(
                    Some(1),
                    "openrouter",
                    "OpenRouter",
                    &["synthetic/model-1"],
                    CredentialState::KeyInFile,
                    "provider.fallbacks[1].api_key",
                ),
                connected(
                    Some(2),
                    "google_ai_studio",
                    "Google Gemini",
                    &["gemini-synthetic"],
                    CredentialState::KeyMissing,
                    "provider.fallbacks[2].api_key",
                ),
            ],
            chatgpt_signed_in: true,
            ..Settings::default()
        })
    })
}

static FORGOTTEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

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
    // Uploads work, so pictures and their viewer can be tried.
    let files = aidaemon_client_gateway::files::FileStore::open(
        dir.path().join("files"),
        dir.path().join("uploads"),
    )?;
    let gateway = ClientGateway::start(GatewayDeps {
        files: Some(Arc::new(files)),
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

/// What a demo reply's tools were given and gave back.
fn demo_calls() -> Vec<aidaemon_client_protocol::ToolCallInfo> {
    let call = |name: &str, input: &str, output: &str| aidaemon_client_protocol::ToolCallInfo {
        name: name.into(),
        input: input.into(),
        output: output.into(),
    };
    vec![
        call(
            "read_file",
            "{\n  \"path\": \"notes/synthetic-week.md\"\n}",
            "# Week\n- Monday: review the synthetic report\n- Tuesday: the demo",
        ),
        call(
            "web_search",
            "{\n  \"query\": \"Acme Corp demo agenda\"\n}",
            "1. Acme Corp — Demo day agenda (example.invalid)",
        ),
        call("read_file", "{\n  \"path\": \"notes/agenda.md\"\n}", ""),
    ]
}
