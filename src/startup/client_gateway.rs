//! The Client Gateway (`/client/v1`) on the daemon: its host (sessions,
//! transcripts and slash commands from the daemon's stores) and its start.
//! Off unless `[client_gateway] enabled = true`.

use std::sync::Arc;

use aidaemon_client_gateway::{ClientGateway, ClientGatewayConfig, ClientHost, GatewayDeps};
use aidaemon_client_protocol::{
    HistoryMessage, MemoryItem, SearchHit, SessionSummary, SlashCommand,
};
use async_trait::async_trait;

use crate::channels::{ChannelHub, ChannelRuntimeDeps};
use crate::config::AppConfig;
use crate::traits::StateStore;
use crate::types::UserRole;

/// The `[client_gateway]` table of the config file, read without resolving
/// the rest (which would touch the keychain). Missing or invalid: defaults.
pub(crate) fn settings_from_file(config_path: &std::path::Path) -> ClientGatewayConfig {
    with_installed_app(
        std::fs::read_to_string(config_path)
            .ok()
            .and_then(|text| text.parse::<toml::Table>().ok())
            .and_then(|table| table.get("client_gateway").cloned())
            .and_then(|value| value.try_into().ok())
            .unwrap_or_default(),
    )
}

/// Where `install.sh` puts the app's web build.
pub(crate) fn installed_app_dir() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|home| home.join(".aidaemon").join("app-web"))
}

/// `config` serving the installed web build when it names none and one is
/// installed: an install needs no `app_dir` line.
pub(crate) fn with_installed_app(mut config: ClientGatewayConfig) -> ClientGatewayConfig {
    let unset = config
        .app_dir
        .as_deref()
        .is_none_or(|d| d.trim().is_empty());
    if unset
        && let Some(dir) = installed_app_dir()
        && dir.join("index.html").is_file()
    {
        config.app_dir = Some(dir.to_string_lossy().into_owned());
    }
    config
}

/// The local token, minting one when there is none that still works.
pub(crate) fn local_token_or_mint(config: &ClientGatewayConfig) -> anyhow::Result<String> {
    if let Some(token) = aidaemon_client_gateway::token::local_token(config) {
        return Ok(token);
    }
    let path = aidaemon_client_gateway::token::mint_local_token(config)?;
    eprintln!(
        "Minted a local Client Gateway token (saved to {}).",
        path.display()
    );
    aidaemon_client_gateway::token::local_token(config)
        .ok_or_else(|| anyhow::anyhow!("the token just minted does not verify"))
}

/// The configured `public_url`, as the origins served while no tunnel
/// publishes a live one.
pub(crate) fn static_origins(config: &ClientGatewayConfig) -> Vec<String> {
    config
        .public_url
        .iter()
        .map(|url| url.trim().trim_end_matches('/').to_string())
        .filter(|url| !url.is_empty())
        .collect()
}

/// The daemon as the gateway's host.
pub(crate) struct DaemonClientHost {
    pub state: Arc<dyn StateStore>,
    pub commands: crate::channels::commands::CommandContext,
    pub system: SystemSources,
    pub settings: crate::startup::client_settings::SettingsHost,
    /// First-run setup's chat step and greeting.
    pub setup: crate::setup::channel::ChannelSetup,
    /// Where the files of messages are shown from.
    pub files: Option<Arc<aidaemon_client_gateway::files::FileStore>>,
    /// The live providers and models, for a conversation's model picker.
    pub llm: aidaemon_runtime::llm::SharedLlmRuntime,
    /// The agent, for what kind of work each stored tool call was.
    pub agent: Arc<dyn crate::runtime_ports::DaemonAgentRuntime>,
}

impl DaemonClientHost {
    /// config.toml as it is now, for names the owner sees (as Settings shows them).
    fn config_table(&self) -> Option<toml::Table> {
        std::fs::read_to_string(&self.commands.config_path)
            .ok()?
            .parse()
            .ok()
    }

    /// The owner's automations as the agent's own tool sees and changes
    /// them, so the app and a chat apply the same rules.
    fn automation_tool(&self) -> crate::tools::ManageAutomationsTool {
        crate::tools::ManageAutomationsTool::new(self.state.clone())
            .with_health_probes(self.system.health.clone())
    }

    /// The channels of every session, for the chat step's "connected".
    async fn session_channels(&self) -> Vec<String> {
        self.sessions()
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.channel)
            .collect()
    }
}

/// What the app's System area reads: the daemon's models, token use, model
/// call timings, schedules, background jobs, queues and health checks.
pub(crate) struct SystemSources {
    /// The live providers and models.
    pub llm: aidaemon_runtime::llm::SharedLlmRuntime,
    pub started_at: std::time::Instant,
    pub models: aidaemon_providers::routing::ModelsConfig,
    pub pool: sqlx::SqlitePool,
    pub events: Arc<crate::events::EventStore>,
    pub health: Option<Arc<crate::health::HealthProbeStore>>,
    pub jobs: Option<Arc<crate::heartbeat::HeartbeatTelemetry>>,
    pub queues: Arc<crate::queue_telemetry::QueueTelemetry>,
    /// What the daemon remembers, for the app's Memory page.
    pub memory: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    /// The MCP servers, for the app's Settings page (taken by it).
    pub mcp: Option<crate::mcp::McpRegistry>,
}

/// A stored time (RFC 3339 or SQLite's `YYYY-MM-DD HH:MM:SS`, UTC) in ms.
fn stored_ms(text: &str) -> Option<u64> {
    let at = chrono::DateTime::parse_from_rfc3339(text)
        .map(|t| t.with_timezone(&chrono::Utc))
        .or_else(|_| {
            chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").map(|t| t.and_utc())
        })
        .ok()?;
    u64::try_from(at.timestamp_millis()).ok()
}

impl SystemSources {
    async fn status(&self) -> aidaemon_client_protocol::SystemStatus {
        use aidaemon_client_protocol::*;
        let usage = sqlx::query_as::<_, (Option<String>, String, i64, i64, i64)>(
            "SELECT date(created_at) AS day, model, SUM(input_tokens), SUM(output_tokens), \
             COUNT(*) FROM token_usage WHERE created_at >= datetime('now', '-7 days') \
             GROUP BY day, model ORDER BY day DESC, model",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|(day, model, input, output, requests)| UsageDay {
            day: day.unwrap_or_default(),
            model,
            input_tokens: input.max(0) as u64,
            output_tokens: output.max(0) as u64,
            requests: requests.max(0) as u64,
        })
        .collect();
        let since = chrono::Utc::now() - chrono::Duration::hours(24);
        // The timings come from the latest 2000 calls; the count is all of them.
        let calls_in_window: Option<i64> = sqlx::query_scalar(
            "SELECT COUNT(*) FROM events WHERE event_type = 'llm_call' AND created_at >= ?",
        )
        .bind(since.to_rfc3339())
        .fetch_one(&self.pool)
        .await
        .ok();
        let model_calls = self
            .events
            .get_llm_stats(since)
            .await
            .ok()
            .filter(|stats| stats.total_calls > 0)
            .map(|stats| ModelCalls {
                total: calls_in_window
                    .and_then(|n| u64::try_from(n).ok())
                    .unwrap_or(stats.total_calls)
                    .max(stats.total_calls),
                failed: stats.failed_calls,
                fell_back: stats.fell_back_count,
                p50_ms: stats.p50_latency_ms,
                p95_ms: stats.p95_latency_ms,
                max_ms: stats.max_latency_ms,
            });
        type ScheduleRow = (
            String,
            String,
            String,
            String,
            i64,
            i64,
            Option<String>,
            String,
        );
        let schedules = sqlx::query_as::<_, ScheduleRow>(
            "SELECT s.goal_id, g.description, g.status, s.original_schedule, s.is_paused, \
             s.is_one_shot, s.last_run_at, s.next_run_at FROM goal_schedules s \
             JOIN goals g ON g.id = s.goal_id ORDER BY s.next_run_at ASC LIMIT 100",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(
            |(goal_id, description, status, schedule, paused, one_shot, last, next)| ScheduleInfo {
                goal_id,
                description,
                status,
                schedule,
                paused: paused != 0,
                one_shot: one_shot != 0,
                next_run_ms: stored_ms(&next),
                last_run_ms: last.as_deref().and_then(stored_ms),
            },
        )
        .collect();
        let jobs = self
            .jobs
            .as_ref()
            .map(|jobs| jobs.snapshots())
            .unwrap_or_default()
            .into_iter()
            .map(|job| JobInfo {
                name: job.name,
                every_secs: job.interval_secs,
                running: job.is_running,
                consecutive_failures: job.consecutive_failures,
                last_success_ms: job.last_success_at.as_deref().and_then(stored_ms),
                last_error: job.last_error.filter(|_| job.consecutive_failures > 0),
            })
            .collect();
        let queues = self
            .queues
            .snapshots()
            .into_iter()
            .map(|queue| QueueInfo {
                name: queue.name,
                depth: queue.current_depth as u64,
                capacity: queue.capacity as u64,
                dropped: queue.dropped_total,
                failed: queue.failed_total,
                overloaded: queue.overload_active,
            })
            .collect();
        let mut checks = Vec::new();
        if let Some(store) = &self.health {
            for probe in store.list_probes().await.unwrap_or_default() {
                let latest = store.get_latest_result(&probe.id).await.ok().flatten();
                let failures = store
                    .count_consecutive_failures(&probe.id)
                    .await
                    .unwrap_or(0);
                let status = match latest.as_ref().map(|r| &r.status) {
                    None => CheckStatus::Unknown,
                    Some(crate::health::probes::ProbeStatus::Healthy) => {
                        let slow = probe
                            .latency_threshold_ms
                            .zip(latest.as_ref().and_then(|r| r.latency_ms))
                            .is_some_and(|(limit, took)| took > limit);
                        if slow {
                            CheckStatus::Degraded
                        } else {
                            CheckStatus::Healthy
                        }
                    }
                    Some(_) => CheckStatus::Unhealthy,
                };
                checks.push(HealthCheckInfo {
                    name: probe.name,
                    target: probe.target,
                    status,
                    paused: probe.is_paused,
                    consecutive_failures: failures,
                    latency_ms: latest.as_ref().and_then(|r| r.latency_ms),
                    checked_ms: latest
                        .as_ref()
                        .and_then(|r| u64::try_from(r.checked_at.timestamp_millis()).ok()),
                    error: latest.and_then(|r| r.error_message),
                });
            }
        }
        type RunRow = (
            String,
            String,
            String,
            Option<String>,
            String,
            Option<String>,
            String,
        );
        let runs = sqlx::query_as::<_, RunRow>(
            "SELECT r.goal_id, g.description, r.status, r.outcome_summary, r.started_at, \
             r.completed_at, g.session_id FROM goal_runs r JOIN goals g ON g.id = r.goal_id \
             WHERE r.started_at >= datetime('now', '-1 day') \
             ORDER BY r.started_at DESC LIMIT 50",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(
            |(goal_id, description, status, outcome, started, finished, session_id)| GoalRunInfo {
                goal_id,
                description,
                status,
                outcome: outcome.as_deref().and_then(outcome_words),
                started_ms: stored_ms(&started).unwrap_or(0),
                finished_ms: finished.as_deref().and_then(stored_ms),
                session_id: Some(session_id).filter(|s| !s.is_empty()),
            },
        )
        .collect();
        let model = Some(self.models.default_model.clone()).filter(|m| !m.is_empty());
        SystemStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.started_at.elapsed().as_secs(),
            model,
            fallback_models: self.models.fallback_models.clone(),
            usage,
            model_calls,
            schedules,
            jobs,
            queues,
            checks,
            runs,
            nodes: crate::startup::nodes::infos(&self.pool).await,
        }
    }
}

/// How a stored run ended, as a client reads it. A missed run (the daemon
/// was off when it was due) is its own outcome, never `Failed`.
fn run_outcome(status: &str) -> aidaemon_client_protocol::RunOutcome {
    use aidaemon_client_protocol::RunOutcome;
    match status {
        "pending" | "running" => RunOutcome::Running,
        "completed" => RunOutcome::Done,
        "failed" => RunOutcome::Failed,
        "blocked" => RunOutcome::Blocked,
        "cancelled" => RunOutcome::Cancelled,
        aidaemon_work::goals::GOAL_RUN_MISSED => RunOutcome::Missed,
        _ => RunOutcome::Unknown,
    }
}

/// A run's report as the owner reads it: a bare status code
/// (`mandate_act_satisfied`) becomes words ("Mandate act satisfied").
fn outcome_words(outcome: &str) -> Option<String> {
    let outcome = outcome.trim();
    if outcome.is_empty() {
        return None;
    }
    if outcome.contains(char::is_whitespace) || !outcome.contains('_') {
        return Some(outcome.to_owned());
    }
    let words = outcome.replace('_', " ");
    let mut chars = words.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
}

/// A call of a reply's turn, still waiting for its result.
struct PendingCall {
    id: String,
    info: aidaemon_client_protocol::ToolCallInfo,
}

/// The calls a stored assistant message made, in order, each with its
/// arguments as the owner sees them.
fn tool_calls(
    json: &str,
    kind_of: impl Fn(&str, &str) -> Option<aidaemon_client_protocol::ToolKind>,
) -> Vec<PendingCall> {
    let Ok(serde_json::Value::Array(calls)) = serde_json::from_str::<serde_json::Value>(json)
    else {
        return Vec::new();
    };
    calls
        .iter()
        .filter_map(|call| {
            let function = call.get("function");
            let name = call
                .get("name")
                .or_else(|| function.and_then(|f| f.get("name")))
                .and_then(|n| n.as_str())?;
            let arguments = call
                .get("arguments")
                .or_else(|| function.and_then(|f| f.get("arguments")));
            Some(PendingCall {
                id: call
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned(),
                info: aidaemon_client_protocol::ToolCallInfo {
                    name: name.to_owned(),
                    input: arguments.map(shown_input).unwrap_or_default(),
                    output: String::new(),
                    kind: kind_of(name, &arguments.map(raw_arguments).unwrap_or_default()),
                },
            })
        })
        .collect()
}

/// A call's arguments as the tool itself reads them: a JSON string as is,
/// any other value as its JSON.
fn raw_arguments(arguments: &serde_json::Value) -> String {
    match arguments {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// A call's arguments as the owner's app shows them: one JSON object,
/// secrets redacted, trimmed. A string holding JSON (the nested shape)
/// reads as that JSON; `{}` reads as nothing.
fn shown_input(arguments: &serde_json::Value) -> String {
    aidaemon_tools::shown::call_input(arguments)
}

/// A call's result as the owner's app shows it (`aidaemon_tools::shown`).
fn shown_output(content: &str) -> String {
    aidaemon_tools::shown::call_output(content)
}

/// Markdown as the words a reader sees: links keep their text, emphasis
/// and code marks go.
fn plain(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' | '`' => {}
            // `[text](url)`: keep the text, drop the url.
            ']' if chars.peek() == Some(&'(') => {
                for c in chars.by_ref() {
                    if c == ')' {
                        break;
                    }
                }
            }
            '[' => {}
            c => out.push(c),
        }
    }
    out
}

/// The text around the first of `words` in `text`, on one line.
fn snippet(text: &str, words: &[String]) -> String {
    const SPAN: usize = 140;
    let flat: String = plain(text).split_whitespace().collect::<Vec<_>>().join(" ");
    let lower = flat.to_lowercase();
    let at = words
        .iter()
        .filter_map(|w| lower.find(w.as_str()))
        .min()
        .unwrap_or(0);
    // Back up to a char boundary about a third of the span before the match.
    let mut start = at.saturating_sub(SPAN / 3);
    while start > 0 && !flat.is_char_boundary(start) {
        start -= 1;
    }
    let body: String = flat[start..].chars().take(SPAN).collect();
    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(&body);
    if flat[start..].chars().count() > SPAN {
        out.push('…');
    }
    out
}

/// Per conversation: when it last had a message (`last`), and what the
/// owner said first and latest, the conversation list's names.
struct SessionEnds {
    last: std::collections::HashMap<String, String>,
    first: std::collections::HashMap<String, String>,
    latest: std::collections::HashMap<String, String>,
}

/// The model each of `ids` (a session's replies) was recorded under, from
/// its `assistant_response` event; a reply with none is left out.
async fn reply_models(
    pool: &sqlx::SqlitePool,
    session_id: &str,
    ids: &[&str],
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for chunk in ids.chunks(200) {
        let marks = vec!["?"; chunk.len()].join(",");
        // Only fixed text and `?` placeholders are joined; every id is bound.
        let sql = format!(
            "SELECT json_extract(data, '$.message_id'), json_extract(data, '$.model') \
             FROM events WHERE session_id = ? AND event_type = 'assistant_response' \
             AND json_extract(data, '$.message_id') IN ({marks})"
        );
        let mut query =
            sqlx::query_as::<_, (Option<String>, Option<String>)>(sqlx::AssertSqlSafe(sql))
                .bind(session_id);
        for id in chunk {
            query = query.bind(*id);
        }
        match query.fetch_all(pool).await {
            Ok(rows) => out.extend(
                rows.into_iter()
                    .filter_map(|(id, model)| Some((id?, model?))),
            ),
            Err(error) => tracing::warn!(%error, "Could not read the models of replies"),
        }
    }
    out
}

async fn session_ends(pool: &sqlx::SqlitePool) -> SessionEnds {
    // Each reads one row per session: the per-session ends come from
    // `idx_events_type_session`, then only those rows are opened.
    let last: std::collections::HashMap<String, String> = sqlx::query_as::<_, (String, String)>(
        "SELECT e.session_id, e.created_at FROM events e JOIN \
             (SELECT MAX(id) AS id FROM events \
              WHERE event_type IN ('user_message', 'assistant_response') \
              GROUP BY session_id) m ON e.id = m.id",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default()
    .into_iter()
    .collect();
    // A name from what the owner said: an app conversation is about its
    // first message; a chat (Telegram, Slack...) runs for months, so it
    // goes by its latest one.
    let said = |pick: &'static str| {
        let pool = pool.clone();
        async move {
            sqlx::query_as::<_, (String, Option<String>)>(sqlx::AssertSqlSafe(format!(
                "SELECT e.session_id, json_extract(e.data, '$.content') FROM events e JOIN \
                 (SELECT {pick}(id) AS id FROM events WHERE event_type = 'user_message' \
                  GROUP BY session_id) m ON e.id = m.id"
            )))
            .fetch_all(&pool)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(id, text)| Some((id, text?)))
            .collect::<std::collections::HashMap<String, String>>()
        }
    };
    let (first, latest) = (said("MIN").await, said("MAX").await);
    SessionEnds {
        last,
        first,
        latest,
    }
}

#[async_trait]
impl ClientHost for DaemonClientHost {
    async fn sessions(&self) -> anyhow::Result<Vec<SessionSummary>> {
        let SessionEnds {
            last,
            first,
            latest,
        } = session_ends(&self.system.pool).await;
        // A chat is direct only when its messages said so; one not heard
        // from since this was recorded counts as shared.
        let shared: std::collections::HashSet<String> = sqlx::query_scalar::<_, String>(
            "SELECT session_id FROM session_channels \
             WHERE shared IS NOT 0 AND channel_name != 'app'",
        )
        .fetch_all(&self.system.pool)
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();
        let marks = super::client_sessions::marks(&self.system.pool).await;
        let projects: std::collections::HashMap<String, String> = self
            .state
            .session_work_projects()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        Ok(self
            .state
            .load_session_channels()
            .await?
            .into_iter()
            .map(|(id, channel)| {
                let words = if id.starts_with(aidaemon_client_gateway::APP_SESSION_PREFIX) {
                    first.get(&id)
                } else {
                    latest.get(&id)
                };
                let (named, archived) = marks.get(&id).cloned().unwrap_or_default();
                // The owner's own name wins over one taken from what was said.
                let title = named.or_else(|| {
                    words
                        // The words typed, not the note about a file sent
                        // with them.
                        .map(|text| aidaemon_channels::attachments::title_words(text))
                        .map(|text| aidaemon_channels::presentation::word_preview(text.trim(), 60))
                        .filter(|t| !t.is_empty())
                });
                SessionSummary {
                    // The app replies into chat conversations too (routed to
                    // their chat by the gateway).
                    writable: true,
                    last_activity_ms: last.get(&id).and_then(|at| stored_ms(at)),
                    project_id: projects.get(&id).cloned(),
                    group: shared.contains(&id),
                    archived,
                    id,
                    channel,
                    title,
                }
            })
            .collect())
    }

    async fn projects(&self) -> anyhow::Result<Vec<aidaemon_client_protocol::ProjectSummary>> {
        super::client_projects::projects(self.state.as_ref()).await
    }

    async fn create_project(
        &self,
        name: &str,
        path: Option<&str>,
        create_folder: bool,
    ) -> Result<aidaemon_client_protocol::ProjectSummary, aidaemon_client_protocol::ProtocolError>
    {
        super::client_projects::create(self.state.as_ref(), name, path, create_folder).await
    }

    async fn update_project(
        &self,
        id: &str,
        name: Option<&str>,
        path: Option<&str>,
        create_folder: bool,
    ) -> Result<aidaemon_client_protocol::ProjectSummary, aidaemon_client_protocol::ProtocolError>
    {
        super::client_projects::update(self.state.as_ref(), id, name, path, create_folder).await
    }

    async fn delete_project(
        &self,
        id: &str,
    ) -> Result<(), aidaemon_client_protocol::ProtocolError> {
        super::client_projects::delete(self.state.as_ref(), id).await
    }

    async fn model_options(&self) -> anyhow::Result<aidaemon_client_protocol::ModelOptions> {
        Ok(super::client_models::named(
            super::client_models::options(&self.llm),
            self.config_table().as_ref(),
        ))
    }

    async fn session_model(
        &self,
        session_id: &str,
    ) -> anyhow::Result<aidaemon_client_protocol::SessionModel> {
        Ok(super::client_models::current(self.state.as_ref(), &self.llm, session_id).await)
    }

    async fn set_session_model(
        &self,
        session_id: &str,
        model: Option<aidaemon_client_protocol::ModelChoice>,
        effort: Option<aidaemon_client_protocol::Effort>,
    ) -> Result<aidaemon_client_protocol::SessionModel, String> {
        super::client_models::set(self.state.as_ref(), &self.llm, session_id, model, effort).await
    }

    async fn set_session_project(
        &self,
        session_id: &str,
        project_id: Option<&str>,
    ) -> Result<(), String> {
        super::client_projects::place(self.state.as_ref(), session_id, project_id).await
    }

    async fn set_channel_project(
        &self,
        channel: &str,
        project_id: Option<&str>,
    ) -> Result<(), String> {
        super::client_projects::send_bot(self.state.as_ref(), channel, project_id).await
    }

    async fn automations(
        &self,
    ) -> anyhow::Result<Vec<aidaemon_client_protocol::AutomationSummary>> {
        use aidaemon_client_protocol::{AutomationKind as Kind, AutomationRun, AutomationSummary};
        use aidaemon_work::automations::AutomationKind;
        let ms = |at: chrono::DateTime<chrono::Utc>| u64::try_from(at.timestamp_millis()).ok();
        Ok(self
            .automation_tool()
            .entries()
            .await?
            .into_iter()
            .map(|entry| AutomationSummary {
                kind: match entry.kind {
                    AutomationKind::Scheduled => Kind::Scheduled,
                    AutomationKind::Watching => Kind::Watching,
                    AutomationKind::Probe => Kind::Probe,
                },
                id: entry.id,
                name: entry.name,
                what: entry.what,
                how_often: entry.how_often,
                status: entry.status,
                next_run_ms: entry.next.and_then(ms),
                recent_runs: entry
                    .runs
                    .into_iter()
                    .filter_map(|run| {
                        Some(AutomationRun {
                            started_ms: ms(run.started)?,
                            finished_ms: run.finished.and_then(ms),
                            outcome: run_outcome(&run.status),
                            summary: run.summary,
                        })
                    })
                    .collect(),
                details: entry.details,
                session_id: entry.session_id,
                edit: entry
                    .edit
                    .map(|edit| aidaemon_client_protocol::AutomationEdit {
                        when: edit.when,
                        instructions: edit.instructions,
                        notify: edit.notify.map(|mode| match mode {
                            aidaemon_work::automations::NotifyMode::EveryRun => {
                                aidaemon_client_protocol::AutomationNotify::EveryRun
                            }
                            aidaemon_work::automations::NotifyMode::Changes => {
                                aidaemon_client_protocol::AutomationNotify::Changes
                            }
                            aidaemon_work::automations::NotifyMode::Problems => {
                                aidaemon_client_protocol::AutomationNotify::Problems
                            }
                        }),
                    }),
            })
            .collect())
    }

    async fn change_automation(
        &self,
        id: &str,
        change: aidaemon_client_protocol::AutomationChange,
    ) -> Result<(bool, String), String> {
        use aidaemon_client_protocol::AutomationChange;
        let action = match change {
            AutomationChange::Pause => "pause",
            AutomationChange::Resume => "resume",
            AutomationChange::Cancel => "cancel",
            AutomationChange::RunNow => "run_now",
            AutomationChange::Unknown => return Err("This app asked for an unknown change".into()),
        };
        self.automation_tool()
            .owner_change(id, action)
            .await
            .map_err(|e| e.to_string())
    }

    async fn edit_automation(
        &self,
        id: &str,
        name: Option<&str>,
        when: Option<&str>,
        instructions: Option<&str>,
        notify: Option<aidaemon_client_protocol::AutomationNotify>,
    ) -> Result<(bool, String), String> {
        use aidaemon_client_protocol::AutomationNotify as Notify;
        use aidaemon_work::automations::NotifyMode;
        let notify = match notify {
            None => None,
            Some(Notify::EveryRun) => Some(NotifyMode::EveryRun),
            Some(Notify::Changes) => Some(NotifyMode::Changes),
            Some(Notify::Problems) => Some(NotifyMode::Problems),
            Some(Notify::Unknown) => return Err("This app asked for an unknown setting".into()),
        };
        self.automation_tool()
            .owner_edit(id, name, when, instructions, notify)
            .await
            .map_err(|e| e.to_string())
    }

    async fn rename_session(&self, session_id: &str, title: Option<&str>) -> Result<(), String> {
        super::client_sessions::rename(&self.system.pool, session_id, title).await
    }

    async fn archive_session(&self, session_id: &str, archived: bool) -> Result<(), String> {
        super::client_sessions::archive(&self.system.pool, session_id, archived).await
    }

    async fn revert_file(
        &self,
        session_id: &str,
        root: &str,
        path: &str,
    ) -> Result<aidaemon_client_protocol::SessionChanges, String> {
        super::client_changes::revert_file(&self.system.pool, session_id, root, path).await
    }

    fn is_secret_path(&self, path: &std::path::Path) -> bool {
        // The daemon's own config and env files too, as the file tools
        // judge them.
        let dir = crate::runtime_working_dir();
        let env_file = crate::resolve_runtime_env_file_path(&dir);
        crate::tools::DaemonPaths::new(vec![dir.join("config.toml"), env_file, dir.join(".env")])
            .is_secret(path)
    }

    async fn system(&self) -> Option<aidaemon_client_protocol::SystemStatus> {
        Some(self.system.status().await)
    }

    async fn changes(
        &self,
        session_id: &str,
    ) -> anyhow::Result<aidaemon_client_protocol::SessionChanges> {
        super::client_changes::session_changes(&self.system.pool, session_id).await
    }

    async fn shell_folder(&self, session_id: &str) -> anyhow::Result<Option<std::path::PathBuf>> {
        super::shells::session_folder(self.state.as_ref(), session_id).await
    }

    async fn settings(&self) -> anyhow::Result<Option<aidaemon_client_protocol::Settings>> {
        self.settings.settings().await.map(Some)
    }

    async fn settings_gate(
        &self,
        change: &aidaemon_client_protocol::SettingsChange,
    ) -> Result<aidaemon_client_gateway::SettingsGate, String> {
        self.settings.gate(change).await
    }

    async fn change_settings(
        &self,
        change: aidaemon_client_protocol::SettingsChange,
        approved: bool,
    ) -> Result<aidaemon_client_gateway::SettingsDone, String> {
        self.settings.change(change, approved).await
    }

    fn client_seen(&self) {
        self.settings.client_seen();
    }

    async fn setup(&self) -> Option<aidaemon_client_protocol::SetupState> {
        let channels = self.session_channels().await;
        self.setup.state(&channels).await
    }

    async fn setup_step(
        &self,
        action: aidaemon_client_protocol::SetupAction,
    ) -> Result<aidaemon_client_protocol::SetupState, String> {
        let channels = self.session_channels().await;
        self.setup.step(action, &channels).await
    }

    fn served_by(&self, model: &str) -> Option<aidaemon_client_protocol::ServedBy> {
        super::client_models::served_by(&self.llm, self.config_table().as_ref(), model)
    }

    async fn history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<HistoryMessage>> {
        let messages = self.state.get_history(session_id, limit).await?;
        let reply_ids: Vec<&str> = messages
            .iter()
            .filter(|m| m.role == "assistant")
            .map(|m| m.id.as_str())
            .collect();
        let models = reply_models(&self.system.pool, session_id, &reply_ids).await;
        let config = self.config_table();
        // Runtime continuations and system notices ride as user-role turns
        // for the model; they are not what the owner said. A reply carries
        // the tools its turn used before it.
        let messages = messages.into_iter().filter(|m| {
            matches!(m.role.as_str(), "user" | "assistant" | "tool")
                && !m.annotations.iter().any(|a| {
                    matches!(
                        a,
                        crate::traits::MessageAnnotation::InternalContinuation
                            | crate::traits::MessageAnnotation::EntireSystemNotice
                    )
                })
        });
        let mut pending: Vec<PendingCall> = Vec::new();
        let mut out = Vec::new();
        for m in messages {
            if m.role == "user" {
                pending.clear();
            }
            if let Some(json) = &m.tool_calls_json {
                pending.extend(tool_calls(json, |name, arguments| {
                    self.agent
                        .tool_activity(name, arguments)
                        .map(aidaemon_client_gateway::tool_kind)
                }));
                continue;
            }
            // A tool's result goes to the call it answers.
            if m.role == "tool" {
                if let Some(call) = m
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| pending.iter_mut().rev().find(|c| c.id == id))
                {
                    call.info.output = shown_output(m.content.as_deref().unwrap_or_default());
                }
                continue;
            }
            // A message's files come as themselves, not as the notes the
            // agent read about them.
            let attachments = match &self.files {
                Some(files) if !m.attachments.is_empty() => files.show(&m.attachments),
                _ => Vec::new(),
            };
            // Files the agent sent: shown with the caption the owner saw;
            // the note's text is for the model.
            let delivered = m.annotations.iter().find_map(|a| match a {
                crate::traits::MessageAnnotation::DeliveredFiles { caption } => {
                    Some(caption.clone())
                }
                _ => None,
            });
            let text = match &delivered {
                Some(caption) => caption.clone(),
                None => m.content.unwrap_or_default(),
            };
            // A helper's result as the owner was shown it, without the
            // label that tells the model where it came from.
            let text = match aidaemon_runtime::turn::notes::parent_note_shown_text(&text) {
                Some(shown) if m.role == "assistant" => shown.to_string(),
                _ => text,
            };
            // A reply reads as it was shown when it arrived.
            let text = if delivered.is_some() {
                text
            } else if m.role == "assistant" {
                aidaemon_channels::presentation::prepare_chat_message(&text)
            } else {
                text
            };
            let text = if m.attachments.is_empty() || delivered.is_some() {
                text
            } else {
                aidaemon_channels::attachments::user_authored_text(&text)
            };
            if text.trim().is_empty() && attachments.is_empty() {
                continue;
            }
            // A delivery lands mid-turn: the turn's tools stay for its reply.
            let calls: Vec<aidaemon_client_protocol::ToolCallInfo> =
                if m.role == "assistant" && delivered.is_none() {
                    std::mem::take(&mut pending)
                        .into_iter()
                        .map(|c| c.info)
                        .collect()
                } else {
                    Vec::new()
                };
            let tools = calls.iter().map(|c| c.name.clone()).collect();
            let handed_off = m.role == "assistant"
                && delivered.is_none()
                && aidaemon_runtime::background_run::is_launch_line(&text);
            let served_by = (m.role == "assistant")
                .then(|| models.get(&m.id))
                .flatten()
                .and_then(|model| {
                    super::client_models::served_by(&self.llm, config.as_ref(), model)
                });
            out.push(HistoryMessage {
                served_by,
                handed_off,
                attachments,
                id: m.id,
                role: m.role,
                text,
                at_ms: u64::try_from(m.created_at.timestamp_millis()).unwrap_or(0),
                tools,
                calls,
            });
        }
        Ok(out)
    }

    async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        let words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .take(8)
            .map(str::to_owned)
            .collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let mut sql = String::from(
            "SELECT id, session_id, event_type, json_extract(data, '$.content'), created_at \
             FROM events WHERE event_type IN ('user_message', 'assistant_response') \
             AND json_extract(data, '$.content') IS NOT NULL",
        );
        for _ in &words {
            sql.push_str(" AND lower(json_extract(data, '$.content')) LIKE ? ESCAPE '\\'");
        }
        sql.push_str(" ORDER BY id DESC LIMIT ?");
        // Only fixed text and `?` placeholders are joined; every word is bound.
        let mut q =
            sqlx::query_as::<_, (i64, String, String, String, String)>(sqlx::AssertSqlSafe(sql));
        for word in &words {
            let escaped = word
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_");
            q = q.bind(format!("%{escaped}%"));
        }
        q = q.bind(i64::try_from(limit).unwrap_or(50));
        Ok(q.fetch_all(&self.system.pool)
            .await?
            .into_iter()
            .map(|(id, session_id, kind, content, at)| SearchHit {
                session_id,
                message_id: id.to_string(),
                role: if kind == "user_message" {
                    "user"
                } else {
                    "assistant"
                }
                .into(),
                snippet: snippet(&content, &words),
                at_ms: stored_ms(&at).unwrap_or(0),
            })
            .collect())
    }

    fn commands(&self) -> Vec<SlashCommand> {
        let mut defs = aidaemon_channels::slash::shared_commands();
        defs.extend(self.commands.extension_commands());
        let mut seen = std::collections::HashSet::new();
        defs.into_iter()
            .filter(|d| seen.insert(d.name))
            .map(|d| SlashCommand {
                name: d.name.to_string(),
                description: d.description.to_string(),
            })
            .collect()
    }

    async fn memories(&self) -> anyhow::Result<Vec<MemoryItem>> {
        let mut facts = self.system.memory.get_facts(None).await?;
        facts.retain(|f| f.superseded_at.is_none());
        facts.sort_by_key(|f| std::cmp::Reverse(f.updated_at));
        facts.truncate(500);
        Ok(facts
            .into_iter()
            .map(|f| MemoryItem {
                id: f.id.to_string(),
                category: f.category,
                key: f.key,
                value: f.value,
                source: f.source,
                updated_ms: u64::try_from(f.updated_at.timestamp_millis()).unwrap_or(0),
            })
            .collect())
    }

    async fn forget(&self, id: &str) -> anyhow::Result<bool> {
        let Ok(id) = id.parse::<i64>() else {
            return Ok(false);
        };
        let memory = &self.system.memory;
        if !memory.get_facts(None).await?.iter().any(|f| f.id == id) {
            return Ok(false);
        }
        memory.delete_fact(id).await?;
        Ok(true)
    }

    async fn slash(&self, session_id: &str, command: &str, args: &str) -> Option<String> {
        if let Some(reply) = super::client_models::slash(
            self.state.as_ref(),
            &self.llm,
            self.config_table().as_ref(),
            session_id,
            command,
            args,
        )
        .await
        {
            return Some(reply);
        }
        self.commands
            .dispatch(
                &format!("/{command}"),
                args,
                session_id,
                None,
                UserRole::Owner,
            )
            .await
    }
}

/// The agent-driven view on the running gateway: `client_view` moves the
/// app connection that sent the turn's message, and only that one.
pub(crate) struct GatewayViews;

impl aidaemon_client_gateway::client_view::ClientViews for GatewayViews {
    fn can_show(&self, session_id: &str) -> bool {
        running().is_some_and(|gateway| gateway.views().can_show(session_id))
    }

    fn show(
        &self,
        session_id: &str,
        action: aidaemon_client_protocol::ViewAction,
        reason: &str,
    ) -> Result<(), String> {
        let gateway = running().ok_or_else(|| "the Client Gateway is off".to_string())?;
        gateway
            .views()
            .show(session_id, action, reason)
            .map_err(|error| error.to_string())
    }
}

/// Features the agent may ask the owner to turn on: those off in the
/// config the daemon started with (each takes effect at a restart), shown
/// on the running gateway's apps.
pub(crate) struct GatewayFeatures {
    off: Vec<aidaemon_client_protocol::Feature>,
}

impl GatewayFeatures {
    pub(crate) fn of(config: &AppConfig) -> Self {
        use aidaemon_client_protocol::Feature;
        let mut off = Vec::new();
        if cfg!(feature = "browser") && !config.browser.enabled {
            off.push(Feature::Browser);
        }
        if !config.files.stt.enabled {
            off.push(Feature::Voice);
        }
        if cfg!(feature = "computer_use") && !config.computer_use.enabled {
            off.push(Feature::ComputerUse);
        }
        Self { off }
    }
}

impl aidaemon_client_gateway::client_feature::FeatureRequests for GatewayFeatures {
    fn off(&self) -> Vec<aidaemon_client_protocol::Feature> {
        if running().is_none() {
            return Vec::new();
        }
        self.off.clone()
    }

    fn request(
        &self,
        session_id: &str,
        feature: aidaemon_client_protocol::Feature,
        reason: &str,
    ) -> Result<(), String> {
        let gateway = running().ok_or_else(|| "the Client Gateway is off".to_string())?;
        gateway.feature_requested(session_id, feature, reason);
        Ok(())
    }
}

/// The running gateway, for the chat's device commands; unset while it is
/// off.
static GATEWAY: std::sync::OnceLock<Arc<ClientGateway>> = std::sync::OnceLock::new();

pub(crate) fn running() -> Option<Arc<ClientGateway>> {
    GATEWAY.get().cloned()
}

/// Start the gateway when it is enabled: it sees every chat's turns and
/// approval cards from now on, and serves on its loopback port. Pairing
/// confirmations go out on `approvals`, as the agent's cards do.
pub async fn start(
    config: &AppConfig,
    pool: sqlx::SqlitePool,
    deps: &ChannelRuntimeDeps,
    hub: Arc<ChannelHub>,
    approvals: crate::tools::ApprovalBroker,
    endpoints: aidaemon_core::EndpointSource,
    mut system: SystemSources,
) -> anyhow::Result<()> {
    let commands = deps.command_context();
    if !config.client_gateway.enabled {
        // An address change that turned the gateway off still rolls back
        // when no app can reach it.
        let _ = crate::startup::settings_upkeep::arm_trial(&commands.config_path);
        return Ok(());
    }
    let (confirm_tx, mut confirm_rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        while let Some(request) = confirm_rx.recv().await {
            let _ = approvals.send(request).await;
        }
    });
    let log = aidaemon_store_sqlite::SqliteClientEventLog::open(pool)
        .await
        .map_err(|error| anyhow::anyhow!("client event log: {error}"))?;
    // Files of messages: kept beside the gateway's token; uploads go to the
    // inbox, where the agent reads any file a chat sent.
    let files = aidaemon_client_gateway::token::token_file(&config.client_gateway)
        .parent()
        .map(|dir| dir.join("client-files"))
        .and_then(|dir| {
            aidaemon_client_gateway::files::FileStore::open(
                dir,
                shellexpand::tilde(&config.files.inbox_dir).to_string(),
            )
            .map_err(|e| tracing::warn!(error = %e, "Client files are off: their folder failed"))
            .ok()
        })
        .map(Arc::new);
    let gateway = ClientGateway::start(GatewayDeps {
        files: files.clone(),
        turns: aidaemon_channels::turns::TurnDeps {
            agent: deps.agent.clone(),
            session_map: deps.session_map.clone(),
            sessions: deps.state.clone(),
            task_registry: deps.task_registry.clone(),
            watchdog_stale_threshold_secs: deps.watchdog_stale_threshold_secs,
            watch: deps.watch.clone(),
        },
        hub,
        host: Arc::new(DaemonClientHost {
            agent: deps.agent.clone(),
            llm: system.llm.clone(),
            state: deps.state.clone(),
            settings: crate::startup::client_settings::SettingsHost::new(
                commands.config_path.clone(),
                deps.state.clone(),
                system.mcp.take(),
                Some(deps.command_context()),
            ),
            files,
            setup: crate::setup::channel::ChannelSetup {
                config_path: commands.config_path.clone(),
                state: deps.state.clone(),
                events: system.events.clone(),
            },
            commands,
            system,
        }),
        log,
        config: with_installed_app(config.client_gateway.clone()),
        endpoints,
        token_file: aidaemon_client_gateway::token::token_file(&config.client_gateway),
        approvals: Some(confirm_tx),
    })
    .await?;
    if config.surfaces.enabled {
        gateway.attach_surfaces(crate::startup::surfaces::registry(config));
    }
    if config.shells.enabled {
        gateway.attach_shells(crate::startup::shells::registry(config));
    }
    let _ = GATEWAY.set(gateway.clone());
    if config.client_gateway.remote.enabled {
    if config.exposure.client_gateway.is_some() {
        start_exposed(gateway.clone()).await;
    }
        start_remote(&config.client_gateway, gateway.clone()).await;
    }
    tokio::spawn(async move {
        if let Err(error) = aidaemon_client_gateway::serve(gateway).await {
            tracing::error!(%error, "Client Gateway stopped");
        }
    });
    Ok(())
}

/// The daemon's iroh endpoint, while remote access is on.
/// The loopback listener tunnels publish: the gateway as it is served off
/// this computer ([`aidaemon_client_gateway::exposed_router`]).
static EXPOSED: std::sync::OnceLock<std::net::SocketAddr> = std::sync::OnceLock::new();

/// Where a tunnel adapter should forward the Client Gateway to, once the
/// gateway is up. Never the loopback listener, where the local token works.
pub(crate) fn exposed_addr() -> Option<std::net::SocketAddr> {
    EXPOSED.get().copied()
}

/// Serve the gateway for tunnels on a loopback port of its own. A failure
/// leaves nothing to expose: tunnels stay down rather than publish the
/// listener the local token works on.
async fn start_exposed(gateway: Arc<ClientGateway>) {
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", 0)).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::warn!(%error, "Client Gateway not exposable: no tunnel listener");
            return;
        }
    };
    let Ok(addr) = listener.local_addr() else {
        return;
    };
    let _ = EXPOSED.set(addr);
    tracing::info!(%addr, "Client Gateway tunnel listener");
    tokio::spawn(async move {
        let app = aidaemon_client_gateway::exposed_router(gateway);
        if let Err(error) = axum::serve(listener, app).await {
            tracing::error!(%error, "Client Gateway tunnel listener stopped");
        }
    });
}

static REMOTE: std::sync::OnceLock<Arc<aidaemon_remote::RemoteListener>> =
    std::sync::OnceLock::new();

/// The relays `[client_gateway.remote] relays` names; none: n0's public
/// relays.
fn remote_relays(config: &ClientGatewayConfig) -> anyhow::Result<aidaemon_remote::Relays> {
    if config.remote.relays.is_empty() {
        return Ok(aidaemon_remote::Relays::N0);
    }
    let urls = config
        .remote
        .relays
        .iter()
        .map(|url| {
            url.trim()
                .parse::<aidaemon_remote::RelayUrl>()
                .map_err(|e| anyhow::anyhow!("client_gateway.remote.relays: {url}: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(aidaemon_remote::Relays::Custom(urls))
}

/// `[client_gateway.remote]`: the gateway over iroh, for the owner's apps
/// anywhere. The handshake admits only the endpoint keys of paired devices
/// (any key while a pairing is open: [`aidaemon_client_gateway::Pairing::admits`]);
/// behind it the peer guard opens pairing and sign-in to an admitted key
/// and the rest only with a token issued to that key. Pairing links carry
/// its address once it is online. A failure leaves the loopback gateway
/// serving.
async fn start_remote(config: &ClientGatewayConfig, gateway: Arc<ClientGateway>) {
    let key_path = aidaemon_client_gateway::token::token_file(config).with_file_name("remote-key");
    let router =
        aidaemon_client_gateway::remote_router::<aidaemon_remote::RemotePeer>(gateway.clone());
    let started = async {
        let key = aidaemon_remote::load_or_create_key(&key_path)?;
        // Unread, nobody but an open pairing would be admitted: stay off
        // rather than admit every key.
        gateway
            .pairing()
            .load_admission()
            .await
            .map_err(|error| anyhow::anyhow!("remote admission: {}", error.message))?;
        let relays = remote_relays(config)?;
        let pairing = gateway.pairing().clone();
        let access = aidaemon_remote::Access::Only(Arc::new(move |peer| {
            pairing.admits(&aidaemon_remote::endpoint_hex(peer))
        }));
        aidaemon_remote::RemoteListener::bind(key, &relays, access, router).await
    };
    let listener = match started.await {
        Ok(listener) => Arc::new(listener),
        Err(error) => {
            tracing::warn!(error = %format!("{error:#}"), "Remote access is off");
            return;
        }
    };
    tracing::info!(endpoint = %listener.id(), "Remote access listening (iroh)");
    let _ = REMOTE.set(listener.clone());
    // The address names the home relay, known once connected; follow it
    // when the endpoint moves to another.
    tokio::spawn(async move {
        listener.online().await;
        let mut shown: Option<String> = None;
        loop {
            let url = listener.url().to_string();
            if shown.as_deref() != Some(url.as_str()) {
                tracing::info!(%url, "Remote access address");
                gateway.pairing().set_remote_url(Some(url.clone()));
                shown = Some(url);
            }
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        }
    });
}

/// `/pair_device`, `/devices` and `/revoke_device <id>`: the owner's devices
/// for the Client Gateway, from any chat.
pub(crate) struct DeviceCommands;

const GATEWAY_OFF: &str = "The Client Gateway is off. Set `[client_gateway] enabled = true` in \
config.toml and restart to pair apps.";

#[async_trait]
impl aidaemon_channels::slash::SlashCommand for DeviceCommands {
    fn handles(&self, cmd: &str) -> bool {
        matches!(cmd, "/pair_device" | "/devices" | "/revoke_device")
    }

    fn commands(&self) -> Vec<aidaemon_channels::slash::CommandDef> {
        use aidaemon_channels::slash::{CommandCategory, CommandDef};
        vec![
            CommandDef {
                name: "pair_device",
                description: "Pair a phone or browser with the app",
                usage: None,
                category: CommandCategory::Core,
            },
            CommandDef {
                name: "devices",
                description: "List paired devices",
                usage: None,
                category: CommandCategory::Core,
            },
            CommandDef {
                name: "revoke_device",
                description: "Unpair a device",
                usage: Some("/revoke_device <id>"),
                category: CommandCategory::Core,
            },
        ]
    }

    async fn run(
        &self,
        cmd: &str,
        args: &str,
        call: &aidaemon_channels::slash::SlashCall<'_>,
    ) -> String {
        if call.user_role != UserRole::Owner {
            return "Only the owner can manage paired devices.".to_string();
        }
        let Some(gateway) = running() else {
            return GATEWAY_OFF.to_string();
        };
        let pairing = gateway.pairing();
        match cmd {
            "/pair_device" => match pairing.offer(Some(call.session_id.to_string())).await {
                Ok(offer) => format!(
                    "To pair your phone, open this link on it (it opens the AIDaemon \
                     app): {}\n\n\
                     To pair a browser instead, open {} in it.\n\n\
                     The code is **{}**, valid 10 minutes. I'll ask you here before the \
                     device can do anything.\n\n\
                     Daemon fingerprint: `{}`",
                    offer.page_link(),
                    web_link(&gateway, &offer),
                    offer.code,
                    offer.fingerprint
                ),
                Err(error) => format!("Could not start pairing: {}", error.message),
            },
            "/devices" => match pairing.devices().await {
                Ok(devices) if devices.iter().all(|d| d.revoked) => {
                    "No paired devices. Send /pair_device to pair one.".to_string()
                }
                Ok(devices) => devices_text(&devices),
                Err(error) => format!("Could not list devices: {}", error.message),
            },
            _ => {
                let id = args.trim();
                if id.is_empty() {
                    return "Usage: /revoke_device <device id> (see /devices)".to_string();
                }
                match pairing.revoke(id).await {
                    Ok(true) => format!("Revoked `{id}`: its sessions are closed."),
                    Ok(false) => format!("No active device `{id}`."),
                    Err(error) => format!("Could not revoke: {}", error.message),
                }
            }
        }
    }
}

/// The web client's address with the offer filled in.
pub(crate) fn web_link(
    gateway: &ClientGateway,
    offer: &aidaemon_client_protocol::PairingOffer,
) -> String {
    offer.web_link(&gateway.public_url(), gateway.web_app_path())
}

fn devices_text(devices: &[aidaemon_client_protocol::DeviceInfo]) -> String {
    let mut text = "Paired devices:".to_string();
    for device in devices.iter().filter(|d| !d.revoked) {
        let scopes: Vec<String> = device
            .scopes
            .iter()
            .map(|s| format!("{s:?}").to_lowercase())
            .collect();
        text.push_str(&format!(
            "\n• {} ({:?}) `{}`: {}",
            device.name,
            device.kind,
            device.device_id,
            scopes.join(", ")
        ));
        if let Some(ceiling) = device.risk_ceiling {
            text.push_str(&format!(", approves up to {ceiling:?} risk"));
        }
    }
    text.push_str("\n\nRevoke one with /revoke_device <id>.");
    text
}

#[cfg(test)]
mod tests {
    use super::{outcome_words, run_outcome, session_ends, shown_output, snippet, tool_calls};
    use aidaemon_tools::shown::CALL_OUTPUT_CHARS;

    /// A run the scheduler skipped reaches the app as `missed`, never as
    /// `failed`; a status this build does not know stays `unknown`.
    #[test]
    fn a_missed_run_reaches_the_app_as_missed_not_failed() {
        use aidaemon_client_protocol::RunOutcome;
        assert_eq!(
            run_outcome(aidaemon_work::goals::GOAL_RUN_MISSED),
            RunOutcome::Missed
        );
        assert_eq!(run_outcome("failed"), RunOutcome::Failed);
        assert_eq!(run_outcome("completed"), RunOutcome::Done);
        assert_eq!(run_outcome("pending"), RunOutcome::Running);
        assert_eq!(run_outcome("something_new"), RunOutcome::Unknown);
    }

    #[tokio::test]
    async fn the_conversation_list_reads_each_session_s_first_and_latest_message() {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        aidaemon_events::schema::migrate_events(&pool)
            .await
            .unwrap();
        for (session, kind, text, at) in [
            (
                "app:1",
                "user_message",
                "Plan the trip",
                "2026-01-01 10:00:00",
            ),
            (
                "app:1",
                "assistant_response",
                "Here is a plan",
                "2026-01-01 10:01:00",
            ),
            ("app:1", "user_message", "Add a day", "2026-01-01 10:02:00"),
            (
                "telegram:2",
                "user_message",
                "Fix the footer",
                "2026-01-02 09:00:00",
            ),
        ] {
            sqlx::query(
                "INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, ?, ?)",
            )
            .bind(session)
            .bind(kind)
            .bind(serde_json::json!({ "content": text }).to_string())
            .bind(at)
            .execute(&pool)
            .await
            .unwrap();
        }
        let ends = session_ends(&pool).await;
        assert_eq!(ends.first["app:1"], "Plan the trip");
        assert_eq!(ends.latest["app:1"], "Add a day");
        assert_eq!(ends.last["app:1"], "2026-01-01 10:02:00");
        assert_eq!(ends.first["telegram:2"], "Fix the footer");
    }

    #[test]
    fn a_status_code_outcome_reads_as_words() {
        assert_eq!(
            outcome_words("mandate_act_satisfied").as_deref(),
            Some("Mandate act satisfied")
        );
        assert_eq!(
            outcome_words("Sent the digest.").as_deref(),
            Some("Sent the digest.")
        );
        assert_eq!(outcome_words("  "), None);
    }

    #[test]
    fn tool_calls_read_both_stored_shapes() {
        let names = |json: &str| -> Vec<String> {
            tool_calls(json, |_, _| None)
                .into_iter()
                .map(|c| c.info.name)
                .collect()
        };
        let flat =
            r#"[{"id":"1","name":"read_file","arguments":{}},{"id":"2","name":"web_search"}]"#;
        assert_eq!(names(flat), ["read_file", "web_search"]);
        let nested =
            r#"[{"id":"1","function":{"name":"terminal","arguments":"{\"command\":\"ls\"}"}}]"#;
        let calls = tool_calls(nested, |_, _| None);
        assert_eq!(calls[0].info.name, "terminal");
        assert_eq!(calls[0].id, "1");
        assert!(
            calls[0].info.input.contains("\"command\": \"ls\""),
            "{}",
            calls[0].info.input
        );
        assert!(tool_calls(flat, |_, _| None)[0].info.input.is_empty());
        assert!(tool_calls("not json", |_, _| None).is_empty());
        // The kind comes from the tool, given the arguments it reads.
        let kinds = tool_calls(nested, |name, arguments| {
            (name == "terminal" && arguments == r#"{"command":"ls"}"#)
                .then_some(aidaemon_client_protocol::ToolKind::Run)
        });
        assert_eq!(
            kinds[0].info.kind,
            Some(aidaemon_client_protocol::ToolKind::Run)
        );
    }

    #[test]
    fn a_call_shows_its_data_without_the_frame_and_trimmed() {
        let framed = aidaemon_policies::wrap_untrusted("web_fetch", "Synthetic page body");
        assert_eq!(shown_output(&framed), "Synthetic page body");
        let long = "x".repeat(CALL_OUTPUT_CHARS * 2);
        let shown = shown_output(&long);
        assert!(
            shown.chars().count() < CALL_OUTPUT_CHARS + 40,
            "{}",
            shown.len()
        );
        assert!(shown.ends_with("(truncated)"));
    }

    #[test]
    fn snippet_centres_on_the_first_match_and_marks_cuts() {
        let text = format!(
            "{} the synthetic weekly summary {}",
            "lead ".repeat(40),
            "tail ".repeat(40)
        );
        let words = vec!["weekly".to_string()];
        let s = snippet(&text, &words);
        assert!(s.starts_with('…') && s.ends_with('…'), "{s}");
        assert!(s.contains("synthetic weekly summary"), "{s}");
        assert_eq!(snippet("Short ünïcode note", &words), "Short ünïcode note");
        assert_eq!(
            snippet(
                "Quito is **17.8°C** now. [Source: Open-Meteo](https://h.example.invalid/x?a=1)",
                &["quito".into()]
            ),
            "Quito is 17.8°C now. Source: Open-Meteo"
        );
    }
}
