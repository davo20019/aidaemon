use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::AppConfig;
use crate::events::EventStore;
use crate::health::HealthProbeStore;
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::state::SqliteStateStore;
use crate::tools::terminal::ApprovalRequest;
#[cfg(feature = "browser")]
use crate::tools::BrowserTool;
#[cfg(feature = "slack")]
use crate::tools::ReadChannelHistoryTool;
use crate::tools::{
    CheckEnvironmentTool, CliAgentTool, ConfigManagerTool, DiagnoseTool, EditFileTool,
    GitCommitTool, GitInfoTool, GoalTraceTool, HealthProbeTool, HttpRequestTool,
    ManageCliAgentsTool, ManageMcpTool, ManageMemoriesTool, ManageOAuthTool, ManagePeopleTool,
    PolicyMetricsTool, ProjectInspectTool, ReadFileTool, RememberFactTool, RunCommandTool,
    ScheduledGoalRunsTool, SearchFilesTool, SendFileTool, ServiceStatusTool, ShareMemoryTool,
    SpawnAgentTool, SystemInfoTool, TerminalTool, ToolTraceTool, WebFetchTool, WebSearchTool,
    WriteFileTool,
};
use crate::traits::store_prelude::*;
use crate::traits::Tool;
use crate::types::MediaMessage;

pub struct BaseToolsBundle {
    pub tools: Vec<Arc<dyn Tool>>,
    pub approval_tx: mpsc::Sender<ApprovalRequest>,
    pub approval_rx: mpsc::Receiver<ApprovalRequest>,
    pub media_tx: mpsc::Sender<MediaMessage>,
    pub media_rx: mpsc::Receiver<MediaMessage>,
    pub terminal_tool: Option<Arc<TerminalTool>>,
}

pub struct OptionalToolsOutcome {
    pub has_cli_agents: bool,
    pub inbox_dir: String,
    pub cli_agent_tool: Option<Arc<CliAgentTool>>,
}

pub struct RuntimeToolsOutcome {
    pub spawn_tool: Option<Arc<SpawnAgentTool>>,
    pub oauth_gateway: Option<crate::oauth::OAuthGateway>,
}

#[derive(Debug, Clone, Copy)]
enum BaseToolId {
    SystemInfo,
    Terminal,
    RememberFact,
    ShareMemory,
    ManageMemories,
    ScheduledGoalRuns,
    GoalTrace,
    ToolTrace,
    PolicyMetrics,
    ConfigManager,
    WebFetch,
    WebSearch,
    ReadFile,
    WriteFile,
    EditFile,
    SearchFiles,
    ProjectInspect,
    RunCommand,
    GitInfo,
    GitCommit,
    CheckEnvironment,
    ServiceStatus,
}

impl BaseToolId {
    fn name(self) -> &'static str {
        match self {
            BaseToolId::SystemInfo => "system_info",
            BaseToolId::Terminal => "terminal",
            BaseToolId::RememberFact => "remember_fact",
            BaseToolId::ShareMemory => "share_memory",
            BaseToolId::ManageMemories => "manage_memories",
            BaseToolId::ScheduledGoalRuns => "scheduled_goal_runs",
            BaseToolId::GoalTrace => "goal_trace",
            BaseToolId::ToolTrace => "tool_trace",
            BaseToolId::PolicyMetrics => "policy_metrics",
            BaseToolId::ConfigManager => "manage_config",
            BaseToolId::WebFetch => "web_fetch",
            BaseToolId::WebSearch => "web_search",
            BaseToolId::ReadFile => "read_file",
            BaseToolId::WriteFile => "write_file",
            BaseToolId::EditFile => "edit_file",
            BaseToolId::SearchFiles => "search_files",
            BaseToolId::ProjectInspect => "project_inspect",
            BaseToolId::RunCommand => "run_command",
            BaseToolId::GitInfo => "git_info",
            BaseToolId::GitCommit => "git_commit",
            BaseToolId::CheckEnvironment => "check_environment",
            BaseToolId::ServiceStatus => "service_status",
        }
    }
}

const BASE_TOOL_MANIFEST: &[BaseToolId] = &[
    BaseToolId::SystemInfo,
    BaseToolId::Terminal,
    BaseToolId::RememberFact,
    BaseToolId::ShareMemory,
    BaseToolId::ManageMemories,
    BaseToolId::ScheduledGoalRuns,
    BaseToolId::GoalTrace,
    BaseToolId::ToolTrace,
    BaseToolId::PolicyMetrics,
    BaseToolId::ConfigManager,
    BaseToolId::WebFetch,
    BaseToolId::WebSearch,
    BaseToolId::ReadFile,
    BaseToolId::WriteFile,
    BaseToolId::EditFile,
    BaseToolId::SearchFiles,
    BaseToolId::ProjectInspect,
    BaseToolId::RunCommand,
    BaseToolId::GitInfo,
    BaseToolId::GitCommit,
    BaseToolId::CheckEnvironment,
    BaseToolId::ServiceStatus,
];

#[derive(Debug, Clone, Copy)]
enum OptionalToolId {
    Diagnose,
    #[cfg(feature = "browser")]
    Browser,
    SendFile,
    CliAgents,
    HealthProbe,
    #[cfg(feature = "slack")]
    SlackChannelHistory,
}

impl OptionalToolId {
    fn name(self) -> &'static str {
        match self {
            OptionalToolId::Diagnose => "diagnose",
            #[cfg(feature = "browser")]
            OptionalToolId::Browser => "browser",
            OptionalToolId::SendFile => "send_file",
            OptionalToolId::CliAgents => "cli_agents",
            OptionalToolId::HealthProbe => "health_probe",
            #[cfg(feature = "slack")]
            OptionalToolId::SlackChannelHistory => "read_channel_history",
        }
    }
}

const OPTIONAL_TOOL_MANIFEST: &[OptionalToolId] = &[
    OptionalToolId::Diagnose,
    #[cfg(feature = "browser")]
    OptionalToolId::Browser,
    OptionalToolId::SendFile,
    OptionalToolId::CliAgents,
    OptionalToolId::HealthProbe,
    #[cfg(feature = "slack")]
    OptionalToolId::SlackChannelHistory,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RuntimeToolId {
    ManageMcp,
    ManagePeople,
    HttpRequest,
    ManageOauth,
    SpawnAgent,
}

impl RuntimeToolId {
    fn name(self) -> &'static str {
        match self {
            RuntimeToolId::ManageMcp => "manage_mcp",
            RuntimeToolId::ManagePeople => "manage_people",
            RuntimeToolId::HttpRequest => "http_request",
            RuntimeToolId::ManageOauth => "manage_oauth",
            RuntimeToolId::SpawnAgent => "spawn_agent",
        }
    }
}

struct RuntimeToolSpec {
    id: RuntimeToolId,
    enabled_if: fn(&AppConfig) -> bool,
    depends_on: &'static [RuntimeToolId],
}

impl RuntimeToolSpec {
    fn name(&self) -> &'static str {
        self.id.name()
    }
}

fn runtime_enabled_always(_: &AppConfig) -> bool {
    true
}

fn runtime_enabled_http_request(config: &AppConfig) -> bool {
    !config.http_auth.is_empty() || config.oauth.enabled
}

fn runtime_enabled_oauth(config: &AppConfig) -> bool {
    config.oauth.enabled
}

fn runtime_enabled_spawn(config: &AppConfig) -> bool {
    config.subagents.enabled
}

const RUNTIME_TOOL_MANIFEST: &[RuntimeToolSpec] = &[
    RuntimeToolSpec {
        id: RuntimeToolId::ManageMcp,
        enabled_if: runtime_enabled_always,
        depends_on: &[],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManagePeople,
        enabled_if: runtime_enabled_always,
        depends_on: &[],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::HttpRequest,
        enabled_if: runtime_enabled_http_request,
        depends_on: &[],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManageOauth,
        enabled_if: runtime_enabled_oauth,
        depends_on: &[RuntimeToolId::HttpRequest],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::SpawnAgent,
        enabled_if: runtime_enabled_spawn,
        depends_on: &[],
    },
];

fn validate_runtime_manifest(manifest: &[RuntimeToolSpec]) -> anyhow::Result<()> {
    let mut seen_ids: HashSet<RuntimeToolId> = HashSet::new();
    let mut seen_names: HashSet<&'static str> = HashSet::new();
    for spec in manifest {
        anyhow::ensure!(
            seen_ids.insert(spec.id),
            "runtime manifest is invalid: duplicate tool id {}",
            spec.name()
        );
        anyhow::ensure!(
            seen_names.insert(spec.name()),
            "runtime manifest is invalid: duplicate tool name {}",
            spec.name()
        );
    }

    let defined_ids: HashSet<RuntimeToolId> = manifest.iter().map(|spec| spec.id).collect();
    let mut declared: HashSet<RuntimeToolId> = HashSet::new();
    for spec in manifest {
        for dep in spec.depends_on {
            anyhow::ensure!(
                defined_ids.contains(dep),
                "runtime manifest is invalid: tool {} declares unknown dependency {}",
                spec.name(),
                dep.name()
            );
            anyhow::ensure!(
                declared.contains(dep),
                "runtime manifest is invalid: tool {} depends on {} but dependency appears after it",
                spec.name(),
                dep.name()
            );
        }
        declared.insert(spec.id);
    }

    Ok(())
}

pub async fn build_base_tools(
    config: &AppConfig,
    config_path: PathBuf,
    state: Arc<SqliteStateStore>,
    event_store: Arc<EventStore>,
    approval_queue_capacity: usize,
    media_queue_capacity: usize,
) -> anyhow::Result<BaseToolsBundle> {
    let (approval_tx, approval_rx) = mpsc::channel(approval_queue_capacity);
    let (media_tx, media_rx) = mpsc::channel::<MediaMessage>(media_queue_capacity);

    let mut tools: Vec<Arc<dyn Tool>> = Vec::with_capacity(BASE_TOOL_MANIFEST.len());
    let mut terminal_tool: Option<Arc<TerminalTool>> = None;

    for tool_id in BASE_TOOL_MANIFEST {
        let built = build_base_tool(
            *tool_id,
            config,
            &config_path,
            state.clone(),
            event_store.clone(),
            approval_tx.clone(),
        )
        .await?;
        info!(
            tool = tool_id.name(),
            "Registered base tool from startup manifest"
        );
        if terminal_tool.is_none() {
            terminal_tool = built.terminal_tool.clone();
        }
        tools.push(built.tool);
    }

    Ok(BaseToolsBundle {
        tools,
        approval_tx,
        approval_rx,
        media_tx,
        media_rx,
        terminal_tool,
    })
}

struct BuiltBaseTool {
    tool: Arc<dyn Tool>,
    terminal_tool: Option<Arc<TerminalTool>>,
}

#[allow(clippy::too_many_arguments)]
pub async fn register_optional_tools(
    tools: &mut Vec<Arc<dyn Tool>>,
    config: &AppConfig,
    state: Arc<SqliteStateStore>,
    event_store: Arc<EventStore>,
    llm_runtime: SharedLlmRuntime,
    health_store: Option<Arc<HealthProbeStore>>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
    media_tx: mpsc::Sender<MediaMessage>,
) -> anyhow::Result<OptionalToolsOutcome> {
    let inbox_dir = shellexpand::tilde(&config.files.inbox_dir).to_string();
    let mut has_cli_agents = false;
    let mut cli_agent_tool: Option<Arc<CliAgentTool>> = None;

    for tool_id in OPTIONAL_TOOL_MANIFEST {
        match tool_id {
            OptionalToolId::Diagnose => {
                if !config.diagnostics.enabled {
                    continue;
                }
                tools.push(Arc::new(DiagnoseTool::new(
                    event_store.clone(),
                    state.clone(),
                    llm_runtime.clone(),
                    config.diagnostics.max_events,
                    config.diagnostics.include_raw_tool_args,
                )));
            }
            #[cfg(feature = "browser")]
            OptionalToolId::Browser => {
                if !config.browser.enabled {
                    continue;
                }
                tools.push(Arc::new(BrowserTool::new(
                    config.browser.clone(),
                    media_tx.clone(),
                )));
            }
            OptionalToolId::SendFile => {
                if !config.files.enabled {
                    continue;
                }
                std::fs::create_dir_all(&inbox_dir)?;
                tools.push(Arc::new(SendFileTool::new(
                    media_tx.clone(),
                    &config.files.outbox_dirs,
                    &inbox_dir,
                )));
            }
            OptionalToolId::CliAgents => {
                if !config.cli_agents.enabled {
                    continue;
                }
                let cli_tool = CliAgentTool::discover(
                    config.cli_agents.clone(),
                    state.clone(),
                    llm_runtime.clone(),
                    approval_tx.clone(),
                )
                .await;
                has_cli_agents = cli_tool.has_tools();
                let arc = Arc::new(cli_tool);
                cli_agent_tool = Some(arc.clone());
                if has_cli_agents {
                    tools.push(arc.clone());
                } else {
                    info!("CLI agents enabled but no tools found on system");
                }
                let manage_cli = ManageCliAgentsTool::new(arc, state.clone(), approval_tx.clone());
                tools.push(Arc::new(manage_cli));
            }
            OptionalToolId::HealthProbe => {
                let Some(store) = &health_store else {
                    continue;
                };
                tools.push(Arc::new(HealthProbeTool::new(store.clone())));
            }
            #[cfg(feature = "slack")]
            OptionalToolId::SlackChannelHistory => {
                let mut slack_tokens: Vec<String> = config
                    .all_slack_bots()
                    .iter()
                    .map(|bot| bot.bot_token.clone())
                    .collect();
                if let Ok(dynamic_bots) = state.get_dynamic_bots().await {
                    for bot in dynamic_bots {
                        if bot.channel_type == "slack" && !bot.bot_token.is_empty() {
                            slack_tokens.push(bot.bot_token.clone());
                        }
                    }
                }
                if slack_tokens.is_empty() {
                    continue;
                }
                info!(count = slack_tokens.len(), "Channel history tool enabled");
                tools.push(Arc::new(ReadChannelHistoryTool::new(slack_tokens)));
            }
        }

        info!(
            tool = tool_id.name(),
            "Registered optional tool from startup manifest"
        );
    }

    Ok(OptionalToolsOutcome {
        has_cli_agents,
        inbox_dir,
        cli_agent_tool,
    })
}

pub async fn register_runtime_tools(
    tools: &mut Vec<Arc<dyn Tool>>,
    config: &AppConfig,
    state: Arc<SqliteStateStore>,
    mcp_registry: McpRegistry,
    approval_tx: mpsc::Sender<ApprovalRequest>,
) -> anyhow::Result<RuntimeToolsOutcome> {
    validate_runtime_manifest(RUNTIME_TOOL_MANIFEST)?;

    let http_profiles: crate::oauth::SharedHttpProfiles =
        Arc::new(RwLock::new(config.http_auth.clone()));
    let mut oauth_gateway: Option<crate::oauth::OAuthGateway> = None;
    let mut spawn_tool: Option<Arc<SpawnAgentTool>> = None;
    let mut registered_runtime_tools: std::collections::HashSet<RuntimeToolId> =
        std::collections::HashSet::new();

    for spec in RUNTIME_TOOL_MANIFEST {
        if !(spec.enabled_if)(config) {
            info!(
                tool = spec.name(),
                "Skipped runtime tool from startup manifest (condition disabled)"
            );
            continue;
        }

        let missing_dependencies: Vec<&'static str> = spec
            .depends_on
            .iter()
            .filter(|dep| !registered_runtime_tools.contains(dep))
            .map(|dep| dep.name())
            .collect();
        if !missing_dependencies.is_empty() {
            info!(
                tool = spec.name(),
                missing_dependencies = ?missing_dependencies,
                "Skipped runtime tool from startup manifest (dependencies not satisfied)"
            );
            continue;
        }

        register_runtime_tool_by_id(
            spec.id,
            tools,
            config,
            state.clone(),
            mcp_registry.clone(),
            approval_tx.clone(),
            http_profiles.clone(),
            &mut spawn_tool,
            &mut oauth_gateway,
        )
        .await?;
        registered_runtime_tools.insert(spec.id);
        info!(
            tool = spec.name(),
            dependencies = ?spec.depends_on.iter().map(|dep| dep.name()).collect::<Vec<_>>(),
            "Registered runtime tool from startup manifest"
        );
    }

    Ok(RuntimeToolsOutcome {
        spawn_tool,
        oauth_gateway,
    })
}

#[allow(clippy::too_many_arguments)]
async fn register_runtime_tool_by_id(
    tool_id: RuntimeToolId,
    tools: &mut Vec<Arc<dyn Tool>>,
    config: &AppConfig,
    state: Arc<SqliteStateStore>,
    mcp_registry: McpRegistry,
    approval_tx: mpsc::Sender<ApprovalRequest>,
    http_profiles: crate::oauth::SharedHttpProfiles,
    spawn_tool: &mut Option<Arc<SpawnAgentTool>>,
    oauth_gateway: &mut Option<crate::oauth::OAuthGateway>,
) -> anyhow::Result<()> {
    match tool_id {
        RuntimeToolId::ManageMcp => {
            let manage_mcp = ManageMcpTool::new(mcp_registry, approval_tx);
            tools.push(Arc::new(manage_mcp));
        }
        RuntimeToolId::ManagePeople => {
            tools.push(Arc::new(ManagePeopleTool::new(state.clone())));
            // Seed the DB setting from config if not already set.
            if state.get_setting("people_enabled").await?.is_none() {
                state
                    .set_setting(
                        "people_enabled",
                        if config.people.enabled {
                            "true"
                        } else {
                            "false"
                        },
                    )
                    .await?;
            }
        }
        RuntimeToolId::HttpRequest => {
            tools.push(Arc::new(HttpRequestTool::new(http_profiles, approval_tx)));
        }
        RuntimeToolId::ManageOauth => {
            let callback_url = config
                .oauth
                .callback_url
                .clone()
                .unwrap_or_else(|| format!("http://localhost:{}", config.daemon.health_port));
            let gateway =
                crate::oauth::OAuthGateway::new(state.clone(), http_profiles, callback_url);

            // Register built-in providers.
            for name in crate::oauth::providers::builtin_provider_names() {
                if let Some(provider) = crate::oauth::providers::get_builtin_provider(name) {
                    gateway.register_provider(provider).await;
                }
            }

            // Register custom providers from config.
            for (name, provider_config) in &config.oauth.providers {
                gateway
                    .register_config_provider(name, provider_config)
                    .await;
            }

            // Restore existing connections from DB + keychain.
            gateway.restore_connections().await;

            tools.push(Arc::new(ManageOAuthTool::new(gateway.clone(), state)));
            *oauth_gateway = Some(gateway);
        }
        RuntimeToolId::SpawnAgent => {
            let st = Arc::new(
                SpawnAgentTool::new_deferred(
                    config.subagents.max_response_chars,
                    config.subagents.timeout_secs,
                )
                .with_state(state.clone() as Arc<dyn crate::traits::StateStore>),
            );
            tools.push(st.clone());
            *spawn_tool = Some(st);
        }
    }

    Ok(())
}

async fn build_base_tool(
    tool_id: BaseToolId,
    config: &AppConfig,
    config_path: &Path,
    state: Arc<SqliteStateStore>,
    event_store: Arc<EventStore>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
) -> anyhow::Result<BuiltBaseTool> {
    let built = match tool_id {
        BaseToolId::SystemInfo => BuiltBaseTool {
            tool: Arc::new(SystemInfoTool),
            terminal_tool: None,
        },
        BaseToolId::Terminal => {
            let terminal = Arc::new(
                TerminalTool::new(
                    config.terminal.allowed_prefixes.clone(),
                    approval_tx,
                    config.terminal.initial_timeout_secs,
                    config.terminal.max_output_chars,
                    config.terminal.permission_mode,
                    state.pool(),
                )
                .await
                .with_event_store(event_store)
                .with_state(state as Arc<dyn crate::traits::StateStore>),
            );
            BuiltBaseTool {
                tool: terminal.clone(),
                terminal_tool: Some(terminal),
            }
        }
        BaseToolId::RememberFact => BuiltBaseTool {
            tool: Arc::new(RememberFactTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::ShareMemory => BuiltBaseTool {
            tool: Arc::new(ShareMemoryTool::new(state, approval_tx)),
            terminal_tool: None,
        },
        BaseToolId::ManageMemories => BuiltBaseTool {
            tool: Arc::new(ManageMemoriesTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::ScheduledGoalRuns => BuiltBaseTool {
            tool: Arc::new(ScheduledGoalRunsTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::GoalTrace => BuiltBaseTool {
            tool: Arc::new(GoalTraceTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::ToolTrace => BuiltBaseTool {
            tool: Arc::new(ToolTraceTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::PolicyMetrics => BuiltBaseTool {
            tool: Arc::new(PolicyMetricsTool),
            terminal_tool: None,
        },
        BaseToolId::ConfigManager => BuiltBaseTool {
            tool: Arc::new(ConfigManagerTool::new(
                config_path.to_path_buf(),
                approval_tx,
            )),
            terminal_tool: None,
        },
        BaseToolId::WebFetch => BuiltBaseTool {
            tool: Arc::new(WebFetchTool::new()),
            terminal_tool: None,
        },
        BaseToolId::WebSearch => BuiltBaseTool {
            tool: Arc::new(WebSearchTool::new(&config.search)),
            terminal_tool: None,
        },
        BaseToolId::ReadFile => BuiltBaseTool {
            tool: Arc::new(ReadFileTool),
            terminal_tool: None,
        },
        BaseToolId::WriteFile => BuiltBaseTool {
            tool: Arc::new(WriteFileTool),
            terminal_tool: None,
        },
        BaseToolId::EditFile => BuiltBaseTool {
            tool: Arc::new(EditFileTool),
            terminal_tool: None,
        },
        BaseToolId::SearchFiles => BuiltBaseTool {
            tool: Arc::new(SearchFilesTool),
            terminal_tool: None,
        },
        BaseToolId::ProjectInspect => BuiltBaseTool {
            tool: Arc::new(ProjectInspectTool),
            terminal_tool: None,
        },
        BaseToolId::RunCommand => BuiltBaseTool {
            tool: Arc::new(RunCommandTool),
            terminal_tool: None,
        },
        BaseToolId::GitInfo => BuiltBaseTool {
            tool: Arc::new(GitInfoTool),
            terminal_tool: None,
        },
        BaseToolId::GitCommit => BuiltBaseTool {
            tool: Arc::new(GitCommitTool),
            terminal_tool: None,
        },
        BaseToolId::CheckEnvironment => BuiltBaseTool {
            tool: Arc::new(CheckEnvironmentTool),
            terminal_tool: None,
        },
        BaseToolId::ServiceStatus => BuiltBaseTool {
            tool: Arc::new(ServiceStatusTool),
            terminal_tool: None,
        },
    };

    Ok(built)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::llm_runtime::{router_from_models, SharedLlmRuntime};
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::MockProvider;
    use crate::traits::{ModelProvider, StateStore};
    use proptest::prelude::*;
    use serde_json::json;
    use std::collections::HashSet;
    use tempfile::{NamedTempFile, TempDir};

    async fn build_tool_schemas_for_contract_validation(
        diagnostics_enabled: bool,
        files_enabled: bool,
        subagents_enabled: bool,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test-key"
            "#,
        )?;
        config.provider.models.apply_defaults(&config.provider.kind);
        config.diagnostics.enabled = diagnostics_enabled;
        config.files.enabled = files_enabled;
        config.subagents.enabled = subagents_enabled;
        config.cli_agents.enabled = false;
        config.oauth.enabled = false;

        let io_temp = TempDir::new()?;
        config.files.inbox_dir = io_temp.path().join("inbox").display().to_string();
        config.files.outbox_dirs = vec![io_temp.path().join("outbox").display().to_string()];

        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new()?);
        let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);
        let event_store = Arc::new(EventStore::new(state.pool()).await?);
        let config_file = NamedTempFile::new()?;

        let mut bundle = build_base_tools(
            &config,
            config_file.path().to_path_buf(),
            state.clone(),
            event_store.clone(),
            32,
            8,
        )
        .await?;
        let model_provider = Arc::new(MockProvider::new()) as Arc<dyn ModelProvider>;
        let llm_runtime = SharedLlmRuntime::new(
            model_provider,
            router_from_models(config.provider.models.clone()),
            config.provider.kind.clone(),
            config.provider.models.primary.clone(),
        );
        let _optional = register_optional_tools(
            &mut bundle.tools,
            &config,
            state.clone(),
            event_store.clone(),
            llm_runtime,
            None,
            bundle.approval_tx.clone(),
            bundle.media_tx.clone(),
        )
        .await?;

        let mcp_registry = McpRegistry::new(state.clone() as Arc<dyn StateStore>);
        let _runtime = register_runtime_tools(
            &mut bundle.tools,
            &config,
            state.clone(),
            mcp_registry,
            bundle.approval_tx.clone(),
        )
        .await?;

        Ok(bundle
            .tools
            .iter()
            .map(|tool| json!({"type":"function","function": tool.schema()}))
            .collect())
    }

    #[test]
    fn manifest_has_unique_tool_names() {
        let mut seen = HashSet::new();
        for tool in BASE_TOOL_MANIFEST {
            assert!(
                seen.insert(tool.name()),
                "duplicate tool name in manifest: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn optional_manifest_has_unique_tool_names() {
        let mut seen = HashSet::new();
        for tool in OPTIONAL_TOOL_MANIFEST {
            assert!(
                seen.insert(tool.name()),
                "duplicate optional tool name in manifest: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn runtime_manifest_has_unique_tool_names() {
        let mut seen = HashSet::new();
        for tool in RUNTIME_TOOL_MANIFEST {
            assert!(
                seen.insert(tool.name()),
                "duplicate runtime tool name in manifest: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn runtime_manifest_matches_expected_pipeline_order() {
        let names: Vec<&'static str> = RUNTIME_TOOL_MANIFEST
            .iter()
            .map(|tool| tool.name())
            .collect();
        assert_eq!(
            names,
            vec![
                "manage_mcp",
                "manage_people",
                "http_request",
                "manage_oauth",
                "spawn_agent",
            ]
        );
    }

    #[test]
    fn runtime_manifest_dependencies_are_declared_before_dependents() {
        let mut seen = HashSet::new();
        for spec in RUNTIME_TOOL_MANIFEST {
            for dep in spec.depends_on {
                assert!(
                    seen.contains(dep),
                    "runtime tool {} depends on {} but dependency appears after it",
                    spec.name(),
                    dep.name()
                );
            }
            seen.insert(spec.id);
        }
    }

    #[test]
    fn runtime_manifest_validator_accepts_current_manifest() {
        assert!(validate_runtime_manifest(RUNTIME_TOOL_MANIFEST).is_ok());
    }

    #[test]
    fn runtime_manifest_validator_rejects_duplicate_ids() {
        let manifest = [
            RuntimeToolSpec {
                id: RuntimeToolId::ManageMcp,
                enabled_if: runtime_enabled_always,
                depends_on: &[],
            },
            RuntimeToolSpec {
                id: RuntimeToolId::ManageMcp,
                enabled_if: runtime_enabled_always,
                depends_on: &[],
            },
        ];
        let err = validate_runtime_manifest(&manifest)
            .unwrap_err()
            .to_string();
        assert!(err.contains("duplicate tool id"));
    }

    #[test]
    fn runtime_manifest_validator_rejects_unknown_dependencies() {
        let manifest = [RuntimeToolSpec {
            id: RuntimeToolId::ManageOauth,
            enabled_if: runtime_enabled_always,
            depends_on: &[RuntimeToolId::HttpRequest],
        }];
        let err = validate_runtime_manifest(&manifest)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown dependency"));
    }

    #[test]
    fn runtime_manifest_validator_rejects_dependency_order() {
        let manifest = [
            RuntimeToolSpec {
                id: RuntimeToolId::ManageOauth,
                enabled_if: runtime_enabled_always,
                depends_on: &[RuntimeToolId::HttpRequest],
            },
            RuntimeToolSpec {
                id: RuntimeToolId::HttpRequest,
                enabled_if: runtime_enabled_always,
                depends_on: &[],
            },
        ];
        let err = validate_runtime_manifest(&manifest)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dependency appears after it"));
    }

    #[tokio::test]
    async fn built_tools_have_schemas_that_match_agent_contract() {
        let schemas = build_tool_schemas_for_contract_validation(true, true, true)
            .await
            .unwrap();
        assert!(!schemas.is_empty(), "expected startup to register tools");
        for schema in schemas {
            let tool_name = schema["function"]["name"]
                .as_str()
                .unwrap_or("<unknown>")
                .to_string();
            let result = Agent::validate_tool_definition_contract(&schema);
            assert!(
                result.is_ok(),
                "tool schema contract failed for {}: {:?}",
                tool_name,
                result.err()
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]
        #[test]
        fn built_tool_schema_contract_holds_under_random_feature_flags(
            diagnostics_enabled in any::<bool>(),
            files_enabled in any::<bool>(),
            subagents_enabled in any::<bool>(),
        ) {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async move {
                let schemas = build_tool_schemas_for_contract_validation(
                    diagnostics_enabled,
                    files_enabled,
                    subagents_enabled,
                )
                .await
                .unwrap();
                assert!(!schemas.is_empty());
                for schema in schemas {
                    let tool_name = schema["function"]["name"]
                        .as_str()
                        .unwrap_or("<unknown>")
                        .to_string();
                    let result = Agent::validate_tool_definition_contract(&schema);
                    assert!(
                        result.is_ok(),
                        "tool schema contract failed for {}: {:?}",
                        tool_name,
                        result.err()
                    );
                }
            });
        }
    }
}
