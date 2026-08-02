use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::info;

use crate::config::AppConfig;
use crate::events::EventStore;
use crate::health::HealthProbeStore;
use crate::llm_runtime::SharedLlmRuntime;
use crate::mcp::McpRegistry;
use crate::state::SqliteStateStore;
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
#[cfg(feature = "browser")]
use crate::tools::BrowserTool;
#[cfg(feature = "computer_use")]
use crate::tools::ComputerUseTool;
#[cfg(feature = "slack")]
use crate::tools::ReadChannelHistoryTool;
use crate::tools::{
    CheckEnvironmentTool, CliAgentTool, ConfigManagerTool, DiagnoseTool, EditFileTool,
    GenerateImageTool, GitCommitTool, GitInfoTool, GoalTraceTool, HealthProbeTool, HttpRequestTool,
    ListCheckpointsTool, ManageApiTool, ManageCliAgentsTool, ManageHttpAuthTool,
    ManageMandatesTool, ManageMcpTool, ManageMemoriesTool, ManageOAuthTool, ManagePeopleTool,
    PolicyMetricsTool, ProjectInspectTool, ReadFileTool, RememberFactTool, RunCommandTool,
    ScheduledGoalRunsTool, SearchFilesTool, SearchHistoryTool, SendFileTool, ServiceStatusTool,
    ShareMemoryTool, SpawnAgentTool, SystemInfoTool, TerminalTool, ToolTraceTool, WebFetchTool,
    WebSearchTool, WriteFileTool,
};
use crate::traits::store_prelude::*;
use crate::traits::Tool;
use crate::types::MediaMessage;

pub struct BaseToolsBundle {
    pub tools: Vec<Arc<dyn Tool>>,
    pub execution_backend: crate::execution::SharedExecutionBackend,
    pub approval_tx: ApprovalBroker,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumCount, strum::EnumIter)]
enum BaseToolId {
    SystemInfo,
    Terminal,
    RememberFact,
    ShareMemory,
    ManageMemories,
    ManageMandates,
    SearchHistory,
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

struct BaseToolSpec {
    id: BaseToolId,
    name: &'static str,
}

const BASE_TOOL_REGISTRY: &[BaseToolSpec] = &[
    BaseToolSpec {
        id: BaseToolId::SystemInfo,
        name: "system_info",
    },
    BaseToolSpec {
        id: BaseToolId::Terminal,
        name: "terminal",
    },
    BaseToolSpec {
        id: BaseToolId::RememberFact,
        name: "remember_fact",
    },
    BaseToolSpec {
        id: BaseToolId::ShareMemory,
        name: "share_memory",
    },
    BaseToolSpec {
        id: BaseToolId::ManageMemories,
        name: "manage_memories",
    },
    BaseToolSpec {
        id: BaseToolId::ManageMandates,
        name: "manage_mandates",
    },
    BaseToolSpec {
        id: BaseToolId::SearchHistory,
        name: "search_history",
    },
    BaseToolSpec {
        id: BaseToolId::ScheduledGoalRuns,
        name: "scheduled_goal_runs",
    },
    BaseToolSpec {
        id: BaseToolId::GoalTrace,
        name: "goal_trace",
    },
    BaseToolSpec {
        id: BaseToolId::ToolTrace,
        name: "tool_trace",
    },
    BaseToolSpec {
        id: BaseToolId::PolicyMetrics,
        name: "policy_metrics",
    },
    BaseToolSpec {
        id: BaseToolId::ConfigManager,
        name: "manage_config",
    },
    BaseToolSpec {
        id: BaseToolId::WebFetch,
        name: "web_fetch",
    },
    BaseToolSpec {
        id: BaseToolId::WebSearch,
        name: "web_search",
    },
    BaseToolSpec {
        id: BaseToolId::ReadFile,
        name: "read_file",
    },
    BaseToolSpec {
        id: BaseToolId::WriteFile,
        name: "write_file",
    },
    BaseToolSpec {
        id: BaseToolId::EditFile,
        name: "edit_file",
    },
    BaseToolSpec {
        id: BaseToolId::SearchFiles,
        name: "search_files",
    },
    BaseToolSpec {
        id: BaseToolId::ProjectInspect,
        name: "project_inspect",
    },
    BaseToolSpec {
        id: BaseToolId::RunCommand,
        name: "run_command",
    },
    BaseToolSpec {
        id: BaseToolId::GitInfo,
        name: "git_info",
    },
    BaseToolSpec {
        id: BaseToolId::GitCommit,
        name: "git_commit",
    },
    BaseToolSpec {
        id: BaseToolId::CheckEnvironment,
        name: "check_environment",
    },
    BaseToolSpec {
        id: BaseToolId::ServiceStatus,
        name: "service_status",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumCount, strum::EnumIter)]
enum OptionalToolId {
    Diagnose,
    #[cfg(feature = "browser")]
    Browser,
    #[cfg(feature = "computer_use")]
    ComputerUse,
    GenerateImage,
    SendFile,
    CliAgents,
    HealthProbe,
    #[cfg(feature = "slack")]
    SlackChannelHistory,
}

struct OptionalToolSpec {
    id: OptionalToolId,
    name: &'static str,
    enabled_if: fn(&AppConfig) -> bool,
}

fn optional_enabled_diagnostics(config: &AppConfig) -> bool {
    config.diagnostics.enabled
}

#[cfg(feature = "browser")]
fn optional_enabled_browser(config: &AppConfig) -> bool {
    config.browser.enabled
}

#[cfg(feature = "computer_use")]
fn optional_enabled_computer_use(config: &AppConfig) -> bool {
    config.computer_use.enabled
}

fn optional_enabled_files(config: &AppConfig) -> bool {
    config.files.enabled
}

fn optional_enabled_cli_agents(config: &AppConfig) -> bool {
    config.cli_agents.enabled
}

fn optional_enabled_always(_: &AppConfig) -> bool {
    true
}

const OPTIONAL_TOOL_REGISTRY: &[OptionalToolSpec] = &[
    OptionalToolSpec {
        id: OptionalToolId::Diagnose,
        name: "diagnose",
        enabled_if: optional_enabled_diagnostics,
    },
    #[cfg(feature = "browser")]
    OptionalToolSpec {
        id: OptionalToolId::Browser,
        name: "browser",
        enabled_if: optional_enabled_browser,
    },
    #[cfg(feature = "computer_use")]
    OptionalToolSpec {
        id: OptionalToolId::ComputerUse,
        name: "computer_use",
        enabled_if: optional_enabled_computer_use,
    },
    OptionalToolSpec {
        id: OptionalToolId::GenerateImage,
        name: "generate_image",
        enabled_if: optional_enabled_files,
    },
    OptionalToolSpec {
        id: OptionalToolId::SendFile,
        name: "send_file",
        enabled_if: optional_enabled_files,
    },
    OptionalToolSpec {
        id: OptionalToolId::CliAgents,
        name: "cli_agents",
        enabled_if: optional_enabled_cli_agents,
    },
    OptionalToolSpec {
        id: OptionalToolId::HealthProbe,
        name: "health_probe",
        enabled_if: optional_enabled_always,
    },
    #[cfg(feature = "slack")]
    OptionalToolSpec {
        id: OptionalToolId::SlackChannelHistory,
        name: "read_channel_history",
        enabled_if: optional_enabled_always,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::EnumCount, strum::EnumIter)]
enum RuntimeToolId {
    ManageMcp,
    ManagePeople,
    HttpRequest,
    ManageHttpAuth,
    ManageOauth,
    ManageApi,
    SpawnAgent,
}

struct RuntimeToolSpec {
    id: RuntimeToolId,
    name: &'static str,
    enabled_if: fn(&AppConfig) -> bool,
    depends_on: &'static [RuntimeToolId],
}

fn runtime_enabled_always(_: &AppConfig) -> bool {
    true
}

fn runtime_enabled_http_request(_: &AppConfig) -> bool {
    true
}

fn runtime_enabled_oauth(_: &AppConfig) -> bool {
    true
}

fn runtime_enabled_spawn(config: &AppConfig) -> bool {
    config.subagents.enabled
}

const RUNTIME_TOOL_MANIFEST: &[RuntimeToolSpec] = &[
    RuntimeToolSpec {
        id: RuntimeToolId::ManageMcp,
        name: "manage_mcp",
        enabled_if: runtime_enabled_always,
        depends_on: &[],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManagePeople,
        name: "manage_people",
        enabled_if: runtime_enabled_always,
        depends_on: &[],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::HttpRequest,
        name: "http_request",
        enabled_if: runtime_enabled_http_request,
        depends_on: &[],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManageHttpAuth,
        name: "manage_http_auth",
        enabled_if: runtime_enabled_always,
        depends_on: &[RuntimeToolId::HttpRequest],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManageOauth,
        name: "manage_oauth",
        enabled_if: runtime_enabled_oauth,
        depends_on: &[RuntimeToolId::HttpRequest],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManageApi,
        name: "manage_api",
        enabled_if: runtime_enabled_always,
        depends_on: &[RuntimeToolId::HttpRequest, RuntimeToolId::ManageOauth],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::SpawnAgent,
        name: "spawn_agent",
        enabled_if: runtime_enabled_spawn,
        depends_on: &[],
    },
];

fn runtime_tool_name(manifest: &[RuntimeToolSpec], id: RuntimeToolId) -> &'static str {
    manifest
        .iter()
        .find(|spec| spec.id == id)
        .map(|spec| spec.name)
        .unwrap_or("<unknown>")
}

fn validate_runtime_manifest(manifest: &[RuntimeToolSpec]) -> anyhow::Result<()> {
    let mut seen_ids: HashSet<RuntimeToolId> = HashSet::new();
    let mut seen_names: HashSet<&'static str> = HashSet::new();
    for spec in manifest {
        anyhow::ensure!(
            seen_ids.insert(spec.id),
            "runtime manifest is invalid: duplicate tool id {}",
            spec.name
        );
        anyhow::ensure!(
            seen_names.insert(spec.name),
            "runtime manifest is invalid: duplicate tool name {}",
            spec.name
        );
    }

    let defined_ids: HashSet<RuntimeToolId> = manifest.iter().map(|spec| spec.id).collect();
    let mut declared: HashSet<RuntimeToolId> = HashSet::new();
    for spec in manifest {
        for dep in spec.depends_on {
            anyhow::ensure!(
                defined_ids.contains(dep),
                "runtime manifest is invalid: tool {} declares unknown dependency {}",
                spec.name,
                runtime_tool_name(manifest, *dep)
            );
            anyhow::ensure!(
                declared.contains(dep),
                "runtime manifest is invalid: tool {} depends on {} but dependency appears after it",
                spec.name,
                runtime_tool_name(manifest, *dep)
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
    let execution_backend = crate::execution::install_execution_backend(
        crate::execution::build_execution_backend(&config.execution, &config_path).await?,
    )?;
    info!(
        backend = execution_backend.kind().as_str(),
        backend_id = execution_backend.id(),
        workspace = execution_backend.workspace_root().as_str(),
        confined = !execution_backend.allows_outside_workspace(),
        "Initialized agent execution environment"
    );
    let checkpoint_manager = crate::checkpoints::install_manager(
        &config.checkpoints,
        state.pool(),
        event_store.clone(),
        execution_backend.clone(),
    )
    .await?;

    let (approval_tx, approval_rx) = mpsc::channel(approval_queue_capacity);
    let approval_tx = ApprovalBroker::new(approval_tx).with_event_store(event_store.clone());
    let (media_tx, media_rx) = mpsc::channel::<MediaMessage>(media_queue_capacity);

    let mut tools: Vec<Arc<dyn Tool>> = Vec::with_capacity(BASE_TOOL_REGISTRY.len());
    let mut terminal_tool: Option<Arc<TerminalTool>> = None;

    for spec in BASE_TOOL_REGISTRY {
        if !config.tools.is_enabled(spec.name) {
            info!(
                tool = spec.name,
                "Skipped base tool from startup manifest (disabled in [tools].disabled)"
            );
            continue;
        }
        let built = build_base_tool(
            spec.id,
            config,
            &config_path,
            state.clone(),
            event_store.clone(),
            approval_tx.clone(),
        )
        .await?;
        info!(
            tool = spec.name,
            "Registered base tool from startup manifest"
        );
        if terminal_tool.is_none() {
            terminal_tool = built.terminal_tool.clone();
        }
        tools.push(built.tool);
    }
    if checkpoint_manager.is_some() && config.tools.is_enabled("list_checkpoints") {
        tools.push(Arc::new(ListCheckpointsTool));
        info!("Registered list_checkpoints for the enabled checkpoint manager");
    }

    Ok(BaseToolsBundle {
        tools,
        execution_backend,
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
    approval_tx: ApprovalBroker,
    media_tx: mpsc::Sender<MediaMessage>,
) -> anyhow::Result<OptionalToolsOutcome> {
    let inbox_dir = shellexpand::tilde(&config.files.inbox_dir).to_string();
    let mut has_cli_agents = false;
    let mut cli_agent_tool: Option<Arc<CliAgentTool>> = None;

    for spec in OPTIONAL_TOOL_REGISTRY {
        if !(spec.enabled_if)(config) {
            continue;
        }

        let tools_before = tools.len();
        match spec.id {
            OptionalToolId::Diagnose => {
                if !config.tools.is_enabled("self_diagnose") {
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
                if !config.tools.is_enabled("browser") {
                    continue;
                }
                let browser_tool = BrowserTool::new(
                    config.browser.clone(),
                    media_tx.clone(),
                    approval_tx.clone(),
                    inbox_dir.clone(),
                )
                .map_err(|e| anyhow::anyhow!("Invalid browser configuration: {e}"))?;
                tools.push(Arc::new(browser_tool));
            }
            #[cfg(feature = "computer_use")]
            OptionalToolId::ComputerUse => {
                if !config.tools.is_enabled("computer_use") {
                    continue;
                }
                std::fs::create_dir_all(&inbox_dir)?;
                let vision = crate::config::VisionConfig::from_files(&config.files);
                tools.push(Arc::new(ComputerUseTool::new(
                    config.computer_use.clone(),
                    vision,
                    std::path::PathBuf::from(&inbox_dir),
                    approval_tx.clone(),
                    media_tx.clone(),
                    Some(event_store.clone()),
                )));
            }
            OptionalToolId::SendFile => {
                if !config.tools.is_enabled("send_file") {
                    continue;
                }
                std::fs::create_dir_all(&inbox_dir)?;
                tools.push(Arc::new(
                    SendFileTool::new(media_tx.clone(), &config.files.outbox_dirs, &inbox_dir)
                        .with_event_store(event_store.clone()),
                ));
            }
            OptionalToolId::GenerateImage => {
                if !config.tools.is_enabled("generate_image") {
                    continue;
                }
                std::fs::create_dir_all(&inbox_dir)?;
                tools.push(Arc::new(
                    GenerateImageTool::chatgpt_subscription(PathBuf::from(&inbox_dir))
                        .map_err(anyhow::Error::msg)?
                        .with_event_store(event_store.clone()),
                ));
            }
            OptionalToolId::CliAgents => {
                let register_cli = config.tools.is_enabled("cli_agent");
                let register_manage_cli = config.tools.is_enabled("manage_cli_agents");
                if !register_cli && !register_manage_cli {
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
                // Always register cli_agent so runtime-added agents become
                // immediately usable without restart.
                if register_cli {
                    tools.push(arc.clone());
                    if !has_cli_agents {
                        info!(
                            "CLI agents enabled but no tools found on system (cli_agent registered for runtime discovery)"
                        );
                    }
                }
                if register_manage_cli {
                    let manage_cli =
                        ManageCliAgentsTool::new(arc, state.clone(), approval_tx.clone());
                    tools.push(Arc::new(manage_cli));
                }
            }
            OptionalToolId::HealthProbe => {
                if !config.tools.is_enabled("health_probe") {
                    continue;
                }
                let Some(store) = &health_store else {
                    continue;
                };
                tools.push(Arc::new(HealthProbeTool::new(store.clone())));
            }
            #[cfg(feature = "slack")]
            OptionalToolId::SlackChannelHistory => {
                if !config.tools.is_enabled("read_channel_history") {
                    continue;
                }
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

        // Log the tools actually pushed this iteration by their real schema
        // names. The `CliAgents` arm registers two tools (cli_agent +
        // manage_cli_agents), and the early-`continue` arms register none, so
        // reporting `spec.name` (a config-group pseudo-name) would mislead.
        for tool in &tools[tools_before..] {
            info!(
                group = spec.name,
                tool = tool.name(),
                "Registered optional tool from startup manifest"
            );
        }
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
    config_path: &Path,
    http_profiles: crate::oauth::SharedHttpProfiles,
    state: Arc<SqliteStateStore>,
    mcp_registry: McpRegistry,
    approval_tx: ApprovalBroker,
) -> anyhow::Result<RuntimeToolsOutcome> {
    validate_runtime_manifest(RUNTIME_TOOL_MANIFEST)?;

    let mut oauth_gateway: Option<crate::oauth::OAuthGateway> = None;
    let mut spawn_tool: Option<Arc<SpawnAgentTool>> = None;
    let mut http_request_tool: Option<Arc<HttpRequestTool>> = None;
    let mut registered_runtime_tools: std::collections::HashSet<RuntimeToolId> =
        std::collections::HashSet::new();

    for spec in RUNTIME_TOOL_MANIFEST {
        if !(spec.enabled_if)(config) {
            info!(
                tool = spec.name,
                "Skipped runtime tool from startup manifest (condition disabled)"
            );
            continue;
        }

        let missing_dependencies: Vec<&'static str> = spec
            .depends_on
            .iter()
            .filter(|dep| !registered_runtime_tools.contains(dep))
            .map(|dep| runtime_tool_name(RUNTIME_TOOL_MANIFEST, *dep))
            .collect();
        if !missing_dependencies.is_empty() {
            info!(
                tool = spec.name,
                missing_dependencies = ?missing_dependencies,
                "Skipped runtime tool from startup manifest (dependencies not satisfied)"
            );
            continue;
        }

        if !config.tools.is_enabled(spec.name) {
            info!(
                tool = spec.name,
                "Skipped runtime tool from startup manifest (disabled in [tools].disabled)"
            );
            continue;
        }

        register_runtime_tool_by_id(
            spec.id,
            tools,
            config,
            config_path,
            state.clone(),
            mcp_registry.clone(),
            approval_tx.clone(),
            http_profiles.clone(),
            &mut http_request_tool,
            &mut spawn_tool,
            &mut oauth_gateway,
        )
        .await?;
        registered_runtime_tools.insert(spec.id);
        info!(
            tool = spec.name,
            dependencies = ?spec.depends_on.iter().map(|dep| runtime_tool_name(RUNTIME_TOOL_MANIFEST, *dep)).collect::<Vec<_>>(),
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
    config_path: &Path,
    state: Arc<SqliteStateStore>,
    mcp_registry: McpRegistry,
    approval_tx: ApprovalBroker,
    http_profiles: crate::oauth::SharedHttpProfiles,
    http_request_tool: &mut Option<Arc<HttpRequestTool>>,
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
            let tool = Arc::new(HttpRequestTool::new(http_profiles, approval_tx));
            tools.push(tool.clone());
            *http_request_tool = Some(tool);
        }
        RuntimeToolId::ManageHttpAuth => {
            tools.push(Arc::new(ManageHttpAuthTool::new(
                config_path.to_path_buf(),
                http_profiles,
                approval_tx,
                state as Arc<dyn crate::traits::StateStore>,
            )));
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

            if let Some(http_tool) = http_request_tool.as_ref() {
                http_tool.set_oauth_gateway(gateway.clone()).await;
            }

            tools.push(Arc::new(ManageOAuthTool::new(
                gateway.clone(),
                state.clone() as Arc<dyn crate::traits::StateStore>,
                config_path.to_path_buf(),
                approval_tx,
            )));
            *oauth_gateway = Some(gateway);
        }
        RuntimeToolId::ManageApi => {
            let gateway = oauth_gateway.clone().ok_or_else(|| {
                anyhow::anyhow!("manage_api requires manage_oauth to be initialized first")
            })?;
            let skills_dir = if config.skills.enabled {
                let dir = config_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(&config.skills.dir);
                std::fs::create_dir_all(&dir).ok();
                Some(dir)
            } else {
                None
            };

            tools.push(Arc::new(ManageApiTool::new(
                config_path.to_path_buf(),
                skills_dir,
                config.skills.registries.clone(),
                config.search.clone(),
                http_profiles,
                approval_tx,
                state.clone() as Arc<dyn crate::traits::StateStore>,
                gateway,
            )));
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
    approval_tx: ApprovalBroker,
) -> anyhow::Result<BuiltBaseTool> {
    let built = match tool_id {
        BaseToolId::SystemInfo => BuiltBaseTool {
            tool: Arc::new(SystemInfoTool),
            terminal_tool: None,
        },
        BaseToolId::Terminal => {
            // Mirror the dirs `send_file` is constructed with so harness-side
            // background deliverable delivery routes through the same shared
            // validation/recovery (`prepare_delivery`).
            let inbox_dir = PathBuf::from(shellexpand::tilde(&config.files.inbox_dir).to_string());
            let outbox_dirs: Vec<PathBuf> = config
                .files
                .outbox_dirs
                .iter()
                .map(|d| PathBuf::from(shellexpand::tilde(d).to_string()))
                .collect();
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
                .with_state(state as Arc<dyn crate::traits::StateStore>)
                .with_self_correction(config.self_correction.clone())
                .with_delivery_dirs(inbox_dir, outbox_dirs),
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
            tool: Arc::new(ManageMemoriesTool::new(state).with_approval_tx(approval_tx.clone())),
            terminal_tool: None,
        },
        BaseToolId::ManageMandates => BuiltBaseTool {
            tool: Arc::new(ManageMandatesTool::new(state, approval_tx).with_skills_dir(
                config.skills.enabled.then(|| {
                    config_path
                        .parent()
                        .unwrap_or_else(|| std::path::Path::new("."))
                        .join(&config.skills.dir)
                }),
            )),
            terminal_tool: None,
        },
        BaseToolId::SearchHistory => BuiltBaseTool {
            tool: Arc::new(SearchHistoryTool::new(
                state,
                config.state.retention.messages_days,
            )),
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
        // Validate the full manifest⇄construction contract: every base tool must
        // be constructible. The default `[tools].disabled` set would otherwise
        // omit several base tools and break the positional alignment with
        // BASE_TOOL_REGISTRY. The disable behavior itself is covered by
        // `disabled_tools_are_omitted_from_base_registry`.
        config.tools.disabled.clear();

        let io_temp = TempDir::new()?;
        config.files.inbox_dir = io_temp.path().join("inbox").display().to_string();
        config.files.outbox_dirs = vec![io_temp.path().join("outbox").display().to_string()];

        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::new()?);
        let state = Arc::new(SqliteStateStore::new(&db_path, 100, None, embedding_service).await?);
        let event_store = Arc::new(EventStore::new(state.pool()).await?);
        let config_file = NamedTempFile::new()?;

        let model_provider = Arc::new(MockProvider::new()) as Arc<dyn ModelProvider>;
        let mut bundle = build_base_tools(
            &config,
            config_file.path().to_path_buf(),
            state.clone(),
            event_store.clone(),
            32,
            8,
        )
        .await?;
        let llm_runtime = SharedLlmRuntime::new(
            model_provider,
            router_from_models(config.provider.models.clone()),
            config.provider.kind,
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
        let http_profiles: crate::oauth::SharedHttpProfiles =
            Arc::new(tokio::sync::RwLock::new(config.http_auth.clone()));
        let _runtime = register_runtime_tools(
            &mut bundle.tools,
            &config,
            config_file.path(),
            http_profiles,
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
        for tool in BASE_TOOL_REGISTRY {
            assert!(
                seen.insert(tool.name),
                "duplicate tool name in manifest: {}",
                tool.name
            );
        }
    }

    #[tokio::test]
    async fn base_tool_registry_names_match_built_schema_names() {
        let schemas = build_tool_schemas_for_contract_validation(false, false, false)
            .await
            .unwrap();
        let built_base_names: Vec<&str> = schemas
            .iter()
            .take(BASE_TOOL_REGISTRY.len())
            .map(|schema| {
                schema["function"]["name"]
                    .as_str()
                    .expect("built tool schema should expose a function name")
            })
            .collect();
        let registry_names: Vec<&str> = BASE_TOOL_REGISTRY.iter().map(|tool| tool.name).collect();

        assert_eq!(built_base_names, registry_names);
    }

    // Completeness guards: every enum variant MUST have a registry entry.
    // Without these, adding a `BaseToolId`/`OptionalToolId`/`RuntimeToolId`
    // variant (plus its construction match arm) but forgetting the registry
    // entry compiles cleanly and passes the name tests — the tool is silently
    // never registered. The `EnumCount`/`EnumIter` derives make the registry's
    // completeness checkable: a new variant bumps `COUNT` and these fail until
    // the registry is updated.
    #[test]
    fn base_registry_covers_every_variant() {
        use strum::{EnumCount, IntoEnumIterator};
        assert_eq!(
            BASE_TOOL_REGISTRY.len(),
            BaseToolId::COUNT,
            "BASE_TOOL_REGISTRY is missing an entry for a BaseToolId variant"
        );
        for id in BaseToolId::iter() {
            assert!(
                BASE_TOOL_REGISTRY.iter().any(|spec| spec.id == id),
                "BaseToolId::{id:?} has no BASE_TOOL_REGISTRY entry (silently unregistered)"
            );
        }
    }

    #[test]
    fn optional_registry_covers_every_variant() {
        use strum::{EnumCount, IntoEnumIterator};
        assert_eq!(
            OPTIONAL_TOOL_REGISTRY.len(),
            OptionalToolId::COUNT,
            "OPTIONAL_TOOL_REGISTRY is missing an entry for an OptionalToolId variant"
        );
        for id in OptionalToolId::iter() {
            assert!(
                OPTIONAL_TOOL_REGISTRY.iter().any(|spec| spec.id == id),
                "OptionalToolId::{id:?} has no OPTIONAL_TOOL_REGISTRY entry (silently unregistered)"
            );
        }
    }

    #[test]
    fn runtime_registry_covers_every_variant() {
        use strum::{EnumCount, IntoEnumIterator};
        assert_eq!(
            RUNTIME_TOOL_MANIFEST.len(),
            RuntimeToolId::COUNT,
            "RUNTIME_TOOL_MANIFEST is missing an entry for a RuntimeToolId variant"
        );
        for id in RuntimeToolId::iter() {
            assert!(
                RUNTIME_TOOL_MANIFEST.iter().any(|spec| spec.id == id),
                "RuntimeToolId::{id:?} has no RUNTIME_TOOL_MANIFEST entry (silently unregistered)"
            );
        }
    }

    #[test]
    fn optional_manifest_has_unique_tool_names() {
        let mut seen = HashSet::new();
        for tool in OPTIONAL_TOOL_REGISTRY {
            assert!(
                seen.insert(tool.name),
                "duplicate optional tool name in manifest: {}",
                tool.name
            );
        }
    }

    #[test]
    fn runtime_manifest_has_unique_tool_names() {
        let mut seen = HashSet::new();
        for tool in RUNTIME_TOOL_MANIFEST {
            assert!(
                seen.insert(tool.name),
                "duplicate runtime tool name in manifest: {}",
                tool.name
            );
        }
    }

    #[test]
    fn runtime_manifest_matches_expected_pipeline_order() {
        let names: Vec<&'static str> = RUNTIME_TOOL_MANIFEST.iter().map(|tool| tool.name).collect();
        assert_eq!(
            names,
            vec![
                "manage_mcp",
                "manage_people",
                "http_request",
                "manage_http_auth",
                "manage_oauth",
                "manage_api",
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
                    spec.name,
                    runtime_tool_name(RUNTIME_TOOL_MANIFEST, *dep)
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
                name: "manage_mcp",
                enabled_if: runtime_enabled_always,
                depends_on: &[],
            },
            RuntimeToolSpec {
                id: RuntimeToolId::ManageMcp,
                name: "manage_mcp_duplicate",
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
            name: "manage_oauth",
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
                name: "manage_oauth",
                enabled_if: runtime_enabled_always,
                depends_on: &[RuntimeToolId::HttpRequest],
            },
            RuntimeToolSpec {
                id: RuntimeToolId::HttpRequest,
                name: "http_request",
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

    #[tokio::test]
    async fn disabled_tools_are_omitted_from_base_registry() {
        let mut config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test-key"
            [tools]
            disabled = ["policy_metrics", "goal_trace", "check_environment"]
            "#,
        )
        .unwrap();
        config.provider.models.apply_defaults(&config.provider.kind);

        let db_file = NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(
                &db_file.path().display().to_string(),
                100,
                None,
                embedding_service,
            )
            .await
            .unwrap(),
        );
        let event_store = Arc::new(EventStore::new(state.pool()).await.unwrap());
        let config_file = NamedTempFile::new().unwrap();

        let bundle = build_base_tools(
            &config,
            config_file.path().to_path_buf(),
            state,
            event_store,
            8,
            8,
        )
        .await
        .unwrap();

        let names: Vec<&str> = bundle.tools.iter().map(|t| t.name()).collect();
        assert!(!names.contains(&"policy_metrics"));
        assert!(!names.contains(&"goal_trace"));
        assert!(!names.contains(&"check_environment"));
        assert!(names.contains(&"terminal"));
    }

    #[tokio::test]
    async fn generic_api_tools_register_without_existing_profiles() {
        let schemas = build_tool_schemas_for_contract_validation(true, true, true)
            .await
            .unwrap();
        let names: Vec<String> = schemas
            .iter()
            .filter_map(|schema| schema["function"]["name"].as_str())
            .map(ToString::to_string)
            .collect();

        assert!(names.contains(&"manage_api".to_string()));
        assert!(names.contains(&"http_request".to_string()));
        assert!(names.contains(&"manage_http_auth".to_string()));
        assert!(names.contains(&"manage_oauth".to_string()));
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
