use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::info;

use crate::config::AppConfig;
use crate::events::EventStore;
use crate::health::HealthProbeStore;
use crate::mcp::McpRegistry;
use crate::nodes::tool::{
    ManageNodeMonitorsTool, ReadNodeHealthTool, ReadNodeSensorsTool, SendNodeAudioTool,
};
use crate::tools::ApprovalBroker;
#[cfg(feature = "browser")]
use crate::tools::BrowserTool;
#[cfg(feature = "computer_use")]
use crate::tools::ComputerUseTool;
#[cfg(feature = "slack")]
use crate::tools::ReadChannelHistoryTool;
use crate::tools::terminal::ApprovalRequest;
use crate::tools::{
    CheckEnvironmentTool, CliAgentTool, ConfigManagerTool, DiagnoseTool, EditFileTool,
    GenerateImageTool, GoalTraceTool, HealthProbeTool, HttpRequestTool, ListCheckpointsTool,
    ManageApiTool, ManageCliAgentsTool, ManageHttpAuthTool, ManageMandatesTool, ManageMcpTool,
    ManageMemoriesTool, ManageOAuthTool, ManagePeopleTool, PolicyMetricsTool, ReadFileTool,
    RememberFactTool, SearchFilesTool, SearchHistoryTool, SendFileTool, SpawnAgentTool,
    SystemInfoTool, TerminalTool, WebFetchTool, WebSearchTool, WriteFileTool,
};
use crate::traits::Tool;
use crate::traits::store_prelude::*;
use crate::types::MediaMessage;
use sqlx::SqlitePool;

/// Complete tool subsystem assembled during startup.
///
/// Keeping this bundle in the startup layer means the application composition
/// root does not need to know how base, optional, runtime, MCP, or skill tools
/// are constructed. The runtime only consumes the capabilities it needs after
/// this point.
pub struct ToolSetup {
    pub tools: Vec<Arc<dyn Tool>>,
    pub approval_tx: ApprovalBroker,
    pub approval_rx: mpsc::Receiver<ApprovalRequest>,
    pub media_rx: mpsc::Receiver<MediaMessage>,
    pub terminal_tool: Option<Arc<TerminalTool>>,
    pub spawn_tool: Option<Arc<SpawnAgentTool>>,
    pub oauth_gateway: Option<crate::oauth::OAuthGateway>,
    pub mcp_registry: McpRegistry,
    pub skills_dir: Option<PathBuf>,
    pub inbox_dir: String,
    pub cli_agent_tool: Option<Arc<CliAgentTool>>,
}

/// Domain-facing dependencies required to assemble tools.
///
/// The tool layer receives the state contract and the shared database handle,
/// not the concrete persistence implementation. This keeps SQLite-specific
/// construction inside `startup::stores` while still allowing infrastructure
/// tools to use the pool directly where their APIs require it.
/// How tools reach the daemon's own memory.
#[derive(Clone)]
pub struct ToolMemoryAccess {
    pub store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    /// False when another memory system replaced the daemon's. Its tools are
    /// hidden then, and tools outside memory stop reading the daemon's store.
    pub own_memory: bool,
}

pub struct ToolSetupDependencies {
    pub execution_backend: crate::execution::SharedExecutionBackend,
    pub state: Arc<dyn StateStore>,
    pub memory: ToolMemoryAccess,
    pub pool: SqlitePool,
    pub event_store: Arc<EventStore>,
    pub side_calls: crate::llm_runtime::gateway::SideCallGateway,
    pub health_store: Option<Arc<HealthProbeStore>>,
    pub approval_capacity: usize,
    pub media_capacity: usize,
}

/// Assemble all tools and their startup-only dependencies.
///
/// This is intentionally the only function that coordinates the individual
/// tool registration phases. Callers should treat the result as a capability
/// bundle instead of reaching into the registration details.
pub async fn setup_tools(
    config: &AppConfig,
    config_path: &Path,
    daemon_paths: crate::tools::DaemonPaths,
    dependencies: ToolSetupDependencies,
) -> anyhow::Result<ToolSetup> {
    let ToolSetupDependencies {
        execution_backend,
        state,
        memory,
        pool,
        event_store,
        side_calls,
        health_store,
        approval_capacity,
        media_capacity,
    } = dependencies;
    let BaseToolsBundle {
        mut tools,
        approval_tx,
        approval_rx,
        media_tx,
        media_rx,
        terminal_tool,
    } = build_base_tools(
        config,
        config_path.to_path_buf(),
        daemon_paths,
        execution_backend.clone(),
        state.clone(),
        memory.clone(),
        pool.clone(),
        event_store.clone(),
        approval_capacity,
        media_capacity,
        health_store.clone(),
    )
    .await?;

    if let Some(terminal) = &terminal_tool {
        terminal.set_side_calls(side_calls);
        crate::startup::trust_all::publish(terminal.clone());
    }

    let OptionalToolsOutcome {
        has_cli_agents: _has_cli_agents,
        inbox_dir,
        cli_agent_tool,
    } = register_optional_tools(
        &mut tools,
        config,
        state.clone(),
        memory.clone(),
        pool.clone(),
        event_store,
        health_store,
        media_tx.clone(),
        execution_backend.clone(),
    )
    .await?;

    let mcp_registry = crate::startup::mcp::setup_mcp_registry(
        config,
        state.clone() as Arc<dyn crate::traits::McpRegistryStore>,
    )
    .await?;
    let http_profiles: crate::oauth::SharedHttpProfiles =
        Arc::new(tokio::sync::RwLock::new(config.http_auth.clone()));

    let skills_dir = crate::startup::skills::register_skills_tools(
        config,
        config_path,
        http_profiles.clone(),
        state.clone() as Arc<dyn crate::traits::SkillsStore>,
        &mut tools,
    )
    .await?;

    let RuntimeToolsOutcome {
        spawn_tool,
        oauth_gateway,
    } = register_runtime_tools(
        &mut tools,
        config,
        config_path,
        http_profiles,
        state,
        memory.clone(),
        mcp_registry.clone(),
    )
    .await?;

    for tool in &tools {
        info!(
            name = tool.name(),
            desc = tool.description(),
            "Registered tool"
        );
    }

    Ok(ToolSetup {
        tools,
        approval_tx,
        approval_rx,
        media_rx,
        terminal_tool,
        spawn_tool,
        oauth_gateway,
        mcp_registry,
        skills_dir,
        inbox_dir,
        cli_agent_tool,
    })
}

pub struct BaseToolsBundle {
    pub tools: Vec<Arc<dyn Tool>>,
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
    ManageMemories,
    SetReminder,
    ScheduleTask,
    ManageMandates,
    ManageAutomations,
    SearchHistory,
    ConfigManager,
    WebFetch,
    WebSearch,
    ReadFile,
    WriteFile,
    EditFile,
    SearchFiles,
    CheckEnvironment,
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
        id: BaseToolId::ManageMemories,
        name: "manage_memories",
    },
    BaseToolSpec {
        id: BaseToolId::SetReminder,
        name: "set_reminder",
    },
    BaseToolSpec {
        id: BaseToolId::ScheduleTask,
        name: "schedule_task",
    },
    BaseToolSpec {
        id: BaseToolId::ManageMandates,
        name: "manage_mandates",
    },
    BaseToolSpec {
        id: BaseToolId::ManageAutomations,
        name: "manage_automations",
    },
    BaseToolSpec {
        id: BaseToolId::SearchHistory,
        name: "search_history",
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
        id: BaseToolId::CheckEnvironment,
        name: "check_environment",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumCount, strum::EnumIter)]
enum OptionalToolId {
    Diagnose,
    GoalTrace,
    PolicyMetrics,
    NodeHealth,
    NodeSensors,
    NodeAudio,
    NodeRestart,
    NodeMonitoring,
    #[cfg(feature = "browser")]
    Browser,
    #[cfg(feature = "computer_use")]
    ComputerUse,
    GenerateImage,
    SendFile,
    CliAgents,
    HealthProbe,
    ClientView,
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

fn optional_enabled_node_sensors(config: &AppConfig) -> bool {
    config.nodes.gateway.enabled
}

fn optional_enabled_node_audio(config: &AppConfig) -> bool {
    config.nodes.gateway.enabled
        && config.nodes.announcements.enabled
        && config.nodes.speech.enabled
}

fn optional_enabled_node_restart(config: &AppConfig) -> bool {
    config.nodes.gateway.enabled
}

fn optional_enabled_node_monitoring(config: &AppConfig) -> bool {
    config.nodes.gateway.enabled && config.nodes.monitoring.enabled
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

fn optional_enabled_client_gateway(config: &AppConfig) -> bool {
    config.client_gateway.enabled
}

fn optional_enabled_always(_: &AppConfig) -> bool {
    true
}

const OPTIONAL_TOOL_REGISTRY: &[OptionalToolSpec] = &[
    OptionalToolSpec {
        id: OptionalToolId::Diagnose,
        name: "self_diagnose",
        enabled_if: optional_enabled_diagnostics,
    },
    OptionalToolSpec {
        id: OptionalToolId::GoalTrace,
        name: "goal_trace",
        enabled_if: optional_enabled_diagnostics,
    },
    OptionalToolSpec {
        id: OptionalToolId::PolicyMetrics,
        name: "policy_metrics",
        enabled_if: optional_enabled_diagnostics,
    },
    OptionalToolSpec {
        id: OptionalToolId::NodeHealth,
        name: "read_node_health",
        enabled_if: optional_enabled_node_sensors,
    },
    OptionalToolSpec {
        id: OptionalToolId::NodeSensors,
        name: "read_node_sensors",
        enabled_if: optional_enabled_node_sensors,
    },
    OptionalToolSpec {
        id: OptionalToolId::NodeAudio,
        name: "send_node_audio",
        enabled_if: optional_enabled_node_audio,
    },
    OptionalToolSpec {
        id: OptionalToolId::NodeRestart,
        name: "node_restart",
        enabled_if: optional_enabled_node_restart,
    },
    OptionalToolSpec {
        id: OptionalToolId::NodeMonitoring,
        name: "manage_node_monitors",
        enabled_if: optional_enabled_node_monitoring,
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
    OptionalToolSpec {
        id: OptionalToolId::ClientView,
        name: "client_view",
        enabled_if: optional_enabled_client_gateway,
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
    ManageAuth,
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
        id: RuntimeToolId::ManageAuth,
        name: "manage_auth",
        enabled_if: runtime_enabled_oauth,
        depends_on: &[RuntimeToolId::HttpRequest],
    },
    RuntimeToolSpec {
        id: RuntimeToolId::ManageApi,
        name: "manage_api",
        enabled_if: runtime_enabled_always,
        depends_on: &[RuntimeToolId::HttpRequest, RuntimeToolId::ManageAuth],
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

#[allow(clippy::too_many_arguments)]
pub async fn build_base_tools(
    config: &AppConfig,
    config_path: PathBuf,
    daemon_paths: crate::tools::DaemonPaths,
    execution_backend: crate::execution::SharedExecutionBackend,
    state: Arc<dyn StateStore>,
    memory: ToolMemoryAccess,
    pool: SqlitePool,
    event_store: Arc<EventStore>,
    approval_queue_capacity: usize,
    media_queue_capacity: usize,
    health_store: Option<Arc<HealthProbeStore>>,
) -> anyhow::Result<BaseToolsBundle> {
    let checkpoint_manager = aidaemon_tools::checkpoints::install_manager(
        &config.checkpoints,
        pool.clone(),
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
            execution_backend.clone(),
            daemon_paths.clone(),
            config,
            &config_path,
            state.clone(),
            memory.clone(),
            pool.clone(),
            event_store.clone(),
            approval_tx.clone(),
            health_store.clone(),
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
    state: Arc<dyn StateStore>,
    memory: ToolMemoryAccess,
    pool: SqlitePool,
    event_store: Arc<EventStore>,
    health_store: Option<Arc<HealthProbeStore>>,
    media_tx: mpsc::Sender<MediaMessage>,
    execution_backend: crate::execution::SharedExecutionBackend,
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
                    config.diagnostics.max_events,
                    config.diagnostics.include_raw_tool_args,
                )));
            }
            OptionalToolId::GoalTrace => {
                if !config.tools.is_enabled("goal_trace") {
                    continue;
                }
                tools.push(Arc::new(GoalTraceTool::new(state.clone())));
            }
            OptionalToolId::PolicyMetrics => {
                if !config.tools.is_enabled("policy_metrics") {
                    continue;
                }
                tools.push(Arc::new(PolicyMetricsTool));
            }
            OptionalToolId::NodeSensors => {
                if !config.tools.is_enabled("read_node_sensors") {
                    continue;
                }
                tools.push(Arc::new(ReadNodeSensorsTool::new(pool.clone())));
            }
            OptionalToolId::NodeHealth => {
                if !config.tools.is_enabled("read_node_health") {
                    continue;
                }
                tools.push(Arc::new(ReadNodeHealthTool::new(pool.clone())));
            }
            OptionalToolId::NodeAudio => {
                if !config.tools.is_enabled("send_node_audio") {
                    continue;
                }
                let store = Arc::new(crate::nodes::NodeStore::new(
                    pool.clone(),
                    crate::nodes::auth::load_or_create_instance_key()?,
                ));
                let speech = crate::nodes::speech::configured_synthesizer(&config.nodes.speech)?;
                let announcements = crate::nodes::announcement::NodeAnnouncementService::new(
                    store,
                    config.nodes.clone(),
                    speech,
                );
                tools.push(Arc::new(SendNodeAudioTool::new(announcements)));
            }
            OptionalToolId::NodeRestart => {
                if !config.tools.is_enabled("node_restart") {
                    continue;
                }
                let store = Arc::new(crate::nodes::NodeStore::new(
                    pool.clone(),
                    crate::nodes::auth::load_or_create_instance_key()?,
                ));
                tools.push(Arc::new(crate::nodes::restart::NodeRestartTool::new(store)));
            }
            OptionalToolId::NodeMonitoring => {
                if !config.tools.is_enabled("manage_node_monitors") {
                    continue;
                }
                let monitoring = crate::nodes::monitoring::NodeMonitoringService::new(
                    pool.clone(),
                    config.nodes.monitoring.clone(),
                );
                tools.push(Arc::new(ManageNodeMonitorsTool::new(monitoring)));
            }
            #[cfg(feature = "browser")]
            OptionalToolId::Browser => {
                if !config.tools.is_enabled("browser") {
                    continue;
                }
                let browser_tool = BrowserTool::new(
                    config.browser.settings(),
                    media_tx.clone(),
                    inbox_dir.clone(),
                    execution_backend.clone(),
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
                    media_tx.clone(),
                    Some(event_store.clone()),
                    crate::tools::computer_use::Desktop::Real,
                )));
            }
            OptionalToolId::SendFile => {
                if !config.tools.is_enabled("send_file") {
                    continue;
                }
                std::fs::create_dir_all(&inbox_dir)?;
                tools.push(Arc::new(
                    SendFileTool::new(
                        execution_backend.clone(),
                        media_tx.clone(),
                        &config.files.outbox_dirs,
                        &inbox_dir,
                    )
                    .with_event_store(event_store.clone()),
                ));
            }
            OptionalToolId::GenerateImage => {
                if !config.tools.is_enabled("generate_image") {
                    continue;
                }
                std::fs::create_dir_all(&inbox_dir)?;
                tools.push(Arc::new(
                    GenerateImageTool::chatgpt_subscription(
                        PathBuf::from(&inbox_dir),
                        crate::providers::chatgpt_login::shared_credential_manager(),
                    )
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
                let mut cli_tool = CliAgentTool::discover(
                    execution_backend.clone(),
                    config.cli_agents.clone(),
                    state.clone(),
                )
                .await;
                if memory.own_memory {
                    cli_tool = cli_tool.with_context_facts(memory.store.clone());
                }
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
                    let manage_cli = ManageCliAgentsTool::new(arc, state.clone());
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
            OptionalToolId::ClientView => {
                if !config.tools.is_enabled("client_view") {
                    continue;
                }
                tools.push(Arc::new(crate::tools::ClientViewTool::new(Arc::new(
                    crate::startup::client_gateway::GatewayViews,
                ))));
            }
            #[cfg(feature = "slack")]
            OptionalToolId::SlackChannelHistory => {
                if !config.tools.is_enabled("read_channel_history") {
                    continue;
                }
                // Opted in, so registered even before any Slack bot exists:
                // the tokens are the configured bots plus the bots connected
                // at runtime, read on every call, so a bot `/connect`ed after
                // startup is seen.
                let configured: Vec<String> = config
                    .all_slack_bots()
                    .iter()
                    .map(|bot| bot.bot_token.clone())
                    .collect();
                let connected: Arc<dyn aidaemon_channels::hub::DynamicBotStore> = state.clone();
                info!(
                    configured = configured.len(),
                    "Channel history tool enabled"
                );
                tools.push(Arc::new(ReadChannelHistoryTool::with_tokens(
                    crate::tools::SlackBotTokens::with_connected(configured, connected),
                )));
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
    state: Arc<dyn StateStore>,
    memory: ToolMemoryAccess,
    mcp_registry: McpRegistry,
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
            memory.clone(),
            mcp_registry.clone(),
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
    state: Arc<dyn StateStore>,
    memory: ToolMemoryAccess,
    mcp_registry: McpRegistry,
    http_profiles: crate::oauth::SharedHttpProfiles,
    http_request_tool: &mut Option<Arc<HttpRequestTool>>,
    spawn_tool: &mut Option<Arc<SpawnAgentTool>>,
    oauth_gateway: &mut Option<crate::oauth::OAuthGateway>,
) -> anyhow::Result<()> {
    match tool_id {
        RuntimeToolId::ManageMcp => {
            let manage_mcp = ManageMcpTool::new(mcp_registry);
            tools.push(Arc::new(manage_mcp));
        }
        RuntimeToolId::ManagePeople => {
            tools.push(Arc::new(ManagePeopleTool::new(
                state.clone(),
                memory.store.clone(),
            )));
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
            let tool = Arc::new(HttpRequestTool::new(http_profiles));
            tools.push(tool.clone());
            *http_request_tool = Some(tool);
        }
        RuntimeToolId::ManageAuth => {
            let manage_http_auth = ManageHttpAuthTool::new(
                crate::config::config_file(config_path),
                http_profiles.clone(),
                state.clone() as Arc<dyn crate::traits::StateStore>,
                crate::config::shared_secret_store(),
            );
            let callback_url = config
                .oauth
                .callback_url
                .clone()
                .unwrap_or_else(|| format!("http://localhost:{}", config.daemon.health_port));
            let gateway = crate::startup::oauth::daemon_gateway(
                state.clone() as Arc<dyn crate::traits::OAuthGatewayStore>,
                http_profiles,
                callback_url,
            );

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

            let manage_oauth = ManageOAuthTool::new(
                gateway.clone(),
                state.clone() as Arc<dyn crate::traits::StateStore>,
                crate::config::config_file(config_path),
            );
            tools.push(Arc::new(
                aidaemon_tools::services::manage_auth::ManageAuthTool::new(
                    manage_http_auth,
                    manage_oauth,
                ),
            ));
            *oauth_gateway = Some(gateway);
        }
        RuntimeToolId::ManageApi => {
            let gateway = oauth_gateway.clone().ok_or_else(|| {
                anyhow::anyhow!("manage_api requires manage_auth to be initialized first")
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
                crate::config::config_file(config_path),
                skills_dir,
                config.skills.registries.clone(),
                config.search.clone(),
                http_profiles,
                state.clone() as Arc<dyn crate::traits::OAuthStore>,
                state.clone() as Arc<dyn crate::traits::SkillsStore>,
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

#[allow(clippy::too_many_arguments)]
async fn build_base_tool(
    tool_id: BaseToolId,
    execution_backend: crate::execution::SharedExecutionBackend,
    daemon_paths: crate::tools::DaemonPaths,
    config: &AppConfig,
    config_path: &Path,
    state: Arc<dyn StateStore>,
    memory: ToolMemoryAccess,
    pool: SqlitePool,
    event_store: Arc<EventStore>,
    approval_tx: ApprovalBroker,
    health_store: Option<Arc<HealthProbeStore>>,
) -> anyhow::Result<BuiltBaseTool> {
    let built = match tool_id {
        BaseToolId::SystemInfo => BuiltBaseTool {
            tool: Arc::new(SystemInfoTool::new(execution_backend)),
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
                    execution_backend,
                    config.terminal.allowed_prefixes.clone(),
                    approval_tx,
                    config.terminal.initial_timeout_secs,
                    config.terminal.max_output_chars,
                    config.terminal.permission_mode,
                    pool,
                )
                .await
                .with_confinement(config.terminal.confinement)
                .with_event_store(event_store)
                .with_stores(
                    state.clone() as Arc<dyn crate::traits::GoalStore>,
                    state as Arc<dyn crate::traits::NotificationStore>,
                )
                .with_delivery_dirs(inbox_dir, outbox_dirs),
            );
            BuiltBaseTool {
                tool: terminal.clone(),
                terminal_tool: Some(terminal),
            }
        }
        BaseToolId::RememberFact => BuiltBaseTool {
            tool: Arc::new(RememberFactTool::new(memory.store.clone())),
            terminal_tool: None,
        },
        BaseToolId::ManageMemories => BuiltBaseTool {
            tool: Arc::new(ManageMemoriesTool::new(state, memory.store.clone())),
            terminal_tool: None,
        },
        BaseToolId::SetReminder => BuiltBaseTool {
            tool: Arc::new(crate::tools::SetReminderTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::ScheduleTask => BuiltBaseTool {
            tool: Arc::new(crate::tools::ScheduleTaskTool::new(state)),
            terminal_tool: None,
        },
        BaseToolId::ManageMandates => BuiltBaseTool {
            tool: Arc::new(ManageMandatesTool::new(state).with_strategy_skills(
                config.skills.enabled.then(|| {
                    crate::startup::tool_config::strategy_skills_dir(
                        config_path
                            .parent()
                            .unwrap_or_else(|| std::path::Path::new("."))
                            .join(&config.skills.dir),
                    )
                }),
            )),
            terminal_tool: None,
        },
        BaseToolId::ManageAutomations => BuiltBaseTool {
            tool: Arc::new(
                crate::tools::ManageAutomationsTool::new(state).with_health_probes(health_store),
            ),
            terminal_tool: None,
        },
        BaseToolId::SearchHistory => BuiltBaseTool {
            tool: {
                let search = SearchHistoryTool::new(state, config.state.retention.messages_days);
                Arc::new(if memory.own_memory {
                    search.with_episodes(memory.store.clone())
                } else {
                    search
                })
            },
            terminal_tool: None,
        },
        BaseToolId::ConfigManager => BuiltBaseTool {
            tool: Arc::new(ConfigManagerTool::new(
                crate::config::config_file(config_path),
                crate::config::shared_secret_store(),
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
            tool: Arc::new(ReadFileTool::on_backend(execution_backend, daemon_paths)),
            terminal_tool: None,
        },
        BaseToolId::WriteFile => BuiltBaseTool {
            tool: Arc::new(WriteFileTool::on_backend(execution_backend, daemon_paths)),
            terminal_tool: None,
        },
        BaseToolId::EditFile => BuiltBaseTool {
            tool: Arc::new(EditFileTool::on_backend(execution_backend, daemon_paths)),
            terminal_tool: None,
        },
        BaseToolId::SearchFiles => BuiltBaseTool {
            tool: Arc::new(SearchFilesTool::on_backend(execution_backend, daemon_paths)),
            terminal_tool: None,
        },
        BaseToolId::CheckEnvironment => BuiltBaseTool {
            tool: Arc::new(CheckEnvironmentTool::new(execution_backend)),
            terminal_tool: None,
        },
    };

    Ok(built)
}

/// The tool catalog as the model can see it, assembled for the schema lint
/// and the per-tool goldens: every startup-registered tool with its feature
/// switched on, plus the tools added outside startup (skills, the
/// checklist, sub-agents and the role-scoped tools a task, mandate run or
/// coordinated child gets).
#[cfg(test)]
pub(crate) mod lint_catalog {
    use super::*;
    use crate::events::EventStoreForTests as _;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use tempfile::{NamedTempFile, TempDir};

    pub(crate) struct Catalog {
        pub tools: Vec<Arc<dyn Tool>>,
        _guards: (TempDir, NamedTempFile, NamedTempFile),
        pub state: Arc<SqliteStateStore>,
        /// Node tools read their instance key from the secret store: never
        /// the developer's real keychain.
        _keychain: Option<crate::testing::KeychainIsolation>,
    }

    /// The tools startup registers (base, optional, runtime), with the
    /// config `configure` leaves.
    pub(crate) async fn build_registered(
        configure: impl FnOnce(&mut AppConfig),
    ) -> anyhow::Result<Catalog> {
        let mut config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test-key"
            "#,
        )?;
        config.provider.models.apply_defaults(&config.provider.kind);
        config.cli_agents.enabled = false;
        config.oauth.enabled = false;
        // Validate the full manifest⇄construction contract: every base tool must
        // be constructible. The default `[tools].disabled` set would otherwise
        // omit several base tools and break the positional alignment with
        // BASE_TOOL_REGISTRY. The disable behavior itself is covered by
        // `disabled_tools_are_omitted_from_base_registry`.
        config.tools.disabled.clear();
        configure(&mut config);

        let io_temp = TempDir::new()?;
        config.files.inbox_dir = io_temp.path().join("inbox").display().to_string();
        config.files.outbox_dirs = vec![io_temp.path().join("outbox").display().to_string()];

        let db_file = NamedTempFile::new()?;
        let db_path = db_file.path().display().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state =
            Arc::new(SqliteStateStore::for_tests(&db_path, 100, None, embedding_service).await?);
        let event_store = Arc::new(EventStore::new(state.pool()).await?);
        let config_file = NamedTempFile::new()?;

        let mut bundle = build_base_tools(
            &config,
            config_file.path().to_path_buf(),
            crate::tools::DaemonPaths::default(),
            crate::execution::active_execution_backend(),
            state.clone(),
            ToolMemoryAccess {
                store: state.memory_stores(),
                own_memory: true,
            },
            state.pool(),
            event_store.clone(),
            32,
            8,
            None,
        )
        .await?;
        let _optional = register_optional_tools(
            &mut bundle.tools,
            &config,
            state.clone(),
            ToolMemoryAccess {
                store: state.memory_stores(),
                own_memory: true,
            },
            state.pool(),
            event_store.clone(),
            None,
            bundle.media_tx.clone(),
            crate::execution::active_execution_backend(),
        )
        .await?;

        let mcp_registry = McpRegistry::new(
            state.clone() as Arc<dyn crate::traits::McpRegistryStore>,
            crate::config::secret_store().clone(),
        );
        let http_profiles: crate::oauth::SharedHttpProfiles =
            Arc::new(tokio::sync::RwLock::new(config.http_auth.clone()));
        let _runtime = register_runtime_tools(
            &mut bundle.tools,
            &config,
            config_file.path(),
            http_profiles,
            state.clone(),
            ToolMemoryAccess {
                store: state.memory_stores(),
                own_memory: true,
            },
            mcp_registry,
        )
        .await?;

        Ok(Catalog {
            tools: bundle.tools,
            _guards: (io_temp, db_file, config_file),
            state,
            _keychain: None,
        })
    }

    /// Every tool the model can be offered, each feature on.
    pub(crate) async fn build_full() -> anyhow::Result<Catalog> {
        let keychain = crate::testing::KeychainIsolation::acquire();
        let mut catalog = build_registered(|config| {
            config.diagnostics.enabled = true;
            config.files.enabled = true;
            config.subagents.enabled = true;
            config.nodes.gateway.enabled = true;
            config.nodes.announcements.enabled = true;
            config.nodes.speech.enabled = true;
            config.nodes.monitoring.enabled = true;
            config.browser.enabled = cfg!(feature = "browser");
            config.computer_use.enabled = cfg!(feature = "computer_use");
        })
        .await?;
        let state = catalog.state.clone();
        let skills_dir = catalog._guards.0.path().join("skills");
        std::fs::create_dir_all(&skills_dir)?;
        let resolver = Arc::new(crate::skills::FileSystemResolver::new());
        let plan_store = Arc::new(crate::plans::PlanStore::new(state.pool()).await?);
        let extra: Vec<Arc<dyn Tool>> = vec![
            Arc::new(crate::tools::UseSkillTool::new(skills_dir.clone())),
            Arc::new(crate::tools::SkillResourcesTool::new(
                skills_dir.clone(),
                resolver as Arc<dyn crate::skills::ResourceResolver>,
            )),
            Arc::new(crate::tools::ManageSkillsTool::new(
                skills_dir,
                state.clone(),
            )),
            Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(plan_store)),
            Arc::new(crate::tools::EscalateToParentTool),
            // Registered by startup only with a checkpoint manager and a
            // health store.
            Arc::new(crate::tools::ListCheckpointsTool),
            Arc::new(crate::tools::HealthProbeTool::new(Arc::new(
                crate::health::HealthProbeStore::new(state.pool()).await?,
            ))),
            Arc::new(
                aidaemon_work::tools::manage_goal_tasks::ManageGoalTasksTool::new(
                    "synthetic-goal".to_string(),
                    state.clone(),
                ),
            ),
            Arc::new(crate::tools::report_blocker::ReportBlockerTool::new(
                "synthetic-task".to_string(),
                state.clone(),
            )),
            // Registered only in a mandate review's task-lead run.
            Arc::new(aidaemon_work::tools::mandate_review::MandateReviewTool::new(state.clone())),
        ];
        catalog.tools.extend(extra);
        catalog._keychain = Some(keychain);
        Ok(catalog)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventStoreForTests as _;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use proptest::prelude::*;
    use serde_json::json;
    use std::collections::HashSet;
    use tempfile::NamedTempFile;

    async fn build_tool_schemas_for_contract_validation(
        diagnostics_enabled: bool,
        files_enabled: bool,
        subagents_enabled: bool,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let catalog = super::lint_catalog::build_registered(|config| {
            config.diagnostics.enabled = diagnostics_enabled;
            config.files.enabled = files_enabled;
            config.subagents.enabled = subagents_enabled;
        })
        .await?;
        Ok(catalog
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
    fn node_audio_tool_requires_explicit_opt_in() {
        let mut config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test-key"
            "#,
        )
        .unwrap();
        assert!(!optional_enabled_node_audio(&config));

        config.nodes.gateway.enabled = true;
        config.nodes.speech.enabled = true;
        assert!(!optional_enabled_node_audio(&config));

        config.nodes.announcements.enabled = true;
        assert!(optional_enabled_node_audio(&config));
    }

    #[test]
    fn node_monitoring_tool_requires_explicit_opt_in() {
        let mut config: AppConfig = toml::from_str(
            r#"
            [provider]
            api_key = "test-key"
            "#,
        )
        .unwrap();
        assert!(!optional_enabled_node_monitoring(&config));

        config.nodes.gateway.enabled = true;
        assert!(!optional_enabled_node_monitoring(&config));

        config.nodes.monitoring.enabled = true;
        assert!(optional_enabled_node_monitoring(&config));
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
                "manage_auth",
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
            id: RuntimeToolId::ManageAuth,
            name: "manage_auth",
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
                id: RuntimeToolId::ManageAuth,
                name: "manage_auth",
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
            let result =
                aidaemon_runtime::engine::tool_roster::validate_tool_definition_contract(&schema);
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
            disabled = ["web_search", "check_environment"]
            "#,
        )
        .unwrap();
        config.provider.models.apply_defaults(&config.provider.kind);

        let db_file = NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(
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
            crate::tools::DaemonPaths::default(),
            crate::execution::active_execution_backend(),
            state.clone(),
            ToolMemoryAccess {
                store: state.memory_stores(),
                own_memory: true,
            },
            state.pool(),
            event_store,
            8,
            8,
            None,
        )
        .await
        .unwrap();

        let names: Vec<&str> = bundle.tools.iter().map(|t| t.name()).collect();
        assert!(!names.contains(&"web_search"));
        assert!(!names.contains(&"check_environment"));
        assert!(names.contains(&"terminal"));
    }

    #[tokio::test]
    async fn forensics_tools_register_only_with_diagnostics() {
        for diagnostics in [false, true] {
            let schemas = build_tool_schemas_for_contract_validation(diagnostics, false, false)
                .await
                .unwrap();
            let names: Vec<&str> = schemas
                .iter()
                .filter_map(|schema| schema["function"]["name"].as_str())
                .collect();
            for tool in ["goal_trace", "policy_metrics", "self_diagnose"] {
                assert_eq!(names.contains(&tool), diagnostics, "{tool}");
            }
        }
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
        assert!(names.contains(&"manage_auth".to_string()));
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
                    let result = aidaemon_runtime::engine::tool_roster::validate_tool_definition_contract(&schema);
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
