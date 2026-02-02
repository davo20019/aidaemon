use std::sync::Arc;
use std::time::Duration;

use tracing::info;

use crate::agent::Agent;
use crate::channels::TelegramChannel;
use crate::config::AppConfig;
use crate::daemon;
use crate::mcp;

use crate::memory::embeddings::EmbeddingService;
use crate::memory::manager::MemoryManager;

use crate::router::{Router, Tier};
use crate::skills;
use crate::state::SqliteStateStore;
use crate::tools::{BrowserTool, CliAgentTool, ConfigManagerTool, RememberFactTool, SpawnAgentTool, SystemInfoTool, TerminalTool};
use crate::traits::Tool;
use crate::triggers::{self, TriggerManager};

pub async fn run(config: AppConfig, config_path: std::path::PathBuf) -> anyhow::Result<()> {
    // 0. Embeddings (Vector Memory)
    let embedding_service = Arc::new(
        EmbeddingService::new().map_err(|e| anyhow::anyhow!("Failed to init embeddings: {}", e))?
    );
    info!("Embedding service initialized (AllMiniLML6V2)");

    // 1. State store
    let state = Arc::new(
        SqliteStateStore::new(
            &config.state.db_path,
            config.state.working_memory_cap,
            embedding_service.clone()
        ).await?,
    );
    info!("State store initialized ({})", config.state.db_path);

    // 2. Provider (moved before MemoryManager so provider is available)
    let provider: Arc<dyn crate::traits::ModelProvider> = match config.provider.kind {
        crate::config::ProviderKind::OpenaiCompatible => Arc::new(crate::providers::OpenAiCompatibleProvider::new(
            &config.provider.base_url,
            &config.provider.api_key,
        )),
        crate::config::ProviderKind::GoogleGenai => Arc::new(crate::providers::GoogleGenAiProvider::new(
            &config.provider.api_key,
        )),
        crate::config::ProviderKind::Anthropic => Arc::new(crate::providers::AnthropicNativeProvider::new(
            &config.provider.api_key,
        )),
    };

    // 3. Router
    let router = Router::new(config.provider.models.clone());
    let model = router.select(Tier::Primary).to_string();
    info!(
        primary = router.select(Tier::Primary),
        fast = router.select(Tier::Fast),
        smart = router.select(Tier::Smart),
        "Model router configured"
    );

    // 3b. Memory Manager (Background Tasks — needs provider + fast model)
    let consolidation_interval = Duration::from_secs(config.state.consolidation_interval_hours * 3600);
    let memory_manager = Arc::new(MemoryManager::new(
        state.pool(),
        embedding_service.clone(),
        provider.clone(),
        router.select(Tier::Fast).to_string(),
        consolidation_interval,
    ));
    memory_manager.start_background_tasks();

    // 4. Tools
    let (approval_tx, approval_rx) = tokio::sync::mpsc::channel(16);
    let (media_tx, media_rx) = tokio::sync::mpsc::channel(16);
    let mut tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(SystemInfoTool),
        Arc::new(TerminalTool::new(config.terminal.allowed_prefixes.clone(), approval_tx)),
        Arc::new(RememberFactTool::new(state.clone())),
        Arc::new(ConfigManagerTool::new(config_path.clone())),
    ];

    // Browser tool (conditional)
    if config.browser.enabled {
        tools.push(Arc::new(BrowserTool::new(config.browser.clone(), media_tx.clone())));
        info!("Browser tool enabled");
    }

    // CLI agent tools (conditional)
    if config.cli_agents.enabled {
        let cli_tool = CliAgentTool::discover(config.cli_agents.clone()).await;
        if cli_tool.has_tools() {
            tools.push(Arc::new(cli_tool));
            info!("CLI agent tool enabled");
        } else {
            info!("CLI agents enabled but no tools found on system");
        }
    }

    // 5. MCP tools
    if !config.mcp.is_empty() {
        let mcp_tools = mcp::discover_mcp_tools(&config.mcp).await?;
        tools.extend(mcp_tools);
    }

    for tool in &tools {
        info!(name = tool.name(), desc = tool.description(), "Registered tool");
    }

    // 6. Skills
    let loaded_skills = if config.skills.enabled {
        let skills_dir = config_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(&config.skills.dir);
        let s = skills::load_skills(&skills_dir);
        info!(count = s.len(), dir = %skills_dir.display(), "Skills loaded");
        s
    } else {
        info!("Skills system disabled");
        Vec::new()
    };

    // 7. Agent (with deferred spawn tool wiring to break the circular dep)
    let base_system_prompt = build_base_system_prompt(&config);

    // Spawn-agent tool (conditional, like browser)
    let spawn_tool: Option<Arc<SpawnAgentTool>> = if config.subagents.enabled {
        let st = Arc::new(SpawnAgentTool::new_deferred(
            config.subagents.max_response_chars,
            config.subagents.timeout_secs,
        ));
        tools.push(st.clone());
        info!("Spawn-agent tool enabled");
        Some(st)
    } else {
        info!("Spawn-agent tool disabled");
        None
    };

    let agent = Arc::new(Agent::new(
        provider,
        state.clone(),
        tools,
        model,
        base_system_prompt,
        config_path.clone(),
        loaded_skills,
        config.subagents.max_depth,
        config.subagents.max_iterations,
        config.subagents.max_response_chars,
        config.subagents.timeout_secs,
        config.provider.models.clone(),
    ));

    // Close the loop: give the spawn tool a weak reference to the agent.
    if let Some(st) = spawn_tool {
        st.set_agent(Arc::downgrade(&agent));
    }

    // 8. Event bus for triggers
    let (event_tx, mut event_rx) = triggers::event_bus(64);

    // 9. Triggers
    let trigger_manager = Arc::new(TriggerManager::new(config.triggers.clone(), event_tx));
    trigger_manager.spawn();

    // 10. Telegram channel
    let telegram = Arc::new(TelegramChannel::new(
        &config.telegram.bot_token,
        config.telegram.allowed_user_ids.clone(),
        Arc::clone(&agent),
        config_path,
        approval_rx,
        media_rx,
    ));

    // 11. Health server
    let health_port = config.daemon.health_port;
    tokio::spawn(async move {
        if let Err(e) = daemon::start_health_server(health_port).await {
            tracing::error!("Health server error: {}", e);
        }
    });

    // 12. Event listener: route trigger events to agent -> Telegram notification
    let telegram_for_events = Arc::clone(&telegram);
    let agent_for_events = Arc::clone(&agent);
    let notify_chat_ids = config.telegram.allowed_user_ids.clone();
    tokio::spawn(async move {
        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    info!(source = %event.source, "Received trigger event");
                    match agent_for_events
                        .handle_message(&event.session_id, &event.content)
                        .await
                    {
                        Ok(reply) => {
                            for &uid in &notify_chat_ids {
                                let _ = telegram_for_events
                                    .send_to_chat(uid as i64, &reply)
                                    .await;
                            }
                        }
                        Err(e) => {
                            tracing::error!("Agent error handling trigger event: {}", e);
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("Event listener lagged by {} events", n);
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    break;
                }
            }
        }
    });

    // 13. Start Telegram with auto-retry (blocks)
    info!("Starting aidaemon v0.1.0");
    telegram.start_with_retry().await;

    Ok(())
}

fn build_base_system_prompt(config: &AppConfig) -> String {
    let spawn_table_row = if config.subagents.enabled {
        "\n| Complex sub-tasks needing focused reasoning | spawn_agent | — |"
    } else {
        ""
    };

    let cli_agent_table_row = if config.cli_agents.enabled {
        "\n| Coding tasks, refactoring, debugging | cli_agent | terminal (running AI tools manually) |"
    } else {
        ""
    };

    let spawn_tool_doc = if config.subagents.enabled {
        format!(
            "\n- `spawn_agent`: Spawn a sub-agent to handle a complex sub-task autonomously. \
            Parameters: `mission` (high-level role, e.g. 'Research assistant for Python packaging') \
            and `task` (the specific question or job). The sub-agent gets its own reasoning loop \
            with access to all tools. Use this when a task benefits from isolated, focused context. \
            Sub-agents can nest up to {} levels deep.",
            config.subagents.max_depth
        )
    } else {
        String::new()
    };

    let cli_agent_tool_doc = if config.cli_agents.enabled {
        "\n- `cli_agent`: Delegate tasks to CLI-based AI coding agents (e.g. Claude Code, Gemini CLI, Codex). \
        These are full AI agents that can read/write files, run commands, and solve complex coding tasks. \
        Parameters: `tool` (which agent to use), `prompt` (the task), `working_dir` (optional project directory). \
        Prefer this over running AI CLI tools via terminal — it handles timeouts, output parsing, and \
        non-interactive mode automatically."
    } else {
        ""
    };

    format!(
        "\
## Identity
You are aidaemon, a personal AI assistant running as a background daemon.
You have access to tools and should use them proactively to answer questions.

## Planning
Before using any tool, STOP and think:
1. What is the user asking for?
2. Which tool is the RIGHT one? (See Tool Selection Guide below)
3. What is the sequence of steps?
4. What information do I need first?

Narrate your plan before executing — tell the user what you're about to do and why.

## Tool Selection Guide
| Task | Correct Tool | WRONG Tool |
|------|-------------|------------|
| Visit website, search web | browser | terminal (curl/wget) |
| Run commands, scripts | terminal | — |
| Get system specs | system_info | terminal (uname, etc.) |
| Store user info | remember_fact | — |
| Fix config | manage_config | terminal (editing files) |{spawn_table_row}{cli_agent_table_row}

## Tools
- `terminal`: Run ANY command available on this system. This includes shell commands, \
CLI tools (python, node, cargo, docker, git, claude, gemini, etc.), package managers, \
scripts, and anything else installed on the machine. You have full access to the system \
through this tool. If a command is not pre-approved, the user will be asked to approve it \
via an inline button — so don't hesitate to try commands even if they're not in the \
pre-approved list. The user can allow them with one tap.
- `system_info`: Get CPU, memory, and OS information.
- `remember_fact`: Store important facts about the user for long-term memory.
- `manage_config`: Read and update your own config.toml. Use this to fix configuration issues.
- `browser`: Control a headless browser for web interactions. Actions: navigate (go to URL), \
screenshot (capture page and send as photo), click (click element by CSS selector), \
fill (type text into input), get_text (extract visible text), execute_js (run JavaScript), \
wait (wait for element to appear), close (end browser session). The browser session persists \
across tool calls so you can chain multi-step workflows (e.g. navigate -> fill form -> click -> screenshot).{spawn_tool_doc}{cli_agent_tool_doc}

## Self-Maintenance
You are responsible for your own maintenance. When you encounter errors:
1. Diagnose the issue using your tools (read logs, check config, test commands).
2. Fix it yourself using `manage_config` to update settings, or `terminal` to run commands.
3. Tell the user to run /reload if you changed the config, so changes take effect.
4. If a model name is wrong, use `manage_config` to read the config, then fix the model name.

## Behavior
- Use tools proactively. Do NOT say you can't do something — try it with `terminal` first.
- Never refuse to run a command because you think you don't have access. The approval system \
handles permissions — just call the tool and let the user decide.
- When you learn important facts about the user, store them with `remember_fact`.
- Narrate your plan before executing — tell the user what you're about to do and why.
- Be concise and helpful."
    )
}
