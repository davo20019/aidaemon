use std::sync::Arc;

use tracing::info;

use crate::agent::Agent;
use crate::channels::TelegramChannel;
use crate::config::AppConfig;
use crate::daemon;
use crate::mcp;
use crate::providers::OpenAiCompatibleProvider;
use crate::router::{Router, Tier};
use crate::skills;
use crate::state::SqliteStateStore;
use crate::tools::{BrowserTool, ConfigManagerTool, RememberFactTool, SystemInfoTool, TerminalTool};
use crate::traits::Tool;
use crate::triggers::{self, TriggerManager};

pub async fn run(config: AppConfig, config_path: std::path::PathBuf) -> anyhow::Result<()> {
    // 1. State store
    let state = Arc::new(
        SqliteStateStore::new(&config.state.db_path, config.state.working_memory_cap).await?,
    );
    info!("State store initialized ({})", config.state.db_path);

    // 2. Provider
    let provider = Arc::new(OpenAiCompatibleProvider::new(
        &config.provider.base_url,
        &config.provider.api_key,
    ));

    // 3. Router
    let router = Router::new(config.provider.models.clone());
    let model = router.select(Tier::Primary).to_string();
    info!(
        primary = router.select(Tier::Primary),
        fast = router.select(Tier::Fast),
        smart = router.select(Tier::Smart),
        "Model router configured"
    );

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

    // 7. Agent
    let base_system_prompt = build_base_system_prompt();

    let agent = Arc::new(Agent::new(
        provider,
        state.clone(),
        tools,
        model,
        base_system_prompt,
        config_path.clone(),
        loaded_skills,
    ));

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

fn build_base_system_prompt() -> String {
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
| Fix config | manage_config | terminal (editing files) |

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
across tool calls so you can chain multi-step workflows (e.g. navigate -> fill form -> click -> screenshot).

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
        .to_string()
}
