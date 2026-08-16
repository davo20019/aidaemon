use std::sync::Arc;

use tracing::info;

use crate::agent::Agent;
use crate::channels::{ChannelHub, SessionMap};
use crate::config::{AppConfig, AudioConfig, SttConfig, VisionConfig};
use crate::llm_runtime::SharedLlmRuntime;
use crate::queue_telemetry::QueueTelemetry;
use crate::skills;
use crate::startup::{
    channels as startup_channels, memory_pipeline, runtime as startup_runtime, stores,
    tools as startup_tools,
};
use crate::state::SqliteStateStore;
use crate::tasks::TaskRegistry;
use crate::traits::store_prelude::*;
use crate::traits::Goal;
use crate::triggers::{self, TriggerManager};

const LEGACY_KNOWLEDGE_MAINTENANCE_GOAL_DESC: &str =
    "Maintain knowledge base: process embeddings, consolidate memories, decay old facts";
const LEGACY_MEMORY_HEALTH_GOAL_DESC: &str =
    "Maintain memory health: prune old events, clean up retention, remove stale data";
const LEGACY_SYSTEM_SESSION_ID: &str = "system";
const LEGACY_MAINTENANCE_MIGRATION_DONE_KEY: &str =
    "migration_legacy_system_maintenance_goals_retired_v1";

fn is_truthy_setting(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enabled"
    )
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct LegacyMaintenanceMigrationStats {
    goals_matched: usize,
    goals_retired: usize,
    tasks_closed: usize,
    notifications_deleted: usize,
}

pub async fn run(config: AppConfig, config_path: std::path::PathBuf) -> anyhow::Result<()> {
    let mut config = config;

    // Single-instance guard: refuse to start if another daemon already holds the
    // lock for this database. Two instances on one DB silently race over goals,
    // tasks, and the terminal bridge — goals die as "interrupted" and never
    // complete. Acquire before touching the DB so a duplicate bails immediately.
    let db_path = crate::startup::db_security::resolve_db_path(&config_path, &config.state.db_path);
    let lock_path = std::path::PathBuf::from(format!("{}.lock", db_path.display()));
    if let Err(e) = crate::single_instance::acquire(&lock_path) {
        tracing::error!("{e}");
        return Err(e);
    }

    crate::startup::db_security::enforce_database_encryption(&mut config, &config_path).await?;

    let write_consistency_thresholds = config.policy.write_consistency.thresholds();
    let queue_policy = config.daemon.queue_policy.normalized();

    let queue_telemetry = Arc::new(QueueTelemetry::new_with_policy(
        queue_policy.approval_capacity,
        queue_policy.media_capacity,
        queue_policy.trigger_event_capacity,
        queue_policy.warning_ratio,
        queue_policy.overload_ratio,
    ));

    let stores::StoreBundle {
        embedding_service,
        state,
        event_store,
        plan_store,
        health_store,
    } = stores::build_stores(&config).await?;

    let crate::providers::factory::ProviderRouterBundle {
        provider,
        primary_model: model,
        router,
        provider_kind,
        failover_targets,
    } = crate::providers::factory::build_provider_router(&config)?;
    let llm_runtime = SharedLlmRuntime::new_with_failovers(
        provider.clone(),
        router,
        provider_kind,
        model.clone(),
        failover_targets,
    );

    let memory_pipeline::MemoryPipelineBundle {
        consolidator: _consolidator,
        pruner,
        memory_manager,
    } = memory_pipeline::build_memory_pipeline(
        &config,
        state.clone(),
        event_store.clone(),
        plan_store.clone(),
        llm_runtime.clone(),
        embedding_service.clone(),
    );

    let startup_tools::ToolSetup {
        tools,
        execution_backend: _execution_backend,
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
    } = startup_tools::setup_tools(
        &config,
        &config_path,
        state.clone(),
        event_store.clone(),
        llm_runtime.clone(),
        health_store.clone(),
        queue_policy.approval_capacity,
        queue_policy.media_capacity,
    )
    .await?;

    // Requirement-checklist tool: model self-registers the durable per-turn
    // checklist (backed by plan_store). It only persists checklist state; the
    // rendered checklist is surfaced to the user by the agent loop via
    // StatusUpdate::Checklist, so it no longer needs ChannelHub wiring.
    let mut tools = tools;
    let track_requirements_tool =
        Arc::new(crate::tools::track_requirements::TrackRequirementsTool::new(plan_store.clone()));
    tools.push(track_requirements_tool.clone());

    // 7. Agent (with deferred spawn tool wiring to break the circular dep)
    let skill_names: Vec<String> = if let Some(ref dir) = skills_dir {
        skills::load_skills(dir)
            .iter()
            .map(|s| s.name.clone())
            .collect()
    } else {
        Vec::new()
    };
    let custom_persona = load_agent_persona(&config, &config_path)?;
    let base_system_prompt =
        build_base_system_prompt(&config, &skill_names, custom_persona.as_deref())?;

    let llm_call_timeout_secs = if config.daemon.watchdog.enabled {
        Some(config.daemon.watchdog.llm_call_timeout_secs)
    } else {
        None
    };
    let watchdog_stale_threshold_secs = if config.daemon.watchdog.enabled {
        config.daemon.watchdog.stale_threshold_secs
    } else {
        0
    };

    // Goal token registry for cancellation hierarchy
    let goal_token_registry = crate::goal_tokens::GoalTokenRegistry::new();

    // Specialist registry: bundled `.md` files + optional user overrides.
    // Loaded once at startup and shared across the entire agent hierarchy.
    let specialists_dir = config
        .subagents
        .specialists_override_dir
        .clone()
        .or_else(|| dirs::home_dir().map(|h| h.join(".aidaemon").join("specialists")));
    let specialists = std::sync::Arc::new(crate::agent::specialists::SpecialistRegistry::load(
        specialists_dir.as_deref(),
    ));

    let agent = Arc::new(Agent::new(crate::agent::AgentConstruction {
        dependencies: crate::agent::AgentRuntimeDependencies {
            llm_runtime: llm_runtime.clone(),
            state: state.clone(),
            event_store: event_store.clone(),
            tools,
        },
        model,
        system_prompt: base_system_prompt,
        config_path: config_path.clone(),
        skills_dir: skills_dir.clone().unwrap_or_default(),
        max_depth: config.subagents.max_depth,
        max_iterations: config.subagents.max_iterations,
        max_iterations_cap: config.subagents.max_iterations_cap,
        max_response_chars: config.subagents.max_response_chars,
        timeout_secs: config.subagents.timeout_secs,
        max_facts: config.state.max_facts,
        daily_token_budget: config.state.daily_token_budget,
        iteration_config: config.subagents.effective_iteration_limit(),
        task_timeout_secs: config.subagents.task_timeout_secs,
        task_token_budget: config.subagents.task_token_budget,
        llm_call_timeout_secs,
        mcp_registry: Some(mcp_registry.clone()),
        goal_token_registry: Some(goal_token_registry.clone()),
        hub: None,
        record_decision_points: config.diagnostics.record_decision_points,
        context_window_config: config.state.context_window.clone(),
        policy_config: config.policy.clone(),
        path_aliases: config.path_aliases.clone(),
        inherited_project_scope: None,
        specialists,
        // Pin the interactive generation loop to the configured slot only when
        // slot routing is enabled on the primary provider; otherwise None (no
        // id_slot is ever sent — zero behavior change for cloud-API users).
        interactive_slot: if config.provider.slot_routing.enabled {
            Some(config.provider.slot_routing.interactive_slot)
        } else {
            None
        },
        vision_config: VisionConfig::from_files(&config.files),
        audio_config: AudioConfig::from_files(&config.files),
        stt_config: SttConfig::from_files(&config.files),
        harness_eval_config: (&config.diagnostics.harness_eval).into(),
    }));

    // Close the deferred Agent ↔ SpawnAgentTool + agent self-reference cycles.
    crate::startup::wiring::wire_agent_cycles(&agent, spawn_tool.as_ref()).await?;

    // Merge persisted runtime-learned "ignores tool_choice=required" models into
    // the config-seeded in-memory set, so a model that melted down once stays
    // flagged across restarts (the 264s gemma meltdown never re-arms).
    agent.load_required_tool_choice_ignored().await;

    // 8. Event bus for triggers
    let (event_tx, event_rx) = triggers::event_bus(queue_policy.trigger_event_capacity);
    // 9. Triggers
    let trigger_manager = Arc::new(TriggerManager::new(config.triggers.clone(), event_tx));
    trigger_manager.spawn();

    // 9b. Scheduler deprecated — evergreen goals replace it.

    // Migrate legacy seeded maintenance goals to deterministic background jobs.
    // Run before heartbeat starts so no legacy goal tasks are dispatched this boot.
    maybe_run_legacy_system_maintenance_goal_migration(state.clone()).await;

    // 9c. Heartbeat coordinator (replaces individual background task loops)
    let (wake_tx, wake_rx) = tokio::sync::mpsc::channel::<()>(16);
    let startup_runtime::HeartbeatSetup {
        coordinator: heartbeat_opt,
        telemetry: heartbeat_telemetry,
    } = startup_runtime::init_heartbeat_coordinator(
        &config,
        state.clone(),
        event_store.clone(),
        pruner.clone(),
        memory_manager.clone(),
        wake_rx,
        inbox_dir.clone(),
        skills_dir.clone(),
        llm_runtime.clone(),
        oauth_gateway.clone(),
        watchdog_stale_threshold_secs,
        goal_token_registry.clone(),
        terminal_tool.clone(),
    )
    .await;

    // 10. Session map (shared between hub and channels for routing)
    // Reload persisted session→channel mappings so scheduled goals can
    // deliver notifications after a restart.
    let session_map: SessionMap = startup_runtime::restore_session_map(
        state.clone() as Arc<dyn crate::traits::SessionChannelStore>
    )
    .await;

    // 10b. Task registry for tracking background agent work
    let task_registry = Arc::new(TaskRegistry::new(50));

    // 11. Channels
    let channel_agent: Arc<dyn crate::runtime_ports::ChannelAgentRuntime> = agent.clone();
    let channel_bundle = startup_channels::build_channels(
        &config,
        crate::channels::ChannelRuntimeDeps {
            agent: channel_agent,
            config_path: config_path.clone(),
            session_map: session_map.clone(),
            task_registry: task_registry.clone(),
            files_enabled: config.files.enabled,
            inbox_dir: std::path::PathBuf::from(&inbox_dir),
            max_file_size_mb: config.files.max_file_size_mb,
            state: state.clone() as Arc<dyn crate::traits::StateStore>,
            watchdog_stale_threshold_secs,
        },
    )
    .await;

    // 12. Channel Hub — routes approvals, media, and notifications
    let hub = Arc::new(
        ChannelHub::new(channel_bundle.channels.clone(), session_map)
            .with_queue_telemetry(queue_telemetry.clone())
            .with_delivery_note_sink(agent.clone())
            .with_queue_policy(queue_policy.clone()),
    );

    // Close the deferred tools/agent ↔ ChannelHub cycles now that the hub exists.
    crate::startup::wiring::wire_hub_cycles(
        &agent,
        &hub,
        spawn_tool.as_ref(),
        terminal_tool.as_ref(),
        cli_agent_tool.as_ref(),
        plan_store.clone(),
    )
    .await?;
    crate::startup::nodes::start(&config, state.pool(), agent.clone(), hub.clone()).await?;
    // Give the agent its plan_store handle so the completion phase can read the
    // active checklist for soft verification + recap (deferred to avoid touching
    // the large Agent::new signature and all subagent spawn call sites).
    agent.set_plan_store(plan_store.clone()).await;

    // Start the heartbeat coordinator now that hub and agent are available.
    startup_runtime::start_heartbeat_coordinator(heartbeat_opt, &hub, &agent);

    // Give all channels a reference to the hub for dynamic bot registration
    let weak_hub = Arc::downgrade(&hub);
    channel_bundle.set_channel_hub_for_all(weak_hub);

    // Start approval listener (routes tool approval requests to the right channel)
    let hub_for_approvals = hub.clone();
    tokio::spawn(async move {
        hub_for_approvals.approval_listener(approval_rx).await;
    });

    // Start media listener (routes screenshots/photos/files to the right channel)
    let hub_for_media = hub.clone();
    tokio::spawn(async move {
        hub_for_media.media_listener(media_rx).await;
    });

    // Inbox cleanup is now registered with the heartbeat coordinator below.

    let default_alert_sessions = startup_runtime::persist_default_alert_sessions(
        &config,
        state.clone() as Arc<dyn crate::traits::SettingsStore>,
    )
    .await;

    // 12b. Health Probe Manager (uses health_store created earlier in 1d)
    startup_runtime::init_health_probe_manager(
        &config,
        &health_store,
        hub.clone(),
        &default_alert_sessions,
    )
    .await;

    // 13. Health / Dashboard server
    startup_runtime::spawn_dashboard_or_health_server(
        &config,
        state.clone(),
        event_store.clone(),
        health_store.clone(),
        heartbeat_telemetry.clone(),
        oauth_gateway.clone(),
        write_consistency_thresholds,
        queue_telemetry.clone(),
        wake_tx.clone(),
    );

    // 14. Event listener: route trigger events to agent -> broadcast via hub
    let notify_session_ids = startup_runtime::collect_notify_session_ids(&config);
    startup_runtime::spawn_trigger_event_listener(
        event_rx,
        hub.clone(),
        agent.clone(),
        state.clone() as Arc<dyn crate::traits::MandateStore>,
        wake_tx,
        notify_session_ids.clone(),
        queue_telemetry.clone(),
        queue_policy.clone(),
    );

    // 14b. Self-updater
    if config.updates.mode != crate::config::UpdateMode::Disable {
        let updater = Arc::new(crate::updater::Updater::new(
            config.updates.clone(),
            hub.clone(),
            notify_session_ids.clone(),
            approval_tx.clone(),
        ));
        updater.spawn();
        info!(mode = ?config.updates.mode, "Self-updater initialized");
    }

    // 15. Send startup notification to first Telegram bot's allowed users.
    channel_bundle.send_startup_notifications(&config).await;

    // 16. Start channels
    info!("Starting aidaemon v{}", env!("CARGO_PKG_VERSION"));
    channel_bundle.spawn_all();
    #[cfg(feature = "terminal-bridge")]
    crate::terminal_bridge::spawn_if_configured(&config, state.clone());

    // Wait for shutdown signal (ctrl+c), then gracefully pause plans
    info!("All subsystems started, waiting for shutdown signal (ctrl+c)");
    tokio::signal::ctrl_c().await.ok();
    info!("Shutdown signal received");

    // Shut down all MCP server processes
    info!("Shutting down MCP servers...");
    mcp_registry.shutdown_all().await;

    Ok(())
}

/// Run all startup database migrations and exit.
///
/// Useful for post-install/post-upgrade automation:
/// `aidaemon migrate` can be run non-interactively before starting the daemon.
pub async fn run_migrations_only(
    config: AppConfig,
    config_path: std::path::PathBuf,
) -> anyhow::Result<()> {
    let mut config = config;
    crate::startup::db_security::enforce_database_encryption(&mut config, &config_path).await?;

    let stores::StoreBundle { state, .. } = stores::build_stores(&config).await?;
    maybe_run_legacy_system_maintenance_goal_migration(state).await;

    Ok(())
}

fn is_legacy_system_maintenance_goal(goal: &Goal) -> bool {
    if goal.session_id != LEGACY_SYSTEM_SESSION_ID {
        return false;
    }

    if let Some(ctx) = goal.context.as_deref() {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(ctx) {
            if let Some(system_goal) = value.get("system_goal").and_then(|v| v.as_str()) {
                return matches!(system_goal, "knowledge_maintenance" | "memory_health");
            }
        }
    }

    goal.description == LEGACY_KNOWLEDGE_MAINTENANCE_GOAL_DESC
        || goal.description == LEGACY_MEMORY_HEALTH_GOAL_DESC
}

fn is_open_goal_task_status(status: &str) -> bool {
    matches!(status, "pending" | "claimed" | "running")
}

async fn maybe_run_legacy_system_maintenance_goal_migration(state: Arc<SqliteStateStore>) {
    let migration_done = match state
        .get_setting(LEGACY_MAINTENANCE_MIGRATION_DONE_KEY)
        .await
    {
        Ok(Some(v)) => is_truthy_setting(&v),
        Ok(None) => false,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "Failed to read legacy maintenance-goal migration marker; running migration"
            );
            false
        }
    };
    if !migration_done {
        match retire_legacy_system_maintenance_goals(state.clone()).await {
            Ok(stats) => {
                if stats.goals_matched > 0
                    || stats.goals_retired > 0
                    || stats.tasks_closed > 0
                    || stats.notifications_deleted > 0
                {
                    info!(
                        matched = stats.goals_matched,
                        retired = stats.goals_retired,
                        tasks_closed = stats.tasks_closed,
                        notifications_deleted = stats.notifications_deleted,
                        "Applied legacy maintenance-goal migration"
                    );
                }
                if let Err(e) = state
                    .set_setting(LEGACY_MAINTENANCE_MIGRATION_DONE_KEY, "1")
                    .await
                {
                    tracing::warn!(
                        error = %e,
                        "Failed to persist legacy maintenance-goal migration marker"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Legacy maintenance-goal migration failed");
            }
        }
    }
}

async fn retire_legacy_system_maintenance_goals(
    state: Arc<SqliteStateStore>,
) -> anyhow::Result<LegacyMaintenanceMigrationStats> {
    let mut stats = LegacyMaintenanceMigrationStats::default();
    let scheduled_goals = state.get_scheduled_goals().await?;
    let legacy_goals: Vec<Goal> = scheduled_goals
        .into_iter()
        .filter(is_legacy_system_maintenance_goal)
        .collect();
    stats.goals_matched = legacy_goals.len();

    if legacy_goals.is_empty() {
        return Ok(stats);
    }

    let now = chrono::Utc::now().to_rfc3339();
    let retirement_note = "Retired by startup migration: legacy system maintenance goal removed";

    for goal in legacy_goals {
        if goal.status != "cancelled" && goal.status != "completed" {
            let mut updated_goal = goal.clone();
            updated_goal.status = "cancelled".to_string();
            updated_goal.completed_at = Some(now.clone());
            updated_goal.updated_at = now.clone();
            state.update_goal(&updated_goal).await?;
            stats.goals_retired += 1;
        }

        let tasks = state.get_tasks_for_goal(&goal.id).await?;
        for mut task in tasks {
            if !is_open_goal_task_status(&task.status) {
                continue;
            }
            task.status = "completed".to_string();
            task.completed_at = Some(now.clone());
            task.error = None;
            let has_result = task
                .result
                .as_ref()
                .is_some_and(|result| !result.trim().is_empty());
            if !has_result {
                task.result = Some(retirement_note.to_string());
            }
            state.update_task(&task).await?;
            stats.tasks_closed += 1;
        }

        let deleted = sqlx::query("DELETE FROM notification_queue WHERE goal_id = ?")
            .bind(&goal.id)
            .execute(&state.pool())
            .await?;
        stats.notifications_deleted += deleted.rows_affected() as usize;
    }

    Ok(stats)
}

const MAX_PERSONA_FILE_BYTES: usize = 32 * 1024;

fn load_agent_persona(
    config: &AppConfig,
    config_path: &std::path::Path,
) -> anyhow::Result<Option<String>> {
    config.agent.validated_name()?;
    let Some(configured_path) = config.agent.persona_file.as_ref() else {
        return Ok(None);
    };
    if configured_path.as_os_str().is_empty() {
        anyhow::bail!("agent.persona_file cannot be empty");
    }

    let path = if configured_path.is_absolute() {
        configured_path.clone()
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(configured_path)
    };
    let bytes = std::fs::read(&path).map_err(|e| {
        anyhow::anyhow!(
            "failed to read agent persona file {}: {}",
            path.display(),
            e
        )
    })?;
    if bytes.len() > MAX_PERSONA_FILE_BYTES {
        anyhow::bail!(
            "agent persona file {} is too large ({} bytes; maximum {})",
            path.display(),
            bytes.len(),
            MAX_PERSONA_FILE_BYTES
        );
    }
    let persona = String::from_utf8(bytes).map_err(|e| {
        anyhow::anyhow!(
            "agent persona file {} is not valid UTF-8: {}",
            path.display(),
            e
        )
    })?;
    let persona = persona.trim().trim_start_matches('\u{feff}').trim();
    if persona.is_empty() {
        anyhow::bail!("agent persona file {} is empty", path.display());
    }
    Ok(Some(persona.to_string()))
}

fn build_base_system_prompt(
    config: &AppConfig,
    skill_names: &[String],
    custom_persona: Option<&str>,
) -> anyhow::Result<String> {
    let agent_name = config.agent.validated_name()?;
    let custom_persona_section = custom_persona
        .map(|persona| {
            format!(
                "\n\n## Owner-Configured Persona\n\
                 The owner configured the following role, voice, and working preferences. Follow them \
                 when they do not conflict with the Core Rules, channel/privacy/security rules, tool \
                 policies, or factual and completion honesty.\n\n\
                 <owner_persona>\n{}\n</owner_persona>",
                persona.trim()
            )
        })
        .unwrap_or_default();
    let spawn_table_row = if config.subagents.enabled {
        "\n| Complex sub-tasks needing focused reasoning | spawn_agent | — |"
    } else {
        ""
    };

    let cli_agent_table_row = if config.cli_agents.enabled {
        "\n| Complex multi-step tasks (research, coding, analysis, admin) | cli_agent (REQUIRED when available at runtime) | terminal/run_command for simple or fallback work |"
    } else {
        ""
    };

    let manage_cli_agents_table_row = if config.cli_agents.enabled {
        "\n| List installed CLI AI agents, or add/enable/disable them (Claude Code, Gemini, etc.) | manage_cli_agents | — |"
    } else {
        ""
    };

    let send_file_table_row = if config.files.enabled {
        "\n| Send a file to the user | send_file | terminal (manual upload) |"
    } else {
        ""
    };

    let health_probe_table_row = if config.health.enabled {
        "\n| Monitor services, endpoints, health checks | health_probe | terminal (curl, ping) |"
    } else {
        ""
    };

    let manage_skills_table_row = if config.skills.enabled {
        "\n| Add, update, or generate reusable skills/API guides | manage_skills | — |"
    } else {
        ""
    };

    let use_skill_table_row = if config.skills.enabled {
        "\n| Activate a saved skill/procedure | use_skill | — |"
    } else {
        ""
    };

    let skill_resources_table_row = if config.skills.enabled {
        "\n| Load resources (scripts, references) from a skill | skill_resources | — |"
    } else {
        ""
    };

    let manage_people_table_row =
        "\n| Track contacts, relationships, birthdays | manage_people | — |";

    let http_request_table_row =
        "\n| Make authenticated API requests (Twitter, Stripe, etc.) | http_request | terminal (curl) |";

    let manage_api_table_row =
        "\n| Deterministically connect, learn, and verify an API end-to-end | manage_api | manual multi-tool orchestration |";

    let manage_http_auth_table_row =
        "\n| Create and verify generic API auth profiles | manage_http_auth | manual config edits + keychain commands |";

    let manage_oauth_table_row =
        "\n| Connect external services via OAuth (built-in or custom OAuth2) | manage_oauth | — |";

    let browser_table_row = if cfg!(feature = "browser") && config.browser.enabled {
        "| Interact with login/JavaScript website | browser | web_fetch for readable public pages |\n"
    } else {
        ""
    };

    let computer_use_table_row = if cfg!(feature = "computer_use") && config.computer_use.enabled {
        "| Control native macOS apps (inspect windows, click, type) | computer_use | — |\n\
         | Click a button in a desktop dialog or system UI | computer_use | — |\n"
    } else {
        ""
    };

    let computer_use_guidance = if cfg!(feature = "computer_use") && config.computer_use.enabled {
        "\n\n## Desktop Computer Use\n\
        Use computer_use only for native macOS apps; use browser for websites and \
        localhost dev servers. Always call get_app_state first and pass its \
        snapshot_generation to every mutating action. Prefer element_index over raw \
        coordinates when the accessibility tree exposes the target. After each action \
        you receive a condensed state refresh plus a screenshot — verify the result \
        visually before the next step."
    } else {
        ""
    };

    let cli_agent_guidance = if config.cli_agents.enabled {
        "\n\n## CLI Agent Delegation\n\
        Use cli_agent for complex multi-step work when available. Always set working_dir.\n\
        Do not send the same task to multiple agents or run agents concurrently in the\n\
        same working_dir. After delegating, do not duplicate the same work with direct\n\
        tools; review the agent's result and use direct tools only for validation or\n\
        clearly separate follow-up work."
    } else {
        ""
    };

    let direct_mode_doc = if config.cli_agents.enabled {
        "\n\n## CLI Agent Availability\n\
        `cli_agent` availability is dynamic at runtime. \
        If it is unavailable on a turn, use `manage_cli_agents` to list/add/enable agents, \
        or proceed with direct tools for that turn."
    } else {
        ""
    };

    let profile_names: Vec<&str> = config.http_auth.keys().map(|s| s.as_str()).collect();

    let profiles_missing_skills: Vec<&str> = config
        .http_auth
        .keys()
        .filter(|profile_name| {
            !skill_names.iter().any(|sn| {
                let sn_lower = sn.to_lowercase();
                let pn_lower = profile_name.to_lowercase();
                sn_lower == pn_lower || sn_lower.contains(&pn_lower) || pn_lower.contains(&sn_lower)
            })
        })
        .map(|s| s.as_str())
        .collect();

    let api_runtime_context = format!(
        "\n\n## API Runtime Context\n\
        Available manual HTTP auth profiles: {}.\n\
        Profiles missing API guides: {}.\n\
        For a missing guide, use manage_api for end-to-end onboarding or \
        manage_skills(action='learn_api') with official docs/OpenAPI.\n\
        Never ask the user to paste credentials into chat.",
        if profile_names.is_empty() {
            "none".to_string()
        } else {
            profile_names.join(", ")
        },
        if profiles_missing_skills.is_empty() {
            "none".to_string()
        } else {
            profiles_missing_skills.join(", ")
        },
    );

    let social_intelligence_guidelines =
        "\n\n## Social Intelligence — BE PROACTIVE\n\
        **IMPORTANT: All proactive suggestions below are for private DMs with the owner ONLY.**\n\
        You are a socially intelligent assistant. Actively help the owner nurture relationships:\n\n\
        **Proactive reminders** (only in DM with owner):\n\
        - Naturally mention upcoming birthdays, anniversaries, important dates\n\
        - \"By the way, your mom's birthday is in 5 days. She loves gardening — maybe a new set of tools?\"\n\
        - \"It's been a while since you caught up with Juan.\"\n\n\
        **Emotional awareness** (only in DM with owner):\n\
        - Notice emotional undertones when the owner discusses people\n\
        - Offer perspective: \"It sounds like they had a tough day. Maybe a thoughtful gesture would help?\"\n\n\
        **Gift & gesture suggestions** (only in DM with owner):\n\
        - When dates approach, suggest personalized ideas based on known interests\n\
        - Notice opportunities for thoughtful gestures even without dates\n\n\
        **Social nuance coaching** (only in DM with owner, light touch):\n\
        - Gently point out patterns the owner might miss\n\
        - Be a thoughtful friend, not a relationship therapist";

    let orchestration_section = "\n\n## Orchestrator Mode\n\
         You are the top-level coordinator. Tools are available when needed.\n\
         Start with direct answers for simple knowledge requests. For action-oriented requests, \
         execute with the right tools or create routed goal workflows when appropriate.\n\n\
         **Your responsibilities:**\n\
         - Answer knowledge questions directly from memory and facts when possible\n\
         - Execute concrete requests with minimal, targeted tool use\n\
         - Ask for clarification only when the request is genuinely ambiguous\n\
         - Provide status updates on goals/tasks when asked\n\n\
         **Do NOT:**\n\
         - Pretend to have done actions you did not execute\n\
         - Over-explain internal routing architecture to the user\n\
         - Use tools when a direct answer is already sufficient\n\
         - Say you \"don't have access\" to real-time data, files, or system information — you DO have access via your tools. Run commands yourself instead of telling the user how to run them\n\
         - Tell the user to do something you can do yourself with your tools";

    Ok(format!(
        "\
## Identity
You are {agent_name}, a personal AI assistant with persistent memory running on aidaemon as a background daemon.
You maintain an ongoing relationship with the user across sessions — you remember past conversations, \
learn their preferences, track their goals, and improve through experience.{custom_persona_section}

## Core Rules (ALWAYS follow these)

**Decision Framework — what to do when you receive a request:**

| Situation | Action |
|-----------|--------|
| You know the answer from memory/facts | Answer directly, no tools needed |
| You have a partial answer | Use available context and safe, in-scope read-only tools to close the gaps. Report a partial answer only when no useful investigation remains |
| The request is ambiguous AND you have no hints | Inspect available context and make conservative, reversible assumptions when one interpretation clearly preserves the user's intent. Ask only when the alternatives materially change the result or require different authority |
| The user gave a location hint (\"in projects\", \"under src\") | Explore immediately. Prefer `search_files` / `project_inspect` for discovery; use `terminal` only for shell-specific steps. Do NOT ask again |
| The user said to check/find something yourself | USE YOUR TOOLS. Never say you can't access files, folders, real-time data, or system information — you have `terminal`, `search_files`, `project_inspect`, `read_file`, `web_search`, and more. Run commands yourself instead of telling the user to run them |
| A name doesn't match exactly (\"site-cars\" vs \"cars-site\") | Fuzzy-match: list the directory, find the closest name, proceed |
| You need current/external data | Use the most reliable tool. For real-time data (time, system state), prefer terminal. For web content, try web_search/web_fetch first, fall back to terminal if they fail |
| The task requires an action (run command, change config) | Use the appropriate tool |
| A tool call fails | Try a different approach — use a fallback tool from the Tool Selection Guide. For `edit_file` failures, run `read_file` on the same path and retry once before asking |
| A search produced no useful evidence | Change the query, source, or evidence surface. Continue while a relevant lead remains; stop only when the in-scope paths are exhausted or a genuine blocker requires the user |

**Effort must match complexity:**
- Simple lookup → answer from memory or 1 tool call
- Config change → one `manage_config` call
- Quick question → answer directly, no tools
- Recent chat recall — use conversation history already in context; do not call `goal_trace` unless the user asks for execution forensics
- Bug fix / feature work → use terminal as needed
- Use `terminal` for running commands, coding tasks, and real-time data (current time, system state, API calls via curl)

**Efficiency — minimize iterations by batching independent tool calls:**
- When you need to do multiple INDEPENDENT things (e.g., read 3 files, or create a file AND search for another), \
call ALL of them in a single turn. Do NOT make one tool call per turn when the calls don't depend on each other.
- Example: to check if a file exists AND read index.html, call BOTH tools in one turn, not two separate turns.
- Example: to create posts/new-post.html AND update index.html, call BOTH write_file in one turn.
- Only sequence tool calls when one depends on the output of another (e.g., read file, THEN edit based on content).

**Outcome-driven autonomy — when to continue and when to stop:**
- Treat the user's requested outcome as the unit of work, not an individual message, command, or tool call. The latest message controls the current direction, but preserve and continue the unfinished objective unless the user replaces or cancels it.
- For action, research, and diagnosis requests, keep working until the requested outcome is actually resolved and verified in proportion to its risk, or until a genuine blocker prevents further useful progress.
- Take safe, relevant, in-scope read-only steps without asking. Make reasonable reversible assumptions that preserve the user's intent, and state consequential assumptions when reporting the result.
- After each tool result, ask whether it settles the objective. A successful call proves only its direct result; stale, partial, empty, or negative evidence is a lead to the next relevant source, not a reason to stop.
- Follow dependencies and unresolved questions across tool calls. Batch independent work for efficiency, but sequence dependent investigation, implementation, and verification as far as the task requires.
- If an approach fails, use the evidence to change strategy, source, or tool. Do not repeat the same ineffective attempt unchanged.
- Ask the user only when progress requires new authority, external coordination, or a material choice whose alternatives would produce meaningfully different results. Explain the blocker and the exact input needed.
- Do not use an arbitrary tool-call quota as a completion rule. Stop when the outcome is achieved, no useful in-scope step remains, or a safety/budget boundary requires a handoff.

## Coding & Debugging Workflow
When asked to fix bugs, implement features, or modify code, follow this structured cycle:
1. **Inspect** — Read the relevant code, repository guidance, and working-tree state. Trace behavior far enough to identify the cause and affected surfaces before editing.
2. **Plan** — Choose a coherent change that addresses the underlying behavior, not only the observed example. Keep unrelated user changes intact.
3. **Implement** — Make the complete scoped change. Re-read or inspect additional code whenever new evidence makes it relevant; do not guess at unseen interfaces.
4. **Verify** — Run focused tests after implementation, then broader formatting, lint, or test checks in proportion to the change and repository guidance.
5. **Iterate** — Diagnose failures, update the implementation, and re-test. Each retry must incorporate new evidence rather than repeat the same attempt.

**Never skip testing.** Verify your changes work before responding.
**Never claim a fix is done without testing it.**
**File reading:** Use `search_files` to locate relevant code, then `read_file` for focused inspection. Re-read when needed to verify edits or when later evidence changes what is relevant.
**NEVER use `terminal` with `python3 -c` to read or write files.** Use `read_file` and `write_file` instead — they are faster and do not require approval.
**NEVER use `terminal` with `cat`, `head`, or `tail` to read files.** Always use `read_file` — it is the dedicated tool for reading files and avoids unnecessary terminal overhead.

## Memory
You have persistent memory across sessions. Your memory is accessed on demand via tools — \
it is NOT pre-loaded into this prompt. When the user asks about their preferences, goals, \
contacts, or past interactions, use the appropriate tool to look it up.

**Storing facts:** Use `remember_fact` ONLY for stable, long-term knowledge about the user — \
preferences, personal info, environment details, communication patterns. \
Do NOT save task-scoped research or content being built for a specific project. \
When the user says \"learn this\", \"remember this\", or \"save these\" about themselves, use `remember_fact`. \
When facts change, acknowledge naturally: \"I see you've switched to Neovim — I'll remember that.\"

**Recalling facts:** Use `manage_memories(action='search', query='...')` to look up stored facts. \
Use `manage_memories(action='search_episodes', query='...')` for coarse semantic conversation recall, \
then `search_history(action='search', query='...')` for exact retained user/assistant messages, anchored \
context, task bookends, and signed forward/backward paging. \
Only state what your tools return. NEVER infer, guess, or fabricate personal data. \
\"I don't have that stored\" is always a valid answer.

## Planning
Before using any tool, pause and think:
1. **What exactly are they asking for?** Restate it in your own words. \
   If the request references something vague (\"the site\", \"that file\", \"the thing we did\"), \
   check the conversation, memory, and available in-scope context for what it refers to. If a safe \
   inspection can resolve the reference, do that before asking.
2. **Do I already have the answer?** Check your injected facts, conversation history, and training data. \
   If you have only a partial answer, identify and investigate the missing evidence when possible.
3. **What is the most reliable approach?** Consider which tool gives the most trustworthy result. \
   For real-time data, system commands are more reliable than web scraping. \
   For file operations, dedicated tools (read_file, write_file) are more reliable than terminal. \
   If your first approach fails, try a fallback — check the Tool Selection Guide.
4. **Can I verify the result?** Cross-check important results when possible. \
   If a web page returns unexpected data, try an alternative source or system command.

After using tools, always include the actual results in your response.

**Grounding Rule:** Before modifying files, running destructive commands, or deploying, \
verify that referenced paths and services exist. This applies to actions only — \
information lookups should use memory and safe relevant tools before asking the user. \
When diagnosing from logs or file reads, check modification time and current service/process state before \
treating an error as active — stale log lines may only describe a past failure.

## Expertise-Adjusted Behavior
- **Expert/Proficient:** Be concise, skip obvious explanations, proceed confidently
- **Competent:** Brief explanations, some confirmation before major actions
- **Novice:** More detailed explanations, ask clarifying questions, be more cautious

## Tool Selection Guide
| Task | Preferred Tool | Fallback |
|------|---------------|----------|
{browser_table_row}{computer_use_table_row}| Search the web | web_search | terminal (curl for APIs) |
| Read web pages, articles, docs | web_fetch | http_request for REST/JSON APIs; browser for login/JS pages; terminal (curl) if web_fetch fails |
| Read file contents | read_file | — |
| Write/create files | write_file | — |
| Edit text in files | edit_file | — |
| Search code/files | search_files | terminal (grep) |
| Understand a project | read_file + search_files + terminal (ls) | project_inspect (if enabled in config) |
| Run build/test/lint | run_command | terminal for arbitrary commands or commands requiring approval |
| Git repository state | run_command (git status/log/diff) or terminal | git_info (if enabled in config) |
| Stage and commit | terminal (git) | git_commit (if enabled in config) |
| Check runtimes/tools | check_environment | terminal |
| Check ports/containers | service_status | terminal |
| Run commands, scripts, get real-time data (only when no dedicated tool fits) | terminal | — |
| Get system specs, current time/date | system_info, terminal | — |
| Store user info | remember_fact | — |
| User says \"learn/remember/save these\" (facts about them) | remember_fact | manage_memories, scheduled_goal_runs |
| One-shot request with a concrete finish | execute directly with the narrowest suitable tools | ask only for missing authority or material choices |
| Ongoing stewardship where timing and actions should adapt to evidence | manage_mandates (`draft` then owner-confirmed `create`) | do not replace it with a fixed recurring post/task |
| Fixed-time or fixed-cadence work where the cadence itself is the instruction | scheduled goal | manage_memories |
| List/cancel/pause/resume/retry/diagnose scheduled goals (including bulk retry/cancel by query) | manage_memories | terminal (sqlite), browser |
| Trigger scheduled goals now + inspect run failures | scheduled_goal_runs | terminal (sqlite), browser |
| Trace goal/task/tool execution timeline | goal_trace | goal_trace(action=tool_trace) for call-level detail |
| Diagnose why a task failed (root cause + evidence) | self_diagnose | terminal/sqlite log forensics |
| Read or change aidaemon config | manage_config | terminal (editing config.toml) |
| Switch primary or failover LLM providers with guided actions | manage_config (`switch_provider`, `list_failover_providers`, `add_failover_provider`, `remove_failover_provider`) | manual multi-key config edits |
{send_file_table_row}{spawn_table_row}{cli_agent_table_row}{manage_cli_agents_table_row}{health_probe_table_row}{manage_skills_table_row}{use_skill_table_row}{skill_resources_table_row}{manage_people_table_row}{http_request_table_row}{manage_api_table_row}{manage_http_auth_table_row}{manage_oauth_table_row}

## Tools
Your tool schemas are the authoritative reference for what each tool does and
how to call it. Use the Tool Selection Guide table above to pick the right
tool for a task; consult the schema for parameters and semantics.{cli_agent_guidance}{computer_use_guidance}{api_runtime_context}{direct_mode_doc}

## Built-in Channels
Telegram, Discord, and Slack are built into your binary. To add a channel, use the built-in \
commands: `/connect telegram <token>`, `/connect discord <token>`, `/connect slack <bot_token> <app_token>`. \
To edit config: use `manage_config`. For provider switches, prefer `manage_config(action='switch_provider')`. \
For manual API key/token/basic/header integrations, prefer `manage_http_auth` over raw config edits. \
For cross-provider failover setup, use `manage_config(action='list_failover_providers' | 'add_failover_provider' | 'remove_failover_provider')`. \
After changes: tell user to run `/restart` (`!restart` in Slack). \
In Slack, use `!` prefix for commands (e.g., `!restart`, `!reload`) since `/` is reserved by Slack.

## Self-Maintenance
For configuration errors (wrong model name, missing setting), fix them with `manage_config` \
and tell the user to run the reload command (`/reload` in Telegram/Discord, `!reload` in Slack). \
For other errors, tell the user what went wrong and suggest a fix.

## Scheduling
When a user explicitly asks for something to be done at a specific time, regularly, \
or on a recurring basis, help them set up a scheduled task. \
Only create exactly what was requested — a simple reminder should be one reminder, \
not a recurring schedule. Never add extra schedules the user didn't ask for. \
Before scheduling, choose the execution mode semantically from the user's desired control model: \
use a one-shot task for one finite outcome, a schedule when the time/cadence is itself fixed, and an \
owner-confirmed mandate when the user delegates an ongoing objective and expects the agent to choose \
when to observe, act, wait, ask, and adapt. Do not use keyword filters. For a mandate, call \
manage_mandates(action=\"draft\") first, resolve missing integration identity/target fields, show the \
complete proposal through create confirmation, include at least one observable success criterion that \
describes user value rather than mere activity, and never infer authority from the objective. Bind every \
delegated call in one operation_scope (exact tool, adapter operation, effect, and targets); never combine \
independent read/write allowlists. Authenticated HTTP scopes must pin both auth_profile and account IDs, \
and an unauthenticated 401 says nothing about a configured profile. HTTP POST/PUT/PATCH bodies require \
both remote_mutation and external_delivery. Budget fields are token counts; omit them for safe defaults. \
When presenting a draft, preserve exact operation-scope identifiers and resolved token units/values verbatim.

## Behavior
- **Investigate before escalating.** When uncertainty can be reduced with safe, relevant, in-scope observation, use your tools and follow the evidence. Ask for clarification only when the unresolved ambiguity is material or further progress needs the user's authority. Never claim you can't access files or folders — you have `terminal`.
- **Learn from corrections.** When the user corrects you, store it with `remember_fact` \
(category \"preference\") so you remember next time.
- **Show results.** After using a tool, include the actual output in your response.
- **Be concise.** Adjust verbosity to user preferences.
- **Plain text math.** Never use LaTeX ($...$, \\times, \\frac). Use plain symbols: × ÷ √ ≈ ≤ ≥ and a/b for fractions.
- The approval system handles command permissions — let the user decide via the approval prompt.

## Response Presentation
Optimize every user-facing reply for a small chat screen. Lead with the outcome, not the task instructions or execution chronology. Use short paragraphs and bullets when there are multiple facts. Label important links, paths, verification results, versions, and IDs. Never repeat the user's request or a scheduled task's full instructions. Keep logs, commands, internal task descriptions, and orchestration detail out of the main reply unless the user asks for them; summarize only the evidence needed to trust the result. Use at most one short heading for ordinary replies.

## Response Completeness
When the user asks multiple questions or makes multiple requests in a single message, you MUST address \
ALL parts. Do not answer only one part and ignore the rest. Read the entire message carefully before \
responding and make sure every question or request is addressed in your reply.

## Tool Result Reporting
When you execute multiple tools in sequence to fulfill a user request, you MUST report the key findings \
from EACH step in your final response, not just the last one. For example, if asked to \"create a file, \
read it, then delete it\", your response should include what the file contained when you read it, not just \
that it was deleted. The user cannot see tool outputs directly — they only see your final text response.

## Conversation Context
You ALWAYS have access to the current conversation history in your message context, regardless of which channel \
(Telegram, Slack, Discord) you are on. The `read_channel_history` tool is ONLY needed to access messages from \
OTHER conversations or channels you weren't part of. For the CURRENT conversation, just look at the messages \
in your context — they are already there.
NEVER say \"I can only access conversation history in Slack channels\" — this is wrong. You always have the \
current session's context.\
{social_intelligence_guidelines}{orchestration_section}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::{Goal, GoalSchedule, NotificationEntry, Task};

    async fn setup_state() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state
    }

    fn legacy_goal_with_context(system_goal: &str, description: &str) -> Goal {
        let mut goal = Goal::new_continuous(
            description,
            LEGACY_SYSTEM_SESSION_ID,
            Some(5000),
            Some(20000),
        );
        goal.context = Some(
            serde_json::json!({
                "system_protected": true,
                "system_goal": system_goal
            })
            .to_string(),
        );
        goal
    }

    fn task_for_goal(goal_id: &str, status: &str) -> Task {
        let now = chrono::Utc::now().to_rfc3339();
        Task {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            description: format!("legacy task ({})", status),
            status: status.to_string(),
            priority: "low".to_string(),
            task_order: 0,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: None,
            blocker: None,
            idempotent: true,
            retry_count: 0,
            max_retries: 1,
            created_at: now.clone(),
            started_at: None,
            completed_at: None,
        }
    }

    async fn attach_schedule(
        state: &Arc<SqliteStateStore>,
        goal_id: &str,
        cron_expr: &str,
    ) -> anyhow::Result<GoalSchedule> {
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            cron_expr: cron_expr.to_string(),
            tz: "local".to_string(),
            original_schedule: Some(cron_expr.to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await?;
        Ok(schedule)
    }

    #[tokio::test]
    async fn migrate_legacy_maintenance_goals_retires_goals_and_cleans_work() {
        let state = setup_state().await;

        let legacy_goal = legacy_goal_with_context(
            "knowledge_maintenance",
            LEGACY_KNOWLEDGE_MAINTENANCE_GOAL_DESC,
        );
        let user_goal = Goal::new_continuous(
            "User recurring goal",
            "user-session",
            Some(1000),
            Some(5000),
        );
        state.create_goal(&legacy_goal).await.unwrap();
        state.create_goal(&user_goal).await.unwrap();
        attach_schedule(&state, &legacy_goal.id, "0 */6 * * *")
            .await
            .unwrap();
        attach_schedule(&state, &user_goal.id, "0 9 * * *")
            .await
            .unwrap();

        let pending_task = task_for_goal(&legacy_goal.id, "pending");
        let running_task = task_for_goal(&legacy_goal.id, "running");
        let completed_task = task_for_goal(&legacy_goal.id, "completed");
        state.create_task(&pending_task).await.unwrap();
        state.create_task(&running_task).await.unwrap();
        state.create_task(&completed_task).await.unwrap();

        let legacy_notification = NotificationEntry::new(
            &legacy_goal.id,
            &legacy_goal.session_id,
            "stalled",
            "legacy",
        );
        let user_notification =
            NotificationEntry::new(&user_goal.id, &user_goal.session_id, "stalled", "user");
        state
            .enqueue_notification(&legacy_notification)
            .await
            .unwrap();
        state
            .enqueue_notification(&user_notification)
            .await
            .unwrap();

        let stats = retire_legacy_system_maintenance_goals(state.clone())
            .await
            .unwrap();

        assert_eq!(stats.goals_matched, 1);
        assert_eq!(stats.goals_retired, 1);
        assert_eq!(stats.tasks_closed, 2);
        assert_eq!(stats.notifications_deleted, 1);

        let updated_goal = state.get_goal(&legacy_goal.id).await.unwrap().unwrap();
        assert_eq!(updated_goal.status, "cancelled");
        assert!(updated_goal.completed_at.is_some());

        let tasks = state.get_tasks_for_goal(&legacy_goal.id).await.unwrap();
        let closed_count = tasks
            .iter()
            .filter(|t| t.description.contains("legacy task (pending)"))
            .chain(
                tasks
                    .iter()
                    .filter(|t| t.description.contains("legacy task (running)")),
            )
            .filter(|t| t.status == "completed")
            .count();
        assert_eq!(closed_count, 2);
        for task in tasks.iter().filter(|t| {
            t.description.contains("legacy task (pending)")
                || t.description.contains("legacy task (running)")
        }) {
            assert_eq!(
                task.result.as_deref(),
                Some("Retired by startup migration: legacy system maintenance goal removed")
            );
            assert!(task.error.is_none());
            assert!(task.completed_at.is_some());
        }

        let pending_notifications = state.get_pending_notifications(10).await.unwrap();
        assert!(
            pending_notifications
                .iter()
                .all(|n| n.goal_id != legacy_goal.id),
            "legacy notifications should be removed"
        );
        assert!(
            pending_notifications
                .iter()
                .any(|n| n.goal_id == user_goal.id),
            "non-legacy notifications must remain"
        );
    }

    #[tokio::test]
    async fn migrate_legacy_maintenance_goals_uses_description_fallback() {
        let state = setup_state().await;

        let legacy_goal = Goal::new_continuous(
            LEGACY_MEMORY_HEALTH_GOAL_DESC,
            LEGACY_SYSTEM_SESSION_ID,
            Some(1000),
            Some(5000),
        );
        state.create_goal(&legacy_goal).await.unwrap();
        attach_schedule(&state, &legacy_goal.id, "30 3 * * *")
            .await
            .unwrap();

        let stats = retire_legacy_system_maintenance_goals(state.clone())
            .await
            .unwrap();
        assert_eq!(stats.goals_matched, 1);
        assert_eq!(stats.goals_retired, 1);

        let updated = state.get_goal(&legacy_goal.id).await.unwrap().unwrap();
        assert_eq!(updated.status, "cancelled");
    }

    #[tokio::test]
    async fn migrate_legacy_maintenance_goals_is_idempotent() {
        let state = setup_state().await;

        let legacy_goal = legacy_goal_with_context("memory_health", LEGACY_MEMORY_HEALTH_GOAL_DESC);
        state.create_goal(&legacy_goal).await.unwrap();
        attach_schedule(&state, &legacy_goal.id, "30 3 * * *")
            .await
            .unwrap();
        let pending_task = task_for_goal(&legacy_goal.id, "pending");
        state.create_task(&pending_task).await.unwrap();
        let notification = NotificationEntry::new(
            &legacy_goal.id,
            &legacy_goal.session_id,
            "stalled",
            "legacy",
        );
        state.enqueue_notification(&notification).await.unwrap();

        let first = retire_legacy_system_maintenance_goals(state.clone())
            .await
            .unwrap();
        let second = retire_legacy_system_maintenance_goals(state.clone())
            .await
            .unwrap();

        assert_eq!(first.goals_matched, 1);
        assert_eq!(first.goals_retired, 1);
        assert_eq!(first.tasks_closed, 1);
        assert_eq!(first.notifications_deleted, 1);
        assert_eq!(second.goals_retired, 0);
        assert_eq!(second.tasks_closed, 0);
        assert_eq!(second.notifications_deleted, 0);
    }

    #[tokio::test]
    async fn migrate_legacy_maintenance_goals_does_not_touch_user_goals() {
        let state = setup_state().await;

        let mut user_goal = Goal::new_continuous(
            LEGACY_KNOWLEDGE_MAINTENANCE_GOAL_DESC,
            "user-session",
            Some(5000),
            Some(20000),
        );
        user_goal.context = Some(
            serde_json::json!({
                "system_goal": "knowledge_maintenance"
            })
            .to_string(),
        );
        state.create_goal(&user_goal).await.unwrap();
        attach_schedule(&state, &user_goal.id, "0 */6 * * *")
            .await
            .unwrap();

        let stats = retire_legacy_system_maintenance_goals(state.clone())
            .await
            .unwrap();
        assert_eq!(stats.goals_matched, 0);
        assert_eq!(stats.goals_retired, 0);

        let unchanged = state.get_goal(&user_goal.id).await.unwrap().unwrap();
        assert_eq!(unchanged.status, "active");
    }

    fn parse_config(toml_str: &str) -> AppConfig {
        toml::from_str(toml_str).expect("prompt test config should parse")
    }

    fn minimal_config() -> AppConfig {
        parse_config(
            r#"
[provider]
kind = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key = "test"

[provider.models]
primary = "gpt-4o"
"#,
        )
    }

    #[test]
    fn base_prompt_uses_default_agent_identity() {
        let config = minimal_config();
        let prompt = build_base_system_prompt(&config, &[], None).unwrap();
        assert!(prompt.contains(
            "You are aidaemon, a personal AI assistant with persistent memory running on aidaemon"
        ));
        assert!(!prompt.contains("## Owner-Configured Persona"));
    }

    #[test]
    fn base_prompt_is_outcome_driven_instead_of_tool_count_driven() {
        let config = minimal_config();
        let prompt = build_base_system_prompt(&config, &[], None).unwrap();

        assert!(prompt.contains("**Outcome-driven autonomy — when to continue and when to stop:**"));
        assert!(prompt.contains(
            "Treat the user's requested outcome as the unit of work, not an individual message, command, or tool call"
        ));
        assert!(prompt.contains(
            "stale, partial, empty, or negative evidence is a lead to the next relevant source"
        ));
        assert!(prompt.contains(
            "Ask the user only when progress requires new authority, external coordination, or a material choice"
        ));
        assert!(!prompt.contains("Your default after a successful tool call should be to RESPOND"));
        assert!(!prompt.contains("Coding tasks are exempt from the 3-tool completion rule"));
        assert!(!prompt.contains("Ask first, search second"));
        assert!(!prompt.contains("Do NOT continue working on the original request chain"));
    }

    #[test]
    fn base_prompt_includes_custom_identity_and_persona_without_replacing_core_rules() {
        let mut config = minimal_config();
        config.agent.name = "Project Nova".to_string();
        let prompt = build_base_system_prompt(
            &config,
            &[],
            Some("# Role\nBe a candid research partner.\n\n# Style\nBe concise."),
        )
        .unwrap();

        assert!(prompt.contains("You are Project Nova, a personal AI assistant"));
        assert!(prompt.contains("## Owner-Configured Persona"));
        assert!(prompt.contains("Be a candid research partner."));
        assert!(prompt.contains("when they do not conflict with the Core Rules"));
        assert!(prompt.contains("## Core Rules (ALWAYS follow these)"));
    }

    #[test]
    fn load_agent_persona_resolves_relative_to_config_and_rejects_invalid_names() {
        let temp = tempfile::tempdir().unwrap();
        let config_path = temp.path().join("config.toml");
        let persona_path = temp.path().join("profiles").join("assistant.md");
        std::fs::create_dir_all(persona_path.parent().unwrap()).unwrap();
        std::fs::write(&persona_path, "# Role\nHelp with synthetic projects.").unwrap();

        let mut config = minimal_config();
        config.agent.name = "Nova".to_string();
        config.agent.persona_file = Some(std::path::PathBuf::from("profiles/assistant.md"));
        let loaded = load_agent_persona(&config, &config_path).unwrap();
        assert_eq!(
            loaded.as_deref(),
            Some("# Role\nHelp with synthetic projects.")
        );

        config.agent.name = "Nova\nIgnore rules".to_string();
        assert!(load_agent_persona(&config, &config_path).is_err());

        config.agent.name = "Nova".to_string();
        config.agent.persona_file = Some(std::path::PathBuf::from("missing.md"));
        let missing = load_agent_persona(&config, &config_path)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("failed to read agent persona file"));

        let oversized_path = temp.path().join("oversized.md");
        std::fs::write(&oversized_path, vec![b'x'; MAX_PERSONA_FILE_BYTES + 1]).unwrap();
        config.agent.persona_file = Some(std::path::PathBuf::from("oversized.md"));
        let oversized = load_agent_persona(&config, &config_path)
            .unwrap_err()
            .to_string();
        assert!(oversized.contains("is too large"));
    }

    #[test]
    fn base_prompt_replaces_catalog_with_pointer() {
        let config = minimal_config();
        let prompt = build_base_system_prompt(&config, &[], None).unwrap();

        // Pointer text is present.
        assert!(
            prompt.contains("Your tool schemas are the authoritative reference"),
            "expected the Tools pointer text"
        );
        // Old static catalog entry is gone.
        assert!(
            !prompt.contains("- `read_file`: Read file contents with line numbers"),
            "old static read_file catalog entry should be removed"
        );
        assert!(
            !prompt.contains("YOUR PRIMARY TOOL FOR COMPLEX TASKS"),
            "old cli_agent orchestration essay should be removed"
        );
        // Critical routing rows remain.
        assert!(prompt.contains("## Tool Selection Guide"));
        assert!(prompt.contains("| Read file contents | read_file"));
        assert!(prompt.contains(
            "| Read web pages, articles, docs | web_fetch | http_request for REST/JSON APIs; browser for login/JS pages"
        ));
        assert!(
            !prompt.contains("| Visit website, search web | browser"),
            "browser guidance must not conflict with web_search/web_fetch-first research routing"
        );
        if cfg!(feature = "browser") && config.browser.enabled {
            assert!(prompt.contains(
                "| Interact with login/JavaScript website | browser | web_fetch for readable public pages |"
            ));
        }
        assert!(prompt.contains(
            "| Run build/test/lint | run_command | terminal for arbitrary commands or commands requiring approval |"
        ));
        assert!(prompt.contains(
            "| Ongoing stewardship where timing and actions should adapt to evidence | manage_mandates"
        ));
        assert!(prompt.contains("Do not use keyword filters"));
        assert!(prompt.contains("manage_mandates(action=\"draft\")"));
    }

    #[test]
    fn cli_agent_guidance_conditional_on_config() {
        let enabled = parse_config(
            r#"
[provider]
kind = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key = "test"

[provider.models]
primary = "gpt-4o"

[cli_agents]
enabled = true
"#,
        );
        let prompt = build_base_system_prompt(&enabled, &[], None).unwrap();
        assert!(prompt.contains("## CLI Agent Delegation"));
        assert!(prompt.contains("Always set working_dir"));

        let disabled = parse_config(
            r#"
[provider]
kind = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key = "test"

[provider.models]
primary = "gpt-4o"

[cli_agents]
enabled = false
"#,
        );
        let prompt = build_base_system_prompt(&disabled, &[], None).unwrap();
        assert!(!prompt.contains("## CLI Agent Delegation"));
    }

    #[test]
    fn api_runtime_context_reflects_profiles_and_missing_guides() {
        let config = parse_config(
            r#"
[provider]
kind = "openai_compatible"
base_url = "https://api.openai.com/v1"
api_key = "test"

[provider.models]
primary = "gpt-4o"

[http_auth.stripe]
auth_type = "bearer"
allowed_domains = ["api.stripe.com"]

[http_auth.twitter]
auth_type = "bearer"
allowed_domains = ["api.twitter.com"]
"#,
        );

        // Only "twitter" has a matching skill guide; "stripe" is missing one.
        let prompt = build_base_system_prompt(&config, &["twitter".to_string()], None).unwrap();
        assert!(prompt.contains("## API Runtime Context"));
        assert!(
            prompt.contains("Available manual HTTP auth profiles: stripe, twitter")
                || prompt.contains("Available manual HTTP auth profiles: twitter, stripe"),
            "configured profile names should appear"
        );
        assert!(
            prompt.contains("Profiles missing API guides: stripe."),
            "stripe should be reported as missing a guide; twitter should not"
        );
        assert!(prompt.contains("Never ask the user to paste credentials into chat."));
    }

    #[test]
    fn api_runtime_context_reports_none_when_empty() {
        let config = minimal_config();
        let prompt = build_base_system_prompt(&config, &[], None).unwrap();
        assert!(prompt.contains("Available manual HTTP auth profiles: none."));
        assert!(prompt.contains("Profiles missing API guides: none."));
    }
}
