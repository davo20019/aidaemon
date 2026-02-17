use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use chrono::Utc;
use serenity::all::{
    ChannelId, Command, CommandInteraction, ComponentInteraction, Context, CreateAttachment,
    CreateButton, CreateCommand, CreateInteractionResponse, CreateInteractionResponseMessage,
    CreateMessage, EditInteractionResponse, EventHandler, GatewayIntents, GuildId, Interaction,
    Message as SerenityMessage, Ready,
};
use serenity::builder::CreateActionRow;
use serenity::model::application::ButtonStyle;
use serenity::Client;
use tokio::sync::Mutex;
use tracing::{info, warn};

use super::formatting::{build_help_text, format_number, sanitize_filename, split_message};
use crate::agent::Agent;
use crate::channels::{should_ignore_lightweight_interjection, ChannelHub, SessionMap};
use crate::config::AppConfig;
use crate::tasks::TaskRegistry;
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::traits::{Channel, ChannelCapabilities, StateStore};
use crate::types::{ApprovalResponse, MediaKind, MediaMessage};
use crate::types::{ChannelContext, ChannelVisibility, StatusUpdate, UserRole};

/// Discord channel implementation using the serenity library.
pub struct DiscordChannel {
    /// Bot username fetched from Discord API (e.g., "my_bot").
    /// Populated on first start() call.
    bot_username: std::sync::RwLock<String>,
    bot_token: String,
    allowed_user_ids: std::sync::RwLock<Vec<u64>>,
    /// Discord user IDs recognized as owners (from `users.owner_ids.discord`).
    owner_user_ids: Vec<u64>,
    guild_id: Option<u64>,
    agent: Arc<Agent>,
    config_path: PathBuf,
    pending_approvals: Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalResponse>>>,
    session_map: SessionMap,
    task_registry: Arc<TaskRegistry>,
    files_enabled: bool,
    inbox_dir: PathBuf,
    max_file_size_mb: u64,
    state: Arc<dyn StateStore>,
    /// Stored after the client starts so we can send messages via the REST API.
    http: Mutex<Option<Arc<serenity::http::Http>>>,
    /// Reference to the channel hub for dynamic bot registration.
    channel_hub: std::sync::RwLock<Option<Weak<ChannelHub>>>,
    /// Seconds of no heartbeat before declaring the agent stuck (0 = disabled).
    watchdog_stale_threshold_secs: u64,
    /// Daemon start time used for post-restart UX guardrails.
    started_at: Instant,
}

impl DiscordChannel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bot_token: &str,
        allowed_user_ids: Vec<u64>,
        owner_user_ids: Vec<u64>,
        guild_id: Option<u64>,
        agent: Arc<Agent>,
        config_path: PathBuf,
        session_map: SessionMap,
        task_registry: Arc<TaskRegistry>,
        files_enabled: bool,
        inbox_dir: PathBuf,
        max_file_size_mb: u64,
        state: Arc<dyn StateStore>,
        watchdog_stale_threshold_secs: u64,
    ) -> Self {
        Self {
            bot_username: std::sync::RwLock::new("discord".to_string()),
            bot_token: bot_token.to_string(),
            allowed_user_ids: std::sync::RwLock::new(allowed_user_ids),
            owner_user_ids,
            guild_id,
            agent,
            config_path,
            pending_approvals: Mutex::new(HashMap::new()),
            session_map,
            task_registry,
            files_enabled,
            inbox_dir,
            max_file_size_mb,
            state,
            http: Mutex::new(None),
            channel_hub: std::sync::RwLock::new(None),
            watchdog_stale_threshold_secs,
            started_at: Instant::now(),
        }
    }

    /// Set the channel hub reference for dynamic bot registration.
    pub fn set_channel_hub(&self, hub: Weak<ChannelHub>) {
        if let Ok(mut guard) = self.channel_hub.write() {
            *guard = Some(hub);
        }
    }

    /// Get the bot's username (cached after first start).
    fn get_bot_username(&self) -> String {
        self.bot_username
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Set the bot's username (called during start() after fetching from API).
    fn set_bot_username(&self, username: String) {
        if let Ok(mut guard) = self.bot_username.write() {
            *guard = username;
        }
    }

    /// Build the session ID for a channel/user, prefixing with bot name if not "discord".
    fn session_id(&self, base_id: &str) -> String {
        let username = self.get_bot_username();
        if username == "discord" {
            base_id.to_string()
        } else {
            format!("{}:{}", username, base_id)
        }
    }

    /// Get the channel identifier for the session map.
    fn channel_name(&self) -> String {
        let username = self.get_bot_username();
        if username == "discord" {
            "discord".to_string()
        } else {
            format!("discord:{}", username)
        }
    }

    /// Start the Discord client with automatic retry on crash.
    /// Uses exponential backoff: 5s → 10s → 20s → 40s → 60s cap.
    pub async fn start_with_retry(self: Arc<Self>) {
        let initial_backoff = Duration::from_secs(5);
        let max_backoff = Duration::from_secs(60);
        let stable_threshold = Duration::from_secs(60);
        let mut backoff = initial_backoff;

        loop {
            info!("Starting Discord client");
            let started = tokio::time::Instant::now();
            self.clone().start().await;
            let ran_for = started.elapsed();

            if ran_for >= stable_threshold {
                backoff = initial_backoff;
            }

            warn!(
                backoff_secs = backoff.as_secs(),
                ran_for_secs = ran_for.as_secs(),
                "Discord client stopped, restarting"
            );
            tokio::time::sleep(backoff).await;
            backoff = std::cmp::min(backoff * 2, max_backoff);
        }
    }

    async fn start(self: Arc<Self>) {
        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let handler = DiscordHandler {
            channel: Arc::clone(&self),
        };

        let mut client = match Client::builder(&self.bot_token, intents)
            .event_handler(handler)
            .await
        {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to create Discord client: {}", e);
                return;
            }
        };

        // Store the HTTP client so Channel trait methods can send messages.
        {
            let mut http = self.http.lock().await;
            *http = Some(client.http.clone());
        }

        if let Err(e) = client.start().await {
            warn!("Discord client error: {}", e);
        }
    }

    /// Resolve a session ID to a ChannelId we can send messages to.
    async fn resolve_channel_id(&self, session_id: &str) -> anyhow::Result<ChannelId> {
        // Session ID format: "{bot_name}:discord:dm:{user_id}" or "discord:dm:{user_id}" (for default)
        // Strip the bot name prefix if present
        let base_session_id = if self.name() != "default" {
            session_id
                .strip_prefix(&format!("{}:", self.name()))
                .unwrap_or(session_id)
        } else {
            session_id
        };

        let http = self.get_http().await?;
        if let Some(user_id_str) = base_session_id.strip_prefix("discord:dm:") {
            let user_id: u64 = user_id_str.parse()?;
            let user = serenity::model::id::UserId::new(user_id);
            let dm_channel = user.create_dm_channel(&http).await?;
            Ok(dm_channel.id)
        } else if let Some(channel_id_str) = base_session_id.strip_prefix("discord:ch:") {
            let channel_id: u64 = channel_id_str.parse()?;
            Ok(ChannelId::new(channel_id))
        } else {
            anyhow::bail!("Invalid Discord session ID: {}", session_id)
        }
    }

    async fn get_http(&self) -> anyhow::Result<Arc<serenity::http::Http>> {
        let guard = self.http.lock().await;
        guard
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Discord HTTP client not ready"))
    }

    fn is_authorized(&self, user_id: u64) -> bool {
        let ids = self
            .allowed_user_ids
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if ids.is_empty() {
            // No allowed users configured — will be handled by auto-claim in handle_message_event.
            return false;
        }
        ids.contains(&user_id)
    }

    /// Auto-claim: when allowed_user_ids is empty OR contains only stale IDs from
    /// another platform (e.g. Telegram IDs saved for a Discord bot via /connect),
    /// register the first DM user as the allowed user and persist to DB.
    ///
    /// Only applies to dynamic bots (owner_user_ids is empty). Config-based bots
    /// with explicit owner_user_ids are never auto-claimable.
    async fn try_auto_claim(&self, user_id: u64, is_dm: bool) -> bool {
        if !is_dm {
            return false;
        }
        // Config-based bots have owner_user_ids set — never auto-claim those.
        if !self.owner_user_ids.is_empty() {
            return false;
        }

        // Claim: replace any stale IDs with this user.
        {
            let mut ids = self
                .allowed_user_ids
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if ids.contains(&user_id) {
                return true; // Already authorized.
            }
            info!(
                user_id,
                old_ids = ?*ids,
                "Auto-claiming Discord bot for DM user (replacing stale IDs)"
            );
            ids.clear();
            ids.push(user_id);
        }

        // Persist to DB so it survives restarts
        if let Err(e) = self
            .state
            .update_dynamic_bot_allowed_users(&self.bot_token, &[user_id.to_string()])
            .await
        {
            warn!("Failed to persist auto-claimed user to DB: {}", e);
        }

        true
    }

    /// Build a session ID from a Discord message.
    fn session_id_from_message(&self, msg: &SerenityMessage) -> String {
        let base = if msg.guild_id.is_some() {
            format!("discord:ch:{}", msg.channel_id)
        } else {
            format!("discord:dm:{}", msg.author.id)
        };
        self.session_id(&base)
    }

    fn spawn_typing_indicator(
        typing_channel: ChannelId,
        typing_http: Arc<serenity::http::Http>,
        typing_cancel: tokio_util::sync::CancellationToken,
        heartbeat: Arc<AtomicU64>,
        stale_threshold_secs: u64,
    ) {
        let typing_token = typing_cancel.clone();
        tokio::spawn(async move {
            loop {
                let _ = typing_channel.broadcast_typing(&typing_http).await;
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(8)) => {
                        if stale_threshold_secs > 0 {
                            let last_hb = heartbeat.load(Ordering::Relaxed);
                            let now = SystemTime::now().duration_since(UNIX_EPOCH)
                                .unwrap_or_default().as_secs();
                            if now.saturating_sub(last_hb) > stale_threshold_secs {
                                break;
                            }
                        }
                    }
                    _ = typing_token.cancelled() => break,
                }
            }
        });
    }

    fn spawn_status_task(
        mut status_rx: tokio::sync::mpsc::Receiver<StatusUpdate>,
        status_http: Arc<serenity::http::Http>,
        status_channel_id: ChannelId,
        is_dm: bool,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut last_sent = tokio::time::Instant::now() - Duration::from_secs(10);
            let min_interval = Duration::from_secs(3);
            while let Some(update) = status_rx.recv().await {
                // In non-DM channels: suppress all status messages.
                // Discord's native "typing..." indicator is sufficient.
                // (except BudgetExtended which must always reach the user)
                if !is_dm && !matches!(&update, StatusUpdate::BudgetExtended { .. }) {
                    continue;
                }
                let now = tokio::time::Instant::now();
                // Skip rate limiting for ToolProgress with URLs (e.g., OAuth authorize links)
                // and for BudgetExtended (cost notifications should always be delivered)
                let has_url = matches!(&update, StatusUpdate::ToolProgress { chunk, .. }
                    if chunk.contains("https://") || chunk.contains("http://"));
                let is_budget_ext = matches!(&update, StatusUpdate::BudgetExtended { .. });
                if !has_url && !is_budget_ext && now.duration_since(last_sent) < min_interval {
                    continue;
                }
                let text = match &update {
                    StatusUpdate::Thinking(_) => "Thinking...".to_string(),
                    StatusUpdate::ToolStart { name, summary } => {
                        if summary.is_empty() {
                            format!("Using {}...", name)
                        } else {
                            format!("Using {}: {}...", name, summary)
                        }
                    }
                    StatusUpdate::ToolProgress { name, chunk } => {
                        if chunk.contains("https://") || chunk.contains("http://") {
                            format!("📤 {}\n{}", name, chunk)
                        } else {
                            let preview: String = chunk.chars().take(100).collect();
                            if chunk.len() > 100 {
                                format!("📤 {}: {}...", name, preview)
                            } else {
                                format!("📤 {}: {}", name, chunk)
                            }
                        }
                    }
                    StatusUpdate::ToolComplete { name, summary } => {
                        format!("✓ {}: {}", name, summary)
                    }
                    StatusUpdate::ToolCancellable { name, task_id } => {
                        format!("⏳ {} started (task_id: {})", name, task_id)
                    }
                    StatusUpdate::ProgressSummary {
                        elapsed_mins,
                        summary,
                    } => {
                        format!("📊 Progress ({} min): {}", elapsed_mins, summary)
                    }
                    StatusUpdate::IterationWarning { current, threshold } => {
                        format!(
                            "⚠️ Approaching soft limit: {} of {} iterations",
                            current, threshold
                        )
                    }
                    StatusUpdate::PlanCreated {
                        description,
                        total_steps,
                        ..
                    } => {
                        format!("📋 Plan created: {} ({} steps)", description, total_steps)
                    }
                    StatusUpdate::PlanStepStart {
                        step_index,
                        total_steps,
                        description,
                        ..
                    } => {
                        format!(
                            "▶️ Step {}/{}: {}",
                            step_index + 1,
                            total_steps,
                            description
                        )
                    }
                    StatusUpdate::PlanStepComplete {
                        step_index,
                        total_steps,
                        description,
                        summary,
                        ..
                    } => {
                        let base = format!(
                            "✅ Step {}/{} done: {}",
                            step_index + 1,
                            total_steps,
                            description
                        );
                        if let Some(s) = summary {
                            format!("{} - {}", base, s)
                        } else {
                            base
                        }
                    }
                    StatusUpdate::PlanStepFailed {
                        step_index,
                        description,
                        error,
                        ..
                    } => {
                        format!(
                            "❌ Step {} failed: {} - {}",
                            step_index + 1,
                            description,
                            error
                        )
                    }
                    StatusUpdate::PlanComplete {
                        description,
                        total_steps,
                        duration_secs,
                        ..
                    } => {
                        let mins = duration_secs / 60;
                        let secs = duration_secs % 60;
                        format!(
                            "🎉 Plan complete: {} ({} steps in {}m {}s)",
                            description, total_steps, mins, secs
                        )
                    }
                    StatusUpdate::PlanAbandoned { description, .. } => {
                        format!("🚫 Plan abandoned: {}", description)
                    }
                    StatusUpdate::PlanRevised {
                        description,
                        reason,
                        new_total_steps,
                        ..
                    } => {
                        format!(
                            "🔄 Plan revised: {} ({} steps) - {}",
                            description, new_total_steps, reason
                        )
                    }
                    StatusUpdate::BudgetExtended {
                        old_budget,
                        new_budget,
                        extension,
                        max_extensions,
                    } => {
                        format!(
                            "💰 Auto-extended token budget {} → {} ({}/{}) — continuing.",
                            old_budget, new_budget, extension, max_extensions
                        )
                    }
                };
                let _ = status_channel_id.say(&status_http, &text).await;
                last_sent = tokio::time::Instant::now();
            }
        })
    }

    /// Handle an incoming Discord message.
    async fn handle_message_event(self: &Arc<Self>, ctx: &Context, msg: SerenityMessage) {
        // Ignore bot messages
        if msg.author.bot {
            return;
        }

        let user_id = msg.author.id.get();
        let is_dm = msg.guild_id.is_none();
        // In guild channels, let all server members through — they get Guest role
        // from determine_role(). Auth gate only applies to DMs.
        if is_dm && !self.is_authorized(user_id) && !self.try_auto_claim(user_id, is_dm).await {
            warn!(user_id, "Unauthorized Discord user attempted access");
            let _ = msg.channel_id.say(&ctx.http, "Unauthorized.").await;
            return;
        }

        let text = if msg.content.is_empty() && !msg.attachments.is_empty() {
            // File message
            if self.files_enabled {
                match self.handle_file_message(ctx, &msg).await {
                    Ok(file_text) => file_text,
                    Err(e) => {
                        let _ = msg
                            .channel_id
                            .say(&ctx.http, format!("File error: {}", e))
                            .await;
                        return;
                    }
                }
            } else {
                let _ = msg
                    .channel_id
                    .say(&ctx.http, "I can only process text messages.")
                    .await;
                return;
            }
        } else if !msg.content.is_empty() {
            // Text might be accompanied by attachments
            let mut text = msg.content.clone();
            if self.files_enabled && !msg.attachments.is_empty() {
                match self.handle_file_message(ctx, &msg).await {
                    Ok(file_text) => {
                        text = format!("{}\n{}", file_text, text);
                    }
                    Err(e) => {
                        let _ = msg
                            .channel_id
                            .say(&ctx.http, format!("File error: {}", e))
                            .await;
                        return;
                    }
                }
            }
            text
        } else {
            return;
        };

        // Handle slash-style commands (text commands starting with /)
        if text.starts_with('/') {
            self.handle_text_command(ctx, &msg, &text).await;
            return;
        }

        let session_id = self.session_id_from_message(&msg);
        let user_role = super::telegram::determine_role(&self.owner_user_ids, user_id);

        // Register this session with the channel hub (in-memory + persistent)
        {
            let channel_name = self.channel_name();
            let mut map = self.session_map.write().await;
            map.insert(session_id.clone(), channel_name.clone());
            let _ = self
                .state
                .save_session_channel(&session_id, &channel_name)
                .await;
        }

        info!(session_id, "Received Discord message from user {}", user_id);

        // Handle cancel/stop commands - these bypass the queue.
        let text_lower = text.to_ascii_lowercase();
        if text_lower == "cancel" || text_lower == "stop" || text_lower == "abort" {
            if user_role != UserRole::Owner {
                let _ = msg
                    .channel_id
                    .say(
                        &ctx.http,
                        "Only the owner can cancel running work in this session.",
                    )
                    .await;
                return;
            }

            let cancelled = self
                .task_registry
                .cancel_running_for_session(&session_id)
                .await;
            let queue_cleared = self.task_registry.queue_len(&session_id).await;
            self.task_registry.clear_queue(&session_id).await;
            let cancelled_goals = self
                .agent
                .cancel_active_goals_for_session(&session_id)
                .await;

            if cancelled.is_empty() {
                if cancelled_goals.is_empty() {
                    let _ = msg
                        .channel_id
                        .say(&ctx.http, "No running task to cancel.")
                        .await;
                } else if cancelled_goals.len() == 1 {
                    let _ = msg
                        .channel_id
                        .say(
                            &ctx.http,
                            format!("⏹️ Cancelled goal: {}", cancelled_goals[0]),
                        )
                        .await;
                } else {
                    let response = format!(
                        "⏹️ Cancelled {} goals:\n{}",
                        cancelled_goals.len(),
                        cancelled_goals
                            .iter()
                            .map(|d| format!("- {}", d))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                    let _ = msg.channel_id.say(&ctx.http, response).await;
                }
            } else {
                let desc = cancelled
                    .first()
                    .map(|(_, d)| d.as_str())
                    .unwrap_or("unknown");
                let mut response = format!("⏹️ Cancelled: {}", desc);
                if queue_cleared > 0 {
                    response.push_str(&format!(" (+{} queued messages cleared)", queue_cleared));
                }
                if !cancelled_goals.is_empty() {
                    response.push_str(&format!(" (+{} goal(s) cancelled)", cancelled_goals.len()));
                }
                let _ = msg.channel_id.say(&ctx.http, response).await;
            }
            return;
        }

        // Check if a task is already running for this session - if so, queue this message.
        if self.task_registry.has_running_task(&session_id).await {
            let daemon_uptime = self.started_at.elapsed();
            if should_ignore_lightweight_interjection(&text, daemon_uptime) {
                let current_task = self
                    .task_registry
                    .get_running_task_description(&session_id)
                    .await
                    .unwrap_or_else(|| "processing".to_string());
                let _ = msg
                    .channel_id
                    .say(
                        &ctx.http,
                        format!(
                            "⏳ Still working on: {}. I ignored that short check-in. \
                             Send `cancel` to stop the current task.",
                            current_task
                        ),
                    )
                    .await;
                return;
            }

            let queue_result = self.task_registry.queue_message(&session_id, &text).await;
            match queue_result {
                Some(queue_pos) => {
                    let current_task = self
                        .task_registry
                        .get_running_task_description(&session_id)
                        .await
                        .unwrap_or_else(|| "processing".to_string());
                    let preview: String = text.chars().take(50).collect();
                    let suffix = if text.len() > 50 { "..." } else { "" };
                    let _ = msg
                        .channel_id
                        .say(
                            &ctx.http,
                            format!(
                                "📥 Queued ({}): \"{}{}\" | Currently: {}",
                                queue_pos, preview, suffix, current_task
                            ),
                        )
                        .await;
                }
                None => {
                    info!(session_id, "Dropped duplicate queued message");
                }
            }
            return;
        }

        // Register task for tracking
        let description: String = text.chars().take(80).collect();
        let (task_id, cancel_token) = self.task_registry.register(&session_id, &description).await;

        // Build channel context from Discord message
        let author_name = msg.author.name.clone();
        let author_id = msg.author.id;
        let channel_ctx = if msg.guild_id.is_some() {
            ChannelContext {
                visibility: ChannelVisibility::Public,
                platform: "discord".to_string(),
                channel_name: None,
                channel_id: Some(format!("discord:{}", msg.channel_id)),
                sender_name: Some(author_name),
                sender_id: Some(format!("discord:{}", author_id)),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            }
        } else {
            ChannelContext {
                visibility: ChannelVisibility::Private,
                platform: "discord".to_string(),
                channel_name: None,
                channel_id: Some(format!("discord:dm:{}", author_id)),
                sender_name: Some(author_name),
                sender_id: Some(format!("discord:{}", author_id)),
                channel_member_names: vec![],
                user_id_map: std::collections::HashMap::new(),
                trusted: false,
            }
        };

        // Create heartbeat for watchdog — agent bumps this on every activity point.
        let heartbeat = Arc::new(AtomicU64::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        ));
        let stale_threshold_secs = self.watchdog_stale_threshold_secs;
        let is_dm = msg.guild_id.is_none();

        // Start typing indicator for this task.
        let typing_cancel = tokio_util::sync::CancellationToken::new();
        Self::spawn_typing_indicator(
            msg.channel_id,
            ctx.http.clone(),
            typing_cancel.clone(),
            heartbeat.clone(),
            stale_threshold_secs,
        );

        // Status updates for this task.
        let (status_tx, status_rx) = tokio::sync::mpsc::channel::<StatusUpdate>(16);
        let status_task =
            Self::spawn_status_task(status_rx, ctx.http.clone(), msg.channel_id, is_dm);

        self.task_registry
            .set_typing_cancel(task_id, typing_cancel.clone())
            .await;

        let registry = Arc::clone(&self.task_registry);

        let agent = Arc::clone(&self.agent);
        let channel_id = msg.channel_id;
        let http = ctx.http.clone();
        tokio::spawn(async move {
            let mut current_text = text;
            let mut current_task_id = task_id;
            let mut current_cancel_token = cancel_token;
            let mut current_status_tx = status_tx;
            let mut current_status_task = status_task;
            let mut current_heartbeat = heartbeat;
            let mut current_typing_cancel = typing_cancel;

            loop {
                let result = tokio::select! {
                    r = agent.handle_message(
                        &session_id,
                        &current_text,
                        Some(current_status_tx.clone()),
                        user_role,
                        channel_ctx.clone(),
                        Some(current_heartbeat.clone()),
                    ) => r,
                    _ = current_cancel_token.cancelled() => Err(anyhow::anyhow!("Task cancelled")),
                    stale_mins = super::wait_for_stale_heartbeat(current_heartbeat.clone(), stale_threshold_secs, 8), if stale_threshold_secs > 0 => {
                        Err(anyhow::anyhow!(
                            "Task auto-cancelled due to inactivity ({} minute{} without progress).",
                            stale_mins,
                            if stale_mins == 1 { "" } else { "s" }
                        ))
                    },
                };
                current_typing_cancel.cancel();
                current_status_task.abort();

                let mut task_error: Option<String> = None;
                match result {
                    Ok(reply) => {
                        let chunks = split_message(&reply, 2000);
                        for chunk in &chunks {
                            if let Err(e) = channel_id.say(&http, chunk).await {
                                warn!("Failed to send Discord message: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        if error_msg == "Task cancelled" {
                            registry.fail(current_task_id, &error_msg).await;
                            info!("Task #{} cancelled", current_task_id);
                            return;
                        }
                        task_error = Some(error_msg.clone());
                        if error_msg.starts_with("Task auto-cancelled due to inactivity") {
                            info!("Task #{} auto-cancelled by stale watchdog", current_task_id);
                            let _ = channel_id.say(&http, format!("⚠️ {}", error_msg)).await;
                        } else {
                            warn!("Agent error: {}", e);
                            let _ = channel_id.say(&http, format!("Error: {}", e)).await;
                        }
                    }
                }

                if let Some(ref err) = task_error {
                    registry.fail(current_task_id, err).await;
                } else {
                    registry.complete(current_task_id).await;
                }

                if let Some(queued) = registry.pop_queued_message(&session_id).await {
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    info!(
                        session_id,
                        "Processing queued message: {}",
                        queued.text.chars().take(50).collect::<String>()
                    );
                    let preview: String = queued.text.chars().take(50).collect();
                    let suffix = if queued.text.len() > 50 { "..." } else { "" };
                    let _ = channel_id
                        .say(
                            &http,
                            format!("▶️ Processing queued: \"{}{}\"", preview, suffix),
                        )
                        .await;

                    current_text = queued.text;
                    let desc: String = current_text.chars().take(80).collect();
                    let (new_task_id, new_cancel_token) =
                        registry.register(&session_id, &desc).await;
                    current_task_id = new_task_id;
                    current_cancel_token = new_cancel_token;

                    let (new_status_tx, new_status_rx) =
                        tokio::sync::mpsc::channel::<StatusUpdate>(16);
                    current_status_tx = new_status_tx;
                    current_status_task =
                        Self::spawn_status_task(new_status_rx, http.clone(), channel_id, is_dm);

                    current_heartbeat = Arc::new(AtomicU64::new(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    ));
                    current_typing_cancel = tokio_util::sync::CancellationToken::new();
                    Self::spawn_typing_indicator(
                        channel_id,
                        http.clone(),
                        current_typing_cancel.clone(),
                        current_heartbeat.clone(),
                        stale_threshold_secs,
                    );
                    registry
                        .set_typing_cancel(current_task_id, current_typing_cancel.clone())
                        .await;
                } else {
                    break;
                }
            }
        });
    }

    /// Handle text commands (e.g. /model, /help)
    async fn handle_text_command(&self, ctx: &Context, msg: &SerenityMessage, text: &str) {
        let reply = self
            .dispatch_command(text, &self.session_id_from_message(msg))
            .await;
        let chunks = split_message(&reply, 2000);
        for chunk in &chunks {
            let _ = msg.channel_id.say(&ctx.http, chunk).await;
        }
    }

    /// Handle a slash command interaction.
    async fn handle_slash_command(&self, ctx: &Context, command: &CommandInteraction) {
        let user_id = command.user.id.get();
        let is_dm = command.guild_id.is_none();
        // In guild channels, let all server members through (Guest role).
        // Auth gate only applies to DMs.
        if is_dm && !self.is_authorized(user_id) {
            let _ = command
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Unauthorized.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }

        // Defer to buy time for longer operations
        let _ = command
            .create_response(
                &ctx.http,
                CreateInteractionResponse::Defer(CreateInteractionResponseMessage::new()),
            )
            .await;

        let base_session = if command.guild_id.is_some() {
            format!("discord:ch:{}", command.channel_id)
        } else {
            format!("discord:dm:{}", command.user.id)
        };
        let session_id = self.session_id(&base_session);

        let arg = command
            .data
            .options
            .first()
            .and_then(|o| o.value.as_str())
            .unwrap_or("");

        let cmd_text = format!("/{} {}", command.data.name, arg);
        let reply = self.dispatch_command(cmd_text.trim(), &session_id).await;

        let _ = command
            .edit_response(&ctx.http, EditInteractionResponse::new().content(reply))
            .await;
    }

    /// Dispatch a command string to the appropriate handler and return the reply text.
    async fn dispatch_command(&self, text: &str, session_id: &str) -> String {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd = parts[0];
        let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

        match cmd {
            "/model" => {
                if arg.is_empty() {
                    let current = self.agent.current_model().await;
                    format!("Current model: {}\n\nUsage: /model <model-name>", current)
                } else {
                    self.agent.set_model(arg.to_string()).await;
                    format!(
                        "Model switched to: {}\nAuto-routing disabled. Use /auto to re-enable.",
                        arg
                    )
                }
            }
            "/models" => match self.agent.list_models().await {
                Ok(models) => {
                    if models.is_empty() {
                        "No models found from provider.".to_string()
                    } else {
                        let current = self.agent.current_model().await;
                        let list: Vec<String> = models
                            .iter()
                            .map(|m| {
                                if *m == current {
                                    format!("• {} (active)", m)
                                } else {
                                    format!("• {}", m)
                                }
                            })
                            .collect();
                        format!("Available models:\n{}", list.join("\n"))
                    }
                }
                Err(e) => format!("Failed to list models: {}", e),
            },
            "/auto" => {
                self.agent.clear_model_override().await;
                "Auto-routing re-enabled. Model will be selected automatically based on query complexity.".to_string()
            }
            "/reload" => match AppConfig::load(&self.config_path) {
                Ok(new_config) => match self.agent.reload_provider(&new_config).await {
                    Ok(status) => format!("Config reloaded. {}", status),
                    Err(e) => format!("Provider reload failed: {}", e),
                },
                Err(e) => {
                    let backup = self.config_path.with_extension("toml.bak");
                    if backup.exists() {
                        if std::fs::copy(&backup, &self.config_path).is_ok() {
                            format!("Config reload failed: {}\n\nAuto-restored from backup.", e)
                        } else {
                            format!("Config reload failed: {}\n\nBackup restore also failed.", e)
                        }
                    } else {
                        format!("Config reload failed: {}\n\nNo backup available.", e)
                    }
                }
            },
            "/tasks" => {
                let entries = self.task_registry.list_for_session(session_id).await;
                if entries.is_empty() {
                    "No tasks found.".to_string()
                } else {
                    let lines: Vec<String> = entries
                        .iter()
                        .map(|e| {
                            let elapsed = match e.finished_at {
                                Some(fin) => {
                                    let d = fin - e.started_at;
                                    format!("{}s", d.num_seconds())
                                }
                                None => {
                                    let d = Utc::now() - e.started_at;
                                    format!("{}s elapsed", d.num_seconds())
                                }
                            };
                            format!("#{} [{}] {} ({})", e.id, e.status, e.description, elapsed)
                        })
                        .collect();
                    lines.join("\n")
                }
            }
            "/cancel" => {
                if arg.is_empty() {
                    "Usage: /cancel <task-id>".to_string()
                } else {
                    match arg.parse::<u64>() {
                        Ok(task_id) => {
                            if self.task_registry.cancel(task_id).await {
                                format!("Task #{} cancelled.", task_id)
                            } else {
                                format!("Task #{} not found or not running.", task_id)
                            }
                        }
                        Err(_) => "Invalid task ID. Usage: /cancel <task-id>".to_string(),
                    }
                }
            }
            "/clear" => match self.agent.clear_session(session_id).await {
                Ok(_) => "Context cleared. Starting fresh.".to_string(),
                Err(e) => format!("Failed to clear context: {}", e),
            },
            "/cost" => self.handle_cost_command().await,
            "/help" | "/start" => build_help_text(false, false, "/"),
            _ => format!(
                "Unknown command: {}\nType /help for available commands.",
                cmd
            ),
        }
    }

    async fn handle_cost_command(&self) -> String {
        let now = Utc::now();
        let since_24h = (now - chrono::Duration::hours(24))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();
        let since_7d = (now - chrono::Duration::days(7))
            .format("%Y-%m-%d %H:%M:%S")
            .to_string();

        let records_24h = match self.state.get_token_usage_since(&since_24h).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };
        let records_7d = match self.state.get_token_usage_since(&since_7d).await {
            Ok(r) => r,
            Err(e) => return format!("Failed to query token usage: {}", e),
        };

        let (input_24h, output_24h) = records_24h.iter().fold((0i64, 0i64), |(i, o), r| {
            (i + r.input_tokens, o + r.output_tokens)
        });
        let (input_7d, output_7d) = records_7d.iter().fold((0i64, 0i64), |(i, o), r| {
            (i + r.input_tokens, o + r.output_tokens)
        });

        let mut model_totals: HashMap<&str, i64> = HashMap::new();
        for r in &records_7d {
            *model_totals.entry(&r.model).or_insert(0) += r.input_tokens + r.output_tokens;
        }
        let mut models_sorted: Vec<(&&str, &i64)> = model_totals.iter().collect();
        models_sorted.sort_by(|a, b| b.1.cmp(a.1));

        let mut reply = format!(
            "Token usage (last 24h):\n  Input:  {} tokens\n  Output: {} tokens\n\n\
             Token usage (last 7d):\n  Input:  {} tokens\n  Output: {} tokens",
            format_number(input_24h),
            format_number(output_24h),
            format_number(input_7d),
            format_number(output_7d),
        );

        if !models_sorted.is_empty() {
            reply.push_str("\n\nTop models (7d):");
            for (model, total) in models_sorted.iter().take(5) {
                reply.push_str(&format!("\n  {}: {} tokens", model, format_number(**total)));
            }
        }

        reply
    }

    /// Handle file attachments from a Discord message.
    async fn handle_file_message(
        &self,
        _ctx: &Context,
        msg: &SerenityMessage,
    ) -> anyhow::Result<String> {
        let mut contexts = Vec::new();

        for attachment in &msg.attachments {
            let max_bytes = self.max_file_size_mb * 1_048_576;
            if attachment.size as u64 > max_bytes {
                anyhow::bail!(
                    "File too large ({:.1} MB). Maximum is {} MB.",
                    attachment.size as f64 / 1_048_576.0,
                    self.max_file_size_mb
                );
            }

            let bytes = attachment.download().await?;

            let filename = &attachment.filename;
            let mime_type = attachment
                .content_type
                .as_deref()
                .unwrap_or("application/octet-stream");

            let sanitized = sanitize_filename(filename);
            let uuid_prefix = uuid::Uuid::new_v4().to_string()[..8].to_string();
            let dest_name = format!("{}_{}", uuid_prefix, sanitized);
            let dest_path = self.inbox_dir.join(&dest_name);

            std::fs::create_dir_all(&self.inbox_dir)?;
            std::fs::write(&dest_path, &bytes)?;

            info!(
                file = %dest_path.display(),
                size = bytes.len(),
                mime = %mime_type,
                "Saved inbound Discord file"
            );

            let size_display = if bytes.len() > 1_048_576 {
                format!("{:.1} MB", bytes.len() as f64 / 1_048_576.0)
            } else {
                format!("{:.0} KB", bytes.len() as f64 / 1024.0)
            };

            contexts.push(format!(
                "[File received: {} ({}, {})\nSaved to: {}]",
                sanitized,
                size_display,
                mime_type,
                dest_path.display()
            ));
        }

        Ok(contexts.join("\n"))
    }

    /// Handle a button interaction (approval flow).
    async fn handle_component_interaction(
        &self,
        ctx: &Context,
        interaction: &ComponentInteraction,
    ) {
        let user_id = interaction.user.id.get();
        // Approval buttons require owner or allowed-user authorization
        // (guild membership alone is not sufficient for approving actions).
        if !self.is_authorized(user_id) && !self.owner_user_ids.contains(&user_id) {
            let _ = interaction
                .create_response(
                    &ctx.http,
                    CreateInteractionResponse::Message(
                        CreateInteractionResponseMessage::new()
                            .content("Unauthorized.")
                            .ephemeral(true),
                    ),
                )
                .await;
            return;
        }

        let data = &interaction.data.custom_id;
        let parts: Vec<&str> = data.splitn(3, ':').collect();
        if parts.len() != 3 || parts[0] != "approve" {
            return;
        }

        let action = parts[1];
        let approval_id = parts[2];

        let response = match action {
            "once" => ApprovalResponse::AllowOnce,
            "session" => ApprovalResponse::AllowSession,
            "always" => ApprovalResponse::AllowAlways,
            "deny" => ApprovalResponse::Deny,
            _ => return,
        };

        let label = match &response {
            ApprovalResponse::AllowOnce => "Allowed (once)",
            ApprovalResponse::AllowSession => "Allowed (this session)",
            ApprovalResponse::AllowAlways => "Allowed (always)",
            ApprovalResponse::Deny => "Denied",
        };

        // Send the response
        {
            let mut pending = self.pending_approvals.lock().await;
            if let Some(tx) = pending.remove(approval_id) {
                let _ = tx.send(response);
            } else {
                warn!(approval_id, "Stale approval callback (no pending request)");
            }
        }

        // Acknowledge the interaction and update the message to remove buttons
        let original_content = interaction.message.content.clone();
        let updated = format!("{} — {}", original_content, label);

        let _ = interaction
            .create_response(
                &ctx.http,
                CreateInteractionResponse::UpdateMessage(
                    CreateInteractionResponseMessage::new()
                        .content(updated)
                        .components(vec![]), // Remove buttons
                ),
            )
            .await;
    }

    /// Register slash commands with Discord.
    async fn register_commands(&self, ctx: &Context) {
        let commands = vec![
            CreateCommand::new("model")
                .description("Show or switch the current LLM model")
                .add_option(
                    serenity::builder::CreateCommandOption::new(
                        serenity::all::CommandOptionType::String,
                        "name",
                        "Model name to switch to",
                    )
                    .required(false),
                ),
            CreateCommand::new("models").description("List available models from provider"),
            CreateCommand::new("auto").description("Re-enable automatic model routing"),
            CreateCommand::new("reload").description("Reload config.toml"),
            CreateCommand::new("tasks").description("List running and recent tasks"),
            CreateCommand::new("cancel")
                .description("Cancel a running task")
                .add_option(
                    serenity::builder::CreateCommandOption::new(
                        serenity::all::CommandOptionType::String,
                        "id",
                        "Task ID to cancel",
                    )
                    .required(true),
                ),
            CreateCommand::new("clear").description("Clear conversation context and start fresh"),
            CreateCommand::new("cost").description("Show token usage statistics"),
            CreateCommand::new("help").description("Show available commands"),
        ];

        if let Some(guild_id) = self.guild_id {
            // Guild-scoped: instant propagation
            let guild = GuildId::new(guild_id);
            if let Err(e) = guild.set_commands(&ctx.http, commands).await {
                warn!("Failed to register guild slash commands: {}", e);
            } else {
                info!(guild_id, "Registered guild slash commands");
            }
        } else {
            // Global: ~1hr propagation
            if let Err(e) = Command::set_global_commands(&ctx.http, commands).await {
                warn!("Failed to register global slash commands: {}", e);
            } else {
                info!("Registered global slash commands");
            }
        }
    }
}

#[async_trait]
impl Channel for DiscordChannel {
    fn name(&self) -> String {
        self.channel_name()
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            markdown: true,
            inline_buttons: true,
            media: true,
            max_message_len: 2000,
        }
    }

    async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        let http = self.get_http().await?;
        let channel_id = self.resolve_channel_id(session_id).await?;
        let mut first_err: Option<anyhow::Error> = None;
        for chunk in split_message(text, 2000) {
            if let Err(e) = channel_id.say(&http, &chunk).await {
                warn!("Failed to send Discord message: {}", e);
                if first_err.is_none() {
                    first_err = Some(anyhow::anyhow!("Failed to send Discord message: {}", e));
                }
            }
        }
        if let Some(err) = first_err {
            return Err(err);
        }
        Ok(())
    }

    async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        let http = self.get_http().await?;
        let channel_id = self.resolve_channel_id(session_id).await?;
        match &media.kind {
            MediaKind::Photo { data } => {
                let attachment = CreateAttachment::bytes(data.clone(), "screenshot.png");
                let mut msg = CreateMessage::new();
                if !media.caption.is_empty() {
                    msg = msg.content(&media.caption);
                }
                msg = msg.add_file(attachment);
                channel_id
                    .send_message(&http, msg)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send photo: {}", e))?;
            }
            MediaKind::Document {
                file_path,
                filename,
            } => {
                let data = tokio::fs::read(file_path).await?;
                let attachment = CreateAttachment::bytes(data, filename.as_str());
                let mut msg = CreateMessage::new();
                if !media.caption.is_empty() {
                    msg = msg.content(&media.caption);
                }
                msg = msg.add_file(attachment);
                channel_id
                    .send_message(&http, msg)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send document: {}", e))?;
            }
        }
        Ok(())
    }

    async fn request_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse> {
        let http = self.get_http().await?;
        let channel_id = self.resolve_channel_id(session_id).await?;

        let approval_id = uuid::Uuid::new_v4().to_string();
        let short_id = &approval_id[..8];

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = self.pending_approvals.lock().await;
            pending.insert(approval_id.clone(), response_tx);
            info!(
                approval_id = %short_id,
                pending_count = pending.len(),
                risk = %risk_level,
                mode = %permission_mode,
                "Stored pending Discord approval"
            );
        }

        // Determine which buttons to show based on permission_mode and risk_level
        let use_session_button = match permission_mode {
            PermissionMode::Cautious => true,
            PermissionMode::Default => risk_level >= RiskLevel::Critical,
            PermissionMode::Yolo => false,
        };

        let buttons = if use_session_button {
            CreateActionRow::Buttons(vec![
                CreateButton::new(format!("approve:once:{}", approval_id))
                    .label("Allow Once")
                    .style(ButtonStyle::Primary),
                CreateButton::new(format!("approve:session:{}", approval_id))
                    .label("Allow Session")
                    .style(ButtonStyle::Success),
                CreateButton::new(format!("approve:deny:{}", approval_id))
                    .label("Deny")
                    .style(ButtonStyle::Danger),
            ])
        } else {
            CreateActionRow::Buttons(vec![
                CreateButton::new(format!("approve:once:{}", approval_id))
                    .label("Allow Once")
                    .style(ButtonStyle::Primary),
                CreateButton::new(format!("approve:always:{}", approval_id))
                    .label("Allow Always")
                    .style(ButtonStyle::Success),
                CreateButton::new(format!("approve:deny:{}", approval_id))
                    .label("Deny")
                    .style(ButtonStyle::Danger),
            ])
        };

        // Build message with risk info
        let (risk_icon, risk_label) = match risk_level {
            RiskLevel::Safe => ("ℹ️", "New command"),
            RiskLevel::Medium => ("⚠️", "Medium risk"),
            RiskLevel::High => ("🔶", "High risk"),
            RiskLevel::Critical => ("🚨", "Critical risk"),
        };

        let mut text = format!("{} **{}**\n\n```\n{}\n```", risk_icon, risk_label, command);

        if !warnings.is_empty() {
            text.push('\n');
            for warning in warnings {
                text.push_str(&format!("\n• {}", warning));
            }
        }

        // Add explanation based on which button is shown
        if use_session_button {
            text.push_str("\n\n*\"Allow Session\" approves this command type until restart.*");
        } else {
            text.push_str("\n\n*\"Allow Always\" permanently approves this command type.*");
        }

        text.push_str(&format!("\n\n*[{}]*", short_id));

        let msg = CreateMessage::new()
            .content(&text)
            .components(vec![buttons]);

        match channel_id.send_message(&http, msg).await {
            Ok(_) => {
                info!(approval_id = %short_id, "Approval message sent to Discord");
            }
            Err(e) => {
                warn!("Failed to send Discord approval request: {}", e);
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                return Ok(ApprovalResponse::Deny);
            }
        }

        // Wait for response with 5-minute timeout to prevent memory leak
        info!(approval_id = %short_id, "Waiting for user approval response...");
        match tokio::time::timeout(Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                warn!(approval_id = %short_id, "Approval channel closed");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                warn!(approval_id = %short_id, "Approval timed out after 5 minutes");
                let mut pending = self.pending_approvals.lock().await;
                pending.remove(&approval_id);
                Ok(ApprovalResponse::Deny)
            }
        }
    }
}

/// Serenity event handler that bridges to our DiscordChannel methods.
struct DiscordHandler {
    channel: Arc<DiscordChannel>,
}

#[async_trait]
impl EventHandler for DiscordHandler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        let username = ready.user.name.clone();
        info!(username = %username, "Discord bot connected");
        self.channel.set_bot_username(username);
        self.channel.register_commands(&ctx).await;
    }

    async fn message(&self, ctx: Context, msg: SerenityMessage) {
        self.channel.handle_message_event(&ctx, msg).await;
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        match interaction {
            Interaction::Command(command) => {
                self.channel.handle_slash_command(&ctx, &command).await;
            }
            Interaction::Component(component) => {
                self.channel
                    .handle_component_interaction(&ctx, &component)
                    .await;
            }
            _ => {}
        }
    }
}

/// Spawn a DiscordChannel in a background task.
/// This is a separate function to avoid async type inference cycles.
pub fn spawn_discord_channel(channel: Arc<DiscordChannel>) {
    tokio::spawn(async move {
        channel.start_with_retry().await;
    });
}
