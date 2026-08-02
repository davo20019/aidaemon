use std::collections::HashMap;
use std::sync::{Arc, Weak};

use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn};

use crate::agent::Agent;
use crate::config::QueuePolicyConfig;
use crate::queue_policy::{should_shed_due_to_overload, SessionFairnessBudget};
use crate::queue_telemetry::{QueuePressure, QueueTelemetry};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::traits::Channel;
use crate::types::{ApprovalKind, ApprovalResponse, MediaKind, MediaMessage};

/// Shared map of session_id → channel name.
/// Written by channels when they receive incoming messages,
/// read by the hub to route outbound messages (approvals, media, notifications).
pub type SessionMap = Arc<RwLock<HashMap<String, String>>>;

/// Cancellation-safe lifetime for an internal child-session route.
///
/// Child agent futures can be aborted while awaiting an LLM or a tool. Keeping
/// cleanup in `Drop` prevents those cancellations from leaving stale routes in
/// the hub indefinitely. The parent value is checked before removal so an old
/// guard can never erase a newer route for the same child identifier.
pub(crate) struct SessionRouteGuard {
    hub: Weak<ChannelHub>,
    child_session: String,
    parent_session: String,
}

impl Drop for SessionRouteGuard {
    fn drop(&mut self) {
        let Some(hub) = self.hub.upgrade() else {
            return;
        };
        if let Ok(mut routes) = hub.session_routes.try_write() {
            if routes.get(&self.child_session) == Some(&self.parent_session) {
                routes.remove(&self.child_session);
            }
            return;
        }

        let child_session = self.child_session.clone();
        let parent_session = self.parent_session.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                hub.unregister_session_route(&child_session, &parent_session)
                    .await;
            });
        }
    }
}

/// Central router for outbound messages across all channels.
///
/// The hub routes approval requests, media, and notifications to the
/// correct channel based on which channel originated the session.
/// Unknown sessions are refused (returns None) to prevent cross-channel
/// privacy leaks.
pub struct ChannelHub {
    /// Registered channels. Uses RwLock to support dynamic registration.
    channels: RwLock<Vec<Arc<dyn Channel>>>,
    session_map: SessionMap,
    /// Ephemeral runtime-owned routes from a specialist session to the human
    /// conversation that spawned it. This lets approvals use the real parent
    /// chat without making internal worker IDs externally addressable.
    session_routes: RwLock<HashMap<String, String>>,
    queue_telemetry: Option<Arc<QueueTelemetry>>,
    queue_policy: Option<QueuePolicyConfig>,
    delivery_note_agent: Option<Arc<Agent>>,
    /// Best-effort duplicate suppression for rapid-fire identical messages.
    /// Keyed by session_id.
    last_sent_text: RwLock<HashMap<String, (String, tokio::time::Instant)>>,
    /// Short-window de-duplication of identical document deliveries to the same
    /// session. Without it, a file the background notifier delivers
    /// deterministically (Task 4) is sent a SECOND time when the model also calls
    /// `send_file` for it (e.g. the user's request ends with "…then send me the
    /// file"). Keyed by `(session_id, canonical file path)`; genuinely different
    /// files (different paths) are never suppressed. Photos are never deduped.
    recent_media_deliveries: RwLock<HashMap<String, tokio::time::Instant>>,
    /// Per-session editable message id of a delivered background-handoff reply
    /// ("⏳ Still on it — …"). The terminal notifier consumes it (take) so the
    /// completion ping EDITS that message in place instead of stacking a new
    /// one. Last writer wins per session; take() removes the entry, so a
    /// second concurrent background command degrades to a fresh ping message.
    background_status_surfaces: RwLock<HashMap<String, String>>,
}

impl ChannelHub {
    pub fn new(channels: Vec<Arc<dyn Channel>>, session_map: SessionMap) -> Self {
        Self {
            channels: RwLock::new(channels),
            session_map,
            session_routes: RwLock::new(HashMap::new()),
            queue_telemetry: None,
            queue_policy: None,
            delivery_note_agent: None,
            last_sent_text: RwLock::new(HashMap::new()),
            recent_media_deliveries: RwLock::new(HashMap::new()),
            background_status_surfaces: RwLock::new(HashMap::new()),
        }
    }

    /// Register the editable message id of a just-delivered background-handoff
    /// reply for this session. See `background_status_surfaces`.
    pub async fn register_background_status_surface(&self, session_id: &str, message_id: &str) {
        self.background_status_surfaces
            .write()
            .await
            .insert(session_id.to_string(), message_id.to_string());
    }

    /// Take (and remove) the registered background-handoff message id for this
    /// session, if any.
    pub async fn take_background_status_surface(&self, session_id: &str) -> Option<String> {
        self.background_status_surfaces
            .write()
            .await
            .remove(session_id)
    }

    /// How long a delivered document suppresses an identical re-delivery.
    const MEDIA_DEDUPE_WINDOW: std::time::Duration = std::time::Duration::from_secs(120);

    /// De-dup key for a media message, or `None` for media that should never be
    /// deduped (in-memory photos have no stable identity).
    fn media_dedupe_key(session_id: &str, media: &MediaMessage) -> Option<String> {
        match &media.kind {
            MediaKind::Document { file_path, .. } => Some(format!("{session_id}\u{1}{file_path}")),
            MediaKind::Photo { .. } => None,
        }
    }

    /// Claim a document delivery within the dedupe window. Returns `true` if this
    /// is the first delivery (caller should proceed) or the media is unkeyable;
    /// `false` if an identical document was already delivered to this session
    /// recently (caller should skip). On a failed send, call
    /// [`release_media_delivery`](Self::release_media_delivery) so a retry isn't
    /// wrongly suppressed.
    async fn claim_media_delivery(&self, session_id: &str, media: &MediaMessage) -> bool {
        let Some(key) = Self::media_dedupe_key(session_id, media) else {
            return true;
        };
        let now = tokio::time::Instant::now();
        let mut log = self.recent_media_deliveries.write().await;
        log.retain(|_, t| now.duration_since(*t) < Self::MEDIA_DEDUPE_WINDOW);
        if log.contains_key(&key) {
            return false;
        }
        log.insert(key, now);
        true
    }

    /// Release a previously-claimed document delivery (used when the actual send
    /// failed, so a later retry of the same file is not suppressed).
    async fn release_media_delivery(&self, session_id: &str, media: &MediaMessage) {
        if let Some(key) = Self::media_dedupe_key(session_id, media) {
            self.recent_media_deliveries.write().await.remove(&key);
        }
    }

    pub fn with_queue_telemetry(mut self, queue_telemetry: Arc<QueueTelemetry>) -> Self {
        self.queue_telemetry = Some(queue_telemetry);
        self
    }

    pub fn with_queue_policy(mut self, queue_policy: QueuePolicyConfig) -> Self {
        self.queue_policy = Some(queue_policy);
        self
    }

    pub fn with_delivery_note_agent(mut self, agent: Arc<Agent>) -> Self {
        self.delivery_note_agent = Some(agent);
        self
    }

    async fn record_media_delivery_note(&self, session_id: &str, media: &MediaMessage) {
        let Some(agent) = self.delivery_note_agent.as_ref() else {
            return;
        };
        let MediaKind::Document {
            file_path,
            filename,
        } = &media.kind
        else {
            return;
        };
        let summary = format!(
            "Delivery note: I sent the attachment {} in chat. Local copy: {}",
            filename, file_path
        );
        if let Err(err) = agent
            .record_auxiliary_assistant_note(session_id, &summary)
            .await
        {
            warn!(
                session_id,
                error = %err,
                "Failed to persist outbound media delivery summary"
            );
        }
    }

    /// Register a new channel dynamically.
    /// Returns the channel name after registration.
    #[allow(dead_code)]
    pub async fn register_channel(&self, channel: Arc<dyn Channel>) -> String {
        let name = channel.name();
        let mut channels = match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.channels.write(),
        )
        .await
        {
            Ok(guard) => guard,
            Err(_) => {
                warn!(channel = %name, "Timed out acquiring channels write lock while registering channel");
                return name;
            }
        };
        channels.push(channel);
        info!(channel = %name, total = channels.len(), "Registered new channel");
        name
    }

    /// Get a reference to the shared session map.
    #[allow(dead_code)]
    pub fn session_map(&self) -> &SessionMap {
        &self.session_map
    }

    /// Route an internal child session through its originating human session.
    /// Only the agent runtime calls this; tool/model arguments cannot register
    /// routes. The parent may itself be a child, so resolution follows a short
    /// bounded chain.
    pub(crate) async fn register_session_route(
        self: &Arc<Self>,
        child_session: &str,
        parent_session: &str,
    ) -> Option<SessionRouteGuard> {
        if child_session.is_empty() || parent_session.is_empty() || child_session == parent_session
        {
            return None;
        }
        self.session_routes
            .write()
            .await
            .insert(child_session.to_string(), parent_session.to_string());
        Some(SessionRouteGuard {
            hub: Arc::downgrade(self),
            child_session: child_session.to_string(),
            parent_session: parent_session.to_string(),
        })
    }

    async fn unregister_session_route(&self, child_session: &str, parent_session: &str) {
        let mut routes = self.session_routes.write().await;
        if routes.get(child_session).map(String::as_str) == Some(parent_session) {
            routes.remove(child_session);
        }
    }

    async fn routed_session_id(&self, session_id: &str) -> String {
        let routes = self.session_routes.read().await;
        let mut current = session_id.to_string();
        for _ in 0..8 {
            let Some(parent) = routes.get(&current) else {
                return current;
            };
            if parent == &current {
                break;
            }
            current = parent.clone();
        }
        warn!(session_id, "Session route chain was cyclic or too deep");
        session_id.to_string()
    }

    /// Resolve both the channel and the externally addressable session. Every
    /// outbound operation must use the returned session, not the internal child
    /// identifier that was supplied by the caller.
    async fn routed_channel(&self, session_id: &str) -> Option<(Arc<dyn Channel>, String)> {
        let routed_session = self.routed_session_id(session_id).await;
        let map =
            match tokio::time::timeout(std::time::Duration::from_secs(2), self.session_map.read())
                .await
            {
                Ok(guard) => guard,
                Err(_) => {
                    warn!(
                        session_id,
                        "Timed out acquiring session_map read lock while routing session"
                    );
                    return None;
                }
            };
        let channels =
            match tokio::time::timeout(std::time::Duration::from_secs(2), self.channels.read())
                .await
            {
                Ok(guard) => guard,
                Err(_) => {
                    warn!(
                        session_id,
                        "Timed out acquiring channels read lock while routing session"
                    );
                    return None;
                }
            };
        if let Some(channel_name) = map.get(&routed_session) {
            if let Some(ch) = channels.iter().find(|c| &c.name() == channel_name) {
                return Some((ch.clone(), routed_session));
            }
        }
        None
    }

    /// Find the channel that owns a session.
    /// Returns None for unknown sessions to prevent cross-channel privacy leaks.
    #[cfg(test)]
    async fn channel_for_session(&self, session_id: &str) -> Option<Arc<dyn Channel>> {
        self.routed_channel(session_id)
            .await
            .map(|(channel, _)| channel)
    }

    /// Request approval through a channel that supports inline buttons.
    ///
    /// Used for UX flows that require button consistency (for example scheduled
    /// goal confirmation) while preserving text fallback in non-inline channels.
    pub async fn request_inline_approval(
        &self,
        session_id: &str,
        command: &str,
        risk_level: RiskLevel,
        warnings: &[String],
        permission_mode: PermissionMode,
    ) -> anyhow::Result<ApprovalResponse> {
        let (channel, routed_session) = self
            .routed_channel(session_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("No channel found for session {}", session_id))?;
        if !channel.capabilities().inline_buttons {
            anyhow::bail!(
                "Channel {} does not support inline approval buttons",
                channel.name()
            );
        }
        channel
            .request_approval(
                &routed_session,
                command,
                risk_level,
                warnings,
                permission_mode,
                false,
            )
            .await
    }

    /// Route approval requests from tools to the appropriate channel.
    ///
    /// Each approval is handled in its own task so the listener doesn't
    /// block while waiting for the user to respond.
    pub async fn approval_listener(self: Arc<Self>, mut rx: mpsc::Receiver<ApprovalRequest>) {
        let mut fair_session_budget: SessionFairnessBudget = HashMap::new();
        loop {
            let request = match rx.recv().await {
                Some(r) => r,
                None => break, // channel closed
            };
            let approval_depth = rx.len().saturating_add(1);
            let mut pressure = QueuePressure::Normal;
            if let Some(queue_telemetry) = &self.queue_telemetry {
                queue_telemetry.mark_approval_received();
                let observation = queue_telemetry.observe_approval_depth(approval_depth);
                pressure = observation.pressure;
                if observation.entered_warning {
                    warn!(
                        queue = "approval",
                        depth = approval_depth,
                        "Approval queue entered warning state"
                    );
                }
                if observation.entered_overload {
                    warn!(
                        queue = "approval",
                        depth = approval_depth,
                        "Approval queue entered overload state"
                    );
                }
            }

            let should_shed = if let Some(queue_policy) = &self.queue_policy {
                should_shed_due_to_overload(
                    &queue_policy.lanes.approval,
                    pressure,
                    &mut fair_session_budget,
                    &request.session_id,
                )
            } else {
                false
            };

            if should_shed {
                if let Some(queue_telemetry) = &self.queue_telemetry {
                    queue_telemetry.mark_approval_dropped(1);
                    queue_telemetry.mark_approval_failed();
                    queue_telemetry.mark_approval_completed();
                }
                warn!(
                    session_id = %request.session_id,
                    "Dropping approval request due to configured overload shedding policy"
                );
                // Drop response_tx without manufacturing a user denial. Tools
                // still fail closed, but can distinguish infrastructure
                // unavailability from an explicit Deny response.
                continue;
            }

            let hub = self.clone();
            tokio::spawn(async move {
                let queue_telemetry = hub.queue_telemetry.clone();
                let route = hub.routed_channel(&request.session_id).await;
                let mut had_error = false;
                let response = match route {
                    Some((ch, routed_session)) => match request.kind {
                        ApprovalKind::GoalConfirmation => {
                            match ch
                                .request_goal_confirmation(
                                    &routed_session,
                                    &request.command,
                                    &request.warnings,
                                )
                                .await
                            {
                                Ok(true) => Some(ApprovalResponse::AllowOnce),
                                Ok(false) => Some(ApprovalResponse::Deny),
                                Err(e) => {
                                    warn!("Goal confirmation failed on {}: {}", ch.name(), e);
                                    had_error = true;
                                    None
                                }
                            }
                        }
                        ApprovalKind::Command | ApprovalKind::CommandOnce => {
                            let one_time_only = matches!(request.kind, ApprovalKind::CommandOnce);
                            match ch
                                .request_approval(
                                    &routed_session,
                                    &request.command,
                                    request.risk_level,
                                    &request.warnings,
                                    request.permission_mode,
                                    one_time_only,
                                )
                                .await
                            {
                                Ok(resp) => Some(if one_time_only {
                                    // Enforce the one-shot boundary even if an
                                    // older or forged channel callback returns a
                                    // broader choice that is no longer offered.
                                    match resp {
                                        ApprovalResponse::AllowSession
                                        | ApprovalResponse::AllowAlways => {
                                            ApprovalResponse::AllowOnce
                                        }
                                        other => other,
                                    }
                                } else {
                                    resp
                                }),
                                Err(e) => {
                                    warn!("Approval request failed on {}: {}", ch.name(), e);
                                    had_error = true;
                                    None
                                }
                            }
                        }
                    },
                    None => {
                        warn!(
                            "No channel found for session {}; approval unavailable",
                            request.session_id
                        );
                        had_error = true;
                        None
                    }
                };
                if let Some(response) = response {
                    if request.response_tx.send(response).is_err() {
                        had_error = true;
                        warn!(
                            session_id = %request.session_id,
                            "Approval response receiver dropped before response could be sent"
                        );
                    }
                }
                if let Some(queue_telemetry) = queue_telemetry {
                    if had_error {
                        queue_telemetry.mark_approval_failed();
                    }
                    queue_telemetry.mark_approval_completed();
                }
            });
        }
    }

    /// Route media messages from tools to the appropriate channel.
    pub async fn media_listener(self: Arc<Self>, mut rx: mpsc::Receiver<MediaMessage>) {
        let mut fair_session_budget: SessionFairnessBudget = HashMap::new();
        loop {
            let mut msg = match rx.recv().await {
                Some(m) => m,
                None => break, // channel closed
            };
            let media_depth = rx.len().saturating_add(1);
            let mut pressure = QueuePressure::Normal;
            if let Some(queue_telemetry) = &self.queue_telemetry {
                queue_telemetry.mark_media_received();
                let observation = queue_telemetry.observe_media_depth(media_depth);
                pressure = observation.pressure;
                if observation.entered_warning {
                    warn!(
                        queue = "media",
                        depth = media_depth,
                        "Media queue entered warning state"
                    );
                }
                if observation.entered_overload {
                    warn!(
                        queue = "media",
                        depth = media_depth,
                        "Media queue entered overload state; shedding non-critical media work"
                    );
                }
            }

            let should_shed = if let Some(queue_policy) = &self.queue_policy {
                should_shed_due_to_overload(
                    &queue_policy.lanes.media,
                    pressure,
                    &mut fair_session_budget,
                    &msg.session_id,
                )
            } else {
                false
            };

            if should_shed {
                let mut had_error = false;
                if let Some((channel, routed_session)) = self.routed_channel(&msg.session_id).await
                {
                    if let Err(e) = channel
                        .send_text(
                            &routed_session,
                            "[Media skipped due high system load. Please retry shortly.]",
                        )
                        .await
                    {
                        had_error = true;
                        warn!(
                            "Failed to send overload media fallback via {}: {}",
                            channel.name(),
                            e
                        );
                    }
                } else {
                    had_error = true;
                    warn!(
                        "No channel found for overloaded media session {}",
                        msg.session_id
                    );
                }
                if let Some(queue_telemetry) = &self.queue_telemetry {
                    queue_telemetry.mark_media_dropped();
                    if had_error {
                        queue_telemetry.mark_media_failed();
                    }
                    queue_telemetry.mark_media_completed();
                }
                // The sender (if it asked) deserves to know the media was NOT
                // delivered — it was shed under system overload.
                if let Some(result_tx) = msg.result_tx.take() {
                    let _ = result_tx.send(Err("system overload".to_string()));
                }
                continue;
            }

            let mut had_error = false;
            // The honest delivery outcome reported back to the enqueuing tool via
            // `result_tx` (if present). `Ok(())` ONLY when the media (or its text
            // fallback) was actually handed to the channel successfully. Reasons
            // are concise and free of secrets/URLs.
            let mut delivery_result: Result<(), String> = Ok(());
            if let Some((channel, routed_session)) = self.routed_channel(&msg.session_id).await {
                if channel.capabilities().media {
                    // Skip a document already delivered to this session recently
                    // (e.g. the notifier's deterministic delivery beat this
                    // `send_file` to the same file). The file is already in the
                    // chat, so report success to the enqueuing tool.
                    if !self.claim_media_delivery(&routed_session, &msg).await {
                        info!(
                            session_id = %routed_session,
                            "Suppressed duplicate document delivery from media queue (already delivered recently)"
                        );
                    } else if let Err(e) = channel.send_media(&routed_session, &msg).await {
                        self.release_media_delivery(&routed_session, &msg).await;
                        had_error = true;
                        delivery_result = Err(e.to_string());
                        warn!("Failed to send media via {}: {}", channel.name(), e);
                    } else {
                        self.record_media_delivery_note(&routed_session, &msg).await;
                    }
                } else {
                    // Channel doesn't support media — send caption as text
                    if let Err(e) = channel
                        .send_text(&routed_session, &format!("[Media] {}", msg.caption))
                        .await
                    {
                        had_error = true;
                        delivery_result = Err(e.to_string());
                        warn!("Failed to send media caption via {}: {}", channel.name(), e);
                    }
                }
            } else {
                had_error = true;
                delivery_result = Err("no channel found for session".to_string());
                warn!("No channel found for media session {}", msg.session_id);
            }
            if let Some(result_tx) = msg.result_tx.take() {
                let _ = result_tx.send(delivery_result);
            }
            if let Some(queue_telemetry) = &self.queue_telemetry {
                if had_error {
                    queue_telemetry.mark_media_failed();
                }
                queue_telemetry.mark_media_completed();
            }
        }
    }

    /// Send text to the channel that owns a specific session.
    #[allow(dead_code)]
    pub async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
        let (channel, routed_session) = self
            .routed_channel(session_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("No channel found for session {}", session_id))?;

        // Deduplicate identical spam (e.g. multiple heartbeats) within a short window.
        // This intentionally remains best-effort: it favors reducing noise over
        // perfect delivery guarantees.
        let mut delivery_claim = None;
        {
            let now = tokio::time::Instant::now();
            let text_norm = text.trim();
            match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                self.last_sent_text.write(),
            )
            .await
            {
                Ok(mut last) => {
                    if let Some((prev, prev_at)) = last.get(&routed_session) {
                        if prev.trim() == text_norm
                            && now.duration_since(*prev_at) < std::time::Duration::from_secs(10)
                        {
                            return Ok(());
                        }
                    }
                    last.insert(routed_session.clone(), (text_norm.to_string(), now));
                    delivery_claim = Some((text_norm.to_string(), now));
                }
                Err(_) => {
                    warn!(
                        session_id = %routed_session,
                        "Timed out acquiring dedupe lock in send_text; continuing without dedupe"
                    );
                }
            }
        }

        let result = channel.send_text(&routed_session, text).await;
        if result.is_err() {
            if let Some((claimed_text, claimed_at)) = delivery_claim {
                if let Ok(mut last) = tokio::time::timeout(
                    std::time::Duration::from_secs(2),
                    self.last_sent_text.write(),
                )
                .await
                {
                    let owns_claim = last
                        .get(&routed_session)
                        .is_some_and(|(text, at)| text == &claimed_text && *at == claimed_at);
                    if owns_claim {
                        last.remove(&routed_session);
                    }
                }
            }
        }
        result
    }

    /// Send text and return an editable message id when the owning channel
    /// supports it (`None` otherwise). Bypasses the send_text dedup window so the
    /// tracked message is always delivered and its id captured.
    pub async fn send_text_tracked(
        &self,
        session_id: &str,
        text: &str,
    ) -> anyhow::Result<Option<String>> {
        if let Some((channel, routed_session)) = self.routed_channel(session_id).await {
            channel.send_text_tracked(&routed_session, text).await
        } else {
            anyhow::bail!("No channel found for session {}", session_id)
        }
    }

    /// Edit a previously-sent message in place. Returns `Ok(false)` when no
    /// channel owns the session or the channel can't edit.
    pub async fn edit_text(
        &self,
        session_id: &str,
        message_id: &str,
        text: &str,
    ) -> anyhow::Result<bool> {
        if let Some((channel, routed_session)) = self.routed_channel(session_id).await {
            channel.edit_text(&routed_session, message_id, text).await
        } else {
            Ok(false)
        }
    }

    /// Send media to the channel that owns a specific session.
    /// Falls back to text caption for channels without media support.
    pub async fn send_media(&self, session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
        if let Some((channel, routed_session)) = self.routed_channel(session_id).await {
            if channel.capabilities().media {
                if !self.claim_media_delivery(&routed_session, media).await {
                    info!(
                        session_id = %routed_session,
                        "Suppressed duplicate document delivery (already delivered recently)"
                    );
                    return Ok(());
                }
                if let Err(e) = channel.send_media(&routed_session, media).await {
                    self.release_media_delivery(&routed_session, media).await;
                    return Err(e);
                }
                self.record_media_delivery_note(&routed_session, media)
                    .await;
                Ok(())
            } else {
                channel
                    .send_text(&routed_session, &format!("[File] {}", media.caption))
                    .await
            }
        } else {
            anyhow::bail!("No channel found for session {}", session_id)
        }
    }

    /// Send media to the channel that owns a session, requiring REAL media
    /// support. Unlike [`send_media`], this never silently falls back to a text
    /// caption: if the owning channel cannot deliver documents/photos it returns
    /// an error so callers (e.g. the terminal deliverable notifier) can report an
    /// honest delivery failure instead of mistaking a text fallback for success.
    pub async fn send_media_strict(
        &self,
        session_id: &str,
        media: &MediaMessage,
    ) -> anyhow::Result<()> {
        let Some((channel, routed_session)) = self.routed_channel(session_id).await else {
            anyhow::bail!("No channel found for session {}", session_id);
        };
        if !channel.capabilities().media {
            anyhow::bail!(
                "Channel '{}' for session {} cannot deliver media documents",
                channel.name(),
                routed_session
            );
        }
        // Suppress a duplicate of a document already delivered to this session
        // within the dedupe window (e.g. the model's `send_file` racing the
        // notifier's deterministic delivery of the same file). Treat as success —
        // the file IS in the chat.
        if !self.claim_media_delivery(&routed_session, media).await {
            info!(
                session_id = %routed_session,
                "Suppressed duplicate document delivery (already delivered recently)"
            );
            return Ok(());
        }
        if let Err(e) = channel.send_media(&routed_session, media).await {
            self.release_media_delivery(&routed_session, media).await;
            return Err(e);
        }
        self.record_media_delivery_note(&routed_session, media)
            .await;
        Ok(())
    }

    /// Broadcast text to a list of session IDs (e.g., trigger notifications).
    /// Errors are logged but don't stop the broadcast.
    pub async fn broadcast_text(&self, session_ids: &[String], text: &str) {
        for session_id in session_ids {
            if let Some((channel, routed_session)) = self.routed_channel(session_id).await {
                if let Err(e) = channel.send_text(&routed_session, text).await {
                    warn!(
                        channel = channel.name(),
                        session_id, "Broadcast send failed: {}", e
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    use async_trait::async_trait;
    use tokio::sync::Mutex;

    use crate::tools::command_risk::{PermissionMode, RiskLevel};
    use crate::traits::{Channel, ChannelCapabilities};
    use crate::types::{ApprovalResponse, MediaMessage};

    /// A test channel with a configurable name, used to verify routing.
    struct NamedTestChannel {
        channel_name: String,
        messages: Mutex<Vec<(String, String)>>, // (session_id, text)
        approvals: Mutex<Vec<String>>,
    }

    impl NamedTestChannel {
        fn new(name: &str) -> Self {
            Self {
                channel_name: name.to_string(),
                messages: Mutex::new(Vec::new()),
                approvals: Mutex::new(Vec::new()),
            }
        }

        async fn captured_messages(&self) -> Vec<(String, String)> {
            self.messages.lock().await.clone()
        }

        async fn captured_approvals(&self) -> Vec<String> {
            self.approvals.lock().await.clone()
        }
    }

    #[async_trait]
    impl Channel for NamedTestChannel {
        fn name(&self) -> String {
            self.channel_name.clone()
        }

        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities {
                markdown: true,
                inline_buttons: false,
                media: false,
                max_message_len: 4096,
            }
        }

        async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
            self.messages
                .lock()
                .await
                .push((session_id.to_string(), text.to_string()));
            Ok(())
        }

        async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn request_approval(
            &self,
            session_id: &str,
            _command: &str,
            _risk_level: RiskLevel,
            _warnings: &[String],
            _permission_mode: PermissionMode,
            _one_time_only: bool,
        ) -> anyhow::Result<ApprovalResponse> {
            self.approvals.lock().await.push(session_id.to_string());
            Ok(ApprovalResponse::AllowOnce)
        }
    }

    fn empty_session_map() -> SessionMap {
        Arc::new(RwLock::new(HashMap::new()))
    }

    /// A media-capable channel that counts how many documents it actually sent,
    /// so tests can prove de-duplication suppressed a redundant delivery.
    struct CountingMediaChannel {
        sent_docs: Mutex<Vec<String>>, // file paths
    }

    struct FlakyTextChannel {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Channel for FlakyTextChannel {
        fn name(&self) -> String {
            "flaky".to_string()
        }

        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities {
                markdown: true,
                inline_buttons: false,
                media: false,
                max_message_len: 4096,
            }
        }

        async fn send_text(&self, _session_id: &str, _text: &str) -> anyhow::Result<()> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("synthetic first-send failure");
            }
            Ok(())
        }

        async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn request_approval(
            &self,
            _session_id: &str,
            _command: &str,
            _risk_level: RiskLevel,
            _warnings: &[String],
            _permission_mode: PermissionMode,
            _one_time_only: bool,
        ) -> anyhow::Result<ApprovalResponse> {
            Ok(ApprovalResponse::AllowOnce)
        }
    }

    impl CountingMediaChannel {
        fn new() -> Self {
            Self {
                sent_docs: Mutex::new(Vec::new()),
            }
        }
        async fn doc_count(&self) -> usize {
            self.sent_docs.lock().await.len()
        }
    }

    #[async_trait]
    impl Channel for CountingMediaChannel {
        fn name(&self) -> String {
            "counting_media".to_string()
        }
        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities {
                markdown: true,
                inline_buttons: false,
                media: true,
                max_message_len: 4096,
            }
        }
        async fn send_text(&self, _session_id: &str, _text: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn send_media(&self, _session_id: &str, media: &MediaMessage) -> anyhow::Result<()> {
            if let crate::types::MediaKind::Document { file_path, .. } = &media.kind {
                self.sent_docs.lock().await.push(file_path.clone());
            }
            Ok(())
        }
        async fn request_approval(
            &self,
            _session_id: &str,
            _command: &str,
            _risk_level: RiskLevel,
            _warnings: &[String],
            _permission_mode: PermissionMode,
            _one_time_only: bool,
        ) -> anyhow::Result<ApprovalResponse> {
            Ok(ApprovalResponse::AllowOnce)
        }
    }

    fn doc_msg(session_id: &str, file_path: &str, filename: &str) -> MediaMessage {
        MediaMessage {
            session_id: session_id.to_string(),
            caption: filename.to_string(),
            kind: crate::types::MediaKind::Document {
                file_path: file_path.to_string(),
                filename: filename.to_string(),
            },
            result_tx: None,
        }
    }

    #[tokio::test]
    async fn send_media_strict_dedupes_identical_document_within_window() {
        let channel = Arc::new(CountingMediaChannel::new());
        let session_map = empty_session_map();
        session_map
            .write()
            .await
            .insert("s1".to_string(), "counting_media".to_string());
        let hub = ChannelHub::new(vec![channel.clone() as Arc<dyn Channel>], session_map);

        // Same file delivered twice (e.g. notifier then model send_file) → one send.
        hub.send_media_strict("s1", &doc_msg("s1", "/inbox/r.txt", "r.txt"))
            .await
            .unwrap();
        hub.send_media_strict("s1", &doc_msg("s1", "/inbox/r.txt", "r.txt"))
            .await
            .unwrap();
        assert_eq!(
            channel.doc_count().await,
            1,
            "identical document must be delivered only once within the dedupe window"
        );

        // A genuinely different file is NOT suppressed.
        hub.send_media_strict("s1", &doc_msg("s1", "/inbox/other.txt", "other.txt"))
            .await
            .unwrap();
        assert_eq!(
            channel.doc_count().await,
            2,
            "a different document must still be delivered"
        );
    }

    fn session_map_with(entries: Vec<(&str, &str)>) -> SessionMap {
        let mut map = HashMap::new();
        for (session, channel) in entries {
            map.insert(session.to_string(), channel.to_string());
        }
        Arc::new(RwLock::new(map))
    }

    #[tokio::test]
    async fn test_channel_for_session_known() {
        let ch_telegram: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("slack"));

        let session_map = session_map_with(vec![("sess_1", "slack")]);
        let hub = ChannelHub::new(vec![ch_telegram, ch_slack], session_map);

        let found = hub.channel_for_session("sess_1").await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().name(), "slack");
    }

    #[tokio::test]
    async fn test_channel_for_session_unknown_returns_none() {
        let ch_telegram: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("slack"));

        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![ch_telegram, ch_slack], session_map);

        // Unknown session should return None to prevent cross-channel leaks
        let found = hub.channel_for_session("unknown_session").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn child_approval_is_delivered_through_parent_session() {
        let channel = Arc::new(NamedTestChannel::new("telegram"));
        let session_map = session_map_with(vec![("parent-session", "telegram")]);
        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn Channel>],
            session_map,
        ));
        let route_guard = hub
            .register_session_route("specialist:browser:child", "parent-session")
            .await
            .expect("route guard");

        hub.send_text("specialist:browser:child", "child result")
            .await
            .unwrap();
        assert_eq!(
            channel.captured_messages().await,
            vec![("parent-session".to_string(), "child result".to_string())],
            "all child output must use the externally addressable parent session"
        );

        let (approval_tx, approval_rx) = mpsc::channel(1);
        let listener = tokio::spawn(hub.clone().approval_listener(approval_rx));
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        approval_tx
            .send(ApprovalRequest {
                command: "Navigate to public site".to_string(),
                session_id: "specialist:browser:child".to_string(),
                risk_level: RiskLevel::Medium,
                warnings: Vec::new(),
                permission_mode: PermissionMode::Default,
                response_tx,
                kind: Default::default(),
            })
            .await
            .unwrap();

        assert!(matches!(
            response_rx.await.unwrap(),
            ApprovalResponse::AllowOnce
        ));
        drop(route_guard);
        tokio::task::yield_now().await;
        let (unrouted_tx, unrouted_rx) = tokio::sync::oneshot::channel();
        approval_tx
            .send(ApprovalRequest {
                command: "Navigate to public site".to_string(),
                session_id: "specialist:browser:child".to_string(),
                risk_level: RiskLevel::Medium,
                warnings: Vec::new(),
                permission_mode: PermissionMode::Default,
                response_tx: unrouted_tx,
                kind: Default::default(),
            })
            .await
            .unwrap();
        assert!(
            unrouted_rx.await.is_err(),
            "missing runtime route must close the request as unavailable, not fabricate Deny"
        );
        drop(approval_tx);
        listener.await.unwrap();
        assert_eq!(channel.captured_approvals().await, vec!["parent-session"]);
    }

    #[tokio::test]
    async fn failed_text_send_releases_dedupe_claim_for_retry() {
        let channel = Arc::new(FlakyTextChannel {
            calls: AtomicUsize::new(0),
        });
        let session_map = session_map_with(vec![("session", "flaky")]);
        let hub = ChannelHub::new(vec![channel.clone() as Arc<dyn Channel>], session_map);

        assert!(hub.send_text("session", "important result").await.is_err());
        hub.send_text("session", "important result")
            .await
            .expect("retry must reach the channel");
        assert_eq!(channel.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn stale_route_guard_cannot_remove_newer_route() {
        let channel = Arc::new(NamedTestChannel::new("telegram"));
        let session_map =
            session_map_with(vec![("parent-one", "telegram"), ("parent-two", "telegram")]);
        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn Channel>],
            session_map,
        ));
        let old_guard = hub
            .register_session_route("child", "parent-one")
            .await
            .unwrap();
        let new_guard = hub
            .register_session_route("child", "parent-two")
            .await
            .unwrap();

        drop(old_guard);
        hub.send_text("child", "still routed")
            .await
            .expect("new route must survive old cleanup");
        assert_eq!(
            channel.captured_messages().await,
            vec![("parent-two".to_string(), "still routed".to_string())]
        );
        drop(new_guard);
    }

    #[tokio::test]
    async fn aborting_child_future_removes_session_route() {
        let channel = Arc::new(NamedTestChannel::new("telegram"));
        let session_map = session_map_with(vec![("parent", "telegram")]);
        let hub = Arc::new(ChannelHub::new(
            vec![channel as Arc<dyn Channel>],
            session_map,
        ));
        let child_hub = hub.clone();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let child = tokio::spawn(async move {
            let _route = child_hub
                .register_session_route("child", "parent")
                .await
                .unwrap();
            let _ = ready_tx.send(());
            std::future::pending::<()>().await;
        });
        ready_rx.await.unwrap();
        child.abort();
        let _ = child.await;
        tokio::task::yield_now().await;

        assert!(
            hub.send_text("child", "must not leak").await.is_err(),
            "aborted child route must not remain externally addressable"
        );
    }

    #[tokio::test]
    async fn test_channel_for_session_empty() {
        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![], session_map);

        let found = hub.channel_for_session("any_session").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_send_text_routes_correctly() {
        let ch_telegram = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack = Arc::new(NamedTestChannel::new("slack"));

        let ch_telegram_dyn: Arc<dyn Channel> = ch_telegram.clone();
        let ch_slack_dyn: Arc<dyn Channel> = ch_slack.clone();

        let session_map = session_map_with(vec![("sess_1", "slack")]);
        let hub = ChannelHub::new(vec![ch_telegram_dyn, ch_slack_dyn], session_map);

        hub.send_text("sess_1", "Hello Slack!").await.unwrap();

        // Slack channel should have the message
        let slack_msgs = ch_slack.captured_messages().await;
        assert_eq!(slack_msgs.len(), 1);
        assert_eq!(slack_msgs[0].0, "sess_1");
        assert_eq!(slack_msgs[0].1, "Hello Slack!");

        // Telegram channel should have no messages
        let telegram_msgs = ch_telegram.captured_messages().await;
        assert_eq!(telegram_msgs.len(), 0);
    }

    #[tokio::test]
    async fn test_tracked_send_and_edit_fall_back_on_non_editing_channel() {
        // A channel that implements only send_text uses the trait defaults:
        // send_text_tracked delivers the message but returns None (no id), and
        // edit_text reports unsupported. This is the cross-channel fallback that
        // keeps the checklist's start-post + recap behavior intact off Telegram.
        let ch = Arc::new(NamedTestChannel::new("slack"));
        let ch_dyn: Arc<dyn Channel> = ch.clone();
        let session_map = session_map_with(vec![("sess_1", "slack")]);
        let hub = ChannelHub::new(vec![ch_dyn], session_map);

        let id = hub
            .send_text_tracked("sess_1", "📋 Plan")
            .await
            .expect("tracked send ok");
        assert!(
            id.is_none(),
            "non-editing channel must report no message id"
        );
        // The message was still delivered via the default send_text path.
        let msgs = ch.captured_messages().await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].1, "📋 Plan");

        let edited = hub
            .edit_text("sess_1", "123", "📋 Plan ✅")
            .await
            .expect("edit ok");
        assert!(!edited, "non-editing channel must report edit unsupported");
    }

    #[tokio::test]
    async fn test_edit_text_no_channel_is_noop() {
        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![], session_map);
        let edited = hub.edit_text("sess_1", "1", "x").await.expect("ok");
        assert!(!edited);
    }

    #[tokio::test]
    async fn test_send_text_no_channels_errors() {
        let session_map = empty_session_map();
        let hub = ChannelHub::new(vec![], session_map);

        let result = hub.send_text("sess_1", "Hello?").await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("No channel found"),
            "Expected 'No channel found' error, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    async fn test_broadcast_sends_to_all() {
        let ch_telegram = Arc::new(NamedTestChannel::new("telegram"));
        let ch_slack = Arc::new(NamedTestChannel::new("slack"));

        let ch_telegram_dyn: Arc<dyn Channel> = ch_telegram.clone();
        let ch_slack_dyn: Arc<dyn Channel> = ch_slack.clone();

        let session_map =
            session_map_with(vec![("sess_telegram", "telegram"), ("sess_slack", "slack")]);
        let hub = ChannelHub::new(vec![ch_telegram_dyn, ch_slack_dyn], session_map);

        let ids = vec!["sess_telegram".to_string(), "sess_slack".to_string()];
        hub.broadcast_text(&ids, "Broadcast!").await;

        let telegram_msgs = ch_telegram.captured_messages().await;
        assert_eq!(telegram_msgs.len(), 1);
        assert_eq!(telegram_msgs[0].1, "Broadcast!");

        let slack_msgs = ch_slack.captured_messages().await;
        assert_eq!(slack_msgs.len(), 1);
        assert_eq!(slack_msgs[0].1, "Broadcast!");
    }

    #[tokio::test]
    async fn test_register_channel_dynamically() {
        let session_map = session_map_with(vec![("sess_1", "discord")]);
        let hub = ChannelHub::new(vec![], session_map);

        // Initially no channels, so send_text should fail
        assert!(hub.send_text("sess_1", "test").await.is_err());

        // Register a channel dynamically
        let ch_discord: Arc<dyn Channel> = Arc::new(NamedTestChannel::new("discord"));
        let name = hub.register_channel(ch_discord).await;
        assert_eq!(name, "discord");

        // Now send_text should succeed
        let result = hub.send_text("sess_1", "Hello Discord!").await;
        assert!(result.is_ok());
    }
}
