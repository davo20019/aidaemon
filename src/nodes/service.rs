use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use crate::agent::Agent;
use crate::config::NodesConfig;
use crate::traits::MessageAttachment;
use crate::types::{ChannelContext, ChannelVisibility, UserRole};

use super::auth;
use super::domain::{
    AuthenticatedNodeContext, NodeAction, NodeTurnState, CHILD_COMPANION_POLICY,
    NODE_PROTOCOL_MAJOR,
};
use super::protocol::*;
use super::store::{CreateTurnOutcome, NodeStore, PairingOffer};

#[async_trait]
pub trait NodeConversationIngress: Send + Sync {
    async fn respond(
        &self,
        context: &AuthenticatedNodeContext,
        text: &str,
    ) -> anyhow::Result<String>;
    async fn respond_with_attachment(
        &self,
        context: &AuthenticatedNodeContext,
        text: &str,
        attachment: MessageAttachment,
    ) -> anyhow::Result<String> {
        let _ = attachment;
        self.respond(context, text).await
    }
}

pub struct AgentNodeConversationIngress {
    agent: Arc<Agent>,
}

impl AgentNodeConversationIngress {
    pub fn new(agent: Arc<Agent>) -> Self {
        Self { agent }
    }
}

#[async_trait]
impl NodeConversationIngress for AgentNodeConversationIngress {
    async fn respond(
        &self,
        context: &AuthenticatedNodeContext,
        text: &str,
    ) -> anyhow::Result<String> {
        let channel_context = ChannelContext {
            visibility: ChannelVisibility::PublicExternal,
            platform: "node".to_string(),
            channel_name: Some("Companion".to_string()),
            channel_id: Some(format!("node:{}", context.node_channel_id)),
            workspace_id: None,
            sender_name: Some("Companion user".to_string()),
            sender_id: Some(format!("node-principal:{}", context.node_id)),
            channel_member_names: Vec::new(),
            user_id_map: HashMap::new(),
            workspace_grant: None,
            trusted: false,
        };
        self.agent
            .handle_message(
                &context.conversation_session_id,
                text,
                None,
                UserRole::Public,
                channel_context,
                None,
            )
            .await
    }

    async fn respond_with_attachment(
        &self,
        context: &AuthenticatedNodeContext,
        text: &str,
        attachment: MessageAttachment,
    ) -> anyhow::Result<String> {
        let channel_context = node_channel_context(context);
        self.agent
            .handle_message_with_attachments(
                &context.conversation_session_id,
                text,
                &[attachment],
                None,
                UserRole::Public,
                channel_context,
                None,
            )
            .await
    }
}

fn node_channel_context(context: &AuthenticatedNodeContext) -> ChannelContext {
    ChannelContext {
        visibility: ChannelVisibility::PublicExternal,
        platform: "node".to_string(),
        channel_name: Some("Companion".to_string()),
        channel_id: Some(format!("node:{}", context.node_channel_id)),
        workspace_id: None,
        sender_name: Some("Companion user".to_string()),
        sender_id: Some(format!("node-principal:{}", context.node_id)),
        channel_member_names: Vec::new(),
        user_id_map: HashMap::new(),
        workspace_grant: None,
        trusted: false,
    }
}

fn validate_recovery_code(label: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty()
            && value.len() <= 48
            && value.bytes().all(|byte| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || matches!(byte, b'_' | b'-' | b'.')
            }),
        "{label} must be a bounded structural code"
    );
    Ok(())
}

fn validate_runtime_recovery_report(report: &RuntimeRecoveryReport) -> anyhow::Result<()> {
    anyhow::ensure!(
        report.schema == "aidaemon.runtime.recovery.v1",
        "unsupported Runtime recovery schema"
    );
    validate_recovery_code("reset_reason", &report.reset_reason)?;
    if let Some(stage) = report.previous_stage.as_deref() {
        validate_recovery_code("previous_stage", stage)?;
    }
    if let Some(reason) = report.planned_restart_reason.as_deref() {
        validate_recovery_code("planned_restart_reason", reason)?;
    }
    anyhow::ensure!(
        report.previous_uptime_ms.unwrap_or(0) <= i64::MAX as u64,
        "previous_uptime_ms is out of range"
    );
    anyhow::ensure!(
        report.consecutive_unhealthy_resets <= u8::MAX as u16,
        "consecutive_unhealthy_resets is out of range"
    );
    anyhow::ensure!(
        report.connectivity_restart_count <= u8::MAX as u16,
        "connectivity_restart_count is out of range"
    );
    if let Some(timeout) = report.watchdog_timeout_seconds {
        anyhow::ensure!(
            (1..=3600).contains(&timeout),
            "watchdog_timeout_seconds is out of range"
        );
    }
    Ok(())
}

#[derive(Clone)]
pub struct NodeService {
    store: Arc<NodeStore>,
    config: NodesConfig,
    speech: Arc<dyn super::speech::NodeSpeechSynthesizer>,
    firmware_release: Option<Arc<super::ota::FirmwareRelease>>,
}

impl NodeService {
    pub fn new(store: Arc<NodeStore>, config: NodesConfig) -> Self {
        Self {
            store,
            config,
            speech: Arc::new(super::speech::DisabledSpeech),
            firmware_release: None,
        }
    }
    pub fn with_speech(mut self, speech: Arc<dyn super::speech::NodeSpeechSynthesizer>) -> Self {
        self.speech = speech;
        self
    }
    pub fn with_firmware_release(mut self, release: super::ota::FirmwareRelease) -> Self {
        self.firmware_release = Some(Arc::new(release));
        self
    }
    pub fn store(&self) -> &Arc<NodeStore> {
        &self.store
    }

    pub async fn create_pairing_offer(
        &self,
        owner_id: &str,
        kind: &str,
        display_name: &str,
        policy_profile: &str,
    ) -> anyhow::Result<PairingOffer> {
        self.store
            .create_pairing_offer(
                owner_id,
                kind,
                display_name,
                policy_profile,
                self.config.gateway.pairing_ttl_seconds,
            )
            .await
    }

    pub async fn redeem(
        &self,
        request: RedeemEnrollmentRequest,
    ) -> anyhow::Result<RedeemEnrollmentResponse> {
        ensure_protocol(&request.protocol)?;
        anyhow::ensure!(
            request.capabilities.len() <= 32,
            "too many capability observations"
        );
        let public_key = auth::decode_public_key(&request.public_key_sec1)?;
        let result = self
            .store
            .redeem_enrollment(
                &request.offer_id,
                &request.offer_secret,
                &request.node_kind,
                &public_key,
                &request.metadata.runtime_version,
                &request.metadata.firmware_version,
                &request.capabilities,
            )
            .await?;
        Ok(RedeemEnrollmentResponse {
            node_id: result.node.node_id,
            credential_id: result.credential_id,
            node_channel_id: result.node.node_channel_id,
            policy_profile: result.node.policy_profile,
            public_key_fingerprint: result.public_key_fingerprint,
            protocol: ProtocolVersion::default(),
        })
    }

    pub async fn challenge(
        &self,
        request: SessionChallengeRequest,
    ) -> anyhow::Result<SessionChallengeResponse> {
        ensure_protocol(&request.protocol)?;
        let record = self
            .store
            .create_challenge(
                &request.credential_id,
                request.protocol.major,
                self.config.gateway.challenge_ttl_seconds,
            )
            .await?;
        Ok(SessionChallengeResponse {
            challenge_id: record.challenge_id,
            nonce: record.nonce,
            instance_id: self.store.instance_id().to_string(),
            expires_in_seconds: self.config.gateway.challenge_ttl_seconds,
        })
    }

    pub async fn open_session(
        &self,
        request: OpenSessionRequest,
    ) -> anyhow::Result<OpenSessionResponse> {
        ensure_protocol(&request.protocol)?;
        super::domain::validate_identifier("boot id", &request.boot_id, 8, 80)?;
        let challenge = self
            .store
            .get_challenge(&request.challenge_id, &request.credential_id)
            .await?;
        let canonical = auth::canonical_session_challenge(
            &request.credential_id,
            &request.challenge_id,
            &challenge.nonce,
            request.protocol.major,
            self.store.instance_id(),
            &request.boot_id,
        );
        auth::verify_session_signature(
            &challenge.public_key_sec1,
            &canonical,
            &request.signature_der,
        )?;
        let created = self
            .store
            .consume_challenge_and_create_session(
                &challenge,
                request.protocol.major,
                &request.boot_id,
                self.config.gateway.session_ttl_seconds,
            )
            .await?;
        Ok(OpenSessionResponse {
            node_session_id: created.context.node_session_id,
            access_token: created.access_token,
            expires_in_seconds: self.config.gateway.session_ttl_seconds,
            policy_profile: created.context.policy_profile,
            policy_revision: created.context.policy_revision,
        })
    }

    pub async fn create_credential_rotation_challenge(
        &self,
        context: &AuthenticatedNodeContext,
        request: CredentialRotationChallengeRequest,
    ) -> anyhow::Result<CredentialRotationChallengeResponse> {
        super::domain::validate_identifier("request id", &request.request_id, 8, 80)?;
        let record = self
            .store
            .create_credential_rotation_challenge(
                context,
                self.config.gateway.challenge_ttl_seconds,
            )
            .await?;
        Ok(CredentialRotationChallengeResponse {
            rotation_id: record.rotation_id,
            nonce: record.nonce,
            instance_id: self.store.instance_id().to_string(),
            expires_in_seconds: self.config.gateway.challenge_ttl_seconds,
        })
    }

    pub async fn rotate_credential(
        &self,
        context: &AuthenticatedNodeContext,
        request: RotateCredentialRequest,
    ) -> anyhow::Result<RotateCredentialResponse> {
        let new_public_key = auth::decode_public_key(&request.new_public_key_sec1)?;
        let record = self
            .store
            .credential_rotation_challenge(context, &request.rotation_id)
            .await?;
        let canonical = auth::canonical_credential_rotation(
            &context.node_id,
            &context.node_session_id,
            &record.rotation_id,
            &record.nonce,
            self.store.instance_id(),
            &request.new_public_key_sec1,
        );
        auth::verify_session_signature(
            &new_public_key,
            &canonical,
            &request.new_key_signature_der,
        )?;
        let result = self
            .store
            .consume_credential_rotation(&record, &new_public_key)
            .await?;
        Ok(RotateCredentialResponse {
            credential_id: result.credential_id,
            public_key_fingerprint: result.public_key_fingerprint,
            sessions_closed: true,
        })
    }

    pub async fn authenticate(&self, token: &str) -> anyhow::Result<AuthenticatedNodeContext> {
        self.store.authenticate(token).await
    }

    pub async fn authorize(
        &self,
        context: &AuthenticatedNodeContext,
        action: NodeAction,
    ) -> anyhow::Result<()> {
        // This is a policy ceiling, separate from the mutable authorization table.
        if context.policy_profile == CHILD_COMPANION_POLICY {
            anyhow::ensure!(
                matches!(
                    action,
                    NodeAction::SubmitTextTurn
                        | NodeAction::SubmitAudioTurn
                        | NodeAction::SubmitStillImage
                        | NodeAction::ReportTelemetry
                        | NodeAction::ReportSensor
                        | NodeAction::ReceiveAudio
                        | NodeAction::ReceiveOta
                ),
                "child companion policy forbids this action"
            );
        }
        anyhow::ensure!(
            self.store.is_authorized(&context.node_id, action).await?,
            "Node is not authorized for this action"
        );
        Ok(())
    }

    pub async fn heartbeat(
        &self,
        context: &AuthenticatedNodeContext,
        heartbeat: HeartbeatRequest,
    ) -> anyhow::Result<HeartbeatResponse> {
        self.authorize(context, NodeAction::ReportTelemetry).await?;
        if let Some(recovery) = heartbeat.recovery.as_ref() {
            validate_runtime_recovery_report(recovery)?;
        }
        let audio_capable = heartbeat.capabilities.iter().any(|capability| {
            capability.capability_id == "output.audio" && capability.version >= 1
        });
        let ota_capability = heartbeat.capabilities.iter().find(|capability| {
            capability.capability_id == "firmware.ota"
                && capability.version >= 1
                && capability
                    .limits
                    .get("board")
                    .and_then(|value| value.as_str())
                    == Some(super::ota::K10_BOARD_ID)
        });
        let ota_capable = ota_capability.is_some();
        let observed_ota_sequence = ota_capability
            .and_then(|capability| capability.limits.get("sequence"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let current_firmware_version = heartbeat.firmware_version.clone();
        self.store.record_heartbeat(context, &heartbeat).await?;
        let outbox_poll_seconds = if self.config.announcements.enabled
            && audio_capable
            && self
                .store
                .is_authorized(&context.node_id, NodeAction::ReceiveAudio)
                .await?
        {
            self.config.announcements.poll_interval_seconds
        } else {
            0
        };
        let firmware_update = if self.config.ota.enabled
            && ota_capable
            && self
                .store
                .is_authorized(&context.node_id, NodeAction::ReceiveOta)
                .await?
        {
            self.firmware_release
                .as_ref()
                .filter(|release| {
                    release.manifest().version != current_firmware_version
                        && release.manifest().sequence > observed_ota_sequence
                })
                .map(|release| release.offer())
        } else {
            None
        };
        Ok(HeartbeatResponse {
            accepted: true,
            next_heartbeat_seconds: self.config.gateway.heartbeat_interval_seconds,
            policy_revision: context.policy_revision,
            authorization_revision: context.authorization_revision,
            outbox_poll_seconds,
            firmware_update,
            gateway_endpoints: self.config.gateway.advertised_endpoints.clone(),
        })
    }

    pub async fn firmware_image(
        &self,
        context: &AuthenticatedNodeContext,
        release_id: &str,
    ) -> anyhow::Result<Arc<[u8]>> {
        anyhow::ensure!(self.config.ota.enabled, "Node OTA is disabled");
        self.authorize(context, NodeAction::ReceiveOta).await?;
        self.firmware_release
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("firmware release is unavailable"))?
            .image_for(release_id)
    }

    pub async fn node_outbox(
        &self,
        context: &AuthenticatedNodeContext,
        after: u64,
    ) -> anyhow::Result<NodeOutboxResponse> {
        anyhow::ensure!(
            self.config.announcements.enabled,
            "Node audio announcements are disabled"
        );
        anyhow::ensure!(after <= i64::MAX as u64, "outbox cursor is out of range");
        self.authorize(context, NodeAction::ReceiveAudio).await?;
        self.cleanup_outbound_media(Some(&context.node_id)).await;
        let events = self
            .store
            .pending_node_outbox(&context.node_id, after, 1)
            .await?;
        let next_cursor = events.last().map(|event| event.cursor).unwrap_or(after);
        Ok(NodeOutboxResponse {
            events,
            next_cursor,
        })
    }

    pub async fn acknowledge_node_outbox(
        &self,
        context: &AuthenticatedNodeContext,
        cursor: u64,
        request: AckNodeOutboxRequest,
    ) -> anyhow::Result<AckNodeOutboxResponse> {
        anyhow::ensure!(
            self.config.announcements.enabled,
            "Node audio announcements are disabled"
        );
        anyhow::ensure!(cursor <= i64::MAX as u64, "outbox cursor is out of range");
        self.authorize(context, NodeAction::ReceiveAudio).await?;
        if let Some(detail_code) = request.detail_code.as_deref() {
            super::domain::validate_identifier("outbox detail code", detail_code, 1, 80)?;
        }
        let cleanup = self
            .store
            .acknowledge_node_outbox(
                &context.node_id,
                cursor,
                request.status,
                request.detail_code.as_deref(),
            )
            .await?;
        if let Some(cleanup) = cleanup {
            match tokio::fs::remove_file(&cleanup.local_path).await {
                Ok(()) => {
                    let _ = self
                        .store
                        .mark_outbound_media_deleted(&cleanup.media_id)
                        .await;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let _ = self
                        .store
                        .mark_outbound_media_deleted(&cleanup.media_id)
                        .await;
                }
                Err(_) => {}
            }
        }
        Ok(AckNodeOutboxResponse {
            accepted: true,
            status: request.status,
        })
    }

    async fn cleanup_outbound_media(&self, node_id: Option<&str>) {
        let Ok(candidates) = self.store.outbound_media_cleanup_candidates(node_id).await else {
            return;
        };
        for candidate in candidates {
            match tokio::fs::remove_file(&candidate.local_path).await {
                Ok(()) => {
                    let _ = self
                        .store
                        .mark_outbound_media_deleted(&candidate.media_id)
                        .await;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    let _ = self
                        .store
                        .mark_outbound_media_deleted(&candidate.media_id)
                        .await;
                }
                Err(_) => {}
            }
        }
    }

    pub async fn report_sensor_readings(
        &self,
        context: &AuthenticatedNodeContext,
        request: ReportSensorReadingsRequest,
    ) -> anyhow::Result<ReportSensorReadingsResponse> {
        self.authorize(context, NodeAction::ReportSensor).await?;
        super::domain::validate_identifier("request id", &request.request_id, 8, 80)?;
        anyhow::ensure!(
            request.sample_uptime_ms <= i64::MAX as u64,
            "sample uptime is out of range"
        );
        anyhow::ensure!(
            (1..=8).contains(&request.readings.len()),
            "sensor report must contain 1-8 readings"
        );
        let mut seen = HashSet::with_capacity(request.readings.len());
        for reading in &request.readings {
            anyhow::ensure!(
                seen.insert(reading.capability_id.as_str()),
                "sensor report contains duplicate capabilities"
            );
            validate_sensor_reading(reading)?;
        }
        let received_at = self
            .store
            .record_sensor_readings(context, &request, Some(&self.config.monitoring))
            .await?;
        if self.config.monitoring.enabled {
            let monitoring = super::monitoring::NodeMonitoringService::new(
                self.store.pool().clone(),
                self.config.monitoring.clone(),
            );
            if let Err(error) = monitoring.evaluate_node_readings(&context.node_id).await {
                tracing::warn!(
                    %error,
                    "Node sensor readings were accepted but monitor evaluation failed"
                );
            }
        }
        Ok(ReportSensorReadingsResponse {
            accepted: true,
            received_at: received_at.to_rfc3339(),
        })
    }

    pub async fn create_text_turn(
        &self,
        context: &AuthenticatedNodeContext,
        idempotency_key: &str,
        request_id: &str,
        text: &str,
    ) -> anyhow::Result<CreateTurnOutcome> {
        self.authorize(context, NodeAction::SubmitTextTurn).await?;
        anyhow::ensure!(
            (16..=128).contains(&idempotency_key.len()),
            "Idempotency-Key must be 16-128 bytes"
        );
        super::domain::validate_identifier("request id", request_id, 8, 80)?;
        anyhow::ensure!(!text.trim().is_empty(), "turn text is empty");
        anyhow::ensure!(
            text.chars().count() <= self.config.limits.text_chars,
            "turn text exceeds configured limit"
        );
        self.store
            .create_text_turn(context, idempotency_key, request_id, text)
            .await
    }

    pub async fn create_media_turn(
        &self,
        context: &AuthenticatedNodeContext,
        idempotency_key: &str,
        request: CreateTurnRequest,
    ) -> anyhow::Result<(CreateTurnOutcome, UploadSlot)> {
        anyhow::ensure!(
            (16..=128).contains(&idempotency_key.len()),
            "Idempotency-Key must be 16-128 bytes"
        );
        let (
            action,
            request_id,
            media_kind,
            content_type,
            size_bytes,
            sha256,
            duration_ms,
            max_bytes,
        ) = match request {
            CreateTurnRequest::Audio {
                request_id,
                content_type,
                size_bytes,
                sha256,
                duration_ms,
            } => {
                anyhow::ensure!(
                    content_type == "audio/wav",
                    "K10 MVP accepts audio/wav only"
                );
                anyhow::ensure!(
                    (100..=10_000).contains(&duration_ms),
                    "audio duration must be 100-10000 ms"
                );
                (
                    NodeAction::SubmitAudioTurn,
                    request_id,
                    "audio",
                    content_type,
                    size_bytes,
                    sha256,
                    Some(duration_ms),
                    self.config.limits.audio_upload_bytes as u64,
                )
            }
            CreateTurnRequest::StillImage {
                request_id,
                content_type,
                size_bytes,
                sha256,
                width,
                height,
                capture_duration_ms,
            } => {
                anyhow::ensure!(
                    content_type == "image/jpeg",
                    "K10 MVP accepts image/jpeg only"
                );
                if let Some(width) = width {
                    anyhow::ensure!((1..=4096).contains(&width), "image width is invalid");
                }
                if let Some(height) = height {
                    anyhow::ensure!((1..=4096).contains(&height), "image height is invalid");
                }
                if let Some(duration) = capture_duration_ms {
                    anyhow::ensure!(duration <= 30_000, "capture duration is invalid");
                }
                (
                    NodeAction::SubmitStillImage,
                    request_id,
                    "still_image",
                    content_type,
                    size_bytes,
                    sha256,
                    capture_duration_ms,
                    self.config.limits.image_upload_bytes as u64,
                )
            }
            CreateTurnRequest::Text { .. } => anyhow::bail!("text turn was sent to the media flow"),
        };
        self.authorize(context, action).await?;
        super::domain::validate_identifier("request id", &request_id, 8, 80)?;
        anyhow::ensure!(
            (1..=max_bytes).contains(&size_bytes),
            "media size exceeds configured limit"
        );
        anyhow::ensure!(
            sha256.len() == 64 && sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "sha256 must be 64 hexadecimal characters"
        );
        let (outcome, slot_id) = self
            .store
            .create_media_turn(
                context,
                idempotency_key,
                &request_id,
                media_kind,
                &content_type,
                size_bytes,
                &sha256.to_ascii_lowercase(),
                duration_ms,
            )
            .await?;
        let slot = UploadSlot {
            slot_id: slot_id.clone(),
            content_type,
            max_bytes,
            upload_path: format!("/node/v1/uploads/{slot_id}"),
        };
        Ok((outcome, slot))
    }

    pub async fn process_media_turn(
        &self,
        context: AuthenticatedNodeContext,
        upload: super::store::MediaUploadRecord,
        ingress: Arc<dyn NodeConversationIngress>,
    ) {
        if self
            .store
            .begin_turn_processing(&context.node_id, &upload.turn_id)
            .await
            .is_err()
        {
            return;
        }
        let Some(path) = upload.local_path.clone() else {
            return;
        };
        let filename = if upload.media_kind == "audio" {
            "voice.wav"
        } else {
            "photo.jpg"
        };
        let prompt = if upload.media_kind == "audio" {
            "The child sent this voice message. Answer aloud in one or two short, child-friendly sentences."
        } else {
            "The child explicitly took and sent this photo."
        };
        let attachment = MessageAttachment {
            resource_id: Some(format!("node-upload:{}", upload.slot_id)),
            local_path: path.clone(),
            filename: filename.to_string(),
            mime_type: upload.content_type.clone(),
            size_bytes: upload.expected_bytes,
            provenance: Default::default(),
            source_tool: None,
            sha256: Some(upload.expected_sha256.clone()),
        };
        let result = ingress
            .respond_with_attachment(&context, prompt, attachment)
            .await;
        let deleted = if self.config.retention.retain_raw_media {
            false
        } else {
            tokio::fs::remove_file(&path).await.is_ok()
        };
        let _ = self
            .store
            .mark_slot_consumed(&context.node_id, &upload.slot_id, deleted)
            .await;
        match result {
            Ok(response)
                if !self
                    .store
                    .turn_is_cancelled(&context.node_id, &upload.turn_id)
                    .await
                    .unwrap_or(false) =>
            {
                let _ = self
                    .complete_with_response(&context.node_id, &upload.turn_id, &response)
                    .await;
            }
            Err(error)
                if !self
                    .store
                    .turn_is_cancelled(&context.node_id, &upload.turn_id)
                    .await
                    .unwrap_or(false) =>
            {
                let _ = self.store.update_turn(&context.node_id, &upload.turn_id, NodeTurnState::Error, json!({"state":"error", "code":"processing_failed", "message":error.to_string()})).await;
            }
            _ => {}
        }
    }

    pub async fn process_text_turn(
        &self,
        context: AuthenticatedNodeContext,
        turn_id: String,
        ingress: Arc<dyn NodeConversationIngress>,
    ) {
        if self
            .store
            .begin_turn_processing(&context.node_id, &turn_id)
            .await
            .is_err()
        {
            return;
        }
        let result = async {
            let text = self
                .store
                .turn_input_text(&context.node_id, &turn_id)
                .await?;
            let response = ingress.respond(&context, &text).await?;
            anyhow::ensure!(
                !self
                    .store
                    .turn_is_cancelled(&context.node_id, &turn_id)
                    .await?,
                "turn cancelled"
            );
            self.complete_with_response(&context.node_id, &turn_id, &response)
                .await?;
            anyhow::Ok(())
        }
        .await;
        if let Err(error) = result {
            if !self
                .store
                .turn_is_cancelled(&context.node_id, &turn_id)
                .await
                .unwrap_or(false)
            {
                let _ = self.store.update_turn(
                    &context.node_id,
                    &turn_id,
                    NodeTurnState::Error,
                    json!({"state":"error", "code":"processing_failed", "message":error.to_string()}),
                ).await;
            }
        }
    }

    async fn complete_with_response(
        &self,
        node_id: &str,
        turn_id: &str,
        response: &str,
    ) -> anyhow::Result<()> {
        let output_dir = std::path::PathBuf::from(
            shellexpand::tilde(&self.config.retention.media_dir).into_owned(),
        )
        .join("responses");
        let spoken_response = bounded_spoken_response(response, 170);
        let audio = self
            .speech
            .synthesize(
                &spoken_response,
                &output_dir,
                self.config.limits.response_audio_bytes,
            )
            .await?;
        let mut payload = json!({"state":"complete", "text":response});
        if let Some(artifact) = audio {
            let media_id = self
                .store
                .register_response_media(
                    node_id,
                    turn_id,
                    &artifact.content_type,
                    artifact.size_bytes,
                    &artifact.path.to_string_lossy(),
                )
                .await?;
            payload["audio"] = json!({
                "media_id": media_id,
                "content_type": artifact.content_type,
                "size_bytes": artifact.size_bytes,
                "download_path": format!("/node/v1/media/{media_id}")
            });
        }
        self.store
            .update_turn(node_id, turn_id, NodeTurnState::Complete, payload)
            .await?;
        Ok(())
    }
}

fn bounded_spoken_response(response: &str, maximum_characters: usize) -> String {
    let trimmed = response.trim();
    if trimmed.chars().count() <= maximum_characters {
        return trimmed.to_string();
    }

    let mut excerpt: String = trimmed.chars().take(maximum_characters).collect();
    if let Some(sentence_end) = excerpt
        .char_indices()
        .filter(|(_, character)| matches!(character, '.' | '!' | '?'))
        .map(|(index, character)| index + character.len_utf8())
        .rfind(|index| *index >= maximum_characters / 2)
    {
        excerpt.truncate(sentence_end);
    } else if let Some(word_end) = excerpt.rfind(char::is_whitespace) {
        excerpt.truncate(word_end);
        excerpt.push('.');
    }
    excerpt
}

fn validate_sensor_reading(reading: &SensorReading) -> anyhow::Result<()> {
    anyhow::ensure!(reading.value.is_finite(), "sensor value must be finite");
    anyhow::ensure!(
        reading.capability_version == 1,
        "unsupported sensor version"
    );
    let (expected_unit, minimum, maximum) = match reading.capability_id.as_str() {
        "sensor.environment.temperature" => ("celsius", -40.0, 85.0),
        "sensor.environment.humidity" => ("percent_rh", 0.0, 100.0),
        _ => anyhow::bail!("unsupported sensor capability"),
    };
    anyhow::ensure!(
        reading.unit == expected_unit,
        "sensor unit does not match capability"
    );
    anyhow::ensure!(
        (minimum..=maximum).contains(&reading.value),
        "sensor value is outside the hardware range"
    );
    Ok(())
}

fn ensure_protocol(version: &ProtocolVersion) -> anyhow::Result<()> {
    anyhow::ensure!(
        version.major == NODE_PROTOCOL_MAJOR,
        "unsupported Node Protocol major version"
    );
    Ok(())
}

#[cfg(test)]
mod sensor_tests {
    use super::*;

    #[test]
    fn environmental_sensor_validation_is_bounded() {
        assert!(validate_sensor_reading(&SensorReading {
            capability_id: "sensor.environment.temperature".into(),
            capability_version: 1,
            value: 22.75,
            unit: "celsius".into(),
        })
        .is_ok());
        assert!(validate_sensor_reading(&SensorReading {
            capability_id: "sensor.environment.humidity".into(),
            capability_version: 1,
            value: 101.0,
            unit: "percent_rh".into(),
        })
        .is_err());
        assert!(validate_sensor_reading(&SensorReading {
            capability_id: "sensor.environment.temperature".into(),
            capability_version: 1,
            value: f64::NAN,
            unit: "celsius".into(),
        })
        .is_err());
    }

    #[test]
    fn runtime_recovery_validation_accepts_codes_and_rejects_free_form_text() {
        let mut report = RuntimeRecoveryReport {
            schema: "aidaemon.runtime.recovery.v1".into(),
            reset_reason: "task_watchdog".into(),
            previous_stage: Some("sensor_report".into()),
            previous_uptime_ms: Some(300_000),
            planned_restart_reason: None,
            consecutive_unhealthy_resets: 3,
            connectivity_restart_count: 0,
            safe_mode: true,
            watchdog_ready: true,
            watchdog_timeout_seconds: Some(90),
        };
        assert!(validate_runtime_recovery_report(&report).is_ok());

        report.previous_stage = Some("capturing child conversation".into());
        assert!(validate_runtime_recovery_report(&report).is_err());
        report.previous_stage = Some("turn".into());
        report.watchdog_timeout_seconds = Some(3_601);
        assert!(validate_runtime_recovery_report(&report).is_err());
    }
}
