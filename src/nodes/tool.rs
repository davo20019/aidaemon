use anyhow::Context;
use async_trait::async_trait;
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::SqlitePool;

use crate::traits::{
    EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, Tool, ToolCallMetadata,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolEvidenceCapability,
    ToolMutationEffects, ToolOutcomeStatus, ToolRole, ToolSemanticScope, ToolVerificationMode,
};

const FRESH_READING_SECONDS: i64 = 90;
const FRESH_NODE_SECONDS: i64 = 90;

pub struct ReadNodeSensorsTool {
    pool: SqlitePool,
}

pub struct ReadNodeHealthTool {
    pool: SqlitePool,
}

pub struct SendNodeAudioTool {
    announcements: super::announcement::NodeAnnouncementService,
}

pub struct ManageNodeMonitorsTool {
    monitoring: super::monitoring::NodeMonitoringService,
}

impl ManageNodeMonitorsTool {
    pub fn new(monitoring: super::monitoring::NodeMonitoringService) -> Self {
        Self { monitoring }
    }
}

impl SendNodeAudioTool {
    pub fn new(announcements: super::announcement::NodeAnnouncementService) -> Self {
        Self { announcements }
    }

    async fn execute_delivery(&self, arguments: &str) -> anyhow::Result<(String, String)> {
        let args: SendNodeAudioArgs = serde_json::from_str(arguments)
            .context("send_node_audio arguments must be valid JSON")?;
        anyhow::ensure!(
            args.user_role
                .as_deref()
                .is_some_and(|role| role.eq_ignore_ascii_case("owner")),
            "Only the AIdaemon owner may send Node audio"
        );
        let delivery = self
            .announcements
            .queue_and_wait(args.node.as_deref(), &args.text)
            .await?;
        let now = Utc::now();
        let last_seen_age_seconds = delivery
            .queued
            .last_seen_at
            .map(|seen| (now - seen).num_seconds().max(0));
        let (status, acknowledged_at, detail_code) = delivery
            .receipt
            .as_ref()
            .map(|receipt| {
                (
                    receipt.status.as_str(),
                    Some(receipt.acknowledged_at.to_rfc3339()),
                    receipt.detail_code.as_deref(),
                )
            })
            .unwrap_or(("queued", None, None));
        let output = serde_json::to_string_pretty(&json!({
            "node": delivery.queued.display_name,
            "delivery_id": delivery.queued.cursor,
            "delivery_status": status,
            "acknowledged_at": acknowledged_at,
            "detail_code": detail_code,
            "audio_bytes": delivery.queued.size_bytes,
            "expires_at": delivery.queued.expires_at.to_rfc3339(),
            "node_last_seen_age_seconds": last_seen_age_seconds,
            "interpretation": if status == "played" {
                "The Device acknowledged completed playback."
            } else if status == "queued" {
                "The announcement is queued but playback has not been acknowledged; do not claim that it played."
            } else {
                "The Device acknowledged the announcement without successful playback."
            }
        }))?;
        Ok((output, status.to_string()))
    }
}

fn node_evidence(purposes: &[EvidencePurpose]) -> Vec<ToolEvidenceCapability> {
    vec![ToolEvidenceCapability::new(
        ToolSemanticScope::ExternalRemote,
        purposes,
        EvidenceAuthority::Direct,
        EvidenceTemporalScope::Current,
    )]
}

fn node_observation_semantics(purposes: &[EvidencePurpose]) -> ToolCallSemantics {
    ToolCallSemantics::observation()
        .with_verification_mode(ToolVerificationMode::ResultContent)
        .with_evidence(node_evidence(purposes))
}

fn node_audio_receipt_semantics() -> ToolCallSemantics {
    ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY)
        .with_verification_mode(ToolVerificationMode::ResultContent)
        .with_evidence(node_evidence(&[EvidencePurpose::Outcome]))
}

fn node_audio_outcome_status(status: &str) -> ToolOutcomeStatus {
    match status {
        "played" => ToolOutcomeStatus::Succeeded,
        // Queuing changed outbox state, but it did not establish completed
        // playback and should remain an explicit incomplete domain result.
        "queued" => ToolOutcomeStatus::CompletedWithNegativeResult,
        // The Device explicitly rejected or failed playback. Retrying the same
        // non-idempotent delivery automatically would be unsafe.
        "failed" | "dismissed" => ToolOutcomeStatus::FailedPermanent,
        _ => ToolOutcomeStatus::FailedPermanent,
    }
}

impl ReadNodeSensorsTool {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

impl ReadNodeHealthTool {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[derive(Debug, Deserialize)]
struct ReadNodeSensorsArgs {
    #[serde(default)]
    node: Option<String>,
    #[serde(default, rename = "_user_role")]
    user_role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReadNodeHealthArgs {
    #[serde(default)]
    node: Option<String>,
    #[serde(default, rename = "_user_role")]
    user_role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SendNodeAudioArgs {
    text: String,
    #[serde(default)]
    node: Option<String>,
    #[serde(default, rename = "_user_role")]
    user_role: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ManageNodeMonitorsArgs {
    action: String,
    #[serde(default)]
    monitor_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    node: Option<String>,
    #[serde(default)]
    capability_id: Option<String>,
    #[serde(default)]
    comparison: Option<String>,
    #[serde(default)]
    threshold: Option<f64>,
    #[serde(default)]
    clear_threshold: Option<f64>,
    #[serde(default)]
    duration_seconds: Option<u64>,
    #[serde(default)]
    stale_after_seconds: Option<u64>,
    #[serde(default)]
    offline_after_seconds: Option<u64>,
    #[serde(default)]
    repeat_seconds: Option<u64>,
    #[serde(default)]
    send_recovery: Option<bool>,
    #[serde(default)]
    duration_minutes: Option<u64>,
    #[serde(default)]
    mandate_id: Option<String>,
    #[serde(default)]
    since_hours: Option<u64>,
    #[serde(default)]
    limit: Option<u32>,
    #[serde(default, rename = "_user_role")]
    user_role: Option<String>,
    #[serde(default, rename = "_channel_visibility")]
    channel_visibility: Option<String>,
    #[serde(default, rename = "_session_id")]
    session_id: Option<String>,
}

fn required_arg<'a>(value: Option<&'a str>, name: &str) -> anyhow::Result<&'a str> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{name} is required"))
}

fn monitor_mandate_for_action(
    action: &str,
    mandate_id: Option<String>,
) -> anyhow::Result<Option<String>> {
    match action {
        // Standalone creation is intentionally robust against an LLM copying a
        // nearby goal, requirement, or mandate identifier into an optional
        // field. Lifecycle binding requires the distinct explicit action.
        "create" => Ok(None),
        "create_linked" => Ok(Some(
            required_arg(mandate_id.as_deref(), "mandate_id")?.to_string(),
        )),
        _ => Ok(mandate_id),
    }
}

#[async_trait]
impl Tool for ReadNodeSensorsTool {
    fn name(&self) -> &str {
        "read_node_sensors"
    }

    fn description(&self) -> &str {
        "Read the latest authenticated environmental sensor readings reported by an enrolled AIdaemon Node. Use when the owner asks for the K10 temperature or humidity."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_node_sensors",
            "description": "Read the latest server-received temperature and humidity values from an enrolled AIdaemon Node, including reading age. This reads stored sensor state; it does not activate a microphone, camera, or actuator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "description": "Optional Node display name. Omit when exactly one active Node is enrolled."
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        node_observation_semantics(&[EvidencePurpose::CurrentState, EvidencePurpose::Content])
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ReadNodeSensorsArgs = serde_json::from_str(if arguments.trim().is_empty() {
            "{}"
        } else {
            arguments
        })
        .context("read_node_sensors arguments must be valid JSON")?;
        anyhow::ensure!(
            args.user_role.as_deref() == Some("Owner"),
            "Only the AIdaemon owner may read Node sensor values"
        );
        let readings =
            super::store::latest_sensor_readings(&self.pool, args.node.as_deref()).await?;
        anyhow::ensure!(
            !readings.is_empty(),
            "The selected Node has not reported any sensor readings yet"
        );
        let now = Utc::now();
        let mut output_readings = Vec::with_capacity(readings.len());
        let mut all_fresh = true;
        for reading in &readings {
            let age_seconds = (now - reading.received_at).num_seconds().max(0);
            all_fresh &= age_seconds <= FRESH_READING_SECONDS;
            let mut rendered = json!({
                "capability_id": reading.capability_id,
                "capability_version": reading.capability_version,
                "value": reading.value,
                "unit": reading.unit,
                "reading_age_seconds": age_seconds,
                "server_received_at": reading.received_at.to_rfc3339(),
                "sample_uptime_ms": reading.sample_uptime_ms,
            });
            if reading.capability_id == "sensor.environment.temperature"
                && reading.unit == "celsius"
            {
                rendered["fahrenheit"] = json!(reading.value.mul_add(1.8, 32.0));
            }
            output_readings.push(rendered);
        }
        let last_seen_age_seconds = readings[0]
            .node_last_seen_at
            .map(|seen| (now - seen).num_seconds().max(0));
        Ok(serde_json::to_string_pretty(&json!({
            "node": readings[0].display_name,
            "freshness": if all_fresh { "fresh" } else { "stale" },
            "freshness_threshold_seconds": FRESH_READING_SECONDS,
            "node_last_seen_age_seconds": last_seen_age_seconds,
            "readings": output_readings,
            "interpretation": "These are the latest Device-reported ambient readings. A stale reading must be described as stale, not current."
        }))?)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        Ok(ToolCallOutcome {
            output: self.call(arguments).await?,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: self.call_semantics(arguments),
                ..ToolCallMetadata::default()
            },
        })
    }
}

#[async_trait]
impl Tool for ReadNodeHealthTool {
    fn name(&self) -> &str {
        "read_node_health"
    }

    fn description(&self) -> &str {
        "Read an enrolled AIdaemon Node's latest authenticated heartbeat, reported capabilities, current gateway authorizations, and bounded Runtime recovery evidence."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "read_node_health",
            "description": "Inspect stored Node connection freshness, Runtime/firmware versions, reported protocol capabilities and limits, current gateway authorizations, bounded resource health, and the most recent Runtime recovery report. Use this before claiming that a Companion control is supported or unavailable. This is evidence reported during an authenticated Node Session; it cannot contact or restart an offline Device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "description": "Optional Node display name. Omit when exactly one active Node is enrolled."
                    }
                },
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        node_observation_semantics(&[EvidencePurpose::CurrentState, EvidencePurpose::Content])
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ReadNodeHealthArgs = serde_json::from_str(if arguments.trim().is_empty() {
            "{}"
        } else {
            arguments
        })
        .context("read_node_health arguments must be valid JSON")?;
        anyhow::ensure!(
            args.user_role.as_deref() == Some("Owner"),
            "Only the AIdaemon owner may read Node health"
        );
        let health = super::store::node_health_snapshot(&self.pool, args.node.as_deref()).await?;
        let now = Utc::now();
        let last_seen_age_seconds = health
            .last_seen_at
            .map(|seen| (now - seen).num_seconds().max(0));
        let connection_state = match last_seen_age_seconds {
            Some(age) if age <= FRESH_NODE_SECONDS => "recently_connected",
            Some(_) => "stale",
            None => "never_connected",
        };
        Ok(serde_json::to_string_pretty(&json!({
            "node": health.display_name,
            "connection_state": connection_state,
            "freshness_threshold_seconds": FRESH_NODE_SECONDS,
            "last_seen_age_seconds": last_seen_age_seconds,
            "last_seen_at": health.last_seen_at.map(|value| value.to_rfc3339()),
            "runtime_version": health.runtime_version,
            "firmware_version": health.firmware_version,
            "uptime_ms": health.uptime_ms,
            "free_internal_heap": health.free_internal_heap,
            "largest_internal_allocation": health.largest_internal_allocation,
            "psram_free": health.psram_free,
            "recovery": health.recovery,
            "reported_capabilities": health.capabilities,
            "current_authorizations": health.authorizations,
            "interpretation": "This is the latest authenticated Runtime report stored by AIdaemon. Capability absence means the Node did not report that protocol control; it does not prove the physical hardware lacks it. Authorization absence means the gateway has not granted that action. A stale report does not prove the Device is currently reachable, and a recovery report does not prove physical hardware health."
        }))?)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        Ok(ToolCallOutcome {
            output: self.call(arguments).await?,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                semantics: self.call_semantics(arguments),
                ..ToolCallMetadata::default()
            },
        })
    }
}

#[async_trait]
impl Tool for SendNodeAudioTool {
    fn name(&self) -> &str {
        "send_node_audio"
    }

    fn description(&self) -> &str {
        "Deliver a short owner-requested spoken announcement to an enrolled, explicitly authorized audio-capable Node. Use when the owner asks AIdaemon to tell or announce something on a Companion."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "send_node_audio",
            "description": "Generate and deliver a short spoken announcement to an enrolled Node. This is owner-only external delivery; use only when the owner explicitly asks to play or say something on a Node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact short message to speak. Do not add private context the owner did not request."
                    },
                    "node": {
                        "type": "string",
                        "description": "Exact Node display name. Omit only when one eligible Node exists."
                    }
                },
                "required": ["text"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        node_audio_receipt_semantics()
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self.execute_delivery(arguments).await?.0)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<crate::types::StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let (output, status) = self.execute_delivery(arguments).await?;
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(node_audio_outcome_status(&status)),
                semantics: node_audio_receipt_semantics(),
                ..ToolCallMetadata::default()
            },
        })
    }
}

#[async_trait]
impl Tool for ManageNodeMonitorsTool {
    fn name(&self) -> &str {
        "manage_node_monitors"
    }

    fn description(&self) -> &str {
        "Create and govern explicit, time-bounded temperature or humidity monitors for the owner; a direct owner request needs no separate monitoring authorization or mandate."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_node_monitors",
            "description": "Create, list, inspect, pause, resume, or cancel deterministic environmental monitors. Create only after an explicit owner request in a private Channel. A direct owner request is sufficient: do not create a mandate or a separate monitoring authorization. The Node's existing report_sensor grant is the only Node authorization required. Every monitor expires. V1 supports temperature and humidity only; it never activates a camera, microphone, or actuator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "create_linked", "list", "get", "history", "pause", "resume", "cancel"],
                        "description": "Use create for normal direct owner requests. Use create_linked only when the current owner message explicitly requests linkage to an exact existing mandate ID."
                    },
                    "monitor_id": {"type": "string", "description": "Required for get, history, pause, resume, and cancel."},
                    "name": {"type": "string", "description": "Create only: short owner-facing monitor name."},
                    "node": {"type": "string", "description": "Create only: exact Node display name. Omit only when one eligible Node exists."},
                    "capability_id": {
                        "type": "string",
                        "enum": ["sensor.environment.temperature", "sensor.environment.humidity"],
                        "description": "Create only."
                    },
                    "comparison": {"type": "string", "enum": ["above", "below"], "description": "Create only."},
                    "threshold": {"type": "number", "description": "Create only: Celsius or percent relative humidity alert point."},
                    "clear_threshold": {"type": "number", "description": "Create only: hysteresis recovery point; <= threshold for above, >= for below."},
                    "duration_seconds": {"type": "integer", "minimum": 0, "maximum": 86400, "description": "Create only: continuous crossing time before alerting."},
                    "stale_after_seconds": {"type": "integer", "minimum": 30, "maximum": 86400, "description": "Create only: optional missing-reading alert interval."},
                    "offline_after_seconds": {"type": "integer", "minimum": 30, "maximum": 86400, "description": "Create only: optional Node check-in alert interval. Values shorter than stale_after_seconds are safely normalized to stale_after_seconds."},
                    "repeat_seconds": {"type": "integer", "minimum": 0, "maximum": 604800, "description": "Create only: 0 sends one alert per incident; otherwise repeat interval."},
                    "send_recovery": {"type": "boolean", "description": "Create only: send an alert when the condition recovers."},
                    "duration_minutes": {"type": "integer", "minimum": 1, "description": "Create only: required monitor lifetime."},
                    "mandate_id": {"type": "string", "description": "Required only for create_linked and ignored by ordinary create. Use create_linked only when the current owner message explicitly names an exact existing, confirmed, active mandate ID. Never use a goal ID, requirement ID, inferred ID, or a mandate created merely to authorize monitoring; direct owner requests need no mandate."},
                    "since_hours": {"type": "integer", "minimum": 1, "maximum": 720, "description": "History only; defaults to 24."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "description": "History only; defaults to 100."}
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Management
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
        match action.as_deref() {
            Some("list" | "get" | "history") => ToolCallSemantics::observation(),
            _ => ToolCallSemantics::mutation_with(ToolMutationEffects::CONFIGURATION),
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageNodeMonitorsArgs = serde_json::from_str(arguments)
            .context("manage_node_monitors arguments must be valid JSON")?;
        anyhow::ensure!(
            args.user_role
                .as_deref()
                .is_some_and(|role| role.eq_ignore_ascii_case("owner")),
            "Only the AIdaemon owner may manage Node monitors"
        );
        anyhow::ensure!(
            args.channel_visibility
                .as_deref()
                .is_some_and(|visibility| visibility.eq_ignore_ascii_case("private")),
            "Node monitors may only be managed from a private Channel"
        );
        let session_id = required_arg(args.session_id.as_deref(), "owner session")?;
        let output = match args.action.as_str() {
            "create" | "create_linked" => {
                let mandate_id = monitor_mandate_for_action(&args.action, args.mandate_id)?;
                let request = super::monitoring::CreateNodeMonitor {
                    name: required_arg(args.name.as_deref(), "name")?.to_string(),
                    owner_session_id: session_id.to_string(),
                    node: args.node,
                    capability_id: required_arg(args.capability_id.as_deref(), "capability_id")?
                        .to_string(),
                    comparison: super::monitoring::MonitorComparison::parse(required_arg(
                        args.comparison.as_deref(),
                        "comparison",
                    )?)?,
                    threshold: args.threshold.context("threshold is required")?,
                    clear_threshold: args
                        .clear_threshold
                        .context("clear_threshold is required")?,
                    duration_seconds: args.duration_seconds.unwrap_or(0),
                    stale_after_seconds: args.stale_after_seconds,
                    offline_after_seconds: args.offline_after_seconds,
                    repeat_seconds: args.repeat_seconds.unwrap_or(0),
                    send_recovery: args.send_recovery.unwrap_or(true),
                    duration_minutes: args
                        .duration_minutes
                        .context("duration_minutes is required")?,
                    mandate_id,
                };
                json!({"monitor": self.monitoring.create(request).await?})
            }
            "list" => json!({"monitors": self.monitoring.list(session_id).await?}),
            "get" => {
                let monitor_id = required_arg(args.monitor_id.as_deref(), "monitor_id")?;
                json!({"monitor": self.monitoring.get(monitor_id, session_id).await?})
            }
            "history" => {
                let monitor_id = required_arg(args.monitor_id.as_deref(), "monitor_id")?;
                let since_hours = args.since_hours.unwrap_or(24);
                let limit = args.limit.unwrap_or(100);
                json!({
                    "monitor": self.monitoring.get(monitor_id, session_id).await?,
                    "events": self.monitoring.history(
                        monitor_id,
                        session_id,
                        since_hours,
                        limit,
                    ).await?,
                    "sensor_readings": self.monitoring.sensor_history(
                        monitor_id,
                        session_id,
                        since_hours,
                        limit,
                    ).await?,
                })
            }
            "pause" | "resume" | "cancel" => {
                let monitor_id = required_arg(args.monitor_id.as_deref(), "monitor_id")?;
                json!({"monitor": self.monitoring.change_status(monitor_id, session_id, &args.action).await?})
            }
            other => anyhow::bail!("unknown manage_node_monitors action `{other}`"),
        };
        Ok(serde_json::to_string_pretty(&output)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn played_audio_receipt_is_authoritative_delivery_and_outcome_evidence() {
        assert_eq!(
            node_audio_outcome_status("played"),
            ToolOutcomeStatus::Succeeded
        );
        let semantics = node_audio_receipt_semantics();
        assert!(semantics.observes_state());
        assert!(semantics.mutates_state());
        assert!(semantics
            .mutation_effects
            .contains(ToolMutationEffects::EXTERNAL_DELIVERY));
        assert!(semantics.can_verify_with_result_content());
        assert!(semantics.evidence.iter().any(|capability| {
            capability.scope == ToolSemanticScope::ExternalRemote
                && capability.purposes.contains(&EvidencePurpose::Outcome)
                && capability.authority == EvidenceAuthority::Direct
        }));
    }

    #[test]
    fn incomplete_audio_receipts_do_not_satisfy_completed_playback() {
        assert_eq!(
            node_audio_outcome_status("queued"),
            ToolOutcomeStatus::CompletedWithNegativeResult
        );
        for status in ["failed", "dismissed"] {
            assert_eq!(
                node_audio_outcome_status(status),
                ToolOutcomeStatus::FailedPermanent
            );
        }
    }

    #[tokio::test]
    async fn rejects_non_owner_before_reading_node_state() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let tool = ReadNodeSensorsTool::new(pool.clone());
        let error = tool.call(r#"{"_user_role":"Public"}"#).await.unwrap_err();
        assert!(error.to_string().contains("Only the AIdaemon owner"));
        let health_tool = ReadNodeHealthTool::new(pool);
        let error = health_tool
            .call(r#"{"_user_role":"Public"}"#)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Only the AIdaemon owner"));
    }

    #[tokio::test]
    async fn audio_delivery_is_owner_only_and_disabled_by_default() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let store = std::sync::Arc::new(super::super::store::NodeStore::new(pool, [9_u8; 32]));
        let announcements = super::super::announcement::NodeAnnouncementService::new(
            store,
            crate::config::NodesConfig::default(),
            std::sync::Arc::new(super::super::speech::DisabledSpeech),
        );
        let tool = SendNodeAudioTool::new(announcements);

        let non_owner = tool
            .call(r#"{"text":"hello","_user_role":"Public"}"#)
            .await
            .unwrap_err();
        assert!(non_owner.to_string().contains("Only the AIdaemon owner"));

        let disabled = tool
            .call(r#"{"text":"hello","_user_role":"Owner"}"#)
            .await
            .unwrap_err();
        assert!(disabled
            .to_string()
            .contains("Node audio announcements are disabled"));
    }

    #[tokio::test]
    async fn monitor_management_requires_owner_and_private_channel() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let service = super::super::monitoring::NodeMonitoringService::new(
            pool,
            crate::config::NodeMonitoringConfig::default(),
        );
        let tool = ManageNodeMonitorsTool::new(service);

        let non_owner = tool
            .call(r#"{"action":"list","_user_role":"Public","_channel_visibility":"private","_session_id":"public-session"}"#)
            .await
            .unwrap_err();
        assert!(non_owner.to_string().contains("Only the AIdaemon owner"));

        let public_channel = tool
            .call(r#"{"action":"list","_user_role":"Owner","_channel_visibility":"public_external","_session_id":"owner-session"}"#)
            .await
            .unwrap_err();
        assert!(public_channel.to_string().contains("private Channel"));

        let disabled = tool
            .call(r#"{"action":"list","_user_role":"Owner","_channel_visibility":"private","_session_id":"owner-session"}"#)
            .await
            .unwrap_err();
        assert!(disabled.to_string().contains("monitoring is disabled"));
    }

    #[tokio::test]
    async fn monitor_schema_tells_agents_not_to_invent_mandate_authorization() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let service = super::super::monitoring::NodeMonitoringService::new(
            pool,
            crate::config::NodeMonitoringConfig::default(),
        );
        let tool = ManageNodeMonitorsTool::new(service);
        let schema = tool.schema();

        let description = schema["description"].as_str().unwrap();
        assert!(description.contains("direct owner request is sufficient"));
        assert!(description.contains("do not create a mandate"));
        assert!(description.contains("report_sensor"));
        let mandate_description = schema["parameters"]["properties"]["mandate_id"]["description"]
            .as_str()
            .unwrap();
        assert!(mandate_description.contains("Never use a goal ID"));
        assert!(mandate_description.contains("direct owner requests need no mandate"));
    }

    #[test]
    fn standalone_monitor_creation_ignores_inferred_mandate_ids() {
        assert_eq!(
            monitor_mandate_for_action("create", Some("not-a-mandate".to_string())).unwrap(),
            None
        );

        let error = monitor_mandate_for_action("create_linked", None).unwrap_err();
        assert!(error.to_string().contains("mandate_id is required"));
        assert_eq!(
            monitor_mandate_for_action("create_linked", Some("explicit-mandate".to_string()))
                .unwrap(),
            Some("explicit-mandate".to_string())
        );
    }
}
