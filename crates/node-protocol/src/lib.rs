use serde::{Deserialize, Serialize};

pub const NODE_PROTOCOL_MAJOR: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}
impl Default for ProtocolVersion {
    fn default() -> Self {
        Self { major: 1, minor: 0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityObservation {
    pub capability_id: String,
    pub version: u16,
    #[serde(default)]
    pub limits: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeTurnState {
    Accepted,
    Thinking,
    Complete,
    Cancelled,
    Error,
}
impl NodeTurnState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Thinking => "thinking",
            Self::Complete => "complete",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetadata {
    pub runtime_version: String,
    pub firmware_version: String,
    #[serde(default)]
    pub board: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedeemEnrollmentRequest {
    pub protocol: ProtocolVersion,
    pub offer_id: String,
    pub offer_secret: String,
    pub node_kind: String,
    pub public_key_sec1: String,
    pub metadata: DeviceMetadata,
    #[serde(default)]
    pub capabilities: Vec<CapabilityObservation>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedeemEnrollmentResponse {
    pub node_id: String,
    pub credential_id: String,
    pub node_channel_id: String,
    pub policy_profile: String,
    pub public_key_fingerprint: String,
    pub protocol: ProtocolVersion,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionChallengeRequest {
    pub credential_id: String,
    pub protocol: ProtocolVersion,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionChallengeResponse {
    pub challenge_id: String,
    pub nonce: String,
    pub instance_id: String,
    pub expires_in_seconds: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenSessionRequest {
    pub credential_id: String,
    pub challenge_id: String,
    pub protocol: ProtocolVersion,
    pub boot_id: String,
    pub signature_der: String,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct OpenSessionResponse {
    pub node_session_id: String,
    pub access_token: String,
    pub expires_in_seconds: u64,
    pub policy_profile: String,
    pub policy_revision: u64,
}
impl std::fmt::Debug for OpenSessionResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenSessionResponse")
            .field("node_session_id", &self.node_session_id)
            .field("access_token", &"[REDACTED]")
            .field("expires_in_seconds", &self.expires_in_seconds)
            .field("policy_profile", &self.policy_profile)
            .field("policy_revision", &self.policy_revision)
            .finish()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRotationChallengeRequest {
    pub request_id: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRotationChallengeResponse {
    pub rotation_id: String,
    pub nonce: String,
    pub instance_id: String,
    pub expires_in_seconds: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotateCredentialRequest {
    pub rotation_id: String,
    pub new_public_key_sec1: String,
    pub new_key_signature_der: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotateCredentialResponse {
    pub credential_id: String,
    pub public_key_fingerprint: String,
    pub sessions_closed: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRecoveryReport {
    pub schema: String,
    pub reset_reason: String,
    #[serde(default)]
    pub previous_stage: Option<String>,
    #[serde(default)]
    pub previous_uptime_ms: Option<u64>,
    #[serde(default)]
    pub planned_restart_reason: Option<String>,
    #[serde(default)]
    pub consecutive_unhealthy_resets: u16,
    #[serde(default)]
    pub connectivity_restart_count: u16,
    #[serde(default)]
    pub safe_mode: bool,
    #[serde(default)]
    pub watchdog_ready: bool,
    #[serde(default)]
    pub watchdog_timeout_seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub boot_id: String,
    pub uptime_ms: u64,
    pub runtime_version: String,
    pub firmware_version: String,
    #[serde(default)]
    pub battery_percent: Option<u8>,
    #[serde(default)]
    pub free_internal_heap: Option<u64>,
    #[serde(default)]
    pub largest_internal_allocation: Option<u64>,
    #[serde(default)]
    pub psram_free: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery: Option<RuntimeRecoveryReport>,
    #[serde(default)]
    pub capabilities: Vec<CapabilityObservation>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub accepted: bool,
    pub next_heartbeat_seconds: u64,
    pub policy_revision: u64,
    pub authorization_revision: u64,
    /// Zero means the server-side announcement feature is disabled or this
    /// Node is not authorized to receive announcements.
    #[serde(default)]
    pub outbox_poll_seconds: u64,
    /// Present only when OTA is globally enabled, the Node advertises the
    /// matching capability, and the Node has an explicit receive_ota grant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firmware_update: Option<FirmwareUpdateOffer>,
    /// Ordered HTTPS transport-adapter origins for the same Node Gateway.
    /// An empty list leaves the Device's current list unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gateway_endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FirmwareReleaseManifest {
    pub schema: String,
    pub release_id: String,
    pub version: String,
    pub board: String,
    pub sequence: u64,
    pub size_bytes: u64,
    pub sha256: String,
    pub signature_der: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FirmwareUpdateOffer {
    pub schema: String,
    pub release_id: String,
    pub version: String,
    pub board: String,
    pub sequence: u64,
    pub size_bytes: u64,
    pub sha256: String,
    pub signature_der: String,
    pub download_path: String,
}

impl FirmwareReleaseManifest {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        firmware_release_canonical(
            &self.schema,
            &self.release_id,
            &self.version,
            &self.board,
            self.sequence,
            self.size_bytes,
            &self.sha256,
        )
    }
}

pub fn firmware_release_canonical(
    schema: &str,
    release_id: &str,
    version: &str,
    board: &str,
    sequence: u64,
    size_bytes: u64,
    sha256: &str,
) -> Vec<u8> {
    format!(
        "AIDAEMON-FIRMWARE-RELEASE-V1\n{schema}\n{release_id}\n{version}\n{board}\n{sequence}\n{size_bytes}\n{sha256}"
    )
    .into_bytes()
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SensorReading {
    pub capability_id: String,
    pub capability_version: u16,
    pub value: f64,
    pub unit: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReportSensorReadingsRequest {
    pub request_id: String,
    pub sample_uptime_ms: u64,
    pub readings: Vec<SensorReading>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportSensorReadingsResponse {
    pub accepted: bool,
    pub received_at: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "input_kind", rename_all = "snake_case")]
pub enum CreateTurnRequest {
    Text {
        request_id: String,
        text: String,
    },
    Audio {
        request_id: String,
        content_type: String,
        size_bytes: u64,
        sha256: String,
        duration_ms: u64,
    },
    StillImage {
        request_id: String,
        content_type: String,
        size_bytes: u64,
        sha256: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        width: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        height: Option<u16>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        capture_duration_ms: Option<u64>,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadSlot {
    pub slot_id: String,
    pub content_type: String,
    pub max_bytes: u64,
    pub upload_path: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateTurnResponse {
    pub turn_id: String,
    pub state: NodeTurnState,
    pub cursor: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub upload_slots: Vec<UploadSlot>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvent {
    pub cursor: u64,
    pub turn_id: String,
    pub event_type: String,
    pub created_at: String,
    pub payload: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEventsResponse {
    pub events: Vec<TurnEvent>,
    pub next_cursor: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeOutboxEvent {
    pub cursor: u64,
    pub event_type: String,
    pub created_at: String,
    pub expires_at: String,
    pub payload: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeOutboxResponse {
    pub events: Vec<NodeOutboxEvent>,
    pub next_cursor: u64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeOutboxAckStatus {
    Played,
    Failed,
    Dismissed,
}
impl NodeOutboxAckStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Played => "played",
            Self::Failed => "failed",
            Self::Dismissed => "dismissed",
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AckNodeOutboxRequest {
    pub status: NodeOutboxAckStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail_code: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AckNodeOutboxResponse {
    pub accepted: bool,
    pub status: NodeOutboxAckStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitTurnResponse {
    pub turn_id: String,
    pub state: NodeTurnState,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitTurnRequest {
    pub slot_id: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadMediaResponse {
    pub slot_id: String,
    pub received_bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelTurnResponse {
    pub turn_id: String,
    pub state: NodeTurnState,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolErrorBody {
    pub error: ProtocolError,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolError {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn golden_fixtures_match_models() {
        let enrollment: RedeemEnrollmentRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/enrollment.json"
        ))
        .unwrap();
        assert_eq!(enrollment.node_kind, "simulator");
        let heartbeat: HeartbeatRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/heartbeat.json"
        ))
        .unwrap();
        assert_eq!(heartbeat.capabilities.len(), 4);
        let recovery = heartbeat.recovery.expect("fixture recovery report");
        assert_eq!(recovery.schema, "aidaemon.runtime.recovery.v1");
        assert_eq!(recovery.reset_reason, "software");
        assert_eq!(recovery.previous_stage.as_deref(), Some("online"));
        let sensors: ReportSensorReadingsRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/sensor-readings.json"
        ))
        .unwrap();
        assert_eq!(sensors.readings.len(), 2);
        let audio: CreateTurnRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/audio-turn.json"
        ))
        .unwrap();
        assert!(matches!(audio, CreateTurnRequest::Audio { .. }));
        let image: CreateTurnRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/still-image-turn.json"
        ))
        .unwrap();
        assert!(matches!(image, CreateTurnRequest::StillImage { .. }));
        let rotation: RotateCredentialRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/credential-rotation.json"
        ))
        .unwrap();
        assert_eq!(rotation.rotation_id, "rotation_synthetic_01");
        let outbox_ack: AckNodeOutboxRequest = serde_json::from_str(include_str!(
            "../../../tests/fixtures/node-protocol-v1/outbox-ack.json"
        ))
        .unwrap();
        assert_eq!(outbox_ack.status, NodeOutboxAckStatus::Played);
    }
}
