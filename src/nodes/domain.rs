use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use aidaemon_node_protocol::{CapabilityObservation, NodeTurnState};

pub const NODE_PROTOCOL_MAJOR: u16 = aidaemon_node_protocol::NODE_PROTOCOL_MAJOR;
pub const CHILD_COMPANION_POLICY: &str = "child_companion_v1";
pub const NODE_CHANNEL_NAME: &str = "node";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeRecord {
    pub node_id: String,
    pub owner_id: String,
    pub kind: String,
    pub display_name: String,
    pub policy_profile: String,
    pub policy_revision: u64,
    pub authorization_revision: u64,
    pub conversation_session_id: String,
    pub node_channel_id: String,
    pub created_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub last_seen_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthenticatedNodeContext {
    pub node_id: String,
    pub node_session_id: String,
    pub credential_id: String,
    pub kind: String,
    pub policy_profile: String,
    pub policy_revision: u64,
    pub authorization_revision: u64,
    pub conversation_session_id: String,
    pub node_channel_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeAction {
    SubmitTextTurn,
    SubmitAudioTurn,
    SubmitStillImage,
    ReportTelemetry,
    ReportSensor,
    ReceiveAudio,
    ReceiveDisplayCommand,
    ReceiveActuatorCommand,
    ReceiveOta,
}

impl NodeAction {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        Ok(match value {
            "submit_text_turn" => Self::SubmitTextTurn,
            "submit_audio_turn" => Self::SubmitAudioTurn,
            "submit_still_image" => Self::SubmitStillImage,
            "report_telemetry" => Self::ReportTelemetry,
            "report_sensor" => Self::ReportSensor,
            "receive_audio" => Self::ReceiveAudio,
            "receive_display_command" => Self::ReceiveDisplayCommand,
            "receive_actuator_command" => Self::ReceiveActuatorCommand,
            "receive_ota" => Self::ReceiveOta,
            _ => anyhow::bail!("unknown Node action: {value}"),
        })
    }
}

pub fn validate_node_kind(value: &str) -> anyhow::Result<()> {
    validate_identifier("node kind", value, 1, 48)
}

pub fn validate_policy_profile(value: &str) -> anyhow::Result<()> {
    validate_identifier("policy profile", value, 1, 64)
}

pub fn validate_identifier(label: &str, value: &str, min: usize, max: usize) -> anyhow::Result<()> {
    let len = value.len();
    anyhow::ensure!(
        (min..=max).contains(&len),
        "{label} must be between {min} and {max} bytes"
    );
    anyhow::ensure!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':')),
        "{label} contains unsupported characters"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_bounded_and_structural() {
        assert!(validate_node_kind("k10").is_ok());
        assert!(validate_node_kind("robot.arm-v2").is_ok());
        assert!(validate_node_kind("").is_err());
        assert!(validate_node_kind("k10/../../owner").is_err());
        assert!(validate_policy_profile(CHILD_COMPANION_POLICY).is_ok());
    }
}
