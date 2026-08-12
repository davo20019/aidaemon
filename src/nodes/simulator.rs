use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature, SigningKey};
use sha2::{Digest, Sha256};

use super::auth;
use super::domain::CapabilityObservation;
use super::protocol::*;

pub struct SimulatorIdentity {
    signing_key: SigningKey,
    pub credential_id: Option<String>,
    pub node_id: Option<String>,
    pub node_session_id: Option<String>,
    pub access_token: Option<String>,
}

impl SimulatorIdentity {
    pub fn generate() -> Self {
        Self {
            signing_key: SigningKey::random(&mut rand::thread_rng()),
            credential_id: None,
            node_id: None,
            node_session_id: None,
            access_token: None,
        }
    }
    fn public_key(&self) -> String {
        URL_SAFE_NO_PAD.encode(
            self.signing_key
                .verifying_key()
                .to_encoded_point(true)
                .as_bytes(),
        )
    }

    pub fn private_key_base64url(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.signing_key.to_bytes())
    }
}

pub struct NodeSimulator {
    base_url: String,
    client: reqwest::Client,
    pub identity: SimulatorIdentity,
}

impl NodeSimulator {
    pub fn new(base_url: impl Into<String>) -> anyhow::Result<Self> {
        Ok(Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(40))
                .build()?,
            identity: SimulatorIdentity::generate(),
        })
    }

    pub async fn enroll(
        &mut self,
        offer_id: &str,
        offer_secret: &str,
    ) -> anyhow::Result<RedeemEnrollmentResponse> {
        self.enroll_as(offer_id, offer_secret, "simulator").await
    }

    pub async fn enroll_as(
        &mut self,
        offer_id: &str,
        offer_secret: &str,
        node_kind: &str,
    ) -> anyhow::Result<RedeemEnrollmentResponse> {
        let response = self
            .client
            .post(format!("{}/node/v1/enrollments/redeem", self.base_url))
            .json(&RedeemEnrollmentRequest {
                protocol: ProtocolVersion::default(),
                offer_id: offer_id.to_string(),
                offer_secret: offer_secret.to_string(),
                node_kind: node_kind.to_string(),
                public_key_sec1: self.identity.public_key(),
                metadata: DeviceMetadata {
                    runtime_version: env!("CARGO_PKG_VERSION").to_string(),
                    firmware_version: "simulator".to_string(),
                    board: None,
                },
                capabilities: vec![CapabilityObservation {
                    capability_id: "input.text".to_string(),
                    version: 1,
                    limits: serde_json::json!({"max_chars":4096}),
                }],
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let response: RedeemEnrollmentResponse = response;
        self.identity.credential_id = Some(response.credential_id.clone());
        self.identity.node_id = Some(response.node_id.clone());
        Ok(response)
    }

    pub async fn open_session(&mut self) -> anyhow::Result<OpenSessionResponse> {
        let credential_id = self
            .identity
            .credential_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("simulator is not enrolled"))?;
        let challenge: SessionChallengeResponse = self
            .client
            .post(format!("{}/node/v1/sessions/challenge", self.base_url))
            .json(&SessionChallengeRequest {
                credential_id: credential_id.clone(),
                protocol: ProtocolVersion::default(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let boot_id = format!("sim-{}", uuid::Uuid::new_v4().simple());
        let canonical = auth::canonical_session_challenge(
            &credential_id,
            &challenge.challenge_id,
            &challenge.nonce,
            1,
            &challenge.instance_id,
            &boot_id,
        );
        let signature: Signature = self.identity.signing_key.sign(&canonical);
        let opened: OpenSessionResponse = self
            .client
            .post(format!("{}/node/v1/sessions", self.base_url))
            .json(&OpenSessionRequest {
                credential_id,
                challenge_id: challenge.challenge_id,
                protocol: ProtocolVersion::default(),
                boot_id,
                signature_der: URL_SAFE_NO_PAD.encode(signature.to_der().as_bytes()),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        self.identity.access_token = Some(opened.access_token.clone());
        self.identity.node_session_id = Some(opened.node_session_id.clone());
        Ok(opened)
    }

    pub async fn heartbeat(
        &self,
        capabilities: Vec<CapabilityObservation>,
    ) -> anyhow::Result<HeartbeatResponse> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        Ok(self
            .client
            .post(format!("{}/node/v1/heartbeats", self.base_url))
            .bearer_auth(token)
            .json(&HeartbeatRequest {
                boot_id: "simulator-runtime".to_string(),
                uptime_ms: 42_000,
                runtime_version: env!("CARGO_PKG_VERSION").to_string(),
                firmware_version: "simulator".to_string(),
                battery_percent: None,
                free_internal_heap: None,
                largest_internal_allocation: None,
                psram_free: None,
                recovery: Some(RuntimeRecoveryReport {
                    schema: "aidaemon.runtime.recovery.v1".to_string(),
                    reset_reason: "software".to_string(),
                    previous_stage: Some("online".to_string()),
                    previous_uptime_ms: Some(41_000),
                    planned_restart_reason: Some("test_restart".to_string()),
                    consecutive_unhealthy_resets: 0,
                    connectivity_restart_count: 0,
                    safe_mode: false,
                    watchdog_ready: true,
                    watchdog_timeout_seconds: Some(90),
                }),
                capabilities,
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn report_sensor_readings(
        &self,
        readings: Vec<SensorReading>,
    ) -> anyhow::Result<ReportSensorReadingsResponse> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        Ok(self
            .client
            .post(format!("{}/node/v1/sensor-readings", self.base_url))
            .bearer_auth(token)
            .json(&ReportSensorReadingsRequest {
                request_id: format!("sensor_{}", uuid::Uuid::new_v4().simple()),
                sample_uptime_ms: 43_000,
                readings,
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn outbox(&self, after: u64) -> anyhow::Result<NodeOutboxResponse> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        Ok(self
            .client
            .get(format!(
                "{}/node/v1/outbox?after={after}&wait_seconds=0",
                self.base_url
            ))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn download_media(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(path.starts_with("/node/v1/media/"), "invalid media path");
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        Ok(self
            .client
            .get(format!("{}{}", self.base_url, path))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec())
    }

    pub async fn download_firmware(&self, path: &str) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            path.starts_with("/node/v1/firmware/") && !path.contains(".."),
            "invalid firmware path"
        );
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        Ok(self
            .client
            .get(format!("{}{}", self.base_url, path))
            .bearer_auth(token)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec())
    }

    pub async fn acknowledge_outbox(
        &self,
        cursor: u64,
        status: NodeOutboxAckStatus,
        detail_code: Option<&str>,
    ) -> anyhow::Result<AckNodeOutboxResponse> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        Ok(self
            .client
            .post(format!("{}/node/v1/outbox/{cursor}/ack", self.base_url))
            .bearer_auth(token)
            .json(&AckNodeOutboxRequest {
                status,
                detail_code: detail_code.map(str::to_string),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn rotate_credential(&mut self) -> anyhow::Result<RotateCredentialResponse> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        let node_id = self
            .identity
            .node_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator is not enrolled"))?;
        let challenge: CredentialRotationChallengeResponse = self
            .client
            .post(format!(
                "{}/node/v1/credentials/rotation-challenge",
                self.base_url
            ))
            .bearer_auth(token)
            .json(&CredentialRotationChallengeRequest {
                request_id: format!("req_{}", uuid::Uuid::new_v4().simple()),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let replacement = SigningKey::random(&mut rand::thread_rng());
        let new_public_key_sec1 = URL_SAFE_NO_PAD.encode(
            replacement
                .verifying_key()
                .to_encoded_point(true)
                .as_bytes(),
        );
        let session_id = self
            .identity
            .node_session_id
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active Node Session"))?;
        let canonical = auth::canonical_credential_rotation(
            node_id,
            session_id,
            &challenge.rotation_id,
            &challenge.nonce,
            &challenge.instance_id,
            &new_public_key_sec1,
        );
        let signature: Signature = replacement.sign(&canonical);
        let response: RotateCredentialResponse = self
            .client
            .post(format!("{}/node/v1/credentials/rotate", self.base_url))
            .bearer_auth(token)
            .json(&RotateCredentialRequest {
                rotation_id: challenge.rotation_id,
                new_public_key_sec1,
                new_key_signature_der: URL_SAFE_NO_PAD.encode(signature.to_der().as_bytes()),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        self.identity.signing_key = replacement;
        self.identity.credential_id = Some(response.credential_id.clone());
        self.identity.node_session_id = None;
        self.identity.access_token = None;
        Ok(response)
    }

    pub async fn text_turn(&self, text: &str) -> anyhow::Result<Vec<TurnEvent>> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
        let created: CreateTurnResponse = self
            .client
            .post(format!("{}/node/v1/turns", self.base_url))
            .bearer_auth(token)
            .header(
                "Idempotency-Key",
                format!("idem_{}", uuid::Uuid::new_v4().simple()),
            )
            .json(&CreateTurnRequest::Text {
                request_id,
                text: text.to_string(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let mut cursor = 0;
        let mut all = Vec::new();
        for _ in 0..12 {
            let response: TurnEventsResponse = self
                .client
                .get(format!(
                    "{}/node/v1/turns/{}/events",
                    self.base_url, created.turn_id
                ))
                .bearer_auth(token)
                .query(&[("after", cursor), ("wait_seconds", 10_u64)])
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            cursor = response.next_cursor;
            let done = response.events.iter().any(|event| {
                matches!(
                    event.event_type.as_str(),
                    "turn.complete" | "turn.error" | "turn.cancelled"
                )
            });
            all.extend(response.events);
            if done {
                return Ok(all);
            }
        }
        anyhow::bail!("turn did not complete before simulator deadline")
    }

    pub async fn media_turn(
        &self,
        bytes: &[u8],
        content_type: &str,
        duration_ms: Option<u64>,
    ) -> anyhow::Result<Vec<TurnEvent>> {
        let token = self
            .identity
            .access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("simulator has no active session"))?;
        let request_id = format!("req_{}", uuid::Uuid::new_v4().simple());
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        let request = if content_type == "audio/wav" {
            CreateTurnRequest::Audio {
                request_id,
                content_type: content_type.to_string(),
                size_bytes: bytes.len() as u64,
                sha256,
                duration_ms: duration_ms.unwrap_or(1000),
            }
        } else {
            CreateTurnRequest::StillImage {
                request_id,
                content_type: content_type.to_string(),
                size_bytes: bytes.len() as u64,
                sha256,
                width: None,
                height: None,
                capture_duration_ms: None,
            }
        };
        let created: CreateTurnResponse = self
            .client
            .post(format!("{}/node/v1/turns", self.base_url))
            .bearer_auth(token)
            .header(
                "Idempotency-Key",
                format!("idem_{}", uuid::Uuid::new_v4().simple()),
            )
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let slot = created
            .upload_slots
            .first()
            .ok_or_else(|| anyhow::anyhow!("Gateway did not return an upload slot"))?;
        self.client
            .put(format!("{}{}", self.base_url, slot.upload_path))
            .bearer_auth(token)
            .header("Content-Type", content_type)
            .body(bytes.to_vec())
            .send()
            .await?
            .error_for_status()?;
        self.client
            .post(format!(
                "{}/node/v1/turns/{}/commit",
                self.base_url, created.turn_id
            ))
            .bearer_auth(token)
            .json(&CommitTurnRequest {
                slot_id: slot.slot_id.clone(),
            })
            .send()
            .await?
            .error_for_status()?;
        self.poll_turn(token, &created.turn_id).await
    }

    async fn poll_turn(&self, token: &str, turn_id: &str) -> anyhow::Result<Vec<TurnEvent>> {
        let mut cursor = 0;
        let mut all = Vec::new();
        for _ in 0..12 {
            let response: TurnEventsResponse = self
                .client
                .get(format!(
                    "{}/node/v1/turns/{}/events",
                    self.base_url, turn_id
                ))
                .bearer_auth(token)
                .query(&[("after", cursor), ("wait_seconds", 10_u64)])
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            cursor = response.next_cursor;
            let done = response.events.iter().any(|event| {
                matches!(
                    event.event_type.as_str(),
                    "turn.complete" | "turn.error" | "turn.cancelled"
                )
            });
            all.extend(response.events);
            if done {
                return Ok(all);
            }
        }
        anyhow::bail!("turn did not complete before simulator deadline")
    }
}
