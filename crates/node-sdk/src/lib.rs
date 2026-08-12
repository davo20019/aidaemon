use aidaemon_node_protocol::*;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature, SigningKey};

pub use aidaemon_node_protocol as protocol;

pub struct NodeIdentity {
    signing_key: SigningKey,
    pub node_id: Option<String>,
    pub credential_id: Option<String>,
}

impl NodeIdentity {
    pub fn generate() -> Self {
        Self {
            signing_key: SigningKey::random(&mut rand::thread_rng()),
            node_id: None,
            credential_id: None,
        }
    }
    pub fn from_private_key(bytes: &[u8; 32]) -> anyhow::Result<Self> {
        Ok(Self {
            signing_key: SigningKey::from_bytes(bytes.into())?,
            node_id: None,
            credential_id: None,
        })
    }
    pub fn private_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes().into()
    }
    pub fn public_key_base64url(&self) -> String {
        URL_SAFE_NO_PAD.encode(
            self.signing_key
                .verifying_key()
                .to_encoded_point(true)
                .as_bytes(),
        )
    }
    fn sign(&self, message: &[u8]) -> String {
        let signature: Signature = self.signing_key.sign(message);
        URL_SAFE_NO_PAD.encode(signature.to_der().as_bytes())
    }
}

pub struct NodeClient {
    gateway: String,
    http: reqwest::Client,
    pub identity: NodeIdentity,
    access_token: Option<String>,
    node_session_id: Option<String>,
}

impl NodeClient {
    pub fn new(gateway: impl Into<String>, identity: NodeIdentity) -> anyhow::Result<Self> {
        let gateway = gateway.into().trim_end_matches('/').to_string();
        anyhow::ensure!(
            gateway.starts_with("https://") || gateway.starts_with("http://127.0.0.1:"),
            "Gateway must use HTTPS or loopback HTTP"
        );
        Ok(Self {
            gateway,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(40))
                .build()?,
            identity,
            access_token: None,
            node_session_id: None,
        })
    }

    pub async fn enroll(
        &mut self,
        offer_id: &str,
        offer_secret: &str,
        node_kind: &str,
        metadata: DeviceMetadata,
        capabilities: Vec<CapabilityObservation>,
    ) -> anyhow::Result<RedeemEnrollmentResponse> {
        let response: RedeemEnrollmentResponse = self
            .http
            .post(format!("{}/node/v1/enrollments/redeem", self.gateway))
            .json(&RedeemEnrollmentRequest {
                protocol: ProtocolVersion::default(),
                offer_id: offer_id.into(),
                offer_secret: offer_secret.into(),
                node_kind: node_kind.into(),
                public_key_sec1: self.identity.public_key_base64url(),
                metadata,
                capabilities,
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        self.identity.node_id = Some(response.node_id.clone());
        self.identity.credential_id = Some(response.credential_id.clone());
        Ok(response)
    }

    pub async fn open_session(&mut self, boot_id: &str) -> anyhow::Result<OpenSessionResponse> {
        let credential_id = self
            .identity
            .credential_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Node is not enrolled"))?;
        let challenge: SessionChallengeResponse = self
            .http
            .post(format!("{}/node/v1/sessions/challenge", self.gateway))
            .json(&SessionChallengeRequest {
                credential_id: credential_id.clone(),
                protocol: ProtocolVersion::default(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let canonical = format!(
            "AIDAEMON-NODE-V1\n{}\n{}\n{}\n1\n{}\n{}",
            credential_id, challenge.challenge_id, challenge.nonce, challenge.instance_id, boot_id
        );
        let opened: OpenSessionResponse = self
            .http
            .post(format!("{}/node/v1/sessions", self.gateway))
            .json(&OpenSessionRequest {
                credential_id,
                challenge_id: challenge.challenge_id,
                protocol: ProtocolVersion::default(),
                boot_id: boot_id.into(),
                signature_der: self.identity.sign(canonical.as_bytes()),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        self.access_token = Some(opened.access_token.clone());
        self.node_session_id = Some(opened.node_session_id.clone());
        Ok(opened)
    }

    pub async fn heartbeat(
        &self,
        heartbeat: HeartbeatRequest,
    ) -> anyhow::Result<HeartbeatResponse> {
        Ok(self
            .http
            .post(format!("{}/node/v1/heartbeats", self.gateway))
            .bearer_auth(self.token()?)
            .json(&heartbeat)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn report_sensor_readings(
        &self,
        report: ReportSensorReadingsRequest,
    ) -> anyhow::Result<ReportSensorReadingsResponse> {
        Ok(self
            .http
            .post(format!("{}/node/v1/sensor-readings", self.gateway))
            .bearer_auth(self.token()?)
            .json(&report)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn outbox(
        &self,
        after: u64,
        wait_seconds: u64,
    ) -> anyhow::Result<NodeOutboxResponse> {
        Ok(self
            .http
            .get(format!("{}/node/v1/outbox", self.gateway))
            .bearer_auth(self.token()?)
            .query(&[("after", after), ("wait_seconds", wait_seconds.min(25))])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn acknowledge_outbox(
        &self,
        cursor: u64,
        request: &AckNodeOutboxRequest,
    ) -> anyhow::Result<AckNodeOutboxResponse> {
        Ok(self
            .http
            .post(format!("{}/node/v1/outbox/{cursor}/ack", self.gateway))
            .bearer_auth(self.token()?)
            .json(request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn create_turn(
        &self,
        request: &CreateTurnRequest,
        idempotency_key: &str,
    ) -> anyhow::Result<CreateTurnResponse> {
        Ok(self
            .http
            .post(format!("{}/node/v1/turns", self.gateway))
            .bearer_auth(self.token()?)
            .header("Idempotency-Key", idempotency_key)
            .json(request)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn events(
        &self,
        turn_id: &str,
        after: u64,
        wait_seconds: u64,
    ) -> anyhow::Result<TurnEventsResponse> {
        Ok(self
            .http
            .get(format!("{}/node/v1/turns/{turn_id}/events", self.gateway))
            .bearer_auth(self.token()?)
            .query(&[("after", after), ("wait_seconds", wait_seconds.min(25))])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn upload_media(
        &self,
        slot: &UploadSlot,
        media: Vec<u8>,
    ) -> anyhow::Result<UploadMediaResponse> {
        anyhow::ensure!(
            media.len() as u64 <= slot.max_bytes,
            "Media exceeds the server-issued upload limit"
        );
        Ok(self
            .http
            .put(self.gateway_url(&slot.upload_path)?)
            .bearer_auth(self.token()?)
            .header(reqwest::header::CONTENT_TYPE, &slot.content_type)
            .body(media)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn commit_turn(
        &self,
        turn_id: &str,
        slot_id: &str,
    ) -> anyhow::Result<CommitTurnResponse> {
        Ok(self
            .http
            .post(format!("{}/node/v1/turns/{turn_id}/commit", self.gateway))
            .bearer_auth(self.token()?)
            .json(&CommitTurnRequest {
                slot_id: slot_id.into(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn cancel_turn(&self, turn_id: &str) -> anyhow::Result<CancelTurnResponse> {
        Ok(self
            .http
            .post(format!("{}/node/v1/turns/{turn_id}/cancel", self.gateway))
            .bearer_auth(self.token()?)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }

    pub async fn download_media(&self, download_path: &str) -> anyhow::Result<Vec<u8>> {
        Ok(self
            .http
            .get(self.gateway_url(download_path)?)
            .bearer_auth(self.token()?)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec())
    }

    pub async fn download_firmware(&self, download_path: &str) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            download_path.starts_with("/node/v1/firmware/") && !download_path.contains(".."),
            "Firmware path must be a Gateway-relative firmware route"
        );
        Ok(self
            .http
            .get(self.gateway_url(download_path)?)
            .bearer_auth(self.token()?)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?
            .to_vec())
    }

    pub async fn rotate_credential(
        &mut self,
        mut replacement: NodeIdentity,
    ) -> anyhow::Result<RotateCredentialResponse> {
        let node_id = self
            .identity
            .node_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Node is not enrolled"))?;
        let node_session_id = self
            .node_session_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Node has no active session"))?;
        let challenge: CredentialRotationChallengeResponse = self
            .http
            .post(format!(
                "{}/node/v1/credentials/rotation-challenge",
                self.gateway
            ))
            .bearer_auth(self.token()?)
            .json(&CredentialRotationChallengeRequest {
                request_id: uuid::Uuid::new_v4().to_string(),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let public_key = replacement.public_key_base64url();
        let canonical = format!(
            "AIDAEMON-NODE-ROTATE-V1\n{}\n{}\n{}\n{}\n{}\n{}",
            node_id,
            node_session_id,
            challenge.rotation_id,
            challenge.nonce,
            challenge.instance_id,
            public_key
        );
        let response: RotateCredentialResponse = self
            .http
            .post(format!("{}/node/v1/credentials/rotate", self.gateway))
            .bearer_auth(self.token()?)
            .json(&RotateCredentialRequest {
                rotation_id: challenge.rotation_id,
                new_public_key_sec1: public_key,
                new_key_signature_der: replacement.sign(canonical.as_bytes()),
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        replacement.node_id = Some(node_id);
        replacement.credential_id = Some(response.credential_id.clone());
        self.identity = replacement;
        self.access_token = None;
        self.node_session_id = None;
        Ok(response)
    }

    fn gateway_url(&self, path: &str) -> anyhow::Result<String> {
        anyhow::ensure!(path.starts_with('/'), "Gateway path must be absolute");
        anyhow::ensure!(!path.starts_with("//"), "Gateway path is invalid");
        Ok(format!("{}{}", self.gateway, path))
    }

    fn token(&self) -> anyhow::Result<&str> {
        self.access_token
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Node has no active session"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_insecure_non_loopback_gateway() {
        assert!(NodeClient::new("http://example.test", NodeIdentity::generate()).is_err());
        assert!(NodeClient::new("http://127.0.0.1:8787", NodeIdentity::generate()).is_ok());
    }

    #[test]
    fn accepts_only_gateway_relative_media_paths() {
        let client = NodeClient::new("https://nodes.example.test", NodeIdentity::generate())
            .expect("secure gateway");
        assert_eq!(
            client.gateway_url("/node/v1/media/example").unwrap(),
            "https://nodes.example.test/node/v1/media/example"
        );
        assert!(client.gateway_url("https://attacker.test/media").is_err());
        assert!(client.gateway_url("//attacker.test/media").is_err());
    }
}
