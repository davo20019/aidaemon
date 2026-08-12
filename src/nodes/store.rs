use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde_json::json;
use sqlx::{Row, SqlitePool};

use super::auth;
use super::domain::{AuthenticatedNodeContext, NodeAction, NodeRecord, NodeTurnState};
use super::protocol::{RuntimeRecoveryReport, TurnEvent};

#[derive(Clone)]
pub struct NodeStore {
    pool: SqlitePool,
    instance_key: Arc<[u8; 32]>,
    instance_id: Arc<str>,
}

#[derive(Debug, Clone)]
pub struct PairingOffer {
    pub offer_id: String,
    pub offer_secret: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct EnrollmentResult {
    pub node: NodeRecord,
    pub credential_id: String,
    pub public_key_fingerprint: String,
}

#[derive(Debug, Clone)]
pub struct ChallengeRecord {
    pub challenge_id: String,
    pub credential_id: String,
    pub nonce: String,
    pub public_key_sec1: Vec<u8>,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CredentialRotationRecord {
    pub rotation_id: String,
    pub node_id: String,
    pub credential_id: String,
    pub node_session_id: String,
    pub nonce: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CreatedSession {
    pub context: AuthenticatedNodeContext,
    pub access_token: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct CreateTurnOutcome {
    pub turn_id: String,
    pub state: NodeTurnState,
    pub cursor: u64,
    pub duplicate: bool,
}

#[derive(Debug, Clone)]
pub struct MediaUploadRecord {
    pub slot_id: String,
    pub turn_id: String,
    pub node_id: String,
    pub media_kind: String,
    pub content_type: String,
    pub expected_bytes: u64,
    pub expected_sha256: String,
    pub duration_ms: Option<u64>,
    pub local_path: Option<String>,
    pub state: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LatestSensorReading {
    pub display_name: String,
    pub capability_id: String,
    pub capability_version: u16,
    pub value: f64,
    pub unit: String,
    pub sample_uptime_ms: u64,
    pub received_at: DateTime<Utc>,
    pub node_last_seen_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeHealthSnapshot {
    pub display_name: String,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub runtime_version: Option<String>,
    pub firmware_version: Option<String>,
    pub uptime_ms: Option<u64>,
    pub free_internal_heap: Option<u64>,
    pub largest_internal_allocation: Option<u64>,
    pub psram_free: Option<u64>,
    pub recovery: Option<RuntimeRecoveryReport>,
}

#[derive(Debug, Clone)]
pub struct AudioAnnouncementTarget {
    pub node_id: String,
    pub display_name: String,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub maximum_audio_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct QueuedAudioAnnouncement {
    pub cursor: u64,
    pub node_id: String,
    pub display_name: String,
    pub size_bytes: u64,
    pub expires_at: DateTime<Utc>,
    pub last_seen_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct NodeOutboxReceipt {
    pub status: String,
    pub detail_code: Option<String>,
    pub acknowledged_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct OutboundMediaCleanup {
    pub media_id: String,
    pub local_path: String,
}

fn parse_timestamp(value: &str) -> anyhow::Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc))
}

fn action_name(action: NodeAction) -> &'static str {
    match action {
        NodeAction::SubmitTextTurn => "submit_text_turn",
        NodeAction::SubmitAudioTurn => "submit_audio_turn",
        NodeAction::SubmitStillImage => "submit_still_image",
        NodeAction::ReportTelemetry => "report_telemetry",
        NodeAction::ReportSensor => "report_sensor",
        NodeAction::ReceiveAudio => "receive_audio",
        NodeAction::ReceiveDisplayCommand => "receive_display_command",
        NodeAction::ReceiveActuatorCommand => "receive_actuator_command",
        NodeAction::ReceiveOta => "receive_ota",
    }
}

pub async fn migrate(pool: &SqlitePool) -> anyhow::Result<()> {
    let statements = [
        r#"CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            display_name TEXT NOT NULL,
            policy_profile TEXT NOT NULL,
            policy_revision INTEGER NOT NULL DEFAULT 1,
            authorization_revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            last_seen_at TEXT,
            last_runtime_version TEXT,
            last_firmware_version TEXT,
            last_health_json TEXT
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_credentials (
            credential_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            public_key_sec1 BLOB NOT NULL,
            public_key_fingerprint TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('pending','active','retired','revoked')),
            revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            revoked_at TEXT
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_pairing_offers (
            offer_id TEXT PRIMARY KEY,
            secret_digest BLOB NOT NULL,
            owner_id TEXT NOT NULL,
            expected_kind TEXT NOT NULL,
            display_name TEXT NOT NULL,
            policy_profile TEXT NOT NULL,
            conversation_session_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            consumed_at TEXT,
            created_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_auth_challenges (
            challenge_id TEXT PRIMARY KEY,
            credential_id TEXT NOT NULL REFERENCES node_credentials(credential_id) ON DELETE CASCADE,
            nonce TEXT NOT NULL,
            protocol_major INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_credential_rotation_challenges (
            rotation_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            credential_id TEXT NOT NULL REFERENCES node_credentials(credential_id) ON DELETE CASCADE,
            node_session_id TEXT NOT NULL REFERENCES node_sessions(node_session_id) ON DELETE CASCADE,
            nonce TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_sessions (
            node_session_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            credential_id TEXT NOT NULL REFERENCES node_credentials(credential_id) ON DELETE CASCADE,
            token_digest BLOB NOT NULL UNIQUE,
            credential_revision INTEGER NOT NULL,
            policy_revision INTEGER NOT NULL,
            authorization_revision INTEGER NOT NULL,
            protocol_major INTEGER NOT NULL,
            boot_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            closed_at TEXT,
            close_reason TEXT
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_channels (
            node_channel_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            conversation_session_id TEXT NOT NULL UNIQUE,
            principal_id TEXT NOT NULL,
            memory_scope TEXT NOT NULL DEFAULT 'disabled',
            created_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_capabilities (
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            capability_id TEXT NOT NULL,
            capability_version INTEGER NOT NULL,
            limits_json TEXT NOT NULL,
            observed_session_id TEXT,
            observed_at TEXT NOT NULL,
            PRIMARY KEY(node_id, capability_id)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_authorizations (
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            action TEXT NOT NULL,
            constraints_json TEXT NOT NULL DEFAULT '{}',
            revision INTEGER NOT NULL,
            granted_at TEXT NOT NULL,
            revoked_at TEXT,
            PRIMARY KEY(node_id, action)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_sensor_readings_latest (
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            capability_id TEXT NOT NULL,
            capability_version INTEGER NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            sample_uptime_ms INTEGER NOT NULL,
            request_id TEXT NOT NULL,
            observed_session_id TEXT NOT NULL,
            received_at TEXT NOT NULL,
            PRIMARY KEY(node_id, capability_id)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_sensor_readings_history (
            reading_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            capability_id TEXT NOT NULL,
            capability_version INTEGER NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            sample_uptime_ms INTEGER NOT NULL,
            request_id TEXT NOT NULL,
            observed_session_id TEXT NOT NULL,
            received_at TEXT NOT NULL,
            UNIQUE(node_id, capability_id, request_id)
        )"#,
        r#"CREATE INDEX IF NOT EXISTS idx_node_sensor_history_capability_time
            ON node_sensor_readings_history(node_id, capability_id, received_at DESC)"#,
        r#"CREATE TABLE IF NOT EXISTS node_monitors (
            monitor_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            owner_session_id TEXT NOT NULL,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            capability_id TEXT NOT NULL,
            capability_version INTEGER NOT NULL,
            unit TEXT NOT NULL,
            comparison TEXT NOT NULL CHECK(comparison IN ('above','below')),
            threshold REAL NOT NULL,
            clear_threshold REAL NOT NULL,
            duration_seconds INTEGER NOT NULL CHECK(duration_seconds BETWEEN 0 AND 86400),
            stale_after_seconds INTEGER NOT NULL CHECK(stale_after_seconds BETWEEN 30 AND 86400),
            offline_after_seconds INTEGER NOT NULL CHECK(offline_after_seconds BETWEEN 30 AND 86400),
            repeat_seconds INTEGER NOT NULL CHECK(repeat_seconds BETWEEN 0 AND 604800),
            send_recovery INTEGER NOT NULL DEFAULT 1 CHECK(send_recovery IN (0,1)),
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active','paused','suspended','cancelled','expired')),
            condition_since TEXT,
            threshold_triggered_at TEXT,
            last_threshold_alert_at TEXT,
            availability_state TEXT NOT NULL DEFAULT 'normal'
                CHECK(availability_state IN ('normal','stale','offline')),
            last_availability_alert_at TEXT,
            last_value REAL,
            last_received_at TEXT,
            mandate_id TEXT REFERENCES mandates(id) ON DELETE SET NULL,
            mandate_goal_id TEXT,
            mandate_version INTEGER,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"#,
        r#"CREATE INDEX IF NOT EXISTS idx_node_monitors_evaluation
            ON node_monitors(status, node_id, capability_id, expires_at)"#,
        r#"CREATE INDEX IF NOT EXISTS idx_node_monitors_owner
            ON node_monitors(owner_session_id, created_at DESC)"#,
        r#"CREATE TABLE IF NOT EXISTS node_monitor_events (
            event_id TEXT PRIMARY KEY,
            monitor_id TEXT NOT NULL REFERENCES node_monitors(monitor_id) ON DELETE CASCADE,
            event_kind TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            notification_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )"#,
        r#"CREATE INDEX IF NOT EXISTS idx_node_monitor_events_monitor_time
            ON node_monitor_events(monitor_id, created_at DESC)"#,
        r#"CREATE TABLE IF NOT EXISTS node_turns (
            turn_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            node_channel_id TEXT NOT NULL REFERENCES node_channels(node_channel_id) ON DELETE CASCADE,
            request_id TEXT NOT NULL,
            input_kind TEXT NOT NULL,
            input_text TEXT,
            state TEXT NOT NULL,
            deadline_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error_code TEXT,
            result_text TEXT,
            UNIQUE(node_id, request_id)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_uploads (
            slot_id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL REFERENCES node_turns(turn_id) ON DELETE CASCADE,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            media_kind TEXT NOT NULL CHECK(media_kind IN ('audio','still_image')),
            content_type TEXT NOT NULL,
            expected_bytes INTEGER NOT NULL,
            expected_sha256 TEXT NOT NULL,
            duration_ms INTEGER,
            state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','uploaded','consumed','deleted')),
            local_path TEXT,
            received_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(turn_id)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_outbox (
            cursor INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            turn_id TEXT REFERENCES node_turns(turn_id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_outbox_receipts (
            cursor INTEGER NOT NULL REFERENCES node_outbox(cursor) ON DELETE CASCADE,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK(status IN ('played','failed','dismissed')),
            detail_code TEXT,
            acknowledged_at TEXT NOT NULL,
            PRIMARY KEY(cursor, node_id)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_outbound_media (
            media_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            outbox_cursor INTEGER NOT NULL UNIQUE REFERENCES node_outbox(cursor) ON DELETE CASCADE,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            local_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            deleted_at TEXT
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_response_media (
            media_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            turn_id TEXT NOT NULL REFERENCES node_turns(turn_id) ON DELETE CASCADE,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            local_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_request_dedup (
            node_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
            operation TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_digest BLOB NOT NULL,
            result_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            PRIMARY KEY(node_id, operation, idempotency_key)
        )"#,
        r#"CREATE TABLE IF NOT EXISTS node_audit_events (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            event_type TEXT NOT NULL,
            detail_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )"#,
        "CREATE INDEX IF NOT EXISTS idx_node_sessions_node_expiry ON node_sessions(node_id, expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_node_outbox_node_cursor ON node_outbox(node_id, cursor)",
        "CREATE INDEX IF NOT EXISTS idx_node_outbound_media_expiry ON node_outbound_media(expires_at, deleted_at)",
        "CREATE INDEX IF NOT EXISTS idx_node_turns_node_created ON node_turns(node_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_node_challenges_expiry ON node_auth_challenges(expires_at)",
    ];

    for statement in statements {
        sqlx::query(statement).execute(pool).await?;
    }
    Ok(())
}

impl NodeStore {
    pub fn new(pool: SqlitePool, instance_key: [u8; 32]) -> Self {
        let instance_digest = auth::keyed_digest(&instance_key, "instance-id", b"local-instance");
        let instance_id = format!(
            "aid_{}",
            instance_digest[..8]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        Self {
            pool,
            instance_key: Arc::new(instance_key),
            instance_id: Arc::from(instance_id),
        }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    pub async fn create_pairing_offer(
        &self,
        owner_id: &str,
        expected_kind: &str,
        display_name: &str,
        policy_profile: &str,
        ttl_seconds: u64,
    ) -> anyhow::Result<PairingOffer> {
        super::domain::validate_node_kind(expected_kind)?;
        super::domain::validate_policy_profile(policy_profile)?;
        anyhow::ensure!(
            !owner_id.trim().is_empty() && owner_id.len() <= 128,
            "owner id is invalid"
        );
        anyhow::ensure!(
            !display_name.trim().is_empty() && display_name.len() <= 80,
            "display name is invalid"
        );
        let now = Utc::now();
        let expires_at = now + Duration::seconds(ttl_seconds.clamp(60, 3600) as i64);
        let offer_id = format!("offer_{}", uuid::Uuid::new_v4().simple());
        let offer_secret = auth::random_secret(32);
        let digest =
            auth::keyed_digest(&self.instance_key, "pairing-offer", offer_secret.as_bytes());
        let conversation_session_id =
            format!("node:conversation:{}", uuid::Uuid::new_v4().simple());
        sqlx::query(
            "INSERT INTO node_pairing_offers
             (offer_id, secret_digest, owner_id, expected_kind, display_name, policy_profile,
              conversation_session_id, expires_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&offer_id)
        .bind(digest)
        .bind(owner_id.trim())
        .bind(expected_kind)
        .bind(display_name.trim())
        .bind(policy_profile)
        .bind(conversation_session_id)
        .bind(expires_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(PairingOffer {
            offer_id,
            offer_secret,
            expires_at,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn redeem_enrollment(
        &self,
        offer_id: &str,
        offer_secret: &str,
        node_kind: &str,
        public_key_sec1: &[u8],
        runtime_version: &str,
        firmware_version: &str,
        capabilities: &[super::domain::CapabilityObservation],
    ) -> anyhow::Result<EnrollmentResult> {
        let mut transaction = self.pool.begin().await?;
        let row = sqlx::query(
            "SELECT secret_digest, owner_id, expected_kind, display_name, policy_profile,
                    conversation_session_id, expires_at, consumed_at, attempts
             FROM node_pairing_offers WHERE offer_id = ?",
        )
        .bind(offer_id)
        .fetch_optional(&mut *transaction)
        .await?
        .ok_or_else(|| anyhow::anyhow!("pairing offer not found"))?;

        let attempts: i64 = row.get("attempts");
        anyhow::ensure!(attempts < 8, "pairing offer attempt limit exceeded");
        sqlx::query("UPDATE node_pairing_offers SET attempts = attempts + 1 WHERE offer_id = ?")
            .bind(offer_id)
            .execute(&mut *transaction)
            .await?;
        let consumed_at: Option<String> = row.get("consumed_at");
        anyhow::ensure!(consumed_at.is_none(), "pairing offer was already consumed");
        let expires_at: String = row.get("expires_at");
        anyhow::ensure!(
            parse_timestamp(&expires_at)? > Utc::now(),
            "pairing offer expired"
        );
        let expected_kind: String = row.get("expected_kind");
        anyhow::ensure!(
            expected_kind == node_kind,
            "Node kind does not match pairing offer"
        );
        let stored_digest: Vec<u8> = row.get("secret_digest");
        anyhow::ensure!(
            auth::constant_time_digest_matches(
                &self.instance_key,
                "pairing-offer",
                offer_secret.as_bytes(),
                &stored_digest,
            ),
            "pairing offer secret is invalid"
        );

        let node_id = format!("node_{}", uuid::Uuid::new_v4().simple());
        let credential_id = format!("cred_{}", uuid::Uuid::new_v4().simple());
        let node_channel_id = format!("nch_{}", uuid::Uuid::new_v4().simple());
        let now = Utc::now();
        let owner_id: String = row.get("owner_id");
        let display_name: String = row.get("display_name");
        let policy_profile: String = row.get("policy_profile");
        let conversation_session_id: String = row.get("conversation_session_id");
        let fingerprint = auth::public_key_fingerprint(public_key_sec1);

        sqlx::query(
            "INSERT INTO nodes
             (node_id, owner_id, kind, display_name, policy_profile, created_at,
              last_runtime_version, last_firmware_version)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&node_id)
        .bind(&owner_id)
        .bind(node_kind)
        .bind(&display_name)
        .bind(&policy_profile)
        .bind(now.to_rfc3339())
        .bind(runtime_version)
        .bind(firmware_version)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_credentials
             (credential_id, node_id, public_key_sec1, public_key_fingerprint, created_at)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&credential_id)
        .bind(&node_id)
        .bind(public_key_sec1)
        .bind(&fingerprint)
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_channels
             (node_channel_id, node_id, conversation_session_id, principal_id, memory_scope, created_at)
             VALUES (?, ?, ?, ?, 'disabled', ?)",
        )
        .bind(&node_channel_id)
        .bind(&node_id)
        .bind(&conversation_session_id)
        .bind(format!("node-principal:{node_id}"))
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        for action in [NodeAction::SubmitTextTurn, NodeAction::ReportTelemetry] {
            sqlx::query(
                "INSERT INTO node_authorizations
                 (node_id, action, constraints_json, revision, granted_at)
                 VALUES (?, ?, '{}', 1, ?)",
            )
            .bind(&node_id)
            .bind(action_name(action))
            .bind(now.to_rfc3339())
            .execute(&mut *transaction)
            .await?;
        }
        for capability in capabilities.iter().take(32) {
            sqlx::query(
                "INSERT INTO node_capabilities
                 (node_id, capability_id, capability_version, limits_json, observed_at)
                 VALUES (?, ?, ?, ?, ?)",
            )
            .bind(&node_id)
            .bind(&capability.capability_id)
            .bind(i64::from(capability.version))
            .bind(serde_json::to_string(&capability.limits)?)
            .bind(now.to_rfc3339())
            .execute(&mut *transaction)
            .await?;
        }
        sqlx::query(
            "UPDATE node_pairing_offers SET consumed_at = ? WHERE offer_id = ? AND consumed_at IS NULL",
        )
        .bind(now.to_rfc3339())
        .bind(offer_id)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_audit_events (node_id, event_type, detail_json, created_at)
             VALUES (?, 'node_enrolled', ?, ?)",
        )
        .bind(&node_id)
        .bind(json!({"kind": node_kind, "policy": policy_profile}).to_string())
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;

        Ok(EnrollmentResult {
            node: NodeRecord {
                node_id,
                owner_id,
                kind: node_kind.to_string(),
                display_name,
                policy_profile,
                policy_revision: 1,
                authorization_revision: 1,
                conversation_session_id,
                node_channel_id,
                created_at: now,
                revoked_at: None,
                last_seen_at: None,
            },
            credential_id,
            public_key_fingerprint: fingerprint,
        })
    }

    pub async fn create_challenge(
        &self,
        credential_id: &str,
        protocol_major: u16,
        ttl_seconds: u64,
    ) -> anyhow::Result<ChallengeRecord> {
        let row = sqlx::query(
            "SELECT c.public_key_sec1
             FROM node_credentials c JOIN nodes n ON n.node_id = c.node_id
             WHERE c.credential_id = ? AND c.state = 'active' AND c.revoked_at IS NULL
               AND n.revoked_at IS NULL",
        )
        .bind(credential_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("credential is unavailable"))?;
        let now = Utc::now();
        let expires_at = now + Duration::seconds(ttl_seconds.clamp(15, 300) as i64);
        let record = ChallengeRecord {
            challenge_id: format!("chal_{}", uuid::Uuid::new_v4().simple()),
            credential_id: credential_id.to_string(),
            nonce: auth::random_secret(24),
            public_key_sec1: row.get("public_key_sec1"),
            expires_at,
        };
        sqlx::query(
            "INSERT INTO node_auth_challenges
             (challenge_id, credential_id, nonce, protocol_major, expires_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.challenge_id)
        .bind(&record.credential_id)
        .bind(&record.nonce)
        .bind(i64::from(protocol_major))
        .bind(record.expires_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(record)
    }

    pub async fn get_challenge(
        &self,
        challenge_id: &str,
        credential_id: &str,
    ) -> anyhow::Result<ChallengeRecord> {
        let row = sqlx::query(
            "SELECT a.challenge_id, a.credential_id, a.nonce, a.expires_at, a.used_at,
                    c.public_key_sec1
             FROM node_auth_challenges a
             JOIN node_credentials c ON c.credential_id = a.credential_id
             JOIN nodes n ON n.node_id = c.node_id
             WHERE a.challenge_id = ? AND a.credential_id = ?
               AND c.state = 'active' AND c.revoked_at IS NULL AND n.revoked_at IS NULL",
        )
        .bind(challenge_id)
        .bind(credential_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("challenge is unavailable"))?;
        let used_at: Option<String> = row.get("used_at");
        anyhow::ensure!(used_at.is_none(), "challenge was already used");
        let expires_at = parse_timestamp(row.get::<String, _>("expires_at").as_str())?;
        anyhow::ensure!(expires_at > Utc::now(), "challenge expired");
        Ok(ChallengeRecord {
            challenge_id: row.get("challenge_id"),
            credential_id: row.get("credential_id"),
            nonce: row.get("nonce"),
            public_key_sec1: row.get("public_key_sec1"),
            expires_at,
        })
    }

    pub async fn consume_challenge_and_create_session(
        &self,
        challenge: &ChallengeRecord,
        protocol_major: u16,
        boot_id: &str,
        session_ttl_seconds: u64,
    ) -> anyhow::Result<CreatedSession> {
        let mut transaction = self.pool.begin().await?;
        let now = Utc::now();
        let updated = sqlx::query(
            "UPDATE node_auth_challenges SET used_at = ?
             WHERE challenge_id = ? AND credential_id = ? AND used_at IS NULL AND expires_at > ?",
        )
        .bind(now.to_rfc3339())
        .bind(&challenge.challenge_id)
        .bind(&challenge.credential_id)
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        anyhow::ensure!(
            updated.rows_affected() == 1,
            "challenge was already consumed or expired"
        );
        let row = sqlx::query(
            "SELECT n.node_id, n.kind, n.policy_profile, n.policy_revision,
                    n.authorization_revision, c.revision AS credential_revision,
                    ch.node_channel_id, ch.conversation_session_id
             FROM node_credentials c
             JOIN nodes n ON n.node_id = c.node_id
             JOIN node_channels ch ON ch.node_id = n.node_id
             WHERE c.credential_id = ? AND c.state = 'active' AND c.revoked_at IS NULL
               AND n.revoked_at IS NULL",
        )
        .bind(&challenge.credential_id)
        .fetch_optional(&mut *transaction)
        .await?
        .ok_or_else(|| anyhow::anyhow!("credential was revoked"))?;
        let access_token = auth::random_secret(32);
        let token_digest =
            auth::keyed_digest(&self.instance_key, "node-session", access_token.as_bytes());
        let node_session_id = format!("ns_{}", uuid::Uuid::new_v4().simple());
        let expires_at = now + Duration::seconds(session_ttl_seconds.clamp(60, 3600) as i64);
        let context = AuthenticatedNodeContext {
            node_id: row.get("node_id"),
            node_session_id: node_session_id.clone(),
            credential_id: challenge.credential_id.clone(),
            kind: row.get("kind"),
            policy_profile: row.get("policy_profile"),
            policy_revision: row.get::<i64, _>("policy_revision") as u64,
            authorization_revision: row.get::<i64, _>("authorization_revision") as u64,
            conversation_session_id: row.get("conversation_session_id"),
            node_channel_id: row.get("node_channel_id"),
        };
        sqlx::query(
            "INSERT INTO node_sessions
             (node_session_id, node_id, credential_id, token_digest, credential_revision,
              policy_revision, authorization_revision, protocol_major, boot_id, created_at,
              expires_at, last_seen_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&node_session_id)
        .bind(&context.node_id)
        .bind(&context.credential_id)
        .bind(token_digest)
        .bind(row.get::<i64, _>("credential_revision"))
        .bind(context.policy_revision as i64)
        .bind(context.authorization_revision as i64)
        .bind(i64::from(protocol_major))
        .bind(boot_id)
        .bind(now.to_rfc3339())
        .bind(expires_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(CreatedSession {
            context,
            access_token,
            expires_at,
        })
    }

    pub async fn authenticate(
        &self,
        access_token: &str,
    ) -> anyhow::Result<AuthenticatedNodeContext> {
        anyhow::ensure!(
            (32..=128).contains(&access_token.len()),
            "session token is invalid"
        );
        let token_digest =
            auth::keyed_digest(&self.instance_key, "node-session", access_token.as_bytes());
        let now = Utc::now().to_rfc3339();
        let row = sqlx::query(
            "SELECT s.node_session_id, s.credential_id, n.node_id, n.kind, n.policy_profile,
                    n.policy_revision, n.authorization_revision,
                    ch.node_channel_id, ch.conversation_session_id
             FROM node_sessions s
             JOIN nodes n ON n.node_id = s.node_id
             JOIN node_credentials c ON c.credential_id = s.credential_id
             JOIN node_channels ch ON ch.node_id = n.node_id
             WHERE s.token_digest = ? AND s.closed_at IS NULL AND s.expires_at > ?
               AND n.revoked_at IS NULL AND c.revoked_at IS NULL AND c.state = 'active'
               AND s.credential_revision = c.revision
               AND s.policy_revision = n.policy_revision
               AND s.authorization_revision = n.authorization_revision",
        )
        .bind(token_digest)
        .bind(&now)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Node Session is invalid, expired, stale, or revoked"))?;
        sqlx::query("UPDATE node_sessions SET last_seen_at = ? WHERE node_session_id = ?")
            .bind(now)
            .bind(row.get::<String, _>("node_session_id"))
            .execute(&self.pool)
            .await?;
        Ok(AuthenticatedNodeContext {
            node_id: row.get("node_id"),
            node_session_id: row.get("node_session_id"),
            credential_id: row.get("credential_id"),
            kind: row.get("kind"),
            policy_profile: row.get("policy_profile"),
            policy_revision: row.get::<i64, _>("policy_revision") as u64,
            authorization_revision: row.get::<i64, _>("authorization_revision") as u64,
            conversation_session_id: row.get("conversation_session_id"),
            node_channel_id: row.get("node_channel_id"),
        })
    }

    pub async fn create_credential_rotation_challenge(
        &self,
        context: &AuthenticatedNodeContext,
        ttl_seconds: u64,
    ) -> anyhow::Result<CredentialRotationRecord> {
        let now = Utc::now();
        let record = CredentialRotationRecord {
            rotation_id: format!("rot_{}", uuid::Uuid::new_v4().simple()),
            node_id: context.node_id.clone(),
            credential_id: context.credential_id.clone(),
            node_session_id: context.node_session_id.clone(),
            nonce: auth::random_secret(24),
            expires_at: now + Duration::seconds(ttl_seconds.clamp(15, 300) as i64),
        };
        sqlx::query(
            "INSERT INTO node_credential_rotation_challenges
             (rotation_id, node_id, credential_id, node_session_id, nonce, expires_at, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&record.rotation_id)
        .bind(&record.node_id)
        .bind(&record.credential_id)
        .bind(&record.node_session_id)
        .bind(&record.nonce)
        .bind(record.expires_at.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(record)
    }

    pub async fn credential_rotation_challenge(
        &self,
        context: &AuthenticatedNodeContext,
        rotation_id: &str,
    ) -> anyhow::Result<CredentialRotationRecord> {
        let row = sqlx::query(
            "SELECT rotation_id, node_id, credential_id, node_session_id, nonce, expires_at, used_at
             FROM node_credential_rotation_challenges
             WHERE rotation_id = ? AND node_id = ? AND credential_id = ? AND node_session_id = ?",
        )
        .bind(rotation_id)
        .bind(&context.node_id)
        .bind(&context.credential_id)
        .bind(&context.node_session_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("credential rotation challenge is unavailable"))?;
        anyhow::ensure!(
            row.get::<Option<String>, _>("used_at").is_none(),
            "credential rotation challenge was already used"
        );
        let expires_at = parse_timestamp(row.get::<String, _>("expires_at").as_str())?;
        anyhow::ensure!(
            expires_at > Utc::now(),
            "credential rotation challenge expired"
        );
        Ok(CredentialRotationRecord {
            rotation_id: row.get("rotation_id"),
            node_id: row.get("node_id"),
            credential_id: row.get("credential_id"),
            node_session_id: row.get("node_session_id"),
            nonce: row.get("nonce"),
            expires_at,
        })
    }

    pub async fn consume_credential_rotation(
        &self,
        record: &CredentialRotationRecord,
        new_public_key_sec1: &[u8],
    ) -> anyhow::Result<EnrollmentResult> {
        let now = Utc::now();
        let mut transaction = self.pool.begin().await?;
        let consumed = sqlx::query(
            "UPDATE node_credential_rotation_challenges SET used_at = ?
             WHERE rotation_id = ? AND used_at IS NULL AND expires_at > ?",
        )
        .bind(now.to_rfc3339())
        .bind(&record.rotation_id)
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        anyhow::ensure!(
            consumed.rows_affected() == 1,
            "credential rotation replayed or expired"
        );

        let node = sqlx::query(
            "SELECT n.node_id, n.owner_id, n.kind, n.display_name, n.policy_profile,
                    n.policy_revision, n.authorization_revision, n.created_at, n.revoked_at,
                    n.last_seen_at, ch.conversation_session_id, ch.node_channel_id
             FROM nodes n JOIN node_channels ch ON ch.node_id = n.node_id
             WHERE n.node_id = ? AND n.revoked_at IS NULL",
        )
        .bind(&record.node_id)
        .fetch_optional(&mut *transaction)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Node is unavailable"))?;
        let credential_id = format!("cred_{}", uuid::Uuid::new_v4().simple());
        let fingerprint = auth::public_key_fingerprint(new_public_key_sec1);
        sqlx::query(
            "INSERT INTO node_credentials
             (credential_id, node_id, public_key_sec1, public_key_fingerprint, created_at)
             VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&credential_id)
        .bind(&record.node_id)
        .bind(new_public_key_sec1)
        .bind(&fingerprint)
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "UPDATE node_credentials SET state = 'retired', revoked_at = ?
             WHERE credential_id = ? AND node_id = ? AND state = 'active'",
        )
        .bind(now.to_rfc3339())
        .bind(&record.credential_id)
        .bind(&record.node_id)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "UPDATE node_sessions SET closed_at = ?, close_reason = 'credential_rotated'
             WHERE node_id = ? AND closed_at IS NULL",
        )
        .bind(now.to_rfc3339())
        .bind(&record.node_id)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_audit_events (node_id, event_type, detail_json, created_at)
             VALUES (?, 'credential_rotated', ?, ?)",
        )
        .bind(&record.node_id)
        .bind(json!({"new_public_key_fingerprint": fingerprint}).to_string())
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;

        Ok(EnrollmentResult {
            node: node_from_row(node)?,
            credential_id,
            public_key_fingerprint: fingerprint,
        })
    }

    pub async fn is_authorized(&self, node_id: &str, action: NodeAction) -> anyhow::Result<bool> {
        let exists: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_authorizations
             WHERE node_id = ? AND action = ? AND revoked_at IS NULL",
        )
        .bind(node_id)
        .bind(action_name(action))
        .fetch_one(&self.pool)
        .await?;
        Ok(exists == 1)
    }

    pub async fn set_authorization(
        &self,
        node_id: &str,
        action: NodeAction,
        enabled: bool,
        constraints: serde_json::Value,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        let mut transaction = self.pool.begin().await?;
        let next_revision: i64 = sqlx::query_scalar(
            "UPDATE nodes SET authorization_revision = authorization_revision + 1
             WHERE node_id = ? AND revoked_at IS NULL RETURNING authorization_revision",
        )
        .bind(node_id)
        .fetch_one(&mut *transaction)
        .await?;
        if enabled {
            sqlx::query(
                "INSERT INTO node_authorizations
                 (node_id, action, constraints_json, revision, granted_at, revoked_at)
                 VALUES (?, ?, ?, ?, ?, NULL)
                 ON CONFLICT(node_id, action) DO UPDATE SET
                   constraints_json = excluded.constraints_json,
                   revision = excluded.revision,
                   granted_at = excluded.granted_at,
                   revoked_at = NULL",
            )
            .bind(node_id)
            .bind(action_name(action))
            .bind(serde_json::to_string(&constraints)?)
            .bind(next_revision)
            .bind(&now)
            .execute(&mut *transaction)
            .await?;
        } else {
            sqlx::query(
                "UPDATE node_authorizations SET revoked_at = ?, revision = ?
                 WHERE node_id = ? AND action = ?",
            )
            .bind(&now)
            .bind(next_revision)
            .bind(node_id)
            .bind(action_name(action))
            .execute(&mut *transaction)
            .await?;
        }
        sqlx::query(
            "UPDATE node_sessions SET closed_at = ?, close_reason = 'authorization_changed'
             WHERE node_id = ? AND closed_at IS NULL",
        )
        .bind(&now)
        .bind(node_id)
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(())
    }

    pub async fn record_heartbeat(
        &self,
        context: &AuthenticatedNodeContext,
        heartbeat: &super::protocol::HeartbeatRequest,
    ) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        let health = json!({
            "boot_id": heartbeat.boot_id,
            "uptime_ms": heartbeat.uptime_ms,
            "battery_percent": heartbeat.battery_percent,
            "free_internal_heap": heartbeat.free_internal_heap,
            "largest_internal_allocation": heartbeat.largest_internal_allocation,
            "psram_free": heartbeat.psram_free,
            "recovery": heartbeat.recovery.as_ref(),
        });
        let mut transaction = self.pool.begin().await?;
        sqlx::query(
            "UPDATE nodes SET last_seen_at = ?, last_runtime_version = ?,
                    last_firmware_version = ?, last_health_json = ?
             WHERE node_id = ? AND revoked_at IS NULL",
        )
        .bind(&now)
        .bind(&heartbeat.runtime_version)
        .bind(&heartbeat.firmware_version)
        .bind(health.to_string())
        .bind(&context.node_id)
        .execute(&mut *transaction)
        .await?;
        for capability in heartbeat.capabilities.iter().take(32) {
            sqlx::query(
                "INSERT INTO node_capabilities
                 (node_id, capability_id, capability_version, limits_json,
                  observed_session_id, observed_at)
                 VALUES (?, ?, ?, ?, ?, ?)
                 ON CONFLICT(node_id, capability_id) DO UPDATE SET
                   capability_version = excluded.capability_version,
                   limits_json = excluded.limits_json,
                   observed_session_id = excluded.observed_session_id,
                   observed_at = excluded.observed_at",
            )
            .bind(&context.node_id)
            .bind(&capability.capability_id)
            .bind(i64::from(capability.version))
            .bind(serde_json::to_string(&capability.limits)?)
            .bind(&context.node_session_id)
            .bind(&now)
            .execute(&mut *transaction)
            .await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    pub async fn record_sensor_readings(
        &self,
        context: &AuthenticatedNodeContext,
        request: &super::protocol::ReportSensorReadingsRequest,
        monitoring: Option<&crate::config::NodeMonitoringConfig>,
    ) -> anyhow::Result<DateTime<Utc>> {
        let received_at = Utc::now();
        let received_at_text = received_at.to_rfc3339();
        let mut transaction = self.pool.begin().await?;
        for reading in &request.readings {
            if monitoring.is_some_and(|config| config.enabled) {
                let duplicate: Option<i64> = sqlx::query_scalar(
                    "SELECT reading_id FROM node_sensor_readings_history
                     WHERE node_id = ? AND capability_id = ? AND request_id = ?",
                )
                .bind(&context.node_id)
                .bind(&reading.capability_id)
                .bind(&request.request_id)
                .fetch_optional(&mut *transaction)
                .await?;
                if duplicate.is_some() {
                    continue;
                }
            }
            let observed_version: Option<i64> = sqlx::query_scalar(
                "SELECT capability_version FROM node_capabilities
                 WHERE node_id = ? AND capability_id = ?",
            )
            .bind(&context.node_id)
            .bind(&reading.capability_id)
            .fetch_optional(&mut *transaction)
            .await?;
            anyhow::ensure!(
                observed_version == Some(i64::from(reading.capability_version)),
                "sensor capability is not currently observed at the reported version"
            );
            sqlx::query(
                "INSERT INTO node_sensor_readings_latest
                 (node_id, capability_id, capability_version, value, unit,
                  sample_uptime_ms, request_id, observed_session_id, received_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                 ON CONFLICT(node_id, capability_id) DO UPDATE SET
                   capability_version = excluded.capability_version,
                   value = excluded.value,
                   unit = excluded.unit,
                   sample_uptime_ms = excluded.sample_uptime_ms,
                   request_id = excluded.request_id,
                   observed_session_id = excluded.observed_session_id,
                   received_at = excluded.received_at",
            )
            .bind(&context.node_id)
            .bind(&reading.capability_id)
            .bind(i64::from(reading.capability_version))
            .bind(reading.value)
            .bind(&reading.unit)
            .bind(request.sample_uptime_ms as i64)
            .bind(&request.request_id)
            .bind(&context.node_session_id)
            .bind(&received_at_text)
            .execute(&mut *transaction)
            .await?;
            if let Some(config) = monitoring.filter(|config| config.enabled) {
                sqlx::query(
                    "INSERT OR IGNORE INTO node_sensor_readings_history
                     (node_id, capability_id, capability_version, value, unit,
                      sample_uptime_ms, request_id, observed_session_id, received_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(&context.node_id)
                .bind(&reading.capability_id)
                .bind(i64::from(reading.capability_version))
                .bind(reading.value)
                .bind(&reading.unit)
                .bind(request.sample_uptime_ms as i64)
                .bind(&request.request_id)
                .bind(&context.node_session_id)
                .bind(&received_at_text)
                .execute(&mut *transaction)
                .await?;
                sqlx::query(
                    "DELETE FROM node_sensor_readings_history
                     WHERE node_id = ? AND capability_id = ? AND reading_id NOT IN (
                       SELECT reading_id FROM node_sensor_readings_history
                       WHERE node_id = ? AND capability_id = ?
                       ORDER BY received_at DESC, reading_id DESC LIMIT ?
                     )",
                )
                .bind(&context.node_id)
                .bind(&reading.capability_id)
                .bind(&context.node_id)
                .bind(&reading.capability_id)
                .bind(config.max_history_rows_per_capability as i64)
                .execute(&mut *transaction)
                .await?;
            }
        }
        transaction.commit().await?;
        Ok(received_at)
    }

    pub async fn create_text_turn(
        &self,
        context: &AuthenticatedNodeContext,
        idempotency_key: &str,
        request_id: &str,
        text: &str,
    ) -> anyhow::Result<CreateTurnOutcome> {
        let request_digest = auth::keyed_digest(
            &self.instance_key,
            "turn-request",
            format!("text\0{request_id}\0{text}").as_bytes(),
        );
        let mut transaction = self.pool.begin().await?;
        if let Some(row) = sqlx::query(
            "SELECT request_digest, result_id FROM node_request_dedup
             WHERE node_id = ? AND operation = 'create_turn' AND idempotency_key = ?",
        )
        .bind(&context.node_id)
        .bind(idempotency_key)
        .fetch_optional(&mut *transaction)
        .await?
        {
            let stored: Vec<u8> = row.get("request_digest");
            anyhow::ensure!(
                stored == request_digest,
                "idempotency key conflicts with another request"
            );
            let turn_id: String = row.get("result_id");
            let state: String =
                sqlx::query_scalar("SELECT state FROM node_turns WHERE turn_id = ?")
                    .bind(&turn_id)
                    .fetch_one(&mut *transaction)
                    .await?;
            let cursor: i64 = sqlx::query_scalar(
                "SELECT COALESCE(MAX(cursor), 0) FROM node_outbox WHERE turn_id = ?",
            )
            .bind(&turn_id)
            .fetch_one(&mut *transaction)
            .await?;
            transaction.commit().await?;
            return Ok(CreateTurnOutcome {
                turn_id,
                state: parse_turn_state(&state),
                cursor: cursor as u64,
                duplicate: true,
            });
        }

        let now = Utc::now();
        let turn_id = format!("turn_{}", uuid::Uuid::new_v4().simple());
        let deadline = now + Duration::seconds(120);
        sqlx::query(
            "INSERT INTO node_turns
             (turn_id, node_id, node_channel_id, request_id, input_kind, input_text,
              state, deadline_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, 'text', ?, 'accepted', ?, ?, ?)",
        )
        .bind(&turn_id)
        .bind(&context.node_id)
        .bind(&context.node_channel_id)
        .bind(request_id)
        .bind(text)
        .bind(deadline.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        let cursor = append_event_tx(
            &mut transaction,
            &context.node_id,
            Some(&turn_id),
            "turn.accepted",
            json!({"state": "accepted"}),
        )
        .await?;
        sqlx::query(
            "INSERT INTO node_request_dedup
             (node_id, operation, idempotency_key, request_digest, result_id, created_at, expires_at)
             VALUES (?, 'create_turn', ?, ?, ?, ?, ?)",
        )
        .bind(&context.node_id)
        .bind(idempotency_key)
        .bind(request_digest)
        .bind(&turn_id)
        .bind(now.to_rfc3339())
        .bind((now + Duration::hours(24)).to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(CreateTurnOutcome {
            turn_id,
            state: NodeTurnState::Accepted,
            cursor,
            duplicate: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_media_turn(
        &self,
        context: &AuthenticatedNodeContext,
        idempotency_key: &str,
        request_id: &str,
        media_kind: &str,
        content_type: &str,
        size_bytes: u64,
        sha256: &str,
        duration_ms: Option<u64>,
    ) -> anyhow::Result<(CreateTurnOutcome, String)> {
        anyhow::ensure!(
            matches!(media_kind, "audio" | "still_image"),
            "unsupported media kind"
        );
        let request_digest = auth::keyed_digest(
            &self.instance_key,
            "turn-request",
            format!(
                "{media_kind}\0{request_id}\0{content_type}\0{size_bytes}\0{sha256}\0{:?}",
                duration_ms
            )
            .as_bytes(),
        );
        let mut transaction = self.pool.begin().await?;
        if let Some(row) = sqlx::query(
            "SELECT request_digest, result_id FROM node_request_dedup
             WHERE node_id = ? AND operation = 'create_turn' AND idempotency_key = ?",
        )
        .bind(&context.node_id)
        .bind(idempotency_key)
        .fetch_optional(&mut *transaction)
        .await?
        {
            anyhow::ensure!(
                row.get::<Vec<u8>, _>("request_digest") == request_digest,
                "idempotency key conflicts with another request"
            );
            let turn_id: String = row.get("result_id");
            let upload = sqlx::query("SELECT slot_id FROM node_uploads WHERE turn_id = ?")
                .bind(&turn_id)
                .fetch_one(&mut *transaction)
                .await?;
            let state: String =
                sqlx::query_scalar("SELECT state FROM node_turns WHERE turn_id = ?")
                    .bind(&turn_id)
                    .fetch_one(&mut *transaction)
                    .await?;
            let cursor: i64 = sqlx::query_scalar(
                "SELECT COALESCE(MAX(cursor), 0) FROM node_outbox WHERE turn_id = ?",
            )
            .bind(&turn_id)
            .fetch_one(&mut *transaction)
            .await?;
            transaction.commit().await?;
            return Ok((
                CreateTurnOutcome {
                    turn_id,
                    state: parse_turn_state(&state),
                    cursor: cursor as u64,
                    duplicate: true,
                },
                upload.get("slot_id"),
            ));
        }
        let now = Utc::now();
        let turn_id = format!("turn_{}", uuid::Uuid::new_v4().simple());
        let slot_id = format!("slot_{}", uuid::Uuid::new_v4().simple());
        sqlx::query(
            "INSERT INTO node_turns
             (turn_id, node_id, node_channel_id, request_id, input_kind, state, deadline_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, 'accepted', ?, ?, ?)",
        ).bind(&turn_id).bind(&context.node_id).bind(&context.node_channel_id).bind(request_id)
            .bind(media_kind).bind((now + Duration::seconds(120)).to_rfc3339())
            .bind(now.to_rfc3339()).bind(now.to_rfc3339()).execute(&mut *transaction).await?;
        sqlx::query(
            "INSERT INTO node_uploads
             (slot_id, turn_id, node_id, media_kind, content_type, expected_bytes, expected_sha256, duration_ms, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ).bind(&slot_id).bind(&turn_id).bind(&context.node_id).bind(media_kind).bind(content_type)
            .bind(size_bytes as i64).bind(sha256).bind(duration_ms.map(|value| value as i64))
            .bind(now.to_rfc3339()).execute(&mut *transaction).await?;
        let cursor = append_event_tx(
            &mut transaction,
            &context.node_id,
            Some(&turn_id),
            "turn.accepted",
            json!({"state":"accepted", "upload_required":true}),
        )
        .await?;
        sqlx::query(
            "INSERT INTO node_request_dedup
             (node_id, operation, idempotency_key, request_digest, result_id, created_at, expires_at)
             VALUES (?, 'create_turn', ?, ?, ?, ?, ?)",
        ).bind(&context.node_id).bind(idempotency_key).bind(request_digest).bind(&turn_id)
            .bind(now.to_rfc3339()).bind((now + Duration::hours(24)).to_rfc3339()).execute(&mut *transaction).await?;
        transaction.commit().await?;
        Ok((
            CreateTurnOutcome {
                turn_id,
                state: NodeTurnState::Accepted,
                cursor,
                duplicate: false,
            },
            slot_id,
        ))
    }

    pub async fn upload_slot(
        &self,
        node_id: &str,
        slot_id: &str,
    ) -> anyhow::Result<MediaUploadRecord> {
        let row = sqlx::query(
            "SELECT slot_id, turn_id, node_id, media_kind, content_type, expected_bytes,
                    expected_sha256, duration_ms, local_path, state
             FROM node_uploads WHERE slot_id = ? AND node_id = ?",
        )
        .bind(slot_id)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("upload slot not found"))?;
        Ok(MediaUploadRecord {
            slot_id: row.get("slot_id"),
            turn_id: row.get("turn_id"),
            node_id: row.get("node_id"),
            media_kind: row.get("media_kind"),
            content_type: row.get("content_type"),
            expected_bytes: row.get::<i64, _>("expected_bytes") as u64,
            expected_sha256: row.get("expected_sha256"),
            duration_ms: row
                .get::<Option<i64>, _>("duration_ms")
                .map(|value| value as u64),
            local_path: row.get("local_path"),
            state: row.get("state"),
        })
    }

    pub async fn mark_slot_uploaded(
        &self,
        node_id: &str,
        slot_id: &str,
        local_path: &str,
    ) -> anyhow::Result<()> {
        let changed = sqlx::query(
            "UPDATE node_uploads SET state = 'uploaded', local_path = ?, received_at = ?
             WHERE slot_id = ? AND node_id = ? AND state = 'pending'",
        )
        .bind(local_path)
        .bind(Utc::now().to_rfc3339())
        .bind(slot_id)
        .bind(node_id)
        .execute(&self.pool)
        .await?;
        anyhow::ensure!(
            changed.rows_affected() == 1,
            "upload slot is unavailable or already used"
        );
        Ok(())
    }

    pub async fn mark_slot_consumed(
        &self,
        node_id: &str,
        slot_id: &str,
        deleted: bool,
    ) -> anyhow::Result<()> {
        sqlx::query("UPDATE node_uploads SET state = ? WHERE slot_id = ? AND node_id = ?")
            .bind(if deleted { "deleted" } else { "consumed" })
            .bind(slot_id)
            .bind(node_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn turn_input_text(&self, node_id: &str, turn_id: &str) -> anyhow::Result<String> {
        sqlx::query_scalar(
            "SELECT input_text FROM node_turns WHERE turn_id = ? AND node_id = ? AND input_kind = 'text'",
        )
        .bind(turn_id)
        .bind(node_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten()
        .ok_or_else(|| anyhow::anyhow!("text turn not found"))
    }

    pub async fn update_turn(
        &self,
        node_id: &str,
        turn_id: &str,
        state: NodeTurnState,
        payload: serde_json::Value,
    ) -> anyhow::Result<u64> {
        let mut transaction = self.pool.begin().await?;
        let now = Utc::now().to_rfc3339();
        let result_text = payload.get("text").and_then(|value| value.as_str());
        let error_code = payload.get("code").and_then(|value| value.as_str());
        let updated = sqlx::query(
            "UPDATE node_turns SET state = ?, updated_at = ?, result_text = COALESCE(?, result_text),
                    error_code = COALESCE(?, error_code)
             WHERE turn_id = ? AND node_id = ? AND state NOT IN ('cancelled')",
        )
        .bind(state.as_str())
        .bind(&now)
        .bind(result_text)
        .bind(error_code)
        .bind(turn_id)
        .bind(node_id)
        .execute(&mut *transaction)
        .await?;
        anyhow::ensure!(
            updated.rows_affected() == 1,
            "turn is unavailable or cancelled"
        );
        let cursor = append_event_tx(
            &mut transaction,
            node_id,
            Some(turn_id),
            &format!("turn.{}", state.as_str()),
            payload,
        )
        .await?;
        transaction.commit().await?;
        Ok(cursor)
    }

    pub async fn begin_turn_processing(&self, node_id: &str, turn_id: &str) -> anyhow::Result<u64> {
        let mut transaction = self.pool.begin().await?;
        let changed = sqlx::query(
            "UPDATE node_turns SET state = 'thinking', updated_at = ?
             WHERE turn_id = ? AND node_id = ? AND state = 'accepted'",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(turn_id)
        .bind(node_id)
        .execute(&mut *transaction)
        .await?;
        anyhow::ensure!(
            changed.rows_affected() == 1,
            "turn is already processing or unavailable"
        );
        let cursor = append_event_tx(
            &mut transaction,
            node_id,
            Some(turn_id),
            "turn.thinking",
            json!({"state":"thinking"}),
        )
        .await?;
        transaction.commit().await?;
        Ok(cursor)
    }

    pub async fn cancel_turn(&self, node_id: &str, turn_id: &str) -> anyhow::Result<NodeTurnState> {
        let mut transaction = self.pool.begin().await?;
        let current: String =
            sqlx::query_scalar("SELECT state FROM node_turns WHERE turn_id = ? AND node_id = ?")
                .bind(turn_id)
                .bind(node_id)
                .fetch_optional(&mut *transaction)
                .await?
                .ok_or_else(|| anyhow::anyhow!("turn not found"))?;
        if matches!(current.as_str(), "complete" | "error" | "cancelled") {
            transaction.commit().await?;
            return Ok(parse_turn_state(&current));
        }
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE node_turns SET state = 'cancelled', updated_at = ? WHERE turn_id = ?")
            .bind(&now)
            .bind(turn_id)
            .execute(&mut *transaction)
            .await?;
        append_event_tx(
            &mut transaction,
            node_id,
            Some(turn_id),
            "turn.cancelled",
            json!({"state": "cancelled"}),
        )
        .await?;
        transaction.commit().await?;
        Ok(NodeTurnState::Cancelled)
    }

    pub async fn turn_is_cancelled(&self, node_id: &str, turn_id: &str) -> anyhow::Result<bool> {
        let state: Option<String> =
            sqlx::query_scalar("SELECT state FROM node_turns WHERE turn_id = ? AND node_id = ?")
                .bind(turn_id)
                .bind(node_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(state.as_deref() == Some("cancelled"))
    }

    pub async fn events_after(
        &self,
        node_id: &str,
        turn_id: &str,
        after: u64,
        limit: usize,
    ) -> anyhow::Result<Vec<TurnEvent>> {
        let owns: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM node_turns WHERE turn_id = ? AND node_id = ?")
                .bind(turn_id)
                .bind(node_id)
                .fetch_one(&self.pool)
                .await?;
        anyhow::ensure!(owns == 1, "turn not found");
        let rows = sqlx::query(
            "SELECT cursor, turn_id, event_type, payload_json, created_at
             FROM node_outbox WHERE node_id = ? AND turn_id = ? AND cursor > ?
             ORDER BY cursor ASC LIMIT ?",
        )
        .bind(node_id)
        .bind(turn_id)
        .bind(after as i64)
        .bind(limit.clamp(1, 100) as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(TurnEvent {
                    cursor: row.get::<i64, _>("cursor") as u64,
                    turn_id: row.get("turn_id"),
                    event_type: row.get("event_type"),
                    created_at: row.get("created_at"),
                    payload: serde_json::from_str(&row.get::<String, _>("payload_json"))?,
                })
            })
            .collect()
    }

    pub async fn append_outbound_for_conversation(
        &self,
        conversation_session_id: &str,
        text: &str,
    ) -> anyhow::Result<u64> {
        let row = sqlx::query(
            "SELECT ch.node_id FROM node_channels ch JOIN nodes n ON n.node_id = ch.node_id
             WHERE ch.conversation_session_id = ? AND n.revoked_at IS NULL",
        )
        .bind(conversation_session_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Node Channel is unavailable"))?;
        let mut transaction = self.pool.begin().await?;
        let cursor = append_event_tx(
            &mut transaction,
            row.get::<String, _>("node_id").as_str(),
            None,
            "channel.text",
            json!({"text": text}),
        )
        .await?;
        transaction.commit().await?;
        Ok(cursor)
    }

    pub async fn resolve_audio_announcement_target(
        &self,
        selector: Option<&str>,
    ) -> anyhow::Result<AudioAnnouncementTarget> {
        let rows = if let Some(selector) = selector.map(str::trim).filter(|value| !value.is_empty())
        {
            sqlx::query(
                "SELECT n.node_id, n.display_name, n.last_seen_at, c.limits_json
                 FROM nodes n
                 JOIN node_capabilities c ON c.node_id = n.node_id
                 JOIN node_authorizations a ON a.node_id = n.node_id
                 WHERE n.revoked_at IS NULL
                   AND (n.node_id = ? OR n.display_name = ? COLLATE NOCASE)
                   AND c.capability_id = 'output.audio'
                   AND c.capability_version >= 1
                   AND c.observed_at = n.last_seen_at
                   AND a.action = 'receive_audio'
                   AND a.revoked_at IS NULL
                 ORDER BY n.created_at ASC",
            )
            .bind(selector)
            .bind(selector)
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "SELECT n.node_id, n.display_name, n.last_seen_at, c.limits_json
                 FROM nodes n
                 JOIN node_capabilities c ON c.node_id = n.node_id
                 JOIN node_authorizations a ON a.node_id = n.node_id
                 WHERE n.revoked_at IS NULL
                   AND c.capability_id = 'output.audio'
                   AND c.capability_version >= 1
                   AND c.observed_at = n.last_seen_at
                   AND a.action = 'receive_audio'
                   AND a.revoked_at IS NULL
                 ORDER BY n.created_at ASC",
            )
            .fetch_all(&self.pool)
            .await?
        };
        anyhow::ensure!(
            !rows.is_empty(),
            "No active Node matches the target with current output.audio capability and receive_audio authorization"
        );
        anyhow::ensure!(
            rows.len() == 1,
            "More than one eligible Node matches; specify the exact Node display name"
        );
        let row = &rows[0];
        let last_seen_at: Option<String> = row.get("last_seen_at");
        let limits: String = row.get("limits_json");
        let maximum_audio_bytes = serde_json::from_str::<serde_json::Value>(&limits)
            .ok()
            .and_then(|value| value.get("max_bytes").and_then(serde_json::Value::as_u64));
        Ok(AudioAnnouncementTarget {
            node_id: row.get("node_id"),
            display_name: row.get("display_name"),
            last_seen_at: last_seen_at.as_deref().map(parse_timestamp).transpose()?,
            maximum_audio_bytes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn queue_audio_announcement(
        &self,
        target: &AudioAnnouncementTarget,
        content_type: &str,
        size_bytes: u64,
        sha256: &str,
        local_path: &str,
        ttl_seconds: u64,
        max_pending: usize,
    ) -> anyhow::Result<QueuedAudioAnnouncement> {
        let now = Utc::now();
        let expires_at = now + Duration::seconds(ttl_seconds as i64);
        let mut transaction = self.pool.begin().await?;
        let eligible: i64 = sqlx::query_scalar(
            "SELECT COUNT(*)
             FROM nodes n
             JOIN node_capabilities c ON c.node_id = n.node_id
             JOIN node_authorizations a ON a.node_id = n.node_id
             WHERE n.node_id = ? AND n.revoked_at IS NULL
               AND c.capability_id = 'output.audio'
               AND c.capability_version >= 1
               AND c.observed_at = n.last_seen_at
               AND a.action = 'receive_audio' AND a.revoked_at IS NULL",
        )
        .bind(&target.node_id)
        .fetch_one(&mut *transaction)
        .await?;
        anyhow::ensure!(
            eligible == 1,
            "Node announcement eligibility changed before queueing"
        );
        let pending: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_outbox o
             WHERE o.node_id = ? AND o.turn_id IS NULL AND o.event_type = 'channel.audio'
               AND o.expires_at > ?
               AND NOT EXISTS (
                   SELECT 1 FROM node_outbox_receipts r
                   WHERE r.cursor = o.cursor AND r.node_id = o.node_id
               )",
        )
        .bind(&target.node_id)
        .bind(now.to_rfc3339())
        .fetch_one(&mut *transaction)
        .await?;
        anyhow::ensure!(
            pending < max_pending as i64,
            "Node announcement queue is full; wait for delivery or expiry"
        );
        let media_id = format!("media_{}", uuid::Uuid::new_v4().simple());
        let payload = json!({
            "audio": {
                "media_id": media_id,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "download_path": format!("/node/v1/media/{media_id}")
            }
        });
        let inserted = sqlx::query(
            "INSERT INTO node_outbox
             (node_id, turn_id, event_type, payload_json, priority, created_at, expires_at)
             VALUES (?, NULL, 'channel.audio', ?, 50, ?, ?)",
        )
        .bind(&target.node_id)
        .bind(payload.to_string())
        .bind(now.to_rfc3339())
        .bind(expires_at.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        let cursor = inserted.last_insert_rowid() as u64;
        sqlx::query(
            "INSERT INTO node_outbound_media
             (media_id, node_id, outbox_cursor, content_type, size_bytes, sha256,
              local_path, created_at, expires_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&media_id)
        .bind(&target.node_id)
        .bind(cursor as i64)
        .bind(content_type)
        .bind(size_bytes as i64)
        .bind(sha256)
        .bind(local_path)
        .bind(now.to_rfc3339())
        .bind(expires_at.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_audit_events (node_id, event_type, detail_json, created_at)
             VALUES (?, 'audio_announcement_queued', ?, ?)",
        )
        .bind(&target.node_id)
        .bind(json!({"cursor": cursor, "size_bytes": size_bytes}).to_string())
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(QueuedAudioAnnouncement {
            cursor,
            node_id: target.node_id.clone(),
            display_name: target.display_name.clone(),
            size_bytes,
            expires_at,
            last_seen_at: target.last_seen_at,
        })
    }

    pub async fn pending_node_outbox(
        &self,
        node_id: &str,
        after: u64,
        limit: usize,
    ) -> anyhow::Result<Vec<super::protocol::NodeOutboxEvent>> {
        let rows = sqlx::query(
            "SELECT o.cursor, o.event_type, o.payload_json, o.created_at, o.expires_at
             FROM node_outbox o
             WHERE o.node_id = ? AND o.turn_id IS NULL AND o.cursor > ?
               AND o.expires_at > ?
               AND NOT EXISTS (
                   SELECT 1 FROM node_outbox_receipts r
                   WHERE r.cursor = o.cursor AND r.node_id = o.node_id
               )
             ORDER BY o.priority DESC, o.cursor ASC LIMIT ?",
        )
        .bind(node_id)
        .bind(after as i64)
        .bind(Utc::now().to_rfc3339())
        .bind(limit.clamp(1, 8) as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(super::protocol::NodeOutboxEvent {
                    cursor: row.get::<i64, _>("cursor") as u64,
                    event_type: row.get("event_type"),
                    created_at: row.get("created_at"),
                    expires_at: row.get("expires_at"),
                    payload: serde_json::from_str(&row.get::<String, _>("payload_json"))?,
                })
            })
            .collect()
    }

    pub async fn acknowledge_node_outbox(
        &self,
        node_id: &str,
        cursor: u64,
        status: super::protocol::NodeOutboxAckStatus,
        detail_code: Option<&str>,
    ) -> anyhow::Result<Option<OutboundMediaCleanup>> {
        let now = Utc::now();
        let mut transaction = self.pool.begin().await?;
        let event_exists: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_outbox
             WHERE cursor = ? AND node_id = ? AND turn_id IS NULL AND event_type = 'channel.audio'",
        )
        .bind(cursor as i64)
        .bind(node_id)
        .fetch_one(&mut *transaction)
        .await?;
        anyhow::ensure!(event_exists == 1, "Node outbox event was not found");
        sqlx::query(
            "INSERT INTO node_outbox_receipts
             (cursor, node_id, status, detail_code, acknowledged_at)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(cursor, node_id) DO NOTHING",
        )
        .bind(cursor as i64)
        .bind(node_id)
        .bind(status.as_str())
        .bind(detail_code)
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_audit_events (node_id, event_type, detail_json, created_at)
             VALUES (?, 'audio_announcement_acknowledged', ?, ?)",
        )
        .bind(node_id)
        .bind(
            json!({"cursor": cursor, "status": status.as_str(), "detail_code": detail_code})
                .to_string(),
        )
        .bind(now.to_rfc3339())
        .execute(&mut *transaction)
        .await?;
        let media = sqlx::query(
            "SELECT media_id, local_path FROM node_outbound_media
             WHERE node_id = ? AND outbox_cursor = ? AND deleted_at IS NULL",
        )
        .bind(node_id)
        .bind(cursor as i64)
        .fetch_optional(&mut *transaction)
        .await?
        .map(|row| OutboundMediaCleanup {
            media_id: row.get("media_id"),
            local_path: row.get("local_path"),
        });
        transaction.commit().await?;
        Ok(media)
    }

    pub async fn node_outbox_receipt(
        &self,
        node_id: &str,
        cursor: u64,
    ) -> anyhow::Result<Option<NodeOutboxReceipt>> {
        let row = sqlx::query(
            "SELECT status, detail_code, acknowledged_at FROM node_outbox_receipts
             WHERE node_id = ? AND cursor = ?",
        )
        .bind(node_id)
        .bind(cursor as i64)
        .fetch_optional(&self.pool)
        .await?;
        row.map(|row| {
            let acknowledged_at: String = row.get("acknowledged_at");
            Ok(NodeOutboxReceipt {
                status: row.get("status"),
                detail_code: row.get("detail_code"),
                acknowledged_at: parse_timestamp(&acknowledged_at)?,
            })
        })
        .transpose()
    }

    pub async fn outbound_media_cleanup_candidates(
        &self,
        node_id: Option<&str>,
    ) -> anyhow::Result<Vec<OutboundMediaCleanup>> {
        let rows = sqlx::query(
            "SELECT m.media_id, m.local_path
             FROM node_outbound_media m
             WHERE m.deleted_at IS NULL AND (? IS NULL OR m.node_id = ?)
               AND (m.expires_at <= ? OR EXISTS (
                   SELECT 1 FROM node_outbox_receipts r
                   WHERE r.cursor = m.outbox_cursor AND r.node_id = m.node_id
               ))
             ORDER BY m.created_at ASC LIMIT 32",
        )
        .bind(node_id)
        .bind(node_id)
        .bind(Utc::now().to_rfc3339())
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|row| OutboundMediaCleanup {
                media_id: row.get("media_id"),
                local_path: row.get("local_path"),
            })
            .collect())
    }

    pub async fn mark_outbound_media_deleted(&self, media_id: &str) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE node_outbound_media SET deleted_at = COALESCE(deleted_at, ?)
             WHERE media_id = ?",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(media_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn register_response_media(
        &self,
        node_id: &str,
        turn_id: &str,
        content_type: &str,
        size_bytes: u64,
        local_path: &str,
    ) -> anyhow::Result<String> {
        let media_id = format!("media_{}", uuid::Uuid::new_v4().simple());
        let now = Utc::now();
        sqlx::query(
            "INSERT INTO node_response_media
             (media_id, node_id, turn_id, content_type, size_bytes, local_path, created_at, expires_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ).bind(&media_id).bind(node_id).bind(turn_id).bind(content_type).bind(size_bytes as i64).bind(local_path)
            .bind(now.to_rfc3339()).bind((now + Duration::minutes(30)).to_rfc3339()).execute(&self.pool).await?;
        Ok(media_id)
    }

    pub async fn response_media(
        &self,
        node_id: &str,
        media_id: &str,
    ) -> anyhow::Result<(String, String, u64)> {
        let row = sqlx::query(
            "SELECT local_path, content_type, size_bytes FROM node_response_media
             WHERE media_id = ? AND node_id = ? AND expires_at > ?
             UNION ALL
             SELECT local_path, content_type, size_bytes FROM node_outbound_media
             WHERE media_id = ? AND node_id = ? AND expires_at > ? AND deleted_at IS NULL
             LIMIT 1",
        )
        .bind(media_id)
        .bind(node_id)
        .bind(Utc::now().to_rfc3339())
        .bind(media_id)
        .bind(node_id)
        .bind(Utc::now().to_rfc3339())
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| anyhow::anyhow!("response media not found or expired"))?;
        Ok((
            row.get("local_path"),
            row.get("content_type"),
            row.get::<i64, _>("size_bytes") as u64,
        ))
    }

    pub async fn list_nodes(&self) -> anyhow::Result<Vec<NodeRecord>> {
        let rows = sqlx::query(
            "SELECT n.node_id, n.owner_id, n.kind, n.display_name, n.policy_profile,
                    n.policy_revision, n.authorization_revision, n.created_at, n.revoked_at,
                    n.last_seen_at, ch.conversation_session_id, ch.node_channel_id
             FROM nodes n JOIN node_channels ch ON ch.node_id = n.node_id
             ORDER BY n.created_at ASC",
        )
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(node_from_row).collect()
    }

    pub async fn latest_sensor_readings(
        &self,
        node_selector: Option<&str>,
    ) -> anyhow::Result<Vec<LatestSensorReading>> {
        latest_sensor_readings(&self.pool, node_selector).await
    }

    pub async fn revoke_node(&self, node_id: &str) -> anyhow::Result<bool> {
        let now = Utc::now().to_rfc3339();
        let mut transaction = self.pool.begin().await?;
        let changed = sqlx::query(
            "UPDATE nodes SET revoked_at = COALESCE(revoked_at, ?),
                    policy_revision = policy_revision + 1,
                    authorization_revision = authorization_revision + 1
             WHERE node_id = ? AND revoked_at IS NULL",
        )
        .bind(&now)
        .bind(node_id)
        .execute(&mut *transaction)
        .await?;
        if changed.rows_affected() == 0 {
            transaction.commit().await?;
            return Ok(false);
        }
        sqlx::query(
            "UPDATE node_credentials SET state = 'revoked', revoked_at = ? WHERE node_id = ?",
        )
        .bind(&now)
        .bind(node_id)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "UPDATE node_sessions SET closed_at = ?, close_reason = 'node_revoked'
             WHERE node_id = ? AND closed_at IS NULL",
        )
        .bind(&now)
        .bind(node_id)
        .execute(&mut *transaction)
        .await?;
        sqlx::query(
            "INSERT INTO node_audit_events (node_id, event_type, created_at)
             VALUES (?, 'node_revoked', ?)",
        )
        .bind(node_id)
        .bind(&now)
        .execute(&mut *transaction)
        .await?;
        transaction.commit().await?;
        Ok(true)
    }
}

pub async fn latest_sensor_readings(
    pool: &SqlitePool,
    node_selector: Option<&str>,
) -> anyhow::Result<Vec<LatestSensorReading>> {
    let active_nodes = sqlx::query(
        "SELECT node_id, display_name FROM nodes
             WHERE revoked_at IS NULL ORDER BY created_at ASC",
    )
    .fetch_all(pool)
    .await?;
    let selected = if let Some(selector) = node_selector.map(str::trim).filter(|s| !s.is_empty()) {
        let mut matches = active_nodes.iter().filter(|row| {
            row.get::<String, _>("node_id") == selector
                || row
                    .get::<String, _>("display_name")
                    .eq_ignore_ascii_case(selector)
        });
        let first = matches
            .next()
            .ok_or_else(|| anyhow::anyhow!("No active Node matches '{selector}'"))?;
        anyhow::ensure!(matches.next().is_none(), "Node selector is ambiguous");
        first.get::<String, _>("node_id")
    } else {
        anyhow::ensure!(!active_nodes.is_empty(), "No active Nodes are enrolled");
        anyhow::ensure!(
            active_nodes.len() == 1,
            "Multiple active Nodes are enrolled; specify one by display name"
        );
        active_nodes[0].get::<String, _>("node_id")
    };
    let rows = sqlx::query(
        "SELECT n.display_name, n.last_seen_at, r.capability_id,
                    r.capability_version, r.value, r.unit, r.sample_uptime_ms, r.received_at
             FROM node_sensor_readings_latest r
             JOIN nodes n ON n.node_id = r.node_id
             WHERE r.node_id = ? AND n.revoked_at IS NULL
             ORDER BY r.capability_id ASC",
    )
    .bind(selected)
    .fetch_all(pool)
    .await?;
    rows.into_iter()
        .map(|row| {
            let received_at: String = row.get("received_at");
            let last_seen_at: Option<String> = row.get("last_seen_at");
            Ok(LatestSensorReading {
                display_name: row.get("display_name"),
                capability_id: row.get("capability_id"),
                capability_version: row.get::<i64, _>("capability_version") as u16,
                value: row.get("value"),
                unit: row.get("unit"),
                sample_uptime_ms: row.get::<i64, _>("sample_uptime_ms") as u64,
                received_at: parse_timestamp(&received_at)?,
                node_last_seen_at: last_seen_at.as_deref().map(parse_timestamp).transpose()?,
            })
        })
        .collect()
}

pub async fn node_health_snapshot(
    pool: &SqlitePool,
    node_selector: Option<&str>,
) -> anyhow::Result<NodeHealthSnapshot> {
    let active_nodes = sqlx::query(
        "SELECT node_id, display_name FROM nodes
         WHERE revoked_at IS NULL ORDER BY created_at ASC",
    )
    .fetch_all(pool)
    .await?;
    let selected = if let Some(selector) = node_selector.map(str::trim).filter(|s| !s.is_empty()) {
        let mut matches = active_nodes.iter().filter(|row| {
            row.get::<String, _>("node_id") == selector
                || row
                    .get::<String, _>("display_name")
                    .eq_ignore_ascii_case(selector)
        });
        let first = matches
            .next()
            .ok_or_else(|| anyhow::anyhow!("No active Node matches '{selector}'"))?;
        anyhow::ensure!(matches.next().is_none(), "Node selector is ambiguous");
        first.get::<String, _>("node_id")
    } else {
        anyhow::ensure!(!active_nodes.is_empty(), "No active Nodes are enrolled");
        anyhow::ensure!(
            active_nodes.len() == 1,
            "Multiple active Nodes are enrolled; specify one by display name"
        );
        active_nodes[0].get::<String, _>("node_id")
    };
    let row = sqlx::query(
        "SELECT display_name, last_seen_at, last_runtime_version,
                last_firmware_version, last_health_json
         FROM nodes WHERE node_id = ? AND revoked_at IS NULL",
    )
    .bind(selected)
    .fetch_one(pool)
    .await?;
    let last_seen_at: Option<String> = row.get("last_seen_at");
    let health_json: Option<String> = row.get("last_health_json");
    let health: serde_json::Value = health_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()?
        .unwrap_or_else(|| json!({}));
    let recovery = health
        .get("recovery")
        .filter(|value| !value.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()?;
    Ok(NodeHealthSnapshot {
        display_name: row.get("display_name"),
        last_seen_at: last_seen_at.as_deref().map(parse_timestamp).transpose()?,
        runtime_version: row.get("last_runtime_version"),
        firmware_version: row.get("last_firmware_version"),
        uptime_ms: health.get("uptime_ms").and_then(serde_json::Value::as_u64),
        free_internal_heap: health
            .get("free_internal_heap")
            .and_then(serde_json::Value::as_u64),
        largest_internal_allocation: health
            .get("largest_internal_allocation")
            .and_then(serde_json::Value::as_u64),
        psram_free: health.get("psram_free").and_then(serde_json::Value::as_u64),
        recovery,
    })
}

async fn append_event_tx(
    transaction: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    node_id: &str,
    turn_id: Option<&str>,
    event_type: &str,
    payload: serde_json::Value,
) -> anyhow::Result<u64> {
    let now = Utc::now();
    let result = sqlx::query(
        "INSERT INTO node_outbox
         (node_id, turn_id, event_type, payload_json, created_at, expires_at)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(node_id)
    .bind(turn_id)
    .bind(event_type)
    .bind(payload.to_string())
    .bind(now.to_rfc3339())
    .bind((now + Duration::hours(24)).to_rfc3339())
    .execute(&mut **transaction)
    .await?;
    Ok(result.last_insert_rowid() as u64)
}

fn parse_turn_state(value: &str) -> NodeTurnState {
    match value {
        "accepted" => NodeTurnState::Accepted,
        "thinking" => NodeTurnState::Thinking,
        "complete" => NodeTurnState::Complete,
        "cancelled" => NodeTurnState::Cancelled,
        _ => NodeTurnState::Error,
    }
}

fn node_from_row(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<NodeRecord> {
    let created_at: String = row.get("created_at");
    let revoked_at: Option<String> = row.get("revoked_at");
    let last_seen_at: Option<String> = row.get("last_seen_at");
    Ok(NodeRecord {
        node_id: row.get("node_id"),
        owner_id: row.get("owner_id"),
        kind: row.get("kind"),
        display_name: row.get("display_name"),
        policy_profile: row.get("policy_profile"),
        policy_revision: row.get::<i64, _>("policy_revision") as u64,
        authorization_revision: row.get::<i64, _>("authorization_revision") as u64,
        conversation_session_id: row.get("conversation_session_id"),
        node_channel_id: row.get("node_channel_id"),
        created_at: parse_timestamp(&created_at)?,
        revoked_at: revoked_at.as_deref().map(parse_timestamp).transpose()?,
        last_seen_at: last_seen_at.as_deref().map(parse_timestamp).transpose()?,
    })
}
