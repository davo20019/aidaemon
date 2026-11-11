//! Owner-approved restart of a reachable Node.
//!
//! The Node is outbound-only, so a restart waits in a queue for its next
//! heartbeat. The owner approves each request, the Node needs a
//! `device_restart` grant and a live `runtime.restart` capability, and the
//! command is delivered at most once. Delivery is not success: only a later
//! authenticated heartbeat from a new boot confirms the restart.

use std::sync::Arc;

use aidaemon_node_protocol::DeviceRestartCommand;
use anyhow::Context;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;

use crate::tools::owner_answers::OwnerAnsweredCalls;
use crate::traits::{
    ApprovalCall, EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, OwnerApprovalNeed,
    Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolDispatchFacts, ToolEvidenceCapability, ToolMutationEffects, ToolOutcomeStatus,
    ToolSemanticScope, ToolSemantics, ToolVerificationMode,
};
use crate::types::{ApprovalKind, ApprovalResponse, UserRole};

use super::domain::NodeAction;
use super::store::NodeStore;

pub const RESTART_CAPABILITY: &str = "runtime.restart";
pub const RESTART_SCHEMA: &str = "aidaemon.device.restart.v1";
const RESTART_REASON: &str = "owner_requested";

/// An approval that has not reached the Node by then must not reboot it later.
const PENDING_TTL_SECONDS: i64 = 300;
/// How long a delivered command may wait for a heartbeat from a new boot.
const CONFIRM_TIMEOUT_SECONDS: i64 = 180;
/// Minimum time between deliveries, so restarts cannot loop.
const COOLDOWN_SECONDS: i64 = 900;
use super::tool::FRESH_NODE_SECONDS;
/// Declining one restart binds the run for every Node restart.
const RESTART_EFFECT_KEY: &str = "node_restart";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRestartRecord {
    pub command_id: String,
    pub display_name: String,
    pub state: String,
    pub created_at: String,
    pub delivered_at: Option<String>,
    pub finished_at: Option<String>,
}

struct RestartTarget {
    node_id: String,
    display_name: String,
}

fn parse_time(value: &str) -> anyhow::Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)?.with_timezone(&Utc))
}

fn stored_uptime(uptime_ms: u64) -> i64 {
    i64::try_from(uptime_ms).unwrap_or(i64::MAX)
}

impl NodeStore {
    async fn restart_target(&self, selector: Option<&str>) -> anyhow::Result<RestartTarget> {
        let selector = selector.map(str::trim).filter(|value| !value.is_empty());
        let rows = sqlx::query(
            "SELECT node_id, display_name FROM nodes
             WHERE revoked_at IS NULL
               AND (?1 IS NULL OR node_id = ?1 OR display_name = ?1 COLLATE NOCASE)
             ORDER BY created_at ASC",
        )
        .bind(selector)
        .fetch_all(self.pool())
        .await?;
        anyhow::ensure!(!rows.is_empty(), "No active Node matches the target");
        anyhow::ensure!(
            rows.len() == 1,
            "More than one active Node matches; specify the exact Node display name"
        );
        Ok(RestartTarget {
            node_id: rows[0].get("node_id"),
            display_name: rows[0].get("display_name"),
        })
    }

    /// The Node a request may restart now, or the reason it may not.
    async fn eligible_restart_target(
        &self,
        selector: Option<&str>,
    ) -> anyhow::Result<RestartTarget> {
        let target = self.restart_target(selector).await?;
        let name = &target.display_name;
        let last_seen: Option<String> =
            sqlx::query_scalar("SELECT last_seen_at FROM nodes WHERE node_id = ?")
                .bind(&target.node_id)
                .fetch_one(self.pool())
                .await?;
        let seen_recently = match last_seen.as_deref() {
            Some(seen) => Utc::now() - parse_time(seen)? <= Duration::seconds(FRESH_NODE_SECONDS),
            None => false,
        };
        anyhow::ensure!(seen_recently, "Node '{name}' is not connected right now");
        anyhow::ensure!(
            self.is_authorized(&target.node_id, NodeAction::DeviceRestart)
                .await?,
            "Node '{name}' has no device_restart grant"
        );
        // A capability counts only if the latest heartbeat reported it.
        let capable: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_capabilities c
             JOIN nodes n ON n.node_id = c.node_id
             WHERE c.node_id = ? AND c.capability_id = ?
               AND c.capability_version >= 1 AND c.observed_at = n.last_seen_at",
        )
        .bind(&target.node_id)
        .bind(RESTART_CAPABILITY)
        .fetch_one(self.pool())
        .await?;
        anyhow::ensure!(
            capable == 1,
            "Node '{name}' does not report the {RESTART_CAPABILITY} capability; its firmware cannot restart on request"
        );
        self.settle_device_restart(&target.node_id, None).await?;
        let active: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM node_restart_commands
             WHERE node_id = ? AND state IN ('pending', 'delivered')",
        )
        .bind(&target.node_id)
        .fetch_one(self.pool())
        .await?;
        anyhow::ensure!(active == 0, "A restart of '{name}' is already in progress");
        let last_delivery: Option<String> = sqlx::query_scalar(
            "SELECT delivered_at FROM node_restart_commands
             WHERE node_id = ? AND delivered_at IS NOT NULL
             ORDER BY rowid DESC LIMIT 1",
        )
        .bind(&target.node_id)
        .fetch_optional(self.pool())
        .await?;
        if let Some(delivered_at) = last_delivery {
            let remaining =
                COOLDOWN_SECONDS - (Utc::now() - parse_time(&delivered_at)?).num_seconds();
            anyhow::ensure!(
                remaining <= 0,
                "Node '{name}' is in its restart cooldown for another {remaining} seconds"
            );
        }
        Ok(target)
    }

    pub async fn request_device_restart(
        &self,
        selector: Option<&str>,
    ) -> anyhow::Result<DeviceRestartRecord> {
        let target = self.eligible_restart_target(selector).await?;
        let now = Utc::now();
        let created_at = now.to_rfc3339();
        let command_id = format!("restart_{}", uuid::Uuid::new_v4().simple());
        let mut transaction = self.pool().begin().await?;
        // The unique index on active commands settles a concurrent request.
        sqlx::query(
            "INSERT INTO node_restart_commands
             (command_id, node_id, reason, state, created_at, expires_at)
             VALUES (?, ?, ?, 'pending', ?, ?)",
        )
        .bind(&command_id)
        .bind(&target.node_id)
        .bind(RESTART_REASON)
        .bind(&created_at)
        .bind((now + Duration::seconds(PENDING_TTL_SECONDS)).to_rfc3339())
        .execute(&mut *transaction)
        .await
        .map_err(|error| {
            let duplicate = error
                .as_database_error()
                .is_some_and(|database| database.is_unique_violation());
            if duplicate {
                anyhow::anyhow!(
                    "A restart of '{}' is already in progress",
                    target.display_name
                )
            } else {
                anyhow::Error::from(error)
            }
        })?;
        insert_restart_audit(
            &mut transaction,
            &target.node_id,
            &command_id,
            "device_restart_requested",
        )
        .await?;
        transaction.commit().await?;
        Ok(DeviceRestartRecord {
            command_id,
            display_name: target.display_name,
            state: "pending".to_string(),
            created_at,
            delivered_at: None,
            finished_at: None,
        })
    }

    pub async fn latest_device_restart(
        &self,
        selector: Option<&str>,
    ) -> anyhow::Result<Option<DeviceRestartRecord>> {
        let target = self.restart_target(selector).await?;
        self.settle_device_restart(&target.node_id, None).await?;
        let row = sqlx::query(
            "SELECT command_id, state, created_at, delivered_at, finished_at
             FROM node_restart_commands WHERE node_id = ?
             ORDER BY rowid DESC LIMIT 1",
        )
        .bind(&target.node_id)
        .fetch_optional(self.pool())
        .await?;
        Ok(row.map(|row| DeviceRestartRecord {
            command_id: row.get("command_id"),
            display_name: target.display_name,
            state: row.get("state"),
            created_at: row.get("created_at"),
            delivered_at: row.get("delivered_at"),
            finished_at: row.get("finished_at"),
        }))
    }

    /// Record what this authenticated heartbeat proves about an active command.
    pub(super) async fn advance_device_restart(
        &self,
        node_id: &str,
        boot_id: &str,
        uptime_ms: u64,
    ) -> anyhow::Result<()> {
        self.settle_device_restart(node_id, Some((boot_id, uptime_ms)))
            .await
    }

    /// Move an active command on. `heartbeat` is the evidence of a new boot;
    /// without one, only the deadlines can apply.
    async fn settle_device_restart(
        &self,
        node_id: &str,
        heartbeat: Option<(&str, u64)>,
    ) -> anyhow::Result<()> {
        let Some(row) = sqlx::query(
            "SELECT command_id, state, expires_at, delivered_at, delivered_boot_id,
                    delivered_uptime_ms
             FROM node_restart_commands
             WHERE node_id = ? AND state IN ('pending', 'delivered')",
        )
        .bind(node_id)
        .fetch_optional(self.pool())
        .await?
        else {
            return Ok(());
        };
        let command_id: String = row.get("command_id");
        let state: String = row.get("state");
        let now = Utc::now();
        let outcome = if state == "pending" {
            let expires_at: String = row.get("expires_at");
            (now >= parse_time(&expires_at)?).then_some(("expired", "device_restart_expired"))
        } else {
            let delivered_boot: Option<String> = row.get("delivered_boot_id");
            let delivered_uptime: Option<i64> = row.get("delivered_uptime_ms");
            let delivered_at: Option<String> = row.get("delivered_at");
            let delivered_at = delivered_at.context("delivered command has no delivery time")?;
            // A shipped firmware sends a constant boot id, so falling uptime
            // is also proof of a reboot.
            let new_boot = heartbeat.is_some_and(|(boot_id, uptime_ms)| {
                delivered_boot.as_deref() != Some(boot_id)
                    || delivered_uptime.is_some_and(|before| stored_uptime(uptime_ms) < before)
            });
            if new_boot {
                Some(("confirmed", "device_restart_confirmed"))
            } else if now - parse_time(&delivered_at)? >= Duration::seconds(CONFIRM_TIMEOUT_SECONDS)
            {
                Some(("timed_out", "device_restart_timed_out"))
            } else {
                None
            }
        };
        let Some((next, event)) = outcome else {
            return Ok(());
        };
        let mut transaction = self.pool().begin().await?;
        let changed = sqlx::query(
            "UPDATE node_restart_commands SET state = ?, finished_at = ?
             WHERE command_id = ? AND state = ?",
        )
        .bind(next)
        .bind(now.to_rfc3339())
        .bind(&command_id)
        .bind(&state)
        .execute(&mut *transaction)
        .await?
        .rows_affected();
        if changed == 1 {
            insert_restart_audit(&mut transaction, node_id, &command_id, event).await?;
        }
        transaction.commit().await?;
        Ok(())
    }

    /// Take the pending command for this heartbeat. It is marked delivered
    /// before the response is sent, so a lost response times out and is
    /// never sent twice.
    pub(super) async fn claim_device_restart(
        &self,
        node_id: &str,
        boot_id: &str,
        uptime_ms: u64,
    ) -> anyhow::Result<Option<DeviceRestartCommand>> {
        let mut transaction = self.pool().begin().await?;
        let claimed = sqlx::query(
            "UPDATE node_restart_commands
             SET state = 'delivered', delivered_at = ?, delivered_boot_id = ?,
                 delivered_uptime_ms = ?
             WHERE node_id = ? AND state = 'pending'
             RETURNING command_id, reason",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(boot_id)
        .bind(stored_uptime(uptime_ms))
        .bind(node_id)
        .fetch_optional(&mut *transaction)
        .await?;
        let Some(claimed) = claimed else {
            return Ok(None);
        };
        let command_id: String = claimed.get("command_id");
        insert_restart_audit(
            &mut transaction,
            node_id,
            &command_id,
            "device_restart_delivered",
        )
        .await?;
        transaction.commit().await?;
        Ok(Some(DeviceRestartCommand {
            schema: RESTART_SCHEMA.to_string(),
            command_id,
            reason: claimed.get("reason"),
        }))
    }
}

async fn insert_restart_audit(
    transaction: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    node_id: &str,
    command_id: &str,
    event_type: &str,
) -> anyhow::Result<()> {
    sqlx::query(
        "INSERT INTO node_audit_events (node_id, event_type, detail_json, created_at)
         VALUES (?, ?, ?, ?)",
    )
    .bind(node_id)
    .bind(event_type)
    .bind(json!({ "command_id": command_id }).to_string())
    .bind(Utc::now().to_rfc3339())
    .execute(&mut **transaction)
    .await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct NodeRestartArgs {
    action: String,
    #[serde(default)]
    node: Option<String>,
}

pub struct NodeRestartTool {
    store: Arc<NodeStore>,
    owner_answers: OwnerAnsweredCalls,
}

impl NodeRestartTool {
    pub fn new(store: Arc<NodeStore>) -> Self {
        Self {
            store,
            owner_answers: OwnerAnsweredCalls::default(),
        }
    }

    fn subject(node_id: &str) -> String {
        format!("node_restart:{node_id}")
    }

    fn is_request(arguments: &str) -> Option<bool> {
        serde_json::from_str::<NodeRestartArgs>(arguments)
            .ok()
            .map(|args| args.action == "request")
    }

    async fn request(
        &self,
        args: &NodeRestartArgs,
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let target = self
            .store
            .eligible_restart_target(args.node.as_deref())
            .await?;
        let approved = facts.session().is_some_and(|session| {
            self.owner_answers
                .take(session, &Self::subject(&target.node_id))
        });
        anyhow::ensure!(
            approved,
            "A Node restart requires the owner's approval through the agent kernel"
        );
        let record = self
            .store
            .request_device_restart(Some(&target.node_id))
            .await?;
        Ok(serde_json::to_string_pretty(&json!({
            "node": record.display_name,
            "command_id": record.command_id,
            "restart_status": record.state,
            "interpretation": "The restart is queued for the Node's next heartbeat. It has not happened yet; check the status action before claiming it did.",
        }))?)
    }

    async fn status(&self, args: &NodeRestartArgs) -> anyhow::Result<(String, ToolOutcomeStatus)> {
        let record = self
            .store
            .latest_device_restart(args.node.as_deref())
            .await?;
        let state = record
            .as_ref()
            .map_or("none", |record| record.state.as_str());
        let interpretation = match state {
            "confirmed" => "The Node reported from a new boot after receiving the restart.",
            "pending" => "The restart is waiting for the Node's next heartbeat.",
            "delivered" => "The Node was sent the restart, but no heartbeat from a new boot has arrived yet.",
            "expired" => "The Node never collected the restart, so it was not sent.",
            "timed_out" => "The restart was sent, but no heartbeat from a new boot arrived in time. Do not claim the Node restarted.",
            _ => "No restart has been requested for this Node.",
        };
        let output = serde_json::to_string_pretty(&json!({
            "command_id": record.as_ref().map(|record| record.command_id.as_str()),
            "node": record.as_ref().map(|record| record.display_name.as_str()),
            "restart_status": state,
            "created_at": record.as_ref().map(|record| record.created_at.as_str()),
            "delivered_at": record.as_ref().and_then(|record| record.delivered_at.as_deref()),
            "finished_at": record.as_ref().and_then(|record| record.finished_at.as_deref()),
            "interpretation": interpretation,
        }))?;
        let outcome = if state == "confirmed" {
            ToolOutcomeStatus::Succeeded
        } else {
            ToolOutcomeStatus::CompletedWithNegativeResult
        };
        Ok((output, outcome))
    }
}

#[async_trait]
impl Tool for NodeRestartTool {
    fn name(&self) -> &str {
        "node_restart"
    }

    fn description(&self) -> &str {
        "Restart a connected AIdaemon Node after the owner approves it, or check the last restart's status."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "node_restart",
            "description": "Ask a connected AIdaemon Node to restart, or read the status of its latest restart. Each request needs the owner's approval, a device_restart grant and firmware that reports the runtime.restart capability. A request only queues the restart for the Node's next heartbeat; it is done only when status reports confirmed. This cannot reach an offline Device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["request", "status"],
                        "description": "request queues a restart; status reports the latest one."
                    },
                    "node": {
                        "type": "string",
                        "description": "Optional Node display name. Omit when exactly one active Node is enrolled."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    /// The card authorizes this one restart, never a standing grant.
    fn approval_card_kind(&self, _request: &aidaemon_core::OwnerApprovalRequest) -> ApprovalKind {
        ApprovalKind::CommandOnce
    }

    async fn record_owner_answer(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
        response: Option<ApprovalResponse>,
    ) {
        if !matches!(response, Some(ApprovalResponse::AllowOnce)) {
            return;
        }
        let Some(session) = facts.session() else {
            return;
        };
        let Ok(args) = serde_json::from_value::<NodeRestartArgs>(call.arguments.clone()) else {
            return;
        };
        if let Ok(target) = self.store.restart_target(args.node.as_deref()).await {
            self.owner_answers
                .approve(session, &Self::subject(&target.node_id));
        }
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: NodeRestartArgs =
            serde_json::from_str(arguments).context("node_restart arguments must be valid JSON")?;
        anyhow::ensure!(
            facts.user_role == Some(UserRole::Owner),
            "Only the AIdaemon owner may restart a Node or read its restart status"
        );
        let (output, outcome_status) = match args.action.as_str() {
            "request" => (
                self.request(&args, facts).await?,
                // Queuing proves nothing about the restart itself.
                ToolOutcomeStatus::CompletedWithNegativeResult,
            ),
            "status" => self.status(&args).await?,
            other => anyhow::bail!("unknown node_restart action: {other}"),
        };
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(outcome_status),
                semantics: self.call_semantics(arguments),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

#[async_trait]
impl ToolSemantics for NodeRestartTool {
    fn dispatch_facts(&self) -> ToolDispatchFacts {
        ToolDispatchFacts {
            keeps_own_grants: true,
            ..Default::default()
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    /// Every request the Node could act on asks the owner. A request that
    /// would be refused anyway shows no card; execution returns the reason.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        let Ok(args) = serde_json::from_value::<NodeRestartArgs>(call.arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if args.action != "request"
            || facts.user_role != Some(UserRole::Owner)
            || facts.session().is_none()
        {
            return OwnerApprovalNeed::NotNeeded;
        }
        let Ok(target) = self
            .store
            .eligible_restart_target(args.node.as_deref())
            .await
        else {
            return OwnerApprovalNeed::NotNeeded;
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            effects: vec![aidaemon_core::EffectKey::named(
                aidaemon_core::EffectClass::Process,
                RESTART_EFFECT_KEY,
            )],
            summary: format!("Restart Node '{}'", target.display_name),
            subject: Self::subject(&target.node_id),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![
                "The Device reboots at its next heartbeat and stops whatever it is doing."
                    .to_string(),
            ],
        })
    }

    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        use aidaemon_core::{CallEffects, EffectClass, EffectKey};
        if Self::is_request(arguments) == Some(false) {
            return CallEffects::none();
        }
        let mut effects = CallEffects::unknown();
        effects
            .keys
            .push(EffectKey::named(EffectClass::Process, RESTART_EFFECT_KEY));
        effects
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let base = if Self::is_request(arguments) == Some(false) {
            ToolCallSemantics::observation()
        } else {
            ToolCallSemantics::observation_and_mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY)
        };
        base.with_verification_mode(ToolVerificationMode::ResultContent)
            .with_evidence(vec![ToolEvidenceCapability::new(
                ToolSemanticScope::ExternalRemote,
                &[EvidencePurpose::Outcome],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodesConfig;
    use crate::nodes::domain::{AuthenticatedNodeContext, NodeAction, CHILD_COMPANION_POLICY};
    use crate::nodes::ota::FirmwareRelease;
    use crate::nodes::protocol::{CapabilityObservation, HeartbeatRequest, HeartbeatResponse};
    use crate::nodes::service::NodeService;
    use crate::testing::{call_through_kernel, CallWithFacts};
    use crate::traits::{OwnerApprovalNeed, Tool, ToolCallContext, ToolSemantics};
    use crate::types::{ApprovalKind, ApprovalResponse, UserRole};
    use chrono::{Duration, Utc};
    use sqlx::sqlite::SqlitePoolOptions;
    use sqlx::Row;

    const BOOT: &str = "k10-runtime";
    const NODE: &str = "node-test";

    struct Fixture {
        store: Arc<NodeStore>,
        service: NodeService,
        context: AuthenticatedNodeContext,
    }

    fn restart_capability() -> Vec<CapabilityObservation> {
        vec![CapabilityObservation {
            capability_id: RESTART_CAPABILITY.to_string(),
            version: 1,
            limits: serde_json::json!({}),
        }]
    }

    fn ota_capability() -> CapabilityObservation {
        CapabilityObservation {
            capability_id: "firmware.ota".to_string(),
            version: 1,
            limits: serde_json::json!({"board": crate::nodes::ota::K10_BOARD_ID}),
        }
    }

    async fn fixture_with(config: NodesConfig, release: Option<FirmwareRelease>) -> Fixture {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::nodes::store::migrate(&pool).await.unwrap();
        let now = (Utc::now() - Duration::seconds(2)).to_rfc3339();
        sqlx::query(
            "INSERT INTO nodes
             (node_id, owner_id, kind, display_name, policy_profile, created_at, last_seen_at)
             VALUES (?, 'owner-test', 'k10', 'Test Companion', ?, ?, ?)",
        )
        .bind(NODE)
        .bind(CHILD_COMPANION_POLICY)
        .bind(&now)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
        let store = Arc::new(NodeStore::new(pool, [7; 32]));
        let mut service = NodeService::new(store.clone(), config);
        if let Some(release) = release {
            service = service.with_firmware_release(release);
        }
        for action in [NodeAction::ReportTelemetry, NodeAction::ReceiveOta] {
            store
                .set_authorization(NODE, action, true, serde_json::json!({}))
                .await
                .unwrap();
        }
        let context = AuthenticatedNodeContext {
            node_id: NODE.to_string(),
            node_session_id: "session-test".to_string(),
            credential_id: "credential-test".to_string(),
            kind: "k10".to_string(),
            policy_profile: CHILD_COMPANION_POLICY.to_string(),
            policy_revision: 1,
            authorization_revision: 1,
            conversation_session_id: "conversation-test".to_string(),
            node_channel_id: "channel-test".to_string(),
        };
        Fixture {
            store,
            service,
            context,
        }
    }

    async fn fixture() -> Fixture {
        fixture_with(NodesConfig::default(), None).await
    }

    impl Fixture {
        async fn grant(&self) {
            self.store
                .set_authorization(NODE, NodeAction::DeviceRestart, true, serde_json::json!({}))
                .await
                .unwrap();
        }

        async fn beat_with(
            &self,
            boot_id: &str,
            uptime_ms: u64,
            capabilities: Vec<CapabilityObservation>,
        ) -> HeartbeatResponse {
            self.service
                .heartbeat(
                    &self.context,
                    HeartbeatRequest {
                        boot_id: boot_id.to_string(),
                        uptime_ms,
                        runtime_version: "0.8.9".to_string(),
                        firmware_version: "0.8.9".to_string(),
                        battery_percent: None,
                        free_internal_heap: None,
                        largest_internal_allocation: None,
                        psram_free: None,
                        recovery: None,
                        capabilities,
                    },
                )
                .await
                .unwrap()
        }

        async fn beat(&self, uptime_ms: u64) -> HeartbeatResponse {
            self.beat_with(BOOT, uptime_ms, restart_capability()).await
        }

        /// A granted Node that has reported the capability.
        async fn ready() -> Self {
            let fixture = fixture().await;
            fixture.grant().await;
            fixture.beat(1_000).await;
            fixture
        }

        async fn state(&self) -> String {
            self.store
                .latest_device_restart(None)
                .await
                .unwrap()
                .expect("a restart command")
                .state
        }

        async fn age(&self, column: &str, seconds: i64) {
            let at = (Utc::now() - Duration::seconds(seconds)).to_rfc3339();
            sqlx::query(&format!(
                "UPDATE node_restart_commands SET {column} = ? WHERE node_id = ?"
            ))
            .bind(at)
            .bind(NODE)
            .execute(self.store.pool())
            .await
            .unwrap();
        }

        async fn audit_events(&self) -> Vec<String> {
            sqlx::query(
                "SELECT event_type FROM node_audit_events
                 WHERE node_id = ? AND event_type LIKE 'device_restart_%'
                 ORDER BY audit_id",
            )
            .bind(NODE)
            .fetch_all(self.store.pool())
            .await
            .unwrap()
            .iter()
            .map(|row| row.get("event_type"))
            .collect()
        }
    }

    fn owner() -> ToolCallContext {
        ToolCallContext {
            user_role: Some(UserRole::Owner),
            session_id: "owner-session".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn a_node_without_the_grant_is_refused_with_the_reason() {
        let fixture = fixture().await;
        fixture.beat(1_000).await;
        let error = fixture
            .store
            .request_device_restart(None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("device_restart"), "{error}");
        assert!(fixture
            .store
            .latest_device_restart(None)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn firmware_without_the_capability_is_refused_with_the_reason() {
        let fixture = fixture().await;
        fixture.grant().await;
        fixture.beat_with(BOOT, 1_000, Vec::new()).await;
        let error = fixture
            .store
            .request_device_restart(None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains(RESTART_CAPABILITY), "{error}");
    }

    #[tokio::test]
    async fn a_node_not_seen_recently_is_refused() {
        let fixture = Fixture::ready().await;
        sqlx::query("UPDATE nodes SET last_seen_at = ? WHERE node_id = ?")
            .bind((Utc::now() - Duration::seconds(600)).to_rfc3339())
            .bind(NODE)
            .execute(fixture.store.pool())
            .await
            .unwrap();
        let error = fixture
            .store
            .request_device_restart(None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not connected"), "{error}");
    }

    #[tokio::test]
    async fn the_command_is_delivered_exactly_once() {
        let fixture = Fixture::ready().await;
        let queued = fixture.store.request_device_restart(None).await.unwrap();
        assert_eq!(queued.state, "pending");
        assert_eq!(queued.display_name, "Test Companion");

        let command = fixture
            .beat(2_000)
            .await
            .device_restart
            .expect("the next heartbeat carries the command");
        assert_eq!(command.command_id, queued.command_id);
        assert_eq!(command.schema, RESTART_SCHEMA);
        assert_eq!(fixture.state().await, "delivered");

        assert!(fixture.beat(3_000).await.device_restart.is_none());
    }

    #[tokio::test]
    async fn delivery_does_not_confirm_until_a_new_boot_reports() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.beat(2_000).await;
        fixture.beat(3_000).await;
        assert_eq!(fixture.state().await, "delivered");

        fixture.beat(500).await;
        assert_eq!(fixture.state().await, "confirmed");
    }

    #[tokio::test]
    async fn a_new_boot_id_confirms_even_when_uptime_is_higher() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.beat(2_000).await;
        fixture
            .beat_with("k10boot-new", 9_000, restart_capability())
            .await;
        assert_eq!(fixture.state().await, "confirmed");
    }

    #[tokio::test]
    async fn firmware_that_stops_reporting_the_capability_gets_no_command() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        let response = fixture.beat_with(BOOT, 2_000, Vec::new()).await;
        assert!(response.device_restart.is_none());
        assert_eq!(fixture.state().await, "pending");
    }

    #[tokio::test]
    async fn a_revoked_grant_stops_a_queued_command() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture
            .store
            .set_authorization(
                NODE,
                NodeAction::DeviceRestart,
                false,
                serde_json::json!({}),
            )
            .await
            .unwrap();
        assert!(fixture.beat(2_000).await.device_restart.is_none());
        assert_eq!(fixture.state().await, "pending");
    }

    #[tokio::test]
    async fn a_stale_approval_expires_instead_of_rebooting_later() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.age("expires_at", 1).await;
        assert!(fixture.beat(2_000).await.device_restart.is_none());
        assert_eq!(fixture.state().await, "expired");
    }

    #[tokio::test]
    async fn a_delivered_command_times_out_without_a_new_boot() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.beat(2_000).await;
        fixture.age("delivered_at", 3_600).await;
        fixture.beat(3_000).await;
        assert_eq!(fixture.state().await, "timed_out");
    }

    #[tokio::test]
    async fn only_one_restart_can_be_active() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        let error = fixture
            .store
            .request_device_restart(None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("already"), "{error}");
    }

    #[tokio::test]
    async fn a_recent_delivery_blocks_another_restart_until_the_cooldown_passes() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.beat(2_000).await;
        fixture.beat(500).await;
        let error = fixture
            .store
            .request_device_restart(None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cooldown"), "{error}");

        fixture.age("delivered_at", 3_600).await;
        fixture.store.request_device_restart(None).await.unwrap();
    }

    #[tokio::test]
    async fn an_undelivered_expiry_does_not_start_the_cooldown() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.age("expires_at", 1).await;
        fixture.beat(2_000).await;
        fixture.store.request_device_restart(None).await.unwrap();
    }

    #[tokio::test]
    async fn a_firmware_offer_holds_the_restart_back() {
        let mut config = NodesConfig::default();
        config.ota.enabled = true;
        let release = FirmwareRelease::signed_for_test(b"synthetic image", "9.9.9", 5);
        let fixture = fixture_with(config, Some(release)).await;
        fixture.grant().await;
        let mut capabilities = restart_capability();
        capabilities.push(ota_capability());
        fixture.beat_with(BOOT, 1_000, capabilities.clone()).await;
        fixture.store.request_device_restart(None).await.unwrap();

        let response = fixture.beat_with(BOOT, 2_000, capabilities).await;
        assert!(response.firmware_update.is_some());
        assert!(response.device_restart.is_none());
        assert_eq!(fixture.state().await, "pending");
    }

    #[tokio::test]
    async fn every_transition_is_audited() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.beat(2_000).await;
        fixture.beat(500).await;
        assert_eq!(
            fixture.audit_events().await,
            [
                "device_restart_requested",
                "device_restart_delivered",
                "device_restart_confirmed"
            ]
        );
    }

    #[tokio::test]
    async fn each_request_needs_its_own_owner_approval() {
        let fixture = Fixture::ready().await;
        let tool = NodeRestartTool::new(fixture.store.clone());
        assert!(tool.dispatch_facts().keeps_own_grants);

        let request = r#"{"action":"request"}"#;
        let call = call_through_kernel(&tool, request, owner(), Some(ApprovalResponse::AllowOnce))
            .await
            .unwrap();
        let card = call.asked.expect("the owner is asked");
        assert!(card.summary.contains("Test Companion"));
        assert!(
            matches!(tool.approval_card_kind(&card), ApprovalKind::CommandOnce),
            "the card must not offer a standing grant"
        );
        assert_eq!(fixture.state().await, "pending");

        // Clear the first command so only the missing approval can refuse this.
        fixture.age("expires_at", 1).await;
        fixture.beat(2_000).await;
        assert_eq!(fixture.state().await, "expired");
        let error = tool.call_with(request, owner()).await.unwrap_err();
        assert!(error.to_string().contains("approval"), "{error}");
        assert_eq!(fixture.state().await, "expired");
    }

    #[tokio::test]
    async fn a_decline_queues_nothing() {
        let fixture = Fixture::ready().await;
        let tool = NodeRestartTool::new(fixture.store.clone());
        call_through_kernel(
            &tool,
            r#"{"action":"request"}"#,
            owner(),
            Some(ApprovalResponse::Deny),
        )
        .await
        .unwrap();
        assert!(fixture
            .store
            .latest_device_restart(None)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn a_non_owner_can_neither_request_nor_read() {
        let fixture = Fixture::ready().await;
        let tool = NodeRestartTool::new(fixture.store.clone());
        for action in ["request", "status"] {
            let error = tool
                .call_with(
                    &format!(r#"{{"action":"{action}"}}"#),
                    ToolCallContext {
                        user_role: Some(UserRole::Public),
                        session_id: "public-session".to_string(),
                        ..Default::default()
                    },
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains("owner"), "{error}");
        }
    }

    #[tokio::test]
    async fn status_does_not_ask_and_reports_the_state() {
        let fixture = Fixture::ready().await;
        fixture.store.request_device_restart(None).await.unwrap();
        fixture.beat(2_000).await;
        let tool = NodeRestartTool::new(fixture.store.clone());
        let arguments = serde_json::json!({"action": "status"});
        let need = tool
            .owner_approval_request(&crate::traits::ApprovalCall::as_sent(&arguments), &owner())
            .await;
        assert!(matches!(need, OwnerApprovalNeed::NotNeeded));
        let output = tool
            .call_with(r#"{"action":"status"}"#, owner())
            .await
            .unwrap();
        assert!(output.contains("delivered"), "{output}");
    }

    #[tokio::test]
    async fn a_request_that_would_be_refused_shows_no_card() {
        let fixture = fixture().await;
        fixture.beat(1_000).await;
        let tool = NodeRestartTool::new(fixture.store.clone());
        let arguments = serde_json::json!({"action": "request"});
        let need = tool
            .owner_approval_request(&crate::traits::ApprovalCall::as_sent(&arguments), &owner())
            .await;
        assert!(matches!(need, OwnerApprovalNeed::NotNeeded));
        let error = tool
            .call_with(r#"{"action":"request"}"#, owner())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("device_restart"), "{error}");
    }

    #[test]
    fn the_restart_grant_parses() {
        assert_eq!(
            NodeAction::parse("device_restart").unwrap(),
            NodeAction::DeviceRestart
        );
    }
}
