use serde::{Deserialize, Serialize};

const MAX_MANDATE_OBJECTIVE_TEXT: usize = 2 * 1024;
const MAX_MANDATE_POLICY_ENTRIES: usize = 16;
const MAX_MANDATE_POLICY_ENTRY_TEXT: usize = 500;
const MAX_MANDATE_POLICY_TEXT: usize = 8 * 1024;
const MAX_MANDATE_AUTHORITY_JSON: usize = 16 * 1024;
const MAX_DECISION_RATIONALE_TEXT: usize = 2 * 1024;
const MAX_DECISION_OBSERVATIONS: usize = 8;
const MAX_DECISION_OBSERVATION_TEXT: usize = 750;
const MAX_DECISION_OBSERVATIONS_JSON: usize = 6 * 1024;
const MAX_DECISION_QUESTION_TEXT: usize = 500;
const MAX_INTENTION_DESCRIPTION_TEXT: usize = 1024;
const MAX_INTENTION_METADATA_TEXT: usize = 4 * 1024;
const MAX_MANDATE_STRATEGY_BODY_TEXT: usize = 16 * 1024;
const MAX_MANDATE_LEARNING_NOTE_TEXT: usize = 1024;
const MAX_MANDATE_EVIDENCE_REFS: usize = 16;

fn text_within(value: &str, max: usize) -> bool {
    value.chars().count() <= max && value.len() <= max
}

fn canonical_text_within(value: &str, max: usize) -> bool {
    !value.is_empty() && value.trim() == value && text_within(value, max)
}

fn canonical_scoped_resource_suffix(value: &str) -> bool {
    if value.is_empty() || value.len() > 192 {
        return false;
    }
    let bytes = value.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-') {
            index += 1;
        } else if byte == b'%'
            && index + 2 < bytes.len()
            && bytes[index + 1].is_ascii_hexdigit()
            && bytes[index + 2].is_ascii_hexdigit()
            && !bytes[index + 1].is_ascii_lowercase()
            && !bytes[index + 2].is_ascii_lowercase()
        {
            index += 3;
        } else {
            return false;
        }
    }
    true
}

/// Owner-approved authority envelope for autonomous work.
///
/// A mandate delegates an objective and a bounded set of actions to the agent.
/// It is deliberately separate from a personal goal (a desired outcome) and
/// from an intention (the agent's revocable commitment for one decision cycle).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MandateStatus {
    Active,
    Paused,
    AwaitingInput,
    Completed,
    Cancelled,
}

impl MandateStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::AwaitingInput => "awaiting_input",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "active" => Some(Self::Active),
            "paused" => Some(Self::Paused),
            "awaiting_input" => Some(Self::AwaitingInput),
            "completed" => Some(Self::Completed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }

    /// Legal authority-epoch transitions. Terminal states never reopen.
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Active,
                Self::Paused | Self::AwaitingInput | Self::Completed | Self::Cancelled
            ) | (Self::Paused, Self::Active | Self::Cancelled)
                | (Self::AwaitingInput, Self::Active | Self::Cancelled)
        )
    }

    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled)
    }
}

impl std::fmt::Display for MandateStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Mandate {
    pub id: String,
    /// Continuous orchestration goal used by the durable scheduler/executor.
    pub goal_id: String,
    /// Optional personal goal that motivated this delegation.
    pub source_goal_id: Option<String>,
    pub objective: String,
    pub status: MandateStatus,
    pub authority: MandateAuthority,
    /// Owner-confirmed, immutable strategy material. This may guide how the
    /// objective is pursued, but it never contributes authority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strategy: Option<MandateStrategySnapshot>,
    /// Typed reason for a non-active, non-terminal lifecycle state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suspension: Option<MandateSuspension>,
    pub constraints: Vec<String>,
    pub success_criteria: Vec<String>,
    pub stop_conditions: Vec<String>,
    /// Lower and upper bounds for an agent-selected next review.
    pub min_review_secs: i64,
    pub max_review_secs: i64,
    pub default_review_secs: i64,
    /// Durable next wake time for the MAPE-K deliberation loop.
    pub next_review_at: String,
    /// Short-lived single-writer lease used by heartbeat dispatch.
    pub review_lease_token: Option<String>,
    pub review_lease_expires_at: Option<String>,
    pub expires_at: Option<String>,
    /// Durable proof that the owner explicitly confirmed this delegation.
    /// Pending-confirmation mandates keep this NULL and cannot be activated or
    /// resumed through ordinary lifecycle transitions.
    pub confirmed_at: Option<String>,
    /// Authority epoch, incremented whenever owner-controlled configuration or
    /// lifecycle state changes so stale decisions and updates fail closed.
    pub version: i64,
    pub created_by_session: String,
    pub created_at: String,
    pub updated_at: String,
}

impl Mandate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        goal_id: &str,
        source_goal_id: Option<String>,
        objective: &str,
        created_by_session: &str,
        authority: MandateAuthority,
        min_review_secs: i64,
        max_review_secs: i64,
        default_review_secs: i64,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let next_review_at =
            (chrono::Utc::now() + chrono::Duration::seconds(default_review_secs)).to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            source_goal_id,
            objective: objective.trim().to_string(),
            status: MandateStatus::Active,
            authority,
            strategy: None,
            suspension: None,
            constraints: Vec::new(),
            success_criteria: Vec::new(),
            stop_conditions: Vec::new(),
            min_review_secs,
            max_review_secs,
            default_review_secs,
            next_review_at,
            review_lease_token: None,
            review_lease_expires_at: None,
            expires_at: None,
            // `new` constructs an active mandate. Callers creating a deferred
            // controller must explicitly clear this proof before persistence.
            confirmed_at: Some(now.clone()),
            version: 1,
            created_by_session: created_by_session.to_string(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn is_active(&self) -> bool {
        if self.status != MandateStatus::Active || self.confirmed_at.is_none() {
            return false;
        }
        match self.expires_at.as_deref() {
            None => true,
            Some(value) => chrono::DateTime::parse_from_rfc3339(value)
                .is_ok_and(|expires| expires > chrono::Utc::now()),
        }
    }

    pub fn clamp_review_secs(&self, requested: Option<i64>) -> i64 {
        requested
            .unwrap_or(self.default_review_secs)
            .clamp(self.min_review_secs, self.max_review_secs)
    }

    /// Content bounds shared by every mandate ingestion path. Persistence
    /// calls this again so callers cannot bypass the owner-facing tool schema.
    pub fn validate_content_bounds(&self) -> Result<(), String> {
        if !canonical_text_within(&self.objective, MAX_MANDATE_OBJECTIVE_TEXT) {
            return Err(
                "mandate objective must be canonical non-empty text of at most 2 KiB in both characters and bytes"
                    .to_string(),
            );
        }
        if let Some(strategy) = self.strategy.as_ref() {
            strategy.validate()?;
        }
        if let Some(suspension) = self.suspension.as_ref() {
            suspension.validate()?;
        }

        let mut total_chars = 0usize;
        let mut total_bytes = 0usize;
        for (label, values) in [
            ("constraints", &self.constraints),
            ("success criteria", &self.success_criteria),
            ("stop conditions", &self.stop_conditions),
        ] {
            if values.len() > MAX_MANDATE_POLICY_ENTRIES {
                return Err(format!("mandate {label} cannot exceed 16 entries"));
            }
            for value in values {
                if !canonical_text_within(value, MAX_MANDATE_POLICY_ENTRY_TEXT) {
                    return Err(format!(
                        "each mandate {label} entry must be canonical non-empty text of at most 500 characters and bytes"
                    ));
                }
                total_chars = total_chars.saturating_add(value.chars().count());
                total_bytes = total_bytes.saturating_add(value.len());
            }
        }
        if total_chars > MAX_MANDATE_POLICY_TEXT || total_bytes > MAX_MANDATE_POLICY_TEXT {
            return Err(
                "mandate constraints, success criteria, and stop conditions exceed their combined 8 KiB character/byte bound"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// A content-addressed copy of one owner-approved skill. The snapshot body is
/// persisted so a later filesystem edit cannot silently change a live mandate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateStrategySnapshot {
    pub skill_name: String,
    pub snapshot_version: u16,
    pub content_sha256: String,
    pub description: String,
    pub body: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl MandateStrategySnapshot {
    pub const SCHEMA_VERSION: u16 = 1;

    pub fn validate(&self) -> Result<(), String> {
        if self.snapshot_version != Self::SCHEMA_VERSION {
            return Err("unsupported mandate strategy snapshot version".to_string());
        }
        if !canonical_text_within(&self.skill_name, 256) {
            return Err(
                "strategy skill_name must be canonical text of at most 256 bytes".to_string(),
            );
        }
        if self.content_sha256.len() != 64
            || !self
                .content_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("strategy content_sha256 must be a 64-character hex digest".to_string());
        }
        if !text_within(&self.description, 1024)
            || !canonical_text_within(&self.body, MAX_MANDATE_STRATEGY_BODY_TEXT)
        {
            return Err(
                "strategy description/body exceed their bounded canonical form".to_string(),
            );
        }
        if self
            .source
            .as_deref()
            .is_some_and(|source| !canonical_text_within(source, 512))
        {
            return Err("strategy source exceeds its canonical 512-byte bound".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MandateSuspensionKind {
    OwnerPaused,
    AwaitingAnswer,
    ReconciliationRequired,
    ExecutionLeaseLost,
    ReviewFailed,
    AuthorityRevokedWithUnresolvedMutation,
}

impl MandateSuspensionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OwnerPaused => "owner_paused",
            Self::AwaitingAnswer => "awaiting_answer",
            Self::ReconciliationRequired => "reconciliation_required",
            Self::ExecutionLeaseLost => "execution_lease_lost",
            Self::ReviewFailed => "review_failed",
            Self::AuthorityRevokedWithUnresolvedMutation => {
                "authority_revoked_with_unresolved_mutation"
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateSuspension {
    pub kind: MandateSuspensionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_cycle_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_run_id: Option<String>,
    pub created_at: String,
}

impl MandateSuspension {
    pub fn new(kind: MandateSuspensionKind, reason_code: Option<String>) -> Self {
        Self {
            kind,
            reason_code,
            decision_cycle_id: None,
            goal_run_id: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self
            .reason_code
            .as_deref()
            .is_some_and(|value| !canonical_text_within(value, 256))
            || self
                .decision_cycle_id
                .as_deref()
                .is_some_and(|value| !canonical_text_within(value, 256))
            || self
                .goal_run_id
                .as_deref()
                .is_some_and(|value| !canonical_text_within(value, 256))
        {
            return Err("mandate suspension identifiers exceed their canonical bounds".to_string());
        }
        chrono::DateTime::parse_from_rfc3339(&self.created_at)
            .map_err(|_| "mandate suspension created_at must be RFC3339".to_string())?;
        Ok(())
    }
}

/// Deterministic action policy attached to a mandate.
///
/// `allowed_tools` applies to both observations and mutations so a mandate
/// cannot read unrelated private state and then exfiltrate it through an
/// otherwise-authorized output tool. Patterns are exact names or a scoped
/// exact v1 data-tool names only; wildcard authority is rejected so a future
/// adapter cannot silently inherit an old delegation.
/// V1 rejects MCP tools until their manifests and effect semantics can be
/// owner-pinned instead of trusted from a live server advertisement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateAuthority {
    #[serde(default = "default_allow_observations")]
    pub allow_observations: bool,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    #[serde(default)]
    pub allowed_mutation_effects: Vec<String>,
    #[serde(default)]
    pub allowed_target_prefixes: Vec<String>,
    #[serde(default)]
    pub max_mutating_actions_per_cycle: u32,
    /// Cross-cycle dispatch ceiling. Every claimed mutation attempt, including
    /// a failed or ambiguous one, consumes one slot for a rolling 24-hour
    /// window; a reservation abandoned before the final I/O claim does not.
    #[serde(default)]
    pub max_mutating_actions_per_rolling_24h: u32,
    /// Minimum wall-clock separation between two claimed mutation dispatches
    /// for this mandate. Observation cycles are deliberately unaffected.
    #[serde(default)]
    pub min_seconds_between_mutations: u32,
}

fn default_allow_observations() -> bool {
    true
}

impl Default for MandateAuthority {
    fn default() -> Self {
        Self {
            allow_observations: true,
            allowed_tools: Vec::new(),
            allowed_mutation_effects: Vec::new(),
            allowed_target_prefixes: Vec::new(),
            max_mutating_actions_per_cycle: 0,
            max_mutating_actions_per_rolling_24h: 0,
            min_seconds_between_mutations: 0,
        }
    }
}

impl MandateAuthority {
    /// V1 delegates only directly governed network data adapters. Protocol
    /// controls are classified separately and never gain authority merely by
    /// appearing in this list.
    pub const DELEGABLE_DATA_TOOLS: &'static [&'static str] = &["http_request", "web_fetch"];
    /// Adapters that can execute an opaque nested action loop. Their inner
    /// reads and writes cannot be target-scoped or metered one-by-one by the
    /// mandate gate, so they are never delegable as a single allowed tool.
    pub const NON_DELEGABLE_TOOLS: &'static [&'static str] = &[
        "cli_agent",
        "terminal",
        "run_command",
        "browser",
        "computer_use",
        "health_probe",
        "scheduled_goal_runs",
        "read_file",
        "write_file",
        "edit_file",
        "search_files",
        "project_inspect",
        "send_file",
        "git_info",
        "git_commit",
        "check_environment",
    ];

    pub const EFFECT_NAMES: &'static [&'static str] = &[
        "local_source_write",
        "local_workspace_write",
        "local_derived_write",
        "repository_write",
        "remote_mutation",
        "remote_deploy",
        "external_delivery",
        "process_state",
        "configuration",
        "destructive",
        "unspecified",
    ];

    pub fn validate(&self) -> Result<(), String> {
        let authority_text = self
            .allowed_tools
            .iter()
            .chain(&self.allowed_mutation_effects)
            .chain(&self.allowed_target_prefixes);
        let authority_text_chars = authority_text
            .clone()
            .map(|value| value.chars().count())
            .sum::<usize>();
        let authority_text_bytes = authority_text.map(String::len).sum::<usize>();
        if authority_text_chars > MAX_MANDATE_AUTHORITY_JSON
            || authority_text_bytes > MAX_MANDATE_AUTHORITY_JSON
        {
            return Err(
                "mandate authority exceeds its total 16 KiB text character/byte bound".to_string(),
            );
        }
        let serialized = serde_json::to_string(self)
            .map_err(|error| format!("mandate authority cannot be serialized: {error}"))?;
        if !text_within(&serialized, MAX_MANDATE_AUTHORITY_JSON) {
            return Err(
                "mandate authority exceeds its total serialized 16 KiB character/byte bound"
                    .to_string(),
            );
        }
        if self.allowed_tools.len() > 64
            || self.allowed_target_prefixes.len() > 64
            || self.allowed_mutation_effects.len() > Self::EFFECT_NAMES.len()
        {
            return Err("mandate authority lists exceed their bounded size".to_string());
        }
        for (index, pattern) in self.allowed_tools.iter().enumerate() {
            let trimmed = pattern.trim();
            let wildcard_count = trimmed.bytes().filter(|byte| *byte == b'*').count();
            if trimmed.is_empty() || trimmed != pattern || wildcard_count != 0 {
                return Err(
                    "allowed_tools must use canonical exact names; wildcards are not allowed"
                        .to_string(),
                );
            }
            if !text_within(pattern, 256)
                || self.allowed_tools[..index]
                    .iter()
                    .any(|seen| seen == pattern)
            {
                return Err(
                    "allowed_tools must be unique and at most 256 characters and bytes each"
                        .to_string(),
                );
            }
            if trimmed.starts_with("mcp__") {
                return Err(
                    "MCP tools cannot be delegated by a v1 mandate because server-advertised safety hints are not owner-pinned authority facts; use a directly governed adapter such as http_request"
                        .to_string(),
                );
            }
            if !(Self::DELEGABLE_DATA_TOOLS.contains(&trimmed)
                || (cfg!(test) && trimmed == "mandate_mutation_spy"))
            {
                return Err(format!(
                    "{trimmed} is not a directly governed v1 mandate data tool"
                ));
            }
            let covered_non_delegable = Self::NON_DELEGABLE_TOOLS.iter().find(|tool_name| {
                trimmed == **tool_name
                    || trimmed
                        .strip_suffix('*')
                        .is_some_and(|prefix| tool_name.starts_with(prefix))
            });
            if let Some(tool_name) = covered_non_delegable {
                return Err(format!(
                    "{tool_name} cannot be delegated by a mandate because its nested actions cannot be target-scoped and metered individually"
                ));
            }
        }
        if let Some(invalid) = self.allowed_mutation_effects.iter().find(|effect| {
            effect.trim() != effect.as_str() || !Self::EFFECT_NAMES.contains(&effect.as_str())
        }) {
            return Err(format!(
                "unknown mutation effect `{invalid}`; valid effects: {}",
                Self::EFFECT_NAMES.join(", ")
            ));
        }
        if self
            .allowed_mutation_effects
            .iter()
            .enumerate()
            .any(|(index, effect)| {
                self.allowed_mutation_effects[..index]
                    .iter()
                    .any(|seen| seen == effect)
            })
        {
            return Err("allowed_mutation_effects must be unique".to_string());
        }
        if self.max_mutating_actions_per_cycle > 0 && self.allowed_tools.is_empty() {
            return Err(
                "max_mutating_actions_per_cycle is non-zero but no mutation tools are allowed"
                    .to_string(),
            );
        }
        if self.max_mutating_actions_per_cycle > 0 && self.allowed_mutation_effects.is_empty() {
            return Err(
                "max_mutating_actions_per_cycle is non-zero but no mutation effects are allowed"
                    .to_string(),
            );
        }
        if self.max_mutating_actions_per_cycle > 0 && self.allowed_target_prefixes.is_empty() {
            return Err(
                "mutation-enabled mandates require explicit typed target prefixes".to_string(),
            );
        }
        if !self.allowed_tools.is_empty() && self.allowed_target_prefixes.is_empty() {
            return Err(
                "delegated observations require explicit typed target prefixes".to_string(),
            );
        }
        if self.max_mutating_actions_per_cycle == 0 {
            if self.max_mutating_actions_per_rolling_24h != 0
                || self.min_seconds_between_mutations != 0
            {
                return Err(
                    "rolling mutation quota and cooldown require a positive per-cycle mutation budget"
                        .to_string(),
                );
            }
        } else {
            if self.max_mutating_actions_per_rolling_24h == 0 {
                return Err(
                    "mutation-enabled mandates require max_mutating_actions_per_rolling_24h"
                        .to_string(),
                );
            }
            if self.max_mutating_actions_per_rolling_24h > 24 {
                return Err(
                    "max_mutating_actions_per_rolling_24h cannot exceed the hard ceiling of 24"
                        .to_string(),
                );
            }
            if self.max_mutating_actions_per_rolling_24h < self.max_mutating_actions_per_cycle {
                return Err(
                    "max_mutating_actions_per_rolling_24h must be at least the per-cycle mutation budget"
                        .to_string(),
                );
            }
            if self.min_seconds_between_mutations < 900 {
                return Err(
                    "mutation-enabled mandates require at least 900 seconds between mutations"
                        .to_string(),
                );
            }
        }
        for (index, prefix) in self.allowed_target_prefixes.iter().enumerate() {
            if prefix.trim().is_empty()
                || prefix.trim() != prefix
                || !text_within(prefix, 2_048)
                || self.allowed_target_prefixes[..index]
                    .iter()
                    .any(|seen| seen == prefix)
            {
                return Err(
                    "allowed_target_prefixes must be unique non-empty canonical prefixes of at most 2048 characters and bytes"
                        .to_string(),
                );
            }
            if prefix.contains("://") {
                let url = reqwest::Url::parse(prefix)
                    .map_err(|_| "URL target prefixes must be canonical URLs".to_string())?;
                if !matches!(url.scheme(), "http" | "https")
                    || url.host_str().is_none()
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || url.fragment().is_some()
                    || matches!(url.path(), "" | "/")
                {
                    return Err(
                        "URL target prefixes require an http(s) origin plus a non-root path and cannot contain userinfo or fragments"
                            .to_string(),
                    );
                }
            } else {
                let suffix = prefix
                    .strip_prefix("auth_profile:")
                    .or_else(|| prefix.strip_prefix("account:"));
                if !suffix.is_some_and(canonical_scoped_resource_suffix) {
                    return Err(
                        "v1 resource targets must be exact auth_profile: or account: identifiers with a non-empty canonical suffix"
                            .to_string(),
                    );
                }
            }
        }
        Ok(())
    }

    pub fn allows_tool(&self, tool_name: &str) -> bool {
        self.allowed_tools.iter().any(|raw| raw == tool_name)
    }

    pub fn allows_effect(&self, effect: &str) -> bool {
        self.allowed_mutation_effects
            .iter()
            .any(|allowed| allowed.trim() == effect)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateDecisionOutcome {
    Act,
    Wait,
    Ask,
    Stop,
}

/// Why a STOP decision may close a mandate. Evidence-dependent reasons must
/// cite one or more current-run structured tool receipts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MandateTerminationKind {
    SuccessCriteriaSatisfied,
    StopConditionMet,
    SafetyTermination,
}

impl MandateTerminationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SuccessCriteriaSatisfied => "success_criteria_satisfied",
            Self::StopConditionMet => "stop_condition_met",
            Self::SafetyTermination => "safety_termination",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "success_criteria_satisfied" => Some(Self::SuccessCriteriaSatisfied),
            "stop_condition_met" => Some(Self::StopConditionMet),
            "safety_termination" => Some(Self::SafetyTermination),
            _ => None,
        }
    }

    pub const fn requires_receipt_evidence(self) -> bool {
        matches!(
            self,
            Self::SuccessCriteriaSatisfied | Self::StopConditionMet
        )
    }
}

impl MandateDecisionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Act => "act",
            Self::Wait => "wait",
            Self::Ask => "ask",
            Self::Stop => "stop",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "act" => Some(Self::Act),
            "wait" => Some(Self::Wait),
            "ask" => Some(Self::Ask),
            "stop" => Some(Self::Stop),
            _ => None,
        }
    }
}

/// One durable deliberation result for one goal run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateDecisionCycle {
    pub id: String,
    pub mandate_id: String,
    pub goal_run_id: String,
    /// Owner-policy revision observed when this decision was committed.
    pub mandate_version: i64,
    pub outcome: MandateDecisionOutcome,
    pub rationale: String,
    /// Sourced observations used by the deliberator, serialized as JSON.
    pub belief_snapshot: Option<String>,
    /// Exact current-run tool-call receipts that support the belief or STOP
    /// decision. These identifiers are verified against durable events before
    /// the decision is committed.
    #[serde(default)]
    pub evidence_receipt_ids: Vec<String>,
    pub question: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_kind: Option<MandateTerminationKind>,
    /// Exact owner-authored success/stop entry matched by a STOP decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_match: Option<String>,
    pub reconsider_at: Option<String>,
    /// Number of mutation attempts authorized during this cycle.
    pub action_attempts: i64,
    pub created_at: String,
    pub updated_at: String,
}

impl MandateDecisionCycle {
    pub fn new(
        mandate_id: &str,
        goal_run_id: &str,
        outcome: MandateDecisionOutcome,
        rationale: &str,
        mandate_version: i64,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            mandate_id: mandate_id.to_string(),
            goal_run_id: goal_run_id.to_string(),
            mandate_version,
            outcome,
            rationale: rationale.trim().to_string(),
            belief_snapshot: None,
            evidence_receipt_ids: Vec::new(),
            question: None,
            termination_kind: None,
            termination_match: None,
            reconsider_at: None,
            action_attempts: 0,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn validate_content_bounds(&self) -> Result<(), String> {
        if !canonical_text_within(&self.rationale, MAX_DECISION_RATIONALE_TEXT) {
            return Err(
                "decision rationale must be canonical non-empty text of at most 2 KiB in both characters and bytes"
                    .to_string(),
            );
        }
        if let Some(snapshot) = self.belief_snapshot.as_deref() {
            if !text_within(snapshot, MAX_DECISION_OBSERVATIONS_JSON) {
                return Err(
                    "decision observations exceed their serialized 6 KiB character/byte bound"
                        .to_string(),
                );
            }
            let observations: Vec<String> = serde_json::from_str(snapshot)
                .map_err(|_| "decision observations must be a JSON string array".to_string())?;
            if observations.len() > MAX_DECISION_OBSERVATIONS {
                return Err("decision observations cannot exceed 8 entries".to_string());
            }
            if observations
                .iter()
                .any(|value| !canonical_text_within(value, MAX_DECISION_OBSERVATION_TEXT))
            {
                return Err(
                    "each decision observation must be canonical non-empty text of at most 750 characters and bytes"
                        .to_string(),
                );
            }
        }
        if let Some(question) = self.question.as_deref() {
            if !canonical_text_within(question, MAX_DECISION_QUESTION_TEXT) {
                return Err(
                    "decision question must be canonical non-empty text of at most 500 characters and bytes"
                        .to_string(),
                );
            }
        }
        if self.evidence_receipt_ids.len() > MAX_MANDATE_EVIDENCE_REFS
            || self.evidence_receipt_ids.iter().any(|value| {
                !canonical_text_within(value, 256)
                    || !value.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':' | b'.')
                    })
            })
        {
            return Err("decision evidence receipt IDs exceed their canonical bounds".to_string());
        }
        if let Some(matched) = self.termination_match.as_deref() {
            if !canonical_text_within(matched, MAX_MANDATE_POLICY_ENTRY_TEXT) {
                return Err("decision termination match exceeds its canonical bound".to_string());
            }
        }
        Ok(())
    }
}

/// Bounded, mandate-local strategy learning. Notes are advisory continuity,
/// never authority, and every note must cite durable evidence from this same
/// mandate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateLearningNote {
    pub id: String,
    pub mandate_id: String,
    pub mandate_version: i64,
    pub learned_in_decision_cycle_id: String,
    pub summary: String,
    pub evidence_receipt_ids: Vec<String>,
    pub created_at: String,
}

impl MandateLearningNote {
    pub fn new(
        mandate_id: &str,
        mandate_version: i64,
        decision_cycle_id: &str,
        summary: &str,
        evidence_receipt_ids: Vec<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            mandate_id: mandate_id.to_string(),
            mandate_version,
            learned_in_decision_cycle_id: decision_cycle_id.to_string(),
            summary: summary.trim().to_string(),
            evidence_receipt_ids,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !canonical_text_within(&self.id, 256)
            || !canonical_text_within(&self.mandate_id, 256)
            || self.mandate_version <= 0
            || !canonical_text_within(&self.learned_in_decision_cycle_id, 256)
            || !canonical_text_within(&self.summary, MAX_MANDATE_LEARNING_NOTE_TEXT)
            || self.evidence_receipt_ids.is_empty()
            || self.evidence_receipt_ids.len() > MAX_MANDATE_EVIDENCE_REFS
            || self
                .evidence_receipt_ids
                .iter()
                .any(|value| !canonical_text_within(value, 256))
        {
            return Err("invalid bounded mandate learning note".to_string());
        }
        chrono::DateTime::parse_from_rfc3339(&self.created_at)
            .map_err(|_| "mandate learning note created_at must be RFC3339".to_string())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MandateReconciliationResolution {
    ConfirmedEffectOccurred,
    ConfirmedNoEffect,
    AbandonAttempt,
}

impl MandateReconciliationResolution {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConfirmedEffectOccurred => "confirmed_effect_occurred",
            Self::ConfirmedNoEffect => "confirmed_no_effect",
            Self::AbandonAttempt => "abandon_attempt",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "confirmed_effect_occurred" => Some(Self::ConfirmedEffectOccurred),
            "confirmed_no_effect" => Some(Self::ConfirmedNoEffect),
            "abandon_attempt" => Some(Self::AbandonAttempt),
            _ => None,
        }
    }
}

/// Agent-selected, revocable commitment made under an owner mandate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum IntentionStatus {
    Committed,
    Satisfied,
    Failed,
    Suspended,
    Abandoned,
}

impl IntentionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Committed => "committed",
            Self::Satisfied => "satisfied",
            Self::Failed => "failed",
            Self::Suspended => "suspended",
            Self::Abandoned => "abandoned",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "committed" => Some(Self::Committed),
            "satisfied" => Some(Self::Satisfied),
            "failed" => Some(Self::Failed),
            "suspended" => Some(Self::Suspended),
            "abandoned" => Some(Self::Abandoned),
            _ => None,
        }
    }

    /// An intention belongs to one decision cycle. Once its commitment ends,
    /// a later cycle must create a fresh intention rather than resurrecting it.
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Committed,
                Self::Satisfied | Self::Failed | Self::Suspended | Self::Abandoned
            )
        )
    }
}

impl std::fmt::Display for IntentionStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Intention {
    pub id: String,
    pub mandate_id: String,
    pub decision_cycle_id: String,
    pub goal_run_id: String,
    pub description: String,
    pub rationale: String,
    pub expected_benefit: Option<String>,
    pub risk: Option<String>,
    pub invalidation_criteria: Option<String>,
    pub status: IntentionStatus,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
}

impl Intention {
    pub fn new(
        mandate_id: &str,
        decision_cycle_id: &str,
        goal_run_id: &str,
        description: &str,
        rationale: &str,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            mandate_id: mandate_id.to_string(),
            decision_cycle_id: decision_cycle_id.to_string(),
            goal_run_id: goal_run_id.to_string(),
            description: description.trim().to_string(),
            rationale: rationale.trim().to_string(),
            expected_benefit: None,
            risk: None,
            invalidation_criteria: None,
            status: IntentionStatus::Committed,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
        }
    }

    pub fn validate_content_bounds(&self) -> Result<(), String> {
        if !canonical_text_within(&self.description, MAX_INTENTION_DESCRIPTION_TEXT) {
            return Err(
                "intention description must be canonical non-empty text of at most 1 KiB in both characters and bytes"
                    .to_string(),
            );
        }
        if !canonical_text_within(&self.rationale, MAX_DECISION_RATIONALE_TEXT) {
            return Err(
                "intention rationale must be canonical non-empty text of at most 2 KiB in both characters and bytes"
                    .to_string(),
            );
        }
        let mut metadata_chars = 0usize;
        let mut metadata_bytes = 0usize;
        for value in [
            self.expected_benefit.as_deref(),
            self.risk.as_deref(),
            self.invalidation_criteria.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if value.is_empty() || value.trim() != value {
                return Err("intention metadata must be canonical non-empty text".to_string());
            }
            metadata_chars = metadata_chars.saturating_add(value.chars().count());
            metadata_bytes = metadata_bytes.saturating_add(value.len());
        }
        if metadata_chars > MAX_INTENTION_METADATA_TEXT
            || metadata_bytes > MAX_INTENTION_METADATA_TEXT
        {
            return Err(
                "intention metadata exceeds its combined 4 KiB character/byte bound".to_string(),
            );
        }
        Ok(())
    }
}

/// Action-bound grant issued by the deterministic mandate gate. It is carried
/// only through Rust control-plane structs, never model-visible arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateAuthorityGrant {
    pub mandate_id: String,
    pub mandate_version: i64,
    pub decision_cycle_id: String,
    pub action_digest: String,
    pub counts_toward_cycle_budget: bool,
    /// One-based fenced reservation number for a mutation; zero for a legacy
    /// or non-mutating grant. The state-store CAS must reserve this exact slot.
    #[serde(default)]
    pub reserved_action_attempt: i64,
    /// Dispatcher-owned tool call identity, bound after deterministic policy
    /// authorization and before the atomic ledger reservation. Never supplied
    /// by or exposed to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Content-safe target identifier persisted for one governed mutation. Raw
/// arguments, request bodies, query strings, responses, and credentials never
/// enter the mandate mutation ledger.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct MandateMutationTarget {
    /// `url` or `resource_id` in v1.
    pub kind: String,
    /// Canonical audit identifier. URL query/fragment/userinfo are removed;
    /// resource identifiers outside the small account/profile vocabulary are
    /// represented by a digest.
    pub identifier: String,
}

/// Rust-only request to reserve one exact governed mutation. All authority
/// fields derive from a grant already issued by the deterministic policy gate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateMutationReservation {
    pub grant: MandateAuthorityGrant,
    pub goal_run_id: String,
    pub root_task_id: String,
    pub root_task_attempt_id: String,
    pub task_id: String,
    pub task_attempt_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub mutation_effects: Vec<String>,
    pub targets: Vec<MandateMutationTarget>,
    pub account_identifiers: Vec<String>,
    /// Caller-supplied clock solely for deterministic state tests. Production
    /// callers always use `Utc::now()` and the store validates RFC3339.
    pub reserved_at: String,
}

/// Rust-only exact one-use claim made at the last common dispatcher before
/// adapter I/O. The store atomically binds this claim to the still-reserved
/// ledger row and every live task/run/authority fence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateMutationDispatchClaim {
    pub grant: MandateAuthorityGrant,
    pub goal_run_id: String,
    pub root_task_id: String,
    pub root_task_attempt_id: String,
    pub task_id: String,
    pub task_attempt_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub claimed_at: String,
}

/// Durable state of a mutation attempt. `reserved` is intentionally not a
/// success: a crash before strict receipt projection remains ambiguous.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateMutationAttemptStatus {
    Reserved,
    /// The action slot was reserved, but the final dispatcher never claimed it
    /// before the run or authority epoch was invalidated. This is terminal and
    /// explicitly distinct from an externally ambiguous dispatch.
    NeverDispatched,
    Succeeded,
    Failed,
    Ambiguous,
}

impl MandateMutationAttemptStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reserved => "reserved",
            Self::NeverDispatched => "never_dispatched",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Ambiguous => "ambiguous",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "reserved" => Some(Self::Reserved),
            "never_dispatched" => Some(Self::NeverDispatched),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            "ambiguous" => Some(Self::Ambiguous),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateMutationEvidence {
    ToolReported,
    StructuredMetadata,
}

impl MandateMutationEvidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolReported => "tool_reported",
            Self::StructuredMetadata => "structured_metadata",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tool_reported" => Some(Self::ToolReported),
            "structured_metadata" => Some(Self::StructuredMetadata),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateMutationAttempt {
    pub id: String,
    pub mandate_id: String,
    pub mandate_version: i64,
    pub decision_cycle_id: String,
    pub goal_run_id: String,
    pub intention_id: String,
    pub root_task_id: String,
    pub root_task_attempt_id: String,
    pub task_id: String,
    pub task_attempt_id: String,
    pub reserved_action_attempt: i64,
    pub action_digest: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub mutation_effects: Vec<String>,
    pub targets: Vec<MandateMutationTarget>,
    pub account_identifiers: Vec<String>,
    pub status: MandateMutationAttemptStatus,
    pub outcome_evidence: Option<MandateMutationEvidence>,
    pub http_status: Option<u16>,
    pub exit_code: Option<i32>,
    pub reserved_at: String,
    pub completed_at: Option<String>,
}

/// Strict, content-free receipt projection emitted by the common dispatcher
/// after the durable ToolResult event append succeeds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateMutationOutcomeProjection {
    pub grant: MandateAuthorityGrant,
    pub goal_run_id: String,
    pub task_id: String,
    pub task_attempt_id: String,
    pub tool_call_id: String,
    pub status: MandateMutationAttemptStatus,
    pub receipt_schema_version: u16,
    pub outcome_evidence: Option<MandateMutationEvidence>,
    pub timed_out: bool,
    pub background_started: bool,
    pub detached: bool,
    pub completion_notifications_enabled: bool,
    pub transport_error_present: bool,
    pub semantics_match: bool,
    pub http_status: Option<u16>,
    pub exit_code: Option<i32>,
    pub completed_at: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateMutationQuotaBlockReason {
    MutationDisabled,
    Rolling24hExhausted,
    Cooldown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateMutationQuotaState {
    pub mandate_id: String,
    pub as_of: String,
    pub max_mutating_actions_per_rolling_24h: u32,
    pub min_seconds_between_mutations: u32,
    pub reserved_in_rolling_24h: u32,
    pub remaining_in_rolling_24h: u32,
    pub last_reserved_at: Option<String>,
    pub available_now: bool,
    pub block_reason: Option<MandateMutationQuotaBlockReason>,
    /// Earliest instant at which both the rolling quota and cooldown can admit
    /// another reservation. None means mutation is disabled or available now.
    pub earliest_next_slot_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateRunFinalizationRequest {
    pub mandate_id: String,
    pub expected_mandate_version: i64,
    pub goal_run_id: String,
    pub finalized_at: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateRunProofCounts {
    pub non_root_tasks: u32,
    pub completed_tasks: u32,
    pub incomplete_tasks: u32,
    pub failed_or_blocked_tasks: u32,
    pub mutation_reservations: u32,
    pub succeeded_mutations: u32,
    pub failed_mutations: u32,
    pub never_dispatched_mutations: u32,
    pub ambiguous_or_reserved_mutations: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateReconciliationReason {
    RootTaskNotSuccessful,
    ActMissingIntention,
    ActMissingWorkTask,
    WorkTasksIncomplete,
    ActMissingVerifiedMutation,
    ActionLedgerMismatch,
    MutationOutcomeFailed,
    MutationOutcomeAmbiguous,
    NonActionCreatedWork,
    NonActionReservedMutation,
}

impl MandateReconciliationReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RootTaskNotSuccessful => "root_task_not_successful",
            Self::ActMissingIntention => "act_missing_intention",
            Self::ActMissingWorkTask => "act_missing_work_task",
            Self::WorkTasksIncomplete => "work_tasks_incomplete",
            Self::ActMissingVerifiedMutation => "act_missing_verified_mutation",
            Self::ActionLedgerMismatch => "action_ledger_mismatch",
            Self::MutationOutcomeFailed => "mutation_outcome_failed",
            Self::MutationOutcomeAmbiguous => "mutation_outcome_ambiguous",
            Self::NonActionCreatedWork => "non_action_created_work",
            Self::NonActionReservedMutation => "non_action_reserved_mutation",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateFinalizationStaleReason {
    MandateMissingOrVersionChanged,
    RunNotCurrent,
    DecisionVersionChanged,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MandateFinalizationRejectReason {
    InvalidRequest,
    DecisionMissing,
    InvalidDecisionState,
}

impl MandateFinalizationRejectReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::DecisionMissing => "decision_missing",
            Self::InvalidDecisionState => "invalid_decision_state",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum MandateRunFinalizationResult {
    ActSatisfied {
        counts: MandateRunProofCounts,
    },
    NonActionSatisfied {
        outcome: MandateDecisionOutcome,
        counts: MandateRunProofCounts,
    },
    ReconciliationRequired {
        reason: MandateReconciliationReason,
        counts: MandateRunProofCounts,
    },
    Stale {
        reason: MandateFinalizationStaleReason,
    },
    Rejected {
        reason: MandateFinalizationRejectReason,
    },
}

/// Typed, content-free owner notice committed with a mandate run's terminal
/// proof state. No model-authored rationale, question, task, tool, response, or
/// error text can enter this structure or its rendered notification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MandateRunNotificationKind {
    ActSatisfied,
    Ask,
    Stopped,
    ReconciliationRequired {
        reason: MandateReconciliationReason,
    },
    ReviewFailed {
        reason: MandateFinalizationRejectReason,
    },
    ExecutionLeaseLost,
    AuthorityRevokedWithUnresolvedMutation,
}

impl MandateRunNotificationKind {
    pub fn notification_type(&self) -> &'static str {
        match self {
            Self::ActSatisfied => "mandate_action",
            Self::Ask => "mandate_ask",
            Self::Stopped => "mandate_stopped",
            Self::ReconciliationRequired { .. }
            | Self::ExecutionLeaseLost
            | Self::AuthorityRevokedWithUnresolvedMutation => "mandate_reconciliation_required",
            Self::ReviewFailed { .. } => "mandate_review_failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MandateRunNotification {
    pub mandate_id: String,
    pub mandate_version: i64,
    pub goal_id: String,
    pub goal_run_id: String,
    pub owner_session_id: String,
    pub kind: MandateRunNotificationKind,
    pub counts: MandateRunProofCounts,
    pub created_at: String,
}

impl MandateRunNotification {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mandate_id: &str,
        mandate_version: i64,
        goal_id: &str,
        goal_run_id: &str,
        owner_session_id: &str,
        kind: MandateRunNotificationKind,
        counts: MandateRunProofCounts,
        created_at: &str,
    ) -> Self {
        Self {
            mandate_id: mandate_id.to_string(),
            mandate_version,
            goal_id: goal_id.to_string(),
            goal_run_id: goal_run_id.to_string(),
            owner_session_id: owner_session_id.to_string(),
            kind,
            counts,
            created_at: created_at.to_string(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for (label, value) in [
            ("mandate id", self.mandate_id.as_str()),
            ("goal id", self.goal_id.as_str()),
            ("goal run id", self.goal_run_id.as_str()),
            ("owner session id", self.owner_session_id.as_str()),
        ] {
            if value.is_empty()
                || value.trim() != value
                || value.chars().count() > 256
                || value.len() > 256
                || value.chars().any(char::is_control)
            {
                return Err(format!("invalid mandate notification {label}"));
            }
        }
        if self.mandate_version <= 0 {
            return Err("invalid mandate notification version".to_string());
        }
        chrono::DateTime::parse_from_rfc3339(&self.created_at)
            .map_err(|_| "invalid mandate notification timestamp".to_string())?;
        Ok(())
    }

    /// Stable primary key: exactly one terminal owner notice can exist for a
    /// mandate run, even if finalization is retried after a crash.
    pub fn notification_id(&self) -> String {
        format!("mandate-run-notice:{}", self.goal_run_id)
    }

    pub fn message(&self) -> String {
        let mandate_ref = self.mandate_id.chars().take(8).collect::<String>();
        let run_ref = self.goal_run_id.chars().take(8).collect::<String>();
        let inspect = format!(
            "Inspect mandate {mandate_ref} with manage_mandates(action=\"get\", mandate_id=\"{mandate_ref}\")."
        );
        match self.kind {
            MandateRunNotificationKind::ActSatisfied => format!(
                "Mandate {mandate_ref} completed bounded action review {run_ref} under policy version {}; work_tasks={}; verified_mutations={}; mutation_reservations={}. The mandate remains active. {inspect}",
                self.mandate_version,
                self.counts.non_root_tasks,
                self.counts.succeeded_mutations,
                self.counts.mutation_reservations,
            ),
            MandateRunNotificationKind::Ask => format!(
                "Mandate {mandate_ref} is awaiting owner input after review {run_ref} under policy version {}. Its generated question is stored as untrusted mandate-local data and is intentionally not copied into assistant history. {inspect}",
                self.mandate_version,
            ),
            MandateRunNotificationKind::Stopped => format!(
                "Mandate {mandate_ref} stopped after review {run_ref} under policy version {}. Its generated rationale remains untrusted mandate-local data and is intentionally not copied into assistant history. {inspect}",
                self.mandate_version,
            ),
            MandateRunNotificationKind::ReconciliationRequired { reason } => format!(
                "Mandate {mandate_ref} paused for owner reconciliation after review {run_ref}; reason={}; work_tasks={}; mutation_reservations={}; verified_mutations={}; failed_mutations={}; never_dispatched_mutations={}; unresolved_mutations={}. No generated task, tool, error, question, rationale, or external-response text is included. {inspect}",
                reason.as_str(),
                self.counts.non_root_tasks,
                self.counts.mutation_reservations,
                self.counts.succeeded_mutations,
                self.counts.failed_mutations,
                self.counts.never_dispatched_mutations,
                self.counts.ambiguous_or_reserved_mutations,
            ),
            MandateRunNotificationKind::ReviewFailed { reason } => format!(
                "Mandate {mandate_ref} could not verify review {run_ref}; reason={}. No generated task, tool, error, question, rationale, or external-response text is included. {inspect}",
                reason.as_str(),
            ),
            MandateRunNotificationKind::ExecutionLeaseLost => format!(
                "Mandate {mandate_ref} review {run_ref} lost its execution lease before its effects could be reconciled. The mandate is paused for safety. Inspect the external target for a partial or duplicate action before resuming. No generated task, tool, error, question, rationale, or external-response text is included. {inspect}"
            ),
            MandateRunNotificationKind::AuthorityRevokedWithUnresolvedMutation => format!(
                "Mandate {mandate_ref} review {run_ref} was invalidated after one or more external mutations crossed the final dispatch boundary without a durable outcome. Inspect the external target for a partial or duplicate action. No generated task, tool, error, question, rationale, or external-response text is included. {inspect}"
            ),
        }
    }

    pub fn to_notification_entry(&self) -> super::goals::NotificationEntry {
        let id = self.notification_id();
        super::goals::NotificationEntry {
            id: id.clone(),
            goal_id: self.goal_id.clone(),
            session_id: self.owner_session_id.clone(),
            notification_type: self.kind.notification_type().to_string(),
            priority: "critical".to_string(),
            message: self.message(),
            created_at: self.created_at.clone(),
            delivered_at: None,
            attempts: 0,
            expires_at: None,
            task_id: None,
            action_token: Some(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mandate_status_legal_transition_matrix_is_explicit() {
        let statuses = [
            MandateStatus::Active,
            MandateStatus::Paused,
            MandateStatus::AwaitingInput,
            MandateStatus::Completed,
            MandateStatus::Cancelled,
        ];
        let legal = [
            (MandateStatus::Active, MandateStatus::Paused),
            (MandateStatus::Active, MandateStatus::AwaitingInput),
            (MandateStatus::Active, MandateStatus::Completed),
            (MandateStatus::Active, MandateStatus::Cancelled),
            (MandateStatus::Paused, MandateStatus::Active),
            (MandateStatus::Paused, MandateStatus::Cancelled),
            (MandateStatus::AwaitingInput, MandateStatus::Active),
            (MandateStatus::AwaitingInput, MandateStatus::Cancelled),
        ];

        for from in statuses {
            for to in statuses {
                assert_eq!(
                    from.can_transition_to(to),
                    legal.contains(&(from, to)),
                    "unexpected mandate transition {from} -> {to}"
                );
            }
            assert_eq!(
                from.is_terminal(),
                matches!(from, MandateStatus::Completed | MandateStatus::Cancelled)
            );
        }
    }

    #[test]
    fn intention_status_legal_transition_matrix_is_explicit() {
        let statuses = [
            IntentionStatus::Committed,
            IntentionStatus::Satisfied,
            IntentionStatus::Failed,
            IntentionStatus::Suspended,
            IntentionStatus::Abandoned,
        ];
        let legal = [
            (IntentionStatus::Committed, IntentionStatus::Satisfied),
            (IntentionStatus::Committed, IntentionStatus::Failed),
            (IntentionStatus::Committed, IntentionStatus::Suspended),
            (IntentionStatus::Committed, IntentionStatus::Abandoned),
        ];

        for from in statuses {
            for to in statuses {
                assert_eq!(
                    from.can_transition_to(to),
                    legal.contains(&(from, to)),
                    "unexpected intention transition {from} -> {to}"
                );
            }
        }
    }

    #[test]
    fn lifecycle_status_wire_values_remain_stable() {
        for status in [
            MandateStatus::Active,
            MandateStatus::Paused,
            MandateStatus::AwaitingInput,
            MandateStatus::Completed,
            MandateStatus::Cancelled,
        ] {
            let encoded = format!("\"{}\"", status.as_str());
            assert_eq!(serde_json::to_string(&status).unwrap(), encoded);
            assert_eq!(
                serde_json::from_str::<MandateStatus>(&encoded).unwrap(),
                status
            );
            assert_eq!(MandateStatus::parse(status.as_str()), Some(status));
        }
        for status in [
            IntentionStatus::Committed,
            IntentionStatus::Satisfied,
            IntentionStatus::Failed,
            IntentionStatus::Suspended,
            IntentionStatus::Abandoned,
        ] {
            let encoded = format!("\"{}\"", status.as_str());
            assert_eq!(serde_json::to_string(&status).unwrap(), encoded);
            assert_eq!(
                serde_json::from_str::<IntentionStatus>(&encoded).unwrap(),
                status
            );
            assert_eq!(IntentionStatus::parse(status.as_str()), Some(status));
        }
    }

    #[test]
    fn authority_rejects_global_wildcard_and_unknown_effects() {
        let mut authority = MandateAuthority {
            allowed_tools: vec!["*".to_string()],
            max_mutating_actions_per_cycle: 1,
            ..MandateAuthority::default()
        };
        assert!(authority.validate().is_err());

        authority.allowed_tools = vec!["mcp__x__*".to_string()];
        authority.allowed_mutation_effects = vec!["invented_effect".to_string()];
        assert!(authority.validate().is_err());
    }

    #[test]
    fn only_exact_positive_list_data_tools_can_be_delegated() {
        let authority = MandateAuthority {
            allowed_tools: vec!["http_request".to_string(), "web_fetch".to_string()],
            allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
            ..MandateAuthority::default()
        };
        assert!(authority.validate().is_ok());
        assert!(authority.allows_tool("http_request"));
        assert!(!authority.allows_tool("http_request_admin"));
        assert!(!authority.allows_tool("terminal"));

        let mut wildcard = authority.clone();
        wildcard.allowed_tools = vec!["http_*".to_string()];
        assert!(wildcard.validate().is_err());
        let mut private_read = authority;
        private_read.allowed_tools = vec!["search_history".to_string()];
        assert!(private_read.validate().is_err());
    }

    #[test]
    fn observation_data_tools_require_explicit_typed_target_scopes() {
        let mut authority = MandateAuthority {
            allowed_tools: vec!["http_request".to_string()],
            ..MandateAuthority::default()
        };
        assert!(authority.validate().is_err());
        authority.allowed_target_prefixes = vec!["https://api.x.com/2/users/me".to_string()];
        assert!(authority.validate().is_ok());
        authority.allowed_target_prefixes = vec!["/private/history".to_string()];
        assert!(authority.validate().is_err());
        for invalid_resource in ["account:", "auth_profile:", "account:123:"] {
            authority.allowed_target_prefixes = vec![invalid_resource.to_string()];
            assert!(authority.validate().is_err(), "scope={invalid_resource}");
        }
    }

    #[test]
    fn opaque_nested_action_tools_cannot_be_delegated() {
        for pattern in [
            "cli_agent",
            "cli_*",
            "terminal",
            "term*",
            "run_command",
            "run_*",
            "browser",
            "brow*",
            "computer_use",
            "computer_*",
            "health_probe",
            "health_*",
            "scheduled_goal_runs",
            "scheduled_goal_*",
            "read_file",
            "read_*",
            "write_file",
            "write_*",
            "edit_file",
            "edit_*",
            "search_files",
            "search_*",
            "project_inspect",
            "project_*",
            "send_file",
            "send_*",
            "git_info",
            "git_commit",
            "git_*",
            "check_environment",
            "check_*",
            "mcp__x__post",
            "mcp__x__*",
        ] {
            let authority = MandateAuthority {
                allowed_tools: vec![pattern.to_string()],
                ..MandateAuthority::default()
            };
            assert!(authority.validate().is_err(), "pattern {pattern:?}");
        }
    }

    #[test]
    fn mutation_authority_requires_bounded_daily_quota_cooldown_and_non_root_target() {
        let mut authority = MandateAuthority {
            allowed_tools: vec!["http_request".to_string()],
            allowed_mutation_effects: vec!["external_delivery".to_string()],
            max_mutating_actions_per_cycle: 1,
            max_mutating_actions_per_rolling_24h: 1,
            min_seconds_between_mutations: 900,
            ..MandateAuthority::default()
        };
        assert!(
            authority.validate().is_err(),
            "empty targets must fail closed"
        );

        authority.allowed_target_prefixes = vec!["https://api.x.com/".to_string()];
        assert!(
            authority.validate().is_err(),
            "origin-wide scope is too broad"
        );

        authority.allowed_target_prefixes = vec![
            "https://api.x.com/2/tweets".to_string(),
            "auth_profile:Twitter%20Prod%2Faccount".to_string(),
        ];
        assert!(authority.validate().is_ok());

        authority.min_seconds_between_mutations = 899;
        assert!(authority.validate().is_err());
        authority.min_seconds_between_mutations = 900;
        authority.max_mutating_actions_per_rolling_24h = 25;
        assert!(authority.validate().is_err());
        authority.max_mutating_actions_per_rolling_24h = 1;
        authority.max_mutating_actions_per_cycle = 2;
        assert!(authority.validate().is_err());
    }

    #[test]
    fn malformed_expiry_fails_closed() {
        let mut mandate = Mandate::new(
            "goal-id",
            None,
            "Test objective",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.expires_at = Some("not-a-timestamp".to_string());
        assert!(!mandate.is_active());
    }

    #[test]
    fn mandate_content_bounds_reject_utf8_byte_overflow_entry_overflow_and_combined_policy() {
        let mut mandate = Mandate::new(
            "goal-id",
            None,
            "bounded objective",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        assert!(mandate.validate_content_bounds().is_ok());

        mandate.objective = "é".repeat(1_025);
        assert!(mandate.validate_content_bounds().is_err());
        mandate.objective = "bounded objective".to_string();

        mandate.constraints = (0..17).map(|index| format!("constraint {index}")).collect();
        assert!(mandate.validate_content_bounds().is_err());

        mandate.constraints = vec!["c".repeat(500); 6];
        mandate.success_criteria = vec!["s".repeat(500); 6];
        mandate.stop_conditions = vec!["x".repeat(500); 6];
        assert!(mandate.validate_content_bounds().is_err());
    }

    #[test]
    fn authority_rejects_aggregate_serialized_size_and_utf8_byte_overflow() {
        let mut authority = MandateAuthority {
            allowed_tools: vec!["é".repeat(129)],
            ..MandateAuthority::default()
        };
        assert!(authority.validate().is_err());

        authority.allowed_tools = vec!["web_fetch".to_string()];
        authority.allowed_target_prefixes = (0..16)
            .map(|index| format!("account:{index}-{}", "x".repeat(1_020)))
            .collect();
        assert!(authority.validate().is_err());
    }

    #[test]
    fn decision_and_intention_bounds_validate_shape_bytes_and_combined_metadata() {
        let mut decision = MandateDecisionCycle::new(
            "mandate-id",
            "run-id",
            MandateDecisionOutcome::Wait,
            "bounded rationale",
            1,
        );
        decision.rationale = "é".repeat(1_025);
        assert!(decision.validate_content_bounds().is_err());
        decision.rationale = "bounded rationale".to_string();
        decision.belief_snapshot = Some(serde_json::json!({"not": "an array"}).to_string());
        assert!(decision.validate_content_bounds().is_err());
        decision.belief_snapshot = Some(serde_json::to_string(&vec!["observation"; 9]).unwrap());
        assert!(decision.validate_content_bounds().is_err());
        decision.belief_snapshot = Some(serde_json::to_string(&vec!["\"".repeat(750); 8]).unwrap());
        assert!(decision.validate_content_bounds().is_err());
        decision.belief_snapshot =
            Some(serde_json::to_string(&vec!["sourced observation"]).unwrap());
        decision.question = Some("é".repeat(251));
        assert!(decision.validate_content_bounds().is_err());

        let mut intention = Intention::new(
            "mandate-id",
            &decision.id,
            "run-id",
            "bounded intention",
            "bounded rationale",
        );
        intention.description = "é".repeat(513);
        assert!(intention.validate_content_bounds().is_err());
        intention.description = "bounded intention".to_string();
        intention.expected_benefit = Some("b".repeat(2_000));
        intention.risk = Some("r".repeat(2_000));
        intention.invalidation_criteria = Some("i".repeat(97));
        assert!(intention.validate_content_bounds().is_err());
    }
}
