use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::types::StatusUpdate;

/// Role assigned to an agent for role-based tool scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Root agent — routes, classifies, full tool access (legacy behavior).
    Orchestrator,
    /// Plans & delegates — management tools only.
    TaskLead,
    /// Executes a single task — action tools + report_blocker.
    Executor,
}

/// Stable specialist profile used to label child-agent sessions and events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialistKind {
    TaskLead,
    Executor,
    Research,
    ArtifactWriter,
    Code,
    BrowserVerifier,
    Review,
    CommsDraft,
    Generic,
}

impl SpecialistKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SpecialistKind::TaskLead => "task_lead",
            SpecialistKind::Executor => "executor",
            SpecialistKind::Research => "research",
            SpecialistKind::ArtifactWriter => "artifact_writer",
            SpecialistKind::Code => "code",
            SpecialistKind::BrowserVerifier => "browser_verifier",
            SpecialistKind::Review => "review",
            SpecialistKind::CommsDraft => "comms_draft",
            SpecialistKind::Generic => "generic",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "task_lead" => Some(SpecialistKind::TaskLead),
            "executor" => Some(SpecialistKind::Executor),
            "research" => Some(SpecialistKind::Research),
            "artifact_writer" => Some(SpecialistKind::ArtifactWriter),
            "code" => Some(SpecialistKind::Code),
            "browser_verifier" => Some(SpecialistKind::BrowserVerifier),
            "review" => Some(SpecialistKind::Review),
            "comms_draft" => Some(SpecialistKind::CommsDraft),
            "generic" => Some(SpecialistKind::Generic),
            _ => None,
        }
    }

    // Test-only oracle: iterates every variant for exhaustiveness checks in
    // `SpecialistRegistry` unit tests. Production code dispatches on
    // `SpecialistKind` values directly, so this helper is unused there.
    #[allow(dead_code)]
    pub fn all() -> &'static [SpecialistKind] {
        &[
            SpecialistKind::TaskLead,
            SpecialistKind::Executor,
            SpecialistKind::Research,
            SpecialistKind::ArtifactWriter,
            SpecialistKind::Code,
            SpecialistKind::BrowserVerifier,
            SpecialistKind::Review,
            SpecialistKind::CommsDraft,
            SpecialistKind::Generic,
        ]
    }
}

/// Categorization of a tool for role-based scoping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRole {
    /// Terminal, web_search, web_fetch, browser, etc.
    Action,
    /// ManageGoalTasksTool, ReportBlockerTool — task lead tools.
    Management,
    /// SystemInfoTool, RememberFactTool — available to all roles.
    Universal,
}

/// Safety and execution metadata for policy-driven tool selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub read_only: bool,
    pub external_side_effect: bool,
    pub needs_approval: bool,
    pub idempotent: bool,
    pub high_impact_write: bool,
}

/// Effect classification for a specific tool call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolCallEffect {
    #[default]
    Unknown,
    Administrative,
    Observation,
    Mutation,
    ObservationAndMutation,
}

/// Typed mutation effects for one tool call.
///
/// `ToolCallEffect` deliberately remains the compact observation/mutation
/// summary used by older persisted events. This set carries the detail needed
/// by policy and completion accounting so an incidental build-cache write does
/// not satisfy a requested source edit, and a local write does not satisfy a
/// requested remote deployment.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(transparent)]
pub struct ToolMutationEffects(u32);

impl ToolMutationEffects {
    pub const PROTOCOL_NAMES: [&str; 11] = [
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
    pub const NONE: Self = Self(0);
    /// A write to user-authored files through a path-aware editing tool.
    pub const LOCAL_SOURCE_WRITE: Self = Self(1 << 0);
    /// A local write whose target is not typed precisely (shell redirect,
    /// copy, generated file, and similar workspace activity).
    pub const LOCAL_WORKSPACE_WRITE: Self = Self(1 << 1);
    /// Compiler caches, dependency directories, and other reproducible output.
    pub const LOCAL_DERIVED_WRITE: Self = Self(1 << 2);
    /// Git refs, index state, commits, and related repository metadata.
    pub const REPOSITORY_WRITE: Self = Self(1 << 3);
    /// A remote state change that is not specifically a deployment.
    pub const REMOTE_MUTATION: Self = Self(1 << 4);
    /// Publishing a new application/site deployment or version.
    pub const REMOTE_DEPLOY: Self = Self(1 << 5);
    /// Sending, posting, uploading, or otherwise delivering content.
    pub const EXTERNAL_DELIVERY: Self = Self(1 << 6);
    /// Starting, stopping, restarting, or killing a process/service.
    pub const PROCESS_STATE: Self = Self(1 << 7);
    /// Mutating configuration, permissions, packages, or runtime setup.
    pub const CONFIGURATION: Self = Self(1 << 8);
    /// Deletion or another intentionally destructive local/remote operation.
    pub const DESTRUCTIVE: Self = Self(1 << 9);
    /// A command can mutate, but static inspection cannot state how. This is
    /// intentionally not allowed to satisfy a more specific required outcome.
    pub const UNSPECIFIED: Self = Self(1 << 10);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Parse a protocol enum value. These names are structured tool metadata,
    /// not natural-language routing keywords.
    pub fn from_protocol_name(name: &str) -> Option<Self> {
        match name {
            "local_source_write" => Some(Self::LOCAL_SOURCE_WRITE),
            "local_workspace_write" => Some(Self::LOCAL_WORKSPACE_WRITE),
            "local_derived_write" => Some(Self::LOCAL_DERIVED_WRITE),
            "repository_write" => Some(Self::REPOSITORY_WRITE),
            "remote_mutation" => Some(Self::REMOTE_MUTATION),
            "remote_deploy" => Some(Self::REMOTE_DEPLOY),
            "external_delivery" => Some(Self::EXTERNAL_DELIVERY),
            "process_state" => Some(Self::PROCESS_STATE),
            "configuration" => Some(Self::CONFIGURATION),
            "destructive" => Some(Self::DESTRUCTIVE),
            "unspecified" => Some(Self::UNSPECIFIED),
            _ => None,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub const fn has_specific_effects(self) -> bool {
        (self.0 & !Self::UNSPECIFIED.0) != 0
    }

    /// Whether observed effects fulfill every typed requirement.
    ///
    /// Known effects are checked strictly: cache output, repository metadata,
    /// and directory scaffolding cannot masquerade as authored source. Opaque
    /// execution surfaces report `UNSPECIFIED`, but unknown evidence is never
    /// allowed to prove a more specific effect. A caller that requires a source
    /// edit, deployment, or delivery must observe that exact typed effect.
    pub const fn satisfies(self, required: Self) -> bool {
        if required.is_empty() {
            return true;
        }
        let generic_required = required.intersects(Self::UNSPECIFIED);
        let generic_satisfied = !generic_required || !self.is_empty();
        let specific_required = Self(required.0 & !Self::UNSPECIFIED.0);
        generic_satisfied && self.contains(specific_required)
    }
}

impl ToolCallEffect {
    pub fn observes_state(self) -> bool {
        matches!(self, Self::Observation | Self::ObservationAndMutation)
    }

    pub fn mutates_state(self) -> bool {
        matches!(self, Self::Mutation | Self::ObservationAndMutation)
    }
}

/// How a tool call can contribute verification evidence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ToolVerificationMode {
    #[default]
    None,
    ResultContent,
}

/// Typed target hint emitted by a tool call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolTargetHintKind {
    Url,
    Path,
    ProjectScope,
    /// Stable opaque identity from the session resource registry. Unlike a
    /// filename or prose reference this is exact, session-scoped, and safe to
    /// compare without natural-language interpretation.
    ResourceId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolTargetHint {
    pub kind: ToolTargetHintKind,
    pub value: String,
}

impl ToolTargetHint {
    pub fn new(kind: ToolTargetHintKind, value: impl Into<String>) -> Option<Self> {
        let value = value.into().trim().to_string();
        if value.is_empty() {
            None
        } else {
            Some(Self { kind, value })
        }
    }
}

/// High-level domain a tool can inspect or mutate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolSemanticScope {
    GoalState,
    UserMemory,
    ConversationHistory,
    ExternalRemote,
    LocalWorkspace,
    HostLocal,
}

impl ToolSemanticScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GoalState => "goal_state",
            Self::UserMemory => "user_memory",
            Self::ConversationHistory => "conversation_history",
            Self::ExternalRemote => "external_remote",
            Self::LocalWorkspace => "local_workspace",
            Self::HostLocal => "host_local",
        }
    }
}

/// The kind of claim an observation can support. These are evidence roles,
/// not request-language keywords or tool names.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EvidencePurpose {
    CurrentState,
    HistoricalRecord,
    Content,
    Outcome,
    Attribution,
    CausalExplanation,
}

impl EvidencePurpose {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CurrentState => "current_state",
            Self::HistoricalRecord => "historical_record",
            Self::Content => "content",
            Self::Outcome => "outcome",
            Self::Attribution => "attribution",
            Self::CausalExplanation => "causal_explanation",
        }
    }
}

/// Strength of the source behind an observation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceAuthority {
    /// A compressed memory, model-derived summary, or other discovery lead.
    Advisory,
    /// A direct observation of the resource being discussed.
    Direct,
    /// The daemon's authoritative ledger for the historical fact in question.
    Canonical,
}

impl EvidenceAuthority {
    const fn rank(self) -> u8 {
        match self {
            Self::Advisory => 0,
            Self::Direct => 1,
            Self::Canonical => 2,
        }
    }

    pub const fn satisfies(self, required: Self) -> bool {
        self.rank() >= required.rank()
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Advisory => "advisory",
            Self::Direct => "direct",
            Self::Canonical => "canonical",
        }
    }
}

/// Time coverage of an observation surface.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTemporalScope {
    Current,
    Historical,
    Both,
}

impl EvidenceTemporalScope {
    pub const fn satisfies(self, required: Self) -> bool {
        matches!(required, Self::Both) && matches!(self, Self::Both)
            || matches!(required, Self::Current) && matches!(self, Self::Current | Self::Both)
            || matches!(required, Self::Historical) && matches!(self, Self::Historical | Self::Both)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Historical => "historical",
            Self::Both => "both",
        }
    }
}

/// Claim-support metadata for one successful tool result. Unlike
/// [`ToolCapabilities`], this describes what information the result can prove,
/// not whether executing the tool is safe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolEvidenceCapability {
    pub scope: ToolSemanticScope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub purposes: Vec<EvidencePurpose>,
    pub authority: EvidenceAuthority,
    pub temporal_scope: EvidenceTemporalScope,
}

impl ToolEvidenceCapability {
    pub fn new(
        scope: ToolSemanticScope,
        purposes: &[EvidencePurpose],
        authority: EvidenceAuthority,
        temporal_scope: EvidenceTemporalScope,
    ) -> Self {
        Self {
            scope,
            purposes: purposes.to_vec(),
            authority,
            temporal_scope,
        }
    }
}

/// Fine-grained semantic capability advertised by a tool.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolSemanticFacet {
    GoalState,
}

/// Structured semantic affordances for matching tools to contextual requests.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolSemanticAffordances {
    pub scope: ToolSemanticScope,
    #[serde(default)]
    pub facets: Vec<ToolSemanticFacet>,
}

#[allow(dead_code)]
impl ToolSemanticAffordances {
    pub fn new(scope: ToolSemanticScope, facets: &[ToolSemanticFacet]) -> Self {
        Self {
            scope,
            facets: facets.to_vec(),
        }
    }

    pub fn supports(&self, facet: ToolSemanticFacet) -> bool {
        self.facets.contains(&facet)
    }
}

/// Canonical adapter operation used by deterministic operation-scoped policy.
/// Unknown/new HTTP verbs deliberately remain untyped and therefore cannot
/// match a delegated mandate scope.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "UPPERCASE")]
pub enum ToolCallOperation {
    Get,
    Head,
    Options,
    Post,
    Put,
    Patch,
    Delete,
}

impl ToolCallOperation {
    pub const fn from_http_method(method: &str) -> Option<Self> {
        match method.as_bytes() {
            b"GET" => Some(Self::Get),
            b"HEAD" => Some(Self::Head),
            b"OPTIONS" => Some(Self::Options),
            b"POST" => Some(Self::Post),
            b"PUT" => Some(Self::Put),
            b"PATCH" => Some(Self::Patch),
            b"DELETE" => Some(Self::Delete),
            _ => None,
        }
    }
}

/// Structured completion semantics for a specific tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolCallSemantics {
    #[serde(default)]
    pub effect: ToolCallEffect,
    #[serde(default)]
    pub verification_mode: ToolVerificationMode,
    /// Fine-grained mutation effects. Older persisted rows omit this field and
    /// continue to deserialize through the default empty set.
    #[serde(default, skip_serializing_if = "ToolMutationEffects::is_empty")]
    pub mutation_effects: ToolMutationEffects,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_hints: Vec<ToolTargetHint>,
    /// Adapter-defined canonical operation for policy matching. This is an
    /// exact, typed identifier (for example `GET` or `POST`), not a natural-
    /// language action inferred by the policy layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<ToolCallOperation>,
    /// Typed claim-support affordances for this exact call/result. These are
    /// persisted in the durable receipt so recovery and completion do not have
    /// to reconstruct evidence from prose.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<ToolEvidenceCapability>,
}

/// Prepared filesystem access requested by one exact tool call.
///
/// Execution location, readable context, and mutation authority are different
/// roles. Keeping them separate prevents a read-only working directory from
/// becoming an implied write grant and gives adapters one canonical contract
/// to enforce and persist.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallAccessManifest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_cwd: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub read_targets: Vec<ToolTargetHint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub write_targets: Vec<ToolTargetHint>,
}

impl ToolCallSemantics {
    pub fn administrative() -> Self {
        Self {
            effect: ToolCallEffect::Administrative,
            ..Self::default()
        }
    }

    pub fn observation() -> Self {
        Self {
            effect: ToolCallEffect::Observation,
            ..Self::default()
        }
    }

    /// Apply a dispatcher-owned hard observation boundary while retaining
    /// target and evidence metadata from the original call. Use only when the
    /// adapter itself enforces that boundary before I/O.
    pub fn constrained_to_observation(mut self) -> Self {
        self.effect = ToolCallEffect::Observation;
        self.mutation_effects = ToolMutationEffects::NONE;
        self
    }

    pub fn mutation() -> Self {
        Self {
            effect: ToolCallEffect::Mutation,
            mutation_effects: ToolMutationEffects::UNSPECIFIED,
            ..Self::default()
        }
    }

    pub fn observation_and_mutation() -> Self {
        Self {
            effect: ToolCallEffect::ObservationAndMutation,
            mutation_effects: ToolMutationEffects::UNSPECIFIED,
            ..Self::default()
        }
    }

    pub fn mutation_with(mutation_effects: ToolMutationEffects) -> Self {
        Self {
            effect: ToolCallEffect::Mutation,
            mutation_effects: if mutation_effects.is_empty() {
                ToolMutationEffects::UNSPECIFIED
            } else {
                mutation_effects
            },
            ..Self::default()
        }
    }

    pub fn observation_and_mutation_with(mutation_effects: ToolMutationEffects) -> Self {
        Self {
            effect: ToolCallEffect::ObservationAndMutation,
            mutation_effects: if mutation_effects.is_empty() {
                ToolMutationEffects::UNSPECIFIED
            } else {
                mutation_effects
            },
            ..Self::default()
        }
    }

    pub fn with_verification_mode(mut self, verification_mode: ToolVerificationMode) -> Self {
        self.verification_mode = verification_mode;
        self
    }

    pub fn with_target_hint(mut self, kind: ToolTargetHintKind, value: impl Into<String>) -> Self {
        if let Some(target) = ToolTargetHint::new(kind, value) {
            self.target_hints.push(target);
        }
        self
    }

    pub fn with_operation(mut self, operation: ToolCallOperation) -> Self {
        self.operation = Some(operation);
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<ToolEvidenceCapability>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn observes_state(&self) -> bool {
        self.effect.observes_state()
    }

    pub fn mutates_state(&self) -> bool {
        self.effect.mutates_state() || !self.mutation_effects.is_empty()
    }

    pub fn can_verify_with_result_content(&self) -> bool {
        self.verification_mode == ToolVerificationMode::ResultContent
    }

    pub fn is_empty(&self) -> bool {
        self.effect == ToolCallEffect::Unknown
            && self.verification_mode == ToolVerificationMode::None
            && self.mutation_effects.is_empty()
            && self.target_hints.is_empty()
            && self.operation.is_none()
            && self.evidence.is_empty()
    }

    pub fn merge_missing_from(&mut self, fallback: Self) {
        if self.effect == ToolCallEffect::Unknown {
            self.effect = fallback.effect;
        }
        if self.verification_mode == ToolVerificationMode::None {
            self.verification_mode = fallback.verification_mode;
        }
        // An explicit observational effect is authoritative. Filling its empty
        // mutation bitset from a conservative pre-dispatch fallback would turn
        // a tool-enforced read-only execution back into a mutation receipt.
        if self.mutation_effects.is_empty()
            && (self.effect == ToolCallEffect::Unknown || self.effect.mutates_state())
        {
            self.mutation_effects = fallback.mutation_effects;
        }
        if self.target_hints.is_empty() {
            self.target_hints = fallback.target_hints;
        }
        if self.operation.is_none() {
            self.operation = fallback.operation;
        }
        if self.evidence.is_empty() {
            self.evidence = fallback.evidence;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadFileSelectionMetadata {
    Full,
    BoundedRange { start_line: usize, end_line: usize },
    OpenEndedRange { start_line: usize },
    Tail { requested_lines: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadFileResultMetadata {
    pub display_path: String,
    pub canonical_path: String,
    pub selection: ReadFileSelectionMetadata,
    pub returned_start_line: Option<usize>,
    pub returned_end_line: Option<usize>,
    pub total_lines: usize,
    pub file_size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified: Option<String>,
    pub selected_lines: Vec<String>,
    /// True when the read returned fewer lines than requested because the
    /// per-call output cap was hit. The rendered output then carries an
    /// explicit continuation hint.
    #[serde(default)]
    pub truncated: bool,
}

/// Structured record of tool-output truncation. Populated at the tool
/// boundary — the instructional notice text is rendered into the
/// model-visible message by the loop (single site), never embedded in the
/// tool's returned content, so text-scanning consumers (error-line
/// extraction, classifiers, summaries) always see clean output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TruncationInfo {
    pub shown_chars: usize,
    pub total_chars: usize,
    /// Tool-specific remediation sentence (see `utils::truncation_notice_with_hint`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remediation_hint: Option<String>,
}

/// Whether a particular retained view contains the authoritative tool result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultCompleteness {
    Complete,
    Truncated,
    #[default]
    Unavailable,
}

impl ToolResultCompleteness {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Truncated => "truncated",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Origin of the bytes represented by a durable observation receipt.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultContentSource {
    ToolOutput,
    DurableReplay,
    SpillPreview,
    PersistentSummary,
    #[default]
    Unavailable,
}

impl ToolResultContentSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ToolOutput => "tool_output",
            Self::DurableReplay => "durable_replay",
            Self::SpillPreview => "spill_preview",
            Self::PersistentSummary => "persistent_summary",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Content provenance shared by live completion and durable continuation.
/// The digest is computed over the authoritative pre-compression result; the
/// view-specific fields state exactly what the model and event log retained.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ToolResultProvenance {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default)]
    pub source: ToolResultContentSource,
    #[serde(default)]
    pub model_view_completeness: ToolResultCompleteness,
    #[serde(default)]
    pub durable_view_completeness: ToolResultCompleteness,
    #[serde(default)]
    pub authoritative_chars: usize,
    #[serde(default)]
    pub model_visible_chars: usize,
    #[serde(default)]
    pub durable_chars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_range: Option<ReadFileSelectionMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returned_start_line: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub returned_end_line: Option<usize>,
}

/// Deterministic credential/target readiness checked immediately before an
/// autonomous adapter is allowed to perform I/O. The accompanying durable
/// receipt carries the mandate grant, which binds this record to the stable
/// mandate identity and policy version without exposing credentials.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthorizationPreflightStatus {
    Ready,
    Blocked,
    Unverifiable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorizationPreflightRecord {
    pub schema_version: u16,
    pub status: AuthorizationPreflightStatus,
    pub intended_operation: String,
    pub target_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    /// Provider-declared OAuth scopes known at the readiness boundary. Static
    /// non-OAuth profiles legitimately leave this empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authorized_scopes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_generation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_result: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub checked_at: String,
}

impl AuthorizationPreflightRecord {
    pub const SCHEMA_VERSION: u16 = 1;

    pub const fn permits_io(&self) -> bool {
        matches!(self.status, AuthorizationPreflightStatus::Ready)
    }
}

impl ToolResultProvenance {
    pub fn from_authoritative_result(
        result: &str,
        metadata: &ToolCallMetadata,
        source: ToolResultContentSource,
    ) -> Self {
        use sha2::{Digest, Sha256};

        let sha256 = format!("{:x}", Sha256::digest(result.as_bytes()));
        let chars = result.chars().count();
        let tool_truncated = metadata.truncation.is_some()
            || metadata
                .read_file
                .as_ref()
                .is_some_and(|read| read.truncated);
        let completeness = if tool_truncated {
            ToolResultCompleteness::Truncated
        } else {
            ToolResultCompleteness::Complete
        };
        let read = metadata.read_file.as_ref();
        Self {
            result_id: Some(format!("sha256:{sha256}")),
            sha256: Some(sha256),
            source,
            model_view_completeness: completeness,
            durable_view_completeness: completeness,
            authoritative_chars: chars,
            model_visible_chars: chars,
            durable_chars: chars,
            requested_range: read.map(|read| read.selection.clone()),
            returned_start_line: read.and_then(|read| read.returned_start_line),
            returned_end_line: read.and_then(|read| read.returned_end_line),
        }
    }

    pub fn mark_model_view_truncated(&mut self, visible_result: &str) {
        self.model_visible_chars = visible_result.chars().count();
        self.model_view_completeness = ToolResultCompleteness::Truncated;
    }

    pub fn record_durable_view(&mut self, durable_result: &str, is_summary: bool) {
        self.durable_chars = durable_result.chars().count();
        if is_summary {
            self.durable_view_completeness = ToolResultCompleteness::Truncated;
            self.source = ToolResultContentSource::PersistentSummary;
        } else {
            self.durable_view_completeness = self.model_view_completeness;
        }
    }
}

/// Structured outcome of a tool invocation. This separates transport/execution
/// success from the domain result: a test runner can execute correctly while
/// reporting failing tests, and a lookup can complete with no matches.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcomeStatus {
    Succeeded,
    CompletedWithNegativeResult,
    FailedRetryable,
    FailedPermanent,
    Blocked,
    Backgrounded,
}

/// Shape of the machine receipt produced by a tool invocation.
///
/// Completion predicates are compiled against this protocol type instead of
/// assuming every tool is a subprocess. In particular, an absent exit code on
/// a management/API receipt means "not applicable", not "the invocation did
/// not happen".
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolReceiptKind {
    #[default]
    Generic,
    Process,
    Http,
}

impl ToolOutcomeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::CompletedWithNegativeResult => "completed_with_negative_result",
            Self::FailedRetryable => "failed_retryable",
            Self::FailedPermanent => "failed_permanent",
            Self::Blocked => "blocked",
            Self::Backgrounded => "backgrounded",
        }
    }

    pub fn satisfies_requested_condition(self) -> bool {
        matches!(self, Self::Succeeded)
    }

    pub fn is_failure(self) -> bool {
        matches!(
            self,
            Self::FailedRetryable | Self::FailedPermanent | Self::Blocked
        )
    }
}

/// How the response layer should treat a successful tool result.
///
/// This is control-plane metadata, not a canned user response. Tools use it to
/// distinguish an owner-facing operational outcome from an explicitly
/// requested diagnostic view while leaving the wording to the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultPresentation {
    NaturalSummary,
    DiagnosticDetail,
}

/// Structured execution metadata returned by tools.
///
/// This is intentionally minimal and backward-compatible: tools can continue
/// returning plain text while selectively populating structured fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallMetadata {
    /// Protocol family for fields whose applicability differs by adapter.
    /// The dispatcher stamps this from the registered tool before completion
    /// evaluation; tools do not need to repeat the declaration per result.
    #[serde(default)]
    pub receipt_kind: ToolReceiptKind,
    /// Effective filesystem access contract enforced for this invocation.
    /// This is an audit receipt, not a reusable authority grant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_manifest: Option<ToolCallAccessManifest>,
    /// Authoritative domain outcome when the tool can provide one. Older tools
    /// may omit it; the loop then derives a conservative status from structured
    /// exit/HTTP/transport metadata before falling back to legacy text parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_status: Option<ToolOutcomeStatus>,
    /// The adapter deterministically rejected the invocation contract before
    /// performing domain I/O. This is an authoritative negative outcome, not
    /// evidence that the requested read/write itself occurred.
    #[serde(default)]
    pub contract_rejected: bool,
    /// Actual adapter that produced the result when dispatch was routed from
    /// the requested tool through another implementation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_tool_name: Option<String>,
    /// True when the dispatcher reconstructed this outcome from a prior
    /// durable receipt instead of invoking the side effect again.
    #[serde(default)]
    pub receipt_replayed: bool,
    /// Process exit code when applicable (e.g. terminal/run_command style tools).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// True when tool execution exceeded a timeout threshold.
    #[serde(default)]
    pub timed_out: bool,
    /// True when execution moved to background tracking.
    #[serde(default)]
    pub background_started: bool,
    /// True when the process is detached and intentionally long-lived.
    #[serde(default)]
    pub detached: bool,
    /// True when the tool guarantees automatic completion delivery for a
    /// backgrounded operation in the current run.
    #[serde(default)]
    pub completion_notifications_enabled: bool,
    /// Transport/runtime failure outside normal tool semantics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport_error: Option<String>,
    /// HTTP status code for API tools (http_request, web_fetch).
    /// Populated at the tool boundary — not scraped from result text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    /// Output truncation performed at the tool boundary. Rendered into the
    /// model-visible text by the loop after ledger/classifier consumption.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationInfo>,
    /// Strong provenance for the result body. Populated centrally by the tool
    /// dispatcher so individual adapters cannot silently omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_provenance: Option<ToolResultProvenance>,
    /// Exact pre-I/O authorization readiness for autonomous authenticated
    /// adapters. It is persisted with the receipt and never contains secrets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorization_preflight: Option<AuthorizationPreflightRecord>,
    /// Optional final user-facing reply. When set, the root agent may close the
    /// turn directly from the tool result instead of running another LLM pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_response: Option<String>,
    /// Completion semantics for this specific tool call.
    #[serde(default, skip_serializing_if = "ToolCallSemantics::is_empty")]
    pub semantics: ToolCallSemantics,
    /// Complete typed result for successful text-file reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_file: Option<ReadFileResultMetadata>,
    /// Image (or other) files produced by the tool for agent vision context.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<crate::traits::MessageAttachment>,
    /// Preserve exact, untrusted bytes while still wrapping them as untrusted
    /// tool data. Intended for canonical-history evidence whose wording must
    /// not be rewritten by prompt-injection sanitization.
    #[serde(default)]
    pub untrusted_verbatim: bool,
    /// Keep bounded tool output inline and never write it to a temporary spill
    /// file or run lossy generic compression.
    #[serde(default)]
    pub preserve_inline: bool,
    /// Durable audit representation. The full output remains available to the
    /// current model turn, but only this stub is written to canonical events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistent_output: Option<String>,
    /// Do not copy result content into executor task-activity telemetry.
    #[serde(default)]
    pub suppress_activity_result: bool,
    /// Presentation policy for this result. `NaturalSummary` asks the response
    /// layer to communicate the user-level outcome without exposing control-
    /// plane bookkeeping. `DiagnosticDetail` is selected only when the caller
    /// explicitly requested those details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<ToolResultPresentation>,
    /// Exact control-plane identifiers observed by the tool. They remain
    /// available to the agent/runtime but are omitted from a natural summary
    /// unless the owner supplied the same identifier or requested diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub internal_identifiers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallOutcome {
    pub output: String,
    #[serde(default)]
    pub metadata: ToolCallMetadata,
}

impl ToolCallOutcome {
    pub fn from_output(output: String) -> Self {
        Self {
            output,
            metadata: ToolCallMetadata::default(),
        }
    }

    pub fn contract_rejection(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
                contract_rejected: true,
                // No requested domain effect ran, but the adapter directly
                // observed and completed its deterministic validation. This
                // lets an explicit receipt predicate prove the negative result
                // without relabeling it as a successful mutation.
                semantics: ToolCallSemantics::observation()
                    .with_verification_mode(ToolVerificationMode::ResultContent),
                ..ToolCallMetadata::default()
            },
        }
    }

    pub fn blocked(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Blocked),
                ..ToolCallMetadata::default()
            },
        }
    }

    pub fn completed_negative_result(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
                ..ToolCallMetadata::default()
            },
        }
    }

    pub fn with_semantics(mut self, semantics: ToolCallSemantics) -> Self {
        self.metadata.semantics = semantics;
        self
    }
}

impl Default for ToolCapabilities {
    fn default() -> Self {
        Self {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: false,
        }
    }
}

fn string_to_target_hint(key: &str, value: &str) -> Option<ToolTargetHint> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower_key = key.to_ascii_lowercase();
    if lower_key == "resource_id" {
        return ToolTargetHint::new(ToolTargetHintKind::ResourceId, trimmed);
    }

    if matches!(
        lower_key.as_str(),
        "url" | "verify_url" | "callback_url" | "target_url" | "auth_url"
    ) || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
    {
        return ToolTargetHint::new(ToolTargetHintKind::Url, trimmed);
    }

    if matches!(
        lower_key.as_str(),
        "path"
            | "file_path"
            | "working_dir"
            | "cwd"
            | "directory"
            | "dir"
            | "repo_path"
            | "repo_dir"
            | "resource_path"
    ) || trimmed.starts_with('/')
        || trimmed.starts_with("./")
        || trimmed.starts_with("../")
        || trimmed.starts_with("~/")
    {
        return ToolTargetHint::new(ToolTargetHintKind::Path, trimmed);
    }

    if matches!(
        lower_key.as_str(),
        "project_path" | "project_dir" | "scope" | "project_scope"
    ) {
        return ToolTargetHint::new(ToolTargetHintKind::ProjectScope, trimmed);
    }

    None
}

fn push_unique_target_hint(hints: &mut Vec<ToolTargetHint>, candidate: Option<ToolTargetHint>) {
    let Some(candidate) = candidate else {
        return;
    };
    if !hints.iter().any(|existing| existing == &candidate) {
        hints.push(candidate);
    }
}

fn collect_common_target_hints(arguments: &str) -> Vec<ToolTargetHint> {
    let parsed = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => map,
        _ => return Vec::new(),
    };

    let mut hints = Vec::new();
    for (key, value) in &parsed {
        match value {
            Value::String(s) => push_unique_target_hint(&mut hints, string_to_target_hint(key, s)),
            Value::Array(items) if matches!(key.as_str(), "paths" | "urls") => {
                for item in items.iter().filter_map(|item| item.as_str()) {
                    push_unique_target_hint(&mut hints, string_to_target_hint(key, item));
                }
            }
            _ => {}
        }
    }
    hints
}

fn default_access_manifest(
    arguments: &str,
    semantics: &ToolCallSemantics,
) -> ToolCallAccessManifest {
    let parsed = serde_json::from_str::<Value>(arguments).ok();
    let execution_cwd = parsed.as_ref().and_then(|value| {
        value
            .get("working_dir")
            .or_else(|| value.get("cwd"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    });
    let targets = if semantics.target_hints.is_empty() {
        collect_common_target_hints(arguments)
    } else {
        semantics.target_hints.clone()
    };
    let non_cwd_targets = targets
        .into_iter()
        .filter(|target| {
            matches!(
                target.kind,
                ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
            )
        })
        .filter(|target| execution_cwd.as_deref() != Some(target.value.as_str()))
        .collect::<Vec<_>>();
    let mut read_targets = Vec::new();
    if let Some(cwd) = execution_cwd.as_deref() {
        if let Some(target) = ToolTargetHint::new(ToolTargetHintKind::ProjectScope, cwd) {
            read_targets.push(target);
        }
    }
    if semantics.observes_state() {
        read_targets.extend(non_cwd_targets.iter().cloned());
    }
    let write_targets = if semantics.mutates_state() {
        non_cwd_targets
    } else {
        Vec::new()
    };
    ToolCallAccessManifest {
        execution_cwd,
        read_targets,
        write_targets,
    }
}

/// Classify a mixed-operation tool from the exact `action` enum in its schema.
///
/// This helper intentionally performs no tokenization, substring matching, or
/// interpretation of unknown action names. Tools pass their exhaustive set of
/// read-only enum variants; every missing, malformed, or new variant fails
/// conservatively as a mutation until the tool implementation classifies it.
pub fn semantics_for_exact_read_actions(
    arguments: &str,
    read_actions: &[&str],
    mutation_effects: ToolMutationEffects,
) -> ToolCallSemantics {
    let action = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("action")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    let mut semantics = if action
        .as_deref()
        .is_some_and(|action| read_actions.contains(&action))
    {
        ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
    } else if mutation_effects.is_empty() {
        ToolCallSemantics::mutation()
    } else {
        ToolCallSemantics::mutation_with(mutation_effects)
    };
    for target_hint in collect_common_target_hints(arguments) {
        semantics = semantics.with_target_hint(target_hint.kind, target_hint.value);
    }
    semantics
}

fn default_semantics_from_identity(
    _name: &str,
    _description: &str,
    arguments: &str,
    caps: ToolCapabilities,
) -> ToolCallSemantics {
    let mut semantics = if caps.read_only {
        ToolCallSemantics::observation().with_verification_mode(ToolVerificationMode::ResultContent)
    } else {
        // Unknown non-read-only tools fail conservatively as mutations. Mixed
        // tools must override `call_semantics` with an exact match on their
        // schema enum; names, descriptions, and arbitrary action vocabulary are
        // never an authority source.
        ToolCallSemantics::mutation()
    };

    for target_hint in collect_common_target_hints(arguments) {
        semantics = semantics.with_target_hint(target_hint.kind, target_hint.value);
    }
    semantics
}

/// Dispatcher-owned execution context passed to tools via the trait method.
///
/// This is a Rust-side control-plane type — it is NEVER serialized, never appears
/// in model-visible JSON, tool schemas, logs, or persisted tool arguments.
/// Do NOT add fields here that duplicate information already in the enriched args.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToolExecutionContext {
    /// When true, the correction sandbox has already classified this exact tool
    /// call as safe for unattended execution. Tools that honor this (currently
    /// `terminal` and `http_request`) may skip their redundant approval prompt.
    /// Hard blocks, command-safety checks, scope, and all other validations run.
    pub correction_preapproved: bool,
    /// True only after the final common dispatcher revalidated an exact,
    /// action-bound `MandateAuthorityGrant` for this mutating call. This is
    /// deliberately separate from correction preapproval so tools can honor
    /// owner-delegated autonomy without conflating the two authority sources.
    pub mandate_preapproved: bool,
    /// True only when the common dispatcher resolved this call to an active
    /// autonomous mandate. Network adapters use this control-plane bit to keep
    /// one target-bound call from following redirects outside the target that
    /// was evaluated by the mandate authority kernel.
    pub mandate_execution: bool,
    /// Hard negative task contract propagated from the validated request.
    /// Open-ended adapters must prove an operation is observational before I/O.
    pub mutation_forbidden: bool,
}

/// A tool-owned decision about whether a previously successful durable receipt
/// still represents current state.
///
/// Most mutations cannot be repeated safely and therefore use `Replay`. Tools
/// with an exact deterministic postcondition may opt into current-state
/// reconciliation and either re-execute when that postcondition is missing or
/// block when the target drifted in a way that could make repetition unsafe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableReplayDecision {
    Replay,
    Reexecute { reason: String },
    Block { reason: String },
}

/// Tool trait — system tools, terminal, MCP-proxied tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgumentContractViolation {
    pub reason: String,
    pub recovery_hint: Option<String>,
}

impl ToolArgumentContractViolation {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            recovery_hint: None,
        }
    }

    pub fn with_recovery_hint(mut self, hint: impl Into<String>) -> Self {
        self.recovery_hint = Some(hint.into());
        self
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Returns the OpenAI-format function schema as a JSON Value.
    fn schema(&self) -> Value;
    /// Adapter-owned, deterministic argument invariants evaluated by the
    /// common dispatcher before any tool I/O. The default accepts all schema-
    /// valid calls. Rejections become typed `contract_rejected` receipts.
    fn validate_arguments(&self, _arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        Ok(())
    }
    /// Execute the tool with the given JSON arguments string, returns result text.
    async fn call(&self, arguments: &str) -> anyhow::Result<String>;

    /// Execute the tool with access to a status update channel for streaming feedback.
    /// Default implementation just calls `call()` - override for tools that emit progress.
    async fn call_with_status(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<String> {
        // Default: ignore status channel and just call the basic method
        let _ = status_tx;
        self.call(arguments).await
    }

    /// Structured execution path used by the agent loop.
    ///
    /// Default behavior preserves compatibility for existing tools by wrapping
    /// plain text output with empty metadata.
    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output = self.call_with_status(arguments, status_tx).await?;
        Ok(ToolCallOutcome::from_output(output))
    }

    /// Context-aware execution path used by the agent loop for correction-gate calls.
    ///
    /// The default implementation discards `exec_ctx` and delegates to
    /// `call_with_status_outcome`, so existing tools are completely unaffected.
    /// Tools that support correction preapproval (`terminal`, `http_request`) override
    /// this method and inspect `exec_ctx.correction_preapproved` to decide whether to
    /// skip the user-approval prompt.
    ///
    /// **Security invariant:** `exec_ctx` is a Rust-side control-plane value — it must
    /// never be serialized, logged at info level, or reflected back into tool args.
    async fn call_with_execution_context(
        &self,
        arguments: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        exec_ctx: ToolExecutionContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let _ = exec_ctx;
        self.call_with_status_outcome(arguments, status_tx).await
    }

    /// Task lifecycle callback fired after the agent emits `TaskEnd`.
    /// Tools that spawn background activity can use this to clean up
    /// task-scoped resources.
    async fn on_task_end(&self, _task_id: &str, _session_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Categorize this tool for role-based scoping.
    /// Default: Action (most tools are action tools).
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    /// Capability metadata used by the execution policy and risk gate.
    /// Defaults are intentionally conservative.
    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities::default()
    }

    /// Machine-receipt protocol emitted by this adapter.
    ///
    /// Keep this independent from natural-language descriptions and call
    /// semantics: it exists solely to type-check completion predicates.
    fn receipt_kind(&self, _arguments: &str) -> ToolReceiptKind {
        ToolReceiptKind::Generic
    }

    /// Structured completion semantics for a specific call.
    ///
    /// Default behavior derives a conservative fallback from `capabilities()`.
    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        default_semantics_from_identity(
            self.name(),
            self.description(),
            arguments,
            self.capabilities(),
        )
    }

    /// Filesystem access roles for this exact call. Open-ended adapters should
    /// override this when their schema exposes explicit read/write grants.
    fn call_access_manifest(&self, arguments: &str) -> ToolCallAccessManifest {
        let semantics = self.call_semantics(arguments);
        default_access_manifest(arguments, &semantics)
    }

    /// Reconcile a successful durable receipt with current state before replay.
    ///
    /// The conservative default always replays. An override may return
    /// `Reexecute` only when repeating the exact same invocation is safe after
    /// the observed state transition, and `Block` when current state cannot be
    /// reconciled safely.
    async fn durable_replay_decision(&self, _arguments: &str) -> DurableReplayDecision {
        DurableReplayDecision::Replay
    }

    /// Semantic domains this tool can answer or affect.
    #[allow(dead_code)]
    fn semantic_affordances(&self) -> Option<ToolSemanticAffordances> {
        None
    }

    /// Whether this tool is currently operational.
    ///
    /// Default: true. Override for tools with dynamic backends that may be
    /// temporarily unavailable at runtime.
    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct AlwaysAvailableTool;

    #[async_trait]
    impl Tool for AlwaysAvailableTool {
        fn name(&self) -> &str {
            "always_available"
        }

        fn description(&self) -> &str {
            "test"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "always_available",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    struct UnavailableTool;

    #[async_trait]
    impl Tool for UnavailableTool {
        fn name(&self) -> &str {
            "unavailable"
        }

        fn description(&self) -> &str {
            "test"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "unavailable",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }

        fn is_available(&self) -> bool {
            false
        }
    }

    #[test]
    fn default_is_available_returns_true() {
        let tool = AlwaysAvailableTool;
        assert!(tool.is_available());
    }

    #[test]
    fn override_is_available_returns_false() {
        let tool = UnavailableTool;
        assert!(!tool.is_available());
    }

    struct ManageTool;

    #[async_trait]
    impl Tool for ManageTool {
        fn name(&self) -> &str {
            "manage_demo"
        }

        fn description(&self) -> &str {
            "Manage demo entities"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "manage_demo",
                "description": "Manage demo entities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "path": {"type": "string"}
                    },
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    #[test]
    fn default_call_semantics_uses_capabilities_not_action_vocabulary() {
        let tool = ManageTool;
        let list = tool.call_semantics(r#"{"action":"list","path":"/tmp/demo"}"#);
        assert!(!list.observes_state());
        assert!(list.mutates_state());
        assert_eq!(
            list.target_hints,
            vec![ToolTargetHint {
                kind: ToolTargetHintKind::Path,
                value: "/tmp/demo".to_string()
            }]
        );

        let remove = tool.call_semantics(r#"{"action":"remove","path":"/tmp/demo"}"#);
        assert!(remove.mutates_state());
        assert!(!remove.observes_state());
    }

    #[test]
    fn default_filesystem_manifest_excludes_remote_and_resource_targets() {
        let tool = ManageTool;
        let manifest = tool.call_access_manifest(
            r#"{"url":"https://example.test/status","resource_id":"remote-1"}"#,
        );
        assert!(manifest.read_targets.is_empty());
        assert!(manifest.write_targets.is_empty());
    }

    #[test]
    fn default_semantics_does_not_interpret_action_names() {
        let tool = ManageTool;
        for action in ["review", "remove_provider", "run_history", "made_up"] {
            let semantics = tool.call_semantics(&format!(r#"{{"action":"{action}"}}"#));
            assert!(semantics.mutates_state(), "action={action}");
            assert!(!semantics.observes_state(), "action={action}");
        }
    }

    #[test]
    fn exact_read_action_helper_never_guesses_unknown_actions() {
        let read = semantics_for_exact_read_actions(
            r#"{"action":"list","resource_id":"res_123"}"#,
            &["list", "get"],
            ToolMutationEffects::CONFIGURATION,
        );
        assert!(read.observes_state());
        assert!(!read.mutates_state());
        assert_eq!(
            read.target_hints,
            vec![ToolTargetHint {
                kind: ToolTargetHintKind::ResourceId,
                value: "res_123".to_string(),
            }]
        );

        for arguments in [
            r#"{"action":"listed"}"#,
            r#"{"action":"made_up"}"#,
            r#"{"action":7}"#,
            r#"{}"#,
            "not json",
        ] {
            let semantics = semantics_for_exact_read_actions(
                arguments,
                &["list", "get"],
                ToolMutationEffects::CONFIGURATION,
            );
            assert!(semantics.mutates_state(), "arguments={arguments}");
            assert!(
                semantics
                    .mutation_effects
                    .contains(ToolMutationEffects::CONFIGURATION),
                "arguments={arguments}"
            );
        }
    }

    #[test]
    fn unspecified_evidence_cannot_prove_a_specific_mutation() {
        assert!(
            !ToolMutationEffects::UNSPECIFIED.satisfies(ToolMutationEffects::LOCAL_SOURCE_WRITE)
        );
        assert!(
            !ToolMutationEffects::LOCAL_DERIVED_WRITE.satisfies(ToolMutationEffects::REMOTE_DEPLOY)
        );
        assert!(ToolMutationEffects::LOCAL_SOURCE_WRITE.satisfies(ToolMutationEffects::UNSPECIFIED));
        assert!(ToolMutationEffects::REMOTE_DEPLOY.satisfies(ToolMutationEffects::REMOTE_DEPLOY));
    }

    #[test]
    fn explicit_observation_is_not_widened_by_conservative_fallback() {
        let mut actual = ToolCallSemantics::observation();
        actual.merge_missing_from(ToolCallSemantics::observation_and_mutation());
        assert!(actual.observes_state());
        assert!(!actual.mutates_state());
    }

    struct RememberTool;

    #[async_trait]
    impl Tool for RememberTool {
        fn name(&self) -> &str {
            "remember_fact"
        }

        fn description(&self) -> &str {
            "Store one or more long-lived facts for later"
        }

        fn schema(&self) -> Value {
            json!({
                "name": "remember_fact",
                "description": "Store facts",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            })
        }

        async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
            Ok("ok".to_string())
        }
    }

    #[test]
    fn conservative_capability_default_covers_non_action_tools() {
        let tool = RememberTool;
        let semantics = tool.call_semantics("{}");
        assert!(semantics.mutates_state());
    }

    #[test]
    fn tool_call_metadata_http_status_defaults_to_none() {
        let meta = ToolCallMetadata::default();
        assert_eq!(meta.http_status, None);
    }

    #[test]
    fn tool_call_metadata_with_http_status() {
        let meta = ToolCallMetadata {
            http_status: Some(201),
            ..Default::default()
        };
        assert_eq!(meta.http_status, Some(201));
    }

    #[test]
    fn pre_io_policy_blocks_and_expected_rejections_have_distinct_typed_outcomes() {
        let blocked = ToolCallOutcome::blocked("policy denied execution");
        assert_eq!(
            blocked.metadata.outcome_status,
            Some(ToolOutcomeStatus::Blocked)
        );
        assert!(!blocked.metadata.contract_rejected);
        assert!(!ToolOutcomeStatus::Blocked.satisfies_requested_condition());

        let rejected = ToolCallOutcome::contract_rejection("invalid requested arguments");
        assert_eq!(
            rejected.metadata.outcome_status,
            Some(ToolOutcomeStatus::CompletedWithNegativeResult)
        );
        assert!(rejected.metadata.contract_rejected);
        assert!(rejected.metadata.semantics.observes_state());
        assert!(!rejected.metadata.semantics.mutates_state());
        assert_eq!(
            rejected.metadata.semantics.verification_mode,
            ToolVerificationMode::ResultContent
        );
    }

    #[test]
    fn specialist_kind_from_str_round_trips_for_every_variant() {
        let kinds = [
            SpecialistKind::TaskLead,
            SpecialistKind::Executor,
            SpecialistKind::Research,
            SpecialistKind::ArtifactWriter,
            SpecialistKind::Code,
            SpecialistKind::BrowserVerifier,
            SpecialistKind::Review,
            SpecialistKind::CommsDraft,
            SpecialistKind::Generic,
        ];
        for kind in kinds {
            let s = kind.as_str();
            assert_eq!(
                SpecialistKind::from_str(s),
                Some(kind),
                "round-trip for {:?}",
                kind
            );
        }
        assert_eq!(SpecialistKind::from_str("not_a_kind"), None);
        assert_eq!(SpecialistKind::from_str(""), None);
        assert_eq!(SpecialistKind::from_str("CODE"), None);
    }
}
