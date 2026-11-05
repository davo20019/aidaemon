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

    /// Stable bit representation retained for durable telemetry and audits.
    /// Human-readable names are emitted alongside it; consumers must never
    /// interpret a raw bitset as a count.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Rehydrate a durable effect set without interpreting the bits as a
    /// count. Unknown future bits remain visible to `telemetry_value` and do
    /// not silently become a known effect.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Return the protocol names represented by this effect set. These names
    /// describe typed adapter effects, not natural-language task wording.
    pub fn protocol_names(self) -> Vec<&'static str> {
        [
            (Self::LOCAL_SOURCE_WRITE, "local_source_write"),
            (Self::LOCAL_WORKSPACE_WRITE, "local_workspace_write"),
            (Self::LOCAL_DERIVED_WRITE, "local_derived_write"),
            (Self::REPOSITORY_WRITE, "repository_write"),
            (Self::REMOTE_MUTATION, "remote_mutation"),
            (Self::REMOTE_DEPLOY, "remote_deploy"),
            (Self::EXTERNAL_DELIVERY, "external_delivery"),
            (Self::PROCESS_STATE, "process_state"),
            (Self::CONFIGURATION, "configuration"),
            (Self::DESTRUCTIVE, "destructive"),
            (Self::UNSPECIFIED, "unspecified"),
        ]
        .into_iter()
        .filter_map(|(effect, name)| self.intersects(effect).then_some(name))
        .collect()
    }

    /// Machine-readable telemetry projection that keeps unknown future bits
    /// visible instead of silently dropping them.
    pub fn telemetry_value(self) -> serde_json::Value {
        let known_bits = Self::UNSPECIFIED.bits() | ((1 << 10) - 1);
        serde_json::json!({
            "bits": self.bits(),
            "names": self.protocol_names(),
            "unknown_bits": self.bits() & !known_bits,
        })
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Keep only effects present in both sets.  This is used when an adapter
    /// projects a task's typed obligation onto the subset it can actually
    /// report; it never interprets the bitset as a count.
    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
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
        // A precisely typed local write (source file, derived output) is a
        // workspace write: the broader requirement is satisfied by the more
        // specific receipt, never the reverse.
        let widened = if self.intersects(Self::LOCAL_SOURCE_WRITE)
            || self.intersects(Self::LOCAL_DERIVED_WRITE)
        {
            Self(self.0 | Self::LOCAL_WORKSPACE_WRITE.0)
        } else {
            self
        };
        generic_satisfied && widened.contains(specific_required)
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

/// Time coverage of the subject records an observation surface can return.
/// Receipt timestamps independently prove when the observation itself ran; a
/// current query may therefore support historical subject data when the tool
/// exposes both current and historical records.
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
    /// Legacy coarse goal-state capability. New exact observations should use
    /// the narrow facet that was actually returned.
    GoalState,
    Schedule,
    RunState,
    Recovery,
    Control,
    Measurement,
    Authorization,
    Ownership,
}

impl ToolSemanticFacet {
    pub const PROTOCOL_NAMES: [&'static str; 8] = [
        Self::GoalState.as_str(),
        Self::Schedule.as_str(),
        Self::RunState.as_str(),
        Self::Recovery.as_str(),
        Self::Control.as_str(),
        Self::Measurement.as_str(),
        Self::Authorization.as_str(),
        Self::Ownership.as_str(),
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GoalState => "goal_state",
            Self::Schedule => "schedule",
            Self::RunState => "run_state",
            Self::Recovery => "recovery",
            Self::Control => "control",
            Self::Measurement => "measurement",
            Self::Authorization => "authorization",
            Self::Ownership => "ownership",
        }
    }
}

/// Whether an adapter returned the whole requested collection. A truncated or
/// otherwise partial list may prove facts about the subjects it contains, but
/// it can never prove that another subject is absent.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ToolCollectionCompleteness {
    #[default]
    Unknown,
    Partial,
    Complete,
}

impl ToolCollectionCompleteness {
    pub const fn satisfies(self, required: Self) -> bool {
        matches!(required, Self::Unknown)
            || matches!(required, Self::Partial) && matches!(self, Self::Partial | Self::Complete)
            || matches!(required, Self::Complete) && matches!(self, Self::Complete)
    }
}

/// One exact subject/facet tuple observed in a successful tool result. This is
/// result evidence, not a capability prediction: adapters populate it only
/// after domain I/O has returned.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolObservationEvidence {
    pub subject: ToolTargetHint,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub facets: Vec<ToolSemanticFacet>,
    /// Collections from which this row was selected. Membership is lineage;
    /// absence still requires a separate complete collection observation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_collections: Vec<ToolTargetHint>,
}

impl ToolObservationEvidence {
    pub fn new(subject: ToolTargetHint, facets: &[ToolSemanticFacet]) -> Self {
        Self {
            subject,
            facets: facets.to_vec(),
            source_collections: Vec::new(),
        }
    }

    pub fn with_source_collection(mut self, collection: ToolTargetHint) -> Self {
        if !self.source_collections.contains(&collection) {
            self.source_collections.push(collection);
        }
        self
    }
}

/// Exact coverage of one collection returned by an adapter. `members` contains
/// stable identities, never display labels. Only `Complete` coverage can prove
/// non-membership; counts remain audit facts and are not interpreted as proof
/// on their own.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCollectionObservation {
    pub collection: ToolTargetHint,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub facets: Vec<ToolSemanticFacet>,
    #[serde(default)]
    pub completeness: ToolCollectionCompleteness,
    #[serde(default)]
    pub returned_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<ToolTargetHint>,
}

impl ToolCollectionObservation {
    pub fn complete(
        collection: ToolTargetHint,
        facets: &[ToolSemanticFacet],
        members: Vec<ToolTargetHint>,
    ) -> Self {
        let returned_count = members.len();
        Self {
            collection,
            facets: facets.to_vec(),
            completeness: ToolCollectionCompleteness::Complete,
            returned_count,
            total_count: Some(returned_count),
            members,
        }
    }
}

/// A receipt subject an adapter reports verbatim, published ahead of any call
/// so the executor's checklist can bind to identities that receipts will
/// actually carry. A checklist target outside every advertised subject and
/// member namespace can never be credited, so the checklist tool refuses to
/// compile it into an exact obligation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StableObservationSubject {
    /// Exact `ResourceId` the adapter reports for a fixed collection or
    /// singleton subject, when that identity is known before the call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_id: Option<String>,
    /// Facets a receipt for this subject or its members can carry.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub facets: Vec<ToolSemanticFacet>,
    /// `ResourceId` prefixes of exact member identities the adapter or the
    /// runtime mints (for example `auth_profile:`). A value under a prefix is
    /// exact and is credited only by a receipt naming it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub member_namespaces: Vec<String>,
    /// Short model-facing meaning of the subject.
    pub summary: String,
}

impl StableObservationSubject {
    pub fn collection(
        resource_id: impl Into<String>,
        facets: &[ToolSemanticFacet],
        member_namespace: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            resource_id: Some(resource_id.into()),
            facets: facets.to_vec(),
            member_namespaces: vec![member_namespace.into()],
            summary: summary.into(),
        }
    }

    pub fn namespace(prefix: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            resource_id: None,
            facets: Vec::new(),
            member_namespaces: vec![prefix.into()],
            summary: summary.into(),
        }
    }

    /// Whether `value` is this subject itself or an exact member under one of
    /// its namespaces. A bare prefix with nothing after it is not a member.
    pub fn binds(&self, value: &str) -> bool {
        self.resource_id.as_deref() == Some(value)
            || self
                .member_namespaces
                .iter()
                .any(|prefix| value.len() > prefix.len() && value.starts_with(prefix.as_str()))
    }
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
    /// Immutable capabilities owned by the adapter itself (for example the
    /// system executable roots a confined terminal needs to start a process).
    /// These are deliberately separate from task data grants: callers may use
    /// them for runtime loading, but they do not authorize arbitrary user
    /// paths or writes and are never inferred from command prose.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adapter_read_targets: Vec<ToolTargetHint>,
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

/// Where an invocation reached in the common execution pipeline.
///
/// This is deliberately orthogonal to [`ToolOutcomeStatus`]. A deterministic
/// argument rejection can be a completed validation observation, but it did
/// not dispatch the requested domain operation and therefore cannot inherit
/// that operation's effects, targets, or evidence capabilities.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolInvocationStage {
    /// Legacy receipt or adapter that did not report a stage.
    #[default]
    Unknown,
    /// The common policy/retry boundary refused dispatch.
    RejectedBeforeDispatch,
    /// The adapter rejected its typed argument contract before domain I/O.
    RejectedBeforeIo,
    /// The adapter was invoked for the prepared operation.
    Dispatched,
    /// The result was reconstructed from a durable prior receipt.
    Replayed,
}

impl ToolInvocationStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::RejectedBeforeDispatch => "rejected_before_dispatch",
            Self::RejectedBeforeIo => "rejected_before_io",
            Self::Dispatched => "dispatched",
            Self::Replayed => "replayed",
        }
    }

    pub const fn reached_dispatch(self) -> bool {
        matches!(self, Self::Dispatched | Self::Replayed)
    }

    /// True only for a newly executed adapter call. A durable replay carries
    /// evidence forward but does not consume another operation attempt.
    pub const fn performed_io(self) -> bool {
        matches!(self, Self::Dispatched)
    }
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

/// Strength of the boundary that enforced a tool's declared filesystem
/// manifest. This reports enforcement, not merely manifest presence.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolAccessEnforcement {
    /// Legacy or uninstrumented adapter; no enforcement claim is made.
    #[default]
    Unknown,
    /// The operation has no filesystem access contract.
    NotApplicable,
    /// A delegated sandbox adapter accepted the manifest.
    AdapterEnforced,
    /// The daemon controller rejected the proposal before adapter dispatch
    /// because it exceeded the compiled task capability.
    ControllerEnforced,
    /// The daemon installed a kernel-enforced policy from the exact manifest.
    KernelEnforced,
}

/// Typed diagnostic emitted when a proposal is rejected by the controller or
/// an adapter is denied by its installed filesystem policy. This is audit
/// telemetry only; the manifest remains the authority and this record never
/// widens it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolAccessDenial {
    pub reason_code: String,
    pub enforcement: ToolAccessEnforcement,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    /// Evidence the refused operation WOULD have produced. A rejection
    /// receipt carries no observed evidence (nothing ran), but the run ledger
    /// needs to know which observation obligation the refusal was attempting
    /// to satisfy; that binding is proposal data, typed on the denial.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proposed_evidence: Vec<ToolEvidenceCapability>,
}

impl ToolOutcomeStatus {
    /// Normalize the execution backend's process status convention. Unix
    /// commands that exit normally produce a nonnegative code; the backend
    /// uses a negative sentinel when no normal exit status exists (for
    /// example, a sandbox kill). Such termination is not a completed negative
    /// predicate observation.
    pub const fn from_process_exit_code(code: i32) -> Self {
        if code == 0 {
            Self::Succeeded
        } else if code > 0 {
            Self::CompletedWithNegativeResult
        } else {
            Self::FailedPermanent
        }
    }

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
    /// Whether the declared access manifest was installed at an execution
    /// boundary. A populated manifest with `Unknown` is audit data only.
    #[serde(default)]
    pub access_enforcement: ToolAccessEnforcement,
    /// Kernel/adapter denial details when a dispatched process was prevented
    /// from opening a path. The declared manifest is retained separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_denial: Option<ToolAccessDenial>,
    /// Authoritative domain outcome when the tool can provide one. Older tools
    /// may omit it; the loop then derives a conservative status from structured
    /// exit/HTTP/transport metadata before falling back to legacy text parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_status: Option<ToolOutcomeStatus>,
    /// Authoritative pipeline stage for this result. Completion and effect
    /// projection must never infer dispatch from intended call semantics.
    #[serde(default)]
    pub invocation_stage: ToolInvocationStage,
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
    /// Exact subject/facet observations made by this dispatched result. These
    /// are intentionally separate from `ToolCallSemantics`: proposed call
    /// semantics must never become proof merely because dispatch was planned.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observations: Vec<ToolObservationEvidence>,
    /// Collection membership and coverage observed by this dispatched result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collection_observations: Vec<ToolCollectionObservation>,
    /// Complete typed result for successful text-file reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_file: Option<ReadFileResultMetadata>,
    /// Image (or other) files produced by the tool for agent vision context.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<crate::traits::MessageAttachment>,
    /// The runtime's own pre-dispatch preparation failed (for example
    /// materializing a declared write root), attributed to the exact argument
    /// declaration that caused it. Set only when no domain I/O ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_preparation_failure: Option<Box<RuntimePreparationFailure>>,
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
    /// Who authored this exact output, when the tool knows better than its
    /// declared `Tool::output_trust` (a delegated run reports whether it read
    /// untrusted content). The dispatcher stamps this over the declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_trust: Option<aidaemon_core::SourceTrust>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallOutcome {
    pub output: String,
    #[serde(default)]
    pub metadata: ToolCallMetadata,
}

/// A failure of the runtime's own pre-dispatch preparation, attributed to the
/// exact argument declaration that caused it. The dispatcher uses this to
/// decide whether the declaration was its own projection (repairable without
/// involving the model) or the model's (surfaced as a typed rejection).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimePreparationFailure {
    /// Argument field the declaration came from (for example `write_roots`).
    pub field: String,
    /// The exact declared value.
    pub value: String,
    /// The underlying runtime error.
    pub error: String,
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
                // A rejected invocation observed no domain condition. Keep it
                // distinct from an executed observation whose answer is
                // legitimately negative (for example process exit 1).
                outcome_status: Some(ToolOutcomeStatus::Blocked),
                invocation_stage: ToolInvocationStage::RejectedBeforeIo,
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

    /// The runtime could not prepare the invocation (no domain I/O ran). The
    /// failure names the exact declaration so the dispatcher can repair its
    /// own projections instead of reporting its mistake to the model.
    pub fn runtime_preparation_failure(
        field: impl Into<String>,
        value: impl Into<String>,
        error: impl std::fmt::Display,
    ) -> Self {
        let field = field.into();
        let value = value.into();
        let error = error.to_string();
        Self {
            output: format!(
                "Error: the runtime could not prepare `{field}` entry `{value}` before running the command: {error}"
            ),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Blocked),
                invocation_stage: ToolInvocationStage::RejectedBeforeIo,
                semantics: ToolCallSemantics::observation()
                    .with_verification_mode(ToolVerificationMode::ResultContent),
                runtime_preparation_failure: Some(Box::new(RuntimePreparationFailure {
                    field,
                    value,
                    error,
                })),
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

    pub fn with_observation_evidence(
        mut self,
        observations: Vec<ToolObservationEvidence>,
        collection_observations: Vec<ToolCollectionObservation>,
    ) -> Self {
        self.metadata.observations = observations;
        self.metadata.collection_observations = collection_observations;
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
    ) {
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
        adapter_read_targets: Vec::new(),
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

/// The scope of an approval the agent kernel gave a call (see
/// [`ToolExecutionContext::kernel_approved`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalScope {
    /// This call came through the kernel, which judged it.
    Call,
    /// One exact inner request (`METHOD url`) of an outer call the kernel
    /// approved, for example `manage_api`'s live probe.
    Request { method: String, url: String },
}

/// Dispatcher-owned execution context passed to tools via the trait method.
///
/// This is a Rust-side control-plane type — it is NEVER serialized, never appears
/// in model-visible JSON, tool schemas, logs, or persisted tool arguments.
/// Trust and approval facts belong here, not in the enriched args: a tool
/// reads them from this context only (`trusted_session`, `user_role`,
/// `untrusted_source`, `scope_escalation`, `kernel_approved`). The
/// `_trusted_session` and `_user_role` arguments the dispatcher still adds
/// remain for tools not yet moved off them.
#[derive(Debug, Clone, Default)]
pub struct ToolExecutionContext {
    /// True only after the final common dispatcher revalidated an exact,
    /// action-bound `MandateAuthorityGrant` for this mutating call, so tools
    /// can honor owner-delegated autonomy.
    pub mandate_preapproved: bool,
    /// True only when the common dispatcher resolved this call to an active
    /// autonomous mandate. Network adapters use this control-plane bit to keep
    /// one target-bound call from following redirects outside the target that
    /// was evaluated by the mandate authority kernel.
    pub mandate_execution: bool,
    /// Hard negative task contract propagated from the validated request.
    /// Open-ended adapters must prove an operation is observational before I/O.
    pub mutation_forbidden: bool,
    /// What the agent kernel approved, set only by the dispatcher (never
    /// from arguments): `Call` for a call that came through the kernel,
    /// whose `ApprovalPolicy` already asked the owner where the daemon's
    /// rules require it; `Request` for one exact inner request an approved
    /// outer call makes. A tool that needs the owner's approval refuses a
    /// call without a matching scope rather than asking itself.
    pub kernel_approved: Option<ApprovalScope>,
    /// The turn is a trusted scheduled session (never one under a mandate),
    /// set only by the dispatcher from the channel context and the goal's
    /// provenance. Tools read trust here, never from their arguments.
    pub trusted_session: bool,
    /// The role of the turn's user, set only by the dispatcher (`None`
    /// outside it: an unknown caller is not the owner).
    pub user_role: Option<crate::types::UserRole>,
    /// The session was started by an automated trigger (an email, an
    /// event), not by the owner. Set only by the dispatcher.
    pub untrusted_source: bool,
    /// Exact out-of-scope targets the controller routed to the owner's
    /// approval instead of refusing them. Set only by the dispatcher.
    pub scope_escalation: Option<ScopeEscalation>,
}

/// Filesystem targets a tool call declared that fall outside the compiled
/// task scope but are not protected host data or credential stores. The
/// controller does not reject such a call outright; it hands the exact
/// out-of-scope targets to the tool, which must obtain explicit user approval
/// (or a persisted capability grant) before running with them.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScopeEscalation {
    #[serde(default)]
    pub read_paths: Vec<String>,
    #[serde(default)]
    pub write_paths: Vec<String>,
}

impl ScopeEscalation {
    pub fn is_empty(&self) -> bool {
        self.read_paths.is_empty() && self.write_paths.is_empty()
    }
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
    /// The concrete tool, for daemon wiring that needs it (the kernel's
    /// approval rules find the terminal's approval ladder this way). `None`
    /// for tools nothing downcasts.
    fn as_any(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        None
    }
    fn description(&self) -> &str;
    /// Returns the OpenAI-format function schema as a JSON Value.
    fn schema(&self) -> Value;
    /// Canonicalize provider argument projections before any semantics,
    /// authority, retry, or access decision is made. The default is identity;
    /// adapters with representational aliases or optional-empty projection
    /// quirks can normalize them without teaching the orchestration layer
    /// adapter-specific rules.
    fn canonicalize_arguments(
        &self,
        arguments: &str,
    ) -> Result<String, ToolArgumentContractViolation> {
        Ok(arguments.to_string())
    }

    /// Produce the single typed invocation object consumed by the execution
    /// control plane. Validation deliberately precedes semantics and access
    /// derivation so a rejected proposal can never acquire intended effects.
    fn prepare_invocation(
        &self,
        arguments: &str,
    ) -> Result<PreparedToolInvocation, ToolArgumentContractViolation> {
        let canonical_arguments = self.canonicalize_arguments(arguments)?;
        self.validate_arguments(&canonical_arguments)?;
        Ok(PreparedToolInvocation {
            semantics: self.call_semantics(&canonical_arguments),
            access_manifest: self.call_access_manifest(&canonical_arguments),
            receipt_kind: self.receipt_kind(&canonical_arguments),
            canonical_arguments,
        })
    }
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

    /// Context-aware execution path used by the agent loop.
    ///
    /// The default implementation discards `exec_ctx` and delegates to
    /// `call_with_status_outcome`, so existing tools are completely unaffected.
    /// Tools that honor dispatcher-owned authority (`terminal`, `http_request`)
    /// override this method and inspect `exec_ctx` (e.g. `mandate_preapproved`).
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

    /// Live state this tool holds for a session that outlives a turn (for
    /// example background processes still running), rendered for the next
    /// turn's task context so a follow-up sees it instead of guessing.
    async fn session_context_note(&self, _session_id: &str) -> Option<String> {
        None
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

    /// The tool only manages the owner's own durable policy (mandates). A
    /// request assessment that narrows a turn's tools still offers it to the
    /// owner in a private chat, but only calls `owner_control_plane_call`
    /// accepts are admitted there.
    fn owner_control_plane(&self) -> bool {
        false
    }

    /// Whether this exact call is safe on a turn whose assessment did not
    /// name the tool: read-only, or committing only through an explicit
    /// owner confirmation. Decided from the tool's own schema enum.
    fn owner_control_plane_call(&self, _arguments: &str) -> bool {
        false
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

    /// Capabilities supplied by the adapter rather than requested as task
    /// data.  The common dispatcher uses this typed lane when attenuating the
    /// call against the task manifest.  The default is empty so an adapter
    /// cannot accidentally widen authority by inheriting a generic grant.
    fn adapter_owned_access_manifest(&self, _arguments: &str) -> ToolCallAccessManifest {
        ToolCallAccessManifest::default()
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

    /// Exact receipt subjects this adapter reports verbatim: fixed collection
    /// IDs and the namespaces of member identities it mints. The checklist
    /// tool publishes the union so declared targets bind only to identities a
    /// receipt can carry. The default advertises nothing.
    fn stable_observation_subjects(&self) -> Vec<StableObservationSubject> {
        Vec::new()
    }

    /// Authority annotations for this exact call: read-only, destructive,
    /// idempotent, and whether it sends data or acts outside the owner's
    /// machine and workspace (`egress`). Derived from the call's typed
    /// semantics, never from the tool's name.
    fn call_annotations(&self, arguments: &str) -> aidaemon_core::ToolAnnotations {
        default_call_annotations(self.capabilities(), &self.call_semantics(arguments))
    }

    /// The typed effects of this exact call (effect class and normalized
    /// target), which the kernel checks against the owner's declines and
    /// hands the approval and egress policies. Build keys only with the
    /// kernel constructors, through `command_effects::{path_effect,
    /// egress_effect, url_effect}`. The default is `CallEffects::none()` for a
    /// tool whose capabilities are read-only and `CallEffects::unknown()`
    /// otherwise; a tool that can name what a call does overrides it.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        if self.capabilities().read_only {
            aidaemon_core::CallEffects::none()
        } else {
            aidaemon_core::CallEffects::unknown()
        }
    }

    /// `call_annotations` judged against the turn's trusted roots, for
    /// effects whose class depends on where they land (a delete inside the
    /// owner's roots is not egress). The default ignores the roots.
    fn call_annotations_in(
        &self,
        arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        self.call_annotations(arguments)
    }

    /// Who authored what this exact call returns. The default is
    /// `Untrusted` iff the tool reaches outside the machine
    /// (`external_side_effect`); local tools override with the owner's
    /// trusted roots.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        if self.capabilities().external_side_effect {
            aidaemon_core::SourceTrust::Untrusted
        } else {
            aidaemon_core::SourceTrust::Trusted
        }
    }

    /// Whether this tool is currently operational.
    ///
    /// Default: true. Override for tools with dynamic backends that may be
    /// temporarily unavailable at runtime.
    fn is_available(&self) -> bool {
        true
    }
}

/// Effects that send data or act outside the owner's machine and workspace,
/// or cannot be undone. A call with any of these is egress.
pub const EGRESS_EFFECTS: ToolMutationEffects = ToolMutationEffects::EXTERNAL_DELIVERY
    .union(ToolMutationEffects::REMOTE_MUTATION)
    .union(ToolMutationEffects::REMOTE_DEPLOY)
    .union(ToolMutationEffects::DESTRUCTIVE)
    .union(ToolMutationEffects::CONFIGURATION);

/// Per-call authority annotations from a tool's capabilities and the call's
/// typed semantics.
pub fn default_call_annotations(
    cap: ToolCapabilities,
    semantics: &ToolCallSemantics,
) -> aidaemon_core::ToolAnnotations {
    let read_only = cap.read_only
        || (matches!(
            semantics.effect,
            ToolCallEffect::Observation | ToolCallEffect::Administrative
        ) && !semantics.mutates_state());
    let effects = semantics.mutation_effects;
    let egress = !read_only
        && (effects.intersects(EGRESS_EFFECTS)
            || (cap.external_side_effect && !effects.has_specific_effects()));
    aidaemon_core::ToolAnnotations {
        read_only,
        destructive: !read_only && effects.intersects(ToolMutationEffects::DESTRUCTIVE),
        idempotent: read_only,
        egress,
    }
}

/// The owner's trusted file roots for one turn. A path is trusted only when
/// it normalizes without escaping and equals a root or sits below it. A
/// relative path resolves against `cwd` (where the file tools resolve it);
/// without one it is never trusted.
pub struct TrustRoots<'a> {
    roots: &'a [String],
    cwd: Option<&'a str>,
}

impl<'a> TrustRoots<'a> {
    pub fn new(roots: &'a [String]) -> Self {
        Self { roots, cwd: None }
    }

    /// Resolve relative paths against `cwd` (an absolute directory).
    pub fn with_cwd(mut self, cwd: Option<&'a str>) -> Self {
        self.cwd = cwd;
        self
    }

    /// The same roots with relative paths resolved against `cwd` instead
    /// (a call that runs in its own directory).
    pub fn anchored_at<'b>(&'b self, cwd: Option<&'b str>) -> TrustRoots<'b> {
        TrustRoots {
            roots: self.roots,
            cwd,
        }
    }

    /// The directory relative paths resolve against, if any.
    pub fn cwd(&self) -> Option<&str> {
        self.cwd
    }

    pub fn contains(&self, path: &str) -> bool {
        let resolved = match self.cwd {
            Some(cwd) if !is_absolute_or_home(path) => {
                format!("{}/{}", cwd.trim_end_matches('/'), path.trim())
            }
            _ => path.to_string(),
        };
        let Some(path) = resolve_absolute(&resolved) else {
            return false;
        };
        self.roots
            .iter()
            .any(|root| resolve_absolute(root).is_some_and(|root| path.starts_with(&root)))
    }
}

/// Expand `~`, then resolve the path the way the filesystem will: every
/// existing prefix through its symlinks (a link inside a root that points
/// elsewhere leaves the root), the rest lexically. `None` for a relative
/// path or one whose `..` climbs above the root.
fn resolve_absolute(path: &str) -> Option<std::path::PathBuf> {
    use std::path::{Component, PathBuf};
    let expanded = shellexpand::tilde(path.trim()).into_owned();
    let path = std::path::Path::new(&expanded);
    if !path.is_absolute() {
        return None;
    }
    let mut resolved = PathBuf::new();
    let mut existing = true;
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !resolved.pop() {
                    return None;
                }
            }
            Component::Normal(part) => {
                resolved.push(part);
                if existing {
                    match std::fs::canonicalize(&resolved) {
                        Ok(canonical) => resolved = canonical,
                        Err(_) => existing = false,
                    }
                }
            }
            other => resolved.push(other.as_os_str()),
        }
    }
    Some(resolved)
}

fn is_absolute_or_home(path: &str) -> bool {
    let path = path.trim();
    path.starts_with('~') || std::path::Path::new(path).is_absolute()
}

/// Validated, canonical tool input. This is the boundary object shared by
/// policy, retry, dispatch, and receipt projection; raw model JSON is not an
/// execution fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedToolInvocation {
    pub canonical_arguments: String,
    pub semantics: ToolCallSemantics,
    pub access_manifest: ToolCallAccessManifest,
    pub receipt_kind: ToolReceiptKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps(external: bool) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: external,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }

    #[test]
    fn default_call_annotations_types_egress_from_effects() {
        let ann = |semantics: ToolCallSemantics, external: bool| {
            default_call_annotations(caps(external), &semantics)
        };
        let delivery = ann(
            ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY),
            false,
        );
        assert!(delivery.egress && !delivery.read_only);
        let local = ann(
            ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
            true,
        );
        assert!(!local.egress && !local.read_only);
        let config = ann(
            ToolCallSemantics::mutation_with(ToolMutationEffects::CONFIGURATION),
            false,
        );
        assert!(config.egress);
        let destructive = ann(
            ToolCallSemantics::mutation_with(ToolMutationEffects::DESTRUCTIVE),
            false,
        );
        assert!(destructive.egress && destructive.destructive);
        let observed = ann(ToolCallSemantics::observation(), true);
        assert!(observed.read_only && !observed.egress && observed.idempotent);
        assert!(ann(ToolCallSemantics::mutation(), true).egress);
        assert!(!ann(ToolCallSemantics::mutation(), false).egress);
    }

    #[test]
    fn trust_roots_contains_only_descendants() {
        let roots = vec!["/tmp/aidaemon-test/ws".to_string()];
        let roots = TrustRoots::new(&roots);
        assert!(roots.contains("/tmp/aidaemon-test/ws"));
        assert!(roots.contains("/tmp/aidaemon-test/ws/a.txt"));
        assert!(!roots.contains("/tmp/aidaemon-test/ws2/a.txt"));
        assert!(!roots.contains("/tmp/aidaemon-test/ws/../x"));
        assert!(!roots.contains("src/lib.rs"), "relative with no cwd");
        let list = vec!["/tmp/aidaemon-test/ws".to_string()];
        let anchored = TrustRoots::new(&list).with_cwd(Some("/tmp/aidaemon-test/ws"));
        assert!(anchored.contains("src/lib.rs"));
        assert!(anchored.contains("./src/../Cargo.toml"));
        assert!(!anchored.contains("../other/a.txt"));
        assert!(!anchored.contains("/etc/hosts"));
        assert!(!roots.contains("a.txt"));
        assert!(!TrustRoots::new(&[]).contains("/tmp/aidaemon-test/ws/a.txt"));
    }

    /// A symlink inside a root that points outside it does not carry the
    /// root's trust: existing paths are compared after resolving links.
    #[cfg(unix)]
    #[test]
    fn trust_roots_resolve_symlinks_before_comparing() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("x.md"), "synthetic").unwrap();
        std::fs::write(root.join("a.txt"), "synthetic").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("docs")).unwrap();
        let list = vec![root.to_string_lossy().into_owned()];
        let roots = TrustRoots::new(&list);
        assert!(roots.contains(&root.join("a.txt").to_string_lossy()));
        assert!(!roots.contains(&root.join("docs/x.md").to_string_lossy()));
        assert!(!roots.contains(&root.join("docs/new.md").to_string_lossy()));
        // A path that does not exist yet is compared lexically.
        assert!(roots.contains(&root.join("later/b.txt").to_string_lossy()));
    }

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
    fn default_manifest_does_not_treat_opaque_payload_text_as_a_path_capability() {
        let tool = ManageTool;
        let manifest = tool
            .call_access_manifest(r#"{"command":"/usr/bin/false","prompt":"inspect /etc/hosts"}"#);
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
    fn process_status_distinguishes_normal_negative_exit_from_termination() {
        assert_eq!(
            ToolOutcomeStatus::from_process_exit_code(0),
            ToolOutcomeStatus::Succeeded
        );
        assert_eq!(
            ToolOutcomeStatus::from_process_exit_code(1),
            ToolOutcomeStatus::CompletedWithNegativeResult
        );
        assert_eq!(
            ToolOutcomeStatus::from_process_exit_code(-1),
            ToolOutcomeStatus::FailedPermanent
        );
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
            Some(ToolOutcomeStatus::Blocked)
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
