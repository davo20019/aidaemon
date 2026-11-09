//! The semantic half of a daemon tool: plain data that describes a call
//! without executing it (capabilities, typed call semantics, access
//! manifests, receipt metadata, execution facts) and the helpers that derive
//! the conservative defaults.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use aidaemon_core::ChannelVisibility;

/// The sender's role, shared across the SDK.
pub use aidaemon_core::UserRole;

/// Where an attachment originated — drives vision retention in archived turns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentProvenance {
    /// User sent via a channel (Telegram, Slack, Discord).
    #[default]
    Inbound,
    /// Produced by a tool observation (browser screenshot, MCP image, etc.).
    ToolObservation,
    /// Produced as a durable artifact by a tool (PDF, document, archive, etc.).
    ToolArtifact,
}

/// Metadata for a file saved from an inbound channel message or tool observation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct MessageAttachment {
    /// Stable opaque handle used by tools and policy. Legacy attachments omit
    /// it and receive one when they next cross the canonical event boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_id: Option<String>,
    pub local_path: String,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub provenance: AttachmentProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_tool: Option<String>,
    /// Content digest captured when the attachment was registered. Execution
    /// re-checks this before sensitive use to detect stale/replaced files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
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

/// Typed facts about how the dispatcher and the engine treat a tool. A tool
/// states them about itself; nothing keys on its name. Several can hold for
/// one tool, so they are flags beside [`ToolRole`] (which scopes tools by
/// agent role) rather than more roles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolDispatchFacts {
    /// The kernel's approval rules judge this tool's calls in the canonical
    /// form the dispatcher executes, and the dispatcher then runs each call
    /// exactly as judged (the terminal).
    pub runs_as_judged: bool,
    /// The tool keeps the owner's answers to its cards in its own grant
    /// stores, consulted by its own ladder (the terminal's command
    /// prefixes), instead of the kernel's session grants.
    pub keeps_own_grants: bool,
    /// The call's `command` or `script` is a shell command line, which the
    /// kernel's access manifest carries so the approval policy's
    /// catastrophic-command check sees it (the terminal).
    pub shell_command_line: bool,
    /// The tool survives every profile and approval pruning of the offered
    /// tools: without it the agent cannot do its core work (write and edit
    /// files, run commands, keep memory, reach the web, deliver files).
    pub always_offered: bool,
    /// A task lead keeps the tool through pruning: it is its delegation
    /// surface (`spawn_agent`, `cli_agent`, `manage_goal_tasks`).
    pub task_lead_delegation: bool,
    /// How long a run may call the tool consecutively before the
    /// same-tool guard treats it as a loop.
    pub same_tool_streak: SameToolStreak,
    /// The tool delivers a file to the user; the same file and caption go
    /// out once per task.
    pub delivers_files: bool,
    /// The tool spawns child agents, which inherit the turn's authority as
    /// it is when the call runs.
    pub spawns_child_agents: bool,
    /// The tool is the executor's requirement checklist: the status line
    /// shows the checklist, and its items become ledger expectations.
    pub requirement_checklist: bool,
    /// The call runs inside a process the daemon cannot see into (an MCP
    /// server), so its downstream effects escape per-action scoping and
    /// accounting. A mandate never delegates such a tool.
    pub opaque_execution: bool,
    /// The tool manages its own timeout and backgrounding, so the generic
    /// watchdog never wraps it (it would cancel the call early and could
    /// orphan the process it started).
    pub manages_own_timeout: bool,
    /// The tool records the trusted roots its launch was judged with (the
    /// terminal), so a background result keeps the trust it ran under.
    pub records_launch_trust: bool,
    /// Which goal facts the call receives.
    pub goal_context: GoalContext,
    /// Which addresses a call may name, judged by `web::NetworkGuard`.
    pub network_reach: NetworkReach,
    /// Which of a call's paths `ProtectedPathGuard` judges.
    pub file_guard: FileGuard,
    /// The dispatcher takes a file checkpoint before the call runs, so its
    /// writes can be rolled back.
    pub checkpoints_writes: bool,
    /// The call receives the live model, provider and model chain (a tool
    /// that runs its own model loop).
    pub needs_live_model: bool,
    /// The status line summarizes a finished call from its arguments, not
    /// its result (memory and people edits, whose result text is terse).
    pub status_from_arguments: bool,
}

/// Which of a tool's paths are judged against the secret and sensitive
/// patterns. A shell is not: the owner is asked about its commands instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FileGuard {
    #[default]
    None,
    /// The paths it reads must not hold secrets.
    Reads,
    /// The paths it reads must not be sensitive (a superset of secrets, for
    /// a tool that publishes what it reads).
    SensitiveReads,
    /// The paths it changes must not be sensitive.
    Writes,
}

/// Which addresses a tool's egress may name. Cloud metadata endpoints are
/// refused for every tool, whatever this says.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NetworkReach {
    /// Anything else, as a shell reaches the owner's own network.
    #[default]
    Any,
    /// Public hosts and this computer.
    PublicAndLoopback,
    /// Public hosts only.
    PublicOnly,
}

/// Which goal facts a call receives (see
/// [`ToolDispatchFacts::goal_context`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GoalContext {
    /// None.
    #[default]
    None,
    /// The goal the run works for.
    Goal,
    /// The goal, found through the executor's task when the run carries
    /// only a task: the call's background notifications go to the goal's
    /// own session, since a child agent's session is not routable.
    BackgroundDelivery,
    /// The goal and, in a mandate run, the run's mandate fence (goal run,
    /// attempt, mandate and version).
    MandateRun,
}

/// The consecutive-call allowance of a tool (see
/// [`ToolDispatchFacts::same_tool_streak`]). Defined in the kernel so the
/// same-tool guard can take it without a dependency on the tools crate.
pub use aidaemon_core::SameToolStreak;

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
    pub const PROTOCOL_NAMES: [&str; 12] = [
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
        "scheduled_work",
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
    /// Creating or changing the daemon's own goals, schedules and reminders.
    /// Nothing leaves the machine; the work it sets up is judged when it runs.
    pub const SCHEDULED_WORK: Self = Self(1 << 11);

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
            (Self::SCHEDULED_WORK, "scheduled_work"),
        ]
        .into_iter()
        .filter_map(|(effect, name)| self.intersects(effect).then_some(name))
        .collect()
    }

    /// Machine-readable telemetry projection that keeps unknown future bits
    /// visible instead of silently dropping them.
    pub fn telemetry_value(self) -> serde_json::Value {
        let known_bits = Self::SCHEDULED_WORK.bits() | Self::UNSPECIFIED.bits() | ((1 << 10) - 1);
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
            "scheduled_work" => Some(Self::SCHEDULED_WORK),
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
        // A schedule is what a model means by `configuration` or
        // `process_state` when it plans one: the goal tools have nothing else
        // to report, so those plans must be able to close.
        let widened = if widened.intersects(Self::SCHEDULED_WORK) {
            Self(widened.0 | Self::CONFIGURATION.0 | Self::PROCESS_STATE.0)
        } else {
            widened
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
    /// The call handed the rest of the turn's work to a background job whose
    /// result reaches the session later (a launch moved to background
    /// tracking, a background sub-agent). Set with the call's finish by
    /// [`ToolCallOutcome::handed_to_background`]; unlike
    /// `background_started` it says nothing about the receipt's status.
    #[serde(default)]
    pub background_handoff: bool,
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
    pub attachments: Vec<MessageAttachment>,
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
    /// The tool ends the run on this call, from its own typed outcome (never
    /// from argument or output text). The dispatcher hands it to the kernel
    /// as [`aidaemon_core::ToolOutput::finish`]; a finish without a reply
    /// ships the call's own output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish: Option<aidaemon_core::Finish>,
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
            finish: None,
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
            finish: None,
        }
    }

    pub fn blocked(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Blocked),
                ..ToolCallMetadata::default()
            },
            finish: None,
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
            finish: None,
        }
    }

    pub fn completed_negative_result(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::CompletedWithNegativeResult),
                ..ToolCallMetadata::default()
            },
            finish: None,
        }
    }

    /// End the run on this call (see [`Self::finish`]).
    pub fn with_finish(mut self, finish: aidaemon_core::Finish) -> Self {
        self.finish = Some(finish);
        self
    }

    /// The call moved the rest of the work to the background: the run ends
    /// on it as partial work, and the dispatcher writes the handoff reply
    /// (it knows the user's request and whether completion notifications
    /// reach them).
    pub fn handed_to_background(mut self) -> Self {
        self.metadata.background_handoff = true;
        self.finish = Some(aidaemon_core::Finish::new(
            aidaemon_core::FinishOutcome::Partial,
            None,
        ));
        self
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

fn default_semantics_from_identity(arguments: &str, caps: ToolCapabilities) -> ToolCallSemantics {
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
/// [`ToolCallContext::kernel_approved`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalScope {
    /// This call came through the kernel, which judged it.
    Call,
    /// One exact inner request (`METHOD url`) of an outer call the kernel
    /// approved, for example `manage_api`'s live probe.
    Request { method: String, url: String },
}

/// One call as the kernel's approval rules show it to
/// [`ToolSemantics::owner_approval_request`].
#[derive(Debug, Clone, Copy)]
pub struct ApprovalCall<'a> {
    /// The arguments the dispatcher will execute (for a tool whose calls the
    /// kernel judges in canonical form, the canonical form; otherwise the
    /// model's own).
    pub arguments: &'a Value,
    /// The arguments as the model sent them, for a card about a call whose
    /// arguments cannot be read.
    pub as_sent: &'a Value,
    /// The dispatcher refuses this call before it runs (an out-of-scope
    /// target nobody may authorize), so nothing about it needs the owner.
    pub refused: bool,
}

impl<'a> ApprovalCall<'a> {
    /// A call the dispatcher runs as the model sent it.
    pub fn as_sent(arguments: &'a Value) -> Self {
        Self {
            arguments,
            as_sent: arguments,
            refused: false,
        }
    }
}

/// What a tool says about the owner's approval for one call.
#[derive(Debug, Clone, PartialEq)]
pub enum OwnerApprovalNeed {
    /// The tool has no opinion: the approval policy's allow rules and mode
    /// decide.
    Unjudged,
    /// The tool judged the call and it needs no approval.
    NotNeeded,
    /// Ask the owner with this request (unless a grant covers it).
    Ask(aidaemon_core::OwnerApprovalRequest),
}

/// The dispatcher's facts about one tool call, passed to the tool beside its
/// arguments.
///
/// This is a Rust-side control-plane type: it is NEVER serialized and never
/// appears in model-visible JSON, tool schemas, logs, or persisted tool
/// arguments. Every fact about who is calling, from where, and under what
/// authority lives here and only here: a tool's arguments are the model's
/// own, and the dispatcher strips any `_`-prefixed key a model sends.
#[derive(Clone, Default)]
pub struct ToolCallContext {
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
    /// provenance.
    pub trusted_session: bool,
    /// The role of the turn's user (`None` outside the dispatcher: an
    /// unknown caller is not the owner).
    pub user_role: Option<UserRole>,
    /// The session was started by an automated trigger (an email, an
    /// event), not by the owner.
    pub untrusted_source: bool,
    /// Exact out-of-scope targets the controller routed to the owner's
    /// approval instead of refusing them.
    pub scope_escalation: Option<ScopeEscalation>,
    /// The session the call belongs to (empty outside the dispatcher).
    pub session_id: String,
    /// Visibility of the channel the turn came from (`None` outside the
    /// dispatcher; each tool keeps its own fallback).
    pub channel_visibility: Option<ChannelVisibility>,
    /// The channel the turn came from, when the channel has an id.
    pub channel_id: Option<String>,
    /// The task the call runs under.
    pub task_id: Option<String>,
    /// The model's id for this tool call.
    pub tool_call_id: Option<String>,
    /// The call is the only one in a root turn's model response, so a reply
    /// the tool finishes the run with answers the user's request directly.
    pub lone_root_call: bool,
    /// The session's current turn.
    pub turn_id: Option<String>,
    /// The goal the call serves: the agent's goal, or (for `cli_agent` and
    /// `terminal`) the goal of the executor's task, so background
    /// notifications reach the origin session.
    pub goal_id: Option<String>,
    /// The goal run of the mandate fence the call runs under.
    pub goal_run_id: Option<String>,
    /// The task attempt of the mandate fence the call runs under.
    pub task_attempt_id: Option<String>,
    /// The mandate of the fence the call runs under.
    pub mandate_id: Option<String>,
    /// The mandate version of the fence the call runs under.
    pub mandate_version: Option<i64>,
    /// The single project the turn is scoped to.
    pub project_scope: Option<String>,
    /// A guest's granted workspace root (only under an active grant).
    pub workspace_collaboration_root: Option<String>,
    /// The turn's trusted file roots (set for `terminal`), so a run moved to
    /// the background records the trust its launch was judged with.
    pub trust_roots: Option<Vec<String>>,
    /// The cwd relative paths resolve against for [`Self::trust_roots`].
    pub trust_cwd: Option<String>,
    /// The live model (set for `computer_use`).
    pub model: Option<String>,
    /// The provider kind behind [`Self::model`], as the daemon names it.
    pub provider_kind: Option<String>,
    /// The router's models in order, when a router is configured.
    pub model_chain: Option<Vec<String>>,
    /// Application facts whose types cannot live in the SDK (they depend on
    /// the application's configuration), read by the tool that needs them
    /// through a typed getter.
    pub extensions: aidaemon_core::Extensions,
}

impl std::fmt::Debug for ToolCallContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallContext")
            .field("mandate_preapproved", &self.mandate_preapproved)
            .field("mandate_execution", &self.mandate_execution)
            .field("mutation_forbidden", &self.mutation_forbidden)
            .field("kernel_approved", &self.kernel_approved)
            .field("trusted_session", &self.trusted_session)
            .field("user_role", &self.user_role)
            .field("untrusted_source", &self.untrusted_source)
            .field("scope_escalation", &self.scope_escalation)
            .field("session_id", &self.session_id)
            .field("channel_visibility", &self.channel_visibility)
            .field("channel_id", &self.channel_id)
            .field("task_id", &self.task_id)
            .field("tool_call_id", &self.tool_call_id)
            .field("lone_root_call", &self.lone_root_call)
            .field("turn_id", &self.turn_id)
            .field("goal_id", &self.goal_id)
            .field("goal_run_id", &self.goal_run_id)
            .field("task_attempt_id", &self.task_attempt_id)
            .field("mandate_id", &self.mandate_id)
            .field("mandate_version", &self.mandate_version)
            .field("project_scope", &self.project_scope)
            .field(
                "workspace_collaboration_root",
                &self
                    .workspace_collaboration_root
                    .as_ref()
                    .map(|_| "[SCOPED WORKSPACE]"),
            )
            .field("trust_roots", &self.trust_roots)
            .field("trust_cwd", &self.trust_cwd)
            .field("model", &self.model)
            .field("provider_kind", &self.provider_kind)
            .field("model_chain", &self.model_chain)
            .field("extensions", &self.extensions)
            .finish()
    }
}

impl ToolCallContext {
    /// The session id, or `None` when the call carries none.
    pub fn session(&self) -> Option<&str> {
        Some(self.session_id.as_str()).filter(|id| !id.trim().is_empty())
    }
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

/// Permission persistence mode for commands requiring owner approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionMode {
    /// Safe/Medium/High persist; Critical remains session-scoped.
    #[default]
    Default,
    /// All approvals remain session-scoped.
    Cautious,
    /// All approvals persist, including Critical.
    Yolo,
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PermissionMode::Default => write!(f, "default"),
            PermissionMode::Cautious => write!(f, "cautious"),
            PermissionMode::Yolo => write!(f, "yolo"),
        }
    }
}

/// Distinguishes different kinds of approval requests so channels can render
/// appropriate buttons (e.g., Allow Once / Deny for commands vs Confirm / Cancel
/// for scheduled-goal confirmations).
#[derive(Debug, Clone, Copy, Default)]
pub enum ApprovalKind {
    /// Standard command approval (Allow Once / Allow Session / Allow Always / Deny).
    #[default]
    Command,
    /// Point-of-action approval that can authorize only this invocation.
    /// Channels must not offer session or persistent approval buttons.
    CommandOnce,
    /// Scheduled-goal confirmation (Confirm / Cancel).
    GoalConfirmation,
    /// Exact owner confirmation for a persisted Autopilot mandate policy.
    /// Kept distinct so channels never infer high-autonomy presentation from
    /// user or model prose.
    AutopilotConfirmation,
}

/// Response to an approval request from the user.
#[derive(Debug, Clone)]
pub enum ApprovalResponse {
    /// Allow this specific command only
    AllowOnce,
    /// Allow this command prefix for the current session (resets on restart)
    AllowSession,
    /// Allow this command prefix forever (persists to database)
    AllowAlways,
    /// Deny the command
    Deny,
}

/// What a tool says about a call without executing it: its capabilities,
/// typed semantics, access manifest, receipt protocol, effects and trust.
/// Every method has a conservative default; a tool overrides only what it
/// can state exactly.
#[async_trait]
pub trait ToolSemantics: Send + Sync {
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

    /// Adapter-owned, deterministic argument invariants evaluated by the
    /// common dispatcher before any tool I/O. The default accepts all schema-
    /// valid calls. Rejections become typed `contract_rejected` receipts.
    fn validate_arguments(&self, _arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        Ok(())
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

    /// How the dispatcher and the engine treat this tool. Default: nothing
    /// special.
    fn dispatch_facts(&self) -> ToolDispatchFacts {
        ToolDispatchFacts::default()
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

    /// The owner approval this exact call needs before it runs, judged from
    /// the call and the dispatcher's facts for it (never from `_` arguments).
    /// The kernel's approval rules ask it for every call and send the card
    /// it returns. The default has no opinion, so the approval policy's mode
    /// decides.
    async fn owner_approval_request(
        &self,
        _call: &ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        OwnerApprovalNeed::Unjudged
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
        default_semantics_from_identity(arguments, self.capabilities())
    }

    /// Filesystem access roles for this exact call. Open-ended adapters should
    /// override this when their schema exposes explicit read/write grants.
    fn call_access_manifest(&self, arguments: &str) -> ToolCallAccessManifest {
        let semantics = self.call_semantics(arguments);
        default_access_manifest(arguments, &semantics)
    }

    /// The paths this call reads and writes, as `ProtectedPathGuard` judges
    /// them. Defaults to the access manifest; a tool overrides it for a path
    /// that scope checks must not see as a task read.
    fn guarded_paths(&self, arguments: &str) -> (Vec<String>, Vec<String>) {
        let manifest = self.call_access_manifest(arguments);
        let values =
            |targets: Vec<ToolTargetHint>| targets.into_iter().map(|target| target.value).collect();
        (
            values(manifest.read_targets),
            values(manifest.write_targets),
        )
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

    /// Exact receipt subjects this adapter reports verbatim: fixed collection
    /// IDs and the namespaces of member identities it mints. The checklist
    /// tool publishes the union so declared targets bind only to identities a
    /// receipt can carry. The default advertises nothing.
    fn stable_observation_subjects(&self) -> Vec<StableObservationSubject> {
        Vec::new()
    }

    /// What this call carries out of the machine if it only reads.
    fn call_outbound(&self, _arguments: &str) -> aidaemon_core::Outbound {
        aidaemon_core::Outbound::Nothing
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

    /// Authority annotations for this exact call, judged against the turn's
    /// trusted roots: read-only, destructive, idempotent, and whether it
    /// sends data or acts outside the owner's machine and workspace
    /// (`egress`). Derived from the call's typed semantics, never from the
    /// tool's name. Roots matter for effects whose class depends on where
    /// they land (a delete inside the owner's roots is not egress); empty
    /// roots mean every such target counts as outside them. The default
    /// ignores the roots.
    fn call_annotations_in(
        &self,
        arguments: &str,
        _roots: &TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        default_call_annotations(self.capabilities(), &self.call_semantics(arguments))
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

/// Output trust of a file read: the owner's own files (inside the trusted
/// roots) are `Trusted`; anything else, or a call whose first present path
/// argument in `keys` is missing, is someone else's content.
pub fn file_output_trust(
    arguments: &str,
    keys: &[&str],
    roots: &TrustRoots<'_>,
) -> aidaemon_core::SourceTrust {
    let args = serde_json::from_str::<Value>(arguments).ok();
    let path = args.as_ref().and_then(|args| {
        keys.iter()
            .find_map(|key| args.get(*key).and_then(Value::as_str))
    });
    match path {
        Some(path) if roots.contains(path) => aidaemon_core::SourceTrust::Trusted,
        _ => aidaemon_core::SourceTrust::Untrusted,
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

/// Produce the single typed invocation object consumed by the execution
/// control plane. Validation deliberately precedes semantics and access
/// derivation so a rejected proposal can never acquire intended effects.
pub fn prepare_invocation<T: ToolSemantics + ?Sized>(
    tool: &T,
    arguments: &str,
) -> Result<PreparedToolInvocation, ToolArgumentContractViolation> {
    let canonical_arguments = tool.canonicalize_arguments(arguments)?;
    tool.validate_arguments(&canonical_arguments)?;
    Ok(PreparedToolInvocation {
        semantics: tool.call_semantics(&canonical_arguments),
        access_manifest: tool.call_access_manifest(&canonical_arguments),
        receipt_kind: tool.receipt_kind(&canonical_arguments),
        canonical_arguments,
    })
}

#[cfg(test)]
mod file_trust_tests {
    use super::{file_output_trust, TrustRoots};
    use aidaemon_core::SourceTrust::{Trusted, Untrusted};

    #[test]
    fn files_inside_the_trusted_roots_are_trusted() {
        let roots = vec!["/tmp/aidaemon-test/ws".to_string()];
        let roots = TrustRoots::new(&roots);
        let keys = ["path", "file_path"];
        assert_eq!(
            file_output_trust(r#"{"path":"/tmp/aidaemon-test/ws/a.txt"}"#, &keys, &roots),
            Trusted
        );
        assert_eq!(
            file_output_trust(
                r#"{"file_path":"/tmp/aidaemon-test/ws/a.txt"}"#,
                &keys,
                &roots
            ),
            Trusted
        );
        assert_eq!(
            file_output_trust(r#"{"path":"/etc/hosts"}"#, &keys, &roots),
            Untrusted
        );
        assert_eq!(file_output_trust(r#"{}"#, &keys, &roots), Untrusted);
        assert_eq!(file_output_trust("not json", &keys, &roots), Untrusted);
    }
}
