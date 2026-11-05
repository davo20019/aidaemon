use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::types::StatusUpdate;

pub use aidaemon_tools::semantics::*;

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

    #[test]
    fn semantics_types_have_one_definition() {
        use std::any::TypeId;
        // The daemon's paths are re-exports of the SDK's types, not copies.
        assert_eq!(
            TypeId::of::<crate::traits::ToolCallSemantics>(),
            TypeId::of::<aidaemon_tools::semantics::ToolCallSemantics>()
        );
        assert_eq!(
            TypeId::of::<crate::traits::ToolCallMetadata>(),
            TypeId::of::<aidaemon_tools::semantics::ToolCallMetadata>()
        );
        assert_eq!(
            TypeId::of::<crate::traits::ToolExecutionContext>(),
            TypeId::of::<aidaemon_tools::semantics::ToolExecutionContext>()
        );
        assert_eq!(
            TypeId::of::<crate::types::UserRole>(),
            TypeId::of::<aidaemon_tools::semantics::UserRole>()
        );
    }

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
