#[cfg(test)]
use async_trait::async_trait;
#[cfg(test)]
use serde_json::Value;

pub use aidaemon_tools::semantics::*;

pub use aidaemon_agents::specialist::SpecialistKind;

/// A tool: the SDK's [`aidaemon_tools::OutcomeTool`]. What it says about a
/// call without running it lives on the [`ToolSemantics`] supertrait.
pub use aidaemon_tools::OutcomeTool as Tool;

pub use aidaemon_tools::status::{status_sender, with_status, ToolStatus};
pub use aidaemon_tools::NothingSentError;

/// Calling a tool directly from a test.
#[cfg(test)]
pub use aidaemon_tools::testing::ToolCalls;

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
            TypeId::of::<crate::traits::ToolCallContext>(),
            TypeId::of::<aidaemon_tools::semantics::ToolCallContext>()
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

        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> = async move { Ok("ok".to_string()) }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    impl crate::traits::ToolSemantics for AlwaysAvailableTool {}

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

        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> = async move { Ok("ok".to_string()) }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    impl crate::traits::ToolSemantics for UnavailableTool {
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

        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> = async move { Ok("ok".to_string()) }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    impl crate::traits::ToolSemantics for ManageTool {}

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

        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> = async move { Ok("ok".to_string()) }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    impl crate::traits::ToolSemantics for RememberTool {}

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
