//! MCP servers and their tools: the registry lives in `aidaemon_mcp`
//! (`registry` feature).

pub use aidaemon_mcp::registry::McpRegistry;
#[allow(unused_imports)]
pub use aidaemon_mcp::registry::RegisteredMcpTool as McpTool;

#[cfg(test)]
mod annotation_tests {
    use super::*;
    use crate::traits::Tool as _;
    use serde_json::Value;

    fn mcp_tool(annotations: Value) -> McpTool {
        // The definition carries the server's hints, as `tools/list` does.
        McpTool::detached(
            "acme",
            serde_json::json!({
                "name": "synthetic_tool",
                "description": "",
                "inputSchema": {"type": "object"},
                "annotations": annotations,
            }),
        )
    }

    /// The model sees `{server}__{tool}` with the name the server gave,
    /// not the SDK's sanitized one, and the server's own description.
    #[test]
    fn wrapped_tools_keep_the_daemon_name_and_schema() {
        let tool = McpTool::detached(
            "acme",
            serde_json::json!({
                "name": "files.read",
                "title": "Reads files",
                "inputSchema": {"$schema": "x", "type": "object"},
            }),
        );
        assert_eq!(tool.name(), "acme__files.read");
        assert_eq!(
            tool.schema(),
            serde_json::json!({
                "name": "acme__files.read",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            })
        );
    }

    /// A mandate never delegates an MCP tool, whatever the server and tool
    /// are called: the tool declares opaque execution.
    #[test]
    fn mcp_tools_are_never_delegated_to_a_mandate() {
        let tool = McpTool::detached(
            "x",
            serde_json::json!({"name": "post", "inputSchema": {"type": "object"}}),
        );
        assert_eq!(tool.name(), "x__post");
        assert!(crate::mandates::is_non_delegable(&tool));
        assert!(!crate::mandates::is_non_delegable_tool("x__post"));
    }

    fn hook_ctx(untrusted_seen: bool) -> aidaemon_core::HookContext<'static> {
        aidaemon_core::HookContext {
            run_id: aidaemon_core::RunId::new(),
            agent_id: "aidaemon",
            session_id: "telegram:synthetic-user-1",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: aidaemon_core::Usage::default(),
            run_usage: aidaemon_core::Usage::default(),
            untrusted_seen,
            declined: &[],
        }
    }

    /// A server's hints are its own claims: a tool claiming to be read-only
    /// and closed-world is still egress, refused under read-only authority,
    /// checked by the untrusted verifier after untrusted content, opaque in
    /// its effects and untrusted in its output.
    #[tokio::test]
    async fn mcp_hints_never_loosen_safeguards() {
        use crate::traits::ToolSemantics;
        use aidaemon_core::Hook;
        let roots = crate::traits::TrustRoots::new(&[]);
        let claims_read_only =
            mcp_tool(serde_json::json!({"readOnlyHint": true, "openWorldHint": false}));
        let unhinted = mcp_tool(serde_json::json!({}));
        for tool in [&claims_read_only, &unhinted] {
            let annotations = tool.call_annotations_in("{}", &roots);
            assert!(annotations.egress, "{annotations:?}");
            assert!(!annotations.read_only, "{annotations:?}");
            let caps = tool.capabilities();
            assert!(!caps.read_only && caps.needs_approval && caps.external_side_effect);
            assert!(!caps.idempotent && caps.high_impact_write);
            assert_eq!(
                tool.call_effects("{}"),
                aidaemon_core::CallEffects::unknown()
            );
            assert_eq!(
                tool.output_trust("{}", &roots),
                aidaemon_core::SourceTrust::Untrusted
            );

            let info = aidaemon_core::ToolInfo {
                annotations,
                access: aidaemon_core::AccessManifest::default(),
                effects: tool.call_effects("{}"),
                outbound: Default::default(),
            };
            let call = aidaemon_core::ToolCall::new("c1", "synthetic_tool", serde_json::json!({}));
            let read_only =
                aidaemon_core::AuthorityHook::new(aidaemon_core::Authority::read_only());
            assert!(
                matches!(
                    read_only.before_tool(&hook_ctx(false), &call, &info).await,
                    aidaemon_core::ToolDecision::Deny { .. }
                ),
                "read-only authority must refuse it"
            );
            let untrusted = aidaemon_core::UntrustedHook::new(
                aidaemon_core::UntrustedMode::Enforce,
                |_, content| content.to_string(),
            );
            assert!(
                matches!(
                    untrusted.before_tool(&hook_ctx(true), &call, &info).await,
                    aidaemon_core::ToolDecision::Interrupt { .. }
                ),
                "after untrusted content it must go through the verifier"
            );
        }
    }

    fn hook_ctx_declined(
        untrusted_seen: bool,
        declined: &'static [aidaemon_core::EffectKey],
    ) -> aidaemon_core::HookContext<'static> {
        aidaemon_core::HookContext {
            declined,
            ..hook_ctx(untrusted_seen)
        }
    }

    fn tool_info(tool: &McpTool) -> aidaemon_core::ToolInfo {
        use crate::traits::ToolSemantics;
        let roots = crate::traits::TrustRoots::new(&[]);
        aidaemon_core::ToolInfo {
            annotations: tool.call_annotations_in("{}", &roots),
            access: aidaemon_core::AccessManifest::default(),
            effects: tool.call_effects("{}"),
            outbound: tool.call_outbound("{}"),
        }
    }

    /// A tool the owner's config lists read-only is a read of its server:
    /// read-only authority runs it and a decline elsewhere in the run asks
    /// nothing more. Its output is still untrusted, what it sends still goes
    /// to the untrusted verifier after an untrusted read, and it declares the
    /// server it reaches, so a declined server binds it.
    #[tokio::test]
    async fn owner_listed_read_only_tool_is_a_read_with_untrusted_output() {
        use crate::traits::ToolSemantics;
        use aidaemon_core::Hook;
        let roots = crate::traits::TrustRoots::new(&[]);
        let listed = mcp_tool(serde_json::json!({})).with_owner_read_only(true);
        assert!(listed.owner_read_only());
        assert_eq!(
            listed.call_annotations_in("{}", &roots),
            aidaemon_core::ToolAnnotations::READ_ONLY
        );
        let caps = listed.capabilities();
        assert!(caps.read_only && !caps.needs_approval && !caps.high_impact_write);
        assert_eq!(
            listed.output_trust("{}", &roots),
            aidaemon_core::SourceTrust::Untrusted
        );
        let server =
            aidaemon_core::EffectKey::named(aidaemon_core::EffectClass::Process, "mcp-server:acme");
        let effects = listed.call_effects("{}");
        assert!(effects.complete, "{effects:?}");
        assert_eq!(effects.keys, vec![server.clone()]);

        let call =
            aidaemon_core::ToolCall::new("c1", "acme__synthetic_tool", serde_json::json!({}));
        let info = tool_info(&listed);
        let read_only = aidaemon_core::AuthorityHook::new(aidaemon_core::Authority::read_only());
        assert_eq!(
            read_only.before_tool(&hook_ctx(false), &call, &info).await,
            aidaemon_core::ToolDecision::Allow
        );
        let untrusted = aidaemon_core::UntrustedHook::new(
            aidaemon_core::UntrustedMode::Enforce,
            |_, content| content.to_string(),
        );
        assert!(
            matches!(
                untrusted.before_tool(&hook_ctx(true), &call, &info).await,
                aidaemon_core::ToolDecision::Interrupt { .. }
            ),
            "its arguments reach the server: checked after untrusted content"
        );

        // After an unrelated decline the listed read runs; an unlisted tool
        // of the same server, even one whose server claims it is read-only,
        // is asked again.
        static DECLINED_POST: std::sync::LazyLock<Vec<aidaemon_core::EffectKey>> =
            std::sync::LazyLock::new(|| {
                vec![aidaemon_core::EffectKey::named(
                    aidaemon_core::EffectClass::Egress,
                    "h.example.invalid:443",
                )]
            });
        let policy = aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never);
        let ctx = hook_ctx_declined(false, DECLINED_POST.as_slice());
        assert_eq!(
            policy.before_tool(&ctx, &call, &info).await,
            aidaemon_core::ToolDecision::Allow
        );
        for unlisted in [
            mcp_tool(serde_json::json!({})),
            mcp_tool(serde_json::json!({"readOnlyHint": true, "openWorldHint": false})),
        ] {
            assert!(!unlisted.owner_read_only());
            assert!(matches!(
                policy.before_tool(&ctx, &call, &tool_info(&unlisted)).await,
                aidaemon_core::ToolDecision::Interrupt {
                    kind: aidaemon_core::InterruptKind::OwnerApproval(_),
                    ..
                }
            ));
        }
        // A declined start of the server overlaps the listed read.
        assert!(effects.keys.iter().any(|key| key.overlaps(&server)));
    }
}
