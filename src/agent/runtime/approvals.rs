//! Owner approvals live in `aidaemon-runtime`; the old paths stay.

#[cfg(test)]
pub(crate) use aidaemon_runtime::approvals::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use async_trait::async_trait;

    use crate::tools::command_risk::PermissionMode;
    use crate::types::ApprovalResponse;
    use aidaemon_core as sdk;
    use aidaemon_policies::{ApprovalRules, RuleDecision};
    use serde_json::json;

    fn hook_ctx() -> sdk::HookContext<'static> {
        sdk::HookContext {
            run_id: sdk::RunId::new(),
            agent_id: "aidaemon",
            session_id: "telegram:synthetic-user-1",
            turn: 1,
            messages: &[],
            receipts: &[],
            usage: sdk::Usage::default(),
            run_usage: sdk::Usage::default(),
            untrusted_seen: false,
            declined: &[],
        }
    }

    fn tool_info() -> sdk::ToolInfo {
        sdk::ToolInfo {
            annotations: sdk::ToolAnnotations::default(),
            access: sdk::AccessManifest::default(),
            effects: sdk::CallEffects::unknown(),
            outbound: Default::default(),
        }
    }

    /// `manage_api`'s approval judgment without its stores (which need a
    /// database): the same `ManageApiTool::owner_approval` the tool runs.
    struct ManageApiApprovals;

    #[async_trait]
    impl crate::traits::ToolSemantics for ManageApiApprovals {
        async fn owner_approval_request(
            &self,
            call: &crate::traits::ApprovalCall<'_>,
            facts: &crate::traits::ToolCallContext,
        ) -> crate::traits::OwnerApprovalNeed {
            crate::tools::ManageApiTool::owner_approval(call.arguments, facts)
        }
    }

    #[async_trait]
    impl crate::traits::Tool for ManageApiApprovals {
        fn name(&self) -> &str {
            "manage_api"
        }
        fn description(&self) -> &str {
            "manage_api approvals"
        }
        fn schema(&self) -> serde_json::Value {
            json!({"name": "manage_api", "description": "", "parameters": {"type": "object"}})
        }
        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> =
                async move { anyhow::bail!("not run in these tests") }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    /// `manage_mcp`'s approval judgment without its registry (which needs a
    /// database): the same `ManageMcpTool::owner_approval` the tool runs.
    struct ManageMcpApprovals;

    #[async_trait]
    impl crate::traits::ToolSemantics for ManageMcpApprovals {
        async fn owner_approval_request(
            &self,
            call: &crate::traits::ApprovalCall<'_>,
            _facts: &crate::traits::ToolCallContext,
        ) -> crate::traits::OwnerApprovalNeed {
            crate::tools::ManageMcpTool::owner_approval(call.arguments)
        }
    }

    #[async_trait]
    impl crate::traits::Tool for ManageMcpApprovals {
        fn name(&self) -> &str {
            "manage_mcp"
        }
        fn description(&self) -> &str {
            "manage_mcp approvals"
        }
        fn schema(&self) -> serde_json::Value {
            json!({"name": "manage_mcp", "description": "", "parameters": {"type": "object"}})
        }
        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> =
                async move { anyhow::bail!("not run in these tests") }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    /// Rules over the turn's approval-judging tools: a real `http_request`,
    /// `manage_api`'s judgment and `manage_mcp`'s judgment.
    fn rules() -> AppApprovalRules {
        let http: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::HttpRequestTool::new(
            Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        ));
        AppApprovalRules {
            tools: ApprovalTools::new([
                http,
                Arc::new(ManageApiApprovals) as Arc<_>,
                Arc::new(ManageMcpApprovals) as Arc<_>,
            ]),
            ..AppApprovalRules::default()
        }
    }

    async fn judge(rules: &AppApprovalRules, args: serde_json::Value) -> RuleDecision {
        let call = sdk::ToolCall::new("c1", "http_request", args);
        rules.judge(&hook_ctx(), &call, &tool_info()).await
    }

    /// An MCP start asks the owner in every session, and a call the add path
    /// refuses needs no card (it never runs).
    #[tokio::test]
    async fn mcp_add_always_asks_before_dispatch() {
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..rules()
            };
            let add = sdk::ToolCall::new(
                "c1",
                "manage_mcp",
                json!({"action":"add","name":"synthetic","command":"node","args":["server.js"]}),
            );
            assert!(
                matches!(
                    rules.judge(&hook_ctx(), &add, &tool_info()).await,
                    RuleDecision::Ask(_)
                ),
                "trusted={trusted}"
            );
            let refused = sdk::ToolCall::new(
                "c2",
                "manage_mcp",
                json!({"action":"add","name":"synthetic","command":"sh"}),
            );
            assert_eq!(
                rules.judge(&hook_ctx(), &refused, &tool_info()).await,
                RuleDecision::Allow
            );
            let list = sdk::ToolCall::new("c3", "manage_mcp", json!({"action":"list"}));
            assert_eq!(
                rules.judge(&hook_ctx(), &list, &tool_info()).await,
                RuleDecision::Allow
            );
        }
    }

    /// Exercise the real app rules and SDK policy with a process-effect
    /// probe. The alternate name proves declines follow effects, not names.
    #[tokio::test]
    async fn mcp_approval_binds_process_effects_across_tools() {
        use sdk::testing::{ScriptedProvider, call, spec, text, tool_calls};
        struct ProcessProbe(&'static str, Arc<AtomicU64>);
        #[async_trait]
        impl sdk::Tool for ProcessProbe {
            fn spec(&self) -> sdk::ToolSpec {
                spec(self.0, "synthetic process probe", &[])
            }
            fn annotations(&self) -> sdk::ToolAnnotations {
                sdk::ToolAnnotations::WRITE
            }
            fn effects(&self, args: &serde_json::Value) -> sdk::CallEffects {
                crate::tools::ManageMcpTool::effects_for(&args.to_string())
            }
            async fn call(&self, _: serde_json::Value, _: &sdk::ToolContext) -> sdk::ToolResult {
                self.1.fetch_add(1, Ordering::SeqCst);
                Ok(sdk::ToolOutput::text("started"))
            }
        }
        for approve in [false, true] {
            let executed = Arc::new(AtomicU64::new(0));
            let args =
                json!({"action":"add","name":"synthetic","command":"node","args":["server.js"]});
            let policy =
                aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never)
                    .with_rules(Arc::new(rules()));
            let agent = sdk::Agent::builder("synthetic")
                .allow_unguarded()
                .model("mock")
                .provider(ScriptedProvider::new([
                    tool_calls([call("c1", "manage_mcp", args.clone())]),
                    tool_calls([call("c2", "alternate_process", args)]),
                    text("Finished considering the request."),
                ]))
                .tool(ProcessProbe("manage_mcp", executed.clone()))
                .tool(ProcessProbe("alternate_process", executed.clone()))
                .hook(policy)
                .build()
                .unwrap();
            let pending = agent
                .run("synthetic-session", "Add the server")
                .await
                .unwrap();
            assert!(matches!(pending.status, sdk::RunStatus::Interrupted { .. }));
            assert_eq!(executed.load(Ordering::SeqCst), 0);
            let resolution = if approve {
                sdk::Resolution::Approve
            } else {
                sdk::Resolution::Deny {
                    reason: "No".into(),
                }
            };
            let finished = agent.resume(pending.state, resolution).await.unwrap();
            assert!(matches!(finished.status, sdk::RunStatus::Completed));
            assert_eq!(executed.load(Ordering::SeqCst), if approve { 2 } else { 0 });
            if !approve {
                assert_eq!(
                    finished.state.declined,
                    vec![
                        sdk::EffectKey::named(sdk::EffectClass::Process, "node"),
                        sdk::EffectKey::named(sdk::EffectClass::Process, "mcp-server:synthetic"),
                    ]
                );
                assert!(
                    finished
                        .receipts()
                        .iter()
                        .all(|r| r.status == sdk::ReceiptStatus::Denied)
                );
            }
        }
    }

    #[tokio::test]
    async fn observation_methods_without_body_are_allowed() {
        for method in ["GET", "HEAD", "OPTIONS", "get"] {
            let decision = judge(
                &rules(),
                json!({"method": method, "url": "https://h.example.invalid/x"}),
            )
            .await;
            assert_eq!(decision, RuleDecision::Allow, "{method}");
        }
    }

    #[tokio::test]
    async fn writes_and_bodies_ask_the_owner() {
        let body = "b".repeat(700);
        for (method, body) in [
            ("POST", Some(body.as_str())),
            ("DELETE", None),
            ("GET", Some("q=1")),
        ] {
            let mut args = json!({"method": method, "url": "https://h.example.invalid/x"});
            if let Some(body) = body {
                args["body"] = json!(body);
            }
            let RuleDecision::Ask(request) = judge(&rules(), args).await else {
                panic!("{method} must ask");
            };
            assert_eq!(request.risk, sdk::ApprovalRisk::High);
            assert_eq!(
                request.subject,
                format!("{method} https://h.example.invalid/x")
            );
            match body {
                Some(body) => {
                    let shown: String = body.chars().take(500).collect();
                    assert_eq!(
                        request.summary,
                        format!("{method} https://h.example.invalid/x\nBody: {shown}")
                    );
                }
                None => assert_eq!(request.summary, request.subject),
            }
        }
    }

    #[tokio::test]
    async fn method_override_header_asks() {
        let decision = judge(
            &rules(),
            json!({
                "method": "GET",
                "url": "https://h.example.invalid/x",
                "headers": {"X-HTTP-Method-Override": "DELETE"},
            }),
        )
        .await;
        assert!(matches!(decision, RuleDecision::Ask(_)), "{decision:?}");
    }

    #[tokio::test]
    async fn trusted_session_and_mandate_preapproval_allow() {
        let post = json!({"method": "POST", "url": "https://h.example.invalid/x", "body": "x"});
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..rules()
        };
        assert_eq!(judge(&trusted, post.clone()).await, RuleDecision::Allow);
        let mandate = AppApprovalRules {
            mandate_preapproved: true,
            ..rules()
        };
        assert_eq!(judge(&mandate, post).await, RuleDecision::Allow);
    }

    #[tokio::test]
    async fn other_tools_defer() {
        let call = sdk::ToolCall::new("c1", "write_file", json!({"path": "/tmp/aidaemon-test/x"}));
        let decision = rules().judge(&hook_ctx(), &call, &tool_info()).await;
        assert_eq!(decision, RuleDecision::Defer);
    }

    /// Widening who may see a stored memory asks the owner in every
    /// session; narrowing it, a call naming no active fact and a call the
    /// dispatcher refuses need no card.
    #[tokio::test]
    async fn set_privacy_asks_before_widening_a_memory() {
        use crate::traits::ToolSemantics;
        use aidaemon_memory::stores::FactStore as _;
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        aidaemon_memory::sqlite::migrate(&pool).await.unwrap();
        let facts = Arc::new(aidaemon_memory::sqlite::SqliteMemory::new(
            pool,
            Arc::new(aidaemon_memory::HashEmbedder::new(384)),
        ));
        facts
            .upsert_fact(
                "preference",
                "report_format",
                "short summaries for Jordan Lee",
                "owner",
                Some("telegram:synthetic-channel-1"),
                crate::types::FactPrivacy::Channel,
            )
            .await
            .unwrap();
        let tool = Arc::new(aidaemon_memory::full::tools::ManageMemoriesTool::new(facts));
        let set = |key: &str, privacy: &str| json!({"action": "set_privacy", "category": "preference", "key": key, "privacy": privacy});
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                tools: ApprovalTools::new([tool.clone() as Arc<dyn crate::traits::Tool>]),
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..AppApprovalRules::default()
            };
            let widen = sdk::ToolCall::new("c1", "manage_memories", set("report_format", "global"));
            let RuleDecision::Ask(request) = rules.judge(&hook_ctx(), &widen, &tool_info()).await
            else {
                panic!("widening must ask (trusted={trusted})");
            };
            assert_eq!(request.risk, sdk::ApprovalRisk::High);
            for (id, args) in [
                ("c2", set("absent", "global")),
                ("c3", set("report_format", "private")),
            ] {
                let call = sdk::ToolCall::new(id, "manage_memories", args);
                assert_eq!(
                    rules.judge(&hook_ctx(), &call, &tool_info()).await,
                    RuleDecision::Allow,
                    "{id}"
                );
            }
        }
        let args = set("report_format", "global");
        let refused = crate::traits::ApprovalCall {
            refused: true,
            ..crate::traits::ApprovalCall::as_sent(&args)
        };
        assert!(matches!(
            tool.owner_approval_request(&refused, &Default::default())
                .await,
            crate::traits::OwnerApprovalNeed::NotNeeded
        ));
    }

    /// A security-sensitive config change asks the owner in every session;
    /// a read and an ordinary setting need no card.
    #[tokio::test]
    async fn sensitive_config_change_asks_before_dispatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[provider]\nkind = \"openai_compatible\"\n").unwrap();
        let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::ConfigManagerTool::new(
            crate::config::config_file(path),
            Arc::new(aidaemon_secrets::MemorySecretStore::new()),
        ));
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                tools: ApprovalTools::new([tool.clone()]),
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..AppApprovalRules::default()
            };
            let set = sdk::ToolCall::new(
                "c1",
                "manage_config",
                json!({"action": "set", "key": "provider.base_url", "value": "\"https://h.example.invalid/v1\""}),
            );
            let RuleDecision::Ask(request) = rules.judge(&hook_ctx(), &set, &tool_info()).await
            else {
                panic!("a sensitive change must ask (trusted={trusted})");
            };
            assert_eq!(
                request.summary,
                "Config change: provider.base_url = \"https://h.example.invalid/v1\""
            );
            for no_card in [
                json!({"action": "read"}),
                json!({"action": "set", "key": "provider.kind", "value": "\"openai_compatible\""}),
            ] {
                let call = sdk::ToolCall::new("c2", "manage_config", no_card);
                assert_eq!(
                    rules.judge(&hook_ctx(), &call, &tool_info()).await,
                    RuleDecision::Allow
                );
            }
        }
        let args = json!({"action": "set", "key": "provider.base_url", "value": "\"x\""});
        let refused = crate::traits::ApprovalCall {
            refused: true,
            ..crate::traits::ApprovalCall::as_sent(&args)
        };
        assert!(matches!(
            tool.owner_approval_request(&refused, &Default::default())
                .await,
            crate::traits::OwnerApprovalNeed::NotNeeded
        ));
    }

    /// `manage_cli_agents`'s approval judgment without its CLI registry: the
    /// same `ManageCliAgentsTool::owner_approval` the tool runs.
    struct ManageCliAgentsApprovals;

    #[async_trait]
    impl crate::traits::ToolSemantics for ManageCliAgentsApprovals {
        async fn owner_approval_request(
            &self,
            call: &crate::traits::ApprovalCall<'_>,
            _facts: &crate::traits::ToolCallContext,
        ) -> crate::traits::OwnerApprovalNeed {
            if call.refused {
                return crate::traits::OwnerApprovalNeed::NotNeeded;
            }
            crate::tools::ManageCliAgentsTool::owner_approval(call.arguments)
        }
    }

    #[async_trait]
    impl crate::traits::Tool for ManageCliAgentsApprovals {
        fn name(&self) -> &str {
            "manage_cli_agents"
        }
        fn description(&self) -> &str {
            "manage_cli_agents approvals"
        }
        fn schema(&self) -> serde_json::Value {
            json!({"name": "manage_cli_agents", "description": "", "parameters": {"type": "object"}})
        }
        async fn execute(
            &self,
            _arguments: &str,
            _facts: &crate::traits::ToolCallContext,
        ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
            let output: anyhow::Result<String> =
                async move { anyhow::bail!("not run in these tests") }.await;
            output.map(crate::traits::ToolCallOutcome::from_output)
        }
    }

    /// Registering a CLI agent asks the owner in every session; listing,
    /// history and a call the add path refuses need no card.
    #[tokio::test]
    async fn cli_agent_add_always_asks_before_dispatch() {
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                tools: ApprovalTools::new([
                    Arc::new(ManageCliAgentsApprovals) as Arc<dyn crate::traits::Tool>
                ]),
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..AppApprovalRules::default()
            };
            let add = sdk::ToolCall::new(
                "c1",
                "manage_cli_agents",
                json!({"action":"add","name":"synthetic","command":"synthetic-cli"}),
            );
            assert!(
                matches!(
                    rules.judge(&hook_ctx(), &add, &tool_info()).await,
                    RuleDecision::Ask(_)
                ),
                "trusted={trusted}"
            );
            for no_card in [
                json!({"action":"add","name":"synthetic"}),
                json!({"action":"list"}),
                json!({"action":"history","limit":5}),
            ] {
                let call = sdk::ToolCall::new("c2", "manage_cli_agents", no_card);
                assert_eq!(
                    rules.judge(&hook_ctx(), &call, &tool_info()).await,
                    RuleDecision::Allow
                );
            }
        }
    }

    #[test]
    fn manage_api_derived_probe_asks_at_the_outer_call() {
        let onboard = |extra: serde_json::Value| {
            let mut args = json!({
                "action": "onboard",
                "service": "synthetic-api",
                "learn_url": "https://docs.example.invalid/graphql",
            });
            for (key, value) in extra.as_object().unwrap() {
                args[key] = value.clone();
            }
            crate::tools::ManageApiTool::probe_approval_request(&args)
        };
        let request = onboard(json!({})).expect("a derived probe may POST");
        assert_eq!(
            request.subject,
            "manage_api probe POST https://docs.example.invalid/graphql"
        );
        assert_eq!(request.risk, sdk::ApprovalRisk::High);
        // A verify_url probe is GET or HEAD; an OpenAPI source probes with
        // GET or HEAD only.
        assert!(onboard(json!({"verify_url": "https://api.example.invalid/me"})).is_none());
        assert!(
            onboard(json!({"openapi_url": "https://api.example.invalid/openapi.json"})).is_none()
        );
    }

    #[tokio::test]
    async fn manage_api_rules_ask_unless_trusted() {
        let args = json!({
            "action": "onboard",
            "service": "synthetic-api",
            "learn_url": "https://docs.example.invalid/graphql",
        });
        let call = sdk::ToolCall::new("c1", "manage_api", args);
        let asked = rules().judge(&hook_ctx(), &call, &tool_info()).await;
        assert!(matches!(asked, RuleDecision::Ask(_)), "{asked:?}");
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..rules()
        };
        assert_eq!(
            trusted.judge(&hook_ctx(), &call, &tool_info()).await,
            RuleDecision::Defer
        );
    }

    /// An OAuth store holding at most one connection, for the judgments of
    /// the auth tools (which read it before asking).
    #[derive(Default)]
    struct SyntheticOAuthStore(Option<crate::traits::OAuthConnection>);

    #[async_trait]
    impl crate::traits::OAuthStore for SyntheticOAuthStore {
        async fn get_oauth_connection(
            &self,
            service: &str,
        ) -> anyhow::Result<Option<crate::traits::OAuthConnection>> {
            Ok(self.0.clone().filter(|conn| conn.service == service))
        }

        async fn list_oauth_connections(
            &self,
        ) -> anyhow::Result<Vec<crate::traits::OAuthConnection>> {
            Ok(self.0.clone().into_iter().collect())
        }
    }

    /// A config file with one synthetic manual auth profile.
    fn synthetic_auth_config(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[provider]\napi_key = \"synthetic-key\"\n\n[http_auth.synthetic]\n\
             auth_type = \"bearer\"\nallowed_domains = [\"api.example.invalid\"]\n",
        )
        .unwrap();
        path
    }

    /// Rules over exactly `tools`, in a trusted (scheduled, mandate) session
    /// or not.
    fn rules_over(tools: Vec<Arc<dyn crate::traits::Tool>>, trusted: bool) -> AppApprovalRules {
        AppApprovalRules {
            tools: ApprovalTools::new(tools),
            trusted_session: trusted,
            mandate_preapproved: trusted,
            ..AppApprovalRules::default()
        }
    }

    async fn judge_call(
        rules: &AppApprovalRules,
        tool: &str,
        args: serde_json::Value,
    ) -> RuleDecision {
        let call = sdk::ToolCall::new("c1", tool, args);
        rules.judge(&hook_ctx(), &call, &tool_info()).await
    }

    /// A profile change asks the owner in every session, with the card the
    /// tool used to send itself; reads and calls that stop before writing
    /// need no card, and neither does a call the dispatcher refuses.
    #[tokio::test]
    async fn http_auth_changes_always_ask_before_dispatch() {
        use crate::traits::ToolSemantics;
        let dir = tempfile::TempDir::new().unwrap();
        let concrete = Arc::new(crate::tools::ManageHttpAuthTool::new(
            crate::config::config_file(synthetic_auth_config(&dir)),
            Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            Arc::new(SyntheticOAuthStore::default()),
            crate::config::shared_secret_store(),
        ));
        let tool: Arc<dyn crate::traits::Tool> = concrete.clone();
        for trusted in [false, true] {
            let rules = rules_over(vec![tool.clone()], trusted);
            let upsert =
                json!({"action":"upsert","profile":"synthetic","user_id":"synthetic-account-2"});
            let RuleDecision::Ask(update) =
                judge_call(&rules, "manage_http_auth", upsert.clone()).await
            else {
                panic!("an upsert must ask (trusted={trusted})");
            };
            assert_eq!(
                update.summary,
                "Update manual API auth profile 'synthetic' (type=bearer, \
                 domains=api.example.invalid, user_id=synthetic-account-2)"
            );
            assert_eq!(update.risk, sdk::ApprovalRisk::High);
            assert_eq!(update.warnings.len(), 3);
            // A changed argument is a new card.
            let RuleDecision::Ask(other) = judge_call(
                &rules,
                "manage_http_auth",
                json!({"action":"upsert","profile":"synthetic","user_id":"synthetic-account-3"}),
            )
            .await
            else {
                panic!("an upsert must ask");
            };
            assert_ne!(update.subject, other.subject);
            let RuleDecision::Ask(remove) = judge_call(
                &rules,
                "manage_http_auth",
                json!({"action":"remove","profile":"synthetic"}),
            )
            .await
            else {
                panic!("a remove must ask (trusted={trusted})");
            };
            assert_eq!(
                remove.summary,
                "Remove manual API auth profile 'synthetic' and delete its stored credentials"
            );
            for args in [
                json!({"action":"list"}),
                json!({"action":"describe","profile":"synthetic"}),
                json!({"action":"verify","profile":"synthetic"}),
                // Nothing to remove; a new profile without its type.
                json!({"action":"remove","profile":"absent"}),
                json!({"action":"upsert","profile":"absent"}),
            ] {
                assert_eq!(
                    judge_call(&rules, "manage_http_auth", args.clone()).await,
                    RuleDecision::Allow,
                    "{args}"
                );
            }
            let refused = crate::traits::ApprovalCall {
                arguments: &upsert,
                as_sent: &upsert,
                refused: true,
            };
            assert_eq!(
                concrete
                    .owner_approval_request(&refused, &crate::traits::ToolCallContext::default())
                    .await,
                crate::traits::OwnerApprovalNeed::NotNeeded
            );
        }
    }

    /// Disconnecting a service and registering or removing a custom provider
    /// ask the owner in every session, with the card the tool used to send
    /// itself and never a client secret; reads, actions that never asked and
    /// calls that stop before changing anything need no card, and neither
    /// does a call the dispatcher refuses.
    #[tokio::test]
    async fn oauth_changes_always_ask_before_dispatch() {
        use crate::traits::ToolSemantics;
        let dir = tempfile::TempDir::new().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            "[provider]\napi_key = \"synthetic-key\"\n\n[oauth.providers.synthetic-custom]\n\
             auth_type = \"oauth2_pkce\"\nauthorize_url = \"https://auth.example.invalid/authorize\"\n\
             token_url = \"https://auth.example.invalid/token\"\nscopes = []\n\
             allowed_domains = [\"api.example.invalid\"]\n",
        )
        .unwrap();
        let store = Arc::new(SyntheticOAuthStore(Some(crate::traits::OAuthConnection {
            id: 1,
            service: "synthetic".to_string(),
            auth_type: "oauth2_pkce".to_string(),
            username: None,
            account_id: None,
            scopes: "[]".to_string(),
            token_expires_at: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        })));
        let gateway = crate::startup::oauth::daemon_gateway(
            store.clone() as Arc<dyn crate::traits::OAuthGatewayStore>,
            Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            "http://localhost:8080".to_string(),
        );
        let concrete = Arc::new(crate::tools::ManageOAuthTool::new(
            gateway.clone(),
            store.clone() as Arc<dyn crate::traits::OAuthStore>,
            crate::config::config_file(config.clone()),
        ));
        // `manage_auth` routes each kind to its tool, with that tool's card.
        let auth_dir = tempfile::TempDir::new().unwrap();
        let manage_auth: Arc<dyn crate::traits::Tool> =
            Arc::new(aidaemon_tools::services::manage_auth::ManageAuthTool::new(
                crate::tools::ManageHttpAuthTool::new(
                    crate::config::config_file(synthetic_auth_config(&auth_dir)),
                    Arc::new(tokio::sync::RwLock::new(HashMap::new())),
                    Arc::new(SyntheticOAuthStore::default()),
                    crate::config::shared_secret_store(),
                ),
                crate::tools::ManageOAuthTool::new(
                    gateway,
                    store as Arc<dyn crate::traits::OAuthStore>,
                    crate::config::config_file(config),
                ),
            ));
        let routed = rules_over(vec![manage_auth], false);
        let RuleDecision::Ask(disconnect) = judge_call(
            &routed,
            "manage_auth",
            json!({"kind":"oauth","action":"remove","service":"synthetic","confirm_disconnect":true}),
        )
        .await
        else {
            panic!("a disconnect through manage_auth must ask");
        };
        assert_eq!(disconnect.summary, "Disconnect OAuth service 'synthetic'");
        let RuleDecision::Ask(remove_profile) = judge_call(
            &routed,
            "manage_auth",
            json!({"kind":"profile","action":"remove","profile":"synthetic"}),
        )
        .await
        else {
            panic!("a profile removal through manage_auth must ask");
        };
        assert_eq!(
            remove_profile.summary,
            "Remove manual API auth profile 'synthetic' and delete its stored credentials"
        );
        for args in [
            json!({"kind":"oauth","action":"list"}),
            json!({"kind":"profile","action":"list"}),
            json!({"action":"remove","profile":"synthetic"}),
        ] {
            assert_eq!(
                judge_call(&routed, "manage_auth", args.clone()).await,
                RuleDecision::Allow,
                "{args}"
            );
        }
        let tool: Arc<dyn crate::traits::Tool> = concrete.clone();
        for trusted in [false, true] {
            let rules = rules_over(vec![tool.clone()], trusted);
            let RuleDecision::Ask(remove) = judge_call(
                &rules,
                "manage_oauth",
                json!({"action":"remove","service":"synthetic","confirm_disconnect":true}),
            )
            .await
            else {
                panic!("a disconnect must ask (trusted={trusted})");
            };
            assert_eq!(remove.summary, "Disconnect OAuth service 'synthetic'");
            assert_eq!(remove.risk, sdk::ApprovalRisk::High);
            let register = json!({
                "action": "register_provider",
                "service": "synthetic-other",
                "authorize_url": "https://auth.example.invalid/authorize",
                "token_url": "https://auth.example.invalid/token",
                "allowed_domains": ["api.example.invalid"],
                "client_secret": "synthetic-client-secret",
            });
            let RuleDecision::Ask(registered) =
                judge_call(&rules, "manage_oauth", register.clone()).await
            else {
                panic!("a registration must ask (trusted={trusted})");
            };
            assert!(
                registered
                    .summary
                    .starts_with("Register custom OAuth provider 'synthetic-other' (authorize=")
            );
            for text in [&registered.summary, &registered.subject] {
                assert!(!text.contains("synthetic-client-secret"), "{text}");
            }
            let RuleDecision::Ask(removed) = judge_call(
                &rules,
                "manage_oauth",
                json!({"action":"remove_provider","service":"synthetic-custom"}),
            )
            .await
            else {
                panic!("a provider removal must ask (trusted={trusted})");
            };
            assert_eq!(
                removed.summary,
                "Remove custom OAuth provider 'synthetic-custom' and delete its client credentials"
            );
            for args in [
                json!({"action":"list"}),
                json!({"action":"providers"}),
                json!({"action":"connect","service":"synthetic"}),
                json!({"action":"refresh","service":"synthetic"}),
                json!({"action":"set_credentials","service":"synthetic","client_id":"synthetic-id","client_secret":"synthetic-client-secret"}),
                // No confirmation; nothing connected; still connected; unknown.
                json!({"action":"remove","service":"synthetic"}),
                json!({"action":"remove","service":"absent","confirm_disconnect":true}),
                json!({"action":"remove_provider","service":"synthetic"}),
                json!({"action":"remove_provider","service":"absent"}),
                // Missing endpoints.
                json!({"action":"register_provider","service":"synthetic-other"}),
            ] {
                assert_eq!(
                    judge_call(&rules, "manage_oauth", args.clone()).await,
                    RuleDecision::Allow,
                    "{args}"
                );
            }
            let refused = crate::traits::ApprovalCall {
                arguments: &register,
                as_sent: &register,
                refused: true,
            };
            assert_eq!(
                concrete
                    .owner_approval_request(&refused, &crate::traits::ToolCallContext::default())
                    .await,
                crate::traits::OwnerApprovalNeed::NotNeeded
            );
        }
    }

    /// Rules over a real terminal with `allowed` prefixes and no semantic
    /// assessor (an unknown command then fails closed to a Critical ask).
    async fn terminal_rules(allowed: &[&str]) -> AppApprovalRules {
        let pool_file = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", pool_file.path().display()))
            .await
            .unwrap();
        std::mem::forget(pool_file);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let terminal: Arc<dyn crate::traits::Tool> = Arc::new(
            crate::tools::TerminalTool::new(
                crate::execution::active_execution_backend(),
                allowed.iter().map(|p| p.to_string()).collect(),
                crate::tools::ApprovalBroker::new(tx),
                1,
                1000,
                crate::types::PermissionMode::Default,
                pool,
            )
            .await,
        );
        AppApprovalRules {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_role: Some(crate::types::UserRole::Owner),
            terminal: TerminalApprovals::find(&[terminal]),
            ..rules()
        }
    }

    /// The terminal with every call judged to have `scope`.
    fn with_scope(terminal: TerminalApprovals, scope: TerminalScope) -> TerminalApprovals {
        terminal.with_canonicalizer(
            move |_, _, raw| JudgedTerminal {
                call: CanonicalCall::stripped(None, raw),
                scope: scope.clone(),
            },
            JudgedCalls::default(),
        )
    }

    async fn judge_terminal(rules: &AppApprovalRules, args: serde_json::Value) -> RuleDecision {
        let call = sdk::ToolCall::new("c1", "terminal", args);
        rules.judge(&hook_ctx(), &call, &tool_info()).await
    }

    fn run(command: &str) -> serde_json::Value {
        json!({"action": "run", "command": command})
    }

    fn asks_with(decision: &RuleDecision, warning: &str) -> bool {
        matches!(decision, RuleDecision::Ask(request)
            if request.warnings.iter().any(|w| w.contains(warning)))
    }

    /// The T8 ladder, in order: an untrusted trigger asks before the
    /// trusted-session and explicit-allow rungs; a trusted session and an
    /// explicit allow both come before the approval floor; the confined
    /// tier comes after it; an unknown command with no assessor fails
    /// closed, Critical.
    #[tokio::test]
    async fn terminal_ladder_keeps_its_order() {
        let base = terminal_rules(&["echo", "cat"]).await;
        let trusted = AppApprovalRules {
            trusted_session: true,
            ..base.clone()
        };
        let trigger = AppApprovalRules {
            untrusted_trigger_session: true,
            ..trusted.clone()
        };
        // Explicit allow.
        assert_eq!(
            judge_terminal(&base, run("echo hi")).await,
            RuleDecision::Allow
        );
        // Untrusted trigger beats both the trusted session and the allowlist.
        let asked = judge_terminal(&trigger, run("echo hi")).await;
        assert!(asks_with(&asked, "untrusted external trigger"), "{asked:?}");
        // A trusted session runs an unknown command without asking.
        assert_eq!(
            judge_terminal(&trusted, run("printf synthetic")).await,
            RuleDecision::Allow
        );
        // The approval floor precedes the explicit allow (owner ruling,
        // closing the T10 hole): an allowlisted command touching a
        // credential path asks; a plain allowlisted command does not.
        for sensitive in ["cat ~/.ssh/config", "cat ~/.ssh/id_rsa"] {
            let asked = judge_terminal(&base, run(sensitive)).await;
            assert!(
                matches!(asked, RuleDecision::Ask(_)),
                "{sensitive}: {asked:?}"
            );
        }
        let plain = terminal_rules(&["ls"]).await;
        assert_eq!(judge_terminal(&plain, run("ls")).await, RuleDecision::Allow);
        // The floor asks for a credential path that no grant covers.
        let floor = judge_terminal(&base, run("head ~/.ssh/config")).await;
        assert!(matches!(floor, RuleDecision::Ask(_)), "{floor:?}");
        // The confined tier: an allowlisted inspection or build command
        // runs confined without asking (the floor above still precedes it:
        // `head` is in the tier).
        for confined in ["printf synthetic", "git status", "cargo check"] {
            assert_eq!(
                judge_terminal(&base, run(confined)).await,
                RuleDecision::Allow,
                "{confined}"
            );
        }
        // Unknown command, no assessor: Critical, failed closed.
        let unknown = judge_terminal(&base, run("node --version")).await;
        let RuleDecision::Ask(request) = &unknown else {
            panic!("{unknown:?}");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert_eq!(request.subject, "node --version");
        assert!(asks_with(&unknown, "not configured"), "{unknown:?}");
        // A novel detached command asks; an allowlisted one does not.
        let detached = json!({"action": "run", "command": "printf x", "detach": true});
        assert!(matches!(
            judge_terminal(&base, detached).await,
            RuleDecision::Ask(_)
        ));
        let allowed_detached = json!({"action": "run", "command": "echo x", "detach": true});
        assert_eq!(
            judge_terminal(&base, allowed_detached).await,
            RuleDecision::Allow
        );
    }

    /// Daemonization: the owner gets a Critical card; a non-owner's call or
    /// one without `detach=true` is refused by the tool's hard block, so the
    /// rules never ask about it.
    #[tokio::test]
    async fn terminal_daemonization_asks_the_owner_critical() {
        let owner = terminal_rules(&["*"]).await;
        let detached = json!({"action": "run", "command": "nohup sleep 1 &", "detach": true});
        let RuleDecision::Ask(request) = judge_terminal(&owner, detached.clone()).await else {
            panic!("the owner is asked even under trust-all");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert!(
            request
                .warnings
                .iter()
                .any(|w| w.contains("Daemonization primitives detected"))
        );
        let guest = AppApprovalRules {
            user_role: Some(crate::types::UserRole::Guest),
            ..owner.clone()
        };
        assert_eq!(judge_terminal(&guest, detached).await, RuleDecision::Allow);
        let attached = json!({"action": "run", "command": "nohup sleep 1 &"});
        assert_eq!(judge_terminal(&owner, attached).await, RuleDecision::Allow);
    }

    /// Trust markers in a terminal call's arguments change nothing: the
    /// rules read the turn's facts only.
    #[tokio::test]
    async fn terminal_trust_arguments_bypass_nothing() {
        let rules = terminal_rules(&[]).await;
        let spoofed = json!({
            "action": "run",
            "command": "node --version",
            "_trusted_session": true,
            "_user_role": "Owner",
        });
        assert!(matches!(
            judge_terminal(&rules, spoofed).await,
            RuleDecision::Ask(_)
        ));
    }

    /// Trust-all is the owner's `/trustall` command, not a tool action: a
    /// model call to the retired action changes nothing and asks nothing.
    #[tokio::test]
    async fn retired_trust_all_action_asks_nothing() {
        let rules = terminal_rules(&[]).await;
        assert!(!matches!(
            judge_terminal(&rules, json!({"action": "trust_all"})).await,
            RuleDecision::Ask(_)
        ));
    }

    /// A terminal call whose arguments cannot be read is never allowed: it
    /// asks, Critical, with the raw arguments on the card.
    #[tokio::test]
    async fn unreadable_terminal_call_asks_critical_with_raw_arguments() {
        let rules = terminal_rules(&["*"]).await;
        let raw = json!({"action": "run", "command": 5, "detach": "yes"});
        let RuleDecision::Ask(request) = judge_terminal(&rules, raw.clone()).await else {
            panic!("an unreadable call must ask");
        };
        assert_eq!(request.risk, sdk::ApprovalRisk::Critical);
        assert!(request.summary.contains(&raw.to_string()), "{request:?}");
        assert!(
            request
                .warnings
                .iter()
                .any(|w| w.contains("could not be read")),
            "{request:?}"
        );
        // The model's `_` keys of the wrong type are dropped first, as the
        // dispatcher drops them: the call is judged on what runs.
        let spoofed = json!({"action": "run", "command": "ls", "_session_id": 1});
        assert_eq!(judge_terminal(&rules, spoofed).await, RuleDecision::Allow);
    }

    /// Terminal cards go out in the terminal's configured mode (which picks
    /// the buttons); other tools' cards stay Cautious.
    #[tokio::test]
    async fn terminal_cards_use_the_terminal_permission_mode() {
        let rules = terminal_rules(&[]).await;
        assert_eq!(
            rules.card_permission_mode("terminal"),
            PermissionMode::Default
        );
        assert_eq!(
            rules.card_permission_mode("http_request"),
            PermissionMode::Cautious
        );
    }

    /// Allow Session adds the session prefix, as the in-tool prompt did:
    /// the next call runs without a card, but a trigger session still asks.
    #[tokio::test]
    async fn terminal_allow_session_adds_the_session_prefix() {
        let rules = terminal_rules(&[]).await;
        let call = sdk::ToolCall::new("c1", "terminal", run("printf first"));
        rules
            .record_owner_answer(
                sdk::RunId::new(),
                &call,
                Some(ApprovalResponse::AllowSession),
            )
            .await;
        assert_eq!(
            judge_terminal(&rules, run("printf second")).await,
            RuleDecision::Allow
        );
        let trigger = AppApprovalRules {
            untrusted_trigger_session: true,
            ..rules.clone()
        };
        assert!(matches!(
            judge_terminal(&trigger, run("printf second")).await,
            RuleDecision::Ask(_)
        ));
        // A decline grants nothing.
        let other = sdk::ToolCall::new("c2", "terminal", run("seq 1 2"));
        rules
            .record_owner_answer(sdk::RunId::new(), &other, Some(ApprovalResponse::Deny))
            .await;
        assert!(matches!(
            judge_terminal(&rules, run("seq 1 2")).await,
            RuleDecision::Ask(_)
        ));
    }

    /// The dispatcher refuses an out-of-scope call it cannot escalate, so
    /// the rules do not ask about it; on a confined terminal an escalatable
    /// target is asked about even for an allowlisted command, and an Allow
    /// Session grants that capability for the command.
    #[tokio::test]
    async fn terminal_scope_check_shapes_the_ask() {
        let refused = AppApprovalRules {
            terminal: terminal_rules(&["echo"])
                .await
                .terminal
                .map(|t| with_scope(t, TerminalScope::Refused("protected host data".to_string()))),
            ..terminal_rules(&[]).await
        };
        assert_eq!(
            judge_terminal(&refused, run("printf synthetic")).await,
            RuleDecision::Allow
        );

        let pool_file = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", pool_file.path().display()))
            .await
            .unwrap();
        std::mem::forget(pool_file);
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let confined: Arc<dyn crate::traits::Tool> = Arc::new(
            crate::tools::TerminalTool::new(
                crate::execution::active_execution_backend(),
                vec!["cat".to_string()],
                crate::tools::ApprovalBroker::new(tx),
                1,
                1000,
                crate::types::PermissionMode::Default,
                pool,
            )
            .await
            .with_confinement(crate::types::TerminalConfinement::Sandbox),
        );
        let escalation = crate::traits::ScopeEscalation {
            read_paths: vec!["/tmp/aidaemon-test/sibling/notes.txt".to_string()],
            write_paths: Vec::new(),
        };
        let scoped = escalation.clone();
        let rules = AppApprovalRules {
            terminal: TerminalApprovals::find(&[confined])
                .map(|t| with_scope(t, TerminalScope::Escalate(scoped.clone()))),
            ..terminal_rules(&[]).await
        };
        let args = run("cat /tmp/aidaemon-test/sibling/notes.txt");
        let asked = judge_terminal(&rules, args.clone()).await;
        assert!(
            asks_with(
                &asked,
                "read outside task scope: /tmp/aidaemon-test/sibling/notes.txt"
            ),
            "{asked:?}"
        );
        let call = sdk::ToolCall::new("c1", "terminal", args.clone());
        rules
            .record_owner_answer(
                sdk::RunId::new(),
                &call,
                Some(ApprovalResponse::AllowSession),
            )
            .await;
        assert_eq!(judge_terminal(&rules, args).await, RuleDecision::Allow);
    }

    /// A temp SQLite store for the tools whose judgment reads state.
    async fn approval_test_store() -> Arc<crate::state::SqliteStateStore> {
        let db = tempfile::NamedTempFile::new().unwrap();
        let path = db.path().to_str().unwrap().to_string();
        std::mem::forget(db);
        Arc::new(
            crate::state::SqliteStateStore::for_tests(
                &path,
                100,
                None,
                Arc::new(crate::memory::embeddings::EmbeddingService::deterministic()),
            )
            .await
            .unwrap(),
        )
    }

    /// Rules over one real tool.
    fn rules_over_one(tool: Arc<dyn crate::traits::Tool>) -> AppApprovalRules {
        AppApprovalRules {
            session_id: "telegram:synthetic-user-1".to_string(),
            user_role: Some(crate::types::UserRole::Owner),
            channel_visibility: Some(crate::types::ChannelVisibility::Private),
            tools: ApprovalTools::new([tool]),
            ..AppApprovalRules::default()
        }
    }

    /// Fetching external skill content asks the owner in every session, on a
    /// one-time card; reading and local edits need no card.
    #[tokio::test]
    async fn skill_fetches_ask_before_dispatch_on_a_one_time_card() {
        let dir = tempfile::TempDir::new().unwrap();
        let tool: Arc<dyn crate::traits::Tool> = Arc::new(
            crate::tools::ManageSkillsTool::new(
                dir.path().to_path_buf(),
                approval_test_store().await as Arc<dyn crate::traits::SkillsStore>,
            )
            .with_registries(vec!["https://registry.example.com/index.json".to_string()]),
        );
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                trusted_session: trusted,
                ..rules_over_one(tool.clone())
            };
            for args in [
                json!({"action":"add","url":"https://example.com/skill.md"}),
                json!({"action":"install","name":"synthetic"}),
                json!({"action":"update","name":"synthetic"}),
            ] {
                let call = sdk::ToolCall::new("c1", "manage_skills", args.clone());
                let RuleDecision::Ask(request) =
                    rules.judge(&hook_ctx(), &call, &tool_info()).await
                else {
                    panic!("{args} must ask (trusted={trusted})");
                };
                assert!(matches!(
                    rules.card_kind("manage_skills", &request),
                    crate::types::ApprovalKind::CommandOnce
                ));
            }
            for args in [
                json!({"action":"list"}),
                json!({"action":"remove","name":"synthetic"}),
            ] {
                let call = sdk::ToolCall::new("c2", "manage_skills", args.clone());
                assert_eq!(
                    rules.judge(&hook_ctx(), &call, &tool_info()).await,
                    RuleDecision::Allow,
                    "{args}"
                );
            }
        }
    }

    /// Scheduled work that runs the agent is confirmed by the owner on a
    /// goal card before anything is created, in every session; a guest's
    /// call and an internal run need no card (the call answers them without
    /// creating anything).
    #[tokio::test]
    async fn scheduled_work_asks_before_dispatch_on_a_goal_card() {
        let state = approval_test_store().await;
        let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::ScheduleTaskTool::new(
            state as Arc<dyn crate::traits::StateStore>,
        ));
        let work = json!({ "task": "Check API health", "when": "every 6h" });
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                trusted_session: trusted,
                ..rules_over_one(tool.clone())
            };
            let call = sdk::ToolCall::new("c1", "schedule_task", work.clone());
            let RuleDecision::Ask(request) = rules.judge(&hook_ctx(), &call, &tool_info()).await
            else {
                panic!("scheduled work must ask (trusted={trusted})");
            };
            assert_eq!(request.summary, "Check API health");
            assert!(matches!(
                rules.card_kind("schedule_task", &request),
                crate::types::ApprovalKind::GoalConfirmation
            ));
        }
        for rules in [
            AppApprovalRules {
                user_role: Some(crate::types::UserRole::Guest),
                ..rules_over_one(tool.clone())
            },
            AppApprovalRules {
                channel_visibility: Some(crate::types::ChannelVisibility::Internal),
                ..rules_over_one(tool.clone())
            },
        ] {
            let call = sdk::ToolCall::new("c4", "schedule_task", work.clone());
            assert_eq!(
                rules.judge(&hook_ctx(), &call, &tool_info()).await,
                RuleDecision::Allow
            );
        }
    }

    /// Creating a mandate is confirmed by the owner on the mandate's card
    /// before anything is stored, in every session (a mandate run's
    /// pre-approval does not cover delegating new authority); outside a
    /// verified private owner chat the call is answered without a card.
    #[tokio::test]
    async fn mandate_create_asks_before_dispatch_on_its_confirmation_card() {
        let store = approval_test_store().await;
        let tool: Arc<dyn crate::traits::Tool> = Arc::new(crate::tools::ManageMandatesTool::new(
            store.clone() as Arc<dyn crate::traits::StateStore>,
        ));
        let create = json!({
            "action": "create",
            "objective": "Never activate without confirmation",
            "max_mutating_actions_per_cycle": 0,
            "success_criteria": ["The bounded source is reviewed without unauthorized action"]
        });
        for trusted in [false, true] {
            let rules = AppApprovalRules {
                trusted_session: trusted,
                mandate_preapproved: trusted,
                ..rules_over_one(tool.clone())
            };
            let call = sdk::ToolCall::new("c1", "manage_mandates", create.clone());
            let RuleDecision::Ask(request) = rules.judge(&hook_ctx(), &call, &tool_info()).await
            else {
                panic!("a mandate create must ask (trusted={trusted})");
            };
            assert!(request.summary.starts_with("Delegate mandate:"));
            assert_eq!(request.risk, sdk::ApprovalRisk::Medium);
            assert!(matches!(
                rules.card_kind("manage_mandates", &request),
                crate::types::ApprovalKind::GoalConfirmation
            ));
            let call = sdk::ToolCall::new("c2", "manage_mandates", json!({"action": "list"}));
            assert_eq!(
                rules.judge(&hook_ctx(), &call, &tool_info()).await,
                RuleDecision::Allow
            );
        }
        let public = AppApprovalRules {
            channel_visibility: Some(crate::types::ChannelVisibility::Public),
            ..rules_over_one(tool.clone())
        };
        let call = sdk::ToolCall::new("c3", "manage_mandates", create);
        assert_eq!(
            public.judge(&hook_ctx(), &call, &tool_info()).await,
            RuleDecision::Allow
        );
        // Judging stores nothing.
        use crate::traits::store_prelude::*;
        assert!(
            store
                .list_mandates(Some("telegram:synthetic-user-1"), true)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn asks_always_carry_a_subject() {
        // Even an unparseable URL names the call it asks about.
        let RuleDecision::Ask(request) =
            judge(&rules(), json!({"method": "POST", "url": "not a url"})).await
        else {
            panic!("must ask");
        };
        assert!(!request.subject.trim().is_empty());
    }
}
