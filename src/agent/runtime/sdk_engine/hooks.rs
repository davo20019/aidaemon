//! The kernel-side adapters: hooks and the bridged daemon tools. Each one
//! only forwards to the daemon over the turn's `DaemonCall` channel.

use super::*;

/// The ledger closeout as an SDK stop hook: before a final answer ships, the
/// run ledger decides whether a still-reachable expectation needs another
/// pass. Receipts, not the reply text, are the evidence.
pub(super) struct LedgerStop {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for LedgerStop {
    async fn on_stop(&self, ctx: &sdk::HookContext<'_>, final_text: &str) -> sdk::StopDecision {
        let completed_results = ctx
            .run_receipts()
            .filter(|receipt| receipt.status == sdk::ReceiptStatus::Succeeded)
            .count();
        let failed_without_success = completed_results == 0
            && ctx
                .run_receipts()
                .last()
                .is_some_and(|receipt| receipt.status == sdk::ReceiptStatus::Failed);
        let (reply, answer) = oneshot::channel();
        if self
            .calls
            .send(DaemonCall::Stop(StopCheck {
                completed_results,
                failed_without_success,
                final_text: final_text.to_string(),
                reply,
            }))
            .await
            .is_err()
        {
            return sdk::StopDecision::Accept;
        }
        answer.await.unwrap_or(sdk::StopDecision::Accept)
    }
}

/// The same-tool guard's limits: 8 consecutive calls
/// (`MAX_CONSECUTIVE_SAME_TOOL`), 16 for a `Versatile` tool, and 40 for a
/// `Stepwise` one (computer use). The rule itself is
/// `aidaemon_policies::SameToolStreakGuard`.
pub(super) const SAME_TOOL_LIMITS: aidaemon_policies::StreakLimits =
    aidaemon_policies::StreakLimits {
        base: crate::agent::MAX_CONSECUTIVE_SAME_TOOL,
        versatile: crate::agent::MAX_CONSECUTIVE_SAME_TOOL + 8,
        stepwise: 40,
    };

/// Each tool's consecutive-call allowance (`dispatch_facts().same_tool_streak`).
pub(super) fn same_tool_streaks<'a>(
    tools: impl IntoIterator<Item = &'a Arc<dyn crate::traits::Tool>>,
) -> std::collections::HashMap<String, crate::traits::SameToolStreak> {
    tools
        .into_iter()
        .map(|tool| {
            (
                tool.name().to_string(),
                tool.dispatch_facts().same_tool_streak,
            )
        })
        .collect()
}

/// Names of the tools that are read-only for every call. The same-tool
/// guard's distinct-arguments exemption is judged by the tool's worst case
/// (`capabilities().read_only`), not the per-call annotation: a mixed tool
/// such as `terminal` types most calls read-only, and must stay limited.
pub(super) fn read_only_tool_names<'a>(
    tools: impl IntoIterator<Item = &'a Arc<dyn crate::traits::Tool>>,
) -> std::collections::HashSet<String> {
    tools
        .into_iter()
        .filter(|tool| tool.capabilities().read_only)
        .map(|tool| tool.name().to_string())
        .collect()
}

/// Where a hook sits in the turn's `before_tool` chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HookSlot {
    Authority,
    SameToolLimit,
    LedgerStop,
    Memory,
    Egress,
    Untrusted,
    Approval,
}

/// The turn's hooks, registered in the pinned order: authority first; the
/// daemon's loop guard and ledger stop; memory; the outbound secret scan,
/// so no secret reaches the untrusted-content verifier or an approval card;
/// the untrusted-content rule; and owner approval last, so the owner is
/// asked only about calls every other hook allows. No hook after
/// `UntrustedHook` may rewrite a call. The kernel's decline check runs
/// before all of them.
pub(super) struct TurnHooks {
    pub(super) authority: sdk::AuthorityHook,
    pub(super) same_tool: aidaemon_policies::SameToolStreakGuard,
    pub(super) ledger: LedgerStop,
    pub(super) memory: Vec<Arc<dyn sdk::Hook>>,
    pub(super) egress: aidaemon_policies::EgressGuard,
    pub(super) untrusted: sdk::UntrustedHook,
    pub(super) approval: aidaemon_policies::ApprovalPolicy,
}

impl TurnHooks {
    pub(super) fn chain(self) -> Vec<(HookSlot, Arc<dyn sdk::Hook>)> {
        let mut chain: Vec<(HookSlot, Arc<dyn sdk::Hook>)> = vec![
            (HookSlot::Authority, Arc::new(self.authority)),
            (HookSlot::SameToolLimit, Arc::new(self.same_tool)),
            (HookSlot::LedgerStop, Arc::new(self.ledger)),
        ];
        chain.extend(self.memory.into_iter().map(|hook| (HookSlot::Memory, hook)));
        chain.push((HookSlot::Egress, Arc::new(self.egress)));
        chain.push((HookSlot::Untrusted, Arc::new(self.untrusted)));
        chain.push((HookSlot::Approval, Arc::new(self.approval)));
        chain
    }
}

/// A daemon tool as the SDK sees it. It only forwards the call; the daemon
/// dispatcher executes it on the turn's own task.
pub(super) struct BridgedTool {
    pub(super) spec: sdk::ToolSpec,
    pub(super) tool: Arc<dyn crate::traits::Tool>,
    /// `manage_memories` offered with only its goal actions.
    pub(super) goal_actions_only: bool,
    pub(super) calls: mpsc::Sender<DaemonCall>,
    /// The turn's trusted roots: where an effect lands can decide its class.
    pub(super) trust_roots: Arc<Vec<String>>,
}

#[async_trait]
impl sdk::Tool for BridgedTool {
    fn spec(&self) -> sdk::ToolSpec {
        self.spec.clone()
    }

    /// Worst case for the tool as a whole: only a tool that is read-only
    /// for every call says so; any other may be destructive and egress.
    fn annotations(&self) -> sdk::ToolAnnotations {
        if self.tool.capabilities().read_only {
            sdk::ToolAnnotations::READ_ONLY
        } else {
            sdk::ToolAnnotations {
                read_only: false,
                destructive: true,
                idempotent: false,
                egress: true,
            }
        }
    }

    /// Judged per call: a mixed tool such as `terminal` reads for `ls` and
    /// writes for `rm`.
    fn annotations_for(&self, arguments: &Value) -> sdk::ToolAnnotations {
        self.tool.call_annotations_in(
            &arguments.to_string(),
            &crate::traits::TrustRoots::new(&self.trust_roots),
        )
    }

    /// The daemon tool's own declaration for the arguments it will be
    /// dispatched with (after its argument canonicalization).
    fn effects(&self, arguments: &Value) -> sdk::CallEffects {
        let raw = arguments.to_string();
        let canonical = self.tool.canonicalize_arguments(&raw).unwrap_or(raw);
        self.tool.call_effects(&canonical)
    }

    /// The declared effects as the kernel's access manifest: egress
    /// targets as hosts, path effects as writes, and the terminal's command
    /// line of a `shell_command_line` tool, given as `command` or `script`
    /// (so the approval policy's catastrophic-command check sees it).
    fn access(&self, arguments: &Value) -> sdk::AccessManifest {
        let mut manifest = sdk::AccessManifest::default();
        if self.tool.dispatch_facts().shell_command_line {
            for field in ["command", "script"] {
                if let Some(text) = arguments
                    .get(field)
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                {
                    manifest.commands.push(text.to_string());
                }
            }
        }
        for key in self.effects(arguments).keys {
            let list = match key.class {
                sdk::EffectClass::Egress => &mut manifest.hosts,
                sdk::EffectClass::Write
                | sdk::EffectClass::Delete
                | sdk::EffectClass::Configuration => &mut manifest.writes,
                _ => continue,
            };
            if !list.contains(&key.target) {
                list.push(key.target);
            }
        }
        manifest
    }

    /// Fallback only: the dispatcher stamps each result with the exact
    /// call's trust (`ToolSemantics::output_trust`).
    fn source_trust(&self, _arguments: &Value) -> sdk::SourceTrust {
        sdk::SourceTrust::Untrusted
    }

    /// Daemon tools check their own arguments (some accept aliases for
    /// schema-required keys) and report bad ones through the dispatcher.
    /// Only the goal-actions view of `manage_memories` is narrowed here; the
    /// kernel records that refusal in the run ledger.
    fn validate(&self, arguments: &Value) -> Result<(), String> {
        if self.goal_actions_only {
            crate::agent::memory_plugin::check_goal_action(arguments)?;
        }
        Ok(())
    }

    fn timeout(&self) -> Option<Duration> {
        Some(TOOL_TIMEOUT)
    }

    async fn call(&self, arguments: Value, ctx: &sdk::ToolContext) -> sdk::ToolResult {
        let (reply, answer) = oneshot::channel();
        self.calls
            .send(DaemonCall::Tool(ToolRequest {
                run_id: ctx.run_id,
                name: self.spec.name.clone(),
                arguments,
                call_id: ctx.call_id.clone(),
                reply,
            }))
            .await
            .map_err(|_| sdk::ToolError("tool dispatcher stopped".to_string()))?;
        answer
            .await
            .map_err(|_| sdk::ToolError("tool dispatcher dropped the call".to_string()))?
    }
}

/// Build an SDK tool spec from a daemon tool's OpenAI-style function schema.
pub(super) fn tool_spec(tool: &dyn crate::traits::Tool) -> sdk::ToolSpec {
    tool_spec_from_schema(tool, &tool.schema())
}

pub(super) fn tool_spec_from_schema(
    tool: &dyn crate::traits::Tool,
    schema: &Value,
) -> sdk::ToolSpec {
    let parameters = schema
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
    let description = schema
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_else(|| tool.description())
        .to_string();
    sdk::ToolSpec {
        name: tool.name().to_string(),
        description,
        parameters,
    }
}
