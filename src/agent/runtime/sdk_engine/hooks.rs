//! The kernel-side adapters: hooks and the bridged daemon tools. Each one
//! only forwards to the daemon over the turn's `DaemonCall` channel.

use super::model_call::{raised_output_ceiling, thinking_cutoff_effort};
use super::*;

/// The ledger closeout as an SDK stop hook: before a final answer ships, the
/// run ledger decides whether a still-reachable expectation needs another
/// pass. Receipts, not the reply text, are the evidence.
pub(super) struct LedgerStop {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for LedgerStop {
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(super::TOOL_TIMEOUT)
    }

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
    AfterCall,
    Canonical,
    Authority,
    SameToolLimit,
    LedgerStop,
    TextOnly,
    Budget,
    Stuck,
    ModelPin,
    Cutoff,
    Retry,
    Memory,
    ProtectedPaths,
    Network,
    Egress,
    Untrusted,
    Approval,
    Dispatch,
    Coordination,
}

/// The turn's hooks, registered in the pinned order: the canonical form of
/// the call first, so every later hook judges the call that runs; authority; the
/// daemon's loop guard and ledger stop; memory; the outbound secret scan,
/// so no secret reaches the untrusted-content verifier or an approval card;
/// the untrusted-content rule; and owner approval last, so the owner is
/// asked only about calls every other hook allows. Stale writes are judged
/// after everything else, against the file as it is when the call would run.
/// No hook after `UntrustedHook` may rewrite a call. The kernel's decline
/// check runs before all of them.
pub(super) struct TurnHooks {
    pub(super) canonical: super::tool_calls::CanonicalCallHook,
    pub(super) authority: sdk::AuthorityHook,
    pub(super) same_tool: aidaemon_policies::SameToolStreakGuard,
    pub(super) ledger: LedgerStop,
    pub(super) text_only: TextOnlyNext,
    pub(super) budget: BudgetGate,
    pub(super) stuck: StuckNotice,
    pub(super) model_pin: ModelPin,
    pub(super) cutoff: CutoffAdjust,
    pub(super) retry: ModelRetry,
    pub(super) memory: Vec<Arc<dyn sdk::Hook>>,
    pub(super) protected_paths: aidaemon_tools::ProtectedPathGuard,
    pub(super) network: aidaemon_tools::NetworkGuard,
    pub(super) egress: aidaemon_policies::EgressGuard,
    pub(super) untrusted: sdk::UntrustedHook,
    pub(super) approval: aidaemon_policies::ApprovalPolicy,
    pub(super) dispatch: DispatchGate,
    pub(super) coordination: crate::agent::coordination::CoordinationHook,
    pub(super) after_call: AfterCallRecord,
}

impl TurnHooks {
    pub(super) fn chain(self) -> Vec<(HookSlot, Arc<dyn sdk::Hook>)> {
        let mut chain: Vec<(HookSlot, Arc<dyn sdk::Hook>)> = vec![
            (HookSlot::AfterCall, Arc::new(self.after_call)),
            (HookSlot::Canonical, Arc::new(self.canonical)),
            (HookSlot::Authority, Arc::new(self.authority)),
            (HookSlot::SameToolLimit, Arc::new(self.same_tool)),
            (HookSlot::LedgerStop, Arc::new(self.ledger)),
            (HookSlot::TextOnly, Arc::new(self.text_only)),
            (HookSlot::Budget, Arc::new(self.budget)),
            (HookSlot::Stuck, Arc::new(self.stuck)),
            (HookSlot::ModelPin, Arc::new(self.model_pin)),
            (HookSlot::Cutoff, Arc::new(self.cutoff)),
            (HookSlot::Retry, Arc::new(self.retry)),
        ];
        chain.extend(self.memory.into_iter().map(|hook| (HookSlot::Memory, hook)));
        chain.push((HookSlot::ProtectedPaths, Arc::new(self.protected_paths)));
        chain.push((HookSlot::Network, Arc::new(self.network)));
        chain.push((HookSlot::Egress, Arc::new(self.egress)));
        chain.push((HookSlot::Untrusted, Arc::new(self.untrusted)));
        chain.push((HookSlot::Approval, Arc::new(self.approval)));
        chain.push((HookSlot::Dispatch, Arc::new(self.dispatch)));
        chain.push((HookSlot::Coordination, Arc::new(self.coordination)));
        chain
    }
}

/// Withholds tools from the next request only, when a retry must answer in
/// text from the receipts it already holds.
pub(super) struct TextOnlyNext {
    pub(super) flag: Arc<AtomicBool>,
}

#[async_trait]
impl sdk::Hook for TextOnlyNext {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        if self.flag.swap(false, Ordering::Relaxed) {
            request.tool_choice = sdk::ToolChoice::None;
        }
    }
}

/// Charges each response to the token budgets before anything acts on it.
/// A budget is a durable balance spanning runs (and may ask the owner to
/// extend it), so it is the daemon's to judge; the kernel's own limits only
/// count this run. Served on the turn's task.
pub(super) struct BudgetGate {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for BudgetGate {
    /// Longer than an owner's budget card can stay open.
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(std::time::Duration::from_secs(60 * 60))
    }

    async fn after_model(
        &self,
        _ctx: &sdk::HookContext<'_>,
        response: &mut sdk::ModelResponse,
    ) -> sdk::ModelDecision {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(DaemonCall::Budget(BudgetCheck {
                response: response.clone(),
                reply,
            }))
            .await;
        if sent.is_err() {
            return sdk::ModelDecision::Accept;
        }
        answer.await.unwrap_or(sdk::ModelDecision::Accept)
    }
}

/// A computer_use GUI loop pins a vision-capable model for its task; every
/// later call of the task uses it. Early in the chain, so hooks that size the
/// request by its model see the one that will answer.
pub(super) struct ModelPin {
    pub(super) task_id: String,
}

#[async_trait]
impl sdk::Hook for ModelPin {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        #[cfg(feature = "computer_use")]
        {
            request.model =
                crate::agent::computer_use::resolve_model_for_task(&self.task_id, &request.model)
                    .await;
        }
        #[cfg(not(feature = "computer_use"))]
        let _ = (&self.task_id, request);
    }
}

/// A reasoning model that spent its whole output budget thinking was usually
/// on a hard step: give it more output room first, and lower reasoning
/// effort only when the cutoffs repeat.
#[derive(Default)]
pub(super) struct CutoffAdjust {
    consecutive: std::sync::atomic::AtomicU8,
    raised_output_tokens: std::sync::atomic::AtomicU32,
}

impl CutoffAdjust {
    #[cfg(test)]
    pub(super) fn cutoffs(&self) -> u8 {
        self.consecutive.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl sdk::Hook for CutoffAdjust {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        let raised = self.raised_output_tokens.load(Ordering::Relaxed);
        if raised > 0 {
            request.max_output_tokens = Some(
                request
                    .max_output_tokens
                    .map_or(raised, |cap| cap.max(raised)),
            );
        }
        if let Some(effort) = thinking_cutoff_effort(self.consecutive.load(Ordering::Relaxed)) {
            request.options.reasoning_effort = Some(effort);
        }
    }

    async fn after_model(
        &self,
        ctx: &sdk::HookContext<'_>,
        response: &mut sdk::ModelResponse,
    ) -> sdk::ModelDecision {
        let produced_output = !response.tool_calls.is_empty() || !response.text.trim().is_empty();
        if produced_output {
            self.consecutive.store(0, Ordering::Relaxed);
        } else if response
            .note
            .as_deref()
            .is_some_and(|note| note.contains("truncated"))
        {
            let count = self.consecutive.fetch_add(1, Ordering::Relaxed) + 1;
            let raised = raised_output_ceiling(
                u32::try_from(response.usage.output_tokens).ok(),
                self.raised_output_tokens.load(Ordering::Relaxed),
            );
            if let Some(raised) = raised {
                self.raised_output_tokens.store(raised, Ordering::Relaxed);
            }
            warn!(
                session_id = ctx.session_id,
                consecutive_truncations = count,
                raised_output_tokens = ?raised,
                next_effort = ?thinking_cutoff_effort(count),
                "SDK engine: response truncated at the token limit with no output"
            );
        }
        sdk::ModelDecision::Accept
    }
}

/// Asks the daemon whether a failed model call is worth another try.
/// Served on the turn's task.
pub(super) struct ModelRetry {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for ModelRetry {
    async fn on_model_error(
        &self,
        _ctx: &sdk::HookContext<'_>,
        error: &sdk::ProviderError,
    ) -> sdk::ModelErrorDecision {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(DaemonCall::ModelError(ModelErrorCheck {
                timed_out: matches!(error, sdk::ProviderError::Transient(_)),
                reply,
            }))
            .await;
        if sent.is_err() {
            return sdk::ModelErrorDecision::Fail;
        }
        answer.await.unwrap_or(sdk::ModelErrorDecision::Fail)
    }
}

/// Records that the run repeated itself. What to do about it stays the
/// kernel's default: nudge, then stop.
pub(super) struct StuckNotice {
    pub(super) notices: mpsc::UnboundedSender<KernelNotice>,
}

#[async_trait]
impl sdk::Hook for StuckNotice {
    async fn on_stuck(
        &self,
        _ctx: &sdk::HookContext<'_>,
        _signal: &sdk::StuckSignal,
    ) -> Option<sdk::StuckDecision> {
        let _ = self.notices.send(KernelNotice::Stuck);
        None
    }
}

/// The dispatcher's own gates as the last `before_tool` hook, after the
/// owner was asked (so a call approved after a wait is judged as of now):
/// the session's live mode, the untrusted-reference block, the filesystem
/// scope and the mandate authority refuse (`Deny`); deferral for new
/// project instructions, a duplicate `send_file` and a durable replay
/// answer in the tool's place (`Answer`). Served on the turn's task.
pub(super) struct DispatchGate {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for DispatchGate {
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(super::TOOL_TIMEOUT)
    }

    async fn before_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(DaemonCall::Gate(GateRequest {
                call: KernelCall {
                    run_id: ctx.run_id,
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    call_id: call.id.clone(),
                },
                reply,
            }))
            .await;
        let stopped = || sdk::ToolDecision::Deny {
            reason: "the tool dispatcher stopped before this call was checked".to_string(),
        };
        if sent.is_err() {
            return stopped();
        }
        answer.await.unwrap_or_else(|_| stopped())
    }
}

/// Records a call the dispatcher ran: result fitting, receipt, status line,
/// learning and the checklist, on the turn's own task. It runs first among
/// the after-tool hooks, so the others see what the model will read.
pub(super) struct AfterCallRecord {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl sdk::Hook for AfterCallRecord {
    fn timeout(&self) -> Option<std::time::Duration> {
        Some(super::TOOL_TIMEOUT)
    }

    async fn after_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        receipt: &sdk::Receipt,
        output: &mut sdk::ToolOutput,
    ) {
        let (reply, answer) = oneshot::channel();
        let sent = self
            .calls
            .send(DaemonCall::AfterCall(AfterCallCheck {
                run_id: ctx.run_id,
                call_id: receipt.call_id.clone(),
                reply,
            }))
            .await;
        match (sent, answer.await) {
            (Ok(()), Ok(None)) => {}
            (Ok(()), Ok(Some(recorded))) => *output = recorded,
            _ => {
                *output = sdk::ToolOutput::error(
                    "the tool dispatcher stopped before recording this result",
                );
            }
        }
    }
}

/// Runs a turn's tool calls on the daemon dispatcher, on the turn's own task.
pub(super) struct TurnDispatch {
    pub(super) calls: mpsc::Sender<DaemonCall>,
}

#[async_trait]
impl aidaemon_tools::OutcomeRunner for TurnDispatch {
    async fn run(&self, tool: &str, arguments: Value, ctx: &sdk::ToolContext) -> sdk::ToolResult {
        let (reply, answer) = oneshot::channel();
        let abandoned = tokio_util::sync::CancellationToken::new();
        let _stop_when_dropped = abandoned.clone().drop_guard();
        self.calls
            .send(DaemonCall::Tool(ToolRequest {
                abandoned,
                call: KernelCall {
                    run_id: ctx.run_id,
                    name: tool.to_string(),
                    arguments,
                    call_id: ctx.call_id.clone(),
                },
                reply,
            }))
            .await
            .map_err(|_| sdk::ToolError("tool dispatcher stopped".to_string()))?;
        answer
            .await
            .map_err(|_| sdk::ToolError("tool dispatcher dropped the call".to_string()))?
    }
}

/// A daemon tool as this turn's kernel agent runs it. Daemon tools check
/// their own arguments (some accept aliases for schema-required keys) and
/// report bad ones through the dispatcher; only the goal-actions view of
/// the memory tool is narrowed here, so the kernel records that refusal in
/// the run ledger.
pub(super) fn turn_tool(
    spec: sdk::ToolSpec,
    tool: Arc<dyn crate::traits::Tool>,
    goal_actions_only: bool,
    calls: mpsc::Sender<DaemonCall>,
    trust_roots: Arc<Vec<String>>,
) -> aidaemon_tools::KernelAdapter<dyn crate::traits::Tool> {
    aidaemon_tools::KernelAdapter::new(tool)
        .with_spec(spec)
        .with_trust_roots(trust_roots)
        .with_timeout(TOOL_TIMEOUT)
        .with_validator(Arc::new(move |arguments| {
            if goal_actions_only {
                crate::agent::memory_plugin::check_goal_action(arguments)?;
            }
            Ok(())
        }))
        .with_runner(Arc::new(TurnDispatch { calls }))
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
