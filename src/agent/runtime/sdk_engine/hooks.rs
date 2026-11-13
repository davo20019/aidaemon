//! The turn's hook chain and the daemon-only kernel-side adapters. The
//! forwarding hooks and bridged tools live in `aidaemon_runtime::engine::hooks`.

use super::*;
pub(super) use aidaemon_runtime::engine::hooks::{
    read_only_tool_names, same_tool_streaks, tool_spec, tool_spec_from_schema, AfterCallRecord,
    BudgetGate, CutoffAdjust, DispatchGate, LedgerStop, ModelRetry, StuckNotice, TextOnlyNext,
    SAME_TOOL_LIMITS,
};

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

/// A computer_use GUI loop pins a vision-capable model for its task; every
/// later call of the task uses it. Early in the chain, so hooks that size the
/// request by its model see the one that will answer.
pub(super) struct ModelPin {
    pub(super) task_id: String,
    pub(super) computer_use: Option<Arc<dyn aidaemon_runtime::host::ComputerUseState>>,
}

#[async_trait]
impl sdk::Hook for ModelPin {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        if let Some(computer_use) = &self.computer_use {
            if let Some(model) = computer_use.pinned_model(&self.task_id).await {
                request.model = model;
            }
        }
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
    aidaemon_runtime::engine::hooks::turn_tool(
        spec,
        tool,
        Arc::new(move |arguments| {
            if goal_actions_only {
                crate::agent::memory_plugin::check_goal_action(arguments)?;
            }
            Ok(())
        }),
        calls,
        trust_roots,
    )
}
