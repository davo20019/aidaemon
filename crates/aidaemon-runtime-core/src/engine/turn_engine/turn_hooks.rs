//! The turn's hook chain and the daemon-only kernel-side adapters. The
//! forwarding hooks and bridged tools live in `crate::engine::hooks`.

use super::*;
pub use crate::engine::hooks::{
    AfterCallRecord, BudgetGate, CutoffAdjust, DispatchGate, LedgerStop, ModelRetry,
    SAME_TOOL_LIMITS, StuckNotice, TextOnlyNext, read_only_tool_names, same_tool_streaks,
    tool_spec, tool_spec_from_schema,
};

/// Where a hook sits in the turn's `before_tool` chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookSlot {
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
    /// A run capped by its device (`approvals::device_ceiling`), inserted
    /// right after `Approval` by `with_device_ceiling`.
    DeviceCeiling,
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
pub struct TurnHooks {
    pub canonical: super::tool_calls::CanonicalCallHook,
    pub authority: sdk::AuthorityHook,
    pub same_tool: aidaemon_policies::SameToolStreakGuard,
    pub ledger: LedgerStop,
    pub text_only: TextOnlyNext,
    pub budget: BudgetGate,
    pub stuck: StuckNotice,
    pub model_pin: ModelPin,
    pub cutoff: CutoffAdjust,
    pub retry: ModelRetry,
    pub memory: Vec<Arc<dyn sdk::Hook>>,
    pub protected_paths: aidaemon_tools::ProtectedPathGuard,
    pub network: aidaemon_tools::NetworkGuard,
    pub egress: aidaemon_policies::EgressGuard,
    pub untrusted: sdk::UntrustedHook,
    pub approval: aidaemon_policies::ApprovalPolicy,
    pub dispatch: DispatchGate,
    pub coordination: crate::coordination::CoordinationHook,
    pub after_call: AfterCallRecord,
}

impl TurnHooks {
    pub fn chain(self) -> Vec<(HookSlot, Arc<dyn sdk::Hook>)> {
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

/// The turn's hook chain with the device ceiling
/// (`approvals::device_ceiling::DeviceCeilingHook`) right after owner
/// approval: after every hook that may still refuse or rewrite the call (so
/// the owner is asked only about a call that would otherwise run) and before
/// the dispatcher's gate. The kernel runs a call only when every hook allows
/// it, so no allow in `ApprovalPolicy` (a mode, an allow rule, a grant)
/// stands in for this hook's answer.
pub fn with_device_ceiling(
    mut chain: Vec<(HookSlot, Arc<dyn sdk::Hook>)>,
    cap: crate::device_cap::DeviceCap,
) -> Vec<(HookSlot, Arc<dyn sdk::Hook>)> {
    let hook: Arc<dyn sdk::Hook> = Arc::new(
        crate::approvals::device_ceiling::DeviceCeilingHook::new(cap),
    );
    let at = chain
        .iter()
        .position(|(slot, _)| *slot == HookSlot::Approval)
        .map_or(chain.len(), |approval| approval + 1);
    chain.insert(at, (HookSlot::DeviceCeiling, hook));
    chain
}

/// A computer_use GUI loop pins a vision-capable model for its task; every
/// later call of the task uses it. Early in the chain, so hooks that size the
/// request by its model see the one that will answer.
pub struct ModelPin {
    pub task_id: String,
    pub computer_use: Option<Arc<dyn crate::host::ComputerUseState>>,
}

#[async_trait]
impl sdk::Hook for ModelPin {
    async fn before_model(&self, _ctx: &sdk::HookContext<'_>, request: &mut sdk::ModelRequest) {
        if let Some(computer_use) = &self.computer_use
            && let Some(model) = computer_use.pinned_model(&self.task_id).await
        {
            request.model = model;
        }
    }
}

/// A daemon tool as this turn's kernel agent runs it. Daemon tools check
/// their schema's required keys themselves (some accept aliases for them)
/// and report bad ones through the dispatcher. Their own argument contract
/// is checked here, before any hook, so the owner is never asked to approve
/// a call that cannot run; and the goal-actions view of the memory tool is
/// narrowed here, so the kernel records that refusal in the run ledger.
pub fn turn_tool(
    spec: sdk::ToolSpec,
    tool: Arc<dyn aidaemon_tools::OutcomeTool>,
    goal_actions_only: bool,
    calls: mpsc::Sender<DaemonCall>,
    trust_roots: Arc<Vec<String>>,
) -> aidaemon_tools::KernelAdapter<dyn aidaemon_tools::OutcomeTool> {
    let checked = tool.clone();
    crate::engine::hooks::turn_tool(
        spec,
        tool,
        Arc::new(move |arguments| {
            aidaemon_tools::outcome::check_argument_contract(checked.as_ref(), arguments)?;
            if goal_actions_only {
                crate::turn::memory::check_goal_action(arguments)?;
            }
            Ok(())
        }),
        calls,
        trust_roots,
    )
}

#[cfg(test)]
mod device_ceiling_slot_tests {
    use super::*;

    #[test]
    fn the_device_ceiling_sits_right_after_owner_approval() {
        struct Nop;
        impl sdk::Hook for Nop {}
        let nop = || -> Arc<dyn sdk::Hook> { Arc::new(Nop) };
        let chain = vec![
            (HookSlot::Authority, nop()),
            (HookSlot::Untrusted, nop()),
            (HookSlot::Approval, nop()),
            (HookSlot::Dispatch, nop()),
        ];
        let slots: Vec<HookSlot> = with_device_ceiling(chain, crate::device_cap::DeviceCap::new())
            .into_iter()
            .map(|(slot, _)| slot)
            .collect();
        assert_eq!(
            slots,
            [
                HookSlot::Authority,
                HookSlot::Untrusted,
                HookSlot::Approval,
                HookSlot::DeviceCeiling,
                HookSlot::Dispatch
            ]
        );
    }
}
