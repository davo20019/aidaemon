//! The calls a kernel run makes to its host. Each turn's kernel-side
//! adapters ([`super::hooks`], [`super::provider`]) put them on the turn's
//! channel; the host serves them on the turn's own task and answers on the
//! reply channel each carries.

use aidaemon_core as sdk;
use serde_json::Value;
use tokio::sync::oneshot;

/// What the kernel's event sink hands to the turn's dispatcher.
pub enum KernelNotice {
    /// A call the kernel refused before dispatch.
    Refusal(sdk::Receipt),
    /// The stuck detector fired.
    Stuck,
}

/// Everything the SDK run asks of its host, served on the turn's own task.
pub enum HostCall {
    Tool(ToolRequest),
    AfterCall(AfterCallCheck),
    Gate(GateRequest),
    Model(ModelCall),
    Stop(StopCheck),
    Budget(BudgetCheck),
    ModelError(ModelErrorCheck),
}

/// A model response to charge against the budgets before it is acted on.
pub struct BudgetCheck {
    pub response: sdk::ModelResponse,
    pub reply: oneshot::Sender<sdk::ModelDecision>,
}

/// A failed model call, for the host to say whether to try again.
pub struct ModelErrorCheck {
    /// The provider stack gave up on a timeout; it retried the rest itself.
    pub timed_out: bool,
    pub reply: oneshot::Sender<sdk::ModelErrorDecision>,
}

/// The model wants to stop; the host asks the run ledger.
pub struct StopCheck {
    /// Operations of this run that completed.
    pub completed_results: usize,
    /// The run's latest tool call failed and none succeeded.
    pub failed_without_success: bool,
    pub final_text: String,
    pub reply: oneshot::Sender<sdk::StopDecision>,
}

/// One kernel tool call as the host dispatcher sees it.
pub struct KernelCall {
    /// The kernel run the call belongs to (call ids are unique per run).
    pub run_id: sdk::RunId,
    pub name: String,
    pub arguments: Value,
    pub call_id: String,
}

/// A dispatched call came back to the kernel; record it.
pub struct AfterCallCheck {
    pub run_id: sdk::RunId,
    pub call_id: String,
    /// What the model reads; `None` when the dispatcher ran no such call.
    pub reply: oneshot::Sender<Option<sdk::ToolOutput>>,
}

/// One tool call handed from the SDK run to the host dispatcher.
pub struct ToolRequest {
    pub call: KernelCall,
    pub reply: oneshot::Sender<sdk::ToolResult>,
    /// Cancelled when the kernel stops waiting for this call (its timeout, or
    /// the run ending), so the tool stops too instead of running on unseen.
    pub abandoned: tokio_util::sync::CancellationToken,
}

impl std::ops::Deref for ToolRequest {
    type Target = KernelCall;
    fn deref(&self) -> &KernelCall {
        &self.call
    }
}

impl std::ops::DerefMut for ToolRequest {
    fn deref_mut(&mut self) -> &mut KernelCall {
        &mut self.call
    }
}

/// A call the kernel's hooks allowed so far, put to the dispatcher's gates
/// ([`super::hooks::DispatchGate`]) before it runs.
pub struct GateRequest {
    pub call: KernelCall,
    pub reply: oneshot::Sender<sdk::ToolDecision>,
}

/// One model request handed from the SDK run to the host dispatcher.
pub struct ModelCall {
    pub request: sdk::ModelRequest,
    pub reply: oneshot::Sender<Result<sdk::ModelResponse, sdk::ProviderError>>,
}
