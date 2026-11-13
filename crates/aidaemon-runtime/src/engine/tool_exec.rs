//! Running one tool call on the host: what the turn hands its host's tool
//! execution ([`crate::host::TurnHost::execute_tool_with_watchdog_outcome`]).

use std::sync::Arc;

use aidaemon_channels::context::WorkspaceGrant;
use aidaemon_core::ChannelVisibility;
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use tokio::sync::mpsc;

/// How one tool call runs: the turn's identity, channel, authority and
/// boundaries the tool's call context is built from.
pub struct ToolExecCtx<'a> {
    /// The tree's file fence for this call, when agents share the workspace.
    pub file_fence: Option<Arc<crate::coordination::CallFence>>,
    pub session_id: &'a str,
    pub task_id: Option<&'a str>,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_visibility: ChannelVisibility,
    pub channel_id: Option<&'a str>,
    pub project_scope: Option<&'a str>,
    pub trusted: bool,
    pub user_role: UserRole,
    /// Present only after `ChannelContext::active_workspace_grant` validated
    /// the exact guest/workspace/channel/sender binding.
    pub workspace_grant: Option<&'a WorkspaceGrant>,
    /// Exact action-bound authority for a call made under an autonomous
    /// mandate. It is issued by the execution loop and never model-visible.
    pub mandate_authority: Option<&'a aidaemon_work::mandates::MandateAuthorityGrant>,
    /// Dispatcher-owned identity of the originating model tool call.
    ///
    /// This is general causal lineage, not mandate-specific state: transparent
    /// adapters and detached work must retain it so their result can be joined
    /// back to the request. Mandate validation also binds its grant to this
    /// independently supplied identity.
    pub tool_call_id: Option<&'a str>,
    /// Rust-side hard boundary inherited from the request contract.
    pub mutation_forbidden: bool,
    /// The call came through the agent kernel, whose `ApprovalPolicy` asked
    /// the owner where needed (passed on to the tool as
    /// `ToolCallContext::kernel_approved` as `ApprovalScope::Call`).
    pub owner_approval_by_kernel: bool,
    /// Exact out-of-scope filesystem targets the controller declined to
    /// reject outright. Loop-owned; the tool must obtain user authority for
    /// them before running.
    pub scope_escalation: Option<&'a aidaemon_tools::semantics::ScopeEscalation>,
    /// The turn's actual authority, taint and root request, handed to
    /// `spawn_agent` in its call context's extensions so a child inherits
    /// what this turn had rather than a re-resolution at spawn time.
    pub turn_authority: Option<&'a crate::authority::InheritedAuthority>,
    /// The turn's trusted roots and the cwd relative paths resolve against,
    /// handed to `terminal` as `ToolCallContext::trust_roots`/`trust_cwd` so
    /// a run moved to the background records the trust its launch was
    /// judged with.
    pub trust_roots: Option<(&'a [String], &'a str)>,
    /// The call is the only one in a root turn's model response (passed on
    /// as `ToolCallContext::lone_root_call`).
    pub lone_root_call: bool,
    /// The MCP-registry tool the kernel bridged and judged at turn start
    /// for this call. A kernel call executes exactly this instance, never a
    /// fresh registry lookup, so a server reload mid-turn cannot swap in a
    /// tool nobody judged.
    pub mcp_tool: Option<&'a Arc<dyn aidaemon_tools::OutcomeTool>>,
}

/// No adapter (native or MCP) is registered under the requested name, so the
/// call was refused before dispatch. Typed so the receipt boundary can record
/// `RejectedBeforeDispatch` instead of the `Dispatched` stamp it gives legacy
/// adapter errors.
#[derive(Debug)]
pub struct UnknownToolError {
    pub name: String,
    pub available: Vec<String>,
}

impl std::fmt::Display for UnknownToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unknown tool '{}'. Available tools: [{}]. Use one of these or respond with text only.",
            self.name,
            self.available.join(", ")
        )
    }
}

impl std::error::Error for UnknownToolError {}
