//! Agent turns on the aidaemon SDK kernel (`crates/aidaemon-core`).
//!
//! Every path that reaches `handle_message_impl` (channel ingress, internal
//! continuations, goal task leads, spawned children) runs one SDK run per
//! turn.
//!
//! What the daemon keeps owning:
//! - the durable turn record: `TaskStart`, `UserMessage`, `AssistantResponse`
//!   and `TaskEnd` are emitted here, so channels get a valid response
//!   envelope and history reloads across turns;
//! - tool dispatch: every daemon tool is exposed to the SDK, but calls are
//!   executed by `Agent::execute_tool_outcome` on this turn's task, so argument
//!   enrichment, role checks, workspace grants and the tools' own approval
//!   prompts all apply.
//!
//! The system prompt comes from the daemon's prompt builder (identity,
//! skills, memory facts, project instructions, mandate isolation).
//!
//! Images: the current turn's image attachments go to the model on the user
//! message, and images a tool observed (`ToolObservation` attachments, such
//! as an image file `read_file` opened) go with that tool's result, both as
//! [`sdk::Image`]. Gated by `[files]` vision settings and the engine model.
//! Replayed history stays text-only, like every archived turn.
//!
//! Layout: this file builds the per-turn kernel agent and runs the
//! `DaemonCall` select loop. The daemon side of each call lives in
//! `model_call` (provider requests), `tool_calls` (tool dispatch), `stop`
//! (closeout and stop-time demands) and `budgets`; `hooks` holds the
//! kernel-side adapters and `outcome` turns a finished run into the reply.
//!
//! There is no intent gate, no pre-execution plan / critique / evidence gate
//! and no guided `tool_choice` forcing: completion is the ledger closeout
//! (`LedgerStop`), judged from receipts.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_core as sdk;
use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::FutureExt;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};
use uuid::Uuid;

use super::agent_helpers::touch_heartbeat;
use super::memory_plugin::{daemon_tool_offer, memory_parts, DaemonToolOffer, MemoryTurn};
use super::sdk_provider::{
    fit_run_values, from_provider_response, message_text, prefix_with_notes, split_injected_notes,
    to_chat_options, to_daemon_messages_from, to_daemon_tools, values_to_sdk, DaemonProvider,
    ModelCall,
};
use super::tool_exec::ToolExecCtx;
use super::Agent;
use crate::agent::vision::encode_image_attachment;
use crate::config::{AppConfig, ProviderKind, VisionConfig};
use crate::events::{EventEmitter, EventType, TaskOutcome, TaskStatus};
use crate::traits::{AttachmentProvenance, Message, MessageAttachment};
use crate::types::{ChannelContext, StatusUpdate, UserRole};

#[path = "sdk_engine/budgets.rs"]
mod budgets;
#[path = "sdk_engine/compaction.rs"]
mod compaction;
#[path = "sdk_engine/hooks.rs"]
mod hooks;
#[path = "sdk_engine/model_call.rs"]
mod model_call;
#[path = "sdk_engine/outcome.rs"]
mod outcome;
#[path = "sdk_engine/stop.rs"]
mod stop;
#[path = "sdk_engine/tool_calls.rs"]
mod tool_calls;

use budgets::*;
use hooks::*;
pub(crate) use outcome::TurnVerdict;
use stop::*;

/// Model turns per run. Token budgets, the same-tool guard and the stuck
/// detector bound runaway work; this only stops a run that defeats all three.
/// Long coding tasks routinely need well over a hundred tool rounds.
const DEFAULT_MAX_TURNS: u32 = 150;
/// Daemon tools enforce their own timeouts (terminal backgrounding, CLI
/// agents); this only stops a wedged call from holding the turn forever.
const TOOL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const HEARTBEAT_TICK: Duration = Duration::from_secs(15);
/// The run ended on a terminal tool result (background handoff, executor
/// blocker, direct response) whose reply ships as is.
const TERMINAL_REPLY_STOP: &str = "terminal tool result";
/// Prefix of the outbound secret scan's refusals (`EgressGuard`).
const EGRESS_BLOCKED: &str = "[EGRESS BLOCKED]";
/// How turns run on the SDK kernel. Held by the root agent and inherited by
/// every child it spawns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SdkEngineSettings {
    /// `[agent.engine] model`: pins every turn to this model instead of the
    /// daemon's own selection.
    pub(crate) model_override: Option<String>,
    pub(crate) max_turns: u32,
    /// `[authority] owner_default`: the owner's level when no session mode
    /// sets one.
    pub(crate) owner_default: crate::config::OwnerLevel,
    /// `[agent] task_assessment`: whether the pre-run assessment runs
    /// (`off` skips it) and whether would-refuse comparisons are recorded.
    pub(crate) task_assessment: crate::config::TaskAssessmentSetting,
    /// `[authority] trusted_roots`: directories whose files are the owner's
    /// own, so reading them is not untrusted content.
    pub(crate) trusted_roots: Vec<String>,
    /// The configured API keys and tokens the outbound secret scan refuses.
    pub(crate) outbound_secrets: super::approvals::ConfiguredSecrets,
}

impl Default for SdkEngineSettings {
    fn default() -> Self {
        Self {
            model_override: None,
            max_turns: DEFAULT_MAX_TURNS,
            owner_default: crate::config::OwnerLevel::default(),
            task_assessment: crate::config::TaskAssessmentSetting::default(),
            trusted_roots: Vec::new(),
            outbound_secrets: Default::default(),
        }
    }
}

/// The engine settings from `[agent.engine]`.
pub(crate) fn settings_from_config(config: &AppConfig) -> SdkEngineSettings {
    let engine = &config.agent.engine;
    if engine.kind.is_some() {
        warn!("[agent.engine] kind is ignored: the SDK kernel is the only agent loop");
    }
    if config.state.daily_token_budget.is_some() {
        warn!(
            "[state] daily_token_budget is ignored: no global daily token cap is enforced; \
             use per-goal daily budgets, subagents.task_token_budget or a provider spend limit"
        );
    }
    let mut model_override = engine
        .model
        .clone()
        .filter(|model| !model.trim().is_empty());
    if !engine.api_key.trim().is_empty() {
        warn!(
            "[agent.engine] api_key is ignored: the SDK engine uses the daemon's configured \
             providers"
        );
        // An engine api_key marks a config written for the engine's former
        // dedicated Anthropic client; its model pin named a model for that
        // client, which the daemon's own provider may not serve.
        if config.provider.kind != ProviderKind::Anthropic {
            if let Some(model) = model_override.take() {
                warn!(
                    model = %model,
                    "[agent.engine] model is ignored: it was configured for the removed \
                     Anthropic engine client; set it without api_key to pin a model the \
                     daemon's provider serves"
                );
            }
        }
    }
    let settings = SdkEngineSettings {
        model_override,
        max_turns: engine.max_turns.unwrap_or(DEFAULT_MAX_TURNS).max(1),
        owner_default: config.authority.owner_default,
        task_assessment: config.agent.task_assessment,
        trusted_roots: config.authority.trusted_roots.clone(),
        outbound_secrets: super::approvals::ConfiguredSecrets::new(config.outbound_secret_values()),
    };
    info!(model_override = ?settings.model_override, "Agent engine: SDK kernel");
    log_configured_trust_roots(&settings.trusted_roots);
    settings
}

/// Where a turn's trusted file roots come from. Trusted roots, like
/// authority, come only from explicit owner grants: they are trusted only by
/// explicit grant, never by inference from the request. A path the request
/// or a mission names (the primary project scope, the assessor's execution
/// cwd, a scope a parent handed down) may steer where tools run, but it
/// never vouches for content: callers pass only explicit values here.
#[derive(Default)]
pub(in crate::agent) struct TrustRootSources<'a> {
    /// Owner configuration: `[authority] trusted_roots`
    /// (`Agent::configured_trust_roots`). Path aliases are not grants.
    pub configured: &'a [String],
    /// The execution workspace (`turn_cwd()`, from config).
    pub cwd: Option<&'a str>,
    /// The turn's explicit workspace binding (`TurnContext::explicit_workspace`:
    /// a goal's `bind_workspace`), never an inferred one.
    pub bound_workspace: Option<&'a str>,
    /// The project root of the chat's active workspace grant (guest turns
    /// in a granted group only).
    pub workspace_grant: Option<&'a str>,
    /// The resolved authority's confinement, passed only alongside an
    /// explicit binding (its write targets derive from it).
    pub task_access: Option<&'a crate::traits::ToolCallAccessManifest>,
    /// The spawning parent turn's trusted roots: the child receives them and
    /// never trusts a file outside them. `None` for a root turn.
    pub parent_roots: Option<&'a [String]>,
}

/// The owner's trusted file roots for one turn, from explicit grants only
/// (see [`TrustRootSources`]). Every source must be a narrow root (after
/// `~` expansion): the filesystem root, the home directory or a shared
/// temp root would trust every file the owner has. A child's roots are its
/// parent's, plus its own sources that lie inside them.
pub(in crate::agent) fn turn_trust_roots(sources: TrustRootSources<'_>) -> Vec<String> {
    let TrustRootSources {
        configured,
        cwd,
        bound_workspace,
        workspace_grant,
        task_access,
        parent_roots,
    } = sources;
    let write_targets = task_access
        .into_iter()
        .flat_map(|access| access.write_targets.iter())
        .map(|target| target.value.as_str());
    let mut roots: Vec<String> = configured
        .iter()
        .map(String::as_str)
        .chain(
            [cwd, bound_workspace, workspace_grant]
                .into_iter()
                .flatten(),
        )
        .chain(write_targets)
        .map(|root| shellexpand::tilde(root.trim()).into_owned())
        .filter(|root| is_narrow_root_resolved(root))
        .collect();
    if let Some(parent) = parent_roots {
        let within = crate::traits::TrustRoots::new(parent);
        roots.retain(|root| within.contains(root));
        roots.splice(0..0, parent.iter().cloned());
    }
    let mut seen = std::collections::HashSet::new();
    roots.retain(|root| seen.insert(root.clone()));
    roots
}

impl Agent {
    /// The trusted roots the owner configured: `[authority] trusted_roots`.
    /// `[path_aliases] projects` is a naming convenience, not a grant.
    pub(in crate::agent) fn configured_trust_roots(&self) -> Vec<String> {
        self.sdk_engine.trusted_roots.clone()
    }
}

/// Log, once at startup, the trusted roots the owner configured plus the
/// execution workspace, and which of them are dropped as too broad.
fn log_configured_trust_roots(configured: &[String]) {
    let workspace = turn_cwd();
    let (kept, dropped): (Vec<String>, Vec<String>) = configured
        .iter()
        .map(|root| shellexpand::tilde(root.trim()).into_owned())
        .chain(std::iter::once(workspace))
        .partition(|root| is_narrow_root_resolved(root));
    info!(
        trusted_roots = ?kept,
        dropped_as_too_broad = ?dropped,
        "Trusted roots: [authority] trusted_roots and the execution workspace \
         (a goal's bind_workspace and a channel workspace grant add to these per turn)"
    );
}

/// `is_narrow_root` for both the root as written and, when it exists, the
/// directory it resolves to: a link to the home directory or `/` is as
/// broad as its target.
fn is_narrow_root_resolved(root: &str) -> bool {
    is_narrow_root(root)
        && std::fs::canonicalize(root)
            .map(|resolved| is_narrow_root(&resolved.to_string_lossy()))
            .unwrap_or(true)
}

/// A directory that holds only the owner's work: not the filesystem root,
/// the home directory or any of its ancestors (`/Users`), nor a shared temp
/// root (`/tmp`, `/private/tmp`, `/var/tmp`, the system temp directory).
/// Folders below them are fine.
fn is_narrow_root(dir: &str) -> bool {
    let dir = std::path::Path::new(dir.trim().trim_end_matches('/'));
    if !dir.is_absolute() || dir.parent().is_none() {
        return false;
    }
    if dirs::home_dir().is_some_and(|home| home.starts_with(dir)) {
        return false;
    }
    let temp = std::env::temp_dir();
    let mut shared: Vec<std::path::PathBuf> = ["/tmp", "/private/tmp", "/var/tmp"]
        .iter()
        .map(std::path::PathBuf::from)
        .collect();
    shared.extend(std::fs::canonicalize(&temp).ok());
    shared.push(temp);
    !shared.iter().any(|root| root.as_path() == dir)
}

/// Where the file tools resolve a relative path: the execution backend's
/// workspace (the daemon's working directory on the local backend).
pub(in crate::agent) fn turn_cwd() -> String {
    crate::execution::active_execution_backend()
        .workspace_root()
        .as_str()
        .to_string()
}

/// Everything the SDK run asks of the daemon, served on the turn's own task.
/// What the kernel's event sink hands to the turn's dispatcher.
enum KernelNotice {
    /// A call the kernel refused before dispatch.
    Refusal(sdk::Receipt),
    /// The stuck detector fired.
    Stuck,
}

pub(super) enum DaemonCall {
    Tool(ToolRequest),
    Model(ModelCall),
    Stop(StopCheck),
}

/// What turn setup (`main_loop.rs`) settled before the SDK run starts:
/// the recorded task, the assessed contract, the offered tools, the selected
/// model and the system prompt.
pub(in crate::agent) struct SdkTurnSetup {
    /// The persisted user text (bootstrap may have added a transcript).
    pub(in crate::agent) user_text: String,
    pub(in crate::agent) task_id: String,
    pub(in crate::agent) emitter: EventEmitter,
    pub(in crate::agent) offered_tools: HashSet<String>,
    pub(in crate::agent) model: String,
    /// The transcript `message_build` projected for the run's first model
    /// call, in the daemon providers' wire format.
    pub(in crate::agent) prefix: Vec<Value>,
    pub(in crate::agent) contract: crate::agent::completion_contract::CompletionContract,
    /// Post-task learning state from bootstrap; the run adds its tool calls.
    pub(in crate::agent) learning_ctx: crate::agent::post_task::LearningContext,
    /// The assessed task shape continues inline work after a background start.
    pub(in crate::agent) continue_inline_after_background: bool,
    /// Matched skills reference untrusted external content: tools that could
    /// act on it are blocked, naming these skills.
    pub(in crate::agent) restrict_untrusted_reference_tools: bool,
    pub(in crate::agent) untrusted_reference_skills: Vec<String>,
    /// Discovers nested project instructions before deeper subtree work.
    pub(in crate::agent) instruction_tracker:
        Option<crate::project_instructions::ProjectInstructionTracker>,
    /// The compiled memory policy allows recall on this turn.
    pub(in crate::agent) memory_recall_allowed: bool,
    /// Where calls without a directory run: the contract's execution cwd,
    /// else the request's primary project scope (a scheduled goal's bound
    /// workspace). `None` when a channel workspace grant scopes the file
    /// tools itself.
    pub(in crate::agent) known_project_dir: Option<String>,
    /// The turn's explicit workspace binding (`TurnContext::explicit_workspace`),
    /// a trusted root. An inferred binding is never passed here.
    pub(in crate::agent) explicit_workspace: Option<String>,
    /// The single-project scope lock, passed to tools as `_project_scope`.
    pub(in crate::agent) project_scope: Option<String>,
    /// What this turn may do, resolved from explicit facts: the kernel's
    /// `AuthorityHook` enforces its level and tool set, and the
    /// controller-side scope check judges each call's access manifest
    /// against its `task_access`, else the directories the turn authorizes.
    pub(in crate::agent) authority: crate::agent::authority::ResolvedAuthority,
    pub(in crate::agent) authorized_scopes: Vec<String>,
    /// Persisted message and turn ids whose content reached the projected
    /// prefix, recorded on every model call's telemetry.
    pub(in crate::agent) projected_source_message_ids: Vec<String>,
    pub(in crate::agent) projected_source_turn_ids: Vec<String>,
}

/// The model wants to stop; the dispatcher asks the run ledger.
pub(super) struct StopCheck {
    /// Operations of this run that completed.
    completed_results: usize,
    /// The run's latest tool call failed and none succeeded.
    failed_without_success: bool,
    final_text: String,
    reply: oneshot::Sender<sdk::StopDecision>,
}

/// One tool call handed from the SDK run to the daemon dispatcher.
pub(super) struct ToolRequest {
    name: String,
    arguments: Value,
    call_id: String,
    reply: oneshot::Sender<sdk::ToolResult>,
}

/// Encode the image attachments `model` may see. Returns the images and
/// whether an eligible image was left out (vision off, a model without
/// vision, or a file that is unreadable or over the size cap).
fn sdk_images(
    attachments: &[&MessageAttachment],
    vision: &VisionConfig,
    model: &str,
) -> (Vec<sdk::Image>, bool) {
    let eligible: Vec<_> = attachments
        .iter()
        .filter(|attachment| vision.mime_allowed(&attachment.mime_type))
        .collect();
    if eligible.is_empty() {
        return (Vec::new(), false);
    }
    if !vision.model_supports_vision(model) {
        return (Vec::new(), true);
    }
    let mut images = Vec::new();
    let mut skipped = false;
    for attachment in eligible {
        match encode_image_attachment(
            Path::new(&attachment.local_path),
            &attachment.mime_type,
            vision.max_image_bytes,
        ) {
            Ok(data) => images.push(sdk::Image::base64(attachment.mime_type.clone(), data)),
            Err(error) => {
                skipped = true;
                warn!(
                    path = %attachment.local_path,
                    mime = %attachment.mime_type,
                    %error,
                    "SDK engine: image not sent to the model"
                );
            }
        }
    }
    (images, skipped)
}

/// The goal whose token budget this turn charges.
struct BudgetGoal {
    goal_id: String,
    /// Scheduled runs charge their per-run budget instead of the daily one.
    scheduled: bool,
}

impl TurnScope<'_> {
    /// End the run before the next model call and ship `reply`.
    fn end_with(&self, reply: String, outcome: TaskOutcome) {
        if let Ok(mut terminal) = self.verdict.terminal_reply.lock() {
            *terminal = Some((reply, outcome));
        }
        self.end_run.store(true, Ordering::Relaxed);
    }
}

/// Everything a dispatched call needs from the turn.
pub(in crate::agent) struct TurnScope<'a> {
    pub(in crate::agent) session_id: &'a str,
    /// The user's message for this turn.
    user_text: &'a str,
    pub(in crate::agent) task_id: &'a str,
    pub(in crate::agent) emitter: &'a EventEmitter,
    /// The projected starting transcript, sent byte-for-byte on every call.
    prefix: Vec<Value>,
    /// Kernel messages that stand for `prefix`; later ones are this run's.
    prefix_len: usize,
    budget_goal: Option<BudgetGoal>,
    /// Set once the goal budget is used up; later model calls are refused.
    budget_exhausted: AtomicBool,
    model_calls: AtomicU64,
    /// The verify-once and failure-recovery passes have been used.
    verification_demanded: AtomicBool,
    recovery_demanded: AtomicBool,
    /// Extra passes the ledger demanded so far.
    closeout_demands: std::sync::atomic::AtomicU32,
    /// `completed_results` at the last checklist demand (`u64::MAX`: none).
    executor_demanded_at: AtomicU64,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    channel_ctx: &'a ChannelContext,
    user_role: UserRole,
    /// The engine model, for vision gating of tool-observed images.
    model: &'a str,
    turn_id: &'a str,
    /// The assessed contract: its expectations are credited from receipts;
    /// its authority view is only compared in shadow logging.
    contract: &'a crate::agent::completion_contract::CompletionContract,
    progress: std::sync::Mutex<crate::agent::completion_contract::CompletionProgress>,
    /// Shared with the memory writer, which learns from it at run end.
    learning: Arc<std::sync::Mutex<crate::agent::post_task::LearningContext>>,
    /// The model-visible output of the latest tool call, for paste checks.
    latest_tool_output: std::sync::Mutex<Option<String>>,
    /// The one reply retry has been spent.
    reply_retry_used: AtomicBool,
    /// The next model call runs with tools disabled (empty-reply retry).
    text_only_next_call: AtomicBool,
    /// Consecutive responses cut off at the token limit with no output (the
    /// whole budget went to thinking). The next call first gets more output
    /// room; only repeated cutoffs lower reasoning effort.
    thinking_truncations: std::sync::atomic::AtomicU8,
    /// Raised output-token ceiling after a thinking cutoff (0: none).
    raised_output_tokens: std::sync::atomic::AtomicU32,
    /// The assessed task shape keeps inline work going after a detach.
    continue_inline_after_background: bool,
    /// Where calls without a directory run; refined by the model's own
    /// explicit directories (see `SdkTurnSetup::known_project_dir`).
    known_project_dir: std::sync::Mutex<Option<String>>,
    /// The single-project scope lock passed to tools.
    project_scope: Option<String>,
    /// See `SdkTurnSetup::authority`.
    authority: &'a crate::agent::authority::ResolvedAuthority,
    /// The chat's live `/readonly` ceiling, read by the hook per call and by
    /// the dispatcher's `mutation_forbidden`.
    pub(in crate::agent) ceiling: Arc<crate::agent::authority::SessionModeCeiling>,
    /// The owner's trusted file roots: configured roots, the working
    /// directory, the project scope and directory, and the task's write
    /// grants.
    trust_roots: Vec<String>,
    /// Where relative file paths resolve (see `turn_cwd`).
    cwd: String,
    /// A dispatched call returned untrusted content this turn.
    untrusted_seen: AtomicBool,
    /// The root request egress is judged against (a child's is its root's),
    /// and whether the owner wrote it.
    origin_request: &'a str,
    origin_trusted: bool,
    authorized_scopes: Vec<String>,
    /// See `SdkTurnSetup::projected_source_message_ids`.
    projected_source_message_ids: Vec<String>,
    projected_source_turn_ids: Vec<String>,
    /// Terminal replies, external acks and run failures: the turn's verdict.
    verdict: Arc<TurnVerdict>,
    /// Daemon memory tools the active memory plugin owns on this turn.
    owned_memory_tools: &'static [&'static str],
    /// Tool calls in the latest model response (a direct response needs one).
    last_batch_size: std::sync::atomic::AtomicUsize,
    /// Ends the run at the next turn boundary (a background handoff).
    end_run: Arc<AtomicBool>,
    /// Skills referencing untrusted external content are active.
    restrict_untrusted_reference_tools: bool,
    untrusted_reference_skills: Vec<String>,
    /// Billable tokens this run spent, and the task budget (0: none).
    task_tokens: AtomicU64,
    task_budget: AtomicU64,
    task_budget_extensions: std::sync::atomic::AtomicUsize,
    /// Distinct successful operations (tool and canonical arguments).
    succeeded_operations: std::sync::Mutex<HashSet<String>>,
    /// The latest successful call repeated an earlier operation.
    repeating: AtomicBool,
    /// Distinct successful operations so far, and the count at the last
    /// extension.
    succeeded_calls: std::sync::atomic::AtomicUsize,
    successes_at_task_extension: std::sync::atomic::AtomicUsize,
    /// The goal's effective daily budget after extensions, and their count.
    goal_daily_budget: std::sync::Mutex<Option<i64>>,
    goal_budget_extensions: std::sync::atomic::AtomicU32,
    /// Mandate decision correction turns used.
    mandate_commit_retries: std::sync::atomic::AtomicU32,
    /// Verify-before-claim demands for an unverified GUI coordinate click.
    #[cfg(feature = "computer_use")]
    gui_verify_demands: std::sync::atomic::AtomicU32,
    /// The one 80% budget warning (task or scheduled run) was given.
    budget_warning_sent: AtomicBool,
    /// Timed-out model calls retried in this run.
    model_call_timeouts: std::sync::atomic::AtomicU32,
    /// The latest model call was rejected as too long for the model's
    /// window (now recorded); the call is retried once compacted.
    context_overflowed: AtomicBool,
    context_overflow_retries: std::sync::atomic::AtomicU32,
    /// `send_file` dedupe keys that already succeeded in this task.
    sent_files: std::sync::Mutex<HashSet<String>>,
    /// Nested project instructions discovered just in time.
    instruction_tracker:
        Option<tokio::sync::Mutex<crate::project_instructions::ProjectInstructionTracker>>,
    /// System context raised during the run and not yet sent; the next model
    /// call pins it at the run's current end (`placed_directives`).
    extra_system: std::sync::Mutex<Vec<String>>,
    /// Directives and mid-turn user messages, with the run position they
    /// arrived at.
    placed_directives: std::sync::Mutex<Vec<(usize, Value)>>,
    /// The run's summarized older rounds, once it outgrew the context.
    compaction: std::sync::Mutex<Option<compaction::RunCompaction>>,
    /// Set when a call moved work to the background with completion
    /// notifications on: the daemon will deliver its result later, so the
    /// reply is a background handoff, not the terminal answer.
    background_handoff: AtomicBool,
}

impl Agent {
    async fn record_kernel_notice(&self, scope: &TurnScope<'_>, notice: KernelNotice) {
        match notice {
            KernelNotice::Refusal(receipt) => self.record_kernel_refusal(scope, receipt).await,
            KernelNotice::Stuck => {
                self.with_harness_eval(|eval| eval.record_repetition_guard())
                    .await;
            }
        }
    }

    /// Record a call the kernel refused before dispatch in the run ledger:
    /// an unknown tool or invalid arguments is a typed failed proposal; a
    /// hook denial is a policy denial.
    async fn record_kernel_refusal(&self, scope: &TurnScope<'_>, receipt: sdk::Receipt) {
        let call = crate::traits::ToolCall {
            id: receipt.call_id.clone(),
            name: receipt.tool.clone(),
            arguments: receipt.arguments.to_string(),
            extra_content: None,
        };
        let reason = receipt
            .reason
            .clone()
            .unwrap_or_else(|| format!("`{}` was refused before it ran", receipt.tool));
        let semantics = self
            .semantics_for_exact_tool_call(&call.name, &call.arguments)
            .await;
        let recorded = if receipt.status == sdk::ReceiptStatus::Denied {
            // Deny hooks and resolutions on SDK turns: the authority hook,
            // whose refusals carry its fixed marker, the untrusted-content
            // verifier (its own marker), the owner's declines (the kernel's
            // marker), the outbound secret scan, and the same-tool limit.
            let authority_denial = reason.starts_with("[AUTHORITY BLOCKED]");
            let untrusted_denial = reason.starts_with(crate::agent::untrusted::UNTRUSTED_BLOCKED);
            let owner_declined = reason.starts_with(sdk::OWNER_DECLINED);
            let egress_blocked = reason.starts_with(EGRESS_BLOCKED);
            if authority_denial {
                if !receipt.read_only {
                    self.with_harness_eval(|eval| eval.record_forbidden_mutation_attempt())
                        .await;
                }
            } else if egress_blocked {
                self.emit_decision_point(
                    scope.emitter,
                    scope.task_id,
                    scope.model_calls.load(Ordering::Relaxed) as usize,
                    crate::events::DecisionType::GateTelemetry,
                    format!("outbound secret blocked in `{}`", receipt.tool),
                    serde_json::json!({
                        "condition": "egress_secret_blocked",
                        "tool": receipt.tool,
                        "reason": reason,
                    }),
                )
                .await;
            } else if !untrusted_denial && !owner_declined {
                self.with_harness_eval(|eval| eval.record_repetition_guard())
                    .await;
            }
            self.persist_pre_dispatch_policy_denial(
                scope.emitter,
                scope.session_id,
                self.ledger_task_id(scope),
                &call,
                &call.arguments,
                reason,
                if authority_denial {
                    "authority_hook_denied"
                } else if untrusted_denial {
                    "untrusted_egress_denied"
                } else if owner_declined {
                    "owner_approval_declined"
                } else if egress_blocked {
                    "egress_secret_blocked"
                } else {
                    "kernel_hook_denied"
                },
                semantics,
                None,
            )
            .await
        } else {
            self.persist_pre_dispatch_outcome_with_kernel_claim(
                scope.emitter,
                scope.session_id,
                self.ledger_task_id(scope),
                &call,
                &call.arguments,
                reason,
                crate::traits::ToolOutcomeStatus::FailedPermanent,
                crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
                false,
                semantics,
                None,
                None,
                None,
            )
            .await
        };
        if let Err(error) = recorded {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a kernel refusal");
        }
    }

    /// The task an operation is ledgered on: a fenced mandate worker's
    /// durable worker task (it belongs to the goal run, where run-scoped
    /// receipt checks join), else the turn's task. Every tool call and its
    /// result use this one id so a call never sits apart from its result.
    fn ledger_task_id<'s>(&'s self, scope: &TurnScope<'s>) -> &'s str {
        self.mandate_execution
            .as_ref()
            .map_or(scope.task_id, |fence| fence.worker_task_id.as_str())
    }

    /// The outbound secret scan: the SDK's secret shapes plus the daemon's
    /// configured API keys and tokens and those of dynamically connected
    /// bots.
    async fn egress_guard(&self, engine: &SdkEngineSettings) -> aidaemon_policies::EgressGuard {
        let mut guard = aidaemon_policies::EgressGuard::new();
        for secret in engine.outbound_secrets.values() {
            guard = guard.secret(secret.clone());
        }
        match self.state.get_dynamic_bots().await {
            Ok(bots) => {
                for bot in bots {
                    guard = guard.secret(bot.bot_token);
                    if let Some(app_token) = bot.app_token {
                        guard = guard.secret(app_token);
                    }
                }
            }
            Err(error) => warn!(%error, "Could not read dynamic bots for the egress guard"),
        }
        guard
    }

    /// One turn on the SDK kernel, after turn setup built `setup`. See the
    /// module docs for what the daemon owns and what the kernel drives.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_message_sdk(
        &self,
        engine: &SdkEngineSettings,
        setup: SdkTurnSetup,
        session_id: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String> {
        let started = Instant::now();
        let SdkTurnSetup {
            user_text,
            task_id,
            emitter,
            offered_tools,
            model,
            prefix,
            contract,
            learning_ctx,
            continue_inline_after_background,
            instruction_tracker,
            restrict_untrusted_reference_tools,
            untrusted_reference_skills,
            memory_recall_allowed,
            known_project_dir,
            project_scope,
            explicit_workspace,
            authority,
            authorized_scopes,
            projected_source_message_ids,
            projected_source_turn_ids,
        } = setup;
        let user_text = user_text.as_str();
        let model = engine.model_override.clone().unwrap_or(model);
        touch_heartbeat(&heartbeat);
        let cwd = turn_cwd();
        // Trusted only by explicit grant, never by inference from the
        // request: `project_scope` and `known_project_dir` steer where tools
        // run but are not trust sources.
        let configured_roots = self.configured_trust_roots();
        let trust_roots = turn_trust_roots(TrustRootSources {
            configured: &configured_roots,
            cwd: Some(&cwd),
            bound_workspace: explicit_workspace.as_deref(),
            workspace_grant: channel_ctx
                .active_workspace_grant(user_role)
                .map(|grant| grant.project_root.as_str()),
            task_access: authority
                .task_access
                .as_ref()
                .filter(|_| explicit_workspace.is_some()),
            parent_roots: self
                .inherited_authority
                .as_ref()
                .map(|parent| parent.trusted_roots.as_slice()),
        });
        let turn_id = self
            .current_turn_ids
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let (calls_tx, mut calls_rx) = mpsc::channel::<DaemonCall>(16);
        // Calls the kernel refuses itself (unknown tool, invalid arguments, a
        // hook denial) never reach the daemon dispatcher; the event sink hands
        // their receipts over so the run ledger records them too.
        let (refusals_tx, mut refusals_rx) = mpsc::unbounded_channel::<KernelNotice>();
        // The turn's resolved authority is enforced by the kernel hook, the
        // outbound secret scan by `EgressGuard`, the untrusted-content rule
        // by `UntrustedHook` and owner approvals by `ApprovalPolicy` (all in
        // the pinned order of `TurnHooks`, below). The daemon dispatcher
        // still enforces Scope itself:
        // `sdk_scope_check` against the turn's filesystem grant.
        // `/readonly` sent while this turn runs narrows it: the hook and the
        // dispatcher both read the chat's live mode (a child reads its
        // parent's chat through `mode_session_for`).
        let ceiling = Arc::new(crate::agent::authority::SessionModeCeiling::new(
            self.state.clone(),
            self.mode_session_for(session_id),
        ));
        let authority_hook =
            sdk::AuthorityHook::new(authority.authority.clone()).with_ceiling(ceiling.clone());
        let mut builder = sdk::Agent::builder("aidaemon")
            .attest(sdk::Safeguard::Scope)
            .model(model.clone())
            .provider(DaemonProvider {
                calls: calls_tx.clone(),
            })
            .limits(sdk::Limits {
                max_turns: engine.max_turns,
                tool_timeout: TOOL_TIMEOUT,
                ..sdk::Limits::default()
            });
        let memory_persistence_allowed = learning_ctx.memory_persistence_allowed;
        let learning = Arc::new(std::sync::Mutex::new(learning_ctx));
        let verdict = Arc::new(TurnVerdict::default());
        // Memory comes from the active plugin; a mandate worker gets none.
        let memory_turn = MemoryTurn {
            session_id: session_id.to_string(),
            task_id: task_id.clone(),
            emitter: emitter.clone(),
            user_text: user_text.to_string(),
            user_role,
            channel_ctx: channel_ctx.clone(),
            memory_pipeline_allowed: memory_recall_allowed,
            memory_persistence_allowed,
            mandate: self.mandate_execution.is_some(),
            depth: self.depth,
            model: model.clone(),
            learning: learning.clone(),
            verdict: verdict.clone(),
        };
        let memory = memory_parts(self.memory.as_ref(), &memory_turn).await;
        // Tools: the set bootstrap offered for this turn (role, policy and
        // contract filtering already applied), with the daemon memory tools
        // only while the active plugin owns them; `manage_memories` keeps
        // its goal actions under any plugin.
        let bridge_roots = Arc::new(trust_roots.clone());
        if user_role != UserRole::Public {
            for tool in self
                .tools
                .iter()
                .filter(|tool| offered_tools.contains(tool.name()))
                // A tool whose backend is down is not offered (the offered
                // set already drops it; this keeps the bridge consistent).
                .filter(|tool| tool.is_available())
            {
                let (spec, goal_actions_only) =
                    match daemon_tool_offer(tool.name(), memory.owned_daemon_tools) {
                        DaemonToolOffer::Full => (tool_spec(tool.as_ref()), false),
                        DaemonToolOffer::GoalActions => (
                            tool_spec_from_schema(
                                tool.as_ref(),
                                &crate::agent::memory_plugin::goal_actions_view(&tool.schema()),
                            ),
                            true,
                        ),
                        DaemonToolOffer::Hidden => continue,
                    };
                builder = builder.tool(BridgedTool {
                    spec,
                    tool: tool.clone(),
                    goal_actions_only,
                    calls: calls_tx.clone(),
                    trust_roots: bridge_roots.clone(),
                });
            }
        }
        // A background handoff ends the run before the next model call.
        let end_run = Arc::new(AtomicBool::new(false));
        let end_flag = end_run.clone();
        builder = builder.stop_when(move |_: &sdk::RunSnapshot| {
            end_flag
                .load(Ordering::Relaxed)
                .then(|| TERMINAL_REPLY_STOP.to_string())
        });
        // Memory hooks run after the ledger stop, so a rejected final answer
        // never reaches a memory writer. Summarization and learning belong to
        // the turn, so they run whatever the memory plugin.
        let owned_memory_tools = memory.owned_daemon_tools;
        let mut memory_hooks: Vec<Arc<dyn sdk::Hook>> = Vec::new();
        if !memory_turn.mandate {
            memory_hooks.push(Arc::new(
                crate::agent::memory_plugin::DaemonMemoryWriter::new(
                    memory_turn.clone(),
                    crate::agent::memory_plugin::WriterSettings {
                        progressive_facts: false,
                        summarization: self.context_window_config.enabled,
                        learning: true,
                    },
                    self.transcript_writes.clone(),
                ),
            ));
        }
        memory_hooks.extend(memory.hooks);
        if user_role != UserRole::Public {
            for tool in memory.tools {
                builder = builder.tool_arc(tool);
            }
        }
        // Owner approvals: the daemon's rules for this turn and the chat's
        // grants (a child shares its parent chat's).
        let grant_session = self
            .approval_session_id
            .clone()
            .unwrap_or_else(|| session_id.to_string());
        let mandate_run = self.mandate_execution.is_some();
        let rules = super::approvals::AppApprovalRules {
            trusted_session: !mandate_run
                && self
                    .session_trusted(channel_ctx.trusted, Some(task_id.as_str()))
                    .await,
            mandate_preapproved: mandate_run,
        };
        let approvals = self.owner_approvals.clone();
        let approval =
            aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never)
                .with_rules(Arc::new(rules))
                .with_grants(approvals.grants_for(&grant_session));
        let egress = self.egress_guard(engine).await;
        let hooks = TurnHooks {
            authority: authority_hook,
            same_tool: SameToolLimit {
                read_only_tools: read_only_tool_names(self.tools.iter()),
            },
            ledger: LedgerStop {
                calls: calls_tx.clone(),
            },
            memory: memory_hooks,
            egress,
            untrusted: sdk::UntrustedHook::new(
                crate::agent::untrusted::untrusted_mode(authority.owner_level),
                aidaemon_policies::wrap_untrusted,
            ),
            approval,
        };
        for (_, hook) in hooks.chain() {
            builder = builder.hook_arc(hook);
        }
        drop(calls_tx);
        let sink_status = status_tx.clone();
        let sink_heartbeat = heartbeat.clone();
        // The kernel holds one sink: heartbeat, kernel refusals for the
        // ledger, and the thinking status all go through it.
        builder = builder.event_sink(move |_run, event: &sdk::Event| {
            touch_heartbeat(&sink_heartbeat);
            match event {
                sdk::Event::ToolFinished { receipt }
                    if matches!(
                        receipt.status,
                        sdk::ReceiptStatus::UnknownTool
                            | sdk::ReceiptStatus::InvalidArguments
                            | sdk::ReceiptStatus::Denied
                    ) =>
                {
                    let _ = refusals_tx.send(KernelNotice::Refusal(receipt.clone()));
                }
                sdk::Event::Stuck { .. } => {
                    let _ = refusals_tx.send(KernelNotice::Stuck);
                }
                _ => {}
            }
            let Some(status_tx) = sink_status.as_ref() else {
                return;
            };
            let update = match event {
                sdk::Event::TurnStarted { turn } if *turn > 0 => {
                    StatusUpdate::Thinking(*turn as usize)
                }
                _ => return,
            };
            let _ = status_tx.try_send(update);
        });
        let sdk_agent = builder
            .build()
            .map_err(|error| anyhow::anyhow!("SDK engine agent: {error}"))?;

        // The kernel's own view of the starting transcript (for its stuck
        // detection and bookkeeping); model calls send the projected prefix
        // bytes unchanged.
        let mut state = sdk::RunState::new("aidaemon", session_id, sdk::Message::user(user_text));
        let projected = values_to_sdk(&prefix);
        if !projected.is_empty() {
            state.messages = projected;
        }
        // A turn whose input is someone else's, or whose transcript still
        // holds a turn that read untrusted content, starts tainted.
        let tainted_turns = crate::agent::untrusted::tainted_turns(self.state.as_ref(), session_id)
            .await
            .inspect_err(|error| {
                warn!(session_id, %error, "Could not read tainted turns; the turn starts tainted");
            })
            .ok();
        let starts_tainted =
            crate::agent::untrusted::starts_tainted(&crate::agent::untrusted::TaintFacts {
                user_role,
                visibility: channel_ctx.visibility,
                session_id,
                projected_turn_ids: &projected_source_turn_ids,
                tainted_turns: tainted_turns.as_deref(),
                inherited: self
                    .inherited_authority
                    .as_ref()
                    .is_some_and(|parent| parent.untrusted_context),
            });
        if starts_tainted {
            state = state.with_untrusted_context();
        }
        let prefix_len = state.messages.len();

        // Drive the run and the tool dispatcher together on this task.
        let budget_goal = match self.budget_goal_id(session_id).await {
            Some(goal_id) => Some(BudgetGoal {
                scheduled: crate::agent::goal_dispatch::goal_has_scheduled_provenance(
                    &self.state,
                    &goal_id,
                    self.task_id.as_deref(),
                )
                .await,
                goal_id,
            }),
            None => None,
        };
        let budget_goal_for_daily = budget_goal.as_ref().map(|goal| goal.goal_id.clone());
        let scheduled_budget_per_check = match budget_goal.as_ref().filter(|goal| goal.scheduled) {
            Some(goal) => self
                .state
                .get_goal(&goal.goal_id)
                .await
                .ok()
                .flatten()
                .and_then(|goal| goal.budget_per_check),
            None => None,
        };
        // A child judges egress against its root's request, which is the
        // owner's only if every turn up to the root was.
        let inherited = self.inherited_authority.as_ref();
        let origin_request = inherited
            .and_then(|parent| parent.origin_request.as_deref())
            .unwrap_or(user_text);
        let origin_trusted =
            crate::agent::untrusted::origin_is_owner_authored(user_role, session_id)
                && inherited.is_none_or(|parent| parent.origin_trusted);
        let scope = TurnScope {
            session_id,
            user_text,
            task_id: &task_id,
            status_tx: status_tx.clone(),
            channel_ctx: &channel_ctx,
            user_role,
            model: &model,
            turn_id: &turn_id,
            contract: &contract,
            learning: learning.clone(),
            latest_tool_output: std::sync::Mutex::new(None),
            reply_retry_used: AtomicBool::new(false),
            text_only_next_call: AtomicBool::new(false),
            thinking_truncations: std::sync::atomic::AtomicU8::new(0),
            raised_output_tokens: std::sync::atomic::AtomicU32::new(0),
            continue_inline_after_background,
            known_project_dir: std::sync::Mutex::new(known_project_dir),
            project_scope,
            authority: &authority,
            ceiling: ceiling.clone(),
            trust_roots,
            cwd,
            untrusted_seen: AtomicBool::new(starts_tainted),
            origin_request,
            origin_trusted,
            authorized_scopes,
            projected_source_message_ids,
            projected_source_turn_ids,
            instruction_tracker: instruction_tracker.map(tokio::sync::Mutex::new),
            extra_system: std::sync::Mutex::new(Vec::new()),
            placed_directives: std::sync::Mutex::new(Vec::new()),
            compaction: std::sync::Mutex::new(None),
            sent_files: std::sync::Mutex::new(HashSet::new()),
            mandate_commit_retries: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "computer_use")]
            gui_verify_demands: std::sync::atomic::AtomicU32::new(0),
            budget_warning_sent: AtomicBool::new(false),
            model_call_timeouts: std::sync::atomic::AtomicU32::new(0),
            context_overflowed: AtomicBool::new(false),
            context_overflow_retries: std::sync::atomic::AtomicU32::new(0),
            task_tokens: AtomicU64::new(0),
            task_budget: AtomicU64::new(task_token_budget(
                self.limits.task_token_budget,
                scheduled_budget_per_check,
            )),
            task_budget_extensions: std::sync::atomic::AtomicUsize::new(0),
            succeeded_calls: std::sync::atomic::AtomicUsize::new(0),
            succeeded_operations: std::sync::Mutex::new(HashSet::new()),
            repeating: AtomicBool::new(false),
            successes_at_task_extension: std::sync::atomic::AtomicUsize::new(0),
            goal_daily_budget: std::sync::Mutex::new(match &budget_goal_for_daily {
                Some(goal_id) => match &self.goal_token_registry {
                    Some(registry) => registry.get_effective_daily_budget(goal_id).await,
                    None => None,
                },
                None => None,
            }),
            goal_budget_extensions: std::sync::atomic::AtomicU32::new(0),
            restrict_untrusted_reference_tools,
            untrusted_reference_skills,
            verdict: verdict.clone(),
            owned_memory_tools,
            last_batch_size: std::sync::atomic::AtomicUsize::new(0),
            end_run: end_run.clone(),
            progress: std::sync::Mutex::new(
                crate::agent::completion_contract::CompletionProgress::new(&contract, &task_id),
            ),
            background_handoff: AtomicBool::new(false),
            emitter: &emitter,
            prefix,
            prefix_len,
            budget_goal,
            budget_exhausted: AtomicBool::new(false),
            model_calls: AtomicU64::new(0),
            closeout_demands: std::sync::atomic::AtomicU32::new(0),
            verification_demanded: AtomicBool::new(false),
            recovery_demanded: AtomicBool::new(false),
            executor_demanded_at: AtomicU64::new(u64::MAX),
        };
        // A parent or goal cancellation stops the run at its next check.
        let cancel = self
            .cancel_token
            .as_ref()
            .map(|token| token.child_token())
            .unwrap_or_default();
        // Egress the untrusted-content rule interrupts is settled by a clean
        // check that sees only this request and the proposed call.
        // Owner approvals are asked of the owner through the approval
        // broker.
        let resolver = super::approvals::DaemonInterruptResolver {
            untrusted: crate::agent::untrusted::UntrustedVerifier {
                agent: self,
                scope: &scope,
                origin_request,
                origin_trusted,
                cancel: cancel.clone(),
            },
            owner: super::approvals::OwnerApprovalResolver {
                agent: self,
                approvals,
                card_session: session_id,
                grant_session,
                emitter: &emitter,
                task_id: &task_id,
                cancel: cancel.clone(),
            },
        };
        let run = sdk_agent.run_resolving(state, cancel, &resolver);
        tokio::pin!(run);
        let mut in_flight = FuturesUnordered::new();
        // Ledger writes for kernel notices run apart from the run's own calls:
        // they are finished after the run ends, never dropped mid-write.
        let mut ledger_writes = FuturesUnordered::new();
        let mut tick = tokio::time::interval(HEARTBEAT_TICK);
        let outcome = loop {
            tokio::select! {
                outcome = &mut run => break outcome,
                Some(call) = calls_rx.recv() => match call {
                    DaemonCall::Tool(request) => {
                        in_flight.push(self.dispatch_sdk_tool_call(&scope, request).boxed());
                    }
                    DaemonCall::Model(call) => {
                        in_flight.push(self.serve_sdk_model_call(&scope, call).boxed());
                    }
                    DaemonCall::Stop(check) => {
                        in_flight.push(self.serve_sdk_stop(&scope, check).boxed());
                    }
                },
                Some(notice) = refusals_rx.recv() => {
                    ledger_writes.push(self.record_kernel_notice(&scope, notice).boxed());
                }
                Some(()) = in_flight.next(), if !in_flight.is_empty() => {}
                Some(()) = ledger_writes.next(), if !ledger_writes.is_empty() => {}
                _ = tick.tick() => touch_heartbeat(&heartbeat),
            }
        };
        // The run is over: record the notices still queued and finish every
        // ledger write before the task end is computed and appended.
        while let Ok(notice) = refusals_rx.try_recv() {
            ledger_writes.push(self.record_kernel_notice(&scope, notice).boxed());
        }
        while ledger_writes.next().await.is_some() {}
        // A turn that read untrusted content taints later turns that still
        // see it. The dispatcher records it when it first reads untrusted
        // content; this covers seeded turns and the kernel's own tools
        // (recording is idempotent).
        let tainted = scope.untrusted_seen.load(Ordering::SeqCst)
            || outcome
                .as_ref()
                .is_ok_and(|outcome| outcome.state.untrusted_seen);
        if tainted {
            self.read_untrusted.store(true, Ordering::SeqCst);
            if let Err(error) = crate::agent::untrusted::record_tainted_turn(
                self.state.as_ref(),
                session_id,
                &turn_id,
            )
            .await
            {
                warn!(session_id, %error, "Could not record a tainted turn");
            }
        }

        let (reply, status, task_outcome, error) = match outcome {
            Ok(outcome) => {
                let tool_calls = scope
                    .learning
                    .lock()
                    .map(|learning| learning.tool_calls.clone())
                    .unwrap_or_default();
                let (reply, status, task_outcome) =
                    scope
                        .verdict
                        .final_reply(&outcome, &tool_calls, channel_ctx.visibility);
                let run_failure = scope.verdict.run_failure();
                if status == TaskStatus::Completed {
                    self.reconcile_checklist_projection(&emitter, session_id, &task_id)
                        .await;
                }
                info!(
                    session_id,
                    task_id,
                    status = ?outcome.status,
                    turns = outcome.turns(),
                    tool_calls = outcome.state.tool_calls,
                    "SDK engine run finished"
                );
                let usage = outcome.run_usage();
                let mut assistant =
                    Message::new_runtime(Uuid::new_v4().to_string(), session_id, "assistant");
                assistant.content = Some(reply.clone());
                assistant.turn_id = Some(turn_id.clone());
                let input_tokens = Some(usage.input_tokens.min(u32::MAX as u64) as u32);
                let output_tokens = Some(usage.output_tokens.min(u32::MAX as u64) as u32);
                if scope.background_handoff.load(Ordering::Relaxed) {
                    self.append_background_handoff_message_with_event(
                        &emitter,
                        &assistant,
                        &model,
                        input_tokens,
                        output_tokens,
                    )
                    .await?;
                } else {
                    self.append_assistant_message_with_event(
                        &emitter,
                        &assistant,
                        &model,
                        input_tokens,
                        output_tokens,
                    )
                    .await?;
                }
                let error = match &outcome.status {
                    sdk::RunStatus::ProviderFailed { error } => Some(error.clone()),
                    _ => run_failure.clone(),
                };
                (
                    match run_failure {
                        Some(failure) => Err(anyhow::anyhow!(failure)),
                        None => Ok(reply),
                    },
                    status,
                    task_outcome,
                    (error, outcome.turns(), outcome.state.tool_calls),
                )
            }
            Err(error) => {
                let message = error.to_string();
                (
                    Err(anyhow::anyhow!("SDK engine run failed: {message}")),
                    TaskStatus::Failed,
                    TaskOutcome::Failed,
                    (Some(message), 0, 0),
                )
            }
        };
        let (error, iterations, tool_calls_count) = error;
        // The replayed run ledger has the final say on the outcome.
        self.emit_task_end(
            &emitter,
            &task_id,
            status,
            task_outcome,
            started,
            iterations as usize,
            tool_calls_count as usize,
            error,
            None,
        )
        .await;
        // Post-turn memory and learning run in the memory plugin's run-end
        // hook.
        for tool in &self.tools {
            let _ = tool.on_task_end(&task_id, session_id).await;
        }
        self.current_turn_ids.write().await.remove(session_id);
        reply
    }
}

#[cfg(test)]
#[path = "sdk_engine/tests.rs"]
mod tests;
