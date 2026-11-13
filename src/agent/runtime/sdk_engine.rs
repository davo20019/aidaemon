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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_core as sdk;
use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::FutureExt;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use super::agent_helpers::touch_heartbeat;
use super::memory_plugin::{daemon_tool_offer, memory_parts, DaemonToolOffer, MemoryTurn};
use super::sdk_provider::{
    fit_run_values, from_provider_response, prefix_with_notes, split_injected_notes,
    to_chat_options, to_daemon_messages_from, to_daemon_tools, values_to_sdk, DaemonProvider,
    ModelCall,
};
use super::tool_exec::ToolExecCtx;
use super::Agent;
use crate::agent::vision::encode_image_attachment;
use crate::config::{AppConfig, ProviderKind, VisionConfig};
use crate::events::{EventEmitter, EventType, TaskOutcome, TaskStatus};
use crate::tools::ProgressClock;
use crate::traits::{AttachmentProvenance, Message, MessageAttachment};
use crate::types::{ChannelContext, StatusUpdate, UserRole};
use aidaemon_runtime::engine::calls::{
    AfterCallCheck, BudgetCheck, GateRequest, HostCall as DaemonCall, KernelCall, KernelNotice,
    ModelErrorCheck, StopCheck, ToolRequest,
};

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
use aidaemon_runtime::engine::recovery;
#[path = "sdk_engine/stop.rs"]
mod stop;
#[path = "sdk_engine/tool_calls.rs"]
mod tool_calls;

pub(crate) use aidaemon_runtime::engine::outcome::TurnVerdict;
pub(in crate::agent) use aidaemon_runtime::engine::scope::{
    AbsorbedSteering, BudgetGoal, TurnScope,
};
use budgets::*;
use hooks::*;
#[cfg(test)]
use stop::*;

/// Model turns per run. Token budgets, the same-tool guard and the stuck
/// detector bound runaway work; this only stops a run that defeats all three.
/// Long coding tasks routinely need well over a hundred tool rounds.
const DEFAULT_MAX_TURNS: u32 = 150;
use aidaemon_runtime::engine::TOOL_TIMEOUT;
const HEARTBEAT_TICK: Duration = Duration::from_secs(15);
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
            trusted_roots: Vec::new(),
            outbound_secrets: Default::default(),
        }
    }
}

/// The engine settings from `[agent.engine]`.
pub(crate) fn settings_from_config(
    config: &AppConfig,
    backend: &dyn crate::execution::ExecutionBackend,
) -> SdkEngineSettings {
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
    if config.subagents.task_timeout_secs.is_some() {
        warn!(
            "[subagents] task_timeout_secs is ignored: runs are bounded by progress and cost \
             (inactivity watchdog, per-call timeouts, limits, token budgets), not elapsed time"
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
        trusted_roots: config.authority.trusted_roots.clone(),
        outbound_secrets: super::approvals::ConfiguredSecrets::new(config.outbound_secret_values()),
    };
    info!(model_override = ?settings.model_override, "Agent engine: SDK kernel");
    log_configured_trust_roots(
        &settings.trusted_roots,
        backend.workspace_root().as_str().to_string(),
    );
    settings
}

/// Where a turn's trusted file roots come from. Trusted roots, like
/// authority, come only from explicit owner grants: they are trusted only by
/// explicit grant, never by inference from the request. A path the request
/// or a mission names (the primary project scope, a scope a parent
/// handed down) may steer where tools run, but it
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
fn log_configured_trust_roots(configured: &[String], workspace: String) {
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
pub(in crate::agent) fn turn_cwd(backend: &dyn crate::execution::ExecutionBackend) -> String {
    backend.workspace_root().as_str().to_string()
}

/// What turn setup (`main_loop.rs`) settled before the SDK run starts:
/// the recorded task, the structural contract, the offered tools, the selected
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
    /// Matched skills reference untrusted external content: tools that could
    /// act on it are blocked, naming these skills.
    pub(in crate::agent) restrict_untrusted_reference_tools: bool,
    pub(in crate::agent) untrusted_reference_skills: Vec<String>,
    /// Discovers nested project instructions before deeper subtree work.
    pub(in crate::agent) instruction_tracker:
        Option<crate::project_instructions::ProjectInstructionTracker>,
    /// The compiled memory policy allows recall on this turn.
    pub(in crate::agent) memory_recall_allowed: bool,
    /// Whether the turn may be learned from; `/private` opted it out before
    /// it started, and `skip_learning` can during the run.
    pub(in crate::agent) learning: aidaemon_policies::TurnLearning,
    /// The conversation summary standing for the turns `prefix` leaves out,
    /// on a turn that continues the conversation.
    pub(in crate::agent) conversation_summary: Option<crate::traits::ConversationSummary>,
    /// Where calls without a directory run: the contract's execution cwd,
    /// else the request's primary project scope (a scheduled goal's bound
    /// workspace). `None` when a channel workspace grant scopes the file
    /// tools itself.
    pub(in crate::agent) known_project_dir: Option<String>,
    /// The turn's explicit workspace binding (`TurnContext::explicit_workspace`),
    /// a trusted root. An inferred binding is never passed here.
    pub(in crate::agent) explicit_workspace: Option<String>,
    /// The single-project scope lock, passed to tools as
    /// `ToolCallContext::project_scope`.
    pub(in crate::agent) project_scope: Option<String>,
    /// What this turn may do, resolved from explicit facts: the kernel's
    /// `AuthorityHook` enforces its level and tool set, and the
    /// controller-side scope check judges each call's access manifest
    /// against its `task_access`, else the directories the turn authorizes.
    pub(in crate::agent) authority: crate::agent::authority::ResolvedAuthority,
    pub(in crate::agent) authorized_scopes: Vec<String>,
    /// `authorized_scopes` comes only from conversation text, for an owner
    /// who can answer a card: it may not refuse their file tools.
    pub(in crate::agent) scopes_are_guess: bool,
    /// Persisted message and turn ids whose content reached the projected
    /// prefix, recorded on every model call's telemetry.
    pub(in crate::agent) projected_source_message_ids: Vec<String>,
    pub(in crate::agent) projected_source_turn_ids: Vec<String>,
    /// A run that fails on the provider before any tool call is queued for
    /// an automatic re-run (`deferred_retry`). False for a continuation of
    /// another task, which that task's own recovery owns.
    pub(in crate::agent) defer_on_provider_failure: bool,
}

/// What a turn's terminal canonicalizer reads (see [`terminal_canonicalizer`]).
pub(super) struct TerminalCanonTurn {
    pub tool: Arc<dyn crate::traits::Tool>,
    pub known_project_dir: Arc<std::sync::Mutex<Option<String>>>,
    pub session_id: String,
    pub task_access: Option<crate::traits::ToolCallAccessManifest>,
    pub authorized_scopes: Vec<String>,
    pub workspace_grant: bool,
    /// How `CanonicalCallHook` canonicalized each call.
    pub(super) records: tool_calls::CanonicalCalls,
    pub paths: crate::tools::DaemonPaths,
    pub backend: crate::execution::SharedExecutionBackend,
}

/// How the kernel's terminal rules see a call: in the canonical form the
/// dispatcher executes ([`tool_calls::canonicalize_call`]), scope-checked
/// exactly as the dispatcher will check it.
pub(super) fn terminal_canonicalizer(
    turn: TerminalCanonTurn,
) -> impl Fn(sdk::RunId, &str, &Value) -> super::approvals::JudgedTerminal + Send + Sync + 'static {
    move |run, call_id, raw| {
        let (call, known) = match turn.records.get(run, call_id) {
            // Already canonical: judged as sent, against the directory the
            // hook recorded for it.
            Some(record) => (
                crate::agent::approvals::CanonicalCall {
                    arguments: raw.clone(),
                    explicit_dir: record.explicit_dir,
                    injected_dir: record.injected_dir,
                    stripped_read_roots: Vec::new(),
                },
                record.known_project_dir,
            ),
            None => {
                let known = turn
                    .known_project_dir
                    .lock()
                    .ok()
                    .and_then(|known| known.clone());
                let call = tool_calls::canonicalize_call(
                    Some(turn.tool.as_ref()),
                    "terminal",
                    raw,
                    &tool_calls::CanonFacts {
                        backend: &*turn.backend,
                        workspace_grant: turn.workspace_grant,
                        known_project_dir: known.as_deref(),
                        task_access: turn.task_access.as_ref(),
                        authorized_scopes: &turn.authorized_scopes,
                        paths: &turn.paths,
                    },
                );
                // `known` as the dispatcher's scope check reads it, after
                // this call's explicit directory refined it.
                let known = call.explicit_dir.clone().or(known);
                (call, known)
            }
        };
        let facts = tool_calls::CallScopeFacts {
            backend: &*turn.backend,
            session_id: &turn.session_id,
            task_access: turn.task_access.as_ref(),
            authorized_scopes: &turn.authorized_scopes,
            scopes_are_guess: false,
            known_project_dir: known.as_deref(),
            paths: &turn.paths,
        };
        let scope = match tool_calls::call_scope_check(
            turn.tool.as_ref(),
            &call.arguments.to_string(),
            &facts,
        ) {
            Some(crate::agent::tool_execution::ScopeCheck::Escalatable { escalation, .. }) => {
                super::approvals::TerminalScope::Escalate(escalation)
            }
            Some(crate::agent::tool_execution::ScopeCheck::Rejected(reason)) => {
                super::approvals::TerminalScope::Refused(reason)
            }
            None => super::approvals::TerminalScope::InScope,
        };
        #[cfg(test)]
        let scope = match super::approvals::judgment_seam(&turn.session_id) {
            Some(super::approvals::JudgmentSeam::JudgedScope(judged)) => judged,
            _ => scope,
        };
        super::approvals::JudgedTerminal { call, scope }
    }
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

/// The kernel's tool-name rule: 1-64 characters from `[a-zA-Z0-9_-]`.
fn kernel_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
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
        // The dispatcher's gate recorded its own refusal, with its reason.
        if receipt.status == sdk::ReceiptStatus::Denied
            && scope
                .gate_outcomes
                .refused(receipt.run_id, &receipt.call_id)
        {
            return;
        }
        // A refused call may be refused because it carries a secret: the
        // ledger never stores its arguments unredacted.
        let call = crate::traits::ToolCall {
            id: receipt.call_id.clone(),
            name: receipt.tool.clone(),
            arguments: scope.ledger_redactor.redact(&receipt.arguments.to_string()),
            extra_content: None,
        };
        let reason = receipt
            .reason
            .clone()
            .unwrap_or_else(|| format!("`{}` was refused before it ran", receipt.tool));
        let semantics = self.sdk_call_semantics(scope, &call.name, &call.arguments);
        // Stale writes are retried by the writer and land, and a write kept
        // off a sibling's lines is the tree settling its own work: a failed
        // attempt, not something the owner refused. A child that never makes
        // the change ends `Partial` through the tree's own record.
        let retryable_collision = receipt.status == sdk::ReceiptStatus::Denied
            && (reason.starts_with(crate::agent::coordination::STALE_WRITE)
                || reason.starts_with(crate::agent::coordination::CONTESTED_WRITE));
        let recorded = if receipt.status == sdk::ReceiptStatus::Denied && !retryable_collision {
            // Deny hooks and resolutions on SDK turns: the authority hook,
            // whose refusals carry its fixed marker, the untrusted-content
            // verifier (its own marker), the owner's declines (the kernel's
            // marker), the outbound secret scan, and the same-tool limit.
            let authority_denial = reason.starts_with("[AUTHORITY BLOCKED]");
            let untrusted_denial = reason.starts_with(crate::agent::untrusted::UNTRUSTED_BLOCKED);
            let owner_declined = reason.starts_with(sdk::OWNER_DECLINED);
            let egress_blocked = reason.starts_with(sdk::EGRESS_BLOCKED);
            let coordination_stop =
                reason.starts_with(crate::agent::coordination::COORDINATION_STOP);
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
            } else if !untrusted_denial && !owner_declined && !coordination_stop {
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
                    crate::events::OWNER_APPROVAL_DECLINED
                } else if egress_blocked {
                    "egress_secret_blocked"
                } else if coordination_stop {
                    "coordination_stopped"
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
                if retryable_collision {
                    crate::traits::ToolOutcomeStatus::FailedRetryable
                } else {
                    crate::traits::ToolOutcomeStatus::FailedPermanent
                },
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

    /// The daemon's configured API keys and tokens and those of
    /// dynamically connected bots: refused in outbound calls by
    /// `EgressGuard` and redacted from refusal rows written to the ledger.
    async fn configured_secret_literals(&self, engine: &SdkEngineSettings) -> Vec<String> {
        let mut secrets = engine.outbound_secrets.values().to_vec();
        match self.state.get_dynamic_bots().await {
            Ok(bots) => {
                for bot in bots {
                    secrets.push(bot.bot_token);
                    secrets.extend(bot.app_token);
                }
            }
            Err(error) => warn!(%error, "Could not read dynamic bots for the secret scan"),
        }
        secrets
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
            instruction_tracker,
            restrict_untrusted_reference_tools,
            untrusted_reference_skills,
            memory_recall_allowed,
            learning: learning_consent,
            conversation_summary,
            known_project_dir,
            project_scope,
            explicit_workspace,
            authority,
            authorized_scopes,
            scopes_are_guess,
            projected_source_message_ids,
            projected_source_turn_ids,
            defer_on_provider_failure,
        } = setup;
        let user_text = user_text.as_str();
        let model = engine.model_override.clone().unwrap_or(model);
        touch_heartbeat(&heartbeat);
        // The chat's running turn, so the sub-agents it starts are listed
        // (and stopped) as part of this request.
        let _running_request =
            (self.depth == 0).then(|| self.running_children.begin_request(session_id, &task_id));
        let cwd = turn_cwd(&*self.execution_backend);
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
        let text_only_next_call = Arc::new(AtomicBool::new(false));
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
        let mut authority_hook =
            sdk::AuthorityHook::new(authority.authority.clone()).with_ceiling(ceiling.clone());
        // A refusal by a limit the owner can lift says how to lift it.
        if let Some(note) = crate::agent::authority::own_limit_note(&authority) {
            authority_hook = authority_hook.with_limit_note(note);
        }
        // A child's authority is what its parent gave it: a refusal says so.
        if self.inherited_authority.is_some() {
            authority_hook = authority_hook.delegated();
        }
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
            turn_id: turn_id.clone(),
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
            learning_consent,
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
                builder = builder.tool(turn_tool(
                    spec,
                    tool.clone(),
                    goal_actions_only,
                    calls_tx.clone(),
                    bridge_roots.clone(),
                ));
            }
        }
        // The offered MCP-registry tools, bridged the same way so every
        // kernel hook judges them. A daemon tool of the same name wins (the
        // dispatcher resolves it first). The set is fixed at turn start: a
        // server added mid-run is offered from the next turn.
        let mut mcp_tools: Vec<Arc<dyn crate::traits::Tool>> = Vec::new();
        if user_role != UserRole::Public {
            if let Some(registry) = self.mcp_registry.as_ref() {
                let mut names: Vec<&String> = offered_tools
                    .iter()
                    .filter(|name| !self.tools.iter().any(|tool| tool.name() == name.as_str()))
                    .collect();
                names.sort();
                for name in names {
                    let Some(tool) = registry.find_tool(name).await else {
                        continue;
                    };
                    if !tool.is_available() {
                        continue;
                    }
                    // A server may advertise a name the kernel cannot take
                    // (dots, over 64 characters): that tool is left out
                    // rather than failing the turn.
                    if !kernel_tool_name(name) {
                        warn!(tool = %name, "MCP tool name is not a valid kernel tool name; not offered");
                        continue;
                    }
                    builder = builder.tool(turn_tool(
                        tool_spec(tool.as_ref()),
                        tool.clone(),
                        false,
                        calls_tx.clone(),
                        bridge_roots.clone(),
                    ));
                    mcp_tools.push(tool);
                }
            }
        }
        // Memory hooks run after the ledger stop, so a rejected final answer
        // never reaches a memory writer. Summarization and learning belong to
        // the turn, so they run whatever the memory plugin.
        let owned_memory_tools = memory.owned_daemon_tools;
        let mut memory_hooks: Vec<Arc<dyn sdk::Hook>> = Vec::new();
        // A turn opted out of learning stores nothing, through any memory
        // tool either.
        memory_hooks.push(Arc::new(
            crate::agent::memory_plugin::PrivateTurnGuard::new(
                memory_turn.learning_consent.clone(),
            ),
        ));
        if !memory_turn.mandate {
            memory_hooks.push(Arc::new(
                crate::agent::memory_plugin::DaemonMemoryWriter::new(
                    memory_turn.clone(),
                    crate::agent::memory_plugin::WriterSettings {
                        progressive_facts: false,
                        summarization: self.context_window_config.enabled,
                        learning: self.memory.is_daemons_own(),
                    },
                    self.transcript_writes.clone(),
                ),
            ));
        }
        // The summary first: notes keep their order after the task-context
        // tail, where the summary stood before.
        if let Some(note) =
            crate::agent::conversation_compaction::summary_note(conversation_summary.as_ref())
        {
            memory_hooks.push(Arc::new(note));
        }
        memory_hooks.extend(memory.hooks);
        if user_role != UserRole::Public {
            for tool in memory.tools {
                crate::agent::memory_plugin::register_memory_system_tool(
                    &tool.spec().name,
                    tool.annotations().read_only,
                );
                builder = builder.tool_arc(tool);
            }
            // "Don't remember this" is the host's, so it is offered under any
            // memory system, on a turn that would otherwise learn. A turn
            // offered no tools stays tool-free (`/private` still works there).
            if !offered_tools.is_empty()
                && !memory_turn.mandate
                && memory_turn.memory_persistence_allowed
                && !memory_turn.memory_isolated()
                && memory_turn.learning_consent.allowed()
            {
                builder = builder.tool(crate::agent::memory_plugin::SkipLearningTool::new(
                    memory_turn.learning_consent.clone(),
                    emitter.clone(),
                    task_id.clone(),
                    turn_id.clone(),
                ));
            }
        }
        // Owner approvals: the daemon's rules for this turn and the chat's
        // grants (a child shares its parent chat's).
        let grant_session =
            super::approvals::grant_session(self.approval_session_id.as_deref(), session_id);
        let mandate_run = self.mandate_execution.is_some();
        // The terminal's ladder sees a call's capability escalation the way
        // the dispatcher's scope check will (same facts, same known
        // project directory as it is refined during the run).
        let known_project_dir = Arc::new(std::sync::Mutex::new(known_project_dir));
        let judged_terminal_calls = super::approvals::JudgedCalls::default();
        let canonical_calls = tool_calls::CanonicalCalls::default();
        let run_declines = super::approvals::RunDeclines::default();
        let workspace_grant = channel_ctx.active_workspace_grant(user_role).is_some();
        let terminal_approvals =
            super::approvals::TerminalApprovals::find(&self.tools).map(|terminal| {
                let tool = terminal.tool();
                let known = known_project_dir.clone();
                let session = session_id.to_string();
                let task_access = authority.task_access.clone();
                let scopes = authorized_scopes.clone();
                terminal.with_canonicalizer(
                    terminal_canonicalizer(TerminalCanonTurn {
                        backend: self.execution_backend.clone(),
                        paths: self.daemon_paths.clone(),
                        tool,
                        known_project_dir: known,
                        session_id: session,
                        task_access,
                        authorized_scopes: scopes,
                        workspace_grant,
                        records: canonical_calls.clone(),
                    }),
                    judged_terminal_calls.clone(),
                )
            });
        let trusted_session = !mandate_run
            && self
                .session_trusted(channel_ctx.trusted, Some(task_id.as_str()))
                .await;
        // A run nobody is watching cannot answer a card: its guess still binds.
        let scopes_are_guess = scopes_are_guess && !trusted_session && !mandate_run;
        let rules = super::approvals::AppApprovalRules {
            paths: self.daemon_paths.clone(),
            trusted_session,
            mandate_preapproved: mandate_run,
            guessed_scopes: if scopes_are_guess {
                authorized_scopes.clone()
            } else {
                Vec::new()
            },
            known_project_dir: Some(known_project_dir.clone()),
            session_id: session_id.to_string(),
            grant_session: grant_session.clone(),
            user_role: Some(user_role),
            channel_visibility: Some(channel_ctx.visibility),
            untrusted_trigger_session: super::is_trigger_session(session_id),
            terminal: terminal_approvals,
            tools: super::approvals::ApprovalTools::new(self.tools.iter().cloned()),
            run_declines: run_declines.clone(),
        };
        let rules = Arc::new(rules);
        let approvals = self.owner_approvals.clone();
        let approval =
            aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never)
                .with_rules(rules.clone())
                .with_grants(approvals.grants_for(&grant_session))
                .workspace_root(self.execution_backend.workspace_root().as_str());
        let secrets = self.configured_secret_literals(engine).await;
        let egress = secrets
            .iter()
            .fold(aidaemon_policies::EgressGuard::new(), |guard, secret| {
                guard.secret(secret.clone())
            });
        let ledger_redactor = secrets
            .iter()
            .fold(aidaemon_policies::Redactor::new(), |redactor, secret| {
                redactor.secret(secret.clone())
            });
        // The turn's tools by name, as the dispatcher resolves them: a
        // daemon tool wins over an MCP tool of the same name.
        let mut turn_tools: std::collections::HashMap<String, Arc<dyn crate::traits::Tool>> =
            mcp_tools
                .iter()
                .map(|tool| (tool.name().to_string(), tool.clone()))
                .collect();
        for tool in self.tools.iter() {
            turn_tools.insert(tool.name().to_string(), tool.clone());
        }
        let network = turn_tools.iter().fold(
            aidaemon_tools::NetworkGuard::new(),
            |guard, (name, tool)| guard.reach(name.clone(), tool.dispatch_facts().network_reach),
        );
        let protected_paths = turn_tools.iter().fold(
            aidaemon_tools::ProtectedPathGuard::new(
                self.execution_backend.clone(),
                crate::tools::daemon_file_policy(self.daemon_paths.clone()),
            ),
            |guard, (name, tool)| guard.tool(name.clone(), tool.dispatch_facts().file_guard),
        );
        let turn_tools = Arc::new(turn_tools);
        let spawn_family = self
            .inherited_authority
            .as_ref()
            .and_then(|parent| parent.spawn_family.clone())
            .unwrap_or_else(|| crate::agent::authority::SpawnFamily::new(self.limits.spawn_family));
        if self.inherited_authority.is_none() {
            spawn_family
                .coordination()
                .bind_root(session_id, user_text, &turn_id);
        }
        let hooks = TurnHooks {
            coordination: crate::agent::coordination::CoordinationHook {
                family: spawn_family.clone(),
                member: session_id.to_string(),
                tools: turn_tools.clone(),
                backend: self.execution_backend.clone(),
                escalator: crate::agent::coordination::Escalator::new(
                    self.side_calls.clone(),
                    self.state.clone(),
                    self.state.clone(),
                ),
            },
            canonical: tool_calls::CanonicalCallHook {
                tools: turn_tools,
                known_project_dir: known_project_dir.clone(),
                workspace_grant,
                task_access: authority.task_access.clone(),
                authorized_scopes: authorized_scopes.clone(),
                records: canonical_calls.clone(),
                paths: self.daemon_paths.clone(),
                backend: self.execution_backend.clone(),
            },
            authority: authority_hook,
            same_tool: aidaemon_policies::SameToolStreakGuard {
                read_only_tools: read_only_tool_names(self.tools.iter()),
                allowances: same_tool_streaks(self.tools.iter()),
                limits: SAME_TOOL_LIMITS,
            },
            ledger: LedgerStop {
                calls: calls_tx.clone(),
            },
            memory: memory_hooks,
            protected_paths,
            network,
            egress,
            untrusted: sdk::UntrustedHook::new(
                crate::agent::untrusted::untrusted_mode(authority.owner_level),
                // The result boundary (`tool_result_fit`) already marked it,
                // for the stored history as well as the model.
                |_tool: &str, content: &str| content.to_string(),
            ),
            approval,
            text_only: TextOnlyNext {
                flag: text_only_next_call.clone(),
            },
            budget: BudgetGate {
                calls: calls_tx.clone(),
            },
            stuck: StuckNotice {
                notices: refusals_tx.clone(),
            },
            model_pin: ModelPin {
                task_id: task_id.clone(),
            },
            cutoff: CutoffAdjust::default(),
            retry: ModelRetry {
                calls: calls_tx.clone(),
            },
            dispatch: DispatchGate {
                calls: calls_tx.clone(),
            },
            after_call: AfterCallRecord {
                calls: calls_tx.clone(),
            },
        };
        for (_, hook) in hooks.chain() {
            builder = builder.hook_arc(hook);
        }
        drop(calls_tx);
        let sink_status = status_tx.clone();
        let sink_heartbeat = heartbeat.clone();
        let sink_progress = self.child_progress.clone();
        // The kernel holds one sink: heartbeat, kernel refusals for the
        // ledger, and the thinking status all go through it.
        // Every run journals its events and checkpoints its state, so a
        // crash leaves the kernel's own record (`in_flight` / `pending`).
        if let Some(store) = self.run_store.clone() {
            let store: Arc<dyn sdk::RunStore> = store;
            builder = builder.store(store);
        }
        if let Some(sink) = crate::otel_export::sink() {
            builder = builder.event_sink_arc(sink);
        }
        builder = builder.event_sink(move |_run, event: &sdk::Event| {
            touch_heartbeat(&sink_heartbeat);
            // Every kernel event (a turn, a model reply, a tool result) is
            // the run making progress.
            if let Some(progress) = &sink_progress {
                progress.touch();
            }
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
        // `/clear` deletes the summary, and the taint goes with it.
        let summary_tainted = conversation_summary.is_some()
            && crate::agent::conversation_compaction::summary_is_tainted(
                self.state.as_ref(),
                session_id,
            )
            .await;
        let starts_tainted =
            crate::agent::untrusted::starts_tainted(&crate::agent::untrusted::TaintFacts {
                user_role,
                visibility: channel_ctx.visibility,
                session_id,
                projected_turn_ids: &projected_source_turn_ids,
                tainted_turns: tainted_turns.as_deref(),
                summary_tainted,
                inherited: self
                    .inherited_authority
                    .as_ref()
                    .is_some_and(|parent| parent.untrusted_context),
            });
        if starts_tainted {
            state = state.with_untrusted_context();
        }
        // A child starts bound by what the owner declined in its parent's
        // run (set after the user message: a new run's declines start empty).
        if let Some(parent) = self.inherited_authority.as_ref() {
            state.declined =
                crate::agent::authority::union_declines(&state.declined, &parent.declined);
        }
        // From then on the run shares its declines live with its parent and
        // its children (the kernel exchanges them before every call); a root
        // turn starts a family of its own.
        let shared_declines = self
            .inherited_authority
            .as_ref()
            .and_then(|parent| parent.shared_declines.clone())
            .unwrap_or_default();
        let mut state = state.share_declines(shared_declines.clone());
        state.parent_run = self
            .inherited_authority
            .as_ref()
            .and_then(|parent| parent.parent_run);
        let prefix_len = state.messages.len();
        let run_id = state.run_id;
        run_declines.bind(run_id);

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
        // A root turn judges egress against the owner's request lineage:
        // the open request a follow-up advances, or the unfinished earlier
        // request a short continuation may refer to. Judging the bare current
        // message is how "Try again" lost the request it was about.
        // A mandate run's request is the mandate the owner confirmed: its
        // session is the owner's chat, whose open request is about something
        // else, and its task text is the runtime's own.
        let inherited = self.inherited_authority.as_ref();
        let mandate_origin = self.mandate_origin().await;
        let lineage_request = if inherited.is_none() && mandate_origin.is_none() {
            let dialogue =
                super::dialogue_state::get_or_rebuild_dialogue_state(self, session_id).await;
            super::dialogue_state::owner_request_for_verifier(
                &dialogue,
                user_text,
                chrono::Utc::now(),
            )
        } else {
            None
        };
        let origin_request: &str = mandate_origin
            .as_ref()
            .map(|origin| origin.request.as_str())
            .or_else(|| inherited.and_then(|parent| parent.origin_request.as_deref()))
            .or(lineage_request.as_deref())
            .unwrap_or(user_text);
        let origin_trusted =
            crate::agent::untrusted::origin_is_owner_authored(user_role, session_id)
                && inherited.is_none_or(|parent| parent.origin_trusted)
                && mandate_origin.as_ref().is_none_or(|origin| origin.trusted);
        // A site on this computer the owner's own request names is one the
        // browser may open for this work without a card.
        if origin_trusted {
            aidaemon_tools::local_sites::note_sites_in(
                &super::approvals::grant_session(self.approval_session_id.as_deref(), session_id),
                origin_request,
            )
            .await;
        }
        let scope = TurnScope {
            daemon_paths: &self.daemon_paths,
            execution_backend: &*self.execution_backend,
            run_id,
            session_id,
            mcp_tools,
            ledger_redactor,
            user_text,
            task_id: &task_id,
            status_tx: status_tx.clone(),
            channel_ctx: &channel_ctx,
            user_role,
            model: &model,
            turn_id: &turn_id,
            contract: &contract,
            learning: learning.clone(),
            learning_consent: memory_turn.learning_consent.clone(),
            latest_tool_output: std::sync::Mutex::new(None),
            reply_retry_used: AtomicBool::new(false),
            text_only_next_call: text_only_next_call.clone(),
            pending_charge: std::sync::Mutex::new(None),
            known_project_dir: known_project_dir.clone(),
            judged_terminal_calls,
            canonical_calls,
            gate_outcomes: Default::default(),
            executed_calls: Default::default(),
            run_declines,
            shared_declines,
            project_scope,
            authority: &authority,
            ceiling: ceiling.clone(),
            trust_roots,
            cwd,
            untrusted_seen: AtomicBool::new(starts_tainted),
            origin_request,
            origin_trusted,
            authorized_scopes,
            scopes_are_guess,
            projected_source_message_ids,
            projected_source_turn_ids,
            instruction_tracker: instruction_tracker.map(tokio::sync::Mutex::new),
            extra_system: std::sync::Mutex::new(Vec::new()),
            notices_seen: std::sync::Mutex::new((chrono::Utc::now(), HashSet::new())),
            placed_directives: std::sync::Mutex::new(Vec::new()),
            absorbed_steering: std::sync::Mutex::new(Vec::new()),
            compaction: std::sync::Mutex::new(None),
            sent_files: std::sync::Mutex::new(HashSet::new()),
            combined_verification_demands: std::sync::atomic::AtomicU32::new(0),
            mandate_commit_retries: std::sync::atomic::AtomicU32::new(0),
            gui_verify_demands: std::sync::atomic::AtomicU32::new(0),
            unreported_jobs_wait_spent: AtomicBool::new(false),
            budget_warning_sent: AtomicBool::new(false),
            model_call_timeouts: std::sync::atomic::AtomicU32::new(0),
            context_overflowed: AtomicBool::new(false),
            provider_failure: std::sync::Mutex::new(None),
            context_overflow_retries: std::sync::atomic::AtomicU32::new(0),
            task_tokens: spawn_family.tokens(),
            spawn_family,
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
            progress: std::sync::Mutex::new(
                crate::agent::completion_contract::CompletionProgress::new(&contract, &task_id),
            ),
            background_handoff: AtomicBool::new(false),
            open_background_jobs: AtomicUsize::new(0),
            emitter: &emitter,
            prefix,
            prefix_len,
            budget_goal,
            model_calls: AtomicU64::new(0),
            closeout_demands: std::sync::atomic::AtomicU32::new(0),
            closeout_ended_unmet: AtomicBool::new(false),
            verification_demanded: AtomicBool::new(false),
            recovery_demanded: AtomicBool::new(false),
            executor_demanded_at: AtomicU64::new(u64::MAX),
            open_checklist_items: AtomicU64::new(0),
            checklist_declared: AtomicBool::new(false),
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
                tools: &self.tools,
                gateway: &self.side_calls,
                decisions: self,
                emitter: scope.emitter,
                task_id: scope.task_id,
                origin_request,
                origin_trusted,
                cancel: cancel.clone(),
                // The request's project folder.
                working_area: scope.project_scope.as_deref(),
            },
            owner: super::approvals::OwnerApprovalResolver {
                decisions: self,
                mission: self.inherited_authority.as_ref().and_then(|inherited| {
                    inherited
                        .card_title
                        .as_deref()
                        .or(inherited.lineage.last().map(String::as_str))
                }),
                approvals,
                rules,
                card_session: session_id,
                grant_session,
                emitter: &emitter,
                task_id: &task_id,
                cancel: cancel.clone(),
            },
        };
        // A run the daemon stopped on (a crash, a restart) is settled first,
        // so its batch finishes under the kernel's rules before this turn.
        let stranded = recovery::stranded_run(self.run_store.as_deref(), session_id).await;
        let abandoned = super::approvals::AbandonedRunResolver {
            turn: &resolver,
            not_done: std::sync::Mutex::new(Vec::new()),
        };
        let run = async {
            if let Some(stranded) = stranded {
                let settled =
                    recovery::settle(&sdk_agent, stranded, cancel.clone(), &abandoned).await;
                match settled {
                    Ok(Some(_)) => info!(session_id, "Settled the run a restart left unfinished"),
                    Ok(None) => warn!(
                        session_id,
                        "The unfinished run was not settled; it stays for a later turn"
                    ),
                    Err(error) => warn!(session_id, %error, "Settling the unfinished run failed"),
                }
            }
            sdk_agent.run_resolving(state, cancel, &resolver).await
        };
        tokio::pin!(run);
        let end_if_dropped = recovery::EndIfDropped::new(self.run_store.clone(), run_id);
        let mut in_flight = FuturesUnordered::new();
        // Ledger writes for kernel notices run apart from the run's own calls:
        // they are finished after the run ends, never dropped mid-write.
        let mut ledger_writes = FuturesUnordered::new();
        let mut tick = tokio::time::interval(HEARTBEAT_TICK);
        let outcome = loop {
            tokio::select! {
                outcome = &mut run => break outcome,
                Some(call) = calls_rx.recv() => match call {
                    // A tool call running is progress: each has its own
                    // bound (its timeout or stall watchdog), so a child that
                    // runs a long install is never stopped as idle.
                    DaemonCall::Tool(request) => {
                        let running = self.child_progress.as_ref().map(ProgressClock::begin);
                        // Each tool call is one of the run's steps: a
                        // helper's time runs from its first step to its
                        // report, its start-up left out.
                        let step = crate::tools::OwnerWaitClock::begin_step();
                        let call = self.dispatch_sdk_tool_call(&scope, request);
                        in_flight.push(
                            async move {
                                call.await;
                                drop(step);
                                drop(running);
                            }
                            .boxed(),
                        );
                    }
                    DaemonCall::AfterCall(check) => {
                        in_flight.push(self.finish_sdk_tool_call(&scope, check).boxed());
                    }
                    DaemonCall::Gate(request) => {
                        in_flight.push(self.serve_sdk_gate(&scope, request).boxed());
                    }
                    DaemonCall::Model(call) => {
                        in_flight.push(self.serve_sdk_model_call(&scope, call).boxed());
                    }
                    DaemonCall::Stop(check) => {
                        in_flight.push(self.serve_sdk_stop(&scope, check).boxed());
                    }
                    DaemonCall::ModelError(check) => {
                        in_flight.push(self.serve_sdk_model_error(&scope, check).boxed());
                    }
                    DaemonCall::Budget(check) => {
                        in_flight.push(self.serve_sdk_budget(&scope, check).boxed());
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
        end_if_dropped.disarm();
        self.sdk_return_unanswered_steering(&scope).await;
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
                let (mut reply, status, task_outcome) =
                    scope
                        .verdict
                        .final_reply(&outcome, &tool_calls, channel_ctx.visibility);
                // Nothing committed: the request is a pure re-runnable input.
                // Queue it for an automatic re-run (after a used-up quota's
                // reset) instead of making the user resend it; a re-run that
                // fails the same way re-queues with attempts + 1.
                let provider_failure = scope
                    .provider_failure
                    .lock()
                    .ok()
                    .and_then(|failure| failure.clone());
                if let (sdk::RunStatus::ProviderFailed { .. }, Some(failure), true, 0, false) = (
                    &outcome.status,
                    provider_failure,
                    defer_on_provider_failure,
                    outcome.state.tool_calls,
                    scope.background_handoff.load(Ordering::Relaxed),
                ) {
                    let failure = anyhow::Error::new(failure);
                    if let Some(attempt) = self
                        .defer_request_for_provider_recovery(
                            session_id,
                            user_text,
                            user_role,
                            &channel_ctx,
                            &failure,
                        )
                        .await
                    {
                        reply = crate::agent::deferred_retry::deferral_notice(attempt, &failure);
                    }
                }
                let not_done = abandoned
                    .not_done
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                if let Some(note) = recovery::not_done_note(&not_done) {
                    reply = format!("{note}\n\n{reply}");
                }
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

#[cfg(test)]
#[path = "sdk_engine/finish_tests.rs"]
mod finish_tests;
#[cfg(test)]
#[path = "sdk_engine/hook_port_tests.rs"]
mod hook_port_tests;
#[cfg(test)]
#[path = "sdk_engine/status_tests.rs"]
mod status_tests;
