//! The turn engine: one turn on the aidaemon SDK kernel, on any
//! [`TurnHost`].
//!
//! [`TurnEngine`] is a view of a host: it derefs to `dyn TurnHost`, so the
//! engine reads the host's typed surface as its own, and it passes itself
//! wherever a `&dyn TurnHost` is wanted (`turn::task_end`, the side-call
//! gateway). It owns no state; a turn's state lives in its
//! [`TurnScope`](crate::engine::scope::TurnScope).
//!
//! Layout: `model_call` serves the kernel's provider requests,
//! `compaction` compacts a run that outgrows its window, `outcome` turns a
//! finished run into the checklist's final shape, and `events` records the
//! turn's task end and decision points.

use crate::engine::deferral::DeferredRetries as _;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aidaemon_channels::context::ChannelContext;
use aidaemon_core as sdk;
use aidaemon_events::conversation::{AttachmentProvenance, Message, MessageAttachment};
use aidaemon_events::{EventEmitter, EventType, TaskOutcome, TaskStatus};
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::FutureExt;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::engine::budgets::*;
use crate::engine::calls::{
    AfterCallCheck, GateRequest, HostCall as DaemonCall, KernelCall, KernelNotice, ModelCall,
    ModelErrorCheck, StopCheck, ToolRequest,
};
use crate::engine::outcome::TurnVerdict;
use crate::engine::provider::{
    fit_run_values, from_provider_response, prefix_with_notes, split_injected_notes,
    to_chat_options, to_daemon_messages_from, to_daemon_tools, values_to_sdk, HostProvider,
};
use crate::engine::recovery;
use crate::engine::scope::{AbsorbedSteering, BudgetGoal, TurnScope};
use crate::engine::settings::SdkEngineSettings;
use crate::engine::settings::{turn_cwd, turn_trust_roots, TrustRootSources};
use crate::engine::tool_exec::ToolExecCtx;
use crate::engine::work_scope::WorkScope;
use crate::engine::TOOL_TIMEOUT;
use crate::host::TurnHost;
use crate::media::vision::encode_image_attachment;
use crate::settings::VisionConfig;
use crate::turn::helpers::touch_heartbeat;
use crate::turn::memory::{daemon_tool_offer, memory_parts, DaemonToolOffer, MemoryTurn};
use crate::turn::notes::TurnNotes;
use crate::turn::turn_context::TurnHistory;
use aidaemon_channels::owner_wait::ProgressClock;

pub mod compaction;
pub mod entry;
mod events;
pub mod main_loop;
pub mod model_call;
pub mod model_pin;
pub mod outcome;
pub mod spawn;
pub mod stop;
pub mod tool_calls;
pub mod turn_hooks;

use turn_hooks::*;

const HEARTBEAT_TICK: Duration = Duration::from_secs(15);

/// The turn engine over a host. See the module docs.
#[derive(Clone, Copy)]
pub struct TurnEngine<'h> {
    host: &'h dyn TurnHost,
}

impl<'h> TurnEngine<'h> {
    pub fn new(host: &'h dyn TurnHost) -> Self {
        Self { host }
    }

    /// The host this engine runs on.
    pub fn host(&self) -> &'h dyn TurnHost {
        self.host
    }
}

impl<'h> std::ops::Deref for TurnEngine<'h> {
    type Target = dyn TurnHost + 'h;

    fn deref(&self) -> &Self::Target {
        self.host
    }
}

/// What turn setup (`main_loop.rs`) settled before the SDK run starts:
/// the recorded task, the structural contract, the offered tools, the selected
/// model and the system prompt.
pub struct SdkTurnSetup {
    /// The persisted user text (bootstrap may have added a transcript).
    pub user_text: String,
    pub task_id: String,
    pub emitter: EventEmitter,
    pub offered_tools: HashSet<String>,
    pub model: String,
    /// The transcript `message_build` projected for the run's first model
    /// call, in the daemon providers' wire format.
    pub prefix: Vec<Value>,
    pub contract: crate::turn::completion_contract::CompletionContract,
    /// Post-task learning state from bootstrap; the run adds its tool calls.
    pub learning_ctx: crate::turn::post_task::LearningContext,
    /// Matched skills reference untrusted external content: tools that could
    /// act on it are blocked, naming these skills.
    pub restrict_untrusted_reference_tools: bool,
    pub untrusted_reference_skills: Vec<String>,
    /// Discovers nested project instructions before deeper subtree work.
    pub instruction_tracker: Option<crate::turn::project_instructions::ProjectInstructionTracker>,
    /// The compiled memory policy allows recall on this turn.
    pub memory_recall_allowed: bool,
    /// Whether the turn may be learned from; `/private` opted it out before
    /// it started, and `skip_learning` can during the run.
    pub learning: aidaemon_policies::TurnLearning,
    /// The conversation summary standing for the turns `prefix` leaves out,
    /// on a turn that continues the conversation.
    pub conversation_summary: Option<aidaemon_events::conversation::ConversationSummary>,
    /// Where calls without a directory run: the contract's execution cwd,
    /// else the request's primary project scope (a scheduled goal's bound
    /// workspace). `None` when a channel workspace grant scopes the file
    /// tools itself.
    pub known_project_dir: Option<String>,
    /// The turn's explicit workspace binding (`TurnContext::explicit_workspace`),
    /// a trusted root. An inferred binding is never passed here.
    pub explicit_workspace: Option<String>,
    /// The single-project scope lock, passed to tools as
    /// `ToolCallContext::project_scope`.
    pub project_scope: Option<String>,
    /// What this turn may do, resolved from explicit facts: the kernel's
    /// `AuthorityHook` enforces its level and tool set, and the
    /// controller-side scope check judges each call's access manifest
    /// against its `task_access`, else the directories the turn authorizes.
    pub authority: crate::authority::ResolvedAuthority,
    pub authorized_scopes: Vec<String>,
    /// `authorized_scopes` comes only from conversation text, for an owner
    /// who can answer a card: it may not refuse their file tools.
    pub scopes_are_guess: bool,
    /// Persisted message and turn ids whose content reached the projected
    /// prefix, recorded on every model call's telemetry.
    pub projected_source_message_ids: Vec<String>,
    pub projected_source_turn_ids: Vec<String>,
    /// A run that fails on the provider before any tool call is queued for
    /// an automatic re-run (`deferred_retry`). False for a continuation of
    /// another task, which that task's own recovery owns.
    pub defer_on_provider_failure: bool,
    /// The turn re-enters the conversation with runtime-produced evidence
    /// (a background command's or helper's result), not with words of the
    /// owner's.
    pub internal_continuation: bool,
}

/// What a turn's terminal canonicalizer reads (see [`terminal_canonicalizer`]).
pub struct TerminalCanonTurn {
    pub tool: Arc<dyn aidaemon_tools::OutcomeTool>,
    pub known_project_dir: Arc<std::sync::Mutex<Option<String>>>,
    pub session_id: String,
    pub task_access: Option<aidaemon_tools::semantics::ToolCallAccessManifest>,
    pub authorized_scopes: Vec<String>,
    pub workspace_grant: bool,
    /// How `CanonicalCallHook` canonicalized each call.
    pub records: tool_calls::CanonicalCalls,
    pub paths: aidaemon_tools::path_class::DaemonPaths,
    pub backend: aidaemon_tools::backend::SharedExecutionBackend,
}

/// How the kernel's terminal rules see a call: in the canonical form the
/// dispatcher executes ([`tool_calls::canonicalize_call`]), scope-checked
/// exactly as the dispatcher will check it.
pub fn terminal_canonicalizer(
    turn: TerminalCanonTurn,
) -> impl Fn(sdk::RunId, &str, &Value) -> crate::approvals::JudgedTerminal + Send + Sync + 'static {
    move |run, call_id, raw| {
        let (call, known) = match turn.records.get(run, call_id) {
            // Already canonical: judged as sent, against the directory the
            // hook recorded for it.
            Some(record) => (
                crate::approvals::CanonicalCall {
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
            Some(crate::turn::tool_dispatch::ScopeCheck::Escalatable { escalation, .. }) => {
                crate::approvals::TerminalScope::Escalate(escalation)
            }
            Some(crate::turn::tool_dispatch::ScopeCheck::Rejected(reason)) => {
                crate::approvals::TerminalScope::Refused(reason)
            }
            None => crate::approvals::TerminalScope::InScope,
        };
        #[cfg(any(test, feature = "testing"))]
        let scope = match crate::approvals::judgment_seam(&turn.session_id) {
            Some(crate::approvals::JudgmentSeam::JudgedScope(judged)) => judged,
            _ => scope,
        };
        crate::approvals::JudgedTerminal { call, scope }
    }
}

/// Encode the image attachments `model` may see. Returns the images and
/// whether an eligible image was left out (vision off, a model without
/// vision, or a file that is unreadable or over the size cap).
pub fn sdk_images(
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
pub fn kernel_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

impl TurnEngine<'_> {
    pub async fn record_kernel_notice(&self, scope: &TurnScope<'_>, notice: KernelNotice) {
        match notice {
            KernelNotice::Refusal(receipt) => self.record_kernel_refusal(scope, receipt).await,
            KernelNotice::Stuck => {
                self.turn_runtime()
                    .with_harness_eval(|eval| eval.record_repetition_guard())
                    .await;
            }
        }
    }

    /// Record a call the kernel refused before dispatch in the run ledger:
    /// an unknown tool or invalid arguments is a typed failed proposal; a
    /// hook denial is a policy denial.
    pub async fn record_kernel_refusal(&self, scope: &TurnScope<'_>, receipt: sdk::Receipt) {
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
        let call = aidaemon_events::conversation::ToolCall {
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
            && (reason.starts_with(crate::coordination::STALE_WRITE)
                || reason.starts_with(crate::coordination::CONTESTED_WRITE));
        let recorded = if receipt.status == sdk::ReceiptStatus::Denied && !retryable_collision {
            // Deny hooks and resolutions on SDK turns: the authority hook,
            // whose refusals carry its fixed marker, the untrusted-content
            // verifier (its own marker), the owner's declines (the kernel's
            // marker), the outbound secret scan, and the same-tool limit.
            let authority_denial = reason.starts_with("[AUTHORITY BLOCKED]");
            let untrusted_denial = reason.starts_with(crate::untrusted::UNTRUSTED_BLOCKED);
            let owner_declined = reason.starts_with(sdk::OWNER_DECLINED);
            let egress_blocked = reason.starts_with(sdk::EGRESS_BLOCKED);
            let coordination_stop = reason.starts_with(crate::coordination::COORDINATION_STOP);
            if authority_denial {
                if !receipt.read_only {
                    self.turn_runtime()
                        .with_harness_eval(|eval| eval.record_forbidden_mutation_attempt())
                        .await;
                }
            } else if egress_blocked {
                self.emit_decision_point(
                    scope.emitter,
                    scope.task_id,
                    scope.model_calls.load(Ordering::Relaxed) as usize,
                    aidaemon_events::payloads::DecisionType::GateTelemetry,
                    format!("outbound secret blocked in `{}`", receipt.tool),
                    serde_json::json!({
                        "condition": "egress_secret_blocked",
                        "tool": receipt.tool,
                        "reason": reason,
                    }),
                )
                .await;
            } else if !untrusted_denial && !owner_declined && !coordination_stop {
                self.turn_runtime()
                    .with_harness_eval(|eval| eval.record_repetition_guard())
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
                    aidaemon_events::run_aggregate::OWNER_APPROVAL_DECLINED
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
                    aidaemon_tools::semantics::ToolOutcomeStatus::FailedRetryable
                } else {
                    aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent
                },
                aidaemon_tools::semantics::ToolInvocationStage::RejectedBeforeDispatch,
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
    pub fn ledger_task_id<'s>(&'s self, scope: &TurnScope<'s>) -> &'s str {
        self.mandate_execution()
            .as_ref()
            .map_or(scope.task_id, |fence| fence.worker_task_id.as_str())
    }

    /// The daemon's configured API keys and tokens and those of
    /// dynamically connected bots: refused in outbound calls by
    /// `EgressGuard` and redacted from refusal rows written to the ledger.
    pub async fn configured_secret_literals(&self, engine: &SdkEngineSettings) -> Vec<String> {
        let mut secrets = engine.outbound_secrets.values().to_vec();
        match self.dynamic_bots().get_dynamic_bots().await {
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
    pub async fn handle_message_sdk(
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
            internal_continuation,
        } = setup;
        let user_text = user_text.as_str();
        let model = engine.model_override.clone().unwrap_or(model);
        touch_heartbeat(&heartbeat);
        // The chat's running turn, so the sub-agents it starts are listed
        // (and stopped) as part of this request.
        let _running_request = (self.depth() == 0)
            .then(|| self.running_children().begin_request(session_id, &task_id));
        let cwd = turn_cwd(&*self.execution_backend_handle());
        // Trusted only by explicit grant, never by inference from the
        // request: `project_scope` and `known_project_dir` steer where tools
        // run but are not trust sources.
        let configured_roots = self.engine_settings().trusted_roots.clone();
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
                .inherited_authority()
                .as_ref()
                .map(|parent| parent.trusted_roots.as_slice()),
        });
        let turn_id = self
            .turn_runtime()
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
        let ceiling = Arc::new(crate::authority::SessionModeCeiling::new(
            self.settings_store(),
            self.mode_session_for(session_id),
        ));
        let mut authority_hook =
            sdk::AuthorityHook::new(authority.authority.clone()).with_ceiling(ceiling.clone());
        // A refusal by a limit the owner can lift says how to lift it.
        if let Some(note) = crate::authority::own_limit_note(&authority) {
            authority_hook = authority_hook.with_limit_note(note);
        }
        // A child's authority is what its parent gave it: a refusal says so.
        if self.inherited_authority().is_some() {
            authority_hook = authority_hook.delegated();
        }
        let mut builder = sdk::Agent::builder("aidaemon")
            .attest(sdk::Safeguard::Scope)
            .model(model.clone())
            .provider(HostProvider {
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
            mandate: self.mandate_execution().is_some(),
            depth: self.depth(),
            model: model.clone(),
            learning: learning.clone(),
            verdict: verdict.clone(),
            learning_consent,
        };
        let memory = memory_parts(self.memory_plugin(), &memory_turn).await;
        // Tools: the set bootstrap offered for this turn (role, policy and
        // contract filtering already applied), with the daemon memory tools
        // only while the active plugin owns them; `manage_memories` keeps
        // its goal actions under any plugin.
        let bridge_roots = Arc::new(trust_roots.clone());
        if user_role != UserRole::Public {
            for tool in self
                .tools()
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
                                &crate::turn::memory::goal_actions_view(&tool.schema()),
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
        let mut mcp_tools: Vec<Arc<dyn aidaemon_tools::OutcomeTool>> = Vec::new();
        if user_role != UserRole::Public {
            {
                let mut names: Vec<&String> = offered_tools
                    .iter()
                    .filter(|name| !self.tools().iter().any(|tool| tool.name() == name.as_str()))
                    .collect();
                names.sort();
                for name in names {
                    let Some(tool) = self.mcp_tool(name).await else {
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
        memory_hooks.push(Arc::new(crate::turn::opt_out::PrivateTurnGuard::new(
            memory_turn.learning_consent.clone(),
        )));
        if !memory_turn.mandate {
            memory_hooks.push(Arc::new(
                crate::turn::memory_writer::DaemonMemoryWriter::new(
                    memory_turn.clone(),
                    crate::turn::memory_writer::WriterSettings {
                        progressive_facts: false,
                        summarization: self.turn_runtime().context_window_config.enabled,
                        learning: self.memory_plugin().is_daemons_own(),
                    },
                    self.transcript_writes().clone(),
                ),
            ));
        }
        // The summary first: notes keep their order after the task-context
        // tail, where the summary stood before.
        if let Some(note) = crate::turn::compaction::summary_note(conversation_summary.as_ref()) {
            memory_hooks.push(Arc::new(note));
        }
        memory_hooks.extend(memory.hooks);
        if user_role != UserRole::Public {
            for tool in memory.tools {
                crate::turn::tool_facts::register_memory_system_tool(
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
                builder = builder.tool(crate::turn::opt_out::SkipLearningTool::new(
                    memory_turn.learning_consent.clone(),
                    emitter.clone(),
                    task_id.clone(),
                    turn_id.clone(),
                ));
            }
        }
        // Owner approvals: the daemon's rules for this turn and the chat's
        // grants (a child shares its parent chat's).
        let grant_session = crate::approvals::grant_session(self.approval_session_id(), session_id);
        let mandate_run = self.mandate_execution().is_some();
        // The terminal's ladder sees a call's capability escalation the way
        // the dispatcher's scope check will (same facts, same known
        // project directory as it is refined during the run).
        let known_project_dir = Arc::new(std::sync::Mutex::new(known_project_dir));
        let judged_terminal_calls = crate::approvals::JudgedCalls::default();
        let canonical_calls = tool_calls::CanonicalCalls::default();
        let run_declines = crate::approvals::RunDeclines::default();
        let workspace_grant = channel_ctx.active_workspace_grant(user_role).is_some();
        let terminal_approvals =
            crate::approvals::TerminalApprovals::find(self.tools()).map(|terminal| {
                let tool = terminal.tool();
                let known = known_project_dir.clone();
                let session = session_id.to_string();
                let task_access = authority.task_access.clone();
                let scopes = authorized_scopes.clone();
                terminal.with_canonicalizer(
                    terminal_canonicalizer(TerminalCanonTurn {
                        backend: self.execution_backend_handle().clone(),
                        paths: self.daemon_paths().clone(),
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
        let rules = crate::approvals::AppApprovalRules {
            paths: self.daemon_paths().clone(),
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
            untrusted_trigger_session: crate::turn::loop_utils::is_trigger_session(session_id),
            terminal: terminal_approvals,
            tools: crate::approvals::ApprovalTools::new(self.tools().iter().cloned()),
            run_declines: run_declines.clone(),
        };
        let rules = Arc::new(rules);
        let approvals = self.owner_approvals().clone();
        let approval =
            aidaemon_policies::ApprovalPolicy::new(aidaemon_policies::ApprovalMode::Never)
                .with_rules(rules.clone())
                .with_grants(approvals.grants_for(&grant_session))
                .workspace_root(self.execution_backend_handle().workspace_root().as_str());
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
        let mut turn_tools: std::collections::HashMap<
            String,
            Arc<dyn aidaemon_tools::OutcomeTool>,
        > = mcp_tools
            .iter()
            .map(|tool| (tool.name().to_string(), tool.clone()))
            .collect();
        for tool in self.tools().iter() {
            turn_tools.insert(tool.name().to_string(), tool.clone());
        }
        let network = turn_tools.iter().fold(
            aidaemon_tools::NetworkGuard::new(),
            |guard, (name, tool)| guard.reach(name.clone(), tool.dispatch_facts().network_reach),
        );
        let protected_paths = turn_tools.iter().fold(
            aidaemon_tools::ProtectedPathGuard::new(
                self.execution_backend_handle().clone(),
                self.file_policy(),
            ),
            |guard, (name, tool)| guard.tool(name.clone(), tool.dispatch_facts().file_guard),
        );
        let turn_tools = Arc::new(turn_tools);
        let spawn_family = self
            .inherited_authority()
            .as_ref()
            .and_then(|parent| parent.spawn_family.clone())
            .unwrap_or_else(|| crate::authority::SpawnFamily::new(self.turn_limits().spawn_family));
        if self.inherited_authority().is_none() {
            spawn_family
                .coordination()
                .bind_root(session_id, user_text, &turn_id);
        }
        let hooks = TurnHooks {
            coordination: crate::coordination::CoordinationHook {
                family: spawn_family.clone(),
                member: session_id.to_string(),
                tools: turn_tools.clone(),
                backend: self.execution_backend_handle().clone(),
                escalator: crate::coordination::Escalator::new(
                    self.side_calls().clone(),
                    self.settings_store(),
                    self.message_log(),
                ),
            },
            canonical: tool_calls::CanonicalCallHook {
                tools: turn_tools,
                known_project_dir: known_project_dir.clone(),
                workspace_grant,
                task_access: authority.task_access.clone(),
                authorized_scopes: authorized_scopes.clone(),
                records: canonical_calls.clone(),
                paths: self.daemon_paths().clone(),
                backend: self.execution_backend_handle().clone(),
            },
            authority: authority_hook,
            same_tool: aidaemon_policies::SameToolStreakGuard {
                read_only_tools: read_only_tool_names(self.tools().iter()),
                allowances: same_tool_streaks(self.tools().iter()),
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
                crate::untrusted::untrusted_mode(authority.owner_level),
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
                computer_use: self.computer_use(),
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
        let sink_progress = self.child_progress().cloned();
        // Reply text streams only from the owner's own turn; a helper's text
        // is its parent's input, not the reply.
        let reply_text = (self.depth() == 0).then(|| status_tx.clone()).flatten();
        let pending_text = std::sync::Mutex::new(PendingReplyText::default());
        // The kernel holds one sink: heartbeat, kernel refusals for the
        // ledger, and the thinking status all go through it.
        // Every run journals its events and checkpoints its state, so a
        // crash leaves the kernel's own record (`in_flight` / `pending`).
        if let Some(store) = self.run_store() {
            let store: Arc<dyn sdk::RunStore> = store.clone();
            builder = builder.store(store);
        }
        if let Some(sink) = self.trace_sink() {
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
            if let Some(tx) = reply_text.as_ref() {
                if let Ok(mut pending) = pending_text.lock() {
                    match event {
                        sdk::Event::TextDelta { text, .. } => pending.push(text),
                        sdk::Event::TurnStarted { .. } => pending.next_call(),
                        _ => {}
                    }
                    pending.flush(tx);
                }
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
        let tainted_turns = crate::untrusted::tainted_turns(self.settings(), session_id)
            .await
            .inspect_err(|error| {
                warn!(session_id, %error, "Could not read tainted turns; the turn starts tainted");
            })
            .ok();
        // `/clear` deletes the summary, and the taint goes with it.
        let summary_tainted = conversation_summary.is_some()
            && crate::turn::compaction::summary_is_tainted(&*self.compaction_store(), session_id)
                .await;
        let starts_tainted = crate::untrusted::starts_tainted(&crate::untrusted::TaintFacts {
            user_role,
            visibility: channel_ctx.visibility,
            session_id,
            projected_turn_ids: &projected_source_turn_ids,
            tainted_turns: tainted_turns.as_deref(),
            summary_tainted,
            inherited: self
                .inherited_authority()
                .as_ref()
                .is_some_and(|parent| parent.untrusted_context),
        });
        if starts_tainted {
            state = state.with_untrusted_context();
        }
        // A child starts bound by what the owner declined in its parent's
        // run (set after the user message: a new run's declines start empty).
        if let Some(parent) = self.inherited_authority().as_ref() {
            state.declined = crate::authority::union_declines(&state.declined, &parent.declined);
        }
        // From then on the run shares its declines live with its parent and
        // its children (the kernel exchanges them before every call); a root
        // turn starts a family of its own.
        let shared_declines = self
            .inherited_authority()
            .as_ref()
            .and_then(|parent| parent.shared_declines.clone())
            .unwrap_or_default();
        let mut state = state.share_declines(shared_declines.clone());
        state.parent_run = self
            .inherited_authority()
            .as_ref()
            .and_then(|parent| parent.parent_run);
        let prefix_len = state.messages.len();
        let run_id = state.run_id;
        run_declines.bind(run_id);

        // Drive the run and the tool dispatcher together on this task.
        let budget_goal = match self.budget_goal_id(session_id).await {
            Some(goal_id) => Some(BudgetGoal {
                scheduled: aidaemon_work::scheduled_runs::goal_has_scheduled_provenance(
                    self.work(),
                    &goal_id,
                    self.task_id(),
                )
                .await,
                goal_id,
            }),
            None => None,
        };
        let budget_goal_for_daily = budget_goal.as_ref().map(|goal| goal.goal_id.clone());
        let scheduled_budget_per_check = match budget_goal.as_ref().filter(|goal| goal.scheduled) {
            Some(goal) => self
                .goals()
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
        let inherited = self.inherited_authority();
        let mandate_origin = self.mandate_origin().await;
        let lineage_request = if inherited.is_none() && mandate_origin.is_none() {
            let dialogue =
                crate::turn::dialogue_state::get_or_rebuild_dialogue_state(self.host, session_id)
                    .await;
            if internal_continuation {
                // A continuation's text is the runtime's report of work that
                // ended, never the owner's follow-up: the request is the
                // owner's own unfinished one (live 2026-11-13: a CLI agent's
                // "Delegated step: … Drupal …" read as the owner narrowing
                // "Drupal and WordPress" to Drupal, and the WordPress install
                // was refused as out of scope).
                crate::turn::dialogue_state::owner_requests_for_continuation(
                    &dialogue,
                    chrono::Utc::now(),
                )
            } else {
                crate::turn::dialogue_state::owner_request_for_verifier(
                    &dialogue,
                    user_text,
                    chrono::Utc::now(),
                )
            }
        } else {
            None
        };
        let origin_request: &str = mandate_origin
            .as_ref()
            .map(|origin| origin.request.as_str())
            .or_else(|| inherited.and_then(|parent| parent.origin_request.as_deref()))
            .or(lineage_request.as_deref())
            .unwrap_or(user_text);
        let origin_trusted = crate::untrusted::origin_is_owner_authored(user_role, session_id)
            && inherited.is_none_or(|parent| parent.origin_trusted)
            && mandate_origin.as_ref().is_none_or(|origin| origin.trusted);
        // A site on this computer the owner's own request names is one the
        // browser may open for this work without a card.
        if origin_trusted {
            aidaemon_tools::local_sites::note_sites_in(
                &crate::approvals::grant_session(self.approval_session_id(), session_id),
                origin_request,
            )
            .await;
        }
        let scope = TurnScope {
            daemon_paths: self.daemon_paths(),
            execution_backend: &*self.execution_backend_handle(),
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
            spawn_family: {
                // The tree's first turn sets the request's one task budget.
                spawn_family.budget().start(task_token_budget(
                    self.turn_limits().task_token_budget,
                    scheduled_budget_per_check,
                ));
                spawn_family
            },
            succeeded_calls: std::sync::atomic::AtomicUsize::new(0),
            succeeded_operations: std::sync::Mutex::new(HashSet::new()),
            repeating: AtomicBool::new(false),
            goal_daily_budget: std::sync::Mutex::new(match &budget_goal_for_daily {
                Some(goal_id) => match &self.goal_tokens() {
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
                crate::turn::completion_contract::CompletionProgress::new(&contract, &task_id),
            ),
            background_handoff: AtomicBool::new(false),
            open_background_jobs: AtomicUsize::new(0),
            background_ack: std::sync::Mutex::new(None),
            background_runs: std::sync::Mutex::new(Vec::new()),
            own_work_since: std::sync::Mutex::new(None),
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
            .cancel_token()
            .as_ref()
            .map(|token| token.child_token())
            .unwrap_or_default();
        // Egress the untrusted-content rule interrupts is settled by a clean
        // check that sees only this request and the proposed call.
        // Owner approvals are asked of the owner through the approval
        // broker.
        let resolver = crate::approvals::DaemonInterruptResolver {
            untrusted: crate::untrusted::UntrustedVerifier {
                tools: self.tools(),
                gateway: self.side_calls(),
                decisions: self.host,
                emitter: scope.emitter,
                task_id: scope.task_id,
                origin_request,
                origin_trusted,
                cancel: cancel.clone(),
                // The request's project folder.
                working_area: scope.project_scope.as_deref(),
            },
            owner: crate::approvals::OwnerApprovalResolver {
                decisions: self.host,
                mission: self.inherited_authority().as_ref().and_then(|inherited| {
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
        let stranded =
            recovery::stranded_run(self.run_store().map(|store| &**store), session_id).await;
        let abandoned = crate::approvals::AbandonedRunResolver {
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
        let end_if_dropped = recovery::EndIfDropped::new(self.run_store().cloned(), run_id);
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
                        let running = self.child_progress().map(ProgressClock::begin);
                        // Each tool call is one of the run's steps: a
                        // helper's time runs from its first step to its
                        // report, its start-up left out.
                        let step = aidaemon_channels::owner_wait::OwnerWaitClock::begin_step();
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
                        in_flight.push(serve_sdk_budget(self.host, &scope, check).boxed());
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
            self.untrusted_read_flag().store(true, Ordering::SeqCst);
            if let Err(error) =
                crate::untrusted::record_tainted_turn(self.settings(), session_id, &turn_id).await
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
                        reply = crate::engine::deferral::deferral_notice(attempt, &failure);
                    }
                }
                // Work the run finished itself after a helper it waited on
                // failed has a measured end: the run's own, from the failure
                // to now, on the card and under the reply, so its time to
                // finish is quoted from a measurement, never estimated.
                let own_work_since = scope.own_work_since.lock().ok().and_then(|since| *since);
                if let (Some(since), TaskStatus::Completed, false) = (
                    own_work_since,
                    status,
                    scope.background_handoff.load(Ordering::Relaxed),
                ) {
                    let span = crate::run_state::measured_span(since, chrono::Utc::now());
                    let took = crate::spawn_tool::took_line(span, std::time::Duration::ZERO);
                    if let Some(tx) = scope.status_tx.as_ref() {
                        let _ = tx.try_send(StatusUpdate::OwnWork { took: took.clone() });
                    }
                    reply = format!(
                        "{}\n\n{}Finished here after the helper failed. {took}",
                        reply.trim_end(),
                        crate::turn::sliding_window::MEASURED_TIMING_MARK
                    );
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
        for tool in self.tools() {
            let _ = tool.on_task_end(&task_id, session_id).await;
        }
        self.turn_runtime()
            .current_turn_ids
            .write()
            .await
            .remove(session_id);
        reply
    }
}

/// Reply text not yet handed to the status channel. A full channel keeps
/// the text here for the next try instead of losing it, and a model call
/// after one that already wrote text starts on a new paragraph.
#[derive(Default)]
struct PendingReplyText {
    text: String,
    wrote: bool,
    paragraph: bool,
}

impl PendingReplyText {
    fn push(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        if self.paragraph {
            self.text.push_str("\n\n");
            self.paragraph = false;
        }
        self.text.push_str(delta);
        self.wrote = true;
    }

    fn next_call(&mut self) {
        if self.wrote {
            self.paragraph = true;
        }
    }

    fn flush(&mut self, tx: &mpsc::Sender<StatusUpdate>) {
        if self.text.is_empty() {
            return;
        }
        if tx
            .try_send(StatusUpdate::ReplyText(self.text.clone()))
            .is_ok()
        {
            self.text.clear();
        }
    }
}

#[cfg(test)]
mod pending_reply_text_tests {
    use super::*;

    #[test]
    fn text_waits_out_a_full_channel_and_calls_start_paragraphs() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(StatusUpdate::Thinking(1)).unwrap();
        let mut pending = PendingReplyText::default();
        pending.next_call();
        pending.push("Checking");
        pending.flush(&tx);
        pending.push(" now.");
        pending.flush(&tx);
        assert!(matches!(rx.try_recv(), Ok(StatusUpdate::Thinking(1))));
        pending.next_call();
        pending.push("Done");
        pending.flush(&tx);
        match rx.try_recv() {
            Ok(StatusUpdate::ReplyText(text)) => assert_eq!(text, "Checking now.\n\nDone"),
            other => panic!("expected reply text, got {other:?}"),
        }
    }
}
