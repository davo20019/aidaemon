//! The pure parts of the engine's tool-call handling: the controller's
//! filesystem scope check, the canonical form of a call that both the
//! kernel's approval rules and the dispatcher read, the dispatcher's gate
//! outcomes, and how a receipt is credited to the completion contract and to
//! post-task learning.

use std::collections::HashMap;
use std::sync::Arc;

use aidaemon_core as sdk;
use async_trait::async_trait;
use serde_json::Value;
use tracing::info;

/// Let a call read the session's own spilled tool results. An oversized
/// result is saved under the session's spill directory and the model is told
/// to read it there; a task scope (a goal's bound workspace) would otherwise
/// refuse the daemon's own output. The grant is read-only and covers this
/// session's directory alone, never another session's or the spill root.
/// What the controller's filesystem scope check reads besides the call.
pub struct CallScopeFacts<'a> {
    pub backend: &'a dyn aidaemon_tools::ExecutionBackend,
    pub session_id: &'a str,
    pub task_access: Option<&'a aidaemon_tools::semantics::ToolCallAccessManifest>,
    pub authorized_scopes: &'a [String],
    /// `authorized_scopes` is only the turn's guess from conversation text.
    pub scopes_are_guess: bool,
    pub known_project_dir: Option<&'a str>,
    pub paths: &'a aidaemon_tools::path_class::DaemonPaths,
}

/// The controller-side filesystem check of one call: protected host data
/// and undeclared writes outside the task boundary are refused; an exact
/// out-of-scope target a user may authorize is escalatable. Shared by the
/// dispatcher and the kernel's terminal approval rules, so the owner is
/// asked about exactly the escalation the dispatcher then grants.
pub fn call_scope_check(
    tool: &dyn aidaemon_tools::OutcomeTool,
    arguments: &str,
    facts: &CallScopeFacts<'_>,
) -> Option<crate::turn::tool_dispatch::ScopeCheck> {
    let mut manifest = aidaemon_tools::semantics::prepare_invocation(tool, arguments)
        .map(|prepared| prepared.access_manifest)
        .unwrap_or_else(|_| tool.call_access_manifest(arguments));
    manifest.adapter_read_targets = tool
        .adapter_owned_access_manifest(arguments)
        .adapter_read_targets;
    grant_own_spill_reads(facts.backend, &mut manifest, facts.session_id);
    // A guess picks where the agent starts; it never refuses the owner. The
    // kernel's approval rules send writes outside it to a card. The terminal
    // keeps its own escalation card.
    let guess_binds = !facts.scopes_are_guess
        || crate::turn::tool_dispatch::tool_supports_scope_escalation(tool.name())
        || crate::turn::tool_dispatch::names_sensitive_path(&manifest, facts.paths);
    crate::turn::tool_dispatch::access_manifest_scope_check(
        tool.name(),
        &manifest,
        facts.task_access,
        if guess_binds {
            facts.authorized_scopes
        } else {
            &[]
        },
        facts.known_project_dir,
        facts.paths,
    )
}

pub fn grant_own_spill_reads(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    manifest: &mut aidaemon_tools::semantics::ToolCallAccessManifest,
    session_id: &str,
) {
    let Some(dir) =
        aidaemon_tools::result_spill::session_spill_dir_for_backend(backend, session_id)
    else {
        return;
    };
    // macOS temp paths have a `/private` canonical twin; grant both spellings.
    let canonical = std::fs::canonicalize(&dir)
        .ok()
        .map(|path| path.to_string_lossy().into_owned());
    for path in std::iter::once(dir).chain(canonical) {
        if let Some(hint) = aidaemon_tools::semantics::ToolTargetHint::new(
            aidaemon_tools::semantics::ToolTargetHintKind::ProjectScope,
            path,
        ) && !manifest.adapter_read_targets.contains(&hint)
        {
            manifest.adapter_read_targets.push(hint);
        }
    }
}

/// What canonicalizing a call reads from the turn.
pub struct CanonFacts<'a> {
    pub backend: &'a dyn aidaemon_tools::ExecutionBackend,
    /// A channel workspace grant scopes the file tools itself: no anchor.
    pub workspace_grant: bool,
    pub known_project_dir: Option<&'a str>,
    pub task_access: Option<&'a aidaemon_tools::semantics::ToolCallAccessManifest>,
    pub authorized_scopes: &'a [String],
    pub paths: &'a aidaemon_tools::path_class::DaemonPaths,
}

/// THE canonical form of a call, used by both the kernel's approval rules
/// (which judge it) and the dispatcher (which executes it), so the judged
/// call is the executed call:
/// - every tool: the model's `_` keys and unset optional arguments dropped,
///   exactly as the dispatcher drops them (`tool_exec::strip_model_arguments`);
/// - a call without a directory is anchored to the turn's project
///   directory: a scheduled goal's terminal command runs in its bound
///   workspace, not the daemon's cwd. The model's own explicit directory
///   refines the anchor; an injected one never does (it may be a parent
///   directory picked because the target does not exist yet);
/// - terminal: declared read roots the task does not authorize are
///   dropped, so a superfluous root (`/tmp`) cannot veto an otherwise
///   authorized call. Dropping a read root does not change what it does.
pub fn canonicalize_call(
    tool: Option<&dyn aidaemon_tools::OutcomeTool>,
    name: &str,
    raw: &Value,
    facts: &CanonFacts<'_>,
) -> CanonicalCall {
    let terminal = tool.is_some_and(|tool| tool.dispatch_facts().runs_as_judged);
    // Every tool, not only the terminal: the dispatcher strips the model's
    // `_` keys and unset optional arguments before any tool runs, so a later
    // hook (an approval card, the untrusted verifier) judging the raw
    // arguments would judge a call that never runs. A tool that binds the
    // owner's answer to the exact call (`browser`) then never finds it.
    let mut call = CanonicalCall::stripped(tool, raw);
    let mut known = facts.known_project_dir.map(str::to_string);
    if !facts.workspace_grant
        && let Ok(text) = serde_json::to_string(&call.arguments)
    {
        if let Some(explicit) =
            crate::turn::project_dir::project_dir_from_tool_args(facts.backend, name, &text)
        {
            known = Some(explicit.clone());
            call.explicit_dir = Some(explicit);
        }
        if let Some((updated, injected)) =
            crate::turn::project_dir::maybe_inject_project_dir_into_tool_args(
                facts.backend,
                name,
                &text,
                known.as_deref(),
            )
            && let Ok(updated) = serde_json::from_str(&updated)
        {
            call.arguments = updated;
            call.injected_dir = Some(injected);
        }
    }
    if terminal {
        call.stripped_read_roots = crate::turn::tool_dispatch::attenuate_unauthorized_read_roots(
            &mut call.arguments,
            facts.task_access,
            facts.authorized_scopes,
            known.as_deref(),
            facts.paths,
        );
    }
    call
}

/// How one call was put in canonical form before the kernel judged it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CanonRecord {
    /// The model's own explicit directory (it refined the known directory).
    pub explicit_dir: Option<String>,
    /// The project directory injected into the call, if any.
    pub injected_dir: Option<String>,
    /// The known project directory the call's scope is checked against:
    /// its explicit directory, else the turn's as of the call. An injected
    /// directory never counts (it may be a parent picked because the
    /// target does not exist yet).
    pub known_project_dir: Option<String>,
}

/// The canonical form of each call, keyed by run and call id: written by
/// [`CanonicalCallHook`] before any other hook judges the call, read by the
/// approval rules and the dispatcher, so nobody canonicalizes again.
#[derive(Debug, Clone, Default)]
pub struct CanonicalCalls(Arc<std::sync::Mutex<HashMap<(sdk::RunId, String), CanonRecord>>>);

impl CanonicalCalls {
    pub fn get(&self, run: sdk::RunId, call_id: &str) -> Option<CanonRecord> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(run, call_id.to_string()))
            .cloned()
    }

    pub fn take(&self, run: sdk::RunId, call_id: &str) -> Option<CanonRecord> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(run, call_id.to_string()))
    }

    pub fn insert(&self, run: sdk::RunId, call_id: &str, record: CanonRecord) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((run, call_id.to_string()), record);
    }
}

/// Puts every call in its canonical form ([`canonicalize_call`]) as the
/// first `before_tool` hook, so the kernel computes effects, scope and
/// approvals for the call that runs: a terminal call without a directory
/// is judged with the directory it will run in, not as relative paths of
/// unknown location. The kernel restarts the chain after the rewrite; a
/// call already recorded is only re-checked, never re-anchored.
pub struct CanonicalCallHook {
    /// The turn's tools by name (a daemon tool wins over an MCP tool).
    pub tools: Arc<HashMap<String, Arc<dyn aidaemon_tools::OutcomeTool>>>,
    pub known_project_dir: Arc<std::sync::Mutex<Option<String>>>,
    pub workspace_grant: bool,
    pub task_access: Option<aidaemon_tools::semantics::ToolCallAccessManifest>,
    pub authorized_scopes: Vec<String>,
    pub records: CanonicalCalls,
    pub paths: aidaemon_tools::path_class::DaemonPaths,
    pub backend: aidaemon_tools::backend::SharedExecutionBackend,
}

#[async_trait]
impl sdk::Hook for CanonicalCallHook {
    async fn before_tool(
        &self,
        ctx: &sdk::HookContext<'_>,
        call: &sdk::ToolCall,
        _tool: &sdk::ToolInfo,
    ) -> sdk::ToolDecision {
        let tool = self.tools.get(&call.name).map(|tool| tool.as_ref());
        let recorded = self.records.get(ctx.run_id, &call.id);
        let known = match &recorded {
            Some(record) => record.known_project_dir.clone(),
            None => self
                .known_project_dir
                .lock()
                .ok()
                .and_then(|known| known.clone()),
        };
        let canonical = canonicalize_call(
            tool,
            &call.name,
            &call.arguments,
            &CanonFacts {
                backend: &*self.backend,
                workspace_grant: self.workspace_grant,
                known_project_dir: known.as_deref(),
                task_access: self.task_access.as_ref(),
                authorized_scopes: &self.authorized_scopes,
                paths: &self.paths,
            },
        );
        if recorded.is_none() {
            if let Some(explicit) = &canonical.explicit_dir
                && let Ok(mut known) = self.known_project_dir.lock()
            {
                *known = Some(explicit.clone());
            }
            if !canonical.stripped_read_roots.is_empty() {
                info!(
                    tool = %call.name,
                    stripped_read_roots = ?canonical.stripped_read_roots,
                    "SDK engine: attenuated unauthorized declared read roots"
                );
            }
            self.records.insert(
                ctx.run_id,
                &call.id,
                CanonRecord {
                    known_project_dir: canonical.explicit_dir.clone().or(known),
                    explicit_dir: canonical.explicit_dir,
                    injected_dir: canonical.injected_dir,
                },
            );
        }
        if canonical.arguments == call.arguments {
            sdk::ToolDecision::Allow
        } else {
            sdk::ToolDecision::Rewrite {
                arguments: canonical.arguments,
            }
        }
    }
}

/// What the dispatcher's gates decided for one call.
#[derive(Debug, Clone)]
pub enum GateOutcome {
    /// Refused, with the text the model sees (the ledger already has it).
    Refuse(String),
    /// Answered in the tool's place (the ledger already has it).
    Answer(sdk::ToolOutput),
    /// May run.
    Clear(GateClearance),
}

impl GateOutcome {
    pub fn decision(&self) -> sdk::ToolDecision {
        match self {
            Self::Refuse(reason) => sdk::ToolDecision::Deny {
                reason: reason.clone(),
            },
            Self::Answer(output) => sdk::ToolDecision::Answer {
                output: output.clone(),
            },
            Self::Clear(_) => sdk::ToolDecision::Allow,
        }
    }
}

/// What a cleared call runs with.
#[derive(Debug, Clone)]
pub struct GateClearance {
    /// The filesystem scope check the call was cleared with.
    pub scope_check: Option<crate::turn::tool_dispatch::ScopeCheck>,
    /// The mandate run's grant for this exact call.
    pub grant: Option<aidaemon_work::mandates::MandateAuthorityGrant>,
}

/// Gate outcomes by run and call id. A call is gated once: the kernel may
/// put a call to its hooks again, and gets the same decision.
#[derive(Debug, Clone, Default)]
pub struct GateOutcomes(Arc<std::sync::Mutex<HashMap<(sdk::RunId, String), GateOutcome>>>);

impl GateOutcomes {
    pub fn get(&self, run: sdk::RunId, call_id: &str) -> Option<GateOutcome> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(run, call_id.to_string()))
            .cloned()
    }

    pub fn take(&self, run: sdk::RunId, call_id: &str) -> Option<GateOutcome> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(run, call_id.to_string()))
    }

    pub fn insert(&self, run: sdk::RunId, call_id: &str, outcome: GateOutcome) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((run, call_id.to_string()), outcome);
    }

    /// The gate refused this call and already recorded the refusal.
    pub fn refused(&self, run: sdk::RunId, call_id: &str) -> bool {
        matches!(self.get(run, call_id), Some(GateOutcome::Refuse(_)))
    }
}

/// How the kernel judged a dispatched call's filesystem scope.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchJudgment {
    /// Not a call the kernel's terminal rules judge: scope-checked now.
    CheckNow,
    /// A terminal call the kernel judged: the scope check it was judged
    /// with, reused as is, so the escalation granted is the one the owner
    /// was shown even if the turn's known project directory moved since.
    /// The remaining window is the filesystem itself: a path created or a
    /// symlink swapped between the judgment and the run is not re-judged;
    /// in Sandbox confinement the sandbox enforces the granted, resolved
    /// paths when the command runs.
    Judged(Option<crate::turn::tool_dispatch::ScopeCheck>),
    /// A terminal call that reached the dispatcher without the kernel's
    /// judgment. Its ladder was never run, so it is refused.
    Unjudged,
}

/// The refusal for a terminal call that reached the dispatcher without the
/// kernel's judgment.
pub fn unjudged_terminal_refusal() -> String {
    "terminal call requires approval through the agent kernel; it reached the dispatcher \
     without the kernel's judgment and was not run."
        .to_string()
}

/// A replayed call's result, marked so the model does not read the earlier
/// success as something that just happened (a reminder re-"set" after the
/// user had it cancelled is still cancelled).
pub fn replay_notice(previous_result: &str) -> String {
    format!(
        "[Not run again: an identical call already ran earlier in this task; its earlier result is \
         repeated below. Nothing new happened now, and anything done since then (for example \
         cancelling what it created) still stands.]\n\n{previous_result}"
    )
}

/// A call that succeeded and changed something outside this machine, by its
/// typed effects: a remote write, a deploy or a delivery.
pub fn changed_remotely(
    metadata: &aidaemon_tools::semantics::ToolCallMetadata,
    status: aidaemon_tools::semantics::ToolOutcomeStatus,
) -> bool {
    use aidaemon_tools::semantics::ToolMutationEffects as Effects;
    let external = Effects::REMOTE_MUTATION.bits()
        | Effects::REMOTE_DEPLOY.bits()
        | Effects::EXTERNAL_DELIVERY.bits();
    metadata.semantics.mutates_state()
        && metadata.semantics.mutation_effects.bits() & external != 0
        && metadata.invocation_stage.performed_io()
        && !metadata.receipt_replayed
        && status.satisfies_requested_condition()
}

/// Credit one receipt to the contract obligations it proves: a satisfied
/// mutation marks its effects, and a
/// verifiable observation marks the evidence requirements it matches.
/// Returns the obligation ids the receipt closes.
#[allow(clippy::too_many_arguments)]
pub fn bind_receipt_to_contract(
    progress: &mut crate::turn::completion_contract::CompletionProgress,
    contract: &crate::turn::completion_contract::CompletionContract,
    call_id: &str,
    tool_name: &str,
    arguments: &str,
    result_text: &str,
    metadata: &aidaemon_tools::semantics::ToolCallMetadata,
    status: aidaemon_tools::semantics::ToolOutcomeStatus,
) -> Vec<String> {
    use crate::turn::tool_dispatch as evidence;
    let semantics = &metadata.semantics;
    if metadata.invocation_stage.performed_io()
        && semantics.mutates_state()
        && !metadata.contract_rejected
        && !metadata.receipt_replayed
    {
        progress.record_mutation_attempt(status);
    }
    // Mutation before observation: a mutation invalidates older evidence, so
    // a combined receipt must not invalidate its own post-action observation.
    if status.satisfies_requested_condition() && semantics.mutates_state() {
        progress.mark_mutation_receipt(contract, semantics, call_id);
    }
    if semantics.observes_state() {
        let matched_contract = evidence::observation_matches_completion_contract(
            contract,
            semantics,
            arguments,
            result_text,
            metadata,
        );
        let matched = evidence::matching_evidence_requirement_indices(
            contract,
            tool_name,
            semantics,
            arguments,
            result_text,
            metadata,
        );
        let available = matches!(
            status,
            aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded
                | aidaemon_tools::semantics::ToolOutcomeStatus::CompletedWithNegativeResult
        ) || matched.iter().any(|index| {
            contract
                .evidence_requirements
                .get(*index)
                .is_some_and(|requirement| {
                    evidence::evidence_requirement_accepts_nonstandard_outcome(
                        requirement,
                        metadata,
                    )
                })
        });
        if available
            && evidence::tool_result_or_metadata_contains_verifiable_evidence(
                semantics,
                result_text,
                metadata,
            )
        {
            progress.mark_observation_receipt(contract, &matched, matched_contract, call_id);
        }
    }
    progress.completion_obligations_for_receipt(call_id)
}

/// Record one call for post-task learning. Errors and the recovery that
/// follows feed error solutions, failure patterns and the outcome label: a
/// success after an error credits it as recovered.
pub fn record_call_for_learning(
    learning: &mut crate::turn::post_task::LearningContext,
    summary: String,
    output: &str,
    failed: bool,
) {
    if failed {
        if learning.first_error.is_none() {
            learning.first_error = Some(output.to_string());
        }
        learning.errors.push((output.to_string(), false));
    } else if !learning.errors.is_empty() {
        learning.recovery_actions.push(summary.clone());
        if let Some((_, recovered)) = learning.errors.last_mut() {
            *recovered = true;
        }
    }
    learning.tool_calls.push(summary);
}

/// One tool call as post-task learning records it (a
/// `tool_name(summary)` line, marked when it was rejected or failed so
/// activity summaries never claim work that did not happen).
pub fn learning_tool_summary(name: &str, arguments: &str, rejected: bool, failed: bool) -> String {
    let line = format!(
        "{name}({})",
        crate::turn::tool_summary::summarize_tool_args(name, arguments)
    );
    if rejected {
        format!("{line} [REJECTED]")
    } else if failed {
        format!("{line} [FAILED]")
    } else {
        line
    }
}

/// A call as the dispatcher executes it: the model's `_` keys and unset
/// optional arguments dropped, the turn's project directory injected and
/// unauthorized read roots dropped (`sdk_engine::tool_calls::canonicalize_call`).
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalCall {
    pub arguments: serde_json::Value,
    /// The model's own explicit directory; it refines the turn's known
    /// project directory when the call is dispatched.
    pub explicit_dir: Option<String>,
    /// The project directory injected into the call, if any.
    pub injected_dir: Option<String>,
    /// Declared read roots dropped as unauthorized.
    pub stripped_read_roots: Vec<String>,
}

impl CanonicalCall {
    /// Only the model's `_` keys and unset optional arguments dropped (no
    /// turn to anchor against).
    pub fn stripped(
        tool: Option<&dyn aidaemon_tools::OutcomeTool>,
        raw: &serde_json::Value,
    ) -> Self {
        let mut arguments = raw.clone();
        if let serde_json::Value::Object(map) = &mut arguments {
            strip_model_arguments(map, tool);
        }
        Self {
            arguments,
            explicit_dir: None,
            injected_dir: None,
            stripped_read_roots: Vec::new(),
        }
    }
}

/// Drop optional top-level arguments the model filled with an empty string
/// or an empty list. Some models send every declared field and use `""` or
/// `[]` for "not provided"; a tool would otherwise read them as values (an
/// empty session filter that matches nothing, an empty `schedules` list that
/// hides `schedule`). Required fields, numbers and booleans are untouched.
/// The model's arguments as the dispatcher reads them: any `_`-prefixed key
/// the model sent is dropped (no tool reads one: the dispatcher's facts
/// reach a tool as its typed `ToolCallContext`), and optional arguments the
/// model left unset are dropped against the tool's schema. Every kernel hook
/// judges a call after this same step (`sdk_engine::tool_calls::canonicalize_call`),
/// so the judged call is the executed one.
pub fn strip_model_arguments(
    map: &mut serde_json::Map<String, Value>,
    tool: Option<&dyn aidaemon_tools::OutcomeTool>,
) {
    // The kernel's own rule, so the daemon and the SDK strip alike.
    aidaemon_core::strip_reserved_arguments(map);
    if let Some(tool) = tool {
        drop_unset_optional_arguments(map, &tool.schema());
    }
}

fn drop_unset_optional_arguments(map: &mut serde_json::Map<String, Value>, schema: &Value) {
    let parameters = schema.get("parameters").unwrap_or(schema);
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    map.retain(|key, value| {
        if required.contains(&key.as_str()) {
            return true;
        }
        match value {
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(items) => !items.is_empty(),
            _ => true,
        }
    });
}

/// How a call ended, before its result is recorded.
pub enum ExecutedOutcome {
    Returned {
        outcome: aidaemon_tools::semantics::ToolCallOutcome,
        status: aidaemon_tools::semantics::ToolOutcomeStatus,
    },
    /// The tool returned `Err`: an explicit typed failure.
    Raised {
        metadata: aidaemon_tools::semantics::ToolCallMetadata,
        error_text: String,
    },
}

impl ExecutedOutcome {
    pub fn metadata(&self) -> &aidaemon_tools::semantics::ToolCallMetadata {
        match self {
            Self::Returned { outcome, .. } => &outcome.metadata,
            Self::Raised { metadata, .. } => metadata,
        }
    }

    /// The same test the receipt makes.
    pub fn failed(&self) -> bool {
        match self {
            Self::Returned { outcome, status } => {
                status.is_failure() || outcome.metadata.contract_rejected
            }
            Self::Raised { .. } => true,
        }
    }
}

/// A call that ran, waiting for the after-tool hook to record it.
pub struct ExecutedCall {
    pub request: crate::engine::calls::KernelCall,
    pub arguments: String,
    pub injected_dir: Option<String>,
    pub facts: aidaemon_tools::semantics::ToolDispatchFacts,
    pub grant: Option<aidaemon_events::mandate_grant::MandateAuthorityGrant>,
    pub idempotency_key: Option<String>,
    pub send_file_key: Option<String>,
    pub duration_ms: u64,
    pub trust: sdk::SourceTrust,
    pub executed: ExecutedOutcome,
}

/// Calls that ran and are not recorded yet, by run and call id.
#[derive(Clone, Default)]
pub struct ExecutedCalls(Arc<std::sync::Mutex<HashMap<(sdk::RunId, String), ExecutedCall>>>);

impl ExecutedCalls {
    pub fn insert(&self, run: sdk::RunId, call_id: &str, call: ExecutedCall) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((run, call_id.to_string()), call);
    }

    pub fn take(&self, run: sdk::RunId, call_id: &str) -> Option<ExecutedCall> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(run, call_id.to_string()))
    }
}

#[cfg(test)]
mod unset_argument_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_optional_arguments_are_dropped_and_required_ones_kept() {
        let schema = json!({
            "name": "search_history",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": ["action", "content"]
            }
        });
        let mut args = serde_json::from_value::<serde_json::Map<String, Value>>(json!({
            "action": "search",
            "content": "",
            "query": "broken.py",
            "session_id": "",
            "task_id": "  ",
            "schedules": [],
            "tags": ["a"],
            "event_id": 0,
            "include_subagents": false
        }))
        .unwrap();
        drop_unset_optional_arguments(&mut args, &schema);
        let mut keys: Vec<&str> = args.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "action",
                "content",
                "event_id",
                "include_subagents",
                "query",
                "tags"
            ]
        );
    }
}
