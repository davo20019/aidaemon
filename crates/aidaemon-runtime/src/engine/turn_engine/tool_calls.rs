//! Tool calls: the daemon side of `DaemonCall::Tool`. Scope and authority
//! checks, durable replay, project anchoring, dispatch through
//! `execute_tool_outcome`, receipts and the persisted tool exchange.

use super::*;
use crate::engine::dispatcher::ToolDispatch as _;
#[allow(unused_imports)]
use crate::engine::{tool_result_fit::ToolResultFit, work_scope::WorkScope};

pub use crate::engine::tool_calls::*;

/// Stands in for a call's result between the tool returning and the
/// after-tool hook writing what the model reads.
const RESULT_PENDING: &str = "[tool result pending]";

/// The kernel's judgment of a dispatched call, removed from `judged`:
/// `None` when the tool's calls are not judged in canonical form
/// (`dispatch_facts().runs_as_judged`, the terminal, whose approval ladder
/// the kernel runs and the tool then skips), `Some(None)` when the call was
/// never judged in `run`.
pub fn take_terminal_judgment(
    judged: &crate::approvals::JudgedCalls,
    run: sdk::RunId,
    tool: Option<&dyn aidaemon_tools::OutcomeTool>,
    call_id: &str,
) -> Option<Option<crate::approvals::JudgedTerminal>> {
    let runs_as_judged = tool.is_some_and(|tool| tool.dispatch_facts().runs_as_judged);
    runs_as_judged.then(|| judged.take(run, call_id))
}

/// Put a dispatched call in its canonical form: a terminal call the kernel
/// judged runs exactly as judged, with the scope it was judged with; any
/// other call is canonicalized now. The model's explicit directory refines
/// the turn's known project directory. Returns the injected directory and
/// the kernel's judgment.
pub fn canonicalize_dispatched_call(
    scope: &TurnScope<'_>,
    tool: Option<&dyn aidaemon_tools::OutcomeTool>,
    request: &mut ToolRequest,
) -> (Option<String>, DispatchJudgment) {
    let taken = take_terminal_judgment(
        &scope.judged_terminal_calls,
        request.run_id,
        tool,
        &request.call_id,
    );
    #[cfg(any(test, feature = "testing"))]
    let taken = match crate::approvals::judgment_seam(scope.session_id) {
        Some(crate::approvals::JudgmentSeam::DropJudgment) => taken.map(|_| None),
        _ => taken,
    };
    let (judged, judgment) = match taken {
        Some(Some(judged)) => {
            let check = judged.scope.as_scope_check();
            (Some(judged.call), DispatchJudgment::Judged(check))
        }
        Some(None) => (None, DispatchJudgment::Unjudged),
        None => (None, DispatchJudgment::CheckNow),
    };
    // The kernel already holds the canonical form (`CanonicalCallHook`):
    // the arguments are run as sent, with the directories recorded then.
    if let Some(record) = scope.canonical_calls.take(request.run_id, &request.call_id) {
        return (record.injected_dir, judgment);
    }
    // Only a call that bypassed the hook chain gets here (tests).
    let canonical = match judged {
        Some(canonical) => canonical,
        None => {
            let known = scope
                .known_project_dir
                .lock()
                .ok()
                .and_then(|known| known.clone());
            canonicalize_call(
                tool,
                &request.name,
                &request.arguments,
                &CanonFacts {
                    backend: scope.execution_backend,
                    workspace_grant: scope
                        .channel_ctx
                        .active_workspace_grant(scope.user_role)
                        .is_some(),
                    known_project_dir: known.as_deref(),
                    task_access: scope.authority.task_access.as_ref(),
                    authorized_scopes: &scope.authorized_scopes,
                    paths: scope.daemon_paths,
                },
            )
        }
    };
    if let Some(explicit) = &canonical.explicit_dir {
        if let Ok(mut known) = scope.known_project_dir.lock() {
            *known = Some(explicit.clone());
        }
    }
    if !canonical.stripped_read_roots.is_empty() {
        info!(
            task_id = scope.task_id,
            stripped_read_roots = ?canonical.stripped_read_roots,
            "SDK engine: attenuated unauthorized declared read roots"
        );
    }
    request.arguments = canonical.arguments;
    (canonical.injected_dir, judgment)
}

impl TurnEngine<'_> {
    /// The tool this turn runs under `name`: a daemon tool, else an
    /// MCP-registry tool bridged for the turn.
    pub fn turn_tool<'s>(
        &'s self,
        scope: &'s TurnScope<'_>,
        name: &str,
    ) -> Option<&'s Arc<dyn aidaemon_tools::OutcomeTool>> {
        self.tools()
            .iter()
            .find(|tool| tool.name() == name)
            .or_else(|| scope.mcp_tools.iter().find(|tool| tool.name() == name))
    }

    /// What a call of the tool this turn runs under `name` does: the
    /// instance judged at turn start, never a fresh registry lookup (an MCP
    /// server reloaded mid-run could answer differently).
    pub fn sdk_call_semantics(
        &self,
        scope: &TurnScope<'_>,
        name: &str,
        arguments: &str,
    ) -> aidaemon_tools::semantics::ToolCallSemantics {
        self.turn_tool(scope, name)
            .map(|tool| tool.call_semantics(arguments))
            .unwrap_or_default()
    }

    /// The dispatch facts of the tool this turn runs under `name` (nothing
    /// special for an unknown tool).
    /// Who authored what this call returned: only executed calls carry the
    /// tool's own class (or what the tool reported for this exact output);
    /// every other output is the daemon's.
    pub fn call_output_trust(
        &self,
        scope: &TurnScope<'_>,
        name: &str,
        arguments: &str,
        metadata: Option<&aidaemon_tools::semantics::ToolCallMetadata>,
    ) -> sdk::SourceTrust {
        let reported = metadata.and_then(|metadata| metadata.output_trust);
        self.turn_tool(scope, name)
            .map(|tool| {
                reported.unwrap_or_else(|| {
                    tool.output_trust(
                        arguments,
                        &aidaemon_tools::semantics::TrustRoots::new(&scope.trust_roots)
                            .with_cwd(Some(&scope.cwd)),
                    )
                })
            })
            .unwrap_or_default()
    }

    /// How a step reads on the owner's status line: the plain title the
    /// call gave it ("running the site installer"), else the tool's
    /// friendly label and a cleaned summary.
    pub fn step_activity(
        &self,
        scope: &TurnScope<'_>,
        tool: &str,
        arguments: &str,
        summary: impl FnOnce() -> String,
    ) -> (String, String) {
        if let Some(title) = self
            .turn_tool(scope, tool)
            .and_then(|known| known.call_title(arguments))
        {
            return (title, String::new());
        }
        aidaemon_tools::sanitize::user_facing_tool_activity(
            tool,
            &summary(),
            scope.channel_ctx.visibility,
        )
    }

    pub fn turn_tool_facts(
        &self,
        scope: &TurnScope<'_>,
        name: &str,
    ) -> aidaemon_tools::semantics::ToolDispatchFacts {
        self.turn_tool(scope, name)
            .map(|tool| tool.dispatch_facts())
            .unwrap_or_default()
    }

    /// Judge the call's access manifest against the task's filesystem
    /// authority. `None`: in scope (or not a known tool).
    pub fn sdk_scope_check(
        &self,
        scope: &TurnScope<'_>,
        tool_name: &str,
        arguments: &str,
    ) -> Option<crate::turn::tool_dispatch::ScopeCheck> {
        let tool = self.turn_tool(scope, tool_name)?;
        let known = scope
            .known_project_dir
            .lock()
            .ok()
            .and_then(|known| known.clone());
        call_scope_check(
            tool.as_ref(),
            arguments,
            &CallScopeFacts {
                backend: &*self.execution_backend_handle(),
                session_id: scope.session_id,
                task_access: scope.authority.task_access.as_ref(),
                authorized_scopes: &scope.authorized_scopes,
                scopes_are_guess: scope.scopes_are_guess,
                known_project_dir: known.as_deref(),
                paths: self.daemon_paths(),
            },
        )
    }

    pub async fn dispatch_sdk_tool_call(&self, scope: &TurnScope<'_>, mut request: ToolRequest) {
        let tool = self.turn_tool(scope, &request.name).cloned();
        let (injected_dir, judgment) =
            canonicalize_dispatched_call(scope, tool.as_deref(), &mut request);
        let arguments = serde_json::to_string(&request.arguments).unwrap_or_else(|_| "{}".into());
        let abandoned = request.abandoned.clone();
        let mut output = tokio::select! {
            output = self.run_sdk_tool_call(
                scope,
                &request,
                &arguments,
                injected_dir.as_deref(),
                &judgment,
            ) => output,
            _ = abandoned.cancelled() => {
                warn!(tool = %request.name, "Tool call stopped: the run no longer waits for it");
                return;
            }
        };
        // Refusals, deferrals, replays and suppressed duplicates are the
        // daemon's own words: trusted.
        output.trust.get_or_insert(sdk::SourceTrust::Trusted);
        let _ = request.reply.send(Ok(output));
    }

    /// An unrestricted owner's egress after untrusted content runs without
    /// a check (the kernel only marks it); every such call is recorded.
    pub async fn record_unrestricted_egress(
        &self,
        scope: &TurnScope<'_>,
        tool_name: &str,
        arguments: &str,
    ) {
        if scope.authority.owner_level != Some(crate::authority::OwnerLevel::Unrestricted)
            || !scope
                .untrusted_seen
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        let egress = self.turn_tool(scope, tool_name).is_some_and(|tool| {
            tool.call_annotations_in(
                arguments,
                &aidaemon_tools::semantics::TrustRoots::new(&scope.trust_roots),
            )
            .egress
                || tool.call_outbound(arguments) != sdk::Outbound::Nothing
        });
        if !egress {
            return;
        }
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            0,
            aidaemon_events::payloads::DecisionType::GateTelemetry,
            "unrestricted owner: egress after untrusted content ran unchecked",
            serde_json::json!({
                "condition": "untrusted_egress_unrestricted",
                "tool": tool_name,
            }),
        )
        .await;
    }

    /// What a child spawned from this turn inherits: the turn's resolved
    /// authority, owner level and workspace, the chat whose `/readonly`
    /// narrows it, whether the turn has seen untrusted content so far, and
    /// the root request with its trust.
    pub async fn turn_authority_for_children(
        &self,
        scope: &TurnScope<'_>,
    ) -> crate::authority::InheritedAuthority {
        // What this run has reached so far, with what its transcript and its
        // own parent already held: the child starts from these identities.
        let known_resources = crate::known_resources::known_resources(
            self.events(),
            self.inherited_authority(),
            scope.session_id,
            Some(scope.task_id),
            &scope.projected_source_turn_ids,
        )
        .await;
        crate::authority::InheritedAuthority {
            authority: scope.authority.authority.clone(),
            owner_level: scope.authority.owner_level,
            task_access: scope.authority.task_access.clone(),
            mode_session: Some(self.mode_session_for(scope.session_id)),
            untrusted_context: scope.untrusted_seen.load(Ordering::SeqCst),
            origin_request: Some(
                scope
                    .origin_request
                    .chars()
                    .take(crate::authority::INHERITED_REQUEST_CHARS)
                    .collect(),
            ),
            origin_trusted: scope.origin_trusted,
            trusted_roots: scope.trust_roots.clone(),
            // What the owner declined in this run so far (the run's
            // declines already hold what this turn itself inherited).
            declined: crate::authority::union_declines(
                &crate::authority::union_declines(
                    self.inherited_authority()
                        .as_ref()
                        .map(|parent| parent.declined.as_slice())
                        .unwrap_or_default(),
                    &scope.run_declines.snapshot(),
                ),
                &scope.shared_declines.snapshot(),
            ),
            // The child shares this run's live declines: each is refused
            // what the owner declined in the other, even after the spawn.
            shared_declines: Some(scope.shared_declines.clone()),
            // The child shares this turn's learning: a sub-agent of a turn
            // opted out of learning (before or during the run) learns nothing.
            learning: Some(scope.learning_consent.clone()),
            spawn_family: Some(scope.spawn_family.clone()),
            child_declaration: None,
            parent_run: Some(scope.run_id),
            known_resources,
            // The spawn tool appends the child's own mission.
            card_title: None,
            lineage: self
                .inherited_authority()
                .as_ref()
                .map(|parent| parent.lineage.clone())
                .unwrap_or_default(),
            parent_task: Some(aidaemon_events::TaskRef {
                session_id: scope.session_id.to_string(),
                task_id: scope.task_id.to_string(),
            }),
        }
    }

    /// Run one call and record it: a `ToolCall` event, then the tool message with a typed receipt, so the run ledger
    /// sees every operation while the run is still going.
    pub async fn run_sdk_tool_call(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        arguments: &str,
        injected_dir: Option<&str>,
        judgment: &DispatchJudgment,
    ) -> sdk::ToolOutput {
        // The dispatcher's gates ran as the kernel's last hook
        // (`DispatchGate`); a call that bypassed the hook chain is gated now.
        let outcome = match scope.gate_outcomes.take(request.run_id, &request.call_id) {
            Some(outcome) => outcome,
            None => {
                self.sdk_gate_call(scope, request, arguments, judgment)
                    .await
            }
        };
        let clearance = match outcome {
            GateOutcome::Clear(clearance) => clearance,
            GateOutcome::Refuse(refusal) => return sdk::ToolOutput::error(refusal),
            GateOutcome::Answer(output) => return output,
        };
        let scope_escalation = match &clearance.scope_check {
            Some(crate::turn::tool_dispatch::ScopeCheck::Escalatable { escalation, .. }) => {
                Some(escalation.clone())
            }
            _ => None,
        };
        // How the dispatcher treats this tool (typed, never by name).
        let facts = self.turn_tool_facts(scope, &request.name);
        // A child spawned by this call inherits what this turn has now.
        let turn_authority = if facts.spawns_child_agents {
            Some(self.turn_authority_for_children(scope).await)
        } else {
            None
        };
        // The session's mode as of now (`/readonly` can arrive mid-turn).
        let ceiling = sdk::AuthorityCeiling::ceiling(scope.ceiling.as_ref()).await;
        let mut ctx = self.sdk_exec_ctx(
            scope,
            request,
            ceiling.is_some(),
            scope_escalation.as_ref(),
            turn_authority.as_ref(),
        );
        let grant = clearance.grant;
        ctx.mandate_authority = grant.as_ref();
        let started = Instant::now();
        let call_semantics = self.sdk_call_semantics(scope, &request.name, arguments);
        // The same file and caption go out once per task.
        let send_file_key = facts
            .delivers_files
            .then(|| crate::turn::loop_utils::extract_send_file_dedupe_key_from_args(arguments))
            .flatten();
        let idempotency_key = self.sdk_idempotency_key(scope, request, &call_semantics);
        self.record_unrestricted_egress(scope, &request.name, arguments)
            .await;
        let mut call_data = aidaemon_events::payloads::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(self.ledger_task_id(scope).to_string()),
        )
        .with_policy_metadata(idempotency_key.clone(), None, None);
        call_data.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = scope.emitter.emit(EventType::ToolCall, call_data).await {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool call");
        }

        // User-facing activity labels for the channel's status line;
        // `track_requirements` shows its checklist instead.
        if !facts.requirement_checklist {
            let (name, summary) = self.step_activity(scope, &request.name, arguments, || {
                crate::turn::tool_summary::summarize_tool_args(&request.name, arguments)
            });
            let kind = step_kind(
                &request.name,
                arguments,
                &facts,
                call_semantics.mutates_state(),
            );
            if let Some(status_tx) = &scope.status_tx {
                let _ = status_tx.try_send(StatusUpdate::ToolStart {
                    name,
                    summary,
                    kind,
                });
            }
        }
        let executed = match self
            .execute_tool_with_watchdog_outcome(&request.name, arguments, &ctx)
            .await
        {
            Ok(mut outcome) => {
                // An executed call keeps the semantics it was admitted with
                // (an observation stays an observation even when it returns a
                // negative result).
                crate::turn::tool_dispatch::complete_tool_result_semantics(
                    &request.name,
                    arguments,
                    &call_semantics,
                    &mut outcome.metadata,
                );
                let status = crate::turn::loop_utils::classify_tool_outcome_status(
                    &request.name,
                    &outcome.output,
                    Some(arguments),
                    Some(&outcome.metadata),
                );
                ExecutedOutcome::Returned { outcome, status }
            }
            Err(error) => {
                // A tool that returns `Err` made an explicit typed failure: it
                // gets a `FailedPermanent` receipt, so the run ledger and the task outcome see it.
                let prepared = self.turn_tool(scope, &request.name).and_then(|tool| {
                    aidaemon_tools::semantics::prepare_invocation(tool.as_ref(), arguments).ok()
                });
                let mut metadata = aidaemon_tools::semantics::ToolCallMetadata {
                    outcome_status: Some(
                        if error
                            .downcast_ref::<aidaemon_tools::NothingSentError>()
                            .is_some()
                        {
                            aidaemon_tools::semantics::ToolOutcomeStatus::FailedRetryable
                        } else {
                            aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent
                        },
                    ),
                    transport_error: Some(error.to_string()),
                    invocation_stage: if error
                        .downcast_ref::<crate::engine::tool_exec::UnknownToolError>()
                        .is_some()
                    {
                        aidaemon_tools::semantics::ToolInvocationStage::RejectedBeforeDispatch
                    } else {
                        aidaemon_tools::semantics::ToolInvocationStage::Dispatched
                    },
                    ..aidaemon_tools::semantics::ToolCallMetadata::default()
                };
                if let Some(prepared) = prepared {
                    metadata.receipt_kind = prepared.receipt_kind;
                    metadata.access_manifest = Some(prepared.access_manifest);
                }
                crate::turn::tool_dispatch::complete_tool_result_semantics(
                    &request.name,
                    arguments,
                    &call_semantics,
                    &mut metadata,
                );
                let error_text = format!("Error: {error}");
                ExecutedOutcome::Raised {
                    metadata,
                    error_text,
                }
            }
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        let failed = executed.failed();
        let trust =
            self.call_output_trust(scope, &request.name, arguments, Some(executed.metadata()));
        // A site on this computer named in a trusted result, or in the
        // output of a call that set something up here within this run's
        // authority (its own install or start command, whatever program;
        // untrusted as a whole when it needed the network), is one the work
        // created: in the run's scope, so the browser opens it without a
        // card, here and in every sub-agent of the chat (the grant session
        // is the chat's). Never from a read of outside content (live
        // 2026-11-13: a WordPress helper was asked to approve opening the
        // site it had just set up in its own folder).
        // A command that ran inside this run's own write scope reports the
        // sites that folder serves: those are in the run's scope whoever
        // set them up (live 2026-11-13: two helpers each asked to open the
        // site their own scoped folder served, which existed before them).
        let write_roots: Vec<String> = scope
            .authority
            .task_access
            .as_ref()
            .map(|access| {
                access
                    .write_targets
                    .iter()
                    .map(|target| target.value.clone())
                    .collect()
            })
            .unwrap_or_default();
        let reports_sites = trust == sdk::SourceTrust::Trusted
            || matches!(
                &executed,
                ExecutedOutcome::Returned { outcome, status }
                    if aidaemon_tools::local_sites::set_up_locally(*status, &outcome.metadata.semantics)
            )
            || matches!(
                &executed,
                ExecutedOutcome::Returned { outcome, status }
                    if !status.is_failure()
                        && outcome
                            .metadata
                            .access_manifest
                            .as_ref()
                            .and_then(|access| access.execution_cwd.as_deref())
                            .is_some_and(|cwd| {
                                aidaemon_tools::local_sites::served_within(cwd, &write_roots)
                            })
            );
        if reports_sites {
            if let ExecutedOutcome::Returned { outcome, .. } = &executed {
                let grant =
                    crate::approvals::grant_session(self.approval_session_id(), scope.session_id);
                aidaemon_tools::local_sites::note_sites_in(&grant, &outcome.output).await;
            }
        }
        // The first untrusted read records the turn as tainted at once, so a
        // run that never finishes still taints later turns that see it.
        if trust == sdk::SourceTrust::Untrusted {
            self.untrusted_read_flag()
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if trust == sdk::SourceTrust::Untrusted
            && !scope
                .untrusted_seen
                .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            if let Err(error) = crate::untrusted::record_tainted_turn(
                self.settings(),
                scope.session_id,
                scope.turn_id,
            )
            .await
            {
                warn!(session_id = scope.session_id, %error, "Could not record a tainted turn");
            }
        }
        scope.executed_calls.insert(
            request.run_id,
            &request.call_id,
            ExecutedCall {
                request: KernelCall {
                    run_id: request.run_id,
                    name: request.name.clone(),
                    arguments: request.arguments.clone(),
                    call_id: request.call_id.clone(),
                },
                arguments: arguments.to_string(),
                injected_dir: injected_dir.map(str::to_string),
                facts,
                grant,
                idempotency_key,
                send_file_key,
                duration_ms,
                trust,
                executed,
            },
        );
        // The result itself is written by the after-tool hook
        // (`AfterCallRecord`), which replaces this text.
        sdk::ToolOutput {
            is_error: failed,
            ..sdk::ToolOutput::text(RESULT_PENDING)
        }
        .with_trust(trust)
    }

    /// What happens after a call returned, run from the kernel's `after_tool`
    /// hook: the result is fitted for the model, the receipt, status line and
    /// learning are written, and the kernel gets what the model will read.
    pub async fn finish_sdk_tool_call(&self, scope: &TurnScope<'_>, check: AfterCallCheck) {
        let Some(call) = scope.executed_calls.take(check.run_id, &check.call_id) else {
            let _ = check.reply.send(None);
            return;
        };
        let output = self.record_sdk_tool_result(scope, call).await;
        let _ = check.reply.send(Some(output));
    }

    /// A durable task's call that changed something outside this machine
    /// (a post, a remote write, a deploy) is logged on the task as
    /// `remote_change`, from its typed effects whatever the tool: the
    /// heartbeat reads it so a goal run never repeats an external action
    /// that already happened.
    pub async fn record_remote_change(
        &self,
        tool: &str,
        arguments: &str,
        metadata: &aidaemon_tools::semantics::ToolCallMetadata,
        status: aidaemon_tools::semantics::ToolOutcomeStatus,
    ) {
        let Some(task_id) = self.task_id() else {
            return;
        };
        if !changed_remotely(metadata, status) {
            return;
        }
        let activity = aidaemon_work::goals::TaskActivity {
            id: 0,
            task_id: task_id.to_string(),
            activity_type: aidaemon_work::goals::REMOTE_CHANGE_ACTIVITY.to_string(),
            tool_name: Some(tool.to_string()),
            tool_args: Some(arguments.to_string()),
            result: None,
            success: Some(true),
            tokens_used: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        if let Err(error) = self.tasks().log_task_activity(&activity).await {
            warn!(task_id, %error, "Could not record a remote change on its task");
        }
    }

    pub async fn record_sdk_tool_result(
        &self,
        scope: &TurnScope<'_>,
        call: ExecutedCall,
    ) -> sdk::ToolOutput {
        let ExecutedCall {
            request,
            arguments,
            injected_dir,
            facts,
            grant,
            idempotency_key,
            send_file_key,
            duration_ms,
            trust,
            executed,
        } = call;
        let arguments = arguments.as_str();
        let injected_dir = injected_dir.as_deref();
        let request = &request;
        let (output, images, metadata, receipt, finish) = match executed {
            ExecutedOutcome::Returned {
                mut outcome,
                status,
            } => {
                let evidence =
                    crate::turn::loop_utils::tool_outcome_evidence_source(&outcome.metadata);
                // What the model sees: untrusted output wrapped, oversized
                // output capped. Fitted before the receipt so its provenance
                // records how much the model saw; status and contract binding
                // stay on the raw result.
                let fitted = self
                    .fit_tool_result_for_model(
                        &crate::engine::tool_result_fit::FitScope {
                            tool_name: &request.name,
                            session_id: scope.session_id,
                            task_id: scope.task_id,
                            model: scope.model,
                            emitter: scope.emitter,
                            mandate_execution: self.mandate_execution().is_some(),
                            trust: self.call_output_trust(
                                scope,
                                &request.name,
                                arguments,
                                Some(&outcome.metadata),
                            ),
                        },
                        outcome.output.clone(),
                        false,
                        &mut outcome.metadata,
                    )
                    .await;
                // What the store keeps: the tool's own summary when it gives
                // one, else what the model saw.
                let durable_is_summary = outcome.metadata.persistent_output.is_some();
                let durable = outcome
                    .metadata
                    .persistent_output
                    .clone()
                    .unwrap_or_else(|| fitted.clone());
                if let Some(provenance) = outcome.metadata.result_provenance.as_mut() {
                    provenance.record_durable_view(&durable, durable_is_summary);
                }
                let mut receipt = aidaemon_events::payloads::ToolReceiptV1::from_metadata(
                    &outcome.metadata,
                    status,
                    evidence,
                    idempotency_key.clone(),
                );
                receipt.mandate_authority = grant.clone();
                self.record_remote_change(&request.name, arguments, &outcome.metadata, status)
                    .await;
                if let Ok(mut progress) = scope.progress.lock() {
                    receipt.completion_obligation_ids = bind_receipt_to_contract(
                        &mut progress,
                        scope.contract,
                        &request.call_id,
                        &request.name,
                        arguments,
                        &outcome.output,
                        &outcome.metadata,
                        status,
                    );
                }
                let mut output = fitted;
                // Tool-boundary truncation is rendered into the model-visible
                // text only.
                if let Some(info) = outcome.metadata.truncation.as_ref() {
                    output.push('\n');
                    output.push_str(&crate::turn::helpers::render_truncation_notice(info));
                }
                if let Some(dir) = injected_dir {
                    output.push_str("\n\n");
                    output.push_str(
                        &crate::turn::tool_result_notices::ToolResultNotice::PathAutoInjectedFromProjectContext {
                            injected_dir: dir.to_string(),
                        }
                        .render(),
                    );
                }
                let observed: Vec<&MessageAttachment> = outcome
                    .metadata
                    .attachments
                    .iter()
                    .filter(|a| a.provenance == AttachmentProvenance::ToolObservation)
                    .collect();
                let (images, skipped) =
                    sdk_images(&observed, &self.turn_runtime().vision_config, scope.model);
                if skipped {
                    output.push_str(
                        "\n\n[SYSTEM] The image this tool observed could not be shown to you \
                         (vision is off, unsupported, or the file is too large or unreadable).",
                    );
                }
                // The tool's own typed finish, applied once the batch
                // completes.
                let mut finish = outcome.finish.take();
                // The detach transition, decided at the typed receipt. A
                // launch the model asked to detach is a long-lived service
                // (server, tunnel, watcher) that never "finishes": its useful
                // result is its startup output, so the run continues and the
                // model reports it. Work a tool handed to the background ends
                // the run on the tool's finish.
                let mut handed_off = false;
                // The runs this turn leaves running: a call that left one
                // there names it, and the turn's launch line names them all.
                let launched = outcome
                    .metadata
                    .background_run_label
                    .as_deref()
                    .filter(|_| {
                        !outcome.metadata.detached
                            && outcome.metadata.outcome_status
                                != Some(
                                    aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent,
                                )
                    })
                    .map(|label| {
                        let mut runs = scope
                            .background_runs
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        runs.push(label.to_string());
                        crate::background_run::started_group_in_background(&runs)
                    });
                // Track the jobs this run left running whose result the
                // daemon will deliver: a notifying, non-detached launch opens
                // one; a receipt that observed a job's end closes one.
                if outcome.metadata.background_started
                    && outcome.metadata.completion_notifications_enabled
                    && !outcome.metadata.detached
                    && outcome.metadata.outcome_status
                        != Some(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent)
                {
                    scope.open_background_jobs.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut ack) = scope.background_ack.lock() {
                        *ack = Some(launched.clone().unwrap_or_else(|| {
                            crate::turn::tool_dispatch::build_background_detach_ack(&output)
                        }));
                    }
                }
                if outcome.metadata.background_settled {
                    let jobs = &scope.open_background_jobs;
                    let mut open = jobs.load(Ordering::Relaxed);
                    while open > 0 {
                        match jobs.compare_exchange_weak(
                            open,
                            open - 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(now) => open = now,
                        }
                    }
                }
                if outcome.metadata.background_started || outcome.metadata.background_handoff {
                    // A call the tool itself typed as a permanent failure hands
                    // nothing off, whatever it started: the run goes on.
                    let typed_failure = outcome.metadata.outcome_status
                        == Some(aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent);
                    // The call that started the job is one open item at most;
                    // a second means the model declared work beyond it.
                    let work_remains = scope.open_checklist_items.load(Ordering::Relaxed) > 1;
                    // A job that reports back re-engages the session when it
                    // ends, and that continuation finishes the request's
                    // remaining steps from the job's result. So declared work
                    // beyond it does not keep this run going: left running, the
                    // run polled the job, started a sub-agent to watch it, and
                    // posted half-results while the job worked (live
                    // 2026-11-11). Only a job without notifications keeps the
                    // run going for the remaining work.
                    let reports_back = outcome.metadata.completion_notifications_enabled;
                    let hands_off = finish.is_some()
                        && !typed_failure
                        && (!work_remains || reports_back)
                        && !outcome.metadata.detached;
                    handed_off = hands_off;
                    let directive = if hands_off {
                        // The reply is the call's own first line, which names
                        // the run ("Started in the background: …"); the run
                        // itself is on the turn's card.
                        if outcome.metadata.completion_notifications_enabled {
                            scope.background_handoff.store(true, Ordering::Relaxed);
                        }
                        let reply = launched.clone().unwrap_or_else(|| {
                            crate::turn::tool_dispatch::build_background_detach_ack(&output)
                        });
                        // The kernel keeps the batch's first finish; the
                        // turn's reply is the launch line naming every run it
                        // left running, so the latest one is recorded.
                        if launched.is_some() {
                            scope.verdict.record_handoff_reply(reply.clone());
                        }
                        if let Some(finish) = finish.as_mut() {
                            finish.reply = Some(reply);
                        }
                        crate::turn::system_directives::SystemDirective::BackgroundHandoff {
                            notifications_active: outcome.metadata.completion_notifications_enabled,
                        }
                    } else {
                        finish = None;
                        crate::turn::system_directives::SystemDirective::BackgroundProcessContinue
                    };
                    output = format!("{output}\n\n{}", directive.render());
                }
                let satisfied = !status.is_failure();
                // The tool's own typed finish ends the run once the batch
                // completes; a failed call's finish is dropped, except a
                // handoff, whose work is running whatever its output reads
                // like (as before, a detach was never judged by status). One
                // without a reply ships the call's output, the untrusted-data
                // wrapper removed so reply sanitization does not empty it.
                let finish = finish
                    .filter(|_| satisfied || handed_off)
                    .map(|mut finish| {
                        if finish
                            .reply
                            .as_deref()
                            .is_none_or(|reply| reply.trim().is_empty())
                        {
                            finish.reply = Some(
                                crate::turn::completion_checks::strip_untrusted_wrapper(&output)
                                    .to_string(),
                            );
                        }
                        finish
                    });
                (
                    output,
                    images,
                    Some(outcome.metadata),
                    Some(receipt),
                    finish,
                )
            }
            ExecutedOutcome::Raised {
                mut metadata,
                error_text,
            } => {
                // Oversized errors (cli_agent process output, stack traces)
                // are capped like results.
                let fitted = self
                    .fit_tool_result_for_model(
                        &crate::engine::tool_result_fit::FitScope {
                            tool_name: &request.name,
                            session_id: scope.session_id,
                            task_id: scope.task_id,
                            model: scope.model,
                            emitter: scope.emitter,
                            mandate_execution: self.mandate_execution().is_some(),
                            trust: self.call_output_trust(
                                scope,
                                &request.name,
                                arguments,
                                Some(&metadata),
                            ),
                        },
                        error_text.clone(),
                        true,
                        &mut metadata,
                    )
                    .await;
                let status = aidaemon_tools::semantics::ToolOutcomeStatus::FailedPermanent;
                let evidence = crate::turn::loop_utils::tool_outcome_evidence_source(&metadata);
                let mut receipt = aidaemon_events::payloads::ToolReceiptV1::from_metadata(
                    &metadata,
                    status,
                    evidence,
                    idempotency_key.clone(),
                );
                receipt.mandate_authority = grant.clone();
                if let Ok(mut progress) = scope.progress.lock() {
                    receipt.completion_obligation_ids = bind_receipt_to_contract(
                        &mut progress,
                        scope.contract,
                        &request.call_id,
                        &request.name,
                        arguments,
                        &error_text,
                        &metadata,
                        status,
                    );
                }
                (fitted, Vec::new(), Some(metadata), Some(receipt), None)
            }
        };
        let failed = receipt
            .as_ref()
            .map(|receipt| receipt.outcome_status.is_failure() || receipt.contract_rejected)
            .unwrap_or(true);
        if let Ok(mut latest) = scope.latest_tool_output.lock() {
            *latest = Some(output.clone());
        }
        // A helper the run waited on failed: the work it was given is the
        // run's own from here, and its end is measured when the run ends.
        let handed_on = metadata
            .as_ref()
            .is_some_and(|metadata| metadata.background_handoff || metadata.background_started);
        if failed && facts.spawns_child_agents && !handed_on {
            if let Ok(mut since) = scope.own_work_since.lock() {
                since.get_or_insert_with(chrono::Utc::now);
            }
        }
        // Budget progress counts distinct successful operations: repeating
        // an identical call is not new evidence.
        if !failed {
            let operation = format!(
                "{}:{}",
                request.name,
                crate::prompt::fingerprint::hash_canonical(&request.arguments)
            );
            if let Ok(mut seen) = scope.succeeded_operations.lock() {
                let new_operation = seen.insert(operation);
                if new_operation {
                    scope.succeeded_calls.fetch_add(1, Ordering::Relaxed);
                    scope.spawn_family.budget().note_success(scope.session_id);
                }
                scope.repeating.store(!new_operation, Ordering::Relaxed);
            }
        }
        if let (Some(key), false) = (send_file_key, failed) {
            if let Ok(mut sent) = scope.sent_files.lock() {
                sent.insert(key);
            }
        }
        if let Some(status_tx) = &scope.status_tx {
            let update = if facts.requirement_checklist {
                // The card is the plan as stored, with its step count: never
                // read back out of the tool's text.
                if failed {
                    None
                } else {
                    self.plan_card(scope.session_id, scope.task_id)
                        .await
                        .map(StatusUpdate::Checklist)
                }
            } else {
                // A step that handed its work to a helper or a background
                // job has not finished: its line says the work goes on
                // (typed from the outcome, never from its text).
                let handed_off = metadata.as_ref().is_some_and(|metadata| {
                    metadata.background_handoff || metadata.background_started
                });
                Some(if handed_off {
                    StatusUpdate::StepHandedOff {
                        kind: step_kind(
                            &request.name,
                            arguments,
                            &facts,
                            metadata
                                .as_ref()
                                .is_some_and(|metadata| metadata.semantics.mutates_state()),
                        ),
                    }
                } else {
                    let (name, summary) =
                        self.step_activity(scope, &request.name, arguments, || {
                            if facts.status_from_arguments {
                                crate::turn::tool_summary::summarize_tool_args(
                                    &request.name,
                                    arguments,
                                )
                            } else {
                                crate::turn::tool_dispatch::summarize_completed_tool_result(&output)
                            }
                        });
                    StatusUpdate::ToolComplete { name, summary }
                })
            };
            if let Some(update) = update {
                let _ = status_tx.try_send(update);
            }
        }
        // A receipt-backed confirmation of an external change, delivered if
        // the model cannot be reached afterwards.
        let external_mutation = metadata.as_ref().is_some_and(|metadata| {
            metadata.semantics.mutates_state()
                && !(metadata.background_started || metadata.detached)
        }) && self
            .turn_tool(scope, &request.name)
            .is_some_and(|tool| tool.capabilities().external_side_effect);
        if external_mutation {
            if let Ok(mut ack) = scope.verdict.external_action_ack.lock() {
                *ack = (!failed
                    && crate::turn::tool_dispatch::should_build_external_action_ack(&output))
                .then(|| crate::turn::tool_dispatch::build_external_action_completion_ack(&output));
            }
        }
        if let Ok(mut learning) = scope.learning.lock() {
            let rejected = metadata
                .as_ref()
                .is_some_and(|metadata| metadata.contract_rejected);
            let summary = learning_tool_summary(&request.name, arguments, rejected, failed);
            record_call_for_learning(&mut learning, summary, &output, failed);
        }

        // The model sees the receipt footer replayed history renders
        // (`conversation_turn`), so it can cite the result by `result_id`
        // (mandate evidence_receipt_ids). The stored result stays raw; the
        // footer is projected from the stored receipt on replay.
        let model_view = match &receipt {
            Some(receipt) => format!("{}\n{output}", receipt.context_header()),
            None => output.clone(),
        };
        let mut tool = Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "tool");
        tool.content = Some(output.clone());
        tool.tool_name = Some(request.name.clone());
        tool.tool_call_id = Some(request.call_id.clone());
        tool.turn_id = Some(scope.turn_id.to_string());
        if let Some(metadata) = &metadata {
            tool.attachments = metadata.attachments.clone();
        }
        if let Err(error) = self
            .append_tool_message_with_receipt_event_policy(
                scope.emitter,
                &tool,
                !failed,
                duration_ms,
                failed.then(|| output.clone()),
                Some(self.ledger_task_id(scope)),
                metadata
                    .as_ref()
                    .and_then(|metadata| metadata.persistent_output.as_deref()),
                receipt,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool result");
        }
        // The executor's own checklist becomes ledger expectations.
        if facts.requirement_checklist && !failed {
            if let Some(declared) =
                aidaemon_work::plans::checklist_args::executor_expectations_from_checklist_arguments(
                    scope.task_id,
                    arguments,
                )
            {
                let open = declared
                    .items
                    .iter()
                    .filter(|item| {
                        !matches!(
                            item.status.as_str(),
                            "completed" | "deferred" | "skipped" | "failed" | "background"
                        )
                    })
                    .count();
                scope
                    .open_checklist_items
                    .store(open as u64, Ordering::Relaxed);
                scope.checklist_declared.store(true, Ordering::Relaxed);
                if let Err(error) = scope
                    .emitter
                    .emit(EventType::ExecutorExpectationsDeclared, declared)
                    .await
                {
                    warn!(task_id = scope.task_id, %error, "SDK engine: could not record a checklist");
                }
                // A step the model ticked shows as done only once receipts
                // prove it: the card is redrawn with what they proved.
                if let Some(plan) = self
                    .confirm_checklist_from_receipts(scope.session_id, scope.task_id)
                    .await
                {
                    if let Some(status_tx) = &scope.status_tx {
                        let _ = status_tx.try_send(StatusUpdate::Checklist(plan));
                    }
                }
            }
        }
        info!(
            session_id = scope.session_id,
            task_id = scope.task_id,
            tool = %request.name,
            failed,
            duration_ms,
            "SDK engine tool call"
        );
        let output = sdk::ToolOutput {
            is_error: failed,
            ..sdk::ToolOutput::text(model_view).with_images(images)
        }
        .with_trust(trust);
        match finish {
            Some(finish) => output.finish(finish),
            _ => output,
        }
    }

    /// The dispatcher's context for executing or gating `request`.
    pub fn sdk_exec_ctx<'s>(
        &'s self,
        scope: &'s TurnScope<'_>,
        request: &'s KernelCall,
        ceiling_set: bool,
        scope_escalation: Option<&'s aidaemon_tools::semantics::ScopeEscalation>,
        turn_authority: Option<&'s crate::authority::InheritedAuthority>,
    ) -> ToolExecCtx<'s> {
        ToolExecCtx {
            file_fence: scope
                .spawn_family
                .active_coordination()
                .map(|coordination| {
                    Arc::new(crate::coordination::CallFence {
                        coordination: coordination.clone(),
                        member: scope.session_id.to_string(),
                        call_id: request.call_id.clone(),
                        backend: self.execution_backend_handle().clone(),
                        escalator: crate::coordination::Escalator::new(
                            self.side_calls().clone(),
                            self.settings_store(),
                            self.message_log(),
                        ),
                        untrusted_seen: scope.untrusted_seen.load(Ordering::SeqCst),
                    })
                }),
            // The MCP tool bridged and judged at turn start, if this is one.
            mcp_tool: scope
                .mcp_tools
                .iter()
                .find(|tool| tool.name() == request.name),
            owner_approval_by_kernel: true,
            session_id: scope.session_id,
            task_id: Some(scope.task_id),
            status_tx: scope.status_tx.clone(),
            channel_visibility: scope.channel_ctx.visibility,
            channel_id: scope.channel_ctx.channel_id.as_deref(),
            project_scope: scope.project_scope.as_deref(),
            trusted: scope.channel_ctx.trusted,
            user_role: scope.user_role,
            workspace_grant: scope.channel_ctx.active_workspace_grant(scope.user_role),
            mandate_authority: None,
            tool_call_id: Some(&request.call_id),
            mutation_forbidden: scope.authority.authority.level == sdk::AccessLevel::ReadOnly
                || ceiling_set,
            scope_escalation,
            turn_authority,
            trust_roots: Some((scope.trust_roots.as_slice(), scope.cwd.as_str())),
            lone_root_call: self.depth() == 0 && scope.last_batch_size.load(Ordering::Relaxed) == 1,
        }
    }

    /// How the kernel judged a call's filesystem scope, without taking the
    /// judgment (the dispatcher takes it when the call runs).
    pub fn peek_judgment(&self, scope: &TurnScope<'_>, request: &KernelCall) -> DispatchJudgment {
        let runs_as_judged = self
            .turn_tool(scope, &request.name)
            .is_some_and(|tool| tool.dispatch_facts().runs_as_judged);
        if !runs_as_judged {
            return DispatchJudgment::CheckNow;
        }
        let judged = scope
            .judged_terminal_calls
            .get(request.run_id, &request.call_id);
        #[cfg(any(test, feature = "testing"))]
        let judged = match crate::approvals::judgment_seam(scope.session_id) {
            Some(crate::approvals::JudgmentSeam::DropJudgment) => None,
            _ => judged,
        };
        match judged {
            Some(judged) => DispatchJudgment::Judged(judged.scope.as_scope_check()),
            None => DispatchJudgment::Unjudged,
        }
    }

    /// Serve [`hooks::DispatchGate`]: gate the call once and tell the
    /// kernel what was decided.
    pub async fn serve_sdk_gate(&self, scope: &TurnScope<'_>, request: GateRequest) {
        let call = &request.call;
        let outcome = match scope.gate_outcomes.get(call.run_id, &call.call_id) {
            Some(outcome) => outcome,
            None => {
                let arguments =
                    serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".into());
                let judgment = self.peek_judgment(scope, call);
                let outcome = self.sdk_gate_call(scope, call, &arguments, &judgment).await;
                scope
                    .gate_outcomes
                    .insert(call.run_id, &call.call_id, outcome.clone());
                outcome
            }
        };
        let _ = request.reply.send(outcome.decision());
    }

    /// The dispatcher's gates for one call, in order: a terminal call the
    /// kernel never judged, the session's live mode, the untrusted-reference
    /// block, the filesystem scope check and the mandate authority refuse;
    /// new nested project instructions defer the call; a duplicate
    /// `send_file` and a durable replay answer in its place. Refusals and
    /// answers are recorded in the run ledger here.
    pub async fn sdk_gate_call(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        arguments: &str,
        judgment: &DispatchJudgment,
    ) -> GateOutcome {
        // The controller-side filesystem check: protected host data and
        // undeclared writes outside the task boundary are refused; an exact
        // out-of-scope target a user may authorize goes to the tool's
        // approval prompt. A terminal call the kernel judged keeps the scope
        // it was judged (and the owner asked) with.
        let scope_check = match judgment {
            DispatchJudgment::Judged(check) => check.clone(),
            DispatchJudgment::CheckNow | DispatchJudgment::Unjudged => {
                self.sdk_scope_check(scope, &request.name, arguments)
            }
        };
        let scope_escalation = match &scope_check {
            Some(crate::turn::tool_dispatch::ScopeCheck::Escalatable { reason, escalation }) => {
                info!(
                    tool = %request.name,
                    reads = ?escalation.read_paths,
                    writes = ?escalation.write_paths,
                    "SDK engine: scope escalation routed to tool approval: {reason}"
                );
                Some(escalation.clone())
            }
            _ => None,
        };
        // The session's mode as of now (`/readonly` can arrive mid-turn),
        // judged again because a call approved after an interruption
        // resumes past the kernel's authority check.
        let ceiling = sdk::AuthorityCeiling::ceiling(scope.ceiling.as_ref()).await;
        let ceiling_denial = ceiling.as_ref().and_then(|ceiling| {
            let annotations = self
                .turn_tool(scope, &request.name)
                .map(|tool| {
                    tool.call_annotations_in(
                        arguments,
                        &aidaemon_tools::semantics::TrustRoots::new(&scope.trust_roots),
                    )
                })
                .unwrap_or_default();
            ceiling.permits(&request.name, annotations).err()
        });
        // A matched skill that references untrusted external content blocks
        // the tools that could act on it.
        let untrusted_block = scope.restrict_untrusted_reference_tools
            && crate::turn::helpers::is_untrusted_external_reference_blocked_tool(&request.name);
        let gate = if *judgment == DispatchJudgment::Unjudged {
            Err((unjudged_terminal_refusal(), "kernel_judgment_missing"))
        } else if let Some(reason) = ceiling_denial {
            Err((
                format!(
                    "[AUTHORITY BLOCKED] `{}` is outside what this session allows ({reason}). \
                     Use a permitted tool or report the limitation. {}",
                    request.name,
                    crate::authority::READ_ONLY_MODE_NOTE
                ),
                "session_mode_ceiling",
            ))
        } else if untrusted_block {
            Err((
                crate::turn::tool_dispatch::blocked_for_untrusted_external_reference_message(
                    &request.name,
                    &scope.untrusted_reference_skills,
                ),
                "untrusted_external_reference_blocked",
            ))
        } else if let Some(crate::turn::tool_dispatch::ScopeCheck::Rejected(reason)) = &scope_check
        {
            Err((
                crate::turn::tool_result_notices::ToolResultNotice::ScopeLockBlockedResult {
                    tool_name: request.name.clone(),
                    reason: reason.clone(),
                }
                .render(),
                "controller_scope_contract_rejected",
            ))
        } else {
            let facts = self.turn_tool_facts(scope, &request.name);
            let turn_authority = if facts.spawns_child_agents {
                Some(self.turn_authority_for_children(scope).await)
            } else {
                None
            };
            let ctx = self.sdk_exec_ctx(
                scope,
                request,
                ceiling.is_some(),
                scope_escalation.as_ref(),
                turn_authority.as_ref(),
            );
            self.sdk_mandate_gate(scope, &ctx, request, arguments)
                .await
                .map_err(|refusal| (refusal, "mandate_authority"))
        };
        let grant = match gate {
            Ok(grant) => grant,
            Err((refusal, reason_code)) => {
                info!(
                    session_id = scope.session_id,
                    task_id = scope.task_id,
                    tool = %request.name,
                    reason_code,
                    "SDK engine: authority gate refused the call"
                );
                let call = aidaemon_events::conversation::ToolCall {
                    id: request.call_id.clone(),
                    name: request.name.clone(),
                    arguments: arguments.to_string(),
                    extra_content: None,
                };
                let semantics = self.sdk_call_semantics(scope, &request.name, arguments);
                if let Err(error) = self
                    .persist_pre_dispatch_policy_denial(
                        scope.emitter,
                        scope.session_id,
                        self.ledger_task_id(scope),
                        &call,
                        arguments,
                        refusal.clone(),
                        reason_code,
                        semantics,
                        None,
                    )
                    .await
                {
                    warn!(task_id = scope.task_id, %error, "SDK engine: could not record a denial");
                }
                return GateOutcome::Refuse(refusal);
            }
        };
        // The daemon's own words: trusted.
        let answer = |output: sdk::ToolOutput| {
            GateOutcome::Answer(output.with_trust(sdk::SourceTrust::Trusted))
        };
        let call_semantics = self.sdk_call_semantics(scope, &request.name, arguments);
        // Before a call enters a deeper subtree, newly applicable nested
        // project instructions are loaded and the call is deliberately not
        // executed, so the model reviews them and retries (CLAUDE.md,
        // "Project Instruction Loading").
        if let Some(deferral) = self
            .sdk_load_nested_instructions(scope, request, arguments, &call_semantics)
            .await
        {
            return answer(deferral);
        }
        // The same file and caption go out once per task.
        let facts = self.turn_tool_facts(scope, &request.name);
        let send_file_key = facts
            .delivers_files
            .then(|| crate::turn::loop_utils::extract_send_file_dedupe_key_from_args(arguments))
            .flatten();
        if let Some(key) = &send_file_key {
            let already_sent = scope.sent_files.lock().is_ok_and(|sent| sent.contains(key));
            if already_sent {
                return answer(self.sdk_suppress_duplicate_send(scope, request).await);
            }
        }
        // Calls with side effects carry an idempotency key, bound to the task
        // and the canonical arguments: a repeat reuses the durable receipt
        // instead of performing the effect twice.
        let idempotency_key = self.sdk_idempotency_key(scope, request, &call_semantics);
        let preflight = crate::engine::tool_gate::durable_replay_preflight(
            self.host,
            scope.session_id,
            &request.name,
            arguments,
            idempotency_key.as_deref(),
        )
        .await;
        if let Some(output) = self
            .sdk_replay_or_block(scope, request, arguments, &preflight)
            .await
        {
            return answer(output);
        }
        GateOutcome::Clear(GateClearance { scope_check, grant })
    }

    /// Record an assistant message that calls tools before the calls run, so
    /// history and the run ledger see each call with the message that made it.
    /// The idempotency key for a step-plan call: any call that is not a pure
    /// observation and mutates, has an external side effect, or is not
    /// idempotent.
    pub fn sdk_idempotency_key(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        semantics: &aidaemon_tools::semantics::ToolCallSemantics,
    ) -> Option<String> {
        let capabilities = self
            .turn_tool(scope, &request.name)
            .map(|tool| tool.capabilities())?;
        let pure_observation = semantics.observes_state() && !semantics.mutates_state();
        let needs_key = !pure_observation
            && (semantics.mutates_state()
                || capabilities.external_side_effect
                || !capabilities.idempotent);
        needs_key.then(|| {
            let hash = crate::prompt::fingerprint::hash_canonical(&request.arguments);
            format!("exec:{}:{}:{hash}", scope.task_id, request.name)
        })
    }

    /// A duplicate `send_file` in the same task: record a no-op result and
    /// tell the model the file already went out.
    pub async fn sdk_suppress_duplicate_send(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
    ) -> sdk::ToolOutput {
        info!(
            task_id = scope.task_id,
            tool_call_id = %request.call_id,
            "SDK engine: suppressed a duplicate send_file"
        );
        let mut call_data = aidaemon_events::payloads::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(self.ledger_task_id(scope).to_string()),
        );
        call_data.turn_id = Some(scope.turn_id.to_string());
        let _ = scope.emitter.emit(EventType::ToolCall, call_data).await;
        let result_text = "Duplicate send_file suppressed: this exact file+caption was already \
                           sent in this task.";
        let mut tool = Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "tool");
        tool.content = Some(result_text.to_string());
        tool.tool_name = Some(request.name.clone());
        tool.tool_call_id = Some(request.call_id.clone());
        tool.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = self
            .append_tool_message_with_receipt_event_policy(
                scope.emitter,
                &tool,
                true,
                0,
                None,
                Some(self.ledger_task_id(scope)),
                None,
                None,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a suppressed send");
        }
        sdk::ToolOutput::text(format!(
            "{result_text}\n\n{}",
            crate::turn::system_directives::SystemDirective::DuplicateSendFileAlreadySent.render()
        ))
    }

    /// Act on the durable preflight: replay a completed earlier result
    /// (recorded as a replayed receipt) or refuse an indeterminate repeat.
    pub async fn sdk_replay_or_block(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        arguments: &str,
        preflight: &crate::engine::tool_gate::DurableReplayPreflight,
    ) -> Option<sdk::ToolOutput> {
        let call = aidaemon_events::conversation::ToolCall {
            id: request.call_id.clone(),
            name: request.name.clone(),
            arguments: arguments.to_string(),
            extra_content: None,
        };
        if let Some((operation_id, result_id, reason)) = &preflight.replay_invalidation {
            // The earlier effect no longer holds, so the call runs again.
            self.emit_decision_point(
                scope.emitter,
                scope.task_id,
                scope.model_calls.load(Ordering::Relaxed) as usize,
                aidaemon_events::payloads::DecisionType::IdempotencyReceiptInvalidated,
                format!(
                    "Reconciled durable receipt for {} is no longer authoritative",
                    request.name
                ),
                serde_json::json!({
                    "condition": "idempotency_receipt_invalidated",
                    "tool": request.name,
                    "tool_call_id": request.call_id,
                    "operation_id": operation_id,
                    "result_id": result_id,
                    "reason": reason,
                    "engine": "sdk",
                }),
            )
            .await;
        }
        if let Some(reason) = &preflight.block_reason {
            if let Err(error) = self
                .persist_pre_dispatch_outcome_with_kernel_claim(
                    scope.emitter,
                    scope.session_id,
                    scope.task_id,
                    &call,
                    arguments,
                    format!("Error: {reason}"),
                    aidaemon_tools::semantics::ToolOutcomeStatus::Blocked,
                    aidaemon_tools::semantics::ToolInvocationStage::RejectedBeforeDispatch,
                    false,
                    aidaemon_tools::semantics::ToolCallSemantics::default(),
                    None,
                    None,
                    None,
                )
                .await
            {
                warn!(task_id = scope.task_id, %error, "SDK engine: could not record a blocked repeat");
            }
            return Some(sdk::ToolOutput::error(format!("Error: {reason}")));
        }
        let previous = preflight.replayed_result.as_ref()?;
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            aidaemon_events::payloads::DecisionType::IdempotencyReceiptReplayed,
            format!(
                "Replayed durable receipt for {} without repeating I/O",
                request.name
            ),
            serde_json::json!({
                "condition": "idempotency_receipt_replayed",
                "tool": request.name,
                "tool_call_id": request.call_id,
                "receipt_outcome": previous.receipt.as_ref().map(|receipt| receipt.outcome_status),
                "engine": "sdk",
            }),
        )
        .await;
        info!(
            task_id = scope.task_id,
            tool = %request.name,
            source = %previous.tool_call_id,
            "SDK engine: replayed a durable receipt instead of repeating the call"
        );
        let mut call_data = aidaemon_events::payloads::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(self.ledger_task_id(scope).to_string()),
        );
        call_data.turn_id = Some(scope.turn_id.to_string());
        let _ = scope.emitter.emit(EventType::ToolCall, call_data).await;
        let mut receipt = previous.receipt.clone();
        if let Some(receipt) = receipt.as_mut() {
            receipt.invocation_stage = aidaemon_tools::semantics::ToolInvocationStage::Replayed;
            receipt.outcome_evidence =
                aidaemon_events::payloads::ToolOutcomeEvidenceSource::DurableReplay;
        }
        let output = replay_notice(&previous.result);
        let mut tool = Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "tool");
        tool.content = Some(output.clone());
        tool.tool_name = Some(request.name.clone());
        tool.tool_call_id = Some(request.call_id.clone());
        tool.turn_id = Some(scope.turn_id.to_string());
        let failed = !previous.success;
        if let Err(error) = self
            .append_tool_message_with_receipt_event_policy(
                scope.emitter,
                &tool,
                !failed,
                0,
                previous.error.clone(),
                Some(self.ledger_task_id(scope)),
                None,
                receipt,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a replay");
        }
        Some(sdk::ToolOutput {
            is_error: failed,
            ..sdk::ToolOutput::text(output)
        })
    }

    /// Resolve nested project instructions for a call's targets. When new
    /// guidance applies, it joins the run's system context and the call is
    /// recorded as deferred, not executed.
    pub async fn sdk_load_nested_instructions(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        arguments: &str,
        semantics: &aidaemon_tools::semantics::ToolCallSemantics,
    ) -> Option<sdk::ToolOutput> {
        if self.mandate_execution().is_some() {
            return None;
        }
        let tracker = scope.instruction_tracker.as_ref()?;
        let targets = crate::turn::project_dir::project_instruction_targets_for_tool_call(
            &*self.execution_backend_handle(),
            &request.name,
            arguments,
            semantics,
        );
        if targets.is_empty() {
            return None;
        }
        let discovered = tracker
            .lock()
            .await
            .discover_for_targets(self.execution_backend_handle().clone(), &targets)
            .await;
        let instructions = match discovered {
            Ok(Some(instructions)) => instructions,
            Ok(None) => return None,
            Err(error) => {
                warn!(
                    task_id = scope.task_id,
                    tool = %request.name,
                    %error,
                    "SDK engine: could not discover nested project instructions"
                );
                return None;
            }
        };
        let sources = instructions.source_paths();
        if let Ok(mut extra) = scope.extra_system.lock() {
            extra.push(format!(
                "[Just-in-time Project Instructions — loaded before subtree work]\n{}",
                instructions.render_for_prompt()
            ));
        }
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            scope.model_calls.load(Ordering::Relaxed) as usize,
            aidaemon_events::payloads::DecisionType::InstructionsSnapshot,
            format!("Loaded nested project instructions before {}", request.name),
            serde_json::json!({
                "condition": "jit_project_instructions_discovered",
                "tool": request.name,
                "targets": &targets,
                "sources": &sources,
                "action_executed": false,
            }),
        )
        .await;
        let notice =
            crate::turn::tool_result_notices::ToolResultNotice::ProjectInstructionsDiscovered {
                tool_name: request.name.clone(),
                sources: sources.join(", "),
            }
            .render();
        let call = aidaemon_events::conversation::ToolCall {
            id: request.call_id.clone(),
            name: request.name.clone(),
            arguments: arguments.to_string(),
            extra_content: None,
        };
        if let Err(error) = self
            .persist_pre_dispatch_outcome_with_kernel_claim(
                scope.emitter,
                scope.session_id,
                self.ledger_task_id(scope),
                &call,
                arguments,
                notice.clone(),
                aidaemon_tools::semantics::ToolOutcomeStatus::Blocked,
                aidaemon_tools::semantics::ToolInvocationStage::RejectedBeforeDispatch,
                false,
                aidaemon_tools::semantics::ToolCallSemantics::default(),
                None,
                None,
                None,
            )
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a deferred call");
        }
        Some(sdk::ToolOutput::error(notice))
    }

    /// Outside a mandate run: `Ok(None)`. Inside one: the grant for this exact
    /// call, or the refusal text the model should see.
    pub async fn sdk_mandate_gate(
        &self,
        scope: &TurnScope<'_>,
        ctx: &ToolExecCtx<'_>,
        request: &KernelCall,
        arguments: &str,
    ) -> Result<Option<aidaemon_work::mandates::MandateAuthorityGrant>, String> {
        let goal_id = match self.mandate_goal_id_for_tool_exec(ctx.task_id).await {
            Ok(Some(goal_id)) => goal_id,
            Ok(None) => return Ok(None),
            Err(error) => return Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        };
        let semantics = self.sdk_call_semantics(scope, &request.name, arguments);
        match crate::engine::tool_gate::mandate_gate_for_tool_call(
            self.host,
            &goal_id,
            &request.call_id,
            &request.name,
            arguments,
            &semantics,
        )
        .await
        {
            Ok(crate::engine::tool_gate::MandateGateDecision::NotApplicable) => Ok(None),
            Ok(crate::engine::tool_gate::MandateGateDecision::Allow { grant }) => Ok(grant),
            Ok(crate::engine::tool_gate::MandateGateDecision::Deny { reason }) => {
                let is_task_lead = self
                    .mandate_execution()
                    .as_ref()
                    .is_none_or(|fence| fence.worker_task_id == fence.root_task_id);
                Err(format!(
                    "[MANDATE AUTHORITY BLOCKED] `{}` cannot run in this decision cycle ({reason}). {}",
                    request.name,
                    aidaemon_work::mandates::policy::mandate_denial_next_step(&reason, is_task_lead)
                ))
            }
            Err(error) => Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        }
    }
}

/// What kind of work a call is, as every card shows it: from the tool's
/// typed identity, its dispatch facts and its admitted semantics.
fn step_kind(
    tool: &str,
    arguments: &str,
    facts: &aidaemon_tools::semantics::ToolDispatchFacts,
    mutates: bool,
) -> aidaemon_core::card::ActivityKind {
    aidaemon_tools::sanitize::tool_activity_kind(
        tool,
        &crate::turn::tool_summary::summarize_tool_args(tool, arguments),
        facts.spawns_child_agents,
        mutates,
    )
}
