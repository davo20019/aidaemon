//! Tool calls: the daemon side of `DaemonCall::Tool`. Scope and authority
//! checks, durable replay, project anchoring, dispatch through
//! `execute_tool_outcome`, receipts and the persisted tool exchange.

use super::*;

pub(in crate::agent) use aidaemon_runtime::engine::tool_calls::*;

/// Stands in for a call's result between the tool returning and the
/// after-tool hook writing what the model reads.
const RESULT_PENDING: &str = "[tool result pending]";

/// The kernel's judgment of a dispatched call, removed from `judged`:
/// `None` when the tool's calls are not judged in canonical form
/// (`dispatch_facts().runs_as_judged`, the terminal, whose approval ladder
/// the kernel runs and the tool then skips), `Some(None)` when the call was
/// never judged in `run`.
pub(super) fn take_terminal_judgment(
    judged: &crate::agent::approvals::JudgedCalls,
    run: sdk::RunId,
    tool: Option<&dyn crate::traits::Tool>,
    call_id: &str,
) -> Option<Option<crate::agent::approvals::JudgedTerminal>> {
    let runs_as_judged = tool.is_some_and(|tool| tool.dispatch_facts().runs_as_judged);
    runs_as_judged.then(|| judged.take(run, call_id))
}

/// Put a dispatched call in its canonical form: a terminal call the kernel
/// judged runs exactly as judged, with the scope it was judged with; any
/// other call is canonicalized now. The model's explicit directory refines
/// the turn's known project directory. Returns the injected directory and
/// the kernel's judgment.
pub(super) fn canonicalize_dispatched_call(
    scope: &TurnScope<'_>,
    tool: Option<&dyn crate::traits::Tool>,
    request: &mut ToolRequest,
) -> (Option<String>, DispatchJudgment) {
    let taken = take_terminal_judgment(
        &scope.judged_terminal_calls,
        request.run_id,
        tool,
        &request.call_id,
    );
    #[cfg(test)]
    let taken = match crate::agent::approvals::judgment_seam(scope.session_id) {
        Some(crate::agent::approvals::JudgmentSeam::DropJudgment) => taken.map(|_| None),
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

impl Agent {
    /// The tool this turn runs under `name`: a daemon tool, else an
    /// MCP-registry tool bridged for the turn.
    pub(super) fn turn_tool<'s>(
        &'s self,
        scope: &'s TurnScope<'_>,
        name: &str,
    ) -> Option<&'s Arc<dyn crate::traits::Tool>> {
        self.tools
            .iter()
            .find(|tool| tool.name() == name)
            .or_else(|| scope.mcp_tools.iter().find(|tool| tool.name() == name))
    }

    /// What a call of the tool this turn runs under `name` does: the
    /// instance judged at turn start, never a fresh registry lookup (an MCP
    /// server reloaded mid-run could answer differently).
    pub(super) fn sdk_call_semantics(
        &self,
        scope: &TurnScope<'_>,
        name: &str,
        arguments: &str,
    ) -> crate::traits::ToolCallSemantics {
        self.turn_tool(scope, name)
            .map(|tool| tool.call_semantics(arguments))
            .unwrap_or_default()
    }

    /// The dispatch facts of the tool this turn runs under `name` (nothing
    /// special for an unknown tool).
    /// Who authored what this call returned: only executed calls carry the
    /// tool's own class (or what the tool reported for this exact output);
    /// every other output is the daemon's.
    fn call_output_trust(
        &self,
        scope: &TurnScope<'_>,
        name: &str,
        arguments: &str,
        metadata: Option<&crate::traits::ToolCallMetadata>,
    ) -> sdk::SourceTrust {
        let reported = metadata.and_then(|metadata| metadata.output_trust);
        self.turn_tool(scope, name)
            .map(|tool| {
                reported.unwrap_or_else(|| {
                    tool.output_trust(
                        arguments,
                        &crate::traits::TrustRoots::new(&scope.trust_roots)
                            .with_cwd(Some(&scope.cwd)),
                    )
                })
            })
            .unwrap_or_default()
    }

    pub(super) fn turn_tool_facts(
        &self,
        scope: &TurnScope<'_>,
        name: &str,
    ) -> crate::traits::ToolDispatchFacts {
        self.turn_tool(scope, name)
            .map(|tool| tool.dispatch_facts())
            .unwrap_or_default()
    }

    /// Judge the call's access manifest against the task's filesystem
    /// authority. `None`: in scope (or not a known tool).
    pub(super) fn sdk_scope_check(
        &self,
        scope: &TurnScope<'_>,
        tool_name: &str,
        arguments: &str,
    ) -> Option<crate::agent::tool_execution::ScopeCheck> {
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
                backend: &*self.execution_backend,
                session_id: scope.session_id,
                task_access: scope.authority.task_access.as_ref(),
                authorized_scopes: &scope.authorized_scopes,
                scopes_are_guess: scope.scopes_are_guess,
                known_project_dir: known.as_deref(),
                paths: &self.daemon_paths,
            },
        )
    }

    pub(super) async fn dispatch_sdk_tool_call(
        &self,
        scope: &TurnScope<'_>,
        mut request: ToolRequest,
    ) {
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
    async fn record_unrestricted_egress(
        &self,
        scope: &TurnScope<'_>,
        tool_name: &str,
        arguments: &str,
    ) {
        if scope.authority.owner_level != Some(crate::config::OwnerLevel::Unrestricted)
            || !scope
                .untrusted_seen
                .load(std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        let egress = self.turn_tool(scope, tool_name).is_some_and(|tool| {
            tool.call_annotations_in(
                arguments,
                &crate::traits::TrustRoots::new(&scope.trust_roots),
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
            crate::events::DecisionType::GateTelemetry,
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
    async fn turn_authority_for_children(
        &self,
        scope: &TurnScope<'_>,
    ) -> crate::agent::authority::InheritedAuthority {
        // What this run has reached so far, with what its transcript and its
        // own parent already held: the child starts from these identities.
        let known_resources = self
            .known_resources(
                scope.session_id,
                Some(scope.task_id),
                &scope.projected_source_turn_ids,
            )
            .await;
        crate::agent::authority::InheritedAuthority {
            authority: scope.authority.authority.clone(),
            owner_level: scope.authority.owner_level,
            task_access: scope.authority.task_access.clone(),
            mode_session: Some(self.mode_session_for(scope.session_id)),
            untrusted_context: scope.untrusted_seen.load(Ordering::SeqCst),
            origin_request: Some(
                scope
                    .origin_request
                    .chars()
                    .take(crate::agent::authority::INHERITED_REQUEST_CHARS)
                    .collect(),
            ),
            origin_trusted: scope.origin_trusted,
            trusted_roots: scope.trust_roots.clone(),
            // What the owner declined in this run so far (the run's
            // declines already hold what this turn itself inherited).
            declined: crate::agent::authority::union_declines(
                &crate::agent::authority::union_declines(
                    self.inherited_authority
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
            lineage: self
                .inherited_authority
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
    pub(super) async fn run_sdk_tool_call(
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
            Some(crate::agent::tool_execution::ScopeCheck::Escalatable { escalation, .. }) => {
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
            .then(|| crate::agent::loop_utils::extract_send_file_dedupe_key_from_args(arguments))
            .flatten();
        let idempotency_key = self.sdk_idempotency_key(scope, request, &call_semantics);
        self.record_unrestricted_egress(scope, &request.name, arguments)
            .await;
        let mut call_data = crate::events::ToolCallData::from_tool_call(
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
            let (name, summary) = crate::tools::sanitize::user_facing_tool_activity(
                &request.name,
                &crate::agent::agent_helpers::summarize_tool_args(&request.name, arguments),
                scope.channel_ctx.visibility,
            );
            if let Some(status_tx) = &scope.status_tx {
                let _ = status_tx.try_send(StatusUpdate::ToolStart { name, summary });
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
                crate::agent::tool_execution::complete_tool_result_semantics(
                    &request.name,
                    arguments,
                    &call_semantics,
                    &mut outcome.metadata,
                );
                let status = crate::agent::loop_utils::classify_tool_outcome_status(
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
                    crate::traits::prepare_invocation(tool.as_ref(), arguments).ok()
                });
                let mut metadata = crate::traits::ToolCallMetadata {
                    outcome_status: Some(
                        if error
                            .downcast_ref::<crate::traits::NothingSentError>()
                            .is_some()
                        {
                            crate::traits::ToolOutcomeStatus::FailedRetryable
                        } else {
                            crate::traits::ToolOutcomeStatus::FailedPermanent
                        },
                    ),
                    transport_error: Some(error.to_string()),
                    invocation_stage: if error
                        .downcast_ref::<crate::agent::tool_exec::UnknownToolError>()
                        .is_some()
                    {
                        crate::traits::ToolInvocationStage::RejectedBeforeDispatch
                    } else {
                        crate::traits::ToolInvocationStage::Dispatched
                    },
                    ..crate::traits::ToolCallMetadata::default()
                };
                if let Some(prepared) = prepared {
                    metadata.receipt_kind = prepared.receipt_kind;
                    metadata.access_manifest = Some(prepared.access_manifest);
                }
                crate::agent::tool_execution::complete_tool_result_semantics(
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
        // The first untrusted read records the turn as tainted at once, so a
        // run that never finishes still taints later turns that see it.
        if trust == sdk::SourceTrust::Untrusted {
            self.read_untrusted
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if trust == sdk::SourceTrust::Untrusted
            && !scope
                .untrusted_seen
                .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            if let Err(error) = crate::agent::untrusted::record_tainted_turn(
                self.state.as_ref(),
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
    pub(super) async fn finish_sdk_tool_call(&self, scope: &TurnScope<'_>, check: AfterCallCheck) {
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
    async fn record_remote_change(
        &self,
        tool: &str,
        arguments: &str,
        metadata: &crate::traits::ToolCallMetadata,
        status: crate::traits::ToolOutcomeStatus,
    ) {
        let Some(task_id) = self.task_id.as_deref() else {
            return;
        };
        if !changed_remotely(metadata, status) {
            return;
        }
        let activity = crate::traits::TaskActivity {
            id: 0,
            task_id: task_id.to_string(),
            activity_type: crate::traits::REMOTE_CHANGE_ACTIVITY.to_string(),
            tool_name: Some(tool.to_string()),
            tool_args: Some(arguments.to_string()),
            result: None,
            success: Some(true),
            tokens_used: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        if let Err(error) = self.state.log_task_activity(&activity).await {
            warn!(task_id, %error, "Could not record a remote change on its task");
        }
    }

    async fn record_sdk_tool_result(
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
                    crate::agent::loop_utils::tool_outcome_evidence_source(&outcome.metadata);
                // What the model sees: untrusted output wrapped, oversized
                // output capped. Fitted before the receipt so its provenance
                // records how much the model saw; status and contract binding
                // stay on the raw result.
                let fitted = self
                    .fit_tool_result_for_model(
                        &crate::agent::tool_result_fit::FitScope {
                            tool_name: &request.name,
                            session_id: scope.session_id,
                            task_id: scope.task_id,
                            model: scope.model,
                            emitter: scope.emitter,
                            mandate_execution: self.mandate_execution.is_some(),
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
                let mut receipt = crate::events::ToolReceiptV1::from_metadata(
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
                    output.push_str(&crate::utils::render_truncation_notice(info));
                }
                if let Some(dir) = injected_dir {
                    output.push_str("\n\n");
                    output.push_str(
                        &crate::agent::ToolResultNotice::PathAutoInjectedFromProjectContext {
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
                let (images, skipped) = sdk_images(&observed, &self.vision_config, scope.model);
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
                // Track the jobs this run left running whose result the
                // daemon will deliver: a notifying, non-detached launch opens
                // one; a receipt that observed a job's end closes one.
                if outcome.metadata.background_started
                    && outcome.metadata.completion_notifications_enabled
                    && !outcome.metadata.detached
                    && outcome.metadata.outcome_status
                        != Some(crate::traits::ToolOutcomeStatus::FailedPermanent)
                {
                    scope.open_background_jobs.fetch_add(1, Ordering::Relaxed);
                }
                if outcome.metadata.background_settled {
                    let _ = scope.open_background_jobs.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |open| open.checked_sub(1),
                    );
                }
                if outcome.metadata.background_started || outcome.metadata.background_handoff {
                    // A call the tool itself typed as a permanent failure hands
                    // nothing off, whatever it started: the run goes on.
                    let typed_failure = outcome.metadata.outcome_status
                        == Some(crate::traits::ToolOutcomeStatus::FailedPermanent);
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
                        // With notifications on, the result follows by
                        // itself: a plain handoff, no pid or tool jargon.
                        // Without them, the ack says how to check on it.
                        let reply = if outcome.metadata.completion_notifications_enabled {
                            scope.background_handoff.store(true, Ordering::Relaxed);
                            crate::agent::post_task::build_friendly_background_handoff(
                                scope.user_text,
                            )
                        } else {
                            crate::agent::tool_execution::build_background_detach_ack(
                                &output,
                                &outcome.metadata,
                            )
                        };
                        if let Some(finish) = finish.as_mut() {
                            finish.reply = Some(reply);
                        }
                        crate::agent::system_directives::SystemDirective::BackgroundHandoff {
                            notifications_active: outcome.metadata.completion_notifications_enabled,
                        }
                    } else {
                        finish = None;
                        crate::agent::system_directives::SystemDirective::BackgroundProcessContinue
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
                                crate::agent::completion_checks::strip_untrusted_wrapper(&output)
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
                        &crate::agent::tool_result_fit::FitScope {
                            tool_name: &request.name,
                            session_id: scope.session_id,
                            task_id: scope.task_id,
                            model: scope.model,
                            emitter: scope.emitter,
                            mandate_execution: self.mandate_execution.is_some(),
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
                let status = crate::traits::ToolOutcomeStatus::FailedPermanent;
                let evidence = crate::agent::loop_utils::tool_outcome_evidence_source(&metadata);
                let mut receipt = crate::events::ToolReceiptV1::from_metadata(
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
        // Budget progress counts distinct successful operations: repeating
        // an identical call is not new evidence.
        if !failed {
            let operation = format!(
                "{}:{}",
                request.name,
                crate::agent::prefix_fingerprint::hash_canonical(&request.arguments)
            );
            if let Ok(mut seen) = scope.succeeded_operations.lock() {
                let new_operation = seen.insert(operation);
                if new_operation {
                    scope.succeeded_calls.fetch_add(1, Ordering::Relaxed);
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
                crate::agent::tool_execution::parse_checklist_from_result(&output)
                    .map(|text| StatusUpdate::Checklist { text })
            } else {
                let source = if facts.status_from_arguments {
                    crate::agent::agent_helpers::summarize_tool_args(&request.name, arguments)
                } else {
                    crate::agent::tool_execution::summarize_completed_tool_result(&output)
                };
                let (name, summary) = crate::tools::sanitize::user_facing_tool_activity(
                    &request.name,
                    &source,
                    scope.channel_ctx.visibility,
                );
                Some(StatusUpdate::ToolComplete { name, summary })
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
                    && crate::agent::tool_execution::should_build_external_action_ack(&output))
                .then(|| {
                    crate::agent::tool_execution::build_external_action_completion_ack(&output)
                });
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
                crate::tools::track_requirements::executor_expectations_from_checklist_arguments(
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
    fn sdk_exec_ctx<'s>(
        &'s self,
        scope: &'s TurnScope<'_>,
        request: &'s KernelCall,
        ceiling_set: bool,
        scope_escalation: Option<&'s crate::traits::ScopeEscalation>,
        turn_authority: Option<&'s crate::agent::authority::InheritedAuthority>,
    ) -> ToolExecCtx<'s> {
        ToolExecCtx {
            file_fence: scope
                .spawn_family
                .active_coordination()
                .map(|coordination| {
                    Arc::new(crate::agent::coordination::CallFence {
                        coordination: coordination.clone(),
                        member: scope.session_id.to_string(),
                        call_id: request.call_id.clone(),
                        backend: self.execution_backend.clone(),
                        escalator: crate::agent::coordination::Escalator::new(
                            self.side_calls.clone(),
                            self.state.clone(),
                            self.state.clone(),
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
            lone_root_call: self.depth == 0 && scope.last_batch_size.load(Ordering::Relaxed) == 1,
        }
    }

    /// How the kernel judged a call's filesystem scope, without taking the
    /// judgment (the dispatcher takes it when the call runs).
    fn peek_judgment(&self, scope: &TurnScope<'_>, request: &KernelCall) -> DispatchJudgment {
        let runs_as_judged = self
            .turn_tool(scope, &request.name)
            .is_some_and(|tool| tool.dispatch_facts().runs_as_judged);
        if !runs_as_judged {
            return DispatchJudgment::CheckNow;
        }
        let judged = scope
            .judged_terminal_calls
            .get(request.run_id, &request.call_id);
        #[cfg(test)]
        let judged = match crate::agent::approvals::judgment_seam(scope.session_id) {
            Some(crate::agent::approvals::JudgmentSeam::DropJudgment) => None,
            _ => judged,
        };
        match judged {
            Some(judged) => DispatchJudgment::Judged(judged.scope.as_scope_check()),
            None => DispatchJudgment::Unjudged,
        }
    }

    /// Serve [`hooks::DispatchGate`]: gate the call once and tell the
    /// kernel what was decided.
    pub(super) async fn serve_sdk_gate(&self, scope: &TurnScope<'_>, request: GateRequest) {
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
    pub(super) async fn sdk_gate_call(
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
            Some(crate::agent::tool_execution::ScopeCheck::Escalatable { reason, escalation }) => {
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
                        &crate::traits::TrustRoots::new(&scope.trust_roots),
                    )
                })
                .unwrap_or_default();
            ceiling.permits(&request.name, annotations).err()
        });
        // A matched skill that references untrusted external content blocks
        // the tools that could act on it.
        let untrusted_block = scope.restrict_untrusted_reference_tools
            && crate::agent::is_untrusted_external_reference_blocked_tool(&request.name);
        let gate = if *judgment == DispatchJudgment::Unjudged {
            Err((unjudged_terminal_refusal(), "kernel_judgment_missing"))
        } else if let Some(reason) = ceiling_denial {
            Err((
                format!(
                    "[AUTHORITY BLOCKED] `{}` is outside what this session allows ({reason}). \
                     Use a permitted tool or report the limitation. {}",
                    request.name,
                    crate::agent::authority::READ_ONLY_MODE_NOTE
                ),
                "session_mode_ceiling",
            ))
        } else if untrusted_block {
            Err((
                crate::agent::tool_execution::blocked_for_untrusted_external_reference_message(
                    &request.name,
                    &scope.untrusted_reference_skills,
                ),
                "untrusted_external_reference_blocked",
            ))
        } else if let Some(crate::agent::tool_execution::ScopeCheck::Rejected(reason)) =
            &scope_check
        {
            Err((
                crate::agent::ToolResultNotice::ScopeLockBlockedResult {
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
                let call = crate::traits::ToolCall {
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
            .then(|| crate::agent::loop_utils::extract_send_file_dedupe_key_from_args(arguments))
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
        let preflight = crate::agent::tool_execution::durable_replay_preflight(
            self,
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
    pub(super) async fn persist_sdk_tool_call_message(
        &self,
        scope: &TurnScope<'_>,
        response: &sdk::ModelResponse,
    ) {
        let calls: Vec<crate::traits::ToolCall> = response
            .tool_calls
            .iter()
            .map(|call| crate::traits::ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
                extra_content: None,
            })
            .collect();
        let mut assistant =
            Message::new_runtime(Uuid::new_v4().to_string(), scope.session_id, "assistant");
        let text = response.text.trim();
        assistant.content = (!text.is_empty()).then(|| text.to_string());
        assistant.tool_calls_json = serde_json::to_string(&calls).ok();
        assistant.turn_id = Some(scope.turn_id.to_string());
        if let Err(error) = self
            .append_assistant_message_with_event(scope.emitter, &assistant, scope.model, None, None)
            .await
        {
            warn!(task_id = scope.task_id, %error, "SDK engine: could not record a tool-call message");
        }
    }

    /// The idempotency key for a step-plan call: any call that is not a pure
    /// observation and mutates, has an external side effect, or is not
    /// idempotent.
    pub(super) fn sdk_idempotency_key(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        semantics: &crate::traits::ToolCallSemantics,
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
            let hash = crate::agent::prefix_fingerprint::hash_canonical(&request.arguments);
            format!("exec:{}:{}:{hash}", scope.task_id, request.name)
        })
    }

    /// A duplicate `send_file` in the same task: record a no-op result and
    /// tell the model the file already went out.
    pub(super) async fn sdk_suppress_duplicate_send(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
    ) -> sdk::ToolOutput {
        info!(
            task_id = scope.task_id,
            tool_call_id = %request.call_id,
            "SDK engine: suppressed a duplicate send_file"
        );
        let mut call_data = crate::events::ToolCallData::from_tool_call(
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
            crate::agent::system_directives::SystemDirective::DuplicateSendFileAlreadySent.render()
        ))
    }

    /// Act on the durable preflight: replay a completed earlier result
    /// (recorded as a replayed receipt) or refuse an indeterminate repeat.
    pub(super) async fn sdk_replay_or_block(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        arguments: &str,
        preflight: &crate::agent::tool_execution::DurableReplayPreflight,
    ) -> Option<sdk::ToolOutput> {
        let call = crate::traits::ToolCall {
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
                crate::events::DecisionType::IdempotencyReceiptInvalidated,
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
                    crate::traits::ToolOutcomeStatus::Blocked,
                    crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
                    false,
                    crate::traits::ToolCallSemantics::default(),
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
            crate::events::DecisionType::IdempotencyReceiptReplayed,
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
        let mut call_data = crate::events::ToolCallData::from_tool_call(
            request.call_id.clone(),
            request.name.clone(),
            request.arguments.clone(),
            Some(self.ledger_task_id(scope).to_string()),
        );
        call_data.turn_id = Some(scope.turn_id.to_string());
        let _ = scope.emitter.emit(EventType::ToolCall, call_data).await;
        let mut receipt = previous.receipt.clone();
        if let Some(receipt) = receipt.as_mut() {
            receipt.invocation_stage = crate::traits::ToolInvocationStage::Replayed;
            receipt.outcome_evidence = crate::events::ToolOutcomeEvidenceSource::DurableReplay;
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
    pub(super) async fn sdk_load_nested_instructions(
        &self,
        scope: &TurnScope<'_>,
        request: &KernelCall,
        arguments: &str,
        semantics: &crate::traits::ToolCallSemantics,
    ) -> Option<sdk::ToolOutput> {
        if self.mandate_execution.is_some() {
            return None;
        }
        let tracker = scope.instruction_tracker.as_ref()?;
        let targets = crate::agent::tool_execution::project_instruction_targets_for_tool_call(
            &*self.execution_backend,
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
            .discover_for_targets(self.execution_backend.clone(), &targets)
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
            crate::events::DecisionType::InstructionsSnapshot,
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
        let notice = crate::agent::ToolResultNotice::ProjectInstructionsDiscovered {
            tool_name: request.name.clone(),
            sources: sources.join(", "),
        }
        .render();
        let call = crate::traits::ToolCall {
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
                crate::traits::ToolOutcomeStatus::Blocked,
                crate::traits::ToolInvocationStage::RejectedBeforeDispatch,
                false,
                crate::traits::ToolCallSemantics::default(),
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
    pub(super) async fn sdk_mandate_gate(
        &self,
        scope: &TurnScope<'_>,
        ctx: &ToolExecCtx<'_>,
        request: &KernelCall,
        arguments: &str,
    ) -> Result<Option<crate::traits::MandateAuthorityGrant>, String> {
        let goal_id = match self.mandate_goal_id_for_tool_exec(ctx).await {
            Ok(Some(goal_id)) => goal_id,
            Ok(None) => return Ok(None),
            Err(error) => return Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        };
        let semantics = self.sdk_call_semantics(scope, &request.name, arguments);
        match crate::agent::tool_execution::mandate_gate_for_tool_call(
            self,
            &goal_id,
            &request.call_id,
            &request.name,
            arguments,
            &semantics,
        )
        .await
        {
            Ok(crate::agent::tool_execution::MandateGateDecision::NotApplicable) => Ok(None),
            Ok(crate::agent::tool_execution::MandateGateDecision::Allow { grant }) => Ok(grant),
            Ok(crate::agent::tool_execution::MandateGateDecision::Deny { reason }) => {
                let is_task_lead = self
                    .mandate_execution
                    .as_ref()
                    .is_none_or(|fence| fence.worker_task_id == fence.root_task_id);
                Err(format!(
                    "[MANDATE AUTHORITY BLOCKED] `{}` cannot run in this decision cycle ({reason}). {}",
                    request.name,
                    crate::mandates::mandate_denial_next_step(&reason, is_task_lead)
                ))
            }
            Err(error) => Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        }
    }
}
