//! Tool calls: the daemon side of `DaemonCall::Tool`. Scope and authority
//! checks, durable replay, project anchoring, dispatch through
//! `execute_tool_outcome`, receipts and the persisted tool exchange.

use super::*;

/// Let a call read the session's own spilled tool results. An oversized
/// result is saved under the session's spill directory and the model is told
/// to read it there; a task scope (a goal's bound workspace) would otherwise
/// refuse the daemon's own output. The grant is read-only and covers this
/// session's directory alone, never another session's or the spill root.
pub(super) fn grant_own_spill_reads(
    manifest: &mut crate::traits::ToolCallAccessManifest,
    session_id: &str,
) {
    let Some(dir) = crate::tools::result_spill::session_spill_dir_for_backend(session_id) else {
        return;
    };
    // macOS temp paths have a `/private` canonical twin; grant both spellings.
    let canonical = std::fs::canonicalize(&dir)
        .ok()
        .map(|path| path.to_string_lossy().into_owned());
    for path in std::iter::once(dir).chain(canonical) {
        if let Some(hint) = crate::traits::ToolTargetHint::new(
            crate::traits::ToolTargetHintKind::ProjectScope,
            path,
        ) {
            if !manifest.adapter_read_targets.contains(&hint) {
                manifest.adapter_read_targets.push(hint);
            }
        }
    }
}

/// Anchor a call without a directory to the turn's project directory: a
/// scheduled goal's terminal command runs in its
/// bound workspace, not the daemon's cwd. The model's own explicit directory
/// refines the anchor; an injected one never does (it may be a parent
/// directory picked because the target does not exist yet). Returns the
/// injected directory.
pub(super) fn anchor_call_to_project_dir(
    scope: &TurnScope<'_>,
    request: &mut ToolRequest,
) -> Option<String> {
    // A channel workspace grant scopes the file tools itself.
    if scope
        .channel_ctx
        .active_workspace_grant(scope.user_role)
        .is_some()
    {
        return None;
    }
    let arguments = serde_json::to_string(&request.arguments).ok()?;
    let mut known = scope.known_project_dir.lock().ok()?;
    if let Some(explicit) =
        crate::agent::tool_execution::project_dir_from_tool_args(&request.name, &arguments)
    {
        *known = Some(explicit);
    }
    let (updated, injected) =
        crate::agent::tool_execution::maybe_inject_project_dir_into_tool_args(
            &request.name,
            &arguments,
            known.as_deref(),
        )?;
    request.arguments = serde_json::from_str(&updated).ok()?;
    Some(injected)
}

/// Drop declared terminal read roots the task does not authorize before the
/// scope check, so a superfluous root (`/tmp`) cannot veto an otherwise
/// authorized call. Dropping a read root does not change what the call does.
pub(super) fn attenuate_sdk_read_roots(scope: &TurnScope<'_>, request: &mut ToolRequest) {
    if request.name != "terminal" {
        return;
    }
    let known = scope
        .known_project_dir
        .lock()
        .ok()
        .and_then(|known| known.clone());
    let stripped = crate::agent::tool_execution::attenuate_unauthorized_read_roots(
        &mut request.arguments,
        scope.authority.task_access.as_ref(),
        &scope.authorized_scopes,
        known.as_deref(),
    );
    if !stripped.is_empty() {
        info!(
            task_id = scope.task_id,
            stripped_read_roots = ?stripped,
            "SDK engine: attenuated unauthorized declared read roots"
        );
    }
}

/// A replayed call's result, marked so the model does not read the earlier
/// success as something that just happened (a reminder re-"set" after the
/// user had it cancelled is still cancelled).
pub(super) fn replay_notice(previous_result: &str) -> String {
    format!(
        "[Not run again: an identical call already ran earlier in this task; its earlier result is \
         repeated below. Nothing new happened now, and anything done since then (for example \
         cancelling what it created) still stands.]\n\n{previous_result}"
    )
}

/// Credit one receipt to the contract obligations it proves: a satisfied
/// mutation marks its effects, and a
/// verifiable observation marks the evidence requirements it matches.
/// Returns the obligation ids the receipt closes.
#[allow(clippy::too_many_arguments)]
pub(super) fn bind_receipt_to_contract(
    progress: &mut crate::agent::completion_contract::CompletionProgress,
    contract: &crate::agent::completion_contract::CompletionContract,
    call_id: &str,
    tool_name: &str,
    arguments: &str,
    result_text: &str,
    metadata: &crate::traits::ToolCallMetadata,
    status: crate::traits::ToolOutcomeStatus,
) -> Vec<String> {
    use crate::agent::tool_execution as evidence;
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
            crate::traits::ToolOutcomeStatus::Succeeded
                | crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult
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
pub(super) fn record_call_for_learning(
    learning: &mut crate::agent::post_task::LearningContext,
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
pub(super) fn learning_tool_summary(
    name: &str,
    arguments: &str,
    rejected: bool,
    failed: bool,
) -> String {
    let line = format!(
        "{name}({})",
        crate::agent::agent_helpers::summarize_tool_args(name, arguments)
    );
    if rejected {
        format!("{line} [REJECTED]")
    } else if failed {
        format!("{line} [FAILED]")
    } else {
        line
    }
}

impl Agent {
    /// Judge the call's access manifest against the task's filesystem
    /// authority. `None`: in scope (or not a known tool).
    pub(super) fn sdk_scope_check(
        &self,
        scope: &TurnScope<'_>,
        tool_name: &str,
        arguments: &str,
    ) -> Option<crate::agent::tool_execution::ScopeCheck> {
        let tool = self.tools.iter().find(|tool| tool.name() == tool_name)?;
        let mut manifest = tool
            .prepare_invocation(arguments)
            .map(|prepared| prepared.access_manifest)
            .unwrap_or_else(|_| tool.call_access_manifest(arguments));
        manifest.adapter_read_targets = tool
            .adapter_owned_access_manifest(arguments)
            .adapter_read_targets;
        grant_own_spill_reads(&mut manifest, scope.session_id);
        let known = scope
            .known_project_dir
            .lock()
            .ok()
            .and_then(|known| known.clone());
        crate::agent::tool_execution::access_manifest_scope_check(
            tool_name,
            &manifest,
            scope.authority.task_access.as_ref(),
            &scope.authorized_scopes,
            known.as_deref(),
        )
    }

    pub(super) async fn dispatch_sdk_tool_call(
        &self,
        scope: &TurnScope<'_>,
        mut request: ToolRequest,
    ) {
        let injected_dir = anchor_call_to_project_dir(scope, &mut request);
        attenuate_sdk_read_roots(scope, &mut request);
        let arguments = serde_json::to_string(&request.arguments).unwrap_or_else(|_| "{}".into());
        let mut output = self
            .run_sdk_tool_call(scope, &request, &arguments, injected_dir.as_deref())
            .await;
        // Refusals, deferrals, replays and suppressed duplicates are the
        // daemon's own words: trusted.
        output.trust.get_or_insert(sdk::SourceTrust::Trusted);
        let _ = request.reply.send(Ok(output));
    }

    /// Shadow comparison: record when the assessor's own authority view would
    /// have refused a call the explicit authority admitted. Never blocks or
    /// alters the call.
    async fn record_assessor_would_refuse(
        &self,
        scope: &TurnScope<'_>,
        tool_name: &str,
        arguments: &str,
    ) {
        let read_only = self
            .tools
            .iter()
            .find(|tool| tool.name() == tool_name)
            .map(|tool| tool.call_annotations(arguments).read_only);
        let assessed = crate::agent::closeout::ClosingAuthority::from_authority(
            scope.contract.authority_as_assessed(),
        );
        let Some(reason) = assessed.refusal_reason(tool_name, read_only) else {
            return;
        };
        self.emit_decision_point(
            scope.emitter,
            scope.task_id,
            0,
            crate::events::DecisionType::GateTelemetry,
            "assessor would have refused this call",
            serde_json::json!({
                "condition": "assessor_would_refuse",
                "tool": tool_name,
                "reason": reason,
            }),
        )
        .await;
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
        let egress = self
            .tools
            .iter()
            .find(|tool| tool.name() == tool_name)
            .is_some_and(|tool| {
                tool.call_annotations_in(
                    arguments,
                    &crate::traits::TrustRoots::new(&scope.trust_roots),
                )
                .egress
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
    fn turn_authority_for_children(
        &self,
        scope: &TurnScope<'_>,
    ) -> crate::agent::authority::InheritedAuthority {
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
        }
    }

    /// Run one call and record it: a `ToolCall` event, then the tool message with a typed receipt, so the run ledger
    /// sees every operation while the run is still going.
    pub(super) async fn run_sdk_tool_call(
        &self,
        scope: &TurnScope<'_>,
        request: &ToolRequest,
        arguments: &str,
        injected_dir: Option<&str>,
    ) -> sdk::ToolOutput {
        // The controller-side filesystem check: protected host data and undeclared writes outside the task
        // boundary are refused; an exact out-of-scope target a user may
        // authorize goes to the tool's approval prompt.
        let scope_check = self.sdk_scope_check(scope, &request.name, arguments);
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
        // A child spawned by this call inherits what this turn has now.
        let turn_authority =
            (request.name == "spawn_agent").then(|| self.turn_authority_for_children(scope));
        // The session's mode as of now (`/readonly` can arrive mid-turn).
        let ceiling = sdk::AuthorityCeiling::ceiling(scope.ceiling.as_ref()).await;
        let mut ctx = ToolExecCtx {
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
                || ceiling.is_some(),
            scope_escalation: scope_escalation.as_ref(),
            turn_authority: turn_authority.as_ref(),
            trust_roots: Some((scope.trust_roots.as_slice(), scope.cwd.as_str())),
        };
        let started = Instant::now();
        let call_semantics = self
            .semantics_for_exact_tool_call(&request.name, arguments)
            .await;
        // The turn's resolved authority (level and tool set) was enforced
        // by the kernel's `AuthorityHook` before this call reached the
        // dispatcher. Here: the session's mode as of now, the
        // untrusted-reference block, the filesystem scope check, then for
        // mandate runs the per-call mandate authority gate, which issues the
        // bounded grant or refuses the call.
        // A call approved after an interruption resumes past the kernel's
        // authority check, so the mode is judged again for every call.
        let ceiling_denial = ceiling.as_ref().and_then(|ceiling| {
            let annotations = self
                .tools
                .iter()
                .find(|tool| tool.name() == request.name)
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
        let gate = if let Some(reason) = ceiling_denial {
            Err((
                format!(
                    "[AUTHORITY BLOCKED] `{}` is outside what this session allows ({reason}). \
                     Use a permitted tool or report the limitation.",
                    request.name
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
            self.sdk_mandate_gate(&ctx, request, arguments)
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
                let semantics = self
                    .semantics_for_exact_tool_call(&request.name, arguments)
                    .await;
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
                return sdk::ToolOutput::error(refusal);
            }
        };
        ctx.mandate_authority = grant.as_ref();

        // Before a call enters a deeper subtree, newly applicable nested
        // project instructions are loaded and the call is deliberately not
        // executed, so the model reviews them and retries (CLAUDE.md,
        // "Project Instruction Loading").
        if let Some(deferral) = self
            .sdk_load_nested_instructions(scope, request, arguments, &call_semantics)
            .await
        {
            return deferral;
        }

        // The same file and caption go out once per task.
        let send_file_key = (request.name == "send_file")
            .then(|| crate::agent::loop_utils::extract_send_file_dedupe_key_from_args(arguments))
            .flatten();
        if let Some(key) = &send_file_key {
            let already_sent = scope.sent_files.lock().is_ok_and(|sent| sent.contains(key));
            if already_sent {
                return self.sdk_suppress_duplicate_send(scope, request).await;
            }
        }
        // Calls with side effects carry an idempotency key, bound to
        // the task and the canonical arguments: a repeat reuses the durable
        // receipt instead of performing the effect twice.
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
            return output;
        }
        // Shadow comparison, once per call that actually executes: after
        // deferral, duplicate suppression and durable replay, so a call that
        // is deferred, suppressed or replayed is not counted (the shadow
        // data feeds the decision to turn the assessor off).
        if self.sdk_engine.task_assessment == crate::config::TaskAssessmentSetting::Shadow {
            self.record_assessor_would_refuse(scope, &request.name, arguments)
                .await;
        }
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
        if request.name != "track_requirements" {
            let (name, summary) = crate::tools::sanitize::user_facing_tool_activity(
                &request.name,
                &crate::agent::agent_helpers::summarize_tool_args(&request.name, arguments),
                scope.channel_ctx.visibility,
            );
            if let Some(status_tx) = &scope.status_tx {
                let _ = status_tx.try_send(StatusUpdate::ToolStart { name, summary });
            }
        }
        let (output, images, metadata, receipt) = match self
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
                // The detach transition, decided at the typed receipt. A
                // launch the model asked to detach is a long-lived service
                // (server, tunnel, watcher) that never "finishes": its useful
                // result is its startup output, so the run continues and the
                // model reports it. Work the daemon pushed to the background
                // is the handoff unless the assessed task shape keeps
                // independent inline work going.
                if crate::agent::tool_execution::tool_result_indicates_background_detach(
                    &request.name,
                    &output,
                    &outcome.metadata,
                ) {
                    let directive = if scope.continue_inline_after_background
                        || outcome.metadata.detached
                    {
                        crate::agent::system_directives::SystemDirective::BackgroundProcessContinue
                    } else {
                        // With notifications on, the result follows by
                        // itself: a plain handoff, no pid or tool jargon.
                        // Without them, the ack says how to check on it.
                        let reply = if outcome.metadata.completion_notifications_enabled {
                            crate::agent::post_task::build_friendly_background_handoff(
                                scope.user_text,
                            )
                        } else {
                            crate::agent::tool_execution::build_background_detach_ack(
                                &request.name,
                                &output,
                                &outcome.metadata,
                            )
                        };
                        scope.end_with(reply, TaskOutcome::Partial);
                        if outcome.metadata.completion_notifications_enabled {
                            scope.background_handoff.store(true, Ordering::Relaxed);
                        }
                        crate::agent::system_directives::SystemDirective::BackgroundHandoff {
                            notifications_active: outcome.metadata.completion_notifications_enabled,
                        }
                    };
                    output = format!("{output}\n\n{}", directive.render());
                }
                let satisfied = !status.is_failure();
                // An executor's typed blocker ends its run with the structured
                // summary; the untrusted-data wrapper is removed first so
                // reply sanitization does not empty it.
                if satisfied && request.name == "report_blocker" && self.task_id.is_some() {
                    scope.end_with(
                        crate::agent::completion_checks::strip_untrusted_wrapper(&output)
                            .to_string(),
                        TaskOutcome::Partial,
                    );
                }
                // A committed non-action mandate decision (WAIT/ASK/STOP)
                // completes the review: read from the durable decision, the
                // run ends without another model call.
                if satisfied
                    && request.name == "manage_mandates"
                    && request.arguments.get("action").and_then(Value::as_str)
                        == Some("record_decision")
                    && self.mandate_execution.is_some()
                {
                    if let Some(summary) =
                        crate::agent::mandate_run::committed_non_action_mandate_summary(self).await
                    {
                        scope.end_with(summary, TaskOutcome::Succeeded);
                    }
                }
                // A tool's direct response answers a root turn's lone call.
                if satisfied
                    && self.depth == 0
                    && scope.last_batch_size.load(Ordering::Relaxed) == 1
                    && !(outcome.metadata.background_started || outcome.metadata.detached)
                {
                    if let Some(reply) = outcome.metadata.direct_response.clone() {
                        scope.end_with(reply, TaskOutcome::Succeeded);
                    }
                }
                (output, images, Some(outcome.metadata), Some(receipt))
            }
            Err(error) => {
                // A tool that returns `Err` made an explicit typed failure: it
                // gets a `FailedPermanent` receipt, so the run ledger and the task outcome see it.
                let prepared = self
                    .tools
                    .iter()
                    .find(|tool| tool.name() == request.name)
                    .and_then(|tool| tool.prepare_invocation(arguments).ok());
                let mut metadata = crate::traits::ToolCallMetadata {
                    outcome_status: Some(crate::traits::ToolOutcomeStatus::FailedPermanent),
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
                (fitted, Vec::new(), Some(metadata), Some(receipt))
            }
        };
        let failed = receipt
            .as_ref()
            .map(|receipt| receipt.outcome_status.is_failure() || receipt.contract_rejected)
            .unwrap_or(true);
        let duration_ms = started.elapsed().as_millis() as u64;
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
            let update = if request.name == "track_requirements" {
                crate::agent::tool_execution::parse_checklist_from_result(&output)
                    .map(|text| StatusUpdate::Checklist { text })
            } else {
                let source = if matches!(request.name.as_str(), "manage_memories" | "manage_people")
                {
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
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
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
        if request.name == "track_requirements" && !failed {
            if let Some(declared) =
                crate::tools::track_requirements::executor_expectations_from_checklist_arguments(
                    scope.task_id,
                    arguments,
                )
            {
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
        // Who authored what this call returned: only executed calls carry
        // the tool's own class (or what the tool reported for this exact
        // output); every other output is the daemon's.
        let reported = metadata.as_ref().and_then(|metadata| metadata.output_trust);
        let trust = self
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
            .map(|tool| {
                reported.unwrap_or_else(|| {
                    tool.output_trust(
                        arguments,
                        &crate::traits::TrustRoots::new(&scope.trust_roots)
                            .with_cwd(Some(&scope.cwd)),
                    )
                })
            })
            .unwrap_or_default();
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
        sdk::ToolOutput {
            is_error: failed,
            ..sdk::ToolOutput::text(model_view).with_images(images)
        }
        .with_trust(trust)
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
        request: &ToolRequest,
        semantics: &crate::traits::ToolCallSemantics,
    ) -> Option<String> {
        let capabilities = self
            .tools
            .iter()
            .find(|tool| tool.name() == request.name)
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
        request: &ToolRequest,
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
        request: &ToolRequest,
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
        request: &ToolRequest,
        arguments: &str,
        semantics: &crate::traits::ToolCallSemantics,
    ) -> Option<sdk::ToolOutput> {
        if self.mandate_execution.is_some() {
            return None;
        }
        let tracker = scope.instruction_tracker.as_ref()?;
        let targets = crate::agent::tool_execution::project_instruction_targets_for_tool_call(
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
            .discover_for_targets(crate::execution::active_execution_backend(), &targets)
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
        ctx: &ToolExecCtx<'_>,
        request: &ToolRequest,
        arguments: &str,
    ) -> Result<Option<crate::traits::MandateAuthorityGrant>, String> {
        let goal_id = match self.mandate_goal_id_for_tool_exec(ctx).await {
            Ok(Some(goal_id)) => goal_id,
            Ok(None) => return Ok(None),
            Err(error) => return Err(format!("[MANDATE AUTHORITY BLOCKED] {error}")),
        };
        let semantics = self
            .semantics_for_exact_tool_call(&request.name, arguments)
            .await;
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
