use super::types::AbortOnDrop;
use crate::agent::*;
use crate::execution_policy::PolicyBundle;
use regex::RegexBuilder;
use serde_json::{json, Value};

pub(super) struct ToolExecutionIoResult {
    pub result_text: String,
    pub tool_duration_ms: u64,
    pub result_metadata: crate::traits::ToolCallMetadata,
}

pub(super) struct ToolExecutionIoCtx<'a> {
    pub effective_arguments: &'a str,
    /// Model selected for this turn — used to resolve per-model tool result caps.
    pub model: &'a str,
    pub idempotency_key: Option<&'a str>,
    pub injected_project_dir: Option<&'a str>,
    pub project_scope: Option<&'a str>,
    pub session_id: &'a str,
    pub task_id: &'a str,
    pub iteration: usize,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub channel_ctx: &'a ChannelContext,
    pub user_role: UserRole,
    pub heartbeat: &'a Option<Arc<AtomicU64>>,
    pub emitter: &'a crate::events::EventEmitter,
    pub policy_bundle: &'a PolicyBundle,
    /// Set by the correction gate when this specific tool call has already
    /// been classified as allowed for unattended execution.
    /// False on all normal (non-correction) paths.
    pub correction_preapproved: bool,
    /// When true, the `_trusted_session` enrichment flag must NOT be injected
    /// into tool args (the correction sandbox overrides trusted-session semantics).
    /// False on all normal paths.
    pub suppress_trusted_session: bool,
    pub mandate_authority: Option<&'a crate::traits::MandateAuthorityGrant>,
    /// Mandate results must remain bounded inline; spilling would create an
    /// ungranted local file containing potentially sensitive observations.
    pub mandate_execution: bool,
}

fn should_replay_durable_result(
    result: &crate::events::ToolResultData,
    tool_is_idempotent: bool,
) -> bool {
    // A producer-owned retryable failure may be attempted again only when the
    // adapter also declares repeated invocation safe. Every other durable
    // outcome—including backgrounded and blocked—is replayed to avoid
    // duplicating an indeterminate or already-completed effect.
    !(tool_is_idempotent
        && result.receipt.as_ref().is_some_and(|receipt| {
            receipt.outcome_status == crate::traits::ToolOutcomeStatus::FailedRetryable
        }))
}

pub(super) async fn execute_tool_call_io(
    agent: &Agent,
    tc: &ToolCall,
    ctx: &ToolExecutionIoCtx<'_>,
) -> ToolExecutionIoResult {
    let (start_label, start_summary) = crate::tools::sanitize::user_facing_tool_activity(
        &tc.name,
        &summarize_tool_args(&tc.name, ctx.effective_arguments),
        ctx.channel_ctx.visibility,
    );
    // `track_requirements` renders its own checklist as the live surface (emitted
    // as StatusUpdate::Checklist right after the call). Skip its generic ToolStart
    // so the surface shows the plan, not a raw "track_requirements…" line first.
    if tc.name != "track_requirements" {
        send_status(
            ctx.status_tx,
            StatusUpdate::ToolStart {
                name: start_label,
                summary: start_summary,
            },
        );
    }

    // Claim/replay preflight happens before the new ToolCall row is appended.
    // A completed key reuses its durable receipt; a claimed key with no result
    // is indeterminate after a crash and is blocked for reconciliation.
    let tool_is_idempotent = agent
        .tools
        .iter()
        .find(|tool| tool.name() == tc.name && tool.is_available())
        .is_some_and(|tool| tool.capabilities().idempotent);
    let mut replayed_result: Option<crate::events::ToolResultData> = None;
    let mut replay_invalidation_reason: Option<String> = None;
    let mut idempotency_block_reason: Option<String> = None;
    if let Some(idempotency_key) = ctx.idempotency_key {
        match agent
            .event_store
            .get_tool_result_by_idempotency_key(ctx.session_id, idempotency_key)
            .await
        {
            Ok(Some(result)) => {
                if should_replay_durable_result(&result, tool_is_idempotent) {
                    let receipt_succeeded = result.receipt.as_ref().is_some_and(|receipt| {
                        receipt.outcome_status == crate::traits::ToolOutcomeStatus::Succeeded
                    });
                    let replay_decision = if receipt_succeeded {
                        match agent
                            .tools
                            .iter()
                            .find(|tool| tool.name() == tc.name && tool.is_available())
                        {
                            Some(tool) => {
                                tool.durable_replay_decision(ctx.effective_arguments).await
                            }
                            None => crate::traits::DurableReplayDecision::Replay,
                        }
                    } else {
                        crate::traits::DurableReplayDecision::Replay
                    };
                    match replay_decision {
                        crate::traits::DurableReplayDecision::Replay => {
                            replayed_result = Some(result);
                        }
                        crate::traits::DurableReplayDecision::Reexecute { reason } => {
                            replay_invalidation_reason = Some(reason);
                        }
                        crate::traits::DurableReplayDecision::Block { reason } => {
                            idempotency_block_reason = Some(reason);
                        }
                    }
                }
            }
            Ok(None) => match agent
                .event_store
                .has_unresolved_tool_call_for_idempotency_key(ctx.session_id, idempotency_key)
                .await
            {
                Ok(true) => {
                    idempotency_block_reason = Some(format!(
                        "A prior `{}` invocation with idempotency key `{}` has no durable result. Its side effect is indeterminate after interruption; inspect the target before issuing a different operation.",
                        tc.name, idempotency_key
                    ));
                }
                Ok(false) => {}
                Err(error) => {
                    idempotency_block_reason = Some(format!(
                        "Could not verify whether `{}` was already executed ({error}); refusing a potentially duplicate side effect.",
                        tc.name
                    ));
                }
            },
            Err(error) => {
                idempotency_block_reason = Some(format!(
                    "Could not read the durable receipt for `{}` ({error}); refusing a potentially duplicate side effect.",
                    tc.name
                ));
            }
        }
    }

    // Emit ToolCall event
    let claim_result = ctx
        .emitter
        .emit(
            EventType::ToolCall,
            ToolCallData::from_tool_call(
                tc.id.clone(),
                tc.name.clone(),
                serde_json::from_str(ctx.effective_arguments).unwrap_or(serde_json::json!({})),
                Some(ctx.task_id.to_string()),
            )
            .with_policy_metadata(
                ctx.idempotency_key
                    .map(str::to_string)
                    .or_else(|| Some(format!("{}:{}:{}", ctx.task_id, tc.name, tc.id))),
                Some(ctx.policy_bundle.policy.policy_rev),
                Some(ctx.policy_bundle.risk_score),
            ),
        )
        .await;

    if let (Some(idempotency_key), Err(error)) = (ctx.idempotency_key, claim_result) {
        let reason = format!(
            "Could not persist the durable execution claim for `{}` ({error}); refusing to run a mutation without replay protection.",
            tc.name
        );
        agent
            .emit_warning_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::IdempotencyIndeterminateBlock,
                "Blocked mutation because its durable claim could not be persisted".to_string(),
                serde_json::json!({
                    "condition": "idempotency_claim_persist_failed",
                    "tool": tc.name,
                    "tool_call_id": tc.id,
                    "idempotency_key": idempotency_key,
                    "error": error.to_string(),
                }),
            )
            .await;
        let semantics = agent
            .tools
            .iter()
            .find(|tool| tool.name() == tc.name && tool.is_available())
            .map(|tool| tool.call_semantics(ctx.effective_arguments))
            .unwrap_or_default();
        return ToolExecutionIoResult {
            result_text: format!("Error: {reason}"),
            tool_duration_ms: 0,
            result_metadata: crate::traits::ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Blocked),
                semantics,
                ..crate::traits::ToolCallMetadata::default()
            },
        };
    }

    if let Some(reason) = replay_invalidation_reason {
        agent
            .emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::IdempotencyReceiptInvalidated,
                format!(
                    "Re-executing {} because its durable effect no longer matches current state",
                    tc.name
                ),
                serde_json::json!({
                    "condition": "idempotency_receipt_invalidated",
                    "tool": tc.name,
                    "tool_call_id": tc.id,
                    "idempotency_key": ctx.idempotency_key,
                    "reason": reason,
                }),
            )
            .await;
    }

    if let Some(previous) = replayed_result {
        let mut metadata = previous
            .receipt
            .as_ref()
            .map(crate::events::ToolReceiptV1::to_metadata)
            .unwrap_or_default();
        metadata.attachments = previous.attachments;
        agent
            .emit_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::IdempotencyReceiptReplayed,
                format!("Replayed durable receipt for {} without repeating I/O", tc.name),
                serde_json::json!({
                    "condition": "idempotency_receipt_replayed",
                    "tool": tc.name,
                    "tool_call_id": tc.id,
                    "idempotency_key": ctx.idempotency_key,
                    "receipt_outcome": previous.receipt.as_ref().map(|receipt| receipt.outcome_status),
                }),
            )
            .await;
        return ToolExecutionIoResult {
            result_text: previous.result,
            tool_duration_ms: 0,
            result_metadata: metadata,
        };
    }

    if let Some(reason) = idempotency_block_reason {
        let semantics = agent
            .tools
            .iter()
            .find(|tool| tool.name() == tc.name && tool.is_available())
            .map(|tool| tool.call_semantics(ctx.effective_arguments))
            .unwrap_or_default();
        agent
            .emit_warning_decision_point(
                ctx.emitter,
                ctx.task_id,
                ctx.iteration,
                DecisionType::IdempotencyIndeterminateBlock,
                format!("Blocked indeterminate replay for {}", tc.name),
                serde_json::json!({
                    "condition": "idempotency_indeterminate_block",
                    "tool": tc.name,
                    "tool_call_id": tc.id,
                    "idempotency_key": ctx.idempotency_key,
                    "reason": reason,
                }),
            )
            .await;
        return ToolExecutionIoResult {
            result_text: format!("Error: {reason}"),
            tool_duration_ms: 0,
            result_metadata: crate::traits::ToolCallMetadata {
                outcome_status: Some(crate::traits::ToolOutcomeStatus::Blocked),
                semantics,
                ..crate::traits::ToolCallMetadata::default()
            },
        };
    }

    let tool_exec_start = Instant::now();
    touch_heartbeat(ctx.heartbeat);

    // For long-running tools (cli_agent, terminal), spawn a background
    // task that keeps the heartbeat alive so the channel-level stale
    // watchdog doesn't auto-cancel the task while the tool is still
    // actively working.
    // Wrap in AbortOnDrop so the keeper is automatically cancelled if
    // handle_message is dropped by an outer select! (e.g. stale watchdog).
    // Without this, a detached keeper loop continues touching the heartbeat
    // forever, preventing the typing indicator's stale check from firing.
    let _heartbeat_keeper = if matches!(tc.name.as_str(), "cli_agent" | "terminal" | "spawn_agent")
    {
        ctx.heartbeat.as_ref().map(|hb| {
            let hb = Arc::clone(hb);
            AbortOnDrop(tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    hb.store(now, Ordering::Relaxed);
                }
            }))
        })
    } else {
        None
    };

    let result = agent
        .execute_tool_with_watchdog_outcome(
            &tc.name,
            ctx.effective_arguments,
            &tool_exec::ToolExecCtx {
                session_id: ctx.session_id,
                task_id: Some(ctx.task_id),
                status_tx: ctx.status_tx.clone(),
                channel_visibility: ctx.channel_ctx.visibility,
                channel_id: ctx.channel_ctx.channel_id.as_deref(),
                project_scope: ctx.project_scope,
                trusted: ctx.channel_ctx.trusted,
                user_role: ctx.user_role,
                workspace_grant: ctx.channel_ctx.active_workspace_grant(ctx.user_role),
                correction_preapproved: ctx.correction_preapproved,
                suppress_trusted_session: ctx.suppress_trusted_session,
                mandate_authority: ctx.mandate_authority,
                mandate_tool_call_id: Some(tc.id.as_str()),
            },
        )
        .await;

    // _heartbeat_keeper is dropped here (or when the scope ends),
    // which triggers AbortOnDrop to cancel the background task.
    drop(_heartbeat_keeper);
    touch_heartbeat(ctx.heartbeat);
    let mut result_metadata = crate::traits::ToolCallMetadata::default();
    let mut result_is_err = result.is_err();
    let mut result_text = match result {
        Ok(outcome) => {
            result_metadata = outcome.metadata;
            let text = outcome.output;
            // Sanitize and wrap untrusted tool outputs
            if !crate::tools::sanitize::is_trusted_tool(&tc.name) {
                let body = if result_metadata.untrusted_verbatim {
                    text
                } else {
                    crate::tools::sanitize::sanitize_external_content(&text)
                };
                crate::tools::sanitize::wrap_untrusted_output(&tc.name, &body)
            } else if tc.name == "terminal" {
                crate::tools::sanitize::strip_internal_control_markers(&text)
            } else {
                text
            }
        }
        Err(e) => {
            result_metadata.transport_error = Some(e.to_string());
            // A legacy tool returning `Err` has made an explicit typed Rust
            // failure. Classify it once at this adapter boundary instead of
            // teaching the orchestration loop phrases from its display text.
            // Retryable conditions (notably watchdog timeouts) return a typed
            // outcome directly before reaching this branch.
            result_metadata.outcome_status =
                Some(crate::traits::ToolOutcomeStatus::FailedPermanent);
            format!("Error: {}", e)
        }
    };

    if result_is_err && tc.name == "edit_file" {
        if let Some(recovered_text) =
            maybe_retry_edit_file_not_found_recovery(agent, &tc.arguments, &result_text, ctx).await
        {
            result_text = recovered_text;
            result_is_err = false;
            result_metadata.transport_error = None;
        }
    }

    if let Some(injected_dir) = ctx.injected_project_dir {
        result_text = format!(
            "{}\n\n{}",
            result_text,
            ToolResultNotice::PathAutoInjectedFromProjectContext {
                injected_dir: injected_dir.to_string(),
            }
            .render()
        );
    }

    // `cli_agent` errors can be extremely large (process output, stack traces).
    // Truncate aggressively to prevent context blow-up and runaway retries.
    if tc.name == "cli_agent" && result_is_err {
        let char_len = result_text.chars().count();
        if char_len > 2000 {
            let head: String = result_text.chars().take(500).collect();
            let tail: String = result_text
                .chars()
                .rev()
                .take(500)
                .collect::<Vec<char>>()
                .into_iter()
                .rev()
                .collect();
            result_text = format!(
                "{}\n\n[... cli_agent error output truncated ({} chars total) ...]\n\n{}",
                head, char_len, tail
            );
        }
    }

    // Compress large tool results to save context budget. The cap is
    // per-model so small-context local models get tighter results while
    // big-context models keep more.
    if agent.context_window_config.enabled {
        let max_chars = agent.context_window_config.tool_result_chars_for(ctx.model);
        // read_file results with typed metadata get line-boundary paging with
        // an explicit continuation hint instead of destructive mid-drop
        // compression — the model can keep reading from the exact cut point.
        let read_file_metadata = (tc.name == "read_file" && !result_is_err)
            .then_some(result_metadata.read_file.as_ref())
            .flatten();
        if let Some(read_metadata) = read_file_metadata {
            if result_text.chars().count() > max_chars {
                result_text =
                    crate::tools::render_read_file_output_within(read_metadata, max_chars);
                if let Some(injected_dir) = ctx.injected_project_dir {
                    result_text = format!(
                        "{}\n\n{}",
                        result_text,
                        ToolResultNotice::PathAutoInjectedFromProjectContext {
                            injected_dir: injected_dir.to_string(),
                        }
                        .render()
                    );
                }
            }
        } else {
            // Structured API bodies: pretty-printed JSON costs ~2.5-3x the
            // tokens of its compact form with identical information (observed
            // live: an inline clinical-trials response helped time out the
            // compose call). Compact it losslessly BEFORE the size checks —
            // this can also keep a result inline that would otherwise spill.
            // Small bodies stay pretty-printed for model readability.
            const STRUCTURED_JSON_COMPACT_MIN_CHARS: usize = 2_000;
            if matches!(tc.name.as_str(), "http_request" | "web_fetch")
                && !result_is_err
                && result_text.chars().count() > STRUCTURED_JSON_COMPACT_MIN_CHARS
            {
                if let Some(compacted) = crate::utils::compact_embedded_json(&result_text) {
                    tracing::info!(
                        tool = %tc.name,
                        original_chars = result_text.chars().count(),
                        compact_chars = compacted.chars().count(),
                        "Losslessly compacted embedded JSON in tool result"
                    );
                    result_text = compacted;
                }
            }
            // Oversized successful results are spilled to a temp file when the
            // model has a filesystem recovery path (read_file or terminal): it
            // then pages / jq / greps the full data instead of losing the
            // middle to lossy head+tail compression. No recovery path → fall
            // back to lossy compression. Errors are never spilled.
            let original_chars = result_text.chars().count();
            let over_cap = original_chars > max_chars;
            // The computer_use accessibility tree is the model's working data: it
            // must stay inline and complete so the model can target an element
            // directly. Spilling it to a file (whose notice tells the model to
            // read_file/grep it) sends the planner into a read_file/terminal loop
            // instead of clicking; lossy head+tail compression would drop mid-tree
            // controls (e.g. a feed's Like buttons). So never spill or compress it.
            let keep_inline = tc.name == "computer_use" || result_metadata.preserve_inline;
            let has_fs_recovery = agent
                .tools
                .iter()
                .any(|t| matches!(t.name(), "read_file" | "terminal") && t.is_available());
            let spilled = if over_cap
                && !result_is_err
                && has_fs_recovery
                && !keep_inline
                && !ctx.mandate_execution
            {
                crate::tools::result_spill::build_spilled_preview_for_backend(
                    &tc.name,
                    ctx.session_id,
                    &result_text,
                    max_chars,
                )
                .await
            } else {
                None
            };
            let was_spilled = spilled.is_some();
            result_text = match spilled {
                Some(preview) => {
                    tracing::info!(
                        target: "inline_dump_spill",
                        tool = %tc.name,
                        "Large tool result spilled to file this turn"
                    );
                    preview
                }
                // Keep the full GUI tree inline (do not compress away the middle).
                None if keep_inline => result_text,
                None => crate::memory::context_window::compress_tool_result(
                    &tc.name,
                    &result_text,
                    max_chars,
                ),
            };
            // Record how the harness transformed the result before the model saw
            // it — spill/compress/kept-inline + how many chars were hidden. This
            // is the mutation that caused the read_file/terminal derailment and
            // was previously only visible in stdout; persist it so a single
            // db_probe --task query explains "why did the model do that".
            if over_cap && !result_is_err {
                let shown_chars = result_text.chars().count();
                let decision_type = if was_spilled {
                    "tool_result_spilled"
                } else if keep_inline {
                    "tool_result_kept_inline"
                } else {
                    "tool_result_compressed"
                };
                let _ = ctx
                    .emitter
                    .emit(
                        EventType::DecisionPoint,
                        serde_json::json!({
                            "decision_type": decision_type,
                            "name": tc.name,
                            "task_id": ctx.task_id,
                            "original_chars": original_chars,
                            "shown_chars": shown_chars,
                            "hidden_chars": original_chars.saturating_sub(shown_chars),
                            "max_chars": max_chars,
                        }),
                    )
                    .await;
            }
        }
    }

    let tool_duration_ms = tool_exec_start.elapsed().as_millis().min(u64::MAX as u128) as u64;
    ToolExecutionIoResult {
        result_text,
        tool_duration_ms,
        result_metadata,
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    fn result_with_status(
        status: crate::traits::ToolOutcomeStatus,
    ) -> crate::events::ToolResultData {
        let metadata = crate::traits::ToolCallMetadata {
            outcome_status: Some(status),
            semantics: crate::traits::ToolCallSemantics::mutation(),
            ..crate::traits::ToolCallMetadata::default()
        };
        crate::events::ToolResultData {
            message_id: None,
            tool_call_id: "call-1".to_string(),
            name: "probe".to_string(),
            result: "result".to_string(),
            success: status == crate::traits::ToolOutcomeStatus::Succeeded,
            duration_ms: 1,
            error: None,
            task_id: None,
            annotations: Vec::new(),
            turn_id: None,
            attachments: Vec::new(),
            receipt: Some(crate::events::ToolReceiptV1::from_metadata(
                &metadata,
                status,
                crate::events::ToolOutcomeEvidenceSource::ToolReported,
                Some("exec:e1:probe:abc".to_string()),
            )),
        }
    }

    #[test]
    fn durable_replay_only_reexecutes_safe_retryable_failures() {
        let retryable = result_with_status(crate::traits::ToolOutcomeStatus::FailedRetryable);
        assert!(!should_replay_durable_result(&retryable, true));
        assert!(should_replay_durable_result(&retryable, false));

        for status in [
            crate::traits::ToolOutcomeStatus::Succeeded,
            crate::traits::ToolOutcomeStatus::CompletedWithNegativeResult,
            crate::traits::ToolOutcomeStatus::FailedPermanent,
            crate::traits::ToolOutcomeStatus::Blocked,
            crate::traits::ToolOutcomeStatus::Backgrounded,
        ] {
            assert!(should_replay_durable_result(
                &result_with_status(status),
                true
            ));
        }
    }
}

// NOTE (3b): `edit_file` retry calls here bypass the correction gate because
// this is an internal deterministic recovery path, not a model-directed retry.
// Future specs that allow model-directed mutating retries MUST route through
// the correction gate in `execute_tool_call_io` — not through this function.
async fn maybe_retry_edit_file_not_found_recovery(
    agent: &Agent,
    arguments: &str,
    initial_error: &str,
    ctx: &ToolExecutionIoCtx<'_>,
) -> Option<String> {
    if !initial_error.contains("Text not found in ") {
        return None;
    }

    // Mandate grants are bound to the exact action arguments and consume one
    // attempt. This recovery rewrites the edit arguments, so it must return to
    // deliberation instead of silently reusing or minting authority.
    if ctx.mandate_authority.is_some() {
        return None;
    }

    // The generic recovery below performs a direct backend read so it can map
    // whitespace before retrying. Scoped collaborators must never enter a
    // secondary path that is not itself rooted and canonicalized by the
    // workspace gate; they can retry with an exact edit after `read_file`.
    if ctx
        .channel_ctx
        .active_workspace_grant(ctx.user_role)
        .is_some()
    {
        return None;
    }

    let args: Value = serde_json::from_str(arguments).ok()?;
    let path = args.get("path")?.as_str()?.to_string();
    let old_text = args.get("old_text")?.as_str()?.to_string();
    if old_text.trim().is_empty() {
        return None;
    }

    let exec_ctx = tool_exec::ToolExecCtx {
        session_id: ctx.session_id,
        task_id: Some(ctx.task_id),
        status_tx: ctx.status_tx.clone(),
        channel_visibility: ctx.channel_ctx.visibility,
        channel_id: ctx.channel_ctx.channel_id.as_deref(),
        project_scope: ctx.project_scope,
        trusted: ctx.channel_ctx.trusted,
        user_role: ctx.user_role,
        workspace_grant: ctx.channel_ctx.active_workspace_grant(ctx.user_role),
        correction_preapproved: ctx.correction_preapproved,
        suppress_trusted_session: ctx.suppress_trusted_session,
        mandate_authority: None,
        mandate_tool_call_id: None,
    };

    // Deterministic self-recovery path:
    // 1) Read current file state.
    // 2) Attempt one whitespace-tolerant mapping from old_text to exact on-disk text.
    // 3) Retry edit_file once with exact recovered old_text.
    let read_args = json!({ "path": path }).to_string();
    let read_probe_ok = agent
        .execute_tool_with_watchdog("read_file", &read_args, &exec_ctx)
        .await
        .is_ok();

    let backend = crate::execution::active_execution_backend();
    let resolved_path = backend.resolve_path(&path).await.ok()?;
    let file_content = String::from_utf8(backend.read(&resolved_path).await.ok()?).ok()?;
    let recovered_old_text = recover_old_text_with_whitespace_tolerance(&file_content, &old_text)?;

    if recovered_old_text == old_text {
        return None;
    }

    let mut retry_args = args;
    retry_args["old_text"] = Value::String(recovered_old_text);
    let retry_args_str = serde_json::to_string(&retry_args).ok()?;
    match agent
        .execute_tool_with_watchdog("edit_file", &retry_args_str, &exec_ctx)
        .await
    {
        Ok(retry_output) => {
            let read_note = if read_probe_ok {
                "read_file probe succeeded"
            } else {
                "read_file probe failed, but direct file read succeeded"
            };
            Some(format!(
                "{}\n\n{}",
                retry_output,
                ToolResultNotice::InternalEditFileRecoverySucceeded {
                    read_note: read_note.to_string(),
                }
                .render()
            ))
        }
        Err(e) => {
            warn!(
                path = %path,
                error = %e,
                "Internal edit_file recovery retry failed"
            );
            None
        }
    }
}

fn build_whitespace_tolerant_pattern(old_text: &str) -> Option<String> {
    let mut pattern = String::new();
    let mut has_non_whitespace = false;
    let mut in_ws = false;

    for ch in old_text.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                pattern.push_str(r"\s+");
                in_ws = true;
            }
        } else {
            has_non_whitespace = true;
            in_ws = false;
            pattern.push_str(&regex::escape(&ch.to_string()));
        }
    }

    if has_non_whitespace {
        Some(pattern)
    } else {
        None
    }
}

fn recover_old_text_with_whitespace_tolerance(content: &str, old_text: &str) -> Option<String> {
    let pattern = build_whitespace_tolerant_pattern(old_text)?;
    let regex = RegexBuilder::new(&pattern)
        .dot_matches_new_line(true)
        .build()
        .ok()?;

    let mut matches = regex.find_iter(content);
    let first = matches.next()?;
    if matches.next().is_some() {
        return None;
    }
    Some(content[first.start()..first.end()].to_string())
}

#[cfg(test)]
mod tests {
    use super::{build_whitespace_tolerant_pattern, recover_old_text_with_whitespace_tolerance};

    #[test]
    fn whitespace_tolerant_pattern_collapses_runs() {
        let pattern = build_whitespace_tolerant_pattern("foo   bar\tbaz\nqux").unwrap();
        assert_eq!(pattern, "foo\\s+bar\\s+baz\\s+qux");
    }

    #[test]
    fn recover_old_text_with_indentation_mismatch() {
        let content = "<section>\n    <h1>Dog World</h1>\n</section>\n";
        let old_text = "<section>\n  <h1>Dog World</h1>\n</section>\n";
        let recovered = recover_old_text_with_whitespace_tolerance(content, old_text).unwrap();
        assert_eq!(recovered, "<section>\n    <h1>Dog World</h1>\n</section>\n");
    }

    #[test]
    fn recover_old_text_returns_none_when_ambiguous() {
        let content = "alpha beta\nalpha    beta\n";
        let old_text = "alpha beta";
        assert!(recover_old_text_with_whitespace_tolerance(content, old_text).is_none());
    }
}
