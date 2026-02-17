use super::types::AbortOnDrop;
use crate::agent::*;
use crate::execution_policy::PolicyBundle;
use regex::RegexBuilder;
use serde_json::{json, Value};

pub(super) struct ToolExecutionIoResult {
    pub result_text: String,
    pub tool_duration_ms: u64,
}

pub(super) struct ToolExecutionIoCtx<'a> {
    pub effective_arguments: &'a str,
    pub injected_project_dir: Option<&'a str>,
    pub session_id: &'a str,
    pub task_id: &'a str,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
    pub channel_ctx: &'a ChannelContext,
    pub user_role: UserRole,
    pub heartbeat: &'a Option<Arc<AtomicU64>>,
    pub emitter: &'a crate::events::EventEmitter,
    pub policy_bundle: &'a PolicyBundle,
}

impl Agent {
    pub(super) async fn execute_tool_call_io(
        &self,
        tc: &ToolCall,
        ctx: &ToolExecutionIoCtx<'_>,
    ) -> ToolExecutionIoResult {
        send_status(
            ctx.status_tx,
            StatusUpdate::ToolStart {
                name: tc.name.clone(),
                summary: summarize_tool_args(&tc.name, ctx.effective_arguments),
            },
        );

        // Emit ToolCall event
        let _ = ctx
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
                    Some(format!("{}:{}:{}", ctx.task_id, tc.name, tc.id)),
                    Some(ctx.policy_bundle.policy.policy_rev),
                    Some(ctx.policy_bundle.risk_score),
                ),
            )
            .await;

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
        let _heartbeat_keeper =
            if matches!(tc.name.as_str(), "cli_agent" | "terminal" | "spawn_agent") {
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

        let result = self
            .execute_tool_with_watchdog(
                &tc.name,
                ctx.effective_arguments,
                &tool_exec::ToolExecCtx {
                    session_id: ctx.session_id,
                    task_id: Some(ctx.task_id),
                    status_tx: ctx.status_tx.clone(),
                    channel_visibility: ctx.channel_ctx.visibility,
                    channel_id: ctx.channel_ctx.channel_id.as_deref(),
                    trusted: ctx.channel_ctx.trusted,
                    user_role: ctx.user_role,
                },
            )
            .await;

        // _heartbeat_keeper is dropped here (or when the scope ends),
        // which triggers AbortOnDrop to cancel the background task.
        drop(_heartbeat_keeper);
        touch_heartbeat(ctx.heartbeat);
        let mut result_is_err = result.is_err();
        let mut result_text = match result {
            Ok(text) => {
                // Sanitize and wrap untrusted tool outputs
                if !crate::tools::sanitize::is_trusted_tool(&tc.name) {
                    let sanitized = crate::tools::sanitize::sanitize_external_content(&text);
                    crate::tools::sanitize::wrap_untrusted_output(&tc.name, &sanitized)
                } else if tc.name == "terminal" {
                    crate::tools::sanitize::strip_internal_control_markers(&text)
                } else {
                    text
                }
            }
            Err(e) => format!("Error: {}", e),
        };

        if result_is_err && tc.name == "edit_file" {
            if let Some(recovered_text) = self
                .maybe_retry_edit_file_not_found_recovery(&tc.arguments, &result_text, ctx)
                .await
            {
                result_text = recovered_text;
                result_is_err = false;
            }
        }

        if let Some(injected_dir) = ctx.injected_project_dir {
            result_text = format!(
                "{}\n\n[SYSTEM] Path was auto-injected from known project context: {}",
                result_text, injected_dir
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

        // Compress large tool results to save context budget
        if self.context_window_config.enabled {
            result_text = crate::memory::context_window::compress_tool_result(
                &tc.name,
                &result_text,
                self.context_window_config.max_tool_result_chars,
            );
        }

        let tool_duration_ms = tool_exec_start.elapsed().as_millis().min(u64::MAX as u128) as u64;
        ToolExecutionIoResult {
            result_text,
            tool_duration_ms,
        }
    }

    async fn maybe_retry_edit_file_not_found_recovery(
        &self,
        arguments: &str,
        initial_error: &str,
        ctx: &ToolExecutionIoCtx<'_>,
    ) -> Option<String> {
        if !initial_error.contains("Text not found in ") {
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
            trusted: ctx.channel_ctx.trusted,
            user_role: ctx.user_role,
        };

        // Deterministic self-recovery path:
        // 1) Read current file state.
        // 2) Attempt one whitespace-tolerant mapping from old_text to exact on-disk text.
        // 3) Retry edit_file once with exact recovered old_text.
        let read_args = json!({ "path": path }).to_string();
        let read_probe_ok = self
            .execute_tool_with_watchdog("read_file", &read_args, &exec_ctx)
            .await
            .is_ok();

        let resolved_path = crate::tools::fs_utils::validate_path(&path).ok()?;
        let file_content = tokio::fs::read_to_string(&resolved_path).await.ok()?;
        let recovered_old_text =
            recover_old_text_with_whitespace_tolerance(&file_content, &old_text)?;

        if recovered_old_text == old_text {
            return None;
        }

        let mut retry_args = args;
        retry_args["old_text"] = Value::String(recovered_old_text);
        let retry_args_str = serde_json::to_string(&retry_args).ok()?;
        match self
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
                    "{}\n\n[SYSTEM] Internal edit_file recovery succeeded: {}. Retried once with exact on-disk text matched via whitespace-tolerant mapping.",
                    retry_output, read_note
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
