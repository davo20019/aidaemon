//! Running a review job: the prompt, the reviewer process and its events.

use crate::*;

pub(crate) fn build_review_prompt(
    reviewer: &str,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    context: &str,
    notes: Option<&str>,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are a strict software reviewer.\n");
    prompt.push_str("Analyze the provided context and return ONLY JSON with this schema:\n");
    prompt.push_str("{\"verdict\":\"approve|needs_changes\",\"summary\":\"...\",\"findings\":[{\"severity\":\"critical|high|medium|low\",\"file\":\"path/to/file\",\"line\":123,\"issue\":\"...\",\"fix\":\"...\"}],\"blocking\":[\"...\"],\"risks\":[\"...\"],\"suggestions\":[\"...\"]}\n");
    prompt.push_str("Rules:\n");
    prompt.push_str("- Prefer concrete findings over generic advice.\n");
    prompt.push_str("- \"verdict\" is exactly \"approve\" or \"needs_changes\"; any other value, or none, counts as \"needs_changes\".\n");
    prompt.push_str("- If there is any meaningful risk, use verdict \"needs_changes\".\n");
    prompt.push_str("- Put actionable issues in findings with file + line when possible.\n");
    prompt.push_str("- Use concise, specific issue and fix text (no fluff).\n");
    prompt.push_str("- Keep each list item concise and actionable.\n");
    prompt.push_str("- No markdown, no extra text, JSON only.\n\n");
    prompt.push_str(&format!(
        "Reviewer: {}\nScope: {}\nDiff base: {}\n\n",
        reviewer,
        scope.as_str(),
        diff_base.as_str()
    ));
    if let Some(n) = notes {
        let cleaned = truncate_with_note(n.trim(), REVIEW_MAX_SECTION_CHARS);
        if !cleaned.trim().is_empty() {
            prompt.push_str("User notes:\n");
            prompt.push_str(&cleaned);
            prompt.push_str("\n\n");
        }
    }
    prompt.push_str("Review context:\n");
    prompt.push_str(context);
    truncate_with_note(&prompt, REVIEW_MAX_CONTEXT_CHARS)
}

pub(crate) async fn run_capture_command(
    command: &str,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    clear_env: bool,
) -> anyhow::Result<std::process::Output> {
    let mut cmd = Command::new(command);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.kill_on_drop(true);
    if clear_env {
        cmd.env_clear();
        for (k, v) in build_child_env() {
            cmd.env(k, v);
        }
    }
    cmd.env_remove("CLAUDECODE");
    cmd.env_remove("CLAUDE_CODE");
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(err)) => Err(anyhow::anyhow!("{}", err)),
        Err(_) => Err(anyhow::anyhow!("command timed out after {:?}", timeout)),
    }
}

pub(crate) fn join_process_output(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if stdout.is_empty() {
        stderr
    } else if stderr.is_empty() {
        stdout
    } else {
        format!("{}\n{}", stdout, stderr)
    }
}

pub(crate) fn render_process_output(output: &std::process::Output, max_chars: usize) -> String {
    truncate_with_note(&join_process_output(output), max_chars)
}

pub(crate) fn normalize_review_request_id(value: Option<&str>) -> Option<String> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }
    let cleaned = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .collect::<String>();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.chars().take(64).collect())
    }
}

pub(crate) fn next_review_request_id() -> String {
    let short = uuid::Uuid::new_v4()
        .to_string()
        .chars()
        .filter(|c| *c != '-')
        .take(12)
        .collect::<String>();
    format!("rvw-{}", short)
}

pub(crate) fn collect_replay_tail(active: &ActiveSession, max_chars: usize) -> String {
    if max_chars == 0 || active.replay.is_empty() {
        return String::new();
    }
    let mut chunks = Vec::new();
    let mut used = 0usize;
    for frame in active.replay.iter().rev() {
        if used >= max_chars {
            break;
        }
        let remaining = max_chars.saturating_sub(used);
        if remaining == 0 {
            break;
        }
        let chunk = truncate_with_note(&frame.data, remaining);
        used = used.saturating_add(chunk.chars().count());
        chunks.push(chunk);
    }
    chunks.reverse();
    truncate_with_note(&chunks.join(""), max_chars)
}

pub(crate) fn send_review_progress_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    stage: &str,
    message: &str,
) {
    let _ = tx.send(ShellEvent::ReviewProgress {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        stage: stage.to_string(),
        message: message.to_string(),
    });
}

pub(crate) fn send_review_error_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    code: &str,
    message: &str,
) {
    let _ = tx.send(ShellEvent::ReviewError {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        code: code.to_string(),
        message: message.to_string(),
    });
}

pub(crate) fn send_review_result_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    payload: Value,
) {
    let _ = tx.send(ShellEvent::ReviewResult {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        payload,
    });
}

pub(crate) fn send_review_stream_event(
    tx: &tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: &str,
    request_id: &str,
    stream: &str,
    data: &str,
) {
    if data.is_empty() {
        return;
    }
    let _ = tx.send(ShellEvent::ReviewStream {
        session_id: session_id.to_string(),
        request_id: request_id.to_string(),
        stream: stream.to_string(),
        data: data.to_string(),
    });
}

pub(crate) async fn read_reviewer_stream<R>(
    mut reader: R,
    tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: String,
    request_id: String,
    stream: &'static str,
) -> anyhow::Result<String>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let mut buf = vec![0u8; REVIEW_STREAM_CHUNK_MAX_BYTES];
    let mut combined = String::new();
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
        if chunk.is_empty() {
            continue;
        }
        combined.push_str(&chunk);
        for part in split_utf8_chunks(&chunk, REVIEW_STREAM_CHUNK_MAX_BYTES) {
            send_review_stream_event(&tx, &session_id, &request_id, stream, &part);
        }
    }
    Ok(combined)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_streaming_capture_command(
    tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: String,
    request_id: String,
    command: &str,
    args: &[String],
    cwd: &Path,
    timeout: Duration,
    clear_env: bool,
) -> anyhow::Result<(std::process::ExitStatus, String)> {
    let mut cmd = Command::new(command);
    for arg in args {
        cmd.arg(arg);
    }
    cmd.current_dir(cwd);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.kill_on_drop(true);
    if clear_env {
        cmd.env_clear();
        for (k, v) in build_child_env() {
            cmd.env(k, v);
        }
    }
    cmd.env_remove("CLAUDECODE");
    cmd.env_remove("CLAUDE_CODE");

    let mut child = cmd.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture reviewer stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture reviewer stderr"))?;

    let out_task = tokio::spawn(read_reviewer_stream(
        stdout,
        tx.clone(),
        session_id.clone(),
        request_id.clone(),
        "stdout",
    ));
    let err_task = tokio::spawn(read_reviewer_stream(
        stderr, tx, session_id, request_id, "stderr",
    ));

    let status = match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(err)) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = out_task.await;
            let _ = err_task.await;
            return Err(anyhow::anyhow!("{}", err));
        }
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = out_task.await;
            let _ = err_task.await;
            return Err(anyhow::anyhow!("command timed out after {:?}", timeout));
        }
    };

    let stdout_combined = out_task
        .await
        .map_err(|err| anyhow::anyhow!("stdout task failed: {}", err))??;
    let stderr_combined = err_task
        .await
        .map_err(|err| anyhow::anyhow!("stderr task failed: {}", err))??;
    let combined = if stdout_combined.is_empty() {
        stderr_combined
    } else if stderr_combined.is_empty() {
        stdout_combined
    } else {
        format!("{}\n{}", stdout_combined, stderr_combined)
    };
    Ok((status, combined))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_review_job(
    tx: tokio::sync::mpsc::UnboundedSender<ShellEvent>,
    session_id: String,
    request_id: String,
    reviewer: String,
    profile: ReviewProfile,
    cwd: PathBuf,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    notes: Option<String>,
    replay_tail: String,
) {
    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "collect_context",
        "Collecting review context...",
    );

    let (context, context_stats) =
        collect_review_context(&cwd, scope, diff_base, &replay_tail).await;
    let prompt = build_review_prompt(&reviewer, scope, diff_base, &context, notes.as_deref());

    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "context_ready",
        &format!(
            "Context ready (changed files: {}, included: {}, skipped generated/binary/large/overflow: {}/{}/{}/{}).",
            context_stats.total_changed_files,
            context_stats.included_files,
            context_stats.skipped_generated_files,
            context_stats.skipped_binary_files,
            context_stats.skipped_large_files,
            context_stats.skipped_overflow_files
        ),
    );

    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "run_reviewer",
        &format!("Running {} reviewer...", reviewer),
    );

    let mut args = review_launch_args(&reviewer, &profile.args);
    args.push(prompt);
    let timeout = Duration::from_secs(profile.timeout_secs.clamp(30, 30 * 60));
    let (status, raw_combined) = match run_streaming_capture_command(
        tx.clone(),
        session_id.clone(),
        request_id.clone(),
        &profile.command,
        &args,
        &cwd,
        timeout,
        true,
    )
    .await
    {
        Ok(out) => out,
        Err(err) => {
            send_review_error_event(
                &tx,
                &session_id,
                &request_id,
                "reviewer_exec_failed",
                &format!("Failed to run {} review: {}", reviewer, err),
            );
            return;
        }
    };

    let rendered = truncate_with_note(&raw_combined, profile.max_output_chars);
    if !status.success() {
        let code = status
            .code()
            .map(|v| v.to_string())
            .unwrap_or_else(|| "terminated".to_string());
        send_review_error_event(
            &tx,
            &session_id,
            &request_id,
            "reviewer_exit_nonzero",
            &format!(
                "{} reviewer exited with status {}.\n{}",
                reviewer, code, rendered
            ),
        );
        return;
    }

    if raw_combined.trim().is_empty() {
        send_review_error_event(
            &tx,
            &session_id,
            &request_id,
            "reviewer_empty_output",
            &format!("{} reviewer returned no output.", reviewer),
        );
        return;
    }

    send_review_progress_event(
        &tx,
        &session_id,
        &request_id,
        "parse_result",
        "Parsing review result...",
    );
    match parse_review_output_payload(&reviewer, &request_id, scope, diff_base, &raw_combined) {
        Ok(mut payload) => {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert(
                    "meta".to_string(),
                    json!({
                        "raw_output_chars": raw_combined.chars().count(),
                        "context_chars": context.chars().count(),
                        "context": {
                            "total_changed_files": context_stats.total_changed_files,
                            "included_files": context_stats.included_files,
                            "skipped_generated_files": context_stats.skipped_generated_files,
                            "skipped_binary_files": context_stats.skipped_binary_files,
                            "skipped_large_files": context_stats.skipped_large_files,
                            "skipped_overflow_files": context_stats.skipped_overflow_files,
                        }
                    }),
                );
            }
            send_review_result_event(&tx, &session_id, &request_id, payload);
        }
        Err(message) => {
            send_review_error_event(
                &tx,
                &session_id,
                &request_id,
                "review_parse_failed",
                &format!(
                    "{} reviewer returned unparseable output.\n{}\nContext stats: changed={}, included={}, skipped generated/binary/large/overflow={}/{}/{}/{}",
                    reviewer,
                    message,
                    context_stats.total_changed_files,
                    context_stats.included_files,
                    context_stats.skipped_generated_files,
                    context_stats.skipped_binary_files,
                    context_stats.skipped_large_files,
                    context_stats.skipped_overflow_files
                ),
            );
        }
    }
}
