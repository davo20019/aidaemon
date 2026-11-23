//! The follow-up turn a finished background command starts.

use super::*;

/// Friendly, pid-free delivery message for a short background result. Inline
/// code for a one-liner (e.g. a count); a fenced block for a few lines.
pub(super) fn format_short_background_result(output_trimmed: &str) -> String {
    if output_trimmed.contains('\n') {
        format!("Result:\n```\n{}\n```", output_trimmed)
    } else {
        format!("Result: `{}`", output_trimmed)
    }
}

pub(super) fn format_background_continuation_failure(unchecked: &[String]) -> String {
    if unchecked.is_empty() {
        return "⚠️ The background step finished, but the automatic continuation could not prepare the final response."
            .to_string();
    }

    let mut message = String::from(
        "⚠️ The background step finished, but I couldn't complete the remaining request. Still needed:\n",
    );
    for item in unchecked.iter().take(5) {
        message.push_str("- ");
        message.push_str(item);
        message.push('\n');
    }
    message.trim_end().to_string()
}

/// Serializes background-notifier re-engagements of the agent loop. A
/// completion arriving while another re-engagement turn is still running must
/// wait, not spawn a CONCURRENT loop on the same daemon: two racing loops each
/// launched their own duplicate `find` sweeps and posted their own "Done"
/// pings (live 2026-07-12). The guard is held across the whole re-engaged
/// `handle_message` call; later completions then see the earlier turn's
/// results in history instead of redoing the work.
pub(super) static REENGAGE_SERIALIZER: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// A command as INFO and WARN logs show it: its program and a fingerprint,
/// never the full text. The agent reads the daemon's own log when it
/// troubleshoots, and a full command line there (`grep trigger_pattern …`)
/// is later found as if it were what the command searched for. The full
/// command stays in the tool call's record and at DEBUG ("Approval check").
pub(crate) struct LoggedCommand<'a>(pub(crate) &'a str);

/// The program a shell command line runs, for a log label: past leading
/// `NAME=value` settings and redirects, without a trailing `;`, `&` or `|`,
/// its last path part. "shell" when that is not a plain program name.
pub(super) fn logged_program(command: &str) -> &str {
    // Machine format: shell command line words.
    let mut words = command.split_whitespace();
    let mut word = None;
    while let Some(next) = words.next() {
        let is_setting = next.split_once('=').is_some_and(|(name, _)| {
            !name.is_empty()
                && name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric())
                && !name.starts_with(|c: char| c.is_ascii_digit())
        });
        let redirect = next.trim_start_matches(|c: char| c.is_ascii_digit());
        if is_setting {
            continue;
        }
        if redirect.starts_with('<') || redirect.starts_with('>') {
            // A bare operator's target is the next word.
            if redirect.trim_start_matches(['<', '>', '&', '|']).is_empty() {
                words.next();
            }
            continue;
        }
        word = Some(next);
        break;
    }
    let word = word.unwrap_or("").trim_end_matches([';', '&', '|']);
    let program = word.rsplit('/').next().unwrap_or(word);
    let plain = !program.is_empty()
        && !program.starts_with('-')
        && program
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+'));
    if plain { program } else { "shell" }
}

impl std::fmt::Display for LoggedCommand<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use sha2::{Digest, Sha256};
        let program = truncate_str(logged_program(self.0), 40);
        let digest = Sha256::digest(self.0.as_bytes());
        write!(f, "{program} [{} chars, ", self.0.chars().count())?;
        for byte in &digest[..4] {
            write!(f, "{byte:02x}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
pub async fn acquire_reengagement_slot() -> tokio::sync::MutexGuard<'static, ()> {
    REENGAGE_SERIALIZER.lock().await
}

/// Run one background completion continuation under the shared serializer and
/// a single wall-clock budget that includes waiting for the serializer. This is
/// the authoritative boundary for terminal and delegated CLI completions.
///
/// The [`FollowupEnd`] tells watchers the turn ended when it is dropped:
/// keep it until the answer is delivered.
pub async fn run_background_continuation(
    agent: &dyn ConversationRuntime,
    request: ConversationRequest,
) -> (
    anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope>,
    FollowupEnd,
) {
    run_background_continuation_with_timeout(agent, request, BACKGROUND_CONTINUATION_TIMEOUT).await
}

/// Numbers for follow-up turns as the turn observer sees them, far above
/// the task registry's own (which count up from 1), so the two never meet.
pub(super) static NEXT_FOLLOWUP_TURN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1 << 48);

/// How a follow-up the owner is shown is named in the running-work list.
pub(super) const FOLLOWUP_DESCRIPTION: &str = "Following up on background work";

pub(super) async fn run_background_continuation_with_timeout(
    agent: &dyn ConversationRuntime,
    mut request: ConversationRequest,
    timeout: Duration,
) -> (
    anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope>,
    FollowupEnd,
) {
    // The follow-up is a turn of the conversation like any other: whoever
    // watches turns (the app) sees it start, its progress and its end, not
    // only the message it ends with.
    let observer = agent.turn_watch().and_then(|watch| watch.observer());
    let watched = match observer {
        Some(observer) if request.status_tx.is_none() => {
            let turn = NEXT_FOLLOWUP_TURN.fetch_add(1, Ordering::Relaxed);
            observer.followup_started(&request.session_id, turn, FOLLOWUP_DESCRIPTION);
            let (status_tx, mut updates) = mpsc::channel::<StatusUpdate>(16);
            request.status_tx = Some(status_tx);
            let forward = {
                let observer = observer.clone();
                let session_id = request.session_id.clone();
                tokio::spawn(async move {
                    while let Some(update) = updates.recv().await {
                        observer.turn_status(&session_id, turn, &update);
                    }
                })
            };
            Some((observer, turn, forward))
        }
        _ => None,
    };
    let session_id = request.session_id.clone();
    let result = run_continuation_turn(agent, request, timeout).await;
    let Some((observer, turn, forward)) = watched else {
        return (result, FollowupEnd(None));
    };
    // The turn has ended and dropped its sender: let what it said reach the
    // watcher before its end does.
    let _ = tokio::time::timeout(Duration::from_secs(2), forward).await;
    let end = match &result {
        Ok(envelope) if envelope.text.trim().is_empty() => aidaemon_channels::turns::TurnEnd::Empty,
        Ok(_) => aidaemon_channels::turns::TurnEnd::Delivered,
        Err(_) => aidaemon_channels::turns::TurnEnd::Failed,
    };
    let ended = FollowupEnded {
        observer,
        session_id,
        turn,
        end,
        // A follow-up may start more background work (a fix's check): the
        // conversation stays working until that reports.
        handed_off: result.as_ref().is_ok_and(|envelope| {
            envelope.disposition == aidaemon_events::AssistantResponseDisposition::BackgroundHandoff
        }),
        error: result.as_ref().err().map(|error| error.to_string()),
    };
    (result, FollowupEnd(Some(ended)))
}

/// A watched follow-up's end, told when this is dropped: after the caller
/// delivered its answer, so the conversation reads "working in the
/// background" from a hand-off on, never idle in between.
pub struct FollowupEnd(pub(super) Option<FollowupEnded>);

pub(super) struct FollowupEnded {
    pub(super) observer: Arc<dyn aidaemon_channels::turns::TurnObserver>,
    pub(super) session_id: String,
    pub(super) turn: u64,
    pub(super) end: aidaemon_channels::turns::TurnEnd,
    pub(super) handed_off: bool,
    pub(super) error: Option<String>,
}

impl Drop for FollowupEnd {
    fn drop(&mut self) {
        if let Some(ended) = self.0.take() {
            ended
                .observer
                .turn_finished(&aidaemon_channels::turns::TurnFinish {
                    session_id: &ended.session_id,
                    turn: ended.turn,
                    platform: "background",
                    end: ended.end,
                    // The answer went out as the conversation's message, by
                    // the caller's delivery: the turn itself carries none.
                    reply: "",
                    handed_off: ended.handed_off,
                    error: ended.error.as_deref(),
                    model: None,
                });
        }
    }
}

pub(super) async fn run_continuation_turn(
    agent: &dyn ConversationRuntime,
    mut request: ConversationRequest,
    timeout: Duration,
) -> anyhow::Result<aidaemon_channels::ports::AgentResponseEnvelope> {
    let session_id = request.session_id.clone();
    let deadline = tokio::time::Instant::now() + timeout;
    let _slot = tokio::time::timeout_at(deadline, REENGAGE_SERIALIZER.lock())
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Background continuation for session {session_id} timed out waiting for its execution slot after {}s",
                timeout.as_secs()
            )
        })?;
    let clock = request.clock;
    // Bounded by inactivity, never by the clock: a continuation still
    // making progress (model calls, tools) runs as long as its work takes,
    // and its token budget bounds its cost; one that goes quiet for
    // `timeout` is stopped (live 2026-11-15: a coding
    // follow-up that fixed, rebuilt and re-ran its tests was cut off at
    // ten minutes and its raw test output posted instead).
    let unix_now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    };
    let heartbeat = request
        .heartbeat
        .get_or_insert_with(|| Arc::new(std::sync::atomic::AtomicU64::new(unix_now())))
        .clone();
    heartbeat.store(unix_now(), Ordering::Relaxed);
    let work = agent.continue_conversation(request);
    tokio::pin!(work);
    let mut envelope = loop {
        tokio::select! {
            result = &mut work => break result?,
            _ = tokio::time::sleep(Duration::from_secs(2).min(timeout)) => {
                let idle = unix_now().saturating_sub(heartbeat.load(Ordering::Relaxed));
                if idle >= timeout.as_secs().max(1) {
                    anyhow::bail!(
                        "Background continuation for session {session_id} timed out after {}s without progress",
                        timeout.as_secs()
                    );
                }
            }
        }
    };
    // The final answer to a request carries its total, measured from the
    // owner's message to now and rendered from the records, never the
    // model's estimate (live 2026-11-13: an answer said the total "wasn't
    // measured"), when it adds to the runs' own times.
    let total = clock
        .filter(|_| !envelope.text.trim().is_empty())
        .and_then(|clock| {
            crate::run_state::timing::answer_sent_line(clock, false, chrono::Utc::now())
        });
    if let Some(total) = total {
        envelope.text = format!(
            "{}\n\n{}{}",
            envelope.text.trim_end(),
            crate::turn::sliding_window::MEASURED_TIMING_MARK,
            total
        );
    }
    Ok(envelope)
}

/// Ends a foreground command's process group when the call that started it
/// is dropped while the command still runs.
pub(super) struct StopWithCall(pub(super) Option<u32>);

impl Drop for StopWithCall {
    fn drop(&mut self) {
        if let Some(pid) = self.0.take() {
            send_sigkill(pid);
        }
    }
}

/// Build the internal follow-up that re-engages the agent loop after a
/// background command completes. Beyond replaying the output, it explicitly
/// steers the model to finish any *deferred deliverable* the user requested
/// before the command was backgrounded. This matters because the original turn
/// ends the moment a long-running command detaches: a request like "send me the
/// file when done" would otherwise be silently dropped, since the model would
/// just summarize the output instead of completing the requested action.
pub(super) fn build_background_reengagement_followup(
    command_summary: &str,
    output: &str,
) -> String {
    // The typed lifecycle facts below are invariants of this code path: a
    // continuation only exists because the launching turn handed off
    // nonterminally, the invocation's receipt recorded it as backgrounded,
    // and its completion output was queued for delivery through this very
    // continuation. Stating them here makes the lifecycle readable instead
    // of something the model must reconstruct from prose — with identical
    // contexts the self-reported lifecycle used to flip between typed values
    // and "unknown" across runs.
    format!(
        "[Background command completed]\n\
         Command: `{command_summary}`\n\
         Output:\n{output}\n\n\
         Typed invocation lifecycle (authoritative, from the durable receipt): \
         the launching turn ended in a nonterminal background handoff; the \
         invocation's terminal result is recorded as backgrounded; its \
         completion output was queued for delivery through this continuation.\n\n\
         Continue the exact parent request identified by the attached runtime \
         continuation edge. Treat the linked terminal receipt as authoritative \
         evidence for that invocation and complete only obligations retained by \
         that parent request: when it asked you to fix what this command \
         found, fix it now. Do NOT re-run this command just to see its result \
         again: its completion receipt is already recorded and credited, and a \
         redundant check counts as an extra tool call against the user's \
         request. Running it again after you changed files is a new check, not \
         a redundant one. When reporting how many calls the request used, count \
         only the original invocation and checks after changes."
    )
}
