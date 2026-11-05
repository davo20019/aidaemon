//! The loop. Everything that is policy lives in hooks; this file only
//! sequences model calls and tool calls, enforces hard limits, records
//! receipts and events, and maps every exit to a [`RunStatus`].

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::time::{Duration, Instant};

use futures::future::join_all;
use futures::{FutureExt, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::agent::{AgentInner, Resolution};
use crate::approval::{EffectKey, InterruptKind, OWNER_DECLINED};
use crate::error::CoreError;
use crate::event::Event;
use crate::hook::{HookContext, StopDecision, StuckDecision, ToolDecision};
use crate::limits::{LimitKind, RunSnapshot};
use crate::message::{Message, MessageOrigin, ToolCall};
use crate::outcome::{RunOutcome, RunStatus};
use crate::provider::{ModelRequest, ModelResponse, StreamEvent, ToolChoice};
use crate::receipt::{InvocationStage, Receipt, ReceiptStatus};
use crate::state::{InFlight, Interruption, Pending, RunId, RunState};
use crate::stuck::StuckSignal;
use crate::tool::{SourceTrust, ToolAnnotations, ToolContext, ToolError, ToolInfo, ToolOutput};

pub(crate) struct Driver<'a> {
    agent: &'a AgentInner,
    state: RunState,
    cancel: CancellationToken,
    segment_start: Instant,
    base_elapsed_ms: u64,
    /// Whether this segment's end is the run's end for hooks. A segment
    /// driven by `run_resolving` is not: the caller reports the final one.
    report_run_end: bool,
}

const STUCK_SKIP_REASON: &str =
    "the run was stopped by the stuck detector before this call started";

/// How a tool batch ended.
enum BatchEnd {
    Done,
    Interrupted(Interruption),
    Stuck(StuckSignal),
    Limit(LimitKind),
    Cancelled,
}

/// What the stuck policy asked for while recording a result.
enum StuckAction {
    Nudge(String),
    Stop(StuckSignal),
}

/// How one model call ended.
enum ModelCall {
    Response(ModelResponse),
    Cancelled,
    Failed(String),
}

/// The settled policy decision for one call (rewrites already applied).
enum Verdict {
    Allow,
    Deny(String),
    /// The reason, its kind, and the index of the hook that interrupted.
    Interrupt(String, InterruptKind, usize),
}

/// A call that passed checks and policy, waiting to execute.
struct Approved {
    call: ToolCall,
    annotations: ToolAnnotations,
}

impl<'a> Driver<'a> {
    pub(crate) fn new(agent: &'a AgentInner, state: RunState, cancel: CancellationToken) -> Self {
        let base_elapsed_ms = state.elapsed_ms;
        Self {
            agent,
            state,
            cancel,
            segment_start: Instant::now(),
            base_elapsed_ms,
            report_run_end: true,
        }
    }

    /// Apply what a resolution means for the run before the pending call is
    /// settled: an owner's denial is remembered for the rest of the run
    /// (its effects join `declined`) and its reason is marked as the
    /// owner's.
    fn settle(
        &mut self,
        interruption: &Interruption,
        resolution: Option<Resolution>,
    ) -> Option<Resolution> {
        let (InterruptKind::OwnerApproval(request), Some(Resolution::Deny { reason })) =
            (&interruption.kind, &resolution)
        else {
            return resolution;
        };
        for effect in &request.effects {
            if !self.state.declined.contains(effect) {
                self.state.declined.push(effect.clone());
            }
        }
        let reason = if reason.starts_with(OWNER_DECLINED) {
            reason.clone()
        } else {
            format!("{OWNER_DECLINED} {reason}")
        };
        Some(Resolution::Deny { reason })
    }

    /// Leave `on_run_end` to the caller (see `report_run_end`).
    pub(crate) fn without_run_end(mut self) -> Self {
        self.report_run_end = false;
        self
    }

    pub(crate) async fn drive(
        mut self,
        resolution: Option<Resolution>,
    ) -> Result<RunOutcome, CoreError> {
        if let Some(pending) = self.state.pending.take() {
            let resolution = self.settle(&pending.interruption, resolution);
            self.emit(Event::RunResumed).await?;
            let earlier: Vec<String> = pending.calls[..pending.next]
                .iter()
                .map(|c| c.id.clone())
                .collect();
            self.state.in_flight = Some(InFlight {
                calls: pending.calls.clone(),
                started: earlier.clone(),
                done: earlier,
            });
            match self
                .execute_batch(&pending.calls, pending.next, resolution, pending.hook_index)
                .await?
            {
                BatchEnd::Done => self.checkpoint().await?,
                end => return self.finish_batch(end).await,
            }
        } else if let Some(in_flight) = self.state.in_flight.clone() {
            // A checkpoint taken mid-batch: the process stopped while tools
            // were running. Finish the batch without repeating side effects.
            match self.recover(in_flight).await? {
                BatchEnd::Done => self.checkpoint().await?,
                end => return self.finish_batch(end).await,
            }
        } else {
            let session_id = self.state.session_id.clone();
            self.emit(Event::RunStarted { session_id }).await?;
        }

        loop {
            // 1. Guards: cancellation, hard limits, custom stop conditions.
            if self.cancel.is_cancelled() {
                return self.finish(RunStatus::Cancelled, None).await;
            }
            let snapshot = self.snapshot();
            if let Some(limit) = self.agent.limits.exceeded(&snapshot) {
                return self.finish(RunStatus::LimitReached { limit }, None).await;
            }
            if let Some(reason) = self
                .agent
                .stop_conditions
                .iter()
                .find_map(|c| c.check(&snapshot))
            {
                return self.finish(RunStatus::Stopped { reason }, None).await;
            }

            // 2. Build the request and let hooks shape it.
            self.state.turn += 1;
            let turn = self.state.turn;
            self.emit(Event::TurnStarted { turn }).await?;
            let mut request = self.build_request();
            {
                let ctx = self.ctx();
                for hook in &self.agent.hooks {
                    hook.before_model(&ctx, &mut request).await;
                }
            }

            // 3. Call the model (streamed; non-streaming providers yield a
            //    single final event), bounded by the remaining wall-clock budget.
            let response = match within(self.remaining(), self.call_model(&request, turn)).await {
                None => {
                    let limit = LimitKind::WallClock;
                    return self.finish(RunStatus::LimitReached { limit }, None).await;
                }
                Some(call) => call?,
            };
            let response = match response {
                ModelCall::Response(response) => response,
                ModelCall::Cancelled => return self.finish(RunStatus::Cancelled, None).await,
                ModelCall::Failed(error) => {
                    return self.finish(RunStatus::ProviderFailed { error }, None).await;
                }
            };
            self.state.usage.add(response.usage);
            self.state.run_usage.add(response.usage);
            let calls: Vec<ToolCall> = response
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(i, mut call)| {
                    if call.id.trim().is_empty() {
                        call.id = format!("call_{turn}_{i}");
                    }
                    call
                })
                .collect();
            self.emit(Event::ModelResponded {
                turn,
                usage: response.usage,
                text: response.text.clone(),
                tool_calls: calls.len(),
            })
            .await?;
            self.state
                .messages
                .push(Message::assistant(response.text.clone(), calls.clone()));

            // 4a. No tool calls: the model thinks it is done. Ask stop hooks.
            if calls.is_empty() {
                let rejection = {
                    let ctx = self.ctx();
                    let mut rejection = None;
                    for hook in &self.agent.hooks {
                        if let StopDecision::Continue { message } =
                            hook.on_stop(&ctx, &response.text).await
                        {
                            rejection = Some(message);
                            break;
                        }
                    }
                    rejection
                };
                match rejection {
                    Some(message) => {
                        self.emit(Event::StopRejected {
                            message: message.clone(),
                        })
                        .await?;
                        self.state
                            .messages
                            .push(Message::from_origin(MessageOrigin::StopHook, message));
                        self.checkpoint().await?;
                        continue;
                    }
                    None => {
                        return self.finish(RunStatus::Completed, Some(response.text)).await;
                    }
                }
            }

            // 4b. Run the requested tools, then go around again. The batch is
            //     checkpointed as in flight so a crash can be recovered.
            self.state.in_flight = Some(InFlight {
                calls: calls.clone(),
                ..InFlight::default()
            });
            self.checkpoint().await?;
            match self.execute_batch(&calls, 0, None, None).await? {
                BatchEnd::Done => self.checkpoint().await?,
                end => return self.finish_batch(end).await,
            }
        }
    }

    /// Run `calls[start..]`. Consecutive approved read-only calls run
    /// concurrently; anything else runs alone, in order. Results are always
    /// appended in call order.
    async fn execute_batch(
        &mut self,
        calls: &[ToolCall],
        start: usize,
        mut resolution: Option<Resolution>,
        resume_after: Option<usize>,
    ) -> Result<BatchEnd, CoreError> {
        let mut group: Vec<Approved> = Vec::new();
        let mut nudge: Option<String> = None;
        let mut stop: Option<StuckSignal> = None;

        for index in start..calls.len() {
            if self.cancel.is_cancelled() {
                self.flush(&mut group, &mut nudge, &mut stop).await?;
                self.skip_rest(
                    &calls[index..],
                    ReceiptStatus::Cancelled,
                    "the run was cancelled before this call started",
                )
                .await?;
                return Ok(BatchEnd::Cancelled);
            }
            // The stuck policy already decided to stop: nothing else in this
            // batch may run, mutating calls least of all.
            if let Some(signal) = stop.take() {
                self.skip_rest(&calls[index..], ReceiptStatus::Skipped, STUCK_SKIP_REASON)
                    .await?;
                return Ok(BatchEnd::Stuck(signal));
            }
            // Budgets hold inside a batch too, not only between turns.
            if let Some(limit) = self.call_limit(group.len()) {
                self.flush(&mut group, &mut nudge, &mut stop).await?;
                let reason = format!("the run reached its {} limit", limit_name(limit));
                self.skip_rest(&calls[index..], ReceiptStatus::Skipped, &reason)
                    .await?;
                return Ok(BatchEnd::Limit(limit));
            }

            let mut call = calls[index].clone();
            let turn = self.state.turn;
            if self.is_done(&call.id) {
                continue;
            }

            // Deterministic checks come before any policy, so a malformed
            // proposal never reaches a hook or a human.
            let Some(tool) = self.agent.tools.get(&call.name).cloned() else {
                if let Some(end) = self
                    .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                    .await?
                {
                    return Ok(end);
                }
                let available: Vec<&str> =
                    self.agent.specs.iter().map(|s| s.name.as_str()).collect();
                let reason = format!(
                    "unknown tool `{}`. Available tools: {}",
                    call.name,
                    available.join(", ")
                );
                let (receipt, output) = rejection(
                    self.state.run_id,
                    &call,
                    turn,
                    ReceiptStatus::UnknownTool,
                    reason,
                );
                track(self.record(receipt, output).await?, &mut nudge, &mut stop);
                continue;
            };
            if let Err(reason) = tool.validate(&call.arguments) {
                if let Some(end) = self
                    .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                    .await?
                {
                    return Ok(end);
                }
                let reason = format!("invalid arguments for `{}`: {reason}", call.name);
                let (receipt, output) = rejection(
                    self.state.run_id,
                    &call,
                    turn,
                    ReceiptStatus::InvalidArguments,
                    reason,
                );
                track(self.record(receipt, output).await?, &mut nudge, &mut stop);
                continue;
            }
            let annotations = tool.annotations_for(&call.arguments);
            // Authority and receipts use the per-call value; concurrent
            // scheduling uses only the tool's worst case, since a per-call
            // relaxation is a statement about authority, not isolation.
            let may_run_concurrently = tool.annotations().read_only;
            let mut info = ToolInfo {
                annotations,
                access: tool.access(&call.arguments),
                effects: tool.effects(&call.arguments),
            };

            // Hooks must see the results of earlier calls in this batch (a
            // fetch that taints the run before a send is judged), so queued
            // read-only calls run before a call that is not concurrent-safe
            // is put to them.
            if !may_run_concurrently || annotations.egress {
                if let Some(end) = self
                    .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                    .await?
                {
                    return Ok(end);
                }
            }

            // Policy: a pending resolution settles the first call; otherwise
            // ask the hooks.
            let settled = if index == start {
                resolution.take()
            } else {
                None
            };
            let verdict = match settled {
                // An approval settles only the hook that interrupted: the
                // hooks after it still judge the call.
                Some(Resolution::Approve) => {
                    self.decide(&mut call, &mut info, resume_after.map_or(0, |i| i + 1))
                        .await
                }
                Some(Resolution::Deny { reason }) => Verdict::Deny(reason),
                None => self.decide(&mut call, &mut info, 0).await,
            };
            // A hook may have rewritten the call; judge and record what runs.
            let annotations = info.annotations;

            match verdict {
                Verdict::Allow if may_run_concurrently => {
                    group.push(Approved { call, annotations });
                }
                Verdict::Allow => {
                    if let Some(end) = self
                        .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                        .await?
                    {
                        return Ok(end);
                    }
                    let mut single = vec![Approved { call, annotations }];
                    self.flush(&mut single, &mut nudge, &mut stop).await?;
                }
                Verdict::Deny(reason) => {
                    if let Some(end) = self
                        .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                        .await?
                    {
                        return Ok(end);
                    }
                    let (mut receipt, output) = rejection(
                        self.state.run_id,
                        &call,
                        turn,
                        ReceiptStatus::Denied,
                        reason,
                    );
                    receipt.read_only = annotations.read_only;
                    receipt.after_untrusted = self.state.untrusted_seen;
                    track(self.record(receipt, output).await?, &mut nudge, &mut stop);
                }
                Verdict::Interrupt(reason, kind, hook_index) => {
                    if let Some(end) = self
                        .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                        .await?
                    {
                        return Ok(end);
                    }
                    // Save the call exactly as the hooks left it: a resumed
                    // `Approve` runs what the approver was shown, including
                    // any rewrite, without consulting the hooks again.
                    let mut pending_calls = calls.to_vec();
                    pending_calls[index] = call.clone();
                    let interruption = Interruption { call, reason, kind };
                    self.state.in_flight = None;
                    self.state.pending = Some(Pending {
                        calls: pending_calls,
                        next: index,
                        interruption: interruption.clone(),
                        hook_index: Some(hook_index),
                    });
                    self.emit(Event::Interrupted {
                        interruption: interruption.clone(),
                    })
                    .await?;
                    return Ok(BatchEnd::Interrupted(interruption));
                }
            }
        }
        self.flush(&mut group, &mut nudge, &mut stop).await?;
        self.state.in_flight = None;

        if let Some(signal) = stop {
            return Ok(BatchEnd::Stuck(signal));
        }
        if let Some(message) = nudge {
            // After all tool results, so the transcript stays well formed.
            self.state
                .messages
                .push(Message::from_origin(MessageOrigin::Harness, message));
        }
        Ok(BatchEnd::Done)
    }

    /// Flush the queued read-only calls before `rest[0]` is handled. If
    /// their results made the stuck policy stop the run, `rest` is skipped
    /// and the batch ends.
    async fn flush_or_stop(
        &mut self,
        group: &mut Vec<Approved>,
        nudge: &mut Option<String>,
        stop: &mut Option<StuckSignal>,
        rest: &[ToolCall],
    ) -> Result<Option<BatchEnd>, CoreError> {
        self.flush(group, nudge, stop).await?;
        let Some(signal) = stop.take() else {
            return Ok(None);
        };
        self.skip_rest(rest, ReceiptStatus::Skipped, STUCK_SKIP_REASON)
            .await?;
        Ok(Some(BatchEnd::Stuck(signal)))
    }

    /// Ask `before_tool` hooks in order. Rewrites are applied (and the new
    /// arguments re-validated) before the next hook sees the call.
    ///
    /// A call with an effect the owner declined earlier in the run is
    /// refused before any hook sees it, and again after any rewrite: no
    /// hook or application can opt out, so a model cannot retry a declined
    /// action through another tool.
    async fn decide(&self, call: &mut ToolCall, info: &mut ToolInfo, from: usize) -> Verdict {
        if let Some(reason) = declined_reason(&self.state.declined, info) {
            return Verdict::Deny(reason);
        }
        let ctx = self.ctx();
        for (hook_index, hook) in self.agent.hooks.iter().enumerate().skip(from) {
            match hook.before_tool(&ctx, call, info).await {
                ToolDecision::Allow => {}
                ToolDecision::Deny { reason } => return Verdict::Deny(reason),
                ToolDecision::Interrupt { reason, kind } => {
                    return Verdict::Interrupt(reason, kind, hook_index)
                }
                ToolDecision::Rewrite { arguments } => {
                    call.arguments = arguments;
                    if let Some(tool) = self.agent.tools.get(&call.name) {
                        if let Err(reason) = tool.validate(&call.arguments) {
                            return Verdict::Deny(format!(
                                "a hook rewrote the arguments into an invalid call: {reason}"
                            ));
                        }
                        // The rewritten call is a different call: its
                        // effect and access are those of the new arguments.
                        info.annotations = tool.annotations_for(&call.arguments);
                        info.access = tool.access(&call.arguments);
                        info.effects = tool.effects(&call.arguments);
                    }
                    if let Some(reason) = declined_reason(&self.state.declined, info) {
                        return Verdict::Deny(reason);
                    }
                }
            }
        }
        Verdict::Allow
    }

    /// Execute approved calls concurrently and record results in order.
    async fn flush(
        &mut self,
        group: &mut Vec<Approved>,
        nudge: &mut Option<String>,
        stop: &mut Option<StuckSignal>,
    ) -> Result<(), CoreError> {
        if group.is_empty() {
            return Ok(());
        }
        let items = std::mem::take(group);
        if let Some(in_flight) = self.state.in_flight.as_mut() {
            in_flight
                .started
                .extend(items.iter().map(|item| item.call.id.clone()));
        }
        self.checkpoint().await?;
        for item in &items {
            self.emit(Event::ToolStarted {
                call_id: item.call.id.clone(),
                tool: item.call.name.clone(),
            })
            .await?;
        }
        let this = &*self;
        let results = join_all(items.iter().map(|item| this.dispatch(item))).await;
        for (receipt, output) in results {
            let action = self.record(receipt, output).await?;
            track(action, nudge, stop);
        }
        self.checkpoint().await?;
        Ok(())
    }

    /// Run one tool with a timeout and cancellation, producing its receipt.
    async fn dispatch(&self, item: &Approved) -> (Receipt, ToolOutput) {
        let call = &item.call;
        let tool = self
            .agent
            .tools
            .get(&call.name)
            .expect("dispatch only sees calls that passed lookup")
            .clone();
        let ctx = ToolContext::new(self.state.session_id.clone(), call.id.clone())
            .with_run(self.state.run_id, self.state.turn)
            .with_cancel(self.cancel.child_token());
        // A tool's own timeout hint replaces the default, but nothing may
        // outlive the run's remaining wall-clock budget.
        let per_tool = tool.timeout().unwrap_or(self.agent.limits.tool_timeout);
        let timeout = match self.remaining() {
            Some(remaining) => remaining.min(per_tool),
            None => per_tool,
        };
        let started = Instant::now();
        let result = tokio::time::timeout(timeout, async {
            tokio::select! {
                biased;
                _ = ctx.cancel.cancelled() => None,
                // A panicking tool is a failed call, not a crashed run.
                result = AssertUnwindSafe(tool.call(call.arguments.clone(), &ctx)).catch_unwind() => {
                    Some(result.map_err(|panic| panic_message(&*panic)))
                }
            }
        })
        .await;
        let duration_ms = started.elapsed().as_millis() as u64;

        // Text the kernel wrote itself (timeout, cancel, panic) is trusted;
        // anything the tool returned, an error included, carries the tool's
        // trust, because an error page can hold an injection too.
        let tool_trust = |output: &ToolOutput| {
            output
                .trust
                .unwrap_or_else(|| tool.source_trust(&call.arguments))
        };
        let (status, reason, output, source) = match result {
            Ok(Some(Ok(Ok(output)))) if output.is_error => {
                let reason = output.content.chars().take(300).collect::<String>();
                let source = tool_trust(&output);
                (ReceiptStatus::Failed, Some(reason), output, source)
            }
            Ok(Some(Ok(Ok(output)))) => {
                let source = tool_trust(&output);
                (ReceiptStatus::Succeeded, None, output, source)
            }
            Ok(Some(Ok(Err(err)))) => (
                ReceiptStatus::Failed,
                Some(err.0.clone()),
                ToolOutput::error(format!("Error: {err}")),
                tool.source_trust(&call.arguments),
            ),
            Ok(Some(Err(panic))) => {
                let err = ToolError(format!("the tool panicked: {panic}"));
                (
                    ReceiptStatus::Failed,
                    Some(err.0.clone()),
                    ToolOutput::error(format!("Error: {err}")),
                    SourceTrust::Trusted,
                )
            }
            Ok(None) => (
                ReceiptStatus::Cancelled,
                Some("cancelled".to_string()),
                ToolOutput::error("Cancelled: the run was cancelled while this tool was running."),
                SourceTrust::Trusted,
            ),
            Err(_) => {
                let secs = timeout.as_secs_f64();
                (
                    ReceiptStatus::TimedOut,
                    Some(format!("timed out after {secs:.1}s")),
                    ToolOutput::error(format!(
                        "Error: `{}` timed out after {secs:.1}s.",
                        call.name
                    )),
                    SourceTrust::Trusted,
                )
            }
        };

        let receipt = Receipt {
            run_id: self.state.run_id,
            call_id: call.id.clone(),
            tool: call.name.clone(),
            arguments: call.arguments.clone(),
            turn: self.state.turn,
            stage: InvocationStage::Dispatched,
            status,
            read_only: item.annotations.read_only,
            duration_ms,
            reason,
            details: output.details.clone(),
            source,
            after_untrusted: self.state.untrusted_seen,
        };
        (receipt, output)
    }

    /// Record calls that will never start, without feeding the stuck
    /// detector (they say nothing about what the model is doing).
    async fn skip_rest(
        &mut self,
        calls: &[ToolCall],
        status: ReceiptStatus,
        reason: &str,
    ) -> Result<(), CoreError> {
        for call in calls {
            let (receipt, mut output) = rejection(
                self.state.run_id,
                call,
                self.state.turn,
                status,
                reason.to_string(),
            );
            self.append(receipt, &mut output).await?;
        }
        Ok(())
    }

    /// Record a result and feed the stuck detector.
    async fn record(
        &mut self,
        receipt: Receipt,
        mut output: ToolOutput,
    ) -> Result<Option<StuckAction>, CoreError> {
        let receipt = self.append(receipt, &mut output).await?;
        self.observe_stuck(&receipt, &output.content).await
    }

    /// Shape the output, run `after_tool` hooks, append the tool message and
    /// receipt, and emit the event.
    async fn append(
        &mut self,
        receipt: Receipt,
        output: &mut ToolOutput,
    ) -> Result<Receipt, CoreError> {
        output.content = truncate_middle(&output.content, self.agent.limits.max_tool_output_chars);
        {
            let ctx = self.ctx();
            for hook in &self.agent.hooks {
                hook.after_tool(&ctx, &receipt, output).await;
            }
        }
        self.state.messages.push(
            Message::tool_result(receipt.call_id.clone(), output.content.clone())
                .with_images(output.images.iter().cloned())
                .with_error(!receipt.succeeded()),
        );
        if receipt.stage == InvocationStage::Dispatched && receipt.source == SourceTrust::Untrusted
        {
            self.state.untrusted_seen = true;
        }
        self.state.receipts.push(receipt.clone());
        self.state.tool_calls += 1;
        self.state.usage.add(output.usage);
        self.state.run_usage.add(output.usage);
        if let Some(in_flight) = self.state.in_flight.as_mut() {
            in_flight.done.push(receipt.call_id.clone());
        }
        self.emit(Event::ToolFinished {
            receipt: receipt.clone(),
        })
        .await?;
        Ok(receipt)
    }

    async fn observe_stuck(
        &mut self,
        receipt: &Receipt,
        output: &str,
    ) -> Result<Option<StuckAction>, CoreError> {
        let Some(signal) = self.state.stuck.observe(&self.agent.stuck, receipt, output) else {
            return Ok(None);
        };
        self.emit(Event::Stuck {
            signal: signal.clone(),
        })
        .await?;

        let decision = {
            let ctx = self.ctx();
            let mut decision = None;
            for hook in &self.agent.hooks {
                if let Some(d) = hook.on_stuck(&ctx, &signal).await {
                    decision = Some(d);
                    break;
                }
            }
            decision
        }
        .unwrap_or_else(|| {
            if self.state.nudges < self.agent.stuck.max_nudges {
                StuckDecision::Nudge {
                    message: signal.nudge_text(),
                }
            } else {
                StuckDecision::Stop
            }
        });

        self.state.stuck.reset();
        Ok(match decision {
            StuckDecision::Nudge { message } => {
                self.state.nudges += 1;
                Some(StuckAction::Nudge(message))
            }
            StuckDecision::Stop => Some(StuckAction::Stop(signal)),
            StuckDecision::Ignore => None,
        })
    }

    /// Stream one model call, forwarding text deltas as events.
    async fn call_model(&self, request: &ModelRequest, turn: u32) -> Result<ModelCall, CoreError> {
        let mut events = self.agent.provider.stream(request);
        loop {
            let next = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => return Ok(ModelCall::Cancelled),
                next = events.next() => next,
            };
            match next {
                Some(Ok(StreamEvent::TextDelta(text))) => {
                    self.emit(Event::TextDelta { turn, text }).await?;
                }
                Some(Ok(StreamEvent::Done(response))) => return Ok(ModelCall::Response(response)),
                Some(Err(err)) => return Ok(ModelCall::Failed(err.to_string())),
                None => {
                    return Ok(ModelCall::Failed(
                        "provider stream ended without a final response".into(),
                    ))
                }
            }
        }
    }

    /// Finish a batch that was running when the process stopped. Calls that
    /// never started run normally. Calls that started but have no result are
    /// re-run if their tool is idempotent; otherwise they get an
    /// `Indeterminate` receipt telling the model to verify before retrying.
    async fn recover(&mut self, in_flight: InFlight) -> Result<BatchEnd, CoreError> {
        let mut reran = Vec::new();
        let mut indeterminate = Vec::new();
        for call in &in_flight.calls {
            let started = in_flight.started.contains(&call.id);
            if !started || in_flight.done.contains(&call.id) {
                continue;
            }
            // Replay safety is a property of this call's arguments.
            let idempotent = self
                .agent
                .tools
                .get(&call.name)
                .map(|tool| tool.annotations_for(&call.arguments).idempotent)
                .unwrap_or(false);
            if idempotent {
                reran.push(call.id.clone());
                continue;
            }
            indeterminate.push(call.id.clone());
            let (mut receipt, output) = rejection(
                self.state.run_id,
                call,
                self.state.turn,
                ReceiptStatus::Indeterminate,
                format!(
                    "`{}` was running when the agent restarted and may or may not have \
                     completed. Check its effect before retrying it.",
                    call.name
                ),
            );
            receipt.stage = InvocationStage::Dispatched;
            // The stuck policy cannot act mid-recovery; it sees the next call.
            let _ = self.record(receipt, output).await?;
        }
        self.emit(Event::Recovered {
            reran,
            indeterminate,
        })
        .await?;
        self.execute_batch(&in_flight.calls, 0, None, None).await
    }

    fn is_done(&self, call_id: &str) -> bool {
        self.state
            .in_flight
            .as_ref()
            .is_some_and(|f| f.done.iter().any(|id| id == call_id))
    }

    fn build_request(&self) -> ModelRequest {
        let mut messages = Vec::with_capacity(self.state.messages.len() + 1);
        if !self.agent.instructions.is_empty() {
            messages.push(Message::system(self.agent.instructions.clone()));
        }
        messages.extend(self.state.messages.iter().cloned());
        ModelRequest {
            model: self.agent.model.clone(),
            messages,
            tools: self.agent.specs.clone(),
            tool_choice: ToolChoice::Auto,
            max_output_tokens: self.agent.max_output_tokens,
        }
    }

    fn ctx(&self) -> HookContext<'_> {
        HookContext {
            run_id: self.state.run_id,
            agent_id: &self.agent.id,
            session_id: &self.state.session_id,
            turn: self.state.turn,
            messages: &self.state.messages,
            receipts: &self.state.receipts,
            usage: self.state.usage,
            run_usage: self.state.run_usage,
            untrusted_seen: self.state.untrusted_seen,
            declined: &self.state.declined,
        }
    }

    fn elapsed(&self) -> Duration {
        Duration::from_millis(self.base_elapsed_ms) + self.segment_start.elapsed()
    }

    fn snapshot(&self) -> RunSnapshot {
        RunSnapshot {
            turn: self.state.turn,
            tool_calls: self.state.tool_calls,
            usage: self.state.run_usage,
            elapsed: self.elapsed(),
        }
    }

    /// Wall-clock budget left, if the run has one.
    fn remaining(&self) -> Option<Duration> {
        self.agent
            .limits
            .timeout
            .map(|timeout| timeout.saturating_sub(self.elapsed()))
    }

    /// Limits that stop a batch before its next call. `queued` approved
    /// read-only calls are waiting to run and count toward `max_tool_calls`.
    fn call_limit(&self, queued: usize) -> Option<LimitKind> {
        let limits = &self.agent.limits;
        if let Some(max) = limits.max_tool_calls {
            if u64::from(self.state.tool_calls) + queued as u64 >= u64::from(max) {
                return Some(LimitKind::ToolCalls);
            }
        }
        if self.remaining() == Some(Duration::ZERO) {
            return Some(LimitKind::WallClock);
        }
        None
    }

    async fn emit(&self, event: Event) -> Result<(), CoreError> {
        if let Some(sink) = &self.agent.sink {
            sink.emit(self.state.run_id, &event);
        }
        if let Some(store) = &self.agent.store {
            store.append_event(self.state.run_id, &event).await?;
        }
        Ok(())
    }

    async fn checkpoint(&mut self) -> Result<(), CoreError> {
        self.state.elapsed_ms = self.elapsed().as_millis() as u64;
        if let Some(store) = &self.agent.store {
            store.save_state(&self.state).await?;
        }
        Ok(())
    }

    async fn finish_batch(self, end: BatchEnd) -> Result<RunOutcome, CoreError> {
        let status = match end {
            BatchEnd::Interrupted(interruption) => RunStatus::Interrupted { interruption },
            BatchEnd::Stuck(signal) => RunStatus::Stuck { signal },
            BatchEnd::Limit(limit) => RunStatus::LimitReached { limit },
            BatchEnd::Cancelled => RunStatus::Cancelled,
            BatchEnd::Done => unreachable!("finish_batch is only called for terminal batch ends"),
        };
        self.finish(status, None).await
    }

    async fn finish(
        mut self,
        status: RunStatus,
        final_text: Option<String>,
    ) -> Result<RunOutcome, CoreError> {
        self.checkpoint().await?;
        self.emit(Event::Finished {
            status: status.clone(),
            usage: self.state.usage,
            turns: self.state.turn,
        })
        .await?;
        let outcome = RunOutcome {
            status,
            final_text,
            state: self.state,
        };
        if self.report_run_end {
            report_run_end(self.agent, &outcome).await;
        }
        Ok(outcome)
    }
}

/// Tell every hook the run is over, with its final outcome.
pub(crate) async fn report_run_end(agent: &AgentInner, outcome: &RunOutcome) {
    let ctx = HookContext {
        run_id: outcome.state.run_id,
        agent_id: &agent.id,
        session_id: &outcome.state.session_id,
        turn: outcome.state.turn,
        messages: &outcome.state.messages,
        receipts: &outcome.state.receipts,
        usage: outcome.state.usage,
        run_usage: outcome.state.run_usage,
        untrusted_seen: outcome.state.untrusted_seen,
        declined: &outcome.state.declined,
    };
    for hook in &agent.hooks {
        hook.on_run_end(&ctx, outcome).await;
    }
}

fn track(action: Option<StuckAction>, nudge: &mut Option<String>, stop: &mut Option<StuckSignal>) {
    match action {
        Some(StuckAction::Nudge(message)) => *nudge = Some(message),
        Some(StuckAction::Stop(signal)) => *stop = Some(signal),
        None => {}
    }
}

/// A receipt and model-facing message for a call that never ran.
fn rejection(
    run_id: RunId,
    call: &ToolCall,
    turn: u32,
    status: ReceiptStatus,
    reason: String,
) -> (Receipt, ToolOutput) {
    let prefix = match status {
        ReceiptStatus::Denied => "Denied",
        ReceiptStatus::Cancelled => "Cancelled",
        ReceiptStatus::Skipped => "Skipped",
        _ => "Error",
    };
    let output = ToolOutput::error(format!("{prefix}: {reason}"));
    let receipt = Receipt {
        run_id,
        call_id: call.id.clone(),
        tool: call.name.clone(),
        arguments: call.arguments.clone(),
        turn,
        stage: InvocationStage::RejectedBeforeDispatch,
        status,
        read_only: false,
        duration_ms: 0,
        reason: Some(reason),
        details: serde_json::Value::Null,
        source: SourceTrust::Trusted,
        after_untrusted: false,
    };
    (receipt, output)
}

/// Run `future` within `budget`; `None` if the budget ran out first.
async fn within<F: Future>(budget: Option<Duration>, future: F) -> Option<F::Output> {
    match budget {
        Some(budget) => tokio::time::timeout(budget, future).await.ok(),
        None => Some(future.await),
    }
}

fn limit_name(limit: LimitKind) -> &'static str {
    match limit {
        LimitKind::Turns => "turn",
        LimitKind::Tokens => "token",
        LimitKind::ToolCalls => "tool-call",
        LimitKind::WallClock => "wall-clock",
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = panic.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = panic.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Keep the head and tail of long output; the middle is usually the least
/// useful part of logs and listings.
fn truncate_middle(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars || max_chars == 0 {
        return text.to_string();
    }
    let head = max_chars * 2 / 3;
    let tail = max_chars - head;
    let omitted = total - head - tail;
    let head_text: String = text.chars().take(head).collect();
    let tail_text: String = text.chars().skip(total - tail).collect();
    format!(
        "{head_text}\n\n[... {omitted} characters omitted by the harness; narrow the request to see them ...]\n\n{tail_text}"
    )
}

/// The refusal for a call with an effect the owner declined earlier in the
/// run, if it has one.
fn declined_reason(declined: &[EffectKey], info: &ToolInfo) -> Option<String> {
    let effect = info
        .effects
        .keys
        .iter()
        .find(|key| declined.iter().any(|d| d.overlaps(key)))?;
    Some(format!(
        "{OWNER_DECLINED} The owner declined {} {} earlier in this run. Do not attempt it \
         another way; tell the owner it was not done.",
        effect.class, effect.target
    ))
}

#[cfg(test)]
mod tests {
    use super::truncate_middle;

    #[test]
    fn truncation_keeps_head_and_tail() {
        let text: String = (0..1000)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let cut = truncate_middle(&text, 90);
        assert!(cut.starts_with(&text[..60]));
        assert!(cut.ends_with(&text[970..]));
        assert!(cut.contains("910 characters omitted"));
        assert_eq!(truncate_middle("short", 90), "short");
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        let text = "é".repeat(500);
        let cut = truncate_middle(&text, 30);
        assert!(cut.starts_with(&"é".repeat(20)));
    }
}
