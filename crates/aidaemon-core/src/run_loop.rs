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
use crate::approval::{
    EffectClass, EffectKey, InterruptKind, SharedDeclines, OWNER_DECLINED, OWNER_UNANSWERED,
};
use crate::authority::Authority;
use crate::error::CoreError;
use crate::event::Event;
use crate::hook::{
    Hook, HookContext, HookFailure, HookPoint, LimitDecision, ModelDecision, ModelErrorDecision,
    StopDecision, StuckDecision, ToolDecision,
};
use crate::limits::{LimitKind, Limits, RunSnapshot, HARD_TURN_CAP};
use crate::message::{Message, MessageOrigin, Role, ToolCall};
use crate::outcome::{RunOutcome, RunStatus};
use crate::provider::{ModelRequest, ModelResponse, ProviderError, StreamEvent, ToolChoice};
use crate::receipt::{InvocationStage, Receipt, ReceiptStatus};
use crate::state::{InFlight, Interruption, Pending, PendingFinish, RunId, RunState};
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
    /// The decline set was made for this segment, not given by the caller;
    /// it does not outlive the segment.
    own_declines: bool,
}

const STUCK_SKIP_REASON: &str =
    "the run was stopped by the stuck detector before this call started";

/// How a tool batch ended.
#[allow(clippy::large_enum_variant)] // short-lived; moved into `RunStatus`
enum BatchEnd {
    Done,
    Interrupted(Interruption),
    Stuck(StuckSignal),
    Limit(LimitKind),
    Cancelled,
    /// A call asked to end the run and its batch ended (see
    /// `execute_batch`).
    Finished(PendingFinish),
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
    Failed(ProviderError),
}

/// The settled policy decision for one call (rewrites already applied).
enum Verdict {
    Allow,
    Deny(String),
    /// The reason, its kind, and the index of the hook that interrupted.
    Interrupt(String, InterruptKind, usize),
    /// A hook answered in the tool's place.
    Answer(ToolOutput),
}

/// A call that passed checks and policy, waiting to execute.
struct Approved {
    call: ToolCall,
    annotations: ToolAnnotations,
}

/// Who asked the question a batch is held on.
enum Asker {
    /// The hook at `index`, after the owner answered the hooks in `waived`
    /// for this same call.
    Hook { index: usize, waived: Vec<usize> },
    /// A child run the call started, paused on its own question.
    Child(Box<RunState>),
}

/// What a dispatch asks of the tool.
enum Invocation {
    Call,
    /// Pass an answer to the child run the tool suspended.
    Resume {
        child: Box<RunState>,
        resolution: Resolution,
    },
}

impl<'a> Driver<'a> {
    pub(crate) fn new(agent: &'a AgentInner, state: RunState, cancel: CancellationToken) -> Self {
        let base_elapsed_ms = state.elapsed_ms;
        // Whatever version it was saved at, it is written back at this one.
        let mut state = state;
        state.schema = crate::state::RUN_STATE_SCHEMA;
        // Tools that start other runs share this set, so a decline in any of
        // them binds the rest.
        let own_declines = state.shared_declines().is_none();
        let mut state = if own_declines {
            state.share_declines(SharedDeclines::new())
        } else {
            state
        };
        // What the caller put in (the user's message, an earlier run's
        // transcript) is redacted before anything reads it.
        let scrub = crate::redaction::Scrubber {
            hooks: &agent.hooks,
        };
        for message in &mut state.messages {
            scrub.message(message);
        }
        Self {
            agent,
            state,
            cancel,
            segment_start: Instant::now(),
            base_elapsed_ms,
            own_declines,
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
        if let Some(Resolution::Unanswered { reason }) = &resolution {
            let reason = if reason.starts_with(OWNER_UNANSWERED) {
                reason.clone()
            } else {
                format!("{OWNER_UNANSWERED} {reason}")
            };
            return Some(Resolution::Deny { reason });
        }
        let (request, reason) = match (&interruption.kind, &resolution) {
            (_, Some(Resolution::OwnerDeclined { request, reason }))
            | (InterruptKind::OwnerApproval(request), Some(Resolution::Deny { reason })) => {
                (request, reason)
            }
            _ => return resolution,
        };
        // A request without effects is remembered by its exact subject; a
        // blank subject names no call, so it binds nothing.
        let call_key;
        let effects: &[EffectKey] = if !request.effects.is_empty() {
            &request.effects
        } else if request.subject.trim().is_empty() {
            &[]
        } else {
            call_key = [EffectKey::named(EffectClass::Call, &request.subject)];
            &call_key
        };
        for effect in effects {
            if !self.state.declined.contains(effect) {
                self.state.declined.push(effect.clone());
            }
        }
        // The runs sharing this run's declines (its parent, its children)
        // are refused the effect from their next call on.
        self.state.sync_declines();
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
        if let Some(mut pending) = self.state.pending.take() {
            let resolution = self.settle(&pending.interruption, resolution);
            let earlier: Vec<String> = pending.calls[..pending.next]
                .iter()
                .map(|c| c.id.clone())
                .collect();
            self.state.in_flight = Some(InFlight {
                calls: pending.calls.clone(),
                started: earlier.clone(),
                done: earlier,
            });
            // An approval waives the hooks it answered, and only for the
            // exact call it was given for; anything else is judged from
            // the first hook.
            let waived = match resolution {
                Some(Resolution::Approve) if pending.child.is_none() => {
                    self.verify_approval(&pending).await?
                }
                _ => Vec::new(),
            };
            self.emit(Event::RunResumed).await?;
            let child = pending.child.take();
            match self
                .execute_batch(&pending.calls, pending.next, resolution, waived, child)
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
            if let Some(parent_run) = self.state.parent_run {
                self.emit(Event::RunLinked { parent_run }).await?;
            }
            // A process restart between turns also lands here; it is not a
            // new run.
            if self.state.turn == 0 {
                let ctx = self.ctx();
                let guard = self.guard(HookPoint::OnRunStart);
                for hook in &self.agent.hooks {
                    guard.run(hook.as_ref(), hook.on_run_start(&ctx)).await;
                }
            }
        }

        loop {
            // 1. Guards: cancellation, hard limits, custom stop conditions.
            if self.cancel.is_cancelled() {
                return self.finish(RunStatus::Cancelled, None).await;
            }
            // A hook is asked once per limit per turn, so a grant too small
            // to clear the limit ends the run instead of asking forever.
            let mut raised: Vec<LimitKind> = Vec::new();
            while let Some(limit) = self.limits().exceeded(&self.snapshot()) {
                if raised.contains(&limit) || !self.extend_limit(limit).await? {
                    return self.finish(RunStatus::LimitReached { limit }, None).await;
                }
                raised.push(limit);
            }
            let snapshot = self.snapshot();
            if let Some(reason) = self
                .agent
                .stop_conditions
                .iter()
                .find_map(|c| c.check(&snapshot))
            {
                return self.finish(RunStatus::Stopped { reason }, None).await;
            }

            // 2. Build the request and let hooks shape it. Hooks see what
            // the runs sharing this run's declines declined meanwhile.
            self.state.sync_declines();
            self.state.turn += 1;
            let turn = self.state.turn;
            self.emit(Event::TurnStarted { turn }).await?;
            self.apply_transcript_edits().await?;
            let mut request = self.build_request();
            {
                let ctx = self.ctx();
                let guard = self.guard(HookPoint::BeforeModel);
                for hook in &self.agent.hooks {
                    guard
                        .run(hook.as_ref(), hook.before_model(&ctx, &mut request))
                        .await;
                }
            }
            // Whatever the hooks added, no secret reaches the model.
            for message in &mut request.messages {
                self.scrubber().message(message);
            }

            // 3. Call the model (streamed; non-streaming providers yield a
            //    single final event), bounded by the remaining wall-clock budget.
            let call = match self.answer_from_hooks(&request).await {
                Some((hook, response)) => {
                    self.emit(Event::ModelAnswered { turn, hook }).await?;
                    Some(Ok(ModelCall::Response(response)))
                }
                None => {
                    self.emit(Event::ModelRequested {
                        turn,
                        model: request.model.clone(),
                        messages: request.messages.len(),
                        tools: request.tools.len(),
                    })
                    .await?;
                    for sink in &self.agent.sinks {
                        sink.observe_request(self.state.run_id, &request);
                    }
                    within(self.remaining(), self.call_model(&request, turn)).await
                }
            };
            let response = match call {
                None => {
                    let limit = LimitKind::WallClock;
                    if self.extend_limit(limit).await? {
                        continue;
                    }
                    return self.finish(RunStatus::LimitReached { limit }, None).await;
                }
                Some(call) => call?,
            };
            let response = match response {
                ModelCall::Response(response) => response,
                ModelCall::Cancelled => return self.finish(RunStatus::Cancelled, None).await,
                ModelCall::Failed(error) => match self.judge_model_error(&error).await {
                    (_, ModelErrorDecision::Fail | ModelErrorDecision::Abort) => {
                        let error = error.to_string();
                        return self.finish(RunStatus::ProviderFailed { error }, None).await;
                    }
                    (_, ModelErrorDecision::Retry { after }) => {
                        self.emit(Event::ModelRetried {
                            turn,
                            error: error.to_string(),
                            after_ms: after.as_millis() as u64,
                        })
                        .await?;
                        self.pause(after).await;
                        continue;
                    }
                    (hook, ModelErrorDecision::Recover { response }) => {
                        self.emit(Event::ModelAnswered { turn, hook }).await?;
                        response
                    }
                },
            };
            // Counted before any hook runs: a rejected response was still
            // paid for, and no hook can hide what it cost.
            let usage = response.usage;
            self.state.usage.add(usage);
            self.state.run_usage.add(usage);
            let mut response = response;
            let rejection = match self.judge_response(&mut response).await {
                Ok(rejection) => rejection,
                Err(reason) => return self.finish(RunStatus::Stopped { reason }, None).await,
            };
            let calls: Vec<ToolCall> = response
                .tool_calls
                .into_iter()
                .enumerate()
                .map(|(i, mut call)| {
                    if call.id.trim().is_empty() {
                        call.id = format!("call_{turn}_{i}");
                    }
                    // `_` keys are the dispatcher's; the model cannot send
                    // them. Dropped before anything records or judges it.
                    if let Some(map) = call.arguments.as_object_mut() {
                        crate::tool::strip_reserved_arguments(map);
                    }
                    call
                })
                .collect();
            self.emit(Event::ModelResponded {
                turn,
                usage,
                text: response.text.clone(),
                tool_calls: calls.len(),
                note: response.note.clone(),
            })
            .await?;
            if let Some(message) = rejection {
                self.emit(Event::ModelRejected {
                    turn,
                    message: message.clone(),
                })
                .await?;
                // Without the calls: a call with no result is a request
                // providers refuse, and these will never run.
                if !response.text.is_empty() {
                    let mut message = Message::assistant(response.text, Vec::new())
                        .with_reasoning(response.reasoning);
                    self.scrubber().message(&mut message);
                    self.state.messages.push(message);
                }
                self.state
                    .messages
                    .push(Message::from_origin(MessageOrigin::Harness, message));
                self.checkpoint().await?;
                continue;
            }
            // The transcript keeps the calls redacted; `calls` keeps the
            // exact arguments the tools run with.
            let mut message = Message::assistant(response.text.clone(), calls.clone())
                .with_reasoning(response.reasoning.clone());
            self.scrubber().message(&mut message);
            self.state.messages.push(message);

            // 4a. No tool calls: the model thinks it is done. Ask stop hooks.
            if calls.is_empty() {
                // The first hook that asks to continue wins. Otherwise the
                // run stops, carrying every requirement a hook reported as
                // unmet: stopping is permitted, success is not implied.
                let (rejection, unmet) = {
                    let ctx = self.ctx();
                    let guard = self.guard(HookPoint::OnStop);
                    let mut rejection = None;
                    let mut halted = false;
                    let mut unmet: Vec<String> = Vec::new();
                    for hook in &self.agent.hooks {
                        let answer = guard
                            .run(hook.as_ref(), hook.on_stop(&ctx, &response.text))
                            .await
                            .unwrap_or_else(|| StopDecision::AcceptUnmet {
                                unmet: vec![format!(
                                    "the completion check `{}` gave no answer",
                                    hook.name()
                                )],
                            });
                        match answer {
                            StopDecision::Accept => {}
                            StopDecision::Continue { message } => {
                                guard.decided(hook.name(), "continue", Some(&message)).await;
                                rejection.get_or_insert(message);
                            }
                            StopDecision::Halt { reason } => {
                                guard.decided(hook.name(), "halt", Some(&reason)).await;
                                halted = true;
                                if !unmet.contains(&reason) {
                                    unmet.push(reason);
                                }
                            }
                            StopDecision::AcceptUnmet { unmet: lines } => {
                                guard
                                    .decided(hook.name(), "accept_unmet", Some(&lines.join("; ")))
                                    .await;
                                for line in lines {
                                    let line = line.trim().to_string();
                                    if !line.is_empty() && !unmet.contains(&line) {
                                        unmet.push(line);
                                    }
                                }
                            }
                        }
                    }
                    (if halted { None } else { rejection }, unmet)
                };
                let (rejection, mut unmet) = (rejection, unmet);
                if self.cancel.is_cancelled() {
                    return self.finish(RunStatus::Cancelled, None).await;
                }
                let rejection = rejection.filter(|_| {
                    let room = self.record_continuation();
                    if !room {
                        unmet.push(format!(
                            "the work was sent back {} times in a row with no new successful \
                             call, so the run stopped",
                            self.limits().max_idle_continuations
                        ));
                    }
                    room
                });
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
                        self.state.unmet = unmet;
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
            match self
                .execute_batch(&calls, 0, None, Vec::new(), None)
                .await?
            {
                BatchEnd::Done => self.checkpoint().await?,
                end => return self.finish_batch(end).await,
            }
        }
    }

    /// Run `calls[start..]` and apply a finish a call returned
    /// (`RunState::finishing`, which survives interruptions and restarts).
    /// A finish whose tool already ran ends the run once the batch ends,
    /// whether it ran to the end or stopped on a limit or the stuck
    /// detector. A cancel drops it: the owner's cancel wins. An
    /// interruption keeps it for the resumed batch.
    async fn execute_batch(
        &mut self,
        calls: &[ToolCall],
        start: usize,
        resolution: Option<Resolution>,
        waived: Vec<usize>,
        child: Option<Box<RunState>>,
    ) -> Result<BatchEnd, CoreError> {
        let end = self
            .run_batch(calls, start, resolution, waived, child)
            .await?;
        Ok(match end {
            BatchEnd::Done | BatchEnd::Limit(_) | BatchEnd::Stuck(_) => {
                match self.state.finishing.take() {
                    Some(finishing) => BatchEnd::Finished(finishing),
                    None => end,
                }
            }
            BatchEnd::Cancelled => {
                self.state.finishing = None;
                end
            }
            BatchEnd::Interrupted(_) | BatchEnd::Finished(_) => end,
        })
    }

    /// Run `calls[start..]`. Consecutive approved read-only calls run
    /// concurrently; anything else runs alone, in order. Results are always
    /// appended in call order.
    async fn run_batch(
        &mut self,
        calls: &[ToolCall],
        start: usize,
        mut resolution: Option<Resolution>,
        mut waived: Vec<usize>,
        mut child: Option<Box<RunState>>,
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
                // Asked after the flush, so a hook sees every result so far.
                let raised = stop.is_none()
                    && !self.cancel.is_cancelled()
                    && self.extend_limit(limit).await?;
                if let Some(limit) = if raised {
                    self.call_limit(0)
                } else {
                    Some(limit)
                } {
                    let reason = format!("the run reached its {} limit", limit_name(limit));
                    self.skip_rest(&calls[index..], ReceiptStatus::Skipped, &reason)
                        .await?;
                    return Ok(BatchEnd::Limit(limit));
                }
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
            if let Err(reason) = self.check_call(tool.as_ref(), &call) {
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
            let suspends = tool.suspends();
            let peers = tool.concurrent_peers();
            let may_run_concurrently = (tool.annotations().read_only || peers > 0) && !suspends;
            let mut info = ToolInfo {
                annotations,
                access: tool.access(&call.arguments),
                effects: tool.effects(&call.arguments),
                outbound: tool.outbound(&call.arguments),
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

            // This call is already inside its tool: it passed policy when it
            // started, and the answer is for the run paused below it.
            if let Some(child) = child.take() {
                let resolution = resolution.take().unwrap_or_else(|| Resolution::Deny {
                    reason: "the run was resumed without an answer".to_string(),
                });
                let item = Approved { call, annotations };
                let invocation = Invocation::Resume { child, resolution };
                if let Some(end) = self
                    .run_suspendable(item, invocation, calls, index, &mut nudge, &mut stop)
                    .await?
                {
                    return Ok(end);
                }
                continue;
            }

            // Policy: a pending resolution settles the first call; otherwise
            // ask the hooks.
            let settled = if index == start {
                resolution.take()
            } else {
                None
            };
            // Hooks answered for this call; a rewrite makes it another
            // call and forgets them.
            let mut waived_now = Vec::new();
            let mut verdict = match settled {
                // An approval waives only the questions it answered: every
                // other hook, before the one that asked as well as after
                // it, judges the call again.
                Some(Resolution::Approve) => {
                    waived_now = std::mem::take(&mut waived);
                    self.decide(&mut call, &mut info, &mut waived_now).await
                }
                Some(
                    Resolution::Deny { reason }
                    | Resolution::OwnerDeclined { reason, .. }
                    | Resolution::Unanswered { reason },
                ) => Verdict::Deny(reason),
                None => self.decide(&mut call, &mut info, &mut waived_now).await,
            };
            // A rewrite may introduce egress after reads were queued. Let
            // those reads taint the run before judging the rewritten call.
            if info.annotations.egress && !group.is_empty() {
                if let Some(end) = self
                    .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                    .await?
                {
                    return Ok(end);
                }
                verdict = self.decide(&mut call, &mut info, &mut waived_now).await;
            }
            // A hook may have rewritten the call; judge and record what runs.
            let annotations = info.annotations;

            match verdict {
                Verdict::Allow if may_run_concurrently => {
                    // A coordinated tool runs at most `peers` of its calls at
                    // once: a full share flushes the queue before this one.
                    let queued_peers = group
                        .iter()
                        .filter(|queued| queued.call.name == call.name)
                        .count();
                    if peers > 0 && queued_peers >= peers {
                        if let Some(end) = self
                            .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                            .await?
                        {
                            return Ok(end);
                        }
                    }
                    group.push(Approved { call, annotations });
                }
                Verdict::Allow if suspends => {
                    if let Some(end) = self
                        .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                        .await?
                    {
                        return Ok(end);
                    }
                    let item = Approved { call, annotations };
                    if let Some(end) = self
                        .run_suspendable(
                            item,
                            Invocation::Call,
                            calls,
                            index,
                            &mut nudge,
                            &mut stop,
                        )
                        .await?
                    {
                        return Ok(end);
                    }
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
                Verdict::Answer(mut output) => {
                    if let Some(end) = self
                        .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                        .await?
                    {
                        return Ok(end);
                    }
                    // An answer is not the tool's result: it never ends the
                    // run.
                    output.finish = None;
                    let receipt = Receipt {
                        run_id: self.state.run_id,
                        call_id: call.id.clone(),
                        tool: call.name.clone(),
                        arguments: call.arguments.clone(),
                        turn,
                        stage: InvocationStage::RejectedBeforeDispatch,
                        status: ReceiptStatus::Answered,
                        read_only: annotations.read_only,
                        duration_ms: 0,
                        reason: None,
                        details: output.details.clone(),
                        source: output.trust.unwrap_or(SourceTrust::Trusted),
                        after_untrusted: self.state.untrusted_seen,
                    };
                    track(self.record(receipt, output).await?, &mut nudge, &mut stop);
                }
                Verdict::Interrupt(reason, kind, hook_index) => {
                    if let Some(end) = self
                        .flush_or_stop(&mut group, &mut nudge, &mut stop, &calls[index..])
                        .await?
                    {
                        return Ok(end);
                    }
                    // Save the call exactly as the hooks left it, bound to
                    // the question by a digest: a resumed `Approve` waives
                    // this hook (and those already answered) for this call
                    // only.
                    let interruption = Interruption {
                        call: call.clone(),
                        reason,
                        kind,
                        digest: None,
                    };
                    return self
                        .hold_batch(
                            calls,
                            index,
                            call,
                            interruption,
                            Asker::Hook {
                                index: hook_index,
                                waived: waived_now,
                            },
                        )
                        .await;
                }
            }
        }
        self.flush(&mut group, &mut nudge, &mut stop).await?;
        self.state.in_flight = None;

        // Every call ran: a tool's finish ends the run (`execute_batch`),
        // ahead of any stuck verdict or nudge about the batch.
        if self.state.finishing.is_some() {
            return Ok(BatchEnd::Done);
        }
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

    /// Save the batch as waiting on `calls[index]`, sent as `call`.
    async fn hold_batch(
        &mut self,
        calls: &[ToolCall],
        index: usize,
        call: ToolCall,
        interruption: Interruption,
        asker: Asker,
    ) -> Result<BatchEnd, CoreError> {
        let (hook_index, child, mut waived) = match asker {
            Asker::Hook { index, waived } => (Some(index), None, waived),
            Asker::Child(child) => (None, Some(child), Vec::new()),
        };
        let hook_chain = self.hook_chain();
        let mut interruption = interruption;
        waived.sort_unstable();
        waived.dedup();
        if let Some(hook_index) = hook_index {
            let question = crate::binding::Question {
                run_id: self.state.run_id,
                turn: self.state.turn,
                call: &call,
                hook_index,
                hook_name: hook_chain.get(hook_index).map_or("", String::as_str),
                waived: &waived,
            };
            interruption.digest = Some(crate::binding::digest(&self.approval_key(), &question));
        }
        // The owner sees the question without secrets; the digest and the
        // pending call keep the exact arguments.
        self.scrubber().json(&mut interruption.call);
        self.scrubber().text(&mut interruption.reason);
        self.scrubber().json(&mut interruption.kind);
        let mut pending_calls = calls.to_vec();
        pending_calls[index] = call;
        self.state.in_flight = None;
        self.state.pending = Some(Pending {
            calls: pending_calls,
            next: index,
            interruption: interruption.clone(),
            hook_index,
            hook_chain,
            child,
            waived,
        });
        self.emit(Event::Interrupted {
            interruption: interruption.clone(),
        })
        .await?;
        Ok(BatchEnd::Interrupted(interruption))
    }

    /// Run a call of a tool that may suspend, alone. Marking it started is
    /// checkpointed first, which also consumes an answer being passed down:
    /// after a crash the call is recovered like any other that was running,
    /// so the answer is not applied twice and the child's finished calls do
    /// not run again. Returns how the batch ends if the child paused.
    async fn run_suspendable(
        &mut self,
        item: Approved,
        invocation: Invocation,
        calls: &[ToolCall],
        index: usize,
        nudge: &mut Option<String>,
        stop: &mut Option<StuckSignal>,
    ) -> Result<Option<BatchEnd>, CoreError> {
        if let Some(in_flight) = self.state.in_flight.as_mut() {
            in_flight.started.push(item.call.id.clone());
        }
        self.checkpoint().await?;
        if matches!(invocation, Invocation::Call) {
            self.emit(Event::ToolStarted {
                call_id: item.call.id.clone(),
                tool: item.call.name.clone(),
            })
            .await?;
        }
        let (receipt, mut output) = self.dispatch(&item, invocation).await;
        let paused = output.suspended.take().and_then(|child| {
            let interruption = child.pending.as_ref()?.interruption.clone();
            Some((interruption, child))
        });
        if let Some((interruption, mut child)) = paused {
            // A runtime handle: the resumed child gets the set in force then.
            child.shared_declines = None;
            let end = self
                .hold_batch(calls, index, item.call, interruption, Asker::Child(child))
                .await?;
            return Ok(Some(end));
        }
        track(self.record(receipt, output).await?, nudge, stop);
        self.checkpoint().await?;
        Ok(None)
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

    /// Ask `before_tool` hooks in order. Changed arguments restart all
    /// judgments; identical rewrites are no-ops. Non-converging chains fail closed.
    ///
    /// A call with an effect the owner declined earlier in the run is
    /// refused before any hook sees it, and again after any rewrite: no
    /// hook or application can opt out, so a model cannot retry a declined
    /// action through another tool. That includes effects declined in a
    /// run sharing this run's declines ([`crate::SharedDeclines`]), however
    /// recently: they are taken in first.
    ///
    /// Hooks in `waived` are skipped: the owner answered their question
    /// about this exact call. A rewrite makes it another call, so it clears
    /// `waived`.
    async fn decide(
        &mut self,
        call: &mut ToolCall,
        info: &mut ToolInfo,
        waived: &mut Vec<usize>,
    ) -> Verdict {
        self.state.sync_declines();
        if let Some(reason) = declined_reason(&self.state.declined, info) {
            return Verdict::Deny(reason);
        }
        let ctx = self.ctx();
        let guard = self.guard(HookPoint::BeforeTool);
        let mut hook_index = 0;
        let mut rewrites = 0;
        while let Some(hook) = self.agent.hooks.get(hook_index) {
            if waived.contains(&hook_index) {
                hook_index += 1;
                continue;
            }
            let answer = guard
                .run(hook.as_ref(), hook.before_tool(&ctx, call, info))
                .await
                .unwrap_or_else(|| ToolDecision::Deny {
                    reason: format!("the safety check `{}` gave no answer", hook.name()),
                });
            let name = hook.name();
            match &answer {
                ToolDecision::Deny { reason } => {
                    guard.decided(name, "deny", Some(reason)).await;
                }
                ToolDecision::Interrupt { reason, .. } => {
                    guard.decided(name, "interrupt", Some(reason)).await;
                }
                ToolDecision::Answer { .. } => guard.decided(name, "answer", None).await,
                ToolDecision::Rewrite { arguments } if *arguments != call.arguments => {
                    guard.decided(name, "rewrite", None).await;
                }
                _ => {}
            }
            match answer {
                ToolDecision::Allow => {}
                ToolDecision::Deny { reason } => return Verdict::Deny(reason),
                ToolDecision::Interrupt { reason, kind } => {
                    return Verdict::Interrupt(reason, kind, hook_index)
                }
                ToolDecision::Answer { output } => return Verdict::Answer(output),
                ToolDecision::Rewrite { arguments } if arguments != call.arguments => {
                    rewrites += 1;
                    if rewrites > 16 {
                        return Verdict::Deny(
                            "tool hooks did not converge after 16 rewrites".into(),
                        );
                    }
                    call.arguments = arguments;
                    if let Some(tool) = self.agent.tools.get(&call.name) {
                        if let Err(reason) = self.check_call(tool.as_ref(), call) {
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
                    // Every judgment applies to exact arguments. A rewrite
                    // invalidates earlier authority, scans and approvals.
                    hook_index = 0;
                    waived.clear();
                    continue;
                }
                ToolDecision::Rewrite { .. } => {}
            }
            hook_index += 1;
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
        let results = join_all(
            items
                .iter()
                .map(|item| this.dispatch(item, Invocation::Call)),
        )
        .await;
        for (receipt, output) in results {
            let action = self.record(receipt, output).await?;
            track(action, nudge, stop);
        }
        self.checkpoint().await?;
        Ok(())
    }

    /// Run one tool with a timeout and cancellation, producing its receipt.
    async fn dispatch(&self, item: &Approved, invocation: Invocation) -> (Receipt, ToolOutput) {
        let call = &item.call;
        let tool = self
            .agent
            .tools
            .get(&call.name)
            .expect("dispatch only sees calls that passed lookup")
            .clone();
        let ctx = ToolContext::new(self.state.session_id.clone(), call.id.clone())
            .with_run(self.state.run_id, self.state.turn)
            .with_cancel(self.cancel.child_token())
            .with_declines(self.state.shared_declines().cloned().unwrap_or_default())
            .with_authority(self.run_authority())
            .with_extensions(self.agent.extensions.clone());
        // A tool's own timeout hint replaces the default, but nothing may
        // outlive the run's remaining wall-clock budget.
        let per_tool = tool.timeout().unwrap_or(self.agent.limits.tool_timeout);
        let timeout = match self.remaining() {
            Some(remaining) => remaining.min(per_tool),
            None => per_tool,
        };
        let started = Instant::now();
        let work = async {
            match invocation {
                Invocation::Call => tool.call(call.arguments.clone(), &ctx).await,
                Invocation::Resume { child, resolution } => {
                    tool.resume_child(*child, resolution, &ctx).await
                }
            }
        };
        let result = tokio::time::timeout(timeout, async {
            tokio::select! {
                biased;
                _ = ctx.cancel.cancelled() => None,
                // A panicking tool is a failed call, not a crashed run.
                result = AssertUnwindSafe(work).catch_unwind() => {
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
        let cancelled = || {
            (
                ReceiptStatus::Cancelled,
                Some("cancelled".to_string()),
                ToolOutput::error("Cancelled: the run was cancelled while this tool was running."),
                SourceTrust::Trusted,
            )
        };
        let (status, reason, output, source) = match result {
            // The owner's cancel wins over a pause that raced it.
            Ok(Some(Ok(Ok(output)))) if output.suspended.is_some() && ctx.cancel.is_cancelled() => {
                cancelled()
            }
            Ok(Some(Ok(Ok(output))))
                if output
                    .suspended
                    .as_ref()
                    .is_some_and(|child| !tool.suspends() || child.pending.is_none()) =>
            {
                let reason = format!(
                    "`{}` returned a paused run it cannot pause this run for",
                    call.name
                );
                (
                    ReceiptStatus::Failed,
                    Some(reason.clone()),
                    ToolOutput::error(format!("Error: {reason}")),
                    SourceTrust::Trusted,
                )
            }
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
            Ok(None) => cancelled(),
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
            let guard = self.guard(HookPoint::AfterTool);
            for hook in &self.agent.hooks {
                let screened = guard
                    .run(hook.as_ref(), hook.after_tool(&ctx, &receipt, output))
                    .await;
                if screened.is_none() {
                    *output = ToolOutput::error(format!(
                        "This result was withheld because the check `{}` gave no answer.",
                        hook.name()
                    ));
                }
            }
        }
        // A hook may have rewritten the result; the limit still holds.
        output.content = truncate_middle(&output.content, self.agent.limits.max_tool_output_chars);
        let mut receipt = receipt;
        self.scrubber().text(&mut output.content);
        self.scrubber().value(&mut output.details);
        self.scrubber().receipt(&mut receipt);
        self.state.messages.push(
            Message::tool_result(receipt.call_id.clone(), output.content.clone())
                .with_images(output.images.iter().cloned())
                .with_error(match receipt.status {
                    // A hook's answer is an error only when it says so.
                    ReceiptStatus::Answered => output.is_error,
                    _ => !receipt.succeeded(),
                }),
        );
        // Results are appended in call order, so the first finish kept is
        // the first in call order.
        if receipt.stage == InvocationStage::Dispatched && self.state.finishing.is_none() {
            if let Some(finish) = output.finish.clone() {
                self.state.finishing = Some(PendingFinish {
                    finish,
                    tool: receipt.tool.clone(),
                });
            }
        }
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
                let answer = self
                    .guard(HookPoint::OnStuck)
                    .run(hook.as_ref(), hook.on_stuck(&ctx, &signal))
                    .await
                    .flatten();
                if let Some(d) = answer {
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
                Some(Ok(StreamEvent::Switched(switch))) => {
                    self.emit(Event::ProviderSwitched { turn, switch }).await?;
                }
                Some(Ok(StreamEvent::Done(response))) => return Ok(ModelCall::Response(response)),
                Some(Err(err)) => return Ok(ModelCall::Failed(err)),
                None => {
                    return Ok(ModelCall::Failed(ProviderError::Other(
                        "provider stream ended without a final response".into(),
                    )))
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
        self.execute_batch(&in_flight.calls, 0, None, Vec::new(), None)
            .await
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
            ..Default::default()
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

    /// The narrowest authority any installed hook enforces.
    fn run_authority(&self) -> Option<Authority> {
        self.agent
            .hooks
            .iter()
            .filter_map(|hook| hook.authority())
            .reduce(|narrowed, next| narrowed.narrow(&next))
    }

    /// The names of the installed hooks, in order.
    /// The deterministic checks every call passes before a hook sees it:
    /// the types the schema declares (from the specs computed at build), then
    /// the tool's own `validate` (by default the full schema, required keys
    /// included).
    fn check_call(&self, tool: &dyn crate::Tool, call: &ToolCall) -> Result<(), String> {
        if let Some(spec) = self.agent.specs.iter().find(|s| s.name == call.name) {
            crate::tool::check_argument_types(&spec.parameters, &call.arguments)?;
        }
        tool.validate(&call.arguments)
    }

    fn scrubber(&self) -> crate::redaction::Scrubber<'_> {
        crate::redaction::Scrubber {
            hooks: &self.agent.hooks,
        }
    }

    fn approval_key(&self) -> [u8; 32] {
        self.agent
            .approval_key
            .unwrap_or_else(crate::binding::process_key)
    }

    /// The hooks an `Approve` of `pending` waives: the one that asked and
    /// those answered before it, when the saved call is still the one the
    /// digest names and the chain is the one that asked. Anything else
    /// (an edited or swapped call, another chain, state saved before
    /// digests) waives nothing: the call is judged from the first hook. An
    /// approval already used, in this process or (with a store) by a run
    /// that has moved on, is refused.
    async fn verify_approval(&self, pending: &Pending) -> Result<Vec<usize>, CoreError> {
        let (Some(hook_index), Some(given)) =
            (pending.hook_index, pending.interruption.digest.as_deref())
        else {
            return Ok(Vec::new());
        };
        let chain = self.hook_chain();
        let Some(call) = pending.calls.get(pending.next) else {
            return Ok(Vec::new());
        };
        if pending.hook_chain != chain || hook_index >= chain.len() {
            return Ok(Vec::new());
        }
        let mut waived = pending.waived.clone();
        waived.sort_unstable();
        waived.dedup();
        if waived.iter().any(|&i| i == hook_index || i >= chain.len()) {
            return Ok(Vec::new());
        }
        let question = crate::binding::Question {
            run_id: self.state.run_id,
            turn: self.state.turn,
            call,
            hook_index,
            hook_name: &chain[hook_index],
            waived: &waived,
        };
        let expected = crate::binding::digest(&self.approval_key(), &question);
        if !crate::binding::same(&expected, given) {
            return Ok(Vec::new());
        }
        if let Some(store) = &self.agent.store {
            if let Some(stored) = store.load_state(self.state.run_id).await? {
                let still_waiting = stored.pending.as_ref().is_some_and(|p| {
                    p.interruption
                        .digest
                        .as_deref()
                        .is_none_or(|d| crate::binding::same(d, given))
                });
                if !still_waiting {
                    return Err(CoreError::ApprovalAlreadyUsed);
                }
            }
        }
        if !crate::binding::consume(given) {
            return Err(CoreError::ApprovalAlreadyUsed);
        }
        waived.push(hook_index);
        Ok(waived)
    }

    fn hook_chain(&self) -> Vec<String> {
        self.agent
            .hooks
            .iter()
            .map(|hook| hook.name().to_string())
            .collect()
    }

    /// Containment for hooks called at `point` from here.
    fn guard(&self, point: HookPoint) -> HookGuard<'_> {
        // Being out of time is what `on_limit` is asked about, and at zero
        // left any other hook would time out before it could answer.
        let budget = self
            .remaining()
            .filter(|left| point != HookPoint::OnLimit && !left.is_zero());
        HookGuard {
            agent: self.agent,
            run_id: self.state.run_id,
            point,
            cancel: Some(&self.cancel),
            budget,
        }
    }

    /// The agent's limits, raised by what hooks granted this run.
    fn limits(&self) -> Limits {
        self.agent.limits.with_grants(&self.state.limit_grants)
    }

    /// Wall-clock budget left, if the run has one.
    fn remaining(&self) -> Option<Duration> {
        self.limits()
            .timeout
            .map(|timeout| timeout.saturating_sub(self.elapsed()))
    }

    /// Ask the hooks to raise `limit`. Returns whether one did.
    async fn extend_limit(&mut self, limit: LimitKind) -> Result<bool, CoreError> {
        if limit == LimitKind::Turns && self.limits().max_turns >= HARD_TURN_CAP {
            return Ok(false);
        }
        let mut granted = None;
        {
            let ctx = self.ctx();
            let guard = self.guard(HookPoint::OnLimit);
            for hook in &self.agent.hooks {
                let answer = guard.run(hook.as_ref(), hook.on_limit(&ctx, limit)).await;
                match answer {
                    Some(LimitDecision::Forbid) => {
                        guard.decided(hook.name(), "forbid", None).await;
                        return Ok(false);
                    }
                    Some(LimitDecision::Extend { amount }) if amount > 0 => {
                        guard.decided(hook.name(), "extend", None).await;
                        granted.get_or_insert(amount);
                    }
                    _ => {}
                }
            }
        }
        let Some(amount) = granted else {
            return Ok(false);
        };
        self.state.limit_grants.add(limit, amount);
        self.emit(Event::LimitExtended { limit, amount }).await?;
        Ok(true)
    }

    /// Let `after_model` hooks check the response. Returns why it was
    /// rejected, if it was, or why the run must stop here.
    async fn judge_response(&self, response: &mut ModelResponse) -> Result<Option<String>, String> {
        let ctx = self.ctx();
        let guard = self.guard(HookPoint::AfterModel);
        for hook in &self.agent.hooks {
            match guard
                .run(hook.as_ref(), hook.after_model(&ctx, response))
                .await
            {
                Some(ModelDecision::Accept) => {}
                Some(ModelDecision::Reject { message }) => {
                    guard.decided(hook.name(), "reject", Some(&message)).await;
                    return Ok(Some(message));
                }
                Some(ModelDecision::Stop { reason }) => {
                    guard.decided(hook.name(), "stop", Some(&reason)).await;
                    return Err(reason);
                }
                None => {
                    return Err(format!(
                        "the response check `{}` gave no answer",
                        hook.name()
                    ))
                }
            }
        }
        Ok(None)
    }

    /// The first answer a hook gives in the model's place, with its name.
    async fn answer_from_hooks(&self, request: &ModelRequest) -> Option<(String, ModelResponse)> {
        let ctx = self.ctx();
        let guard = self.guard(HookPoint::AnswerModel);
        for hook in &self.agent.hooks {
            let answer = guard
                .run(hook.as_ref(), hook.answer_model(&ctx, request))
                .await
                .flatten();
            if let Some(response) = answer {
                return Some((hook.name().to_string(), response));
            }
        }
        None
    }

    /// Count a continuation. Returns whether the run has room for it.
    fn record_continuation(&mut self) -> bool {
        let successes = self
            .state
            .run_receipts()
            .filter(|receipt| receipt.status == ReceiptStatus::Succeeded)
            .count();
        if successes > self.state.continued_at_successes {
            self.state.continued_at_successes = successes;
            self.state.idle_continuations = 0;
        }
        self.state.idle_continuations = self.state.idle_continuations.saturating_add(1);
        self.state.idle_continuations <= self.limits().max_idle_continuations
    }

    /// What the hooks want done about a failed model call, and which hook
    /// said so.
    async fn judge_model_error(&self, error: &ProviderError) -> (String, ModelErrorDecision) {
        let ctx = self.ctx();
        let guard = self.guard(HookPoint::OnModelError);
        let mut chosen = None;
        for hook in &self.agent.hooks {
            let answer = guard
                .run(hook.as_ref(), hook.on_model_error(&ctx, error))
                .await;
            match answer {
                None | Some(ModelErrorDecision::Fail) => {}
                Some(ModelErrorDecision::Abort) => {
                    guard.decided(hook.name(), "abort", None).await;
                    return (hook.name().to_string(), ModelErrorDecision::Abort);
                }
                Some(decision) => {
                    let kind = match decision {
                        ModelErrorDecision::Recover { .. } => "recover",
                        _ => "retry",
                    };
                    guard.decided(hook.name(), kind, None).await;
                    chosen.get_or_insert((hook.name().to_string(), decision));
                }
            }
        }
        chosen.unwrap_or_else(|| (String::new(), ModelErrorDecision::Fail))
    }

    /// Wait, but no longer than the wall-clock budget and not past a
    /// cancel. The guards at the top of the turn then see either.
    async fn pause(&self, wait: Duration) {
        let wait = self.remaining().map_or(wait, |left| wait.min(left));
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {}
            _ = tokio::time::sleep(wait) => {}
        }
    }

    /// Let `edit_transcript` hooks replace the transcript, one after another.
    async fn apply_transcript_edits(&mut self) -> Result<(), CoreError> {
        let turn = self.state.turn;
        let mut edited = false;
        for index in 0..self.agent.hooks.len() {
            let replacement = {
                let ctx = self.ctx();
                let hook = &self.agent.hooks[index];
                self.guard(HookPoint::EditTranscript)
                    .run(hook.as_ref(), hook.edit_transcript(&ctx))
                    .await
                    .flatten()
            };
            let Some(replacement) = replacement else {
                continue;
            };
            if replacement == self.state.messages {
                continue;
            }
            if let Some(reason) = transcript_defect(&replacement) {
                self.emit(Event::TranscriptEditRefused { turn, reason })
                    .await?;
                continue;
            }
            let before = self.state.messages.len();
            let after = replacement.len();
            self.state.messages = replacement;
            self.emit(Event::TranscriptEdited {
                turn,
                before,
                after,
            })
            .await?;
            edited = true;
        }
        if edited {
            self.checkpoint().await?;
        }
        Ok(())
    }

    /// Limits that stop a batch before its next call. `queued` approved
    /// read-only calls are waiting to run and count toward `max_tool_calls`.
    fn call_limit(&self, queued: usize) -> Option<LimitKind> {
        let limits = self.limits();
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
        let mut event = event;
        self.scrubber().json(&mut event);
        for sink in &self.agent.sinks {
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

    async fn finish_batch(mut self, end: BatchEnd) -> Result<RunOutcome, CoreError> {
        self.state.sync_declines();
        let (status, final_text) = match end {
            BatchEnd::Interrupted(interruption) => (RunStatus::Interrupted { interruption }, None),
            BatchEnd::Stuck(signal) => (RunStatus::Stuck { signal }, None),
            BatchEnd::Limit(limit) => (RunStatus::LimitReached { limit }, None),
            BatchEnd::Cancelled => (RunStatus::Cancelled, None),
            BatchEnd::Finished(PendingFinish { finish, tool }) => (
                RunStatus::Finished {
                    // A run in which the owner declined something never
                    // reports success, whatever the tool judged.
                    outcome: match finish.outcome {
                        crate::FinishOutcome::Succeeded if !self.state.declined.is_empty() => {
                            crate::FinishOutcome::Partial
                        }
                        outcome => outcome,
                    },
                    tool,
                },
                finish.reply,
            ),
            BatchEnd::Done => unreachable!("finish_batch is only called for terminal batch ends"),
        };
        self.finish(status, final_text).await
    }

    async fn finish(
        mut self,
        status: RunStatus,
        final_text: Option<String>,
    ) -> Result<RunOutcome, CoreError> {
        // A decline in a run sharing this one's (a child that finished
        // inside its last call) is part of this run's record: it never
        // ends `Succeeded`.
        self.state.sync_declines();
        // The run is over: whatever the batch was doing will not continue,
        // and a recovery must not mistake it for a run a crash cut short.
        self.state.in_flight = None;
        if self.own_declines {
            self.state.shared_declines = None;
        }
        self.checkpoint().await?;
        self.emit(Event::Finished {
            status: status.clone(),
            usage: self.state.usage,
            turns: self.state.turn,
        })
        .await?;
        let mut final_text = final_text;
        if let Some(text) = &mut final_text {
            self.scrubber().text(text);
        }
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
    // Not raced against the cancel: the run has ended, and what a hook
    // does here (writing memory) is worth finishing.
    let guard = HookGuard {
        agent,
        run_id: outcome.state.run_id,
        point: HookPoint::OnRunEnd,
        cancel: None,
        budget: None,
    };
    for hook in &agent.hooks {
        guard
            .run(hook.as_ref(), hook.on_run_end(&ctx, outcome))
            .await;
    }
}

/// Runs a hook so that a panic, a hang or a cancel costs one answer rather
/// than the run.
struct HookGuard<'a> {
    agent: &'a AgentInner,
    run_id: RunId,
    point: HookPoint,
    cancel: Option<&'a CancellationToken>,
    budget: Option<Duration>,
}

impl HookGuard<'_> {
    /// The hook's answer, or `None` if it gave none.
    async fn run<T>(&self, hook: &dyn Hook, call: impl Future<Output = T>) -> Option<T> {
        let allowed = hook.timeout().unwrap_or(self.agent.limits.hook_timeout);
        let allowed = self.budget.map_or(allowed, |left| allowed.min(left));
        let contained = tokio::time::timeout(allowed, AssertUnwindSafe(call).catch_unwind());
        let cancelled = async {
            match self.cancel {
                Some(cancel) => cancel.cancelled().await,
                None => std::future::pending().await,
            }
        };
        let failure = tokio::select! {
            biased;
            _ = cancelled => HookFailure::Cancelled,
            outcome = contained => match outcome {
                Ok(Ok(answer)) => return Some(answer),
                Ok(Err(panic)) => HookFailure::Panicked { message: panic_message(&*panic) },
                Err(_) => HookFailure::TimedOut { after_ms: allowed.as_millis() as u64 },
            },
        };
        self.report(hook.name(), failure).await;
        None
    }

    /// Record the failure. A store error here must not change what the run
    /// does about the hook; the next checkpoint reports it.
    async fn report(&self, hook: &str, failure: HookFailure) {
        self.record(Event::HookFailed {
            point: self.point,
            hook: hook.to_string(),
            failure,
        })
        .await;
    }

    /// Record a decision other than letting things proceed.
    async fn decided(&self, hook: &str, decision: &str, reason: Option<&str>) {
        self.record(Event::HookDecided {
            point: self.point,
            hook: hook.to_string(),
            decision: decision.to_string(),
            reason: reason.map(str::to_string),
        })
        .await;
    }

    async fn record(&self, event: Event) {
        let mut event = event;
        crate::redaction::Scrubber {
            hooks: &self.agent.hooks,
        }
        .json(&mut event);
        for sink in &self.agent.sinks {
            sink.emit(self.run_id, &event);
        }
        if let Some(store) = &self.agent.store {
            let _ = store.append_event(self.run_id, &event).await;
        }
    }
}

/// Why a hook's replacement transcript cannot be sent to a provider, if so.
/// Every call must be answered before the next assistant message, and every
/// result must answer a call.
fn transcript_defect(messages: &[Message]) -> Option<String> {
    if !messages.iter().any(|m| m.role == Role::User) {
        return Some("the transcript has no user message".into());
    }
    let mut open: Vec<&str> = Vec::new();
    for message in messages {
        match message.role {
            Role::Assistant => {
                if let Some(id) = open.first() {
                    return Some(format!("tool call `{id}` has no result"));
                }
                open = message.tool_calls.iter().map(|c| c.id.as_str()).collect();
            }
            Role::Tool => {
                let id = message.tool_call_id.as_deref().unwrap_or_default();
                let Some(position) = open.iter().position(|open| *open == id) else {
                    return Some(format!("tool result `{id}` answers no call"));
                };
                open.remove(position);
            }
            Role::System | Role::User => {}
        }
    }
    open.first()
        .map(|id| format!("tool call `{id}` has no result"))
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
