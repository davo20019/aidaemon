//! How a background helper's result reaches its chat: the result composed
//! from its run record, pushed when the run ends (or queued), and the one
//! answer for a group of runs a turn started together.

use std::sync::{Arc, Weak};

use aidaemon_channels::ports::OutboundRouter;
use aidaemon_core::SourceTrust;
use aidaemon_work::stores::NotificationStore as StateStore;
use futures::FutureExt as _;
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::card::settle_background_steps;
use crate::run_state::RunPhase;
use crate::run_state::timing::{background_timing, since_request_view};
use crate::spawn::ChildAgentRuntime;
use crate::spawn_tool::bounded_child_response;

/// A background helper's result for the chat, composed from its typed run
/// record around the helper's report: the outcome line (its phase, and why
/// it did not succeed), the report as the body, then what the record says
/// of the work the report alone vouches for, what reached the owner, and
/// the time the runtime measured. The helper's report is only the body:
/// its words never decide the outcome, the deliveries or the time, and the
/// text is never filtered. How it ended is said once: when its card shows
/// it (`on_card`), the result has no header of its own.
pub fn render_run_result(
    state: &crate::run_state::RunState,
    name: &str,
    report: &str,
    max_response_chars: usize,
    on_card: bool,
) -> String {
    // This goes straight to the chat, past the loop that renders truncation
    // metadata for tool results.
    let body = match bounded_child_response(report, max_response_chars) {
        (shown, Some(info)) => {
            format!(
                "{shown}\n\n{}",
                crate::turn::helpers::render_truncation_notice(&info)
            )
        }
        (shown, None) => shown.to_string(),
    };
    let mut parts: Vec<String> = Vec::new();
    let mut lead = if on_card {
        String::new()
    } else {
        format!("{}: {name}", state.view().headline())
    };
    // A header that already says the owner stopped it needs no reason.
    if let Some(reason) = state
        .reason()
        .filter(|_| on_card || !state.stopped_by_owner())
    {
        if !lead.is_empty() {
            lead.push('\n');
        }
        lead.push_str(&reason);
    }
    if !lead.is_empty() {
        parts.push(lead);
    }
    let has_body = !body.trim().is_empty();
    if has_body {
        parts.push(body.trim_end().to_string());
    }
    if state.reason().is_none()
        && let Some(evidence) = state.evidence.after_report(has_body)
    {
        parts.push(evidence);
    }
    if let Some(delivered) =
        crate::coordination::owner_delivery_lines(state.evidence.delivered.as_deref())
    {
        parts.push(delivered);
    }
    if let Some(footer) = background_timing(state) {
        parts.push(footer);
    }
    parts.join("\n\n")
}

/// The notification kind a run's phase is queued under.
pub fn run_notification_type(phase: RunPhase) -> &'static str {
    match phase {
        RunPhase::Succeeded => "completed",
        RunPhase::Stopped => "partial",
        RunPhase::Failed | RunPhase::Running => "failed",
    }
}

/// The plan steps a run's end settles: its phase, and for a run that
/// stopped because the owner declined one of its steps, that reason.
pub fn job_end(
    phase: RunPhase,
    failure: Option<&crate::run_state::RunFailure>,
) -> aidaemon_work::plans::JobEnd {
    match phase {
        RunPhase::Succeeded => aidaemon_work::plans::JobEnd::Succeeded,
        RunPhase::Stopped
            if matches!(failure, Some(crate::run_state::RunFailure::Declined { .. })) =>
        {
            aidaemon_work::plans::JobEnd::Declined
        }
        RunPhase::Stopped
            if matches!(
                failure,
                Some(crate::run_state::RunFailure::Cancelled {
                    by: crate::run_state::CancelledBy::Owner
                })
            ) =>
        {
            aidaemon_work::plans::JobEnd::StoppedByOwner
        }
        RunPhase::Stopped => aidaemon_work::plans::JobEnd::Stopped,
        RunPhase::Failed | RunPhase::Running => aidaemon_work::plans::JobEnd::Failed,
    }
}

/// Tell each chat which of its background sub-agents the last shutdown lost.
pub async fn report_interrupted_background_children(state: &Arc<dyn StateStore>) {
    let spawns = match state.take_background_spawns().await {
        Ok(spawns) => spawns,
        Err(error) => {
            warn!(%error, "Could not read interrupted background sub-agents");
            return;
        }
    };
    for spawn in spawns {
        let message = format!(
            "\u{26a0}\u{fe0f} A restart stopped this background job before it finished: {}\n\n\
             It may have done part of the work. Check before running it again.",
            spawn.mission
        );
        enqueue_background_notification(
            Some(state),
            "global",
            &spawn.session_id,
            "failed",
            &message,
            "interrupted background sub-agent notifier",
            true,
        )
        .await;
    }
}

/// Queue a background job's notification for later delivery.
#[allow(clippy::too_many_arguments)]
pub async fn enqueue_background_notification(
    state: Option<&Arc<dyn StateStore>>,
    goal_id: &str,
    session_id: &str,
    notification_type: &str,
    message: &str,
    context: &str,
    untrusted: bool,
) {
    let message = aidaemon_channels::presentation::present_notification(notification_type, message);
    let Some(state_store) = state else {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            "{context}: no queue configured; update dropped"
        );
        return;
    };
    let entry = aidaemon_work::goals::NotificationEntry::new(
        goal_id,
        session_id,
        notification_type,
        &message,
    )
    .with_untrusted(untrusted);
    if let Err(e) = state_store.enqueue_notification(&entry).await {
        warn!(
            session_id = %session_id,
            goal_id = %goal_id,
            notification_type = %notification_type,
            error = %e,
            "{context}: enqueue failed"
        );
    }
}

/// Where a background child's result goes: straight to the parent session
/// through the hub, else the queue.
pub struct BackgroundDelivery<'a> {
    pub agent: &'a dyn ChildAgentRuntime,
    pub hub: Option<&'a Weak<dyn OutboundRouter>>,
    pub state: Option<&'a Arc<dyn StateStore>>,
    pub session_id: &'a str,
    pub goal_id: &'a str,
}

impl BackgroundDelivery<'_> {
    pub async fn deliver(&self, notification_type: &str, message: &str, trust: SourceTrust) {
        // Without a hub the queue delivers it (and the heartbeat records
        // it in the parent session then), so it is recorded once.
        let direct = match self.hub {
            Some(hub) => {
                self.agent
                    .deliver_background_child_result(Some(hub), self.session_id, message, trust)
                    .await
            }
            None => Ok(false),
        };
        let delivered = match direct {
            Ok(sent) => {
                // The chat's record of the push: the daemon's log showed
                // nothing for a finished helper whose result had in fact
                // been sent (live 2026-11-13).
                if sent {
                    info!(
                        session_id = %self.session_id,
                        notification_type = %notification_type,
                        "Delivered a background helper's result to its chat"
                    );
                }
                sent
            }
            Err(e) => {
                warn!(
                    session_id = %self.session_id,
                    goal_id = %self.goal_id,
                    notification_type = %notification_type,
                    error = %e,
                    "spawn_agent background completion notifier: parent delivery failed"
                );
                false
            }
        };

        if !delivered {
            // The queued copy carries the child's taint into whichever
            // parent turn records it.
            enqueue_background_notification(
                self.state,
                self.goal_id,
                self.session_id,
                notification_type,
                message,
                "spawn_agent background completion notifier",
                trust == SourceTrust::Untrusted,
            )
            .await;
        }
    }
}

/// Where a background run's result goes when the run ends, and what it
/// settles there.
pub struct ResultRoute {
    pub agent: Arc<dyn ChildAgentRuntime>,
    pub runs: Arc<crate::run_state::RunStates>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub state: Option<Arc<dyn StateStore>>,
    pub plans: Option<Arc<aidaemon_work::plans::PlanStore>>,
    pub plan_session: Option<String>,
    pub goal_id: String,
    pub name: String,
    pub max_response_chars: usize,
    /// The turn that started the run, when one did: the run's end waits
    /// until it is over.
    pub turn: Option<aidaemon_channels::turns::TurnOver>,
    /// The chat turn's task, when the run is one of that turn's group.
    pub parent_task_id: Option<String>,
}

/// The subscriber on one run's terminal transition: when the run ends it
/// pushes the result, rendered from the run record, to the chat (or the
/// queue), then settles the plan steps the run was handed and redraws the
/// card. It waits on nothing but the transition: no turn and no inbound
/// message is involved.
pub async fn deliver_when_run_ends(
    mut ended: mpsc::UnboundedReceiver<crate::run_state::RunEnded>,
    route: ResultRoute,
) {
    let Some(ended) = ended.recv().await else {
        return;
    };
    // The turn that started it shows everything first (its reply, the
    // "Started in the background" ack, and the card it kept): the run's end
    // never lands above the turn's own reply.
    if let Some(turn) = route.turn.as_ref() {
        turn.wait().await;
    }
    let state = ended.state;
    // One of a group its turn started: the request's answer, a continuation
    // of that turn, does the turn's own steps and settles the card once it
    // is written ([`answer_group`]). The plan is known by this run's job
    // until its end settles its steps.
    let answer_follows = route.runs.answered_together(&state.id);
    let plan_id = if answer_follows {
        super::card::job_plan(
            route.plans.as_ref(),
            route.plan_session.as_deref(),
            &state.id,
        )
        .await
        .map(|plan| plan.id)
    } else {
        None
    };
    // The card settles to its final header first; the result follows,
    // without saying again how it ended.
    let on_card = settle_background_steps(
        &route.runs,
        route.plans.as_ref(),
        route.hub.as_ref(),
        route.plan_session.as_deref(),
        &state.chat,
        &state.id,
        job_end(state.phase, state.failure.as_ref()),
        answer_follows,
    )
    .await;
    let message = render_run_result(
        &state,
        &route.name,
        &ended.report.text,
        route.max_response_chars,
        on_card,
    );
    // One answer per request: a run alone is answered by its own result; the
    // runs a turn started together are answered once, when the last ends,
    // with all their results. Until then the card shows each one's end.
    let end = crate::run_state::GroupEnd {
        state: state.clone(),
        report: ended.report.clone(),
        message,
    };
    match route.runs.group_end(&state.id, end.clone()) {
        crate::run_state::GroupTurn::Alone => deliver_run_message(&route, &end).await,
        crate::run_state::GroupTurn::Waiting => {
            info!(
                session_id = %state.chat,
                job_id = %state.id,
                "A helper's result waits for the rest of its group"
            );
        }
        crate::run_state::GroupTurn::Complete(ends) => {
            answer_group(&route, ends, plan_id.as_deref()).await
        }
    }
}

/// Send one run's own result message to its chat (queued when it cannot be
/// sent now).
async fn deliver_run_message(route: &ResultRoute, end: &crate::run_state::GroupEnd) {
    let state = &end.state;
    let trust = if end.report.untrusted {
        SourceTrust::Untrusted
    } else {
        SourceTrust::Trusted
    };
    let notification_type = run_notification_type(state.phase);
    let delivery = BackgroundDelivery {
        agent: route.agent.as_ref(),
        hub: route.hub.as_ref(),
        state: route.state.as_ref(),
        session_id: &state.chat,
        goal_id: &route.goal_id,
    };
    let delivered =
        std::panic::AssertUnwindSafe(delivery.deliver(notification_type, &end.message, trust))
            .catch_unwind()
            .await;
    if delivered.is_err() {
        warn!(session_id = %state.chat, job_id = %state.id, "Delivering a helper's result panicked; queuing it");
        enqueue_background_notification(
            route.state.as_ref(),
            &route.goal_id,
            &state.chat,
            notification_type,
            &end.message,
            "spawn_agent background completion notifier",
            end.report.untrusted,
        )
        .await;
    }
}

/// The last run of a turn's group ended: the turn's request gets its one
/// answer, written by the parent from every run's typed result
/// ([`group_followup`]). When no answer can be written, each run's own
/// result goes to the chat instead, so nothing is lost.
async fn answer_group(
    route: &ResultRoute,
    ends: Vec<crate::run_state::GroupEnd>,
    plan_id: Option<&str>,
) {
    let Some(chat) = ends.first().map(|end| end.state.chat.clone()) else {
        return;
    };
    let jobs: Vec<String> = ends.iter().map(|end| end.state.id.clone()).collect();
    let answered = route
        .agent
        .answer_after_runs(crate::spawn::RunsAnswer {
            hub: route.hub.clone(),
            chat: chat.clone(),
            parent_task_id: route.parent_task_id.clone(),
            followup: group_followup(&ends, route.max_response_chars),
            clock: crate::run_state::RequestClock::of_group(&ends),
        })
        .await;
    // The request's turn is over, answered or not: the card settles from
    // the plan as the answer left it (its own steps done by the answer's
    // turn), above the answer or the results that replace it.
    super::card::settle_answered_request(
        &route.runs,
        route.plans.as_ref(),
        route.hub.as_ref(),
        &chat,
        plan_id,
        &jobs,
    )
    .await;
    match answered {
        Ok(true) => {
            info!(session_id = %chat, runs = ends.len(), "Answered a request once its helpers all ended");
        }
        Ok(false) | Err(_) => {
            if let Err(error) = answered {
                warn!(session_id = %chat, %error, "Could not answer a request from its helpers' results; sending each result");
            }
            for end in &ends {
                deliver_run_message(route, end).await;
            }
        }
    }
}

/// The continuation a turn's group of runs gets once the last of them
/// ended: each run's name, how it ended and its one time set from its
/// record ([`crate::run_state::RunTimes::line`], the line its card shows),
/// then its report (told never to state a time: the record is the only
/// source of one). The parent answers
/// the owner's request once from this, and the record's labelled times are
/// the only durations it sees (live 2026-11-13: the answer said "its agent
/// took 3m 55s" and "stopped after 2m", work-start figures, under a card
/// that said "Done 5m 54s after your request"). The card already shows each
/// run's end and the group's total, so the answer is told not to list them
/// again; a request that asked how long gets these lines as given.
pub fn group_followup(ends: &[crate::run_state::GroupEnd], max_report_chars: usize) -> String {
    let mut out = String::from(
        "[Background helpers finished]\n\
         Every helper this request started has ended. Their run records follow \
         (authoritative: how each ended and its times are the daemon's measurements).\n",
    );
    for (index, end) in ends.iter().enumerate() {
        let state = &end.state;
        let ended = match state.phase {
            RunPhase::Succeeded => "finished",
            RunPhase::Stopped => "didn't finish",
            RunPhase::Failed => "failed",
            RunPhase::Running => "still running",
        };
        out.push_str(&format!(
            "\n{}. {} ({ended})\n",
            index + 1,
            state.label.trim()
        ));
        if let Some(times) = state.times() {
            out.push_str(&format!("Times: {}\n", times.line()));
        }
        if let Some(line) = state.line() {
            out.push_str(&line);
            out.push('\n');
        }
        if let Some(lines) =
            crate::coordination::delivery_lines(state.evidence.delivered.as_deref())
        {
            out.push_str(&lines);
            out.push('\n');
        }
        let (report, _) = bounded_child_response(end.report.text.trim(), max_report_chars);
        let report = if end.report.untrusted {
            aidaemon_tools::sanitize::wrap_untrusted_output("spawn_agent", report)
        } else {
            report.to_string()
        };
        out.push_str(&format!("Its report:\n{}\n", report.trim()));
    }
    let last = ends
        .iter()
        .filter_map(|end| end.state.times()?.after_request())
        .max_by_key(|span| span.to);
    if let (true, Some(last)) = (ends.len() > 1, last) {
        out.push_str(&format!(
            "\nThe last of them ended {}.\n",
            since_request_view(last.from, last.to).words
        ));
    }
    out.push_str(
        "\nAnswer the owner's original request now, once, from these results. The chat's card \
         already shows how each helper ended and its time, and each helper already sent what it \
         delivered: don't list them again line by line or resend anything. What reached the \
         owner is each record's \"Delivered to the owner\" list, the only record of it: if you \
         mention what was sent, say exactly that (a helper whose list says nothing sent \
         nothing), never what a report claims it sent. The \"Times\" lines \
         are the only durations there are: \"took\" is a helper's own work from its start to its \
         end, \"ended … after your request\" is when a helper ended, counted from the owner's \
         request. If you mention a time, quote one of these figures with its own words \
         (\"took …\" or \"ended … after your request\"); never call one by the other's name and \
         never work a time out yourself. When the answer's own time adds to these, the daemon \
         measures it and shows it under your answer as \"Answer sent … after your request\": \
         never say it wasn't measured, and don't state one of your own. A record's \
         \"Next step\" is the daemon's offer: make it in your answer, as given. Don't call tools \
         to re-check this work.",
    );
    out
}
