//! A background helper's card: the card its starting turn kept, redrawn
//! from the run records while the helper works and settled when it ends.

use std::sync::{Arc, Weak};
use std::time::Duration;

use aidaemon_channels::ports::OutboundRouter;
use aidaemon_tools::status::StatusUpdate;
use tokio::sync::mpsc;
use tracing::warn;

use crate::run_state::RunPhase;

/// A kept card, drawn from the records: the plan (as it renders itself) and
/// each run drawn on that card, running or ended, by the same renderer as
/// the live card.
pub fn kept_card(
    plan: Option<&aidaemon_work::plans::TaskPlan>,
    runs: &[crate::run_state::RunState],
) -> Option<String> {
    let plan = plan.map(aidaemon_work::plans::TaskPlan::card);
    aidaemon_channels::live_status::render_card(&aidaemon_channels::live_status::CardView {
        plan: plan.as_ref(),
        helpers: runs.iter().map(|run| (run.view(), None)).collect(),
        ..Default::default()
    })
}

/// The runs drawn on card `card` in `chat`: each run whose own card it is
/// (a run still waiting to learn its card is resolved first).
pub async fn runs_on_card(
    hub: &dyn OutboundRouter,
    runs: &crate::run_state::RunStates,
    chat: &str,
    card: &str,
) -> Vec<crate::run_state::RunState> {
    for run in runs.for_chat(chat) {
        if matches!(run.card, crate::run_state::RunCard::Pending { .. }) {
            run_card(hub, runs, &run.id, chat).await;
        }
    }
    runs.on_card(chat, card)
}

/// One writer at a time for a chat's kept card. A helper's step redraw and
/// the redraw its end makes take turns, and a redraw that starts after the
/// helper ended draws nothing: a step edit still in flight (a slow or
/// rate-limited Telegram edit) can never land after the finished card and
/// leave a stale step on it (live 2026-11-13: the card stayed on "running a
/// command: 13" after the count ended).
pub fn card_writer(chat: &str) -> Arc<tokio::sync::Mutex<()>> {
    static WRITERS: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    > = std::sync::LazyLock::new(Default::default);
    let mut writers = WRITERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    writers.retain(|_, writer| Arc::strong_count(writer) > 1);
    writers.entry(chat.to_string()).or_default().clone()
}

/// The card a background helper's turn leaves in the chat, for showing
/// the helper's steps after that turn ended.
pub struct HelperCard {
    pub runs: Arc<crate::run_state::RunStates>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub plans: Option<Arc<aidaemon_work::plans::PlanStore>>,
    pub plan_session: Option<String>,
    pub chat: String,
    pub job: String,
    pub label: String,
    /// The chat's kept card when the helper started: one that differs
    /// later is the card its own turn left.
    pub at_start: Option<String>,
    /// What this helper last drew, so an unchanged card is not sent again.
    pub drawn: std::sync::Mutex<Option<String>>,
}

/// How often a helper's steps may edit its card after its turn.
pub const HELPER_CARD_EVERY: Duration = Duration::from_secs(3);

impl HelperCard {
    /// Redraw the card the helper's turn left (its plan, when it has one,
    /// and each helper's record), when that turn has ended and left one.
    /// While the turn runs, its own live card shows the steps.
    pub async fn refresh(&self) {
        let writer = card_writer(&self.chat);
        let _turn = writer.lock().await;
        // Ended meanwhile: its end drew the finished card.
        if self
            .runs
            .get(&self.job)
            .is_none_or(|run| run.phase != RunPhase::Running)
        {
            return;
        }
        let Some(hub) = self.hub.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        if self
            .plan_session
            .as_deref()
            .is_some_and(|plan_session| plan_session != self.chat)
        {
            return;
        }
        let now = hub.checklist_surface(&self.chat).await;
        if now == self.at_start {
            return;
        }
        let plan = job_plan(self.plans.as_ref(), self.plan_session.as_deref(), &self.job).await;
        // Without a plan the card is the helpers' own: the one the turn that
        // started this helper kept, never a later turn's.
        let surface = match run_card(hub.as_ref(), &self.runs, &self.job, &self.chat).await {
            Some(card) => card,
            None => match (&plan, now) {
                (Some(_), Some(card)) => card,
                _ => return,
            },
        };
        let on_card = runs_on_card(hub.as_ref(), &self.runs, &self.chat, &surface).await;
        let Some(card) = kept_card(plan.as_ref(), &on_card) else {
            return;
        };
        let unchanged = self
            .drawn
            .lock()
            .is_ok_and(|drawn| drawn.as_deref() == Some(card.as_str()));
        if unchanged {
            return;
        }
        match hub.edit_text(&self.chat, &surface, &card).await {
            Ok(true) => {
                if let Ok(mut drawn) = self.drawn.lock() {
                    *drawn = Some(card);
                }
            }
            Ok(false) => {}
            Err(error) => warn!(
                session_id = %self.chat,
                job_id = %self.job,
                %error,
                "Could not show a background helper's step on its card"
            ),
        }
    }
}

/// The recent plan of `plan_session` that background job `job` works on.
async fn job_plan(
    plans: Option<&Arc<aidaemon_work::plans::PlanStore>>,
    plan_session: Option<&str>,
    job: &str,
) -> Option<aidaemon_work::plans::TaskPlan> {
    plans?
        .get_recent_for_session(plan_session?, 5)
        .await
        .ok()?
        .into_iter()
        .find(|plan| {
            plan.steps.iter().any(|step| {
                step.background_jobs
                    .iter()
                    .chain(&step.waited_jobs)
                    .any(|known| known == job)
            })
        })
}

/// A waited-on helper's status channel: its own steps go on its run record,
/// and its line on the card is drawn from that record ([`RunState::view`]),
/// as a background helper's is. Every other update passes through.
///
/// [`RunState::view`]: crate::run_state::RunState::view
pub fn record_waited_steps(
    parent: Option<mpsc::Sender<StatusUpdate>>,
    runs: Arc<crate::run_state::RunStates>,
    id: String,
    label: String,
) -> Option<mpsc::Sender<StatusUpdate>> {
    let parent = parent?;
    let (tx, mut rx) = mpsc::channel::<StatusUpdate>(64);
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            let update = match update {
                StatusUpdate::Helper(view)
                    if view
                        .run
                        .as_deref()
                        .map_or(view.label == label, |run| run == id)
                        && !view.phase.is_terminal() =>
                {
                    if let Some(step) = view.step.clone() {
                        runs.step(&id, step);
                    }
                    match runs.get(&id) {
                        Some(run) if !run.phase.is_terminal() => StatusUpdate::Helper(run.view()),
                        _ => continue,
                    }
                }
                other => other,
            };
            if parent.send(update).await.is_err() {
                break;
            }
        }
    });
    Some(tx)
}

/// A background helper's status channel: every update goes on to its
/// parent's, and its own steps are also noted for its line on the card and
/// shown there once its turn has ended (live 2026-11-13: the turn's live
/// card stopped with the turn, and a 60 s count showed nothing until it
/// ended).
pub fn show_helper_steps_on_card(
    parent: Option<mpsc::Sender<StatusUpdate>>,
    card: HelperCard,
) -> Option<mpsc::Sender<StatusUpdate>> {
    let parent = parent?;
    let card = Arc::new(card);
    let (tx, mut rx) = mpsc::channel::<StatusUpdate>(64);
    tokio::spawn(async move {
        let mut redraw = tokio::time::interval(HELPER_CARD_EVERY);
        redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut changed = false;
        // The redraw runs beside the updates: a slow edit never holds up
        // the helper's steps, and the next redraw shows the latest one.
        let mut drawing: Option<tokio::task::JoinHandle<()>> = None;
        loop {
            tokio::select! {
                update = rx.recv() => {
                    let Some(update) = update else { break };
                    // The helper's own view goes on as its record: the live
                    // card shows what the kept card will.
                    let update = match update {
                        StatusUpdate::Helper(view)
                            if view.label == card.label && !view.phase.is_terminal() =>
                        {
                            if let Some(step) = view.step.clone() {
                                card.runs.step(&card.job, step);
                                changed = true;
                            }
                            match card.runs.get(&card.job) {
                                Some(run) if !run.phase.is_terminal() => {
                                    StatusUpdate::Helper(run.view())
                                }
                                _ => continue,
                            }
                        }
                        other => other,
                    };
                    let _ = parent.send(update).await;
                }
                _ = redraw.tick(), if changed => {
                    if drawing.as_ref().is_some_and(|draw| !draw.is_finished()) {
                        continue;
                    }
                    changed = false;
                    let card = card.clone();
                    drawing = Some(tokio::spawn(async move { card.refresh().await }));
                }
            }
        }
    });
    Some(tx)
}

/// A background run ended: settle the checklist steps it was given, then
/// redraw its own card (the one its starting turn kept) from the records:
/// the plan and every run drawn on that card. Once nothing on the card is
/// still running, the plan's steps still open settle as not done (nothing
/// is left to do them) and the card settles to its final state, by the same
/// rule as a turn's card ([`aidaemon_core::card::CardEnd::of`]); the
/// records of its runs are then let go. A run whose card is unknown edits
/// the chat's plan card (or sends it fresh) when it had a plan. `true` when
/// a card now shows how the run ended; the result message follows on its
/// own and never says it again.
pub async fn settle_background_steps(
    runs: &crate::run_state::RunStates,
    plan_store: Option<&Arc<aidaemon_work::plans::PlanStore>>,
    hub: Option<&Weak<dyn OutboundRouter>>,
    plan_session: Option<&str>,
    chat_session: &str,
    job_id: &str,
    end: aidaemon_work::plans::JobEnd,
) -> bool {
    // Its run record already says it ended, so a step redraw that starts
    // now draws nothing; this waits for any still in flight.
    let writer = card_writer(chat_session);
    let _turn = writer.lock().await;
    let plan_id = job_plan(plan_store, plan_session, job_id)
        .await
        .map(|plan| plan.id);
    let settled = match (plan_store, plan_session) {
        (Some(plans), Some(plan_session)) => {
            // The folders its work made, on the steps it took on, first:
            // the card then names the real ones.
            let made = runs
                .get(job_id)
                .map(|run| run.made_folders)
                .unwrap_or_default();
            if let Err(error) = plans.note_job_folders(plan_session, job_id, &made).await {
                warn!(%error, job_id, "Could not record the folders a background job made");
            }
            match plans.settle_background_job(plan_session, job_id, end).await {
                Ok(plan) => plan,
                Err(error) => {
                    warn!(%error, job_id, "Could not settle a background job's checklist steps");
                    None
                }
            }
        }
        _ => None,
    };
    let plan_id = plan_id.or_else(|| settled.as_ref().map(|plan| plan.id.clone()));
    // A sub-agent's own checklist is not the card in the chat.
    if plan_session.is_some_and(|plan_session| plan_session != chat_session) {
        runs.forget(job_id);
        return false;
    }
    let Some(hub) = hub.and_then(Weak::upgrade) else {
        runs.forget(job_id);
        return false;
    };
    let Some(surface) = run_card(hub.as_ref(), runs, job_id, chat_session).await else {
        // Its turn kept no card it can be told by: the plan's card shows
        // the plan and this run, when that card is still the chat's card
        // from before the run began. A card some later turn kept belongs to
        // that turn and its own runs: drawing this run over it would hide
        // them (live 2026-11-13: a shell loop's card was overwritten with
        // "codex coding agent … Done 9m 40s" and stayed wrong). Its result
        // then says how it ended itself.
        let this: Vec<_> = runs.get(job_id).into_iter().collect();
        let began_on = match this.first().map(|run| &run.card) {
            Some(crate::run_state::RunCard::Pending { before }) => before.clone(),
            _ => None,
        };
        runs.forget(job_id);
        let current = hub.checklist_surface(chat_session).await;
        let still_its_card = current.is_some()
            && current == began_on
            && current
                .as_deref()
                .is_some_and(|card| runs.on_card(chat_session, card).is_empty());
        return match kept_card(settled.as_ref(), &this) {
            Some(card) if settled.is_some() && still_its_card => {
                update_checklist_card(hub.as_ref(), chat_session, &card).await
            }
            _ => false,
        };
    };
    let on_card = runs_on_card(hub.as_ref(), runs, chat_session, &surface).await;
    let all_ended = on_card.iter().all(|run| run.phase.is_terminal());
    let mut plan = settled;
    if let (Some(plans), Some(plan_id)) = (plan_store, plan_id.as_deref()) {
        let latest = if all_ended {
            plans.settle_open_steps(plan_id).await
        } else {
            plans.get(plan_id).await
        };
        match latest {
            Ok(Some(latest)) => plan = Some(latest),
            Ok(None) => {}
            Err(error) => warn!(%error, job_id, "Could not read a background run's plan"),
        }
    }
    let shown = match kept_card(plan.as_ref(), &on_card) {
        Some(card) => match hub.edit_text(chat_session, &surface, &card).await {
            Ok(edited) => edited,
            Err(error) => {
                warn!(session_id = %chat_session, %error, "Could not settle a run's card");
                false
            }
        },
        None => false,
    };
    if all_ended {
        for run in &on_card {
            runs.forget(&run.id);
        }
    }
    shown
}

/// The card run `job` is drawn on: the one its run record names, else the
/// first card the chat kept after the run began (the card its starting turn
/// kept), which is then recorded on the run. `None` while that turn has not
/// kept one, or once the chat's card history no longer tells which it was.
pub async fn run_card(
    hub: &dyn OutboundRouter,
    runs: &crate::run_state::RunStates,
    job: &str,
    chat: &str,
) -> Option<String> {
    match runs.get(job)?.card {
        crate::run_state::RunCard::Kept(card) => Some(card),
        crate::run_state::RunCard::Pending { before } => {
            let card = hub.checklist_surface_after(chat, before.as_deref()).await?;
            runs.keep_card(job, &card);
            Some(card)
        }
    }
}

/// Show `card` in the session's chat: edit the plan card the turn left in
/// place, else send it as a new card. `true` when the chat shows it.
pub async fn update_checklist_card(hub: &dyn OutboundRouter, session_id: &str, card: &str) -> bool {
    if let Some(id) = hub.checklist_surface(session_id).await
        && matches!(hub.edit_text(session_id, &id, card).await, Ok(true))
    {
        return true;
    }
    match hub.send_text_tracked(session_id, card).await {
        Ok(Some(id)) => {
            hub.register_checklist_surface(session_id, &id).await;
            true
        }
        Ok(None) => true,
        Err(error) => {
            warn!(session_id, %error, "Could not show the updated checklist");
            false
        }
    }
}
