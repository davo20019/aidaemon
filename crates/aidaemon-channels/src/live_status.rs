//! Single self-editing status surface per turn. One message is created lazily on
//! the first progress event and edited in place for every subsequent update, so a
//! task shows one calm, updating line/checklist instead of many messages.
//!
//! Every card, the live one during a turn and the one a turn leaves behind,
//! is drawn by [`render_card`] from one view: the plan's card, each helper's
//! record ([`HelperView`]) and the turn's own typed [`Step`]. How a card ends
//! with its turn is decided once, from that view ([`CardView::end`]): every
//! channel path keeps or collapses the same card the same way.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::card::{CardEnd, HelperView, PlanCard, RunPhase, Step};
use aidaemon_tools::status::StatusUpdate;

use crate::hub::ChannelHub;

/// Whether a plan card (a rendered checklist) shows every step finished.
/// Such a card drops the turn's own step line: the work is done.
pub fn is_completed_checklist(rendered: &str) -> bool {
    rendered.trim_start().starts_with('✅')
}

/// One card's state, as every surface draws it.
#[derive(Debug, Clone, Default)]
pub struct CardView<'a> {
    /// The plan's card, as the plan renders itself.
    pub plan: Option<&'a PlanCard>,
    /// Each helper's record, in the order they started, with how long its
    /// latest step has run when that is known.
    pub helpers: Vec<(HelperView, Option<Duration>)>,
    /// The turn's own current step and how long it has run.
    pub activity: Option<(&'a Step, Duration)>,
    /// The turn finished itself work a failed helper left: its measured
    /// time.
    pub own_work: Option<&'a str>,
}

impl CardView<'_> {
    /// How this card ends with its turn ([`CardEnd::of`]).
    pub fn end(&self) -> CardEnd {
        CardEnd::of(self.plan, self.helpers.len())
    }
}

/// The card for `view`: the plan (when there is one), each helper's header
/// and its latest step while it runs (its measured time once it ended), and
/// the turn's own step, each long step with how long it has run. A plan
/// never hides the helpers or the step running now; a finished plan drops
/// the turn's own step.
pub fn render_card(view: &CardView<'_>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(plan) = view.plan {
        parts.push(plan_with_run_header(plan.text.trim_end(), &view.helpers));
    }
    for (helper, elapsed) in &view.helpers {
        parts.push(helper_block(helper, *elapsed));
    }
    if let Some(total) = since_request_total(&view.helpers) {
        parts.push(total);
    }
    if let Some(took) = view.own_work {
        parts.push(format!(
            "✅ Finished here after the helper failed\n{}",
            took.trim()
        ));
    }
    let done = view
        .plan
        .is_some_and(|plan| is_completed_checklist(&plan.text));
    if !done {
        if let Some((step, elapsed)) = view.activity.filter(|(step, _)| step.is_shown()) {
            parts.push(format!("⏳ {}", line_with_elapsed(&step.words(), elapsed)));
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// The plan's card under a header that agrees with its helpers' run
/// records. A plan whose steps all read done, on a card where a helper
/// ended without doing its work, is not "Done": the header says how that
/// helper ended instead (live 2026-11-13: "Done · 3 of 3 steps" above "A
/// helper didn't finish"). The plan's own step marks stay as its records
/// set them.
fn plan_with_run_header(plan: &str, helpers: &[(HelperView, Option<Duration>)]) -> String {
    let worst = helpers
        .iter()
        .map(|(helper, _)| helper.phase)
        .filter(|phase| matches!(phase, RunPhase::Failed | RunPhase::Stopped))
        .max_by_key(|phase| matches!(phase, RunPhase::Failed));
    let Some(worst) = worst.filter(|_| is_completed_checklist(plan)) else {
        return plan.to_string();
    };
    let (header, body) = plan.split_once('\n').unwrap_or((plan, ""));
    let headline = worst.helper_headline(false);
    let header = match header.split_once(" · ") {
        // "✅ Done · 3 of 3 steps": the count stays, the verdict is the run's.
        Some((_, count)) => format!("{headline} · {count}"),
        // A one-step plan is its step: only its mark changes.
        None => {
            let mark = headline.split(' ').next().unwrap_or_default();
            format!("{mark}{}", header.trim_start_matches('✅'))
        }
    };
    if body.is_empty() {
        header
    } else {
        format!("{header}\n{body}")
    }
}

/// With several helpers on the card, once every one ended: how long after
/// the owner's request the last of them ended, from their run records.
/// One helper's own line already says it.
fn since_request_total(helpers: &[(HelperView, Option<Duration>)]) -> Option<String> {
    if helpers.len() < 2
        || helpers
            .iter()
            .any(|(helper, _)| !helper.phase.is_terminal())
    {
        return None;
    }
    let last = helpers
        .iter()
        .map(|(helper, _)| helper.since_request.as_ref())
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max_by_key(|since| since.secs)?;
    let all_done = helpers
        .iter()
        .all(|(helper, _)| helper.phase == RunPhase::Succeeded);
    Some(if all_done {
        format!("All {} done {}.", helpers.len(), last.words.trim())
    } else {
        format!("The last of them ended {}.", last.words.trim())
    })
}

/// Where a helper stands, in a few plain words: its latest step while it
/// runs, else how it ended. For one-line replies that list helpers.
pub fn helper_status_words(helper: &HelperView) -> String {
    match (helper.phase, helper.step.as_ref()) {
        (RunPhase::Running, Some(step)) if step.is_shown() => step.words(),
        (RunPhase::Running, Some(_)) => "Working".to_string(),
        (RunPhase::Running, None) => "Getting started".to_string(),
        (RunPhase::Succeeded, _) => "Finished".to_string(),
        (RunPhase::Stopped, _) => "Didn't finish".to_string(),
        (RunPhase::Failed, _) => "Failed".to_string(),
    }
}

/// A helper's lines on a card: its header, then its latest step while it
/// runs, or how long it took once it ended.
fn helper_block(helper: &HelperView, elapsed: Option<Duration>) -> String {
    let mut block = format!(
        "{}: {}",
        helper.phase.helper_headline(helper.background),
        helper.label.trim()
    );
    match (helper.phase, helper.step.as_ref(), helper.took.as_deref()) {
        (RunPhase::Running, Some(step), _) if step.is_shown() => {
            block.push('\n');
            block.push_str(&line_with_elapsed(
                &step.words(),
                elapsed.unwrap_or_default(),
            ));
        }
        (phase, _, Some(took)) if phase.is_terminal() => {
            block.push('\n');
            block.push_str(took.trim());
        }
        _ => {}
    }
    if let (true, Some(since)) = (helper.phase.is_terminal(), helper.since_request.as_ref()) {
        let verb = if helper.phase == RunPhase::Succeeded {
            "Done"
        } else {
            "Ended"
        };
        block.push_str(if helper.took.is_some() { " " } else { "\n" });
        block.push_str(&format!("{verb} {}.", since.words.trim()));
    }
    block
}

/// Abstraction over the chat surface so `LiveStatus` is testable without a bot.
#[async_trait::async_trait]
pub trait SurfaceSink: Send + Sync {
    /// Create a new surface message; returns its channel message id (if any).
    async fn create(&self, text: &str) -> anyhow::Result<Option<String>>;
    /// Edit an existing surface message. `Ok(true)` edited, `Ok(false)` can't edit.
    async fn edit(&self, message_id: &str, text: &str) -> anyhow::Result<bool>;
    /// Delete an existing surface message. `Ok(true)` deleted, `Ok(false)` can't delete.
    async fn delete(&self, message_id: &str) -> anyhow::Result<bool>;
}

/// Per-turn live-status state. Shared as `Arc<tokio::sync::Mutex<LiveStatus>>`
/// between the status-consumer task and the turn runner, which owns how the
/// card ends.
pub struct LiveStatus {
    message_id: Option<String>,
    checklist: Option<PlanCard>,
    /// The turn's own current step and when it began.
    activity: Option<(Step, std::time::Instant)>,
    /// Each helper's record, in the order they started, and when its
    /// latest step began.
    helpers: Vec<(HelperView, std::time::Instant)>,
    /// The measured time of work the turn finished itself after a helper
    /// failed.
    own_work: Option<String>,
    /// What the surface shows now, so an unchanged card is never re-sent.
    shown: Option<String>,
    edit_supported: bool,
}

/// How long a step must have run before its line says for how long.
const ELAPSED_SHOWN_AFTER: Duration = Duration::from_secs(30);

/// `line` with how long its step has run, in whole minutes once it passes
/// [`ELAPSED_SHOWN_AFTER`] ("Running a command · 3 min so far"), so a long
/// step reads as alive and the card changes at most once a minute.
pub fn line_with_elapsed(line: &str, elapsed: Duration) -> String {
    if elapsed < ELAPSED_SHOWN_AFTER {
        return line.to_string();
    }
    let minutes = elapsed.as_secs() / 60;
    if minutes == 0 {
        format!("{line} · under a minute so far")
    } else {
        format!("{line} · {minutes} min so far")
    }
}

impl LiveStatus {
    pub fn new() -> Self {
        Self {
            message_id: None,
            checklist: None,
            activity: None,
            helpers: Vec::new(),
            own_work: None,
            shown: None,
            edit_supported: true,
        }
    }

    /// The card as it is now.
    fn view(&self, with_activity: bool) -> CardView<'_> {
        CardView {
            plan: self.checklist.as_ref(),
            helpers: self
                .helpers
                .iter()
                .map(|(helper, since)| (helper.clone(), Some(since.elapsed())))
                .collect(),
            activity: with_activity
                .then_some(self.activity.as_ref())
                .flatten()
                .map(|(step, since)| (step, since.elapsed())),
            own_work: self.own_work.as_deref(),
        }
    }

    fn render(&self) -> Option<String> {
        render_card(&self.view(true))
    }

    /// What the card would show now.
    pub fn preview(&self) -> Option<String> {
        self.render()
    }

    /// How this card ends with its turn: the one rule ([`CardView::end`]).
    pub fn end(&self) -> CardEnd {
        self.view(false).end()
    }

    /// Note one of the turn's status updates on the card; `true` when the
    /// card should show it at once: the plan, a handoff and a helper's
    /// start or end are never held back. Each step is its typed kind; a
    /// tool ending is not a step, and between steps (the model choosing
    /// the next one) the card keeps the latest step while the working
    /// indicator says the turn is busy.
    pub fn note(&mut self, update: &StatusUpdate) -> bool {
        match update {
            StatusUpdate::Checklist(plan) => {
                self.note_checklist(plan.clone());
                true
            }
            StatusUpdate::ToolStart { kind, .. } => {
                self.note_activity(Step::Doing(*kind));
                false
            }
            StatusUpdate::StepHandedOff { kind } => {
                self.note_activity(Step::HandedOff(*kind));
                true
            }
            StatusUpdate::Helper(helper) => self.note_helper(helper.clone()),
            StatusUpdate::OwnWork { took } => {
                self.own_work = Some(took.clone());
                true
            }
            _ => false,
        }
    }

    /// Note the plan card without sending it.
    pub fn note_checklist(&mut self, plan: PlanCard) {
        self.checklist = Some(plan);
    }

    /// Note the turn's current step without sending it; the same step
    /// keeps its start time.
    pub fn note_activity(&mut self, step: Step) {
        if self.activity.as_ref().map(|(shown, _)| shown) != Some(&step) {
            self.activity = Some((step, std::time::Instant::now()));
        }
    }

    /// Note a helper's record: one entry per helper, its latest step
    /// keeping its start time while it stays the same. `true` when its
    /// header changed (it is new, or it ended): a step alone can wait for
    /// the next refresh.
    pub fn note_helper(&mut self, helper: HelperView) -> bool {
        match self
            .helpers
            .iter_mut()
            .find(|(known, _)| known.label == helper.label)
        {
            Some((known, _)) if *known == helper => false,
            Some((known, since)) => {
                // A late step never undoes how the helper ended.
                if known.phase.is_terminal() && !helper.phase.is_terminal() {
                    return false;
                }
                let header_changed = known.phase != helper.phase || known.took != helper.took;
                if known.step != helper.step {
                    *since = std::time::Instant::now();
                }
                *known = helper;
                header_changed
            }
            None => {
                self.helpers.push((helper, std::time::Instant::now()));
                true
            }
        }
    }

    /// Send the card when what it would show changed since it was last
    /// sent (a step's elapsed time included); `true` when it was sent.
    pub async fn flush_if_changed(&mut self, sink: &dyn SurfaceSink) -> bool {
        if self.render().is_none() || self.render() == self.shown {
            return false;
        }
        self.flush(sink).await;
        true
    }

    async fn flush(&mut self, sink: &dyn SurfaceSink) {
        let Some(body) = self.render() else { return };
        self.shown = Some(body.clone());
        let body = crate::presentation::render_internal_markers(&body);
        match (&self.message_id, self.edit_supported) {
            (Some(id), true) => match sink.edit(id, &body).await {
                Ok(true) => {}
                _ => {
                    // First edit failure: stop editing for the rest of the turn and
                    // send a fresh message instead.
                    self.edit_supported = false;
                    if let Ok(Some(new_id)) = sink.create(&body).await {
                        self.message_id = Some(new_id);
                    }
                }
            },
            (Some(_), false) => {
                // Editing is disabled for this turn after an earlier failure: fall back
                // to today's throttled send behavior — a fresh message per update, so
                // progress is never lost (the consumer applies the 3s throttle).
                if let Ok(Some(new_id)) = sink.create(&body).await {
                    self.message_id = Some(new_id);
                }
            }
            (None, _) => {
                if let Ok(Some(id)) = sink.create(&body).await {
                    self.message_id = Some(id);
                }
            }
        }
    }

    pub async fn set_checklist(&mut self, sink: &dyn SurfaceSink, plan: PlanCard) {
        self.note_checklist(plan);
        self.flush(sink).await;
    }

    pub async fn set_activity(&mut self, sink: &dyn SurfaceSink, step: Step) {
        self.note_activity(step);
        self.flush(sink).await;
    }

    /// End the card with its turn, by the one rule ([`Self::end`]). The
    /// turn's own step always goes. A card that is kept (a helper on it, or
    /// a plan of more than one step) is settled to its final state and
    /// stays as the record of the work; a background helper still running
    /// settles it again when it ends. Any other card is removed: the reply
    /// says what was done. Safe to call more than once.
    pub async fn close(&mut self, sink: &dyn SurfaceSink) {
        self.activity = None;
        match self.end() {
            CardEnd::Keep => {
                let card = render_card(&self.view(false));
                if card.is_none() || self.shown == card {
                    return;
                }
                if let (Some(id), true, Some(body)) =
                    (&self.message_id, self.edit_supported, card.as_deref())
                {
                    let body = crate::presentation::render_internal_markers(body);
                    if matches!(sink.edit(id, &body).await, Ok(true)) {
                        self.shown = card;
                    }
                }
            }
            CardEnd::Collapse => {
                if let Some(id) = self.message_id.take() {
                    let _ = sink.delete(&id).await;
                }
                self.shown = None;
            }
        }
    }

    /// The card's editable message id, when one exists and editing still
    /// works this turn.
    pub fn surface_message_id(&self) -> Option<String> {
        if self.edit_supported {
            self.message_id.clone()
        } else {
            None
        }
    }

    /// The message showing this turn's kept card, when one is on screen
    /// and still editable, so work that outlives the turn (a background
    /// job settling its steps, a helper ending) can update the card.
    pub fn kept_card_message_id(&self) -> Option<String> {
        if self.end() == CardEnd::Collapse {
            return None;
        }
        self.surface_message_id()
    }
}

impl Default for LiveStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// `SurfaceSink` backed by the long-lived `ChannelHub`. Survives the per-request
/// status task being aborted, and is channel-agnostic (edit degrades to Ok(false)).
pub struct HubSurfaceSink {
    hub: Arc<ChannelHub>,
    session_id: String,
}

impl HubSurfaceSink {
    pub fn new(hub: Arc<ChannelHub>, session_id: String) -> Self {
        Self { hub, session_id }
    }
}

#[async_trait::async_trait]
impl SurfaceSink for HubSurfaceSink {
    async fn create(&self, text: &str) -> anyhow::Result<Option<String>> {
        self.hub.send_text_tracked(&self.session_id, text).await
    }
    async fn edit(&self, message_id: &str, text: &str) -> anyhow::Result<bool> {
        self.hub.edit_text(&self.session_id, message_id, text).await
    }
    async fn delete(&self, message_id: &str) -> anyhow::Result<bool> {
        self.hub.delete_message(&self.session_id, message_id).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use aidaemon_core::card::ActivityKind;

    use super::*;

    /// Records create/edit calls; create returns a fixed id, edit returns a flag.
    struct FakeSink {
        creates: StdMutex<Vec<String>>,
        edits: StdMutex<Vec<(String, String)>>,
        deletes: StdMutex<Vec<String>>,
        edit_ok: bool,
    }
    impl FakeSink {
        fn new(edit_ok: bool) -> Self {
            Self {
                creates: StdMutex::new(vec![]),
                edits: StdMutex::new(vec![]),
                deletes: StdMutex::new(vec![]),
                edit_ok,
            }
        }
    }
    #[async_trait::async_trait]
    impl SurfaceSink for FakeSink {
        async fn create(&self, text: &str) -> anyhow::Result<Option<String>> {
            self.creates.lock().unwrap().push(text.to_string());
            Ok(Some("m1".to_string()))
        }
        async fn edit(&self, message_id: &str, text: &str) -> anyhow::Result<bool> {
            self.edits
                .lock()
                .unwrap()
                .push((message_id.to_string(), text.to_string()));
            Ok(self.edit_ok)
        }
        async fn delete(&self, message_id: &str) -> anyhow::Result<bool> {
            self.deletes.lock().unwrap().push(message_id.to_string());
            Ok(true)
        }
    }

    fn doing(kind: ActivityKind) -> Step {
        Step::Doing(kind)
    }

    fn plan(text: &str, steps: usize) -> PlanCard {
        PlanCard {
            text: text.to_string(),
            steps,
        }
    }

    fn ended(label: &str, phase: RunPhase, background: bool) -> HelperView {
        HelperView {
            phase,
            ..HelperView::running(label, background, None)
        }
    }

    #[tokio::test]
    async fn creates_once_then_edits_in_place() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, doing(ActivityKind::Write)).await;
        s.set_activity(&sink, doing(ActivityKind::Run)).await;
        s.set_activity(&sink, doing(ActivityKind::Search)).await;
        assert_eq!(sink.creates.lock().unwrap().len(), 1);
        assert_eq!(sink.edits.lock().unwrap().len(), 2);
        assert_eq!(sink.edits.lock().unwrap()[0].1, "⏳ Running a command");
    }

    /// A plan card never hides the step running now or the helpers: each
    /// helper has its own header, under the plan, with its latest step.
    #[tokio::test]
    async fn the_plan_card_shows_each_helper_and_the_step_running_now() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, plan("📋 Plan\n☐ a\n☐ b", 2)).await;
        s.note_helper(HelperView::running("Count A", true, None));
        s.note_helper(HelperView::running("Count B", true, None));
        s.note_helper(HelperView::running(
            "Count B",
            true,
            Some(doing(ActivityKind::Run)),
        ));
        s.set_activity(&sink, doing(ActivityKind::Read)).await;
        assert_eq!(sink.creates.lock().unwrap().len(), 1);
        let card = sink.edits.lock().unwrap().last().unwrap().1.clone();
        assert_eq!(
            card,
            "📋 Plan\n☐ a\n☐ b\n\n⏳ Running in the background: Count A\n\n⏳ Running in the \
             background: Count B\nRunning a command\n\n⏳ Reading files"
        );
        // Nothing changed: nothing is re-sent.
        assert!(!s.flush_if_changed(&sink).await);
        assert_eq!(sink.edits.lock().unwrap().len(), 1);
        // The turn's own step goes when the turn ends; the helpers stay.
        s.close(&sink).await;
        assert_eq!(
            sink.edits.lock().unwrap().last().unwrap().1,
            "📋 Plan\n☐ a\n☐ b\n\n⏳ Running in the background: Count A\n\n⏳ Running in the \
             background: Count B\nRunning a command"
        );
        assert_eq!(s.kept_card_message_id().as_deref(), Some("m1"));
    }

    /// Two helpers started together each get their own header on the card,
    /// whatever order or batch their updates arrive in (live 2026-11-12:
    /// only "Counter B" showed), and every line is a typed kind in plain
    /// words: a tool ending and the model choosing its next step add
    /// nothing.
    #[test]
    fn parallel_helpers_each_get_their_own_line() {
        let mut live = LiveStatus::new();
        assert!(live.note(&StatusUpdate::Helper(HelperView::running(
            "Count A", true, None
        ))));
        assert!(live.note(&StatusUpdate::Helper(HelperView::running(
            "Count B", true, None
        ))));
        assert!(!live.note(&StatusUpdate::Helper(HelperView::running(
            "Count B",
            true,
            Some(Step::Doing(ActivityKind::Run)),
        ))));
        assert!(!live.note(&StatusUpdate::ToolComplete {
            name: "reading a file".into(),
            summary: "done".into(),
        }));
        assert!(live.note(&StatusUpdate::StepHandedOff {
            kind: ActivityKind::Delegate,
        }));
        assert!(!live.note(&StatusUpdate::Thinking(3)));
        assert_eq!(
            live.preview().unwrap(),
            "⏳ Running in the background: Count A\n\n⏳ Running in the background: \
             Count B\nRunning a command\n\n⏳ Handed to a helper; it works in the background"
        );
    }

    /// How a card ends is decided once from its typed state: a helper on
    /// it, or a plan of more than one step, keeps it; anything else
    /// collapses. The rendered text never decides.
    #[test]
    fn a_cards_end_comes_from_its_typed_state() {
        let mut live = LiveStatus::new();
        live.note_activity(doing(ActivityKind::Run));
        assert_eq!(live.end(), CardEnd::Collapse);
        live.note_checklist(plan("✅ Remember X", 1));
        assert_eq!(live.end(), CardEnd::Collapse);
        live.note_checklist(plan("✅ Done · 2 of 2 steps\n\n✅ a\n✅ b", 2));
        assert_eq!(live.end(), CardEnd::Keep);

        let mut helpers = LiveStatus::new();
        helpers.note_helper(ended("Count lines", RunPhase::Succeeded, false));
        assert_eq!(helpers.end(), CardEnd::Keep);
    }

    /// A foreground turn whose helpers ran side by side keeps its card,
    /// settled to each helper's final header and its measured time, after
    /// the reply (live 2026-11-13: the card vanished at turn end).
    #[tokio::test]
    async fn a_card_with_helpers_is_kept_and_settled() {
        let sink = FakeSink::new(true);
        let mut live = LiveStatus::new();
        live.note_helper(HelperView::running("Count lines", false, None));
        live.note_helper(HelperView::running("Count words", false, None));
        live.set_activity(&sink, Step::Doing(ActivityKind::Delegate))
            .await;
        live.note_helper(HelperView {
            took: Some("Took 21s (2:21:16 PM to 2:21:37 PM ET).".into()),
            ..ended("Count lines", RunPhase::Succeeded, false)
        });
        live.note_helper(ended("Count words", RunPhase::Failed, false));
        live.close(&sink).await;
        assert!(sink.deletes.lock().unwrap().is_empty());
        assert_eq!(
            sink.edits.lock().unwrap().last().unwrap().1,
            "✅ A helper finished: Count lines\nTook 21s (2:21:16 PM to 2:21:37 PM ET).\n\n❌ A \
             helper failed: Count words"
        );
        assert_eq!(live.kept_card_message_id().as_deref(), Some("m1"));
    }

    /// The live turn (2026-11-13): two helpers installed a site each. The
    /// header agrees with how each helper's run ended, and once both ended
    /// the card says how long after the owner's request the last one did.
    #[test]
    fn the_header_and_the_total_follow_the_helpers_run_records() {
        let plan = plan("✅ Done · 3 of 3 steps\n\n✅ a\n\n✅ b\n\n✅ c", 3);
        let helper = |label: &str, phase: RunPhase, secs: u64, at: &str| HelperView {
            took: Some(format!("Took {secs}s.")),
            since_request: Some(aidaemon_core::card::SinceRequest {
                secs,
                words: format!("{secs}s after your request (5:37:00 AM to {at} AM ET)"),
            }),
            ..ended(label, phase, false)
        };
        let both_done = render_card(&CardView {
            plan: Some(&plan),
            helpers: vec![
                (
                    helper("Install Drupal", RunPhase::Succeeded, 310, "5:42:10"),
                    None,
                ),
                (
                    helper("Install WordPress", RunPhase::Succeeded, 300, "5:42:00"),
                    None,
                ),
            ],
            ..Default::default()
        })
        .unwrap();
        assert!(
            both_done.starts_with("✅ Done · 3 of 3 steps"),
            "{both_done}"
        );
        assert!(
            both_done.contains("Took 310s. Done 310s after your request"),
            "{both_done}"
        );
        assert!(
            both_done
                .ends_with("All 2 done 310s after your request (5:37:00 AM to 5:42:10 AM ET)."),
            "{both_done}"
        );

        let one_stopped = render_card(&CardView {
            plan: Some(&plan),
            helpers: vec![
                (
                    helper("Install Drupal", RunPhase::Stopped, 310, "5:42:10"),
                    None,
                ),
                (
                    helper("Install WordPress", RunPhase::Succeeded, 300, "5:42:00"),
                    None,
                ),
            ],
            ..Default::default()
        })
        .unwrap();
        assert!(
            one_stopped.starts_with("⚠️ A helper didn't finish · 3 of 3 steps"),
            "{one_stopped}"
        );
        assert!(!one_stopped.contains("✅ Done"), "{one_stopped}");
        assert!(
            one_stopped.ends_with(
                "The last of them ended 310s after your request (5:37:00 AM to 5:42:10 AM ET)."
            ),
            "{one_stopped}"
        );

        // While one still runs there is no total yet.
        let running = render_card(&CardView {
            helpers: vec![
                (
                    helper("Install Drupal", RunPhase::Succeeded, 310, "5:42:10"),
                    None,
                ),
                (HelperView::running("Install WordPress", false, None), None),
            ],
            ..Default::default()
        })
        .unwrap();
        assert!(!running.contains("All 2 done"), "{running}");
    }

    /// A step of a generic kind ("Checking things") says nothing: the card
    /// shows the typed latest step, or none.
    #[test]
    fn a_generic_step_is_never_shown() {
        for generic in [ActivityKind::Check, ActivityKind::Change] {
            let step = Step::Doing(generic);
            let card = render_card(&CardView {
                helpers: vec![(
                    HelperView::running("Install WordPress", false, Some(step.clone())),
                    None,
                )],
                activity: Some((&step, Duration::from_secs(1))),
                ..Default::default()
            })
            .unwrap();
            assert!(!card.contains(&generic.words().to_string()), "{card}");
        }
    }

    /// A single-step card with no helper is only progress: it is removed
    /// once the reply is out, never turned into a message of its own.
    #[tokio::test]
    async fn a_single_step_card_collapses() {
        let sink = FakeSink::new(true);
        let mut live = LiveStatus::new();
        live.set_activity(&sink, doing(ActivityKind::Read)).await;
        live.close(&sink).await;
        assert_eq!(*sink.deletes.lock().unwrap(), vec!["m1".to_string()]);

        let sink = FakeSink::new(true);
        let mut live = LiveStatus::new();
        live.set_checklist(&sink, plan("⏳ Remember X", 1)).await;
        live.set_checklist(&sink, plan("✅ Remember X", 1)).await;
        live.close(&sink).await;
        assert_eq!(*sink.deletes.lock().unwrap(), vec!["m1".to_string()]);
        assert_eq!(live.kept_card_message_id(), None);
    }

    /// A plan of more than one step stays as the record of the work,
    /// finished or not.
    #[tokio::test]
    async fn a_plan_of_several_steps_is_kept() {
        for card in [
            "⚠️ Stopped short · 1 of 2 done\n\n✅ Step 1\n⏭️ Step 2 (not done)",
            "✅ Done · 2 of 2 steps\n\n✅ a\n✅ b",
        ] {
            let sink = FakeSink::new(true);
            let mut live = LiveStatus::new();
            live.set_checklist(&sink, plan(card, 2)).await;
            live.close(&sink).await;
            assert!(sink.deletes.lock().unwrap().is_empty(), "{card}");
            assert_eq!(live.kept_card_message_id().as_deref(), Some("m1"));
        }
    }

    /// The live card and the card a turn leaves behind are one renderer
    /// over one view: the same plan and helper records draw the same card,
    /// whichever path draws it.
    #[tokio::test]
    async fn the_live_and_the_kept_card_agree_for_the_same_records() {
        let plan = plan(
            "⏳ Continuing in the background · 0 of 1 done\n\n⏳ Count (running in the \
             background)",
            1,
        );
        let record = HelperView::running("Slow count", true, Some(Step::Output("13".into())));
        let sink = FakeSink::new(true);
        let mut live = LiveStatus::new();
        live.set_checklist(&sink, plan.clone()).await;
        live.note_helper(record.clone());
        live.set_activity(&sink, Step::HandedOff(ActivityKind::Delegate))
            .await;
        live.close(&sink).await;
        let kept = sink.edits.lock().unwrap().last().unwrap().1.clone();
        let drawn_after_the_turn = render_card(&CardView {
            plan: Some(&plan),
            helpers: vec![(record.clone(), None)],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(kept, drawn_after_the_turn);

        // Its end settles the card to the final header.
        let ended = HelperView {
            phase: RunPhase::Succeeded,
            step: None,
            ..record
        };
        let mut live = LiveStatus::new();
        live.note_helper(ended.clone());
        assert_eq!(
            live.preview(),
            render_card(&CardView {
                helpers: vec![(ended, None)],
                ..Default::default()
            })
        );
        assert_eq!(
            live.preview().unwrap(),
            "✅ Finished in the background: Slow count"
        );
    }

    /// A running command's latest line reads as its output, never as a
    /// label or a counter of the step.
    #[test]
    fn a_commands_latest_line_reads_as_its_output() {
        assert_eq!(
            Step::Output("3".into()).words(),
            "Running a command · last output: 3"
        );
        assert_eq!(Step::Output("  ".into()).words(), "Running a command");
    }

    /// Nothing the owner reads names a tool, an internal label or a raw
    /// step: every line is plain words from a typed kind.
    #[test]
    fn no_owner_facing_line_carries_an_internal_label() {
        use ActivityKind::*;
        let kinds = [
            Read, Search, Write, Run, Web, Delegate, Recall, Remember, Remind, Check, Change,
        ];
        let mut lines: Vec<String> = Vec::new();
        for kind in kinds {
            lines.push(Step::Doing(kind).words());
            lines.push(Step::HandedOff(kind).words());
        }
        lines.push(Step::Output("42".into()).words());
        lines.push(Step::Reporting.words());
        for phase in [
            RunPhase::Running,
            RunPhase::Succeeded,
            RunPhase::Stopped,
            RunPhase::Failed,
        ] {
            for background in [true, false] {
                lines.push(
                    render_card(&CardView {
                        helpers: vec![(
                            HelperView {
                                label: "Count the letters".into(),
                                phase,
                                background,
                                step: Some(Step::Doing(Search)),
                                took: None,
                                since_request: None,
                            },
                            None,
                        )],
                        ..Default::default()
                    })
                    .unwrap(),
                );
            }
        }
        for line in &lines {
            let lower = line.to_lowercase();
            for internal in [
                "tool",
                "_",
                "spawn",
                "specialist",
                "delegat",
                "searching files",
                "started in the background",
                "finished reading",
                "next step",
                "...",
            ] {
                assert!(!lower.contains(internal), "{internal:?} in {line:?}");
            }
        }
    }

    /// A long step says for how long it has run, in whole minutes, so the
    /// card changes at most once a minute.
    #[test]
    fn a_long_step_says_how_long_it_has_run() {
        assert_eq!(
            line_with_elapsed("Running a command", Duration::from_secs(10)),
            "Running a command"
        );
        assert_eq!(
            line_with_elapsed("Running a command", Duration::from_secs(45)),
            "Running a command · under a minute so far"
        );
        assert_eq!(
            line_with_elapsed("Running a command", Duration::from_secs(6 * 60 + 5)),
            "Running a command · 6 min so far"
        );
    }

    /// A step that handed its work on never reads as finished, and a late
    /// step never undoes how a helper ended.
    #[test]
    fn a_handed_off_step_never_reads_as_finished() {
        assert_eq!(
            Step::HandedOff(ActivityKind::Delegate).words(),
            "Handed to a helper; it works in the background"
        );
        assert_eq!(
            Step::HandedOff(ActivityKind::Run).words(),
            "Running a command · goes on in the background"
        );
        let mut live = LiveStatus::new();
        live.note_helper(ended("Count", RunPhase::Succeeded, false));
        live.note_helper(HelperView::running(
            "Count",
            false,
            Some(doing(ActivityKind::Run)),
        ));
        assert_eq!(live.preview().unwrap(), "✅ A helper finished: Count");
    }

    #[tokio::test]
    async fn keeps_sending_after_edit_fallback() {
        let sink = FakeSink::new(false); // edits always fail
        let mut s = LiveStatus::new();
        s.set_activity(&sink, doing(ActivityKind::Read)).await; // create #1
        s.set_activity(&sink, doing(ActivityKind::Run)).await; // edit fails -> create #2
        s.set_activity(&sink, doing(ActivityKind::Web)).await; // still no edit -> create #3
        assert_eq!(
            sink.creates.lock().unwrap().len(),
            3,
            "post-fallback updates keep sending fresh messages; progress is never lost"
        );
        assert_eq!(
            sink.edits.lock().unwrap().len(),
            1,
            "edit attempted once, then disabled"
        );
    }

    /// A card left in the chat can still be edited after the turn (a
    /// background job settling its steps); an activity line or a dropped
    /// card leaves nothing to edit.
    #[tokio::test]
    async fn kept_card_stays_editable_after_the_turn() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, doing(ActivityKind::Check)).await;
        assert_eq!(s.kept_card_message_id(), None);
        s.set_checklist(
            &sink,
            plan("⏳ Continuing in the background · 0 of 2 done", 2),
        )
        .await;
        s.close(&sink).await;
        assert_eq!(s.kept_card_message_id().as_deref(), Some("m1"));
    }
}
