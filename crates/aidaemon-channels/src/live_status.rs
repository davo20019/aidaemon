//! Single self-editing status surface per turn. One message is created lazily on
//! the first progress event and edited in place for every subsequent update, so a
//! task shows one calm, updating line/checklist instead of many messages.
//!
//! Every card, the live one during a turn and the one a turn leaves behind,
//! is drawn by [`render_card`] from one view: the plan's rendered steps, each
//! helper's record ([`HelperView`]) and the turn's own typed [`Step`].

use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::card::{HelperView, RunPhase, Step};

use crate::hub::ChannelHub;

/// Whether a plan card (a rendered checklist) shows every step finished.
/// Such a card is only progress: the reply covers the result.
pub fn is_completed_checklist(rendered: &str) -> bool {
    rendered.trim_start().starts_with('✅')
}

/// One card's state, as every surface draws it.
#[derive(Debug, Clone, Default)]
pub struct CardView<'a> {
    /// The plan's card, as the plan renders itself.
    pub plan: Option<&'a str>,
    /// Each helper's record, in the order they started, with how long its
    /// latest step has run when that is known.
    pub helpers: Vec<(HelperView, Option<Duration>)>,
    /// The turn's own current step and how long it has run.
    pub activity: Option<(&'a Step, Duration)>,
}

/// The card for `view`: the plan (when there is one), each helper's header
/// and its latest step while it runs, and the turn's own step, each long
/// step with how long it has run. A plan never hides the helpers or the
/// step running now; a finished plan drops the turn's own step.
pub fn render_card(view: &CardView<'_>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(plan) = view.plan {
        parts.push(plan.trim_end().to_string());
    }
    for (helper, elapsed) in &view.helpers {
        parts.push(helper_block(helper, *elapsed));
    }
    let done = view.plan.is_some_and(is_completed_checklist);
    if !done {
        if let Some((step, elapsed)) = view.activity {
            parts.push(format!("⏳ {}", line_with_elapsed(&step.words(), elapsed)));
        }
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Where a helper stands, in a few plain words: its latest step while it
/// runs, else how it ended. For one-line replies that list helpers.
pub fn helper_status_words(helper: &HelperView) -> String {
    match (helper.phase, helper.step.as_ref()) {
        (RunPhase::Running, Some(step)) => step.words(),
        (RunPhase::Running, None) => "Getting started".to_string(),
        (RunPhase::Succeeded, _) => "Finished".to_string(),
        (RunPhase::Stopped, _) => "Didn't finish".to_string(),
        (RunPhase::Failed, _) => "Failed".to_string(),
    }
}

/// A helper's lines on a card: its header, then its latest step while it
/// runs.
fn helper_block(helper: &HelperView, elapsed: Option<Duration>) -> String {
    let mut block = format!(
        "{}: {}",
        helper.phase.helper_headline(helper.background),
        helper.label.trim()
    );
    if let (RunPhase::Running, Some(step)) = (helper.phase, helper.step.as_ref()) {
        block.push('\n');
        block.push_str(&line_with_elapsed(
            &step.words(),
            elapsed.unwrap_or_default(),
        ));
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
/// between the status-consumer task and the outer request handler.
pub struct LiveStatus {
    message_id: Option<String>,
    checklist: Option<String>,
    /// The turn's own current step and when it began.
    activity: Option<(Step, std::time::Instant)>,
    /// Each helper's record, in the order they started, and when its
    /// latest step began.
    helpers: Vec<(HelperView, std::time::Instant)>,
    /// What the surface shows now, so an unchanged card is never re-sent.
    shown: Option<String>,
    /// The surface was edited into the turn's reply.
    finalized: bool,
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
            shown: None,
            finalized: false,
            edit_supported: true,
        }
    }

    /// The card as it is now.
    fn view(&self, with_activity: bool) -> CardView<'_> {
        CardView {
            plan: self.checklist.as_deref(),
            helpers: self
                .helpers
                .iter()
                .map(|(helper, since)| (helper.clone(), Some(since.elapsed())))
                .collect(),
            activity: with_activity
                .then_some(self.activity.as_ref())
                .flatten()
                .map(|(step, since)| (step, since.elapsed())),
        }
    }

    fn render(&self) -> Option<String> {
        render_card(&self.view(true))
    }

    /// What the card would show now.
    pub fn preview(&self) -> Option<String> {
        self.render()
    }

    /// Note the plan card without sending it.
    pub fn note_checklist(&mut self, text: String) {
        self.checklist = Some(text);
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
                let header_changed = known.phase != helper.phase;
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

    /// Whether a helper on the card still runs: the card stays its own
    /// message, to settle when the helper ends, and is never turned into
    /// the turn's reply.
    pub fn has_running_helpers(&self) -> bool {
        self.helpers
            .iter()
            .any(|(helper, _)| helper.phase == RunPhase::Running)
    }

    /// Whether the card outlives the turn: a plan with work left (or with
    /// no reply to say what was done), or a helper still running.
    fn kept(&self, reply_delivered: bool) -> bool {
        let plan_kept = self
            .checklist
            .as_deref()
            .is_some_and(|plan| !reply_delivered || !is_completed_checklist(plan));
        plan_kept || self.has_running_helpers()
    }

    /// Send the card when what it would show changed since it was last
    /// sent (a step's elapsed time included); `true` when it was sent.
    pub async fn flush_if_changed(&mut self, sink: &dyn SurfaceSink) -> bool {
        if self.finalized || self.render().is_none() || self.render() == self.shown {
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

    pub async fn set_checklist(&mut self, sink: &dyn SurfaceSink, text: String) {
        self.note_checklist(text);
        self.flush(sink).await;
    }

    pub async fn set_activity(&mut self, sink: &dyn SurfaceSink, step: Step) {
        self.note_activity(step);
        self.flush(sink).await;
    }

    /// Edit the surface into the final reply. Returns true if the caller no
    /// longer needs to send the reply separately.
    pub async fn finalize_text(&mut self, sink: &dyn SurfaceSink, reply: &str) -> bool {
        if reply.trim().is_empty() {
            return false;
        }
        if let (Some(id), true) = (&self.message_id, self.edit_supported) {
            if matches!(sink.edit(id, reply).await, Ok(true)) {
                // The surface is the reply now: no later refresh may turn
                // it back into the card.
                self.finalized = true;
                return true;
            }
        }
        false
    }

    /// Remove the temporary surface when the turn is over. The turn's own
    /// step always goes. The card stays as the record of the work when
    /// [`Self::kept`]: a plan with unfinished, failed or deferred steps, a
    /// finished plan no reply described, or a helper still running (it
    /// settles this card to its final header when it ends). A finished
    /// plan the reply describes, or a bare activity line, is removed.
    pub async fn clear(&mut self, sink: &dyn SurfaceSink, reply_delivered: bool) {
        self.activity = None;
        if self.kept(reply_delivered) {
            let card = render_card(&self.view(false));
            if card.is_some() && self.shown != card {
                if let (Some(id), true, Some(body)) =
                    (&self.message_id, self.edit_supported, card.as_deref())
                {
                    let body = crate::presentation::render_internal_markers(body);
                    if matches!(sink.edit(id, &body).await, Ok(true)) {
                        self.shown = card;
                    }
                }
            }
            return;
        }
        if let Some(id) = self.message_id.take() {
            let _ = sink.delete(&id).await;
        }
        self.helpers.clear();
        self.shown = None;
        self.edit_supported = true;
    }

    /// The surface's editable message id, when one exists and editing still
    /// works this turn. Read before `reset()` — e.g. to register a delivered
    /// background-handoff reply so a later completion ping can edit it.
    pub fn surface_message_id(&self) -> Option<String> {
        if self.edit_supported {
            self.message_id.clone()
        } else {
            None
        }
    }

    /// The message showing this turn's kept card, when one is on screen
    /// and still editable. Read before `reset()`, so work that outlives the
    /// turn (a background job settling its steps, a helper ending) can
    /// update the card.
    pub fn kept_card_message_id(&self) -> Option<String> {
        if self.checklist.is_none() && self.helpers.is_empty() {
            return None;
        }
        self.surface_message_id()
    }

    pub fn reset(&mut self) {
        self.message_id = None;
        self.checklist = None;
        self.activity = None;
        self.helpers.clear();
        self.shown = None;
        self.finalized = false;
        self.edit_supported = true;
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
        s.set_checklist(&sink, "📋 Plan\n☐ a".into()).await;
        s.note_helper(HelperView::running("Counter A", true, None));
        s.note_helper(HelperView::running("Counter B", true, None));
        s.note_helper(HelperView::running(
            "Counter B",
            true,
            Some(doing(ActivityKind::Run)),
        ));
        s.set_activity(&sink, doing(ActivityKind::Think)).await;
        assert_eq!(sink.creates.lock().unwrap().len(), 1);
        let card = sink.edits.lock().unwrap().last().unwrap().1.clone();
        assert_eq!(
            card,
            "📋 Plan\n☐ a\n\n⏳ Running in the background: Counter A\n\n⏳ Running in the \
             background: Counter B\nRunning a command\n\n⏳ Working out the next step"
        );
        // Nothing changed: nothing is re-sent.
        assert!(!s.flush_if_changed(&sink).await);
        assert_eq!(sink.edits.lock().unwrap().len(), 1);
        // The turn's own step goes when the turn ends; the helpers stay.
        s.clear(&sink, true).await;
        assert_eq!(
            sink.edits.lock().unwrap().last().unwrap().1,
            "📋 Plan\n☐ a\n\n⏳ Running in the background: Counter A\n\n⏳ Running in the \
             background: Counter B\nRunning a command"
        );
        assert_eq!(s.kept_card_message_id().as_deref(), Some("m1"));
    }

    /// The live card and the card a turn leaves behind are one renderer
    /// over one view: the same plan and helper records draw the same card,
    /// whichever path draws it.
    #[tokio::test]
    async fn the_live_and_the_kept_card_agree_for_the_same_records() {
        let plan = "⏳ Continuing in the background · 0 of 1 done\n\n⏳ Count (running in the \
                    background)";
        let record = HelperView::running("Slow count", true, Some(Step::Output("13".into())));
        let sink = FakeSink::new(true);
        let mut live = LiveStatus::new();
        live.set_checklist(&sink, plan.into()).await;
        live.note_helper(record.clone());
        live.set_activity(&sink, Step::HandedOff(ActivityKind::Delegate))
            .await;
        live.clear(&sink, true).await;
        let kept = sink.edits.lock().unwrap().last().unwrap().1.clone();
        let drawn_after_the_turn = render_card(&CardView {
            plan: Some(plan),
            helpers: vec![(record.clone(), None)],
            activity: None,
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

    /// Nothing the owner reads names a tool, an internal label or a raw
    /// step: every line is plain words from a typed kind.
    #[test]
    fn no_owner_facing_line_carries_an_internal_label() {
        use ActivityKind::*;
        let kinds = [
            Read, Search, Write, Run, Web, Delegate, Recall, Remember, Remind, Check, Change, Think,
        ];
        let mut lines: Vec<String> = Vec::new();
        for kind in kinds {
            lines.push(Step::Doing(kind).words());
            lines.push(Step::HandedOff(kind).words());
        }
        lines.push(Step::Output("42".into()).words());
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
                                label: "Letters file helper".into(),
                                phase,
                                background,
                                step: Some(Step::Doing(Search)),
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
        live.note_helper(HelperView {
            phase: RunPhase::Succeeded,
            ..HelperView::running("Count", false, None)
        });
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

    #[tokio::test]
    async fn finalize_text_edits_surface_into_reply() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, doing(ActivityKind::Check)).await;
        let handled = s.finalize_text(&sink, "Done — results attached.").await;
        assert!(handled, "final reply consumed by editing the surface");
        assert_eq!(
            sink.edits.lock().unwrap().last().unwrap().1,
            "Done — results attached."
        );
    }

    #[tokio::test]
    async fn finalize_text_false_when_no_surface() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        let handled = s.finalize_text(&sink, "Answer").await;
        assert!(!handled, "no surface -> caller sends normally");
    }

    #[tokio::test]
    async fn finalize_text_false_when_edit_fails() {
        let sink = FakeSink::new(false); // edit always returns Ok(false)
        let mut s = LiveStatus::new();
        s.set_activity(&sink, doing(ActivityKind::Check)).await;
        assert!(
            !s.finalize_text(&sink, "huge reply").await,
            "edit failed -> finalize returns false -> caller sends normally"
        );
    }

    #[tokio::test]
    async fn clear_removes_surface_without_done_marker() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, doing(ActivityKind::Check)).await;
        s.clear(&sink, false).await;
        assert_eq!(*sink.deletes.lock().unwrap(), vec!["m1".to_string()]);
    }

    #[tokio::test]
    async fn clear_keeps_unfinished_checklist() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(
            &sink,
            "⚠️ Stopped short · 1 of 2 done\n\n✅ Step 1\n⏭️ Step 2 (not done)".into(),
        )
        .await;
        s.clear(&sink, true).await;
        assert_eq!(
            sink.deletes.lock().unwrap().len(),
            0,
            "a card with unfinished steps stays as the record of what was left"
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
            "⏳ Continuing in the background · 0 of 1 done".into(),
        )
        .await;
        s.clear(&sink, true).await;
        assert_eq!(s.kept_card_message_id().as_deref(), Some("m1"));

        let mut done = LiveStatus::new();
        done.set_checklist(&sink, "✅ Remember X".into()).await;
        done.clear(&sink, true).await;
        assert_eq!(done.kept_card_message_id(), None);
    }

    #[tokio::test]
    async fn clear_drops_fully_done_checklist() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "⏳ Remember X".into()).await;
        s.set_checklist(&sink, "✅ Remember X".into()).await;
        s.clear(&sink, true).await;
        assert_eq!(*sink.deletes.lock().unwrap(), vec!["m1".to_string()]);
    }

    #[tokio::test]
    async fn clear_keeps_fully_done_checklist_when_no_reply_went_out() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "✅ Done · 2 of 2 steps\n\n✅ a\n✅ b".into())
            .await;
        s.clear(&sink, false).await;
        assert_eq!(sink.deletes.lock().unwrap().len(), 0);
    }
}
