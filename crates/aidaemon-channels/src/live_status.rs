//! Single self-editing status surface per turn. One message is created lazily on
//! the first progress event and edited in place for every subsequent update, so a
//! task shows one calm, updating line/checklist instead of many messages.

use std::sync::Arc;

use crate::hub::ChannelHub;

/// Whether a plan card (a rendered checklist) shows every step finished.
/// Such a card is only progress: the reply covers the result.
pub fn is_completed_checklist(rendered: &str) -> bool {
    rendered.trim_start().starts_with('✅')
}

/// The status line once a step has ended: it says the step finished, so a
/// line left on screen never reads as work still running ("Finished running
/// a command", not "Running a command…").
pub fn step_ended_line(step: &str) -> String {
    let step = step.trim().trim_end_matches(['…', '.']);
    let mut chars = step.chars();
    let step = match chars.next() {
        Some(first) => first.to_lowercase().chain(chars).collect::<String>(),
        None => return "finished a step".to_string(),
    };
    format!("finished {step}")
}

/// The line for a step that handed its work on: the work goes on, so it
/// never reads as finished ("Finished delegating to a specialist" while the
/// helper still ran, live 2026-11-12).
pub fn step_handed_off_line(step: &str, to_helper: bool) -> String {
    if to_helper {
        return "Handed off to a helper; it's working in the background".to_string();
    }
    let step = step.trim().trim_end_matches(['…', '.']);
    let mut chars = step.chars();
    match chars.next() {
        Some(first) => format!(
            "{}{} goes on in the background",
            first.to_uppercase(),
            chars.as_str()
        ),
        None => "A step goes on in the background".to_string(),
    }
}

/// A helper's start line, one per helper. It stays on the card while the
/// helper works, its latest step under it.
pub fn helper_started_line(label: &str) -> String {
    format!("Started in the background: {}", label.trim())
}

/// A plan card with a start line under it for each helper in `labels`,
/// one per helper, in order.
pub fn with_helper_start_lines<'a>(
    card: &str,
    labels: impl IntoIterator<Item = &'a str>,
) -> String {
    let mut out = card.trim_end().to_string();
    for label in labels {
        let line = helper_started_line(label);
        if !out.lines().any(|shown| shown == line) {
            out.push_str("\n\n");
            out.push_str(&line);
        }
    }
    out
}

/// A plan card with each helper's start line under it and, when it has
/// reported one, its latest step under that, the way the live card shows
/// them: for a card redrawn after its turn while helpers still work.
pub fn with_helper_lines<'a>(
    card: &str,
    helpers: impl IntoIterator<Item = (&'a str, Option<&'a str>)>,
) -> String {
    let mut out = card.trim_end().to_string();
    for (label, step) in helpers {
        let line = helper_started_line(label);
        if out.lines().any(|shown| shown == line) {
            continue;
        }
        out.push_str("\n\n");
        out.push_str(&line);
        if let Some(step) = step.map(str::trim).filter(|step| !step.is_empty()) {
            out.push_str("\n🤖 ");
            out.push_str(step);
        }
    }
    out
}

/// A helper's own step that went on in the background, as its progress
/// line ("counting 1 to 60 goes on in the background").
pub fn helper_step_handed_off(step: &str, to_helper: bool) -> String {
    let line = step_handed_off_line(step, to_helper);
    let mut chars = line.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().chain(chars).collect(),
        None => line,
    }
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
    current_line: Option<(String, std::time::Instant)>,
    /// Each helper's line, in the order they started: its label, what it
    /// is doing and since when.
    helpers: Vec<(String, HelperLine, std::time::Instant)>,
    /// What the surface shows now, so an unchanged card is never re-sent.
    shown: Option<String>,
    /// The surface was edited into the turn's reply.
    finalized: bool,
    edit_supported: bool,
}

/// What one helper's line on the card says.
#[derive(Debug, Clone, PartialEq, Eq)]
enum HelperLine {
    Started,
    Progress(String),
}

/// How long a step must have run before its line says for how long.
const ELAPSED_SHOWN_AFTER: std::time::Duration = std::time::Duration::from_secs(30);

/// `line` with how long its step has run, in whole minutes once it passes
/// [`ELAPSED_SHOWN_AFTER`] ("Running a command · 3 min so far"), so a long
/// step reads as alive and the card changes at most once a minute.
pub fn line_with_elapsed(line: &str, elapsed: std::time::Duration) -> String {
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
            current_line: None,
            helpers: Vec::new(),
            shown: None,
            finalized: false,
            edit_supported: true,
        }
    }

    /// The card: the plan (when there is one), each helper's line and the
    /// turn's own current step, each long step with how long it has run. A
    /// plan never hides the helpers or the step running now (live
    /// 2026-11-12: six minutes of a plan card and "typing" while two
    /// helpers worked).
    fn render(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(c) = &self.checklist {
            parts.push(c.clone());
        }
        let done = self
            .checklist
            .as_deref()
            .is_some_and(crate::live_status::is_completed_checklist);
        // Each helper keeps its start line, the same words for every helper
        // (live 2026-11-13: a first step replaced it, "Slow count to 60:
        // counting 1 to 60 slowly"), with its latest step under it.
        for (label, line, since) in &self.helpers {
            parts.push(match line {
                HelperLine::Started => helper_started_line(label),
                HelperLine::Progress(progress) => format!(
                    "{}\n🤖 {}",
                    helper_started_line(label),
                    line_with_elapsed(progress, since.elapsed())
                ),
            });
        }
        if !done {
            if let Some((line, since)) = &self.current_line {
                parts.push(format!(
                    "⏳ {}",
                    line_with_elapsed(&capitalize_first(line), since.elapsed())
                ));
            }
        }
        (!parts.is_empty()).then(|| parts.join("\n\n"))
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
    pub fn note_activity(&mut self, line: String) {
        if self.current_line.as_ref().map(|(shown, _)| shown) != Some(&line) {
            self.current_line = Some((line, std::time::Instant::now()));
        }
    }

    /// Note that a helper started: its own line, one per helper.
    pub fn note_helper_started(&mut self, label: &str) {
        if !self.helpers.iter().any(|(known, ..)| known == label) {
            self.helpers.push((
                label.to_string(),
                HelperLine::Started,
                std::time::Instant::now(),
            ));
        }
    }

    /// Note a helper's latest step on its own line.
    pub fn note_helper_progress(&mut self, label: &str, progress: &str) {
        let line = HelperLine::Progress(progress.to_string());
        match self.helpers.iter_mut().find(|(known, ..)| known == label) {
            Some(entry) if entry.1 == line => {}
            Some(entry) => {
                entry.1 = line;
                entry.2 = std::time::Instant::now();
            }
            None => self
                .helpers
                .push((label.to_string(), line, std::time::Instant::now())),
        }
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

    pub async fn set_activity(&mut self, sink: &dyn SurfaceSink, line: String) {
        self.note_activity(line);
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

    /// Remove the temporary surface when the turn is over. An activity line
    /// is always removed. A checklist with every step done is removed only
    /// when `reply_delivered`: the reply says what was done, so the card would
    /// repeat it; without a reply the card is the only sign the work happened.
    /// A checklist with unfinished, failed or deferred steps always stays as
    /// the record of what was left.
    pub async fn clear(&mut self, sink: &dyn SurfaceSink, reply_delivered: bool) {
        if let Some(checklist) = self.checklist.clone() {
            if !reply_delivered || !crate::live_status::is_completed_checklist(&checklist) {
                // The card stays as the record of the plan, with each
                // helper's start line: helpers started from a plan go on
                // after the turn (live 2026-11-13: two helpers started in
                // parallel and the card kept only "Continuing in the
                // background · 0 of 5 done"). Their latest steps and the
                // turn's own step were only live progress, so they go.
                let card = with_helper_start_lines(
                    &checklist,
                    self.helpers.iter().map(|(label, ..)| label.as_str()),
                );
                self.current_line = None;
                self.helpers.clear();
                if self.shown.as_deref() != Some(card.as_str()) {
                    if let (Some(id), true) = (&self.message_id, self.edit_supported) {
                        let body = crate::presentation::render_internal_markers(&card);
                        if matches!(sink.edit(id, &body).await, Ok(true)) {
                            self.shown = Some(card);
                        }
                    }
                }
                return;
            }
        }
        if let Some(id) = self.message_id.take() {
            let _ = sink.delete(&id).await;
        }
        self.current_line = None;
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

    /// The message showing this turn's plan card, when one is on screen and
    /// still editable. Read before `reset()`, so work that outlives the turn
    /// (a background job settling its steps) can update the card.
    pub fn checklist_message_id(&self) -> Option<String> {
        self.checklist.as_ref()?;
        self.surface_message_id()
    }

    pub fn reset(&mut self) {
        self.message_id = None;
        self.checklist = None;
        self.current_line = None;
        self.helpers.clear();
        self.shown = None;
        self.finalized = false;
        self.edit_supported = true;
    }
}

/// "checking memory" -> "Checking memory": tool labels are lowercase so they
/// read inside a sentence; a status line starts one.
fn capitalize_first(line: &str) -> String {
    let mut chars = line.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
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

    use super::*;

    /// A step that ended never leaves a line that reads as still running.
    #[test]
    fn an_ended_step_reads_as_finished() {
        assert_eq!(
            step_ended_line("running a command"),
            "finished running a command"
        );
        assert_eq!(
            step_ended_line("Running the site installer…"),
            "finished running the site installer"
        );
        assert_eq!(step_ended_line(" "), "finished a step");
    }

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

    #[tokio::test]
    async fn creates_once_then_edits_in_place() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, "Writing netprobe3.py".into()).await;
        s.set_activity(&sink, "Running the script".into()).await;
        s.set_activity(&sink, "Still running (45s)".into()).await;
        assert_eq!(
            sink.creates.lock().unwrap().len(),
            1,
            "exactly one message created"
        );
        assert_eq!(
            sink.edits.lock().unwrap().len(),
            2,
            "subsequent updates edit in place"
        );
        assert!(sink.edits.lock().unwrap()[0]
            .1
            .contains("⏳ Running the script"));
    }

    /// A plan card never hides the step running now or the helpers: each
    /// helper has its own line, under the plan (live 2026-11-12: six
    /// minutes of a plan card and "typing" while two helpers worked).
    #[tokio::test]
    async fn the_plan_card_shows_each_helper_and_the_step_running_now() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "📋 Plan\n☐ a".into()).await;
        s.note_helper_started("Counter A");
        s.note_helper_started("Counter B");
        s.note_helper_progress("Counter B", "running a command");
        s.set_activity(&sink, "waiting for the helpers".into())
            .await;
        assert_eq!(sink.creates.lock().unwrap().len(), 1);
        let card = sink.edits.lock().unwrap().last().unwrap().1.clone();
        assert_eq!(
            card,
            "📋 Plan\n☐ a\n\nStarted in the background: Counter A\n\nStarted in the background: \
             Counter B\n🤖 running a command\n\n⏳ Waiting for the helpers"
        );
        // Nothing changed: nothing is re-sent.
        assert!(!s.flush_if_changed(&sink).await);
        assert_eq!(sink.edits.lock().unwrap().len(), 1);
        // The kept plan card loses the live lines when the turn ends, but
        // each helper keeps its start line (live 2026-11-13: two helpers
        // started in parallel from a plan, and the card kept only the plan).
        s.clear(&sink, true).await;
        assert_eq!(
            sink.edits.lock().unwrap().last().unwrap().1,
            "📋 Plan\n☐ a\n\nStarted in the background: Counter A\n\nStarted in the \
             background: Counter B"
        );
        // Once one has finished, the card the plan is redrawn as keeps the
        // other's.
        assert_eq!(
            with_helper_start_lines("📋 Plan\n✅ a", ["Counter B"]),
            "📋 Plan\n✅ a\n\nStarted in the background: Counter B"
        );
    }

    /// A long step says for how long it has run, in whole minutes, so the
    /// card changes at most once a minute.
    #[test]
    fn a_long_step_says_how_long_it_has_run() {
        use std::time::Duration;
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

    /// A step that handed its work on never reads as finished.
    #[test]
    fn a_handed_off_step_never_reads_as_finished() {
        assert_eq!(
            step_handed_off_line("delegating to a specialist", true),
            "Handed off to a helper; it's working in the background"
        );
        assert_eq!(
            step_handed_off_line("running a command", false),
            "Running a command goes on in the background"
        );
        assert_eq!(
            helper_started_line("Counter A"),
            "Started in the background: Counter A"
        );
    }

    #[tokio::test]
    async fn keeps_sending_after_edit_fallback() {
        let sink = FakeSink::new(false); // edits always fail
        let mut s = LiveStatus::new();
        s.set_activity(&sink, "first".into()).await; // create #1
        s.set_activity(&sink, "second".into()).await; // edit fails -> create #2
        s.set_activity(&sink, "third".into()).await; // still no edit -> create #3
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
        s.set_activity(&sink, "working".into()).await;
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
        s.set_activity(&sink, "working".into()).await;
        assert!(
            !s.finalize_text(&sink, "huge reply").await,
            "edit failed -> finalize returns false -> caller sends normally"
        );
    }

    #[tokio::test]
    async fn clear_removes_surface_without_done_marker() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, "Working…".into()).await;
        s.clear(&sink, false).await;
        assert_eq!(
            *sink.deletes.lock().unwrap(),
            vec!["m1".to_string()],
            "clear deletes the temporary surface without editing it to '✅ Done'"
        );
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
        s.set_activity(&sink, "working".into()).await;
        assert_eq!(s.checklist_message_id(), None);
        s.set_checklist(
            &sink,
            "⏳ Continuing in the background · 0 of 1 done".into(),
        )
        .await;
        s.clear(&sink, true).await;
        assert_eq!(s.checklist_message_id().as_deref(), Some("m1"));

        let mut done = LiveStatus::new();
        done.set_checklist(&sink, "✅ Remember X".into()).await;
        done.clear(&sink, true).await;
        assert_eq!(done.checklist_message_id(), None);
    }

    #[tokio::test]
    async fn clear_drops_fully_done_checklist() {
        // Live 2026-11-11: "Remember X" left a "✅ Planned steps complete /
        // 1 of 1 steps complete / ✅ Remember X / Finalizing…" card above a
        // reply saying the same thing.
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "⏳ Remember X".into()).await;
        s.set_checklist(&sink, "✅ Remember X".into()).await;
        s.clear(&sink, true).await;
        assert_eq!(*sink.deletes.lock().unwrap(), vec!["m1".to_string()]);
    }

    #[tokio::test]
    async fn clear_keeps_fully_done_checklist_when_no_reply_went_out() {
        // Cancelled, errored or failed send: the card is the only sign the
        // work happened, so it stays.
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "✅ Done · 2 of 2 steps\n\n✅ a\n✅ b".into())
            .await;
        s.clear(&sink, false).await;
        assert_eq!(sink.deletes.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn activity_line_starts_with_a_capital() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_activity(&sink, "checking memory…".into()).await;
        assert_eq!(sink.creates.lock().unwrap()[0], "⏳ Checking memory…");
    }
}
