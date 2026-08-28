//! Single self-editing status surface per turn. One message is created lazily on
//! the first progress event and edited in place for every subsequent update, so a
//! task shows one calm, updating line/checklist instead of many messages.

use std::sync::Arc;

use crate::channels::ChannelHub;

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
    current_line: Option<String>,
    edit_supported: bool,
}

impl LiveStatus {
    pub fn new() -> Self {
        Self {
            message_id: None,
            checklist: None,
            current_line: None,
            edit_supported: true,
        }
    }

    fn render(&self) -> Option<String> {
        if let Some(c) = &self.checklist {
            return Some(c.clone());
        }
        self.current_line.as_ref().map(|l| format!("⏳ {l}"))
    }

    async fn flush(&mut self, sink: &dyn SurfaceSink) {
        let Some(body) = self.render() else { return };
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
        self.checklist = Some(text);
        self.flush(sink).await;
    }

    pub async fn set_activity(&mut self, sink: &dyn SurfaceSink, line: String) {
        // A checklist, once present, owns the surface; ignore single-line activity.
        if self.checklist.is_some() {
            return;
        }
        self.current_line = Some(line);
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
                return true;
            }
        }
        false
    }

    /// Remove the temporary activity surface once the final response has been
    /// delivered. A checklist is retained as the completed plan summary, but
    /// an activity-only surface should not become a separate "Done" bubble.
    pub async fn clear(&mut self, sink: &dyn SurfaceSink) {
        if self.checklist.is_some() {
            return;
        }
        if let Some(id) = self.message_id.take() {
            let _ = sink.delete(&id).await;
        }
        self.current_line = None;
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

    pub fn reset(&mut self) {
        self.message_id = None;
        self.checklist = None;
        self.current_line = None;
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

    #[tokio::test]
    async fn checklist_owns_the_surface() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "📋 Plan\n☐ a".into()).await;
        s.set_activity(&sink, "Running".into()).await; // ignored
        assert_eq!(sink.creates.lock().unwrap().len(), 1);
        assert_eq!(sink.edits.lock().unwrap().len(), 0);
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
        s.clear(&sink).await;
        assert_eq!(
            *sink.deletes.lock().unwrap(),
            vec!["m1".to_string()],
            "clear deletes the temporary surface without editing it to '✅ Done'"
        );
    }

    #[tokio::test]
    async fn clear_keeps_checklist_without_done_marker() {
        let sink = FakeSink::new(true);
        let mut s = LiveStatus::new();
        s.set_checklist(&sink, "📋 Plan\n✅ Step 1\n☐ Step 2".into())
            .await;
        s.clear(&sink).await;
        assert_eq!(
            sink.deletes.lock().unwrap().len(),
            0,
            "clear keeps the completed checklist instead of adding a done marker"
        );
    }
}
