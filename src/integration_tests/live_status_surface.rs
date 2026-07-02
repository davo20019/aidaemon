// Integration test: hub-backed live status surface creates exactly ONE tracked
// message and edits it in place (activity, checklist, final reply).

mod live_status_surface_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use tokio::sync::{Mutex, RwLock};

    use crate::channels::live_status::{HubSurfaceSink, LiveStatus, SurfaceSink};
    use crate::channels::{ChannelHub, SessionMap};
    use crate::tools::command_risk::{PermissionMode, RiskLevel};
    use crate::traits::{Channel, ChannelCapabilities};
    use crate::types::{ApprovalResponse, MediaMessage};

    /// Test-only channel that supports `send_text_tracked` (returns a stable id)
    /// and `edit_text` (records the edit). Used to prove the hub routes both calls
    /// to the same underlying channel.
    struct EditableChannel {
        messages: Mutex<Vec<(String, String)>>,
        edits: Mutex<Vec<(String, String)>>,
    }

    #[async_trait]
    impl Channel for EditableChannel {
        fn name(&self) -> String {
            "editable-test".to_string()
        }

        fn capabilities(&self) -> ChannelCapabilities {
            ChannelCapabilities {
                markdown: true,
                inline_buttons: false,
                media: false,
                max_message_len: 4096,
            }
        }

        async fn send_text(&self, session_id: &str, text: &str) -> anyhow::Result<()> {
            self.messages
                .lock()
                .await
                .push((session_id.to_string(), text.to_string()));
            Ok(())
        }

        async fn send_text_tracked(
            &self,
            session_id: &str,
            text: &str,
        ) -> anyhow::Result<Option<String>> {
            self.send_text(session_id, text).await?;
            Ok(Some("m1".to_string()))
        }

        async fn edit_text(
            &self,
            _session_id: &str,
            message_id: &str,
            text: &str,
        ) -> anyhow::Result<bool> {
            self.edits
                .lock()
                .await
                .push((message_id.to_string(), text.to_string()));
            Ok(true)
        }

        async fn send_media(&self, _session_id: &str, _media: &MediaMessage) -> anyhow::Result<()> {
            Ok(())
        }

        async fn request_approval(
            &self,
            _session_id: &str,
            _command: &str,
            _risk_level: RiskLevel,
            _warnings: &[String],
            _permission_mode: PermissionMode,
        ) -> anyhow::Result<ApprovalResponse> {
            Ok(ApprovalResponse::AllowOnce)
        }
    }

    /// Verify that a hub-backed `HubSurfaceSink` drives `LiveStatus` through:
    ///   1. First `set_activity`  → creates ONE tracked message (creates = 1, edits = 0)
    ///   2. Second `set_activity` → edits that message in place      (edits = 1)
    ///   3. `set_checklist`       → edits in place again             (edits = 2)
    ///   4. `finalize_text`       → edits the surface into the final reply (edits = 3)
    ///
    /// Also asserts that no raw command (`cd '/tmp'`) or pid (`pid=`) leaks into
    /// any edit payload.
    #[tokio::test]
    async fn hub_backed_live_status_creates_once_and_edits() {
        let channel = Arc::new(EditableChannel {
            messages: Mutex::new(vec![]),
            edits: Mutex::new(vec![]),
        });

        let session_map: SessionMap = Arc::new(RwLock::new(HashMap::new()));
        session_map.write().await.insert(
            "telegram:synthetic-user-1".to_string(),
            "editable-test".to_string(),
        );

        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn Channel>],
            session_map,
        ));

        let sink = HubSurfaceSink::new(hub, "telegram:synthetic-user-1".to_string());
        let sink_ref: &dyn SurfaceSink = &sink;

        let mut live = LiveStatus::new();

        // Activity 1 — lazy create (no message yet → send_text_tracked)
        live.set_activity(sink_ref, "Writing netprobe3.py".to_string())
            .await;
        // Activity 2 — edit in place
        live.set_activity(sink_ref, "Running the script".to_string())
            .await;
        // Checklist — edit in place (checklist owns the surface from here)
        live.set_checklist(sink_ref, "📋 Plan\n✅ Create script\n✅ Run it".to_string())
            .await;
        // Finalize — edits the surface into the final reply
        let handled = live
            .finalize_text(sink_ref, "Done — result file attached.")
            .await;

        assert!(handled, "finalize_text must return true when edit succeeds");
        assert_eq!(
            channel.messages.lock().await.len(),
            1,
            "exactly one tracked status message created"
        );

        let edits = channel.edits.lock().await;
        assert_eq!(
            edits.len(),
            3,
            "activity edit + checklist edit + final reply edit"
        );

        // No raw command or pid must appear in any edit payload.
        for (_, text) in edits.iter() {
            assert!(
                !text.contains("cd '/tmp'"),
                "raw command leaked into edit payload: {text}"
            );
            assert!(
                !text.contains("pid="),
                "pid leaked into edit payload: {text}"
            );
        }
    }

    /// Background completion pings EDIT the registered "⏳ Still on it —"
    /// handoff bubble in place (one evolving status message); when no surface
    /// is registered (or it was already consumed) they fall back to a fresh
    /// message. Exercises the real ChannelHub registry + the terminal
    /// notifier's delivery helper end to end.
    #[tokio::test]
    async fn background_completion_ping_edits_registered_handoff_bubble() {
        let channel = Arc::new(EditableChannel {
            messages: Mutex::new(vec![]),
            edits: Mutex::new(vec![]),
        });
        let session_map: SessionMap = Arc::new(RwLock::new(HashMap::new()));
        session_map.write().await.insert(
            "telegram:synthetic-user-1".to_string(),
            "editable-test".to_string(),
        );
        let hub = Arc::new(ChannelHub::new(
            vec![channel.clone() as Arc<dyn Channel>],
            session_map,
        ));

        // The delivered handoff reply registered its message id ("m1").
        hub.register_background_status_surface("telegram:synthetic-user-1", "m1")
            .await;

        let ping = "✅ Done — finished in 1m 3s. Writing up the result now…";
        crate::tools::terminal::deliver_background_completion_ping(
            Some(&hub),
            None,
            "telegram:synthetic-user-1",
            "goal-1",
            ping,
            4242,
        )
        .await;

        {
            let edits = channel.edits.lock().await;
            let messages = channel.messages.lock().await;
            assert_eq!(edits.len(), 1, "ping must edit the handoff bubble");
            assert_eq!(edits[0].0, "m1");
            assert!(edits[0].1.contains("Done — finished in"));
            assert!(messages.is_empty(), "no fresh message when the edit works");
        }

        // The registry entry is consumed: a second ping (e.g. another
        // background command whose handoff was never registered) falls back
        // to a fresh message.
        crate::tools::terminal::deliver_background_completion_ping(
            Some(&hub),
            None,
            "telegram:synthetic-user-1",
            "goal-1",
            ping,
            4243,
        )
        .await;
        let edits = channel.edits.lock().await;
        let messages = channel.messages.lock().await;
        assert_eq!(edits.len(), 1, "no second edit — registry entry consumed");
        assert_eq!(messages.len(), 1, "fallback fresh message delivered");
    }
}
