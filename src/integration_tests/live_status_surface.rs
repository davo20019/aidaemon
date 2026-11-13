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
    use crate::traits::{HubChannel, ChannelCapabilities};
    use crate::types::{ApprovalResponse, MediaMessage};

    /// Test-only channel that supports `send_text_tracked` (returns a stable id)
    /// and `edit_text` (records the edit). Used to prove the hub routes both calls
    /// to the same underlying channel.
    struct EditableChannel {
        messages: Mutex<Vec<(String, String)>>,
        edits: Mutex<Vec<(String, String)>>,
    }

    #[async_trait]
    impl HubChannel for EditableChannel {
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
            _one_time_only: bool,
        ) -> anyhow::Result<ApprovalResponse> {
            Ok(ApprovalResponse::AllowOnce)
        }
    }

    /// Verify that a hub-backed `HubSurfaceSink` drives `LiveStatus` through:
    ///   1. First `set_activity`  → creates ONE tracked message (creates = 1, edits = 0)
    ///   2. Second `set_activity` → edits that message in place      (edits = 1)
    ///   3. `set_checklist`       → edits in place again             (edits = 2)
    ///   4. `close`               → a two-step plan is kept, settled without the
    ///      turn's own step (edits = 3), and never becomes the reply
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
            vec![channel.clone() as Arc<dyn HubChannel>],
            session_map,
        ));
        let sink = HubSurfaceSink::new(hub, "telegram:synthetic-user-1".to_string());
        let sink_ref: &dyn SurfaceSink = &sink;

        let mut live = LiveStatus::new();

        // Activity 1 — lazy create (no message yet → send_text_tracked)
        live.set_activity(
            sink_ref,
            aidaemon_core::card::Step::Doing(aidaemon_core::card::ActivityKind::Write),
        )
        .await;
        // Activity 2 — edit in place
        live.set_activity(
            sink_ref,
            aidaemon_core::card::Step::Doing(aidaemon_core::card::ActivityKind::Run),
        )
        .await;
        // Checklist — edit in place (checklist owns the surface from here)
        live.set_checklist(
            sink_ref,
            aidaemon_core::card::PlanCard {
                text: "📋 Plan\n✅ Create script\n✅ Run it".to_string(),
                steps: 2,
            },
        )
        .await;
        // A plan of two steps is the record of the work: the turn's end
        // keeps it.
        live.close(sink_ref).await;

        assert_eq!(
            channel.messages.lock().await.len(),
            1,
            "exactly one tracked status message created"
        );

        let edits = channel.edits.lock().await;
        assert_eq!(
            edits.len(),
            3,
            "activity edit + checklist edit + the kept card settled without the turn's step"
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

    /// Two runs a turn left running share its one card. When the first
    /// ends, the card shows it ended and the other still running; when the
    /// last ends, nothing on the card is running: the plan step the turn
    /// left open settles as not done, and the card settles to its final
    /// state (live 2026-11-13: the card stayed "Working · step 4 of 4" with
    /// two "Running in the background" rows 14 minutes after both ended).
    #[tokio::test]
    async fn a_card_settles_once_nothing_on_it_runs() {
        use aidaemon_runtime::background_run::{BackgroundRun, RunHome, RunStart};
        use aidaemon_runtime::run_state::RunPhase;
        let session = "telegram:synthetic-two-runs";
        let chat = crate::testing::CardChat::new(session).await;
        let db = tempfile::NamedTempFile::new().unwrap();
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .connect(&format!("sqlite://{}?mode=rwc", db.path().display()))
            .await
            .unwrap();
        let plans = Arc::new(crate::plans::PlanStore::new(pool).await.unwrap());
        let plan = plans
            .upsert_checklist(
                session,
                Some("turn-1"),
                "test",
                &[
                    ("Install site one".to_string(), crate::plans::StepStatus::InProgress),
                    ("Install site two".to_string(), crate::plans::StepStatus::Pending),
                    ("Report each time".to_string(), crate::plans::StepStatus::Pending),
                ],
            )
            .await
            .unwrap();
        let home = RunHome {
            runs: Arc::default(),
            hub: Some(chat.outbound()),
            plans: Some(plans.clone()),
        };
        let (first, second) = chat
            .turn(
                |tx| {
                    let home = home.clone();
                    let plans = plans.clone();
                    let plan_id = plan.id.clone();
                    async move {
                        if let Ok(Some(plan)) = plans.get(&plan_id).await {
                            let _ = tx.try_send(crate::types::StatusUpdate::Checklist(plan.card()));
                        }
                        let start = |id: &'static str, label: &'static str| RunStart {
                            id,
                            chat: session,
                            label,
                            task_id: Some("turn-1"),
                            status_tx: Some(tx.clone()),
                            work_began: chrono::Utc::now(),
                        };
                        let first =
                            BackgroundRun::start(home.clone(), start("job-1", "Install site one"))
                                .await;
                        let second =
                            BackgroundRun::start(home.clone(), start("job-2", "Install site two"))
                                .await;
                        (first, second)
                    }
                },
                |_| "Started in the background: Install site one.".to_string(),
            )
            .await;

        let (_, on_card) = first.finish(RunPhase::Succeeded, Default::default()).await;
        assert!(on_card);
        let card = chat.shown().await[0].clone();
        assert!(card.contains("✅ Finished in the background: Install site one"), "{card}");
        assert!(card.contains("⏳ Running in the background: Install site two"), "{card}");

        let (_, on_card) = second.finish(RunPhase::Succeeded, Default::default()).await;
        assert!(on_card);
        let shown = chat.shown().await;
        let card = &shown[0];
        assert!(card.contains("✅ Finished in the background: Install site one"), "{card}");
        assert!(card.contains("✅ Finished in the background: Install site two"), "{card}");
        assert!(!card.contains("Running"), "{card}");
        assert!(!card.contains("Working"), "{card}");
        assert!(card.contains("Report each time"), "{card}");
        assert!(card.contains("Report each time (not done)"), "{card}");
        let cards = shown
            .iter()
            .filter(|text| text.contains("in the background:") && !text.starts_with("Started"))
            .count();
        assert_eq!(cards, 1, "{shown:#?}");
        assert!(home.runs.for_chat(session).is_empty(), "records let go once settled");
    }
}
