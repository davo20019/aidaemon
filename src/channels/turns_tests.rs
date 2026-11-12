//! The turn runner on the daemon's agent: what only a real agent shows.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::turns::{status_line, Turn, TurnRunner, TurnSurface};
use super::ChannelRuntimeDeps;
use crate::testing::{setup_test_agent, MockProvider};
use crate::types::{ChannelContext, StatusUpdate, UserRole};

/// Records what the runner says.
#[derive(Default)]
struct Recording(std::sync::Mutex<Vec<String>>);

#[async_trait]
impl TurnSurface for Recording {
    fn platform(&self) -> &'static str {
        "test"
    }
    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        self.0.lock().unwrap().push(text.to_string());
        Ok(Vec::new())
    }
    fn show_working(&self, _: CancellationToken, _: Arc<AtomicU64>, _: u64) {}
}

fn turn(text: &str, key: &str) -> Turn {
    Turn {
        session_id: "test:wall-clock".to_string(),
        text: text.to_string(),
        attachments: Vec::new(),
        user_role: UserRole::Owner,
        channel_ctx: ChannelContext::private("test"),
        dedup_key: Some(key.to_string()),
        ingress_timing: None,
    }
}

#[tokio::test]
async fn stop_ends_finite_work_but_keeps_a_recurring_goal() {
    use crate::traits::store_prelude::*;
    let harness = setup_test_agent(MockProvider::new()).await.unwrap();
    let session = "test:stop";
    let recurring =
        crate::traits::Goal::new_continuous("Publish a daily blog post", session, None, None);
    let finite = crate::traits::Goal::new_finite("Answer the current question", session);
    harness.state.create_goal(&recurring).await.unwrap();
    harness.state.create_goal(&finite).await.unwrap();
    let inbox = tempfile::tempdir().unwrap().keep();
    let state = harness.state.clone();
    let runner = TurnRunner::new(
        ChannelRuntimeDeps {
            agent: Arc::new(harness.agent),
            config_path: inbox.join("config.toml"),
            session_map: Arc::default(),
            task_registry: Arc::new(crate::tasks::TaskRegistry::new(50)),
            files_enabled: false,
            inbox_dir: inbox,
            max_file_size_mb: 10,
            state: state.clone(),
            watchdog_stale_threshold_secs: 0,
        }
        .turn_deps(),
    );
    let surface = Arc::new(Recording::default());
    let mut stop = turn("stop", "m1");
    stop.session_id = session.to_string();
    runner.submit(surface.clone(), stop).await;

    let said = surface.0.lock().unwrap().clone();
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(said[0].contains("Answer the current question"), "{said:?}");
    assert!(!said[0].contains("daily blog"), "{said:?}");
    let status = |goal: &crate::traits::Goal| {
        let state = state.clone();
        let id = goal.id.clone();
        async move { state.get_goal(&id).await.unwrap().unwrap().status }
    };
    assert_eq!(status(&finite).await, "cancelled");
    assert_ne!(status(&recurring).await, "cancelled");
}

/// A surface whose status display is slow, as a loaded Telegram edit is:
/// each update shows a while after it arrives.
#[derive(Default)]
struct SlowStatus {
    log: Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait]
impl TurnSurface for SlowStatus {
    fn platform(&self) -> &'static str {
        "test"
    }
    async fn say(&self, text: &str) -> anyhow::Result<Vec<String>> {
        self.log.lock().unwrap().push(format!("say:{text}"));
        Ok(vec!["1".to_string()])
    }
    fn show_working(&self, _: CancellationToken, _: Arc<AtomicU64>, _: u64) {}
    fn status_settles_before_reply(&self) -> bool {
        true
    }
    fn show_status(
        &self,
        mut updates: mpsc::Receiver<StatusUpdate>,
        _private: bool,
    ) -> Result<JoinHandle<()>, mpsc::Receiver<StatusUpdate>> {
        let log = self.log.clone();
        Ok(tokio::spawn(async move {
            while let Some(update) = updates.recv().await {
                log.lock().unwrap().push("status-received".to_string());
                tokio::time::sleep(Duration::from_millis(150)).await;
                log.lock()
                    .unwrap()
                    .push(format!("status-shown:{}", status_line(&update)));
            }
        }))
    }
}

#[tokio::test]
async fn status_sent_before_the_reply_is_shown_before_the_reply() {
    let tool = Arc::new(crate::testing::MockTool::new(
        "lookup_widget",
        "Look up a synthetic widget",
        "widget-7 is blue",
    ));
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("lookup_widget", "{}"),
        MockProvider::text_response("Widget-7 is blue."),
    ]);
    let harness = crate::testing::setup_test_agent_with_extra_tools_and_llm_timeout(
        provider,
        vec![tool as Arc<dyn crate::traits::Tool>],
        None,
    )
    .await
    .unwrap();
    let inbox = tempfile::tempdir().unwrap().keep();
    let runner = TurnRunner::new(
        ChannelRuntimeDeps {
            agent: Arc::new(harness.agent),
            config_path: inbox.join("config.toml"),
            session_map: Arc::default(),
            task_registry: Arc::new(crate::tasks::TaskRegistry::new(50)),
            files_enabled: false,
            inbox_dir: inbox,
            max_file_size_mb: 10,
            state: harness.state.clone(),
            watchdog_stale_threshold_secs: 0,
        }
        .turn_deps(),
    );
    let surface = Arc::new(SlowStatus::default());
    runner
        .clone()
        .run(surface.clone(), turn("What colour is widget-7?", "m1"))
        .await;

    let log = surface.log.lock().unwrap().clone();
    let reply = log
        .iter()
        .position(|entry| entry.starts_with("say:") && entry.contains("blue"))
        .unwrap_or_else(|| panic!("no reply in {log:#?}"));
    let received = log[..reply]
        .iter()
        .filter(|entry| *entry == "status-received")
        .count();
    let shown = log[..reply]
        .iter()
        .filter(|entry| entry.starts_with("status-shown:"))
        .count();
    assert!(received > 0, "the turn sent no status: {log:#?}");
    assert_eq!(
        shown, received,
        "every status sent during the turn must show before the reply: {log:#?}"
    );
}
