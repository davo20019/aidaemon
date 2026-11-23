//! Background runs' answers owed to their chats.
//!
//! When a background run ends, its chat's answer is written by a
//! continuation turn. A stop between the run's end and that answer (a
//! restart in the seconds before the turn kept its first checkpoint) left
//! the run's chat with nothing on record to resume: the work was done and
//! no one said so. Each answer is noted as owed before its turn starts and
//! cleared once it is given; the next boot gives every one still owed
//! (`startup::resume_turns`).

use aidaemon_tools::services::SettingsStore;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// The settings key the owed answers are kept under.
const KEY: &str = "runtime.owed_run_reports";

/// One owed answer: the continuation that gives it, as it was to run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwedReport {
    /// The run whose answer it is (one owed answer per run).
    pub run: String,
    pub chat: String,
    /// The turn's task the continuation continues.
    pub parent_task_id: Option<String>,
    /// The continuation's message.
    pub followup: String,
}

static HELD: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn read(store: &dyn SettingsStore) -> Vec<OwedReport> {
    match store.get_setting(KEY).await {
        Ok(Some(raw)) => serde_json::from_str(&raw).unwrap_or_default(),
        Ok(None) => Vec::new(),
        Err(error) => {
            warn!(%error, "Could not read the answers owed to chats");
            Vec::new()
        }
    }
}

async fn write(store: &dyn SettingsStore, owed: &[OwedReport]) {
    let Ok(raw) = serde_json::to_string(owed) else {
        return;
    };
    if let Err(error) = store.set_setting(KEY, &raw).await {
        warn!(%error, "Could not keep the answers owed to chats");
    }
}

/// Note `report` as owed (replacing an earlier note for its run).
pub async fn owe(store: &dyn SettingsStore, report: OwedReport) {
    let _held = HELD.lock().await;
    let mut owed = read(store).await;
    owed.retain(|other| other.run != report.run);
    owed.push(report);
    write(store, &owed).await;
}

/// Run `run`'s answer was given (or will not be): no longer owed.
pub async fn paid(store: &dyn SettingsStore, run: &str) {
    let _held = HELD.lock().await;
    let mut owed = read(store).await;
    let before = owed.len();
    owed.retain(|other| other.run != run);
    if owed.len() != before {
        write(store, &owed).await;
    }
}

/// Every answer still owed, cleared: the caller gives them once.
pub async fn take_all(store: &dyn SettingsStore) -> Vec<OwedReport> {
    let _held = HELD.lock().await;
    let owed = read(store).await;
    if !owed.is_empty() {
        write(store, &[]).await;
    }
    owed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Settings(std::sync::Mutex<std::collections::HashMap<String, String>>);

    #[async_trait::async_trait]
    impl SettingsStore for Settings {
        async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }
        async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(key.to_string(), value.to_string());
            Ok(())
        }
    }

    fn report(run: &str) -> OwedReport {
        OwedReport {
            run: run.into(),
            chat: "app:synthetic".into(),
            parent_task_id: Some("task-1".into()),
            followup: format!("[Background command completed] {run}"),
        }
    }

    #[tokio::test]
    async fn an_answer_stays_owed_until_it_is_given() {
        let store = Settings::default();
        owe(&store, report("cli-agent-a")).await;
        owe(&store, report("cli-agent-b")).await;
        paid(&store, "cli-agent-a").await;
        // The stop came before b's answer: the boot gives it, once.
        assert_eq!(take_all(&store).await, vec![report("cli-agent-b")]);
        assert!(take_all(&store).await.is_empty());
    }
}
