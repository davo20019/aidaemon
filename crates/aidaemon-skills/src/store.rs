//! What the skills feature keeps in a host's database: skill drafts awaiting
//! the owner's review (and the deprecated dynamic skills), plus settings.

use std::collections::HashMap;
use std::sync::Mutex;

use aidaemon_memory::skills::{SkillDraft, SkillStore};
use aidaemon_tools::services::SettingsStore;
use async_trait::async_trait;

/// The store `manage_skills` needs: skill drafts and settings.
pub trait SkillsStore: SkillStore + SettingsStore {}

impl<T> SkillsStore for T where T: SkillStore + SettingsStore {}

/// Skill drafts and settings held in memory, for tests and hosts without a
/// database. Drafts are numbered from 1 in insertion order.
#[derive(Default)]
pub struct InMemorySkillStore {
    drafts: Mutex<Vec<SkillDraft>>,
    settings: Mutex<HashMap<String, String>>,
}

impl InMemorySkillStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SkillStore for InMemorySkillStore {
    async fn add_skill_draft(&self, draft: &SkillDraft) -> anyhow::Result<i64> {
        let mut drafts = self.drafts.lock().unwrap_or_else(|e| e.into_inner());
        let id = drafts.len() as i64 + 1;
        drafts.push(SkillDraft {
            id,
            status: "pending".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            ..draft.clone()
        });
        Ok(id)
    }

    async fn get_pending_skill_drafts(&self) -> anyhow::Result<Vec<SkillDraft>> {
        let drafts = self.drafts.lock().unwrap_or_else(|e| e.into_inner());
        Ok(drafts
            .iter()
            .filter(|draft| draft.status == "pending")
            .cloned()
            .collect())
    }

    async fn get_skill_draft(&self, id: i64) -> anyhow::Result<Option<SkillDraft>> {
        let drafts = self.drafts.lock().unwrap_or_else(|e| e.into_inner());
        Ok(drafts.iter().find(|draft| draft.id == id).cloned())
    }

    async fn update_skill_draft_status(&self, id: i64, status: &str) -> anyhow::Result<()> {
        let mut drafts = self.drafts.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(draft) = drafts.iter_mut().find(|draft| draft.id == id) {
            draft.status = status.to_string();
        }
        Ok(())
    }

    async fn skill_draft_exists_for_procedure(&self, procedure_name: &str) -> anyhow::Result<bool> {
        let drafts = self.drafts.lock().unwrap_or_else(|e| e.into_inner());
        Ok(drafts
            .iter()
            .any(|draft| draft.source_procedure == procedure_name))
    }
}

#[async_trait]
impl SettingsStore for InMemorySkillStore {
    async fn get_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
        let settings = self.settings.lock().unwrap_or_else(|e| e.into_inner());
        Ok(settings.get(key).cloned())
    }

    async fn set_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
        let mut settings = self.settings.lock().unwrap_or_else(|e| e.into_inner());
        settings.insert(key.to_string(), value.to_string());
        Ok(())
    }
}
