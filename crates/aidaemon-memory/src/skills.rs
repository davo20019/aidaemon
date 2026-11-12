//! Skills a host keeps in its database: drafts promoted from successful
//! procedures, pending the owner's review, and the deprecated dynamic skills
//! kept for migration compatibility.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::Procedure;

/// A dynamically added skill (stored in database).
/// Deprecated: kept for migration compatibility. Use filesystem skills instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicSkill {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub triggers_json: String, // JSON array of trigger strings
    pub body: String,
    pub source: String, // "url", "inline", "auto", "registry"
    pub source_url: Option<String>,
    pub enabled: bool,
    pub version: Option<String>,
    pub created_at: String,
    /// JSON-serialized Vec<ResourceEntry> for directory-based skills
    #[serde(default = "default_empty_json")]
    pub resources_json: String,
}

/// A skill draft created by auto-promotion, pending user review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDraft {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub triggers_json: String,
    pub body: String,
    pub source_procedure: String,
    pub status: String, // "pending", "approved", "dismissed"
    pub created_at: String,
}

fn default_empty_json() -> String {
    "[]".to_string()
}

/// Skills storage (deprecated dynamic skills + skill drafts).
#[async_trait]
pub trait SkillStore: Send + Sync {
    /// Store a dynamically added skill.
    /// Deprecated: use filesystem skills instead.
    #[allow(dead_code)]
    async fn add_dynamic_skill(&self, _skill: &DynamicSkill) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all dynamic skills.
    /// Deprecated: use filesystem skills instead.
    async fn get_dynamic_skills(&self) -> anyhow::Result<Vec<DynamicSkill>> {
        Ok(vec![])
    }

    /// Delete a dynamic skill by ID.
    /// Deprecated: use filesystem skills instead.
    #[allow(dead_code)]
    async fn delete_dynamic_skill(&self, _id: i64) -> anyhow::Result<()> {
        Ok(())
    }

    /// Update the enabled flag of a dynamic skill.
    /// Deprecated: file existence = active, no enable/disable needed.
    #[allow(dead_code)]
    async fn update_dynamic_skill_enabled(&self, _id: i64, _enabled: bool) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get procedures eligible for skill promotion (success_count >= min_success, success rate >= min_rate).
    async fn get_promotable_procedures(
        &self,
        _min_success: i32,
        _min_rate: f32,
    ) -> anyhow::Result<Vec<Procedure>> {
        Ok(vec![])
    }

    /// Store a skill draft from auto-promotion. Returns the draft ID.
    async fn add_skill_draft(&self, _draft: &SkillDraft) -> anyhow::Result<i64> {
        Ok(0)
    }

    /// Get all pending skill drafts.
    async fn get_pending_skill_drafts(&self) -> anyhow::Result<Vec<SkillDraft>> {
        Ok(vec![])
    }

    /// Get a skill draft by ID.
    async fn get_skill_draft(&self, _id: i64) -> anyhow::Result<Option<SkillDraft>> {
        Ok(None)
    }

    /// Update a skill draft's status ("approved" or "dismissed").
    async fn update_skill_draft_status(&self, _id: i64, _status: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Check if any draft record already exists for a given procedure name
    /// (pending, approved, or dismissed).
    async fn skill_draft_exists_for_procedure(
        &self,
        _procedure_name: &str,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }
}
