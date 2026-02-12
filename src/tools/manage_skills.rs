use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::skills::{self, Skill};
use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::skill_registry;
use crate::tools::terminal::ApprovalRequest;
use crate::tools::web_fetch::{build_browser_client, validate_url_for_ssrf};
use crate::traits::{StateStore, Tool};
use crate::types::ApprovalResponse;

pub struct ManageSkillsTool {
    skills_dir: PathBuf,
    state: Arc<dyn StateStore>,
    #[allow(dead_code)] // Available for future approval flow on add/install
    approval_tx: mpsc::Sender<ApprovalRequest>,
    client: reqwest::Client,
    /// Configured registry URLs for browse/install.
    registry_urls: Vec<String>,
}

impl ManageSkillsTool {
    pub fn new(
        skills_dir: PathBuf,
        state: Arc<dyn StateStore>,
        approval_tx: mpsc::Sender<ApprovalRequest>,
    ) -> Self {
        Self {
            skills_dir,
            state,
            approval_tx,
            client: build_browser_client(),
            registry_urls: Vec::new(),
        }
    }

    pub fn with_registries(mut self, registries: Vec<String>) -> Self {
        self.registry_urls = registries;
        self
    }

    #[allow(dead_code)] // Available for future approval flow on add/install
    async fn request_approval(
        &self,
        session_id: &str,
        description: &str,
    ) -> anyhow::Result<ApprovalResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: description.to_string(),
                session_id: session_id.to_string(),
                risk_level: RiskLevel::Medium,
                warnings: vec![
                    "This will add a new skill that can influence AI behavior".to_string()
                ],
                permission_mode: PermissionMode::Default,
                response_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Approval channel closed"))?;
        match tokio::time::timeout(std::time::Duration::from_secs(300), response_rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                tracing::warn!(description, "Approval response channel closed");
                Ok(ApprovalResponse::Deny)
            }
            Err(_) => {
                tracing::warn!(
                    description,
                    "Approval request timed out (300s), auto-denying"
                );
                Ok(ApprovalResponse::Deny)
            }
        }
    }

    /// Write a skill to the filesystem, checking for duplicate names.
    async fn persist_to_filesystem(&self, skill: Skill) -> anyhow::Result<String> {
        let existing = skills::load_skills(&self.skills_dir);
        if skills::find_skill_by_name(&existing, &skill.name).is_some() {
            return Ok(format!(
                "A skill named '{}' already exists. Remove it first or choose a different name.",
                skill.name
            ));
        }

        let name = skill.name.clone();
        let desc = skill.description.clone();
        let path = skills::write_skill_to_file(&self.skills_dir, &skill)?;

        info!(name = %name, path = %path.display(), "Skill added to filesystem");
        Ok(format!(
            "Skill '{}' added and saved to {}. Description: {}",
            name,
            path.display(),
            desc
        ))
    }

    async fn handle_add_url(&self, url: &str) -> anyhow::Result<String> {
        // SSRF validation
        validate_url_for_ssrf(url).map_err(|e| anyhow::anyhow!("URL blocked: {}", e))?;

        // Fetch content
        let response = self.client.get(url).send().await?;
        if !response.status().is_success() {
            return Ok(format!("Failed to fetch URL: HTTP {}", response.status()));
        }

        let content = response.text().await?;
        let skill = match Skill::parse(&content) {
            Some(mut s) => {
                s.source = Some("url".to_string());
                s.source_url = Some(url.to_string());
                s
            }
            None => return Ok("Failed to parse skill from URL. The content must be valid skill markdown with --- frontmatter (name, description, triggers) and a body.".to_string()),
        };

        self.persist_to_filesystem(skill).await
    }

    async fn handle_add_inline(&self, content: &str) -> anyhow::Result<String> {
        let skill = match Skill::parse(content) {
            Some(mut s) => {
                s.source = Some("inline".to_string());
                s
            }
            None => return Ok("Failed to parse skill. Expected markdown with --- frontmatter containing name, description, triggers fields, followed by the skill body.".to_string()),
        };

        self.persist_to_filesystem(skill).await
    }

    async fn handle_list(&self) -> anyhow::Result<String> {
        let mut all_skills = skills::load_skills(&self.skills_dir);
        if all_skills.is_empty() {
            return Ok("No skills loaded.".to_string());
        }

        all_skills.sort_by(|a, b| a.name.cmp(&b.name));

        let mut output = format!("**{} skills:**\n", all_skills.len());
        for skill in &all_skills {
            let source = skill.source.as_deref().unwrap_or("filesystem");
            output.push_str(&format!(
                "- **{}**: {} [source: {}]\n",
                skill.name, skill.description, source
            ));
            if !skill.triggers.is_empty() {
                output.push_str(&format!("  triggers: {}\n", skill.triggers.join(", ")));
            }
            if !skill.resources.is_empty() {
                let mut by_category: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::new();
                for r in &skill.resources {
                    *by_category.entry(r.category.as_str()).or_insert(0) += 1;
                }
                let summary: Vec<String> = by_category
                    .iter()
                    .map(|(k, v)| format!("{} {}(s)", v, k))
                    .collect();
                output.push_str(&format!("  resources: {}\n", summary.join(", ")));
            }
        }
        Ok(output)
    }

    async fn handle_remove(&self, name: &str) -> anyhow::Result<String> {
        match skills::remove_skill_file(&self.skills_dir, name)? {
            true => {
                info!(name = %name, "Skill removed from filesystem");
                Ok(format!("Skill '{}' removed.", name))
            }
            false => Ok(format!("Skill '{}' not found.", name)),
        }
    }

    async fn handle_browse(&self, query: Option<&str>) -> anyhow::Result<String> {
        if self.registry_urls.is_empty() {
            return Ok("No skill registries configured. Add registry URLs to [skills.registries] in config.toml.".to_string());
        }

        let mut all_entries = Vec::new();
        for url in &self.registry_urls {
            match skill_registry::fetch_registry(&self.client, url).await {
                Ok(entries) => all_entries.extend(entries),
                Err(e) => {
                    warn!(url = %url, error = %e, "Failed to fetch registry");
                }
            }
        }

        if let Some(q) = query {
            let filtered = skill_registry::search_registry(&all_entries, q);
            let owned: Vec<_> = filtered.into_iter().cloned().collect();
            Ok(skill_registry::format_registry_listing(&owned))
        } else {
            Ok(skill_registry::format_registry_listing(&all_entries))
        }
    }

    async fn handle_install(&self, name: &str) -> anyhow::Result<String> {
        if self.registry_urls.is_empty() {
            return Ok("No skill registries configured.".to_string());
        }

        // Find the entry across all registries
        let mut target_entry = None;
        for url in &self.registry_urls {
            match skill_registry::fetch_registry(&self.client, url).await {
                Ok(entries) => {
                    if let Some(entry) = entries.into_iter().find(|e| e.name == name) {
                        target_entry = Some(entry);
                        break;
                    }
                }
                Err(e) => {
                    warn!(url = %url, error = %e, "Failed to fetch registry for install");
                }
            }
        }

        let entry = match target_entry {
            Some(e) => e,
            None => {
                return Ok(format!(
                    "Skill '{}' not found in any configured registry.",
                    name
                ))
            }
        };

        // Fetch the skill content
        let content = skill_registry::fetch_skill_content(&self.client, &entry).await?;
        let skill = match Skill::parse(&content) {
            Some(mut s) => {
                s.source = Some("registry".to_string());
                s.source_url = Some(entry.url.clone());
                s
            }
            None => {
                return Ok(format!(
                    "Failed to parse skill '{}' from registry URL.",
                    name
                ))
            }
        };

        self.persist_to_filesystem(skill).await
    }

    async fn handle_update(&self, name: &str) -> anyhow::Result<String> {
        // Find the existing skill on disk
        let all_skills = skills::load_skills(&self.skills_dir);
        let existing = skills::find_skill_by_name(&all_skills, name);

        let existing = match existing {
            Some(s) => s,
            None => return Ok(format!("Skill '{}' not found.", name)),
        };

        let source_url = match &existing.source_url {
            Some(url) => url.clone(),
            None => {
                return Ok(format!(
                    "Skill '{}' has no source URL and cannot be updated.",
                    name
                ))
            }
        };

        // Re-fetch from source URL
        validate_url_for_ssrf(&source_url).map_err(|e| anyhow::anyhow!("URL blocked: {}", e))?;
        let response = self.client.get(&source_url).send().await?;
        if !response.status().is_success() {
            return Ok(format!(
                "Failed to fetch skill update: HTTP {}",
                response.status()
            ));
        }

        let content = response.text().await?;
        let new_skill = match Skill::parse(&content) {
            Some(s) => s,
            None => return Ok("Failed to parse updated skill content.".to_string()),
        };

        // Remove old file and write new one
        skills::remove_skill_file(&self.skills_dir, name)?;

        let mut skill = new_skill;
        skill.source = existing.source.clone();
        skill.source_url = Some(source_url);
        let path = skills::write_skill_to_file(&self.skills_dir, &skill)?;

        info!(name = %name, path = %path.display(), "Skill updated from source");
        Ok(format!(
            "Skill '{}' updated from source. Saved to {}.",
            name,
            path.display()
        ))
    }

    async fn handle_review(
        &self,
        draft_id: Option<i64>,
        approve: Option<bool>,
    ) -> anyhow::Result<String> {
        // If a draft_id is given with an approve/dismiss decision
        if let Some(id) = draft_id {
            let draft = match self.state.get_skill_draft(id).await? {
                Some(d) => d,
                None => return Ok(format!("Skill draft #{} not found.", id)),
            };

            if draft.status != "pending" {
                return Ok(format!(
                    "Skill draft #{} has already been {} and cannot be changed.",
                    id, draft.status
                ));
            }

            if approve == Some(true) {
                // Parse draft into Skill and write to filesystem
                let triggers: Vec<String> =
                    serde_json::from_str(&draft.triggers_json).unwrap_or_default();
                let skill = Skill {
                    name: draft.name.clone(),
                    description: draft.description.clone(),
                    triggers,
                    body: draft.body.clone(),
                    source: Some("auto".to_string()),
                    source_url: None,
                    dir_path: None,
                    resources: vec![],
                };

                // Check for duplicates
                let existing = skills::load_skills(&self.skills_dir);
                if skills::find_skill_by_name(&existing, &skill.name).is_some() {
                    return Ok(format!(
                        "A skill named '{}' already exists on disk. Dismiss this draft or remove the existing skill first.",
                        skill.name
                    ));
                }

                let path = skills::write_skill_to_file(&self.skills_dir, &skill)?;
                self.state.update_skill_draft_status(id, "approved").await?;
                info!(name = %draft.name, path = %path.display(), "Skill draft approved and written to filesystem");
                Ok(format!(
                    "Skill draft #{} '{}' approved and saved to {}.",
                    id,
                    draft.name,
                    path.display()
                ))
            } else {
                self.state
                    .update_skill_draft_status(id, "dismissed")
                    .await?;
                info!(name = %draft.name, id, "Skill draft dismissed");
                Ok(format!("Skill draft #{} '{}' dismissed.", id, draft.name))
            }
        } else {
            // List all pending drafts
            let drafts = self.state.get_pending_skill_drafts().await?;
            if drafts.is_empty() {
                return Ok("No pending skill drafts.".to_string());
            }

            let mut output = format!("**{} pending skill draft(s):**\n", drafts.len());
            for draft in &drafts {
                output.push_str(&format!(
                    "- **#{}** '{}': {} (from procedure: '{}', created: {})\n",
                    draft.id,
                    draft.name,
                    draft.description,
                    draft.source_procedure,
                    draft.created_at
                ));
                let triggers: Vec<String> =
                    serde_json::from_str(&draft.triggers_json).unwrap_or_default();
                if !triggers.is_empty() {
                    output.push_str(&format!("  triggers: {}\n", triggers.join(", ")));
                }
            }
            output.push_str(
                "\nUse `manage_skills review` with `draft_id` and `approve: true/false` to approve or dismiss.",
            );
            Ok(output)
        }
    }
}

#[derive(Deserialize)]
struct ManageSkillsArgs {
    action: String,
    url: Option<String>,
    content: Option<String>,
    name: Option<String>,
    query: Option<String>,
    draft_id: Option<i64>,
    approve: Option<bool>,
}

#[async_trait]
impl Tool for ManageSkillsTool {
    fn name(&self) -> &str {
        "manage_skills"
    }

    fn description(&self) -> &str {
        "Manage skills at runtime. Actions: add (from URL), add_inline (raw markdown), list, remove, browse (search registries), install (from registry), update (re-fetch from source), review (approve/dismiss auto-promoted skill drafts)."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_skills",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "add_inline", "list", "remove", "browse", "install", "update", "review"],
                        "description": "The action to perform"
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to fetch skill from (for 'add' action)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Raw skill markdown content (for 'add_inline' action)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name (for 'remove', 'install', 'update' actions)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for 'browse' action)"
                    },
                    "draft_id": {
                        "type": "integer",
                        "description": "Skill draft ID to approve or dismiss (for 'review' action)"
                    },
                    "approve": {
                        "type": "boolean",
                        "description": "true to approve a draft, false to dismiss (for 'review' action with draft_id)"
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageSkillsArgs = serde_json::from_str(arguments)
            .map_err(|e| anyhow::anyhow!("Invalid arguments: {}", e))?;

        match args.action.as_str() {
            "add" => {
                let url = args.url.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'url' parameter required for 'add' action"))?;
                self.handle_add_url(url).await
            }
            "add_inline" => {
                let content = args.content.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'content' parameter required for 'add_inline' action"))?;
                self.handle_add_inline(content).await
            }
            "list" => self.handle_list().await,
            "remove" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'remove' action"))?;
                self.handle_remove(name).await
            }
            "browse" => {
                self.handle_browse(args.query.as_deref()).await
            }
            "install" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'install' action"))?;
                self.handle_install(name).await
            }
            "update" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'update' action"))?;
                self.handle_update(name).await
            }
            "review" => {
                self.handle_review(args.draft_id, args.approve).await
            }
            other => Ok(format!("Unknown action '{}'. Valid actions: add, add_inline, list, remove, browse, install, update, review", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_skill_markdown() {
        let content = "---\nname: deploy\ndescription: Deploy the app\ntriggers: deploy, ship\n---\nRun cargo build --release && deploy.sh";
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name, "deploy");
        assert_eq!(skill.description, "Deploy the app");
        assert_eq!(skill.triggers, vec!["deploy", "ship"]);
        assert!(skill.body.contains("cargo build"));
    }

    #[test]
    fn parse_invalid_skill_markdown() {
        assert!(Skill::parse("no frontmatter here").is_none());
        assert!(Skill::parse("---\ndescription: no name\n---\nbody").is_none());
    }

    #[test]
    fn ssrf_rejection() {
        assert!(validate_url_for_ssrf("http://localhost/evil").is_err());
        assert!(validate_url_for_ssrf("http://127.0.0.1/evil").is_err());
        assert!(validate_url_for_ssrf("http://169.254.169.254/metadata").is_err());
        assert!(validate_url_for_ssrf("ftp://example.com/file").is_err());
    }

    #[test]
    fn ssrf_valid_urls() {
        assert!(
            validate_url_for_ssrf("https://raw.githubusercontent.com/user/repo/main/skill.md")
                .is_ok()
        );
        assert!(validate_url_for_ssrf("https://example.com/skills/deploy.md").is_ok());
    }
}
