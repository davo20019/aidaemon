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
use crate::traits::{StateStore, Tool, ToolCapabilities};
use crate::types::ApprovalResponse;

pub struct ManageSkillsTool {
    skills_dir: PathBuf,
    state: Arc<dyn StateStore>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
    client: reqwest::Client,
    /// Configured registry URLs for browse/install.
    registry_urls: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoveOutcomeKind {
    Removed,
    DraftsOnly,
    NotFound,
    Ambiguous,
}

#[derive(Debug, Clone)]
struct RemoveOutcome {
    kind: RemoveOutcomeKind,
    requested: String,
    target_name: String,
    dismissed_draft_ids: Vec<i64>,
    ambiguous_candidates: Vec<String>,
    available_skills: Vec<String>,
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
                kind: Default::default(),
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
        let existing = skills::load_skills_with_status(&self.skills_dir);
        if let Some(conflicting_name) = Self::find_conflicting_skill(&existing, &skill.name) {
            return Ok(format!(
                "Skill '{}' conflicts with existing skill '{}' (same canonical filename). Remove or rename the existing skill first.",
                skill.name,
                conflicting_name
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
                s.origin = Some(skills::SKILL_ORIGIN_CUSTOM.to_string());
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
                s.origin = Some(skills::SKILL_ORIGIN_CUSTOM.to_string());
                s.source = Some("inline".to_string());
                s
            }
            None => return Ok("Failed to parse skill. Expected markdown with --- frontmatter containing name, description, triggers fields, followed by the skill body.".to_string()),
        };

        self.persist_to_filesystem(skill).await
    }

    async fn handle_list(&self) -> anyhow::Result<String> {
        let mut all_skills = skills::load_skills_with_status(&self.skills_dir);
        if all_skills.is_empty() {
            return Ok("No skills loaded.".to_string());
        }

        all_skills.sort_by(|a, b| a.skill.name.cmp(&b.skill.name));

        let mut custom_count = 0usize;
        let mut contrib_count = 0usize;
        let mut enabled_count = 0usize;
        let mut disabled_count = 0usize;
        for skill in &all_skills {
            match skills::infer_skill_origin(
                skill.skill.origin.as_deref(),
                skill.skill.source.as_deref(),
            ) {
                skills::SKILL_ORIGIN_CONTRIB => contrib_count += 1,
                _ => custom_count += 1,
            }
            if skill.enabled {
                enabled_count += 1;
            } else {
                disabled_count += 1;
            }
        }

        let mut output = format!(
            "**{} skills:** (enabled: {}, disabled: {}, {}: {}, {}: {})\n",
            all_skills.len(),
            enabled_count,
            disabled_count,
            skills::SKILL_ORIGIN_CUSTOM,
            custom_count,
            skills::SKILL_ORIGIN_CONTRIB,
            contrib_count
        );
        for skill in &all_skills {
            let origin = skills::infer_skill_origin(
                skill.skill.origin.as_deref(),
                skill.skill.source.as_deref(),
            );
            let source = skill.skill.source.as_deref().unwrap_or("filesystem");
            let status = if skill.enabled { "enabled" } else { "disabled" };
            output.push_str(&format!(
                "- **{}**: {} [origin: {}] [source: {}] [status: {}]\n",
                skill.skill.name, skill.skill.description, origin, source, status
            ));
            if !skill.skill.triggers.is_empty() {
                output.push_str(&format!(
                    "  triggers: {}\n",
                    skill.skill.triggers.join(", ")
                ));
            }
            if !skill.skill.resources.is_empty() {
                let mut by_category: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::new();
                for r in &skill.skill.resources {
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

    fn find_conflicting_skill<'a>(
        all_skills: &'a [skills::SkillWithStatus],
        name: &str,
    ) -> Option<&'a str> {
        let normalized_target = skills::sanitize_skill_filename(name);
        all_skills
            .iter()
            .find(|skill| skills::sanitize_skill_filename(&skill.skill.name) == normalized_target)
            .map(|skill| skill.skill.name.as_str())
    }

    fn resolve_skill_name<'a>(
        all_skills: &'a [skills::SkillWithStatus],
        name: &str,
    ) -> Result<Option<&'a skills::SkillWithStatus>, Vec<String>> {
        if all_skills.is_empty() {
            return Ok(None);
        }

        // Exact case-sensitive match first.
        if let Some(found) = all_skills.iter().find(|s| s.skill.name == name) {
            return Ok(Some(found));
        }

        // Case-insensitive fallback.
        let mut matches: Vec<&skills::SkillWithStatus> = all_skills
            .iter()
            .filter(|s| s.skill.name.eq_ignore_ascii_case(name))
            .collect();
        if matches.len() > 1 {
            return Err(matches.into_iter().map(|s| s.skill.name.clone()).collect());
        }
        if let Some(found) = matches.pop() {
            return Ok(Some(found));
        }

        // Canonical filename-based fallback (handles spaces/underscores/hyphen variants).
        let normalized_query = skills::sanitize_skill_filename(name);
        let mut normalized_matches: Vec<&skills::SkillWithStatus> = all_skills
            .iter()
            .filter(|s| skills::sanitize_skill_filename(&s.skill.name) == normalized_query)
            .collect();
        if normalized_matches.len() > 1 {
            return Err(normalized_matches
                .into_iter()
                .map(|s| s.skill.name.clone())
                .collect());
        }
        if let Some(found) = normalized_matches.pop() {
            return Ok(Some(found));
        }

        Ok(None)
    }

    async fn matching_pending_draft_ids(&self, skill_name: &str) -> anyhow::Result<Vec<i64>> {
        let pending = self.state.get_pending_skill_drafts().await?;
        if pending.is_empty() {
            return Ok(Vec::new());
        }

        let normalized_target = skills::sanitize_skill_filename(skill_name);
        let mut dismissed_ids = Vec::new();
        for draft in pending {
            let name_matches = draft.name.eq_ignore_ascii_case(skill_name)
                || skills::sanitize_skill_filename(&draft.name) == normalized_target;
            if name_matches {
                dismissed_ids.push(draft.id);
            }
        }

        Ok(dismissed_ids)
    }

    async fn dismiss_pending_drafts(&self, draft_ids: &[i64]) -> anyhow::Result<()> {
        for id in draft_ids {
            self.state
                .update_skill_draft_status(*id, "dismissed")
                .await?;
        }
        Ok(())
    }

    async fn remove_skill_internal(
        &self,
        name: &str,
        dry_run: bool,
    ) -> anyhow::Result<RemoveOutcome> {
        let all_skills = skills::load_skills_with_status(&self.skills_dir);
        let mut available: Vec<String> = all_skills.iter().map(|s| s.skill.name.clone()).collect();
        available.sort();

        let resolved_skill = match Self::resolve_skill_name(&all_skills, name) {
            Ok(skill) => skill,
            Err(candidates) => {
                return Ok(RemoveOutcome {
                    kind: RemoveOutcomeKind::Ambiguous,
                    requested: name.to_string(),
                    target_name: name.to_string(),
                    dismissed_draft_ids: Vec::new(),
                    ambiguous_candidates: candidates,
                    available_skills: available,
                });
            }
        };

        let target_name = resolved_skill
            .map(|s| s.skill.name.clone())
            .unwrap_or_else(|| name.to_string());
        let dismissed_draft_ids = self.matching_pending_draft_ids(&target_name).await?;

        let removed = if dry_run {
            let sanitized = skills::sanitize_skill_filename(&target_name);
            let md_path = self.skills_dir.join(format!("{}.md", sanitized));
            let dir_path = self.skills_dir.join(&sanitized);
            resolved_skill.is_some() || md_path.exists() || dir_path.is_dir()
        } else {
            skills::remove_skill_file(&self.skills_dir, &target_name)?
        };

        if !dry_run && !dismissed_draft_ids.is_empty() {
            self.dismiss_pending_drafts(&dismissed_draft_ids).await?;
        }

        if removed {
            return Ok(RemoveOutcome {
                kind: RemoveOutcomeKind::Removed,
                requested: name.to_string(),
                target_name,
                dismissed_draft_ids,
                ambiguous_candidates: Vec::new(),
                available_skills: available,
            });
        }

        if !dismissed_draft_ids.is_empty() {
            return Ok(RemoveOutcome {
                kind: RemoveOutcomeKind::DraftsOnly,
                requested: name.to_string(),
                target_name,
                dismissed_draft_ids,
                ambiguous_candidates: Vec::new(),
                available_skills: available,
            });
        }

        Ok(RemoveOutcome {
            kind: RemoveOutcomeKind::NotFound,
            requested: name.to_string(),
            target_name,
            dismissed_draft_ids: Vec::new(),
            ambiguous_candidates: Vec::new(),
            available_skills: available,
        })
    }

    async fn handle_remove(&self, name: &str) -> anyhow::Result<String> {
        let outcome = self.remove_skill_internal(name, false).await?;

        match outcome.kind {
            RemoveOutcomeKind::Ambiguous => Ok(format!(
                "Skill name '{}' is ambiguous. Matches: {}. Use the exact skill name from `manage_skills list`.",
                outcome.requested,
                outcome.ambiguous_candidates.join(", ")
            )),
            RemoveOutcomeKind::Removed => {
                info!(name = %outcome.target_name, "Skill removed from filesystem");
                if outcome.dismissed_draft_ids.is_empty() {
                    Ok(format!("Skill '{}' removed.", outcome.target_name))
                } else {
                    let ids = outcome
                        .dismissed_draft_ids
                        .iter()
                        .map(|id| format!("#{}", id))
                        .collect::<Vec<String>>()
                        .join(", ");
                    Ok(format!(
                        "Skill '{}' removed. Dismissed {} pending draft(s): {}.",
                        outcome.target_name,
                        outcome.dismissed_draft_ids.len(),
                        ids
                    ))
                }
            }
            RemoveOutcomeKind::DraftsOnly => {
                let ids = outcome
                    .dismissed_draft_ids
                    .iter()
                    .map(|id| format!("#{}", id))
                    .collect::<Vec<String>>()
                    .join(", ");
                Ok(format!(
                    "Skill '{}' was not installed, but dismissed {} pending draft(s): {}.",
                    outcome.target_name,
                    outcome.dismissed_draft_ids.len(),
                    ids
                ))
            }
            RemoveOutcomeKind::NotFound => {
                if outcome.available_skills.is_empty() {
                    Ok(format!(
                        "Skill '{}' not found. No skills are currently loaded.",
                        outcome.requested
                    ))
                } else {
                    Ok(format!(
                        "Skill '{}' not found. Available skills: {}",
                        outcome.requested,
                        outcome.available_skills.join(", ")
                    ))
                }
            }
        }
    }

    async fn handle_remove_all(&self, names: &[String], dry_run: bool) -> anyhow::Result<String> {
        let mut requested: Vec<String> = Vec::new();
        for name in names {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !requested.iter().any(|n| n.eq_ignore_ascii_case(trimmed)) {
                requested.push(trimmed.to_string());
            }
        }

        if requested.is_empty() {
            return Ok("No valid skill names were provided.".to_string());
        }

        let mut removed = Vec::new();
        let mut drafts_only = Vec::new();
        let mut not_found = Vec::new();
        let mut ambiguous: Vec<(String, Vec<String>)> = Vec::new();
        let mut draft_ids: Vec<i64> = Vec::new();

        for name in &requested {
            let outcome = self.remove_skill_internal(name, dry_run).await?;
            draft_ids.extend(outcome.dismissed_draft_ids.iter().copied());
            match outcome.kind {
                RemoveOutcomeKind::Removed => removed.push(outcome.target_name),
                RemoveOutcomeKind::DraftsOnly => drafts_only.push(outcome.target_name),
                RemoveOutcomeKind::NotFound => not_found.push(outcome.requested),
                RemoveOutcomeKind::Ambiguous => {
                    ambiguous.push((outcome.requested, outcome.ambiguous_candidates))
                }
            }
        }

        removed.sort();
        removed.dedup();
        drafts_only.sort();
        drafts_only.dedup();
        not_found.sort();
        not_found.dedup();
        ambiguous.sort_by(|a, b| a.0.cmp(&b.0));
        draft_ids.sort_unstable();
        draft_ids.dedup();

        let mut output = if dry_run {
            format!(
                "Dry run for remove_all processed {} skill request(s):\n",
                requested.len()
            )
        } else {
            format!(
                "remove_all processed {} skill request(s):\n",
                requested.len()
            )
        };

        if !removed.is_empty() {
            if dry_run {
                output.push_str(&format!(
                    "- Would remove {}: {}\n",
                    removed.len(),
                    removed.join(", ")
                ));
            } else {
                output.push_str(&format!(
                    "- Removed {}: {}\n",
                    removed.len(),
                    removed.join(", ")
                ));
            }
        }
        if !drafts_only.is_empty() {
            output.push_str(&format!(
                "- No installed skill matched, but pending drafts matched {}: {}\n",
                drafts_only.len(),
                drafts_only.join(", ")
            ));
        }
        if !draft_ids.is_empty() {
            let ids = draft_ids
                .iter()
                .map(|id| format!("#{}", id))
                .collect::<Vec<String>>()
                .join(", ");
            if dry_run {
                output.push_str(&format!("- Would dismiss pending draft(s): {}\n", ids));
            } else {
                output.push_str(&format!("- Dismissed pending draft(s): {}\n", ids));
            }
        }
        if !not_found.is_empty() {
            output.push_str(&format!(
                "- Not found {}: {}\n",
                not_found.len(),
                not_found.join(", ")
            ));
        }
        for (name, candidates) in &ambiguous {
            output.push_str(&format!(
                "- Ambiguous '{}': {}. Use exact names from `manage_skills list`.\n",
                name,
                candidates.join(", ")
            ));
        }

        if removed.is_empty()
            && drafts_only.is_empty()
            && draft_ids.is_empty()
            && not_found.is_empty()
            && ambiguous.is_empty()
        {
            output.push_str("- No changes.");
        }

        Ok(output)
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
                s.origin = Some(skills::SKILL_ORIGIN_CONTRIB.to_string());
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
        let all_skills = skills::load_skills_with_status(&self.skills_dir);
        let existing = match Self::resolve_skill_name(&all_skills, name) {
            Ok(Some(s)) => s,
            Ok(None) => return Ok(format!("Skill '{}' not found.", name)),
            Err(candidates) => {
                return Ok(format!(
                    "Skill name '{}' is ambiguous. Matches: {}. Use the exact skill name from `manage_skills list`.",
                    name,
                    candidates.join(", ")
                ));
            }
        };

        let source_url = match &existing.skill.source_url {
            Some(url) => url.clone(),
            None => {
                return Ok(format!(
                    "Skill '{}' has no source URL and cannot be updated.",
                    existing.skill.name
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
        skills::remove_skill_file(&self.skills_dir, &existing.skill.name)?;

        let mut skill = new_skill;
        skill.origin = Some(
            skills::infer_skill_origin(
                existing.skill.origin.as_deref(),
                existing.skill.source.as_deref(),
            )
            .to_string(),
        );
        skill.source = existing.skill.source.clone();
        skill.source_url = Some(source_url);
        let path = skills::write_skill_to_file(&self.skills_dir, &skill)?;
        if !existing.enabled {
            let _ = skills::set_skill_enabled(&self.skills_dir, &skill.name, false)?;
        }

        info!(name = %name, path = %path.display(), "Skill updated from source");
        Ok(format!(
            "Skill '{}' updated from source. Saved to {}.",
            name,
            path.display()
        ))
    }

    async fn handle_enable(&self, name: &str) -> anyhow::Result<String> {
        let all_skills = skills::load_skills_with_status(&self.skills_dir);
        let target = match Self::resolve_skill_name(&all_skills, name) {
            Ok(Some(skill)) => skill,
            Ok(None) => return Ok(format!("Skill '{}' not found.", name)),
            Err(candidates) => {
                return Ok(format!(
                    "Skill name '{}' is ambiguous. Matches: {}. Use the exact skill name from `manage_skills list`.",
                    name,
                    candidates.join(", ")
                ));
            }
        };

        match skills::set_skill_enabled(&self.skills_dir, &target.skill.name, true)? {
            None => Ok(format!("Skill '{}' not found.", name)),
            Some(false) => Ok(format!("Skill '{}' is already enabled.", target.skill.name)),
            Some(true) => Ok(format!("Skill '{}' enabled.", target.skill.name)),
        }
    }

    async fn handle_disable(&self, name: &str) -> anyhow::Result<String> {
        let all_skills = skills::load_skills_with_status(&self.skills_dir);
        let target = match Self::resolve_skill_name(&all_skills, name) {
            Ok(Some(skill)) => skill,
            Ok(None) => return Ok(format!("Skill '{}' not found.", name)),
            Err(candidates) => {
                return Ok(format!(
                    "Skill name '{}' is ambiguous. Matches: {}. Use the exact skill name from `manage_skills list`.",
                    name,
                    candidates.join(", ")
                ));
            }
        };

        match skills::set_skill_enabled(&self.skills_dir, &target.skill.name, false)? {
            None => Ok(format!("Skill '{}' not found.", name)),
            Some(false) => Ok(format!(
                "Skill '{}' is already disabled.",
                target.skill.name
            )),
            Some(true) => Ok(format!("Skill '{}' disabled.", target.skill.name)),
        }
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

            let approve = match approve {
                Some(value) => value,
                None => {
                    return Ok(format!(
                        "Skill draft #{} requires `approve: true` to approve or `approve: false` to dismiss.",
                        id
                    ));
                }
            };

            if approve {
                // Parse draft into Skill and write to filesystem
                let triggers: Vec<String> =
                    serde_json::from_str(&draft.triggers_json).unwrap_or_default();
                let skill = Skill {
                    name: draft.name.clone(),
                    description: draft.description.clone(),
                    triggers,
                    body: draft.body.clone(),
                    origin: Some(skills::SKILL_ORIGIN_CUSTOM.to_string()),
                    source: Some("auto".to_string()),
                    source_url: None,
                    dir_path: None,
                    resources: vec![],
                };

                // Check for duplicates
                let existing = skills::load_skills_with_status(&self.skills_dir);
                if let Some(conflicting_name) = Self::find_conflicting_skill(&existing, &skill.name)
                {
                    return Ok(format!(
                        "Skill draft '{}' conflicts with existing skill '{}' (same canonical filename). Dismiss this draft or remove/rename the existing skill first.",
                        skill.name,
                        conflicting_name
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
    names: Option<Vec<String>>,
    query: Option<String>,
    draft_id: Option<i64>,
    approve: Option<bool>,
    dry_run: Option<bool>,
    #[serde(default)]
    _session_id: Option<String>,
}

#[async_trait]
impl Tool for ManageSkillsTool {
    fn name(&self) -> &str {
        "manage_skills"
    }

    fn description(&self) -> &str {
        "Manage skills at runtime. Actions: add (from URL), add_inline (raw markdown), list, remove (also dismiss matching pending drafts), remove_all (bulk remove with optional dry_run), enable, disable, browse (search registries), install (from registry), update (re-fetch from source), review (approve/dismiss auto-promoted skill drafts)."
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
                        "enum": ["add", "add_inline", "list", "remove", "remove_all", "enable", "disable", "browse", "install", "update", "review"],
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
                        "description": "Skill name (for 'remove', 'enable', 'disable', 'install', 'update' actions)"
                    },
                    "names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of skill names (for 'remove_all' action)"
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
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, preview `remove_all` changes without removing/dismissing anything."
                    }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: true,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageSkillsArgs = serde_json::from_str(arguments)
            .map_err(|e| anyhow::anyhow!("Invalid arguments: {}", e))?;

        match args.action.as_str() {
            "add" => {
                let url = args.url.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'url' parameter required for 'add' action"))?;
                let session_id = args._session_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("Missing session context for approval on 'add' action")
                })?;
                let approval = self
                    .request_approval(
                        session_id,
                        &format!(
                            "Add skill from URL '{}'\n\
                             WARNING: This will fetch external instructions that can influence AI behavior.",
                            url
                        ),
                    )
                    .await?;
                if matches!(approval, ApprovalResponse::Deny) {
                    return Ok("Skill add from URL denied by user.".to_string());
                }
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
            "remove_all" => {
                let names = args
                    .names
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'names' parameter required for 'remove_all' action"))?;
                self.handle_remove_all(names, args.dry_run.unwrap_or(false)).await
            }
            "enable" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'enable' action"))?;
                self.handle_enable(name).await
            }
            "disable" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'disable' action"))?;
                self.handle_disable(name).await
            }
            "browse" => {
                self.handle_browse(args.query.as_deref()).await
            }
            "install" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'install' action"))?;
                if !self.registry_urls.is_empty() {
                    let session_id = args._session_id.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("Missing session context for approval on 'install' action")
                    })?;
                    let approval = self
                        .request_approval(
                            session_id,
                            &format!(
                                "Install registry skill '{}'\n\
                                 WARNING: This will fetch external instructions from a registry URL.",
                                name
                            ),
                        )
                        .await?;
                    if matches!(approval, ApprovalResponse::Deny) {
                        return Ok("Skill installation denied by user.".to_string());
                    }
                }
                self.handle_install(name).await
            }
            "update" => {
                let name = args.name.as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'name' parameter required for 'update' action"))?;
                let session_id = args._session_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("Missing session context for approval on 'update' action")
                })?;
                let approval = self
                    .request_approval(
                        session_id,
                        &format!(
                            "Update skill '{}'\n\
                             WARNING: This will re-fetch external skill content from its source URL.",
                            name
                        ),
                    )
                    .await?;
                if matches!(approval, ApprovalResponse::Deny) {
                    return Ok("Skill update denied by user.".to_string());
                }
                self.handle_update(name).await
            }
            "review" => {
                self.handle_review(args.draft_id, args.approve).await
            }
            other => Ok(format!("Unknown action '{}'. Valid actions: add, add_inline, list, remove, remove_all, enable, disable, browse, install, update, review", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::tools::terminal::ApprovalRequest;
    use crate::traits::store_prelude::*;
    use crate::traits::SkillDraft;

    async fn setup_tool() -> (ManageSkillsTool, Arc<dyn StateStore>, tempfile::TempDir) {
        let skills_dir = tempfile::TempDir::new().unwrap();
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let sqlite_state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);

        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(4);
        let tool = ManageSkillsTool::new(
            skills_dir.path().to_path_buf(),
            sqlite_state.clone() as Arc<dyn StateStore>,
            approval_tx,
        );

        (tool, sqlite_state as Arc<dyn StateStore>, skills_dir)
    }

    async fn setup_tool_with_approval_channel() -> (
        ManageSkillsTool,
        Arc<dyn StateStore>,
        tempfile::TempDir,
        tokio::sync::mpsc::Receiver<ApprovalRequest>,
    ) {
        let skills_dir = tempfile::TempDir::new().unwrap();
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let sqlite_state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);

        let (approval_tx, approval_rx) = tokio::sync::mpsc::channel(4);
        let tool = ManageSkillsTool::new(
            skills_dir.path().to_path_buf(),
            sqlite_state.clone() as Arc<dyn StateStore>,
            approval_tx,
        );

        (
            tool,
            sqlite_state as Arc<dyn StateStore>,
            skills_dir,
            approval_rx,
        )
    }

    fn make_skill(name: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{} skill", name),
            triggers: vec!["deploy".to_string()],
            body: "Do the thing.".to_string(),
            origin: Some(skills::SKILL_ORIGIN_CUSTOM.to_string()),
            source: Some("inline".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        }
    }

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

    #[tokio::test]
    async fn add_url_requires_approval_and_respects_denial() {
        let (tool, _state, _skills_dir, mut approval_rx) = setup_tool_with_approval_channel().await;

        let approval_task = tokio::spawn(async move {
            let req = approval_rx.recv().await.expect("approval request");
            assert!(req.command.contains("Add skill from URL"));
            let _ = req.response_tx.send(crate::types::ApprovalResponse::Deny);
        });

        let result = tool
            .call(
                r#"{
                    "action":"add",
                    "url":"https://example.com/skill.md",
                    "_session_id":"test:owner"
                }"#,
            )
            .await
            .unwrap();
        approval_task.await.unwrap();

        assert!(result.contains("denied by user"));
    }

    #[tokio::test]
    async fn update_requires_approval_and_respects_denial() {
        let (tool, _state, skills_dir, mut approval_rx) = setup_tool_with_approval_channel().await;

        let skill = Skill {
            name: "updatable".to_string(),
            description: "Update me".to_string(),
            triggers: vec!["update".to_string()],
            body: "body".to_string(),
            origin: Some(skills::SKILL_ORIGIN_CUSTOM.to_string()),
            source: Some("url".to_string()),
            source_url: Some("https://example.com/updatable.md".to_string()),
            dir_path: None,
            resources: vec![],
        };
        skills::write_skill_to_file(skills_dir.path(), &skill).unwrap();

        let approval_task = tokio::spawn(async move {
            let req = approval_rx.recv().await.expect("approval request");
            assert!(req.command.contains("Update skill 'updatable'"));
            let _ = req.response_tx.send(crate::types::ApprovalResponse::Deny);
        });

        let result = tool
            .call(
                r#"{
                    "action":"update",
                    "name":"updatable",
                    "_session_id":"test:owner"
                }"#,
            )
            .await
            .unwrap();
        approval_task.await.unwrap();

        assert!(result.contains("denied by user"));
    }

    #[tokio::test]
    async fn remove_fuzzy_name_removes_skill_and_dismisses_matching_draft() {
        let (tool, state, skills_dir) = setup_tool().await;
        let skill = make_skill("send-resume");
        skills::write_skill_to_file(skills_dir.path(), &skill).unwrap();

        let draft = SkillDraft {
            id: 0,
            name: "Send Resume".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: "[]".to_string(),
            body: "draft body".to_string(),
            source_procedure: "resume-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        state.add_skill_draft(&draft).await.unwrap();

        let result = tool
            .call(r#"{"action":"remove","name":"send resume"}"#)
            .await
            .unwrap();

        assert!(result.contains("Skill 'send-resume' removed."));
        assert!(result.contains("Dismissed 1 pending draft(s)"));
        assert!(skills::load_skills(skills_dir.path()).is_empty());
        assert!(state.get_pending_skill_drafts().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn remove_nonexistent_skill_still_dismisses_pending_draft() {
        let (tool, state, _skills_dir) = setup_tool().await;

        let draft = SkillDraft {
            id: 0,
            name: "deploy-helper".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: "[]".to_string(),
            body: "draft body".to_string(),
            source_procedure: "deploy-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        state.add_skill_draft(&draft).await.unwrap();

        let result = tool
            .call(r#"{"action":"remove","name":"deploy helper"}"#)
            .await
            .unwrap();

        let result_lower = result.to_lowercase();
        assert!(result_lower.contains("was not installed"));
        assert!(result_lower.contains("dismissed 1 pending draft(s)"));
        assert!(state.get_pending_skill_drafts().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn remove_all_bulk_removes_skills_and_dismisses_drafts() {
        let (tool, state, skills_dir) = setup_tool().await;
        skills::write_skill_to_file(skills_dir.path(), &make_skill("send-resume")).unwrap();
        skills::write_skill_to_file(skills_dir.path(), &make_skill("confirm")).unwrap();

        let draft1 = SkillDraft {
            id: 0,
            name: "Send Resume".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: "[]".to_string(),
            body: "draft body".to_string(),
            source_procedure: "resume-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        let draft2 = SkillDraft {
            id: 0,
            name: "deploy-helper".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: "[]".to_string(),
            body: "draft body".to_string(),
            source_procedure: "deploy-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        state.add_skill_draft(&draft1).await.unwrap();
        state.add_skill_draft(&draft2).await.unwrap();

        let result = tool
            .call(
                r#"{
                    "action":"remove_all",
                    "names":["send resume","confirm","deploy helper","missing"]
                }"#,
            )
            .await
            .unwrap();

        assert!(result.contains("remove_all processed 4 skill request(s):"));
        assert!(result.contains("Removed 2: confirm, send-resume"));
        assert!(result.contains("pending drafts matched 1: deploy helper"));
        assert!(result.contains("Dismissed pending draft(s): #"));
        assert!(result.contains("Not found 1: missing"));
        assert!(skills::load_skills(skills_dir.path()).is_empty());
        assert!(state.get_pending_skill_drafts().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn remove_all_dry_run_does_not_modify_skills_or_drafts() {
        let (tool, state, skills_dir) = setup_tool().await;
        skills::write_skill_to_file(skills_dir.path(), &make_skill("send-resume")).unwrap();

        let draft = SkillDraft {
            id: 0,
            name: "Send Resume".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: "[]".to_string(),
            body: "draft body".to_string(),
            source_procedure: "resume-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        state.add_skill_draft(&draft).await.unwrap();

        let result = tool
            .call(
                r#"{
                    "action":"remove_all",
                    "names":["send resume"],
                    "dry_run":true
                }"#,
            )
            .await
            .unwrap();

        assert!(result.contains("Dry run for remove_all processed 1 skill request(s):"));
        assert!(result.contains("Would remove 1: send-resume"));
        assert!(result.contains("Would dismiss pending draft(s): #"));
        assert_eq!(skills::load_skills(skills_dir.path()).len(), 1);
        assert_eq!(state.get_pending_skill_drafts().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn review_requires_explicit_approve_flag() {
        let (tool, state, _skills_dir) = setup_tool().await;
        let draft = SkillDraft {
            id: 0,
            name: "deploy-helper".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: r#"["deploy helper"]"#.to_string(),
            body: "1. Validate prerequisites.\n2. Deploy and verify.".to_string(),
            source_procedure: "deploy-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        let draft_id = state.add_skill_draft(&draft).await.unwrap();

        let result = tool
            .call(&format!(r#"{{"action":"review","draft_id":{}}}"#, draft_id))
            .await
            .unwrap();

        assert!(result.contains("requires `approve: true`"));
        assert_eq!(state.get_pending_skill_drafts().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn review_approve_blocks_canonical_filename_collision() {
        let (tool, state, skills_dir) = setup_tool().await;
        skills::write_skill_to_file(skills_dir.path(), &make_skill("send-resume")).unwrap();

        let draft = SkillDraft {
            id: 0,
            name: "send resume".to_string(),
            description: "Draft replacement".to_string(),
            triggers_json: r#"["resume send"]"#.to_string(),
            body: "1. Gather inputs.\n2. Send the resume.".to_string(),
            source_procedure: "resume-proc".to_string(),
            status: "pending".to_string(),
            created_at: String::new(),
        };
        let draft_id = state.add_skill_draft(&draft).await.unwrap();

        let result = tool
            .call(&format!(
                r#"{{"action":"review","draft_id":{},"approve":true}}"#,
                draft_id
            ))
            .await
            .unwrap();

        assert!(result.contains("conflicts with existing skill"));
        assert_eq!(state.get_pending_skill_drafts().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn disable_and_enable_actions_toggle_skill_state() {
        let (tool, _state, skills_dir) = setup_tool().await;
        skills::write_skill_to_file(skills_dir.path(), &make_skill("toggle-skill")).unwrap();
        assert_eq!(skills::load_skills(skills_dir.path()).len(), 1);

        let disable_result = tool
            .call(r#"{"action":"disable","name":"toggle skill"}"#)
            .await
            .unwrap();
        assert!(disable_result.contains("disabled"));
        assert!(skills::load_skills(skills_dir.path()).is_empty());

        let list_result = tool.call(r#"{"action":"list"}"#).await.unwrap();
        assert!(list_result.contains("status: disabled"));

        let enable_result = tool
            .call(r#"{"action":"enable","name":"toggle-skill"}"#)
            .await
            .unwrap();
        assert!(enable_result.contains("enabled"));
        assert_eq!(skills::load_skills(skills_dir.path()).len(), 1);
    }
}
