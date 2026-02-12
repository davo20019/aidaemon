use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::cron_utils::{compute_next_run_local, is_one_shot_schedule};
use crate::traits::{StateStore, Tool};
use crate::types::FactPrivacy;

pub struct ManageMemoriesTool {
    state: Arc<dyn StateStore>,
}

impl ManageMemoriesTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }

    /// Resolve a goal identifier provided by the model/user.
    /// Accepts:
    /// - exact full V3 goal ID
    /// - unique prefix (e.g., the 8-char short ID shown in list output)
    async fn resolve_goal_id_v3(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        // Fast path: exact ID match.
        if self.state.get_goal_v3(trimmed).await?.is_some() {
            return Ok(trimmed.to_string());
        }

        // Prefix fallback: match against scheduled goals because this tool's
        // schedule operations operate on scheduled goals.
        let goals = self.state.get_scheduled_goals_v3().await?;
        let mut matches: Vec<&crate::traits::GoalV3> =
            goals.iter().filter(|g| g.id.starts_with(trimmed)).collect();

        if matches.is_empty() {
            anyhow::bail!("Scheduled goal not found: {}", trimmed);
        }

        if matches.len() == 1 {
            return Ok(matches.remove(0).id.clone());
        }

        // Deterministic ambiguity handling: prefer non-terminal states first.
        matches.sort_by_key(|g| match g.status.as_str() {
            "active" => 0usize,
            "paused" => 1,
            "pending_confirmation" => 2,
            "failed" => 3,
            "cancelled" => 4,
            "completed" => 5,
            _ => 6,
        });

        let preview = matches
            .iter()
            .take(5)
            .map(|g| {
                let short: String = g.id.chars().take(8).collect();
                format!("{} ({}, {})", short, g.status, g.description)
            })
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::bail!(
            "Goal ID prefix '{}' is ambiguous ({} matches): {}. Use full goal_id_v3.",
            trimmed,
            matches.len(),
            preview
        );
    }

    fn is_protected_system_maintenance_goal(goal: &crate::traits::GoalV3) -> bool {
        const KNOWLEDGE_GOAL_DESC: &str =
            "Maintain knowledge base: process embeddings, consolidate memories, decay old facts";
        const HEALTH_GOAL_DESC: &str =
            "Maintain memory health: prune old events, clean up retention, remove stale data";

        if goal.description == KNOWLEDGE_GOAL_DESC || goal.description == HEALTH_GOAL_DESC {
            return true;
        }

        if let Some(ctx) = &goal.context {
            if let Ok(v) = serde_json::from_str::<Value>(ctx) {
                if v.get("system_protected").and_then(|x| x.as_bool()) == Some(true) {
                    return true;
                }
            }
        }

        false
    }

    fn goal_matches_query(goal: &crate::traits::GoalV3, query: &str) -> bool {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return false;
        }
        goal.id.to_ascii_lowercase().starts_with(&q)
            || goal.description.to_ascii_lowercase().contains(&q)
    }

    fn truncate_chars(s: &str, max: usize) -> String {
        s.chars().take(max).collect()
    }
}

#[derive(Deserialize)]
struct ManageArgs {
    action: String,
    category: Option<String>,
    key: Option<String>,
    privacy: Option<String>,
    query: Option<String>,
    goal_id: Option<i64>,
    goal_id_v3: Option<String>,
}

#[async_trait]
impl Tool for ManageMemoriesTool {
    fn name(&self) -> &str {
        "manage_memories"
    }

    fn description(&self) -> &str {
        "List/search/forget memories, and list/cancel/pause/resume/retry/diagnose scheduled goals (accepts full or unique prefix goal_id_v3; includes bulk retry for failed schedules)"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_memories",
            "description": "List, search, forget, or change privacy of stored memories and goals. Also list/cancel/pause/resume/retry/diagnose scheduled goals. IMPORTANT for scheduled-goal management: first call action='list_scheduled' or 'list_scheduled_matching' to get exact goal IDs, then call cancel_scheduled/pause_scheduled/resume_scheduled/retry_scheduled/diagnose_scheduled with goal_id_v3. Use retry_failed_scheduled for one-shot recovery of failed goals (optionally filtered by query). Do not use terminal/sqlite for scheduled-goal management when this tool can do it. Protected system maintenance goals cannot be cancelled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "forget", "set_privacy", "search", "list_goals", "complete_goal", "abandon_goal", "list_scheduled", "list_scheduled_matching", "cancel_scheduled", "pause_scheduled", "resume_scheduled", "retry_scheduled", "retry_failed_scheduled", "cancel_scheduled_matching", "retry_scheduled_matching", "diagnose_scheduled"],
                        "description": "Action to perform. For schedule operations: use list_scheduled or list_scheduled_matching first, then cancel_scheduled/pause_scheduled/resume_scheduled/retry_scheduled/diagnose_scheduled with exact goal_id_v3. For bulk operations, use retry_failed_scheduled (all failed, optionally filtered), cancel_scheduled_matching, or retry_scheduled_matching with query."
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (for list/forget/set_privacy)"
                    },
                    "key": {
                        "type": "string",
                        "description": "Fact key (for forget/set_privacy)"
                    },
                    "privacy": {
                        "type": "string",
                        "enum": ["global", "channel", "private"],
                        "description": "Target privacy level (for set_privacy)"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search term (for search action). For scheduled bulk operations, this matches goal ID prefix or description text."
                    },
                    "goal_id": {
                        "type": "integer",
                        "description": "Goal ID (for complete_goal/abandon_goal)"
                    },
                    "goal_id_v3": {
                        "type": "string",
                        "description": "Exact V3 goal ID for cancel_scheduled, pause_scheduled, resume_scheduled, retry_scheduled, or diagnose_scheduled. Retrieve via list_scheduled first."
                    }
                },
                "required": ["action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            "list" => {
                let facts = self.state.get_all_facts_with_provenance().await?;
                if facts.is_empty() {
                    return Ok("No memories stored.".to_string());
                }

                let filtered = if let Some(ref cat) = args.category {
                    facts.into_iter().filter(|f| f.category == *cat).collect::<Vec<_>>()
                } else {
                    facts
                };

                let mut output = format!("**Stored Memories** ({} facts)\n\n", filtered.len());
                let mut current_cat = String::new();
                for f in &filtered {
                    if f.category != current_cat {
                        current_cat = f.category.clone();
                        output.push_str(&format!("### {}\n", current_cat));
                    }
                    let privacy_label = f.privacy.to_string();
                    let channel_label = f.channel_id.as_deref().unwrap_or("global");
                    let age = chrono::Utc::now().signed_duration_since(f.updated_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        "just now".to_string()
                    };
                    output.push_str(&format!(
                        "- **{}**: {} (privacy: {}, from: {}, updated: {})\n",
                        f.key, f.value, privacy_label, channel_label, age_str
                    ));
                }
                Ok(output)
            }
            "forget" => {
                let key = args.key.as_deref().ok_or_else(|| anyhow::anyhow!("'key' is required for forget action"))?;
                let category = args.category.as_deref().ok_or_else(|| anyhow::anyhow!("'category' is required for forget action"))?;

                let facts = self.state.get_facts(Some(category)).await?;
                let fact = facts.iter().find(|f| f.key == key && f.superseded_at.is_none());

                match fact {
                    Some(f) => {
                        self.state.delete_fact(f.id).await?;
                        Ok(format!("Forgotten: [{}] {}", category, key))
                    }
                    None => Ok(format!("No active fact found: [{}] {}", category, key)),
                }
            }
            "set_privacy" => {
                let key = args.key.as_deref().ok_or_else(|| anyhow::anyhow!("'key' is required for set_privacy action"))?;
                let category = args.category.as_deref().ok_or_else(|| anyhow::anyhow!("'category' is required for set_privacy action"))?;
                let privacy_str = args.privacy.as_deref().ok_or_else(|| anyhow::anyhow!("'privacy' is required for set_privacy action"))?;
                let privacy = FactPrivacy::from_str_lossy(privacy_str);

                let facts = self.state.get_facts(Some(category)).await?;
                let fact = facts.iter().find(|f| f.key == key && f.superseded_at.is_none());

                match fact {
                    Some(f) => {
                        self.state.update_fact_privacy(f.id, privacy).await?;
                        Ok(format!("Updated privacy: [{}] {} → {}", category, key, privacy))
                    }
                    None => Ok(format!("No active fact found: [{}] {}", category, key)),
                }
            }
            "search" => {
                let query = args.query.as_deref().unwrap_or("");
                if query.is_empty() {
                    return Ok("Please provide a search query.".to_string());
                }

                let facts = self.state.get_all_facts_with_provenance().await?;
                let query_lower = query.to_lowercase();
                let matches: Vec<_> = facts
                    .iter()
                    .filter(|f| {
                        f.key.to_lowercase().contains(&query_lower)
                            || f.value.to_lowercase().contains(&query_lower)
                            || f.category.to_lowercase().contains(&query_lower)
                    })
                    .collect();

                if matches.is_empty() {
                    return Ok(format!("No memories matching '{}'.", query));
                }

                let mut output = format!("**Search results for '{}'** ({} matches)\n\n", query, matches.len());
                for f in matches.iter().take(20) {
                    let privacy_label = f.privacy.to_string();
                    let channel_label = f.channel_id.as_deref().unwrap_or("global");
                    output.push_str(&format!(
                        "- [{}] **{}**: {} (privacy: {}, from: {})\n",
                        f.category, f.key, f.value, privacy_label, channel_label
                    ));
                }
                Ok(output)
            }
            "list_goals" => {
                let goals = self.state.get_active_goals().await?;
                if goals.is_empty() {
                    return Ok("No active goals.".to_string());
                }

                let mut output = format!("**Active Goals** ({} goals)\n\n", goals.len());
                for g in &goals {
                    let age = chrono::Utc::now().signed_duration_since(g.created_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        "just now".to_string()
                    };
                    let notes_count = g.progress_notes.as_ref().map_or(0, |n| n.len());
                    output.push_str(&format!(
                        "- **[ID: {}]** {} (priority: {}, created: {}, {} progress notes)\n",
                        g.id, g.description, g.priority, age_str, notes_count
                    ));
                }
                Ok(output)
            }
            "complete_goal" => {
                let goal_id = args.goal_id.ok_or_else(|| anyhow::anyhow!("'goal_id' is required for complete_goal action"))?;
                self.state.update_goal(goal_id, Some("completed"), None).await?;
                Ok(format!("Goal {} marked as completed.", goal_id))
            }
            "abandon_goal" => {
                let goal_id = args.goal_id.ok_or_else(|| anyhow::anyhow!("'goal_id' is required for abandon_goal action"))?;
                self.state.update_goal(goal_id, Some("abandoned"), None).await?;
                Ok(format!("Goal {} marked as abandoned. It will not be re-created by automatic analysis.", goal_id))
            }
            "list_scheduled" => {
                let goals = self.state.get_scheduled_goals_v3().await?;
                if goals.is_empty() {
                    return Ok("No scheduled goals.".to_string());
                }

                let mut active = Vec::new();
                let mut paused = Vec::new();
                let mut pending_confirmation = Vec::new();
                let mut failed = Vec::new();
                let mut cancelled = Vec::new();
                let mut completed = Vec::new();
                let mut other = Vec::new();

                for g in &goals {
                    match g.status.as_str() {
                        "active" => active.push(g),
                        "paused" => paused.push(g),
                        "pending_confirmation" => pending_confirmation.push(g),
                        "failed" => failed.push(g),
                        "cancelled" => cancelled.push(g),
                        "completed" => completed.push(g),
                        _ => other.push(g),
                    }
                }

                let active_count = active.len() + paused.len() + pending_confirmation.len();
                let mut output = format!("**Scheduled Goals** ({} total)\n\n", goals.len());
                if active_count == 0 {
                    output.push_str("No active scheduled tasks.\n\n");
                }

                let mut append_group = |title: &str, items: &[&crate::traits::GoalV3]| {
                    if items.is_empty() {
                        return;
                    }
                    output.push_str(&format!("**{}** ({})\n", title, items.len()));
                    for g in items {
                        let desc: String = g.description.chars().take(80).collect();
                        let schedule = g
                            .schedule
                            .clone()
                            .unwrap_or_else(|| "(none)".to_string());
                        let goal_type = if g.goal_type == "finite"
                            || g
                                .schedule
                                .as_ref()
                                .is_some_and(|s| is_one_shot_schedule(s))
                        {
                            "one-time"
                        } else {
                            "recurring"
                        };
                        let next_run = g
                            .schedule
                            .as_deref()
                            .and_then(|s| compute_next_run_local(s).ok())
                            .map(|dt| dt.format("%Y-%m-%d %H:%M %Z").to_string())
                            .unwrap_or_else(|| "n/a".to_string());
                        output.push_str(&format!(
                            "- **{}** {} (type: {}, status: {}, schedule: {}, next: {})\n",
                            g.id, desc, goal_type, g.status, schedule, next_run
                        ));
                    }
                    output.push('\n');
                };

                append_group("Active", &active);
                append_group("Paused", &paused);
                append_group("Pending Confirmation", &pending_confirmation);
                append_group("Failed", &failed);
                append_group("Cancelled", &cancelled);
                append_group("Completed", &completed);
                append_group("Other", &other);
                Ok(output)
            }
            "list_scheduled_matching" => {
                let query = args
                    .query
                    .as_deref()
                    .map(str::trim)
                    .filter(|q| !q.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'query' is required for list_scheduled_matching action"))?;
                let goals = self.state.get_scheduled_goals_v3().await?;
                let mut matched: Vec<&crate::traits::GoalV3> = goals
                    .iter()
                    .filter(|g| Self::goal_matches_query(g, query))
                    .collect();
                if matched.is_empty() {
                    return Ok(format!("No scheduled goals matched query '{}'.", query));
                }
                matched.sort_by(|a, b| b.created_at.cmp(&a.created_at));

                let mut output = format!(
                    "**Matching Scheduled Goals** for '{}' ({} matches)\n\n",
                    query,
                    matched.len()
                );
                for g in matched {
                    let desc: String = g.description.chars().take(80).collect();
                    let schedule = g
                        .schedule
                        .clone()
                        .unwrap_or_else(|| "(none)".to_string());
                    let goal_type = if g.goal_type == "finite"
                        || g
                            .schedule
                            .as_ref()
                            .is_some_and(|s| is_one_shot_schedule(s))
                    {
                        "one-time"
                    } else {
                        "recurring"
                    };
                    let next_run = g
                        .schedule
                        .as_deref()
                        .and_then(|s| compute_next_run_local(s).ok())
                        .map(|dt| dt.format("%Y-%m-%d %H:%M %Z").to_string())
                        .unwrap_or_else(|| "n/a".to_string());
                    output.push_str(&format!(
                        "- **{}** {} (type: {}, status: {}, schedule: {}, next: {})\n",
                        g.id, desc, goal_type, g.status, schedule, next_run
                    ));
                }
                Ok(output)
            }
            "cancel_scheduled" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for cancel_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id_v3(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if Self::is_protected_system_maintenance_goal(&goal) {
                    return Ok(format!(
                        "Cannot cancel protected system maintenance goal {}.",
                        resolved_goal_id
                    ));
                }
                goal.status = "cancelled".to_string();
                goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                goal.updated_at = chrono::Utc::now().to_rfc3339();
                self.state.update_goal_v3(&goal).await?;
                Ok(format!("Cancelled scheduled goal {}.", resolved_goal_id))
            }
            "pause_scheduled" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for pause_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id_v3(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.schedule.is_none() {
                    return Ok("Only scheduled goals can be paused.".to_string());
                }
                if goal.status != "active" {
                    return Ok(format!(
                        "Only active scheduled goals can be paused (current status: {}).",
                        goal.status
                    ));
                }
                goal.status = "paused".to_string();
                goal.updated_at = chrono::Utc::now().to_rfc3339();
                self.state.update_goal_v3(&goal).await?;
                Ok(format!("Paused scheduled goal {}.", resolved_goal_id))
            }
            "resume_scheduled" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for resume_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id_v3(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.schedule.is_none() {
                    return Ok("Only scheduled goals can be resumed.".to_string());
                }
                if goal.status != "paused" {
                    return Ok(format!(
                        "Only paused scheduled goals can be resumed (current status: {}).",
                        goal.status
                    ));
                }
                if goal.goal_type == "continuous" {
                    let active_evergreen = self.state.count_active_evergreen_goals().await?;
                    if active_evergreen >= 10 {
                        return Ok(format!(
                            "Cannot resume recurring goal: hard cap of 10 active evergreen goals reached (current: {}).",
                            active_evergreen
                        ));
                    }
                }
                goal.status = "active".to_string();
                goal.updated_at = chrono::Utc::now().to_rfc3339();
                self.state.update_goal_v3(&goal).await?;
                Ok(format!("Resumed scheduled goal {}.", resolved_goal_id))
            }
            "retry_scheduled" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for retry_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id_v3(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.schedule.is_none() {
                    return Ok("Only scheduled goals can be retried.".to_string());
                }
                if goal.status != "failed" {
                    return Ok(format!(
                        "Only failed scheduled goals can be retried (current status: {}).",
                        goal.status
                    ));
                }
                if goal.goal_type == "continuous" {
                    let active_evergreen = self.state.count_active_evergreen_goals().await?;
                    if active_evergreen >= 10 {
                        return Ok(format!(
                            "Cannot retry recurring goal: hard cap of 10 active evergreen goals reached (current: {}).",
                            active_evergreen
                        ));
                    }
                }
                goal.status = "active".to_string();
                goal.completed_at = None;
                goal.updated_at = chrono::Utc::now().to_rfc3339();
                self.state.update_goal_v3(&goal).await?;
                Ok(format!(
                    "Retried scheduled goal {}. It is active again.",
                    resolved_goal_id
                ))
            }
            "retry_failed_scheduled" => {
                let query = args.query.as_deref().map(str::trim).unwrap_or("");
                let goals = self.state.get_scheduled_goals_v3().await?;
                let mut matched: Vec<&crate::traits::GoalV3> = goals
                    .iter()
                    .filter(|g| {
                        g.status == "failed"
                            && (query.is_empty() || Self::goal_matches_query(g, query))
                    })
                    .collect();
                if matched.is_empty() {
                    if query.is_empty() {
                        return Ok("No failed scheduled goals to retry.".to_string());
                    }
                    return Ok(format!(
                        "No failed scheduled goals matched query '{}'.",
                        query
                    ));
                }
                matched.sort_by(|a, b| b.created_at.cmp(&a.created_at));

                let mut retried = Vec::new();
                let mut cap_blocked = Vec::new();
                let mut errors = Vec::new();
                let mut active_evergreen = self.state.count_active_evergreen_goals().await?;

                for g in matched {
                    let mut updated = g.clone();
                    if updated.goal_type == "continuous" && active_evergreen >= 10 {
                        cap_blocked.push(updated.id.clone());
                        continue;
                    }
                    updated.status = "active".to_string();
                    updated.completed_at = None;
                    updated.updated_at = chrono::Utc::now().to_rfc3339();
                    match self.state.update_goal_v3(&updated).await {
                        Ok(()) => {
                            if updated.goal_type == "continuous" {
                                active_evergreen += 1;
                            }
                            retried.push(updated.id);
                        }
                        Err(e) => errors.push(format!("{} ({})", g.id, e)),
                    }
                }

                let label = if query.is_empty() {
                    "retry_failed_scheduled(*)".to_string()
                } else {
                    format!("retry_failed_scheduled('{}')", query)
                };
                let mut out = format!(
                    "{}: retried {}, cap-blocked {}, errors {}.",
                    label,
                    retried.len(),
                    cap_blocked.len(),
                    errors.len()
                );
                if !retried.is_empty() {
                    out.push_str(&format!("\nRetried:\n- {}", retried.join("\n- ")));
                }
                if !cap_blocked.is_empty() {
                    out.push_str(&format!(
                        "\nCap blocked (unchanged):\n- {}",
                        cap_blocked.join("\n- ")
                    ));
                }
                if !errors.is_empty() {
                    out.push_str(&format!("\nErrors:\n- {}", errors.join("\n- ")));
                }
                Ok(out)
            }
            "cancel_scheduled_matching" => {
                let query = args
                    .query
                    .as_deref()
                    .map(str::trim)
                    .filter(|q| !q.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'query' is required for cancel_scheduled_matching action"))?;
                let goals = self.state.get_scheduled_goals_v3().await?;
                let mut matched: Vec<&crate::traits::GoalV3> = goals
                    .iter()
                    .filter(|g| Self::goal_matches_query(g, query))
                    .collect();
                if matched.is_empty() {
                    return Ok(format!("No scheduled goals matched query '{}'.", query));
                }
                matched.sort_by(|a, b| b.created_at.cmp(&a.created_at));

                let mut cancelled = Vec::new();
                let mut protected = Vec::new();
                let mut already_terminal = Vec::new();
                let mut errors = Vec::new();

                for g in matched {
                    if Self::is_protected_system_maintenance_goal(g) {
                        protected.push(g.id.clone());
                        continue;
                    }
                    if g.status == "cancelled" || g.status == "completed" {
                        already_terminal.push(format!("{} ({})", g.id, g.status));
                        continue;
                    }
                    let mut updated = g.clone();
                    updated.status = "cancelled".to_string();
                    updated.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    updated.updated_at = chrono::Utc::now().to_rfc3339();
                    match self.state.update_goal_v3(&updated).await {
                        Ok(()) => cancelled.push(updated.id),
                        Err(e) => errors.push(format!("{} ({})", g.id, e)),
                    }
                }

                let mut out = format!(
                    "cancel_scheduled_matching('{}'): cancelled {}, protected {}, already terminal {}, errors {}.",
                    query,
                    cancelled.len(),
                    protected.len(),
                    already_terminal.len(),
                    errors.len()
                );
                if !cancelled.is_empty() {
                    out.push_str(&format!("\nCancelled:\n- {}", cancelled.join("\n- ")));
                }
                if !protected.is_empty() {
                    out.push_str(&format!("\nProtected (not cancelled):\n- {}", protected.join("\n- ")));
                }
                if !already_terminal.is_empty() {
                    out.push_str(&format!(
                        "\nAlready terminal:\n- {}",
                        already_terminal.join("\n- ")
                    ));
                }
                if !errors.is_empty() {
                    out.push_str(&format!("\nErrors:\n- {}", errors.join("\n- ")));
                }
                Ok(out)
            }
            "retry_scheduled_matching" => {
                let query = args
                    .query
                    .as_deref()
                    .map(str::trim)
                    .filter(|q| !q.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'query' is required for retry_scheduled_matching action"))?;
                let goals = self.state.get_scheduled_goals_v3().await?;
                let mut matched: Vec<&crate::traits::GoalV3> = goals
                    .iter()
                    .filter(|g| Self::goal_matches_query(g, query))
                    .collect();
                if matched.is_empty() {
                    return Ok(format!("No scheduled goals matched query '{}'.", query));
                }
                matched.sort_by(|a, b| b.created_at.cmp(&a.created_at));

                let mut retried = Vec::new();
                let mut non_failed = Vec::new();
                let mut cap_blocked = Vec::new();
                let mut errors = Vec::new();

                let mut active_evergreen = self.state.count_active_evergreen_goals().await?;
                for g in matched {
                    if g.status != "failed" {
                        non_failed.push(format!("{} ({})", g.id, g.status));
                        continue;
                    }
                    let mut updated = g.clone();
                    if updated.goal_type == "continuous" && active_evergreen >= 10 {
                        cap_blocked.push(updated.id.clone());
                        continue;
                    }
                    updated.status = "active".to_string();
                    updated.completed_at = None;
                    updated.updated_at = chrono::Utc::now().to_rfc3339();
                    match self.state.update_goal_v3(&updated).await {
                        Ok(()) => {
                            if updated.goal_type == "continuous" {
                                active_evergreen += 1;
                            }
                            retried.push(updated.id);
                        }
                        Err(e) => errors.push(format!("{} ({})", g.id, e)),
                    }
                }

                let mut out = format!(
                    "retry_scheduled_matching('{}'): retried {}, non-failed {}, cap-blocked {}, errors {}.",
                    query,
                    retried.len(),
                    non_failed.len(),
                    cap_blocked.len(),
                    errors.len()
                );
                if !retried.is_empty() {
                    out.push_str(&format!("\nRetried:\n- {}", retried.join("\n- ")));
                }
                if !non_failed.is_empty() {
                    out.push_str(&format!("\nNot failed (unchanged):\n- {}", non_failed.join("\n- ")));
                }
                if !cap_blocked.is_empty() {
                    out.push_str(&format!(
                        "\nCap blocked (unchanged):\n- {}",
                        cap_blocked.join("\n- ")
                    ));
                }
                if !errors.is_empty() {
                    out.push_str(&format!("\nErrors:\n- {}", errors.join("\n- ")));
                }
                Ok(out)
            }
            "diagnose_scheduled" => {
                let goal_id = args
                    .goal_id_v3
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id_v3' is required for diagnose_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id_v3(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(goal) = self.state.get_goal_v3(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.schedule.is_none() {
                    return Ok("Only scheduled goals can be diagnosed with this action.".to_string());
                }

                let schedule = goal
                    .schedule
                    .clone()
                    .unwrap_or_else(|| "(none)".to_string());
                let next_run = goal
                    .schedule
                    .as_deref()
                    .and_then(|s| compute_next_run_local(s).ok())
                    .map(|dt| dt.format("%Y-%m-%d %H:%M %Z").to_string())
                    .unwrap_or_else(|| "n/a".to_string());
                let goal_type = if goal.goal_type == "finite"
                    || goal
                        .schedule
                        .as_ref()
                        .is_some_and(|s| is_one_shot_schedule(s))
                {
                    "one-time"
                } else {
                    "recurring"
                };

                let tasks = self.state.get_tasks_for_goal_v3(&goal.id).await?;
                let mut task_total = 0usize;
                let mut task_failed = 0usize;
                let mut task_completed = 0usize;
                let mut task_running = 0usize;
                let mut task_pending = 0usize;
                for t in &tasks {
                    task_total += 1;
                    match t.status.as_str() {
                        "failed" => task_failed += 1,
                        "completed" => task_completed += 1,
                        "running" | "claimed" => task_running += 1,
                        "pending" => task_pending += 1,
                        _ => {}
                    }
                }

                let mut out = format!(
                    "**Scheduled Goal Diagnosis**\n\n- ID: {}\n- Description: {}\n- Type: {}\n- Status: {}\n- Schedule: {}\n- Next: {}\n- Tasks: total {}, failed {}, running {}, pending {}, completed {}",
                    goal.id,
                    goal.description,
                    goal_type,
                    goal.status,
                    schedule,
                    next_run,
                    task_total,
                    task_failed,
                    task_running,
                    task_pending,
                    task_completed
                );

                if let Some(last_failed_task) = tasks.iter().filter(|t| t.status == "failed").max_by(
                    |a, b| {
                        let a_key = a
                            .completed_at
                            .clone()
                            .or_else(|| a.started_at.clone())
                            .unwrap_or_else(|| a.created_at.clone());
                        let b_key = b
                            .completed_at
                            .clone()
                            .or_else(|| b.started_at.clone())
                            .unwrap_or_else(|| b.created_at.clone());
                        a_key.cmp(&b_key)
                    },
                ) {
                    out.push_str(&format!(
                        "\n\n**Latest Failed Task**\n- Task ID: {}\n- Description: {}\n- Error: {}",
                        last_failed_task.id,
                        last_failed_task.description,
                        last_failed_task
                            .error
                            .as_deref()
                            .map(|e| Self::truncate_chars(e, 220))
                            .unwrap_or_else(|| "n/a".to_string())
                    ));

                    let activities = self
                        .state
                        .get_task_activities_v3(&last_failed_task.id)
                        .await
                        .unwrap_or_default();
                    if !activities.is_empty() {
                        out.push_str("\n\n**Recent Activity**");
                        for a in activities.iter().rev().take(3).rev() {
                            let tool = a.tool_name.as_deref().unwrap_or("-");
                            let ok = a.success.map(|v| if v { "ok" } else { "err" }).unwrap_or("n/a");
                            let result = a
                                .result
                                .as_deref()
                                .map(|r| Self::truncate_chars(r, 120))
                                .unwrap_or_else(|| "".to_string());
                            if result.is_empty() {
                                out.push_str(&format!(
                                    "\n- {} [{}] tool={} at {}",
                                    a.activity_type, ok, tool, a.created_at
                                ));
                            } else {
                                out.push_str(&format!(
                                    "\n- {} [{}] tool={} at {} => {}",
                                    a.activity_type, ok, tool, a.created_at, result
                                ));
                            }
                        }
                    }
                }

                Ok(out)
            }
            other => Ok(format!("Unknown action: '{}'. Use list, forget, set_privacy, search, list_goals, complete_goal, abandon_goal, list_scheduled, list_scheduled_matching, cancel_scheduled, pause_scheduled, resume_scheduled, retry_scheduled, retry_failed_scheduled, cancel_scheduled_matching, retry_scheduled_matching, or diagnose_scheduled.", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::GoalV3;

    async fn setup_state() -> Arc<dyn StateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state as Arc<dyn StateStore>
    }

    #[tokio::test]
    async fn cancel_scheduled_blocks_protected_system_maintenance_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = GoalV3::new_continuous(
            "Maintain memory health: prune old events, clean up retention, remove stale data",
            "system",
            "30 3 * * *",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Cannot cancel protected system maintenance goal"));
        let fetched = state.get_goal_v3(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
    }

    #[tokio::test]
    async fn cancel_scheduled_allows_non_system_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = GoalV3::new_continuous(
            "English research for pronunciation",
            "user-session",
            "0 5,12,19 * * *",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Cancelled scheduled goal"));
        let fetched = state.get_goal_v3(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "cancelled");
    }

    #[tokio::test]
    async fn retry_scheduled_reactivates_failed_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut goal = GoalV3::new_continuous(
            "Retryable scheduled goal",
            "user-session",
            "0 */6 * * *",
            Some(2000),
            Some(20000),
        );
        goal.status = "failed".to_string();
        goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Retried scheduled goal"));
        let fetched = state.get_goal_v3(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
        assert!(fetched.completed_at.is_none());
    }

    #[tokio::test]
    async fn retry_scheduled_rejects_non_failed_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = GoalV3::new_continuous(
            "Already active scheduled goal",
            "user-session",
            "0 */6 * * *",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal_v3(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id_v3": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Only failed scheduled goals can be retried"));
        let fetched = state.get_goal_v3(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
    }

    #[tokio::test]
    async fn retry_failed_scheduled_reactivates_all_failed() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut failed_one = GoalV3::new_continuous(
            "Maintain knowledge base: process embeddings, consolidate memories, decay old facts",
            "system",
            "0 */6 * * *",
            Some(1000),
            Some(5000),
        );
        failed_one.status = "failed".to_string();
        let failed_one_id = failed_one.id.clone();
        state.create_goal_v3(&failed_one).await.unwrap();

        let mut failed_two = GoalV3::new_continuous(
            "English pronunciation slot A",
            "user-session",
            "0 5 * * *",
            Some(1000),
            Some(5000),
        );
        failed_two.status = "failed".to_string();
        let failed_two_id = failed_two.id.clone();
        state.create_goal_v3(&failed_two).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "retry_failed_scheduled"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("retried 2"));
        let fetched_one = state.get_goal_v3(&failed_one_id).await.unwrap().unwrap();
        let fetched_two = state.get_goal_v3(&failed_two_id).await.unwrap().unwrap();
        assert_eq!(fetched_one.status, "active");
        assert_eq!(fetched_two.status, "active");
    }

    #[tokio::test]
    async fn cancel_scheduled_matching_cancels_matching_goals() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let g1 = GoalV3::new_continuous(
            "English pronunciation slot A",
            "user-session",
            "0 5 * * *",
            Some(2000),
            Some(20000),
        );
        let g2 = GoalV3::new_continuous(
            "English pronunciation slot B",
            "user-session",
            "0 12 * * *",
            Some(2000),
            Some(20000),
        );
        let g3 = GoalV3::new_continuous(
            "Unrelated recurring task",
            "user-session",
            "0 19 * * *",
            Some(2000),
            Some(20000),
        );
        state.create_goal_v3(&g1).await.unwrap();
        state.create_goal_v3(&g2).await.unwrap();
        state.create_goal_v3(&g3).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled_matching",
                    "query": "english pronunciation"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("cancelled 2"));
        let g1_after = state.get_goal_v3(&g1.id).await.unwrap().unwrap();
        let g2_after = state.get_goal_v3(&g2.id).await.unwrap().unwrap();
        let g3_after = state.get_goal_v3(&g3.id).await.unwrap().unwrap();
        assert_eq!(g1_after.status, "cancelled");
        assert_eq!(g2_after.status, "cancelled");
        assert_eq!(g3_after.status, "active");
    }

    #[tokio::test]
    async fn diagnose_scheduled_reports_latest_failed_task() {
        use crate::traits::{TaskActivityV3, TaskV3};

        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut goal = GoalV3::new_continuous(
            "Diagnose me",
            "user-session",
            "0 */6 * * *",
            Some(2000),
            Some(20000),
        );
        goal.status = "failed".to_string();
        state.create_goal_v3(&goal).await.unwrap();

        let task = TaskV3 {
            id: "diag-task-1".to_string(),
            goal_id: goal.id.clone(),
            description: "Run maintenance".to_string(),
            status: "failed".to_string(),
            priority: "medium".to_string(),
            task_order: 1,
            parallel_group: None,
            depends_on: None,
            agent_id: None,
            context: None,
            result: None,
            error: Some("Connection timeout to embedding backend".to_string()),
            blocker: None,
            idempotent: true,
            retry_count: 1,
            max_retries: 3,
            created_at: chrono::Utc::now().to_rfc3339(),
            started_at: None,
            completed_at: Some(chrono::Utc::now().to_rfc3339()),
        };
        state.create_task_v3(&task).await.unwrap();

        let activity = TaskActivityV3 {
            id: 0,
            task_id: task.id.clone(),
            activity_type: "tool_result".to_string(),
            tool_name: Some("http_request".to_string()),
            tool_args: Some("{\"url\":\"https://example.com\"}".to_string()),
            result: Some("timeout while contacting backend".to_string()),
            success: Some(false),
            tokens_used: None,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        state.log_task_activity_v3(&activity).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "diagnose_scheduled",
                    "goal_id_v3": goal.id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Scheduled Goal Diagnosis"));
        assert!(result.contains("Latest Failed Task"));
        assert!(result.contains("Connection timeout to embedding backend"));
        assert!(result.contains("Recent Activity"));
    }
}
