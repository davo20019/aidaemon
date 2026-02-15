use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

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
    /// - exact full goal ID
    /// - unique prefix (e.g., the 8-char short ID shown in list output)
    async fn resolve_goal_id(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        // Fast path: exact ID match.
        if self.state.get_goal(trimmed).await?.is_some() {
            return Ok(trimmed.to_string());
        }

        // Prefix fallback: match against scheduled goals because this tool's
        // schedule operations operate on scheduled goals.
        let goals = self.state.get_scheduled_goals().await?;
        let mut matches: Vec<&crate::traits::Goal> =
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
            "Goal ID prefix '{}' is ambiguous ({} matches): {}. Use full goal_id.",
            trimmed,
            matches.len(),
            preview
        );
    }

    /// Resolve a personal goal identifier (domain = "personal") by exact ID or unique prefix.
    async fn resolve_personal_goal_id(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        // Fast path: exact match + domain check.
        if let Some(g) = self.state.get_goal(trimmed).await? {
            if g.domain == "personal" {
                return Ok(trimmed.to_string());
            }
        }

        let goals = self.state.get_active_personal_goals(100).await?;
        let mut matches: Vec<&crate::traits::Goal> =
            goals.iter().filter(|g| g.id.starts_with(trimmed)).collect();

        if matches.is_empty() {
            anyhow::bail!("Personal goal not found: {}", trimmed);
        }
        if matches.len() == 1 {
            return Ok(matches.remove(0).id.clone());
        }

        // Prefer most recently created in ambiguous cases.
        matches.sort_by(|a, b| b.created_at.cmp(&a.created_at));

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
            "Goal ID prefix '{}' is ambiguous ({} matches): {}. Use full goal_id.",
            trimmed,
            matches.len(),
            preview
        );
    }

    fn is_protected_system_maintenance_goal(goal: &crate::traits::Goal) -> bool {
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

    fn goal_matches_query(goal: &crate::traits::Goal, query: &str) -> bool {
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

    fn parse_ts(ts: &str) -> Option<chrono::DateTime<chrono::Utc>> {
        chrono::DateTime::parse_from_rfc3339(ts)
            .ok()
            .map(|d| d.with_timezone(&chrono::Utc))
    }

    fn format_age(ts: &str) -> String {
        let Some(dt) = Self::parse_ts(ts) else {
            return "n/a".to_string();
        };
        let age = chrono::Utc::now() - dt;
        if age.num_days() > 0 {
            format!("{}d ago", age.num_days())
        } else if age.num_hours() > 0 {
            format!("{}h ago", age.num_hours())
        } else if age.num_minutes() > 0 {
            format!("{}m ago", age.num_minutes())
        } else {
            "just now".to_string()
        }
    }

    fn format_local(ts: &str) -> String {
        chrono::DateTime::parse_from_rfc3339(ts)
            .ok()
            .map(|dt| {
                dt.with_timezone(&chrono::Local)
                    .format("%Y-%m-%d %H:%M %Z")
                    .to_string()
            })
            .unwrap_or_else(|| ts.to_string())
    }
}

#[derive(Deserialize)]
struct ManageArgs {
    action: String,
    category: Option<String>,
    key: Option<String>,
    privacy: Option<String>,
    query: Option<String>,
    goal: Option<String>,
    priority: Option<String>,
    goal_id: Option<String>,
    schedule_id: Option<String>,
    schedule: Option<String>,
    schedules: Option<Vec<String>>,
    fire_policy: Option<String>,
    is_one_shot: Option<bool>,
    is_paused: Option<bool>,
    #[serde(default)]
    _session_id: Option<String>,
    #[serde(default)]
    _user_role: Option<String>,
}

#[async_trait]
impl Tool for ManageMemoriesTool {
    fn name(&self) -> &str {
        "manage_memories"
    }

    fn description(&self) -> &str {
        "List/search/forget memories, and list/add/cancel/pause/resume/retry/diagnose scheduled goals (accepts full or unique prefix goal_id; includes bulk retry for failed schedules)"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_memories",
            "description": "List, search, forget, or change privacy of stored memories and goals. Also create/list/manage personal goals, and list/create/add/cancel/pause/resume/retry/diagnose scheduled goals. IMPORTANT for scheduled-goal management: first call action='list_scheduled' or 'list_scheduled_matching' to get exact goal IDs (and schedule IDs), then call add_schedule/cancel_scheduled/pause_scheduled/resume_scheduled/retry_scheduled/diagnose_scheduled with goal_id (and optionally schedule_id). Use create_scheduled_goal to create a new scheduled goal from scratch. Use retry_failed_scheduled for one-shot recovery of failed goals (optionally filtered by query). Do not use terminal/sqlite for scheduled-goal management when this tool can do it. Protected system maintenance goals cannot be cancelled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "forget", "set_privacy", "search", "create_personal_goal", "list_goals", "complete_goal", "abandon_goal", "create_scheduled_goal", "list_scheduled", "list_scheduled_matching", "add_schedule", "cancel_scheduled", "pause_scheduled", "resume_scheduled", "retry_scheduled", "retry_failed_scheduled", "cancel_scheduled_matching", "retry_scheduled_matching", "diagnose_scheduled"],
                        "description": "Action to perform. For schedule operations: use list_scheduled or list_scheduled_matching first, then add_schedule/cancel_scheduled/pause_scheduled/resume_scheduled/retry_scheduled/diagnose_scheduled with exact goal_id (and optionally schedule_id). For bulk operations, use retry_failed_scheduled (all failed, optionally filtered), cancel_scheduled_matching, or retry_scheduled_matching with query."
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
                    "goal": {
                        "type": "string",
                        "description": "Goal description for create_personal_goal or create_scheduled_goal."
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Optional priority for create_personal_goal (default medium)."
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Goal ID for goal/schedule actions. Retrieve via list_goals/list_scheduled first."
                    },
                    "schedule_id": {
                        "type": "string",
                        "description": "Optional schedule ID for schedule-specific pause/resume/cancel operations. Retrieve via list_scheduled first."
                    },
                    "schedule": {
                        "type": "string",
                        "description": "Schedule string for add_schedule or create_scheduled_goal. Accepts natural text (e.g. 'daily at 9am', 'every day at 6am, 12pm, 6pm', 'every 6h') or a 5-field cron expression."
                    },
                    "schedules": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional list of schedules for create_scheduled_goal. Use when the user needs multiple schedules with different minutes (e.g. 8:05, 13:10, 21:30)."
                    },
                    "fire_policy": {
                        "type": "string",
                        "enum": ["coalesce", "always_fire"],
                        "description": "Optional schedule fire policy (default 'coalesce'). coalesce: skip if open tasks exist; always_fire: enqueue even if open tasks exist (capped)."
                    },
                    "is_one_shot": {
                        "type": "boolean",
                        "description": "Optional. When true, the schedule is deleted after it fires once."
                    },
                    "is_paused": {
                        "type": "boolean",
                        "description": "Optional. When true, the new schedule starts paused."
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
                let goals = self.state.get_active_personal_goals(50).await?;
                if goals.is_empty() {
                    return Ok("No active personal goals.".to_string());
                }

                let mut output =
                    format!("**Active Personal Goals** ({} goals)\n\n", goals.len());
                for g in &goals {
                    let notes_count = g.progress_notes.as_ref().map_or(0, |n| n.len());
                    let age_str = Self::format_age(&g.created_at);
                    output.push_str(&format!(
                        "- **[ID: {}]** {} (priority: {}, created: {}, {} progress notes)\n",
                        g.id, g.description, g.priority, age_str, notes_count
                    ));
                }
                Ok(output)
            }
            "create_personal_goal" => {
                let desc = args
                    .goal
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'goal' is required for create_personal_goal action"))?;

                let mut goal = crate::traits::Goal::new_personal(desc, "_global");
                if let Some(p) = args.priority.as_deref() {
                    let p = p.trim().to_ascii_lowercase();
                    if matches!(p.as_str(), "low" | "medium" | "high" | "critical") {
                        goal.priority = p;
                    } else {
                        return Ok(format!(
                            "Invalid priority '{}'. Use low, medium, high, or critical.",
                            p
                        ));
                    }
                }

                self.state.create_goal(&goal).await?;
                Ok(format!(
                    "Created personal goal {}: {}",
                    goal.id, goal.description
                ))
            }
            "complete_goal" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for complete_goal action"))?;
                let resolved_goal_id = match self.resolve_personal_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                self.state
                    .update_personal_goal(&resolved_goal_id, Some("completed"), None)
                    .await?;
                Ok(format!("Goal {} marked as completed.", resolved_goal_id))
            }
            "abandon_goal" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for abandon_goal action"))?;
                let resolved_goal_id = match self.resolve_personal_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                self.state
                    .update_personal_goal(&resolved_goal_id, Some("abandoned"), None)
                    .await?;
                Ok(format!("Goal {} marked as abandoned. It will not be re-created by automatic analysis.", resolved_goal_id))
            }
            "create_scheduled_goal" => {
                let is_owner = args
                    ._user_role
                    .as_deref()
                    .is_some_and(|r| r.eq_ignore_ascii_case("owner"));
                if !is_owner {
                    return Ok("Only owners can create scheduled goals.".to_string());
                }

                let session_id = args._session_id.as_deref().unwrap_or("");
                if session_id.trim().is_empty() {
                    return Ok("Internal error: create_scheduled_goal requires _session_id.".to_string());
                }

                let desc = args
                    .goal
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'goal' is required for create_scheduled_goal action"))?;

                let schedule_inputs: Vec<String> = if let Some(list) = args.schedules.as_ref() {
                    list.iter()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                } else {
                    let schedule_raw = args
                        .schedule
                        .as_deref()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| anyhow::anyhow!("'schedule' is required for create_scheduled_goal action"))?;
                    vec![schedule_raw.to_string()]
                };
                if schedule_inputs.is_empty() {
                    return Ok("At least one schedule is required.".to_string());
                }

                // Parse schedules first so we don't create partial goals on failure.
                struct ParsedSchedule {
                    original: String,
                    cron: String,
                    is_one_shot: bool,
                    next_local: chrono::DateTime<chrono::Local>,
                }
                let mut parsed = Vec::new();
                for schedule_raw in &schedule_inputs {
                    let cron_expr = match crate::cron_utils::parse_schedule(schedule_raw) {
                        Ok(expr) => expr,
                        Err(e) => {
                            return Ok(format!(
                                "Couldn't parse schedule '{}': {}",
                                schedule_raw, e
                            ))
                        }
                    };
                    let next_local = match crate::cron_utils::compute_next_run_local(&cron_expr) {
                        Ok(dt) => dt,
                        Err(e) => {
                            return Ok(format!(
                                "Couldn't compute next run for '{}': {}",
                                cron_expr, e
                            ))
                        }
                    };
                    let is_one_shot = args
                        .is_one_shot
                        .unwrap_or_else(|| crate::cron_utils::is_one_shot_schedule(&cron_expr));
                    parsed.push(ParsedSchedule {
                        original: schedule_raw.to_string(),
                        cron: cron_expr,
                        is_one_shot,
                        next_local,
                    });
                }

	                let has_recurring = parsed.iter().any(|p| !p.is_one_shot);
	                let mut goal = if has_recurring {
	                    crate::traits::Goal::new_continuous_pending(desc, session_id, None, None)
	                } else {
	                    crate::traits::Goal::new_deferred_finite(desc, session_id)
	                };
                goal.domain = "orchestration".to_string();

                self.state.create_goal(&goal).await?;

                let fire_policy = match args.fire_policy.as_deref() {
                    Some("always_fire") => "always_fire".to_string(),
                    Some("coalesce") | None => "coalesce".to_string(),
                    Some(other) => {
                        return Ok(format!(
                            "Invalid fire_policy '{}'. Use 'coalesce' or 'always_fire'.",
                            other
                        ))
                    }
                };
                let paused = args.is_paused.unwrap_or(false);
                let now = chrono::Utc::now().to_rfc3339();

                for p in &parsed {
                    let schedule = crate::traits::GoalSchedule {
                        id: uuid::Uuid::new_v4().to_string(),
                        goal_id: goal.id.clone(),
                        cron_expr: p.cron.clone(),
                        tz: "local".to_string(),
                        original_schedule: Some(p.original.clone()),
                        fire_policy: fire_policy.clone(),
                        is_one_shot: p.is_one_shot,
                        is_paused: paused,
                        last_run_at: None,
                        next_run_at: p.next_local
                            .with_timezone(&chrono::Utc)
                            .to_rfc3339(),
                        created_at: now.clone(),
                        updated_at: now.clone(),
                    };
                    self.state.create_goal_schedule(&schedule).await?;
                }

                let tz_label = crate::cron_utils::system_timezone_display();
                let next_run = parsed
                    .iter()
                    .map(|p| p.next_local)
                    .min_by_key(|dt| dt.timestamp())
                    .map(|dt| dt.format("%Y-%m-%d %H:%M %Z").to_string())
                    .unwrap_or_else(|| "n/a".to_string());

                Ok(format!(
                    "Created scheduled goal {} (pending confirmation) with {} schedule(s). Next: {}. System timezone: {}. Reply **confirm** to activate or **cancel** to discard.",
                    goal.id,
                    parsed.len(),
                    next_run,
                    tz_label
                ))
            }
            "list_scheduled" => {
                let goals = self.state.get_scheduled_goals().await?;
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

                output.push_str(&format!(
                    "System timezone: {}.\nTip: use `schedule_id` for schedule-specific pause/resume/cancel.\n\n",
                    crate::cron_utils::system_timezone_display()
                ));

                let groups: [(&str, &Vec<&crate::traits::Goal>); 7] = [
                    ("Active", &active),
                    ("Paused", &paused),
                    ("Pending Confirmation", &pending_confirmation),
                    ("Failed", &failed),
                    ("Cancelled", &cancelled),
                    ("Completed", &completed),
                    ("Other", &other),
                ];

                for (title, items) in groups {
                    if items.is_empty() {
                        continue;
                    }
                    output.push_str(&format!("**{}** ({})\n", title, items.len()));
                    for g in items.iter() {
                        let desc: String = g.description.chars().take(80).collect();
                        let schedules = self
                            .state
                            .get_schedules_for_goal(&g.id)
                            .await
                            .unwrap_or_default();
                        let has_recurring = schedules.iter().any(|s| !s.is_one_shot);
                        let has_one_shot = schedules.iter().any(|s| s.is_one_shot);
                        let goal_type = if has_recurring {
                            "recurring"
                        } else if has_one_shot {
                            "one-time"
                        } else if g.goal_type == "continuous" {
                            "recurring"
                        } else {
                            "one-time"
                        };
                        output.push_str(&format!(
                            "- **{}** {} (type: {}, status: {}, schedules: {})\n",
                            g.id,
                            desc,
                            goal_type,
                            g.status,
                            schedules.len()
                        ));
                        for s in &schedules {
                            let next_local = Self::format_local(&s.next_run_at);
                            output.push_str(&format!(
                                "  schedule {}: next {}, paused {}, policy {}, one_shot {}, cron {}\n",
                                s.id,
                                next_local,
                                s.is_paused,
                                s.fire_policy,
                                s.is_one_shot,
                                s.cron_expr
                            ));
                        }
                    }
                    output.push('\n');
                }
                Ok(output)
            }
            "list_scheduled_matching" => {
                let query = args
                    .query
                    .as_deref()
                    .map(str::trim)
                    .filter(|q| !q.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'query' is required for list_scheduled_matching action"))?;
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::traits::Goal> = goals
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
                    let schedules = self
                        .state
                        .get_schedules_for_goal(&g.id)
                        .await
                        .unwrap_or_default();
                    let has_recurring = schedules.iter().any(|s| !s.is_one_shot);
                    let has_one_shot = schedules.iter().any(|s| s.is_one_shot);
                    let goal_type = if has_recurring {
                        "recurring"
                    } else if has_one_shot {
                        "one-time"
                    } else if g.goal_type == "continuous" {
                        "recurring"
                    } else {
                        "one-time"
                    };
                    let next_run = schedules
                        .iter()
                        .filter_map(|s| chrono::DateTime::parse_from_rfc3339(&s.next_run_at).ok())
                        .min_by_key(|dt| dt.timestamp())
                        .map(|dt| dt.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M %Z").to_string())
                        .unwrap_or_else(|| "n/a".to_string());
                    output.push_str(&format!(
                        "- **{}** {} (type: {}, status: {}, schedules: {}, next: {})\n",
                        g.id, desc, goal_type, g.status, schedules.len(), next_run
                    ));
                }
                Ok(output)
            }
            "add_schedule" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "'goal_id' is required for add_schedule action. Call action='list_scheduled' first to get the goal_id, or use action='create_scheduled_goal' to create a new scheduled goal."
                        )
                    })?;
                let schedule_raw = args
                    .schedule
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| anyhow::anyhow!("'schedule' is required for add_schedule action"))?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };

                let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.domain != "orchestration" {
                    return Ok(format!(
                        "Personal goals cannot be scheduled. Goal {} has domain '{}'.",
                        goal.id, goal.domain
                    ));
                }

                let cron_expr = match crate::cron_utils::parse_schedule(schedule_raw) {
                    Ok(expr) => expr,
                    Err(e) => {
                        return Ok(format!(
                            "Couldn't parse schedule '{}': {}",
                            schedule_raw, e
                        ))
                    }
                };
                let next_local = match crate::cron_utils::compute_next_run_local(&cron_expr) {
                    Ok(dt) => dt,
                    Err(e) => {
                        return Ok(format!(
                            "Couldn't compute next run for '{}': {}",
                            cron_expr, e
                        ))
                    }
                };

                let now = chrono::Utc::now().to_rfc3339();
                let schedule = crate::traits::GoalSchedule {
                    id: uuid::Uuid::new_v4().to_string(),
                    goal_id: goal.id.clone(),
                    cron_expr: cron_expr.clone(),
                    tz: "local".to_string(),
                    original_schedule: Some(schedule_raw.to_string()),
                    fire_policy: match args.fire_policy.as_deref() {
                        Some("always_fire") => "always_fire".to_string(),
                        Some("coalesce") | None => "coalesce".to_string(),
                        Some(other) => {
                            return Ok(format!(
                                "Invalid fire_policy '{}'. Use 'coalesce' or 'always_fire'.",
                                other
                            ))
                        }
                    },
                    is_one_shot: args
                        .is_one_shot
                        .unwrap_or_else(|| crate::cron_utils::is_one_shot_schedule(&cron_expr)),
                    is_paused: args.is_paused.unwrap_or(false),
                    last_run_at: None,
                    next_run_at: next_local.with_timezone(&chrono::Utc).to_rfc3339(),
                    created_at: now.clone(),
                    updated_at: now,
                };
                self.state.create_goal_schedule(&schedule).await?;

                Ok(format!(
                    "Added schedule {} to goal {} (next: {}).",
                    schedule.id,
                    goal.id,
                    next_local.format("%Y-%m-%d %H:%M %Z")
                ))
            }
            "cancel_scheduled" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for cancel_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if let Some(schedule_id) = args.schedule_id.as_deref() {
                    let Some(sched) = self.state.get_goal_schedule(schedule_id).await? else {
                        return Ok(format!("Schedule not found: {}", schedule_id));
                    };
                    if sched.goal_id != goal.id {
                        return Ok(format!(
                            "Schedule {} does not belong to goal {}.",
                            sched.id, goal.id
                        ));
                    }
                    let deleted = self.state.delete_goal_schedule(&sched.id).await?;
                    if deleted {
                        return Ok(format!(
                            "Cancelled schedule {} for goal {}.",
                            sched.id, goal.id
                        ));
                    }
                    return Ok(format!("Schedule {} was already removed.", sched.id));
                }

                if Self::is_protected_system_maintenance_goal(&goal) {
                    return Ok(format!(
                        "Cannot cancel protected system maintenance goal {}.",
                        resolved_goal_id
                    ));
                }

                // Capture schedules before transitioning status; state.update_goal now
                // purges schedules for terminal goals as a safety net.
                let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
                goal.status = "cancelled".to_string();
                goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                goal.updated_at = chrono::Utc::now().to_rfc3339();
                self.state.update_goal(&goal).await?;

                // Clean up schedules so they no longer appear in listings.
                for s in &schedules {
                    let _ = self.state.delete_goal_schedule(&s.id).await;
                }

                Ok(format!(
                    "Cancelled scheduled goal {} (deleted {} schedule(s)).",
                    resolved_goal_id,
                    schedules.len()
                ))
            }
            "pause_scheduled" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for pause_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                if let Some(schedule_id) = args.schedule_id.as_deref() {
                    let Some(mut sched) = self.state.get_goal_schedule(schedule_id).await? else {
                        return Ok(format!("Schedule not found: {}", schedule_id));
                    };
                    if sched.goal_id != resolved_goal_id {
                        return Ok(format!(
                            "Schedule {} does not belong to goal {}.",
                            sched.id, resolved_goal_id
                        ));
                    }
                    sched.is_paused = true;
                    sched.updated_at = chrono::Utc::now().to_rfc3339();
                    self.state.update_goal_schedule(&sched).await?;
                    return Ok(format!("Paused schedule {}.", sched.id));
                }

                let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.status != "active" {
                    return Ok(format!(
                        "Only active scheduled goals can be paused (current status: {}).",
                        goal.status
                    ));
                }
                goal.status = "paused".to_string();
                goal.updated_at = chrono::Utc::now().to_rfc3339();
                self.state.update_goal(&goal).await?;
                Ok(format!("Paused scheduled goal {}.", resolved_goal_id))
            }
            "resume_scheduled" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for resume_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                if let Some(schedule_id) = args.schedule_id.as_deref() {
                    let Some(mut sched) = self.state.get_goal_schedule(schedule_id).await? else {
                        return Ok(format!("Schedule not found: {}", schedule_id));
                    };
                    if sched.goal_id != resolved_goal_id {
                        return Ok(format!(
                            "Schedule {} does not belong to goal {}.",
                            sched.id, resolved_goal_id
                        ));
                    }
                    sched.is_paused = false;
                    sched.updated_at = chrono::Utc::now().to_rfc3339();
                    self.state.update_goal_schedule(&sched).await?;
                    return Ok(format!("Resumed schedule {}.", sched.id));
                }

                let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
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
                self.state.update_goal(&goal).await?;
                Ok(format!("Resumed scheduled goal {}.", resolved_goal_id))
            }
            "retry_scheduled" => {
                let goal_id = args
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for retry_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
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
                self.state.update_goal(&goal).await?;
                Ok(format!(
                    "Retried scheduled goal {}. It is active again.",
                    resolved_goal_id
                ))
            }
            "retry_failed_scheduled" => {
                let query = args.query.as_deref().map(str::trim).unwrap_or("");
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::traits::Goal> = goals
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
                    match self.state.update_goal(&updated).await {
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
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::traits::Goal> = goals
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
                    match self.state.update_goal(&updated).await {
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
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::traits::Goal> = goals
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
                    match self.state.update_goal(&updated).await {
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
                    .goal_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'goal_id' is required for diagnose_scheduled action"))?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };

                let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
                let has_recurring = schedules.iter().any(|s| !s.is_one_shot);
                let has_one_shot = schedules.iter().any(|s| s.is_one_shot);
                let goal_type = if has_recurring {
                    "recurring"
                } else if has_one_shot {
                    "one-time"
                } else if goal.goal_type == "continuous" {
                    "recurring"
                } else {
                    "one-time"
                };
                let next_run = schedules
                    .iter()
                    .filter_map(|s| chrono::DateTime::parse_from_rfc3339(&s.next_run_at).ok())
                    .min_by_key(|dt| dt.timestamp())
                    .map(|dt| dt.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M %Z").to_string())
                    .unwrap_or_else(|| "n/a".to_string());

                let tasks = self.state.get_tasks_for_goal(&goal.id).await?;
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
                    "**Scheduled Goal Diagnosis**\n\n- ID: {}\n- Description: {}\n- Type: {}\n- Status: {}\n- Next: {}\n- Schedules: {}\n- Tasks: total {}, failed {}, running {}, pending {}, completed {}",
                    goal.id,
                    goal.description,
                    goal_type,
                    goal.status,
                    next_run,
                    schedules.len(),
                    task_total,
                    task_failed,
                    task_running,
                    task_pending,
                    task_completed
                );

                if schedules.is_empty() {
                    out.push_str("\n\nNo schedules found for this goal.");
                } else {
                    out.push_str("\n\n**Schedules**");
                    for s in &schedules {
                        out.push_str(&format!(
                            "\n- {} next={} paused={} one_shot={} policy={} cron={}",
                            s.id,
                            Self::format_local(&s.next_run_at),
                            s.is_paused,
                            s.is_one_shot,
                            s.fire_policy,
                            s.cron_expr
                        ));
                    }
                }

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
                        .get_task_activities(&last_failed_task.id)
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
            other => Ok(format!("Unknown action: '{}'. Use list, forget, set_privacy, search, create_personal_goal, list_goals, complete_goal, abandon_goal, create_scheduled_goal, list_scheduled, list_scheduled_matching, add_schedule, cancel_scheduled, pause_scheduled, resume_scheduled, retry_scheduled, retry_failed_scheduled, cancel_scheduled_matching, retry_scheduled_matching, or diagnose_scheduled.", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::{Goal, GoalSchedule, Task, TaskActivity};

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

    async fn add_schedule(state: &Arc<dyn StateStore>, goal_id: &str, cron_expr: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let next_run = crate::cron_utils::compute_next_run(cron_expr)
            .unwrap()
            .to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal_id.to_string(),
            cron_expr: cron_expr.to_string(),
            tz: "local".to_string(),
            original_schedule: Some(cron_expr.to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: next_run,
            created_at: now.clone(),
            updated_at: now,
        };
        state.create_goal_schedule(&schedule).await.unwrap();
    }

    #[tokio::test]
    async fn add_schedule_creates_schedule_row() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = Goal::new_continuous(
            "English research for pronunciation",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "add_schedule",
                    "goal_id": goal_id,
                    "schedule": "daily at 9am"
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Added schedule"));

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0].cron_expr, "0 9 * * *");
        assert!(!schedules[0].is_paused);
    }

    #[tokio::test]
    async fn add_schedule_rejects_personal_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut goal = Goal::new_finite("Personal goal", "user-session");
        goal.domain = "personal".to_string();
        state.create_goal(&goal).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "add_schedule",
                    "goal_id": goal.id,
                    "schedule": "daily at 9am"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Personal goals cannot be scheduled"));
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules.is_empty());
    }

    #[tokio::test]
    async fn cancel_scheduled_blocks_protected_system_maintenance_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Maintain memory health: prune old events, clean up retention, remove stale data",
            "system",
            Some(1000),
            Some(5000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "30 3 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Cannot cancel protected system maintenance goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
    }

    #[tokio::test]
    async fn cancel_scheduled_allows_non_system_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = Goal::new_continuous(
            "English research for pronunciation",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 5,12,19 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "cancel_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Cancelled scheduled goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "cancelled");
    }

    #[tokio::test]
    async fn retry_scheduled_reactivates_failed_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut goal = Goal::new_continuous(
            "Retryable scheduled goal",
            "user-session",
            Some(2000),
            Some(20000),
        );
        goal.status = "failed".to_string();
        goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 */6 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Retried scheduled goal"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
        assert!(fetched.completed_at.is_none());
    }

    #[tokio::test]
    async fn retry_scheduled_rejects_non_failed_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let goal = Goal::new_continuous(
            "Already active scheduled goal",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let goal_id = goal.id.clone();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 */6 * * *").await;

        let result = tool
            .call(
                &json!({
                    "action": "retry_scheduled",
                    "goal_id": goal_id
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Only failed scheduled goals can be retried"));
        let fetched = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(fetched.status, "active");
    }

    #[tokio::test]
    async fn retry_failed_scheduled_reactivates_all_failed() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut failed_one = Goal::new_continuous(
            "Maintain knowledge base: process embeddings, consolidate memories, decay old facts",
            "system",
            Some(1000),
            Some(5000),
        );
        failed_one.status = "failed".to_string();
        let failed_one_id = failed_one.id.clone();
        state.create_goal(&failed_one).await.unwrap();
        add_schedule(&state, &failed_one.id, "0 */6 * * *").await;

        let mut failed_two = Goal::new_continuous(
            "English pronunciation slot A",
            "user-session",
            Some(1000),
            Some(5000),
        );
        failed_two.status = "failed".to_string();
        let failed_two_id = failed_two.id.clone();
        state.create_goal(&failed_two).await.unwrap();
        add_schedule(&state, &failed_two.id, "0 5 * * *").await;

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
        let fetched_one = state.get_goal(&failed_one_id).await.unwrap().unwrap();
        let fetched_two = state.get_goal(&failed_two_id).await.unwrap().unwrap();
        assert_eq!(fetched_one.status, "active");
        assert_eq!(fetched_two.status, "active");
    }

    #[tokio::test]
    async fn cancel_scheduled_matching_cancels_matching_goals() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let g1 = Goal::new_continuous(
            "English pronunciation slot A",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let g2 = Goal::new_continuous(
            "English pronunciation slot B",
            "user-session",
            Some(2000),
            Some(20000),
        );
        let g3 = Goal::new_continuous(
            "Unrelated recurring task",
            "user-session",
            Some(2000),
            Some(20000),
        );
        state.create_goal(&g1).await.unwrap();
        state.create_goal(&g2).await.unwrap();
        state.create_goal(&g3).await.unwrap();
        add_schedule(&state, &g1.id, "0 5 * * *").await;
        add_schedule(&state, &g2.id, "0 12 * * *").await;
        add_schedule(&state, &g3.id, "0 19 * * *").await;

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
        let g1_after = state.get_goal(&g1.id).await.unwrap().unwrap();
        let g2_after = state.get_goal(&g2.id).await.unwrap().unwrap();
        let g3_after = state.get_goal(&g3.id).await.unwrap().unwrap();
        assert_eq!(g1_after.status, "cancelled");
        assert_eq!(g2_after.status, "cancelled");
        assert_eq!(g3_after.status, "active");

        // Schedules for cancelled goals should be purged.
        assert_eq!(state.get_schedules_for_goal(&g1.id).await.unwrap().len(), 0);
        assert_eq!(state.get_schedules_for_goal(&g2.id).await.unwrap().len(), 0);
        assert_eq!(state.get_schedules_for_goal(&g3.id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn diagnose_scheduled_reports_latest_failed_task() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let mut goal = Goal::new_continuous("Diagnose me", "user-session", Some(2000), Some(20000));
        goal.status = "failed".to_string();
        state.create_goal(&goal).await.unwrap();
        add_schedule(&state, &goal.id, "0 */6 * * *").await;

        let task = Task {
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
        state.create_task(&task).await.unwrap();

        let activity = TaskActivity {
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
        state.log_task_activity(&activity).await.unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "diagnose_scheduled",
                    "goal_id": goal.id
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

    // ==================== create_personal_goal tests ====================

    #[tokio::test]
    async fn create_personal_goal_creates_goal_in_registry() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Learn conversational Spanish"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created personal goal"));
        assert!(result.contains("Learn conversational Spanish"));

        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].description, "Learn conversational Spanish");
        assert_eq!(goals[0].domain, "personal");
        assert_eq!(goals[0].priority, "medium");
        assert_eq!(goals[0].status, "active");
    }

    #[tokio::test]
    async fn create_personal_goal_with_priority() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Exercise daily",
                    "priority": "high"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created personal goal"));
        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert_eq!(goals[0].priority, "high");
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_invalid_priority() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Some goal",
                    "priority": "ultra"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Invalid priority"));
        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_empty_description() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "   "
                })
                .to_string(),
            )
            .await;

        assert!(result.is_err() || result.unwrap().contains("required"));
    }

    // ==================== create_scheduled_goal tests ====================

    #[tokio::test]
    async fn create_scheduled_goal_success_recurring() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Check API health",
                    "schedule": "every 6h",
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created scheduled goal"));
        assert!(result.contains("pending confirmation"));
        assert!(result.contains("1 schedule(s)"));
        assert!(result.contains("confirm"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Check API health")
            .unwrap();
        assert_eq!(goal.status, "pending_confirmation");
        assert_eq!(goal.domain, "orchestration");
        assert_eq!(goal.goal_type, "continuous");
        assert_eq!(goal.budget_per_check, Some(50_000));
        assert_eq!(goal.budget_daily, Some(200_000));

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(!schedules[0].is_one_shot);
        assert_eq!(schedules[0].cron_expr, "0 */6 * * *");
    }

    #[tokio::test]
    async fn create_scheduled_goal_success_one_shot() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Deploy release",
                    "schedule": "in 2h",
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created scheduled goal"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Deploy release")
            .unwrap();
        assert_eq!(goal.goal_type, "finite");

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(schedules[0].is_one_shot);
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_non_owner() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Steal data",
                    "schedule": "every 1h",
                    "_user_role": "guest",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Only owners"));
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_missing_session_id() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Do something",
                    "schedule": "daily at 9am",
                    "_user_role": "owner"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("requires _session_id"));
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_bad_schedule() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Do something",
                    "schedule": "whenever the moon is full",
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Couldn't parse schedule"));
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_scheduled_goal_rejects_missing_schedule() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Do something",
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await;

        assert!(result.is_err() || result.unwrap().contains("required"));
    }

    #[tokio::test]
    async fn create_scheduled_goal_multiple_schedules() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Multi-slot monitoring",
                    "schedules": ["daily at 8am", "daily at 2pm", "daily at 9pm"],
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created scheduled goal"));
        assert!(result.contains("3 schedule(s)"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Multi-slot monitoring")
            .unwrap();
        assert_eq!(goal.goal_type, "continuous");

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert_eq!(schedules.len(), 3);
        let crons: Vec<&str> = schedules.iter().map(|s| s.cron_expr.as_str()).collect();
        assert!(crons.contains(&"0 8 * * *"));
        assert!(crons.contains(&"0 14 * * *"));
        assert!(crons.contains(&"0 21 * * *"));
    }

    #[tokio::test]
    async fn create_scheduled_goal_starts_paused() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Paused from start",
                    "schedule": "daily at 9am",
                    "is_paused": true,
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Created scheduled goal"));

        let goals = state.get_scheduled_goals().await.unwrap();
        let goal = goals
            .iter()
            .find(|g| g.description == "Paused from start")
            .unwrap();
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        assert!(schedules[0].is_paused);
    }

    #[tokio::test]
    async fn create_scheduled_goal_bad_schedule_in_multi_does_not_create_goal() {
        let state = setup_state().await;
        let tool = ManageMemoriesTool::new(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_scheduled_goal",
                    "goal": "Partial failure",
                    "schedules": ["daily at 8am", "whenever I feel like it"],
                    "_user_role": "owner",
                    "_session_id": "test-session"
                })
                .to_string(),
            )
            .await
            .unwrap();

        assert!(result.contains("Couldn't parse schedule"));
        // No goal should be created — fail-fast before DB writes
        let goals = state.get_scheduled_goals().await.unwrap();
        assert!(
            goals.iter().all(|g| g.description != "Partial failure"),
            "No goal should be created when any schedule fails to parse"
        );
    }
}
