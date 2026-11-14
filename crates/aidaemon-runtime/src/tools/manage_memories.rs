use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::traits::{
    AutomationStore, Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus, semantics_for_exact_read_actions,
};

/// `manage_memories` as the daemon offers it: the memory actions of
/// [`aidaemon_memory::full::tools::ManageMemoriesTool`] and the goal
/// actions, which are the daemon's work, as one tool with one schema.
pub struct ManageMemoriesTool {
    state: Arc<dyn AutomationStore>,
    memory: aidaemon_memory::full::tools::ManageMemoriesTool,
}

/// Schedule actions this tool no longer has, and where each one went.
/// A call to one is answered with its new home and runs nothing.
const MOVED_ACTIONS: &[(&str, &str)] = &[
    (
        "create_scheduled_goal",
        "schedule_task (for a plain reminder message, set_reminder)",
    ),
    ("activate_scheduled", "manage_automations list"),
    ("list_scheduled", "manage_automations list"),
    ("list_scheduled_matching", "manage_automations list"),
    ("add_schedule", "manage_automations edit with `when`"),
    (
        "cancel_scheduled",
        "manage_automations cancel (by name, ref, or `ids` from manage_automations list)",
    ),
    (
        "cancel_scheduled_matching",
        "manage_automations list, then manage_automations cancel with the chosen refs as `ids`",
    ),
    ("pause_scheduled", "manage_automations pause"),
    ("resume_scheduled", "manage_automations resume"),
    ("retry_scheduled", "manage_automations resume"),
    (
        "retry_failed_scheduled",
        "manage_automations list, then manage_automations resume with the chosen refs as `ids`",
    ),
    (
        "retry_scheduled_matching",
        "manage_automations list, then manage_automations resume with the chosen refs as `ids`",
    ),
    (
        "diagnose_scheduled",
        "manage_automations last_failure or history",
    ),
    ("trigger_now", "manage_automations run_now"),
];

/// Where a schedule action this tool no longer has went, if `action` is one.
fn moved_action(action: &str) -> Option<&'static str> {
    MOVED_ACTIONS
        .iter()
        .find(|(name, _)| *name == action)
        .map(|(_, home)| *home)
}

impl ManageMemoriesTool {
    pub fn new(
        state: Arc<dyn AutomationStore>,
        store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    ) -> Self {
        Self {
            state,
            memory: aidaemon_memory::full::tools::ManageMemoriesTool::new(store),
        }
    }

    /// Resolve a personal goal identifier (domain = "personal") by exact ID or unique prefix.
    async fn resolve_personal_goal_id(&self, input_id: &str) -> anyhow::Result<String> {
        let trimmed = input_id.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Empty goal ID");
        }

        // Fast path: exact match + domain check.
        if let Some(g) = self.state.get_goal(trimmed).await?
            && g.domain == "personal"
        {
            return Ok(trimmed.to_string());
        }

        let goals = self.state.get_active_personal_goals(100).await?;
        let mut matches: Vec<&crate::tools::traits::Goal> =
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
}

#[derive(Deserialize)]
// Every field is parsed, so a malformed call fails here whatever its
// action; the memory actions' fields are read by the memory tool.
#[allow(dead_code)]
struct ManageArgs {
    action: String,
    limit: Option<usize>,
    category: Option<String>,
    key: Option<String>,
    privacy: Option<String>,
    query: Option<String>,
    goal: Option<String>,
    priority: Option<String>,
    goal_id: Option<String>,
}

/// The tool's one description: the schema the model reads and
/// `description()` both use it.
const TOOL_DESCRIPTION: &str =
    "Memories/goals. search=facts; search_episodes=past chats. Store via remember_fact.";

fn manage_memories_schema() -> Value {
    json!({
        "name": "manage_memories",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "health", "repair", "forget", "set_privacy", "search", "search_episodes", "create_personal_goal", "list_goals", "complete_goal", "abandon_goal"]
                },
                "limit": { "type": "integer" },
                "category": { "type": "string" },
                "key": { "type": "string" },
                "privacy": { "type": "string", "enum": ["global", "channel", "private"] },
                "query": { "type": "string" },
                "goal": { "type": "string" },
                "priority": { "type": "string", "enum": ["low", "medium", "high", "critical"] },
                "goal_id": { "type": "string" }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

/// Typed effects for memory-store mutations. Memory is persistent agent
/// state, so writes are `configuration` and a forget is also `destructive`;
/// an untyped receipt could never close a checklist step the model declared
/// with either label. Personal-goal actions are goal bookkeeping.
fn memory_action_effects(arguments: &str) -> ToolMutationEffects {
    match action_of(arguments).as_deref() {
        Some("forget") => {
            ToolMutationEffects::CONFIGURATION.union(ToolMutationEffects::DESTRUCTIVE)
        }
        Some("set_privacy") => ToolMutationEffects::CONFIGURATION,
        Some("create_personal_goal" | "complete_goal" | "abandon_goal") => {
            ToolMutationEffects::SCHEDULED_WORK
        }
        _ => ToolMutationEffects::NONE,
    }
}

/// Actions that only read, including a moved schedule action, which runs
/// nothing.
fn is_read_action(action: &str) -> bool {
    matches!(
        action,
        "list" | "health" | "search" | "search_episodes" | "list_goals"
    ) || moved_action(action).is_some()
}

#[async_trait]
impl Tool for ManageMemoriesTool {
    fn name(&self) -> &str {
        "manage_memories"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        manage_memories_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        self.outcome(arguments, facts).await
    }
}

impl crate::tools::traits::ToolSemantics for ManageMemoriesTool {
    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            offering: aidaemon_tools::semantics::Offering::Core,
            status_from_arguments: true,
            ..Default::default()
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        if action_of(arguments).is_some_and(|action| moved_action(&action).is_some()) {
            return ToolCallSemantics::observation();
        }
        semantics_for_exact_read_actions(
            arguments,
            &["list", "health", "search", "search_episodes", "list_goals"],
            memory_action_effects(arguments),
        )
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    /// Reads, `repair` and `forget` touch only the agent's own memory store,
    /// and a moved schedule action runs nothing. `set_privacy` widens who
    /// may see a fact, and the goal actions start work, so those stay
    /// undeclared.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        match action_of(arguments).as_deref() {
            Some(action) if is_read_action(action) => aidaemon_core::CallEffects::none(),
            Some("repair" | "forget") => aidaemon_core::CallEffects::none(),
            _ => aidaemon_core::CallEffects::unknown(),
        }
    }
}

/// The `action` argument, if any.
fn action_of(arguments: &str) -> Option<String> {
    serde_json::from_str::<Value>(arguments)
        .ok()?
        .get("action")?
        .as_str()
        .map(str::to_string)
}

impl ManageMemoriesTool {
    /// Run one call with the dispatcher's `facts` (session, role, channel,
    /// trust), never facts from the arguments.
    pub(crate) async fn run(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<String> {
        let args: ManageArgs = serde_json::from_str(arguments)?;

        match args.action.as_str() {
            action if aidaemon_memory::full::tools::MEMORY_ACTIONS.contains(&action) => {
                self.memory.run(arguments, facts).await
            }
            "list_goals" => {
                let limit = args.limit.unwrap_or(50).clamp(1, 200) as i64;
                let mut goals = self.state.get_active_personal_goals(limit).await?;
                if goals.is_empty() {
                    return Ok("No active personal goals.".to_string());
                }

                let owner_session = facts
                    .session()
                    .map(str::trim)
                    .filter(|session| !session.is_empty());
                goals.sort_by_key(|goal| {
                    if owner_session.is_some_and(|session| goal.session_id == session) {
                        0
                    } else {
                        1
                    }
                });

                let mut output = format!("**Active Personal Goals** ({} goals)\n\n", goals.len());
                output.push_str(
                    "Only goals marked `mandate-source compatible` may be supplied as a new mandate's source_goal_id. Existing mandates retain their current source; omit source_goal_id when creating a mandate without compatible provenance.\n\n",
                );
                for g in &goals {
                    let notes_count = g.progress_notes.as_ref().map_or(0, |n| n.len());
                    let age_str = Self::format_age(&g.created_at);
                    let compatibility =
                        if owner_session.is_some_and(|session| g.session_id == session) {
                            "mandate-source compatible"
                        } else {
                            "not compatible with this mandate owner session"
                        };
                    output.push_str(&format!(
                        "- **[ID: {}]** {} (priority: {}, created: {}, {} progress notes; {})\n",
                        g.id, g.description, g.priority, age_str, notes_count, compatibility
                    ));
                }
                Ok(output)
            }
            "create_personal_goal" => {
                let session_id = facts
                    .session()
                    .map(str::trim)
                    .filter(|session| !session.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Internal error: create_personal_goal: the session fact is missing from the call context."
                        )
                    })?;
                let desc = args
                    .goal
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!("'goal' is required for create_personal_goal action")
                    })?;

                // Personal goals may be visible across the owner's private contexts, but the
                // session that actually created them is an authorization boundary. Mandates use
                // this binding to prevent a goal from another conversation from delegating
                // external authority.
                let mut goal = crate::tools::traits::Goal::new_personal(desc, session_id);
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
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for complete_goal action")
                })?;
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
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for abandon_goal action")
                })?;
                let resolved_goal_id = match self.resolve_personal_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                self.state
                    .update_personal_goal(&resolved_goal_id, Some("abandoned"), None)
                    .await?;
                Ok(format!(
                    "Goal {} marked as abandoned. It will not be re-created by automatic analysis.",
                    resolved_goal_id
                ))
            }
            action if moved_action(action).is_some() => Ok(format!(
                "manage_memories no longer has `{action}`; nothing was run. Use {}.",
                moved_action(action).unwrap_or_default()
            )),
            other => Ok(format!(
                "Unknown action: '{}'. Use list, health, repair, forget, set_privacy, search, search_episodes, create_personal_goal, list_goals, complete_goal, or abandon_goal. Scheduled work is schedule_task, set_reminder and manage_automations.",
                other
            )),
        }
    }

    /// A call's typed outcome, with the dispatcher's `facts`.
    pub(crate) async fn outcome(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let moved = action_of(arguments).is_some_and(|action| moved_action(&action).is_some());
        let output = self.run(arguments, facts).await?;
        let trimmed = output.trim_start();
        let outcome_status = if moved {
            ToolOutcomeStatus::FailedPermanent
        } else if trimmed.starts_with("Internal error:")
            || trimmed.starts_with("Invalid priority")
            || trimmed.starts_with("Unknown action:")
            || trimmed.starts_with("Personal goal not found:")
        {
            ToolOutcomeStatus::FailedPermanent
        } else if trimmed.starts_with("No matching") {
            ToolOutcomeStatus::CompletedWithNegativeResult
        } else {
            ToolOutcomeStatus::Succeeded
        };
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(outcome_status),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_personal_goal_action_reports_what_it_did_without_counting_as_outbound() {
        use crate::tools::traits::ToolMutationEffects as Effects;
        let effects = memory_action_effects(r#"{"action":"create_personal_goal"}"#);
        assert!(effects.satisfies(Effects::SCHEDULED_WORK));
        assert!(!effects.intersects(crate::tools::traits::EGRESS_EFFECTS));
    }

    #[test]
    fn every_moved_schedule_action_names_its_new_home_and_runs_nothing() {
        for (action, home) in MOVED_ACTIONS {
            assert!(
                home.starts_with("manage_automations") || home.starts_with("schedule_task"),
                "{action} -> {home}"
            );
            assert!(is_read_action(action), "{action}");
        }
        assert!(moved_action("search").is_none());
    }
}
