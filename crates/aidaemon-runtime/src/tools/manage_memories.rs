use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::traits::{
    AutomationStore, Tool, ToolArgumentContractViolation, ToolCallContext, ToolCallMetadata,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus,
    semantics_for_exact_read_actions,
};
use crate::tools::traits::{ChannelVisibility, UserRole};

/// `manage_memories` as the daemon offers it: the memory actions of
/// [`aidaemon_memory::full::tools::ManageMemoriesTool`] and the goal
/// actions, which are the daemon's work, as one tool with one schema.
pub struct ManageMemoriesTool {
    state: Arc<dyn AutomationStore>,
    memory: aidaemon_memory::full::tools::ManageMemoriesTool,
}

/// One parsed schedule of a `create_scheduled_goal` call.
struct ParsedSchedule {
    original: String,
    cron: String,
    is_one_shot: bool,
    next_local: chrono::DateTime<chrono::Local>,
}

/// A validated `create_scheduled_goal` call, before anything is stored.
struct ScheduledGoalPlan {
    desc: String,
    session_id: String,
    parsed: Vec<ParsedSchedule>,
    fire_policy: String,
    reminder: Option<String>,
    /// The owner-given name, if any.
    name: Option<String>,
    next_run: String,
    /// The exact arguments the owner approves (the approval subject).
    subject: String,
}

/// What `create_scheduled_goal` does before storing anything: store the
/// plan, or answer without storing anything.
enum ScheduledGoalPrep {
    Plan(ScheduledGoalPlan),
    Answer(String),
}

impl ScheduledGoalPlan {
    /// The owner confirms a scheduled goal that runs the agent; a typed
    /// reminder runs no agent and only messages the owner, so it needs none.
    fn needs_owner_approval(&self) -> bool {
        self.reminder.is_none()
    }

    /// The card the owner confirms the goal with: the five lines every
    /// automation card has, then the next run and the name as details.
    fn approval_request(&self) -> aidaemon_core::OwnerApprovalRequest {
        let mut card = aidaemon_work::automations::scheduled_card(
            &self.desc,
            self.reminder.as_deref(),
            &self.schedule_specs(),
            &aidaemon_work::schedule::system_timezone_display(),
        );
        card.details.push(format!("Next: {}", self.next_run));
        if let Some(name) = self.name.as_deref() {
            card.details.push(format!("Name: {name}"));
        }
        aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: self.desc.clone(),
            subject: self.subject.clone(),
            risk: aidaemon_core::ApprovalRisk::Medium,
            warnings: card.into_lines(),
        }
    }

    fn schedule_specs(&self) -> Vec<aidaemon_work::automations::ScheduleSpec> {
        self.parsed
            .iter()
            .map(|p| aidaemon_work::automations::ScheduleSpec {
                cron: p.cron.clone(),
                one_shot: p.is_one_shot,
                next: Some(aidaemon_work::schedule::owner_time(&p.next_local)),
            })
            .collect()
    }
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

    /// Validate a `create_scheduled_goal` call with the dispatcher's facts
    /// and the stored goals, exactly as the call does before it stores
    /// anything. Shared by the call and its approval judgment, so the owner
    /// is asked only about a goal the call would create.
    async fn prepare_scheduled_goal(
        &self,
        args: &ManageArgs,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ScheduledGoalPrep> {
        use self::ScheduledGoalPrep::Answer;
        let is_owner = facts.user_role == Some(UserRole::Owner);
        if !is_owner {
            return Ok(Answer(
                "Only owners can create scheduled goals.".to_string(),
            ));
        }

        let session_id = facts.session().unwrap_or("");
        if session_id.trim().is_empty() {
            return Ok(Answer("Internal error: create_scheduled_goal: the session fact is missing from the call context.".to_string()));
        }
        let is_internal_visibility = facts.channel_visibility == Some(ChannelVisibility::Internal);
        // `sub-` is the legacy child-session prefix (replaced by
        // `specialist:`); accept both so an in-flight legacy session
        // can't bypass this guard.
        if is_internal_visibility
            || session_id.starts_with("sub-")
            || session_id.starts_with("specialist:")
        {
            return Ok(Answer("Cannot create scheduled goals from within internal scheduled-task execution. Execute the task directly instead.".to_string()));
        }

        let raw_desc = args
            .goal
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("'goal' is required for create_scheduled_goal action")
            })?;

        // Strict-schema providers send every field, so an unused
        // `schedules` arrives as `[]` beside the real `schedule`.
        let mut schedule_inputs: Vec<String> = args
            .schedules
            .iter()
            .flatten()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if schedule_inputs.is_empty()
            && let Some(schedule) = args.schedule.as_deref().map(str::trim)
            && !schedule.is_empty()
        {
            schedule_inputs.push(schedule.to_string());
        }
        if schedule_inputs.is_empty() {
            anyhow::bail!(
                "'schedule' (or a non-empty 'schedules' list) is required for create_scheduled_goal"
            );
        }

        // Keep tool-path descriptions aligned with fast-path scheduling:
        // remove explicit schedule phrases and leading filler prompts.
        let mut desc = raw_desc.to_string();
        for schedule_raw in &schedule_inputs {
            desc = aidaemon_work::schedule::clean_task_description(&desc, schedule_raw);
        }
        desc = aidaemon_work::schedule::clean_task_description(&desc, "");
        if desc.trim().is_empty() {
            desc = raw_desc.to_string();
        }

        // Parse schedules first so we don't create partial goals on failure.
        let mut parsed = Vec::new();
        for schedule_raw in &schedule_inputs {
            let cron_expr = match aidaemon_work::schedule::parse_schedule(schedule_raw) {
                Ok(expr) => expr,
                Err(e) => {
                    return Ok(Answer(format!(
                        "Couldn't parse schedule '{}': {}",
                        schedule_raw, e
                    )));
                }
            };
            let next_local = match aidaemon_work::schedule::compute_next_run_local(&cron_expr) {
                Ok(dt) => dt,
                Err(e) => {
                    return Ok(Answer(format!(
                        "Couldn't compute next run for '{}': {}",
                        cron_expr, e
                    )));
                }
            };
            let is_one_shot = args
                .is_one_shot
                .unwrap_or_else(|| aidaemon_work::schedule::is_one_shot_schedule(&cron_expr));
            parsed.push(ParsedSchedule {
                original: schedule_raw.to_string(),
                cron: cron_expr,
                is_one_shot,
                next_local,
            });
        }

        // Prevent duplicate schedules when the model repeats the same create request.
        let target_desc = Self::canonicalize_schedule_goal_description(&desc);
        let target_crons: BTreeSet<String> = parsed
            .iter()
            .map(|p| p.cron.trim().to_ascii_lowercase())
            .collect();
        let existing_goals = self.state.get_scheduled_goals().await.unwrap_or_default();
        for existing in existing_goals {
            if existing.session_id != session_id {
                continue;
            }
            if !matches!(
                existing.status.as_str(),
                "active" | "pending_confirmation" | "paused"
            ) {
                continue;
            }
            if Self::canonicalize_schedule_goal_description(&existing.description) != target_desc {
                continue;
            }
            let existing_schedules = self
                .state
                .get_schedules_for_goal(&existing.id)
                .await
                .unwrap_or_default();
            let existing_crons: BTreeSet<String> = existing_schedules
                .iter()
                .map(|s| s.cron_expr.trim().to_ascii_lowercase())
                .collect();
            if !existing_crons.is_empty() && existing_crons == target_crons {
                return Ok(Answer(format!(
                    "A similar scheduled goal already exists ({}). Use list_scheduled to inspect existing goals.",
                    existing.id
                )));
            }
        }

        let name = args
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        if let Some(name) = name.as_deref() {
            if name.chars().count() > MAX_AUTOMATION_NAME_CHARS
                || name.chars().any(char::is_control)
            {
                return Ok(Answer(format!(
                    "The name must be one line of at most {MAX_AUTOMATION_NAME_CHARS} characters."
                )));
            }
            if let Some(taken) = crate::tools::manage_automations::live_name_holder(
                self.state.as_ref(),
                session_id,
                name,
            )
            .await?
            {
                return Ok(Answer(format!(
                    "An automation is already named \"{name}\" ({taken}); pick another name or cancel that one first."
                )));
            }
        }

        // Checked before anything is stored, so an invalid policy leaves no
        // half-created goal behind.
        let fire_policy = match args.fire_policy.as_deref() {
            Some("always_fire") => "always_fire".to_string(),
            Some("coalesce") | None => "coalesce".to_string(),
            Some(other) => {
                return Ok(Answer(format!(
                    "Invalid fire_policy '{}'. Use 'coalesce' or 'always_fire'.",
                    other
                )));
            }
        };
        let reminder = args
            .reminder
            .as_deref()
            .map(str::trim)
            .filter(|message| !message.is_empty())
            .map(str::to_string);

        let next_run = parsed
            .iter()
            .map(|p| p.next_local)
            .min_by_key(|dt| dt.timestamp())
            .map(|dt| aidaemon_work::schedule::owner_time(&dt))
            .unwrap_or_else(|| "n/a".to_string());
        let subject = json!({
            "action": "create_scheduled_goal",
            "goal": desc,
            "schedules": parsed.iter().map(|p| p.cron.clone()).collect::<Vec<_>>(),
            "one_shot": parsed.iter().map(|p| p.is_one_shot).collect::<Vec<_>>(),
            "fire_policy": fire_policy,
            "is_paused": args.is_paused.unwrap_or(false),
            "name": name,
        })
        .to_string();
        Ok(ScheduledGoalPrep::Plan(ScheduledGoalPlan {
            desc,
            session_id: session_id.to_string(),
            parsed,
            fire_policy,
            reminder,
            name,
            next_run,
            subject,
        }))
    }

    /// The owner approval one call needs: a scheduled goal that runs the
    /// agent is confirmed by the owner before it is created, whatever the
    /// session; a reminder and every call the tool answers without creating
    /// a goal need none.
    pub async fn owner_approval(
        &self,
        arguments: &Value,
        facts: &ToolCallContext,
    ) -> crate::tools::traits::OwnerApprovalNeed {
        use crate::tools::traits::OwnerApprovalNeed;
        let Ok(args) = serde_json::from_value::<ManageArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if args.action != "create_scheduled_goal" {
            return OwnerApprovalNeed::NotNeeded;
        }
        match self.prepare_scheduled_goal(&args, facts).await {
            Ok(ScheduledGoalPrep::Plan(plan)) if plan.needs_owner_approval() => {
                OwnerApprovalNeed::Ask(plan.approval_request())
            }
            _ => OwnerApprovalNeed::NotNeeded,
        }
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

        // Schedule operations act on scheduled goals.
        let goals = self.state.get_scheduled_goals().await?;
        aidaemon_work::tools::goal_ids::pick_goal(trimmed, &goals, "Scheduled goal")
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

    pub(crate) fn is_protected_system_maintenance_goal(goal: &crate::tools::traits::Goal) -> bool {
        const KNOWLEDGE_GOAL_DESC: &str =
            "Maintain knowledge base: process embeddings, consolidate memories, decay old facts";
        const HEALTH_GOAL_DESC: &str =
            "Maintain memory health: prune old events, clean up retention, remove stale data";

        if goal.description == KNOWLEDGE_GOAL_DESC || goal.description == HEALTH_GOAL_DESC {
            return true;
        }

        if let Some(ctx) = &goal.context
            && let Ok(v) = serde_json::from_str::<Value>(ctx)
            && v.get("system_protected").and_then(|x| x.as_bool()) == Some(true)
        {
            return true;
        }

        false
    }

    fn goal_matches_query(goal: &crate::tools::traits::Goal, query: &str) -> bool {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return false;
        }
        goal.id.to_ascii_lowercase().starts_with(&q)
            || goal.description.to_ascii_lowercase().contains(&q)
    }

    fn canonicalize_schedule_goal_description(input: &str) -> String {
        let mut normalized = input.trim().to_ascii_lowercase();
        let system_suffix = "[system: already scheduled and firing now; do not reschedule.]";
        if let Some(idx) = normalized.find(system_suffix) {
            normalized.truncate(idx);
        }
        normalized = normalized.trim().to_string();

        for prefix in ["execute scheduled goal:", "scheduled check:"] {
            if let Some(rest) = normalized.strip_prefix(prefix) {
                normalized = rest.trim().to_string();
                break;
            }
        }

        normalized.split_whitespace().collect::<Vec<_>>().join(" ")
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
            .map(|dt| aidaemon_work::schedule::owner_time(&dt))
            .unwrap_or_else(|| ts.to_string())
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
    schedule_id: Option<String>,
    schedule: Option<String>,
    /// A plain reminder's message, delivered as-is when the schedule fires.
    /// Set by `set_reminder`; not offered in this tool's schema.
    reminder: Option<String>,
    schedules: Option<Vec<String>>,
    fire_policy: Option<String>,
    is_one_shot: Option<bool>,
    is_paused: Option<bool>,
    /// create_scheduled_goal: the short name the owner gave it.
    name: Option<String>,
}

/// The longest name an automation may carry.
pub(crate) const MAX_AUTOMATION_NAME_CHARS: usize = 80;

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
                    "enum": ["list", "health", "repair", "forget", "set_privacy", "search", "search_episodes", "create_personal_goal", "list_goals", "complete_goal", "abandon_goal", "create_scheduled_goal", "activate_scheduled", "list_scheduled", "list_scheduled_matching", "add_schedule", "cancel_scheduled", "pause_scheduled", "resume_scheduled", "retry_scheduled", "retry_failed_scheduled", "cancel_scheduled_matching", "retry_scheduled_matching", "diagnose_scheduled"]
                },
                "limit": { "type": "integer" },
                "category": { "type": "string" },
                "key": { "type": "string" },
                "privacy": { "type": "string", "enum": ["global", "channel", "private"] },
                "query": { "type": "string" },
                "goal": { "type": "string" },
                "priority": { "type": "string", "enum": ["low", "medium", "high", "critical"] },
                "goal_id": { "type": "string" },
                "schedule_id": { "type": "string" },
                "schedule": { "type": "string" },
                "schedules": { "type": "array", "items": { "type": "string" } },
                "fire_policy": { "type": "string", "enum": ["coalesce", "always_fire"] },
                "is_one_shot": { "type": "boolean" },
                "is_paused": { "type": "boolean" },
                "name": { "type": "string" }
            },
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

/// Typed effects for memory-store mutations. Memory is persistent agent
/// state, so writes are `configuration` and a forget is also `destructive`;
/// an untyped receipt could never close a checklist step the model declared
/// with either label. Scheduled-goal actions keep their existing typing.
fn memory_action_effects(arguments: &str) -> ToolMutationEffects {
    let action = serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("action")
                .and_then(Value::as_str)
                .map(str::to_string)
        });
    match action.as_deref() {
        Some("forget") => {
            ToolMutationEffects::CONFIGURATION.union(ToolMutationEffects::DESTRUCTIVE)
        }
        Some("set_privacy") => ToolMutationEffects::CONFIGURATION,
        Some(
            "create_personal_goal"
            | "complete_goal"
            | "abandon_goal"
            | "create_scheduled_goal"
            | "activate_scheduled"
            | "add_schedule"
            | "cancel_scheduled"
            | "pause_scheduled"
            | "resume_scheduled"
            | "retry_scheduled"
            | "retry_failed_scheduled"
            | "cancel_scheduled_matching"
            | "retry_scheduled_matching"
            | "trigger_now",
        ) => ToolMutationEffects::SCHEDULED_WORK,
        _ => ToolMutationEffects::NONE,
    }
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

    /// A scheduled goal is confirmed on a goal card (Approve & activate /
    /// Cancel), which offers no session or persistent grant.
    fn approval_card_kind(
        &self,
        _request: &aidaemon_core::OwnerApprovalRequest,
    ) -> crate::tools::traits::ApprovalKind {
        crate::tools::traits::ApprovalKind::GoalConfirmation
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        let exec_ctx = facts.clone();

        self.outcome(arguments, &exec_ctx).await
    }
}

#[async_trait]
impl crate::tools::traits::ToolSemantics for ManageMemoriesTool {
    /// A scheduled goal that runs the agent is confirmed by the owner
    /// before it is created. The dispatcher runs the call only with that
    /// answer; a decline is recorded by the kernel and creates nothing.
    async fn owner_approval_request(
        &self,
        call: &crate::tools::traits::ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> crate::tools::traits::OwnerApprovalNeed {
        if call.refused {
            return crate::tools::traits::OwnerApprovalNeed::NotNeeded;
        }
        self.owner_approval(call.arguments, facts).await
    }

    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            always_offered: true,
            status_from_arguments: true,
            ..Default::default()
        }
    }

    fn validate_arguments(&self, arguments: &str) -> Result<(), ToolArgumentContractViolation> {
        const ACTIONS_REQUIRING_GOAL_ID: &[&str] = &[
            "add_schedule",
            "cancel_scheduled",
            "pause_scheduled",
            "resume_scheduled",
            "retry_scheduled",
            "diagnose_scheduled",
        ];
        let parsed = serde_json::from_str::<Value>(arguments).ok();
        let action = parsed
            .as_ref()
            .and_then(|value| value.get("action"))
            .and_then(Value::as_str);
        if !action.is_some_and(|action| ACTIONS_REQUIRING_GOAL_ID.contains(&action)) {
            return Ok(());
        }
        let has_goal_id = parsed
            .as_ref()
            .and_then(|value| value.get("goal_id"))
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());
        if has_goal_id {
            return Ok(());
        }
        Err(ToolArgumentContractViolation::new(format!(
            "action `{}` requires `goal_id` for `manage_memories`",
            action.unwrap_or("<missing>")
        ))
        .with_recovery_hint("Obtain a concrete scheduled goal identifier before retrying."))
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(
            arguments,
            &[
                "list",
                "health",
                "search",
                "search_episodes",
                "list_goals",
                "list_scheduled",
                "list_scheduled_matching",
                "diagnose_scheduled",
            ],
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

    /// Reads, `repair` and `forget` touch only the agent's own memory store.
    /// `set_privacy` widens who may see a fact, and the goal and schedule
    /// actions start or schedule work, so those stay undeclared. A declined
    /// `create_scheduled_goal` (no typed target to name: the work it
    /// schedules is open-ended) is recorded by its approval subject, so the
    /// identical goal is not put to the owner again.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        match action_of(arguments).as_deref() {
            Some(
                "list"
                | "health"
                | "search"
                | "search_episodes"
                | "list_goals"
                | "list_scheduled"
                | "list_scheduled_matching"
                | "diagnose_scheduled"
                | "repair"
                | "forget",
            ) => aidaemon_core::CallEffects::none(),
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
            "create_scheduled_goal" => {
                let plan = match self.prepare_scheduled_goal(&args, facts).await? {
                    ScheduledGoalPrep::Answer(answer) => return Ok(answer),
                    ScheduledGoalPrep::Plan(plan) => plan,
                };
                // The owner confirmed this goal through the agent kernel
                // (`owner_approval_request`) before the call ran. Outside
                // the kernel nothing is created.
                if plan.needs_owner_approval()
                    && facts.kernel_approved != Some(crate::tools::traits::ApprovalScope::Call)
                {
                    anyhow::bail!(
                        "manage_memories create_scheduled_goal requires approval through the \
                         agent kernel; it was called outside an approved agent call"
                    );
                }
                let ScheduledGoalPlan {
                    desc,
                    session_id,
                    parsed,
                    fire_policy,
                    reminder,
                    name,
                    next_run,
                    subject: _,
                } = plan;

                let has_recurring = parsed.iter().any(|p| !p.is_one_shot);
                let mut goal = if has_recurring {
                    crate::tools::traits::Goal::new_continuous_pending(
                        &desc,
                        &session_id,
                        None,
                        None,
                    )
                } else {
                    crate::tools::traits::Goal::new_deferred_finite(&desc, &session_id)
                };
                goal.domain = "orchestration".to_string();
                if let Some(message) = reminder.as_deref() {
                    goal.context = Some(aidaemon_work::reminders::context_for_message(message));
                }
                if let Some(name) = name.as_deref() {
                    goal.context = Some(aidaemon_work::automations::context_with_name(
                        goal.context.as_deref(),
                        name,
                    ));
                }

                self.state.create_goal(&goal).await?;

                let paused = args.is_paused.unwrap_or(false);
                let now = chrono::Utc::now().to_rfc3339();

                for p in &parsed {
                    let schedule = crate::tools::traits::GoalSchedule {
                        id: uuid::Uuid::new_v4().to_string(),
                        goal_id: goal.id.clone(),
                        cron_expr: p.cron.clone(),
                        tz: "local".to_string(),
                        original_schedule: Some(p.original.clone()),
                        fire_policy: fire_policy.clone(),
                        is_one_shot: p.is_one_shot,
                        is_paused: paused,
                        last_run_at: None,
                        next_run_at: p.next_local.with_timezone(&chrono::Utc).to_rfc3339(),
                        created_at: now.clone(),
                        updated_at: now.clone(),
                    };
                    self.state.create_goal_schedule(&schedule).await?;
                }

                self.state.activate_goal(&goal.id).await?;
                // A reminder runs no agent and only messages the owner: no
                // confirmation step.
                if let Some(message) = reminder {
                    return Ok(format!(
                        "Reminder set: \"{}\". {} schedule(s). Next: {}.",
                        message,
                        parsed.len(),
                        next_run
                    ));
                }
                Ok(format!(
                    "Confirmed and activated scheduled goal {}. {} schedule(s). Next: {}.",
                    goal.id,
                    parsed.len(),
                    next_run
                ))
            }
            "activate_scheduled" => {
                let is_owner = facts.user_role == Some(UserRole::Owner);
                if !is_owner {
                    return Ok("Only the owner can activate scheduled goals.".to_string());
                }
                let session_id = facts.session().unwrap_or("").trim();
                if session_id.is_empty() {
                    return Ok(
                        "Internal error: activate_scheduled: the session fact is missing from the call context.".to_string(),
                    );
                }
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for activate_scheduled action")
                })?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(error) => return Ok(error.to_string()),
                };
                let Some(goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if goal.session_id != session_id {
                    return Ok(format!(
                        "Cannot activate scheduled goal {} from a different owner session.",
                        resolved_goal_id
                    ));
                }
                if goal.status != "pending_confirmation" {
                    return Ok(format!(
                        "Scheduled goal {} is not awaiting confirmation (current status: {}).",
                        resolved_goal_id, goal.status
                    ));
                }
                if !self.state.activate_goal(&resolved_goal_id).await? {
                    return Ok(format!(
                        "Scheduled goal {} was not activated because its state changed.",
                        resolved_goal_id
                    ));
                }
                let next_run = self
                    .state
                    .get_schedules_for_goal(&resolved_goal_id)
                    .await?
                    .iter()
                    .filter_map(|schedule| {
                        chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at).ok()
                    })
                    .min_by_key(|at| at.timestamp())
                    .map(|at| {
                        aidaemon_work::schedule::humanize_run_time(at.with_timezone(&chrono::Local))
                    })
                    .unwrap_or_else(|| "unscheduled".to_string());
                Ok(format!(
                    "Activated scheduled goal {} (next run {}).",
                    resolved_goal_id, next_run
                ))
            }
            "list_scheduled" => {
                let all_goals = self.state.get_scheduled_goals().await?;
                // Active goals left with no schedule (and no mandate) never run;
                // show them so they can be cancelled by goal_id.
                let unscheduled: Vec<crate::tools::traits::Goal> = self
                    .state
                    .get_unscheduled_active_goals()
                    .await
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|goal| !Self::is_protected_system_maintenance_goal(goal))
                    .collect();
                let mut unscheduled_section = String::new();
                if !unscheduled.is_empty() {
                    unscheduled_section.push_str(&format!(
                        "**Active without a schedule** ({}) — these never run; cancel with cancel_scheduled and the goal_id\n",
                        unscheduled.len()
                    ));
                    for g in &unscheduled {
                        let desc: String = g.description.chars().take(80).collect();
                        unscheduled_section.push_str(&format!("- **{}** {}\n", g.id, desc));
                    }
                    unscheduled_section.push('\n');
                }
                if all_goals.is_empty() {
                    return Ok(if unscheduled_section.is_empty() {
                        "No scheduled goals.".to_string()
                    } else {
                        format!("No scheduled goals.\n\n{unscheduled_section}")
                    });
                }
                let limit = args.limit.unwrap_or(50).clamp(1, 200);
                let mut goals = all_goals;
                let total_goals = goals.len();
                goals.truncate(limit);

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
                let mut output = format!(
                    "**Scheduled Goals** (showing {} of {} total)\n\n",
                    goals.len(),
                    total_goals
                );
                if active_count == 0 {
                    output.push_str("No active scheduled tasks.\n\n");
                }

                output.push_str(&format!(
                    "System timezone: {}.\nTip: use `schedule_id` for schedule-specific pause/resume/cancel.\n\n",
                    aidaemon_work::schedule::system_timezone_display()
                ));

                let groups: [(&str, &Vec<&crate::tools::traits::Goal>); 7] = [
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
                output.push_str(&unscheduled_section);
                if total_goals > goals.len() {
                    output.push_str(&format!(
                        "(Results truncated to {} goals. Provide a higher `limit` to see more.)\n",
                        goals.len()
                    ));
                }
                Ok(output)
            }
            "list_scheduled_matching" => {
                let query = args
                    .query
                    .as_deref()
                    .map(str::trim)
                    .filter(|q| !q.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!("'query' is required for list_scheduled_matching action")
                    })?;
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::tools::traits::Goal> = goals
                    .iter()
                    .filter(|g| Self::goal_matches_query(g, query))
                    .collect();
                if matched.is_empty() {
                    return Ok(format!("No scheduled goals matched query '{}'.", query));
                }
                matched.sort_by(|a, b| b.created_at.cmp(&a.created_at));
                let limit = args.limit.unwrap_or(50).clamp(1, 200);

                let mut output = format!(
                    "**Matching Scheduled Goals** for '{}' (showing {} of {} matches)\n\n",
                    query,
                    matched.len().min(limit),
                    matched.len()
                );
                for g in matched.into_iter().take(limit) {
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
                        .map(|dt| aidaemon_work::schedule::owner_time(&dt))
                        .unwrap_or_else(|| "n/a".to_string());
                    output.push_str(&format!(
                        "- **{}** {} (type: {}, status: {}, schedules: {}, next: {})\n",
                        g.id,
                        desc,
                        goal_type,
                        g.status,
                        schedules.len(),
                        next_run
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
                    .ok_or_else(|| {
                        anyhow::anyhow!("'schedule' is required for add_schedule action")
                    })?;
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

                let cron_expr = match aidaemon_work::schedule::parse_schedule(schedule_raw) {
                    Ok(expr) => expr,
                    Err(e) => {
                        return Ok(format!("Couldn't parse schedule '{}': {}", schedule_raw, e));
                    }
                };
                let next_local = match aidaemon_work::schedule::compute_next_run_local(&cron_expr) {
                    Ok(dt) => dt,
                    Err(e) => {
                        return Ok(format!(
                            "Couldn't compute next run for '{}': {}",
                            cron_expr, e
                        ));
                    }
                };

                let now = chrono::Utc::now().to_rfc3339();
                let schedule = crate::tools::traits::GoalSchedule {
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
                            ));
                        }
                    },
                    is_one_shot: args.is_one_shot.unwrap_or_else(|| {
                        aidaemon_work::schedule::is_one_shot_schedule(&cron_expr)
                    }),
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
                    aidaemon_work::schedule::owner_time(&next_local)
                ))
            }
            "cancel_scheduled" => {
                let goal_id_input = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for cancel_scheduled action")
                })?;
                let goal_id_trimmed = goal_id_input.trim();

                if goal_id_trimmed.eq_ignore_ascii_case("all") || goal_id_trimmed == "*" {
                    let session_id = facts.session().unwrap_or("").trim();
                    if session_id.is_empty() {
                        return Ok(
                            "Internal error: cancel_scheduled with goal_id='all': the session fact is missing from the call context."
                                .to_string(),
                        );
                    }
                    let goals = self.state.get_scheduled_goals().await?;
                    let mut cancelled = 0usize;
                    let mut protected = 0usize;
                    let mut skipped = 0usize;
                    let mut errors = 0usize;

                    for mut goal in goals {
                        if goal.session_id != session_id {
                            continue;
                        }
                        if !matches!(
                            goal.status.as_str(),
                            "active" | "pending_confirmation" | "paused"
                        ) {
                            skipped += 1;
                            continue;
                        }
                        if Self::is_protected_system_maintenance_goal(&goal) {
                            protected += 1;
                            continue;
                        }

                        let schedules = self
                            .state
                            .get_schedules_for_goal(&goal.id)
                            .await
                            .unwrap_or_default();
                        goal.status = "cancelled".to_string();
                        goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        goal.updated_at = chrono::Utc::now().to_rfc3339();
                        if self.state.update_goal(&goal).await.is_ok() {
                            for s in &schedules {
                                let _ = self.state.delete_goal_schedule(&s.id).await;
                            }
                            cancelled += 1;
                        } else {
                            errors += 1;
                        }
                    }

                    let mut msg = format!("Cancelled {} scheduled goals.", cancelled);
                    if protected > 0 || skipped > 0 || errors > 0 {
                        msg.push_str(&format!(
                            " Skipped protected: {}. Skipped non-active: {}. Errors: {}.",
                            protected, skipped, errors
                        ));
                    }
                    return Ok(msg);
                }

                let resolved_goal_id = match self.resolve_goal_id(goal_id_trimmed).await {
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
                    if !deleted {
                        return Ok(format!("Schedule {} was already removed.", sched.id));
                    }
                    // A goal with no schedule left would stay "active" in
                    // listings and never run: its last schedule ends it.
                    let remaining = self.state.get_schedules_for_goal(&goal.id).await?;
                    if remaining.is_empty() && !Self::is_protected_system_maintenance_goal(&goal) {
                        goal.status = "cancelled".to_string();
                        goal.completed_at = Some(chrono::Utc::now().to_rfc3339());
                        goal.updated_at = chrono::Utc::now().to_rfc3339();
                        self.state.update_goal(&goal).await?;
                        return Ok(format!(
                            "Cancelled schedule {}; it was the last schedule of goal {}, so the goal is cancelled too.",
                            sched.id, goal.id
                        ));
                    }
                    return Ok(format!(
                        "Cancelled schedule {} for goal {}.",
                        sched.id, goal.id
                    ));
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
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for pause_scheduled action")
                })?;
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
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for resume_scheduled action")
                })?;
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
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for retry_scheduled action")
                })?;
                let resolved_goal_id = match self.resolve_goal_id(goal_id).await {
                    Ok(id) => id,
                    Err(e) => return Ok(e.to_string()),
                };
                let Some(mut goal) = self.state.get_goal(&resolved_goal_id).await? else {
                    return Ok(format!("Scheduled goal not found: {}", resolved_goal_id));
                };
                if !matches!(goal.status.as_str(), "failed" | "stalled") {
                    return Ok(format!(
                        "Only failed or stalled scheduled goals can be retried (current status: {}).",
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
                goal.dispatch_failures = 0;
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
                let mut matched: Vec<&crate::tools::traits::Goal> = goals
                    .iter()
                    .filter(|g| {
                        matches!(g.status.as_str(), "failed" | "stalled")
                            && (query.is_empty() || Self::goal_matches_query(g, query))
                    })
                    .collect();
                if matched.is_empty() {
                    if query.is_empty() {
                        return Ok("No failed or stalled scheduled goals to retry.".to_string());
                    }
                    return Ok(format!(
                        "No failed or stalled scheduled goals matched query '{}'.",
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
                    updated.dispatch_failures = 0;
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
                    .ok_or_else(|| {
                        anyhow::anyhow!("'query' is required for cancel_scheduled_matching action")
                    })?;
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::tools::traits::Goal> = goals
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
                    out.push_str(&format!(
                        "\nProtected (not cancelled):\n- {}",
                        protected.join("\n- ")
                    ));
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
                    .ok_or_else(|| {
                        anyhow::anyhow!("'query' is required for retry_scheduled_matching action")
                    })?;
                let goals = self.state.get_scheduled_goals().await?;
                let mut matched: Vec<&crate::tools::traits::Goal> = goals
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
                    if !matches!(g.status.as_str(), "failed" | "stalled") {
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
                    updated.dispatch_failures = 0;
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
                    "retry_scheduled_matching('{}'): retried {}, not-retryable {}, cap-blocked {}, errors {}.",
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
                    out.push_str(&format!(
                        "\nNot failed/stalled (unchanged):\n- {}",
                        non_failed.join("\n- ")
                    ));
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
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for diagnose_scheduled action")
                })?;
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
                    .map(|dt| aidaemon_work::schedule::owner_time(&dt))
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

                if let Some(last_failed_task) = tasks
                    .iter()
                    .filter(|t| t.status == "failed")
                    .max_by(|a, b| {
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
                    })
                {
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
                            let ok = a
                                .success
                                .map(|v| if v { "ok" } else { "err" })
                                .unwrap_or("n/a");
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
            "trigger_now" => {
                let goal_id = args.goal_id.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'goal_id' is required for trigger_now action")
                })?;
                // Keep this compatibility action on the same implementation as
                // `scheduled_goal_runs(run_now)`. The former duplicate path did
                // not copy goal context or establish a fresh rooted goal run,
                // which could silently drop updated instructions and attach a
                // manual task to stale ownership state.
                aidaemon_work::tools::scheduled_goal_runs::ScheduledGoalRunsTool::new(
                    self.state.clone(),
                )
                .run_now(goal_id, None)
                .await
            }
            other => Ok(format!(
                "Unknown action: '{}'. Use list, forget, set_privacy, search, create_personal_goal, list_goals, complete_goal, abandon_goal, create_scheduled_goal, activate_scheduled, list_scheduled, list_scheduled_matching, add_schedule, cancel_scheduled, pause_scheduled, resume_scheduled, retry_scheduled, retry_failed_scheduled, cancel_scheduled_matching, retry_scheduled_matching, diagnose_scheduled, or trigger_now.",
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
        let output = self.run(arguments, facts).await?;
        let trimmed = output.trim_start();
        let outcome_status = if trimmed.starts_with("Internal error:")
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
    fn a_goal_action_reports_what_it_did_without_counting_as_outbound() {
        use crate::tools::traits::ToolMutationEffects as Effects;
        let effects = memory_action_effects(r#"{"action":"create_scheduled_goal"}"#);
        // A plan may name it any of these; each must be able to close.
        for planned in [
            Effects::SCHEDULED_WORK,
            Effects::CONFIGURATION,
            Effects::PROCESS_STATE,
        ] {
            assert!(effects.satisfies(planned));
        }
        assert!(!effects.satisfies(Effects::REMOTE_MUTATION));
        assert!(!effects.intersects(crate::tools::traits::EGRESS_EFFECTS));
    }

    #[test]
    fn canonicalize_schedule_goal_description_strips_execution_wrappers() {
        let wrapped = "Scheduled check: Check API health [SYSTEM: already scheduled and firing now; do not reschedule.]";
        let execute_wrapped = "Execute scheduled goal:   Check   API health  ";
        assert_eq!(
            ManageMemoriesTool::canonicalize_schedule_goal_description(wrapped),
            "check api health"
        );
        assert_eq!(
            ManageMemoriesTool::canonicalize_schedule_goal_description(execute_wrapped),
            "check api health"
        );
    }
}
