//! `schedule_task`: scheduled work in three typed fields, and the one path
//! that creates scheduled work.
//!
//! [`ScheduledWork::create`] is shared with `set_reminder` (a reminder is
//! scheduled work that only messages the owner). It answers with a typed
//! [`ScheduleOutcome`], so a refusal (duplicate, owner-only, a time it cannot
//! read, a name in use) is never read as a success from its wording.
//! Changing or stopping scheduled work is `manage_automations`.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::traits::{
    AutomationStore, Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus,
};
use crate::tools::traits::{ChannelVisibility, UserRole};

/// The longest name an automation may carry.
pub(crate) const MAX_AUTOMATION_NAME_CHARS: usize = 80;

/// One request for scheduled work.
#[derive(Debug, Clone)]
pub(crate) struct ScheduleRequest {
    /// The instruction each run carries out (for a reminder, its message).
    pub task: String,
    /// When it runs, as the owner said it.
    pub when: String,
    /// The short name the owner gave it.
    pub name: Option<String>,
    /// A plain reminder's message: delivered as-is, no agent runs.
    pub reminder: Option<String>,
}

/// What a creation call did.
#[derive(Debug, Clone)]
pub(crate) enum ScheduleOutcome {
    /// The work is stored and active.
    Created(Created),
    /// Nothing was stored, for this reason.
    Refused(String),
}

#[derive(Debug, Clone)]
pub(crate) struct Created {
    pub goal_id: String,
    pub reminder: Option<String>,
    pub name: Option<String>,
    pub task: String,
    pub next_run: String,
}

impl Created {
    /// The answer the model reads.
    pub fn message(&self) -> String {
        match (&self.reminder, &self.name) {
            (Some(message), _) => {
                format!("Reminder set: \"{message}\". Next: {}.", self.next_run)
            }
            (None, Some(name)) => format!(
                "Scheduled \"{name}\": {}. Next run: {}.",
                self.task, self.next_run
            ),
            (None, None) => format!("Scheduled: {}. Next run: {}.", self.task, self.next_run),
        }
    }
}

impl ScheduleOutcome {
    /// The model's answer and its typed status.
    pub fn into_tool_outcome(self) -> ToolCallOutcome {
        let (output, status, internal_identifiers) = match self {
            Self::Created(created) => (
                created.message(),
                ToolOutcomeStatus::Succeeded,
                vec![created.goal_id],
            ),
            Self::Refused(reason) => (reason, ToolOutcomeStatus::FailedPermanent, Vec::new()),
        };
        ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(status),
                internal_identifiers,
                ..ToolCallMetadata::default()
            },
            finish: None,
        }
    }
}

/// A validated request, before anything is stored.
struct Plan {
    desc: String,
    session_id: String,
    cron: String,
    original: String,
    is_one_shot: bool,
    next_local: chrono::DateTime<chrono::Local>,
    reminder: Option<String>,
    name: Option<String>,
    next_run: String,
    /// The exact work the owner approves (the approval subject).
    subject: String,
}

impl Plan {
    /// The owner confirms scheduled work that runs the agent; a reminder
    /// runs no agent and only messages the owner, so it needs none.
    fn needs_owner_approval(&self) -> bool {
        self.reminder.is_none()
    }

    /// The card the owner confirms the work with: the five lines every
    /// automation card has, then the next run and the name as details.
    fn approval_request(&self) -> aidaemon_core::OwnerApprovalRequest {
        let spec = aidaemon_work::automations::ScheduleSpec {
            cron: self.cron.clone(),
            one_shot: self.is_one_shot,
            next: Some(aidaemon_work::schedule::owner_time(&self.next_local)),
        };
        let mut card = aidaemon_work::automations::scheduled_card(
            &self.desc,
            self.reminder.as_deref(),
            &[spec],
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
}

/// The description two creations compare by: wrappers a run adds and
/// spacing removed.
pub(crate) fn canonical_description(input: &str) -> String {
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

/// Why `name` cannot name an automation, if it cannot.
pub(crate) async fn name_refusal(
    state: &dyn AutomationStore,
    session_id: &str,
    name: &str,
) -> anyhow::Result<Option<String>> {
    if name.chars().count() > MAX_AUTOMATION_NAME_CHARS || name.chars().any(char::is_control) {
        return Ok(Some(format!(
            "The name must be one line of at most {MAX_AUTOMATION_NAME_CHARS} characters."
        )));
    }
    Ok(
        crate::tools::manage_automations::live_name_holder(state, session_id, name)
            .await?
            .map(|taken| {
                format!(
                    "An automation is already named \"{name}\" ({taken}); pick another name or cancel that one first."
                )
            }),
    )
}

/// The one path that creates scheduled work.
pub(crate) struct ScheduledWork {
    state: Arc<dyn AutomationStore>,
}

impl ScheduledWork {
    pub fn new(state: Arc<dyn AutomationStore>) -> Self {
        Self { state }
    }

    /// Validate a request with the dispatcher's facts and the stored work,
    /// exactly as creation does before it stores anything. Shared by the
    /// call and its approval judgment, so the owner is asked only about work
    /// the call would create.
    async fn prepare(
        &self,
        request: &ScheduleRequest,
        facts: &ToolCallContext,
    ) -> anyhow::Result<Result<Plan, String>> {
        if facts.user_role != Some(UserRole::Owner) {
            return Ok(Err("Only the owner can schedule work.".to_string()));
        }
        let session_id = facts.session().unwrap_or("").trim();
        if session_id.is_empty() {
            return Ok(Err(
                "Internal error: the session fact is missing from the call context.".to_string(),
            ));
        }
        // `sub-` is the legacy child-session prefix (replaced by
        // `specialist:`); accept both so an in-flight legacy session can't
        // bypass this guard.
        if facts.channel_visibility == Some(ChannelVisibility::Internal)
            || session_id.starts_with("sub-")
            || session_id.starts_with("specialist:")
        {
            return Ok(Err("Cannot schedule work from within internal scheduled-task execution. Execute the task directly instead.".to_string()));
        }
        let raw_desc = request.task.trim();
        let when = request.when.trim();
        if raw_desc.is_empty() {
            return Ok(Err("Say what to do: `task` is empty.".to_string()));
        }
        if when.is_empty() {
            return Ok(Err("Say when: `when` is empty.".to_string()));
        }

        // Remove the schedule phrase and leading filler from the task.
        let mut desc = aidaemon_work::schedule::clean_task_description(raw_desc, when);
        desc = aidaemon_work::schedule::clean_task_description(&desc, "");
        if desc.trim().is_empty() {
            desc = raw_desc.to_string();
        }

        let cron = match aidaemon_work::schedule::parse_schedule(when) {
            Ok(cron) => cron,
            Err(error) => return Ok(Err(format!("Couldn't read the time '{when}': {error}"))),
        };
        let next_local = match aidaemon_work::schedule::compute_next_run_local(&cron) {
            Ok(at) => at,
            Err(error) => {
                return Ok(Err(format!(
                    "Couldn't compute the next run for '{when}': {error}"
                )));
            }
        };
        let is_one_shot = aidaemon_work::schedule::is_one_shot_schedule(&cron);

        // The model repeating the same request must not create it twice.
        let target = canonical_description(&desc);
        let target_crons = BTreeSet::from([cron.trim().to_ascii_lowercase()]);
        for existing in self.state.get_scheduled_goals().await.unwrap_or_default() {
            if existing.session_id != session_id
                || !matches!(
                    existing.status.as_str(),
                    "active" | "pending_confirmation" | "paused"
                )
                || canonical_description(&existing.description) != target
            {
                continue;
            }
            let existing_crons: BTreeSet<String> = self
                .state
                .get_schedules_for_goal(&existing.id)
                .await
                .unwrap_or_default()
                .iter()
                .map(|schedule| schedule.cron_expr.trim().to_ascii_lowercase())
                .collect();
            if existing_crons == target_crons {
                return Ok(Err(format!(
                    "The same scheduled work already exists [ref {}]; see it with manage_automations list.",
                    existing.id.get(..8).unwrap_or(&existing.id)
                )));
            }
        }

        let name = request
            .name
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        if let Some(name) = name.as_deref()
            && let Some(refusal) = name_refusal(self.state.as_ref(), session_id, name).await?
        {
            return Ok(Err(refusal));
        }
        let reminder = request
            .reminder
            .as_deref()
            .map(str::trim)
            .filter(|message| !message.is_empty())
            .map(str::to_string);

        let next_run = aidaemon_work::schedule::owner_time(&next_local);
        let subject = json!({
            "action": "schedule_task",
            "task": desc,
            "schedule": cron,
            "one_shot": is_one_shot,
            "name": name,
        })
        .to_string();
        Ok(Ok(Plan {
            desc,
            session_id: session_id.to_string(),
            cron,
            original: when.to_string(),
            is_one_shot,
            next_local,
            reminder,
            name,
            next_run,
            subject,
        }))
    }

    /// The owner approval one request needs: work that runs the agent is
    /// confirmed by the owner before it is created; a reminder and every
    /// request that would be refused need none.
    pub async fn owner_approval(
        &self,
        request: &ScheduleRequest,
        facts: &ToolCallContext,
    ) -> crate::tools::traits::OwnerApprovalNeed {
        use crate::tools::traits::OwnerApprovalNeed;
        match self.prepare(request, facts).await {
            Ok(Ok(plan)) if plan.needs_owner_approval() => {
                OwnerApprovalNeed::Ask(plan.approval_request())
            }
            _ => OwnerApprovalNeed::NotNeeded,
        }
    }

    /// Store the work and activate it, or say why not.
    pub async fn create(
        &self,
        request: &ScheduleRequest,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ScheduleOutcome> {
        let plan = match self.prepare(request, facts).await? {
            Ok(plan) => plan,
            Err(reason) => return Ok(ScheduleOutcome::Refused(reason)),
        };
        // The owner confirmed this work through the agent kernel
        // (`owner_approval_request`) before the call ran. Outside the
        // kernel nothing is created.
        if plan.needs_owner_approval()
            && facts.kernel_approved != Some(crate::tools::traits::ApprovalScope::Call)
        {
            anyhow::bail!(
                "schedule_task requires approval through the agent kernel; it was called \
                 outside an approved agent call"
            );
        }
        let mut goal = if plan.is_one_shot {
            crate::tools::traits::Goal::new_deferred_finite(&plan.desc, &plan.session_id)
        } else {
            crate::tools::traits::Goal::new_continuous_pending(
                &plan.desc,
                &plan.session_id,
                None,
                None,
            )
        };
        goal.domain = "orchestration".to_string();
        if let Some(message) = plan.reminder.as_deref() {
            goal.context = Some(aidaemon_work::reminders::context_for_message(message));
        }
        if let Some(name) = plan.name.as_deref() {
            goal.context = Some(aidaemon_work::automations::context_with_name(
                goal.context.as_deref(),
                name,
            ));
        }
        self.state.create_goal(&goal).await?;

        let now = chrono::Utc::now().to_rfc3339();
        self.state
            .create_goal_schedule(&crate::tools::traits::GoalSchedule {
                id: uuid::Uuid::new_v4().to_string(),
                goal_id: goal.id.clone(),
                cron_expr: plan.cron.clone(),
                tz: "local".to_string(),
                original_schedule: Some(plan.original.clone()),
                fire_policy: "coalesce".to_string(),
                is_one_shot: plan.is_one_shot,
                is_paused: false,
                last_run_at: None,
                next_run_at: plan.next_local.with_timezone(&chrono::Utc).to_rfc3339(),
                created_at: now.clone(),
                updated_at: now,
            })
            .await?;
        self.state.activate_goal(&goal.id).await?;
        Ok(ScheduleOutcome::Created(Created {
            goal_id: goal.id,
            reminder: plan.reminder,
            name: plan.name,
            task: plan.desc,
            next_run: plan.next_run,
        }))
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleTaskArgs {
    task: String,
    when: String,
    name: Option<String>,
}

impl ScheduleTaskArgs {
    fn request(self) -> ScheduleRequest {
        ScheduleRequest {
            task: self.task,
            when: self.when,
            name: self.name,
            reminder: None,
        }
    }
}

const TOOL_DESCRIPTION: &str = "Schedule work you do on your own at a time or on a cadence: each run carries out `task` with the user's access, once the user confirms. A plain reminder message is set_reminder; seeing, changing or stopping it is manage_automations.";

pub struct ScheduleTaskTool {
    work: ScheduledWork,
}

impl ScheduleTaskTool {
    pub fn new(state: Arc<dyn AutomationStore>) -> Self {
        Self {
            work: ScheduledWork::new(state),
        }
    }
}

#[async_trait]
impl Tool for ScheduleTaskTool {
    fn name(&self) -> &str {
        "schedule_task"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        json!({
            "name": "schedule_task",
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": { "type": "string", "description": "The instruction each run carries out." },
                    "when": { "type": "string", "description": "e.g. 'in 2 hours', 'tomorrow at 9am', 'every weekday at 8am'" },
                    "name": { "type": "string", "description": "The short name the user gave it, exactly as given; omit when none." }
                },
                "required": ["task", "when"],
                "additionalProperties": false
            }
        })
    }

    /// Scheduled work is confirmed on a goal card (Approve & activate /
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
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let request = serde_json::from_str::<ScheduleTaskArgs>(arguments)?.request();
        Ok(self.work.create(&request, facts).await?.into_tool_outcome())
    }
}

#[async_trait]
impl crate::tools::traits::ToolSemantics for ScheduleTaskTool {
    /// Work that runs the agent is confirmed by the owner before it is
    /// created. The dispatcher runs the call only with that answer; a
    /// decline is recorded by the kernel and creates nothing.
    async fn owner_approval_request(
        &self,
        call: &crate::tools::traits::ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> crate::tools::traits::OwnerApprovalNeed {
        if call.refused {
            return crate::tools::traits::OwnerApprovalNeed::NotNeeded;
        }
        let Ok(args) = serde_json::from_value::<ScheduleTaskArgs>(call.arguments.clone()) else {
            return crate::tools::traits::OwnerApprovalNeed::NotNeeded;
        };
        self.work.owner_approval(&args.request(), facts).await
    }

    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            offering: aidaemon_tools::semantics::Offering::Core,
            ..Default::default()
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::mutation_with(ToolMutationEffects::SCHEDULED_WORK)
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_description_strips_execution_wrappers() {
        let wrapped = "Scheduled check: Check API health [SYSTEM: already scheduled and firing now; do not reschedule.]";
        let execute_wrapped = "Execute scheduled goal:   Check   API health  ";
        assert_eq!(canonical_description(wrapped), "check api health");
        assert_eq!(canonical_description(execute_wrapped), "check api health");
    }

    #[test]
    fn scheduled_work_reports_what_it_did_without_counting_as_outbound() {
        use crate::tools::traits::ToolMutationEffects as Effects;
        let effects = Effects::SCHEDULED_WORK;
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
}
