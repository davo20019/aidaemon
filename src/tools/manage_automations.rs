//! `manage_automations`: the owner's one list of everything that runs on its
//! own, and pause/resume/cancel by name across both kinds.
//!
//! - *Scheduled*: runs a fixed instruction at set times (scheduled goals and
//!   reminders, created with `set_reminder` or `manage_memories
//!   create_scheduled_goal`).
//! - *Watching*: works toward an objective within limits the owner approved
//!   (mandates, created with `manage_mandates draft` then `create`).
//!
//! The engines and their data are unchanged: a change here runs the same
//! `manage_memories` or `manage_mandates` action with this call's dispatcher
//! facts, and the result is read back from the stored state.

use std::sync::Arc;

use aidaemon_core::mandate_card::MandateConfirmationCard;
use aidaemon_work::automations::{self, AutomationKind, ScheduleSpec};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::tools::{ManageMandatesTool, ManageMemoriesTool};
use crate::traits::{
    StateStore, Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus,
};
use crate::types::{ChannelVisibility, UserRole};

const TOOL_DESCRIPTION: &str = "The user's automations, in one list: scheduled ones (a fixed instruction or reminder at set times) and watching ones (work toward an objective within limits the user approved). List them, show one's card, or pause, resume or cancel one by its name.";

const ACTIONS: &[&str] = &["list", "show", "pause", "resume", "cancel"];

pub struct ManageAutomationsTool {
    state: Arc<dyn StateStore>,
    scheduler: ManageMemoriesTool,
    mandates: ManageMandatesTool,
}

impl ManageAutomationsTool {
    pub fn new(
        state: Arc<dyn StateStore>,
        store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    ) -> Self {
        Self {
            scheduler: ManageMemoriesTool::new(state.clone(), store),
            mandates: ManageMandatesTool::new(state.clone()),
            state,
        }
    }
}

#[derive(Debug, Deserialize)]
struct Args {
    action: String,
    name: Option<String>,
    id: Option<String>,
    kind: Option<String>,
}

/// One automation as the owner sees it.
#[derive(Debug, Clone)]
struct Automation {
    kind: AutomationKind,
    /// The goal ID (scheduled) or mandate ID (watching).
    id: String,
    name: Option<String>,
    /// The goal or mandate status.
    status: String,
    card: MandateConfirmationCard,
    next: Option<DateTime<Utc>>,
}

impl Automation {
    fn title(&self) -> String {
        match self.name.as_deref() {
            Some(name) => format!("\"{name}\""),
            None => aidaemon_events::text::truncate_str(&self.card.what, 80),
        }
    }

    fn short_id(&self) -> &str {
        self.id.get(..8).unwrap_or(&self.id)
    }

    fn next_words(&self) -> Option<String> {
        self.next.map(|at| crate::cron_utils::owner_time(&at))
    }

    /// One list row: name, kind, cadence, state, next run.
    fn row(&self) -> String {
        let mut row = format!(
            "{} · {} · {} · {}",
            self.title(),
            self.kind.label(),
            self.card.how_often,
            automations::state_words(&self.status)
        );
        if let Some(next) = self.next_words() {
            row.push_str(&format!(" · next {next}"));
        }
        row.push_str(&format!(" [ref {}]", self.short_id()));
        row
    }

    fn render_card(&self) -> String {
        let mut text = format!("{} ({})\n", self.title(), self.kind.label());
        for (essential, line) in self.card.essentials() {
            text.push_str(&format!("{}: {line}\n", essential.label()));
        }
        text.push_str(&format!(
            "State: {}",
            automations::state_words(&self.status)
        ));
        if let Some(next) = self.next_words() {
            text.push_str(&format!(" · next {next}"));
        }
        text
    }
}

/// Who already holds `name` among the owner's live automations, as the
/// owner would say it ("a scheduled one"); `None` when it is free.
pub(crate) async fn live_name_holder(
    state: &dyn StateStore,
    session_id: &str,
    name: &str,
) -> anyhow::Result<Option<&'static str>> {
    let scheduled = state.get_scheduled_goals().await?.into_iter().any(|goal| {
        matches!(
            goal.status.as_str(),
            "active" | "paused" | "pending_confirmation" | "failed"
        ) && automations::name_from_context(goal.context.as_deref())
            .is_some_and(|own| own.eq_ignore_ascii_case(name.trim()))
    });
    if scheduled {
        return Ok(Some("a scheduled one"));
    }
    let watching = state
        .list_mandates(Some(session_id), false)
        .await?
        .iter()
        .any(|mandate| mandate.has_name(name));
    Ok(watching.then_some("a watching one"))
}

/// Only the owner, in a private chat outside internal work, sees and changes
/// automations (the same bar the watching ones always had).
fn private_owner(facts: &ToolCallContext) -> bool {
    let session = facts.session_id.as_str();
    facts.user_role == Some(UserRole::Owner)
        && facts.channel_visibility == Some(ChannelVisibility::Private)
        && !session.starts_with("specialist:")
        && !session.starts_with("sub-")
}

const OWNER_ONLY: &str = "Automations can only be seen or changed by the owner in a private chat.";

impl ManageAutomationsTool {
    async fn scheduled(&self) -> anyhow::Result<Vec<Automation>> {
        let zone = crate::cron_utils::system_timezone_display();
        let mut found = Vec::new();
        for goal in self.state.get_scheduled_goals().await? {
            if !matches!(
                goal.status.as_str(),
                "active" | "paused" | "pending_confirmation" | "failed"
            ) || ManageMemoriesTool::is_protected_system_maintenance_goal(&goal)
                || self.state.get_mandate_for_goal(&goal.id).await?.is_some()
            {
                continue;
            }
            let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
            let runs: Vec<DateTime<Utc>> = schedules
                .iter()
                .filter(|schedule| !schedule.is_paused)
                .filter_map(|schedule| DateTime::parse_from_rfc3339(&schedule.next_run_at).ok())
                .map(|at| at.with_timezone(&Utc))
                .collect();
            let specs: Vec<ScheduleSpec> = schedules
                .iter()
                .map(|schedule| ScheduleSpec {
                    cron: schedule.cron_expr.clone(),
                    one_shot: schedule.is_one_shot,
                    next: crate::cron_utils::owner_time_rfc3339(&schedule.next_run_at),
                })
                .collect();
            let reminder = crate::reminders::message_from_context(goal.context.as_deref());
            let card =
                automations::scheduled_card(&goal.description, reminder.as_deref(), &specs, &zone);
            found.push(Automation {
                kind: AutomationKind::Scheduled,
                name: automations::name_from_context(goal.context.as_deref()),
                next: (goal.status == "active")
                    .then(|| runs.iter().min().copied())
                    .flatten(),
                status: goal.status,
                id: goal.id,
                card,
            });
        }
        Ok(found)
    }

    async fn watching(&self, session_id: &str) -> anyhow::Result<Vec<Automation>> {
        Ok(self
            .state
            .list_mandates(Some(session_id), false)
            .await?
            .into_iter()
            .map(|mandate| Automation {
                kind: AutomationKind::Watching,
                card: ManageMandatesTool::owner_card(&mandate, None),
                next: mandate
                    .scheduled_next_review()
                    .and_then(|at| DateTime::parse_from_rfc3339(at).ok())
                    .map(|at| at.with_timezone(&Utc)),
                name: mandate.name.clone(),
                status: mandate.status.as_str().to_string(),
                id: mandate.id,
            })
            .collect())
    }

    /// Every automation of `kind` (or both), next to run first.
    async fn all(
        &self,
        session_id: &str,
        kind: Option<AutomationKind>,
    ) -> anyhow::Result<Vec<Automation>> {
        let mut all = Vec::new();
        if kind != Some(AutomationKind::Watching) {
            all.extend(self.scheduled().await?);
        }
        if kind != Some(AutomationKind::Scheduled) {
            all.extend(self.watching(session_id).await?);
        }
        all.sort_by(|left, right| match (left.next, right.next) {
            (Some(l), Some(r)) => l.cmp(&r),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.id.cmp(&right.id),
        });
        Ok(all)
    }

    fn render_list(all: &[Automation], kind: Option<AutomationKind>) -> String {
        let heading = match kind {
            None => "Your automations".to_string(),
            Some(kind) => format!("Your {} automations", kind.as_str()),
        };
        if all.is_empty() {
            return format!("{heading}: none.");
        }
        let mut text = format!("{heading} ({}):\n", all.len());
        for (index, automation) in all.iter().enumerate() {
            text.push_str(&format!("{}. {}\n", index + 1, automation.row()));
        }
        text.push_str(
            "Pause, resume or cancel one by its name (or its ref when it has none). The refs are for your calls, not for the user.",
        );
        text
    }

    /// The one automation a call names, or the answer when it names none or
    /// several.
    async fn select(
        &self,
        args: &Args,
        session_id: &str,
        kind: Option<AutomationKind>,
    ) -> anyhow::Result<Result<Automation, String>> {
        let all = self.all(session_id, kind).await?;
        let name = args
            .name
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty());
        let id = args.id.as_deref().map(str::trim).filter(|v| !v.is_empty());
        let matches: Vec<&Automation> = match (id, name) {
            (Some(id), _) => all
                .iter()
                .filter(|automation| automation.id.starts_with(id))
                .collect(),
            (None, Some(name)) => all
                .iter()
                .filter(|automation| {
                    automation
                        .name
                        .as_deref()
                        .is_some_and(|own| own.eq_ignore_ascii_case(name))
                })
                .collect(),
            (None, None) => {
                return Ok(Err(
                    "Say which automation: its name, or its ref from the list.".to_string(),
                ))
            }
        };
        let label = name.or(id).unwrap_or_default();
        match matches.as_slice() {
            [one] => Ok(Ok((*one).clone())),
            [] => Ok(Err(format!(
                "No automation is named \"{label}\". {}",
                Self::render_list(&all, kind)
            ))),
            several => {
                let kinds: Vec<&str> = several
                    .iter()
                    .map(|automation| automation.kind.as_str())
                    .collect();
                Ok(Err(format!(
                    "More than one automation matches \"{label}\" ({}). Pass `kind` or the ref to pick one.",
                    kinds.join(", ")
                )))
            }
        }
    }

    /// The status the automation has now, or `None` once it is gone.
    async fn current_status(&self, automation: &Automation) -> anyhow::Result<Option<String>> {
        Ok(match automation.kind {
            AutomationKind::Scheduled => self
                .state
                .get_goal(&automation.id)
                .await?
                .map(|goal| goal.status),
            AutomationKind::Watching => self
                .state
                .get_mandate(&automation.id)
                .await?
                .map(|mandate| mandate.status.as_str().to_string()),
        })
    }

    async fn change(
        &self,
        automation: &Automation,
        action: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<(String, ToolOutcomeStatus)> {
        // A scheduled one that stopped after a problem resumes by retrying.
        let engine_action = if action == "resume" && automation.status == "failed" {
            "retry"
        } else {
            action
        };
        let call = match automation.kind {
            AutomationKind::Scheduled => json!({
                "action": format!("{engine_action}_scheduled"),
                "goal_id": automation.id,
            }),
            AutomationKind::Watching => json!({
                "action": action,
                "mandate_id": automation.id,
            }),
        }
        .to_string();
        let result = match automation.kind {
            AutomationKind::Scheduled => self.scheduler.run(&call, facts).await,
            AutomationKind::Watching => self.mandates.run(&call, facts).await,
        };
        if let Err(error) = &result {
            tracing::debug!(%error, action, "automation change was refused");
        }
        let wanted = match action {
            "pause" => "paused",
            "resume" => "active",
            _ => "cancelled",
        };
        let now = self.current_status(automation).await?;
        let title = automation.title();
        let kind = automation.kind.as_str();
        // Already so before the call: nothing changed.
        if now.as_deref() == Some(wanted) && automation.status != wanted {
            let text = match action {
                "pause" => {
                    format!("Paused {title} ({kind}). It won't run until you resume it.")
                }
                "resume" => {
                    let mut text = format!("Resumed {title} ({kind}).");
                    if let Some(refreshed) = self
                        .all(&facts.session_id, Some(automation.kind))
                        .await?
                        .into_iter()
                        .find(|candidate| candidate.id == automation.id)
                        .and_then(|candidate| candidate.next_words())
                    {
                        text.push_str(&format!(" Next run: {refreshed}."));
                    }
                    text
                }
                _ => format!("Cancelled {title} ({kind}). It won't run again."),
            };
            return Ok((text, ToolOutcomeStatus::Succeeded));
        }
        let state = automations::state_words(now.as_deref().unwrap_or("cancelled"));
        let mut text = format!("Couldn't {action} {title} ({kind}): it is {state}.");
        if now.as_deref() == Some("awaiting_input") {
            text.push_str(" It is waiting for the user's answer to its question.");
        }
        Ok((text, ToolOutcomeStatus::CompletedWithNegativeResult))
    }

    async fn run(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Args = serde_json::from_str(arguments)?;
        let outcome = |output: String, status: ToolOutcomeStatus| ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(status),
                ..ToolCallMetadata::default()
            },
            finish: None,
        };
        if !private_owner(facts) {
            return Ok(outcome(OWNER_ONLY.to_string(), ToolOutcomeStatus::Blocked));
        }
        let kind = match args
            .kind
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            None => None,
            Some(raw) => match AutomationKind::parse(raw) {
                Some(kind) => Some(kind),
                None => anyhow::bail!("kind must be scheduled or watching"),
            },
        };
        let session_id = facts.session_id.as_str();
        match args.action.as_str() {
            "list" => {
                let all = self.all(session_id, kind).await?;
                Ok(outcome(
                    Self::render_list(&all, kind),
                    ToolOutcomeStatus::Succeeded,
                ))
            }
            "show" => Ok(match self.select(&args, session_id, kind).await? {
                Ok(automation) => outcome(automation.render_card(), ToolOutcomeStatus::Succeeded),
                Err(answer) => outcome(answer, ToolOutcomeStatus::CompletedWithNegativeResult),
            }),
            action @ ("pause" | "resume" | "cancel") => {
                Ok(match self.select(&args, session_id, kind).await? {
                    Ok(automation) => {
                        let (text, status) = self.change(&automation, action, facts).await?;
                        outcome(text, status)
                    }
                    Err(answer) => outcome(answer, ToolOutcomeStatus::CompletedWithNegativeResult),
                })
            }
            other => anyhow::bail!("unknown manage_automations action `{other}`"),
        }
    }
}

fn action_of(arguments: &str) -> Option<String> {
    serde_json::from_str::<Value>(arguments)
        .ok()?
        .get("action")?
        .as_str()
        .map(str::to_string)
}

#[async_trait]
impl Tool for ManageAutomationsTool {
    fn name(&self) -> &str {
        "manage_automations"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_automations",
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ACTIONS },
                    "name": { "type": "string", "description": "The automation's name, as the user gave it." },
                    "id": { "type": "string", "description": "The ref from the list, for one with no name." },
                    "kind": { "type": "string", "enum": ["scheduled", "watching"], "description": "Narrows the list, or picks between two automations with the same name." }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.run(arguments, facts).await
    }
}

impl crate::traits::ToolSemantics for ManageAutomationsTool {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            always_offered: true,
            ..Default::default()
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        match action_of(arguments).as_deref() {
            Some("list" | "show") => ToolCallSemantics::observation(),
            _ => ToolCallSemantics::mutation_with(
                ToolMutationEffects::SCHEDULED_WORK.union(ToolMutationEffects::CONFIGURATION),
            ),
        }
    }

    /// Reads have no effects. A pause, resume or cancel changes what runs
    /// on its own, so it stays undeclared.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        match action_of(arguments).as_deref() {
            Some("list" | "show") => aidaemon_core::CallEffects::none(),
            _ => aidaemon_core::CallEffects::unknown(),
        }
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
    use crate::testing::CallWithFacts;
    use crate::testing::{setup_test_agent, MockProvider};

    fn owner() -> ToolCallContext {
        ToolCallContext {
            session_id: "owner-session".into(),
            user_role: Some(UserRole::Owner),
            channel_visibility: Some(ChannelVisibility::Private),
            ..Default::default()
        }
    }

    fn approved_owner() -> ToolCallContext {
        ToolCallContext {
            kernel_approved: Some(crate::traits::ApprovalScope::Call),
            ..owner()
        }
    }

    struct Fixture {
        state: Arc<dyn StateStore>,
        memory: Arc<dyn aidaemon_memory::stores::MemoryStores>,
        tool: ManageAutomationsTool,
        /// Scheduled automations by name, recorded when created.
        scheduled: std::sync::Mutex<std::collections::HashMap<String, String>>,
        /// Keeps the test database alive.
        _harness: crate::testing::TestHarness,
    }

    async fn fixture() -> Fixture {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state: Arc<dyn StateStore> = harness.state.clone();
        let memory = harness.state.memory_stores();
        let tool = ManageAutomationsTool::new(state.clone(), memory.clone());
        Fixture {
            state,
            memory,
            tool,
            scheduled: Default::default(),
            _harness: harness,
        }
    }

    impl Fixture {
        async fn reminder(&self, name: &str, when: &str) -> String {
            let output = crate::tools::SetReminderTool::new(self.state.clone(), self.memory())
                .call_with(
                    &json!({"message": "Check the oven.", "when": when, "name": name}).to_string(),
                    owner(),
                )
                .await
                .unwrap();
            self.record(name).await;
            output
        }

        /// Remember the goal now carrying `name`, if one does.
        async fn record(&self, name: &str) {
            let found = self
                .state
                .get_scheduled_goals()
                .await
                .unwrap()
                .into_iter()
                .find(|goal| {
                    automations::name_from_context(goal.context.as_deref()).as_deref() == Some(name)
                });
            if let Some(goal) = found {
                self.scheduled
                    .lock()
                    .unwrap()
                    .insert(name.to_string(), goal.id);
            }
        }

        async fn scheduled_work(&self, name: &str, schedule: &str) -> String {
            let output = ManageMemoriesTool::new(self.state.clone(), self.memory())
                .call_with(
                    &json!({
                        "action": "create_scheduled_goal",
                        "goal": "Summarize the synthetic inbox",
                        "schedule": schedule,
                        "name": name,
                    })
                    .to_string(),
                    approved_owner(),
                )
                .await
                .unwrap();
            self.record(name).await;
            output
        }

        async fn watching(&self, name: &str) -> String {
            ManageMandatesTool::new(self.state.clone())
                .call_with(
                    &json!({
                        "action": "create",
                        "name": name,
                        "objective": "Watch the synthetic status page",
                        "max_mutating_actions_per_cycle": 0,
                        "min_review_minutes": 60,
                        "max_review_minutes": 60,
                        "default_review_minutes": 60,
                        "success_criteria": ["The status page is reviewed without unauthorized action"],
                    })
                    .to_string(),
                    approved_owner(),
                )
                .await
                .unwrap()
        }

        fn memory(&self) -> Arc<dyn aidaemon_memory::stores::MemoryStores> {
            self.memory.clone()
        }

        async fn call(&self, args: Value) -> String {
            self.tool
                .call_with(&args.to_string(), owner())
                .await
                .unwrap()
        }

        async fn scheduled_status(&self, name: &str) -> String {
            let id = self.scheduled.lock().unwrap().get(name).cloned();
            let id = id.unwrap_or_else(|| panic!("no scheduled automation named {name}"));
            self.state.get_goal(&id).await.unwrap().unwrap().status
        }

        async fn mandate_status(&self, name: &str) -> String {
            self.state
                .list_mandates(Some("owner-session"), true)
                .await
                .unwrap()
                .into_iter()
                .find(|mandate| mandate.has_name(name))
                .map(|mandate| mandate.status.as_str().to_string())
                .unwrap_or_default()
        }
    }

    #[tokio::test]
    async fn one_list_shows_both_kinds_with_cadence_state_and_next_run() {
        let f = fixture().await;
        assert!(f
            .reminder("Oven", "every weekday at 8am")
            .await
            .starts_with("Reminder set"));
        let work = f.scheduled_work("Inbox digest", "every day at 9am").await;
        assert!(work.contains("Confirmed and activated"), "{work}");
        let watch = f.watching("Status watch").await;
        assert!(watch.contains("Activated mandate"), "{watch}");

        let list = f.call(json!({"action": "list"})).await;
        assert!(list.starts_with("Your automations (3):"), "{list}");
        for row in [
            "\"Oven\" · Scheduled · Every weekday at 8:00 AM",
            "\"Inbox digest\" · Scheduled · Every day at 9:00 AM",
            "\"Status watch\" · Watching · Every hour · On · next ",
        ] {
            assert!(list.contains(row), "missing {row:?} in {list}");
        }
        assert!(
            !list.contains("mandate") && !list.contains("goal"),
            "{list}"
        );

        // The typed kind narrows the list.
        let watching = f.call(json!({"action": "list", "kind": "watching"})).await;
        assert!(
            watching.starts_with("Your watching automations (1):"),
            "{watching}"
        );
        assert!(!watching.contains("Oven"), "{watching}");
    }

    #[tokio::test]
    async fn show_gives_the_same_five_line_card_for_both_kinds() {
        let f = fixture().await;
        f.scheduled_work("Inbox digest", "every day at 9am").await;
        f.watching("Status watch").await;

        let scheduled = f
            .call(json!({"action": "show", "name": "inbox digest"}))
            .await;
        for line in [
            "\"Inbox digest\" (Scheduled)",
            "What: Summarize the synthetic inbox",
            "How often: Every day at 9:00 AM",
            "Can do: Anything you can do – it runs with your full access",
            "Tells you when: a run finishes, or it hits a problem",
            "Ends: Only when you stop it",
            "State: On · next ",
        ] {
            assert!(scheduled.contains(line), "missing {line:?} in {scheduled}");
        }
        let watching = f
            .call(json!({"action": "show", "name": "Status watch"}))
            .await;
        for line in [
            "\"Status watch\" (Watching)",
            "What: Watch the synthetic status page",
            "How often: Every hour",
            "Can do: Read only – makes no changes",
            "Tells you when: ",
            "Ends: Only when you stop it",
        ] {
            assert!(watching.contains(line), "missing {line:?} in {watching}");
        }
    }

    #[tokio::test]
    async fn cancel_by_name_works_for_each_kind() {
        let f = fixture().await;
        f.reminder("Oven", "every weekday at 8am").await;
        f.watching("Status watch").await;

        let cancelled = f.call(json!({"action": "cancel", "name": "oven"})).await;
        assert_eq!(
            cancelled,
            "Cancelled \"Oven\" (scheduled). It won't run again."
        );
        assert_eq!(f.scheduled_status("Oven").await, "cancelled");

        let cancelled = f
            .call(json!({"action": "cancel", "name": "Status watch"}))
            .await;
        assert_eq!(
            cancelled,
            "Cancelled \"Status watch\" (watching). It won't run again."
        );
        assert_eq!(f.mandate_status("Status watch").await, "cancelled");

        let list = f.call(json!({"action": "list"})).await;
        assert_eq!(list, "Your automations: none.");
        let missing = f.call(json!({"action": "pause", "name": "Oven"})).await;
        assert!(
            missing.starts_with("No automation is named \"Oven\""),
            "{missing}"
        );
    }

    #[tokio::test]
    async fn pause_and_resume_by_name_round_trip_for_each_kind() {
        let f = fixture().await;
        f.scheduled_work("Inbox digest", "every day at 9am").await;
        f.watching("Status watch").await;
        for (name, kind) in [("Inbox digest", "scheduled"), ("Status watch", "watching")] {
            let paused = f.call(json!({"action": "pause", "name": name})).await;
            assert_eq!(
                paused,
                format!("Paused \"{name}\" ({kind}). It won't run until you resume it.")
            );
            let again = f.call(json!({"action": "pause", "name": name})).await;
            assert_eq!(
                again,
                format!("Couldn't pause \"{name}\" ({kind}): it is Paused.")
            );
            let resumed = f.call(json!({"action": "resume", "name": name})).await;
            assert!(
                resumed.starts_with(&format!("Resumed \"{name}\" ({kind}).")),
                "{resumed}"
            );
        }
        assert_eq!(f.scheduled_status("Inbox digest").await, "active");
        assert_eq!(f.mandate_status("Status watch").await, "active");
    }

    #[tokio::test]
    async fn the_typed_kind_picks_between_two_automations_with_one_name() {
        let f = fixture().await;
        f.reminder("Prices", "every weekday at 8am").await;
        f.watching("Prices").await;

        let ambiguous = f.call(json!({"action": "pause", "name": "Prices"})).await;
        assert!(
            ambiguous.starts_with("More than one automation matches"),
            "{ambiguous}"
        );
        assert_eq!(f.scheduled_status("Prices").await, "active");
        assert_eq!(f.mandate_status("Prices").await, "active");

        let paused = f
            .call(json!({"action": "pause", "name": "Prices", "kind": "watching"}))
            .await;
        assert!(
            paused.starts_with("Paused \"Prices\" (watching)"),
            "{paused}"
        );
        assert_eq!(f.mandate_status("Prices").await, "paused");
        assert_eq!(f.scheduled_status("Prices").await, "active");

        let cancelled = f
            .call(json!({"action": "cancel", "name": "Prices", "kind": "scheduled"}))
            .await;
        assert!(
            cancelled.starts_with("Cancelled \"Prices\" (scheduled)"),
            "{cancelled}"
        );
        assert_eq!(f.mandate_status("Prices").await, "paused");
    }

    #[tokio::test]
    async fn a_name_already_in_use_is_refused_and_unnamed_ones_go_by_ref() {
        let f = fixture().await;
        f.watching("Oven").await;
        let refused = f.reminder("oven", "in 2h").await;
        assert!(
            refused.starts_with("An automation is already named \"oven\" (a watching one)"),
            "{refused}"
        );

        let set = crate::tools::SetReminderTool::new(f.state.clone(), f.memory())
            .call_with(
                &json!({"message": "Stretch.", "when": "every day at 3pm"}).to_string(),
                owner(),
            )
            .await
            .unwrap();
        assert!(set.starts_with("Reminder set"), "{set}");
        let list = f.call(json!({"action": "list", "kind": "scheduled"})).await;
        let row = list
            .lines()
            .find(|line| line.contains("Reminds you: Stretch."))
            .unwrap_or_else(|| panic!("{list}"));
        let reference = row
            .rsplit_once("[ref ")
            .and_then(|(_, rest)| rest.strip_suffix(']'))
            .unwrap();
        let cancelled = f.call(json!({"action": "cancel", "id": reference})).await;
        assert!(
            cancelled.starts_with("Cancelled Reminds you: Stretch."),
            "{cancelled}"
        );
    }

    #[tokio::test]
    async fn only_the_owner_in_a_private_chat_sees_automations() {
        let f = fixture().await;
        for facts in [
            ToolCallContext {
                user_role: Some(UserRole::Guest),
                ..owner()
            },
            ToolCallContext {
                channel_visibility: Some(ChannelVisibility::Public),
                ..owner()
            },
            ToolCallContext {
                channel_visibility: Some(ChannelVisibility::Internal),
                ..owner()
            },
        ] {
            let answer = f
                .tool
                .call_with(&json!({"action": "list"}).to_string(), facts)
                .await
                .unwrap();
            assert_eq!(answer, OWNER_ONLY);
        }
    }
}
