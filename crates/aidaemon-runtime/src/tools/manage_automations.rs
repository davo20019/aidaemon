//! `manage_automations`: the owner's one list of everything that runs on its
//! own, and every change to it.
//!
//! - *Scheduled*: runs a fixed instruction at set times (scheduled goals and
//!   reminders, created with `schedule_task` or `set_reminder`).
//! - *Watching*: works toward an objective within limits the owner approved
//!   (mandates, created with `manage_mandates draft` then `create`).
//! - *Probe*: checks a service, port, command or file on a schedule (health
//!   probes, created with `health_probe create`), when the daemon runs them.
//!
//! List, show, edit (reschedule, rename, snooze), pause, resume and cancel
//! are the owner's, in a private chat. Running scheduled work now, its run
//! history, and scheduled work's last failure, budget, instructions and
//! workspace keep the checks they always had.
//! Every change is judged by reading the stored state back, never by the
//! wording of an answer.

use std::sync::Arc;

use aidaemon_core::mandate_card::MandateConfirmationCard;
use aidaemon_work::automations::{self, AutomationKind, ScheduleSpec};
use aidaemon_work::health::HealthProbeStore;
use aidaemon_work::tools::scheduled_goal_runs::ScheduledGoalRuns;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::traits::{
    AutomationStore, Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus, ToolResultPresentation,
};
use crate::tools::traits::{ChannelVisibility, UserRole};
use aidaemon_work::tools::manage_mandates::ManageMandatesTool;

const TOOL_DESCRIPTION: &str = "The user's automations: scheduled (an instruction or reminder at set times), watching (mandates) and health checks. List, show, history, edit (when, new_name, snooze_until, tell_me), pause, resume or cancel one by name; for scheduled work also run_now, last_failure, set_budget, update_instructions, bind_workspace. Watching history shows goal runs and decisions in words; health-check history shows stored status, latency and error. Create with schedule_task or set_reminder.";

const ACTIONS: &[&str] = &[
    "list",
    "show",
    "edit",
    "pause",
    "resume",
    "cancel",
    "run_now",
    "history",
    "last_failure",
    "set_budget",
    "update_instructions",
    "bind_workspace",
];

/// Who may make a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gate {
    /// The owner, in a private chat, outside internal work.
    PrivateOwner,
    /// The owner, wherever (scheduled runs act as the owner).
    Owner,
    /// Anyone but a guest (task leads and executors bind their workspace).
    NotGuest,
}

fn gate(action: &str) -> Gate {
    match action {
        "run_now" | "set_budget" | "update_instructions" => Gate::Owner,
        "history" | "last_failure" | "bind_workspace" => Gate::NotGuest,
        _ => Gate::PrivateOwner,
    }
}

pub struct ManageAutomationsTool {
    state: Arc<dyn AutomationStore>,
    mandates: ManageMandatesTool,
    runs: ScheduledGoalRuns,
    /// Health probes, when the daemon runs them.
    probes: Option<Arc<HealthProbeStore>>,
}

impl ManageAutomationsTool {
    pub fn new(state: Arc<dyn AutomationStore>) -> Self {
        Self {
            mandates: ManageMandatesTool::new(state.clone()),
            runs: ScheduledGoalRuns::new(state.clone()),
            probes: None,
            state,
        }
    }

    /// Also list and change health probes.
    pub fn with_health_probes(mut self, probes: Option<Arc<HealthProbeStore>>) -> Self {
        self.probes = probes;
        self
    }
}

#[derive(Debug, Default, Deserialize)]
struct Args {
    action: String,
    name: Option<String>,
    id: Option<String>,
    ids: Option<Vec<String>>,
    kind: Option<String>,
    when: Option<String>,
    snooze_until: Option<String>,
    new_name: Option<String>,
    tell_me: Option<String>,
    limit: Option<usize>,
    budget_per_check: Option<i64>,
    budget_daily: Option<i64>,
    instructions: Option<String>,
    run_requirement: Option<String>,
    insufficient_evidence: Option<String>,
    workspace_path: Option<String>,
}

/// A string argument, if it says anything (strict-schema providers send
/// every field, empty when unused).
fn given(value: &Option<String>) -> Option<&str> {
    value.as_deref().map(str::trim).filter(|v| !v.is_empty())
}

/// One automation as the owner sees it.
#[derive(Debug, Clone)]
struct Automation {
    kind: AutomationKind,
    /// The goal ID (scheduled), mandate ID (watching) or probe ID (probe).
    id: String,
    name: Option<String>,
    /// The goal or mandate status; `active` or `paused` for a probe.
    status: String,
    card: MandateConfirmationCard,
    next: Option<DateTime<Utc>>,
    /// A probe from the config file, which startup would add back.
    from_config: bool,
    /// What a client may change in place (scheduled work only).
    edit: Option<Editable>,
    /// Human-only context for a scheduled automation held by recovery.
    held_after_failures: Option<HeldAfterFailures>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HeldAfterFailures {
    consecutive_failures: u16,
    last_failure_reason: Option<String>,
}

fn held_after_failures(
    recovery: &aidaemon_work::goals::ScheduledRecoveryState,
    runs: &[aidaemon_work::goals::GoalRun],
) -> Option<HeldAfterFailures> {
    (recovery.disposition == aidaemon_work::goals::ScheduledRecoveryDisposition::Escalated).then(
        || HeldAfterFailures {
            consecutive_failures: recovery.consecutive_failures,
            last_failure_reason: recovery.last_failed_run_id.as_deref().and_then(|run_id| {
                runs.iter()
                    .find(|run| run.id == run_id)
                    .and_then(|run| run.outcome_summary.as_deref())
                    .map(automations::run_summary_words)
                    .filter(|summary| !summary.is_empty())
            }),
        },
    )
}

/// A scheduled automation's changeable fields as they are now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Editable {
    /// When it runs, in the words it was set with.
    pub when: String,
    /// What its runs do; `None` for a reminder.
    pub instructions: Option<String>,
    /// What its runs tell the owner; `None` for a reminder (it always does).
    pub notify: Option<automations::NotifyMode>,
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
        self.next.map(|at| aidaemon_work::schedule::owner_time(&at))
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
        if let Some(held) = &self.held_after_failures {
            row.push_str(&format!(
                " · {} consecutive failures",
                held.consecutive_failures
            ));
            if let Some(reason) = &held.last_failure_reason {
                row.push_str(&format!(" · last failure: {reason}"));
            }
        }
        row
    }

    fn render_card(&self) -> String {
        let mut text = format!("{} ({})\n", self.title(), self.kind.label());
        for (essential, line) in self.card.essentials() {
            text.push_str(&format!("{}: {line}\n", essential.label()));
        }
        for detail in &self.card.details {
            text.push_str(&format!("{detail}\n"));
        }
        text.push_str(&format!(
            "State: {}",
            automations::state_words(&self.status)
        ));
        if let Some(next) = self.next_words() {
            text.push_str(&format!(" · next {next}"));
        }
        if let Some(held) = &self.held_after_failures {
            text.push_str(&format!(
                "\nConsecutive failures: {}",
                held.consecutive_failures
            ));
            if let Some(reason) = &held.last_failure_reason {
                text.push_str(&format!("\nLast failure: {reason}"));
            }
        }
        text
    }
}

/// Who already holds `name` among the owner's live automations, as the
/// owner would say it ("a scheduled one"); `None` when it is free.
pub(crate) async fn live_name_holder(
    state: &dyn AutomationStore,
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

/// The daemon's own maintenance goals, which nobody lists or cancels.
pub(crate) fn is_protected_system_maintenance_goal(goal: &crate::tools::traits::Goal) -> bool {
    const KNOWLEDGE_GOAL_DESC: &str =
        "Maintain knowledge base: process embeddings, consolidate memories, decay old facts";
    const HEALTH_GOAL_DESC: &str =
        "Maintain memory health: prune old events, clean up retention, remove stale data";
    if goal.description == KNOWLEDGE_GOAL_DESC || goal.description == HEALTH_GOAL_DESC {
        return true;
    }
    goal.context
        .as_deref()
        .and_then(|ctx| serde_json::from_str::<Value>(ctx).ok())
        .and_then(|v| v.get("system_protected").and_then(Value::as_bool))
        == Some(true)
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

fn allowed(gate: Gate, facts: &ToolCallContext) -> bool {
    match gate {
        Gate::PrivateOwner => private_owner(facts),
        Gate::Owner => facts.user_role == Some(UserRole::Owner),
        Gate::NotGuest => facts.user_role != Some(UserRole::Guest),
    }
}

/// The refusal outside the owner's private chat.
pub const OWNER_ONLY: &str =
    "Automations can only be seen or changed by the owner in a private chat.";

/// The cap on recurring goals that are on at once.
const MAX_ACTIVE_EVERGREEN: i64 = 10;

/// What one change wrote, before it is read back.
type ChangeNote = Option<String>;

/// One automation as a client lists it (the app's Automations page).
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationEntry {
    pub kind: AutomationKind,
    pub id: String,
    pub name: Option<String>,
    /// What it does, in the owner's words.
    pub what: String,
    /// When it runs ("Every weekday at 9:00 AM").
    pub how_often: String,
    /// `active`, `paused`, `failed`, `awaiting_input`...
    pub status: String,
    pub next: Option<DateTime<Utc>>,
    /// Its latest runs, newest first.
    pub runs: Vec<RunEntry>,
    /// The rest of its card: what it may do, when it tells you, when it
    /// ends, and its details.
    pub details: Vec<String>,
    /// What a client may change in place.
    pub edit: Option<Editable>,
    /// The conversation its runs work in (a goal's session), when it has
    /// one: where a run waits for the owner.
    pub session_id: Option<String>,
}

/// One run of an automation, as its records left it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunEntry {
    pub started: DateTime<Utc>,
    pub finished: Option<DateTime<Utc>>,
    /// `pending`, `running`, `completed`, `failed`, `blocked`, `cancelled`.
    pub status: String,
    pub summary: Option<String>,
}

/// How many runs a client is shown per automation.
pub const RECENT_RUNS: usize = 5;

fn parse_time(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|at| at.with_timezone(&Utc))
}

/// What the owner can do to an automation from a client.
pub const CLIENT_CHANGES: [&str; 4] = ["pause", "resume", "cancel", "run_now"];

fn outcome(output: String, status: ToolOutcomeStatus) -> ToolCallOutcome {
    ToolCallOutcome {
        output,
        metadata: ToolCallMetadata {
            outcome_status: Some(status),
            ..ToolCallMetadata::default()
        },
        finish: None,
    }
}

fn negative(output: String) -> ToolCallOutcome {
    outcome(output, ToolOutcomeStatus::CompletedWithNegativeResult)
}

impl ManageAutomationsTool {
    /// Scheduled work, whichever of the owner's sessions created it: a
    /// scheduled goal belongs to the owner, not to one channel session, and
    /// every owner surface reports one membership for the scheduled
    /// collection (R53). Watching ones are listed for this session, because
    /// a mandate's authority is bound to the session that approved it.
    async fn scheduled(&self) -> anyhow::Result<Vec<Automation>> {
        let zone = aidaemon_work::schedule::system_timezone_display();
        let mut found = Vec::new();
        for goal in self.state.get_scheduled_goals().await? {
            if !matches!(
                goal.status.as_str(),
                "active" | "paused" | "pending_confirmation" | "failed"
            ) || is_protected_system_maintenance_goal(&goal)
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
                    zone: zone_label(&schedule.tz),
                    one_shot: schedule.is_one_shot,
                    next: aidaemon_work::schedule::owner_time_rfc3339(&schedule.next_run_at),
                    following: Vec::new(),
                })
                .collect();
            let reminder = aidaemon_work::reminders::message_from_context(goal.context.as_deref());
            let edit = Editable {
                when: schedules
                    .iter()
                    .find_map(|schedule| schedule.original_schedule.clone())
                    .filter(|words| !words.trim().is_empty())
                    .unwrap_or_else(|| {
                        schedules
                            .first()
                            .map(|schedule| schedule.cron_expr.clone())
                            .unwrap_or_default()
                    }),
                notify: reminder
                    .is_none()
                    .then(|| automations::NotifyMode::from_context(goal.context.as_deref())),
                instructions: match reminder {
                    Some(_) => None,
                    None => Some(
                        goal.context
                            .as_deref()
                            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                            .and_then(|context| {
                                context
                                    .get("run_instructions")
                                    .and_then(Value::as_str)
                                    .map(str::to_string)
                            })
                            .unwrap_or_else(|| goal.description.clone()),
                    ),
                },
            };
            // What its runs do now: the owner's latest instructions when it
            // was given some, never the compiled run policy around them.
            let what = edit
                .instructions
                .clone()
                .unwrap_or_else(|| goal.description.clone());
            let mut card = automations::scheduled_card(&what, reminder.as_deref(), &specs, &zone);
            match edit.notify {
                Some(automations::NotifyMode::Changes) => {
                    card.tells_you_when =
                        "only when a run finds something new, or hits a problem".to_string();
                }
                Some(automations::NotifyMode::Problems) => {
                    card.tells_you_when = "only when a run fails or needs you".to_string();
                }
                _ => {}
            }
            if let Some(workspace) = goal
                .context
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .and_then(|context| {
                    context
                        .get("project_scope")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
            {
                card.details.push(format!("Workspace: {workspace}"));
            }
            // What it spends, so cost is never a surprise (research
            // 2026-11-15: opaque credit costs were a top complaint).
            if let Some(spend) = automations::spend_line(
                goal.tokens_used_today,
                &goal.tokens_used_day,
                &Utc::now().date_naive().to_string(),
                goal.budget_per_check,
                goal.budget_daily,
            ) {
                card.details.push(spend);
            }
            // Its runs kept failing and the recovery ledger paused its
            // schedules: on, but not running, which the owner must see.
            let recovery = self.state.get_scheduled_recovery_state(&goal.id).await?;
            let held_after_failures = if goal.status == "active" {
                match recovery.as_ref() {
                    Some(recovery)
                        if recovery.disposition
                            == aidaemon_work::goals::ScheduledRecoveryDisposition::Escalated =>
                    {
                        let runs = self.state.get_goal_runs(&goal.id).await?;
                        held_after_failures(recovery, &runs)
                    }
                    _ => None,
                }
            } else {
                None
            };
            let held = held_after_failures.is_some();
            found.push(Automation {
                kind: AutomationKind::Scheduled,
                name: automations::name_from_context(goal.context.as_deref()),
                next: (goal.status == "active" && !held)
                    .then(|| runs.iter().min().copied())
                    .flatten(),
                status: if held {
                    automations::HELD_AFTER_FAILURES.to_string()
                } else {
                    goal.status
                },
                id: goal.id,
                card,
                from_config: false,
                edit: Some(edit),
                held_after_failures,
            });
        }
        Ok(found)
    }

    /// Watching automations of `session_id`, or of every session.
    async fn watching(&self, session_id: Option<&str>) -> anyhow::Result<Vec<Automation>> {
        Ok(self
            .state
            .list_mandates(session_id, false)
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
                from_config: false,
                edit: None,
                held_after_failures: None,
            })
            .collect())
    }

    async fn probes(&self) -> anyhow::Result<Vec<Automation>> {
        let Some(store) = &self.probes else {
            return Ok(Vec::new());
        };
        let zone = aidaemon_work::schedule::system_timezone_display();
        Ok(store
            .list_probes()
            .await?
            .into_iter()
            .map(|probe| Automation {
                kind: AutomationKind::Probe,
                card: automations::probe_card(
                    probe.probe_type.as_str(),
                    &probe.target,
                    &probe.schedule,
                    &zone,
                ),
                next: (!probe.is_paused).then_some(probe.next_run_at),
                status: if probe.is_paused { "paused" } else { "active" }.to_string(),
                edit: (probe.source != "config").then(|| Editable {
                    when: probe.schedule.clone(),
                    notify: None,
                    instructions: None,
                }),
                name: Some(probe.name),
                from_config: probe.source == "config",
                id: probe.id,
                held_after_failures: None,
            })
            .collect())
    }

    /// Every automation the owner has, for a client: scheduled work, every
    /// session's watching automations, and health probes; next to run first.
    pub async fn entries(&self) -> anyhow::Result<Vec<AutomationEntry>> {
        let mut all = self.scheduled().await?;
        all.extend(self.watching(None).await?);
        all.extend(self.probes().await?);
        all.sort_by(|left, right| match (left.next, right.next) {
            (Some(l), Some(r)) => l.cmp(&r),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.id.cmp(&right.id),
        });
        let mut entries = Vec::with_capacity(all.len());
        for automation in all {
            // A run's records are no reason to hide the automation.
            let runs = self
                .recent_runs(automation.kind, &automation.id)
                .await
                .unwrap_or_default();
            let details = [
                format!("Can do: {}", automation.card.can_do),
                format!("Tells you when: {}", automation.card.tells_you_when),
                format!("Ends: {}", automation.card.ends),
            ]
            .into_iter()
            .chain(automation.card.details.iter().cloned())
            .filter(|line| !line.trim_end().ends_with(':'))
            .collect();
            let session_id = self.session_of(automation.kind, &automation.id).await;
            entries.push(AutomationEntry {
                session_id,
                kind: automation.kind,
                id: automation.id,
                name: automation.name,
                what: automation.card.what,
                how_often: automation.card.how_often,
                status: automation.status,
                next: automation.next,
                runs,
                details,
                edit: automation.edit,
            });
        }
        Ok(entries)
    }

    /// The conversation automation `id` runs in: its goal's session.
    async fn session_of(&self, kind: AutomationKind, id: &str) -> Option<String> {
        let goal_id = match kind {
            AutomationKind::Scheduled => id.to_string(),
            AutomationKind::Watching => self.state.get_mandate(id).await.ok()??.goal_id,
            AutomationKind::Probe => return None,
        };
        let goal = self.state.get_goal(&goal_id).await.ok()??;
        (!goal.session_id.is_empty()).then_some(goal.session_id)
    }

    /// The latest runs of automation `id`, newest first: a scheduled goal's
    /// and a watching automation's goal runs, a health check's results.
    async fn recent_runs(&self, kind: AutomationKind, id: &str) -> anyhow::Result<Vec<RunEntry>> {
        let goal_id = match kind {
            AutomationKind::Scheduled => id.to_string(),
            AutomationKind::Watching => match self.state.get_mandate(id).await? {
                Some(mandate) => mandate.goal_id,
                None => return Ok(Vec::new()),
            },
            AutomationKind::Probe => {
                let Some(store) = &self.probes else {
                    return Ok(Vec::new());
                };
                return Ok(store
                    .get_results(id, RECENT_RUNS)
                    .await?
                    .into_iter()
                    .map(|result| {
                        let healthy = matches!(
                            result.status,
                            aidaemon_work::health::probes::ProbeStatus::Healthy
                        );
                        RunEntry {
                            started: result.checked_at,
                            finished: Some(result.checked_at),
                            status: if healthy { "completed" } else { "failed" }.to_string(),
                            summary: if healthy {
                                result.latency_ms.map(|ms| format!("Healthy in {ms} ms"))
                            } else {
                                Some(result.error_message.unwrap_or_else(|| {
                                    format!("Check result: {}", result.status.as_str())
                                }))
                            },
                        }
                    })
                    .collect());
            }
        };
        let mut runs = self.state.get_goal_runs(&goal_id).await?;
        runs.truncate(RECENT_RUNS);
        Ok(runs
            .into_iter()
            .filter_map(|run| {
                Some(RunEntry {
                    started: parse_time(&run.started_at)?,
                    finished: run.completed_at.as_deref().and_then(parse_time),
                    status: run.status,
                    summary: run
                        .outcome_summary
                        .map(|summary| automations::run_summary_words(&summary))
                        .filter(|summary| !summary.is_empty()),
                })
            })
            .collect())
    }

    async fn history(&self, automation: &Automation, limit: usize) -> anyhow::Result<String> {
        let limit = limit.clamp(1, 50);
        match automation.kind {
            AutomationKind::Scheduled => self.runs.run_history(&automation.id, limit).await,
            AutomationKind::Watching => {
                let Some(mandate) = self.state.get_mandate(&automation.id).await? else {
                    return Ok(format!("No history found for {}.", automation.title()));
                };
                let runs = self
                    .state
                    .get_goal_runs(&mandate.goal_id)
                    .await?
                    .into_iter()
                    .take(limit)
                    .collect::<Vec<_>>();
                if runs.is_empty() {
                    return Ok(format!("No runs found yet for {}.", automation.title()));
                }
                let decisions = self.state.list_mandate_decisions(&mandate.id, 200).await?;
                let mut out = format!(
                    "**Watching History**\n\n- Automation: {}\n- Showing: {} run(s)",
                    automation.title(),
                    runs.len()
                );
                for run in runs {
                    out.push_str(&format!("\n\n- {} · {}", run.started_at, run.status));
                    if let Some(summary) = run.outcome_summary.as_deref() {
                        let summary = automations::run_summary_words(summary);
                        if !summary.is_empty() {
                            out.push_str(&format!(" · {summary}"));
                        }
                    }
                    if let Some(decision) = decisions
                        .iter()
                        .find(|decision| decision.goal_run_id == run.id)
                    {
                        let word = match decision.outcome {
                            aidaemon_work::mandates::MandateDecisionOutcome::Act => "Act",
                            aidaemon_work::mandates::MandateDecisionOutcome::Wait => "Wait",
                            aidaemon_work::mandates::MandateDecisionOutcome::Ask => "Ask",
                            aidaemon_work::mandates::MandateDecisionOutcome::Stop => "Stop",
                        };
                        out.push_str(&format!("\n  Decision: {word} — {}", decision.rationale));
                    }
                }
                Ok(out)
            }
            AutomationKind::Probe => {
                let Some(store) = &self.probes else {
                    return Ok(format!("No history found for {}.", automation.title()));
                };
                let results = store.get_results(&automation.id, limit).await?;
                if results.is_empty() {
                    return Ok(format!("No checks found yet for {}.", automation.title()));
                }
                let mut out = format!(
                    "**Health Check History**\n\n- Automation: {}\n- Showing: {} check(s)",
                    automation.title(),
                    results.len()
                );
                for result in results {
                    let latency = result
                        .latency_ms
                        .map_or_else(|| "not recorded".to_string(), |ms| format!("{ms} ms"));
                    let error = result.error_message.as_deref().unwrap_or("none");
                    out.push_str(&format!(
                        "\n\n- {} · status: {} · latency: {} · error: {}",
                        result.checked_at,
                        result.status.as_str(),
                        latency,
                        error
                    ));
                }
                Ok(out)
            }
        }
    }

    /// Pause, resume, cancel or (scheduled work) run automation `id` now, as
    /// the owner in a private chat: through this tool's own checks and
    /// read-back, as when the owner asks the agent. A watching automation is
    /// changed in the session that approved it. Returns whether it changed,
    /// and what to tell the owner.
    pub async fn owner_change(&self, id: &str, action: &str) -> anyhow::Result<(bool, String)> {
        if !CLIENT_CHANGES.contains(&action) {
            return Ok((
                false,
                format!("Automations cannot be changed with {action}."),
            ));
        }
        let all = self.all("", None).await?;
        let mut found = all.into_iter().find(|automation| automation.id == id);
        if found.is_none() {
            found = self
                .watching(None)
                .await?
                .into_iter()
                .find(|automation| automation.id == id);
        }
        let Some(automation) = found else {
            return Ok((false, "That automation is gone.".to_string()));
        };
        if action == "run_now" && automation.kind != AutomationKind::Scheduled {
            return Ok((false, "Only scheduled work can run now.".to_string()));
        }
        let session_id = match automation.kind {
            AutomationKind::Watching => self
                .state
                .get_mandate(&automation.id)
                .await?
                .map(|mandate| mandate.created_by_session)
                .unwrap_or_default(),
            _ => String::new(),
        };
        let facts = ToolCallContext {
            session_id,
            user_role: Some(UserRole::Owner),
            channel_visibility: Some(ChannelVisibility::Private),
            ..ToolCallContext::default()
        };
        let call = json!({
            "action": action,
            "id": automation.id,
            "kind": automation.kind.as_str(),
        })
        .to_string();
        let result = self.run(&call, &facts).await?;
        let changed = result.metadata.outcome_status == Some(ToolOutcomeStatus::Succeeded);
        Ok((changed, result.output))
    }

    /// Rename, reschedule or re-instruct scheduled automation `id` as the
    /// owner in a private chat, through this tool's own checks and
    /// read-back; a `None` field stays as it is. Returns whether anything
    /// changed, and what to tell the owner.
    pub async fn owner_edit(
        &self,
        id: &str,
        name: Option<&str>,
        when: Option<&str>,
        instructions: Option<&str>,
        notify: Option<automations::NotifyMode>,
    ) -> anyhow::Result<(bool, String)> {
        let Some(automation) = self
            .all("", None)
            .await?
            .into_iter()
            .find(|automation| automation.id == id)
        else {
            return Ok((
                false,
                "Only scheduled work and reminders change here; ask in a conversation to change this one."
                    .to_string(),
            ));
        };
        if automation.kind == AutomationKind::Watching {
            return Ok((
                false,
                "Watching automations change through their mandate.".to_string(),
            ));
        }
        if automation.kind == AutomationKind::Probe && (instructions.is_some() || notify.is_some())
        {
            return Ok((
                false,
                "Health checks can only change their name or schedule.".to_string(),
            ));
        }
        let editable = automation.edit.clone();
        fn differs<'a>(new: Option<&'a str>, old: Option<&str>) -> Option<&'a str> {
            new.map(str::trim)
                .filter(|new| !new.is_empty() && Some(*new) != old.map(str::trim))
        }
        let name = differs(name, automation.name.as_deref());
        let when = differs(when, editable.as_ref().map(|e| e.when.as_str()));
        let instructions = differs(
            instructions,
            editable.as_ref().and_then(|e| e.instructions.as_deref()),
        );
        if instructions.is_some() && editable.as_ref().is_some_and(|e| e.instructions.is_none()) {
            return Ok((
                false,
                "A reminder's message is fixed; set a new reminder instead.".to_string(),
            ));
        }
        let facts = ToolCallContext {
            user_role: Some(UserRole::Owner),
            channel_visibility: Some(ChannelVisibility::Private),
            ..ToolCallContext::default()
        };
        let notify = notify.filter(|mode| editable.as_ref().and_then(|e| e.notify) != Some(*mode));
        let mut said = Vec::new();
        let mut changed = false;
        if name.is_some() || when.is_some() || notify.is_some() {
            let mut call =
                json!({ "action": "edit", "id": automation.id, "kind": automation.kind.as_str() });
            if let Some(mode) = notify {
                call["tell_me"] = json!(mode.as_str());
            }
            if let Some(name) = name {
                call["new_name"] = json!(name);
            }
            if let Some(when) = when {
                call["when"] = json!(when);
            }
            let result = self.run(&call.to_string(), &facts).await?;
            let ok = result.metadata.outcome_status == Some(ToolOutcomeStatus::Succeeded);
            said.push(result.output);
            if !ok {
                return Ok((changed, said.join(" ")));
            }
            changed = true;
        }
        if let Some(instructions) = instructions {
            let call = json!({
                "action": "update_instructions",
                "id": automation.id,
                "kind": "scheduled",
                "instructions": instructions,
            });
            let result = self.run(&call.to_string(), &facts).await?;
            changed |= result.metadata.outcome_status == Some(ToolOutcomeStatus::Succeeded);
            said.push(result.output);
        }
        if said.is_empty() {
            return Ok((false, "Nothing changed.".to_string()));
        }
        Ok((changed, said.join(" ")))
    }

    /// Every automation of `kind` (or every kind), next to run first.
    async fn all(
        &self,
        session_id: &str,
        kind: Option<AutomationKind>,
    ) -> anyhow::Result<Vec<Automation>> {
        let mut all = Vec::new();
        let wants = |candidate: AutomationKind| kind.is_none_or(|kind| kind == candidate);
        if wants(AutomationKind::Scheduled) {
            all.extend(self.scheduled().await?);
        }
        if wants(AutomationKind::Watching) {
            all.extend(self.watching(Some(session_id)).await?);
        }
        if wants(AutomationKind::Probe) {
            all.extend(self.probes().await?);
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
            "Change one by its name (or its ref when it has none). The refs are for your calls, not for the user.",
        );
        text
    }

    /// The one automation `name` or `id` names, or the answer when it names
    /// none or several.
    async fn select(
        &self,
        name: Option<&str>,
        id: Option<&str>,
        session_id: &str,
        kind: Option<AutomationKind>,
    ) -> anyhow::Result<Result<Automation, String>> {
        let all = self.all(session_id, kind).await?;
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
                ));
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
            AutomationKind::Probe => match &self.probes {
                Some(store) => store
                    .get_probe(&automation.id)
                    .await?
                    .map(|probe| if probe.is_paused { "paused" } else { "active" }.to_string()),
                None => None,
            },
        })
    }

    /// Pause, resume (a failed one resumes by retrying) or cancel one
    /// scheduled goal. Returns why it was refused, when the store says why.
    async fn change_scheduled(&self, goal_id: &str, action: &str) -> anyhow::Result<ChangeNote> {
        let Some(mut goal) = self.state.get_goal(goal_id).await? else {
            return Ok(None);
        };
        let now = Utc::now().to_rfc3339();
        match action {
            "pause" => {
                if goal.status != "active" {
                    return Ok(None);
                }
                goal.status = "paused".to_string();
            }
            "resume" => {
                let retry = matches!(goal.status.as_str(), "failed" | "stalled");
                if goal.status != "paused" && !retry {
                    return Ok(None);
                }
                if goal.goal_type == "continuous" {
                    let active = self.state.count_active_evergreen_goals().await?;
                    if active >= MAX_ACTIVE_EVERGREEN {
                        return Ok(Some(format!(
                            "{active} recurring automations are already on, the most there can be; pause or cancel one first."
                        )));
                    }
                }
                goal.status = "active".to_string();
                if retry {
                    goal.completed_at = None;
                    goal.dispatch_failures = 0;
                }
            }
            _ => {
                if is_protected_system_maintenance_goal(&goal) {
                    return Ok(Some("It is the daemon's own maintenance.".to_string()));
                }
                // Captured before the status changes: the store purges the
                // schedules of a terminal goal as a safety net.
                let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
                goal.status = "cancelled".to_string();
                goal.completed_at = Some(now.clone());
                goal.updated_at = now;
                self.state.update_goal(&goal).await?;
                for schedule in &schedules {
                    let _ = self.state.delete_goal_schedule(&schedule.id).await;
                }
                return Ok(None);
            }
        }
        goal.updated_at = now;
        self.state.update_goal(&goal).await?;
        Ok(None)
    }

    async fn change_probe(
        &self,
        automation: &Automation,
        action: &str,
    ) -> anyhow::Result<ChangeNote> {
        let Some(store) = &self.probes else {
            return Ok(None);
        };
        match action {
            "pause" => {
                store.pause_probe(&automation.id).await?;
            }
            "resume" => {
                let Some(probe) = store.get_probe(&automation.id).await? else {
                    return Ok(None);
                };
                let next = aidaemon_work::schedule::compute_next_run(&probe.schedule)?;
                store.resume_probe(&automation.id, next).await?;
            }
            _ => {
                if automation.from_config {
                    return Ok(Some(
                        "It comes from the config file, so it would come back at the next start; remove it from the config, or pause it."
                            .to_string(),
                    ));
                }
                store.delete_probe(&automation.id).await?;
            }
        }
        Ok(None)
    }

    async fn change(
        &self,
        automation: &Automation,
        action: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<(String, ToolOutcomeStatus)> {
        let note = match automation.kind {
            AutomationKind::Scheduled => self.change_scheduled(&automation.id, action).await?,
            AutomationKind::Probe => self.change_probe(automation, action).await?,
            AutomationKind::Watching => {
                let call = json!({ "action": action, "mandate_id": automation.id }).to_string();
                if let Err(error) = self.mandates.run(&call, facts).await {
                    tracing::debug!(%error, action, "automation change was refused");
                }
                None
            }
        };
        let wanted = match action {
            "pause" => "paused",
            "resume" => "active",
            _ => "cancelled",
        };
        let now = self.current_status(automation).await?;
        // A deleted probe is gone, which is what cancelling it means.
        let now = match (automation.kind, now) {
            (AutomationKind::Probe, None) => Some("cancelled".to_string()),
            (_, now) => now,
        };
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
        if let Some(note) = note {
            text.push(' ');
            text.push_str(&note);
        }
        Ok((text, ToolOutcomeStatus::CompletedWithNegativeResult))
    }

    /// Pause, resume or cancel the automation one call names, or each of the
    /// refs it lists. Choosing several is a separate read step (`list`):
    /// nothing is changed by matching words.
    async fn change_named(
        &self,
        args: &Args,
        action: &str,
        facts: &ToolCallContext,
        kind: Option<AutomationKind>,
    ) -> anyhow::Result<ToolCallOutcome> {
        let session_id = facts.session_id.as_str();
        let ids: Vec<&str> = args
            .ids
            .iter()
            .flatten()
            .map(|id| id.trim())
            .filter(|id| !id.is_empty())
            .collect();
        if ids.is_empty() {
            return Ok(
                match self
                    .select(given(&args.name), given(&args.id), session_id, kind)
                    .await?
                {
                    Ok(automation) => {
                        let (text, status) = self.change(&automation, action, facts).await?;
                        outcome(text, status)
                    }
                    Err(answer) => negative(answer),
                },
            );
        }
        let mut lines = Vec::new();
        let mut all_changed = true;
        for id in ids {
            match self.select(None, Some(id), session_id, kind).await? {
                Ok(automation) => {
                    let (text, status) = self.change(&automation, action, facts).await?;
                    all_changed &= status == ToolOutcomeStatus::Succeeded;
                    lines.push(text);
                }
                Err(_) => {
                    all_changed = false;
                    lines.push(format!("No automation has the ref {id}."));
                }
            }
        }
        Ok(outcome(
            lines.join("\n"),
            if all_changed {
                ToolOutcomeStatus::Succeeded
            } else {
                ToolOutcomeStatus::CompletedWithNegativeResult
            },
        ))
    }

    async fn edit_probe(
        &self,
        automation: &Automation,
        args: &Args,
    ) -> anyhow::Result<ToolCallOutcome> {
        if automation.from_config {
            return Ok(negative(
                "This health check comes from the config file, so it stays as configured; edit the config file to change it.".to_string(),
            ));
        }
        let Some(store) = &self.probes else {
            return Ok(negative("Health check storage is unavailable.".to_string()));
        };
        if given(&args.snooze_until).is_some() || given(&args.tell_me).is_some() {
            return Ok(negative(
                "Health checks can only change `when` or `new_name`.".to_string(),
            ));
        }
        let Some(probe) = store.get_probe(&automation.id).await? else {
            return Ok(negative("That health check is gone.".to_string()));
        };
        let name = given(&args.new_name).unwrap_or(&probe.name).to_string();
        let schedule = given(&args.when).unwrap_or(&probe.schedule).to_string();
        if name == probe.name && schedule == probe.schedule {
            return Ok(negative("Nothing changed.".to_string()));
        }
        if name != probe.name
            && store
                .list_probes()
                .await?
                .iter()
                .any(|other| other.id != probe.id && other.name.eq_ignore_ascii_case(&name))
        {
            return Ok(negative(format!(
                "A health check is already named \"{name}\"."
            )));
        }
        let next = if schedule == probe.schedule {
            probe.next_run_at
        } else {
            match aidaemon_work::schedule::parse_schedule(&schedule)
                .and_then(|cron| aidaemon_work::schedule::compute_next_run_local(&cron))
            {
                Ok(next) => next.with_timezone(&Utc),
                Err(error) => {
                    return Ok(negative(format!(
                        "Couldn't read the schedule '{schedule}': {error}"
                    )));
                }
            }
        };
        if !store.edit_probe(&probe.id, &name, &schedule, next).await? {
            return Ok(negative(
                "That health check is gone or comes from the config file.".to_string(),
            ));
        }
        let probe = store
            .get_probe(&probe.id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("edited health check disappeared"))?;
        if probe.name != name || probe.schedule != schedule || probe.next_run_at != next {
            return Ok(negative(
                "The health check edit could not be verified.".to_string(),
            ));
        }
        let mut edits = Vec::new();
        if name != automation.name.as_deref().unwrap_or_default() {
            edits.push(format!("renamed it \"{name}\""));
        }
        if schedule != automation.card.how_often && given(&args.when).is_some() {
            edits.push(format!("its schedule is now {schedule}"));
        }
        Ok(outcome(
            format!(
                "Edited {}: {}. Next run: {}.",
                automation.title(),
                edits.join("; "),
                aidaemon_work::schedule::owner_time(&probe.next_run_at)
            ),
            ToolOutcomeStatus::Succeeded,
        ))
    }

    /// Reschedule, rename or snooze one scheduled automation. Everything is
    /// checked before anything is written.
    async fn edit(
        &self,
        automation: &Automation,
        args: &Args,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        match automation.kind {
            AutomationKind::Scheduled => {}
            AutomationKind::Watching => {
                return Ok(negative(
                    "A watching automation's limits and cadence change through manage_mandates; pause, resume or cancel it here.".to_string(),
                ));
            }
            AutomationKind::Probe => return self.edit_probe(automation, args).await,
        }
        let when = given(&args.when);
        let snooze = given(&args.snooze_until);
        let new_name = given(&args.new_name);
        let tell_me = match given(&args.tell_me) {
            None => None,
            Some(raw) => match automations::NotifyMode::parse(raw) {
                Some(mode) => Some(mode),
                None => {
                    return Ok(negative(
                        "`tell_me` is every_run, changes or problems.".to_string(),
                    ));
                }
            },
        };
        if when.is_none() && snooze.is_none() && new_name.is_none() && tell_me.is_none() {
            return Ok(negative(
                "Say what to change: `when` (a new time), `new_name`, `snooze_until` or `tell_me`."
                    .to_string(),
            ));
        }
        if when.is_some() && snooze.is_some() {
            return Ok(negative(
                "Pass `when` to change its schedule or `snooze_until` to move only its next run, not both."
                    .to_string(),
            ));
        }
        let title = automation.title();
        let Some(mut goal) = self.state.get_goal(&automation.id).await? else {
            return Ok(negative(format!("{title} is gone.")));
        };
        let schedules = self.state.get_schedules_for_goal(&goal.id).await?;

        if let Some(new_name) = new_name {
            let same = automation
                .name
                .as_deref()
                .is_some_and(|own| own.eq_ignore_ascii_case(new_name));
            if !same
                && let Some(refusal) = crate::tools::schedule_task::name_refusal(
                    self.state.as_ref(),
                    &facts.session_id,
                    new_name,
                )
                .await?
            {
                return Ok(negative(refusal));
            }
        }
        let new_schedule = match when {
            None => None,
            Some(when) => {
                let aidaemon_work::schedule::ParsedSchedule { cron, tz } =
                    match aidaemon_work::schedule::parse_schedule_with_zone(when) {
                        Ok(parsed) => parsed,
                        Err(error) => {
                            return Ok(negative(format!(
                                "Couldn't read the time '{when}': {error}"
                            )));
                        }
                    };
                let next = match aidaemon_work::schedule::compute_next_run_in(&cron, &tz) {
                    Ok(next) => next.with_timezone(&chrono::Local),
                    Err(error) => {
                        return Ok(negative(format!(
                            "Couldn't compute the next run for '{when}': {error}"
                        )));
                    }
                };
                let one_shot = aidaemon_work::schedule::is_one_shot_schedule(&cron);
                let was_one_shot = !schedules.is_empty() && schedules.iter().all(|s| s.is_one_shot);
                if !schedules.is_empty() && one_shot != was_one_shot {
                    let (from, to) = if was_one_shot {
                        ("runs once", "a recurring one")
                    } else {
                        ("recurs", "a one-time one")
                    };
                    return Ok(negative(format!(
                        "{title} {from}; to make it {to}, create it again with schedule_task or set_reminder and cancel this one."
                    )));
                }
                Some((when, cron, tz, next, one_shot))
            }
        };
        let snooze_to = match snooze {
            None => None,
            Some(snooze) => {
                let cron = match aidaemon_work::schedule::parse_schedule(snooze) {
                    Ok(cron) => cron,
                    Err(error) => {
                        return Ok(negative(format!(
                            "Couldn't read the time '{snooze}': {error}"
                        )));
                    }
                };
                if !aidaemon_work::schedule::is_one_shot_schedule(&cron) {
                    return Ok(negative(
                        "`snooze_until` is one time, e.g. 'tomorrow at 9am'; to change how often it runs, pass `when`."
                            .to_string(),
                    ));
                }
                if schedules.is_empty() {
                    return Ok(negative(format!("{title} has no schedule to snooze.")));
                }
                match aidaemon_work::schedule::compute_next_run_local(&cron) {
                    Ok(at) => Some(at),
                    Err(error) => {
                        return Ok(negative(format!(
                            "Couldn't compute the time '{snooze}': {error}"
                        )));
                    }
                }
            }
        };

        if tell_me.is_some()
            && aidaemon_work::reminders::message_from_context(goal.context.as_deref()).is_some()
        {
            return Ok(negative(format!(
                "{title} is a reminder: it always tells you; there is nothing to quiet."
            )));
        }
        let now = Utc::now().to_rfc3339();
        let mut changed = Vec::new();
        if let Some(mode) = tell_me {
            let mut context = goal
                .context
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({}));
            context[automations::NotifyMode::KEY] = json!(mode.as_str());
            goal.context = Some(context.to_string());
            goal.updated_at = now.clone();
            self.state.update_goal(&goal).await?;
            changed.push(format!("it now tells you {}", mode.words()));
        }
        if let Some(new_name) = new_name {
            goal.context = Some(automations::context_with_name(
                goal.context.as_deref(),
                new_name,
            ));
            goal.updated_at = now.clone();
            self.state.update_goal(&goal).await?;
            changed.push(format!("renamed it \"{new_name}\""));
        }
        if let Some((when, cron, tz, next, one_shot)) = new_schedule {
            let fire_policy = schedules
                .first()
                .map(|schedule| schedule.fire_policy.clone())
                .unwrap_or_else(|| "coalesce".to_string());
            for schedule in &schedules {
                self.state.delete_goal_schedule(&schedule.id).await?;
            }
            self.state
                .create_goal_schedule(&crate::tools::traits::GoalSchedule {
                    id: uuid::Uuid::new_v4().to_string(),
                    goal_id: goal.id.clone(),
                    cron_expr: cron.clone(),
                    tz: tz.clone(),
                    original_schedule: Some(when.to_string()),
                    fire_policy,
                    is_one_shot: one_shot,
                    is_paused: false,
                    last_run_at: None,
                    next_run_at: next.with_timezone(&Utc).to_rfc3339(),
                    created_at: now.clone(),
                    updated_at: now.clone(),
                })
                .await?;
            let spec = ScheduleSpec {
                cron,
                zone: zone_label(&tz),
                one_shot,
                next: Some(aidaemon_work::schedule::owner_time(&next)),
                following: Vec::new(),
            };
            changed.push(format!(
                "its schedule is now {}",
                automations::cadence_words(
                    &spec,
                    &aidaemon_work::schedule::system_timezone_display()
                )
            ));
        }
        if let Some(at) = snooze_to {
            for mut schedule in schedules.into_iter().filter(|s| !s.is_paused) {
                schedule.next_run_at = at.with_timezone(&Utc).to_rfc3339();
                schedule.updated_at = now.clone();
                self.state.update_goal_schedule(&schedule).await?;
            }
            changed.push(format!(
                "its next run moved to {}; after that it follows its schedule",
                aidaemon_work::schedule::owner_time(&at)
            ));
        }

        let refreshed = self
            .scheduled()
            .await?
            .into_iter()
            .find(|candidate| candidate.id == automation.id);
        let mut text = format!("Edited {title}: {}.", changed.join("; "));
        if let Some(next) = refreshed.as_ref().and_then(Automation::next_words) {
            text.push_str(&format!(" Next run: {next}."));
        } else if automation.status != "active" {
            text.push_str(&format!(
                " It is {}.",
                automations::state_words(&automation.status)
            ));
        }
        Ok(outcome(text, ToolOutcomeStatus::Succeeded))
    }

    /// The scheduled goal a run or settings call names: an automation's
    /// name, or a ref, id or distinctive objective phrase (any scheduled
    /// goal, as these actions always reached).
    async fn scheduled_target(
        &self,
        args: &Args,
        session_id: &str,
    ) -> anyhow::Result<Result<String, String>> {
        if let Some(id) = given(&args.id) {
            return Ok(self
                .runs
                .resolve_goal_id(id)
                .await
                .map_err(|e| e.to_string()));
        }
        Ok(self
            .select(
                given(&args.name),
                None,
                session_id,
                Some(AutomationKind::Scheduled),
            )
            .await?
            .map(|automation| automation.id))
    }

    /// The settings a call can change, to tell a change from a refusal.
    async fn settings_of(&self, goal_id: &str) -> anyhow::Result<Option<Value>> {
        Ok(self.state.get_goal(goal_id).await?.map(|goal| {
            json!([
                goal.budget_per_check,
                goal.budget_daily,
                goal.description,
                goal.context
            ])
        }))
    }

    /// What a run now can change: the goal's tasks and its status.
    async fn run_state(&self, goal_id: &str) -> anyhow::Result<(Vec<String>, Option<String>)> {
        let mut tasks: Vec<String> = self
            .state
            .get_tasks_for_goal(goal_id)
            .await?
            .into_iter()
            .map(|task| task.id)
            .collect();
        tasks.sort();
        let status = self.state.get_goal(goal_id).await?.map(|goal| goal.status);
        Ok((tasks, status))
    }

    /// Run now, history, last failure and the goal settings.
    async fn on_scheduled_goal(
        &self,
        args: &Args,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let goal_id = match self.scheduled_target(args, &facts.session_id).await? {
            Ok(goal_id) => goal_id,
            Err(answer) => return Ok(negative(answer)),
        };
        let action = args.action.as_str();
        let before = self.settings_of(&goal_id).await?;
        let run_before = self.run_state(&goal_id).await?;
        let output = match action {
            "run_now" => self.runs.run_now(&goal_id, None).await?,
            "history" => {
                self.runs
                    .run_history(&goal_id, args.limit.unwrap_or(10))
                    .await?
            }
            "last_failure" => self.runs.last_failure(&goal_id).await?,
            "set_budget" => {
                self.runs
                    .set_budget(&goal_id, args.budget_per_check, args.budget_daily)
                    .await?
            }
            "update_instructions" => {
                let Some(instructions) = given(&args.instructions) else {
                    return Ok(negative(
                        "Pass the new `instructions` for its runs.".to_string(),
                    ));
                };
                self.runs
                    .update_instructions(
                        &goal_id,
                        instructions,
                        given(&args.run_requirement),
                        given(&args.insufficient_evidence),
                    )
                    .await?
            }
            _ => {
                let Some(path) = given(&args.workspace_path) else {
                    return Ok(negative(
                        "Pass the project directory as `workspace_path`.".to_string(),
                    ));
                };
                self.runs.bind_workspace(&goal_id, path).await?
            }
        };
        let status = match action {
            "history" | "last_failure" => ToolOutcomeStatus::Succeeded,
            // A run started (a new task), or a stalled one was taken up
            // again (the goal is on again). Keeping an open run, a budget
            // stop or a paused goal starts nothing.
            "run_now" => {
                let (tasks, status) = self.run_state(&goal_id).await?;
                let started = tasks.iter().any(|task| !run_before.0.contains(task));
                let reactivated = status.as_deref() == Some("active")
                    && run_before.1.as_deref() != Some("active");
                if started || reactivated {
                    ToolOutcomeStatus::Succeeded
                } else {
                    ToolOutcomeStatus::CompletedWithNegativeResult
                }
            }
            _ if self.settings_of(&goal_id).await? != before => ToolOutcomeStatus::Succeeded,
            _ => ToolOutcomeStatus::CompletedWithNegativeResult,
        };
        let presentation = matches!(action, "run_now" | "set_budget" | "update_instructions")
            .then_some(ToolResultPresentation::NaturalSummary);
        let mut internal_identifiers = aidaemon_tools::sanitize::extract_uuid_identifiers(&output);
        internal_identifiers.push(goal_id);
        internal_identifiers.sort();
        internal_identifiers.dedup();
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(status),
                presentation,
                internal_identifiers,
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }

    async fn run(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Args = serde_json::from_str(arguments)?;
        let action = args.action.as_str();
        if !ACTIONS.contains(&action) {
            anyhow::bail!("unknown manage_automations action `{action}`");
        }
        if !allowed(gate(action), facts) {
            return Ok(outcome(OWNER_ONLY.to_string(), ToolOutcomeStatus::Blocked));
        }
        let kind = match given(&args.kind) {
            None => None,
            Some(raw) => match AutomationKind::parse(raw) {
                Some(kind) => Some(kind),
                None => anyhow::bail!("kind must be scheduled, watching or probe"),
            },
        };
        let session_id = facts.session_id.as_str();
        match action {
            "list" => {
                let all = self.all(session_id, kind).await?;
                let mut listed =
                    outcome(Self::render_list(&all, kind), ToolOutcomeStatus::Succeeded);
                // The scheduled collection was read in full: record it as
                // every owner surface does.
                if kind.is_none_or(|kind| kind == AutomationKind::Scheduled) {
                    let (_, portfolio) = self.runs.overview_snapshot(500, false).await?;
                    let (observations, collections) = portfolio.receipt_evidence();
                    listed = listed.with_observation_evidence(observations, collections);
                }
                Ok(listed)
            }
            "show" => Ok(
                match self
                    .select(given(&args.name), given(&args.id), session_id, kind)
                    .await?
                {
                    Ok(automation) => {
                        outcome(automation.render_card(), ToolOutcomeStatus::Succeeded)
                    }
                    Err(answer) => negative(answer),
                },
            ),
            "edit" => Ok(
                match self
                    .select(given(&args.name), given(&args.id), session_id, kind)
                    .await?
                {
                    Ok(automation) => self.edit(&automation, &args, facts).await?,
                    Err(answer) => negative(answer),
                },
            ),
            "pause" | "resume" | "cancel" => self.change_named(&args, action, facts, kind).await,
            "history" => {
                let automation = match self
                    .select(given(&args.name), given(&args.id), session_id, kind)
                    .await?
                {
                    Ok(automation) => automation,
                    // A goal the list does not show (one with no schedule
                    // yet) still has its runs, found by its id as before.
                    Err(_) if kind.is_none_or(|kind| kind == AutomationKind::Scheduled) => {
                        return self.on_scheduled_goal(&args, facts).await;
                    }
                    Err(answer) => return Ok(negative(answer)),
                };
                let output = self.history(&automation, args.limit.unwrap_or(10)).await?;
                Ok(outcome(output, ToolOutcomeStatus::Succeeded))
            }
            _ => {
                if kind.is_some_and(|kind| kind != AutomationKind::Scheduled) {
                    return Ok(negative(format!(
                        "`{action}` is for scheduled work. A health check runs now with health_probe run_now; a watching one through manage_mandates."
                    )));
                }
                self.on_scheduled_goal(&args, facts).await
            }
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

fn is_read(action: Option<&str>) -> bool {
    matches!(action, Some("list" | "show" | "history" | "last_failure"))
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
        let text = |description: &str| json!({ "type": "string", "description": description });
        let integer = |description: &str| json!({ "type": "integer", "description": description });
        json!({
            "name": "manage_automations",
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ACTIONS },
                    "name": text("Its name, as the user gave it."),
                    "id": text("Its ref from the list."),
                    "ids": { "type": "array", "items": { "type": "string" }, "description": "pause/resume/cancel: several refs." },
                    "kind": { "type": "string", "enum": ["scheduled", "watching", "probe"], "description": "Narrows the list or picks between equal names." },
                    "when": text("edit: new schedule."),
                    "snooze_until": text("edit: move only the next run to this time."),
                    "new_name": text("edit: new name."),
                    "tell_me": { "type": "string", "enum": ["every_run", "changes", "problems"], "description": "edit, scheduled work: what its runs tell the user: every run, only changes, or only problems." },
                    "limit": integer("history: runs or health-check results to show."),
                    "budget_per_check": integer("Tokens per run."),
                    "budget_daily": integer("Tokens per day."),
                    "instructions": text("What later runs do."),
                    "run_requirement": { "type": "string", "enum": ["must_complete", "best_effort"], "description": "Must a run finish." },
                    "insufficient_evidence": { "type": "string", "enum": ["use_public_sources", "use_best_available", "skip_run"], "description": "When evidence is short." },
                    "workspace_path": text("Absolute project directory for its runs.")
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

impl crate::tools::traits::ToolSemantics for ManageAutomationsTool {
    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            offering: aidaemon_tools::semantics::Offering::Core,
            evidence: aidaemon_tools::semantics::EvidenceSource::GoalRecords,
            semantic_scope: Some(aidaemon_tools::semantics::ToolSemanticScope::GoalState),
            ..Default::default()
        }
    }

    fn stable_observation_subjects(&self) -> Vec<crate::tools::traits::StableObservationSubject> {
        vec![
            aidaemon_work::tools::objective_status::ObjectiveCollection::ScheduledGoals
                .stable_subject(),
        ]
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let action = action_of(arguments);
        match action.as_deref() {
            action if is_read(action) => ToolCallSemantics::observation(),
            // A workspace binding is a trusted root for every later run.
            Some("bind_workspace") => {
                ToolCallSemantics::mutation_with(ToolMutationEffects::CONFIGURATION)
            }
            Some("run_now") => {
                ToolCallSemantics::mutation_with(ToolMutationEffects::SCHEDULED_WORK)
            }
            _ => ToolCallSemantics::mutation_with(
                ToolMutationEffects::SCHEDULED_WORK.union(ToolMutationEffects::CONFIGURATION),
            ),
        }
    }

    /// Reads have no effects. Every change alters what runs on its own, so
    /// it stays undeclared.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        if is_read(action_of(arguments).as_deref()) {
            aidaemon_core::CallEffects::none()
        } else {
            aidaemon_core::CallEffects::unknown()
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

/// A stored schedule zone as the owner reads it beside a clock time;
/// `None` for the owner's own zone (or one this cannot read).
fn zone_label(tz: &str) -> Option<String> {
    aidaemon_work::schedule::ScheduleZone::from_stored(tz)
        .ok()
        .and_then(|zone| zone.label())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_work::goals::{GoalRun, ScheduledRecoveryDisposition, ScheduledRecoveryState};

    fn recovery(last_failed_run_id: Option<&str>) -> ScheduledRecoveryState {
        ScheduledRecoveryState {
            goal_id: "goal-1".to_string(),
            consecutive_failures: 4,
            failure_budget: 3,
            disposition: ScheduledRecoveryDisposition::Escalated,
            latest_failure_kind: None,
            last_failed_run_id: last_failed_run_id.map(str::to_string),
            last_recovery_run_id: None,
            recovery_attempts: 0,
            last_recovery_attempt_at: None,
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn failed_run(id: &str, summary: &str) -> GoalRun {
        let mut run = GoalRun::new("goal-1", "project-1", "scheduled");
        run.id = id.to_string();
        run.status = "failed".to_string();
        run.outcome_summary = Some(summary.to_string());
        run
    }

    fn held_automation(held: HeldAfterFailures) -> Automation {
        Automation {
            kind: AutomationKind::Scheduled,
            id: "scheduled-goal-1".to_string(),
            name: Some("Repo check".to_string()),
            status: automations::HELD_AFTER_FAILURES.to_string(),
            card: MandateConfirmationCard {
                what: "Check the synthetic repository".to_string(),
                how_often: "Every day".to_string(),
                can_do: "Read only".to_string(),
                tells_you_when: "a run fails".to_string(),
                ends: "Only when you stop it".to_string(),
                details: Vec::new(),
            },
            next: None,
            from_config: false,
            edit: None,
            held_after_failures: Some(held),
        }
    }

    #[test]
    fn held_failure_uses_only_the_ledger_named_run_summary() {
        let runs = vec![
            failed_run("newer-run", "A different failure."),
            failed_run("ledger-run", "The synthetic token expired."),
        ];

        let held = held_after_failures(&recovery(Some("ledger-run")), &runs).unwrap();
        assert_eq!(held.consecutive_failures, 4);
        assert_eq!(
            held.last_failure_reason.as_deref(),
            Some("The synthetic token expired.")
        );

        let held = held_after_failures(&recovery(Some("missing-run")), &runs).unwrap();
        assert_eq!(held.last_failure_reason, None);
    }

    #[test]
    fn held_failure_omits_blank_run_summary_but_keeps_count() {
        let runs = vec![failed_run("ledger-run", "   ")];

        let held = held_after_failures(&recovery(Some("ledger-run")), &runs).unwrap();
        assert_eq!(held.consecutive_failures, 4);
        assert_eq!(held.last_failure_reason, None);

        let automation = held_automation(held);
        let shown = automation.render_card();
        assert!(shown.contains("Consecutive failures: 4"), "{shown}");
        assert!(!shown.contains("Last failure:"), "{shown}");
    }

    #[test]
    fn held_failure_count_and_reason_render_in_list_and_show() {
        let automation = held_automation(HeldAfterFailures {
            consecutive_failures: 4,
            last_failure_reason: Some("The synthetic token expired.".to_string()),
        });

        let list = ManageAutomationsTool::render_list(
            std::slice::from_ref(&automation),
            Some(AutomationKind::Scheduled),
        );
        assert!(list.contains("4 consecutive failures"), "{list}");
        assert!(
            list.contains("last failure: The synthetic token expired."),
            "{list}"
        );

        let shown = automation.render_card();
        assert!(shown.contains("Consecutive failures: 4"), "{shown}");
        assert!(
            shown.contains("Last failure: The synthetic token expired."),
            "{shown}"
        );
    }
}
