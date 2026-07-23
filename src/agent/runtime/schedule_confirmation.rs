//! Ephemeral schedule proposals and post-confirmation persistence.
//!
//! A proposed schedule is intentionally not written to durable state until the
//! owner confirms the current proposal. This prevents an intent-classification
//! heuristic from creating durable side effects on its own.

use super::Agent;
use crate::traits::{Goal, GoalSchedule};
use chrono::{Duration, Utc};

const PROPOSAL_TTL_MINUTES: i64 = 30;

#[derive(Debug, Clone)]
pub(crate) struct PendingScheduleProposal {
    pub(crate) goals_and_schedules: Vec<(Goal, GoalSchedule)>,
    created_at: chrono::DateTime<Utc>,
}

impl PendingScheduleProposal {
    pub(crate) fn new(goals_and_schedules: Vec<(Goal, GoalSchedule)>) -> Self {
        Self {
            goals_and_schedules,
            created_at: Utc::now(),
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at < Utc::now() - Duration::minutes(PROPOSAL_TTL_MINUTES)
    }
}

#[derive(Debug, Default)]
pub(crate) struct ScheduleActivationReport {
    pub(crate) activated: Vec<String>,
    pub(crate) errors: Vec<String>,
}

pub(crate) async fn store_pending_schedule_proposal(
    agent: &Agent,
    session_id: &str,
    goals_and_schedules: Vec<(Goal, GoalSchedule)>,
) {
    agent.pending_schedule_proposals.write().await.insert(
        session_id.to_string(),
        PendingScheduleProposal::new(goals_and_schedules),
    );
}

pub(crate) async fn take_pending_schedule_proposal(
    agent: &Agent,
    session_id: &str,
) -> Option<PendingScheduleProposal> {
    let proposal = agent
        .pending_schedule_proposals
        .write()
        .await
        .remove(session_id)?;
    (!proposal.is_expired()).then_some(proposal)
}

pub(crate) async fn has_pending_schedule_proposal(agent: &Agent, session_id: &str) -> bool {
    let mut proposals = agent.pending_schedule_proposals.write().await;
    let expired = proposals
        .get(session_id)
        .is_some_and(PendingScheduleProposal::is_expired);
    if expired {
        proposals.remove(session_id);
        return false;
    }
    proposals.contains_key(session_id)
}

pub(crate) async fn discard_pending_schedule_proposal(agent: &Agent, session_id: &str) -> bool {
    agent
        .pending_schedule_proposals
        .write()
        .await
        .remove(session_id)
        .is_some()
}

/// Persist and activate a confirmed proposal.
///
/// Persistence happens only here, after confirmation. Failures are isolated per
/// schedule; a goal whose schedule cannot be written is immediately cancelled.
pub(crate) async fn persist_and_activate_schedule_proposal(
    agent: &Agent,
    proposal: &PendingScheduleProposal,
) -> ScheduleActivationReport {
    let mut report = ScheduleActivationReport::default();

    for (goal, schedule) in &proposal.goals_and_schedules {
        if let Err(error) = agent.state.create_goal(goal).await {
            report.errors.push(format!(
                "{}: could not create goal: {error}",
                crate::tools::sanitize::short_goal_label(&goal.description)
            ));
            continue;
        }

        if let Err(error) = agent.state.create_goal_schedule(schedule).await {
            let mut cancelled = goal.clone();
            let now = Utc::now().to_rfc3339();
            cancelled.status = "cancelled".to_string();
            cancelled.completed_at = Some(now.clone());
            cancelled.updated_at = now;
            let _ = agent.state.update_goal(&cancelled).await;
            report.errors.push(format!(
                "{}: could not create schedule: {error}",
                crate::tools::sanitize::short_goal_label(&goal.description)
            ));
            continue;
        }

        match agent.state.activate_goal(&goal.id).await {
            Ok(true) => {
                if let Some(ref registry) = agent.goal_token_registry {
                    registry.register(&goal.id).await;
                }
                let next_run = chrono::DateTime::parse_from_rfc3339(&schedule.next_run_at)
                    .ok()
                    .map(|dt| {
                        crate::cron_utils::humanize_run_time(dt.with_timezone(&chrono::Local))
                    })
                    .unwrap_or_else(|| "n/a".to_string());
                report.activated.push(format!(
                    "{} (next run {})",
                    crate::tools::sanitize::short_goal_label(&goal.description),
                    next_run
                ));
            }
            Ok(false) => report.errors.push(format!(
                "{}: goal was not pending confirmation",
                crate::tools::sanitize::short_goal_label(&goal.description)
            )),
            Err(error) => report.errors.push(format!(
                "{}: could not activate goal: {error}",
                crate::tools::sanitize::short_goal_label(&goal.description)
            )),
        }
    }

    report
}

pub(crate) fn schedule_activation_message(report: &ScheduleActivationReport) -> String {
    if !report.activated.is_empty() && report.errors.is_empty() {
        if report.activated.len() == 1 {
            format!("✅ Scheduled: {}.", report.activated[0])
        } else {
            format!(
                "✅ Scheduled {} goals:\n- {}",
                report.activated.len(),
                report.activated.join("\n- ")
            )
        }
    } else if !report.activated.is_empty() {
        format!(
            "Scheduled {} goals:\n- {}\nBut {} could not be activated: {}",
            report.activated.len(),
            report.activated.join("\n- "),
            report.errors.len(),
            report.errors.join("; ")
        )
    } else {
        format!(
            "I couldn't activate scheduled goals: {}",
            report.errors.join("; ")
        )
    }
}
