//! One authoritative lifecycle readout for goal-backed objectives.
//!
//! An objective's health lives in typed durable state (schedules, goal runs,
//! recovery disposition, and mandate control). The scheduled-goal and mandate
//! collections are different universes, so a shared projection must preserve
//! both the exact goal identity and which collection supplied each row. A
//! complete collection may prove absence only inside that collection; a
//! truncated or unqueried collection may not.

use std::collections::{BTreeMap, BTreeSet};

/// Durable collections that can contribute an objective row. These names are
/// deliberately about source membership, not about what a row happens to say:
/// a mandate-controller goal with no cron row is still a mandate controller,
/// and does not prove that some unrelated scheduled objective is absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ObjectiveCollection {
    ScheduledGoals,
    MandateControllers,
}

impl ObjectiveCollection {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ScheduledGoals => "scheduled_goals",
            Self::MandateControllers => "mandate_controllers",
        }
    }

    pub(crate) const fn resource_id(self) -> &'static str {
        match self {
            Self::ScheduledGoals => "objective_collection:scheduled_goals",
            Self::MandateControllers => "objective_collection:mandate_controllers",
        }
    }

    fn target_hint(self) -> crate::traits::ToolTargetHint {
        crate::traits::ToolTargetHint::new(
            crate::traits::ToolTargetHintKind::ResourceId,
            self.resource_id(),
        )
        .expect("objective collection IDs are nonempty constants")
    }

    /// What an adapter enumerating this collection advertises to the
    /// checklist: the collection ID, every objective facet, and the
    /// `objective:` member namespace `objective_resource_id` mints.
    pub(crate) fn stable_subject(self) -> crate::traits::StableObservationSubject {
        crate::traits::StableObservationSubject::collection(
            self.resource_id(),
            &OBJECTIVE_STATUS_FACETS,
            OBJECTIVE_MEMBER_NAMESPACE,
            match self {
                Self::ScheduledGoals => "every scheduled goal",
                Self::MandateControllers => "every mandate-controlled goal",
            },
        )
    }
}

pub(crate) const OBJECTIVE_MEMBER_NAMESPACE: &str = "objective:";

pub(crate) const OBJECTIVE_STATUS_FACETS: [crate::traits::ToolSemanticFacet; 6] = [
    crate::traits::ToolSemanticFacet::Schedule,
    crate::traits::ToolSemanticFacet::RunState,
    crate::traits::ToolSemanticFacet::Recovery,
    crate::traits::ToolSemanticFacet::Control,
    crate::traits::ToolSemanticFacet::Measurement,
    crate::traits::ToolSemanticFacet::Ownership,
];

pub(crate) fn objective_resource_id(goal_id: &str) -> String {
    use sha2::{Digest, Sha256};

    // The checklist needs a stable exact identity, while owner-facing tool
    // output must not expose internal database UUIDs. A full digest keeps the
    // mapping deterministic without relying on display prose or row order.
    format!(
        "{OBJECTIVE_MEMBER_NAMESPACE}sha256:{:x}",
        Sha256::digest(goal_id.as_bytes())
    )
}

/// Coverage for one collection enumeration. `complete` is derived rather than
/// supplied so callers cannot accidentally label a limited result authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObjectiveCollectionCoverage {
    pub collection: ObjectiveCollection,
    pub total: usize,
    pub returned: usize,
}

impl ObjectiveCollectionCoverage {
    pub(crate) fn new(
        collection: ObjectiveCollection,
        total: usize,
        returned: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            returned <= total,
            "objective collection {} returned {returned} rows from a total of {total}",
            collection.as_str()
        );
        Ok(Self {
            collection,
            total,
            returned,
        })
    }

    pub(crate) const fn is_complete(self) -> bool {
        self.returned == self.total
    }

    pub(crate) fn as_json(self) -> serde_json::Value {
        serde_json::json!({
            "name": self.collection.as_str(),
            "resource_id": self.collection.resource_id(),
            "complete": self.is_complete(),
            "total": self.total,
            "returned": self.returned,
            "absence_semantics": if self.is_complete() {
                "authoritative_within_collection"
            } else {
                "unknown_outside_returned_rows"
            },
        })
    }
}

/// Upper bound on scheduled rows any surface returns in one enumeration.
pub(crate) const SCHEDULED_COLLECTION_LIMIT: usize = 100;

/// Enumerate the scheduled-goal collection the same way on every owner
/// surface. The collection is daemon-wide: a goal's `session_id` records the
/// channel session that created it, not an authority boundary, and one owner
/// routinely reaches the daemon through several sessions (two Telegram bots,
/// a Slack DM). A surface that filters this collection by session while still
/// labeling its coverage `complete` turns a goal created through a sibling
/// owner session into an authoritative "absent" (R53). `total` therefore
/// counts every scheduled goal, so truncation stays honest as `partial`.
pub(crate) async fn load_scheduled_goal_collection<S>(
    state: &S,
    limit: usize,
) -> anyhow::Result<(Vec<crate::traits::Goal>, ObjectiveCollectionCoverage)>
where
    S: crate::traits::GoalScheduleStore + ?Sized,
{
    let mut goals = state.get_scheduled_goals().await?;
    goals.sort_by(|left, right| {
        let left_rank = usize::from(left.status != "active");
        let right_rank = usize::from(right.status != "active");
        left_rank
            .cmp(&right_rank)
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| left.id.cmp(&right.id))
    });
    let total = goals.len();
    goals.truncate(limit.clamp(1, SCHEDULED_COLLECTION_LIMIT));
    let coverage =
        ObjectiveCollectionCoverage::new(ObjectiveCollection::ScheduledGoals, total, goals.len())?;
    Ok((goals, coverage))
}

/// The only three defensible answers to collection membership. In particular,
/// not seeing a subject in a limited collection is `Unknown`, never `Absent`.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObjectiveCollectionMembership {
    Present,
    Absent,
    Unknown,
}

/// Subject-keyed row in the canonical objective portfolio. The stable goal ID
/// is retained even when a public renderer elects not to display internal IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObjectivePortfolioRow {
    pub goal_id: String,
    pub source_membership: BTreeSet<ObjectiveCollection>,
    pub status: ObjectiveStatus,
}

impl ObjectivePortfolioRow {
    pub(crate) fn subject_hint(&self) -> crate::traits::ToolTargetHint {
        crate::traits::ToolTargetHint::new(
            crate::traits::ToolTargetHintKind::ResourceId,
            objective_resource_id(&self.goal_id),
        )
        .expect("validated objective goal IDs are nonempty")
    }

    pub(crate) fn source_membership_json(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.source_membership
                .iter()
                .map(|collection| serde_json::Value::String(collection.as_str().to_string()))
                .collect(),
        )
    }

    pub(crate) fn source_membership_label(&self) -> String {
        self.source_membership
            .iter()
            .map(|collection| collection.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// A portfolio joins rows only by stable goal ID. It intentionally does not
/// attempt to infer that similar objective prose, account configuration, row
/// order, or one collection's first result refers to another collection's row.
#[derive(Debug, Default)]
pub(crate) struct ObjectivePortfolio {
    subjects: BTreeMap<String, ObjectivePortfolioRow>,
    coverage: BTreeMap<ObjectiveCollection, ObjectiveCollectionCoverage>,
}

impl ObjectivePortfolio {
    pub(crate) fn record_collection(
        &mut self,
        coverage: ObjectiveCollectionCoverage,
    ) -> anyhow::Result<()> {
        if let Some(existing) = self.coverage.get(&coverage.collection) {
            anyhow::ensure!(
                existing == &coverage,
                "objective collection {} was enumerated with conflicting coverage",
                coverage.collection.as_str()
            );
            return Ok(());
        }
        self.coverage.insert(coverage.collection, coverage);
        Ok(())
    }

    pub(crate) fn insert(&mut self, row: ObjectivePortfolioRow) -> anyhow::Result<()> {
        match self.subjects.get_mut(&row.goal_id) {
            Some(existing) => {
                anyhow::ensure!(
                    existing.status == row.status,
                    "objective subject {} was projected with conflicting lifecycle state",
                    row.goal_id
                );
                existing.source_membership.extend(row.source_membership);
            }
            None => {
                self.subjects.insert(row.goal_id.clone(), row);
            }
        }
        Ok(())
    }

    pub(crate) fn subject(&self, goal_id: &str) -> Option<&ObjectivePortfolioRow> {
        self.subjects.get(goal_id)
    }

    #[cfg(test)]
    pub(crate) fn subjects(&self) -> impl Iterator<Item = &ObjectivePortfolioRow> {
        self.subjects.values()
    }

    #[cfg(test)]
    pub(crate) fn membership(
        &self,
        goal_id: &str,
        collection: ObjectiveCollection,
    ) -> ObjectiveCollectionMembership {
        if self
            .subjects
            .get(goal_id)
            .is_some_and(|row| row.source_membership.contains(&collection))
        {
            return ObjectiveCollectionMembership::Present;
        }
        match self.coverage.get(&collection) {
            Some(coverage) if coverage.is_complete() => ObjectiveCollectionMembership::Absent,
            _ => ObjectiveCollectionMembership::Unknown,
        }
    }

    pub(crate) fn collection_scope_json(
        &self,
        primary: ObjectiveCollection,
    ) -> anyhow::Result<serde_json::Value> {
        let coverage = self.coverage.get(&primary).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "objective collection {} has no coverage record",
                primary.as_str()
            )
        })?;
        let not_enumerated = [
            ObjectiveCollection::ScheduledGoals,
            ObjectiveCollection::MandateControllers,
        ]
        .into_iter()
        .filter(|collection| !self.coverage.contains_key(collection))
        .map(|collection| serde_json::Value::String(collection.as_str().to_string()))
        .collect::<Vec<_>>();
        let mut value = coverage.as_json();
        value["not_enumerated"] = serde_json::Value::Array(not_enumerated);
        Ok(value)
    }

    /// Convert the canonical subject-keyed snapshot into durable receipt
    /// assertions. Renderers may reorder or truncate prose, but the adapter's
    /// exact subject/facet and collection coverage remain stable evidence.
    pub(crate) fn receipt_evidence(
        &self,
    ) -> (
        Vec<crate::traits::ToolObservationEvidence>,
        Vec<crate::traits::ToolCollectionObservation>,
    ) {
        let observations = self
            .subjects
            .values()
            .map(|row| {
                let mut observation = crate::traits::ToolObservationEvidence::new(
                    row.subject_hint(),
                    &OBJECTIVE_STATUS_FACETS,
                );
                for collection in &row.source_membership {
                    observation = observation.with_source_collection(collection.target_hint());
                }
                observation
            })
            .collect::<Vec<_>>();
        let collections = self
            .coverage
            .values()
            .map(|coverage| {
                let members = self
                    .subjects
                    .values()
                    .filter(|row| row.source_membership.contains(&coverage.collection))
                    .map(ObjectivePortfolioRow::subject_hint)
                    .collect::<Vec<_>>();
                crate::traits::ToolCollectionObservation {
                    collection: coverage.collection.target_hint(),
                    facets: OBJECTIVE_STATUS_FACETS.to_vec(),
                    completeness: if coverage.is_complete() {
                        crate::traits::ToolCollectionCompleteness::Complete
                    } else {
                        crate::traits::ToolCollectionCompleteness::Partial
                    },
                    returned_count: coverage.returned,
                    total_count: Some(coverage.total),
                    members,
                }
            })
            .collect();
        (observations, collections)
    }
}

/// Typed lifecycle snapshot for one goal-backed objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObjectiveStatus {
    /// "active", "paused", or "missing".
    pub schedule_state: &'static str,
    /// Earliest next occurrence among active schedules.
    pub next_run_at: Option<String>,
    /// Latest goal run: (status, finished-or-started timestamp, summary).
    pub latest_run: Option<(String, String, Option<String>)>,
    /// Recovery ledger: (disposition, consecutive_failures, failure_budget).
    pub recovery: Option<(&'static str, u16, u16)>,
    /// Terminal run history counts (bounded by the store's returned window).
    pub runs_completed: usize,
    pub runs_failed: usize,
    /// Delegated control-loop facet; `None` means no mandate governs this
    /// goal, which every surface must render as an explicit "control absent"
    /// rather than leaving the question unanswerable.
    pub control: Option<ObjectiveControlFacet>,
}

/// Typed delegation/control readout for one goal-backed objective, derived
/// from the goal's mandate row and its recorded objective measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObjectiveControlFacet {
    pub mandate_status: String,
    pub autonomy_mode: String,
    /// Owner-confirmed control-loop metric, when the mandate has one.
    pub metric_name: Option<String>,
    pub measurement_count: usize,
    /// Distinct `account:` / `auth_profile:` identities from the mandate's
    /// authority scopes: the typed answer to "which owned account and
    /// credentials is this objective delegated to?".
    pub delegated_identities: Vec<String>,
}

/// Project the goal's mandate row into the control facet. `None` in means no
/// mandate exists for the goal; measurement rows without a mandate cannot
/// exist, so the absent case always carries zero measurements.
pub(crate) fn objective_control_facet(
    mandate: Option<&crate::traits::Mandate>,
    measurement_count: usize,
) -> Option<ObjectiveControlFacet> {
    mandate.map(|mandate| {
        let mut delegated_identities = Vec::new();
        for scope in &mandate.authority.operation_scopes {
            for prefix in &scope.target_prefixes {
                if (prefix.starts_with("account:") || prefix.starts_with("auth_profile:"))
                    && !delegated_identities.contains(prefix)
                {
                    delegated_identities.push(prefix.clone());
                }
            }
        }
        ObjectiveControlFacet {
            mandate_status: mandate.status.to_string(),
            autonomy_mode: mandate.autonomy_mode.to_string(),
            metric_name: mandate
                .objective_control
                .as_ref()
                .map(|control| control.metric_name.clone()),
            measurement_count,
            delegated_identities,
        }
    })
}

/// The control facet as typed JSON for structured surfaces. The absent case
/// is the literal string "absent" rather than an object: R47 showed that an
/// audit reads the mere presence of a populated `objective_control` object as
/// "control exists", so absence must not be expressed through a shape that
/// exists. Present control is an object; absent control is unmistakably not.
pub(crate) fn objective_control_json(control: Option<&ObjectiveControlFacet>) -> serde_json::Value {
    match control {
        Some(facet) => serde_json::json!({
            "mandate_status": facet.mandate_status,
            "autonomy_mode": facet.autonomy_mode,
            "metric": facet.metric_name,
            "measurement_count": facet.measurement_count,
            "delegated_identities": facet.delegated_identities,
        }),
        None => serde_json::json!("absent"),
    }
}

/// Project one exact goal into a portfolio row. All typed inputs are checked
/// against the requested subject before any field is aggregated. This is the
/// central defense against projecting goal B's mandate or run state onto goal
/// A merely because the rows were adjacent or fetched by the same audit.
pub(crate) fn objective_portfolio_row(
    goal_id: &str,
    enumerated_from: ObjectiveCollection,
    schedules: &[crate::traits::GoalSchedule],
    runs: &[crate::traits::GoalRun],
    recovery: Option<&crate::traits::ScheduledRecoveryState>,
    mandate: Option<&crate::traits::Mandate>,
    measurement_count: usize,
) -> anyhow::Result<ObjectivePortfolioRow> {
    anyhow::ensure!(!goal_id.trim().is_empty(), "objective subject is empty");
    for schedule in schedules {
        anyhow::ensure!(
            schedule.goal_id == goal_id,
            "schedule {} belongs to goal {}, not objective subject {goal_id}",
            schedule.id,
            schedule.goal_id
        );
    }
    for run in runs {
        anyhow::ensure!(
            run.goal_id == goal_id,
            "goal run {} belongs to goal {}, not objective subject {goal_id}",
            run.id,
            run.goal_id
        );
    }
    if let Some(recovery) = recovery {
        anyhow::ensure!(
            recovery.goal_id == goal_id,
            "recovery state belongs to goal {}, not objective subject {goal_id}",
            recovery.goal_id
        );
    }
    if let Some(mandate) = mandate {
        anyhow::ensure!(
            mandate.goal_id == goal_id,
            "mandate {} belongs to goal {}, not objective subject {goal_id}",
            mandate.id,
            mandate.goal_id
        );
    } else {
        anyhow::ensure!(
            measurement_count == 0,
            "objective subject {goal_id} has measurements without a mandate"
        );
    }

    // Membership records which collection enumeration returned this subject,
    // not which facet rows happened to be joined while projecting it. A
    // scheduled audit may read a mandate facet without enumerating (or being
    // authorized to enumerate) the mandate collection; that cannot become a
    // collection-membership claim.
    let source_membership = BTreeSet::from([enumerated_from]);

    let active_count = schedules
        .iter()
        .filter(|schedule| !schedule.is_paused)
        .count();
    let schedule_state = if schedules.is_empty() {
        "missing"
    } else if active_count > 0 {
        "active"
    } else {
        "paused"
    };
    let next_run_at = schedules
        .iter()
        .filter(|schedule| !schedule.is_paused)
        .map(|schedule| schedule.next_run_at.clone())
        .min();
    let latest_run = runs.first().map(|run| {
        (
            run.status.clone(),
            run.completed_at
                .clone()
                .unwrap_or_else(|| run.started_at.clone()),
            run.outcome_summary.clone(),
        )
    });
    let runs_completed = runs.iter().filter(|run| run.status == "completed").count();
    let runs_failed = runs
        .iter()
        .filter(|run| matches!(run.status.as_str(), "failed" | "blocked" | "cancelled"))
        .count();
    let recovery = recovery.map(|recovery| {
        (
            recovery.disposition.as_str(),
            recovery.consecutive_failures,
            recovery.failure_budget,
        )
    });
    Ok(ObjectivePortfolioRow {
        goal_id: goal_id.to_string(),
        source_membership,
        status: ObjectiveStatus {
            schedule_state,
            next_run_at,
            latest_run,
            recovery,
            runs_completed,
            runs_failed,
            control: objective_control_facet(mandate, measurement_count),
        },
    })
}

/// Render the snapshot as a single plain-text line for list surfaces.
pub(crate) fn render_objective_status_line(status: &ObjectiveStatus) -> String {
    let mut line = format!("schedule {}", status.schedule_state);
    if let Some(next) = &status.next_run_at {
        line.push_str(&format!(" (next {next})"));
    }
    match &status.latest_run {
        Some((run_status, at, summary)) => {
            line.push_str(&format!("; last run {run_status} at {at}"));
            if let Some(summary) = summary {
                let summary = summary.trim();
                if !summary.is_empty() {
                    line.push_str(" — ");
                    line.push_str(&crate::utils::truncate_str(summary, 140));
                }
            }
        }
        None => line.push_str("; no runs recorded"),
    }
    if let Some((disposition, failures, budget)) = &status.recovery {
        line.push_str(&format!(
            "; recovery {disposition} ({failures}/{budget} failures)"
        ));
    }
    line.push_str(&format!(
        "; run history {} completed / {} failed",
        status.runs_completed, status.runs_failed
    ));
    match &status.control {
        Some(control) => {
            let metric = control
                .metric_name
                .as_deref()
                .map(|name| format!("metric {name}"))
                .unwrap_or_else(|| "no metric loop".to_string());
            let identities = if control.delegated_identities.is_empty() {
                "no delegated account".to_string()
            } else {
                format!("delegated to {}", control.delegated_identities.join(", "))
            };
            line.push_str(&format!(
                "; control {} ({}; {}; {}; {} measurements)",
                control.mandate_status,
                control.autonomy_mode,
                metric,
                identities,
                control.measurement_count
            ));
        }
        None => line.push_str(
            "; control absent (no mandate; no delegated account or credentials; 0 measurements)",
        ),
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::store_prelude::*;
    use crate::traits::{Goal, GoalSchedule};

    #[tokio::test]
    async fn objective_status_projects_schedule_run_and_recovery_from_typed_state() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: "2026-09-01T10:00:00+00:00".to_string(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        let run = state
            .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
            .await
            .unwrap();
        assert!(state
            .finish_goal_run(
                &run.id,
                "completed",
                Some("Published and verified the daily post.")
            )
            .await
            .unwrap());

        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let recovery = state.get_scheduled_recovery_state(&goal.id).await.unwrap();
        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::ScheduledGoals,
            &schedules,
            &runs,
            recovery.as_ref(),
            None,
            0,
        )
        .unwrap()
        .status;
        assert_eq!(status.schedule_state, "active");
        assert_eq!(
            status.next_run_at.as_deref(),
            Some("2026-09-01T10:00:00+00:00")
        );
        let (run_status, _, summary) = status.latest_run.as_ref().unwrap();
        assert_eq!(run_status, "completed");
        assert_eq!(
            summary.as_deref(),
            Some("Published and verified the daily post.")
        );
        assert_eq!(status.runs_completed, 1);
        assert_eq!(status.runs_failed, 0);

        let line = render_objective_status_line(&status);
        assert!(line.contains("schedule active (next 2026-09-01T10:00:00+00:00)"));
        assert!(line.contains("last run completed"));
        assert!(line.contains("Published and verified the daily post."));
        assert!(line.contains("run history 1 completed / 0 failed"));
        // Without a mandate the control question still has a typed answer:
        // explicitly absent with zero measurements, never silently missing.
        assert!(status.control.is_none());
        assert!(line.contains(
            "control absent (no mandate; no delegated account or credentials; 0 measurements)"
        ));
        assert_eq!(objective_control_json(None), serde_json::json!("absent"));
    }

    #[tokio::test]
    async fn objective_status_reports_paused_schedule_and_recovery_disposition() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: uuid::Uuid::new_v4().to_string(),
            goal_id: goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: now.clone(),
            created_at: now.clone(),
            updated_at: now.clone(),
        };
        state.create_goal_schedule(&schedule).await.unwrap();
        // Three failed runs escalate and pause through the recovery ledger.
        for _ in 0..3 {
            let run = state
                .start_goal_run(&goal.id, "scheduled", Some(&schedule.id), None)
                .await
                .unwrap();
            assert!(state
                .finish_goal_run(&run.id, "failed", Some("synthetic failure"))
                .await
                .unwrap());
        }
        let schedules = state.get_schedules_for_goal(&goal.id).await.unwrap();
        let runs = state.get_goal_runs(&goal.id).await.unwrap();
        let recovery = state.get_scheduled_recovery_state(&goal.id).await.unwrap();
        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::ScheduledGoals,
            &schedules,
            &runs,
            recovery.as_ref(),
            None,
            0,
        )
        .unwrap()
        .status;
        assert_eq!(status.schedule_state, "paused");
        assert_eq!(status.next_run_at, None);
        let (disposition, failures, _) = status.recovery.unwrap();
        assert_eq!(disposition, "escalated");
        assert_eq!(failures, 3);
        assert_eq!(status.runs_failed, 3);
        let line = render_objective_status_line(&status);
        assert!(line.contains("schedule paused"));
        assert!(line.contains("recovery escalated (3/3 failures)"));
    }

    #[tokio::test]
    async fn objective_status_projects_mandate_control_loop_and_measurements() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("publish synthetic report", "session-a", None, None);
        state.create_goal(&goal).await.unwrap();
        let authority = crate::traits::MandateAuthority::from_operation_scopes(
            true,
            serde_json::from_value(serde_json::json!([
                {
                    "tool": "http_request",
                    "operation": "POST",
                    "kind": "mutation",
                    "target_prefixes": [
                        "https://api.x.com/2/tweets",
                        "auth_profile:twitter",
                        "account:12345"
                    ],
                    "mutation_effects": ["remote_mutation", "external_delivery"]
                }
            ]))
            .unwrap(),
            1,
            4,
            900,
        );
        let mut mandate = crate::traits::Mandate::new(
            &goal.id,
            None,
            "publish synthetic report",
            "session-a",
            authority,
            3_600,
            21_600,
            10_800,
        );
        mandate.objective_control = Some(crate::traits::MandateObjectiveControl {
            schema_version: crate::traits::MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "daily_visits".to_string(),
            unit: "visits".to_string(),
            baseline_micros: 0,
            target_micros: 1_000_000,
            direction: crate::traits::ObjectiveMetricDirection::AtLeast,
            measurement_source: "https://analytics.example.com/api".to_string(),
            measurement_cadence_secs: 7_200,
            experiment_cohort: "cohort-a".to_string(),
            experiment_window_secs: 604_800,
            minimum_effect_micros: 1,
            max_stagnant_measurements: 5,
            run_failure_budget: 3,
            baseline_observed_at: chrono::Utc::now().to_rfc3339(),
        });

        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::MandateControllers,
            &[],
            &[],
            None,
            Some(&mandate),
            3,
        )
        .unwrap()
        .status;
        let control = status.control.as_ref().unwrap();
        assert_eq!(control.mandate_status, "active");
        assert_eq!(control.autonomy_mode, "bounded");
        assert_eq!(control.metric_name.as_deref(), Some("daily_visits"));
        assert_eq!(control.measurement_count, 3);
        assert_eq!(
            control.delegated_identities,
            vec![
                "auth_profile:twitter".to_string(),
                "account:12345".to_string()
            ]
        );

        let line = render_objective_status_line(&status);
        assert!(
            line.contains(
                "control active (bounded; metric daily_visits; delegated to auth_profile:twitter, account:12345; 3 measurements)"
            ),
            "{line}"
        );
        let json = objective_control_json(status.control.as_ref());
        assert_eq!(json["mandate_status"], "active");
        assert_eq!(json["metric"], "daily_visits");
        assert_eq!(json["measurement_count"], 3);
        assert_eq!(
            json["delegated_identities"],
            serde_json::json!(["auth_profile:twitter", "account:12345"])
        );

        // A mandate without an owner-confirmed metric loop still projects a
        // typed answer rather than an unknown.
        mandate.objective_control = None;
        mandate.authority.operation_scopes.clear();
        let status = objective_portfolio_row(
            &goal.id,
            ObjectiveCollection::MandateControllers,
            &[],
            &[],
            None,
            Some(&mandate),
            0,
        )
        .unwrap()
        .status;
        let line = render_objective_status_line(&status);
        assert!(
            line.contains(
                "control active (bounded; no metric loop; no delegated account; 0 measurements)"
            ),
            "{line}"
        );
    }

    #[test]
    fn portfolio_keeps_unrelated_collection_subjects_distinct_and_scopes_absence() {
        let mut scheduled_goal =
            Goal::new_continuous("Publish the Acme digest", "owner-session", None, None);
        scheduled_goal.id = "goal-scheduled-a".to_string();
        let mut controller_goal =
            Goal::new_continuous("Review Acme engagement", "owner-session", None, None);
        controller_goal.id = "goal-controller-b".to_string();
        let now = chrono::Utc::now().to_rfc3339();
        let schedule = GoalSchedule {
            id: "schedule-a".to_string(),
            goal_id: scheduled_goal.id.clone(),
            cron_expr: "0 6 * * *".to_string(),
            tz: "local".to_string(),
            original_schedule: Some("daily".to_string()),
            fire_policy: "coalesce".to_string(),
            is_one_shot: false,
            is_paused: false,
            last_run_at: None,
            next_run_at: "2026-09-02T10:00:00+00:00".to_string(),
            created_at: now.clone(),
            updated_at: now,
        };
        let mandate = crate::traits::Mandate::new(
            &controller_goal.id,
            None,
            "Review Acme engagement",
            "owner-session",
            crate::traits::MandateAuthority::default(),
            3_600,
            21_600,
            10_800,
        );

        let scheduled_row = objective_portfolio_row(
            &scheduled_goal.id,
            ObjectiveCollection::ScheduledGoals,
            std::slice::from_ref(&schedule),
            &[],
            None,
            None,
            0,
        )
        .unwrap();
        let controller_row = objective_portfolio_row(
            &controller_goal.id,
            ObjectiveCollection::MandateControllers,
            &[],
            &[],
            None,
            Some(&mandate),
            1,
        )
        .unwrap();

        // Insert in the reverse of the collection/subject order. Identity,
        // never position, controls the merge.
        let mut portfolio = ObjectivePortfolio::default();
        portfolio
            .record_collection(
                ObjectiveCollectionCoverage::new(ObjectiveCollection::ScheduledGoals, 1, 1)
                    .unwrap(),
            )
            .unwrap();
        portfolio
            .record_collection(
                ObjectiveCollectionCoverage::new(ObjectiveCollection::MandateControllers, 1, 1)
                    .unwrap(),
            )
            .unwrap();
        portfolio.insert(controller_row.clone()).unwrap();
        portfolio.insert(scheduled_row.clone()).unwrap();

        assert_eq!(portfolio.subjects().count(), 2);
        let scheduled = portfolio.subject(&scheduled_goal.id).unwrap();
        assert_eq!(scheduled.status.schedule_state, "active");
        assert!(scheduled.status.control.is_none());
        let controller = portfolio.subject(&controller_goal.id).unwrap();
        assert_eq!(controller.status.schedule_state, "missing");
        assert_eq!(
            controller
                .status
                .control
                .as_ref()
                .map(|control| control.measurement_count),
            Some(1)
        );
        assert_eq!(
            portfolio.membership(&scheduled_goal.id, ObjectiveCollection::ScheduledGoals),
            ObjectiveCollectionMembership::Present
        );
        assert_eq!(
            portfolio.membership(&scheduled_goal.id, ObjectiveCollection::MandateControllers),
            ObjectiveCollectionMembership::Absent
        );
        assert_eq!(
            portfolio.membership(&controller_goal.id, ObjectiveCollection::ScheduledGoals),
            ObjectiveCollectionMembership::Absent
        );
        assert_eq!(
            portfolio.membership(&controller_goal.id, ObjectiveCollection::MandateControllers),
            ObjectiveCollectionMembership::Present
        );

        // A limit makes non-membership unknown, and never global absence.
        let mut limited = ObjectivePortfolio::default();
        limited
            .record_collection(
                ObjectiveCollectionCoverage::new(ObjectiveCollection::ScheduledGoals, 2, 1)
                    .unwrap(),
            )
            .unwrap();
        limited.insert(scheduled_row).unwrap();
        assert_eq!(
            limited.membership(&controller_goal.id, ObjectiveCollection::ScheduledGoals),
            ObjectiveCollectionMembership::Unknown
        );
        let scope = limited
            .collection_scope_json(ObjectiveCollection::ScheduledGoals)
            .unwrap();
        assert_eq!(scope["complete"], false);
        assert_eq!(scope["absence_semantics"], "unknown_outside_returned_rows");
        assert_eq!(
            scope["not_enumerated"],
            serde_json::json!(["mandate_controllers"])
        );

        // Passing one subject's durable row while naming another fails before
        // any lifecycle or control field can be projected.
        let error = objective_portfolio_row(
            &controller_goal.id,
            ObjectiveCollection::MandateControllers,
            &[schedule],
            &[],
            None,
            Some(&mandate),
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not objective subject"));
    }

    fn scheduled_collection_of(
        metadata: &crate::traits::ToolCallMetadata,
    ) -> crate::traits::ToolCollectionObservation {
        metadata
            .collection_observations
            .iter()
            .find(|collection| {
                collection.collection.value == ObjectiveCollection::ScheduledGoals.resource_id()
            })
            .cloned()
            .expect("scheduled-goal collection coverage")
    }

    #[tokio::test]
    async fn every_owner_surface_enumerates_the_same_scheduled_goal_collection() {
        // R53: an owner audit consulted only `manage_mandates list`, which
        // reported the scheduled-goal collection as `0/0 (complete)` because
        // the one scheduled objective had been created through a sibling
        // channel session of the same owner. `scheduled_goal_runs overview`
        // saw it. One collection ID must have one membership everywhere.
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let now = chrono::Utc::now().to_rfc3339();
        let mut created = Vec::new();
        for (description, session) in [
            ("Publish the synthetic daily digest", "owner-bot-a:1001"),
            ("Refresh the synthetic status page", "owner-bot-b:1001"),
            ("Roll the synthetic weekly summary", "owner-slack-dm:1001"),
        ] {
            let goal = Goal::new_continuous(description, session, None, None);
            state.create_goal(&goal).await.unwrap();
            state
                .create_goal_schedule(&GoalSchedule {
                    id: uuid::Uuid::new_v4().to_string(),
                    goal_id: goal.id.clone(),
                    cron_expr: "0 6 * * *".to_string(),
                    tz: "local".to_string(),
                    original_schedule: Some("daily".to_string()),
                    fire_policy: "coalesce".to_string(),
                    is_one_shot: false,
                    is_paused: false,
                    last_run_at: None,
                    next_run_at: "2026-09-03T10:00:00+00:00".to_string(),
                    created_at: now.clone(),
                    updated_at: now.clone(),
                })
                .await
                .unwrap();
            created.push(objective_resource_id(&goal.id));
        }

        use crate::traits::Tool as _;
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let mandates = crate::tools::manage_mandates::ManageMandatesTool::new(
            state.clone(),
            crate::tools::ApprovalBroker::new(approval_tx),
        );
        let runs = crate::tools::scheduled_goal_runs::ScheduledGoalRunsTool::new(state.clone());

        let mandate_view = mandates
            .call_with_status_outcome(
                &serde_json::json!({
                    "action": "list",
                    "_session_id": "owner-bot-a:1001",
                    "_user_role": "owner",
                    "_channel_visibility": "private"
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();
        let runs_view = runs
            .call_with_status_outcome(r#"{"action":"overview"}"#, None)
            .await
            .unwrap();

        let from_mandates = scheduled_collection_of(&mandate_view.metadata);
        let from_runs = scheduled_collection_of(&runs_view.metadata);
        assert_eq!(from_mandates.total_count, Some(3));
        assert_eq!(from_mandates.returned_count, 3);
        assert_eq!(
            from_mandates.completeness,
            crate::traits::ToolCollectionCompleteness::Complete
        );
        let mut mandate_members = from_mandates
            .members
            .iter()
            .map(|member| member.value.clone())
            .collect::<Vec<_>>();
        let mut runs_members = from_runs
            .members
            .iter()
            .map(|member| member.value.clone())
            .collect::<Vec<_>>();
        mandate_members.sort();
        runs_members.sort();
        created.sort();
        assert_eq!(mandate_members, created);
        assert_eq!(runs_members, created);
        assert_eq!(from_runs.total_count, from_mandates.total_count);
        assert_eq!(from_runs.completeness, from_mandates.completeness);

        // The rendered audit text names every objective, not just the ones
        // created through the calling session.
        for member in &created {
            assert!(
                mandate_view.output.contains(member),
                "{}",
                mandate_view.output
            );
        }
    }

    #[tokio::test]
    async fn scheduled_goal_collection_reports_truncation_as_partial_coverage() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let now = chrono::Utc::now().to_rfc3339();
        for index in 0..3 {
            let goal = Goal::new_continuous(
                &format!("Synthetic scheduled objective {index}"),
                "owner-bot-a:1001",
                None,
                None,
            );
            state.create_goal(&goal).await.unwrap();
            state
                .create_goal_schedule(&GoalSchedule {
                    id: uuid::Uuid::new_v4().to_string(),
                    goal_id: goal.id.clone(),
                    cron_expr: "0 6 * * *".to_string(),
                    tz: "local".to_string(),
                    original_schedule: Some("daily".to_string()),
                    fire_policy: "coalesce".to_string(),
                    is_one_shot: false,
                    is_paused: false,
                    last_run_at: None,
                    next_run_at: "2026-09-03T10:00:00+00:00".to_string(),
                    created_at: now.clone(),
                    updated_at: now.clone(),
                })
                .await
                .unwrap();
        }

        let (goals, coverage) = load_scheduled_goal_collection(state.as_ref(), 2)
            .await
            .unwrap();
        assert_eq!(goals.len(), 2);
        assert_eq!(coverage.total, 3);
        assert_eq!(coverage.returned, 2);
        assert!(!coverage.is_complete());
        let (goals, coverage) = load_scheduled_goal_collection(state.as_ref(), 50)
            .await
            .unwrap();
        assert_eq!(goals.len(), 3);
        assert!(coverage.is_complete());
    }
}
