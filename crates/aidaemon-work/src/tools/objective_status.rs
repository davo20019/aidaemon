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
pub enum ObjectiveCollection {
    ScheduledGoals,
    MandateControllers,
}

impl ObjectiveCollection {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScheduledGoals => "scheduled_goals",
            Self::MandateControllers => "mandate_controllers",
        }
    }

    pub const fn resource_id(self) -> &'static str {
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
    pub fn stable_subject(self) -> crate::traits::StableObservationSubject {
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

pub const OBJECTIVE_MEMBER_NAMESPACE: &str = "objective:";

pub const OBJECTIVE_STATUS_FACETS: [crate::traits::ToolSemanticFacet; 6] = [
    crate::traits::ToolSemanticFacet::Schedule,
    crate::traits::ToolSemanticFacet::RunState,
    crate::traits::ToolSemanticFacet::Recovery,
    crate::traits::ToolSemanticFacet::Control,
    crate::traits::ToolSemanticFacet::Measurement,
    crate::traits::ToolSemanticFacet::Ownership,
];

pub fn objective_resource_id(goal_id: &str) -> String {
    use sha2::{Digest, Sha256};

    // The checklist needs a stable exact identity, while owner-facing tool
    // output must not expose internal database UUIDs. A full digest keeps the
    // mapping deterministic without relying on display prose or row order.
    format!(
        "{OBJECTIVE_MEMBER_NAMESPACE}sha256:{}",
        hex::encode(Sha256::digest(goal_id.as_bytes()))
    )
}

/// Coverage for one collection enumeration. `complete` is derived rather than
/// supplied so callers cannot accidentally label a limited result authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectiveCollectionCoverage {
    pub collection: ObjectiveCollection,
    pub total: usize,
    pub returned: usize,
}

impl ObjectiveCollectionCoverage {
    pub fn new(
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

    pub const fn is_complete(self) -> bool {
        self.returned == self.total
    }

    pub fn as_json(self) -> serde_json::Value {
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
pub const SCHEDULED_COLLECTION_LIMIT: usize = 100;

/// Enumerate the scheduled-goal collection the same way on every owner
/// surface. The collection is daemon-wide: a goal's `session_id` records the
/// channel session that created it, not an authority boundary, and one owner
/// routinely reaches the daemon through several sessions (two Telegram bots,
/// a Slack DM). A surface that filters this collection by session while still
/// labeling its coverage `complete` turns a goal created through a sibling
/// owner session into an authoritative "absent" (R53). `total` therefore
/// counts every scheduled goal, so truncation stays honest as `partial`.
pub async fn load_scheduled_goal_collection<S>(
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveCollectionMembership {
    Present,
    Absent,
    Unknown,
}

/// Subject-keyed row in the canonical objective portfolio. The stable goal ID
/// is retained even when a public renderer elects not to display internal IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectivePortfolioRow {
    pub goal_id: String,
    pub source_membership: BTreeSet<ObjectiveCollection>,
    pub status: ObjectiveStatus,
}

impl ObjectivePortfolioRow {
    pub fn subject_hint(&self) -> crate::traits::ToolTargetHint {
        crate::traits::ToolTargetHint::new(
            crate::traits::ToolTargetHintKind::ResourceId,
            objective_resource_id(&self.goal_id),
        )
        .expect("validated objective goal IDs are nonempty")
    }

    pub fn source_membership_json(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.source_membership
                .iter()
                .map(|collection| serde_json::Value::String(collection.as_str().to_string()))
                .collect(),
        )
    }

    pub fn source_membership_label(&self) -> String {
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
pub struct ObjectivePortfolio {
    subjects: BTreeMap<String, ObjectivePortfolioRow>,
    coverage: BTreeMap<ObjectiveCollection, ObjectiveCollectionCoverage>,
}

impl ObjectivePortfolio {
    pub fn record_collection(
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

    pub fn insert(&mut self, row: ObjectivePortfolioRow) -> anyhow::Result<()> {
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

    pub fn subject(&self, goal_id: &str) -> Option<&ObjectivePortfolioRow> {
        self.subjects.get(goal_id)
    }

    pub fn subjects(&self) -> impl Iterator<Item = &ObjectivePortfolioRow> {
        self.subjects.values()
    }

    pub fn membership(
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

    pub fn collection_scope_json(
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
    pub fn receipt_evidence(
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
pub struct ObjectiveStatus {
    /// What wakes this objective next.
    pub wake: ObjectiveWake,
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

/// What wakes an objective. A mandate wakes from its own review clock and
/// never uses `goal_schedules`, so an empty schedule list says nothing about
/// a mandate controller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveWake {
    GoalSchedule {
        /// Earliest next occurrence among active schedules; `None` when all
        /// are paused.
        next_run_at: Option<String>,
    },
    MandateClock {
        next_review_at: String,
    },
    /// A mandate that is not active: nothing will wake it.
    MandateHalted {
        status: String,
        /// The typed suspension reason, when it is suspended.
        reason: Option<String>,
    },
    None,
}

impl ObjectiveWake {
    /// `wake_source` on structured surfaces.
    pub fn source(&self) -> &'static str {
        match self {
            Self::GoalSchedule { .. } => "goal_schedule",
            Self::MandateClock { .. } => "mandate_clock",
            Self::MandateHalted { .. } | Self::None => "none",
        }
    }

    /// `schedule_state` on structured surfaces.
    pub fn state(&self) -> &'static str {
        match self {
            Self::GoalSchedule {
                next_run_at: Some(_),
            } => "active",
            Self::GoalSchedule { next_run_at: None } => "paused",
            Self::MandateClock { .. } => "mandate_clock",
            Self::MandateHalted { .. } => "mandate_halted",
            Self::None => "missing",
        }
    }

    /// When the objective wakes next, whatever wakes it.
    pub fn next_at(&self) -> Option<&str> {
        match self {
            Self::GoalSchedule { next_run_at } => next_run_at.as_deref(),
            Self::MandateClock { next_review_at } => Some(next_review_at),
            Self::MandateHalted { .. } | Self::None => None,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::GoalSchedule {
                next_run_at: Some(next),
            } => format!("schedule active (next {})", shown_time(next)),
            Self::GoalSchedule { next_run_at: None } => "schedule paused".to_string(),
            Self::MandateClock { next_review_at } => {
                format!("mandate clock (next review {})", shown_time(next_review_at))
            }
            Self::MandateHalted {
                reason: Some(reason),
                ..
            } => format!("suspended ({reason}), no review scheduled"),
            Self::MandateHalted { status, .. } => format!("{status}, no review scheduled"),
            Self::None => "schedule missing".to_string(),
        }
    }
}

/// Typed delegation/control readout for one goal-backed objective, derived
/// from the goal's mandate row and its recorded objective measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveControlFacet {
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
pub fn objective_control_facet(
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
pub fn objective_control_json(control: Option<&ObjectiveControlFacet>) -> serde_json::Value {
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
pub fn objective_portfolio_row(
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

    let wake = match mandate {
        Some(mandate) => match mandate.scheduled_next_review() {
            Some(next_review_at) => ObjectiveWake::MandateClock {
                next_review_at: next_review_at.to_string(),
            },
            None => ObjectiveWake::MandateHalted {
                status: mandate.status.to_string(),
                reason: mandate.suspension.as_ref().map(|suspension| {
                    suspension
                        .reason_code
                        .clone()
                        .unwrap_or_else(|| suspension.kind.as_str().to_string())
                }),
            },
        },
        None if schedules.is_empty() => ObjectiveWake::None,
        None => ObjectiveWake::GoalSchedule {
            next_run_at: schedules
                .iter()
                .filter(|schedule| !schedule.is_paused)
                .map(|schedule| schedule.next_run_at.clone())
                .min(),
        },
    };
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
            wake,
            latest_run,
            recovery,
            runs_completed,
            runs_failed,
            control: objective_control_facet(mandate, measurement_count),
        },
    })
}

/// Render the snapshot as a single plain-text line for list surfaces.
pub fn render_objective_status_line(status: &ObjectiveStatus) -> String {
    let mut line = status.wake.describe();
    match &status.latest_run {
        Some((run_status, at, summary)) => {
            line.push_str(&format!("; last run {run_status} at {at}"));
            if let Some(summary) = summary {
                let summary = summary.trim();
                if !summary.is_empty() {
                    line.push_str(" — ");
                    line.push_str(&aidaemon_events::text::truncate_str(summary, 140));
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

/// A stored time as the owner reads it ("2:39 PM ET"); unreadable values
/// as stored.
fn shown_time(value: &str) -> String {
    crate::schedule::owner_time_rfc3339(value).unwrap_or_else(|| value.to_string())
}
