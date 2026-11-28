//! Privacy-minimal continuity for an autonomous mandate.
//!
//! This projection is deliberately narrower than conversational memory. It
//! carries only same-mandate protocol outcomes, typed mutation receipts, and
//! quota state. Model-authored rationale, questions, task prose, arguments,
//! bodies, outputs, errors, credentials, and internal lease identifiers never
//! enter the block.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Value, json};

use crate::traits::{
    Intention, MandateActionOutcome, MandateDecisionCycle, MandateLearningNote,
    MandateMutationAttempt, MandateMutationAttemptStatus, MandateMutationQuotaState,
    MandateObjectiveControl, MandateObjectiveMeasurement, MandateStore, ObjectiveMetricDirection,
};

const MAX_DECISIONS: i64 = 10;
const MAX_INTENTIONS: i64 = 16;
const MAX_ACTIONS: usize = 24;
const MAX_ACTION_OUTCOMES: i64 = 64;
const UNTAGGED_FORMAT: &str = "untagged";

/// Consecutive latest credible readings that neither reach the target nor
/// improve after which an objective counts as stalled.
pub const STALLED_MEASUREMENTS_THRESHOLD: usize = 6;

/// How many of the most recent credible (`confidence_bps > 0`)
/// measurements in a row neither reached the target nor improved on the
/// credible reading before them (the baseline, for the first). `ordered`
/// runs oldest to newest; `AtLeast` improves upward, `AtMost` downward.
pub fn objective_flat_streak(
    control: &MandateObjectiveControl,
    ordered: &[&MandateObjectiveMeasurement],
) -> usize {
    let credible = ordered
        .iter()
        .filter(|measurement| measurement.confidence_bps > 0)
        .map(|measurement| measurement.value_micros)
        .collect::<Vec<_>>();
    let mut streak = 0;
    for index in (0..credible.len()).rev() {
        let value = credible[index];
        let prior = index
            .checked_sub(1)
            .map_or(control.baseline_micros, |prior| credible[prior]);
        let improved = match control.direction {
            ObjectiveMetricDirection::AtLeast => value > prior,
            ObjectiveMetricDirection::AtMost => value < prior,
        };
        if control.target_reached(value) || improved {
            break;
        }
        streak += 1;
    }
    streak
}
const MAX_LEARNING_NOTES: i64 = 12;
const MAX_MEASUREMENTS: i64 = 12;
const MAX_HISTORY_BYTES: usize = 24 * 1024;

#[derive(Debug)]
struct HistoryRecord {
    timestamp: String,
    value: Value,
}

/// Build the sole historical context available to a mandate worker.
///
/// `objective_control` is the mandate's current owner-confirmed control
/// policy. It is passed in (never re-read here) so the derived
/// `next_measurement_due` cannot disagree with the immutable policy the
/// worker was fenced with.
pub async fn build_mandate_history_block(
    state: &(impl MandateStore + ?Sized),
    mandate_id: &str,
    as_of: &str,
    objective_control: Option<&MandateObjectiveControl>,
) -> anyhow::Result<String> {
    let decisions = state
        .list_mandate_decisions(mandate_id, MAX_DECISIONS)
        .await?;
    anyhow::ensure!(
        decisions
            .iter()
            .all(|decision| decision.mandate_id == mandate_id),
        "mandate history query crossed its authority boundary"
    );

    let intentions = state.list_intentions(mandate_id, MAX_INTENTIONS).await?;
    anyhow::ensure!(
        intentions
            .iter()
            .all(|intention| intention.mandate_id == mandate_id),
        "mandate intention history crossed its authority boundary"
    );
    let learning_notes = state
        .list_mandate_learning_notes(mandate_id, MAX_LEARNING_NOTES)
        .await?;
    anyhow::ensure!(
        learning_notes
            .iter()
            .all(|note| note.mandate_id == mandate_id),
        "mandate learning history crossed its authority boundary"
    );

    let mut decision_records = decisions
        .iter()
        .map(|decision| HistoryRecord {
            timestamp: decision.created_at.clone(),
            value: decision_value(decision, &intentions),
        })
        .collect::<Vec<_>>();
    decision_records.sort_by(|left, right| left.timestamp.cmp(&right.timestamp));

    let mut attempts = Vec::new();
    for decision in &decisions {
        for attempt in state
            .list_mandate_mutation_attempts_for_run(&decision.goal_run_id)
            .await?
        {
            anyhow::ensure!(
                attempt.mandate_id == mandate_id
                    && attempt.decision_cycle_id == decision.id
                    && attempt.goal_run_id == decision.goal_run_id,
                "mandate action history crossed its decision boundary"
            );
            attempts.push(attempt);
        }
    }
    attempts.sort_by(|left, right| left.reserved_at.cmp(&right.reserved_at));
    if attempts.len() > MAX_ACTIONS {
        attempts.drain(..attempts.len() - MAX_ACTIONS);
    }
    let action_records = attempts
        .iter()
        .map(|attempt| HistoryRecord {
            timestamp: attempt.reserved_at.clone(),
            value: action_value(attempt, &intentions),
        })
        .collect::<Vec<_>>();

    let outcomes = state
        .list_mandate_action_outcomes(mandate_id, MAX_ACTION_OUTCOMES)
        .await?;
    anyhow::ensure!(
        outcomes
            .iter()
            .all(|outcome| outcome.mandate_id == mandate_id),
        "mandate outcome history crossed its authority boundary"
    );
    let outcome_projection = OutcomeProjection::build(&attempts, &intentions, &outcomes);

    let quota = state
        .get_mandate_mutation_quota_state(mandate_id, as_of)
        .await?;
    anyhow::ensure!(
        quota
            .as_ref()
            .is_none_or(|quota| quota.mandate_id == mandate_id),
        "mandate quota history crossed its authority boundary"
    );

    let measurements = if objective_control.is_some() {
        let measurements = state
            .list_mandate_objective_measurements(mandate_id, MAX_MEASUREMENTS)
            .await?;
        anyhow::ensure!(
            measurements
                .iter()
                .all(|measurement| measurement.mandate_id == mandate_id),
            "mandate measurement history crossed its authority boundary"
        );
        measurements
    } else {
        Vec::new()
    };
    let objective = objective_control.map(|control| {
        let latest_observed_at = measurements
            .iter()
            .map(|measurement| measurement.observed_at.as_str())
            .max();
        objective_value(control, &measurements, latest_observed_at)
    });

    render_bounded_history(
        decision_records,
        action_records,
        learning_notes,
        quota,
        objective,
        outcome_projection,
    )
}

/// What earlier published resources did, by format: only ids, format keys
/// and counts.
#[derive(Debug, Default)]
struct OutcomeProjection {
    /// One entry per format, most posts first.
    format_performance: Vec<Value>,
    /// Succeeded recent actions' resources with no outcome yet, oldest
    /// first.
    unmeasured_resource_ids: Vec<String>,
}

impl OutcomeProjection {
    fn build(
        attempts: &[MandateMutationAttempt],
        intentions: &[Intention],
        outcomes: &[MandateActionOutcome],
    ) -> Self {
        #[derive(Default)]
        struct FormatStats {
            posts: u64,
            measured_posts: u64,
            total_engagements_micros: i64,
            max_engagements_micros: Option<i64>,
            max_impressions: Option<i64>,
        }
        // The newest outcome per resource.
        let mut latest: HashMap<&str, &MandateActionOutcome> = HashMap::new();
        for outcome in outcomes {
            latest
                .entry(outcome.external_resource_id.as_str())
                .and_modify(|current| {
                    if outcome.observed_at > current.observed_at {
                        *current = outcome;
                    }
                })
                .or_insert(outcome);
        }
        // Every resource the window knows of, with its format.
        let mut resources: BTreeMap<String, String> = BTreeMap::new();
        let mut unmeasured = Vec::new();
        let mut seen = HashSet::new();
        for attempt in attempts {
            if attempt.status != MandateMutationAttemptStatus::Succeeded {
                continue;
            }
            let format = attempt_format_key(attempt, intentions)
                .unwrap_or(UNTAGGED_FORMAT)
                .to_string();
            match attempt.external_resource_id.as_deref() {
                Some(id) => {
                    if seen.insert(id.to_string()) && !latest.contains_key(id) {
                        unmeasured.push(id.to_string());
                    }
                    resources.insert(format!("id:{id}"), format);
                }
                None => {
                    resources.insert(format!("attempt:{}", attempt.id), format);
                }
            }
        }
        for outcome in latest.values() {
            resources
                .entry(format!("id:{}", outcome.external_resource_id))
                .or_insert_with(|| {
                    outcome
                        .format_key
                        .clone()
                        .unwrap_or_else(|| UNTAGGED_FORMAT.to_string())
                });
        }
        let mut by_format: BTreeMap<String, FormatStats> = BTreeMap::new();
        for (resource, format) in &resources {
            let stats = by_format.entry(format.clone()).or_default();
            stats.posts += 1;
            let Some(outcome) = resource.strip_prefix("id:").and_then(|id| latest.get(id)) else {
                continue;
            };
            stats.measured_posts += 1;
            stats.total_engagements_micros = stats
                .total_engagements_micros
                .saturating_add(outcome.engagements_micros);
            stats.max_engagements_micros = stats
                .max_engagements_micros
                .max(Some(outcome.engagements_micros));
            stats.max_impressions = stats.max_impressions.max(outcome.impressions);
        }
        let mut formats = by_format.into_iter().collect::<Vec<_>>();
        formats.sort_by(|(left_key, left), (right_key, right)| {
            right.posts.cmp(&left.posts).then(left_key.cmp(right_key))
        });
        Self {
            format_performance: formats
                .into_iter()
                .map(|(format_key, stats)| {
                    json!({
                        "format_key": format_key,
                        "posts": stats.posts,
                        "measured_posts": stats.measured_posts,
                        "total_engagements_micros": stats.total_engagements_micros,
                        "max_engagements_micros": stats.max_engagements_micros,
                        "max_impressions": stats.max_impressions,
                    })
                })
                .collect(),
            unmeasured_resource_ids: unmeasured,
        }
    }
}

fn attempt_format_key<'a>(
    attempt: &MandateMutationAttempt,
    intentions: &'a [Intention],
) -> Option<&'a str> {
    intentions
        .iter()
        .find(|intention| intention.id == attempt.intention_id)
        .and_then(|intention| intention.format_key.as_deref())
}

/// Measured-objective state: baseline, current, target, trend, and when
/// the next measurement is due. Measurement records carry only the typed
/// value, confidence, receipt IDs, and time; run and row identifiers stay
/// private.
fn objective_value(
    control: &MandateObjectiveControl,
    measurements: &[MandateObjectiveMeasurement],
    latest_observed_at: Option<&str>,
) -> Value {
    let mut ordered = measurements.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.observed_at.cmp(&right.observed_at));
    let current = ordered.last().map(|measurement| measurement.value_micros);
    let flat_streak = objective_flat_streak(control, &ordered);
    json!({
        "metric_name": control.metric_name,
        "unit": control.unit,
        "baseline_micros": control.baseline_micros,
        "baseline_observed_at": control.baseline_observed_at,
        "current_micros": current,
        "target_micros": control.target_micros,
        "direction": control.direction,
        "target_reached": current.map(|value| control.target_reached(value)),
        "measurement_source": control.measurement_source,
        "measurement_cadence_secs": control.measurement_cadence_secs,
        "next_measurement_due": control.next_measurement_at(latest_observed_at),
        "flat_measurements_streak": flat_streak,
        "stalled": flat_streak >= STALLED_MEASUREMENTS_THRESHOLD,
        "objective_measurements": ordered.iter().map(|measurement| json!({
            "mandate_version": measurement.mandate_version,
            "value_micros": measurement.value_micros,
            "confidence_bps": measurement.confidence_bps,
            "observed_at": measurement.observed_at,
        })).collect::<Vec<_>>(),
    })
}

fn decision_value(decision: &MandateDecisionCycle, intentions: &[Intention]) -> Value {
    let intention = intentions.iter().find(|intention| {
        intention.decision_cycle_id == decision.id && intention.goal_run_id == decision.goal_run_id
    });
    json!({
        "mandate_version": decision.mandate_version,
        "outcome": decision.outcome,
        "activity_level": decision.activity_level,
        "action_attempts": decision.action_attempts,
        "termination_kind": decision.termination_kind,
        "termination_match": decision.termination_match,
        "value_criterion": intention.and_then(|intention| intention.value_criterion.as_deref()),
        "format_key": intention.and_then(|intention| intention.format_key.as_deref()),
        "intention_status": intention.map(|intention| intention.status.as_str()),
        "reconsider_at": decision.reconsider_at,
        "created_at": decision.created_at,
        "updated_at": decision.updated_at,
    })
}

fn action_value(attempt: &MandateMutationAttempt, intentions: &[Intention]) -> Value {
    json!({
        "mandate_version": attempt.mandate_version,
        "reserved_action_attempt": attempt.reserved_action_attempt,
        "action_digest": attempt.action_digest,
        "tool_name": attempt.tool_name,
        "mutation_effects": attempt.mutation_effects,
        "targets": attempt.targets,
        "account_identifiers": attempt.account_identifiers,
        "status": attempt.status,
        "outcome_evidence": attempt.outcome_evidence,
        "http_status": attempt.http_status,
        "exit_code": attempt.exit_code,
        "external_resource_id": attempt.external_resource_id,
        "format_key": attempt_format_key(attempt, intentions),
        "reserved_at": attempt.reserved_at,
        "completed_at": attempt.completed_at,
    })
}

fn render_bounded_history(
    mut decisions: Vec<HistoryRecord>,
    mut actions: Vec<HistoryRecord>,
    mut learning_notes: Vec<MandateLearningNote>,
    quota: Option<MandateMutationQuotaState>,
    objective: Option<Value>,
    mut outcomes: OutcomeProjection,
) -> anyhow::Result<String> {
    learning_notes.sort_by(|left, right| left.created_at.cmp(&right.created_at));
    loop {
        let rendered = serde_json::to_string(&json!({
            "provenance": "autonomous_mandate_history_untrusted",
            "authority": false,
            "scope": "same_mandate_typed_history_only",
            "objective_control_state": objective,
            "decision_outcomes": decisions.iter().map(|record| &record.value).collect::<Vec<_>>(),
            "mutation_receipts": actions.iter().map(|record| &record.value).collect::<Vec<_>>(),
            "advisory_learning_notes": learning_notes.iter().map(|note| json!({
                "mandate_version": note.mandate_version,
                "summary": note.summary,
                "created_at": note.created_at,
            })).collect::<Vec<_>>(),
            "mutation_quota": quota,
            "format_performance": outcomes.format_performance,
            "unmeasured_resource_ids": outcomes.unmeasured_resource_ids,
        }))?;
        if rendered.len() <= MAX_HISTORY_BYTES {
            return Ok(rendered);
        }

        // Drop the oldest whole record across both streams. Never truncate a
        // JSON string or typed receipt into a misleading partial record.
        match (decisions.first(), actions.first()) {
            (Some(decision), Some(action)) if decision.timestamp <= action.timestamp => {
                decisions.remove(0);
            }
            (Some(_), Some(_)) => {
                actions.remove(0);
            }
            (Some(_), None) => {
                decisions.remove(0);
            }
            (None, Some(_)) => {
                actions.remove(0);
            }
            (None, None) if !learning_notes.is_empty() => {
                learning_notes.remove(0);
            }
            (None, None) if !outcomes.unmeasured_resource_ids.is_empty() => {
                outcomes.unmeasured_resource_ids.remove(0);
            }
            (None, None) if !outcomes.format_performance.is_empty() => {
                outcomes.format_performance.pop();
            }
            (None, None) => {
                anyhow::bail!("typed mandate quota projection exceeded its hard prompt bound")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{
        Intention, MandateDecisionOutcome, MandateMutationAttemptStatus, MandateMutationEvidence,
        MandateMutationTarget,
    };

    fn attempt() -> MandateMutationAttempt {
        MandateMutationAttempt {
            id: "ledger-private-id".to_string(),
            mandate_id: "mandate-1".to_string(),
            mandate_version: 3,
            decision_cycle_id: "cycle-1".to_string(),
            goal_run_id: "run-private-id".to_string(),
            intention_id: "intention-private-id".to_string(),
            root_task_id: "root-private-id".to_string(),
            root_task_attempt_id: "root-attempt-private-id".to_string(),
            task_id: "task-private-id".to_string(),
            task_attempt_id: "task-attempt-private-id".to_string(),
            reserved_action_attempt: 1,
            action_digest: "a".repeat(64),
            tool_call_id: "tool-call-private-id".to_string(),
            tool_name: "http_request".to_string(),
            mutation_effects: vec!["remote_mutation".to_string()],
            targets: vec![MandateMutationTarget {
                kind: "resource_id".to_string(),
                identifier: "auth_profile:twitter-prod".to_string(),
            }],
            account_identifiers: vec!["auth_profile:twitter-prod".to_string()],
            status: MandateMutationAttemptStatus::Succeeded,
            outcome_evidence: Some(MandateMutationEvidence::ToolReported),
            http_status: Some(201),
            exit_code: None,
            external_resource_id: Some("1000000000000000001".to_string()),
            reserved_at: "2026-08-01T12:00:00Z".to_string(),
            completed_at: Some("2026-08-01T12:00:01Z".to_string()),
        }
    }

    #[test]
    fn objective_state_reports_trend_and_next_measurement_without_private_ids() {
        let control = MandateObjectiveControl {
            schema_version: MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "synthetic followers".to_string(),
            unit: "count".to_string(),
            baseline_micros: 100_000_000,
            target_micros: 150_000_000,
            direction: crate::traits::ObjectiveMetricDirection::AtLeast,
            measurement_source: "metric_source:synthetic-analytics".to_string(),
            measurement_cadence_secs: 86_400,
            baseline_observed_at: "2026-08-01T00:00:00Z".to_string(),
        };
        let mut older = MandateObjectiveMeasurement::new(
            "mandate-1",
            2,
            "run-private-id",
            120_000_000,
            9_000,
            vec!["receipt-1".to_string()],
            "2026-08-02T00:00:00Z",
        );
        older.id = "measurement-private-id".to_string();
        let newer = MandateObjectiveMeasurement::new(
            "mandate-1",
            2,
            "run-private-id",
            130_000_000,
            9_500,
            vec!["receipt-2".to_string()],
            "2026-08-03T00:00:00Z",
        );
        let value = objective_value(&control, &[newer, older], Some("2026-08-03T00:00:00Z"));
        assert_eq!(value["current_micros"], 130_000_000);
        assert_eq!(value["target_reached"], false);
        assert_eq!(value["next_measurement_due"], "2026-08-04T00:00:00Z");
        let rendered = value.to_string();
        let series = &rendered[rendered.find("objective_measurements").unwrap()..];
        assert!(series.find("120000000").unwrap() < series.find("130000000").unwrap());
        assert!(!rendered.contains("run-private-id"));
        assert!(!rendered.contains("measurement-private-id"));

        let unmeasured = objective_value(&control, &[], None);
        assert!(unmeasured["current_micros"].is_null());
        assert_eq!(unmeasured["next_measurement_due"], "2026-08-02T00:00:00Z");
    }

    #[test]
    fn action_history_excludes_internal_ids_and_content_fields() {
        let rendered = action_value(&attempt(), &[]).to_string();
        assert!(rendered.contains("1000000000000000001"));
        assert!(rendered.contains("auth_profile:twitter-prod"));
        for private in [
            "ledger-private-id",
            "run-private-id",
            "intention-private-id",
            "root-private-id",
            "root-attempt-private-id",
            "task-private-id",
            "task-attempt-private-id",
            "tool-call-private-id",
        ] {
            assert!(!rendered.contains(private));
        }
        for forbidden_key in ["arguments", "body", "output", "error", "credentials"] {
            assert!(!rendered.contains(forbidden_key));
        }
    }

    #[test]
    fn decision_history_retains_owner_value_anchor_without_model_plan_text() {
        let decision = MandateDecisionCycle::new(
            "mandate-1",
            "run-1",
            MandateDecisionOutcome::Act,
            "MODEL_RATIONALE_MUST_NOT_PERSIST",
            3,
        );
        let mut intention = Intention::new(
            "mandate-1",
            &decision.id,
            "run-1",
            "MODEL_PLAN_MUST_NOT_PERSIST",
            "MODEL_INTENTION_RATIONALE_MUST_NOT_PERSIST",
        );
        intention.value_criterion =
            Some("Provide verified useful information to the audience".to_string());

        let rendered = decision_value(&decision, &[intention]).to_string();
        assert!(rendered.contains("Provide verified useful information"));
        assert!(!rendered.contains("MODEL_RATIONALE_MUST_NOT_PERSIST"));
        assert!(!rendered.contains("MODEL_PLAN_MUST_NOT_PERSIST"));
        assert!(!rendered.contains("MODEL_INTENTION_RATIONALE_MUST_NOT_PERSIST"));
    }

    #[test]
    fn history_drops_oldest_whole_records_to_fit_hard_byte_bound() {
        let huge = HistoryRecord {
            timestamp: "2026-08-01T00:00:00Z".to_string(),
            value: json!({"typed_identifier": "x".repeat(MAX_HISTORY_BYTES)}),
        };
        let recent = HistoryRecord {
            timestamp: "2026-08-01T01:00:00Z".to_string(),
            value: json!({"status": "succeeded"}),
        };
        let rendered = render_bounded_history(
            vec![huge, recent],
            Vec::new(),
            Vec::new(),
            None,
            None,
            OutcomeProjection::default(),
        )
        .unwrap();
        assert!(rendered.len() <= MAX_HISTORY_BYTES);
        assert!(!rendered.contains("typed_identifier"));
        assert!(rendered.contains("succeeded"));
    }

    fn control(direction: crate::traits::ObjectiveMetricDirection) -> MandateObjectiveControl {
        MandateObjectiveControl {
            schema_version: MandateObjectiveControl::SCHEMA_VERSION,
            metric_name: "weekly_engagements".to_string(),
            unit: "engagements".to_string(),
            baseline_micros: 1_000_000,
            target_micros: match direction {
                crate::traits::ObjectiveMetricDirection::AtLeast => 100_000_000,
                crate::traits::ObjectiveMetricDirection::AtMost => 0,
            },
            direction,
            measurement_source: "metric_source:synthetic-analytics".to_string(),
            measurement_cadence_secs: 86_400,
            baseline_observed_at: "2026-08-01T00:00:00Z".to_string(),
        }
    }

    fn readings(values: &[(i64, u16)]) -> Vec<MandateObjectiveMeasurement> {
        values
            .iter()
            .enumerate()
            .map(|(day, (value, confidence))| {
                MandateObjectiveMeasurement::new(
                    "mandate-1",
                    1,
                    "run-1",
                    *value,
                    *confidence,
                    vec![format!("receipt-{day}")],
                    &format!("2026-08-{:02}T00:00:00Z", day + 2),
                )
            })
            .collect()
    }

    #[test]
    fn the_flat_streak_counts_latest_credible_readings_that_did_not_improve() {
        use crate::traits::ObjectiveMetricDirection::{AtLeast, AtMost};
        let at_least = control(AtLeast);
        let streak = |control: &MandateObjectiveControl, values: &[(i64, u16)]| {
            let measurements = readings(values);
            objective_flat_streak(control, &measurements.iter().collect::<Vec<_>>())
        };
        assert_eq!(streak(&at_least, &[]), 0);
        // Equal to the baseline is flat; an uptick then flat readings.
        assert_eq!(streak(&at_least, &[(1_000_000, 9_000)]), 1);
        assert_eq!(
            streak(
                &at_least,
                &[
                    (1_000_000, 9_000),
                    (2_000_000, 9_000),
                    (2_000_000, 9_000),
                    (1_000_000, 9_000)
                ]
            ),
            2
        );
        // Zero-confidence readings are skipped, not counted or compared.
        assert_eq!(
            streak(
                &at_least,
                &[
                    (2_000_000, 9_000),
                    (9_000_000, 0),
                    (2_000_000, 9_000),
                    (2_000_000, 9_000)
                ]
            ),
            2
        );
        let flat = vec![(1_000_000, 9_000); 7];
        assert_eq!(streak(&at_least, &flat), 7);
        assert!(streak(&at_least, &flat) >= STALLED_MEASUREMENTS_THRESHOLD);
        // A reading at the target ends the streak.
        assert_eq!(
            streak(&at_least, &[(1_000_000, 9_000), (100_000_000, 9_000)]),
            0
        );
        // AtMost improves downward.
        let at_most = control(AtMost);
        assert_eq!(streak(&at_most, &[(500_000, 9_000)]), 0);
        assert_eq!(streak(&at_most, &[(500_000, 9_000), (700_000, 9_000)]), 1);

        let measurements = readings(&flat);
        let value = objective_value(&at_least, &measurements, None);
        assert_eq!(value["flat_measurements_streak"], 7);
        assert_eq!(value["stalled"], true);
        let value = objective_value(&at_least, &measurements[..2], None);
        assert_eq!(value["stalled"], false);
    }

    #[test]
    fn outcomes_project_by_format_and_name_unmeasured_resources() {
        let mut tip = Intention::new("mandate-1", "cycle-1", "run-1", "plan", "why");
        tip.format_key = Some("question_hook".to_string());
        let posted = |id: &str, resource: Option<&str>, intention: &Intention| {
            let mut attempt = attempt();
            attempt.id = id.to_string();
            attempt.intention_id = intention.id.clone();
            attempt.external_resource_id = resource.map(str::to_string);
            attempt
        };
        let untagged = Intention::new("mandate-1", "cycle-2", "run-2", "plan", "why");
        let mut failed = posted("a-failed", Some("40"), &tip);
        failed.status = MandateMutationAttemptStatus::Failed;
        let attempts = vec![
            posted("a-1", Some("11"), &tip),
            posted("a-2", Some("12"), &tip),
            posted("a-3", Some("21"), &untagged),
            posted("a-4", None, &untagged),
            failed,
        ];
        let outcome = |resource: &str, engagements: i64, observed_at: &str| {
            let mut outcome = MandateActionOutcome::new(
                "mandate-1",
                1,
                "cycle-9",
                resource,
                engagements,
                Some(engagements / 10_000),
                vec!["receipt-9".to_string()],
                observed_at,
            );
            outcome.format_key = Some("question_hook".to_string());
            outcome
        };
        let outcomes = vec![
            outcome("11", 2_000_000, "2026-08-03T00:00:00Z"),
            outcome("11", 5_000_000, "2026-08-04T00:00:00Z"),
            // A resource older than the action window still counts.
            outcome("09", 1_000_000, "2026-08-02T00:00:00Z"),
        ];
        let projection =
            OutcomeProjection::build(&attempts, &[tip.clone(), untagged.clone()], &outcomes);
        assert_eq!(projection.unmeasured_resource_ids, vec!["12", "21"]);
        let hook = &projection.format_performance[0];
        assert_eq!(hook["format_key"], "question_hook");
        assert_eq!(hook["posts"], 3);
        assert_eq!(hook["measured_posts"], 2);
        assert_eq!(hook["total_engagements_micros"], 6_000_000);
        assert_eq!(hook["max_engagements_micros"], 5_000_000);
        assert_eq!(hook["max_impressions"], 500);
        let other = &projection.format_performance[1];
        assert_eq!(other["format_key"], "untagged");
        assert_eq!(other["posts"], 2);
        assert_eq!(other["measured_posts"], 0);
        assert!(other["max_engagements_micros"].is_null());

        let rendered = action_value(&attempts[0], &[tip]).to_string();
        assert!(
            rendered.contains("\"format_key\":\"question_hook\""),
            "{rendered}"
        );

        let rendered =
            render_bounded_history(Vec::new(), Vec::new(), Vec::new(), None, None, projection)
                .unwrap();
        assert!(rendered.contains("format_performance"));
        assert!(rendered.contains("unmeasured_resource_ids"));
    }

    #[test]
    fn outcome_projection_is_dropped_last_to_fit_the_byte_bound() {
        let projection = OutcomeProjection {
            format_performance: (0..400)
                .map(|index| json!({"format_key": format!("format-{index:03}-{}", "x".repeat(40))}))
                .collect(),
            unmeasured_resource_ids: (0..200).map(|index| format!("{index:064}")).collect(),
        };
        let rendered =
            render_bounded_history(Vec::new(), Vec::new(), Vec::new(), None, None, projection)
                .unwrap();
        assert!(rendered.len() <= MAX_HISTORY_BYTES);
        assert!(
            rendered.contains("format-000"),
            "the busiest formats are kept"
        );
    }
}
