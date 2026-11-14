//! `mandate_review`: the tool a mandate's review run commits its typed
//! result with. It exists only inside that run (the task lead of a mandate
//! controller), never in an owner's chat: `manage_mandates` is the owner's
//! tool, and its review actions moved here.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::mandates::MAX_INTENTION_METADATA_TEXT;
use crate::mandates::validate::{
    MAX_EVIDENCE_RECEIPTS, MAX_LEARNING_NOTE_TEXT, MAX_POLICY_ENTRY_TEXT, clean_strings,
    learning_note_fields_complete, normalized_learning_note, normalized_termination_fields,
    required_bounded_trimmed, required_call_fact, required_trimmed, validate_act_evidence,
    validate_bounded_strings, validate_bounded_text,
};
use crate::traits::{
    Intention, MandateActivityLevel, MandateDecisionCycle, MandateDecisionOutcome,
    MandateLearningNote, MandateObjectiveMeasurement, MandateOperatingUpdates, MandateToolStore,
    ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolOutcomeStatus, ToolRole,
};
use aidaemon_core::ChannelVisibility;
use aidaemon_tools::OutcomeTool as Tool;

/// The tool's name, as the mandate kernel classifies its calls.
pub const MANDATE_REVIEW_TOOL: &str = "mandate_review";

/// The review actions, which `manage_mandates` once had: a call to one of
/// them there runs nothing and names this tool.
pub const REVIEW_ACTIONS: &[&str] = &["record_decision", "record_measurement"];

/// Whether `action` is a review action `manage_mandates` no longer has.
pub fn moved_action(action: &str) -> bool {
    REVIEW_ACTIONS.contains(&action)
}

const MAX_OBJECTIVE_TEXT: usize = 2 * 1024;
const MAX_RATIONALE_TEXT: usize = 2 * 1024;
const MAX_OBSERVATIONS: usize = 8;
const MAX_OBSERVATION_TEXT: usize = 750;
const MAX_OBSERVATIONS_JSON: usize = 6 * 1024;
const MAX_QUESTION_TEXT: usize = 500;
const MAX_INTENTION_TEXT: usize = 1024;
/// Evidence a learning note may cite (`MandateLearningNote::validate`).
const MAX_LEARNING_EVIDENCE_RECEIPTS: usize = 8;

const MANDATE_OUTCOMES: &[&str] = &["act", "wait", "ask", "stop"];
const ACTIVITY_LEVELS: &[&str] = &["quiet", "active", "urgent"];
const TERMINATION_KINDS: &[&str] = &[
    "success_criteria_satisfied",
    "stop_condition_met",
    "safety_termination",
];

const EVIDENCE_RECEIPT_IDS_DESCRIPTION: &str = "Optional. Omit to cite this run's successful observations automatically. To cite specific ones, use the tool_call_id you issued (e.g. call_...) or the result_id in that result's receipt footer (e.g. sha256:...); receipts from other runs, failed calls, or invented IDs are rejected.";
const LEARNING_EVIDENCE_RECEIPT_IDS_DESCRIPTION: &str = "Optional. Receipts backing learning_note; they must also back this decision. Omit to cite the decision's evidence.";
const OWNER_ALERT_DESCRIPTION: &str = "WAIT only, when the policy grants owner_alerts: the one finding the owner asked to hear about, in a sentence.";
const STOP_ONLY_DESCRIPTION: &str = "STOP only; omit for ACT, WAIT, and ASK.";
const VALUE_CRITERION_DESCRIPTION: &str =
    "ACT: exact owner-confirmed success criterion advanced; legacy uses the exact objective.";
const EXPECTED_BENEFIT_DESCRIPTION: &str = "ACT: benefit of acting now instead of waiting.";
const RISK_DESCRIPTION: &str = "ACT: expected cost or downside, or why none is material.";
const INVALIDATION_DESCRIPTION: &str =
    "ACT: evidence making the intervention unsafe or not worthwhile.";

/// The review run's commit point: one ACT/WAIT/ASK/STOP decision per run,
/// and, for a measured objective, the metric read before it.
pub struct MandateReviewTool {
    state: Arc<dyn MandateToolStore>,
}

impl MandateReviewTool {
    pub fn new(state: Arc<dyn MandateToolStore>) -> Self {
        Self { state }
    }

    fn is_internal(args: &MandateReviewArgs) -> bool {
        internal_call(args.call_session.as_deref(), args.call_visibility)
    }

    /// The receipts the model cited, or, when it cited none, the run's own
    /// successful observations. The runtime already knows which reads
    /// happened; copying their identifiers is bookkeeping, not judgment.
    async fn cited_or_current_run_evidence(
        &self,
        cited: Option<&[String]>,
        goal_run_id: &str,
    ) -> anyhow::Result<Vec<String>> {
        let cited = clean_strings(cited);
        if !cited.is_empty() {
            return Ok(cited);
        }
        self.state
            .current_run_evidence_receipt_ids(goal_run_id, MAX_EVIDENCE_RECEIPTS)
            .await
    }

    async fn record_measurement(&self, args: &MandateReviewArgs) -> anyhow::Result<String> {
        if !Self::is_internal(args) {
            return Ok(
                "record_measurement is reserved for the internal mandate deliberator.".to_string(),
            );
        }
        let goal_id = required_call_fact(args.call_goal_id.as_deref(), "goal")?;
        let goal_run_id = required_call_fact(args.call_goal_run_id.as_deref(), "goal run")?;
        let mandate = self
            .state
            .get_mandate_for_goal(goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("current goal is not a mandate controller"))?;
        anyhow::ensure!(
            mandate.is_active() && mandate.objective_control.is_some(),
            "mandate has no active objective control"
        );
        let value_micros = args
            .measurement_value_micros
            .ok_or_else(|| anyhow::anyhow!("measurement_value_micros is required"))?;
        let receipts = self
            .cited_or_current_run_evidence(args.evidence_receipt_ids.as_deref(), goal_run_id)
            .await?;
        let observed_at = args
            .measurement_observed_at
            .clone()
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
        let measurement = MandateObjectiveMeasurement::new(
            &mandate.id,
            mandate.version,
            goal_run_id,
            value_micros,
            args.measurement_confidence_bps.unwrap_or(10_000),
            receipts,
            &observed_at,
        );
        self.state
            .record_mandate_objective_measurement(&measurement)
            .await?;
        let target_reached = mandate
            .objective_control
            .as_ref()
            .is_some_and(|control| control.target_reached(value_micros));
        Ok(format!(
            "Objective measurement {} recorded for mandate {} (target_reached={target_reached}).",
            measurement.id, mandate.id
        ))
    }

    async fn record_decision(&self, args: &MandateReviewArgs) -> anyhow::Result<String> {
        if !Self::is_internal(args) {
            return Ok(
                "record_decision is reserved for the internal mandate deliberator.".to_string(),
            );
        }
        let goal_id = required_call_fact(args.call_goal_id.as_deref(), "goal")?;
        let pinned_run_id = required_call_fact(args.call_goal_run_id.as_deref(), "goal run")?;
        let pinned_attempt_id =
            required_call_fact(args.call_task_attempt_id.as_deref(), "task attempt")?;
        let mandate = self
            .state
            .get_mandate_for_goal(goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("current goal is not a mandate controller"))?;
        anyhow::ensure!(mandate.is_active(), "mandate is not active");
        let run = self
            .state
            .get_goal_runs(goal_id)
            .await?
            .into_iter()
            .find(|run| run.id == pinned_run_id)
            .ok_or_else(|| anyhow::anyhow!("pinned mandate decision run does not exist"))?;
        anyhow::ensure!(
            run.trigger_type == "mandate" && run.status == "running",
            "pinned mandate decision run is no longer executable"
        );
        let current_run = self
            .state
            .get_current_goal_run(goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("mandate has no active decision run"))?;
        anyhow::ensure!(
            current_run.id == pinned_run_id,
            "pinned mandate decision run is no longer current"
        );
        anyhow::ensure!(
            self.state
                .get_mandate_decision_for_run(&run.id)
                .await?
                .is_none(),
            "this mandate run already has a decision"
        );
        let outcome_raw =
            required_trimmed(args.outcome.as_deref(), "outcome")?.to_ascii_lowercase();
        let outcome = MandateDecisionOutcome::parse(&outcome_raw)
            .ok_or_else(|| anyhow::anyhow!("outcome must be act, wait, ask, or stop"))?;
        // Whether ACT is admissible now (mutation budget, quota, cooldown)
        // is decided by the store when it commits the decision.
        let rationale =
            required_bounded_trimmed(args.rationale.as_deref(), "rationale", MAX_RATIONALE_TEXT)?;
        let mut decision =
            MandateDecisionCycle::new(&mandate.id, &run.id, outcome, rationale, mandate.version);
        decision.activity_level =
            MandateActivityLevel::parse(args.activity_level.as_deref().unwrap_or("quiet"))
                .ok_or_else(|| {
                    anyhow::anyhow!("activity_level must be quiet, active, or urgent")
                })?;
        let observations = clean_strings(args.observations.as_deref());
        validate_bounded_strings(
            &observations,
            "observations",
            MAX_OBSERVATIONS,
            MAX_OBSERVATION_TEXT,
            MAX_OBSERVATIONS_JSON,
        )?;
        if !observations.is_empty() {
            decision.belief_snapshot = Some(serde_json::to_string(&observations)?);
        }
        decision.evidence_receipt_ids = self
            .cited_or_current_run_evidence(args.evidence_receipt_ids.as_deref(), pinned_run_id)
            .await?;
        validate_bounded_strings(
            &decision.evidence_receipt_ids,
            "evidence_receipt_ids",
            MAX_EVIDENCE_RECEIPTS,
            256,
            4 * 1024,
        )?;
        validate_act_evidence(outcome, &observations, &decision.evidence_receipt_ids)?;
        decision.question = args
            .question
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        if let Some(question) = decision.question.as_deref() {
            validate_bounded_text(question, "question", MAX_QUESTION_TEXT)?;
        }
        if outcome == MandateDecisionOutcome::Ask {
            anyhow::ensure!(
                decision.question.is_some(),
                "question is required when outcome is ask"
            );
        }
        if let Some(alert) = args.owner_alert.as_deref() {
            validate_bounded_text(alert, "owner_alert", MAX_QUESTION_TEXT)?;
            anyhow::ensure!(
                outcome == MandateDecisionOutcome::Wait,
                "owner_alert is carried only by a WAIT"
            );
            anyhow::ensure!(
                mandate.authority.owner_alerts,
                "this mandate's confirmed policy does not include telling the owner what it finds; record WAIT without owner_alert, or ASK the owner to confirm a policy update that adds it"
            );
            decision.owner_alert = Some(alert.to_string());
        }
        (decision.termination_kind, decision.termination_match) = normalized_termination_fields(
            outcome,
            args.termination_kind.as_deref(),
            args.termination_match.as_deref(),
        )?;
        let review_secs = args.reconsider_minutes.map_or_else(
            || mandate.review_secs_for_activity(decision.activity_level),
            |minutes| mandate.clamp_review_secs(Some(minutes.saturating_mul(60))),
        );
        if outcome != MandateDecisionOutcome::Stop {
            decision.reconsider_at =
                Some(mandate.bounded_next_review_at(Some(review_secs), chrono::Utc::now()));
        }
        let intention = if outcome == MandateDecisionOutcome::Act {
            let description = required_bounded_trimmed(
                args.intention.as_deref(),
                "intention",
                MAX_INTENTION_TEXT,
            )?;
            let mut intention =
                Intention::new(&mandate.id, &decision.id, &run.id, description, rationale);
            let value_criterion = required_bounded_trimmed(
                args.value_criterion.as_deref(),
                "value_criterion",
                MAX_OBJECTIVE_TEXT,
            )?;
            let criterion_matches = if mandate.success_criteria.is_empty() {
                // Compatibility for mandates created before value criteria
                // became mandatory. Their immutable objective is the only
                // owner-authored value anchor available.
                value_criterion == mandate.objective
            } else {
                mandate
                    .success_criteria
                    .iter()
                    .any(|criterion| criterion == value_criterion)
            };
            anyhow::ensure!(
                criterion_matches,
                "value_criterion must exactly match one owner-confirmed success criterion (or the immutable objective on a legacy mandate)"
            );
            intention.value_criterion = Some(value_criterion.to_string());
            intention.expected_benefit = Some(
                required_bounded_trimmed(
                    args.expected_benefit.as_deref(),
                    "expected_benefit",
                    MAX_INTENTION_METADATA_TEXT,
                )?
                .to_string(),
            );
            intention.risk = Some(
                required_bounded_trimmed(
                    args.risk.as_deref(),
                    "risk",
                    MAX_INTENTION_METADATA_TEXT,
                )?
                .to_string(),
            );
            intention.invalidation_criteria = Some(
                required_bounded_trimmed(
                    args.invalidation_criteria.as_deref(),
                    "invalidation_criteria",
                    MAX_INTENTION_METADATA_TEXT,
                )?
                .to_string(),
            );
            intention
                .validate_value_contract()
                .map_err(anyhow::Error::msg)?;
            let metadata = [
                intention.expected_benefit.as_deref(),
                intention.risk.as_deref(),
                intention.invalidation_criteria.as_deref(),
            ];
            anyhow::ensure!(
                metadata
                    .iter()
                    .flatten()
                    .map(|value| value.chars().count())
                    .sum::<usize>()
                    <= MAX_INTENTION_METADATA_TEXT
                    && metadata
                        .iter()
                        .flatten()
                        .map(|value| value.len())
                        .sum::<usize>()
                        <= MAX_INTENTION_METADATA_TEXT,
                "intention metadata exceeds its combined 4 KiB bound"
            );
            Some(intention)
        } else {
            None
        };
        // Learning notes are optional advisory metadata, never
        // the authority-bearing decision itself. Structured providers can
        // populate only part of an optional field family; do not let that
        // prevent a valid ACT/WAIT/ASK/STOP decision from being committed.
        // A note whose receipt ids are omitted cites the decision's own
        // evidence (learning evidence must be a subset of it, and the model
        // may not see ids the runtime attached), capped at the note's limit.
        let cited_learning_evidence = clean_strings(args.learning_evidence_receipt_ids.as_deref());
        let learning_evidence = if cited_learning_evidence.is_empty() {
            decision
                .evidence_receipt_ids
                .iter()
                .take(MAX_LEARNING_EVIDENCE_RECEIPTS)
                .cloned()
                .collect()
        } else {
            cited_learning_evidence
        };
        let learning_fields_complete =
            learning_note_fields_complete(args.learning_note.as_deref(), Some(&learning_evidence));
        let learning = if learning_fields_complete {
            normalized_learning_note(args.learning_note.as_deref(), Some(&learning_evidence))?
        } else {
            None
        };
        let learning_note = learning.map(|(summary, evidence)| {
            MandateLearningNote::new(
                &mandate.id,
                mandate.version,
                &decision.id,
                &summary,
                evidence,
            )
        });
        let operating_updates = MandateOperatingUpdates { learning_note };
        self.state
            .record_mandate_decision_with_updates(
                &decision,
                intention.as_ref(),
                Some(&operating_updates),
                Some(pinned_attempt_id),
            )
            .await?;
        Ok(match outcome {
            MandateDecisionOutcome::Act => {
                format!(
                    "ACT committed for mandate {}. Create only tasks necessary for this intention: {}",
                    mandate.id,
                    intention
                        .as_ref()
                        .map(|value| value.description.as_str())
                        .unwrap_or("")
                )
            }
            MandateDecisionOutcome::Wait => format!(
                "WAIT recorded. No action tasks should be created; review again at {}.",
                decision
                    .reconsider_at
                    .as_deref()
                    .unwrap_or("the bounded default")
            ),
            MandateDecisionOutcome::Ask => format!(
                "ASK recorded. The mandate is awaiting owner input: {}",
                decision.question.as_deref().unwrap_or("")
            ),
            MandateDecisionOutcome::Stop => {
                "STOP recorded. The mandate has been completed; create no tasks.".to_string()
            }
        })
    }

    /// The summary of the committed decision of the review this call is
    /// fenced to, when that decision is WAIT, ASK or STOP.
    async fn committed_non_action_summary(&self, facts: &ToolCallContext) -> Option<String> {
        let goal_run_id = facts.goal_run_id.as_deref()?;
        let decision = self
            .state
            .get_mandate_decision_for_run(goal_run_id)
            .await
            .ok()
            .flatten()?;
        if Some(decision.mandate_id.as_str()) != facts.mandate_id.as_deref()
            || Some(decision.mandate_version) != facts.mandate_version
        {
            return None;
        }
        decision.non_action_summary()
    }

    /// Run one call with the dispatcher's `facts`.
    pub async fn run(&self, arguments: &str, facts: &ToolCallContext) -> anyhow::Result<String> {
        let mut args: MandateReviewArgs = serde_json::from_str(arguments)?;
        args.normalize();
        args.bind_facts(facts);
        match args.action.as_str() {
            "record_decision" => self.record_decision(&args).await,
            "record_measurement" => self.record_measurement(&args).await,
            other => Ok(format!(
                "Unknown mandate_review action `{other}`. Use record_decision or record_measurement."
            )),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct MandateReviewArgs {
    action: String,
    outcome: Option<String>,
    activity_level: Option<String>,
    rationale: Option<String>,
    observations: Option<Vec<String>>,
    evidence_receipt_ids: Option<Vec<String>>,
    measurement_value_micros: Option<i64>,
    measurement_confidence_bps: Option<u16>,
    measurement_observed_at: Option<String>,
    question: Option<String>,
    owner_alert: Option<String>,
    termination_kind: Option<String>,
    termination_match: Option<String>,
    reconsider_minutes: Option<i64>,
    intention: Option<String>,
    value_criterion: Option<String>,
    expected_benefit: Option<String>,
    risk: Option<String>,
    invalidation_criteria: Option<String>,
    learning_note: Option<String>,
    learning_evidence_receipt_ids: Option<Vec<String>>,
    /// The dispatcher's facts about this call ([`Self::bind_facts`]),
    /// never arguments.
    #[serde(skip)]
    call_session: Option<String>,
    #[serde(skip)]
    call_visibility: Option<ChannelVisibility>,
    #[serde(skip)]
    call_goal_id: Option<String>,
    #[serde(skip)]
    call_goal_run_id: Option<String>,
    #[serde(skip)]
    call_task_attempt_id: Option<String>,
}

impl MandateReviewArgs {
    /// Take the caller's identity and run fence from the dispatcher's facts.
    fn bind_facts(&mut self, facts: &ToolCallContext) {
        self.call_session = Some(facts.session_id.clone());
        self.call_visibility = facts.channel_visibility;
        self.call_goal_id = facts.goal_id.clone();
        self.call_goal_run_id = facts.goal_run_id.clone();
        self.call_task_attempt_id = facts.task_attempt_id.clone();
    }

    /// Some structured-tool providers fill every optional string with "":
    /// an empty field is an absent one.
    fn normalize(&mut self) {
        for value in [
            &mut self.outcome,
            &mut self.activity_level,
            &mut self.rationale,
            &mut self.measurement_observed_at,
            &mut self.question,
            &mut self.owner_alert,
            &mut self.termination_kind,
            &mut self.termination_match,
            &mut self.intention,
            &mut self.value_criterion,
            &mut self.expected_benefit,
            &mut self.risk,
            &mut self.invalidation_criteria,
            &mut self.learning_note,
        ] {
            *value = value
                .take()
                .map(|raw| raw.trim().to_string())
                .filter(|raw| !raw.is_empty());
        }
    }
}

/// The review runs in the mandate's internal task-lead session.
fn internal_call(session: Option<&str>, visibility: Option<ChannelVisibility>) -> bool {
    visibility == Some(ChannelVisibility::Internal)
        || session.is_some_and(|session| {
            session.starts_with("specialist:") || session.starts_with("sub-")
        })
}

fn bounded_text(max_length: usize, description: &str) -> Value {
    json!({ "type": "string", "maxLength": max_length, "description": description })
}

fn text_enum(values: &[&str], description: &str) -> Value {
    json!({ "type": "string", "enum": values, "description": description })
}

fn bounded_text_array(max_items: usize, max_length: usize, description: &str) -> Value {
    json!({
        "type": "array",
        "maxItems": max_items,
        "items": { "type": "string", "maxLength": max_length },
        "description": description
    })
}

#[async_trait]
impl Tool for MandateReviewTool {
    fn name(&self) -> &str {
        MANDATE_REVIEW_TOOL
    }

    fn description(&self) -> &str {
        "Commit this mandate review's one decision (record_decision: act, wait, ask or stop), \
         and for a measured objective first record the metric (record_measurement)."
    }

    fn schema(&self) -> Value {
        json!({
            "name": MANDATE_REVIEW_TOOL,
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": text_enum(REVIEW_ACTIONS, "record_decision once per review; record_measurement before it when the mandate has a measured objective."),
                    "outcome": text_enum(MANDATE_OUTCOMES, "record_decision: the one decision."),
                    "activity_level": text_enum(ACTIVITY_LEVELS, "Next-review cadence: quiet = default, active = sooner, urgent = the confirmed minimum."),
                    "rationale": bounded_text(MAX_RATIONALE_TEXT, "Why this outcome, from this run's observations."),
                    "observations": bounded_text_array(MAX_OBSERVATIONS, MAX_OBSERVATION_TEXT, "What this run observed, one finding per item; ACT needs at least one."),
                    "evidence_receipt_ids": described_array(MAX_EVIDENCE_RECEIPTS, EVIDENCE_RECEIPT_IDS_DESCRIPTION),
                    "question": bounded_text(MAX_QUESTION_TEXT, "ASK only: the one question for the owner."),
                    "owner_alert": bounded_text(MAX_QUESTION_TEXT, OWNER_ALERT_DESCRIPTION),
                    "measurement_value_micros": { "type": "integer", "description": "record_measurement: the metric in millionths of one unit (25 engagements = 25000000); 0 when unreadable." },
                    "measurement_confidence_bps": { "type": "integer", "minimum": 0, "maximum": 10000, "description": "record_measurement: confidence in basis points (10000 = certain; 0 = unreadable)." },
                    "measurement_observed_at": { "type": "string", "description": "record_measurement: RFC3339 time the metric was read; omit for now." },
                    "termination_kind": text_enum(TERMINATION_KINDS, STOP_ONLY_DESCRIPTION),
                    "termination_match": bounded_text(MAX_POLICY_ENTRY_TEXT, STOP_ONLY_DESCRIPTION),
                    "reconsider_minutes": { "type": "integer", "minimum": 1, "description": "Minutes until the next review, clamped to the confirmed bounds; omit to follow activity_level." },
                    "intention": bounded_text(MAX_INTENTION_TEXT, "ACT: the one concrete intervention to carry out."),
                    "value_criterion": bounded_text(MAX_OBJECTIVE_TEXT, VALUE_CRITERION_DESCRIPTION),
                    "expected_benefit": bounded_text(MAX_INTENTION_METADATA_TEXT, EXPECTED_BENEFIT_DESCRIPTION),
                    "risk": bounded_text(MAX_INTENTION_METADATA_TEXT, RISK_DESCRIPTION),
                    "invalidation_criteria": bounded_text(MAX_INTENTION_METADATA_TEXT, INVALIDATION_DESCRIPTION),
                    "learning_note": bounded_text(MAX_LEARNING_NOTE_TEXT, "Optional advisory lesson for later reviews; never changes authority."),
                    "learning_evidence_receipt_ids": described_array(MAX_EVIDENCE_RECEIPTS, LEARNING_EVIDENCE_RECEIPT_IDS_DESCRIPTION)
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
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| value.get("action")?.as_str().map(str::to_string));
        let output = self.run(arguments, facts).await?;
        let outcome_status = if !action.as_deref().is_some_and(moved_action) {
            ToolOutcomeStatus::FailedPermanent
        } else if !internal_call(Some(&facts.session_id), facts.channel_visibility) {
            ToolOutcomeStatus::Blocked
        } else {
            ToolOutcomeStatus::Succeeded
        };
        let mut outcome = ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(outcome_status),
                ..ToolCallMetadata::default()
            },
            finish: None,
        };
        // A committed non-action decision (WAIT, ASK, STOP) completes the
        // fenced review: the run ends on the durable decision's summary,
        // without another model call.
        if action.as_deref() == Some("record_decision")
            && outcome_status == ToolOutcomeStatus::Succeeded
            && let Some(summary) = self.committed_non_action_summary(facts).await
        {
            outcome = outcome.with_finish(aidaemon_core::Finish::succeeded(summary));
        }
        Ok(outcome)
    }
}

fn described_array(max_items: usize, description: &str) -> Value {
    bounded_text_array(max_items, 256, description)
}

#[async_trait]
impl crate::traits::ToolSemantics for MandateReviewTool {
    /// The review's commit is the mandate kernel's to judge (the run's
    /// fence, role and budget); the owner confirmed the policy already.
    async fn owner_approval_request(
        &self,
        _call: &crate::traits::ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> crate::traits::OwnerApprovalNeed {
        crate::traits::OwnerApprovalNeed::NotNeeded
    }

    /// A decision schedules the next review and may start an ACT's work,
    /// so its effects stay undeclared.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::unknown()
    }

    /// Sent on every call of the review run that has it; its calls get the
    /// run's mandate fence.
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            goal_context: crate::traits::GoalContext::MandateRun,
            offering: aidaemon_tools::semantics::Offering::Scoped,
            semantic_scope: Some(aidaemon_tools::semantics::ToolSemanticScope::GoalState),
            ..Default::default()
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Management
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::administrative()
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
}
