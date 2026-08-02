use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::tools::command_risk::{PermissionMode, RiskLevel};
use crate::tools::terminal::ApprovalRequest;
use crate::tools::ApprovalBroker;
use crate::traits::{
    Intention, Mandate, MandateAuthority, MandateDecisionCycle, MandateDecisionOutcome,
    MandateLearningNote, MandateOperationScope, MandateReconciliationResolution, MandateStatus,
    MandateStrategySnapshot, MandateSuspensionKind, MandateTerminationKind, StateStore, Tool,
    ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolRole,
};
use crate::types::{ApprovalKind, ApprovalResponse};

const DEFAULT_MIN_REVIEW_SECS: i64 = 15 * 60;
const DEFAULT_MAX_REVIEW_SECS: i64 = 24 * 60 * 60;
const DEFAULT_REVIEW_SECS: i64 = 4 * 60 * 60;
const MIN_MANDATE_TOKEN_BUDGET: i64 = 10_000;
const MAX_OBJECTIVE_TEXT: usize = 2 * 1024;
const MAX_POLICY_ENTRIES: usize = 16;
const MAX_POLICY_ENTRY_TEXT: usize = 500;
const MAX_POLICY_TEXT: usize = 8 * 1024;
const MAX_RATIONALE_TEXT: usize = 2 * 1024;
const MAX_OBSERVATIONS: usize = 8;
const MAX_OBSERVATION_TEXT: usize = 750;
const MAX_OBSERVATIONS_JSON: usize = 6 * 1024;
const MAX_QUESTION_TEXT: usize = 500;
const MAX_INTENTION_TEXT: usize = 1024;
const MAX_INTENTION_METADATA_TEXT: usize = 4 * 1024;
const MAX_GUIDANCE_ENTRIES: usize = 10;
const MAX_GUIDANCE_ENTRY_TEXT: usize = 1024;
const MAX_GUIDANCE_TEXT: usize = 8 * 1024;
const MAX_LEARNING_NOTE_TEXT: usize = 1024;
const MAX_EVIDENCE_RECEIPTS: usize = 16;
const CREATE_ONLY_FIELD_DESCRIPTION: &str = "Create only; update rejects.";

/// Owner-facing mandate administration and the internal deliberation commit point.
///
/// This tool deliberately does not execute delegated actions. It records the
/// owner's authority envelope and the task lead's ACT/WAIT/ASK/STOP choice; the
/// dispatcher enforces the envelope when an action is attempted.
pub struct ManageMandatesTool {
    state: Arc<dyn StateStore>,
    approval_tx: ApprovalBroker,
    skills_dir: Option<PathBuf>,
}

impl ManageMandatesTool {
    pub fn new(state: Arc<dyn StateStore>, approval_tx: ApprovalBroker) -> Self {
        Self {
            state,
            approval_tx,
            skills_dir: None,
        }
    }

    pub fn with_skills_dir(mut self, skills_dir: Option<PathBuf>) -> Self {
        self.skills_dir = skills_dir;
        self
    }

    fn is_owner(args: &ManageMandatesArgs) -> bool {
        args._user_role
            .as_deref()
            .is_some_and(|role| role.eq_ignore_ascii_case("owner"))
    }

    fn is_internal(args: &ManageMandatesArgs) -> bool {
        args._channel_visibility
            .as_deref()
            .is_some_and(|visibility| visibility.eq_ignore_ascii_case("internal"))
            || args._session_id.as_deref().is_some_and(|session| {
                session.starts_with("specialist:") || session.starts_with("sub-")
            })
    }

    fn is_private_owner_control(args: &ManageMandatesArgs) -> bool {
        Self::is_owner(args)
            && !Self::is_internal(args)
            && args
                ._channel_visibility
                .as_deref()
                .is_some_and(|visibility| visibility.eq_ignore_ascii_case("private"))
    }

    fn owner_session(args: &ManageMandatesArgs) -> anyhow::Result<&str> {
        let session = args
            ._session_id
            .as_deref()
            .map(str::trim)
            .filter(|session| !session.is_empty())
            .ok_or_else(|| anyhow::anyhow!("manage_mandates requires an owner session"))?;
        Ok(session)
    }

    fn authority_from_args(args: &ManageMandatesArgs) -> anyhow::Result<MandateAuthority> {
        if let Some(operation_scopes) = args.operation_scopes.clone() {
            anyhow::ensure!(
                !operation_scopes.is_empty(),
                "operation_scopes must contain at least one exact operation"
            );
            anyhow::ensure!(
                args.allowed_tools.is_none()
                    && args.allowed_mutation_effects.is_none()
                    && args.allowed_target_prefixes.is_none(),
                "operation_scopes replace the legacy independent allowed_tools, allowed_mutation_effects, and allowed_target_prefixes fields; do not send both"
            );
            return Ok(MandateAuthority::from_operation_scopes(
                args.allow_observations.unwrap_or(true),
                operation_scopes,
                args.max_mutating_actions_per_cycle.unwrap_or(0),
                args.max_mutating_actions_per_rolling_24h.unwrap_or(0),
                args.min_seconds_between_mutations.unwrap_or(0),
            ));
        }
        Ok(MandateAuthority {
            allow_observations: args.allow_observations.unwrap_or(true),
            allowed_tools: args.allowed_tools.clone().unwrap_or_default(),
            allowed_mutation_effects: args.allowed_mutation_effects.clone().unwrap_or_default(),
            allowed_target_prefixes: args.allowed_target_prefixes.clone().unwrap_or_default(),
            operation_scopes: Vec::new(),
            max_mutating_actions_per_cycle: args.max_mutating_actions_per_cycle.unwrap_or(0),
            max_mutating_actions_per_rolling_24h: args
                .max_mutating_actions_per_rolling_24h
                .unwrap_or(0),
            min_seconds_between_mutations: args.min_seconds_between_mutations.unwrap_or(0),
        })
    }

    fn resolved_token_budgets(args: &ManageMandatesArgs) -> anyhow::Result<(i64, i64)> {
        let goal = crate::traits::Goal::new_continuous_pending(
            "Mandate draft budget resolution",
            "mandate-draft",
            args.budget_per_cycle,
            args.budget_daily,
        );
        let per_cycle = goal.budget_per_check.unwrap_or_default();
        let daily = goal.budget_daily.unwrap_or_default();
        anyhow::ensure!(
            per_cycle >= MIN_MANDATE_TOKEN_BUDGET,
            "budget_per_cycle is a token budget and must be at least {MIN_MANDATE_TOKEN_BUDGET}; omit it to use the default"
        );
        anyhow::ensure!(
            daily >= MIN_MANDATE_TOKEN_BUDGET,
            "budget_daily is a token budget and must be at least {MIN_MANDATE_TOKEN_BUDGET}; omit it to use the default"
        );
        anyhow::ensure!(
            daily >= per_cycle,
            "budget_daily must be at least budget_per_cycle"
        );
        Ok((per_cycle, daily))
    }

    fn strategy_snapshot(
        &self,
        skill_name: Option<&str>,
    ) -> anyhow::Result<Option<MandateStrategySnapshot>> {
        let Some(skill_name) = skill_name.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        let skills_dir = self.skills_dir.as_ref().ok_or_else(|| {
            anyhow::anyhow!("strategy_skill requires the filesystem skills system")
        })?;
        let skills = crate::skills::load_skills(skills_dir);
        let skill = crate::skills::find_skill_by_name(&skills, skill_name)
            .ok_or_else(|| anyhow::anyhow!("strategy skill `{skill_name}` was not found"))?;
        let canonical = skill.to_markdown();
        let body = crate::tools::sanitize::sanitize_external_content(&skill.body)
            .trim()
            .to_string();
        anyhow::ensure!(
            !body.is_empty(),
            "strategy skill body is empty after sanitization"
        );
        let snapshot = MandateStrategySnapshot {
            skill_name: skill.name.clone(),
            snapshot_version: MandateStrategySnapshot::SCHEMA_VERSION,
            content_sha256: format!("{:x}", Sha256::digest(canonical.as_bytes())),
            description: skill.description.trim().to_string(),
            body,
            source: skill.source.clone(),
        };
        snapshot.validate().map_err(anyhow::Error::msg)?;
        Ok(Some(snapshot))
    }

    fn confirmation_warnings(
        mandate: &Mandate,
        budget_per_cycle: i64,
        budget_daily: i64,
    ) -> Vec<String> {
        vec![
            format!("Objective: {}", mandate.objective),
            format!("Constraints: {}", display_policy(&mandate.constraints)),
            format!(
                "Success criteria: {}",
                display_policy(&mandate.success_criteria)
            ),
            format!("Stop conditions: {}", display_policy(&mandate.stop_conditions)),
            format!(
                "Pinned strategy: {}",
                mandate.strategy.as_ref().map_or_else(
                    || "none".to_string(),
                    |strategy| format!(
                        "{} (sha256:{})",
                        strategy.skill_name,
                        &strategy.content_sha256[..12]
                    )
                )
            ),
            format!(
                "Observations allowed: {}; exact observation/action tools: {}",
                mandate.authority.allow_observations,
                display_allowlist(&mandate.authority.allowed_tools)
            ),
            format!(
                "Allowed mutation effects: {}",
                display_allowlist(&mandate.authority.allowed_mutation_effects)
            ),
            format!(
                "Allowed targets: {}",
                display_target_scope(&mandate.authority.allowed_target_prefixes)
            ),
            format!(
                "Exact operation scopes: {}",
                serde_json::to_string(&mandate.authority.operation_scopes)
                    .unwrap_or_else(|_| "unavailable".to_string())
            ),
            format!(
                "Mutation limits: {} per decision cycle; {} per rolling 24 hours; minimum spacing {} seconds",
                mandate.authority.max_mutating_actions_per_cycle,
                mandate.authority.max_mutating_actions_per_rolling_24h,
                mandate.authority.min_seconds_between_mutations
            ),
            format!(
                "Review interval: {}–{} minutes (default {})",
                mandate.min_review_secs / 60,
                mandate.max_review_secs / 60,
                mandate.default_review_secs / 60
            ),
            format!(
                "Expiration: {}",
                mandate.expires_at.as_deref().unwrap_or("none")
            ),
            format!(
                "Resolved token budgets: {budget_per_cycle} tokens per decision cycle; {budget_daily} tokens per UTC day"
            ),
        ]
    }

    async fn draft(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandate drafts can only be prepared for the owner in a verified private channel."
                    .to_string(),
            );
        }
        let objective =
            required_bounded_trimmed(args.objective.as_deref(), "objective", MAX_OBJECTIVE_TEXT)?;
        let authority = Self::authority_from_args(args)?;
        let (budget_per_cycle, budget_daily) = Self::resolved_token_budgets(args)?;
        let constraints = clean_strings(args.constraints.as_deref());
        let success_criteria = clean_strings(args.success_criteria.as_deref());
        let stop_conditions = clean_strings(args.stop_conditions.as_deref());
        validate_policy_text(&constraints, &success_criteria, &stop_conditions)?;
        let strategy = self.strategy_snapshot(args.strategy_skill.as_deref())?;
        let mut missing = Vec::new();
        if authority.operation_scopes.is_empty()
            && (authority.max_mutating_actions_per_cycle > 0
                || !authority.allowed_tools.is_empty()
                || !authority.allowed_mutation_effects.is_empty()
                || !authority.allowed_target_prefixes.is_empty())
        {
            missing.push("operation_scopes");
        }
        if authority.max_mutating_actions_per_cycle > 0 {
            if authority.allowed_tools.is_empty() {
                missing.push("allowed_tools");
            }
            if authority.allowed_mutation_effects.is_empty() {
                missing.push("allowed_mutation_effects");
            }
            if authority.allowed_target_prefixes.is_empty() {
                missing.push("allowed_target_prefixes");
            }
            if authority.max_mutating_actions_per_rolling_24h == 0 {
                missing.push("max_mutating_actions_per_rolling_24h");
            }
            if authority.min_seconds_between_mutations < 900 {
                missing.push("min_seconds_between_mutations_at_least_900");
            }
        }
        let validation_error = authority.validate().err();
        let ready_to_confirm = missing.is_empty() && validation_error.is_none();
        Ok(serde_json::to_string_pretty(&json!({
            "execution_mode": "ongoing_mandate",
            "writes_performed": false,
            "ready_to_confirm": ready_to_confirm,
            "required_inputs": missing,
            "validation_status": validation_error.as_deref().unwrap_or("ready"),
            "proposal": {
                "objective": objective,
                "constraints": constraints,
                "success_criteria": success_criteria,
                "stop_conditions": stop_conditions,
                "strategy": strategy.as_ref().map(|value| json!({
                    "skill_name": value.skill_name,
                    "content_sha256": value.content_sha256,
                })),
                "authority": authority,
                "token_budgets": {
                    "units": "tokens",
                    "per_decision_cycle": budget_per_cycle,
                    "per_utc_day": budget_daily,
                },
                "review_minutes": {
                    "minimum": args.min_review_minutes.unwrap_or(DEFAULT_MIN_REVIEW_SECS / 60),
                    "default": args.default_review_minutes.unwrap_or(DEFAULT_REVIEW_SECS / 60),
                    "maximum": args.max_review_minutes.unwrap_or(DEFAULT_MAX_REVIEW_SECS / 60),
                },
                "expires_at": args.expires_at,
                "priority": args.priority.as_deref().unwrap_or("high"),
            },
            "next_step": if ready_to_confirm {
                "Call create with this proposal to show the complete owner confirmation."
            } else {
                "Supply every required input, then draft again before create."
            }
        }))?)
    }

    fn reject_create_only_update_fields(args: &ManageMandatesArgs) -> anyhow::Result<()> {
        let mut fields = Vec::new();
        if args.source_goal_id.is_some() {
            fields.push("source_goal_id");
        }
        if args.priority.is_some() {
            fields.push("priority");
        }
        if args.budget_per_cycle.is_some() {
            fields.push("budget_per_cycle");
        }
        if args.budget_daily.is_some() {
            fields.push("budget_daily");
        }
        anyhow::ensure!(
            fields.is_empty(),
            "update does not support create-only fields in v1: {}; create a new mandate to change them",
            fields.join(", ")
        );
        Ok(())
    }

    async fn resolve_owned_mandate(
        &self,
        raw_id: &str,
        owner_session: &str,
    ) -> anyhow::Result<Mandate> {
        let raw_id = raw_id.trim();
        anyhow::ensure!(!raw_id.is_empty(), "mandate_id must not be empty");
        if let Some(mandate) = self.state.get_mandate(raw_id).await? {
            anyhow::ensure!(
                mandate.created_by_session == owner_session,
                "mandate not found in this owner session"
            );
            return Ok(mandate);
        }
        let matches = self
            .state
            .list_mandates(Some(owner_session), true)
            .await?
            .into_iter()
            .filter(|mandate| mandate.id.starts_with(raw_id))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [mandate] => Ok(mandate.clone()),
            [] => anyhow::bail!("mandate not found: {raw_id}"),
            _ => anyhow::bail!("mandate ID prefix is ambiguous; use the full ID"),
        }
    }

    async fn cancel_unconfirmed(&self, mandate: &Mandate) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.state
                .transition_mandate_status(
                    &mandate.id,
                    MandateStatus::Paused,
                    MandateStatus::Cancelled,
                )
                .await?,
            "unconfirmed mandate could not be cancelled"
        );
        Ok(())
    }

    async fn create(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandates can only be created by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let session_id = Self::owner_session(args)?;
        let objective =
            required_bounded_trimmed(args.objective.as_deref(), "objective", MAX_OBJECTIVE_TEXT)?;
        let authority = Self::authority_from_args(args)?;
        anyhow::ensure!(
            authority.uses_operation_scopes()
                || (authority.allowed_tools.is_empty()
                    && authority.allowed_mutation_effects.is_empty()
                    && authority.allowed_target_prefixes.is_empty()
                    && authority.max_mutating_actions_per_cycle == 0),
            "delegated operations require operation_scopes from a validated draft; legacy independent authority lists cannot create new mandates"
        );
        authority.validate().map_err(anyhow::Error::msg)?;

        let min_review_secs = minutes_to_secs(
            args.min_review_minutes,
            DEFAULT_MIN_REVIEW_SECS,
            "min_review_minutes",
        )?;
        let max_review_secs = minutes_to_secs(
            args.max_review_minutes,
            DEFAULT_MAX_REVIEW_SECS,
            "max_review_minutes",
        )?;
        let default_review_secs = minutes_to_secs(
            args.default_review_minutes,
            DEFAULT_REVIEW_SECS,
            "default_review_minutes",
        )?;
        anyhow::ensure!(
            min_review_secs <= default_review_secs && default_review_secs <= max_review_secs,
            "review bounds must satisfy min <= default <= max"
        );

        let mut goal = crate::traits::Goal::new_continuous_pending(
            &format!("Mandate: {objective}"),
            session_id,
            args.budget_per_cycle,
            args.budget_daily,
        );
        let (budget_per_cycle, budget_daily) = Self::resolved_token_budgets(args)?;
        goal.priority = args.priority.clone().unwrap_or_else(|| "high".to_string());
        anyhow::ensure!(
            matches!(
                goal.priority.as_str(),
                "low" | "medium" | "high" | "critical"
            ),
            "priority must be low, medium, high, or critical"
        );

        let mut mandate = Mandate::new(
            &goal.id,
            args.source_goal_id.clone(),
            objective,
            session_id,
            authority,
            min_review_secs,
            max_review_secs,
            default_review_secs,
        );
        // Pending/paused records are safe if approval delivery is interrupted.
        // Once activated, this timestamp makes the first deliberation immediately due.
        mandate.status = MandateStatus::Paused;
        mandate.confirmed_at = None;
        mandate.next_review_at = chrono::Utc::now().to_rfc3339();
        mandate.constraints = clean_strings(args.constraints.as_deref());
        mandate.success_criteria = clean_strings(args.success_criteria.as_deref());
        mandate.stop_conditions = clean_strings(args.stop_conditions.as_deref());
        mandate.strategy = self.strategy_snapshot(args.strategy_skill.as_deref())?;
        validate_policy_text(
            &mandate.constraints,
            &mandate.success_criteria,
            &mandate.stop_conditions,
        )?;
        mandate.expires_at = args.expires_at.clone();
        if let Some(expires_at) = mandate.expires_at.as_deref() {
            let parsed = chrono::DateTime::parse_from_rfc3339(expires_at)
                .map_err(|_| anyhow::anyhow!("expires_at must be an RFC3339 timestamp"))?;
            anyhow::ensure!(
                parsed > chrono::Utc::now(),
                "expires_at must be in the future"
            );
        }
        goal.conditions = if mandate.success_criteria.is_empty() {
            None
        } else {
            Some(mandate.success_criteria.join("\n"))
        };
        goal.context = Some(serde_json::to_string(&json!({
            "mandate_id": &mandate.id,
            "source_goal_id": &mandate.source_goal_id,
        }))?);

        self.state
            .create_mandate_controller(&goal, &mandate)
            .await?;

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let warnings = Self::confirmation_warnings(&mandate, budget_per_cycle, budget_daily);
        let request = ApprovalRequest {
            command: format!("Delegate mandate: {objective}"),
            session_id: session_id.to_string(),
            risk_level: if mandate.authority.max_mutating_actions_per_cycle == 0 {
                RiskLevel::Medium
            } else {
                RiskLevel::High
            },
            warnings,
            permission_mode: PermissionMode::Cautious,
            response_tx,
            kind: ApprovalKind::GoalConfirmation,
        };
        if let Err(error) = self.approval_tx.send(request).await {
            self.cancel_unconfirmed(&mandate).await?;
            return Ok(format!(
                "Mandate confirmation could not be delivered, so the pending mandate was cancelled: {error}"
            ));
        }

        match response_rx.await {
            Ok(
                ApprovalResponse::AllowOnce
                | ApprovalResponse::AllowSession
                | ApprovalResponse::AllowAlways,
            ) => {
                anyhow::ensure!(
                    self.state.confirm_mandate(&mandate.id).await?,
                    "mandate could not be activated"
                );
                Ok(format!(
                    "Activated mandate {}. Its first review is due now; future reviews are chosen within {}–{} minutes.",
                    mandate.id,
                    mandate.min_review_secs / 60,
                    mandate.max_review_secs / 60
                ))
            }
            Ok(ApprovalResponse::Deny) => {
                self.cancel_unconfirmed(&mandate).await?;
                Ok(format!(
                    "Cancelled mandate {}; confirmation was declined.",
                    mandate.id
                ))
            }
            Err(_) => {
                self.cancel_unconfirmed(&mandate).await?;
                Ok(format!(
                    "Mandate {} was cancelled because the confirmation response became unavailable.",
                    mandate.id
                ))
            }
        }
    }

    async fn list(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandates can only be listed by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let session_id = Self::owner_session(args)?;
        let mandates = self
            .state
            .list_mandates(Some(session_id), args.include_terminal.unwrap_or(false))
            .await?;
        if mandates.is_empty() {
            return Ok("No mandates in this owner session.".to_string());
        }
        let mut output = format!("Mandates ({}):\n", mandates.len());
        for mandate in mandates {
            output.push_str(&format!(
                "- {} [{} v{}] {} — next review {}; {}/cycle, {}/rolling-24h mutations\n",
                mandate.id,
                mandate.status,
                mandate.version,
                mandate.objective,
                mandate.next_review_at,
                mandate.authority.max_mutating_actions_per_cycle,
                mandate.authority.max_mutating_actions_per_rolling_24h
            ));
        }
        Ok(output)
    }

    async fn get(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandates can only be inspected by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let owner = Self::owner_session(args)?;
        let id = required_trimmed(args.mandate_id.as_deref(), "mandate_id")?;
        let mandate = self.resolve_owned_mandate(id, owner).await?;
        let decisions = self.state.list_mandate_decisions(&mandate.id, 5).await?;
        let recent = self.state.list_intentions(&mandate.id, 5).await?;
        let learning = self
            .state
            .list_mandate_learning_notes(&mandate.id, 10)
            .await?;
        let mut mutation_receipts = Vec::new();
        for decision in &decisions {
            mutation_receipts.extend(
                self.state
                    .list_mandate_mutation_attempts_for_run(&decision.goal_run_id)
                    .await?,
            );
        }
        Ok(serde_json::to_string_pretty(&json!({
            "mandate": mandate,
            "recent_decisions": decisions,
            "recent_intentions": recent,
            "recent_learning_notes": learning,
            "recent_mutation_receipts": mutation_receipts,
        }))?)
    }

    async fn transition(&self, args: &ManageMandatesArgs, action: &str) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandates can only be changed by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let owner = Self::owner_session(args)?;
        let id = required_trimmed(args.mandate_id.as_deref(), "mandate_id")?;
        let mandate = self.resolve_owned_mandate(id, owner).await?;
        anyhow::ensure!(
            !mandate.status.is_terminal(),
            "terminal mandates cannot be changed"
        );
        if !matches!(
            action,
            "resume" | "answer_question" | "resolve_reconciliation"
        ) {
            anyhow::ensure!(
                args.guidance.is_none(),
                "guidance is valid only with resume, answer_question, or resolve_reconciliation"
            );
        }
        if matches!(
            action,
            "resume" | "answer_question" | "resolve_reconciliation"
        ) {
            anyhow::ensure!(
                mandate.confirmed_at.is_some(),
                "an unconfirmed mandate cannot be resumed; cancel it and create a fresh confirmed mandate"
            );
        }
        let controller = if matches!(
            action,
            "resume" | "answer_question" | "resolve_reconciliation"
        ) {
            Some(
                self.state
                    .get_goal(&mandate.goal_id)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("mandate controller goal is missing"))?,
            )
        } else {
            None
        };
        let guidance = args
            .guidance
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if let Some(guidance) = guidance {
            validate_bounded_text(guidance, "guidance", MAX_GUIDANCE_ENTRY_TEXT)?;
        }
        let resumed_context = guidance
            .map(|guidance| {
                append_owner_guidance(
                    controller.as_ref().and_then(|goal| goal.context.as_deref()),
                    guidance,
                )
            })
            .transpose()?;

        if action == "answer_question" || action == "resolve_reconciliation" {
            anyhow::ensure!(
                mandate.status == MandateStatus::AwaitingInput,
                "{action} requires an awaiting-input mandate"
            );
            let suspension = mandate
                .suspension
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("awaiting-input mandate has no typed suspension"))?;
            let expected_kind = suspension.kind;
            let resolution = if action == "answer_question" {
                anyhow::ensure!(
                    expected_kind == MandateSuspensionKind::AwaitingAnswer,
                    "this mandate is awaiting safety reconciliation, not a question answer"
                );
                anyhow::ensure!(
                    guidance.is_some(),
                    "guidance is required for answer_question"
                );
                None
            } else {
                anyhow::ensure!(
                    expected_kind != MandateSuspensionKind::AwaitingAnswer,
                    "this mandate is awaiting an answer; use answer_question"
                );
                let raw = required_trimmed(
                    args.reconciliation_resolution.as_deref(),
                    "reconciliation_resolution",
                )?;
                Some(MandateReconciliationResolution::parse(raw).ok_or_else(|| {
                    anyhow::anyhow!(
                        "reconciliation_resolution must be confirmed_effect_occurred, confirmed_no_effect, or abandon_attempt"
                    )
                })?)
            };
            let guidance =
                guidance.ok_or_else(|| anyhow::anyhow!("guidance is required for {action}"))?;
            anyhow::ensure!(
                self.state
                    .resolve_mandate_suspension(
                        &mandate.id,
                        mandate.version,
                        expected_kind,
                        resumed_context.as_deref(),
                        resolution,
                        guidance,
                        owner,
                    )
                    .await?,
                "mandate suspension changed before it could be resolved"
            );
            return Ok(format!(
                "Mandate {} is active after typed {}.",
                mandate.id, action
            ));
        }

        let (from, to) = match action {
            "pause" => (MandateStatus::Active, MandateStatus::Paused),
            "resume" => {
                anyhow::ensure!(
                    mandate.status == MandateStatus::Paused
                        && mandate.suspension.as_ref().is_some_and(|value| {
                            value.kind == MandateSuspensionKind::OwnerPaused
                        }),
                    "resume is only for owner-paused mandates; use answer_question or resolve_reconciliation for awaiting-input states"
                );
                (MandateStatus::Paused, MandateStatus::Active)
            }
            "cancel" => (mandate.status, MandateStatus::Cancelled),
            _ => unreachable!(),
        };
        let transitioned = if action == "resume" {
            self.state
                .resume_mandate_with_context(
                    &mandate.id,
                    from,
                    mandate.version,
                    resumed_context.as_deref(),
                )
                .await?
        } else {
            self.state
                .transition_mandate_status(&mandate.id, from, to)
                .await?
        };
        anyhow::ensure!(
            transitioned,
            "mandate is {}, so it cannot be changed with {action}",
            mandate.status
        );
        let current = self
            .state
            .get_mandate(&mandate.id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("mandate disappeared after transition"))?;
        Ok(format!("Mandate {} is now {}.", mandate.id, current.status))
    }

    async fn update(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandate authority can only be changed by the owner in a verified private channel."
                    .to_string(),
            );
        }
        Self::reject_create_only_update_fields(args)?;
        let owner = Self::owner_session(args)?;
        let id = required_trimmed(args.mandate_id.as_deref(), "mandate_id")?;
        let mut mandate = self.resolve_owned_mandate(id, owner).await?;
        let has_change = args.objective.is_some()
            || args.allow_observations.is_some()
            || args.operation_scopes.is_some()
            || args.allowed_tools.is_some()
            || args.allowed_mutation_effects.is_some()
            || args.allowed_target_prefixes.is_some()
            || args.max_mutating_actions_per_cycle.is_some()
            || args.max_mutating_actions_per_rolling_24h.is_some()
            || args.min_seconds_between_mutations.is_some()
            || args.constraints.is_some()
            || args.success_criteria.is_some()
            || args.stop_conditions.is_some()
            || args.min_review_minutes.is_some()
            || args.max_review_minutes.is_some()
            || args.default_review_minutes.is_some()
            || args.expires_at.is_some()
            || args.strategy_skill.is_some()
            || args.clear_strategy.unwrap_or(false);
        anyhow::ensure!(has_change, "update requires at least one changed field");
        anyhow::ensure!(
            !mandate.status.is_terminal(),
            "terminal mandates cannot be updated"
        );
        if let Some(objective) = args.objective.as_deref() {
            mandate.objective =
                required_bounded_trimmed(Some(objective), "objective", MAX_OBJECTIVE_TEXT)?
                    .to_string();
        }
        if args.allow_observations.is_some()
            || args.operation_scopes.is_some()
            || args.allowed_tools.is_some()
            || args.allowed_mutation_effects.is_some()
            || args.allowed_target_prefixes.is_some()
            || args.max_mutating_actions_per_cycle.is_some()
            || args.max_mutating_actions_per_rolling_24h.is_some()
            || args.min_seconds_between_mutations.is_some()
        {
            let scopes = args
                .operation_scopes
                .clone()
                .unwrap_or_else(|| mandate.authority.operation_scopes.clone());
            if !scopes.is_empty() {
                anyhow::ensure!(
                    args.allowed_tools.is_none()
                        && args.allowed_mutation_effects.is_none()
                        && args.allowed_target_prefixes.is_none(),
                    "operation-scoped mandates cannot be updated through legacy independent authority lists"
                );
                mandate.authority = MandateAuthority::from_operation_scopes(
                    args.allow_observations
                        .unwrap_or(mandate.authority.allow_observations),
                    scopes,
                    args.max_mutating_actions_per_cycle
                        .unwrap_or(mandate.authority.max_mutating_actions_per_cycle),
                    args.max_mutating_actions_per_rolling_24h
                        .unwrap_or(mandate.authority.max_mutating_actions_per_rolling_24h),
                    args.min_seconds_between_mutations
                        .unwrap_or(mandate.authority.min_seconds_between_mutations),
                );
            } else {
                mandate.authority = MandateAuthority {
                    allow_observations: args
                        .allow_observations
                        .unwrap_or(mandate.authority.allow_observations),
                    allowed_tools: args
                        .allowed_tools
                        .clone()
                        .unwrap_or(mandate.authority.allowed_tools),
                    allowed_mutation_effects: args
                        .allowed_mutation_effects
                        .clone()
                        .unwrap_or(mandate.authority.allowed_mutation_effects),
                    allowed_target_prefixes: args
                        .allowed_target_prefixes
                        .clone()
                        .unwrap_or(mandate.authority.allowed_target_prefixes),
                    operation_scopes: Vec::new(),
                    max_mutating_actions_per_cycle: args
                        .max_mutating_actions_per_cycle
                        .unwrap_or(mandate.authority.max_mutating_actions_per_cycle),
                    max_mutating_actions_per_rolling_24h: args
                        .max_mutating_actions_per_rolling_24h
                        .unwrap_or(mandate.authority.max_mutating_actions_per_rolling_24h),
                    min_seconds_between_mutations: args
                        .min_seconds_between_mutations
                        .unwrap_or(mandate.authority.min_seconds_between_mutations),
                };
            }
        }
        if let Some(values) = args.constraints.as_deref() {
            mandate.constraints = clean_strings(Some(values));
        }
        if let Some(values) = args.success_criteria.as_deref() {
            mandate.success_criteria = clean_strings(Some(values));
        }
        if let Some(values) = args.stop_conditions.as_deref() {
            mandate.stop_conditions = clean_strings(Some(values));
        }
        anyhow::ensure!(
            !(args.strategy_skill.is_some() && args.clear_strategy.unwrap_or(false)),
            "strategy_skill and clear_strategy are mutually exclusive"
        );
        if args.clear_strategy.unwrap_or(false) {
            mandate.strategy = None;
        } else if args.strategy_skill.is_some() {
            mandate.strategy = self.strategy_snapshot(args.strategy_skill.as_deref())?;
        }
        validate_policy_text(
            &mandate.constraints,
            &mandate.success_criteria,
            &mandate.stop_conditions,
        )?;
        mandate.min_review_secs = minutes_to_secs(
            args.min_review_minutes,
            mandate.min_review_secs,
            "min_review_minutes",
        )?;
        mandate.max_review_secs = minutes_to_secs(
            args.max_review_minutes,
            mandate.max_review_secs,
            "max_review_minutes",
        )?;
        mandate.default_review_secs = minutes_to_secs(
            args.default_review_minutes,
            mandate.default_review_secs,
            "default_review_minutes",
        )?;
        anyhow::ensure!(
            mandate.min_review_secs <= mandate.default_review_secs
                && mandate.default_review_secs <= mandate.max_review_secs,
            "review bounds must satisfy min <= default <= max"
        );
        if let Some(expires_at) = args.expires_at.as_ref() {
            let parsed = chrono::DateTime::parse_from_rfc3339(expires_at)
                .map_err(|_| anyhow::anyhow!("expires_at must be an RFC3339 timestamp"))?;
            anyhow::ensure!(
                parsed > chrono::Utc::now(),
                "expires_at must be in the future"
            );
            mandate.expires_at = Some(expires_at.clone());
        }
        mandate.authority.validate().map_err(anyhow::Error::msg)?;

        let controller = self
            .state
            .get_goal(&mandate.goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("mandate controller goal is missing"))?;
        let warnings = Self::confirmation_warnings(
            &mandate,
            controller.budget_per_check.unwrap_or_default(),
            controller.budget_daily.unwrap_or_default(),
        );
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.approval_tx
            .send(ApprovalRequest {
                command: format!("Update delegated mandate: {}", mandate.objective),
                session_id: owner.to_string(),
                risk_level: RiskLevel::High,
                warnings,
                permission_mode: PermissionMode::Cautious,
                response_tx,
                kind: ApprovalKind::GoalConfirmation,
            })
            .await
            .map_err(|error| anyhow::anyhow!("mandate update confirmation unavailable: {error}"))?;
        match response_rx.await {
            Ok(
                ApprovalResponse::AllowOnce
                | ApprovalResponse::AllowSession
                | ApprovalResponse::AllowAlways,
            ) => {}
            Ok(ApprovalResponse::Deny) => {
                return Ok(
                    "Mandate update cancelled; the existing envelope is unchanged.".to_string(),
                )
            }
            Err(_) => anyhow::bail!("mandate update confirmation became unavailable"),
        }
        mandate.version += 1;
        mandate.updated_at = chrono::Utc::now().to_rfc3339();
        self.state.update_mandate(&mandate).await?;
        Ok(format!(
            "Updated mandate {} to policy version {}. In-flight decisions on older versions are revoked.",
            mandate.id, mandate.version
        ))
    }

    async fn record_decision(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_internal(args) {
            return Ok(
                "record_decision is reserved for the internal mandate deliberator.".to_string(),
            );
        }
        let goal_id = required_trimmed(args._goal_id.as_deref(), "internal _goal_id")?;
        let pinned_run_id =
            required_trimmed(args._goal_run_id.as_deref(), "internal _goal_run_id")?;
        let pinned_attempt_id = required_trimmed(
            args._task_attempt_id.as_deref(),
            "internal _task_attempt_id",
        )?;
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
        anyhow::ensure!(
            outcome != MandateDecisionOutcome::Act
                || mandate.authority.max_mutating_actions_per_cycle > 0,
            "ACT requires a positive governed mutation budget; choose WAIT, ASK, or STOP for an observation-only cycle"
        );
        if outcome == MandateDecisionOutcome::Act {
            let quota = self
                .state
                .get_mandate_mutation_quota_state(&mandate.id, &chrono::Utc::now().to_rfc3339())
                .await?
                .ok_or_else(|| anyhow::anyhow!("mandate mutation quota state is missing"))?;
            if !quota.available_now {
                let earliest = quota
                    .earliest_next_slot_at
                    .as_deref()
                    .unwrap_or("unavailable");
                anyhow::bail!(
                    "mandate_mutation_quota_unavailable: choose WAIT; earliest_next_slot_at={earliest}"
                );
            }
        }
        let rationale =
            required_bounded_trimmed(args.rationale.as_deref(), "rationale", MAX_RATIONALE_TEXT)?;
        let mut decision =
            MandateDecisionCycle::new(&mandate.id, &run.id, outcome, rationale, mandate.version);
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
        decision.evidence_receipt_ids = clean_strings(args.evidence_receipt_ids.as_deref());
        validate_bounded_strings(
            &decision.evidence_receipt_ids,
            "evidence_receipt_ids",
            MAX_EVIDENCE_RECEIPTS,
            256,
            4 * 1024,
        )?;
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
        decision.termination_kind = args
            .termination_kind
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                MandateTerminationKind::parse(value).ok_or_else(|| {
                    anyhow::anyhow!(
                        "termination_kind must be success_criteria_satisfied, stop_condition_met, or safety_termination"
                    )
                })
            })
            .transpose()?;
        decision.termination_match = bounded_optional_text(
            args.termination_match.as_deref(),
            "termination_match",
            MAX_POLICY_ENTRY_TEXT,
        )?;
        if outcome == MandateDecisionOutcome::Stop {
            anyhow::ensure!(
                decision.termination_kind.is_some(),
                "STOP requires termination_kind"
            );
        } else {
            anyhow::ensure!(
                decision.termination_kind.is_none() && decision.termination_match.is_none(),
                "termination fields are valid only with STOP"
            );
        }
        let review_secs = mandate.clamp_review_secs(
            args.reconsider_minutes
                .map(|minutes| minutes.saturating_mul(60)),
        );
        if outcome != MandateDecisionOutcome::Stop {
            decision.reconsider_at =
                Some((chrono::Utc::now() + chrono::Duration::seconds(review_secs)).to_rfc3339());
        }
        let intention = if outcome == MandateDecisionOutcome::Act {
            let description = required_bounded_trimmed(
                args.intention.as_deref(),
                "intention",
                MAX_INTENTION_TEXT,
            )?;
            let mut intention =
                Intention::new(&mandate.id, &decision.id, &run.id, description, rationale);
            intention.expected_benefit = bounded_optional_text(
                args.expected_benefit.as_deref(),
                "expected_benefit",
                MAX_INTENTION_METADATA_TEXT,
            )?;
            intention.risk =
                bounded_optional_text(args.risk.as_deref(), "risk", MAX_INTENTION_METADATA_TEXT)?;
            intention.invalidation_criteria = bounded_optional_text(
                args.invalidation_criteria.as_deref(),
                "invalidation_criteria",
                MAX_INTENTION_METADATA_TEXT,
            )?;
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
        self.state
            .record_mandate_decision(&decision, intention.as_ref(), Some(pinned_attempt_id))
            .await?;
        if let Some(summary) = bounded_optional_text(
            args.learning_note.as_deref(),
            "learning_note",
            MAX_LEARNING_NOTE_TEXT,
        )? {
            let evidence = clean_strings(args.learning_evidence_receipt_ids.as_deref());
            validate_bounded_strings(
                &evidence,
                "learning_evidence_receipt_ids",
                MAX_EVIDENCE_RECEIPTS,
                256,
                4 * 1024,
            )?;
            anyhow::ensure!(
                !evidence.is_empty(),
                "learning_note requires learning_evidence_receipt_ids"
            );
            let note = MandateLearningNote::new(
                &mandate.id,
                mandate.version,
                &decision.id,
                &summary,
                evidence,
            );
            self.state.record_mandate_learning_note(&note).await?;
        } else {
            anyhow::ensure!(
                args.learning_evidence_receipt_ids.is_none(),
                "learning_evidence_receipt_ids require learning_note"
            );
        }
        Ok(match outcome {
            MandateDecisionOutcome::Act => {
                format!(
                "ACT committed for mandate {}. Create only tasks necessary for this intention: {}",
                mandate.id,
                intention.as_ref().map(|value| value.description.as_str()).unwrap_or("")
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

    async fn list_intentions(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandate intentions can only be inspected by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let owner = Self::owner_session(args)?;
        let id = required_trimmed(args.mandate_id.as_deref(), "mandate_id")?;
        let mandate = self.resolve_owned_mandate(id, owner).await?;
        let intentions = self
            .state
            .list_intentions(&mandate.id, args.limit.unwrap_or(20).clamp(1, 100))
            .await?;
        Ok(serde_json::to_string_pretty(&intentions)?)
    }
}

#[derive(Debug, Deserialize)]
struct ManageMandatesArgs {
    action: String,
    mandate_id: Option<String>,
    objective: Option<String>,
    source_goal_id: Option<String>,
    allow_observations: Option<bool>,
    operation_scopes: Option<Vec<MandateOperationScope>>,
    allowed_tools: Option<Vec<String>>,
    allowed_mutation_effects: Option<Vec<String>>,
    allowed_target_prefixes: Option<Vec<String>>,
    max_mutating_actions_per_cycle: Option<u32>,
    max_mutating_actions_per_rolling_24h: Option<u32>,
    min_seconds_between_mutations: Option<u32>,
    constraints: Option<Vec<String>>,
    success_criteria: Option<Vec<String>>,
    stop_conditions: Option<Vec<String>>,
    strategy_skill: Option<String>,
    clear_strategy: Option<bool>,
    min_review_minutes: Option<i64>,
    max_review_minutes: Option<i64>,
    default_review_minutes: Option<i64>,
    expires_at: Option<String>,
    priority: Option<String>,
    budget_per_cycle: Option<i64>,
    budget_daily: Option<i64>,
    include_terminal: Option<bool>,
    limit: Option<i64>,
    outcome: Option<String>,
    rationale: Option<String>,
    observations: Option<Vec<String>>,
    evidence_receipt_ids: Option<Vec<String>>,
    question: Option<String>,
    termination_kind: Option<String>,
    termination_match: Option<String>,
    reconsider_minutes: Option<i64>,
    intention: Option<String>,
    expected_benefit: Option<String>,
    risk: Option<String>,
    invalidation_criteria: Option<String>,
    learning_note: Option<String>,
    learning_evidence_receipt_ids: Option<Vec<String>>,
    guidance: Option<String>,
    reconciliation_resolution: Option<String>,
    #[serde(default)]
    _session_id: Option<String>,
    #[serde(default)]
    _user_role: Option<String>,
    #[serde(default)]
    _channel_visibility: Option<String>,
    #[serde(default)]
    _goal_id: Option<String>,
    #[serde(default)]
    _goal_run_id: Option<String>,
    #[serde(default)]
    _task_attempt_id: Option<String>,
}

#[async_trait]
impl Tool for ManageMandatesTool {
    fn name(&self) -> &str {
        "manage_mandates"
    }

    fn description(&self) -> &str {
        "Manage ongoing mandates. Draft before create. Delegated calls require exact non-combinable operation scopes. Budgets are tokens; ASK, reconciliation, STOP, and learning use typed evidence."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_mandates",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["draft", "create", "list", "get", "update", "pause", "resume", "answer_question", "resolve_reconciliation", "cancel", "record_decision", "list_intentions"] },
                    "mandate_id": { "type": "string", "maxLength": 256 },
                    "objective": { "type": "string", "maxLength": MAX_OBJECTIVE_TEXT },
                    "source_goal_id": { "type": "string", "maxLength": 256, "description": CREATE_ONLY_FIELD_DESCRIPTION },
                    "allow_observations": { "type": "boolean" },
                    "operation_scopes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 64,
                        "description": "Exact tool + operation + kind + targets + effects tuples; authenticated scopes pair auth_profile and account targets.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": { "type": "string", "enum": ["http_request", "web_fetch"] },
                                "operation": { "type": "string", "enum": ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"] },
                                "kind": { "type": "string", "enum": ["observation", "mutation"] },
                                "target_prefixes": { "type": "array", "minItems": 1, "maxItems": 16, "items": { "type": "string", "maxLength": 2048 } },
                                "mutation_effects": { "type": "array", "maxItems": 2, "items": { "type": "string", "enum": ["remote_mutation", "external_delivery"] } }
                            },
                            "required": ["tool", "operation", "kind", "target_prefixes", "mutation_effects"],
                            "additionalProperties": false
                        }
                    },
                    "max_mutating_actions_per_cycle": { "type": "integer", "minimum": 0, "maximum": 24 },
                    "max_mutating_actions_per_rolling_24h": { "type": "integer", "minimum": 0, "maximum": 24 },
                    "min_seconds_between_mutations": { "type": "integer", "minimum": 0 },
                    "constraints": { "type": "array", "maxItems": MAX_POLICY_ENTRIES, "items": { "type": "string", "maxLength": MAX_POLICY_ENTRY_TEXT } },
                    "success_criteria": { "type": "array", "maxItems": MAX_POLICY_ENTRIES, "items": { "type": "string", "maxLength": MAX_POLICY_ENTRY_TEXT } },
                    "stop_conditions": { "type": "array", "maxItems": MAX_POLICY_ENTRIES, "items": { "type": "string", "maxLength": MAX_POLICY_ENTRY_TEXT } },
                    "strategy_skill": { "type": "string", "maxLength": 256 },
                    "clear_strategy": { "type": "boolean" },
                    "min_review_minutes": { "type": "integer", "minimum": 1 },
                    "max_review_minutes": { "type": "integer", "minimum": 1 },
                    "default_review_minutes": { "type": "integer", "minimum": 1 },
                    "expires_at": { "type": "string" },
                    "priority": { "type": "string", "enum": ["low", "medium", "high", "critical"], "description": CREATE_ONLY_FIELD_DESCRIPTION },
                    "budget_per_cycle": { "type": "integer", "minimum": MIN_MANDATE_TOKEN_BUDGET, "description": "Create only; update rejects. Cycle token limit; default 400000." },
                    "budget_daily": { "type": "integer", "minimum": MIN_MANDATE_TOKEN_BUDGET, "description": "Create only; update rejects. UTC-day token limit; default 1000000." },
                    "include_terminal": { "type": "boolean" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                    "outcome": { "type": "string", "enum": ["act", "wait", "ask", "stop"] },
                    "rationale": { "type": "string", "maxLength": MAX_RATIONALE_TEXT },
                    "observations": { "type": "array", "maxItems": MAX_OBSERVATIONS, "items": { "type": "string", "maxLength": MAX_OBSERVATION_TEXT } },
                    "evidence_receipt_ids": { "type": "array", "maxItems": MAX_EVIDENCE_RECEIPTS, "items": { "type": "string", "maxLength": 256 } },
                    "question": { "type": "string", "maxLength": MAX_QUESTION_TEXT },
                    "termination_kind": { "type": "string", "enum": ["success_criteria_satisfied", "stop_condition_met", "safety_termination"] },
                    "termination_match": { "type": "string", "maxLength": MAX_POLICY_ENTRY_TEXT },
                    "reconsider_minutes": { "type": "integer", "minimum": 1 },
                    "intention": { "type": "string", "maxLength": MAX_INTENTION_TEXT },
                    "expected_benefit": { "type": "string", "maxLength": MAX_INTENTION_METADATA_TEXT },
                    "risk": { "type": "string", "maxLength": MAX_INTENTION_METADATA_TEXT },
                    "invalidation_criteria": { "type": "string", "maxLength": MAX_INTENTION_METADATA_TEXT },
                    "learning_note": { "type": "string", "maxLength": MAX_LEARNING_NOTE_TEXT },
                    "learning_evidence_receipt_ids": { "type": "array", "maxItems": MAX_EVIDENCE_RECEIPTS, "items": { "type": "string", "maxLength": 256 } },
                    "guidance": { "type": "string", "maxLength": MAX_GUIDANCE_ENTRY_TEXT },
                    "reconciliation_resolution": { "type": "string", "enum": ["confirmed_effect_occurred", "confirmed_no_effect", "abandon_attempt"] }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: ManageMandatesArgs = serde_json::from_str(arguments)?;
        match args.action.as_str() {
            "draft" => self.draft(&args).await,
            "create" => self.create(&args).await,
            "list" => self.list(&args).await,
            "get" => self.get(&args).await,
            "update" => self.update(&args).await,
            "pause" | "resume" | "answer_question" | "resolve_reconciliation" | "cancel" => {
                self.transition(&args, &args.action).await
            }
            "record_decision" => self.record_decision(&args).await,
            "list_intentions" => self.list_intentions(&args).await,
            other => anyhow::bail!("unknown manage_mandates action `{other}`"),
        }
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Management
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            });
        match action.as_deref() {
            Some("draft" | "list" | "get" | "list_intentions") => ToolCallSemantics::observation(),
            Some("record_decision") => ToolCallSemantics::administrative(),
            _ => ToolCallSemantics::mutation_with(ToolMutationEffects::CONFIGURATION),
        }
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

fn required_trimmed<'a>(value: Option<&'a str>, name: &str) -> anyhow::Result<&'a str> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{name} is required"))
}

fn required_bounded_trimmed<'a>(
    value: Option<&'a str>,
    name: &str,
    max_text: usize,
) -> anyhow::Result<&'a str> {
    let value = required_trimmed(value, name)?;
    validate_bounded_text(value, name, max_text)?;
    Ok(value)
}

fn validate_bounded_text(value: &str, name: &str, max_text: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.chars().count() <= max_text && value.len() <= max_text,
        "{name} exceeds its {max_text}-character/byte bound"
    );
    Ok(())
}

fn bounded_optional_text(
    value: Option<&str>,
    name: &str,
    max_text: usize,
) -> anyhow::Result<Option<String>> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    validate_bounded_text(value, name, max_text)?;
    Ok(Some(value.to_string()))
}

fn clean_strings(values: Option<&[String]>) -> Vec<String> {
    values
        .unwrap_or_default()
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn validate_bounded_strings(
    values: &[String],
    name: &str,
    max_entries: usize,
    max_entry_text: usize,
    max_serialized_bytes: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        values.len() <= max_entries,
        "{name} cannot contain more than {max_entries} entries"
    );
    for value in values {
        validate_bounded_text(value, &format!("one {name} entry"), max_entry_text)?;
    }
    anyhow::ensure!(
        serde_json::to_vec(values)?.len() <= max_serialized_bytes,
        "{name} exceed their serialized {max_serialized_bytes}-byte bound"
    );
    Ok(())
}

fn validate_policy_text(
    constraints: &[String],
    success_criteria: &[String],
    stop_conditions: &[String],
) -> anyhow::Result<()> {
    for (name, values) in [
        ("constraints", constraints),
        ("success_criteria", success_criteria),
        ("stop_conditions", stop_conditions),
    ] {
        validate_bounded_strings(
            values,
            name,
            MAX_POLICY_ENTRIES,
            MAX_POLICY_ENTRY_TEXT,
            MAX_POLICY_TEXT,
        )?;
    }
    let values = constraints
        .iter()
        .chain(success_criteria)
        .chain(stop_conditions);
    let chars = values
        .clone()
        .map(|value| value.chars().count())
        .sum::<usize>();
    let bytes = values.map(String::len).sum::<usize>();
    anyhow::ensure!(
        chars <= MAX_POLICY_TEXT && bytes <= MAX_POLICY_TEXT,
        "constraints, success_criteria, and stop_conditions exceed their combined 8 KiB bound"
    );
    Ok(())
}

fn minutes_to_secs(value: Option<i64>, default_secs: i64, field: &str) -> anyhow::Result<i64> {
    let Some(minutes) = value else {
        return Ok(default_secs);
    };
    anyhow::ensure!(minutes > 0, "{field} must be greater than zero");
    minutes
        .checked_mul(60)
        .ok_or_else(|| anyhow::anyhow!("{field} is too large"))
}

fn append_owner_guidance(existing: Option<&str>, guidance: &str) -> anyhow::Result<String> {
    validate_bounded_text(guidance, "guidance", MAX_GUIDANCE_ENTRY_TEXT)?;
    let mut context = existing
        .and_then(|value| serde_json::from_str::<Value>(value).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    let entries = context
        .as_object_mut()
        .expect("context was normalized to an object")
        .entry("owner_guidance")
        .or_insert_with(|| json!([]));
    if !entries.is_array() {
        *entries = json!([]);
    }
    let entries = entries
        .as_array_mut()
        .expect("owner_guidance was normalized to an array");
    entries.push(json!({
        "guidance": guidance,
        "recorded_at": chrono::Utc::now().to_rfc3339(),
    }));
    let mut newest_first = Vec::new();
    let mut total_chars = 0usize;
    let mut total_bytes = 0usize;
    for entry in entries.iter().rev() {
        if newest_first.len() == MAX_GUIDANCE_ENTRIES {
            break;
        }
        let Some(text) = entry
            .get("guidance")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
        else {
            continue;
        };
        let chars = text.chars().count();
        let bytes = text.len();
        if chars > MAX_GUIDANCE_ENTRY_TEXT
            || bytes > MAX_GUIDANCE_ENTRY_TEXT
            || total_chars.saturating_add(chars) > MAX_GUIDANCE_TEXT
            || total_bytes.saturating_add(bytes) > MAX_GUIDANCE_TEXT
        {
            continue;
        }
        total_chars += chars;
        total_bytes += bytes;
        newest_first.push(entry.clone());
    }
    newest_first.reverse();
    *entries = newest_first;
    Ok(serde_json::to_string(&context)?)
}

fn display_allowlist(values: &[String]) -> String {
    if values.is_empty() {
        "none (controller protocol only)".to_string()
    } else {
        values.join(", ")
    }
}

fn display_target_scope(values: &[String]) -> String {
    if values.is_empty() {
        "none configured (mutation authority requires explicit targets)".to_string()
    } else {
        values.join(", ")
    }
}

fn display_policy(values: &[String]) -> String {
    if values.is_empty() {
        "none specified".to_string()
    } else {
        values
            .iter()
            .enumerate()
            .map(|(index, value)| format!("{}. {}", index + 1, value))
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{setup_test_agent, MockProvider};
    use crate::traits::store_prelude::*;

    #[test]
    fn clean_strings_removes_empty_constraints() {
        let values = vec!["  truthful ".to_string(), " ".to_string()];
        assert_eq!(clean_strings(Some(&values)), vec!["truthful"]);
    }

    #[test]
    fn review_minutes_are_checked() {
        assert_eq!(minutes_to_secs(Some(15), 1, "review").unwrap(), 900);
        assert!(minutes_to_secs(Some(0), 1, "review").is_err());
    }

    #[test]
    fn owner_guidance_is_bounded_durable_context() {
        let context =
            append_owner_guidance(Some(r#"{"mandate_id":"m-1"}"#), "Prefer replies").unwrap();
        let value: Value = serde_json::from_str(&context).unwrap();
        assert_eq!(value["mandate_id"], "m-1");
        assert_eq!(value["owner_guidance"][0]["guidance"], "Prefer replies");
    }

    #[test]
    fn mandate_text_bounds_are_byte_authoritative() {
        assert!(validate_bounded_text(&"é".repeat(1_025), "objective", 2_048).is_err());
        assert!(validate_bounded_text(&"x".repeat(2_048), "objective", 2_048).is_ok());

        let constraints = vec!["x".repeat(500); 16];
        let success = vec!["y".repeat(500)];
        assert!(validate_policy_text(&constraints, &success, &[]).is_err());
        assert!(validate_policy_text(&vec!["x".to_string(); 17], &[], &[]).is_err());
    }

    #[test]
    fn owner_guidance_write_path_keeps_only_whole_bounded_entries() {
        let mut context = None;
        for index in 0..12 {
            let guidance = format!("{index:02}{}", "x".repeat(1_022));
            context = Some(append_owner_guidance(context.as_deref(), &guidance).unwrap());
        }
        let value: Value = serde_json::from_str(context.as_deref().unwrap()).unwrap();
        let entries = value["owner_guidance"].as_array().unwrap();
        assert_eq!(entries.len(), 8);
        let total = entries
            .iter()
            .filter_map(|entry| entry["guidance"].as_str())
            .map(str::len)
            .sum::<usize>();
        assert!(total <= MAX_GUIDANCE_TEXT);
        assert!(entries[0]["guidance"].as_str().unwrap().starts_with("04"));
        assert!(entries[7]["guidance"].as_str().unwrap().starts_with("11"));
    }

    #[tokio::test]
    async fn owner_confirmation_activates_one_unscheduled_controller_atomically() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let (approval_tx, mut approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(state.clone(), ApprovalBroker::new(approval_tx));
        let call = tokio::spawn(async move {
            tool.call(
                r#"{
                    "action":"create",
                    "objective":"Steward @aidaemon_ai thoughtfully",
                    "operation_scopes":[{
                        "tool":"http_request",
                        "operation":"POST",
                        "kind":"mutation",
                        "target_prefixes":["https://api.x.com/2/tweets","auth_profile:twitter","account:12345"],
                        "mutation_effects":["remote_mutation","external_delivery"]
                    }],
                    "max_mutating_actions_per_cycle":1,
                    "max_mutating_actions_per_rolling_24h":8,
                    "min_seconds_between_mutations":900,
                    "constraints":["No fabrication"],
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
        });

        let request = approval_rx.recv().await.expect("confirmation request");
        assert!(matches!(request.kind, ApprovalKind::GoalConfirmation));
        assert!(request.command.contains("Steward @aidaemon_ai"));
        assert!(request
            .warnings
            .iter()
            .any(|warning| warning.contains("http_request")));
        assert!(request.warnings.iter().any(|warning| {
            warning
                == "Resolved token budgets: 400000 tokens per decision cycle; 1000000 tokens per UTC day"
        }));
        request
            .response_tx
            .send(ApprovalResponse::AllowOnce)
            .unwrap();
        let result = call.await.unwrap().unwrap();
        assert!(result.contains("Activated mandate"));

        let mandates = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap();
        assert_eq!(mandates.len(), 1);
        assert_eq!(mandates[0].status, MandateStatus::Active);
        let controller = state.get_goal(&mandates[0].goal_id).await.unwrap().unwrap();
        assert_eq!(controller.status, "active");
        assert_eq!(controller.domain, "orchestration");
        assert_eq!(controller.goal_type, "continuous");
        assert_eq!(controller.budget_per_check, Some(400_000));
        assert_eq!(controller.budget_daily, Some(1_000_000));
        assert!(state
            .get_schedules_for_goal(&controller.id)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn pinned_strategy_is_content_addressed_and_confirmation_is_complete() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let skills_dir = tempfile::TempDir::new().unwrap();
        let original_skill = crate::skills::Skill {
            name: "x-stewardship".to_string(),
            description: "Thoughtful account strategy".to_string(),
            triggers: vec![],
            body: "Prefer useful replies and verify outcomes.".to_string(),
            origin: Some("custom".to_string()),
            source: Some("filesystem".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &original_skill).unwrap();

        let (approval_tx, mut approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(state.clone(), ApprovalBroker::new(approval_tx))
            .with_skills_dir(Some(skills_dir.path().to_path_buf()));
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(30)).to_rfc3339();
        let call = tokio::spawn(async move {
            tool.call(&format!(
                r#"{{
                    "action":"create",
                    "objective":"Steward @aidaemon_ai thoughtfully",
                    "allow_observations":false,
                    "constraints":["No fabrication"],
                    "success_criteria":["Owner ends stewardship"],
                    "stop_conditions":["Credentials are revoked"],
                    "strategy_skill":"x-stewardship",
                    "expires_at":"{expires_at}",
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }}"#,
            ))
            .await
        });

        let request = approval_rx.recv().await.expect("complete confirmation");
        for expected in [
            "Objective:",
            "Constraints:",
            "Success criteria:",
            "Stop conditions:",
            "Pinned strategy:",
            "Observations allowed: false",
            "Allowed mutation effects:",
            "Allowed targets:",
            "Mutation limits:",
            "Review interval:",
            "Expiration:",
            "Resolved token budgets:",
        ] {
            assert!(
                request
                    .warnings
                    .iter()
                    .any(|warning| warning.contains(expected)),
                "missing confirmation field {expected}"
            );
        }

        let changed_skill = crate::skills::Skill {
            body: "A later filesystem edit must not alter the pending snapshot.".to_string(),
            ..original_skill
        };
        crate::skills::write_skill_to_file(skills_dir.path(), &changed_skill).unwrap();
        request
            .response_tx
            .send(ApprovalResponse::AllowOnce)
            .unwrap();
        call.await.unwrap().unwrap();

        let mandate = state
            .list_mandates(Some("owner-session"), false)
            .await
            .unwrap()
            .pop()
            .unwrap();
        let strategy = mandate.strategy.expect("pinned strategy");
        assert_eq!(strategy.skill_name, "x-stewardship");
        assert!(strategy.body.contains("Prefer useful replies"));
        assert!(!strategy.body.contains("later filesystem edit"));
        assert_eq!(strategy.content_sha256.len(), 64);
    }

    #[tokio::test]
    async fn draft_reports_missing_authority_without_writing() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(state.clone(), ApprovalBroker::new(approval_tx));
        let result = tool
            .call(
                r#"{
                    "action":"draft",
                    "objective":"Manage the account as an ongoing presence",
                    "max_mutating_actions_per_cycle":1,
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["execution_mode"], "ongoing_mandate");
        assert_eq!(value["writes_performed"], false);
        assert_eq!(value["ready_to_confirm"], false);
        assert!(value["required_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "operation_scopes"));
        assert!(state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn draft_labels_resolved_token_budgets_and_rejects_action_count_values() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(harness.state, ApprovalBroker::new(approval_tx));
        let result = tool
            .call(
                r#"{
                    "action":"draft",
                    "objective":"Maintain a bounded internal posture",
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
            .unwrap();
        let value: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(value["proposal"]["token_budgets"]["units"], "tokens");
        assert_eq!(
            value["proposal"]["token_budgets"]["per_decision_cycle"],
            400_000
        );
        assert_eq!(value["proposal"]["token_budgets"]["per_utc_day"], 1_000_000);

        let error = tool
            .call(
                r#"{
                    "action":"draft",
                    "objective":"Do not confuse actions with tokens",
                    "budget_per_cycle":1,
                    "budget_daily":1,
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("token budget"));
        assert!(error.to_string().contains("at least 10000"));
    }

    #[tokio::test]
    async fn create_rejects_legacy_cross_product_authority() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(harness.state, ApprovalBroker::new(approval_tx));
        let error = tool
            .call(
                r#"{
                    "action":"create",
                    "objective":"Do not combine unrelated reads and writes",
                    "allowed_tools":["http_request","web_fetch"],
                    "allowed_mutation_effects":["remote_mutation","external_delivery"],
                    "allowed_target_prefixes":["https://blog.aidaemon.ai/posts/","https://api.x.com/2/tweets"],
                    "max_mutating_actions_per_cycle":1,
                    "max_mutating_actions_per_rolling_24h":1,
                    "min_seconds_between_mutations":900,
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("operation_scopes"));
    }

    #[tokio::test]
    async fn update_rejects_and_documents_create_only_fields() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(harness.state, ApprovalBroker::new(approval_tx));

        let schema = tool.schema();
        for field in [
            "source_goal_id",
            "priority",
            "budget_per_cycle",
            "budget_daily",
        ] {
            let description = schema["parameters"]["properties"][field]["description"]
                .as_str()
                .expect("create-only field description");
            assert!(description.contains("Create only"), "{field}");
            let description = description.to_ascii_lowercase();
            assert!(description.contains("update"), "{field}");
            assert!(description.contains("reject"), "{field}");
        }

        let error = tool
            .call(
                r#"{
                    "action":"update",
                    "mandate_id":"not-consulted",
                    "source_goal_id":"source-goal",
                    "priority":"critical",
                    "budget_per_cycle":12345,
                    "budget_daily":67890,
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("create-only fields in v1"));
        for field in [
            "source_goal_id",
            "priority",
            "budget_per_cycle",
            "budget_daily",
        ] {
            assert!(message.contains(field), "{field}");
        }
    }

    #[tokio::test]
    async fn unavailable_confirmation_cancels_instead_of_leaving_resumable_authority() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let (approval_tx, approval_rx) = tokio::sync::mpsc::channel(1);
        drop(approval_rx);
        let tool = ManageMandatesTool::new(state.clone(), ApprovalBroker::new(approval_tx));
        let result = tool
            .call(
                r#"{
                    "action":"create",
                    "objective":"Never activate without confirmation",
                    "max_mutating_actions_per_cycle":0,
                    "_session_id":"owner-session",
                    "_user_role":"owner",
                    "_channel_visibility":"private"
                }"#,
            )
            .await
            .unwrap();
        assert!(result.contains("pending mandate was cancelled"));
        let mandates = state
            .list_mandates(Some("owner-session"), true)
            .await
            .unwrap();
        assert_eq!(mandates.len(), 1);
        assert_eq!(mandates[0].status, MandateStatus::Cancelled);
        assert_eq!(
            state
                .get_goal(&mandates[0].goal_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "cancelled"
        );
    }

    #[tokio::test]
    async fn resume_requires_durable_confirmation_not_only_a_paused_status() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = crate::traits::Goal::new_continuous_pending(
            "Unconfirmed mandate controller",
            "owner-session",
            None,
            None,
        );
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Do not resume without confirmation",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.status = MandateStatus::Paused;
        mandate.confirmed_at = None;
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();

        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(state, ApprovalBroker::new(approval_tx));
        let error = tool
            .call(&format!(
                r#"{{"action":"resume","mandate_id":"{}","_session_id":"owner-session","_user_role":"owner","_channel_visibility":"private"}}"#,
                mandate.id
            ))
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("unconfirmed mandate cannot be resumed"));
    }

    #[tokio::test]
    async fn ask_resume_requires_and_persists_owner_guidance() {
        use crate::traits::{
            Goal, Mandate, MandateAuthority, MandateDecisionCycle, MandateDecisionOutcome,
            MandateRunFinalizationRequest, MandateRunFinalizationResult, Task,
        };

        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("Await owner judgment", "owner-session", None, None);
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Use owner judgment when quality is ambiguous",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        state
            .claim_due_mandates(1, "test-heartbeat", 300)
            .await
            .unwrap();
        let root_task_id = uuid::Uuid::new_v4().to_string();
        let run = state
            .start_goal_run(&goal.id, "mandate", None, Some(&root_task_id))
            .await
            .unwrap();
        let created_at = chrono::Utc::now().to_rfc3339();
        state
            .create_task(&Task {
                id: root_task_id.clone(),
                goal_id: goal.id.clone(),
                description: "Deliberate and record one ASK decision".to_string(),
                status: "pending".to_string(),
                priority: "high".to_string(),
                task_order: 0,
                parallel_group: None,
                depends_on: None,
                agent_id: None,
                context: None,
                result: None,
                error: None,
                blocker: None,
                idempotent: false,
                retry_count: 0,
                max_retries: 0,
                created_at,
                started_at: None,
                completed_at: None,
            })
            .await
            .unwrap();
        let root_attempt = state
            .claim_task_with_lease(
                &root_task_id,
                "ask-resume-task-lead",
                Some("profile-task-lead"),
                7_200,
            )
            .await
            .unwrap()
            .unwrap();
        let mut ask = MandateDecisionCycle::new(
            &mandate.id,
            &run.id,
            MandateDecisionOutcome::Ask,
            "Owner preference is material",
            mandate.version,
        );
        ask.question = Some("Should replies be preferred?".to_string());
        state
            .record_mandate_decision(&ask, None, Some(&root_attempt.id))
            .await
            .unwrap();
        assert!(state
            .patch_task_from_attempt(
                &root_attempt.id,
                &root_attempt.lease_token,
                &crate::traits::TaskAttemptPatch {
                    status: "completed".to_string(),
                    result: Some("ASK decision durably recorded".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap());
        assert!(matches!(
            state
                .finalize_mandate_run_from_proof(&MandateRunFinalizationRequest {
                    mandate_id: mandate.id.clone(),
                    expected_mandate_version: mandate.version,
                    goal_run_id: run.id.clone(),
                    finalized_at: chrono::Utc::now().to_rfc3339(),
                })
                .await
                .unwrap(),
            MandateRunFinalizationResult::NonActionSatisfied {
                outcome: MandateDecisionOutcome::Ask,
                ..
            }
        ));

        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(state.clone(), ApprovalBroker::new(approval_tx));
        let missing = tool
            .call(&format!(
                r#"{{"action":"answer_question","mandate_id":"{}","_session_id":"owner-session","_user_role":"owner","_channel_visibility":"private"}}"#,
                mandate.id
            ))
            .await;
        assert!(missing
            .unwrap_err()
            .to_string()
            .contains("guidance is required"));

        let resumed = tool
            .call(&format!(
                r#"{{"action":"answer_question","mandate_id":"{}","guidance":"Prefer thoughtful replies over original posts.","_session_id":"owner-session","_user_role":"owner","_channel_visibility":"private"}}"#,
                mandate.id
            ))
            .await
            .unwrap();
        assert!(resumed.contains("is active"));
        let resumed_mandate = state.get_mandate(&mandate.id).await.unwrap().unwrap();
        assert_eq!(resumed_mandate.status, MandateStatus::Active);
        let resumed_review_at =
            chrono::DateTime::parse_from_rfc3339(&resumed_mandate.next_review_at)
                .unwrap()
                .with_timezone(&chrono::Utc);
        assert!(resumed_review_at <= chrono::Utc::now() + chrono::Duration::seconds(2));
        assert!(resumed_review_at >= chrono::Utc::now() - chrono::Duration::seconds(2));
        let controller = state.get_goal(&goal.id).await.unwrap().unwrap();
        assert_eq!(controller.status, "active");
        assert!(controller
            .context
            .as_deref()
            .unwrap()
            .contains("Prefer thoughtful replies"));
    }

    #[tokio::test]
    async fn stale_internal_decision_cannot_rebind_to_a_newer_goal_run() {
        use crate::traits::{Goal, Mandate, MandateAuthority};

        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let state = harness.state.clone();
        let goal = Goal::new_continuous("Pinned decision cycle", "owner-session", None, None);
        let mut mandate = Mandate::new(
            &goal.id,
            None,
            "Never move a stale decision onto a newer run",
            "owner-session",
            MandateAuthority::default(),
            60,
            3_600,
            300,
        );
        mandate.next_review_at = (chrono::Utc::now() - chrono::Duration::minutes(1)).to_rfc3339();
        state
            .create_mandate_controller(&goal, &mandate)
            .await
            .unwrap();
        state
            .claim_due_mandates(1, "pinned-run-test", 300)
            .await
            .unwrap();

        let stale_run = state
            .start_goal_run(&goal.id, "mandate", None, None)
            .await
            .unwrap();
        state
            .finish_goal_run(&stale_run.id, "failed", Some("superseded"))
            .await
            .unwrap();
        let current_run = state
            .start_goal_run(&goal.id, "mandate", None, None)
            .await
            .unwrap();

        let (approval_tx, _approval_rx) = tokio::sync::mpsc::channel(1);
        let tool = ManageMandatesTool::new(state.clone(), ApprovalBroker::new(approval_tx));
        let error = tool
            .call(&format!(
                r#"{{"action":"record_decision","outcome":"wait","rationale":"stale","_goal_id":"{}","_goal_run_id":"{}","_task_attempt_id":"stale-attempt","_session_id":"sub-task-lead","_channel_visibility":"internal"}}"#,
                goal.id, stale_run.id
            ))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("no longer executable"));
        assert!(state
            .get_mandate_decision_for_run(&current_run.id)
            .await
            .unwrap()
            .is_none());
    }
}
