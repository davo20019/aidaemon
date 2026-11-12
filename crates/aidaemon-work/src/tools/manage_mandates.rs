use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::traits::UserRole;
use crate::traits::{
    Intention, Mandate, MandateActivityLevel, MandateAuthority, MandateAutonomyMode,
    MandateDecisionCycle, MandateDecisionOutcome, MandateLearningNote,
    MandateObjectiveMeasurement, MandateOperatingUpdates, MandateOperationScope,
    MandateReconciliationResolution, MandateStatus, MandateStrategySnapshot, MandateSuspensionKind,
    MandateTerminationKind, MandateToolStore, ToolCallContext, ToolCallMetadata, ToolCallOperation,
    ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus,
    ToolRole,
};
use aidaemon_core::ChannelVisibility;
use aidaemon_tools::OutcomeTool as Tool;

const DEFAULT_MIN_REVIEW_SECS: i64 = 15 * 60;
const DEFAULT_MAX_REVIEW_SECS: i64 = 24 * 60 * 60;
const DEFAULT_REVIEW_SECS: i64 = 4 * 60 * 60;
/// The review interval an Autopilot mandate gets unless the owner sets one.
pub const AUTOPILOT_DEFAULT_REVIEW_SECS: i64 = 3 * 60 * 60;
// Token counts are an internal runaway guard, not an owner-facing unit. The
// owner chooses a human-readable review effort; cadence and the selected mode
// derive enough aggregate capacity to fund every expected review in a UTC day.
// Legacy raw values remain parseable for backward compatibility but are not
// advertised in the tool schema.
const MIN_MANDATE_TOKEN_BUDGET: i64 = 30_000;
const MAX_MANDATE_TOKEN_BUDGET_PER_CYCLE: i64 = 1_000_000;
const MAX_MANDATE_TOKEN_BUDGET_DAILY: i64 = 50_000_000;
const MAX_OBJECTIVE_TEXT: usize = 2 * 1024;
const MAX_NAME_TEXT: usize = 80;
const MAX_POLICY_ENTRIES: usize = 16;
const MAX_POLICY_ENTRY_TEXT: usize = 500;
const MAX_POLICY_TEXT: usize = 8 * 1024;
const MAX_RATIONALE_TEXT: usize = 2 * 1024;
const MAX_OBSERVATIONS: usize = 8;
const MAX_OBSERVATION_TEXT: usize = 750;
const MAX_OBSERVATIONS_JSON: usize = 6 * 1024;
const MAX_QUESTION_TEXT: usize = 500;
const MAX_INTENTION_TEXT: usize = 1024;
use crate::mandates::card::{
    CanDo, EndLimit, Ends, MandateConfirmationCard, NoticeTrigger, ReviewCadence,
};
use crate::mandates::MAX_INTENTION_METADATA_TEXT;
const MAX_GUIDANCE_ENTRIES: usize = 10;
const MAX_GUIDANCE_ENTRY_TEXT: usize = 1024;
const MAX_GUIDANCE_TEXT: usize = 8 * 1024;
const MAX_LEARNING_NOTE_TEXT: usize = 1024;
const MAX_EVIDENCE_RECEIPTS: usize = 16;
/// Evidence a learning note may cite (`MandateLearningNote::validate`).
const MAX_LEARNING_EVIDENCE_RECEIPTS: usize = 8;
const EVIDENCE_RECEIPT_IDS_DESCRIPTION: &str = "Optional. Omit to cite this run's successful observations automatically. To cite specific ones, use the tool_call_id you issued (e.g. call_...) or the result_id in that result's receipt footer (e.g. sha256:...); receipts from other runs, failed calls, or invented IDs are rejected.";
const LEARNING_EVIDENCE_RECEIPT_IDS_DESCRIPTION: &str = "Optional. Receipts backing learning_note; they must also back this decision. Omit to cite the decision's evidence.";
const NAME_FIELD_DESCRIPTION: &str = "draft/create: the short name the owner gave the mandate, exactly as given; omit when none. Other actions: selects the mandate by that name when mandate_id is omitted.";
const SOURCE_GOAL_FIELD_DESCRIPTION: &str = "Create only. Optional provenance link to a personal goal created in this exact private owner session; omit it when no compatible goal exists. Updates preserve the existing binding and ignore this field.";
const DURATION_FIELD_DESCRIPTION: &str = "Create only, relative to activation. Updates ignore this field; use expires_at to change an existing mandate's expiry.";
const PRIORITY_FIELD_DESCRIPTION: &str = "Create only controller scheduling priority. Updates preserve the existing priority and ignore this field.";
const CONSTRAINT_FIELD_DESCRIPTION: &str = "Owner-facing guardrail. Use one short plain-language rule per item; split compound rules and avoid implementation jargon.";
const SUCCESS_FIELD_DESCRIPTION: &str =
    "Owner-facing, observable success measure. Use one short plain-language result per item.";
const STOP_FIELD_DESCRIPTION: &str =
    "Owner-facing reason to stop. Use one short plain-language condition per item.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReviewEffort {
    Efficient,
    Balanced,
    Thorough,
}

impl ReviewEffort {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "efficient" => Some(Self::Efficient),
            "balanced" => Some(Self::Balanced),
            "thorough" => Some(Self::Thorough),
            _ => None,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Efficient => "efficient",
            Self::Balanced => "balanced",
            Self::Thorough => "thorough",
        }
    }

    const fn per_cycle_capacity(self) -> i64 {
        match self {
            Self::Efficient => 100_000,
            Self::Balanced => 250_000,
            Self::Thorough => 500_000,
        }
    }

    const fn daily_capacity_floor(self) -> i64 {
        match self {
            Self::Efficient => 1_000_000,
            Self::Balanced => 2_000_000,
            Self::Thorough => 4_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedReviewCapacity {
    effort: Option<ReviewEffort>,
    per_cycle: i64,
    daily: i64,
}

impl ResolvedReviewCapacity {
    const fn automatically_managed(self) -> bool {
        self.effort.is_some()
    }

    fn display_mode(self) -> &'static str {
        self.effort.map_or("legacy_custom", ReviewEffort::as_str)
    }
}

/// Owner-facing mandate administration and the internal deliberation commit point.
///
/// This tool deliberately does not execute delegated actions. It records the
/// owner's authority envelope and the task lead's ACT/WAIT/ASK/STOP choice; the
/// dispatcher enforces the envelope when an action is attempted.
/// A skill a mandate may take as its strategy, as the host's skill library
/// holds it.
#[derive(Debug, Clone)]
pub struct StrategySkill {
    pub name: String,
    pub description: String,
    /// The skill's body with untrusted content already neutralized by the
    /// host (it is shown to the model on every review).
    pub body: String,
    pub source: Option<String>,
    /// The skill's whole canonical text; its digest pins the version the
    /// owner confirmed.
    pub canonical: String,
}

/// The host's skill library, as `manage_mandates` reads it for strategies.
pub trait StrategySkills: Send + Sync {
    /// The skill named `name`, if the library has one.
    fn find(&self, name: &str) -> Option<StrategySkill>;
}

pub struct ManageMandatesTool {
    state: Arc<dyn MandateToolStore>,
    strategy_skills: Option<Arc<dyn StrategySkills>>,
    /// The strategy skill content each card was judged with, by session and
    /// approved arguments, so a commit pins exactly the content the owner
    /// was asked about (a skill edited while the card was open is refused).
    judged_strategies: std::sync::Mutex<std::collections::HashMap<String, String>>,
}

/// The owner's confirmation card for a mandate change, as the tool asked
/// before the kernel did.
struct MandateCard {
    summary: String,
    risk: aidaemon_core::ApprovalRisk,
    warnings: Vec<String>,
    /// The confirmed policy is Autopilot (the card is the Autopilot one).
    autopilot: bool,
}

/// A validated `create`, before anything is stored.
struct CreatePlan {
    goal: crate::traits::Goal,
    mandate: Mandate,
    duration_secs: Option<i64>,
    card: MandateCard,
}

/// A validated `update`: the mandate as it will be stored.
struct UpdatePlan {
    mandate: Mandate,
    capacity: ResolvedReviewCapacity,
    card: MandateCard,
}

impl ManageMandatesTool {
    pub fn new(state: Arc<dyn MandateToolStore>) -> Self {
        Self {
            state,
            strategy_skills: None,
            judged_strategies: Default::default(),
        }
    }

    /// A committing owner change (create, update) runs only
    /// after the owner confirmed it through the agent kernel.
    fn require_kernel_approval(args: &ManageMandatesArgs) -> anyhow::Result<()> {
        anyhow::ensure!(
            args.call_kernel_approved,
            "manage_mandates {} requires approval through the agent kernel; it was called \
             outside an approved agent call",
            args.action
        );
        Ok(())
    }

    /// The owner approval one call needs: creating and updating a mandate
    /// are confirmed by the owner, whatever the
    /// session, exactly where the call would commit them. Every other call
    /// (including one these actions answer without committing) needs none.
    pub async fn owner_approval(
        &self,
        arguments: &Value,
        facts: &ToolCallContext,
    ) -> crate::traits::OwnerApprovalNeed {
        use crate::traits::OwnerApprovalNeed;
        let Ok(mut args) = serde_json::from_value::<ManageMandatesArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        args.normalize_for_action();
        args.bind_facts(facts);
        let card = match args.action.as_str() {
            "create" => match self.create_plan(&args).await {
                Ok(Ok(plan)) => plan.card,
                _ => return OwnerApprovalNeed::NotNeeded,
            },
            "update" => match self.update_plan(&args).await {
                Ok(Ok(plan)) => plan.card,
                _ => return OwnerApprovalNeed::NotNeeded,
            },
            _ => return OwnerApprovalNeed::NotNeeded,
        };
        // The exact arguments the owner approves (never `_` keys); a changed
        // argument is a new card.
        let approved: serde_json::Map<String, Value> = arguments
            .as_object()
            .map(|map| {
                map.iter()
                    .filter(|(key, _)| !key.starts_with('_'))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default();
        let strategy_sha256 = self.judged_strategy_hash(&args);
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            effects: Vec::new(),
            summary: card.summary,
            subject: json!({
                "arguments": approved,
                "autopilot": card.autopilot,
                "strategy_sha256": strategy_sha256,
            })
            .to_string(),
            risk: card.risk,
            warnings: card.warnings,
        })
    }

    /// The card kind for a request from [`Self::owner_approval`]: the
    /// Autopilot confirmation for an Autopilot policy, else the goal one.
    pub fn card_kind_for(
        request: &aidaemon_core::OwnerApprovalRequest,
    ) -> crate::traits::ApprovalKind {
        let autopilot = serde_json::from_str::<Value>(&request.subject)
            .ok()
            .and_then(|subject| subject.get("autopilot")?.as_bool())
            .unwrap_or(false);
        if autopilot {
            crate::traits::ApprovalKind::AutopilotConfirmation
        } else {
            crate::traits::ApprovalKind::GoalConfirmation
        }
    }

    /// The host's skill library, which a mandate's `strategy_skill` names.
    /// Without one, a strategy skill is refused.
    pub fn with_strategy_skills(mut self, skills: Option<Arc<dyn StrategySkills>>) -> Self {
        self.strategy_skills = skills;
        self
    }

    fn is_owner(args: &ManageMandatesArgs) -> bool {
        args.call_role == Some(UserRole::Owner)
    }

    fn is_internal(args: &ManageMandatesArgs) -> bool {
        args.call_visibility == Some(ChannelVisibility::Internal)
            || args.call_session.as_deref().is_some_and(|session| {
                session.starts_with("specialist:") || session.starts_with("sub-")
            })
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

    fn is_private_owner_control(args: &ManageMandatesArgs) -> bool {
        Self::is_owner(args)
            && !Self::is_internal(args)
            && args.call_visibility == Some(ChannelVisibility::Private)
    }

    fn owner_session(args: &ManageMandatesArgs) -> anyhow::Result<&str> {
        let session = args
            .call_session
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
            )
            .with_owner_alerts(args.owner_alerts.unwrap_or(false)));
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
            owner_alerts: args.owner_alerts.unwrap_or(false),
        })
    }

    fn autonomy_mode_from_args(
        args: &ManageMandatesArgs,
        existing: Option<MandateAutonomyMode>,
    ) -> anyhow::Result<MandateAutonomyMode> {
        args.autonomy_mode
            .as_deref()
            .map(|value| {
                MandateAutonomyMode::parse(value)
                    .ok_or_else(|| anyhow::anyhow!("autonomy_mode must be bounded or autopilot"))
            })
            .transpose()
            .map(|mode| mode.or(existing).unwrap_or_default())
    }

    fn resolved_review_capacity(
        args: &ManageMandatesArgs,
        default_review_secs: i64,
        timing: &ActivationTiming,
        existing: Option<(i64, i64)>,
        existing_effort: Option<ReviewEffort>,
        default_effort: Option<ReviewEffort>,
    ) -> anyhow::Result<ResolvedReviewCapacity> {
        anyhow::ensure!(
            args.review_effort.is_none()
                || (args.budget_per_cycle.is_none() && args.budget_daily.is_none()),
            "review_effort cannot be combined with legacy raw token budgets"
        );
        let requested_effort = args
            .review_effort
            .as_deref()
            .map(|value| {
                ReviewEffort::parse(value).ok_or_else(|| {
                    anyhow::anyhow!("review_effort must be efficient, balanced, or thorough")
                })
            })
            .transpose()?;
        let has_legacy_raw = args.budget_per_cycle.is_some() || args.budget_daily.is_some();

        let effort = (!has_legacy_raw)
            .then(|| requested_effort.or(existing_effort).or(default_effort))
            .flatten();
        let per_cycle = if let Some(effort) = effort {
            effort.per_cycle_capacity()
        } else {
            args.budget_per_cycle
                .or(existing.map(|value| value.0))
                .unwrap_or(ReviewEffort::Balanced.per_cycle_capacity())
        };
        anyhow::ensure!(
            (MIN_MANDATE_TOKEN_BUDGET..=MAX_MANDATE_TOKEN_BUDGET_PER_CYCLE)
                .contains(&per_cycle),
            "legacy per-review token capacity must be between {MIN_MANDATE_TOKEN_BUDGET} and {MAX_MANDATE_TOKEN_BUDGET_PER_CYCLE}"
        );
        let review_cycles = expected_default_review_cycles(default_review_secs, timing)?;
        let cadence_floor = per_cycle
            .checked_mul(review_cycles)
            .ok_or_else(|| anyhow::anyhow!("review cadence token budget is too large"))?;
        let daily = if let Some(effort) = effort {
            effort.daily_capacity_floor().max(cadence_floor)
        } else {
            args.budget_daily
                .or(existing.map(|value| value.1))
                .unwrap_or(
                    default_effort
                        .unwrap_or(ReviewEffort::Balanced)
                        .daily_capacity_floor()
                        .max(cadence_floor),
                )
        };
        if effort.is_some() {
            anyhow::ensure!(
                daily <= MAX_MANDATE_TOKEN_BUDGET_DAILY,
                "default_review_minutes is too short: {review_cycles} reviews per day exceed the daily review capacity ceiling; keep the current review bounds or lengthen default_review_minutes"
            );
        }
        anyhow::ensure!(
            (MIN_MANDATE_TOKEN_BUDGET..=MAX_MANDATE_TOKEN_BUDGET_DAILY).contains(&daily),
            "legacy daily token capacity must be between {MIN_MANDATE_TOKEN_BUDGET} and {MAX_MANDATE_TOKEN_BUDGET_DAILY}"
        );
        anyhow::ensure!(
            daily >= per_cycle,
            "daily review capacity must be at least one review cycle"
        );
        anyhow::ensure!(
            daily >= cadence_floor,
            "daily review capacity cannot fund {review_cycles} default review cycle(s); select automatic review_effort or lengthen default_review_minutes"
        );
        Ok(ResolvedReviewCapacity {
            effort,
            per_cycle,
            daily,
        })
    }

    fn strategy_key(args: &ManageMandatesArgs) -> Option<String> {
        let skill = args.strategy_skill.as_deref()?.trim();
        if skill.is_empty() || !matches!(args.action.as_str(), "create" | "update") {
            return None;
        }
        Some(format!(
            "{}\n{}\n{}\n{}",
            args.call_session.as_deref().unwrap_or_default(),
            args.action,
            args.mandate_id.as_deref().unwrap_or_default(),
            skill
        ))
    }

    /// Judging a card: remember the strategy content it shows (its hash).
    fn judged_strategy_hash(&self, args: &ManageMandatesArgs) -> Option<String> {
        let key = Self::strategy_key(args)?;
        let hash = self
            .strategy_snapshot(args.strategy_skill.as_deref())
            .ok()
            .flatten()?
            .content_sha256;
        self.judged_strategies
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, hash.clone());
        Some(hash)
    }

    /// The strategy snapshot a committing call pins: refused when the skill
    /// changed since the owner's card was judged.
    fn approved_strategy_snapshot(
        &self,
        args: &ManageMandatesArgs,
    ) -> anyhow::Result<Option<MandateStrategySnapshot>> {
        let snapshot = self.strategy_snapshot(args.strategy_skill.as_deref())?;
        if !args.call_kernel_approved {
            return Ok(snapshot);
        }
        let judged = Self::strategy_key(args).and_then(|key| {
            self.judged_strategies
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&key)
        });
        if let (Some(judged), Some(current)) = (judged, snapshot.as_ref()) {
            anyhow::ensure!(
                judged == current.content_sha256,
                "strategy skill `{}` changed after the owner was asked; call again so the owner \
                 confirms the current version",
                current.skill_name
            );
        }
        Ok(snapshot)
    }

    fn strategy_snapshot(
        &self,
        skill_name: Option<&str>,
    ) -> anyhow::Result<Option<MandateStrategySnapshot>> {
        let Some(skill_name) = skill_name.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        let skills = self.strategy_skills.as_ref().ok_or_else(|| {
            anyhow::anyhow!("strategy_skill requires the filesystem skills system")
        })?;
        let skill = skills
            .find(skill_name)
            .ok_or_else(|| anyhow::anyhow!("strategy skill `{skill_name}` was not found"))?;
        let canonical = skill.canonical;
        let body = skill.body.trim().to_string();
        anyhow::ensure!(
            !body.is_empty(),
            "strategy skill body is empty after sanitization"
        );
        let snapshot = MandateStrategySnapshot {
            skill_name: skill.name.clone(),
            snapshot_version: MandateStrategySnapshot::SCHEMA_VERSION,
            content_sha256: hex::encode(Sha256::digest(canonical.as_bytes())),
            description: skill.description.trim().to_string(),
            body,
            source: skill.source.clone(),
        };
        snapshot.validate().map_err(anyhow::Error::msg)?;
        Ok(Some(snapshot))
    }

    /// The owner's confirmation card: five essentials (what, how often,
    /// what it can do, when it tells you, when it ends), then the details.
    fn confirmation_card(
        mandate: &Mandate,
        capacity: ResolvedReviewCapacity,
        activation_duration_secs: Option<i64>,
        proposed_policy_version: i64,
    ) -> MandateConfirmationCard {
        let authority = &mandate.authority;
        let labels = access_labels(authority);
        let changes = can_change(authority, &labels);
        let can_do = if changes {
            CanDo::Changes {
                what: labels.changes.iter().cloned().collect(),
                per_review: authority.max_mutating_actions_per_cycle,
                per_day: authority.max_mutating_actions_per_rolling_24h,
                min_gap_secs: authority.min_seconds_between_mutations,
            }
        } else {
            CanDo::ReadOnly
        };
        let mut triggers = Vec::new();
        if authority.owner_alerts {
            triggers.push(NoticeTrigger::Finding);
        }
        if changes {
            triggers.push(NoticeTrigger::Change);
        }
        triggers.extend([NoticeTrigger::NeedsInput, NoticeTrigger::Problem]);
        let ends = Ends {
            limit: match (activation_duration_secs, mandate.expires_at.as_deref()) {
                (Some(secs), _) => EndLimit::AfterApproval { secs },
                (None, Some(at)) => EndLimit::At(
                    crate::schedule::owner_time_rfc3339(at)
                        .unwrap_or_else(|| "at the end date you set".to_string()),
                ),
                (None, None) => EndLimit::None,
            },
            has_stop_conditions: !mandate.stop_conditions.is_empty(),
        };

        let mut details = Vec::new();
        if !mandate.constraints.is_empty() {
            details.push(format!(
                "Constraints: {}",
                display_policy(&mandate.constraints)
            ));
        }
        details.push(format!(
            "Success criteria: {}",
            display_policy(&mandate.success_criteria)
        ));
        if !mandate.stop_conditions.is_empty() {
            details.push(format!(
                "Stop conditions: {}",
                display_policy(&mandate.stop_conditions)
            ));
        }
        if let Some(strategy) = mandate.strategy.as_ref() {
            details.push(format!("Guidance: {}", strategy.description.trim()));
        }
        details.push(format!(
            "Access summary: {}",
            display_access_summary(authority)
        ));
        details.push(format!(
            "Review depth: {}",
            friendly_review_effort(capacity.display_mode())
        ));
        details.push(format!(
            "Autonomy mode: {}",
            match mandate.autonomy_mode {
                MandateAutonomyMode::Bounded => {
                    "Bounded — works on its own only within the rules shown here"
                }
                MandateAutonomyMode::Autopilot => {
                    "Autopilot — handles routine decisions within the rules shown here"
                }
            }
        ));
        if mandate.autonomy_mode.is_autopilot() {
            details.push(
                "Owner checkpoints: It asks you before using a new account, capability, destination, sensitive data, spending authority, destructive action, or a judgment only you can make."
                    .to_string(),
            );
            details.push(
                "Recovery policy: It may retry safe internal failures and resume after a restart, but it never blindly repeats an external action when the result is unclear."
                    .to_string(),
            );
        }
        details.push(format!(
            "Policy reference: Your approval covers only the rules and limits shown here (policy version {proposed_policy_version})."
        ));

        MandateConfirmationCard {
            what: mandate
                .objective
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
            how_often: ReviewCadence {
                min_secs: mandate.min_review_secs,
                default_secs: mandate.default_review_secs,
                max_secs: mandate.max_review_secs,
            }
            .render(),
            can_do: can_do.render(),
            tells_you_when: NoticeTrigger::render(&triggers),
            ends: ends.render(),
            details,
        }
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
        let name = mandate_name(args)?;
        let autonomy_mode = Self::autonomy_mode_from_args(args, None)?;
        let authority = Self::authority_from_args(args)?;
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
            if autonomy_mode.is_autopilot() {
                AUTOPILOT_DEFAULT_REVIEW_SECS
            } else {
                DEFAULT_REVIEW_SECS
            },
            "default_review_minutes",
        )?;
        anyhow::ensure!(
            min_review_secs <= default_review_secs && default_review_secs <= max_review_secs,
            "{}",
            review_bounds_error(min_review_secs, default_review_secs, max_review_secs)
        );
        let timing = activation_timing(args)?;
        let default_effort = if autonomy_mode.is_autopilot() {
            ReviewEffort::Thorough
        } else {
            ReviewEffort::Balanced
        };
        let capacity = Self::resolved_review_capacity(
            args,
            default_review_secs,
            &timing,
            None,
            None,
            Some(default_effort),
        )?;
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
        if success_criteria.is_empty() {
            missing.push("success_criteria");
        }
        let validation_error = authority
            .validate()
            .err()
            .or_else(|| unreachable_scope_targets(&authority.operation_scopes));
        let ready_to_confirm = missing.is_empty() && validation_error.is_none();
        Ok(serde_json::to_string_pretty(&json!({
            "execution_mode": "ongoing_mandate",
            "writes_performed": false,
            "ready_to_confirm": ready_to_confirm,
            "required_inputs": missing,
            "validation_status": validation_error.as_deref().unwrap_or("ready"),
            "proposal": {
                "name": name,
                "objective": objective,
                "autonomy_mode": autonomy_mode,
                "constraints": constraints,
                "success_criteria": success_criteria,
                "stop_conditions": stop_conditions,
                "strategy": strategy.as_ref().map(|value| json!({
                    "skill_name": value.skill_name,
                    "content_sha256": value.content_sha256,
                })),
                "owner_alerts": authority.owner_alerts,
                "authority": authority,
                "resource_policy": {
                    "review_effort": capacity.display_mode(),
                    "automatically_managed": capacity.automatically_managed(),
                    "cadence_funded": true,
                    "runaway_protection": true,
                },
                "review_minutes": {
                    "minimum": min_review_secs / 60,
                    "default": default_review_secs / 60,
                    "maximum": max_review_secs / 60,
                },
                "expires_at": timing.expires_at,
                "owner_times": owner_times([("ends", timing.expires_at.as_deref())]),
                "duration_minutes": timing.duration_secs.map(|value| value / 60),
                "timing_normalization": timing.normalized_redundant_expiry.then_some(
                    "duration_minutes is authoritative; the redundant expires_at value was ignored"
                ),
                "priority": args.priority.as_deref().unwrap_or("high"),
            },
            // The owner confirms a ready proposal on the card create shows,
            // so a text "reply create" confirmation before it asks twice.
            "owner_confirms_on_card": ready_to_confirm,
            "next_step": if ready_to_confirm {
                "If the owner asked for this to run, call create with this proposal now: create shows the owner the confirmation card, so do not ask for a separate confirmation in chat first."
            } else {
                "Supply every required input, then draft again before create."
            }
        }))?)
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
                self.state
                    .is_mandate_session_authorized(&mandate.id, owner_session)
                    .await?,
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

    /// The mandate a call names: by `mandate_id` (full ID or unique prefix),
    /// else by the owner-given `name` it was created with (case-insensitive,
    /// exact). A name prefers live mandates; only when none carries it does
    /// it select a finished one.
    async fn resolve_selected_mandate(
        &self,
        args: &ManageMandatesArgs,
        owner_session: &str,
    ) -> anyhow::Result<Mandate> {
        if let Some(id) = args.mandate_id.as_deref() {
            return self.resolve_owned_mandate(id, owner_session).await;
        }
        let Some(name) = args.name.as_deref() else {
            anyhow::bail!("mandate_id is required (or the mandate's name)");
        };
        for include_terminal in [false, true] {
            let named = self
                .state
                .list_mandates(Some(owner_session), include_terminal)
                .await?
                .into_iter()
                .filter(|mandate| mandate.has_name(name))
                .collect::<Vec<_>>();
            match named.as_slice() {
                [] => continue,
                [mandate] => return Ok(mandate.clone()),
                several => anyhow::bail!(
                    "more than one mandate is named \"{name}\" ({}); use mandate_id",
                    several
                        .iter()
                        .map(|mandate| mandate.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
        anyhow::bail!("no mandate named \"{name}\" in this owner session")
    }

    /// Validate a `create` exactly as it commits, without storing anything.
    /// `Ok(Err(answer))`: the call answers without creating a mandate.
    async fn create_plan(
        &self,
        args: &ManageMandatesArgs,
    ) -> anyhow::Result<Result<CreatePlan, String>> {
        if !Self::is_private_owner_control(args) {
            return Ok(Err(
                "Mandates can only be created by the owner in a verified private channel."
                    .to_string(),
            ));
        }
        let session_id = Self::owner_session(args)?;
        let objective =
            required_bounded_trimmed(args.objective.as_deref(), "objective", MAX_OBJECTIVE_TEXT)?;
        let name = mandate_name(args)?;
        if let Some(name) = name.as_deref() {
            // A name selects a mandate later, so it must stay unambiguous
            // among the owner's live mandates.
            if let Some(taken) = self
                .state
                .list_mandates(Some(session_id), false)
                .await?
                .into_iter()
                .find(|mandate| mandate.has_name(name))
            {
                anyhow::bail!(
                    "a live mandate is already named \"{name}\" ({}); pick another name or cancel that one first",
                    taken.id
                );
            }
        }
        let autonomy_mode = Self::autonomy_mode_from_args(args, None)?;
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
        if let Some(reason) = unreachable_scope_targets(&authority.operation_scopes) {
            anyhow::bail!(reason);
        }

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
            if autonomy_mode.is_autopilot() {
                AUTOPILOT_DEFAULT_REVIEW_SECS
            } else {
                DEFAULT_REVIEW_SECS
            },
            "default_review_minutes",
        )?;
        anyhow::ensure!(
            min_review_secs <= default_review_secs && default_review_secs <= max_review_secs,
            "{}",
            review_bounds_error(min_review_secs, default_review_secs, max_review_secs)
        );

        let timing = activation_timing(args)?;
        let default_effort = if autonomy_mode.is_autopilot() {
            ReviewEffort::Thorough
        } else {
            ReviewEffort::Balanced
        };
        let capacity = Self::resolved_review_capacity(
            args,
            default_review_secs,
            &timing,
            None,
            None,
            Some(default_effort),
        )?;
        let mut goal = crate::traits::Goal::new_continuous_pending(
            &format!("Mandate: {objective}"),
            session_id,
            Some(capacity.per_cycle),
            Some(capacity.daily),
        );
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
        mandate.name = name;
        mandate.autonomy_mode = autonomy_mode;
        mandate.next_review_at = chrono::Utc::now().to_rfc3339();
        mandate.constraints = clean_strings(args.constraints.as_deref());
        mandate.success_criteria = clean_strings(args.success_criteria.as_deref());
        mandate.stop_conditions = clean_strings(args.stop_conditions.as_deref());
        mandate.strategy = self.approved_strategy_snapshot(args)?;
        mandate.review_effort = capacity.display_mode().to_string();
        validate_policy_text(
            &mandate.constraints,
            &mandate.success_criteria,
            &mandate.stop_conditions,
        )?;
        anyhow::ensure!(
            !mandate.success_criteria.is_empty(),
            "new mandates require at least one observable success criterion so autonomous reviews can judge value instead of merely repeating activity"
        );
        mandate.expires_at = timing.expires_at.clone();
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

        let proposed_policy_version = mandate.version + 1;
        let mut card = Self::confirmation_card(
            &mandate,
            capacity,
            timing.duration_secs,
            proposed_policy_version,
        );
        if timing.normalized_redundant_expiry {
            card.details.push(
                "Timing note: You gave both a length and an end date; it uses the length, so the end date was ignored."
                    .to_string(),
            );
        }
        let warnings = card.into_lines();
        let titled = match mandate.name.as_deref() {
            Some(name) => format!(" \"{name}\": {objective}"),
            None => format!(": {objective}"),
        };
        let card = MandateCard {
            summary: if autonomy_mode.is_autopilot() {
                format!("Enable Autopilot{titled}")
            } else {
                format!("Delegate mandate{titled}")
            },
            risk: if mandate.authority.max_mutating_actions_per_cycle == 0 {
                aidaemon_core::ApprovalRisk::Medium
            } else {
                aidaemon_core::ApprovalRisk::High
            },
            warnings,
            autopilot: autonomy_mode.is_autopilot(),
        };
        Ok(Ok(CreatePlan {
            goal,
            mandate,
            duration_secs: timing.duration_secs,
            card,
        }))
    }

    /// Create and activate a mandate. The owner confirmed it through the
    /// agent kernel (`owner_approval_request`) before this runs; a declined
    /// confirmation stores nothing.
    async fn create(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        let plan = match self.create_plan(args).await? {
            Err(answer) => return Ok(answer),
            Ok(plan) => plan,
        };
        Self::require_kernel_approval(args)?;
        let CreatePlan {
            goal,
            mandate,
            duration_secs,
            card: _,
        } = plan;
        // Stored paused, then confirmed: the confirmation timestamp makes the
        // first deliberation immediately due.
        self.state
            .create_mandate_controller(&goal, &mandate)
            .await?;
        anyhow::ensure!(
            self.state
                .confirm_mandate(&mandate.id, mandate.version, duration_secs)
                .await?,
            "mandate could not be activated"
        );
        let activated = self
            .state
            .get_mandate(&mandate.id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("activated mandate could not be reloaded"))?;
        let ends = match activated.expires_at.as_deref() {
            Some(expires_at) => format!(
                "it ends {} (expires_at {expires_at})",
                crate::schedule::owner_time_rfc3339(expires_at)
                    .unwrap_or_else(|| expires_at.to_string())
            ),
            None => "it has no end date".to_string(),
        };
        Ok(format!(
            "{} {}{} at {} under policy version {}; {ends}. Its first review is due now; future reviews are chosen within {}–{} minutes and capped at expiry.",
            if activated.autonomy_mode.is_autopilot() {
                "Autopilot enabled for mandate"
            } else {
                "Activated mandate"
            },
            activated.id,
            named_suffix(&activated),
            activated
                .confirmed_at
                .as_deref()
                .and_then(crate::schedule::owner_time_rfc3339)
                .unwrap_or_else(|| "unavailable".to_string()),
            activated.version,
            activated.min_review_secs / 60,
            activated.max_review_secs / 60
        ))
    }

    async fn list_snapshot(
        &self,
        args: &ManageMandatesArgs,
    ) -> anyhow::Result<(
        String,
        Option<crate::tools::objective_status::ObjectivePortfolio>,
    )> {
        if !Self::is_private_owner_control(args) {
            return Ok((
                "Mandates can only be listed by the owner in a verified private channel."
                    .to_string(),
                None,
            ));
        }
        let session_id = Self::owner_session(args)?;
        let mandates = self
            .state
            .list_mandates(Some(session_id), args.include_terminal.unwrap_or(false))
            .await?;
        // `list` is the owner audit surface, so enumerate both durable
        // objective collections. They are deliberately tracked as separate
        // collection memberships: a mandate controller with no cron schedule
        // cannot stand in for an unrelated scheduled-only goal.
        //
        // Mandates are session-scoped by typed authority (`list_mandates`,
        // `resolve_owned_mandate`). Scheduled goals have
        // no such authority model: `goal.session_id` only records which owner
        // channel created the goal, and the owner reaches the daemon through
        // several sessions. Enumerating that collection through the shared
        // loader keeps its membership identical to `scheduled_goal_runs`, so a
        // `complete` coverage here really does prove absence (R53).
        let (scheduled_goals, scheduled_coverage) =
            crate::tools::objective_status::load_scheduled_goal_collection(
                self.state.as_ref(),
                crate::tools::objective_status::SCHEDULED_COLLECTION_LIMIT,
            )
            .await?;

        let mandate_coverage = crate::tools::objective_status::ObjectiveCollectionCoverage::new(
            crate::tools::objective_status::ObjectiveCollection::MandateControllers,
            mandates.len(),
            mandates.len(),
        )?;
        let mut portfolio = crate::tools::objective_status::ObjectivePortfolio::default();
        portfolio.record_collection(mandate_coverage)?;
        portfolio.record_collection(scheduled_coverage)?;

        let mut mandate_details = String::new();
        for mandate in &mandates {
            let admission = match self.state.get_goal(&mandate.goal_id).await? {
                Some(goal) => controller_budget_admission_label(&goal, &mandate.review_effort),
                None => "controller-missing".to_string(),
            };
            mandate_details.push_str(&format!(
                "- {}{} [objective {}] [{} {} v{}] {} — next review {}; review admission {}; {}/cycle, {}/rolling-24h mutations\n",
                mandate.id,
                named_suffix(mandate),
                crate::tools::objective_status::objective_resource_id(&mandate.goal_id),
                mandate.status,
                mandate.autonomy_mode,
                mandate.version,
                mandate.objective,
                mandate.scheduled_next_review().unwrap_or("none, not active"),
                admission,
                mandate.authority.max_mutating_actions_per_cycle,
                mandate.authority.max_mutating_actions_per_rolling_24h
            ));
            // The mandate's controller goal carries the objective's live
            // lifecycle (schedule, latest run, recovery). Project the shared
            // authoritative readout here so an owner audit that consults the
            // mandate surface sees the same typed answer as
            // `scheduled_goal_runs` instead of reporting "unknown".
            let schedules = self.state.get_schedules_for_goal(&mandate.goal_id).await?;
            let runs = self.state.get_goal_runs(&mandate.goal_id).await?;
            let recovery = self
                .state
                .get_scheduled_recovery_state(&mandate.goal_id)
                .await?;
            let measurement_count = self
                .state
                .list_mandate_objective_measurements(&mandate.id, 500)
                .await?
                .len();
            let row = crate::tools::objective_status::objective_portfolio_row(
                &mandate.goal_id,
                crate::tools::objective_status::ObjectiveCollection::MandateControllers,
                &schedules,
                &runs,
                recovery.as_ref(),
                Some(mandate),
                measurement_count,
            )?;
            mandate_details.push_str(&format!(
                "  objective sources: {}\n",
                row.source_membership_label()
            ));
            mandate_details.push_str(&format!(
                "  objective status: {}\n",
                crate::tools::objective_status::render_objective_status_line(&row.status)
            ));
            portfolio.insert(row)?;
        }

        // Project every returned scheduled row even if the same subject was
        // already seen through a mandate. The portfolio merges only equal
        // goal IDs and verifies that both projections agree.
        for goal in &scheduled_goals {
            let schedules = self.state.get_schedules_for_goal(&goal.id).await?;
            let runs = self.state.get_goal_runs(&goal.id).await?;
            let recovery = self.state.get_scheduled_recovery_state(&goal.id).await?;
            let mandate = self.state.get_mandate_for_goal(&goal.id).await?;
            let measurement_count = match mandate.as_ref() {
                Some(mandate) => self
                    .state
                    .list_mandate_objective_measurements(&mandate.id, 500)
                    .await?
                    .len(),
                None => 0,
            };
            let row = crate::tools::objective_status::objective_portfolio_row(
                &goal.id,
                crate::tools::objective_status::ObjectiveCollection::ScheduledGoals,
                &schedules,
                &runs,
                recovery.as_ref(),
                mandate.as_ref(),
                measurement_count,
            )?;
            portfolio.insert(row)?;
        }

        let completeness =
            |coverage: crate::tools::objective_status::ObjectiveCollectionCoverage| {
                if coverage.is_complete() {
                    "complete"
                } else {
                    "partial"
                }
            };
        let mut output = format!(
            "Objective portfolio collections (absence is scoped to each collection):\n- {} [{}]: {}/{} returned ({})\n- {} [{}]: {}/{} returned ({})\n",
            mandate_coverage.collection.as_str(),
            mandate_coverage.collection.resource_id(),
            mandate_coverage.returned,
            mandate_coverage.total,
            completeness(mandate_coverage),
            scheduled_coverage.collection.as_str(),
            scheduled_coverage.collection.resource_id(),
            scheduled_coverage.returned,
            scheduled_coverage.total,
            completeness(scheduled_coverage),
        );

        // Put the compact cross-collection index before verbose mandate
        // policy details. Even if downstream display truncates, active
        // scheduled-only objectives are visible before the large tail.
        output.push_str(&format!(
            "Scheduled objective rows ({}):\n",
            scheduled_goals.len()
        ));
        if scheduled_goals.is_empty() {
            output.push_str("- none in the returned scheduled-goal collection\n");
        } else {
            for goal in &scheduled_goals {
                let row = portfolio.subject(&goal.id).ok_or_else(|| {
                    anyhow::anyhow!("scheduled objective {} is missing from portfolio", goal.id)
                })?;
                output.push_str(&format!(
                    "- {} — {} [{}]\n  objective sources: {}\n  objective status: {}\n",
                    crate::tools::objective_status::objective_resource_id(&goal.id),
                    aidaemon_events::text::truncate_str(&goal.description, 240),
                    goal.status,
                    row.source_membership_label(),
                    crate::tools::objective_status::render_objective_status_line(&row.status),
                ));
            }
        }
        if mandates.is_empty() {
            output.push_str("No mandates in this owner session.\n");
        } else {
            output.push_str(&format!("Mandates ({}):\n", mandates.len()));
            output.push_str(&mandate_details);
        }
        Ok((output, Some(portfolio)))
    }

    async fn list(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        self.list_snapshot(args).await.map(|(output, _)| output)
    }

    async fn get(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandates can only be inspected by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let owner = Self::owner_session(args)?;
        let mandate = self.resolve_selected_mandate(args, owner).await?;
        let strategy = mandate.strategy.as_ref().map(|value| {
            json!({
                "skill_name": value.skill_name,
                "snapshot_version": value.snapshot_version,
                "content_sha256": value.content_sha256,
                "description": value.description,
                "source": value.source,
                "body_persisted": true,
                "body_included": false,
            })
        });
        let section = args.section.as_deref().unwrap_or("summary");
        let output = match section {
            "summary" => {
                let controller = self
                    .state
                    .get_goal(&mandate.goal_id)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("mandate controller goal is missing"))?;
                let latest_decision = self
                    .state
                    .list_mandate_decisions(&mandate.id, 1)
                    .await?
                    .into_iter()
                    .next();
                let latest_intention = self
                    .state
                    .list_intentions(&mandate.id, 1)
                    .await?
                    .into_iter()
                    .next();
                let latest_learning_note = self
                    .state
                    .list_mandate_learning_notes(&mandate.id, 1)
                    .await?
                    .into_iter()
                    .next();
                let latest_goal_run = self
                    .state
                    .get_goal_runs(&mandate.goal_id)
                    .await?
                    .into_iter()
                    .next();
                let latest_mutation_receipts = if let Some(decision) = latest_decision.as_ref() {
                    self.state
                        .list_mandate_mutation_attempts_for_run(&decision.goal_run_id)
                        .await?
                } else {
                    Vec::new()
                };
                let recent_objective_measurements = self
                    .state
                    .list_mandate_objective_measurements(&mandate.id, 10)
                    .await?;
                let latest_measurement_observed_at = recent_objective_measurements
                    .iter()
                    .map(|measurement| measurement.observed_at.as_str())
                    .max();
                let next_measurement_at = mandate.objective_control.as_ref().and_then(|control| {
                    control.next_measurement_at(latest_measurement_observed_at)
                });
                let objective_progress = mandate.objective_control.as_ref().map(|control| {
                    let current = recent_objective_measurements
                        .iter()
                        .max_by(|left, right| left.observed_at.cmp(&right.observed_at))
                        .map(|measurement| measurement.value_micros);
                    json!({
                        "metric_name": control.metric_name,
                        "unit": control.unit,
                        "baseline_micros": control.baseline_micros,
                        "current_micros": current,
                        "target_micros": control.target_micros,
                        "direction": control.direction,
                        "target_reached": current.map(|value| control.target_reached(value)),
                        "measurement_count": recent_objective_measurements.len(),
                    })
                });
                json!({
                    "schema_version": 2,
                    "section": "summary",
                    "mandate": {
                        "id": mandate.id,
                        "controller_goal_id": mandate.goal_id,
                        "source_goal_id": mandate.source_goal_id,
                        "name": mandate.name,
                        "objective": mandate.objective,
                        "status": mandate.status,
                        "autonomy_mode": mandate.autonomy_mode,
                        "version": mandate.version,
                        "confirmed_at": mandate.confirmed_at,
                        "expires_at": mandate.expires_at,
                        "next_review_at": mandate.scheduled_next_review(),
                        "review_lease_active": mandate.review_lease_token.is_some(),
                        "review_lease_expires_at": mandate.review_lease_expires_at,
                        "suspension": mandate.suspension,
                        "created_at": mandate.created_at,
                        "updated_at": mandate.updated_at,
                        // The same times as the owner reads them; quote these
                        // to the owner, never the RFC 3339 values.
                        "owner_times": owner_times([
                            ("confirmed", mandate.confirmed_at.as_deref()),
                            ("next_review", mandate.scheduled_next_review()),
                            ("ends", mandate.expires_at.as_deref()),
                            ("created", Some(mandate.created_at.as_str())),
                        ]),
                    },
                    "review_policy": {
                        "effort": mandate.review_effort,
                        "minimum_seconds": mandate.min_review_secs,
                        "default_seconds": mandate.default_review_secs,
                        "maximum_seconds": mandate.max_review_secs,
                    },
                    "resource_policy": controller_budget_snapshot(&controller, &mandate.review_effort),
                    "authority_summary": {
                        "observations_allowed": mandate.authority.allow_observations,
                        "operation_scope_count": mandate.authority.operation_scopes.len(),
                        "tools": mandate.authority.allowed_tools,
                        "mutation_effects": mandate.authority.allowed_mutation_effects,
                        "max_mutations_per_cycle": mandate.authority.max_mutating_actions_per_cycle,
                        "max_mutations_per_rolling_24h": mandate.authority.max_mutating_actions_per_rolling_24h,
                        "minimum_seconds_between_mutations": mandate.authority.min_seconds_between_mutations,
                        "owner_alerts": mandate.authority.owner_alerts,
                    },
                    "owner_input_contract": {
                        "answer_question_records_guidance_only": true,
                        "answer_question_changes_authority": false,
                        "authority_changes_require_confirmed_update": true,
                        "exact_policy": "Call get with section=policy before constructing an update."
                    },
                    "strategy": strategy,
                    "objective_control": mandate.objective_control,
                    "objective_progress": objective_progress,
                    "next_measurement_at": next_measurement_at,
                    "recent_objective_measurements": recent_objective_measurements,
                    "latest_goal_run": latest_goal_run,
                    "latest_decision": latest_decision,
                    "latest_intention": latest_intention,
                    "latest_learning_note": latest_learning_note,
                    "latest_mutation_receipts": latest_mutation_receipts,
                    "more": {
                        "policy": "Call get again with section=policy for exact scopes and owner policy.",
                        "history": "Call get again with section=history and limit=1..10 for recent durable history."
                    }
                })
            }
            "policy" => json!({
                "schema_version": 2,
                "section": "policy",
                "mandate_id": mandate.id,
                "controller_goal_id": mandate.goal_id,
                "version": mandate.version,
                "objective": mandate.objective,
                "autonomy_mode": mandate.autonomy_mode,
                "authority": mandate.authority,
                "constraints": mandate.constraints,
                "success_criteria": mandate.success_criteria,
                "stop_conditions": mandate.stop_conditions,
                "review_effort": mandate.review_effort,
                // Update-native names and units, so an update built from this
                // view round-trips the current review bounds instead of
                // leaving a structured-output provider to invent them.
                "min_review_minutes": mandate.min_review_secs / 60,
                "default_review_minutes": mandate.default_review_secs / 60,
                "max_review_minutes": mandate.max_review_secs / 60,
                "expires_at": mandate.expires_at,
                "strategy": strategy,
                "objective_control": mandate.objective_control,
            }),
            "history" => {
                let limit = args.limit.unwrap_or(3).clamp(1, 10);
                let decisions = self
                    .state
                    .list_mandate_decisions(&mandate.id, limit)
                    .await?;
                let intentions = self.state.list_intentions(&mandate.id, limit).await?;
                let learning = self
                    .state
                    .list_mandate_learning_notes(&mandate.id, limit)
                    .await?;
                let goal_runs = self
                    .state
                    .get_goal_runs(&mandate.goal_id)
                    .await?
                    .into_iter()
                    .take(limit as usize)
                    .collect::<Vec<_>>();
                let mut mutation_receipts = Vec::new();
                for decision in &decisions {
                    mutation_receipts.extend(
                        self.state
                            .list_mandate_mutation_attempts_for_run(&decision.goal_run_id)
                            .await?,
                    );
                }
                json!({
                    "schema_version": 2,
                    "section": "history",
                    "mandate_id": mandate.id,
                    "controller_goal_id": mandate.goal_id,
                    "limit": limit,
                    "recent_goal_runs": goal_runs,
                    "recent_decisions": decisions,
                    "recent_intentions": intentions,
                    "recent_learning_notes": learning,
                    "recent_mutation_receipts": mutation_receipts,
                })
            }
            _ => anyhow::bail!("section must be summary, policy, or history"),
        };
        Ok(serde_json::to_string(&output)?)
    }

    async fn transition(&self, args: &ManageMandatesArgs, action: &str) -> anyhow::Result<String> {
        if !Self::is_private_owner_control(args) {
            return Ok(
                "Mandates can only be changed by the owner in a verified private channel."
                    .to_string(),
            );
        }
        let owner = Self::owner_session(args)?;
        let mandate = self.resolve_selected_mandate(args, owner).await?;
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

        // An unanswered ASK resumes on its own at its reconsider time. A late
        // answer still lands as bounded owner guidance on the active mandate.
        if action == "answer_question" && mandate.status == MandateStatus::Active {
            let context = resumed_context
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("guidance is required for answer_question"))?;
            let asked = self
                .state
                .list_mandate_decisions(&mandate.id, 50)
                .await?
                .iter()
                .any(|decision| decision.outcome == MandateDecisionOutcome::Ask);
            anyhow::ensure!(
                asked,
                "this mandate has no open question; use update to change its policy"
            );
            anyhow::ensure!(
                self.state
                    .record_active_mandate_owner_guidance(
                        &mandate.id,
                        mandate.version,
                        owner,
                        context,
                    )
                    .await?,
                "mandate changed before the answer could be recorded"
            );
            return Ok(format!(
                "Mandate {} recorded your late answer as bounded owner guidance for its next review. Immutable authority is unchanged: no tool, operation, effect, account, URL, or query scope was added. Use the separately owner-confirmed update workflow for any authority change.",
                mandate.id
            ));
        }
        if action == "answer_question"
            || action == "resolve_reconciliation"
            || (action == "resume" && mandate.status == MandateStatus::AwaitingInput)
        {
            anyhow::ensure!(
                mandate.status == MandateStatus::AwaitingInput,
                "{action} requires an awaiting-input mandate"
            );
            let suspension = mandate
                .suspension
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("awaiting-input mandate has no typed suspension"))?;
            let expected_kind = suspension.kind;
            let resolution = match expected_kind {
                MandateSuspensionKind::AwaitingAnswer => {
                    anyhow::ensure!(
                        action == "answer_question",
                        "this mandate is awaiting an answer; use answer_question"
                    );
                    anyhow::ensure!(
                        guidance.is_some(),
                        "guidance is required for answer_question"
                    );
                    None
                }
                MandateSuspensionKind::OwnerPaused => {
                    anyhow::bail!("owner-paused mandates must use ordinary resume")
                }
                MandateSuspensionKind::ReconciliationRequired
                | MandateSuspensionKind::ExecutionLeaseLost
                | MandateSuspensionKind::ReviewFailed
                | MandateSuspensionKind::AuthorityRevokedWithUnresolvedMutation => {
                    anyhow::ensure!(
                        action == "resolve_reconciliation",
                        "this safety suspension requires resolve_reconciliation"
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
                }
            };
            let guidance = guidance.unwrap_or("Owner resolved the mandate's suspension.");
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
            return Ok(
                if action == "answer_question" {
                    format!(
                    "Mandate {} is active after recording bounded owner guidance. Immutable authority is unchanged: no tool, operation, effect, account, URL, or query scope was added. Use the separately owner-confirmed update workflow for any authority change, and do not claim this answer authorized one.",
                    mandate.id
                )
                } else {
                    format!("Mandate {} is active after typed {}.", mandate.id, action)
                },
            );
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

    /// Validate an `update` exactly as it commits, without storing it.
    /// `Ok(Err(answer))`: the call answers without changing the mandate.
    async fn update_plan(
        &self,
        args: &ManageMandatesArgs,
    ) -> anyhow::Result<Result<UpdatePlan, String>> {
        if !Self::is_private_owner_control(args) {
            return Ok(Err(
                "Mandate authority can only be changed by the owner in a verified private channel."
                    .to_string(),
            ));
        }
        let owner = Self::owner_session(args)?;
        let mut mandate = self.resolve_selected_mandate(args, owner).await?;
        let previous_autonomy_mode = mandate.autonomy_mode;
        let has_change = args.objective.is_some()
            || args.autonomy_mode.is_some()
            || args.allow_observations.is_some()
            || args.operation_scopes.is_some()
            || args.allowed_tools.is_some()
            || args.allowed_mutation_effects.is_some()
            || args.allowed_target_prefixes.is_some()
            || args.max_mutating_actions_per_cycle.is_some()
            || args.max_mutating_actions_per_rolling_24h.is_some()
            || args.min_seconds_between_mutations.is_some()
            || args.owner_alerts.is_some()
            || args.constraints.is_some()
            || args.success_criteria.is_some()
            || args.stop_conditions.is_some()
            || args.min_review_minutes.is_some()
            || args.max_review_minutes.is_some()
            || args.default_review_minutes.is_some()
            || args.expires_at.is_some()
            || args.review_effort.is_some()
            || args.budget_per_cycle.is_some()
            || args.budget_daily.is_some()
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
        mandate.autonomy_mode = Self::autonomy_mode_from_args(args, Some(mandate.autonomy_mode))?;
        let enabled_autopilot =
            !previous_autonomy_mode.is_autopilot() && mandate.autonomy_mode.is_autopilot();
        let previous_owner_alerts = mandate.authority.owner_alerts;
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
                    owner_alerts: previous_owner_alerts,
                };
            }
        }
        mandate.authority.owner_alerts = args.owner_alerts.unwrap_or(previous_owner_alerts);
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
            mandate.strategy = self.approved_strategy_snapshot(args)?;
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
            if enabled_autopilot
                && (mandate.min_review_secs..=mandate.max_review_secs)
                    .contains(&AUTOPILOT_DEFAULT_REVIEW_SECS)
            {
                AUTOPILOT_DEFAULT_REVIEW_SECS
            } else {
                mandate.default_review_secs
            },
            "default_review_minutes",
        )?;
        anyhow::ensure!(
            mandate.min_review_secs <= mandate.default_review_secs
                && mandate.default_review_secs <= mandate.max_review_secs,
            "{}",
            review_bounds_error(
                mandate.min_review_secs,
                mandate.default_review_secs,
                mandate.max_review_secs
            )
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
        if let Some(reason) = unreachable_scope_targets(&mandate.authority.operation_scopes) {
            anyhow::bail!(reason);
        }

        let controller = self
            .state
            .get_goal(&mandate.goal_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("mandate controller goal is missing"))?;
        let current_capacity = (
            controller
                .budget_per_check
                .ok_or_else(|| anyhow::anyhow!("mandate controller capacity is missing"))?,
            controller
                .budget_daily
                .ok_or_else(|| anyhow::anyhow!("mandate controller daily capacity is missing"))?,
        );
        let timing = ActivationTiming {
            duration_secs: None,
            expires_at: mandate.expires_at.clone(),
            normalized_redundant_expiry: false,
        };
        let capacity = Self::resolved_review_capacity(
            args,
            mandate.default_review_secs,
            &timing,
            Some(current_capacity),
            if enabled_autopilot && args.review_effort.is_none() {
                None
            } else {
                ReviewEffort::parse(&mandate.review_effort)
            },
            enabled_autopilot.then_some(ReviewEffort::Thorough),
        )?;
        mandate.review_effort = capacity.display_mode().to_string();
        let proposed_policy_version = mandate.version + 1;
        let warnings =
            Self::confirmation_card(&mandate, capacity, None, proposed_policy_version).into_lines();
        let card = MandateCard {
            summary: if mandate.autonomy_mode.is_autopilot() {
                format!("Enable or update Autopilot: {}", mandate.objective)
            } else {
                format!("Update delegated mandate: {}", mandate.objective)
            },
            risk: aidaemon_core::ApprovalRisk::High,
            warnings,
            autopilot: mandate.autonomy_mode.is_autopilot(),
        };
        Ok(Ok(UpdatePlan {
            mandate,
            capacity,
            card,
        }))
    }

    /// Store an update the owner confirmed through the agent kernel; a
    /// declined confirmation leaves the existing envelope unchanged.
    async fn update(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
        let plan = match self.update_plan(args).await? {
            Err(answer) => return Ok(answer),
            Ok(plan) => plan,
        };
        Self::require_kernel_approval(args)?;
        let UpdatePlan {
            mut mandate,
            capacity,
            card: _,
        } = plan;
        mandate.version += 1;
        let updated_at = chrono::Utc::now().to_rfc3339();
        mandate.next_review_at = updated_at.clone();
        mandate.updated_at = updated_at;
        self.state.update_mandate(&mandate).await?;
        self.state
            .set_goal_budgets(
                &mandate.goal_id,
                Some(capacity.per_cycle),
                Some(capacity.daily),
            )
            .await?;
        Ok(format!(
            "Updated mandate {} to policy version {} in {} mode with {} review effort and automatically managed capacity. In-flight decisions on older versions are revoked; the next review is due now.",
            mandate.id, mandate.version, mandate.autonomy_mode, capacity.display_mode(),
        ))
    }

    async fn record_measurement(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
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

    async fn record_decision(&self, args: &ManageMandatesArgs) -> anyhow::Result<String> {
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
        let mandate = self.resolve_selected_mandate(args, owner).await?;
        let intentions = self
            .state
            .list_intentions(&mandate.id, args.limit.unwrap_or(20).clamp(1, 100))
            .await?;
        Ok(serde_json::to_string_pretty(&intentions)?)
    }
}

#[derive(Debug, Default, Deserialize)]
struct ManageMandatesArgs {
    action: String,
    mandate_id: Option<String>,
    name: Option<String>,
    objective: Option<String>,
    autonomy_mode: Option<String>,
    source_goal_id: Option<String>,
    allow_observations: Option<bool>,
    operation_scopes: Option<Vec<MandateOperationScope>>,
    allowed_tools: Option<Vec<String>>,
    allowed_mutation_effects: Option<Vec<String>>,
    allowed_target_prefixes: Option<Vec<String>>,
    max_mutating_actions_per_cycle: Option<u32>,
    max_mutating_actions_per_rolling_24h: Option<u32>,
    min_seconds_between_mutations: Option<u32>,
    owner_alerts: Option<bool>,
    constraints: Option<Vec<String>>,
    success_criteria: Option<Vec<String>>,
    stop_conditions: Option<Vec<String>>,
    strategy_skill: Option<String>,
    clear_strategy: Option<bool>,
    min_review_minutes: Option<i64>,
    max_review_minutes: Option<i64>,
    default_review_minutes: Option<i64>,
    duration_minutes: Option<i64>,
    expires_at: Option<String>,
    priority: Option<String>,
    review_effort: Option<String>,
    /// Backward-compatible internal fields. Raw token arithmetic is no longer
    /// part of the owner/model-facing schema; use review_effort instead.
    budget_per_cycle: Option<i64>,
    budget_daily: Option<i64>,
    include_terminal: Option<bool>,
    section: Option<String>,
    limit: Option<i64>,
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
    guidance: Option<String>,
    reconciliation_resolution: Option<String>,
    /// The dispatcher's facts about this call ([`Self::bind_facts`]),
    /// never arguments.
    #[serde(skip)]
    call_session: Option<String>,
    #[serde(skip)]
    call_role: Option<UserRole>,
    #[serde(skip)]
    call_visibility: Option<ChannelVisibility>,
    #[serde(skip)]
    call_goal_id: Option<String>,
    #[serde(skip)]
    call_goal_run_id: Option<String>,
    #[serde(skip)]
    call_task_attempt_id: Option<String>,
    /// The agent kernel judged this call (and asked the owner where
    /// `owner_approval_request` requires it).
    #[serde(skip)]
    call_kernel_approved: bool,
}

impl ManageMandatesArgs {
    /// Take the caller's identity and run fence from the dispatcher's facts.
    fn bind_facts(&mut self, facts: &ToolCallContext) {
        self.call_session = Some(facts.session_id.clone());
        self.call_role = facts.user_role;
        self.call_visibility = facts.channel_visibility;
        self.call_goal_id = facts.goal_id.clone();
        self.call_goal_run_id = facts.goal_run_id.clone();
        self.call_task_attempt_id = facts.task_attempt_id.clone();
        self.call_kernel_approved =
            facts.kernel_approved == Some(crate::traits::ApprovalScope::Call);
    }

    /// Normalize compatibility noise from structured-tool providers before any
    /// action-specific validation. Some providers materialize every optional
    /// scalar using an empty string or its schema minimum. Those placeholders
    /// are not evidence that the caller intended to change a field.
    fn normalize_for_action(&mut self) {
        fn trim_optional(value: &mut Option<String>) {
            let normalized = value
                .take()
                .map(|raw| raw.trim().to_string())
                .filter(|raw| !raw.is_empty());
            *value = normalized;
        }

        trim_optional(&mut self.mandate_id);
        trim_optional(&mut self.name);
        trim_optional(&mut self.objective);
        trim_optional(&mut self.autonomy_mode);
        trim_optional(&mut self.source_goal_id);
        trim_optional(&mut self.strategy_skill);
        trim_optional(&mut self.expires_at);
        trim_optional(&mut self.priority);
        trim_optional(&mut self.review_effort);
        trim_optional(&mut self.section);
        trim_optional(&mut self.outcome);
        trim_optional(&mut self.activity_level);
        trim_optional(&mut self.rationale);
        trim_optional(&mut self.measurement_observed_at);
        trim_optional(&mut self.question);
        trim_optional(&mut self.owner_alert);
        trim_optional(&mut self.termination_kind);
        trim_optional(&mut self.termination_match);
        trim_optional(&mut self.intention);
        trim_optional(&mut self.value_criterion);
        trim_optional(&mut self.expected_benefit);
        trim_optional(&mut self.risk);
        trim_optional(&mut self.invalidation_criteria);
        trim_optional(&mut self.learning_note);
        trim_optional(&mut self.guidance);
        trim_optional(&mut self.reconciliation_resolution);

        if self.action == "update" {
            // These are creation metadata, not authority. A monolithic tool
            // schema is exposed to providers that populate all optional fields,
            // so presence cannot safely mean "change requested" on update.
            // Preserve the existing values and use the update-native expires_at
            // field for expiry changes.
            self.source_goal_id = None;
            self.priority = None;
            self.duration_minutes = None;
        }
    }
}

fn bounded_text(max_length: usize) -> Value {
    json!({ "type": "string", "maxLength": max_length })
}

fn text_enum(values: &[&str]) -> Value {
    json!({ "type": "string", "enum": values })
}

fn integer_min(minimum: i64) -> Value {
    json!({ "type": "integer", "minimum": minimum })
}

fn integer_range(minimum: i64, maximum: i64) -> Value {
    json!({ "type": "integer", "minimum": minimum, "maximum": maximum })
}

fn bounded_text_array(max_items: usize, max_length: usize) -> Value {
    json!({
        "type": "array",
        "maxItems": max_items,
        "items": bounded_text(max_length)
    })
}

fn described(mut schema: Value, description: &str) -> Value {
    schema["description"] = Value::String(description.to_string());
    schema
}

const MANAGE_MANDATE_ACTIONS: &[&str] = &[
    "draft",
    "create",
    "list",
    "get",
    "update",
    "pause",
    "resume",
    "answer_question",
    "resolve_reconciliation",
    "cancel",
    "record_measurement",
    "record_decision",
    "list_intentions",
];
const OPERATION_SCOPE_REQUIRED: &[&str] = &[
    "tool",
    "operation",
    "kind",
    "target_prefixes",
    "allowed_query_params",
    "mutation_effects",
];
const MANDATE_OUTCOMES: &[&str] = &["act", "wait", "ask", "stop"];
const ACTIVITY_LEVELS: &[&str] = &["quiet", "active", "urgent"];
const TERMINATION_KINDS: &[&str] = &[
    "success_criteria_satisfied",
    "stop_condition_met",
    "safety_termination",
];
const RECONCILIATION_RESOLUTIONS: &[&str] = &[
    "confirmed_effect_occurred",
    "confirmed_no_effect",
    "abandon_attempt",
];
const HTTP_TOOLS: &[&str] = &["http_request", "web_fetch"];
const HTTP_METHODS: &[&str] = &["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"];
const OPERATION_KINDS: &[&str] = &["observation", "mutation"];
const MUTATION_EFFECTS: &[&str] = &["remote_mutation", "external_delivery"];
const AUTONOMY_MODES: &[&str] = &["bounded", "autopilot"];
const PRIORITIES: &[&str] = &["low", "medium", "high", "critical"];
const REVIEW_EFFORTS: &[&str] = &["efficient", "balanced", "thorough"];
const GET_SECTIONS: &[&str] = &["summary", "policy", "history"];
const DEFAULT_REVIEW_FIELD_DESCRIPTION: &str =
    "How often the mandate runs, in minutes (every hour = 60). For a fixed interval set min_review_minutes and max_review_minutes to the same value.";
const FIXED_DEADLINE_DESCRIPTION: &str =
    "RFC3339 fixed deadline. Supported by both create and update.";
const OWNER_ALERTS_DESCRIPTION: &str =
    "create/update: true when the owner asked to be told what the mandate finds (e.g. tell me when it is down). Lets a review's WAIT carry owner_alert; grants no tool, target or effect.";
const OWNER_ALERT_DESCRIPTION: &str =
    "WAIT only, when the policy grants owner_alerts: the one finding the owner asked to hear about, in a sentence.";
const STOP_ONLY_DESCRIPTION: &str = "STOP only; omit for ACT, WAIT, and ASK.";
const VALUE_CRITERION_DESCRIPTION: &str =
    "ACT: exact owner-confirmed success criterion advanced; legacy uses the exact objective.";
const EXPECTED_BENEFIT_DESCRIPTION: &str = "ACT: benefit of acting now instead of waiting.";
const RISK_DESCRIPTION: &str = "ACT: expected cost or downside, or why none is material.";
const INVALIDATION_DESCRIPTION: &str =
    "ACT: evidence making the intervention unsafe or not worthwhile.";

fn op_schema() -> Value {
    json!({
        "type": "array", "minItems": 1, "maxItems": 64,
        "items": { "type": "object",
            "properties": {
                "tool": text_enum(HTTP_TOOLS),
                "operation": text_enum(HTTP_METHODS),
                "kind": text_enum(OPERATION_KINDS),
                "target_prefixes": { "type": "array", "minItems": 1, "maxItems": 16, "items": { "type": "string", "maxLength": 2048 } },
                "allowed_query_params": { "type": "array", "maxItems": 16, "items": { "type": "string", "maxLength": 128 } },
                "mutation_effects": { "type": "array", "maxItems": 2, "items": text_enum(MUTATION_EFFECTS) }
            },
            "required": OPERATION_SCOPE_REQUIRED,
            "additionalProperties": false
        }
    })
}

#[async_trait]
impl Tool for ManageMandatesTool {
    fn name(&self) -> &str {
        "manage_mandates"
    }

    /// A mandate change is confirmed on its goal or Autopilot card (confirm
    /// or cancel), which offers no session or persistent grant.
    fn approval_card_kind(
        &self,
        request: &aidaemon_core::OwnerApprovalRequest,
    ) -> crate::traits::ApprovalKind {
        Self::card_kind_for(request)
    }

    fn description(&self) -> &str {
        "Draft, create, inspect, and govern owner-confirmed ongoing mandates."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "manage_mandates",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": text_enum(MANAGE_MANDATE_ACTIONS),
                    "mandate_id": bounded_text(256), "name": described(bounded_text(MAX_NAME_TEXT), NAME_FIELD_DESCRIPTION), "objective": bounded_text(MAX_OBJECTIVE_TEXT), "autonomy_mode": text_enum(AUTONOMY_MODES), "source_goal_id": described(bounded_text(256), SOURCE_GOAL_FIELD_DESCRIPTION),
                    "allow_observations": { "type": "boolean" },
                    "operation_scopes": op_schema(),
                    "max_mutating_actions_per_cycle": integer_range(0, 24), "max_mutating_actions_per_rolling_24h": integer_range(0, 24), "min_seconds_between_mutations": integer_min(0),
                    "owner_alerts": described(json!({"type":"boolean"}), OWNER_ALERTS_DESCRIPTION), "owner_alert": described(bounded_text(MAX_QUESTION_TEXT), OWNER_ALERT_DESCRIPTION),
                    "constraints": described(bounded_text_array(MAX_POLICY_ENTRIES, MAX_POLICY_ENTRY_TEXT), CONSTRAINT_FIELD_DESCRIPTION), "success_criteria": described(bounded_text_array(MAX_POLICY_ENTRIES, MAX_POLICY_ENTRY_TEXT), SUCCESS_FIELD_DESCRIPTION), "stop_conditions": described(bounded_text_array(MAX_POLICY_ENTRIES, MAX_POLICY_ENTRY_TEXT), STOP_FIELD_DESCRIPTION),
                    "strategy_skill": bounded_text(256), "clear_strategy": { "type": "boolean" }, "min_review_minutes": integer_min(1), "max_review_minutes": integer_min(1), "default_review_minutes": described(integer_min(1), DEFAULT_REVIEW_FIELD_DESCRIPTION),
                    "duration_minutes": described(integer_min(1), DURATION_FIELD_DESCRIPTION), "expires_at": described(json!({"type":"string"}), FIXED_DEADLINE_DESCRIPTION), "priority": described(text_enum(PRIORITIES), PRIORITY_FIELD_DESCRIPTION), "review_effort": text_enum(REVIEW_EFFORTS),
                    "include_terminal": { "type": "boolean" }, "section": text_enum(GET_SECTIONS), "limit": integer_range(1, 10),
                    "outcome": text_enum(MANDATE_OUTCOMES), "activity_level": text_enum(ACTIVITY_LEVELS), "rationale": bounded_text(MAX_RATIONALE_TEXT),
                    "observations": bounded_text_array(MAX_OBSERVATIONS, MAX_OBSERVATION_TEXT), "evidence_receipt_ids": described(bounded_text_array(MAX_EVIDENCE_RECEIPTS, 256), EVIDENCE_RECEIPT_IDS_DESCRIPTION), "question": bounded_text(MAX_QUESTION_TEXT),
                    "measurement_value_micros": { "type": "integer", "description": "record_measurement only, for a mandate confirmed with a measured objective. Millionths of one unit (25 engagements = 25000000)." }, "measurement_confidence_bps": integer_range(0, 10000), "measurement_observed_at": { "type": "string" },
                    "termination_kind": described(text_enum(TERMINATION_KINDS), STOP_ONLY_DESCRIPTION), "termination_match": described(bounded_text(MAX_POLICY_ENTRY_TEXT), STOP_ONLY_DESCRIPTION), "reconsider_minutes": integer_min(1),
                    "intention": bounded_text(MAX_INTENTION_TEXT), "value_criterion": described(bounded_text(MAX_OBJECTIVE_TEXT), VALUE_CRITERION_DESCRIPTION), "expected_benefit": described(bounded_text(MAX_INTENTION_METADATA_TEXT), EXPECTED_BENEFIT_DESCRIPTION), "risk": described(bounded_text(MAX_INTENTION_METADATA_TEXT), RISK_DESCRIPTION), "invalidation_criteria": described(bounded_text(MAX_INTENTION_METADATA_TEXT), INVALIDATION_DESCRIPTION),
                    "learning_note": bounded_text(MAX_LEARNING_NOTE_TEXT), "learning_evidence_receipt_ids": described(bounded_text_array(MAX_EVIDENCE_RECEIPTS, 256), LEARNING_EVIDENCE_RECEIPT_IDS_DESCRIPTION),
                    "guidance": bounded_text(MAX_GUIDANCE_ENTRY_TEXT), "reconciliation_resolution": text_enum(RECONCILIATION_RESOLUTIONS)
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::traits::ToolCallContext,
    ) -> anyhow::Result<crate::traits::ToolCallOutcome> {
        let facts = facts.clone();

        let mut args: ManageMandatesArgs = serde_json::from_str(arguments)?;
        args.normalize_for_action();
        args.bind_facts(&facts);
        let (output, objective_evidence) = if args.action == "list" {
            let (output, portfolio) = self.list_snapshot(&args).await?;
            (
                output,
                portfolio.map(|portfolio| portfolio.receipt_evidence()),
            )
        } else {
            (self.run(arguments, &facts).await?, None)
        };
        let trimmed = output.trim_start();
        let outcome_status = if trimmed.starts_with("Mandates can only")
            || trimmed.starts_with("Mandate authority can only")
            || trimmed.starts_with("Mandate intentions can only")
        {
            ToolOutcomeStatus::Blocked
        } else if trimmed.contains("confirmation was declined")
            || trimmed.starts_with("Mandate update cancelled")
            || trimmed.contains("pending mandate was cancelled")
        {
            ToolOutcomeStatus::CompletedWithNegativeResult
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
        if let Some((observations, collections)) = objective_evidence {
            outcome = outcome.with_observation_evidence(observations, collections);
        }
        // A committed non-action decision (WAIT, ASK, STOP) completes the
        // fenced review: the run ends on the durable decision's summary,
        // without another model call.
        if args.action == "record_decision" && outcome_status == ToolOutcomeStatus::Succeeded {
            if let Some(summary) = self.committed_non_action_summary(&facts).await {
                outcome = outcome.with_finish(aidaemon_core::Finish::succeeded(summary));
            }
        }
        Ok(outcome)
    }
}

impl ManageMandatesTool {
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
}

impl ManageMandatesTool {
    /// Run one call with the dispatcher's `facts`.
    pub async fn run(&self, arguments: &str, facts: &ToolCallContext) -> anyhow::Result<String> {
        let mut args: ManageMandatesArgs = serde_json::from_str(arguments)?;
        args.normalize_for_action();
        args.bind_facts(facts);
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
            "record_measurement" => self.record_measurement(&args).await,
            "list_intentions" => self.list_intentions(&args).await,
            other => anyhow::bail!("unknown manage_mandates action `{other}`"),
        }
    }
}

#[async_trait]
impl crate::traits::ToolSemantics for ManageMandatesTool {
    /// Creating and updating a mandate are confirmed by the
    /// owner before they commit. The dispatcher runs the call only with that
    /// answer; a decline is recorded by the kernel and changes nothing.
    async fn owner_approval_request(
        &self,
        call: &crate::traits::ApprovalCall<'_>,
        facts: &ToolCallContext,
    ) -> crate::traits::OwnerApprovalNeed {
        if call.refused {
            return crate::traits::OwnerApprovalNeed::NotNeeded;
        }
        self.owner_approval(call.arguments, facts).await
    }

    /// Reads (draft, list, get, list_intentions) have no effects. Every
    /// other action changes durable delegated policy or schedules work, so
    /// it stays undeclared; a declined create or update is
    /// recorded by its approval subject, so the identical change is not put
    /// to the owner again.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| value.get("action")?.as_str().map(str::to_string));
        match action.as_deref() {
            Some("draft" | "list" | "get" | "list_intentions") => {
                aidaemon_core::CallEffects::none()
            }
            _ => aidaemon_core::CallEffects::unknown(),
        }
    }

    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            always_offered: true,
            goal_context: crate::traits::GoalContext::MandateRun,
            ..Default::default()
        }
    }

    fn owner_control_plane(&self) -> bool {
        true
    }

    fn owner_control_plane_call(&self, arguments: &str) -> bool {
        // Read-only actions, plus create and update, which commit only after
        // the owner confirms through the agent kernel. Transitions (pause,
        // resume, cancel, answer_question, resolve_reconciliation) take
        // effect without a confirmation and are not admitted here.
        serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|args| args.get("action")?.as_str().map(str::to_string))
            .is_some_and(|action| {
                matches!(
                    action.as_str(),
                    "draft" | "create" | "update" | "list" | "get" | "list_intentions"
                )
            })
    }

    fn tool_role(&self) -> ToolRole {
        ToolRole::Management
    }

    fn stable_observation_subjects(&self) -> Vec<crate::traits::StableObservationSubject> {
        vec![
            crate::tools::objective_status::ObjectiveCollection::MandateControllers
                .stable_subject(),
            crate::tools::objective_status::ObjectiveCollection::ScheduledGoals.stable_subject(),
        ]
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
            Some(action)
                if crate::mandates::policy::deliberator_protocol_class(action).is_some() =>
            {
                ToolCallSemantics::administrative()
            }
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

/// A dispatcher fact the call needs (from its `ToolCallContext`).
fn required_call_fact<'a>(value: Option<&'a str>, fact: &str) -> anyhow::Result<&'a str> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("the {fact} fact is missing from the call context"))
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

/// The owner-given name a draft or create carries, bounded and single-line.
fn mandate_name(args: &ManageMandatesArgs) -> anyhow::Result<Option<String>> {
    let Some(name) = args.name.as_deref() else {
        return Ok(None);
    };
    validate_bounded_text(name, "name", MAX_NAME_TEXT)?;
    anyhow::ensure!(
        !name.chars().any(char::is_control),
        "name must be a single line"
    );
    Ok(Some(name.to_string()))
}

/// ` "name"` after a mandate's ID in owner-facing lines, or nothing.
fn named_suffix(mandate: &Mandate) -> String {
    mandate
        .name
        .as_deref()
        .map(|name| format!(" \"{name}\""))
        .unwrap_or_default()
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

fn normalized_termination_fields(
    outcome: MandateDecisionOutcome,
    termination_kind: Option<&str>,
    termination_match: Option<&str>,
) -> anyhow::Result<(Option<MandateTerminationKind>, Option<String>)> {
    // Some structured-tool providers populate every optional schema field with
    // a default. The explicit typed outcome is authoritative, so STOP-only
    // metadata on ACT/WAIT/ASK must not turn a safe non-action into a failed
    // review. It is ignored rather than reinterpreted as a termination.
    if outcome != MandateDecisionOutcome::Stop {
        return Ok((None, None));
    }
    let kind = termination_kind
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
    anyhow::ensure!(kind.is_some(), "STOP requires termination_kind");
    let matched = bounded_optional_text(
        termination_match,
        "termination_match",
        MAX_POLICY_ENTRY_TEXT,
    )?;
    Ok((kind, matched))
}

/// A scope whose tool can never reach its target fails at the first review,
/// long after the owner confirmed it. `web_fetch` refuses local and private
/// hosts by design; reject such a scope while it is still a draft.
fn unreachable_scope_targets(scopes: &[MandateOperationScope]) -> Option<String> {
    scopes
        .iter()
        .filter(|scope| scope.tool == "web_fetch")
        .flat_map(|scope| scope.target_prefixes.iter())
        .find_map(|prefix| {
            aidaemon_tools::web::validate_url_for_ssrf(prefix)
                .err()
                .map(|reason| {
                    format!(
                        "web_fetch cannot reach {prefix} ({reason}); use http_request for local or private targets"
                    )
                })
        })
}

fn validate_act_evidence(
    outcome: MandateDecisionOutcome,
    observations: &[String],
    evidence_receipt_ids: &[String],
) -> anyhow::Result<()> {
    if outcome == MandateDecisionOutcome::Act {
        anyhow::ensure!(
            !observations.is_empty() && !evidence_receipt_ids.is_empty(),
            "ACT requires at least one current-run sourced observation and its durable evidence receipt; choose WAIT when no evidence shows that intervention is worthwhile"
        );
    }
    Ok(())
}

fn normalized_learning_note(
    learning_note: Option<&str>,
    learning_evidence_receipt_ids: Option<&[String]>,
) -> anyhow::Result<Option<(String, Vec<String>)>> {
    let Some(summary) =
        bounded_optional_text(learning_note, "learning_note", MAX_LEARNING_NOTE_TEXT)?
    else {
        // Structured-tool providers may populate this optional companion field
        // even when learning_note is empty. With no note there is nothing to
        // persist, so ignore the orphan defaults before the decision commit.
        return Ok(None);
    };
    let evidence = clean_strings(learning_evidence_receipt_ids);
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
    Ok(Some((summary, evidence)))
}

fn learning_note_fields_complete(learning_note: Option<&str>, evidence: Option<&[String]>) -> bool {
    learning_note.is_some_and(|value| !value.trim().is_empty())
        && evidence.is_some_and(|values| values.iter().any(|value| !value.trim().is_empty()))
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

/// Which review setting is out of order, with the allowed range.
fn review_bounds_error(min_secs: i64, default_secs: i64, max_secs: i64) -> String {
    if min_secs > max_secs {
        format!(
            "min_review_minutes ({}) must not be more than max_review_minutes ({})",
            min_secs / 60,
            max_secs / 60
        )
    } else {
        format!(
            "default_review_minutes is {}; allowed: min_review_minutes ({}) to max_review_minutes ({})",
            default_secs / 60,
            min_secs / 60,
            max_secs / 60
        )
    }
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

#[derive(Debug, PartialEq, Eq)]
struct ActivationTiming {
    duration_secs: Option<i64>,
    expires_at: Option<String>,
    normalized_redundant_expiry: bool,
}

fn activation_timing(args: &ManageMandatesArgs) -> anyhow::Result<ActivationTiming> {
    let duration_secs = args
        .duration_minutes
        .map(|minutes| minutes_to_secs(Some(minutes), 0, "duration_minutes"))
        .transpose()?;
    let normalized_redundant_expiry = duration_secs.is_some() && args.expires_at.is_some();
    Ok(ActivationTiming {
        duration_secs,
        expires_at: if duration_secs.is_some() {
            None
        } else {
            args.expires_at.clone()
        },
        normalized_redundant_expiry,
    })
}

fn expected_default_review_cycles(
    default_review_secs: i64,
    timing: &ActivationTiming,
) -> anyhow::Result<i64> {
    anyhow::ensure!(
        default_review_secs > 0,
        "default review interval must be positive"
    );
    let horizon_secs = if let Some(duration_secs) = timing.duration_secs {
        duration_secs
    } else if let Some(expires_at) = timing.expires_at.as_deref() {
        let expires_at = chrono::DateTime::parse_from_rfc3339(expires_at)
            .map_err(|_| anyhow::anyhow!("expires_at must be an RFC3339 timestamp"))?
            .with_timezone(&chrono::Utc);
        (expires_at - chrono::Utc::now()).num_seconds().max(1)
    } else {
        24 * 60 * 60
    }
    .clamp(1, 24 * 60 * 60);

    // The first review is immediate, then subsequent reviews occur at the
    // default interval. Exclude a cycle exactly at expiration.
    Ok(((horizon_secs - 1) / default_review_secs) + 1)
}

fn controller_budget_snapshot(goal: &crate::traits::Goal, review_effort: &str) -> Value {
    let now = chrono::Utc::now();
    let today = now.date_naive().to_string();
    let used_today = if goal.tokens_used_day == today {
        goal.tokens_used_today.max(0)
    } else {
        0
    };
    let remaining = goal
        .budget_daily
        .map(|daily| daily.saturating_sub(used_today).max(0));
    let can_fund_full_cycle = match (remaining, goal.budget_per_check) {
        (Some(remaining), Some(per_cycle)) => remaining >= per_cycle.max(0),
        _ => true,
    };
    let blocked_until_utc = (!can_fund_full_cycle).then(|| {
        let midnight = now
            .date_naive()
            .succ_opt()
            .expect("the next UTC date is representable")
            .and_hms_opt(0, 0, 0)
            .expect("UTC midnight is representable");
        chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(midnight, chrono::Utc)
            .to_rfc3339()
    });
    let used_percent = goal
        .budget_daily
        .filter(|daily| *daily > 0)
        .map(|daily| ((used_today as f64 / daily as f64) * 100.0).clamp(0.0, 100.0));
    let effort = ReviewEffort::parse(review_effort);
    json!({
        "review_effort": effort.map_or("legacy_custom", ReviewEffort::as_str),
        "automatically_managed": effort.is_some(),
        "runaway_protection": true,
        "usage_percent_today": used_percent,
        "usage_utc_day": today,
        "can_fund_full_cycle": can_fund_full_cycle,
        "blocked_until_utc": blocked_until_utc,
    })
}

fn controller_budget_admission_label(goal: &crate::traits::Goal, review_effort: &str) -> String {
    let snapshot = controller_budget_snapshot(goal, review_effort);
    if snapshot["can_fund_full_cycle"] == Value::Bool(true) {
        "ready".to_string()
    } else {
        format!(
            "budget-blocked-until {}",
            snapshot["blocked_until_utc"]
                .as_str()
                .unwrap_or("next UTC reset")
        )
    }
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

/// Named times in the owner's zone and format ("2:39 PM ET"), leaving out
/// the ones that are unset or unreadable.
fn owner_times<const N: usize>(times: [(&str, Option<&str>); N]) -> Value {
    Value::Object(
        times
            .into_iter()
            .filter_map(|(name, at)| {
                let shown = crate::schedule::owner_time_rfc3339(at?)?;
                Some((name.to_string(), Value::String(shown)))
            })
            .collect(),
    )
}

fn friendly_review_effort(value: &str) -> &str {
    match value {
        "efficient" => "Light — faster checks for straightforward decisions",
        "balanced" => "Balanced — standard checks before acting",
        "thorough" => "Thorough — extra care before acting",
        _ => "Custom decision checks",
    }
}

fn operation_target_label(scope: &MandateOperationScope) -> String {
    let url = scope
        .target_prefixes
        .iter()
        .find_map(|target| reqwest::Url::parse(target).ok());
    let Some(url) = url else {
        return match scope.kind {
            crate::traits::MandateOperationKind::Observation => "approved information".to_string(),
            crate::traits::MandateOperationKind::Mutation => "the approved destination".to_string(),
        };
    };

    let host = url.host_str().unwrap_or("the approved service");
    let path = url.path().trim_end_matches('/');
    if matches!(host, "api.x.com" | "api.twitter.com") {
        if path == "/2/tweets" {
            return match scope.kind {
                crate::traits::MandateOperationKind::Observation => {
                    "read public posts on X".to_string()
                }
                crate::traits::MandateOperationKind::Mutation => {
                    "publish public posts and replies on X".to_string()
                }
            };
        }
        if path == "/2/tweets/search/recent" {
            return "search recent public posts on X".to_string();
        }
        if path.ends_with("/mentions") {
            return "read mentions of the connected X account".to_string();
        }
        if path.starts_with("/2/users/") && path.ends_with("/tweets") {
            return "read posts from the connected X account".to_string();
        }
    }

    let destination = if path.is_empty() {
        host.to_string()
    } else {
        format!("{host}{path}")
    };
    match (scope.kind, scope.operation) {
        (crate::traits::MandateOperationKind::Observation, _) => {
            format!("read information from {destination}")
        }
        (_, ToolCallOperation::Post) => format!("create content at {destination}"),
        (_, ToolCallOperation::Put | ToolCallOperation::Patch) => {
            format!("update content at {destination}")
        }
        (_, ToolCallOperation::Delete) => format!("delete content at {destination}"),
        _ => format!("use {destination}"),
    }
}

/// What a policy lets a mandate read and change, in plain words.
struct AccessLabels {
    reads: BTreeSet<String>,
    changes: BTreeSet<String>,
    account_ids: BTreeSet<String>,
    connection_names: BTreeSet<String>,
}

fn access_labels(authority: &MandateAuthority) -> AccessLabels {
    let mut reads = BTreeSet::new();
    let mut changes = BTreeSet::new();
    let mut account_ids = BTreeSet::new();
    let mut connection_names = BTreeSet::new();

    for scope in &authority.operation_scopes {
        let label = operation_target_label(scope);
        match scope.kind {
            crate::traits::MandateOperationKind::Observation => {
                reads.insert(label);
            }
            crate::traits::MandateOperationKind::Mutation => {
                changes.insert(label);
            }
        }
        for target in &scope.target_prefixes {
            if let Some(account) = target.strip_prefix("account:") {
                account_ids.insert(account.to_string());
            } else if let Some(profile) = target.strip_prefix("auth_profile:") {
                connection_names.insert(match profile {
                    "twitter" | "x" => "X".to_string(),
                    other => other.replace(['_', '-'], " "),
                });
            }
        }
    }

    // Older persisted mandates may predate tuple-scoped policy. Keep their
    // confirmation truthful without surfacing adapter names or raw target data.
    if authority.operation_scopes.is_empty() {
        if authority.allow_observations {
            reads.insert("approved sources only".to_string());
        }
        if authority.max_mutating_actions_per_cycle > 0
            && !authority.allowed_mutation_effects.is_empty()
        {
            changes.insert("approved destinations only".to_string());
        }
        for target in &authority.allowed_target_prefixes {
            if let Some(account) = target.strip_prefix("account:") {
                account_ids.insert(account.to_string());
            } else if let Some(profile) = target.strip_prefix("auth_profile:") {
                connection_names.insert(match profile {
                    "twitter" | "x" => "X".to_string(),
                    other => other.replace(['_', '-'], " "),
                });
            }
        }
    }

    AccessLabels {
        reads,
        changes,
        account_ids,
        connection_names,
    }
}

/// Whether a policy may change anything outside the daemon.
fn can_change(authority: &MandateAuthority, labels: &AccessLabels) -> bool {
    authority.max_mutating_actions_per_cycle > 0 && !labels.changes.is_empty()
}

fn display_access_summary(authority: &MandateAuthority) -> String {
    let labels = access_labels(authority);
    let may_change = can_change(authority, &labels);
    let AccessLabels {
        reads,
        changes,
        account_ids,
        connection_names,
    } = labels;
    let mut lines = Vec::new();
    if reads.is_empty() {
        lines.push("Read: No autonomous read access".to_string());
    } else {
        lines.push(format!(
            "Read: {}",
            reads.into_iter().collect::<Vec<_>>().join("; ")
        ));
    }
    if !may_change {
        lines.push("Change: No external changes".to_string());
    } else {
        lines.push(format!(
            "Change: {}",
            changes.into_iter().collect::<Vec<_>>().join("; ")
        ));
    }
    if !account_ids.is_empty() {
        let service = if connection_names.is_empty() {
            "saved connection".to_string()
        } else {
            format!(
                "saved {} connection",
                connection_names.into_iter().collect::<Vec<_>>().join("/")
            )
        };
        lines.push(format!(
            "Account: Only account {} through the {service}",
            account_ids.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if authority.owner_alerts {
        lines.push("Tell you: what it finds that you asked to hear about".to_string());
    }
    lines.push("Everything else remains blocked".to_string());
    lines.join("\n")
}

fn display_policy(values: &[String]) -> String {
    if values.is_empty() {
        "none specified".to_string()
    } else {
        values
            .iter()
            .map(|value| format!("• {}", value.replace("; ", ";\n  ↳ ")))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmation_access_summary_translates_scopes_without_leaking_adapter_json() {
        let authority = MandateAuthority {
            operation_scopes: serde_json::from_value(json!([
            {
                "tool": "http_request",
                "operation": "GET",
                "kind": "observation",
                "target_prefixes": [
                    "https://api.x.com/2/users/12345/mentions",
                    "auth_profile:twitter",
                    "account:12345"
                ],
                "allowed_query_params": ["max_results"],
                "mutation_effects": []
            },
            {
                "tool": "http_request",
                "operation": "GET",
                "kind": "observation",
                "target_prefixes": [
                    "https://api.x.com/2/tweets/search/recent",
                    "auth_profile:twitter",
                    "account:12345"
                ],
                "allowed_query_params": ["query"],
                "mutation_effects": []
            },
            {
                "tool": "http_request",
                "operation": "POST",
                "kind": "mutation",
                "target_prefixes": [
                    "https://api.x.com/2/tweets",
                    "auth_profile:twitter",
                    "account:12345"
                ],
                "allowed_query_params": [],
                "mutation_effects": ["remote_mutation", "external_delivery"]
            }
            ]))
            .unwrap(),
            max_mutating_actions_per_cycle: 1,
            max_mutating_actions_per_rolling_24h: 8,
            ..MandateAuthority::default()
        };

        let summary = display_access_summary(&authority);
        // With no change allowed per review, the same scopes change nothing.
        let read_only = display_access_summary(&MandateAuthority {
            max_mutating_actions_per_cycle: 0,
            ..authority.clone()
        });
        assert!(
            read_only.contains("Change: No external changes"),
            "{read_only}"
        );
        assert!(summary.contains("read mentions of the connected X account"));
        assert!(summary.contains("search recent public posts on X"));
        assert!(summary.contains("publish public posts and replies on X"));
        assert!(summary.contains("Only account 12345 through the saved X connection"));
        for internal_term in [
            "http_request",
            "api.x.com",
            "auth_profile",
            "allowed_query_params",
            "mutation_effects",
        ] {
            assert!(!summary.contains(internal_term), "leaked {internal_term}");
        }
    }

    /// A read-only hourly mandate with zero change limits reads as five
    /// plain essentials, with no "0 changes", "no delay", "none specified",
    /// version or decision-check line among them.
    #[test]
    fn confirmation_card_is_five_essentials_then_details() {
        let mut mandate = Mandate::new(
            "goal-synthetic",
            None,
            "Every hour, check the synthetic status page",
            "owner-session",
            MandateAuthority {
                owner_alerts: true,
                ..MandateAuthority::default()
            },
            3_600,
            3_600,
            3_600,
        );
        mandate.success_criteria = vec!["You hear about outages within the hour".to_string()];
        let capacity = ResolvedReviewCapacity {
            effort: Some(ReviewEffort::Efficient),
            per_cycle: 1,
            daily: 1,
        };
        let lines = ManageMandatesTool::confirmation_card(&mandate, capacity, None, 2).into_lines();
        assert_eq!(
            &lines[..5],
            [
                "What: Every hour, check the synthetic status page",
                "How often: Every hour",
                "Can do: Read only – makes no changes",
                "Tells you when: it finds something you asked to hear about, it needs your input, or it hits a problem",
                "Ends: Only when you stop it",
            ]
        );
        let card = MandateConfirmationCard::from_lines(&lines).expect("all five essentials");
        assert_eq!(card.how_often, "Every hour");
        let rendered = lines.join("\n");
        for nonsense in [
            "0 change",
            "no delay",
            "none specified",
            "Stop conditions:",
            "Mutation limits:",
            "Review interval:",
            "Expiration:",
        ] {
            assert!(!rendered.contains(nonsense), "{nonsense}: {rendered}");
        }
        // Version and decision checks stay available, as details only.
        assert!(card
            .details
            .iter()
            .any(|detail| detail.starts_with("Policy reference:") && detail.contains("version 2")));
        assert!(card
            .details
            .iter()
            .any(|detail| detail.starts_with("Review depth:")));

        // An end date reads in the owner's zone and format, never RFC 3339.
        let expires_at = (chrono::Utc::now() + chrono::Duration::days(3)).to_rfc3339();
        mandate.expires_at = Some(expires_at.clone());
        let card = ManageMandatesTool::confirmation_card(&mandate, capacity, None, 2);
        assert_eq!(
            card.ends,
            crate::schedule::owner_time_rfc3339(&expires_at).unwrap()
        );
        assert_ne!(card.ends, expires_at);
    }

    #[test]
    fn confirmation_policy_is_scannable_plain_text() {
        let rendered = display_policy(&[
            "Use only the connected account; never impersonate a person; never buy engagement"
                .to_string(),
            "Do not disclose private information".to_string(),
        ]);
        assert!(rendered.starts_with("• Use only the connected account"));
        assert!(rendered.contains("\n  ↳ never impersonate a person"));
        assert!(rendered.contains("\n• Do not disclose private information"));
        assert!(!rendered.contains(" | "));
    }

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
    fn non_stop_outcomes_ignore_generator_supplied_stop_metadata() {
        let (kind, matched) = normalized_termination_fields(
            MandateDecisionOutcome::Wait,
            Some("success_criteria_satisfied"),
            Some("provider-filled default"),
        )
        .unwrap();
        assert!(kind.is_none());
        assert!(matched.is_none());

        assert!(normalized_termination_fields(MandateDecisionOutcome::Stop, None, None).is_err());
    }

    #[test]
    fn act_requires_sourced_evidence_while_non_actions_do_not_manufacture_it() {
        let observations = vec!["The monitored value crossed its confirmed threshold".to_string()];
        let receipts = vec!["tool-call-1".to_string()];
        assert!(validate_act_evidence(MandateDecisionOutcome::Act, &[], &receipts).is_err());
        assert!(validate_act_evidence(MandateDecisionOutcome::Act, &observations, &[]).is_err());
        assert!(
            validate_act_evidence(MandateDecisionOutcome::Act, &observations, &receipts).is_ok()
        );
        assert!(validate_act_evidence(MandateDecisionOutcome::Wait, &[], &[]).is_ok());
    }

    #[test]
    fn empty_learning_note_ignores_generator_supplied_receipt_defaults_before_commit() {
        let receipts = vec!["provider-filled-default".to_string()];
        assert!(normalized_learning_note(Some(""), Some(&receipts))
            .unwrap()
            .is_none());
        assert!(normalized_learning_note(Some("real learning"), None).is_err());
    }

    #[test]
    fn partial_optional_learning_metadata_is_not_decision_blocking() {
        assert!(!learning_note_fields_complete(
            Some("possible learning"),
            None
        ));
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

    #[test]
    fn update_ignores_provider_filled_create_only_fields() {
        let mut args: ManageMandatesArgs = serde_json::from_str(
            r#"{
                "action":"update",
                "mandate_id":"mandate-1",
                "source_goal_id":"",
                "priority":"critical",
                "duration_minutes":1,
                "expires_at":"2027-02-15T13:31:00Z"
            }"#,
        )
        .unwrap();
        args.normalize_for_action();
        assert_eq!(args.source_goal_id, None);
        assert_eq!(args.priority, None);
        assert_eq!(args.duration_minutes, None);
        assert_eq!(args.expires_at.as_deref(), Some("2027-02-15T13:31:00Z"));
    }
}
