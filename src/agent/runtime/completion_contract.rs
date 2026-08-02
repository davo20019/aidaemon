//! Completion-contract inference: deriving what a turn must observe/mutate/verify.
//!
//! Moved verbatim from `runtime/history.rs` (Phase 4 decoupling); logic is
//! unchanged. Owns the completion types ([`CompletionContract`],
//! [`CompletionProgress`], [`CompletionTaskKind`], [`VerificationTarget`],
//! [`VerificationTargetKind`], `CompletionSignals`) and the `infer_*` /
//! `extract_verification_*` helpers.

use super::followup::{looks_like_retry_followup, text_contains_any_phrase};
use super::project_scope::{
    normalize_project_scope_path_with_aliases, push_project_scope, token_looks_like_filesystem_path,
};
use once_cell::sync::Lazy;
use regex::Regex;

use crate::traits::{ToolCallSemantics, ToolMutationEffects};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum CompletionTaskKind {
    #[default]
    Conversational,
    Answer,
    Check,
    Find,
    Change,
    Deliver,
    Schedule,
    Monitor,
    Diagnose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerificationTargetKind {
    Url,
    Path,
    ProjectScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerificationTarget {
    pub kind: VerificationTargetKind,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ForbiddenMutationAction {
    Create,
    Delete,
    Deploy,
    Publish,
    Post,
    Send,
}

impl ForbiddenMutationAction {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Delete => "delete",
            Self::Deploy => "deploy",
            Self::Publish => "publish",
            Self::Post => "post",
            Self::Send => "send",
        }
    }
}

/// Parse a task-assessor operation constraint. Unknown values are ignored so a
/// malformed model response cannot invent a new policy action.
pub(super) fn parse_planned_forbidden_action(value: &str) -> Option<ForbiddenMutationAction> {
    match value.trim().to_ascii_lowercase().as_str() {
        "create" => Some(ForbiddenMutationAction::Create),
        "delete" => Some(ForbiddenMutationAction::Delete),
        "deploy" => Some(ForbiddenMutationAction::Deploy),
        "publish" => Some(ForbiddenMutationAction::Publish),
        "post" => Some(ForbiddenMutationAction::Post),
        "send" => Some(ForbiddenMutationAction::Send),
        _ => None,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CompletionContract {
    pub task_kind: CompletionTaskKind,
    pub expects_mutation: bool,
    /// Typed outcomes that must be observed before a mutating request is
    /// considered fulfilled. An empty set retains legacy count-based behavior.
    pub required_mutation_effects: ToolMutationEffects,
    /// The user explicitly constrained this turn to observation/reporting.
    /// Unlike `expects_mutation = false`, this is a hard negative obligation:
    /// mutation attempts are contract violations and must be blocked.
    pub forbids_mutation: bool,
    /// Operation-specific negative obligations. These do not turn an otherwise
    /// mutating task into a report-only task; they block only the named action.
    pub forbidden_mutation_actions: Vec<ForbiddenMutationAction>,
    pub requires_observation: bool,
    pub requires_reverification_after_mutation: bool,
    pub explicit_verification_requested: bool,
    pub connected_content_mode: super::intent_routing::ConnectedContentMode,
    pub verification_targets: Vec<VerificationTarget>,
}

impl CompletionContract {
    pub(super) fn primary_target_hint(&self) -> Option<String> {
        self.verification_targets
            .first()
            .map(|target| target.value.clone())
    }
}

/// Authoritative, finalized per-turn assessment of whether completing the
/// request requires execution.
///
/// Contract, route, and deterministic lexical signals are folded into one
/// value. Downstream routing and recovery code must consume
/// [`Self::requires_execution`] instead of independently recalculating whether
/// tools are needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ExecutionRequirement {
    requires_execution: bool,
    deterministic_intent: bool,
    recognized_artifact_request: bool,
    mutation_contract: bool,
    observation_contract: bool,
    complex_route: bool,
    supplied_observation: bool,
}

impl ExecutionRequirement {
    pub(super) fn from_finalized_contract(
        contract: &CompletionContract,
        deterministic_tool_need: Option<bool>,
        complex_route: bool,
        recognized_artifact_request: bool,
    ) -> Self {
        let mutation_contract =
            contract.expects_mutation && !contract.connected_content_mode.is_authoring_only();
        let observation_contract = contract.requires_observation;
        let supplied_observation = deterministic_tool_need == Some(false);
        let deterministic_intent = deterministic_tool_need == Some(true);
        let requires_execution = !supplied_observation
            && (deterministic_intent
                || recognized_artifact_request
                || mutation_contract
                || observation_contract
                || complex_route);

        Self {
            requires_execution,
            deterministic_intent,
            recognized_artifact_request,
            mutation_contract,
            observation_contract,
            complex_route,
            // `Some(false)` is reserved for synthetic turns that already carry
            // the requested observation (for example background-command output).
            supplied_observation,
        }
    }

    pub(super) fn requires_execution(&self) -> bool {
        self.requires_execution
    }

    /// Recognized artifact creation is the narrow case where the first model
    /// turn must be an execution call rather than an explanatory text response.
    pub(super) fn requires_initial_tool_call(&self) -> bool {
        self.requires_execution && self.recognized_artifact_request
    }

    pub(super) fn reason_codes(&self) -> Vec<&'static str> {
        if self.supplied_observation {
            return vec!["supplied_observation"];
        }

        let mut reasons = Vec::new();
        if self.deterministic_intent {
            reasons.push("deterministic_intent");
        }
        if self.recognized_artifact_request {
            reasons.push("recognized_artifact_request");
        }
        if self.mutation_contract {
            reasons.push("mutation_contract");
        }
        if self.observation_contract {
            reasons.push("observation_contract");
        }
        if self.complex_route {
            reasons.push("complex_route");
        }
        if reasons.is_empty() {
            reasons.push("none");
        }
        reasons
    }
}

/// Change and Deliver contracts may only enter text-only mode after their
/// mutation obligation has actually been fulfilled.
pub(super) fn mutation_contract_fulfilled(
    contract: &CompletionContract,
    progress: &CompletionProgress,
) -> bool {
    if !contract.expects_mutation {
        return true;
    }
    progress.mutation_count > 0
        && (contract.required_mutation_effects.is_empty()
            || progress
                .observed_mutation_effects
                .satisfies(contract.required_mutation_effects))
}

/// A delivery task has produced a local artifact but has not actually sent it.
///
/// This is deliberately narrower than a generic unfulfilled delivery: a request
/// to locate and send an existing file may truthfully end with "not found".
/// Once the agent has authored a local artifact, however, an honest converter
/// failure is usually recoverable by trying a different local conversion path.
pub(super) fn authored_artifact_still_needs_delivery_recovery(
    contract: &CompletionContract,
    progress: &CompletionProgress,
) -> bool {
    let local_artifact_effects =
        ToolMutationEffects::LOCAL_SOURCE_WRITE.union(ToolMutationEffects::LOCAL_WORKSPACE_WRITE);

    contract.task_kind == CompletionTaskKind::Deliver
        && contract.expects_mutation
        && contract
            .required_mutation_effects
            .intersects(ToolMutationEffects::EXTERNAL_DELIVERY)
        && progress
            .observed_mutation_effects
            .intersects(local_artifact_effects)
        && !progress
            .observed_mutation_effects
            .intersects(ToolMutationEffects::EXTERNAL_DELIVERY)
}

pub(super) fn completion_contract_allows_force_text(
    contract: &CompletionContract,
    progress: &CompletionProgress,
) -> bool {
    !matches!(
        contract.task_kind,
        CompletionTaskKind::Change | CompletionTaskKind::Deliver
    ) || mutation_contract_fulfilled(contract, progress)
}

/// Map a planner-supplied task-kind string to the enum. Unknown values map
/// to None so a hallucinated kind never overrides the keyword inference.
pub(super) fn parse_planned_task_kind(value: &str) -> Option<CompletionTaskKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "conversational" => Some(CompletionTaskKind::Conversational),
        "answer" => Some(CompletionTaskKind::Answer),
        "check" => Some(CompletionTaskKind::Check),
        "find" => Some(CompletionTaskKind::Find),
        "change" => Some(CompletionTaskKind::Change),
        "deliver" => Some(CompletionTaskKind::Deliver),
        "schedule" => Some(CompletionTaskKind::Schedule),
        "monitor" => Some(CompletionTaskKind::Monitor),
        "diagnose" => Some(CompletionTaskKind::Diagnose),
        _ => None,
    }
}

/// Parse semantic outcome requirements from the task assessment. Unknown
/// values reject the whole refinement so a malformed list cannot silently
/// weaken or distort the completion contract.
pub(super) fn parse_planned_mutation_effects(values: &[String]) -> Option<ToolMutationEffects> {
    if values.is_empty() {
        return None;
    }
    let mut effects = ToolMutationEffects::NONE;
    for value in values {
        let effect = match value.trim().to_ascii_lowercase().as_str() {
            "local_source_write" => ToolMutationEffects::LOCAL_SOURCE_WRITE,
            "repository_write" => ToolMutationEffects::REPOSITORY_WRITE,
            "remote_mutation" => ToolMutationEffects::REMOTE_MUTATION,
            "remote_deploy" => ToolMutationEffects::REMOTE_DEPLOY,
            "external_delivery" => ToolMutationEffects::EXTERNAL_DELIVERY,
            "process_state" => ToolMutationEffects::PROCESS_STATE,
            "configuration" => ToolMutationEffects::CONFIGURATION,
            "destructive" => ToolMutationEffects::DESTRUCTIVE,
            _ => return None,
        };
        effects = effects.union(effect);
    }
    Some(effects)
}

/// Add semantically classified positive proof obligations. This never grants
/// permission and never erases deterministic requirements; it only prevents a
/// mismatched mutation from being treated as completion.
pub(super) fn apply_planned_required_mutation_effects(
    contract: &mut CompletionContract,
    effects: Option<ToolMutationEffects>,
) {
    if contract.expects_mutation && !contract.forbids_mutation {
        if let Some(effects) = effects.filter(|effects| !effects.is_empty()) {
            contract.required_mutation_effects = contract.required_mutation_effects.union(effects);
        }
    }
}

/// Refine a keyword-inferred contract with the planning LLM's classification.
/// The planner read the actual request (any language), so its signals win —
/// with one exception: an explicit user verification request ("verify it",
/// "make sure") is never relaxed by a planner saying observation isn't needed.
pub(super) fn apply_planned_contract_signals(
    contract: &mut CompletionContract,
    expects_mutation: Option<bool>,
    requires_observation: Option<bool>,
    task_kind: Option<CompletionTaskKind>,
) {
    let scoped_mutation_obligation =
        contract.expects_mutation && !contract.forbidden_mutation_actions.is_empty();
    if let Some(kind) = task_kind {
        contract.task_kind = kind;
    }
    if let Some(mutation) = expects_mutation {
        if contract.forbids_mutation {
            contract.expects_mutation = false;
            contract.required_mutation_effects = ToolMutationEffects::NONE;
            contract.requires_reverification_after_mutation = false;
        } else if !mutation && scoped_mutation_obligation {
            // A planner can mistake "build locally, but do not deploy" for a
            // report-only request. Preserve the positive work obligation while
            // the operation-specific restriction remains independently enforced.
        } else {
            contract.expects_mutation = mutation;
            if mutation && contract.required_mutation_effects.is_empty() {
                contract.required_mutation_effects = ToolMutationEffects::UNSPECIFIED;
            }
        }
        if !contract.expects_mutation {
            contract.required_mutation_effects = ToolMutationEffects::NONE;
            // No mutation expected → nothing to re-verify after one.
            contract.requires_reverification_after_mutation = false;
        }
    }
    if let Some(observation) = requires_observation {
        if observation {
            contract.requires_observation = true;
        } else if !contract.explicit_verification_requested {
            contract.requires_observation = false;
        }
    }
}

/// Apply semantic negative mutation constraints from the task assessment.
///
/// This is additive by design: a model classification may discover a
/// non-English constraint, but it may never erase a deterministic constraint
/// already found in the request. `read_only` is global; `scoped` blocks only
/// the supplied operations and leaves other required mutations available.
pub(super) fn apply_planned_mutation_constraints(
    contract: &mut CompletionContract,
    mutation_scope: Option<&str>,
    forbidden_actions: &[ForbiddenMutationAction],
) {
    let scope = mutation_scope
        .map(|value| value.trim().to_ascii_lowercase())
        .unwrap_or_default();

    match scope.as_str() {
        "read_only" | "read-only" => {
            contract.forbids_mutation = true;
            contract.expects_mutation = false;
            contract.required_mutation_effects = ToolMutationEffects::NONE;
            contract.requires_reverification_after_mutation = false;
        }
        "scoped" => {
            for action in forbidden_actions {
                if !contract.forbidden_mutation_actions.contains(action) {
                    contract.forbidden_mutation_actions.push(*action);
                }
            }
        }
        // `allowed`, absent, and unknown values cannot relax a deterministic
        // negative obligation.
        _ => {}
    }
}

fn explicitly_forbids_mutation(lower: &str) -> bool {
    let trimmed = lower.trim();
    let explicit_mode = ["read-only", "inspect-only", "report-only"]
        .iter()
        .any(|mode| {
            trimmed == *mode
                || trimmed.starts_with(&format!("{mode}:"))
                || trimmed.starts_with(&format!("{mode} request"))
                || trimmed.starts_with(&format!("this is a {mode}"))
        });
    if explicit_mode {
        return true;
    }

    // A global prohibition must carry a global object (anything, files, or
    // any changes). Bare phrases such as "read only the README" and qualified
    // constraints such as "do not make changes outside src" are not blanket
    // write bans.
    [
        "do not make any changes",
        "don't make any changes",
        "without making any changes",
        "do not modify anything",
        "don't modify anything",
        "do not change anything",
        "don't change anything",
        "do not edit anything",
        "don't edit anything",
        "do not write anything",
        "don't write anything",
        "do not modify files",
        "don't modify files",
    ]
    .iter()
    .any(|phrase| {
        let Some(index) = lower.find(phrase) else {
            return false;
        };
        let tail = lower[index + phrase.len()..]
            .trim_start_matches(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';' | ':'));
        ![
            "outside",
            "except",
            "other than",
            "beyond",
            "under",
            "inside",
        ]
        .iter()
        .any(|qualifier| tail.starts_with(qualifier))
    })
}

const SCOPED_NEGATIVE_MUTATION_PHRASES: &[(ForbiddenMutationAction, &[&str])] = &[
    (
        ForbiddenMutationAction::Create,
        &["do not create", "don't create", "without creating"],
    ),
    (
        ForbiddenMutationAction::Delete,
        &["do not delete", "don't delete", "without deleting"],
    ),
    (
        ForbiddenMutationAction::Deploy,
        &["do not deploy", "don't deploy", "without deploying"],
    ),
    (
        ForbiddenMutationAction::Publish,
        &["do not publish", "don't publish", "without publishing"],
    ),
    (
        ForbiddenMutationAction::Post,
        &["do not post", "don't post", "without posting"],
    ),
    (
        ForbiddenMutationAction::Send,
        &["do not send", "don't send", "without sending"],
    ),
];

fn scoped_forbidden_mutation_actions(lower: &str) -> Vec<ForbiddenMutationAction> {
    SCOPED_NEGATIVE_MUTATION_PHRASES
        .iter()
        .filter_map(|(action, phrases)| {
            phrases
                .iter()
                .any(|phrase| negative_phrase_is_operation_wide(lower, phrase))
                .then_some(*action)
        })
        .collect()
}

/// Return true only when a negative action phrase is unambiguously a ban on
/// the operation itself. Content and precondition guards such as "don't post
/// filler" or "do not post if identity verification fails" must remain task
/// instructions; promoting them to a blanket hard gate makes the requested
/// operation impossible.
///
/// This intentionally prefers false negatives for ambiguous noun phrases. The
/// completion contract is an execution backstop, not a natural-language policy
/// engine, so it should hard-block only high-confidence operation-wide bans.
fn negative_phrase_is_operation_wide(lower: &str, phrase: &str) -> bool {
    lower.match_indices(phrase).any(|(index, _)| {
        let before = lower[..index].chars().next_back();
        let after_index = index + phrase.len();
        let after = lower[after_index..].chars().next();
        if before.is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
            || after.is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
        {
            return false;
        }

        if negative_phrase_has_conditional_context(lower, index, phrase) {
            return false;
        }

        let tail = lower[after_index..].trim_start();
        if tail.is_empty()
            || tail.starts_with(|ch: char| {
                matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | '\n' | '\r' | '—')
            })
        {
            return true;
        }

        // These complements still prohibit the operation as a whole. Other
        // noun/conditional complements are deliberately left to the executor
        // because they commonly constrain content, retries, or prerequisites.
        [
            "anything",
            "anything else",
            "it",
            "this",
            "that",
            "them",
            "again",
            "yet",
            "now",
            "at all",
            "anywhere",
            "publicly",
            "externally",
            "automatically",
            "directly",
            "live",
            "online",
            "to ",
            "on ",
            "via ",
            "into ",
        ]
        .iter()
        .any(|complement| {
            tail == *complement
                || (tail.starts_with(complement)
                    && complement
                        .chars()
                        .next_back()
                        .is_some_and(char::is_whitespace))
                || tail
                    .strip_prefix(complement)
                    .is_some_and(|rest| rest.starts_with(|ch: char| !ch.is_alphanumeric()))
        })
    })
}

fn negative_phrase_has_conditional_context(lower: &str, index: usize, phrase: &str) -> bool {
    let before = &lower[..index];
    let sentence_start = before
        .rfind(['.', '!', '?', '\n'])
        .map_or(0, |position| position + 1);
    let sentence_prefix = before[sentence_start..].trim_start();
    if ["if ", "unless ", "when ", "whenever ", "only if "]
        .iter()
        .any(|marker| sentence_prefix.starts_with(marker))
    {
        return true;
    }

    // Scheduled publishing goals commonly express their skip gate as two
    // sentences: establish the condition, then say to finish without posting.
    // Keep that conditional outcome out of the operation-wide hard gate.
    if !phrase.starts_with("without ") {
        return false;
    }
    let paragraph_start = before.rfind("\n\n").map_or(0, |position| position + 2);
    let paragraph_prefix = before[paragraph_start..].trim_start();
    let immediate_prefix = sentence_prefix
        .rsplit_once(';')
        .map_or(sentence_prefix, |(_, tail)| tail)
        .trim();
    ["finish", "skip", "stop"]
        .iter()
        .any(|verb| immediate_prefix.ends_with(verb))
        && (paragraph_prefix.contains("skip gate")
            || paragraph_prefix.contains("skip if")
            || paragraph_prefix.contains("if nothing"))
}

fn remove_scoped_negative_mutation_phrases(lower: &str) -> String {
    SCOPED_NEGATIVE_MUTATION_PHRASES
        .iter()
        .flat_map(|(_, phrases)| phrases.iter())
        .fold(lower.to_string(), |text, phrase| text.replace(phrase, " "))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CompletionProgress {
    pub observation_count: usize,
    pub mutation_count: usize,
    /// Effects observed from successful calls. This is separate from the raw
    /// count so incidental cache writes cannot fulfill source/deploy outcomes.
    pub observed_mutation_effects: ToolMutationEffects,
    pub verification_count: usize,
    pub verification_pending: bool,
    /// Count of external mutation attempts that failed (4xx/5xx, etc.)
    pub failed_external_mutation_count: usize,
    /// Count of external mutation attempts that succeeded
    pub successful_external_mutation_count: usize,
    /// True after we have already given the LLM one reconciliation-informed retry.
    pub external_mutation_reconciliation_attempted: bool,
    /// True after the one-shot RECOVERY pass: before asking the model to
    /// report failed external mutations honestly, it first gets one
    /// evidence-fed chance to fix them with a DIFFERENT approach.
    pub external_mutation_recovery_attempted: bool,
    /// True after the one-shot recovery for an authored local artifact that
    /// has not yet been delivered to the user.
    pub undelivered_artifact_recovery_attempted: bool,
    /// Cumulative count of times the verification guard blocked completion.
    /// Unlike `stall_count` (which is shared with other stall mechanisms and
    /// gets reset), this counter only increments and is used as a safety valve
    /// to prevent infinite verification loops.
    pub verification_block_count: usize,
    /// Count of times the response-quality nudge has been injected.
    /// Used to prevent infinite nudge loops — only fire once.
    pub quality_nudge_count: usize,
    /// Count of times the locate-file retry nudge has been injected (model
    /// asked the user to upload a file it could find itself). Fires once.
    pub file_access_retry_count: usize,
    /// Count of times the answer-grounding nudge has been injected (final
    /// reply enumerated entities absent from all tool outputs). Fires once.
    pub grounding_nudge_count: usize,
    /// Count of times the single-source corroboration nudge has been
    /// injected (enumeration answer from web research with <2 source pages
    /// read). Fires once.
    pub corroboration_nudge_count: usize,
    /// Count of times the search-before-deny gate has fired (reply denies/
    /// asserts a personal fact about an entity that was not looked up in
    /// memory this turn). Bounded to 1 to prevent infinite loops.
    pub denial_gate_count: usize,
    /// True when the coreference grounding gate already fired this turn
    /// (a pronoun-referent follow-up that was anchored to the prior exchange).
    /// When set, the denial gate must NOT also fire — coreference gate takes
    /// precedence and the denial would be a false positive.
    pub coreference_fired: bool,
}

impl CompletionProgress {
    pub(super) fn new(contract: &CompletionContract) -> Self {
        Self {
            verification_pending: contract.requires_observation,
            ..Self::default()
        }
    }

    pub(super) fn mark_mutation(
        &mut self,
        contract: &CompletionContract,
        semantics: &ToolCallSemantics,
    ) {
        self.mutation_count = self.mutation_count.saturating_add(1);
        let effects = if semantics.mutation_effects.is_empty() {
            ToolMutationEffects::UNSPECIFIED
        } else {
            semantics.mutation_effects
        };
        self.observed_mutation_effects = self.observed_mutation_effects.union(effects);
        if contract.requires_reverification_after_mutation {
            self.verification_pending = true;
        }
    }

    pub(super) fn mark_observation(&mut self, contract: &CompletionContract, matched_target: bool) {
        self.observation_count = self.observation_count.saturating_add(1);
        if !contract.requires_observation {
            return;
        }
        if matched_target || contract.verification_targets.is_empty() {
            self.verification_pending = false;
            self.verification_count = self.verification_count.saturating_add(1);
        }
    }

    pub(super) fn mark_failed_external_mutation(&mut self) {
        self.failed_external_mutation_count += 1;
    }

    pub(super) fn mark_successful_external_mutation(&mut self) {
        self.successful_external_mutation_count += 1;
    }

    pub(super) fn mark_external_mutation_reconciliation_attempted(&mut self) {
        self.external_mutation_reconciliation_attempted = true;
    }

    pub(super) fn mark_external_mutation_recovery_attempted(&mut self) {
        self.external_mutation_recovery_attempted = true;
    }

    pub(super) fn mark_undelivered_artifact_recovery_attempted(&mut self) {
        self.undelivered_artifact_recovery_attempted = true;
    }

    pub(super) fn clear_failed_external_mutation_gate(&mut self) {
        self.failed_external_mutation_count = 0;
        self.external_mutation_reconciliation_attempted = false;
        self.external_mutation_recovery_attempted = false;
    }
}

pub(super) static HTTP_URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)\bhttps?://[^\s"'()<>]+"#).expect("valid http url regex"));

fn should_allow_verification_nickname_scope(text: &str, token: &str) -> bool {
    super::should_allow_contextual_project_nickname_scope(text, token)
}

fn extract_verification_project_scopes(
    text: &str,
    scopes: &mut Vec<String>,
    max_scopes: usize,
    alias_roots: &[String],
) {
    for raw in text.split_whitespace() {
        if scopes.len() >= max_scopes {
            break;
        }
        let token = raw
            .trim_matches(|c: char| {
                c.is_ascii_whitespace()
                    || matches!(
                        c,
                        '`' | '\''
                            | '"'
                            | ','
                            | ';'
                            | ':'
                            | '.'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                    )
            })
            .trim();
        if token.is_empty() || token.contains("://") {
            continue;
        }

        let scope = if token_looks_like_filesystem_path(token) {
            normalize_project_scope_path_with_aliases(token, alias_roots)
        } else {
            should_allow_verification_nickname_scope(text, token)
                .then(|| {
                    crate::tools::fs_utils::resolve_named_project_root(token, alias_roots)
                        .or_else(|| {
                            crate::tools::fs_utils::resolve_contextual_project_nickname_in_explicit_roots(token, alias_roots)
                        })
                })
                .flatten()
                .map(|path| path.to_string_lossy().to_string())
        };

        if let Some(scope) = scope {
            push_project_scope(scopes, scope, max_scopes);
        }
    }
}

fn extract_verification_targets(text: &str, alias_roots: &[String]) -> Vec<VerificationTarget> {
    let mut targets = Vec::new();

    for capture in HTTP_URL_RE.captures_iter(text) {
        let raw = capture
            .get(0)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim_end_matches(['.', ',', ';', ')', ']', '}'])
            .to_string();
        if raw.is_empty()
            || targets.iter().any(|existing: &VerificationTarget| {
                existing.kind == VerificationTargetKind::Url && existing.value == raw
            })
        {
            continue;
        }
        targets.push(VerificationTarget {
            kind: VerificationTargetKind::Url,
            value: raw,
        });
    }

    let mut scopes = Vec::new();
    extract_verification_project_scopes(text, &mut scopes, 4, alias_roots);
    for scope in scopes {
        if targets.iter().any(|existing| existing.value == scope) {
            continue;
        }
        targets.push(VerificationTarget {
            kind: VerificationTargetKind::ProjectScope,
            value: scope,
        });
    }

    if targets.is_empty() && super::user_text_references_filesystem_path(text) {
        for raw in text.split_whitespace() {
            let token = raw
                .trim_matches(|c: char| {
                    c.is_ascii_whitespace()
                        || matches!(
                            c,
                            '`' | '\''
                                | '"'
                                | ','
                                | ';'
                                | ':'
                                | '.'
                                | '!'
                                | '?'
                                | '('
                                | ')'
                                | '['
                                | ']'
                                | '{'
                                | '}'
                        )
                })
                .trim();
            if token.is_empty() || token.contains("://") || !token_looks_like_filesystem_path(token)
            {
                continue;
            }
            let resolved = if crate::execution::active_execution_backend().kind()
                == crate::execution::BackendKind::Local
            {
                crate::tools::fs_utils::validate_path(token)
                    .map(|path| path.to_string_lossy().to_string())
            } else {
                crate::execution::normalize_active_path_lexically(token)
                    .map(|path| path.as_str().to_string())
            };
            if let Ok(value) = resolved {
                if !targets.iter().any(|existing| existing.value == value) {
                    targets.push(VerificationTarget {
                        kind: VerificationTargetKind::Path,
                        value,
                    });
                }
            }
        }
    }

    targets
}

#[derive(Debug, Clone, Default)]
struct CompletionSignals {
    is_question: bool,
    asks_schedule: bool,
    asks_monitor: bool,
    asks_check: bool,
    asks_find: bool,
    asks_deliver: bool,
    asks_change: bool,
    asks_diagnose: bool,
    mutation_obligation: bool,
    has_verification_target: bool,
    claimed_side_effect: bool,
    explicit_verification_requested: bool,
    observable_target_request: bool,
    visible_state_problem: bool,
    read_only_request: bool,
    /// A question whose answer requires observing live external/system state
    /// (e.g. "how many users?", "who are the admins?", "any blocked users?").
    /// These have no file/URL target and no "show me / on this site" phrasing,
    /// so they are NOT `observable_target_request`, yet they still need a tool
    /// to observe. Without this signal the contract treats them as plain-text
    /// knowledge questions and the prelude hard-blocks the lookup tool.
    live_state_query: bool,
}

pub(super) fn looks_like_question_request(lower_text: &str) -> bool {
    lower_text.ends_with('?')
        || [
            "what ", "when ", "where ", "why ", "who ", "how ", "is ", "are ", "do ", "does ",
            "did ", "can ", "could ", "will ", "would ",
        ]
        .iter()
        .any(|prefix| lower_text.starts_with(prefix))
}

/// Mutation-flavored request verbs, split so run-and-report detection can ask
/// "does this text ask for any change BEYOND merely running something?".
const CHANGE_KEYWORDS: &[&str] = &[
    "change",
    "update",
    "edit",
    "write",
    "rewrite",
    "overwrite",
    "modify",
    "replace",
    "create",
    "delete",
    "remove",
    "deploy",
    "build",
    "connect",
    "set up",
    "setup",
    "install",
    "restart",
    "reload",
    "enable",
    "disable",
    "remember",
    "store",
    "save",
    "note",
    "pull",
    "push",
    "fetch",
    "merge",
    "start",
    "stop",
    "compile",
    "download",
    "clone",
    "migrate",
    "fix",
    "retry",
    "redo",
    "try again",
    "try one more time",
    "one more time",
    "do it again",
];

/// Verbs that only mean "execute something" — mutating ONLY if the executed
/// thing mutates, which the other keywords capture.
const RUN_KEYWORDS: &[&str] = &["run", "execute", "rerun", "re-run"];

/// Keywords that are strong enough to arm a mutation contract when they appear
/// as direct requests. Broader content verbs such as "write" stay out of this
/// list unless there is an artifact cue; "write a tweet" is reply generation,
/// while "write a test file" is a filesystem mutation.
const STRONG_MUTATION_KEYWORDS: &[&str] = &[
    "change",
    "update",
    "edit",
    "rewrite",
    "overwrite",
    "modify",
    "replace",
    "create",
    "delete",
    "remove",
    "deploy",
    "connect",
    "set up",
    "setup",
    "install",
    "restart",
    "reload",
    "enable",
    "disable",
    "remember",
    "store",
    "save",
    "pull",
    "push",
    "fetch",
    "merge",
    "start",
    "stop",
    "compile",
    "download",
    "clone",
    "migrate",
    "fix",
    "retry",
    "redo",
    "try again",
    "try one more time",
    "one more time",
    "do it again",
];

const WRITE_ARTIFACT_CUES: &[&str] = &[
    "file", "script", "test", "tests", "readme", "code", "module", "function", "class", "document",
    "doc", "page", "record", "database", "db",
];

/// "Run X and tell me what it said" — the deliverable is the OBSERVATION.
/// True when the text asks to report/return/provide output AND the only
/// execution verbs are the bare run/execute family (no genuine change verb).
fn is_run_and_report_only(lower_text: &str) -> bool {
    let report_intent = text_contains_any_phrase(
        lower_text,
        &[
            "provide the output",
            "return the output",
            "report the output",
            "show the output",
            "print the output",
            "provide the result",
            "return the result",
            "report the result",
            "provide the count",
            "return the count",
            "report the count",
        ],
    );
    report_intent
        && text_contains_any_phrase(lower_text, RUN_KEYWORDS)
        && !text_contains_any_phrase(lower_text, CHANGE_KEYWORDS)
}

fn infer_completion_signals(
    lower_text: &str,
    verification_targets: &[VerificationTarget],
) -> CompletionSignals {
    let has_verification_target = !verification_targets.is_empty();
    let is_question = looks_like_question_request(lower_text);
    let asks_schedule = text_contains_any_phrase(
        lower_text,
        &[
            "remind me",
            "schedule",
            "set a reminder",
            "add reminder",
            "scheduled task",
            "scheduled goal",
            "recurring task",
            "recurring goal",
        ],
    ) || crate::cron_utils::extract_schedule_from_text(lower_text).is_some();
    let asks_monitor =
        text_contains_any_phrase(lower_text, &["monitor", "watch", "keep an eye on"]);
    let asks_check = text_contains_any_phrase(
        lower_text,
        &[
            "check",
            "verify",
            "confirm",
            "see if",
            "test whether",
            "test if",
            "is there",
            "do i have",
            "did it",
            "did you",
            "status",
        ],
    );
    let asks_find = text_contains_any_phrase(
        lower_text,
        &["find", "locate", "list", "show me", "search for", "look up"],
    );
    let asks_deliver = text_contains_any_phrase(
        lower_text,
        &[
            "send",
            "post this",
            "post it",
            "post a",
            "post an",
            "post to",
            "post on",
            "upload",
            "tweet this",
            "tweet it",
            "email",
            "message",
            "share",
        ],
    );
    let asks_change = text_contains_any_phrase(lower_text, CHANGE_KEYWORDS)
        || text_contains_any_phrase(lower_text, RUN_KEYWORDS);
    let visible_state_problem = text_contains_any_phrase(
        lower_text,
        &[
            "still dont see",
            "still don't see",
            "not showing",
            "doesnt show",
            "doesn't show",
            "isnt showing",
            "isn't showing",
            "not visible",
            "missing from",
            "missing on",
            "broken on",
            "not working",
            "failed to load",
            "in production",
            "on the site",
            "on the page",
            "go live",
        ],
    );
    // "why is/isn't" alone is a weak diagnose signal: it matches innocent
    // knowledge questions ("why is it called america?" scored partial and
    // paid a verification-block LLM call, task 9ae13321). The interrogative
    // only counts when the text corroborates an observable problem — a
    // visible-state complaint, a verification target, or a strong diagnose
    // verb. The planner refinement layer can still re-arm observation for
    // genuine diagnostics this heuristic misses.
    let strong_diagnose_signal = text_contains_any_phrase(
        lower_text,
        &[
            "fix",
            "fixing",
            "debug",
            "diagnose",
            "troubleshoot",
            "issue",
            "problem",
            "error",
            "fails to",
            "failing to",
        ],
    );
    let weak_why_interrogative =
        text_contains_any_phrase(lower_text, &["why is", "why isnt", "why isn't"]);
    let asks_diagnose = visible_state_problem
        || strong_diagnose_signal
        || (weak_why_interrogative && has_verification_target);
    let claimed_side_effect = text_contains_any_phrase(
        lower_text,
        &[
            "did it",
            "did you",
            "did that work",
            "did this work",
            "went through",
            "was it sent",
            "was it posted",
            "was it deployed",
        ],
    );
    let explicit_verification_requested = text_contains_any_phrase(
        lower_text,
        &[
            "verify",
            "confirm",
            "make sure",
            "double check",
            "double-check",
            "validate",
            "look it up",
            "look this up",
        ],
    );
    let observable_target_request = has_verification_target
        && text_contains_any_phrase(
            lower_text,
            &[
                "here",
                "there",
                "read",
                "open",
                "summarize",
                "show me",
                "what's on",
                "what is on",
                "what does",
                "what do you see",
                "in this file",
                "on this page",
                "on this site",
                "at this url",
                "at this link",
            ],
        );
    let read_only_request = observable_target_request
        || text_contains_any_phrase(
            lower_text,
            &[
                "read",
                "open",
                "summarize",
                "show me",
                "tell me",
                "what's on",
                "what is on",
                "what does",
                "what do you see",
                "find",
                "locate",
                "list",
                "search for",
                "look up",
                "check",
                "verify",
                "confirm",
                "see if",
                "status",
            ],
        );

    // If a read-only turn or question only hits memory-flavored change words
    // ("remember", "store", "save", "note"), demote asks_change. "Find the
    // note" and "what did I ask you to remember?" are recalls, not mutations.
    let asks_change = if asks_change && (is_question || read_only_request) {
        text_contains_any_phrase(
            lower_text,
            &[
                "change",
                "update",
                "edit",
                "write",
                "rewrite",
                "overwrite",
                "modify",
                "replace",
                "create",
                "delete",
                "remove",
                "deploy",
                "build",
                "connect",
                "set up",
                "setup",
                "install",
                "restart",
                "reload",
                "enable",
                "disable",
                "pull",
                "push",
                "run",
                "execute",
                "fetch",
                "merge",
                "start",
                "stop",
                "compile",
                "download",
                "clone",
                "migrate",
                "fix",
                "retry",
                "redo",
                "rerun",
                "try again",
                "try one more time",
                "one more time",
                "do it again",
            ],
        )
    } else {
        asks_change
    };

    // Interrogatives that enumerate or quantify live system/entity state. These
    // need a tool to observe the answer even though they carry no file/URL
    // target. Scoped to questions so plain conversational text is unaffected.
    let live_state_query = is_question
        && (lower_text.starts_with("any ")
            || text_contains_any_phrase(
                lower_text,
                &[
                    "how many",
                    "how much",
                    "who are",
                    "who is",
                    "who's",
                    "list the",
                    "list all",
                    "what are the",
                    "which ",
                    "are there any",
                    "is there any",
                    "do i have any",
                    "do we have any",
                ],
            ));

    let instructional_question = is_question
        && text_contains_any_phrase(
            lower_text,
            &[
                "how do i",
                "how can i",
                "how should i",
                "what is the best way",
                "what's the best way",
                "why should i",
            ],
        )
        && !text_contains_any_phrase(lower_text, &["can you", "could you", "will you"]);
    let write_artifact_request = text_contains_any_phrase(lower_text, &["write"])
        && (has_verification_target || text_contains_any_phrase(lower_text, WRITE_ARTIFACT_CUES));
    let execution_request =
        text_contains_any_phrase(lower_text, RUN_KEYWORDS) && !is_run_and_report_only(lower_text);
    let strong_mutation_request = text_contains_any_phrase(lower_text, STRONG_MUTATION_KEYWORDS)
        || write_artifact_request
        || execution_request;
    let mutation_obligation = !instructional_question
        && (asks_schedule
            || asks_monitor
            || asks_deliver
            || (strong_mutation_request && asks_change));

    CompletionSignals {
        is_question,
        asks_schedule,
        asks_monitor,
        asks_check,
        asks_find,
        asks_deliver,
        asks_change,
        asks_diagnose,
        mutation_obligation,
        has_verification_target,
        claimed_side_effect,
        explicit_verification_requested,
        observable_target_request,
        visible_state_problem,
        read_only_request,
        live_state_query,
    }
}

fn infer_completion_task_kind(signals: &CompletionSignals) -> CompletionTaskKind {
    if signals.asks_schedule {
        return CompletionTaskKind::Schedule;
    }
    if signals.asks_monitor {
        return CompletionTaskKind::Monitor;
    }
    if signals.asks_diagnose {
        return CompletionTaskKind::Diagnose;
    }
    if signals.asks_deliver {
        return CompletionTaskKind::Deliver;
    }
    if signals.asks_change {
        return CompletionTaskKind::Change;
    }
    if signals.asks_check {
        return CompletionTaskKind::Check;
    }
    if signals.asks_find {
        return CompletionTaskKind::Find;
    }
    if signals.observable_target_request {
        return CompletionTaskKind::Answer;
    }
    if signals.is_question {
        return CompletionTaskKind::Answer;
    }

    CompletionTaskKind::Conversational
}

/// Extract the current-turn request from text that may have been enriched with
/// the prior request. `turn_context` enriches follow-ups as
/// "Original request:\n…\n\nCurrent request:\n…", then `sanitize_carryover_blocks`
/// strips the "Original request:" / "Assistant asked:" / "Follow-up:" labels
/// (but NOT "Current request:") before this contract is inferred. So the
/// surviving "Current request:" marker — not "Original request:", which is gone
/// by the time we run — is what reliably delimits the current turn.
///
/// The completion contract must reflect what THIS turn asks: an observational
/// follow-up ("what's in that file?") after a prior mutation ("create a file …")
/// must not inherit the prior request's task-kind and `expects_mutation=true`,
/// which would block completion for extra iterations and score the turn failed.
/// Returns the full text unchanged when no marker is present (the common,
/// non-enriched case → byte-identical behavior).
fn current_request_segment(text: &str) -> &str {
    let segment_start = ["Current request:", "Follow-up:"]
        .iter()
        .filter_map(|marker| text.rfind(marker).map(|idx| idx + marker.len()))
        .max();
    if let Some(start) = segment_start {
        let segment = text[start..].trim();
        if !segment.is_empty() {
            return segment;
        }
    }
    text
}

/// Return the request that preceded an enriched `Current request:` segment.
/// `sanitize_carryover_blocks` may already have removed the `Original request:`
/// label, so the surviving current marker is the reliable split point.
fn prior_request_segment(text: &str) -> Option<&str> {
    let marker_start = ["Current request:", "Follow-up:"]
        .iter()
        .filter_map(|marker| text.rfind(marker))
        .max()?;
    let prior = text[..marker_start].trim();
    (!prior.is_empty()).then_some(prior)
}

/// Narrow a freshly inferred contract to what a delegated executor child can
/// actually satisfy on its own.
///
/// An executor runs one narrowly-scoped step of a decomposed plan; the
/// orchestrating parent owns goal-level verification and routinely delegates
/// it to a sibling task. Reverification inferred from mutation-flavored
/// phrases ("deploy", "publish") in the mission would demand a verification
/// the child was never asked to perform — and stamp a fully successful step
/// `partial`. Reverification survives only when the mission itself asks for
/// it.
pub(super) fn scope_contract_for_delegated_executor(
    mut contract: CompletionContract,
) -> CompletionContract {
    if !contract.explicit_verification_requested {
        contract.requires_reverification_after_mutation = false;
    }
    contract
}

fn infer_required_mutation_effects(
    lower_text: &str,
    expects_mutation: bool,
    task_kind: CompletionTaskKind,
) -> ToolMutationEffects {
    if !expects_mutation {
        return ToolMutationEffects::NONE;
    }

    let mut required = ToolMutationEffects::NONE;
    // Delivery is a semantic outcome, not a synonym list. Once the request is
    // classified as Deliver, a local write must never satisfy it. Live repro:
    // "Send me a pdf about imerkar" wrote HTML and PostScript, failed both
    // conversions, and was incorrectly persisted as succeeded without calling
    // send_file because the old phrase list only recognized "send me the file".
    if task_kind == CompletionTaskKind::Deliver {
        required = required.union(ToolMutationEffects::EXTERNAL_DELIVERY);
    }
    if text_contains_any_phrase(
        lower_text,
        &[
            "deploy",
            "go live",
            "release to production",
            "publish the site",
            "push to a worker",
            "push to worker",
        ],
    ) {
        required = required.union(ToolMutationEffects::REMOTE_DEPLOY);
    }

    let local_artifact = text_contains_any_phrase(
        lower_text,
        &[
            "file",
            "folder",
            "directory",
            "script",
            "test",
            "tests",
            "readme",
            "code",
            "module",
            "function",
            "class",
            "document",
            "page",
            "website",
            "web site",
            "site",
            "app",
            "application",
            "project",
            "worker",
            "dashboard",
            "api",
        ],
    );
    let authors_artifact = text_contains_any_phrase(
        lower_text,
        &[
            "change",
            "update",
            "edit",
            "write",
            "rewrite",
            "overwrite",
            "modify",
            "replace",
            "remove",
            "create",
            "implement",
            "develop",
            "fix",
            "migrate",
        ],
    );
    if local_artifact && authors_artifact {
        required = required.union(ToolMutationEffects::LOCAL_SOURCE_WRITE);
    }

    let deletes_local_path = text_contains_any_phrase(lower_text, &["delete", "remove", "erase"])
        && text_contains_any_phrase(
            lower_text,
            &["file", "folder", "directory", "local project", "repository"],
        );
    if deletes_local_path {
        required = required
            .union(ToolMutationEffects::LOCAL_SOURCE_WRITE)
            .union(ToolMutationEffects::DESTRUCTIVE);
    }

    if text_contains_any_phrase(
        lower_text,
        &[
            "restart",
            "reload",
            "start the service",
            "stop the service",
            "kill the process",
        ],
    ) {
        required = required.union(ToolMutationEffects::PROCESS_STATE);
    }

    if text_contains_any_phrase(
        lower_text,
        &[
            "install the package",
            "install dependencies",
            "change the configuration",
            "update the configuration",
            "change permissions",
        ],
    ) {
        required = required.union(ToolMutationEffects::CONFIGURATION);
    }

    if text_contains_any_phrase(
        lower_text,
        &[
            "send the file",
            "send me the file",
            "deliver the file",
            "upload the file",
            "email the file",
            "post this",
            "post it",
            "send the message",
        ],
    ) {
        required = required.union(ToolMutationEffects::EXTERNAL_DELIVERY);
    }

    if required.is_empty() {
        ToolMutationEffects::UNSPECIFIED
    } else {
        required
    }
}

pub(super) fn infer_completion_contract(text: &str, alias_roots: &[String]) -> CompletionContract {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return CompletionContract::default();
    }

    // Derive the task-kind and mutation expectation from the CURRENT request
    // when the text was enriched with a prior request (see
    // `current_request_segment`). Verification targets are still taken from the
    // full text so prior URLs/paths the follow-up refers back to remain
    // observable.
    let current = current_request_segment(trimmed);
    // A retry-only turn inherits the substantive request it asks us to repeat.
    // Other follow-ups continue to use only the current segment, preserving the
    // observational-followup guard below.
    let contract_request = if looks_like_retry_followup(current) {
        prior_request_segment(trimmed).unwrap_or(current)
    } else {
        current
    };
    let lower = contract_request.to_ascii_lowercase();
    let forbids_mutation = explicitly_forbids_mutation(&lower);
    let forbidden_mutation_actions = scoped_forbidden_mutation_actions(&lower);
    let mutation_signal_text = remove_scoped_negative_mutation_phrases(&lower);

    let verification_targets = extract_verification_targets(text, alias_roots);
    let signals = infer_completion_signals(&mutation_signal_text, &verification_targets);
    let connected_content_mode =
        super::intent_routing::classify_connected_content_mode(contract_request);
    let mut task_kind = infer_completion_task_kind(&signals);
    // "Run X and provide the output" is an observation delivered as text —
    // the bare run verb must not make it a Change (observed live: successful
    // read-and-report executor tasks scored partial via expects_mutation).
    if task_kind == CompletionTaskKind::Change && is_run_and_report_only(&lower) {
        task_kind = CompletionTaskKind::Check;
    }
    // DraftOnly used to override task_kind to Compose and force
    // expects_mutation=false, but this caused false tool disablement
    // for requests like "create blog posts in ~/projects/X and commit".
    // Keyword-based content-mode classification is too brittle to make
    // hard execution decisions.  DraftOnly is now advisory only — it
    // still influences system prompt hints and budget routing but does
    // not override the signal-derived task kind or mutation expectation.

    // The live-delivery content mode must not override an explicitly
    // read-only request: "check if the post is live" is a Check whose
    // fulfillment is the observation itself, not a delivery.
    let live_delivery_override = connected_content_mode.expects_live_delivery()
        && !signals.read_only_request
        && !matches!(
            task_kind,
            CompletionTaskKind::Check | CompletionTaskKind::Find
        );
    let expects_mutation = if forbids_mutation {
        false
    } else if live_delivery_override {
        true
    } else {
        signals.mutation_obligation
    };
    let required_mutation_effects =
        infer_required_mutation_effects(&mutation_signal_text, expects_mutation, task_kind);
    let requires_observation = signals.explicit_verification_requested
        || signals.observable_target_request
        || signals.visible_state_problem
        || signals.live_state_query
        || task_kind == CompletionTaskKind::Diagnose
        || task_kind == CompletionTaskKind::Find
        || (matches!(task_kind, CompletionTaskKind::Check)
            && (signals.has_verification_target || signals.claimed_side_effect));
    let requires_reverification_after_mutation = matches!(
        task_kind,
        CompletionTaskKind::Diagnose | CompletionTaskKind::Monitor
    ) || (expects_mutation
        && (signals.explicit_verification_requested
            || text_contains_any_phrase(
                &mutation_signal_text,
                &["deploy", "publish", "release", "go live"],
            )
            || signals.visible_state_problem));

    CompletionContract {
        task_kind,
        expects_mutation,
        required_mutation_effects,
        forbids_mutation,
        forbidden_mutation_actions,
        requires_observation,
        requires_reverification_after_mutation,
        explicit_verification_requested: signals.explicit_verification_requested,
        connected_content_mode,
        verification_targets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_and_report_missions_are_checks_not_changes() {
        // Live repro (executors 3f4c6a8d / bb711de5, 2026-07-02): "Run 'df -h /'
        // and provide the output." classified Change via the "run" keyword →
        // expects_mutation=true → a perfectly successful read-and-report
        // scored partial. Running a command to REPORT its output is an
        // observation, not a mutation.
        for mission in [
            "Run 'df -h /' and provide the output.",
            "Run 'find ~/projects/aidaemon/src -name \"*.rs\" | wc -l' and return the count.",
            "Execute 'uptime' and report the output.",
        ] {
            let contract = infer_completion_contract(mission, &[]);
            assert!(
                !contract.expects_mutation,
                "{mission:?} must not expect mutation, got kind={:?}",
                contract.task_kind
            );
        }

        // Control: a run-and-report phrasing around a genuinely mutating verb
        // keeps its mutation expectation. (Backtick-quoted like real task-lead
        // missions — NOTE: single-quoted commands hide keywords from
        // contains_keyword_as_words because apostrophes count as word chars;
        // pre-existing matcher quirk, documented here.)
        let deploy = infer_completion_contract(
            "Run `npm run deploy` in ~/projects/example-blog and report the output.",
            &[],
        );
        assert!(deploy.expects_mutation);
    }

    #[test]
    fn verify_only_check_mission_does_not_expect_mutation_despite_live_content_mode() {
        // Live repro (task 6680789b): a sub-agent's verification mission
        // mentioning "the post is live" classified as DeliverOnly content
        // mode, whose expects_live_delivery() override forced
        // expects_mutation=true onto a Check task. The child made its
        // observation perfectly and still scored partial.
        let contract = infer_completion_contract(
            "Use `curl -I https://blog.example.com/posts/synthetic-post/` to check if the post is live. Expect an HTTP 200 status code.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Check);
        assert!(
            !contract.expects_mutation,
            "a read-only check must not expect a mutation"
        );
        assert!(contract.requires_observation);
    }

    #[test]
    fn genuine_live_delivery_request_still_expects_mutation() {
        // Control: the live-delivery override must keep applying to requests
        // that actually ask for content to be delivered.
        let contract = infer_completion_contract(
            "Write a short announcement and post it to the blog so it goes live.",
            &[],
        );
        assert!(contract.expects_mutation);
    }

    #[test]
    fn executor_scope_clears_implicit_reverification_only() {
        // Live repro (task 9437a263): an executor child whose whole mission
        // was "run `npm run deploy`" carried
        // requires_reverification_after_mutation=true from the "deploy"
        // phrase, though the parent decomposed verification into a sibling
        // task. The child deployed successfully and still scored partial.
        let deploy = infer_completion_contract(
            "1. Navigate to `~/projects/example-blog`.\n2. Run `npm run deploy`.\n3. Report the output of the command.",
            &[],
        );
        assert!(deploy.requires_reverification_after_mutation);
        let scoped = scope_contract_for_delegated_executor(deploy);
        assert!(
            !scoped.requires_reverification_after_mutation,
            "implicit reverification must not survive executor scoping"
        );

        // Explicit verification requested in the mission itself must survive.
        let explicit = infer_completion_contract(
            "Run `npm run deploy` in ~/projects/example-blog, then verify the site returns HTTP 200.",
            &[],
        );
        assert!(explicit.explicit_verification_requested);
        let scoped = scope_contract_for_delegated_executor(explicit);
        assert!(scoped.requires_reverification_after_mutation);
    }

    #[test]
    fn visible_issue_contract_requires_observation_and_reverification() {
        let contract = infer_completion_contract(
            "I still don't see the posts here: https://blog.aidaemon.ai",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Diagnose);
        assert!(contract.requires_observation);
        assert!(contract.requires_reverification_after_mutation);
        assert_eq!(contract.verification_targets.len(), 1);
        assert_eq!(
            contract.verification_targets[0],
            VerificationTarget {
                kind: VerificationTargetKind::Url,
                value: "https://blog.aidaemon.ai".to_string(),
            }
        );
    }
    #[test]
    fn create_record_contract_does_not_force_verification() {
        let contract = infer_completion_contract("Create the remote record.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }
    #[test]
    fn knowledge_why_questions_are_answers_not_diagnoses() {
        // Live repro (task 9ae13321, 2026-07-14): "why is it called america?"
        // matched the "why is" diagnose keyword → task_kind=Diagnose → the
        // contract armed BOTH requires_observation and expects_mutation for a
        // plain knowledge question. The correct zero-tool answer then paid a
        // verification-block retry (a full extra LLM call) and scored partial.
        // A bare "why is/isn't" is a weak signal: it only means Diagnose when
        // the text corroborates an observable problem.
        for question in [
            "why is it called america?",
            "why is the sky blue?",
            "why isn't Pluto a planet?",
        ] {
            let contract = infer_completion_contract(question, &[]);
            assert_eq!(
                contract.task_kind,
                CompletionTaskKind::Answer,
                "{question:?} is a knowledge question, not a diagnosis"
            );
            assert!(!contract.requires_observation, "{question:?}");
            assert!(!contract.expects_mutation, "{question:?}");
        }

        // Controls: "why is/isn't" plus corroboration must stay Diagnose.
        let with_target = infer_completion_contract("why is https://blog.aidaemon.ai down?", &[]);
        assert_eq!(with_target.task_kind, CompletionTaskKind::Diagnose);
        assert!(with_target.requires_observation);

        let with_strong_keyword =
            infer_completion_contract("why is the daemon throwing an error?", &[]);
        assert_eq!(with_strong_keyword.task_kind, CompletionTaskKind::Diagnose);
        assert!(with_strong_keyword.requires_observation);

        let with_visible_problem = infer_completion_contract("why is the chart not showing?", &[]);
        assert_eq!(with_visible_problem.task_kind, CompletionTaskKind::Diagnose);
        assert!(with_visible_problem.requires_observation);
    }
    #[test]
    fn still_phrase_alone_does_not_force_diagnose() {
        let contract = infer_completion_contract("I still want you to deploy the app.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(contract.requires_reverification_after_mutation);
    }
    #[test]
    fn target_reference_in_change_request_does_not_force_verification() {
        let contract = infer_completion_contract(
            "Update /tmp/aidaemon/config.toml to point at the new endpoint.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
        assert_eq!(contract.verification_targets.len(), 1);
    }
    #[test]
    fn reading_target_requires_observation_without_change() {
        let contract = infer_completion_contract(
            "Read https://blog.aidaemon.ai and summarize the latest post.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Answer);
        assert!(
            !contract.expects_mutation,
            "read-only URL summarization must not inherit a delivery contract from the word 'post'"
        );
        assert!(contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }
    #[test]
    fn deploy_and_verify_stays_change_with_reverification() {
        let contract = infer_completion_contract(
            "Deploy the app and verify it is live at https://blog.aidaemon.ai",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
        assert!(contract.requires_observation);
        assert!(contract.requires_reverification_after_mutation);
    }
    #[test]
    fn schedule_request_tracks_mutation_without_forcing_verification() {
        let contract = infer_completion_contract("Remind me tomorrow at 9am to call Alice.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Schedule);
        assert!(contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }
    #[test]
    fn live_state_questions_require_observation_so_lookup_tools_are_not_blocked() {
        // Regression: factual questions about live system state were classified
        // as plain-text knowledge questions (requires_observation=false), so the
        // prelude hard-blocked the shell/lookup tool and the model fabricated an
        // answer or returned a bare "Done.".
        for q in [
            "How many users?",
            "Who are admin users?",
            "Any blocked/inactive users?",
            "Which modules are enabled?",
            "How much disk is free?",
        ] {
            let contract = infer_completion_contract(q, &[]);
            assert!(
                contract.requires_observation,
                "expected requires_observation for live-state question: {q:?}"
            );
            assert!(
                !contract.expects_mutation,
                "live-state question should not expect mutation: {q:?}"
            );
        }
        // "How many users?" carries no file/URL target and requests no explicit
        // verification, so requires_observation alone does NOT arm a completion
        // verification block (has_concrete_verification_reason stays false) —
        // confirming requires_observation flips off the plain-text gate without
        // introducing loop risk for a plain answer.
        let bare = infer_completion_contract("How many users?", &[]);
        assert!(!bare.explicit_verification_requested);
        assert!(bare.verification_targets.is_empty());
    }
    #[test]
    fn conversational_questions_stay_plain_text() {
        // A question that does not enumerate live state must not be flagged.
        let contract = infer_completion_contract("What do you think of this idea?", &[]);
        assert!(!contract.requires_observation);
        assert!(!contract.expects_mutation);
    }
    #[test]
    fn observational_followup_does_not_inherit_prior_request_mutation() {
        // Regression: the completion contract is inferred from `goal_user_text`,
        // which prepends the prior request ("Original request:\n…\n\nCurrent
        // request:\n…"). An observational follow-up after a prior mutation
        // request inherited expects_mutation=true and was blocked for extra
        // iterations, then scored failed/partial.
        let enriched = "Original request:\nCreate a python script that pings davidloor.com \
             every 5 seconds and logs the latency.\n\nCurrent request:\nHow stable is the latency?";
        let contract = infer_completion_contract(enriched, &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Answer);
        assert!(
            !contract.expects_mutation,
            "observational follow-up must not inherit the prior request's mutation expectation"
        );
    }
    #[test]
    fn observational_screenshot_followup_is_answer_not_mutation() {
        let enriched = "Original request:\nOpen davidloor.com in the browser and take a \
             screenshot.\n\nCurrent request:\nWhat did you see in the screenshot?";
        let contract = infer_completion_contract(enriched, &[]);
        assert!(
            !contract.expects_mutation,
            "asking what was seen is observational, not a mutation"
        );
    }

    #[test]
    fn retry_only_followup_inherits_pdf_delivery_contract() {
        let enriched = "Original request:\nSend me a pdf about Imerkar. Make it nice since we are pitching the idea to investors.\n\nCurrent request:\nCan you try one more time?";
        let contract = infer_completion_contract(enriched, &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Deliver);
        assert!(contract.expects_mutation);
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::EXTERNAL_DELIVERY));
    }

    #[test]
    fn bare_one_more_time_is_still_a_mutation_retry() {
        let contract = infer_completion_contract("Can you try one more time?", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
    }

    #[test]
    fn sanitized_enriched_followup_uses_current_request_not_prior_mutation() {
        // Regression (live-confirmed): turn_context enriches a follow-up as
        // "Original request:\n…\n\nCurrent request:\n…", but sanitize_carryover_blocks
        // strips the "Original request:" label BEFORE the contract is inferred,
        // leaving the prior mutation text bare at the start with only "Current
        // request:" surviving. The contract must still key off "Current request:"
        // so the observational follow-up is not classified as the prior mutation.
        let sanitized = "Create a file at /tmp/aidaemon_probe2.txt containing the word hello.\
             \n\nCurrent request:\nWhat's in that file?";
        let c = infer_completion_contract(sanitized, &[]);
        assert_eq!(
            c.task_kind,
            CompletionTaskKind::Answer,
            "sanitized follow-up should classify by the current request"
        );
        assert!(
            !c.expects_mutation,
            "observational follow-up must not inherit the prior request's mutation expectation"
        );
    }
    #[test]
    fn enriched_current_request_mutation_still_expects_mutation() {
        // The fix must not suppress a mutation expressed in the CURRENT request.
        let enriched =
            "Original request:\nWhat files are in ~/projects?\n\nCurrent request:\nCreate a \
             README in that folder.";
        let contract = infer_completion_contract(enriched, &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
    }
    #[test]
    fn generic_check_request_does_not_force_verification_without_target() {
        let contract = infer_completion_contract("Check system health.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Check);
        assert!(!contract.expects_mutation);
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }
    #[test]
    fn targeted_check_request_requires_observation_without_mutation() {
        let contract = infer_completion_contract("Check https://status.example.com.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Check);
        assert!(!contract.expects_mutation);
        assert!(contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }
    #[test]
    fn deliver_request_does_not_force_observation_without_explicit_verification() {
        let contract = infer_completion_contract("Email this note to Alice.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Deliver);
        assert!(contract.expects_mutation);
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::EXTERNAL_DELIVERY));
        assert!(!contract.requires_observation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn send_me_pdf_requires_delivery_not_just_a_local_write() {
        let contract = infer_completion_contract(
            "Send me a pdf about imerkar. Make it nice since we are pitching the idea to investors.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Deliver);
        assert!(contract.expects_mutation);
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::EXTERNAL_DELIVERY));

        let mut progress = CompletionProgress::new(&contract);
        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
        );

        assert!(!mutation_contract_fulfilled(&contract, &progress));
        assert!(authored_artifact_still_needs_delivery_recovery(
            &contract, &progress
        ));

        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::mutation_with(ToolMutationEffects::EXTERNAL_DELIVERY),
        );

        assert!(mutation_contract_fulfilled(&contract, &progress));
        assert!(!authored_artifact_still_needs_delivery_recovery(
            &contract, &progress
        ));
    }

    #[test]
    fn missing_existing_file_does_not_arm_authored_artifact_recovery() {
        let contract =
            infer_completion_contract("Send me the SOW PDF from the Lodestar project.", &[]);
        let progress = CompletionProgress::new(&contract);

        assert!(!authored_artifact_still_needs_delivery_recovery(
            &contract, &progress
        ));
    }
    #[test]
    fn rewrite_request_expects_mutation() {
        let contract = infer_completion_contract(
            "Rewrite ~/projects/blog/tweets.md so the thread better promotes the blog content.",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);
    }
    #[test]
    fn connected_content_draft_only_request_preserves_signal_derived_task_kind() {
        let contract = infer_completion_contract("Help me write a tweet about our launch.", &[]);
        assert_eq!(
            contract.connected_content_mode,
            super::super::intent_routing::ConnectedContentMode::DraftOnly
        );
        assert!(
            !contract.expects_mutation,
            "draft-only reply generation must not require a filesystem/external mutation"
        );
    }
    #[test]
    fn connected_content_draft_then_deliver_request_keeps_delivery_contract() {
        let contract = infer_completion_contract(
            "Can you post a tweet about your new stuff and make it engaging so people want to comment?",
            &[],
        );
        assert_eq!(contract.task_kind, CompletionTaskKind::Deliver);
        assert!(contract.expects_mutation);
        assert_eq!(
            contract.connected_content_mode,
            super::super::intent_routing::ConnectedContentMode::DraftThenDeliver
        );
    }

    #[test]
    fn finalized_execution_requirement_keeps_draft_only_text_optional() {
        let contract = infer_completion_contract("Help me write a tweet about our launch.", &[]);
        let requirement =
            ExecutionRequirement::from_finalized_contract(&contract, None, false, false);

        assert!(!requirement.requires_execution());
        assert_eq!(requirement.reason_codes(), vec!["none"]);
    }

    #[test]
    fn finalized_execution_requirement_uses_planner_refined_observation_contract() {
        let mut contract = CompletionContract::default();
        apply_planned_contract_signals(
            &mut contract,
            Some(false),
            Some(true),
            Some(CompletionTaskKind::Check),
        );
        let requirement =
            ExecutionRequirement::from_finalized_contract(&contract, None, false, false);

        assert!(requirement.requires_execution());
        assert!(requirement.reason_codes().contains(&"observation_contract"));
    }

    #[test]
    fn evidence_grounded_observation_requires_execution() {
        let contract = CompletionContract {
            requires_observation: true,
            explicit_verification_requested: true,
            verification_targets: vec![VerificationTarget {
                kind: VerificationTargetKind::Url,
                value: "https://example.com/status".to_string(),
            }],
            ..CompletionContract::default()
        };
        let requirement =
            ExecutionRequirement::from_finalized_contract(&contract, None, false, false);

        assert!(requirement.requires_execution());
    }

    #[test]
    fn supplied_observation_disarms_tool_recovery() {
        let contract = CompletionContract {
            requires_observation: true,
            ..CompletionContract::default()
        };
        let requirement =
            ExecutionRequirement::from_finalized_contract(&contract, Some(false), true, true);

        assert!(!requirement.requires_execution());
        assert!(!requirement.requires_initial_tool_call());
        assert_eq!(requirement.reason_codes(), vec!["supplied_observation"]);
    }

    #[test]
    fn recognized_artifact_requires_initial_execution_call() {
        let contract = CompletionContract {
            task_kind: CompletionTaskKind::Change,
            expects_mutation: true,
            ..CompletionContract::default()
        };
        let requirement =
            ExecutionRequirement::from_finalized_contract(&contract, Some(true), false, true);

        assert!(requirement.requires_execution());
        assert!(requirement.requires_initial_tool_call());
        assert!(requirement
            .reason_codes()
            .contains(&"recognized_artifact_request"));
    }

    #[test]
    fn unfulfilled_change_and_deliver_contracts_disallow_force_text() {
        for task_kind in [CompletionTaskKind::Change, CompletionTaskKind::Deliver] {
            let contract = CompletionContract {
                task_kind,
                expects_mutation: true,
                ..CompletionContract::default()
            };
            let mut progress = CompletionProgress::new(&contract);

            assert!(!completion_contract_allows_force_text(&contract, &progress));
            progress.mutation_count = 1;
            assert!(completion_contract_allows_force_text(&contract, &progress));
        }
    }
    #[test]
    fn recall_question_about_remembered_facts_does_not_expect_mutation() {
        // "What did I ask you to remember?" contains "remember" but is a
        // recall question, not a mutation request. Should NOT expect mutation.
        let contract = infer_completion_contract(
            "What do you know about my coding preferences? What did I ask you to remember?",
            &[],
        );
        assert!(
            !contract.expects_mutation,
            "Recall question about 'remember' should not expect mutation, got task_kind={:?}",
            contract.task_kind
        );
    }
    #[test]
    fn store_request_with_remember_does_expect_mutation() {
        // "Remember that I prefer dark themes" is a store request, not recall.
        let contract =
            infer_completion_contract("Remember that I prefer dark themes and large fonts", &[]);
        assert!(
            contract.expects_mutation,
            "Store request with 'remember' should expect mutation"
        );
    }
    #[test]
    fn generic_find_request_does_not_force_verification_without_target() {
        let contract =
            infer_completion_contract("Find the most relevant note from last week.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Find);
        assert!(
            !contract.expects_mutation,
            "finding a note is a read, not a memory/file write"
        );
        assert!(contract.requires_observation);
    }
    #[test]
    fn completion_progress_resets_after_mutation_and_clears_after_observation() {
        let contract = infer_completion_contract(
            "I still don't see the posts here: https://blog.aidaemon.ai",
            &[],
        );
        let mut progress = CompletionProgress::new(&contract);
        assert!(progress.verification_pending);

        progress.mark_observation(&contract, true);
        assert!(!progress.verification_pending);

        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::mutation_with(ToolMutationEffects::UNSPECIFIED),
        );
        assert!(progress.verification_pending);

        progress.mark_observation(&contract, true);
        assert!(!progress.verification_pending);
        assert_eq!(progress.verification_count, 2);
    }
    #[test]
    fn verification_targets_do_not_resolve_plain_word_nicknames_without_local_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        let targets = extract_verification_targets(
            "Find recruiting studies in Fairfax, Virginia and summarize them.",
            &alias_roots,
        );

        assert!(
            targets
                .iter()
                .all(|target| target.kind != VerificationTargetKind::ProjectScope),
            "plain-word nickname should not resolve without local scope cues: {:?}",
            targets
        );
    }
    #[test]
    fn verification_targets_allow_plain_word_nicknames_with_explicit_project_scope_cues() {
        let root = tempfile::tempdir().expect("tempdir");
        let alias_root = root.path().join("projects-root");
        let project = alias_root.join("fairfax-va-site");
        std::fs::create_dir_all(&project).expect("create project");
        std::fs::write(project.join("wrangler.toml"), "name = \"fairfax\"\n").expect("wrangler");
        let alias_roots = vec![alias_root.to_string_lossy().to_string()];

        assert!(should_allow_verification_nickname_scope(
            "Check the Fairfax project for broken links.",
            "Fairfax"
        ));
        assert_eq!(
            crate::tools::fs_utils::resolve_contextual_project_nickname_in_explicit_roots(
                "Fairfax",
                &alias_roots
            ),
            Some(project.clone())
        );

        let targets = extract_verification_targets(
            "Check the Fairfax project for broken links.",
            &alias_roots,
        );

        assert!(
            targets.iter().any(|target| {
                target.kind == VerificationTargetKind::ProjectScope
                    && target.value == project.to_string_lossy()
            }),
            "explicit local scope cue should still allow nickname resolution: {:?}",
            targets
        );
    }
    #[test]
    fn failed_external_mutation_tracking() {
        let contract = CompletionContract {
            task_kind: CompletionTaskKind::Deliver,
            expects_mutation: true,
            requires_observation: false,
            ..Default::default()
        };
        let mut progress = CompletionProgress::new(&contract);
        assert_eq!(progress.failed_external_mutation_count, 0);
        assert!(!progress.external_mutation_reconciliation_attempted);

        progress.mark_failed_external_mutation();
        assert_eq!(progress.failed_external_mutation_count, 1);
        assert!(!progress.external_mutation_reconciliation_attempted);

        progress.mark_successful_external_mutation();
        assert_eq!(progress.successful_external_mutation_count, 1);

        progress.mark_external_mutation_reconciliation_attempted();
        assert!(progress.external_mutation_reconciliation_attempted);

        progress.clear_failed_external_mutation_gate();
        assert_eq!(progress.failed_external_mutation_count, 0);
        assert!(!progress.external_mutation_reconciliation_attempted);
    }

    // Regression tests from the 2026-06-06 attribution run: the mutation
    // expectation misfired in BOTH directions across builds. Read-only
    // phrasings were blocked with expects_mutation=true on the older build,
    // while the turn-10 delete request must keep expects_mutation=true so
    // the zero-tool fabrication guard can fire.
    #[test]
    fn read_only_file_inspection_does_not_expect_mutation() {
        let contract = infer_completion_contract(
            "Read each of the three files back and tell me which one is longest.",
            &[],
        );
        assert!(
            !contract.expects_mutation,
            "read-only inspection must not expect a mutation (got {:?})",
            contract.task_kind
        );

        let contract = infer_completion_contract(
            "Open all four files and tell me which two are most similar in length.",
            &[],
        );
        assert!(!contract.expects_mutation);
    }

    #[test]
    fn explicit_negative_mutation_constraint_is_hard_and_planner_cannot_override_it() {
        let mut contract = infer_completion_contract(
            "Inspect the repository and report only. Do not modify files or deploy.",
            &[],
        );
        assert!(contract.forbids_mutation);
        assert!(!contract.expects_mutation);

        apply_planned_contract_signals(
            &mut contract,
            Some(true),
            Some(true),
            Some(CompletionTaskKind::Change),
        );
        assert!(contract.forbids_mutation);
        assert!(!contract.expects_mutation);
    }

    #[test]
    fn object_scoped_modify_constraint_is_not_a_global_write_ban() {
        let contract = infer_completion_contract(
            "Update the application, but do not modify the deployment configuration.",
            &[],
        );
        assert!(!contract.forbids_mutation);
        assert!(contract.expects_mutation);

        let global = infer_completion_contract(
            "Inspect the application only and do not make any changes.",
            &[],
        );
        assert!(global.forbids_mutation);
        assert!(!global.expects_mutation);
    }

    #[test]
    fn contextual_read_and_scoped_change_phrases_do_not_become_global_bans() {
        for request in [
            "Read only the README, then fix the failing test.",
            "Do not make any changes outside src; update src/main.rs.",
            "Report only the files changed after implementing the fix.",
        ] {
            let contract = infer_completion_contract(request, &[]);
            assert!(
                !contract.forbids_mutation,
                "contextual phrase must not globally forbid work: {request}"
            );
            assert!(
                contract.expects_mutation,
                "requested edit must remain actionable: {request}"
            );
        }
    }

    #[test]
    fn local_build_with_deployment_reserved_for_parent_remains_mutating() {
        let request = "Inspect ~/projects for an appropriate existing app or create a new \
            project directory there. Implement a polished website. Configure static hosting \
            tooling and run the relevant build/check. Do not deploy externally; report the \
            directory, changes, and verification results.";
        let mut contract = infer_completion_contract(request, &[]);

        assert!(!contract.forbids_mutation);
        assert!(contract.expects_mutation);
        assert_eq!(
            contract.forbidden_mutation_actions,
            vec![ForbiddenMutationAction::Deploy]
        );

        apply_planned_contract_signals(
            &mut contract,
            Some(false),
            Some(true),
            Some(CompletionTaskKind::Deliver),
        );
        assert!(
            contract.expects_mutation,
            "planner must not turn scoped deployment restraint into a blanket write ban"
        );
    }

    #[test]
    fn observation_with_deployment_restriction_does_not_invent_a_mutation() {
        let contract =
            infer_completion_contract("Inspect the current site, but do not deploy.", &[]);
        assert!(!contract.forbids_mutation);
        assert!(!contract.expects_mutation);
        assert_eq!(
            contract.forbidden_mutation_actions,
            vec![ForbiddenMutationAction::Deploy]
        );
    }

    #[test]
    fn tweet_content_and_precondition_guards_do_not_become_a_blanket_post_ban() {
        let request = "Post one short tweet from the authenticated account. \
            Skip if nothing genuinely noteworthy happened; don't post filler. \
            A quiet day is fine; finish without posting. Never post personal data or \
            security-sensitive internals. Resolve the account \
            through GET /2/users/me and do not post if the identity check fails. \
            Compose and POST exactly one tweet, then verify it once.";
        let contract = infer_completion_contract(request, &[]);

        assert!(!contract
            .forbidden_mutation_actions
            .contains(&ForbiddenMutationAction::Post));
    }

    #[test]
    fn unqualified_post_bans_remain_hard_constraints() {
        for request in [
            "Draft the tweet, but do not post.",
            "Prepare the message without posting.",
            "Write the announcement, but don't post it.",
        ] {
            let contract = infer_completion_contract(request, &[]);
            assert!(
                contract
                    .forbidden_mutation_actions
                    .contains(&ForbiddenMutationAction::Post),
                "operation-wide prohibition must remain enforced: {request}"
            );
        }
    }

    #[test]
    fn delete_and_create_requests_expect_mutation() {
        // Turn-10 fabrication case: "delete" must keep the contract armed.
        let contract = infer_completion_contract("Delete the cachetest2 folder entirely.", &[]);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
        assert!(contract.expects_mutation);

        let contract = infer_completion_contract(
            "Make a folder ~/tmp/cachetest2 and create four files in it.",
            &[],
        );
        assert!(contract.expects_mutation);

        let contract =
            infer_completion_contract("Remove west.txt and execute tally.py once more.", &[]);
        assert!(contract.expects_mutation);
    }

    #[test]
    fn typed_outcomes_do_not_let_build_cache_satisfy_source_edit() {
        let contract = infer_completion_contract("Fix the code in this project.", &[]);
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::LOCAL_SOURCE_WRITE));
        let mut progress = CompletionProgress::new(&contract);
        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::observation_and_mutation_with(
                ToolMutationEffects::LOCAL_DERIVED_WRITE,
            ),
        );
        assert!(!mutation_contract_fulfilled(&contract, &progress));
        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
        );
        assert!(mutation_contract_fulfilled(&contract, &progress));
    }

    #[test]
    fn deterministic_effect_fallback_does_not_overfit_ambiguous_build_or_remove() {
        let build = infer_completion_contract(
            "Build the existing project and report any compiler errors.",
            &[],
        );
        assert!(!build
            .required_mutation_effects
            .contains(ToolMutationEffects::LOCAL_SOURCE_WRITE));

        let edit = infer_completion_contract("Remove the unused function from the code.", &[]);
        assert!(edit
            .required_mutation_effects
            .contains(ToolMutationEffects::LOCAL_SOURCE_WRITE));
        assert!(!edit
            .required_mutation_effects
            .contains(ToolMutationEffects::DESTRUCTIVE));
    }

    #[test]
    fn semantic_effects_add_positive_proof_without_granting_permission() {
        let parsed = parse_planned_mutation_effects(&[
            "local_source_write".to_string(),
            "remote_deploy".to_string(),
        ])
        .expect("valid effects");
        let mut contract = CompletionContract {
            expects_mutation: true,
            required_mutation_effects: ToolMutationEffects::UNSPECIFIED,
            ..CompletionContract::default()
        };
        apply_planned_required_mutation_effects(&mut contract, Some(parsed));
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::LOCAL_SOURCE_WRITE));
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::REMOTE_DEPLOY));
        assert!(parse_planned_mutation_effects(&["made_up".to_string()]).is_none());

        contract.forbids_mutation = true;
        contract.expects_mutation = false;
        let before = contract.required_mutation_effects;
        apply_planned_required_mutation_effects(
            &mut contract,
            Some(ToolMutationEffects::REMOTE_MUTATION),
        );
        assert_eq!(contract.required_mutation_effects, before);
    }

    #[test]
    fn build_and_deploy_requires_both_local_source_and_remote_deploy() {
        let contract = infer_completion_contract(
            "Create a website in the project, deploy it, verify it, and return the URL.",
            &[],
        );
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::LOCAL_SOURCE_WRITE));
        assert!(contract
            .required_mutation_effects
            .contains(ToolMutationEffects::REMOTE_DEPLOY));
        let mut progress = CompletionProgress::new(&contract);
        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::mutation_with(ToolMutationEffects::LOCAL_SOURCE_WRITE),
        );
        assert!(!mutation_contract_fulfilled(&contract, &progress));
        progress.mark_mutation(
            &contract,
            &ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_DEPLOY),
        );
        assert!(mutation_contract_fulfilled(&contract, &progress));
    }

    #[test]
    fn planned_task_kind_parses_all_variants_and_rejects_garbage() {
        for (s, expected) in [
            ("conversational", CompletionTaskKind::Conversational),
            ("answer", CompletionTaskKind::Answer),
            ("check", CompletionTaskKind::Check),
            ("find", CompletionTaskKind::Find),
            ("change", CompletionTaskKind::Change),
            ("deliver", CompletionTaskKind::Deliver),
            ("schedule", CompletionTaskKind::Schedule),
            ("monitor", CompletionTaskKind::Monitor),
            ("diagnose", CompletionTaskKind::Diagnose),
        ] {
            assert_eq!(parse_planned_task_kind(s), Some(expected), "kind {s}");
        }
        assert_eq!(
            parse_planned_task_kind(" Change "),
            Some(CompletionTaskKind::Change)
        );
        assert_eq!(parse_planned_task_kind("destroy_everything"), None);
        assert_eq!(parse_planned_task_kind(""), None);
    }

    #[test]
    fn planned_forbidden_actions_parse_known_values_only() {
        for (value, expected) in [
            ("create", ForbiddenMutationAction::Create),
            ("DELETE", ForbiddenMutationAction::Delete),
            (" deploy ", ForbiddenMutationAction::Deploy),
            ("publish", ForbiddenMutationAction::Publish),
            ("post", ForbiddenMutationAction::Post),
            ("send", ForbiddenMutationAction::Send),
        ] {
            assert_eq!(parse_planned_forbidden_action(value), Some(expected));
        }
        assert_eq!(parse_planned_forbidden_action("rewrite-history"), None);
    }

    #[test]
    fn semantic_read_only_constraint_blocks_all_mutation() {
        let mut contract = CompletionContract {
            expects_mutation: true,
            requires_reverification_after_mutation: true,
            ..Default::default()
        };

        apply_planned_mutation_constraints(&mut contract, Some("read_only"), &[]);

        assert!(contract.forbids_mutation);
        assert!(!contract.expects_mutation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn semantic_scoped_constraint_preserves_other_mutation_work() {
        let mut contract = CompletionContract {
            expects_mutation: true,
            ..Default::default()
        };

        apply_planned_mutation_constraints(
            &mut contract,
            Some("scoped"),
            &[ForbiddenMutationAction::Deploy],
        );

        assert!(!contract.forbids_mutation);
        assert!(contract.expects_mutation);
        assert_eq!(
            contract.forbidden_mutation_actions,
            vec![ForbiddenMutationAction::Deploy]
        );

        apply_planned_contract_signals(&mut contract, Some(false), Some(true), None);
        assert!(
            contract.expects_mutation,
            "a scoped restriction must not collapse the remaining work into read-only mode"
        );
    }

    #[test]
    fn semantic_allowed_scope_cannot_erase_deterministic_restrictions() {
        let mut contract = infer_completion_contract("Inspect the site, but do not deploy.", &[]);
        let before = contract.clone();

        apply_planned_mutation_constraints(&mut contract, Some("allowed"), &[]);

        assert_eq!(contract, before);
    }

    #[test]
    fn planned_signals_override_keyword_inference() {
        // "Escribe un script en deploy.sh" — Spanish; keyword inference sees
        // nothing and produces a conversational no-mutation contract.
        let mut contract = CompletionContract::default();
        assert!(!contract.expects_mutation);

        apply_planned_contract_signals(
            &mut contract,
            Some(true),
            Some(true),
            Some(CompletionTaskKind::Change),
        );
        assert!(contract.expects_mutation);
        assert!(contract.requires_observation);
        assert_eq!(contract.task_kind, CompletionTaskKind::Change);
    }

    #[test]
    fn planned_mutation_false_clears_reverification() {
        // Keyword false positive: "write a tweet about rust" infers a file
        // mutation. The planner classifying it as pure text generation must
        // clear both the mutation expectation and the dependent re-verify.
        let mut contract = CompletionContract {
            expects_mutation: true,
            requires_reverification_after_mutation: true,
            ..Default::default()
        };
        apply_planned_contract_signals(&mut contract, Some(false), None, None);
        assert!(!contract.expects_mutation);
        assert!(!contract.requires_reverification_after_mutation);
    }

    #[test]
    fn explicit_verification_request_is_never_relaxed() {
        let mut contract = CompletionContract {
            requires_observation: true,
            explicit_verification_requested: true,
            ..Default::default()
        };
        apply_planned_contract_signals(&mut contract, None, Some(false), None);
        assert!(
            contract.requires_observation,
            "user's explicit 'verify it' must survive planner relaxation"
        );

        // Without the explicit request, the planner may relax it.
        let mut contract = CompletionContract {
            requires_observation: true,
            ..Default::default()
        };
        apply_planned_contract_signals(&mut contract, None, Some(false), None);
        assert!(!contract.requires_observation);
    }

    #[test]
    fn absent_signals_leave_contract_untouched() {
        let mut contract = CompletionContract {
            task_kind: CompletionTaskKind::Find,
            expects_mutation: true,
            requires_observation: true,
            ..Default::default()
        };
        let before = contract.clone();
        apply_planned_contract_signals(&mut contract, None, None, None);
        assert_eq!(contract, before);
    }
}
