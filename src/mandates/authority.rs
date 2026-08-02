//! Pure, deterministic authority evaluator for one tool action proposed under
//! an owner mandate.
//!
//! This module deliberately performs no I/O, persistence, approval prompting,
//! or tool execution. Callers must load the current mandate and ACT decision
//! cycle, call [`authorize_mandate_action`], then atomically reserve a mutation
//! attempt before carrying an allowed grant into the execution boundary.

use chrono::{DateTime, Utc};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::traits::{
    Mandate, MandateAuthorityGrant, MandateDecisionCycle, MandateDecisionOutcome,
    MandateMutationTarget, ToolCallEffect, ToolCallSemantics, ToolMutationEffects,
    ToolTargetHintKind,
};

const ACTION_DIGEST_DOMAIN: &[u8] = b"aidaemon.mandate.action.v1\0";

const MUTATION_EFFECTS: [(&str, ToolMutationEffects); 11] = [
    (
        "local_source_write",
        ToolMutationEffects::LOCAL_SOURCE_WRITE,
    ),
    (
        "local_workspace_write",
        ToolMutationEffects::LOCAL_WORKSPACE_WRITE,
    ),
    (
        "local_derived_write",
        ToolMutationEffects::LOCAL_DERIVED_WRITE,
    ),
    ("repository_write", ToolMutationEffects::REPOSITORY_WRITE),
    ("remote_mutation", ToolMutationEffects::REMOTE_MUTATION),
    ("remote_deploy", ToolMutationEffects::REMOTE_DEPLOY),
    ("external_delivery", ToolMutationEffects::EXTERNAL_DELIVERY),
    ("process_state", ToolMutationEffects::PROCESS_STATE),
    ("configuration", ToolMutationEffects::CONFIGURATION),
    ("destructive", ToolMutationEffects::DESTRUCTIVE),
    ("unspecified", ToolMutationEffects::UNSPECIFIED),
];

/// Stable, non-sensitive reason code for a rejected mandate action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MandateAuthorityDenial {
    InvalidMandate,
    InvalidAuthority,
    MandateInactive,
    MandateExpired,
    InvalidDecisionCycle,
    DecisionCycleNotAct,
    MandateVersionMismatch,
    InvalidCycleBudget,
    CycleBudgetExhausted,
    InvalidToolName,
    InvalidArguments,
    UnknownCallSemantics,
    UnsupportedCallSemantics,
    ObservationNotAllowed,
    ToolNotAllowed,
    UnknownMutationEffect,
    MutationEffectNotAllowed,
    TargetRequired,
    TargetNotAllowed,
    GrantMismatch,
}

impl MandateAuthorityDenial {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidMandate => "invalid_mandate",
            Self::InvalidAuthority => "invalid_authority",
            Self::MandateInactive => "mandate_inactive",
            Self::MandateExpired => "mandate_expired",
            Self::InvalidDecisionCycle => "invalid_decision_cycle",
            Self::DecisionCycleNotAct => "decision_cycle_not_act",
            Self::MandateVersionMismatch => "mandate_version_mismatch",
            Self::InvalidCycleBudget => "invalid_cycle_budget",
            Self::CycleBudgetExhausted => "cycle_budget_exhausted",
            Self::InvalidToolName => "invalid_tool_name",
            Self::InvalidArguments => "invalid_arguments",
            Self::UnknownCallSemantics => "unknown_call_semantics",
            Self::UnsupportedCallSemantics => "unsupported_call_semantics",
            Self::ObservationNotAllowed => "observation_not_allowed",
            Self::ToolNotAllowed => "tool_not_allowed",
            Self::UnknownMutationEffect => "unknown_mutation_effect",
            Self::MutationEffectNotAllowed => "mutation_effect_not_allowed",
            Self::TargetRequired => "target_required",
            Self::TargetNotAllowed => "target_not_allowed",
            Self::GrantMismatch => "grant_mismatch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MandateAuthorityDecision {
    Allow(MandateAuthorityGrant),
    Deny(MandateAuthorityDenial),
}

impl MandateAuthorityDecision {
    #[cfg(test)]
    pub(crate) fn grant(&self) -> Option<&MandateAuthorityGrant> {
        match self {
            Self::Allow(grant) => Some(grant),
            Self::Deny(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CanonicalTarget {
    kind: &'static str,
    value: String,
}

/// Evaluate one exact tool call against the current owner mandate and current
/// decision cycle.
///
/// `evaluated_at` is supplied by the caller so the function remains pure and
/// expiration behavior is deterministic in tests. The cycle's
/// `action_attempts` field is the current per-cycle mutation-budget input. An
/// execution integration must still reserve the attempt atomically before it
/// issues the returned grant to a mutating call.
pub(crate) fn authorize_mandate_action(
    mandate: &Mandate,
    decision_cycle: &MandateDecisionCycle,
    tool_name: &str,
    arguments_json: &str,
    semantics: &ToolCallSemantics,
    evaluated_at: &DateTime<Utc>,
) -> MandateAuthorityDecision {
    if let Err(reason) = validate_current_context(mandate, decision_cycle, evaluated_at) {
        return deny(reason);
    }

    if tool_name.is_empty() || tool_name.trim() != tool_name || tool_name.contains('*') {
        return deny(MandateAuthorityDenial::InvalidToolName);
    }
    if !tool_is_allowed(&mandate.authority.allowed_tools, tool_name) {
        return deny(MandateAuthorityDenial::ToolNotAllowed);
    }
    let Ok(canonical_arguments) = canonical_arguments(arguments_json) else {
        return deny(MandateAuthorityDenial::InvalidArguments);
    };

    let call_mutates = match call_mutates(semantics) {
        Ok(call_mutates) => call_mutates,
        Err(reason) => return deny(reason),
    };
    if let Err(reason) = validate_http_identity_binding(
        tool_name,
        arguments_json,
        semantics,
        call_mutates,
        &mandate.authority.allowed_target_prefixes,
    ) {
        return deny(reason);
    }

    let effect_names = if call_mutates {
        let effects = if semantics.mutation_effects.is_empty() {
            ToolMutationEffects::UNSPECIFIED
        } else {
            semantics.mutation_effects
        };
        let Ok(names) = mutation_effect_names(effects) else {
            return deny(MandateAuthorityDenial::UnknownMutationEffect);
        };
        names
    } else {
        Vec::new()
    };

    if semantics
        .target_hints
        .iter()
        .any(|target| target.value.is_empty() || target.value.trim() != target.value)
    {
        return deny(MandateAuthorityDenial::TargetNotAllowed);
    }
    if has_unmediated_local_target(semantics) {
        return deny(MandateAuthorityDenial::TargetNotAllowed);
    }
    let targets = canonical_targets(semantics);
    if targets.is_empty() {
        return deny(MandateAuthorityDenial::TargetRequired);
    }
    if !mandate.authority.allowed_target_prefixes.is_empty()
        && targets.iter().any(|target| {
            !mandate
                .authority
                .allowed_target_prefixes
                .iter()
                .any(|prefix| target_is_within_scope(target, prefix))
        })
    {
        return deny(MandateAuthorityDenial::TargetNotAllowed);
    }

    if call_mutates {
        if effect_names
            .iter()
            .any(|effect| !mandate.authority.allows_effect(effect))
        {
            return deny(MandateAuthorityDenial::MutationEffectNotAllowed);
        }
        if decision_cycle.action_attempts
            >= i64::from(mandate.authority.max_mutating_actions_per_cycle)
        {
            return deny(MandateAuthorityDenial::CycleBudgetExhausted);
        }
    } else if !mandate.authority.allow_observations {
        return deny(MandateAuthorityDenial::ObservationNotAllowed);
    }

    let action_digest = action_digest(
        mandate,
        decision_cycle,
        tool_name,
        &canonical_arguments,
        call_mutates,
        &effect_names,
        &targets,
    );

    MandateAuthorityDecision::Allow(MandateAuthorityGrant {
        mandate_id: mandate.id.clone(),
        mandate_version: mandate.version,
        decision_cycle_id: decision_cycle.id.clone(),
        action_digest,
        counts_toward_cycle_budget: call_mutates,
        reserved_action_attempt: if call_mutates {
            decision_cycle.action_attempts + 1
        } else {
            0
        },
        tool_call_id: None,
    })
}

/// Check a pre-decision observation against the active mandate's read scope.
///
/// Unlike mutations, observations do not require an ACT cycle and do not
/// receive a reusable grant. They still require an exact allowed tool, valid
/// typed semantics, and any applicable target prefix so an output mandate
/// cannot roam across unrelated owner data during deliberation.
pub(crate) fn authorize_mandate_observation(
    mandate: &Mandate,
    tool_name: &str,
    arguments_json: &str,
    semantics: &ToolCallSemantics,
    evaluated_at: &DateTime<Utc>,
) -> Result<(), MandateAuthorityDenial> {
    validate_current_mandate(mandate, evaluated_at)?;
    if !mandate.authority.allow_observations {
        return Err(MandateAuthorityDenial::ObservationNotAllowed);
    }
    if tool_name.is_empty() || tool_name.trim() != tool_name || tool_name.contains('*') {
        return Err(MandateAuthorityDenial::InvalidToolName);
    }
    if !tool_is_allowed(&mandate.authority.allowed_tools, tool_name) {
        return Err(MandateAuthorityDenial::ToolNotAllowed);
    }
    canonical_arguments(arguments_json).map_err(|_| MandateAuthorityDenial::InvalidArguments)?;
    if call_mutates(semantics)? {
        return Err(MandateAuthorityDenial::UnsupportedCallSemantics);
    }
    validate_http_identity_binding(
        tool_name,
        arguments_json,
        semantics,
        false,
        &mandate.authority.allowed_target_prefixes,
    )?;
    if semantics
        .target_hints
        .iter()
        .any(|target| target.value.is_empty() || target.value.trim() != target.value)
    {
        return Err(MandateAuthorityDenial::TargetNotAllowed);
    }
    if has_unmediated_local_target(semantics) {
        return Err(MandateAuthorityDenial::TargetNotAllowed);
    }
    let targets = canonical_targets(semantics);
    if targets.is_empty() {
        return Err(MandateAuthorityDenial::TargetRequired);
    }
    if mandate.authority.allowed_target_prefixes.is_empty() {
        return Err(MandateAuthorityDenial::TargetRequired);
    }
    if targets.iter().any(|target| {
        !mandate
            .authority
            .allowed_target_prefixes
            .iter()
            .any(|prefix| target_is_within_scope(target, prefix))
    }) {
        return Err(MandateAuthorityDenial::TargetNotAllowed);
    }
    Ok(())
}

fn validate_current_mandate(
    mandate: &Mandate,
    evaluated_at: &DateTime<Utc>,
) -> Result<(), MandateAuthorityDenial> {
    if mandate.id.trim().is_empty() || mandate.goal_id.trim().is_empty() || mandate.version <= 0 {
        return Err(MandateAuthorityDenial::InvalidMandate);
    }
    if validate_authority(mandate).is_err() {
        return Err(MandateAuthorityDenial::InvalidAuthority);
    }
    // `active` is only executable when backed by durable owner confirmation.
    // Treat an impossible/corrupt active-without-confirmation row exactly like
    // any other inactive authority state so the pure gate cannot be bypassed
    // even if a caller loads a mandate outside the guarded SQL paths.
    if mandate.status != "active" || mandate.confirmed_at.is_none() {
        return Err(MandateAuthorityDenial::MandateInactive);
    }
    if let Some(expires_at) = mandate.expires_at.as_deref() {
        let Ok(expires_at) = DateTime::parse_from_rfc3339(expires_at) else {
            return Err(MandateAuthorityDenial::InvalidMandate);
        };
        if expires_at.with_timezone(&Utc) <= *evaluated_at {
            return Err(MandateAuthorityDenial::MandateExpired);
        }
    }
    Ok(())
}

fn validate_current_context(
    mandate: &Mandate,
    decision_cycle: &MandateDecisionCycle,
    evaluated_at: &DateTime<Utc>,
) -> Result<(), MandateAuthorityDenial> {
    validate_current_mandate(mandate, evaluated_at)?;

    if decision_cycle.id.trim().is_empty()
        || decision_cycle.goal_run_id.trim().is_empty()
        || decision_cycle.mandate_id != mandate.id
    {
        return Err(MandateAuthorityDenial::InvalidDecisionCycle);
    }
    if decision_cycle.outcome != MandateDecisionOutcome::Act {
        return Err(MandateAuthorityDenial::DecisionCycleNotAct);
    }
    if decision_cycle.mandate_version != mandate.version {
        return Err(MandateAuthorityDenial::MandateVersionMismatch);
    }
    if decision_cycle.action_attempts < 0 {
        return Err(MandateAuthorityDenial::InvalidCycleBudget);
    }
    Ok(())
}

/// Revalidate an action-bound grant at the final dispatch boundary after a
/// mutating action attempt has been atomically reserved.
///
/// Authorization evaluates a mutation against the count before reservation;
/// final dispatch sees the count after reservation. For that reason this
/// function accepts `1..=max_mutating_actions_per_cycle` for mutation grants,
/// reconstructs the pre-reservation count, and recomputes the exact expected
/// grant. It compares the complete grant, including its mandate/cycle binding,
/// digest, and budget classification.
pub(crate) fn validate_mandate_grant(
    mandate: &Mandate,
    decision_cycle: &MandateDecisionCycle,
    tool_name: &str,
    arguments_json: &str,
    semantics: &ToolCallSemantics,
    evaluated_at: &DateTime<Utc>,
    grant: &MandateAuthorityGrant,
) -> Result<(), MandateAuthorityDenial> {
    validate_current_context(mandate, decision_cycle, evaluated_at)?;
    let mutation = call_mutates(semantics)?;
    let mut authorization_cycle = decision_cycle.clone();
    if mutation {
        let max_attempts = i64::from(mandate.authority.max_mutating_actions_per_cycle);
        if grant.reserved_action_attempt < 1
            || grant.reserved_action_attempt > max_attempts
            || decision_cycle.action_attempts != grant.reserved_action_attempt
            || decision_cycle.action_attempts > max_attempts
        {
            return Err(MandateAuthorityDenial::InvalidCycleBudget);
        }
        authorization_cycle.action_attempts = grant.reserved_action_attempt - 1;
    } else if grant.reserved_action_attempt != 0 {
        return Err(MandateAuthorityDenial::GrantMismatch);
    }

    match authorize_mandate_action(
        mandate,
        &authorization_cycle,
        tool_name,
        arguments_json,
        semantics,
        evaluated_at,
    ) {
        MandateAuthorityDecision::Deny(reason) => Err(reason),
        MandateAuthorityDecision::Allow(mut expected) => {
            expected.tool_call_id.clone_from(&grant.tool_call_id);
            if &expected == grant {
                Ok(())
            } else {
                Err(MandateAuthorityDenial::GrantMismatch)
            }
        }
    }
}

fn deny(reason: MandateAuthorityDenial) -> MandateAuthorityDecision {
    MandateAuthorityDecision::Deny(reason)
}

fn validate_authority(mandate: &Mandate) -> Result<(), ()> {
    mandate.authority.validate().map_err(|_| ())
}

fn tool_is_allowed(patterns: &[String], tool_name: &str) -> bool {
    patterns.iter().any(|pattern| pattern == tool_name)
}

fn call_mutates(semantics: &ToolCallSemantics) -> Result<bool, MandateAuthorityDenial> {
    match semantics.effect {
        ToolCallEffect::Unknown => Err(MandateAuthorityDenial::UnknownCallSemantics),
        ToolCallEffect::Administrative => Err(MandateAuthorityDenial::UnsupportedCallSemantics),
        ToolCallEffect::Observation => Ok(!semantics.mutation_effects.is_empty()),
        ToolCallEffect::Mutation | ToolCallEffect::ObservationAndMutation => Ok(true),
    }
}

fn mutation_effect_names(
    effects: ToolMutationEffects,
) -> Result<Vec<&'static str>, MandateAuthorityDenial> {
    let mut recognized = ToolMutationEffects::NONE;
    let mut names = Vec::new();
    for (name, effect) in MUTATION_EFFECTS {
        if effects.contains(effect) {
            recognized = recognized.union(effect);
            names.push(name);
        }
    }
    if recognized != effects {
        return Err(MandateAuthorityDenial::UnknownMutationEffect);
    }
    Ok(names)
}

fn validate_http_identity_binding(
    tool_name: &str,
    arguments_json: &str,
    semantics: &ToolCallSemantics,
    _call_mutates: bool,
    allowed_target_prefixes: &[String],
) -> Result<(), MandateAuthorityDenial> {
    if tool_name != "http_request" {
        return Ok(());
    }
    let arguments = serde_json::from_str::<Value>(arguments_json)
        .map_err(|_| MandateAuthorityDenial::InvalidArguments)?;
    let Some(arguments) = arguments.as_object() else {
        return Err(MandateAuthorityDenial::InvalidArguments);
    };
    let profile_name = match arguments.get("auth_profile") {
        None | Some(Value::Null) => {
            if arguments
                .get("account_id")
                .is_some_and(|value| !value.is_null())
            {
                return Err(MandateAuthorityDenial::TargetNotAllowed);
            }
            return Ok(());
        }
        Some(Value::String(value)) if !value.is_empty() && value.trim() == value => value,
        _ => return Err(MandateAuthorityDenial::TargetNotAllowed),
    };
    let expected_profile = scoped_resource_id("auth_profile", profile_name);
    let has_profile = semantics.target_hints.iter().any(|target| {
        target.kind == ToolTargetHintKind::ResourceId && target.value == expected_profile
    });
    if !has_profile {
        return Err(MandateAuthorityDenial::TargetRequired);
    }
    if !allowed_target_prefixes
        .iter()
        .any(|scope| scope == &expected_profile)
    {
        return Err(MandateAuthorityDenial::TargetNotAllowed);
    }
    let account_id = match arguments.get("account_id") {
        Some(Value::String(value)) if !value.is_empty() && value.trim() == value => value,
        None | Some(Value::Null) => return Err(MandateAuthorityDenial::TargetRequired),
        _ => return Err(MandateAuthorityDenial::TargetNotAllowed),
    };
    let expected_account = scoped_resource_id("account", account_id);
    let has_account = semantics.target_hints.iter().any(|target| {
        target.kind == ToolTargetHintKind::ResourceId && target.value == expected_account
    });
    if !has_account {
        return Err(MandateAuthorityDenial::TargetRequired);
    }
    if !allowed_target_prefixes
        .iter()
        .any(|scope| scope == &expected_account)
    {
        return Err(MandateAuthorityDenial::TargetNotAllowed);
    }
    Ok(())
}

fn scoped_resource_id(kind: &str, identifier: &str) -> String {
    let mut encoded = String::with_capacity(identifier.len());
    for byte in identifier.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    format!("{kind}:{encoded}")
}

fn canonical_targets(semantics: &ToolCallSemantics) -> Vec<CanonicalTarget> {
    let mut targets = Vec::with_capacity(semantics.target_hints.len());
    for target in &semantics.target_hints {
        targets.push(CanonicalTarget {
            kind: target_kind_name(target.kind),
            value: target.value.clone(),
        });
    }
    targets.sort();
    targets.dedup();
    targets
}

/// Derive the content-safe audit fields persisted with a mutation reservation.
/// This runs only after the exact semantics passed authority evaluation. URL
/// query strings and unrecognized resource identifiers are never persisted.
pub(crate) type MandateMutationAuditScope = (Vec<String>, Vec<MandateMutationTarget>, Vec<String>);

pub(crate) fn mutation_audit_scope(
    semantics: &ToolCallSemantics,
) -> Result<MandateMutationAuditScope, MandateAuthorityDenial> {
    if !call_mutates(semantics)? {
        return Err(MandateAuthorityDenial::UnsupportedCallSemantics);
    }
    let effects = if semantics.mutation_effects.is_empty() {
        ToolMutationEffects::UNSPECIFIED
    } else {
        semantics.mutation_effects
    };
    let mutation_effects = mutation_effect_names(effects)?
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut targets = Vec::new();
    let mut accounts = Vec::new();
    for target in canonical_targets(semantics) {
        let identifier = match target.kind {
            "url" => {
                let mut url = reqwest::Url::parse(&target.value)
                    .map_err(|_| MandateAuthorityDenial::TargetNotAllowed)?;
                if !url.username().is_empty() || url.password().is_some() || url.cannot_be_a_base()
                {
                    return Err(MandateAuthorityDenial::TargetNotAllowed);
                }
                url.set_query(None);
                url.set_fragment(None);
                url.to_string()
            }
            "resource_id" => {
                if safe_account_identifier(&target.value) {
                    accounts.push(target.value.clone());
                    target.value
                } else {
                    format!("sha256:{:x}", Sha256::digest(target.value.as_bytes()))
                }
            }
            _ => return Err(MandateAuthorityDenial::TargetNotAllowed),
        };
        targets.push(MandateMutationTarget {
            kind: target.kind.to_string(),
            identifier,
        });
    }
    targets.sort();
    targets.dedup();
    accounts.sort();
    accounts.dedup();
    Ok((mutation_effects, targets, accounts))
}

fn safe_account_identifier(value: &str) -> bool {
    let suffix = value
        .strip_prefix("auth_profile:")
        .or_else(|| value.strip_prefix("account:"));
    suffix.is_some_and(|suffix| {
        if suffix.is_empty() || suffix.len() > 192 {
            return false;
        }
        let bytes = suffix.as_bytes();
        let mut index = 0usize;
        while index < bytes.len() {
            let byte = bytes[index];
            if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-') {
                index += 1;
                continue;
            }
            if byte == b'%'
                && index + 2 < bytes.len()
                && bytes[index + 1].is_ascii_hexdigit()
                && bytes[index + 2].is_ascii_hexdigit()
                && !bytes[index + 1].is_ascii_lowercase()
                && !bytes[index + 2].is_ascii_lowercase()
            {
                index += 3;
                continue;
            }
            return false;
        }
        true
    })
}

fn has_unmediated_local_target(semantics: &ToolCallSemantics) -> bool {
    semantics.target_hints.iter().any(|target| {
        matches!(
            target.kind,
            ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope
        )
    })
}

fn target_is_within_scope(target: &CanonicalTarget, raw_scope: &str) -> bool {
    let scope = raw_scope.trim();
    match target.kind {
        "url" => url_is_within_scope(&target.value, scope),
        // V1 identity scopes are always exact. Prefix/hierarchy semantics
        // would let `account:` or `auth_profile:` authorize every credential.
        "resource_id" => !scope.is_empty() && target.value == scope,
        // Local path/project mediation is intentionally unavailable in v1.
        "path" | "project_scope" => false,
        _ => false,
    }
}

fn url_is_within_scope(raw_target: &str, raw_scope: &str) -> bool {
    let Ok(target) = reqwest::Url::parse(raw_target) else {
        return false;
    };
    let Ok(scope) = reqwest::Url::parse(raw_scope) else {
        return false;
    };
    if !target.username().is_empty()
        || target.password().is_some()
        || !scope.username().is_empty()
        || scope.password().is_some()
        || target.fragment().is_some()
        || scope.fragment().is_some()
    {
        return false;
    }
    if target.scheme() != scope.scheme()
        || target.host_str().is_none()
        || target.host_str() != scope.host_str()
        || target.port_or_known_default() != scope.port_or_known_default()
    {
        return false;
    }

    // `Url` resolves dot segments while parsing. Compare normalized paths on
    // a segment boundary so `/2` cannot authorize `/20`; a root scope is the
    // only path that covers its entire origin.
    let target_path = target.path();
    let scope_path = scope.path();
    let path_matches = if scope_path == "/" {
        true
    } else {
        let scope_path = scope_path.trim_end_matches('/');
        target_path == scope_path
            || target_path
                .strip_prefix(scope_path)
                .is_some_and(|suffix| suffix.starts_with('/'))
    };
    if !path_matches {
        return false;
    }

    // Query data is outbound data. It must be owner-pinned exactly instead of
    // inheriting authority from a queryless path scope.
    scope.query() == target.query()
}

fn target_kind_name(kind: ToolTargetHintKind) -> &'static str {
    match kind {
        ToolTargetHintKind::Url => "url",
        ToolTargetHintKind::Path => "path",
        ToolTargetHintKind::ProjectScope => "project_scope",
        ToolTargetHintKind::ResourceId => "resource_id",
    }
}

fn canonical_arguments(arguments_json: &str) -> Result<String, ()> {
    let parsed = serde_json::from_str::<Value>(arguments_json).map_err(|_| ())?;
    if !parsed.is_object() {
        return Err(());
    }
    let mut canonical = String::new();
    write_canonical_json(&parsed, &mut canonical);
    Ok(canonical)
}

fn write_canonical_json(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical_json(&map[key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn action_digest(
    mandate: &Mandate,
    decision_cycle: &MandateDecisionCycle,
    tool_name: &str,
    canonical_arguments: &str,
    call_mutates: bool,
    effect_names: &[&str],
    targets: &[CanonicalTarget],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(ACTION_DIGEST_DOMAIN);
    hash_field(&mut hasher, b"mandate_id", mandate.id.as_bytes());
    hash_field(
        &mut hasher,
        b"mandate_version",
        mandate.version.to_string().as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"decision_cycle_id",
        decision_cycle.id.as_bytes(),
    );
    hash_field(&mut hasher, b"tool_name", tool_name.as_bytes());
    hash_field(&mut hasher, b"arguments", canonical_arguments.as_bytes());
    if call_mutates {
        hash_field(
            &mut hasher,
            b"reserved_action_attempt",
            (decision_cycle.action_attempts + 1).to_string().as_bytes(),
        );
    }
    hash_field(
        &mut hasher,
        b"call_class",
        if call_mutates {
            b"mutation"
        } else {
            b"observation"
        },
    );
    for effect in effect_names {
        hash_field(&mut hasher, b"mutation_effect", effect.as_bytes());
    }
    for target in targets {
        hash_field(&mut hasher, b"target_kind", target.kind.as_bytes());
        hash_field(&mut hasher, b"target_value", target.value.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_field(hasher: &mut Sha256, label: &[u8], value: &[u8]) {
    hasher.update((label.len() as u64).to_be_bytes());
    hasher.update(label);
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{MandateAuthority, ToolTargetHint};

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-08-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn authority() -> MandateAuthority {
        MandateAuthority {
            allow_observations: true,
            allowed_tools: vec!["http_request".to_string(), "web_fetch".to_string()],
            allowed_mutation_effects: vec![
                "remote_mutation".to_string(),
                "external_delivery".to_string(),
            ],
            allowed_target_prefixes: vec!["https://api.x.com/2/".to_string()],
            max_mutating_actions_per_cycle: 2,
            max_mutating_actions_per_rolling_24h: 8,
            min_seconds_between_mutations: 900,
        }
    }

    fn mandate() -> Mandate {
        let mut mandate = Mandate::new(
            "goal-1",
            None,
            "Steward @aidaemon_ai",
            "owner-session",
            authority(),
            60,
            3_600,
            300,
        );
        mandate.id = "mandate-1".to_string();
        mandate.next_review_at = "2026-08-01T12:05:00Z".to_string();
        mandate.created_at = "2026-08-01T00:00:00Z".to_string();
        mandate.updated_at = mandate.created_at.clone();
        mandate
    }

    fn cycle(mandate: &Mandate) -> MandateDecisionCycle {
        let mut cycle = MandateDecisionCycle::new(
            &mandate.id,
            "goal-run-1",
            MandateDecisionOutcome::Act,
            "A useful reply is warranted",
            mandate.version,
        );
        cycle.id = "cycle-1".to_string();
        cycle.created_at = "2026-08-01T12:00:00Z".to_string();
        cycle.updated_at = cycle.created_at.clone();
        cycle
    }

    fn post_semantics() -> ToolCallSemantics {
        ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        )
        .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/tweets")
    }

    fn decide(
        mandate: &Mandate,
        cycle: &MandateDecisionCycle,
        tool_name: &str,
        arguments: &str,
        semantics: &ToolCallSemantics,
    ) -> MandateAuthorityDecision {
        authorize_mandate_action(mandate, cycle, tool_name, arguments, semantics, &now())
    }

    fn denial(decision: MandateAuthorityDecision) -> MandateAuthorityDenial {
        match decision {
            MandateAuthorityDecision::Deny(reason) => reason,
            MandateAuthorityDecision::Allow(grant) => {
                panic!("expected denial, got grant {}", grant.action_digest)
            }
        }
    }

    #[test]
    fn allows_exact_mutation_and_returns_action_bound_grant() {
        let mandate = mandate();
        let base_cycle = cycle(&mandate);
        let decision = decide(
            &mandate,
            &base_cycle,
            "http_request",
            r#"{"url":"https://api.x.com/2/tweets","method":"POST","body":{"text":"hello"}}"#,
            &post_semantics(),
        );
        let grant = decision.grant().expect("action should be authorized");
        assert_eq!(grant.mandate_id, mandate.id);
        assert_eq!(grant.mandate_version, mandate.version);
        assert_eq!(grant.decision_cycle_id, base_cycle.id);
        assert!(grant.counts_toward_cycle_budget);
        assert_eq!(grant.action_digest.len(), 64);
        assert!(grant
            .action_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn authenticated_http_action_requires_url_profile_and_exact_account_targets() {
        let semantics = ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        )
        .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/tweets")
        .with_target_hint(ToolTargetHintKind::ResourceId, "auth_profile:twitter-prod")
        .with_target_hint(ToolTargetHintKind::ResourceId, "account:2244994945");
        let arguments = r#"{"url":"https://api.x.com/2/tweets","method":"POST","auth_profile":"twitter-prod","account_id":"2244994945"}"#;

        let mut allowed = mandate();
        allowed.authority.allowed_tools = vec!["http_request".to_string()];
        allowed
            .authority
            .allowed_target_prefixes
            .push("auth_profile:twitter-prod".to_string());
        allowed
            .authority
            .allowed_target_prefixes
            .push("account:2244994945".to_string());
        let base_cycle = cycle(&allowed);
        let allowed_decision = decide(&allowed, &base_cycle, "http_request", arguments, &semantics);
        assert!(
            allowed_decision.grant().is_some(),
            "the exact URL, profile, and account identity should be authorized, got {allowed_decision:?}"
        );

        let semantics_without_account = ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        )
        .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/tweets")
        .with_target_hint(ToolTargetHintKind::ResourceId, "auth_profile:twitter-prod");
        let arguments_without_account =
            r#"{"url":"https://api.x.com/2/tweets","method":"POST","auth_profile":"twitter-prod"}"#;
        assert_eq!(
            denial(decide(
                &allowed,
                &cycle(&allowed),
                "http_request",
                arguments_without_account,
                &semantics_without_account,
            )),
            MandateAuthorityDenial::TargetRequired,
            "an authenticated HTTP mutation without account_id must fail before grant issuance"
        );
        assert_eq!(
            denial(decide(
                &allowed,
                &cycle(&allowed),
                "http_request",
                arguments,
                &semantics_without_account,
            )),
            MandateAuthorityDenial::TargetRequired,
            "account_id without its exact account resource target must fail before grant issuance"
        );

        let mut missing_profile_scope = mandate();
        missing_profile_scope.authority.allowed_tools = vec!["http_request".to_string()];
        missing_profile_scope
            .authority
            .allowed_target_prefixes
            .push("account:2244994945".to_string());
        assert_eq!(
            denial(decide(
                &missing_profile_scope,
                &cycle(&missing_profile_scope),
                "http_request",
                arguments,
                &semantics,
            )),
            MandateAuthorityDenial::TargetNotAllowed,
            "URL and account scope must not substitute for the exact auth profile"
        );

        let mut wrong_profile_scope = mandate();
        wrong_profile_scope.authority.allowed_tools = vec!["http_request".to_string()];
        wrong_profile_scope
            .authority
            .allowed_target_prefixes
            .push("auth_profile:twitter-personal".to_string());
        wrong_profile_scope
            .authority
            .allowed_target_prefixes
            .push("account:2244994945".to_string());
        assert_eq!(
            denial(decide(
                &wrong_profile_scope,
                &cycle(&wrong_profile_scope),
                "http_request",
                arguments,
                &semantics,
            )),
            MandateAuthorityDenial::TargetNotAllowed,
            "a sibling profile must not satisfy the profile target"
        );

        let mut missing_account_scope = mandate();
        missing_account_scope.authority.allowed_tools = vec!["http_request".to_string()];
        missing_account_scope
            .authority
            .allowed_target_prefixes
            .push("auth_profile:twitter-prod".to_string());
        assert_eq!(
            denial(decide(
                &missing_account_scope,
                &cycle(&missing_account_scope),
                "http_request",
                arguments,
                &semantics,
            )),
            MandateAuthorityDenial::TargetNotAllowed,
            "URL and profile scope must not substitute for the exact account"
        );

        let mut wrong_account_scope = allowed.clone();
        wrong_account_scope.authority.allowed_target_prefixes = vec![
            "https://api.x.com/2/".to_string(),
            "auth_profile:twitter-prod".to_string(),
            "account:9999999999".to_string(),
        ];
        assert_eq!(
            denial(decide(
                &wrong_account_scope,
                &cycle(&wrong_account_scope),
                "http_request",
                arguments,
                &semantics,
            )),
            MandateAuthorityDenial::TargetNotAllowed,
            "a sibling account must not satisfy the account target"
        );

        let mut broader_looking_exact_identity_scope = allowed.clone();
        broader_looking_exact_identity_scope
            .authority
            .allowed_target_prefixes = vec![
            "https://api.x.com/2/".to_string(),
            "auth_profile:twitter".to_string(),
            "account:2244".to_string(),
        ];
        assert_eq!(
            denial(decide(
                &broader_looking_exact_identity_scope,
                &cycle(&broader_looking_exact_identity_scope),
                "http_request",
                arguments,
                &semantics,
            )),
            MandateAuthorityDenial::TargetNotAllowed,
            "valid but broader-looking identity IDs must not authorize different exact identities"
        );
    }

    #[test]
    fn authenticated_http_observation_requires_exact_profile_and_account_binding() {
        let semantics = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me")
            .with_target_hint(ToolTargetHintKind::ResourceId, "auth_profile:twitter-prod")
            .with_target_hint(ToolTargetHintKind::ResourceId, "account:2244994945");
        let arguments = r#"{"url":"https://api.x.com/2/users/me","method":"GET","auth_profile":"twitter-prod","account_id":"2244994945"}"#;
        let mut allowed = mandate();
        allowed.authority.allowed_tools = vec!["http_request".to_string()];
        allowed.authority.allowed_target_prefixes = vec![
            "https://api.x.com/2/users/me".to_string(),
            "auth_profile:twitter-prod".to_string(),
            "account:2244994945".to_string(),
        ];
        assert_eq!(
            authorize_mandate_observation(&allowed, "http_request", arguments, &semantics, &now(),),
            Ok(())
        );

        let missing_account = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me")
            .with_target_hint(ToolTargetHintKind::ResourceId, "auth_profile:twitter-prod");
        assert_eq!(
            authorize_mandate_observation(
                &allowed,
                "http_request",
                arguments,
                &missing_account,
                &now(),
            ),
            Err(MandateAuthorityDenial::TargetRequired)
        );

        let mut wrong_profile_scope = allowed;
        wrong_profile_scope.authority.allowed_target_prefixes[1] =
            "auth_profile:twitter-personal".to_string();
        assert_eq!(
            authorize_mandate_observation(
                &wrong_profile_scope,
                "http_request",
                arguments,
                &semantics,
                &now(),
            ),
            Err(MandateAuthorityDenial::TargetNotAllowed)
        );
    }

    #[test]
    fn mutation_audit_scope_preserves_canonical_encoded_profile_and_strips_url_query() {
        let semantics = ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        )
        .with_target_hint(
            ToolTargetHintKind::Url,
            "https://api.x.com/2/tweets?access_token=never-persist",
        )
        .with_target_hint(
            ToolTargetHintKind::ResourceId,
            "auth_profile:Twitter%20Prod%2Faccount",
        )
        .with_target_hint(ToolTargetHintKind::ResourceId, "account:2244994945");
        let (_effects, targets, accounts) =
            mutation_audit_scope(&semantics).expect("audit scope should be content-safe");
        assert_eq!(
            accounts,
            vec![
                "account:2244994945".to_string(),
                "auth_profile:Twitter%20Prod%2Faccount".to_string(),
            ]
        );
        assert!(targets.iter().any(|target| {
            target.kind == "url"
                && target.identifier == "https://api.x.com/2/tweets"
                && !target.identifier.contains("access_token")
        }));
    }

    #[test]
    fn exact_tools_are_allowed_but_near_matches_and_mcp_are_not() {
        let mandate = mandate();
        let cycle = cycle(&mandate);
        let semantics = post_semantics();
        let args = r#"{"url":"https://api.x.com/2/tweets"}"#;

        assert!(matches!(
            decide(&mandate, &cycle, "http_request", args, &semantics),
            MandateAuthorityDecision::Allow(_)
        ));
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request_extra",
                args,
                &semantics,
            )),
            MandateAuthorityDenial::ToolNotAllowed
        );
        assert_eq!(
            denial(decide(&mandate, &cycle, "mcp__x__post", args, &semantics,)),
            MandateAuthorityDenial::ToolNotAllowed
        );
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "mcp__other__post",
                args,
                &semantics,
            )),
            MandateAuthorityDenial::ToolNotAllowed
        );
    }

    #[test]
    fn malformed_tool_patterns_fail_closed() {
        let mut mandate = mandate();
        let cycle = cycle(&mandate);
        for invalid in ["*", "mcp__*__post", "mcp__x__**", ""] {
            mandate.authority.allowed_tools = vec![invalid.to_string()];
            assert_eq!(
                denial(decide(
                    &mandate,
                    &cycle,
                    "mcp__x__post",
                    r#"{"url":"https://api.x.com/2/tweets"}"#,
                    &post_semantics(),
                )),
                MandateAuthorityDenial::InvalidAuthority,
                "pattern {invalid:?} should fail closed"
            );
        }
    }

    #[test]
    fn paused_expired_and_malformed_expiration_are_denied() {
        let mut mandate = mandate();
        let cycle = cycle(&mandate);
        mandate.status = "paused".to_string();
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::MandateInactive
        );

        mandate.status = "active".to_string();
        mandate.confirmed_at = None;
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::MandateInactive
        );

        mandate.confirmed_at = Some("2026-08-01T00:00:00Z".to_string());
        mandate.expires_at = Some("2026-08-01T12:00:00Z".to_string());
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::MandateExpired
        );

        mandate.expires_at = Some("not-a-time".to_string());
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::InvalidMandate
        );
    }

    #[test]
    fn only_current_matching_act_cycle_is_authorized() {
        let mandate = mandate();

        let mut wrong_mandate = cycle(&mandate);
        wrong_mandate.mandate_id = "mandate-2".to_string();
        assert_eq!(
            denial(decide(
                &mandate,
                &wrong_mandate,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::InvalidDecisionCycle
        );

        let mut wait_cycle = cycle(&mandate);
        wait_cycle.outcome = MandateDecisionOutcome::Wait;
        assert_eq!(
            denial(decide(
                &mandate,
                &wait_cycle,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::DecisionCycleNotAct
        );

        let mut stale_cycle = cycle(&mandate);
        stale_cycle.mandate_version -= 1;
        assert_eq!(
            denial(decide(
                &mandate,
                &stale_cycle,
                "http_request",
                "{}",
                &post_semantics(),
            )),
            MandateAuthorityDenial::MandateVersionMismatch
        );
    }

    #[test]
    fn every_mutation_effect_must_be_explicitly_allowed() {
        let mut mandate = mandate();
        let cycle = cycle(&mandate);
        mandate.authority.allowed_mutation_effects = vec!["remote_mutation".to_string()];
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                r#"{"url":"https://api.x.com/2/tweets"}"#,
                &post_semantics(),
            )),
            MandateAuthorityDenial::MutationEffectNotAllowed
        );
    }

    #[test]
    fn mutation_effect_names_are_complete_and_unknown_bits_fail_closed() {
        let mut all_effects = ToolMutationEffects::NONE;
        for (_, effect) in MUTATION_EFFECTS {
            all_effects = all_effects.union(effect);
        }
        assert_eq!(
            mutation_effect_names(all_effects).unwrap(),
            MandateAuthority::EFFECT_NAMES
        );

        let unknown: ToolMutationEffects = serde_json::from_str("2147483648").unwrap();
        assert_eq!(
            mutation_effect_names(unknown),
            Err(MandateAuthorityDenial::UnknownMutationEffect)
        );
    }

    #[test]
    fn typed_mutation_effect_on_observation_is_governed_as_mutation() {
        let mandate = mandate();
        let cycle = cycle(&mandate);
        let semantics = ToolCallSemantics {
            effect: ToolCallEffect::Observation,
            mutation_effects: ToolMutationEffects::REMOTE_MUTATION,
            target_hints: vec![ToolTargetHint::new(
                ToolTargetHintKind::Url,
                "https://api.x.com/2/tweets",
            )
            .unwrap()],
            ..ToolCallSemantics::default()
        };
        let grant = decide(
            &mandate,
            &cycle,
            "http_request",
            r#"{"url":"https://api.x.com/2/tweets"}"#,
            &semantics,
        )
        .grant()
        .cloned()
        .expect("typed mutation evidence must take the conservative mutation path");
        assert!(grant.counts_toward_cycle_budget);
    }

    #[test]
    fn unspecified_mutation_fails_closed_unless_explicitly_allowed() {
        let mut mandate = mandate();
        let cycle = cycle(&mandate);
        let semantics = ToolCallSemantics::mutation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/tweets");
        assert_eq!(
            denial(decide(&mandate, &cycle, "http_request", "{}", &semantics,)),
            MandateAuthorityDenial::MutationEffectNotAllowed
        );

        mandate
            .authority
            .allowed_mutation_effects
            .push("unspecified".to_string());
        assert!(matches!(
            decide(&mandate, &cycle, "http_request", "{}", &semantics,),
            MandateAuthorityDecision::Allow(_)
        ));
    }

    #[test]
    fn configured_target_scope_requires_targets_and_checks_every_one() {
        let mandate = mandate();
        let cycle = cycle(&mandate);
        let no_target = ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION);
        assert_eq!(
            denial(decide(&mandate, &cycle, "http_request", "{}", &no_target,)),
            MandateAuthorityDenial::TargetRequired
        );

        let mixed_targets = ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION)
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/tweets")
            .with_target_hint(ToolTargetHintKind::Url, "https://evil.example/post");
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                "{}",
                &mixed_targets,
            )),
            MandateAuthorityDenial::TargetNotAllowed
        );
    }

    #[test]
    fn url_scope_uses_origin_and_normalized_segment_containment() {
        let mut mandate = mandate();
        mandate.authority.allowed_target_prefixes = vec!["https://api.x.com/2".to_string()];
        let cycle = cycle(&mandate);

        for denied_url in [
            "https://api.x.com.evil/2/tweets",
            "https://api.x.com/20/tweets",
            "https://api.x.com/2/../admin",
            "https://user@api.x.com/2/tweets",
            "https://api.x.com:444/2/tweets",
        ] {
            let semantics = ToolCallSemantics::mutation_with(
                ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
            )
            .with_target_hint(ToolTargetHintKind::Url, denied_url);
            assert_eq!(
                denial(decide(&mandate, &cycle, "http_request", "{}", &semantics,)),
                MandateAuthorityDenial::TargetNotAllowed,
                "URL {denied_url:?} must not escape the delegated origin/path"
            );
        }

        let queryful = ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        )
        .with_target_hint(
            ToolTargetHintKind::Url,
            "https://api.x.com/2/tweets?expansions=author_id",
        );
        assert_eq!(
            denial(decide(&mandate, &cycle, "http_request", "{}", &queryful,)),
            MandateAuthorityDenial::TargetNotAllowed
        );
    }

    #[test]
    fn url_scope_query_is_restrictive_when_present() {
        let mut mandate = mandate();
        mandate.authority.allowed_target_prefixes =
            vec!["https://api.x.com/2/users?tenant=owner".to_string()];
        let exact = ToolCallSemantics::observation().with_target_hint(
            ToolTargetHintKind::Url,
            "https://api.x.com/2/users?tenant=owner",
        );
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &exact, &now()),
            Ok(())
        );
        let changed = ToolCallSemantics::observation().with_target_hint(
            ToolTargetHintKind::Url,
            "https://api.x.com/2/users?tenant=other",
        );
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &changed, &now()),
            Err(MandateAuthorityDenial::TargetNotAllowed)
        );
    }

    #[test]
    fn local_path_and_project_targets_fail_closed_even_with_a_matching_prefix() {
        let mut mandate = mandate();
        mandate.authority.allowed_tools = vec!["web_fetch".to_string()];
        mandate.authority.allowed_target_prefixes = vec!["/srv/project".to_string()];
        for kind in [ToolTargetHintKind::Path, ToolTargetHintKind::ProjectScope] {
            let semantics = ToolCallSemantics::observation()
                .with_target_hint(kind, "/srv/project/safe-looking");
            assert_eq!(
                authorize_mandate_observation(&mandate, "web_fetch", "{}", &semantics, &now(),),
                Err(MandateAuthorityDenial::InvalidAuthority)
            );
        }
    }

    #[test]
    fn resource_id_scope_is_always_exact_and_hierarchy_scopes_are_invalid() {
        let mut mandate = mandate();
        mandate.authority.allowed_target_prefixes = vec!["account:123".to_string()];

        let exact = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::ResourceId, "account:123");
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &exact, &now()),
            Ok(())
        );

        let sibling = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::ResourceId, "account:123evil");
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &sibling, &now()),
            Err(MandateAuthorityDenial::TargetNotAllowed)
        );

        mandate.authority.allowed_target_prefixes = vec!["account:123:".to_string()];
        let descendant = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::ResourceId, "account:123:messages:7");
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &descendant, &now()),
            Err(MandateAuthorityDenial::InvalidAuthority)
        );
    }

    #[test]
    fn mutation_enabled_authority_requires_an_explicit_target_scope() {
        let mut mandate = mandate();
        mandate.authority.allowed_target_prefixes.clear();
        let cycle = cycle(&mandate);
        let semantics = ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION);
        assert_eq!(
            denial(decide(&mandate, &cycle, "http_request", "{}", &semantics,)),
            MandateAuthorityDenial::InvalidAuthority
        );
    }

    #[test]
    fn mutation_budget_is_per_cycle_and_observations_do_not_consume_it() {
        let mandate = mandate();
        let mut cycle = cycle(&mandate);
        cycle.action_attempts = 2;
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "http_request",
                r#"{"url":"https://api.x.com/2/tweets"}"#,
                &post_semantics(),
            )),
            MandateAuthorityDenial::CycleBudgetExhausted
        );

        let observation = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me");
        let grant = decide(&mandate, &cycle, "web_fetch", "{}", &observation)
            .grant()
            .cloned()
            .expect("observation remains allowed after mutation budget is spent");
        assert!(!grant.counts_toward_cycle_budget);

        cycle.action_attempts = -1;
        assert_eq!(
            denial(decide(&mandate, &cycle, "web_fetch", "{}", &observation,)),
            MandateAuthorityDenial::InvalidCycleBudget
        );
    }

    #[test]
    fn observations_require_a_scoped_tool_but_not_a_mutation_envelope() {
        let mut mandate = mandate();
        mandate.authority.allowed_tools = vec!["web_fetch".to_string()];
        mandate.authority.allowed_mutation_effects.clear();
        mandate.authority.max_mutating_actions_per_cycle = 0;
        mandate.authority.max_mutating_actions_per_rolling_24h = 0;
        mandate.authority.min_seconds_between_mutations = 0;
        let cycle = cycle(&mandate);
        let observation = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me");
        assert!(matches!(
            decide(&mandate, &cycle, "web_fetch", "{}", &observation),
            MandateAuthorityDecision::Allow(_)
        ));

        mandate.authority.allowed_tools = vec!["http_request".to_string()];
        assert_eq!(
            denial(decide(&mandate, &cycle, "web_fetch", "{}", &observation)),
            MandateAuthorityDenial::ToolNotAllowed
        );

        mandate.authority.allow_observations = false;
        mandate.authority.allowed_tools = vec!["web_fetch".to_string()];
        assert_eq!(
            denial(decide(&mandate, &cycle, "web_fetch", "{}", &observation,)),
            MandateAuthorityDenial::ObservationNotAllowed
        );
    }

    #[test]
    fn pre_act_observations_use_the_same_tool_and_target_scope() {
        let mandate = mandate();
        let x_read = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me");
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &x_read, &now()),
            Ok(())
        );
        assert_eq!(
            authorize_mandate_observation(&mandate, "read_file", "{}", &x_read, &now()),
            Err(MandateAuthorityDenial::ToolNotAllowed)
        );
        let outside = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://private.example/data");
        assert_eq!(
            authorize_mandate_observation(&mandate, "web_fetch", "{}", &outside, &now()),
            Err(MandateAuthorityDenial::TargetNotAllowed)
        );
    }

    #[test]
    fn unknown_administrative_and_invalid_arguments_fail_closed() {
        let mandate = mandate();
        let cycle = cycle(&mandate);
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "web_fetch",
                "{}",
                &ToolCallSemantics::default(),
            )),
            MandateAuthorityDenial::UnknownCallSemantics
        );
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "web_fetch",
                "{}",
                &ToolCallSemantics::administrative(),
            )),
            MandateAuthorityDenial::UnsupportedCallSemantics
        );
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "web_fetch",
                "not-json",
                &ToolCallSemantics::observation(),
            )),
            MandateAuthorityDenial::InvalidArguments
        );
        assert_eq!(
            denial(decide(
                &mandate,
                &cycle,
                "web_fetch",
                "[]",
                &ToolCallSemantics::observation(),
            )),
            MandateAuthorityDenial::InvalidArguments
        );
    }

    #[test]
    fn digest_is_canonical_but_bound_to_action_and_authority_context() {
        let mandate = mandate();
        let base_cycle = cycle(&mandate);
        let semantics = post_semantics();
        let first = decide(
            &mandate,
            &base_cycle,
            "http_request",
            r#"{"method":"POST","body":{"b":2,"a":1},"url":"https://api.x.com/2/tweets"}"#,
            &semantics,
        )
        .grant()
        .unwrap()
        .action_digest
        .clone();
        let reordered = decide(
            &mandate,
            &base_cycle,
            "http_request",
            r#"{"url":"https://api.x.com/2/tweets","body":{"a":1,"b":2},"method":"POST"}"#,
            &semantics,
        )
        .grant()
        .unwrap()
        .action_digest
        .clone();
        assert_eq!(first, reordered);

        let changed_args = decide(
            &mandate,
            &base_cycle,
            "http_request",
            r#"{"url":"https://api.x.com/2/tweets","body":{"a":1,"b":3},"method":"POST"}"#,
            &semantics,
        )
        .grant()
        .unwrap()
        .action_digest
        .clone();
        assert_ne!(first, changed_args);

        let mut other_cycle = base_cycle.clone();
        other_cycle.id = "cycle-2".to_string();
        let changed_cycle = decide(
            &mandate,
            &other_cycle,
            "http_request",
            r#"{"method":"POST","body":{"b":2,"a":1},"url":"https://api.x.com/2/tweets"}"#,
            &semantics,
        )
        .grant()
        .unwrap()
        .action_digest
        .clone();
        assert_ne!(first, changed_cycle);

        let mut new_version = mandate.clone();
        new_version.version += 1;
        let new_cycle = cycle(&new_version);
        let changed_version = decide(
            &new_version,
            &new_cycle,
            "http_request",
            r#"{"method":"POST","body":{"b":2,"a":1},"url":"https://api.x.com/2/tweets"}"#,
            &semantics,
        )
        .grant()
        .unwrap()
        .action_digest
        .clone();
        assert_ne!(first, changed_version);
    }

    #[test]
    fn target_order_and_duplicates_do_not_change_digest() {
        let mandate = mandate();
        let cycle = cycle(&mandate);
        let first_semantics =
            ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION)
                .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/a")
                .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/b");
        let reordered = ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION)
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/b")
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/a")
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/a");

        let digest = |semantics: &ToolCallSemantics| {
            decide(
                &mandate,
                &cycle,
                "http_request",
                r#"{"url":"https://api.x.com/2/a"}"#,
                semantics,
            )
            .grant()
            .unwrap()
            .action_digest
            .clone()
        };
        assert_eq!(digest(&first_semantics), digest(&reordered));
    }

    #[test]
    fn observation_target_hints_are_scoped_when_the_tool_reports_them() {
        let mandate = mandate();
        let cycle = cycle(&mandate);
        let semantics = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me");
        assert!(matches!(
            decide(&mandate, &cycle, "web_fetch", "{}", &semantics,),
            MandateAuthorityDecision::Allow(_)
        ));

        let outside = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://outside.example/read");
        assert_eq!(
            denial(decide(&mandate, &cycle, "web_fetch", "{}", &outside)),
            MandateAuthorityDenial::TargetNotAllowed
        );
    }

    #[test]
    fn malformed_mutation_target_hints_fail_closed() {
        let mandate = mandate();
        let cycle = cycle(&mandate);

        for invalid_value in [
            "",
            " https://api.x.com/2/tweets",
            "https://api.x.com/2/tweets ",
        ] {
            let semantics = ToolCallSemantics {
                effect: ToolCallEffect::Mutation,
                mutation_effects: ToolMutationEffects::REMOTE_MUTATION,
                target_hints: vec![ToolTargetHint {
                    kind: ToolTargetHintKind::Url,
                    value: invalid_value.to_string(),
                }],
                ..ToolCallSemantics::default()
            };
            assert_eq!(
                denial(decide(&mandate, &cycle, "http_request", "{}", &semantics,)),
                MandateAuthorityDenial::TargetNotAllowed,
                "target {invalid_value:?} must fail closed"
            );
        }
    }

    #[test]
    fn final_dispatch_revalidation_accepts_reserved_mutation_at_cap() {
        let mandate = mandate();
        let mut authorization_cycle = cycle(&mandate);
        authorization_cycle.action_attempts = 1;
        let semantics = post_semantics();
        let arguments = r#"{"url":"https://api.x.com/2/tweets","method":"POST"}"#;
        let grant = decide(
            &mandate,
            &authorization_cycle,
            "http_request",
            arguments,
            &semantics,
        )
        .grant()
        .cloned()
        .expect("the second and final mutation should be authorizable");

        let mut reserved_cycle = authorization_cycle;
        reserved_cycle.action_attempts = 2;
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &reserved_cycle,
                "http_request",
                arguments,
                &semantics,
                &now(),
                &grant,
            ),
            Ok(())
        );
    }

    #[test]
    fn final_dispatch_requires_one_valid_reserved_mutation_attempt() {
        let mandate = mandate();
        let authorization_cycle = cycle(&mandate);
        let semantics = post_semantics();
        let arguments = r#"{"url":"https://api.x.com/2/tweets","method":"POST"}"#;
        let grant = decide(
            &mandate,
            &authorization_cycle,
            "http_request",
            arguments,
            &semantics,
        )
        .grant()
        .cloned()
        .unwrap();

        for invalid_attempts in [0, 3] {
            let mut stored_cycle = authorization_cycle.clone();
            stored_cycle.action_attempts = invalid_attempts;
            assert_eq!(
                validate_mandate_grant(
                    &mandate,
                    &stored_cycle,
                    "http_request",
                    arguments,
                    &semantics,
                    &now(),
                    &grant,
                ),
                Err(MandateAuthorityDenial::InvalidCycleBudget),
                "stored mutation count {invalid_attempts} must fail closed"
            );
        }
    }

    #[test]
    fn final_dispatch_compares_every_grant_field_and_exact_action() {
        let mandate = mandate();
        let authorization_cycle = cycle(&mandate);
        let semantics = post_semantics();
        let arguments = r#"{"url":"https://api.x.com/2/tweets","method":"POST"}"#;
        let grant = decide(
            &mandate,
            &authorization_cycle,
            "http_request",
            arguments,
            &semantics,
        )
        .grant()
        .cloned()
        .unwrap();
        let mut reserved_cycle = authorization_cycle;
        reserved_cycle.action_attempts = 1;

        let mut altered_grants = Vec::new();
        let mut altered = grant.clone();
        altered.mandate_id = "other-mandate".to_string();
        altered_grants.push(altered);
        let mut altered = grant.clone();
        altered.mandate_version += 1;
        altered_grants.push(altered);
        let mut altered = grant.clone();
        altered.decision_cycle_id = "other-cycle".to_string();
        altered_grants.push(altered);
        let mut altered = grant.clone();
        altered.action_digest = "0".repeat(64);
        altered_grants.push(altered);
        let mut altered = grant.clone();
        altered.counts_toward_cycle_budget = false;
        altered_grants.push(altered);

        for altered in altered_grants {
            assert_eq!(
                validate_mandate_grant(
                    &mandate,
                    &reserved_cycle,
                    "http_request",
                    arguments,
                    &semantics,
                    &now(),
                    &altered,
                ),
                Err(MandateAuthorityDenial::GrantMismatch)
            );
        }

        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &reserved_cycle,
                "http_request",
                r#"{"url":"https://api.x.com/2/tweets","method":"POST","body":"changed"}"#,
                &semantics,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::GrantMismatch)
        );
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &reserved_cycle,
                "mcp__x__post",
                arguments,
                &semantics,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::ToolNotAllowed)
        );

        let changed_effect = ToolCallSemantics::mutation_with(ToolMutationEffects::REMOTE_MUTATION)
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/tweets");
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &reserved_cycle,
                "http_request",
                arguments,
                &changed_effect,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::GrantMismatch)
        );

        let changed_target = ToolCallSemantics::mutation_with(
            ToolMutationEffects::REMOTE_MUTATION.union(ToolMutationEffects::EXTERNAL_DELIVERY),
        )
        .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me");
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &reserved_cycle,
                "http_request",
                arguments,
                &changed_target,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::GrantMismatch)
        );
    }

    #[test]
    fn final_dispatch_rechecks_current_mandate_and_act_cycle() {
        let mandate = mandate();
        let authorization_cycle = cycle(&mandate);
        let semantics = post_semantics();
        let arguments = r#"{"url":"https://api.x.com/2/tweets"}"#;
        let grant = decide(
            &mandate,
            &authorization_cycle,
            "http_request",
            arguments,
            &semantics,
        )
        .grant()
        .cloned()
        .unwrap();
        let mut reserved_cycle = authorization_cycle;
        reserved_cycle.action_attempts = 1;

        let mut paused = mandate.clone();
        paused.status = "paused".to_string();
        assert_eq!(
            validate_mandate_grant(
                &paused,
                &reserved_cycle,
                "http_request",
                arguments,
                &semantics,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::MandateInactive)
        );

        let mut non_act = reserved_cycle.clone();
        non_act.outcome = MandateDecisionOutcome::Wait;
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &non_act,
                "http_request",
                arguments,
                &semantics,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::DecisionCycleNotAct)
        );

        let mut stale = reserved_cycle;
        stale.mandate_version -= 1;
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &stale,
                "http_request",
                arguments,
                &semantics,
                &now(),
                &grant,
            ),
            Err(MandateAuthorityDenial::MandateVersionMismatch)
        );
    }

    #[test]
    fn observation_grant_revalidates_without_budget_reservation() {
        let mandate = mandate();
        let mut cycle = cycle(&mandate);
        cycle.action_attempts = 2;
        let semantics = ToolCallSemantics::observation()
            .with_target_hint(ToolTargetHintKind::Url, "https://api.x.com/2/users/me");
        let grant = decide(&mandate, &cycle, "web_fetch", "{}", &semantics)
            .grant()
            .cloned()
            .unwrap();
        assert!(!grant.counts_toward_cycle_budget);
        assert_eq!(
            validate_mandate_grant(
                &mandate,
                &cycle,
                "web_fetch",
                "{}",
                &semantics,
                &now(),
                &grant,
            ),
            Ok(())
        );
    }

    #[test]
    fn denial_codes_are_stable_machine_values() {
        assert_eq!(
            MandateAuthorityDenial::CycleBudgetExhausted.as_str(),
            "cycle_budget_exhausted"
        );
        assert_eq!(
            MandateAuthorityDenial::MutationEffectNotAllowed.as_str(),
            "mutation_effect_not_allowed"
        );
        assert_eq!(
            MandateAuthorityDenial::GrantMismatch.as_str(),
            "grant_mismatch"
        );
    }
}
