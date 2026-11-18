//! The typed projections of a `track_requirements` call's arguments: the
//! executor's expectation declaration and the evidence targets, shared by
//! the tool and the turn engine.

use serde_json::Value;

use aidaemon_events::{ExecutorExpectationItem, ExecutorExpectationsDeclaredData};

use aidaemon_events::dialogue::RequestObservationTarget;

use crate::traits::ToolMutationEffects;

/// Item-level key under which canonicalization records resource-ID targets
/// that no registered tool advertises. Kept on the canonical arguments so the
/// persisted checklist and the durable expectation event see the same drop.
pub const UNBOUND_TARGETS_KEY: &str = "_unbound_targets";

pub fn unbound_targets_of(item: &Value) -> Vec<String> {
    item.get(UNBOUND_TARGETS_KEY)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Project a `track_requirements` call into the executor's typed expectation
/// declaration. Only typed content survives: effects, the observation flag,
/// exact targets, the recorded unbound targets, and the declared status.
pub fn executor_expectations_from_checklist_arguments(
    task_id: &str,
    arguments: &str,
) -> Option<ExecutorExpectationsDeclaredData> {
    let value: Value = serde_json::from_str(arguments).ok()?;
    let raw_items = value.get("items")?.as_array()?;
    let mut items = Vec::with_capacity(raw_items.len());
    for (index, item) in raw_items.iter().enumerate() {
        let description = item.get("text")?.as_str()?.to_string();
        let mut mutation_effects = ToolMutationEffects::NONE;
        if let Some(effects) = item.get("mutation_effects").and_then(Value::as_array) {
            for effect in effects.iter().filter_map(Value::as_str) {
                if let Some(effect) = ToolMutationEffects::from_protocol_name(effect) {
                    mutation_effects = mutation_effects.union(effect);
                }
            }
        }
        let (targets, observation_targets) = parse_expectation_targets(item.get("targets")).ok()?;
        items.push(ExecutorExpectationItem {
            index,
            description,
            requires_observation: item
                .get("requires_observation")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            mutation_effects,
            targets,
            observation_targets,
            unbound_targets: unbound_targets_of(item),
            depends_on: indices(item.get("depends_on")),
            owner_units: indices(item.get("owner_units")),
            status: item
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("pending")
                .to_string(),
        });
    }
    (!items.is_empty()).then(|| ExecutorExpectationsDeclaredData {
        schema_version: ExecutorExpectationsDeclaredData::SCHEMA_VERSION,
        task_id: task_id.to_string(),
        items,
        outcome: value
            .get("outcome")
            .and_then(Value::as_str)
            .and_then(aidaemon_events::ExecutorClaimedOutcome::from_protocol_name),
        short_of: value
            .get("short_of")
            .and_then(Value::as_str)
            .and_then(aidaemon_events::ShortResource::from_protocol_name),
    })
}

/// A list of indices (`depends_on`, `owner_units`); anything else is none.
fn indices(value: Option<&Value>) -> Vec<usize> {
    value
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_u64().and_then(|n| usize::try_from(n).ok()))
                .collect()
        })
        .unwrap_or_default()
}

/// Decode checklist evidence targets once for both persistence and the durable
/// executor-expectation event. String targets retain the original path/URL
/// behavior; every other string is an exact opaque resource ID and therefore
/// cannot silently degrade into an untargeted observation.
pub fn parse_expectation_targets(
    raw_targets: Option<&Value>,
) -> Result<(Vec<String>, Vec<RequestObservationTarget>), String> {
    let Some(raw_targets) = raw_targets else {
        return Ok((Vec::new(), Vec::new()));
    };
    let values = raw_targets
        .as_array()
        .ok_or_else(|| "targets must be an array".to_string())?;
    let mut legacy_targets = Vec::with_capacity(values.len());
    let mut observation_targets = Vec::with_capacity(values.len());
    for value in values {
        let target = parse_expectation_target(value)?;
        legacy_targets.push(target.subject.value.clone());
        observation_targets.push(target);
    }
    Ok((legacy_targets, observation_targets))
}

pub fn parse_expectation_target(value: &Value) -> Result<RequestObservationTarget, String> {
    let target = if let Some(value) = value.as_str() {
        RequestObservationTarget::from_legacy_exact(value)
            .ok_or_else(|| "target strings must not be empty".to_string())?
    } else {
        serde_json::from_value::<RequestObservationTarget>(value.clone())
            .map_err(|error| format!("invalid structured target: {error}"))?
    };
    if target.subject.value.trim().is_empty() {
        return Err("target subject must not be empty".to_string());
    }
    if let Some(coverage) = target.collection_coverage.as_ref() {
        if coverage.collection.value.trim().is_empty() {
            return Err("collection target must not be empty".to_string());
        }
        if target.facets.is_empty() {
            return Err(
                "collection-backed targets must declare at least one semantic facet".to_string(),
            );
        }
    }
    Ok(target)
}
