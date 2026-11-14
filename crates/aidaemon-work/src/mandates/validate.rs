//! Bounds and normalization for the text and lists a mandate tool call
//! carries. The owner tool (`manage_mandates`) and the review tool
//! (`mandate_review`) both validate through these, so a limit is stated once.

use crate::traits::{MandateDecisionOutcome, MandateTerminationKind};

/// Longest single policy entry (constraint, criterion, stop condition).
pub const MAX_POLICY_ENTRY_TEXT: usize = 500;
/// Longest learning note a review records.
pub const MAX_LEARNING_NOTE_TEXT: usize = 1024;
/// Most evidence receipts one decision cites.
pub const MAX_EVIDENCE_RECEIPTS: usize = 16;

pub fn required_trimmed<'a>(value: Option<&'a str>, name: &str) -> anyhow::Result<&'a str> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{name} is required"))
}

/// A dispatcher fact the call needs (from its `ToolCallContext`).
pub fn required_call_fact<'a>(value: Option<&'a str>, fact: &str) -> anyhow::Result<&'a str> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("the {fact} fact is missing from the call context"))
}

pub fn required_bounded_trimmed<'a>(
    value: Option<&'a str>,
    name: &str,
    max_text: usize,
) -> anyhow::Result<&'a str> {
    let value = required_trimmed(value, name)?;
    validate_bounded_text(value, name, max_text)?;
    Ok(value)
}

pub fn validate_bounded_text(value: &str, name: &str, max_text: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.chars().count() <= max_text && value.len() <= max_text,
        "{name} exceeds its {max_text}-character/byte bound"
    );
    Ok(())
}

pub fn bounded_optional_text(
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

pub fn normalized_termination_fields(
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

pub fn validate_act_evidence(
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

pub fn normalized_learning_note(
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

pub fn learning_note_fields_complete(
    learning_note: Option<&str>,
    evidence: Option<&[String]>,
) -> bool {
    learning_note.is_some_and(|value| !value.trim().is_empty())
        && evidence.is_some_and(|values| values.iter().any(|value| !value.trim().is_empty()))
}

pub fn clean_strings(values: Option<&[String]>) -> Vec<String> {
    values
        .unwrap_or_default()
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

pub fn validate_bounded_strings(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_strings_removes_empty_constraints() {
        let values = vec!["  truthful ".to_string(), " ".to_string()];
        assert_eq!(clean_strings(Some(&values)), vec!["truthful"]);
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
        assert!(
            normalized_learning_note(Some(""), Some(&receipts))
                .unwrap()
                .is_none()
        );
        assert!(normalized_learning_note(Some("real learning"), None).is_err());
    }

    #[test]
    fn partial_optional_learning_metadata_is_not_decision_blocking() {
        assert!(!learning_note_fields_complete(
            Some("possible learning"),
            None
        ));
    }
}
