use super::*;

#[test]
fn detects_explicit_positive_signals_only() {
    let detected = detect_explicit_outcome_signal("thanks, that worked");
    assert_eq!(detected, Some(("positive", true)));
}

#[test]
fn detects_explicit_negative_signals_only() {
    let detected = detect_explicit_outcome_signal("you misunderstood");
    assert_eq!(detected, Some(("negative", false)));
}

#[test]
fn ignores_non_explicit_feedback() {
    let detected = detect_explicit_outcome_signal("can you try a different approach");
    assert!(detected.is_none());
}

#[test]
fn risk_scoring_prefers_word_boundary_action_terms() {
    let caps = std::collections::HashMap::new();
    let direct = build_policy_bundle("please deploy to production", &caps, false).risk_score;
    let morph =
        build_policy_bundle("the deployed artifact is production-like", &caps, false).risk_score;
    assert!(
        direct > morph,
        "expected direct action wording to score higher than morphological variants"
    );
}

#[test]
fn risk_scoring_does_not_treat_overwritten_as_overwrite_command() {
    let caps = std::collections::HashMap::new();
    let destructive = build_policy_bundle("overwrite the file now", &caps, false).risk_score;
    let descriptive = build_policy_bundle("this is an overwritten file", &caps, false).risk_score;
    assert!(
        destructive > descriptive,
        "expected imperative overwrite command to score higher than descriptive text"
    );
}
