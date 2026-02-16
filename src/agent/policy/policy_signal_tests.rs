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
