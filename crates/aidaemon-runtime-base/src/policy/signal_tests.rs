use super::*;

#[test]
fn policy_bootstrap_does_not_infer_risk_from_request_wording() {
    let caps = std::collections::HashMap::new();
    let direct = build_policy_bundle("please deploy to production", &caps, false).risk_score;
    let morph =
        build_policy_bundle("the deployed artifact is production-like", &caps, false).risk_score;
    assert_eq!(direct, morph);
}

#[test]
fn policy_bootstrap_does_not_infer_destructive_intent_from_prose() {
    let caps = std::collections::HashMap::new();
    let destructive = build_policy_bundle("overwrite the file now", &caps, false).risk_score;
    let descriptive = build_policy_bundle("this is an overwritten file", &caps, false).risk_score;
    assert_eq!(destructive, descriptive);
}
