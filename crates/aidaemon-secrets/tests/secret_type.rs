//! What a store hands out is a `Secret`: it never prints itself, and the
//! kernel redacts it from everything a run sends out.

use aidaemon_secrets::{MemorySecretStore, Secret, SecretStore};

#[test]
fn a_fetched_secret_does_not_print_and_is_registered_for_redaction() {
    let store = MemorySecretStore::new();
    store
        .set("deploy_token", "tok_live_store_0a1b2c3d4e5f")
        .unwrap();
    let secret: Secret = store.get("deploy_token").unwrap();
    assert_eq!(format!("{secret:?}"), "Secret([REDACTED])");
    let shown = format!("{:?}", store.get("deploy_token"));
    assert!(!shown.contains("tok_live_store"), "{shown}");
    assert_eq!(secret.expose(), "tok_live_store_0a1b2c3d4e5f");
    assert_eq!(
        aidaemon_core::redaction::redact_registered("Authorization: tok_live_store_0a1b2c3d4e5f"),
        "Authorization: [REDACTED:secret]"
    );
}

#[test]
fn a_secret_too_short_to_redact_says_so() {
    let secret = Secret::new("1234");
    assert!(!secret.is_redacted());
    assert!(Secret::new("long-enough-value").is_redacted());
}
