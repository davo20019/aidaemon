//! What a store hands out is a `Secret`: it never prints itself, and the
//! kernel redacts it from everything a run sends out.

use aidaemon_secrets::{MemorySecretStore, Secret, SecretStore, SecretTooShort};

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
    assert!(secret.is_redacted());
    assert_eq!(
        aidaemon_core::redaction::redact_registered("Authorization: tok_live_store_0a1b2c3d4e5f"),
        "Authorization: [REDACTED:secret]"
    );
}

/// A secret too short to redact is not built and registered quietly: the
/// constructor refuses it, and only the explicit unredacted path wraps it.
#[test]
fn a_secret_too_short_to_redact_is_refused() {
    assert_eq!(Secret::new("1234"), Err(SecretTooShort { chars: 4 }));
    assert!(Secret::new("long-enough-value").unwrap().is_redacted());
    let plain = Secret::unredacted("1234");
    assert!(!plain.is_redacted());
    assert_eq!(format!("{plain:?}"), "Secret([REDACTED])");
}

/// A short value a store already holds (a client id kept beside its
/// secret) still comes out, marked as not redacted.
#[test]
fn a_short_stored_value_comes_out_marked_unredacted() {
    let store = MemorySecretStore::new();
    store.set("oauth_client_id", "id42").unwrap();
    let secret = store.get("oauth_client_id").unwrap();
    assert_eq!(secret.expose(), "id42");
    assert!(!secret.is_redacted());
}
