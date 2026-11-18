//! The checked-in JSON Schema matches the Rust types.

use std::path::PathBuf;

#[test]
fn checked_in_schema_is_current() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/client-protocol/v1.schema.json");
    let on_disk = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "{} is missing: run `cargo run -p aidaemon-client-protocol --bin export-client-schema`",
            path.display()
        )
    });
    let on_disk: serde_json::Value = serde_json::from_str(&on_disk).unwrap();
    assert!(
        on_disk == aidaemon_client_protocol::schema(),
        "{} is stale: run `cargo run -p aidaemon-client-protocol --bin export-client-schema` \
         and commit the result",
        path.display()
    );
}

/// Every frame type is reachable from the schema's root.
#[test]
fn schema_names_every_frame_and_the_http_page() {
    let schema = aidaemon_client_protocol::schema();
    let defs = schema["$defs"].as_object().unwrap();
    for name in [
        "ClientFrame",
        "ServerFrame",
        "Command",
        "CommandResult",
        "Event",
        "Approval",
        "EventsPage",
    ] {
        assert!(defs.contains_key(name), "$defs lacks {name}");
    }
}
