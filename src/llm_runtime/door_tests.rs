//! The side-call gateway is the only door to a model outside the agent loop.
//! These tests scan the source for code that reaches a provider, or records a
//! model call, without it.

use std::path::Path;

/// Files allowed to reach a provider without the gateway, each with its reason.
const ALLOWED: &[(&str, &str)] = &[
    (
        "crates/aidaemon-runtime/src/llm/",
        "defines the runtime and the gateway",
    ),
    (
        "crates/aidaemon-runtime/src/chat_bridge/",
        "calls providers",
    ),
    ("src/llm_runtime/door_tests.rs", "these tests"),
    ("src/providers/", "builds providers"),
    (
        "src/agent/runtime/sdk_provider.rs",
        "the main loop's provider",
    ),
    (
        "src/agent/runtime/sdk_engine/",
        "the main loop, including its own compaction",
    ),
    ("src/agent/runtime/llm.rs", "the main loop's failover"),
];
const RAW_ACCESS: &[&str] = &[
    "snapshot.provider()",
    "snapshot().provider()",
    "record_background_model_call_telemetry(",
    "record_background_model_call_failure(",
];

/// Every `.rs` file under `dir` (relative to the crate root), as a path
/// relative to the crate root with `/` separators.
fn rust_files_under(dir: &str) -> impl Iterator<Item = String> + use<> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut pending = vec![root.join(dir)];
    let mut files = Vec::new();
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let relative = path.strip_prefix(root).unwrap();
                let parts: Vec<_> = relative
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect();
                files.push(parts.join("/"));
            }
        }
    }
    files.sort();
    files.into_iter()
}

/// The daemon's code and the runtime crate's.
const SCANNED: &[&str] = &["src", "crates/aidaemon-runtime/src"];

fn scanned_files() -> impl Iterator<Item = String> {
    SCANNED.iter().flat_map(|dir| rust_files_under(dir))
}

fn reaches_a_provider(path: &str) -> bool {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap();
    RAW_ACCESS.iter().any(|needle| text.contains(needle))
}

#[test]
fn nothing_outside_the_gateway_reaches_a_provider() {
    let offenders = scanned_files()
        .filter(|path| !ALLOWED.iter().any(|(prefix, _)| path.starts_with(prefix)))
        .filter(|path| reaches_a_provider(path))
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "call the model through SideCallGateway: {offenders:?}"
    );
}

/// The scan sees the gateway's own raw access, so an empty offender list
/// means the files were read, not skipped.
#[test]
fn the_scan_sees_the_gateway() {
    assert!(scanned_files()
        .any(|path| path == "crates/aidaemon-runtime/src/llm/gateway.rs"
            && reaches_a_provider(&path)));
}

/// What reaches a decision backend directly. Callers ask through
/// `SideCallGateway::decide`, which applies modes, deadlines, thresholds
/// and telemetry.
const RAW_DECIDER: &[&str] = &[
    "GeminiDecider",
    "ProviderDecider",
    "dyn Decider",
    "impl Decider",
    "Decider::decide",
];

#[test]
fn nothing_outside_the_gateway_holds_a_decider() {
    let offenders = scanned_files()
        .filter(|path| !path.starts_with("crates/aidaemon-runtime/src/llm/"))
        .filter(|path| path != "src/llm_runtime/door_tests.rs")
        .filter(|path| {
            let text =
                std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap();
            RAW_DECIDER.iter().any(|needle| text.contains(needle))
        })
        .collect::<Vec<_>>();
    assert!(
        offenders.is_empty(),
        "ask decisions through SideCallGateway::decide: {offenders:?}"
    );
}
