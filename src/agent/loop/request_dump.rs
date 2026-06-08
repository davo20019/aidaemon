//! Opt-in debug dump of the exact provider request payload.
//!
//! Gated by the `AIDAEMON_DUMP_LLM_REQUESTS` env var. When enabled, every
//! finalized provider call (the same payload the Phase 0 prefix fingerprint
//! hashes — after security-message injection and force-text tool selection)
//! is written as one pretty-printed JSON file so the composition of the input
//! tokens can be inspected byte-for-byte.
//!
//! Values: unset / empty / `0` / `false` → disabled; `1` / `true` → dump to
//! the default `llm_request_dumps/` directory under the working directory;
//! any other value → treated as the dump directory path.
//!
//! SECURITY NOTE: dumps contain raw conversation content, tool results, and
//! the full system prompt. This is a local debugging aid — never enable it in
//! a deployment where the dump directory is readable by other users.

use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Default dump directory (relative to the working directory) when the env
/// var is a bare truthy value rather than a path.
const DEFAULT_DUMP_DIR: &str = "llm_request_dumps";

/// Process-wide sequence number so concurrent sessions / same-second dumps
/// never collide on file names.
static DUMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Resolve the dump directory from the raw env var value.
///
/// `None` / empty / `"0"` / `"false"` (case-insensitive) → disabled.
/// `"1"` / `"true"` (case-insensitive) → default `llm_request_dumps` dir.
/// Anything else → the value itself as a directory path.
pub(crate) fn dump_dir_from_env(value: Option<&str>) -> Option<PathBuf> {
    let value = value?.trim();
    match value.to_ascii_lowercase().as_str() {
        "" | "0" | "false" => None,
        "1" | "true" => Some(PathBuf::from(DEFAULT_DUMP_DIR)),
        _ => Some(PathBuf::from(value)),
    }
}

/// Write one provider request payload to `dir` as a pretty-printed JSON file.
///
/// Creates `dir` (and parents) if missing. Returns the path of the written
/// file. File names embed a sanitized session id, the iteration, and a
/// process-wide sequence number so concurrent sessions never collide.
pub(crate) fn write_request_dump(
    dir: &Path,
    session_id: &str,
    iteration: usize,
    model: &str,
    messages: &[Value],
    tools: &[Value],
    force_text: bool,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;

    let seq = DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let timestamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ");
    let safe_session: String = session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!(
        "{timestamp}_{seq:06}_{safe_session}_iter{iteration:03}.json"
    ));

    let dump = json!({
        "timestamp": timestamp.to_string(),
        "session_id": session_id,
        "iteration": iteration,
        "model": model,
        "force_text": force_text,
        "message_count": messages.len(),
        "messages": messages,
        "tools": tools,
    });
    std::fs::write(&path, serde_json::to_string_pretty(&dump)?)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn env_unset_or_disabled_returns_none() {
        assert_eq!(dump_dir_from_env(None), None);
        assert_eq!(dump_dir_from_env(Some("")), None);
        assert_eq!(dump_dir_from_env(Some("0")), None);
        assert_eq!(dump_dir_from_env(Some("false")), None);
        assert_eq!(dump_dir_from_env(Some("FALSE")), None);
    }

    #[test]
    fn env_truthy_uses_default_dir() {
        assert_eq!(
            dump_dir_from_env(Some("1")),
            Some(PathBuf::from("llm_request_dumps"))
        );
        assert_eq!(
            dump_dir_from_env(Some("true")),
            Some(PathBuf::from("llm_request_dumps"))
        );
        assert_eq!(
            dump_dir_from_env(Some("TRUE")),
            Some(PathBuf::from("llm_request_dumps"))
        );
    }

    #[test]
    fn env_path_value_used_as_dir() {
        assert_eq!(
            dump_dir_from_env(Some("/tmp/my_dumps")),
            Some(PathBuf::from("/tmp/my_dumps"))
        );
    }

    #[test]
    fn write_dump_round_trips_payload() {
        let dir = tempfile::tempdir().unwrap();
        let messages = vec![
            json!({"role": "system", "content": "sys prompt"}),
            json!({"role": "user", "content": "hi"}),
        ];
        let tools = vec![json!({"name": "terminal", "description": "run commands"})];
        let path = write_request_dump(
            dir.path(),
            "telegram:123",
            2,
            "gpt-test",
            &messages,
            &tools,
            false,
        )
        .unwrap();

        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(parsed["session_id"], "telegram:123");
        assert_eq!(parsed["iteration"], 2);
        assert_eq!(parsed["model"], "gpt-test");
        assert_eq!(parsed["force_text"], false);
        assert_eq!(parsed["message_count"], 2);
        assert_eq!(parsed["messages"], Value::Array(messages));
        assert_eq!(parsed["tools"], Value::Array(tools));
    }

    #[test]
    fn filename_sanitizes_session_id_and_is_unique() {
        let dir = tempfile::tempdir().unwrap();
        let messages = vec![json!({"role": "user", "content": "x"})];
        let p1 =
            write_request_dump(dir.path(), "slack:C1/D2", 1, "m", &messages, &[], false).unwrap();
        let p2 =
            write_request_dump(dir.path(), "slack:C1/D2", 1, "m", &messages, &[], false).unwrap();
        assert_ne!(p1, p2, "consecutive dumps must not collide");
        let fname = p1.file_name().unwrap().to_str().unwrap();
        assert!(
            !fname.contains(':') && !fname.contains('/'),
            "session separators must be sanitized out of the file name: {fname}"
        );
        assert!(fname.contains("slack_C1_D2"), "file name: {fname}");
    }

    #[test]
    fn write_dump_creates_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a").join("b");
        let path = write_request_dump(&nested, "s", 1, "m", &[], &[], true).unwrap();
        assert!(path.exists());
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(parsed["force_text"], true);
    }
}
