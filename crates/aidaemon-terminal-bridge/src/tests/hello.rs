//! Client hello parsing: agent arguments, session ids, fresh start.

use super::*;

#[test]
fn test_normalize_agent_args_drops_shell_metacharacters() {
    let payload = serde_json::json!(["--model", "gpt-5", "foo;bar", "$(whoami)", "--json"]);
    let args = normalize_agent_args(Some(&payload), Some("codex"));
    assert_eq!(args, vec!["--model", "gpt-5", "--json"]);
}

#[test]
fn test_normalize_agent_args_splits_single_string_payload() {
    let payload = serde_json::json!("--model gpt-5 --json");
    let args = normalize_agent_args(Some(&payload), Some("codex"));
    assert_eq!(args, vec!["--model", "gpt-5", "--json"]);
}

#[test]
fn test_requested_agent_args_from_client_hello_rewrites_claude_allow_alias() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "agent": "claude",
        "agent_args": ["--allow-dangerously-skip-permissions", "--output-format", "json"],
    });
    let args = requested_agent_args_from_client_hello(&frame);
    assert_eq!(
        args,
        vec!["--dangerously-skip-permissions", "--output-format", "json"]
    );
}

#[test]
fn test_shell_token_keeps_plain_flags_unquoted() {
    assert_eq!(shell_token("codex"), "codex");
    assert_eq!(
        shell_token("--dangerously-bypass-approvals-and-sandbox"),
        "--dangerously-bypass-approvals-and-sandbox"
    );
}

#[test]
fn test_shell_token_quotes_tokens_with_whitespace_or_quotes() {
    assert_eq!(shell_token("--model gpt-5"), "'--model gpt-5'");
    assert_eq!(shell_token("it's"), "'it'\\''s'");
}

#[test]
fn test_requested_agent_args_from_client_hello_accepts_legacy_args_key() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "agent": "codex",
        "args": ["--model", "gpt-5", "bad;arg"],
    });
    let args = requested_agent_args_from_client_hello(&frame);
    assert_eq!(args, vec!["--model", "gpt-5"]);
}

#[test]
fn test_requested_agent_args_from_client_hello_accepts_camel_case_key() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "agent": "codex",
        "agentArgs": ["--json"],
    });
    let args = requested_agent_args_from_client_hello(&frame);
    assert_eq!(args, vec!["--json"]);
}

#[test]
fn test_requested_telegram_session_id_from_client_hello_accepts_snake_case() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "telegram_session_id": "telegrambot:12345",
    });
    let value = requested_telegram_session_id_from_client_hello(&frame);
    assert_eq!(value.as_deref(), Some("telegrambot:12345"));
}

#[test]
fn test_requested_telegram_session_id_from_client_hello_accepts_camel_case() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "telegramSessionId": "telegrambot:999",
    });
    let value = requested_telegram_session_id_from_client_hello(&frame);
    assert_eq!(value.as_deref(), Some("telegrambot:999"));
}

#[test]
fn test_requested_telegram_session_id_from_client_hello_rejects_invalid_chars() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "telegram_session_id": "telegrambot:12345/../../etc/passwd",
    });
    assert!(requested_telegram_session_id_from_client_hello(&frame).is_none());
}

#[test]
fn test_requested_relay_session_id_from_client_hello_accepts_snake_case() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "relay_session_id": "native-1234abcd",
    });
    let value = requested_relay_session_id_from_client_hello(&frame);
    assert_eq!(value.as_deref(), Some("native-1234abcd"));
}

#[test]
fn test_requested_relay_session_id_from_client_hello_accepts_camel_case() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "relaySessionId": "native-5678efgh",
    });
    let value = requested_relay_session_id_from_client_hello(&frame);
    assert_eq!(value.as_deref(), Some("native-5678efgh"));
}

#[test]
fn test_requested_relay_session_id_from_client_hello_accepts_nested_relay() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "relay": {
            "session_id": "native-zxyw9876"
        },
    });
    let value = requested_relay_session_id_from_client_hello(&frame);
    assert_eq!(value.as_deref(), Some("native-zxyw9876"));
}

#[test]
fn test_requested_relay_session_id_from_client_hello_rejects_invalid_chars() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "relay_session_id": "native-1234/../../etc/passwd",
    });
    assert!(requested_relay_session_id_from_client_hello(&frame).is_none());
}

#[test]
fn test_requested_fresh_start_from_client_hello_accepts_boolean_flag() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "fresh_start": true,
    });
    assert!(requested_fresh_start_from_client_hello(&frame));
}

#[test]
fn test_requested_fresh_start_from_client_hello_accepts_string_flag() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "freshStart": "yes",
    });
    assert!(requested_fresh_start_from_client_hello(&frame));
}

#[test]
fn test_requested_fresh_start_from_client_hello_accepts_nested_launch_flag() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "launch": {
            "force_fresh_start": 1
        }
    });
    assert!(requested_fresh_start_from_client_hello(&frame));
}

#[test]
fn test_requested_fresh_start_from_client_hello_defaults_false() {
    let frame = serde_json::json!({
        "type": "e2ee_client_hello",
        "fresh_start": "maybe",
    });
    assert!(!requested_fresh_start_from_client_hello(&frame));
}
