//! A tool call as the owner's apps show it: its arguments and its result,
//! secrets redacted and trimmed. One shape for a call seen live and the same
//! call read back from history, so a step says as much either way.

use serde_json::Value;

/// The most of a call's arguments an app is sent.
pub const CALL_INPUT_CHARS: usize = 600;
/// The most of a call's result an app is sent.
pub const CALL_OUTPUT_CHARS: usize = 1_500;

/// A call's arguments: a JSON string read as JSON, empty for none, secrets
/// redacted, pretty-printed and trimmed.
pub fn call_input(arguments: &Value) -> String {
    let mut value = match arguments {
        Value::String(text) => serde_json::from_str(text).unwrap_or_else(|_| arguments.clone()),
        other => other.clone(),
    };
    if value.as_object().is_some_and(|o| o.is_empty()) || value.is_null() {
        return String::new();
    }
    crate::sanitize::redact_json_for_owner(&mut value);
    let text = match &value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    };
    crate::text::truncate_with_note(&text, CALL_INPUT_CHARS)
}

/// A call's result: without the untrusted-data frame the model saw it in,
/// secrets redacted, trimmed.
pub fn call_output(content: &str) -> String {
    let mut text = content.trim();
    // Machine marker: the frame the daemon itself writes around tool output.
    if let Some(rest) = text.strip_prefix(aidaemon_policies::UNTRUSTED_PREFIX)
        && let Some(rest) = rest
            .trim_end()
            .strip_suffix(aidaemon_policies::UNTRUSTED_SUFFIX)
    {
        // The frame's first line names the tool; the data follows it.
        text = rest.split_once('\n').map_or("", |(_, data)| data).trim();
    }
    let text = crate::sanitize::redact_for_owner(text);
    crate::text::truncate_with_note(&text, CALL_OUTPUT_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_and_results_are_shown_alike() {
        assert_eq!(call_input(&serde_json::json!({})), "");
        assert!(
            call_input(&serde_json::json!("{\"command\":\"ls -la\"}"))
                .contains("\"command\": \"ls -la\"")
        );
        let framed = format!(
            "{}terminal]\nhello\n{}",
            aidaemon_policies::UNTRUSTED_PREFIX,
            aidaemon_policies::UNTRUSTED_SUFFIX
        );
        assert_eq!(call_output(&framed), "hello");
    }
}
