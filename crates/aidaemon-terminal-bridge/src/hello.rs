//! Reading a client hello: agent, arguments, session ids and launch flags.

use super::*;

pub(crate) fn normalize_agent(value: Option<&str>) -> Option<String> {
    let v = value.unwrap_or("").trim().to_ascii_lowercase();
    if SUPPORTED_TERMINAL_AGENTS.contains(&v.as_str()) {
        Some(v)
    } else {
        None
    }
}

pub(crate) fn normalize_relay_session_id(raw: Option<&str>) -> Option<String> {
    let value = raw.unwrap_or("").trim();
    if value.is_empty() || value.len() > 128 {
        return None;
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.'))
    {
        return None;
    }
    Some(value.to_string())
}

pub(crate) fn normalize_agent_args(value: Option<&Value>, agent: Option<&str>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    let mut raw_values: Vec<String> = Vec::new();
    match value {
        Value::Array(raw) => {
            for item in raw {
                if let Some(text) = item.as_str() {
                    raw_values.push(text.to_string());
                }
            }
        }
        Value::String(text) => {
            if let Ok(parsed) = shell_words::split(text) {
                raw_values.extend(parsed);
            } else {
                raw_values.push(text.clone());
            }
        }
        _ => return Vec::new(),
    }
    let out = raw_values
        .into_iter()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty() && !v.contains('\0'))
        .filter(|v| is_safe_agent_bootstrap_arg(v))
        .map(|v| v.chars().take(MAX_AGENT_ARG_CHARS).collect::<String>())
        .take(MAX_AGENT_ARGS)
        .collect();
    let (out, _) =
        aidaemon_tools::permission_bypass::normalize_terminal_agent_permission_aliases(agent, out);
    out
}

pub(crate) fn requested_agent_args_from_client_hello(frame: &Value) -> Vec<String> {
    // Accept multiple key shapes to remain compatible with older/newer Mini App builds.
    let requested_agent = normalize_agent(frame.get("agent").and_then(|v| v.as_str()));
    let key_order = ["agent_args", "agentArgs", "args", "argv", "arg"];
    for key in key_order {
        if let Some(raw) = frame.get(key) {
            return normalize_agent_args(Some(raw), requested_agent.as_deref());
        }
    }
    Vec::new()
}

pub(crate) fn normalize_telegram_session_id(raw: Option<&str>) -> Option<String> {
    let session_id = raw?.trim();
    if session_id.is_empty() || session_id.len() > 128 {
        return None;
    }
    if !session_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_' | '.'))
    {
        return None;
    }
    Some(session_id.to_string())
}

pub(crate) fn requested_telegram_session_id_from_client_hello(frame: &Value) -> Option<String> {
    let key_order = [
        "telegram_session_id",
        "telegramSessionId",
        "chat_session_id",
        "chatSessionId",
    ];
    for key in key_order {
        if let Some(value) = normalize_telegram_session_id(frame.get(key).and_then(Value::as_str)) {
            return Some(value);
        }
    }
    if let Some(telegram_obj) = frame.get("telegram").and_then(Value::as_object) {
        for key in key_order {
            if let Some(value) =
                normalize_telegram_session_id(telegram_obj.get(key).and_then(Value::as_str))
            {
                return Some(value);
            }
        }
    }
    None
}

pub(crate) fn requested_relay_session_id_from_client_hello(frame: &Value) -> Option<String> {
    let key_order = [
        "relay_session_id",
        "relaySessionId",
        "requested_relay_session_id",
        "requestedRelaySessionId",
        "target_relay_session_id",
        "targetRelaySessionId",
    ];
    for key in key_order {
        if let Some(value) = normalize_relay_session_id(frame.get(key).and_then(Value::as_str)) {
            return Some(value);
        }
    }
    if let Some(relay_obj) = frame.get("relay").and_then(Value::as_object) {
        for key in key_order {
            if let Some(value) =
                normalize_relay_session_id(relay_obj.get(key).and_then(Value::as_str))
            {
                return Some(value);
            }
        }
        if let Some(value) =
            normalize_relay_session_id(relay_obj.get("session_id").and_then(Value::as_str))
        {
            return Some(value);
        }
        if let Some(value) =
            normalize_relay_session_id(relay_obj.get("sessionId").and_then(Value::as_str))
        {
            return Some(value);
        }
    }
    None
}

pub(crate) fn parse_boolish_client_hello_value(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(v) => Some(*v),
        Value::Number(n) => n
            .as_u64()
            .map(|v| v != 0)
            .or_else(|| n.as_i64().map(|v| v != 0)),
        Value::String(text) => {
            let normalized = text.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                return None;
            }
            match normalized.as_str() {
                "1" | "true" | "yes" | "y" | "on" => Some(true),
                "0" | "false" | "no" | "n" | "off" => Some(false),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(crate) fn requested_fresh_start_from_client_hello(frame: &Value) -> bool {
    let key_order = [
        "fresh_start",
        "freshStart",
        "force_fresh_start",
        "forceFreshStart",
        "start_fresh",
        "startFresh",
    ];
    for key in key_order {
        if let Some(raw) = frame.get(key)
            && let Some(value) = parse_boolish_client_hello_value(raw)
        {
            return value;
        }
    }
    if let Some(launch_obj) = frame.get("launch").and_then(Value::as_object) {
        for key in key_order {
            if let Some(raw) = launch_obj.get(key)
                && let Some(value) = parse_boolish_client_hello_value(raw)
            {
                return value;
            }
        }
    }
    false
}

pub(crate) fn is_safe_agent_bootstrap_arg(value: &str) -> bool {
    if value.contains("$(") || value.contains('\n') || value.contains('\r') {
        return false;
    }
    !value
        .chars()
        .any(|ch| matches!(ch, ';' | '|' | '&' | '>' | '<' | '`'))
}

pub(crate) fn should_quote_shell_token(value: &str) -> bool {
    value.is_empty()
        || value.chars().any(|ch| {
            !(ch.is_ascii_alphanumeric()
                || matches!(ch, '_' | '-' | '.' | '/' | ':' | '=' | '+' | ',' | '@'))
        })
}

pub(crate) fn shell_token(value: &str) -> String {
    if !should_quote_shell_token(value) {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
}

pub(crate) fn resolve_session_cwd(raw: Option<&str>, default_cwd: &Path) -> PathBuf {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() || raw == "~" {
        return dirs::home_dir().unwrap_or_else(|| default_cwd.to_path_buf());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| default_cwd.join(rest));
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return path;
    }
    default_cwd.join(path)
}

pub(crate) fn resolve_terminal_inbox_dir(raw: &str, default_cwd: &Path) -> PathBuf {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return default_cwd.join(".aidaemon/files/inbox");
    }
    if trimmed == "~" {
        return dirs::home_dir()
            .unwrap_or_else(|| default_cwd.to_path_buf())
            .join(".aidaemon/files/inbox");
    }
    if let Some(rest) = trimmed.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| default_cwd.join(rest));
    }
    let p = PathBuf::from(trimmed);
    if p.is_absolute() {
        p
    } else {
        default_cwd.join(p)
    }
}
