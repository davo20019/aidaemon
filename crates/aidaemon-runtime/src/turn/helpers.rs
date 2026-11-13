//! Small helpers the turn's handlers share: status sends, the heartbeat,
//! bounded text, and the provider tool array's canonical order.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aidaemon_tools::status::StatusUpdate;
use serde_json::Value;
use tokio::sync::mpsc;

/// Best-effort send — never blocks the agent loop if the receiver is slow/full.
pub fn send_status(tx: &Option<mpsc::Sender<StatusUpdate>>, update: StatusUpdate) {
    if let Some(ref tx) = tx {
        let _ = tx.try_send(update);
    }
}

/// Update the heartbeat timestamp to signal the agent is alive.
/// No-op when heartbeat is None (sub-agents, triggers, tests).
pub fn touch_heartbeat(hb: &Option<Arc<AtomicU64>>) {
    if let Some(ref hb) = hb {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        hb.store(now, Ordering::Relaxed);
    }
}

/// The first `max_chars` characters of `text`, with `...` when cut.
pub fn truncate_for_resume(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (count, ch) in text.chars().enumerate() {
        if count >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

/// The tool name of an OpenAI-style function definition.
pub fn tool_name_from_definition(def: &Value) -> Option<&str> {
    def.get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
}

/// Sort tool definitions in place by tool name, using serialized definition
/// bytes as a deterministic tie-breaker, so the provider tool array has a
/// stable order.
///
/// Sort keys (name, serialized bytes) are computed ONCE per element so
/// serialization cost is O(n), not O(n log n).
pub fn sort_tool_definitions_by_name(defs: &mut Vec<Value>) {
    // Compute each element's sort key exactly once, then sort by key.
    let mut keyed: Vec<(String, String, Value)> = defs
        .drain(..)
        .map(|d| {
            let name = tool_name_from_definition(&d).unwrap_or("").to_string();
            let serialized = d.to_string();
            (name, serialized, d)
        })
        .collect();
    keyed.sort_by(|(an, as_, _), (bn, bs, _)| an.cmp(bn).then_with(|| as_.cmp(bs)));
    defs.extend(keyed.into_iter().map(|(_, _, d)| d));
}

/// Extract a JSON object from LLM output, handling code fences and preamble text.
/// Tries direct parse first, then falls back to finding `{...}` bounds.
pub fn extract_json_object(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let candidate = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```JSON")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else {
        trimmed.to_string()
    };
    if serde_json::from_str::<serde_json::Value>(&candidate)
        .ok()
        .is_some_and(|v| v.is_object())
    {
        return Some(candidate);
    }

    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let sliced = raw[start..=end].trim().to_string();
    if serde_json::from_str::<serde_json::Value>(&sliced)
        .ok()
        .is_some_and(|v| v.is_object())
    {
        Some(sliced)
    } else {
        None
    }
}
