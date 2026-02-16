use serde_json::Value;

use super::{IntentGateDecision, INTENT_GATE_MARKER};

pub(super) fn parse_intent_gate_json(text: &str) -> Option<IntentGateDecision> {
    let value: Value = serde_json::from_str(text).ok()?;
    Some(IntentGateDecision {
        can_answer_now: value.get("can_answer_now").and_then(|v| v.as_bool()),
        needs_tools: value.get("needs_tools").and_then(|v| v.as_bool()),
        needs_clarification: value.get("needs_clarification").and_then(|v| v.as_bool()),
        clarifying_question: value
            .get("clarifying_question")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        missing_info: value
            .get("missing_info")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        complexity: value
            .get("complexity")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty()),
        cancel_intent: value.get("cancel_intent").and_then(|v| v.as_bool()),
        cancel_scope: value
            .get("cancel_scope")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| s == "generic" || s == "targeted"),
        is_acknowledgment: value.get("is_acknowledgment").and_then(|v| v.as_bool()),
        schedule: value
            .get("schedule")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        schedule_type: value
            .get("schedule_type")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty()),
        schedule_cron: value
            .get("schedule_cron")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        domains: value
            .get("domains")
            .and_then(|v| v.as_array())
            .map(|arr| {
                let mut out = Vec::new();
                for item in arr {
                    if let Some(raw) = item.as_str() {
                        let domain = raw.trim().to_ascii_lowercase();
                        if !domain.is_empty() && !out.contains(&domain) {
                            out.push(domain);
                        }
                    }
                }
                out
            })
            .unwrap_or_default(),
    })
}

pub(super) fn extract_intent_gate(text: &str) -> (String, Option<IntentGateDecision>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut cleaned = Vec::with_capacity(lines.len());
    let mut decision: Option<IntentGateDecision> = None;
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i];
        if decision.is_none() {
            if let Some(pos) = line.find(INTENT_GATE_MARKER) {
                let after = line[(pos + INTENT_GATE_MARKER.len())..].trim();
                if !after.is_empty() {
                    decision = parse_intent_gate_json(after);
                } else if i + 1 < lines.len() {
                    let next = lines[i + 1].trim();
                    if next.starts_with('{') {
                        if let Some(parsed) = parse_intent_gate_json(next) {
                            decision = Some(parsed);
                            i += 2;
                            continue;
                        }
                    }
                }
                i += 1;
                continue;
            }
        }
        cleaned.push(line.to_string());
        i += 1;
    }

    // If no [INTENT_GATE] marker was found, check for a trailing JSON block
    // at the end of the response. The model sometimes omits the marker and just
    // appends the JSON (single-line, multi-line, or code-fenced).
    if decision.is_none() {
        decision = try_extract_trailing_intent_json(&mut cleaned);
    }

    (cleaned.join("\n").trim().to_string(), decision)
}

/// Scan backwards from the end of `lines` looking for a trailing JSON object
/// that contains intent gate fields (complexity, can_answer_now, needs_tools).
/// If found, remove those lines from `lines` and return the parsed decision.
fn try_extract_trailing_intent_json(lines: &mut Vec<String>) -> Option<IntentGateDecision> {
    // Find last non-empty line
    let mut end = lines.len();
    while end > 0 && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    if end == 0 {
        return None;
    }

    // Check for code-fence closing: strip trailing ```
    let mut has_closing_fence = false;
    let mut fence_end = end;
    if lines[end - 1].trim() == "```" {
        has_closing_fence = true;
        fence_end = end;
        end -= 1;
        // Skip blanks before the closing fence
        while end > 0 && lines[end - 1].trim().is_empty() {
            end -= 1;
        }
    }

    // Now find the JSON block: look for a line ending with `}` (end of JSON)
    if end == 0 || !lines[end - 1].trim().ends_with('}') {
        return None;
    }

    // Find a parseable trailing JSON object by trying increasingly large suffixes.
    // This avoids naive brace counting (which breaks on `{`/`}` inside JSON strings).
    let json_end = end;
    for json_start in (0..json_end).rev() {
        let first = lines[json_start].trim();
        if first.is_empty() {
            continue;
        }
        // Skip code-fence lines; we'll handle them when removing.
        if first.starts_with("```") {
            continue;
        }
        if !first.starts_with('{') {
            continue;
        }

        let json_text: String = lines[json_start..json_end]
            .iter()
            .map(|l| l.trim())
            .collect::<Vec<_>>()
            .join("");
        let Some(parsed) = parse_intent_gate_json(&json_text) else {
            continue;
        };

        // Only strip if it contains intent gate fields.
        if parsed.complexity.is_none()
            && parsed.can_answer_now.is_none()
            && parsed.needs_tools.is_none()
        {
            continue;
        }

        // Check for opening code fence before the JSON block.
        let mut has_opening_fence = false;
        let mut actual_start = json_start;
        if json_start > 0 {
            let prev = lines[json_start - 1].trim();
            if prev == "```json" || prev == "```JSON" || prev == "```" {
                has_opening_fence = true;
                actual_start = json_start - 1;
            }
        }

        // Remove the JSON block (and fences if present).
        let remove_end = if has_closing_fence {
            fence_end
        } else {
            json_end
        };
        lines.drain(actual_start..remove_end);

        // Also remove trailing empty lines that were before the JSON block.
        while lines.last().is_some_and(|l| l.trim().is_empty()) {
            lines.pop();
        }

        // Require code fences to match (both present or neither).
        if has_opening_fence != has_closing_fence {
            // Mismatched fences — still return the parsed result but don't
            // worry about the fence mismatch (model output is imperfect).
        }

        return Some(parsed);
    }

    None
}
