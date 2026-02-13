use super::{contains_keyword_as_words, CONSULTANT_TEXT_ONLY_MARKER, INTENT_GATE_MARKER};

fn is_pseudo_tool_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("[tool_use:")
        || lower.starts_with("[tool_call:")
        || lower.starts_with("[function_call:")
        || lower.starts_with("[functioncall:")
}

fn is_tool_name_like(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "terminal"
            | "browser"
            | "web_search"
            | "web_fetch"
            | "system_info"
            | "remember_fact"
            | "manage_config"
            | "send_file"
            | "spawn_agent"
            | "cli_agent"
            | "manage_cli_agents"
            | "health_probe"
            | "manage_skills"
            | "use_skill"
            | "skill_resources"
            | "manage_people"
            | "http_request"
            | "manage_oauth"
            | "read_channel_history"
    ) || lower.starts_with("mcp__")
        || lower.contains("__")
}

fn parse_name_field(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let (key, value) = trimmed.split_once(':')?;
    if !key.trim().eq_ignore_ascii_case("name") {
        return None;
    }
    let name = value.trim();
    if name.is_empty() || name.contains(' ') {
        return None;
    }
    Some(name.to_string())
}

pub(super) fn looks_like_deferred_action_response(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();

    // Pattern-based detection: catch "I'll [verb]", "I will [verb]", "Let me [verb]",
    // "Shall I [verb]", "Would you like me to [verb]" where verb needs tools.
    // This is dynamic — any new action verb the LLM uses is automatically caught.
    if has_action_promise(&lower) {
        return true;
    }

    // Special-case phrases that don't fit the prefix+verb pattern
    if contains_keyword_as_words(&lower, "i would typically") {
        return true;
    }

    // Structural format markers — substring match appropriate for these patterns
    lower.contains("[consultation]")
        || lower.contains(&INTENT_GATE_MARKER.to_ascii_lowercase())
        || lower.contains("[tool_use:")
        || lower.contains("[tool_call:")
        || lower.contains("arguments:")
}

/// Detect action-promise patterns like "I'll create", "I will run", "Let me check".
/// Returns true when the verb following the prefix is NOT a knowledge-only verb
/// (e.g., "explain", "describe", "summarize"), meaning the LLM needs tools to fulfill it.
pub(super) fn has_action_promise(text: &str) -> bool {
    // Normalize common Unicode apostrophes so contractions like "I’ll"
    // are treated the same as "I'll".
    let normalized = text.replace(['\u{2018}', '\u{2019}', '`', '\u{02BC}'], "'");

    // Verbs the LLM can fulfill without tools — pure knowledge/explanation verbs
    const KNOWLEDGE_ONLY_VERBS: &[&str] = &[
        "explain",
        "describe",
        "summarize",
        "clarify",
        "elaborate",
        "outline",
        "note",
        "mention",
        "address",
        "highlight",
        "tell",
        "share",
        "say",
        "answer",
        "provide",
        "be",
        "give",
        "offer",
        "rephrase",
        "restate",
    ];

    let words: Vec<String> = normalized
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();

    for i in 0..words.len() {
        // Determine the index of the verb after the action-promise prefix
        let verb_idx = if words[i] == "i'll" {
            // "I'll [verb]"
            Some(i + 1)
        } else if words[i] == "i" && words.get(i + 1).is_some_and(|w| w == "will") {
            // "I will [verb]"
            Some(i + 2)
        } else if words[i] == "let" && words.get(i + 1).is_some_and(|w| w == "me") {
            // "Let me [verb]"
            Some(i + 2)
        } else if words[i] == "shall" && words.get(i + 1).is_some_and(|w| w == "i") {
            // "Shall I [verb]"
            Some(i + 2)
        } else if words[i] == "would"
            && words.get(i + 1).is_some_and(|w| w == "you")
            && words.get(i + 2).is_some_and(|w| w == "like")
            && words.get(i + 3).is_some_and(|w| w == "me")
            && words.get(i + 4).is_some_and(|w| w == "to")
        {
            // "Would you like me to [verb]"
            Some(i + 5)
        } else {
            None
        };

        if let Some(vi) = verb_idx {
            if let Some(verb) = words.get(vi) {
                if !KNOWLEDGE_ONLY_VERBS.contains(&verb.as_str()) {
                    return true;
                }
            }
        }
    }

    false
}

/// Remove leaked consultant control markers and pseudo tool-call text.
pub(super) fn sanitize_consultant_analysis(analysis: &str) -> String {
    let lines: Vec<&str> = analysis.lines().collect();
    let has_pseudo_tool_block = lines.iter().any(|line| is_pseudo_tool_line(line));

    let mut cleaned: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();

        if lower == "arguments:" {
            let mut j = i + 1;
            let mut block_has_tool_signature = false;
            while j < lines.len() {
                let next = lines[j].trim();
                if next.is_empty() {
                    break;
                }
                if let Some(name) = parse_name_field(next) {
                    if is_tool_name_like(&name) {
                        block_has_tool_signature = true;
                    }
                }
                let next_lower = next.to_ascii_lowercase();
                if next_lower.starts_with("cmd:")
                    || next_lower.starts_with("command:")
                    || next_lower.starts_with("args:")
                    || next_lower.starts_with("arguments:")
                {
                    block_has_tool_signature = true;
                }
                j += 1;
            }

            if block_has_tool_signature {
                i = j;
                continue;
            }
        }

        if is_pseudo_tool_line(line) {
            i += 1;
            continue;
        }

        let replaced = line.replace(CONSULTANT_TEXT_ONLY_MARKER, "");
        let trimmed_replaced = replaced.trim();
        let lower_replaced = trimmed_replaced.to_ascii_lowercase();

        if lower_replaced == "[consultation]" {
            i += 1;
            continue;
        }

        if lower_replaced.starts_with(&INTENT_GATE_MARKER.to_ascii_lowercase()) {
            i += 1;
            continue;
        }

        // Some models echo the consultant control instruction verbatim.
        if lower_replaced.starts_with("[important:")
            && lower_replaced.contains("you are being consulted")
            && lower_replaced.contains("respond with text only")
        {
            i += 1;
            continue;
        }

        if has_pseudo_tool_block
            && (lower_replaced.starts_with("cmd:")
                || lower_replaced.starts_with("command:")
                || lower_replaced.starts_with("args:")
                || lower_replaced.starts_with("arguments:")
                || parse_name_field(trimmed_replaced)
                    .as_deref()
                    .is_some_and(is_tool_name_like))
        {
            i += 1;
            continue;
        }

        if trimmed_replaced.is_empty() {
            if cleaned.last().is_some_and(|prev| prev.is_empty()) {
                i += 1;
                continue;
            }
            cleaned.push(String::new());
        } else {
            cleaned.push(replaced.trim_end().to_string());
        }
        i += 1;
    }

    cleaned.join("\n").trim().to_string()
}
