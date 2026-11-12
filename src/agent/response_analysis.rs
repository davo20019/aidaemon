#[cfg(test)]
use super::contains_keyword_as_words;
#[cfg(test)]
use crate::llm_markers::INTENT_GATE_MARKER;

#[cfg(test)]
fn is_pseudo_tool_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("[tool_use:")
        || lower.starts_with("[tool_call:")
        || lower.starts_with("[function_call:")
        || lower.starts_with("[functioncall:")
}

#[cfg(test)]
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
            | "manage_api"
            | "http_request"
            | "manage_http_auth"
            | "manage_oauth"
            | "read_channel_history"
    ) || lower.starts_with("mcp__")
        || lower.contains("__")
}

#[cfg(test)]
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

/// Detect a long status/plan response that reports a failed attempt and promises
/// unsupported future recovery instead of delivering the requested result.
#[cfg(test)]
pub(super) fn looks_like_incomplete_retry_plan(text: &str) -> bool {
    let normalized = text
        .replace(['\u{2018}', '\u{2019}', '`', '\u{02BC}'], "'")
        .trim()
        .to_ascii_lowercase();

    let has_failure = [
        "timed out",
        "timeout",
        "failed",
        "couldn't complete",
        "could not complete",
    ]
    .iter()
    .any(|phrase| contains_keyword_as_words(&normalized, phrase));
    let has_unsupported_future = [
        "will retry",
        "i'll retry",
        "monitoring the system",
        "when the connection is stable",
        "as soon as the connection",
    ]
    .iter()
    .any(|phrase| contains_keyword_as_words(&normalized, phrase));
    let has_plan_scaffold = [
        "current plan",
        "research phase",
        "synthesis phase",
        "next phase",
    ]
    .iter()
    .any(|phrase| contains_keyword_as_words(&normalized, phrase));

    has_failure && has_unsupported_future && has_plan_scaffold
}

/// Detect a reply that punts file access back to the user ("please upload
/// the file", "provide the full path") instead of locating it with tools.
/// Multi-word phrases — substring matching is appropriate here (see
/// keyword-matching guidance in CLAUDE.md).
#[cfg(test)]
pub(super) fn reply_defers_file_access(text: &str) -> bool {
    let normalized = text
        .replace('\u{2019}', "'")
        .trim()
        .to_ascii_lowercase()
        .replace(char::is_whitespace, " ");
    const DEFER_PHRASES: &[&str] = &[
        "upload the file",
        "upload that file",
        "upload it to",
        "attach the file",
        "attach that file",
        "provide the full path",
        "provide the path",
        "provide the file",
        "provide a path",
        "share the file",
        "don't have access to that",
        "don't have access to the file",
        "don't have direct access",
        "do not have access to that",
        "do not have direct access",
        "once i have access",
    ];
    DEFER_PHRASES
        .iter()
        .any(|phrase| normalized.contains(phrase))
}

/// Detect whether the user's message references a concrete file: a token
/// with a known document/code extension, or an explicit path.
#[cfg(test)]
pub(super) fn user_text_references_file(text: &str) -> bool {
    const FILE_EXTENSIONS: &[&str] = &[
        ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".pptx", ".json", ".log", ".rs",
        ".py", ".js", ".ts", ".html", ".css", ".toml", ".yaml", ".yml", ".png", ".jpg", ".jpeg",
        ".zip", ".epub",
    ];
    for raw_token in text.split_whitespace() {
        let token = raw_token
            .trim_matches(|c: char| c.is_ascii_punctuation() && c != '.' && c != '/' && c != '~')
            .to_ascii_lowercase();
        if token.starts_with("~/") || (token.starts_with('/') && token.len() > 1) {
            return true;
        }
        if FILE_EXTENSIONS
            .iter()
            .any(|ext| token.ends_with(ext) && token.len() > ext.len())
        {
            return true;
        }
    }
    false
}

/// Extract an explicit requested source count (for example, "two primary
/// sources"). This is a language-level requirement, so bounded phrase parsing
/// is appropriate; it never decides authority or whether an action succeeded.
#[cfg(test)]
pub(super) fn requested_source_count(text: &str) -> Option<usize> {
    let tokens = text
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_ascii_alphanumeric())
                .to_ascii_lowercase()
        })
        .collect::<Vec<_>>();
    let number = |token: &str| match token {
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        other => other
            .parse::<usize>()
            .ok()
            .filter(|value| (1..=10).contains(value)),
    };
    for (index, token) in tokens.iter().enumerate() {
        if token == "source" || token == "sources" {
            for preceding in tokens[index.saturating_sub(3)..index].iter().rev() {
                if let Some(count) = number(preceding) {
                    return Some(count);
                }
            }
        }
    }
    None
}

#[cfg(test)]
pub(super) fn requests_exact_conversation_history(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "exact stop condition",
        "exact wording",
        "what did you say",
        "what did i say",
        "your earlier plan",
        "your previous plan",
        "immediately earlier",
        "earlier message",
        "previous message",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

#[cfg(test)]
pub(super) fn reply_claims_history_unavailable(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "don't have the exact",
        "do not have the exact",
        "can't recall",
        "cannot recall",
        "can't retrieve",
        "cannot retrieve",
        "not available in context",
        "isn't available in context",
        "is not available in context",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase))
}

pub(in crate::agent) use aidaemon_runtime::turn::reply_checks::*;

/// Remove leaked text-only control markers and pseudo tool-call text.
#[cfg(test)]
pub(super) fn sanitize_response_analysis(analysis: &str) -> String {
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

        let replaced = line.replace(crate::llm_markers::TEXT_ONLY_RESPONSE_MARKER, "");
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

        // Some models echo the text-only control instructions verbatim.
        // Strip the control header and nearby instruction lines so they don't
        // pollute the injected warm-start context for iteration 2.
        if lower_replaced.starts_with("[important:")
            && (lower_replaced.contains("consultation")
                || (lower_replaced.contains("you are being consulted")
                    && lower_replaced.contains("respond with text only")))
        {
            i += 1;
            continue;
        }
        if lower_replaced.contains("text only")
            && (lower_replaced.contains("no tools")
                || lower_replaced.contains("no function calls")
                || lower_replaced.contains("tool_use")
                || lower_replaced.contains("functioncall"))
        {
            i += 1;
            continue;
        }
        if lower_replaced.starts_with("end your response with")
            || lower_replaced.starts_with("end with one line")
            || lower_replaced == "guidelines:"
            || lower_replaced.starts_with("- complexity:")
            || lower_replaced.starts_with("- only include schedule")
            || lower_replaced.starts_with("- domains is optional")
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

#[cfg(test)]
mod unbacked_promise_tests {
    use super::{
        reply_claims_history_unavailable, requested_source_count,
        requests_exact_conversation_history,
    };

    #[test]
    fn parses_explicit_source_counts() {
        assert_eq!(
            requested_source_count("Verify this against two primary sources."),
            Some(2)
        );
        assert_eq!(requested_source_count("Use 3 independent sources"), Some(3));
        assert_eq!(requested_source_count("Research this thoroughly"), None);
    }

    #[test]
    fn recognizes_exact_history_lookup_shortfall() {
        assert!(requests_exact_conversation_history(
            "What was the exact stop condition in your earlier plan?"
        ));
        assert!(reply_claims_history_unavailable(
            "I don't have the exact wording available in context."
        ));
        assert!(!requests_exact_conversation_history(
            "What is a reasonable stop condition?"
        ));
    }
}
