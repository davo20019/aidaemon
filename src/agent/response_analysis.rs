use super::contains_keyword_as_words;
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

pub(super) fn looks_like_deferred_action_response(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();

    // Pattern-based detection: catch "I'll [verb]", "I will [verb]", "Let me [verb]",
    // "Shall I [verb]", "Would you like me to [verb]" where verb needs tools.
    // This is dynamic — any new action verb the LLM uses is automatically caught.
    if has_action_promise(&lower) {
        return true;
    }

    // Structural format markers — substring match appropriate for these patterns
    lower.contains("[consultation]")
        || lower.contains(&INTENT_GATE_MARKER.to_ascii_lowercase())
        || lower.contains("[tool_use:")
        || lower.contains("[tool_call:")
}

/// Detect a long status/plan response that reports a failed attempt and promises
/// unsupported future recovery instead of delivering the requested result.
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

/// Detect first-person past-tense side-effect claims like "I have deleted…",
/// "I've removed…", "I created…". Used by the completion phase to catch
/// fabricated action claims: a reply asserting a completed side effect in a
/// task that made zero tool calls cannot be truthful.
pub(super) fn claims_completed_side_effect(text: &str) -> bool {
    // Same normalization as has_action_promise: unify Unicode apostrophes
    // so "I’ve" matches "I've".
    let normalized = text
        .trim()
        .to_ascii_lowercase()
        .replace(['\u{2018}', '\u{2019}', '`', '\u{02BC}'], "'");

    // Past-tense verbs that imply a tool-mediated side effect. Knowledge
    // verbs ("explained", "summarized") are intentionally absent — those
    // can be fulfilled without tools.
    const PAST_ACTION_VERBS: &[&str] = &[
        "deleted",
        "removed",
        "created",
        "wrote",
        "written",
        "updated",
        "edited",
        "modified",
        "replaced",
        "moved",
        "renamed",
        "executed",
        "ran",
        "installed",
        "saved",
        "stored",
        "copied",
        "combined",
        "merged",
        "deployed",
        "restarted",
        "stopped",
        "killed",
        "downloaded",
        "cloned",
        "pushed",
        "pulled",
        "committed",
        "cleaned",
        "cleared",
        "built",
        "generated",
        "appended",
        "uploaded",
        "sent",
        "added",
    ];
    // Adverbs allowed between the subject and the verb:
    // "I have successfully executed", "I've just removed".
    const FILLER_ADVERBS: &[&str] = &["now", "just", "successfully", "already", "also"];

    let words: Vec<String> = normalized
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'')
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();

    let is_action_verb = |w: &str| PAST_ACTION_VERBS.contains(&w);
    let verb_after = |start: usize| -> bool {
        let mut idx = start;
        while words
            .get(idx)
            .is_some_and(|w| FILLER_ADVERBS.contains(&w.as_str()))
        {
            idx += 1;
        }
        words.get(idx).is_some_and(|w| is_action_verb(w))
    };

    for i in 0..words.len() {
        let claim = if words[i] == "i've" {
            // "I've removed …"
            verb_after(i + 1)
        } else if words[i] == "i" && words.get(i + 1).is_some_and(|w| w == "have") {
            // "I have deleted …"
            verb_after(i + 2)
        } else if words[i] == "i" {
            // Simple past: "I deleted …"
            verb_after(i + 1)
        } else {
            false
        };
        if claim {
            return true;
        }
    }
    false
}

/// Detect claims that a delegated agent has already been started. Without a
/// corresponding tool call, these statements fabricate asynchronous work that
/// will never produce a result.
pub(super) fn claims_delegation_started(text: &str) -> bool {
    let normalized = text
        .trim()
        .to_ascii_lowercase()
        .replace(['\u{2018}', '\u{2019}', '`', '\u{02BC}'], "'");

    let mentions_delegated_agent = [
        "specialist agent",
        "specialized review agent",
        "review agent",
        "research agent",
        "sub-agent",
        "subagent",
        "background agent",
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase));
    if !mentions_delegated_agent {
        return false;
    }

    [
        "i've initiated",
        "i have initiated",
        "i initiated",
        "i've started",
        "i have started",
        "i started",
        "i've launched",
        "i have launched",
        "i launched",
        "i've spawned",
        "i have spawned",
        "i spawned",
        "i've delegated",
        "i have delegated",
        "i delegated",
        "is now running",
        "is running in the background",
        "has been started",
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase))
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
        "know",
        "rephrase",
        "restate",
        // Memory/recall verbs — can be answered from conversation context or stored facts
        "recall",
        "confirm",
        "remember",
        "think",
        "point",
        "help",
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

/// Check whether a model response contains substantive text content rather
/// than just deferred-action phrases.  Used to decide whether to accept a
/// text-only response after repeated deferred-no-tool retries: if the model
/// finally produced real content (greeting, explanation, joke, etc.) we should
/// Short, human-presentable tool excerpt: at most a few lines of plain prose,
/// not structured data. The shared discrimination for every "surface the tool
/// output directly to the user" path (family rule: user-facing fallback text
/// is model-composed or minimal canned text — NEVER a raw data dump).
pub(in crate::agent) fn is_short_prose_excerpt(text: &str) -> bool {
    let t = text.trim();
    t.chars().count() <= 200
        && t.lines().count() <= 3
        && !t.starts_with('{')
        && !t.starts_with('[')
        && !t.contains("\":")
}

/// A reply whose LAST line is a promise ("I will answer both clearly.") is
/// unfulfilled by construction — nothing follows it. The deferred-action
/// detector exempts knowledge verbs ("answer", "explain") because such
/// promises normally PRECEDE their delivery in the same reply; that
/// exemption cannot apply to the closing line (live repro 2026-07-03: a
/// two-part question answered by half, ending literally with "I will answer
/// both clearly.").
pub(in crate::agent) fn reply_ends_with_unfulfilled_promise(reply: &str) -> bool {
    let Some(last) = reply
        .trim_end()
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
    else {
        return false;
    };
    let t = last.trim();
    if t.chars().count() > 120 || t.contains('?') {
        return false;
    }
    let lower = t.to_lowercase();
    if lower.starts_with("let me know") {
        return false;
    }
    let opener_len = [
        "i will ",
        "i'll ",
        "let me ",
        "i am going to ",
        "i'm going to ",
    ]
    .iter()
    .find(|o| lower.starts_with(**o))
    .map(|o| o.len());
    let Some(opener_len) = opener_len else {
        return false;
    };
    // Performative acknowledgments self-fulfill in the saying ("I'll
    // remember that...", "I'll keep that in mind.") — the act already
    // happened. Content-DELIVERY verbs ("I will answer both clearly.")
    // promise material that must follow, and nothing follows a closing line.
    const PERFORMATIVE_ACK_VERBS: [&str; 7] = [
        "remember", "note", "keep", "bear", "confirm", "recall", "treat",
    ];
    let verb = lower[opener_len..].split_whitespace().next().unwrap_or("");
    !PERFORMATIVE_ACK_VERBS.contains(&verb)
}

/// A final reply that is a raw structured-data dump: an optional short
/// lead-in line ("Here's the command output:") followed by a large JSON/
/// bracket blob. Sibling of the pasted-file-page detector — same disease
/// (data shipped instead of an answer), no harness page header (live repro
/// 2026-07-03: 334-study ClinicalTrials JSON pasted inline as the final
/// answer). One-shot retry, never a hard block: a genuinely-requested JSON
/// snippet costs one bounce, then ships.
pub(in crate::agent) fn reply_is_raw_data_dump(reply: &str) -> bool {
    let t = reply.trim();
    let body = match t.split_once('\n') {
        // Tolerate a short lead-in line ending with ':'.
        Some((first, rest)) if first.trim_end().ends_with(':') && first.chars().count() <= 80 => {
            rest.trim_start()
        }
        _ => t,
    };
    (body.starts_with('{') || body.starts_with('[')) && body.chars().count() > 600
}

/// A final reply that is a verbatim read_file page — the harness's own page
/// header (`File: <path> (lines A-B of N, X bytes...)`) followed by
/// line-numbered content — is a paste, never an answer. Observed live: after
/// a large API result was spilled to a file, the model paged it linearly and
/// shipped page 5 of raw JSON as its reply. The header and `NNN | ` line
/// format are harness-generated, so this detection is structural, not a
/// guess about model prose.
pub(super) fn reply_is_pasted_file_page(reply: &str) -> bool {
    let has_page_header = reply.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("File: ") && t.contains("(lines ") && t.contains(" bytes")
    });
    if !has_page_header {
        return false;
    }
    let numbered_lines = reply
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
            digits >= 1 && t[digits..].trim_start().starts_with('|')
        })
        .count();
    numbered_lines >= 5
}

/// let it through instead of stalling further.
///
/// The heuristic:
/// 1. The text must be at least `min_len` characters (default 50).
/// 2. After stripping lines that are purely deferred-action phrases, there must
///    still be substantive content left.
pub(super) fn is_substantive_text_response(text: &str, min_len: usize) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < min_len {
        return false;
    }

    // Strip lines that are purely deferred-action phrases.
    // If most of the text survives, the response is substantive.
    let substantive_lines: Vec<&str> = trimmed
        .lines()
        .filter(|line| {
            let l = line.trim();
            if l.is_empty() {
                return false;
            }
            // Keep lines that do NOT look like pure deferral text
            !has_action_promise(&l.to_ascii_lowercase())
        })
        .collect();

    let substantive_text: String = substantive_lines.join(" ");
    let substantive_len = substantive_text.trim().len();

    // Must have at least min_len chars of non-deferred content
    substantive_len >= min_len
}

/// Heuristic: does the user's message look like a multi-part request that
/// warrants a detailed response?  We check for numbered lists, explanation
/// keywords, and conjunction-heavy compound tasks.
pub(super) fn looks_like_multi_part_request(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();

    // Count numbered/lettered items: "1)", "2)", "a)", "b)", "step 1", etc.
    let numbered_items = {
        let re = regex::Regex::new(r"(?:^|\s)(?:\d+[.)]\s|[a-e][.)]\s|step\s+\d)").unwrap();
        re.find_iter(&lower).count()
    };
    if numbered_items >= 2 {
        return true;
    }

    // Explanation keywords: user explicitly wants reasoning
    let explanation_words = [
        "explain why",
        "explain how",
        "tell me why",
        "describe how",
        "show me",
        "what did you",
        "summarize what",
        "thorough review",
        "find all",
        "list all",
        "review it",
        "review the",
        "audit",
    ];
    let has_explanation_request = explanation_words.iter().any(|w| lower.contains(w));

    // Compound task indicators
    let compound_signals = [
        "also ",
        "then ",
        "after that",
        "additionally",
        "finally ",
        "and then",
        "before ",
        "as well",
    ];
    let compound_count = compound_signals
        .iter()
        .filter(|s| lower.contains(*s))
        .count();

    // Multi-part if explanation requested, or ≥2 compound signals
    has_explanation_request || compound_count >= 2
}

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

/// A final reply that PROMISES imminent or in-progress action ("I'm searching
/// the API now...", "Let me check that.") is only honest if work actually
/// exists to back it. The caller pairs this detector with the task ledger
/// (zero tool calls = nothing running, nothing spawned); the detector itself
/// only classifies the reply text. Live repro 2026-07-03 (task ab7b318d):
/// "I'm searching the ClinicalTrials.gov API ... now..." shipped as the FINAL
/// answer of a task with zero tool calls, then the task ended — nothing was
/// running.
pub(super) fn reply_is_unbacked_action_promise(reply: &str) -> bool {
    let t = reply.trim();
    // A bare promise is short; long analytical answers that happen to contain
    // intent phrasing are real answers.
    if t.is_empty() || t.chars().count() > 400 {
        return false;
    }
    // Questions back to the user are legitimate final replies.
    if t.contains('?') {
        return false;
    }
    let lower = t.to_lowercase();
    // "Let me know ..." is a closing formula, not an action promise.
    if lower.starts_with("let me know") {
        return false;
    }
    // PRESENT-PROGRESSIVE claims of current activity only. Future intent
    // ("I'll ...", "Let me ...") is the existing tier-gated deferred-action
    // gate's territory (`has_action_promise`): the Autonomous tier chooses to
    // trust stated plans. A claim that work is happening RIGHT NOW with an
    // empty tool ledger is not a plan — it is a false status report, and
    // anti-fabrication guards are never tier-gated.
    const STATUS_OPENERS: [&str; 6] = [
        "i'm ",
        "i am ",
        "one moment",
        "give me a moment",
        "working on ",
        "on it",
    ];
    const PROGRESSIVE_OPENERS: [&str; 9] = [
        "searching",
        "checking",
        "looking",
        "fetching",
        "querying",
        "running",
        "scanning",
        "pulling",
        "starting",
    ];
    let opener = STATUS_OPENERS.iter().any(|o| lower.starts_with(o));
    let progressive = PROGRESSIVE_OPENERS.iter().any(|o| lower.starts_with(o));
    if !opener && !progressive {
        return false;
    }
    // Imminence: trailing ellipsis, an explicit "now"/"currently", or an
    // opener that is already an unambiguous in-progress report. Without one
    // of these, "I'm confident the answer is 42." style replies stay
    // untouched.
    t.ends_with("...")
        || t.ends_with('…')
        || contains_keyword_as_words(&lower, "now")
        || contains_keyword_as_words(&lower, "currently")
        || lower.starts_with("one moment")
        || lower.starts_with("give me a moment")
        || lower.starts_with("working on ")
        || lower.starts_with("on it")
}

#[cfg(test)]
mod unbacked_promise_tests {
    use super::reply_is_unbacked_action_promise;

    #[test]
    fn fires_on_false_in_progress_status_claims() {
        // The live repro, verbatim shape.
        assert!(reply_is_unbacked_action_promise(
            "I'm searching the ClinicalTrials.gov API specifically for recruiting \
             skin cancer trials in the Fairfax/Chantilly area now..."
        ));
        assert!(reply_is_unbacked_action_promise(
            "Searching the trials database now..."
        ));
        assert!(reply_is_unbacked_action_promise(
            "Starting the send-resume workflow..."
        ));
        assert!(reply_is_unbacked_action_promise("One moment..."));
        assert!(reply_is_unbacked_action_promise("Working on it."));
        assert!(reply_is_unbacked_action_promise(
            "I am currently checking the registry for matches."
        ));
    }

    #[test]
    fn future_intent_is_the_deferred_gates_territory_not_ours() {
        // "I'll ..." / "Let me ..." are stated PLANS — handled by the
        // existing tier-gated deferred-action gate (has_action_promise),
        // where the Autonomous tier deliberately trusts them. This detector
        // must not re-gate them on all tiers.
        assert!(!reply_is_unbacked_action_promise(
            "I'll look into that and get back to you."
        ));
        assert!(!reply_is_unbacked_action_promise(
            "Let me check the deploy logs."
        ));
        assert!(!reply_is_unbacked_action_promise(
            "I will search ClinicalTrials.gov and report back."
        ));
    }

    #[test]
    fn ignores_real_answers_admissions_and_questions() {
        // Completed answer.
        assert!(!reply_is_unbacked_action_promise(
            "I found 3 recruiting trials near Fairfax: NCT001, NCT002, NCT003."
        ));
        // Honest inability admission (past tense, no imminence).
        assert!(!reply_is_unbacked_action_promise(
            "I'm sorry, I couldn't find any recruiting trials in that area."
        ));
        // Confidence statement with an intent-like opener but no imminence.
        assert!(!reply_is_unbacked_action_promise(
            "I'm confident the answer is 42."
        ));
        // Question back to the user.
        assert!(!reply_is_unbacked_action_promise(
            "Should I search ClinicalTrials.gov for recruiting studies now?"
        ));
        // Closing formula.
        assert!(!reply_is_unbacked_action_promise(
            "Let me know if you need anything else."
        ));
        // Long analytical answer containing intent phrasing stays untouched.
        let long = format!(
            "I'm going to summarize the findings. {}",
            "The data shows a consistent pattern across all sites. ".repeat(12)
        );
        assert!(!reply_is_unbacked_action_promise(&long));
    }
}
