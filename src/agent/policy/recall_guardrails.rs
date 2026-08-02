#[cfg(test)]
use serde_json::Value;

use super::intent_routing::contains_keyword_as_words;
use crate::traits::{Fact, Message};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CriticalFactQuery {
    OwnerName,
    AssistantName,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CriticalFactSummary {
    pub owner_name: Option<String>,
    pub assistant_name: Option<String>,
}

pub(super) fn is_personal_memory_tool(name: &str) -> bool {
    matches!(
        name,
        "manage_people" | "manage_memories" | "remember_fact" | "search_history"
    )
}

fn normalize_name_candidate(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'));
    if trimmed.is_empty() || trimmed.len() > 80 {
        return None;
    }
    if trimmed
        .chars()
        .any(|c| matches!(c, '\n' | '\r' | '[' | ']' | '{' | '}'))
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn extract_name_from_phrase(value: &str) -> Option<String> {
    let lower = value.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    for prefix in ["my name is ", "i am ", "i'm ", "call me "] {
        if lower.starts_with(prefix) {
            let name = value.trim()[prefix.len()..].trim();
            return normalize_name_candidate(name);
        }
    }

    if let Some(idx) = lower.find(" is myself") {
        return normalize_name_candidate(value[..idx].trim());
    }

    None
}

pub(super) fn detect_critical_fact_query(user_text: &str) -> Option<CriticalFactQuery> {
    let lower = user_text
        .trim()
        .trim_end_matches(['?', '!', '.'])
        .trim()
        .to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    // Multi-part questions should NOT be handled deterministically.
    // Questions like "What's my name, what do I like, and what's my dog's name?"
    // need the full LLM with context to answer comprehensively.
    let comma_count = lower.matches(',').count();
    let question_mark_count = lower.matches('?').count();
    let has_conjunction_joining = lower.contains(" and what")
        || lower.contains(" and who")
        || lower.contains(" and tell")
        || lower.contains(" and my");
    if comma_count >= 2 || question_mark_count >= 2 || has_conjunction_joining {
        return None;
    }

    let asks_owner_name = matches!(
        lower.as_str(),
        "what is my name" | "what's my name" | "who am i" | "tell me my name" | "my full name"
    );
    if asks_owner_name {
        return Some(CriticalFactQuery::OwnerName);
    }

    let asks_assistant_name = matches!(
        lower.as_str(),
        "what is your name"
            | "what's your name"
            | "who are you"
            | "what should i call you"
            | "what is your bot name"
            | "what's your bot name"
    );
    if asks_assistant_name {
        return Some(CriticalFactQuery::AssistantName);
    }

    None
}

pub(super) fn extract_critical_fact_summary(facts: &[Fact]) -> CriticalFactSummary {
    let mut summary = CriticalFactSummary::default();

    for fact in facts {
        let key = fact.key.trim();
        let value = fact.value.trim();
        if key.is_empty() || value.is_empty() {
            continue;
        }
        let lower_key = key.to_ascii_lowercase();
        let lower_cat = fact.category.trim().to_ascii_lowercase();

        if summary.owner_name.is_none() {
            let owner_name_key = matches!(
                lower_key.as_str(),
                "name" | "owner_name" | "user_name" | "full_name" | "my_name" | "owner"
            );
            let owner_name_category = matches!(
                lower_cat.as_str(),
                "user" | "personal" | "profile" | "identity"
            );
            if (owner_name_key && owner_name_category) || lower_key == "owner_name" {
                summary.owner_name = normalize_name_candidate(value);
            } else if let Some(name) = extract_name_from_phrase(value) {
                if lower_key.contains("name") || lower_key.contains("owner") {
                    summary.owner_name = Some(name);
                }
            }
        }

        if summary.assistant_name.is_none() {
            let assistant_key = matches!(
                lower_key.as_str(),
                "assistant_name" | "bot_name" | "ai_name" | "daemon_name"
            ) || (lower_key == "name"
                && matches!(lower_cat.as_str(), "assistant" | "bot"));
            if assistant_key {
                summary.assistant_name = normalize_name_candidate(value);
            }
        }
    }

    summary
}

#[cfg(test)]
pub(super) fn deterministic_reply_for_critical_query(
    query: CriticalFactQuery,
    summary: &CriticalFactSummary,
) -> String {
    match query {
        CriticalFactQuery::OwnerName => summary.owner_name.as_ref().map_or_else(
            || {
                "I don't have your name saved in critical memory yet. Tell me \"my name is ...\" and I'll pin it.".to_string()
            },
            |name| format!("Your name is {}.", name),
        ),
        CriticalFactQuery::AssistantName => summary.assistant_name.as_ref().map_or_else(
            || "I don't have a pinned assistant name in critical memory right now.".to_string(),
            |name| format!("My name is {}.", name),
        ),
    }
}

pub(super) fn build_critical_facts_prompt_block(summary: &CriticalFactSummary) -> Option<String> {
    let mut lines = vec![
        "═══ CRITICAL FACTS — USE THESE EXACT VALUES ═══".to_string(),
        "These pinned values cover ONLY your user's identity and your own — no other entity."
            .to_string(),
        "When asked about a fact below, reply with the EXACT value shown here.".to_string(),
        "Do NOT substitute, paraphrase, or infer different values from training data.".to_string(),
        "They do NOT apply to subjects from the current conversation: if the question refers to \
         an entity just discussed (e.g. \"the owner\"/\"the founder\" of a company), answer from \
         that conversation context, not from this block."
            .to_string(),
    ];

    let mut fact_count = 0;
    if let Some(owner_name) = summary.owner_name.as_ref() {
        lines.push(format!("• Owner name → {}", owner_name));
        fact_count += 1;
    }
    if let Some(assistant_name) = summary.assistant_name.as_ref() {
        lines.push(format!("• Assistant name → {}", assistant_name));
        fact_count += 1;
    }
    lines.push("═══════════════════════════════════════════════".to_string());

    if fact_count == 0 {
        None
    } else {
        Some(lines.join("\n"))
    }
}

pub(super) fn text_relates_to_critical_identity(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    if detect_critical_fact_query(&lower).is_some() {
        return true;
    }

    contains_keyword_as_words(&lower, "my name is")
        || contains_keyword_as_words(&lower, "call me")
        || contains_keyword_as_words(&lower, "i am")
        || contains_keyword_as_words(&lower, "owner name")
        || contains_keyword_as_words(&lower, "bot name")
        || contains_keyword_as_words(&lower, "assistant name")
        || contains_keyword_as_words(&lower, "wife")
        || contains_keyword_as_words(&lower, "husband")
        || contains_keyword_as_words(&lower, "spouse")
        || contains_keyword_as_words(&lower, "daughter")
        || contains_keyword_as_words(&lower, "son")
        || contains_keyword_as_words(&lower, "children")
        || lower.contains(" is myself")
        || (lower.contains("[user]") && lower.contains("name:"))
        || (lower.contains("[user]") && lower.contains("name ="))
        || (lower.contains("saved fact") && lower.contains("name"))
        || (lower.contains("remembered:") && lower.contains("name"))
}

#[cfg(test)]
pub(super) fn filter_tool_defs_for_personal_memory(defs: &[Value]) -> Vec<Value> {
    defs.iter()
        .filter_map(|def| {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())?;
            if is_personal_memory_tool(name) {
                Some(def.clone())
            } else {
                None
            }
        })
        .collect()
}

/// Execution tools blocked when delegation mode is active.
/// Keep spawn_agent available for task-lead orchestration.
pub(super) fn is_delegation_blocked_tool(name: &str) -> bool {
    matches!(name, "terminal" | "browser" | "run_command")
}

#[cfg(test)]
pub(super) fn filter_tool_defs_for_delegation(defs: &[Value]) -> Vec<Value> {
    defs.iter()
        .filter_map(|def| {
            let name = def
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())?;
            if is_delegation_blocked_tool(name) {
                None
            } else {
                Some(def.clone())
            }
        })
        .collect()
}

/// Detect store/write intent — user is asking the agent to SAVE facts, not recall them.
/// When true, the personal-memory recall restriction should NOT apply because
/// the agent needs to make multiple `remember_fact` / `manage_people` calls.
pub(super) fn looks_like_personal_memory_store_request(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    // Recall guardrail is the lenient site: a false positive here just keeps
    // the tool palette wider, while a false negative blocks a legitimate
    // store request. Combine the lenient single-word verbs with the strict
    // multi-word phrases so any keyword recognized by the schedule gate is
    // also recognized here.
    crate::agent::intent_keywords::MEMORY_STORE_LENIENT_VERBS
        .iter()
        .chain(crate::agent::intent_keywords::MEMORY_STORE_STRICT_PHRASES.iter())
        .any(|kw| contains_keyword_as_words(&lower, kw))
}

pub(crate) fn looks_like_personal_memory_recall_question(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }

    // Store intent takes priority — "remember these facts about me" is a write, not a read.
    if looks_like_personal_memory_store_request(user_text) {
        return false;
    }

    // Compound task detection: if the message contains BOTH recall keywords AND
    // action verbs that require non-memory tools (create, write, build, etc.),
    // it's a compound task, not a pure recall question. Don't restrict tools.
    let has_action_verbs = contains_keyword_as_words(&lower, "create")
        || contains_keyword_as_words(&lower, "write")
        || contains_keyword_as_words(&lower, "build")
        || contains_keyword_as_words(&lower, "generate")
        || contains_keyword_as_words(&lower, "make")
        || contains_keyword_as_words(&lower, "code")
        || contains_keyword_as_words(&lower, "script")
        || contains_keyword_as_words(&lower, "deploy")
        || contains_keyword_as_words(&lower, "install")
        || contains_keyword_as_words(&lower, "run")
        || contains_keyword_as_words(&lower, "execute")
        || contains_keyword_as_words(&lower, "search")
        || contains_keyword_as_words(&lower, "fetch")
        || contains_keyword_as_words(&lower, "download")
        || contains_keyword_as_words(&lower, "send")
        || contains_keyword_as_words(&lower, "post")
        || contains_keyword_as_words(&lower, "tweet");
    if has_action_verbs {
        return false;
    }

    let mentions_personal_entities = contains_keyword_as_words(&lower, "daughter")
        || contains_keyword_as_words(&lower, "daughters")
        || contains_keyword_as_words(&lower, "daughter's")
        || contains_keyword_as_words(&lower, "daughters'")
        || contains_keyword_as_words(&lower, "son")
        || contains_keyword_as_words(&lower, "sons")
        || contains_keyword_as_words(&lower, "son's")
        || contains_keyword_as_words(&lower, "sons'")
        || contains_keyword_as_words(&lower, "kid")
        || contains_keyword_as_words(&lower, "kids")
        || contains_keyword_as_words(&lower, "child")
        || contains_keyword_as_words(&lower, "children")
        || contains_keyword_as_words(&lower, "pet")
        || contains_keyword_as_words(&lower, "pets")
        || contains_keyword_as_words(&lower, "dog")
        || contains_keyword_as_words(&lower, "cat")
        || contains_keyword_as_words(&lower, "family")
        || contains_keyword_as_words(&lower, "wife")
        || contains_keyword_as_words(&lower, "husband")
        || contains_keyword_as_words(&lower, "mom")
        || contains_keyword_as_words(&lower, "dad")
        || contains_keyword_as_words(&lower, "mom's")
        || contains_keyword_as_words(&lower, "dad's")
        || contains_keyword_as_words(&lower, "mother")
        || contains_keyword_as_words(&lower, "father");

    let explicitly_about_owner = contains_keyword_as_words(&lower, "my")
        || contains_keyword_as_words(&lower, "i")
        || contains_keyword_as_words(&lower, "me");
    let mentions_owner_profile = contains_keyword_as_words(&lower, "birthday")
        || contains_keyword_as_words(&lower, "birth date")
        || contains_keyword_as_words(&lower, "live")
        || contains_keyword_as_words(&lower, "residence")
        || contains_keyword_as_words(&lower, "work")
        || contains_keyword_as_words(&lower, "employer")
        || contains_keyword_as_words(&lower, "job")
        || contains_keyword_as_words(&lower, "username")
        || contains_keyword_as_words(&lower, "handle");
    let recall_question = contains_keyword_as_words(&lower, "who is")
        || contains_keyword_as_words(&lower, "who's")
        || contains_keyword_as_words(&lower, "what is")
        || contains_keyword_as_words(&lower, "what's")
        || contains_keyword_as_words(&lower, "where is")
        || contains_keyword_as_words(&lower, "where's")
        || contains_keyword_as_words(&lower, "where does")
        || contains_keyword_as_words(&lower, "where do")
        || contains_keyword_as_words(&lower, "what are")
        || contains_keyword_as_words(&lower, "what do")
        || contains_keyword_as_words(&lower, "do i");

    contains_keyword_as_words(&lower, "what do you know about me")
        || contains_keyword_as_words(&lower, "about me")
        || (explicitly_about_owner
            && (mentions_personal_entities || mentions_owner_profile)
            && recall_question)
        || (contains_keyword_as_words(&lower, "do i have") && mentions_personal_entities)
        || (contains_keyword_as_words(&lower, "what about") && mentions_personal_entities)
        || (contains_keyword_as_words(&lower, "do i") && mentions_personal_entities)
        || (contains_keyword_as_words(&lower, "i have")
            && mentions_personal_entities
            && user_is_reaffirmation_challenge(user_text))
}

pub(super) fn user_is_reaffirmation_challenge(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    contains_keyword_as_words(&lower, "are you sure")
        || contains_keyword_as_words(&lower, "really")
        || contains_keyword_as_words(&lower, "you sure")
        || contains_keyword_as_words(&lower, "certain")
}

/// Detect a follow-up question whose person referent is carried ONLY by a
/// third-person pronoun ("...what can you infer about her?"), with no explicit
/// subject named in the message itself.
///
/// On a small model these turns are prone to a *coreference hijack*: the pronoun
/// gets bound to whoever is most salient in the injected core-profile block
/// (e.g. the pinned partner) instead of the actual subject of the immediately
/// preceding exchange. Telemetry: user asked "Where is my mom from?" then
/// "...what can you infer?" and the model answered about the pinned partner.
/// When this fires, the loop anchors the pronoun to the prior exchange and
/// forces a memory lookup before answering.
pub(super) fn looks_like_pronoun_referent_followup(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() || lower.len() > 200 {
        return false;
    }

    // Must lean on a third-person *personal* pronoun for the referent.
    let has_person_pronoun = contains_keyword_as_words(&lower, "her")
        || contains_keyword_as_words(&lower, "him")
        || contains_keyword_as_words(&lower, "she")
        || contains_keyword_as_words(&lower, "he")
        || contains_keyword_as_words(&lower, "his")
        || contains_keyword_as_words(&lower, "hers");
    if !has_person_pronoun {
        return false;
    }

    // If the message names its own subject ("my mom", "my wife", ...) there is
    // no ambiguity to resolve — the pronoun is anchored within the message.
    let names_own_subject = contains_keyword_as_words(&lower, "my")
        && (contains_keyword_as_words(&lower, "mom")
            || contains_keyword_as_words(&lower, "mother")
            || contains_keyword_as_words(&lower, "dad")
            || contains_keyword_as_words(&lower, "father")
            || contains_keyword_as_words(&lower, "wife")
            || contains_keyword_as_words(&lower, "husband")
            || contains_keyword_as_words(&lower, "partner")
            || contains_keyword_as_words(&lower, "spouse")
            || contains_keyword_as_words(&lower, "daughter")
            || contains_keyword_as_words(&lower, "son")
            || contains_keyword_as_words(&lower, "kid")
            || contains_keyword_as_words(&lower, "kids")
            || contains_keyword_as_words(&lower, "child")
            || contains_keyword_as_words(&lower, "children"));
    if names_own_subject {
        return false;
    }

    // Recall / inference shaped: the model is being asked to reason about or
    // recall facts about the pronoun's referent.
    contains_keyword_as_words(&lower, "infer")
        || contains_keyword_as_words(&lower, "guess")
        || contains_keyword_as_words(&lower, "think")
        || contains_keyword_as_words(&lower, "know about")
        || contains_keyword_as_words(&lower, "tell me about")
        || contains_keyword_as_words(&lower, "what about")
        || contains_keyword_as_words(&lower, "what do you know")
        || contains_keyword_as_words(&lower, "where is")
        || contains_keyword_as_words(&lower, "where's")
        || contains_keyword_as_words(&lower, "who is")
        || contains_keyword_as_words(&lower, "who's")
}

/// A reaffirmation challenge is only "vague" — and safe to anchor hard to the
/// immediately previous exchange — when the message is short and carries no
/// new task of its own. Long or statement-shaped messages that merely contain
/// "really"/"certain" must NOT be pinned to the prior exchange.
pub(super) fn is_vague_reaffirmation_challenge(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    // Long messages carry their own topic/task — anchoring them to the prior
    // exchange would fight the user's actual request (same length heuristic
    // as cancel-intent detection in main_loop.rs).
    if lower.len() >= 80 {
        return false;
    }
    if contains_keyword_as_words(&lower, "are you sure")
        || contains_keyword_as_words(&lower, "you sure")
        || contains_keyword_as_words(&lower, "are you certain")
    {
        return true;
    }
    // Bare "really"/"certain" are too ambiguous on their own: require a short,
    // question-shaped message so statements like "I really need you to
    // refactor X" don't get pinned to the previous exchange.
    (contains_keyword_as_words(&lower, "really") || contains_keyword_as_words(&lower, "certain"))
        && lower.contains('?')
        && lower.len() <= 40
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ReaffirmationAnchorContext {
    pub prior_user_request: Option<String>,
    pub prior_assistant_reply: String,
}

/// Resolve the immediately previous user/assistant exchange that a vague
/// reaffirmation challenge ("Are you sure?") should address.
pub(super) fn resolve_reaffirmation_anchor(
    history: &[Message],
    challenge_text: &str,
) -> Option<ReaffirmationAnchorContext> {
    let challenge = challenge_text.trim();
    if challenge.is_empty() {
        return None;
    }

    let mut skipped_challenge = false;
    let mut prior_assistant: Option<&Message> = None;

    for msg in history.iter().rev() {
        match msg.role.as_str() {
            "user" => {
                // None-content rows (synthetic/tool-related) are skipped, not
                // treated as a resolution failure.
                let Some(content) = msg.content.as_deref() else {
                    continue;
                };
                let content = content.trim();
                if content.is_empty() {
                    continue;
                }
                if !skipped_challenge && content.eq_ignore_ascii_case(challenge) {
                    skipped_challenge = true;
                    continue;
                }
                if let Some(assistant) = prior_assistant {
                    return Some(ReaffirmationAnchorContext {
                        prior_user_request: Some(content.to_string()),
                        prior_assistant_reply: assistant.content.clone()?,
                    });
                }
            }
            "assistant" if prior_assistant.is_none() => {
                // Tool-call-only assistant rows can have None content — skip
                // them rather than aborting the whole resolution.
                let Some(content) = msg.content.as_deref() else {
                    continue;
                };
                if !content.trim().is_empty() {
                    prior_assistant = Some(msg);
                }
            }
            _ => {}
        }
    }

    prior_assistant.and_then(|assistant| {
        let reply = assistant.content.as_deref()?.trim();
        if reply.is_empty() {
            return None;
        }
        Some(ReaffirmationAnchorContext {
            prior_user_request: None,
            prior_assistant_reply: reply.to_string(),
        })
    })
}

pub(super) fn user_requests_external_verification(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    contains_keyword_as_words(&lower, "actually check")
        || contains_keyword_as_words(&lower, "double check")
        || contains_keyword_as_words(&lower, "double-check")
        || contains_keyword_as_words(&lower, "fact check")
        || contains_keyword_as_words(&lower, "fact-check")
        || contains_keyword_as_words(&lower, "verify")
        || contains_keyword_as_words(&lower, "verify this")
        || contains_keyword_as_words(&lower, "look it up")
        || contains_keyword_as_words(&lower, "look this up")
        || contains_keyword_as_words(&lower, "check online")
        || contains_keyword_as_words(&lower, "search the web")
        || contains_keyword_as_words(&lower, "use tools")
}

pub(super) fn tool_result_indicates_no_evidence(result_text: &str) -> bool {
    let lower = result_text.to_ascii_lowercase();
    lower.contains("no matches found")
        || lower.contains("person not found")
        || lower.contains("no active fact found")
        || lower.contains("none recorded")
        || lower.contains("no results")
        || lower.contains("not found")
        || lower.contains("couldn't find")
        || lower.contains("could not find")
        || lower.contains("i don't have any information")
        || lower.contains("i don't have information")
        || lower.contains("no information")
        || lower.contains("no relevant")
        || lower.contains("no evidence")
        // Terminal's empty-output marker: a grep/find that printed nothing.
        || lower.contains("(no output)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FactPrivacy;
    use chrono::Utc;
    use serde_json::json;

    fn make_fact(category: &str, key: &str, value: &str) -> Fact {
        Fact {
            id: 1,
            category: category.to_string(),
            key: key.to_string(),
            value: value.to_string(),
            source: "test".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            superseded_at: None,
            recall_count: 0,
            last_recalled_at: None,
            channel_id: None,
            privacy: FactPrivacy::Global,
            first_seen_at: None,
            source_excerpt: None,
        }
    }

    #[test]
    fn detects_personal_memory_recall_questions() {
        assert!(looks_like_personal_memory_recall_question(
            "Do I have daughters?"
        ));
        assert!(looks_like_personal_memory_recall_question(
            "What about pets?"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Do I have node installed?"
        ));
    }

    #[test]
    fn store_requests_not_classified_as_recall() {
        // Store intent should NOT trigger recall restriction
        assert!(!looks_like_personal_memory_recall_question(
            "Remember these facts about me: I like coffee"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Please remember my dog is named Luna"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Save this about me: I work in Miami"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Note that I prefer dark mode"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Keep in mind I work 8am-6pm"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Update my work hours to 9am-5pm"
        ));
    }

    #[test]
    fn compound_tasks_not_classified_as_recall() {
        // Compound tasks that include recall + action should NOT be recall-only
        assert!(!looks_like_personal_memory_recall_question(
            "What do you know about me? After answering, create a Python script showing my info"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Tell me about me and then write a summary document"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "What's my schedule? Also generate a calendar export"
        ));
        assert!(!looks_like_personal_memory_recall_question(
            "Do I have any pets? Search the web for pet care tips"
        ));
        // Pure recall should still match
        assert!(looks_like_personal_memory_recall_question(
            "What do you know about me?"
        ));
        assert!(looks_like_personal_memory_recall_question(
            "Do I have a dog?"
        ));
    }

    #[test]
    fn detects_personal_memory_store_requests() {
        assert!(looks_like_personal_memory_store_request(
            "Remember these facts about me"
        ));
        assert!(looks_like_personal_memory_store_request(
            "Please save my preferences"
        ));
        assert!(looks_like_personal_memory_store_request(
            "Note that I like dark mode"
        ));
        assert!(looks_like_personal_memory_store_request(
            "Keep in mind I work remotely"
        ));
        assert!(looks_like_personal_memory_store_request(
            "Update my schedule"
        ));
        // Pure recall should NOT trigger store detection
        assert!(!looks_like_personal_memory_store_request(
            "What do you know about me?"
        ));
        assert!(!looks_like_personal_memory_store_request(
            "Do I have a dog?"
        ));
    }

    #[test]
    fn detects_pronoun_referent_followup() {
        // The telemetry case: pronoun-only referent + inference cue.
        assert!(looks_like_pronoun_referent_followup(
            "Based on what you know about me and her what can you infer?"
        ));
        assert!(looks_like_pronoun_referent_followup(
            "What do you know about him?"
        ));
        assert!(looks_like_pronoun_referent_followup("Where is she from?"));
        assert!(looks_like_pronoun_referent_followup("Tell me about her"));
    }

    #[test]
    fn pronoun_referent_followup_ignores_explicit_subject() {
        // Message names its own subject — no ambiguity to resolve.
        assert!(!looks_like_pronoun_referent_followup(
            "Where is my mom from?"
        ));
        assert!(!looks_like_pronoun_referent_followup(
            "What do you know about my wife?"
        ));
        // No referential person pronoun.
        assert!(!looks_like_pronoun_referent_followup("What can you infer?"));
        assert!(!looks_like_pronoun_referent_followup(
            "What do you know about me?"
        ));
        // No recall/inference cue.
        assert!(!looks_like_pronoun_referent_followup("Tell her I said hi"));
        // Empty / overly long.
        assert!(!looks_like_pronoun_referent_followup(""));
    }

    #[test]
    fn distinguishes_challenge_vs_external_verification() {
        assert!(user_is_reaffirmation_challenge("Are you sure?"));
        assert!(!user_requests_external_verification("Are you sure?"));
        assert!(user_requests_external_verification(
            "Please check online and verify this."
        ));
    }

    fn history_msg(role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test".to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: chrono::Utc::now(),
            importance: 1.0,
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn resolve_reaffirmation_anchor_targets_immediately_previous_exchange() {
        let history = vec![
            history_msg("user", "Whats ecuador?"),
            history_msg(
                "assistant",
                "Ecuador is a country in northwestern South America.",
            ),
            history_msg("user", "How many R's in strawberry?"),
            history_msg("assistant", "There are 3 R's in strawberry."),
            history_msg("user", "Are you sure?"),
        ];

        let anchor =
            resolve_reaffirmation_anchor(&history, "Are you sure?").expect("anchor should resolve");

        assert_eq!(
            anchor.prior_user_request.as_deref(),
            Some("How many R's in strawberry?")
        );
        assert_eq!(
            anchor.prior_assistant_reply,
            "There are 3 R's in strawberry."
        );
    }

    #[test]
    fn resolve_reaffirmation_anchor_skips_empty_assistant_messages() {
        let history = vec![
            history_msg("user", "2 + 2 ?"),
            history_msg("assistant", ""),
            history_msg("assistant", "2 + 2 = 4"),
            history_msg("user", "Are you sure?"),
        ];

        let anchor =
            resolve_reaffirmation_anchor(&history, "Are you sure?").expect("anchor should resolve");

        assert_eq!(anchor.prior_user_request.as_deref(), Some("2 + 2 ?"));
        assert_eq!(anchor.prior_assistant_reply, "2 + 2 = 4");
    }

    #[test]
    fn resolve_reaffirmation_anchor_skips_none_content_messages() {
        // Tool-call-only assistant rows and synthetic user rows can have
        // content: None — they must be skipped, not abort the resolution.
        let mut none_user = history_msg("user", "");
        none_user.content = None;
        let mut none_assistant = history_msg("assistant", "");
        none_assistant.content = None;
        let history = vec![
            history_msg("user", "2 + 2 ?"),
            none_user,
            history_msg("assistant", "2 + 2 = 4"),
            none_assistant,
            history_msg("user", "Are you sure?"),
        ];

        let anchor = resolve_reaffirmation_anchor(&history, "Are you sure?")
            .expect("None-content messages should be skipped, not abort resolution");

        assert_eq!(anchor.prior_user_request.as_deref(), Some("2 + 2 ?"));
        assert_eq!(anchor.prior_assistant_reply, "2 + 2 = 4");
    }

    #[test]
    fn vague_reaffirmation_challenge_accepts_short_challenges() {
        assert!(is_vague_reaffirmation_challenge("Are you sure?"));
        assert!(is_vague_reaffirmation_challenge("really?"));
        assert!(is_vague_reaffirmation_challenge("you sure?"));
        assert!(is_vague_reaffirmation_challenge(
            "Are you certain about that?"
        ));
    }

    #[test]
    fn vague_reaffirmation_challenge_rejects_compound_or_statement_messages() {
        // Statements that merely contain a challenge keyword.
        assert!(!is_vague_reaffirmation_challenge(
            "I really need you to refactor the auth module"
        ));
        assert!(!is_vague_reaffirmation_challenge(
            "I'm certain the config is in the projects folder"
        ));
        // Long compound message carries its own new task.
        assert!(!is_vague_reaffirmation_challenge(
            "Are you sure? Anyway, now please write a long and detailed blog post about Ecuador for me"
        ));
        // Not a challenge at all.
        assert!(!is_vague_reaffirmation_challenge("Please deploy the app"));
    }

    #[test]
    fn detects_no_evidence_tool_results() {
        assert!(tool_result_indicates_no_evidence(
            "No matches found (40 files scanned)"
        ));
        assert!(tool_result_indicates_no_evidence(
            "Person 'Alice' not found."
        ));
        assert!(!tool_result_indicates_no_evidence(
            "Found 2 matches in profile data."
        ));
    }

    #[test]
    fn filters_tool_defs_to_personal_memory_scope() {
        let defs = vec![
            json!({"type":"function","function":{"name":"manage_people"}}),
            json!({"type":"function","function":{"name":"manage_memories"}}),
            json!({"type":"function","function":{"name":"browser"}}),
            json!({"type":"function","function":{"name":"terminal"}}),
        ];
        let filtered = filter_tool_defs_for_personal_memory(&defs);
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|d| d.get("function"))
            .filter_map(|f| f.get("name"))
            .filter_map(|n| n.as_str())
            .collect();
        assert_eq!(names, vec!["manage_people", "manage_memories"]);
    }

    #[test]
    fn detects_direct_first_person_relationship_recall() {
        for prompt in [
            "What's my dad's name?",
            "Who is my mom?",
            "What are my daughters' names?",
            "Where does my father live?",
            "Where do I live?",
            "Where do I work?",
            "What's my birthday?",
        ] {
            assert!(
                looks_like_personal_memory_recall_question(prompt),
                "prompt should be personal recall: {prompt}"
            );
        }
        assert!(!looks_like_personal_memory_recall_question(
            "Send me my daughter's resume"
        ));
    }

    #[test]
    fn identifies_delegation_blocked_tools() {
        assert!(is_delegation_blocked_tool("terminal"));
        assert!(is_delegation_blocked_tool("browser"));
        assert!(is_delegation_blocked_tool("run_command"));
        assert!(!is_delegation_blocked_tool("spawn_agent"));
        assert!(!is_delegation_blocked_tool("cli_agent"));
        assert!(!is_delegation_blocked_tool("web_search"));
    }

    #[test]
    fn filters_tool_defs_for_delegation_mode() {
        let defs = vec![
            json!({"type":"function","function":{"name":"terminal"}}),
            json!({"type":"function","function":{"name":"cli_agent"}}),
            json!({"type":"function","function":{"name":"web_search"}}),
            json!({"type":"function","function":{"name":"browser"}}),
            json!({"type":"function","function":{"name":"run_command"}}),
            json!({"type":"function","function":{"name":"spawn_agent"}}),
            json!({"type":"function","function":{"name":"remember_fact"}}),
        ];
        let filtered = filter_tool_defs_for_delegation(&defs);
        let names: Vec<&str> = filtered
            .iter()
            .filter_map(|d| d.get("function"))
            .filter_map(|f| f.get("name"))
            .filter_map(|n| n.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["cli_agent", "web_search", "spawn_agent", "remember_fact"]
        );
    }

    #[test]
    fn detects_critical_fact_queries() {
        assert_eq!(
            detect_critical_fact_query("What's my name?"),
            Some(CriticalFactQuery::OwnerName)
        );
        assert_eq!(
            detect_critical_fact_query("What is your bot name?"),
            Some(CriticalFactQuery::AssistantName)
        );
    }

    #[test]
    fn multi_part_questions_bypass_deterministic_resolver() {
        // Multi-part questions should go to the LLM for comprehensive answers
        assert_eq!(
            detect_critical_fact_query(
                "What's my name, what programming languages do I love, and what's my dog's name?"
            ),
            None
        );
        assert_eq!(
            detect_critical_fact_query("What's my name and what do I do for work?"),
            None
        );
        assert_eq!(
            detect_critical_fact_query("Who am I? What do I like? Where do I live?"),
            None
        );
        // Single-part questions still work
        assert_eq!(
            detect_critical_fact_query("What's my name?"),
            Some(CriticalFactQuery::OwnerName)
        );
    }

    #[test]
    fn extracts_critical_fact_summary() {
        let facts = vec![
            make_fact("user", "name", "Test Owner"),
            make_fact("assistant", "bot_name", "TestBot"),
            make_fact("user", "daughter_name", "Sofia"),
        ];
        let summary = extract_critical_fact_summary(&facts);
        assert_eq!(summary.owner_name.as_deref(), Some("Test Owner"));
        assert_eq!(summary.assistant_name.as_deref(), Some("TestBot"));
    }

    #[test]
    fn deterministic_reply_uses_critical_facts() {
        let summary = CriticalFactSummary {
            owner_name: Some("Test Owner".to_string()),
            assistant_name: Some("TestBot".to_string()),
        };
        assert_eq!(
            deterministic_reply_for_critical_query(CriticalFactQuery::OwnerName, &summary),
            "Your name is Test Owner."
        );
        assert_eq!(
            deterministic_reply_for_critical_query(CriticalFactQuery::AssistantName, &summary),
            "My name is TestBot."
        );
    }

    #[test]
    fn critical_facts_block_scopes_to_user_and_assistant_identity() {
        // Telemetry case: after a SpaceX conversation, "Who's the owner?" was
        // answered with the pinned owner name instead of the company's owner.
        // The block must say its values cover ONLY the user's/assistant's own
        // identity and that conversational antecedents win.
        let summary = CriticalFactSummary {
            owner_name: Some("Test Owner".to_string()),
            assistant_name: None,
        };
        let block = build_critical_facts_prompt_block(&summary).expect("block should render");
        assert!(
            block.contains("ONLY your user's identity and your own"),
            "block must scope pinned values to user/assistant identity: {}",
            block
        );
        assert!(
            block.contains("current conversation"),
            "block must defer to conversational antecedents: {}",
            block
        );
        // Existing consumers depend on these markers.
        assert!(block.contains("CRITICAL FACTS"));
        assert!(block.contains("• Owner name → Test Owner"));
    }

    #[test]
    fn critical_facts_block_empty_summary_renders_nothing() {
        let block = build_critical_facts_prompt_block(&CriticalFactSummary::default());
        assert!(block.is_none(), "no pinned facts → no block: {:?}", block);
    }

    #[test]
    fn detects_identity_related_text_snippets() {
        assert!(text_relates_to_critical_identity("my name is David"));
        assert!(text_relates_to_critical_identity(
            "Saved fact [user] name: David"
        ));
        assert!(!text_relates_to_critical_identity("run the tests"));
    }
}
