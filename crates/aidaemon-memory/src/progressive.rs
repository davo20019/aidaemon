//! Progressive extraction: after each owner exchange, a fast model proposes
//! durable facts, and the ones that are new, evidenced and not already
//! stored are written.

use std::sync::Arc;

use aidaemon_core::{ChannelVisibility, Message, ModelRequest, Provider, UserRole};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::records::{ExtractedMemoryGraph, Fact, FactPrivacy, PersonalMemoryWrite};
use crate::stores::FactStore;
use crate::task_learning::{FactLifetime, FACT_LIFETIME_INSTRUCTION};

/// How the user's message relates to a proposed fact. Only [`Asserted`]
/// facts are stored; a missing or unrecognized stance is [`Unstated`].
///
/// [`Asserted`]: FactStance::Asserted
/// [`Unstated`]: FactStance::Unstated
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactStance {
    /// The user states the fact in their own message.
    Asserted,
    /// The user asks whether the fact holds.
    Asked,
    /// The user's message assumes the fact without stating it (a question's
    /// premise, a conditional).
    Presupposed,
    /// Only the assistant's reply carries the fact.
    AssistantOnly,
    /// No stance given, or one this schema does not know.
    #[default]
    #[serde(other)]
    Unstated,
}

impl FactStance {
    /// The user's own message states the fact.
    pub fn is_user_assertion(self) -> bool {
        self == FactStance::Asserted
    }
}

/// A fact proposed by progressive extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineFact {
    pub category: String,
    pub key: String,
    pub value: String,
    /// The user's stance toward the fact. Missing, null or unknown values
    /// read as [`FactStance::Unstated`], which is never stored.
    #[serde(default, deserialize_with = "lenient_stance")]
    pub stance: FactStance,
    /// A verbatim quote from the user's message that states the fact.
    /// Missing or non-string reads as empty, which is never stored.
    #[serde(default, deserialize_with = "lenient_string")]
    pub evidence: String,
    /// The key of an already-stored fact this proposal carries the same
    /// information as. A restatement is never stored again.
    #[serde(default, deserialize_with = "lenient_key")]
    pub restates: Option<String>,
    #[serde(default)]
    pub graph: ExtractedMemoryGraph,
    #[serde(default)]
    pub personal_memory: Option<PersonalMemoryWrite>,
    #[serde(default)]
    pub lifetime: FactLifetime,
}

fn lenient_stance<'de, D: Deserializer<'de>>(deserializer: D) -> Result<FactStance, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default())
}

fn lenient_string<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string())
}

fn lenient_key<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    let key = lenient_string(deserializer)?;
    let key = key.trim();
    Ok((!key.is_empty()).then(|| key.to_string()))
}

/// Why a proposed fact is not stored, before any store call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inadmissible {
    /// The user did not state it (asked, presupposed, assistant-only or
    /// no stance).
    NotAsserted(FactStance),
    /// No quote from the user's message was given.
    NoEvidence,
    /// The quote does not occur in the user's message.
    EvidenceNotInUserText,
    /// It carries the same information as this already-stored fact.
    Restates(String),
}

/// Whether a proposal may be stored: the user asserted it, its evidence is
/// a quote found in the user's own message, and it does not restate a fact
/// already stored. The evidence check validates a citation against its
/// source; it does not interpret what the words mean.
pub fn admissibility(fact: &InlineFact, user_text: &str) -> Result<(), Inadmissible> {
    if let Some(key) = &fact.restates {
        return Err(Inadmissible::Restates(key.clone()));
    }
    if !fact.stance.is_user_assertion() {
        return Err(Inadmissible::NotAsserted(fact.stance));
    }
    let evidence = normalize_quote(&fact.evidence);
    if evidence.is_empty() {
        return Err(Inadmissible::NoEvidence);
    }
    if !normalize_quote(user_text).contains(&evidence) {
        return Err(Inadmissible::EvidenceNotInUserText);
    }
    Ok(())
}

/// Lowercase, typographic quotes folded to ASCII, whitespace runs collapsed,
/// and surrounding punctuation trimmed, so a faithful quote matches its
/// source despite case, spacing or a trailing period.
fn normalize_quote(text: &str) -> String {
    let folded: String = text
        .chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{02BC}' => '\'',
            '\u{201C}' | '\u{201D}' => '"',
            other => other,
        })
        .collect::<String>()
        .to_lowercase();
    let collapsed = folded.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_string()
}

/// Facts owner-authored (stated by the owner or saved by the agent on the
/// owner's behalf) that were written or refreshed at or after `since`:
/// what the agent itself stored during a run that started then. Superseded
/// rows are left out.
pub fn owner_writes_since(facts: &[Fact], since: DateTime<Utc>) -> Vec<Fact> {
    facts
        .iter()
        .filter(|fact| {
            fact.superseded_at.is_none()
                && fact.updated_at >= since
                && crate::sqlite::is_owner_authored_fact_source(&fact.source)
        })
        .cloned()
        .collect()
}

/// [`owner_writes_since`] over the store's active facts. A store error reads
/// as no writes, so extraction proceeds as it would without any.
pub async fn owner_writes_during_run<S>(store: &S, run_started_at: DateTime<Utc>) -> Vec<Fact>
where
    S: FactStore + ?Sized,
{
    match store.get_facts(None).await {
        Ok(facts) => owner_writes_since(&facts, run_started_at),
        Err(error) => {
            debug!(%error, "Could not read facts written during the run");
            Vec::new()
        }
    }
}

/// Whether owner memory may be extracted from a turn: only the owner's own
/// words, and never on an untrusted public platform.
pub fn may_extract(role: UserRole, visibility: ChannelVisibility) -> bool {
    role.can_persist_owner_memory() && visibility != ChannelVisibility::PublicExternal
}

/// Whether a user message is long enough to hold a durable fact. Shorter
/// messages (acknowledgments, greetings, single emoji) are not worth an
/// extraction call.
pub fn worth_extracting(user_text: &str) -> bool {
    /// Bytes below which a message cannot carry a durable fact.
    const MIN_BYTES: usize = 20;
    user_text.trim().len() >= MIN_BYTES
}

/// The exchange extraction reads, and where its facts are scoped.
#[derive(Debug, Clone, Copy)]
pub struct Exchange<'a> {
    pub user_text: &'a str,
    pub assistant_reply: &'a str,
    pub channel_id: Option<&'a str>,
    /// Facts already written during this run (see [`owner_writes_during_run`]).
    /// The model is shown them and must mark a proposal that restates one,
    /// which is then not stored a second time. Empty for most runs.
    pub already_stored: &'a [Fact],
}

/// At most this many already-stored facts are shown to the model.
const MAX_ALREADY_STORED_SHOWN: usize = 20;

/// The extraction call's user message: the exchange, and the facts already
/// stored during the run when there are any.
fn extraction_input(exchange: &Exchange<'_>) -> String {
    let mut input = format!(
        "User said: {}\n\nAssistant replied: {}",
        truncate_for_extraction(exchange.user_text, 500),
        truncate_for_extraction(exchange.assistant_reply, 500)
    );
    if !exchange.already_stored.is_empty() {
        input.push_str(
            "\n\nALREADY STORED during this exchange (category/key: value). Any fact you propose \
             that carries the same information as one of these must set \"restates\" to that \
             entry's key:",
        );
        for fact in exchange
            .already_stored
            .iter()
            .take(MAX_ALREADY_STORED_SHOWN)
        {
            input.push_str(&format!(
                "\n- {}/{}: {}",
                fact.category,
                fact.key,
                truncate_for_extraction(&fact.value, 200)
            ));
        }
    }
    input
}

/// Proposes facts from an exchange with a model, a bounded number of calls
/// at a time, and writes the ones worth keeping.
#[derive(Clone)]
pub struct ProgressiveExtractor {
    provider: Arc<dyn Provider>,
    model: String,
    permits: Arc<Semaphore>,
}

impl ProgressiveExtractor {
    /// At most two extraction calls run at once.
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self::with_permits(provider, model, Arc::new(Semaphore::new(2)))
    }

    /// An extractor sharing `permits` with others, so one concurrency limit
    /// holds across them.
    pub fn with_permits(
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
        permits: Arc<Semaphore>,
    ) -> Self {
        Self {
            provider,
            model: model.into(),
            permits,
        }
    }

    /// Ask the model for durable facts in `exchange`. Unparseable output is
    /// no facts; provider failures are errors.
    pub async fn propose(&self, exchange: &Exchange<'_>) -> anyhow::Result<Vec<InlineFact>> {
        let _permit = self.permits.acquire().await?;
        let system = ["You extract durable facts from conversations. Only extract facts that would be useful to remember long-term. \
                        Return a JSON array of objects with 'category', 'key', 'value', 'stance', 'evidence', and optional 'restates', 'graph' or 'personal_memory' fields.\n\n\
                        STANCE (required): how the USER's message relates to the fact. One of: \
                        \"asserted\" (the user states it), \"asked\" (the user asks whether it is so), \
                        \"presupposed\" (the user's question or request assumes it without stating it), \
                        \"assistant_only\" (only the assistant's reply says it). Only \"asserted\" facts are stored; \
                        a question never asserts its own premise.\n\
                        EVIDENCE (required): an exact, verbatim quote copied from the USER's message that states the fact. \
                        Never quote the assistant, never paraphrase. If you cannot quote the user stating it, do not propose it.\n\
                        RESTATES (optional): when an ALREADY STORED list is given, set this to the key of the stored entry \
                        that carries the same information as your proposal; such proposals are not stored again.\n\n\
                        Categories: user (personal info), preference (likes/dislikes), project (project details), technical (technical facts).\n\
                        Use snake_case keys for non-personal facts. For personal/profile facts involving names, aliases, handles, birth dates, residence, accounts, or family relationships, include a 'personal_memory' write plan and do not encode identity in a dynamic key. The plan has entities [{local_id,entity_type,canonical_name,is_reference,canonical_name_confirmed}], aliases [{entity_local_id,value,alias_type}], facts [{subject_local_id,predicate,value,display_value,valid_from,valid_to}], relationships [{source_local_id,relationship_type,target_local_id,valid_from,valid_to}], direct_user_statement, and correction. Use local_id 'owner' for the user. A later alias mention uses is_reference=true; a newly declared person uses false. Relationship types are PARENT_OF, CHILD_OF, LIVES_WITH, LIVES_IN, USES_HANDLE, HAS_ACCOUNT.\n\n\
                        CORRECTIONS: If the user is correcting or updating previously stated information (e.g., \"actually\", \"not X, it's Y\", \
                        \"I changed\", \"I meant\"), extract the CORRECTED fact using the same key format as the original would have used. \
                        The corrected value will automatically supersede the old one.\n\n\
                        IDENTITY: The 'user' category is ONLY for facts the user states about THEMSELVES in first person. \
                        Never store another person's name (friend, family member, client, applicant, public figure) under a user \
                        identity key like 'name'. When the user talks about someone else, do not extract a 'user' fact from it.\n\n\
                        GRAPH: When the fact contains named entities and a useful relationship, include: \
                        \"graph\":{\"entities\":[{\"local_id\":\"e1\",\"name\":\"...\",\"entity_type\":\"person|project|technology|organization|place|concept\",\"aliases\":[],\"confidence\":0.0}], \
                        \"relationships\":[{\"source_id\":\"e1\",\"target_id\":\"e2\",\"relation\":\"snake_case_relation\",\"confidence\":0.0}]}. \
                        Use \"owner\" as the name for the user when needed. Only include entities and relationships directly supported by the user's words. \
                        Do not infer entity types from fact category/key names, invent entities, or add a relationship unless both endpoints are present.\n\n\
                        If nothing is worth remembering, return an empty array: []\n\n\
                        Examples:\n\
                        - \"My dog's name is Mia\" → [{\"category\":\"user\",\"key\":\"dog_name\",\"value\":\"Mia\",\"stance\":\"asserted\",\"evidence\":\"My dog's name is Mia\"}]\n\
                        - \"Actually my dog's name is Max, not Mia\" → [{\"category\":\"user\",\"key\":\"dog_name\",\"value\":\"Max\",\"stance\":\"asserted\",\"evidence\":\"my dog's name is Max\"}]\n\
                        - \"I prefer dark mode\" → [{\"category\":\"preference\",\"key\":\"ui_theme\",\"value\":\"dark mode\",\"stance\":\"asserted\",\"evidence\":\"I prefer dark mode\"}]\n\
                        - \"My sister Alice lives in Tokyo, not Paris\" → emit personal_memory entities for Alice and Tokyo plus a LIVES_IN relationship, with stance \"asserted\" and evidence \"My sister Alice lives in Tokyo\"; do not create sister_location\n\
                        - \"Do you know what kind of car I drive?\" → [] (a question: the user asks, and only presupposes owning a car)\n\
                        - \"How's the weather?\" → []\n\n\
                        IMPORTANT: Return ONLY the JSON array, no other text.\n\n", FACT_LIFETIME_INSTRUCTION].concat();
        let user = extraction_input(exchange);
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![Message::system(system), Message::user(user)],
            ..Default::default()
        };
        let response = self.provider.complete(&request).await?;
        Ok(parse_inline_facts(&response.text))
    }

    /// Write the proposed `facts` that are worth keeping. Returns what was
    /// written, for the host's audit record.
    pub async fn write<S>(
        &self,
        store: &S,
        exchange: &Exchange<'_>,
        facts: Vec<InlineFact>,
    ) -> Vec<Value>
    where
        S: FactStore + ?Sized,
    {
        // Only facts the user asserted, quoted from their own message, and not
        // already stored this run, reach the store.
        let facts: Vec<InlineFact> = facts
            .into_iter()
            .filter(|fact| match admissibility(fact, exchange.user_text) {
                Ok(()) => true,
                Err(reason) => {
                    debug!(
                        key = fact.key,
                        ?reason,
                        "Skipping progressive fact the user did not state"
                    );
                    false
                }
            })
            .collect();
        if facts.is_empty() {
            return Vec::new();
        }
        let first_seen_at = chrono::Utc::now();
        // One snapshot of active facts, to suppress recall restatements.
        let existing_facts = store.get_facts(None).await.unwrap_or_default();
        let mut written: Vec<Value> = Vec::new();
        for fact in facts {
            if fact.lifetime.is_transient() && fact.personal_memory.is_none() {
                continue;
            }
            // The user's quoted words are the fact's provenance.
            let source_excerpt = crate::text::truncate_chars(fact.evidence.trim(), 200);
            if let Some(mut personal) = fact.personal_memory.clone() {
                personal.direct_user_statement = fact.stance.is_user_assertion();
                match store
                    .reconcile_personal_memory(
                        &personal,
                        "progressive",
                        Some(source_excerpt.as_str()),
                        exchange.channel_id,
                        FactPrivacy::Private,
                    )
                    .await
                {
                    Ok(result) => written.push(json!({
                        "structured_personal_memory": result.concise_summary(),
                        "unresolved": result.unresolved,
                    })),
                    Err(error) => warn!(%error, "Failed to reconcile progressive personal memory"),
                }
                continue;
            }
            // Identity facts (user.name etc.) must be evidenced by the
            // user's own words: a name merely mentioned is not the user's.
            if identity_fact_lacks_user_evidence(
                &fact.category,
                &fact.key,
                &fact.value,
                exchange.user_text,
            ) {
                warn!(
                    key = fact.key,
                    value = fact.value,
                    "Skipping user identity fact without evidence in the user's message"
                );
                continue;
            }
            // The assistant recalling a stored fact is not new information.
            if is_redundant_recall_fact(&fact.key, &fact.value, exchange.user_text, &existing_facts)
            {
                debug!(
                    key = fact.key,
                    value = fact.value,
                    "Skipping recall-restatement of a fact already in memory"
                );
                continue;
            }
            if duplicates_active_fact_in_other_category(
                &fact.category,
                &fact.key,
                &fact.value,
                &existing_facts,
            ) {
                debug!(
                    key = fact.key,
                    category = fact.category,
                    "Skipping progressive fact already stored under another category"
                );
                continue;
            }
            // Personal facts default to private, others to the channel,
            // until explicitly promoted.
            let privacy = if fact.category.trim().eq_ignore_ascii_case("user") {
                FactPrivacy::Private
            } else {
                FactPrivacy::Channel
            };
            if let Err(e) = store
                .upsert_fact_with_provenance(
                    &fact.category,
                    &fact.key,
                    &fact.value,
                    "progressive",
                    exchange.channel_id,
                    privacy,
                    Some(first_seen_at),
                    Some(source_excerpt.as_str()),
                )
                .await
            {
                warn!(error = %e, key = fact.key, "Failed to store progressive fact");
                continue;
            }
            if let Err(e) = store
                .project_extracted_fact_graph(
                    &fact.category,
                    &fact.key,
                    source_excerpt.as_str(),
                    &fact.graph,
                )
                .await
            {
                debug!(error = %e, key = fact.key, "Failed to project extracted graph");
            }
            written.push(json!({
                "category": fact.category,
                "key": fact.key,
                "value": fact.value,
            }));
        }
        written
    }
}

/// The JSON array in a reply, leniently: anything around it is ignored, and
/// a reply with no parseable array has no facts.
pub fn parse_inline_facts(text: &str) -> Vec<InlineFact> {
    let trimmed = text.trim();
    let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) else {
        return Vec::new();
    };
    if end < start {
        return Vec::new();
    }
    match serde_json::from_str::<Vec<InlineFact>>(&trimmed[start..=end]) {
        Ok(facts) => {
            if !facts.is_empty() {
                info!(count = facts.len(), "Progressive extraction found facts");
            }
            facts
        }
        Err(e) => {
            debug!(error = %e, response = trimmed, "Failed to parse extraction response");
            Vec::new()
        }
    }
}

/// Truncate text for extraction prompts to avoid excessive token usage.
fn truncate_for_extraction(text: &str, max_len: usize) -> &str {
    if text.len() <= max_len {
        text
    } else {
        let mut end = max_len;
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        &text[..end]
    }
}

/// Identity-class `user` fact keys that require first-person evidence in the
/// user's own message before they may be persisted.
const USER_IDENTITY_KEYS: &[&str] = &["name", "full_name", "first_name", "last_name", "nickname"];

/// Returns true when an extracted `user` identity fact (e.g. `name`) has no
/// supporting evidence in the user's own words.
///
/// Guards against the extraction model misattributing a third-party name
/// (someone merely mentioned in conversation) — or a hallucinated one — as
/// the user's identity, which then poisons every future prompt that injects
/// the user profile.
pub fn identity_fact_lacks_user_evidence(
    category: &str,
    key: &str,
    value: &str,
    user_text: &str,
) -> bool {
    if !category.trim().eq_ignore_ascii_case("user") {
        return false;
    }
    let key_norm = key.trim().to_ascii_lowercase();
    if !USER_IDENTITY_KEYS.contains(&key_norm.as_str()) {
        return false;
    }
    let value_norm = value.trim().to_lowercase();
    if value_norm.len() <= 1 {
        return true;
    }
    !user_text.to_lowercase().contains(&value_norm)
}

/// Normalize a fact key for cross-category comparison: lowercase, every run of
/// non-alphanumeric characters collapses to a single `_`, trimmed. So `db_port`,
/// `DB Port` and `db-port` all compare equal.
fn normalize_fact_key_for_match(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    let mut last_sep = false;
    for ch in key.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_sep = false;
        } else if !last_sep {
            out.push('_');
            last_sep = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// True when an extracted fact is merely the assistant recalling/restating a
/// fact we already store — not new information the user provided this turn.
///
/// Progressive extraction is fed the assistant's reply as well as the user's
/// message, so on a recall turn ("what's my DB port?" → "You prefer port 54329")
/// it would otherwise re-extract `54329` and persist it AGAIN — often under a
/// different category than the original — duplicating the fact and letting
/// "forgotten" facts resurface every time they're recalled.
///
/// Fires when the value is absent from the user's message (so it came from the
/// assistant, not the user) AND an active fact already holds that exact value,
/// under EITHER the same canonical key OR — for a distinctive multi-word value —
/// any key. The second case catches recall duplication where the extractor
/// invents a fresh key for a value it just recalled (e.g. "yerba mate" recalled
/// as `late_night_coding_beverage` / `beverage` when it's already stored as
/// `programming_beverage`).
///
/// The distinctive-value gate (≥2 word tokens) keeps this safe: a common
/// single-word value like "blue" or "daily" re-appearing under a new key is left
/// alone, since it may legitimately describe a different attribute. Corrections
/// (the value differs) and user-stated values are never dropped.
fn is_redundant_recall_fact(
    fact_key: &str,
    fact_value: &str,
    user_message: &str,
    existing: &[Fact],
) -> bool {
    let value_norm = fact_value.trim().to_lowercase();
    if value_norm.is_empty() {
        return false;
    }
    // If the user actually stated the value this turn, treat it as new/affirmed.
    if user_message.to_lowercase().contains(&value_norm) {
        return false;
    }
    let key_norm = normalize_fact_key_for_match(fact_key);
    let value_is_distinctive = value_norm
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .count()
        >= 2;
    existing.iter().any(|f| {
        if f.superseded_at.is_some() {
            return false;
        }
        if f.value.trim().to_lowercase() != value_norm {
            return false;
        }
        normalize_fact_key_for_match(&f.key) == key_norm || value_is_distinctive
    })
}

/// True when an active fact already holds this exact canonical key and value
/// under a DIFFERENT category, so upserting would create a second row rather
/// than refresh the existing one.
///
/// This is the turn-level idempotency check: when the agent stored a fact via
/// `remember_fact` this turn, progressive extraction re-reads the same exchange
/// and re-files it under its own category guess (`test` → `technical`). Unlike
/// [`is_redundant_recall_fact`] this does not care who stated the value — an
/// identical active fact is already stored, so the write adds nothing.
/// Same-category writes are left to the upsert, which refreshes in place.
fn duplicates_active_fact_in_other_category(
    category: &str,
    fact_key: &str,
    fact_value: &str,
    existing: &[Fact],
) -> bool {
    let value_norm = fact_value.trim().to_lowercase();
    if value_norm.is_empty() {
        return false;
    }
    let key_norm = normalize_fact_key_for_match(fact_key);
    existing.iter().any(|f| {
        f.superseded_at.is_none()
            && !f.category.trim().eq_ignore_ascii_case(category.trim())
            && f.value.trim().to_lowercase() == value_norm
            && normalize_fact_key_for_match(&f.key) == key_norm
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_fact(category: &str, key: &str, value: &str) -> Fact {
        let now = chrono::Utc::now();
        Fact {
            id: 1,
            category: category.to_string(),
            key: key.to_string(),
            value: value.to_string(),
            source: "test".to_string(),
            created_at: now,
            updated_at: now,
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
    fn normalize_fact_key_matches_across_separators() {
        assert_eq!(
            normalize_fact_key_for_match("local_dev_db_port"),
            "local_dev_db_port"
        );
        assert_eq!(normalize_fact_key_for_match("DB Port"), "db_port");
        assert_eq!(normalize_fact_key_for_match("db-port"), "db_port");
    }

    #[test]
    fn redundant_recall_fact_blocks_restatement_of_known_fact() {
        // The live bug: recalling the port ("You prefer port 54329") must NOT
        // re-store it, especially under a different category.
        let existing = vec![active_fact("preference", "local_dev_db_port", "54329")];
        assert!(is_redundant_recall_fact(
            "local_dev_db_port",
            "54329",
            "what database port did I tell you I prefer?",
            &existing,
        ));
    }

    #[test]
    fn redundant_recall_fact_allows_user_stated_value() {
        // User states the value this turn → legitimately new/affirmed, keep it.
        let existing = vec![active_fact("preference", "local_dev_db_port", "54329")];
        assert!(!is_redundant_recall_fact(
            "local_dev_db_port",
            "54329",
            "my local dev db port is 54329",
            &existing,
        ));
    }

    #[test]
    fn redundant_recall_fact_allows_correction_to_new_value() {
        // Different value (a correction) is never suppressed, even if the user
        // didn't restate it verbatim.
        let existing = vec![active_fact("preference", "local_dev_db_port", "54329")];
        assert!(!is_redundant_recall_fact(
            "local_dev_db_port",
            "9090",
            "actually change my dev db port",
            &existing,
        ));
    }

    #[test]
    fn redundant_recall_fact_blocks_distinctive_value_under_new_key() {
        // The yerba-mate case: recalled value re-extracted under a fresh key.
        // "yerba mate" is distinctive (2 words) → suppress the duplicate.
        let existing = vec![active_fact(
            "preference",
            "programming_beverage",
            "yerba mate",
        )];
        assert!(is_redundant_recall_fact(
            "late_night_coding_beverage",
            "yerba mate",
            "what do I sip on during late-night coding sessions?",
            &existing,
        ));
    }

    #[test]
    fn redundant_recall_fact_allows_common_value_under_new_key() {
        // A common single-word value ("blue") under a different key may describe a
        // genuinely different attribute → must NOT be suppressed.
        let existing = vec![active_fact("user", "car_color", "blue")];
        assert!(!is_redundant_recall_fact(
            "laptop_color",
            "blue",
            "my laptop matches my car",
            &existing,
        ));
    }

    #[test]
    fn redundant_recall_fact_allows_genuinely_new_fact() {
        // No existing fact with this key → it's new information.
        let existing = vec![active_fact("preference", "ui_theme", "dark")];
        assert!(!is_redundant_recall_fact(
            "local_dev_db_port",
            "54329",
            "the assistant mentioned a port",
            &existing,
        ));
    }

    #[test]
    fn redundant_recall_fact_ignores_superseded_existing() {
        // A superseded duplicate must not anchor the guard.
        let mut superseded = active_fact("preference", "local_dev_db_port", "54329");
        superseded.superseded_at = Some(chrono::Utc::now());
        assert!(!is_redundant_recall_fact(
            "local_dev_db_port",
            "54329",
            "what is my port?",
            &[superseded],
        ));
    }

    #[test]
    fn progressive_skips_fact_the_agent_already_stored_this_turn() {
        // The agent saved it via remember_fact; extraction re-files the same
        // user-stated key/value under its own category guess. Must not duplicate,
        // even though the user stated the value (the recall guard allows that).
        let existing = vec![active_fact("test", "qa_color", "teal")];
        let user_text = "remember qa_color=teal";
        assert!(!is_redundant_recall_fact(
            "qa_color", "teal", user_text, &existing
        ));
        assert!(duplicates_active_fact_in_other_category(
            "technical",
            "qa-color",
            "Teal",
            &existing,
        ));
    }

    #[test]
    fn progressive_duplicate_check_keeps_corrections_and_same_category_refresh() {
        let existing = vec![active_fact("test", "qa_color", "teal")];
        // A correction (new value) is new information.
        assert!(!duplicates_active_fact_in_other_category(
            "technical",
            "qa_color",
            "navy",
            &existing,
        ));
        // Same category → the upsert refreshes in place; no duplicate row.
        assert!(!duplicates_active_fact_in_other_category(
            "test", "qa_color", "teal", &existing,
        ));
        // A different key with the same value is a different attribute.
        assert!(!duplicates_active_fact_in_other_category(
            "technical",
            "qa_shade",
            "teal",
            &existing,
        ));
        // Superseded rows don't anchor the check.
        let mut gone = active_fact("test", "qa_color", "teal");
        gone.superseded_at = Some(chrono::Utc::now());
        assert!(!duplicates_active_fact_in_other_category(
            "technical",
            "qa_color",
            "teal",
            &[gone],
        ));
    }

    #[test]
    fn identity_guard_blocks_unevidenced_user_name() {
        // Third-party or hallucinated name not present in the user's words.
        assert!(identity_fact_lacks_user_evidence(
            "user",
            "name",
            "Edison Mendez",
            "Tell me about the beca applicant and their nationality"
        ));
    }

    #[test]
    fn identity_guard_allows_first_person_name() {
        assert!(!identity_fact_lacks_user_evidence(
            "user",
            "name",
            "Jordan Lee",
            "Hi, my name is Jordan Lee and I live in Lisbon"
        ));
        // Case-insensitive match.
        assert!(!identity_fact_lacks_user_evidence(
            "user",
            "Name",
            "jordan lee",
            "I'm Jordan Lee"
        ));
    }

    #[test]
    fn identity_guard_ignores_non_identity_facts() {
        // Other user facts are not gated (dog_name is not the user's identity).
        assert!(!identity_fact_lacks_user_evidence(
            "user",
            "dog_name",
            "Mia",
            "what's the weather?"
        ));
        // Non-user categories are never gated.
        assert!(!identity_fact_lacks_user_evidence(
            "project",
            "name",
            "aidaemon",
            "unrelated text"
        ));
    }

    #[test]
    fn identity_guard_blocks_trivial_values() {
        assert!(identity_fact_lacks_user_evidence("user", "name", "x", "x"));
        assert!(identity_fact_lacks_user_evidence(
            "user", "name", " ", "anything"
        ));
    }

    #[test]
    fn test_inline_fact_deserialization() {
        let json = r#"[{"category":"user","key":"dog_name","value":"Mia"}]"#;
        let facts: Vec<InlineFact> = serde_json::from_str(json).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, "user");
        assert_eq!(facts[0].key, "dog_name");
        assert_eq!(facts[0].value, "Mia");
    }

    #[test]
    fn missing_or_unknown_stance_fields_read_as_not_asserted() {
        let facts = parse_inline_facts(
            r#"[{"category":"user","key":"a","value":"x"},
                {"category":"user","key":"b","value":"x","stance":"definitely","evidence":7},
                {"category":"user","key":"c","value":"x","stance":null,"evidence":null,"restates":""},
                {"category":"user","key":"d","value":"x","stance":"asserted","evidence":"x","restates":" k "}]"#,
        );
        assert_eq!(facts.len(), 4);
        for fact in &facts[..3] {
            assert_eq!(fact.stance, FactStance::Unstated, "{}", fact.key);
            assert!(fact.evidence.is_empty(), "{}", fact.key);
            assert_eq!(fact.restates, None, "{}", fact.key);
            assert_eq!(
                admissibility(fact, "x marks the spot"),
                Err(Inadmissible::NotAsserted(FactStance::Unstated))
            );
        }
        assert_eq!(facts[3].stance, FactStance::Asserted);
        assert_eq!(facts[3].restates.as_deref(), Some("k"));
    }

    // --- write() through a recording store ---

    use std::sync::Mutex;

    use aidaemon_core::testing::{text, ScriptedProvider};
    use async_trait::async_trait;

    #[derive(Debug, Clone, PartialEq)]
    struct Stored {
        key: String,
        value: String,
        source: String,
        excerpt: Option<String>,
    }

    #[derive(Default)]
    struct RecordingFacts {
        existing: Vec<Fact>,
        stored: Mutex<Vec<Stored>>,
        personal: Mutex<Vec<PersonalMemoryWrite>>,
    }

    #[async_trait]
    impl FactStore for RecordingFacts {
        async fn reconcile_personal_memory(
            &self,
            write: &PersonalMemoryWrite,
            _source: &str,
            _source_excerpt: Option<&str>,
            _channel_id: Option<&str>,
            _privacy: FactPrivacy,
        ) -> anyhow::Result<crate::records::PersonalMemoryWriteResult> {
            self.personal.lock().unwrap().push(write.clone());
            Ok(Default::default())
        }
        async fn upsert_fact_with_provenance(
            &self,
            _category: &str,
            key: &str,
            value: &str,
            source: &str,
            _channel_id: Option<&str>,
            _privacy: FactPrivacy,
            _first_seen_at: Option<DateTime<Utc>>,
            source_excerpt: Option<&str>,
        ) -> anyhow::Result<()> {
            self.stored.lock().unwrap().push(Stored {
                key: key.to_string(),
                value: value.to_string(),
                source: source.to_string(),
                excerpt: source_excerpt.map(str::to_string),
            });
            Ok(())
        }
        async fn get_facts(&self, _: Option<&str>) -> anyhow::Result<Vec<Fact>> {
            Ok(self.existing.clone())
        }
    }

    fn exchange<'a>(user_text: &'a str, already_stored: &'a [Fact]) -> Exchange<'a> {
        Exchange {
            user_text,
            assistant_reply: "Noted.",
            channel_id: None,
            already_stored,
        }
    }

    fn extractor(replies: &[&str]) -> (ProgressiveExtractor, Arc<ScriptedProvider>) {
        let provider = Arc::new(ScriptedProvider::new(replies.iter().map(|r| text(*r))));
        (
            ProgressiveExtractor::new(provider.clone(), "mock"),
            provider,
        )
    }

    async fn propose_and_write(
        store: &RecordingFacts,
        exchange: &Exchange<'_>,
        reply: &str,
    ) -> (Vec<Value>, Arc<ScriptedProvider>) {
        let (extractor, provider) = extractor(&[reply]);
        let facts = extractor.propose(exchange).await.unwrap();
        (extractor.write(store, exchange, facts).await, provider)
    }

    #[tokio::test]
    async fn a_question_premise_is_not_stored() {
        // The live bug, translated: the user asks about something they are
        // assumed to own; extraction proposes the premise as a fact.
        let store = RecordingFacts::default();
        let user = "Search your memory: do you know what kind of kayak I own?";
        for stance in ["asked", "presupposed", "assistant_only"] {
            let reply = format!(
                r#"[{{"category":"user","key":"kayak_ownership","value":"Owns a kayak; type unspecified","stance":"{stance}","evidence":"what kind of kayak I own"}}]"#
            );
            let (written, _) = propose_and_write(&store, &exchange(user, &[]), &reply).await;
            assert!(written.is_empty(), "{stance}");
        }
        assert!(store.stored.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn evidence_absent_from_the_user_message_is_not_stored() {
        let store = RecordingFacts::default();
        let user = "Can you summarize the orion launch checklist for me?";
        // Evidence quoted from the assistant's reply, or invented.
        let reply = r#"[{"category":"project","key":"orion_owner","value":"Jordan Lee","stance":"asserted","evidence":"Jordan Lee owns orion"},
                        {"category":"project","key":"orion_phase","value":"beta","stance":"asserted","evidence":""}]"#;
        let (written, _) = propose_and_write(&store, &exchange(user, &[]), reply).await;
        assert!(written.is_empty());
        assert!(store.stored.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_asserted_fact_with_a_faithful_quote_is_stored_with_it() {
        let store = RecordingFacts::default();
        let user = "FYI I work at Acme Corp now, on the orion team.";
        // Case, spacing and a trailing period do not break a quote.
        let reply = r#"[{"category":"user","key":"employer","value":"Acme Corp","stance":"asserted","evidence":"I work at  ACME Corp now."}]"#;
        let (written, _) = propose_and_write(&store, &exchange(user, &[]), reply).await;
        assert_eq!(written.len(), 1);
        let stored = store.stored.lock().unwrap();
        assert_eq!(
            *stored,
            vec![Stored {
                key: "employer".into(),
                value: "Acme Corp".into(),
                source: "progressive".into(),
                excerpt: Some("I work at  ACME Corp now.".into()),
            }]
        );
    }

    #[tokio::test]
    async fn personal_memory_takes_its_direct_statement_from_the_stance() {
        let store = RecordingFacts::default();
        let user = "My sister Alice Rivera lives in Lisbon";
        let reply = r#"[{"category":"user","key":"sister","value":"Alice Rivera","stance":"asserted","evidence":"My sister Alice Rivera lives in Lisbon","personal_memory":{"direct_user_statement":false}},
                        {"category":"user","key":"brother","value":"Jordan Lee","stance":"presupposed","evidence":"My sister","personal_memory":{"direct_user_statement":true}}]"#;
        let (written, _) = propose_and_write(&store, &exchange(user, &[]), reply).await;
        assert_eq!(written.len(), 1);
        let personal = store.personal.lock().unwrap();
        assert_eq!(personal.len(), 1);
        assert!(personal[0].direct_user_statement);
    }

    fn agent_fact(key: &str, value: &str, updated_at: DateTime<Utc>) -> Fact {
        Fact {
            source: "agent".to_string(),
            updated_at,
            ..active_fact("project", key, value)
        }
    }

    #[tokio::test]
    async fn a_fact_the_agent_stored_this_run_is_not_extracted_again() {
        let run_started_at = Utc::now() - chrono::Duration::seconds(5);
        let before_run = run_started_at - chrono::Duration::hours(1);
        let store = RecordingFacts {
            existing: vec![
                agent_fact(
                    "orion_staging_deployment_schedule",
                    "Project orion deploys to staging every Tuesday.",
                    Utc::now(),
                ),
                // Written before the run, or by extraction: not this run's.
                agent_fact("orion_owner", "Alice Rivera", before_run),
                Fact {
                    source: "progressive".into(),
                    ..active_fact("project", "orion_repo", "acme/orion")
                },
            ],
            ..Default::default()
        };
        let already = owner_writes_during_run(&store, run_started_at).await;
        assert_eq!(
            already.iter().map(|f| f.key.as_str()).collect::<Vec<_>>(),
            vec!["orion_staging_deployment_schedule"]
        );
        let user = "Remember: project orion deploys to staging every Tuesday. Also I prefer short standups.";
        let reply = r#"[{"category":"project","key":"orion_deployment_schedule","value":"Deploys to staging every Tuesday","stance":"asserted","evidence":"project orion deploys to staging every Tuesday","restates":"orion_staging_deployment_schedule"},
                        {"category":"preference","key":"standup_length","value":"short","stance":"asserted","evidence":"I prefer short standups"}]"#;
        let (written, provider) = propose_and_write(&store, &exchange(user, &already), reply).await;
        // The model was shown what the agent stored.
        let prompt = &provider.requests()[0].messages[1];
        let prompt = serde_json::to_string(prompt).unwrap();
        assert!(prompt.contains("ALREADY STORED"));
        assert!(prompt.contains("project/orion_staging_deployment_schedule"));
        assert!(!prompt.contains("orion_owner"));
        // The restatement is dropped; the new fact from the same turn is kept.
        assert_eq!(written.len(), 1);
        let stored = store.stored.lock().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].key, "standup_length");
    }

    #[tokio::test]
    async fn without_agent_writes_extraction_is_unchanged() {
        let store = RecordingFacts {
            existing: vec![agent_fact(
                "orion_owner",
                "Alice Rivera",
                Utc::now() - chrono::Duration::hours(2),
            )],
            ..Default::default()
        };
        let already =
            owner_writes_during_run(&store, Utc::now() - chrono::Duration::seconds(5)).await;
        assert!(already.is_empty());
        let user = "Project orion deploys to staging every Tuesday.";
        let reply = r#"[{"category":"project","key":"orion_deployment_schedule","value":"Deploys to staging every Tuesday","stance":"asserted","evidence":"Project orion deploys to staging every Tuesday"}]"#;
        let (written, provider) = propose_and_write(&store, &exchange(user, &already), reply).await;
        let prompt = serde_json::to_string(&provider.requests()[0].messages[1]).unwrap();
        assert!(!prompt.contains("ALREADY STORED"));
        assert_eq!(written.len(), 1);
        assert_eq!(
            store.stored.lock().unwrap()[0].key,
            "orion_deployment_schedule"
        );
    }

    #[test]
    fn test_inline_fact_empty_array() {
        let json = "[]";
        let facts: Vec<InlineFact> = serde_json::from_str(json).unwrap();
        assert!(facts.is_empty());
    }

    #[test]
    fn trivial_messages_are_not_worth_extracting() {
        for trivial in [
            "ok",
            "thanks",
            "yes",
            "lol",
            "short",
            "thank you",
            "understood",
            "👍",
        ] {
            assert!(!worth_extracting(trivial), "{trivial}");
        }
        assert!(worth_extracting(
            "My dog's name is Mia and she's a golden retriever"
        ));
        assert!(worth_extracting(
            "I work at Acme Corp in the engineering department"
        ));
    }

    #[test]
    fn only_the_owner_off_public_platforms_is_extracted() {
        assert!(may_extract(UserRole::Owner, ChannelVisibility::Private));
        assert!(!may_extract(UserRole::Guest, ChannelVisibility::Private));
        assert!(!may_extract(
            UserRole::Owner,
            ChannelVisibility::PublicExternal
        ));
    }

    #[test]
    fn replies_are_parsed_leniently() {
        let facts = parse_inline_facts(
            "Here you go: [{\"category\":\"user\",\"key\":\"dog_name\",\"value\":\"Mia\"}] done",
        );
        assert_eq!(facts.len(), 1);
        assert!(parse_inline_facts("nothing here").is_empty());
        assert!(parse_inline_facts("] backwards [").is_empty());
    }
}
