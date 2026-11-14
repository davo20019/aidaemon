//! Fact consolidation: a background pass over owner conversation history
//! that asks a model for durable facts the turn-by-turn extraction missed.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use aidaemon_core::{ChannelVisibility, Message, ModelRequest, Provider};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::history::{ConsolidationCursors, ConversationSource, TurnRole};
use crate::progressive::{identity_fact_lacks_user_evidence, worth_extracting};
use crate::records::{ExtractedMemoryGraph, FactPrivacy, PersonalMemoryWrite};
use crate::stores::FactStore;
use crate::task_learning::{FACT_LIFETIME_INSTRUCTION, FactLifetime};

/// Turns read per session per pass.
const TURNS_PER_PASS: usize = 300;
/// Candidate messages sent to the model per session per pass.
const CANDIDATES_PER_SESSION: usize = 30;

const CONSOLIDATION_PROMPT: &str = "You are a memory consolidation system. Given a conversation excerpt, \
            extract durable facts worth remembering long-term. Output ONLY a JSON array. Each item has \
            category, key, value, privacy, and an optional graph or personal_memory object. Graph entities have local_id, \
            name, entity_type, aliases, confidence. Relationships have source_id, target_id, relation, \
            confidence. Use owner for the user. Include graph data only when directly supported by the \
            owner's words; never infer entity types from category or key names. For names, aliases, handles, dates of birth, residence, accounts, or family relationships, emit personal_memory with entities, aliases, facts, relationships, direct_user_statement, and correction instead of dynamic person-name keys. Use owner as the owner's local_id, stable entity references, ISO dates, and typed relationships PARENT_OF, CHILD_OF, LIVES_WITH, LIVES_IN, USES_HANDLE, HAS_ACCOUNT. \
            Categories:\n\
            - user: Personal info about the OWNER (name, location, job)\n\
            - preference: Tool, workflow, and communication preferences\n\
            - project: Projects, tech stacks, goals\n\
            - technical: Environment details, installed tools\n\
            - relationship: Communication patterns with the AI\n\
            - behavior: Observed tool-usage patterns and recurring workflows\n\
            - people: Information about OTHER individuals mentioned or participating in conversation\n\n\
            ONLY learn from owner-authenticated user messages. NEVER extract or store facts from guest/public speakers, and NEVER treat another person's self-description as owner data. \
            A name mentioned in conversation belongs in \"people\" — it is NEVER the owner's \"name\" unless the owner explicitly states it as their own.\n\n\
            For \"behavior\", look for:\n\
            - Which tools the user prefers for specific tasks\n\
            - Recurring workflows or action sequences\n\
            - Types of tasks frequently delegated\n\n\
            For \"people\", extract:\n\
            - Names and relationships mentioned (e.g., \"my wife Alice\", \"coworker Juan\")\n\
            - Personal details about others (birthdays, preferences, interests, jobs)\n\
            - Important dates related to people\n\
            - Always include personal_memory; never encode a person's name in key\n\
            - Give each person a local entity ID, store nicknames/handles as aliases, attach dates/properties as facts, and model family/residence links as relationships\n\
            - Keep separate people as separate entities and normalize dates to YYYY-MM-DD\n\
            - Set person_name only as a legacy display hint, not as an identity key\n\
            - NEVER extract health info, financial details, political opinions, or religious beliefs about people\n\
            Example: represent \"My daughter Alice, nicknamed Allie, was born March 15, 2015\" with one Alice entity, an Allie nickname alias, a 2015-03-15 birth_date fact, and PARENT_OF(owner, Alice)\n\n\
            Also classify each fact's privacy:\n\
            - \"global\": General facts useful anywhere (name, job, timezone, tech preferences)\n\
            - \"channel\": Context-specific facts from this conversation\n\
            - \"private\": Sensitive personal info the user would want kept private\n\
            Default to \"channel\" if unsure. People facts should default to \"private\".\n\n\
            Only extract facts useful in future conversations. If nothing worth remembering, return [].";

/// Consolidates durable facts from owner conversation history with a model.
#[derive(Clone)]
pub struct FactConsolidator {
    provider: Arc<dyn Provider>,
    model: String,
}

/// What one consolidation pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConsolidationReport {
    /// Sessions whose facts were consolidated (not skipped as isolated or
    /// public-external), for follow-up passes over the same sessions.
    pub consolidated_sessions: Vec<String>,
    pub facts_seen: usize,
}

impl FactConsolidator {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// Consolidate history older than one hour before `now`. Only
    /// owner-authored user messages and tool results are read: assistant
    /// text is skipped so a hallucinated value can never overwrite what the
    /// owner said. A session's cursor advances when its batch is done, or
    /// when there is nothing to learn from it; a failed model call or an
    /// unparseable reply leaves it, so the batch is retried.
    pub async fn consolidate<S>(
        &self,
        source: &dyn ConversationSource,
        cursors: &dyn ConsolidationCursors,
        store: &S,
        now: DateTime<Utc>,
    ) -> anyhow::Result<ConsolidationReport>
    where
        S: FactStore + ?Sized,
    {
        let cutoff = now - chrono::Duration::hours(1);
        let candidates = source.sessions_with_turns_before(cutoff).await?;
        let mut report = ConsolidationReport::default();
        if candidates.is_empty() {
            return Ok(report);
        }

        // session -> candidate messages (role, content), and the last turn
        // id seen this pass (for cursor advancement).
        let mut sessions: HashMap<String, Vec<(TurnRole, String)>> = HashMap::new();
        let mut last_seen: HashMap<String, i64> = HashMap::new();
        for session_id in candidates {
            let cursor = cursors.cursor(&session_id).await;
            if !source.owner_only_since(&session_id, cursor, cutoff).await? {
                if let Some(last) = source.last_turn_id(&session_id, cursor, cutoff).await? {
                    let _ = cursors.set_cursor(&session_id, last).await;
                }
                continue;
            }
            let turns = source
                .turns(&session_id, cursor, cutoff, TURNS_PER_PASS)
                .await?;
            let Some(last) = turns.last().map(|turn| turn.id) else {
                continue;
            };
            last_seen.insert(session_id.clone(), last);
            let entry = sessions.entry(session_id).or_default();
            for turn in turns {
                if turn.role == TurnRole::Assistant {
                    continue;
                }
                let Some(content) = turn.content.filter(|c| !c.trim().is_empty()) else {
                    continue;
                };
                // Every message long enough to hold a fact is a candidate;
                // the model decides what is durable.
                if worth_extracting(&content) && entry.len() < CANDIDATES_PER_SESSION {
                    entry.push((turn.role, content));
                }
            }
        }

        info!(
            "Consolidation: processing {} messages from {} sessions (event-native)",
            sessions.values().map(Vec::len).sum::<usize>(),
            sessions.len()
        );

        let system_prompt = format!("{CONSOLIDATION_PROMPT}\n\n{FACT_LIFETIME_INSTRUCTION}");
        let mut skipped: HashSet<String> = HashSet::new();
        for (session_id, messages) in &sessions {
            // Defense in depth right before the model call: provenance may
            // have arrived after selection.
            if source.is_memory_isolated(session_id).await? {
                skipped.insert(session_id.clone());
                info!(
                    session_id = session_id.as_str(),
                    "Skipping memory fact consolidation for isolated worker session"
                );
                continue;
            }
            // Untrusted public platforms never teach durable facts.
            if source.visibility(session_id).await == Some(ChannelVisibility::PublicExternal) {
                skipped.insert(session_id.clone());
                if let Some(last) = last_seen.get(session_id) {
                    let _ = cursors.set_cursor(session_id, *last).await;
                }
                info!(
                    session_id = session_id.as_str(),
                    "Skipping memory fact consolidation for PublicExternal session"
                );
                continue;
            }

            // Nothing worth a model call: move the cursor on, so low-signal
            // history is not rescanned every pass.
            if messages.is_empty() {
                if let Some(last) = last_seen.get(session_id) {
                    let _ = cursors.set_cursor(session_id, *last).await;
                }
                continue;
            }

            let conversation = messages
                .iter()
                .map(|(role, content)| format!("{}: {content}", role.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            // The owner's own words, for identity evidence.
            let user_text = messages
                .iter()
                .filter(|(role, _)| *role == TurnRole::User)
                .map(|(_, content)| content.as_str())
                .collect::<Vec<_>>()
                .join("\n");

            let request = ModelRequest {
                model: self.model.clone(),
                messages: vec![
                    Message::system(system_prompt.as_str()),
                    Message::user(conversation),
                ],
                ..Default::default()
            };
            let response = match self.provider.complete(&request).await {
                Ok(response) => response,
                Err(e) => {
                    warn!("LLM call failed during consolidation for session {session_id}: {e}");
                    continue; // Not consolidated: retried next pass.
                }
            };
            let facts = match parse_consolidation_response(&response.text) {
                Ok(facts) => facts,
                Err(e) => {
                    warn!(
                        "Failed to parse consolidation response for session {session_id}: {e} — events will be retried next cycle"
                    );
                    continue;
                }
            };
            report.facts_seen += facts.len();
            let channel_id = source.channel_id(session_id);
            store_consolidated(store, &facts, &user_text, channel_id.as_deref()).await;
            info!(
                "Consolidation: extracted {} facts from session {session_id}",
                facts.len()
            );
            if let Some(last) = last_seen.get(session_id) {
                let _ = cursors.set_cursor(session_id, *last).await;
            }
        }

        let mut consolidated: Vec<String> = sessions
            .into_keys()
            .filter(|session| !skipped.contains(session))
            .collect();
        consolidated.sort();
        report.consolidated_sessions = consolidated;
        Ok(report)
    }
}

/// Write consolidated facts through the store, which owns deduplication,
/// the owner-overwrite guard, deletion tombstones and embeddings.
async fn store_consolidated<S>(
    store: &S,
    facts: &[ExtractedFact],
    user_text: &str,
    channel_id: Option<&str>,
) where
    S: FactStore + ?Sized,
{
    let excerpt = crate::text::truncate_chars(user_text, 200);
    for fact in facts {
        if let Some(personal) = &fact.personal_memory {
            let privacy = fact
                .privacy
                .as_deref()
                .map(FactPrivacy::from_str_lossy)
                .unwrap_or(FactPrivacy::Private);
            if let Err(error) = store
                .reconcile_personal_memory(
                    personal,
                    "consolidation",
                    Some(excerpt.as_str()),
                    channel_id,
                    privacy,
                )
                .await
            {
                warn!(%error, "Failed to reconcile consolidated personal memory");
            }
            continue;
        }
        if fact.lifetime.is_transient() {
            debug!(key = fact.key, "Skipping transient consolidated fact");
            continue;
        }
        // People are recorded only as structured personal memory.
        if fact.category == "people" {
            warn!(
                key = fact.key,
                "Skipping unstructured people fact; extractor must emit personal_memory"
            );
            continue;
        }
        // Identity facts must appear in the owner's own messages.
        if identity_fact_lacks_user_evidence(&fact.category, &fact.key, &fact.value, user_text) {
            warn!(
                "Skipping consolidated user identity fact [{}={}] — no evidence in user messages",
                fact.key, fact.value
            );
            continue;
        }
        let privacy = fact
            .privacy
            .as_deref()
            .map(FactPrivacy::from_str_lossy)
            .unwrap_or(FactPrivacy::Channel);
        if let Err(e) = store
            .upsert_fact_with_provenance(
                &fact.category,
                &fact.key,
                &fact.value,
                "consolidation",
                channel_id,
                privacy,
                Some(Utc::now()),
                Some(excerpt.as_str()),
            )
            .await
        {
            warn!(
                "Failed to upsert consolidated fact [{}/{}]: {e}",
                fact.category, fact.key
            );
            continue;
        }
        if let Err(e) = store.refresh_fact_memory(&fact.category, &fact.key).await {
            debug!(error = %e, key = fact.key, "Failed to refresh consolidated fact projection");
        } else if let Err(e) = store
            .project_extracted_fact_graph(&fact.category, &fact.key, excerpt.as_str(), &fact.graph)
            .await
        {
            debug!(error = %e, key = fact.key, "Failed to project consolidated graph");
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ExtractedFact {
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub category: String,
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub key: String,
    #[serde(default, deserialize_with = "deserialize_nullable_string")]
    pub value: String,
    pub privacy: Option<String>,
    #[serde(default, deserialize_with = "deserialize_lossy_graph")]
    pub graph: ExtractedMemoryGraph,
    #[serde(default, deserialize_with = "deserialize_lossy_personal_memory")]
    pub personal_memory: Option<PersonalMemoryWrite>,
    #[serde(default)]
    pub lifetime: FactLifetime,
}

fn deserialize_nullable_string<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

fn deserialize_lossy_graph<'de, D>(deserializer: D) -> Result<ExtractedMemoryGraph, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    Ok(serde_json::from_value(value).unwrap_or_default())
}

fn deserialize_lossy_personal_memory<'de, D>(
    deserializer: D,
) -> Result<Option<PersonalMemoryWrite>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    if value.is_null() {
        return Ok(None);
    }
    Ok(serde_json::from_value(value).ok())
}

/// Parse a consolidation reply: a JSON array (code fences allowed). One
/// malformed item is skipped; all malformed is an error so the batch is
/// retried. Facts with an empty or placeholder field are dropped.
pub fn parse_consolidation_response(text: &str) -> anyhow::Result<Vec<ExtractedFact>> {
    // Try to find JSON array in the response (LLM may include markdown fences)
    let trimmed = text.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            trimmed
        }
    } else {
        trimmed
    };

    let raw_facts: Vec<serde_json::Value> = serde_json::from_str(json_str)?;
    let mut malformed = 0_usize;
    let mut facts = Vec::with_capacity(raw_facts.len());
    for (index, raw_fact) in raw_facts.into_iter().enumerate() {
        match serde_json::from_value::<ExtractedFact>(raw_fact) {
            Ok(fact) => facts.push(fact),
            Err(error) => {
                malformed += 1;
                warn!(
                    fact_index = index,
                    %error,
                    "Skipping one malformed memory fact from consolidation response"
                );
            }
        }
    }

    if facts.is_empty() && malformed > 0 {
        anyhow::bail!("all {malformed} memory facts were malformed");
    }

    // Filter out facts with empty/meaningless values to prevent hallucination
    let facts = facts
        .into_iter()
        .filter(|f| {
            [&f.category, &f.key, &f.value].into_iter().all(|field| {
                let field = field.trim();
                !field.is_empty()
                    && !field.eq_ignore_ascii_case("none")
                    && !field.eq_ignore_ascii_case("null")
                    && !field.eq_ignore_ascii_case("n/a")
                    && !field.eq_ignore_ascii_case("unknown")
            })
        })
        .collect();
    Ok(facts)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use aidaemon_core::ProviderError;
    use aidaemon_core::testing::{ScriptedProvider, text};
    use async_trait::async_trait;

    use super::*;
    use crate::history::HistoryTurn;
    use crate::records::Fact;

    #[test]
    fn parser_isolates_bad_items_and_nullable_fields() {
        let response = r#"```json
        [
          {
            "category": "preferences",
            "key": "editor",
            "value": "Neovim",
            "privacy": null,
            "graph": {
              "entities": [{"local_id":"editor","name":null,"entity_type":"tool"}]
            }
          },
          {"category": null, "key": "placeholder", "value": null},
          42,
          {"category": "profile", "key": "timezone", "value": "America/New_York"}
        ]
        ```"#;
        let facts = parse_consolidation_response(response).unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].key, "editor");
        assert!(facts[0].graph.entities.is_empty());
        assert_eq!(facts[1].key, "timezone");
    }

    #[test]
    fn parser_retries_when_every_item_is_structurally_bad() {
        let error = parse_consolidation_response(r#"[{"category": 7}, false]"#)
            .expect_err("an entirely malformed batch must be retried");
        assert!(
            error
                .to_string()
                .contains("all 2 memory facts were malformed")
        );
    }

    /// A one-session history: the owner speaks unless `guest` is set.
    struct History {
        turns: Vec<HistoryTurn>,
        owner_only: bool,
        visibility: Option<ChannelVisibility>,
    }

    fn turn(id: i64, role: TurnRole, content: &str) -> HistoryTurn {
        HistoryTurn {
            id,
            role,
            content: Some(content.to_string()),
            created_at: Utc::now() - chrono::Duration::hours(2),
        }
    }

    #[async_trait]
    impl ConversationSource for History {
        async fn sessions_with_turns_before(
            &self,
            _: DateTime<Utc>,
        ) -> anyhow::Result<Vec<String>> {
            Ok(vec!["telegram:synthetic-chat".to_string()])
        }
        async fn is_memory_isolated(&self, _: &str) -> anyhow::Result<bool> {
            Ok(false)
        }
        async fn owner_only_since(
            &self,
            _: &str,
            _: i64,
            _: DateTime<Utc>,
        ) -> anyhow::Result<bool> {
            Ok(self.owner_only)
        }
        async fn last_turn_id(
            &self,
            _: &str,
            after: i64,
            _: DateTime<Utc>,
        ) -> anyhow::Result<Option<i64>> {
            Ok(self
                .turns
                .iter()
                .map(|t| t.id)
                .filter(|id| *id > after)
                .max())
        }
        async fn turns(
            &self,
            _: &str,
            after: i64,
            _: DateTime<Utc>,
            limit: usize,
        ) -> anyhow::Result<Vec<HistoryTurn>> {
            Ok(self
                .turns
                .iter()
                .filter(|t| t.id > after)
                .take(limit)
                .cloned()
                .collect())
        }
        async fn visibility(&self, _: &str) -> Option<ChannelVisibility> {
            self.visibility
        }
        fn channel_id(&self, _: &str) -> Option<String> {
            Some("telegram:synthetic-chat".to_string())
        }
    }

    #[derive(Default)]
    struct Cursors(Mutex<HashMap<String, i64>>);

    #[async_trait]
    impl ConsolidationCursors for Cursors {
        async fn cursor(&self, session_id: &str) -> i64 {
            self.0.lock().unwrap().get(session_id).copied().unwrap_or(0)
        }
        async fn set_cursor(&self, session_id: &str, turn_id: i64) -> anyhow::Result<()> {
            self.0
                .lock()
                .unwrap()
                .insert(session_id.to_string(), turn_id);
            Ok(())
        }
    }

    #[derive(Default)]
    struct Facts(Mutex<Vec<(String, String, FactPrivacy)>>);

    #[async_trait]
    impl FactStore for Facts {
        async fn upsert_fact_with_provenance(
            &self,
            _category: &str,
            key: &str,
            value: &str,
            source: &str,
            _channel_id: Option<&str>,
            privacy: FactPrivacy,
            _first_seen_at: Option<DateTime<Utc>>,
            _source_excerpt: Option<&str>,
        ) -> anyhow::Result<()> {
            assert_eq!(source, "consolidation");
            self.0
                .lock()
                .unwrap()
                .push((key.to_string(), value.to_string(), privacy));
            Ok(())
        }
        async fn get_facts(&self, _: Option<&str>) -> anyhow::Result<Vec<Fact>> {
            Ok(Vec::new())
        }
    }

    fn owner_history() -> History {
        History {
            turns: vec![
                turn(
                    1,
                    TurnRole::User,
                    "I moved to Lisbon last month and work remotely",
                ),
                turn(
                    2,
                    TurnRole::Assistant,
                    "Your cat is called Pixel, as you told me before",
                ),
                turn(3, TurnRole::User, "ok"),
            ],
            owner_only: true,
            visibility: Some(ChannelVisibility::Private),
        }
    }

    const REPLY: &str = r#"[{"category":"user","key":"city","value":"Lisbon","privacy":"global"},
        {"category":"user","key":"name","value":"Jordan Lee"}]"#;

    #[tokio::test]
    async fn owner_facts_are_stored_and_the_cursor_advances() {
        let provider = Arc::new(ScriptedProvider::new([text(REPLY)]));
        let (cursors, facts) = (Cursors::default(), Facts::default());
        let report = FactConsolidator::new(provider.clone(), "mock")
            .consolidate(&owner_history(), &cursors, &facts, Utc::now())
            .await
            .unwrap();

        assert_eq!(report.consolidated_sessions, ["telegram:synthetic-chat"]);
        // The name is not in the owner's words: refused.
        let stored = facts.0.lock().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].0, "city");
        assert_eq!(stored[0].2, FactPrivacy::Global);
        assert_eq!(cursors.0.lock().unwrap()["telegram:synthetic-chat"], 3);
        // The assistant's words and the too-short message never reach the model.
        let sent = &provider.requests()[0].messages[1].content;
        assert!(sent.contains("Lisbon") && !sent.contains("Pixel") && !sent.contains("user: ok"));
    }

    #[tokio::test]
    async fn a_failed_model_call_leaves_the_batch_for_the_next_pass() {
        let provider = Arc::new(ScriptedProvider::with_results([Err(ProviderError::Auth(
            "synthetic outage".into(),
        ))]));
        let (cursors, facts) = (Cursors::default(), Facts::default());
        FactConsolidator::new(provider, "mock")
            .consolidate(&owner_history(), &cursors, &facts, Utc::now())
            .await
            .unwrap();
        assert!(facts.0.lock().unwrap().is_empty());
        assert!(cursors.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn guests_and_public_platforms_teach_nothing() {
        for history in [
            History {
                owner_only: false,
                ..owner_history()
            },
            History {
                visibility: Some(ChannelVisibility::PublicExternal),
                ..owner_history()
            },
        ] {
            let provider = Arc::new(ScriptedProvider::new([text(REPLY)]));
            let (cursors, facts) = (Cursors::default(), Facts::default());
            let report = FactConsolidator::new(provider.clone(), "mock")
                .consolidate(&history, &cursors, &facts, Utc::now())
                .await
                .unwrap();
            assert!(report.consolidated_sessions.is_empty());
            assert!(provider.requests().is_empty(), "no model call");
            assert!(facts.0.lock().unwrap().is_empty());
            // Either way the history is not rescanned.
            assert_eq!(cursors.0.lock().unwrap()["telegram:synthetic-chat"], 3);
        }
    }
}
