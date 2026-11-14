//! [`FullRecall`]: the full memory system's recall, rendered as one system
//! note by `aidaemon_policies::MemoryInjection`. It reads the SQLite memory
//! (facts, episodes, procedures, people, the core profile) and scopes every
//! section by the turn's audience: personal memory only in the owner's
//! private chats, nothing on a public channel with outsiders.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use aidaemon_core::ChannelVisibility;
use async_trait::async_trait;
use serde_json::json;
use tracing::{info, warn};

use crate::records::Fact;
use crate::render::{
    MemoryContext, RecallGoal, build_critical_facts_prompt_block, extract_critical_fact_summary,
    render_memory_section,
};
use crate::stores::MemoryStores;

/// Identity/profile categories pinned as critical facts in owner DMs.
pub const IDENTITY_CATEGORIES: &[&str] = &[
    "identity",
    "personal",
    "profile",
    "user",
    "assistant",
    "bot",
    "relationship",
    "preference",
    "family",
];

/// What recall knows about one turn, as the host decided it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallTurn {
    pub session_id: String,
    pub task_id: String,
    /// The user's message: recall is scoped to the current request. `None`
    /// recalls for the latest user message of the request instead.
    pub user_text: Option<String>,
    /// The owner wrote the message.
    pub owner: bool,
    /// The owner wrote it in a private chat: personal memory (identity,
    /// goals, profile, people) may be shown.
    pub personal: bool,
    pub visibility: ChannelVisibility,
    pub channel_id: Option<String>,
    /// The speaker's platform id, to recognise a known person.
    pub sender_id: Option<String>,
    /// The turn may read memory at all.
    pub allowed: bool,
    /// Sub-agent depth: the core profile is for the top-level run only.
    pub depth: usize,
}

/// A recall decision the host may record for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallDecision {
    /// Which entities the core profile selected (id and content hash each).
    CoreProfileSelection,
    /// What recall fetched and rendered.
    MemoryRetrieval,
}

/// What [`FullRecall`] needs from its host beyond the memory stores.
#[async_trait]
pub trait RecallHost: Send + Sync {
    /// The owner's active personal goals (owner private chats only).
    async fn active_goals(&self) -> anyhow::Result<Vec<RecallGoal>>;

    /// Whether people intelligence is on.
    async fn people_enabled(&self) -> anyhow::Result<bool>;

    /// The host offers `manage_automations` for scheduled tasks and
    /// reminders, so the memory guide may point at it.
    fn offers_scheduled_goals(&self) -> bool {
        false
    }

    /// Whether [`RecallHost::relevance`] is used at all.
    fn relevance_enabled(&self) -> bool {
        false
    }

    /// For each `(key, question)`, the probability that the answer is yes,
    /// for the questions the host's judge answered. `message` is what the
    /// questions are about. Never fails: no answer leaves a key out.
    async fn relevance(
        &self,
        _turn: &RecallTurn,
        _message: String,
        _questions: Vec<(String, String)>,
    ) -> HashMap<String, f32> {
        HashMap::new()
    }

    /// Record a recall decision for diagnostics.
    async fn decision(
        &self,
        _turn: &RecallTurn,
        _kind: RecallDecision,
        _summary: String,
        _metadata: serde_json::Value,
    ) {
    }
}

/// Settings of [`FullRecall`].
#[derive(Debug, Clone, Default)]
pub struct RecallSettings {
    pub max_facts: usize,
    /// The configured assistant name, pinned among the critical facts.
    pub assistant_name: Option<String>,
}

/// State shared by every turn's [`FullRecall`].
pub struct RecallSource {
    store: Arc<dyn MemoryStores>,
    settings: RecallSettings,
    /// Per session, the entities the core profile selected last, so the
    /// selection stays stable across turns.
    core_profile_ids: tokio::sync::RwLock<HashMap<String, Vec<String>>>,
}

impl RecallSource {
    pub fn new(store: Arc<dyn MemoryStores>, settings: RecallSettings) -> Self {
        Self {
            store,
            settings,
            core_profile_ids: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

/// A recall read that fails soft: the section renders empty, and the error is
/// logged so a broken store does not look like an empty memory.
fn or_logged<T: Default, E: std::fmt::Display>(result: Result<T, E>, section: &str) -> T {
    result.unwrap_or_else(|error| {
        warn!(section, %error, "memory recall read failed; section left empty");
        T::default()
    })
}

/// Recall over the memory stores for one turn. It already knows the channel,
/// role and memory policy, so the `Recall` contract's session and query are
/// enough.
pub struct FullRecall {
    source: Arc<RecallSource>,
    host: Arc<dyn RecallHost>,
    turn: RecallTurn,
    /// The turn's memory block, started by [`Self::prefetch`] before the
    /// first model call asks for it.
    pub(super) prefetched: std::sync::Mutex<Option<tokio::task::JoinHandle<String>>>,
}

impl FullRecall {
    pub fn new(source: Arc<RecallSource>, host: Arc<dyn RecallHost>, turn: RecallTurn) -> Self {
        Self {
            source,
            host,
            turn,
            prefetched: std::sync::Mutex::new(None),
        }
    }

    /// Starts rendering the turn's memory block now, so the first model
    /// call finds it ready instead of waiting for every lookup. Only a turn
    /// that names its request can: the block is rendered for that text,
    /// whatever query the model call later passes. Outside a Tokio runtime
    /// nothing starts and the block renders on demand as before.
    pub fn prefetch(self) -> Self {
        if self.turn.user_text.is_none() {
            return self;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return self;
        };
        let worker = Self::new(self.source.clone(), self.host.clone(), self.turn.clone());
        let handle = runtime.spawn(async move { worker.render_block("").await });
        if let Ok(mut prefetched) = self.prefetched.lock() {
            *prefetched = Some(handle);
        }
        self
    }

    /// The prefetched block, once; `None` when none was started or it
    /// failed, so the caller renders it itself.
    async fn take_prefetched(&self) -> Option<String> {
        let handle = self.prefetched.lock().ok()?.take()?;
        handle.await.ok()
    }

    /// Starts the recall filter over `facts`, when the host judges
    /// relevance. Resolves to the ids of facts to drop: only an answer with
    /// P(relevant) below 0.5 drops one, and facts in identity categories are
    /// never asked about, so they are always kept. On any failure nothing is
    /// dropped.
    fn spawn_fact_filter(
        &self,
        query: &str,
        facts: &[Fact],
    ) -> Option<tokio::task::JoinHandle<Option<HashSet<i64>>>> {
        let host = self.host.clone();
        if !host.relevance_enabled() {
            return None;
        }
        let message = format!(
            "User message:\n{}",
            crate::text::truncate_chars(query, 3000)
        );
        let mut questions = Vec::new();
        let mut asked = std::collections::BTreeMap::new();
        for fact in facts
            .iter()
            .filter(|fact| !IDENTITY_CATEGORIES.contains(&fact.category.as_str()))
        {
            let key = format!("f{}", asked.len());
            questions.push((
                key.clone(),
                format!(
                    "Would this remembered fact help answer or carry out the user's message? Fact: [{}] {}: {}",
                    fact.category,
                    fact.key,
                    crate::text::truncate_chars(&fact.value, 300)
                ),
            ));
            asked.insert(key, fact.id);
        }
        if asked.is_empty() {
            return None;
        }
        let turn = self.turn.clone();
        Some(tokio::spawn(async move {
            let answers = host.relevance(&turn, message, questions).await;
            Some(
                asked
                    .iter()
                    .filter(|(key, _)| answers.get(*key).is_some_and(|p| *p < 0.5))
                    .map(|(_, id)| *id)
                    .collect(),
            )
        }))
    }

    /// Core profile: owner DMs at the top level, with a per-session stable
    /// entity selection.
    async fn core_profile(&self, people_enabled: bool) -> String {
        let turn = &self.turn;
        let source = &self.source;
        if !(turn.allowed && turn.personal && turn.depth == 0) {
            return String::new();
        }
        let cached_ids = source
            .core_profile_ids
            .read()
            .await
            .get(&turn.session_id)
            .cloned();
        let (profile_str, new_ids, digest) = or_logged(
            crate::profile::build_core_profile(source.store.as_ref(), cached_ids, people_enabled)
                .await,
            "core_profile",
        );
        if let Some(ids) = new_ids {
            source
                .core_profile_ids
                .write()
                .await
                .insert(turn.session_id.clone(), ids);
        }
        // The selection digest (id + content hash per entity) makes a
        // profile change self-explaining across turns.
        if !digest.is_empty() {
            self.host
                .decision(
                    turn,
                    RecallDecision::CoreProfileSelection,
                    format!("Core profile selection: {} entities", digest.len()),
                    json!({
                        "code": "core_profile_digest",
                        "count": digest.len(),
                        "entities": digest
                            .iter()
                            .map(|(id, h)| json!({ "id": id, "h": h }))
                            .collect::<Vec<_>>(),
                    }),
                )
                .await;
        }
        profile_str
    }

    /// The memory block: critical facts, core profile, then the query-ranked
    /// memory sections. Channel, role and policy scoping are applied here.
    /// `query` is the request's latest user message, used when the turn
    /// gives no user text.
    pub async fn render_block(&self, query: &str) -> String {
        let source = &self.source;
        let turn = &self.turn;
        let store = &source.store;
        let host = &self.host;
        let allowed = turn.allowed;
        // Memory retrieval is current-request-scoped.
        let query = turn.user_text.as_deref().unwrap_or(query);
        let inject_personal = turn.personal;
        let public_external = matches!(turn.visibility, ChannelVisibility::PublicExternal);

        // Facts: always channel-scoped semantic retrieval, so unrelated facts
        // never bleed into prompts for unrelated queries.
        let recall_started = std::time::Instant::now();
        let facts = if !allowed {
            vec![]
        } else {
            or_logged(
                store
                    .get_relevant_facts_for_channel(
                        query,
                        source.settings.max_facts,
                        turn.channel_id.as_deref(),
                        turn.visibility,
                        turn.owner,
                    )
                    .await,
                "facts",
            )
        };

        // The optional recall filter asks which of these facts matter to the
        // message. It runs beside the lookups below and is awaited where the
        // facts are rendered.
        let facts_ms = recall_started.elapsed().as_millis() as u64;
        let fact_filter = self.spawn_fact_filter(query, &facts);

        // The sections below read independent stores, so they run together:
        // run one after another they put ten round trips (and the query
        // embeddings) in front of every turn's first model call.
        let lookups_started = std::time::Instant::now();
        let critical_facts = async {
            // Critical facts (identity/profile): bounded identity categories
            // only, owner DMs (delegated agents included).
            if allowed && inject_personal {
                let mut identity_facts = Vec::new();
                for category in IDENTITY_CATEGORIES {
                    identity_facts.append(&mut or_logged(
                        store.get_facts(Some(category)).await,
                        "critical_facts",
                    ));
                }
                extract_critical_fact_summary(&identity_facts)
            } else {
                Default::default()
            }
        };
        // Episodes: channel-scoped for non-DM channels.
        let episodes = async {
            if !allowed {
                vec![]
            } else if inject_personal {
                or_logged(store.get_relevant_episodes(query, 3).await, "episodes")
            } else if public_external {
                vec![]
            } else {
                or_logged(
                    store
                        .get_relevant_episodes_for_channel(query, 3, turn.channel_id.as_deref())
                        .await,
                    "episodes",
                )
            }
        };
        // Personal goals/profile remain DM-only. Operational failure patterns
        // are safe more broadly: they encode agent-side recovery guidance, not
        // user-private preferences.
        let goals = async {
            if allowed && inject_personal {
                or_logged(host.active_goals().await, "goals")
            } else {
                vec![]
            }
        };
        let patterns = async {
            if !allowed || public_external {
                vec![]
            } else if inject_personal {
                or_logged(store.get_behavior_patterns(0.5).await, "patterns")
            } else {
                or_logged(store.get_behavior_patterns(0.5).await, "patterns")
                    .into_iter()
                    .filter(|pattern| pattern.pattern_type == "failure")
                    .collect()
            }
        };
        // Procedures and error solutions are operational: always load, except
        // on PublicExternal where everything is restricted.
        let procedures = async {
            if !allowed || public_external {
                vec![]
            } else {
                or_logged(store.get_relevant_procedures(query, 5).await, "procedures")
            }
        };
        let error_solutions = async {
            if !allowed || public_external {
                vec![]
            } else {
                or_logged(
                    store.get_relevant_error_solutions(query, 5).await,
                    "error_solutions",
                )
            }
        };
        let profile = async {
            if allowed && inject_personal {
                or_logged(store.get_user_profile().await, "profile")
            } else {
                None
            }
        };
        // People context, then the core profile, which reads whether people
        // are enabled.
        let people_and_core = async {
            // Resolve the current speaker (only when enabled).
            let people_enabled =
                allowed && or_logged(host.people_enabled().await, "people_enabled");
            let (people, current_person, current_person_facts) = if !people_enabled {
                (vec![], None, vec![])
            } else if inject_personal {
                // In owner DMs the people list only selects the privacy rules;
                // the owner's own profile arrives through the core profile.
                (
                    or_logged(store.get_all_people().await, "people"),
                    None,
                    vec![],
                )
            } else if let Some(ref sender_id) = turn.sender_id {
                match store.get_person_by_platform_id(sender_id).await {
                    Ok(Some(person)) => {
                        if let Err(error) = store.touch_person_interaction(person.id).await {
                            warn!(%error, "failed to record person interaction");
                        }
                        let facts = or_logged(
                            store.get_person_facts(person.id, None).await,
                            "person_facts",
                        );
                        (vec![], Some(person), facts)
                    }
                    Ok(None) => (vec![], None, vec![]),
                    Err(error) => {
                        warn!(%error, "memory recall read failed; section left empty");
                        (vec![], None, vec![])
                    }
                }
            } else {
                (vec![], None, vec![])
            };
            let core_profile = self.core_profile(people_enabled).await;
            (people, current_person, current_person_facts, core_profile)
        };
        let (
            mut critical_fact_summary,
            episodes,
            goals,
            patterns,
            procedures,
            error_solutions,
            profile,
            (people, current_person, current_person_facts, core_profile),
        ) = tokio::join!(
            critical_facts,
            episodes,
            goals,
            patterns,
            procedures,
            error_solutions,
            profile,
            people_and_core
        );
        if let Some(configured_name) = source.settings.assistant_name.clone() {
            critical_fact_summary.assistant_name = Some(configured_name);
        }
        let lookups_ms = lookups_started.elapsed().as_millis() as u64;

        let filter_wait_started = std::time::Instant::now();
        let facts = match fact_filter {
            Some(filter) => match filter.await {
                Ok(Some(drop)) if !drop.is_empty() => {
                    let before = facts.len();
                    let kept: Vec<_> = facts
                        .into_iter()
                        .filter(|fact| !drop.contains(&fact.id))
                        .collect();
                    info!(
                        dropped = before - kept.len(),
                        kept = kept.len(),
                        "Recall filter dropped facts irrelevant to the message"
                    );
                    kept
                }
                _ => facts,
            },
            None => facts,
        };
        let filter_wait_ms = filter_wait_started.elapsed().as_millis() as u64;

        let memory_context = MemoryContext {
            facts: &facts,
            episodes: &episodes,
            goals: &goals,
            patterns: &patterns,
            procedures: &procedures,
            error_solutions: &error_solutions,
            profile: profile.as_ref(),
            people: &people,
            current_person: current_person.as_ref(),
            current_person_facts: &current_person_facts,
            scheduled_goals_tool: self.host.offers_scheduled_goals(),
        };
        let report = render_memory_section(&memory_context, source.settings.max_facts);
        if let Err(error) = store
            .record_memory_recall(&report.rendered_fact_ids, &report.rendered_episode_ids)
            .await
        {
            tracing::debug!(%error, "Failed to record rendered memory recall");
        }
        host.decision(
            turn,
            RecallDecision::MemoryRetrieval,
            format!(
                "Memory fetched/rendered: facts={}/{} episodes={}/{} procedures={}/{}",
                facts.len(),
                report.rendered_fact_ids.len(),
                episodes.len(),
                report.rendered_episode_ids.len(),
                procedures.len(),
                report.rendered_procedure_ids.len()
            ),
            json!({
                "policy": if allowed { "allowed" } else { "suppressed" },
                "pipeline_suppressed": !allowed,
                "fetched": {
                    "facts": facts.len(),
                    "episodes": episodes.len(),
                    "goals": goals.len(),
                    "patterns": patterns.len(),
                    "procedures": procedures.len(),
                    "error_solutions": error_solutions.len(),
                    "people": people.len(),
                    "current_person_facts": current_person_facts.len()
                },
                "rendered": {
                    "fact_ids": report.rendered_fact_ids,
                    "episode_ids": report.rendered_episode_ids,
                    "goal_ids": report.rendered_goal_ids,
                    "procedure_ids": report.rendered_procedure_ids,
                    "error_solution_ids": report.rendered_error_solution_ids,
                    "pattern_ids": report.rendered_pattern_ids
                }
            }),
        )
        .await;
        info!(
            session_id = %turn.session_id,
            facts = facts.len(),
            episodes = episodes.len(),
            goals = goals.len(),
            patterns = patterns.len(),
            procedures = procedures.len(),
            elapsed_ms = recall_started.elapsed().as_millis() as u64,
            facts_ms,
            lookups_ms,
            filter_wait_ms,
            "Memory context loaded"
        );
        let mut block =
            build_critical_facts_prompt_block(&critical_fact_summary).unwrap_or_default();
        if !core_profile.is_empty() {
            if !block.is_empty() {
                block.push_str("\n\n");
            }
            block.push_str(&core_profile);
        }
        block.push_str(&report.prompt);
        block.trim_start().to_string()
    }
}

impl Drop for FullRecall {
    /// A turn that ends before its first model call stops the prefetch.
    fn drop(&mut self) {
        if let Ok(mut prefetched) = self.prefetched.lock()
            && let Some(handle) = prefetched.take()
        {
            handle.abort();
        }
    }
}

#[async_trait]
impl aidaemon_policies::Recall for FullRecall {
    /// Channel-scoped fact lines, for generic callers.
    async fn recall(&self, _session_id: &str, query: &str, limit: usize) -> Vec<String> {
        if !self.turn.allowed {
            return Vec::new();
        }
        let facts = self
            .source
            .store
            .get_relevant_facts_for_channel(
                query,
                limit,
                self.turn.channel_id.as_deref(),
                self.turn.visibility,
                self.turn.owner,
            )
            .await;
        or_logged(facts, "facts")
            .into_iter()
            .map(|fact| format!("[{}] {}: {}", fact.category, fact.key, fact.value))
            .collect()
    }

    /// The full memory block with its section headings.
    async fn render(&self, _session_id: &str, query: &str, _limit: usize) -> Option<String> {
        let block = match self.take_prefetched().await {
            Some(block) => block,
            None => self.render_block(query).await,
        };
        (!block.trim().is_empty()).then_some(block)
    }
}
