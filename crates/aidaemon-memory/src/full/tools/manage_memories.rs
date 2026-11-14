//! `manage_memories`: read, search, forget and re-scope what the memory
//! holds. A host that manages more than memory through the same tool (the
//! daemon's goals) presents its own combined tool and runs these actions
//! through [`ManageMemoriesTool::run`].

use std::sync::Arc;

use aidaemon_core::ChannelVisibility;
use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ApprovalCall, OwnerApprovalNeed, ToolCallContext, ToolCallMetadata, ToolCallOutcome,
    ToolCallSemantics, ToolCapabilities, ToolMutationEffects, ToolOutcomeStatus,
    semantics_for_exact_read_actions,
};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::records::{Fact, FactPrivacy};
use crate::stores::MemoryStores;

/// The actions this tool runs.
pub const MEMORY_ACTIONS: &[&str] = &[
    "list",
    "health",
    "repair",
    "forget",
    "set_privacy",
    "search",
    "search_episodes",
];

/// The actions that only read.
const READ_ACTIONS: &[&str] = &["list", "health", "search", "search_episodes"];

fn compact_provenance_excerpt(text: &str) -> String {
    let single_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    crate::text::truncate_chars(&single_line, 180)
}

fn canonical_personal_display_key(key: &str) -> String {
    match key.trim().to_ascii_lowercase().as_str() {
        "birthday" | "date_of_birth" | "dob" => "birth_date".to_string(),
        "current_residence" | "home" | "location" => "residence".to_string(),
        "place_of_birth" => "birthplace".to_string(),
        "favorite_name" | "preferred_first_name" => "preferred_name".to_string(),
        other => other.to_string(),
    }
}

/// The stored facts and episodes, managed by the model: list, search,
/// forget, change who may see a fact, and check or repair the store.
pub struct ManageMemoriesTool {
    store: Arc<dyn MemoryStores>,
}

impl ManageMemoriesTool {
    pub fn new(store: Arc<dyn MemoryStores>) -> Self {
        Self { store }
    }

    /// The active fact a `set_privacy` call names, if any.
    async fn find_fact(&self, category: &str, key: &str) -> anyhow::Result<Option<Fact>> {
        Ok(self
            .store
            .get_facts(Some(category))
            .await?
            .into_iter()
            .find(|f| f.key == key && f.superseded_at.is_none()))
    }

    /// The owner card for a `set_privacy` call that widens who may see a
    /// fact; narrowing, a call naming no active fact, and every other
    /// action need none. The subject is the exact arguments approved.
    pub async fn privacy_approval(&self, arguments: &Value) -> OwnerApprovalNeed {
        let Ok(args) = serde_json::from_value::<ManageArgs>(arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        let (Some(category), Some(key), Some(privacy)) = (
            args.category.as_deref(),
            args.key.as_deref(),
            args.privacy.as_deref(),
        ) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        if args.action != "set_privacy" {
            return OwnerApprovalNeed::NotNeeded;
        }
        let to = FactPrivacy::from_str_lossy(privacy);
        let value = match self.find_fact(category, key).await {
            Ok(Some(fact)) if fact.privacy.widens_to(to) => {
                format!(" = \"{}\"", crate::text::truncate_chars(&fact.value, 83))
            }
            Ok(_) => return OwnerApprovalNeed::NotNeeded,
            // The lookup failed: ask with what the call names rather than
            // waving it through (the run reads the store again).
            Err(_) => String::new(),
        };
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!("Make memory {to}: [{category}] {key}{value}"),
            subject: json!({"action": "set_privacy", "category": category, "key": key, "privacy": to.to_string()})
                .to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec![match to {
                FactPrivacy::Global => "This will make this memory visible across all channels",
                _ => "This will make this memory visible in its channel, not only in DMs",
            }
            .to_string()],
        })
    }
}

#[derive(Deserialize)]
struct ManageArgs {
    action: String,
    limit: Option<usize>,
    category: Option<String>,
    key: Option<String>,
    privacy: Option<String>,
    query: Option<String>,
}

const TOOL_DESCRIPTION: &str = "Memories. search=stored facts; search_episodes=generated summaries of past chats (exact messages: search_history). Store via remember_fact.";

/// The memory actions' parameters, each described. A host that offers more
/// actions on the same tool (personal goals) extends this map.
pub fn memory_parameters() -> serde_json::Map<String, Value> {
    let Value::Object(map) = json!({
        "action": { "type": "string", "enum": MEMORY_ACTIONS, "description": "forget and set_privacy name one fact by category and key." },
        "limit": { "type": "integer", "description": "Max results." },
        "category": { "type": "string", "description": "forget/set_privacy: the fact's category; list: filter." },
        "key": { "type": "string", "description": "forget/set_privacy: the fact's key." },
        "privacy": { "type": "string", "enum": ["global", "channel", "private"], "description": "set_privacy: who may see it." },
        "query": { "type": "string", "description": "search: what to find. search_episodes: a topic, or omit for the latest chats." }
    }) else {
        unreachable!("a JSON object literal")
    };
    map
}

/// The answer to an action the tool does not have, naming the ones it has.
pub fn unknown_action_message(action: &str, actions: &[&str]) -> String {
    format!("Unknown action: '{action}'. Use {}.", actions.join(", "))
}

fn manage_memories_schema() -> Value {
    json!({
        "name": "manage_memories",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": memory_parameters(),
            "required": ["action"],
            "additionalProperties": false
        }
    })
}

/// Memory is persistent agent state, so writes are `configuration` and a
/// forget is also `destructive`.
fn memory_action_effects(arguments: &str) -> ToolMutationEffects {
    match action_of(arguments).as_deref() {
        Some("forget") => {
            ToolMutationEffects::CONFIGURATION.union(ToolMutationEffects::DESTRUCTIVE)
        }
        Some("set_privacy") => ToolMutationEffects::CONFIGURATION,
        _ => ToolMutationEffects::NONE,
    }
}

/// The `action` argument, if any.
fn action_of(arguments: &str) -> Option<String> {
    serde_json::from_str::<Value>(arguments)
        .ok()?
        .get("action")?
        .as_str()
        .map(str::to_string)
}

#[async_trait]
impl Tool for ManageMemoriesTool {
    fn name(&self) -> &str {
        "manage_memories"
    }

    fn description(&self) -> &str {
        TOOL_DESCRIPTION
    }

    fn schema(&self) -> Value {
        manage_memories_schema()
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let output = self.run(arguments, facts).await?;
        let trimmed = output.trim_start();
        let outcome_status = if trimmed.starts_with("Unknown action:") {
            ToolOutcomeStatus::FailedPermanent
        } else if trimmed.starts_with("No matching") {
            ToolOutcomeStatus::CompletedWithNegativeResult
        } else {
            ToolOutcomeStatus::Succeeded
        };
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(outcome_status),
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

#[async_trait]
impl aidaemon_tools::semantics::ToolSemantics for ManageMemoriesTool {
    fn dispatch_facts(&self) -> aidaemon_tools::semantics::ToolDispatchFacts {
        aidaemon_tools::semantics::ToolDispatchFacts {
            offering: aidaemon_tools::semantics::Offering::Core,
            status_from_arguments: true,
            activity: aidaemon_tools::semantics::ToolActivity::Memory,
            ..Default::default()
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        semantics_for_exact_read_actions(arguments, READ_ACTIONS, memory_action_effects(arguments))
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }

    /// `set_privacy` asks the owner when it widens who may see a fact.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        self.privacy_approval(call.arguments).await
    }

    /// Reads, `repair` and `forget` touch only the agent's own memory
    /// store. `set_privacy` widens who may see a fact, so it stays
    /// undeclared.
    fn call_effects(&self, arguments: &str) -> aidaemon_core::CallEffects {
        match action_of(arguments).as_deref() {
            Some("list" | "health" | "search" | "search_episodes" | "repair" | "forget") => {
                aidaemon_core::CallEffects::none()
            }
            _ => aidaemon_core::CallEffects::unknown(),
        }
    }
}

impl ManageMemoriesTool {
    /// Run one call with the dispatcher's `facts` (session, role, channel,
    /// trust), never facts from the arguments. An action outside
    /// [`MEMORY_ACTIONS`] is answered as unknown.
    pub async fn run(&self, arguments: &str, facts: &ToolCallContext) -> anyhow::Result<String> {
        let args: ManageArgs = serde_json::from_str(arguments)?;
        let trusted_session = facts.trusted_session;

        match args.action.as_str() {
            "health" => {
                let report = self.store.memory_health_report().await?;
                Ok(serde_json::to_string_pretty(&report)?)
            }
            "repair" => {
                let report = self.store.repair_memory_projections().await?;
                Ok(serde_json::to_string_pretty(&report)?)
            }
            "list" => {
                let mut facts = self.store.get_canonical_memory_facts().await?;
                let canonical_owner_keys: std::collections::HashSet<String> = facts
                    .iter()
                    .filter(|fact| fact.category == "user")
                    .map(|fact| canonical_personal_display_key(&fact.key))
                    .collect();
                facts.extend(
                    self.store
                        .get_all_facts_with_provenance()
                        .await?
                        .into_iter()
                        .filter(|fact| {
                            fact.category != "user"
                                || !canonical_owner_keys
                                    .contains(&canonical_personal_display_key(&fact.key))
                        }),
                );
                if facts.is_empty() {
                    return Ok("No memories stored.".to_string());
                }

                let filtered = if let Some(ref cat) = args.category {
                    facts
                        .into_iter()
                        .filter(|f| f.category == *cat)
                        .collect::<Vec<_>>()
                } else {
                    facts
                };

                let limit = args.limit.unwrap_or(100).clamp(1, 200);
                let total = filtered.len();
                let shown = filtered.into_iter().take(limit).collect::<Vec<_>>();

                let mut output = format!(
                    "**Stored Memories** (showing {} of {} facts)\n\n",
                    shown.len(),
                    total
                );
                let mut current_cat = String::new();
                for f in &shown {
                    if f.category != current_cat {
                        current_cat = f.category.clone();
                        output.push_str(&format!("### {}\n", current_cat));
                    }
                    let privacy_label = f.privacy.to_string();
                    let channel_label = f.channel_id.as_deref().unwrap_or("global");
                    let age = chrono::Utc::now().signed_duration_since(f.updated_at);
                    let age_str = if age.num_days() > 0 {
                        format!("{}d ago", age.num_days())
                    } else if age.num_hours() > 0 {
                        format!("{}h ago", age.num_hours())
                    } else {
                        "just now".to_string()
                    };
                    output.push_str(&format!(
                        "- **{}**: {} (privacy: {}, from: {}, updated: {})\n",
                        f.key, f.value, privacy_label, channel_label, age_str
                    ));
                }
                if total > shown.len() {
                    output.push_str(&format!(
                        "\n(Results truncated to {}. Provide a higher `limit` to see more.)",
                        shown.len()
                    ));
                }
                Ok(output)
            }
            "forget" => {
                let key_raw = args
                    .key
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'key' is required for forget action"))?;
                let category = args
                    .category
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'category' is required for forget action"))?;

                // Reject empty / whitespace-only / too-short keys. A blank key would
                // make partial key matching true for every
                // fact, wiping the store. A single character is also too broad to be
                // a safe substring match across all facts.
                let key = key_raw.trim();
                if key.is_empty() {
                    anyhow::bail!("'key' must not be empty or whitespace for forget action");
                }
                if key.chars().count() < 2 {
                    anyhow::bail!(
                        "'key' must be at least 2 characters for forget action (got {:?}); use manage_memories(action='list') to see exact keys",
                        key
                    );
                }

                // Canonicalize the requested key for fuzzy matching
                let canon = |k: &str| -> String {
                    let mut out = String::with_capacity(k.len());
                    let mut last_sep = false;
                    for ch in k.trim().chars() {
                        if ch.is_ascii_alphanumeric() {
                            out.push(ch.to_ascii_lowercase());
                            last_sep = false;
                        } else if !last_sep {
                            out.push('_');
                            last_sep = true;
                        }
                    }
                    out.trim_matches('_').to_string()
                };

                let key_canonical = canon(key);

                // Within the explicitly-requested category an exact or canonical
                // key match is authoritative. Only when none exists may a partial
                // key select facts, and then only as whole `_`-separated key
                // segments ("color" -> "favorite_color"). A fact key contained
                // in the requested key is never a match: forgetting
                // `qa_pet_name` used to also delete the owner's `name` fact.
                let exact_in_category = |f: &crate::records::Fact| -> bool {
                    f.superseded_at.is_none() && (f.key == key || canon(&f.key) == key_canonical)
                };
                let key_segments: Vec<&str> = key_canonical.split('_').collect();
                let partial_in_category = |f: &crate::records::Fact| -> bool {
                    if f.superseded_at.is_some() || key_canonical.is_empty() {
                        return false;
                    }
                    let fk_canon = canon(&f.key);
                    let fact_segments: Vec<&str> = fk_canon.split('_').collect();
                    fact_segments
                        .windows(key_segments.len())
                        .any(|window| window == key_segments.as_slice())
                };

                // Across OTHER categories (the "duplicate sweep") only accept exact
                // or canonical matches. Substring matching outside the requested
                // category is too broad and can delete unrelated facts.
                let matches_key_cross_category = |f: &crate::records::Fact| -> bool {
                    if f.superseded_at.is_some() {
                        return false;
                    }
                    f.key == key || canon(&f.key) == key_canonical
                };

                let mut to_delete: Vec<(i64, String, String)> = Vec::new();
                let mut seen_ids = std::collections::HashSet::new();

                let facts = self.store.get_facts(Some(category)).await?;
                let has_exact = facts.iter().any(&exact_in_category);
                for f in &facts {
                    let selected = if has_exact {
                        exact_in_category(f)
                    } else {
                        partial_in_category(f)
                    };
                    if selected && seen_ids.insert(f.id) {
                        to_delete.push((f.id, f.category.clone(), f.key.clone()));
                    }
                }

                // Also check OTHER categories for exact/canonical duplicates only.
                let all_facts = self.store.get_facts(None).await?;
                for f in &all_facts {
                    if f.category == category {
                        continue;
                    }
                    if matches_key_cross_category(f) && seen_ids.insert(f.id) {
                        to_delete.push((f.id, f.category.clone(), f.key.clone()));
                    }
                }

                if to_delete.is_empty() {
                    Ok(format!(
                        "No active fact found matching key '{}' in any category. Use manage_memories(action='list') to see stored facts and their exact keys.",
                        key
                    ))
                } else {
                    let mut forgotten = Vec::new();
                    for (id, cat, k) in &to_delete {
                        self.store.delete_fact(*id).await?;
                        forgotten.push(format!("[{}] {}", cat, k));
                    }
                    Ok(format!(
                        "Forgotten {} fact(s): {}",
                        forgotten.len(),
                        forgotten.join(", ")
                    ))
                }
            }
            "set_privacy" => {
                let key = args
                    .key
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("'key' is required for set_privacy action"))?;
                let category = args.category.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'category' is required for set_privacy action")
                })?;
                let privacy_str = args.privacy.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("'privacy' is required for set_privacy action")
                })?;
                let privacy = FactPrivacy::from_str_lossy(privacy_str);

                let fact = self.find_fact(category, key).await?;

                match fact {
                    Some(f) => {
                        // The kernel asked the owner before a widening
                        // runs; one that reaches here without its
                        // judgment is refused.
                        if f.privacy.widens_to(privacy) && facts.kernel_approved.is_none() {
                            anyhow::bail!(
                                "widening a memory's privacy requires approval through the agent \
                                 kernel; it was called outside an approved agent call"
                            );
                        }
                        self.store.update_fact_privacy(f.id, privacy).await?;
                        Ok(format!(
                            "Updated privacy: [{}] {} → {}",
                            category, key, privacy
                        ))
                    }
                    None => Ok(format!("No active fact found: [{}] {}", category, key)),
                }
            }
            "search" => {
                let query = args.query.as_deref().unwrap_or("");
                if query.is_empty() {
                    return Ok("Please provide a search query.".to_string());
                }

                let limit = args.limit.unwrap_or(10).clamp(1, 200);
                let report = self
                    .store
                    .search_facts(query, limit, &crate::search::FactSearchConfig::default())
                    .await?;
                if report.hits.is_empty() {
                    return Ok(format!("No memories matching '{}'.", query));
                }

                // "Relevant" only when a relevance floor judged the pool;
                // otherwise every candidate would be called relevant.
                let counts = if report.floor_applied {
                    format!(
                        "{} shown, {} relevant of {} considered",
                        report.hits.len(),
                        report.passed_floor,
                        report.considered
                    )
                } else {
                    format!(
                        "{} shown of {} considered",
                        report.hits.len(),
                        report.considered
                    )
                };
                let mut output = format!(
                    "══ Stored facts matching '{}' ({}) ══\n\
                     IMPORTANT: Use these EXACT values when answering — do not substitute or infer.\n\n",
                    query, counts
                );
                for hit in &report.hits {
                    let f = &hit.fact;
                    let privacy_label = f.privacy.to_string();
                    let channel_label = f.channel_id.as_deref().unwrap_or("global");
                    let marker = if hit.kind == crate::search::MatchKind::Semantic {
                        " (semantic match)"
                    } else {
                        ""
                    };
                    // Surface WHEN the fact was learned so the model can answer
                    // temporal questions ("when did I tell you X?") directly from
                    // search results instead of reaching for the wrong tool.
                    let learned_ts = f.first_seen_at.unwrap_or(f.created_at).to_rfc3339();
                    let learned = aidaemon_tools::duration_format::format_age(&learned_ts);
                    let evidence = f
                        .source_excerpt
                        .as_deref()
                        .map(compact_provenance_excerpt)
                        .filter(|excerpt| !excerpt.is_empty())
                        .map(|excerpt| {
                            format!(
                                "; evidence: {}",
                                serde_json::to_string(&excerpt)
                                    .unwrap_or_else(|_| "\"[unavailable]\"".to_string())
                            )
                        })
                        .unwrap_or_default();
                    output.push_str(&format!(
                        "• [{}] {} → \"{}\" (fact_id: {}, source: {}, privacy: {}, from: {}, learned: {}{}){}\n",
                        f.category,
                        f.key,
                        f.value,
                        f.id,
                        f.source,
                        privacy_label,
                        channel_label,
                        learned,
                        evidence,
                        marker
                    ));
                }
                if report.floor_applied && report.passed_floor > report.hits.len() {
                    output.push_str(&format!(
                        "({} more relevant; raise limit to see them.)\n",
                        report.passed_floor - report.hits.len()
                    ));
                }
                if !report.related.is_empty() {
                    output.push_str("\n── Related context (connected facts) ──\n");
                    for hit in &report.related {
                        output.push_str(&format!(
                            "• [{}] {} → \"{}\"\n",
                            hit.fact.category, hit.fact.key, hit.fact.value
                        ));
                    }
                }

                Ok(output)
            }
            "search_episodes" => {
                // Active recall over episodic memory (past-conversation summaries).
                // Facts answer "what is X"; episodes answer "what did we discuss /
                // do, and when". With a query → semantic search; without one →
                // most-recent conversations (browse). This tool is owner-only (the
                // session tool roster is empty for non-owners), with explicit
                // channel/session scoping for non-private origins.
                let query = args.query.as_deref().unwrap_or("").trim();
                let limit = args.limit.unwrap_or(8).clamp(1, 50);
                let visibility = facts
                    .channel_visibility
                    .unwrap_or(ChannelVisibility::Internal);
                let episodes = match visibility {
                    ChannelVisibility::Private => {
                        self.store.get_relevant_episodes(query, limit).await?
                    }
                    ChannelVisibility::PrivateGroup | ChannelVisibility::Public => {
                        self.store
                            .get_relevant_episodes_for_channel(
                                query,
                                limit,
                                facts.channel_id.as_deref(),
                            )
                            .await?
                    }
                    ChannelVisibility::PublicExternal => {
                        return Ok(
                            "Past-conversation recall is disabled in public-external channels."
                                .to_string(),
                        );
                    }
                    ChannelVisibility::Internal if trusted_session => {
                        self.store.get_relevant_episodes(query, limit).await?
                    }
                    ChannelVisibility::Internal => {
                        let session_id = facts.session().unwrap_or("");
                        self.store
                            .get_relevant_episodes_for_session(query, limit, session_id)
                            .await?
                    }
                };
                let recalled: Vec<i64> = episodes.iter().map(|episode| episode.id).collect();
                let _ = self.store.record_memory_recall(&[], &recalled).await;
                if episodes.is_empty() {
                    return Ok(if query.is_empty() {
                        "No past conversations recorded yet.".to_string()
                    } else {
                        format!("No past conversations found matching '{}'.", query)
                    });
                }

                let header = if query.is_empty() {
                    format!(
                        "══ Generated episode summaries: recent conversations ({}) ══\n\n",
                        episodes.len()
                    )
                } else {
                    format!(
                        "══ Generated episode summaries matching '{}' ({}) ══\n\n",
                        query,
                        episodes.len()
                    )
                };
                let mut output = header;
                for e in &episodes {
                    let when =
                        aidaemon_tools::duration_format::format_age(&e.created_at.to_rfc3339());
                    output.push_str(&format!("• [{}] {}", when, e.summary.trim()));
                    if let Some(topics) = e.topics.as_ref().filter(|t| !t.is_empty()) {
                        output.push_str(&format!(" (topics: {})", topics.join(", ")));
                    }
                    if let Some(outcome) = e.outcome.as_deref().filter(|o| !o.is_empty()) {
                        output.push_str(&format!(" [outcome: {}]", outcome));
                    }
                    output.push('\n');
                }
                output.push_str(
                    "\nThese are semantic, generated summaries. Use search_history for exact \
                     retained user/assistant wording, anchored context, and message paging.\n",
                );
                Ok(output)
            }
            other => Ok(unknown_action_message(other, MEMORY_ACTIONS)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_excerpt_is_single_line_utf8_safe_and_bounded() {
        let input = format!("User said:\n\t{}", "🦀".repeat(250));
        let compact = compact_provenance_excerpt(&input);
        assert!(!compact.contains('\n'));
        assert!(!compact.contains('\t'));
        assert!(compact.chars().count() <= 180);
        assert!(compact.ends_with("..."));
    }

    async fn tool_with_channel_fact() -> (Arc<crate::sqlite::SqliteMemory>, ManageMemoriesTool) {
        use crate::stores::FactStore as _;
        let store = crate::full::tools::test_support::memory().await;
        store
            .upsert_fact(
                "preference",
                "report_format",
                "short summaries for Alice Rivera",
                "owner",
                Some("telegram:synthetic-channel-1"),
                FactPrivacy::Channel,
            )
            .await
            .unwrap();
        (store.clone(), ManageMemoriesTool::new(store))
    }

    async fn privacy_of(store: &crate::sqlite::SqliteMemory) -> FactPrivacy {
        use crate::stores::FactStore as _;
        store.get_facts(Some("preference")).await.unwrap()[0].privacy
    }

    #[tokio::test]
    async fn widening_privacy_asks_with_its_exact_arguments() {
        let (_, tool) = tool_with_channel_fact().await;
        let args = json!({"action": "set_privacy", "category": "preference", "key": "report_format", "privacy": "global"});
        let OwnerApprovalNeed::Ask(request) = tool.privacy_approval(&args).await else {
            panic!("widening a fact must ask");
        };
        assert_eq!(
            request.summary,
            "Make memory global: [preference] report_format = \"short summaries for Alice Rivera\""
        );
        assert_eq!(request.risk, aidaemon_core::ApprovalRisk::High);
        let subject: Value = serde_json::from_str(&request.subject).unwrap();
        assert_eq!(subject, args);
        for narrower in [
            json!({"action": "set_privacy", "category": "preference", "key": "report_format", "privacy": "private"}),
            json!({"action": "set_privacy", "category": "preference", "key": "absent", "privacy": "global"}),
            json!({"action": "list"}),
        ] {
            assert!(matches!(
                tool.privacy_approval(&narrower).await,
                OwnerApprovalNeed::NotNeeded
            ));
        }
    }

    #[tokio::test]
    async fn widening_without_the_kernel_is_refused_and_narrowing_is_not() {
        let (store, tool) = tool_with_channel_fact().await;
        let widen = r#"{"action":"set_privacy","category":"preference","key":"report_format","privacy":"global"}"#;
        let err = tool
            .run(widen, &ToolCallContext::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("requires approval through the agent kernel"),
            "{err}"
        );
        assert_eq!(privacy_of(&store).await, FactPrivacy::Channel);
        tool.run(
            widen,
            &ToolCallContext {
                kernel_approved: Some(aidaemon_tools::semantics::ApprovalScope::Call),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(privacy_of(&store).await, FactPrivacy::Global);
        let narrow = r#"{"action":"set_privacy","category":"preference","key":"report_format","privacy":"private"}"#;
        tool.run(narrow, &ToolCallContext::default()).await.unwrap();
        assert_eq!(privacy_of(&store).await, FactPrivacy::Private);
    }
}
