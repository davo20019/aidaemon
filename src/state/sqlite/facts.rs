use super::*;

const FACT_SEMANTIC_MIN_SCORE: f32 = 0.3;
/// Recall-oriented threshold for the EXPLICIT, user-initiated memory-search tool
/// (`search_facts_semantic`), lower than the passive-injection cutoff above.
/// Explicit search is owner-only, deliberately requested, and merged AFTER the
/// high-precision lexical pass, so favouring recall is safe here. The gap matters
/// for near-synonyms the small embedding model rates just under 0.3 — e.g. "spouse"
/// (~0.28) / "wife" (~0.23) against a stored `partner` fact — which would otherwise
/// be missed on the search path even though the fact is plainly relevant.
const EXPLICIT_SEARCH_SEMANTIC_MIN_SCORE: f32 = 0.22;
/// Bi-encoder candidate-pool size handed to the cross-encoder reranker in
/// explicit search. Deep enough to contain a weakly-scoring correct fact
/// (measured rank ~30 for "spouse"→partner_name) while keeping reranking — a
/// per-candidate cross-encoder pass — bounded in latency.
const EXPLICIT_SEARCH_CANDIDATE_POOL: usize = 50;
const FACT_LEXICAL_MIN_SCORE: f32 = 0.3;
const FACT_LEXICAL_MAX_SCORE: f32 = 0.55;
const FACT_FRESHNESS_MAX_BOOST: f32 = 0.15;
const FACT_FRESHNESS_DECAY_HOURS: f32 = 168.0; // 7 days
const FACT_PAD_LOW_CONFIDENCE_RESULTS: bool = true;

/// Keywords that signal the user wants ALL facts, not a filtered subset.
const EXHAUSTIVE_QUERY_MARKERS: &[&str] = &[
    "everything you know",
    "everything about me",
    "all about me",
    "all you know",
    "tell me everything",
    "complete list",
    "full list",
    "what do you know about me",
    "what do you remember",
    "list all",
    "list everything",
    "verify your memory",
    "memory check",
    "memory test",
    "dump all",
];

fn is_exhaustive_query(query: &str) -> bool {
    let q = query.to_lowercase();
    EXHAUSTIVE_QUERY_MARKERS.iter().any(|m| q.contains(m))
}

/// Build a key-only embedding text for semantic dedup comparison.
///
/// Focuses on category + readable key (no value) so that dedup detects
/// keys referring to the same concept regardless of their values.
fn build_dedup_key_text(category: &str, key: &str) -> String {
    let readable_key = key.replace(['_', '-'], " ");
    format!(
        "The user's {} {}. {} attribute: {}",
        category, readable_key, category, readable_key
    )
}

fn hybrid_fact_score(
    fact: &Fact,
    semantic: f32,
    lexical: f32,
    graph_match: bool,
    freshness: f32,
) -> f32 {
    crate::memory::hybrid::fused_score(crate::memory::hybrid::HybridSignals {
        semantic,
        lexical,
        graph: if graph_match { 1.0 } else { 0.0 },
        freshness: (freshness / FACT_FRESHNESS_MAX_BOOST).clamp(0.0, 1.0),
        confidence: if fact.source == "consolidation" {
            0.7
        } else {
            1.0
        },
        provenance: if fact.source_excerpt.is_some() {
            1.0
        } else {
            0.3
        },
    })
}

/// Build a natural-language embedding text for a fact.
///
/// Instead of `[project] inter_service: gRPC`, this produces something like:
/// `"Project: inter-service communication technology is gRPC"`
///
/// Natural language embeds much better against natural language queries
/// like "What tech stack does my project use?".
pub(crate) fn build_fact_embedding_text(category: &str, key: &str, value: &str) -> String {
    // Convert underscore/hyphen keys to readable form
    let readable_key = key.replace(['_', '-'], " ");

    // Build a natural-language sentence
    format!(
        "{} - {}: {}. The user's {} is {}.",
        category, readable_key, value, readable_key, value
    )
}

async fn bump_fact_recall(pool: &SqlitePool, facts: &[Fact]) {
    if facts.is_empty() {
        return;
    }
    let now = Utc::now().to_rfc3339();
    let ids: Vec<i64> = facts.iter().map(|f| f.id).collect();
    let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
    let query = format!(
        "UPDATE facts SET recall_count = recall_count + 1, last_recalled_at = ? WHERE id IN ({})",
        placeholders.join(",")
    );
    let mut q = sqlx::query(&query).bind(&now);
    for id in ids {
        q = q.bind(id);
    }
    let _ = q.execute(pool).await;
}

fn canonicalize_fact_key(key: &str) -> String {
    let raw = key.trim();
    if raw.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(raw.len());
    let mut last_was_sep = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }

    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        raw.to_string()
    } else {
        trimmed.to_string()
    }
}

fn canonical_personal_key(key: &str) -> String {
    match canonicalize_fact_key(key).as_str() {
        "birthday" | "date_of_birth" | "dob" => "birth_date".to_string(),
        "current_residence" | "home" | "location" => "residence".to_string(),
        "place_of_birth" => "birthplace".to_string(),
        "favorite_name" | "preferred_first_name" => "preferred_name".to_string(),
        other => other.to_string(),
    }
}

fn prefer_canonical_facts(
    mut canonical: Vec<Fact>,
    legacy: Vec<Fact>,
    query: &str,
    max: usize,
) -> Vec<Fact> {
    let query_lower = query.to_lowercase();
    let query_words = query_tokens(&query_lower);
    if !query.trim().is_empty() && !is_exhaustive_query(query) {
        canonical.retain(|fact| {
            lexical_fallback_score(&query_lower, &query_words, fact) > 0.0
                || fact.value.to_lowercase().contains(query_lower.trim())
        });
    }

    let canonical_owner_keys: std::collections::HashSet<String> = canonical
        .iter()
        .filter(|fact| fact.category == "user")
        .map(|fact| canonical_personal_key(&fact.key))
        .collect();
    let has_relationships = canonical
        .iter()
        .any(|fact| fact.category == "relationships");
    canonical.extend(legacy.into_iter().filter(|fact| {
        !(fact.category == "user"
            && canonical_owner_keys.contains(&canonical_personal_key(&fact.key))
            || has_relationships
                && matches!(
                    canonicalize_fact_key(&fact.key).as_str(),
                    "daughter"
                        | "son"
                        | "child"
                        | "mother"
                        | "father"
                        | "parent"
                        | "daughter_name"
                        | "son_name"
                        | "child_name"
                        | "mother_name"
                        | "father_name"
                ))
    }));
    // Preserve semantic/graph ranking within each group, but always surface an
    // exact structured-key hit before loosely related embedding matches. This
    // is critical for short profile questions such as "my dad" (`dad_name`) or
    // "my mom" (`mom_name`), where generic identity facts otherwise outrank the
    // answer-bearing record.
    canonical.sort_by_key(|fact| {
        let key = fact.key.to_ascii_lowercase();
        std::cmp::Reverse(
            query_words
                .iter()
                .filter(|word| contains_word(&key, word))
                .count(),
        )
    });
    canonical.truncate(max);
    canonical
}

fn fact_freshness_boost(now: DateTime<Utc>, updated_at: DateTime<Utc>) -> f32 {
    let age_hours = (now - updated_at).num_hours().max(0) as f32;
    FACT_FRESHNESS_MAX_BOOST * (1.0 - (age_hours / FACT_FRESHNESS_DECAY_HOURS).min(1.0))
}

pub(super) fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "me"
            | "my"
            | "of"
            | "on"
            | "or"
            | "please"
            | "tell"
            | "that"
            | "the"
            | "to"
            | "was"
            | "we"
            | "were"
            | "what"
            | "when"
            | "where"
            | "who"
            | "why"
            | "you"
            | "your"
    )
}

fn query_tokens(query_lower: &str) -> Vec<&str> {
    query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .filter(|t| !is_stopword(t))
        .collect()
}

fn contains_word(haystack_lower: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    haystack_lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| w == needle)
}

fn lexical_fallback_score(query_lower: &str, tokens: &[&str], fact: &Fact) -> f32 {
    let q = query_lower.trim();
    if q.is_empty() || tokens.is_empty() {
        return 0.0;
    }

    let key = fact.key.to_lowercase();
    let value = fact.value.to_lowercase();
    let category = fact.category.to_lowercase();

    let q_is_token = q.chars().all(|c| c.is_alphanumeric());
    let value_has_q = q_is_token && contains_word(&value, q);

    // Keys/categories are structured; substring matching is fine.
    // Values are natural language; prefer word-boundary matching to avoid
    // false positives like "dog" matching "dodger".
    if key.contains(q) || category.contains(q) || value_has_q {
        return FACT_LEXICAL_MAX_SCORE;
    }

    // If any query token matches the structured key exactly (word boundary),
    // treat as high-confidence lexical relevance.
    if tokens.iter().any(|t| contains_word(&key, t)) {
        return FACT_LEXICAL_MAX_SCORE;
    }

    let mut matched = 0usize;
    for t in tokens {
        if key.contains(t) || category.contains(t) || contains_word(&value, t) {
            matched += 1;
        }
    }
    (matched as f32 / tokens.len() as f32) * FACT_LEXICAL_MAX_SCORE
}

#[async_trait]
impl crate::traits::FactStore for SqliteStateStore {
    async fn reconcile_personal_memory(
        &self,
        write: &crate::traits::PersonalMemoryWrite,
        source: &str,
        source_excerpt: Option<&str>,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<crate::traits::PersonalMemoryWriteResult> {
        self.reconcile_structured_personal_memory(
            write,
            source,
            source_excerpt,
            channel_id,
            privacy,
        )
        .await
    }

    async fn get_canonical_memory_facts(&self) -> anyhow::Result<Vec<Fact>> {
        self.canonical_personal_facts().await
    }

    async fn memory_health_report(&self) -> anyhow::Result<crate::traits::MemoryHealthReport> {
        self.canonical_memory_health().await
    }

    async fn repair_memory_projections(&self) -> anyhow::Result<crate::traits::MemoryHealthReport> {
        self.backfill_missing_memory_projections().await?;
        self.canonical_memory_health().await
    }

    async fn refresh_fact_memory(&self, category: &str, key: &str) -> anyhow::Result<()> {
        if let Some(id) = sqlx::query_scalar::<_, i64>(
            "SELECT id FROM facts WHERE lower(category) = lower(?) AND lower(key) = lower(?)
             AND superseded_at IS NULL ORDER BY updated_at DESC LIMIT 1",
        )
        .bind(category)
        .bind(key)
        .fetch_optional(&self.pool)
        .await?
        {
            self.project_fact_memory(id).await?;
        }
        Ok(())
    }

    async fn project_extracted_fact_graph(
        &self,
        category: &str,
        key: &str,
        source_excerpt: &str,
        graph: &crate::traits::ExtractedMemoryGraph,
    ) -> anyhow::Result<()> {
        self.persist_extracted_fact_graph(category, key, source_excerpt, graph)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn upsert_fact_with_provenance(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
        first_seen_at: Option<DateTime<Utc>>,
        source_excerpt: Option<&str>,
    ) -> anyhow::Result<()> {
        // The legacy flat-fact API performs a read/reconcile/write sequence.
        // Serialize that sequence so two concurrent writes for the same key
        // cannot both supersede the active row and race through the unique
        // active-fact constraint. Structured personal-memory writes use their
        // own explicit SQL transaction and do not take this compatibility lock.
        let _upsert_guard = self.fact_upsert_lock.lock().await;
        let now = Utc::now().to_rfc3339();
        let privacy_str = privacy.to_string();

        let category_clean = category.trim();
        let key_clean = key.trim();
        let canonical_key = canonicalize_fact_key(key_clean);

        // Find existing current fact (not superseded).
        // Prefer exact match; fall back to canonical match to avoid key drift
        // ("dog name" vs "dog_name") creating duplicates.
        let mut existing: Option<(i64, String, String)> = None;

        if let Some(row) = sqlx::query(
            "SELECT id, key, value FROM facts WHERE category = ? AND key = ? AND superseded_at IS NULL",
        )
        .bind(category_clean)
        .bind(key_clean)
        .fetch_optional(&self.pool)
        .await?
        {
            existing = Some((row.get("id"), row.get("key"), row.get("value")));
        } else {
            if canonical_key != key_clean {
                if let Some(row) = sqlx::query(
                    "SELECT id, key, value FROM facts WHERE category = ? AND key = ? AND superseded_at IS NULL",
                )
                .bind(category_clean)
                .bind(&canonical_key)
                .fetch_optional(&self.pool)
                .await?
                {
                    existing = Some((row.get("id"), row.get("key"), row.get("value")));
                }
            }

            if existing.is_none() && !canonical_key.is_empty() {
                // Canonical scan: match an existing key by canonical form.
                // Limits churn when historical keys were not canonicalized.
                let rows = sqlx::query(
                    "SELECT id, key, value FROM facts WHERE category = ? AND superseded_at IS NULL ORDER BY updated_at DESC",
                )
                .bind(category_clean)
                .fetch_all(&self.pool)
                .await?;
                for row in rows {
                    let existing_key: String = row.get("key");
                    if canonicalize_fact_key(&existing_key) == canonical_key {
                        existing = Some((row.get("id"), existing_key, row.get("value")));
                        break;
                    }
                }
            }
        }

        // Semantic dedup: if key-based matching found nothing, check if any active
        // fact in the same category is semantically equivalent (different key, same
        // concept). Catches synonym keys like "editor" vs "preferred_editor".
        //
        // We compare key-only embeddings (category + readable key) rather than
        // full fact embeddings (which include value), because dedup should detect
        // that two keys refer to the same concept regardless of their values.
        if existing.is_none() {
            let new_key_text = build_dedup_key_text(category_clean, key_clean);
            if let Ok(new_vec) = self.embedding_service.embed(new_key_text).await {
                let category_facts = sqlx::query(
                    "SELECT id, key, value FROM facts
                     WHERE category = ? AND superseded_at IS NULL",
                )
                .bind(category_clean)
                .fetch_all(&self.pool)
                .await?;

                let new_tokens: std::collections::HashSet<&str> =
                    canonical_key.split('_').filter(|t| !t.is_empty()).collect();

                let mut best_match: Option<(i64, String, String, f32)> = None;
                for row in &category_facts {
                    let existing_key: String = row.get("key");
                    let existing_canonical = canonicalize_fact_key(&existing_key);

                    // Guard against false semantic matches for keys that are
                    // structurally similar but intentionally distinct.
                    //
                    // If keys share SOME tokens, have the same token count, but
                    // differ in at least one token, they're variants of the same
                    // base concept (e.g., "dog_name_old" vs "dog_name_new") — skip.
                    //
                    // But allow through:
                    // - Keys that are a subset/superset (modifier pattern):
                    //   "editor" vs "preferred_editor" — different count, proceed.
                    // - Keys that are just reordered (same tokens, same count):
                    //   "company_previous" vs "previous_company" — all shared, proceed.
                    // - Keys with zero overlap: completely different concepts.
                    let existing_tokens: std::collections::HashSet<&str> = existing_canonical
                        .split('_')
                        .filter(|t| !t.is_empty())
                        .collect();
                    let shared = new_tokens.intersection(&existing_tokens).count();
                    let new_count = new_tokens.len();
                    let existing_count = existing_tokens.len();

                    // Skip when both keys are single-token — short keys embed
                    // too similarly in MiniLM (e.g., "sem0" vs "sem1", "age"
                    // vs "name"). Canonical scan already handles single-word
                    // matches. Semantic dedup is for multi-word synonym patterns.
                    if new_count <= 1 && existing_count <= 1 {
                        continue;
                    }

                    // Skip when keys have partial overlap that indicates
                    // intentionally distinct variants rather than synonyms.
                    //
                    // Allow through (don't skip):
                    // - Zero overlap: different concept names that might be
                    //   synonyms (e.g., no shared tokens at all).
                    // - Subset/superset: one key's tokens are fully contained
                    //   in the other (modifier pattern: "editor" ⊂
                    //   "preferred_editor"). The extra token is a qualifier.
                    // - Full overlap with reordering: same tokens, different
                    //   order (e.g., "company_previous" vs "previous_company").
                    //
                    // Skip (continue):
                    // - Partial overlap where neither is a subset of the other:
                    //   keys share a base but differ in distinguishing tokens
                    //   (e.g., "dog_name_old" vs "dog_name_new" — shared
                    //   {"dog","name"} but "old" vs "new" differ).
                    let is_subset = new_tokens.is_subset(&existing_tokens)
                        || existing_tokens.is_subset(&new_tokens);
                    if shared > 0 && !is_subset {
                        continue;
                    }

                    // Entity-scoping guard: prevent cross-entity dedup.
                    //
                    // When one key is a strict superset of the other (e.g.,
                    // "carlos_birthday" ⊃ "birthday"), the extra tokens could
                    // be either a modifier (safe to dedup: "preferred_editor"
                    // vs "editor") or an entity prefix (dangerous: "carlos_birthday"
                    // vs "birthday" belong to different people).
                    //
                    // We allow dedup only when ALL extra tokens are known
                    // qualifiers/modifiers. If any extra token is unknown
                    // (likely a person name or entity identifier), skip dedup.
                    if is_subset && new_count != existing_count {
                        const SAFE_MODIFIERS: &[&str] = &[
                            "preferred",
                            "favorite",
                            "fav",
                            "default",
                            "primary",
                            "current",
                            "previous",
                            "old",
                            "new",
                            "last",
                            "first",
                            "main",
                            "secondary",
                            "alternate",
                            "alt",
                            "other",
                            "work",
                            "home",
                            "personal",
                            "daily",
                            "weekly",
                            "morning",
                            "evening",
                            "night",
                        ];
                        let (larger, smaller) = if new_count > existing_count {
                            (&new_tokens, &existing_tokens)
                        } else {
                            (&existing_tokens, &new_tokens)
                        };
                        let extra: Vec<&&str> = larger.difference(smaller).collect();
                        let all_modifiers = extra.iter().all(|t| SAFE_MODIFIERS.contains(t));
                        if !all_modifiers {
                            continue;
                        }
                    }

                    let existing_key_text = build_dedup_key_text(category_clean, &existing_key);
                    if let Ok(existing_vec) = self.embedding_service.embed(existing_key_text).await
                    {
                        let sim = crate::memory::math::cosine_similarity(&new_vec, &existing_vec);
                        if sim > 0.85 {
                            let score = best_match.as_ref().map(|m| m.3).unwrap_or(0.0);
                            if sim > score {
                                best_match = Some((
                                    row.get("id"),
                                    existing_key.clone(),
                                    row.get::<String, _>("value"),
                                    sim,
                                ));
                            }
                        }
                    }
                }

                if let Some((id, matched_key, matched_value, sim)) = best_match {
                    tracing::info!(
                        category = category_clean,
                        new_key = key_clean,
                        matched_key = matched_key.as_str(),
                        similarity = sim,
                        "Semantic dedup: matched existing fact by embedding similarity"
                    );
                    existing = Some((id, matched_key, matched_value));
                }
            }
        }

        let key_for_write = existing
            .as_ref()
            .map(|(_, k, _)| k.clone())
            .unwrap_or_else(|| {
                if canonical_key.is_empty() {
                    key_clean.to_string()
                } else {
                    canonical_key.clone()
                }
            });

        // Pre-compute embedding for the fact text (best-effort).
        let fact_text = build_fact_embedding_text(category_clean, &key_for_write, value);
        let embedding_blob = self
            .embedding_service
            .embed(fact_text)
            .await
            .ok()
            .map(|v| encode_embedding(&v));

        if let Some((old_id, _old_key, old_value)) = &existing {
            // If the value is different, mark old as superseded and insert new.
            //
            // Source priority guard: consolidation-derived facts MUST NOT
            // overwrite facts that were stored by the user (via remember_fact,
            // user_message, etc.).  The consolidation pipeline processes
            // conversation history and can inadvertently extract wrong values
            // from hallucinated assistant messages.  User-originated facts are
            // always ground truth.
            if old_value != value && source == "consolidation" {
                let existing_source: Option<String> =
                    sqlx::query_scalar("SELECT source FROM facts WHERE id = ?")
                        .bind(old_id)
                        .fetch_optional(&self.pool)
                        .await?;
                let is_user_originated = existing_source
                    .as_deref()
                    .is_some_and(|s| s != "consolidation");
                if is_user_originated {
                    tracing::info!(
                        category = category_clean,
                        key = key_clean,
                        existing_source = ?existing_source,
                        "Consolidation skipped: refusing to overwrite user-originated fact"
                    );
                    return Ok(());
                }
            }
            if old_value != value {
                sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
                    .bind(&now)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;

                // Insert new fact with embedding — ignore duplicate entry errors (code 2067)
                // that can occur due to active-unique constraint race conditions.
                let insert_result = sqlx::query(
                    "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy, embedding, first_seen_at, source_excerpt)
                     VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                )
                .bind(category_clean)
                .bind(&key_for_write)
                .bind(value)
                .bind(source)
                .bind(&now)
                .bind(&now)
                .bind(channel_id)
                .bind(&privacy_str)
                .bind(&embedding_blob)
                .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
                .bind(source_excerpt)
                .execute(&self.pool)
                .await;

                match insert_result {
                    Ok(_) => {}
                    Err(sqlx::Error::Database(ref db_err))
                        if db_err.code().as_deref() == Some("2067") =>
                    {
                        // Duplicate entry — another active row already exists (concurrent upsert).
                        // Update the active row to "last write wins" without resurrecting the
                        // superseded row (which would violate partial unique indexes).
                        let updated = sqlx::query(
                            "UPDATE facts
                             SET value = ?, source = ?, updated_at = ?,
                                 channel_id = ?, privacy = ?,
                                 embedding = COALESCE(?, embedding),
                                 first_seen_at = COALESCE(?, first_seen_at),
                                 source_excerpt = COALESCE(?, source_excerpt)
                             WHERE category = ? AND key = ? AND superseded_at IS NULL",
                        )
                        .bind(value)
                        .bind(source)
                        .bind(&now)
                        .bind(channel_id)
                        .bind(&privacy_str)
                        .bind(&embedding_blob)
                        .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
                        .bind(source_excerpt)
                        .bind(category_clean)
                        .bind(&key_for_write)
                        .execute(&self.pool)
                        .await?;

                        // Legacy fallback: if no active row exists (e.g., old UNIQUE(category,key)
                        // constraint), keep data consistent by updating in-place.
                        if updated.rows_affected() == 0 {
                            sqlx::query(
                                "UPDATE facts
                                 SET value = ?, source = ?, updated_at = ?, superseded_at = NULL,
                                     channel_id = ?, privacy = ?,
                                     embedding = COALESCE(?, embedding),
                                     first_seen_at = COALESCE(?, first_seen_at),
                                     source_excerpt = COALESCE(?, source_excerpt)
                                 WHERE id = ?",
                            )
                            .bind(value)
                            .bind(source)
                            .bind(&now)
                            .bind(channel_id)
                            .bind(&privacy_str)
                            .bind(&embedding_blob)
                            .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
                            .bind(source_excerpt)
                            .bind(old_id)
                            .execute(&self.pool)
                            .await?;
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            } else {
                // Same value — update timestamp and backfill embedding.
                // When the new source is consolidation, keep the original
                // (higher-trust) source intact. Only promote the source when
                // the caller is a user-originated path (remember_fact, etc.).
                if source == "consolidation" {
                    sqlx::query(
                        "UPDATE facts SET updated_at = ?, embedding = COALESCE(embedding, ?) WHERE id = ?",
                    )
                    .bind(&now)
                    .bind(&embedding_blob)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;
                } else {
                    sqlx::query(
                        "UPDATE facts SET source = ?, updated_at = ?, embedding = COALESCE(embedding, ?), first_seen_at = COALESCE(first_seen_at, ?), source_excerpt = COALESCE(source_excerpt, ?) WHERE id = ?",
                    )
                    .bind(source)
                    .bind(&now)
                    .bind(&embedding_blob)
                    .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
                    .bind(source_excerpt)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;
                }
            }
        } else {
            // No existing fact - insert new with embedding
            // Ignore duplicate entry errors (code 2067) from concurrent inserts
            let insert_result = sqlx::query(
                "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy, embedding, first_seen_at, source_excerpt)
                 VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
            )
            .bind(category_clean)
            .bind(&key_for_write)
            .bind(value)
            .bind(source)
            .bind(&now)
            .bind(&now)
            .bind(channel_id)
            .bind(&privacy_str)
            .bind(&embedding_blob)
            .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
            .bind(source_excerpt)
            .execute(&self.pool)
            .await;

            match insert_result {
                Ok(_) => {}
                Err(sqlx::Error::Database(ref db_err))
                    if db_err.code().as_deref() == Some("2067") =>
                {
                    // Duplicate entry — another active row already exists (concurrent upsert).
                    // Update the active row to "last write wins".
                    let updated = sqlx::query(
                        "UPDATE facts
                         SET value = ?, source = ?, updated_at = ?,
                             channel_id = ?, privacy = ?,
                             embedding = COALESCE(?, embedding),
                             first_seen_at = COALESCE(?, first_seen_at),
                             source_excerpt = COALESCE(?, source_excerpt)
                         WHERE category = ? AND key = ? AND superseded_at IS NULL",
                    )
                    .bind(value)
                    .bind(source)
                    .bind(&now)
                    .bind(channel_id)
                    .bind(&privacy_str)
                    .bind(&embedding_blob)
                    .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
                    .bind(source_excerpt)
                    .bind(category_clean)
                    .bind(&key_for_write)
                    .execute(&self.pool)
                    .await?;

                    if updated.rows_affected() == 0 {
                        // If there is no active row, update the most recent version in-place.
                        // This can happen on legacy schemas where a superseded row still blocks
                        // inserts (old UNIQUE(category,key) constraint).
                        if let Some(row) = sqlx::query(
                            "SELECT id FROM facts WHERE category = ? AND key = ? ORDER BY updated_at DESC LIMIT 1",
                        )
                        .bind(category_clean)
                        .bind(&key_for_write)
                        .fetch_optional(&self.pool)
                        .await?
                        {
                            let id: i64 = row.get("id");
                            sqlx::query(
                                "UPDATE facts
                                 SET value = ?, source = ?, updated_at = ?, superseded_at = NULL,
                                     channel_id = ?, privacy = ?,
                                     embedding = COALESCE(?, embedding),
                                     first_seen_at = COALESCE(?, first_seen_at),
                                     source_excerpt = COALESCE(?, source_excerpt)
                                 WHERE id = ?",
                            )
                            .bind(value)
                            .bind(source)
                            .bind(&now)
                            .bind(channel_id)
                            .bind(&privacy_str)
                            .bind(&embedding_blob)
                            .bind(first_seen_at.as_ref().map(|dt| dt.to_rfc3339()))
                            .bind(source_excerpt)
                            .bind(id)
                            .execute(&self.pool)
                            .await?;
                        }
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }

        // ── Post-store conflict sweep ──────────────────────────────────
        //
        // When the user explicitly stores a fact via remember_fact (source
        // == "agent"), scan other active facts in the same category for
        // semantically similar entries that represent stale/conflicting
        // values for the same concept stored under a different key.
        //
        // Example: user says "my coffee is cappuccino" → stored as
        // (preference, coffee_preference).  We also supersede
        // (preference, beverage = "coffee (cold brew)") which is the same
        // concept under a different key that escaped key-level dedup.
        //
        // Guards:
        // - Only for source=="agent" (user-explicit, not consolidation/test)
        // - Full-fact embedding similarity > 0.82 (stricter than key dedup)
        // - Lexical overlap: at least one significant word from the new
        //   fact's key or value must appear in the other fact's key or value
        //   (prevents cross-concept supersession in the same category)
        if source == "agent" {
            if let Some(ref new_emb_blob) = embedding_blob {
                if let Ok(new_vec) = decode_embedding(new_emb_blob) {
                    let others = sqlx::query(
                        "SELECT id, key, value, embedding FROM facts
                         WHERE category = ? AND superseded_at IS NULL
                           AND key != ?
                           AND embedding IS NOT NULL",
                    )
                    .bind(category_clean)
                    .bind(&key_for_write)
                    .fetch_all(&self.pool)
                    .await
                    .unwrap_or_default();

                    // Build keyword set from the new fact's key + value for
                    // lexical overlap guard.  Only words ≥3 chars to skip
                    // noise like "a", "is", "my".
                    let new_words: std::collections::HashSet<String> = key_clean
                        .split(|c: char| !c.is_alphanumeric())
                        .chain(value.split(|c: char| !c.is_alphanumeric()))
                        .filter(|w| w.len() >= 3)
                        .map(|w| w.to_ascii_lowercase())
                        .collect();

                    for row in &others {
                        let other_id: i64 = row.get("id");
                        let other_key: String = row.get("key");
                        let other_value: String = row.get("value");
                        let blob: Vec<u8> = row.get("embedding");

                        // Lexical overlap guard: at least one meaningful word
                        // from the new fact must appear in the other fact.
                        let other_text =
                            format!("{} {}", other_key, other_value).to_ascii_lowercase();
                        let has_overlap = new_words.iter().any(|w| other_text.contains(w.as_str()));
                        if !has_overlap {
                            continue;
                        }

                        if let Ok(other_vec) = decode_embedding(&blob) {
                            let sim = crate::memory::math::cosine_similarity(&new_vec, &other_vec);
                            // High similarity on full-fact embedding (value-aware)
                            // means these facts describe the same concept.  Only
                            // supersede if the values actually differ (otherwise
                            // it's a harmless duplicate that adds context).
                            if sim > 0.82 && other_value != value {
                                tracing::info!(
                                    category = category_clean,
                                    new_key = key_clean,
                                    new_value = value,
                                    conflicting_key = other_key.as_str(),
                                    conflicting_value = other_value.as_str(),
                                    similarity = sim,
                                    "Conflict sweep: superseding stale fact with different key"
                                );
                                let _ =
                                    sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
                                        .bind(&now)
                                        .bind(other_id)
                                        .execute(&self.pool)
                                        .await;
                            }
                        }
                    }
                }
            }
        }

        if let Err(error) = self.sync_fact_memory_category(category_clean).await {
            tracing::warn!(%error, category = category_clean, "Deferred fact memory projection");
        }
        Ok(())
    }

    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        // Only return current (non-superseded) facts
        let rows = if let Some(cat) = category {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, first_seen_at, source_excerpt FROM facts WHERE category = ? AND superseded_at IS NULL ORDER BY updated_at DESC")
                .bind(cat)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, first_seen_at, source_excerpt FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC")
                .fetch_all(&self.pool)
                .await?
        };

        let mut facts = Vec::with_capacity(rows.len());
        for row in rows {
            facts.push(Self::row_to_fact(&row));
        }
        Ok(facts)
    }

    async fn get_relevant_facts(&self, query: &str, max: usize) -> anyhow::Result<Vec<Fact>> {
        // Load facts with stored embeddings
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding, first_seen_at, source_excerpt
             FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let all_facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();

        if all_facts.is_empty() || query.trim().is_empty() {
            let mut facts = all_facts;
            facts.truncate(max);
            bump_fact_recall(&self.pool, &facts).await;
            let canonical = self.canonical_personal_facts().await.unwrap_or_default();
            return Ok(prefer_canonical_facts(canonical, facts, query, max));
        }

        // Exhaustive queries ("tell me everything", "what do you know about me")
        // bypass semantic search and return all facts to avoid retrieval gaps.
        if is_exhaustive_query(query) {
            tracing::info!(
                query = query,
                total_facts = all_facts.len(),
                "Exhaustive query detected — returning all facts without scoring"
            );
            let mut facts = all_facts;
            facts.truncate(max);
            bump_fact_recall(&self.pool, &facts).await;
            let canonical = self.canonical_personal_facts().await.unwrap_or_default();
            return Ok(prefer_canonical_facts(canonical, facts, query, max));
        }

        // Embed the query
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "Failed to embed query for fact filtering, returning all facts: {}",
                    e
                );
                let mut facts = all_facts;
                facts.truncate(max);
                bump_fact_recall(&self.pool, &facts).await;
                let canonical = self.canonical_personal_facts().await.unwrap_or_default();
                return Ok(prefer_canonical_facts(canonical, facts, query, max));
            }
        };

        let now = Utc::now();
        let query_lower = query.to_lowercase();
        let tokens = query_tokens(&query_lower);
        let indexed_scores = self
            .fact_embedding_scores(&query_vec, rows.len())
            .await
            .unwrap_or_default();
        let fts_fact_ids: std::collections::HashSet<i64> = self
            .search_memory_claims(
                query,
                None,
                ChannelVisibility::Internal,
                true,
                max.saturating_mul(4).max(16),
            )
            .await
            .unwrap_or_default()
            .into_iter()
            .filter_map(|claim| claim.source_fact_id)
            .collect();
        let graph_fact_ids = self
            .graph_fact_ids_for_query(query)
            .await
            .unwrap_or_default();

        // Score facts using stored embeddings, with a small recency boost for sorting.
        // IMPORTANT: the recency boost must not change which facts pass the semantic threshold.
        let mut candidates: Vec<(usize, f32, f32, bool)> = Vec::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            let fact = &all_facts[i];
            let freshness = fact_freshness_boost(now, fact.updated_at);

            let semantic = indexed_scores.get(&fact.id).copied().or_else(|| {
                row.get::<Option<Vec<u8>>, _>("embedding")
                    .and_then(|blob| decode_embedding(&blob).ok())
                    .map(|vec| crate::memory::math::cosine_similarity(&query_vec, &vec))
            });
            let lexical = if fts_fact_ids.contains(&fact.id) {
                FACT_LEXICAL_MAX_SCORE
            } else {
                lexical_fallback_score(&query_lower, &tokens, fact)
            };
            if let Some(semantic) = semantic {
                if semantic > FACT_SEMANTIC_MIN_SCORE {
                    candidates.push((
                        i,
                        semantic,
                        hybrid_fact_score(
                            fact,
                            semantic,
                            lexical,
                            graph_fact_ids.contains(&fact.id),
                            freshness,
                        ),
                        true,
                    ));
                    continue;
                }
                // Below semantic threshold — try lexical as fallback for keyword matches
                let best = semantic.max(lexical);
                let is_semantic = best == semantic;
                candidates.push((
                    i,
                    best,
                    hybrid_fact_score(
                        fact,
                        semantic,
                        lexical,
                        graph_fact_ids.contains(&fact.id),
                        freshness,
                    ),
                    is_semantic,
                ));
                continue;
            }

            // Missing/invalid embedding: fall back to cheap lexical relevance so
            // freshly saved facts can still be retrieved during embedding backfill.
            candidates.push((
                i,
                lexical,
                hybrid_fact_score(
                    fact,
                    0.0,
                    lexical,
                    graph_fact_ids.contains(&fact.id),
                    freshness,
                ),
                false,
            ));
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut relevant: Vec<Fact> = Vec::with_capacity(max.min(all_facts.len()));
        let mut seen_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (i, base_score, _sort_score, is_semantic) in candidates.into_iter() {
            if relevant.len() >= max {
                break;
            }
            let min_score = if is_semantic {
                FACT_SEMANTIC_MIN_SCORE
            } else {
                FACT_LEXICAL_MIN_SCORE
            };
            if base_score > min_score {
                let fact = all_facts[i].clone();
                if seen_ids.insert(fact.id) {
                    relevant.push(fact);
                }
            }
        }

        // If filtering left us with very few facts, pad with most recent ones
        if FACT_PAD_LOW_CONFIDENCE_RESULTS
            && relevant.len() < max / 3
            && all_facts.len() > relevant.len()
        {
            for fact in &all_facts {
                if relevant.len() >= max {
                    break;
                }
                if !seen_ids.contains(&fact.id) {
                    seen_ids.insert(fact.id);
                    relevant.push(fact.clone());
                }
            }
        }

        bump_fact_recall(&self.pool, &relevant).await;
        let canonical = self.canonical_personal_facts().await.unwrap_or_default();
        Ok(prefer_canonical_facts(canonical, relevant, query, max))
    }

    async fn search_facts_semantic(
        &self,
        query: &str,
        max: usize,
    ) -> anyhow::Result<Vec<(Fact, f32)>> {
        if query.trim().is_empty() || max == 0 {
            return Ok(vec![]);
        }

        // Embed the query; if embedding is unavailable, there is no semantic
        // signal to contribute (the caller still has its lexical results).
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("Semantic fact search: embedding failed: {}", e);
                return Ok(vec![]);
            }
        };

        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding, first_seen_at, source_excerpt
             FROM facts WHERE superseded_at IS NULL",
        )
        .fetch_all(&self.pool)
        .await?;

        // Stage 1 — bi-encoder candidate retrieval. Keep everything above the
        // recall-oriented cutoff so near-synonyms ("spouse"→"partner", ~0.28)
        // enter the candidate pool, then cap the pool by cosine so reranking
        // stays bounded. The pool must be deep enough that a weakly-scoring but
        // correct fact (measured: the answer can sit ~rank 30) isn't dropped.
        let indexed_scores = self
            .fact_embedding_scores(&query_vec, rows.len())
            .await
            .unwrap_or_default();
        let mut scored: Vec<(Fact, f32)> = Vec::new();
        for row in &rows {
            let fact = Self::row_to_fact(row);
            let semantic = indexed_scores.get(&fact.id).copied().or_else(|| {
                row.get::<Option<Vec<u8>>, _>("embedding")
                    .and_then(|blob| decode_embedding(&blob).ok())
                    .map(|vec| crate::memory::math::cosine_similarity(&query_vec, &vec))
            });
            if let Some(semantic) = semantic {
                if semantic > EXPLICIT_SEARCH_SEMANTIC_MIN_SCORE {
                    scored.push((fact, semantic));
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(EXPLICIT_SEARCH_CANDIDATE_POOL);

        if scored.len() <= 1 {
            scored.truncate(max);
            return Ok(scored);
        }

        // Stage 2 — cross-encoder rerank. The bi-encoder ranks attribute facts
        // ("wife covers insurance") above the answer-bearing identity fact
        // ("partner name: Alice") for queries like "spouse"; a cross-encoder
        // reads (query, fact) together and reorders correctly. On any reranker
        // error (e.g. model unavailable offline), fall back to the bi-encoder
        // order so search still works.
        let pool_size = scored.len();
        let cosine_top_id = scored.first().map(|(f, _)| f.id);
        let docs: Vec<String> = scored
            .iter()
            .map(|(f, _)| build_fact_embedding_text(&f.category, &f.key, &f.value))
            .collect();
        let started = std::time::Instant::now();
        match self.embedding_service.rerank(query.to_string(), docs).await {
            Ok(ranked) => {
                let mut out: Vec<(Fact, f32)> = Vec::with_capacity(max);
                for (idx, score) in ranked.into_iter().take(max) {
                    if let Some((fact, _)) = scored.get(idx) {
                        out.push((fact.clone(), score));
                    }
                }
                // Telemetry: reranker cost (latency), impact (did it change the top
                // result vs the bi-encoder), and the top results with scores — so
                // recall quality is observable without ad-hoc measurement.
                let reordered = out.first().map(|(f, _)| f.id) != cosine_top_id;
                let top = out
                    .iter()
                    .take(3)
                    .map(|(f, s)| format!("{}={:.3}", f.key, s))
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::info!(
                    target: "memory_recall",
                    candidate_pool = pool_size,
                    returned = out.len(),
                    rerank_ms = started.elapsed().as_millis() as u64,
                    reordered,
                    fallback = false,
                    top = %top,
                    "explicit fact search reranked"
                );
                Ok(out)
            }
            Err(e) => {
                tracing::warn!(
                    target: "memory_recall",
                    candidate_pool = pool_size,
                    rerank_ms = started.elapsed().as_millis() as u64,
                    fallback = true,
                    error = %e,
                    "explicit fact search rerank unavailable; using bi-encoder order"
                );
                scored.truncate(max);
                Ok(scored)
            }
        }
    }

    async fn get_relevant_facts_for_channel(
        &self,
        query: &str,
        max: usize,
        channel_id: Option<&str>,
        visibility: ChannelVisibility,
        requester_is_owner: bool,
    ) -> anyhow::Result<Vec<Fact>> {
        // In DM/Internal contexts, use semantic relevance search so that only
        // facts related to the current query are injected into the prompt.
        // Previously this called get_facts(None) which returned ALL facts
        // without filtering, causing unrelated facts to bleed into context.
        //
        // SECURITY: only the OWNER gets the unfiltered graph in a DM. A non-owner
        // (allowlisted Guest) DMing the bot must NOT receive Private or
        // other-channel facts — fall through to the privacy filter below, same as
        // a group channel. Without this gate, a guest's prompt would be injected
        // with the owner's private memory.
        if requester_is_owner
            && matches!(
                visibility,
                ChannelVisibility::Private | ChannelVisibility::Internal
            )
        {
            return self.get_relevant_facts(query, max).await;
        }

        // PublicExternal: do NOT inject any stored facts (treat as untrusted).
        if matches!(visibility, ChannelVisibility::PublicExternal) {
            return Ok(vec![]);
        }

        // Public/PrivateGroup: global + same-channel facts (no private, no other-channel)
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding, first_seen_at, source_excerpt
             FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        // Build facts and track which indices pass the privacy filter
        let all_facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();
        let filtered_indices: Vec<usize> = all_facts
            .iter()
            .enumerate()
            .filter(|(_, f)| match f.privacy {
                FactPrivacy::Private => false,
                FactPrivacy::Global => {
                    if matches!(visibility, ChannelVisibility::PublicExternal) {
                        !matches!(f.category.as_str(), "personal" | "health" | "finance")
                    } else {
                        true
                    }
                }
                FactPrivacy::Channel => match (channel_id, &f.channel_id) {
                    (Some(current), Some(fact_ch)) => {
                        crate::session::stored_channel_matches_current(fact_ch, current)
                    }
                    (None, None) => true,
                    _ => false,
                },
            })
            .map(|(i, _)| i)
            .collect();

        let filtered: Vec<Fact> = filtered_indices
            .iter()
            .map(|&i| all_facts[i].clone())
            .collect();

        if filtered.is_empty() || query.trim().is_empty() {
            let mut facts = filtered;
            facts.truncate(max);
            bump_fact_recall(&self.pool, &facts).await;
            return Ok(facts);
        }

        // Apply semantic filtering using stored embeddings
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => {
                let mut facts = filtered;
                facts.truncate(max);
                bump_fact_recall(&self.pool, &facts).await;
                return Ok(facts);
            }
        };

        let now = Utc::now();
        let query_lower = query.to_lowercase();
        let tokens = query_tokens(&query_lower);
        let indexed_scores = self
            .fact_embedding_scores(&query_vec, rows.len())
            .await
            .unwrap_or_default();
        let fts_fact_ids: std::collections::HashSet<i64> = self
            .search_memory_claims(
                query,
                channel_id,
                visibility,
                requester_is_owner,
                max.saturating_mul(4).max(16),
            )
            .await
            .unwrap_or_default()
            .into_iter()
            .filter_map(|claim| claim.source_fact_id)
            .collect();
        let graph_fact_ids = self
            .graph_fact_ids_for_query(query)
            .await
            .unwrap_or_default();

        let mut candidates: Vec<(usize, f32, f32, bool)> = Vec::with_capacity(filtered.len());
        for (fi, &ri) in filtered_indices.iter().enumerate() {
            let fact = &filtered[fi];
            let freshness = fact_freshness_boost(now, fact.updated_at);

            let semantic = indexed_scores.get(&fact.id).copied().or_else(|| {
                rows[ri]
                    .get::<Option<Vec<u8>>, _>("embedding")
                    .and_then(|blob| decode_embedding(&blob).ok())
                    .map(|vec| crate::memory::math::cosine_similarity(&query_vec, &vec))
            });
            let lexical = if fts_fact_ids.contains(&fact.id) {
                FACT_LEXICAL_MAX_SCORE
            } else {
                lexical_fallback_score(&query_lower, &tokens, fact)
            };
            if let Some(semantic) = semantic {
                if semantic > FACT_SEMANTIC_MIN_SCORE {
                    candidates.push((
                        fi,
                        semantic,
                        hybrid_fact_score(
                            fact,
                            semantic,
                            lexical,
                            graph_fact_ids.contains(&fact.id),
                            freshness,
                        ),
                        true,
                    ));
                    continue;
                }
                // Below semantic threshold — try lexical as fallback for keyword matches
                let best = semantic.max(lexical);
                let is_semantic = best == semantic;
                candidates.push((
                    fi,
                    best,
                    hybrid_fact_score(
                        fact,
                        semantic,
                        lexical,
                        graph_fact_ids.contains(&fact.id),
                        freshness,
                    ),
                    is_semantic,
                ));
                continue;
            }

            candidates.push((
                fi,
                lexical,
                hybrid_fact_score(
                    fact,
                    0.0,
                    lexical,
                    graph_fact_ids.contains(&fact.id),
                    freshness,
                ),
                false,
            ));
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut relevant: Vec<Fact> = Vec::with_capacity(max.min(filtered.len()));
        let mut seen_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for (fi, base_score, _sort_score, is_semantic) in candidates.into_iter() {
            if relevant.len() >= max {
                break;
            }
            let min_score = if is_semantic {
                FACT_SEMANTIC_MIN_SCORE
            } else {
                FACT_LEXICAL_MIN_SCORE
            };
            if base_score > min_score {
                let fact = filtered[fi].clone();
                if seen_ids.insert(fact.id) {
                    relevant.push(fact);
                }
            }
        }

        if FACT_PAD_LOW_CONFIDENCE_RESULTS
            && relevant.len() < max / 3
            && filtered.len() > relevant.len()
        {
            for fact in &filtered {
                if relevant.len() >= max {
                    break;
                }
                if !seen_ids.contains(&fact.id) {
                    seen_ids.insert(fact.id);
                    relevant.push(fact.clone());
                }
            }
        }

        bump_fact_recall(&self.pool, &relevant).await;
        Ok(relevant)
    }

    async fn get_cross_channel_hints(
        &self,
        query: &str,
        current_channel_id: &str,
        max: usize,
    ) -> anyhow::Result<Vec<Fact>> {
        // Get channel-scoped facts from OTHER channels that are relevant to the query
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding, first_seen_at, source_excerpt
             FROM facts
             WHERE superseded_at IS NULL
               AND privacy = 'channel'
               AND channel_id IS NOT NULL
             ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            return Ok(vec![]);
        }

        let filtered_rows: Vec<_> = rows
            .into_iter()
            .filter(|row| {
                row.try_get::<Option<String>, _>("channel_id")
                    .ok()
                    .flatten()
                    .is_some_and(|stored_channel| {
                        !crate::session::stored_channel_matches_current(
                            &stored_channel,
                            current_channel_id,
                        )
                    })
            })
            .collect();
        let facts: Vec<Fact> = filtered_rows.iter().map(Self::row_to_fact).collect();

        // Apply semantic filtering using stored embeddings
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        };

        let mut scored: Vec<(usize, f32)> = Vec::new();
        let indexed_scores = self
            .fact_embedding_scores(&query_vec, filtered_rows.len().saturating_mul(8).max(64))
            .await
            .unwrap_or_default();
        for (i, row) in filtered_rows.iter().enumerate() {
            let score = indexed_scores.get(&facts[i].id).copied().or_else(|| {
                row.get::<Option<Vec<u8>>, _>("embedding")
                    .and_then(|blob| decode_embedding(&blob).ok())
                    .map(|vec| crate::memory::math::cosine_similarity(&query_vec, &vec))
            });
            if let Some(score) = score {
                scored.push((i, score));
            }
            // Facts without embeddings are skipped for cross-channel hints (conservative)
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hints: Vec<Fact> = scored
            .into_iter()
            .filter(|(_, score)| *score > 0.6) // Higher threshold for cross-channel hints
            .take(max)
            .map(|(i, _)| facts[i].clone())
            .collect();

        bump_fact_recall(&self.pool, &hints).await;
        Ok(hints)
    }

    async fn update_fact_privacy(&self, fact_id: i64, privacy: FactPrivacy) -> anyhow::Result<()> {
        sqlx::query("UPDATE facts SET privacy = ? WHERE id = ?")
            .bind(privacy.to_string())
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        if let Err(error) = self.project_fact_memory(fact_id).await {
            tracing::warn!(%error, fact_id, "Deferred fact privacy projection");
        }
        Ok(())
    }

    async fn delete_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
            .bind(&now)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        if let Err(error) = self.project_fact_memory(fact_id).await {
            tracing::warn!(%error, fact_id, "Deferred deleted-fact projection");
        }
        Ok(())
    }

    async fn delete_fact_by_key(&self, category: &str, key: &str) -> anyhow::Result<bool> {
        let now = Utc::now().to_rfc3339();
        // Canonicalize: lowercase, trim, strip [brackets]
        let cat_clean = category.trim().to_lowercase();
        let key_clean = key.trim().to_lowercase();
        let ids: Vec<i64> = sqlx::query_scalar(
            "SELECT id FROM facts WHERE LOWER(TRIM(category)) = ? AND LOWER(TRIM(key)) = ? AND superseded_at IS NULL",
        )
        .bind(&cat_clean)
        .bind(&key_clean)
        .fetch_all(&self.pool)
        .await?;
        let result = sqlx::query(
            "UPDATE facts SET superseded_at = ? WHERE LOWER(TRIM(category)) = ? AND LOWER(TRIM(key)) = ? AND superseded_at IS NULL",
        )
        .bind(&now)
        .bind(&cat_clean)
        .bind(&key_clean)
        .execute(&self.pool)
        .await?;
        for id in ids {
            if let Err(error) = self.project_fact_memory(id).await {
                tracing::warn!(%error, fact_id = id, "Deferred deleted-fact projection");
            }
        }
        Ok(result.rows_affected() > 0)
    }

    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<Fact>> {
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, first_seen_at, source_excerpt
             FROM facts WHERE superseded_at IS NULL ORDER BY category, key"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_fact).collect())
    }

    async fn assemble_neighborhood(
        &self,
        entities: &[String],
        initial_ids: &std::collections::HashSet<i64>,
    ) -> anyhow::Result<Vec<Fact>> {
        use crate::memory::neighborhood::{
            entity_mentioned_as_words, is_relationship_key, select_neighborhood_facts,
            NeighborhoodCaps,
        };
        use crate::traits::PeopleStore;

        if entities.is_empty() {
            return Ok(vec![]);
        }

        let all = self
            .get_all_facts_with_provenance()
            .await
            .unwrap_or_default();
        let people = PeopleStore::get_all_people(self).await.unwrap_or_default();

        let folded = |s: &str| s.to_ascii_lowercase();
        let mut resolved: Vec<String> = Vec::new();
        let mut owner_relationship = false;

        // If any seed fact (from the initial embedding hit) has a relationship key,
        // the search context is already family/relationship-oriented — enable the
        // owner-relationship cluster so all co-relations travel together.
        if all
            .iter()
            .filter(|f| initial_ids.contains(&f.id))
            .any(|f| is_relationship_key(&f.key))
        {
            owner_relationship = true;
        }

        for ent in entities {
            let ef = folded(ent);
            if let Some(p) = people.iter().find(|p| {
                entity_mentioned_as_words(&p.name, ent)
                    || p.aliases.iter().any(|a| entity_mentioned_as_words(a, ent))
            }) {
                resolved.push(p.name.clone());
                // A resolved person that has a relationship role flips on the
                // owner-relationship cluster rule so co-relations travel together.
                if p.relationship.is_some() {
                    owner_relationship = true;
                }
            } else {
                resolved.push(ent.clone());
            }
            // If the entity name itself matches a relationship word (e.g. "mom",
            // "partner"), also enable the owner cluster.
            if is_relationship_key(&ef) {
                owner_relationship = true;
            }
        }

        Ok(select_neighborhood_facts(
            &all,
            &resolved,
            owner_relationship,
            initial_ids,
            NeighborhoodCaps::default(),
        ))
    }
}

#[cfg(test)]
mod assemble_neighborhood_tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::traits::{FactStore, PeopleStore};
    use crate::types::FactPrivacy;
    use std::sync::Arc;

    async fn setup_store() -> (SqliteStateStore, tempfile::NamedTempFile) {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let store = SqliteStateStore::new(
            db_file.path().to_str().unwrap(),
            100,
            None,
            embedding_service,
        )
        .await
        .unwrap();
        (store, db_file)
    }

    #[tokio::test]
    async fn assemble_neighborhood_pulls_owner_family_cluster() {
        let (store, _db) = setup_store().await;

        store
            .upsert_fact(
                "user",
                "mother_name",
                "Carol Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        store
            .upsert_fact(
                "user",
                "father",
                "Frank Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        store
            .upsert_fact(
                "user",
                "partner_name",
                "Alice Rivera",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // Pretend the search already matched the mother fact (id resolved below).
        let mother = store
            .get_facts(Some("user"))
            .await
            .unwrap()
            .into_iter()
            .find(|f| f.key == "mother_name")
            .unwrap();
        let initial: std::collections::HashSet<i64> = [mother.id].into_iter().collect();

        let out = store
            .assemble_neighborhood(&["Carol".to_string()], &initial)
            .await
            .unwrap();
        let values: Vec<String> = out.iter().map(|f| f.value.clone()).collect();
        assert!(
            values.iter().any(|v| v.contains("Frank")),
            "father pulled into cluster: {:?}",
            values
        );
    }

    /// Tests the *primary* person-resolution path: entity name → resolved Person
    /// with a `relationship` role → `owner_relationship = true` → cluster pulls
    /// in co-relations.
    ///
    /// Critically, `initial_ids` is EMPTY so the fallback "initial-match has a
    /// relationship key" trigger cannot fire.  Only the Person-record path can
    /// enable the owner cluster here.
    #[tokio::test]
    async fn assemble_neighborhood_resolves_person_relationship() {
        let (store, _db) = setup_store().await;

        // Seed a Person record with a relationship role.
        let person = crate::traits::Person {
            id: 0,
            name: "Carol Mendez".to_string(),
            aliases: vec![],
            relationship: Some("mother".to_string()),
            platform_ids: std::collections::HashMap::new(),
            notes: None,
            communication_style: None,
            language_preference: None,
            last_interaction_at: None,
            interaction_count: 0,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.upsert_person(&person).await.unwrap();

        // Seed flat facts that belong to the owner-relationship cluster.
        store
            .upsert_fact(
                "user",
                "father",
                "Frank Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        store
            .upsert_fact(
                "user",
                "partner_name",
                "Alice Rivera",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // Empty initial_ids — the fallback trigger (initial fact has relationship
        // key) CANNOT fire.  Only the Person-record resolution path can enable
        // the owner cluster.
        let initial: std::collections::HashSet<i64> = std::collections::HashSet::new();

        let out = store
            .assemble_neighborhood(&["Carol".to_string()], &initial)
            .await
            .unwrap();
        let values: Vec<String> = out.iter().map(|f| f.value.clone()).collect();
        assert!(
            values.iter().any(|v| v.contains("Frank")),
            "father should be pulled into cluster via person-resolution path: {:?}",
            values
        );
    }

    /// Word-boundary guard: entity "Ana" must NOT resolve the person "Banana Doe".
    #[tokio::test]
    async fn assemble_neighborhood_person_match_is_word_boundary() {
        let (store, _db) = setup_store().await;

        // "Banana Doe" — contains "ana" as a substring but NOT as a whole word.
        let person = crate::traits::Person {
            id: 0,
            name: "Banana Doe".to_string(),
            aliases: vec![],
            relationship: Some("friend".to_string()),
            platform_ids: std::collections::HashMap::new(),
            notes: None,
            communication_style: None,
            language_preference: None,
            last_interaction_at: None,
            interaction_count: 0,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.upsert_person(&person).await.unwrap();

        // A fact whose value would only appear if the person cluster were enabled.
        store
            .upsert_fact(
                "user",
                "best_friend",
                "Banana Doe",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        store
            .upsert_fact(
                "user",
                "hobby",
                "gardening",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // Search for "Ana" — should NOT match "Banana Doe" via word-boundary.
        let initial: std::collections::HashSet<i64> = std::collections::HashSet::new();
        let out = store
            .assemble_neighborhood(&["Ana".to_string()], &initial)
            .await
            .unwrap();
        // The owner cluster must NOT be triggered by false substring match.
        // "hobby" fact is unrelated; if "Ana" matched "Banana Doe", the friend's
        // relationship role would enable the cluster and pull in all relationship
        // facts.  Assert the result is empty (no initial hits, no cluster).
        assert!(
            out.is_empty(),
            "entity 'Ana' must not resolve person 'Banana Doe' via substring: {:?}",
            out
        );
    }
}
