use super::*;

const FACT_SEMANTIC_MIN_SCORE: f32 = 0.3;
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

fn fact_freshness_boost(now: DateTime<Utc>, updated_at: DateTime<Utc>) -> f32 {
    let age_hours = (now - updated_at).num_hours().max(0) as f32;
    FACT_FRESHNESS_MAX_BOOST * (1.0 - (age_hours / FACT_FRESHNESS_DECAY_HOURS).min(1.0))
}

fn is_stopword(token: &str) -> bool {
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
    async fn upsert_fact(
        &self,
        category: &str,
        key: &str,
        value: &str,
        source: &str,
        channel_id: Option<&str>,
        privacy: FactPrivacy,
    ) -> anyhow::Result<()> {
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

        if let Some((old_id, _old_key, old_value)) = existing {
            // If the value is different, mark old as superseded and insert new
            if old_value != value {
                sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
                    .bind(&now)
                    .bind(old_id)
                    .execute(&self.pool)
                    .await?;

                // Insert new fact with embedding — ignore duplicate entry errors (code 2067)
                // that can occur due to active-unique constraint race conditions.
                let insert_result = sqlx::query(
                    "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy, embedding)
                     VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
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
                                 embedding = COALESCE(?, embedding)
                             WHERE category = ? AND key = ? AND superseded_at IS NULL",
                        )
                        .bind(value)
                        .bind(source)
                        .bind(&now)
                        .bind(channel_id)
                        .bind(&privacy_str)
                        .bind(&embedding_blob)
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
                                     embedding = COALESCE(?, embedding)
                                 WHERE id = ?",
                            )
                            .bind(value)
                            .bind(source)
                            .bind(&now)
                            .bind(channel_id)
                            .bind(&privacy_str)
                            .bind(&embedding_blob)
                            .bind(old_id)
                            .execute(&self.pool)
                            .await?;
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            } else {
                // Same value - update timestamp/source and backfill embedding if missing
                sqlx::query(
                    "UPDATE facts SET source = ?, updated_at = ?, embedding = COALESCE(embedding, ?) WHERE id = ?",
                )
                .bind(source)
                .bind(&now)
                .bind(&embedding_blob)
                .bind(old_id)
                .execute(&self.pool)
                .await?;
            }
        } else {
            // No existing fact - insert new with embedding
            // Ignore duplicate entry errors (code 2067) from concurrent inserts
            let insert_result = sqlx::query(
                "INSERT INTO facts (category, key, value, source, created_at, updated_at, recall_count, channel_id, privacy, embedding)
                 VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
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
                             embedding = COALESCE(?, embedding)
                         WHERE category = ? AND key = ? AND superseded_at IS NULL",
                    )
                    .bind(value)
                    .bind(source)
                    .bind(&now)
                    .bind(channel_id)
                    .bind(&privacy_str)
                    .bind(&embedding_blob)
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
                                     embedding = COALESCE(?, embedding)
                                 WHERE id = ?",
                            )
                            .bind(value)
                            .bind(source)
                            .bind(&now)
                            .bind(channel_id)
                            .bind(&privacy_str)
                            .bind(&embedding_blob)
                            .bind(id)
                            .execute(&self.pool)
                            .await?;
                        }
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
        // Only return current (non-superseded) facts
        let rows = if let Some(cat) = category {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy FROM facts WHERE category = ? AND superseded_at IS NULL ORDER BY updated_at DESC")
                .bind(cat)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query("SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC")
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
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding
             FROM facts WHERE superseded_at IS NULL ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        let all_facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();

        if all_facts.is_empty() || query.trim().is_empty() {
            let mut facts = all_facts;
            facts.truncate(max);
            bump_fact_recall(&self.pool, &facts).await;
            return Ok(facts);
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
            return Ok(facts);
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
                return Ok(facts);
            }
        };

        let now = Utc::now();
        let query_lower = query.to_lowercase();
        let tokens = query_tokens(&query_lower);

        // Score facts using stored embeddings, with a small recency boost for sorting.
        // IMPORTANT: the recency boost must not change which facts pass the semantic threshold.
        let mut candidates: Vec<(usize, f32, f32, bool)> = Vec::with_capacity(rows.len());
        let mut unembedded: Vec<usize> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let fact = &all_facts[i];
            let freshness = fact_freshness_boost(now, fact.updated_at);

            let embedding: Option<Vec<u8>> = row.get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let semantic = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    if semantic > FACT_SEMANTIC_MIN_SCORE {
                        candidates.push((i, semantic, semantic + freshness, true));
                        continue;
                    }
                    // Below semantic threshold — try lexical as fallback for keyword matches
                    let lexical = lexical_fallback_score(&query_lower, &tokens, fact);
                    let best = semantic.max(lexical);
                    let is_semantic = best == semantic;
                    candidates.push((i, best, best + freshness, is_semantic));
                    continue;
                }
            }

            // Missing/invalid embedding: fall back to cheap lexical relevance so
            // freshly saved facts can still be retrieved during embedding backfill.
            let lexical = lexical_fallback_score(&query_lower, &tokens, fact);
            candidates.push((i, lexical, lexical + freshness, false));
            unembedded.push(i);
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

        // Preserve prior behavior: include unembedded facts (during backfill) if
        // we still have space, even if lexical relevance was low.
        for i in unembedded {
            if relevant.len() >= max {
                break;
            }
            let fact = all_facts[i].clone();
            if seen_ids.insert(fact.id) {
                relevant.push(fact);
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
        Ok(relevant)
    }

    async fn get_relevant_facts_for_channel(
        &self,
        query: &str,
        max: usize,
        channel_id: Option<&str>,
        visibility: ChannelVisibility,
    ) -> anyhow::Result<Vec<Fact>> {
        // In DM/Internal contexts, use semantic relevance search so that only
        // facts related to the current query are injected into the prompt.
        // Previously this called get_facts(None) which returned ALL facts
        // without filtering, causing unrelated facts to bleed into context.
        if matches!(
            visibility,
            ChannelVisibility::Private | ChannelVisibility::Internal
        ) {
            return self.get_relevant_facts(query, max).await;
        }

        // PublicExternal: do NOT inject any stored facts (treat as untrusted).
        if matches!(visibility, ChannelVisibility::PublicExternal) {
            return Ok(vec![]);
        }

        // Public/PrivateGroup: global + same-channel facts (no private, no other-channel)
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding
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
                    (Some(current), Some(fact_ch)) => current == fact_ch,
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

        let mut candidates: Vec<(usize, f32, f32, bool)> = Vec::with_capacity(filtered.len());
        let mut unembedded: Vec<usize> = Vec::new();
        for (fi, &ri) in filtered_indices.iter().enumerate() {
            let fact = &filtered[fi];
            let freshness = fact_freshness_boost(now, fact.updated_at);

            let embedding: Option<Vec<u8>> = rows[ri].get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let semantic = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    if semantic > FACT_SEMANTIC_MIN_SCORE {
                        candidates.push((fi, semantic, semantic + freshness, true));
                        continue;
                    }
                    // Below semantic threshold — try lexical as fallback for keyword matches
                    let lexical = lexical_fallback_score(&query_lower, &tokens, fact);
                    let best = semantic.max(lexical);
                    let is_semantic = best == semantic;
                    candidates.push((fi, best, best + freshness, is_semantic));
                    continue;
                }
            }

            let lexical = lexical_fallback_score(&query_lower, &tokens, fact);
            candidates.push((fi, lexical, lexical + freshness, false));
            unembedded.push(fi);
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

        for fi in unembedded {
            if relevant.len() >= max {
                break;
            }
            let fact = filtered[fi].clone();
            if seen_ids.insert(fact.id) {
                relevant.push(fact);
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
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy, embedding
             FROM facts
             WHERE superseded_at IS NULL
               AND privacy = 'channel'
               AND channel_id IS NOT NULL
               AND channel_id != ?
             ORDER BY updated_at DESC",
        )
        .bind(current_channel_id)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            return Ok(vec![]);
        }

        let facts: Vec<Fact> = rows.iter().map(Self::row_to_fact).collect();

        // Apply semantic filtering using stored embeddings
        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        };

        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let embedding: Option<Vec<u8>> = row.get("embedding");
            if let Some(blob) = embedding {
                if let Ok(vec) = decode_embedding(&blob) {
                    let score = crate::memory::math::cosine_similarity(&query_vec, &vec);
                    scored.push((i, score));
                }
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
        Ok(())
    }

    async fn delete_fact(&self, fact_id: i64) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query("UPDATE facts SET superseded_at = ? WHERE id = ?")
            .bind(&now)
            .bind(fact_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_all_facts_with_provenance(&self) -> anyhow::Result<Vec<Fact>> {
        let rows = sqlx::query(
            "SELECT id, category, key, value, source, created_at, updated_at, superseded_at, recall_count, last_recalled_at, channel_id, privacy
             FROM facts WHERE superseded_at IS NULL ORDER BY category, key"
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(Self::row_to_fact).collect())
    }
}
