use super::*;

#[async_trait]
impl EpisodeStore for SqliteMemory {
    async fn last_episode_bound(&self, session_id: &str) -> anyhow::Result<Option<EpisodeBound>> {
        let row = sqlx::query(
            "SELECT end_event_id, end_time FROM episodes
             WHERE session_id = ?
             ORDER BY COALESCE(end_event_id, 0) DESC, end_time DESC LIMIT 1",
        )
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| EpisodeBound {
            end_event_id: row.try_get::<Option<i64>, _>("end_event_id").ok().flatten(),
            end_time: row.try_get::<String, _>("end_time").ok(),
        }))
    }

    async fn latest_episode_ends(&self) -> anyhow::Result<Vec<(String, String)>> {
        let rows: Vec<(String, Option<String>)> =
            sqlx::query_as("SELECT session_id, MAX(end_time) FROM episodes GROUP BY session_id")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .into_iter()
            .filter_map(|(session_id, end)| end.map(|end| (session_id, end)))
            .collect())
    }

    /// Insert the episode with the event span it covers, then embed its
    /// summary and project it into the memory index. Multiple episodes per
    /// session are allowed (mid-session ones for long conversations, then a
    /// final one when the session goes idle).
    async fn record_session_episode(
        &self,
        episode: &crate::episodes::SessionEpisode,
    ) -> anyhow::Result<i64> {
        let (start_event_id, end_event_id) = self
            .message_event_bounds(&episode.session_id, episode.start_time, episode.end_time)
            .await?;
        let topics_json = serde_json::to_string(&episode.topics).ok();
        let result = sqlx::query(
            "INSERT INTO episodes
                (session_id, summary, topics, emotional_tone, outcome, importance,
                 recall_count, message_count, start_time, end_time, created_at,
                 channel_id, start_event_id, end_event_id)
             VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&episode.session_id)
        .bind(&episode.summary)
        .bind(&topics_json)
        .bind(&episode.emotional_tone)
        .bind(&episode.outcome)
        .bind(episode.importance)
        .bind(episode.message_count as i32)
        .bind(episode.start_time.to_rfc3339())
        .bind(episode.end_time.to_rfc3339())
        .bind(chrono::Utc::now().to_rfc3339())
        .bind(&episode.channel_id)
        .bind(start_event_id)
        .bind(end_event_id)
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            tracing::info!(session_id = %episode.session_id, "Episode insert returned 0 rows affected");
            let existing: Option<(i64,)> = sqlx::query_as(
                "SELECT id FROM episodes WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            )
            .bind(&episode.session_id)
            .fetch_optional(&self.pool)
            .await?;
            return Ok(existing.map(|(id,)| id).unwrap_or(0));
        }
        let episode_id = result.last_insert_rowid();
        if let Ok(embedding) = self.embedding_service.embed(episode.summary.clone()).await {
            sqlx::query("UPDATE episodes SET embedding = ? WHERE id = ?")
                .bind(encode_embedding(&embedding))
                .bind(episode_id)
                .execute(&self.pool)
                .await?;
            // Keep the canonical span and the embedding index in sync with
            // the legacy episode column.
            let _ = self.project_episode_memory(episode_id).await;
        }
        Ok(episode_id)
    }

    async fn project_episode_memory(&self, episode_id: i64) -> anyhow::Result<()> {
        SqliteMemory::project_episode_memory(self, episode_id).await
    }

    async fn get_relevant_episodes(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Episode>> {
        // Delegate to inherent method
        SqliteMemory::get_relevant_episodes(self, query, limit).await
    }

    async fn get_relevant_episodes_for_channel(
        &self,
        query: &str,
        limit: usize,
        channel_id: Option<&str>,
    ) -> anyhow::Result<Vec<Episode>> {
        // For channel-scoped retrieval, unknown legacy provenance is not
        // authorization to expose an episode in every channel.
        let rows = sqlx::query(
            "SELECT id, session_id, summary, topics, emotional_tone, outcome, importance, recall_count, last_recalled_at, message_count, start_time, end_time, created_at, channel_id, embedding
             FROM episodes ORDER BY created_at DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(vec![]);
        }
        if query.trim().is_empty() {
            let mut episodes = Vec::new();
            for row in rows {
                let ep_channel_id: Option<String> = row.try_get("channel_id").unwrap_or(None);
                let include = match (&ep_channel_id, channel_id) {
                    (None, None) => true,
                    (None, Some(_)) => false,
                    (Some(ep_ch), Some(current_ch)) => {
                        self.channels.same_channel(ep_ch, current_ch)
                    }
                    (Some(_), None) => false,
                };
                if include {
                    episodes.push(self.row_to_episode(&row)?);
                }
                if episodes.len() >= limit {
                    break;
                }
            }
            return Ok(episodes);
        }

        let query_vec = self.embedding_service.embed(query.to_string()).await.ok();
        let indexed_scores = if let Some(query_vec) = query_vec.as_ref() {
            self.episode_embedding_scores(query_vec, rows.len())
                .await
                .unwrap_or_default()
        } else {
            tracing::warn!(
                "Channel episode embedding unavailable; using explicit lexical degraded mode"
            );
            std::collections::HashMap::new()
        };

        let mut scored: Vec<(Episode, f32)> = Vec::new();
        for row in rows {
            // Filter by channel: include episodes from same channel or legacy (no channel_id)
            let ep_channel_id: Option<String> = row.try_get("channel_id").unwrap_or(None);
            let include = match (&ep_channel_id, channel_id) {
                (None, None) => true,
                (None, Some(_)) => false,
                (Some(ep_ch), Some(current_ch)) => self.channels.same_channel(ep_ch, current_ch),
                (Some(_), None) => false, // Has channel but no current: skip
            };
            if !include {
                continue;
            }

            let episode_id: i64 = row.get("id");
            let similarity = if let Some(query_vec) = query_vec.as_ref() {
                indexed_scores
                    .get(&episode_id)
                    .copied()
                    .or_else(|| {
                        row.get::<Option<Vec<u8>>, _>("embedding")
                            .and_then(|blob| decode_embedding(&blob).ok())
                            .map(|vec| cosine_similarity(query_vec, &vec))
                    })
                    .or_else(|| {
                        Some(crate::word_overlap(
                            query,
                            row.get::<String, _>("summary").as_str(),
                        ))
                    })
            } else {
                Some(crate::word_overlap(
                    query,
                    row.get::<String, _>("summary").as_str(),
                ))
            };
            if let Some(similarity) = similarity {
                let episode = self.row_to_episode(&row)?;
                let score = episode_search_score(similarity, episode.recall_count);
                if score > 0.25 {
                    scored.push((episode, score));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let episodes: Vec<Episode> = scored.into_iter().take(limit).map(|(e, _)| e).collect();
        Ok(episodes)
    }

    async fn get_relevant_episodes_for_session(
        &self,
        query: &str,
        limit: usize,
        session_id: &str,
    ) -> anyhow::Result<Vec<Episode>> {
        let rows = sqlx::query(
            "SELECT id, session_id, summary, topics, emotional_tone, outcome,
                    importance, recall_count, last_recalled_at, message_count,
                    start_time, end_time, created_at, channel_id, embedding
             FROM episodes WHERE session_id=? ORDER BY created_at DESC",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            let mut episodes = Vec::new();
            for row in rows.into_iter().take(limit) {
                episodes.push(self.row_to_episode(&row)?);
            }
            return Ok(episodes);
        }

        let query_vec = self.embedding_service.embed(query.to_string()).await.ok();
        let indexed_scores = if let Some(query_vec) = query_vec.as_ref() {
            // The vector index is global. Asking it for only this session's row
            // count could return N higher-scoring rows from other sessions and
            // falsely make the authorized session look empty.
            let global_episode_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM episodes")
                .fetch_one(&self.pool)
                .await?;
            self.episode_embedding_scores(query_vec, global_episode_count.max(0) as usize)
                .await
                .unwrap_or_default()
        } else {
            std::collections::HashMap::new()
        };
        let mut scored = Vec::new();
        for row in rows {
            let episode_id: i64 = row.get("id");
            let similarity = if let Some(query_vec) = query_vec.as_ref() {
                indexed_scores
                    .get(&episode_id)
                    .copied()
                    .or_else(|| {
                        row.get::<Option<Vec<u8>>, _>("embedding")
                            .and_then(|blob| decode_embedding(&blob).ok())
                            .map(|vec| cosine_similarity(query_vec, &vec))
                    })
                    .or_else(|| {
                        Some(crate::word_overlap(
                            query,
                            row.get::<String, _>("summary").as_str(),
                        ))
                    })
            } else {
                Some(crate::word_overlap(
                    query,
                    row.get::<String, _>("summary").as_str(),
                ))
            };
            if let Some(similarity) = similarity {
                let episode = self.row_to_episode(&row)?;
                let score = episode_search_score(similarity, episode.recall_count);
                if score > 0.25 {
                    scored.push((episode, score));
                }
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let episodes: Vec<_> = scored
            .into_iter()
            .take(limit)
            .map(|(episode, _)| episode)
            .collect();
        Ok(episodes)
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;

    async fn memory() -> SqliteMemory {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        migrate(&pool).await.unwrap();
        SqliteMemory::new(pool, Arc::new(crate::HashEmbedder::new(384)))
    }

    async fn episode(memory: &SqliteMemory, session: &str, end: &str, end_event: Option<i64>) {
        sqlx::query(
            "INSERT INTO episodes (session_id, summary, importance, message_count,
                                   start_time, end_time, created_at, end_event_id)
             VALUES (?1, 'summary', 0.5, 5, ?2, ?2, ?2, ?3)",
        )
        .bind(session)
        .bind(end)
        .bind(end_event)
        .execute(&memory.pool)
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_session_is_bounded_by_its_last_episode() {
        let memory = memory().await;
        assert_eq!(memory.last_episode_bound("s1").await.unwrap(), None);
        episode(&memory, "s1", "2026-01-02T00:00:00+00:00", None).await;
        assert_eq!(
            memory.last_episode_bound("s1").await.unwrap(),
            Some(EpisodeBound {
                end_event_id: None,
                end_time: Some("2026-01-02T00:00:00+00:00".into()),
            })
        );
        // A recorded event bound outranks a later end time without one.
        episode(&memory, "s1", "2026-01-01T00:00:00+00:00", Some(7)).await;
        episode(&memory, "s1", "2026-01-03T00:00:00+00:00", None).await;
        let bound = memory.last_episode_bound("s1").await.unwrap().unwrap();
        assert_eq!(bound.end_event_id, Some(7));
    }

    #[tokio::test]
    async fn latest_ends_take_the_newest_per_session() {
        let memory = memory().await;
        episode(&memory, "s1", "2026-01-01T00:00:00+00:00", None).await;
        episode(&memory, "s1", "2026-01-05T00:00:00+00:00", None).await;
        episode(&memory, "s\"2", "2026-01-02T00:00:00+00:00", None).await;
        let mut ends = memory.latest_episode_ends().await.unwrap();
        ends.sort();
        assert_eq!(
            ends,
            [
                ("s\"2".to_string(), "2026-01-02T00:00:00+00:00".to_string()),
                ("s1".to_string(), "2026-01-05T00:00:00+00:00".to_string()),
            ]
        );
    }
}
