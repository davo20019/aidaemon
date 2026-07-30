use super::*;

#[async_trait]
impl crate::traits::EpisodeStore for SqliteStateStore {
    async fn project_episode_memory(&self, episode_id: i64) -> anyhow::Result<()> {
        SqliteStateStore::project_episode_memory(self, episode_id).await
    }

    async fn get_relevant_episodes(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Episode>> {
        // Delegate to inherent method
        SqliteStateStore::get_relevant_episodes(self, query, limit).await
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
                        crate::session::stored_channel_matches_current(ep_ch, current_ch)
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
            for episode in &episodes {
                if let Err(error) = self.increment_episode_recall(episode.id).await {
                    tracing::debug!(%error, episode_id = episode.id, "Episode recall bump deferred");
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
                (Some(ep_ch), Some(current_ch)) => {
                    crate::session::stored_channel_matches_current(ep_ch, current_ch)
                }
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
                            .map(|vec| crate::memory::math::cosine_similarity(query_vec, &vec))
                    })
                    .or_else(|| {
                        Some(crate::memory::scoring::lexical_relevance(
                            query,
                            row.get::<String, _>("summary").as_str(),
                        ))
                    })
            } else {
                Some(crate::memory::scoring::lexical_relevance(
                    query,
                    row.get::<String, _>("summary").as_str(),
                ))
            };
            if let Some(similarity) = similarity {
                let episode = self.row_to_episode(&row)?;
                let score =
                    crate::memory::scoring::episode_search_score(similarity, episode.recall_count);
                if score > 0.25 {
                    scored.push((episode, score));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let episodes: Vec<Episode> = scored.into_iter().take(limit).map(|(e, _)| e).collect();
        for episode in &episodes {
            if let Err(error) = self.increment_episode_recall(episode.id).await {
                tracing::debug!(%error, episode_id = episode.id, "Episode recall bump deferred");
            }
        }
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
            for episode in &episodes {
                let _ = self.increment_episode_recall(episode.id).await;
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
                            .map(|vec| crate::memory::math::cosine_similarity(query_vec, &vec))
                    })
                    .or_else(|| {
                        Some(crate::memory::scoring::lexical_relevance(
                            query,
                            row.get::<String, _>("summary").as_str(),
                        ))
                    })
            } else {
                Some(crate::memory::scoring::lexical_relevance(
                    query,
                    row.get::<String, _>("summary").as_str(),
                ))
            };
            if let Some(similarity) = similarity {
                let episode = self.row_to_episode(&row)?;
                let score =
                    crate::memory::scoring::episode_search_score(similarity, episode.recall_count);
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
        for episode in &episodes {
            let _ = self.increment_episode_recall(episode.id).await;
        }
        Ok(episodes)
    }
}
