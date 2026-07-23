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
        // For channel-scoped episode retrieval, filter episodes by channel_id
        // Episodes without channel_id (legacy) are accessible everywhere
        let rows = sqlx::query(
            "SELECT id, session_id, summary, topics, emotional_tone, outcome, importance, recall_count, last_recalled_at, message_count, start_time, end_time, created_at, channel_id, embedding
             FROM episodes ORDER BY created_at DESC LIMIT 500"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() || query.trim().is_empty() {
            return Ok(vec![]);
        }

        let query_vec = match self.embedding_service.embed(query.to_string()).await {
            Ok(v) => v,
            Err(_) => return Ok(vec![]),
        };
        let indexed_scores = self
            .episode_embedding_scores(&query_vec, rows.len())
            .await
            .unwrap_or_default();

        let mut scored: Vec<(Episode, f32)> = Vec::new();
        for row in rows {
            // Filter by channel: include episodes from same channel or legacy (no channel_id)
            let ep_channel_id: Option<String> = row.try_get("channel_id").unwrap_or(None);
            let include = match (&ep_channel_id, channel_id) {
                (None, _) => true, // Legacy episodes: include
                (Some(ep_ch), Some(current_ch)) => {
                    crate::session::stored_channel_matches_current(ep_ch, current_ch)
                }
                (Some(_), None) => false, // Has channel but no current: skip
            };
            if !include {
                continue;
            }

            let episode_id: i64 = row.get("id");
            let similarity = indexed_scores.get(&episode_id).copied().or_else(|| {
                row.get::<Option<Vec<u8>>, _>("embedding")
                    .and_then(|blob| decode_embedding(&blob).ok())
                    .map(|vec| crate::memory::math::cosine_similarity(&query_vec, &vec))
            });
            if let Some(similarity) = similarity {
                let episode = self.row_to_episode(&row)?;
                let score = crate::memory::scoring::memory_score(
                    similarity,
                    episode.created_at,
                    episode.recall_count,
                    episode.last_recalled_at,
                );
                if score > 0.3 {
                    scored.push((episode, score));
                }
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let episodes: Vec<Episode> = scored.into_iter().take(limit).map(|(e, _)| e).collect();
        Ok(episodes)
    }
}
