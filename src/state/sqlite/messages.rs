use super::*;

#[async_trait]
impl crate::traits::MessageStore for SqliteStateStore {
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        // Canonical persistence is event-sourced (events table). Keep only an
        // in-memory hot window here for low-latency context assembly.
        {
            let mut wm = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                self.working_memory.write(),
            )
            .await
            .map_err(|_| anyhow::anyhow!("append_message: working_memory write lock timed out"))?;
            let deque = wm
                .entry(msg.session_id.clone())
                .or_insert_with(VecDeque::new);
            deque.push_back(msg.clone());

            // Evict old messages but ALWAYS preserve the first user message (anchor)
            // This is critical for Gemini which requires tool_calls to follow user/tool messages
            let mut evicted = 0;
            while deque.len() > self.cap {
                // Find the first user message index
                let anchor_idx = deque.iter().position(|m| m.role == "user");

                if anchor_idx == Some(0) && deque.len() > 1 {
                    // Anchor is at front - evict the second message instead
                    deque.remove(1);
                } else {
                    // Safe to evict from front
                    deque.pop_front();
                }
                evicted += 1;
            }

            tracing::debug!(
                session_id = %msg.session_id,
                role = %msg.role,
                msg_id = %msg.id,
                deque_len = deque.len(),
                cap = self.cap,
                evicted,
                "append_message: added to working memory"
            );
        }

        Ok(())
    }

    async fn get_history(&self, session_id: &str, limit: usize) -> anyhow::Result<Vec<Message>> {
        // Check working memory first
        {
            let wm = match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                self.working_memory.read(),
            )
            .await
            {
                Ok(guard) => guard,
                Err(_) => {
                    tracing::warn!(
                        session_id,
                        "get_history: working_memory read lock timed out, falling back to DB hydrate"
                    );
                    // Fall through to DB hydrate path.
                    return self.hydrate(session_id).await.map(|deque| {
                        let msgs: Vec<_> = deque.iter().cloned().collect();
                        crate::conversation::truncate_with_anchor(msgs, limit)
                    });
                }
            };
            tracing::debug!(
                session_id,
                wm_sessions = wm.len(),
                has_session = wm.contains_key(session_id),
                "get_history: checking working memory"
            );
            if let Some(deque) = wm.get(session_id) {
                let roles: Vec<&str> = deque.iter().map(|m| m.role.as_str()).collect();
                tracing::debug!(
                    session_id,
                    deque_len = deque.len(),
                    roles = ?roles,
                    "get_history: found session in working memory"
                );
                if !deque.is_empty() {
                    let msgs: Vec<_> = deque.iter().cloned().collect();
                    let before_len = msgs.len();
                    let result = crate::conversation::truncate_with_anchor(msgs, limit);
                    tracing::debug!(
                        session_id,
                        before_truncate = before_len,
                        after_truncate = result.len(),
                        "get_history: returning from working memory"
                    );
                    return Ok(result);
                }
            }
        }

        // Cold start: hydrate from DB
        tracing::debug!(session_id, "get_history: cold start, hydrating from DB");
        let deque = self.hydrate(session_id).await?;
        let msgs: Vec<_> = deque.iter().cloned().collect();
        let result = crate::conversation::truncate_with_anchor(msgs, limit);
        tracing::debug!(
            session_id,
            hydrated_count = deque.len(),
            result_count = result.len(),
            "get_history: hydrated from DB"
        );

        // Cache in working memory
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.working_memory.write(),
        )
        .await
        {
            Ok(mut wm) => {
                wm.insert(session_id.to_string(), deque);
            }
            Err(_) => {
                tracing::warn!(
                    session_id,
                    "get_history: working_memory write lock timed out, skipping cache insert"
                );
            }
        }

        Ok(result)
    }

    async fn get_context(
        &self,
        session_id: &str,
        _query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        // Canonical context retrieval is event-backed. The in-memory working
        // window is hydrated from events on cold start by get_history().
        self.get_history(session_id, limit).await
    }

    async fn clear_session(&self, session_id: &str) -> anyhow::Result<()> {
        // Clear working memory
        {
            match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                self.working_memory.write(),
            )
            .await
            {
                Ok(mut wm) => {
                    wm.remove(session_id);
                }
                Err(_) => {
                    tracing::warn!(
                        session_id,
                        "clear_session: working_memory write lock timed out"
                    );
                }
            }
        }

        // Remove every active-database projection that can contain verbatim
        // conversation text. Keep the logical deletion atomic: a damaged FTS
        // index or foreign-key failure rolls everything back instead of
        // reporting a successful partial wipe.
        let mut tx = self.pool.begin().await?;
        super::history_search::remove_session_projection_in_tx(&mut tx, session_id).await?;
        sqlx::query(
            "UPDATE facts SET source_excerpt = NULL
             WHERE id IN (
                 SELECT source_fact_id FROM memory_claims
                 WHERE source_fact_id IS NOT NULL
                   AND (
                       source_span_id IN (
                           SELECT id FROM memory_spans WHERE session_id = ?
                       )
                       OR source_event_id IN (
                           SELECT id FROM events WHERE session_id = ?
                       )
                   )
             )",
        )
        .bind(session_id)
        .bind(session_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE memory_claims
             SET source_span_id = NULL, source_event_id = NULL,
                 provenance = CASE
                     WHEN provenance = '' THEN 'source_session_wiped'
                     ELSE provenance || ';source_session_wiped'
                 END
             WHERE source_span_id IN (
                       SELECT id FROM memory_spans WHERE session_id = ?
                   )
                OR source_event_id IN (
                       SELECT id FROM events WHERE session_id = ?
                   )",
        )
        .bind(session_id)
        .bind(session_id)
        .execute(&mut *tx)
        .await?;
        sqlx::query(
            "DELETE FROM memory_embeddings
             WHERE owner_type = 'span'
               AND owner_id IN (
                   SELECT CAST(id AS TEXT) FROM memory_spans WHERE session_id = ?
               )",
        )
        .bind(session_id)
        .execute(&mut *tx)
        .await?;
        // Durable goals survive the wipe, but cannot retain a foreign-key
        // pointer back to an erased episode.
        sqlx::query(
            "UPDATE goals SET source_episode_id = NULL
             WHERE source_episode_id IN (
                 SELECT id FROM episodes WHERE session_id = ?
             )",
        )
        .bind(session_id)
        .execute(&mut *tx)
        .await?;

        for table in [
            "memory_spans",
            "episodes",
            "events",
            "conversation_summaries",
            "session_context_boundaries",
            "dialogue_states",
            "task_plans",
            "notification_queue",
            "cli_agent_invocations",
            "session_channels",
            "pending_oauth_flows",
        ] {
            let table_exists: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            )
            .bind(table)
            .fetch_one(&mut *tx)
            .await?;
            if table_exists == 0 {
                continue;
            }
            let query = format!("DELETE FROM {table} WHERE session_id = ?");
            sqlx::query(&query)
                .bind(session_id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        super::history_search::checkpoint_after_wipe(&self.pool).await?;
        Ok(())
    }

    async fn clear_session_context(&self, session_id: &str) -> anyhow::Result<()> {
        // Clear the in-memory hot window; it rehydrates from events filtered by
        // the boundary we set below, so the fresh context excludes prior turns.
        {
            match tokio::time::timeout(
                std::time::Duration::from_secs(2),
                self.working_memory.write(),
            )
            .await
            {
                Ok(mut wm) => {
                    wm.remove(session_id);
                }
                Err(_) => {
                    tracing::warn!(
                        session_id,
                        "clear_session_context: working_memory write lock timed out"
                    );
                }
            }
        }

        // Record (or advance) the durable boundary at the current max event id.
        // Context retrieval hides everything with id <= cleared_after_id; NOTHING
        // is deleted — the events remain for the memory pipeline and audit.
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO session_context_boundaries (session_id, cleared_after_id, cleared_at)
             VALUES (?1, COALESCE((SELECT MAX(id) FROM events WHERE session_id = ?1), 0), ?2)
             ON CONFLICT(session_id) DO UPDATE SET
                cleared_after_id = COALESCE((SELECT MAX(id) FROM events WHERE session_id = ?1), 0),
                cleared_at = ?2",
        )
        .bind(session_id)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        // Drop the derived conversation summary: it summarizes now-hidden
        // messages and is regenerable — not source-of-truth history.
        if let Err(e) = sqlx::query("DELETE FROM conversation_summaries WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await
        {
            if !e
                .to_string()
                .contains("no such table: conversation_summaries")
            {
                return Err(e.into());
            }
        }
        Ok(())
    }

    async fn advance_session_context_boundary(
        &self,
        session_id: &str,
        cleared_after_event_id: i64,
        retained_turn_id: &str,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            cleared_after_event_id >= 0,
            "context boundary event id cannot be negative"
        );
        anyhow::ensure!(
            !retained_turn_id.trim().is_empty(),
            "retained context turn id cannot be empty"
        );

        // Keep the current task's already-appended messages while removing
        // older hot-window entries. Without this, the durable SQL boundary
        // would be bypassed until the process restarted or the cache evicted.
        match tokio::time::timeout(
            std::time::Duration::from_secs(2),
            self.working_memory.write(),
        )
        .await
        {
            Ok(mut wm) => {
                if let Some(messages) = wm.get_mut(session_id) {
                    messages.retain(|message| message.turn_id.as_deref() == Some(retained_turn_id));
                }
            }
            Err(_) => {
                anyhow::bail!(
                    "advance_session_context_boundary: working_memory write lock timed out"
                );
            }
        }

        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO session_context_boundaries (session_id, cleared_after_id, cleared_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id) DO UPDATE SET
                cleared_after_id = MAX(session_context_boundaries.cleared_after_id, ?2),
                cleared_at = CASE
                    WHEN ?2 > session_context_boundaries.cleared_after_id THEN ?3
                    ELSE session_context_boundaries.cleared_at
                END",
        )
        .bind(session_id)
        .bind(cleared_after_event_id)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        // A cumulative summary describes turns that are now outside implicit
        // context. Exact older history remains available through its explicit
        // event-backed retrieval path.
        sqlx::query("DELETE FROM conversation_summaries WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}
