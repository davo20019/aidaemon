use super::*;

#[async_trait]
impl crate::traits::MessageStore for SqliteStateStore {
    async fn append_message(&self, msg: &Message) -> anyhow::Result<()> {
        // Canonical persistence is event-sourced (events table). Keep only an
        // in-memory hot window here for low-latency context assembly.
        {
            let mut wm = self.working_memory.write().await;
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
            let wm = self.working_memory.read().await;
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
        let mut wm = self.working_memory.write().await;
        wm.insert(session_id.to_string(), deque);

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
            let mut wm = self.working_memory.write().await;
            wm.remove(session_id);
        }

        // Delete session rows across canonical tables.
        // Some test DBs may not have all tables yet; treat missing tables as best-effort.
        for table in ["events", "conversation_summaries"] {
            let query = format!("DELETE FROM {table} WHERE session_id = ?");
            if let Err(e) = sqlx::query(&query)
                .bind(session_id)
                .execute(&self.pool)
                .await
            {
                let missing_table = format!("no such table: {table}");
                if !e.to_string().contains(&missing_table) {
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }
}
