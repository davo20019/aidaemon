use super::*;
use crate::traits::DialogueState;

#[async_trait]
impl crate::traits::DialogueStateStore for SqliteStateStore {
    async fn get_dialogue_state(&self, session_id: &str) -> anyhow::Result<Option<DialogueState>> {
        let row = sqlx::query("SELECT state_json FROM dialogue_states WHERE session_id = ?")
            .bind(session_id)
            .fetch_optional(&self.pool)
            .await?;

        row.map(|r| {
            let raw: String = r.get("state_json");
            serde_json::from_str::<DialogueState>(&raw).map_err(anyhow::Error::from)
        })
        .transpose()
    }

    async fn upsert_dialogue_state(&self, state: &DialogueState) -> anyhow::Result<()> {
        let state_json = serde_json::to_string(state)?;
        let open_request_status = state
            .open_request
            .as_ref()
            .map(|request| serde_json::to_string(&request.status))
            .transpose()?;
        let awaiting_user_reply = state
            .open_question
            .as_ref()
            .is_some_and(|question| question.awaiting_user_reply);
        let active_task_id = state.active_task.as_ref().map(|task| task.task_id.clone());

        sqlx::query(
            "INSERT INTO dialogue_states (
                session_id, state_json, revision, active_task_id,
                open_request_status, awaiting_user_reply, updated_at
             ) VALUES (?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(session_id) DO UPDATE SET
                state_json = excluded.state_json,
                revision = excluded.revision,
                active_task_id = excluded.active_task_id,
                open_request_status = excluded.open_request_status,
                awaiting_user_reply = excluded.awaiting_user_reply,
                updated_at = excluded.updated_at",
        )
        .bind(&state.session_id)
        .bind(state_json)
        .bind(state.revision)
        .bind(active_task_id)
        .bind(open_request_status)
        .bind(if awaiting_user_reply { 1_i64 } else { 0_i64 })
        .bind(state.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn delete_dialogue_state(&self, session_id: &str) -> anyhow::Result<()> {
        sqlx::query("DELETE FROM dialogue_states WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
