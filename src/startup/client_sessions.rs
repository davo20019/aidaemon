//! What the owner sets on a conversation from an app: its name, over the
//! one taken from what was said, and whether it is archived. Kept on its
//! `session_channels` row.

use std::collections::HashMap;

/// Each conversation's owner-set name and archived flag, by session id.
pub(crate) async fn marks(pool: &sqlx::SqlitePool) -> HashMap<String, (Option<String>, bool)> {
    sqlx::query_as::<_, (String, Option<String>, Option<i64>)>(
        "SELECT session_id, title, archived FROM session_channels \
         WHERE title IS NOT NULL OR archived IS NOT NULL",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|(id, title, archived)| {
        (
            id,
            (
                title.filter(|t| !t.trim().is_empty()),
                archived.unwrap_or(0) != 0,
            ),
        )
    })
    .collect()
}

/// The coding agent each conversation set to one answers with, by
/// conversation id, read in one go for the lists (protocol 1.52).
pub(crate) async fn agents(pool: &sqlx::SqlitePool) -> HashMap<String, String> {
    use aidaemon_runtime::session_model::{KEY_PREFIX, SessionModelSetting};
    // The key prefix is one the daemon writes (`session_model::key`).
    sqlx::query_as::<_, (String, String)>(
        "SELECT key, value FROM settings WHERE substr(key, 1, ?) = ?",
    )
    .bind(i64::try_from(KEY_PREFIX.len()).unwrap_or(i64::MAX))
    .bind(KEY_PREFIX)
    .fetch_all(pool)
    .await
    .unwrap_or_default()
    .into_iter()
    .filter_map(|(key, value)| {
        let id = key.strip_prefix(KEY_PREFIX)?.to_string();
        let agent = serde_json::from_str::<SessionModelSetting>(&value)
            .ok()?
            .agent?;
        Some((id, agent))
    })
    .collect()
}

/// Name conversation `id` (`None` or blank: back to the name taken from
/// what was said). `Err`: there is no such conversation.
pub(crate) async fn rename(
    pool: &sqlx::SqlitePool,
    id: &str,
    title: Option<&str>,
) -> Result<(), String> {
    let title = title
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.chars().take(120).collect::<String>());
    let done = sqlx::query("UPDATE session_channels SET title = ? WHERE session_id = ?")
        .bind(title)
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| format!("Could not rename it: {e}"))?;
    if done.rows_affected() == 0 {
        return Err("There is no such conversation".to_string());
    }
    Ok(())
}

/// Archive conversation `id`, or bring it back. `Err`: there is no such
/// conversation.
pub(crate) async fn archive(
    pool: &sqlx::SqlitePool,
    id: &str,
    archived: bool,
) -> Result<(), String> {
    let done = sqlx::query("UPDATE session_channels SET archived = ? WHERE session_id = ?")
        .bind(i64::from(archived))
        .bind(id)
        .execute(pool)
        .await
        .map_err(|e| format!("Could not archive it: {e}"))?;
    if done.rows_affected() == 0 {
        return Err("There is no such conversation".to_string());
    }
    Ok(())
}

/// The app conversation the owner wrote in last, unless archived: where a
/// notice for an owner who only uses the app goes.
pub(crate) async fn latest_app_session(pool: &sqlx::SqlitePool) -> Option<String> {
    let archived = marks(pool).await;
    let recent: Vec<String> = sqlx::query_scalar(
        "SELECT session_id FROM events \
         WHERE event_type = 'user_message' AND session_id LIKE 'app:%' \
         GROUP BY session_id ORDER BY MAX(id) DESC LIMIT 20",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    recent.into_iter().find(|id| {
        id != aidaemon_client_gateway::SETTINGS_SESSION
            && !archived.get(id).is_some_and(|(_, archived)| *archived)
    })
}

#[cfg(test)]
mod tests {
    use crate::testing::{MockProvider, setup_test_agent};
    use crate::traits::store_prelude::*;

    #[tokio::test]
    async fn a_new_message_brings_an_archived_chat_back() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let (state, pool) = (harness.state.clone(), harness.state.pool());
        let chat = "telegram_bot:synthetic-user-1";
        state
            .save_session_channel(chat, "telegram:telegram_bot")
            .await
            .unwrap();
        super::archive(&pool, chat, true).await.unwrap();
        assert!(super::marks(&pool).await[chat].1);
        state
            .chat_seen(chat, "telegram:telegram_bot", false)
            .await
            .unwrap();
        assert!(!super::marks(&pool).await.get(chat).is_some_and(|m| m.1));
    }

    #[tokio::test]
    async fn the_latest_unarchived_app_conversation_is_found() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let (state, pool) = (harness.state.clone(), harness.state.pool());
        for (session, kind) in [
            ("app:older", "user_message"),
            ("app:newer", "user_message"),
            ("app:settings", "user_message"),
            ("app:archived", "user_message"),
            ("telegram:synthetic-user-1", "user_message"),
            ("app:older", "assistant_response"),
        ] {
            sqlx::query("INSERT INTO events (session_id, event_type, data, created_at) VALUES (?, ?, '{}', datetime('now'))")
                .bind(session)
                .bind(kind)
                .execute(&pool)
                .await
                .unwrap();
        }
        // The settings page is not a conversation; the last one written in wins.
        assert_eq!(
            super::latest_app_session(&pool).await.as_deref(),
            Some("app:archived")
        );
        state
            .save_session_channel("app:archived", "app")
            .await
            .unwrap();
        super::archive(&pool, "app:archived", true).await.unwrap();
        assert_eq!(
            super::latest_app_session(&pool).await.as_deref(),
            Some("app:newer")
        );
    }
}
