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
