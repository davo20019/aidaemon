use super::*;

#[async_trait]
impl crate::traits::DynamicBotStore for SqliteStateStore {
    async fn add_dynamic_bot(&self, bot: &crate::traits::DynamicBot) -> anyhow::Result<i64> {
        let allowed_user_ids_json = serde_json::to_string(&bot.allowed_user_ids)?;

        // Store tokens in OS keychain to avoid plaintext storage in SQLite.
        // We insert first to get the row ID, then store in keychain and update
        // the row to hold a "keychain:key" reference instead of the raw token.
        let result = sqlx::query(
            "INSERT INTO dynamic_bots (channel_type, bot_token, app_token, allowed_user_ids, extra_config, created_at)
             VALUES (?, ?, ?, ?, ?, datetime('now'))"
        )
        .bind(&bot.channel_type)
        .bind(&bot.bot_token) // Temporarily store plaintext; will be replaced below
        .bind(&bot.app_token)
        .bind(&allowed_user_ids_json)
        .bind(&bot.extra_config)
        .execute(&self.pool)
        .await?;
        let row_id = result.last_insert_rowid();

        // Try to move the bot_token to keychain
        let bot_token_key = format!("dynamic_bot_{}_bot_token", row_id);
        if crate::config::store_in_keychain(&bot_token_key, &bot.bot_token).is_ok() {
            // Replace plaintext with keychain reference
            let _ = sqlx::query("UPDATE dynamic_bots SET bot_token = ? WHERE id = ?")
                .bind(format!("keychain:{}", bot_token_key))
                .bind(row_id)
                .execute(&self.pool)
                .await;
        }

        // Try to move the app_token to keychain (Slack bots)
        if let Some(ref app_tok) = bot.app_token {
            let app_token_key = format!("dynamic_bot_{}_app_token", row_id);
            if crate::config::store_in_keychain(&app_token_key, app_tok).is_ok() {
                let _ = sqlx::query("UPDATE dynamic_bots SET app_token = ? WHERE id = ?")
                    .bind(format!("keychain:{}", app_token_key))
                    .bind(row_id)
                    .execute(&self.pool)
                    .await;
            }
        }

        Ok(row_id)
    }

    async fn update_dynamic_bot_allowed_users(
        &self,
        bot_token: &str,
        allowed_user_ids: &[String],
    ) -> anyhow::Result<()> {
        // Find the bot by resolved token (tokens may be stored as keychain refs)
        let bots = self.get_dynamic_bots().await?;
        let bot = bots
            .iter()
            .find(|b| b.bot_token == bot_token)
            .ok_or_else(|| anyhow::anyhow!("Dynamic bot not found for token"))?;

        let allowed_json = serde_json::to_string(allowed_user_ids)?;
        sqlx::query("UPDATE dynamic_bots SET allowed_user_ids = ? WHERE id = ?")
            .bind(&allowed_json)
            .bind(bot.id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn get_dynamic_bots(&self) -> anyhow::Result<Vec<crate::traits::DynamicBot>> {
        let rows = sqlx::query(
            "SELECT id, channel_type, bot_token, app_token, allowed_user_ids, extra_config, created_at
             FROM dynamic_bots ORDER BY created_at ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut bots = Vec::with_capacity(rows.len());
        for row in rows {
            let allowed_user_ids_json: String = row.get("allowed_user_ids");
            let allowed_user_ids: Vec<String> =
                serde_json::from_str(&allowed_user_ids_json).unwrap_or_default();

            // Resolve keychain references: "keychain:key_name" -> actual value
            let raw_bot_token: String = row.get("bot_token");
            let bot_token = resolve_keychain_ref(&raw_bot_token);

            let raw_app_token: Option<String> = row.get("app_token");
            let app_token = raw_app_token.map(|t| resolve_keychain_ref(&t));

            bots.push(crate::traits::DynamicBot {
                id: row.get("id"),
                channel_type: row.get("channel_type"),
                bot_token,
                app_token,
                allowed_user_ids,
                extra_config: row.get("extra_config"),
                created_at: row.get("created_at"),
            });
        }
        Ok(bots)
    }

    async fn delete_dynamic_bot(&self, id: i64) -> anyhow::Result<()> {
        // Clean up keychain entries for this bot
        let _ = crate::config::delete_from_keychain(&format!("dynamic_bot_{}_bot_token", id));
        let _ = crate::config::delete_from_keychain(&format!("dynamic_bot_{}_app_token", id));

        sqlx::query("DELETE FROM dynamic_bots WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}
