//! Saving a claimed bot's access ids in config.toml, in that bot's own
//! entry: `[telegram]` / `[slack]` (the single-bot form) or its element of
//! `[[telegram_bots]]` / `[[slack_bots]]`, found by its token.

use std::path::Path;
use std::sync::LazyLock;

use tokio::sync::Mutex;

/// One writer at a time for the channel code's config edits (claims and
/// Slack workspace grants), so one cannot overwrite another's change with a
/// stale read.
pub(crate) static CONFIG_WRITE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Where a bot's settings live in the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entry {
    /// `[legacy]`.
    Single,
    /// `[[array]]` element `i`.
    Listed(usize),
}

/// The token a config entry holds, as the config loader resolves it: a
/// literal, `${VAR}` references, or `"keychain"` (read from the entry's
/// keychain name).
fn entry_token(key: &str, raw: &str) -> Option<String> {
    if raw == "keychain" {
        return crate::config::resolve_from_keychain(&crate::config::keychain_name_for(key)?).ok();
    }
    crate::config::expand_env_vars(raw).ok()
}

/// The entry of the bot with `bot_token` under `[single]` or `[[array]]`.
fn find_entry(doc: &toml::Table, single: &str, array: &str, bot_token: &str) -> Option<Entry> {
    let token_of = |table: &toml::Table, key: String| {
        table
            .get("bot_token")
            .and_then(toml::Value::as_str)
            .and_then(|raw| entry_token(&key, raw))
    };
    if let Some(bots) = doc.get(array).and_then(toml::Value::as_array) {
        for (i, bot) in bots.iter().enumerate() {
            if let Some(table) = bot.as_table()
                && token_of(table, format!("{array}[{i}].bot_token")).as_deref() == Some(bot_token)
            {
                return Some(Entry::Listed(i));
            }
        }
    }
    doc.get(single)
        .and_then(toml::Value::as_table)
        .filter(|table| {
            token_of(table, format!("{single}.bot_token")).as_deref() == Some(bot_token)
        })
        .map(|_| Entry::Single)
}

/// Set `allowed_user_ids` on the entry of the bot with `bot_token` (under
/// `[single]` or `[[array]]`), let `also` change the rest of the document
/// (owner ids), and save it through [`crate::config::config_file`]: checked,
/// backed up, written atomically and owner-only.
pub(crate) async fn save_allowed_ids(
    config_path: &Path,
    single: &str,
    array: &str,
    bot_token: &str,
    allowed: toml::Value,
    also: impl FnOnce(&mut toml::Table) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let _guard = CONFIG_WRITE_LOCK.lock().await;
    let file = crate::config::config_file(config_path);
    let mut doc = file.load_table().await?;
    let table = match find_entry(&doc, single, array, bot_token) {
        Some(Entry::Single) => doc.get_mut(single).and_then(toml::Value::as_table_mut),
        Some(Entry::Listed(i)) => doc
            .get_mut(array)
            .and_then(toml::Value::as_array_mut)
            .and_then(|bots| bots.get_mut(i))
            .and_then(toml::Value::as_table_mut),
        None => None,
    }
    .ok_or_else(|| {
        anyhow::anyhow!("This bot has no [{single}] or [[{array}]] entry in the config")
    })?;
    table.insert("allowed_user_ids".to_string(), allowed);
    also(&mut doc)?;
    file.save_table(&doc).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_claimed_bot_s_own_entry_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[provider]\nkind = \"openai_chatgpt\"\napi_key = \"\"\n\n\
             [[telegram_bots]]\nbot_token = \"111:synthetic-a\"\n\n\
             [[telegram_bots]]\nbot_token = \"222:synthetic-b\"\n",
        )
        .unwrap();
        save_allowed_ids(
            &path,
            "telegram",
            "telegram_bots",
            "222:synthetic-b",
            toml::Value::Array(vec![toml::Value::Integer(42)]),
            |_| Ok(()),
        )
        .await
        .unwrap();
        let doc: toml::Table = std::fs::read_to_string(&path).unwrap().parse().unwrap();
        let bots = doc["telegram_bots"].as_array().unwrap();
        assert!(bots[0].get("allowed_user_ids").is_none());
        assert_eq!(bots[1]["allowed_user_ids"][0].as_integer(), Some(42));
    }

    #[tokio::test]
    async fn an_unknown_token_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let before = "[provider]\nkind = \"openai_chatgpt\"\napi_key = \"\"\n\n\
                      [telegram]\nbot_token = \"111:synthetic-a\"\n";
        std::fs::write(&path, before).unwrap();
        assert!(
            save_allowed_ids(
                &path,
                "telegram",
                "telegram_bots",
                "999:other",
                toml::Value::Array(vec![]),
                |_| Ok(()),
            )
            .await
            .is_err()
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    }
}
