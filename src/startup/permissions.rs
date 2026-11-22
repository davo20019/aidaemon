//! AIdaemon's permission level at start: the one-time conversions of older
//! shapes into `config.toml`, and the daemon's live value
//! ([`aidaemon_runtime::permissions::Permissions`]) every part shares.
//!
//! Older releases kept the same choices in several places: trust-all in the
//! database (a `*` prefix, then a `terminal_trust_all` row), in
//! `terminal.permission_mode = "yolo"` and in a `*` in
//! `terminal.allowed_prefixes`; each coding agent's mode as a database
//! setting (`cli_agent_mode:<name>`); a level `unrestricted` for every chat.
//! [`upgrade`] moves each into the file once and removes it where it was,
//! never widening what was allowed:
//! - any trust-all becomes `owner_default = "full_access"`;
//! - `unrestricted` becomes `full_access` when trust-all was on, else
//!   `full` (unchecked outward actions after untrusted reads stay a chat's
//!   own `/mode unrestricted`);
//! - each stored agent mode becomes its `[cli_agents.permissions]` entry
//!   (a mode above the level is capped when it runs, and Settings says so);
//! - the unused `terminal.auto_approve` line is removed.

use std::path::Path;
use std::sync::OnceLock;

use aidaemon_runtime::permissions::Permissions;
use aidaemon_tools::config_file::ConfigFile;
use sqlx::SqlitePool;
use tracing::{info, warn};

/// The daemon's live value and its file, published at start for the owner's
/// `/trustall` and the Settings page.
static DAEMON: OnceLock<(Permissions, ConfigFile)> = OnceLock::new();

pub(crate) fn publish(permissions: Permissions, config: ConfigFile) {
    let _ = DAEMON.set((permissions, config));
}

/// The daemon's live value and its file; `None` before start (tests).
pub(crate) fn daemon() -> Option<&'static (Permissions, ConfigFile)> {
    DAEMON.get()
}

const AGENT_MODE_PREFIX: &str = "cli_agent_mode:";

/// What [`upgrade`] changed, for the log.
#[derive(Debug, Default, PartialEq, Eq)]
struct Moved {
    level: Option<&'static str>,
    agents: usize,
    removed: Vec<&'static str>,
}

/// Convert the older shapes into `config.toml` (see the module docs), then
/// read the file's choices. Runs at every start; each conversion acts only
/// while its old shape is there.
pub(crate) async fn upgrade(config_path: &Path, pool: &SqlitePool) -> anyhow::Result<Permissions> {
    let config = crate::config::config_file(config_path);
    let legacy_trust_all = aidaemon_runtime::tools::terminal_grants::legacy_trust_all(pool).await?;
    let agent_modes: Vec<(String, String)> =
        sqlx::query_as("SELECT key, value FROM settings WHERE key LIKE ? ORDER BY key")
            .bind(format!("{AGENT_MODE_PREFIX}%"))
            .fetch_all(pool)
            .await?;
    let text = config.read().await?;
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    let moved = convert(&mut doc, legacy_trust_all, &agent_modes)?;
    if moved != Moved::default() {
        config.save(&doc.to_string()).await?;
        info!(
            level = moved.level.unwrap_or("unchanged"),
            agents = moved.agents,
            removed = ?moved.removed,
            "Moved permission settings into config.toml (Settings: AIdaemon permissions)"
        );
    }
    // Only once the file holds them: a failed write keeps the old shapes
    // for the next start.
    if legacy_trust_all {
        aidaemon_runtime::tools::terminal_grants::drop_legacy_trust_all(pool).await?;
    }
    if !agent_modes.is_empty() {
        sqlx::query("DELETE FROM settings WHERE key LIKE ?")
            .bind(format!("{AGENT_MODE_PREFIX}%"))
            .execute(pool)
            .await?;
    }
    let permissions = Permissions::from_config(&config.load_table().await?);
    info!(
        level = permissions.level().as_str(),
        "AIdaemon's permission level"
    );
    Ok(permissions)
}

/// The conversions on the file's text (see the module docs).
fn convert(
    doc: &mut toml_edit::DocumentMut,
    legacy_trust_all: bool,
    agent_modes: &[(String, String)],
) -> anyhow::Result<Moved> {
    let mut moved = Moved::default();
    let mut trust_all = legacy_trust_all;
    if let Some(terminal) = doc
        .get_mut("terminal")
        .and_then(toml_edit::Item::as_table_like_mut)
    {
        if terminal
            .get("permission_mode")
            .and_then(toml_edit::Item::as_str)
            == Some("yolo")
        {
            terminal.remove("permission_mode");
            moved.removed.push("terminal.permission_mode");
            trust_all = true;
        }
        if let Some(prefixes) = terminal
            .get_mut("allowed_prefixes")
            .and_then(toml_edit::Item::as_array_mut)
        {
            let before = prefixes.len();
            prefixes.retain(|prefix| prefix.as_str() != Some("*"));
            if prefixes.len() != before {
                moved.removed.push("terminal.allowed_prefixes \"*\"");
                trust_all = true;
            }
        }
        if terminal.remove("auto_approve").is_some() {
            moved.removed.push("terminal.auto_approve");
        }
    }

    let current = doc
        .get("authority")
        .and_then(|authority| authority.get("owner_default"))
        .and_then(toml_edit::Item::as_str)
        .map(str::to_string);
    let level = match current.as_deref() {
        Some("unrestricted") => {
            warn!(
                "[authority] owner_default = \"unrestricted\" is now a chat's own \
                 /mode unrestricted; AIdaemon's level becomes {}",
                if trust_all { "Full access" } else { "Standard" }
            );
            Some(if trust_all { "full_access" } else { "full" })
        }
        Some("read_only" | "full_access") => None,
        // `full`, `standard`, unset or unreadable: Standard, raised only to
        // keep trust-all the owner had.
        _ => trust_all.then_some("full_access"),
    };
    if let Some(level) = level {
        authority_table(doc)?.insert("owner_default", toml_edit::value(level));
        moved.level = Some(level);
    }

    for (key, value) in agent_modes {
        let Some(name) = key.strip_prefix(AGENT_MODE_PREFIX) else {
            continue;
        };
        if aidaemon_runtime::permissions::AgentChoice::parse(value).is_none() || name.is_empty() {
            continue;
        }
        let table = agent_table(doc)?;
        // A choice already in the file is the owner's latest.
        if !table.contains_key(name) {
            table.insert(name, toml_edit::value(value.trim()));
            moved.agents += 1;
        }
    }
    Ok(moved)
}

fn authority_table(doc: &mut toml_edit::DocumentMut) -> anyhow::Result<&mut toml_edit::Table> {
    doc.as_table_mut()
        .entry("authority")
        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("`authority` in config.toml is not a table"))
}

fn agent_table(doc: &mut toml_edit::DocumentMut) -> anyhow::Result<&mut toml_edit::Table> {
    let cli_agents = doc
        .as_table_mut()
        .entry("cli_agents")
        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("`cli_agents` in config.toml is not a table"))?;
    let permissions = cli_agents
        .entry("permissions")
        .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()))
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("`cli_agents.permissions` is not a table"))?;
    Ok(permissions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_runtime::permissions::{AgentChoice, OwnerLevel};
    use aidaemon_runtime::tools::agent_mode::PermissionMode as PermissionModeOf;

    fn converted(text: &str, trust_all: bool, modes: &[(&str, &str)]) -> (String, Moved) {
        let mut doc: toml_edit::DocumentMut = text.parse().unwrap();
        let modes: Vec<(String, String)> = modes
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let moved = convert(&mut doc, trust_all, &modes).unwrap();
        (doc.to_string(), moved)
    }

    fn level_of(text: &str) -> OwnerLevel {
        Permissions::from_config(&text.parse().unwrap()).level()
    }

    /// Every older way of saying trust-all becomes Full access, and the old
    /// line goes; nothing is widened.
    #[test]
    fn every_old_trust_all_becomes_full_access_once() {
        let (text, moved) = converted("", true, &[]);
        assert_eq!(level_of(&text), OwnerLevel::FullAccess);
        assert_eq!(moved.level, Some("full_access"));

        let (text, _) = converted(
            "[terminal]\npermission_mode = \"yolo\"\nauto_approve = true\n",
            false,
            &[],
        );
        assert_eq!(level_of(&text), OwnerLevel::FullAccess);
        assert!(!text.contains("permission_mode"), "{text}");
        assert!(!text.contains("auto_approve"), "{text}");

        let (text, _) = converted(
            "# my prefixes\n[terminal]\nallowed_prefixes = [\"ls\", \"*\", \"git\"]\n",
            false,
            &[],
        );
        assert_eq!(level_of(&text), OwnerLevel::FullAccess);
        assert!(text.contains("# my prefixes"), "{text}");
        assert!(
            text.contains("allowed_prefixes = [\"ls\", \"git\"]"),
            "{text}"
        );

        // A second start changes nothing.
        let (again, moved) = converted(&text, false, &[]);
        assert_eq!(again, text);
        assert_eq!(moved, Moved::default());

        // Nothing old: nothing written, Standard.
        let (text, moved) = converted("[terminal]\npermission_mode = \"cautious\"\n", false, &[]);
        assert_eq!(moved, Moved::default());
        assert_eq!(level_of(&text), OwnerLevel::Full);
    }

    /// `unrestricted` becomes a chat's own mode: the level is the narrower
    /// of what the owner had.
    #[test]
    fn unrestricted_narrows_to_what_the_terminal_had() {
        let unrestricted = "[authority]\nowner_default = \"unrestricted\"\n";
        assert_eq!(
            level_of(&converted(unrestricted, false, &[]).0),
            OwnerLevel::Full
        );
        assert_eq!(
            level_of(&converted(unrestricted, true, &[]).0),
            OwnerLevel::FullAccess
        );
        // A level the owner set stays.
        let read_only = "[authority]\nowner_default = \"read_only\"\n";
        let (text, moved) = converted(read_only, true, &[]);
        assert_eq!(level_of(&text), OwnerLevel::ReadOnly);
        assert_eq!(moved.level, None);
    }

    /// Stored agent modes move into the file; one already there wins.
    #[test]
    fn stored_agent_modes_move_into_the_file() {
        let (text, moved) = converted(
            "[cli_agents.permissions]\ncodex = \"every_command\"\n",
            false,
            &[
                ("cli_agent_mode:claude", "full_access"),
                ("cli_agent_mode:codex", "full_access"),
                ("cli_agent_mode:broken", "sideways"),
            ],
        );
        assert_eq!(moved.agents, 1);
        let permissions = Permissions::from_config(&text.parse().unwrap());
        assert_eq!(
            permissions.choice("claude"),
            Some(AgentChoice::Mode(PermissionModeOf::FullAccess))
        );
        assert_eq!(
            permissions.choice("codex"),
            Some(AgentChoice::Mode(PermissionModeOf::EveryCommand))
        );
        assert_eq!(permissions.choice("broken"), None);
    }

    /// The whole start on a database: old rows go only once the file holds
    /// them, and a second start finds nothing to move.
    #[tokio::test]
    async fn the_start_moves_old_rows_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            concat!("[terminal]\nallowed_prefixes = [\"ls\"]\n", "[provider]\nkind = \"openai_compatible\"\napi_key = \"keychain\"\nbase_url = \"https://h.example.invalid/v1\"\n\n[provider.models]\ndefault = \"synthetic-model-1\"\n"),
        )
        .unwrap();
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        for statement in [
            "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT)",
            "INSERT INTO settings (key, value) VALUES ('cli_agent_mode:claude', 'full_access'),
                ('other', 'kept')",
            "CREATE TABLE terminal_trust_all (backend_scope TEXT PRIMARY KEY, created_at TEXT)",
            "INSERT INTO terminal_trust_all (backend_scope) VALUES ('local')",
        ] {
            sqlx::query(statement).execute(&pool).await.unwrap();
        }
        let permissions = upgrade(&path, &pool).await.unwrap();
        assert_eq!(permissions.level(), OwnerLevel::FullAccess);
        assert_eq!(
            permissions.choice("claude"),
            Some(AgentChoice::Mode(PermissionModeOf::FullAccess))
        );
        let left: Vec<String> = sqlx::query_scalar("SELECT key FROM settings")
            .fetch_all(&pool)
            .await
            .unwrap();
        assert_eq!(left, ["other"]);
        assert!(
            !aidaemon_runtime::tools::terminal_grants::legacy_trust_all(&pool)
                .await
                .unwrap()
        );

        // The owner turns it off; the next start keeps it off.
        let file = crate::config::config_file(&path);
        aidaemon_runtime::permissions::save_level(&file, &permissions, OwnerLevel::Full)
            .await
            .unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            upgrade(&path, &pool).await.unwrap().level(),
            OwnerLevel::Full
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), written);
    }
}
