//! The daemon's configuration of crate tools: the file tools' wording and
//! their [`HostFilePolicy`] (the daemon's own files, backups under
//! `~/.aidaemon/file-backups`), `web_fetch` and `web_search` as the daemon
//! words them, and the skill library mandate strategies read.

use std::sync::Arc;

use aidaemon_tools::web_search::{SearchConfig, WebSearch};
use aidaemon_tools::{FilePolicy, HostFilePolicy, HostFilePolicySettings, ToolText};
use aidaemon_work::tools::manage_mandates::{StrategySkill, StrategySkills};

use crate::execution::SharedExecutionBackend;
pub use aidaemon_tools::path_class::DaemonPaths;

/// The file tools' policy on the daemon's host filesystem.
pub(crate) fn daemon_file_policy(paths: DaemonPaths) -> Arc<dyn FilePolicy> {
    Arc::new(HostFilePolicy::new(
        paths,
        HostFilePolicySettings {
            shell_tool_name: "terminal".to_string(),
            delivery_hint: Some(
                "When the file is complete, deliver it with send_file instead of pasting it inline."
                    .to_string(),
            ),
            own_file_hint: Some(
                "This is one of aidaemon's own files, which change only through manage_config: \
                 read the config with manage_config action 'read' (secrets hidden), and change one \
                 setting with 'set' or 'unset'."
                    .to_string(),
            ),
            keeps_backups: true,
            backup_root: file_backup_root(),
        },
    ))
}

#[cfg(not(test))]
pub(crate) fn file_backup_root() -> Option<std::path::PathBuf> {
    Some(dirs::home_dir()?.join(".aidaemon").join("file-backups"))
}

#[cfg(test)]
pub(crate) fn file_backup_root() -> Option<std::path::PathBuf> {
    Some(std::env::temp_dir().join("aidaemon-test-file-backups"))
}

/// The daemon's constructors for the SDK's [`aidaemon_tools::ReadFile`].
pub struct ReadFileTool;

#[allow(clippy::new_ret_no_self)]
impl ReadFileTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> aidaemon_tools::ReadFile {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(
        backend: SharedExecutionBackend,
        paths: DaemonPaths,
    ) -> aidaemon_tools::ReadFile {
        aidaemon_tools::ReadFile::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Read file contents with line numbers and metadata. Read a fitting file in full once; for large files, search first and then read one exact non-overlapping range.".to_string(),
                schema: "Read file contents with line numbers and metadata. Use this instead of terminal cat/head/tail. Read a file in full once when it fits the limit; oversized reads return the first page plus an exact continuation call to follow. For a large file with a known target, use search_files first, then read one exact surrounding range. Sequential ranges must not overlap; previously covered content is replayed from the task-local artifact.".to_string(),
            })
    }
}

/// The daemon's constructors for the SDK's [`aidaemon_tools::WriteFile`].
pub struct WriteFileTool;

#[allow(clippy::new_ret_no_self)]
impl WriteFileTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> aidaemon_tools::WriteFile {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(
        backend: SharedExecutionBackend,
        paths: DaemonPaths,
    ) -> aidaemon_tools::WriteFile {
        aidaemon_tools::WriteFile::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Write or create a file with given content".to_string(),
                schema: "Write content to a file, creating it if it doesn't exist. Use this instead of terminal echo/cat for file creation. Creates a backup of existing files before overwriting. ALWAYS prefer write_file over terminal heredocs (cat > file << EOF) — heredocs trigger the approval flow.".to_string(),
            })
    }
}

/// The daemon's constructors for the SDK's [`aidaemon_tools::EditFile`].
pub struct EditFileTool;

#[allow(clippy::new_ret_no_self)]
impl EditFileTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> aidaemon_tools::EditFile {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(
        backend: SharedExecutionBackend,
        paths: DaemonPaths,
    ) -> aidaemon_tools::EditFile {
        aidaemon_tools::EditFile::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Find and replace text in a file".to_string(),
                schema: "Find and replace text in a file. Use this instead of terminal sed/awk. Shows context around the change. Fails safely if the text isn't found or is ambiguous. On not-found/ambiguous text, read_file the same path and retry once before asking the user.".to_string(),
            })
    }
}

/// The daemon's constructors for the SDK's [`aidaemon_tools::SearchFiles`].
pub struct SearchFilesTool;

#[allow(clippy::new_ret_no_self)]
impl SearchFilesTool {
    /// Without the daemon's own files protected; production passes them.
    #[cfg(test)]
    pub fn new(backend: SharedExecutionBackend) -> aidaemon_tools::SearchFiles {
        Self::on_backend(backend, DaemonPaths::default())
    }

    pub fn on_backend(
        backend: SharedExecutionBackend,
        paths: DaemonPaths,
    ) -> aidaemon_tools::SearchFiles {
        aidaemon_tools::SearchFiles::on_backend(backend, daemon_file_policy(paths)).with_text(ToolText {
                short: "Search for files by name pattern or content regex".to_string(),
                schema: "Search for files by name pattern (glob) and/or content (regex). Use this instead of terminal find/grep. Automatically skips .git, node_modules, target, etc.".to_string(),
            })
    }
}

/// The daemon's constructors for the SDK's [`aidaemon_tools::WebFetch`].
pub struct WebFetchTool;

#[allow(clippy::new_ret_no_self)]
impl WebFetchTool {
    pub fn new() -> aidaemon_tools::WebFetch {
        aidaemon_tools::WebFetch::new()
                .browser_identity()
                .follow_public_redirects()
                .with_text(ToolText {
                    short: "Fetch a readable web page and extract its content; not for REST/JSON API endpoints".to_string(),
                    schema: "Fetch a readable web page and extract its content. Strips ads/navigation. Do NOT use for REST/JSON API endpoints or machine-readable responses; use http_request for APIs. For login-required sites, use browser instead.".to_string(),
                })
    }
}

/// What the daemon's model is told: the crate's description, naming where
/// the owner configures search.
const DAEMON_DESCRIPTION: &str = "Search the web. Queries all configured search backends in parallel and returns merged, deduplicated results (titles, URLs, snippets, publication age when known). Use to find current information, research topics, check facts. One focused search is almost always enough; for single-fact lookups do NOT re-search with rephrased queries — synthesize promptly. Results are snippet previews, NOT full pages: never assemble complete lists, rosters, or enumerations from snippets — fetch a source page with web_fetch and answer from its text. Set 'freshness' for time-sensitive queries (news, prices, releases). If results are consistently empty, the search backend may be blocked — suggest the user set up Brave Search via manage_config (search.backend = 'brave' + search.api_key) or a self-hosted SearxNG instance (search.searxng_url).";

/// The daemon's constructor for the SDK's [`aidaemon_tools::web_search::WebSearch`].
pub struct WebSearchTool;

#[allow(clippy::new_ret_no_self)]
impl WebSearchTool {
    pub fn new(config: &SearchConfig) -> WebSearch {
        WebSearch::new(config).with_description(DAEMON_DESCRIPTION)
    }
}

/// The daemon's filesystem skills under `dir`, as mandate strategies read
/// them.
pub(crate) fn strategy_skills_dir(dir: std::path::PathBuf) -> std::sync::Arc<dyn StrategySkills> {
    std::sync::Arc::new(SkillsDirStrategies(dir))
}

struct SkillsDirStrategies(std::path::PathBuf);

impl StrategySkills for SkillsDirStrategies {
    fn find(&self, name: &str) -> Option<StrategySkill> {
        let skills = crate::skills::load_skills(&self.0);
        let skill = crate::skills::find_skill_by_name(&skills, name)?;
        Some(StrategySkill {
            name: skill.name.clone(),
            description: skill.description.clone(),
            body: crate::tools::sanitize::sanitize_external_content(&skill.body),
            source: skill.source.clone(),
            canonical: skill.to_markdown(),
        })
    }
}
