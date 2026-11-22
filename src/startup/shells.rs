//! Conversation shells in the daemon: one registry for the process. The
//! Client Gateway serves them to the owner's apps; the agent reads and
//! types into its conversation's shell.

use std::path::PathBuf;
use std::sync::OnceLock;

use aidaemon_shells::ShellRegistry;

use crate::config::AppConfig;
use crate::traits::StateStore;

static REGISTRY: OnceLock<ShellRegistry> = OnceLock::new();

/// The process's shell registry, made on first use.
pub(crate) fn registry(config: &AppConfig) -> ShellRegistry {
    REGISTRY
        .get_or_init(|| ShellRegistry::new(config.shells.registry_config()))
        .clone()
}

/// The folder a conversation's shell starts in: its project's, when it has
/// one with a folder that exists; `None` otherwise (the owner's home).
pub(crate) async fn session_folder(
    state: &dyn StateStore,
    session_id: &str,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(project_id) = state
        .session_work_projects()
        .await?
        .into_iter()
        .find_map(|(session, project)| (session == session_id).then_some(project))
    else {
        return Ok(None);
    };
    Ok(state
        .list_work_projects()
        .await?
        .into_iter()
        .find(|project| project.id == project_id)
        .and_then(|project| project.path)
        .map(PathBuf::from)
        .filter(|folder| folder.is_dir()))
}
