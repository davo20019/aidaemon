//! Projects for the app (protocol 1.10): the work-project store's projects,
//! each with the folder its conversations work in. A folder can be anything
//! on this computer (a repository, a site, a folder of designs); it must
//! exist and is stored as an absolute, resolved path. A missing folder is
//! made only when the owner confirms it (protocol 1.24), and only under the
//! home folder. A chat bot can send its direct-message chats to a project
//! (protocol 1.12).

use std::path::{Path, PathBuf};

use aidaemon_client_protocol::{ErrorCode, ProjectSummary, ProtocolError};
use aidaemon_work::traits::{DEFAULT_PROJECT_ID, WorkProject};

use crate::traits::StateStore;

fn summary(project: WorkProject, bots: &[(String, String)]) -> ProjectSummary {
    ProjectSummary {
        channels: bots
            .iter()
            .filter(|(_, project_id)| *project_id == project.id)
            .map(|(channel, _)| channel.clone())
            .collect(),
        id: project.id,
        name: project.name,
        path: project.path,
    }
}

/// Every project but the default one (which holds conversations in no
/// project), by name.
pub(crate) async fn projects(state: &dyn StateStore) -> anyhow::Result<Vec<ProjectSummary>> {
    let bots = state.channel_work_projects().await?;
    Ok(state
        .list_work_projects()
        .await?
        .into_iter()
        .filter(|project| project.id != DEFAULT_PROJECT_ID)
        .map(|project| summary(project, &bots))
        .collect())
}

/// Send chat bot `channel`'s direct-message chats to `project_id`, or to
/// no project.
pub(crate) async fn send_bot(
    state: &dyn StateStore,
    channel: &str,
    project_id: Option<&str>,
) -> Result<(), String> {
    if channel.trim().is_empty() || channel == aidaemon_client_gateway::APP_CHANNEL {
        return Err("Only a chat bot's chats can go to a project this way".to_string());
    }
    match state.set_channel_work_project(channel, project_id).await {
        Ok(true) => Ok(()),
        Ok(false) => Err("There is no such project".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

fn refused(why: impl Into<String>) -> ProtocolError {
    ProtocolError::new(ErrorCode::Invalid, why)
}

pub(crate) async fn create(
    state: &dyn StateStore,
    name: &str,
    path: Option<&str>,
    create_folder: bool,
) -> Result<ProjectSummary, ProtocolError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(refused("A project needs a name"));
    }
    let taken = state
        .list_work_projects()
        .await
        .map_err(|e| refused(e.to_string()))?
        .iter()
        .any(|project| project.name.eq_ignore_ascii_case(name));
    if taken {
        return Err(refused(format!("There is already a project called {name}")));
    }
    // The name is checked first, so a refused project makes no folder.
    let folder = match path.map(str::trim).filter(|p| !p.is_empty()) {
        Some(path) => Some(folder(path, create_folder)?),
        None => None,
    };
    let project = state
        .create_work_project(name, None)
        .await
        .map_err(|e| refused(e.to_string()))?;
    match folder {
        Some(folder) => state
            .update_work_project(&project.id, None, Some(Some(&folder)))
            .await
            .map_err(|e| refused(e.to_string()))?
            .map(|project| summary(project, &[]))
            .ok_or_else(|| refused("The project was made but its folder was not saved")),
        None => Ok(summary(project, &[])),
    }
}

pub(crate) async fn update(
    state: &dyn StateStore,
    id: &str,
    name: Option<&str>,
    path: Option<&str>,
    create_folder: bool,
) -> Result<ProjectSummary, ProtocolError> {
    if id == DEFAULT_PROJECT_ID {
        return Err(refused("Conversations in no project cannot be renamed"));
    }
    let exists = state
        .list_work_projects()
        .await
        .map_err(|e| refused(e.to_string()))?
        .iter()
        .any(|project| project.id == id);
    if !exists {
        return Err(refused("There is no such project"));
    }
    let path = match path.map(str::trim) {
        Some("") => Some(None),
        Some(path) => Some(Some(folder(path, create_folder)?)),
        None => None,
    };
    let clears_folder = matches!(path, Some(None));
    let bots = state
        .channel_work_projects()
        .await
        .map_err(|e| refused(e.to_string()))?;
    let updated = state
        .update_work_project(id, name, path.as_ref().map(|p| p.as_deref()))
        .await
        .map_err(|e| refused(e.to_string()))?
        .map(|project| summary(project, &bots))
        .ok_or_else(|| refused("There is no such project"))?;
    if clears_folder {
        clear_project_agent_pins(state, id).await;
    }
    Ok(updated)
}

/// Delete a project; its conversations go to no project and its folder
/// stays where it is.
pub(crate) async fn delete(state: &dyn StateStore, id: &str) -> Result<(), ProtocolError> {
    if id == DEFAULT_PROJECT_ID {
        return Err(refused("Conversations in no project cannot be deleted"));
    }
    let sessions: Vec<String> = state
        .session_work_projects()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(session_id, project_id)| (project_id == id).then_some(session_id))
        .collect();
    match state.delete_work_project(id).await {
        Ok(true) => {
            for session_id in sessions {
                clear_agent_pin(state, &session_id).await;
            }
            Ok(())
        }
        Ok(false) => Err(refused("There is no such project")),
        Err(e) => Err(refused(e.to_string())),
    }
}

pub(crate) async fn place(
    state: &dyn StateStore,
    session_id: &str,
    project_id: Option<&str>,
) -> Result<(), String> {
    let project_id = project_id.unwrap_or(DEFAULT_PROJECT_ID);
    match state.set_session_work_project(session_id, project_id).await {
        Ok(true) => {
            // A coding-agent pin requires a folder-backed project: clear it
            // when a conversation moves out of one.
            if session_folder(state, session_id).await.is_none() {
                clear_agent_pin(state, session_id).await;
            }
            Ok(())
        }
        Ok(false) => Err("There is no such project".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

async fn clear_agent_pin(state: &dyn StateStore, session_id: &str) {
    let mut setting = aidaemon_runtime::session_model::get(state, session_id).await;
    if setting.agent.take().is_some() {
        let _ = aidaemon_runtime::session_model::set(state, session_id, &setting).await;
    }
}

async fn clear_project_agent_pins(state: &dyn StateStore, project_id: &str) {
    let Ok(assignments) = state.session_work_projects().await else {
        return;
    };
    for (session_id, assigned) in assignments {
        if assigned == project_id {
            clear_agent_pin(state, &session_id).await;
        }
    }
}

/// The folder `session_id`'s project works in, when it is in one that has
/// a folder.
pub(crate) async fn session_folder(state: &dyn StateStore, session_id: &str) -> Option<PathBuf> {
    let project_id = state.get_session_work_project(session_id).await.ok()?;
    if project_id == DEFAULT_PROJECT_ID {
        return None;
    }
    state
        .list_work_projects()
        .await
        .ok()?
        .into_iter()
        .find(|project| project.id == project_id)?
        .path
        .map(PathBuf::from)
}

/// `/project` in a chat: which project the chat works in, or (the owner)
/// put it in one by name, or in none.
pub(crate) struct ProjectCommand {
    pub state: std::sync::Arc<dyn StateStore>,
}

#[async_trait::async_trait]
impl aidaemon_channels::slash::SlashCommand for ProjectCommand {
    fn handles(&self, cmd: &str) -> bool {
        cmd == "/project"
    }

    fn commands(&self) -> Vec<aidaemon_channels::slash::CommandDef> {
        vec![aidaemon_channels::slash::CommandDef {
            name: "project",
            description: "Show or change the project this chat works in",
            usage: Some("/project [name|none]"),
            category: aidaemon_channels::slash::CommandCategory::Core,
        }]
    }

    async fn run(
        &self,
        _cmd: &str,
        args: &str,
        call: &aidaemon_channels::slash::SlashCall<'_>,
    ) -> String {
        project_command(self.state.as_ref(), call.session_id, args, call.user_role).await
    }
}

async fn project_command(
    state: &dyn StateStore,
    session_id: &str,
    args: &str,
    role: crate::types::UserRole,
) -> String {
    let all = match projects(state).await {
        Ok(all) => all,
        Err(e) => return format!("Could not read the projects: {e}"),
    };
    let wanted = args.trim();
    if wanted.is_empty() {
        let current = state
            .get_session_work_project(session_id)
            .await
            .ok()
            .and_then(|id| all.iter().find(|p| p.id == id));
        let here = match current {
            Some(project) => format!("This chat works in {}.", described(project)),
            None => "This chat is in no project.".to_string(),
        };
        if all.is_empty() {
            return format!("{here} There are no projects yet; make one in the app.");
        }
        let names: Vec<&str> = all.iter().map(|p| p.name.as_str()).collect();
        return format!(
            "{here}\nProjects: {}.\nUse /project <name> to work in one, /project none to leave it.",
            names.join(", ")
        );
    }
    if role != crate::types::UserRole::Owner {
        return "Only the owner can change this chat's project.".to_string();
    }
    if wanted.eq_ignore_ascii_case("none") {
        return match place(state, session_id, None).await {
            Ok(()) => "This chat is in no project now.".to_string(),
            Err(why) => format!("Could not change it: {why}"),
        };
    }
    let lower = wanted.to_lowercase();
    let exact: Vec<_> = all
        .iter()
        .filter(|p| p.name.to_lowercase() == lower)
        .collect();
    let matches = if exact.is_empty() {
        all.iter()
            .filter(|p| p.name.to_lowercase().starts_with(&lower))
            .collect()
    } else {
        exact
    };
    match matches.as_slice() {
        [project] => match place(state, session_id, Some(&project.id)).await {
            Ok(()) => format!(
                "This chat works in {} now; its next messages run there.",
                described(project)
            ),
            Err(why) => format!("Could not change it: {why}"),
        },
        [] => format!(
            "No project is called {wanted}. Projects: {}.",
            all.iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        several => format!(
            "{wanted} could be {}. Use the full name.",
            several
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(" or ")
        ),
    }
}

fn described(project: &ProjectSummary) -> String {
    match &project.path {
        Some(path) => format!("{} ({path})", project.name),
        None => project.name.clone(),
    }
}

/// The project's folder as stored: `~` expanded, resolved, and existing.
/// A missing folder is `FolderMissing`, or with `create` is made (with any
/// missing parents) when it lies under the home folder.
fn folder(path: &str, create: bool) -> Result<String, ProtocolError> {
    let home = || {
        dirs::home_dir().ok_or_else(|| refused("This computer has no home folder to start from"))
    };
    // Only `~` and `~/…` are the owner's home; `~bob/x` is not `$HOME/bob/x`.
    let expanded: PathBuf = if path == "~" {
        home()?
    } else if let Some(rest) = path.strip_prefix("~/") {
        home()?.join(rest.trim_start_matches('/'))
    } else {
        PathBuf::from(path)
    };
    if !expanded.is_absolute() {
        return Err(refused(format!(
            "Give the folder's full path (for example ~/projects/{}), not a relative one",
            Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "acme-site".to_string())
        )));
    }
    if !expanded.exists() {
        if !create {
            return Err(ProtocolError::new(
                ErrorCode::FolderMissing,
                format!(
                    "There is no folder at {} on this computer",
                    expanded.display()
                ),
            ));
        }
        make_folder(&expanded, &home()?)?;
    }
    let resolved = dunce::canonicalize(&expanded).map_err(|_| {
        refused(format!(
            "There is no folder at {} on this computer",
            expanded.display()
        ))
    })?;
    if !resolved.is_dir() {
        return Err(refused(format!(
            "{} is a file, not a folder",
            resolved.display()
        )));
    }
    Ok(resolved.to_string_lossy().into_owned())
}

/// Make `path` and its missing parents, only under `home`: no `..`, and its
/// nearest existing parent (links resolved) is a folder inside `home`.
fn make_folder(path: &Path, home: &Path) -> Result<(), ProtocolError> {
    let outside = || {
        refused(format!(
            "Only a folder inside your home folder can be made; make {} yourself",
            path.display()
        ))
    };
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(outside());
    }
    let home = dunce::canonicalize(home).map_err(|_| outside())?;
    let parent = path
        .ancestors()
        .skip(1)
        .find(|p| p.exists())
        .ok_or_else(outside)?;
    let parent_resolved = dunce::canonicalize(parent).map_err(|_| outside())?;
    if !parent_resolved.starts_with(&home) {
        return Err(outside());
    }
    if !parent_resolved.is_dir() {
        return Err(refused(format!(
            "{} is a file, not a folder",
            parent_resolved.display()
        )));
    }
    std::fs::create_dir_all(path)
        .map_err(|e| refused(format!("Could not make {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_deleted_project_leaves_its_folder_and_its_conversations_in_none() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state: &dyn StateStore = harness.state.as_ref();
        let site = tempfile::tempdir().unwrap();
        let made = create(
            state,
            "Acme site",
            Some(&site.path().to_string_lossy()),
            false,
        )
        .await
        .unwrap();
        place(state, "app:synthetic-1", Some(&made.id))
            .await
            .unwrap();

        delete(state, &made.id).await.unwrap();

        assert!(projects(state).await.unwrap().is_empty());
        assert_eq!(
            state
                .get_session_work_project("app:synthetic-1")
                .await
                .unwrap(),
            DEFAULT_PROJECT_ID
        );
        assert!(site.path().is_dir());
        assert!(delete(state, &made.id).await.is_err());
        assert!(delete(state, DEFAULT_PROJECT_ID).await.is_err());
    }

    #[tokio::test]
    async fn a_project_is_made_with_an_existing_folder_and_holds_sessions() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state: &dyn StateStore = harness.state.as_ref();
        let site = tempfile::tempdir().unwrap();
        let made = create(
            state,
            "Acme site",
            Some(&site.path().to_string_lossy()),
            false,
        )
        .await
        .unwrap();
        assert_eq!(
            made.path.as_deref(),
            Some(dunce::canonicalize(site.path()).unwrap().to_str().unwrap())
        );
        // The same name again, a missing folder and a relative one are refused.
        assert!(create(state, "acme SITE", None, false).await.is_err());
        let missing = create(
            state,
            "Gone",
            Some(&crate::testing::abs_path("/nowhere/synthetic-missing")),
            false,
        )
        .await
        .unwrap_err();
        assert_eq!(missing.code, ErrorCode::FolderMissing);
        assert!(missing.message.contains("no folder"), "{missing:?}");
        // Outside the home folder, confirming does not make it.
        let outside = create(
            state,
            "Gone",
            Some(&crate::testing::abs_path("/nowhere/synthetic-missing")),
            true,
        )
        .await
        .unwrap_err();
        assert_eq!(outside.code, ErrorCode::Invalid);
        assert!(projects(state).await.unwrap().len() == 1);
        assert!(
            create(state, "Rel", Some("projects/acme"), false)
                .await
                .is_err()
        );
        // Listed without the default project.
        let listed = projects(state).await.unwrap();
        assert_eq!(listed, vec![made.clone()]);
        // A folder can be removed and the project renamed.
        let bare = update(state, &made.id, Some("Acme Corp"), Some(""), false)
            .await
            .unwrap();
        assert_eq!((bare.name.as_str(), bare.path), ("Acme Corp", None));
        // Sessions move in and out.
        place(state, "app:synthetic-1", Some(&made.id))
            .await
            .unwrap();
        assert_eq!(
            state.session_work_projects().await.unwrap(),
            vec![("app:synthetic-1".to_string(), made.id.clone())]
        );
        place(state, "app:synthetic-1", None).await.unwrap();
        assert!(state.session_work_projects().await.unwrap().is_empty());
        assert!(
            place(state, "app:synthetic-1", Some("project-missing"))
                .await
                .is_err()
        );
        // A bot's direct-message chats go to the project, and it lists the bot.
        send_bot(state, "telegram:synthetic_coding_bot", Some(&made.id))
            .await
            .unwrap();
        assert_eq!(
            projects(state).await.unwrap()[0].channels,
            vec!["telegram:synthetic_coding_bot".to_string()]
        );
        assert!(send_bot(state, "app", Some(&made.id)).await.is_err());
        send_bot(state, "telegram:synthetic_coding_bot", None)
            .await
            .unwrap();
        assert!(projects(state).await.unwrap()[0].channels.is_empty());

        // `/project` shows, places by name (the owner only) and takes out.
        use crate::types::UserRole;
        let chat = "synthetic_coding_bot:42";
        assert!(
            project_command(state, chat, "", UserRole::Owner)
                .await
                .starts_with("This chat is in no project.")
        );
        assert!(
            project_command(state, chat, "acme", UserRole::Guest)
                .await
                .starts_with("Only the owner")
        );
        assert!(
            project_command(state, chat, "acme", UserRole::Owner)
                .await
                .starts_with("This chat works in Acme Corp now")
        );
        assert!(
            project_command(state, chat, "", UserRole::Owner)
                .await
                .starts_with("This chat works in Acme Corp.")
        );
        assert!(
            project_command(state, chat, "Zeta", UserRole::Owner)
                .await
                .starts_with("No project is called Zeta")
        );
        project_command(state, chat, "none", UserRole::Owner).await;
        assert!(state.session_work_projects().await.unwrap().is_empty());
    }

    #[test]
    fn a_missing_folder_is_made_only_inside_home() {
        let home = tempfile::tempdir().unwrap();
        let home_path = home.path();
        // Missing parents are made too.
        let site = home_path.join("projects").join("acme-site");
        make_folder(&site, home_path).unwrap();
        assert!(site.is_dir());
        // `..` and anything outside home are refused.
        assert!(make_folder(&home_path.join("projects/../../escape"), home_path).is_err());
        let elsewhere = tempfile::tempdir().unwrap();
        let outside = elsewhere.path().join("acme-site");
        assert!(make_folder(&outside, home_path).is_err());
        assert!(!outside.exists());
        // A path through a file is refused, and the file is left alone.
        let file = home_path.join("notes.md");
        std::fs::write(&file, "synthetic").unwrap();
        assert!(make_folder(&file.join("acme-site"), home_path).is_err());
        assert!(file.is_file());
    }
}

/// `/memory` in a chat: where the memory of the project this chat works in
/// is kept and what it holds. Read only.
pub(crate) struct ProjectMemoryCommand {
    pub state: std::sync::Arc<dyn StateStore>,
    /// The config file, for `[memory]` as it stands now.
    pub config_path: PathBuf,
}

/// The `[memory]` table of the config file, alone: nothing else is read,
/// so no secret is resolved for a read-only command.
fn project_notes_from(
    config_path: &Path,
) -> Option<std::sync::Arc<aidaemon_memory::project::ProjectNotes>> {
    #[derive(serde::Deserialize, Default)]
    struct MemoryOnly {
        #[serde(default)]
        memory: crate::config::MemoryConfig,
    }
    let memory = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|text| toml::from_str::<MemoryOnly>(&text).ok())
        .unwrap_or_default()
        .memory;
    memory.project_notes()
}

#[async_trait::async_trait]
impl aidaemon_channels::slash::SlashCommand for ProjectMemoryCommand {
    fn handles(&self, cmd: &str) -> bool {
        cmd == "/memory"
    }

    fn commands(&self) -> Vec<aidaemon_channels::slash::CommandDef> {
        vec![aidaemon_channels::slash::CommandDef {
            name: "memory",
            description: "Show the memory of the project this chat works in",
            usage: Some("/memory"),
            category: aidaemon_channels::slash::CommandCategory::Core,
        }]
    }

    async fn run(
        &self,
        _cmd: &str,
        _args: &str,
        call: &aidaemon_channels::slash::SlashCall<'_>,
    ) -> String {
        if call.user_role != crate::types::UserRole::Owner {
            return "Only the owner can see project memory.".to_string();
        }
        let Some(notes) = project_notes_from(&self.config_path) else {
            return "Project memory is off ([memory] project_memory = false).".to_string();
        };
        let Some(project) = aidaemon_runtime::turn::turn_project::resolve_turn_project(
            self.state.as_ref() as &dyn aidaemon_work::stores::WorkCoordinationStore,
            &[call.session_id],
            None,
        )
        .await
        else {
            return "This chat is in no project, so it keeps no project memory. \
                    Use /project <name> to work in one."
                .to_string();
        };
        project_memory_reply(&notes, &project)
    }
}

fn project_memory_reply(
    notes: &aidaemon_memory::project::ProjectNotes,
    project: &aidaemon_core::ProjectRef,
) -> String {
    let Some(dir) = notes.dir(project) else {
        return format!("{} has no memory directory.", project.name);
    };
    let entries = notes.entries(project);
    if entries.is_empty() {
        return format!(
            "{} keeps its memory in {}. Nothing is saved there yet: what this chat learns about \
             the project is kept there as it goes.",
            project.name,
            dir.display()
        );
    }
    let mut reply = format!(
        "{} keeps {} note{} in {}:",
        project.name,
        entries.len(),
        if entries.len() == 1 { "" } else { "s" },
        dir.display()
    );
    const SHOWN: usize = 30;
    for entry in entries.iter().take(SHOWN) {
        let kind = entry.kind.map(|k| k.as_str()).unwrap_or("note");
        reply.push_str(&format!("\n- {kind} {}: {}", entry.name, entry.description));
    }
    if entries.len() > SHOWN {
        reply.push_str(&format!(
            "\n…and {} more; MEMORY.md there lists them all.",
            entries.len() - SHOWN
        ));
    }
    reply.push_str("\nEdit or delete the files directly, or ask me to forget one.");
    reply
}

#[cfg(test)]
mod project_memory_command_tests {
    use super::*;

    #[test]
    fn the_reply_names_the_directory_and_lists_the_notes() {
        let dir = tempfile::tempdir().unwrap();
        let notes = aidaemon_memory::project::ProjectNotes::new(dir.path());
        let project = aidaemon_core::ProjectRef::new("project-a", "Acme Site", None).unwrap();
        let empty = project_memory_reply(&notes, &project);
        assert!(empty.contains("Nothing is saved there yet"), "{empty}");
        notes
            .write(
                &project,
                &aidaemon_memory::project::NoteWrite {
                    kind: aidaemon_memory::project::NoteKind::Fact,
                    key: "test_command".into(),
                    category: None,
                    description: None,
                    body: "Tests run with make test.".into(),
                    source: aidaemon_memory::project::NoteSource::Agent,
                },
            )
            .unwrap();
        let reply = project_memory_reply(&notes, &project);
        assert!(reply.starts_with("Acme Site keeps 1 note in "), "{reply}");
        assert!(
            reply.contains("- fact test_command: Tests run with make test."),
            "{reply}"
        );
    }

    #[test]
    fn the_config_switch_and_directory_are_read_alone() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(
            &config,
            "[memory]\nproject_memory = false\n[provider]\napi_key = \"keychain:x\"\n",
        )
        .unwrap();
        assert!(project_notes_from(&config).is_none());
        std::fs::write(
            &config,
            format!(
                "[memory]\nprojects_dir = '{}'\n",
                dir.path().join("p").display()
            ),
        )
        .unwrap();
        assert_eq!(
            project_notes_from(&config).unwrap().root(),
            dir.path().join("p")
        );
        assert!(project_notes_from(&dir.path().join("missing.toml")).is_some());
    }

    #[tokio::test]
    async fn losing_project_folder_clears_coding_agent_pin() {
        use aidaemon_runtime::session_model::{self, SessionModelSetting};
        use std::sync::Arc;
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap();
        let state: Arc<dyn StateStore> = Arc::new(
            crate::state::SqliteStateStore::for_tests(
                db_path,
                100,
                None,
                Arc::new(crate::memory::embeddings::EmbeddingService::deterministic()),
            )
            .await
            .unwrap(),
        );
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().display().to_string();

        // 1. Moving a conversation out of a folder-backed project clears its agent pin.
        let p1 = create(state.as_ref(), "Proj One", Some(&folder), false)
            .await
            .unwrap();
        place(state.as_ref(), "app:s1", Some(&p1.id)).await.unwrap();
        session_model::set(
            state.as_ref(),
            "app:s1",
            &SessionModelSetting {
                agent: Some("codex".into()),
                ..SessionModelSetting::default()
            },
        )
        .await
        .unwrap();
        place(state.as_ref(), "app:s1", None).await.unwrap();
        assert_eq!(
            session_model::get(state.as_ref(), "app:s1").await.agent,
            None
        );

        // 2. Deleting a project clears its sessions' agent pins.
        place(state.as_ref(), "app:s2", Some(&p1.id)).await.unwrap();
        session_model::set(
            state.as_ref(),
            "app:s2",
            &SessionModelSetting {
                agent: Some("claude".into()),
                ..SessionModelSetting::default()
            },
        )
        .await
        .unwrap();
        delete(state.as_ref(), &p1.id).await.unwrap();
        assert_eq!(
            session_model::get(state.as_ref(), "app:s2").await.agent,
            None
        );

        // 3. Clearing a project's folder via update clears its sessions' agent pins.
        let p2 = create(state.as_ref(), "Proj Two", Some(&folder), false)
            .await
            .unwrap();
        place(state.as_ref(), "app:s3", Some(&p2.id)).await.unwrap();
        session_model::set(
            state.as_ref(),
            "app:s3",
            &SessionModelSetting {
                agent: Some("codex".into()),
                ..SessionModelSetting::default()
            },
        )
        .await
        .unwrap();
        update(state.as_ref(), &p2.id, None, Some(""), false)
            .await
            .unwrap();
        assert_eq!(
            session_model::get(state.as_ref(), "app:s3").await.agent,
            None
        );
    }
}
