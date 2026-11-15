//! Projects for the app (protocol 1.10): the work-project store's projects,
//! each with the folder its conversations work in. A folder can be anything
//! on this computer (a repository, a site, a folder of designs); it must
//! exist and is stored as an absolute, resolved path. A chat bot can send
//! its direct-message chats to a project (protocol 1.12).

use std::path::{Path, PathBuf};

use aidaemon_client_protocol::ProjectSummary;
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

pub(crate) async fn create(
    state: &dyn StateStore,
    name: &str,
    path: Option<&str>,
) -> Result<ProjectSummary, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("A project needs a name".to_string());
    }
    let folder = match path.map(str::trim).filter(|p| !p.is_empty()) {
        Some(path) => Some(folder(path)?),
        None => None,
    };
    let taken = state
        .list_work_projects()
        .await
        .map_err(|e| e.to_string())?
        .iter()
        .any(|project| project.name.eq_ignore_ascii_case(name));
    if taken {
        return Err(format!("There is already a project called {name}"));
    }
    let project = state
        .create_work_project(name, None)
        .await
        .map_err(|e| e.to_string())?;
    match folder {
        Some(folder) => state
            .update_work_project(&project.id, None, Some(Some(&folder)))
            .await
            .map_err(|e| e.to_string())?
            .map(|project| summary(project, &[]))
            .ok_or_else(|| "The project was made but its folder was not saved".to_string()),
        None => Ok(summary(project, &[])),
    }
}

pub(crate) async fn update(
    state: &dyn StateStore,
    id: &str,
    name: Option<&str>,
    path: Option<&str>,
) -> Result<ProjectSummary, String> {
    if id == DEFAULT_PROJECT_ID {
        return Err("Conversations in no project cannot be renamed".to_string());
    }
    let path = match path.map(str::trim) {
        Some("") => Some(None),
        Some(path) => Some(Some(folder(path)?)),
        None => None,
    };
    let bots = state
        .channel_work_projects()
        .await
        .map_err(|e| e.to_string())?;
    state
        .update_work_project(id, name, path.as_ref().map(|p| p.as_deref()))
        .await
        .map_err(|e| e.to_string())?
        .map(|project| summary(project, &bots))
        .ok_or_else(|| "There is no such project".to_string())
}

pub(crate) async fn place(
    state: &dyn StateStore,
    session_id: &str,
    project_id: Option<&str>,
) -> Result<(), String> {
    let project_id = project_id.unwrap_or(DEFAULT_PROJECT_ID);
    match state.set_session_work_project(session_id, project_id).await {
        Ok(true) => Ok(()),
        Ok(false) => Err("There is no such project".to_string()),
        Err(e) => Err(e.to_string()),
    }
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
fn folder(path: &str) -> Result<String, String> {
    let expanded: PathBuf = match path.strip_prefix("~/").or(path.strip_prefix("~")) {
        Some(rest) => dirs::home_dir()
            .ok_or_else(|| "This computer has no home folder to start from".to_string())?
            .join(rest.trim_start_matches('/')),
        None => PathBuf::from(path),
    };
    if !expanded.is_absolute() {
        return Err(format!(
            "Give the folder's full path (for example ~/projects/{}), not a relative one",
            Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "acme-site".to_string())
        ));
    }
    let resolved = dunce::canonicalize(&expanded).map_err(|_| {
        format!(
            "There is no folder at {} on this computer",
            expanded.display()
        )
    })?;
    if !resolved.is_dir() {
        return Err(format!("{} is a file, not a folder", resolved.display()));
    }
    Ok(resolved.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_project_is_made_with_an_existing_folder_and_holds_sessions() {
        let harness = crate::testing::setup_test_agent(crate::testing::MockProvider::new())
            .await
            .unwrap();
        let state: &dyn StateStore = harness.state.as_ref();
        let site = tempfile::tempdir().unwrap();
        let made = create(state, "Acme site", Some(&site.path().to_string_lossy()))
            .await
            .unwrap();
        assert_eq!(
            made.path.as_deref(),
            Some(dunce::canonicalize(site.path()).unwrap().to_str().unwrap())
        );
        // The same name again, a missing folder and a relative one are refused.
        assert!(create(state, "acme SITE", None).await.is_err());
        assert!(
            create(
                state,
                "Gone",
                Some(&crate::testing::abs_path("/nowhere/synthetic-missing"))
            )
            .await
            .unwrap_err()
            .contains("no folder")
        );
        assert!(create(state, "Rel", Some("projects/acme")).await.is_err());
        // Listed without the default project.
        let listed = projects(state).await.unwrap();
        assert_eq!(listed, vec![made.clone()]);
        // A folder can be removed and the project renamed.
        let bare = update(state, &made.id, Some("Acme Corp"), Some(""))
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
}
