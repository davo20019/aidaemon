//! Which project a turn works in, from typed state only: the project its
//! conversation is placed in, else the project whose folder holds the
//! turn's bound workspace. Never from what a message says.

use std::path::{Path, PathBuf};

use aidaemon_core::ProjectRef;
use aidaemon_work::stores::WorkCoordinationStore;
use aidaemon_work::traits::{DEFAULT_PROJECT_ID, WorkProject};

/// The project a turn works in.
///
/// 1. The project `sessions` are placed in, the first that is (the turn's
///    own conversation, then the chat a sub-agent works for).
/// 2. Otherwise, when the turn is bound to a folder, the innermost project
///    whose folder holds it; a git worktree counts as its main checkout, so
///    a task's attempt worktree belongs to the project it was cut from.
pub async fn resolve_turn_project(
    projects: &dyn WorkCoordinationStore,
    sessions: &[&str],
    bound_workspace: Option<&str>,
) -> Option<ProjectRef> {
    let mut placed = None;
    for session in sessions.iter().filter(|s| !s.is_empty()) {
        match projects.get_session_work_project(session).await {
            Ok(id) if id != DEFAULT_PROJECT_ID => {
                placed = Some(id);
                break;
            }
            _ => {}
        }
    }
    if placed.is_none() && bound_workspace.is_none() {
        return None;
    }
    let all = projects.list_work_projects().await.ok()?;
    if let Some(id) = placed
        && let Some(project) = all.iter().find(|p| p.id == id)
    {
        return project_ref(project);
    }
    let folder = bound_workspace?;
    project_for_folder(&all, Path::new(folder)).and_then(project_ref)
}

/// Where conversations are placed, as memory recall asks
/// ([`aidaemon_memory::project::SessionProjects`]).
pub struct PlacedSessions<S: ?Sized>(pub std::sync::Arc<S>);

#[async_trait::async_trait]
impl<S> aidaemon_memory::project::SessionProjects for PlacedSessions<S>
where
    S: WorkCoordinationStore + ?Sized + 'static,
{
    async fn project_of(&self, session_id: &str) -> Option<String> {
        self.0
            .get_session_work_project(session_id)
            .await
            .ok()
            .filter(|id| id != DEFAULT_PROJECT_ID)
    }
}

fn project_ref(project: &WorkProject) -> Option<ProjectRef> {
    let root = project
        .path
        .as_deref()
        .map(PathBuf::from)
        .filter(|path| path.is_dir());
    ProjectRef::new(project.id.clone(), project.name.clone(), root)
}

/// The innermost project whose folder holds `folder` (or the main checkout
/// of the git worktree `folder` is in).
pub fn project_for_folder<'a>(
    projects: &'a [WorkProject],
    folder: &Path,
) -> Option<&'a WorkProject> {
    let folder = resolved(folder)?;
    innermost(projects, &folder).or_else(|| {
        let main = main_checkout(&folder)?;
        innermost(projects, &main)
    })
}

fn innermost<'a>(projects: &'a [WorkProject], folder: &Path) -> Option<&'a WorkProject> {
    projects
        .iter()
        .filter(|project| project.id != DEFAULT_PROJECT_ID)
        .filter_map(|project| {
            let root = resolved(Path::new(project.path.as_deref()?))?;
            folder
                .starts_with(&root)
                .then_some((root.components().count(), project))
        })
        .max_by_key(|(depth, _)| *depth)
        .map(|(_, project)| project)
}

fn resolved(path: &Path) -> Option<PathBuf> {
    path.is_absolute()
        .then(|| dunce::canonicalize(path).ok())
        .flatten()
}

/// For a folder inside a linked git worktree, the main checkout's root.
/// A linked worktree's `.git` is a file `gitdir: <main>/.git/worktrees/<name>`.
fn main_checkout(folder: &Path) -> Option<PathBuf> {
    let mut dir = Some(folder);
    while let Some(current) = dir {
        let dot_git = current.join(".git");
        if dot_git.is_dir() {
            // An ordinary checkout: not a linked worktree.
            return None;
        }
        if dot_git.is_file() {
            let text = std::fs::read_to_string(&dot_git).ok()?;
            let gitdir = text
                .lines()
                .find_map(|line| line.strip_prefix("gitdir:"))?
                .trim();
            let gitdir = if Path::new(gitdir).is_absolute() {
                PathBuf::from(gitdir)
            } else {
                current.join(gitdir)
            };
            let gitdir = dunce::canonicalize(gitdir).ok()?;
            // <main>/.git/worktrees/<name> → <main>
            let worktrees = gitdir.parent()?;
            if worktrees.file_name()? != "worktrees" {
                return None;
            }
            let git = worktrees.parent()?;
            if git.file_name()? != ".git" {
                return None;
            }
            return git.parent().map(Path::to_path_buf);
        }
        dir = current.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(id: &str, path: Option<&Path>) -> WorkProject {
        WorkProject {
            id: id.into(),
            name: format!("Project {id}"),
            description: None,
            path: path.map(|p| p.to_string_lossy().into_owned()),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn the_innermost_project_folder_wins() {
        let temp = tempfile::tempdir().unwrap();
        let outer = temp.path().join("work");
        let inner = outer.join("site");
        std::fs::create_dir_all(inner.join("src")).unwrap();
        let projects = vec![
            project(DEFAULT_PROJECT_ID, Some(temp.path())),
            project("project-outer", Some(&outer)),
            project("project-inner", Some(&inner)),
            project("project-none", None),
        ];
        assert_eq!(
            project_for_folder(&projects, &inner.join("src"))
                .unwrap()
                .id,
            "project-inner"
        );
        assert_eq!(
            project_for_folder(&projects, &outer).unwrap().id,
            "project-outer"
        );
        assert!(project_for_folder(&projects, temp.path()).is_none());
        assert!(project_for_folder(&projects, Path::new("relative/site")).is_none());
        // A sibling whose name starts like the project's is not inside it.
        let sibling = temp.path().join("work-old");
        std::fs::create_dir_all(&sibling).unwrap();
        assert!(project_for_folder(&projects, &sibling).is_none());
    }

    #[test]
    fn a_linked_worktree_belongs_to_its_main_checkout() {
        let temp = tempfile::tempdir().unwrap();
        let main = temp.path().join("repo");
        let gitdir = main.join(".git").join("worktrees").join("attempt-1");
        std::fs::create_dir_all(&gitdir).unwrap();
        let worktree = temp.path().join("elsewhere").join("attempt-1");
        std::fs::create_dir_all(worktree.join("src")).unwrap();
        std::fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", gitdir.display()),
        )
        .unwrap();
        let projects = vec![project("project-repo", Some(&main))];
        assert_eq!(
            project_for_folder(&projects, &worktree.join("src"))
                .unwrap()
                .id,
            "project-repo"
        );
        // A `.git` file that points anywhere else is not a worktree link.
        let odd = temp.path().join("odd");
        std::fs::create_dir_all(&odd).unwrap();
        std::fs::write(odd.join(".git"), format!("gitdir: {}\n", main.display())).unwrap();
        assert!(project_for_folder(&projects, &odd).is_none());
    }
}
