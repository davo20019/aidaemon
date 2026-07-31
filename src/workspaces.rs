use std::time::Duration;

use crate::execution::{active_execution_backend, ExecutionRequest};
use crate::traits::{HandoffArtifact, StateStore, TaskAttempt, TaskWorkspace};

#[derive(Debug, Default)]
pub(crate) struct WorkspaceEvidence {
    pub artifacts: Vec<HandoffArtifact>,
    pub verification: Vec<String>,
}

fn short_component(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(12)
        .collect()
}

fn task_needs_container_scope(description: &str) -> bool {
    let lower = description.to_ascii_lowercase();
    let explicitly_cross_project = [
        "all my projects",
        "all projects",
        "across my projects",
        "across all projects",
    ]
    .iter()
    .any(|phrase| lower.contains(phrase));
    let container_named = lower.contains("projects workspace")
        || lower.contains("projects folder")
        || lower.contains("projects directory");
    let words = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let inspects_container = container_named
        && ["inspect", "list", "scan", "search", "compare", "audit"]
            .iter()
            .any(|verb| words.contains(verb));

    // Cross-project work needs the container for discovery even when later
    // commands mutate individual children. Checkpointing still requires each
    // mutation to resolve to one bounded child project.
    explicitly_cross_project || inspects_container
}

fn safe_slug_component(value: &str) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            previous_dash = false;
        } else if !previous_dash && !slug.is_empty() {
            slug.push('-');
            previous_dash = true;
        }
        if slug.len() >= 48 {
            break;
        }
    }
    slug.trim_matches('-').to_string()
}

fn isolated_workspace_slug(description: &str, task_id: &str) -> String {
    let lower = description.to_ascii_lowercase();
    let words = lower
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let mut topic = Vec::new();
    if let Some(index) = words.iter().position(|word| *word == "about") {
        for word in words.iter().skip(index + 1).take(4) {
            if matches!(*word, "and" | "then" | "with" | "in" | "under" | "for") {
                break;
            }
            topic.push(*word);
        }
    }
    let base = if topic.is_empty() {
        "task".to_string()
    } else {
        let topic = safe_slug_component(&topic.join("-"));
        if lower.contains("website") || lower.contains("web site") {
            format!("{topic}-site")
        } else {
            topic
        }
    };
    format!("{}-{}", base, short_component(task_id))
}

async fn git_output(cwd: &str, args: Vec<String>, timeout: Duration) -> anyhow::Result<String> {
    let backend = active_execution_backend();
    let mut full_args = vec!["-C".to_string(), cwd.to_string()];
    full_args.extend(args);
    let output = backend
        .execute(ExecutionRequest::argv("git", full_args), timeout)
        .await?;
    anyhow::ensure!(
        !output.timed_out,
        "git command timed out while preparing task workspace"
    );
    anyhow::ensure!(
        output.exit_code == 0,
        "git command failed: {}",
        output.stderr_lossy().trim()
    );
    Ok(output.stdout_lossy().trim().to_string())
}

pub(crate) async fn provision_task_workspace(
    state: &dyn StateStore,
    task_id: &str,
    attempt: &TaskAttempt,
    project_scope: Option<&str>,
) -> anyhow::Result<TaskWorkspace> {
    if let Some(existing) = state.get_task_workspace(task_id).await? {
        if existing.attempt_id == attempt.id && existing.status != "failed" {
            return Ok(existing);
        }
    }

    let backend = active_execution_backend();
    let requested_policy = state.get_task_workspace_policy(task_id).await?;
    let task_description = state
        .get_task(task_id)
        .await?
        .map(|task| task.description)
        .unwrap_or_default();
    let now = chrono::Utc::now().to_rfc3339();
    if matches!(requested_policy.as_str(), "shared" | "isolated") {
        let requested_root = project_scope.unwrap_or_else(|| backend.workspace_root().as_str());
        let requested_root = backend.resolve_path(requested_root).await?;
        let workspace_container = backend.workspace_root();
        let at_workspace_container = requested_root.as_str() == workspace_container.as_str();
        let auto_isolate = requested_policy == "isolated"
            || (at_workspace_container && !task_needs_container_scope(&task_description));
        let resolved_root = if auto_isolate && at_workspace_container {
            workspace_container.join(isolated_workspace_slug(&task_description, task_id))
        } else {
            requested_root
        };
        backend.create_dir_all(&resolved_root).await?;
        // Validate the created directory through the backend's canonical
        // boundary check, but retain the caller-facing spelling in handoffs.
        // macOS aliases `/tmp` to `/private/tmp`; rewriting that path makes
        // otherwise stable task scopes appear to change between attempts.
        backend.canonicalize(&resolved_root).await?;
        let root_path = resolved_root.to_string();
        let workspace = TaskWorkspace {
            id: uuid::Uuid::new_v4().to_string(),
            task_id: task_id.to_string(),
            attempt_id: attempt.id.clone(),
            backend_id: backend.id().to_string(),
            policy: if auto_isolate {
                "isolated".to_string()
            } else {
                "shared".to_string()
            },
            root_path,
            branch_name: None,
            base_ref: None,
            head_ref: None,
            status: "active".to_string(),
            created_at: now,
            released_at: None,
        };
        state.create_task_workspace(&workspace).await?;
        return Ok(workspace);
    }

    anyhow::ensure!(
        requested_policy == "worktree",
        "unsupported task workspace policy"
    );
    let scope = project_scope
        .ok_or_else(|| anyhow::anyhow!("worktree policy requires an explicit project scope"))?;
    let resolved_scope = backend.resolve_path(scope).await?;
    let canonical_scope = backend.canonicalize(&resolved_scope).await?;
    let git_root = git_output(
        canonical_scope.as_str(),
        vec!["rev-parse".to_string(), "--show-toplevel".to_string()],
        Duration::from_secs(20),
    )
    .await?;
    let base_ref = git_output(
        &git_root,
        vec!["rev-parse".to_string(), "HEAD".to_string()],
        Duration::from_secs(20),
    )
    .await?;
    let suffix = format!(
        "{}-{}",
        short_component(task_id),
        short_component(&attempt.id)
    );
    let branch_name = format!("aidaemon/task-{suffix}");
    let parent = backend.workspace_root().join(".aidaemon-workspaces");
    backend.create_dir_all(&parent).await?;
    let root = parent.join(&suffix);
    let mut workspace = TaskWorkspace {
        id: uuid::Uuid::new_v4().to_string(),
        task_id: task_id.to_string(),
        attempt_id: attempt.id.clone(),
        backend_id: backend.id().to_string(),
        policy: requested_policy,
        root_path: root.as_str().to_string(),
        branch_name: Some(branch_name.clone()),
        base_ref: Some(base_ref.clone()),
        head_ref: None,
        status: "active".to_string(),
        created_at: now,
        released_at: None,
    };
    let create = git_output(
        &git_root,
        vec![
            "worktree".to_string(),
            "add".to_string(),
            "-b".to_string(),
            branch_name,
            root.as_str().to_string(),
            base_ref,
        ],
        Duration::from_secs(60),
    )
    .await;
    if let Err(error) = create {
        workspace.status = "failed".to_string();
        state.create_task_workspace(&workspace).await?;
        return Err(error);
    }
    state.create_task_workspace(&workspace).await?;
    Ok(workspace)
}

pub(crate) async fn preserve_task_workspace(
    state: &dyn StateStore,
    task_id: &str,
) -> anyhow::Result<WorkspaceEvidence> {
    let Some(mut workspace) = state.get_task_workspace(task_id).await? else {
        return Ok(WorkspaceEvidence::default());
    };
    if workspace.status != "active" {
        return Ok(WorkspaceEvidence::default());
    }

    let mut evidence = WorkspaceEvidence {
        artifacts: vec![HandoffArtifact {
            kind: "path".to_string(),
            reference: workspace.root_path.clone(),
            digest: None,
            metadata: Some(format!("workspace_policy={}", workspace.policy)),
        }],
        verification: Vec::new(),
    };
    if workspace.policy == "worktree" {
        match git_output(
            &workspace.root_path,
            vec!["rev-parse".to_string(), "HEAD".to_string()],
            Duration::from_secs(20),
        )
        .await
        {
            Ok(head) => {
                workspace.head_ref = Some(head.clone());
                evidence.artifacts.push(HandoffArtifact {
                    kind: "commit".to_string(),
                    reference: head,
                    digest: None,
                    metadata: workspace.branch_name.clone(),
                });
            }
            Err(error) => evidence
                .verification
                .push(format!("Could not read workspace HEAD: {error}")),
        }
        match git_output(
            &workspace.root_path,
            vec!["status".to_string(), "--porcelain".to_string()],
            Duration::from_secs(20),
        )
        .await
        {
            Ok(status) if status.is_empty() => evidence
                .verification
                .push("Workspace has no uncommitted changes.".to_string()),
            Ok(status) => evidence.verification.push(format!(
                "Workspace preserved with {} uncommitted path(s).",
                status.lines().count()
            )),
            Err(error) => evidence
                .verification
                .push(format!("Could not inspect workspace status: {error}")),
        }
    } else {
        evidence
            .verification
            .push(format!("{} workspace path recorded.", workspace.policy));
    }
    workspace.status = "preserved".to_string();
    state.update_task_workspace(&workspace).await?;
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::{isolated_workspace_slug, short_component, task_needs_container_scope};

    #[test]
    fn workspace_components_are_bounded_and_safe() {
        assert_eq!(short_component("abc-123_DEF!xyz"), "abc123DEFxyz");
        assert_eq!(short_component("123456789012345"), "123456789012");
    }

    #[test]
    fn new_project_workspace_gets_bounded_human_readable_slug() {
        assert_eq!(
            isolated_workspace_slug(
                "Create a website about AI engineering and deploy it",
                "43b2cf48-bc97"
            ),
            "ai-engineering-site-43b2cf48bc97"
        );
        assert!(!task_needs_container_scope(
            "Create a website in the projects folder"
        ));
    }

    #[test]
    fn cross_project_discovery_keeps_container_scope() {
        assert!(task_needs_container_scope(
            "Inspect the projects workspace and list all projects"
        ));
        assert!(task_needs_container_scope(
            "Inspect all projects and update their dependencies"
        ));
        assert!(!task_needs_container_scope(
            "Create a website in the projects folder"
        ));
    }
}
