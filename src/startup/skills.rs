use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::mpsc;
use tracing::info;

use crate::config::AppConfig;
use crate::state::SqliteStateStore;
use crate::tools::terminal::ApprovalRequest;
use crate::tools::{ManageSkillsTool, SkillResourcesTool, UseSkillTool};
use crate::traits::{StateStore, Tool};

pub async fn register_skills_tools(
    config: &AppConfig,
    config_path: &PathBuf,
    state: Arc<SqliteStateStore>,
    tools: &mut Vec<Arc<dyn Tool>>,
    approval_tx: mpsc::Sender<ApprovalRequest>,
) -> anyhow::Result<Option<PathBuf>> {
    if !config.skills.enabled {
        info!("Skills system disabled");
        return Ok(None);
    }

    let dir = config_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(&config.skills.dir);
    std::fs::create_dir_all(&dir).ok();

    if state
        .get_setting("skill_migration_v1_done")
        .await?
        .is_none()
    {
        match state.get_dynamic_skills().await {
            Ok(dynamic_skills) => {
                let existing = crate::skills::load_skills(&dir);
                let existing_names: Vec<&str> = existing.iter().map(|s| s.name.as_str()).collect();
                let mut migrated = 0;
                for ds in &dynamic_skills {
                    if existing_names.iter().any(|n| *n == ds.name) {
                        info!(name = %ds.name, "Skipping migration — skill already exists on disk");
                        continue;
                    }
                    let triggers: Vec<String> =
                        serde_json::from_str(&ds.triggers_json).unwrap_or_default();
                    let skill = crate::skills::Skill {
                        name: ds.name.clone(),
                        description: ds.description.clone(),
                        triggers,
                        body: ds.body.clone(),
                        source: Some(ds.source.clone()),
                        source_url: ds.source_url.clone(),
                        dir_path: None,
                        resources: vec![],
                    };
                    match crate::skills::write_skill_to_file(&dir, &skill) {
                        Ok(path) => {
                            info!(name = %ds.name, path = %path.display(), "Migrated dynamic skill to filesystem");
                            migrated += 1;
                        }
                        Err(e) => {
                            tracing::warn!(name = %ds.name, error = %e, "Failed to migrate dynamic skill");
                        }
                    }
                }
                if migrated > 0 {
                    info!(count = migrated, "Dynamic skills migrated to filesystem");
                }
            }
            Err(e) => {
                tracing::warn!("Failed to load dynamic skills for migration: {}", e);
            }
        }
        state.set_setting("skill_migration_v1_done", "true").await?;
    }

    let startup_skills = crate::skills::load_skills(&dir);
    info!(
        count = startup_skills.len(),
        dir = %dir.display(),
        "Filesystem skills loaded"
    );

    let fs_resolver = Arc::new(crate::skills::FileSystemResolver::new());
    for skill in &startup_skills {
        if let Some(ref dir_path) = skill.dir_path {
            fs_resolver.register(&skill.name, dir_path.clone()).await;
        }
    }

    tools.push(Arc::new(UseSkillTool::new(dir.clone())));
    info!("use_skill tool enabled");

    tools.push(Arc::new(SkillResourcesTool::new(
        dir.clone(),
        fs_resolver as Arc<dyn crate::skills::ResourceResolver>,
    )));
    info!("skill_resources tool enabled");

    let manage_skills = ManageSkillsTool::new(dir.clone(), state, approval_tx)
        .with_registries(config.skills.registries.clone());
    tools.push(Arc::new(manage_skills));
    info!("manage_skills tool enabled");

    Ok(Some(dir))
}
