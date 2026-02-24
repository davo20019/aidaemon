use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::skills::{self, ResourceResolver};
use crate::traits::{Tool, ToolCapabilities};

pub struct SkillResourcesTool {
    skills_dir: PathBuf,
    resolver: Arc<dyn ResourceResolver>,
}

impl SkillResourcesTool {
    pub fn new(skills_dir: PathBuf, resolver: Arc<dyn ResourceResolver>) -> Self {
        Self {
            skills_dir,
            resolver,
        }
    }
}

#[async_trait]
impl Tool for SkillResourcesTool {
    fn name(&self) -> &str {
        "skill_resources"
    }

    fn description(&self) -> &str {
        "Load resources (scripts, references, assets) from a skill on demand"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "skill_resources",
            "description": "Access resources bundled with a skill. Use 'list' to see available resources, 'read' to load a specific file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": { "type": "string", "description": "Name of the skill" },
                    "action": {
                        "type": "string",
                        "enum": ["list", "read"],
                        "description": "Action: 'list' available resources or 'read' a specific resource"
                    },
                    "resource_path": {
                        "type": "string",
                        "description": "Relative path of resource to read (e.g. 'references/guide.md'). Required for 'read' action."
                    }
                },
                "required": ["skill_name", "action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let skill_name = args["skill_name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing skill_name"))?;
        let action = args["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing action"))?;

        let all_skills = skills::load_skills(&self.skills_dir);
        let skill = match skills::find_skill_by_name(&all_skills, skill_name) {
            Some(s) => s,
            None => return Ok(format!("Skill '{}' not found.", skill_name)),
        };

        if let Some(ref dir_path) = skill.dir_path {
            if let Err(e) = self
                .resolver
                .register_skill_directory(&skill.name, dir_path)
                .await
            {
                return Ok(format!("Error registering skill resources: {}", e));
            }
        }

        match action {
            "list" => {
                if skill.resources.is_empty() {
                    return Ok(format!("Skill '{}' has no bundled resources.", skill_name));
                }
                let mut out = format!("Resources for '{}':\n", skill_name);
                for entry in &skill.resources {
                    out.push_str(&format!("  [{}] {}\n", entry.category, entry.path));
                }
                Ok(out)
            }
            "read" => {
                let resource_path = args["resource_path"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("'read' action requires resource_path"))?;

                // Validate the resource exists in the skill's declared resources
                if !skill.resources.iter().any(|r| r.path == resource_path) {
                    return Ok(format!(
                        "Resource '{}' not found in skill '{}'. Use action='list' to see available resources.",
                        resource_path, skill_name
                    ));
                }

                // Delegate to the resolver for actual I/O
                match self.resolver.read_resource(skill_name, resource_path).await {
                    Ok(content) => Ok(content),
                    Err(e) => Ok(format!("Error reading resource: {}", e)),
                }
            }
            _ => Ok(format!(
                "Unknown action '{}'. Use 'list' or 'read'.",
                action
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::resources::ResourceEntry;
    use crate::skills::Skill;

    fn make_skill_with_resources(name: &str, resources: Vec<ResourceEntry>) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{} skill", name),
            triggers: vec![],
            body: "Do things.".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources,
        }
    }

    fn make_plain_skill(name: &str) -> Skill {
        make_skill_with_resources(name, vec![])
    }

    /// A test resolver that returns canned content.
    struct MockResolver;

    #[async_trait]
    impl ResourceResolver for MockResolver {
        async fn read_resource(
            &self,
            _skill_name: &str,
            resource_path: &str,
        ) -> anyhow::Result<String> {
            Ok(format!("Content of {}", resource_path))
        }
    }

    /// Create a directory-based skill with resources on disk
    fn setup_dir_skill(
        parent: &std::path::Path,
        name: &str,
        resources: &[(&str, &str, &str)], // (subdir, filename, content)
    ) {
        let skill_dir = parent.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let skill = Skill {
            name: name.to_string(),
            description: format!("{} skill", name),
            triggers: vec![],
            body: "Do things.".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: Some(skill_dir.clone()),
            resources: vec![],
        };
        std::fs::write(skill_dir.join("SKILL.md"), skill.to_markdown()).unwrap();
        for (subdir, filename, content) in resources {
            let subdir_path = skill_dir.join(subdir);
            std::fs::create_dir_all(&subdir_path).unwrap();
            std::fs::write(subdir_path.join(filename), content).unwrap();
        }
    }

    #[tokio::test]
    async fn test_list_resources() {
        let dir = tempfile::TempDir::new().unwrap();
        setup_dir_skill(
            dir.path(),
            "deploy",
            &[
                ("scripts", "deploy.sh", "#!/bin/bash"),
                ("references", "guide.md", "# Guide"),
            ],
        );
        let tool = SkillResourcesTool::new(dir.path().to_path_buf(), Arc::new(MockResolver));
        let result = tool
            .call(r#"{"skill_name": "deploy", "action": "list"}"#)
            .await
            .unwrap();
        assert!(result.contains("scripts/deploy.sh"));
        assert!(result.contains("references/guide.md"));
    }

    #[tokio::test]
    async fn test_read_resource() {
        let dir = tempfile::TempDir::new().unwrap();
        setup_dir_skill(
            dir.path(),
            "deploy",
            &[("references", "guide.md", "# Guide")],
        );
        let tool = SkillResourcesTool::new(dir.path().to_path_buf(), Arc::new(MockResolver));
        let result = tool
            .call(r#"{"skill_name": "deploy", "action": "read", "resource_path": "references/guide.md"}"#)
            .await
            .unwrap();
        assert_eq!(result, "Content of references/guide.md");
    }

    #[tokio::test]
    async fn test_read_nonexistent_resource() {
        let dir = tempfile::TempDir::new().unwrap();
        setup_dir_skill(
            dir.path(),
            "deploy",
            &[("references", "guide.md", "# Guide")],
        );
        let tool = SkillResourcesTool::new(dir.path().to_path_buf(), Arc::new(MockResolver));
        let result = tool
            .call(r#"{"skill_name": "deploy", "action": "read", "resource_path": "missing.txt"}"#)
            .await
            .unwrap();
        assert!(result.contains("not found"));
        assert!(result.contains("action='list'"));
    }

    #[tokio::test]
    async fn test_runtime_directory_skill_auto_registers_with_filesystem_resolver() {
        let dir = tempfile::TempDir::new().unwrap();
        setup_dir_skill(
            dir.path(),
            "deploy",
            &[("references", "guide.md", "# Runtime Guide")],
        );

        // Intentionally do NOT pre-register the directory in resolver.
        let resolver = Arc::new(crate::skills::FileSystemResolver::new());
        let tool = SkillResourcesTool::new(dir.path().to_path_buf(), resolver);
        let result = tool
            .call(
                r#"{"skill_name": "deploy", "action": "read", "resource_path": "references/guide.md"}"#,
            )
            .await
            .unwrap();
        assert!(result.contains("Runtime Guide"));
    }

    #[tokio::test]
    async fn test_no_resources_skill() {
        let dir = tempfile::TempDir::new().unwrap();
        skills::write_skill_to_file(dir.path(), &make_plain_skill("simple")).unwrap();
        let tool = SkillResourcesTool::new(dir.path().to_path_buf(), Arc::new(MockResolver));
        let result = tool
            .call(r#"{"skill_name": "simple", "action": "list"}"#)
            .await
            .unwrap();
        assert!(result.contains("no bundled resources"));
    }

    #[tokio::test]
    async fn test_unknown_action() {
        let dir = tempfile::TempDir::new().unwrap();
        skills::write_skill_to_file(dir.path(), &make_plain_skill("simple")).unwrap();
        let tool = SkillResourcesTool::new(dir.path().to_path_buf(), Arc::new(MockResolver));
        let result = tool
            .call(r#"{"skill_name": "simple", "action": "execute"}"#)
            .await
            .unwrap();
        assert!(result.contains("Unknown action"));
    }
}
