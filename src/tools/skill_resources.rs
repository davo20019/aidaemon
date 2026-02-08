use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::skills::{ResourceResolver, SharedSkillRegistry};
use crate::traits::Tool;

pub struct SkillResourcesTool {
    registry: SharedSkillRegistry,
    resolver: Arc<dyn ResourceResolver>,
}

impl SkillResourcesTool {
    pub fn new(registry: SharedSkillRegistry, resolver: Arc<dyn ResourceResolver>) -> Self {
        Self { registry, resolver }
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
                "required": ["skill_name", "action"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments)?;
        let skill_name = args["skill_name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing skill_name"))?;
        let action = args["action"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing action"))?;

        let skills = self.registry.snapshot().await;
        let skill = match skills.iter().find(|s| s.name == skill_name) {
            Some(s) => s,
            None => return Ok(format!("Skill '{}' not found.", skill_name)),
        };

        match action {
            "list" => {
                if skill.resources.is_empty() {
                    return Ok(format!(
                        "Skill '{}' has no bundled resources.",
                        skill_name
                    ));
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
            source: None,
            source_url: None,
            id: None,
            enabled: true,
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

    #[tokio::test]
    async fn test_list_resources() {
        let skill = make_skill_with_resources(
            "deploy",
            vec![
                ResourceEntry {
                    path: "scripts/deploy.sh".to_string(),
                    category: "script".to_string(),
                },
                ResourceEntry {
                    path: "references/guide.md".to_string(),
                    category: "reference".to_string(),
                },
            ],
        );
        let registry = SharedSkillRegistry::new(vec![skill]);
        let tool = SkillResourcesTool::new(registry, Arc::new(MockResolver));

        let result = tool
            .call(r#"{"skill_name": "deploy", "action": "list"}"#)
            .await
            .unwrap();
        assert!(result.contains("scripts/deploy.sh"));
        assert!(result.contains("references/guide.md"));
        assert!(result.contains("[script]"));
        assert!(result.contains("[reference]"));
    }

    #[tokio::test]
    async fn test_read_resource() {
        let skill = make_skill_with_resources(
            "deploy",
            vec![ResourceEntry {
                path: "references/guide.md".to_string(),
                category: "reference".to_string(),
            }],
        );
        let registry = SharedSkillRegistry::new(vec![skill]);
        let tool = SkillResourcesTool::new(registry, Arc::new(MockResolver));

        let result = tool
            .call(r#"{"skill_name": "deploy", "action": "read", "resource_path": "references/guide.md"}"#)
            .await
            .unwrap();
        assert_eq!(result, "Content of references/guide.md");
    }

    #[tokio::test]
    async fn test_read_nonexistent_resource() {
        let skill = make_skill_with_resources(
            "deploy",
            vec![ResourceEntry {
                path: "references/guide.md".to_string(),
                category: "reference".to_string(),
            }],
        );
        let registry = SharedSkillRegistry::new(vec![skill]);
        let tool = SkillResourcesTool::new(registry, Arc::new(MockResolver));

        let result = tool
            .call(r#"{"skill_name": "deploy", "action": "read", "resource_path": "missing.txt"}"#)
            .await
            .unwrap();
        assert!(result.contains("not found"));
        assert!(result.contains("action='list'"));
    }

    #[tokio::test]
    async fn test_no_resources_skill() {
        let skill = make_plain_skill("simple");
        let registry = SharedSkillRegistry::new(vec![skill]);
        let tool = SkillResourcesTool::new(registry, Arc::new(MockResolver));

        let result = tool
            .call(r#"{"skill_name": "simple", "action": "list"}"#)
            .await
            .unwrap();
        assert!(result.contains("no bundled resources"));
    }

    #[tokio::test]
    async fn test_unknown_action() {
        let skill = make_plain_skill("simple");
        let registry = SharedSkillRegistry::new(vec![skill]);
        let tool = SkillResourcesTool::new(registry, Arc::new(MockResolver));

        let result = tool
            .call(r#"{"skill_name": "simple", "action": "execute"}"#)
            .await
            .unwrap();
        assert!(result.contains("Unknown action"));
    }
}
