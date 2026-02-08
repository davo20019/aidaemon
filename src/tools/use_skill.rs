use async_trait::async_trait;
use serde_json::{json, Value};

use crate::skills::SharedSkillRegistry;
use crate::traits::Tool;

pub struct UseSkillTool {
    registry: SharedSkillRegistry,
}

impl UseSkillTool {
    pub fn new(registry: SharedSkillRegistry) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for UseSkillTool {
    fn name(&self) -> &str {
        "use_skill"
    }

    fn description(&self) -> &str {
        "Activate a skill by name to get its instructions"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "use_skill",
            "description": "Activate a skill by name to get its instructions. Use manage_skills list to see available skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the skill to activate"
                    }
                },
                "required": ["skill_name"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: Value = serde_json::from_str(arguments).unwrap_or(json!({}));
        let skill_name = args["skill_name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: skill_name"))?;

        let skills = self.registry.snapshot().await;
        if let Some(skill) = skills.iter().find(|s| s.name == skill_name) {
            Ok(skill.body.clone())
        } else {
            let available: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
            Ok(format!(
                "Skill '{}' not found. Available skills: {}",
                skill_name,
                available.join(", ")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::Skill;

    fn make_skill(name: &str, body: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{} skill", name),
            triggers: vec![],
            body: body.to_string(),
            source: None,
            source_url: None,
            id: None,
            enabled: true,
            dir_path: None,
            resources: vec![],
        }
    }

    #[tokio::test]
    async fn found_skill_returns_body() {
        let registry = SharedSkillRegistry::new(vec![make_skill("deploy", "Run deploy steps")]);
        let tool = UseSkillTool::new(registry);
        let result = tool.call(r#"{"skill_name": "deploy"}"#).await.unwrap();
        assert_eq!(result, "Run deploy steps");
    }

    #[tokio::test]
    async fn not_found_lists_available() {
        let registry = SharedSkillRegistry::new(vec![
            make_skill("deploy", "body1"),
            make_skill("lint", "body2"),
        ]);
        let tool = UseSkillTool::new(registry);
        let result = tool.call(r#"{"skill_name": "missing"}"#).await.unwrap();
        assert!(result.contains("not found"));
        assert!(result.contains("deploy"));
        assert!(result.contains("lint"));
    }

    #[tokio::test]
    async fn empty_skills() {
        let registry = SharedSkillRegistry::new(vec![]);
        let tool = UseSkillTool::new(registry);
        let result = tool.call(r#"{"skill_name": "any"}"#).await.unwrap();
        assert!(result.contains("not found"));
    }

    #[tokio::test]
    async fn disabled_skill_not_found() {
        let mut skill = make_skill("deploy", "Run deploy steps");
        skill.enabled = false;
        let registry = SharedSkillRegistry::new(vec![skill]);
        let tool = UseSkillTool::new(registry);
        let result = tool.call(r#"{"skill_name": "deploy"}"#).await.unwrap();
        assert!(result.contains("not found"));
    }
}
