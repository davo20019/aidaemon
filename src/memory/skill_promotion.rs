//! Automatic skill promotion from learned procedures.
//!
//! When a procedure has been successfully used many times (configurable threshold),
//! it can be automatically promoted to a dynamic skill. This lets the agent
//! turn repeated successful patterns into reusable, named skills.

use std::sync::Arc;

use serde_json::json;
use tracing::{info, warn};

use crate::skills::{SharedSkillRegistry, Skill};
use crate::traits::{DynamicSkill, ModelProvider, Procedure, StateStore};

/// Minimum number of successes before a procedure is eligible for promotion.
const DEFAULT_MIN_SUCCESS: i32 = 5;
/// Minimum success rate (0.0 - 1.0) for promotion eligibility.
const DEFAULT_MIN_RATE: f32 = 0.8;

pub struct SkillPromoter {
    state: Arc<dyn StateStore>,
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    registry: SharedSkillRegistry,
}

impl SkillPromoter {
    pub fn new(
        state: Arc<dyn StateStore>,
        provider: Arc<dyn ModelProvider>,
        fast_model: String,
        registry: SharedSkillRegistry,
    ) -> Self {
        Self {
            state,
            provider,
            fast_model,
            registry,
        }
    }

    /// Run a full promotion cycle: find eligible procedures, generate skills, persist and register.
    pub async fn run_promotion_cycle(&self) -> anyhow::Result<usize> {
        let promotable = self.check_promotable_procedures().await?;
        if promotable.is_empty() {
            return Ok(0);
        }

        let existing_skills = self.registry.snapshot_all().await;
        let existing_names: Vec<&str> = existing_skills.iter().map(|s| s.name.as_str()).collect();

        let mut promoted = 0;
        for procedure in &promotable {
            // Skip if a skill with this name already exists
            let candidate_name = procedure.name.to_lowercase().replace(' ', "-");
            if existing_names.iter().any(|n| n.to_lowercase() == candidate_name) {
                continue;
            }

            match self.promote_procedure(procedure).await {
                Ok(Some(skill)) => {
                    // Persist to database
                    let triggers_json = serde_json::to_string(&skill.triggers)?;
                    let dynamic = DynamicSkill {
                        id: 0,
                        name: skill.name.clone(),
                        description: skill.description.clone(),
                        triggers_json,
                        body: skill.body.clone(),
                        source: "auto".to_string(),
                        source_url: None,
                        enabled: true,
                        version: None,
                        created_at: String::new(),
                    };
                    let db_id = self.state.add_dynamic_skill(&dynamic).await?;

                    // Register
                    let mut registered = skill;
                    registered.id = Some(db_id);
                    registered.source = Some("auto".to_string());
                    info!(
                        name = %registered.name,
                        procedure = %procedure.name,
                        success_count = procedure.success_count,
                        "Auto-promoted procedure to skill"
                    );
                    self.registry.add(registered).await;
                    promoted += 1;
                }
                Ok(None) => {
                    // LLM decided not to promote
                }
                Err(e) => {
                    warn!(
                        procedure = %procedure.name,
                        error = %e,
                        "Failed to promote procedure to skill"
                    );
                }
            }
        }

        Ok(promoted)
    }

    /// Query for procedures that meet the promotion threshold.
    async fn check_promotable_procedures(&self) -> anyhow::Result<Vec<Procedure>> {
        self.state
            .get_promotable_procedures(DEFAULT_MIN_SUCCESS, DEFAULT_MIN_RATE)
            .await
    }

    /// Use a fast LLM to generate a proper skill markdown from procedure data.
    /// Returns None if the LLM decides the procedure isn't suitable for a skill.
    async fn promote_procedure(&self, procedure: &Procedure) -> anyhow::Result<Option<Skill>> {
        let steps_str = procedure
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n");

        let success_rate = if procedure.success_count + procedure.failure_count > 0 {
            procedure.success_count as f32
                / (procedure.success_count + procedure.failure_count) as f32
                * 100.0
        } else {
            0.0
        };

        let prompt = format!(
            "Convert this learned procedure into a skill definition. \
             Return ONLY the skill markdown (with --- frontmatter and body) or 'SKIP' if this \
             procedure is too specific or context-dependent to be a reusable skill.\n\n\
             Procedure name: {}\n\
             Trigger pattern: {}\n\
             Success rate: {:.0}% ({} successes, {} failures)\n\
             Steps:\n{}\n\n\
             The skill markdown format is:\n\
             ---\n\
             name: skill-name-in-kebab-case\n\
             description: Brief description of what this skill does\n\
             triggers: keyword1, keyword2, keyword3\n\
             ---\n\
             Step-by-step instructions for the AI to follow when this skill is activated.\n\n\
             Make the triggers broad enough to match relevant user messages but specific enough \
             to avoid false positives. The body should be clear, actionable instructions.",
            procedure.name,
            procedure.trigger_pattern,
            success_rate,
            procedure.success_count,
            procedure.failure_count,
            steps_str
        );

        let messages = vec![
            json!({"role": "system", "content": "You are a skill generator. Convert proven procedures into reusable skill definitions. Respond with ONLY the skill markdown or SKIP."}),
            json!({"role": "user", "content": prompt}),
        ];

        let response = self.provider.chat(&self.fast_model, &messages, &[]).await?;
        let text = response
            .content
            .ok_or_else(|| anyhow::anyhow!("Empty response from skill generation LLM"))?;

        let trimmed = text.trim();
        if trimmed == "SKIP" || trimmed.to_uppercase() == "SKIP" {
            return Ok(None);
        }

        // Parse the generated markdown
        match Skill::parse(trimmed) {
            Some(mut skill) => {
                skill.source = Some("auto".to_string());
                skill.enabled = true;
                Ok(Some(skill))
            }
            None => {
                warn!(
                    procedure = %procedure.name,
                    "LLM generated invalid skill markdown"
                );
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_constants() {
        assert_eq!(DEFAULT_MIN_SUCCESS, 5);
        assert!(DEFAULT_MIN_RATE > 0.0 && DEFAULT_MIN_RATE <= 1.0);
    }

    #[test]
    fn generated_skill_parses() {
        let markdown = "---\nname: deploy-app\ndescription: Deploy the application\ntriggers: deploy, ship, release\n---\n1. Run cargo build --release\n2. Copy binary to server\n3. Restart service";
        let skill = Skill::parse(markdown).unwrap();
        assert_eq!(skill.name, "deploy-app");
        assert_eq!(skill.triggers.len(), 3);
        assert!(skill.body.contains("cargo build"));
    }

    #[test]
    fn skip_response_handled() {
        // Simulating SKIP response - just verify parsing returns None
        assert!(Skill::parse("SKIP").is_none());
    }
}
