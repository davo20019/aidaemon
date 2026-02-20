//! Automatic skill promotion from learned procedures.
//!
//! When a procedure has been successfully used many times (configurable threshold),
//! it can be automatically promoted to a skill draft pending user review.
//! Drafts are stored in the `skill_drafts` table and can be approved or dismissed
//! via the `manage_skills review` action.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use tracing::{info, warn};

use crate::llm_runtime::SharedLlmRuntime;
use crate::skills::{self, Skill};
use crate::traits::{Procedure, SkillDraft, StateStore};

/// Minimum number of successes before a procedure is eligible for promotion.
const DEFAULT_MIN_SUCCESS: i32 = 5;
/// Minimum success rate (0.0 - 1.0) for promotion eligibility.
const DEFAULT_MIN_RATE: f32 = 0.8;
const _: () = {
    assert!(DEFAULT_MIN_RATE > 0.0 && DEFAULT_MIN_RATE <= 1.0);
};
/// Stricter thresholds when evidence gate enforcement is enabled.
const EVIDENCE_MIN_SUCCESS: i32 = 7;
const EVIDENCE_MIN_RATE: f32 = 0.9;
const EVIDENCE_MIN_MARGIN: i32 = 3;
const MIN_PROCEDURE_STEPS: usize = 2;
const MIN_PROCEDURE_STEP_WORDS: usize = 8;
const MIN_DESCRIPTION_WORDS: usize = 4;
const MIN_BODY_LINES: usize = 2;
const MIN_BODY_WORDS: usize = 20;
const GENERIC_LOW_VALUE_TRIGGERS: &[&str] = &[
    "yes",
    "yeah",
    "yep",
    "yup",
    "ok",
    "okay",
    "sure",
    "indeed",
    "affirmative",
    "correct",
    "true",
    "totally",
    "absolutely",
    "no",
    "nope",
    "nah",
    "thanks",
    "thank you",
    "hello",
    "hi",
];

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

fn normalize_phrase(text: &str) -> String {
    let mut normalized = String::new();
    for raw in text.split_whitespace() {
        let token = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
            .to_lowercase();
        if token.is_empty() {
            continue;
        }
        if !normalized.is_empty() {
            normalized.push(' ');
        }
        normalized.push_str(&token);
    }
    normalized
}

fn is_generic_low_value_trigger(text: &str) -> bool {
    let normalized = normalize_phrase(text);
    if normalized.is_empty() {
        return true;
    }
    GENERIC_LOW_VALUE_TRIGGERS
        .iter()
        .any(|trigger| *trigger == normalized)
}

fn procedure_has_minimum_substance(procedure: &Procedure) -> bool {
    if procedure.steps.len() < MIN_PROCEDURE_STEPS {
        return false;
    }
    let total_step_words: usize = procedure.steps.iter().map(|step| word_count(step)).sum();
    total_step_words >= MIN_PROCEDURE_STEP_WORDS
}

fn skill_is_valuable(skill: &Skill) -> bool {
    if skill.triggers.is_empty() {
        return false;
    }
    if word_count(&skill.description) < MIN_DESCRIPTION_WORDS {
        return false;
    }

    let body_lines = skill
        .body
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    if body_lines < MIN_BODY_LINES {
        return false;
    }
    if word_count(&skill.body) < MIN_BODY_WORDS {
        return false;
    }

    let non_generic_trigger_count = skill
        .triggers
        .iter()
        .filter(|trigger| !is_generic_low_value_trigger(trigger))
        .count();
    if non_generic_trigger_count == 0 {
        return false;
    }

    true
}

pub struct SkillPromoter {
    state: Arc<dyn StateStore>,
    llm_runtime: SharedLlmRuntime,
    skills_dir: PathBuf,
    evidence_gate_enforce: bool,
}

impl SkillPromoter {
    pub fn new(
        state: Arc<dyn StateStore>,
        llm_runtime: SharedLlmRuntime,
        skills_dir: PathBuf,
        evidence_gate_enforce: bool,
    ) -> Self {
        Self {
            state,
            llm_runtime,
            skills_dir,
            evidence_gate_enforce,
        }
    }

    /// Run a full promotion cycle: find eligible procedures, generate skill drafts,
    /// and insert them into the skill_drafts table for user review.
    pub async fn run_promotion_cycle(&self) -> anyhow::Result<usize> {
        let promotable = self.check_promotable_procedures().await?;
        if promotable.is_empty() {
            return Ok(0);
        }

        let existing_skills = skills::load_skills(&self.skills_dir);
        let existing_names: Vec<&str> = existing_skills.iter().map(|s| s.name.as_str()).collect();

        let mut drafted = 0;
        for procedure in &promotable {
            if !procedure_has_minimum_substance(procedure) {
                info!(
                    procedure = %procedure.name,
                    steps = procedure.steps.len(),
                    "Skipping promotion: low-substance procedure"
                );
                continue;
            }
            if self.evidence_gate_enforce
                && procedure.success_count - procedure.failure_count < EVIDENCE_MIN_MARGIN
            {
                info!(
                    procedure = %procedure.name,
                    success_count = procedure.success_count,
                    failure_count = procedure.failure_count,
                    "Skipping promotion: insufficient evidence margin"
                );
                continue;
            }

            // Skip if a skill with this name already exists on disk
            let candidate_name = procedure.name.to_lowercase().replace(' ', "-");
            if existing_names
                .iter()
                .any(|n| n.to_lowercase() == candidate_name)
            {
                continue;
            }

            // Skip if this procedure has already been reviewed as a draft before
            if self
                .state
                .skill_draft_exists_for_procedure(&procedure.name)
                .await?
            {
                continue;
            }

            match self.promote_procedure(procedure).await {
                Ok(Some(skill)) => {
                    let normalized_skill_name = skills::sanitize_skill_filename(&skill.name);
                    if existing_names
                        .iter()
                        .any(|name| skills::sanitize_skill_filename(name) == normalized_skill_name)
                    {
                        info!(
                            skill = %skill.name,
                            procedure = %procedure.name,
                            "Skipping promotion: generated skill name conflicts with installed skill"
                        );
                        continue;
                    }

                    // Insert as draft, not directly to filesystem
                    let triggers_json = serde_json::to_string(&skill.triggers)?;
                    let draft = SkillDraft {
                        id: 0,
                        name: skill.name.clone(),
                        description: skill.description.clone(),
                        triggers_json,
                        body: skill.body.clone(),
                        source_procedure: procedure.name.clone(),
                        status: "pending".to_string(),
                        created_at: String::new(),
                    };
                    self.state.add_skill_draft(&draft).await?;

                    info!(
                        name = %skill.name,
                        procedure = %procedure.name,
                        success_count = procedure.success_count,
                        "Auto-promoted procedure to skill draft (pending review)"
                    );
                    drafted += 1;
                }
                Ok(None) => {
                    // LLM decided not to promote
                }
                Err(e) => {
                    warn!(
                        procedure = %procedure.name,
                        error = %e,
                        "Failed to promote procedure to skill draft"
                    );
                }
            }
        }

        Ok(drafted)
    }

    /// Query for procedures that meet the promotion threshold.
    async fn check_promotable_procedures(&self) -> anyhow::Result<Vec<Procedure>> {
        let (min_success, min_rate) = if self.evidence_gate_enforce {
            (EVIDENCE_MIN_SUCCESS, EVIDENCE_MIN_RATE)
        } else {
            (DEFAULT_MIN_SUCCESS, DEFAULT_MIN_RATE)
        };
        self.state
            .get_promotable_procedures(min_success, min_rate)
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
             procedure is too specific, context-dependent, or too trivial/generic to be a reusable skill \
             (for example: simple confirmations, greetings, or one-line acknowledgements).\n\n\
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
             Only return a skill when it provides meaningful user value and reusable workflow guidance. \
             Skip conversational filler behaviors.\n\
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
            json!({"role": "system", "content": "You are a skill generator. Convert proven procedures into reusable, high-value skill definitions. Skip trivial/generic conversational behaviors. Respond with ONLY the skill markdown or SKIP."}),
            json!({"role": "user", "content": prompt}),
        ];

        let runtime_snapshot = self.llm_runtime.snapshot();
        let fast_model = runtime_snapshot.fast_model();
        let response = runtime_snapshot
            .provider()
            .chat(&fast_model, &messages, &[])
            .await?;

        // Track token usage for background LLM calls
        if let Some(usage) = &response.usage {
            let _ = self
                .state
                .record_token_usage("background:skill_promotion", usage)
                .await;
        }

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
                if !skill_is_valuable(&skill) {
                    info!(
                        procedure = %procedure.name,
                        skill = %skill.name,
                        "Skipping low-value auto-generated skill draft"
                    );
                    return Ok(None);
                }
                skill.source = Some("auto".to_string());
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
        assert_eq!(DEFAULT_MIN_RATE, 0.8);
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

    #[test]
    fn low_value_affirmation_skill_rejected() {
        let skill = Skill {
            name: "respond-yes".to_string(),
            description: "Acknowledge and affirm a user's statement.".to_string(),
            triggers: vec![
                "yes".to_string(),
                "yeah".to_string(),
                "indeed".to_string(),
                "affirmative".to_string(),
            ],
            body: "1. Respond with an affirmative and encouraging statement.".to_string(),
            origin: None,
            source: Some("auto".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };

        assert!(!skill_is_valuable(&skill));
    }

    #[test]
    fn substantive_skill_accepted() {
        let skill = Skill {
            name: "deploy-rust-service".to_string(),
            description: "Build, validate, and deploy a Rust service safely.".to_string(),
            triggers: vec![
                "deploy service".to_string(),
                "release rust app".to_string(),
            ],
            body: "1. Run `cargo fmt`, `cargo clippy --all-features -- -D warnings`, and `cargo test`.\n2. Build the release artifact with `cargo build --release` and verify config values.\n3. Deploy the artifact, restart the service, and check health endpoints/logs before reporting completion."
                .to_string(),
            origin: None,
            source: Some("auto".to_string()),
            source_url: None,
            dir_path: None,
            resources: vec![],
        };

        assert!(skill_is_valuable(&skill));
    }

    #[test]
    fn low_substance_procedure_rejected() {
        let proc = Procedure {
            id: 0,
            name: "confirm".to_string(),
            trigger_pattern: "say yes".to_string(),
            steps: vec!["reply yes".to_string()],
            success_count: 5,
            failure_count: 0,
            avg_duration_secs: None,
            last_used_at: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        assert!(!procedure_has_minimum_substance(&proc));
    }
}
