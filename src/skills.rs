use std::path::Path;
use tracing::{info, warn};

use crate::traits::Fact;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub triggers: Vec<String>,
    pub body: String,
}

impl Skill {
    /// Parse a skill from markdown content with `---` frontmatter.
    pub fn parse(content: &str) -> Option<Skill> {
        let trimmed = content.trim();
        if !trimmed.starts_with("---") {
            return None;
        }

        // Split on the closing `---`
        let after_opening = &trimmed[3..];
        let end_idx = after_opening.find("---")?;
        let frontmatter = &after_opening[..end_idx];
        let body = after_opening[end_idx + 3..].trim().to_string();

        let mut name = None;
        let mut description = None;
        let mut triggers = Vec::new();

        for line in frontmatter.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();
                match key {
                    "name" => name = Some(value.to_string()),
                    "description" => description = Some(value.to_string()),
                    "triggers" => {
                        triggers = value
                            .split(',')
                            .map(|s| s.trim().to_lowercase())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                    _ => {}
                }
            }
        }

        Some(Skill {
            name: name?,
            description: description.unwrap_or_default(),
            triggers,
            body,
        })
    }
}

/// Load all `.md` skill files from the given directory.
pub fn load_skills(dir: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "Could not read skills directory");
            return skills;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                if let Some(skill) = Skill::parse(&content) {
                    info!(name = %skill.name, triggers = ?skill.triggers, "Loaded skill");
                    skills.push(skill);
                } else {
                    warn!(path = %path.display(), "Failed to parse skill file");
                }
            }
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to read skill file");
            }
        }
    }

    skills
}

/// Match skills whose trigger keywords appear in the user message (case-insensitive substring).
pub fn match_skills<'a>(skills: &'a [Skill], user_message: &str) -> Vec<&'a Skill> {
    let lower = user_message.to_lowercase();
    skills
        .iter()
        .filter(|skill| skill.triggers.iter().any(|trigger| lower.contains(trigger)))
        .collect()
}

/// Build the complete system prompt from base prompt, all skills (for listing),
/// active skills (full body injection), and known facts.
pub fn build_system_prompt(
    base: &str,
    skills: &[Skill],
    active: &[&Skill],
    facts: &[Fact],
) -> String {
    let mut prompt = base.to_string();

    // Available skills listing
    if !skills.is_empty() {
        prompt.push_str("\n\n## Available Skills\n");
        for skill in skills {
            prompt.push_str(&format!("- **{}**: {}\n", skill.name, skill.description));
        }
    }

    // Active skill bodies
    for skill in active {
        prompt.push_str(&format!("\n\n## Active Skill: {}\n{}", skill.name, skill.body));
    }

    // Known facts
    if !facts.is_empty() {
        prompt.push_str("\n\n## Known Facts\n");
        for f in facts {
            prompt.push_str(&format!("- [{}] {}: {}\n", f.category, f.key, f.value));
        }
    }

    prompt
}
