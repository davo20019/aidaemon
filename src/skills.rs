use std::path::Path;
use serde_json::json;
use tracing::{info, warn};

use crate::traits::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, Fact, Goal, ModelProvider, Procedure,
    UserProfile,
};

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

/// Check if a character is a word boundary (not alphanumeric and not underscore).
fn is_word_boundary(c: char) -> bool {
    !c.is_alphanumeric() && c != '_'
}

/// Check if `word` appears as a whole word in `text` (case-sensitive; caller should
/// lowercase both if needed). Word boundaries are start/end of string or any character
/// that is not alphanumeric/underscore.
fn contains_whole_word(text: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = text[start..].find(word) {
        let abs_pos = start + pos;
        let before_ok = abs_pos == 0
            || text[..abs_pos]
                .chars()
                .next_back()
                .map_or(true, is_word_boundary);
        let after_pos = abs_pos + word.len();
        let after_ok = after_pos >= text.len()
            || text[after_pos..]
                .chars()
                .next()
                .map_or(true, is_word_boundary);
        if before_ok && after_ok {
            return true;
        }
        // Advance past this occurrence
        start = abs_pos + 1;
        if start >= text.len() {
            break;
        }
    }
    false
}

/// Match skills whose trigger keywords appear as whole words in the user message (case-insensitive).
pub fn match_skills<'a>(skills: &'a [Skill], user_message: &str) -> Vec<&'a Skill> {
    let lower = user_message.to_lowercase();
    skills
        .iter()
        .filter(|skill| {
            skill
                .triggers
                .iter()
                .any(|trigger| contains_whole_word(&lower, trigger))
        })
        .collect()
}

/// Ask a fast LLM to confirm which candidate skills are truly relevant to the user message.
/// Returns only the confirmed subset. On any error, returns the full candidate list (fail-open).
pub async fn confirm_skills<'a>(
    provider: &dyn ModelProvider,
    fast_model: &str,
    candidates: Vec<&'a Skill>,
    user_message: &str,
) -> anyhow::Result<Vec<&'a Skill>> {
    if candidates.is_empty() {
        return Ok(candidates);
    }

    let skills_list: String = candidates
        .iter()
        .map(|s| format!("- {}: {}", s.name, s.description))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Given this user message, which of these skills (if any) are relevant?\n\
         Return ONLY a JSON array of skill names, or [] if none.\n\n\
         Message: \"{}\"\n\n\
         Skills:\n{}",
        user_message, skills_list
    );

    let messages = vec![
        json!({"role": "system", "content": "You are a skill classifier. Respond with only a JSON array of skill names."}),
        json!({"role": "user", "content": prompt}),
    ];

    let response = provider.chat(fast_model, &messages, &[]).await?;
    let text = response
        .content
        .ok_or_else(|| anyhow::anyhow!("Empty response from skill confirmation LLM"))?;

    // Parse: strip markdown fences, find [...], deserialize
    let trimmed = text.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            return Ok(candidates); // malformed, fail-open
        }
    } else {
        return Ok(candidates); // no array found, fail-open
    };

    let names: Vec<String> = match serde_json::from_str(json_str) {
        Ok(n) => n,
        Err(_) => return Ok(candidates), // parse error, fail-open
    };

    let confirmed: Vec<&'a Skill> = candidates
        .into_iter()
        .filter(|s| names.iter().any(|n| n == &s.name))
        .collect();

    Ok(confirmed)
}

/// Build the complete system prompt from base prompt, all skills (for listing),
/// active skills (full body injection), and known facts.
///
/// `max_facts` caps the number of facts injected into the prompt. Facts are
/// assumed to arrive ordered by most-recently-updated first (from `get_facts()`).
#[allow(dead_code)] // Reserved for future use - replaced by build_system_prompt_with_memory
pub fn build_system_prompt(
    base: &str,
    skills: &[Skill],
    active: &[&Skill],
    facts: &[Fact],
    max_facts: usize,
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

    // Known facts (capped to max_facts, already ordered by updated_at DESC)
    let capped_facts = if facts.len() > max_facts {
        &facts[..max_facts]
    } else {
        facts
    };
    if !capped_facts.is_empty() {
        prompt.push_str("\n\n## Known Facts\n");
        for f in capped_facts {
            prompt.push_str(&format!("- [{}] {}: {}\n", f.category, f.key, f.value));
        }
    }

    prompt
}

/// Extended context for memory-rich system prompts.
#[derive(Default)]
pub struct MemoryContext<'a> {
    pub facts: &'a [Fact],
    pub episodes: &'a [Episode],
    pub goals: &'a [Goal],
    pub patterns: &'a [BehaviorPattern],
    pub procedures: &'a [Procedure],
    pub error_solutions: &'a [ErrorSolution],
    pub expertise: &'a [Expertise],
    pub profile: Option<&'a UserProfile>,
}

/// Build the complete system prompt with all memory components.
///
/// This extended version injects episodic memory, goals, procedures, expertise,
/// and behavior patterns in addition to facts and skills.
pub fn build_system_prompt_with_memory(
    base: &str,
    skills: &[Skill],
    active: &[&Skill],
    memory: &MemoryContext,
    max_facts: usize,
) -> String {
    let mut prompt = base.to_string();

    // 1. Communication Style (from user profile)
    if let Some(profile) = memory.profile {
        prompt.push_str("\n\n## Communication Preferences\n");
        prompt.push_str(&format!(
            "- Verbosity: {}\n- Tone: {}\n- Explanation depth: {}\n",
            profile.verbosity_preference, profile.tone_preference, profile.explanation_depth
        ));
        if profile.likes_suggestions {
            prompt.push_str("- User appreciates proactive suggestions\n");
        }
        if !profile.prefers_explanations {
            prompt.push_str("- Keep explanations brief — user prefers direct answers\n");
        }
    }

    // 2. Expertise Levels
    if !memory.expertise.is_empty() {
        prompt.push_str("\n\n## Your Expertise Levels\n");
        prompt.push_str("| Domain | Level | Confidence |\n|--------|-------|------------|\n");
        for exp in memory.expertise.iter().take(10) {
            prompt.push_str(&format!(
                "| {} | {} | {:.0}% |\n",
                exp.domain,
                exp.current_level,
                exp.confidence_score * 100.0
            ));
        }
    }

    // 3. Known Procedures (high success rate)
    let good_procedures: Vec<&Procedure> = memory
        .procedures
        .iter()
        .filter(|p| p.success_count > p.failure_count && p.success_count >= 2)
        .take(5)
        .collect();
    if !good_procedures.is_empty() {
        prompt.push_str("\n\n## Known Procedures\n");
        prompt.push_str("I've successfully used these approaches before:\n");
        for proc in good_procedures {
            let success_rate = proc.success_count as f32
                / (proc.success_count + proc.failure_count) as f32
                * 100.0;
            prompt.push_str(&format!(
                "- **{}** (trigger: '{}', {:.0}% success): {}\n",
                proc.name,
                truncate(&proc.trigger_pattern, 30),
                success_rate,
                proc.steps.first().map(|s| s.as_str()).unwrap_or("...")
            ));
        }
    }

    // 4. Known Error Fixes (high success rate)
    let good_solutions: Vec<&ErrorSolution> = memory
        .error_solutions
        .iter()
        .filter(|s| s.success_count > s.failure_count && s.success_count >= 2)
        .take(5)
        .collect();
    if !good_solutions.is_empty() {
        prompt.push_str("\n\n## Known Error Solutions\n");
        prompt.push_str("I've resolved these types of errors before:\n");
        for sol in good_solutions {
            prompt.push_str(&format!(
                "- **{}**: {} ({}x successful)\n",
                truncate(&sol.error_pattern, 50),
                sol.solution_summary,
                sol.success_count
            ));
        }
    }

    // 5. Active Goals
    let active_goals: Vec<&Goal> = memory
        .goals
        .iter()
        .filter(|g| g.status == "active")
        .take(5)
        .collect();
    if !active_goals.is_empty() {
        prompt.push_str("\n\n## Active Goals\n");
        prompt.push_str("The user is working toward:\n");
        for goal in active_goals {
            let priority_marker = match goal.priority.as_str() {
                "high" => "🔴",
                "medium" => "🟡",
                _ => "⚪",
            };
            prompt.push_str(&format!("- {} {}\n", priority_marker, goal.description));
        }
    }

    // 6. Known Facts (with history indication)
    let capped_facts = if memory.facts.len() > max_facts {
        &memory.facts[..max_facts]
    } else {
        memory.facts
    };
    if !capped_facts.is_empty() {
        prompt.push_str("\n\n## Known Facts\n");
        for f in capped_facts {
            let history_marker = if f.recall_count > 3 { " (frequently used)" } else { "" };
            prompt.push_str(&format!(
                "- [{}] {}: {}{}\n",
                f.category, f.key, f.value, history_marker
            ));
        }
    }

    // 7. Relevant Past Experiences
    if !memory.episodes.is_empty() {
        prompt.push_str("\n\n## Relevant Past Sessions\n");
        for ep in memory.episodes.iter().take(3) {
            let tone = ep.emotional_tone.as_deref().unwrap_or("neutral");
            let outcome = ep.outcome.as_deref().unwrap_or("unknown");
            prompt.push_str(&format!(
                "- {} (tone: {}, outcome: {})\n",
                truncate(&ep.summary, 100),
                tone,
                outcome
            ));
        }
    }

    // 8. Behavior Patterns (high confidence)
    let confident_patterns: Vec<&BehaviorPattern> = memory
        .patterns
        .iter()
        .filter(|p| p.confidence >= 0.7)
        .take(3)
        .collect();
    if !confident_patterns.is_empty() {
        prompt.push_str("\n\n## Observed Patterns\n");
        for pattern in confident_patterns {
            prompt.push_str(&format!("- {}\n", pattern.description));
        }
    }

    // 9. Available Skills
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

    prompt
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- contains_whole_word tests ---

    #[test]
    fn whole_word_at_start() {
        assert!(contains_whole_word("browse the web", "browse"));
    }

    #[test]
    fn whole_word_at_end() {
        assert!(contains_whole_word("open the browser", "browser"));
    }

    #[test]
    fn whole_word_in_middle() {
        assert!(contains_whole_word("please browse now", "browse"));
    }

    #[test]
    fn whole_word_with_punctuation() {
        assert!(contains_whole_word("browse, please", "browse"));
        assert!(contains_whole_word("use browse.", "browse"));
        assert!(contains_whole_word("(browse)", "browse"));
    }

    #[test]
    fn whole_word_hyphenated() {
        assert!(contains_whole_word("web-browse tool", "browse"));
    }

    #[test]
    fn partial_match_rejected() {
        assert!(!contains_whole_word("browseable interface", "browse"));
        assert!(!contains_whole_word("prebrowse step", "browse"));
    }

    #[test]
    fn substring_in_longer_word_rejected() {
        assert!(!contains_whole_word("remembering things", "member"));
    }

    #[test]
    fn exact_match_only_word() {
        assert!(contains_whole_word("browse", "browse"));
    }

    #[test]
    fn empty_word_returns_false() {
        assert!(!contains_whole_word("anything", ""));
    }

    #[test]
    fn empty_text_returns_false() {
        assert!(!contains_whole_word("", "word"));
    }

    #[test]
    fn word_with_underscore_boundary() {
        // underscore is NOT a word boundary, so "disk" inside "disk_check" is NOT a whole word
        assert!(!contains_whole_word("run disk_check", "disk"));
        // but "disk_check" as a whole word IS matched
        assert!(contains_whole_word("run disk_check now", "disk_check"));
    }

    // --- match_skills tests ---

    fn make_skill(name: &str, triggers: &[&str]) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{} skill", name),
            triggers: triggers.iter().map(|t| t.to_lowercase()).collect(),
            body: String::new(),
        }
    }

    #[test]
    fn match_skills_whole_word() {
        let skills = vec![
            make_skill("web-browsing", &["browse", "website"]),
            make_skill("system-admin", &["disk", "memory", "cpu"]),
        ];
        // "browse" as whole word matches
        let matched = match_skills(&skills, "browse google.com");
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "web-browsing");
    }

    #[test]
    fn match_skills_rejects_partial() {
        let skills = vec![make_skill("web-browsing", &["browse", "website"])];
        // "browseable" should NOT match "browse"
        let matched = match_skills(&skills, "this is a browseable page");
        assert!(matched.is_empty());
    }

    #[test]
    fn match_skills_case_insensitive() {
        let skills = vec![make_skill("web-browsing", &["browse"])];
        let matched = match_skills(&skills, "Please BROWSE the site");
        assert_eq!(matched.len(), 1);
    }

    // --- build_system_prompt max_facts tests ---

    fn make_fact(category: &str, key: &str, value: &str) -> Fact {
        Fact {
            id: 0,
            category: category.to_string(),
            key: key.to_string(),
            value: value.to_string(),
            source: "test".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            superseded_at: None,
            recall_count: 0,
            last_recalled_at: None,
        }
    }

    #[test]
    fn build_prompt_caps_facts() {
        let facts: Vec<Fact> = (0..10)
            .map(|i| make_fact("user", &format!("key{}", i), &format!("val{}", i)))
            .collect();
        let prompt = build_system_prompt("base", &[], &[], &facts, 3);
        // Should contain exactly 3 facts
        assert!(prompt.contains("key0"));
        assert!(prompt.contains("key2"));
        assert!(!prompt.contains("key3"));
    }

    #[test]
    fn build_prompt_no_cap_when_under_limit() {
        let facts = vec![make_fact("user", "name", "Alice")];
        let prompt = build_system_prompt("base", &[], &[], &facts, 100);
        assert!(prompt.contains("Alice"));
    }

    #[test]
    fn build_prompt_zero_max_facts() {
        let facts = vec![make_fact("user", "name", "Alice")];
        let prompt = build_system_prompt("base", &[], &[], &facts, 0);
        assert!(!prompt.contains("Known Facts"));
    }
}
