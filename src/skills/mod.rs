use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tracing::{info, warn};

pub mod resources;
pub use resources::{FileSystemResolver, ResourceEntry, ResourceResolver};

use crate::tools::sanitize::sanitize_external_content;
use crate::traits::{
    BehaviorPattern, Episode, ErrorSolution, Expertise, Fact, Goal, ModelProvider, Person,
    PersonFact, Procedure, StateStore, UserProfile,
};
use crate::types::{ChannelVisibility, UserRole};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub triggers: Vec<String>,
    pub body: String,
    /// Ownership/distribution class: "custom" or "contrib"
    #[serde(default)]
    pub origin: Option<String>,
    /// Where this skill came from: "filesystem", "url", "inline", "auto", "registry"
    #[serde(default)]
    pub source: Option<String>,
    /// URL this skill was fetched from (if source is "url")
    #[serde(default)]
    pub source_url: Option<String>,
    /// Filesystem path for directory-based skills (None for single-file/dynamic skills)
    #[serde(default)]
    pub dir_path: Option<PathBuf>,
    /// Available resources (metadata only — file names, not content)
    #[serde(default)]
    pub resources: Vec<ResourceEntry>,
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
        let mut origin = None;
        let mut source = None;
        let mut source_url = None;

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
                    "origin" => {
                        if !value.is_empty() {
                            origin = Some(value.to_string());
                        }
                    }
                    "source" => {
                        if !value.is_empty() {
                            source = Some(value.to_string());
                        }
                    }
                    "source_url" => {
                        if !value.is_empty() {
                            source_url = Some(value.to_string());
                        }
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
            origin,
            source,
            source_url,
            dir_path: None,
            resources: vec![],
        })
    }

    /// Serialize a Skill back to frontmatter + body markdown format.
    pub fn to_markdown(&self) -> String {
        let mut md = String::from("---\n");
        md.push_str(&format!("name: {}\n", self.name));
        if !self.description.is_empty() {
            md.push_str(&format!("description: {}\n", self.description));
        }
        if !self.triggers.is_empty() {
            md.push_str(&format!("triggers: {}\n", self.triggers.join(", ")));
        }
        if let Some(ref origin) = self.origin {
            md.push_str(&format!("origin: {}\n", origin));
        }
        if let Some(ref source) = self.source {
            md.push_str(&format!("source: {}\n", source));
        }
        if let Some(ref url) = self.source_url {
            md.push_str(&format!("source_url: {}\n", url));
        }
        md.push_str("---\n");
        md.push_str(&self.body);
        if !self.body.ends_with('\n') {
            md.push('\n');
        }
        md
    }
}

pub const SKILL_ORIGIN_CUSTOM: &str = "custom";
pub const SKILL_ORIGIN_CONTRIB: &str = "contrib";

fn normalize_skill_origin(origin: Option<&str>) -> Option<&'static str> {
    match origin {
        Some(value) if value.eq_ignore_ascii_case(SKILL_ORIGIN_CUSTOM) => Some(SKILL_ORIGIN_CUSTOM),
        Some(value) if value.eq_ignore_ascii_case(SKILL_ORIGIN_CONTRIB) => {
            Some(SKILL_ORIGIN_CONTRIB)
        }
        _ => None,
    }
}

pub fn infer_skill_origin(origin: Option<&str>, source: Option<&str>) -> &'static str {
    if let Some(explicit) = normalize_skill_origin(origin) {
        return explicit;
    }
    match source {
        Some(value) if value.eq_ignore_ascii_case("registry") => SKILL_ORIGIN_CONTRIB,
        _ => SKILL_ORIGIN_CUSTOM,
    }
}

/// Sanitize a skill name into a safe filename.
/// Lowercase, replace spaces/underscores with hyphens, strip non-alphanumeric except hyphens,
/// collapse consecutive hyphens, strip leading dots.
pub fn sanitize_skill_filename(name: &str) -> String {
    let lower = name.to_lowercase();
    let sanitized: String = lower
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();

    // Collapse consecutive hyphens
    let mut result = String::new();
    let mut prev_hyphen = false;
    for c in sanitized.chars() {
        if c == '-' {
            if !prev_hyphen && !result.is_empty() {
                result.push(c);
            }
            prev_hyphen = true;
        } else {
            result.push(c);
            prev_hyphen = false;
        }
    }

    // Strip trailing hyphens
    let result = result.trim_end_matches('-').to_string();

    if result.is_empty() {
        "skill".to_string()
    } else {
        result
    }
}

/// Write a skill to a `.md` file in the given directory. Uses atomic write
/// (temp file + rename) to prevent partial writes. Returns the path written.
pub fn write_skill_to_file(dir: &Path, skill: &Skill) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;

    let filename = format!("{}.md", sanitize_skill_filename(&skill.name));
    let target = dir.join(&filename);
    let tmp = dir.join(format!(".{}.tmp", filename));

    let content = skill.to_markdown();
    std::fs::write(&tmp, &content)?;
    std::fs::rename(&tmp, &target)?;

    Ok(target)
}

/// Remove a skill file from the directory. Handles both single `.md` files and
/// directory-based skills. Returns true if something was removed.
pub fn remove_skill_file(dir: &Path, skill_name: &str) -> anyhow::Result<bool> {
    // Try exact filename match first
    let sanitized = sanitize_skill_filename(skill_name);
    let md_path = dir.join(format!("{}.md", sanitized));
    if md_path.exists() {
        std::fs::remove_file(&md_path)?;
        return Ok(true);
    }

    // Try directory-based skill
    let dir_path = dir.join(&sanitized);
    if dir_path.is_dir() {
        std::fs::remove_dir_all(&dir_path)?;
        return Ok(true);
    }

    // Fallback: scan all files and parse to find matching name
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skill_md = path.join("SKILL.md");
                if skill_md.exists() {
                    if let Ok(content) = std::fs::read_to_string(&skill_md) {
                        if let Some(parsed) = Skill::parse(&content) {
                            if parsed.name == skill_name {
                                std::fs::remove_dir_all(&path)?;
                                return Ok(true);
                            }
                        }
                    }
                }
            } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Some(parsed) = Skill::parse(&content) {
                        if parsed.name == skill_name {
                            std::fs::remove_file(&path)?;
                            return Ok(true);
                        }
                    }
                }
            }
        }
    }

    Ok(false)
}

/// Find a skill by name (case-sensitive) in a slice.
pub fn find_skill_by_name<'a>(skills: &'a [Skill], name: &str) -> Option<&'a Skill> {
    skills.iter().find(|s| s.name == name)
}

/// Scan a skill directory for resource files in any subdirectory.
fn scan_skill_resources(dir: &Path) -> Vec<ResourceEntry> {
    let mut entries = Vec::new();
    let Ok(subdirs) = std::fs::read_dir(dir) else {
        return entries;
    };

    for subdir_entry in subdirs.flatten() {
        let subdir_path = subdir_entry.path();
        if !subdir_path.is_dir() {
            continue;
        }

        let subdir_name = match subdir_path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Infer category from directory name
        let category = match subdir_name.as_str() {
            "scripts" => "script",
            "references" => "reference",
            "assets" => "asset",
            other => other, // custom directories work too
        }
        .to_string();

        if let Ok(files) = std::fs::read_dir(&subdir_path) {
            for file in files.flatten() {
                let path = file.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        entries.push(ResourceEntry {
                            path: format!("{}/{}", subdir_name, name),
                            category: category.clone(),
                        });
                    }
                }
            }
        }
    }

    entries.sort_by(|a, b| a.path.cmp(&b.path));
    entries
}

/// Load a directory-based skill from a directory containing SKILL.md.
fn load_directory_skill(dir: &Path) -> Option<Skill> {
    let skill_md = dir.join("SKILL.md");
    let content = std::fs::read_to_string(&skill_md).ok()?;
    let mut skill = Skill::parse(&content)?;
    skill.dir_path = Some(dir.to_path_buf());
    skill.resources = scan_skill_resources(dir);
    Some(skill)
}

/// Load skills from a directory. Supports both:
/// - Legacy single `.md` files (e.g. `skills/deploy.md`)
/// - Directory-based skills with `SKILL.md` (e.g. `skills/deploy/SKILL.md`)
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

        if path.is_dir() {
            // Directory-based skill: look for SKILL.md
            if let Some(skill) = load_directory_skill(&path) {
                info!(
                    name = %skill.name,
                    triggers = ?skill.triggers,
                    resources = skill.resources.len(),
                    "Loaded directory skill"
                );
                skills.push(skill);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            // Legacy single-file skill
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
    }

    skills
}

/// Cached skill loader that avoids re-reading files on every message.
/// Checks directory modification time and only reloads when files change.
#[derive(Clone)]
pub struct SkillCache {
    dir: PathBuf,
    inner: Arc<Mutex<SkillCacheInner>>,
}

struct SkillCacheInner {
    skills: Vec<Skill>,
    last_checked: SystemTime,
    dir_mtime: Option<SystemTime>,
}

impl SkillCache {
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            inner: Arc::new(Mutex::new(SkillCacheInner {
                skills: Vec::new(),
                last_checked: SystemTime::UNIX_EPOCH,
                dir_mtime: None,
            })),
        }
    }

    /// Returns cached skills, reloading only if the directory has been modified.
    pub fn get(&self) -> Vec<Skill> {
        let current_mtime = std::fs::metadata(&self.dir).and_then(|m| m.modified()).ok();

        let mut inner = self.inner.lock().unwrap();

        // Reload if directory mtime changed or cache is empty
        if inner.dir_mtime != current_mtime || inner.skills.is_empty() {
            inner.skills = load_skills(&self.dir);
            inner.dir_mtime = current_mtime;
            inner.last_checked = SystemTime::now();
        }

        inner.skills.clone()
    }

    /// Force a reload on next access (e.g., after adding/removing a skill).
    #[allow(dead_code)]
    pub fn invalidate(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.dir_mtime = None;
    }
}

/// Normalize a potential skill reference token into canonical filename form.
fn normalize_skill_ref(token: &str) -> String {
    let trimmed = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
    sanitize_skill_filename(trimmed)
}

/// Extract explicit skill references from the user message.
///
/// Supported explicit forms:
/// - `$skill-name`
/// - `skill:skill-name`
/// - `use skill <skill-name>`
fn extract_explicit_skill_refs(user_message: &str) -> Vec<String> {
    let lower = user_message.to_lowercase();
    let mut refs: Vec<String> = Vec::new();

    for token in lower.split_whitespace() {
        if let Some(raw) = token.strip_prefix('$') {
            let norm = normalize_skill_ref(raw);
            if !norm.is_empty() && !refs.contains(&norm) {
                refs.push(norm);
            }
        }
        if let Some(raw) = token.strip_prefix("skill:") {
            let norm = normalize_skill_ref(raw);
            if !norm.is_empty() && !refs.contains(&norm) {
                refs.push(norm);
            }
        }
    }

    // Parse "use skill <name>"
    let words: Vec<&str> = lower.split_whitespace().collect();
    for window in words.windows(3) {
        if window[0] == "use" && window[1] == "skill" {
            let norm = normalize_skill_ref(window[2]);
            if !norm.is_empty() && !refs.contains(&norm) {
                refs.push(norm);
            }
        }
    }

    refs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillMatchKind {
    None,
    Explicit,
    Trigger,
}

pub struct SkillMatches<'a> {
    pub kind: SkillMatchKind,
    pub skills: Vec<&'a Skill>,
}

fn normalize_for_trigger_match(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_string()
}

fn trigger_matches_message(message_norm_padded: &str, trigger: &str) -> bool {
    let t = normalize_for_trigger_match(trigger);
    // Skip extremely short triggers to reduce false positives ("a", "i", etc.).
    let compact_len = t.chars().filter(|c| c.is_ascii_alphanumeric()).count();
    if compact_len < 3 {
        return false;
    }
    let needle = format!(" {} ", t);
    message_norm_padded.contains(&needle)
}

fn match_skills_by_triggers<'a>(skills: &'a [Skill], user_message: &str) -> Vec<&'a Skill> {
    // Normalize and pad so we can do cheap word-boundary matching with spaces.
    let normalized = normalize_for_trigger_match(user_message);
    if normalized.is_empty() {
        return Vec::new();
    }
    let padded = format!(" {} ", normalized);

    let mut scored: Vec<(&Skill, usize)> = Vec::new();
    for skill in skills {
        if skill.triggers.is_empty() {
            continue;
        }
        let mut score = 0usize;
        for trig in &skill.triggers {
            if trigger_matches_message(&padded, trig) {
                score += 1;
            }
        }
        if score > 0 {
            scored.push((skill, score));
        }
    }

    // Deterministic order: score desc, then name asc.
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.name.cmp(&b.0.name)));

    let mut out: Vec<&Skill> = scored.into_iter().map(|(s, _)| s).collect();
    if out.len() > 8 {
        out.truncate(8);
    }
    out
}

/// Match skills for the given message.
///
/// Rules:
/// - Explicit references always win (`$skill`, `skill:name`, `use skill <name>`).
/// - Trigger matching is only enabled for the Owner in non-PublicExternal channels.
/// - Trigger matches are capped to a small number to keep prompts stable.
pub fn match_skills<'a>(
    skills: &'a [Skill],
    user_message: &str,
    user_role: UserRole,
    visibility: ChannelVisibility,
) -> SkillMatches<'a> {
    let refs = extract_explicit_skill_refs(user_message);
    if !refs.is_empty() {
        let matched: Vec<&Skill> = skills
            .iter()
            .filter(|skill| {
                refs.iter()
                    .any(|r| r == &sanitize_skill_filename(&skill.name))
            })
            .collect();
        if matched.is_empty() {
            return SkillMatches {
                kind: SkillMatchKind::None,
                skills: Vec::new(),
            };
        }
        return SkillMatches {
            kind: SkillMatchKind::Explicit,
            skills: matched,
        };
    }

    // Never trigger skills for untrusted public platforms.
    if matches!(visibility, ChannelVisibility::PublicExternal) {
        return SkillMatches {
            kind: SkillMatchKind::None,
            skills: Vec::new(),
        };
    }

    // Only allow keyword/trigger-based activation for the Owner.
    if user_role != UserRole::Owner {
        return SkillMatches {
            kind: SkillMatchKind::None,
            skills: Vec::new(),
        };
    }

    let triggered = match_skills_by_triggers(skills, user_message);
    if triggered.is_empty() {
        return SkillMatches {
            kind: SkillMatchKind::None,
            skills: Vec::new(),
        };
    }

    SkillMatches {
        kind: SkillMatchKind::Trigger,
        skills: triggered,
    }
}

/// Ask a fast LLM to confirm which candidate skills are truly relevant to the user message.
/// Returns only the confirmed subset. On any error, returns the full candidate list (fail-open).
pub async fn confirm_skills<'a>(
    provider: &dyn ModelProvider,
    fast_model: &str,
    candidates: Vec<&'a Skill>,
    user_message: &str,
    state: Option<&Arc<dyn StateStore>>,
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

    // Track token usage for skill confirmation LLM calls
    if let (Some(state), Some(usage)) = (state, &response.usage) {
        let _ = state
            .record_token_usage("background:skill_confirmation", usage)
            .await;
    }

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
#[allow(dead_code)] // Kept for backwards compatibility, use build_system_prompt_with_memory instead
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

    // Active skill bodies (sanitized to prevent prompt injection from external skill sources)
    for skill in active {
        let sanitized_body = sanitize_external_content(&skill.body);
        prompt.push_str(&format!(
            "\n\n## Active Skill: {}\n{}",
            skill.name, sanitized_body
        ));
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
    /// Trusted command patterns (pattern string, approval count)
    pub trusted_command_patterns: &'a [(String, i32)],
    /// Cross-channel hints: relevant facts from other channels (category+key only, values under confidentiality)
    pub cross_channel_hints: &'a [Fact],
    /// All tracked people (injected only in owner DMs)
    pub people: &'a [Person],
    /// The current speaker (resolved from sender_id), if any
    pub current_person: Option<&'a Person>,
    /// Facts about the current speaker
    pub current_person_facts: &'a [PersonFact],
}

/// Build the complete system prompt with all memory components.
///
/// This extended version injects episodic memory, goals, procedures, expertise,
/// and behavior patterns in addition to facts and skills.
/// Replace known user IDs (e.g., `U04S8KSS932`) with display names in text.
/// Matches both bare IDs and `<@USERID>` Slack mention format.
fn resolve_user_ids(text: &str, user_id_map: &HashMap<String, String>) -> String {
    if user_id_map.is_empty() {
        return text.to_string();
    }
    let mut result = text.to_string();
    for (uid, name) in user_id_map {
        // Replace <@USERID> format (Slack mentions)
        let mention = format!("<@{}>", uid);
        if result.contains(&mention) {
            result = result.replace(&mention, &format!("@{}", name));
        }
        // Replace bare user IDs (e.g., in stored facts)
        if result.contains(uid.as_str()) {
            result = result.replace(uid.as_str(), name);
        }
    }
    result
}

pub fn build_system_prompt_with_memory(
    base: &str,
    skills: &[Skill],
    active: &[&Skill],
    memory: &MemoryContext,
    max_facts: usize,
    suggestions: Option<&[crate::memory::proactive::Suggestion]>,
    user_id_map: &HashMap<String, String>,
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
        if profile.asks_before_acting {
            prompt.push_str(
                "- User prefers confirmation before destructive or system-modifying actions (file deletion, deployment, config changes). For read-only exploration (searching files, listing directories, reading code), proceed directly without asking. For multi-step modification tasks, briefly state your plan and confirm before executing.\n",
            );
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
        .filter(|p| p.success_count > p.failure_count && p.success_count >= 1)
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
        .filter(|s| s.success_count > s.failure_count && s.success_count >= 1)
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
            let history_marker = if f.recall_count > 3 {
                " (frequently used)"
            } else {
                ""
            };
            let resolved_key = resolve_user_ids(&f.key, user_id_map);
            let resolved_value = resolve_user_ids(&f.value, user_id_map);
            // Sanitize fact values to prevent prompt injection via stored data
            let sanitized_value = sanitize_external_content(&resolved_value);
            prompt.push_str(&format!(
                "- [{}] {}: {}{}\n",
                f.category, resolved_key, sanitized_value, history_marker
            ));
        }
    }

    // 6b. Cross-Channel Hints (facts from other channels — redacted values)
    if !memory.cross_channel_hints.is_empty() {
        prompt.push_str("\n\n## Cross-Channel Context — CONFIDENTIAL\n");
        prompt.push_str("You have relevant info from other conversations. You may mention that you have info,\n\
                         but NEVER reveal the actual values — they are redacted below.\n\
                         If the owner approves sharing, use `share_memory` tool to make the info permanently shareable here.\n");
        for f in memory.cross_channel_hints.iter().take(5) {
            let resolved_key = resolve_user_ids(&f.key, user_id_map);
            prompt.push_str(&format!(
                "- [{}] {}: <confidential> (from another channel)\n",
                f.category, resolved_key
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
    let failure_patterns: Vec<&BehaviorPattern> = memory
        .patterns
        .iter()
        .filter(|p| p.pattern_type == "failure" && p.confidence >= 0.5)
        .take(3)
        .collect();
    if !failure_patterns.is_empty() {
        prompt.push_str("\n\n## Failure Patterns To Avoid\n");
        for pattern in failure_patterns {
            prompt.push_str(&format!("- {}\n", pattern.description));
        }
    }

    let confident_patterns: Vec<&BehaviorPattern> = memory
        .patterns
        .iter()
        .filter(|p| p.pattern_type != "failure" && p.confidence >= 0.7)
        .take(3)
        .collect();
    if !confident_patterns.is_empty() {
        prompt.push_str("\n\n## Observed Patterns\n");
        for pattern in confident_patterns {
            prompt.push_str(&format!("- {}\n", pattern.description));
        }
    }

    // 9. Trusted Command Patterns
    if !memory.trusted_command_patterns.is_empty() {
        prompt.push_str("\n\n## Trusted Command Patterns\n");
        prompt.push_str("These command patterns have been approved multiple times and will have reduced approval friction:\n");
        for (pattern, count) in memory.trusted_command_patterns.iter().take(10) {
            prompt.push_str(&format!("- `{}` (approved {}x)\n", pattern, count));
        }
        prompt.push_str(
            "\nWhen suggesting terminal commands, prefer simpler commands over complex pipelines. ",
        );
        prompt.push_str("If a pipeline is necessary, explain why each part is needed.\n");
    }

    // 10. Contextual Suggestions (if user likes them)
    if let Some(suggestions) = suggestions {
        if !suggestions.is_empty() {
            prompt.push_str("\n\n## Contextual Suggestions\n");
            prompt.push_str("Consider these if relevant (do not force):\n");
            for s in suggestions.iter().take(3) {
                prompt.push_str(&format!("- {}\n", s.text));
            }
        }
    }

    // 11. People Privacy Rules (BEFORE data — agent sees constraints first)
    if !memory.people.is_empty() || memory.current_person.is_some() {
        prompt.push_str("\n\n## People Privacy Rules\n\
            - You ARE transparent about: adapting communication style, learning from conversations\n\
            - You are NOT transparent about: the owner's people graph, relationship insights, stored facts about people, or proactive coaching\n\
            - Only share detailed information about people with the OWNER in private DMs\n\
            - If someone asks \"what do you know about me?\", you may share what THEY have told you directly, but NOT facts the owner stored or things learned from other contexts\n\
            - Never say \"the owner told me about you\" or share facts from the owner's private notes\n\
            - In group chats, do not volunteer personal facts about any individual\n\
            - When proactively reminding the owner about dates/events, do so naturally (\"By the way, someone's birthday is coming up next week!\")\n");
    }

    // 12. People Context (only when people data exists)
    if !memory.people.is_empty() {
        prompt.push_str("\n\n## People You Know\n");
        for p in memory.people.iter().take(20) {
            let rel = p.relationship.as_deref().unwrap_or("contact");
            let style = p
                .communication_style
                .as_deref()
                .map(|s| format!(", style: {}", s))
                .unwrap_or_default();
            let lang = p
                .language_preference
                .as_deref()
                .map(|l| format!(", language: {}", l))
                .unwrap_or_default();
            prompt.push_str(&format!("- **{}** ({}){}{}\n", p.name, rel, style, lang));
        }
    }

    // 13. Current Speaker Context (when talking to a known person who is not the owner)
    if let Some(person) = memory.current_person {
        prompt.push_str(&format!(
            "\n\n## Current Speaker Context\nYou are talking to {} ",
            person.name
        ));
        if let Some(ref rel) = person.relationship {
            prompt.push_str(&format!("(the owner's {}). ", rel));
        } else {
            prompt.push_str("(a known contact). ");
        }
        if let Some(ref style) = person.communication_style {
            prompt.push_str(&format!("Communication style: {}. ", style));
        }
        if let Some(ref lang) = person.language_preference {
            prompt.push_str(&format!("Language preference: {}. ", lang));
        }
        prompt.push('\n');

        // Add relevant facts about this person
        if !memory.current_person_facts.is_empty() {
            prompt.push_str("Known facts about them:\n");
            for f in memory.current_person_facts.iter().take(10) {
                prompt.push_str(&format!("- [{}] {}: {}\n", f.category, f.key, f.value));
            }
        }

        // First interaction notice
        if person.interaction_count == 0 {
            prompt.push_str(&format!(
                "\nThis is your first interaction with {}. Naturally mention early in the conversation: \
                 \"I learn from our conversations and adapt my communication style over time. \
                 If you have any preferences, just let me know!\"\n",
                person.name
            ));
        }
    }

    // 14. Available Skills
    if !skills.is_empty() {
        prompt.push_str("\n\n## Available Skills\n");
        for skill in skills {
            prompt.push_str(&format!("- **{}**: {}\n", skill.name, skill.description));
        }
    }

    // Active skill bodies (sanitized to prevent prompt injection from external skill sources)
    for skill in active {
        let sanitized_body = sanitize_external_content(&skill.body);
        prompt.push_str(&format!(
            "\n\n## Active Skill: {}\n{}",
            skill.name, sanitized_body
        ));
        if !skill.resources.is_empty() {
            prompt.push_str(
                "\n\n**Bundled resources** (use `skill_resources` tool to load on demand):",
            );
            for entry in &skill.resources {
                prompt.push_str(&format!("\n- [{}] `{}`", entry.category, entry.path));
            }
        }
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

    // --- match_skills tests ---

    fn make_skill(name: &str, triggers: &[&str]) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{} skill", name),
            triggers: triggers.iter().map(|t| t.to_lowercase()).collect(),
            body: String::new(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources: vec![],
        }
    }

    #[test]
    fn match_skills_explicit_dollar_reference() {
        let skills = vec![
            make_skill("web-browsing", &["browse", "website"]),
            make_skill("system-admin", &["disk", "memory", "cpu"]),
        ];
        let matched = match_skills(
            &skills,
            "please run $web-browsing now",
            crate::types::UserRole::Owner,
            crate::types::ChannelVisibility::Private,
        );
        assert_eq!(matched.kind, SkillMatchKind::Explicit);
        assert_eq!(matched.skills.len(), 1);
        assert_eq!(matched.skills[0].name, "web-browsing");
    }

    #[test]
    fn match_skills_explicit_skill_prefix() {
        let skills = vec![make_skill("web-browsing", &["browse", "website"])];
        let matched = match_skills(
            &skills,
            "skill:web-browsing",
            crate::types::UserRole::Owner,
            crate::types::ChannelVisibility::Private,
        );
        assert_eq!(matched.kind, SkillMatchKind::Explicit);
        assert_eq!(matched.skills.len(), 1);
        assert_eq!(matched.skills[0].name, "web-browsing");
    }

    #[test]
    fn match_skills_use_skill_form() {
        let skills = vec![make_skill("web-browsing", &["browse"])];
        let matched = match_skills(
            &skills,
            "Use skill WEB-BROWSING",
            crate::types::UserRole::Guest,
            crate::types::ChannelVisibility::Public,
        );
        assert_eq!(matched.kind, SkillMatchKind::Explicit);
        assert_eq!(matched.skills.len(), 1);
    }

    #[test]
    fn match_skills_triggers_for_owner_in_private() {
        let skills = vec![make_skill("web-browsing", &["browse", "website"])];
        let matched = match_skills(
            &skills,
            "please browse the site",
            crate::types::UserRole::Owner,
            crate::types::ChannelVisibility::Private,
        );
        assert_eq!(matched.kind, SkillMatchKind::Trigger);
        assert_eq!(matched.skills.len(), 1);
        assert_eq!(matched.skills[0].name, "web-browsing");
    }

    #[test]
    fn match_skills_does_not_trigger_for_guest() {
        let skills = vec![make_skill("web-browsing", &["browse", "website"])];
        let matched = match_skills(
            &skills,
            "please browse the site",
            crate::types::UserRole::Guest,
            crate::types::ChannelVisibility::Private,
        );
        assert_eq!(matched.kind, SkillMatchKind::None);
        assert!(matched.skills.is_empty());
    }

    #[test]
    fn match_skills_does_not_trigger_for_public_external() {
        let skills = vec![make_skill("web-browsing", &["browse", "website"])];
        let matched = match_skills(
            &skills,
            "please browse the site",
            crate::types::UserRole::Owner,
            crate::types::ChannelVisibility::PublicExternal,
        );
        assert_eq!(matched.kind, SkillMatchKind::None);
        assert!(matched.skills.is_empty());
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
            channel_id: None,
            privacy: crate::types::FactPrivacy::Global,
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

    // --- directory skill tests ---

    #[test]
    fn test_load_directory_skill() {
        let dir = tempfile::TempDir::new().unwrap();
        let skill_dir = dir.path().join("test-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::create_dir(skill_dir.join("scripts")).unwrap();
        std::fs::create_dir(skill_dir.join("references")).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: test-skill\ndescription: A test\ntriggers: test\n---\nDo the thing.",
        )
        .unwrap();
        std::fs::write(skill_dir.join("scripts/hello.sh"), "#!/bin/bash\necho hi").unwrap();
        std::fs::write(
            skill_dir.join("references/guide.md"),
            "# Guide\nUse snake_case.",
        )
        .unwrap();

        let skill = load_directory_skill(&skill_dir).unwrap();
        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.resources.len(), 2);
        assert!(skill.dir_path.is_some());
        assert!(skill
            .resources
            .iter()
            .any(|r| r.path == "references/guide.md"));
        assert!(skill.resources.iter().any(|r| r.path == "scripts/hello.sh"));
    }

    #[test]
    fn test_load_skills_mixed() {
        let dir = tempfile::TempDir::new().unwrap();

        // Legacy single-file skill
        std::fs::write(
            dir.path().join("legacy.md"),
            "---\nname: legacy\ndescription: Legacy skill\ntriggers: old\n---\nLegacy body.",
        )
        .unwrap();

        // Directory-based skill
        let skill_dir = dir.path().join("new-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: new-skill\ndescription: New skill\ntriggers: new\n---\nNew body.",
        )
        .unwrap();

        let skills = load_skills(dir.path());
        assert_eq!(skills.len(), 2);
        assert!(skills.iter().any(|s| s.name == "legacy"));
        assert!(skills.iter().any(|s| s.name == "new-skill"));
    }

    #[test]
    fn test_scan_resources_custom_dirs() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("examples")).unwrap();
        std::fs::create_dir(dir.path().join("data")).unwrap();
        std::fs::write(dir.path().join("examples/demo.py"), "print('hi')").unwrap();
        std::fs::write(dir.path().join("data/config.json"), "{}").unwrap();

        let resources = scan_skill_resources(dir.path());
        assert_eq!(resources.len(), 2);
        // Custom directories use their dirname as category
        let examples_entry = resources
            .iter()
            .find(|r| r.path == "data/config.json")
            .unwrap();
        assert_eq!(examples_entry.category, "data");
        let data_entry = resources
            .iter()
            .find(|r| r.path == "examples/demo.py")
            .unwrap();
        assert_eq!(data_entry.category, "examples");
    }

    #[test]
    fn test_directory_without_skill_md_skipped() {
        let dir = tempfile::TempDir::new().unwrap();
        let random_dir = dir.path().join("random-stuff");
        std::fs::create_dir(&random_dir).unwrap();
        std::fs::write(random_dir.join("readme.txt"), "not a skill").unwrap();

        let skills = load_skills(dir.path());
        assert!(skills.is_empty());
    }

    #[test]
    fn test_parse_anthropic_format() {
        // Anthropic format: name + description, no explicit triggers.
        // Trigger inference is intentionally disabled to avoid keyword guessing.
        let content =
            "---\nname: code-review\ndescription: Review code for quality\n---\nCheck for bugs.";
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.name, "code-review");
        assert!(skill.triggers.is_empty());
    }

    // --- to_markdown roundtrip tests ---

    #[test]
    fn to_markdown_roundtrip() {
        let skill = Skill {
            name: "deploy-app".to_string(),
            description: "Deploy the application".to_string(),
            triggers: vec!["deploy".to_string(), "ship".to_string()],
            body: "Run cargo build --release\nCopy binary to server".to_string(),
            origin: Some("contrib".to_string()),
            source: Some("url".to_string()),
            source_url: Some("https://example.com/deploy.md".to_string()),
            dir_path: None,
            resources: vec![],
        };
        let md = skill.to_markdown();
        let parsed = Skill::parse(&md).unwrap();
        assert_eq!(parsed.name, skill.name);
        assert_eq!(parsed.description, skill.description);
        assert_eq!(parsed.triggers, skill.triggers);
        assert_eq!(parsed.body, skill.body);
        assert_eq!(parsed.origin, skill.origin);
        assert_eq!(parsed.source, skill.source);
        assert_eq!(parsed.source_url, skill.source_url);
    }

    #[test]
    fn to_markdown_minimal() {
        let skill = Skill {
            name: "simple".to_string(),
            description: String::new(),
            triggers: vec![],
            body: "Do the thing.".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        let md = skill.to_markdown();
        assert!(md.starts_with("---\n"));
        assert!(md.contains("name: simple"));
        let parsed = Skill::parse(&md).unwrap();
        assert_eq!(parsed.name, "simple");
    }

    // --- sanitize_skill_filename tests ---

    #[test]
    fn sanitize_basic() {
        assert_eq!(sanitize_skill_filename("Deploy App"), "deploy-app");
    }

    #[test]
    fn sanitize_special_chars() {
        assert_eq!(sanitize_skill_filename("my-skill (v2)"), "my-skill-v2");
    }

    #[test]
    fn sanitize_leading_dots() {
        // Leading non-alphanumeric chars are stripped
        assert_eq!(sanitize_skill_filename("...hidden"), "hidden");
    }

    #[test]
    fn sanitize_empty() {
        assert_eq!(sanitize_skill_filename(""), "skill");
        assert_eq!(sanitize_skill_filename("..."), "skill");
    }

    #[test]
    fn sanitize_already_clean() {
        assert_eq!(sanitize_skill_filename("deploy"), "deploy");
        assert_eq!(sanitize_skill_filename("code-review"), "code-review");
    }

    // --- write_skill_to_file tests ---

    #[test]
    fn write_and_read_skill_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let skill = Skill {
            name: "test-write".to_string(),
            description: "A writable skill".to_string(),
            triggers: vec!["test".to_string()],
            body: "Do tests.".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        let path = write_skill_to_file(dir.path(), &skill).unwrap();
        assert!(path.exists());
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), "test-write.md");

        // No leftover temp file
        let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().collect();
        assert_eq!(entries.len(), 1);

        // Roundtrip
        let loaded = load_skills(dir.path());
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "test-write");
    }

    // --- remove_skill_file tests ---

    #[test]
    fn remove_single_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let skill = Skill {
            name: "removable".to_string(),
            description: "Remove me".to_string(),
            triggers: vec![],
            body: "Body.".to_string(),
            origin: None,
            source: None,
            source_url: None,
            dir_path: None,
            resources: vec![],
        };
        write_skill_to_file(dir.path(), &skill).unwrap();
        assert!(remove_skill_file(dir.path(), "removable").unwrap());
        assert!(load_skills(dir.path()).is_empty());
    }

    #[test]
    fn remove_directory_skill() {
        let dir = tempfile::TempDir::new().unwrap();
        let skill_dir = dir.path().join("removable");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: removable\ndescription: test\ntriggers: rem\n---\nbody",
        )
        .unwrap();
        assert!(remove_skill_file(dir.path(), "removable").unwrap());
        assert!(!skill_dir.exists());
    }

    #[test]
    fn remove_not_found() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(!remove_skill_file(dir.path(), "nonexistent").unwrap());
    }

    // --- find_skill_by_name tests ---

    #[test]
    fn find_existing() {
        let skills = vec![make_skill("alpha", &[]), make_skill("beta", &[])];
        assert_eq!(find_skill_by_name(&skills, "beta").unwrap().name, "beta");
    }

    #[test]
    fn find_missing() {
        let skills = vec![make_skill("alpha", &[])];
        assert!(find_skill_by_name(&skills, "nope").is_none());
    }

    // --- parse source/source_url frontmatter ---

    #[test]
    fn parse_with_source_fields() {
        let content = "---\nname: fetched\ndescription: From URL\ntriggers: fetch\norigin: contrib\nsource: url\nsource_url: https://example.com/skill.md\n---\nFetched body.";
        let skill = Skill::parse(content).unwrap();
        assert_eq!(skill.origin.as_deref(), Some("contrib"));
        assert_eq!(skill.source.as_deref(), Some("url"));
        assert_eq!(
            skill.source_url.as_deref(),
            Some("https://example.com/skill.md")
        );
    }

    #[test]
    fn infer_origin_defaults_and_registry() {
        assert_eq!(infer_skill_origin(None, None), SKILL_ORIGIN_CUSTOM);
        assert_eq!(
            infer_skill_origin(None, Some("registry")),
            SKILL_ORIGIN_CONTRIB
        );
        assert_eq!(
            infer_skill_origin(Some("custom"), Some("registry")),
            SKILL_ORIGIN_CUSTOM
        );
    }
}
