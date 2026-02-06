//! Skill registry client for browsing and installing skills from remote registries.
//!
//! A registry is a JSON manifest hosted at a URL with the format:
//! ```json
//! [
//!   {
//!     "name": "skill-name",
//!     "description": "What the skill does",
//!     "triggers": ["keyword1", "keyword2"],
//!     "url": "https://example.com/skills/skill-name.md",
//!     "version": "1.0.0",
//!     "author": "author-name"
//!   }
//! ]
//! ```

use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::tools::web_fetch::validate_url_for_ssrf;

/// A single entry in a skill registry manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub triggers: Vec<String>,
    pub url: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
}

/// Fetch and parse a skill registry manifest from a URL.
pub async fn fetch_registry(client: &Client, registry_url: &str) -> anyhow::Result<Vec<RegistryEntry>> {
    validate_url_for_ssrf(registry_url)
        .map_err(|e| anyhow::anyhow!("Registry URL blocked: {}", e))?;

    let response = client.get(registry_url).send().await?;
    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch registry: HTTP {}", response.status());
    }

    let entries: Vec<RegistryEntry> = response.json().await?;
    Ok(entries)
}

/// Search registry entries by query (matches name and description).
pub fn search_registry<'a>(entries: &'a [RegistryEntry], query: &str) -> Vec<&'a RegistryEntry> {
    let query_lower = query.to_lowercase();
    entries
        .iter()
        .filter(|e| {
            e.name.to_lowercase().contains(&query_lower)
                || e.description.to_lowercase().contains(&query_lower)
                || e.triggers.iter().any(|t| t.to_lowercase().contains(&query_lower))
        })
        .collect()
}

/// Fetch the skill markdown content from a registry entry's URL.
pub async fn fetch_skill_content(client: &Client, entry: &RegistryEntry) -> anyhow::Result<String> {
    validate_url_for_ssrf(&entry.url)
        .map_err(|e| anyhow::anyhow!("Skill URL blocked: {}", e))?;

    let response = client.get(&entry.url).send().await?;
    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch skill content: HTTP {}", response.status());
    }

    response.text().await.map_err(Into::into)
}

/// Format registry entries for display to the user.
pub fn format_registry_listing(entries: &[RegistryEntry]) -> String {
    if entries.is_empty() {
        return "No skills found in registry.".to_string();
    }

    let mut output = format!("**{} skills in registry:**\n", entries.len());
    for entry in entries {
        let version = entry.version.as_deref().unwrap_or("?");
        let author = entry.author.as_deref().unwrap_or("unknown");
        output.push_str(&format!(
            "- **{}** v{} by {}: {}\n",
            entry.name, version, author, entry.description
        ));
        if !entry.triggers.is_empty() {
            output.push_str(&format!("  triggers: {}\n", entry.triggers.join(", ")));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<RegistryEntry> {
        vec![
            RegistryEntry {
                name: "deploy".to_string(),
                description: "Deploy applications".to_string(),
                triggers: vec!["deploy".to_string(), "ship".to_string()],
                url: "https://example.com/skills/deploy.md".to_string(),
                version: Some("1.0.0".to_string()),
                author: Some("alice".to_string()),
            },
            RegistryEntry {
                name: "lint-code".to_string(),
                description: "Run linting tools".to_string(),
                triggers: vec!["lint".to_string(), "check".to_string()],
                url: "https://example.com/skills/lint.md".to_string(),
                version: Some("2.1.0".to_string()),
                author: Some("bob".to_string()),
            },
        ]
    }

    #[test]
    fn search_by_name() {
        let entries = sample_entries();
        let results = search_registry(&entries, "deploy");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "deploy");
    }

    #[test]
    fn search_by_description() {
        let entries = sample_entries();
        let results = search_registry(&entries, "linting");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "lint-code");
    }

    #[test]
    fn search_by_trigger() {
        let entries = sample_entries();
        let results = search_registry(&entries, "ship");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "deploy");
    }

    #[test]
    fn search_no_match() {
        let entries = sample_entries();
        let results = search_registry(&entries, "nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn format_listing() {
        let entries = sample_entries();
        let output = format_registry_listing(&entries);
        assert!(output.contains("2 skills in registry"));
        assert!(output.contains("deploy"));
        assert!(output.contains("lint-code"));
        assert!(output.contains("v1.0.0"));
        assert!(output.contains("alice"));
    }

    #[test]
    fn format_empty_listing() {
        let output = format_registry_listing(&[]);
        assert!(output.contains("No skills found"));
    }
}
