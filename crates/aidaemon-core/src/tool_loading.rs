//! Deferred tools: registered with the agent and callable, but their schemas
//! stay out of the model request until the run loads them.
//!
//! A tool registered with [`crate::AgentBuilder::deferred_tool_arc`] is
//! listed by name and one line in the description of the built-in
//! [`LOAD_TOOLS`] tool. Calling `load_tools` (or calling a deferred tool
//! directly) adds it to the agent's [`LoadedTools`], and its full schema is
//! sent from the next model request on. The host may share one
//! `LoadedTools` across a session's turns, so what was loaded once stays
//! loaded and the request's tool block stays byte-stable.

use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::CallEffects;
use crate::provider::ToolSpec;
use crate::tool::{Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult};

/// The built-in tool that loads deferred tools.
pub const LOAD_TOOLS: &str = "load_tools";

/// The deferred tools whose schemas the model is sent. Cheap to clone;
/// clones share the set.
#[derive(Debug, Clone, Default)]
pub struct LoadedTools(Arc<RwLock<BTreeSet<String>>>);

impl LoadedTools {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(name)
    }

    /// Mark `name` loaded. True when it was not loaded before.
    pub fn load(&self, name: &str) -> bool {
        self.0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_string())
    }

    /// Every loaded name, sorted.
    pub fn names(&self) -> Vec<String> {
        self.0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }
}

/// One deferred tool as the index lists it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeferredEntry {
    pub name: String,
    pub summary: String,
}

/// The first sentence of a description, capped: the index's one line.
pub(crate) fn one_line(description: &str) -> String {
    let first = description
        .split_terminator(['\n'])
        .next()
        .unwrap_or_default()
        .trim();
    let sentence = match first.find(". ") {
        Some(end) => &first[..=end],
        None => first,
    };
    let sentence = sentence.trim();
    if sentence.chars().count() <= 120 {
        sentence.to_string()
    } else {
        let cut: String = sentence.chars().take(117).collect();
        format!("{}...", cut.trim_end())
    }
}

/// `load_tools`: adds deferred tools to the run. Its description is the
/// index of every deferred tool, name-sorted.
pub(crate) struct LoadToolsTool {
    entries: Vec<DeferredEntry>,
    loaded: LoadedTools,
}

impl LoadToolsTool {
    pub(crate) fn new(mut entries: Vec<DeferredEntry>, loaded: LoadedTools) -> Self {
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Self { entries, loaded }
    }

    fn description(&self) -> String {
        let mut text = String::from(
            "Load more tools. These tools exist but their parameters are not shown yet; \
             load the ones a step needs by name (or find them with a query), then call \
             them. Loaded tools stay available.",
        );
        for entry in &self.entries {
            text.push_str(&format!("\n- {}: {}", entry.name, entry.summary));
        }
        text
    }

    /// Entries a free-text query matches: every word of the query appears
    /// in the tool's name or line (case-insensitive).
    fn matching(&self, query: &str) -> Vec<&DeferredEntry> {
        let words: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .map(str::to_lowercase)
            .collect();
        if words.is_empty() {
            return Vec::new();
        }
        let scored = |entry: &DeferredEntry| {
            let haystack =
                format!("{} {}", entry.name.replace('_', " "), entry.summary).to_lowercase();
            words
                .iter()
                .filter(|w| haystack.contains(w.as_str()))
                .count()
        };
        let best = self.entries.iter().map(scored).max().unwrap_or(0);
        if best == 0 {
            return Vec::new();
        }
        self.entries
            .iter()
            .filter(|entry| scored(entry) == best)
            .take(5)
            .collect()
    }
}

#[async_trait]
impl Tool for LoadToolsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: LOAD_TOOLS.to_string(),
            description: self.description(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names from the list to load"
                    },
                    "query": {
                        "type": "string",
                        "description": "Words describing the capability, when the name is unknown"
                    }
                },
                "additionalProperties": false
            }),
        }
    }

    /// Changes only which schemas the next request carries.
    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            idempotent: true,
            ..ToolAnnotations::READ_ONLY
        }
    }

    fn effects(&self, _arguments: &Value) -> CallEffects {
        CallEffects::none()
    }

    fn source_trust(&self, _arguments: &Value) -> crate::tool::SourceTrust {
        crate::tool::SourceTrust::Trusted
    }

    async fn call(&self, arguments: Value, _ctx: &ToolContext) -> ToolResult {
        let mut wanted: Vec<String> = arguments
            .get("names")
            .and_then(Value::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|name| name.trim().to_string())
                    .collect()
            })
            .unwrap_or_default();
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !query.trim().is_empty() {
            wanted.extend(self.matching(query).into_iter().map(|e| e.name.clone()));
        }
        if wanted.is_empty() {
            return Ok(ToolOutput::error(
                "Nothing to load: pass `names` from the list in this tool's description, or a \
                 `query`.",
            ));
        }
        let mut loaded = Vec::new();
        let mut already = Vec::new();
        let mut unknown = Vec::new();
        for name in wanted {
            if !self.entries.iter().any(|entry| entry.name == name) {
                if !unknown.contains(&name) {
                    unknown.push(name);
                }
            } else if self.loaded.load(&name) {
                loaded.push(name);
            } else if !already.contains(&name) && !loaded.contains(&name) {
                already.push(name);
            }
        }
        let mut lines = Vec::new();
        if !loaded.is_empty() {
            lines.push(format!(
                "Loaded: {}. Their parameters are available from your next step.",
                loaded.join(", ")
            ));
        }
        if !already.is_empty() {
            lines.push(format!("Already loaded: {}.", already.join(", ")));
        }
        if !unknown.is_empty() {
            lines.push(format!(
                "Not a deferred tool: {} (tools already listed can be called directly).",
                unknown.join(", ")
            ));
        }
        let output = ToolOutput::text(lines.join("\n"));
        Ok(if loaded.is_empty() && already.is_empty() {
            ToolOutput {
                is_error: true,
                ..output
            }
        } else {
            output
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> (LoadToolsTool, LoadedTools) {
        let loaded = LoadedTools::new();
        let entries = vec![
            DeferredEntry {
                name: "browser".into(),
                summary: "Drive a web browser.".into(),
            },
            DeferredEntry {
                name: "manage_config".into(),
                summary: "Read and change the daemon's config.".into(),
            },
        ];
        (LoadToolsTool::new(entries, loaded.clone()), loaded)
    }

    #[test]
    fn the_description_indexes_every_deferred_tool_by_name() {
        let (tool, _) = tool();
        let description = tool.spec().description;
        let browser = description.find("- browser:").unwrap();
        let config = description.find("- manage_config:").unwrap();
        assert!(browser < config, "{description}");
    }

    #[tokio::test]
    async fn names_and_queries_load_tools() {
        let (tool, loaded) = tool();
        let ctx = ToolContext::new("test:synthetic-session", "call-1");
        let out = tool
            .call(json!({"names": ["browser"]}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error && out.content.contains("Loaded: browser"));
        assert!(loaded.contains("browser"));
        let out = tool
            .call(json!({"query": "change config"}), &ctx)
            .await
            .unwrap();
        assert!(
            out.content.contains("Loaded: manage_config"),
            "{}",
            out.content
        );
        let out = tool
            .call(json!({"names": ["browser"]}), &ctx)
            .await
            .unwrap();
        assert!(out.content.contains("Already loaded: browser"));
        let out = tool.call(json!({"names": ["nope"]}), &ctx).await.unwrap();
        assert!(out.is_error && out.content.contains("Not a deferred tool: nope"));
    }

    #[test]
    fn one_line_keeps_the_first_sentence() {
        assert_eq!(one_line("Fetch a page. Then more."), "Fetch a page.");
        assert_eq!(one_line("No period"), "No period");
        assert!(one_line(&"x".repeat(300)).ends_with("..."));
    }
}
