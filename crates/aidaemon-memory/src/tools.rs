//! [`RememberTool`] and [`RecallTool`]: explicit memory access for the
//! model, as ordinary `aidaemon_core::Tool`s.
//!
//! The scope always comes from the trusted `ToolContext::session_id`
//! through the scope resolver, never from the model's arguments, so a
//! model cannot read or write another user's memory.

use std::sync::Arc;

use aidaemon_core::{
    SourceTrust, Tool, ToolAnnotations, ToolContext, ToolOutput, ToolResult, ToolSpec,
};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::memory::{Memory, ScopeResolver};
use crate::types::{NewFact, Provenance, UpsertOutcome};

fn resolve(memory: &Memory, resolver: &Option<ScopeResolver>, session_id: &str) -> String {
    match resolver {
        Some(f) => f(session_id),
        None => memory.scope_for(session_id),
    }
}

/// Lets the model store a fact. Not read-only, not destructive, and
/// idempotent: the same arguments reinforce instead of duplicating.
pub struct RememberTool {
    memory: Memory,
    resolver: Option<ScopeResolver>,
}

impl RememberTool {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            resolver: None,
        }
    }

    /// Override the memory's scope resolver for this tool.
    pub fn scope_resolver(mut self, f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        self.resolver = Some(Arc::new(f));
        self
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "remember".into(),
            description: "Save a durable fact about the user or their world to long-term memory \
                          (preferences, people, projects, decisions). Reusing a category and key \
                          replaces the old value; the old one is kept as history. Do not save \
                          secrets or transient task details."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Short lowercase noun, e.g. user, preference, person, project"},
                    "key": {"type": "string", "description": "snake_case attribute name, e.g. favorite_color"},
                    "value": {"type": "string", "description": "The fact itself"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["category", "key", "value"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
            egress: false,
        }
    }

    /// The owner's own memory store answers.
    fn source_trust(&self, _arguments: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let field = |name: &str| arguments.get(name).and_then(Value::as_str).unwrap_or("");
        let scope = resolve(&self.memory, &self.resolver, &ctx.session_id);
        let fact = NewFact::new(scope, field("category"), field("key"), field("value"))
            .confidence(
                arguments
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0) as f32,
            )
            .source(Provenance {
                run_id: Some(ctx.run_id.to_string()),
                session_id: Some(ctx.session_id.clone()),
                excerpt: None,
            });
        let audience = self.memory.audience_for(&ctx.session_id);
        match self
            .memory
            .remember_with(fact, &audience.access_for_writes())
            .await
        {
            // The earlier value may be something this audience cannot see.
            Ok(UpsertOutcome::Superseded { id, .. }) if !audience.sees_everything() => {
                Ok(ToolOutput::text("Saved to memory.")
                    .with_details(json!({ "kind": "superseded", "id": id })))
            }
            Ok(outcome) => {
                let text = match &outcome {
                    UpsertOutcome::Inserted { .. } => "Saved to memory.".to_string(),
                    UpsertOutcome::Reinforced { .. } => "Already in memory; refreshed.".to_string(),
                    UpsertOutcome::Superseded { previous_value, .. } => {
                        format!(
                            "Saved to memory, replacing the previous value \"{previous_value}\"."
                        )
                    }
                };
                Ok(ToolOutput::text(text).with_details(json!(outcome)))
            }
            Err(crate::MemoryError::Invalid(reason)) => Ok(ToolOutput::error(reason)),
            Err(other) => Err(other.to_string().into()),
        }
    }
}

/// Lets the model search memory. Read-only.
pub struct RecallTool {
    memory: Memory,
    resolver: Option<ScopeResolver>,
    max_limit: usize,
}

impl RecallTool {
    pub fn new(memory: Memory) -> Self {
        Self {
            memory,
            resolver: None,
            max_limit: 20,
        }
    }

    pub fn scope_resolver(mut self, f: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        self.resolver = Some(Arc::new(f));
        self
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "recall".into(),
            description: "Search long-term memory for facts and past conversations relevant to \
                          a query. Returns the best matches with relevance scores."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": self.max_limit}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        }
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::READ_ONLY
    }

    /// Facts from the owner's own memory store.
    fn source_trust(&self, _arguments: &Value) -> SourceTrust {
        SourceTrust::Trusted
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if query.is_empty() {
            return Ok(ToolOutput::error("query must not be empty"));
        }
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(5)
            .clamp(1, self.max_limit);
        let scope = resolve(&self.memory, &self.resolver, &ctx.session_id);
        let audience = self.memory.audience_for(&ctx.session_id);
        let facts = self
            .memory
            .recall_facts_for(&scope, query, limit, &audience)
            .await
            .map_err(|e| e.to_string())?;
        let episodes = self
            .memory
            .recall_episodes_for(&scope, query, limit.div_ceil(2), &audience)
            .await
            .map_err(|e| e.to_string())?;
        if facts.is_empty() && episodes.is_empty() {
            return Ok(ToolOutput::text("No matching memories."));
        }
        let mut text = String::new();
        for f in &facts {
            text.push_str(&format!(
                "- [{}] {}: {} (score {:.2})\n",
                f.fact.category, f.fact.key, f.fact.value, f.score
            ));
        }
        for e in &episodes {
            text.push_str(&format!(
                "- [episode {}] {} (score {:.2})\n",
                e.episode.created_at.format("%Y-%m-%d"),
                e.episode.summary,
                e.score
            ));
        }
        let details = json!({
            "facts": facts.iter().map(|f| json!({
                "id": f.fact.id, "category": f.fact.category, "key": f.fact.key,
                "value": f.fact.value, "score": f.score,
            })).collect::<Vec<_>>(),
            "episodes": episodes.iter().map(|e| json!({
                "id": e.episode.id, "summary": e.episode.summary, "score": e.score,
            })).collect::<Vec<_>>(),
        });
        Ok(ToolOutput::text(text.trim_end().to_string()).with_details(details))
    }
}
