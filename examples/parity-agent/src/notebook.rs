//! A memory system of our own, in a few lines: a `remember` tool appends a
//! line to a file, and every line is shown to the model before it answers.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aidaemon_sdk::policies::{MemoryInjection, MemoryParts, MemorySystem, MemoryTurn, Recall};
use aidaemon_sdk::{FnTool, SourceTrust, ToolAnnotations, ToolOutput, ToolSpec};
use serde_json::{Value, json};

#[derive(Clone)]
pub struct Notebook(PathBuf);

impl Notebook {
    pub fn open(dir: &Path) -> Self {
        Self(dir.join("notebook.txt"))
    }

    fn lines(&self) -> Vec<String> {
        let text = std::fs::read_to_string(&self.0).unwrap_or_default();
        text.lines().map(str::to_string).collect()
    }

    fn write(&self, line: String) -> std::io::Result<()> {
        let mut lines = self.lines();
        lines.push(line);
        std::fs::write(&self.0, lines.join("\n") + "\n")
    }
}

#[async_trait::async_trait]
impl Recall for Notebook {
    async fn recall(&self, _session_id: &str, _query: &str, limit: usize) -> Vec<String> {
        self.lines().into_iter().rev().take(limit).collect()
    }
}

#[async_trait::async_trait]
impl MemorySystem for Notebook {
    async fn turn(&self, turn: &MemoryTurn) -> MemoryParts {
        let mut parts = MemoryParts::default();
        if turn.read {
            parts
                .hooks
                .push(Arc::new(MemoryInjection::new(self.clone())));
        }
        if turn.write {
            let spec = ToolSpec {
                name: "remember".into(),
                description: "Write a lasting fact about the user in the notebook.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "key": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["category", "key", "value"],
                    "additionalProperties": false
                }),
            };
            let notebook = self.clone();
            let remember = FnTool::new(spec, ToolAnnotations::WRITE, move |args: Value| {
                let notebook = notebook.clone();
                async move {
                    let field = |name: &str| args[name].as_str().unwrap_or_default().to_string();
                    notebook
                        .write(format!("{}: {}", field("key"), field("value")))
                        .map_err(|err| aidaemon_sdk::ToolError(err.to_string()))?;
                    Ok(ToolOutput::text("written"))
                }
            })
            .with_trust(SourceTrust::Trusted);
            parts.tools.push(Arc::new(remember));
        }
        parts
    }

    fn guidance(&self) -> Option<String> {
        Some("Save lasting facts about the user with `remember`.".into())
    }
}
