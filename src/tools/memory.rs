use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::traits::{StateStore, Tool};

pub struct RememberFactTool {
    state: Arc<dyn StateStore>,
}

impl RememberFactTool {
    pub fn new(state: Arc<dyn StateStore>) -> Self {
        Self { state }
    }
}

#[derive(Deserialize)]
struct RememberArgs {
    category: String,
    key: String,
    value: String,
}

#[async_trait]
impl Tool for RememberFactTool {
    fn name(&self) -> &str {
        "remember_fact"
    }

    fn description(&self) -> &str {
        "Store a fact about the user or environment for long-term memory"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "remember_fact",
            "description": "Store a fact about the user or environment for long-term memory. Facts are injected into your system prompt on every request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category for the fact (e.g. 'user', 'preference', 'project')"
                    },
                    "key": {
                        "type": "string",
                        "description": "A unique key for this fact within the category"
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact to remember"
                    }
                },
                "required": ["category", "key", "value"]
            }
        })
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: RememberArgs = serde_json::from_str(arguments)?;
        self.state
            .upsert_fact(&args.category, &args.key, &args.value, "agent")
            .await?;
        Ok(format!(
            "Remembered: [{}] {} = {}",
            args.category, args.key, args.value
        ))
    }
}
