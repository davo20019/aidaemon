use std::collections::HashMap;
use async_trait::async_trait;
use serde_json::{json, Value};
use crate::config::RemoteAgentConfig;
use crate::traits::Tool;

pub struct RemoteAgentTool {
    agents: HashMap<String, RemoteAgentConfig>,
    client: reqwest::Client,
}

impl RemoteAgentTool {
    pub fn new(agents: HashMap<String, RemoteAgentConfig>) -> Self {
        Self {
            agents,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Tool for RemoteAgentTool {
    fn name(&self) -> &str {
        "remote_agent"
    }

    fn description(&self) -> &str {
        "Send a message to a remote AI agent to perform a task or ask a question. \
        Use this to coordinate with other agents (e.g., on AWS, GCP)."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "remote_agent",
            "description": "Send a message to a remote AI agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "The name of the agent to contact."
                    },
                    "message": {
                        "type": "string",
                        "description": "The message or task for the remote agent."
                    }
                },
                "required": ["agent", "message"]
            }
        })
    }

    async fn call(&self, args: &str) -> anyhow::Result<String> {
        let params: Value = serde_json::from_str(args)?;
        let agent_name = params["agent"].as_str().ok_or_else(|| anyhow::anyhow!("Missing 'agent' parameter"))?;
        let message = params["message"].as_str().ok_or_else(|| anyhow::anyhow!("Missing 'message' parameter"))?;
        let session_id = params["_session_id"].as_str().unwrap_or("unknown-session");

        let config = self.agents.get(agent_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown agent: '{}'. Available agents: {:?}", agent_name, self.agents.keys()))?;

        // Assume the config.url is the base URL (e.g. http://localhost:8080 or http://myserver.com)
        // We append /agent/message
        let url = format!("{}/agent/message", config.url.trim_end_matches('/'));

        let mut req = self.client.post(&url)
            .json(&json!({
                "session_id": session_id,
                "message": message
            }));

        if let Some(token) = &config.token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        let resp = req.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Remote agent returned error {}: {}", status, text);
        }

        let resp_json: Value = resp.json().await?;
        let response_text = resp_json["response"].as_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format from remote agent"))?;

        Ok(response_text.to_string())
    }
}
