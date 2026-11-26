//! One real call that shows a model can run a turn here: it answers with a
//! tool call, as the agent's calls need. Made through the provider's own
//! backend (its quota guard), so a refusal that only time or the owner can
//! cure (no credit, a rejected key, a used-up quota) is remembered for every
//! caller.

use std::time::Duration;

use aidaemon_core::{Message, ModelRequest, Provider, RequestOptions, ToolChoice, ToolSpec};
use serde_json::json;

/// How long a trial may take.
pub const TRIAL_WITHIN: Duration = Duration::from_secs(90);

/// Ask `model` of `backend` to call one tool. `Err`: why it can't run a
/// turn, in plain words.
pub async fn trial(backend: &dyn Provider, model: &str) -> Result<(), String> {
    let request = ModelRequest {
        model: model.to_string(),
        messages: vec![Message::user(
            "This checks that a model works with this app. Call the `ready` tool \
             with ok set to true.",
        )],
        tools: vec![ToolSpec {
            name: "ready".to_string(),
            description: "Report that you are ready.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": false
            }),
        }],
        tool_choice: ToolChoice::Auto,
        options: RequestOptions {
            single_attempt: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let response = tokio::time::timeout(TRIAL_WITHIN, backend.complete(&request))
        .await
        .map_err(|_| format!("{model} did not answer within {TRIAL_WITHIN:?}"))?
        .map_err(|error| error.to_string())?;
    if !response.tool_calls.iter().any(|call| call.name == "ready") {
        return Err(format!("{model} answered without calling a tool"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::testing::{ScriptedProvider, call, text, tool_calls};

    #[tokio::test]
    async fn a_model_that_calls_the_tool_passes_and_one_that_only_talks_does_not() {
        let calls = ScriptedProvider::new([tool_calls([call("c1", "ready", json!({"ok": true}))])]);
        assert_eq!(trial(&calls, "synthetic-model").await, Ok(()));
        let talks = ScriptedProvider::new([text("ready")]);
        assert!(trial(&talks, "synthetic-model").await.is_err());
    }
}
