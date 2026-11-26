//! One real call that shows a model can run a turn here: it answers with a
//! tool call, as the agent's calls need. Made through the provider's own
//! backend (its quota guard), so a refusal that only time or the owner can
//! cure (no credit, a rejected key, a used-up quota) is remembered for every
//! caller.

use std::time::Duration;

use aidaemon_core::{
    Message, ModelRequest, Provider, ProviderError, RequestOptions, ToolChoice, ToolSpec,
};
use serde_json::json;

/// How long a trial may take.
pub const TRIAL_WITHIN: Duration = Duration::from_secs(90);

/// Why a trial failed.
#[derive(Debug, Clone, PartialEq)]
pub enum TrialFailure {
    /// The provider refused the call.
    Refused(ProviderError),
    /// It answered without calling the tool.
    NoToolCall,
    /// It did not answer within [`TRIAL_WITHIN`].
    TimedOut,
}

impl TrialFailure {
    /// The provider refused the account, not the model or the request: it
    /// cannot pay (whatever smaller output it could still afford), its key
    /// was rejected, or its whole quota is used up.
    pub fn account_refused(&self) -> bool {
        matches!(
            self,
            TrialFailure::Refused(
                ProviderError::Billing { .. }
                    | ProviderError::Auth(_)
                    | ProviderError::QuotaExhausted {
                        per_model: false,
                        ..
                    }
            )
        )
    }
}

impl std::fmt::Display for TrialFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrialFailure::Refused(error) => write!(f, "{error}"),
            TrialFailure::NoToolCall => write!(f, "it answered without calling a tool"),
            TrialFailure::TimedOut => write!(f, "it did not answer within {TRIAL_WITHIN:?}"),
        }
    }
}

/// Ask `model` of `backend` to call one tool.
pub async fn trial(backend: &dyn Provider, model: &str) -> Result<(), TrialFailure> {
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
        .map_err(|_| TrialFailure::TimedOut)?
        .map_err(TrialFailure::Refused)?;
    if !response.tool_calls.iter().any(|call| call.name == "ready") {
        return Err(TrialFailure::NoToolCall);
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
        assert_eq!(
            trial(&talks, "synthetic-model").await,
            Err(TrialFailure::NoToolCall)
        );
    }

    /// An account that can still afford a little output is refused all the
    /// same: no turn fits in it.
    #[tokio::test]
    async fn an_account_that_cannot_pay_is_refused_whatever_it_could_afford() {
        let broke = ScriptedProvider::with_results([Err(ProviderError::Billing {
            message: "synthetic: can only afford a little".into(),
            affordable_tokens: Some(100),
        })]);
        let failure = trial(&broke, "synthetic-model").await.unwrap_err();
        assert!(failure.account_refused());
        assert!(!TrialFailure::NoToolCall.account_refused());
    }
}
