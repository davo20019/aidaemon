//! `escalate_to_parent`: a sub-agent's way to hand a decision it should not
//! make to the request's top-level agent.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::agent::coordination::{CallFence, Escalation};
use crate::traits::{Tool, ToolCallContext, ToolCallOutcome, ToolCapabilities};
use aidaemon_core::SourceTrust;

pub struct EscalateToParentTool;

#[derive(Deserialize)]
struct EscalateArgs {
    question: String,
    #[serde(default)]
    findings: String,
}

#[async_trait]
impl Tool for EscalateToParentTool {
    fn name(&self) -> &str {
        "escalate_to_parent"
    }

    fn description(&self) -> &str {
        "Ask the agent coordinating this request to decide something you should \
                not decide alone: work another sub-agent may already be doing, or a conflict with \
                its changes. Use it sparingly; the answer is one of proceed, stop or instead \
                (do what the note says), with a note."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "escalate_to_parent",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What you need decided."
                    },
                    "findings": {
                        "type": "string",
                        "description": "What you have found that the coordinator needs to decide."
                    }
                },
                "required": ["question"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: EscalateArgs = serde_json::from_str(arguments)?;
        let Some(call) = facts.extensions.get::<Arc<CallFence>>() else {
            anyhow::bail!("There is no coordinating agent for this run");
        };
        let answer = call
            .escalator
            .ask(
                &call.coordination,
                Escalation {
                    member: &call.member,
                    subject: "a question".to_string(),
                    recent_actions: args.findings,
                    proposed_call: String::new(),
                    facts: String::new(),
                    question: args.question,
                    options: crate::agent::coordination::QUESTION_OPTIONS.to_vec(),
                    untrusted: call.untrusted_seen,
                    voluntary: true,
                },
            )
            .await;
        let mut outcome = ToolCallOutcome::from_output(format!(
            "Decision: {}\nNote: {}",
            answer.decision.as_str(),
            answer.note
        ));
        // The answer was written from a transcript that may hold others' words.
        outcome.metadata.output_trust = Some(SourceTrust::Untrusted);
        Ok(outcome)
    }
}

impl crate::traits::ToolSemantics for EscalateToParentTool {
    fn dispatch_facts(&self) -> crate::traits::ToolDispatchFacts {
        crate::traits::ToolDispatchFacts {
            always_offered: true,
            ..Default::default()
        }
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: false,
            high_impact_write: false,
        }
    }
}
