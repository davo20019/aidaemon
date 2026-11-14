//! `request_feature`: the agent asks the owner to turn on a feature that is
//! off (browsing websites, voice notes, using the computer's screen) when
//! the work in front of it needs one. The owner's apps show a card that
//! turns it on (and walks the system's permission prompt where there is
//! one); nothing changes until the owner presses it.
//!
//! Offered only while at least one of those features is off and an app can
//! show the card. The call itself changes nothing, so it declares no effect.

use std::sync::Arc;

use aidaemon_client_protocol::Feature;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ToolCallContext, ToolCallOutcome, ToolCallSemantics, ToolCapabilities, ToolRole, ToolSemantics,
};

/// Where requests go: the running Client Gateway, or a test double.
pub trait FeatureRequests: Send + Sync {
    /// The features that are off now, which the owner can turn on.
    fn off(&self) -> Vec<Feature>;
    /// Show every app the request.
    fn request(&self, session_id: &str, feature: Feature, reason: &str) -> Result<(), String>;
}

pub struct FeatureRequestTool {
    requests: Arc<dyn FeatureRequests>,
}

impl FeatureRequestTool {
    pub fn new(requests: Arc<dyn FeatureRequests>) -> Self {
        Self { requests }
    }
}

fn name_of(feature: Feature) -> &'static str {
    match feature {
        Feature::Browser => "browser",
        Feature::Voice => "voice",
        Feature::ComputerUse => "computer_use",
        Feature::Unknown => "unknown",
    }
}

fn words(feature: Feature) -> &'static str {
    match feature {
        Feature::Browser => "browsing websites",
        Feature::Voice => "understanding voice notes",
        Feature::ComputerUse => "seeing and using this computer's screen",
        Feature::Unknown => "that feature",
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Args {
    feature: Feature,
    reason: String,
}

#[async_trait]
impl Tool for FeatureRequestTool {
    fn name(&self) -> &str {
        "request_feature"
    }

    fn description(&self) -> &str {
        "Ask the owner to turn on a feature that is off when the task needs it: `browser` \
         (open and read websites, fill forms), `voice` (transcribe voice notes) or \
         `computer_use` (see and use this computer's screen). Their app shows a card with a \
         button that turns it on. It is not available in this turn: say what you will do once \
         it is on, or do the task another way."
    }

    fn schema(&self) -> Value {
        let off: Vec<&str> = self.requests.off().into_iter().map(name_of).collect();
        json!({
            "name": "request_feature",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature": {
                        "type": "string",
                        "enum": off,
                        "description": "The feature the task needs"
                    },
                    "reason": {
                        "type": "string",
                        "description": "What it is for, in a few words the owner will read, e.g. \"to read the venue's booking page\""
                    }
                },
                "required": ["feature", "reason"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args: Args = serde_json::from_str(arguments)?;
        if !self.requests.off().contains(&args.feature) {
            anyhow::bail!(
                "`{}` is already on or cannot be turned on here",
                name_of(args.feature)
            );
        }
        let session = if facts.session_id.is_empty() {
            facts.channel_id.clone().unwrap_or_default()
        } else {
            facts.session_id.clone()
        };
        self.requests
            .request(&session, args.feature, args.reason.trim())
            .map_err(|why| anyhow::anyhow!("The request could not be shown: {why}"))?;
        Ok(ToolCallOutcome::from_output(format!(
            "The owner's app now offers to turn on {}. It is not on yet and not in this turn: \
             tell them what you will do once it is, or do the task another way.",
            words(args.feature)
        )))
    }
}

impl ToolSemantics for FeatureRequestTool {
    fn tool_role(&self) -> ToolRole {
        ToolRole::Action
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, _arguments: &str) -> ToolCallSemantics {
        ToolCallSemantics::administrative()
    }

    /// A card on the owner's own app: no file, process, credential or
    /// outside host is touched, and nothing turns on until the owner does it.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::none()
    }

    fn call_annotations_in(
        &self,
        _arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::ToolAnnotations {
        aidaemon_core::ToolAnnotations {
            read_only: false,
            destructive: false,
            idempotent: true,
            egress: false,
        }
    }

    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &aidaemon_tools::semantics::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Trusted
    }

    fn offered_in_session(&self, _session_id: &str) -> bool {
        !self.requests.off().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Requests {
        sent: Mutex<Vec<(String, Feature, String)>>,
    }

    impl FeatureRequests for Requests {
        fn off(&self) -> Vec<Feature> {
            vec![Feature::Browser]
        }
        fn request(&self, session_id: &str, feature: Feature, reason: &str) -> Result<(), String> {
            self.sent
                .lock()
                .unwrap()
                .push((session_id.into(), feature, reason.into()));
            Ok(())
        }
    }

    #[tokio::test]
    async fn only_a_feature_that_is_off_is_asked_for() {
        let requests = Arc::new(Requests::default());
        let tool = FeatureRequestTool::new(requests.clone());
        assert_eq!(
            tool.schema()["parameters"]["properties"]["feature"]["enum"],
            json!(["browser"])
        );
        let facts = ToolCallContext {
            session_id: "app:s1".into(),
            ..ToolCallContext::default()
        };
        tool.execute(r#"{"feature":"browser","reason":"to read a page"}"#, &facts)
            .await
            .unwrap();
        assert!(
            tool.execute(r#"{"feature":"voice","reason":"x"}"#, &facts)
                .await
                .is_err()
        );
        assert_eq!(
            requests.sent.lock().unwrap().as_slice(),
            &[(
                "app:s1".to_string(),
                Feature::Browser,
                "to read a page".to_string()
            )]
        );
        assert!(tool.offered_in_session("app:s1"));
    }
}
