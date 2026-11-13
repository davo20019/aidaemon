use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use aidaemon_tools::OutcomeTool as Tool;
use aidaemon_tools::semantics::{
    ApprovalCall, OwnerApprovalNeed, ToolCallContext, ToolCapabilities,
};

use crate::records::{Fact, FactPrivacy};
use crate::stores::FactStore;

pub struct ShareMemoryTool {
    state: Arc<dyn FactStore>,
}

impl ShareMemoryTool {
    pub fn new(state: Arc<dyn FactStore>) -> Self {
        Self { state }
    }

    /// The active fact a call names, if any.
    async fn find_fact(&self, args: &ShareArgs) -> anyhow::Result<Option<Fact>> {
        let facts = self.state.get_facts(Some(&args.category)).await?;
        Ok(facts
            .into_iter()
            .find(|f| f.key == args.key && f.superseded_at.is_none()))
    }

    /// The owner approval for sharing `fact`. Making a memory visible across
    /// all channels always asks; the subject is the exact arguments approved.
    fn approval_for(args: &ShareArgs, fact: &Fact) -> OwnerApprovalNeed {
        OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
            card: None,
            effects: Vec::new(),
            summary: format!(
                "Share memory globally: [{}] {} = \"{}\"",
                args.category,
                args.key,
                crate::text::truncate_chars(&fact.value, 83)
            ),
            subject: json!({"category": args.category, "key": args.key}).to_string(),
            risk: aidaemon_core::ApprovalRisk::High,
            warnings: vec!["This will make this memory visible across all channels".to_string()],
        })
    }
}

#[derive(Deserialize)]
struct ShareArgs {
    category: String,
    key: String,
}

#[async_trait]
impl Tool for ShareMemoryTool {
    fn name(&self) -> &str {
        "share_memory"
    }

    fn description(&self) -> &str {
        "Make a memory global: visible in every channel and chat, permanently. Use only when the owner approves sharing it."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "share_memory",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of the fact to share"
                    },
                    "key": {
                        "type": "string",
                        "description": "The key of the fact to share"
                    }
                },
                "required": ["category", "key"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &aidaemon_tools::semantics::ToolCallContext,
    ) -> anyhow::Result<aidaemon_tools::semantics::ToolCallOutcome> {
        let facts = facts.clone();

        self.run(arguments, &facts)
            .await
            .map(aidaemon_tools::semantics::ToolCallOutcome::from_output)
    }
}

impl ShareMemoryTool {
    /// Run one call with the dispatcher's facts (`call`). The kernel asked
    /// the owner before this runs; a call that reaches the share without the
    /// kernel's judgment is refused.
    pub async fn run(&self, arguments: &str, call: &ToolCallContext) -> anyhow::Result<String> {
        let args: ShareArgs = serde_json::from_str(arguments)?;

        match self.find_fact(&args).await? {
            Some(f) => {
                if call.kernel_approved.is_none() {
                    anyhow::bail!(
                        "share_memory requires approval through the agent kernel; it was called \
                         outside an approved agent call"
                    );
                }
                self.state
                    .update_fact_privacy(f.id, FactPrivacy::Global)
                    .await?;
                Ok(format!(
                    "Memory shared: [{}] {} is now globally accessible.",
                    args.category, args.key
                ))
            }
            None => Ok(format!(
                "No active fact found with category '{}' and key '{}'.",
                args.category, args.key
            )),
        }
    }
}

#[async_trait]
impl aidaemon_tools::semantics::ToolSemantics for ShareMemoryTool {
    /// Sharing an existing fact asks the owner, whatever the session. A call
    /// that names no active fact (or cannot be read) changes nothing and
    /// needs no card.
    async fn owner_approval_request(
        &self,
        call: &ApprovalCall<'_>,
        _facts: &ToolCallContext,
    ) -> OwnerApprovalNeed {
        if call.refused {
            return OwnerApprovalNeed::NotNeeded;
        }
        let Ok(args) = serde_json::from_value::<ShareArgs>(call.arguments.clone()) else {
            return OwnerApprovalNeed::NotNeeded;
        };
        match self.find_fact(&args).await {
            Ok(Some(fact)) => Self::approval_for(&args, &fact),
            Ok(None) => OwnerApprovalNeed::NotNeeded,
            // The lookup failed: ask with what the call names rather than
            // waving it through (the run reads the store again).
            Err(_) => OwnerApprovalNeed::Ask(aidaemon_core::OwnerApprovalRequest {
                card: None,
                effects: Vec::new(),
                summary: format!("Share memory globally: [{}] {}", args.category, args.key),
                subject: json!({"category": args.category, "key": args.key}).to_string(),
                risk: aidaemon_core::ApprovalRisk::High,
                warnings: vec![
                    "This will make this memory visible across all channels".to_string(),
                ],
            }),
        }
    }

    /// Sharing widens who can see a memory. It has no path, host or process
    /// to name, so its effects stay undeclared: a decline binds the exact
    /// call (by its subject), and later opaque calls are asked again.
    fn call_effects(&self, _arguments: &str) -> aidaemon_core::CallEffects {
        aidaemon_core::CallEffects::unknown()
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: false,
            external_side_effect: false,
            needs_approval: true,
            idempotent: false,
            high_impact_write: true,
        }
    }
}

/// A fact store for tests: the facts it was given, and the privacy changes
/// made to them.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    pub(crate) struct SyntheticFacts {
        pub(crate) facts: Mutex<Vec<Fact>>,
    }

    impl SyntheticFacts {
        pub(crate) fn with(category: &str, key: &str, value: &str) -> Self {
            let now = chrono::Utc::now();
            Self {
                facts: Mutex::new(vec![Fact {
                    id: 1,
                    category: category.to_string(),
                    key: key.to_string(),
                    value: value.to_string(),
                    source: "owner".to_string(),
                    created_at: now,
                    updated_at: now,
                    superseded_at: None,
                    recall_count: 0,
                    last_recalled_at: None,
                    channel_id: Some("telegram:synthetic-channel-1".to_string()),
                    privacy: FactPrivacy::Channel,
                    first_seen_at: None,
                    source_excerpt: None,
                }]),
            }
        }

        pub(crate) fn privacy(&self, id: i64) -> Option<FactPrivacy> {
            self.facts
                .lock()
                .unwrap()
                .iter()
                .find(|fact| fact.id == id)
                .map(|fact| fact.privacy)
        }
    }

    #[async_trait]
    impl FactStore for SyntheticFacts {
        async fn upsert_fact_with_provenance(
            &self,
            _category: &str,
            _key: &str,
            _value: &str,
            _source: &str,
            _channel_id: Option<&str>,
            _privacy: FactPrivacy,
            _first_seen_at: Option<chrono::DateTime<chrono::Utc>>,
            _source_excerpt: Option<&str>,
        ) -> anyhow::Result<()> {
            anyhow::bail!("not used in these tests")
        }

        async fn get_facts(&self, category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
            Ok(self
                .facts
                .lock()
                .unwrap()
                .iter()
                .filter(|fact| category.is_none_or(|c| fact.category == c))
                .cloned()
                .collect())
        }

        async fn update_fact_privacy(
            &self,
            fact_id: i64,
            privacy: FactPrivacy,
        ) -> anyhow::Result<()> {
            for fact in self.facts.lock().unwrap().iter_mut() {
                if fact.id == fact_id {
                    fact.privacy = privacy;
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::SyntheticFacts;
    use super::*;
    use crate::full::tools::test_support::ToolCalls as _;
    use aidaemon_tools::semantics::ToolSemantics;

    fn tool() -> (Arc<SyntheticFacts>, ShareMemoryTool) {
        let facts = Arc::new(SyntheticFacts::with(
            "preference",
            "report_format",
            "short summaries for Alice Rivera",
        ));
        (facts.clone(), ShareMemoryTool::new(facts))
    }

    #[tokio::test]
    async fn sharing_an_existing_fact_asks_with_its_exact_arguments() {
        let (_, tool) = tool();
        let args = json!({"category": "preference", "key": "report_format"});
        let OwnerApprovalNeed::Ask(request) = tool
            .owner_approval_request(&ApprovalCall::as_sent(&args), &ToolCallContext::default())
            .await
        else {
            panic!("sharing a fact must ask");
        };
        assert_eq!(
            request.summary,
            "Share memory globally: [preference] report_format = \"short summaries for Alice Rivera\""
        );
        assert_eq!(request.risk, aidaemon_core::ApprovalRisk::High);
        let subject: Value = serde_json::from_str(&request.subject).unwrap();
        assert_eq!(subject, args);

        let missing = json!({"category": "preference", "key": "absent"});
        assert!(matches!(
            tool.owner_approval_request(
                &ApprovalCall::as_sent(&missing),
                &ToolCallContext::default()
            )
            .await,
            OwnerApprovalNeed::NotNeeded
        ));
        let refused = ApprovalCall {
            refused: true,
            ..ApprovalCall::as_sent(&args)
        };
        assert!(matches!(
            tool.owner_approval_request(&refused, &ToolCallContext::default())
                .await,
            OwnerApprovalNeed::NotNeeded
        ));
    }

    #[tokio::test]
    async fn sharing_without_the_kernel_is_refused() {
        let (facts, tool) = tool();
        let args = r#"{"category":"preference","key":"report_format"}"#;
        let err = tool.call(args).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("requires approval through the agent kernel"),
            "{err}"
        );
        assert_eq!(facts.privacy(1), Some(FactPrivacy::Channel));

        let shared = tool
            .run(
                args,
                &ToolCallContext {
                    kernel_approved: Some(aidaemon_tools::semantics::ApprovalScope::Call),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(shared.contains("now globally accessible"), "{shared}");
        assert_eq!(facts.privacy(1), Some(FactPrivacy::Global));
    }

    #[test]
    fn sharing_keeps_its_effects_declared() {
        let (_, tool) = tool();
        let effects = tool.call_effects(r#"{"category":"preference","key":"report_format"}"#);
        assert!(!effects.complete);
        assert_ne!(effects, aidaemon_core::CallEffects::none());
    }
}
