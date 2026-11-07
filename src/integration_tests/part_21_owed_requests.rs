// ==================== Owed owner requests across the SDK path ====================
//
// Replays the failure that blocked a real deletion: an owner's request whose
// work was refused, followed by short continuations the classifier treats as
// new requests. The verifier must still be shown the owner's request and the
// owner's neighbouring messages, because "it" and "them" name the work.

/// Removes a synthetic fact (destructive, so the verifier checks it).
struct ForgetProbe;

#[async_trait::async_trait]
impl crate::traits::Tool for ForgetProbe {
    fn name(&self) -> &str {
        "forget_probe"
    }
    fn description(&self) -> &str {
        "Forgets a synthetic memory."
    }
    fn schema(&self) -> serde_json::Value {
        json!({
            "name": "forget_probe",
            "description": "Forgets a synthetic memory.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}}
        })
    }
    async fn call(&self, _arguments: &str) -> anyhow::Result<String> {
        Ok("forgotten".to_string())
    }
}

impl crate::traits::ToolSemantics for ForgetProbe {
    fn capabilities(&self) -> crate::traits::ToolCapabilities {
        crate::traits::ToolCapabilities {
            read_only: false,
            ..Default::default()
        }
    }
    fn call_semantics(&self, _arguments: &str) -> crate::traits::ToolCallSemantics {
        crate::traits::ToolCallSemantics::mutation_with(
            crate::traits::ToolMutationEffects::DESTRUCTIVE,
        )
    }
}

const OWNER_REMOVE_REQUEST: &str = "Remove it from memory";
const OWNER_REFERENT: &str = "Cloud pets?";

#[tokio::test]
async fn misclassified_continuation_still_shows_the_verifier_the_owed_request() {
    let session_id = "owed_request_replay";
    let provider = MockProvider::with_responses(vec![
        // Turn 1: the referent the owner names.
        MockProvider::text_response("CloudPets is a pet monitoring system."),
        // Turn 2: the owner's request. Untrusted read, then a destructive call
        // the verifier checks, then a reply that cannot finish the work.
        MockProvider::tool_call_response("injected_page", r#"{"url":"https://example.invalid/x"}"#),
        MockProvider::tool_call_response("forget_probe", r#"{"key":"other_projects"}"#),
        // The verifier refuses: the owed work did not happen.
        MockProvider::text_response(r#"{"consistent":false,"reason":"request does not identify the entry"}"#),
        MockProvider::text_response("I could not finish that yet."),
        // Turn 3: "Try again", classified as a new request.
        MockProvider::tool_call_response("injected_page", r#"{"url":"https://example.invalid/x"}"#),
        MockProvider::tool_call_response("forget_probe", r#"{"key":"side_project"}"#),
        MockProvider::text_response(r#"{"consistent":true,"reason":"owner asked"}"#),
        MockProvider::text_response("Done for this pass."),
    ])
    .with_task_assessments(vec![
        MockProvider::semantic_task_assessment("answer", false, false, &[], "new_request", "none"),
        MockProvider::semantic_task_assessment("answer", false, false, &[], "new_request", "none"),
        MockProvider::semantic_task_assessment("answer", false, false, &[], "new_request", "none"),
    ]);
    let tools: Vec<Arc<dyn crate::traits::Tool>> =
        vec![Arc::new(InjectedPage(String::new())), Arc::new(ForgetProbe)];
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(provider, tools, None)
        .await
        .unwrap();

    for text in [OWNER_REFERENT, OWNER_REMOVE_REQUEST, "Try again"] {
        let _ = harness
            .agent
            .handle_message(
                session_id,
                text,
                None,
                UserRole::Owner,
                ChannelContext::private("test"),
                None,
            )
            .await
            .unwrap();
    }

    let calls = harness.provider.call_log.lock().await;
    let verifier: Vec<_> = calls
        .iter()
        .filter(|call| is_clean_verifier_call(call))
        .collect();
    assert_eq!(verifier.len(), 2, "one clean check per forget call");
    let last = serde_json::Value::Array(verifier.last().unwrap().messages.clone()).to_string();
    assert!(
        last.contains(OWNER_REMOVE_REQUEST),
        "the owed request must reach the verifier: {last}"
    );
    assert!(
        last.contains(OWNER_REFERENT),
        "the referent the owner named must reach the verifier: {last}"
    );
}
