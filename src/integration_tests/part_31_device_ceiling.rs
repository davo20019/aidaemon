// ==========================================================================
// The device ceiling
//
// A message from a device capped at Medium (an e-ink reader on a public
// bridge, typed by the client gateway as `ChannelContext::sender_ceiling`)
// caps the run it starts or steers: a call above Medium waits for the
// owner on a High card offered once only, even where nothing would have
// asked before, and an answer there leaves no lasting grant.
// ==========================================================================

const CEILING_SESSION: &str = "app:synthetic-ceiling-1";

/// A tool nothing asks about today: no approval rule and the policy's mode
/// `Never`. Its effects are not declared, so a capped run judges it above
/// Medium.
fn opaque_tool() -> Arc<dyn crate::traits::Tool> {
    Arc::new(MockTool::new(
        "post_note",
        "Posts a note somewhere",
        "note posted",
    ))
}

fn capped(mut channel_ctx: ChannelContext) -> ChannelContext {
    channel_ctx.sender_ceiling = Some(aidaemon_channels::context::SenderCeiling::Medium);
    channel_ctx
}

async fn ceiling_harness(
    script: Vec<crate::traits::ProviderResponse>,
    extra: Vec<Arc<dyn crate::traits::Tool>>,
    answers: Vec<Option<crate::types::ApprovalResponse>>,
) -> (
    crate::testing::TestHarness,
    crate::testing::ScriptedApprover,
) {
    let mut tools = vec![opaque_tool()];
    tools.extend(extra);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(
        MockProvider::with_responses(script),
        tools,
        None,
    )
    .await
    .unwrap();
    let (broker, approver) = crate::testing::ScriptedApprover::start(answers);
    harness.agent.owner_approvals().set_broker(broker);
    (harness, approver)
}

async fn ceiling_turn(harness: &crate::testing::TestHarness, channel_ctx: ChannelContext) {
    harness
        .agent
        .handle_message(
            CEILING_SESSION,
            "Post the synthetic notes for Jordan Lee",
            None,
            UserRole::Owner,
            channel_ctx,
            None,
        )
        .await
        .unwrap();
}

fn post_note() -> crate::traits::ProviderResponse {
    MockProvider::tool_call_response("post_note", "{}")
}

/// Uncapped, the call runs without a card; capped, the same call waits for
/// the owner on a High, once-only card, and "Allow for this session" on it
/// covers that call alone: the next one asks again.
#[tokio::test]
async fn a_capped_turn_asks_where_nothing_asked_and_grants_nothing_lasting() {
    let script = || {
        vec![
            post_note(),
            post_note(),
            MockProvider::text_response("Posted the notes."),
        ]
    };
    let (open, open_approver) = ceiling_harness(script(), Vec::new(), Vec::new()).await;
    ceiling_turn(&open, ChannelContext::private("app")).await;
    assert_eq!(open_approver.card_count(), 0, "nothing asks uncapped");

    let (harness, approver) = ceiling_harness(
        script(),
        Vec::new(),
        vec![
            Some(crate::types::ApprovalResponse::AllowSession),
            Some(crate::types::ApprovalResponse::AllowSession),
        ],
    )
    .await;
    ceiling_turn(&harness, capped(ChannelContext::private("app"))).await;

    let cards = approver.cards.lock().unwrap().clone();
    assert_eq!(cards.len(), 2, "a session answer is no lasting grant");
    for card in &cards {
        assert!(
            card.risk_level >= crate::types::RiskLevel::High,
            "no Medium-ceilinged device can answer it: {card:?}"
        );
        assert!(
            matches!(card.kind, crate::types::ApprovalKind::CommandOnce),
            "only Allow once and Deny: {card:?}"
        );
    }
    let results = tool_results_of(&harness).await;
    assert_eq!(
        results.iter().filter(|r| r.contains("note posted")).count(),
        2,
        "{results:?}"
    );
}

/// The owner's Deny on the capped card refuses the call like any decline.
#[tokio::test]
async fn a_declined_capped_card_runs_nothing() {
    let (harness, approver) = ceiling_harness(
        vec![post_note(), MockProvider::text_response("Not posted.")],
        Vec::new(),
        vec![Some(crate::types::ApprovalResponse::Deny)],
    )
    .await;
    ceiling_turn(&harness, capped(ChannelContext::private("app"))).await;
    assert_eq!(approver.card_count(), 1);
    let results = tool_results_of(&harness).await;
    assert!(
        !results.iter().any(|r| r.contains("note posted")),
        "{results:?}"
    );
}

/// A steer from a capped device into a run the owner's app started caps
/// that run from then on: the call after it waits for the owner.
#[tokio::test]
async fn a_capped_steer_caps_a_running_uncapped_run() {
    for (steer_ceiling, expected_cards) in [(false, 0), (true, 1)] {
        let channel_ctx = ChannelContext::private("app");
        let steer_ctx = if steer_ceiling {
            capped(channel_ctx.clone())
        } else {
            channel_ctx.clone()
        };
        let registry = Arc::new(crate::tasks::TaskRegistry::new(10));
        let queue: Arc<dyn crate::traits::Tool> = Arc::new(QueueOnCall {
            registry: registry.clone(),
            session: CEILING_SESSION.to_string(),
            message: crate::tasks::QueuedMessage::new(
                "Also post the note for Alice Rivera.",
                UserRole::Owner,
                steer_ctx,
                Vec::new(),
            ),
        });
        let (harness, approver) = ceiling_harness(
            vec![
                MockProvider::tool_call_response("slow_step", "{}"),
                post_note(),
                MockProvider::text_response("Posted it."),
            ],
            vec![queue],
            vec![Some(crate::types::ApprovalResponse::AllowOnce)],
        )
        .await;
        harness.agent.set_task_registry(registry.clone());
        ceiling_turn(&harness, channel_ctx).await;
        // The steer was taken into the run, not left for a turn of its own.
        assert_eq!(registry.queue_len(CEILING_SESSION).await, 0);
        assert_eq!(
            approver.card_count(),
            expected_cards,
            "steer capped: {steer_ceiling}"
        );
        let results = tool_results_of(&harness).await;
        assert!(
            results.iter().any(|r| r.contains("note posted")),
            "{results:?}"
        );
    }
}

/// The tool results the model saw last, in order.
async fn tool_results_of(harness: &crate::testing::TestHarness) -> Vec<String> {
    let calls = harness.provider.call_log.lock().await;
    let last = calls.last().expect("the model was called");
    last.messages
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| message["content"].as_str().unwrap_or_default().to_string())
        .collect()
}
