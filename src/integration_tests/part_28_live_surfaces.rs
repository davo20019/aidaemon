// ==========================================================================
// Live surfaces: the agent hands a browser tab to the owner
//
// `hand_over` is an ordinary owner card through the kernel: the model's
// call interrupts the run, the owner gets a take-over card, Done runs the
// call (which reports where the tab is now) and Not now binds the run so
// the agent does not ask again. What the owner types during a takeover
// never reaches the model, the history or the event log.
// ==========================================================================

/// A surface registry with one scripted browser tab in the approvals chat.
fn handoff_surfaces() -> (
    aidaemon_surfaces::SurfaceRegistry,
    aidaemon_surfaces::testing::MockSurface,
    aidaemon_surfaces::SurfaceId,
    Arc<dyn crate::traits::Tool>,
) {
    let registry = aidaemon_surfaces::SurfaceRegistry::new(Default::default());
    let mock = aidaemon_surfaces::testing::MockSurface::new();
    mock.set_notifier(registry.notifier());
    registry.add_backend(Arc::new(mock.clone()));
    let id = mock.add(APPROVALS_SESSION, "tab-synthetic-1");
    let tool: Arc<dyn crate::traits::Tool> = Arc::new(
        aidaemon_surfaces::tool::HandOverTool::new(registry.clone(), Arc::new(|_| None)),
    );
    (registry, mock, id, tool)
}

fn hand_over_call() -> crate::traits::ProviderResponse {
    MockProvider::tool_call_response(
        "hand_over",
        &json!({"what": "Sign in to your Acme account"}).to_string(),
    )
}

#[tokio::test]
async fn a_handoff_is_a_take_over_card_and_done_returns_where_the_tab_is() {
    let (registry, _mock, id, tool) = handoff_surfaces();
    let (harness, approver) = approvals_harness_with(
        vec![
            hand_over_call(),
            MockProvider::text_response("Signed in; carrying on."),
        ],
        vec![Some(crate::types::ApprovalResponse::AllowOnce)],
        None,
        vec![tool],
    )
    .await;
    run_owner_turn(&harness, ChannelContext::private("telegram")).await;

    assert_eq!(approver.card_count(), 1);
    let card = approver.cards.lock().unwrap()[0].card.clone().unwrap();
    assert!(
        matches!(&card.action, aidaemon_core::CardAction::TakeOver { surface_id, what, .. }
            if *surface_id == id.to_string() && what == "Sign in to your Acme account"),
        "{card:?}"
    );
    let results = tool_results(&harness).await;
    assert!(
        results.iter().any(|r| r.contains("\"owner_answer\": \"done\"")
            && r.contains("tab-synthetic-1")),
        "{results:?}"
    );
    let info = registry.get(&id).unwrap();
    assert!(!info.handoff_pending);
    assert_eq!(info.controller, aidaemon_surfaces::Controller::Agent);
}

#[tokio::test]
async fn not_now_binds_the_run_so_the_agent_does_not_ask_again() {
    let (registry, _mock, id, tool) = handoff_surfaces();
    let (harness, approver) = approvals_harness_with(
        vec![
            hand_over_call(),
            hand_over_call(),
            MockProvider::text_response("I'll stop here."),
        ],
        vec![Some(crate::types::ApprovalResponse::Deny)],
        None,
        vec![tool],
    )
    .await;
    run_owner_turn(&harness, ChannelContext::private("telegram")).await;

    assert_eq!(approver.card_count(), 1, "a declined handoff is not asked again");
    let results = tool_results(&harness).await;
    assert_eq!(results.len(), 2, "{results:?}");
    assert!(
        results.iter().all(|r| !r.contains("owner_answer")),
        "{results:?}"
    );
    assert!(!registry.get(&id).unwrap().handoff_pending);
}

#[tokio::test]
async fn nothing_typed_during_a_takeover_reaches_the_model_or_the_records() {
    const SENTINEL: &str = "synthetic-secret-7f3e";
    let (registry, mock, id, tool) = handoff_surfaces();
    let (harness, approver) = approvals_harness_with(
        vec![hand_over_call(), MockProvider::text_response("Done.")],
        // Held: the test answers once it has typed.
        vec![None],
        None,
        vec![tool],
    )
    .await;
    let harness = Arc::new(harness);
    let turn = {
        let harness = harness.clone();
        tokio::spawn(async move {
            run_owner_turn(&harness, ChannelContext::private("telegram")).await;
        })
    };
    // The card reached the owner.
    for _ in 0..500 {
        if !approver.held.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(registry.get(&id).unwrap().handoff_pending);

    // The owner takes over from a phone and types.
    let live = registry
        .open_live(
            &id,
            "synthetic-phone",
            aidaemon_surfaces::LiveMode::Control,
            Default::default(),
        )
        .await
        .unwrap();
    registry
        .input(
            &live.live_id,
            &aidaemon_surfaces::InputEvent::Text {
                text: SENTINEL.into(),
            },
        )
        .await
        .unwrap();
    assert_eq!(mock.inputs().len(), 1);
    registry.close_live(&live.live_id);
    let answer = approver.held.lock().unwrap().pop().expect("the card waits");
    let _ = answer.send(crate::types::ApprovalResponse::AllowOnce);
    turn.await.unwrap();

    // Not in what the model was sent.
    let calls = harness.provider.call_log.lock().await;
    assert!(!calls.is_empty());
    for call in calls.iter() {
        assert!(
            !serde_json::to_string(&call.messages).unwrap().contains(SENTINEL),
            "the model saw what the owner typed"
        );
    }
    drop(calls);
    // Not anywhere in the database: every text cell of every table.
    let tables: Vec<String> =
        sqlx::query_scalar("SELECT name FROM sqlite_master WHERE type = 'table'")
            .fetch_all(&harness.state.pool())
            .await
            .unwrap();
    assert!(!tables.is_empty());
    for table in tables {
        let columns: Vec<String> = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
            "SELECT name FROM pragma_table_info('{table}')"
        )))
        .fetch_all(&harness.state.pool())
        .await
        .unwrap();
        for column in columns {
            let hits: i64 = sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
                "SELECT COUNT(*) FROM \"{table}\" WHERE CAST(\"{column}\" AS TEXT) LIKE ?"
            )))
            .bind(format!("%{SENTINEL}%"))
            .fetch_one(&harness.state.pool())
            .await
            .unwrap_or(0);
            assert_eq!(hits, 0, "{table}.{column} holds what the owner typed");
        }
    }
}

#[tokio::test]
async fn a_handoff_with_no_tab_asks_nothing_and_says_why() {
    let registry = aidaemon_surfaces::SurfaceRegistry::new(Default::default());
    let tool: Arc<dyn crate::traits::Tool> = Arc::new(
        aidaemon_surfaces::tool::HandOverTool::new(registry, Arc::new(|_| None)),
    );
    let (harness, approver) = approvals_harness_with(
        vec![hand_over_call(), MockProvider::text_response("No tab.")],
        vec![],
        None,
        vec![tool],
    )
    .await;
    run_owner_turn(&harness, ChannelContext::private("telegram")).await;
    assert_eq!(approver.card_count(), 0);
    let results = tool_results(&harness).await;
    assert!(
        results.iter().any(|r| r.contains("No browser tab")),
        "{results:?}"
    );
}

/// Live 2026-11-17: "open the sign-in page and hand it over so I can sign
/// in". The run opened the page and handed it over, the owner signed in,
/// and the reply still said "I didn't start these parts": a handoff was
/// typed as a look, so a request for a step counted no step done. The
/// owner's step through a handoff is work done.
#[tokio::test]
async fn a_finished_handoff_is_the_step_it_handed_over() {
    let (_registry, _mock, _id, tool) = handoff_surfaces();
    let items = |status: &str| {
        json!({"items": [
            {"text": "Open the synthetic sign-in page and hand it over for the owner's sign-in", "status": status, "mutation_effects": ["process_state"], "owner_units": [0]}
        ]})
    };
    let provider = MockProvider::with_responses(vec![
        MockProvider::tool_call_response("track_requirements", &items("in_progress").to_string()),
        hand_over_call(),
        MockProvider::tool_call_response(
            "track_requirements",
            &{
                let mut done = items("completed");
                done["outcome"] = json!("done");
                done.to_string()
            },
        ),
        MockProvider::text_response("You're signed in; the secure page is open."),
        MockProvider::text_response("UNREACHED"),
    ])
    .with_owner_units(vec![MockProvider::owner_units(&[(
        "Open the synthetic sign-in page and hand it over so I can sign in",
        true,
    )])]);
    let harness = setup_test_agent_with_extra_tools_and_llm_timeout(provider, vec![tool], None)
        .await
        .unwrap();
    let (broker, approver) =
        crate::testing::ScriptedApprover::start(vec![Some(crate::types::ApprovalResponse::AllowOnce)]);
    harness.agent.owner_approvals().set_broker(broker);
    let reply = ask_owner_turn(
        &harness,
        APPROVALS_SESSION,
        "Open the synthetic sign-in page and hand it over so I can sign in.",
    )
    .await;
    assert_eq!(approver.card_count(), 1);
    assert!(!reply.contains("didn't start"), "{reply}");
    assert_eq!(reply, "You're signed in; the secure page is open.");
}
