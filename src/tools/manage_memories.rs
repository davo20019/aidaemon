//! `manage_memories` lives in `aidaemon_runtime::tools::manage_memories`; the tests here run it against the daemon's assembled state store.

pub use aidaemon_runtime::tools::manage_memories::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::testing::CallWithFacts;
    use crate::traits::ToolCalls as _;
    use crate::traits::ToolOutcomeStatus;
    use crate::traits::ToolSemantics;
    use crate::traits::store_prelude::*;
    use crate::types::ChannelVisibility;
    use crate::types::FactPrivacy;
    use serde_json::Value;
    use serde_json::json;
    use std::sync::Arc;

    /// Memory-store actions declare no effects; privacy and goal actions
    /// stay undeclared.
    #[tokio::test]
    async fn memory_store_actions_declare_no_effects() {
        let tool = test_tool(setup_state().await);
        let effects = |args: Value| tool.call_effects(&args.to_string());
        for action in ["forget", "repair", "list", "search"] {
            assert!(effects(json!({"action": action})).complete, "{action}");
        }
        for action in ["set_privacy", "create_personal_goal"] {
            assert!(!effects(json!({"action": action})).complete, "{action}");
        }
        // A schedule action that moved runs nothing: it only names its new
        // home.
        assert!(effects(json!({"action": "cancel_scheduled"})).complete);
    }

    /// Search action appends the relational neighborhood when results contain a
    /// relationship-typed key. No LLM/MockProvider needed — entities are derived
    /// purely from matched result values.
    ///
    /// Seed: mother_name=Carol, father_name=Frank Mendez.
    /// Query "carol's mother" with limit=1 → only mother_name in initial_ids.
    /// Expansion derives "Carol" from mother_name value, assembles neighborhood,
    /// and surfaces father_name under "Related context".
    #[tokio::test]
    async fn test_search_relational_expansion() {
        let state = setup_state().await;

        state
            .upsert_fact(
                "user",
                "mother_name",
                "Carol",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "user",
                "father_name",
                "Frank Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // No MockProvider needed — expansion is driven purely by result entities.
        let tool = test_tool(state.clone());

        // limit=1 keeps only the top-ranked initial hit (mother_name) so
        // initial_ids does NOT contain father_name. The neighborhood expansion
        // is then the only path that can surface "Frank".
        let out = tool
            .call(&json!({"action": "search", "query": "carol's mother", "limit": 1}).to_string())
            .await
            .unwrap();

        // Guard: "Frank" must NOT appear in the primary (pre-neighborhood) results.
        let split = out.find("Related context").unwrap_or(out.len());
        assert!(
            !out[..split].contains("Frank"),
            "Frank must NOT be in primary results for the expansion-isolation test to be meaningful; output:\n{out}"
        );

        assert!(
            out.contains("Frank"),
            "neighborhood expansion should surface father_name fact; got:\n{out}"
        );
        assert!(
            out.contains("Related context"),
            "output should have a neighborhood header; got:\n{out}"
        );
    }

    /// Regression: keyword query (no possessive/recall phrasing) still triggers
    /// neighborhood expansion via the result-signal path.
    ///
    /// Seed: mother_name=Carol, father_name=Frank.
    /// Query "carol" (bare keyword) matches mother_name. Expansion fires
    /// because mother_name is a relationship key in the matched results.
    /// No MockProvider needed — classifier is no longer invoked.
    #[tokio::test]
    async fn test_search_keyword_query_triggers_neighborhood_via_result_signal() {
        let state = setup_state().await;

        state
            .upsert_fact(
                "user",
                "mother_name",
                "Carol",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "user",
                "father_name",
                "Frank",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // No MockProvider — result-signal is sufficient.
        let tool = test_tool(state.clone());

        // limit=1 so only the top-ranked initial hit is in the base results;
        // neighborhood expansion is the only path that can surface "Frank".
        let out = tool
            .call(&json!({"action": "search", "query": "carol", "limit": 1}).to_string())
            .await
            .unwrap();

        // Guard: Frank must NOT be in the primary block.
        let split = out.find("Related context").unwrap_or(out.len());
        assert!(
            !out[..split].contains("Frank"),
            "Frank must NOT be in primary results; output:\n{out}"
        );

        assert!(
            out.contains("Frank"),
            "keyword-query neighborhood expansion should surface father_name; got:\n{out}"
        );
        assert!(
            out.contains("Related context"),
            "output should have a neighborhood header; got:\n{out}"
        );
    }

    /// Expansion works with multi-word values (the common production case where
    /// mother_name="Carol Mendez" and father="Frank Mendez").
    ///
    /// Previously, the classifier was needed because a bare keyword query yielded
    /// no entities from the LLM. Now the value of the matched relationship-key
    /// fact IS the candidate entity — no classifier call at all.
    #[tokio::test]
    async fn test_search_neighborhood_from_result_entities_when_classifier_empty() {
        let state = setup_state().await;

        state
            .upsert_fact(
                "user",
                "mother_name",
                "Carol Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "user",
                "father",
                "Frank Mendez",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        // No MockProvider — the result-derived entity path is the only path.
        let tool = test_tool(state.clone());

        // limit=1 ensures only the top-ranked initial hit (mother_name) is in the
        // primary block; neighborhood expansion is the ONLY path that can surface Frank.
        let out = tool
            .call(&json!({"action": "search", "query": "carol", "limit": 1}).to_string())
            .await
            .unwrap();

        // Guard: Frank must NOT appear before the "Related context" separator.
        let split = out.find("Related context").unwrap_or(out.len());
        assert!(
            !out[..split].contains("Frank"),
            "Frank must NOT be in primary results — expansion-isolation; output:\n{out}"
        );

        assert!(
            out.contains("Frank"),
            "result-derived entity path must surface father=Frank even without a classifier; got:\n{out}"
        );
        assert!(
            out.contains("Related context"),
            "output should have a neighborhood header; got:\n{out}"
        );
    }

    #[tokio::test]
    async fn search_header_reports_honest_counts() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        let topics = [
            "gardening",
            "bicycle",
            "jazz",
            "sourdough",
            "chess",
            "hiking",
            "kettle",
            "piano",
            "camera",
            "pottery",
        ];
        for i in 0..27 {
            state
                .upsert_fact(
                    "general",
                    &format!("hobby_{i}"),
                    &format!("{} session {i}", topics[i % topics.len()]),
                    "test",
                    None,
                    FactPrivacy::Global,
                )
                .await
                .unwrap();
        }
        for i in 0..3 {
            state
                .upsert_fact(
                    "project",
                    &format!("zephyr_item_{i}"),
                    &format!("zephyr build {i}"),
                    "test",
                    None,
                    FactPrivacy::Global,
                )
                .await
                .unwrap();
        }
        let out = tool
            .call(&json!({"action": "search", "query": "zephyr"}).to_string())
            .await
            .unwrap();
        assert!(out.contains("(3 shown of"), "{out}");
        assert!(
            !out.contains("relevant of"),
            "no floor, no relevance claim: {out}"
        );
        assert!(!out.contains("of 30 matches"), "{out}");
        assert!(!out.contains("more relevant"), "{out}");
    }

    #[tokio::test]
    async fn test_search_tool_finds_port_among_substring_noise() {
        // End-to-end regression for the live recall bug: "database port" used to
        // bury `local_dev_db_port` under substring noise (report/portfolio/…) and
        // truncate it away. The hybrid search must surface it.
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        state
            .upsert_fact_with_provenance(
                "preference",
                "local_dev_db_port",
                "54329",
                "progressive",
                None,
                FactPrivacy::Global,
                Some(chrono::Utc::now()),
                Some("You told me:\nThe local database port is 54329."),
            )
            .await
            .unwrap();
        // Substring-noise facts that previously outranked/truncated the real one.
        for (i, (k, v)) in [
            ("report_output_location", "~/reports"),
            ("portfolio_site", "example.com"),
            ("important_note", "ship it"),
            ("support_email", "help@example.com"),
        ]
        .iter()
        .enumerate()
        {
            state
                .upsert_fact(
                    "project",
                    k,
                    v,
                    &format!("test{i}"),
                    None,
                    FactPrivacy::Global,
                )
                .await
                .unwrap();
        }

        let result = tool
            .call(&json!({"action": "search", "query": "database port"}).to_string())
            .await
            .unwrap();

        assert!(
            result.contains("local_dev_db_port") && result.contains("54329"),
            "search for 'database port' must surface the port fact; got:\n{result}"
        );
        assert!(
            !result.contains("report_output_location"),
            "substring noise must not appear; got:\n{result}"
        );
        // Temporal affordance: each result reports when the fact was learned so
        // the model can answer "when did I tell you X?" without thrashing.
        assert!(
            result.contains("learned:"),
            "search results should include a 'learned:' recency; got:\n{result}"
        );
        assert!(result.contains("fact_id:"), "missing fact id:\n{result}");
        assert!(
            result.contains("source: progressive"),
            "missing extraction source:\n{result}"
        );
        assert!(
            result.contains("evidence: \"You told me: The local database port is 54329.\""),
            "missing normalized source evidence:\n{result}"
        );
    }

    #[tokio::test]
    async fn test_search_episodes_recalls_past_conversation() {
        // Concrete store so we can use the inherent insert_episode helper.
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let concrete = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        let tool = test_tool(concrete.clone());
        let now = chrono::Utc::now();
        let ep = crate::traits::Episode {
            id: 0,
            session_id: "test".to_string(),
            summary: "Wrote and ran a Python primes script and a stats module with pytest"
                .to_string(),
            topics: Some(vec!["python".to_string(), "testing".to_string()]),
            emotional_tone: None,
            outcome: Some("completed".to_string()),
            importance: 0.7,
            recall_count: 0,
            last_recalled_at: None,
            message_count: 10,
            start_time: now,
            end_time: now,
            created_at: now,
            channel_id: None,
        };
        concrete.insert_episode(&ep).await.unwrap();

        // Empty query → recency browse (no embedding dependency).
        let result = tool
            .call_with(
                &json!({
                    "action": "search_episodes"
                })
                .to_string(),
                crate::traits::ToolCallContext {
                    session_id: "test".into(),
                    channel_visibility: Some(crate::types::ChannelVisibility::Private),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            result.contains("primes script") || result.contains("stats module"),
            "should recall the episode summary; got:\n{result}"
        );
        assert!(
            result.contains("topics:"),
            "should surface topics; got:\n{result}"
        );
        assert!(
            result.contains("just now") || result.contains("ago"),
            "should include a recency label; got:\n{result}"
        );
    }

    /// Trust comes only from the dispatcher's typed context: a
    /// `_trusted_session` argument does not widen internal recall to every
    /// session's episodes.
    #[tokio::test]
    async fn trusted_session_argument_bypasses_nothing() {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let concrete = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        let tool = test_tool(concrete.clone());
        let now = chrono::Utc::now();
        for (session, summary) in [
            (
                "session-alice",
                "Planned the Acme Corp offsite with Alice Rivera",
            ),
            (
                "session-jordan",
                "Reviewed the synthetic budget with Jordan Lee",
            ),
        ] {
            let ep = crate::traits::Episode {
                id: 0,
                session_id: session.to_string(),
                summary: summary.to_string(),
                topics: None,
                emotional_tone: None,
                outcome: None,
                importance: 0.7,
                recall_count: 0,
                last_recalled_at: None,
                message_count: 4,
                start_time: now,
                end_time: now,
                created_at: now,
                channel_id: None,
            };
            concrete.insert_episode(&ep).await.unwrap();
        }
        let args = json!({
            "action": "search_episodes",
            "_trusted_session": true
        })
        .to_string();
        let facts = crate::traits::ToolCallContext {
            session_id: "session-alice".into(),
            channel_visibility: Some(ChannelVisibility::Internal),
            ..Default::default()
        };
        let result = tool
            .call_with_execution_context(&args, None, facts.clone())
            .await
            .unwrap()
            .output;
        assert!(result.contains("Acme Corp offsite"), "{result}");
        assert!(
            !result.contains("Jordan Lee"),
            "an argument does not widen recall to other sessions: {result}"
        );
        // The dispatcher's typed trust does.
        let ctx = crate::traits::ToolCallContext {
            trusted_session: true,
            ..facts
        };
        let trusted = tool
            .call_with_execution_context(&args, None, ctx)
            .await
            .unwrap()
            .output;
        assert!(trusted.contains("Jordan Lee"), "{trusted}");
    }

    fn test_tool(state: Arc<SqliteStateStore>) -> ManageMemoriesTool {
        let store = state.memory_stores();
        ManageMemoriesTool::new(state, store)
    }

    async fn setup_state() -> Arc<SqliteStateStore> {
        let db_file = tempfile::NamedTempFile::new().unwrap();
        let db_path = db_file.path().to_str().unwrap().to_string();
        let embedding_service = Arc::new(EmbeddingService::deterministic());
        let state = Arc::new(
            SqliteStateStore::for_tests(&db_path, 100, None, embedding_service)
                .await
                .unwrap(),
        );
        std::mem::forget(db_file);
        state
    }

    // ==================== create_personal_goal tests ====================

    #[tokio::test]
    async fn create_personal_goal_creates_goal_in_registry() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Learn conversational Spanish"
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Created personal goal"));
        assert!(result.contains("Learn conversational Spanish"));

        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].description, "Learn conversational Spanish");
        assert_eq!(goals[0].domain, "personal");
        assert_eq!(goals[0].priority, "medium");
        assert_eq!(goals[0].status, "active");
        assert_eq!(goals[0].session_id, "owner-session");
    }

    #[tokio::test]
    async fn created_goal_with_rate_limit_language_reports_typed_success() {
        let state = setup_state().await;
        let tool = test_tool(state);

        let outcome = tool
            .call_with_execution_context(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Steward a synthetic account under owner-confirmed rate limits"
                })
                .to_string(),
                None,
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert_eq!(
            outcome.metadata.outcome_status,
            Some(ToolOutcomeStatus::Succeeded)
        );
        assert!(outcome.output.starts_with("Created personal goal "));
    }

    #[tokio::test]
    async fn list_goals_marks_only_same_session_sources_as_mandate_compatible() {
        let state = setup_state().await;
        let compatible = crate::traits::Goal::new_personal("Compatible goal", "owner-session");
        let foreign = crate::traits::Goal::new_personal("Foreign goal", "other-session");
        state.create_goal(&foreign).await.unwrap();
        state.create_goal(&compatible).await.unwrap();
        let tool = test_tool(state);

        let output = tool
            .call_with(
                &json!({
                    "action": "list_goals",
                    "limit": 10
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        let compatible_line = output
            .lines()
            .find(|line| line.contains("Compatible goal"))
            .unwrap();
        let foreign_line = output
            .lines()
            .find(|line| line.contains("Foreign goal"))
            .unwrap();
        assert!(compatible_line.contains("mandate-source compatible"));
        assert!(foreign_line.contains("not compatible"));
        assert!(output.find("Compatible goal") < output.find("Foreign goal"));
    }

    #[tokio::test]
    async fn create_personal_goal_with_priority() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Exercise daily",
                    "priority": "high"
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Created personal goal"));
        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert_eq!(goals[0].priority, "high");
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_invalid_priority() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Some goal",
                    "priority": "ultra"
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await
            .unwrap();

        assert!(result.contains("Invalid priority"));
        let goals = state.get_active_personal_goals(50).await.unwrap();
        assert!(goals.is_empty());
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_empty_description() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call_with(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "   "
                })
                .to_string(),
                crate::testing::call_facts("owner-session"),
            )
            .await;

        assert!(result.is_err() || result.unwrap().contains("required"));
    }

    #[tokio::test]
    async fn create_personal_goal_rejects_missing_internal_session() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        let result = tool
            .call(
                &json!({
                    "action": "create_personal_goal",
                    "goal": "Cannot become unowned"
                })
                .to_string(),
            )
            .await;

        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains(
                "create_personal_goal: the session fact is missing from the call context"
            )
        );
        assert!(
            state
                .get_active_personal_goals(50)
                .await
                .unwrap()
                .is_empty()
        );
    }

    // ==================== create_scheduled_goal tests ====================

    /// Regression: an empty / whitespace `key` to `forget` previously matched
    /// every fact via `fk.contains("")` and wiped all active facts across all
    /// categories. The action must refuse to run instead.
    #[tokio::test]
    async fn forget_rejects_empty_or_whitespace_key() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        for (cat, key, val) in [
            ("preferences", "favorite_color", "blue"),
            ("preferences", "favorite_food", "tacos"),
            ("personal", "birthday", "1990-01-01"),
        ] {
            state
                .upsert_fact(cat, key, val, "test", None, FactPrivacy::Global)
                .await
                .unwrap();
        }

        for blank in ["", "   ", "\t", "\n"] {
            let err = tool
                .call(
                    &json!({
                        "action": "forget",
                        "category": "preferences",
                        "key": blank,
                    })
                    .to_string(),
                )
                .await
                .expect_err("blank key must be rejected");
            let msg = err.to_string();
            assert!(
                msg.contains("empty") || msg.contains("whitespace"),
                "expected refusal message, got {msg:?}"
            );
        }

        // All facts should still be present.
        let remaining = state.get_facts(None).await.unwrap();
        let active = remaining
            .iter()
            .filter(|f| f.superseded_at.is_none())
            .count();
        assert_eq!(active, 3, "no facts should have been deleted");
    }

    /// Regression: single-character keys are also too broad to use for substring
    /// matching across the fact store.
    #[tokio::test]
    async fn forget_rejects_single_character_key() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        state
            .upsert_fact(
                "preferences",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let err = tool
            .call(
                &json!({
                    "action": "forget",
                    "category": "preferences",
                    "key": "a",
                })
                .to_string(),
            )
            .await
            .expect_err("single-character key must be rejected");
        assert!(err.to_string().contains("at least 2 characters"));

        let remaining = state.get_facts(None).await.unwrap();
        assert_eq!(
            remaining
                .iter()
                .filter(|f| f.superseded_at.is_none())
                .count(),
            1
        );
    }

    /// Substring matching is still allowed within the explicitly-requested
    /// category — the caller knows what they are targeting there.
    #[tokio::test]
    async fn forget_substring_matches_within_requested_category() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        state
            .upsert_fact(
                "preferences",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "personal",
                "color_of_car",
                "red",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "forget",
                    "category": "preferences",
                    "key": "color",
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten"), "got: {result}");

        // Only the `preferences` fact (substring match) should be gone.
        // The `personal.color_of_car` substring match must NOT be deleted
        // because cross-category sweep only allows exact/canonical matches.
        let remaining = state.get_facts(None).await.unwrap();
        let active: Vec<_> = remaining
            .iter()
            .filter(|f| f.superseded_at.is_none())
            .collect();
        assert_eq!(active.len(), 1, "got: {active:?}");
        assert_eq!(active[0].category, "personal");
        assert_eq!(active[0].key, "color_of_car");
    }

    /// Regression: forgetting `qa_pet_name` also deleted the owner's `name`
    /// fact because a fact key contained in the requested key matched, and
    /// partial matching ran even when an exact key existed.
    #[tokio::test]
    async fn forget_never_deletes_keys_contained_in_the_requested_key() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        for (key, value) in [
            ("qa_pet_name", "SYNTH-FINCH"),
            ("name", "Alice Rivera"),
            ("pet_name_history", "Rex"),
        ] {
            state
                .upsert_fact("user", key, value, "test", None, FactPrivacy::Global)
                .await
                .unwrap();
        }

        let result = tool
            .call(
                &json!({"action": "forget", "category": "user", "key": "qa_pet_name"}).to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten 1 fact"), "got: {result}");

        let active: Vec<_> = state
            .get_facts(None)
            .await
            .unwrap()
            .into_iter()
            .filter(|f| f.superseded_at.is_none())
            .map(|f| f.key)
            .collect();
        assert!(active.contains(&"name".to_string()), "got: {active:?}");
        assert!(active.contains(&"pet_name_history".to_string()));
        assert!(!active.contains(&"qa_pet_name".to_string()));
    }

    #[tokio::test]
    async fn forget_partial_key_matches_whole_segments_only() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());
        for key in ["favorite_color", "colorado_trip"] {
            state
                .upsert_fact("preferences", key, "x", "test", None, FactPrivacy::Global)
                .await
                .unwrap();
        }
        let result = tool
            .call(
                &json!({"action": "forget", "category": "preferences", "key": "color"}).to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten 1 fact"), "got: {result}");
        assert!(result.contains("favorite_color"), "got: {result}");
    }

    /// Cross-category sweep should still catch exact-key duplicates (the
    /// original "in case duplicates exist elsewhere" intent), but only via
    /// exact or canonical match — never substring.
    #[tokio::test]
    async fn forget_cross_category_only_exact_or_canonical() {
        let state = setup_state().await;
        let tool = test_tool(state.clone());

        // Exact duplicate across categories should be cleaned up.
        state
            .upsert_fact(
                "preferences",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        state
            .upsert_fact(
                "personal",
                "favorite_color",
                "blue",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();
        // An unrelated fact whose key contains "color" — must NOT be deleted.
        state
            .upsert_fact(
                "personal",
                "color_of_car",
                "red",
                "test",
                None,
                FactPrivacy::Global,
            )
            .await
            .unwrap();

        let result = tool
            .call(
                &json!({
                    "action": "forget",
                    "category": "preferences",
                    "key": "favorite_color",
                })
                .to_string(),
            )
            .await
            .unwrap();
        assert!(result.contains("Forgotten 2 fact"), "got: {result}");

        let remaining = state.get_facts(None).await.unwrap();
        let active: Vec<_> = remaining
            .iter()
            .filter(|f| f.superseded_at.is_none())
            .collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].key, "color_of_car");
    }
}
