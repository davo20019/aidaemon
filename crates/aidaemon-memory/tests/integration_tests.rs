//! Model-driven writers, the hook, the tools and injection, through real
//! `Agent` runs with scripted providers.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_core::testing::{call, text, tool_calls, ScriptedProvider, SCRIPTED_USAGE};
use aidaemon_core::{
    Agent, Message, ModelRequest, ModelResponse, Provider, ProviderError, Role, ToolCall,
};
use aidaemon_memory::{
    EpisodeSummarizer, FactExtractor, HashEmbedder, InMemoryMemoryStore, Memory, MemoryError,
    MemoryRecall, MemoryWriter, NewFact, RecallTool, RememberTool,
};
use aidaemon_policies::MemoryInjection;
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Notify;

const SESSION: &str = "telegram:synthetic-user-1";

fn facts_reply(facts: serde_json::Value) -> ModelResponse {
    tool_calls([ToolCall::new(
        "x1",
        "record_facts",
        json!({ "facts": facts }),
    )])
}

fn memory_with(extractor: Option<FactExtractor>, summarizer: Option<EpisodeSummarizer>) -> Memory {
    let mut builder = Memory::builder(
        Arc::new(InMemoryMemoryStore::new()),
        Arc::new(HashEmbedder::default()),
    );
    if let Some(e) = extractor {
        builder = builder.extractor(e);
    }
    if let Some(s) = summarizer {
        builder = builder.summarizer(s);
    }
    builder.build()
}

fn turn(user: &str, assistant: &str) -> Vec<Message> {
    vec![Message::user(user), Message::assistant(assistant, vec![])]
}

#[tokio::test]
async fn extractor_parses_good_json_and_stores_with_provenance() {
    let provider = Arc::new(ScriptedProvider::new([facts_reply(json!([
        {"category": "person", "key": "daughters", "value": "Mia and Zoe",
         "confidence": 0.95, "evidence": "my daughters Mia and Zoe"},
        {"category": "user", "key": "employer", "value": "Acme Corp", "confidence": 0.8}
    ]))]));
    let memory = memory_with(
        Some(FactExtractor::from_arc(provider.clone(), "small")),
        None,
    );

    let report = memory
        .ingest_turn(
            SESSION,
            Some("run-7"),
            &turn(
                "I work at Acme Corp and my daughters Mia and Zoe start school soon",
                "Congratulations!",
            ),
        )
        .await
        .unwrap();
    assert_eq!(report.facts.len(), 2);
    assert_eq!(report.rejected, 0);
    assert_eq!(report.usage, SCRIPTED_USAGE);

    let facts = memory.facts(SESSION).await.unwrap();
    let daughters = facts.iter().find(|f| f.key == "daughters").unwrap();
    assert_eq!(daughters.source.run_id.as_deref(), Some("run-7"));
    assert_eq!(daughters.source.session_id.as_deref(), Some(SESSION));
    assert_eq!(
        daughters.source.excerpt.as_deref(),
        Some("my daughters Mia and Zoe")
    );
    let employer = facts.iter().find(|f| f.key == "employer").unwrap();
    assert!(employer
        .source
        .excerpt
        .as_deref()
        .unwrap()
        .contains("Acme Corp"));

    // The request forced the typed tool and carried the transcript.
    let request = &provider.requests()[0];
    assert_eq!(
        request.tool_choice,
        aidaemon_core::ToolChoice::Specific("record_facts".into())
    );
    assert!(request.messages[1].content.contains("Mia and Zoe"));
}

#[tokio::test]
async fn extractor_skips_malformed_entries() {
    let provider = ScriptedProvider::new([facts_reply(json!([
        {"category": "user", "key": "home_city", "value": "Lisbon", "confidence": 0.9},
        {"category": "user", "key": "", "value": "empty key"},
        {"category": "user", "value": "no key"},
        {"category": "user", "key": "k", "value": {"nested": true}},
        {"category": "user", "key": "k2", "value": "v", "confidence": "high"},
        "not an object",
        42
    ]))]);
    let memory = memory_with(Some(FactExtractor::new(provider, "small")), None);
    let report = memory
        .ingest_turn(
            SESSION,
            None,
            &turn("I moved to Lisbon last spring for good", "Nice"),
        )
        .await
        .unwrap();
    assert_eq!(report.facts.len(), 1);
    assert_eq!(report.rejected, 6);
}

#[tokio::test]
async fn extractor_survives_garbage_and_provider_errors() {
    for garbage in [
        "",
        "I think you like blue?",
        "{\"facts\": 3}",
        "[[[",
        "null",
    ] {
        let memory = memory_with(
            Some(FactExtractor::new(
                ScriptedProvider::new([text(garbage)]),
                "m",
            )),
            None,
        );
        let report = memory
            .ingest_turn(
                SESSION,
                None,
                &turn("My favorite color is blue, remember it", "ok"),
            )
            .await
            .unwrap();
        assert!(report.facts.is_empty(), "{garbage:?}");
        assert_eq!(memory.stats().await.unwrap().active_facts, 0);
    }

    let failing = ScriptedProvider::with_results([Err(ProviderError::Transient("down".into()))]);
    let memory = memory_with(Some(FactExtractor::new(failing, "m")), None);
    let err = memory
        .ingest_turn(
            SESSION,
            None,
            &turn("My favorite color is blue, remember it", "ok"),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, MemoryError::Provider(_)));
}

#[tokio::test]
async fn short_turns_are_skipped_without_a_model_call() {
    let provider = Arc::new(ScriptedProvider::new([]));
    let memory = memory_with(Some(FactExtractor::from_arc(provider.clone(), "m")), None);
    let report = memory
        .ingest_turn(SESSION, None, &turn("thanks!", "You're welcome"))
        .await
        .unwrap();
    assert!(report.skipped.is_some());
    assert!(provider.requests().is_empty());
}

#[tokio::test]
async fn known_keys_let_corrections_supersede() {
    let provider = Arc::new(ScriptedProvider::new([
        facts_reply(
            json!([{"category": "pet", "key": "dog_name", "value": "Rex", "confidence": 0.9}]),
        ),
        facts_reply(
            json!([{"category": "pet", "key": "dog_name", "value": "Max", "confidence": 0.9}]),
        ),
    ]));
    let memory = memory_with(Some(FactExtractor::from_arc(provider.clone(), "m")), None);
    memory
        .ingest_turn(
            SESSION,
            None,
            &turn("My dog's name is Rex, he is three", "Cute"),
        )
        .await
        .unwrap();
    memory
        .ingest_turn(
            SESSION,
            None,
            &turn("Sorry, my dog is actually called Max", "Got it"),
        )
        .await
        .unwrap();
    assert!(provider.requests()[1].messages[1]
        .content
        .contains("pet/dog_name"));
    let facts = memory.facts(SESSION).await.unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].value, "Max");
    assert_eq!(
        memory
            .fact_history(SESSION, "pet", "dog_name")
            .await
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn summarizer_creates_a_scored_episode() {
    let provider = ScriptedProvider::new([tool_calls([ToolCall::new(
        "e1",
        "record_episode",
        json!({
            "summary": "Alice Rivera planned a birthday party for Mia and Zoe.",
            "topics": ["party", "family", ""],
            "decisions": ["Hold it on Saturday"],
            "goals": [],
            "outcome": "resolved",
            "emotional_intensity": 0.5
        }),
    )])]);
    let memory = memory_with(None, Some(EpisodeSummarizer::new(provider, "m")));
    let messages = turn("Help me plan a party for Mia and Zoe", "Saturday works");
    let (episode, usage) = memory
        .summarize_session(SESSION, &messages, &[])
        .await
        .unwrap();
    assert_eq!(usage, SCRIPTED_USAGE);
    assert_eq!(episode.topics, vec!["party", "family"]);
    // base 0.5 + decisions 0.1 + intensity 0.5 * 0.2
    assert!(
        (episode.importance - 0.7).abs() < 1e-5,
        "{}",
        episode.importance
    );
    let hits = memory
        .recall_episodes(SESSION, "birthday party", 3)
        .await
        .unwrap();
    assert_eq!(hits[0].episode.id, episode.id);

    let bad = memory_with(
        None,
        Some(EpisodeSummarizer::new(
            ScriptedProvider::new([text("no json")]),
            "m",
        )),
    );
    assert!(bad
        .summarize_session(SESSION, &messages, &[])
        .await
        .is_err());
}

/// A provider that waits for a signal before answering, to prove the hook
/// does not hold up the run.
struct GatedProvider {
    gate: Arc<Notify>,
    reply: ModelResponse,
}

#[async_trait]
impl Provider for GatedProvider {
    fn name(&self) -> &str {
        "gated"
    }
    async fn complete(&self, _: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.gate.notified().await;
        Ok(self.reply.clone())
    }
}

#[tokio::test]
async fn memory_writer_does_not_block_or_alter_the_run() {
    let gate = Arc::new(Notify::new());
    let extraction = GatedProvider {
        gate: gate.clone(),
        reply: facts_reply(json!([
            {"category": "preference", "key": "seat", "value": "window", "confidence": 0.9}
        ])),
    };
    let memory = memory_with(Some(FactExtractor::new(extraction, "small")), None);
    let writer = Arc::new(MemoryWriter::new(memory.clone()));
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([text("Booked a window seat.")]))
        .hook_arc(writer.clone())
        .build()
        .unwrap();

    let outcome = tokio::time::timeout(
        Duration::from_secs(5),
        agent.run(SESSION, "I always prefer a window seat when I fly"),
    )
    .await
    .expect("the run must not wait for extraction")
    .unwrap();
    assert!(outcome.is_completed());
    assert_eq!(outcome.final_text.as_deref(), Some("Booked a window seat."));
    assert_eq!(
        outcome.usage(),
        SCRIPTED_USAGE,
        "extraction usage is not billed to the run"
    );
    assert_eq!(outcome.state.messages.len(), 2);
    assert_eq!(memory.stats().await.unwrap().active_facts, 0);
    assert_eq!(writer.pending(), 1);

    gate.notify_one();
    let reports = writer.drain().await;
    assert_eq!(reports.len(), 1);
    let report = reports[0].result.as_ref().unwrap();
    assert_eq!(report.facts.len(), 1);
    assert_eq!(report.usage, SCRIPTED_USAGE);
    assert_eq!(reports[0].run_id, outcome.state.run_id.to_string());
    assert_eq!(memory.facts(SESSION).await.unwrap()[0].value, "window");
}

#[tokio::test]
async fn memory_writer_failures_never_reach_the_run() {
    let failing = ScriptedProvider::with_results([Err(ProviderError::Auth("bad key".into()))]);
    let memory = memory_with(Some(FactExtractor::new(failing, "m")), None);
    let writer = Arc::new(MemoryWriter::new(memory));
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([text("done")]))
        .hook_arc(writer.clone())
        .build()
        .unwrap();
    let outcome = agent
        .run(SESSION, "Please remember that my manager is Jordan Lee")
        .await
        .unwrap();
    assert!(outcome.is_completed());
    let reports = writer.drain().await;
    assert!(reports[0].result.as_ref().unwrap_err().contains("bad key"));
}

/// Rejects the first final answer of every run.
struct RejectOnce(std::sync::atomic::AtomicBool);
#[async_trait]
impl aidaemon_core::Hook for RejectOnce {
    async fn on_stop(
        &self,
        _ctx: &aidaemon_core::HookContext<'_>,
        _final_text: &str,
    ) -> aidaemon_core::StopDecision {
        if self.0.swap(true, std::sync::atomic::Ordering::SeqCst) {
            aidaemon_core::StopDecision::Accept
        } else {
            aidaemon_core::StopDecision::Continue {
                message: "check your work".into(),
            }
        }
    }
}

#[tokio::test]
async fn memory_writer_extracts_once_per_completed_run_only() {
    let extraction = Arc::new(ScriptedProvider::new([facts_reply(json!([
        {"category": "preference", "key": "seat", "value": "window", "confidence": 0.9}
    ]))]));
    let memory = memory_with(Some(FactExtractor::new(extraction.clone(), "small")), None);
    let writer = Arc::new(MemoryWriter::new(memory.clone()));
    // Registered BEFORE a hook that rejects the first answer: the writer must
    // still wait for the run to actually finish.
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([
            text("draft"),
            text("Booked a window seat."),
        ]))
        .hook_arc(writer.clone())
        .hook(RejectOnce(Default::default()))
        .build()
        .unwrap();
    agent
        .run(SESSION, "I always prefer a window seat when I fly")
        .await
        .unwrap();
    assert_eq!(writer.drain().await.len(), 1);
    let requests = extraction.requests();
    assert_eq!(requests.len(), 1);
    let sent: String = requests[0]
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    // Fired after the accepted answer, not at the rejected draft.
    assert!(sent.contains("Booked a window seat."), "{sent}");

    // A failed run writes nothing.
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::with_results([Err(ProviderError::Auth(
            "bad key".into(),
        ))]))
        .hook_arc(writer.clone())
        .build()
        .unwrap();
    agent
        .run(SESSION, "I always prefer a window seat when I fly")
        .await
        .unwrap();
    assert!(writer.drain().await.is_empty());
}

#[tokio::test]
async fn ingest_run_extracts_and_summarizes() {
    let extract = ScriptedProvider::new([facts_reply(json!([
        {"category": "user", "key": "home_city", "value": "Porto", "confidence": 0.9}
    ]))]);
    let summarize = ScriptedProvider::new([tool_calls([ToolCall::new(
        "e",
        "record_episode",
        json!({"summary": "User shared that they live in Porto.", "topics": ["home"]}),
    )])]);
    let memory = memory_with(
        Some(FactExtractor::new(extract, "m")),
        Some(EpisodeSummarizer::new(summarize, "m")),
    );
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(ScriptedProvider::new([text("Porto is lovely.")]))
        .build()
        .unwrap();
    let outcome = agent
        .run(SESSION, "I live in Porto these days, by the river")
        .await
        .unwrap();
    let report = memory.ingest_run(&outcome).await.unwrap();
    assert_eq!(report.facts.len(), 1);
    assert!(report.episode.is_some());
    assert_eq!(report.usage.input_tokens, 2 * SCRIPTED_USAGE.input_tokens);
    let fact = &memory.facts(SESSION).await.unwrap()[0];
    assert_eq!(fact.source.run_id, Some(outcome.state.run_id.to_string()));

    let bare = memory_with(None, None);
    assert!(matches!(
        bare.ingest_run(&outcome).await,
        Err(MemoryError::NotConfigured(_))
    ));
}

#[tokio::test]
async fn remember_and_recall_tools_work_inside_an_agent_run() {
    let memory = Memory::builder(
        Arc::new(InMemoryMemoryStore::new()),
        Arc::new(HashEmbedder::default()),
    )
    .scope_resolver(|session: &str| format!("user:{session}"))
    .build();
    let provider = Arc::new(ScriptedProvider::new([
        tool_calls([call(
            "c1",
            "remember",
            json!({"category": "preference", "key": "favorite color", "value": "teal"}),
        )]),
        tool_calls([call(
            "c2",
            "remember",
            json!({"category": "preference", "key": "favorite color", "value": "teal"}),
        )]),
        tool_calls([call("c3", "recall", json!({"query": "favorite color"}))]),
        text("Your favorite color is teal."),
    ]));
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .provider(provider.clone())
        .tool(RememberTool::new(memory.clone()))
        .tool(RecallTool::new(memory.clone()))
        .build()
        .unwrap();
    let spec = agent
        .tool_specs()
        .iter()
        .find(|s| s.name == "remember")
        .unwrap();
    assert_eq!(spec.parameters["type"], "object");

    let outcome = agent
        .run(SESSION, "Remember my favorite color")
        .await
        .unwrap();
    assert!(outcome.is_completed());
    assert!(outcome.receipts().iter().all(|r| r.succeeded()));
    let recall_receipt = outcome
        .receipts()
        .iter()
        .find(|r| r.tool == "recall")
        .unwrap();
    assert!(recall_receipt.read_only);
    assert_eq!(recall_receipt.details["facts"][0]["value"], "teal");

    let tool_text: Vec<&str> = outcome
        .state
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(tool_text[0], "Saved to memory.");
    assert_eq!(tool_text[1], "Already in memory; refreshed.");
    assert!(tool_text[2].contains("favorite_color: teal"));

    // Scoped through the resolver; idempotent writes left one fact.
    let scope = format!("user:{SESSION}");
    assert_eq!(memory.facts(&scope).await.unwrap().len(), 1);
    assert!(memory.facts(SESSION).await.unwrap().is_empty());
}

#[tokio::test]
async fn remember_tool_reports_invalid_input_as_tool_error() {
    let memory = memory_with(None, None);
    let tool = RememberTool::new(memory);
    let out = aidaemon_core::Tool::call(
        &tool,
        json!({"category": "user", "key": "  ", "value": "x"}),
        &aidaemon_core::testing::tool_context(),
    )
    .await
    .unwrap();
    assert!(out.is_error);
}

#[tokio::test]
async fn memory_recall_injects_into_requests_never_into_state() {
    let memory = memory_with(None, None);
    memory
        .remember(NewFact::new(SESSION, "person", "daughters", "Mia and Zoe"))
        .await
        .unwrap();
    memory
        .remember(NewFact::new(SESSION, "user", "employer", "Acme Corp"))
        .await
        .unwrap();
    memory
        .add_episode(aidaemon_memory::NewEpisode {
            scope: SESSION.into(),
            session_id: SESSION.into(),
            summary: "Planned a party for the daughters".into(),
            topics: vec!["party".into()],
            importance: 0.6,
        })
        .await
        .unwrap();

    let provider = Arc::new(ScriptedProvider::new([text("They are Mia and Zoe.")]));
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .model("m")
        .instructions("You are helpful.")
        .provider(provider.clone())
        .hook(MemoryInjection::new(MemoryRecall::new(memory.clone())))
        .build()
        .unwrap();
    let outcome = agent
        .run(SESSION, "What are my daughters' names?")
        .await
        .unwrap();
    assert!(outcome.is_completed());

    let request = &provider.requests()[0];
    let note = request
        .messages
        .iter()
        .find(|m| m.role == Role::System && m.content.contains("Recalled memory"))
        .expect("memory note injected");
    assert!(note.content.contains("daughters: Mia and Zoe"));
    assert!(note.content.contains("Earlier conversation"));
    assert!(
        !note.content.contains("Acme Corp"),
        "irrelevant fact injected"
    );

    assert!(outcome
        .state
        .messages
        .iter()
        .all(|m| !m.content.contains("Recalled memory")));
    assert_eq!(outcome.state.messages.len(), 2);
}
