//! Runner tests: TestChannel + ScriptedProvider + a real Agent.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aidaemon_channels::testing::{direct, group, message, TestChannel};
use aidaemon_channels::{
    AccessPolicy, ChannelError, ChannelRunner, EventRouter, MemorySessionIndex, RunnerConfig,
    SessionIndex, APPROVE_CHOICE, DENY_CHOICE,
};
use aidaemon_core::testing::{call, spec, text, tool_calls, ScriptedProvider, StreamingProvider};
use aidaemon_core::{
    Agent, FnTool, Hook, HookContext, MemoryStore, ModelRequest, ModelResponse, Provider,
    ProviderError, Role, RunStore, ToolAnnotations, ToolCall, ToolDecision, ToolInfo, ToolOutput,
};
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::{Barrier, Semaphore};
use tokio_util::sync::CancellationToken;

const OWNER: &str = "synthetic-user-1";
const FRIEND: &str = "synthetic-user-2";
const STRANGER: &str = "synthetic-stranger";
const WAIT: Duration = Duration::from_secs(5);

// ---------- fixtures ----------

/// Interrupts every call to `deploy` for a human decision.
struct AskBeforeDeploy;

#[async_trait]
impl Hook for AskBeforeDeploy {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        call: &ToolCall,
        _tool: &ToolInfo,
    ) -> ToolDecision {
        if call.name == "deploy" {
            ToolDecision::Interrupt {
                reason: "deploys need a human".into(),
                kind: aidaemon_core::InterruptKind::default(),
            }
        } else {
            ToolDecision::Allow
        }
    }
}

/// Holds every model call until a permit is released.
struct GatedProvider {
    inner: Arc<ScriptedProvider>,
    gate: Arc<Semaphore>,
    entered: Arc<AtomicUsize>,
}

#[async_trait]
impl Provider for GatedProvider {
    fn name(&self) -> &str {
        "gated"
    }
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.entered.fetch_add(1, Ordering::SeqCst);
        self.gate.acquire().await.expect("gate closed").forget();
        self.inner.complete(request).await
    }
}

/// Every model call waits until two calls are in flight at once.
struct BarrierProvider {
    inner: Arc<ScriptedProvider>,
    barrier: Arc<Barrier>,
}

#[async_trait]
impl Provider for BarrierProvider {
    fn name(&self) -> &str {
        "barrier"
    }
    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        if tokio::time::timeout(Duration::from_secs(1), self.barrier.wait())
            .await
            .is_err()
        {
            return Err(ProviderError::Other("runs were not concurrent".into()));
        }
        self.inner.complete(request).await
    }
}

fn deploy_tool(count: Arc<AtomicUsize>) -> impl aidaemon_core::Tool {
    FnTool::new(
        spec("deploy", "Deploy the app", &["target"]),
        ToolAnnotations::DESTRUCTIVE,
        move |_args| {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutput::text("deployed"))
            }
        },
    )
}

fn slow_tool() -> impl aidaemon_core::Tool {
    FnTool::new(
        spec("slow", "Takes a while", &[]),
        ToolAnnotations::WRITE,
        |_args| async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            Ok(ToolOutput::text("slow done"))
        },
    )
}

struct Setup {
    provider: Arc<dyn Provider>,
    channel: Arc<TestChannel>,
    config: RunnerConfig,
    policy: AccessPolicy,
    store: Option<(Arc<MemoryStore>, Arc<MemorySessionIndex>)>,
    deploys: Arc<AtomicUsize>,
}

impl Setup {
    fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            channel: Arc::new(TestChannel::new("telegram")),
            config: RunnerConfig::default(),
            policy: AccessPolicy::new()
                .owner("telegram", OWNER)
                .allow("telegram", FRIEND),
            store: None,
            deploys: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn channel(mut self, channel: TestChannel) -> Self {
        self.channel = Arc::new(channel);
        self
    }

    fn config(mut self, f: impl FnOnce(&mut RunnerConfig)) -> Self {
        f(&mut self.config);
        self
    }

    fn start(self) -> Running {
        let router = EventRouter::new();
        let agent = Agent::builder("assistant")
            .allow_unguarded()
            .provider(self.provider.clone())
            .model("synthetic-model")
            .instructions("You help Alice Rivera.")
            .tool(deploy_tool(self.deploys.clone()))
            .tool(slow_tool())
            .hook(AskBeforeDeploy)
            .event_sink(router.clone())
            .build()
            .expect("agent builds");
        let mut builder = ChannelRunner::builder(agent)
            .channel(self.channel.clone())
            .access(self.policy)
            .event_router(router)
            .config(self.config);
        if let Some((store, index)) = &self.store {
            builder = builder
                .store(store.clone() as Arc<dyn RunStore>)
                .session_index(index.clone() as Arc<dyn SessionIndex>);
        }
        let runner = builder.build().expect("runner builds");
        let shutdown = CancellationToken::new();
        let handle = {
            let shutdown = shutdown.clone();
            tokio::spawn(async move { runner.run(shutdown).await })
        };
        Running {
            channel: self.channel,
            deploys: self.deploys,
            shutdown,
            handle,
        }
    }
}

struct Running {
    channel: Arc<TestChannel>,
    deploys: Arc<AtomicUsize>,
    shutdown: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl Running {
    fn say(&self, conversation: &str, text: &str) {
        self.channel
            .inject_message(message(direct(conversation), OWNER, text));
    }

    async fn wait_text(&self, conversation: &str, needle: &str) -> bool {
        self.channel
            .wait_for(WAIT, |c| {
                c.sent_texts(conversation)
                    .iter()
                    .any(|t| t.contains(needle))
            })
            .await
    }

    async fn stop(self) {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.handle)
            .await
            .expect("runner stops")
            .expect("runner task ok");
    }
}

fn scripted(replies: Vec<ModelResponse>) -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::new(replies))
}

fn user_texts(request: &ModelRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| m.content.clone())
        .collect()
}

/// The prompt id from a text-fallback approval prompt.
fn fallback_id(texts: &[String]) -> Option<String> {
    texts.iter().find_map(|t| {
        let start = t.find("`/approve ")? + "`/approve ".len();
        let end = t[start..].find('`')?;
        Some(t[start..start + end].to_string())
    })
}

// ---------- tests ----------

#[tokio::test]
async fn direct_message_round_trip() {
    let provider = scripted(vec![text("Hello, Alice Rivera.")]);
    let run = Setup::new(provider.clone()).start();
    run.say("chat-1", "hi there");
    assert!(run.wait_text("chat-1", "Hello, Alice Rivera.").await);
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(user_texts(&requests[0]), vec!["hi there"]);
    // Without edits there is no status message: only the answer.
    assert_eq!(
        run.channel.sent_texts("chat-1"),
        vec!["Hello, Alice Rivera."]
    );
    run.stop().await;
}

#[tokio::test]
async fn group_messages_need_to_address_the_bot() {
    let provider = scripted(vec![text("answered")]);
    let run = Setup::new(provider.clone()).start();
    let mut unaddressed = message(group("group-1"), OWNER, "chatter");
    unaddressed.addressed = false;
    run.channel.inject_message(unaddressed);
    let mut addressed = message(group("group-1"), OWNER, "@bot question");
    addressed.addressed = true;
    let reply_to = addressed.message_id.clone();
    run.channel.inject_message(addressed);
    assert!(run.wait_text("group-1", "answered").await);
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(user_texts(&requests[0]), vec!["@bot question"]);
    // In groups the answer replies to the triggering message.
    let sent = run.channel.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].message.reply_to.as_deref(), Some(reply_to.as_str()));
    run.stop().await;
}

#[tokio::test]
async fn unknown_senders_are_refused_and_bots_ignored() {
    let provider = scripted(vec![text("never")]);
    let run = Setup::new(provider.clone()).start();
    run.channel
        .inject_message(message(direct("chat-9"), STRANGER, "let me in"));
    let mut bot = message(direct("chat-8"), OWNER, "beep");
    bot.sender.is_bot = true;
    run.channel.inject_message(bot);
    assert!(run.wait_text("chat-9", "Sorry").await);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        provider.requests().is_empty(),
        "the agent must not be called"
    );
    assert!(run.channel.sent_texts("chat-8").is_empty());
    run.stop().await;
}

#[tokio::test]
async fn messages_during_a_run_are_queued_in_order() {
    let inner = scripted(vec![text("r1"), text("r2"), text("r3")]);
    let gate = Arc::new(Semaphore::new(0));
    let entered = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(GatedProvider {
        inner: inner.clone(),
        gate: gate.clone(),
        entered: entered.clone(),
    });
    let run = Setup::new(provider).start();
    run.say("chat-1", "m1");
    assert!(
        run.channel
            .wait_for(WAIT, |_| entered.load(Ordering::SeqCst) == 1)
            .await
    );
    run.say("chat-1", "m2");
    run.say("chat-1", "/status");
    assert!(
        run.wait_text("chat-1", "A run is active. Queued messages: 1.")
            .await
    );
    run.say("chat-1", "m3");
    tokio::time::sleep(Duration::from_millis(30)).await;
    gate.add_permits(3);
    assert!(run.wait_text("chat-1", "r3").await);
    let replies: Vec<String> = run
        .channel
        .sent_texts("chat-1")
        .into_iter()
        .filter(|t| t.starts_with('r'))
        .collect();
    assert_eq!(replies, vec!["r1", "r2", "r3"]);
    let lasts: Vec<String> = inner
        .requests()
        .iter()
        .map(|r| user_texts(r).last().cloned().unwrap_or_default())
        .collect();
    assert_eq!(lasts, vec!["m1", "m2", "m3"]);
    run.stop().await;
}

#[tokio::test]
async fn full_queue_drops_the_oldest_with_a_notice() {
    let inner = scripted(vec![text("r1"), text("r3")]);
    let gate = Arc::new(Semaphore::new(0));
    let entered = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(GatedProvider {
        inner: inner.clone(),
        gate: gate.clone(),
        entered: entered.clone(),
    });
    let run = Setup::new(provider)
        .config(|c| c.max_queued_per_session = 1)
        .start();
    run.say("chat-1", "m1");
    assert!(
        run.channel
            .wait_for(WAIT, |_| entered.load(Ordering::SeqCst) == 1)
            .await
    );
    run.say("chat-1", "m2");
    run.say("chat-1", "m3");
    assert!(run.wait_text("chat-1", "dropped the oldest").await);
    gate.add_permits(2);
    assert!(run.wait_text("chat-1", "r3").await);
    let lasts: Vec<String> = inner
        .requests()
        .iter()
        .map(|r| user_texts(r).last().cloned().unwrap_or_default())
        .collect();
    assert_eq!(lasts, vec!["m1", "m3"]);
    run.stop().await;
}

#[tokio::test]
async fn different_sessions_run_concurrently() {
    let inner = scripted(vec![text("done"), text("done")]);
    let provider = Arc::new(BarrierProvider {
        inner,
        barrier: Arc::new(Barrier::new(2)),
    });
    let run = Setup::new(provider).start();
    run.say("chat-1", "first");
    run.say("chat-2", "second");
    assert!(run.wait_text("chat-1", "done").await);
    assert!(run.wait_text("chat-2", "done").await);
    run.stop().await;
}

#[tokio::test]
async fn global_concurrency_limit_serializes_sessions() {
    // With one permit the barrier can never be reached by two runs at once.
    let inner = scripted(vec![text("done"), text("done")]);
    let provider = Arc::new(BarrierProvider {
        inner,
        barrier: Arc::new(Barrier::new(2)),
    });
    let run = Setup::new(provider)
        .config(|c| c.max_concurrent_runs = 1)
        .start();
    run.say("chat-1", "first");
    run.say("chat-2", "second");
    assert!(run.wait_text("chat-1", "not concurrent").await);
    run.stop().await;
}

#[tokio::test]
async fn conversation_continues_from_the_previous_run() {
    let provider = scripted(vec![text("first answer"), text("second answer")]);
    let run = Setup::new(provider.clone()).start();
    run.say("chat-1", "first question");
    assert!(run.wait_text("chat-1", "first answer").await);
    run.say("chat-1", "second question");
    assert!(run.wait_text("chat-1", "second answer").await);
    let requests = provider.requests();
    let second = &requests[1];
    let contents: Vec<&str> = second.messages.iter().map(|m| m.content.as_str()).collect();
    assert!(contents.contains(&"first question"), "{contents:?}");
    assert!(contents.contains(&"first answer"), "{contents:?}");
    assert_eq!(contents.last(), Some(&"second question"));
    run.stop().await;
}

#[tokio::test]
async fn long_answers_are_split_without_breaking_code_fences() {
    let code = format!("```rust\n{}```", "let value = 42;\n".repeat(2));
    let answer = format!(
        "{}\n\n{}\n\n{}",
        "Intro paragraph with some words.", code, "Closing paragraph that is long enough."
    );
    let provider = scripted(vec![text(answer.clone())]);
    let run = Setup::new(provider)
        .channel(TestChannel::new("telegram").with_max_len(60))
        .start();
    run.say("chat-1", "show code");
    assert!(run.wait_text("chat-1", "Closing paragraph").await);
    let parts = run.channel.sent_texts("chat-1");
    assert!(parts.len() >= 3, "{parts:#?}");
    assert!(parts.iter().all(|p| p.chars().count() <= 60), "{parts:#?}");
    assert!(parts.contains(&code), "fence kept whole: {parts:#?}");
    for p in &parts {
        assert_eq!(p.matches("```").count() % 2, 0, "unbalanced: {p}");
    }
    run.stop().await;
}

#[tokio::test]
async fn status_message_is_edited_with_throttling_when_supported() {
    let interval = Duration::from_millis(60);
    let provider = Arc::new(StreamingProvider(scripted(vec![
        tool_calls([
            call("c1", "slow", json!({})),
            call("c2", "slow", json!({})),
            call("c3", "slow", json!({})),
        ]),
        text("all three finished"),
    ])));
    let run = Setup::new(provider)
        .channel(
            TestChannel::new("telegram")
                .with_edits(true)
                .with_typing(true),
        )
        .config(|c| {
            c.status_edit_interval = interval;
            c.typing_interval = Duration::from_millis(100);
        })
        .start();
    run.say("chat-1", "do slow things");
    assert!(run.wait_text("chat-1", "all three finished").await);
    // Give the final edit a moment (it follows the answer's computation).
    tokio::time::sleep(Duration::from_millis(50)).await;

    let sent = run.channel.sent_texts("chat-1");
    assert_eq!(
        sent.len(),
        2,
        "one status message and one answer: {sent:#?}"
    );
    assert!(sent[0].contains("Working"));
    let edits = run.channel.edits();
    assert!(edits.len() >= 3, "progress edits expected: {edits:#?}");
    let status_id = &run.channel.sent()[0].message_ref;
    assert!(edits.iter().all(|e| &e.message_ref == status_id));
    // Every progress edit (all but the final one) respects the interval.
    let progress: Vec<_> = edits[..edits.len() - 1].to_vec();
    for pair in progress.windows(2) {
        let gap = pair[1].at.duration_since(pair[0].at);
        assert!(gap >= interval - Duration::from_millis(2), "gap {gap:?}");
    }
    assert!(progress.iter().any(|e| e.text.contains("`slow`")));
    let last = edits.last().expect("final edit");
    assert!(last.text.contains("Done"), "{}", last.text);
    assert!(run.channel.typing_count() >= 1);
    run.stop().await;
}

#[tokio::test]
async fn no_status_message_without_edit_support() {
    let provider = Arc::new(StreamingProvider(scripted(vec![
        tool_calls([call("c1", "slow", json!({}))]),
        text("finished"),
    ])));
    let run = Setup::new(provider)
        .config(|c| c.status_edit_interval = Duration::from_millis(10))
        .start();
    run.say("chat-1", "go");
    assert!(run.wait_text("chat-1", "finished").await);
    assert_eq!(run.channel.sent_texts("chat-1"), vec!["finished"]);
    assert!(run.channel.edits().is_empty());
    assert_eq!(run.channel.typing_count(), 0);
    run.stop().await;
}

fn deploy_script(final_text: &str) -> Arc<ScriptedProvider> {
    scripted(vec![
        tool_calls([call("d1", "deploy", json!({"target": "staging"}))]),
        text(final_text),
    ])
}

#[tokio::test]
async fn approval_buttons_resume_the_run() {
    let provider = deploy_script("Deployed to staging.");
    let run = Setup::new(provider.clone())
        .channel(
            TestChannel::new("telegram")
                .with_buttons(true)
                .with_edits(true),
        )
        .start();
    run.say("chat-1", "deploy please");
    assert!(run.channel.wait_for(WAIT, |c| c.prompts().len() == 1).await);
    let prompt = run.channel.prompts().remove(0);
    assert!(prompt.text.contains("`deploy`"));
    assert!(prompt.text.contains("\"target\": \"staging\""));
    assert!(prompt.text.contains("deploys need a human"));
    let choices: Vec<&str> = prompt.choices.iter().map(|c| c.id.as_str()).collect();
    assert_eq!(choices, vec![APPROVE_CHOICE, DENY_CHOICE]);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);

    run.channel
        .press(&prompt.id, APPROVE_CHOICE, FRIEND, direct("chat-1"));
    assert!(run.wait_text("chat-1", "Deployed to staging.").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 1);
    // The prompt message was marked as settled.
    assert!(run
        .channel
        .edits()
        .iter()
        .any(|e| e.text.contains("**Approved**")));
    run.stop().await;
}

#[tokio::test]
async fn approval_falls_back_to_text_commands_without_buttons() {
    let provider = deploy_script("Deployed via command.");
    let run = Setup::new(provider).start();
    run.say("chat-1", "deploy please");
    assert!(run.wait_text("chat-1", "/approve ").await);
    let id = fallback_id(&run.channel.sent_texts("chat-1")).expect("prompt id");
    assert!(run.channel.prompts().is_empty());
    run.say("chat-1", &format!("/approve {id}"));
    assert!(run.wait_text("chat-1", "Deployed via command.").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 1);
    run.stop().await;
}

#[tokio::test]
async fn deny_tells_the_model_and_skips_the_tool() {
    let provider = deploy_script("Okay, I will not deploy.");
    let run = Setup::new(provider.clone())
        .channel(TestChannel::new("telegram").with_buttons(true))
        .start();
    run.say("chat-1", "deploy please");
    assert!(run.channel.wait_for(WAIT, |c| c.prompts().len() == 1).await);
    let prompt = run.channel.prompts().remove(0);
    run.channel
        .press(&prompt.id, DENY_CHOICE, OWNER, direct("chat-1"));
    assert!(run.wait_text("chat-1", "Okay, I will not deploy.").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);
    let requests = provider.requests();
    let tool_result = requests[1]
        .messages
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("tool result");
    assert!(
        tool_result.content.contains("denied"),
        "{}",
        tool_result.content
    );
    run.stop().await;
}

#[tokio::test]
async fn stale_and_duplicate_approvals_are_ignored_with_a_notice() {
    let provider = deploy_script("Deployed once.");
    let run = Setup::new(provider.clone())
        .channel(TestChannel::new("telegram").with_buttons(true))
        .start();
    run.say("chat-1", "/approve deadbeef");
    assert!(run.wait_text("chat-1", "`deadbeef` is not pending").await);

    run.say("chat-1", "deploy please");
    assert!(run.channel.wait_for(WAIT, |c| c.prompts().len() == 1).await);
    let prompt = run.channel.prompts().remove(0);
    // Same id from another conversation does not count.
    run.channel
        .press(&prompt.id, APPROVE_CHOICE, OWNER, direct("chat-2"));
    assert!(run.wait_text("chat-2", "is not pending here").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);

    run.channel
        .press(&prompt.id, APPROVE_CHOICE, OWNER, direct("chat-1"));
    run.channel
        .press(&prompt.id, APPROVE_CHOICE, OWNER, direct("chat-1"));
    assert!(run.wait_text("chat-1", "Deployed once.").await);
    assert!(run.wait_text("chat-1", "already answered or expired").await);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(run.deploys.load(Ordering::SeqCst), 1);
    assert_eq!(provider.requests().len(), 2);
    run.stop().await;
}

#[tokio::test]
async fn approvals_from_users_who_may_not_approve_are_ignored() {
    let provider = deploy_script("Deployed after owner approval.");
    let run = Setup::new(provider)
        .channel(TestChannel::new("telegram").with_buttons(true))
        .start();
    run.say("chat-1", "deploy please");
    assert!(run.channel.wait_for(WAIT, |c| c.prompts().len() == 1).await);
    let prompt = run.channel.prompts().remove(0);
    run.channel
        .press(&prompt.id, APPROVE_CHOICE, STRANGER, direct("chat-1"));
    run.channel.inject_message(message(
        direct("chat-1"),
        STRANGER,
        &format!("/approve {}", prompt.id),
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);
    // Still pending: the owner can approve it.
    run.channel
        .press(&prompt.id, APPROVE_CHOICE, OWNER, direct("chat-1"));
    assert!(
        run.wait_text("chat-1", "Deployed after owner approval.")
            .await
    );
    assert_eq!(run.deploys.load(Ordering::SeqCst), 1);
    run.stop().await;
}

#[tokio::test]
async fn stop_cancels_the_active_run() {
    let inner = scripted(vec![text("too late")]);
    let gate = Arc::new(Semaphore::new(0));
    let entered = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(GatedProvider {
        inner,
        gate: gate.clone(),
        entered: entered.clone(),
    });
    let run = Setup::new(provider).start();
    run.say("chat-1", "long task");
    assert!(
        run.channel
            .wait_for(WAIT, |_| entered.load(Ordering::SeqCst) == 1)
            .await
    );
    run.say("chat-1", "/stop");
    assert!(run.wait_text("chat-1", "Stopped.").await);
    run.say("chat-1", "/status");
    assert!(run.wait_text("chat-1", "Idle.").await);
    assert!(!run
        .channel
        .sent_texts("chat-1")
        .iter()
        .any(|t| t.contains("too late")));
    gate.add_permits(10);
    run.stop().await;
}

#[tokio::test]
async fn stop_while_waiting_for_approval_ends_the_run() {
    let provider = deploy_script("never shown");
    let run = Setup::new(provider.clone())
        .channel(TestChannel::new("telegram").with_buttons(true))
        .start();
    run.say("chat-1", "deploy please");
    assert!(run.channel.wait_for(WAIT, |c| c.prompts().len() == 1).await);
    run.say("chat-1", "/stop");
    assert!(run.wait_text("chat-1", "Stopped.").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);
    assert_eq!(provider.requests().len(), 1);
    run.stop().await;
}

#[tokio::test]
async fn new_starts_a_fresh_session() {
    let provider = scripted(vec![text("first answer"), text("fresh answer")]);
    let run = Setup::new(provider.clone()).start();
    run.say("chat-1", "first question");
    assert!(run.wait_text("chat-1", "first answer").await);
    run.say("chat-1", "/new");
    assert!(run.wait_text("chat-1", "Started a new conversation.").await);
    run.say("chat-1", "unrelated question");
    assert!(run.wait_text("chat-1", "fresh answer").await);
    let requests = provider.requests();
    assert_eq!(user_texts(&requests[1]), vec!["unrelated question"]);
    run.stop().await;
}

#[tokio::test]
async fn pending_approvals_expire_with_a_deny() {
    let provider = deploy_script("The approval expired, so I did not deploy.");
    let run = Setup::new(provider.clone())
        .channel(TestChannel::new("telegram").with_buttons(true))
        .config(|c| c.approval_timeout = Duration::from_millis(100))
        .start();
    run.say("chat-1", "deploy please");
    assert!(run.channel.wait_for(WAIT, |c| c.prompts().len() == 1).await);
    let prompt = run.channel.prompts().remove(0);
    assert!(
        run.wait_text("chat-1", "expired, so the action was denied")
            .await
    );
    assert!(run.wait_text("chat-1", "I did not deploy.").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);
    let requests = provider.requests();
    let tool_result = requests[1]
        .messages
        .iter()
        .find(|m| m.role == Role::Tool)
        .expect("tool result");
    assert!(
        tool_result.content.contains("expired"),
        "{}",
        tool_result.content
    );
    // A late press is stale.
    run.channel
        .press(&prompt.id, APPROVE_CHOICE, OWNER, direct("chat-1"));
    assert!(run.wait_text("chat-1", "already answered or expired").await);
    assert_eq!(run.deploys.load(Ordering::SeqCst), 0);
    run.stop().await;
}

#[tokio::test]
async fn run_store_and_session_index_continue_across_runners() {
    let store = Arc::new(MemoryStore::new());
    let index = Arc::new(MemorySessionIndex::new());
    let provider = scripted(vec![text("stored answer"), text("after restart")]);

    let mut setup = Setup::new(provider.clone());
    setup.store = Some((store.clone(), index.clone()));
    let run = setup.start();
    run.say("chat-1", "remember this");
    assert!(run.wait_text("chat-1", "stored answer").await);
    run.stop().await;

    let run_id = index
        .latest_run("telegram:chat-1")
        .await
        .expect("index records the run");
    let saved = store
        .load_state(run_id)
        .await
        .expect("store ok")
        .expect("state saved");
    assert_eq!(saved.session_id, "telegram:chat-1");

    // A fresh runner (as after a restart) with the same store and index.
    let mut setup = Setup::new(provider.clone());
    setup.store = Some((store.clone(), index.clone()));
    let run = setup.start();
    run.say("chat-1", "what did I say?");
    assert!(run.wait_text("chat-1", "after restart").await);
    let requests = provider.requests();
    let contents: Vec<&str> = requests[1]
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect();
    assert!(contents.contains(&"remember this"), "{contents:?}");
    assert!(contents.contains(&"stored answer"), "{contents:?}");
    assert_ne!(index.latest_run("telegram:chat-1").await, Some(run_id));
    run.stop().await;
}

#[tokio::test]
async fn failed_runs_explain_and_keep_partial_text() {
    let provider = Arc::new(ScriptedProvider::with_results(vec![
        Ok(ModelResponse {
            text: "Checking the deploy log first.".into(),
            ..tool_calls([call("s1", "slow", json!({}))])
        }),
        Err(ProviderError::Other("synthetic outage".into())),
    ]));
    let run = Setup::new(provider).start();
    run.say("chat-1", "check it");
    assert!(run.wait_text("chat-1", "model provider").await);
    let texts = run.channel.sent_texts("chat-1");
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Checking the deploy log first.")),
        "{texts:#?}"
    );
    run.stop().await;
}

#[tokio::test]
async fn failing_channels_restart_with_backoff_but_auth_errors_stop_them() {
    let provider = scripted(vec![text("back online")]);
    let channel = TestChannel::new("telegram");
    channel.fail_runs([
        ChannelError::Transport("synthetic disconnect".into()),
        ChannelError::Transport("synthetic disconnect".into()),
    ]);
    let run = Setup::new(provider)
        .channel(channel)
        .config(|c| c.restart_backoff = Duration::from_millis(10))
        .start();
    run.say("chat-1", "hello?");
    assert!(run.wait_text("chat-1", "back online").await);
    assert_eq!(run.channel.run_count(), 3);
    run.stop().await;

    let channel = TestChannel::new("telegram");
    channel.fail_runs([ChannelError::Auth("revoked".into())]);
    let run = Setup::new(scripted(vec![]))
        .channel(channel)
        .config(|c| c.restart_backoff = Duration::from_millis(10))
        .start();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(run.channel.run_count(), 1);
    run.stop().await;
}

#[test]
fn builder_rejects_bad_setups() {
    let agent = Agent::builder("assistant")
        .allow_unguarded()
        .provider(ScriptedProvider::new(vec![]))
        .model("synthetic-model")
        .build()
        .expect("agent");
    let err = ChannelRunner::builder(agent.clone()).build().err();
    assert_eq!(err, Some(aidaemon_channels::RunnerBuildError::NoChannels));
    let err = ChannelRunner::builder(agent)
        .channel(Arc::new(TestChannel::new("telegram")))
        .channel(Arc::new(TestChannel::new("telegram")))
        .build()
        .err();
    assert!(matches!(
        err,
        Some(aidaemon_channels::RunnerBuildError::DuplicateChannel(_))
    ));
}
