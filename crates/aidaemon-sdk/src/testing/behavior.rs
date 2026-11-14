//! The shared behavior suite: what any agent assembled from the SDK should
//! do, checked with a scripted model and an in-process channel, so it runs
//! anywhere without keys.
//!
//! You give it a [`Subject`]: a name and a factory that assembles your
//! agent from the [`Parts`] a case hands it (a scripted provider, the
//! case's tools and wiring, a directory for the memory to keep its store
//! in). Your factory chooses everything else: instructions, safeguards,
//! the memory system. Each case is a function returning a [`Verdict`].
//!
//! | Case | Requires | Checks |
//! |---|---|---|
//! | `memory_write` | any memory | the memory's write tool stores a fact, and the agent's next run sees it |
//! | `memory_recall` | any memory | a new agent on the same store gets the fact in its prompt; one on an empty store does not |
//! | `tool_use` | | a `#[tool]` result reaches the next model request; invalid arguments are refused before the tool runs |
//! | `approvals` | | a destructive call waits for the owner; approved it runs once, declined it never runs and the run is not a success |
//! | `sub_agents` | | a child agent's answer reaches the parent, and its usage counts against the parent's budget |
//! | `channel_round_trip` | | a message into a channel gets a reply out of it, through an approval asked and answered on the channel |
//! | `full_memory_tools` | `FullMemory` | the daemon memory's four tools are offered |
//! | `full_memory_manage` | `FullMemory` | `manage_memories` lists a stored fact, and forgetting it (after approval) removes it from recall |
//!
//! Cases tagged [`Requires::FullMemory`] test aidaemon's own memory system
//! (`aidaemon_memory::full::FullMemory`) and are skipped for a subject that
//! is not [`Subject::full_memory`]. The memory cases written against the
//! `MemorySystem` contract only need a tool that stores a fact from
//! `{category, key, value}` ([`Subject::write_tool`]). Every case runs in
//! one session ([`OWNER_SESSION`]): a memory system may keep sessions
//! apart unless the host maps them to one memory, as `ScopedMemory` does.
//!
//! ```no_run
//! # async fn demo() {
//! use aidaemon_sdk::testing::behavior::{run, Subject};
//!
//! let subject = Subject::new("default preset", |parts| async move {
//!     let preset = aidaemon_sdk::preset(parts.id())
//!         .memory_path(parts.memory_dir().join("memory.db"));
//!     Ok(parts.finish(preset).build().await?)
//! })
//! .full_memory();
//! let report = run(&subject).await;
//! println!("{report}");
//! assert!(report.all_passed());
//! # }
//! ```

use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aidaemon_agents::{AgentTool, ChildBudget};
use aidaemon_channels::testing::{TestChannel, direct, message};
use aidaemon_channels::{APPROVE_CHOICE, AccessPolicy, ChannelRunner, EventRouter};
use serde_json::{Value, json};

use super::{SCRIPTED_USAGE, ScriptedProvider, call, text, tool_calls};
use crate::{
    Agent, AgentBuilder, CancellationToken, FnTool, Limits, Message, ModelRequest, Preset,
    ReceiptStatus, Resolution, Role, RunOutcome, RunStatus, Tool, ToolAnnotations, ToolOutput,
    ToolSpec, tool,
};

/// An error from a subject's factory.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
type Factory = Arc<dyn Fn(Parts) -> BoxFuture<Result<Agent, BoxError>> + Send + Sync>;
type Configure = Box<dyn FnOnce(AgentBuilder) -> AgentBuilder + Send>;

/// The session every case runs as.
pub const OWNER_SESSION: &str = "parity:synthetic-owner";

/// What a case hands the factory for one agent.
pub struct Parts {
    id: String,
    memory_dir: PathBuf,
    provider: Arc<ScriptedProvider>,
    tools: Vec<Arc<dyn Tool>>,
    configure: Vec<Configure>,
}

impl Parts {
    /// The agent's id (`"assistant"`, or `"researcher"` for a child).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// A directory for the memory system's store. Agents built for the same
    /// case with the same directory must share what the memory keeps.
    pub fn memory_dir(&self) -> &Path {
        &self.memory_dir
    }

    /// Add the case's scripted provider (model `"scripted"`), its tools and
    /// its wiring (an event sink, limits) to `preset`.
    pub fn finish(self, mut preset: Preset) -> Preset {
        preset = preset
            .provider_arc(self.provider)
            .model("scripted")
            .tools(self.tools);
        for configure in self.configure {
            preset = preset.configure(configure);
        }
        preset
    }
}

/// An agent assembly under test.
#[derive(Clone)]
pub struct Subject {
    name: String,
    factory: Factory,
    write_tool: String,
    full_memory: bool,
}

impl Subject {
    /// `factory` builds one agent from [`Parts`]; it is called once per
    /// agent a case needs.
    pub fn new<F, Fut>(name: impl Into<String>, factory: F) -> Self
    where
        F: Fn(Parts) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Agent, BoxError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            factory: Arc::new(move |parts| Box::pin(factory(parts))),
            write_tool: "remember".into(),
            full_memory: false,
        }
    }

    /// The memory system's tool that stores a fact from
    /// `{"category", "key", "value"}` (default `remember`, as
    /// `ScopedMemory` names it).
    pub fn write_tool(mut self, name: impl Into<String>) -> Self {
        self.write_tool = name.into();
        self
    }

    /// The subject uses aidaemon's full memory system: its write tool is
    /// `remember_fact`, and the [`Requires::FullMemory`] cases run.
    pub fn full_memory(mut self) -> Self {
        self.full_memory = true;
        self.write_tool = "remember_fact".into();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    async fn build(
        &self,
        id: &str,
        dir: &Path,
        provider: &Arc<ScriptedProvider>,
        tools: Vec<Arc<dyn Tool>>,
        configure: Vec<Configure>,
    ) -> Result<Agent, String> {
        let parts = Parts {
            id: id.to_string(),
            memory_dir: dir.to_path_buf(),
            provider: provider.clone(),
            tools,
            configure,
        };
        (self.factory)(parts)
            .await
            .map_err(|err| format!("the factory failed for `{id}`: {err}"))
    }
}

/// What a case needs from the subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requires {
    /// Any assembly with a memory system.
    Contract,
    /// aidaemon's full memory system and its own tools.
    FullMemory,
}

/// A case's result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Passed,
    Failed(String),
    /// Not run: the subject lacks what the case requires.
    Skipped(Requires),
}

impl Verdict {
    pub fn is_failed(&self) -> bool {
        matches!(self, Verdict::Failed(_))
    }
}

type CaseFn = fn(Subject, PathBuf) -> BoxFuture<Result<(), String>>;

/// One case of the suite.
#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub requires: Requires,
    run: CaseFn,
}

/// Every case, in order.
pub fn cases() -> Vec<Case> {
    fn case(name: &'static str, requires: Requires, run: CaseFn) -> Case {
        Case {
            name,
            requires,
            run,
        }
    }
    use Requires::{Contract, FullMemory};
    vec![
        case("memory_write", Contract, |s, d| {
            Box::pin(memory_write(s, d))
        }),
        case("memory_recall", Contract, |s, d| {
            Box::pin(memory_recall(s, d))
        }),
        case("tool_use", Contract, |s, d| Box::pin(tool_use(s, d))),
        case("approvals", Contract, |s, d| Box::pin(approvals(s, d))),
        case("sub_agents", Contract, |s, d| Box::pin(sub_agents(s, d))),
        case("channel_round_trip", Contract, |s, d| {
            Box::pin(channel_round_trip(s, d))
        }),
        case("full_memory_tools", FullMemory, |s, d| {
            Box::pin(full_memory_tools(s, d))
        }),
        case("full_memory_manage", FullMemory, |s, d| {
            Box::pin(full_memory_manage(s, d))
        }),
    ]
}

/// Run one case by name.
///
/// # Panics
/// If no case has that name.
pub async fn run_case(subject: &Subject, name: &str) -> Verdict {
    let case = cases()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no behavior case named {name:?}"));
    run_one(subject, &case).await
}

async fn run_one(subject: &Subject, case: &Case) -> Verdict {
    if case.requires == Requires::FullMemory && !subject.full_memory {
        return Verdict::Skipped(case.requires);
    }
    let dir = match ScratchDir::new(case.name) {
        Ok(dir) => dir,
        Err(err) => return Verdict::Failed(format!("scratch directory: {err}")),
    };
    // Each case on its own task, so a panic in the subject is a failure,
    // not the end of the suite.
    let task = tokio::spawn((case.run)(subject.clone(), dir.path().to_path_buf()));
    let result = match tokio::time::timeout(CASE_TIMEOUT, task).await {
        Err(_) => Err(format!("timed out after {CASE_TIMEOUT:?}")),
        Ok(Err(join)) => Err(format!("panicked: {join}")),
        Ok(Ok(result)) => result,
    };
    match result {
        Ok(()) => Verdict::Passed,
        Err(reason) => Verdict::Failed(reason),
    }
}

/// Run every case against `subject`.
pub async fn run(subject: &Subject) -> Report {
    let mut results = Vec::new();
    for case in cases() {
        results.push((case.name, run_one(subject, &case).await));
    }
    Report {
        subject: subject.name.clone(),
        results,
    }
}

/// The verdict of every case for one subject.
#[derive(Debug, Clone)]
pub struct Report {
    pub subject: String,
    pub results: Vec<(&'static str, Verdict)>,
}

impl Report {
    /// No case failed (skipped ones do not count).
    pub fn all_passed(&self) -> bool {
        !self.results.iter().any(|(_, verdict)| verdict.is_failed())
    }

    /// The names of the cases with this verdict kind.
    pub fn named(&self, pick: impl Fn(&Verdict) -> bool) -> Vec<&'static str> {
        self.results
            .iter()
            .filter(|(_, verdict)| pick(verdict))
            .map(|(name, _)| *name)
            .collect()
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.subject)?;
        for (name, verdict) in &self.results {
            match verdict {
                Verdict::Passed => writeln!(f, "  pass  {name}")?,
                Verdict::Skipped(requires) => writeln!(f, "  skip  {name} (needs {requires:?})")?,
                Verdict::Failed(reason) => writeln!(f, "  FAIL  {name}: {reason}")?,
            }
        }
        Ok(())
    }
}

const CASE_TIMEOUT: Duration = Duration::from_secs(60);
const CHANNEL_WAIT: Duration = Duration::from_secs(20);
const OWNER: &str = "synthetic-owner";

macro_rules! ensure {
    ($cond:expr, $($why:tt)+) => {
        if !$cond {
            return Err(format!($($why)+));
        }
    };
}

fn scripted(replies: impl IntoIterator<Item = crate::ModelResponse>) -> Arc<ScriptedProvider> {
    Arc::new(ScriptedProvider::new(replies))
}

async fn run_agent(agent: &Agent, session: &str, input: &str) -> Result<RunOutcome, String> {
    agent
        .run(session, input)
        .await
        .map_err(|err| format!("run failed: {err}"))
}

fn request(provider: &ScriptedProvider, index: usize) -> Result<ModelRequest, String> {
    provider.requests().get(index).cloned().ok_or_else(|| {
        format!(
            "the model got {} requests, expected at least {}",
            provider.requests().len(),
            index + 1
        )
    })
}

/// The result the model saw for `call_id`.
fn tool_result<'a>(request: &'a ModelRequest, call_id: &str) -> Option<&'a Message> {
    request
        .messages
        .iter()
        .find(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some(call_id))
}

fn mentions(request: &ModelRequest, needle: &str) -> bool {
    request
        .messages
        .iter()
        .any(|m| m.content.to_lowercase().contains(needle))
}

fn receipt_status(outcome: &RunOutcome, call_id: &str) -> Option<ReceiptStatus> {
    outcome
        .receipts()
        .iter()
        .find(|r| r.call_id == call_id)
        .map(|r| r.status)
}

fn tool_names(agent: &Agent) -> Vec<String> {
    agent.tool_specs().iter().map(|s| s.name.clone()).collect()
}

fn remember(subject: &Subject, id: &str) -> crate::ModelResponse {
    tool_calls([call(
        id,
        &subject.write_tool,
        json!({"category": "preference", "key": "favorite_color", "value": "teal"}),
    )])
}

/// Store a fact with the subject's write tool, in a fresh agent on `dir`.
/// The agent and its provider are returned with one more scripted reply.
async fn store_fact(
    subject: &Subject,
    dir: &Path,
) -> Result<(Agent, Arc<ScriptedProvider>), String> {
    let provider = scripted([remember(subject, "w1"), text("Noted."), text("Teal.")]);
    let agent = subject
        .build("assistant", dir, &provider, Vec::new(), Vec::new())
        .await?;
    ensure!(
        tool_names(&agent).contains(&subject.write_tool),
        "the memory offers no `{}` tool: {:?}",
        subject.write_tool,
        tool_names(&agent)
    );
    let outcome = run_agent(&agent, OWNER_SESSION, "My favorite color is teal.").await?;
    ensure!(
        receipt_status(&outcome, "w1") == Some(ReceiptStatus::Succeeded),
        "the write did not succeed: {:?}",
        outcome.receipts()
    );
    ensure!(
        outcome.is_completed(),
        "the run did not complete: {:?}",
        outcome.status
    );
    let next = request(&provider, 1)?;
    ensure!(
        tool_result(&next, "w1").is_some(),
        "the write's result never reached the model"
    );
    Ok((agent, provider))
}

/// 1. The memory's write tool stores a fact, and it outlives the run: the
///    agent's next run (a fresh transcript) has it in its prompt.
async fn memory_write(subject: Subject, dir: PathBuf) -> Result<(), String> {
    let (agent, provider) = store_fact(&subject, &dir).await?;
    run_agent(&agent, OWNER_SESSION, "What's my favorite color?").await?;
    let next_run = request(&provider, 2)?;
    ensure!(
        !next_run.messages.iter().any(|m| m.role == Role::Tool),
        "the next run carried the last run's transcript"
    );
    ensure!(
        mentions(&next_run, "teal"),
        "the stored fact is not in the next run's prompt"
    );
    Ok(())
}

/// 2. A later run, by a new agent on the same store, gets the fact in its
///    prompt; an agent on an empty store does not.
async fn memory_recall(subject: Subject, dir: PathBuf) -> Result<(), String> {
    let stored = dir.join("stored");
    let empty = dir.join("empty");
    for dir in [&stored, &empty] {
        std::fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    }
    drop(store_fact(&subject, &stored).await?);

    let provider = scripted([text("Teal.")]);
    let agent = subject
        .build("assistant", &stored, &provider, Vec::new(), Vec::new())
        .await?;
    run_agent(&agent, OWNER_SESSION, "What's my favorite color?").await?;
    ensure!(
        mentions(&request(&provider, 0)?, "teal"),
        "the fact is not recalled: {:?}",
        request(&provider, 0)?.messages
    );

    let provider = scripted([text("I don't know.")]);
    let agent = subject
        .build("assistant", &empty, &provider, Vec::new(), Vec::new())
        .await?;
    run_agent(&agent, OWNER_SESSION, "What's my favorite color?").await?;
    ensure!(
        !mentions(&request(&provider, 0)?, "teal"),
        "an agent on an empty store knows the fact: recall does not come from the store"
    );
    Ok(())
}

/// Add two numbers.
#[tool(read_only, crate = crate)]
fn add_numbers(
    /// The first number.
    a: i64,
    /// The second number.
    b: i64,
) -> String {
    format!("sum: {}", a + b)
}

/// 3. A `#[tool]` result reaches the next model request; invalid arguments
///    are refused by validation.
async fn tool_use(subject: Subject, dir: PathBuf) -> Result<(), String> {
    let provider = scripted([
        tool_calls([call("t1", "add_numbers", json!({"a": 2, "b": 40}))]),
        tool_calls([call("t2", "add_numbers", json!({"a": "two", "c": 40}))]),
        text("It is 42."),
    ]);
    let agent = subject
        .build(
            "assistant",
            &dir,
            &provider,
            vec![Arc::new(AddNumbers)],
            Vec::new(),
        )
        .await?;
    let outcome = run_agent(&agent, OWNER_SESSION, "What is 2 + 40?").await?;
    ensure!(
        outcome.is_completed(),
        "the run did not complete: {:?}",
        outcome.status
    );
    ensure!(
        receipt_status(&outcome, "t1") == Some(ReceiptStatus::Succeeded),
        "the valid call did not succeed: {:?}",
        outcome.receipts()
    );
    let second = request(&provider, 1)?;
    let result = tool_result(&second, "t1").ok_or("the tool's result never reached the model")?;
    ensure!(
        result.content.contains("sum: 42"),
        "the model got {:?}, not the tool's result",
        result.content
    );
    ensure!(
        receipt_status(&outcome, "t2") == Some(ReceiptStatus::InvalidArguments),
        "invalid arguments were not refused: {:?}",
        outcome.receipts()
    );
    let third = request(&provider, 2)?;
    let refusal = tool_result(&third, "t2").ok_or("the refusal never reached the model")?;
    ensure!(
        !refusal.content.contains("sum:"),
        "the tool ran on invalid arguments: {:?}",
        refusal.content
    );
    Ok(())
}

/// A destructive tool that counts its runs.
fn delete_record(runs: Arc<AtomicUsize>) -> Arc<dyn Tool> {
    let spec = ToolSpec {
        name: "delete_record".into(),
        description: "Delete a record (synthetic).".into(),
        parameters: json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": false
        }),
    };
    Arc::new(FnTool::new(
        spec,
        ToolAnnotations::DESTRUCTIVE,
        move |args: Value| {
            let runs = runs.clone();
            async move {
                runs.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutput::text(format!(
                    "deleted {}",
                    args["id"].as_str().unwrap_or_default()
                )))
            }
        },
    ))
}

fn delete(id: &str) -> crate::ModelResponse {
    tool_calls([call(id, "delete_record", json!({"id": "REC-1"}))])
}

/// 4. A destructive call waits for the owner; approved, it runs once;
///    declined, it never runs and the run is not a success.
async fn approvals(subject: Subject, dir: PathBuf) -> Result<(), String> {
    for approve in [true, false] {
        let runs = Arc::new(AtomicUsize::new(0));
        let provider = scripted([delete("d1"), text("Done.")]);
        let agent = subject
            .build(
                "assistant",
                &dir,
                &provider,
                vec![delete_record(runs.clone())],
                Vec::new(),
            )
            .await?;
        let session = if approve {
            "parity:approve"
        } else {
            "parity:decline"
        };
        let outcome = run_agent(&agent, session, "Delete REC-1.").await?;
        let RunStatus::Interrupted { interruption } = &outcome.status else {
            return Err(format!(
                "a destructive call did not wait for the owner: {:?}",
                outcome.status
            ));
        };
        ensure!(
            interruption.call.name == "delete_record",
            "the owner was asked about {}",
            interruption.call.name
        );
        ensure!(
            runs.load(Ordering::SeqCst) == 0,
            "the tool ran before the owner answered"
        );
        let resolution = if approve {
            Resolution::Approve
        } else {
            Resolution::Deny {
                reason: "Not this one.".into(),
            }
        };
        let outcome = agent
            .resume(outcome.state, resolution)
            .await
            .map_err(|err| format!("resume failed: {err}"))?;
        let ran = runs.load(Ordering::SeqCst);
        if approve {
            ensure!(ran == 1, "approved, the tool ran {ran} times");
            ensure!(
                outcome.is_succeeded(),
                "approved, the run did not succeed: {:?}",
                outcome.status
            );
        } else {
            ensure!(ran == 0, "declined, the tool ran {ran} times");
            ensure!(
                !outcome.is_succeeded(),
                "declined, the run still reports success"
            );
            ensure!(
                receipt_status(&outcome, "d1") == Some(ReceiptStatus::Denied),
                "declined, the receipt is not Denied: {:?}",
                outcome.receipts()
            );
        }
    }
    Ok(())
}

/// Look up stock in the synthetic warehouse.
#[tool(read_only, crate = crate)]
fn check_stock(
    /// The item.
    item: String,
) -> String {
    format!("{item}: 42 in stock (synthetic)")
}

/// 5. A parent delegates to a child agent tool; the child's answer reaches
///    the parent, and its usage counts against the parent's budget.
async fn sub_agents(subject: Subject, dir: PathBuf) -> Result<(), String> {
    let child_reply = "There are 42 widgets in stock.";
    let child = |dir: PathBuf, provider: Arc<ScriptedProvider>| {
        let subject = subject.clone();
        async move {
            let agent = subject
                .build(
                    "researcher",
                    &dir,
                    &provider,
                    vec![Arc::new(CheckStock)],
                    Vec::new(),
                )
                .await?;
            Ok::<_, String>(
                AgentTool::new(agent)
                    .budget(ChildBudget {
                        max_turns: 4,
                        max_tokens: Some(10_000),
                        ..Default::default()
                    })
                    .description("Ask the researcher about stock.")
                    .read_only(),
            )
        }
    };
    let child_script = || {
        scripted([
            tool_calls([call("r1", "check_stock", json!({"item": "widgets"}))]),
            text(child_reply),
        ])
    };
    let delegate = || {
        tool_calls([call(
            "p1",
            "researcher",
            json!({"task": "Widgets in stock?"}),
        )])
    };
    let child_dir = dir.join("child");
    std::fs::create_dir_all(&child_dir).map_err(|err| err.to_string())?;

    // The answer comes back.
    let child_provider = child_script();
    let tool = child(child_dir.clone(), child_provider.clone()).await?;
    let provider = scripted([delegate(), text("42 widgets.")]);
    let parent = subject
        .build(
            "assistant",
            &dir,
            &provider,
            vec![Arc::new(tool)],
            Vec::new(),
        )
        .await?;
    let outcome = run_agent(&parent, OWNER_SESSION, "How many widgets do we have?").await?;
    ensure!(
        outcome.is_completed(),
        "the parent did not complete: {:?} {:?}",
        outcome.status,
        outcome.receipts()
    );
    ensure!(
        child_provider.requests().len() == 2,
        "the child made {} model calls, expected 2",
        child_provider.requests().len()
    );
    let result = tool_result(&request(&provider, 1)?, "p1")
        .map(|m| m.content.clone())
        .ok_or("the child's answer never reached the parent")?;
    ensure!(
        result.contains(child_reply),
        "the parent got {result:?}, not the child's answer"
    );
    let own = SCRIPTED_USAGE.billable_tokens() * provider.requests().len() as u64;
    let total = outcome.usage().billable_tokens();
    ensure!(
        total >= own + 2 * SCRIPTED_USAGE.billable_tokens(),
        "the child's usage is not on the parent's bill: {total} for {own} of the parent's own"
    );

    // A parent budget the delegation exceeds stops the parent.
    let budget = SCRIPTED_USAGE.billable_tokens() * 2;
    let child_provider = child_script();
    let tool = child(child_dir, child_provider).await?;
    let provider = scripted([delegate(), text("42 widgets.")]);
    let limits: Configure = Box::new(move |builder| {
        builder.limits(Limits {
            max_tokens: Some(budget),
            ..Limits::default()
        })
    });
    let parent = subject
        .build(
            "assistant",
            &dir,
            &provider,
            vec![Arc::new(tool)],
            vec![limits],
        )
        .await?;
    let outcome = run_agent(&parent, "parity:budget", "How many widgets do we have?").await?;
    ensure!(
        !outcome.is_completed() && provider.requests().len() == 1,
        "a parent with a {budget}-token budget went on after a delegation that spent more: {:?}",
        outcome.status
    );
    Ok(())
}

/// 6. A message into an in-process channel gets a reply out of it via
///    `ChannelRunner`, with the approval asked and answered on the channel.
async fn channel_round_trip(subject: Subject, dir: PathBuf) -> Result<(), String> {
    let runs = Arc::new(AtomicUsize::new(0));
    let router = EventRouter::new();
    let sink = router.clone();
    let provider = scripted([delete("c1"), text("REC-1 is deleted.")]);
    let agent = subject
        .build(
            "assistant",
            &dir,
            &provider,
            vec![delete_record(runs.clone())],
            vec![Box::new(move |builder| builder.event_sink(sink))],
        )
        .await?;
    let channel = Arc::new(TestChannel::new("chat").with_buttons(true));
    let runner = ChannelRunner::builder(agent)
        .channel(channel.clone())
        .access(AccessPolicy::new().owner("chat", OWNER))
        .event_router(router)
        .build()
        .map_err(|err| format!("channel runner: {err}"))?;
    let shutdown = CancellationToken::new();
    let task = tokio::spawn({
        let shutdown = shutdown.clone();
        async move { runner.run(shutdown).await }
    });
    let result = async {
        channel.inject_message(message(direct("dm-1"), OWNER, "Delete REC-1."));
        ensure!(
            channel
                .wait_for(CHANNEL_WAIT, |c| !c.prompts().is_empty())
                .await,
            "no approval prompt on the channel; sent: {:?}",
            channel.sent_texts("dm-1")
        );
        ensure!(
            runs.load(Ordering::SeqCst) == 0,
            "the tool ran before the owner answered"
        );
        let prompt = channel.prompts()[0].clone();
        channel.press(&prompt.id, APPROVE_CHOICE, OWNER, direct("dm-1"));
        let replied = |c: &TestChannel| {
            c.sent_texts("dm-1")
                .iter()
                .any(|t| t.contains("REC-1 is deleted."))
        };
        ensure!(
            channel.wait_for(CHANNEL_WAIT, replied).await,
            "no reply on the channel; sent: {:?}",
            channel.sent_texts("dm-1")
        );
        let ran = runs.load(Ordering::SeqCst);
        ensure!(
            ran == 1,
            "approved on the channel, the tool ran {ran} times"
        );
        Ok(())
    }
    .await;
    shutdown.cancel();
    let _ = task.await;
    result
}

/// The daemon memory's own tools.
const FULL_MEMORY_TOOLS: [&str; 3] = ["remember_fact", "manage_memories", "manage_people"];

/// 7. (FullMemory) The three memory tools are offered.
async fn full_memory_tools(subject: Subject, dir: PathBuf) -> Result<(), String> {
    let provider = scripted([text("ok")]);
    let agent = subject
        .build("assistant", &dir, &provider, Vec::new(), Vec::new())
        .await?;
    let names = tool_names(&agent);
    for tool in FULL_MEMORY_TOOLS {
        ensure!(
            names.iter().any(|n| n == tool),
            "`{tool}` is missing: {names:?}"
        );
    }
    Ok(())
}

/// 8. (FullMemory) `manage_memories` lists a stored fact; forgetting it,
///    once the owner approves, removes it from recall.
async fn full_memory_manage(subject: Subject, dir: PathBuf) -> Result<(), String> {
    drop(store_fact(&subject, &dir).await?);
    let provider = scripted([
        tool_calls([call("m1", "manage_memories", json!({"action": "list"}))]),
        tool_calls([call(
            "m2",
            "manage_memories",
            json!({"action": "forget", "category": "preference", "key": "favorite_color"}),
        )]),
        text("Forgotten."),
    ]);
    let agent = subject
        .build("assistant", &dir, &provider, Vec::new(), Vec::new())
        .await?;
    let mut outcome = run_agent(&agent, OWNER_SESSION, "List, then forget my color.").await?;
    let listed = tool_result(&request(&provider, 1)?, "m1")
        .map(|m| m.content.to_lowercase())
        .ok_or("the list never reached the model")?;
    ensure!(
        listed.contains("teal"),
        "the list does not show the fact: {listed:?}"
    );
    let mut asked = false;
    while let RunStatus::Interrupted { .. } = &outcome.status {
        asked = true;
        outcome = agent
            .resume(outcome.state, Resolution::Approve)
            .await
            .map_err(|err| format!("resume failed: {err}"))?;
    }
    ensure!(asked, "forgetting a memory did not ask the owner");
    ensure!(
        receipt_status(&outcome, "m2") == Some(ReceiptStatus::Succeeded),
        "the forget did not succeed: {:?}",
        outcome.receipts()
    );

    let provider = scripted([text("I don't know.")]);
    let agent = subject
        .build("assistant", &dir, &provider, Vec::new(), Vec::new())
        .await?;
    run_agent(&agent, "parity:after-forget", "What's my favorite color?").await?;
    ensure!(
        !mentions(&request(&provider, 0)?, "teal"),
        "a forgotten fact is still recalled"
    );
    Ok(())
}

/// A directory removed on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(label: &str) -> std::io::Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or_default();
        let path = std::env::temp_dir().join(format!(
            "aidaemon-behavior-{}-{}-{nanos}-{label}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
