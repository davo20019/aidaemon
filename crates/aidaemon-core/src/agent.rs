use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::error::{BuildError, CoreError};
use crate::event::EventSink;
use crate::hook::Hook;
use crate::limits::{Limits, StopCondition};
use crate::outcome::RunOutcome;
use crate::provider::{Provider, ToolSpec};
use crate::run_loop::Driver;
use crate::state::RunState;
use crate::store::RunStore;
use crate::stuck::StuckConfig;
use crate::tool::Tool;

/// How to settle the call an interrupted run is waiting on.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// Run the call. `before_tool` hooks are not consulted again for it.
    Approve,
    /// Refuse it. The reason goes back to the model as the tool result.
    Deny { reason: String },
}

#[derive(Clone)]
pub(crate) struct AgentInner {
    pub id: String,
    pub model: String,
    pub instructions: String,
    pub provider: Arc<dyn Provider>,
    pub tools: HashMap<String, Arc<dyn Tool>>,
    /// Specs in registration order, computed once so the tool block of the
    /// prompt is byte-stable across turns (cache friendly).
    pub specs: Vec<ToolSpec>,
    pub hooks: Vec<Arc<dyn Hook>>,
    pub limits: Limits,
    pub stuck: StuckConfig,
    pub stop_conditions: Vec<Arc<dyn StopCondition>>,
    pub sink: Option<Arc<dyn EventSink>>,
    pub store: Option<Arc<dyn RunStore>>,
    pub max_output_tokens: Option<u32>,
}

/// An agent is configuration: a stable id, a model, instructions, tools and
/// hooks. It holds no per-run state, is cheap to clone, and can run many
/// sessions concurrently.
#[derive(Clone)]
pub struct Agent {
    inner: Arc<AgentInner>,
}

impl Agent {
    pub fn builder(id: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(id)
    }

    pub fn id(&self) -> &str {
        &self.inner.id
    }

    pub fn tool_specs(&self) -> &[ToolSpec] {
        &self.inner.specs
    }

    pub fn limits(&self) -> &Limits {
        &self.inner.limits
    }

    /// A copy of this agent with different limits (same id, tools, hooks).
    /// Used to give a child agent its own budget.
    pub fn with_limits(&self, limits: Limits) -> Agent {
        let mut inner = (*self.inner).clone();
        inner.limits = limits;
        Agent {
            inner: Arc::new(inner),
        }
    }

    /// A copy of this agent with one more hook, run after the existing ones.
    pub fn with_hook(&self, hook: impl Hook + 'static) -> Agent {
        let mut inner = (*self.inner).clone();
        inner.hooks.push(Arc::new(hook));
        Agent {
            inner: Arc::new(inner),
        }
    }

    /// Start a new run in `session_id` with one user message.
    pub async fn run(&self, session_id: &str, input: &str) -> Result<RunOutcome, CoreError> {
        self.run_with_cancel(session_id, input, CancellationToken::new())
            .await
    }

    pub async fn run_with_cancel(
        &self,
        session_id: &str,
        input: &str,
        cancel: CancellationToken,
    ) -> Result<RunOutcome, CoreError> {
        let state = RunState::new(&self.inner.id, session_id, input);
        Driver::new(&self.inner, state, cancel).drive(None).await
    }

    /// Continue from saved state that is not interrupted, typically
    /// `previous.state.with_user_message("next question")`.
    pub async fn run_from(
        &self,
        state: RunState,
        cancel: CancellationToken,
    ) -> Result<RunOutcome, CoreError> {
        self.check_owner(&state)?;
        if state.pending.is_some() {
            return Err(CoreError::PendingInterruption);
        }
        Driver::new(&self.inner, state, cancel).drive(None).await
    }

    /// Settle the interruption in `state` and keep going. `state` may come
    /// from another process: it is plain data.
    pub async fn resume(
        &self,
        state: RunState,
        resolution: Resolution,
    ) -> Result<RunOutcome, CoreError> {
        self.resume_with_cancel(state, resolution, CancellationToken::new())
            .await
    }

    pub async fn resume_with_cancel(
        &self,
        state: RunState,
        resolution: Resolution,
        cancel: CancellationToken,
    ) -> Result<RunOutcome, CoreError> {
        self.check_owner(&state)?;
        if state.pending.is_none() {
            return Err(CoreError::NotInterrupted);
        }
        Driver::new(&self.inner, state, cancel)
            .drive(Some(resolution))
            .await
    }

    fn check_owner(&self, state: &RunState) -> Result<(), CoreError> {
        if state.agent_id != self.inner.id {
            return Err(CoreError::AgentMismatch {
                expected: self.inner.id.clone(),
                found: state.agent_id.clone(),
            });
        }
        Ok(())
    }
}

/// Builder for [`Agent`]. Only `provider` and `model` are required.
pub struct AgentBuilder {
    id: String,
    model: Option<String>,
    instructions: String,
    provider: Option<Arc<dyn Provider>>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Vec<Arc<dyn Hook>>,
    limits: Limits,
    stuck: StuckConfig,
    stop_conditions: Vec<Arc<dyn StopCondition>>,
    sink: Option<Arc<dyn EventSink>>,
    store: Option<Arc<dyn RunStore>>,
    max_output_tokens: Option<u32>,
}

impl AgentBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: None,
            instructions: String::new(),
            provider: None,
            tools: Vec::new(),
            hooks: Vec::new(),
            limits: Limits::default(),
            stuck: StuckConfig::default(),
            stop_conditions: Vec::new(),
            sink: None,
            store: None,
            max_output_tokens: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The system prompt. Sent first on every request and never compacted.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    pub fn provider(mut self, provider: impl Provider + 'static) -> Self {
        self.provider = Some(Arc::new(provider));
        self
    }

    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn hook(mut self, hook: impl Hook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    pub fn hook_arc(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn stuck_detection(mut self, config: StuckConfig) -> Self {
        self.stuck = config;
        self
    }

    pub fn stop_when(mut self, condition: impl StopCondition + 'static) -> Self {
        self.stop_conditions.push(Arc::new(condition));
        self
    }

    pub fn event_sink(mut self, sink: impl EventSink + 'static) -> Self {
        self.sink = Some(Arc::new(sink));
        self
    }

    pub fn store(mut self, store: Arc<dyn RunStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    pub fn build(self) -> Result<Agent, BuildError> {
        if self.id.trim().is_empty() {
            return Err(BuildError::EmptyAgentId);
        }
        let provider = self.provider.ok_or(BuildError::MissingProvider)?;
        let model = self.model.ok_or(BuildError::MissingModel)?;

        let mut tools = HashMap::new();
        let mut specs = Vec::with_capacity(self.tools.len());
        for tool in self.tools {
            let spec = tool.spec();
            if !valid_tool_name(&spec.name) {
                return Err(BuildError::InvalidToolName(spec.name));
            }
            if tools.insert(spec.name.clone(), tool).is_some() {
                return Err(BuildError::DuplicateTool(spec.name));
            }
            specs.push(spec);
        }

        Ok(Agent {
            inner: Arc::new(AgentInner {
                id: self.id,
                model,
                instructions: self.instructions,
                provider,
                tools,
                specs,
                hooks: self.hooks,
                limits: self.limits,
                stuck: self.stuck,
                stop_conditions: self.stop_conditions,
                sink: self.sink,
                store: self.store,
                max_output_tokens: self.max_output_tokens,
            }),
        })
    }
}

fn valid_tool_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
