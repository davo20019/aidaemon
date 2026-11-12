//! A short start: an agent with a provider, the required safeguards, a run
//! store and, if you give it a workspace, the default tools, in a few lines.
//!
//! ```no_run
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = aidaemon_sdk::preset("helper")
//!     .provider_from_env() // ANTHROPIC_API_KEY or OPENAI_API_KEY
//!     .instructions("You maintain this repository. Verify changes with tests.")
//!     .workspace(".") // the default file, search and shell tools, confined here
//!     .build()
//!     .await?;
//! let outcome = agent.run("cli:synthetic-user-1", "Fix the failing test").await?;
//! println!("{}", outcome.final_text.unwrap_or_default());
//! # Ok(())
//! # }
//! ```
//!
//! What the preset adds, and how to change each part:
//!
//! | Part | Default | Change with |
//! |---|---|---|
//! | Provider | none: name one | [`Preset::provider`], [`Preset::provider_from_env`], [`Preset::provider_from_env_or`] |
//! | Model | `AIDAEMON_MODEL`, else the provider's default | [`Preset::model`] |
//! | Safeguards | [`AuthorityHook`] (full authority), [`ScopeGuard`](crate::policies::ScopeGuard) over the workspace, [`EgressGuard`](crate::policies::EgressGuard), [`ApprovalPolicy`](crate::policies::ApprovalPolicy) (`DestructiveOnly`), [`Redactor`](crate::policies::Redactor), [`UntrustedOutput`](crate::policies::UntrustedOutput) | [`Preset::authority`], [`Preset::approvals`], [`Preset::undeclared_writes`], [`Preset::without_default_hooks`] |
//! | Tools | none; with a workspace, [`default_tools`](crate::tools::default_tools) | [`Preset::tool`], [`Preset::workspace`] |
//! | Run store | in memory ([`MemoryStore`]) | [`Preset::store`], [`Preset::sqlite`] |
//! | Memory | none | [`Preset::memory`]: any [`MemorySystem`] |
//! | Anything else | | [`Preset::hook`], [`Preset::configure`] (the [`AgentBuilder`]) |
//!
//! A destructive call pauses the run for the owner
//! ([`RunStatus::Interrupted`](crate::RunStatus)): answer it with
//! [`Agent::resume`].

use std::path::PathBuf;
use std::sync::Arc;

use aidaemon_core::{
    Agent, AgentBuilder, Authority, AuthorityHook, BuildError, Hook, MemoryStore, Provider,
    RunStore, Tool,
};
use aidaemon_policies::{
    ApprovalMode, ApprovalPolicy, EgressGuard, MemorySystem, MemoryTurn, Redactor, ScopeGuard,
    UndeclaredAccess, UntrustedOutput,
};
use aidaemon_providers::{Anthropic, OpenAiCompatible};

/// The model used with `ANTHROPIC_API_KEY` when `AIDAEMON_MODEL` is unset.
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-sonnet-5-5";
/// The model used with `OPENAI_API_KEY` when `AIDAEMON_MODEL` is unset.
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-5";

/// Start a [`Preset`] for an agent with this id.
pub fn preset(id: impl Into<String>) -> Preset {
    Preset::new(id)
}

/// Why a preset could not build.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PresetError {
    #[error("no provider: call .provider(..), or set ANTHROPIC_API_KEY or OPENAI_API_KEY")]
    NoProvider,
    #[error("provider configuration: {0}")]
    Provider(String),
    #[error("workspace: {0}")]
    Workspace(String),
    #[error("run store: {0}")]
    Store(String),
    #[error(transparent)]
    Build(#[from] BuildError),
}

/// A provider read from the environment and its default model.
pub type EnvProvider = (Arc<dyn Provider>, &'static str);

/// A provider read from the environment, and its default model:
/// `ANTHROPIC_API_KEY`, else `OPENAI_API_KEY` (with `OPENAI_BASE_URL` for
/// any OpenAI-compatible server). `None` when neither is set.
pub fn provider_from_env() -> Result<Option<EnvProvider>, PresetError> {
    let config = |err: aidaemon_providers::ConfigError| PresetError::Provider(err.to_string());
    if let Some(key) = env("ANTHROPIC_API_KEY") {
        let provider = Anthropic::builder().api_key(key).build().map_err(config)?;
        return Ok(Some((Arc::new(provider), DEFAULT_ANTHROPIC_MODEL)));
    }
    if let Some(key) = env("OPENAI_API_KEY") {
        let mut builder = OpenAiCompatible::builder().api_key(key);
        if let Some(url) = env("OPENAI_BASE_URL") {
            builder = builder.base_url(url);
        }
        let provider = builder.build().map_err(config)?;
        return Ok(Some((Arc::new(provider), DEFAULT_OPENAI_MODEL)));
    }
    Ok(None)
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

enum ProviderChoice {
    None,
    Explicit(Arc<dyn Provider>),
    Env,
    EnvOr(Arc<dyn Provider>),
}

enum StoreChoice {
    Memory,
    Given(Arc<dyn RunStore>),
    #[cfg(feature = "sqlite")]
    Sqlite(PathBuf),
}

/// A builder that starts from working defaults (see the [module](self)).
/// Every part can be replaced.
pub struct Preset {
    id: String,
    provider: ProviderChoice,
    model: Option<String>,
    instructions: String,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Vec<Arc<dyn Hook>>,
    #[cfg_attr(not(feature = "tools"), allow(dead_code))]
    workspace: Option<PathBuf>,
    store: StoreChoice,
    authority: Authority,
    approvals: ApprovalMode,
    default_hooks: bool,
    undeclared_writes: Option<UndeclaredAccess>,
    memory: Option<(Arc<dyn MemorySystem>, MemoryTurn)>,
    configure: Vec<Box<dyn FnOnce(AgentBuilder) -> AgentBuilder + Send>>,
}

impl Preset {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            provider: ProviderChoice::None,
            model: None,
            instructions: String::new(),
            tools: Vec::new(),
            hooks: Vec::new(),
            workspace: None,
            store: StoreChoice::Memory,
            authority: Authority::full(),
            approvals: ApprovalMode::DestructiveOnly,
            default_hooks: true,
            undeclared_writes: None,
            memory: None,
            configure: Vec::new(),
        }
    }

    /// Use this provider. Set the model with [`Preset::model`].
    pub fn provider(mut self, provider: impl Provider + 'static) -> Self {
        self.provider = ProviderChoice::Explicit(Arc::new(provider));
        self
    }

    /// [`Preset::provider`] for a shared provider.
    pub fn provider_arc(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = ProviderChoice::Explicit(provider);
        self
    }

    /// Read the provider from the environment at build time (see
    /// [`provider_from_env`]); building fails if no key is set.
    pub fn provider_from_env(mut self) -> Self {
        self.provider = ProviderChoice::Env;
        self
    }

    /// Read the provider from the environment, and use `fallback` (a
    /// scripted provider, a local server) when no key is set. With the
    /// fallback and no [`Preset::model`], the model is `"scripted"`.
    pub fn provider_from_env_or(mut self, fallback: impl Provider + 'static) -> Self {
        self.provider = ProviderChoice::EnvOr(Arc::new(fallback));
        self
    }

    /// The model id, overriding `AIDAEMON_MODEL` and the provider's default.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// The system prompt.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
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

    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// A hook of your own. It runs after the authority, scope and egress
    /// checks and before the approval policy, so a hook that refuses a call
    /// does so before the owner is asked.
    pub fn hook(mut self, hook: impl Hook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    pub fn hook_arc(mut self, hook: Arc<dyn Hook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Work in `dir`: add the default tools (files, search, sandboxed
    /// shell, web fetch, checklist) confined to it, and scope every path to
    /// it.
    #[cfg(feature = "tools")]
    pub fn workspace(mut self, dir: impl Into<PathBuf>) -> Self {
        self.workspace = Some(dir.into());
        self
    }

    /// Keep runs in this store (default: in memory).
    pub fn store(mut self, store: Arc<dyn RunStore>) -> Self {
        self.store = StoreChoice::Given(store);
        self
    }

    /// Keep runs in a SQLite file, so they survive a crash or restart.
    #[cfg(feature = "sqlite")]
    pub fn sqlite(mut self, path: impl Into<PathBuf>) -> Self {
        self.store = StoreChoice::Sqlite(path.into());
        self
    }

    /// What the agent may do (default: [`Authority::full`]).
    pub fn authority(mut self, authority: Authority) -> Self {
        self.authority = authority;
        self
    }

    /// When the owner is asked (default: [`ApprovalMode::DestructiveOnly`]).
    pub fn approvals(mut self, mode: ApprovalMode) -> Self {
        self.approvals = mode;
        self
    }

    /// What the scope guard does with a call that may change things but
    /// declares no paths ([`Tool::access`]). The default is `Deny` with a
    /// [`Preset::workspace`] (every file tool there declares its paths) and
    /// `Allow` without one, where no directory is in scope to check
    /// against, so tools of your own that do not touch files (a refund, a
    /// ticket) run; the approval policy still asks before destructive ones.
    pub fn undeclared_writes(mut self, policy: UndeclaredAccess) -> Self {
        self.undeclared_writes = Some(policy);
        self
    }

    /// Leave out the default safeguard hooks. Your own hooks (or
    /// [`AgentBuilder::attest`] through [`Preset::configure`]) must then
    /// cover every required safeguard, or building fails.
    pub fn without_default_hooks(mut self) -> Self {
        self.default_hooks = false;
        self
    }

    /// Long-term memory: the hooks and tools `system` gives for `turn`,
    /// and its guidance appended to the instructions. Any [`MemorySystem`]
    /// works (`aidaemon-memory`'s or your own).
    pub fn memory(mut self, system: Arc<dyn MemorySystem>, turn: MemoryTurn) -> Self {
        self.memory = Some((system, turn));
        self
    }

    /// Change the underlying [`AgentBuilder`] after the preset has filled
    /// it (limits, event sinks, extensions, stop conditions...).
    pub fn configure(
        mut self,
        configure: impl FnOnce(AgentBuilder) -> AgentBuilder + Send + 'static,
    ) -> Self {
        self.configure.push(Box::new(configure));
        self
    }

    pub async fn build(self) -> Result<Agent, PresetError> {
        let (provider, default_model) = match self.provider {
            ProviderChoice::None => return Err(PresetError::NoProvider),
            ProviderChoice::Explicit(provider) => (provider, None),
            ProviderChoice::Env => {
                let (provider, model) = provider_from_env()?.ok_or(PresetError::NoProvider)?;
                (provider, Some(model))
            }
            ProviderChoice::EnvOr(fallback) => match provider_from_env()? {
                Some((provider, model)) => (provider, Some(model)),
                None => (fallback, Some("scripted")),
            },
        };
        let model = self
            .model
            .or_else(|| env("AIDAEMON_MODEL").filter(|_| default_model != Some("scripted")))
            .or_else(|| default_model.map(str::to_string));

        let mut builder = Agent::builder(self.id).provider(provider);
        if let Some(model) = model {
            builder = builder.model(model);
        }

        let mut instructions = self.instructions;
        let mut tools = self.tools;
        let mut extra_hooks = Vec::new();
        if let Some((system, turn)) = &self.memory {
            let parts = system.turn(turn).await;
            extra_hooks.extend(parts.hooks);
            tools.extend(parts.tools);
            if let Some(guidance) = system.guidance() {
                if !instructions.is_empty() {
                    instructions.push_str("\n\n");
                }
                instructions.push_str(&guidance);
            }
        }

        #[cfg_attr(not(feature = "tools"), allow(unused_mut))]
        let mut roots: Vec<PathBuf> = Vec::new();
        #[cfg(feature = "tools")]
        if let Some(dir) = &self.workspace {
            let workspace = crate::tools::Workspace::new(dir)
                .map_err(|err| PresetError::Workspace(format!("{}: {err}", dir.display())))?;
            roots.push(workspace.root().to_path_buf());
            tools.extend(crate::tools::default_tools(workspace));
        }

        let undeclared = self.undeclared_writes.unwrap_or(if roots.is_empty() {
            UndeclaredAccess::Allow
        } else {
            UndeclaredAccess::Deny
        });
        builder = builder.instructions(instructions);
        for tool in tools {
            builder = builder.tool_arc(tool);
        }
        if self.default_hooks {
            builder = builder
                .hook(AuthorityHook::new(self.authority))
                .hook(ScopeGuard::new(&roots).undeclared_writer(undeclared))
                .hook(EgressGuard::new());
        }
        for hook in self.hooks.into_iter().chain(extra_hooks) {
            builder = builder.hook_arc(hook);
        }
        if self.default_hooks {
            builder = builder
                .hook(ApprovalPolicy::new(self.approvals))
                .hook(Redactor::default())
                .hook(UntrustedOutput::default());
        }

        let store: Arc<dyn RunStore> = match self.store {
            StoreChoice::Memory => Arc::new(MemoryStore::new()),
            StoreChoice::Given(store) => store,
            #[cfg(feature = "sqlite")]
            StoreChoice::Sqlite(path) => Arc::new(
                crate::store::SqliteStore::open(&path)
                    .await
                    .map_err(|err| PresetError::Store(format!("{}: {err}", path.display())))?,
            ),
        };
        builder = builder.store(store);
        for configure in self.configure {
            builder = configure(builder);
        }
        Ok(builder.build()?)
    }
}
