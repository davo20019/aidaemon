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
//! | Safeguards | [`AuthorityHook`] (full authority), [`ScopeGuard`](crate::policies::ScopeGuard) over the workspace, [`EgressGuard`](crate::policies::EgressGuard), [`ApprovalPolicy`](crate::policies::ApprovalPolicy) (`DestructiveOnly`), [`Redactor`](crate::policies::Redactor), [`UntrustedOutput`](crate::policies::UntrustedOutput) | [`Preset::authority`], [`Preset::approvals`], [`Preset::undeclared_writes`], [`Preset::undeclared_reads`], [`Preset::path_free_tool`], [`Preset::without_default_hooks`] |
//! | Tools | none; with a workspace, [`default_tools`](crate::tools::default_tools) | [`Preset::tool`], [`Preset::workspace`] |
//! | Run store | in memory ([`MemoryStore`]) | [`Preset::store`], [`Preset::sqlite`] |
//! | Memory | with the `memory` feature (default), aidaemon's full memory system (`aidaemon_memory::full::FullMemory`: recall plus `remember_fact`, `manage_memories`, `manage_people`, `share_memory`), in memory or in [`Preset::memory_path`]; else none | [`Preset::memory`]: any [`MemorySystem`]; [`Preset::without_memory`] |
//! | Anything else | | [`Preset::hook`], [`Preset::configure`] (the [`AgentBuilder`]) |
//!
//! A destructive call pauses the run for the owner
//! ([`RunStatus::Interrupted`](crate::RunStatus)): answer it with
//! [`Agent::resume`]. So does, without a workspace, any call that may
//! change things and declares no paths, since no scope check covers it;
//! with a workspace, a call that declares no paths is refused. Mark tools
//! that touch no files with [`Preset::path_free_tool`].

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
    #[error("memory: {0}")]
    Memory(String),
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

/// The default memory's turn: the agent's owner, reading and writing one
/// memory.
#[cfg(feature = "memory")]
fn owner_turn(id: &str) -> MemoryTurn {
    MemoryTurn {
        session_id: id.to_string(),
        scope: "owner".to_string(),
        read: true,
        write: true,
        explicit_writes_only: false,
        learning: Default::default(),
        host: Default::default(),
    }
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

enum MemoryChoice {
    /// The full memory system, when built with the `memory` feature.
    #[cfg_attr(not(feature = "memory"), allow(dead_code))]
    Default(Option<PathBuf>),
    None,
    Given(Arc<dyn MemorySystem>, MemoryTurn),
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
    undeclared_reads: Option<UndeclaredAccess>,
    path_free: Vec<String>,
    memory: MemoryChoice,
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
            undeclared_reads: None,
            path_free: Vec::new(),
            memory: MemoryChoice::Default(None),
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
    /// declares no access ([`Tool::access`]). Unset, it is `Deny` with a
    /// [`Preset::workspace`] (every file tool there declares its paths).
    /// Without one, no directory is in scope to check against, so the
    /// scope guard lets such a call through and the approval policy asks
    /// the owner before it runs, whatever [`Preset::approvals`] says short
    /// of [`ApprovalMode::Never`]: a refund tool reaches the owner instead
    /// of being refused, and a tool that writes files without declaring
    /// them never runs unseen. Setting `Allow` here runs such calls without
    /// that extra ask; mark tools that touch no files with
    /// [`Preset::path_free_tool`] instead.
    pub fn undeclared_writes(mut self, policy: UndeclaredAccess) -> Self {
        self.undeclared_writes = Some(policy);
        self
    }

    /// What the scope guard does with a read-only call that declares no
    /// access. Unset, it is `Deny` with a [`Preset::workspace`] (a read
    /// outside it cannot be ruled out) and `Allow` without one.
    pub fn undeclared_reads(mut self, policy: UndeclaredAccess) -> Self {
        self.undeclared_reads = Some(policy);
        self
    }

    /// A tool of yours that touches no files (a ticket, a refund, an order
    /// lookup): its empty access manifest means "nothing" to the scope
    /// guard and the approval policy, so it is neither refused for not
    /// declaring paths nor asked about for that reason. The approval mode
    /// still applies. The tools a [`Preset::memory`] system gives are
    /// marked this way.
    pub fn path_free_tool(mut self, name: impl Into<String>) -> Self {
        self.path_free.push(name.into());
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
        self.memory = MemoryChoice::Given(system, turn);
        self
    }

    /// Keep the default memory in this SQLite file, so it survives a
    /// restart (default: in memory, for the life of the agent).
    #[cfg(feature = "memory")]
    pub fn memory_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.memory = MemoryChoice::Default(Some(path.into()));
        self
    }

    /// No long-term memory.
    pub fn without_memory(mut self) -> Self {
        self.memory = MemoryChoice::None;
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

        let mut builder = Agent::builder(self.id.clone()).provider(provider);
        if let Some(model) = model {
            builder = builder.model(model);
        }

        let mut instructions = self.instructions;
        let mut tools = self.tools;
        let mut extra_hooks = Vec::new();
        let mut path_free = self.path_free;
        let memory: Option<(Arc<dyn MemorySystem>, MemoryTurn)> = match self.memory {
            MemoryChoice::None => None,
            MemoryChoice::Given(system, turn) => Some((system, turn)),
            #[cfg(feature = "memory")]
            MemoryChoice::Default(path) => {
                let system = aidaemon_memory::full::FullMemory::open(path.as_deref())
                    .await
                    .map_err(|err| PresetError::Memory(err.to_string()))?;
                Some((Arc::new(system), owner_turn(&self.id)))
            }
            #[cfg(not(feature = "memory"))]
            MemoryChoice::Default(_) => None,
        };
        if let Some((system, turn)) = &memory {
            let parts = system.turn(turn).await;
            extra_hooks.extend(parts.hooks);
            // Memory tools keep the memory store, not workspace files.
            path_free.extend(parts.tools.iter().map(|tool| tool.spec().name));
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

        // With a workspace, a call that declares no access cannot be checked
        // against it, so it is refused. Without one, nothing is in scope:
        // reads run, and a call that may change things asks the owner.
        let confined = !roots.is_empty();
        let ask_undeclared = !confined && self.undeclared_writes.is_none();
        let undeclared_writes = self.undeclared_writes.unwrap_or(if confined {
            UndeclaredAccess::Deny
        } else {
            UndeclaredAccess::Allow
        });
        let undeclared_reads = self.undeclared_reads.unwrap_or(if confined {
            UndeclaredAccess::Deny
        } else {
            UndeclaredAccess::Allow
        });
        let mut scope = ScopeGuard::new(&roots)
            .undeclared_writer(undeclared_writes)
            .undeclared_read_only(undeclared_reads);
        let mut approval =
            ApprovalPolicy::new(self.approvals).ask_undeclared_writes(ask_undeclared);
        for name in &path_free {
            scope = scope.path_free_tool(name.clone());
            approval = approval.path_free_tool(name.clone());
        }
        builder = builder.instructions(instructions);
        for tool in tools {
            builder = builder.tool_arc(tool);
        }
        if self.default_hooks {
            builder = builder
                .hook(AuthorityHook::new(self.authority))
                .hook(scope)
                .hook(EgressGuard::new());
        }
        for hook in self.hooks.into_iter().chain(extra_hooks) {
            builder = builder.hook_arc(hook);
        }
        if self.default_hooks {
            builder = builder
                .hook(approval)
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
