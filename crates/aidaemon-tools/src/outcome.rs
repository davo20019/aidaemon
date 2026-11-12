//! Tools that report a typed [`ToolCallOutcome`].
//!
//! An [`OutcomeTool`] is written once. An application with its own dispatcher
//! calls [`OutcomeTool::execute`] and keeps the typed metadata; a plain
//! [`aidaemon_core`] agent gets the same tool through [`aidaemon_core::Tool`].

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use aidaemon_core::{
    AccessManifest, CallEffects, EffectClass, Outbound, SourceTrust, ToolAnnotations, ToolContext,
    ToolOutput, ToolResult, ToolSpec,
};

use crate::semantics::{
    ApprovalCall, ApprovalKind, ApprovalResponse, FileGuard, PermissionMode, ToolCallContext,
    ToolCallOutcome, ToolSemantics, TrustRoots,
};

/// A tool could not reach its target at all, so nothing took effect and the
/// call may run again.
#[derive(Debug)]
pub struct NothingSentError(pub String);

impl std::fmt::Display for NothingSentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NothingSentError {}

/// A tool whose execution returns a typed outcome.
#[async_trait]
pub trait OutcomeTool: ToolSemantics {
    fn name(&self) -> &str;
    /// One-line description.
    fn description(&self) -> &str;
    /// The full function object: `name`, `description` and `parameters`.
    fn schema(&self) -> Value;
    /// Run one call. `Err` is a failed call whose message the model reads.
    async fn execute(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome>;
    /// Checks made before a kernel agent runs the call.
    fn validate_call(&self, _arguments: &Value) -> Result<(), String> {
        Ok(())
    }
    /// The permission mode the owner's cards for this tool are sent with
    /// (it picks the buttons offered). `None`: the default, Cautious.
    fn approval_card_mode(&self) -> Option<PermissionMode> {
        None
    }
    /// The kind of card the owner is asked `request` (from
    /// [`ToolSemantics::owner_approval_request`]) with, which picks the
    /// buttons offered: a one-time card or a goal confirmation offers no
    /// session or persistent grant. The default is a standard command card.
    fn approval_card_kind(&self, _request: &aidaemon_core::OwnerApprovalRequest) -> ApprovalKind {
        ApprovalKind::Command
    }
    /// Apply the owner's answer to a card asked for `call` (from
    /// [`ToolSemantics::owner_approval_request`]) to the tool's own grant
    /// stores. `None`: the card timed out or never reached the owner. The
    /// default keeps nothing; the kernel's session grants are separate.
    async fn record_owner_answer(
        &self,
        _call: &ApprovalCall<'_>,
        _facts: &ToolCallContext,
        _response: Option<ApprovalResponse>,
    ) {
    }
}

pub(crate) fn spec_of<T: OutcomeTool + ?Sized>(tool: &T) -> ToolSpec {
    let schema = tool.schema();
    ToolSpec {
        name: tool.name().to_string(),
        description: schema["description"]
            .as_str()
            .unwrap_or_else(|| tool.description())
            .to_string(),
        parameters: schema["parameters"].clone(),
    }
}

/// The arguments must be an object holding every key the schema requires,
/// and pass the tool's own checks.
pub(crate) fn validate_for_core<T: OutcomeTool + ?Sized>(
    tool: &T,
    arguments: &Value,
) -> Result<(), String> {
    let Some(object) = arguments.as_object() else {
        return Err("arguments must be a JSON object".into());
    };
    let schema = tool.schema();
    let required = schema["parameters"]["required"].as_array();
    for key in required.into_iter().flatten().filter_map(Value::as_str) {
        if !object.contains_key(key) {
            return Err(format!("`{key}` is required"));
        }
    }
    tool.validate_call(arguments)
}

/// Run an [`OutcomeTool`] for a kernel agent. A failed or rejected call is an
/// error result the model can react to.
pub(crate) async fn run_as_core<T: OutcomeTool + ?Sized>(tool: &T, arguments: Value) -> ToolResult {
    let raw = arguments.to_string();
    let outcome = match tool.execute(&raw, &ToolCallContext::default()).await {
        Ok(outcome) => outcome,
        Err(error) => return Ok(ToolOutput::error(error.to_string())),
    };
    let mut output = if outcome.metadata.contract_rejected {
        ToolOutput::error(outcome.output)
    } else {
        ToolOutput::text(outcome.output)
    };
    if let Some(read) = &outcome.metadata.read_file {
        output = output.with_details(json!({
            "path": read.display_path,
            "start_line": read.returned_start_line,
            "end_line": read.returned_end_line,
            "total_lines": read.total_lines,
            "truncated": read.truncated,
        }));
    }
    output.finish = outcome.finish;
    Ok(output)
}

/// Runs one call of an adapted tool for a kernel agent, in place of the
/// tool's own [`OutcomeTool::execute`]. An application with its own
/// dispatcher (receipts, status lines, result fitting) supplies this.
#[async_trait]
pub trait OutcomeRunner: Send + Sync {
    async fn run(&self, tool: &str, arguments: Value, ctx: &ToolContext) -> ToolResult;
}

/// Extra argument check for an adapted tool, replacing the schema check.
pub type CallValidator = Arc<dyn Fn(&Value) -> Result<(), String> + Send + Sync>;

/// Any [`OutcomeTool`] as a kernel tool: every description of a call
/// (annotations, effects, access, outbound data) comes from the tool's
/// [`ToolSemantics`].
pub struct KernelAdapter<T: ?Sized> {
    tool: Arc<T>,
    spec: ToolSpec,
    trust_roots: Arc<Vec<String>>,
    validator: Option<CallValidator>,
    timeout: Option<Duration>,
    runner: Option<Arc<dyn OutcomeRunner>>,
}

impl<T: OutcomeTool + ?Sized> KernelAdapter<T> {
    pub fn new(tool: Arc<T>) -> Self {
        let spec = spec_of(tool.as_ref());
        Self {
            tool,
            spec,
            trust_roots: Arc::new(Vec::new()),
            validator: None,
            timeout: None,
            runner: None,
        }
    }

    /// Offer the tool under this spec instead of its own schema.
    pub fn with_spec(mut self, spec: ToolSpec) -> Self {
        self.spec = spec;
        self
    }

    /// The run's trusted roots: where an effect lands can decide its class.
    pub fn with_trust_roots(mut self, roots: Arc<Vec<String>>) -> Self {
        self.trust_roots = roots;
        self
    }

    /// Check arguments with this instead of the schema's `required` keys.
    pub fn with_validator(mut self, validator: CallValidator) -> Self {
        self.validator = Some(validator);
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Run calls through `runner` instead of [`OutcomeTool::execute`].
    pub fn with_runner(mut self, runner: Arc<dyn OutcomeRunner>) -> Self {
        self.runner = Some(runner);
        self
    }

    fn canonical(&self, arguments: &Value) -> String {
        let raw = arguments.to_string();
        self.tool.canonicalize_arguments(&raw).unwrap_or(raw)
    }
}

/// How many sub-agents one batch may run at once.
pub const MAX_CONCURRENT_CHILDREN: usize = 3;

#[async_trait]
impl<T: OutcomeTool + ?Sized + 'static> aidaemon_core::Tool for KernelAdapter<T> {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    /// Worst case for the tool as a whole: only a tool that is read-only
    /// for every call says so; any other may be destructive and egress.
    fn annotations(&self) -> ToolAnnotations {
        if self.tool.capabilities().read_only {
            ToolAnnotations::READ_ONLY
        } else {
            ToolAnnotations {
                read_only: false,
                destructive: true,
                idempotent: false,
                egress: true,
            }
        }
    }

    /// Sub-agents spawned in one batch run together, up to
    /// [`MAX_CONCURRENT_CHILDREN`]: the coordination ledger arbitrates what
    /// they write. Every other non-read-only tool runs alone.
    fn concurrent_peers(&self) -> usize {
        if self.tool.dispatch_facts().spawns_child_agents {
            MAX_CONCURRENT_CHILDREN
        } else {
            0
        }
    }

    /// Judged per call: a mixed tool such as a shell reads for `ls` and
    /// writes for `rm`.
    fn annotations_for(&self, arguments: &Value) -> ToolAnnotations {
        self.tool
            .call_annotations_in(&arguments.to_string(), &TrustRoots::new(&self.trust_roots))
    }

    fn outbound(&self, arguments: &Value) -> Outbound {
        self.tool.call_outbound(&arguments.to_string())
    }

    /// The tool's own declaration for the arguments it will run with
    /// (after its argument canonicalization).
    fn effects(&self, arguments: &Value) -> CallEffects {
        self.tool.call_effects(&self.canonical(arguments))
    }

    /// The declared effects as the kernel's access manifest: egress targets
    /// as hosts, path effects as writes, a shell tool's command line given
    /// as `command` or `script`, and a file tool's guarded paths.
    fn access(&self, arguments: &Value) -> AccessManifest {
        let mut manifest = AccessManifest::default();
        let facts = self.tool.dispatch_facts();
        if facts.shell_command_line {
            for field in ["command", "script"] {
                if let Some(text) = arguments
                    .get(field)
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                {
                    manifest.commands.push(text.to_string());
                }
            }
        }
        for key in self.effects(arguments).keys {
            let list = match key.class {
                EffectClass::Egress => &mut manifest.hosts,
                EffectClass::Write | EffectClass::Delete | EffectClass::Configuration => {
                    &mut manifest.writes
                }
                _ => continue,
            };
            if !list.contains(&key.target) {
                list.push(key.target);
            }
        }
        // Effects do not name reads, and name no writes on a remote backend.
        if facts.file_guard != FileGuard::None {
            let (reads, writes) = self.tool.guarded_paths(&self.canonical(arguments));
            manifest.reads.extend(reads);
            for write in writes {
                if !manifest.writes.contains(&write) {
                    manifest.writes.push(write);
                }
            }
        }
        manifest
    }

    /// Fallback only: a result that knows its trust says so itself.
    fn source_trust(&self, _arguments: &Value) -> SourceTrust {
        SourceTrust::Untrusted
    }

    fn validate(&self, arguments: &Value) -> Result<(), String> {
        match &self.validator {
            Some(validator) => validator(arguments),
            None => validate_for_core(self.tool.as_ref(), arguments),
        }
    }

    fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    async fn call(&self, arguments: Value, ctx: &ToolContext) -> ToolResult {
        match &self.runner {
            Some(runner) => runner.run(&self.spec.name, arguments, ctx).await,
            None => run_as_core(self.tool.as_ref(), arguments).await,
        }
    }
}

/// Implement [`aidaemon_core::Tool`] for an [`OutcomeTool`].
macro_rules! core_tool {
    ($ty:ty, annotations: $annotations:expr, access: |$this:ident, $args:ident| $access:expr) => {
        #[async_trait::async_trait]
        impl aidaemon_core::Tool for $ty {
            fn spec(&self) -> aidaemon_core::ToolSpec {
                $crate::outcome::spec_of(self)
            }

            fn annotations(&self) -> aidaemon_core::ToolAnnotations {
                $annotations
            }

            fn validate(&self, arguments: &serde_json::Value) -> Result<(), String> {
                $crate::outcome::validate_for_core(self, arguments)
            }

            fn access(&self, $args: &serde_json::Value) -> aidaemon_core::AccessManifest {
                let $this = self;
                $access
            }

            async fn call(
                &self,
                arguments: serde_json::Value,
                _ctx: &aidaemon_core::ToolContext,
            ) -> aidaemon_core::ToolResult {
                $crate::outcome::run_as_core(self, arguments).await
            }
        }
    };
}
pub(crate) use core_tool;
