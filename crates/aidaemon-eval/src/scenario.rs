//! The YAML scenario format and loading.
//!
//! Every struct rejects unknown fields: a typo such as `tools_usd` would
//! otherwise deserialize to "no assertion" and the scenario would pass
//! vacuously.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use aidaemon_core::testing::SCRIPTED_USAGE;
use aidaemon_core::{
    AgentBuilder, Limits, ModelResponse, ReceiptStatus, RunStatus, ToolCall, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Errors from loading or validating scenarios. Every variant names the
/// file it came from when there is one.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("{path}: cannot read: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: invalid scenario YAML: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("{path}: invalid scenario: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("{path}: no *.yaml or *.yml scenario files found")]
    Empty { path: PathBuf },
    #[error("duplicate scenario name `{name}` in {first} and {second}")]
    DuplicateName {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("invalid report JSON: {0}")]
    ReportJson(String),
}

/// One eval case: an input, optionally a scripted model, and expectations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The user message that starts the run.
    pub input: String,
    /// System prompt, applied by [`Scenario::configure`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Free-form labels an [`crate::AgentFactory`] may use to pick tools,
    /// hooks or policies for this scenario.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Scripted model replies, in order. When present the run is
    /// deterministic and must consume the script exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<Vec<ScriptedReply>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<ScenarioLimits>,
    #[serde(default)]
    pub expect: Expect,
    /// Where the scenario was loaded from, for error messages.
    #[serde(skip)]
    pub source: Option<PathBuf>,
}

/// One scripted model reply: final text, tool calls, or both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptedReply {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ScriptedCall>>,
    /// Token usage reported for this reply. Defaults to
    /// [`aidaemon_core::testing::SCRIPTED_USAGE`] (100 in, 20 out).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<ScriptedUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptedCall {
    /// Defaults to `call_<reply>_<index>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: ScriptedArguments,
}

/// A scripted call's arguments: a JSON document in a string, as a provider
/// sends them, or a YAML object.
///
/// A string that is not valid JSON is kept as it is, so a script can show the
/// loop what a model that emits broken arguments looks like.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScriptedArguments {
    Text(String),
    Object(Value),
}

impl Default for ScriptedArguments {
    fn default() -> Self {
        Self::Object(empty_object())
    }
}

impl From<Value> for ScriptedArguments {
    fn from(value: Value) -> Self {
        Self::Object(value)
    }
}

impl ScriptedArguments {
    /// The arguments as JSON: a parsed string, or the object itself.
    pub fn to_value(&self) -> Value {
        match self {
            Self::Text(raw) => {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.clone()))
            }
            Self::Object(value) => value.clone(),
        }
    }

    /// The arguments as the JSON string a provider would have sent.
    pub fn to_json_string(&self) -> String {
        match self {
            Self::Text(raw) => raw.clone(),
            Self::Object(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptedUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cached_input_tokens: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioLimits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    /// Billable-token budget for the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

/// How a run is expected to end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedStatus {
    Completed,
    Interrupted,
    LimitReached,
    Stopped,
    Stuck,
    Cancelled,
    ProviderFailed,
    /// A tool ended the run with its own finish.
    Finished,
}

impl ExpectedStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Interrupted => "interrupted",
            Self::LimitReached => "limit_reached",
            Self::Stopped => "stopped",
            Self::Stuck => "stuck",
            Self::Cancelled => "cancelled",
            Self::ProviderFailed => "provider_failed",
            Self::Finished => "finished",
        }
    }
}

/// The snake_case name of a run status, matching [`ExpectedStatus`].
pub fn status_name(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::Completed => "completed",
        RunStatus::Interrupted { .. } => "interrupted",
        RunStatus::LimitReached { .. } => "limit_reached",
        RunStatus::Stopped { .. } => "stopped",
        RunStatus::Stuck { .. } => "stuck",
        RunStatus::Cancelled => "cancelled",
        RunStatus::ProviderFailed { .. } => "provider_failed",
        RunStatus::Finished { .. } => "finished",
        _ => "unknown",
    }
}

/// Assertions on a run's outcome and trajectory. Every field is optional;
/// all set fields are checked and every failure is reported.
///
/// "Used" means the model proposed the call and the loop recorded a receipt
/// for it, whatever its status (a denied call is still used). Use
/// `receipt_statuses` to assert what actually happened.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ExpectedStatus>,
    /// Whether the run's objective was met
    /// ([`aidaemon_core::RunOutcome::is_succeeded`]). `completed` only says
    /// the run stopped; a run that stops with unmet requirements is
    /// `completed` but not `succeeded`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub succeeded: Option<bool>,
    /// Case-insensitive substrings the final text must contain.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub final_text_contains: Vec<String>,
    /// Case-insensitive substrings the final text must not contain.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub final_text_not_contains: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_used: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_not_used: Vec<String>,
    /// Tools that must appear in this relative order (subsequence match:
    /// unrelated calls may interleave).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_in_order: Vec<String>,
    /// Exact call count per tool. Tools not listed are unconstrained.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_call_counts: BTreeMap<String, u32>,
    /// Exact sequence of receipt statuses per tool, e.g.
    /// `{delete_file: [denied]}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub receipt_statuses: BTreeMap<String, Vec<ReceiptStatus>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_billable_tokens: Option<u64>,
    /// No receipt may be failed, timed out, cancelled, unknown-tool,
    /// invalid-arguments or indeterminate. Policy denials are allowed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_failed_receipts: bool,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

impl Scenario {
    /// Parse and validate one scenario. `origin` is used in error messages.
    pub fn from_yaml_str(yaml: &str, origin: impl AsRef<Path>) -> Result<Self, EvalError> {
        let path = origin.as_ref().to_path_buf();
        let mut scenario: Scenario =
            serde_yaml::from_str(yaml).map_err(|err| EvalError::Parse {
                path: path.clone(),
                message: err.to_string(),
            })?;
        scenario.source = Some(path.clone());
        scenario
            .validate()
            .map_err(|message| EvalError::Invalid { path, message })?;
        Ok(scenario)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, EvalError> {
        let path = path.as_ref();
        let yaml = std::fs::read_to_string(path).map_err(|source| EvalError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_yaml_str(&yaml, path)
    }

    /// Structural checks serde cannot express.
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("`name` must not be empty".into());
        }
        if let Some(script) = &self.script {
            if script.is_empty() {
                return Err("`script` must have at least one reply (omit it for live runs)".into());
            }
            for (i, reply) in script.iter().enumerate() {
                let no_calls = reply.tool_calls.as_ref().is_none_or(Vec::is_empty);
                if reply.text.is_none() && no_calls {
                    return Err(format!(
                        "script reply {} needs `text` or non-empty `tool_calls`",
                        i + 1
                    ));
                }
                for call in reply.tool_calls.iter().flatten() {
                    if call.name.trim().is_empty() {
                        return Err(format!("script reply {}: tool call without a name", i + 1));
                    }
                }
            }
        }
        if let Some(limits) = &self.limits {
            if limits.max_turns == Some(0) {
                return Err("`limits.max_turns` must be at least 1".into());
            }
        }
        Ok(())
    }

    pub fn is_scripted(&self) -> bool {
        self.script.is_some()
    }

    /// The script as core model responses.
    pub fn scripted_responses(&self) -> Vec<ModelResponse> {
        self.script
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, reply)| reply.to_response(i + 1))
            .collect()
    }

    /// Apply the scenario's instructions and limits to a builder. Factories
    /// call this so per-scenario settings take effect.
    pub fn configure(&self, mut builder: AgentBuilder) -> AgentBuilder {
        if let Some(instructions) = &self.instructions {
            builder = builder.instructions(instructions.clone());
        }
        if let Some(limits) = self.limits {
            let mut core = Limits::default();
            if let Some(turns) = limits.max_turns {
                core.max_turns = turns;
            }
            core.max_tokens = limits.max_tokens;
            builder = builder.limits(core);
        }
        builder
    }
}

impl ScriptedReply {
    pub fn to_response(&self, reply_number: usize) -> ModelResponse {
        let tool_calls = self
            .tool_calls
            .iter()
            .flatten()
            .enumerate()
            .map(|(i, call)| {
                ToolCall::new(
                    call.id
                        .clone()
                        .unwrap_or_else(|| format!("call_{reply_number}_{i}")),
                    call.name.clone(),
                    call.arguments.to_value(),
                )
            })
            .collect();
        let usage = match self.usage {
            Some(u) => Usage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cached_input_tokens: u.cached_input_tokens,
                cache_creation_tokens: 0,
            },
            None => SCRIPTED_USAGE,
        };
        ModelResponse {
            text: self.text.clone().unwrap_or_default(),
            tool_calls,
            usage,
            ..Default::default()
        }
    }
}

/// Load every `*.yaml` / `*.yml` file in `dir` (not recursive), sorted by
/// file name. Fails on the first bad file, naming it.
pub(crate) fn load_dir(dir: &Path) -> Result<Vec<Scenario>, EvalError> {
    let entries = std::fs::read_dir(dir).map_err(|source| EvalError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| EvalError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let is_yaml = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "yaml" || e == "yml");
        if is_yaml && path.is_file() {
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return Err(EvalError::Empty {
            path: dir.to_path_buf(),
        });
    }
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let mut scenarios: Vec<Scenario> = Vec::with_capacity(paths.len());
    for path in paths {
        let scenario = Scenario::from_file(&path)?;
        if let Some(prev) = scenarios.iter().find(|s| s.name == scenario.name) {
            return Err(EvalError::DuplicateName {
                name: scenario.name,
                first: prev.source.clone().unwrap_or_default(),
                second: path,
            });
        }
        scenarios.push(scenario);
    }
    Ok(scenarios)
}
