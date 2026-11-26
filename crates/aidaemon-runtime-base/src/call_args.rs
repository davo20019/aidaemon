//! A tool call's arguments as the dispatcher executes them, below the engine
//! so approvals can name the canonical form.

use serde_json::Value;

/// A call as the dispatcher executes it: the model's `_` keys and unset
/// optional arguments dropped, the turn's project directory injected and
/// unauthorized read roots dropped (`sdk_engine::tool_calls::canonicalize_call`).
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalCall {
    pub arguments: serde_json::Value,
    /// The model's own explicit directory; it refines the turn's known
    /// project directory when the call is dispatched.
    pub explicit_dir: Option<String>,
    /// The project directory injected into the call, if any.
    pub injected_dir: Option<String>,
    /// Declared read roots dropped as unauthorized.
    pub stripped_read_roots: Vec<String>,
}

impl CanonicalCall {
    /// Only the model's `_` keys and unset optional arguments dropped (no
    /// turn to anchor against).
    pub fn stripped(
        tool: Option<&dyn aidaemon_tools::OutcomeTool>,
        raw: &serde_json::Value,
    ) -> Self {
        let mut arguments = raw.clone();
        if let serde_json::Value::Object(map) = &mut arguments {
            strip_model_arguments(map, tool);
        }
        Self {
            arguments,
            explicit_dir: None,
            injected_dir: None,
            stripped_read_roots: Vec::new(),
        }
    }
}

/// Drop optional top-level arguments the model filled with an empty string
/// or an empty list. Some models send every declared field and use `""` or
/// `[]` for "not provided"; a tool would otherwise read them as values (an
/// empty session filter that matches nothing, an empty `schedules` list that
/// hides `schedule`). Required fields, numbers and booleans are untouched.
/// The model's arguments as the dispatcher reads them: any `_`-prefixed key
/// the model sent is dropped (no tool reads one: the dispatcher's facts
/// reach a tool as its typed `ToolCallContext`), and optional arguments the
/// model left unset are dropped against the tool's schema. Every kernel hook
/// judges a call after this same step (`sdk_engine::tool_calls::canonicalize_call`),
/// so the judged call is the executed one.
pub fn strip_model_arguments(
    map: &mut serde_json::Map<String, Value>,
    tool: Option<&dyn aidaemon_tools::OutcomeTool>,
) {
    // The kernel's own rule, so the daemon and the SDK strip alike.
    aidaemon_core::strip_reserved_arguments(map);
    if let Some(tool) = tool {
        drop_unset_optional_arguments(map, &tool.schema());
    }
}

pub fn drop_unset_optional_arguments(map: &mut serde_json::Map<String, Value>, schema: &Value) {
    let parameters = schema.get("parameters").unwrap_or(schema);
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    map.retain(|key, value| {
        if required.contains(&key.as_str()) {
            return true;
        }
        match value {
            Value::Null => false,
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(items) => !items.is_empty(),
            _ => true,
        }
    });
}
