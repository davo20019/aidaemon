//! Tools that report a typed [`ToolCallOutcome`].
//!
//! An [`OutcomeTool`] is written once. An application with its own dispatcher
//! calls [`OutcomeTool::execute`] and keeps the typed metadata; a plain
//! [`aidaemon_core`] agent gets the same tool through [`aidaemon_core::Tool`].

use async_trait::async_trait;
use serde_json::{json, Value};

use aidaemon_core::{ToolOutput, ToolResult, ToolSpec};

use crate::semantics::{ToolCallContext, ToolCallOutcome, ToolSemantics};

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
