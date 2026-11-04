use serde_json::{Map, Value};

use super::ConformanceReport;
use crate::tool::Tool;

/// Stable names of the checks run by [`check_tool`], each documenting the
/// rule it enforces.
pub mod tool_checks {
    /// `spec().name` matches `^[a-zA-Z0-9_-]{1,64}$`, the character set and
    /// length every major provider accepts for function names.
    pub const NAME_FORMAT: &str = "tool.name_format";
    /// `spec().description` is not empty or whitespace: the model chooses
    /// tools by their description.
    pub const DESCRIPTION_NON_EMPTY: &str = "tool.description_non_empty";
    /// `spec().parameters` is a JSON object schema with `"type": "object"`.
    pub const PARAMETERS_OBJECT_SCHEMA: &str = "tool.parameters_object_schema";
    /// `spec().parameters` has `"additionalProperties": false`, so strict
    /// providers accept it and invented arguments are rejected.
    pub const PARAMETERS_CLOSED: &str = "tool.parameters_closed";
    /// `properties` (if present) is an object, `required` (if present) is an
    /// array of strings, and every `required` key is declared in
    /// `properties`.
    pub const REQUIRED_KEYS_DECLARED: &str = "tool.required_keys_declared";
    /// `validate` accepts a minimal valid call: the author-supplied sample
    /// ([`super::ToolKitOptions::valid_arguments`]) or one generated from
    /// the schema by [`super::sample_arguments`].
    pub const VALIDATE_ACCEPTS_MINIMAL_CALL: &str = "tool.validate_accepts_minimal_call";
    /// `validate` rejects the minimal valid call with any one `required`
    /// key removed, so malformed proposals never reach policies or I/O.
    pub const VALIDATE_REJECTS_MISSING_REQUIRED: &str = "tool.validate_rejects_missing_required";
    /// `validate` rejects arguments that are not a JSON object (the kernel
    /// contract for object schemas).
    pub const VALIDATE_REJECTS_NON_OBJECT: &str = "tool.validate_rejects_non_object";
    /// A tool annotated `read_only` is not also `destructive`; the two are
    /// contradictory and break scheduling and approval policies.
    pub const READ_ONLY_NOT_DESTRUCTIVE: &str = "tool.read_only_not_destructive";
    /// `spec()` and `annotations()` return identical values on repeated
    /// calls, so the cacheable request prefix stays stable.
    pub const SPEC_STABLE: &str = "tool.spec_stable";
}

/// Options for [`check_tool_with`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ToolKitOptions {
    /// A known-valid argument object. Use it when `validate` checks more
    /// than the schema can express (a URL that must parse, a path that must
    /// exist) and the generated sample is rejected for that reason. It
    /// should contain only the required keys, so the
    /// missing-required check can remove them one by one.
    pub valid_arguments: Option<Value>,
}

impl ToolKitOptions {
    pub fn valid_arguments(mut self, arguments: Value) -> Self {
        self.valid_arguments = Some(arguments);
        self
    }
}

/// Run every tool check (see [`tool_checks`]) with a schema-generated
/// sample call. Never calls [`Tool::call`], so it is safe for tools with
/// side effects.
pub fn check_tool(tool: &dyn Tool) -> ConformanceReport {
    check_tool_with(tool, ToolKitOptions::default())
}

/// [`check_tool`] with options, for tools whose `validate` needs a
/// hand-written valid call.
pub fn check_tool_with(tool: &dyn Tool, options: ToolKitOptions) -> ConformanceReport {
    use tool_checks::*;

    let spec = tool.spec();
    let mut report = ConformanceReport::new(if spec.name.is_empty() {
        "<unnamed tool>".to_string()
    } else {
        spec.name.clone()
    });
    let params = &spec.parameters;

    report.record(NAME_FORMAT, check_name(&spec.name));
    report.record(
        DESCRIPTION_NON_EMPTY,
        if spec.description.trim().is_empty() {
            Err("description is empty".into())
        } else {
            Ok(())
        },
    );
    let is_object_schema =
        params.is_object() && params.get("type").and_then(Value::as_str) == Some("object");
    report.record(
        PARAMETERS_OBJECT_SCHEMA,
        if is_object_schema {
            Ok(())
        } else {
            Err(format!(
                "parameters must be a JSON object with \"type\": \"object\", got {}",
                abbreviate(params)
            ))
        },
    );
    report.record(
        PARAMETERS_CLOSED,
        match params.get("additionalProperties") {
            Some(Value::Bool(false)) => Ok(()),
            Some(other) => Err(format!(
                "\"additionalProperties\" must be false, got {other}"
            )),
            None => Err("\"additionalProperties\": false is missing".into()),
        },
    );
    let required = required_keys(params);
    report.record(REQUIRED_KEYS_DECLARED, check_required_declared(params));

    let (sample, supplied) = match options.valid_arguments {
        Some(args) => (args, true),
        None => (sample_arguments(params), false),
    };
    let origin = if supplied {
        "supplied valid_arguments"
    } else {
        "schema-generated sample"
    };
    let accepted = tool.validate(&sample);
    report.record(
        VALIDATE_ACCEPTS_MINIMAL_CALL,
        accepted.clone().map_err(|e| {
            let hint = if supplied {
                ""
            } else {
                " (if validate needs more than the schema expresses, pass \
                 ToolKitOptions::valid_arguments)"
            };
            format!("validate rejected the {origin} {sample}: {e}{hint}")
        }),
    );

    let keys = required.as_ref().cloned().unwrap_or_default();
    if keys.is_empty() {
        report.pass(VALIDATE_REJECTS_MISSING_REQUIRED);
        report.note(format!(
            "{VALIDATE_REJECTS_MISSING_REQUIRED}: schema has no required keys; check is vacuous"
        ));
    } else if let Some(object) = sample.as_object() {
        let accepted_with: Vec<String> = keys
            .iter()
            .filter(|key| {
                let mut partial = object.clone();
                partial.remove(key.as_str());
                tool.validate(&Value::Object(partial)).is_ok()
            })
            .cloned()
            .collect();
        report.record(
            VALIDATE_REJECTS_MISSING_REQUIRED,
            if accepted_with.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "validate accepted the {origin} with required key(s) removed: {}",
                    accepted_with.join(", ")
                ))
            },
        );
    } else {
        report.fail(
            VALIDATE_REJECTS_MISSING_REQUIRED,
            format!("the {origin} is not a JSON object: {sample}"),
        );
    }

    if is_object_schema {
        let rejected_all = [
            Value::String("not an object".into()),
            Value::Array(Vec::new()),
            Value::Null,
        ]
        .into_iter()
        .filter(|bad| tool.validate(bad).is_ok())
        .map(|bad| bad.to_string())
        .collect::<Vec<_>>();
        report.record(
            VALIDATE_REJECTS_NON_OBJECT,
            if rejected_all.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "validate accepted non-object arguments: {}",
                    rejected_all.join(", ")
                ))
            },
        );
    } else {
        report.fail(
            VALIDATE_REJECTS_NON_OBJECT,
            "parameters are not an object schema, so non-object arguments cannot be rejected"
                .to_string(),
        );
    }

    let annotations = tool.annotations();
    report.record(
        READ_ONLY_NOT_DESTRUCTIVE,
        if annotations.read_only && annotations.destructive {
            Err("annotated both read_only and destructive".into())
        } else {
            Ok(())
        },
    );

    let again = tool.spec();
    report.record(
        SPEC_STABLE,
        if again != spec {
            Err("spec() changed between two calls".into())
        } else if tool.annotations() != annotations {
            Err("annotations() changed between two calls".into())
        } else {
            Ok(())
        },
    );
    report
}

fn check_name(name: &str) -> Result<(), String> {
    let len = name.chars().count();
    if len == 0 || len > 64 {
        return Err(format!("name must be 1-64 characters, got {len}"));
    }
    match name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-'))
    {
        Some(bad) => Err(format!(
            "name {name:?} contains {bad:?}; allowed: ASCII letters, digits, '_' and '-'"
        )),
        None => Ok(()),
    }
}

fn required_keys(params: &Value) -> Option<Vec<String>> {
    match params.get("required") {
        None => Some(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(_) => None,
    }
}

fn check_required_declared(params: &Value) -> Result<(), String> {
    let properties = match params.get("properties") {
        None => None,
        Some(Value::Object(map)) => Some(map),
        Some(other) => {
            return Err(format!(
                "\"properties\" must be an object, got {}",
                abbreviate(other)
            ))
        }
    };
    let Some(required) = required_keys(params) else {
        return Err("\"required\" must be an array of strings".into());
    };
    let missing: Vec<&str> = required
        .iter()
        .filter(|k| !properties.is_some_and(|p| p.contains_key(k.as_str())))
        .map(String::as_str)
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "required key(s) not declared in properties: {}",
            missing.join(", ")
        ))
    }
}

fn abbreviate(value: &Value) -> String {
    let text = value.to_string();
    if text.chars().count() > 80 {
        format!("{}...", text.chars().take(80).collect::<String>())
    } else {
        text
    }
}

/// Build a minimal argument value from a JSON Schema: only `required`
/// properties, each filled from `const`, `default`, the first `enum` value or
/// the first of `examples` when present, otherwise from its type (non-empty
/// distinct strings honouring `format`/`minLength`, numbers honouring
/// `minimum`, `false`, arrays of `minItems` samples, nested objects).
///
/// Exposed so authors can see what [`check_tool`] sends to `validate`.
pub fn sample_arguments(schema: &Value) -> Value {
    sample_value(schema, "value", 0)
}

fn sample_value(schema: &Value, key: &str, depth: usize) -> Value {
    if depth > 16 {
        return Value::Null;
    }
    let Some(obj) = schema.as_object() else {
        return Value::String(format!("sample-{key}"));
    };
    for field in ["const", "default"] {
        if let Some(v) = obj.get(field) {
            return v.clone();
        }
    }
    for field in ["enum", "examples"] {
        if let Some(first) = obj
            .get(field)
            .and_then(Value::as_array)
            .and_then(|a| a.first())
        {
            return first.clone();
        }
    }
    for field in ["anyOf", "oneOf", "allOf"] {
        if let Some(first) = obj.get(field).and_then(Value::as_array).and_then(|a| {
            a.iter()
                .find(|s| s.get("type").and_then(Value::as_str) != Some("null"))
                .or_else(|| a.first())
        }) {
            return sample_value(first, key, depth + 1);
        }
    }
    let ty = match obj.get("type") {
        Some(Value::String(t)) => t.as_str(),
        Some(Value::Array(types)) => types
            .iter()
            .filter_map(Value::as_str)
            .find(|t| *t != "null")
            .unwrap_or("null"),
        _ if obj.contains_key("properties") => "object",
        _ if obj.contains_key("items") => "array",
        _ => "string",
    };
    match ty {
        "object" => {
            let props = obj.get("properties").and_then(Value::as_object);
            let mut out = Map::new();
            for key in obj
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                let prop = props.and_then(|p| p.get(key)).unwrap_or(&Value::Null);
                out.insert(key.to_string(), sample_value(prop, key, depth + 1));
            }
            Value::Object(out)
        }
        "array" => {
            let n = obj.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
            let item = obj.get("items").unwrap_or(&Value::Null);
            Value::Array((0..n).map(|_| sample_value(item, key, depth + 1)).collect())
        }
        "integer" | "number" => sample_number(obj, ty == "integer"),
        "boolean" => Value::Bool(false),
        "null" => Value::Null,
        _ => sample_string(obj, key),
    }
}

fn sample_number(obj: &Map<String, Value>, integer: bool) -> Value {
    let min = obj.get("minimum").and_then(Value::as_f64);
    let exclusive_min = obj.get("exclusiveMinimum").and_then(Value::as_f64);
    let max = obj
        .get("maximum")
        .and_then(Value::as_f64)
        .or_else(|| obj.get("exclusiveMaximum").and_then(Value::as_f64));
    let mut n = match (min, exclusive_min) {
        (Some(m), _) => m,
        (None, Some(m)) => m + 1.0,
        (None, None) => 1.0,
    };
    if let Some(max) = max {
        if n > max {
            n = max;
        }
    }
    if integer {
        Value::from(n.ceil() as i64)
    } else {
        serde_json::Number::from_f64(n)
            .map(Value::Number)
            .unwrap_or_else(|| Value::from(1))
    }
}

fn sample_string(obj: &Map<String, Value>, key: &str) -> Value {
    let mut s = match obj.get("format").and_then(Value::as_str) {
        Some("uri" | "url" | "iri" | "uri-reference") => "https://example.com/".to_string(),
        Some("email") => "alice.rivera@example.com".to_string(),
        Some("date-time") => "2026-01-01T00:00:00Z".to_string(),
        Some("date") => "2026-01-01".to_string(),
        Some("time") => "00:00:00Z".to_string(),
        Some("uuid") => "00000000-0000-4000-8000-000000000000".to_string(),
        Some("hostname") => "example.com".to_string(),
        _ => format!("sample-{key}"),
    };
    let min = obj.get("minLength").and_then(Value::as_u64).unwrap_or(0) as usize;
    while s.chars().count() < min {
        s.push('x');
    }
    if let Some(max) = obj.get("maxLength").and_then(Value::as_u64) {
        s = s.chars().take(max as usize).collect();
    }
    Value::String(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::spec;
    use crate::tool::{FnTool, ToolAnnotations, ToolOutput};
    use crate::ToolSpec;
    use serde_json::json;

    fn tool(spec: ToolSpec, annotations: ToolAnnotations) -> impl Tool {
        FnTool::new(spec, annotations, |_args: Value| async {
            Ok(ToolOutput::text("ok"))
        })
    }

    #[test]
    fn a_well_formed_tool_passes() {
        let report = check_tool(&tool(
            spec("echo", "Echo", &["text"]),
            ToolAnnotations::READ_ONLY,
        ));
        report.assert_all_passed();
        assert_eq!(report.passed.len(), 10);
    }

    #[test]
    fn every_problem_is_reported_at_once() {
        let bad = ToolSpec {
            name: "bad name!".into(),
            description: " ".into(),
            parameters: json!({"type": "object", "required": ["path"]}),
        };
        let report = check_tool(&tool(
            bad,
            ToolAnnotations {
                read_only: true,
                destructive: true,
                idempotent: true,
                egress: false,
            },
        ));
        for check in [
            tool_checks::NAME_FORMAT,
            tool_checks::DESCRIPTION_NON_EMPTY,
            tool_checks::PARAMETERS_CLOSED,
            tool_checks::REQUIRED_KEYS_DECLARED,
            tool_checks::READ_ONLY_NOT_DESTRUCTIVE,
        ] {
            assert!(report.failure(check).is_some(), "{check}: {report}");
        }
        assert!(report.passed_check(tool_checks::VALIDATE_REJECTS_MISSING_REQUIRED));
    }

    #[test]
    fn samples_follow_the_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
                "limit": {"type": "integer", "minimum": 5},
                "flag": {"type": "boolean"},
                "tags": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 12}},
                "nested": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]},
                "optional": {"type": "string"}
            },
            "required": ["url", "mode", "limit", "flag", "tags", "nested"]
        });
        assert_eq!(
            sample_arguments(&schema),
            json!({
                "url": "https://example.com/",
                "mode": "fast",
                "limit": 5,
                "flag": false,
                "tags": ["sample-tagsx"],
                "nested": {"a": 1.0}
            })
        );
    }
}
