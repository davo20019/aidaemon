//! JSON Schemas from Rust types, shaped the way model providers take them.

use schemars::JsonSchema;
use schemars::generate::SchemaSettings;
use serde_json::Value;

/// The JSON Schema for `T`, inlined (no `$defs`/`$ref` unless the type is
/// recursive) and without the `$schema` and `title` keys, which providers
/// either reject or ignore.
pub fn schema_for<T: JsonSchema + ?Sized>() -> Value {
    let settings = SchemaSettings::draft2020_12().with(|settings| {
        settings.inline_subschemas = true;
        settings.meta_schema = None;
    });
    let mut value = settings
        .into_generator()
        .into_root_schema_for::<T>()
        .to_value();
    if let Some(object) = value.as_object_mut() {
        object.remove("title");
        object.remove("$schema");
    }
    value
}

/// The parameters schema of a tool whose arguments are `T`: an object
/// schema that is closed (`"additionalProperties": false`) at the top, the
/// shape the `Tool` conformance kit and strict providers expect.
pub fn tool_parameters<T: JsonSchema + ?Sized>() -> Value {
    let mut value = schema_for::<T>();
    if let Some(object) = value.as_object_mut() {
        object.insert("type".into(), Value::String("object".into()));
        object
            .entry("properties")
            .or_insert_with(|| Value::Object(Default::default()));
        object.insert("additionalProperties".into(), Value::Bool(false));
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A synthetic nested argument type.
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Outer {
        /// The inner part.
        inner: Inner,
        note: Option<String>,
    }

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Inner {
        count: u32,
    }

    #[test]
    fn nested_types_are_inlined_and_the_top_is_closed() {
        let schema = tool_parameters::<Outer>();
        assert!(schema.get("$defs").is_none(), "{schema}");
        assert!(schema.get("title").is_none());
        assert!(schema.get("$schema").is_none());
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(
            schema["properties"]["inner"]["description"],
            "The inner part."
        );
        assert_eq!(
            schema["properties"]["inner"]["properties"]["count"]["type"],
            "integer"
        );
        assert_eq!(schema["required"], json!(["inner"]));
    }
}
