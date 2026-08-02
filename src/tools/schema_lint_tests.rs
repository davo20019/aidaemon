use std::fs;
use std::path::{Path, PathBuf};

const MAX_SCHEMA_SEGMENT_CHARS: usize = 6_500;
// Reference-image editing adds one bounded input to `generate_image`; retain a
// hard aggregate ceiling while accounting for that user-visible capability.
const MAX_TOTAL_SCHEMA_SEGMENT_CHARS: usize = 101_000;

fn tools_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tools")
}

fn tool_source_files() -> Vec<PathBuf> {
    let mut files = fs::read_dir(tools_dir())
        .expect("read src/tools")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .filter(|path| path.file_name().and_then(|n| n.to_str()) != Some("mod.rs"))
        .filter(|path| path.file_name().and_then(|n| n.to_str()) != Some("schema_lint_tests.rs"))
        .filter(|path| {
            fs::read_to_string(path)
                .ok()
                .is_some_and(|src| src.contains("impl Tool for"))
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

/// Returns the full source region covering schema-related definitions:
/// the inline `fn schema` body plus any private helper such as
/// `fn <tool>_schema() -> Value` that the method may delegate to.
fn schema_source_region(source: &str) -> Option<&str> {
    // If the schema fn delegates to a private helper (Pillar-C pattern), find
    // the helper's definition start so we capture the JSON content.
    // Convention: helper is `fn <name>_schema() -> Value {` appearing before
    // `impl Tool for`.
    let start = if let Some(pos) = source.find("_schema() -> Value {") {
        // Walk back to the `fn ` keyword
        let prefix = &source[..pos];
        prefix.rfind("fn ").unwrap_or(0)
    } else {
        // No helper — fall through to schema_segment
        let (_, after) = source.split_once("fn schema(&self) -> Value {")?;
        return after.split_once("async fn call(").map(|(seg, _)| seg);
    };

    let region = &source[start..];
    let (segment, _) = region.split_once("async fn call(")?;
    Some(segment)
}

#[test]
fn all_tool_schemas_disable_additional_properties() {
    for file in tool_source_files() {
        let source = fs::read_to_string(&file).expect("read tool source");
        let segment = schema_source_region(&source)
            .unwrap_or_else(|| panic!("Could not locate schema segment in {}", file.display()));
        assert!(
            segment.contains("\"additionalProperties\": false"),
            "Schema must include parameters.additionalProperties=false: {}",
            file.display()
        );
    }
}

#[test]
fn all_tools_define_explicit_capabilities() {
    for file in tool_source_files() {
        let source = fs::read_to_string(&file).expect("read tool source");
        assert!(
            source.contains("fn capabilities(&self) -> ToolCapabilities"),
            "Tool must define explicit capabilities: {}",
            file.display()
        );
    }
}

#[test]
fn tools_do_not_silently_fallback_to_empty_arguments() {
    for file in tool_source_files() {
        let source = fs::read_to_string(&file).expect("read tool source");
        assert!(
            !source.contains("from_str(arguments).unwrap_or(json!({}))"),
            "Tool must not swallow argument parse errors via empty-object fallback: {}",
            file.display()
        );
    }
}

#[test]
fn schema_payload_budget_stays_bounded() {
    let mut total = 0usize;
    for file in tool_source_files() {
        let source = fs::read_to_string(&file).expect("read tool source");
        let segment = schema_source_region(&source)
            .unwrap_or_else(|| panic!("Could not locate schema segment in {}", file.display()));
        total += segment.len();
        assert!(
            segment.len() <= MAX_SCHEMA_SEGMENT_CHARS,
            "Schema segment too large ({} chars) in {}",
            segment.len(),
            file.display()
        );
    }

    assert!(
        total <= MAX_TOTAL_SCHEMA_SEGMENT_CHARS,
        "Total schema segment payload too large: {} chars (max {})",
        total,
        MAX_TOTAL_SCHEMA_SEGMENT_CHARS
    );
}

#[test]
fn tools_do_not_embed_truncation_notices_in_output() {
    // Tools must set ToolCallMetadata.truncation instead of embedding
    // truncation_notice text — see 2026-07-01 scaffolding-leak incidents.
    // Background delivery paths render via render_truncation_notice, which is allowed.
    // Strip allowed render_truncation_notice call sites first, then any remaining
    // truncation_notice( / truncation_notice_with_hint( occurrence is a forbidden
    // direct text-embedding call regardless of import style (fully-qualified, bare,
    // or braced-group import).
    for file in tool_source_files() {
        let source = fs::read_to_string(&file).expect("read tool source");
        let stripped = source.replace("render_truncation_notice(", "");
        assert!(
            !stripped.contains("truncation_notice(") && !stripped.contains("truncation_notice_with_hint("),
            "{} embeds a truncation notice in tool output; set metadata.truncation instead \
             (render_truncation_notice is allowed only for delivery paths that bypass the agent loop)",
            file.display()
        );
    }
}
