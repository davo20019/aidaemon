use std::fs;
use std::path::{Path, PathBuf};

const MAX_SCHEMA_SEGMENT_CHARS: usize = 6_500;
const MAX_TOTAL_SCHEMA_SEGMENT_CHARS: usize = 90_000;

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

fn schema_segment(source: &str) -> Option<&str> {
    let (_, after_schema_start) = source.split_once("fn schema(&self) -> Value {")?;
    let (segment, _) = after_schema_start.split_once("async fn call(")?;
    Some(segment)
}

#[test]
fn all_tool_schemas_disable_additional_properties() {
    for file in tool_source_files() {
        let source = fs::read_to_string(&file).expect("read tool source");
        let segment = schema_segment(&source)
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
        let segment = schema_segment(&source)
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
