use std::fs;
use std::path::{Path, PathBuf};

const MAX_SCHEMA_SEGMENT_CHARS: usize = 6_500;
// Reference-image editing and the owner-facing bounded-mandate contract add
// explicit inputs, including non-combinable operation scopes. Retain a hard
// aggregate ceiling while accounting for those user-visible capabilities.
const MAX_TOTAL_SCHEMA_SEGMENT_CHARS: usize = 107_000;

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

/// Return exactly one Rust function, ending at its balanced closing brace.
/// Strings and comments are skipped so braces in schema descriptions do not
/// alter the boundary. This keeps the schema budget independent of whatever
/// adapter hooks happen to follow `schema()` in a `Tool` implementation.
fn rust_function_region(source: &str, start: usize) -> Option<&str> {
    #[derive(Clone, Copy)]
    enum ScanState {
        Code,
        String,
        Char,
        LineComment,
        BlockComment(usize),
        RawString(usize),
    }

    let bytes = source.as_bytes();
    let mut index = source[start..].find('{')? + start;
    let mut depth = 0usize;
    let mut state = ScanState::Code;
    let mut escaped = false;

    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        match state {
            ScanState::Code => {
                if byte == b'/' && next == Some(b'/') {
                    state = ScanState::LineComment;
                    index += 1;
                } else if byte == b'/' && next == Some(b'*') {
                    state = ScanState::BlockComment(1);
                    index += 1;
                } else if byte == b'r' {
                    let mut cursor = index + 1;
                    while bytes.get(cursor) == Some(&b'#') {
                        cursor += 1;
                    }
                    if bytes.get(cursor) == Some(&b'"') {
                        state = ScanState::RawString(cursor - index - 1);
                        index = cursor;
                    }
                } else if byte == b'"' {
                    state = ScanState::String;
                    escaped = false;
                } else if byte == b'\'' {
                    state = ScanState::Char;
                    escaped = false;
                } else if byte == b'{' {
                    depth += 1;
                } else if byte == b'}' {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return source.get(start..=index);
                    }
                }
            }
            ScanState::String => {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    state = ScanState::Code;
                }
            }
            ScanState::Char => {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'\'' {
                    state = ScanState::Code;
                }
            }
            ScanState::LineComment => {
                if byte == b'\n' {
                    state = ScanState::Code;
                }
            }
            ScanState::BlockComment(comment_depth) => {
                if byte == b'/' && next == Some(b'*') {
                    state = ScanState::BlockComment(comment_depth + 1);
                    index += 1;
                } else if byte == b'*' && next == Some(b'/') {
                    state = if comment_depth == 1 {
                        ScanState::Code
                    } else {
                        ScanState::BlockComment(comment_depth - 1)
                    };
                    index += 1;
                }
            }
            ScanState::RawString(hashes) => {
                if byte == b'"'
                    && (0..hashes).all(|offset| bytes.get(index + 1 + offset) == Some(&b'#'))
                {
                    state = ScanState::Code;
                    index += hashes;
                }
            }
        }
        index += 1;
    }
    None
}

/// Returns only the schema-producing function: the private schema helper when
/// present, otherwise the inline `Tool::schema` implementation.
fn schema_source_region(source: &str) -> Option<&str> {
    let start = if let Some(pos) = source.find("_schema() -> Value {") {
        source[..pos].rfind("fn ")?
    } else {
        source.find("fn schema(&self) -> Value {")?
    };
    rust_function_region(source, start)
}

#[test]
fn schema_region_is_not_extended_by_following_adapter_hooks() {
    let source = r###"
        fn synthetic_schema() -> Value {
            json!({"description": "literal } and {", "raw": r#"}"#})
        }

        fn validate_arguments(&self) {
            // This hook is deliberately much larger than the schema.
            if true { nested(); }
        }
    "###;
    let region = schema_source_region(source).expect("schema function");
    assert!(region.contains("literal } and {"));
    assert!(!region.contains("validate_arguments"));
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
