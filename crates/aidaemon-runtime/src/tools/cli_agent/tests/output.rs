//! Output parsing: loop detection, JSON extraction, stream-json.

use super::*;

/// The daemon's state database stays guarded by its file name; a
/// project's own SQLite database is the agent's to use.
#[test]
fn only_the_daemons_database_is_a_prohibited_command() {
    let blocked = |command: &str| prohibited_cli_agent_command_reason(command).is_some();
    // Opening or changing it, or its key.
    assert!(blocked("sqlite3 ~/.aidaemon/aidaemon.db .tables"));
    assert!(blocked("sqlcipher ~/projects/aidaemon/aidaemon.db"));
    assert!(blocked("rm -f aidaemon.db-wal"));
    assert!(blocked("cp backup.db ~/projects/aidaemon/aidaemon.db"));
    assert!(blocked("echo x > aidaemon.db"));
    assert!(blocked("cd /x && mv aidaemon.db old.db"));
    assert!(blocked(
        "python3 -c \"import sqlite3; sqlite3.connect('aidaemon.db').execute('delete from facts')\""
    ));
    assert!(blocked("sqlite3 x.db \"PRAGMA key = 'abc'\""));
    // The key pragma in a here-document, on a copy of the database.
    assert!(blocked(
        "sqlcipher /tmp/copy.db <<EOF\nPRAGMA key = 'abc';\nselect 1;\nEOF"
    ));
    assert!(blocked("brew install sqlcipher"));
    // A project's own database.
    assert!(!blocked("sqlite3 app.db 'select 1'"));
    assert!(!blocked("python3 -c 'import sqlite3'"));
    assert!(!blocked("sqlcipher data.db"));
}

/// The harness working on its own code reads and searches freely (live
/// 2026-11-27: a memory-improvement run was killed for listing the file).
#[test]
fn reading_or_working_on_the_harness_code_is_not_prohibited() {
    let allowed = |command: &str| prohibited_cli_agent_command_reason(command).is_none();
    assert!(allowed(
        "rg -n \"struct ToolExecCtx\" crates/ && rg -n \"v1\\.schema\\.json|UPDATE_SCHEMA\" crates/ scripts/ \
         && echo \"---DB_PROBE---\" && ls -la src/bin/ && echo \"---AIDAEMON_DATA---\" \
         && ls -la ~/projects/aidaemon/aidaemon.db* ~/.aidaemon/aidaemon.db* 2>/dev/null || true"
    ));
    assert!(allowed("stat aidaemon.db && du -sh aidaemon.db-wal"));
    assert!(allowed(
        "rg -n 'aidaemon.db|pragma key|sqlcipher' crates src"
    ));
    assert!(allowed("cargo test -q --lib memory"));
    assert!(allowed("cargo tree -e features | grep sqlcipher"));
    assert!(allowed(
        "sed -i '' 's/default.db/aidaemon.db.example/' docs/notes.md"
    ));
    assert!(allowed(
        "git diff -- crates/aidaemon-memory/src/sqlite/schema.rs"
    ));
}

#[test]
fn test_loop_detector_no_false_positive() {
    let mut detector = LoopDetector::new();
    for i in 0..20 {
        assert!(!detector.add_line(&format!("unique line {}", i)));
    }
}

#[test]
fn test_loop_detector_catches_repetition() {
    let mut detector = LoopDetector::new();
    // Add the same line 50+ times
    for i in 0..LOOP_DETECTION_THRESHOLD + 1 {
        let detected = detector.add_line("stuck in a loop");
        if i >= LOOP_DETECTION_THRESHOLD - 1 {
            // Should trigger around threshold
            if detected {
                return; // Test passes
            }
        }
    }
    panic!("Loop detector should have triggered");
}

#[test]
fn test_loop_detector_ignores_empty_lines() {
    let mut detector = LoopDetector::new();
    for _ in 0..200 {
        assert!(!detector.add_line(""));
        assert!(!detector.add_line("   "));
    }
}

// -----------------------------------------------------------------------
// JSON extraction tests
// -----------------------------------------------------------------------

#[test]
fn test_extract_meaningful_output_plain_text() {
    let output = "Hello, this is plain text output\nLine 2\n";
    let result = extract_meaningful_output(output, 10000);
    assert_eq!(result, output);
}

#[test]
fn test_extract_meaningful_output_json_result() {
    let output = r#"{"result": "The task is complete. Created 3 files."}"#;
    let result = extract_meaningful_output(output, 10000);
    assert_eq!(result, "The task is complete. Created 3 files.");
}

#[test]
fn test_extract_meaningful_output_json_output_field() {
    let output = r#"{"output": "Generated report successfully"}"#;
    let result = extract_meaningful_output(output, 10000);
    assert_eq!(result, "Generated report successfully");
}

#[test]
fn test_extract_meaningful_output_codex_jsonl_uses_final_agent_message() {
    let noisy_prefix = "repository guidance\n".repeat(2_000);
    let output = format!(
        "{}\n{}\n{}\n{}",
        serde_json::json!({"type":"thread.started","thread_id":"synthetic-thread"}),
        serde_json::json!({
            "type":"item.completed",
            "item":{
                "id":"item-1",
                "type":"command_execution",
                "aggregated_output":noisy_prefix,
                "exit_code":0,
                "status":"completed"
            }
        }),
        serde_json::json!({
            "type":"item.completed",
            "item":{
                "id":"item-2",
                "type":"agent_message",
                "text":"Published and verified https://example.test/posts/synthetic/"
            }
        }),
        serde_json::json!({"type":"turn.completed","usage":{"output_tokens":42}}),
    );

    let result = extract_meaningful_output(&output, 200);

    assert_eq!(
        result,
        "Published and verified https://example.test/posts/synthetic/"
    );
    assert!(!result.contains("repository guidance"));
    assert!(!result.contains("truncated"));
}

#[test]
fn test_extract_meaningful_output_codex_jsonl_prefers_last_agent_message() {
    let output = [
        serde_json::json!({
            "type":"item.completed",
            "item":{"type":"agent_message","text":"I will inspect the repository."}
        })
        .to_string(),
        serde_json::json!({
            "type":"item.completed",
            "item":{"type":"agent_message","text":"Publication completed and verified."}
        })
        .to_string(),
        serde_json::json!({"type":"turn.completed"}).to_string(),
    ]
    .join("\n");

    assert_eq!(
        extract_meaningful_output(&output, 10_000),
        "Publication completed and verified."
    );
}

#[test]
fn bounded_output_retains_final_codex_result_after_overflow() {
    let mut output = String::new();
    append_bounded_line(
        &mut output,
        "",
        &serde_json::json!({"type":"thread.started"}).to_string(),
        512,
    );
    append_bounded_line(
        &mut output,
        "",
        &serde_json::json!({
            "type":"item.completed",
            "item":{
                "type":"command_execution",
                "aggregated_output":"inspection noise ".repeat(200)
            }
        })
        .to_string(),
        512,
    );
    append_bounded_line(
        &mut output,
        "",
        &serde_json::json!({
            "type":"item.completed",
            "item":{
                "type":"agent_message",
                "text":"Published exactly once."
            }
        })
        .to_string(),
        512,
    );

    assert!(output.len() <= 512);
    assert!(output.contains(BUFFER_TRUNCATION_MARKER.trim()));
    assert_eq!(
        extract_meaningful_output(&output, 10_000),
        "Published exactly once."
    );
}

#[test]
fn test_extract_meaningful_output_truncation() {
    let output = "a".repeat(20000);
    let result = extract_meaningful_output(&output, 100);
    assert!(result.len() <= 200); // 100 chars + truncation note
    assert!(result.contains("truncated"));
}

#[test]
fn test_extract_progress_from_json_tool_use() {
    let json = r#"{"name":"Read","input":{"file_path":"/src/main.rs"}}"#;
    let progress = extract_progress_from_json(json);
    assert!(progress.is_some());
    assert!(progress.unwrap().contains("main.rs"));
}

#[test]
fn test_extract_progress_from_json_bash_command() {
    let json = r#"{"name":"Bash","input":{"command":"npm install"}}"#;
    let progress = extract_progress_from_json(json);
    assert!(progress.is_some());
    assert!(progress.unwrap().contains("npm install"));
}

#[test]
fn test_extract_progress_from_json_non_json() {
    let text = "This is just regular text";
    assert!(extract_progress_from_json(text).is_none());
}

#[test]
fn test_looks_like_json() {
    assert!(looks_like_json(r#"{"key": "value"}"#));
    assert!(looks_like_json(r#"[1, 2, 3]"#));
    assert!(looks_like_json(r#"  {"indented": true}  "#));
    assert!(!looks_like_json("plain text"));
    assert!(!looks_like_json(""));
}

// -----------------------------------------------------------------------
// Discovery tests
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_claude_stream_json_output_parsing() {
    // Simulates Claude Code's --output-format stream-json output
    let (tool, _db) = setup_bash_tool().await;

    let stream_json = r#"
echo '{"type":"assistant","message":{"content":[{"type":"text","text":"I will create the website."}]}}'
echo '{"type":"tool_use","name":"Bash","input":{"command":"mkdir -p website"}}'
echo '{"type":"tool_result","content":"Directory created"}'
echo '{"type":"result","result":"Website created successfully with index.html, style.css, and script.js"}'
"#;

    let args = json!({
        "action": "run",
        "tool": "bash-agent",
        "prompt": stream_json.trim()
    });
    let result = tool.call(&args.to_string()).await.unwrap();

    // extract_meaningful_output should pull out the "result" field
    assert!(
        result.contains("Website created successfully"),
        "Should extract result from JSON output, got: {}",
        result
    );
}

#[tokio::test]
async fn test_progress_extraction_from_claude_stream() {
    // Test the progress parser with Claude Code's assistant/tool_use events
    let assistant_json = r#"{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"npm install react"}}]}}"#;
    let progress = extract_progress_from_json(assistant_json);
    assert!(
        progress.is_some(),
        "Should extract progress from assistant tool_use event"
    );
    assert!(
        progress.unwrap().contains("npm install react"),
        "Should include the command"
    );

    let thinking_json = r#"{"type":"thinking"}"#;
    let progress = extract_progress_from_json(thinking_json);
    assert_eq!(progress, Some("💭 Thinking...".to_string()));
}

// -----------------------------------------------------------------------
// Multi-line stderr output test
// -----------------------------------------------------------------------

/// A schema-held agent's report reads as its text wherever its output is
/// used (the run's record, a check, a synchronous result), never as JSON.
#[test]
fn a_report_reads_as_its_text_not_its_json() {
    let report = r#"{"summary":"Renamed the button to Send.","changed_files":["form.html"],"checks":[],"open_questions":[],"needs_owner":false}"#;
    let shown = "Renamed the button to Send.\n\nChanged files:\n- form.html";

    // Claude Code's last stream line: the report as `structured_output`
    // and as the JSON text of `result`.
    let claude = format!(
        "{{\"type\":\"system\"}}\n{}",
        serde_json::json!({"type": "result", "result": report, "structured_output": serde_json::from_str::<serde_json::Value>(report).unwrap()})
    );
    assert_eq!(extract_meaningful_output(&claude, 4_000), shown);

    // The report only in `structured_output`, with an empty `result`.
    let empty_result = serde_json::json!({"type": "result", "result": "", "structured_output": serde_json::from_str::<serde_json::Value>(report).unwrap()}).to_string();
    assert_eq!(extract_meaningful_output(&empty_result, 4_000), shown);

    // Codex's final agent message whose text is the report's JSON.
    let codex = serde_json::json!({"type": "item.completed", "item": {"type": "agent_message", "text": report}}).to_string();
    assert_eq!(extract_meaningful_output(&codex, 4_000), shown);

    // Prose stays as it is.
    let prose = serde_json::json!({"type": "result", "result": "Added a test."}).to_string();
    assert_eq!(extract_meaningful_output(&prose, 4_000), "Added a test.");
}

#[test]
fn extract_terminal_commands_covers_claude_codex_and_acp_events() {
    let claude = serde_json::json!({
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "sqlite3 aidaemon.db .tables"}}]}
    })
    .to_string();
    assert_eq!(
        extract_terminal_commands_from_json(&claude),
        vec!["sqlite3 aidaemon.db .tables"]
    );

    let codex = serde_json::json!({
        "type": "item.started",
        "item": {"id": "cmd-1", "type": "command_execution", "command": "sqlite3 ~/.aidaemon/aidaemon.db 'SELECT 1'"}
    })
    .to_string();
    assert_eq!(
        extract_terminal_commands_from_json(&codex),
        vec!["sqlite3 ~/.aidaemon/aidaemon.db 'SELECT 1'"]
    );

    let acp = serde_json::json!({
        "type": "tool_call",
        "kind": "execute",
        "command": "rm -rf ~/.aidaemon/aidaemon.db"
    })
    .to_string();
    assert_eq!(
        extract_terminal_commands_from_json(&acp),
        vec!["rm -rf ~/.aidaemon/aidaemon.db"]
    );
}
