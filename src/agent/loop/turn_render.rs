//! Pillar B: pure per-turn rendering. Spec §Rendering.
//! No timestamps, no map-iteration order, no env-dependent formatting —
//! enforced by golden tests + the debug re-render assertion (Task 5).
//!
//! `render_turn(turn_messages, mode, renderer_version) -> Vec<Value>` is the
//! single pure renderer. **Current** mode = full append-only conversion (the
//! same `&Message → Value` logic that `message_build_phase.rs` performs
//! inline, including the orphan-`tool_calls` filter and the
//! `tool_call_id`/`name` mapping). **Archived** mode = the single permanent
//! survivorship form (user text full; last-substantive assistant truncated;
//! tool results summarized; identity-critical verbatim; terminal-state
//! placeholder when no substantive assistant remains).

// The public renderer and its helpers are consumed by the message-build
// integration in Task 7 (Pillar B). Until then they have no in-crate caller
// outside this module's own golden tests, so silence dead-code lints.
#![allow(dead_code)]

use super::recall_guardrails::text_relates_to_critical_identity;
use super::sliding_window::summarize_tool_result;
// `Message`, `ToolCall`, `json`, `Value`, and `MAX_OLD_ASSISTANT_CONTENT_CHARS`
// all live in the `agent` module scope.
use super::*;
use crate::config::VisionConfig;
use crate::events::TerminalState;

/// Bump when the rendering ALGORITHM changes; invalidates all cached renders.
pub(crate) const RENDERER_VERSION: u32 = 2;

#[derive(Clone, Debug)]
pub(crate) struct RenderOptions {
    pub vision: VisionConfig,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            vision: VisionConfig {
                enabled: true,
                max_image_bytes: 4 * 1_048_576,
                mime_types: vec![
                    "image/jpeg".to_string(),
                    "image/png".to_string(),
                    "image/gif".to_string(),
                    "image/webp".to_string(),
                ],
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenderMode {
    Current,
    Archived { terminal_state: TerminalState },
}

/// Pure entry point. Dispatches to the per-mode renderer.
pub(crate) fn render_turn(
    turn_messages: &[Message],
    mode: RenderMode,
    _version: u32,
    options: &RenderOptions,
) -> Vec<Value> {
    match mode {
        RenderMode::Current => render_current(turn_messages, options),
        RenderMode::Archived { terminal_state } => {
            render_archived(turn_messages, terminal_state, options)
        }
    }
}

/// Shared learned-helplessness / budget-exhaustion failure-pattern predicate.
///
/// BOTH `render_current` (drops matching assistant text) and `render_archived`
/// (excludes from winning-assistant selection, uses the terminal placeholder)
/// call this so the lists cannot drift. This is the list previously inlined in
/// `message_build_phase.rs`'s `&Message → Value` conversion.
pub(crate) fn is_failure_boilerplate(content: &str) -> bool {
    let t = content.trim_start();
    t.starts_with("I wasn't able to process that request.")
        || t.starts_with("I wasn't able to complete this task.")
        || t.starts_with("I made some progress but wasn't able to fully complete")
        || t.starts_with("I seem to be stuck on this task.")
        || t.starts_with("I've reached my processing limit")
        || t.starts_with("This goal hit its daily processing budget")
        || t.starts_with("This scheduled goal hit its daily processing budget")
        || t.starts_with("This scheduled run hit its per-run processing budget")
        || t.starts_with("I sent the requested file(s), but ran into issues")
        || t.starts_with("I completed the main deliverable but wasn't able to finish")
}

/// Char-safe truncation to the old-assistant cap. Matches the legacy inline
/// behaviour: take the first N *characters* (never byte-slicing) and append an
/// ellipsis only when content actually exceeded the cap.
fn truncate_old_assistant(content: &str) -> String {
    if content.chars().count() > MAX_OLD_ASSISTANT_CONTENT_CHARS {
        let truncated: String = content
            .chars()
            .take(MAX_OLD_ASSISTANT_CONTENT_CHARS)
            .collect();
        format!("{}…", truncated)
    } else {
        content.to_string()
    }
}

/// Set of `tool_call_id`s that have a matching `tool` result in this turn.
/// Used to strip orphan `tool_calls` (whose result is absent) so the payload
/// stays provider-valid.
fn tool_result_ids(turn_messages: &[Message]) -> std::collections::HashSet<&str> {
    turn_messages
        .iter()
        .filter(|m| m.role == "tool" && m.tool_name.as_ref().is_some_and(|n| !n.is_empty()))
        .filter_map(|m| m.tool_call_id.as_deref())
        .collect()
}

/// Convert an assistant message's `tool_calls_json` into OpenAI wire-format
/// `tool_calls`, dropping any whose result is missing from this turn. Returns
/// the filtered array (possibly empty).
fn wire_tool_calls(tc_json: &str, result_ids: &std::collections::HashSet<&str>) -> Vec<Value> {
    let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) else {
        return Vec::new();
    };
    tcs.iter()
        .filter(|tc| result_ids.contains(tc.id.as_str()))
        .map(|tc| {
            let mut val = json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments
                }
            });
            if let Some(ref extra) = tc.extra_content {
                val["extra_content"] = extra.clone();
            }
            val
        })
        .collect()
}

/// Attach `name` / `tool_call_id` fields when present.
fn attach_tool_routing(obj: &mut Value, m: &Message) {
    if let Some(name) = &m.tool_name {
        if !name.is_empty() {
            obj["name"] = json!(name);
        }
    }
    if let Some(tcid) = &m.tool_call_id {
        obj["tool_call_id"] = json!(tcid);
    }
}

/// **Current** mode: append-only, full content. Single source of truth for the
/// `&Message → Value` conversion (Task 7 deletes the inline copy in
/// `message_build_phase.rs` and routes through here).
fn render_current(turn_messages: &[Message], options: &RenderOptions) -> Vec<Value> {
    let result_ids = tool_result_ids(turn_messages);
    let mut vision_skipped = false;

    let mut rendered: Vec<Value> = turn_messages
        .iter()
        // Skip tool results with empty/missing tool_name.
        .filter(|m| !(m.role == "tool" && m.tool_name.as_ref().is_none_or(|n| n.is_empty())))
        .filter_map(|m| {
            // Drop learned-helplessness / budget-exhaustion boilerplate so the
            // model never reads its own prior "I failed" text and gives up.
            if m.role == "assistant"
                && m.tool_calls_json.is_none()
                && m.content.as_deref().is_some_and(is_failure_boilerplate)
            {
                return None;
            }

            let content = if m.role == "user" && !m.attachments.is_empty() {
                let text = m.content.as_deref().unwrap_or("");
                let built = crate::agent::vision::build_multimodal_content(
                    text,
                    &m.attachments,
                    RenderMode::Current,
                    &options.vision,
                );
                if built.vision_skipped {
                    vision_skipped = true;
                }
                built.content
            } else {
                json!(m.content)
            };

            let mut obj = json!({
                "role": m.role,
                "content": content,
            });

            if let Some(tc_json) = &m.tool_calls_json {
                let filtered = wire_tool_calls(tc_json, &result_ids);
                if !filtered.is_empty() {
                    obj["tool_calls"] = json!(filtered);
                    if m.content.is_none() {
                        obj["content"] = Value::Null;
                    }
                } else if m.content.is_none()
                    || m.content.as_deref().is_some_and(|c| c.trim().is_empty())
                {
                    // All tool_calls orphaned and no text content — replace with
                    // a placeholder to avoid a dangling user message.
                    obj["content"] = json!("[Action completed]");
                }
            }

            attach_tool_routing(&mut obj, m);
            Some(obj)
        })
        .collect();

    if vision_skipped {
        rendered.insert(
            0,
            json!({
                "role": "system",
                "content": crate::agent::vision::VISION_SKIPPED_SYSTEM_HINT,
            }),
        );
    }

    rendered
}

/// **Archived** mode: single pass IN ORDER (chronological, in-place
/// transforms; NEVER regrouped into user/assistant/tools). The turn's message
/// order is preserved so assistant `tool_calls` keep immediately preceding
/// their `tool` results.
fn render_archived(
    turn_messages: &[Message],
    terminal_state: TerminalState,
    _options: &RenderOptions,
) -> Vec<Value> {
    let result_ids = tool_result_ids(turn_messages);

    // Pre-scan: index of the LAST substantive assistant record — non-empty
    // trimmed content that is NOT failure boilerplate and NOT identity-critical
    // (identity messages are handled by the verbatim override, not as the
    // "winning" assistant).
    let last_substantive_assistant = turn_messages
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            m.role == "assistant"
                && m.content.as_deref().is_some_and(|c| {
                    !c.trim().is_empty()
                        && !is_failure_boilerplate(c)
                        && !text_relates_to_critical_identity(c)
                })
        })
        .map(|(i, _)| i)
        .next_back();

    // Tool-step count feeds the terminal-state placeholder.
    let tool_step_count = turn_messages
        .iter()
        .filter(|m| m.role == "tool" && m.tool_name.as_ref().is_some_and(|n| !n.is_empty()))
        .count();

    // Position of the turn's last assistant/tool record — where the synthetic
    // placeholder is emitted (end of turn) so order stays chronological.
    let last_assistant_or_tool = turn_messages
        .iter()
        .rposition(|m| m.role == "assistant" || m.role == "tool");

    let need_placeholder = last_substantive_assistant.is_none();

    let mut out: Vec<Value> = Vec::with_capacity(turn_messages.len());

    for (idx, m) in turn_messages.iter().enumerate() {
        // Identity override (precedence): emit verbatim in original position,
        // exempt from truncation/drop/summarization — but exactly ONCE (this
        // REPLACES the normal transform for that message).
        if m.content
            .as_deref()
            .is_some_and(text_relates_to_critical_identity)
        {
            let mut obj = json!({
                "role": m.role,
                "content": m.content,
            });
            if let Some(tc_json) = &m.tool_calls_json {
                let filtered = wire_tool_calls(tc_json, &result_ids);
                if !filtered.is_empty() {
                    obj["tool_calls"] = json!(filtered);
                }
            }
            attach_tool_routing(&mut obj, m);
            out.push(obj);
            maybe_emit_placeholder(
                &mut out,
                idx,
                need_placeholder,
                last_assistant_or_tool,
                terminal_state,
                tool_step_count,
            );
            continue;
        }

        match m.role.as_str() {
            // 1. user → verbatim full. (A turn may have NO user message — do
            //    not synthesize one.)
            "user" => {
                out.push(json!({ "role": "user", "content": m.content }));
            }

            // 2. assistant.
            "assistant" => {
                if Some(idx) == last_substantive_assistant {
                    // Winning assistant: truncated content, tool_calls retained.
                    let truncated = m
                        .content
                        .as_deref()
                        .map(truncate_old_assistant)
                        .unwrap_or_default();
                    let mut obj = json!({
                        "role": "assistant",
                        "content": truncated,
                    });
                    if let Some(tc_json) = &m.tool_calls_json {
                        let filtered = wire_tool_calls(tc_json, &result_ids);
                        if !filtered.is_empty() {
                            obj["tool_calls"] = json!(filtered);
                        }
                    }
                    out.push(obj);
                } else if m.content.as_deref().is_some_and(is_failure_boilerplate) {
                    // Failure boilerplate: do not emit text, do not treat as a
                    // winning response. If it carries tool_calls, retain those.
                    emit_tool_call_only_assistant(&mut out, m, &result_ids);
                } else if m.tool_calls_json.is_some() {
                    // Losing (superseded) assistant with tool_calls: keep the
                    // calls so the call/result pairing survives, drop the text.
                    emit_tool_call_only_assistant(&mut out, m, &result_ids);
                }
                // else: empty / superseded plain assistant → drop.
            }

            // 3. tool result → tool-role deterministic summary.
            "tool" if m.tool_name.as_ref().is_some_and(|n| !n.is_empty()) => {
                let args_json = m
                    .tool_call_id
                    .as_deref()
                    .and_then(|cid| tool_args_for_call(turn_messages, cid))
                    .unwrap_or_default();
                let tool_name = m.tool_name.as_deref().unwrap_or("unknown");
                let result = m.content.as_deref().unwrap_or("");
                let summary = summarize_tool_result(tool_name, &args_json, result);
                let mut obj = json!({ "role": "tool", "content": summary });
                attach_tool_routing(&mut obj, m);
                out.push(obj);
            }

            _ => {}
        }

        maybe_emit_placeholder(
            &mut out,
            idx,
            need_placeholder,
            last_assistant_or_tool,
            terminal_state,
            tool_step_count,
        );
    }

    out
}

/// Emit an assistant message that retains only its (orphan-filtered)
/// `tool_calls`, dropping the (losing) text content — but only if some
/// `tool_calls` survive the orphan filter; otherwise nothing is emitted.
fn emit_tool_call_only_assistant(
    out: &mut Vec<Value>,
    m: &Message,
    result_ids: &std::collections::HashSet<&str>,
) {
    if let Some(tc_json) = &m.tool_calls_json {
        let filtered = wire_tool_calls(tc_json, result_ids);
        if !filtered.is_empty() {
            out.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": filtered,
            }));
        }
    }
}

/// Emit the single synthetic terminal-state placeholder at the position of the
/// turn's last assistant/tool record, when no substantive assistant remains.
fn maybe_emit_placeholder(
    out: &mut Vec<Value>,
    idx: usize,
    need_placeholder: bool,
    last_assistant_or_tool: Option<usize>,
    terminal_state: TerminalState,
    tool_step_count: usize,
) {
    if need_placeholder && last_assistant_or_tool == Some(idx) {
        out.push(json!({
            "role": "assistant",
            "content": terminal_state.placeholder(tool_step_count),
        }));
    }
}

/// Look up the arguments JSON for a tool result's `tool_call_id` by scanning
/// the turn's assistant `tool_calls`.
fn tool_args_for_call(turn_messages: &[Message], call_id: &str) -> Option<String> {
    for m in turn_messages.iter() {
        if m.role == "assistant" {
            if let Some(tc_json) = &m.tool_calls_json {
                if let Ok(tcs) = serde_json::from_str::<Vec<ToolCall>>(tc_json) {
                    for tc in &tcs {
                        if tc.id == call_id {
                            return Some(tc.arguments.clone());
                        }
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(c: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Some(c.to_string()),
            ..Message::runtime_defaults()
        }
    }

    fn assistant(c: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Some(c.to_string()),
            ..Message::runtime_defaults()
        }
    }

    fn assistant_empty_with_tool_call() -> Message {
        Message {
            role: "assistant".to_string(),
            content: None,
            tool_calls_json: Some(
                r#"[{"id":"c1","name":"terminal","arguments":"{}"}]"#.to_string(),
            ),
            ..Message::runtime_defaults()
        }
    }

    fn tool(name: &str, call_id: &str, result: &str) -> Message {
        // Real terminal output carries an `exit_code: N` line; the golden tests
        // use the shorthand `"exit 0"`. Normalize that shorthand to the
        // canonical form so `summarize_tool_result` reproduces `-> exit N`
        // without touching the production summarizer (which keys off
        // `exit_code:`). Non-terminal results pass through verbatim.
        let content = if name == "terminal" {
            if let Some(code) = result.strip_prefix("exit ") {
                format!("exit_code: {code}")
            } else {
                result.to_string()
            }
        } else {
            result.to_string()
        };
        Message {
            role: "tool".to_string(),
            content: Some(content),
            tool_call_id: Some(call_id.to_string()),
            tool_name: Some(name.to_string()),
            ..Message::runtime_defaults()
        }
    }

    #[test]
    fn archived_keeps_user_full_and_last_nonempty_assistant_truncated() {
        let turn = vec![
            user("please do the long thing with lots of detail ...full text..."),
            assistant_empty_with_tool_call(),
            tool("terminal", "c1", "exit 0"),
            assistant(&"X".repeat(500)), // last non-empty assistant — wins, truncated
            assistant(""),               // later empty — loses
        ];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let joined = serde_json::to_string(&out).unwrap();
        assert!(
            joined.contains("...full text..."),
            "user text survives in full"
        );
        assert!(
            joined.contains(&"X".repeat(MAX_OLD_ASSISTANT_CONTENT_CHARS)),
            "assistant truncated to cap"
        );
        assert!(!joined.contains(&"X".repeat(MAX_OLD_ASSISTANT_CONTENT_CHARS + 1)));
        assert!(
            joined.contains("terminal: -> exit 0"),
            "tool result summarized deterministically"
        );
    }

    #[test]
    fn archived_no_text_reply_uses_terminal_state_placeholder() {
        let turn = vec![
            user("run it"),
            assistant_empty_with_tool_call(),
            tool("terminal", "c1", "exit 1"),
        ];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Failed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let joined = serde_json::to_string(&out).unwrap();
        assert!(joined.contains("[failed: 1 tool steps, no text reply]"));
    }

    #[test]
    fn archived_interrupted_turn_renders_interrupted_placeholder() {
        let turn = vec![user("hello"), tool("terminal", "c1", "exit 0")];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Interrupted,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        assert!(serde_json::to_string(&out)
            .unwrap()
            .contains("[task interrupted]"));
    }

    #[test]
    fn archived_preserves_identity_critical_verbatim() {
        let turn = vec![user("my name is David Loor"), assistant("noted")];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        assert!(serde_json::to_string(&out)
            .unwrap()
            .contains("my name is David Loor"));
    }

    #[test]
    fn render_is_deterministic() {
        let turn = vec![
            user("hi"),
            assistant("there"),
            tool("read_file", "c1", "12 lines"),
        ];
        let a = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let b = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn current_mode_is_append_only_full() {
        let turn = vec![user("hi"), assistant("there")];
        let out = render_turn(
            &turn,
            RenderMode::Current,
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        // Current keeps full content, both messages, in order.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["content"], "hi");
        assert_eq!(out[1]["content"], "there");
    }

    #[test]
    fn archived_output_order_is_chronological() {
        let turn = vec![
            user("u"),
            assistant_empty_with_tool_call(),
            tool("terminal", "c1", "exit 0"),
            assistant("final"),
        ];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let roles: Vec<&str> = out.iter().map(|m| m["role"].as_str().unwrap()).collect();
        // user → assistant(tool_calls) → tool → assistant(final): order
        // preserved, NOT regrouped.
        assert_eq!(roles, vec!["user", "assistant", "tool", "assistant"]);
    }

    #[test]
    fn archived_no_user_message_turn_renders_without_synthesizing_user() {
        // Scheduled/background turn with no user_message.
        let turn = vec![
            assistant_empty_with_tool_call(),
            tool("terminal", "c1", "exit 0"),
        ];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        assert!(
            out.iter().all(|m| m["role"] != "user"),
            "no synthetic user message"
        );
        // tool_step_count = 1 feeds the placeholder if no assistant text exists.
        assert!(
            serde_json::to_string(&out)
                .unwrap()
                .contains("1 tool steps")
                || out.iter().any(|m| m["role"] == "assistant")
        );
    }

    #[test]
    fn archived_identity_message_emitted_once_not_duplicated() {
        // Identity-critical assistant that is NOT the last non-empty assistant.
        let turn = vec![
            user("hi"),
            assistant("my name is David Loor"),
            assistant("ok done"),
        ];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let joined = serde_json::to_string(&out).unwrap();
        assert!(
            joined.contains("my name is David Loor"),
            "identity survives verbatim"
        );
        assert_eq!(
            joined.matches("my name is David Loor").count(),
            1,
            "emitted once, not duplicated"
        );
    }

    #[test]
    fn current_mode_drops_learned_helplessness_but_archived_uses_placeholder() {
        let turn = vec![
            user("do it"),
            assistant_empty_with_tool_call(),
            tool("terminal", "c1", "exit 1"),
            assistant("I wasn't able to complete this task."),
        ];
        let cur = render_turn(
            &turn,
            RenderMode::Current,
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        assert!(
            !serde_json::to_string(&cur)
                .unwrap()
                .contains("I wasn't able to complete"),
            "learned-helplessness dropped in Current"
        );
        let arch = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Failed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let aj = serde_json::to_string(&arch).unwrap();
        assert!(aj.contains("[failed: 1 tool steps, no text reply]"));
        assert!(
            !aj.contains("I wasn't able to complete"),
            "Archived failure boilerplate is replaced by the deterministic terminal placeholder"
        );
    }

    #[test]
    fn current_mode_encodes_image_attachments_as_multimodal_array() {
        use crate::traits::MessageAttachment;
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();
        let stub = format!(
            "[File received: photo.png (0 KB, image/png)\nSaved to: {}]",
            file.path().display()
        );
        let turn = vec![Message {
            role: "user".to_string(),
            content: Some(stub),
            attachments: vec![MessageAttachment {
                local_path: file.path().to_string_lossy().into_owned(),
                filename: "photo.png".to_string(),
                mime_type: "image/png".to_string(),
                size_bytes: 8,
            }],
            ..Message::runtime_defaults()
        }];
        let out = render_turn(
            &turn,
            RenderMode::Current,
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        let user_msg = out
            .iter()
            .find(|m| m["role"] == "user")
            .expect("user message");
        let blocks = user_msg["content"].as_array().expect("multimodal array");
        assert!(blocks.iter().any(|b| b["type"] == "text"));
        assert!(blocks.iter().any(|b| b["type"] == "image_url"));
    }

    #[test]
    fn archived_mode_keeps_image_attachments_as_text_stub() {
        let turn = vec![Message {
            role: "user".to_string(),
            content: Some(
                "[File received: photo.png (1 KB, image/png)\nSaved to: /tmp/x.png]".to_string(),
            ),
            attachments: vec![crate::traits::MessageAttachment {
                local_path: "/tmp/x.png".to_string(),
                filename: "photo.png".to_string(),
                mime_type: "image/png".to_string(),
                size_bytes: 1024,
            }],
            ..Message::runtime_defaults()
        }];
        let out = render_turn(
            &turn,
            RenderMode::Archived {
                terminal_state: TerminalState::Completed,
            },
            RENDERER_VERSION,
            &RenderOptions::default(),
        );
        assert!(out[0]["content"].is_string());
    }
}
