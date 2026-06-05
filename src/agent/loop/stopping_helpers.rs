use super::*;
use crate::utils::truncate_with_note;

pub(super) fn send_file_completion_reply() -> &'static str {
    "I've sent the requested file. If you want any changes or another file, tell me exactly what to send."
}

pub(super) async fn latest_non_system_tool_result(
    agent: &Agent,
    session_id: &str,
    max_chars: usize,
) -> Option<(String, String)> {
    let history = match tokio::time::timeout(
        Duration::from_secs(5),
        agent.state.get_history(session_id, 80),
    )
    .await
    {
        Ok(Ok(history)) => history,
        Ok(Err(_)) => return None,
        Err(_) => {
            warn!(
                session_id,
                "Timed out while loading history for stall output excerpt"
            );
            return None;
        }
    };

    // Low-information tool names whose output is rarely useful as a user-facing
    // summary (e.g., "File written to /path, 200 bytes").  We still fall back to
    // them if nothing better is available.
    const LOW_INFO_TOOLS: &[&str] = &[
        "write_file",
        "edit_file",
        "manage_memories",
        "manage_people",
        "remember_fact",
        "check_environment", // diagnostic: lists installed tools, never a task result
        "cli_agent",         // sub-agent raw output is file metadata, never user-facing
    ];

    let clean_tool_content = |msg: &crate::traits::Message| -> Option<(String, String)> {
        if msg.role != "tool" {
            return None;
        }
        let cleaned = msg.primary_content()?;
        let cleaned = cleaned.trim();
        if cleaned.is_empty() {
            return None;
        }
        Some((
            msg.tool_name.clone().unwrap_or_default(),
            truncate_with_note(cleaned, max_chars),
        ))
    };

    // Only return output from informative tools (terminal, search, etc.).
    // State-changing tools (remember_fact, manage_memories, write_file, etc.)
    // produce confirmations ("Remembered: ...", "Forgotten: ...") that are
    // self-documenting.  Wrapping them with "Here is the latest tool output:"
    // creates confusing debugging-style messages.  When only low-info tools
    // ran, return None so the LLM's natural response passes through instead.
    //
    // IMPORTANT: Stop at the first `user` message boundary to avoid leaking
    // tool results from previous interactions into the current response.
    let mut hit_user_boundary = false;
    for msg in history.iter().rev() {
        if msg.role == "user" {
            hit_user_boundary = true;
        }
        if hit_user_boundary && msg.role == "tool" {
            // This tool result is from a previous interaction — stop.
            break;
        }
        let tool_name = msg.tool_name.as_deref().unwrap_or("");
        if LOW_INFO_TOOLS.contains(&tool_name) {
            continue;
        }
        if let Some(result) = clean_tool_content(msg) {
            return Some(result);
        }
    }
    None
}

pub(super) async fn latest_non_system_tool_output_excerpt(
    agent: &Agent,
    session_id: &str,
    max_chars: usize,
) -> Option<String> {
    latest_non_system_tool_result(agent, session_id, max_chars)
        .await
        .map(|(_, content)| content)
}
