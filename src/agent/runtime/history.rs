use super::*;

#[derive(Debug, Clone, Default)]
pub(super) struct GoalPromptContext {
    pub goal_user_text: String,
    pub recent_messages: Vec<Value>,
    pub project_hints: Vec<String>,
}

const GOAL_CONTEXT_RECENT_MESSAGES_LIMIT: usize = 6;
const GOAL_CONTEXT_HINT_HISTORY_LIMIT: usize = 30;
const GOAL_CONTEXT_MAX_PROJECT_HINTS: usize = 8;

fn find_previous_turns(
    history: &[Message],
    current_user_text: &str,
) -> (Option<String>, Option<String>) {
    // History includes the current user message (already persisted). Walk backwards to find:
    // - previous assistant message (optional; usually a clarifying question)
    // - previous user message (the original request)
    let mut saw_current_user = false;
    let mut prev_assistant: Option<String> = None;
    let mut prev_user: Option<String> = None;
    for msg in history.iter().rev() {
        match msg.role.as_str() {
            "user" => {
                if !saw_current_user {
                    saw_current_user = true;
                    continue;
                }
                if let Some(content) = msg.content.as_deref() {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        prev_user = Some(trimmed.to_string());
                        break;
                    }
                }
            }
            "assistant" => {
                if saw_current_user && prev_assistant.is_none() {
                    if let Some(content) = msg.content.as_deref() {
                        let trimmed = content.trim();
                        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case(current_user_text) {
                            prev_assistant = Some(trimmed.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    (prev_assistant, prev_user)
}

fn assistant_message_looks_like_clarifying_question(message: &str) -> bool {
    let trimmed = message.trim();
    if !trimmed.contains('?') {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let clarifying_markers = [
        "which",
        "what",
        "how",
        "do you want",
        "would you like",
        "should i",
        "can you clarify",
        "any specific",
        "do you prefer",
        "prefer",
        "what style",
        "what elements",
    ];
    clarifying_markers.iter().any(|m| lower.contains(m))
}

fn looks_like_explicit_task_switch(lower_text: &str) -> bool {
    lower_text.starts_with("new task")
        || lower_text.starts_with("different task")
        || lower_text.starts_with("instead ")
        || lower_text.starts_with("forget that")
        || lower_text.starts_with("ignore that")
}

fn looks_like_style_followup(lower_text: &str) -> bool {
    let style_markers = [
        "do what you consider",
        "do what you think",
        "as you see fit",
        "you decide",
        "your call",
        "best judgment",
        "best judgement",
        "whatever you think",
    ];
    style_markers.iter().any(|m| lower_text.contains(m))
}

fn looks_like_followup_reply(current: &str, prev_assistant: Option<&str>) -> bool {
    let trimmed = current.trim();
    if trimmed.is_empty() {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    let is_short = trimmed.chars().count() <= 260;
    if !is_short || looks_like_explicit_task_switch(&lower) {
        return false;
    }

    let explicit_followup = lower.starts_with("also ")
        || lower.starts_with("and ")
        || lower.starts_with("plus ")
        || contains_keyword_as_words(&lower, "yes")
        || contains_keyword_as_words(&lower, "confirm")
        || contains_keyword_as_words(&lower, "go ahead")
        || contains_keyword_as_words(&lower, "do it")
        || contains_keyword_as_words(&lower, "sure")
        || contains_keyword_as_words(&lower, "ok")
        || contains_keyword_as_words(&lower, "okay")
        || contains_keyword_as_words(&lower, "sounds good")
        || contains_keyword_as_words(&lower, "just use")
        || looks_like_style_followup(&lower);

    if explicit_followup {
        return true;
    }

    prev_assistant.is_some_and(|prev| {
        assistant_message_looks_like_clarifying_question(prev)
            && !trimmed.trim_end().ends_with('?')
            && !looks_like_explicit_task_switch(&lower)
    })
}

fn trim_assistant_context_content(content: &str) -> String {
    let trimmed = content.trim();
    if let Some((before, _)) = trimmed.split_once(INTENT_GATE_MARKER) {
        before.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

fn extract_recent_parent_messages(history: &[Message], max_messages: usize) -> Vec<Value> {
    let mut rows: Vec<Value> = history
        .iter()
        .filter_map(|msg| {
            if msg.role != "user" && msg.role != "assistant" {
                return None;
            }
            let raw = msg.content.as_deref()?.trim();
            if raw.is_empty() {
                return None;
            }
            let content = if msg.role == "assistant" {
                trim_assistant_context_content(raw)
            } else {
                raw.to_string()
            };
            if content.trim().is_empty() {
                return None;
            }
            Some(json!({
                "role": msg.role,
                "content": truncate_for_resume(content.trim(), 500),
            }))
        })
        .collect();
    if rows.len() > max_messages {
        rows = rows.split_off(rows.len() - max_messages);
    }
    rows
}

fn is_generic_non_project_token(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "this"
            | "that"
            | "these"
            | "those"
            | "with"
            | "from"
            | "into"
            | "about"
            | "using"
            | "make"
            | "create"
            | "modern"
            | "frontend"
            | "backend"
            | "design"
            | "developer"
            | "senior"
            | "best"
            | "style"
            | "tailwind"
            | "react"
            | "html"
            | "css"
            | "javascript"
            | "typescript"
            | "project"
            | "workspace"
            | "directory"
            | "folder"
    )
}

fn is_likely_filename(token: &str) -> bool {
    let Some((name, ext)) = token.rsplit_once('.') else {
        return false;
    };
    !name.is_empty()
        && !ext.is_empty()
        && ext.len() <= 8
        && ext.chars().all(|c| c.is_ascii_alphanumeric())
}

fn is_common_path_segment(token: &str) -> bool {
    matches!(
        token,
        "users"
            | "user"
            | "home"
            | "workspace"
            | "workspaces"
            | "projects"
            | "repos"
            | "repo"
            | "src"
            | "apps"
            | "app"
            | "packages"
            | "package"
            | "code"
            | "tmp"
            | "var"
            | "usr"
            | "opt"
            | "local"
            | "dev"
            | "documents"
            | "downloads"
            | "desktop"
    )
}

fn normalize_project_component(raw: &str, allow_plain_names: bool) -> Option<String> {
    let token = raw
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
        .to_ascii_lowercase();
    if token.is_empty() || token.contains("://") {
        return None;
    }
    if token.len() < 3 {
        return None;
    }
    if !token.chars().any(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if is_generic_non_project_token(&token) {
        return None;
    }
    if is_likely_filename(&token) {
        return None;
    }

    if !allow_plain_names {
        let looks_project_like = token.contains("project")
            || token.contains('-')
            || token.contains('_')
            || token.starts_with("app")
            || token.ends_with("app");
        if !looks_project_like {
            return None;
        }
    }

    Some(token)
}

fn extract_project_hint_from_path_like_token(raw_token: &str) -> Option<String> {
    let trimmed = raw_token
        .trim_matches(|c: char| c.is_ascii_whitespace() || c == '`' || c == '\'' || c == '"')
        .trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'))
        .to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }

    let path_source = if let Some((_, after_scheme)) = trimmed.split_once("://") {
        let (_, path) = after_scheme.split_once('/')?;
        path.to_string()
    } else {
        trimmed
    };

    let mut parts: Vec<String> = Vec::new();
    for raw_part in path_source.split(['/', '\\']) {
        let part = raw_part
            .split(['?', '#'])
            .next()
            .unwrap_or("")
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.')
            .to_ascii_lowercase();
        if !part.is_empty() {
            parts.push(part);
        }
    }

    for part in parts.iter().rev() {
        if is_common_path_segment(part) || is_likely_filename(part) {
            continue;
        }
        if let Some(candidate) = normalize_project_component(part, true) {
            return Some(candidate);
        }
    }

    None
}

fn normalize_project_hint(raw: &str, path_like: bool) -> Option<String> {
    let uri_like = raw.contains("://");
    if path_like || uri_like {
        if let Some(path_hint) = extract_project_hint_from_path_like_token(raw) {
            return Some(path_hint);
        }
        if uri_like {
            return None;
        }
    }
    normalize_project_component(raw, false)
}

fn push_project_hint(hints: &mut Vec<String>, hint: String, max_hints: usize) {
    if hints.len() >= max_hints || hints.iter().any(|existing| existing == &hint) {
        return;
    }
    hints.push(hint);
}

fn extract_project_hints_from_text(
    text: &str,
    hints: &mut Vec<String>,
    max_hints: usize,
    path_only: bool,
) {
    for raw in text.split_whitespace() {
        if hints.len() >= max_hints {
            break;
        }
        let uri_like = raw.contains("://");
        let path_like = uri_like
            || raw.contains('/')
            || raw.contains('\\')
            || raw.starts_with("./")
            || raw.starts_with("../")
            || raw.starts_with("~/");
        if path_only && !path_like {
            continue;
        }
        if let Some(normalized) = normalize_project_hint(raw, path_like) {
            push_project_hint(hints, normalized, max_hints);
        }
    }
}

fn extract_project_hints_from_history(
    history: &[Message],
    current_user_text: &str,
    max_hints: usize,
) -> Vec<String> {
    let mut hints: Vec<String> = Vec::new();
    extract_project_hints_from_text(current_user_text, &mut hints, max_hints, false);

    for msg in history.iter().rev() {
        if hints.len() >= max_hints {
            break;
        }
        if let Some(content) = msg.content.as_deref() {
            match msg.role.as_str() {
                "user" | "assistant" => {
                    extract_project_hints_from_text(content, &mut hints, max_hints, false);
                }
                "tool" => {
                    // Tool payloads are noisy; only trust explicit paths/URIs there.
                    extract_project_hints_from_text(content, &mut hints, max_hints, true);
                }
                _ => {}
            }
        }
    }
    hints
}

impl Agent {
    pub(super) async fn build_goal_prompt_context_from_recent_history(
        &self,
        session_id: &str,
        user_text: &str,
    ) -> GoalPromptContext {
        let current = user_text.trim();
        if current.is_empty() {
            return GoalPromptContext::default();
        }

        let history = self
            .state
            .get_history(session_id, GOAL_CONTEXT_HINT_HISTORY_LIMIT)
            .await
            .unwrap_or_default();
        let (prev_assistant, prev_user) = find_previous_turns(&history, current);

        let goal_user_text = match prev_user {
            Some(prev_user_text)
                if looks_like_followup_reply(current, prev_assistant.as_deref())
                    && !prev_user_text.trim().eq_ignore_ascii_case(current) =>
            {
                let mut combined = String::new();
                combined.push_str("Original request:\n");
                combined.push_str(&truncate_for_resume(prev_user_text.trim(), 2000));
                if let Some(prev_assistant_text) = prev_assistant {
                    let trimmed = prev_assistant_text.trim();
                    if !trimmed.is_empty() && trimmed.contains('?') {
                        combined.push_str("\n\nAssistant asked:\n");
                        combined.push_str(&truncate_for_resume(trimmed, 800));
                    }
                }
                combined.push_str("\n\nFollow-up:\n");
                combined.push_str(&truncate_for_resume(current, 800));
                combined
            }
            _ => current.to_string(),
        };

        GoalPromptContext {
            goal_user_text,
            recent_messages: extract_recent_parent_messages(
                &history,
                GOAL_CONTEXT_RECENT_MESSAGES_LIMIT,
            ),
            project_hints: extract_project_hints_from_history(
                &history,
                current,
                GOAL_CONTEXT_MAX_PROJECT_HINTS,
            ),
        }
    }

    pub(super) async fn append_message_canonical(&self, msg: &Message) -> anyhow::Result<()> {
        self.state.append_message(msg).await
    }

    pub(super) async fn append_user_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        channel_ctx: &ChannelContext,
        has_attachments: bool,
    ) -> anyhow::Result<()> {
        emitter
            .emit(
                EventType::UserMessage,
                json!({
                    "content": msg.content.clone().unwrap_or_default(),
                    "message_id": msg.id.clone(),
                    "has_attachments": has_attachments,
                    // Provenance for downstream projections/consolidation.
                    "channel_visibility": channel_ctx.visibility.to_string(),
                    "channel_id": channel_ctx.channel_id.clone(),
                    "platform": channel_ctx.platform.clone(),
                    "sender_id": channel_ctx.sender_id.clone(),
                }),
            )
            .await?;
        self.append_message_canonical(msg).await?;
        Ok(())
    }

    pub(super) async fn append_assistant_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        model: &str,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) -> anyhow::Result<()> {
        let tool_calls = msg.tool_calls_json.as_ref().and_then(|raw| {
            serde_json::from_str::<Vec<ToolCall>>(raw)
                .ok()
                .map(|calls| {
                    calls
                        .into_iter()
                        .map(|tc| ToolCallInfo {
                            id: tc.id,
                            name: tc.name,
                            arguments: serde_json::from_str(&tc.arguments)
                                .unwrap_or(serde_json::json!({})),
                            extra_content: tc.extra_content,
                        })
                        .collect::<Vec<_>>()
                })
        });
        emitter
            .emit(
                EventType::AssistantResponse,
                AssistantResponseData {
                    message_id: Some(msg.id.clone()),
                    content: msg.content.clone(),
                    model: model.to_string(),
                    tool_calls,
                    input_tokens,
                    output_tokens,
                },
            )
            .await?;
        self.append_message_canonical(msg).await?;
        Ok(())
    }

    pub(super) async fn append_tool_message_with_result_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        success: bool,
        duration_ms: u64,
        error: Option<String>,
        task_id: Option<&str>,
    ) -> anyhow::Result<()> {
        emitter
            .emit(
                EventType::ToolResult,
                ToolResultData {
                    message_id: Some(msg.id.clone()),
                    tool_call_id: msg.tool_call_id.clone().unwrap_or_else(|| msg.id.clone()),
                    name: msg
                        .tool_name
                        .clone()
                        .unwrap_or_else(|| "system".to_string()),
                    result: msg.content.clone().unwrap_or_default(),
                    success,
                    duration_ms,
                    error,
                    task_id: task_id.map(str::to_string),
                },
            )
            .await?;
        self.append_message_canonical(msg).await?;
        Ok(())
    }

    pub(super) async fn load_initial_history(
        &self,
        session_id: &str,
        user_text: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        match self
            .event_store
            .get_conversation_history(session_id, limit)
            .await
        {
            Ok(history) if !history.is_empty() => {
                return Ok(history);
            }
            Ok(_) => {}
            Err(e) => {
                warn!(
                    session_id,
                    error = %e,
                    "Event history load failed; falling back to state context retrieval"
                );
            }
        }

        self.state.get_context(session_id, user_text, limit).await
    }

    pub(super) async fn load_recent_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<Message>> {
        match self
            .event_store
            .get_conversation_history(session_id, limit)
            .await
        {
            Ok(history) if !history.is_empty() => Ok(history),
            Ok(_) => self.state.get_history(session_id, limit).await,
            Err(e) => {
                warn!(
                    session_id,
                    error = %e,
                    "Event recent-history load failed; falling back to state history retrieval"
                );
                self.state.get_history(session_id, limit).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: role.to_string(),
            content: Some(content.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            embedding: None,
        }
    }

    #[test]
    fn followup_detects_answer_to_clarifying_question() {
        let followup =
            "You are a senior designer and frontend developer. Do what you consider best.";
        let prev = "Which elements do you want me to modernize?";
        assert!(looks_like_followup_reply(followup, Some(prev)));
    }

    #[test]
    fn followup_rejects_explicit_task_switch() {
        let followup = "New task: build a dashboard from scratch.";
        let prev = "Should I continue with this page or focus on another one?";
        assert!(!looks_like_followup_reply(followup, Some(prev)));
    }

    #[test]
    fn project_hint_extraction_finds_project_name() {
        let history = vec![
            msg(
                "user",
                "Please work in test-project and modernize index.html",
            ),
            msg("assistant", "Which sections should I prioritize?"),
            msg("user", "Do what you consider best."),
        ];
        let hints = extract_project_hints_from_history(&history, "Do what you consider best.", 6);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project in project hints, got {:?}",
            hints
        );
    }

    #[test]
    fn project_hint_extraction_handles_file_uri_paths() {
        let history = vec![msg(
            "assistant",
            "Opened file:///Users/testuser/projects/test-project/index.html",
        )];
        let hints = extract_project_hints_from_history(&history, "Do what you consider best.", 8);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project from file URI, got {:?}",
            hints
        );
        assert!(
            hints.iter().all(|h| h != "index.html"),
            "filename should not be treated as project hint: {:?}",
            hints
        );
    }

    #[test]
    fn project_hint_extraction_scans_tool_messages_path_only() {
        let history = vec![msg(
            "tool",
            "Using terminal: cd ~/projects/test-project && npm run build",
        )];
        let hints = extract_project_hints_from_history(&history, "make it modern", 8);
        assert!(
            hints.iter().any(|h| h == "test-project"),
            "expected test-project from tool output path, got {:?}",
            hints
        );
    }

    #[test]
    fn recent_parent_messages_strip_intent_gate_payload() {
        let history = vec![
            msg("user", "Build a site"),
            msg(
                "assistant",
                "Sure, I can help.\n[INTENT_GATE] {\"can_answer_now\":false}",
            ),
        ];
        let messages = extract_recent_parent_messages(&history, 6);
        assert_eq!(messages.len(), 2);
        let assistant_content = messages[1]
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(!assistant_content.contains("[INTENT_GATE]"));
        assert_eq!(assistant_content, "Sure, I can help.");
    }
}
