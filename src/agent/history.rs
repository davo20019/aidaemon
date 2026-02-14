use super::*;

impl Agent {
    pub(super) async fn append_message_canonical(&self, msg: &Message) -> anyhow::Result<()> {
        self.state.append_message(msg).await
    }

    pub(super) async fn append_user_message_with_event(
        &self,
        emitter: &crate::events::EventEmitter,
        msg: &Message,
        has_attachments: bool,
    ) -> anyhow::Result<()> {
        emitter
            .emit(
                EventType::UserMessage,
                UserMessageData {
                    content: msg.content.clone().unwrap_or_default(),
                    message_id: Some(msg.id.clone()),
                    has_attachments,
                },
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
