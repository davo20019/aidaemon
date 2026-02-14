use super::*;

impl Agent {
    pub(super) async fn build_resume_checkpoint(
        &self,
        session_id: &str,
    ) -> anyhow::Result<Option<ResumeCheckpoint>> {
        let Some(active_task_event) = self.event_store.get_active_task(session_id).await? else {
            return Ok(None);
        };
        let Some(task_id) = active_task_event.task_id.clone() else {
            return Ok(None);
        };

        let start_data = active_task_event.parse_data::<TaskStartData>().ok();
        let description = start_data
            .as_ref()
            .map(|d| d.description.clone())
            .unwrap_or_else(|| "in-progress task".to_string());
        let original_user_message = start_data.and_then(|d| d.user_message);

        let events = self
            .event_store
            .query_task_events_for_session(session_id, &task_id)
            .await
            .unwrap_or_default();

        let mut last_iteration: u32 = 0;
        let mut tool_results_count: u32 = 0;
        let mut pending_tool_calls: HashSet<String> = HashSet::new();
        let mut last_assistant_summary: Option<String> = None;
        let mut last_tool_summary: Option<String> = None;
        let mut last_error: Option<String> = None;

        for event in &events {
            match event.event_type {
                EventType::ThinkingStart => {
                    if let Ok(data) = event.parse_data::<ThinkingStartData>() {
                        last_iteration = last_iteration.max(data.iteration);
                    }
                }
                EventType::AssistantResponse => {
                    if let Ok(data) = event.parse_data::<AssistantResponseData>() {
                        if let Some(calls) = data.tool_calls.as_ref() {
                            for call in calls {
                                pending_tool_calls.insert(call.id.clone());
                            }
                        }
                        if let Some(content) = data.content.as_deref() {
                            let trimmed = content.trim();
                            if !trimmed.is_empty() {
                                last_assistant_summary = Some(truncate_for_resume(trimmed, 180));
                            }
                        }
                    }
                }
                EventType::ToolResult => {
                    if let Ok(data) = event.parse_data::<ToolResultData>() {
                        tool_results_count = tool_results_count.saturating_add(1);
                        pending_tool_calls.remove(&data.tool_call_id);
                        let detail = if data.success {
                            data.result
                        } else {
                            data.error.unwrap_or(data.result)
                        };
                        let detail = detail.trim();
                        if !detail.is_empty() {
                            last_tool_summary = Some(truncate_for_resume(detail, 180));
                        }
                    }
                }
                EventType::Error => {
                    if let Ok(data) = event.parse_data::<ErrorData>() {
                        last_error = Some(truncate_for_resume(&data.message, 180));
                    }
                }
                _ => {}
            }
        }

        let elapsed_secs = (Utc::now() - active_task_event.created_at)
            .num_seconds()
            .max(0) as u64;
        let mut pending_tool_call_ids: Vec<String> = pending_tool_calls.into_iter().collect();
        pending_tool_call_ids.sort();

        Ok(Some(ResumeCheckpoint {
            task_id,
            description,
            original_user_message,
            elapsed_secs,
            last_iteration,
            tool_results_count,
            pending_tool_call_ids,
            last_assistant_summary,
            last_tool_summary,
            last_error,
        }))
    }

    pub(super) async fn mark_task_interrupted_for_resume(
        &self,
        session_id: &str,
        checkpoint: &ResumeCheckpoint,
        resumed_task_id: &str,
    ) {
        // Best-effort: if task already has task_end, skip.
        let already_ended = self
            .event_store
            .query_task_events_for_session(session_id, &checkpoint.task_id)
            .await
            .ok()
            .is_some_and(|events| events.iter().any(|e| e.event_type == EventType::TaskEnd));
        if already_ended {
            return;
        }

        let resume_emitter =
            crate::events::EventEmitter::new(self.event_store.clone(), session_id.to_string())
                .with_task_id(checkpoint.task_id.clone());
        let error = format!(
            "Agent process interrupted before completion. Resumed in task {}.",
            resumed_task_id
        );
        let _ = resume_emitter
            .emit(
                EventType::TaskEnd,
                TaskEndData {
                    task_id: checkpoint.task_id.clone(),
                    status: TaskStatus::Failed,
                    duration_secs: checkpoint.elapsed_secs,
                    iterations: checkpoint.last_iteration,
                    tool_calls_count: checkpoint.tool_results_count,
                    error: Some(error),
                    summary: Some("Recovered from checkpoint after interruption".to_string()),
                },
            )
            .await;
    }
}
