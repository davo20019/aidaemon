use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::events::{
    ApprovalDeniedData, AssistantResponseData, DecisionPointData, DecisionType, ErrorData,
    ErrorType, Event, EventStore, EventType, FailureCategory, RootCauseCandidate, TaskEndData,
    TaskStartData, ThinkingStartData, ToolCallData, ToolResultData,
};
use crate::tools::sanitize::redact_secrets;
use crate::traits::{ModelProvider, ProviderResponse, StateStore, Tool, ToolCapabilities};
use crate::utils::truncate_str;

pub struct DiagnoseTool {
    event_store: Arc<EventStore>,
    #[allow(dead_code)]
    state: Arc<dyn StateStore>,
    provider: Arc<dyn ModelProvider>,
    fast_model: String,
    max_events: usize,
    include_raw_args: bool,
}

impl DiagnoseTool {
    pub fn new(
        event_store: Arc<EventStore>,
        state: Arc<dyn StateStore>,
        provider: Arc<dyn ModelProvider>,
        fast_model: String,
        max_events: usize,
        include_raw_args: bool,
    ) -> Self {
        Self {
            event_store,
            state,
            provider,
            fast_model,
            max_events: max_events.clamp(50, 1000),
            include_raw_args,
        }
    }

    async fn list_tasks(
        &self,
        session_id: &str,
        limit: usize,
        failures_only: bool,
    ) -> anyhow::Result<String> {
        let rows = self
            .event_store
            .query_recent_task_ends(session_id, failures_only, limit.clamp(1, 50))
            .await?;
        if rows.is_empty() {
            return Ok("No matching tasks found in this session.".to_string());
        }

        let mut out = format!(
            "**Recent Tasks**\n\n- Session: {}\n- Count: {}\n- Failures only: {}\n\n",
            session_id,
            rows.len(),
            failures_only
        );
        for event in rows {
            if let Ok(end) = event.parse_data::<TaskEndData>() {
                let status = end.status.as_str();
                let err = end
                    .error
                    .as_deref()
                    .map(|s| truncate_str(s, 120))
                    .unwrap_or_else(|| "-".to_string());
                out.push_str(&format!(
                    "- `{}` status={} duration={}s iterations={} tool_calls={} at {}\n  error: {}\n",
                    end.task_id,
                    status,
                    end.duration_secs,
                    end.iterations,
                    end.tool_calls_count,
                    event.created_at.to_rfc3339(),
                    redact_secrets(&err)
                ));
            }
        }
        Ok(out.trim_end().to_string())
    }

    async fn collect_task_timeline(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<Vec<Event>> {
        let mut events = self
            .event_store
            .query_task_events_for_session(session_id, task_id)
            .await?;
        let mut decision_points = self
            .event_store
            .query_decision_points(session_id, task_id)
            .await?;
        events.append(&mut decision_points);

        events.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
        let mut seen = HashSet::new();
        events.retain(|e| seen.insert(e.id));

        if events.len() > self.max_events {
            let keep_from = events.len() - self.max_events;
            events = events.split_off(keep_from);
        }
        Ok(events)
    }

    fn brief_for_event(&self, event: &Event) -> String {
        match event.event_type {
            EventType::TaskStart => {
                if let Ok(data) = event.parse_data::<TaskStartData>() {
                    let desc = truncate_str(&data.description, 120);
                    format!("task_start: {}", desc)
                } else {
                    "task_start".to_string()
                }
            }
            EventType::ThinkingStart => {
                if let Some(iter) = event.data.get("iteration").and_then(|v| v.as_u64()) {
                    format!("thinking_start: iteration {}", iter)
                } else {
                    "thinking_start".to_string()
                }
            }
            EventType::ToolCall => {
                if let Ok(data) = event.parse_data::<ToolCallData>() {
                    let detail = if self.include_raw_args {
                        truncate_str(&data.arguments.to_string(), 160)
                    } else {
                        data.summary
                            .unwrap_or_else(|| truncate_str(&data.arguments.to_string(), 100))
                    };
                    format!("tool_call {} -> {}", data.name, detail)
                } else {
                    "tool_call".to_string()
                }
            }
            EventType::ToolResult => {
                if let Ok(data) = event.parse_data::<ToolResultData>() {
                    let status = if data.success { "ok" } else { "err" };
                    let detail = if let Some(err) = data.error {
                        truncate_str(&err, 140)
                    } else {
                        truncate_str(&data.result, 140)
                    };
                    format!(
                        "tool_result {} [{}] {}ms -> {}",
                        data.name, status, data.duration_ms, detail
                    )
                } else {
                    "tool_result".to_string()
                }
            }
            EventType::Error => {
                if let Ok(data) = event.parse_data::<ErrorData>() {
                    format!(
                        "error {:?}: {}",
                        data.error_type,
                        truncate_str(&data.message, 160)
                    )
                } else {
                    "error".to_string()
                }
            }
            EventType::TaskEnd => {
                if let Ok(data) = event.parse_data::<TaskEndData>() {
                    let detail = data
                        .error
                        .as_deref()
                        .map(|e| truncate_str(e, 120))
                        .unwrap_or_else(|| "-".to_string());
                    format!(
                        "task_end {} duration={}s tool_calls={} err={}",
                        data.status.as_str(),
                        data.duration_secs,
                        data.tool_calls_count,
                        detail
                    )
                } else {
                    "task_end".to_string()
                }
            }
            EventType::DecisionPoint => {
                if let Ok(data) = event.parse_data::<DecisionPointData>() {
                    format!(
                        ">>> DECISION {:?} (iter {}) {}",
                        data.decision_type,
                        data.iteration,
                        truncate_str(&data.summary, 140)
                    )
                } else {
                    ">>> DECISION".to_string()
                }
            }
            EventType::ApprovalDenied => {
                if let Ok(data) = event.parse_data::<ApprovalDeniedData>() {
                    format!(
                        "approval_denied command={}",
                        truncate_str(&data.command, 120)
                    )
                } else {
                    "approval_denied".to_string()
                }
            }
            _ => event.event_type.as_str().to_string(),
        }
    }

    fn timeline_text(&self, events: &[Event]) -> String {
        if events.is_empty() {
            return "No events found for this task in the current session.".to_string();
        }
        let mut out = String::new();
        for event in events {
            let line = self.brief_for_event(event);
            out.push_str(&format!(
                "{} [#{} {}] {}\n",
                event.created_at.to_rfc3339(),
                event.id,
                event.event_type.as_str(),
                redact_secrets(&line)
            ));
        }
        out.trim_end().to_string()
    }

    async fn timeline(&self, session_id: &str, task_id: &str) -> anyhow::Result<String> {
        let events = self.collect_task_timeline(session_id, task_id).await?;
        let mut out = format!(
            "**Task Timeline**\n\n- Session: {}\n- Task: {}\n- Events: {}\n\n",
            session_id,
            task_id,
            events.len()
        );
        out.push_str(&self.timeline_text(&events));
        Ok(out)
    }

    fn evidence_from_event(&self, event: &Event) -> crate::events::EvidenceRef {
        crate::events::EvidenceRef {
            event_id: event.id,
            event_type: event.event_type.as_str().to_string(),
            timestamp: event.created_at.to_rfc3339(),
            summary: redact_secrets(&truncate_str(&self.brief_for_event(event), 180)),
        }
    }

    fn build_deterministic_analysis(&self, events: &[Event]) -> DeterministicAnalysis {
        let mut first_error: Option<crate::events::EvidenceRef> = None;
        let mut task_end_failed: Option<(TaskEndData, crate::events::EvidenceRef)> = None;
        let mut approval_denied = Vec::new();
        let mut tool_failures = Vec::new();
        let mut provider_errors = Vec::new();
        let mut missing_context = Vec::new();
        let mut loop_signals = Vec::new();

        for event in events {
            let ev = self.evidence_from_event(event);
            match event.event_type {
                EventType::Error => {
                    if first_error.is_none() {
                        first_error = Some(ev.clone());
                    }
                    if let Ok(data) = event.parse_data::<ErrorData>() {
                        let lower = data.message.to_ascii_lowercase();
                        if matches!(
                            data.error_type,
                            ErrorType::LlmError | ErrorType::Timeout | ErrorType::RateLimit
                        ) || lower.contains("rate limit")
                            || lower.contains("provider")
                            || lower.contains("429")
                        {
                            provider_errors.push(ev.clone());
                        }
                        if lower.contains("no such file")
                            || lower.contains("not found")
                            || lower.contains("missing")
                        {
                            missing_context.push(ev.clone());
                        }
                    }
                }
                EventType::ToolResult => {
                    if let Ok(data) = event.parse_data::<ToolResultData>() {
                        if !data.success {
                            if first_error.is_none() {
                                first_error = Some(ev.clone());
                            }
                            tool_failures.push(ev);
                        }
                    }
                }
                EventType::ApprovalDenied => approval_denied.push(ev),
                EventType::DecisionPoint => {
                    if let Ok(data) = event.parse_data::<DecisionPointData>() {
                        if matches!(
                            data.decision_type,
                            DecisionType::RepetitiveCallDetection
                                | DecisionType::ConsecutiveSameToolDetection
                                | DecisionType::AlternatingPatternDetection
                        ) || matches!(data.decision_type, DecisionType::StoppingCondition)
                            && data.summary.to_ascii_lowercase().contains("stall")
                        {
                            loop_signals.push(ev);
                        }
                    }
                }
                EventType::TaskEnd => {
                    if let Ok(data) = event.parse_data::<TaskEndData>() {
                        if matches!(data.status, crate::events::TaskStatus::Failed) {
                            if data
                                .error
                                .as_deref()
                                .is_some_and(|e| e.to_ascii_lowercase().contains("stall"))
                            {
                                loop_signals.push(ev.clone());
                            }
                            task_end_failed = Some((data, ev));
                        }
                    }
                }
                _ => {}
            }
        }

        let what_failed = if let Some((failed, _)) = &task_end_failed {
            failed
                .error
                .clone()
                .or_else(|| failed.summary.clone())
                .unwrap_or_else(|| "Task ended as failed".to_string())
        } else if let Some(err) = &first_error {
            err.summary.clone()
        } else {
            "Task did not complete as expected.".to_string()
        };

        let mut candidates: Vec<RootCauseCandidate> = Vec::new();
        if !approval_denied.is_empty() {
            candidates.push(RootCauseCandidate {
                category: FailureCategory::SandboxPermissionBlock,
                confidence: 0.9,
                description: "Execution required approval and the command was denied or timed out."
                    .to_string(),
                evidence: approval_denied.iter().take(3).cloned().collect(),
                why_previous_step_looked_valid: Some(
                    "The agent selected an executable path, but approval policy blocked execution."
                        .to_string(),
                ),
            });
        }
        if !loop_signals.is_empty() {
            candidates.push(RootCauseCandidate {
                category: FailureCategory::AgentLoop,
                confidence: 0.86,
                description:
                    "Loop guards/stall signals show repeated behavior without meaningful progress."
                        .to_string(),
                evidence: loop_signals.iter().take(3).cloned().collect(),
                why_previous_step_looked_valid: Some(
                    "Each step appeared actionable in isolation, but repeated patterns prevented net progress."
                        .to_string(),
                ),
            });
        }
        if !provider_errors.is_empty() {
            candidates.push(RootCauseCandidate {
                category: FailureCategory::ProviderError,
                confidence: 0.82,
                description: "LLM/provider-side failure signals were detected.".to_string(),
                evidence: provider_errors.iter().take(3).cloned().collect(),
                why_previous_step_looked_valid: None,
            });
        }
        if !tool_failures.is_empty() {
            candidates.push(RootCauseCandidate {
                category: FailureCategory::ToolFailure,
                confidence: 0.76,
                description: "One or more tool executions failed and cascaded into task failure."
                    .to_string(),
                evidence: tool_failures.iter().take(3).cloned().collect(),
                why_previous_step_looked_valid: None,
            });
        }
        if !missing_context.is_empty() {
            candidates.push(RootCauseCandidate {
                category: FailureCategory::MissingContext,
                confidence: 0.72,
                description:
                    "Execution referenced resources that appear missing or unresolved in context."
                        .to_string(),
                evidence: missing_context.iter().take(3).cloned().collect(),
                why_previous_step_looked_valid: None,
            });
        }
        if candidates.is_empty() {
            let fallback_evidence = events
                .last()
                .map(|e| vec![self.evidence_from_event(e)])
                .unwrap_or_default();
            candidates.push(RootCauseCandidate {
                category: FailureCategory::IncorrectResult,
                confidence: 0.5,
                description:
                    "No strong deterministic failure signature was found; likely output quality issue."
                        .to_string(),
                evidence: fallback_evidence,
                why_previous_step_looked_valid: None,
            });
        }

        candidates.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let top = candidates.first().map(|c| c.category);
        let (minimal_fix, verification_steps) = match top {
            Some(FailureCategory::SandboxPermissionBlock) => (
                vec![
                    "Re-run with explicit approval for the blocked command.".to_string(),
                    "If this is expected recurring behavior, grant session/persistent approval as appropriate."
                        .to_string(),
                ],
                vec![
                    "Confirm an `approval_granted` event appears for the command.".to_string(),
                    "Confirm subsequent `tool_result` is successful.".to_string(),
                ],
            ),
            Some(FailureCategory::AgentLoop) => (
                vec![
                    "Add a stricter stop condition or vary tool strategy after first repeated failure."
                        .to_string(),
                    "Inject a clarifying question earlier when progress is ambiguous.".to_string(),
                ],
                vec![
                    "Verify no repetitive/consecutive/alternating decision-point events fire on retry."
                        .to_string(),
                    "Verify task ends with `completed` instead of stall/failure.".to_string(),
                ],
            ),
            Some(FailureCategory::ProviderError) => (
                vec![
                    "Retry with fallback model/provider and backoff.".to_string(),
                    "Check provider auth/rate-limit status.".to_string(),
                ],
                vec![
                    "No new provider/LLM error events in rerun.".to_string(),
                    "Task reaches successful completion.".to_string(),
                ],
            ),
            Some(FailureCategory::ToolFailure) => (
                vec![
                    "Fix tool arguments/environment based on the first failing tool result."
                        .to_string(),
                    "Avoid retrying unchanged failing tool calls repeatedly.".to_string(),
                ],
                vec![
                    "First previously failing tool call now returns success.".to_string(),
                    "Task completes without downstream failures.".to_string(),
                ],
            ),
            _ => (
                vec!["Apply a minimal correction and rerun from the first divergence point."
                    .to_string()],
                vec!["Confirm final output matches expected behavior.".to_string()],
            ),
        };

        DeterministicAnalysis {
            what_failed: redact_secrets(&what_failed),
            candidates,
            why_previous_step_looked_valid: None,
            minimal_fix,
            verification_steps,
        }
    }

    async fn llm_rank_candidates(
        &self,
        task_id: &str,
        timeline_text: &str,
        deterministic: &DeterministicAnalysis,
    ) -> Option<LlmDiagnosisResponse> {
        let allowed_categories: Vec<String> = deterministic
            .candidates
            .iter()
            .map(|c| {
                serde_json::to_string(&c.category)
                    .unwrap_or_else(|_| "\"incorrect_result\"".to_string())
            })
            .map(|s| s.trim_matches('"').to_string())
            .collect();
        let deterministic_json = serde_json::to_string_pretty(&deterministic).ok()?;

        let system =
            "You are a strict diagnostics ranker. Ground every claim in provided event IDs. \
Return JSON only, no markdown.";
        let user = format!(
            "Task: {task_id}\n\
Allowed categories (must choose only from these): {allowed}\n\n\
Deterministic baseline:\n{baseline}\n\n\
Timeline:\n{timeline}\n\n\
Return JSON object:\n\
{{\n\
  \"what_failed\": \"...\",\n\
  \"candidates\": [{{\"category\":\"...\",\"confidence\":0.0,\"description\":\"...\",\"evidence_event_ids\":[1,2],\"why_previous_step_looked_valid\":\"...\"}}],\n\
  \"why_previous_step_looked_valid\": \"...\",\n\
  \"minimal_fix\": [\"...\"],\n\
  \"verification_steps\": [\"...\"]\n\
}}\n\
Rules: no invented event IDs; confidence 0..1.",
            allowed = allowed_categories.join(", "),
            baseline = deterministic_json,
            timeline = truncate_str(timeline_text, 12000)
        );
        let messages = vec![
            json!({"role":"system","content":system}),
            json!({"role":"user","content":user}),
        ];

        let resp: ProviderResponse =
            match self.provider.chat(&self.fast_model, &messages, &[]).await {
                Ok(r) => r,
                Err(_) => return None,
            };
        // Track token usage for diagnostic LLM calls
        if let Some(usage) = &resp.usage {
            let _ = self
                .state
                .record_token_usage("background:diagnose", usage)
                .await;
        }
        let raw = resp.content.or(resp.thinking)?;
        let json_text = extract_json_object(&raw)?;
        serde_json::from_str::<LlmDiagnosisResponse>(&json_text).ok()
    }

    fn merge_llm_analysis(
        &self,
        mut deterministic: DeterministicAnalysis,
        llm: Option<LlmDiagnosisResponse>,
        events: &[Event],
    ) -> DeterministicAnalysis {
        let Some(llm) = llm else {
            return deterministic;
        };

        let mut evidence_by_id: HashMap<i64, crate::events::EvidenceRef> = HashMap::new();
        for event in events {
            evidence_by_id.insert(event.id, self.evidence_from_event(event));
        }
        let mut by_category: HashMap<FailureCategory, usize> = HashMap::new();
        for (idx, c) in deterministic.candidates.iter().enumerate() {
            by_category.insert(c.category, idx);
        }

        if let Some(cands) = llm.candidates {
            for cand in cands {
                if let Some(idx) = by_category.get(&cand.category).copied() {
                    let existing = &mut deterministic.candidates[idx];
                    let llm_conf = cand.confidence.clamp(0.0, 1.0);
                    existing.confidence =
                        (existing.confidence * 0.6 + llm_conf * 0.4).clamp(0.0, 1.0);
                    if !cand.description.trim().is_empty() {
                        existing.description = redact_secrets(&cand.description);
                    }
                    let mut mapped = Vec::new();
                    for id in cand.evidence_event_ids {
                        if let Some(ev) = evidence_by_id.get(&id) {
                            mapped.push(ev.clone());
                        }
                    }
                    if !mapped.is_empty() {
                        existing.evidence = mapped;
                    }
                    if let Some(why) = cand.why_previous_step_looked_valid {
                        if !why.trim().is_empty() {
                            existing.why_previous_step_looked_valid =
                                Some(redact_secrets(&truncate_str(&why, 220)));
                        }
                    }
                }
            }
        }
        deterministic
            .candidates
            .sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        if let Some(what_failed) = llm.what_failed {
            if !what_failed.trim().is_empty() {
                deterministic.what_failed = redact_secrets(&truncate_str(&what_failed, 300));
            }
        }
        if let Some(why) = llm.why_previous_step_looked_valid {
            if !why.trim().is_empty() {
                deterministic.why_previous_step_looked_valid =
                    Some(redact_secrets(&truncate_str(&why, 260)));
            }
        }
        if let Some(fixes) = llm.minimal_fix {
            let filtered: Vec<String> = fixes
                .into_iter()
                .filter(|s| !s.trim().is_empty())
                .map(|s| redact_secrets(&truncate_str(&s, 200)))
                .collect();
            if !filtered.is_empty() {
                deterministic.minimal_fix = filtered;
            }
        }
        if let Some(steps) = llm.verification_steps {
            let filtered: Vec<String> = steps
                .into_iter()
                .filter(|s| !s.trim().is_empty())
                .map(|s| redact_secrets(&truncate_str(&s, 200)))
                .collect();
            if !filtered.is_empty() {
                deterministic.verification_steps = filtered;
            }
        }

        deterministic
    }

    fn parse_resumed_task_id_from_error(&self, text: &str) -> Option<String> {
        const MARKER: &str = "Resumed in task ";
        let idx = text.find(MARKER)?;
        let tail = text[idx + MARKER.len()..].trim();
        let first_token = tail
            .trim_start_matches('`')
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches('.')
            .trim_end_matches('`')
            .trim();
        if first_token.is_empty() {
            None
        } else {
            Some(first_token.to_string())
        }
    }

    fn checkpoint_snapshot_from_events(
        &self,
        task_id: &str,
        events: &[Event],
    ) -> ResumeCheckpointSnapshot {
        let mut description: Option<String> = None;
        let mut original_user_message: Option<String> = None;
        let mut parent_task_id: Option<String> = None;
        let mut resumed_into_task_id: Option<String> = None;
        let mut duration_secs: Option<u64> = None;
        let mut last_iteration: u32 = 0;
        let mut tool_results_count: u32 = 0;
        let mut pending_tool_calls: HashSet<String> = HashSet::new();
        let mut last_assistant_summary: Option<String> = None;
        let mut last_tool_summary: Option<String> = None;
        let mut last_error: Option<String> = None;

        for event in events {
            match event.event_type {
                EventType::TaskStart => {
                    if let Ok(data) = event.parse_data::<TaskStartData>() {
                        if description.is_none() {
                            description = Some(data.description);
                        }
                        if original_user_message.is_none() {
                            original_user_message = data.user_message;
                        }
                        if parent_task_id.is_none() {
                            parent_task_id = data.parent_task_id;
                        }
                    }
                }
                EventType::ThinkingStart => {
                    if let Ok(data) = event.parse_data::<ThinkingStartData>() {
                        last_iteration = last_iteration.max(data.iteration);
                    }
                }
                EventType::AssistantResponse => {
                    if let Ok(data) = event.parse_data::<AssistantResponseData>() {
                        if let Some(calls) = data.tool_calls {
                            for call in calls {
                                pending_tool_calls.insert(call.id);
                            }
                        }
                        if let Some(content) = data.content {
                            let trimmed = content.trim();
                            if !trimmed.is_empty() {
                                last_assistant_summary = Some(truncate_str(trimmed, 180));
                            }
                        }
                    }
                }
                EventType::ToolResult => {
                    if let Ok(data) = event.parse_data::<ToolResultData>() {
                        tool_results_count = tool_results_count.saturating_add(1);
                        pending_tool_calls.remove(&data.tool_call_id);
                        let detail = data.error.unwrap_or(data.result);
                        let trimmed = detail.trim();
                        if !trimmed.is_empty() {
                            last_tool_summary = Some(truncate_str(trimmed, 180));
                        }
                    }
                }
                EventType::Error => {
                    if let Ok(data) = event.parse_data::<ErrorData>() {
                        let trimmed = data.message.trim();
                        if !trimmed.is_empty() {
                            last_error = Some(truncate_str(trimmed, 180));
                        }
                    }
                }
                EventType::TaskEnd => {
                    if let Ok(data) = event.parse_data::<TaskEndData>() {
                        duration_secs = Some(data.duration_secs);
                        if let Some(err) = data.error {
                            let trimmed = err.trim();
                            if !trimmed.is_empty() {
                                last_error = Some(truncate_str(trimmed, 180));
                                resumed_into_task_id =
                                    self.parse_resumed_task_id_from_error(trimmed);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let mut pending_tool_call_ids: Vec<String> = pending_tool_calls.into_iter().collect();
        pending_tool_call_ids.sort();

        ResumeCheckpointSnapshot {
            task_id: task_id.to_string(),
            description,
            original_user_message,
            parent_task_id,
            resumed_into_task_id,
            duration_secs,
            last_iteration,
            tool_results_count,
            pending_tool_call_ids,
            last_assistant_summary,
            last_tool_summary,
            last_error,
        }
    }

    async fn build_resume_recovery_section(
        &self,
        session_id: &str,
        task_id: &str,
        task_events: &[Event],
    ) -> String {
        let current = self.checkpoint_snapshot_from_events(task_id, task_events);

        let mut lines = vec!["### Resume Recovery State".to_string()];
        lines.push(format!(
            "- Current task: `{}` (iteration {}, tool_results {}, pending_tool_calls {})",
            current.task_id,
            current.last_iteration,
            current.tool_results_count,
            current.pending_tool_call_ids.len()
        ));

        if let Some(desc) = current.description.as_ref() {
            lines.push(format!("- Current description: {}", redact_secrets(desc)));
        }
        if let Some(parent) = current.parent_task_id.as_ref() {
            lines.push(format!("- Resumed from previous task: `{}`", parent));
            let parent_events = self
                .event_store
                .query_task_events_for_session(session_id, parent)
                .await
                .unwrap_or_default();
            if !parent_events.is_empty() {
                let parent_snapshot = self.checkpoint_snapshot_from_events(parent, &parent_events);
                if let Some(duration) = parent_snapshot.duration_secs {
                    lines.push(format!(
                        "- Previous elapsed before interruption: {}s",
                        duration
                    ));
                }
                lines.push(format!(
                    "- Previous checkpoint: iteration {}, tool_results {}, pending_tool_calls {}",
                    parent_snapshot.last_iteration,
                    parent_snapshot.tool_results_count,
                    parent_snapshot.pending_tool_call_ids.len()
                ));
                if let Some(desc) = parent_snapshot.description.as_ref() {
                    lines.push(format!("- Previous description: {}", redact_secrets(desc)));
                }
                if let Some(msg) = parent_snapshot.original_user_message.as_ref() {
                    lines.push(format!(
                        "- Previous user request: {}",
                        redact_secrets(&truncate_str(msg, 180))
                    ));
                }
                if let Some(summary) = parent_snapshot.last_assistant_summary.as_ref() {
                    lines.push(format!(
                        "- Previous last assistant output: {}",
                        redact_secrets(summary)
                    ));
                }
                if let Some(summary) = parent_snapshot.last_tool_summary.as_ref() {
                    lines.push(format!(
                        "- Previous last tool result: {}",
                        redact_secrets(summary)
                    ));
                }
                if let Some(err) = parent_snapshot.last_error.as_ref() {
                    lines.push(format!(
                        "- Previous interruption/error: {}",
                        redact_secrets(err)
                    ));
                }
                if let Some(next_task) = parent_snapshot.resumed_into_task_id.as_ref() {
                    lines.push(format!("- Previous task resumed into: `{}`", next_task));
                }
            } else {
                lines.push("- Previous task events not found in this session.".to_string());
            }
        } else if let Some(next_task) = current.resumed_into_task_id.as_ref() {
            lines.push(format!(
                "- This task was interrupted and resumed into: `{}`",
                next_task
            ));
            if let Some(duration) = current.duration_secs {
                lines.push(format!("- Elapsed before interruption: {}s", duration));
            }
        } else {
            lines.push("- No resume linkage detected for this task.".to_string());
        }

        if let Some(err) = current.last_error.as_ref() {
            lines.push(format!("- Current last error: {}", redact_secrets(err)));
        }
        if let Some(summary) = current.last_assistant_summary.as_ref() {
            lines.push(format!(
                "- Current last assistant output: {}",
                redact_secrets(summary)
            ));
        }
        if let Some(summary) = current.last_tool_summary.as_ref() {
            lines.push(format!(
                "- Current last tool result: {}",
                redact_secrets(summary)
            ));
        }

        lines.join("\n")
    }

    fn format_diagnosis(
        &self,
        task_id: &str,
        analysis: &DeterministicAnalysis,
        recovery_section: &str,
    ) -> String {
        let mut out = format!(
            "## Diagnosis for Task {}\n\n### What Failed\n{}\n\n### Most Likely Cause(s)\n",
            task_id, analysis.what_failed
        );
        for (idx, candidate) in analysis.candidates.iter().enumerate() {
            out.push_str(&format!(
                "{}. **{:?}** (confidence: {:.0}%)\n{}\n",
                idx + 1,
                candidate.category,
                candidate.confidence * 100.0,
                candidate.description
            ));
            if candidate.evidence.is_empty() {
                out.push_str("Evidence: (none linked)\n");
            } else {
                let refs: Vec<String> = candidate
                    .evidence
                    .iter()
                    .map(|e| {
                        format!(
                            "Event #{} ({}): {}",
                            e.event_id,
                            e.timestamp,
                            truncate_str(&e.summary, 140)
                        )
                    })
                    .collect();
                out.push_str(&format!("Evidence: {}\n", refs.join(" | ")));
            }
        }

        out.push_str("\n### Why Previous Step Looked Valid\n");
        if let Some(why) = analysis
            .why_previous_step_looked_valid
            .as_ref()
            .or_else(|| {
                analysis
                    .candidates
                    .first()?
                    .why_previous_step_looked_valid
                    .as_ref()
            })
        {
            out.push_str(why);
        } else {
            out.push_str("No explicit misleading-validity signal detected.");
        }

        out.push_str("\n\n### Minimal Fix\n");
        for step in &analysis.minimal_fix {
            out.push_str(&format!("- {}\n", step));
        }

        out.push_str("\n### Verification Steps\n");
        for step in &analysis.verification_steps {
            out.push_str(&format!("- {}\n", step));
        }
        out.push_str("\n");
        out.push_str(recovery_section);
        out.trim_end().to_string()
    }

    async fn diagnose(&self, session_id: &str, task_id: Option<&str>) -> anyhow::Result<String> {
        let resolved_task = if let Some(task_id) = task_id {
            task_id.to_string()
        } else {
            let latest_failed = self
                .event_store
                .query_recent_task_ends(session_id, true, 1)
                .await?;
            let Some(event) = latest_failed.first() else {
                return Ok(
                    "No failed task found in this session. Provide task_id or use action=list_tasks."
                        .to_string(),
                );
            };
            if let Ok(end) = event.parse_data::<TaskEndData>() {
                end.task_id
            } else {
                return Ok("Could not resolve latest failed task ID.".to_string());
            }
        };

        let events = self
            .collect_task_timeline(session_id, &resolved_task)
            .await?;
        if events.is_empty() {
            return Ok(format!(
                "No events found for task `{}` in session `{}`.",
                resolved_task, session_id
            ));
        }

        let timeline_text = self.timeline_text(&events);
        let deterministic = self.build_deterministic_analysis(&events);
        let llm = self
            .llm_rank_candidates(&resolved_task, &timeline_text, &deterministic)
            .await;
        let merged = self.merge_llm_analysis(deterministic, llm, &events);
        let recovery_section = self
            .build_resume_recovery_section(session_id, &resolved_task, &events)
            .await;
        Ok(self.format_diagnosis(&resolved_task, &merged, &recovery_section))
    }

    fn signature_for_event(&self, event: &Event) -> String {
        match event.event_type {
            EventType::DecisionPoint => {
                if let Ok(dp) = event.parse_data::<DecisionPointData>() {
                    format!(
                        "decision:{:?}:{}",
                        dp.decision_type,
                        truncate_str(&dp.summary, 40)
                    )
                } else {
                    "decision".to_string()
                }
            }
            EventType::ToolCall => {
                if let Ok(tc) = event.parse_data::<ToolCallData>() {
                    format!("tool_call:{}", tc.name)
                } else {
                    "tool_call".to_string()
                }
            }
            EventType::ToolResult => {
                if let Ok(tr) = event.parse_data::<ToolResultData>() {
                    format!("tool_result:{}:{}", tr.name, tr.success)
                } else {
                    "tool_result".to_string()
                }
            }
            EventType::TaskEnd => {
                if let Ok(te) = event.parse_data::<TaskEndData>() {
                    format!("task_end:{}", te.status.as_str())
                } else {
                    "task_end".to_string()
                }
            }
            _ => event.event_type.as_str().to_string(),
        }
    }

    async fn compare(
        &self,
        session_id: &str,
        task_a: &str,
        task_b: &str,
    ) -> anyhow::Result<String> {
        let timeline_a = self.collect_task_timeline(session_id, task_a).await?;
        let timeline_b = self.collect_task_timeline(session_id, task_b).await?;
        if timeline_a.is_empty() || timeline_b.is_empty() {
            return Ok("One or both tasks have no events in this session.".to_string());
        }

        let sig_a: Vec<String> = timeline_a
            .iter()
            .map(|e| self.signature_for_event(e))
            .collect();
        let sig_b: Vec<String> = timeline_b
            .iter()
            .map(|e| self.signature_for_event(e))
            .collect();
        let mut divergence_at = None;
        for i in 0..sig_a.len().min(sig_b.len()) {
            if sig_a[i] != sig_b[i] {
                divergence_at = Some(i);
                break;
            }
        }
        let idx = divergence_at.unwrap_or(sig_a.len().min(sig_b.len()));

        let mut out = format!(
            "**Task Comparison**\n\n- Session: {}\n- Task A: {} ({} events)\n- Task B: {} ({} events)\n",
            session_id,
            task_a,
            timeline_a.len(),
            task_b,
            timeline_b.len()
        );
        if let Some(i) = divergence_at {
            out.push_str(&format!("\nFirst divergence at position {}:\n", i + 1));
            out.push_str(&format!(
                "- A: {}\n- B: {}\n",
                redact_secrets(&self.brief_for_event(&timeline_a[i])),
                redact_secrets(&self.brief_for_event(&timeline_b[i]))
            ));
        } else {
            out.push_str("\nNo divergence in shared prefix; differences only in tail events.\n");
        }

        if let Some(a) = timeline_a.get(idx) {
            out.push_str(&format!(
                "\nNext unique A event: {}",
                redact_secrets(&self.brief_for_event(a))
            ));
        }
        if let Some(b) = timeline_b.get(idx) {
            out.push_str(&format!(
                "\nNext unique B event: {}",
                redact_secrets(&self.brief_for_event(b))
            ));
        }

        Ok(out)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeterministicAnalysis {
    what_failed: String,
    candidates: Vec<RootCauseCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    why_previous_step_looked_valid: Option<String>,
    minimal_fix: Vec<String>,
    verification_steps: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DiagnoseArgs {
    action: String,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    task_id_a: Option<String>,
    #[serde(default)]
    task_id_b: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    failures_only: Option<bool>,
    #[serde(default)]
    _session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlmDiagnosisResponse {
    #[serde(default)]
    what_failed: Option<String>,
    #[serde(default)]
    candidates: Option<Vec<LlmCandidate>>,
    #[serde(default)]
    why_previous_step_looked_valid: Option<String>,
    #[serde(default)]
    minimal_fix: Option<Vec<String>>,
    #[serde(default)]
    verification_steps: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct LlmCandidate {
    category: FailureCategory,
    confidence: f32,
    description: String,
    #[serde(default)]
    evidence_event_ids: Vec<i64>,
    #[serde(default)]
    why_previous_step_looked_valid: Option<String>,
}

#[derive(Debug, Clone)]
struct ResumeCheckpointSnapshot {
    task_id: String,
    description: Option<String>,
    original_user_message: Option<String>,
    parent_task_id: Option<String>,
    resumed_into_task_id: Option<String>,
    duration_secs: Option<u64>,
    last_iteration: u32,
    tool_results_count: u32,
    pending_tool_call_ids: Vec<String>,
    last_assistant_summary: Option<String>,
    last_tool_summary: Option<String>,
    last_error: Option<String>,
}

fn extract_json_object(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let candidate = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```JSON")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else {
        trimmed.to_string()
    };
    if serde_json::from_str::<Value>(&candidate)
        .ok()
        .is_some_and(|v| v.is_object())
    {
        return Some(candidate);
    }

    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let sliced = raw[start..=end].trim().to_string();
    if serde_json::from_str::<Value>(&sliced)
        .ok()
        .is_some_and(|v| v.is_object())
    {
        Some(sliced)
    } else {
        None
    }
}

#[async_trait]
impl Tool for DiagnoseTool {
    fn name(&self) -> &str {
        "self_diagnose"
    }

    fn description(&self) -> &str {
        "Diagnose why an agent task failed using event timelines, decision points, and evidence-linked root-cause ranking."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "self_diagnose",
            "description": "Self-diagnostics for agent task failures. Use list_tasks, timeline, diagnose, compare.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type":"string",
                        "enum":["list_tasks","timeline","diagnose","compare"]
                    },
                    "task_id": {
                        "type":"string",
                        "description":"Target task ID for timeline/diagnose."
                    },
                    "task_id_a": {
                        "type":"string",
                        "description":"First task ID for compare."
                    },
                    "task_id_b": {
                        "type":"string",
                        "description":"Second task ID for compare."
                    },
                    "limit": {
                        "type":"integer",
                        "description":"List limit (default 10, max 50)."
                    },
                    "failures_only": {
                        "type":"boolean",
                        "description":"For list_tasks, filter to failed tasks (default true)."
                    }
                },
                "required":["action"],
                "additionalProperties": false
            }
        })
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            read_only: true,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        let args: DiagnoseArgs = serde_json::from_str(arguments)?;
        let session_id = args
            ._session_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("self_diagnose requires internal _session_id"))?;

        match args.action.as_str() {
            "list_tasks" => {
                self.list_tasks(
                    session_id,
                    args.limit.unwrap_or(10),
                    args.failures_only.unwrap_or(true),
                )
                .await
            }
            "timeline" => {
                let task_id = args
                    .task_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("task_id is required for action=timeline"))?;
                self.timeline(session_id, task_id).await
            }
            "diagnose" => self.diagnose(session_id, args.task_id.as_deref()).await,
            "compare" => {
                let a = args
                    .task_id_a
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("task_id_a is required for action=compare"))?;
                let b = args
                    .task_id_b
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("task_id_b is required for action=compare"))?;
                self.compare(session_id, a, b).await
            }
            other => Ok(format!(
                "Unknown action: '{}'. Use list_tasks, timeline, diagnose, or compare.",
                other
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    use crate::events::{DecisionPointData, TaskStatus};
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;
    use crate::traits::TokenUsage;

    struct MockProvider {
        content: String,
    }

    #[async_trait]
    impl ModelProvider for MockProvider {
        async fn chat(
            &self,
            _model: &str,
            _messages: &[Value],
            _tools: &[Value],
        ) -> anyhow::Result<ProviderResponse> {
            Ok(ProviderResponse {
                content: Some(self.content.clone()),
                tool_calls: vec![],
                usage: Some(TokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                    model: "mock".to_string(),
                }),
                thinking: None,
                response_note: None,
            })
        }

        async fn list_models(&self) -> anyhow::Result<Vec<String>> {
            Ok(vec!["mock-fast".to_string()])
        }
    }

    async fn setup_tool() -> DiagnoseTool {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let db_path = db_file.path().to_string_lossy().to_string();
        let embedding_service = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(&db_path, 50, None, embedding_service)
                .await
                .expect("state"),
        );
        std::mem::forget(db_file);

        let event_store = Arc::new(EventStore::new(state.pool()).await.expect("event store"));
        let provider: Arc<dyn ModelProvider> = Arc::new(MockProvider {
            content: json!({
                "what_failed":"Tool execution failed",
                "candidates":[
                    {"category":"tool_failure","confidence":0.8,"description":"Tool failed","evidence_event_ids":[3]}
                ],
                "minimal_fix":["Fix tool arguments"],
                "verification_steps":["Re-run and confirm success"]
            })
            .to_string(),
        });
        DiagnoseTool::new(
            event_store,
            state as Arc<dyn StateStore>,
            provider,
            "mock-fast".to_string(),
            200,
            false,
        )
    }

    async fn append_event<T: serde::Serialize>(
        tool: &DiagnoseTool,
        session_id: &str,
        event_type: EventType,
        data: T,
        task_id: Option<&str>,
    ) {
        let mut event = crate::events::Event::new(
            session_id.to_string(),
            event_type,
            serde_json::to_value(data).unwrap(),
        );
        event.task_id = task_id.map(str::to_string);
        event.created_at = Utc::now();
        tool.event_store.append(event).await.expect("append");
    }

    #[tokio::test]
    async fn test_list_tasks() {
        let tool = setup_tool().await;
        append_event(
            &tool,
            "s1",
            EventType::TaskEnd,
            TaskEndData {
                task_id: "t1".to_string(),
                status: TaskStatus::Failed,
                duration_secs: 3,
                iterations: 1,
                tool_calls_count: 2,
                error: Some("boom".to_string()),
                summary: None,
            },
            Some("t1"),
        )
        .await;
        append_event(
            &tool,
            "s2",
            EventType::TaskEnd,
            TaskEndData {
                task_id: "t2".to_string(),
                status: TaskStatus::Failed,
                duration_secs: 3,
                iterations: 1,
                tool_calls_count: 2,
                error: Some("other".to_string()),
                summary: None,
            },
            Some("t2"),
        )
        .await;

        let res = tool
            .call(r#"{"action":"list_tasks","_session_id":"s1"}"#)
            .await
            .unwrap();
        assert!(res.contains("t1"));
        assert!(!res.contains("t2"));
    }

    #[tokio::test]
    async fn test_timeline_reconstruction_and_decision_points() {
        let tool = setup_tool().await;
        append_event(
            &tool,
            "s1",
            EventType::TaskStart,
            TaskStartData {
                task_id: "t1".to_string(),
                description: "Test".to_string(),
                parent_task_id: None,
                user_message: None,
            },
            Some("t1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::DecisionPoint,
            DecisionPointData {
                decision_type: DecisionType::IntentGate,
                task_id: "t1".to_string(),
                iteration: 1,
                metadata: json!({"needs_tools":true}),
                summary: "intent gate forced tool mode".to_string(),
            },
            Some("t1"),
        )
        .await;
        let res = tool
            .call(r#"{"action":"timeline","task_id":"t1","_session_id":"s1"}"#)
            .await
            .unwrap();
        assert!(res.contains("Task Timeline"));
        assert!(res.contains(">>> DECISION"));
    }

    #[tokio::test]
    async fn test_diagnose_finds_first_error() {
        let tool = setup_tool().await;
        append_event(
            &tool,
            "s1",
            EventType::TaskStart,
            TaskStartData {
                task_id: "t1".to_string(),
                description: "Run".to_string(),
                parent_task_id: None,
                user_message: None,
            },
            Some("t1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::Error,
            ErrorData::tool_error(
                "terminal",
                "No such file or directory",
                Some("t1".to_string()),
            ),
            Some("t1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::TaskEnd,
            TaskEndData {
                task_id: "t1".to_string(),
                status: TaskStatus::Failed,
                duration_secs: 2,
                iterations: 1,
                tool_calls_count: 1,
                error: Some("failed".to_string()),
                summary: None,
            },
            Some("t1"),
        )
        .await;

        let res = tool
            .call(r#"{"action":"diagnose","task_id":"t1","_session_id":"s1"}"#)
            .await
            .unwrap();
        assert!(res.contains("Diagnosis for Task t1"));
        assert!(res.contains("Most Likely Cause(s)"));
    }

    #[tokio::test]
    async fn test_diagnose_includes_resume_recovery_state() {
        let tool = setup_tool().await;

        // Parent task (interrupted and resumed)
        append_event(
            &tool,
            "s1",
            EventType::TaskStart,
            TaskStartData {
                task_id: "parent-1".to_string(),
                description: "Build website and deploy".to_string(),
                parent_task_id: None,
                user_message: Some("Build website and deploy".to_string()),
            },
            Some("parent-1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::ThinkingStart,
            ThinkingStartData {
                iteration: 3,
                task_id: "parent-1".to_string(),
                total_tool_calls: 2,
            },
            Some("parent-1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::AssistantResponse,
            AssistantResponseData {
                message_id: None,
                content: Some("I'll continue by checking deployment config.".to_string()),
                tool_calls: Some(vec![crate::events::ToolCallInfo {
                    id: "call-parent-pending".to_string(),
                    name: "system_info".to_string(),
                    arguments: json!({}),
                    extra_content: None,
                }]),
                model: "mock-fast".to_string(),
                input_tokens: None,
                output_tokens: None,
            },
            Some("parent-1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::TaskEnd,
            TaskEndData {
                task_id: "parent-1".to_string(),
                status: TaskStatus::Failed,
                duration_secs: 42,
                iterations: 3,
                tool_calls_count: 1,
                error: Some(
                    "Agent process interrupted before completion. Resumed in task child-1."
                        .to_string(),
                ),
                summary: Some("Recovered from checkpoint after interruption".to_string()),
            },
            Some("parent-1"),
        )
        .await;

        // Child resumed task
        append_event(
            &tool,
            "s1",
            EventType::TaskStart,
            TaskStartData {
                task_id: "child-1".to_string(),
                description: "resume: Build website and deploy".to_string(),
                parent_task_id: Some("parent-1".to_string()),
                user_message: Some("continue".to_string()),
            },
            Some("child-1"),
        )
        .await;
        append_event(
            &tool,
            "s1",
            EventType::TaskEnd,
            TaskEndData {
                task_id: "child-1".to_string(),
                status: TaskStatus::Failed,
                duration_secs: 10,
                iterations: 1,
                tool_calls_count: 0,
                error: Some("still failing".to_string()),
                summary: None,
            },
            Some("child-1"),
        )
        .await;

        let res = tool
            .call(r#"{"action":"diagnose","task_id":"child-1","_session_id":"s1"}"#)
            .await
            .unwrap();

        assert!(res.contains("### Resume Recovery State"));
        assert!(res.contains("Resumed from previous task: `parent-1`"));
        assert!(res.contains("Previous checkpoint: iteration 3"));
        assert!(res.contains("Previous task resumed into: `child-1`"));
    }

    #[tokio::test]
    async fn test_pii_redaction_in_timeline() {
        let tool = setup_tool().await;
        append_event(
            &tool,
            "s1",
            EventType::ToolResult,
            ToolResultData {
                message_id: None,
                tool_call_id: "c1".to_string(),
                name: "web_fetch".to_string(),
                result: "token sk-abc123456789012345678901234567890 leaked".to_string(),
                success: false,
                duration_ms: 10,
                error: None,
                task_id: Some("t1".to_string()),
            },
            Some("t1"),
        )
        .await;
        let res = tool
            .call(r#"{"action":"timeline","task_id":"t1","_session_id":"s1"}"#)
            .await
            .unwrap();
        assert!(!res.contains("sk-abc"));
        assert!(res.contains("REDACTED"));
    }
}
