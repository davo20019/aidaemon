use super::history::{
    assistant_message_looks_like_clarifying_question, infer_completion_contract,
    looks_like_context_dependent_followup_question, looks_like_explicit_task_switch,
    looks_like_self_contained_mutation_request, looks_like_short_command_request,
    looks_like_standalone_goal_request, looks_like_unanswered_request_reference,
};
use super::*;
use crate::events::{
    AssistantResponseData, EventType, TaskEndData, TaskStartData, TaskStatus, UserMessageData,
};
use crate::traits::{
    extract_primary_message_content, message_content_is_structural_only, ActiveTaskRef,
    ActiveTaskStatus, AssistantTurnKind, AssistantTurnSummary, DialogueState, Message,
    OpenQuestion, OpenRequest, OpenRequestStatus, QuestionKind, ToolSemanticScope, UserTurnKind,
    UserTurnSummary,
};
use chrono::Utc;

fn is_courtesy_only(lower: &str) -> bool {
    contains_keyword_as_words(lower, "thanks")
        || contains_keyword_as_words(lower, "thank you")
        || contains_keyword_as_words(lower, "got it")
        || contains_keyword_as_words(lower, "sounds good")
        || contains_keyword_as_words(lower, "okay thanks")
        || contains_keyword_as_words(lower, "ok thanks")
}

fn completion_contract_looks_actionable(contract: &CompletionContract) -> bool {
    contract.expects_mutation
        || contract.requires_observation
        || contract.explicit_verification_requested
        || !contract.verification_targets.is_empty()
        || contract.task_kind != CompletionTaskKind::Conversational
}

fn infer_open_request_scope(text: &str) -> Option<ToolSemanticScope> {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return None;
    }

    let is_scheduled_execution =
        lower.contains("[system: already scheduled and firing now; do not reschedule.]");

    if !is_scheduled_execution
        && [
            "scheduled goal",
            "scheduled task",
            "schedule",
            "scheduled",
            "scheduler",
            "recurring",
            "reminder",
            "trigger",
            "triggered",
            "run history",
            "goal status",
        ]
        .iter()
        .any(|term| contains_keyword_as_words(&lower, term))
    {
        return Some(ToolSemanticScope::GoalState);
    }

    if [
        "what is my name",
        "what's my name",
        "who am i",
        "what do you know about me",
        "remember",
        "saved fact",
    ]
    .iter()
    .any(|term| contains_keyword_as_words(&lower, term))
    {
        return Some(ToolSemanticScope::UserMemory);
    }

    if [
        "channel history",
        "conversation history",
        "what was discussed",
        "what happened in",
    ]
    .iter()
    .any(|term| contains_keyword_as_words(&lower, term))
    {
        return Some(ToolSemanticScope::ConversationHistory);
    }

    if [
        "http://",
        "https://",
        "website",
        "web page",
        "article",
        "docs",
        "documentation",
        "search the web",
    ]
    .iter()
    .any(|term| lower.contains(term))
    {
        return Some(ToolSemanticScope::ExternalRemote);
    }

    if [
        "repo",
        "repository",
        "project",
        "file",
        "directory",
        "folder",
        "path",
        ".rs",
        ".toml",
        ".json",
        ".md",
        "/",
        "~/",
        "./",
    ]
    .iter()
    .any(|term| lower.contains(term))
    {
        return Some(ToolSemanticScope::LocalWorkspace);
    }

    if [
        "hostname",
        "cpu",
        "memory",
        "ram",
        "disk",
        "uptime",
        "operating system",
        "local time",
        "timezone",
    ]
    .iter()
    .any(|term| contains_keyword_as_words(&lower, term))
    {
        return Some(ToolSemanticScope::HostLocal);
    }

    None
}

/// Finished requests only anchor followups briefly; open ones expire after a
/// hard TTL. Past these bounds the request is history: it must not capture
/// turn classification, tool scoping, or prompt injection (the 2026-07-11
/// incident re-answered a request resolved two hours earlier).
const RESOLVED_REQUEST_ANCHOR_LINGER_MINUTES: i64 = 10;
const OPEN_REQUEST_ANCHOR_TTL_HOURS: i64 = 12;
/// A pending clarifying question binds non-ack replies only briefly; ack-like
/// replies ("yes do 1, 2") bind regardless of age — answering a menu after a
/// long pause is normal, but an arbitrary short message hours later is a new
/// request, not an answer.
const OPEN_QUESTION_IMPLICIT_BIND_TTL_MINUTES: i64 = 30;
/// A question the user just answered stays readable for the answering turn's
/// prompt; the binding is per-exchange, not durable state.
const CLOSED_QUESTION_INJECTION_TTL_MINUTES: i64 = 30;

fn open_request_anchor_expired(request: &OpenRequest, now: chrono::DateTime<Utc>) -> bool {
    match request.status {
        OpenRequestStatus::Answered | OpenRequestStatus::Superseded => {
            request.resolved_at.is_none_or(|resolved| {
                now.signed_duration_since(resolved)
                    > chrono::Duration::minutes(RESOLVED_REQUEST_ANCHOR_LINGER_MINUTES)
            })
        }
        _ => {
            now.signed_duration_since(request.opened_at)
                > chrono::Duration::hours(OPEN_REQUEST_ANCHOR_TTL_HOURS)
        }
    }
}

fn user_reply_is_concise_ack(trimmed: &str, lower: &str) -> bool {
    trimmed.chars().count() <= 120
        && (contains_keyword_as_words(lower, "yes")
            || contains_keyword_as_words(lower, "no")
            || contains_keyword_as_words(lower, "do it")
            || contains_keyword_as_words(lower, "go ahead")
            || contains_keyword_as_words(lower, "sure")
            || contains_keyword_as_words(lower, "confirm")
            || contains_keyword_as_words(lower, "that one")
            || contains_keyword_as_words(lower, "the first one")
            || contains_keyword_as_words(lower, "the second one")
            || contains_keyword_as_words(lower, "use this")
            || contains_keyword_as_words(lower, "use that"))
}

fn user_reply_likely_answers_open_question(trimmed: &str, lower: &str) -> bool {
    user_reply_is_concise_ack(trimmed, lower)
        || (trimmed.chars().count() <= 120
            && !looks_like_explicit_task_switch(lower)
            && !looks_like_standalone_goal_request(lower)
            && !looks_like_self_contained_mutation_request(trimmed, lower)
            && !looks_like_short_command_request(trimmed))
}

fn user_turn_strongly_references_existing_request(
    state: &DialogueState,
    _trimmed: &str,
    lower: &str,
) -> bool {
    if state.open_request.is_none() {
        return false;
    }

    if looks_like_context_dependent_followup_question(lower)
        || looks_like_unanswered_request_reference(lower)
        || lower.starts_with("also ")
        || lower.starts_with("and ")
        || lower.starts_with("plus ")
        || contains_keyword_as_words(lower, "continue")
        || contains_keyword_as_words(lower, "go on")
        || contains_keyword_as_words(lower, "what about")
        || contains_keyword_as_words(lower, "try again")
        || contains_keyword_as_words(lower, "retry")
    {
        return true;
    }

    false
}

fn user_turn_weakly_references_existing_request(
    state: &DialogueState,
    trimmed: &str,
    lower: &str,
) -> bool {
    if state.open_request.is_none() {
        return false;
    }

    trimmed.chars().count() <= 80
        && !looks_like_explicit_task_switch(lower)
        && !looks_like_standalone_goal_request(lower)
        && !looks_like_self_contained_mutation_request(trimmed, lower)
        && !looks_like_short_command_request(trimmed)
        && !is_courtesy_only(lower)
}

fn classify_user_turn(
    state: &DialogueState,
    text: &str,
    alias_roots: &[String],
    observed_at: chrono::DateTime<Utc>,
) -> UserTurnKind {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return UserTurnKind::Unknown;
    }
    let lower = trimmed.to_ascii_lowercase();

    if is_courtesy_only(&lower) {
        return UserTurnKind::Courtesy;
    }

    if state.open_question.as_ref().is_some_and(|question| {
        question.awaiting_user_reply
            && user_reply_likely_answers_open_question(trimmed, &lower)
            && (user_reply_is_concise_ack(trimmed, &lower)
                || observed_at.signed_duration_since(question.asked_at)
                    <= chrono::Duration::minutes(OPEN_QUESTION_IMPLICIT_BIND_TTL_MINUTES))
    }) {
        return UserTurnKind::ClarificationAnswer;
    }

    if user_turn_strongly_references_existing_request(state, trimmed, &lower) {
        return UserTurnKind::Followup;
    }

    let contract = infer_completion_contract(trimmed, alias_roots);
    if looks_like_explicit_task_switch(&lower)
        || looks_like_standalone_goal_request(&lower)
        || looks_like_self_contained_mutation_request(trimmed, &lower)
        || looks_like_short_command_request(trimmed)
        || completion_contract_looks_actionable(&contract)
    {
        return UserTurnKind::NewRequest;
    }

    if user_turn_weakly_references_existing_request(state, trimmed, &lower) {
        return UserTurnKind::Followup;
    }

    if state.open_request.is_none() {
        return UserTurnKind::NewRequest;
    }

    if state.open_request.is_some() {
        UserTurnKind::Followup
    } else {
        UserTurnKind::Unknown
    }
}

fn clarification_question_kind(lower: &str) -> QuestionKind {
    if contains_keyword_as_words(lower, "approve")
        || contains_keyword_as_words(lower, "allow")
        || contains_keyword_as_words(lower, "permission")
        || contains_keyword_as_words(lower, "proceed")
    {
        return QuestionKind::Approval;
    }
    if contains_keyword_as_words(lower, "which")
        || contains_keyword_as_words(lower, "choose")
        || contains_keyword_as_words(lower, "pick")
        || contains_keyword_as_words(lower, "prefer")
    {
        return QuestionKind::Choice;
    }
    if contains_keyword_as_words(lower, "confirm")
        || contains_keyword_as_words(lower, "is that right")
        || contains_keyword_as_words(lower, "correct")
    {
        return QuestionKind::Confirmation;
    }
    QuestionKind::Clarification
}

fn classify_assistant_turn_text(text: &str) -> (AssistantTurnKind, bool) {
    let trimmed = text.trim();
    let lower = trimmed.to_ascii_lowercase();
    if trimmed.is_empty() {
        return (AssistantTurnKind::SystemNotice, false);
    }

    if assistant_message_looks_like_clarifying_question(trimmed) {
        return (AssistantTurnKind::ClarificationQuestion, true);
    }

    let blocked = [
        "i can't",
        "i cannot",
        "couldn't",
        "could not",
        "blocked",
        "need approval",
        "need your approval",
        "need confirmation",
        "failed",
        "error",
        "not able to",
        "unable to",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    if blocked {
        return (AssistantTurnKind::Blocked, true);
    }

    let partial_progress = [
        "here is the latest result",
        "so far",
        "current status",
        "still need",
        "still working",
        "i'm working",
        "i am working",
        "i checked",
        "i found",
        "i started",
        "partial",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    if partial_progress {
        return (AssistantTurnKind::PartialProgress, true);
    }

    if message_content_is_structural_only(trimmed, &[]) {
        return (AssistantTurnKind::SystemNotice, false);
    }

    (AssistantTurnKind::SubstantiveAnswer, false)
}

fn apply_task_start(state: &mut DialogueState, task_id: &str, started_at: chrono::DateTime<Utc>) {
    state.active_task = Some(ActiveTaskRef {
        task_id: task_id.to_string(),
        status: ActiveTaskStatus::Running,
        started_at,
    });
    if let Some(open_request) = state.open_request.as_mut() {
        open_request.task_id = Some(task_id.to_string());
        if open_request.status == OpenRequestStatus::Open {
            open_request.status = OpenRequestStatus::InProgress;
        }
    }
    state.touch();
}

fn apply_user_message(
    state: &mut DialogueState,
    message_id: &str,
    text: &str,
    alias_roots: &[String],
    observed_at: chrono::DateTime<Utc>,
) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    // Expire stale request anchors BEFORE classification so a long-finished
    // request cannot capture this turn's kind, tool scope, or injection.
    if state
        .open_request
        .as_ref()
        .is_some_and(|request| open_request_anchor_expired(request, observed_at))
    {
        state.open_request = None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let turn_kind = classify_user_turn(state, trimmed, alias_roots, observed_at);
    state.last_user_turn = Some(UserTurnSummary {
        message_id: message_id.to_string(),
        kind: turn_kind,
        text: trimmed.to_string(),
    });

    match turn_kind {
        UserTurnKind::ClarificationAnswer => {
            // Stash the question this reply answers: `open_question` is
            // cleared during ingestion, but the turn that ANSWERS it still
            // needs the question text when its prompt is composed.
            if let Some(question) = state.open_question.take() {
                state.last_closed_question = Some(question);
            }
            if let Some(open_request) = state.open_request.as_mut() {
                if matches!(
                    open_request.status,
                    OpenRequestStatus::Blocked | OpenRequestStatus::PartiallyAnswered
                ) {
                    open_request.status = if state.active_task.is_some() {
                        OpenRequestStatus::InProgress
                    } else {
                        OpenRequestStatus::Open
                    };
                    open_request.resolved_at = None;
                }
            }
        }
        UserTurnKind::Followup => {
            if contains_keyword_as_words(&lower, "retry")
                || contains_keyword_as_words(&lower, "try again")
                || contains_keyword_as_words(&lower, "continue")
            {
                if let Some(open_request) = state.open_request.as_mut() {
                    open_request.status = if state.active_task.is_some() {
                        OpenRequestStatus::InProgress
                    } else {
                        OpenRequestStatus::Open
                    };
                    open_request.resolved_at = None;
                }
            }
        }
        UserTurnKind::NewRequest => {
            if let Some(open_request) = state.open_request.as_mut() {
                if !matches!(
                    open_request.status,
                    OpenRequestStatus::Answered | OpenRequestStatus::Superseded
                ) {
                    open_request.status = OpenRequestStatus::Superseded;
                    open_request.resolved_at = Some(observed_at);
                }
            }
            state.open_question = None;
            state.last_closed_question = None;
            state.open_request = Some(OpenRequest {
                user_message_id: message_id.to_string(),
                text: trimmed.to_string(),
                status: if state.active_task.is_some() {
                    OpenRequestStatus::InProgress
                } else {
                    OpenRequestStatus::Open
                },
                task_id: state.active_task.as_ref().map(|task| task.task_id.clone()),
                project_scope: None,
                semantic_scope: infer_open_request_scope(trimmed),
                opened_at: observed_at,
                resolved_at: None,
            });
        }
        UserTurnKind::Courtesy | UserTurnKind::Unknown => {}
    }

    state.touch();
}

fn apply_assistant_message(
    state: &mut DialogueState,
    message_id: &str,
    content: &str,
    observed_at: chrono::DateTime<Utc>,
) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }

    let (kind, left_request_open) = classify_assistant_turn_text(trimmed);
    state.last_assistant_turn = Some(AssistantTurnSummary {
        message_id: message_id.to_string(),
        kind,
        left_request_open,
        text: trimmed.to_string(),
    });

    match kind {
        AssistantTurnKind::ClarificationQuestion => {
            state.open_question = Some(OpenQuestion {
                assistant_message_id: message_id.to_string(),
                text: trimmed.to_string(),
                kind: clarification_question_kind(&trimmed.to_ascii_lowercase()),
                related_user_message_id: state
                    .open_request
                    .as_ref()
                    .map(|request| request.user_message_id.clone()),
                awaiting_user_reply: true,
                asked_at: observed_at,
            });
            if let Some(open_request) = state.open_request.as_mut() {
                if open_request.status == OpenRequestStatus::Open {
                    open_request.status = OpenRequestStatus::PartiallyAnswered;
                }
            }
        }
        AssistantTurnKind::PartialProgress => {
            state.open_question = None;
            if let Some(open_request) = state.open_request.as_mut() {
                if matches!(
                    open_request.status,
                    OpenRequestStatus::Open | OpenRequestStatus::InProgress
                ) {
                    open_request.status = OpenRequestStatus::PartiallyAnswered;
                }
            }
        }
        AssistantTurnKind::Blocked | AssistantTurnKind::Refusal => {
            state.open_question = None;
            if let Some(open_request) = state.open_request.as_mut() {
                open_request.status = OpenRequestStatus::Blocked;
                open_request.resolved_at = Some(observed_at);
            }
        }
        AssistantTurnKind::SubstantiveAnswer => {
            state.open_question = None;
            if let Some(open_request) = state.open_request.as_mut() {
                open_request.status = OpenRequestStatus::Answered;
                open_request.resolved_at = Some(observed_at);
            }
        }
        AssistantTurnKind::SystemNotice => {}
    }

    state.touch();
}

fn apply_task_end(
    state: &mut DialogueState,
    task_id: &str,
    status: TaskStatus,
    observed_at: chrono::DateTime<Utc>,
) {
    let matches_active_task = state
        .active_task
        .as_ref()
        .is_some_and(|task| task.task_id == task_id);
    let matches_request = state
        .open_request
        .as_ref()
        .and_then(|request| request.task_id.as_deref())
        .is_some_and(|request_task_id| request_task_id == task_id);

    if let Some(open_request) = state.open_request.as_mut() {
        if matches_request || open_request.task_id.is_none() {
            match status {
                TaskStatus::Completed => {
                    if state
                        .open_question
                        .as_ref()
                        .is_some_and(|question| question.awaiting_user_reply)
                        || state
                            .last_assistant_turn
                            .as_ref()
                            .is_some_and(|turn| turn.left_request_open)
                    {
                        if open_request.status == OpenRequestStatus::InProgress {
                            open_request.status = OpenRequestStatus::PartiallyAnswered;
                        }
                    } else if !matches!(
                        open_request.status,
                        OpenRequestStatus::Answered | OpenRequestStatus::Superseded
                    ) {
                        open_request.status = OpenRequestStatus::Answered;
                        open_request.resolved_at = Some(observed_at);
                    }
                }
                TaskStatus::Failed | TaskStatus::Cancelled => {
                    open_request.status = OpenRequestStatus::Blocked;
                    open_request.resolved_at = Some(observed_at);
                }
            }
        }
    }

    if matches_active_task {
        state.active_task = None;
    }
    state.touch();
}

pub(in crate::agent) async fn get_or_rebuild_dialogue_state(
    agent: &Agent,
    session_id: &str,
) -> DialogueState {
    match agent.state.get_dialogue_state(session_id).await {
        Ok(Some(state)) if state.schema_version == DialogueState::SCHEMA_VERSION => state,
        Ok(_) | Err(_) => {
            let rebuilt = rebuild_dialogue_state_from_events(agent, session_id).await;
            if let Err(err) = agent.state.upsert_dialogue_state(&rebuilt).await {
                tracing::warn!(
                    session_id,
                    error = %err,
                    "Failed to persist rebuilt dialogue state"
                );
            }
            rebuilt
        }
    }
}

async fn rebuild_dialogue_state_from_events(agent: &Agent, session_id: &str) -> DialogueState {
    let mut state = DialogueState::new(session_id);
    let events = agent
        .event_store
        .query_recent_events(session_id, 200)
        .await
        .unwrap_or_default();

    for event in events {
        match event.event_type {
            EventType::TaskStart => {
                if let Ok(data) = event.parse_data::<TaskStartData>() {
                    apply_task_start(&mut state, &data.task_id, event.created_at);
                }
            }
            EventType::UserMessage => {
                if let Ok(data) = event.parse_data::<UserMessageData>() {
                    apply_user_message(
                        &mut state,
                        data.message_id.as_deref().unwrap_or_default(),
                        &data.content,
                        &agent.path_aliases.projects,
                        event.created_at,
                    );
                }
            }
            EventType::AssistantResponse => {
                if let Ok(data) = event.parse_data::<AssistantResponseData>() {
                    let raw = data.content.unwrap_or_default();
                    let primary = extract_primary_message_content(&raw, &data.annotations);
                    if !primary.trim().is_empty()
                        && data
                            .tool_calls
                            .as_ref()
                            .is_none_or(|calls| calls.is_empty())
                    {
                        apply_assistant_message(
                            &mut state,
                            data.message_id.as_deref().unwrap_or_default(),
                            &primary,
                            event.created_at,
                        );
                    }
                }
            }
            EventType::TaskEnd => {
                if let Ok(data) = event.parse_data::<TaskEndData>() {
                    apply_task_end(&mut state, &data.task_id, data.status, event.created_at);
                }
            }
            _ => {}
        }
    }

    state
}

pub(in crate::agent) async fn record_dialogue_task_start(
    agent: &Agent,
    session_id: &str,
    task_id: &str,
) -> anyhow::Result<()> {
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_task_start(&mut state, task_id, Utc::now());
    agent.state.upsert_dialogue_state(&state).await
}

pub(in crate::agent) async fn record_dialogue_user_message(
    agent: &Agent,
    session_id: &str,
    message: &Message,
) -> anyhow::Result<()> {
    let Some(content) = message.primary_content() else {
        return Ok(());
    };
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_user_message(
        &mut state,
        &message.id,
        &content,
        &agent.path_aliases.projects,
        message.created_at,
    );
    agent.state.upsert_dialogue_state(&state).await
}

pub(in crate::agent) async fn record_dialogue_assistant_message(
    agent: &Agent,
    session_id: &str,
    message: &Message,
) -> anyhow::Result<()> {
    if message.tool_calls_json.is_some() {
        return Ok(());
    }
    let Some(content) = message.content.as_deref() else {
        return Ok(());
    };
    let primary = extract_primary_message_content(content, &message.effective_annotations());
    if primary.trim().is_empty() {
        return Ok(());
    }

    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_assistant_message(&mut state, &message.id, &primary, message.created_at);
    agent.state.upsert_dialogue_state(&state).await
}

/// Tail-preserving clip for injected clarifying questions: the options at the
/// end are the actionable part the user's reply refers to.
fn clip_question_keeping_tail(text: &str) -> String {
    const CAP: usize = 2000;
    let count = text.chars().count();
    if count <= CAP {
        text.to_string()
    } else {
        let tail: String = text.chars().skip(count - CAP).collect();
        format!("…{tail}")
    }
}

/// Compose the LLM-visible user text for the current turn from the dialogue
/// state. Extracted from the main loop so the injection policy is unit-testable.
///
/// Followup turns carry the pending open request — but ONLY while it is
/// genuinely pending (not answered/superseded) and within its TTL; re-injecting
/// a resolved request resurrects questions the user already got answers for.
/// Clarification-answer turns carry the question being answered instead.
pub(in crate::agent) fn compose_llm_user_text(
    followup_mode: Option<super::followup::FollowupMode>,
    state: Option<&DialogueState>,
    user_text: &str,
    now: chrono::DateTime<Utc>,
) -> String {
    let trimmed = user_text.trim();
    match followup_mode {
        Some(super::followup::FollowupMode::Followup) => state
            .and_then(|s| s.open_request.as_ref())
            .filter(|request| !open_request_anchor_expired(request, now))
            .map(|request| request.text.trim())
            .filter(|original| !original.is_empty() && !original.eq_ignore_ascii_case(trimmed))
            .map(|original| format!("Original request:\n{original}\n\nFollow-up:\n{trimmed}"))
            .unwrap_or_else(|| user_text.to_string()),
        Some(super::followup::FollowupMode::ClarificationAnswer) => state
            .and_then(|s| {
                s.open_question
                    .as_ref()
                    .filter(|question| question.awaiting_user_reply)
                    .or(s.last_closed_question.as_ref())
            })
            .filter(|question| {
                now.signed_duration_since(question.asked_at)
                    <= chrono::Duration::minutes(CLOSED_QUESTION_INJECTION_TTL_MINUTES)
            })
            .map(|question| question.text.trim())
            .filter(|question| !question.is_empty())
            .map(|question| {
                format!(
                    "Assistant asked:\n{}\n\nFollow-up:\n{}",
                    clip_question_keeping_tail(question),
                    trimmed
                )
            })
            .unwrap_or_else(|| user_text.to_string()),
        _ => user_text.to_string(),
    }
}

pub(in crate::agent) async fn record_dialogue_task_end(
    agent: &Agent,
    session_id: &str,
    task_id: &str,
    status: TaskStatus,
) -> anyhow::Result<()> {
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_task_end(&mut state, task_id, status, Utc::now());
    agent.state.upsert_dialogue_state(&state).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_reply_to_open_question_is_not_new_request() {
        let mut state = DialogueState::new("s1");
        state.open_request = Some(OpenRequest {
            user_message_id: "u1".to_string(),
            text: "Deploy the site".to_string(),
            status: OpenRequestStatus::Open,
            task_id: None,
            project_scope: None,
            semantic_scope: None,
            opened_at: Utc::now(),
            resolved_at: None,
        });
        state.open_question = Some(OpenQuestion {
            assistant_message_id: "a1".to_string(),
            text: "Which environment should I use?".to_string(),
            kind: QuestionKind::Clarification,
            related_user_message_id: Some("u1".to_string()),
            awaiting_user_reply: true,
            asked_at: Utc::now(),
        });

        apply_user_message(&mut state, "u2", "production", &[], Utc::now());
        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::ClarificationAnswer)
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u1")
        );
        assert!(state.open_question.is_none());
    }

    #[test]
    fn unresolved_request_reference_stays_followup() {
        let mut state = DialogueState::new("s1");
        state.open_request = Some(OpenRequest {
            user_message_id: "u1".to_string(),
            text: "What were the latency regressions?".to_string(),
            status: OpenRequestStatus::Open,
            task_id: None,
            project_scope: None,
            semantic_scope: None,
            opened_at: Utc::now(),
            resolved_at: None,
        });

        apply_user_message(
            &mut state,
            "u2",
            "You didn't answer my question",
            &[],
            Utc::now(),
        );
        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::Followup)
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u1")
        );
    }

    #[test]
    fn schedule_request_records_goal_state_scope() {
        let mut state = DialogueState::new("s1");
        apply_user_message(
            &mut state,
            "u1",
            "What times does the tweet posting schedule trigger?",
            &[],
            Utc::now(),
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .and_then(|request| request.semantic_scope),
            Some(ToolSemanticScope::GoalState)
        );
    }

    #[test]
    fn scheduled_execution_instruction_does_not_use_goal_state_scope() {
        let mut state = DialogueState::new("s1");
        apply_user_message(
            &mut state,
            "u1",
            "Scheduled check: Post daily optimized tweets for the aidaemon Twitter account \
             [SYSTEM: already scheduled and firing now; do not reschedule.]",
            &[],
            Utc::now(),
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .and_then(|request| request.semantic_scope),
            None
        );
    }

    #[test]
    fn short_self_contained_schedule_query_supersedes_previous_request() {
        let mut state = DialogueState::new("s1");
        apply_user_message(
            &mut state,
            "u1",
            "top 3 tallest buildings in the world 2024 height",
            &[],
            Utc::now(),
        );

        let observed_at = Utc::now();
        apply_user_message(
            &mut state,
            "u2",
            "What are your scheduled tasks?",
            &[],
            observed_at,
        );

        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest)
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u2")
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .and_then(|request| request.semantic_scope),
            Some(ToolSemanticScope::GoalState)
        );
    }

    // ── 2026-07-11 stale-open-request incident regressions ──────────────

    const MENU: &str = "I couldn't find a file named \"Acme resume\" in your \
        `~/projects/resume` directory.\n\nWould you like me to:\n\
        1. Search your entire machine for anything related to \"Acme\"?\n\
        2. Look in other common folders like `~/Documents` or `~/Downloads`?\n\
        3. Send you one of your other recent resumes instead?";

    fn request_with(
        status: OpenRequestStatus,
        opened_at: chrono::DateTime<Utc>,
        resolved_at: Option<chrono::DateTime<Utc>>,
    ) -> OpenRequest {
        OpenRequest {
            user_message_id: "u1".to_string(),
            text: "What's a contract vehicle?".to_string(),
            status,
            task_id: None,
            project_scope: None,
            semantic_scope: None,
            opened_at,
            resolved_at,
        }
    }

    fn question_with(asked_at: chrono::DateTime<Utc>, awaiting: bool) -> OpenQuestion {
        OpenQuestion {
            assistant_message_id: "a1".to_string(),
            text: MENU.to_string(),
            kind: QuestionKind::Approval,
            related_user_message_id: None,
            awaiting_user_reply: awaiting,
            asked_at,
        }
    }

    #[test]
    fn compose_followup_skips_answered_open_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::hours(2),
            Some(now - chrono::Duration::hours(2)),
        ));
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::Followup),
            Some(&state),
            "Yes do 1, 2",
            now,
        );
        assert_eq!(
            out, "Yes do 1, 2",
            "answered requests must never be re-injected"
        );
    }

    #[test]
    fn compose_followup_injects_fresh_open_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Open,
            now - chrono::Duration::minutes(1),
            None,
        ));
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::Followup),
            Some(&state),
            "make it shorter",
            now,
        );
        assert!(out.starts_with("Original request:\nWhat's a contract vehicle?"));
        assert!(out.ends_with("Follow-up:\nmake it shorter"));
    }

    #[test]
    fn compose_followup_skips_expired_open_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Open,
            now - chrono::Duration::hours(13),
            None,
        ));
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::Followup),
            Some(&state),
            "make it shorter",
            now,
        );
        assert_eq!(
            out, "make it shorter",
            "requests past the TTL must not be injected"
        );
    }

    #[test]
    fn compose_clarification_answer_injects_closed_question() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.last_closed_question = Some(question_with(now - chrono::Duration::minutes(1), false));
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::ClarificationAnswer),
            Some(&state),
            "Yes do 1, 2",
            now,
        );
        assert!(out.starts_with("Assistant asked:"));
        assert!(out.contains("1. Search your entire machine"));
        assert!(out.ends_with("Follow-up:\nYes do 1, 2"));
    }

    #[test]
    fn compose_clarification_answer_prefers_awaiting_open_question() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        let mut awaiting = question_with(now - chrono::Duration::minutes(1), true);
        awaiting.text = "Should I proceed with the deploy?".to_string();
        state.open_question = Some(awaiting);
        state.last_closed_question = Some(question_with(now - chrono::Duration::minutes(5), false));
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::ClarificationAnswer),
            Some(&state),
            "yes",
            now,
        );
        assert!(out.contains("Should I proceed with the deploy?"));
        assert!(!out.contains("1. Search your entire machine"));
    }

    #[test]
    fn compose_clarification_answer_skips_stale_question() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.last_closed_question = Some(question_with(now - chrono::Duration::hours(2), false));
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::ClarificationAnswer),
            Some(&state),
            "Yes do 1, 2",
            now,
        );
        assert_eq!(out, "Yes do 1, 2");
    }

    #[test]
    fn compose_clarification_answer_without_question_passes_through() {
        let now = Utc::now();
        let state = DialogueState::new("s1");
        let out = compose_llm_user_text(
            Some(super::super::followup::FollowupMode::ClarificationAnswer),
            Some(&state),
            "Yes do 1, 2",
            now,
        );
        assert_eq!(out, "Yes do 1, 2");
    }

    #[test]
    fn clarification_answer_stashes_closed_question() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_question = Some(question_with(now - chrono::Duration::minutes(1), true));
        apply_user_message(&mut state, "u2", "Yes do 1, 2", &[], now);
        assert!(state.open_question.is_none());
        assert_eq!(
            state
                .last_closed_question
                .as_ref()
                .map(|question| question.text.as_str()),
            Some(MENU),
            "the answered question must remain readable for this turn's prompt"
        );
    }

    #[test]
    fn stale_answered_request_does_not_capture_short_new_request() {
        // Incident shape: "What's a contract vehicle?" was answered hours ago,
        // yet "Send me my makpar resume" kept anchoring to it as a followup,
        // so the open_request never rotated.
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::hours(2),
            Some(now - chrono::Duration::hours(2)),
        ));
        apply_user_message(&mut state, "u2", "Send me my makpar resume", &[], now);
        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest)
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u2"),
            "a stale answered request must be superseded by the new request"
        );
    }

    #[test]
    fn stale_open_question_does_not_capture_short_new_request() {
        // Same incident, other capture path: a clarifying menu asked hours ago
        // must not swallow a fresh plain-verb request as a "clarification
        // answer" just because the reply is short and non-command-shaped.
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_question = Some(question_with(now - chrono::Duration::hours(2), true));
        apply_user_message(&mut state, "u2", "Send me my makpar resume", &[], now);
        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest)
        );
    }

    #[test]
    fn fresh_open_request_still_anchors_short_followup() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        let mut request = request_with(OpenRequestStatus::Open, now, None);
        request.text = "Find my tax documents".to_string();
        state.open_request = Some(request);
        apply_user_message(&mut state, "u2", "hmm the second folder maybe", &[], now);
        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::Followup)
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u1")
        );
    }

    #[test]
    fn incident_2026_07_11_ack_binds_to_menu_not_stale_request() {
        // Full replay of the incident chain: a request answered two hours ago
        // sits in state, the assistant presents a clarifying menu, the user
        // acks with option numbers. The composed prompt must carry the menu —
        // not resurrect the stale request, and not pass bare text.
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::hours(2),
            Some(now - chrono::Duration::hours(2)),
        ));
        apply_assistant_message(&mut state, "a2", MENU, now - chrono::Duration::seconds(90));
        apply_user_message(&mut state, "u2", "Yes do 1, 2", &[], now);

        let (mode, _) = super::super::followup::classify_followup_mode("Yes do 1, 2", Some(MENU));
        let out = compose_llm_user_text(Some(mode), Some(&state), "Yes do 1, 2", now);
        assert!(out.starts_with("Assistant asked:"), "got: {out}");
        assert!(out.contains("1. Search your entire machine"));
        assert!(
            !out.contains("contract vehicle"),
            "the stale answered request must not be resurrected"
        );
    }

    #[test]
    fn task_end_completes_substantive_answered_request() {
        let mut state = DialogueState::new("s1");
        apply_task_start(&mut state, "task-1", Utc::now());
        apply_user_message(&mut state, "u1", "Summarize the failures", &[], Utc::now());
        apply_assistant_message(
            &mut state,
            "a1",
            "The three failures were auth timeout, DNS mismatch, and a missing secret.",
            Utc::now(),
        );
        apply_task_end(&mut state, "task-1", TaskStatus::Completed, Utc::now());

        assert_eq!(
            state.open_request.as_ref().map(|request| request.status),
            Some(OpenRequestStatus::Answered)
        );
        assert!(state.active_task.is_none());
    }
}
