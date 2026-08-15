use super::history::assistant_message_looks_like_clarifying_question;
use super::*;
use crate::events::{
    AssistantResponseData, EventType, TaskEndData, TaskOutcome, TaskStartData, TaskStatus,
    UserMessageData,
};
use crate::traits::{
    extract_primary_message_content, message_content_is_structural_only, ActiveTaskRef,
    ActiveTaskStatus, AssistantTurnKind, AssistantTurnSummary, DialogueState, Message,
    OpenQuestion, OpenRequest, OpenRequestStatus, QuestionKind, ToolSemanticScope, UserTurnKind,
    UserTurnSummary,
};
use chrono::Utc;

fn is_courtesy_only(lower: &str) -> bool {
    matches!(
        lower
            .trim_matches(|ch: char| ch.is_ascii_punctuation())
            .trim(),
        "thanks" | "thank you" | "got it" | "sounds good" | "okay thanks" | "ok thanks"
    )
}

fn infer_open_request_scope(text: &str, alias_roots: &[String]) -> Option<ToolSemanticScope> {
    if text.trim().is_empty() {
        return None;
    }

    // Scope is a routing boundary, so only concrete resource identities may
    // establish it here. Domain words such as "project", "memory", "docs",
    // or "schedule" are semantic hints for the model; they are not durable
    // proof that this request belongs to a particular tool namespace.
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|ch: char| {
            matches!(
                ch,
                '`' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
            )
        });
        if token.starts_with("http://") || token.starts_with("https://") {
            return Some(ToolSemanticScope::ExternalRemote);
        }
        if crate::tools::fs_utils::resolve_structural_filesystem_reference(token, alias_roots)
            .is_some()
        {
            return Some(ToolSemanticScope::LocalWorkspace);
        }
    }

    None
}

/// Finished requests only anchor followups briefly; open ones expire after a
/// hard TTL. Past these bounds the request is history: it must not capture
/// turn classification, tool scoping, or prompt injection (the 2026-07-11
/// incident re-answered a request resolved two hours earlier).
const RESOLVED_REQUEST_ANCHOR_LINGER_MINUTES: i64 = 10;
const OPEN_REQUEST_ANCHOR_TTL_HOURS: i64 = 12;
/// A pending clarifying question binds implicitly only for a bounded period.
/// Durable mandate-input questions are linked by typed mandate identity and do
/// not depend on the reply's wording.
const OPEN_QUESTION_IMPLICIT_BIND_TTL_MINUTES: i64 = 30;

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

/// Return the single persisted classification for the current user turn.
///
/// `record_dialogue_user_message` runs during bootstrap before turn-context
/// assembly. Consumers should use this result instead of independently
/// classifying the same text a second time. Text equality protects callers
/// from accidentally reusing a stale dialogue-state row.
pub(in crate::agent) fn resolved_followup_mode(
    state: &DialogueState,
    current_user_text: &str,
) -> Option<super::followup::FollowupMode> {
    let current = current_user_text.trim();
    let turn = state
        .last_user_turn
        .as_ref()
        .filter(|turn| turn.text.trim().eq_ignore_ascii_case(current))?;
    Some(match turn.kind {
        UserTurnKind::Followup => super::followup::FollowupMode::Followup,
        UserTurnKind::ClarificationAnswer => super::followup::FollowupMode::ClarificationAnswer,
        UserTurnKind::NewRequest | UserTurnKind::Courtesy | UserTurnKind::Unknown => {
            super::followup::FollowupMode::NewTask
        }
    })
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
        let durable_mandate_input = question.kind == QuestionKind::MandateInput
            && question
                .mandate_id
                .as_deref()
                .is_some_and(|id| !id.is_empty());
        let separate_scoped_request = infer_open_request_scope(trimmed, alias_roots).is_some();
        question.awaiting_user_reply
            && !separate_scoped_request
            && (durable_mandate_input
                || observed_at.signed_duration_since(question.asked_at)
                    <= chrono::Duration::minutes(OPEN_QUESTION_IMPLICIT_BIND_TTL_MINUTES))
    }) {
        return UserTurnKind::ClarificationAnswer;
    }

    // Until semantic assessment runs, topology is the only safe fallback: an
    // unresolved request is the antecedent, while a terminal/absent one is not.
    if state.open_request.as_ref().is_some_and(|request| {
        matches!(
            request.status,
            OpenRequestStatus::Open
                | OpenRequestStatus::InProgress
                | OpenRequestStatus::PartiallyAnswered
                | OpenRequestStatus::Blocked
        )
    }) {
        UserTurnKind::Followup
    } else {
        UserTurnKind::NewRequest
    }
}

fn classify_assistant_turn_text(text: &str) -> (AssistantTurnKind, bool) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (AssistantTurnKind::SystemNotice, false);
    }

    if assistant_message_looks_like_clarifying_question(trimmed) {
        return (AssistantTurnKind::ClarificationQuestion, true);
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
            // A typed new TaskStart on a related turn reopens the request.
            // The user's choice of words ("retry", "continue", or any
            // paraphrase) is irrelevant to this lifecycle transition.
            if state.active_task.is_some() {
                if let Some(open_request) = state.open_request.as_mut() {
                    open_request.status = OpenRequestStatus::InProgress;
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
                semantic_scope: infer_open_request_scope(trimmed, alias_roots),
                completion_contract: None,
                opened_at: observed_at,
                resolved_at: None,
            });
        }
        UserTurnKind::Courtesy | UserTurnKind::Unknown => {}
    }

    state.touch();
}

fn parse_semantic_scope(value: &str) -> Result<Option<ToolSemanticScope>, ()> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(None),
        "goal_state" => Ok(Some(ToolSemanticScope::GoalState)),
        "user_memory" => Ok(Some(ToolSemanticScope::UserMemory)),
        "conversation_history" => Ok(Some(ToolSemanticScope::ConversationHistory)),
        "external_remote" => Ok(Some(ToolSemanticScope::ExternalRemote)),
        "local_workspace" => Ok(Some(ToolSemanticScope::LocalWorkspace)),
        "host_local" => Ok(Some(ToolSemanticScope::HostLocal)),
        _ => Err(()),
    }
}

fn parse_semantic_relationship(value: &str) -> Option<UserTurnKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "new_request" => Some(UserTurnKind::NewRequest),
        "continuation" => Some(UserTurnKind::Followup),
        "clarification_answer" => Some(UserTurnKind::ClarificationAnswer),
        "courtesy" => Some(UserTurnKind::Courtesy),
        _ => None,
    }
}

/// Commit semantic relationship/scope classification after task assessment.
/// Bootstrap's topology-only classification is deliberately provisional.
fn apply_semantic_user_turn_assessment(
    state: &mut DialogueState,
    text: &str,
    kind: UserTurnKind,
    semantic_scope: Option<ToolSemanticScope>,
    observed_at: chrono::DateTime<Utc>,
) {
    let Some(turn) = state
        .last_user_turn
        .as_mut()
        .filter(|turn| turn.text.trim().eq_ignore_ascii_case(text.trim()))
    else {
        return;
    };
    let message_id = turn.message_id.clone();
    turn.kind = kind;

    match kind {
        UserTurnKind::NewRequest => {
            if state
                .open_request
                .as_ref()
                .is_none_or(|request| request.user_message_id != message_id)
            {
                state.open_request = Some(OpenRequest {
                    user_message_id: message_id,
                    text: text.trim().to_string(),
                    status: if state.active_task.is_some() {
                        OpenRequestStatus::InProgress
                    } else {
                        OpenRequestStatus::Open
                    },
                    task_id: state.active_task.as_ref().map(|task| task.task_id.clone()),
                    project_scope: None,
                    semantic_scope,
                    completion_contract: None,
                    opened_at: observed_at,
                    resolved_at: None,
                });
            } else if let Some(request) = state.open_request.as_mut() {
                request.semantic_scope = semantic_scope;
            }
            state.open_question = None;
            state.last_closed_question = None;
        }
        UserTurnKind::ClarificationAnswer => {
            if let Some(question) = state.open_question.take() {
                state.last_closed_question = Some(question);
            }
        }
        UserTurnKind::Followup => {
            if let Some(request) = state.open_request.as_mut() {
                if semantic_scope.is_some() {
                    request.semantic_scope = semantic_scope;
                }
            }
        }
        UserTurnKind::Courtesy | UserTurnKind::Unknown => {}
    }
    state.touch();
}

pub(in crate::agent) async fn record_dialogue_semantic_user_turn(
    agent: &Agent,
    session_id: &str,
    text: &str,
    relationship: &str,
    semantic_scope: &str,
) -> anyhow::Result<Option<UserTurnKind>> {
    let Some(kind) = parse_semantic_relationship(relationship) else {
        return Ok(None);
    };
    let Ok(scope) = parse_semantic_scope(semantic_scope) else {
        return Ok(None);
    };
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_semantic_user_turn_assessment(&mut state, text, kind, scope, Utc::now());
    agent.state.upsert_dialogue_state(&state).await?;
    Ok(Some(kind))
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

    // Assistant text classification is presentation metadata. Request
    // lifecycle is committed only by typed TaskStart/TaskEnd events below;
    // otherwise prose containing words like "blocked", "partial", or
    // "failed" can silently rewrite durable state.
    match kind {
        AssistantTurnKind::ClarificationQuestion => {
            state.open_question = Some(OpenQuestion {
                assistant_message_id: message_id.to_string(),
                text: trimmed.to_string(),
                // Ordinary model-authored questions are provisional
                // clarification edges. Approval and mandate-input obligations
                // are created only by their typed runtime paths.
                kind: QuestionKind::Clarification,
                related_user_message_id: state
                    .open_request
                    .as_ref()
                    .map(|request| request.user_message_id.clone()),
                mandate_id: None,
                awaiting_user_reply: true,
                asked_at: observed_at,
            });
        }
        AssistantTurnKind::PartialProgress => {
            state.open_question = None;
        }
        AssistantTurnKind::Blocked | AssistantTurnKind::Refusal => {
            state.open_question = None;
        }
        AssistantTurnKind::SubstantiveAnswer => {
            state.open_question = None;
        }
        AssistantTurnKind::SystemNotice => {}
    }

    state.touch();
}

fn apply_task_end(
    state: &mut DialogueState,
    task_id: &str,
    status: TaskStatus,
    outcome: TaskOutcome,
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
                    match outcome {
                        TaskOutcome::Succeeded => {
                            let awaits_mandate_input =
                                state.open_question.as_ref().is_some_and(|question| {
                                    question.awaiting_user_reply
                                        && question.kind == QuestionKind::MandateInput
                                });
                            if awaits_mandate_input {
                                open_request.status = OpenRequestStatus::PartiallyAnswered;
                                open_request.resolved_at = None;
                            } else if !matches!(
                                open_request.status,
                                OpenRequestStatus::Answered | OpenRequestStatus::Superseded
                            ) {
                                open_request.status = OpenRequestStatus::Answered;
                                open_request.resolved_at = Some(observed_at);
                                state.open_question = None;
                            }
                        }
                        TaskOutcome::Partial => {
                            // Semantic outcome is authoritative over response prose. A useful
                            // explanation can accompany unfinished work, but it cannot resolve
                            // the request that produced it.
                            open_request.status = OpenRequestStatus::PartiallyAnswered;
                            open_request.resolved_at = None;
                        }
                        TaskOutcome::Failed => {
                            open_request.status = OpenRequestStatus::Blocked;
                            open_request.resolved_at = Some(observed_at);
                        }
                    }
                }
                TaskStatus::Failed | TaskStatus::Cancelled => {
                    open_request.status = OpenRequestStatus::Blocked;
                    open_request.resolved_at = Some(observed_at);
                }
                TaskStatus::Interrupted => {
                    open_request.status = OpenRequestStatus::PartiallyAnswered;
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
                    if data
                        .annotations
                        .contains(&crate::traits::MessageAnnotation::InternalContinuation)
                    {
                        continue;
                    }
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
                    apply_task_end(
                        &mut state,
                        &data.task_id,
                        data.status,
                        data.effective_outcome(),
                        event.created_at,
                    );
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
    if message
        .effective_annotations()
        .contains(&crate::traits::MessageAnnotation::InternalContinuation)
    {
        return Ok(());
    }
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

/// Attach the finalized typed completion obligations to the currently open
/// request. This lets later continuations inherit the contract without
/// reclassifying the original wording or depending on assessment availability.
pub(in crate::agent) async fn record_dialogue_completion_contract(
    agent: &Agent,
    session_id: &str,
    user_text: &str,
    contract: &super::completion_contract::CompletionContract,
) -> anyhow::Result<()> {
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    let current_turn_matches = state
        .last_user_turn
        .as_ref()
        .is_some_and(|turn| turn.text.trim().eq_ignore_ascii_case(user_text.trim()));
    if !current_turn_matches {
        return Ok(());
    }
    if let Some(request) = state.open_request.as_mut().filter(|request| {
        !matches!(
            request.status,
            OpenRequestStatus::Answered | OpenRequestStatus::Superseded
        )
    }) {
        request.completion_contract = Some(
            super::completion_contract::persistable_completion_contract(contract),
        );
        state.touch();
        agent.state.upsert_dialogue_state(&state).await?;
    }
    Ok(())
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

/// Persist a delivered mandate ASK as a typed owner-input obligation.
///
/// The visible notice intentionally contains no deliberator-authored question,
/// so prose classification cannot safely reconstruct this relationship. Only
/// authenticated runtime identifiers are stored here; mandate-local text must
/// still be retrieved through `manage_mandates(get)` on the related turn.
pub(in crate::agent) async fn record_mandate_owner_input(
    agent: &Agent,
    session_id: &str,
    mandate_id: &str,
    mandate_version: i64,
    notification_id: &str,
) -> anyhow::Result<()> {
    let mandate_id = mandate_id.trim();
    anyhow::ensure!(!mandate_id.is_empty(), "mandate id is required");
    anyhow::ensure!(mandate_version > 0, "mandate version must be positive");
    anyhow::ensure!(
        !notification_id.trim().is_empty(),
        "notification id is required"
    );

    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    let mandate_ref = mandate_id.chars().take(8).collect::<String>();
    state.open_question = Some(OpenQuestion {
        assistant_message_id: notification_id.to_string(),
        text: format!(
            "Mandate {mandate_ref} is awaiting owner input under policy version \
             {mandate_version}; inspect its mandate-local decision before responding."
        ),
        kind: QuestionKind::MandateInput,
        related_user_message_id: None,
        mandate_id: Some(mandate_id.to_string()),
        awaiting_user_reply: true,
        asked_at: Utc::now(),
    });
    state.last_closed_question = None;
    state.touch();
    agent.state.upsert_dialogue_state(&state).await
}

pub(in crate::agent) async fn record_dialogue_task_end(
    agent: &Agent,
    session_id: &str,
    task_id: &str,
    status: TaskStatus,
    outcome: TaskOutcome,
) -> anyhow::Result<()> {
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_task_end(&mut state, task_id, status, outcome, Utc::now());
    agent.state.upsert_dialogue_state(&state).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn internal_continuation_does_not_rotate_open_request() {
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::DialogueStateStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        let now = Utc::now();
        let mut state = DialogueState::new("test-session");
        state.open_request = Some(OpenRequest {
            user_message_id: "request-1".to_string(),
            text: "Find the specifications and tradeoffs for the synthetic product.".to_string(),
            status: OpenRequestStatus::InProgress,
            task_id: Some("task-1".to_string()),
            project_scope: None,
            semantic_scope: Some(ToolSemanticScope::ExternalRemote),
            completion_contract: None,
            opened_at: now,
            resolved_at: None,
        });
        harness
            .state
            .upsert_dialogue_state(&state)
            .await
            .expect("persist dialogue state");
        let continuation = Message {
            content: Some(
                "Worker output: https://goo.gle/gemini-cli-auth-docs#workspace-gca".to_string(),
            ),
            annotations: vec![crate::traits::MessageAnnotation::InternalContinuation],
            ..Message::new_runtime("continuation-1", "test-session", "user")
        };

        record_dialogue_user_message(&harness.agent, "test-session", &continuation)
            .await
            .expect("record continuation");

        let persisted = harness
            .state
            .get_dialogue_state("test-session")
            .await
            .expect("read dialogue state")
            .expect("dialogue state");
        let open = persisted.open_request.expect("open request remains");
        assert_eq!(open.user_message_id, "request-1");
        assert!(!open.text.contains("workspace-gca"));
        assert!(persisted.last_user_turn.is_none());
    }

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
            completion_contract: None,
            opened_at: Utc::now(),
            resolved_at: None,
        });
        state.open_question = Some(OpenQuestion {
            assistant_message_id: "a1".to_string(),
            text: "Which environment should I use?".to_string(),
            kind: QuestionKind::Clarification,
            related_user_message_id: Some("u1".to_string()),
            mandate_id: None,
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
            completion_contract: None,
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
        apply_semantic_user_turn_assessment(
            &mut state,
            "What times does the tweet posting schedule trigger?",
            UserTurnKind::NewRequest,
            Some(ToolSemanticScope::GoalState),
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
        apply_semantic_user_turn_assessment(
            &mut state,
            "What are your scheduled tasks?",
            UserTurnKind::NewRequest,
            Some(ToolSemanticScope::GoalState),
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
            completion_contract: None,
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
            mandate_id: None,
            awaiting_user_reply: awaiting,
            asked_at,
        }
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
            "the answered question remains auditable in dialogue state"
        );
    }

    #[test]
    fn fresh_answered_request_does_not_capture_unrelated_short_question() {
        let now = Utc::now();
        let answered = request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::minutes(1),
            Some(now - chrono::Duration::minutes(1)),
        );

        let mut standalone = DialogueState::new("standalone");
        standalone.open_request = Some(answered);
        apply_user_message(
            &mut standalone,
            "u2",
            "What is dependency injection?",
            &[],
            now,
        );
        assert_eq!(
            resolved_followup_mode(&standalone, "What is dependency injection?"),
            Some(super::super::followup::FollowupMode::NewTask)
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
    fn fresh_open_question_does_not_capture_imperative_request() {
        // Live 2026-07-12: the makpar clarifying question was only seconds old
        // when "Send me my Microsoft resume" arrived; the non-ack fallthrough
        // captured it as a clarification answer. A self-contained imperative
        // that names its own object supersedes the question instead.
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_question = Some(question_with(now - chrono::Duration::seconds(30), true));
        apply_user_message(&mut state, "u2", "Send me my Microsoft resume", &[], now);
        apply_semantic_user_turn_assessment(
            &mut state,
            "Send me my Microsoft resume",
            UserTurnKind::NewRequest,
            None,
            now,
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
    fn durable_mandate_input_anchors_differently_worded_explanation_followup() {
        let now = Utc::now();
        let mandate_id = "08012d3d-synthetic";
        let mut state = DialogueState::new("owner-session");
        state.open_question = Some(OpenQuestion {
            assistant_message_id: "mandate-run-notice:review-synthetic".to_string(),
            text: "Content-free mandate owner notice".to_string(),
            kind: QuestionKind::MandateInput,
            related_user_message_id: None,
            mandate_id: Some(mandate_id.to_string()),
            awaiting_user_reply: true,
            // Mandate input remains a durable obligation, unlike a generic
            // conversational clarification whose implicit binding expires.
            asked_at: now - chrono::Duration::hours(2),
        });

        apply_user_message(
            &mut state,
            "u-explain",
            "Walk me through the reason.",
            &[],
            now,
        );

        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::ClarificationAnswer)
        );
        let closed = state
            .last_closed_question
            .as_ref()
            .expect("typed mandate input should move to the current turn");
        assert_eq!(closed.kind, QuestionKind::MandateInput);
        assert_eq!(closed.mandate_id.as_deref(), Some(mandate_id));
    }

    #[test]
    fn durable_mandate_input_does_not_capture_explicit_separate_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("owner-session");
        state.open_question = Some(OpenQuestion {
            assistant_message_id: "mandate-run-notice:review-synthetic".to_string(),
            text: "Content-free mandate owner notice".to_string(),
            kind: QuestionKind::MandateInput,
            related_user_message_id: None,
            mandate_id: Some("08012d3d-synthetic".to_string()),
            awaiting_user_reply: true,
            asked_at: now,
        });

        apply_user_message(
            &mut state,
            "u-separate",
            "Check disk space on this Mac.",
            &[],
            now,
        );
        apply_semantic_user_turn_assessment(
            &mut state,
            "Check disk space on this Mac.",
            UserTurnKind::NewRequest,
            Some(ToolSemanticScope::HostLocal),
            now,
        );

        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest)
        );
        assert!(state.last_closed_question.is_none());
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
    fn unresolved_request_anchors_elliptical_question_by_state() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        let mut request = request_with(OpenRequestStatus::PartiallyAnswered, now, None);
        request.text = "Repair the failing job and verify the result".to_string();
        state.open_request = Some(request);

        apply_user_message(&mut state, "u2", "Fixed?", &[], now);

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
    fn active_task_reopens_related_request_without_retry_keywords() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        let mut request = request_with(OpenRequestStatus::PartiallyAnswered, now, None);
        request.text = "Repair the failing job and verify the result".to_string();
        state.open_request = Some(request);

        apply_task_start(&mut state, "task-2", now);
        apply_user_message(
            &mut state,
            "u2",
            "Take another pass from the receipt we already have.",
            &[],
            now,
        );

        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::Followup)
        );
        assert_eq!(
            state.open_request.as_ref().map(|request| request.status),
            Some(OpenRequestStatus::InProgress),
            "typed task activity, not retry vocabulary, reopens the request"
        );
    }

    #[test]
    fn incident_2026_07_11_ack_binds_to_menu_not_stale_request() {
        // Full replay of the incident chain: a request answered two hours ago
        // sits in state, the assistant presents a clarifying menu, the user
        // acks with option numbers. The persisted classifier must bind the
        // answer to that menu rather than resurrecting the stale request.
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::hours(2),
            Some(now - chrono::Duration::hours(2)),
        ));
        apply_assistant_message(&mut state, "a2", MENU, now - chrono::Duration::seconds(90));
        apply_user_message(&mut state, "u2", "Yes do 1, 2", &[], now);

        assert_eq!(
            resolved_followup_mode(&state, "Yes do 1, 2"),
            Some(super::super::followup::FollowupMode::ClarificationAnswer)
        );
        assert_eq!(
            state
                .last_closed_question
                .as_ref()
                .map(|question| question.assistant_message_id.as_str()),
            Some("a2")
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
        apply_task_end(
            &mut state,
            "task-1",
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            Utc::now(),
        );

        assert_eq!(
            state.open_request.as_ref().map(|request| request.status),
            Some(OpenRequestStatus::Answered)
        );
        assert!(state.active_task.is_none());
    }

    #[test]
    fn assistant_blocker_words_do_not_override_successful_task_outcome() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        apply_task_start(&mut state, "task-1", now);
        apply_user_message(&mut state, "u1", "Summarize the incident", &[], now);
        apply_assistant_message(
            &mut state,
            "a1",
            "The report explains why the old queue was blocked and how the failed rows were repaired.",
            now,
        );

        assert_eq!(
            state.open_request.as_ref().map(|request| request.status),
            Some(OpenRequestStatus::InProgress)
        );

        apply_task_end(
            &mut state,
            "task-1",
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            now,
        );
        assert_eq!(
            state.open_request.as_ref().map(|request| request.status),
            Some(OpenRequestStatus::Answered)
        );
    }

    #[test]
    fn partial_task_outcome_keeps_request_unresolved_despite_answer_like_prose() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        apply_task_start(&mut state, "task-1", now);
        apply_user_message(
            &mut state,
            "u1",
            "Increase the synthetic Companion speaker volume.",
            &[],
            now,
        );
        apply_assistant_message(
            &mut state,
            "a1",
            "That control is unavailable in the current interface.",
            now,
        );
        assert_eq!(
            state.open_request.as_ref().map(|request| request.status),
            Some(OpenRequestStatus::InProgress),
            "assistant prose must not commit durable request lifecycle"
        );

        apply_task_end(
            &mut state,
            "task-1",
            TaskStatus::Completed,
            TaskOutcome::Partial,
            now,
        );

        let request = state.open_request.as_ref().expect("open request");
        assert_eq!(request.status, OpenRequestStatus::PartiallyAnswered);
        assert!(request.resolved_at.is_none());
        assert!(state.has_unresolved_request_obligation());

        for (message_id, followup) in [("u2", "Why"), ("u3", "Walk me through what prevented it.")]
        {
            let mut followup_state = state.clone();
            apply_user_message(&mut followup_state, message_id, followup, &[], now);
            assert_eq!(
                followup_state.last_user_turn.as_ref().map(|turn| turn.kind),
                Some(UserTurnKind::Followup),
                "unresolved typed state should bind {followup:?} without depending on its phrasing"
            );
            assert_eq!(
                followup_state
                    .open_request
                    .as_ref()
                    .map(|request| request.user_message_id.as_str()),
                Some("u1")
            );
        }

        let mut separate_state = state;
        apply_user_message(
            &mut separate_state,
            "u4",
            "Check disk space on this Mac.",
            &[],
            now,
        );
        apply_semantic_user_turn_assessment(
            &mut separate_state,
            "Check disk space on this Mac.",
            UserTurnKind::NewRequest,
            Some(ToolSemanticScope::HostLocal),
            now,
        );
        assert_eq!(
            separate_state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest),
            "a concrete separate request must supersede rather than inherit"
        );
    }
}
