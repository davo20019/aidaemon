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

    if [
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

fn user_reply_likely_answers_open_question(trimmed: &str, lower: &str) -> bool {
    let concise_ack_like = trimmed.chars().count() <= 120
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
            || contains_keyword_as_words(lower, "use that"));

    concise_ack_like
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

fn classify_user_turn(state: &DialogueState, text: &str, alias_roots: &[String]) -> UserTurnKind {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return UserTurnKind::Unknown;
    }
    let lower = trimmed.to_ascii_lowercase();

    if is_courtesy_only(&lower) {
        return UserTurnKind::Courtesy;
    }

    if state
        .open_question
        .as_ref()
        .is_some_and(|question| question.awaiting_user_reply)
        && user_reply_likely_answers_open_question(trimmed, &lower)
    {
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
    let lower = trimmed.to_ascii_lowercase();
    let turn_kind = classify_user_turn(state, trimmed, alias_roots);
    state.last_user_turn = Some(UserTurnSummary {
        message_id: message_id.to_string(),
        kind: turn_kind,
        text: trimmed.to_string(),
    });

    match turn_kind {
        UserTurnKind::ClarificationAnswer => {
            state.open_question = None;
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

impl Agent {
    pub(super) async fn get_or_rebuild_dialogue_state(&self, session_id: &str) -> DialogueState {
        match self.state.get_dialogue_state(session_id).await {
            Ok(Some(state)) if state.schema_version == DialogueState::SCHEMA_VERSION => state,
            Ok(_) | Err(_) => {
                let rebuilt = self.rebuild_dialogue_state_from_events(session_id).await;
                if let Err(err) = self.state.upsert_dialogue_state(&rebuilt).await {
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

    async fn rebuild_dialogue_state_from_events(&self, session_id: &str) -> DialogueState {
        let mut state = DialogueState::new(session_id);
        let events = self
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
                            &self.path_aliases.projects,
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

    pub(super) async fn record_dialogue_task_start(
        &self,
        session_id: &str,
        task_id: &str,
    ) -> anyhow::Result<()> {
        let mut state = self.get_or_rebuild_dialogue_state(session_id).await;
        apply_task_start(&mut state, task_id, Utc::now());
        self.state.upsert_dialogue_state(&state).await
    }

    pub(super) async fn record_dialogue_user_message(
        &self,
        session_id: &str,
        message: &Message,
    ) -> anyhow::Result<()> {
        let Some(content) = message.primary_content() else {
            return Ok(());
        };
        let mut state = self.get_or_rebuild_dialogue_state(session_id).await;
        apply_user_message(
            &mut state,
            &message.id,
            &content,
            &self.path_aliases.projects,
            message.created_at,
        );
        self.state.upsert_dialogue_state(&state).await
    }

    pub(super) async fn record_dialogue_assistant_message(
        &self,
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

        let mut state = self.get_or_rebuild_dialogue_state(session_id).await;
        apply_assistant_message(&mut state, &message.id, &primary, message.created_at);
        self.state.upsert_dialogue_state(&state).await
    }

    pub(super) async fn record_dialogue_task_end(
        &self,
        session_id: &str,
        task_id: &str,
        status: TaskStatus,
    ) -> anyhow::Result<()> {
        let mut state = self.get_or_rebuild_dialogue_state(session_id).await;
        apply_task_end(&mut state, task_id, status, Utc::now());
        self.state.upsert_dialogue_state(&state).await
    }
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
