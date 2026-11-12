use crate::host::{TurnStores, TurnWorkspace};
use aidaemon_events::{
    AssistantResponseData, EventType, InteractionRequestedData, InteractionResolvedData,
    TaskEndData, TaskOutcome, TaskStartData, TaskStatus, UserMessageData,
};
use aidaemon_events::traits::{
    extract_primary_message_content, message_content_is_structural_only, ActiveTaskRef,
    ActiveTaskStatus, AssistantTurnKind, AssistantTurnSummary, DialogueState, Message,
    OpenQuestion, OpenRequest, OpenRequestStatus, QuestionKind, ToolSemanticScope,
    UserRequestBinding, UserTurnKind, UserTurnSummary,
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

fn infer_open_request_scope(
    backend: &dyn aidaemon_tools::ExecutionBackend,
    text: &str,
    alias_roots: &[String],
) -> Option<ToolSemanticScope> {
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
        if aidaemon_tools::fs_utils::resolve_structural_filesystem_reference(
            backend,
            token,
            alias_roots,
        )
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
/// Relationship assessment receives at most six recent messages. Retaining a
/// wider bounded lineage window lets those IDs survive intervening tool and
/// assistant rows without allowing dialogue state to grow without bound.
const RECENT_REQUEST_BINDING_LIMIT: usize = 32;
/// Owed owner requests kept beside the open request. Oldest entries drop first.
const UNFINISHED_REQUEST_LIMIT: usize = 4;
/// Recent user messages kept so a request's referents can be resolved.
const RECENT_USER_MESSAGE_LIMIT: usize = 6;

/// A request whose owner work has not been answered yet.
fn request_is_owed(request: &OpenRequest) -> bool {
    matches!(
        request.status,
        OpenRequestStatus::Open
            | OpenRequestStatus::InProgress
            | OpenRequestStatus::PartiallyAnswered
            | OpenRequestStatus::Blocked
    )
}

fn bind_user_message_to_request(
    state: &mut DialogueState,
    user_message_id: &str,
    request_user_message_id: &str,
) {
    let user_message_id = user_message_id.trim();
    let request_user_message_id = request_user_message_id.trim();
    if user_message_id.is_empty() || request_user_message_id.is_empty() {
        return;
    }
    state
        .recent_request_bindings
        .retain(|binding| binding.user_message_id != user_message_id);
    state.recent_request_bindings.push(UserRequestBinding {
        user_message_id: user_message_id.to_string(),
        request_user_message_id: request_user_message_id.to_string(),
    });
    let excess = state
        .recent_request_bindings
        .len()
        .saturating_sub(RECENT_REQUEST_BINDING_LIMIT);
    if excess > 0 {
        state.recent_request_bindings.drain(..excess);
    }
}

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
pub fn resolved_followup_mode(
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

/// The owner's request text an untrusted-call verifier must judge a turn
/// against, read from the persisted request lineage rather than from the
/// current message alone.
///
/// A follow-up or clarification answer is judged against the open request it
/// advances, with the follow-up quoted as the owner's refinement. A turn the
/// classifier calls a new request still sees the owner's earlier request when
/// that request is unfinished, so a short continuation such as "try again" is
/// not judged as if the earlier work never existed. The earlier text is quoted
/// context for the verifier, never a grant: the verifier still decides whether
/// the exact proposed call is consistent with what the owner asked for.
///
/// Returns `None` when there is no unfinished owner request to add, in which
/// case the current message is the whole request.
pub fn owner_request_for_verifier(
    state: &DialogueState,
    current_user_text: &str,
    now: chrono::DateTime<Utc>,
) -> Option<String> {
    let current = current_user_text.trim();
    let current_message_id = state
        .last_user_turn
        .as_ref()
        .filter(|turn| turn.text.trim().eq_ignore_ascii_case(current))
        .map(|turn| turn.message_id.as_str());
    // Every owed request in the lineage, oldest first. A single root chosen
    // by timestamp is wrong: continuations are themselves owed requests, so
    // the oldest-by-clock entry can be a bare "try again" that names nothing.
    // Passing the whole owed set keeps the original, content-bearing request
    // visible however the entries were stamped.
    let mut owed: Vec<&OpenRequest> = state
        .open_request
        .iter()
        .chain(state.unfinished_requests.iter())
        .filter(|request| {
            request_is_owed(request)
                && Some(request.user_message_id.as_str()) != current_message_id
                && !request.text.trim().is_empty()
                && !open_request_anchor_expired(request, now)
                && state
                    .last_user_turn
                    .as_ref()
                    .is_none_or(|turn| turn.message_id != request.user_message_id)
        })
        .collect();
    owed.sort_by_key(|request| request.opened_at);
    let mut earlier: Vec<&str> = Vec::new();
    for request in &owed {
        let text = request.text.trim();
        if !earlier.contains(&text) {
            earlier.push(text);
        }
    }
    if earlier.is_empty() {
        return None;
    }
    let neighbours: Vec<String> = state
        .recent_user_messages
        .iter()
        .filter(|text| text.trim() != current)
        .map(|text| format!("- {}", serde_json::Value::from(text.trim())))
        .collect();
    let context = if neighbours.is_empty() {
        String::new()
    } else {
        format!(
            "\n\nRecent messages from the owner in this conversation, oldest first \
             (use them to resolve references such as \"it\" or \"them\"):\n{}",
            neighbours.join("\n")
        )
    };
    let listed = earlier
        .iter()
        .map(|text| format!("- {}", serde_json::Value::from(*text)))
        .collect::<Vec<_>>()
        .join("\n");
    Some(match resolved_followup_mode(state, current_user_text) {
        Some(super::followup::FollowupMode::Followup)
        | Some(super::followup::FollowupMode::ClarificationAnswer) => format!(
            "The owner's unfinished requests in this conversation, oldest first:\n{listed}\n\n\
             The owner's follow-up in the same conversation: {current}{context}"
        ),
        Some(super::followup::FollowupMode::NewTask) | None => format!(
            "{current}\n\nThe owner's earlier requests in this conversation are still \
             unfinished, oldest first, so this message may continue one of them:\n{listed}{context}"
        ),
    })
}

/// The owner's owed requests as continuation candidates for the relationship
/// assessor, oldest first, each under its own stable message id. Continuations
/// of unfinished work are otherwise invisible once the transcript window moves
/// on, and the assessor could only name what it can see.
pub fn owed_request_candidate_rows(
    state: &DialogueState,
    now: chrono::DateTime<Utc>,
) -> Option<String> {
    let mut owed: Vec<&OpenRequest> = state
        .open_request
        .iter()
        .chain(state.unfinished_requests.iter())
        .filter(|request| {
            request_is_owed(request)
                && !request.text.trim().is_empty()
                && !open_request_anchor_expired(request, now)
                && state
                    .last_user_turn
                    .as_ref()
                    .is_none_or(|turn| turn.message_id != request.user_message_id)
        })
        .collect();
    owed.sort_by_key(|request| request.opened_at);
    let mut seen = std::collections::HashSet::new();
    owed.retain(|request| seen.insert(request.user_message_id.clone()));
    if owed.is_empty() {
        return None;
    }
    let rows = owed
        .iter()
        .map(|request| {
            format!(
                "USER message_id=\"{}\" (unfinished request): {}",
                request.user_message_id,
                serde_json::Value::from(request.text.trim())
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    Some(format!(
        "Unfinished owner requests that a continuation may advance (copy the exact message_id):\n{rows}"
    ))
}

/// Canonicalize a recent conversational user-message ID to the live request
/// node it advances. The binding alone is never authority: its target must
/// still be one of the request nodes currently eligible for adoption.
pub fn resolve_request_antecedent<'a>(
    state: &'a DialogueState,
    user_message_id: &str,
) -> Option<&'a OpenRequest> {
    let candidate = user_message_id.trim();
    if candidate.is_empty()
        || state
            .last_user_turn
            .as_ref()
            .is_some_and(|turn| turn.message_id == candidate)
    {
        return None;
    }
    let canonical_id = state
        .recent_request_bindings
        .iter()
        .rev()
        .find(|binding| binding.user_message_id == candidate)
        .map(|binding| binding.request_user_message_id.as_str())
        .unwrap_or(candidate);
    state
        .open_request
        .as_ref()
        .into_iter()
        .chain(state.antecedent_request.as_ref())
        .chain(state.unfinished_requests.iter())
        .find(|request| request.user_message_id == canonical_id)
}

/// Resolve the only structurally possible antecedent for the current user
/// turn. This is used when the semantic assessor identifies a continuation
/// but omits the optional stable ID. The fallback is deliberately narrow: it
/// may select only a persisted request node other than the current ingress
/// message, and it refuses ambiguity. The relationship still comes from the
/// semantic assessment; no request wording is classified here.
pub fn unambiguous_request_antecedent(
    state: &DialogueState,
) -> Option<&OpenRequest> {
    let current_message_id = state
        .last_user_turn
        .as_ref()
        .map(|turn| turn.message_id.as_str());
    // One request can sit in several slots (the open request, its antecedent
    // and the owed list); it is one candidate, not an ambiguity.
    let mut distinct: Vec<&OpenRequest> = Vec::new();
    for request in state
        .open_request
        .as_ref()
        .into_iter()
        .chain(state.antecedent_request.as_ref())
        .chain(state.unfinished_requests.iter())
        .filter(|request| Some(request.user_message_id.as_str()) != current_message_id)
    {
        if !distinct
            .iter()
            .any(|known| known.user_message_id == request.user_message_id)
        {
            distinct.push(request);
        }
    }
    match distinct.as_slice() {
        [only] => Some(*only),
        _ => None,
    }
}

fn classify_user_turn(
    backend: &dyn aidaemon_tools::ExecutionBackend,
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
        let separate_scoped_request =
            infer_open_request_scope(backend, trimmed, alias_roots).is_some();
        question.awaiting_user_reply
            && !separate_scoped_request
            && (durable_mandate_input
                || observed_at.signed_duration_since(question.asked_at)
                    <= chrono::Duration::minutes(OPEN_QUESTION_IMPLICIT_BIND_TTL_MINUTES))
    }) {
        return UserTurnKind::ClarificationAnswer;
    }

    // A prior unresolved request is a candidate antecedent, not evidence that
    // this message continues it. Ingress always opens a provisional request
    // and retains the candidate as a typed node. Only an explicit pending
    // interaction above or a validated semantic relationship may adopt it.
    // This fail-fresh default prevents stale policy, scopes, and obligations
    // from leaking into an unrelated request when assessment is unavailable.
    UserTurnKind::NewRequest
}

fn classify_assistant_turn_text(text: &str) -> (AssistantTurnKind, bool) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return (AssistantTurnKind::SystemNotice, false);
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
        let semantic_followup = state
            .last_user_turn
            .as_ref()
            .is_some_and(|turn| turn.kind == UserTurnKind::Followup);
        if open_request.status == OpenRequestStatus::Open
            || semantic_followup
                && matches!(
                    open_request.status,
                    OpenRequestStatus::Answered
                        | OpenRequestStatus::Superseded
                        | OpenRequestStatus::Blocked
                        | OpenRequestStatus::PartiallyAnswered
                )
        {
            open_request.status = OpenRequestStatus::InProgress;
            open_request.resolved_at = None;
        }
    }
    state.touch();
}

fn apply_user_message(
    backend: &dyn aidaemon_tools::ExecutionBackend,
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
    // Ingestion runs once per user message. The same message can reach this
    // point again (bootstrap records it, then turn-context assembly does); a
    // second pass must not re-open, re-classify or re-queue the request.
    if state
        .last_user_turn
        .as_ref()
        .is_some_and(|turn| turn.message_id == message_id)
    {
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
    state
        .unfinished_requests
        .retain(|request| !open_request_anchor_expired(request, observed_at));
    state.recent_user_messages.push(trimmed.to_string());
    let excess = state
        .recent_user_messages
        .len()
        .saturating_sub(RECENT_USER_MESSAGE_LIMIT);
    state.recent_user_messages.drain(..excess);
    let turn_kind = classify_user_turn(backend, state, trimmed, alias_roots, observed_at);
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
            // Owed work is not answered by a later message. Keep it in the
            // bounded owed list before the open slot is superseded, so a
            // continuation can still reach the owner's earlier request.
            if let Some(open_request) = state.open_request.as_ref() {
                if request_is_owed(open_request)
                    && !state
                        .unfinished_requests
                        .iter()
                        .any(|owed| owed.user_message_id == open_request.user_message_id)
                {
                    state.unfinished_requests.push(open_request.clone());
                    let excess = state
                        .unfinished_requests
                        .len()
                        .saturating_sub(UNFINISHED_REQUEST_LIMIT);
                    state.unfinished_requests.drain(..excess);
                }
            }
            if let Some(open_request) = state.open_request.as_mut() {
                if !matches!(
                    open_request.status,
                    OpenRequestStatus::Answered | OpenRequestStatus::Superseded
                ) {
                    open_request.status = OpenRequestStatus::Superseded;
                    open_request.resolved_at = Some(observed_at);
                }
            }
            // This relationship is provisional until semantic task
            // assessment. Keep the exact prior request and its typed contract
            // so a genuine continuation can restore it without reconstructing
            // policy or obligations from prose.
            state.antecedent_request = state.open_request.take();
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
                completion_contract: None,
                opened_at: observed_at,
                resolved_at: None,
            });
        }
        UserTurnKind::Courtesy | UserTurnKind::Unknown => {}
    }

    if matches!(
        turn_kind,
        UserTurnKind::NewRequest | UserTurnKind::Followup | UserTurnKind::ClarificationAnswer
    ) {
        let request_user_message_id = state
            .open_request
            .as_ref()
            .map(|request| request.user_message_id.clone())
            .or_else(|| {
                state
                    .last_closed_question
                    .as_ref()
                    .and_then(|question| question.related_user_message_id.clone())
            });
        if let Some(request_user_message_id) = request_user_message_id {
            bind_user_message_to_request(state, message_id, &request_user_message_id);
        }
    }

    state.touch();
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
                state.antecedent_request = state.open_request.take();
                state.open_request = Some(OpenRequest {
                    user_message_id: message_id.clone(),
                    text: text.trim().to_string(),
                    status: if state.active_task.is_some() {
                        OpenRequestStatus::InProgress
                    } else {
                        OpenRequestStatus::Open
                    },
                    task_id: state.active_task.as_ref().map(|task| task.task_id.clone()),
                    project_scope: None,
                    completion_contract: None,
                    opened_at: observed_at,
                    resolved_at: None,
                });
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
            // A topology-only ingress classification may have provisionally
            // opened a new request. The semantic relationship names this turn
            // as a continuation, so restore the retained typed antecedent
            // atomically before any task is bound to it.
            if state
                .open_request
                .as_ref()
                .is_some_and(|request| request.user_message_id == message_id)
            {
                if let Some(antecedent) = state.antecedent_request.take() {
                    state.open_request = Some(antecedent);
                }
            }
            if let Some(request) = state.open_request.as_mut() {
                if state.active_task.is_some() {
                    request.status = OpenRequestStatus::InProgress;
                    request.resolved_at = None;
                }
            }
        }
        UserTurnKind::Courtesy | UserTurnKind::Unknown => {}
    }
    if matches!(
        kind,
        UserTurnKind::NewRequest | UserTurnKind::Followup | UserTurnKind::ClarificationAnswer
    ) {
        let request_user_message_id = state
            .open_request
            .as_ref()
            .map(|request| request.user_message_id.clone())
            .or_else(|| {
                state
                    .last_closed_question
                    .as_ref()
                    .and_then(|question| question.related_user_message_id.clone())
            });
        if let Some(request_user_message_id) = request_user_message_id {
            bind_user_message_to_request(state, &message_id, &request_user_message_id);
        }
    }
    state.touch();
}

pub async fn record_dialogue_semantic_user_turn(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
    session_id: &str,
    text: &str,
    relationship: &str,
) -> anyhow::Result<Option<UserTurnKind>> {
    let Some(kind) = parse_semantic_relationship(relationship) else {
        return Ok(None);
    };
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_semantic_user_turn_assessment(&mut state, text, kind, Utc::now());
    agent.dialogue_states().upsert_dialogue_state(&state).await?;
    Ok(Some(kind))
}

fn apply_assistant_message(
    state: &mut DialogueState,
    message_id: &str,
    content: &str,
    _observed_at: chrono::DateTime<Utc>,
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
        // Questions are lifecycle state only when a typed interaction broker
        // creates them. Assistant prose alone cannot open an obligation.
        AssistantTurnKind::ClarificationQuestion => {}
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

fn apply_interaction_requested(
    state: &mut DialogueState,
    interaction: InteractionRequestedData,
    observed_at: chrono::DateTime<Utc>,
) {
    state.open_question = Some(OpenQuestion {
        assistant_message_id: interaction.interaction_id,
        text: interaction.action,
        kind: QuestionKind::Approval,
        related_user_message_id: state
            .open_request
            .as_ref()
            .map(|request| request.user_message_id.clone()),
        mandate_id: None,
        awaiting_user_reply: true,
        asked_at: observed_at,
    });
    state.last_closed_question = None;
    state.touch();
}

fn apply_interaction_resolved(state: &mut DialogueState, interaction: InteractionResolvedData) {
    if state.open_question.as_ref().is_some_and(|question| {
        question.kind == QuestionKind::Approval
            && question.assistant_message_id == interaction.interaction_id
    }) {
        state.last_closed_question = state.open_question.take();
        state.touch();
    }
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
                                // A continuation that completes successfully
                                // discharges the owed work of its chain: the
                                // owner followed up on the same work and it
                                // is done. A new, unrelated request does not
                                // discharge anything; owed entries still expire.
                                if state
                                    .last_user_turn
                                    .as_ref()
                                    .is_some_and(|turn| turn.kind == UserTurnKind::Followup)
                                {
                                    state.unfinished_requests.clear();
                                }
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

pub async fn get_or_rebuild_dialogue_state(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
    session_id: &str,
) -> DialogueState {
    match agent.dialogue_states().get_dialogue_state(session_id).await {
        Ok(Some(state)) if state.schema_version == DialogueState::SCHEMA_VERSION => state,
        Ok(_) | Err(_) => {
            let rebuilt = rebuild_dialogue_state_from_events(agent, session_id).await;
            if let Err(err) = agent.dialogue_states().upsert_dialogue_state(&rebuilt).await {
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

async fn rebuild_dialogue_state_from_events(agent: &(impl TurnStores + TurnWorkspace + ?Sized), session_id: &str) -> DialogueState {
    let mut state = DialogueState::new(session_id);
    let events = agent
        .events()
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
                        .contains(&aidaemon_events::traits::MessageAnnotation::InternalContinuation)
                    {
                        continue;
                    }
                    apply_user_message(
                        agent.execution_backend(),
                        &mut state,
                        data.message_id.as_deref().unwrap_or_default(),
                        &data.content,
                        agent.project_roots(),
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
            EventType::InteractionRequested => {
                if let Ok(data) = event.parse_data::<InteractionRequestedData>() {
                    apply_interaction_requested(&mut state, data, event.created_at);
                }
            }
            EventType::InteractionResolved => {
                if let Ok(data) = event.parse_data::<InteractionResolvedData>() {
                    apply_interaction_resolved(&mut state, data);
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

pub async fn record_dialogue_task_start(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
    session_id: &str,
    task_id: &str,
) -> anyhow::Result<()> {
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_task_start(&mut state, task_id, Utc::now());
    agent.dialogue_states().upsert_dialogue_state(&state).await
}

pub async fn record_dialogue_user_message(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
    session_id: &str,
    message: &Message,
) -> anyhow::Result<()> {
    if message
        .effective_annotations()
        .contains(&aidaemon_events::traits::MessageAnnotation::InternalContinuation)
    {
        return Ok(());
    }
    let Some(content) = message.primary_content() else {
        return Ok(());
    };
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_user_message(
        agent.execution_backend(),
        &mut state,
        &message.id,
        &content,
        agent.project_roots(),
        message.created_at,
    );
    agent.dialogue_states().upsert_dialogue_state(&state).await
}

/// Attach the finalized typed completion obligations to the currently open
/// request. This lets later continuations inherit the contract without
/// reclassifying the original wording or depending on assessment availability.
pub async fn record_dialogue_completion_contract(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
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
        agent.dialogue_states().upsert_dialogue_state(&state).await?;
    }
    Ok(())
}

pub async fn record_dialogue_assistant_message(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
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
    agent.dialogue_states().upsert_dialogue_state(&state).await
}

/// Persist a delivered mandate ASK as a typed owner-input obligation.
///
/// The visible notice intentionally contains no deliberator-authored question,
/// so prose classification cannot safely reconstruct this relationship. Only
/// authenticated runtime identifiers are stored here; mandate-local text must
/// still be retrieved through `manage_mandates(get)` on the related turn.
pub async fn record_mandate_owner_input(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
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
    agent.dialogue_states().upsert_dialogue_state(&state).await
}

pub async fn record_dialogue_task_end(
    agent: &(impl TurnStores + TurnWorkspace + ?Sized),
    session_id: &str,
    task_id: &str,
    status: TaskStatus,
    outcome: TaskOutcome,
) -> anyhow::Result<()> {
    let mut state = get_or_rebuild_dialogue_state(agent, session_id).await;
    apply_task_end(&mut state, task_id, status, outcome, Utc::now());
    agent.dialogue_states().upsert_dialogue_state(&state).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_user_message(
        state: &mut DialogueState,
        message_id: &str,
        text: &str,
        alias_roots: &[String],
        observed_at: chrono::DateTime<Utc>,
    ) {
        super::apply_user_message(
            &*aidaemon_tools::testing::active_execution_backend(),
            state,
            message_id,
            text,
            alias_roots,
            observed_at,
        );
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
    fn unresolved_request_is_retained_as_candidate_not_assumed_followup() {
        let mut state = DialogueState::new("s1");
        state.open_request = Some(OpenRequest {
            user_message_id: "u1".to_string(),
            text: "What were the latency regressions?".to_string(),
            status: OpenRequestStatus::Open,
            task_id: None,
            project_scope: None,
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
            Some(UserTurnKind::NewRequest)
        );
        assert_eq!(
            state
                .antecedent_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u1")
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
    fn semantic_continuation_restores_exact_provisional_antecedent() {
        let now = Utc::now();
        let mut state = DialogueState::new("synthetic-session");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::minutes(1),
            Some(now - chrono::Duration::seconds(30)),
        ));

        apply_user_message(
            &mut state,
            "u2",
            "Return the values from that setup.",
            &[],
            now,
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u2")
        );
        assert_eq!(
            resolve_request_antecedent(&state, "u1")
                .map(|request| request.user_message_id.as_str()),
            Some("u1")
        );

        apply_semantic_user_turn_assessment(
            &mut state,
            "Return the values from that setup.",
            UserTurnKind::Followup,
            now,
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u1"),
            "semantic adoption restores the exact typed request rather than reparsing prose"
        );
        assert!(state.antecedent_request.is_none());
    }

    #[test]
    fn chained_followup_message_resolves_to_canonical_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("synthetic-session");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::minutes(2),
            Some(now - chrono::Duration::minutes(1)),
        ));

        apply_user_message(
            &mut state,
            "u2",
            "Are there any uncommitted files?",
            &[],
            now,
        );
        apply_semantic_user_turn_assessment(
            &mut state,
            "Are there any uncommitted files?",
            UserTurnKind::Followup,
            now,
        );
        assert_eq!(
            state
                .recent_request_bindings
                .iter()
                .find(|binding| binding.user_message_id == "u2")
                .map(|binding| binding.request_user_message_id.as_str()),
            Some("u1"),
            "the follow-up message must remain a typed alias of its root request"
        );

        apply_user_message(
            &mut state,
            "u3",
            "How can that be possible?",
            &[],
            now + chrono::Duration::seconds(1),
        );
        assert_eq!(
            resolve_request_antecedent(&state, "u2")
                .map(|request| request.user_message_id.as_str()),
            Some("u1"),
            "a second-order follow-up must resolve through the prior follow-up"
        );

        apply_semantic_user_turn_assessment(
            &mut state,
            "How can that be possible?",
            UserTurnKind::Followup,
            now + chrono::Duration::seconds(1),
        );
        assert_eq!(
            state
                .open_request
                .as_ref()
                .map(|request| request.user_message_id.as_str()),
            Some("u1")
        );
        assert_eq!(
            state
                .recent_request_bindings
                .iter()
                .find(|binding| binding.user_message_id == "u3")
                .map(|binding| binding.request_user_message_id.as_str()),
            Some("u1")
        );
    }

    #[test]
    fn stale_followup_binding_cannot_revive_an_ineligible_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("synthetic-session");
        state.open_request = Some(request_with(OpenRequestStatus::Answered, now, Some(now)));
        bind_user_message_to_request(&mut state, "u2", "u1");

        let mut next = request_with(OpenRequestStatus::Open, now, None);
        next.user_message_id = "u3".to_string();
        state.open_request = Some(next);
        state.antecedent_request = None;
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "u4".to_string(),
            kind: UserTurnKind::NewRequest,
            text: "Inspect a separate synthetic project.".to_string(),
        });

        assert!(
            resolve_request_antecedent(&state, "u2").is_none(),
            "lineage may canonicalize identity but must not grant eligibility"
        );
    }

    #[test]
    fn antecedent_resolution_requires_one_persisted_candidate() {
        let now = Utc::now();
        let mut state = DialogueState::new("synthetic-session");
        state.open_request = Some(request_with(OpenRequestStatus::Open, now, None));
        assert_eq!(
            unambiguous_request_antecedent(&state).map(|request| request.user_message_id.as_str()),
            Some("u1")
        );

        let mut second = request_with(OpenRequestStatus::Open, now, None);
        second.user_message_id = "u0".to_string();
        state.antecedent_request = Some(second);
        assert!(
            unambiguous_request_antecedent(&state).is_none(),
            "semantic confidence cannot choose between multiple graph nodes"
        );
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
        // yet "Send me my acme resume" kept anchoring to it as a followup,
        // so the open_request never rotated.
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::hours(2),
            Some(now - chrono::Duration::hours(2)),
        ));
        apply_user_message(&mut state, "u2", "Send me my acme resume", &[], now);
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
        // Live 2026-07-12: the acme clarifying question was only seconds old
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
        apply_user_message(&mut state, "u2", "Send me my acme resume", &[], now);
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
            now,
        );

        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest)
        );
        assert!(state.last_closed_question.is_none());
    }

    #[test]
    fn semantic_edge_anchors_short_followup_to_retained_candidate() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        let mut request = request_with(OpenRequestStatus::Open, now, None);
        request.text = "Find my tax documents".to_string();
        state.open_request = Some(request);
        apply_user_message(&mut state, "u2", "hmm the second folder maybe", &[], now);
        assert_eq!(
            state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest)
        );
        apply_semantic_user_turn_assessment(
            &mut state,
            "hmm the second folder maybe",
            UserTurnKind::Followup,
            now,
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
    fn semantic_edge_anchors_elliptical_question_to_unresolved_request() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        let mut request = request_with(OpenRequestStatus::PartiallyAnswered, now, None);
        request.text = "Repair the failing job and verify the result".to_string();
        state.open_request = Some(request);

        apply_user_message(&mut state, "u2", "Fixed?", &[], now);
        apply_semantic_user_turn_assessment(&mut state, "Fixed?", UserTurnKind::Followup, now);

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
        apply_semantic_user_turn_assessment(
            &mut state,
            "Take another pass from the receipt we already have.",
            UserTurnKind::Followup,
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
    fn typed_interaction_binds_reply_without_classifying_assistant_prose() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        state.open_request = Some(request_with(
            OpenRequestStatus::Answered,
            now - chrono::Duration::hours(2),
            Some(now - chrono::Duration::hours(2)),
        ));
        apply_interaction_requested(
            &mut state,
            InteractionRequestedData {
                schema_version: InteractionRequestedData::SCHEMA_VERSION,
                interaction_id: "a2".to_string(),
                interaction_kind: "approval".to_string(),
                action: MENU.to_string(),
                action_sha256: "synthetic".to_string(),
                risk_level: "low".to_string(),
                warnings: Vec::new(),
                permission_mode: "once".to_string(),
            },
            now - chrono::Duration::seconds(90),
        );
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
    fn assistant_question_punctuation_does_not_create_lifecycle_state() {
        let now = Utc::now();
        let mut state = DialogueState::new("s1");
        apply_assistant_message(
            &mut state,
            "a1",
            "Which synthetic environment should be used?",
            now,
        );
        assert!(state.open_question.is_none());
        assert_eq!(
            state.last_assistant_turn.as_ref().map(|turn| turn.kind),
            Some(AssistantTurnKind::SubstantiveAnswer)
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
                Some(UserTurnKind::NewRequest),
                "ingress must not infer a relationship from unresolved state"
            );
            apply_semantic_user_turn_assessment(
                &mut followup_state,
                followup,
                UserTurnKind::Followup,
                now,
            );
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
            now,
        );
        assert_eq!(
            separate_state.last_user_turn.as_ref().map(|turn| turn.kind),
            Some(UserTurnKind::NewRequest),
            "a concrete separate request must supersede rather than inherit"
        );
    }
}

#[cfg(test)]
mod owner_lineage_tests {
    use super::*;

    fn request(status: OpenRequestStatus, text: &str) -> OpenRequest {
        OpenRequest {
            user_message_id: "request-1".to_string(),
            text: text.to_string(),
            status,
            task_id: Some("task-1".to_string()),
            project_scope: None,
            completion_contract: None,
            opened_at: Utc::now(),
            resolved_at: None,
        }
    }

    fn state_with(
        open: OpenRequest,
        current_id: &str,
        current_text: &str,
        kind: UserTurnKind,
    ) -> DialogueState {
        let mut state = DialogueState::new("session-1");
        state.open_request = Some(open);
        state.last_user_turn = Some(UserTurnSummary {
            message_id: current_id.to_string(),
            kind,
            text: current_text.to_string(),
        });
        state
    }

    const EARLIER: &str = "Remove the three incorrect entries from the synthetic list.";

    #[test]
    fn followup_is_judged_against_the_open_request_it_advances() {
        let state = state_with(
            request(OpenRequestStatus::PartiallyAnswered, EARLIER),
            "msg-2",
            "Try again",
            UserTurnKind::Followup,
        );
        let origin = owner_request_for_verifier(&state, "Try again", Utc::now())
            .expect("unfinished request is the verifier's origin");
        assert!(origin.contains(EARLIER), "{origin}");
        assert!(origin.contains("Try again"), "{origin}");
    }

    #[test]
    fn new_task_still_sees_an_unfinished_earlier_request() {
        // A classifier that calls a short continuation a new request must not
        // strip the owner's unfinished request from the verifier's view.
        let state = state_with(
            request(OpenRequestStatus::PartiallyAnswered, EARLIER),
            "msg-2",
            "Try again",
            UserTurnKind::NewRequest,
        );
        let origin = owner_request_for_verifier(&state, "Try again", Utc::now())
            .expect("unfinished request is still context");
        assert!(origin.starts_with("Try again"), "{origin}");
        assert!(origin.contains(EARLIER), "{origin}");
    }

    #[test]
    fn answered_request_adds_nothing() {
        let state = state_with(
            request(OpenRequestStatus::Answered, EARLIER),
            "msg-2",
            "Try again",
            UserTurnKind::NewRequest,
        );
        assert_eq!(
            owner_request_for_verifier(&state, "Try again", Utc::now()),
            None
        );
    }

    #[test]
    fn the_current_message_is_never_its_own_antecedent() {
        let state = state_with(
            request(OpenRequestStatus::InProgress, EARLIER),
            "request-1",
            EARLIER,
            UserTurnKind::NewRequest,
        );
        assert_eq!(
            owner_request_for_verifier(&state, EARLIER, Utc::now()),
            None
        );
    }

    #[test]
    fn continuation_resolves_to_the_earliest_owed_request_not_the_latest() {
        // The owner's CloudPets request was never answered. A later "try
        // again" became the open request, and a further "try again" must
        // still be judged against the original request.
        let now = Utc::now();
        let mut original = request(OpenRequestStatus::PartiallyAnswered, EARLIER);
        original.user_message_id = "request-1".to_string();
        original.opened_at = now - chrono::Duration::minutes(5);
        let mut latest = request(OpenRequestStatus::PartiallyAnswered, "Try again");
        latest.user_message_id = "request-2".to_string();
        latest.opened_at = now - chrono::Duration::minutes(1);
        let mut state = DialogueState::new("session-1");
        state.open_request = Some(latest);
        state.unfinished_requests = vec![original];
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "msg-3".to_string(),
            kind: UserTurnKind::NewRequest,
            text: "Try again".to_string(),
        });
        let origin = owner_request_for_verifier(&state, "Try again", now)
            .expect("owed work is the verifier's origin");
        assert!(origin.contains(EARLIER), "{origin}");
    }

    #[test]
    fn original_request_stays_visible_when_a_bare_continuation_is_older() {
        // A repaired or late-recorded original request can carry a newer
        // timestamp than an older bare "try again" entry. Both must reach the
        // verifier, so the original's words are never hidden by ordering.
        let now = Utc::now();
        let mut bare = request(OpenRequestStatus::PartiallyAnswered, "Try again");
        bare.user_message_id = "request-2".to_string();
        bare.opened_at = now - chrono::Duration::minutes(30);
        let mut original = request(OpenRequestStatus::PartiallyAnswered, EARLIER);
        original.user_message_id = "request-1".to_string();
        original.opened_at = now - chrono::Duration::minutes(1);
        let mut state = DialogueState::new("session-1");
        state.unfinished_requests = vec![bare, original];
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "msg-9".to_string(),
            kind: UserTurnKind::NewRequest,
            text: "Try again".to_string(),
        });
        let origin = owner_request_for_verifier(&state, "Try again", now)
            .expect("owed work is the verifier's origin");
        assert!(origin.contains(EARLIER), "{origin}");
    }

    #[test]
    fn owed_request_carries_the_owners_recent_referents() {
        // "Remove it from memory" names nothing on its own. The owner's
        // neighbouring messages are what "it" refers to, so the verifier must
        // see them with the owed request.
        let now = Utc::now();
        let mut owed = request(
            OpenRequestStatus::PartiallyAnswered,
            "Remove it from memory",
        );
        owed.user_message_id = "request-1".to_string();
        owed.opened_at = now;
        let mut state = DialogueState::new("session-1");
        state.unfinished_requests = vec![owed];
        state.recent_user_messages = vec![
            "Cloud pets?".to_string(),
            "Isn't a project of mine".to_string(),
            "Remove it from memory".to_string(),
            "Try again".to_string(),
        ];
        state.last_user_turn = Some(UserTurnSummary {
            message_id: "msg-9".to_string(),
            kind: UserTurnKind::NewRequest,
            text: "Try again".to_string(),
        });
        let origin = owner_request_for_verifier(&state, "Try again", now)
            .expect("owed work is the verifier's origin");
        assert!(origin.contains("Cloud pets?"), "{origin}");
        assert!(origin.contains("Isn't a project of mine"), "{origin}");
    }

    #[test]
    fn a_successful_continuation_discharges_its_chain_but_a_new_request_does_not() {
        let now = Utc::now();
        let owed = |id: &str, text: &str| {
            let mut request = request(OpenRequestStatus::PartiallyAnswered, text);
            request.user_message_id = id.to_string();
            request.opened_at = now;
            request
        };
        let mut continuation = state_with(
            owed("request-2", "Try again"),
            "msg-3",
            "Try again",
            UserTurnKind::Followup,
        );
        continuation.open_request.as_mut().unwrap().status = OpenRequestStatus::InProgress;
        continuation.open_request.as_mut().unwrap().task_id = Some("task-3".to_string());
        continuation.active_task = Some(ActiveTaskRef {
            task_id: "task-3".to_string(),
            status: ActiveTaskStatus::Running,
            started_at: now,
        });
        continuation.unfinished_requests = vec![owed("request-1", EARLIER)];
        apply_task_end(
            &mut continuation,
            "task-3",
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            now,
        );
        assert!(
            continuation.unfinished_requests.is_empty(),
            "continuation discharges owed work"
        );

        let mut unrelated = continuation.clone();
        unrelated.unfinished_requests = vec![owed("request-1", EARLIER)];
        unrelated.last_user_turn.as_mut().unwrap().kind = UserTurnKind::NewRequest;
        unrelated.open_request.as_mut().unwrap().status = OpenRequestStatus::InProgress;
        unrelated.open_request.as_mut().unwrap().resolved_at = None;
        apply_task_end(
            &mut unrelated,
            "task-3",
            TaskStatus::Completed,
            TaskOutcome::Succeeded,
            now,
        );
        assert_eq!(
            unrelated.unfinished_requests.len(),
            1,
            "a new request leaves owed work alone"
        );
    }

    #[test]
    fn the_current_message_is_never_its_own_continuation_candidate() {
        let now = Utc::now();
        let mut state = DialogueState::new("session-1".to_string());
        let request = |id: &str, text: &str| OpenRequest {
            user_message_id: id.to_string(),
            text: text.to_string(),
            status: OpenRequestStatus::Open,
            task_id: None,
            project_scope: None,
            completion_contract: None,
            opened_at: now,
            resolved_at: None,
        };
        state.open_request = Some(request("message-current", "the provisional request"));
        state.last_user_turn = Some(aidaemon_events::traits::UserTurnSummary {
            message_id: "message-current".to_string(),
            kind: UserTurnKind::NewRequest,
            text: "the provisional request".to_string(),
        });
        assert!(owed_request_candidate_rows(&state, now).is_none());

        state
            .unfinished_requests
            .push(request("message-earlier", "the earlier unfinished request"));
        let rows = owed_request_candidate_rows(&state, now).expect("the earlier request is owed");
        assert!(rows.contains("message-earlier"));
        assert!(!rows.contains("message-current"));
    }

    #[test]
    fn an_owed_request_is_a_resolvable_continuation_target() {
        // The assessor may name an owed request by its message id. Resolution
        // must find it; otherwise a correct continuation is downgraded to a new
        // request for lack of an antecedent.
        let now = Utc::now();
        let mut owed = request(OpenRequestStatus::PartiallyAnswered, EARLIER);
        owed.user_message_id = "request-1".to_string();
        owed.opened_at = now;
        let mut state = DialogueState::new("session-1");
        state.open_request = Some(request(OpenRequestStatus::Answered, "Unrelated"));
        state.open_request.as_mut().unwrap().user_message_id = "request-9".to_string();
        state.unfinished_requests = vec![owed];
        let found = resolve_request_antecedent(&state, "request-1")
            .expect("owed request must resolve by id");
        assert_eq!(found.text, EARLIER);
    }
}
