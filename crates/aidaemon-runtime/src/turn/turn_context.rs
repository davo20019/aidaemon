//! Turn-context assembly: recent-history summarization and the
//! `build_turn_context_from_recent_history` entry point that produces a
//! [`TurnContext`] for each turn, plus the session history appends every
//! turn writes through ([`TurnHistory`], on any [`TurnHost`]).

use std::sync::atomic::Ordering;

use aidaemon_channels::context::ChannelContext;
use aidaemon_events::conversation::{Message, ToolCall};
use aidaemon_events::traits::{ToolCallAccessManifest, ToolCallSemantics};
use aidaemon_events::{AssistantResponseData, EventType, ToolCallInfo, ToolResultData};
use futures::future::BoxFuture;
use serde_json::{json, Value};

use crate::host::TurnHost;
use crate::markers::INTENT_GATE_MARKER;
use crate::policy::metrics::POLICY_METRICS;
use crate::turn::completion_contract::{
    completion_contract_from_persisted, infer_structural_completion_contract_anchored,
    inherit_unfinished_request_contract, scope_contract_for_delegated_executor, CompletionContract,
};
use crate::turn::followup::{
    find_previous_turns, has_project_scope_divergence_with_aliases, sanitize_carryover_blocks,
    FollowupMode, TurnContextReason,
};
use crate::turn::helpers::truncate_for_resume;
use crate::turn::project_scope::{
    choose_primary_project_scope, drop_inbox_scopes,
    extract_explicit_path_scopes_from_text_anchored, extract_project_scopes_from_history_anchored,
    resolve_primary_project_scope, unify_current_turn_scopes,
};

#[derive(Debug, Clone, Default)]
pub struct TurnContext {
    #[allow(dead_code)] // Retained as an inspectable trace of contract-inference input.
    pub goal_user_text: String,
    /// Bounded prior messages exposed only to the relationship assessor. They
    /// carry stable message IDs so a continuation must select an exact
    /// antecedent. They never enter a fresh task's provider transcript.
    pub assessment_recent_messages: Vec<Value>,
    /// Exact prior user message selected by the semantic relationship
    /// assessment. This binds any provider-visible conversational evidence to
    /// one persisted antecedent instead of treating the whole recent tail as
    /// interchangeable proof.
    pub visible_antecedent_user_message_id: Option<String>,
    /// Typed durable workspace binding for this turn: the goal's bound project
    /// scope, or the workspace a parent handed to a spawned worker (including
    /// an executor's provisioned attempt workspace). Both are runtime state,
    /// not request prose. When present it anchors every relative path
    /// reference in the request and is the turn's filesystem authority
    /// boundary.
    pub bound_workspace: Option<String>,
    /// Where `bound_workspace` came from: only an explicit binding is a
    /// trusted root (`explicit_workspace`).
    pub bound_workspace_origin: Option<crate::turn::project_scope::WorkspaceBindingOrigin>,
    pub primary_project_scope: Option<String>,
    /// Exact path authorities named by the current request (or inherited from
    /// its structurally-linked open request). Unlike `primary_project_scope`,
    /// this list is never widened to a common ancestor and is therefore safe
    /// to compile into write confinement.
    pub authorized_project_scopes: Vec<String>,
    pub allow_multi_project_scope: bool,
    pub followup_mode: Option<FollowupMode>,
    pub reasons: Vec<TurnContextReason>,
    pub completion_contract: CompletionContract,
    /// True when an unresolved request supplied a durable contract, which is
    /// then retained instead of being demoted to structural resource identity.
    pub inherited_completion_contract: bool,
    /// Distinguishes an unfinished request's outstanding proof obligations
    /// from a completed antecedent whose durable safety constraints alone may
    /// carry into a related turn.
    pub inherited_outstanding_obligations: bool,
}

impl TurnContext {
    /// The workspace binding that vouches for its files: the bound
    /// workspace, only when it was bound explicitly.
    pub fn explicit_workspace(&self) -> Option<&str> {
        (self.bound_workspace_origin
            == Some(crate::turn::project_scope::WorkspaceBindingOrigin::Explicit))
        .then_some(self.bound_workspace.as_deref())
        .flatten()
    }
}

const GOAL_CONTEXT_RECENT_MESSAGES_LIMIT: usize = 6;
const GOAL_CONTEXT_HINT_HISTORY_LIMIT: usize = 30;
const GOAL_CONTEXT_MAX_PROJECT_SCOPES: usize = 6;

pub fn related_open_request(
    state: Option<&aidaemon_events::traits::DialogueState>,
    followup_mode: FollowupMode,
) -> Option<&aidaemon_events::traits::OpenRequest> {
    (followup_mode != FollowupMode::NewTask)
        .then(|| state.and_then(|state| state.open_request.as_ref()))
        .flatten()
}

pub fn normalize_message_resources(msg: &Message) -> Message {
    let mut normalized = msg.with_inferred_annotations().into_owned();
    for attachment in &mut normalized.attachments {
        aidaemon_channels::attachments::ensure_attachment_resource_identity(attachment);
    }

    // Resource handles are trusted runtime metadata, not an interpretation of
    // the user's wording. Expose them to the model so follow-ups can bind an
    // exact artifact without pronoun or basename heuristics.
    let existing = normalized.content.clone().unwrap_or_default();
    let resource_lines = normalized
        .attachments
        .iter()
        .filter_map(|attachment| {
            let resource_id = attachment.resource_id.as_deref()?;
            (!existing.contains(resource_id)).then(|| {
                format!(
                    "[Resource: {} ({}; {})]",
                    resource_id, attachment.filename, attachment.mime_type
                )
            })
        })
        .collect::<Vec<_>>();
    if !resource_lines.is_empty() {
        let resource_block = resource_lines.join("\n");
        normalized.content = Some(if existing.trim().is_empty() {
            resource_block
        } else {
            format!("{existing}\n{resource_block}")
        });
    }
    normalized
}

pub async fn emit_registered_resources(
    emitter: &aidaemon_events::EventEmitter,
    msg: &Message,
    produced_by_tool_call_id: Option<String>,
    task_id: Option<String>,
    turn_id: Option<String>,
) -> anyhow::Result<()> {
    for attachment in &msg.attachments {
        if let Some(resource) = aidaemon_events::ResourceRegisteredData::from_attachment(
            attachment,
            produced_by_tool_call_id.clone(),
            task_id.clone(),
            turn_id.clone(),
        ) {
            emitter
                .emit(aidaemon_events::EventType::ResourceRegistered, resource)
                .await?;
        }
    }
    Ok(())
}

pub fn trim_assistant_context_content(content: &str) -> String {
    let trimmed = content.trim();
    if let Some((before, _)) = trimmed.split_once(INTENT_GATE_MARKER) {
        before.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn is_low_signal_http_metadata_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.starts_with("content-type:")
        || lower.starts_with("content-length:")
        || lower.starts_with("cache-control:")
        || lower.starts_with("date:")
        || lower.starts_with("etag:")
        || lower.starts_with("expires:")
        || lower.starts_with("last-modified:")
        || lower.starts_with("location:")
        || lower.starts_with("server:")
        || lower.starts_with("strict-transport-security:")
        || lower.starts_with("vary:")
        || lower.starts_with("via:")
        || lower.starts_with("x-")
}

pub fn is_low_info_recent_tool_context(tool_name: &str) -> bool {
    matches!(tool_name, "write_file" | "edit_file" | "check_environment")
        || crate::turn::tool_facts::memory_tool_facts(tool_name)
            .is_some_and(|facts| facts.low_signal_result)
}

pub fn is_low_signal_tool_context_line(line: &str) -> bool {
    let lower = line.trim().to_ascii_lowercase();
    lower.is_empty()
        || is_low_signal_http_metadata_line(line)
        || lower == "[truncated]"
        || lower.starts_with("exit code:")
        || lower.starts_with("[mode:")
}

pub fn summarize_http_request_context(primary: &str) -> Option<String> {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let first = lines.next()?;
    if !first.to_ascii_lowercase().starts_with("http ") {
        return Some(first.to_string());
    }

    let detail = lines.find(|line| !is_low_signal_http_metadata_line(line));
    match detail {
        Some(detail) if !detail.eq_ignore_ascii_case(first) => Some(format!("{first} | {detail}")),
        _ => Some(first.to_string()),
    }
}

pub fn summarize_generic_tool_context(primary: &str) -> Option<String> {
    let mut lines = primary
        .lines()
        .map(str::trim)
        .filter(|line| !is_low_signal_tool_context_line(line));
    let first = lines.next()?;
    let second = lines.next();
    match second {
        Some(second) if !second.eq_ignore_ascii_case(first) => Some(format!("{first} | {second}")),
        _ => Some(first.to_string()),
    }
}

pub fn summarize_recent_tool_context(msg: &Message) -> Option<String> {
    let tool_name = msg.tool_name.as_deref()?;
    if is_low_info_recent_tool_context(tool_name) {
        return None;
    }

    let primary = msg.primary_content()?;
    let summary = match tool_name {
        "http_request" => summarize_http_request_context(&primary)?,
        _ => summarize_generic_tool_context(&primary)?,
    };
    Some(format!(
        "{}: {}",
        tool_name,
        truncate_for_resume(&summary, 240)
    ))
}

pub fn extract_recent_parent_messages(history: &[Message], max_messages: usize) -> Vec<Value> {
    let mut rows_rev: Vec<Value> = Vec::new();
    let mut recent_tool_rows = 0usize;

    for msg in history.iter().rev() {
        let row = match msg.role.as_str() {
            "user" => {
                let Some(raw) = msg.content.as_deref().map(str::trim) else {
                    continue;
                };
                if raw.is_empty() {
                    continue;
                }
                Some(json!({
                    "message_id": msg.id,
                    "role": msg.role,
                    "content": truncate_for_resume(raw, 500),
                }))
            }
            "assistant" => {
                let Some(raw) = msg.content.as_deref().map(str::trim) else {
                    continue;
                };
                if raw.is_empty() {
                    continue;
                }
                let content = trim_assistant_context_content(raw);
                if content.trim().is_empty() {
                    continue;
                }
                Some(json!({
                    "message_id": msg.id,
                    "role": msg.role,
                    "content": truncate_for_resume(content.trim(), 500),
                }))
            }
            "tool" if recent_tool_rows < 2 => summarize_recent_tool_context(msg).map(|content| {
                recent_tool_rows += 1;
                json!({
                    "message_id": msg.id,
                    "role": "tool",
                    "content": content,
                })
            }),
            _ => None,
        };

        if let Some(row) = row {
            rows_rev.push(row);
            if rows_rev.len() >= max_messages {
                break;
            }
        }
    }

    rows_rev.reverse();
    rows_rev
}

/// The session history appends and turn-context assembly every turn runs,
/// for any [`TurnHost`].
#[async_trait::async_trait]
pub trait TurnHistory: TurnHost {
    async fn build_turn_context_from_recent_history(
        &self,
        session_id: &str,
        user_text: &str,
    ) -> TurnContext {
        self.build_turn_context_from_recent_history_with_origin(session_id, user_text, false)
            .await
    }

    async fn build_turn_context_from_recent_history_with_origin(
        &self,
        session_id: &str,
        user_text: &str,
        internal_continuation: bool,
    ) -> TurnContext {
        let stored_current = user_text.trim();
        if stored_current.is_empty() {
            return TurnContext::default();
        }
        let authored_current = aidaemon_channels::attachments::user_authored_text(stored_current);
        // History rows include channel file stubs; match on the full stored text.
        let history = self
            .messages()
            .get_history(session_id, GOAL_CONTEXT_HINT_HISTORY_LIMIT)
            .await
            .unwrap_or_default();
        let (_, prev_user) = find_previous_turns(&history, stored_current);
        let mut reasons = Vec::new();
        // Dialogue ingestion already classified this exact persisted user turn.
        // Reuse that decision so dialogue state and prompt assembly cannot drift
        // into contradictory modes for the same message.
        let dialogue_state = self
            .dialogue_states()
            .get_dialogue_state(session_id)
            .await
            .ok()
            .flatten();
        let mut followup_mode = if internal_continuation {
            reasons.clear();
            reasons.push(TurnContextReason::InternalContinuation);
            FollowupMode::Followup
        } else {
            dialogue_state
                .as_ref()
                .and_then(|state| {
                    crate::turn::dialogue_state::resolved_followup_mode(state, stored_current)
                })
                .unwrap_or(FollowupMode::NewTask)
        };
        if !internal_continuation {
            reasons.clear();
            reasons.push(match followup_mode {
                FollowupMode::NewTask => TurnContextReason::DefaultNewTask,
                FollowupMode::Followup => TurnContextReason::ContextDependentQuestion,
                FollowupMode::ClarificationAnswer => TurnContextReason::ClarificationAnswer,
            });
        }

        let mut goal_user_text = if authored_current.is_empty() {
            String::new()
        } else {
            authored_current.clone()
        };
        let continuation_request_anchor = internal_continuation
            .then(|| {
                related_open_request(dialogue_state.as_ref(), followup_mode)
                    .map(|request| request.text.as_str())
                    .or(prev_user.as_deref())
            })
            .flatten();
        // Mismatch preflight: still used for project scope divergence detection
        // (affects scope extraction below), but no longer gates goal_user_text enrichment.
        if !internal_continuation && followup_mode != FollowupMode::NewTask {
            let request_anchor = related_open_request(dialogue_state.as_ref(), followup_mode)
                .map(|request| request.text.as_str())
                .or(prev_user.as_deref());
            let mismatch_preflight_drop = request_anchor.is_some_and(|prev| {
                has_project_scope_divergence_with_aliases(
                    self.execution_backend(),
                    prev,
                    &authored_current,
                    self.project_roots(),
                )
            });
            if mismatch_preflight_drop {
                followup_mode = FollowupMode::NewTask;
                reasons.push(TurnContextReason::FollowupOverrideMismatchPreflight);
                reasons.push(TurnContextReason::DefaultNewTask);
                POLICY_METRICS
                    .context_mismatch_preflight_drop_total
                    .fetch_add(1, Ordering::Relaxed);
                POLICY_METRICS
                    .followup_mode_overrides_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        // Enrich only genuine follow-ups. New tasks start a fresh provider
        // transcript floor; concatenating the previous request here would leak
        // stale instructions into contracts, schedules, goals, and replies.
        if followup_mode != FollowupMode::NewTask {
            let request_anchor = continuation_request_anchor.or_else(|| {
                related_open_request(dialogue_state.as_ref(), followup_mode)
                    .map(|request| request.text.as_str())
                    .or(prev_user.as_deref())
            });
            if let Some(prev_user_text) =
                request_anchor.filter(|prev| !prev.trim().eq_ignore_ascii_case(stored_current))
            {
                let mut combined = String::new();
                combined.push_str("Original request:\n");
                combined.push_str(&truncate_for_resume(prev_user_text.trim(), 2000));
                combined.push_str("\n\nCurrent request:\n");
                combined.push_str(if authored_current.is_empty() {
                    stored_current
                } else {
                    &authored_current
                });
                goal_user_text = combined;
            }
        }
        // Sanitize carryover blocks regardless of mode (they can appear in any message)
        {
            let (sanitized, changed) = sanitize_carryover_blocks(&goal_user_text);
            if changed {
                reasons.push(TurnContextReason::CarryoverSanitized);
                POLICY_METRICS
                    .context_bleed_prevented_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            if !sanitized.is_empty() {
                goal_user_text = sanitized;
            }
        }

        // A typed durable workspace binding outranks anything inferred from
        // request text. Two sources exist, both runtime state rather than
        // prose:
        //
        // * this agent's own goal binding (`scheduled_goal_runs bind_workspace`
        //   → goal.context.project_scope), re-read from the store each turn so
        //   a bind issued mid-run takes effect on the following turn;
        // * the workspace a parent handed to this spawned worker — for an
        //   executor, its provisioned attempt workspace (`task_workspaces`).
        //
        // Mission and handoff text routinely name other repositories,
        // hostnames containing a sibling project's name, or
        // instruction-relative paths like `src/content/posts`. Such paths are
        // resolved *inside* the binding; they must never relocate the worker
        // to whichever repository the daemon process happens to run from.
        let bound_goal_scope = if self.role() == aidaemon_events::traits::AgentRole::Executor {
            None
        } else if let Some(goal_id) = self.goal_id() {
            self.goals()
                .get_goal(goal_id)
                .await
                .ok()
                .flatten()
                .and_then(|goal| {
                    crate::turn::project_scope::bound_workspace_from_goal_context(
                        goal.context.as_deref(),
                    )
                })
        } else {
            None
        };
        use crate::turn::project_scope::WorkspaceBindingOrigin;
        let (bound_workspace, bound_workspace_origin) = match bound_goal_scope {
            Some(scope) => (Some(scope), Some(WorkspaceBindingOrigin::Explicit)),
            None => match crate::turn::project_scope::typed_workspace_binding(
                self.inherited_project_scope(),
            ) {
                Some(scope) => (Some(scope), Some(WorkspaceBindingOrigin::Inferred)),
                None => (None, None),
            },
        };
        // Relative references in a delegated request mean the directory the
        // parent's work ran in (its handoff), not the daemon's own. Only
        // resolution is anchored: the binding, and so confinement, is unchanged.
        let handed_over_dir = self
            .inherited_authority()
            .and_then(|inherited| inherited.known_resources.working_dir());
        let workspace_anchor = bound_workspace
            .as_deref()
            .or(handed_over_dir.as_deref())
            .map(std::path::Path::new);

        // For the current user message, only extract explicit filesystem paths
        // (~/foo, /foo, ./foo) — NOT contextual nickname matches.  This prevents
        // common English words like "modern" (in "modern website") from resolving
        // to existing project directories like modern-plants-site.
        let intent_scope_text = continuation_request_anchor.unwrap_or(&authored_current);
        let mut current_project_scopes = Vec::new();
        extract_explicit_path_scopes_from_text_anchored(
            self.execution_backend(),
            intent_scope_text,
            &mut current_project_scopes,
            GOAL_CONTEXT_MAX_PROJECT_SCOPES,
            self.project_roots(),
            workspace_anchor,
        );
        // Scope carryover follows the persisted dialogue relationship, not an
        // acknowledgement/command phrase list. New requests must name their
        // scope structurally or receive a grounded semantic project reference.
        let allow_scope_carryover = internal_continuation || followup_mode != FollowupMode::NewTask;
        let mut project_scopes = extract_project_scopes_from_history_anchored(
            self.execution_backend(),
            &history,
            intent_scope_text,
            GOAL_CONTEXT_MAX_PROJECT_SCOPES,
            allow_scope_carryover,
            self.project_roots(),
            workspace_anchor,
        );
        drop_inbox_scopes(&mut project_scopes, self.inbox_dir());
        // A carried-over turn that names no path inherits the scope of the
        // request it continues before any older history. Whole-history
        // extraction prefers whichever scope looks like a project root, so an
        // unrelated repository mentioned turns ago used to outrank the
        // antecedent's own workspace (e.g. a scratch dir under /tmp), and the
        // scope lock then refused the files the antecedent had just created.
        let mut antecedent_project_scopes = Vec::new();
        if allow_scope_carryover && current_project_scopes.is_empty() {
            let request_anchor = continuation_request_anchor.or_else(|| {
                related_open_request(dialogue_state.as_ref(), followup_mode)
                    .map(|request| request.text.as_str())
                    .or(prev_user.as_deref())
            });
            if let Some(anchor) = request_anchor {
                extract_explicit_path_scopes_from_text_anchored(
                    self.execution_backend(),
                    anchor,
                    &mut antecedent_project_scopes,
                    GOAL_CONTEXT_MAX_PROJECT_SCOPES,
                    self.project_roots(),
                    workspace_anchor,
                );
            }
        }
        drop_inbox_scopes(&mut antecedent_project_scopes, self.inbox_dir());
        let allow_multi_project_scope = current_project_scopes.len() > 1;
        // For current-turn scopes, unify all explicitly mentioned paths into a
        // single scope that encompasses them all.  When the user mentions paths
        // in different subdirectories (e.g. ~/projects/blog/posts/file.md and
        // ~/projects/blog/tweets.md), the unified scope is their common ancestor
        // (~/projects/blog/) so writes to sibling directories aren't blocked.
        //
        // Falls back to `.first()` for single-scope messages, preserving the
        // text-order priority that's critical for new-project creation (the
        // user's explicit path may not exist yet on disk).
        // The project-root preference is only appropriate for history scopes
        // where we want to pick the most likely intended project among stale
        // references.
        // Under a binding the workspace is the primary scope outright.
        let primary_project_scope = bound_workspace.clone().or_else(|| {
            resolve_primary_project_scope(
                unify_current_turn_scopes(&current_project_scopes)
                    .or_else(|| unify_current_turn_scopes(&antecedent_project_scopes))
                    .or_else(|| {
                        choose_primary_project_scope(self.execution_backend(), &project_scopes)
                    }),
                self.inherited_project_scope(),
                allow_multi_project_scope,
                allow_scope_carryover,
            )
        });
        // Under a binding, text-extracted paths outside the bound workspace
        // are not authorized: the binding is the objective's authority
        // boundary.
        let authorized_project_scopes =
            if bound_workspace.is_some() || current_project_scopes.is_empty() {
                primary_project_scope.iter().cloned().collect()
            } else {
                current_project_scopes.clone()
            };
        let contract_text = continuation_request_anchor.unwrap_or(&goal_user_text);
        let mut completion_contract = infer_structural_completion_contract_anchored(
            self.execution_backend(),
            contract_text,
            self.project_roots(),
            workspace_anchor,
        );
        let mut inherited_completion_contract = false;
        let mut inherited_outstanding_obligations = false;
        if followup_mode != FollowupMode::NewTask {
            if let Some(request) = related_open_request(dialogue_state.as_ref(), followup_mode) {
                let mut unfinished_contract = request
                    .completion_contract
                    .as_ref()
                    .map(completion_contract_from_persisted)
                    .unwrap_or_else(|| {
                        infer_structural_completion_contract_anchored(
                            self.execution_backend(),
                            &request.text,
                            self.project_roots(),
                            workspace_anchor,
                        )
                    });
                // Legacy persisted contracts predate container-level lineage.
                // The open-request lifecycle already owns an exact task edge,
                // so upgrade that authoritative relationship before adoption.
                if unfinished_contract.scope_task_id.is_none() {
                    unfinished_contract.scope_task_id = request.task_id.clone();
                }
                inherited_completion_contract = request.completion_contract.is_some();
                inherited_outstanding_obligations = dialogue_state
                    .as_ref()
                    .is_some_and(|state| state.has_unresolved_request_obligation());
                completion_contract = if !inherited_outstanding_obligations {
                    crate::turn::completion_contract::inherit_request_constraints(
                        completion_contract,
                        &unfinished_contract,
                    )
                } else {
                    inherit_unfinished_request_contract(completion_contract, &unfinished_contract)
                };
            }
        }
        if self.role() == aidaemon_events::traits::AgentRole::Executor {
            completion_contract = scope_contract_for_delegated_executor(completion_contract);
        }
        TurnContext {
            goal_user_text,
            assessment_recent_messages: extract_recent_parent_messages(
                history
                    .iter()
                    .rposition(|message| {
                        message.role == "user"
                            && message
                                .content
                                .as_deref()
                                .is_some_and(|content| content.trim() == stored_current)
                    })
                    .map(|index| &history[..index])
                    .unwrap_or(history.as_slice()),
                GOAL_CONTEXT_RECENT_MESSAGES_LIMIT,
            ),
            visible_antecedent_user_message_id: None,
            bound_workspace,
            bound_workspace_origin,
            primary_project_scope,
            authorized_project_scopes,
            allow_multi_project_scope,
            followup_mode: Some(followup_mode),
            reasons,
            completion_contract,
            inherited_completion_contract,
            inherited_outstanding_obligations,
        }
    }

    /// Resolve the turn_id to stamp on an emitted event for `msg`. Prefer the
    /// message's own turn_id (set for the user message in bootstrap); fall back
    /// to the active turn in `current_turn_ids` for assistant/tool messages that
    /// were constructed without it. Mirrors the auto-stamp in
    /// `append_message_canonical` so the emitted event and the canonical row
    /// share the same turn_id.
    async fn resolve_event_turn_id(&self, msg: &Message) -> Option<String> {
        if msg.turn_id.is_some() {
            return msg.turn_id.clone();
        }
        self.turn_runtime()
            .current_turn_ids
            .read()
            .await
            .get(&msg.session_id)
            .cloned()
    }

    async fn append_message_canonical(&self, msg: &Message) -> anyhow::Result<()> {
        // Auto-stamp turn_id from the per-session map so every message
        // written during a turn shares the same id. Messages that already
        // carry a turn_id (e.g., the user message in bootstrap, which sets
        // its own id) are left alone.
        if msg.turn_id.is_some() {
            return self.messages().append_message(msg).await;
        }
        let turn_id = self
            .turn_runtime()
            .current_turn_ids
            .read()
            .await
            .get(&msg.session_id)
            .cloned();
        match turn_id {
            Some(tid) => {
                let mut stamped = msg.clone();
                stamped.turn_id = Some(tid);
                self.messages().append_message(&stamped).await
            }
            None => {
                // No active turn for this session — preserve old behavior.
                // This path covers background tasks, sub-agent flows, and
                // test scaffolding that bypass `handle_message_impl`.
                self.messages().append_message(msg).await
            }
        }
    }

    async fn append_user_message_with_event(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        msg: &Message,
        user_role: aidaemon_tools::semantics::UserRole,
        channel_ctx: &ChannelContext,
        has_attachments: bool,
    ) -> anyhow::Result<i64> {
        let normalized_msg = normalize_message_resources(msg);
        let execution_origin = self.works_for_mandate().then(|| "mandate".to_string());
        let event_id = emitter
            .emit(
                EventType::UserMessage,
                json!({
                    "content": normalized_msg.content.clone().unwrap_or_default(),
                    "message_id": normalized_msg.id.clone(),
                    "has_attachments": has_attachments,
                    "attachments": normalized_msg.attachments.clone(),
                    "annotations": normalized_msg.annotations.clone(),
                    // Provenance for downstream projections/consolidation.
                    "channel_visibility": channel_ctx.visibility.to_string(),
                    "channel_id": channel_ctx.channel_id.clone(),
                    "platform": channel_ctx.platform.clone(),
                    "sender_id": channel_ctx.sender_id.clone(),
                    "user_role": user_role.to_string(),
                    // Durable, explicit provenance. Memory isolation must not
                    // depend on the worker's generated session name or on an
                    // owner role inherited by an internal mandate worker.
                    "execution_origin": execution_origin,
                    // Pillar B: stamp the turn_id onto the emitted event so the
                    // turn-anchored fetch (which filters turn_id IS NOT NULL)
                    // returns this message. Without this the row is NULL.
                    "turn_id": normalized_msg.turn_id.clone(),
                }),
            )
            .await?;
        emit_registered_resources(
            emitter,
            &normalized_msg,
            None,
            None,
            normalized_msg.turn_id.clone(),
        )
        .await?;
        self.append_message_canonical(&normalized_msg).await?;
        crate::turn::dialogue_state::record_dialogue_user_message(
            self,
            &normalized_msg.session_id,
            &normalized_msg,
        )
        .await?;
        Ok(event_id)
    }

    fn append_assistant_message_with_event<'a>(
        &'a self,
        emitter: &'a aidaemon_events::EventEmitter,
        msg: &'a Message,
        model: &'a str,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        self.append_assistant_message_with_event_disposition(
            emitter,
            msg,
            model,
            input_tokens,
            output_tokens,
            aidaemon_events::AssistantResponseDisposition::Terminal,
        )
    }

    fn append_background_handoff_message_with_event<'a>(
        &'a self,
        emitter: &'a aidaemon_events::EventEmitter,
        msg: &'a Message,
        model: &'a str,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        self.append_assistant_message_with_event_disposition(
            emitter,
            msg,
            model,
            input_tokens,
            output_tokens,
            aidaemon_events::AssistantResponseDisposition::BackgroundHandoff,
        )
    }

    fn append_assistant_message_with_event_disposition<'a>(
        &'a self,
        emitter: &'a aidaemon_events::EventEmitter,
        msg: &'a Message,
        model: &'a str,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        disposition: aidaemon_events::AssistantResponseDisposition,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async move {
            // The persisted assistant artifact is whatever the completion
            // phase decided to deliver. Response-contract projection is a
            // completion-phase decision with its own decision point; this
            // persistence boundary must never silently rewrite a reply that
            // already passed the completion gates.
            let normalized_msg = normalize_message_resources(msg);
            let turn_id = self.resolve_event_turn_id(&normalized_msg).await;
            let tool_calls = normalized_msg.tool_calls_json.as_ref().and_then(|raw| {
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
            let referenced_receipts = if normalized_msg
                .content
                .as_deref()
                .is_some_and(|content| !content.trim().is_empty())
                && tool_calls.is_none()
            {
                match emitter.task_id() {
                    Some(task_id) => self
                        .events()
                        .task_completion_proof_references(task_id)
                        .await
                        .unwrap_or_default(),
                    None => Vec::new(),
                }
            } else {
                Vec::new()
            };
            emitter
                .emit(
                    EventType::AssistantResponse,
                    AssistantResponseData {
                        message_id: Some(normalized_msg.id.clone()),
                        content: normalized_msg.content.clone(),
                        model: model.to_string(),
                        tool_calls,
                        input_tokens,
                        output_tokens,
                        annotations: normalized_msg.annotations.clone(),
                        turn_id: turn_id.clone(),
                        task_id: emitter.task_id().map(str::to_string),
                        referenced_receipts,
                        disposition,
                    },
                )
                .await?;
            self.append_message_canonical(&normalized_msg).await?;
            crate::turn::dialogue_state::record_dialogue_assistant_message(
                self,
                &normalized_msg.session_id,
                &normalized_msg,
            )
            .await?;
            Ok(())
        })
    }

    async fn append_tool_message_with_result_event(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        msg: &Message,
        success: bool,
        duration_ms: u64,
        error: Option<String>,
        task_id: Option<&str>,
    ) -> anyhow::Result<()> {
        self.append_tool_message_with_result_event_policy(
            emitter,
            msg,
            success,
            duration_ms,
            error,
            task_id,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_tool_message_with_result_event_policy(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        msg: &Message,
        success: bool,
        duration_ms: u64,
        error: Option<String>,
        task_id: Option<&str>,
        persistent_result: Option<&str>,
    ) -> anyhow::Result<()> {
        self.append_tool_message_with_receipt_event_policy(
            emitter,
            msg,
            success,
            duration_ms,
            error,
            task_id,
            persistent_result,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_tool_message_with_receipt_event_policy(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        msg: &Message,
        success: bool,
        duration_ms: u64,
        error: Option<String>,
        task_id: Option<&str>,
        persistent_result: Option<&str>,
        receipt: Option<aidaemon_events::ToolReceiptV1>,
    ) -> anyhow::Result<()> {
        let normalized_msg = normalize_message_resources(msg);
        let turn_id = self.resolve_event_turn_id(&normalized_msg).await;
        let durable_result = persistent_result
            .map(str::to_owned)
            .unwrap_or_else(|| normalized_msg.content.clone().unwrap_or_default());
        let success = receipt.as_ref().map_or(success, |receipt| {
            receipt.outcome_status == aidaemon_events::traits::ToolOutcomeStatus::Succeeded
                && !receipt.contract_rejected
        });
        emitter
            .emit(
                EventType::ToolResult,
                ToolResultData {
                    message_id: Some(normalized_msg.id.clone()),
                    tool_call_id: normalized_msg
                        .tool_call_id
                        .clone()
                        .unwrap_or_else(|| normalized_msg.id.clone()),
                    name: normalized_msg
                        .tool_name
                        .clone()
                        .unwrap_or_else(|| "system".to_string()),
                    result: durable_result,
                    success,
                    duration_ms,
                    error,
                    task_id: task_id.map(str::to_string),
                    annotations: normalized_msg.annotations.clone(),
                    turn_id: turn_id.clone(),
                    attachments: normalized_msg.attachments.clone(),
                    receipt,
                },
            )
            .await?;
        emit_registered_resources(
            emitter,
            &normalized_msg,
            normalized_msg.tool_call_id.clone(),
            task_id.map(str::to_string),
            turn_id,
        )
        .await?;
        self.append_message_canonical(&normalized_msg).await?;
        Ok(())
    }

    /// Persist a typed policy denial: the operation was refused before
    /// dispatch by an authority boundary (read-only contract, capability
    /// constraint, mandate gate, scope lock). The receipt carries the typed
    /// `ToolAccessDenial` so the run ledger — not a caller-maintained counter
    /// — can recognize the denial as a terminal observation.
    #[allow(clippy::too_many_arguments)]
    async fn persist_pre_dispatch_policy_denial(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        session_id: &str,
        task_id: &str,
        tool_call: &ToolCall,
        effective_arguments: &str,
        result_text: String,
        reason_code: &str,
        semantics: ToolCallSemantics,
        access_manifest: Option<ToolCallAccessManifest>,
    ) -> anyhow::Result<()> {
        self.persist_pre_dispatch_outcome_with_kernel_claim(
            emitter,
            session_id,
            task_id,
            tool_call,
            effective_arguments,
            result_text,
            aidaemon_events::traits::ToolOutcomeStatus::Blocked,
            aidaemon_events::traits::ToolInvocationStage::RejectedBeforeDispatch,
            false,
            semantics,
            access_manifest,
            Some(aidaemon_events::traits::ToolAccessDenial {
                reason_code: reason_code.to_string(),
                enforcement: aidaemon_events::traits::ToolAccessEnforcement::ControllerEnforced,
                exit_code: None,
                proposed_evidence: Vec::new(),
            }),
            None,
        )
        .await
    }

    /// Persist a pre-dispatch outcome together with the durable operation and
    /// obligation binding that was proposed. This makes denied attempts part
    /// of the same replayable lifecycle as dispatched calls.
    #[allow(clippy::too_many_arguments)]
    async fn persist_pre_dispatch_outcome_with_kernel_claim(
        &self,
        emitter: &aidaemon_events::EventEmitter,
        session_id: &str,
        task_id: &str,
        tool_call: &ToolCall,
        effective_arguments: &str,
        result_text: String,
        outcome_status: aidaemon_events::traits::ToolOutcomeStatus,
        invocation_stage: aidaemon_events::traits::ToolInvocationStage,
        contract_rejected: bool,
        semantics: ToolCallSemantics,
        access_manifest: Option<ToolCallAccessManifest>,
        access_denial: Option<aidaemon_events::traits::ToolAccessDenial>,
        kernel_claim: Option<&aidaemon_events::TaskKernelOperationClaim>,
    ) -> anyhow::Result<()> {
        let mut call_data = aidaemon_events::ToolCallData::from_tool_call(
            tool_call.id.clone(),
            tool_call.name.clone(),
            serde_json::from_str(effective_arguments).unwrap_or_else(|_| serde_json::json!({})),
            Some(task_id.to_string()),
        );
        if let Some(claim) = kernel_claim {
            call_data = call_data.with_task_kernel_claim(claim);
        }
        if let Some(claim) = kernel_claim {
            // Pre-dispatch denials are still task-kernel proposals. Persist
            // them through the same atomic admission/reducer boundary as
            // adapter I/O so obligation ownership and cardinality cannot
            // diverge by invocation stage.
            let _ = emitter.emit_admitted_tool_call(call_data, claim).await?;
        } else {
            emitter
                .emit(aidaemon_events::EventType::ToolCall, call_data)
                .await?;
        }

        // A typed denial names what the refused operation would have
        // produced, so the run ledger can bind it to the obligation it was
        // attempting. Filled once here for every denial site.
        let access_denial = access_denial.map(|mut denial| {
            if denial.proposed_evidence.is_empty() {
                denial.proposed_evidence =
                    crate::turn::inquiry::evidence_capabilities_for_tool_call(
                        &tool_call.name,
                        effective_arguments,
                    );
            }
            denial
        });
        let metadata = aidaemon_events::traits::ToolCallMetadata {
            access_manifest,
            access_denial,
            outcome_status: Some(outcome_status),
            invocation_stage,
            contract_rejected,
            semantics,
            ..aidaemon_events::traits::ToolCallMetadata::default()
        };
        let receipt = aidaemon_events::ToolReceiptV1::from_metadata(
            &metadata,
            outcome_status,
            aidaemon_events::ToolOutcomeEvidenceSource::StructuredMetadata,
            None,
        );
        let error = outcome_status.is_failure().then_some(result_text.clone());
        let tool_msg = Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: "tool".to_string(),
            content: Some(result_text),
            tool_call_id: Some(tool_call.id.clone()),
            tool_name: Some(tool_call.name.clone()),
            tool_calls_json: None,
            created_at: chrono::Utc::now(),
            importance: 0.2,
            ..Message::runtime_defaults()
        };
        self.append_tool_message_with_receipt_event_policy(
            emitter,
            &tool_msg,
            false,
            0,
            error,
            Some(task_id),
            None,
            Some(receipt),
        )
        .await
    }

    // Pillar B (Task 7): `load_initial_history` and `load_recent_history` were
    // removed. Historical conversation retention is now owned entirely by the
    // turn-anchored fetch (`EventStore::get_turns_from_anchor`) in
    // `message_build`; the legacy "load N recent events then age-collapse"
    // path no longer exists.
}

impl<T: TurnHost + ?Sized> TurnHistory for T {}
