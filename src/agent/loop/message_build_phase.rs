use super::*;
use crate::agent::turn_render::RenderMode;
use crate::agent::vision::{build_multimodal_content, user_message_content_matches};
use crate::execution_policy::PolicyBundle;
use crate::traits::MessageAttachment;

pub(super) struct MessageBuildCtx<'a> {
    pub session_id: &'a str,
    pub iteration: usize,
    pub user_text: &'a str,
    /// Attachments for the current user turn (images saved to inbox).
    pub current_attachments: &'a [MessageAttachment],
    pub completed_tool_calls: &'a [String],
    pub model: &'a str,
    /// Pillar A: message-zero bytes (session-static CORE prompt). Byte-stable
    /// across the within-task loop so the prefix cache reuses it.
    pub core_prompt: &'a str,
    /// Pillar A: the per-task volatile context tail. Inserted at boundary − 1
    /// (immediately before the current user message). The SAME string is reused
    /// every iteration of the within-task loop.
    pub task_context_tail: &'a str,
    pub tool_defs: &'a [Value],
    pub policy_bundle: &'a PolicyBundle,
    pub pending_system_messages: &'a mut Vec<SystemDirective>,
    pub empty_response_retry_pending: bool,
    pub status_tx: &'a Option<mpsc::Sender<StatusUpdate>>,
}

pub(super) struct MessageBuildData {
    pub messages: Vec<Value>,
    pub tool_defs: Vec<Value>,
    /// Estimated input tokens (messages + tool schemas) for this call, used for
    /// est-vs-actual drift telemetry in the `LlmCall` event.
    pub est_input_tokens: u32,
}

const EMPTY_RETRY_MAX_PARENT_CHARS: usize = 800;
const EXECUTION_CHECKPOINT_MAX_REQUEST_CHARS: usize = 240;
const EXECUTION_CHECKPOINT_MAX_ACTIVITY_CHARS: usize = 900;
const EXECUTION_CHECKPOINT_MAX_EVIDENCE_CHARS: usize = 500;
const RESPONSE_RESERVE_TOKENS: usize = 1_536;
const MIN_MESSAGE_BUDGET_TOKENS: usize = 1_024;
const TOKEN_ESTIMATE_SAFETY_MARGIN: usize = 256;
/// Pillar B (Task 7): fixed token reserve for the in-flight current turn — the
/// current user message plus expected tool-chain headroom. Held constant across
/// turns so the archived-region budget does not churn build-to-build (a
/// per-turn-derived reserve would shift the eviction boundary every iteration).
const CURRENT_TURN_RESERVE_TOKENS: usize = 4_000;
/// Pillar B (Task 7): safety margin applied to the archived-region budget.
const ARCHIVED_BUDGET_SAFETY_MARGIN: f64 = 0.10;

fn trimmed_message_content(message: &Value) -> Option<String> {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn truncate_parent_for_empty_retry(content: &str) -> String {
    let mut out: String = content.chars().take(EMPTY_RETRY_MAX_PARENT_CHARS).collect();
    if content.chars().count() > EMPTY_RETRY_MAX_PARENT_CHARS {
        out.push_str("...");
    }
    out
}

fn is_current_user_message(message: &Value, user_text: &str) -> bool {
    message.get("role").and_then(|r| r.as_str()) == Some("user")
        && message
            .get("content")
            .is_some_and(|content| user_message_content_matches(content, user_text))
}

fn find_current_user_position(messages: &[Value], user_text: &str) -> Option<usize> {
    messages
        .iter()
        .rposition(|m| is_current_user_message(m, user_text))
}

fn current_turn_user_attachments(
    turn: &crate::events::FetchedTurn,
    user_text: &str,
) -> Vec<MessageAttachment> {
    turn.messages
        .iter()
        .filter(|m| m.role == "user")
        .find(|m| m.content.as_deref() == Some(user_text))
        .map(|m| m.attachments.clone())
        .unwrap_or_default()
}

fn build_empty_response_retry_messages(existing: &[Value], user_text: &str) -> Vec<Value> {
    let current_idx = find_current_user_position(existing, user_text);
    let search_end = current_idx.unwrap_or(existing.len());

    let prev_assistant = existing
        .iter()
        .take(search_end)
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .and_then(trimmed_message_content);

    let prev_user = existing
        .iter()
        .take(search_end)
        .rev()
        .find(|m| {
            if m.get("role").and_then(|r| r.as_str()) != Some("user") {
                return false;
            }
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|content| content != user_text && !content.trim().is_empty())
        })
        .and_then(trimmed_message_content);

    let mut recovered = Vec::new();
    if let Some(prev_user) = prev_user {
        recovered.push(json!({
            "role": "user",
            "content": truncate_parent_for_empty_retry(&prev_user),
        }));
    }
    if let Some(prev_assistant) = prev_assistant {
        recovered.push(json!({
            "role": "assistant",
            "content": truncate_parent_for_empty_retry(&prev_assistant),
        }));
    }
    recovered.push(json!({
        "role": "user",
        "content": user_text,
    }));

    recovered
}

fn tool_is_low_info_for_checkpoint(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "write_file"
            | "edit_file"
            | "manage_memories"
            | "manage_people"
            | "remember_fact"
            | "check_environment"
    )
}

fn build_execution_checkpoint_message(
    user_text: &str,
    completed_tool_calls: &[String],
    current_interaction: &[&Message],
) -> Option<String> {
    let trimmed_user = user_text.trim();
    if trimmed_user.is_empty() || completed_tool_calls.is_empty() {
        return None;
    }

    let activity = super::post_task::categorize_tool_calls(completed_tool_calls);
    let latest_evidence = current_interaction.iter().rev().find_map(|message| {
        if message.role != "tool" {
            return None;
        }
        let tool_name = message.tool_name.as_deref().unwrap_or("").trim();
        if tool_name.is_empty() || tool_is_low_info_for_checkpoint(tool_name) {
            return None;
        }
        let content = message.primary_content()?;
        let content = content.trim();
        if content.is_empty() {
            return None;
        }
        Some(format!(
            "- {}: {}",
            tool_name,
            truncate_for_resume(content, EXECUTION_CHECKPOINT_MAX_EVIDENCE_CHARS)
        ))
    });

    let mut lines = vec![
        "[SYSTEM] EXECUTION CHECKPOINT: You are still working on the same active request from this turn.".to_string(),
        format!(
            "Active request: {}",
            truncate_for_resume(trimmed_user, EXECUTION_CHECKPOINT_MAX_REQUEST_CHARS)
        ),
    ];

    if !activity.trim().is_empty() {
        lines.push("Completed work so far:".to_string());
        lines.push(truncate_for_resume(
            activity.trim(),
            EXECUTION_CHECKPOINT_MAX_ACTIVITY_CHARS,
        ));
    }

    if let Some(evidence) = latest_evidence {
        lines.push("Latest concrete evidence:".to_string());
        lines.push(evidence);
    }

    lines.push("Continue from this checkpoint. Do NOT reset into a generic availability reply or ask what the user wants help with. Either take the next step for this request, answer with concrete results if it is complete, or state the blocker tied to this request.".to_string());

    Some(lines.join("\n"))
}

pub(super) async fn run_message_build_phase(
    services: &super::services::AgentServices<'_>,
    ctx: &mut MessageBuildCtx<'_>,
) -> anyhow::Result<MessageBuildData> {
    let agent = services.agent;
    let session_id = ctx.session_id;
    let iteration = ctx.iteration;
    let user_text = ctx.user_text;
    let current_attachments = ctx.current_attachments;
    let completed_tool_calls = ctx.completed_tool_calls;
    let model = ctx.model;
    let core_prompt = ctx.core_prompt;
    let task_context_tail = ctx.task_context_tail;
    let original_tool_defs = ctx.tool_defs;
    let policy_bundle = ctx.policy_bundle;
    let pending_system_messages = &mut *ctx.pending_system_messages;
    let empty_response_retry_pending = ctx.empty_response_retry_pending;
    let status_tx = ctx.status_tx;
    let render_options = agent.render_options();

    let total_context_budget =
        crate::memory::context_window::model_context_budget(model, &agent.context_window_config);
    // Pillar A: the system payload is now message zero (core) PLUS the task
    // context tail. Both occupy budget, so reserve for the sum.
    let system_tokens = crate::memory::context_window::estimate_tokens(core_prompt)
        + crate::memory::context_window::estimate_tokens(task_context_tail);
    let original_tool_tokens =
        crate::memory::context_window::estimate_tool_definition_tokens(original_tool_defs);
    let tool_budget = total_context_budget
        .saturating_sub(system_tokens + RESPONSE_RESERVE_TOKENS + MIN_MESSAGE_BUDGET_TOKENS);
    let mut effective_tool_defs = crate::memory::context_window::fit_tool_definitions_to_budget(
        original_tool_defs,
        tool_budget,
    );

    if effective_tool_defs != original_tool_defs {
        info!(
            session_id,
            iteration,
            model,
            total_context_budget,
            original_tool_tokens,
            effective_tool_tokens = crate::memory::context_window::estimate_tool_definition_tokens(
                &effective_tool_defs
            ),
            tool_count = effective_tool_defs.len(),
            "Compacted tool schema descriptions for model context compatibility"
        );
    }
    let mut tool_defs = effective_tool_defs.as_slice();

    // ============================================================
    // Pillar B (Task 7): turn-anchored fetch → render → evict.
    //
    // This replaces the legacy fetch → age-collapse → sliding-window →
    // JSON-conversion stages. The archived region is whole-turn rendered (never
    // partially trimmed), anchored at `agent.turn_anchors[session_id]`, and
    // carried byte-stable across the within-task loop via the per-turn render
    // cache. Only the current turn is full/append-only.
    // ============================================================

    use crate::events::TerminalState;

    // Step 0: archived-region budget. Wire it from the SAME budget sources the
    // legacy code used — no invented numbers. `core` is the cached core bytes
    // (already computed by Pillar A); `tools` is the name-sorted tool array this
    // build will send; `tail` is the `[Task Context]` tail string.
    let core_tokens = crate::memory::context_window::estimate_tokens(core_prompt);
    let tail_tokens = crate::memory::context_window::estimate_tokens(task_context_tail);
    let tool_tokens_for_budget =
        crate::memory::context_window::estimate_tool_definition_tokens(tool_defs);
    let archived_budget = super::turn_eviction::archived_budget(
        total_context_budget,
        core_tokens,
        tool_tokens_for_budget,
        tail_tokens,
        CURRENT_TURN_RESERVE_TOKENS,
        crate::memory::context_window::CONTEXT_RESPONSE_RESERVE_TOKENS,
        ARCHIVED_BUDGET_SAFETY_MARGIN,
    );

    let current_turn_id: Option<String> =
        agent.current_turn_ids.read().await.get(session_id).cloned();

    // Step 1: anchor resolve. Read the per-session anchor; on cold start /
    // restart, initialize via the BOUNDED reverse-walk page (NEVER a
    // full-session `get_turns_from_anchor(.., 0)`).
    let anchor: i64 = {
        let existing = agent.turn_anchors.read().await.get(session_id).copied();
        match existing {
            Some(a) => a,
            None => {
                // Reverse-walk newest→oldest, accumulating each turn's Archived
                // estimate, stopping at the last turn that keeps the running
                // total <= low_water(archived_budget).
                let low_water = super::turn_eviction::low_water(archived_budget);
                let mut before: Option<i64> = None;
                let mut accumulated: usize = 0;
                let mut init_anchor: Option<i64> = None;
                loop {
                    let page = agent
                        .event_store
                        .get_recent_turns_page(session_id, before, 1)
                        .await?;
                    let Some(turn) = page.into_iter().next() else {
                        break;
                    };
                    let terminal_state = TerminalState::from_task_status(turn.terminal_status);
                    let rendered = super::turn_render::render_turn(
                        &turn.messages,
                        super::turn_render::RenderMode::Archived { terminal_state },
                        super::turn_render::RENDERER_VERSION,
                        &render_options,
                    );
                    let est = crate::memory::context_window::estimate_tokens(
                        &serde_json::to_string(&rendered).unwrap_or_default(),
                    );
                    if init_anchor.is_none() {
                        // The newest turn is the (about-to-be) current turn. It is
                        // not part of the archived region, so its tokens are NOT
                        // counted against low_water; set the anchor floor here and
                        // walk past it to reach the archived turns.
                        init_anchor = Some(turn.turn_seq);
                        before = Some(turn.turn_seq);
                        continue;
                    }
                    // Stop once adding this archived turn would exceed low_water.
                    if accumulated + est > low_water {
                        break;
                    }
                    accumulated = accumulated.saturating_add(est);
                    init_anchor = Some(turn.turn_seq);
                    before = Some(turn.turn_seq);
                }
                let resolved = init_anchor.unwrap_or(0);
                agent
                    .turn_anchors
                    .write()
                    .await
                    .insert(session_id.to_string(), resolved);
                info!(
                    session_id,
                    anchor = resolved,
                    archived_budget,
                    "anchor initialized (cold start)"
                );
                resolved
            }
        }
    };

    // Step 2: fetch whole turns from the anchor.
    let mut turns = agent
        .event_store
        .get_turns_from_anchor(session_id, anchor)
        .await?;

    // Identify the current turn = last FetchedTurn whose turn_id == current_turn_id.
    let last_is_current = turns
        .last()
        .map(|t| t.turn_id.as_deref() == current_turn_id.as_deref() && current_turn_id.is_some())
        .unwrap_or(false);

    if !last_is_current {
        // Current-turn fallback (replaces `current_user_injected`). The happy
        // path persists the current UserMessage event before this build runs,
        // so the fetch ends in the current turn. This covers (a) the documented
        // not-yet-committed race and (b) legacy turn_id=NULL rows excluded by
        // the anchored fetch. Synthesize a current turn from the in-process
        // user_text + current_turn_id so the payload always ends in the current
        // user message.
        let synthetic_turn_id = current_turn_id.clone();
        let synthetic_seq = turns.last().map(|t| t.turn_seq + 1).unwrap_or(0);
        let current_user = Message {
            id: format!("synthetic-current-user-{}", uuid::Uuid::new_v4()),
            session_id: session_id.to_string(),
            role: "user".to_string(),
            content: Some(user_text.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: chrono::Utc::now(),
            importance: 1.0,
            turn_id: synthetic_turn_id.clone(),
            attachments: current_attachments.to_vec(),
            ..Message::runtime_defaults()
        };
        turns.push(crate::events::FetchedTurn {
            turn_id: synthetic_turn_id,
            turn_seq: synthetic_seq,
            messages: vec![current_user],
            terminal_status: None,
        });
        warn!(
            session_id,
            "current turn absent from fetch; injected in-process"
        );
    } else {
        // Happy-path assertion: the fetch ends in the current turn.
        debug_assert!(
            turns
                .last()
                .map(|t| t.turn_id.as_deref() == current_turn_id.as_deref())
                .unwrap_or(false),
            "turn-anchored fetch must end in the current turn on the happy path"
        );
    }

    // Split off the current turn (always the last entry now).
    let current_turn = turns.pop().expect("at least the current turn is present");
    // `turns` now holds only archived turns (oldest→newest).

    // Step 3: evict. Render each archived turn (Archived mode) to estimate over
    // the FINAL bytes, then plan_eviction. Cache the renders so step 4 reuses
    // them (and to drive the debug re-render assertion).
    struct ArchivedRender {
        turn_id: Option<String>,
        turn_seq: i64,
        content_fp: String,
        bytes: Vec<Value>,
        cache_hit: bool,
        cache_reason: &'static str,
    }

    let mut archived_renders: Vec<ArchivedRender> = Vec::with_capacity(turns.len());
    {
        // One read-lock snapshot of this session's render cache for the decision;
        // writes happen after the loop to avoid holding the lock across renders.
        let prev_cache = agent.turn_renders.read().await;
        let session_cache = prev_cache.get(session_id);
        for turn in &turns {
            let terminal_state = TerminalState::from_task_status(turn.terminal_status);
            let fp = super::turn_render_cache::content_fp(&turn.messages, terminal_state);
            let prev = turn
                .turn_id
                .as_deref()
                .and_then(|tid| session_cache.and_then(|c| c.get(tid)));
            let render_fn = || {
                super::turn_render::render_turn(
                    &turn.messages,
                    super::turn_render::RenderMode::Archived { terminal_state },
                    super::turn_render::RENDERER_VERSION,
                    &render_options,
                )
            };
            let (bytes, hit, reason) = super::turn_render_cache::render_cache_decision(
                prev,
                &fp,
                super::turn_render::RENDERER_VERSION,
                "archived",
                render_fn,
            );
            // Determinism guard: on a cache HIT, debug/test builds re-render and
            // assert the cached bytes still match (nondeterminism = panic). On a
            // miss `bytes` is itself a fresh render, so re-rendering would compare
            // a fresh render against itself — skip it.
            #[cfg(debug_assertions)]
            if hit {
                let fresh = super::turn_render::render_turn(
                    &turn.messages,
                    super::turn_render::RenderMode::Archived { terminal_state },
                    super::turn_render::RENDERER_VERSION,
                    &render_options,
                );
                assert_eq!(
                    fresh, bytes,
                    "archived render must be deterministic for turn {:?}",
                    turn.turn_id
                );
            }
            archived_renders.push(ArchivedRender {
                turn_id: turn.turn_id.clone(),
                turn_seq: turn.turn_seq,
                content_fp: fp,
                bytes,
                cache_hit: hit,
                cache_reason: reason,
            });
        }
    }

    let rendered_for_plan: Vec<super::turn_eviction::RenderedTurn> = archived_renders
        .iter()
        .map(|r| super::turn_eviction::RenderedTurn {
            turn_seq: r.turn_seq,
            est_tokens: crate::memory::context_window::estimate_tokens(
                &serde_json::to_string(&r.bytes).unwrap_or_default(),
            ),
        })
        .collect();
    let plan = super::turn_eviction::plan_eviction(&rendered_for_plan, archived_budget);

    if plan.degenerate {
        warn!(
            session_id,
            iteration,
            archived_budget,
            core_tokens,
            tool_tokens = tool_tokens_for_budget,
            tail_tokens,
            current_turn_reserve = CURRENT_TURN_RESERVE_TOKENS,
            "Archived budget degenerate: non-evictable components exceed context; zero archived turns"
        );
    }

    if plan.evicted_count > 0 {
        // Advance the anchor, drop evicted renders, and prune their cache entries.
        let evicted: Vec<ArchivedRender> = archived_renders.drain(..plan.evicted_count).collect();
        {
            let mut anchors = agent.turn_anchors.write().await;
            anchors.insert(session_id.to_string(), plan.new_anchor_turn_seq);
        }
        {
            let mut cache = agent.turn_renders.write().await;
            if let Some(session_cache) = cache.get_mut(session_id) {
                for r in &evicted {
                    if let Some(tid) = r.turn_id.as_deref() {
                        session_cache.remove(tid);
                    }
                }
            }
        }
        info!(
            session_id,
            iteration,
            new_anchor = plan.new_anchor_turn_seq,
            turns_evicted = plan.evicted_count,
            kept_est_tokens = plan.kept_est_tokens,
            archived_kept = archived_renders.len(),
            archived_budget,
            "Window decision"
        );
    }
    if plan.degenerate {
        // Degenerate: carry zero archived turns.
        archived_renders.clear();
    }

    // Step 4 (storage): persist the kept archived renders into the cache and
    // emit per-turn hit/miss logging.
    {
        let mut cache = agent.turn_renders.write().await;
        let session_cache = cache.entry(session_id.to_string()).or_default();
        for r in &archived_renders {
            if r.cache_hit {
                tracing::debug!(
                    session_id,
                    turn_id = ?r.turn_id,
                    "archived render cache hit"
                );
            } else {
                info!(
                    session_id,
                    turn_id = ?r.turn_id,
                    reason = r.cache_reason,
                    "archived render cache miss"
                );
            }
            if let Some(tid) = r.turn_id.as_deref() {
                session_cache.insert(
                    tid.to_string(),
                    super::turn_render_cache::CachedRender {
                        content_fp: r.content_fp.clone(),
                        renderer_version: super::turn_render::RENDERER_VERSION,
                        mode_tag: "archived".to_string(),
                        bytes: r.bytes.clone(),
                    },
                );
            }
        }
    }

    // Step 5: render the current turn (full / append-only).
    let current_rendered = super::turn_render::render_turn(
        &current_turn.messages,
        super::turn_render::RenderMode::Current,
        super::turn_render::RENDERER_VERSION,
        &render_options,
    );

    // Step 6 (assembly, part 1): archived turns first, then the current turn.
    // Pillar A tail + core insertion happen further below, unchanged.
    let mut messages: Vec<Value> = Vec::new();
    for r in &archived_renders {
        messages.extend(r.bytes.iter().cloned());
    }
    // The current region begins here. Current-region fitting (step 7) re-locates
    // it by the current user message (content == user_text) AFTER provider-validity
    // fixups, so no fragile index is threaded across the intervening passes.
    messages.extend(current_rendered);

    // Three-pass fixup: merge → drop orphans → merge again (provider-validity).
    fixup_message_ordering(&mut messages);

    tracing::debug!(
        session_id,
        iteration,
        stage = "turn_anchored_assembly",
        archived_turns = archived_renders.len(),
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Execution checkpoint (iteration > 1): a system reminder of the active
    // request + completed work, anchored to the CURRENT turn's messages (the
    // current interaction). Appended at the tail further below.
    let execution_checkpoint = if iteration > 1 {
        let current_interaction: Vec<&Message> = current_turn.messages.iter().collect();
        build_execution_checkpoint_message(user_text, completed_tool_calls, &current_interaction)
    } else {
        None
    };

    // Ensure the current user message is in the context.
    // The DB write (append_user_message_with_event) may not yet be visible
    // to load_recent_history due to a race condition, especially on
    // iteration 1. It can also be missing on iteration 2 after a
    // early-iteration `continue` (no messages are stored between iterations,
    // so the race condition persists). Check all messages on every iteration
    // to be safe — the content match prevents duplicates.
    {
        let has_current_user_msg = messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("user")
                && m.get("content").is_some_and(|content| {
                    crate::agent::vision::user_message_content_matches(content, user_text)
                })
        });

        if !has_current_user_msg {
            let turn_attachments = current_turn_user_attachments(&current_turn, user_text);
            let attachments: &[MessageAttachment] = if !turn_attachments.is_empty() {
                &turn_attachments
            } else {
                current_attachments
            };
            let build = build_multimodal_content(
                user_text,
                attachments,
                RenderMode::Current,
                &render_options.vision,
            );
            messages.push(json!({
                "role": "user",
                "content": build.content,
            }));
        }
    }

    // Task boundary marker: when there are multiple user messages in context
    // (i.e., multiple independent tasks in the same chat session), inject a
    // system separator before the current user message so the LLM knows which
    // task is current. Without this, models confuse old tasks with the new one.
    // Injected on ALL iterations (not just early ones) because on iteration 3+
    // old user messages can mislead the model into responding to them instead of
    // the current task — especially after tool calls push the current user message
    // further up the context.
    {
        let user_positions: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            .map(|(i, _)| i)
            .collect();
        if user_positions.len() >= 2 {
            // Find the position of the *current* user message — match by content,
            // not just "last user message", so we correctly anchor even when
            // stray user messages from other interactions appear after ours.
            let current_pos = user_positions
                .iter()
                .copied()
                .rev()
                .find(|&pos| {
                    messages[pos].get("content").is_some_and(|content| {
                        crate::agent::vision::user_message_content_matches(content, user_text)
                    })
                })
                .or_else(|| user_positions.last().copied());

            if let Some(current_pos) = current_pos {
                let prev_user_content = user_positions
                    .iter()
                    .copied()
                    .filter(|&pos| pos != current_pos)
                    .rev()
                    .find_map(|pos| {
                        messages[pos]
                            .get("content")
                            .and_then(|c| c.as_str())
                            .map(|s| s.to_string())
                    });
                // Only inject if a different task exists in context.
                let has_different_task =
                    prev_user_content.as_deref() != Some(user_text) && prev_user_content.is_some();
                if has_different_task {
                    // Softer marker that tells the LLM which message is current
                    // without telling it to ignore prior context. The old [TASK BOUNDARY]
                    // marker aggressively instructed the LLM to ignore prior messages,
                    // which broke follow-up references like "the ones within 20 miles".
                    let marker = json!({
                        "role": "system",
                        "content": "[Current Task] The message below is the user's current request. \
                                    Prior messages are conversation history for context."
                    });
                    messages.insert(current_pos, marker);
                    info!(
                        session_id,
                        iteration,
                        user_messages = user_positions.len(),
                        "Current task marker injected before current user message"
                    );
                }
            }
        }
    }

    // Phase 0 stage hash: the `[Current Task]` marker is a system message
    // inserted just before the current user (boundary) message, so it falls in
    // the pre-boundary region. Marker movement between turns is a known cache
    // boundary; this stage makes it attributable.
    tracing::debug!(
        session_id,
        iteration,
        stage = "current_task_marker",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Guard against context interleaving: if another user message arrived in
    // this session while the agent was processing (race condition between task
    // registration and queuing), it may appear after the current task's tool
    // chain. Such stray user messages confuse the model into responding to them
    // instead of the current task. Remove them.
    {
        let current_task_pos = find_current_user_position(&messages, user_text);
        if let Some(task_pos) = current_task_pos {
            // Find the end of the current task's tool chain (last assistant/tool after task_pos)
            let chain_end = messages
                .iter()
                .enumerate()
                .rev()
                .find(|(i, m)| {
                    *i > task_pos
                        && matches!(
                            m.get("role").and_then(|r| r.as_str()),
                            Some("assistant") | Some("tool")
                        )
                })
                .map(|(i, _)| i)
                .unwrap_or(task_pos);

            // Check for user messages after the tool chain
            let stray_start = chain_end + 1;
            if stray_start < messages.len() {
                let stray_count = messages[stray_start..]
                    .iter()
                    .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                    .count();
                if stray_count > 0 {
                    messages.truncate(stray_start);
                    info!(
                        session_id,
                        iteration,
                        stray_user_messages = stray_count,
                        "Truncated stray messages after current task's tool chain"
                    );
                }
            }
        }
    }

    // Collapse repeated tool errors in the current interaction to reduce
    // context blow-up during retry loops (keep the latest error details).
    let collapsed_tool_errors = super::loop_utils::collapse_repeated_tool_errors(&mut messages);
    if collapsed_tool_errors > 0 {
        // Pillar B (Task 8): collapsing earlier repeated-error payloads into short
        // notes rewrites stable-region bytes. Attribute it so the prefix-invariant
        // gate (Task 11) does not read this as an unattributed prefix break.
        info!(
            session_id,
            iteration,
            collapsed_tool_errors,
            reason = "repeated_tool_error_collapse",
            "Prefix mutation"
        );
    }
    // Phase 0 stage hash: repeated tool-error collapse. Repeated-error collapse
    // operates on the current interaction (at/after the boundary), but is
    // instrumented for completeness and to confirm it does not perturb the
    // pre-boundary region.
    tracing::debug!(
        session_id,
        iteration,
        stage = "tool_error_collapse",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Context window enforcement: trim messages to fit token budget.
    //
    // Pillar B (Task 7): fitting is scoped to the CURRENT-TURN region only.
    // Archived turns are whole-turn-evicted upstream (spec invariant 3) and must
    // never be trimmed here — only the current turn (everything from the current
    // user message onward) is handed to `fit_messages_with_source_quotas`.
    if agent.context_window_config.enabled {
        // Reserve for BOTH message zero (core) and the task context tail.
        let system_tokens = crate::memory::context_window::estimate_tokens(core_prompt)
            + crate::memory::context_window::estimate_tokens(task_context_tail);
        let model_budget = crate::memory::context_window::compute_available_budget_precomputed(
            model,
            system_tokens,
            tool_defs,
            &agent.context_window_config,
        );
        let policy_budget = policy_bundle.policy.context_budget;
        if agent.policy_config.policy_shadow_mode && !agent.policy_config.policy_enforce {
            info!(
                session_id,
                iteration, model_budget, policy_budget, "Context budget shadow comparison"
            );
        }
        let effective_budget = if agent.policy_config.policy_enforce {
            // Never exceed the model's budget; policy config can be mis-set.
            policy_budget.min(model_budget)
        } else {
            model_budget
        };
        // Locate the current-turn region: from the current user message (last
        // occurrence matching `user_text`) to the end of `messages`. Archived
        // turns precede it and are left untouched.
        let current_region_idx = find_current_user_position(&messages, user_text).unwrap_or(0);
        let archived_prefix: Vec<Value> = messages[..current_region_idx].to_vec();
        let current_region: Vec<Value> = messages[current_region_idx..].to_vec();
        // The archived prefix already consumed `archived_budget`; the current
        // region is fit against the remaining budget.
        let archived_prefix_tokens = crate::memory::context_window::estimate_tokens(
            &serde_json::to_string(&archived_prefix).unwrap_or_default(),
        );
        let current_budget = effective_budget.saturating_sub(archived_prefix_tokens);
        let (fitted_current, dropped) =
            crate::memory::context_window::fit_messages_with_source_quotas(
                current_region,
                current_budget,
            );
        messages = archived_prefix;
        messages.extend(fitted_current);
        if dropped > 0 {
            // Pillar B (Task 8): fitting rewrote stable-region bytes by dropping
            // current-turn messages. Attribute it so the prefix-invariant gate
            // (Task 11) does not read this as an unattributed prefix break.
            info!(
                session_id,
                iteration,
                dropped,
                reason = "history_fitting",
                "Prefix mutation"
            );
        }
    }
    // Phase 0 stage hash: history fitting. Fitting is now current-region only;
    // archived turns are whole-turn-evicted upstream and never trimmed here.
    tracing::debug!(
        session_id,
        iteration,
        stage = "history_fitting",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    // Empty-response recovery: on retry, clear conversational history to avoid
    // repeatedly sending a poisoned context to the provider (Gemini in particular
    // can get "stuck" returning empty candidates for a given session history).
    if empty_response_retry_pending && !is_trigger_session(session_id) {
        let before = messages.len();
        let rebuilt = build_empty_response_retry_messages(&messages, user_text);
        // Fire the prefix-mutation attribution ONLY when the rebuild actually
        // changes the messages (non-zero effect), so quiet turns stay silent.
        let mutated = rebuilt != messages;
        messages = rebuilt;
        info!(
            session_id,
            iteration,
            before,
            after = messages.len(),
            "Empty-response recovery: reduced history while preserving immediate parent context"
        );
        if mutated {
            // Pillar B (Task 8): the retry rebuild rewrote stable-region bytes by
            // collapsing prior history. Attribute it so the prefix-invariant gate
            // (Task 11) does not read this as an unattributed prefix break.
            info!(
                session_id,
                iteration,
                reason = "empty_response_retry",
                "Prefix mutation"
            );
        }
    }

    // Pillar A: insert the per-task context TAIL immediately BEFORE the current
    // user message (boundary − 1). The tail is a single `role:"system"` message
    // whose content starts with `TASK_CONTEXT_TAIL_MARKER`; the provider-call
    // fingerprint locates it by that marker. The session summary, current
    // date/time, session context, query-ranked memory, matched skill bodies, and
    // resume checkpoint all live INSIDE this string (compiled once per task in
    // bootstrap and reused byte-identically across the within-task loop).
    //
    // This insertion happens BEFORE message zero is inserted, so the boundary is
    // located against the current `messages` (no leading system prompt yet).
    if !task_context_tail.is_empty() {
        let tail_insert_pos =
            find_current_user_position(&messages, user_text).unwrap_or(messages.len());
        messages.insert(
            tail_insert_pos,
            json!({
                "role": "system",
                "content": task_context_tail,
            }),
        );
    }

    // Keep message zero byte-stable across iterations so llama.cpp can reuse the
    // expensive system-prompt prefix. Message zero is the session-static CORE
    // prompt ONLY — volatile per-turn material lives in the task context tail
    // inserted above (Pillar A).
    messages.insert(
        0,
        json!({
            "role": "system",
            "content": core_prompt,
        }),
    );

    // Phase 0 stage hash: the task context tail sits at boundary − 1 (inside the
    // pre-boundary region). Message zero is the core prompt;
    // `stage_pre_boundary_hash` skips that leading system message, so the tail is
    // included in the pre-boundary hash and tail churn is attributable. (The
    // session-summary stage was retired with the index-1 summary insertion; the
    // provider-call `tail_hash` covers tail attribution at the call boundary.)
    tracing::debug!(
        session_id,
        iteration,
        stage = "context_tail",
        pre_boundary_hash = %super::prefix_fingerprint::stage_pre_boundary_hash(&messages, user_text),
        "Build stage pre-boundary fingerprint"
    );

    if let Some(checkpoint) = execution_checkpoint {
        messages.push(json!({
            "role": "system",
            "content": checkpoint,
        }));
        info!(
            session_id,
            iteration, "Injected execution checkpoint for in-progress task continuity"
        );
    }
    // Phase 0 stage hash: execution-checkpoint insertion. The checkpoint is
    // appended at the tail (at/after the boundary) and cannot flip the
    // pre-boundary hash, so this stage emits a full-payload hash that tracks
    // tail growth for completeness rather than a pre-boundary hash. The
    // `serde_json::Value::Array` clone is built only inside the `debug!` field
    // expression, so it runs only when debug logging is enabled.
    tracing::debug!(
        session_id,
        iteration,
        stage = "execution_checkpoint",
        full_payload_hash = %super::prefix_fingerprint::hash_canonical(&serde_json::Value::Array(messages.clone())),
        "Build stage tail fingerprint"
    );

    // Fresh-context isolation: when history is empty or only contains the current
    // user message (e.g. first message after /clear), inject a boundary marker to
    // prevent the LLM from drifting toward stale tool-call patterns from pinned
    // memories or prior context.
    {
        let non_system_non_user_count = messages
            .iter()
            .filter(|m| {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                role != "system" && role != "user"
            })
            .count();
        if non_system_non_user_count == 0 {
            messages.retain(|m| {
                m.get("role").and_then(|r| r.as_str()) != Some("user")
                    || m.get("content")
                        .is_some_and(|content| user_message_content_matches(content, user_text))
            });
            pending_system_messages.push(SystemDirective::FreshConversationContext);
        }
    }

    // System nudges (budget warnings, loop-stop reminders, etc.): inject for a single
    // LLM call so they influence the model without polluting stored history.
    for directive in pending_system_messages.drain(..) {
        messages.push(json!({
            "role": "system",
            "content": directive.render(),
        }));
    }

    // Empty-response recovery: if the prior iteration produced no text and no tool calls,
    // inject a system nudge for the next LLM call. (Tool-role nudges are dropped by
    // message-order fixups because they don't correspond to an assistant tool_call_id.)
    if empty_response_retry_pending && !is_trigger_session(session_id) {
        messages.push(json!({
            "role": "system",
            "content": SystemDirective::EmptyResponseRetry.render()
        }));
    }

    // Serialize messages once; reused for final-enforcement token count, debug logging,
    // and the final est_input_tokens. Messages are not mutated after this point.
    let messages_json = serde_json::to_string(&messages).unwrap_or_default();

    // Final enforcement must happen after every prompt component has been inserted.
    // Earlier trimming cannot account for execution checkpoints and one-shot directives.
    if agent.context_window_config.enabled {
        let message_tokens = crate::memory::context_window::estimate_tokens(&messages_json);
        let final_tool_budget = total_context_budget.saturating_sub(
            message_tokens + RESPONSE_RESERVE_TOKENS + TOKEN_ESTIMATE_SAFETY_MARGIN,
        );
        let final_tool_defs = crate::memory::context_window::fit_tool_definitions_to_budget(
            original_tool_defs,
            final_tool_budget,
        );
        if final_tool_defs != effective_tool_defs {
            info!(
                session_id,
                iteration,
                model,
                message_tokens,
                final_tool_budget,
                before_tool_tokens = crate::memory::context_window::estimate_tool_definition_tokens(
                    &effective_tool_defs
                ),
                after_tool_tokens = crate::memory::context_window::estimate_tool_definition_tokens(
                    &final_tool_defs
                ),
                tool_count = final_tool_defs.len(),
                "Recompacted tool schemas after final prompt assembly"
            );
            effective_tool_defs = final_tool_defs;
            tool_defs = effective_tool_defs.as_slice();
        }
    }

    // Emit "Thinking" status for iterations after the first
    if iteration > 1 {
        send_status(status_tx, StatusUpdate::Thinking(iteration));
    }

    // Debug: log message structure and estimated token count
    {
        let summary: Vec<String> = messages
            .iter()
            .map(|m| {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                let name = m.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let tc_id = m
                    .get("tool_call_id")
                    .and_then(|id| id.as_str())
                    .unwrap_or("");
                let tc_count = m
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map_or(0, |a| a.len());
                if role == "tool" {
                    format!("tool({},tc_id={})", name, &tc_id[..tc_id.len().min(12)])
                } else if tc_count > 0 {
                    format!("{}(tc={})", role, tc_count)
                } else {
                    role.to_string()
                }
            })
            .collect();

        // Estimate tokens: ~4 chars per token for English text
        let est_msg_tokens = messages_json.len() / 4;
        let est_tool_tokens =
            crate::memory::context_window::estimate_tool_definition_tokens(tool_defs);
        let est_total_tokens = est_msg_tokens + est_tool_tokens;
        let est_msg_tokens_u64 = est_msg_tokens as u64;
        let est_tool_tokens_u64 = est_tool_tokens as u64;
        let est_total_tokens_u64 = est_total_tokens as u64;
        let est_tool_share_bps = est_tool_tokens_u64
            .saturating_mul(10_000)
            .checked_div(est_total_tokens_u64)
            .unwrap_or(0);

        // Runtime signal: quantify prompt overhead from tool schemas before each LLM call.
        POLICY_METRICS
            .est_input_token_samples
            .fetch_add(1, Ordering::Relaxed);
        POLICY_METRICS
            .est_input_tokens_total
            .fetch_add(est_total_tokens_u64, Ordering::Relaxed);
        POLICY_METRICS
            .est_msg_tokens_total
            .fetch_add(est_msg_tokens_u64, Ordering::Relaxed);
        POLICY_METRICS
            .est_tool_tokens_total
            .fetch_add(est_tool_tokens_u64, Ordering::Relaxed);

        const HIGH_TOOL_SHARE_BPS: u64 = 3500; // >=35% of input estimate
        const HIGH_TOOL_TOKENS_ABS: u64 = 1_500; // large absolute tool-schema cost
        if est_tool_share_bps >= HIGH_TOOL_SHARE_BPS {
            POLICY_METRICS
                .est_tool_tokens_high_share_total
                .fetch_add(1, Ordering::Relaxed);
        }
        if est_tool_tokens_u64 >= HIGH_TOOL_TOKENS_ABS {
            POLICY_METRICS
                .est_tool_tokens_high_abs_total
                .fetch_add(1, Ordering::Relaxed);
        }

        info!(
            session_id,
            iteration,
            est_input_tokens = est_total_tokens,
            est_msg_tokens,
            est_tool_tokens,
            total_context_budget,
            response_reserve_tokens = RESPONSE_RESERVE_TOKENS,
            est_tool_share_pct = est_tool_share_bps as f64 / 100.0,
            msg_count = messages.len(),
            msgs = ?summary,
            "Context before LLM call"
        );
    }

    // Pillar A Task 6: name-sort the emitted roster as the FINAL operation before
    // constructing MessageBuildData. This is the authoritative guarantee that the
    // provider tool array is in canonical order regardless of any late
    // append/filter/widen/compaction that mutated `effective_tool_defs` above.
    // Providers stay order-preserving (no sort in adapters).
    Agent::sort_tool_definitions_by_name(&mut effective_tool_defs);

    let est_input_tokens = {
        let est_msg_tokens = messages_json.len() / 4;
        let est_tool_tokens =
            crate::memory::context_window::estimate_tool_definition_tokens(&effective_tool_defs);
        (est_msg_tokens + est_tool_tokens) as u32
    };

    Ok(MessageBuildData {
        messages,
        tool_defs: effective_tool_defs,
        est_input_tokens,
    })
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
            ..Message::runtime_defaults()
        }
    }

    fn tool_msg(name: &str, content: &str) -> Message {
        Message {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".to_string(),
            role: "tool".to_string(),
            content: Some(content.to_string()),
            tool_call_id: Some(format!("tool-call-{}", uuid::Uuid::new_v4())),
            tool_name: Some(name.to_string()),
            tool_calls_json: None,
            created_at: Utc::now(),
            importance: 0.5,
            ..Message::runtime_defaults()
        }
    }

    // ====================================================================
    // Pillar B (Task 7): turn-anchored build-phase test scaffolding.
    //
    // The turn-anchored fetch reads the CANONICAL `events` table (rows with a
    // non-NULL `turn_id`), not the `messages` table. These helpers seed whole
    // turns into events over the harness's shared pool so the build phase
    // reconstructs them via `get_turns_from_anchor`.
    // ====================================================================

    use crate::events::{Event, EventStore, EventType};

    /// Emit a complete user→assistant(+tool_call)→tool→assistant turn into the
    /// events table under `turn_id`. `tool` is `Some((name, result))` to include
    /// a tool step. `final_assistant` is the turn's winning reply. A `task_end`
    /// with `status` closes the turn so the renderer derives the terminal state.
    async fn seed_turn(
        store: &EventStore,
        session: &str,
        turn_id: &str,
        user: &str,
        tool: Option<(&str, &str)>,
        final_assistant: &str,
        status: &str,
    ) {
        store
            .append(Event::new(
                session,
                EventType::UserMessage,
                json!({ "content": user, "turn_id": turn_id }),
            ))
            .await
            .expect("seed user_message");
        if let Some((name, result)) = tool {
            let call_id = format!("call-{turn_id}-{name}");
            store
                .append(Event::new(
                    session,
                    EventType::AssistantResponse,
                    json!({
                        "content": serde_json::Value::Null,
                        "tool_calls": [{ "id": call_id, "name": name, "arguments": "{}" }],
                        "turn_id": turn_id,
                    }),
                ))
                .await
                .expect("seed assistant tool-call");
            store
                .append(Event::new(
                    session,
                    EventType::ToolResult,
                    json!({
                        "tool_call_id": call_id,
                        "name": name,
                        "result": result,
                        "success": true,
                        "duration_ms": 1,
                        "turn_id": turn_id,
                    }),
                ))
                .await
                .expect("seed tool_result");
        }
        store
            .append(Event::new(
                session,
                EventType::AssistantResponse,
                json!({ "content": final_assistant, "turn_id": turn_id }),
            ))
            .await
            .expect("seed final assistant");
        store
            .append(Event::new(
                session,
                EventType::TaskEnd,
                json!({ "status": status, "turn_id": turn_id }),
            ))
            .await
            .expect("seed task_end");
    }

    /// A sibling `EventStore` over the harness's shared pool — the agent's own
    /// `event_store` reads the same DB, so seeded events are visible to the
    /// build phase.
    async fn seed_store(harness: &crate::testing::TestHarness) -> EventStore {
        EventStore::new(harness.state.pool())
            .await
            .expect("sibling event store")
    }

    /// Stash `turn_id` as the session's current turn so the build phase matches
    /// the current turn by id (mirrors what bootstrap does on a real turn).
    async fn set_current_turn(harness: &crate::testing::TestHarness, session: &str, turn_id: &str) {
        harness
            .agent
            .current_turn_ids
            .write()
            .await
            .insert(session.to_string(), turn_id.to_string());
    }

    #[test]
    fn empty_retry_preserves_parent_pair_and_current_user() {
        let messages = vec![
            json!({"role": "user", "content": "can you clear cache using drush?"}),
            json!({"role": "assistant", "content": "I can see updates available. Should I proceed with updating these?"}),
            json!({"role": "user", "content": "yes, update them"}),
        ];
        let recovered = build_empty_response_retry_messages(&messages, "yes, update them");
        assert_eq!(recovered.len(), 3);
        assert_eq!(recovered[0]["role"], "user");
        assert_eq!(recovered[1]["role"], "assistant");
        assert_eq!(recovered[2]["role"], "user");
        assert_eq!(recovered[2]["content"].as_str(), Some("yes, update them"));
    }

    #[test]
    fn empty_retry_falls_back_to_current_user_when_no_history() {
        let messages = vec![json!({"role": "user", "content": "help"})];
        let recovered = build_empty_response_retry_messages(&messages, "help");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0]["role"], "user");
        assert_eq!(recovered[0]["content"].as_str(), Some("help"));
    }

    #[tokio::test]
    async fn later_iterations_include_execution_checkpoint_after_tool_progress() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        // Pillar B: seed the CURRENT (in-flight) turn into events — user message
        // plus a completed system_info tool step. The current turn is rendered
        // full/append-only, so the checkpoint can surface the tool evidence.
        // No task_end: the turn is still in progress at iteration 2.
        let store = seed_store(&harness).await;
        let turn = "turn-checkpoint";
        store
            .append(Event::new(
                "test-session",
                EventType::UserMessage,
                json!({ "content": "Find the system details and summarize them.", "turn_id": turn }),
            ))
            .await
            .expect("seed current user");
        store
            .append(Event::new(
                "test-session",
                EventType::AssistantResponse,
                json!({
                    "content": serde_json::Value::Null,
                    "tool_calls": [{ "id": "call-sysinfo", "name": "system_info", "arguments": "{}" }],
                    "turn_id": turn,
                }),
            ))
            .await
            .expect("seed assistant tool-call");
        store
            .append(Event::new(
                "test-session",
                EventType::ToolResult,
                json!({
                    "tool_call_id": "call-sysinfo",
                    "name": "system_info",
                    "result": "OS: macOS 15.0\nMemory: 16 GB\nHostname: dev-machine",
                    "success": true,
                    "duration_ms": 1,
                    "turn_id": turn,
                }),
            ))
            .await
            .expect("seed tool_result");
        set_current_turn(&harness, "test-session", turn).await;

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Find the system details and summarize them.",
            current_attachments: &[],
            completed_tool_calls: &completed_tool_calls,
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        assert!(
            serialized.contains("EXECUTION CHECKPOINT"),
            "later iterations should carry a live execution checkpoint: {}",
            serialized
        );
        assert!(
            serialized.contains("Find the system details and summarize them."),
            "checkpoint should restate the active request: {}",
            serialized
        );
        assert!(
            serialized.contains("system_info"),
            "checkpoint should include completed tool/evidence context: {}",
            serialized
        );
        assert!(
            serialized.contains("Do NOT reset into a generic availability reply"),
            "checkpoint should explicitly block idle reset replies: {}",
            serialized
        );
    }

    #[tokio::test]
    async fn later_iterations_preserve_system_prompt_prefix_without_duplicate_guidance() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Inspect the repository."))
            .await
            .expect("append user");

        let system_prompt =
            "## Identity\nStable identity.\n\n## Tools\nVerbose tool guidance.\n\n## Behavior\nBe precise.";
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut first_pending_system_messages = Vec::new();
        let mut first_ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Inspect the repository.",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: system_prompt,
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut first_pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let first = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut first_ctx,
        )
        .await
        .expect("first message build");

        let mut second_pending_system_messages = Vec::new();
        let mut second_ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Inspect the repository.",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: system_prompt,
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut second_pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let second = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut second_ctx,
        )
        .await
        .expect("second message build");
        let first_system = first.messages[0]["content"]
            .as_str()
            .expect("first system content");
        let second_system = second.messages[0]["content"]
            .as_str()
            .expect("second system content");

        assert_eq!(first_system, system_prompt);
        assert_eq!(
            second_system, first_system,
            "message zero must remain byte-identical for prompt-cache reuse"
        );
        for (iteration, built) in [(1, &first), (2, &second)] {
            assert!(
                built.messages.iter().skip(1).all(|message| {
                    !message
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| content.contains("Stable identity."))
                }),
                "iteration {iteration} must not duplicate the system prompt later in the request"
            );
        }
    }

    /// Pillar B (Task 7): historical conversation comes ONLY from the
    /// turn-anchored fetch over canonical `events` (rows with a non-NULL
    /// `turn_id`). Plain `messages`-table rows that never became turn-stamped
    /// events (e.g. legacy/idle-gap pairs) are NOT reconstructed, so they never
    /// leak into the payload — the current user turn is the only content. (This
    /// supersedes the deleted idle-gap sliding-window reset.)
    #[tokio::test]
    async fn non_event_history_is_not_reconstructed_into_payload() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;
        use chrono::Duration as ChronoDuration;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");

        // Insert old messages with timestamps > 2 hours ago.
        let old_time = Utc::now() - ChronoDuration::hours(3);
        let old_user = Message {
            created_at: old_time,
            ..msg("user", "Old stale question from 3 hours ago")
        };
        let old_assistant = Message {
            created_at: old_time,
            ..msg("assistant", "Old stale answer from 3 hours ago")
        };
        harness
            .state
            .append_message(&old_user)
            .await
            .expect("append old user");
        harness
            .state
            .append_message(&old_assistant)
            .await
            .expect("append old assistant");

        // Insert current user message (now).
        harness
            .state
            .append_message(&msg("user", "Fresh question now"))
            .await
            .expect("append current user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Fresh question now",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");
        let serialized = serde_json::to_string(&built.messages).expect("serialize messages");

        // Old stale messages should NOT be present after idle gap reset.
        assert!(
            !serialized.contains("Old stale question from 3 hours ago"),
            "idle gap should reset window to 0, removing old stale pairs: {}",
            serialized
        );
        assert!(
            !serialized.contains("Old stale answer from 3 hours ago"),
            "idle gap should reset window to 0, removing old stale assistant: {}",
            serialized
        );

        // Current user message must still be present.
        assert!(
            serialized.contains("Fresh question now"),
            "current user message should always be present: {}",
            serialized
        );
    }

    /// Pillar A: the session summary now travels INSIDE the per-task context
    /// tail (compiled in bootstrap). Message-build no longer takes a summary
    /// argument; instead it inserts the tail (containing `[Session Summary]`) at
    /// boundary − 1 as a single system message. This test passes the summary via
    /// `task_context_tail` and asserts it lands in the tail message and NOT at a
    /// separate index-1 message.
    #[tokio::test]
    async fn session_summary_travels_inside_task_context_tail() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");

        harness
            .state
            .append_message(&msg("user", "Current question"))
            .await
            .expect("append user");

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Session Summary]\nUser previously asked about deploying a blog. Config was created."
        );
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Current question",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: &tail,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        // The summary lives inside the tail message (starts with the marker).
        let tail_msg = built.messages.iter().find(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("system")
                && m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
        });
        let tail_content = tail_msg
            .and_then(|m| m["content"].as_str())
            .expect("tail message must be present");
        assert!(
            tail_content.contains("[Session Summary]"),
            "summary must live inside the task context tail: {tail_content}"
        );
        assert!(
            tail_content.contains("deploying a blog"),
            "summary content must be present in the tail: {tail_content}"
        );

        // No separate index-1 [Session Summary] message exists anymore.
        let summary_only_messages = built
            .messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Session Summary]"))
            })
            .count();
        assert_eq!(
            summary_only_messages, 1,
            "summary must appear exactly once (inside the tail), not as a separate message"
        );
    }

    #[tokio::test]
    async fn small_context_model_compacts_tool_schemas_without_dropping_tools() {
        use crate::execution_policy::PolicyBundle;
        use crate::memory::context_window::{estimate_tokens, estimate_tool_definition_tokens};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .agent
            .context_window_config
            .model_budgets
            .insert("gemma-4-26b".to_string(), 16_384);
        harness
            .state
            .append_message(&msg("user", "List all available tools."))
            .await
            .expect("append user");

        let verbose =
            "Detailed operational guidance for selecting and safely using this tool. ".repeat(300);
        let tool_defs: Vec<Value> = (0..20)
            .map(|idx| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": verbose
                                },
                                "mode": {
                                    "type": "string",
                                    "description": verbose,
                                    "enum": ["read", "write"]
                                }
                            },
                            "required": ["path"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();
        assert!(estimate_tool_definition_tokens(&tool_defs) > 16_384);

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "List all available tools.",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "gemma-4-26b",
            core_prompt: "You are a helpful test assistant.",
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        assert_eq!(built.tool_defs.len(), tool_defs.len());
        // Pillar A: the emitted roster is now name-sorted (lexicographic), so
        // assert the SET of tool names is preserved and parameter contracts are
        // intact — not a positional numeric order.
        let got_names: std::collections::HashSet<String> = built
            .tool_defs
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect();
        let expected_names: std::collections::HashSet<String> = (0..tool_defs.len())
            .map(|idx| format!("tool_{idx}"))
            .collect();
        assert_eq!(got_names, expected_names, "all tools must be preserved");
        for tool in &built.tool_defs {
            assert_eq!(
                tool["function"]["parameters"]["properties"]["mode"]["enum"],
                json!(["read", "write"])
            );
        }
        // Confirm the final order is name-sorted (the authoritative final sort).
        let ordered_names: Vec<&str> = built
            .tool_defs
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        let mut sorted = ordered_names.clone();
        sorted.sort();
        assert_eq!(ordered_names, sorted, "tool order must be name-sorted");

        let message_tokens =
            estimate_tokens(&serde_json::to_string(&built.messages).expect("serialize messages"));
        let tool_tokens = estimate_tool_definition_tokens(&built.tool_defs);
        assert!(
            message_tokens + tool_tokens + 1_536 <= 16_384,
            "request estimate should fit Gemma context: messages={message_tokens}, tools={tool_tokens}"
        );
    }

    #[tokio::test]
    async fn small_context_model_rechecks_budget_after_final_prompt_assembly() {
        use crate::execution_policy::PolicyBundle;
        use crate::memory::context_window::{estimate_tokens, estimate_tool_definition_tokens};
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let mut harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .agent
            .context_window_config
            .model_budgets
            .insert("gemma-4-26b".to_string(), 16_384);
        harness
            .state
            .append_message(&msg("user", "Can you test your tools?"))
            .await
            .expect("append user");
        harness
            .state
            .append_message(&tool_msg(
                "system_info",
                "OS: macOS\nMemory: 64 GB\nHostname: workstation",
            ))
            .await
            .expect("append tool");

        let verbose = "Detailed parameter guidance for local agent tool execution. ".repeat(40);
        let tool_defs: Vec<Value> = (0..38)
            .map(|idx| {
                let properties: serde_json::Map<String, Value> = (0..8)
                    .map(|prop_idx| {
                        (
                            format!("parameter_{prop_idx}"),
                            json!({
                                "type": "string",
                                "description": verbose,
                                "enum": ["one", "two", "three"]
                            }),
                        )
                    })
                    .collect();
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{idx}"),
                        "description": verbose,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": ["parameter_0"],
                            "additionalProperties": false
                        }
                    }
                })
            })
            .collect();

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let mut pending_system_messages = vec![SystemDirective::FreshConversationContext];
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let completed_tool_calls = vec!["system_info({})".to_string()];
        let system_prompt = "Root agent operating guidance and tool policy. ".repeat(650);

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 2,
            user_text: "Can you test your tools?",
            current_attachments: &[],
            completed_tool_calls: &completed_tool_calls,
            model: "gemma-4-26b",
            core_prompt: &system_prompt,
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        assert_eq!(built.tool_defs.len(), tool_defs.len());
        let message_tokens =
            estimate_tokens(&serde_json::to_string(&built.messages).expect("serialize messages"));
        let tool_tokens = estimate_tool_definition_tokens(&built.tool_defs);
        assert!(
            message_tokens + tool_tokens + RESPONSE_RESERVE_TOKENS <= 16_384,
            "final assembled request should fit: messages={message_tokens}, tools={tool_tokens}"
        );
    }

    // ---- Pillar A Task 6: payload assembly tests ----

    /// Test 1: exactly one system message starts with TASK_CONTEXT_TAIL_MARKER,
    /// positioned immediately BEFORE the current user message (boundary − 1).
    /// Test 2: no standalone `[Session Summary]` message; the summary appears
    /// ONLY inside the tail.
    /// Test 3: message zero equals the core bytes exactly (no volatile suffix).
    #[tokio::test]
    async fn tail_precedes_current_turn_and_summary_lives_only_in_tail() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Old question"))
            .await
            .expect("append old user");
        harness
            .state
            .append_message(&msg("assistant", "Old answer"))
            .await
            .expect("append old assistant");
        harness
            .state
            .append_message(&msg("user", "Current question"))
            .await
            .expect("append current user");

        let core = "You are aidaemon. CORE PROMPT BODY.";
        let tail = format!(
            "{TASK_CONTEXT_TAIL_MARKER}\n\n[Session Summary]\nUser deploying a blog.\n\n[Current Date & Time]\nMonday"
        );

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "test-session",
            iteration: 1,
            user_text: "Current question",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };

        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build");

        // Test 3: message zero equals the core bytes exactly.
        assert_eq!(
            built.messages[0]["content"].as_str(),
            Some(core),
            "message zero must be the core prompt bytes with no volatile suffix"
        );

        // Test 1: exactly one tail message; it precedes the current user message.
        let tail_positions: Vec<usize> = built
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.get("role").and_then(|r| r.as_str()) == Some("system")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tail_positions.len(), 1, "exactly one tail message expected");
        let tail_pos = tail_positions[0];
        let current_user_pos = built
            .messages
            .iter()
            .rposition(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some("Current question")
            })
            .expect("current user message present");
        assert_eq!(
            tail_pos + 1,
            current_user_pos,
            "tail must sit immediately before the current user message (boundary − 1)"
        );

        // Test 2: no standalone `[Session Summary]` message; summary only in tail.
        assert!(
            !built.messages[1]["content"]
                .as_str()
                .unwrap_or("")
                .starts_with("[Session Summary]"),
            "index 1 must not be a standalone session-summary message"
        );
        let summary_msgs = built
            .messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("[Session Summary]"))
            })
            .count();
        assert_eq!(
            summary_msgs, 1,
            "summary appears exactly once, inside the tail"
        );
    }

    /// Test 4: within-task tail reuse — two consecutive build iterations of the
    /// same task produce a byte-identical tail message.
    #[tokio::test]
    async fn within_task_tail_reuse_is_byte_identical() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Same task"))
            .await
            .expect("append user");

        let core = "CORE";
        let tail = format!("{TASK_CONTEXT_TAIL_MARKER}\n\n[Current Date & Time]\nFixed timestamp");
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let extract_tail = |built: &MessageBuildData| -> String {
            built
                .messages
                .iter()
                .find(|m| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|s| s.starts_with(TASK_CONTEXT_TAIL_MARKER))
                })
                .and_then(|m| m["content"].as_str())
                .expect("tail present")
                .to_string()
        };

        let mut p1 = Vec::new();
        let mut ctx1 = MessageBuildCtx {
            session_id: "reuse-session",
            iteration: 1,
            user_text: "Same task",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut p1,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let built1 = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx1,
        )
        .await
        .expect("build 1");

        let mut p2 = Vec::new();
        let mut ctx2 = MessageBuildCtx {
            session_id: "reuse-session",
            iteration: 2,
            user_text: "Same task",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: core,
            task_context_tail: &tail,
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut p2,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let built2 = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx2,
        )
        .await
        .expect("build 2");

        assert_eq!(
            extract_tail(&built1),
            extract_tail(&built2),
            "tail must be byte-identical across within-task iterations"
        );
    }

    /// Test 5: the final emitted tool order is name-sorted even when the input
    /// roster is unsorted — proving the sort happens as the final op before
    /// MessageBuildData, after any mutation.
    #[tokio::test]
    async fn final_tool_order_is_name_sorted_after_mutations() {
        use crate::execution_policy::PolicyBundle;
        use crate::testing::{setup_test_agent, MockProvider};
        use crate::traits::MessageStore;

        let harness = setup_test_agent(MockProvider::new())
            .await
            .expect("test harness");
        harness
            .state
            .append_message(&msg("user", "Do work"))
            .await
            .expect("append user");

        // Deliberately unsorted roster (zebra, alpha, mango).
        let tool_defs: Vec<Value> = ["zebra_tool", "alpha_tool", "mango_tool"]
            .iter()
            .map(|name| {
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "x",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": false}
                    }
                })
            })
            .collect();

        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let mut pending = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;

        let mut ctx = MessageBuildCtx {
            session_id: "sort-session",
            iteration: 1,
            user_text: "Do work",
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "CORE",
            task_context_tail: "",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        let built = run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("build");

        let names: Vec<&str> = built
            .tool_defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str())
            .collect();
        assert_eq!(
            names,
            vec!["alpha_tool", "mango_tool", "zebra_tool"],
            "final tool_defs must be name-sorted"
        );
    }

    // ====================================================================
    // Pillar B (Task 7): turn-anchored build-phase tests (the 10 from the plan).
    // ====================================================================

    use crate::execution_policy::PolicyBundle;
    use crate::testing::{setup_test_agent, MockProvider, TestHarness};

    /// Build a payload for `session`/`user_text` at `iteration`, using the
    /// turn-anchored path. Returns the built messages.
    async fn build_payload(
        harness: &TestHarness,
        session: &str,
        user_text: &str,
        iteration: usize,
    ) -> Vec<Value> {
        let policy_bundle = PolicyBundle::from_scores(0.1, 0.1, 0.9);
        let tool_defs: Vec<Value> = Vec::new();
        let mut pending_system_messages = Vec::new();
        let status_tx: Option<mpsc::Sender<StatusUpdate>> = None;
        let mut ctx = MessageBuildCtx {
            session_id: session,
            iteration,
            user_text,
            current_attachments: &[],
            completed_tool_calls: &[],
            model: "mock-model",
            core_prompt: "CORE-PROMPT-BYTES",
            task_context_tail: "[Task Context] tail",
            tool_defs: &tool_defs,
            policy_bundle: &policy_bundle,
            pending_system_messages: &mut pending_system_messages,
            empty_response_retry_pending: false,
            status_tx: &status_tx,
        };
        run_message_build_phase(
            &crate::agent::services::AgentServices::new(&harness.agent),
            &mut ctx,
        )
        .await
        .expect("message build")
        .messages
    }

    fn count_occurrences(messages: &[Value], needle: &str) -> usize {
        serde_json::to_string(messages)
            .unwrap_or_default()
            .matches(needle)
            .count()
    }

    /// 1. Archived turns are whole and in Archived form, positioned between the
    ///    core (index 0) and the `[Task Context]` tail.
    #[tokio::test]
    async fn archived_turns_are_whole_and_in_archived_form() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        // Two completed archived turns.
        seed_turn(
            &store,
            "s1",
            "t1",
            "first question",
            Some(("terminal", "exit_code: 0")),
            "first answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s1",
            "t2",
            "second question",
            Some(("read_file", "10 lines")),
            "second answer",
            "completed",
        )
        .await;
        // Current turn (in events) under turn id t3.
        seed_turn(
            &store,
            "s1",
            "t3",
            "third question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s1", "t3").await;

        let messages = build_payload(&harness, "s1", "third question", 1).await;
        let serialized = serde_json::to_string(&messages).unwrap();

        // Core at index 0.
        assert_eq!(messages[0]["content"].as_str(), Some("CORE-PROMPT-BYTES"));
        // Both archived user messages present (whole/full).
        assert!(serialized.contains("first question"), "{serialized}");
        assert!(serialized.contains("second question"), "{serialized}");
        // Archived tool results are SUMMARIZED (compact 1-liner form).
        assert!(
            messages.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("tool")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.starts_with("terminal:"))
            }),
            "archived terminal result should be summarized: {serialized}"
        );
        // Exactly one tail marker, and the archived turns precede it.
        let tail_pos = messages.iter().position(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|c| c.starts_with(TASK_CONTEXT_TAIL_MARKER))
        });
        let tail_pos = tail_pos.expect("tail present");
        let first_q_pos = messages
            .iter()
            .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("first question"))
            .expect("first archived user present");
        assert!(
            first_q_pos < tail_pos,
            "archived turns must precede the tail"
        );
    }

    /// 2. Cross-turn archived stability: archived turn 1's bytes are identical
    ///    when built in turn 2 vs turn 3 (render-cache hit; no fp_mismatch).
    #[tokio::test]
    async fn cross_turn_archived_render_is_byte_stable() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s2",
            "t1",
            "marker-ONE question",
            Some(("terminal", "exit_code: 0")),
            "answer one",
            "completed",
        )
        .await;

        // Build turn 2 (current = t2).
        seed_turn(
            &store,
            "s2",
            "t2",
            "marker-TWO question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s2", "t2").await;
        let build_a = build_payload(&harness, "s2", "marker-TWO question", 1).await;

        // Close t2, open t3 as current.
        store
            .append(Event::new(
                "s2",
                EventType::TaskEnd,
                json!({"status":"completed","turn_id":"t2"}),
            ))
            .await
            .unwrap();
        seed_turn(
            &store,
            "s2",
            "t3",
            "marker-THREE question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s2", "t3").await;
        let build_b = build_payload(&harness, "s2", "marker-THREE question", 1).await;

        // The rendered archived-turn-1 region (everything up to its 'answer one')
        // must be byte-identical across both builds.
        let extract_t1 = |msgs: &[Value]| -> Vec<Value> {
            let end = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("answer one"))
                .expect("answer one present");
            msgs[..=end].to_vec()
        };
        assert_eq!(
            extract_t1(&build_a),
            extract_t1(&build_b),
            "archived turn 1 must be byte-identical across builds (render-cache hit)"
        );
    }

    /// 3. The current turn is full/append-only (NOT summarized).
    #[tokio::test]
    async fn current_turn_is_full_not_summarized() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        let long_user = "DETAILED current request ".repeat(8);
        store
            .append(Event::new(
                "s3",
                EventType::UserMessage,
                json!({"content": long_user, "turn_id":"tc"}),
            ))
            .await
            .unwrap();
        set_current_turn(&harness, "s3", "tc").await;

        let messages = build_payload(&harness, "s3", &long_user, 1).await;
        // The full (untruncated) current user text is present verbatim.
        assert!(
            messages.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("user")
                    && m.get("content").and_then(|c| c.as_str()) == Some(long_user.as_str())
            }),
            "current user message must be full/verbatim"
        );
    }

    /// 4. Tail + core position preserved: exactly one tail marker at boundary−1,
    ///    message zero equals the core bytes.
    #[tokio::test]
    async fn tail_and_core_positions_preserved() {
        use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s4",
            "t1",
            "old question",
            None,
            "old answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s4",
            "t2",
            "current question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s4", "t2").await;

        let messages = build_payload(&harness, "s4", "current question", 1).await;
        assert_eq!(messages[0]["content"].as_str(), Some("CORE-PROMPT-BYTES"));
        let tail_positions: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|c| c.starts_with(TASK_CONTEXT_TAIL_MARKER))
            })
            .map(|(i, _)| i)
            .collect();
        assert_eq!(tail_positions.len(), 1, "exactly one tail marker");
        // Tail sits immediately before the current user message (boundary − 1).
        let tail_idx = tail_positions[0];
        assert_eq!(
            messages[tail_idx + 1]["role"].as_str(),
            Some("user"),
            "tail must sit at boundary − 1 (immediately before current user)"
        );
        assert_eq!(
            messages[tail_idx + 1]["content"].as_str(),
            Some("current question")
        );
    }

    /// 5. Eviction advances the anchor: a tiny archived budget evicts the oldest
    ///    whole turns; no archived turn is partially trimmed.
    #[tokio::test]
    async fn eviction_advances_anchor_and_evicts_whole_turns() {
        let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
        // Tiny model budget so archived_budget is small but positive.
        harness.agent.context_window_config.default_budget = 5900;
        let store = seed_store(&harness).await;
        // Several archived turns with sizeable content so they exceed the budget.
        let big = "lots of detail ".repeat(20);
        for i in 1..=4 {
            seed_turn(
                &store,
                "s5",
                &format!("t{i}"),
                &format!("OLDMARK-{i} {big}"),
                None,
                &format!("answer {i}"),
                "completed",
            )
            .await;
        }
        seed_turn(
            &store,
            "s5",
            "tcur",
            "current request",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s5", "tcur").await;

        let anchor_before = harness.agent.turn_anchors.read().await.get("s5").copied();
        let messages = build_payload(&harness, "s5", "current request", 1).await;
        let anchor_after = harness.agent.turn_anchors.read().await.get("s5").copied();

        assert!(anchor_after.is_some(), "anchor recorded");
        assert!(
            anchor_after != anchor_before || anchor_before.is_none(),
            "eviction should advance the anchor"
        );
        // The oldest turn marker must have been evicted (whole-turn).
        let serialized = serde_json::to_string(&messages).unwrap();
        assert!(
            !serialized.contains("OLDMARK-1"),
            "oldest whole turn should be evicted: {serialized}"
        );
        // Current request always present.
        assert!(serialized.contains("current request"));
    }

    /// 6. Late write re-renders exactly one turn: appending a tool message under
    ///    an already-archived turn flips that turn's content_fp (one re-render),
    ///    the OTHER archived turn stays byte-identical.
    #[tokio::test]
    async fn late_write_re_renders_only_the_affected_turn() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s6",
            "t1",
            "alpha question",
            Some(("terminal", "exit_code: 0")),
            "alpha answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s6",
            "t2",
            "beta question",
            Some(("read_file", "5 lines")),
            "beta answer",
            "completed",
        )
        .await;
        seed_turn(
            &store,
            "s6",
            "t3",
            "current question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s6", "t3").await;

        let first = build_payload(&harness, "s6", "current question", 1).await;
        // Late write under already-archived t1: a complete assistant tool-call +
        // its tool result (a realistic background-notifier append). Both share
        // turn_id t1 and land with ids higher than t1's original rows.
        store
            .append(Event::new(
                "s6",
                EventType::AssistantResponse,
                json!({
                    "content": serde_json::Value::Null,
                    "tool_calls": [{ "id": "late-call", "name": "web_search", "arguments": "{}" }],
                    "turn_id": "t1",
                }),
            ))
            .await
            .unwrap();
        store
            .append(Event::new(
                "s6",
                EventType::ToolResult,
                json!({
                    "tool_call_id": "late-call",
                    "name": "web_search",
                    "result": "LATEWRITE-marker",
                    "success": true,
                    "duration_ms": 1,
                    "turn_id": "t1",
                }),
            ))
            .await
            .unwrap();
        let second = build_payload(&harness, "s6", "current question", 1).await;

        // t2's archived region (its user..answer) must be byte-identical across builds.
        let extract_region = |msgs: &[Value], start_marker: &str, end_marker: &str| -> Vec<Value> {
            let s = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some(start_marker))
                .expect("start marker");
            let e = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some(end_marker))
                .expect("end marker");
            msgs[s..=e].to_vec()
        };
        assert_eq!(
            extract_region(&first, "beta question", "beta answer"),
            extract_region(&second, "beta question", "beta answer"),
            "unaffected archived turn t2 must stay byte-identical"
        );
        // t1's archived region (alpha question .. just before beta question) MUST
        // change: the late tool step adds a new summarized tool message at the END
        // of t1 (highest id within the turn), so the region capture must extend
        // past "alpha answer" up to the start of the next turn.
        let extract_t1 = |msgs: &[Value]| -> Vec<Value> {
            let s = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("alpha question"))
                .expect("alpha question");
            let e = msgs
                .iter()
                .position(|m| m.get("content").and_then(|c| c.as_str()) == Some("beta question"))
                .expect("beta question");
            msgs[s..e].to_vec()
        };
        let first_t1 = extract_t1(&first);
        let second_t1 = extract_t1(&second);
        assert_ne!(
            first_t1, second_t1,
            "affected archived turn t1 must be re-rendered after the late write"
        );
        // The late web_search step appears as a new summarized tool message.
        assert!(
            second_t1.iter().any(|m| {
                m.get("role").and_then(|r| r.as_str()) == Some("tool")
                    && m.get("content")
                        .and_then(|c| c.as_str())
                        .is_some_and(|c| c.starts_with("web_search:"))
            }),
            "late web_search step must appear (summarized) in the re-rendered turn"
        );
    }

    /// 7. No synthetic user / no age_collapse fingerprint on the happy path.
    #[tokio::test]
    async fn no_synthetic_user_and_no_age_collapse_on_happy_path() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        seed_turn(
            &store,
            "s7",
            "t1",
            "current question",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s7", "t1").await;

        let messages = build_payload(&harness, "s7", "current question", 1).await;
        let serialized = serde_json::to_string(&messages).unwrap();
        // No synthetic-user id leaks into content (the in-process fallback did
        // not fire because the current turn is present in the fetch).
        assert!(
            !serialized.contains("synthetic-current-user-"),
            "no synthetic user id on the happy path: {serialized}"
        );
        // No legacy age_collapse stage marker anywhere.
        assert!(!serialized.contains("age_collapse"));
    }

    /// 8. Current-turn fallback fires when the current row is absent: simulate
    ///    the race (current turn not yet in events) and assert the payload still
    ///    ends in the current user turn (full/append-only).
    #[tokio::test]
    async fn current_turn_fallback_injects_when_row_absent() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        // One archived turn exists, but the CURRENT turn's user row is NOT yet
        // committed to events (the documented race). current_turn_ids points at
        // a turn id with no rows.
        seed_turn(
            &store,
            "s8",
            "t1",
            "old question",
            None,
            "old answer",
            "completed",
        )
        .await;
        set_current_turn(&harness, "s8", "t-not-committed").await;

        let messages = build_payload(&harness, "s8", "RACE current text", 1).await;
        // Payload must still end in the current user message (full/append-only).
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));
        assert!(
            last_user.is_some_and(
                |m| m.get("content").and_then(|c| c.as_str()) == Some("RACE current text")
            ),
            "fallback must inject the current user message: {:?}",
            serde_json::to_string(&messages)
        );
    }

    /// 9. Render cache prunes on eviction: after an eviction advances the anchor,
    ///    `turn_renders[session]` no longer holds entries for evicted turn ids.
    #[tokio::test]
    async fn render_cache_prunes_evicted_turns() {
        let mut harness = setup_test_agent(MockProvider::new()).await.unwrap();
        harness.agent.context_window_config.default_budget = 5900;
        let store = seed_store(&harness).await;
        let big = "detail ".repeat(20);
        for i in 1..=4 {
            seed_turn(
                &store,
                "s9",
                &format!("t{i}"),
                &format!("turn {i} {big}"),
                None,
                &format!("answer {i}"),
                "completed",
            )
            .await;
        }
        seed_turn(
            &store,
            "s9",
            "tcur",
            "current request",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s9", "tcur").await;

        let _ = build_payload(&harness, "s9", "current request", 1).await;
        let cache = harness.agent.turn_renders.read().await;
        let session_cache = cache.get("s9");
        // The oldest evicted turn ids must NOT remain cached (no unbounded growth).
        if let Some(sc) = session_cache {
            assert!(
                !sc.contains_key("t1"),
                "evicted turn t1 must be pruned from the render cache"
            );
        }
        // Anchor advanced past t1.
        let anchor = harness.agent.turn_anchors.read().await.get("s9").copied();
        assert!(anchor.is_some());
    }

    /// 10. No duplicate pinned-history path: unique content markers appear at
    ///     most once, and every emitted historical marker belongs to a turn at
    ///     or after the anchor.
    #[tokio::test]
    async fn no_duplicate_pinned_history_path() {
        let harness = setup_test_agent(MockProvider::new()).await.unwrap();
        let store = seed_store(&harness).await;
        // More than the former 20-message recency window, each with a unique marker.
        for i in 1..=12 {
            seed_turn(
                &store,
                "s10",
                &format!("t{i}"),
                &format!("UNIQUEMARK-{i:02}-X"),
                None,
                &format!("reply-{i}"),
                "completed",
            )
            .await;
        }
        seed_turn(
            &store,
            "s10",
            "tcur",
            "current request UNIQUEMARK-CUR",
            None,
            "",
            "in_progress",
        )
        .await;
        set_current_turn(&harness, "s10", "tcur").await;

        let messages = build_payload(&harness, "s10", "current request UNIQUEMARK-CUR", 1).await;

        // Every marker present occurs at most once (no duplicate canonical-history path).
        for i in 1..=12 {
            let needle = format!("UNIQUEMARK-{i:02}-X");
            let occ = count_occurrences(&messages, &needle);
            assert!(
                occ <= 1,
                "marker {needle} must occur at most once, got {occ}"
            );
        }

        // Every emitted historical marker belongs to a turn whose turn_seq >= anchor.
        let anchor = harness
            .agent
            .turn_anchors
            .read()
            .await
            .get("s10")
            .copied()
            .expect("anchor recorded");
        let store2 = seed_store(&harness).await;
        let turns = store2.get_turns_from_anchor("s10", anchor).await.unwrap();
        let serialized = serde_json::to_string(&messages).unwrap();
        for i in 1..=12 {
            let needle = format!("UNIQUEMARK-{i:02}-X");
            if serialized.contains(&needle) {
                let belongs = turns.iter().any(|t| {
                    t.turn_seq >= anchor
                        && t.messages
                            .iter()
                            .any(|m| m.content.as_deref().is_some_and(|c| c.contains(&needle)))
                });
                assert!(
                    belongs,
                    "emitted marker {needle} must belong to a turn >= anchor"
                );
            }
        }

        // Compile-time guarantee that the pinned_memories plumbing is gone:
        // MessageBuildCtx has no such field (this references all current fields).
        let _assert_no_pinned = |c: &MessageBuildCtx| {
            let MessageBuildCtx {
                session_id: _,
                iteration: _,
                user_text: _,
                current_attachments: _,
                completed_tool_calls: _,
                model: _,
                core_prompt: _,
                task_context_tail: _,
                tool_defs: _,
                policy_bundle: _,
                pending_system_messages: _,
                empty_response_retry_pending: _,
                status_tx: _,
            } = c;
        };
    }

    #[test]
    fn fresh_context_isolation_preserves_multimodal_user_message() {
        let user_text = "What do you see here?";
        let mut messages = vec![
            json!({"role": "system", "content": "core"}),
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }),
        ];
        let non_system_non_user_count = messages
            .iter()
            .filter(|m| {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
                role != "system" && role != "user"
            })
            .count();
        assert_eq!(non_system_non_user_count, 0);
        messages.retain(|m| {
            m.get("role").and_then(|r| r.as_str()) != Some("user")
                || m.get("content")
                    .is_some_and(|content| user_message_content_matches(content, user_text))
        });
        assert_eq!(messages.len(), 2);
        assert!(
            messages
                .iter()
                .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("user")),
            "multimodal user message must survive fresh-context retain"
        );
    }
}
