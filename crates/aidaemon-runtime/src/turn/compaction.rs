//! The daemon on the SDK's conversation compaction: its event store as the
//! [`sdk_policies::ConversationSource`], its summary table as the
//! [`sdk_policies::SummaryStore`], and the side-call gateway behind the
//! SDK's [`sdk_policies::ProviderSummarizer`]. Every decision (when to compact,
//! which turns to fold in, one pass per session, coalescing) is the SDK's
//! [`sdk_policies::Compactor`].

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use aidaemon_core as sdk;
use aidaemon_policies as sdk_policies;
use async_trait::async_trait;
use serde_json::Value;

use crate::llm::gateway::SideCallGateway;
use crate::llm::side_calls::SideCall;
use aidaemon_events::{EventStore, FetchedTurn};
use aidaemon_tools::semantics::UserRole;

/// What a compaction reads and writes besides the turns: the summary table,
/// and the settings that record a summary's untrusted-content taint.
pub trait CompactionStore:
    aidaemon_events::stores::ConversationSummaryStore + aidaemon_tools::services::SettingsStore
{
}

impl<T> CompactionStore for T where
    T: aidaemon_events::stores::ConversationSummaryStore
        + aidaemon_tools::services::SettingsStore
        + ?Sized
{
}

/// One pass per session and one queued behind it, across every compactor
/// the daemon builds (the background pass after a reply and the emergency
/// pass before a turn share them).
static JOBS: LazyLock<sdk_policies::CompactionJobs> =
    LazyLock::new(sdk_policies::CompactionJobs::default);
/// Summary-model calls at once, independent of fact extraction.
static SUMMARY_PERMITS: LazyLock<Arc<tokio::sync::Semaphore>> =
    LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(2)));

/// The daemon's `conversation_summaries` table. A legacy summary without a
/// turn cursor reads as none, so the next pass rebuilds it from the turns.
pub struct DaemonSummaries {
    /// Where the summaries and their taint records live.
    pub state: Arc<dyn CompactionStore>,
    /// The tainted turns among those this pass folds in.
    pub tainted_seqs: TaintedSeqs,
}

/// The `turn_seq`s, among the turns handed to this pass, of turns that read
/// untrusted content.
pub type TaintedSeqs = Arc<std::sync::Mutex<HashSet<i64>>>;

/// Later passes with no tainted turn before a summary counts as clean again:
/// each rewrites the summary, so what the page said has been diluted away.
pub const CLEAN_PASSES_BEFORE_TRUSTED: u32 = 3;

/// What stands in for a tool result of a tainted turn: untrusted text never
/// goes into a summary.
const OMITTED_TOOL_OUTPUT: &str = "[tool output omitted: it held untrusted content]";

fn summary_taint_key(session_id: &str) -> String {
    format!("untrusted_summary:{session_id}")
}

/// Whether the session's summary folded in a turn that read untrusted
/// content. An unreadable record counts as tainted.
pub async fn summary_is_tainted(state: &dyn CompactionStore, session_id: &str) -> bool {
    match state.get_setting(&summary_taint_key(session_id)).await {
        Ok(Some(passes)) => !passes.is_empty(),
        Ok(None) => false,
        Err(_) => true,
    }
}

#[async_trait]
impl sdk_policies::SummaryStore for DaemonSummaries {
    async fn load(&self, session_id: &str) -> Option<sdk_policies::ConversationSummary> {
        let stored = self
            .state
            .get_conversation_summary(session_id)
            .await
            .ok()
            .flatten()?;
        Some(sdk_policies::ConversationSummary {
            text: stored.summary,
            cursor: stored.last_turn_seq?,
            message_count: stored.message_count,
            last_message_id: Some(stored.last_message_id).filter(|id| !id.is_empty()),
            fingerprint: 0,
        })
    }

    async fn save(&self, session_id: &str, summary: sdk_policies::ConversationSummary) {
        let stored = aidaemon_events::conversation::ConversationSummary {
            session_id: session_id.to_string(),
            summary: summary.text,
            message_count: summary.message_count,
            last_message_id: summary.last_message_id.unwrap_or_default(),
            last_turn_seq: Some(summary.cursor),
            updated_at: chrono::Utc::now(),
        };
        let cursor = stored.last_turn_seq.unwrap_or(i64::MAX);
        if let Err(error) = self.state.upsert_conversation_summary(&stored).await {
            tracing::warn!(session_id, %error, "could not store the conversation summary");
            return;
        }
        let folded_taint = self
            .tainted_seqs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .any(|seq| *seq <= cursor);
        let key = summary_taint_key(session_id);
        let next = if folded_taint {
            Some("0".to_string())
        } else {
            match self.state.get_setting(&key).await.ok().flatten() {
                Some(passes) if !passes.is_empty() => {
                    let passes = passes.parse::<u32>().unwrap_or(0) + 1;
                    Some(if passes >= CLEAN_PASSES_BEFORE_TRUSTED {
                        String::new()
                    } else {
                        passes.to_string()
                    })
                }
                _ => None,
            }
        };
        if let Some(next) = next {
            if let Err(error) = self.state.set_setting(&key, &next).await {
                tracing::warn!(session_id, %error, "could not record the summary's taint");
            }
        }
    }
}

/// The session's canonical turns from the event store, after the summary's
/// cursor (a turn's `seq` is its `turn_seq`). Turns the memory policy
/// suppressed are left out. `finished_turn_id` is the turn whose run just
/// ended: its `task_end` lands after the run, so without it the turn would
/// still look open.
pub struct DaemonTurns {
    /// The session's event log.
    pub event_store: Arc<EventStore>,
    /// Where the turns' taint records live.
    pub state: Arc<dyn CompactionStore>,
    /// Filled with the tainted turns among those handed to the pass.
    pub tainted_seqs: TaintedSeqs,
    /// The turn whose run just ended.
    pub finished_turn_id: Option<String>,
}

#[async_trait]
impl sdk_policies::ConversationSource for DaemonTurns {
    async fn turns_after(
        &self,
        session_id: &str,
        summary: Option<&sdk_policies::ConversationSummary>,
    ) -> Result<sdk_policies::PendingTurns, String> {
        let anchor = summary.map_or(0, |s| s.cursor.saturating_add(1));
        let suppressed: HashSet<String> = self
            .event_store
            .memory_suppressed_turn_ids(session_id)
            .await
            .map_err(|e| e.to_string())?
            .into_iter()
            .collect();
        let turns: Vec<FetchedTurn> = self
            .event_store
            .get_turns_from_anchor(session_id, anchor)
            .await
            .map_err(|e| e.to_string())?
            .into_iter()
            .filter(|turn| {
                !turn
                    .turn_id
                    .as_ref()
                    .is_some_and(|id| suppressed.contains(id))
            })
            .collect();
        // An unreadable record counts every turn as tainted.
        let tainted_turns = crate::untrusted::tainted_turns(self.state.as_ref(), session_id)
            .await
            .ok();
        let is_tainted = |turn: &FetchedTurn| match &tainted_turns {
            None => true,
            Some(tainted) => turn.turn_id.as_ref().is_some_and(|id| tainted.contains(id)),
        };
        {
            let mut seqs = self
                .tainted_seqs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            seqs.clear();
            seqs.extend(turns.iter().filter(|t| is_tainted(t)).map(|t| t.turn_seq));
        }
        let newest_finished = turns.last().is_some_and(|turn| {
            turn.terminal_status.is_some()
                || self
                    .finished_turn_id
                    .as_deref()
                    .is_some_and(|id| turn.turn_id.as_deref() == Some(id))
        });
        Ok(sdk_policies::PendingTurns {
            turns: turns
                .iter()
                .map(|turn| to_sdk_turn(turn, is_tainted(turn)))
                .collect(),
            newest_finished,
            summary_stale: false,
        })
    }
}

/// A canonical turn as an SDK turn, sized with the daemon's estimator (the
/// one its thresholds are calibrated against).
pub fn to_sdk_turn(turn: &FetchedTurn, tainted: bool) -> sdk_policies::ConversationTurn {
    let messages: Vec<sdk::Message> = turn
        .messages
        .iter()
        .map(|message| {
            if tainted && message.role == "tool" {
                let mut omitted = message.clone();
                omitted.content = Some(OMITTED_TOOL_OUTPUT.to_string());
                to_sdk_message(&omitted)
            } else {
                to_sdk_message(message)
            }
        })
        .collect();
    let tokens = crate::context_budget::estimate_multimodal_message_tokens(
        &turn
            .messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content.as_deref().unwrap_or(""),
                    "tool_calls": m.tool_calls_json.as_deref().unwrap_or(""),
                })
            })
            .collect::<Vec<_>>(),
    );
    sdk_policies::ConversationTurn::new(turn.turn_seq, messages)
        .with_message_ids(turn.messages.iter().map(|m| m.id.clone()).collect())
        .with_tokens(tokens)
}

/// A daemon message as an SDK message.
pub fn to_sdk_message(message: &aidaemon_events::conversation::Message) -> sdk::Message {
    let content = message.content.clone().unwrap_or_default();
    match message.role.as_str() {
        "assistant" => {
            let calls = message
                .tool_calls_json
                .as_deref()
                .and_then(|json| {
                    serde_json::from_str::<Vec<aidaemon_events::conversation::ToolCall>>(json).ok()
                })
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    let arguments = serde_json::from_str(&call.arguments)
                        .unwrap_or(Value::String(call.arguments));
                    sdk::ToolCall::new(call.id, call.name, arguments)
                })
                .collect();
            sdk::Message::assistant(content, calls)
        }
        "tool" => {
            // Without a call id the tool's name stands in, so the summary
            // can still say which tool produced the result.
            let call = message
                .tool_call_id
                .clone()
                .or_else(|| message.tool_name.clone())
                .unwrap_or_default();
            sdk::Message::tool_result(call, content)
        }
        "system" => sdk::Message::system(content),
        _ => sdk::Message::user(content),
    }
}

/// The SDK compactor writing summaries through the side-call gateway, which
/// runs the summarization side call's model chain within its total deadline
/// and records every attempt under `background:summarization`.
fn compactor(
    side_calls: SideCallGateway,
    fast_model: &str,
    state: Arc<dyn CompactionStore>,
    tainted_seqs: TaintedSeqs,
) -> sdk_policies::Compactor {
    let provider = side_calls.provider(SideCall::Summarization);
    // The side-call gateway picks the model; `fast_model` is not sent (the
    // summarizer only logs it). `limits` still uses it for the token-estimate
    // correction.
    let summarizer = sdk_policies::ProviderSummarizer::new(provider, fast_model)
        .permits(SUMMARY_PERMITS.clone());
    sdk_policies::Compactor::new(
        summarizer,
        DaemonSummaries {
            state,
            tainted_seqs,
        },
    )
    .with_jobs(JOBS.clone())
    .pin_user_messages()
}

/// Thresholds in the daemon's calibrated tokens, as the SDK's raw estimate
/// counts them.
fn limits(fast_model: &str, threshold: usize, recent: usize) -> sdk_policies::CompactionLimits {
    let correction = crate::context_budget::token_estimate_correction(fast_model);
    sdk_policies::CompactionLimits {
        threshold: ((threshold as f64) / correction).ceil() as usize,
        keep_tokens: ((recent as f64) / correction).ceil() as usize,
    }
}

/// Where a compaction runs: the session and the daemon's handles.
pub struct CompactionTarget<'a> {
    pub side_calls: SideCallGateway,
    pub fast_model: &'a str,
    pub state: Arc<dyn CompactionStore>,
    pub event_store: Arc<EventStore>,
    pub session_id: &'a str,
}

/// One pass now, waiting for any pass already running for the session.
pub async fn compact_now(
    target: CompactionTarget<'_>,
    threshold: usize,
    recent: usize,
) -> Result<Option<sdk_policies::ConversationSummary>, String> {
    let tainted_seqs = TaintedSeqs::default();
    let source = DaemonTurns {
        event_store: target.event_store.clone(),
        state: target.state.clone(),
        tainted_seqs: tainted_seqs.clone(),
        finished_turn_id: None,
    };
    compactor(
        target.side_calls,
        target.fast_model,
        target.state,
        tainted_seqs,
    )
    .compact(
        target.session_id,
        &source,
        limits(target.fast_model, threshold, recent),
    )
    .await
}

/// A pass after the reply to `finished_turn_id`, in the background.
pub fn compact_after_reply(
    target: CompactionTarget<'_>,
    finished_turn_id: String,
    threshold: usize,
    recent: usize,
    user_role: UserRole,
) {
    if !user_role.can_persist_owner_memory() {
        return;
    }
    let tainted_seqs = TaintedSeqs::default();
    let source = Arc::new(DaemonTurns {
        event_store: target.event_store.clone(),
        state: target.state.clone(),
        tainted_seqs: tainted_seqs.clone(),
        finished_turn_id: Some(finished_turn_id),
    });
    compactor(
        target.side_calls,
        target.fast_model,
        target.state,
        tainted_seqs,
    )
    .spawn(
        target.session_id,
        source,
        limits(target.fast_model, threshold, recent),
    );
}

/// Wait for the session's background passes (tests).
pub async fn wait_for_background(session_id: &str) {
    JOBS.wait(session_id).await;
}

/// The note that places the summary in a continuation turn's requests,
/// pinned to the summary the turn's history was built against.
pub fn summary_note(
    summary: Option<&aidaemon_events::conversation::ConversationSummary>,
) -> Option<sdk_policies::SummaryNote> {
    let summary = summary.filter(|s| !s.summary.is_empty())?;
    Some(
        sdk_policies::SummaryNote::fixed(Some(sdk_policies::ConversationSummary {
            text: summary.summary.clone(),
            cursor: summary.last_turn_seq.unwrap_or_default(),
            message_count: summary.message_count,
            last_message_id: Some(summary.last_message_id.clone()),
            fingerprint: 0,
        }))
        .guidance(
            "If exact wording or evidence is absent below, retrieve the canonical conversation \
             with `search_history`; do not guess.",
        ),
    )
}
