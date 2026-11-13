use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::Sha256;
use uuid::Uuid;

use crate::tools::traits::{ChannelVisibility, UserRole};
use crate::tools::traits::{
    HistoryCoverage, HistoryMessage, HistoryScope, HistorySearchStore, HistoryToolActivity,
    TaskBookends, Tool, ToolCallContext, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics,
    ToolCapabilities, ToolOutcomeStatus, ToolVerificationMode,
};

type HmacSha256 = Hmac<Sha256>;

pub struct SearchHistoryTool {
    state: Arc<dyn HistorySearchStore>,
    /// Summaries that narrow a search, when the daemon's own memory keeps them.
    episodes: Option<Arc<dyn aidaemon_memory::stores::EpisodeStore>>,
    retention_days: u32,
    cursor_secret: Vec<u8>,
}

impl SearchHistoryTool {
    pub fn new(state: Arc<dyn HistorySearchStore>, retention_days: u32) -> Self {
        Self {
            state,
            episodes: None,
            retention_days,
            cursor_secret: Uuid::new_v4().as_bytes().to_vec(),
        }
    }

    /// Let searches be narrowed by the sessions these summaries point at.
    pub fn with_episodes(
        mut self,
        episodes: Arc<dyn aidaemon_memory::stores::EpisodeStore>,
    ) -> Self {
        self.episodes = Some(episodes);
        self
    }

    fn encode_cursor(&self, payload: &CursorPayload) -> anyhow::Result<String> {
        let bytes = serde_json::to_vec(payload)?;
        let mut mac = HmacSha256::new_from_slice(&self.cursor_secret)?;
        mac.update(&bytes);
        let signature = mac.finalize().into_bytes();
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        Ok(format!(
            "{}.{}",
            engine.encode(bytes),
            engine.encode(signature)
        ))
    }

    fn decode_cursor(&self, raw: &str) -> anyhow::Result<CursorPayload> {
        let (payload, signature) = raw
            .split_once('.')
            .ok_or_else(|| anyhow::anyhow!("invalid history cursor"))?;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let bytes = engine.decode(payload)?;
        let signature = engine.decode(signature)?;
        let mut mac = HmacSha256::new_from_slice(&self.cursor_secret)?;
        mac.update(&bytes);
        mac.verify_slice(&signature)
            .map_err(|_| anyhow::anyhow!("history cursor signature is invalid"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// The caller's scope, from the dispatcher's `facts` only.
    fn runtime_scope(
        args: &SearchHistoryArgs,
        facts: &ToolCallContext,
        snapshot: i64,
    ) -> anyhow::Result<HistoryScope> {
        let session_id = facts
            .session()
            .ok_or_else(|| anyhow::anyhow!("missing trusted session scope"))?
            .to_string();
        Ok(HistoryScope {
            session_id,
            channel_id: facts.channel_id.clone(),
            visibility: facts
                .channel_visibility
                .unwrap_or(ChannelVisibility::Internal),
            user_role: facts.user_role.unwrap_or(UserRole::Public),
            trusted: facts.trusted_session,
            include_subagents: args.include_subagents,
            session_filter: args.filter_session_id.clone(),
            task_filter: args.task_id.clone(),
            snapshot_max_event_id: snapshot,
        })
    }

    fn cursor_for(
        &self,
        scope: &HistoryScope,
        anchor: i64,
        direction: &str,
    ) -> anyhow::Result<String> {
        self.encode_cursor(&CursorPayload {
            version: 1,
            anchor_event_id: anchor,
            direction: direction.to_string(),
            snapshot_max_event_id: scope.snapshot_max_event_id,
            runtime_session_id: scope.session_id.clone(),
            channel_id: scope.channel_id.clone(),
            visibility: scope.visibility.to_string(),
            trusted: scope.trusted,
            include_subagents: scope.include_subagents,
            session_filter: scope.session_filter.clone(),
            task_filter: scope.task_filter.clone(),
        })
    }

    fn scope_from_cursor(
        args: &SearchHistoryArgs,
        facts: &ToolCallContext,
        cursor: &CursorPayload,
    ) -> anyhow::Result<HistoryScope> {
        let runtime = Self::runtime_scope(args, facts, cursor.snapshot_max_event_id)?;
        if cursor.version != 1
            || cursor.runtime_session_id != runtime.session_id
            || cursor.channel_id != runtime.channel_id
            || cursor.visibility != runtime.visibility.to_string()
            || cursor.trusted != runtime.trusted
        {
            anyhow::bail!("history cursor does not belong to the current authorized scope");
        }
        Ok(HistoryScope {
            include_subagents: cursor.include_subagents,
            session_filter: cursor.session_filter.clone(),
            task_filter: cursor.task_filter.clone(),
            ..runtime
        })
    }

    async fn semantic_sessions(&self, query: &str, scope: &HistoryScope) -> HashSet<String> {
        if query.trim().is_empty() {
            return HashSet::new();
        }
        let Some(store) = &self.episodes else {
            return HashSet::new();
        };
        let episodes = match scope.visibility {
            ChannelVisibility::Private => store.get_relevant_episodes(query, 16).await,
            ChannelVisibility::PrivateGroup | ChannelVisibility::Public => {
                store
                    .get_relevant_episodes_for_channel(query, 16, scope.channel_id.as_deref())
                    .await
            }
            ChannelVisibility::Internal if scope.trusted => {
                store.get_relevant_episodes(query, 16).await
            }
            ChannelVisibility::Internal => {
                store
                    .get_relevant_episodes_for_session(query, 16, &scope.session_id)
                    .await
            }
            ChannelVisibility::PublicExternal => return HashSet::new(),
        };
        episodes
            .unwrap_or_default()
            .into_iter()
            .map(|episode| episode.session_id)
            .collect()
    }

    fn may_report_global_coverage(scope: &HistoryScope) -> bool {
        scope.visibility == ChannelVisibility::Private
            || (scope.visibility == ChannelVisibility::Internal && scope.trusted)
    }

    async fn run_search(
        &self,
        arguments: &str,
        facts: &ToolCallContext,
    ) -> anyhow::Result<ToolCallOutcome> {
        let args = serde_json::from_str::<SearchHistoryArgs>(arguments)?.without_blank_fields();
        let action = args.action.as_str();
        let authorization_scope = Self::runtime_scope(
            &args,
            facts,
            self.state.history_snapshot_max_event_id().await?,
        )?;
        if authorization_scope.user_role != UserRole::Owner {
            anyhow::bail!("exact history is restricted to the owner");
        }
        if authorization_scope.visibility == ChannelVisibility::PublicExternal {
            anyhow::bail!("exact history is disabled in public-external channels");
        }

        if action == "health" {
            let coverage = if Self::may_report_global_coverage(&authorization_scope) {
                Some(self.state.history_coverage().await?)
            } else {
                None
            };
            return self.outcome(
                serde_json::to_string_pretty(&json!({
                    "projection": coverage,
                    "projection_note": if coverage.is_some() {
                        "Global projection coverage is visible in this authorized scope."
                    } else {
                        "Global projection counts are hidden in channel/session-scoped contexts."
                    },
                    "retention": retention_disclosure(self.retention_days)
                }))?,
                "search_history health checked".to_string(),
                false,
            );
        }
        if action == "repair" {
            if !Self::may_report_global_coverage(&authorization_scope) {
                anyhow::bail!(
                    "global exact-history repair is restricted to private or trusted internal contexts"
                );
            }
            let stats = self.state.repair_history_projection(100).await?;
            return self.outcome(
                serde_json::to_string_pretty(&stats)?,
                "search_history projection repair completed".to_string(),
                false,
            );
        }

        let (scope, anchor, older) = if action == "page" {
            let cursor = self.decode_cursor(
                args.cursor
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("cursor is required for page"))?,
            )?;
            let scope = Self::scope_from_cursor(&args, facts, &cursor)?;
            (
                scope,
                Some(cursor.anchor_event_id),
                cursor.direction == "older",
            )
        } else {
            let snapshot = self.state.history_snapshot_max_event_id().await?;
            (
                Self::runtime_scope(&args, facts, snapshot)?,
                args.event_id,
                false,
            )
        };

        let limit = args.limit.unwrap_or(8).clamp(1, 20);
        let radius = args.context.unwrap_or(2).clamp(0, 4);
        let mut response = SearchResponse {
            action: action.to_string(),
            query: args.query.clone(),
            snapshot_max_event_id: scope.snapshot_max_event_id,
            retention: retention_disclosure(self.retention_days),
            coverage: if Self::may_report_global_coverage(&scope) {
                Some(self.state.history_coverage().await?)
            } else {
                None
            },
            matches: Vec::new(),
            messages: Vec::new(),
            tool_activity: Vec::new(),
            older_cursor: None,
            newer_cursor: None,
        };

        match action {
            "search" => {
                let query = args
                    .query
                    .as_deref()
                    .filter(|q| !q.trim().is_empty())
                    .ok_or_else(|| anyhow::anyhow!("query is required for search"))?;
                let semantic = self.semantic_sessions(query, &scope).await;
                let hits = self
                    .state
                    .search_history(query, &scope, limit, &semantic)
                    .await?;
                for hit in hits {
                    // Discovery spans authorized sessions, but scrolling from a
                    // hit follows that hit's exact session timeline.
                    let mut hit_scope = scope.clone();
                    hit_scope.session_filter = Some(hit.session_id.clone());
                    let older_cursor = self.cursor_for(&hit_scope, hit.event_id, "older")?;
                    let newer_cursor = self.cursor_for(&hit_scope, hit.event_id, "newer")?;
                    let context = self
                        .state
                        .history_context(hit.event_id, radius, &scope)
                        .await?;
                    let bookends = self
                        .state
                        .history_task_bookends(hit.task_id.as_deref(), &hit.session_id, &scope)
                        .await?;
                    response.matches.push(SearchHit {
                        anchor: hit,
                        context,
                        bookends,
                        older_cursor,
                        newer_cursor,
                    });
                }
            }
            "open" => {
                let selector_count = usize::from(anchor.is_some())
                    + usize::from(args.message_id.is_some())
                    + usize::from(args.turn_id.is_some());
                anyhow::ensure!(
                    selector_count == 1,
                    "open requires exactly one of event_id, message_id, or turn_id"
                );
                if let Some(turn_id) = args.turn_id.as_deref() {
                    response.messages = self.state.history_turn(turn_id, &scope).await?;
                    response.tool_activity = prepare_tool_activity(
                        self.state
                            .history_turn_tool_activity(turn_id, &scope)
                            .await?,
                        scope.visibility,
                    );
                } else {
                    let event_id = match (anchor, args.message_id.as_deref()) {
                        (Some(event_id), None) => event_id,
                        (None, Some(message_id)) => self
                            .state
                            .history_event_for_message_id(message_id, &scope)
                            .await?
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "message_id was not found in the authorized retained history"
                                )
                            })?,
                        _ => unreachable!("selector count was validated above"),
                    };
                    response.messages =
                        self.state.history_context(event_id, radius, &scope).await?;
                }
            }
            "page" => {
                response.messages = self
                    .state
                    .history_page(anchor.unwrap_or_default(), older, &scope, limit)
                    .await?;
            }
            other => anyhow::bail!("unsupported search_history action: {other}"),
        }

        let bounds: Vec<&HistoryMessage> = response.messages.iter().collect();
        if let (Some(first), Some(last)) = (bounds.first(), bounds.last()) {
            let mut message_scope = scope.clone();
            if response
                .messages
                .iter()
                .all(|message| message.session_id == first.session_id)
            {
                message_scope.session_filter = Some(first.session_id.clone());
            }
            response.older_cursor =
                Some(self.cursor_for(&message_scope, first.event_id, "older")?);
            response.newer_cursor =
                Some(self.cursor_for(&message_scope, last.event_id, "newer")?);
        }

        let count = response.matches.len() + response.messages.len();
        self.outcome(
            serde_json::to_string_pretty(&response)?,
            format!(
                "search_history returned {count} authorized exact-history item(s) at snapshot {}",
                scope.snapshot_max_event_id
            ),
            true,
        )
    }

    fn outcome(
        &self,
        output: String,
        persistent_output: String,
        verbatim: bool,
    ) -> anyhow::Result<ToolCallOutcome> {
        Ok(ToolCallOutcome {
            output,
            metadata: ToolCallMetadata {
                outcome_status: Some(ToolOutcomeStatus::Succeeded),
                untrusted_verbatim: verbatim,
                preserve_inline: verbatim,
                persistent_output: Some(persistent_output),
                suppress_activity_result: true,
                ..ToolCallMetadata::default()
            },
            finish: None,
        })
    }
}

fn retention_disclosure(days: u32) -> String {
    if days == 0 {
        "Exact recall is configured for permanent active-database history; explicit wipe and backup lifecycles still apply.".to_string()
    } else {
        format!(
            "Exact recall is limited to canonical messages retained for {days} days; projection coverage is reported separately."
        )
    }
}

#[derive(Debug, Deserialize)]
struct SearchHistoryArgs {
    action: String,
    query: Option<String>,
    event_id: Option<i64>,
    message_id: Option<String>,
    turn_id: Option<String>,
    cursor: Option<String>,
    limit: Option<usize>,
    context: Option<usize>,
    #[serde(default)]
    include_subagents: bool,
    #[serde(rename = "session_id")]
    filter_session_id: Option<String>,
    task_id: Option<String>,
}

impl SearchHistoryArgs {
    /// Strict function schemas make models send every optional field, so an
    /// unused one arrives as `""` or `0`. Those mean absent.
    fn without_blank_fields(mut self) -> Self {
        for field in [
            &mut self.query,
            &mut self.message_id,
            &mut self.turn_id,
            &mut self.cursor,
            &mut self.filter_session_id,
            &mut self.task_id,
        ] {
            *field = field.take().filter(|value| !value.trim().is_empty());
        }
        self.event_id = self.event_id.filter(|id| *id > 0);
        self.limit = self.limit.filter(|limit| *limit > 0);
        self
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CursorPayload {
    version: u8,
    anchor_event_id: i64,
    direction: String,
    snapshot_max_event_id: i64,
    runtime_session_id: String,
    channel_id: Option<String>,
    visibility: String,
    trusted: bool,
    include_subagents: bool,
    session_filter: Option<String>,
    task_filter: Option<String>,
}

#[derive(Serialize)]
struct SearchHit {
    anchor: HistoryMessage,
    context: Vec<HistoryMessage>,
    bookends: TaskBookends,
    older_cursor: String,
    newer_cursor: String,
}

#[derive(Serialize)]
struct SearchResponse {
    action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<String>,
    snapshot_max_event_id: i64,
    retention: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<HistoryCoverage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    matches: Vec<SearchHit>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    messages: Vec<HistoryMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_activity: Vec<HistoryToolActivity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    older_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    newer_cursor: Option<String>,
}

/// How far back the per-turn note looks, matching the dialogue a new request
/// keeps.
const RECENT_TOOL_TURNS: usize = 6;
const TOOL_OUTPUT_CHARS: usize = 6_000;
const TOOL_ARGUMENT_CHARS: usize = 1_000;

/// Bound each call's output and argument strings, and redact outside a private
/// chat, where a transcript of a shared session is redacted the same way.
fn prepare_tool_activity(
    activity: Vec<HistoryToolActivity>,
    visibility: ChannelVisibility,
) -> Vec<HistoryToolActivity> {
    let private = visibility == ChannelVisibility::Private;
    activity
        .into_iter()
        .map(|mut call| {
            if !private {
                call.output = aidaemon_tools::sanitize::redact_secrets(&call.output);
            }
            let total = call.output.chars().count();
            if total > TOOL_OUTPUT_CHARS {
                call.output = call.output.chars().take(TOOL_OUTPUT_CHARS).collect();
                call.output_omitted_chars = Some(total - TOOL_OUTPUT_CHARS);
            }
            call.arguments = call.arguments.map(|arguments| {
                let text = arguments.to_string();
                let text = if private {
                    text
                } else {
                    aidaemon_tools::sanitize::redact_secrets(&text)
                };
                if text.chars().count() > TOOL_ARGUMENT_CHARS {
                    Value::from(aidaemon_events::text::truncate_str(
                        &text,
                        TOOL_ARGUMENT_CHARS,
                    ))
                } else {
                    serde_json::from_str(&text).unwrap_or(Value::from(text))
                }
            });
            call
        })
        .collect()
}

#[async_trait]
impl Tool for SearchHistoryTool {
    fn name(&self) -> &str {
        "search_history"
    }

    fn description(&self) -> &str {
        "Search exact retained conversation messages. Search returns anchored context and task objective/resolution bookends. Open an event or page older/newer exact messages with signed cursors. Opening a turn_id also returns the tool calls that turn ran and their output, which search does not cover."
    }

    fn schema(&self) -> Value {
        json!({
            "name": "search_history",
            "description": self.description(),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "open", "page", "health", "repair"]
                    },
                    "query": { "type": "string", "description": "Words to find; FTS syntax is treated as literal terms" },
                    "event_id": { "type": "integer", "description": "Authorized anchor event for open" },
                    "message_id": { "type": "string", "maxLength": 256, "description": "Stable canonical message ID for open" },
                    "turn_id": { "type": "string", "maxLength": 256, "description": "Stable canonical turn ID; open returns the whole authorized turn, including its tool calls and their output" },
                    "cursor": { "type": "string", "description": "Opaque signed cursor returned by a prior call" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 20 },
                    "context": { "type": "integer", "minimum": 0, "maximum": 4 },
                    "include_subagents": { "type": "boolean", "description": "Include specialist child sessions; false by default" },
                    "session_id": { "type": "string", "description": "Optional session narrowing filter" },
                    "task_id": { "type": "string", "description": "Optional task narrowing filter" }
                },
                "required": ["action"],
                "additionalProperties": false
            }
        })
    }

    async fn execute(
        &self,
        arguments: &str,
        facts: &crate::tools::traits::ToolCallContext,
    ) -> anyhow::Result<crate::tools::traits::ToolCallOutcome> {
        let facts = facts.clone();

        self.run_search(arguments, &facts).await
    }
}

#[async_trait]
impl crate::tools::traits::ToolSemantics for SearchHistoryTool {
    /// Transcripts of new requests drop earlier tool calls, so a turn whose
    /// reply was brief can read as if nothing ran. This states which turns did
    /// from the event log, and carries no message or tool content.
    async fn session_context_note(&self, session_id: &str) -> Option<String> {
        let turns = self
            .state
            .history_recent_tool_turns(session_id, RECENT_TOOL_TURNS)
            .await
            .ok()?;
        if turns.is_empty() {
            return None;
        }
        let counted = |calls: &[(String, usize)]| {
            calls
                .iter()
                .map(|(tool, count)| format!("{tool} x{count}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let lines = turns
            .iter()
            .map(|turn| {
                let ago = match turn.turns_ago {
                    1 => "previous turn".to_string(),
                    n => format!("{n} turns ago"),
                };
                let mut calls = Vec::new();
                if !turn.tools.is_empty() {
                    calls.push(counted(&turn.tools));
                }
                if !turn.refused.is_empty() {
                    calls.push(format!(
                        "refused before running: {}",
                        counted(&turn.refused)
                    ));
                }
                format!("- {ago} (turn_id={}): {}", turn.turn_id, calls.join("; "))
            })
            .collect::<Vec<_>>()
            .join("\n");
        Some(format!(
            "Tool calls from earlier turns of this conversation:\n{lines}\nThe event log records \
             those calls. The ones not marked refused did run even when the reply was brief. A \
             refused one was attempted and stopped before it ran, so a reply saying it was \
             blocked was accurate. If their output is not in your transcript and you need it, \
             call search_history with action=open and that turn_id to read the exact calls and \
             output."
        ))
    }

    fn dispatch_facts(&self) -> crate::tools::traits::ToolDispatchFacts {
        crate::tools::traits::ToolDispatchFacts {
            always_offered: true,
            ..Default::default()
        }
    }

    /// Retained messages include other people's (group chats, guests) and
    /// assistant turns that echoed pages: someone else's words.
    fn output_trust(
        &self,
        _arguments: &str,
        _roots: &crate::tools::traits::TrustRoots<'_>,
    ) -> aidaemon_core::SourceTrust {
        aidaemon_core::SourceTrust::Untrusted
    }

    fn capabilities(&self) -> ToolCapabilities {
        ToolCapabilities {
            // `repair` mutates only a disposable local projection; the normal
            // search/open/page/health actions are classified as observations
            // by `call_semantics` below.
            read_only: false,
            external_side_effect: false,
            needs_approval: false,
            idempotent: true,
            high_impact_write: false,
        }
    }

    fn call_semantics(&self, arguments: &str) -> ToolCallSemantics {
        let action = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        if action.as_deref() == Some("repair") {
            ToolCallSemantics::mutation()
        } else {
            ToolCallSemantics::observation()
                .with_verification_mode(ToolVerificationMode::ResultContent)
        }
    }
}
