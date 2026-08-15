use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use uuid::Uuid;

use crate::state::sqlite::history_search::{
    HistoryCoverage, HistoryMessage, HistoryScope, HistorySearchStore, TaskBookends,
};
use crate::traits::{
    Tool, ToolCallMetadata, ToolCallOutcome, ToolCallSemantics, ToolCapabilities,
    ToolOutcomeStatus, ToolVerificationMode,
};
use crate::types::{ChannelVisibility, StatusUpdate, UserRole};

type HmacSha256 = Hmac<Sha256>;

pub struct SearchHistoryTool {
    state: Arc<dyn HistorySearchStore>,
    retention_days: u32,
    cursor_secret: Vec<u8>,
}

impl SearchHistoryTool {
    pub fn new(state: Arc<dyn HistorySearchStore>, retention_days: u32) -> Self {
        Self {
            state,
            retention_days,
            cursor_secret: Uuid::new_v4().as_bytes().to_vec(),
        }
    }

    fn parse_role(raw: &str) -> UserRole {
        match raw.to_ascii_lowercase().as_str() {
            "owner" => UserRole::Owner,
            "guest" => UserRole::Guest,
            _ => UserRole::Public,
        }
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

    fn runtime_scope(args: &SearchHistoryArgs, snapshot: i64) -> anyhow::Result<HistoryScope> {
        let session_id = args
            .session_id
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("missing trusted session scope"))?
            .to_string();
        Ok(HistoryScope {
            session_id,
            channel_id: args.channel_id.clone(),
            visibility: ChannelVisibility::from_str_lossy(
                args.channel_visibility.as_deref().unwrap_or("internal"),
            ),
            user_role: Self::parse_role(args.user_role.as_deref().unwrap_or("public")),
            trusted: args.trusted_session,
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
        cursor: &CursorPayload,
    ) -> anyhow::Result<HistoryScope> {
        let runtime = Self::runtime_scope(args, cursor.snapshot_max_event_id)?;
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
        let episodes = match scope.visibility {
            ChannelVisibility::Private => self.state.get_relevant_episodes(query, 16).await,
            ChannelVisibility::PrivateGroup | ChannelVisibility::Public => {
                self.state
                    .get_relevant_episodes_for_channel(query, 16, scope.channel_id.as_deref())
                    .await
            }
            ChannelVisibility::Internal if scope.trusted => {
                self.state.get_relevant_episodes(query, 16).await
            }
            ChannelVisibility::Internal => {
                self.state
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

    async fn execute(&self, arguments: &str) -> anyhow::Result<ToolCallOutcome> {
        let args: SearchHistoryArgs = serde_json::from_str(arguments)?;
        let action = args.action.as_str();
        let authorization_scope =
            Self::runtime_scope(&args, self.state.history_snapshot_max_event_id().await?)?;
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
            let scope = Self::scope_from_cursor(&args, &cursor)?;
            (
                scope,
                Some(cursor.anchor_event_id),
                cursor.direction == "older",
            )
        } else {
            let snapshot = self.state.history_snapshot_max_event_id().await?;
            (Self::runtime_scope(&args, snapshot)?, args.event_id, false)
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
                let event_id =
                    anchor.ok_or_else(|| anyhow::anyhow!("event_id is required for open"))?;
                response.messages = self.state.history_context(event_id, radius, &scope).await?;
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
    cursor: Option<String>,
    limit: Option<usize>,
    context: Option<usize>,
    #[serde(default)]
    include_subagents: bool,
    #[serde(rename = "session_id")]
    filter_session_id: Option<String>,
    task_id: Option<String>,
    #[serde(rename = "_session_id")]
    session_id: Option<String>,
    #[serde(rename = "_channel_id")]
    channel_id: Option<String>,
    #[serde(rename = "_channel_visibility")]
    channel_visibility: Option<String>,
    #[serde(rename = "_user_role")]
    user_role: Option<String>,
    #[serde(default, rename = "_trusted_session")]
    trusted_session: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    older_cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    newer_cursor: Option<String>,
}

#[async_trait]
impl Tool for SearchHistoryTool {
    fn name(&self) -> &str {
        "search_history"
    }

    fn description(&self) -> &str {
        "Search and page exact canonical user/assistant messages across authorized sessions; use manage_memories search_episodes for coarse semantic recall"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "search_history",
            "description": "Search exact retained conversation messages. Search returns anchored context and task objective/resolution bookends. Open an event or page older/newer exact messages with signed cursors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "open", "page", "health", "repair"]
                    },
                    "query": { "type": "string", "description": "Words to find; FTS syntax is treated as literal terms" },
                    "event_id": { "type": "integer", "description": "Authorized anchor event for open" },
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

    async fn call(&self, arguments: &str) -> anyhow::Result<String> {
        Ok(self.execute(arguments).await?.output)
    }

    async fn call_with_status_outcome(
        &self,
        arguments: &str,
        _status_tx: Option<tokio::sync::mpsc::Sender<StatusUpdate>>,
    ) -> anyhow::Result<ToolCallOutcome> {
        self.execute(arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{Event, EventStore, EventType};
    use crate::memory::embeddings::EmbeddingService;
    use crate::state::SqliteStateStore;

    async fn tool() -> (SearchHistoryTool, Arc<EventStore>, tempfile::NamedTempFile) {
        let db = tempfile::NamedTempFile::new().unwrap();
        let path = db.path().to_path_buf();
        let embedding = Arc::new(EmbeddingService::new().unwrap());
        let state = Arc::new(
            SqliteStateStore::new(path.to_str().unwrap(), 50, None, embedding)
                .await
                .unwrap(),
        );
        let events = Arc::new(EventStore::new(state.pool()).await.unwrap());
        (SearchHistoryTool::new(state, 90), events, db)
    }

    #[tokio::test]
    async fn exact_output_is_ephemeral_and_cursor_cannot_change_runtime_scope() {
        let (tool, events, _db) = tool().await;
        events
            .append(Event::new(
                "root-session",
                EventType::UserMessage,
                json!({
                    "content": "needle verbatim secret",
                    "message_id": "m1",
                    "channel_visibility": "private",
                    "user_role": "owner",
                    "turn_id": "t1"
                }),
            ))
            .await
            .unwrap();
        events
            .append(Event::new(
                "other-session",
                EventType::AssistantResponse,
                json!({
                    "content": "interleaved other-session message",
                    "message_id": "m-other"
                }),
            ))
            .await
            .unwrap();
        events
            .append(Event::new(
                "root-session",
                EventType::AssistantResponse,
                json!({
                    "content": "same-session followup",
                    "message_id": "m2"
                }),
            ))
            .await
            .unwrap();

        let outcome = tool
            .call_with_status_outcome(
                &json!({
                    "action": "search",
                    "query": "needle",
                    "_session_id": "root-session",
                    "_channel_visibility": "private",
                    "_user_role": "Owner"
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();
        assert!(outcome.output.contains("verbatim secret"));
        assert!(outcome.metadata.untrusted_verbatim);
        assert!(outcome.metadata.preserve_inline);
        assert!(outcome.metadata.suppress_activity_result);
        assert!(!outcome
            .metadata
            .persistent_output
            .as_deref()
            .unwrap()
            .contains("verbatim secret"));

        let value: Value = serde_json::from_str(&outcome.output).unwrap();
        let cursor = value["matches"][0]["older_cursor"].as_str().unwrap();
        let newer_cursor = value["matches"][0]["newer_cursor"].as_str().unwrap();
        let page = tool
            .call_with_status_outcome(
                &json!({
                    "action": "page",
                    "cursor": newer_cursor,
                    "_session_id": "root-session",
                    "_channel_visibility": "private",
                    "_user_role": "Owner"
                })
                .to_string(),
                None,
            )
            .await
            .unwrap();
        assert!(page.output.contains("same-session followup"));
        assert!(!page.output.contains("interleaved other-session message"));

        let widened = tool
            .call_with_status_outcome(
                &json!({
                    "action": "page",
                    "cursor": cursor,
                    "_session_id": "different-session",
                    "_channel_visibility": "private",
                    "_user_role": "Owner"
                })
                .to_string(),
                None,
            )
            .await;
        assert!(widened.is_err());
    }

    #[tokio::test]
    async fn scoped_contexts_hide_global_projection_metadata_and_cannot_repair() {
        let (tool, _events, _db) = tool().await;
        let scoped_args = json!({
            "action": "health",
            "_session_id": "slack:C1",
            "_channel_id": "slack:C1",
            "_channel_visibility": "private_group",
            "_user_role": "Owner"
        });
        let health = tool.call(&scoped_args.to_string()).await.unwrap();
        assert!(!health.contains("canonical_messages"));
        assert!(health.contains("counts are hidden"));

        let mut repair_args = scoped_args;
        repair_args["action"] = json!("repair");
        assert!(tool.call(&repair_args.to_string()).await.is_err());
    }
}
