# Event-Sourced Architecture Implementation Plan

## Overview

Migrate aidaemon from message-based persistence to pure event sourcing, where:
- Every action is an immutable event
- Messages table is replaced by events
- Session context is compiled from events
- Long-term memory is consolidated from events before pruning

**Goals:**
1. Agent knows "what it's doing" and "what just happened"
2. Single source of truth for all activity
3. Better input for learning/memory system
4. Full audit trail for debugging

**Constraints:**
- 2 users - migration risk is low
- No ToolProgress persistence (ToolResult captures output)
- 7-day retention with consolidation before pruning

---

## Event Schema

### Core Event Structure

```rust
// src/events/mod.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: i64,
    pub session_id: String,
    pub event_type: EventType,
    pub data: JsonValue,
    pub created_at: DateTime<Utc>,
    pub consolidated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    // Session lifecycle
    SessionStart,
    SessionEnd,

    // Messages (replaces messages table)
    UserMessage,
    AssistantResponse,

    // Tool lifecycle
    ToolCall,
    ToolResult,

    // Agent thinking
    ThinkingStart,      // Iteration N begins

    // Task lifecycle
    TaskStart,
    TaskEnd,

    // Errors
    Error,

    // Sub-agents
    SubAgentSpawn,
    SubAgentComplete,

    // Approvals
    ApprovalRequested,
    ApprovalResponse,
}
```

### Event Data Payloads

```rust
// src/events/payloads.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessageData {
    pub content: String,
    pub message_id: Option<String>,  // Platform message ID
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantResponseData {
    pub content: String,
    pub tool_calls: Option<Vec<ToolCallData>>,
    pub model: String,
    pub tokens_used: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallData {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultData {
    pub tool_call_id: String,
    pub name: String,
    pub result: String,
    pub success: bool,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStartData {
    pub task_id: String,
    pub description: String,
    pub parent_task_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEndData {
    pub task_id: String,
    pub status: TaskStatus,  // Completed, Cancelled, Failed
    pub duration_secs: u64,
    pub iterations: u32,
    pub tool_calls_count: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorData {
    pub message: String,
    pub error_type: String,  // tool_error, llm_error, timeout, etc.
    pub context: Option<String>,
    pub recoverable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingStartData {
    pub iteration: u32,
    pub task_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentSpawnData {
    pub child_session_id: String,
    pub mission: String,
    pub task: String,
    pub depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubAgentCompleteData {
    pub child_session_id: String,
    pub success: bool,
    pub result_summary: String,
    pub duration_secs: u64,
}
```

### Database Schema

```sql
-- Migration: create_events_table

CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    data TEXT NOT NULL,  -- JSON
    created_at TEXT NOT NULL,
    consolidated_at TEXT,

    -- For efficient queries
    task_id TEXT,  -- Extracted from data for indexing
    tool_name TEXT  -- Extracted from data for indexing
);

CREATE INDEX idx_events_session_time ON events(session_id, created_at DESC);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_task ON events(task_id) WHERE task_id IS NOT NULL;
CREATE INDEX idx_events_consolidation ON events(consolidated_at) WHERE consolidated_at IS NULL;
CREATE INDEX idx_events_prune ON events(created_at) WHERE consolidated_at IS NOT NULL;
```

---

## Session Context Compilation

### Context Structure

```rust
// src/events/context.rs

#[derive(Debug, Clone, Serialize)]
pub struct SessionContext {
    pub current_task: Option<CurrentTask>,
    pub last_completed_task: Option<CompletedTask>,
    pub last_error: Option<RecentError>,
    pub recent_tools: Vec<RecentTool>,
    pub current_iteration: Option<u32>,
    pub active_sub_agents: Vec<ActiveSubAgent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CurrentTask {
    pub task_id: String,
    pub description: String,
    pub started_at: DateTime<Utc>,
    pub elapsed_secs: u64,
    pub iterations: u32,
    pub tool_calls: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletedTask {
    pub task_id: String,
    pub description: String,
    pub status: TaskStatus,
    pub duration_secs: u64,
    pub error: Option<String>,
    pub completed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentError {
    pub message: String,
    pub error_type: String,
    pub occurred_at: DateTime<Utc>,
    pub task_context: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecentTool {
    pub name: String,
    pub summary: String,
    pub success: bool,
    pub timestamp: DateTime<Utc>,
}
```

### Context Compiler

```rust
// src/events/context.rs

impl EventStore {
    pub async fn compile_session_context(
        &self,
        session_id: &str,
        window: Duration,
    ) -> anyhow::Result<SessionContext> {
        let since = Utc::now() - window;
        let events = self.query_events(session_id, since).await?;

        Ok(SessionContext {
            current_task: self.find_current_task(&events),
            last_completed_task: self.find_last_completed_task(&events),
            last_error: self.find_last_error(&events),
            recent_tools: self.extract_recent_tools(&events, 10),
            current_iteration: self.find_current_iteration(&events),
            active_sub_agents: self.find_active_sub_agents(&events),
        })
    }

    fn find_current_task(&self, events: &[Event]) -> Option<CurrentTask> {
        // Find TaskStart without matching TaskEnd
        // ...
    }

    fn find_last_error(&self, events: &[Event]) -> Option<RecentError> {
        // Find most recent Error event
        // ...
    }

    // ... other extraction methods
}
```

### System Prompt Injection

```rust
// src/events/context.rs

impl SessionContext {
    pub fn format_for_prompt(&self) -> String {
        let mut lines = vec!["## Current Session Activity".to_string()];

        if let Some(task) = &self.current_task {
            lines.push(format!(
                "- **Active task:** \"{}\" (running {}s, iteration {}, {} tool calls)",
                task.description,
                task.elapsed_secs,
                task.iterations,
                task.tool_calls
            ));
        }

        if let Some(task) = &self.last_completed_task {
            let status_str = match task.status {
                TaskStatus::Completed => "completed",
                TaskStatus::Cancelled => "CANCELLED",
                TaskStatus::Failed => "FAILED",
            };
            lines.push(format!(
                "- **Last task:** \"{}\" - {} ({}s){}",
                task.description,
                status_str,
                task.duration_secs,
                task.error.as_ref().map(|e| format!(" - Error: {}", e)).unwrap_or_default()
            ));
        }

        if let Some(error) = &self.last_error {
            lines.push(format!(
                "- **Recent error:** {} ({})",
                error.message,
                error.error_type
            ));
        }

        if !self.recent_tools.is_empty() {
            let tool_summary: Vec<String> = self.recent_tools
                .iter()
                .take(5)
                .map(|t| format!("{}({})", t.name, if t.success { "ok" } else { "err" }))
                .collect();
            lines.push(format!("- **Recent tools:** {}", tool_summary.join(", ")));
        }

        lines.join("\n")
    }
}
```

---

## Messages Migration Strategy

### Current Messages Structure

```rust
// Current (to be replaced)
pub struct Message {
    pub id: String,
    pub session_id: String,
    pub role: String,          // "user", "assistant", "tool"
    pub content: String,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub tool_calls_json: Option<String>,
    pub created_at: DateTime<Utc>,
    pub importance: f32,
    pub embedding: Option<Vec<f32>>,
}
```

### Mapping to Events

| Message Role | Event Type | Notes |
|--------------|------------|-------|
| `user` | `UserMessage` | content → data.content |
| `assistant` | `AssistantResponse` | content + tool_calls_json → data |
| `tool` | `ToolResult` | Already captured separately |

### Conversation History from Events

```rust
// src/events/history.rs

impl EventStore {
    /// Get conversation history in the format needed for LLM providers
    pub async fn get_conversation_history(
        &self,
        session_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ConversationMessage>> {
        let events = self.query_events_by_types(
            session_id,
            &[EventType::UserMessage, EventType::AssistantResponse, EventType::ToolResult],
            limit * 3,  // Fetch more to account for tool results
        ).await?;

        // Convert events to conversation format
        let mut messages = Vec::new();
        for event in events {
            match event.event_type {
                EventType::UserMessage => {
                    let data: UserMessageData = serde_json::from_value(event.data)?;
                    messages.push(ConversationMessage {
                        role: Role::User,
                        content: data.content,
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                EventType::AssistantResponse => {
                    let data: AssistantResponseData = serde_json::from_value(event.data)?;
                    messages.push(ConversationMessage {
                        role: Role::Assistant,
                        content: data.content,
                        tool_calls: data.tool_calls,
                        tool_call_id: None,
                    });
                }
                EventType::ToolResult => {
                    let data: ToolResultData = serde_json::from_value(event.data)?;
                    messages.push(ConversationMessage {
                        role: Role::Tool,
                        content: data.result,
                        tool_calls: None,
                        tool_call_id: Some(data.tool_call_id),
                    });
                }
                _ => {}
            }
        }

        // Apply limit and return in chronological order
        Ok(messages.into_iter().rev().take(limit).rev().collect())
    }
}
```

### Migration Script

```rust
// src/events/migration.rs

pub async fn migrate_messages_to_events(
    pool: &SqlitePool,
    event_store: &EventStore,
) -> anyhow::Result<MigrationStats> {
    let mut stats = MigrationStats::default();

    // Fetch all messages ordered by session and time
    let messages: Vec<Message> = sqlx::query_as(
        "SELECT * FROM messages ORDER BY session_id, created_at"
    )
    .fetch_all(pool)
    .await?;

    for msg in messages {
        let event = match msg.role.as_str() {
            "user" => Event {
                id: 0,  // Auto-generated
                session_id: msg.session_id,
                event_type: EventType::UserMessage,
                data: serde_json::to_value(UserMessageData {
                    content: msg.content,
                    message_id: Some(msg.id),
                })?,
                created_at: msg.created_at,
                consolidated_at: None,
            },
            "assistant" => {
                let tool_calls: Option<Vec<ToolCallData>> = msg.tool_calls_json
                    .and_then(|json| serde_json::from_str(&json).ok());

                Event {
                    id: 0,
                    session_id: msg.session_id,
                    event_type: EventType::AssistantResponse,
                    data: serde_json::to_value(AssistantResponseData {
                        content: msg.content,
                        tool_calls,
                        model: "unknown".to_string(),  // Not in old schema
                        tokens_used: None,
                    })?,
                    created_at: msg.created_at,
                    consolidated_at: None,
                }
            },
            "tool" => Event {
                id: 0,
                session_id: msg.session_id,
                event_type: EventType::ToolResult,
                data: serde_json::to_value(ToolResultData {
                    tool_call_id: msg.tool_call_id.unwrap_or_default(),
                    name: msg.tool_name.unwrap_or_default(),
                    result: msg.content,
                    success: true,  // Can't determine from old data
                    duration_ms: 0,
                })?,
                created_at: msg.created_at,
                consolidated_at: None,
            },
            _ => continue,
        };

        event_store.append(event).await?;
        stats.migrated += 1;
    }

    Ok(stats)
}
```

---

## Consolidation System

### Consolidation Triggers

1. **End of Session** - When task completes or session goes idle
2. **Daily Cron** - Run once per day for all sessions

### Consolidation Process

```rust
// src/events/consolidation.rs

pub struct Consolidator {
    event_store: Arc<EventStore>,
    memory_manager: Arc<MemoryManager>,
}

impl Consolidator {
    /// Consolidate events into long-term memory
    pub async fn consolidate_session(&self, session_id: &str) -> anyhow::Result<ConsolidationResult> {
        // Get unconsolidated events
        let events = self.event_store
            .query_unconsolidated(session_id)
            .await?;

        if events.is_empty() {
            return Ok(ConsolidationResult::empty());
        }

        let mut result = ConsolidationResult::default();

        // 1. Extract procedures from successful task sequences
        let procedures = self.extract_procedures(&events).await?;
        for proc in procedures {
            self.memory_manager.save_procedure(&proc).await?;
            result.procedures_created += 1;
        }

        // 2. Extract error-solution pairs
        let error_solutions = self.extract_error_solutions(&events).await?;
        for es in error_solutions {
            self.memory_manager.save_error_solution(&es).await?;
            result.error_solutions_created += 1;
        }

        // 3. Update expertise from domain activity
        let expertise_updates = self.extract_expertise_updates(&events).await?;
        for update in expertise_updates {
            self.memory_manager.update_expertise(&update).await?;
            result.expertise_updated += 1;
        }

        // 4. Create episode summary
        let episode = self.create_episode_summary(&events).await?;
        self.memory_manager.save_episode(&episode).await?;
        result.episodes_created = 1;

        // 5. Mark events as consolidated
        let event_ids: Vec<i64> = events.iter().map(|e| e.id).collect();
        self.event_store.mark_consolidated(&event_ids).await?;
        result.events_consolidated = event_ids.len();

        Ok(result)
    }

    async fn extract_procedures(&self, events: &[Event]) -> anyhow::Result<Vec<Procedure>> {
        // Find TaskStart → ToolCall* → TaskEnd(success) sequences
        // Extract as procedures with trigger patterns
        // ...
    }

    async fn extract_error_solutions(&self, events: &[Event]) -> anyhow::Result<Vec<ErrorSolution>> {
        // Find Error → (actions) → Success sequences
        // Extract as error-solution mappings
        // ...
    }

    async fn extract_expertise_updates(&self, events: &[Event]) -> anyhow::Result<Vec<ExpertiseUpdate>> {
        // Analyze tool usage and task domains
        // Update expertise levels
        // ...
    }

    async fn create_episode_summary(&self, events: &[Event]) -> anyhow::Result<Episode> {
        // Summarize the session activity
        // Use LLM for summary if needed
        // ...
    }
}
```

### Consolidation Triggers Implementation

```rust
// src/events/consolidation.rs

impl Consolidator {
    /// Trigger: End of session/task
    pub async fn on_session_end(&self, session_id: &str) {
        if let Err(e) = self.consolidate_session(session_id).await {
            warn!("Consolidation failed for session {}: {}", session_id, e);
        }
    }

    /// Trigger: Daily cron job
    pub async fn daily_consolidation(&self) -> anyhow::Result<DailyConsolidationStats> {
        let sessions = self.event_store.get_sessions_needing_consolidation().await?;

        let mut stats = DailyConsolidationStats::default();
        for session_id in sessions {
            match self.consolidate_session(&session_id).await {
                Ok(result) => stats.add(result),
                Err(e) => {
                    warn!("Consolidation failed for {}: {}", session_id, e);
                    stats.failures += 1;
                }
            }
        }

        Ok(stats)
    }
}
```

---

## Pruning System

### Pruning Strategy

- Events older than 7 days AND consolidated → delete
- Events older than 7 days AND NOT consolidated → consolidate first, then delete
- Never delete unconsolidated events without attempting consolidation

```rust
// src/events/pruning.rs

pub struct Pruner {
    event_store: Arc<EventStore>,
    consolidator: Arc<Consolidator>,
    retention_days: u32,
}

impl Pruner {
    pub async fn prune(&self) -> anyhow::Result<PruneStats> {
        let cutoff = Utc::now() - Duration::days(self.retention_days as i64);

        // First, consolidate any old unconsolidated events
        let unconsolidated_sessions = self.event_store
            .get_sessions_with_old_unconsolidated_events(cutoff)
            .await?;

        for session_id in unconsolidated_sessions {
            self.consolidator.consolidate_session(&session_id).await?;
        }

        // Now prune consolidated events older than cutoff
        let deleted = self.event_store.delete_old_consolidated(cutoff).await?;

        Ok(PruneStats { deleted })
    }
}
```

---

## Implementation Phases

### Phase 1: Event Store Foundation
**Files to create:**
- `src/events/mod.rs` - Module root, Event struct, EventType enum
- `src/events/payloads.rs` - Event data structures
- `src/events/store.rs` - EventStore with CRUD operations

**Files to modify:**
- `src/state/sqlite.rs` - Add events table migration
- `src/lib.rs` - Add events module

**Tasks:**
- [ ] Create events module structure
- [ ] Define Event struct and EventType enum
- [ ] Define all payload structs
- [ ] Create EventStore with append/query methods
- [ ] Add database migration for events table
- [ ] Write unit tests for EventStore

### Phase 2: Event Emission
**Files to modify:**
- `src/agent.rs` - Emit Thinking, Error events
- `src/tasks.rs` - Emit TaskStart, TaskEnd events (enhance TaskRegistry)
- `src/tools/mod.rs` - Emit ToolCall, ToolResult events
- `src/tools/spawn.rs` - Emit SubAgentSpawn, SubAgentComplete events
- `src/channels/telegram.rs` - Emit UserMessage, SessionStart events
- `src/channels/slack.rs` - Same as telegram
- `src/channels/discord.rs` - Same as telegram

**Tasks:**
- [ ] Add EventStore to Agent struct
- [ ] Emit TaskStart/TaskEnd from task lifecycle
- [ ] Emit ThinkingStart at each iteration
- [ ] Emit ToolCall before tool execution
- [ ] Emit ToolResult after tool execution
- [ ] Emit Error on failures
- [ ] Emit UserMessage on message receipt
- [ ] Emit AssistantResponse after LLM response
- [ ] Emit SubAgent events from spawn tool

### Phase 3: Context Compilation
**Files to create:**
- `src/events/context.rs` - SessionContext and compiler

**Files to modify:**
- `src/skills.rs` - Inject session context into system prompt
- `src/agent.rs` - Compile context before LLM call

**Tasks:**
- [ ] Create SessionContext struct
- [ ] Implement context compilation from events
- [ ] Implement format_for_prompt()
- [ ] Integrate into system prompt building
- [ ] Test "what are you doing?" scenario
- [ ] Test "what was the error?" scenario

### Phase 4: Messages Migration
**Files to create:**
- `src/events/migration.rs` - Migration script
- `src/events/history.rs` - Conversation history from events

**Files to modify:**
- `src/state/sqlite.rs` - Deprecate messages methods
- `src/agent.rs` - Use events for history instead of messages
- `src/providers/*.rs` - Update to use new history format

**Tasks:**
- [ ] Create migration script
- [ ] Create get_conversation_history from events
- [ ] Run migration on existing data (backup first!)
- [ ] Update agent to use events for history
- [ ] Update tri-hybrid retrieval to use events
- [ ] Remove/deprecate messages table methods
- [ ] Test conversation continuity

### Phase 5: Consolidation System
**Files to create:**
- `src/events/consolidation.rs` - Consolidator

**Files to modify:**
- `src/agent.rs` - Trigger consolidation on task end
- `src/core.rs` - Add daily consolidation cron
- `src/memory/manager.rs` - Ensure memory methods support consolidation input

**Tasks:**
- [ ] Create Consolidator struct
- [ ] Implement procedure extraction
- [ ] Implement error-solution extraction
- [ ] Implement expertise updates
- [ ] Implement episode creation
- [ ] Add end-of-session trigger
- [ ] Add daily cron trigger
- [ ] Test consolidation quality

### Phase 6: Pruning System
**Files to create:**
- `src/events/pruning.rs` - Pruner

**Files to modify:**
- `src/core.rs` - Add pruning to daily cron

**Tasks:**
- [ ] Create Pruner struct
- [ ] Implement safe pruning (consolidate first)
- [ ] Add to daily cron
- [ ] Test retention policy
- [ ] Monitor storage growth

---

## File Structure After Implementation

```
src/
├── events/
│   ├── mod.rs           # Event, EventType, re-exports
│   ├── payloads.rs      # Event data structs
│   ├── store.rs         # EventStore (CRUD)
│   ├── context.rs       # SessionContext, compilation
│   ├── history.rs       # Conversation history from events
│   ├── consolidation.rs # Consolidator
│   ├── pruning.rs       # Pruner
│   └── migration.rs     # Messages → Events migration
├── state/
│   └── sqlite.rs        # Updated with events table
├── memory/
│   └── manager.rs       # Enhanced for consolidation
├── agent.rs             # Event emission + context injection
├── tasks.rs             # TaskStart/TaskEnd events
└── ...
```

---

## Testing Checklist

### Scenario Tests
- [ ] User asks "what are you doing?" during task → gets current activity
- [ ] User asks "what was the error?" after failure → gets error details
- [ ] Task completes → events consolidated to memory
- [ ] Events older than 7 days → pruned after consolidation
- [ ] Process restart → events persisted, context available
- [ ] Sub-agent spawned → events capture hierarchy

### Migration Tests
- [ ] Existing messages migrated correctly
- [ ] Conversation history works from events
- [ ] No data loss during migration
- [ ] Rollback possible if needed

### Performance Tests
- [ ] Event append is fast (<5ms)
- [ ] Context compilation is fast (<50ms)
- [ ] History query is fast (<100ms)
- [ ] Consolidation doesn't block agent

---

## Rollback Plan

1. **Before migration:** Backup entire SQLite database
2. **Keep messages table:** Don't drop until Phase 4 is validated
3. **Feature flag:** Add `use_event_store: bool` config to toggle
4. **If issues:** Revert to messages-based system, restore backup

---

## Success Metrics

1. **Immediate:** "What are you doing?" and "What was the error?" work correctly
2. **Short-term:** Events capture all agent activity
3. **Medium-term:** Consolidation improves long-term memory quality
4. **Long-term:** Storage stays bounded via pruning

---

## Next Steps

1. Review and approve this plan
2. Start Phase 1: Event Store Foundation
3. Iterate through phases with testing at each step
