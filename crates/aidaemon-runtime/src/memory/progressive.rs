//! Progressive extraction on daemon turns: the full memory system extracts
//! and writes the facts ([`extract_progressive`]); the daemon's event log is
//! its provenance and flight recorder.
//!
//! [`extract_progressive`]: aidaemon_memory::full::writer::extract_progressive

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use aidaemon_memory::full::writer::{ExtractionProvenance, ExtractionRecord, ExtractionRun};

use aidaemon_core::ChannelVisibility;
use aidaemon_events::{Event, EventStore, EventType};
use aidaemon_tools::semantics::UserRole;

use crate::llm::gateway::SideCallGateway;
use crate::llm::side_calls::SideCall;

/// Run progressive fact extraction in the background, with the daemon's
/// event log as provenance.
#[allow(clippy::too_many_arguments)]
pub fn spawn_progressive_extraction(
    side_calls: SideCallGateway,
    fast_model: String,
    store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    event_store: Arc<EventStore>,
    source_task_id: String,
    user_text: String,
    assistant_response: String,
    channel_id: Option<String>,
    visibility: ChannelVisibility,
    user_role: UserRole,
    run_started_at: chrono::DateTime<chrono::Utc>,
) {
    tokio::spawn(async move {
        if !aidaemon_memory::progressive::may_extract(user_role, visibility) {
            return;
        }
        let provider = side_calls.provider(SideCall::ProgressiveExtraction);
        // The side-call gateway picks the model; `fast_model` is not sent.
        aidaemon_memory::full::writer::extract_progressive(
            provider,
            &fast_model,
            store,
            Some(Arc::new(EventProvenance(event_store))),
            ExtractionRun {
                task_id: source_task_id,
                user_text,
                reply: assistant_response,
                channel_id,
                visibility,
                user_role,
                run_started_at,
            },
        )
        .await;
    });
}

/// The event log as extraction provenance: the task's user message is
/// projected as the facts' source, and every background memory write gets a
/// flight-recorder entry (which facts, from which excerpt, into which
/// channel).
struct EventProvenance(Arc<EventStore>);

#[async_trait]
impl ExtractionProvenance for EventProvenance {
    async fn source_event(&self, task_id: &str) -> anyhow::Result<Option<i64>> {
        self.0
            .project_selected_user_message_memory_span_for_task(task_id)
            .await
    }

    async fn record(&self, record: ExtractionRecord) -> anyhow::Result<()> {
        let event = Event::new(
            "background:progressive_extraction",
            EventType::DecisionPoint,
            json!({
                "code": "memory_write",
                "decision_type": "memory_write",
                "severity": "info",
                "summary": format!("Progressive extraction stored {} fact(s)", record.facts.len()),
                "metadata": {
                    "source": "progressive",
                    "source_task_id": record.source_task_id,
                    "source_event_id": record.source_event_id,
                    "channel_id": record.channel_id,
                    "facts": record.facts,
                    "source_excerpt": record.source_excerpt,
                },
            }),
        );
        self.0.append(event).await?;
        Ok(())
    }
}
