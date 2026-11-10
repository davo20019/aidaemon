//! Progressive extraction on daemon turns: the SDK memory crate proposes
//! and writes facts; the daemon records each write in its flight recorder.

use std::sync::Arc;

use serde_json::json;
use tracing::debug;

use crate::events::EventStore;
use crate::types::UserRole;

/// Run progressive fact extraction in the background: the SDK memory
/// crate's [`ProgressiveExtractor`] proposes and writes the facts, and the
/// daemon records the write in its flight recorder.
///
/// [`ProgressiveExtractor`]: aidaemon_memory::progressive::ProgressiveExtractor
#[allow(clippy::too_many_arguments)]
pub fn spawn_progressive_extraction(
    side_calls: crate::llm_runtime::gateway::SideCallGateway,
    fast_model: String,
    store: Arc<dyn aidaemon_memory::stores::MemoryStores>,
    event_store: Arc<EventStore>,
    source_task_id: String,
    user_text: String,
    assistant_response: String,
    channel_id: Option<String>,
    visibility: crate::types::ChannelVisibility,
    user_role: UserRole,
) {
    use aidaemon_memory::progressive::{may_extract, Exchange};
    tokio::spawn(async move {
        if !may_extract(user_role, visibility) {
            return;
        }
        let provider =
            side_calls.provider(crate::llm_runtime::side_calls::SideCall::ProgressiveExtraction);
        // The side-call gateway picks the model; `fast_model` is not sent.
        let extractor = extractor(provider, &fast_model);
        let exchange = Exchange {
            user_text: &user_text,
            assistant_reply: &assistant_response,
            channel_id: channel_id.as_deref(),
        };
        let facts = match extractor.propose(&exchange).await {
            Ok(facts) => facts,
            Err(e) => {
                debug!(error = %e, "Progressive fact extraction failed");
                return;
            }
        };
        if facts.is_empty() {
            return; // Expected for most interactions.
        }
        let written = extractor.write(store.as_ref(), &exchange, facts).await;
        if written.is_empty() {
            return;
        }
        // Flight-recorder entry so every background memory write is
        // auditable: which facts, from which excerpt, into which channel.
        let source_excerpt = crate::utils::truncate_str(&user_text, 200);
        let selected_source_event_id = match event_store
            .project_selected_user_message_memory_span_for_task(&source_task_id)
            .await
        {
            Ok(Some(event_id)) => {
                if let Err(error) = store.repair_memory_projections().await {
                    debug!(%error, %source_task_id, "Failed to bind selected message provenance to extracted memory");
                }
                Some(event_id)
            }
            Ok(None) => {
                debug!(%source_task_id, "Selected memory write had no policy-eligible canonical user event");
                None
            }
            Err(error) => {
                debug!(%error, %source_task_id, "Failed to project selected message provenance");
                None
            }
        };
        let event = crate::events::Event::new(
            "background:progressive_extraction",
            crate::events::EventType::DecisionPoint,
            json!({
                "code": "memory_write",
                "decision_type": "memory_write",
                "severity": "info",
                "summary": format!("Progressive extraction stored {} fact(s)", written.len()),
                "metadata": {
                    "source": "progressive",
                    "source_task_id": source_task_id,
                    "source_event_id": selected_source_event_id,
                    "channel_id": channel_id,
                    "facts": written,
                    "source_excerpt": source_excerpt,
                },
            }),
        );
        if let Err(e) = event_store.append(event).await {
            debug!(error = %e, "Failed to record memory_write event");
        }
    });
}

/// One extractor per process, so its two-call concurrency limit holds
/// across every turn. The provider is rebuilt per call (the runtime may
/// switch models), the limit is not.
fn extractor(
    provider: Arc<dyn aidaemon_core::Provider>,
    model: &str,
) -> aidaemon_memory::progressive::ProgressiveExtractor {
    static PERMITS: std::sync::LazyLock<Arc<tokio::sync::Semaphore>> =
        std::sync::LazyLock::new(|| Arc::new(tokio::sync::Semaphore::new(2)));
    aidaemon_memory::progressive::ProgressiveExtractor::with_permits(
        provider,
        model,
        PERMITS.clone(),
    )
}
