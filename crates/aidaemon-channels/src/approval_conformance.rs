//! Conformance kit for [`ApprovalStore`] implementations (feature
//! `conformance`).
//!
//! The runner treats the store as plain durable storage: it decides expiry,
//! answers and who may answer, and only needs the store to keep exactly what
//! it was given until told to drop it. The kit checks that contract.
//!
//! ```
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! use std::sync::Arc;
//! use aidaemon_channels::approval_conformance::check_approval_store;
//! use aidaemon_channels::{ApprovalStore, MemoryApprovalStore};
//!
//! check_approval_store(|| async {
//!     Arc::new(MemoryApprovalStore::new()) as Arc<dyn ApprovalStore>
//! })
//! .await
//! .assert_all_passed();
//! # }
//! ```

use std::future::Future;
use std::sync::Arc;

use aidaemon_core::RunId;
use aidaemon_core::conformance::ConformanceReport;

use crate::approvals::{ApprovalStore, StoredApproval};
use crate::channel::{Conversation, ConversationKind, MessageRef};

/// Check names, each documenting its rule.
pub mod approval_checks {
    /// A fresh store has nothing to load.
    pub const STARTS_EMPTY: &str = "approvals.starts_empty";
    /// A saved record loads back field for field, including absent options.
    pub const ROUND_TRIP: &str = "approvals.round_trip";
    /// Saving an existing `prompt_id` replaces it; it never duplicates.
    pub const SAVE_REPLACES: &str = "approvals.save_replaces";
    /// Removing drops only that record, and removing a missing one is Ok
    /// (a double press settles the approval once).
    pub const REMOVE_IDEMPOTENT: &str = "approvals.remove_idempotent";
    /// Records for different conversations and channels stay separate and
    /// keep their own scope fields.
    pub const SCOPES_KEPT_APART: &str = "approvals.scopes_kept_apart";
    /// Concurrent saves and removes leave exactly the expected records.
    pub const CONCURRENT_WRITES: &str = "approvals.concurrent_writes";
}

fn record(prompt_id: &str, channel: &str, conversation: &str) -> StoredApproval {
    StoredApproval {
        prompt_id: prompt_id.to_string(),
        session_id: format!("{channel}:{conversation}"),
        channel: channel.to_string(),
        conversation: Conversation {
            id: conversation.to_string(),
            kind: ConversationKind::Direct,
        },
        reply_to: None,
        run_id: RunId::new(),
        prompt_text: format!("Allow the synthetic action {prompt_id}?"),
        prompt_ref: None,
        expires_at_ms: 4_000_000_000_000,
    }
}

fn sorted(mut all: Vec<StoredApproval>) -> Vec<StoredApproval> {
    all.sort_by(|a, b| a.prompt_id.cmp(&b.prompt_id));
    all
}

/// Run every check against stores made by `make`. Each check gets a fresh
/// store.
pub async fn check_approval_store<F, Fut>(make: F) -> ConformanceReport
where
    F: Fn() -> Fut,
    Fut: Future<Output = Arc<dyn ApprovalStore>>,
{
    use approval_checks::*;
    let mut report = ConformanceReport::new("ApprovalStore");

    let store = make().await;
    report.record(
        STARTS_EMPTY,
        store
            .load_all()
            .await
            .map_err(|e| e.to_string())
            .and_then(|all| {
                if all.is_empty() {
                    Ok(())
                } else {
                    Err(format!("{} records in a fresh store", all.len()))
                }
            }),
    );

    let store = make().await;
    let mut full = record("p1", "telegram", "chat-1");
    full.reply_to = Some("m-7".to_string());
    full.prompt_ref = Some(MessageRef {
        conversation_id: "chat-1".to_string(),
        message_id: "m-9".to_string(),
    });
    let bare = record("p2", "telegram", "chat-1");
    report.record(
        ROUND_TRIP,
        async {
            store.save(&full).await.map_err(|e| e.to_string())?;
            store.save(&bare).await.map_err(|e| e.to_string())?;
            let got = sorted(store.load_all().await.map_err(|e| e.to_string())?);
            if got == vec![full.clone(), bare.clone()] {
                Ok(())
            } else {
                Err(format!("loaded {got:?}"))
            }
        }
        .await,
    );

    let store = make().await;
    report.record(
        SAVE_REPLACES,
        async {
            let first = record("p1", "telegram", "chat-1");
            let mut second = first.clone();
            second.prompt_ref = Some(MessageRef {
                conversation_id: "chat-1".to_string(),
                message_id: "m-2".to_string(),
            });
            second.expires_at_ms += 1;
            store.save(&first).await.map_err(|e| e.to_string())?;
            store.save(&second).await.map_err(|e| e.to_string())?;
            let got = store.load_all().await.map_err(|e| e.to_string())?;
            if got == vec![second] {
                Ok(())
            } else {
                Err(format!("loaded {got:?}"))
            }
        }
        .await,
    );

    let store = make().await;
    report.record(
        REMOVE_IDEMPOTENT,
        async {
            let keep = record("keep", "telegram", "chat-1");
            store.save(&keep).await.map_err(|e| e.to_string())?;
            store
                .save(&record("drop", "telegram", "chat-1"))
                .await
                .map_err(|e| e.to_string())?;
            store.remove("drop").await.map_err(|e| e.to_string())?;
            store
                .remove("drop")
                .await
                .map_err(|e| format!("second remove failed: {e}"))?;
            store
                .remove("never-saved")
                .await
                .map_err(|e| format!("removing a missing id failed: {e}"))?;
            let got = store.load_all().await.map_err(|e| e.to_string())?;
            if got == vec![keep] {
                Ok(())
            } else {
                Err(format!("loaded {got:?}"))
            }
        }
        .await,
    );

    let store = make().await;
    report.record(
        SCOPES_KEPT_APART,
        async {
            let a = record("a", "telegram", "chat-1");
            let b = record("b", "telegram", "chat-2");
            let c = record("c", "slack", "chat-1");
            for r in [&a, &b, &c] {
                store.save(r).await.map_err(|e| e.to_string())?;
            }
            store.remove("b").await.map_err(|e| e.to_string())?;
            let got = sorted(store.load_all().await.map_err(|e| e.to_string())?);
            if got == vec![a, c] {
                Ok(())
            } else {
                Err(format!("loaded {got:?}"))
            }
        }
        .await,
    );

    let store = make().await;
    report.record(
        CONCURRENT_WRITES,
        async {
            let mut tasks = Vec::new();
            for i in 0..16 {
                let store = store.clone();
                tasks.push(tokio::spawn(async move {
                    let id = format!("p{i:02}");
                    store.save(&record(&id, "telegram", "chat-1")).await?;
                    if i % 2 == 1 {
                        store.remove(&id).await?;
                        store.remove(&id).await?;
                    }
                    Ok::<_, aidaemon_core::StoreError>(())
                }));
            }
            for task in tasks {
                task.await
                    .map_err(|e| e.to_string())?
                    .map_err(|e| e.to_string())?;
            }
            let got: Vec<String> = sorted(store.load_all().await.map_err(|e| e.to_string())?)
                .into_iter()
                .map(|r| r.prompt_id)
                .collect();
            let want: Vec<String> = (0..16)
                .filter(|i| i % 2 == 0)
                .map(|i| format!("p{i:02}"))
                .collect();
            if got == want {
                Ok(())
            } else {
                Err(format!("loaded {got:?}, wanted {want:?}"))
            }
        }
        .await,
    );

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryApprovalStore;

    #[tokio::test]
    async fn the_memory_store_passes_the_kit() {
        check_approval_store(|| async {
            Arc::new(MemoryApprovalStore::new()) as Arc<dyn ApprovalStore>
        })
        .await
        .assert_all_passed();
    }
}
