//! Pending approvals that outlive the process.

use std::collections::HashMap;
use std::sync::Mutex;

use aidaemon_core::{RunId, StoreError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::channel::{Conversation, MessageRef};

/// One open approval, as the runner needs it after a restart. The interrupted
/// run itself is not copied here: it is read back from the runner's
/// [`aidaemon_core::RunStore`] by `run_id`, so the approval can never resume
/// anything other than what that store says is waiting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredApproval {
    pub prompt_id: String,
    pub session_id: String,
    /// The channel's name. Channels are matched by name on restart.
    pub channel: String,
    pub conversation: Conversation,
    pub reply_to: Option<String>,
    pub run_id: RunId,
    pub prompt_text: String,
    pub prompt_ref: Option<MessageRef>,
    /// When the approval is answered with Deny, in milliseconds since the
    /// Unix epoch. Wall-clock, so time spent stopped counts.
    pub expires_at_ms: u64,
}

/// Where the runner keeps open approvals. Implement it on your database to
/// survive restarts; [`MemoryApprovalStore`] is the in-process version. Set it
/// with [`crate::ChannelRunnerBuilder::approval_store`].
///
/// A record is removed before its run resumes. A crash between the two leaves
/// a call that never runs, which is the safe way round: the other order could
/// run an approved call twice.
#[async_trait]
pub trait ApprovalStore: Send + Sync {
    /// Insert or replace the record with this `prompt_id`.
    async fn save(&self, approval: &StoredApproval) -> Result<(), StoreError>;
    async fn remove(&self, prompt_id: &str) -> Result<(), StoreError>;
    async fn load_all(&self) -> Result<Vec<StoredApproval>, StoreError>;
}

#[derive(Default)]
pub struct MemoryApprovalStore {
    approvals: Mutex<HashMap<String, StoredApproval>>,
}

impl MemoryApprovalStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ApprovalStore for MemoryApprovalStore {
    async fn save(&self, approval: &StoredApproval) -> Result<(), StoreError> {
        self.approvals
            .lock()
            .expect("approval store poisoned")
            .insert(approval.prompt_id.clone(), approval.clone());
        Ok(())
    }

    async fn remove(&self, prompt_id: &str) -> Result<(), StoreError> {
        self.approvals
            .lock()
            .expect("approval store poisoned")
            .remove(prompt_id);
        Ok(())
    }

    async fn load_all(&self) -> Result<Vec<StoredApproval>, StoreError> {
        Ok(self
            .approvals
            .lock()
            .expect("approval store poisoned")
            .values()
            .cloned()
            .collect())
    }
}
