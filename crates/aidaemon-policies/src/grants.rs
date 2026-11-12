//! Standing approvals the owner gave during a session.

use std::collections::HashSet;
use std::sync::Mutex;

use aidaemon_core::{EffectKey, OwnerApprovalRequest};
use async_trait::async_trait;

use crate::approval::ApprovalGrants;

/// Approvals the owner granted for the rest of a session, kept in memory.
///
/// A grant is keyed by each of the request's effects together with its
/// `subject` (the exact action string), so approving `POST
/// https://h.example.invalid/x` once does not approve another URL or
/// another host. A request without declared effects is keyed by its
/// subject alone. Persisted grants are the application's business: it
/// implements [`ApprovalGrants`] over its own store.
#[derive(Debug, Default)]
pub struct SessionGrants {
    granted: Mutex<HashSet<(Option<EffectKey>, String)>>,
}

impl SessionGrants {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember that the owner approved `request`.
    pub fn grant(&self, request: &OwnerApprovalRequest) {
        let mut granted = self.granted.lock().unwrap_or_else(|e| e.into_inner());
        granted.extend(keys(request));
    }
}

/// The grant keys a request needs: one per effect, or the subject alone.
fn keys(request: &OwnerApprovalRequest) -> Vec<(Option<EffectKey>, String)> {
    if request.effects.is_empty() {
        return vec![(None, request.subject.clone())];
    }
    request
        .effects
        .iter()
        .map(|effect| (Some(effect.clone()), request.subject.clone()))
        .collect()
}

#[async_trait]
impl ApprovalGrants for SessionGrants {
    async fn covers(&self, request: &OwnerApprovalRequest) -> bool {
        let granted = self.granted.lock().unwrap_or_else(|e| e.into_inner());
        keys(request).iter().all(|key| granted.contains(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::ApprovalGrants;
    use aidaemon_core::{ApprovalRisk, EffectClass, EffectKey, OwnerApprovalRequest};

    fn req(subject: &str, effects: Vec<EffectKey>) -> OwnerApprovalRequest {
        OwnerApprovalRequest {
            card: None,
            effects,
            summary: format!("Run {subject} for Jordan Lee"),
            subject: subject.to_string(),
            risk: ApprovalRisk::High,
            warnings: Vec::new(),
        }
    }

    fn host(name: &str) -> EffectKey {
        EffectKey::egress("https", name, None)
    }

    #[tokio::test]
    async fn a_grant_covers_its_subject_and_effects() {
        let grants = SessionGrants::new();
        let post = req(
            "POST https://h.example.invalid/x",
            vec![host("h.example.invalid")],
        );
        assert!(!grants.covers(&post).await);
        grants.grant(&post);
        assert!(grants.covers(&post).await);

        // Another subject, or an effect not granted, is not covered.
        let other_subject = req(
            "POST https://h.example.invalid/y",
            vec![host("h.example.invalid")],
        );
        assert!(!grants.covers(&other_subject).await);
        let wider = req(
            "POST https://h.example.invalid/x",
            vec![host("h.example.invalid"), host("other.example.invalid")],
        );
        assert!(!grants.covers(&wider).await);
    }

    #[tokio::test]
    async fn a_request_without_effects_is_keyed_by_subject() {
        let grants = SessionGrants::new();
        let opaque = req("python3 /tmp/aidaemon-test/run.py", Vec::new());
        assert!(!grants.covers(&opaque).await);
        grants.grant(&opaque);
        assert!(grants.covers(&opaque).await);
        // A grant with effects does not cover the same subject without them.
        let keyed = req(
            "git push",
            vec![EffectKey::named(EffectClass::Process, "git")],
        );
        grants.grant(&keyed);
        assert!(!grants.covers(&req("git push", Vec::new())).await);
    }
}
