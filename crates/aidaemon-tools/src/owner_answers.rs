//! Owner approvals the agent kernel delivered for exact call subjects.
//!
//! A tool whose ask depends on state it reads before the call (the page a
//! browser action lands on, the app computer_use controls) judges the call
//! in `ToolSemantics::owner_approval_request`, and the kernel's resolver
//! hands the owner's answer back through `Tool::record_owner_answer`. The
//! tool keeps that approval here, for its session and the exact subject the
//! owner saw, and the call it approved consumes it. At execution the tool
//! judges the call again; an ask whose subject was not approved (the target
//! changed after the card, or the call never reached the kernel) is refused
//! rather than run or asked about inside the tool.

use std::collections::HashMap;
use std::sync::Mutex;

/// Which chat's approvals cover a session: a sub-agent's session is bound
/// to its parent chat, whose grants its other agents share.
#[derive(Debug, Default)]
pub struct GrantSessions(Mutex<HashMap<String, String>>);

impl GrantSessions {
    /// Record that `session`'s approvals are those of `grant_session`.
    pub fn bind(&self, session: &str, grant_session: &str) {
        if session.is_empty() || grant_session.is_empty() || session == grant_session {
            return;
        }
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(session.to_string(), grant_session.to_string());
    }

    /// The key `session`'s approvals are kept under.
    pub fn key(&self, session: &str) -> String {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session)
            .cloned()
            .unwrap_or_else(|| session.to_string())
    }
}

/// Approvals by (session, subject), each good for one call.
#[derive(Debug, Default)]
pub struct OwnerAnsweredCalls(Mutex<HashMap<(String, String), u32>>);

impl OwnerAnsweredCalls {
    /// The owner approved `subject` in `session` for one call.
    pub fn approve(&self, session: &str, subject: &str) {
        *self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry((session.to_string(), subject.to_string()))
            .or_default() += 1;
    }

    /// Consume one approval of `subject` in `session`, if there is one.
    pub fn take(&self, session: &str, subject: &str) -> bool {
        let mut approved = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let key = (session.to_string(), subject.to_string());
        match approved.get_mut(&key) {
            Some(count) if *count > 1 => {
                *count -= 1;
                true
            }
            Some(_) => {
                approved.remove(&key);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_approval_is_good_for_one_call_in_its_session() {
        let answers = OwnerAnsweredCalls::default();
        answers.approve("session-a", "subject");
        assert!(!answers.take("session-b", "subject"));
        assert!(!answers.take("session-a", "other"));
        assert!(answers.take("session-a", "subject"));
        assert!(!answers.take("session-a", "subject"));
        answers.approve("session-a", "subject");
        answers.approve("session-a", "subject");
        assert!(answers.take("session-a", "subject"));
        assert!(answers.take("session-a", "subject"));
        assert!(!answers.take("session-a", "subject"));
    }
}
