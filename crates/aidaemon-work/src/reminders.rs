//! Typed reminders.
//!
//! A reminder is a scheduled goal whose only deliverable is a message back to
//! the user, stored as the goal context's typed `reminder_message` (set from
//! `set_reminder` / `schedule_task`'s typed `reminder` argument). The
//! schedule fire path sends that message directly instead of dispatching an
//! agent. Nothing here reads the goal's description.

/// Goal-context key holding a reminder's message. Set by
/// `create_scheduled_goal` from its typed `reminder` argument; a goal
/// carrying it is delivered as that message, with no agent run.
pub const CONTEXT_KEY: &str = "reminder_message";

/// The goal context for a typed reminder.
pub fn context_for_message(message: &str) -> String {
    serde_json::json!({ CONTEXT_KEY: message.trim() }).to_string()
}

/// The typed reminder message stored in a goal's context, if any.
pub fn message_from_context(context: Option<&str>) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(context?).ok()?;
    let message = value.get(CONTEXT_KEY)?.as_str()?.trim();
    (!message.is_empty()).then(|| message.to_string())
}

/// The text sent when a goal fires as a reminder: only its typed message.
///
/// A goal's description is never read for this. An older goal whose
/// description merely starts with "remind me ..." and has no typed message
/// runs as a normal goal: the agent reads the instruction and does what it
/// says, so "remind me to check the server and restart it" restarts it
/// rather than sending canned text.
pub fn fire_text_for_goal(context: Option<&str>) -> Option<String> {
    message_from_context(context).map(|message| format!("⏰ Reminder: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_typed_message_makes_a_goal_a_reminder() {
        let context = context_for_message("Check the oven.");
        assert_eq!(
            fire_text_for_goal(Some(&context)).as_deref(),
            Some("⏰ Reminder: Check the oven.")
        );
        assert_eq!(fire_text_for_goal(None), None);
        assert_eq!(fire_text_for_goal(Some("{\"other\":1}")), None);
        assert_eq!(fire_text_for_goal(Some("not json")), None);
    }
}
