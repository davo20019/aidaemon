//! The one redaction boundary for what the owner is shown.
//!
//! Every platform adapter passes each request it sends (a message, an edit,
//! a card, an approval prompt, a caption) through [`redact_body`] or
//! [`redact_text`] at its wire, the one place all of its traffic leaves
//! through, so no path (a card line, a status, a result, an approval card,
//! a command's answer) can reach a chat unredacted. It hides every
//! registered secret ([`aidaemon_core::redaction::register_secret`]: each
//! value a secret store handed out and each credential a tool printed) and
//! the credential shapes the kernel's `Redactor` knows, and registers a
//! labelled credential it sees for the first time.

/// `text` as the owner may see it, with the process-wide secrets
/// ([`aidaemon_core::RedactionScope::process`]).
pub fn redact_text(text: &str) -> String {
    aidaemon_tools::sanitize::redact_for_owner(text)
}

/// [`redact_text`] with the secrets registered in `scope`.
pub fn redact_text_in(scope: &aidaemon_core::RedactionScope, text: &str) -> String {
    aidaemon_tools::sanitize::redact_for_owner_in(scope, text)
}

/// A platform request's body with every string in it redacted, whatever
/// its fields are called.
pub fn redact_body(body: &serde_json::Value) -> serde_json::Value {
    redact_body_in(&aidaemon_core::RedactionScope::process(), body)
}

/// [`redact_body`] with the secrets registered in `scope`.
pub fn redact_body_in(
    scope: &aidaemon_core::RedactionScope,
    body: &serde_json::Value,
) -> serde_json::Value {
    let mut body = body.clone();
    aidaemon_tools::sanitize::redact_json_for_owner_in(scope, &mut body);
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live 2026-11-13: a helper's card showed `drush site:install`'s
    /// "User password: …" line. Whatever path builds the text, the wire
    /// hides the value, and once seen it stays hidden unlabelled too.
    #[test]
    fn a_password_line_never_reaches_a_chat() {
        let body = serde_json::json!({
            "chat_id": 42,
            "text": "Running a command · [success] User name: admin  User password: Wire-Synthetic-91",
            "reply_markup": {"inline_keyboard": [[{"text": "OK", "callback_data": "a:1"}]]},
        });
        let sent = redact_body(&body);
        let text = sent["text"].as_str().unwrap();
        assert!(!text.contains("Wire-Synthetic-91"), "{text}");
        assert!(text.contains("User password: [REDACTED:"), "{text}");
        assert_eq!(sent["chat_id"], 42);
        assert_eq!(sent["reply_markup"], body["reply_markup"]);
        let later = redact_text("Your admin login is admin / Wire-Synthetic-91.");
        assert!(!later.contains("Wire-Synthetic-91"), "{later}");
    }
}
