//! Plain-reminder detection and rendering.
//!
//! A "plain reminder" is a scheduled goal whose only deliverable is a message
//! back to the user (e.g. "remind me to call my daughter"). These need none of
//! the heavy scheduled-goal machinery: no approval gate, no task lead, no
//! progress heartbeats. The scheduling path auto-confirms them with a single
//! friendly message, and the schedule fire path sends the reminder text
//! directly instead of dispatching an agent.

/// A parsed plain reminder.
#[derive(Debug, Clone, PartialEq)]
pub struct Reminder {
    /// Connective from the original phrasing ("to", "about", "of", "that"),
    /// used to rebuild a natural sentence: "I'll remind you {connective} {body}".
    pub connective: Option<&'static str>,
    /// Reminder content with pronouns flipped to second person and trailing
    /// punctuation trimmed.
    pub body: String,
}

/// Case-insensitive ASCII prefix strip that returns the remainder of the
/// original string (case preserved). Safe on char boundaries because the
/// prefixes are pure ASCII.
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len()
        && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
    {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// Parse a goal description as a plain reminder. Returns `None` when the
/// description is not reminder-shaped or the reminder content looks like a
/// question the agent itself would need to answer (those still go through the
/// normal scheduled-goal pipeline).
pub fn parse_reminder(description: &str) -> Option<Reminder> {
    let mut rest = description.trim();

    // Strip leading politeness so "please remind me ..." still matches.
    for prefix in [
        "please ",
        "can you ",
        "could you ",
        "would you ",
        "will you ",
    ] {
        if let Some(r) = strip_prefix_ci(rest, prefix) {
            rest = r.trim_start();
        }
    }

    let after_verb = strip_prefix_ci(rest, "remind me ")
        .or_else(|| strip_prefix_ci(rest, "remind us "))
        .or_else(|| strip_prefix_ci(rest, "reminder: "))
        .or_else(|| strip_prefix_ci(rest, "reminder "))?;

    let connectives: [(&str, &'static str); 4] = [
        ("to ", "to"),
        ("about ", "about"),
        ("of ", "of"),
        ("that ", "that"),
    ];
    for (prefix, conn) in connectives {
        if let Some(body) = strip_prefix_ci(after_verb, prefix) {
            return build_reminder(body, Some(conn));
        }
    }
    build_reminder(after_verb, None)
}

fn build_reminder(raw_body: &str, connective: Option<&'static str>) -> Option<Reminder> {
    let raw_body = raw_body.trim().trim_end_matches(['?', '.', '!']).trim();
    if raw_body.is_empty() {
        return None;
    }
    // Bodies that read as questions to the agent ("remind me what I said
    // about X") need recall/work, not a canned message — skip the fast path.
    let first_word = raw_body
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase();
    if matches!(
        first_word.as_str(),
        "what" | "when" | "where" | "who" | "whose" | "which" | "how" | "why"
    ) {
        return None;
    }
    Some(Reminder {
        connective,
        body: flip_pronouns(raw_body),
    })
}

/// Flip first-person pronouns to second person so the reminder reads naturally
/// when sent back ("call my daughter" → "call your daughter").
fn flip_pronouns(text: &str) -> String {
    text.split_whitespace()
        .map(|token| {
            let core_start = token.find(|c: char| c.is_alphanumeric() || c == '\'');
            let Some(cs) = core_start else {
                return token.to_string();
            };
            let core_end = token
                .rfind(|c: char| c.is_alphanumeric() || c == '\'')
                .map(|i| i + token[i..].chars().next().map_or(1, char::len_utf8))
                .unwrap_or(token.len());
            let (lead, rest) = token.split_at(cs);
            let (core, tail) = rest.split_at(core_end - cs);
            let flipped = match core.to_lowercase().as_str() {
                "i" => Some("you"),
                "i'm" => Some("you're"),
                "i'll" => Some("you'll"),
                "i've" => Some("you've"),
                "me" => Some("you"),
                "my" => Some("your"),
                "mine" => Some("yours"),
                "myself" => Some("yourself"),
                "am" => None, // "am" only follows "I"; leave it — rare and ambiguous
                _ => None,
            };
            match flipped {
                Some(f) => format!("{}{}{}", lead, f, tail),
                None => token.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Rebuild a canonical reminder description from a bare imperative body
/// ("Check logs" → "Remind me to check logs"). Used when upstream description
/// cleaning stripped the user's leading "remind me to", so the stored goal
/// stays recognizable as a reminder at fire time.
pub fn canonical_description(body: &str) -> String {
    let body = body.trim().trim_end_matches(['?', '.', '!']).trim();
    let first_word = body.split_whitespace().next().unwrap_or("");
    let mut chars = first_word.chars();
    // Lowercase sentence-style capitalization on the leading word ("Check");
    // leave acronyms ("PR") and single letters alone.
    let sentence_case = matches!(chars.next(), Some(c) if c.is_uppercase())
        && chars.all(|c| c.is_lowercase())
        && first_word.chars().count() > 1;
    if sentence_case {
        let mut it = body.chars();
        let first = it.next().expect("non-empty checked above");
        format!("Remind me to {}{}", first.to_lowercase(), it.as_str())
    } else {
        format!("Remind me to {}", body)
    }
}

/// One-line scheduling confirmation, e.g.
/// "⏰ Got it — I'll remind you to call your daughter today at 1:46 PM."
#[cfg(test)]
pub fn confirmation_message(reminder: &Reminder, when: &str) -> String {
    match reminder.connective {
        Some(conn) => format!(
            "⏰ Got it — I'll remind you {} {} {}.",
            conn, reminder.body, when
        ),
        None => format!("⏰ Got it — I'll remind you {}: {}.", when, reminder.body),
    }
}

/// The message sent when the reminder fires, e.g. "⏰ Reminder: call your daughter".
pub fn fire_message(reminder: &Reminder) -> String {
    match reminder.connective {
        Some("that") => format!("⏰ Reminder: remember that {}", reminder.body),
        Some("about") | Some("of") => format!("⏰ Reminder about: {}", reminder.body),
        _ => format!("⏰ Reminder: {}", reminder.body),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_reminder_with_question_mark() {
        let r = parse_reminder("Remind me to call my daughter?").unwrap();
        assert_eq!(r.connective, Some("to"));
        assert_eq!(r.body, "call your daughter");
    }

    #[test]
    fn parses_polite_prefix() {
        let r = parse_reminder("please remind me to take out the trash").unwrap();
        assert_eq!(r.connective, Some("to"));
        assert_eq!(r.body, "take out the trash");
    }

    #[test]
    fn parses_about_connective() {
        let r = parse_reminder("Remind me about the dentist appointment.").unwrap();
        assert_eq!(r.connective, Some("about"));
        assert_eq!(r.body, "the dentist appointment");
    }

    #[test]
    fn parses_that_connective_with_pronoun_flip() {
        let r = parse_reminder("remind me that I'm meeting Maria").unwrap();
        assert_eq!(r.connective, Some("that"));
        assert_eq!(r.body, "you're meeting Maria");
    }

    #[test]
    fn parses_no_connective() {
        let r = parse_reminder("remind me call mom").unwrap();
        assert_eq!(r.connective, None);
        assert_eq!(r.body, "call mom");
    }

    #[test]
    fn rejects_non_reminders() {
        assert!(parse_reminder("check the deploy status").is_none());
        assert!(parse_reminder("send the weekly report").is_none());
        assert!(parse_reminder("").is_none());
        // "reminders" as a topic, not a request
        assert!(parse_reminder("list my reminders").is_none());
    }

    #[test]
    fn rejects_recall_questions() {
        assert!(parse_reminder("remind me what I said about the budget").is_none());
        assert!(parse_reminder("remind me when my flight is").is_none());
        assert!(parse_reminder("remind me how to restart the server").is_none());
    }

    #[test]
    fn preserves_names_and_case_in_body() {
        let r = parse_reminder("remind me to email Maria about Q3").unwrap();
        assert_eq!(r.body, "email Maria about Q3");
    }

    #[test]
    fn flips_possessives_and_contractions() {
        assert_eq!(
            flip_pronouns("pick up my dry cleaning"),
            "pick up your dry cleaning"
        );
        assert_eq!(
            flip_pronouns("tell me I'll be late"),
            "tell you you'll be late"
        );
        assert_eq!(
            flip_pronouns("that book is mine, not yours"),
            "that book is yours, not yours"
        );
    }

    #[test]
    fn canonical_description_restores_reminder_form() {
        assert_eq!(
            canonical_description("Check logs"),
            "Remind me to check logs"
        );
        assert_eq!(
            canonical_description("check logs."),
            "Remind me to check logs"
        );
        // Acronyms keep their casing.
        assert_eq!(canonical_description("PR review"), "Remind me to PR review");
        // Round-trips through the parser.
        let r = parse_reminder(&canonical_description("Call my daughter?")).unwrap();
        assert_eq!(r.body, "call your daughter");
    }

    #[test]
    fn confirmation_and_fire_messages_read_naturally() {
        let r = parse_reminder("Remind me to call my daughter?").unwrap();
        assert_eq!(
            confirmation_message(&r, "today at 1:46 PM"),
            "⏰ Got it — I'll remind you to call your daughter today at 1:46 PM."
        );
        assert_eq!(fire_message(&r), "⏰ Reminder: call your daughter");

        let r2 = parse_reminder("remind me about the standup").unwrap();
        assert_eq!(
            confirmation_message(&r2, "tomorrow at 9:00 AM"),
            "⏰ Got it — I'll remind you about the standup tomorrow at 9:00 AM."
        );
        assert_eq!(fire_message(&r2), "⏰ Reminder about: the standup");
    }

    #[test]
    fn handles_wrapped_scheduled_task_description() {
        // heartbeat wraps descriptions; the fast path uses goal.description,
        // but make sure the wrapped form does NOT match (it goes through the
        // normal pipeline).
        assert!(parse_reminder(
            "Execute scheduled goal: Remind me to call my daughter? [SYSTEM: already scheduled]"
        )
        .is_none());
    }
}
