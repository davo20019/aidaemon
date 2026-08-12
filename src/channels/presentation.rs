//! Shared presentation rules for outbound chat text.
//!
//! Keep channel messages readable here instead of teaching every notification
//! producer how to lay out a mobile-friendly card.

const MAX_INLINE_STEP_CHARS: usize = 180;

/// Prepare arbitrary agent-originated Markdown for a chat surface.
pub(crate) fn prepare_chat_message(text: &str) -> String {
    let cleaned = crate::tools::sanitize::strip_leaked_control_markers(text);
    let mut lines = Vec::new();
    let mut previous_blank = false;

    for raw_line in cleaned.lines() {
        let line = raw_line.trim_end();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !previous_blank && !lines.is_empty() {
                lines.push(String::new());
            }
            previous_blank = true;
            continue;
        }

        // Orchestration steps can contain the user's entire multi-paragraph
        // instruction. They are useful in the task ledger, not in a chat card.
        if is_verbose_step_echo(trimmed) || is_probable_bold_task_prompt(trimmed) {
            continue;
        }

        lines.push(style_known_label(line));
        previous_blank = false;
    }

    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines.join("\n").trim().to_string()
}

/// Apply a consistent status card to queued notifications.
pub(crate) fn present_notification(notification_type: &str, message: &str) -> String {
    let prepared = prepare_chat_message(message);
    if prepared.is_empty() || has_status_heading(&prepared) {
        return prepared;
    }

    let heading = match notification_type {
        "completed" => "✅ **Completed**",
        "failed" => "❌ **Run failed**",
        "escalation" | "mandate_ask" | "mandate_reconciliation_required" => "⚠️ **Action needed**",
        "stalled" => "⏸️ **Work paused**",
        "token_alert" => "⚠️ **Budget alert**",
        "evergreen_alert" => "⚠️ **Schedule needs attention**",
        "mandate_paused" | "mandate_stopped" => "⏸️ **Automation paused**",
        "mandate_review_failed" | "mandate_reconciliation" => "⚠️ **Automation update**",
        "mandate_action" => "✅ **Automation update**",
        "node_monitor_alert" => "🚨 **Node alert**",
        "node_monitor_recovery" => "✅ **Node recovered**",
        "node_monitor_ended" | "node_monitor_suspended" => "⏸️ **Node monitoring**",
        "progress" | "status_update" => "🔄 **Update**",
        _ => "ℹ️ **Update**",
    };
    let body = trim_redundant_lead(notification_type, &prepared);
    if body.is_empty() {
        heading.to_string()
    } else {
        format!("{heading}\n\n{body}")
    }
}

fn is_verbose_step_echo(line: &str) -> bool {
    line.strip_prefix("Step:")
        .or_else(|| line.strip_prefix("**Step:**"))
        .is_some_and(|step| step.trim().chars().count() > MAX_INLINE_STEP_CHARS)
}

fn is_probable_bold_task_prompt(line: &str) -> bool {
    if !line.starts_with("**") || !line.ends_with("**") || line.chars().count() <= 220 {
        return false;
    }
    let lower = line.to_ascii_lowercase();
    [
        "create", "read", "write", "run", "deploy", "verify", "exactly",
    ]
    .iter()
    .filter(|word| lower.contains(*word))
    .count()
        >= 3
}

fn style_known_label(line: &str) -> String {
    let trimmed = line.trim_start();
    if trimmed.starts_with("**") {
        return line.to_string();
    }
    for label in [
        "Blocked",
        "Step",
        "Needed to continue",
        "Needed",
        "Resume",
        "Live",
        "Source",
        "Verification",
        "Status",
        "Version",
        "ID",
    ] {
        let prefix = format!("{label}:");
        if let Some(value) = trimmed.strip_prefix(&prefix) {
            return format!("**{label}:**{}", value);
        }
    }
    line.to_string()
}

fn has_status_heading(message: &str) -> bool {
    let first = message.lines().next().unwrap_or_default().trim_start();
    [
        "✅ **",
        "❌ **",
        "⚠️ **",
        "⏸️ **",
        "🔄 **",
        "⏳ **",
        "🔁 **",
        "ℹ️ **",
    ]
    .iter()
    .any(|prefix| first.starts_with(prefix))
}

fn trim_redundant_lead<'a>(notification_type: &str, message: &'a str) -> &'a str {
    let prefixes: &[&str] = match notification_type {
        "completed" => &["Goal completed:", "Goal completed.", "Completed:"],
        "failed" => &["Goal failed:", "Run failed:", "Failed:"],
        "stalled" => &["Goal stalled:", "Work stalled:"],
        _ => &[],
    };
    for prefix in prefixes {
        if let Some(rest) = message.strip_prefix(prefix) {
            return rest.trim();
        }
    }
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_presentation_removes_long_step_echo_and_styles_labels() {
        let prompt = "Create, read, write, run, deploy, and verify exactly one post. ".repeat(8);
        let input = format!(
            "The deploy failed.\nStep: {prompt}\nNeeded: Fix the build.\nResume: /work unblock 12345678 <resolution>"
        );

        let output = prepare_chat_message(&input);

        assert!(!output.contains("Create, read"));
        assert!(output.contains("**Needed:** Fix the build."));
        assert!(output.contains("**Resume:** /work unblock"));
    }

    #[test]
    fn notifications_get_one_consistent_heading() {
        assert_eq!(
            present_notification("completed", "Goal completed:\n\nPublished the post."),
            "✅ **Completed**\n\nPublished the post."
        );
        assert_eq!(
            present_notification("escalation", "⚠️ **Action needed**\n\nFix it."),
            "⚠️ **Action needed**\n\nFix it."
        );
    }

    #[test]
    fn long_bold_task_prompt_is_removed() {
        let prompt = format!(
            "**{}**",
            "Create, read, write, deploy, and verify exactly one post. ".repeat(8)
        );
        let output = prepare_chat_message(&format!("Published the post.\n\n{prompt}\n\nHTTP 200."));
        assert_eq!(output, "Published the post.\n\nHTTP 200.");
    }
}
